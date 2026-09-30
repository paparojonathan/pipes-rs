//! Command-line surface of the `pipes` binary (clap derive).

use std::path::PathBuf;

use clap::{Args, Parser, Subcommand, ValueEnum};
use pipes_kitti::voxel::VoxelSize;

/// Spin window in ms for the paced driver: sleep in <=1 ms steps until
/// `due - window`, then spin. Measured, not assumed (D6): three release runs
/// of a `sleep(1 ms)` overshoot probe on this host gave p99 = 1.086 / 0.891 /
/// 0.900 ms (rule: p99 rounded up to the next 0.5 ms, min 0.5, so 1.5 / 1.0 /
/// 1.0 ms); the conservative 1.5 ms is kept (D17, tracker Log 2026-09-20). A
/// fixed setting rather than a flag: no run ever needed a different value.
pub const DEFAULT_SPIN_WINDOW_MS: f64 = 1.5;

/// Largest queue capacity whose depth can be recorded faithfully.
/// `Evidence::depth_at_push` is a `u16` and `queue::depth_u16` saturates, so a
/// larger capacity would write 65535 for every deeper queue: evidence that
/// looks like a measurement and is not. Rejecting at parse time keeps the
/// saturation unreachable instead of merely unlikely.
pub const MAX_CAP: usize = u16::MAX as usize;

/// The port `--rerun grpc` looks for a viewer on and starts one at: the SDK's
/// own default, so a `rerun` started by hand listens where a plain run looks.
pub const DEFAULT_VIEWER_PORT: u16 = rerun::DEFAULT_SERVER_PORT;

/// `--cap` parser: rejects 0 and anything above [`MAX_CAP`].
///
/// A capacity of 0 is clamped to 1 by `BoundedQueue::new`, which would silently
/// run a different experiment than the one asked for.
fn parse_cap(s: &str) -> Result<usize, String> {
    let n: usize = s
        .parse()
        .map_err(|_| format!("`{s}` is not a whole number"))?;
    if n == 0 {
        return Err("capacity must be at least 1 (0 would be clamped to 1)".to_string());
    }
    if n > MAX_CAP {
        return Err(format!(
            "capacity must be at most {MAX_CAP}; a deeper queue would record a saturated `depth_at_push` (a u16) rather than a real one"
        ));
    }
    Ok(n)
}

/// Length of the `YYYY_MM_DD` prefix every KITTI drive name starts with.
const DATE_LEN: usize = 10;

/// `--drive` parser. A KITTI drive name carries its capture date as its first
/// ten characters (`2011_09_26_drive_0005_sync` sits under `2011_09_26/`), so
/// the date directory is derived from the name rather than asked for twice.
/// A name without that prefix is refused here, where the message can say what
/// a drive name looks like, instead of at open time as a missing date.
fn parse_drive(s: &str) -> Result<String, String> {
    let b = s.as_bytes();
    let dated = b.len() > DATE_LEN
        && b[DATE_LEN] == b'_'
        && b[..DATE_LEN].iter().enumerate().all(|(i, c)| match i {
            4 | 7 => *c == b'_',
            _ => c.is_ascii_digit(),
        });
    if dated {
        Ok(s.to_string())
    } else {
        Err(format!(
            "`{s}` is not a KITTI drive name; one looks like 2011_09_26_drive_0005_sync, \
             and its first {DATE_LEN} characters name the capture date directory it sits under"
        ))
    }
}

/// Default KITTI data root: relative to the repository root, because the
/// dataset sits *beside* the checkout rather than inside it (README, "Setup"). It was
/// an absolute path into one developer's home directory, so every documented
/// command failed on any other machine - including CI - with a confusing
/// "path not found" instead of "tell me where the dataset is". Override per
/// invocation with `--kitti-root`, or once with `PIPES_KITTI_ROOT`.
pub const DEFAULT_KITTI_ROOT: &str = "../data/kitti";

#[derive(Parser, Debug)]
#[command(
    name = "pipes",
    version,
    about = "Single-stream zero-copy pipeline experiments",
    // Two commands, and `--help` lists exactly those two.
    disable_help_subcommand = true
)]
pub struct Cli {
    #[command(subcommand)]
    pub cmd: Cmd,
}

#[derive(Subcommand, Debug)]
pub enum Cmd {
    /// List the KITTI drives under a data root, with frame count and health.
    Drives(DrivesArgs),
    /// Replay a KITTI drive through one admission and the stages behind it.
    Run(RunArgs),
}

/// Arguments of `pipes drives`.
///
/// `--drive` is an unvalidated name everywhere else, and nothing else in the
/// binary could answer "what can I pass?": this subcommand is that
/// answer, so it takes the same `--kitti-root`, with the same environment
/// override, and nothing else it does not need.
#[derive(Args, Debug)]
pub struct DrivesArgs {
    /// KITTI data root (holds `<date>/<drive>/image_02`). `PIPES_KITTI_ROOT`
    /// is the one-time override for a checkout whose dataset is elsewhere.
    #[arg(long, env = "PIPES_KITTI_ROOT", default_value = DEFAULT_KITTI_ROOT)]
    pub kitti_root: PathBuf,
    /// Print a JSON array instead of the aligned table.
    #[arg(long)]
    pub json: bool,
}

/// Overflow policy of the camera queue `--cap` sizes: `cam0->camdet` when the
/// detector runs, `cam0->proc` when it does not.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum PolicyArg {
    DropOldest,
    DropNewest,
    /// Bounded wait on the driver thread: the negative control (D8/D14).
    Block,
}

/// Whether the run replays the drive's lidar stream beside the camera.
///
/// The default is [`LidarArg::Auto`] — the drive either carries a
/// `velodyne_points/` directory or it does not, and asking the user to repeat
/// a fact that is on disk is how a second stream ends up never being
/// exercised. Many KITTI downloads are `image_02` alone, so a camera-only
/// drive is a common case and must stay the quiet one: under `auto` such a
/// drive opens no lidar queue, spawns no lidar thread and prints nothing
/// about lidar, so its output is what it was before this existed.
///
/// [`LidarArg::On`] is the override that turns a missing stream into an
/// error, and [`LidarArg::Off`] the one that ignores a present stream.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, ValueEnum)]
pub enum LidarArg {
    /// Replay the lidar exactly when the drive has a `velodyne_points/` directory.
    #[default]
    Auto,
    /// Require the lidar; a drive without one is an error.
    On,
    /// Never replay the lidar, even when the drive has one.
    Off,
}

impl LidarArg {
    /// The CLI / `run.json` spelling.
    pub fn name(self) -> &'static str {
        match self {
            LidarArg::Auto => "auto",
            LidarArg::On => "on",
            LidarArg::Off => "off",
        }
    }
}

/// Whether the lidar's sweeps are reduced and handed to a second stage.
///
/// `on` by default, and only ever reachable on a drive whose lidar is
/// replaying at all — so a camera-only drive is untouched by this flag
/// existing.
///
/// [`ReduceArg::Off`] exists so the lidar path can be run exactly as it was
/// before this stage was written, which is what makes the chain's cost
/// measurable rather than merely stated.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, ValueEnum)]
pub enum ReduceArg {
    /// Reduce every sweep and hand the result to a second stage.
    #[default]
    On,
    /// Replay the lidar with leaf consumers only, as M11 left it.
    Off,
}

impl ReduceArg {
    /// The CLI / `run.json` spelling.
    pub fn name(self) -> &'static str {
        match self {
            ReduceArg::On => "on",
            ReduceArg::Off => "off",
        }
    }
}

/// Whether the reduced cloud is turned into detections by a third stage.
///
/// `on` by default, on exactly the terms [`ReduceArg`] is: it is reachable
/// only on a drive whose lidar is replaying AND whose sweeps are being
/// reduced, so a camera-only drive is untouched by the flag existing.
///
/// [`DetectArg::Off`] exists so the chain can be run exactly as M12 left it,
/// which is what makes this stage's cost measurable rather than merely stated.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, ValueEnum)]
pub enum DetectArg {
    /// Find objects in every reduced cloud and hand them to a fourth stage.
    #[default]
    On,
    /// Stop the chain at the reduced cloud, as M12 left it.
    Off,
}

impl DetectArg {
    /// The CLI / `run.json` spelling.
    pub fn name(self) -> &'static str {
        match self {
            DetectArg::On => "on",
            DetectArg::Off => "off",
        }
    }
}

/// Whether the detections are tracked across sweeps, fused with the camera
/// frame, and reduced to an answer by two more stages.
///
/// `on` by default, so a plain `pipes run` shows the whole chain. With it the
/// camera's half of each pair is admitted by a camera stage for every frame
/// it finishes: the frozen detector's batch under `--detector on`, or, under
/// `--detector off`, a bare reference from `proc`, emitted after `proc_end`
/// and counted on its own, so `proc_bytes_per_frame` does not move either
/// way. The integration tests
/// pass `--track off` through their shared helper, so the fixtures without a
/// calibration keep running as they did.
///
/// Reachable only with `--reduce on --detect on` on a drive whose lidar is
/// replaying, so a camera-only drive is untouched by the flag existing.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, ValueEnum)]
pub enum TrackArg {
    /// Track the detections, pair each sweep with its camera frame, and answer.
    #[default]
    On,
    /// Stop the chain at the detections, as M13 left it.
    Off,
}

impl TrackArg {
    /// The CLI / `run.json` spelling.
    pub fn name(self) -> &'static str {
        match self {
            TrackArg::On => "on",
            TrackArg::Off => "off",
        }
    }
}

/// Whether the camera's half of each fused pair is the frozen detector's
/// output or a bare frame reference.
///
/// `on` by default, so a plain `pipes run` fuses what the camera SAW with what
/// the lidar measured. The integration tests pass `--detector off` through
/// their shared helper: their fixtures have no model file, and the frame
/// reference `proc` sends is what the fusion was built and tested against.
///
/// Reachable only where the fusion is -- `--track on` on a drive whose lidar
/// is replaying -- so a camera-only drive is untouched by the flag
/// existing, and a run that does not fuse never needs the model.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, ValueEnum)]
pub enum DetectorArg {
    /// Run the frozen detector on every camera frame and send its detections.
    #[default]
    On,
    /// Send `proc`'s bare frame reference, as before the detector existed.
    Off,
}

impl DetectorArg {
    /// The CLI / `run.json` spelling.
    pub fn name(self) -> &'static str {
        match self {
            DetectorArg::On => "on",
            DetectorArg::Off => "off",
        }
    }
}

/// `--voxel-size-m` parser: refuses everything that would fail *silently*.
///
/// Each rejected value produces a plausible-looking empty or nonsense result
/// rather than an error further in: `0` makes every division infinite so every
/// point falls outside the grid, a negative edge runs the grid backwards, and
/// `NaN` fails every comparison so every point is discarded and the stage
/// reports an empty cloud with nothing anywhere saying why.
fn parse_voxel_size(s: &str) -> Result<VoxelSize, String> {
    let m: f32 = s
        .parse()
        .map_err(|_| format!("`{s}` is not a number of metres"))?;
    VoxelSize::new(m).map_err(|e| e.to_string())
}

/// `--rerun-host`: a bare host name or IP address. The run builds the URL
/// itself from it and `--rerun-port`, so a scheme, a port or a path here
/// would build a wrong one; they are refused with the form that works.
fn parse_rerun_host(s: &str) -> Result<String, String> {
    let h = s.trim();
    if h.is_empty() {
        return Err("the host is empty".to_string());
    }
    if h.contains("://") || h.contains('/') {
        return Err(format!(
            "`{h}` is a URL; give the host alone, such as `host.docker.internal`, and the \
             port with --rerun-port"
        ));
    }
    let bracketed_v6 = h.starts_with('[') && h.ends_with(']');
    if !bracketed_v6 && h.matches(':').count() == 1 {
        return Err(format!(
            "`{h}` carries a port; give the host alone and the port with --rerun-port"
        ));
    }
    Ok(h.to_string())
}

/// Where the `rerun` consumer sends frames.
///
/// `grpc` by default: the run opens the viewer itself (or joins one already
/// listening on port 9876) and streams into it, so seeing the pipeline is one
/// command. `rrd` writes the same recording to a file for later.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, ValueEnum)]
pub enum RerunArg {
    /// No rerun consumer and no `cam0->rerun` edge.
    #[value(hide = true)]
    Off,
    /// The consumer runs and is measured but logs nothing.
    #[value(hide = true)]
    Null,
    /// Write `runs/<name>/cam0.rrd`.
    Rrd,
    /// Stream to the Rerun viewer, starting one if none is listening.
    #[default]
    Grpc,
}

/// Whether the recording carries the pipeline's own picture of itself.
///
/// `on` by default: every evidence row is mirrored as a series or a lane,
/// the lidar clouds, the detections, the tracks, the fusion overlay and the
/// answer are all drawn, and the layout that shows them is sent ahead of the
/// first frame (`crates/pipes/src/dashboard.rs`). `off` leaves the recording
/// with the raw camera frames only, which is what the integration tests use
/// through their helper.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, ValueEnum)]
pub enum DashboardArg {
    /// Draw the whole chain: timing series, images, clouds, boxes, answer.
    #[default]
    On,
    /// Log the camera frames only; nothing is drawn.
    Off,
}

impl DashboardArg {
    pub fn is_on(self) -> bool {
        self == DashboardArg::On
    }
}

/// Arguments of `pipes run`.
///
/// Two tiers. The eleven `--help` shows are what a reader needs to see the
/// pipeline and to make it fail on purpose: `--cap` and `--consumer-delay-ms`
/// act on the camera stage whose output reaches the answer, so slowing the
/// camera reaches the fusion whichever stage that is. The rest (`hide = true`)
/// are the experiment's knobs -- the camera queue's policy, the replay rate,
/// which stages of the lidar chain run -- still accepted, still recorded in
/// `run.json`, but not an invitation on the first screen.
#[derive(Args, Debug, Clone)]
pub struct RunArgs {
    /// KITTI data root (holds `<date>/<drive>/image_02`). `PIPES_KITTI_ROOT`
    /// is the one-time override for a checkout whose dataset is elsewhere.
    #[arg(long, env = "PIPES_KITTI_ROOT", default_value = DEFAULT_KITTI_ROOT)]
    pub kitti_root: PathBuf,
    /// The drive to replay, under `<kitti-root>/<its first 10 characters>/`.
    #[arg(long, default_value = "2011_09_26_drive_0005_sync", value_parser = parse_drive)]
    pub drive: String,
    /// Replay rate factor: 1.0 = real time, 10 = ten times faster, `inf` = unpaced.
    #[arg(long, default_value_t = 1.0, hide = true)]
    pub rate: f64,
    /// Run name (directory under `runs/`); default `run-<unix secs>`.
    #[arg(long)]
    pub name: Option<String>,
    /// Capacity of the camera queue in front of the stage whose output
    /// reaches the answer, 1..=65535 (see `parse_cap` for why the ends are
    /// refused): `cam0->camdet`, the detector's, when the detector runs, and
    /// `cam0->proc` when it does not (`--detector off`, or no fusion for it
    /// to feed). The other stage does not run.
    ///
    /// Omitted, it is 1 on `cam0->camdet` and 4 on `cam0->proc`, as each was
    /// before `--cap` could reach the detector. The detector's 1 is chosen. At
    /// 70 to 90 ms a frame against the camera's 103 ms period, a healthy
    /// real-time run rarely has a frame waiting: depths 1 and 4 both
    /// delivered all 154 frames of drive_0005, to the same bytes. Once the
    /// detector is slower than the camera, a deeper queue delivers no more
    /// frames, only older ones: at `--consumer-delay-ms 60 --pair-wait-ms
    /// 300`, depths 1, 2 and 4 ran on 102, 101 and 104 frames, finished a
    /// median 208, 311 and 517 ms after each was due, and at depth 4 the
    /// fusion, waiting on them, lost 40 sweeps on the lidar side (one run
    /// each, uncertified). That age is the staleness this project measures.
    /// What depth 1 gives up: when one frame holds the detector past the next
    /// two frames' arrival, the first of those is evicted rather than
    /// delivered late.
    #[arg(long, value_parser = parse_cap)]
    pub cap: Option<usize>,
    /// Overflow policy of the same camera queue `--cap` sizes: `cam0->camdet`
    /// when the detector runs, `cam0->proc` when it does not.
    ///
    /// `drop-oldest` by default, and for the detector that is load-bearing
    /// twice: a Drop* edge gets `try_push` in `Admission::admit`, so a
    /// detector slower than the camera can never stall the camera driver -- it
    /// loses frames instead, each one a drop row with reason `evicted` -- and
    /// dropping the OLDEST is what keeps the frame it does run on the newest
    /// one. `block` is the negative control on either queue.
    #[arg(long, value_enum, default_value_t = PolicyArg::DropOldest, hide = true)]
    pub policy: PolicyArg,
    /// `max_wait` of `--policy block`, in ms, on the same queue.
    #[arg(long, default_value_t = 500, hide = true)]
    pub block_max_wait_ms: u64,
    /// Artificial per-frame delay, in ms, in the stage behind the queue
    /// `--cap` sizes: `camdet` when the detector runs, `proc` when it does
    /// not -- either way the stage whose output reaches the answer.
    ///
    /// Slept inside the stage's measured window, so the evidence shows it as
    /// service time. With the detector on, `--consumer-delay-ms 100 --cap 1
    /// --pair-wait-ms 300` makes the camera lose about half of drive_0005's
    /// frames (71 to 84 of 154 on the runs of 2026-09-27), and the fusion
    /// expires exactly those sweeps, each one `pair_dropped`, on every run
    /// tried. At 200 ms that held on 3 runs of 7; on the other 4 the detector
    /// took about three periods a frame, the fusion fell behind the lidar
    /// waiting on it, and `obj->track` evicted sweeps as well (4 to 15), so
    /// use 100 ms.
    #[arg(long, default_value_t = 0)]
    pub consumer_delay_ms: u64,
    /// `proc` reuses one output buffer instead of allocating one per frame.
    #[arg(long, hide = true)]
    pub reuse_output: bool,
    /// Replay the drive's Velodyne sweeps as a second producer through the
    /// same admission. `auto` (the default) does it exactly when the drive has
    /// a `velodyne_points/` directory; `on` makes a drive without one an
    /// error; `off` ignores one that is there. See `cargo run --release -- drives` for which
    /// drives carry lidar.
    #[arg(long, value_enum, default_value_t = LidarArg::Auto, hide = true)]
    pub lidar: LidarArg,
    /// Reduce each sweep to a voxel grid in a `reduce` stage and hand the
    /// result to a second consumer as its own derived stream. This is the
    /// project's only stage-to-stage Arrow hand-off; `off` replays the lidar
    /// with leaf consumers only. No effect on a drive without lidar.
    #[arg(long, value_enum, default_value_t = ReduceArg::On, hide = true)]
    pub reduce: ReduceArg,
    /// Remove the ground from each reduced cloud, group what is left into
    /// objects, and hand those to a fourth stage as their own derived stream.
    /// `off` stops the chain at the reduced cloud. No effect without
    /// `--reduce on`, and none on a drive without lidar.
    ///
    /// Every parameter this stage uses is derived rather than exposed --
    /// there is deliberately no `--ground-threshold` or `--cluster-size`,
    /// because a knob is an invitation to tune the output until it looks
    /// right. See `pipes_kitti::detect` for where each number comes from.
    #[arg(long, value_enum, default_value_t = DetectArg::On, hide = true)]
    pub detect: DetectArg,
    /// Track the detections across sweeps, pair each sweep with the camera
    /// frame whose instant falls inside it, and answer with every track: its
    /// distance, closing speed, age and place in the camera image, the
    /// nearest one in the vehicle's path flagged.
    ///
    /// No effect on a drive without lidar.
    #[arg(long, value_enum, default_value_t = TrackArg::On)]
    pub track: TrackArg,
    /// Run a frozen, pretrained camera detector (YOLOX-Nano, 80 COCO
    /// classes) on the camera frames in a `camdet` stage of its own, and fuse
    /// what it found: each pair's camera half is the frame's detections --
    /// box, class, confidence -- instead of a bare frame reference.
    ///
    /// Visible, and the eleventh flag on this page, because it decides
    /// whether the camera contributes anything to the answer. The weights
    /// are `models\yolox_nano.onnx`, checked against a pinned sha256; a
    /// missing file stops the run with the command that downloads it,
    /// `powershell -ExecutionPolicy Bypass -File scripts\fetch_model.ps1`.
    /// The stage reads its own queue, `cam0->camdet`, and with the detector
    /// on that is the queue `--cap` sizes and the stage `--consumer-delay-ms`
    /// slows. No effect without `--track on`.
    #[arg(long, value_enum, default_value_t = DetectorArg::On)]
    pub detector: DetectorArg,
    /// How long the fusion waits for a camera frame that has not arrived yet,
    /// in ms. Omitted means **one sweep span**, taken from the sweep's own
    /// `Tov::Range` rather than assumed; `0` refuses immediately.
    ///
    /// `0` is the negative control and it is worth running: with the
    /// detector on (the default) -- 70 to 90 ms a frame on the development
    /// host, uncertified, and printed on each run's `camdet ms/frame` line --
    /// the camera path dropped NOTHING at real time, and every frame still
    /// reached the fusion after the sweep it belongs to, so a fusion that
    /// refuses to wait reported all 154 sets expired (`pair_late`) on runs
    /// where nothing was lost; the default wait completed all 154. The
    /// detector's margin under the 103 ms period is small, so a busier host
    /// can drop frames at `cam0->camdet` as well. With `--detector off` the same was
    /// measured at `--consumer-delay-ms 100 --cap 16`, and at 50 ms the camera
    /// was mostly in time: 0 or 9 of 154 expired on two runs.
    #[arg(long)]
    pub pair_wait_ms: Option<u64>,
    /// Permit the fusion to pair a sweep with a camera frame OLDER than its
    /// range, taken up to this many ms before the sweep's trigger, and label
    /// the result stale.
    ///
    /// Counted from the trigger because every stale label -- the answer's
    /// words, the picture, `pair_age_ns` -- counts from it too, so none reads
    /// staler than the window asked for. On drive_0005 the previous frame is
    /// 92.5 to 93.2 ms before the trigger, so it takes 94 or more to admit it
    /// on every sweep; a set the window does not reach expires instead.
    ///
    /// `0` (the default) refuses, and the refusal is the point: the
    /// architecture document says silent stale-data reuse would undermine the
    /// project. A run that wants degraded sets rather than expired ones has to
    /// ask for them by name, and every sample so produced carries
    /// `pair_age_ns` and `pair_outcome = stale` in its payload.
    #[arg(long, default_value_t = 0)]
    pub pair_stale_ms: u64,
    /// Voxel edge in metres for `--reduce on`.
    ///
    /// The default is derived from the smallest object that has to survive the
    /// reduction — one third of a 0.6 m pedestrian width — and NOT from the
    /// shrink ratio it produces. Coarsening it will make that ratio look
    /// better and is exactly the parameter-tuning this project's standards
    /// forbid; the number a run reports is whatever this edge gives on that
    /// scene.
    #[arg(long, default_value_t = VoxelSize::default(), value_parser = parse_voxel_size, hide = true)]
    pub voxel_size_m: VoxelSize,
    /// Where the recording goes: `grpc` streams it to the Rerun viewer,
    /// starting one if none is listening; `rrd` writes `runs/<name>/cam0.rrd`
    /// to open later with `rerun runs/<name>/cam0.rrd`.
    #[arg(long, value_enum, default_value_t = RerunArg::Grpc)]
    pub rerun: RerunArg,
    /// The port `--rerun grpc` looks for a viewer on, and starts one at when
    /// nothing answers there. The SDK joins whatever is listening on that
    /// port before it looks at PATH, so a viewer left open from an earlier
    /// run receives every later run: this is the way to keep a run out of
    /// it, and the only way the no-viewer failure can be observed while one
    /// is open (`crates/pipes/tests/viewer.rs`). Hidden: a plain run wants
    /// the SDK's default, which is where a `rerun` started by hand listens.
    #[arg(long, hide = true, default_value_t = DEFAULT_VIEWER_PORT)]
    pub rerun_port: u16,
    /// The host `--rerun grpc` streams to, for a viewer outside this machine
    /// or container: the run joins the viewer listening there on
    /// `--rerun-port` and never starts one, and stops before its clock starts
    /// if nothing answers. Unset, the viewer is local and started when
    /// needed. In the Docker setup the viewer runs on the host, which the
    /// container reaches as `host.docker.internal`. Hidden: compose.yaml sets
    /// it, through `PIPES_RERUN_HOST`.
    #[arg(long, hide = true, env = "PIPES_RERUN_HOST", value_parser = parse_rerun_host)]
    pub rerun_host: Option<String>,
    /// Test hook: panic in the `proc` stage when this frame arrives, to
    /// exercise the shutdown path in the binary that actually ships. Hidden
    /// from `--help`; used only by
    /// `panic_in_proc_shuts_down_cleanly_and_exits_3`.
    #[arg(long, hide = true)]
    pub panic_at_frame: Option<u64>,
    /// Test hook: the recording takes nothing for this many ms from the
    /// moment the clock starts, as a viewer frozen that long does, and the
    /// Rerun SDK buffers 256 KiB instead of 100 MiB, so the stall reaches
    /// `log` within a few samples of a fixture. Only a recording written to a
    /// file (`--rerun rrd`, or `dashboard.rrd`) is frozen. Hidden from
    /// `--help`; used by `tests/viewer.rs`.
    #[arg(long, hide = true, default_value_t = 0)]
    pub viewer_delay_ms: u64,
    /// Draw the whole chain on the recording, in one layout sent ahead of the
    /// first frame: the answer as a headline and on the camera image with
    /// the tracks by source, the lidar in 3D, the time to collision, the
    /// pairing and the tracks in frame, and the auditor's pages (queues,
    /// latency, fusion, bytes, log). `off` keeps the raw camera frames only.
    /// With `--rerun null|off` the drawing goes to
    /// `runs/<name>/dashboard.rrd`.
    #[arg(long, value_enum, default_value_t = DashboardArg::On)]
    pub dashboard: DashboardArg,
}

impl RunArgs {
    /// The capture date directory: the first ten characters of `--drive`,
    /// which [`parse_drive`] guaranteed are `YYYY_MM_DD`.
    pub fn date(&self) -> &str {
        &self.drive[..DATE_LEN]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_rerun_host_is_a_bare_name_or_address() {
        for good in [
            "host.docker.internal",
            "192.168.1.20",
            "localhost",
            "::1",
            "[::1]",
        ] {
            assert_eq!(parse_rerun_host(good).as_deref(), Ok(good), "{good}");
        }
        for (bad, says) in [
            ("", "empty"),
            ("http://host.docker.internal:9876", "URL"),
            ("host.docker.internal/proxy", "URL"),
            ("host.docker.internal:9876", "--rerun-port"),
        ] {
            let Err(e) = parse_rerun_host(bad) else {
                panic!("`{bad}` was taken for a host");
            };
            assert!(e.contains(says), "{bad}: {e}");
        }
    }

    #[test]
    fn a_drive_name_carries_its_date() {
        assert_eq!(
            parse_drive("2011_09_26_drive_0005_sync").as_deref(),
            Ok("2011_09_26_drive_0005_sync")
        );
        assert!(parse_drive("2011_09_28_drive_0001_sync").is_ok());
    }

    #[test]
    fn a_name_without_a_date_prefix_is_refused_with_an_example() {
        for bad in [
            "drive_0005_sync",
            "2011_09_26",
            "2011-09-26_drive_0005_sync",
            "",
            "20110926_drive_0005",
        ] {
            match parse_drive(bad) {
                Err(e) => {
                    assert!(e.contains("2011_09_26_drive_0005_sync"), "{e}");
                    // One sentence, not a wrapped source line: a lost `\`
                    // continuation once printed 14 spaces mid-message.
                    assert!(!e.contains("  "), "{e}");
                }
                Ok(v) => panic!("`{v}` was accepted as a drive name"),
            }
        }
    }
}
