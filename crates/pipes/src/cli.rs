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
/// dataset sits *beside* the checkout rather than inside it outside Docker. It was
/// an absolute path into one developer's home directory, so every documented
/// command failed on any other machine - including CI - with a confusing
/// "path not found" instead of "tell me where the dataset is". Override per
/// invocation with `--kitti-root`, or once with `PIPES_KITTI_ROOT`.
pub const DEFAULT_KITTI_ROOT: &str = "../data/kitti";

#[derive(Parser, Debug)]
#[command(
    name = "pipes",
    version,
    about = "Sensor fusion on replayed KITTI drives",
    // Four commands, and `--help` lists exactly those four.
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
    /// Time the camera detector on a drive's frames: one model on one backend.
    BenchDetector(BenchArgs),
    /// Time the lidar detector on a drive's sweeps: PointPillars on one backend.
    BenchLidar(BenchLidarArgs),
}

/// Which network `--detector-model` runs. Each is the published YOLOX file
/// with its declared input at the size `pipes` feeds (see
/// `scripts/prepare_models.py`), and each has its own pinned sha256.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, ValueEnum)]
pub enum DetectorModelArg {
    /// YOLOX-Nano at 640x192, 0.9 M parameters: the default; `pipes-gpu` alone runs YOLOX-L.
    #[default]
    #[value(name = "yolox_nano")]
    YoloxNano,
    /// YOLOX-S at 640x192, 9 M parameters.
    #[value(name = "yolox_s")]
    YoloxS,
    /// YOLOX-S at 1024x320: the same network at four fifths of the frame's resolution.
    #[value(name = "yolox_s-1024")]
    YoloxS1024,
    /// YOLOX-M at 1024x320, 25 M parameters.
    #[value(name = "yolox_m-1024")]
    YoloxM1024,
    /// YOLOX-L at 1248x384, the frame's own resolution, 54 M parameters, 16-bit weights: for the GPU.
    #[value(name = "yolox_l-1248")]
    YoloxL1248,
    /// YOLOX-X at 1248x384, 99 M parameters, 16-bit weights: the biggest, for the GPU.
    #[value(name = "yolox_x-1248")]
    YoloxX1248,
}

impl DetectorModelArg {
    /// The model's spec: file, hash, input size.
    pub fn spec(self) -> &'static pipes_kitti::camdet::ModelSpec {
        use pipes_kitti::camdet::{
            YOLOX_L_1248, YOLOX_M_1024, YOLOX_NANO, YOLOX_S, YOLOX_S_1024, YOLOX_X_1248,
        };
        match self {
            DetectorModelArg::YoloxNano => &YOLOX_NANO,
            DetectorModelArg::YoloxS => &YOLOX_S,
            DetectorModelArg::YoloxS1024 => &YOLOX_S_1024,
            DetectorModelArg::YoloxM1024 => &YOLOX_M_1024,
            DetectorModelArg::YoloxL1248 => &YOLOX_L_1248,
            DetectorModelArg::YoloxX1248 => &YOLOX_X_1248,
        }
    }
}

/// Which runtime `--detector-backend` runs the network on.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, ValueEnum)]
pub enum DetectorBackendArg {
    /// tract, pure Rust, one thread: the same bytes on every run and every machine. The default.
    #[default]
    Tract,
    /// ONNX Runtime on the CPU, four threads per stage; needs libonnxruntime.so (ORT_DYLIB_PATH).
    Ort,
    /// ONNX Runtime's CUDA provider: the network on the GPU; needs the GPU image, `docker compose run --rm pipes-gpu`.
    Cuda,
}

impl DetectorBackendArg {
    /// The backend as the detector module names it.
    pub fn backend(self) -> pipes_kitti::camdet::Backend {
        use pipes_kitti::camdet::Backend;
        match self {
            DetectorBackendArg::Tract => Backend::Tract,
            DetectorBackendArg::Ort => Backend::Ort,
            DetectorBackendArg::Cuda => Backend::Cuda,
        }
    }
}

/// Arguments of `pipes bench-detector`: the detector alone, over the first
/// frames of a drive, so a model and a backend can be chosen by measurement
/// before a run commits to them. Decoding the PNGs is outside the timing;
/// the letterbox, the network and the decode are inside it, as they are in a
/// run's `camdet` stage.
#[derive(Args, Debug)]
pub struct BenchArgs {
    /// KITTI data root (holds `<date>/<drive>/image_02`).
    #[arg(long, env = "PIPES_KITTI_ROOT", default_value = DEFAULT_KITTI_ROOT)]
    pub kitti_root: PathBuf,
    /// The drive whose frames are timed.
    #[arg(long, default_value = "2011_09_26_drive_0005_sync", value_parser = parse_drive)]
    pub drive: String,
    /// How many frames, from the first.
    #[arg(long, default_value_t = 40)]
    pub frames: usize,
    /// The model.
    #[arg(long, value_enum, default_value_t = DetectorModelArg::YoloxNano)]
    pub model: DetectorModelArg,
    /// The runtime.
    #[arg(long, value_enum, default_value_t = DetectorBackendArg::Tract)]
    pub backend: DetectorBackendArg,
    /// CPU threads for the `ort` backend.
    #[arg(long, default_value_t = 4)]
    pub threads: usize,
    /// Which camera's frames.
    #[arg(long, default_value = "left")]
    pub camera: String,
    /// How the `cuda` backend picks cuDNN's convolution algorithms: `heuristic`
    /// (by rule, repeatable), `default` (one algorithm for every shape,
    /// repeatable) or `exhaustive` (timed, can differ from run to run).
    #[arg(long, value_enum, default_value_t = CudaSearchArg::Heuristic)]
    pub cuda_search: CudaSearchArg,
    /// Let the `cuda` backend round matmul inputs to TF32 on the tensor cores.
    #[arg(long)]
    pub tf32: bool,
}

/// [`pipes_kitti::camdet::CudaSearch`] on the command line.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, ValueEnum)]
pub enum CudaSearchArg {
    /// cuDNN's ranking by shape; no timing, repeatable.
    #[default]
    Heuristic,
    /// cuDNN's default algorithm for every shape.
    Default,
    /// Every algorithm timed on the first frame, the fastest kept.
    Exhaustive,
}

impl CudaSearchArg {
    /// The choice as the detector module names it.
    pub fn search(self) -> pipes_kitti::camdet::CudaSearch {
        use pipes_kitti::camdet::CudaSearch;
        match self {
            CudaSearchArg::Heuristic => CudaSearch::Heuristic,
            CudaSearchArg::Default => CudaSearch::Default,
            CudaSearchArg::Exhaustive => CudaSearch::Exhaustive,
        }
    }

    /// The `cuda` backend's knobs, with `tf32`.
    pub fn options(self, tf32: bool) -> pipes_kitti::camdet::CudaOptions {
        pipes_kitti::camdet::CudaOptions {
            search: self.search(),
            tf32,
        }
    }
}

/// `--lidar-detector`: whether the frozen lidar detector runs.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, ValueEnum)]
pub enum LidarDetectorArg {
    /// Run PointPillars on every sweep in a `pillars` stage and name the tracks it finds cars, pedestrians and cyclists.
    On,
    /// No lidar detector: structures tracked, named by the cameras alone. The default.
    #[default]
    Off,
}

impl LidarDetectorArg {
    /// The `run.json` spelling.
    pub fn name(self) -> &'static str {
        match self {
            LidarDetectorArg::On => "on",
            LidarDetectorArg::Off => "off",
        }
    }
}

/// Which runtime `--lidar-backend` runs the lidar detector on. No `tract`:
/// the network is 70 GFLOP a sweep, which is a GPU's job or, slowly, a
/// thread pool's.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, ValueEnum)]
pub enum LidarBackendArg {
    /// ONNX Runtime's CUDA provider: the network on the GPU. The default; needs the GPU image, `pipes-gpu`.
    #[default]
    Cuda,
    /// ONNX Runtime on the CPU, four threads: about 240 ms a sweep, so a real-time run drops more than half of them; for a machine without a GPU, or a slowed replay.
    Ort,
}

impl LidarBackendArg {
    /// The backend as the detector modules name it.
    pub fn backend(self) -> pipes_kitti::camdet::Backend {
        use pipes_kitti::camdet::Backend;
        match self {
            LidarBackendArg::Cuda => Backend::Cuda,
            LidarBackendArg::Ort => Backend::Ort,
        }
    }
}

/// `--lidar-views`: which way the lidar detector looks.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, ValueEnum)]
pub enum LidarViewsArg {
    /// All the way round on `cuda`, ahead on `ort`. The default.
    #[default]
    Auto,
    /// One pass a sweep, over the 69 m ahead: what the network was trained on.
    Ahead,
    /// Two passes a sweep: ahead, and the sweep turned half round, for the 69 m behind.
    Around,
}

impl LidarViewsArg {
    /// The views a backend's run takes. `auto` looks behind as well where
    /// a second pass fits inside a sweep, which is the GPU; on the CPU one
    /// pass already takes longer than a sweep lasts.
    pub fn views(
        self,
        backend: pipes_kitti::camdet::Backend,
    ) -> &'static [pipes_kitti::pillars::View] {
        use pipes_kitti::camdet::Backend;
        use pipes_kitti::pillars::View;
        let around = match self {
            LidarViewsArg::Auto => backend == Backend::Cuda,
            LidarViewsArg::Ahead => false,
            LidarViewsArg::Around => true,
        };
        if around {
            &View::AROUND
        } else {
            &[View::Ahead]
        }
    }
}

/// Arguments of `pipes bench-lidar`: the lidar detector alone, over the
/// first sweeps of a drive, on `bench-detector`'s terms. Reading the sweeps
/// is outside the timing; the pillars, the network and the decode are
/// inside it, as they are in a run's `pillars` stage.
#[derive(Args, Debug)]
pub struct BenchLidarArgs {
    /// KITTI data root (holds `<date>/<drive>/velodyne_points`).
    #[arg(long, env = "PIPES_KITTI_ROOT", default_value = DEFAULT_KITTI_ROOT)]
    pub kitti_root: PathBuf,
    /// The drive whose sweeps are timed.
    #[arg(long, default_value = "2011_09_26_drive_0005_sync", value_parser = parse_drive)]
    pub drive: String,
    /// How many sweeps, from the first.
    #[arg(long, default_value_t = 40)]
    pub sweeps: usize,
    /// The runtime.
    #[arg(long, value_enum, default_value_t = LidarBackendArg::Cuda)]
    pub backend: LidarBackendArg,
    /// Which way the network looks: `ahead` (one pass a sweep), `around`
    /// (a second pass over the sweep turned half round, for what is behind)
    /// or `auto`: `around` on `cuda`, `ahead` on `ort`.
    #[arg(long, value_enum, default_value_t = LidarViewsArg::Auto)]
    pub views: LidarViewsArg,
    /// CPU threads for the `ort` backend.
    #[arg(long, default_value_t = 4)]
    pub threads: usize,
    /// How the `cuda` backend picks cuDNN's convolution algorithms.
    #[arg(long, value_enum, default_value_t = CudaSearchArg::Heuristic)]
    pub cuda_search: CudaSearchArg,
    /// Let the `cuda` backend round matmul inputs to TF32 on the tensor cores.
    #[arg(long)]
    pub tf32: bool,
    /// Also print the boxes found in this sweep (its frame number), one a
    /// line: class, score, centre, size and heading.
    #[arg(long)]
    pub show: Option<usize>,
    /// Start each sweep this many ms after the one before, instead of back
    /// to back: at the lidar's 103 the GPU idles between sweeps, as it does
    /// in a run, and a sweep costs what it costs there. 0 (the default) runs
    /// them back to back.
    #[arg(long, default_value_t = 0)]
    pub period_ms: u64,
}

/// Arguments of `pipes drives`.
///
/// `--drive` is an unvalidated name everywhere else, and nothing else in the
/// binary could answer "what can I pass?": this subcommand is that
/// answer, so it takes the same `--kitti-root`, with the same environment
/// override, and nothing else it does not need.
#[derive(Args, Debug)]
pub struct DrivesArgs {
    /// KITTI data root (holds `<date>/<drive>/image_02`). Docker sets it;
    /// `KITTI_DATA=<path>` in `.env` mounts a folder you already have.
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

/// Whether to replay the drive's OXTS rows -- the GPS/IMU unit, the
/// vehicle's own speed and turn rate -- as a third producer, for the fusion
/// to subtract from every track.
///
/// `auto` replays them exactly when the drive has an `oxts/` directory and
/// the fusion runs, which is their only consumer; `on` makes a drive without
/// one an error; `off` ignores one that is there, which is how a run shows
/// what the turning problem looked like before the stream existed.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, ValueEnum)]
pub enum OxtsArg {
    /// Replay the GPS/IMU rows exactly when the drive has an `oxts/` directory and the fusion runs.
    #[default]
    Auto,
    /// Require the GPS/IMU rows; a drive without them is an error.
    On,
    /// Never replay them, even when the drive has them.
    Off,
}

impl OxtsArg {
    /// The CLI / `run.json` spelling.
    pub fn name(self) -> &'static str {
        match self {
            OxtsArg::Auto => "auto",
            OxtsArg::On => "on",
            OxtsArg::Off => "off",
        }
    }
}

/// Whether to replay the drive's right colour camera (`image_03`) as a
/// fourth producer, with its own detector stage, for the fusion to read
/// beside the left one.
///
/// `auto` replays it exactly when the drive has an `image_03/` directory and
/// the detector runs, which is its only consumer; `on` makes a drive without
/// one an error; `off` ignores one that is there, which is how a run shows
/// what one camera alone makes of the same drive.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, ValueEnum)]
pub enum RightCameraArg {
    /// Replay the right camera exactly when the drive has an `image_03/` directory and the detector runs.
    #[default]
    Auto,
    /// Require the right camera; a drive without it is an error.
    On,
    /// Never replay it, even when the drive has it.
    Off,
}

impl RightCameraArg {
    /// The CLI / `run.json` spelling.
    pub fn name(self) -> &'static str {
        match self {
            RightCameraArg::Auto => "auto",
            RightCameraArg::On => "on",
            RightCameraArg::Off => "off",
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
    /// Stop the chain at the detections: no answer.
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
    /// Send `proc`'s bare frame reference: no class from the cameras.
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
/// `grpc` by default: the run streams to the viewer listening on port 9876,
/// the one open on the host under Docker; outside Docker it opens one itself
/// when none is listening. `rrd` writes the same recording to a file for
/// later.
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
    /// Stream to the Rerun viewer, which must be open on the host.
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
/// Two tiers. The sixteen `--help` shows are what a reader needs to see the
/// pipeline and to make it fail on purpose: `--cap` and `--consumer-delay-ms`
/// act on the camera stage whose output reaches the answer, so slowing the
/// camera reaches the fusion whichever stage that is. The rest (`hide = true`)
/// are the experiment's knobs -- the camera queue's policy, the replay rate,
/// which stages of the lidar chain run -- still accepted, still recorded in
/// `run.json`, but not an invitation on the first screen.
#[derive(Args, Debug, Clone)]
pub struct RunArgs {
    /// KITTI data root (holds `<date>/<drive>/image_02`). Docker sets it;
    /// `KITTI_DATA=<path>` in `.env` mounts a folder you already have.
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
    /// to feed). The other stage does not run. The right camera's
    /// `cam1->camdet` is sized the same.
    ///
    /// Omitted, it is 1 on `cam0->camdet` and 4 on `cam0->proc`. The
    /// detector's 1 is chosen. At about 41 ms a frame against the camera's
    /// 103 ms period, a healthy real-time run rarely has a frame waiting:
    /// depths 1 and 4 both delivered all 154 frames of drive_0005, to the
    /// same bytes. Once the detector is slower than the camera, a deeper
    /// queue delivers no more frames, only older ones: when the detector
    /// took 70 to 90 ms a frame, at `--consumer-delay-ms 60 --pair-wait-ms
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
    /// service time. It slows the left camera only. With `--right-camera
    /// off`, `--consumer-delay-ms 100 --cap 1 --pair-wait-ms 300` makes the
    /// camera lose about a quarter of drive_0005's frames (38 of 154 on one
    /// run), and the fusion expires exactly those sweeps, each one
    /// `pair_dropped`. With the right camera on they are answered from its
    /// frame instead, all 154; at 2000 ms the left camera is declared silent
    /// and the right one answers alone.
    #[arg(long, default_value_t = 0)]
    pub consumer_delay_ms: u64,
    /// `proc` reuses one output buffer instead of allocating one per frame.
    #[arg(long, hide = true)]
    pub reuse_output: bool,
    /// Replay the drive's Velodyne sweeps as a second producer through the
    /// same admission. `auto` (the default) does it exactly when the drive has
    /// a `velodyne_points/` directory; `on` makes a drive without one an
    /// error; `off` ignores one that is there. See `pipes drives` for which
    /// drives carry lidar.
    #[arg(long, value_enum, default_value_t = LidarArg::Auto, hide = true)]
    pub lidar: LidarArg,
    /// Replay the drive's OXTS rows -- the GPS/IMU unit: the vehicle's own
    /// speed and turn rate, one row per frame -- as a third producer through
    /// the same admission, for the fusion to subtract from every track so a
    /// stationary object stays put in a turn and velocities are over ground.
    /// `auto` (the default) does it exactly when the drive has an `oxts/`
    /// directory and `--track` is on; `on` makes a drive without one an
    /// error; `off` ignores one that is there.
    #[arg(long, value_enum, default_value_t = OxtsArg::Auto, hide = true)]
    pub oxts: OxtsArg,
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
    /// frame whose instant falls inside it, and answer with every track
    /// (distance, closing speed, time to contact, age, image box), the
    /// nearest in the path flagged.
    ///
    /// No effect on a drive without lidar.
    #[arg(long, value_enum, default_value_t = TrackArg::On)]
    pub track: TrackArg,
    /// Run a frozen, pretrained camera detector (YOLOX, 80 COCO classes) on
    /// each camera's frames in a stage of its own, `camdet` and
    /// `camdet-right`, and fuse what it found: each pair's camera half is the
    /// frame's detections -- box, class, confidence -- instead of a bare
    /// frame reference.
    ///
    /// Visible, one of the sixteen flags on this page, because it decides
    /// whether the cameras contribute anything to the answer. The weights
    /// are the file `--detector-model` names, in the image's `models/`,
    /// checked against a pinned sha256; a missing file stops the run. The
    /// left camera's stage reads its own queue, `cam0->camdet`, and with the
    /// detector on that is the queue `--cap` sizes and the stage
    /// `--consumer-delay-ms` slows. No effect without `--track on`.
    #[arg(long, value_enum, default_value_t = DetectorArg::On)]
    pub detector: DetectorArg,
    /// Which network the detector runs: YOLOX-Nano at 640x192 (the default,
    /// 0.9 M parameters), YOLOX-S at 640x192 or at 1024x320, YOLOX-M at
    /// 1024x320 (25 M), or YOLOX-L (54 M) or YOLOX-X (99 M) at 1248x384, the
    /// frame's own resolution; the last two have 16-bit weights and are for
    /// `--detector-backend cuda`. Each is a file under `models/` with its
    /// own pinned sha256, derived from the published weights by
    /// `scripts/prepare_models.py`; a missing one stops the run naming it.
    /// `pipes bench-detector` times them.
    #[arg(long, value_enum, default_value_t = DetectorModelArg::YoloxNano)]
    pub detector_model: DetectorModelArg,
    /// Which runtime runs it: `tract` (pure Rust, one thread, the same bytes
    /// on every machine), `ort` (ONNX Runtime on the CPU, four threads per
    /// detector stage) or `cuda` (ONNX Runtime on the GPU). The last two load
    /// the library `ORT_DYLIB_PATH` names, which the Docker images carry;
    /// `cuda` needs the GPU image (`docker compose run --rm pipes-gpu ...`).
    #[arg(long, value_enum, default_value_t = DetectorBackendArg::Tract)]
    pub detector_backend: DetectorBackendArg,
    /// CPU threads of the `ort` backend's pool, per detector stage. Hidden:
    /// four is the measured sweet spot on the development host, and a run
    /// records what it used.
    #[arg(long, default_value_t = 4, hide = true)]
    pub detector_threads: usize,
    /// How the `cuda` backend picks cuDNN's convolution algorithms:
    /// `heuristic` (the default: by rule, the same every run), `default`
    /// (cuDNN's one algorithm for every shape, also repeatable, and faster
    /// for YOLOX-M on the development host) or `exhaustive` (timed on the
    /// first frame; can differ from run to run). Hidden; recorded in
    /// run.json.
    #[arg(long, value_enum, default_value_t = CudaSearchArg::Heuristic, hide = true)]
    pub cuda_search: CudaSearchArg,
    /// Let the `cuda` backend round matmul inputs to TF32 on the tensor
    /// cores: faster, repeatable on one machine, and no longer the CPU's
    /// arithmetic. Hidden; recorded in run.json.
    #[arg(long, hide = true)]
    pub cuda_tf32: bool,
    /// Replay the drive's right colour camera (`image_03`) as a fourth
    /// producer, run the detector on its frames in a second stage,
    /// `camdet-right`, on its own thread, and fuse both cameras: a track
    /// keeps its class only when both cameras match it with the same class
    /// (one camera's match is recorded as `one_camera`, which the answer does
    /// not trust), a detection no track matched is kept only when both
    /// cameras see it and gets a range from how far apart their boxes are,
    /// every confirmed track's lidar distance is checked against that range,
    /// and a sweep whose left frame was lost is answered from the right one.
    /// `auto` (the default) does it exactly when the drive has an `image_03/`
    /// directory and the detector runs; `on` makes a drive without one an
    /// error; `off` ignores one that is there. No effect without `--track
    /// on --detector on`.
    #[arg(long, value_enum, default_value_t = RightCameraArg::Auto)]
    pub right_camera: RightCameraArg,
    /// Run a frozen, pretrained lidar detector (PointPillars, trained on
    /// KITTI's cars, pedestrians and cyclists) on every raw sweep in a
    /// `pillars` stage of its own, and name the tracks from it: a track
    /// whose centroid lies in one of its boxes takes the box's class, beside
    /// whatever the cameras said. `off` is the default; a bare `pipes-gpu`
    /// passes `on`. On the GPU it runs twice a sweep, the second time on the sweep
    /// turned half round, so it names what is behind the vehicle as well.
    /// The weights are `models/pointpillars-kitti-fp16.onnx` on the GPU and
    /// `models/pointpillars-kitti.onnx` on the CPU, each checked against a
    /// pinned sha256; the Docker build derives them. Needs `--track on`.
    /// The network is a GPU's job (`--lidar-backend`).
    #[arg(long, value_enum, default_value_t = LidarDetectorArg::Off)]
    pub lidar_detector: LidarDetectorArg,
    /// Which runtime runs it: `cuda` (ONNX Runtime's CUDA provider on the
    /// GPU, the default; `docker compose run --rm pipes-gpu`) or `ort` (ONNX
    /// Runtime on the CPU: about 240 ms a sweep, longer than a sweep lasts
    /// and longer than the fusion waits, so at real time it drops more than
    /// half the sweeps and the boxes of the rest come late; the fusion goes
    /// on without their names and says so).
    #[arg(long, value_enum, default_value_t = LidarBackendArg::Cuda)]
    pub lidar_backend: LidarBackendArg,
    /// Which way the lidar detector looks: `ahead` (one pass a sweep, over
    /// the 69 m ahead), `around` (a second pass over the sweep turned half
    /// round, for the 69 m behind) or `auto`, the default: `around` on
    /// `cuda`, `ahead` on `ort`, where one pass already takes longer than a
    /// sweep. Hidden; recorded in run.json.
    #[arg(long, value_enum, default_value_t = LidarViewsArg::Auto, hide = true)]
    pub lidar_views: LidarViewsArg,
    /// CPU threads of the lidar detector's `ort` pool. Hidden; recorded in
    /// run.json.
    #[arg(long, default_value_t = 4, hide = true)]
    pub lidar_threads: usize,
    /// Test hook: an artificial per-sweep delay, in ms, in the `pillars`
    /// stage, on `--consumer-delay-ms`'s terms. A delay of seconds breaks
    /// the lidar detector on purpose, so a run can show the fusion carrying
    /// on without its names. Hidden from `--help`; used by
    /// `tests/lidar_detector.rs`.
    #[arg(long, hide = true, default_value_t = 0)]
    pub lidar_delay_ms: u64,
    /// How long the fusion waits for a camera frame that has not arrived yet,
    /// in ms. Omitted means **one sweep span**, taken from the sweep's own
    /// `Tov::Range` rather than assumed; `0` refuses immediately. The lidar
    /// detector's boxes are waited for inside the same deadline.
    ///
    /// `0` is the negative control and it is worth running, as
    /// `--pair-wait-ms 0 --consumer-delay-ms 50 --right-camera off`: the
    /// detector -- about 41 ms a frame on the development host, uncertified,
    /// and printed on each run's `camdet ms/frame` line -- then finishes each
    /// frame after its sweep has been processed. The camera path drops
    /// NOTHING, yet a fusion that refuses to wait reports all 154 sets
    /// expired (`pair_late`), and there is no answer. With `--detector off`
    /// the same was measured at `--consumer-delay-ms 100 --cap 16`, and at
    /// 50 ms the camera was mostly in time: 0 or 9 of 154 expired on two
    /// runs.
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
    /// Where the recording goes: `grpc` streams it to the Rerun viewer, which
    /// must be open on the host; `rrd` writes `runs/<name>/cam0.rrd` to open
    /// later with `rerun runs/<name>/cam0.rrd`.
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
    /// Test hook: an artificial per-frame delay, in ms, in the RIGHT
    /// camera's detector stage, on `--consumer-delay-ms`'s terms (slept
    /// inside its measured window). A delay of seconds breaks the right
    /// stream on purpose, so a run can show the left camera carrying on
    /// alone; `--consumer-delay-ms` does the same to the left. Hidden from
    /// `--help`; used by `tests/right_camera.rs`.
    #[arg(long, hide = true, default_value_t = 0)]
    pub right_delay_ms: u64,
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
    /// the tracks by source, the lidar in 3D, and four tabs: Demo (the time
    /// to collision and the pairing), Fusion, Pipeline and Log. `off` keeps
    /// the raw camera frames only. With `--rerun null|off` the drawing goes
    /// to `runs/<name>/dashboard.rrd`.
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
