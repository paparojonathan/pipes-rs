//! `pipes run`: the full pipeline (spec G.6) — driver → Admission →
//! `cam0->proc` + `cam0->rerun` → evidence → `rec` — followed by the summary
//! and the H.3 invariants.
//!
//! M11: a drive that carries `velodyne_points/` also replays its lidar, on a
//! thread of its own, through the **same** `Admission` and onto a `velo->cloud`
//! edge. Two producers is the point: one mutex serialises them, so
//! `arrival_seq` finally means something it could not mean with one driver.
//! The camera path is untouched by it — the edges are stream-routed, so a
//! sweep can never reach `cam0->proc`.
//!
//! M12/M13: that sweep then goes down a CHAIN — `reduce` makes a voxel cloud
//! of it, `detect` finds the objects in that cloud — and each link is a stage
//! that consumes Arrow and produces Arrow through the same `Admission`. The
//! shutdown order below is the thing to read carefully: with a chain it stops
//! being "producers, queues, consumers" and becomes strictly one rung at a
//! time, because closing a queue while something upstream is still admitting
//! into it fails SILENTLY.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use arrow::array::RecordBatch;
use arrow::datatypes::SchemaRef;
use arrow::ipc::writer::StreamWriter;
use pipes_core::alloc::{
    bytes_alloc, set_stage_slot, SLOT_CAMDET, SLOT_DET_CLOUD, SLOT_DRIVER, SLOT_UNTRACKED,
    SLOT_VELO, SLOT_VELO_DRIVER,
};
use pipes_core::clock::{now, wall_now_unix_ns, ClockModel, HostTime};
use pipes_core::evidence::{Event, Evidence, Outcome, RowCtx};
use pipes_core::host::{opt_out_of_power_throttling, PowerThrottling};
use pipes_core::queue::{BoundedQueue, QueuePolicy};
use pipes_core::sample::{Sample, StreamId};
use pipes_core::stats::{percentile, percentile_sorted};
use pipes_kitti::calib::{Calib, CalibError};
use pipes_kitti::cam0::{
    median_period_ns, png_dimensions, Cam0Driver, DriverError, DriverEvent, RunTotals,
};
use pipes_kitti::camdet::{
    cam_det_schema, CamDetError, Model, INPUT_H, INPUT_W, MODEL_FILE, MODEL_NAME, NMS_IOU,
    PAD_VALUE, SCORE_THRESHOLD,
};
use pipes_kitti::detect::{
    detect_params_note, detect_schema, DETECTION_LANES, MIN_CLUSTER_VOXELS, PERSISTENCE_GATE_M,
    RANGE_LIMIT_M,
};
use pipes_kitti::frame::cam0_schema;
use pipes_kitti::fuse::FUSION_RULE;
use pipes_kitti::layout::{frames_phrase, Gap, ABSENT_IN_SOURCE};
use pipes_kitti::state::{
    state_schema, CORRIDOR_HALF_WIDTH_M, OBJECT_BYTES, OBJECT_LANES, VEHICLE_WIDTH_M,
};
use pipes_kitti::track::{
    cam_ref_schema, gate_note, track_schema, Pairing, MAX_MISSES, MIN_OBSERVATIONS, TRACK_LANES,
    YAW_RATE_MAX_RPS,
};
use pipes_kitti::velo::{has_velodyne, velo_schema, VeloDriver, VeloError};
use pipes_kitti::voxel::{voxel_schema, voxel_size_note, VoxelSize};
use rerun::{RecordingStream, RecordingStreamBuilder};
use serde::Serialize;

use crate::admission::{Admission, Edge};
use crate::cli::{
    DetectArg, DetectorArg, LidarArg, PolicyArg, ReduceArg, RerunArg, RunArgs, TrackArg,
    DEFAULT_SPIN_WINDOW_MS,
};
use crate::consumers::{
    answer_words, camdet_thread, cloud_thread, detect_thread, obj_thread, proc_thread,
    reduce_thread, rerun_thread, state_sink_thread, state_thread, track_thread, CamDetCfg,
    CamDetReport, CamRefOut, CloudCfg, CloudReport, DetectCfg, DetectReport, FuseTally, ObjCfg,
    ObjReport, ProcCfg, ProcReport, ReduceCfg, ReduceReport, RerunError, RerunMode, RerunReport,
    ShapeRow, StateCfg, StateReport, StateSinkCfg, StateSinkReport, TrackCfg, TrackReport,
    SWEEP_RGBA, VOXEL_RGBA,
};
use crate::dashboard::{entity, Mode};
use crate::record::{rec_thread, Bounds, Dashboard, EvRow, EvidenceSink, RecError};
use crate::summary::{rss_bytes, summarize, RunSummary};
use crate::viewer::{
    frozen_file, viewer_thread, Freeze, ViewerQueue, ViewerReport, VIEWER_CAP, VIEWER_EDGE,
};

type BoxError = Box<dyn std::error::Error + Send + Sync>;

/// Capacity of the `velo->cloud` queue, in sweeps.
///
/// Capacity is retained memory here in a way it is not on the camera edge. A
/// KITTI sweep is 16 B per point and the point count VARIES -- measured on
/// drive_0005, 98,532 to 124,122 points, so 1.50 to 1.89 MiB, mean 1.86 MiB
/// -- against a 1242x375 RGB frame's fixed 1,397,250 B (1.33 MiB). The queue
/// holds `Arc<Sample>`, so every slot keeps a whole point buffer alive:
/// `--cap 16` on the camera edge costs 21 MiB, the same number here up to
/// 30 MiB, and which end of that range you get is the scene's business.
///
/// 2, deliberately, and not 1 or 16:
/// * 1 turns every scheduling hiccup into an eviction, so the edge would
///   measure the OS scheduler rather than the consumer.
/// * 16 buys latency the consumer cannot use. A sweep is 103.27 ms of
///   geometry (measured mean on drive_0005); the back of a 16-deep queue is
///   1.65 s of stale world, which no downstream stage would want even if it
///   arrived.
/// * 2 is one sweep being read and one in flight: 3.7 MiB retained at
///   drive_0005's mean sweep, 3.8 MiB at its largest.
pub const VELO_CAP: usize = 2;

/// Overflow policy of the `velo->cloud` queue.
///
/// `DropOldest`, fixed rather than taken from `--policy`. Three reasons, in
/// order of weight:
/// 1. It makes `Admission::admit` non-blocking on this edge **by
///    construction** (a Drop* edge gets `try_push`), so the lidar consumer can
///    never stall its own producer, and the stall question does not arise.
/// 2. The freshest sweep is the one a geometry consumer wants; an evicted
///    sweep is recorded as a drop row and counted in the edge's invariant, so
///    the loss is measured rather than hidden.
/// 3. `--policy` is the camera experiment's independent variable, and the 39
///    committed rows vary it. Letting it reach a second edge would change what
///    those rows mean.
pub const VELO_POLICY: QueuePolicy = QueuePolicy::DropOldest;

/// Capacity of the `det->cloud` queue, in derived clouds.
///
/// 4, against [`VELO_CAP`]'s 2, for one reason: a reduced cloud is several
/// times smaller than the sweep it came from, so four slots retain about what
/// two raw slots do. The argument is memory, not latency — the latency
/// argument against a deep queue is the same one [`VELO_CAP`] makes, and it is
/// why this is 4 and not 16.
pub const DET_CAP: usize = 4;

/// Overflow policy of the `det->cloud` queue.
///
/// `DropOldest`, and this one is load-bearing rather than a preference. The
/// producer on this edge is a **consumer thread** — `reduce` admits into it
/// while draining `velo->reduce` — so a `Block` policy would let a slow
/// downstream stage stall `reduce`, which would stall its own queue, which
/// would eventually push back onto the velodyne driver. A Drop* edge gets
/// `try_push` in `Admission::admit` and therefore cannot block **by
/// construction**, so the derived chain can never back-pressure a sensor, let
/// alone the camera path the committed record was measured on.
pub const DET_POLICY: QueuePolicy = QueuePolicy::DropOldest;

/// Capacity of the `obj->sink` queue, in detection batches.
///
/// 4, matching [`DET_CAP`], and the reason is NOT the memory argument that
/// chose that one. A detection batch is about 8 KB against a reduced cloud's
/// 492 KB, so sixty slots would still cost less than one raw sweep and memory
/// stops being the constraint entirely. What is left is the latency argument
/// [`VELO_CAP`] makes -- the back of a deep queue is stale world nobody wants
/// -- plus one depth for the whole derived chain, so a backlog anywhere in it
/// is read against one number rather than three.
pub const OBJ_CAP: usize = 4;

/// Overflow policy of the `obj->sink` queue.
///
/// `DropOldest`, and load-bearing for the same reason [`DET_POLICY`] is,
/// one rung further down. The producer on this edge is a consumer thread --
/// `detect` admits into it while draining `det->detect` -- so a `Block` policy
/// would let a slow sink stall `detect`, which would stall `reduce`, which
/// would eventually push back onto the velodyne driver. A Drop* edge gets
/// `try_push` in `Admission::admit` and therefore cannot block **by
/// construction**.
pub const OBJ_POLICY: QueuePolicy = QueuePolicy::DropOldest;

/// Capacity of `obj->track`, `track->state` and `state->sink`, in samples.
///
/// 4, matching [`OBJ_CAP`], and for its reason rather than the memory one:
/// past the detections the payloads are kilobytes and then bytes, so memory
/// stops being a constraint entirely and what is left is the latency argument
/// [`VELO_CAP`] makes — the back of a deep queue is stale world nobody wants —
/// plus one depth for the whole derived chain, so a backlog anywhere in it is
/// read against one number rather than six.
pub const CHAIN_CAP: usize = 4;

/// Capacity of `cam_det->track`, in camera samples: the detector's batches
/// under `--detector on`, `proc`'s bare references under `--detector off`.
///
/// **Deliberately larger than every other queue in this file, and deliberately
/// not `--cap`.** This edge is not a place where a policy should be exercised:
/// a sample is a few dozen bytes to a couple of kilobytes, and an eviction
/// here would destroy a pair that the camera path had already delivered — a
/// loss invented by the fusion and then reported as the camera's. The camera
/// path drops on its own queue -- `cam0->camdet` with the detector,
/// `cam0->proc` without it, sized by `--cap` either way ([`camera_queues`]) --
/// and a sample only exists here because that stage FINISHED a frame.
///
/// 64 is about 1.5 s of camera at 10 Hz, which is more head start than the
/// camera can take: its frame is due 41.13 ms before the sweep it belongs to,
/// and the detector spends about 90 ms of that head start and more.
pub const CAM_REF_CAP: usize = 64;

/// Capacity of `cam0->camdet`, the frozen detector's queue, when `--cap` is
/// omitted: **1**. The reasons are `--cap`'s own doc comment (`cli.rs`), where
/// a reader of `--help` meets the trade-off.
pub const CAMDET_CAP: usize = 1;

/// Capacity of `cam0->proc` when `--cap` is omitted: 4, `--cap`'s default
/// since M4, so a run without the detector is what it was.
pub const PROC_CAP: usize = 4;

/// One camera consumer's queue and stage, resolved from the arguments.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CameraQueue {
    /// `cam0->camdet` or `cam0->proc`.
    pub edge: &'static str,
    /// Capacity, in frames.
    pub cap: usize,
    /// Overflow policy.
    pub policy: QueuePolicy,
    /// The artificial per-frame sleep, ms, inside the stage's measured window.
    pub delay_ms: u64,
}

/// The camera consumer whose output reaches the answer, and its queue:
/// **exactly one of the two runs**. The detector when it runs; `proc`
/// otherwise, because with the detector on nothing reads `proc`'s output,
/// so it would only cost a queue, a grayscale buffer per frame and rows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CameraQueues {
    queue: CameraQueue,
    detector: bool,
}

impl CameraQueues {
    /// The queue the camera knobs set: the one camera consumer's.
    pub fn knobs(&self) -> CameraQueue {
        self.queue
    }

    /// `cam0->proc`; `None` when the detector runs.
    pub fn proc(&self) -> Option<CameraQueue> {
        (!self.detector).then_some(self.queue)
    }

    /// `cam0->camdet`; `None` when the detector does not run.
    pub fn camdet(&self) -> Option<CameraQueue> {
        self.detector.then_some(self.queue)
    }
}

/// **The camera knobs' rule**, in one place: `--cap`, `--policy`,
/// `--block-max-wait-ms` and `--consumer-delay-ms` act on the camera
/// consumer whose output reaches `track` -- `cam0->camdet` when the detector
/// runs (`detector`), `cam0->proc` when it does not -- so "slow the camera,
/// and the answer loses exactly those frames" holds whichever stage feeds the
/// fusion. The other stage does not run.
///
/// "When it does not" includes every run the detector cannot reach --
/// `--track off`, a drive without lidar -- where `proc` is the only camera
/// consumer there is, so the committed record's runs mean what they meant.
pub fn camera_queues(a: &RunArgs, detector: bool) -> CameraQueues {
    let policy = match a.policy {
        PolicyArg::DropOldest => QueuePolicy::DropOldest,
        PolicyArg::DropNewest => QueuePolicy::DropNewest,
        PolicyArg::Block => QueuePolicy::Block {
            max_wait: Duration::from_millis(a.block_max_wait_ms),
        },
    };
    let (edge, default_cap) = if detector {
        ("cam0->camdet", CAMDET_CAP)
    } else {
        ("cam0->proc", PROC_CAP)
    };
    CameraQueues {
        queue: CameraQueue {
            edge,
            cap: a.cap.unwrap_or(default_cap),
            policy,
            delay_ms: a.consumer_delay_ms,
        },
        detector,
    }
}

/// Overflow policy of the chain's remaining edges.
///
/// `DropOldest`, load-bearing for the reason [`DET_POLICY`] is, repeated at
/// every rung: every producer on these edges is a consumer thread, so a
/// `Block` policy anywhere would let a slow stage stall the one above it, and
/// eventually push back onto a sensor. A Drop* edge gets `try_push` in
/// `Admission::admit` and therefore cannot block **by construction**.
pub const CHAIN_POLICY: QueuePolicy = QueuePolicy::DropOldest;

/// Written into `runs/<name>/` as the final action of a run that completed and
/// wrote its artifacts. Its absence means the evidence may be truncated.
pub const COMPLETE_MARKER: &str = "_COMPLETE";

/// Shared per-run context, passed to every stage as `Arc<RunCtx>`.
pub struct RunCtx {
    pub run_id: String,
    pub t0_host: HostTime,
    pub epoch: u32,
    pub n_frames: usize,
}

impl RunCtx {
    pub fn row_ctx(&self) -> RowCtx {
        RowCtx {
            run_id: self.run_id.clone(),
            t0_host: self.t0_host,
            epoch: self.epoch,
        }
    }
}

/// Contents of `runs/<name>/run.json`.
#[derive(Serialize)]
struct RunJson<'a> {
    clock: &'a ClockModel,
    /// `--rate` as given (`"1"`, `"10"`, `"inf"`): `clock.rate_factor` is `null` for `inf` in JSON.
    rate: String,
    git_sha: String,
    args: Vec<String>,
    spin_window_ms: f64,
    /// Capture date of the replayed scene. Provenance, not configuration: a
    /// run directory that names only the drive cannot be interpreted once a
    /// second scene exists, and KITTI's resolution is a property of the *date*.
    date: &'a str,
    drive: &'a str,
    n_frames: usize,
    /// Frame 0's width and height, read from the PNG header before the clock
    /// starts. `null` when that header could not be read, which is honest and
    /// harmless: nothing computes from these, they only say what was replayed.
    /// The 43 run directories written before these fields existed have neither,
    /// so a reader must treat all three as optional.
    width: Option<u32>,
    height: Option<u32>,
    rerun_mode: &'static str,
    /// `--lidar` as given (`auto` | `on` | `off`). Provenance: `auto` means
    /// the answer came from the disk, so the run directory has to say what
    /// was asked as well as what happened.
    lidar: &'static str,
    /// Sweeps replayed, or `null` when the run had no lidar stream. The 43
    /// run directories written before this existed have neither field.
    n_sweeps: Option<usize>,
    /// `--reduce` as given (`on` | `off`). Provenance: it decides whether the
    /// run had a stage-to-stage hand-off at all, so a directory that does not
    /// say cannot be interpreted.
    reduce: &'static str,
    /// The voxel edge the `reduce` stage used, or `null` when it did not run.
    /// The parameter is recorded with the run because the shrink ratio in
    /// `summary.json` is meaningless without it.
    voxel_size_m: Option<f32>,
    /// `--detect` as given (`on` | `off`). Provenance: it decides whether the
    /// chain has a third link at all.
    detect: &'static str,
    /// The validity bound and the cluster gate `detect` used, or `null` when
    /// it did not run. Recorded for the same reason `voxel_size_m` is: the
    /// detection count in `summary.json` means nothing without them, and a
    /// reader must not have to go and find the commit that chose them.
    detect_range_limit_m: Option<f32>,
    detect_min_cluster_voxels: Option<u32>,
    /// `--track` as given (`on` | `off`). Provenance: it decides whether the
    /// chain ends in an answer or in 177 rows of detections.
    track: &'static str,
    /// The pairing policy this run declared, or `null` when `track` was off.
    /// Recorded because "expired" and "degraded" mean nothing without them: a
    /// run that refused to wait and a run that waited a sweep produce very
    /// different counts from identical inputs.
    pair_wait_ms: Option<u64>,
    pair_stale_ms: Option<u64>,
    /// The corridor half-width the answer used, metres, or `null`.
    corridor_half_width_m: Option<f32>,
    /// `--detector` as given (`on` | `off`). Provenance: it decides whether
    /// the camera's half of every pair is what the camera saw or only when
    /// it looked.
    detector: &'static str,
    /// The model the detector ran, or `null` when it did not run: its name,
    /// the file, and the sha256 of the bytes that were loaded -- checked
    /// against the pinned one before the clock started, so "frozen" is a
    /// statement about this run and not only about the source.
    detector_model: Option<&'static str>,
    detector_model_file: Option<&'static str>,
    detector_model_sha256: Option<String>,
    /// The input the frames were letterboxed to, and the two thresholds the
    /// detections were cut at, or `null`. The counts in `summary.json` mean
    /// nothing without them.
    detector_input: Option<String>,
    detector_score_threshold: Option<f32>,
    detector_nms_iou: Option<f32>,
    /// The rule the fusion associated tracks with detections by, or `null`
    /// when no detector fed it. It has no parameter -- the threshold is
    /// derived -- so the rule is what is recorded.
    fusion_rule: Option<&'static str>,
    /// Whether the OS was told not to throttle this process while another
    /// window has the focus (`off`, `refused (os error N)`, `not
    /// applicable`): on Windows a throttled run's timings are a function of
    /// which window was in front. See `pipes_core::host`.
    power_throttling: String,
}

/// Frame 0's dimensions from its PNG header, or `(None, None)` with a warning.
///
/// Header-only, and called before `ClockModel::start_now`, so it stays outside
/// the pacing window (D16). A frame that cannot be read here is not fatal: the
/// run will report it frame by frame, and refusing to start over a provenance
/// field would be a worse trade than a `null`.
fn frame0_dims(driver: &Cam0Driver) -> (Option<u32>, Option<u32>) {
    // The first PNG on disk: frame 0, unless the camera's source lacks it.
    let path = driver.file_path(0).unwrap_or_else(|| driver.frame_path(0));
    match png_dimensions(&path) {
        Ok((w, h)) => (Some(w), Some(h)),
        Err(e) => {
            eprintln!(
                "run.json: no width/height, {} header unreadable: {e}",
                path.display()
            );
            (None, None)
        }
    }
}

/// Errors of a full run.
#[derive(Debug)]
pub enum RunError {
    Args(String),
    NoFrames,
    Driver(DriverError),
    Velo(VeloError),
    Io(std::io::Error),
    Json(serde_json::Error),
    Csv(csv::Error),
    Rec(RecError),
    Rerun(RerunError),
    Panicked(&'static str),
    /// `--track on` needs the date's calibration and could not read it.
    ///
    /// A hard failure rather than a fallback, because every plausible fallback
    /// here is a wrong answer that looks right: an identity for a missing
    /// `R_rect_00` costs 6.58 px and a zero fourth column costs 2.91 px, and
    /// both would ship. See `pipes_kitti::calib::CalibError`.
    Calib(CalibError),
    /// `--detector on` needs the frozen model and could not load it: the file
    /// is missing, unreadable, not the pinned sha256, or refused by the
    /// runtime.
    ///
    /// A hard failure on the calibration's terms. The fallback -- fusing with
    /// the bare frame reference -- is a different experiment, and a run that
    /// switched to it quietly would report a camera that contributed nothing
    /// as one that looked. The cause names `scriptsetch_model.ps1`.
    Model(CamDetError),
}

impl std::fmt::Display for RunError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            // Each arm names *this* layer only. The wrapped error is reached
            // through `source()` and printed by `main` as a `caused by:` chain,
            // so interpolating it here would print every inner message twice.
            RunError::Args(m) => write!(f, "{m}"),
            RunError::NoFrames => write!(f, "drive has no frames"),
            RunError::Driver(_) => write!(f, "the cam0 driver could not start"),
            RunError::Velo(_) => write!(f, "the velodyne driver could not start"),
            RunError::Io(_) => write!(f, "an i/o operation failed"),
            RunError::Json(_) => write!(f, "run.json/summary.json could not be written"),
            RunError::Csv(_) => write!(f, "an evidence CSV could not be written"),
            RunError::Rec(_) => write!(f, "the rec thread failed"),
            RunError::Rerun(_) => write!(f, "the rerun sink failed"),
            RunError::Panicked(t) => write!(f, "thread `{t}` panicked"),
            RunError::Calib(_) => write!(
                f,
                "--track on needs this date's lidar-to-camera calibration and it could not be read"
            ),
            RunError::Model(_) => write!(
                f,
                "--detector on needs the frozen camera detector's model and it could not be loaded"
            ),
        }
    }
}

impl std::error::Error for RunError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            RunError::Args(_) | RunError::NoFrames | RunError::Panicked(_) => None,
            RunError::Driver(e) => Some(e),
            RunError::Velo(e) => Some(e),
            RunError::Io(e) => Some(e),
            RunError::Json(e) => Some(e),
            RunError::Csv(e) => Some(e),
            RunError::Rec(e) => Some(e),
            RunError::Rerun(e) => Some(e),
            RunError::Calib(e) => Some(e),
            RunError::Model(e) => Some(e),
        }
    }
}

impl From<CalibError> for RunError {
    fn from(e: CalibError) -> Self {
        RunError::Calib(e)
    }
}

impl From<CamDetError> for RunError {
    fn from(e: CamDetError) -> Self {
        RunError::Model(e)
    }
}

impl From<DriverError> for RunError {
    fn from(e: DriverError) -> Self {
        RunError::Driver(e)
    }
}
impl From<VeloError> for RunError {
    fn from(e: VeloError) -> Self {
        RunError::Velo(e)
    }
}
impl From<std::io::Error> for RunError {
    fn from(e: std::io::Error) -> Self {
        RunError::Io(e)
    }
}
impl From<serde_json::Error> for RunError {
    fn from(e: serde_json::Error) -> Self {
        RunError::Json(e)
    }
}
impl From<csv::Error> for RunError {
    fn from(e: csv::Error) -> Self {
        RunError::Csv(e)
    }
}
impl From<RecError> for RunError {
    fn from(e: RecError) -> Self {
        RunError::Rec(e)
    }
}
impl From<RerunError> for RunError {
    fn from(e: RerunError) -> Self {
        RunError::Rerun(e)
    }
}
impl From<rerun::RecordingStreamError> for RunError {
    fn from(e: rerun::RecordingStreamError) -> Self {
        RunError::Rerun(RerunError::Stream(e))
    }
}

/// Owns the evidence sink and the `rec` thread, and guarantees that both are
/// shut down on **every** exit path from [`execute`] — the normal one, an early
/// `?`, or an unwind.
///
/// This is the fix for the failure this project cares most about: if `execute`
/// returns without closing the sink, `rec` blocks forever in `pop()` (which
/// yields `None` only when the queue is closed *and* drained), its `csv::Writer`
/// is never flushed, and the process tears down leaving `evidence.csv`
/// truncated mid-row — a file that claims to account for every frame but does
/// not, and that the Python tooling silently under-counts rather than rejects.
///
/// `Drop` alone is not enough: closing the sink without joining leaves `rec`
/// racing the process exit, so the guard owns the join handle too.
struct Recorder {
    sink: Arc<EvidenceSink>,
    handle: Option<std::thread::JoinHandle<Result<Vec<Evidence>, RecError>>>,
}

impl Recorder {
    /// Ordered shutdown on the success path: close the sink, join `rec`, and
    /// hand back the rows it collected.
    fn finish(mut self) -> Result<Vec<Evidence>, RunError> {
        self.sink.close();
        match self.handle.take() {
            Some(h) => Ok(h.join().map_err(|_| RunError::Panicked("rec"))??),
            None => Ok(Vec::new()),
        }
    }
}

impl Drop for Recorder {
    fn drop(&mut self) {
        // Only reached when `finish` was not called — an early return or an
        // unwind. `close` is idempotent, and the join lets `rec` flush before
        // the process goes away. A panic in `rec` itself is deliberately
        // swallowed here: we are already unwinding, and the original error is
        // the one worth reporting.
        self.sink.close();
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

/// Owns the viewer thread, which is the only code that calls the Rerun SDK's
/// `log` ([`crate::viewer`]).
///
/// Its queue is closed by the last producer that draws: the `rec` thread's
/// dashboard, once it has drawn the shutdown event, or `execute` itself when
/// there is no dashboard. On any other exit from `execute` the guard closes
/// it, so the thread drains what is queued and stops rather than waiting
/// forever. It is not joined there: the viewer may be the reason for the
/// early exit, and an error should not wait on it.
struct ViewerThread {
    queue: Arc<ViewerQueue>,
    handle: Option<std::thread::JoinHandle<(ViewerReport, RecordingStream)>>,
}

impl ViewerThread {
    /// Joins the thread once its queue has been closed, and hands back its
    /// count and the recording it logged to.
    fn join(mut self) -> Result<(ViewerReport, RecordingStream), RunError> {
        let h = self.handle.take().ok_or(RunError::Panicked("viewer"))?;
        h.join().map_err(|_| RunError::Panicked("viewer"))
    }
}

impl Drop for ViewerThread {
    fn drop(&mut self) {
        self.queue.close();
    }
}

/// Result of the camera's `storage_id` check: the driver against `proc`
/// (`--detector off` only) and `rerun` (not under `--rerun off`), so 3, 2
/// or 1 stages -- and at 1 there is nothing to compare.
pub enum StorageCheck {
    Ok {
        stages: u8,
    },
    Mismatch {
        seq: u64,
        driver: usize,
        proc: Option<usize>,
        rerun: Option<usize>,
    },
}

/// A producer -> consumer `storage_id` check over one stream: the recorded
/// producer rows against what the consumer re-derived from the real buffer.
///
/// Separate from [`StorageCheck`] rather than folded into it, because the two
/// answer questions about different streams and one bool would let one
/// stream's silence pass for the other's proof. That is not hypothetical: the
/// camera check selects driver rows by `stage == "driver"`, which excludes
/// every velodyne row, so bolting the lidar onto it would have compared
/// nothing and returned `Ok`.
///
/// Three of these run on a lidar drive, and they are deliberately three
/// separate answers rather than one:
/// * velodyne driver -> `cloud`, the raw sweep's zero-copy proof;
/// * velodyne driver -> `reduce`, the same proof extended to the transform
///   stage — this is what says `reduce` read the driver's buffer rather than a
///   copy of it;
/// * `reduce` -> `det-cloud`, the DERIVED buffer's proof, which is a different
///   claim: the reduced cloud is a new allocation, and this says that new
///   allocation was then shared rather than copied again.
pub enum VeloStorageCheck {
    Ok {
        /// Samples whose producer and consumer ids were compared.
        compared: usize,
    },
    Mismatch {
        seq: u64,
        driver: usize,
        consumer: Option<usize>,
    },
    /// Samples were admitted and none reached the consumer, so there was
    /// nothing to compare. Its own state, and treated as a failure of the
    /// check: a check with an empty population is not a check, and an empty
    /// population is exactly how the camera version would have "passed" on
    /// the lidar path.
    NotChecked { admitted: usize },
}

impl VeloStorageCheck {
    /// Whether the proof holds. `NotChecked` is **not** a pass.
    pub fn passed(&self) -> bool {
        matches!(self, VeloStorageCheck::Ok { .. })
    }

    /// The line a run prints for this check, given what it is a check of.
    fn line(&self, what: &str, unit: &str) -> String {
        match self {
            VeloStorageCheck::Ok { compared } => {
                format!("{what}: OK ({compared} {unit})")
            }
            VeloStorageCheck::Mismatch {
                seq,
                driver,
                consumer,
            } => format!(
                "{what}: MISMATCH (producer={driver:#x} consumer={:#x}) seq={seq}",
                consumer.unwrap_or(0)
            ),
            VeloStorageCheck::NotChecked { admitted } => {
                format!("{what}: NOT CHECKED ({admitted} {unit} admitted, none compared)")
            }
        }
    }
}

/// Result of joining what `reduce` believed it was reducing against what the
/// far end of the chain read out of `Sample::parent`.
///
/// **The check this project shipped without.** `cloud_thread` compared
/// `s.parent.map(|(stream, _)| stream)` — the stream half — and a run printed
/// `det-cloud parent = lidar on 154 of 154 samples` on the strength of it.
/// Replacing `parent: Some((s.stream, s.seq))` with `s.seq.wrapping_add(7)`,
/// so that every derived cloud named a sweep it did not come from, changed
/// nothing: 192 tests passed and the line still said 154 of 154. The stream
/// half says a result came from the lidar; the seq half says which
/// measurement, which is the half provenance is for.
///
/// The join needs both threads, so it happens here rather than in either of
/// them: `reduce` records `(derived seq -> sweep seq)` as it admits, the far
/// consumer records `(derived seq -> the parent seq it read)`, and every
/// sample that arrived must agree with the producer's record.
pub enum ParentCheck {
    Ok {
        /// Samples whose parent seq was compared against the producer's.
        compared: usize,
    },
    Mismatch {
        /// The derived sample that carried the wrong provenance.
        seq: u64,
        /// The sweep `reduce` actually reduced to produce it.
        want: u64,
        /// What the sample claimed, or `None` for no parent at all.
        got: Option<u64>,
    },
    /// Samples were produced and none reached the far end, so nothing was
    /// compared. Not a pass, for the same reason
    /// [`VeloStorageCheck::NotChecked`] is not: a check over an empty
    /// population is not a check.
    NotChecked { produced: usize },
}

impl ParentCheck {
    /// Whether the join holds. `NotChecked` is **not** a pass.
    pub fn passed(&self) -> bool {
        matches!(self, ParentCheck::Ok { .. })
    }

    /// The line a run prints for it: `who` is the consumer that read the
    /// provenance, `unit` what the producer consumed, `producer` the stage
    /// that wrote it.
    ///
    /// Parameterised rather than hard-coded because there are now two of these
    /// joins in a run, and one shared spelling would let either stand in for
    /// the other. The `det-cloud` wording is reproduced exactly -- it is
    /// pinned by `tests/reduce.rs`, and a line that moved would invalidate a
    /// pinned measurement rather than improve it.
    fn line(&self, who: &str, unit: &str, producer: &str) -> String {
        match self {
            ParentCheck::Ok { compared } => format!(
                "{who} parent seq = the {unit} {producer} read: OK ({compared} samples)"
            ),
            ParentCheck::Mismatch { seq, want, got } => match got {
                Some(got) => format!(
                    "{who} parent seq: MISMATCH (sample {seq} names {unit} {got}, {producer} built it from {unit} {want})"
                ),
                None => format!(
                    "{who} parent seq: MISMATCH (sample {seq} names no {unit} at all, {producer} built it from {unit} {want})"
                ),
            },
            ParentCheck::NotChecked { produced } => format!(
                "{who} parent seq: NOT CHECKED ({produced} produced by {producer}, none reached the far end)"
            ),
        }
    }
}

/// The last two links: the `track` fusion, the `state` answer and the leaf
/// that consumes it. `None` when `--track off`, or when there were no
/// detections to track.
pub struct TrackRun {
    /// What `track` consumed, produced, refused and allocated.
    pub report: TrackReport,
    /// What `state` made of it.
    pub state: StateReport,
    /// The far end of the whole chain.
    pub sink: StateSinkReport,
    /// `detect` -> `track`: the detection buffer was shared, not copied.
    pub in_check: VeloStorageCheck,
    /// `track` -> `state`: the track buffer was shared in turn.
    pub out_check: VeloStorageCheck,
    /// `state` -> `state-sink`: and so was the answer's.
    pub state_check: VeloStorageCheck,
    /// `track` -> `state`: each fused sample names the detections it was built
    /// from, seq included.
    pub parent_check: ParentCheck,
    /// `state` -> `state-sink`: the same one link further down.
    pub state_parent_check: ParentCheck,
    /// **The camera half of the provenance**, which `Sample::parent` cannot
    /// hold: every fused sample's `cam_seq` names a frame the camera driver
    /// really admitted, and one whose instant really falls inside that sweep's
    /// range -- or before it and no more than `--pair-stale-ms` before its
    /// trigger, for a set labelled stale. Without this the payload column is a number nothing reads, which
    /// is the exact failure `Sample::parent` already had once.
    pub pair_check: PairCheck,
    /// `camdet` -> `track`: the detector's batches were read where they were
    /// built. `None` under `--detector off`, whose bare references name no
    /// buffer to share.
    pub cam_check: Option<VeloStorageCheck>,
    /// Whether a detector's output reached the fusion at all, so the lines
    /// that read its result can say "off" rather than print zeros.
    pub detector: bool,
    /// `runs/<name>/fused.arrows`: every set the fusion produced.
    pub fused_file: PathBuf,
    /// Whether the calibration was loaded, so a run can say that its image
    /// rectangles are real rather than absent.
    pub calib: Option<Calib>,
    /// The instants a gap in either sensor's source left without a partner:
    /// empty on a drive whose sensors have a sample for every frame.
    pub gap_pairs: SourceGapPairs,
}

/// Where each of the drive's sweeps went on its way to the fusion: the half
/// of the `fusion sets` line that `track`'s own counters cannot see.
///
/// `track` counts only the sweeps it was handed -- completed, degraded,
/// expired or an error. A sweep evicted on a queue in front of it never got
/// there, and the line used to say nothing about it: a run with a stalled
/// viewer printed `COMPLETED 63 | ... | EXPIRED 4` over a drive of 154
/// sweeps and never said where the other 87 went (`obj->track` evicted
/// them). This names every sweep the fusion never saw and where it was lost.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct BeforeTrack {
    /// Sweeps the drive holds.
    pub sweeps: u64,
    /// Sweeps `track` was handed: its `delivered`.
    pub reached: u64,
    /// Sweeps the velodyne driver skipped because their deadline had passed,
    /// or could not read or wrap.
    pub skipped: u64,
    /// Sweeps evicted on each queue in front of `track`, in chain order.
    pub evicted_velo_reduce: u64,
    pub evicted_det_detect: u64,
    pub evicted_obj_track: u64,
    /// Sweeps `reduce` or `detect` failed on, so there was nothing to hand on.
    pub stage_errors: u64,
    /// Frames of the drive whose sweep the source never had: 4 on drive
    /// 0009. Frame slots on the lidar stream, never sweeps, so they are
    /// counted in [`BeforeTrack::frames`] and not in `sweeps`.
    pub absent_in_source: u64,
}

impl BeforeTrack {
    /// Frame slots on the lidar stream: the drive's sweeps and the frames
    /// absent in its source.
    pub fn frames(&self) -> u64 {
        self.sweeps + self.absent_in_source
    }

    /// Frame slots that never reached `track`, absent ones included.
    pub fn never_reached(&self) -> u64 {
        self.frames().saturating_sub(self.reached)
    }

    /// Whether the losses named add up to the frame slots that never
    /// arrived. The edge and stage invariants each balance one link; this is
    /// the sum of them over the whole lidar half of the chain.
    pub fn balanced(&self) -> bool {
        self.reached <= self.sweeps
            && self.skipped
                + self.evicted_velo_reduce
                + self.evicted_det_detect
                + self.evicted_obj_track
                + self.stage_errors
                + self.absent_in_source
                == self.never_reached()
    }

    /// The tail of the `fusion sets` line. A drive with a sweep for every
    /// frame reads as it always has; one with a gap in its source counts in
    /// frames, and names the absent ones as their own term.
    pub fn line(&self) -> String {
        let n = self.never_reached();
        let mut s = if self.absent_in_source == 0 {
            format!("NEVER REACHED track {n} of {} sweeps", self.sweeps)
        } else {
            format!(
                "NEVER REACHED track {n} of {} frames ({} sweeps + {} absent in source)",
                self.frames(),
                self.sweeps,
                self.absent_in_source
            )
        };
        if n > 0 || !self.balanced() {
            s.push_str(&format!(
                " = evicted on obj->track {} + det->detect {} + velo->reduce {} + skipped by the driver {} + stage errors {}",
                self.evicted_obj_track,
                self.evicted_det_detect,
                self.evicted_velo_reduce,
                self.skipped,
                self.stage_errors
            ));
            if self.absent_in_source > 0 {
                s.push_str(&format!(" + absent in source {}", self.absent_in_source));
            }
        }
        if !self.balanced() {
            s.push_str(" -- MISMATCH: those do not add up to the sweeps that never arrived");
        }
        s
    }
}

/// The instants a gap in either sensor's source left without a partner, and
/// what the fusion did with them. Pairing is by time, so the two streams stay
/// in step across a gap; this is where the run says which instants had
/// nothing to pair with, rather than leaving them to be inferred from counts.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
pub struct SourceGapPairs {
    /// The frames whose SWEEP is absent in the lidar's source, ascending.
    pub no_sweep: Vec<u64>,
    /// Of those, the ones whose camera frame reached `track` -- and found no
    /// sweep to be paired into.
    pub no_sweep_camera_reached_track: Vec<u64>,
    /// Of those, the ones a produced set named anyway: only a declared stale
    /// window can do that, for the sweep after the gap.
    pub no_sweep_camera_used_stale: Vec<u64>,
    /// The frames whose CAMERA frame is absent in the camera's source.
    pub no_frame: Vec<u64>,
    /// Of those, the sweeps that reached `track` and expired as
    /// `pair_absent_in_source`: no camera instant inside their range, and
    /// none coming.
    pub no_frame_sweep_expired: Vec<u64>,
}

impl SourceGapPairs {
    /// The lines printed under the `fusion sets` line: none on a drive whose
    /// sensors have a sample for every frame, which prints nothing new.
    pub fn lines(&self) -> Vec<String> {
        let mut out = Vec::new();
        if !self.no_sweep.is_empty() {
            let mut s = format!(
                "fusion sets for {}: none, no sweep in the lidar's source -- camera frames that reached track and had no sweep to pair with: {} of {}",
                frames_phrase(&self.no_sweep),
                self.no_sweep_camera_reached_track.len(),
                self.no_sweep.len()
            );
            if !self.no_sweep_camera_used_stale.is_empty() {
                s.push_str(&format!(
                    "; used as a declared stale frame by the sweep after: {}",
                    frames_phrase(&self.no_sweep_camera_used_stale)
                ));
            }
            out.push(s);
        }
        if !self.no_frame.is_empty() {
            out.push(format!(
                "fusion sets for {}: no camera frame in the source -- sweeps that reached track and expired with no frame to pair ({}): {} of {}",
                frames_phrase(&self.no_frame),
                Pairing::AbsentInSource.name(),
                self.no_frame_sweep_expired.len(),
                self.no_frame.len()
            ));
        }
        out
    }
}

/// The result of joining every fused sample's `cam_seq` against the camera
/// driver's own rows.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum PairCheck {
    /// Every completed or degraded set named a frame the camera admitted:
    /// `checked` completed sets whose frame's instant is inside the sweep,
    /// and `stale` degraded sets whose frame's instant is before the sweep
    /// and inside the declared stale window, counted back from its trigger.
    Ok {
        /// Completed sets compared.
        checked: usize,
        /// Degraded sets compared.
        stale: usize,
        /// The declared stale window, ms.
        window_ms: u64,
    },
    /// A sample named a camera frame that the camera driver never admitted.
    Unknown {
        /// The fused sample's seq.
        seq: u64,
        /// The frame it claimed.
        cam_seq: i64,
    },
    /// A sample named a frame whose instant is NOT inside its sweep's range:
    /// the pairing rule was violated rather than merely unlucky.
    Outside {
        /// The fused sample's seq.
        seq: u64,
        /// The frame it claimed.
        cam_seq: i64,
    },
    /// A sample labelled stale named a frame that is not before its sweep,
    /// or is older than the declared window allows: the stale label was
    /// written on a pair the window did not produce.
    OutsideWindow {
        /// The fused sample's seq.
        seq: u64,
        /// The frame it claimed.
        cam_seq: i64,
    },
    /// With the detector on, a sample named a frame `camdet` never produced
    /// a `CAM_DET` batch for: its camera half would be detections that do
    /// not exist.
    NoDetections {
        /// The fused sample's seq.
        seq: u64,
        /// The frame it claimed.
        cam_seq: i64,
    },
    /// Nothing was produced, so nothing was compared. Not a pass.
    NotChecked {
        /// Samples the stage produced.
        produced: usize,
    },
}

impl PairCheck {
    /// Whether the join held over a non-empty population.
    pub fn passed(&self) -> bool {
        matches!(self, PairCheck::Ok { .. })
    }

    /// The end-of-run line.
    pub fn line(&self) -> String {
        match self {
            PairCheck::Ok {
                checked,
                stale: 0,
                ..
            } => format!(
                "fused cam_seq = a frame the camera really admitted, and inside the sweep, on all {checked} samples"
            ),
            PairCheck::Ok {
                checked,
                stale,
                window_ms,
            } => format!(
                "fused cam_seq = a frame the camera really admitted, and inside the sweep, on all {checked} completed samples; before the sweep and at most {window_ms} ms before its trigger, the declared stale window, on all {stale} degraded ones"
            ),
            PairCheck::Unknown { seq, cam_seq } => format!(
                "MISMATCH fused sample {seq} named camera frame {cam_seq}, which the camera driver never admitted"
            ),
            PairCheck::Outside { seq, cam_seq } => format!(
                "MISMATCH fused sample {seq} named camera frame {cam_seq}, whose instant is OUTSIDE that sweep's range"
            ),
            PairCheck::OutsideWindow { seq, cam_seq } => format!(
                "MISMATCH stale fused sample {seq} named camera frame {cam_seq}, which is not before its sweep and inside the declared stale window back from its trigger"
            ),
            PairCheck::NoDetections { seq, cam_seq } => format!(
                "MISMATCH fused sample {seq} named camera frame {cam_seq}, for which camdet produced no detections"
            ),
            PairCheck::NotChecked { produced } => format!(
                "fused cam_seq NOT CHECKED: {produced} samples produced, none carried a camera frame"
            ),
        }
    }
}

/// The frozen camera detector: the `camdet` stage, what it cost, and what it
/// found. `None` unless `--detector on` reached a fusion to feed.
pub struct CamDetRun {
    /// What the stage consumed, produced, allocated and found.
    pub report: CamDetReport,
    /// The sha256 of the model file that was loaded -- the pinned one, or the
    /// run would have stopped before the clock started.
    pub model_sha256: String,
    /// What loading and optimising the model cost, before the clock started.
    pub load_ms: f64,
    /// The camera's period as this run replayed it -- the drive's median
    /// frame interval divided by `--rate` -- which is the stage's budget per
    /// frame. `None` unpaced (`--rate inf`), where there is no period to keep
    /// up with.
    pub period_ns: Option<i64>,
    /// camera driver -> `camdet`: the frame was read where the driver put it.
    pub in_check: VeloStorageCheck,
    /// `runs/<name>/cam_det.arrows`: every batch the stage produced.
    pub file: PathBuf,
    /// Its queue and delay, as `--cap`, `--policy` and `--consumer-delay-ms`
    /// set them.
    pub queue: CameraQueue,
}

/// The third link: the `detect` stage and the consumer of what it produced.
/// `None` when `--detect off`, or when there was no reduced cloud to detect
/// in.
pub struct DetectRun {
    /// What `detect` consumed, produced and allocated.
    pub report: DetectReport,
    /// The far end: `obj-sink`.
    pub sink: ObjReport,
    /// `reduce` -> `detect`: the reduced cloud was shared, not copied.
    pub in_check: VeloStorageCheck,
    /// `detect` -> `obj-sink`: the detection buffer was shared in turn.
    pub out_check: VeloStorageCheck,
    /// `detect` -> `obj-sink`: each detection batch names the cloud it was
    /// actually built from, seq included.
    pub parent_check: ParentCheck,
    /// The last two links, when `--track on` ran them.
    pub track: Option<TrackRun>,
}

/// The derived half of a lidar run: the `reduce` stage and the consumer of
/// what it produced. `None` when `--reduce off`, or when the drive has no
/// lidar to reduce.
pub struct ReduceRun {
    /// The grid edge this run used, recorded because the ratio below is
    /// meaningless without it.
    pub voxel_size: VoxelSize,
    /// What `reduce` consumed, produced and allocated.
    pub report: ReduceReport,
    /// The far end of the chain: `det-cloud`, the same consumer code that
    /// reads the raw sweep.
    pub cloud: CloudReport,
    /// velodyne driver -> `reduce`: the input was shared, not copied.
    pub in_check: VeloStorageCheck,
    /// `reduce` -> `det-cloud`: the derived buffer was shared in turn.
    pub out_check: VeloStorageCheck,
    /// `reduce` -> `det-cloud`: each derived cloud names the sweep it was
    /// actually built from, seq included.
    pub parent_check: ParentCheck,
    /// The third link hanging off the same clouds; `None` under
    /// `--detect off`.
    pub detect: Option<DetectRun>,
}

/// One consumer edge's counters, as Admission and its queue saw them.
///
/// These used to be four fields on [`RunReport`] named after `proc` and
/// `rerun`, read back with `push_blocked_ns(0)` / `(1)` at the call sites.
/// With a third edge that is a silent hazard: inserting an edge anywhere but
/// the end reassigns another edge's timings, and nothing would say so.
pub struct EdgeReport {
    pub name: &'static str,
    /// Samples Admission **routed onto this edge** -- the denominator of
    /// `delivered + dropped == admitted` (H.3). Not the global admitted
    /// count, which with two producers counts the other stream's samples too.
    pub admitted: u64,
    /// `BoundedQueue::dropped()`: every non-`Accepted` push.
    pub queue_dropped: u64,
    /// Duration of every push on this edge (ns).
    pub blocked_ns: Vec<i64>,
}

/// What the lidar half of a run produced; `None` when it did not run.
pub struct VeloRun {
    /// Sweeps the drive holds (`VeloDriver::len`).
    pub n_sweeps: usize,
    /// Frame slots the replay accounted for: every sweep, plus every frame
    /// whose sweep is absent in the source (`VeloDriver::frame_slots`).
    /// `totals.admitted + totals.missing` must equal it.
    pub frames: usize,
    /// The frames with no sweep in the source, ascending: `[177, 178, 179,
    /// 180]` on drive 0009, empty on a drive with a sweep per frame.
    pub absent_in_source: Vec<u64>,
    /// Those frames as runs, with the real sweeps either side of each.
    pub gaps: Vec<Gap>,
    pub totals: RunTotals,
    /// Bytes the velodyne replay thread requested for its admitted sweeps,
    /// summed. The head of the byte chain the next step has to shrink.
    pub driver_bytes: u64,
    pub cloud: CloudReport,
    pub storage_check: VeloStorageCheck,
    /// The derived chain hanging off the same sweeps; `None` under
    /// `--reduce off`.
    pub reduce: Option<ReduceRun>,
}

/// Everything a full run produced, for the end-of-run prints and the summary.
pub struct RunReport {
    pub run_id: String,
    pub dir: PathBuf,
    /// Frame slots on the camera stream (`Cam0Driver::frame_slots`): every
    /// PNG, and every frame absent in the camera's source.
    pub n_frames: usize,
    /// PNGs on disk (`Cam0Driver::len`): `n_frames` less the absent ones.
    pub cam_on_disk: usize,
    /// The frames, ascending, whose PNG is absent in the source; empty when
    /// the camera has a PNG for every frame.
    pub cam_absent_in_source: Vec<u64>,
    /// The camera queue `--cap`, `--policy` and `--consumer-delay-ms` set
    /// ([`camera_queues`]): `cam0->camdet` with the detector, `cam0->proc`
    /// without it.
    pub camera: CameraQueue,
    pub reuse_output: bool,
    pub rerun_mode: &'static str,
    pub totals: RunTotals,
    /// `Admission::admitted()`.
    pub admitted: u64,
    /// Samples admitted per stream, keyed by `StreamId.0`. The driver
    /// invariant is per producer, so it cannot use `admitted`.
    pub admitted_by_stream: BTreeMap<u8, u64>,
    /// `None` when the detector ran, so `proc` did not.
    pub proc: Option<ProcReport>,
    pub rerun: Option<RerunReport>,
    /// The frozen camera detector; `None` when it did not run.
    pub camdet: Option<CamDetRun>,
    /// The lidar half of the run; `None` when it did not run.
    pub velo: Option<VeloRun>,
    /// One entry per consumer edge, in declaration order.
    pub edges: Vec<EdgeReport>,
    pub sink_blocked_ns: Vec<i64>,
    pub evidence_lost: u64,
    pub recorder_degraded: bool,
    pub rows: Vec<Evidence>,
    pub storage_check: StorageCheck,
    pub untracked_bytes: u64,
    /// The run's Rerun recording (image sink and/or `--dashboard`), flushed
    /// once by `finish` after every producer has stopped. Handed back by the
    /// viewer thread, the only code that logs to it.
    pub stream: Option<RecordingStream>,
    /// The viewer thread's own count; `None` when there was no recording.
    /// Its edge's counters are with the others in `edges`.
    pub viewer: Option<ViewerReport>,
    /// Where `stream` went, for the last lines of the run.
    pub recording: Recording,
}

/// How `--rerun grpc` reached its viewer. Reported at the end of the run
/// because the two look identical from the terminal and mean different
/// things on the desktop: a window opens for a run that started one, and
/// nothing visibly happens for a run that joined one left open by an earlier
/// run, which is the state the README's "First run" leaves a reader in.
pub enum Viewer {
    /// Nothing was listening on `port`, so `rerun` was started from PATH.
    /// `pid` is that process, which is not always the window: the `rerun` a
    /// `pip install rerun-sdk` puts on PATH is a launcher that starts Python,
    /// which starts the viewer, and stopping the launcher leaves the window
    /// open (checked on Windows with rerun-sdk 0.38.1).
    Started { port: u16, pid: u32 },
    /// Something was already listening on `port` and the run streamed into it.
    Joined { port: u16 },
    /// `--rerun-host`: the run streamed to the viewer listening at
    /// `host:port`, outside this machine or container, and started nothing.
    Remote { host: String, port: u16 },
}

/// Where a run's recording went.
pub enum Recording {
    /// `--rerun grpc`: streamed to the viewer the run joined or started.
    Viewer(Viewer),
    /// An `.rrd` file under the run directory: `cam0.rrd` under `--rerun
    /// rrd`, or `dashboard.rrd` when the dashboard is on and the images have
    /// no sink.
    File(PathBuf),
    /// No recording at all (`--rerun null|off --dashboard off`).
    Nothing,
}

impl RunReport {
    /// One consumer edge's counters by name; `None` for an edge this run had
    /// no queue for (`cam0->rerun` under `--rerun off`, `velo->cloud` without
    /// lidar).
    pub fn edge(&self, name: &str) -> Option<&EdgeReport> {
        self.edges.iter().find(|e| e.name == name)
    }

    fn edge_admitted(&self, name: &str) -> u64 {
        self.edge(name).map_or(0, |e| e.admitted)
    }

    fn edge_dropped(&self, name: &str) -> u64 {
        self.edge(name).map_or(0, |e| e.queue_dropped)
    }

    /// Samples admitted on one stream. A stream with no producer in this run
    /// is absent from the map and reads as 0, which is the same statement.
    pub fn admitted_of(&self, stream: StreamId) -> u64 {
        self.admitted_by_stream.get(&stream.0).copied().unwrap_or(0)
    }

    /// Every sweep of the drive against the fusion: how many reached `track`
    /// and where the rest were lost. `None` when the fusion did not run.
    pub fn before_track(&self) -> Option<BeforeTrack> {
        let v = self.velo.as_ref()?;
        let r = v.reduce.as_ref()?;
        let d = r.detect.as_ref()?;
        let t = d.track.as_ref()?;
        let absent = v.absent_in_source.len() as u64;
        Some(BeforeTrack {
            sweeps: v.n_sweeps as u64,
            reached: t.report.delivered,
            // The driver's `Missing` rows count the absent frames too; they
            // are their own term, never the driver's skips.
            skipped: v.totals.missing.saturating_sub(absent),
            evicted_velo_reduce: self.edge_dropped("velo->reduce"),
            evicted_det_detect: self.edge_dropped("det->detect"),
            evicted_obj_track: self.edge_dropped("obj->track"),
            stage_errors: r.report.errors + d.report.errors,
            absent_in_source: absent,
        })
    }

    /// The lidar half of the end-of-run lines, in the camera's shape so the
    /// two streams can be read against each other.
    ///
    /// Printed only when the lidar ran. A camera-only drive prints not one
    /// word about a stream it does not have, because a block of `n/a`s reads
    /// as a stream that failed rather than one that was never there.
    fn print_velo(&self, v: &VeloRun) {
        // A drive with a sweep for every frame prints what it always has; a
        // gap in the source adds the frame slots and names the absent ones,
        // which `missing` counts.
        if v.absent_in_source.is_empty() {
            println!(
                "velo admitted={} missing={} n_sweeps={}",
                v.totals.admitted, v.totals.missing, v.n_sweeps
            );
        } else {
            println!(
                "velo admitted={} missing={} n_sweeps={} frames={} absent_in_source={} ({}: no sweep in the source)",
                v.totals.admitted,
                v.totals.missing,
                v.n_sweeps,
                v.frames,
                v.absent_in_source.len(),
                frames_phrase(&v.absent_in_source)
            );
        }
        println!(
            "edge=velo->cloud delivered={} dropped={} admitted={} storage_mismatch={}",
            v.cloud.delivered,
            self.edge_dropped("velo->cloud"),
            self.edge_admitted("velo->cloud"),
            v.cloud.storage_mismatch
        );
        // Spelled exactly as M11 spelled it: the integration tests grep this
        // line, and "2 stages" still names what it checks (velodyne driver ->
        // `cloud`) even now that a third stage reads the same buffer. The
        // `reduce` stage's own reading of it is a separate line with a
        // separate population, because folding the two would let one stage's
        // silence pass for the other's proof.
        match &v.storage_check {
            VeloStorageCheck::Ok { compared } => {
                println!("velo storage_id equal at 2 stages: OK ({compared} sweeps)");
            }
            VeloStorageCheck::Mismatch {
                seq,
                driver,
                consumer,
            } => println!(
                "velo storage_id equal at 2 stages: MISMATCH (velo-driver={driver:#x} cloud={:#x}) seq={seq}",
                consumer.unwrap_or(0)
            ),
            // Not "OK". A check with an empty population is not a check, and
            // this is exactly how the camera's version would have "passed" on
            // the lidar path had the two been folded into one bool.
            VeloStorageCheck::NotChecked { admitted } => println!(
                "velo storage_id equal at 2 stages: NOT CHECKED ({admitted} sweeps admitted, none compared)"
            ),
        }
        if v.cloud.count_mismatch > 0 {
            println!(
                "velo point_count disagreed with the payload on {} sweeps",
                v.cloud.count_mismatch
            );
        }
        match (
            v.cloud.points_min,
            v.cloud.points_mean(),
            v.cloud.points_max,
        ) {
            (Some(lo), Some(mean), Some(hi)) => {
                println!("velo points/sweep min={lo} mean={mean} max={hi}");
            }
            _ => println!("velo points/sweep = n/a (no sweep reached the consumer)"),
        }
        // Where the head was facing forward inside its own rotation, read
        // back out of the PAYLOAD and checked against the envelope's range.
        // On drive_0005 this is ~51.6 ms into a ~103.3 ms sweep -- which is
        // the offset a fusion stage has to reconcile against a camera
        // instant, and the reason the trigger could not be folded into
        // `Tov::Range`.
        match v.cloud.trigger_offset_mean_ns() {
            Some(ns) => println!(
                "velo trigger offset mean = {:.3} ms into the sweep (outside the range on {} sweeps, absent on {})",
                ns as f64 * 1e-6,
                v.cloud.trigger_outside_range,
                v.cloud.trigger_missing
            ),
            None => println!("velo trigger offset = n/a (no sweep carried one)"),
        }
        match v.cloud.extent {
            Some(e) => println!(
                "velo extent x=[{:.2},{:.2}] y=[{:.2},{:.2}] z=[{:.2},{:.2}] m",
                e.min[0], e.max[0], e.min[1], e.max[1], e.min[2], e.max[2]
            ),
            None => println!("velo extent = n/a (no points)"),
        }
        // The pair the next step exists to move, and the reason both numbers
        // are printed rather than one. CARRIED is the size of the transfer;
        // ALLOCATED is what the stage had to request to receive it. Today
        // they are ~1.95 MB against 0, which is the zero-copy hand-off. A
        // stage that genuinely transforms has to make the CARRIED number fall
        // down the chain -- that, and not the allocation, is "the final,
        // smaller result".
        println!(
            "velo payload bytes/sweep = {} (carried across velo->cloud)",
            show_u64(v.cloud.payload_bytes_mean())
        );
        println!(
            "velo driver bytes/sweep = {} (allocated)",
            v.driver_bytes / v.totals.admitted.max(1)
        );
        println!(
            "velo cloud bytes/sweep = {} (allocated)",
            v.cloud.bytes_total / v.cloud.delivered.max(1)
        );
        self.print_cloud_viz(entity::LIDAR_SWEEP, &v.cloud);
        if let Some(r) = &v.reduce {
            self.print_reduce(r);
        }
    }

    /// What drawing one cloud for the viewer cost, per sweep.
    ///
    /// Printed rather than absorbed, and printed even though it is outside
    /// every measured window, because the alternative is a recording that
    /// quietly got four times heavier with no line anywhere saying why.
    ///
    /// The cost is real and unavoidable: `Points3D` wants positions as
    /// `[f32; 3]` while the payload is interleaved `xyzr`, so the stride is
    /// wrong and the cloud has to be deinterleaved into a new buffer -- 12 B
    /// per point; the colour is one for the whole cloud. rerun 0.38.1 offers no
    /// zero-copy path to take instead. The raw sweep's picture carries one
    /// return in three (`CloudCfg::stride`); the stage read them all, and
    /// every number the run measures is over the buffer, not the picture.
    fn print_cloud_viz(&self, entity: &str, c: &CloudReport) {
        // Nothing drawn at all without `--dashboard`, and a line reading 0 B
        // would suggest the drawing was free rather than absent.
        if c.viz_ns_total == 0 && c.viz_bytes_total == 0 {
            return;
        }
        let n = c.delivered.max(1);
        println!(
            "{entity} drawn: {} B and {:.2} ms per sweep to build and queue ({} B, {:.2} s total), \
             allocated on the stage's own slot AFTER proc_end and excluded from bytes_alloc",
            c.viz_bytes_total / n,
            c.viz_ns_total as f64 * 1e-6 / n as f64,
            c.viz_bytes_total,
            c.viz_ns_total as f64 * 1e-9,
        );
        if c.viz_log_errors > 0 {
            println!(
                "{entity}: {} sweeps could not be logged to the viewer (the pipeline is unaffected)",
                c.viz_log_errors
            );
        }
    }

    /// The derived chain: the project's first stage-to-stage Arrow hand-off,
    /// and the byte chain that is the point of it.
    ///
    /// The chain is printed as three links with an explicit label on each,
    /// because the two numbers that get confused here are CARRIED (what the
    /// edge transfers) and ALLOCATED (what the stage had to request). The
    /// claim is that the first one FALLS down the chain while the second stays
    /// at zero except in the one stage that genuinely transforms, where it is
    /// supposed to rise.
    fn print_reduce(&self, r: &ReduceRun) {
        let rep = &r.report;
        println!(
            "reduce voxel = {} m ({})",
            r.voxel_size.metres(),
            voxel_size_note(r.voxel_size)
        );
        println!(
            "reduce delivered={} produced={} errors={} storage_mismatch={}",
            rep.delivered, rep.produced, rep.errors, rep.storage_mismatch
        );
        println!(
            "edge=det->cloud delivered={} dropped={} admitted={} storage_mismatch={}",
            r.cloud.delivered,
            self.edge_dropped("det->cloud"),
            self.edge_admitted("det->cloud"),
            r.cloud.storage_mismatch
        );
        println!(
            "reduce points/sweep in={} out={} ({})",
            show_u64((rep.delivered > 0).then(|| rep.in_points_total / rep.delivered)),
            show_u64((rep.produced > 0).then(|| rep.out_points_total / rep.produced)),
            rep.point_shrink()
                .map_or_else(|| "n/a".to_string(), |x| format!("{x:.2}x fewer points"))
        );
        // THE TABLE. Three links, one line each, carried bytes per sample.
        println!(
            "byte chain 1 velo-driver -> reduce  carried = {} B/sweep (edge velo->reduce)",
            show_u64(rep.in_payload_bytes_mean())
        );
        println!(
            "byte chain 2 reduce -> det-cloud    carried = {} B/sample (edge det->cloud)",
            show_u64(rep.out_payload_bytes_mean())
        );
        // The ratio, and — attached to it rather than three lines below it —
        // the share of the input the grid threw away. A merge and a filter
        // print the same headline otherwise: `--voxel-size-m 1e-9` addresses
        // +-1 mm, discards every point of a KITTI sweep and reported
        // `11883.34x smaller`. The suffix is empty when nothing was
        // discarded, which is every run on a real drive measured so far.
        println!(
            "byte chain shrink = {}{}",
            rep.payload_shrink()
                .map_or_else(|| "n/a".to_string(), |x| format!("{x:.2}x smaller")),
            match rep.discarded_fraction() {
                Some(f) if f > 0.0 => format!(
                    " (but {:.2}% of the input points were DISCARDED, not merged \
                     — see `reduce points discarded` below)",
                    f * 100.0
                ),
                _ => String::new(),
            }
        );
        // The two halves of the zero-copy claim, which are NOT the same claim
        // and have been confused in this project before. Reading the input
        // costs nothing; producing the output costs one buffer, and must.
        println!(
            "reduce bytes/sweep = {} reading the sweep (borrowed) + {} building the result (allocated)",
            rep.read_bytes_total / rep.delivered.max(1),
            rep.build_bytes_total / rep.produced.max(1)
        );
        println!(
            "det-cloud bytes/sample = {} (allocated)",
            r.cloud.bytes_total / r.cloud.delivered.max(1)
        );
        println!(
            "{}",
            r.in_check.line("reduce read the driver's buffer", "sweeps")
        );
        println!(
            "{}",
            r.out_check
                .line("det-cloud read reduce's buffer", "derived clouds")
        );
        // Provenance, read back at the far end rather than asserted here,
        // and in two halves: which STREAM a result came from, and which
        // SWEEP. Only the first of those was ever checked.
        println!(
            "det-cloud parent = lidar on {} of {} samples",
            r.cloud.delivered - r.cloud.parent_mismatch,
            r.cloud.delivered
        );
        println!("{}", r.parent_check.line("det-cloud", "sweep", "reduce"));
        // The honesty caveat that belongs beside every shrink ratio: a voxel
        // holding one point has a "centroid" that averages nothing.
        println!(
            "reduce voxel occupancy: {} of {} output rows hold exactly one point ({}), max {}",
            rep.singleton_voxels_total,
            rep.out_points_total,
            rep.singleton_fraction()
                .map_or_else(|| "n/a".to_string(), |f| format!("{:.1}%", f * 100.0)),
            rep.max_occupancy
        );
        println!(
            "reduce points discarded: non_finite={} out_of_range={}",
            rep.non_finite_total, rep.out_of_range_total
        );
        // The caveat on the 0 above, printed as a number. "Allocated 0
        // reading the sweep" means per sweep; this is what it excludes, once.
        println!(
            "reduce scratch = {} B, grown {} times, all outside the measured window (excluded from the 0 above)",
            rep.scratch_bytes, rep.scratch_growths
        );
        self.print_cloud_viz(entity::LIDAR_VOXELS, &r.cloud);
        if let Some(d) = &r.detect {
            self.print_detect(d);
        }
    }

    /// The third link, and the one the chain was built for: the point where
    /// the payload stops being "the same thing, decimated".
    ///
    /// The lines are laid out so the claim and its caveat cannot be separated.
    /// `detect detections/sweep` is immediately followed by what those
    /// detections ARE -- a count of connected non-ground structures, roughly
    /// three fifths of them fragments of something bigger -- because the
    /// number alone reads as a count of objects and it is not one.
    fn print_detect(&self, d: &DetectRun) {
        let rep = &d.report;
        println!("detect params: {}", detect_params_note(self.voxel_size()));
        println!(
            "detect delivered={} produced={} errors={} storage_mismatch={} voxel_size_mismatch={}",
            rep.delivered, rep.produced, rep.errors, rep.storage_mismatch, rep.voxel_size_mismatch
        );
        println!(
            "edge=obj->sink delivered={} dropped={} admitted={} storage_mismatch={}",
            d.sink.delivered,
            self.edge_dropped("obj->sink"),
            self.edge_admitted("obj->sink"),
            d.sink.storage_mismatch
        );
        println!(
            "detect voxels/sweep in={} detections/sweep out={} ({})",
            show_u64((rep.delivered > 0).then(|| rep.in_voxels_total / rep.delivered)),
            show_u64(rep.detections_mean()),
            rep.voxel_shrink()
                .map_or_else(|| "n/a".to_string(), |x| format!("{x:.1}x fewer rows"))
        );
        // THE CAVEAT, on the line after the count and not three lines below
        // it. A detection is a connected non-ground structure; most of them
        // are pieces of something larger, and a few are two things fused.
        println!(
            "detect what those are: connected non-ground structures within {} m, NOT objects -- {} of {} are smaller than any KITTI object class ({}), {} span more than one",
            RANGE_LIMIT_M,
            rep.fragment_detections_total,
            rep.out_detections_total,
            rep.fragment_fraction()
                .map_or_else(|| "n/a".to_string(), |f| format!("{:.1}%", f * 100.0)),
            rep.merged_detections_total,
        );
        println!(
            "byte chain 3 reduce -> detect       carried = {} B/sample (edge det->detect)",
            show_u64(rep.in_payload_bytes_mean())
        );
        println!(
            "byte chain 4 detect -> obj-sink     carried = {} B/sample (edge obj->sink)",
            show_u64(rep.out_payload_bytes_mean())
        );
        println!(
            "byte chain shrink at detect = {}",
            rep.payload_shrink()
                .map_or_else(|| "n/a".to_string(), |x| format!("{x:.2}x smaller"))
        );
        // The whole chain, recomputed at the FAR end from the columns the
        // detection batch carries -- not from anything this process
        // remembered. `reduce` and `detect` both state their own ratio; this
        // is the only number in the run that nothing upstream could have got
        // wrong without the batch itself being wrong.
        println!(
            "chain end to end = {} raw returns per detection, read out of the batch at obj-sink",
            d.sink
                .chain_shrink()
                .map_or_else(|| "n/a".to_string(), |x| format!("{x:.0}"))
        );
        println!(
            "detect bytes/sweep = {} reading the cloud (borrowed) + {} building the detections (allocated)",
            rep.read_bytes_total / rep.delivered.max(1),
            rep.build_bytes_total / rep.produced.max(1)
        );
        println!(
            "obj-sink bytes/sample = {} (allocated)",
            d.sink.bytes_total / d.sink.delivered.max(1)
        );
        // The same two figures read at the FAR end, over the population that
        // actually arrived. They agree with `detect`'s own only while nothing
        // was evicted on `obj->sink`, which is exactly why both are printed:
        // one stage's view of what it produced is not evidence about what was
        // delivered.
        println!(
            "obj-sink detections/sample min={} mean={} max={} carried={} B/sample",
            show_u64(d.sink.detections_min.map(u64::from)),
            show_u64(d.sink.detections_mean()),
            show_u64(d.sink.detections_max.map(u64::from)),
            show_u64(d.sink.payload_bytes_mean())
        );
        println!(
            "{}",
            d.in_check.line("detect read reduce's buffer", "clouds")
        );
        println!(
            "{}",
            d.out_check.line("obj-sink read detect's buffer", "batches")
        );
        println!(
            "obj-sink parent = lidar_det on {} of {} samples",
            d.sink.delivered - d.sink.parent_mismatch,
            d.sink.delivered
        );
        println!("{}", d.parent_check.line("obj-sink", "cloud", "detect"));
        // The ground plane, and the assumption it rests on, stated with the
        // measurement that says whether the assumption held on this drive.
        println!(
            "detect ground: {} of {} in-range voxels removed ({}), plane fitted on {} sweeps, FELL BACK to a fixed height on {}",
            rep.ground_voxels_total,
            rep.in_range_voxels_total,
            rep.ground_fraction()
                .map_or_else(|| "n/a".to_string(), |f| format!("{:.1}%", f * 100.0)),
            rep.ground_fitted,
            rep.ground_fallback
        );
        println!(
            "detect ground tilt: mean {} max {} (a SINGLE plane is a flat-world assumption; it is defensible at these angles and collapses on a real hill)",
            rep.tilt_deg_mean()
                .map_or_else(|| "n/a".to_string(), |t| format!("{t:.2} deg")),
            format_args!("{:.2} deg", rep.tilt_mdeg_max as f64 / 1000.0),
        );
        println!(
            "detect clusters={} gated to detections={} at >= {} voxels",
            rep.clusters_total, rep.out_detections_total, MIN_CLUSTER_VOXELS
        );
        println!(
            "detect voxels discarded: non_finite={} out_of_range={} key_collisions={}",
            rep.non_finite_total, rep.out_of_range_total, rep.key_collisions_total
        );
        // The sanity check, with the control beside it, and the label that
        // keeps it honest: this is the UNCOMPENSATED number, because nothing
        // in this pipeline replays the vehicle's own velocity.
        match (
            rep.persistence.fraction(),
            rep.persistence_control_turned.fraction(),
        ) {
            (Some(real), Some(control)) => println!(
                "detect persistence (UNCOMPENSATED, no ego motion -- oxts is not replayed) = {:.1}% of {} detections over {} CONSECUTIVE sweep pairs matched the next sweep within {} m | CONTROL turned 90 deg = {:.1}% | signal/noise {}",
                real * 100.0,
                rep.persistence.total,
                rep.persistence_pairs,
                PERSISTENCE_GATE_M,
                control * 100.0,
                if control > 0.0 {
                    format!("{:.0}x", real / control)
                } else {
                    "n/a (the control matched nothing)".to_string()
                }
            ),
            // Not a silent 0. A run that evicted enough on `velo->reduce`
            // leaves no adjacent pair at all, and the honest answer is that
            // the check did not run rather than that nothing persisted.
            _ => println!(
                "detect persistence = n/a (no two CONSECUTIVE sweeps reached the stage; {} clouds arrived but none of them were adjacent)",
                rep.delivered
            ),
        }
        println!(
            "detect check cost = {} B and {:.3} ms per sweep, measured AFTER proc_end on the stage's own slot and excluded from bytes_alloc",
            rep.check_bytes_total / rep.produced.max(1),
            rep.check_ns_total as f64 * 1e-6 / rep.produced.max(1) as f64
        );
        println!(
            "detect scratch = {} B, grown {} times, all outside the measured window (excluded from the 0 above)",
            rep.scratch_bytes, rep.scratch_growths
        );
        if d.sink.viz_ns_total != 0 || d.sink.viz_bytes_total != 0 {
            println!(
                "{} drawn: {} B and {:.2} ms per sweep, allocated on the stage's own \
                 slot AFTER proc_end and excluded from bytes_alloc",
                entity::LIDAR_DETECTIONS,
                d.sink.viz_bytes_total / d.sink.delivered.max(1),
                d.sink.viz_ns_total as f64 * 1e-6 / d.sink.delivered.max(1) as f64,
            );
        }
        if d.sink.viz_log_errors > 0 {
            println!(
                "{}: {} batches could not be logged to the viewer (the pipeline is unaffected)",
                entity::LIDAR_DETECTIONS,
                d.sink.viz_log_errors
            );
        }
        if let Some(tk) = &d.track {
            self.print_track(tk);
        }
    }

    /// The fusion and the answer: the last two links, the pairing, and the
    /// thing the whole pipeline exists to produce.
    fn print_track(&self, tk: &TrackRun) {
        let t = &tk.report;
        let st = &tk.state;
        println!(
            "track: {} detection batches -> {} fused samples, {} EXPIRED (no camera frame), {} errors",
            t.delivered, t.produced, t.expired, t.errors
        );
        println!(
            "byte chain 5 detect -> track        carried = {} B/sample (edge obj->track)",
            show_u64(t.in_payload_bytes_mean())
        );
        println!(
            "byte chain 6 track -> state         carried = {} B/sample (edge track->state)",
            show_u64(t.out_payload_bytes_mean())
        );
        println!(
            "byte chain 7 state -> state-sink    carried = {} B/sample (edge state->sink): {} records of {} B, one per track, and the rest is provenance",
            show_u64(st.out_payload_bytes_mean()),
            tk.sink
                .objects_mean()
                .map_or_else(|| "n/a".to_string(), |n| format!("{n:.1}")),
            OBJECT_BYTES
        );
        // The honest caveat on link 5, stated where the number is rather than
        // in a doc comment nobody reads next to it.
        // Printed as a GROWTH when it is one. `0.56x smaller` is a true
        // sentence and an unreadable one, and this is the link where the
        // number is most likely to be quoted.
        println!(
            "byte chain shrink at track = {} -- the ONLY link in the chain that adds information rather than removing it: a track carries a velocity, an age, a freshness flag, an image rectangle and the camera's class that no detection had ({} lanes against a detection's {}), and a schema trimmed to improve this ratio would be the tuning this project forbids",
            t.payload_shrink().map_or_else(
                || "n/a".to_string(),
                |x| if x >= 1.0 {
                    format!("{x:.2}x smaller")
                } else {
                    format!("{:.2}x LARGER", 1.0 / x)
                }
            ),
            TRACK_LANES,
            DETECTION_LANES
        );
        // The answer is every track now, one smaller record each, so this
        // link barely shrinks: the chain compresses at `reduce` and `detect`
        // and after that every link adds information. It used to read 46x
        // here because the answer was one object out of ~183.
        println!(
            "byte chain shrink at state = {} -- every track is in the answer, {} lanes a record against a track's {}; the chain compresses at reduce and detect, and the answer is no longer the small end of it",
            st.payload_shrink().map_or_else(
                || "n/a".to_string(),
                |x| if x >= 1.0 {
                    format!("{x:.2}x smaller")
                } else {
                    format!("{:.2}x LARGER", 1.0 / x)
                }
            ),
            OBJECT_LANES,
            TRACK_LANES
        );
        // The same figure read at the FAR end, over the population that
        // actually arrived. It agrees with `state`'s own only while nothing
        // was evicted on `state->sink`, which is exactly why both are printed.
        println!(
            "state-sink: {} answers delivered, carried = {} B/sample, {} of them from a sweep paired inside its range",
            tk.sink.delivered,
            show_u64(tk.sink.payload_bytes_mean()),
            tk.sink.paired
        );
        println!(
            "chain end to end = {} raw returns per ANSWER ({} per object in it), read out of the batch at state-sink",
            tk.sink
                .chain_shrink()
                .map_or_else(|| "n/a".to_string(), |x| format!("{x:.0}")),
            tk.sink
                .chain_shrink_per_object()
                .map_or_else(|| "n/a".to_string(), |x| format!("{x:.0}"))
        );
        println!(
            "track detections/sample={} -> tracks/sample={} (emitted at >= {} observations, because a velocity is a difference between two); live {} per sweep",
            show_u64((t.delivered > 0).then(|| t.in_detections_total / t.delivered)),
            show_u64(t.tracks_mean()),
            MIN_OBSERVATIONS,
            show_u64((t.delivered > 0).then(|| t.live_total / t.delivered))
        );
        println!(
            "track {}",
            gate_note(t.dt_s_last, self.voxel_size(), RANGE_LIMIT_M)
        );
        // The two numbers that have to be read together: a wider gate buys
        // association and pays in ambiguity.
        println!(
            "track association = {} of detections continued a track | AMBIGUOUS (more than one track inside the gate) = {} | contested from the tracks' side = {}",
            t.association_rate()
                .map_or_else(|| "n/a".to_string(), |f| format!("{:.1}%", f * 100.0)),
            t.ambiguous_fraction()
                .map_or_else(|| "n/a".to_string(), |f| format!("{:.1}%", f * 100.0)),
            t.contested_total
        );
        // The ambiguity above is not a surprise and this is the arithmetic
        // that says so BEFORE the measurement does. Detections are spread over
        // a disc of radius `range_limit`; a gate of radius `g` covers
        // `(g/R)^2` of it, so the expected number of OTHER tracks inside one
        // detection's gate by density alone is `n * (g/R)^2`. When that
        // exceeds 1, ambiguity is guaranteed by the scene rather than caused
        // by the tracker, and no gate that satisfies the physics can avoid it.
        let per_sweep = if t.delivered > 0 {
            t.live_total as f64 / t.delivered as f64
        } else {
            0.0
        };
        let expected = per_sweep * (t.gate_m_last / f64::from(RANGE_LIMIT_M)).powi(2);
        println!(
            "track ambiguity is PREDICTED by the density, not caused by the tracker: {per_sweep:.0} live tracks over a {} m disc, a {:.3} m gate covers {:.2}% of it, so {expected:.2} tracks fall inside one detection's gate by chance alone. A narrower gate would break the physics; the ways out are ego compensation (which would remove the {:.0}% of the gate that is un-subtracted yaw) and fewer fragmented detections, not a smaller number",
            RANGE_LIMIT_M,
            t.gate_m_last,
            (t.gate_m_last / f64::from(RANGE_LIMIT_M)).powi(2) * 100.0,
            if t.gate_m_last > 0.0 {
                YAW_RATE_MAX_RPS * t.dt_s_last * f64::from(RANGE_LIMIT_M) / t.gate_m_last * 100.0
            } else {
                0.0
            }
        );
        // The age is sensor time, first seen to last seen on the sweeps'
        // triggers -- the velocity's own baseline, so the noise figure is one
        // division and needs no nominal period. The observation count is
        // printed beside it because it is what the emission rule counts, and
        // because the gap between the two is the coasting a count hid.
        println!(
            "track age: mean {} max {} (sensor time, first seen to last seen; velocity quantisation noise 0.20 m / age = ~{} m/s at the mean) | observations mean {} max {} | {} of emitted tracks coasting, not seen this sweep | born {} died {} coasted {} (at most {} sweep) | ids issued {}",
            t.age_s_mean()
                .map_or_else(|| "n/a".to_string(), |a| format!("{a:.2} s")),
            if t.out_tracks_total > 0 {
                format!("{:.2} s", t.age_s_max)
            } else {
                "n/a".to_string()
            },
            t.age_s_mean()
                .filter(|a| *a > 0.0)
                .map_or_else(|| "n/a".to_string(), |a| format!("{:.2}", 0.20 / a)),
            t.observations_mean()
                .map_or_else(|| "n/a".to_string(), |a| format!("{a:.1}")),
            t.observations_max,
            t.unseen_fraction()
                .map_or_else(|| "n/a".to_string(), |f| format!("{:.1}%", f * 100.0)),
            t.born_total,
            t.died_total,
            t.coasted_total,
            MAX_MISSES,
            t.ids_issued
        );
        // The number that attributes a collapsed tracker to the input. A gap
        // in the source is a reset too -- sweep 181 is not 176 + 1 -- and a
        // true one: a tracker that coasted across 414 ms of nothing would be
        // answering from data it never had. So the line names the gaps.
        let source_gaps = self.velo.as_ref().map_or(0, |v| {
            v.gaps
                .iter()
                .filter(|g| g.before.is_some() && g.after.is_some())
                .count()
        });
        println!(
            "track RESETS = {} sweeps were not `previous + 1`, so every track was discarded and the ids started again -- an upstream eviction{}, not the algorithm",
            t.resets,
            if source_gaps > 0 {
                format!(" or a gap in the source ({source_gaps} in this drive's lidar)")
            } else {
                String::new()
            }
        );
        // ---- the pairing -------------------------------------------------
        // Every sweep of the drive, not only the ones `track` was handed: the
        // sets it made, and then the sweeps lost in front of it, which its
        // own counters cannot see.
        println!(
            "fusion sets: COMPLETED {} (camera instant inside the sweep) | DEGRADED {} (a declared stale frame) | EXPIRED {} = dropped {} + late {} + absent {}{}{} | {}",
            t.pair_ok,
            t.pair_stale,
            t.expired,
            t.pair_dropped,
            t.pair_late,
            t.pair_absent,
            if t.pair_absent_in_source.is_empty() {
                String::new()
            } else {
                format!(
                    " + camera frame absent in source {}",
                    t.pair_absent_in_source.len()
                )
            },
            if t.errors > 0 {
                format!(" | ERRORS {}", t.errors)
            } else {
                String::new()
            },
            self.before_track()
                .map_or_else(|| "sweeps n/a".to_string(), |b| b.line())
        );
        // The instants a gap in either source left without a partner: they
        // made no set, and the run says which rather than leaving them to be
        // inferred from a count.
        for line in tk.gap_pairs.lines() {
            println!("{line}");
        }
        println!(
            "pair age = {} mean (min {} max {}) -- the timestamps say the camera fires 10.503 ms after the sweep's trigger, sd 0.071 ms, so this is the check that the pairing found the RIGHT frame rather than merely a frame",
            t.pair_age_ns_mean()
                .map_or_else(|| "n/a".to_string(), |ns| format!("{:.3} ms", ns as f64 / 1e6)),
            t.pair_age_ns_min
                .map_or_else(|| "n/a".to_string(), |ns| format!("{:.3} ms", ns as f64 / 1e6)),
            t.pair_age_ns_max
                .map_or_else(|| "n/a".to_string(), |ns| format!("{:.3} ms", ns as f64 / 1e6))
        );
        // The line above covers completed sets only, so a run whose every
        // answer was stale printed `n/a` there and nowhere said how stale.
        if t.pair_stale > 0 {
            let ms = |ns: Option<i64>| {
                ns.map_or_else(
                    || "n/a".to_string(),
                    |ns| format!("{:.3} ms", ns as f64 / 1e6),
                )
            };
            println!(
                "pair age DEGRADED = {} mean (min {} max {}) over {} stale sets -- the camera half of every stale answer was taken this long BEFORE its sweep's trigger, the same camera-instant-minus-trigger as above",
                ms(t.stale_age_ns_mean()),
                ms(t.stale_age_ns_min),
                ms(t.stale_age_ns_max),
                t.pair_stale
            );
        }
        println!(
            "pair wait = {:.3} ms per sweep, INSIDE proc_end and therefore inside measurement_age_ns | camera references seen {} ({} unreadable)",
            t.wait_ns_total as f64 / 1e6 / t.delivered.max(1) as f64,
            t.cam_delivered,
            t.cam_bad_format
        );
        println!(
            "tracks in the camera frame = {} of {} ({}), calibration {}",
            t.in_frame_total,
            t.out_tracks_total,
            t.in_frame_fraction()
                .map_or_else(|| "n/a".to_string(), |f| format!("{:.1}%", f * 100.0)),
            if tk.calib.is_some() {
                "loaded"
            } else {
                "NOT LOADED -- no image rectangles"
            }
        );
        // ---- the fusion --------------------------------------------------
        self.print_fusion(tk);
        // ---- the answer --------------------------------------------------
        println!(
            "ANSWER: something in the vehicle's path on {} of {} sweeps ({}), corridor half-width {} m = half the recording vehicle ({} m)",
            st.with_object,
            st.delivered,
            st.with_object_fraction()
                .map_or_else(|| "n/a".to_string(), |f| format!("{:.1}%", f * 100.0)),
            CORRIDOR_HALF_WIDTH_M,
            VEHICLE_WIDTH_M
        );
        println!(
            "ANSWER distance: mean {} min {} max {} m (lidar origin to the object's NEAR FACE, not the bumper) | closing speed mean {} m/s, UNCOMPENSATED and correctly so: the question is how fast the gap shrinks, so a parked car ahead closes at the ego speed",
            st.distance_m_mean()
                .map_or_else(|| "n/a".to_string(), |d| format!("{d:.2}")),
            st.distance_m_min
                .map_or_else(|| "n/a".to_string(), |d| format!("{d:.2}")),
            st.distance_m_max
                .map_or_else(|| "n/a".to_string(), |d| format!("{d:.2}")),
            st.closing_mps_mean()
                .map_or_else(|| "n/a".to_string(), |d| format!("{d:.2}"))
        );
        println!(
            "ANSWER objects: every track, {} per sweep -- {} in the camera frame, the rest carried with in_frame = 0 rather than dropped; {} in the vehicle's path (fresh) on average, the nearest flagged on {} answers of {}",
            tk.sink
                .objects_mean()
                .map_or_else(|| "n/a".to_string(), |n| format!("{n:.1}")),
            tk.sink
                .in_frame_mean()
                .map_or_else(|| "n/a".to_string(), |n| format!("{n:.1}")),
            if st.delivered > 0 {
                format!("{:.1}", st.candidates_total as f64 / st.delivered as f64)
            } else {
                "n/a".to_string()
            },
            tk.sink.flagged_total,
            tk.sink.delivered
        );
        match (&st.most_urgent, st.ttc_s_min) {
            (Some((sweep, a, cam)), Some(ttc)) => println!(
                "ANSWER most urgent: sweep {sweep}, {ttc:.2} s to contact -- {}",
                answer_words(a, *cam)
            ),
            _ => println!("ANSWER most urgent: n/a (nothing was ever closing)"),
        }
        if let Some(line) = &tk.sink.last_line {
            println!("ANSWER last: {line}");
        }
        println!(
            "track bytes/sample = {} reading the detections and waiting for the camera (borrowed) + {} building the tracks (allocated); state = {} + {}; state-sink = {} (allocated)",
            t.read_bytes_total / t.delivered.max(1),
            t.build_bytes_total / t.produced.max(1),
            st.read_bytes_total / st.delivered.max(1),
            st.build_bytes_total / st.produced.max(1),
            tk.sink.bytes_total / tk.sink.delivered.max(1)
        );
        println!(
            "{}",
            tk.in_check.line("track read detect's buffer", "batches")
        );
        println!(
            "{}",
            tk.out_check.line("state read track's buffer", "samples")
        );
        println!(
            "{}",
            tk.state_check
                .line("state-sink read state's buffer", "answers")
        );
        println!("{}", tk.parent_check.line("state", "detections", "track"));
        println!(
            "{}",
            tk.state_parent_check.line("state-sink", "tracks", "state")
        );
        // The camera half of the provenance, which `Sample::parent` cannot
        // hold and which nothing would read if this line did not exist.
        println!("{}", tk.pair_check.line());
        if let Some(c) = &tk.cam_check {
            println!(
                "{}",
                c.line("track read camdet's buffer", "detection batches")
            );
        }
        println!(
            "track scratch = {} B, grown {} times, all outside the measured window (excluded from the read figure above)",
            t.scratch_bytes, t.scratch_growths
        );
        // What the pictures cost, on the terms every other picture in this run
        // is costed on: after `proc_end`, on the drawing stage's own slot, and
        // stated rather than absorbed. Three stages draw now, so three lines.
        self.print_viz(
            "lidar/tracks + pairing/*",
            t.viz_bytes_total,
            t.viz_ns_total,
            t.produced,
            t.viz_log_errors,
        );
        self.print_viz(
            "camera/tracks + camera/answer + lidar/answer",
            st.viz_bytes_total,
            st.viz_ns_total,
            st.delivered,
            st.viz_log_errors,
        );
        self.print_viz(
            "answer/* + answer/line",
            tk.sink.viz_bytes_total,
            tk.sink.viz_ns_total,
            tk.sink.delivered,
            tk.sink.viz_log_errors,
        );
    }

    /// What the association did: the counts, the IoU, the classes, and the
    /// shape table that asks whether the camera separates real objects from
    /// fragments. Completed and degraded sets are printed apart, because the
    /// degraded ones are the stale-camera experiment's result and a mean over
    /// both would hide it.
    fn print_fusion(&self, tk: &TrackRun) {
        let t = &tk.report;
        if !tk.detector {
            println!(
                "fusion: no camera detector (--detector off): the camera half is a frame reference with no detections, so every track is unfused"
            );
            return;
        }
        println!("fusion rule: {FUSION_RULE}");
        let pct =
            |x: Option<f64>| x.map_or_else(|| "n/a".to_string(), |f| format!("{:.1}%", f * 100.0));
        let num =
            |x: Option<f64>, d: usize| x.map_or_else(|| "n/a".to_string(), |v| format!("{v:.d$}"));
        let tally = |what: &str, f: &FuseTally| {
            println!(
                "fusion {what}: {} sets, {} detections, {} candidate tracks (fresh, in frame) -> FUSED {} ({} per set; {} of candidates, {} of detections), camera-only {}, contested {}, crowded {} | fused IoU p10 {} p50 {} p90 {}",
                f.samples,
                f.detections,
                f.candidates,
                f.fused,
                num(f.fused_per_sample(), 2),
                pct(f.fused_fraction_of_candidates()),
                pct(f.fused_fraction_of_detections()),
                f.camera_only,
                f.contested,
                f.crowded,
                num(f.iou_percentile(10.0), 3),
                num(f.iou_percentile(50.0), 3),
                num(f.iou_percentile(90.0), 3),
            );
            if f.fused > 0 {
                println!(
                    "fusion {what} classes: {}",
                    f.class_counts()
                        .iter()
                        .map(|(k, n)| format!("{k} {n}"))
                        .collect::<Vec<_>>()
                        .join(", ")
                );
            }
        };
        tally("completed", &t.fuse_completed);
        if t.fuse_degraded.samples > 0 {
            tally("DEGRADED (stale camera)", &t.fuse_degraded);
        }
        // The shape table: is what the camera confirmed object-shaped, and
        // what it left fragment-shaped?
        let row = |name: &str, r: &ShapeRow| {
            format!(
                "{name} {} (fragment {}, object {}, merged {}; age {} s; extent {} m)",
                r.tracks,
                pct(r.fragment_fraction()),
                pct((r.tracks > 0).then(|| r.object as f64 / r.tracks as f64)),
                pct((r.tracks > 0).then(|| r.merged as f64 / r.tracks as f64)),
                num(r.age_s_mean(), 2),
                r.extent_m_mean().map_or_else(
                    || "n/a".to_string(),
                    |e| format!("{:.2} x {:.2} x {:.2}", e[0], e[1], e[2])
                ),
            )
        };
        let sh = &t.shape;
        println!(
            "fusion shape, completed sets: {} | {} | {} | {} | {}",
            row("FUSED", &sh.fused),
            row("inside a fused object", &sh.inside_fused),
            row("in frame alone", &sh.in_frame_alone),
            row("coasted in frame", &sh.coasted_in_frame),
            row("out of frame", &sh.out_of_frame),
        );
        let by_class = sh.by_class();
        if !by_class.is_empty() {
            println!(
                "fusion shape by class, completed sets: {}",
                by_class
                    .iter()
                    .take(6)
                    .map(|(k, r)| row(k, r))
                    .collect::<Vec<_>>()
                    .join(" | ")
            );
        }
        let k = &tk.sink;
        println!(
            "fusion at the far end: records fused {}, lidar-only {}, unfused {}; camera-only detections {} | fused_count disagreeing with the records on {} answers; in the fusion's own batches on {} sets",
            k.fused_total,
            k.lidar_only_total,
            k.unfused_total,
            k.camera_only_total,
            k.fused_mismatch,
            t.fuse_column_mismatch,
        );
        println!(
            "fusion scratch = {} B, grown {} times, outside both byte windows; fused sets: {} ({} sets, Arrow IPC stream)",
            t.fuse_scratch_bytes,
            t.fuse_scratch_growths,
            tk.fused_file.display(),
            t.batches.len()
        );
    }

    /// One `--dashboard` drawing cost, in the shape [`Self::print_cloud_viz`]
    /// prints the clouds' in.
    ///
    /// Nothing is printed when nothing was drawn: a line reading 0 B would say
    /// the drawing was free rather than absent, and every stage that draws
    /// here draws only under `--dashboard`.
    fn print_viz(&self, what: &str, bytes: u64, ns: i64, n: u64, errors: u64) {
        if ns == 0 && bytes == 0 {
            return;
        }
        let n = n.max(1);
        println!(
            "{what} drawn: {} B and {:.3} ms per sample ({} B, {:.2} s total), allocated on the \
             stage's own slot AFTER proc_end and excluded from bytes_alloc",
            bytes / n,
            ns as f64 * 1e-6 / n as f64,
            bytes,
            ns as f64 * 1e-9
        );
        if errors > 0 {
            println!(
                "{what}: {errors} samples could not be logged to the viewer (the pipeline is unaffected)"
            );
        }
    }

    /// The frozen camera detector: what it ran, what it cost against the
    /// camera's period, what it found, and what it carried.
    ///
    /// The cost lines come before the findings on purpose. A detector that
    /// keeps up only at its median is the one input in this run that can go
    /// stale on its own, and a reader should know how often it did before
    /// reading what it saw.
    fn print_camdet(&self, c: &CamDetRun) {
        let r = &c.report;
        println!(
            "edge=cam0->camdet delivered={} dropped={} admitted={} storage_mismatch={}",
            r.delivered,
            self.edge_dropped("cam0->camdet"),
            self.edge_admitted("cam0->camdet"),
            r.storage_mismatch
        );
        println!(
            "camdet model {} ({}) sha256 {} loaded in {:.0} ms before the clock started; input 1x3x{}x{} letterboxed, score > {}, class-agnostic NMS at IoU > {}",
            MODEL_NAME, MODEL_FILE, c.model_sha256, c.load_ms, INPUT_H, INPUT_W, SCORE_THRESHOLD, NMS_IOU
        );
        let admitted = self.edge_admitted("cam0->camdet");
        let frac = r.delivered as f64 / admitted.max(1) as f64;
        let rate = match c.period_ns {
            Some(p) if p > 0 => {
                let hz = 1e9 / p as f64;
                format!(
                    " = {:.2} Hz of the camera's {hz:.2} Hz as replayed",
                    frac * hz
                )
            }
            _ => " (unpaced: no camera rate to keep up with)".to_string(),
        };
        // The drops by the reason their own rows give, read back out of the
        // evidence rather than assumed from the policy.
        let mut reasons: BTreeMap<&str, u64> = BTreeMap::new();
        for row in self
            .rows
            .iter()
            .filter(|x| x.edge == "cam0->camdet" && x.outcome != Outcome::Delivered)
        {
            *reasons.entry(row.reason).or_insert(0) += 1;
        }
        println!(
            "camdet ran on {} of {} frames ({:.1}%){rate}; dropped at admission (queue cap {}): {}; {} errors",
            r.delivered,
            admitted,
            frac * 100.0,
            c.queue.cap,
            if reasons.is_empty() {
                "none".to_string()
            } else {
                reasons
                    .iter()
                    .map(|(k, n)| format!("{n} {k}"))
                    .collect::<Vec<_>>()
                    .join(", ")
            },
            r.errors
        );
        let ms = |v: &[i64], p: f64| -> String {
            let mut v = v.to_vec();
            percentile(&mut v, p)
                .map_or_else(|| "-".to_string(), |ns| format!("{:.1}", ns as f64 / 1e6))
        };
        println!(
            "camdet ms/frame: service p50 {} p90 {} p99 {} max {} (letterbox+tensor p50 {}, network+decode+NMS p50 {}); {} -- UNCERTIFIED, the host was not checked quiet",
            ms(&r.service_ns, 50.0),
            ms(&r.service_ns, 90.0),
            ms(&r.service_ns, 99.0),
            ms(&r.service_ns, 100.0),
            ms(&r.preprocess_ns, 50.0),
            ms(&r.infer_ns, 50.0),
            match c.period_ns {
                Some(p) => format!(
                    "{} of {} frames over the {:.2} ms period",
                    r.over_period(p),
                    r.delivered,
                    p as f64 / 1e6
                ),
                None => "no period to be over, unpaced".to_string(),
            }
        );
        // Printed only when asked for, so a plain run's lines are unchanged,
        // and printed at all because the figures above would otherwise read
        // as the detector's own cost.
        if c.queue.delay_ms > 0 {
            println!(
                "camdet delay = {} ms a frame (--consumer-delay-ms), slept inside the measured window: every service figure above includes it",
                c.queue.delay_ms
            );
        }
        println!(
            "camdet detections: {} in {} frames ({} per frame): {}",
            r.detections_total,
            r.produced,
            r.detections_mean()
                .map_or_else(|| "n/a".to_string(), |m| format!("{m:.1}")),
            r.classes()
                .iter()
                .map(|(k, n)| format!("{k} {n}"))
                .collect::<Vec<_>>()
                .join(", ")
        );
        // CARRIED against ALLOCATED, as everywhere else in this run, and the
        // one place the second is supposed to be the larger: the frame is read
        // in place, and then the network needs its own copies of it.
        println!(
            "camdet bytes/frame = {} looking at the frame (allocated: {} of it resizing the frame and filling the tensor, the rest the network's own buffers) + {} building the batch (allocated); carried in = {} (the frame, read in place), carried out = {} (edge cam_det->track)",
            r.frame_bytes_total / r.delivered.max(1),
            r.preprocess_bytes_total / r.delivered.max(1),
            r.build_bytes_total / r.produced.max(1),
            r.in_payload_bytes_total / r.delivered.max(1),
            show_u64(r.out_payload_bytes_mean())
        );
        println!(
            "{}",
            c.in_check
                .line("camdet read the camera driver's buffer", "frames")
        );
        println!(
            "camdet batches: {} ({} batches, Arrow IPC stream, one per frame the detector ran on)",
            c.file.display(),
            r.batches.len()
        );
        self.print_viz(
            entity::CAMERA_DET,
            r.viz_bytes_total,
            r.viz_ns_total,
            r.produced,
            r.viz_log_errors,
        );
    }

    /// The voxel edge this run's lidar chain used, or the default when it did
    /// not run. Only ever read on a path where the chain DID run.
    fn voxel_size(&self) -> VoxelSize {
        self.velo
            .as_ref()
            .and_then(|v| v.reduce.as_ref())
            .map_or_else(VoxelSize::default, |r| r.voxel_size)
    }

    /// The end-of-run lines (exact spellings per spec G.6; the verifier greps them).
    pub fn print(&self) {
        let wall_s = self.totals.wall_ns as f64 / 1e9;
        // `camera_queue=` names the edge the three before it are about: the
        // detector's with it on, `proc`'s without.
        println!(
            "run {} cap={} policy={} delay_ms={} camera_queue={} reuse_output={} rerun={} n_frames={}",
            self.run_id,
            self.camera.cap,
            self.camera.policy.name(),
            self.camera.delay_ms,
            self.camera.edge,
            self.reuse_output,
            self.rerun_mode,
            self.n_frames
        );
        println!(
            "driver admitted={} missing={} wall_s={wall_s:.3}",
            self.totals.admitted, self.totals.missing
        );
        // A drive whose camera has a PNG for every frame prints what it
        // always has; a gap in its source is named, as the lidar's is.
        if !self.cam_absent_in_source.is_empty() {
            println!(
                "cam0 on_disk={} n_frames={} absent_in_source={} ({}: no frame in the source)",
                self.cam_on_disk,
                self.n_frames,
                self.cam_absent_in_source.len(),
                frames_phrase(&self.cam_absent_in_source)
            );
        }
        println!("admitted = {}", self.admitted);
        if let Some(p) = &self.proc {
            println!(
                "edge=cam0->proc delivered={} dropped={} admitted={} storage_mismatch={}",
                p.delivered,
                self.edge_dropped("cam0->proc"),
                self.edge_admitted("cam0->proc"),
                p.storage_mismatch
            );
        }
        if let Some(r) = &self.rerun {
            println!(
                "edge=cam0->rerun delivered={} dropped={} admitted={} storage_mismatch={}",
                r.delivered,
                self.edge_dropped("cam0->rerun"),
                self.edge_admitted("cam0->rerun"),
                r.storage_mismatch
            );
        }
        // The viewer's own edge, in drawings: a viewer that did not take them
        // loses them here, and the reasons are its rows'.
        if let Some(v) = &self.viewer {
            let mut why: BTreeMap<&str, u64> = BTreeMap::new();
            for r in &self.rows {
                if r.edge == VIEWER_EDGE && r.outcome != Outcome::Delivered {
                    *why.entry(r.reason).or_default() += 1;
                }
            }
            let why: Vec<String> = why.iter().map(|(r, n)| format!("{r} {n}")).collect();
            println!(
                "edge={VIEWER_EDGE} delivered={} dropped={} admitted={} (drawings, cap {VIEWER_CAP}, drop-oldest{}{})",
                v.delivered,
                self.edge_dropped(VIEWER_EDGE),
                self.edge_admitted(VIEWER_EDGE),
                if why.is_empty() { "" } else { "; " },
                why.join(", ")
            );
            if v.log_errors > 0 {
                println!("viewer: the Rerun SDK refused {} log calls", v.log_errors);
            }
        }
        // The stages the check compared: the driver, `proc` without the
        // detector, `rerun` unless it is off.
        let (has_proc, has_rerun) = (self.proc.is_some(), self.rerun.is_some());
        let stages = match (has_proc, has_rerun) {
            (true, true) => "3 stages",
            (true, false) => "2 stages (rerun off)",
            (false, true) => "2 stages (driver, rerun)",
            (false, false) => "1 stage (driver only)",
        };
        match &self.storage_check {
            StorageCheck::Ok { stages: 1 } => {
                println!("storage_id equal at {stages}: nothing to compare");
            }
            StorageCheck::Ok { .. } => println!("storage_id equal at {stages}: OK"),
            StorageCheck::Mismatch {
                seq,
                driver,
                proc,
                rerun,
            } => {
                let mut ids = format!("driver={driver:#x}");
                if has_proc {
                    ids.push_str(&format!(" proc={:#x}", proc.unwrap_or(0)));
                }
                if has_rerun {
                    ids.push_str(&format!(" rerun={:#x}", rerun.unwrap_or(0)));
                }
                println!("storage_id equal at {stages}: MISMATCH ({ids}) seq={seq}");
            }
        }
        if let Some(c) = &self.camdet {
            self.print_camdet(c);
        }
        if let Some(v) = &self.velo {
            self.print_velo(v);
        }
        if let Some(p) = &self.proc {
            println!("proc bytes/frame = {}", p.bytes_total / p.delivered.max(1));
        }
        match &self.rerun {
            Some(r) => println!("rerun bytes/frame = {}", r.bytes_total / r.delivered.max(1)),
            None => println!("rerun bytes/frame = n/a (off)"),
        }
        println!("bytes/frame = bytes requested from the allocator on that thread (allocations + realloc growth), not resident memory");
        for edge in [
            "cam0->proc",
            "cam0->rerun",
            "cam0->camdet",
            "velo->cloud",
            "velo->reduce",
            "det->cloud",
            "det->detect",
            "obj->sink",
            VIEWER_EDGE,
        ] {
            if let Some(e) = self.edge(edge) {
                println!("push_blocked_ns p99 = {} edge={edge}", p99(&e.blocked_ns));
                println!("push_blocked_ns edge={edge} {}", dist(&e.blocked_ns));
            }
        }
        println!(
            "push_blocked_ns p99 = {} edge=evidence->rec",
            p99(&self.sink_blocked_ns)
        );
        println!(
            "push_blocked_ns edge=evidence->rec {}",
            dist(&self.sink_blocked_ns)
        );
        println!("untracked bytes total = {}", self.untracked_bytes);
        println!(
            "evidence rows = {} lost = {} recorder_degraded = {}",
            self.rows.len(),
            self.evidence_lost,
            self.recorder_degraded
        );
        println!("evidence.csv: {}", self.dir.join("evidence.csv").display());
    }
}

/// `p99` in ns, or `-` when nothing was pushed on that edge. A dash reads as
/// "not measured"; `0` would read as "measured, and it was instant".
fn p99(v: &[i64]) -> String {
    let mut v = v.to_vec();
    show(percentile(&mut v, 99.0))
}

fn show(x: Option<i64>) -> String {
    x.map_or_else(|| "-".to_string(), |x| x.to_string())
}

/// As [`show`], for counts that cannot be negative. A dash is "nothing was
/// measured"; `0` is a measurement this pipeline really does record.
fn show_u64(x: Option<u64>) -> String {
    x.map_or_else(|| "-".to_string(), |x| x.to_string())
}

/// `p50=… p90=… p99=… max=… n=…` (ns), the whole distribution behind the p99 line.
fn dist(v: &[i64]) -> String {
    let mut v = v.to_vec();
    v.sort_unstable();
    let q = |p: f64| show(percentile_sorted(&v, p));
    format!(
        "p50={} p90={} p99={} max={} n={}",
        q(50.0),
        q(90.0),
        q(99.0),
        q(100.0),
        v.len()
    )
}

pub fn main(a: RunArgs) -> Result<(), BoxError> {
    let report = execute(&a)?;
    let (_summary, ok) = finish(&report)?;
    if !ok {
        std::process::exit(2);
    }
    Ok(())
}

/// After a run: summary → `summary.json` → the G.6 lines → the H.3
/// `INVARIANT … -> OK|FAIL` lines. Returns the summary and whether
/// every invariant holds (the caller decides the exit code, after everything
/// has been written).
pub fn finish(report: &RunReport) -> Result<(RunSummary, bool), RunError> {
    let (summary, invariants) = summarize(report, rss_bytes());
    let summary_json = serde_json::to_string_pretty(&summary)?;

    // Console output first, and unconditionally: if the flush below fails, the
    // numbers are still worth seeing to diagnose why. Printing is not an
    // artifact, so it is not behind the gate.
    report.print();
    let mut ok = true;
    for inv in &invariants {
        println!("{} -> {}", inv.text, if inv.ok { "OK" } else { "FAIL" });
        ok &= inv.ok;
    }

    // M8: one flush for both producers on the recording (the rerun consumer
    // and the dashboard). The recording carries no copy of `summary.json`:
    // the file on disk is the summary, and a document panel in the viewer
    // was one more thing to close before the picture could be seen.
    //
    // This flush is a gate, and everything that vouches for the run is written
    // *after* it. It used to sit in the middle: `summary.json` was written
    // first, so a flush failure -- which with `--rerun grpc` means the viewer
    // was never there or went away mid-run -- left a summary behind with no
    // `_COMPLETE` beside it. D26 says every completion artifact sits behind
    // one gate; the flush is part of that gate.
    if let Some(rec) = &report.stream {
        rec.flush_blocking()
            .map_err(|e| RunError::Rerun(RerunError::Flush(e.to_string())))?;
    }

    let summary_path = report.dir.join("summary.json");
    std::fs::write(&summary_path, &summary_json)?;
    println!("summary.json: {}", summary_path.display());
    // The last action of a run that produced its artifacts: a marker saying the
    // evidence in this directory is whole. A run killed or panicked partway
    // leaves no marker, so anything reading the directory can refuse it rather
    // than quietly report a better number from a truncated file.
    std::fs::write(
        report.dir.join(COMPLETE_MARKER),
        format!(
            "run_id={}\nevidence_rows={}\ninvariants_ok={}\n",
            report.run_id,
            report.rows.len(),
            ok
        ),
    )?;
    // Last, and as full paths: `runs/<name>` is relative to wherever the run
    // was started, and the reader of a scrolled-past terminal should not
    // have to know where that was.
    let here = std::env::current_dir()?;
    let full = |p: &Path| {
        if p.is_absolute() {
            p.to_path_buf()
        } else {
            here.join(p)
        }
    };
    println!("run dir: {}", full(&report.dir).display());
    match &report.recording {
        Recording::Viewer(Viewer::Started { port, pid }) => {
            println!(
                "rerun: streamed to the viewer this run opened on port {port} (pid {pid} is the \
                 `rerun` started from PATH; pip's is a launcher, and the window is a separate \
                 process under it)"
            );
        }
        Recording::Viewer(Viewer::Joined { port }) => println!(
            "rerun: streamed to the viewer that was already listening on port {port} \
             (no new window was opened for this run)"
        ),
        Recording::Viewer(Viewer::Remote { host, port }) => {
            println!("rerun: streamed to the viewer at {host}:{port} (--rerun-host)")
        }
        Recording::File(p) => println!("rerun: {}", full(p).display()),
        Recording::Nothing => println!("rerun: nothing recorded (--rerun {})", report.rerun_mode),
    }
    Ok((summary, ok))
}

/// The stream behind `--rerun grpc`: a viewer already listening on `port`
/// (`--rerun-port`, the SDK's 9876 unless moved) is joined, otherwise `rerun`
/// is started from PATH and joined once it listens (`rerun::spawn` then
/// `connect_grpc_opts`, which is what `RecordingStreamBuilder::spawn_opts`
/// does, rerun 0.38.1; the two steps are taken here so the run can say which
/// happened). The run used to `connect_grpc()` and leave starting the viewer
/// to the reader, which was the one step every first run got wrong: with
/// nothing listening the SDK buffered until its flush timeout and the run
/// failed at the end with the frames already gone.
///
/// Only the "no executable" failure is rewritten. It is the one a fresh
/// machine hits, and the SDK's own text recommends `pip3`, which is not what
/// this project's setup line says.
///
/// **A viewer this run starts is listening before this returns**, and this
/// runs before the clock starts, so the replay never begins against a viewer
/// that cannot take its data. `rerun::spawn` returns as soon as the process
/// exists; a cold start of the viewer took 15 s on the development host, and
/// while nothing listened every stage that draws blocked in `rec.log` once
/// the SDK's buffers filled, so the drive replayed past them and their queues
/// evicted: the detector ran on 64 of 154 frames and the run still exited 0
/// with every invariant OK. The stages no longer call the SDK -- the viewer
/// thread does, from its own queue ([`crate::viewer`]) -- so a viewer slow to
/// listen would now cost drawings, not samples; the wait keeps the start of
/// the run on screen. See [`wait_for_listener`] for the wait, and
/// [`VIEWER_LISTEN_BUDGET`] for how long.
fn spawn_viewer(port: u16) -> Result<(RecordingStream, Viewer), RunError> {
    use rerun::{RecordingStreamError, SpawnError};
    let opts = viewer_spawn_options(port);
    let info = rerun::spawn(&opts).map_err(|e| match e {
        SpawnError::ExecutableNotFoundInPath { .. } => {
            RunError::Rerun(RerunError::NoViewer { port })
        }
        e => RunError::Rerun(RerunError::Stream(RecordingStreamError::SpawnViewer(e))),
    })?;
    // `info.port` is the port the SDK settled on; it equals `port` unless the
    // SDK was asked to pick a free one, which this run never does.
    let viewer = match info.child_pid {
        Some(pid) => {
            let port = info.port;
            let waited = wait_for_listener(
                LOCAL_VIEWER_HOST,
                port,
                VIEWER_LISTEN_BUDGET,
                VIEWER_LISTEN_NOTE,
                |w| {
                    println!(
                        "rerun: waiting for the viewer this run started to listen on port \
                         {port} ({:.0} s so far, up to {:.0} s); the clock has not started",
                        w.as_secs_f64(),
                        VIEWER_LISTEN_BUDGET.as_secs_f64()
                    );
                },
            )
            .map_err(|waited| {
                RunError::Rerun(RerunError::NotListening {
                    port,
                    pid,
                    waited_s: waited.as_secs_f64(),
                })
            })?;
            println!(
                "rerun: the viewer this run started was listening on port {port} after {:.1} s; \
                 the clock starts after this",
                waited.as_secs_f64()
            );
            Viewer::Started { port, pid }
        }
        None => Viewer::Joined { port: info.port },
    };
    let stream = RecordingStreamBuilder::new("pipes").connect_grpc_opts(format!(
        "rerun+http://{LOCAL_VIEWER_HOST}:{}/proxy",
        info.port
    ))?;
    Ok((stream, viewer))
}

/// Where a viewer this run starts, or finds already running, listens.
const LOCAL_VIEWER_HOST: &str = "127.0.0.1";

/// The stream behind `--rerun grpc --rerun-host HOST`: the viewer listening
/// at `HOST:port` is joined and nothing is started, because a viewer on
/// another machine, or on the host of the container this runs in, cannot be
/// started from here. The run waits for it exactly as [`spawn_viewer`] waits
/// for its own, so the clock never starts against a viewer that cannot take
/// the data; the wait gives the reader time to open one.
fn join_viewer(host: &str, port: u16) -> Result<(RecordingStream, Viewer), RunError> {
    let waited = wait_for_listener(host, port, VIEWER_LISTEN_BUDGET, VIEWER_LISTEN_NOTE, |w| {
        println!(
            "rerun: waiting for a viewer to listen at {host}:{port} ({:.0} s so far, up to \
                 {:.0} s); open `rerun` there. The clock has not started",
            w.as_secs_f64(),
            VIEWER_LISTEN_BUDGET.as_secs_f64()
        );
    })
    .map_err(|waited| {
        RunError::Rerun(RerunError::NoRemoteViewer {
            host: host.to_string(),
            port,
            waited_s: waited.as_secs_f64(),
        })
    })?;
    println!(
        "rerun: joining the viewer at {host}:{port}, which answered after {:.1} s; the clock \
         starts after this",
        waited.as_secs_f64()
    );
    // An IPv6 address goes in brackets in a URL; `--rerun-host` accepts it
    // either way.
    let url_host = if host.contains(':') && !host.starts_with('[') {
        format!("[{host}]")
    } else {
        host.to_string()
    };
    let stream = RecordingStreamBuilder::new("pipes")
        .connect_grpc_opts(format!("rerun+http://{url_host}:{port}/proxy"))?;
    Ok((
        stream,
        Viewer::Remote {
            host: host.to_string(),
            port,
        },
    ))
}

/// The window a viewer this run starts opens at, in logical points: the size
/// the dashboard's layout is drawn for. Left to itself the viewer reopens at
/// whatever size its last window had -- 1600x1200 on the development host --
/// and in a window that much taller than the layout the camera picture,
/// whose share of the height is fixed, got black bands above and below it.
const VIEWER_WINDOW: &str = "1600x900";

/// How `--rerun grpc` starts a viewer when none listens on `port`: the SDK's
/// defaults, on that port, at [`VIEWER_WINDOW`]. A viewer that is already
/// listening is joined as it is, whatever its size.
///
/// `wait_for_bind` is false on purpose, and said so rather than inherited:
/// the run waits itself ([`wait_for_listener`]). The SDK's own wait (rerun
/// 0.38.1, `spawn.rs`) is 30 tries whose length depends on how fast the OS
/// refuses a connection -- about 36 s on Windows, 6 s on Linux -- after which
/// a release build logs a warning and carries on against a viewer that is
/// not listening, and a debug build panics.
fn viewer_spawn_options(port: u16) -> rerun::SpawnOptions {
    rerun::SpawnOptions {
        port,
        wait_for_bind: false,
        extra_args: vec!["--window-size".to_string(), VIEWER_WINDOW.to_string()],
        ..rerun::SpawnOptions::default()
    }
}

/// How long a run waits for a viewer it started to listen before it stops
/// with [`RerunError::NotListening`], before its clock starts. A cold start
/// of the viewer took 15 s on the development host (the `rerun` on PATH is a
/// Python launcher that starts a 214 MB executable), so the budget is four
/// times that.
const VIEWER_LISTEN_BUDGET: Duration = Duration::from_secs(60);

/// After this long without a listener the run says it is still waiting, so
/// a cold start does not look like a hang.
const VIEWER_LISTEN_NOTE: Duration = Duration::from_secs(5);

/// How long one connection attempt may take, and the pause between two.
const VIEWER_LISTEN_POLL: Duration = Duration::from_millis(100);

/// Polls `host:port` until something accepts a TCP connection, for at most
/// `budget`: `Ok(waited)` once something does, `Err(waited)` when the budget
/// runs out first. `note` is called once, the first time the wait passes
/// `note_after` with nothing listening.
///
/// A TCP accept is the readiness the SDK's own `wait_for_bind` tests, and it
/// is enough: the viewer's gRPC server takes and buffers what the run sends
/// from the moment it listens, whether or not its window is up yet.
///
/// The host is resolved on every attempt, not once: a name such as
/// `host.docker.internal` can fail to resolve until the network it names is
/// up, and a failed lookup counts as nothing listening yet.
fn wait_for_listener(
    host: &str,
    port: u16,
    budget: Duration,
    note_after: Duration,
    mut note: impl FnMut(Duration),
) -> Result<Duration, Duration> {
    use std::net::ToSocketAddrs;
    let bare = host.trim_start_matches('[').trim_end_matches(']');
    let start = now();
    let waited = || Duration::from_nanos(u64::try_from(now() - start).unwrap_or(0));
    let mut noted = false;
    loop {
        let listening = (bare, port).to_socket_addrs().is_ok_and(|mut addrs| {
            addrs.any(|a| std::net::TcpStream::connect_timeout(&a, VIEWER_LISTEN_POLL).is_ok())
        });
        if listening {
            return Ok(waited());
        }
        let w = waited();
        if w >= budget {
            return Err(w);
        }
        if !noted && w >= note_after {
            noted = true;
            note(w);
        }
        std::thread::sleep(VIEWER_LISTEN_POLL);
    }
}

/// The dashboard's reference lines, derived from this run's own arguments.
///
/// The measurement-age bound is the pre-registered `cap × frame period +
/// consumer delay` (spec §I.2), in host ms: the drive's own median sensor
/// period replayed at `--rate`, so a drive captured at a different cadence
/// gets the right line rather than a hard-coded 103.3 ms. An unpaced run has
/// no deadlines, so `due_ns` and `measurement_age_ns` are both empty and there
/// is nothing to bound.
///
/// `cam0->rerun` is always cap 1 with no artificial delay, so its bound is one
/// frame period, and `cam0->camdet` is its cap in periods plus the detector's
/// own service, which is most of one.
///
/// `reduce` and `answer` say which of the dashboard's lanes and pictures this
/// run draws at all -- `velo->reduce`, which the lidar lane watches, and the
/// answer -- so a frame whose sweep is absent in the source marks those and
/// only those.
fn dashboard_bounds(
    a: &RunArgs,
    driver: &Cam0Driver,
    cams: &CameraQueues,
    image_wh: Option<(u32, u32)>,
    reduce: bool,
    answer: bool,
) -> Bounds {
    let mut age_queue_ms = BTreeMap::new();
    if a.rate.is_finite() && a.rate > 0.0 {
        let period_ms = median_period_ns(&driver.timestamps) as f64 * 1e-6 / a.rate;
        // Queueing term only. `--consumer-delay-ms` is deliberately absent: the
        // sleep it causes sits between `proc_start` and `proc_end` of whichever
        // stage it slows, so it is already inside the service time the
        // dashboard adds per frame, and counting it here too would double it.
        let camera = cams.knobs();
        age_queue_ms.insert(camera.edge, camera.cap as f64 * period_ms);
        age_queue_ms.insert("cam0->rerun", period_ms);
    }
    // Every queue this binary can open, with the capacity it opens it at. The
    // dashboard draws a line only where rows exist, so naming an edge this run
    // never opened (`cam0->rerun` under `--rerun off`, the lidar edges on a
    // camera-only drive) costs nothing and asserts nothing.
    //
    // Listed here rather than measured from the queues because this is the one
    // place that already knows all of them, and because the numbers are the
    // point: `--cap` is the camera experiment's independent variable, on
    // whichever camera queue feeds the answer, while the lidar edges are pinned
    // at `VELO_CAP` and `DET_CAP` precisely so it cannot reach them. A viewer
    // that draws one capacity line for all five would say the opposite.
    let mut queue_cap = BTreeMap::from([
        // Fixed at 1 where it is built; not `--cap`.
        ("cam0->rerun", 1.0),
        ("velo->cloud", VELO_CAP as f64),
        ("velo->reduce", VELO_CAP as f64),
        ("det->cloud", DET_CAP as f64),
        ("det->detect", DET_CAP as f64),
        ("obj->sink", OBJ_CAP as f64),
        // The chain's last links are bounded too, at the same cap, and the
        // camera-reference edge at its own: a depth read without its cap
        // answers nothing, on these four as on the others.
        ("obj->track", CHAIN_CAP as f64),
        ("cam_det->track", CAM_REF_CAP as f64),
        ("track->state", CHAIN_CAP as f64),
        ("state->sink", CHAIN_CAP as f64),
    ]);
    // The one camera consumer's queue, `cam0->camdet` or `cam0->proc`.
    queue_cap.insert(cams.knobs().edge, cams.knobs().cap as f64);
    Bounds {
        age_queue_ms,
        queue_cap,
        image_wh: image_wh.map(|(w, h)| (w as f32, h as f32)),
        period_ms: Some(median_period_ns(&driver.timestamps) as f64 * 1e-6),
        camera_edge: Some(cams.knobs().edge),
        n_frames: Some(driver.frame_slots() as f64),
        rate: (a.rate.is_finite() && a.rate > 0.0).then_some(a.rate),
        lidar_lane: reduce,
        answer,
    }
}

/// The fixed spin window (`cli::DEFAULT_SPIN_WINDOW_MS`), as a `Duration`.
fn spin_window() -> Result<Duration, RunError> {
    Duration::try_from_secs_f64(DEFAULT_SPIN_WINDOW_MS / 1000.0)
        .map_err(|e| RunError::Args(format!("spin window: {e}")))
}

/// Rejects arguments that would otherwise produce a schedule of nonsense
/// instead of an error. `--rate 0` is the sharp one: it is finite, so it passes
/// `ClockModel::due`'s guard, and `delta / 0.0` is infinity, which saturates to
/// `i64::MAX` on the cast into the deadline. A negative rate puts every
/// deadline in the past, so every frame is skipped and the run reports all
/// `Missing` with no diagnostic. `inf` stays valid: it is the documented
/// unpaced mode.
fn validate(a: &RunArgs) -> Result<(), RunError> {
    if a.rate.is_nan() || a.rate <= 0.0 {
        return Err(RunError::Args(format!(
            "--rate must be positive (or `inf` for unpaced); got {}",
            a.rate
        )));
    }
    Ok(())
}

/// The full pipeline (spec G.6).
pub fn execute(a: &RunArgs) -> Result<RunReport, RunError> {
    validate(a)?;
    let spin = spin_window()?;
    // First, before any thread exists: the viewer this run spawns takes the
    // focus, and a Windows process in the background is power-throttled onto
    // the efficiency cores -- which halved the detector's speed and expired
    // half the sweeps of every live run. See `pipes_core::host`.
    let throttling = opt_out_of_power_throttling();
    match throttling {
        PowerThrottling::Off => println!(
            "host: power throttling off for this process, so its timings do not depend on which window has the focus"
        ),
        PowerThrottling::Refused(code) => println!(
            "host: WARNING Windows refused to turn power throttling off (os error {code}): \
             with another window in front this run's stages may run on the efficiency cores"
        ),
        PowerThrottling::NotApplicable => {}
    }
    let driver = Cam0Driver::open(&a.kitti_root, a.date(), &a.drive)?;
    let t0_sensor = driver
        .timestamps
        .first()
        .copied()
        .ok_or(RunError::NoFrames)?;

    // D16: everything slow happens before the clock starts.
    let git_sha = git_sha();
    let (width, height) = frame0_dims(&driver);
    let schema = cam0_schema();
    // The second producer, when the drive has one. `auto` asks the disk,
    // because a camera-only download is common and requiring a flag for a
    // fact that is already on disk is how a second stream ends up never
    // being exercised. `on` turns a missing stream into an error (that
    // is `VeloDriver::open`'s own `NoVelodyne`, which names the drive's
    // sibling directories); `off` ignores one that is there.
    //
    // Opened here rather than on the driver thread: `open` parses three
    // timestamp files and walks the sweep directory, which is exactly the
    // kind of work D16 keeps out of the paced window.
    let want_lidar = match a.lidar {
        LidarArg::Off => false,
        LidarArg::On => true,
        LidarArg::Auto => has_velodyne(&a.kitti_root, a.date(), &a.drive),
    };
    let velo = match want_lidar {
        true => Some(VeloDriver::open(&a.kitti_root, a.date(), &a.drive)?),
        false => None,
    };
    // The drive's frame count is the most frame slots any sensor this run
    // replays declares -- trailing blank lines included -- and each stream is
    // extended to it. KITTI numbers every sensor of a drive by one frame
    // index, so a sensor whose timestamp file stops before another's lacks
    // the frames after it: absent in its source like any other, and
    // invisible to that sensor alone. Symmetric, so a camera that stops
    // early is a gap in the camera's source, not a fusion that waited for a
    // frame and called it late. (`--lidar off` replays one sensor, which is
    // then the drive.)
    let n_frames = velo
        .as_ref()
        .map_or(0, VeloDriver::frame_slots)
        .max(driver.frame_slots());
    let driver = driver.with_frame_slots(n_frames);
    let velo: Option<Arc<VeloDriver>> = velo.map(|v| Arc::new(v.with_frame_slots(n_frames)));
    // A gap in either sensor's source is said before the clock starts, so the
    // run that follows is read knowing it: the frames are replayed as missing
    // inputs, not refused and not closed up.
    let cam_gaps = driver.gaps();
    if !cam_gaps.is_empty() {
        println!(
            "cam0 {}: {} frames over {}; no frame in the source for {}; each is replayed at its own place in the schedule as a missing input ({ABSENT_IN_SOURCE}), and a sweep of its number has no camera frame to pair with",
            a.drive,
            driver.len(),
            driver.frame_slots(),
            cam_gaps.iter().map(Gap::describe).collect::<Vec<_>>().join(" and ")
        );
    }
    if let Some(v) = &velo {
        let gaps = v.gaps();
        if !gaps.is_empty() {
            println!(
                "velo {}: {} sweeps over {} frames; no sweep in the source for {}; each is replayed at its own place in the schedule as a missing input ({ABSENT_IN_SOURCE}), and the camera frame of its number has no sweep to pair with",
                a.drive,
                v.len(),
                v.frame_slots(),
                gaps.iter().map(Gap::describe).collect::<Vec<_>>().join(" and ")
            );
        }
    }
    let velo_schema = velo.as_ref().map(|_| velo_schema());
    let n_sweeps = velo.as_ref().map(|v| v.len());
    // The derived chain exists only where there is a lidar to derive from. On
    // a camera-only drive `--reduce on` is not an error and not a warning;
    // there is simply nothing to reduce, and the run says nothing about it,
    // exactly as it says nothing about lidar.
    let want_reduce = velo.is_some() && a.reduce == ReduceArg::On;
    // Built before the clock starts and shared by every derived batch (D16),
    // exactly as `cam0_schema` and `velo_schema` are.
    let det_schema = want_reduce.then(voxel_schema);
    // And the third link, which exists only where the second one does: there
    // is nothing to detect in without a reduced cloud to detect in.
    let want_detect = want_reduce && a.detect == DetectArg::On;
    let obj_schema = want_detect.then(detect_schema);
    // And the last two, which exist only where the third one does.
    let want_track = want_detect && a.track == TrackArg::On;
    let trk_schema = want_track.then(track_schema);
    let ego_schema = want_track.then(state_schema);
    // The camera's half of the fusion: the frozen detector's batches, or
    // `proc`'s bare frame reference -- never both, because two producers on
    // one stream would collide on its seq.
    let want_detector = want_track && a.detector == DetectorArg::On;
    let cam_ref_sch = (want_track && !want_detector).then(cam_ref_schema);
    // The camera knobs go where the answer's camera half comes from.
    let cams = camera_queues(a, want_detector);
    // Read once, before the clock starts, on the same terms as the schemas.
    // A hard failure: `--track on` is a request for image rectangles, and
    // every fallback that would let the run continue is a wrong projection
    // that looks right. See `RunError::Calib`.
    let calib = match want_track {
        true => {
            let c = Calib::load(&a.kitti_root, a.date())?;
            // Cross-checked against the PNG header the drive scanner already
            // read, never assumed: KITTI's resolution varies by capture date
            // and a hardcoded 1242x375 is wrong on three of the other four.
            if let (Some(w), Some(h)) = (width, height) {
                c.check_image_size(&a.kitti_root.join(a.date()), w, h)?;
            }
            println!(
                "calib {} loaded: image {}x{}, projection P_rect_02 . R_rect_00 . (R|T), \
                 frustum tested as w > 0 in homogeneous coordinates",
                a.date(),
                c.width,
                c.height
            );
            Some(c)
        }
        false => None,
    };
    // The model, on the calibration's terms: read, checked against its pinned
    // sha256 and optimised once, before the clock starts (D16), and a hard
    // failure naming the download when it cannot be -- see `RunError::Model`.
    // Its allocations land on the detector's own counter rather than on
    // `untracked`, where they would read as the viewer's.
    let model = match want_detector {
        true => {
            set_stage_slot(SLOT_CAMDET);
            let t = now();
            let loaded = Model::load(Path::new(MODEL_FILE));
            let load_ms = (now() - t) as f64 / 1e6;
            set_stage_slot(SLOT_UNTRACKED);
            let m = loaded?;
            println!(
                "camdet model {MODEL_FILE} loaded: {MODEL_NAME}, sha256 {} (the pinned one), 1x3x{INPUT_H}x{INPUT_W}, in {load_ms:.0} ms",
                m.sha256
            );
            Some((m, load_ms))
        }
        false => None,
    };
    let cam_det_sch = model.as_ref().map(|(m, _)| cam_det_schema(&m.sha256));
    let name = a
        .name
        .clone()
        .unwrap_or_else(|| format!("run-{}", wall_now_unix_ns() / 1_000_000_000));
    let dir = PathBuf::from("runs").join(&name);
    std::fs::create_dir_all(&dir)?;
    // Clear the completion evidence of any previous run that used this name,
    // before a single artifact of this one is written. Re-running a name
    // overwrites `evidence.csv` but would otherwise leave the old `_COMPLETE`
    // and `summary.json` in place, so a run that then panicked would be
    // vouched for by its own successful predecessor - the marker asserting
    // completeness exactly where it is false. Found by running
    // `--panic-at-frame` twice into one directory. Best-effort: absent is the
    // desired state, and a failure to remove surfaces as the later write.
    let _ = std::fs::remove_file(dir.join(COMPLETE_MARKER));
    let _ = std::fs::remove_file(dir.join("summary.json"));
    let rerun_mode = match a.rerun {
        RerunArg::Off => RerunMode::Off,
        RerunArg::Null => RerunMode::Null,
        RerunArg::Rrd => RerunMode::Rrd(dir.join("cam0.rrd")),
        RerunArg::Grpc => RerunMode::Grpc,
    };
    let rerun_mode_name = rerun_mode.name();
    // One recording per run, opened before the clock starts (D16). The rerun
    // consumer draws images for it and, with `--dashboard`, the `rec` thread
    // mirrors evidence rows onto it; `--rerun null|off --dashboard` still gets
    // its own file, so the flag is useful without the image stream.
    let dashboard_on = a.dashboard.is_on();
    // `--viewer-delay-ms`, the test hook: a recording written to a file takes
    // nothing for that long once the clock starts.
    let freeze =
        (a.viewer_delay_ms > 0).then(|| Freeze::new(Duration::from_millis(a.viewer_delay_ms)));
    let save = |path: PathBuf| -> Result<(Recording, Option<RecordingStream>), RunError> {
        let stream = match &freeze {
            Some(f) => frozen_file(&path, Arc::clone(f))?,
            None => RecordingStreamBuilder::new("pipes").save(&path)?,
        };
        Ok((Recording::File(path), Some(stream)))
    };
    let (recording, stream) = match (&rerun_mode, dashboard_on) {
        (RerunMode::Grpc, _) => {
            let (stream, viewer) = match &a.rerun_host {
                Some(host) => join_viewer(host, a.rerun_port)?,
                None => spawn_viewer(a.rerun_port)?,
            };
            (Recording::Viewer(viewer), Some(stream))
        }
        (RerunMode::Rrd(path), _) => save(path.clone())?,
        (RerunMode::Null | RerunMode::Off, true) => save(dir.join("dashboard.rrd"))?,
        (RerunMode::Null | RerunMode::Off, false) => (Recording::Nothing, None),
    };
    // The viewer's own queue, `draw->viewer`, exactly when there is a
    // recording: everything below draws on a canvas of it, and the viewer
    // thread, started with the clock, is the only code that logs to `stream`
    // ([`crate::viewer`]). With `--rerun null|off --dashboard off` there is no
    // queue, no thread and no canvas, so nothing is drawn or queued.
    let viewer: Option<Arc<ViewerQueue>> = stream.as_ref().map(|_| ViewerQueue::new(VIEWER_CAP));
    // Images go only where `--rerun` asked for them.
    let image_canvas = match rerun_mode {
        RerunMode::Rrd(_) | RerunMode::Grpc => viewer.as_ref().map(|v| v.canvas("rerun")),
        RerunMode::Null | RerunMode::Off => None,
    };
    // The lidar's two clouds and everything drawn after them: only under
    // `--dashboard`, on the same recording, drawn outside the measured
    // window. Without the flag those stages hold no canvas at all.
    let canvas = |stage: &'static str| {
        viewer
            .as_ref()
            .filter(|_| dashboard_on)
            .map(|v| v.canvas(stage))
    };
    // Built here, where the caps and the spin window are already in hand, and
    // before the `rec` thread takes it: the dashboard draws these lines but
    // never derives them, so it stays a pure reader of the evidence stream.
    let dashboard = match (dashboard_on, &stream, &viewer) {
        (true, Some(s), Some(v)) => {
            // Live in a viewer, or a file opened later: the layout plays and
            // shows the last eight seconds for the first, and shows the
            // whole run paused for the second.
            let mode = match recording {
                Recording::Viewer(_) => Mode::Live,
                Recording::File(_) | Recording::Nothing => Mode::File,
            };
            Some(Dashboard::new(
                s,
                v.canvas("rec"),
                dashboard_bounds(
                    a,
                    &driver,
                    &cams,
                    width.zip(height),
                    want_reduce,
                    want_track,
                ),
                mode,
            )?)
        }
        _ => None,
    };
    let sink = Arc::new(EvidenceSink::new(4096, Duration::from_secs(1)));
    // From here on every exit path goes through `Recorder`, so the sink is
    // always closed and `rec` always joined — including the `?`s below, which
    // used to be able to skip the shutdown entirely.
    let recorder = Recorder {
        sink: Arc::clone(&sink),
        handle: Some(std::thread::Builder::new().name("rec".to_string()).spawn({
            let q = sink.queue();
            let dir = dir.clone();
            move || rec_thread(q, dir, dashboard)
        })?),
    };
    // `proc`'s queue, only without the detector: with it, nothing reads
    // `proc`'s output, so neither the queue nor the stage exists.
    let proc_q: Option<Arc<BoundedQueue<Arc<Sample>>>> = cams
        .proc()
        .map(|c| Arc::new(BoundedQueue::new(c.cap, c.policy)));
    let rerun_q: Option<Arc<BoundedQueue<Arc<Sample>>>> = match rerun_mode {
        RerunMode::Off => None,
        _ => Some(Arc::new(BoundedQueue::new(1, QueuePolicy::DropOldest))),
    };
    // The detector's own fan-out of the camera frame, in `proc`'s place and
    // beside `rerun`'s: the frame is shared, so the edge costs one `Arc`
    // clone. With the detector on this is the queue `--cap` and `--policy` set.
    let camdet_q: Option<Arc<BoundedQueue<Arc<Sample>>>> = cams
        .camdet()
        .map(|c| Arc::new(BoundedQueue::new(c.cap, c.policy)));
    // Deliberately not `--policy`/`--cap`: see [`VELO_CAP`] and
    // [`VELO_POLICY`] for why a ~1.95 MB payload gets its own two, and why the
    // camera experiment's independent variable must not reach this edge.
    let velo_q: Option<Arc<BoundedQueue<Arc<Sample>>>> = velo
        .as_ref()
        .map(|_| Arc::new(BoundedQueue::new(VELO_CAP, VELO_POLICY)));
    // The derived chain's two queues. `velo->reduce` is a SECOND fan-out of
    // the same sweep, on the same terms as `velo->cloud`: the sweep is shared,
    // so a second edge costs one more `Arc` clone and one more slot's worth of
    // retention, not a second copy of 1.95 MB.
    let reduce_q: Option<Arc<BoundedQueue<Arc<Sample>>>> =
        want_reduce.then(|| Arc::new(BoundedQueue::new(VELO_CAP, VELO_POLICY)));
    let det_q: Option<Arc<BoundedQueue<Arc<Sample>>>> =
        want_reduce.then(|| Arc::new(BoundedQueue::new(DET_CAP, DET_POLICY)));
    // `det->detect` is a SECOND fan-out of the reduced cloud, on the same
    // terms `velo->reduce` is a second fan-out of the sweep: the cloud is
    // shared, so the extra edge costs one `Arc` clone and one slot's retention
    // rather than a second copy of it.
    let detect_q: Option<Arc<BoundedQueue<Arc<Sample>>>> =
        want_detect.then(|| Arc::new(BoundedQueue::new(DET_CAP, DET_POLICY)));
    let obj_q: Option<Arc<BoundedQueue<Arc<Sample>>>> =
        want_detect.then(|| Arc::new(BoundedQueue::new(OBJ_CAP, OBJ_POLICY)));
    // The fusion's two inputs and the last two links. `obj->track` is a SECOND
    // fan-out of the detections, on the same terms `det->detect` is a second
    // fan-out of the cloud. `cam_det->track` is the new shape: an edge that
    // crosses from the CAMERA path to the lidar chain, which is what makes
    // this a fusion rather than a fourth link.
    let obj_track_q: Option<Arc<BoundedQueue<Arc<Sample>>>> =
        want_track.then(|| Arc::new(BoundedQueue::new(CHAIN_CAP, CHAIN_POLICY)));
    let cam_det_q: Option<Arc<BoundedQueue<Arc<Sample>>>> =
        want_track.then(|| Arc::new(BoundedQueue::new(CAM_REF_CAP, CHAIN_POLICY)));
    let state_q: Option<Arc<BoundedQueue<Arc<Sample>>>> =
        want_track.then(|| Arc::new(BoundedQueue::new(CHAIN_CAP, CHAIN_POLICY)));
    let ego_q: Option<Arc<BoundedQueue<Arc<Sample>>>> =
        want_track.then(|| Arc::new(BoundedQueue::new(CHAIN_CAP, CHAIN_POLICY)));

    // The clock starts; from here on only run.json and decode(0) sit in the lead window.
    let clock = ClockModel::start_now(t0_sensor, a.rate);
    let ctx = Arc::new(RunCtx {
        run_id: name.clone(),
        t0_host: clock.t0_host,
        epoch: clock.epoch,
        n_frames: driver.frame_slots(),
    });
    if let Some(f) = &freeze {
        f.start(clock.t0_host);
    }
    // The viewer thread, the only code that logs to the recording: it needs
    // the clock's origin for its rows, and nothing has drawn yet.
    let viewer_thread = match (&viewer, stream) {
        (Some(v), Some(rec)) => Some(ViewerThread {
            queue: Arc::clone(v),
            handle: Some(
                std::thread::Builder::new()
                    .name("viewer".to_string())
                    .spawn({
                        let v = Arc::clone(v);
                        let sink = Arc::clone(&sink);
                        let ctx = Arc::clone(&ctx);
                        move || viewer_thread(v, rec, sink, ctx)
                    })?,
            ),
        }),
        _ => None,
    };
    write_run_json(
        &dir.join("run.json"),
        &RunJson {
            clock: &clock,
            rate: a.rate.to_string(),
            git_sha,
            args: std::env::args().collect(),
            spin_window_ms: DEFAULT_SPIN_WINDOW_MS,
            date: a.date(),
            drive: &a.drive,
            n_frames: driver.frame_slots(),
            width,
            height,
            rerun_mode: rerun_mode_name,
            lidar: a.lidar.name(),
            n_sweeps,
            reduce: a.reduce.name(),
            voxel_size_m: want_reduce.then(|| a.voxel_size_m.metres()),
            detect: a.detect.name(),
            detect_range_limit_m: want_detect.then_some(RANGE_LIMIT_M),
            detect_min_cluster_voxels: want_detect.then_some(MIN_CLUSTER_VOXELS),
            track: a.track.name(),
            pair_wait_ms: want_track.then_some(a.pair_wait_ms).flatten(),
            pair_stale_ms: want_track.then_some(a.pair_stale_ms),
            corridor_half_width_m: want_track.then_some(CORRIDOR_HALF_WIDTH_M),
            detector: a.detector.name(),
            detector_model: want_detector.then_some(MODEL_NAME),
            detector_model_file: want_detector.then_some(MODEL_FILE),
            detector_model_sha256: model.as_ref().map(|(m, _)| m.sha256.clone()),
            detector_input: want_detector
                .then(|| format!("1x3x{INPUT_H}x{INPUT_W} bgr letterbox-top-left pad {PAD_VALUE}")),
            detector_score_threshold: want_detector.then_some(SCORE_THRESHOLD),
            detector_nms_iou: want_detector.then_some(NMS_IOU),
            fusion_rule: want_detector.then_some(FUSION_RULE),
            power_throttling: throttling.name(),
        },
    )?;
    // The clock model is in `run.json` and nowhere else: a `run/clock`
    // document used to be logged onto the recording too, and was one more
    // panel in the viewer saying what the file on disk already says.

    // Admission is built BEFORE the consumer threads, which is a change from
    // the one-producer shape and the reason the reorder is here rather than
    // where it reads most naturally. The velodyne driver runs on a thread of
    // its own and needs the `Arc<Admission>` at spawn time; the queues
    // already exist above, so the only thing that had to move was this.
    //
    // Every edge names the stream it carries. Without that filter `admit`
    // would push a ~1.95 MB sweep onto `cam0->proc`, where `proc` would read a
    // point cloud as a camera frame -- and each edge needs its own admitted
    // count anyway, because `delivered + dropped == admitted` cannot use a
    // global counter that is now the sum of two producers.
    let mut edges = Vec::new();
    if let Some(q) = &proc_q {
        edges.push(Edge {
            name: "cam0->proc",
            stream: StreamId::CAM0,
            queue: Arc::clone(q),
        });
    }
    if let Some(q) = &rerun_q {
        edges.push(Edge {
            name: "cam0->rerun",
            stream: StreamId::CAM0,
            queue: Arc::clone(q),
        });
    }
    if let Some(q) = &camdet_q {
        edges.push(Edge {
            name: "cam0->camdet",
            stream: StreamId::CAM0,
            queue: Arc::clone(q),
        });
    }
    if let Some(q) = &velo_q {
        edges.push(Edge {
            name: "velo->cloud",
            stream: StreamId::LIDAR,
            queue: Arc::clone(q),
        });
    }
    // The chain. `velo->reduce` carries the same stream as `velo->cloud` --
    // one producer, two consumers of the raw sweep -- while `det->cloud`
    // carries a stream no driver produces. That is the new thing: a stream
    // admitted by a CONSUMER thread, with its own routing, its own per-edge
    // denominator and its own rows, on exactly the same terms as a sensor's.
    if let Some(q) = &reduce_q {
        edges.push(Edge {
            name: "velo->reduce",
            stream: StreamId::LIDAR,
            queue: Arc::clone(q),
        });
    }
    if let Some(q) = &det_q {
        edges.push(Edge {
            name: "det->cloud",
            stream: StreamId::LIDAR_DET,
            queue: Arc::clone(q),
        });
    }
    // The third link. `det->detect` carries the same stream as `det->cloud` --
    // one producer, two consumers of the reduced cloud -- while `obj->sink`
    // carries a stream that is neither a sensor's nor a cloud: the first
    // payload in this project that is not the shape it was derived from.
    if let Some(q) = &detect_q {
        edges.push(Edge {
            name: "det->detect",
            stream: StreamId::LIDAR_DET,
            queue: Arc::clone(q),
        });
    }
    if let Some(q) = &obj_q {
        edges.push(Edge {
            name: "obj->sink",
            stream: StreamId::LIDAR_OBJ,
            queue: Arc::clone(q),
        });
    }
    // The fusion. `obj->track` is the same stream as `obj->sink`; the three
    // below are new streams, and `cam_det->track` is the first edge in this
    // project whose producer is on the CAMERA path and whose consumer is on
    // the lidar chain.
    if let Some(q) = &obj_track_q {
        edges.push(Edge {
            name: "obj->track",
            stream: StreamId::LIDAR_OBJ,
            queue: Arc::clone(q),
        });
    }
    if let Some(q) = &cam_det_q {
        edges.push(Edge {
            name: "cam_det->track",
            stream: StreamId::CAM_DET,
            queue: Arc::clone(q),
        });
    }
    if let Some(q) = &state_q {
        edges.push(Edge {
            name: "track->state",
            stream: StreamId::TRACKS,
            queue: Arc::clone(q),
        });
    }
    if let Some(q) = &ego_q {
        edges.push(Edge {
            name: "state->sink",
            stream: StreamId::EGO,
            queue: Arc::clone(q),
        });
    }
    let admission = Arc::new(Admission::new(edges, Arc::clone(&sink), Arc::clone(&ctx)));

    // Consumers start before the driver.
    // `proc` only without the detector: with it, `proc`'s output reaches nothing.
    let proc_handle = match (&proc_q, cams.proc()) {
        (Some(q), Some(c)) => {
            let q = Arc::clone(q);
            let sink = Arc::clone(&sink);
            let ctx = Arc::clone(&ctx);
            let cfg = ProcCfg {
                delay_ms: c.delay_ms,
                reuse_output: a.reuse_output,
                panic_at_frame: a.panic_at_frame,
                // `None` unless `--track on --detector off`, so the camera
                // path is byte for byte what it is without the fusion.
                cam_ref: cam_ref_sch.as_ref().map(|schema| CamRefOut {
                    admission: Arc::clone(&admission),
                    schema: Arc::clone(schema),
                }),
            };
            Some(
                std::thread::Builder::new()
                    .name("proc".to_string())
                    .spawn(move || proc_thread(q, sink, ctx, cfg))?,
            )
        }
        _ => None,
    };
    // The image consumer clears the detector's boxes at every frame it
    // draws, so a frame the detector skipped shows none rather than the last
    // frame's. Only where both are drawn: see `rerun_thread`.
    let clear_detections = want_detector && dashboard_on && image_canvas.is_some();
    let rerun_handle = match &rerun_q {
        Some(q) => Some(
            std::thread::Builder::new()
                .name("rerun".to_string())
                .spawn({
                    let q = Arc::clone(q);
                    let sink = Arc::clone(&sink);
                    let ctx = Arc::clone(&ctx);
                    move || rerun_thread(q, sink, ctx, image_canvas, clear_detections)
                })?,
        ),
        None => None,
    };
    // The frozen detector: the camera's half of the fusion under `--detector
    // on`. It takes the `Arc<Admission>` for the reason `reduce` does -- it is
    // a consumer that admits -- and the model, which it alone runs.
    let (model, model_load_ms) = match model {
        Some((m, ms)) => (Some(m), ms),
        None => (None, 0.0),
    };
    let camdet_handle = match (&camdet_q, model, &cam_det_sch) {
        (Some(q), Some(m), Some(schema)) => Some(
            std::thread::Builder::new()
                .name("camdet".to_string())
                .spawn({
                    let q = Arc::clone(q);
                    let admission = Arc::clone(&admission);
                    let sink = Arc::clone(&sink);
                    let ctx = Arc::clone(&ctx);
                    let cfg = CamDetCfg {
                        model: m,
                        schema: Arc::clone(schema),
                        expect: driver.len(),
                        rec: canvas("camdet"),
                        delay_ms: cams.camdet().map_or(0, |c| c.delay_ms),
                    };
                    move || camdet_thread(q, admission, sink, ctx, cfg)
                })?,
        ),
        _ => None,
    };

    // The lidar's leaf consumer. Deliberately the smallest honest one -- see
    // `cloud_thread`: its job in this step is to prove a second producer
    // reaches a second consumer through the same admission, not to perceive.
    let cloud_handle = match (&velo_q, n_sweeps) {
        (Some(q), Some(n)) => Some(
            std::thread::Builder::new()
                .name("cloud".to_string())
                .spawn({
                    let q = Arc::clone(q);
                    let sink = Arc::clone(&sink);
                    let ctx = Arc::clone(&ctx);
                    // Made out here: the `move` closure below cannot borrow
                    // the queue the canvas is made from.
                    let rec = canvas("cloud");
                    move || {
                        cloud_thread(
                            q,
                            sink,
                            ctx,
                            CloudCfg {
                                edge: "velo->cloud",
                                stage: "cloud",
                                slot: SLOT_VELO,
                                expect: n,
                                // A stream off a sensor has no parent, so
                                // there is nothing here that could be missing.
                                expect_parent: None,
                                rec,
                                entity: entity::LIDAR_SWEEP,
                                // The lidar's teal (#00A8B0), translucent,
                                // and a hairline, so 41k points read as a
                                // surface rather than a fog of spheres.
                                color: SWEEP_RGBA,
                                radius_pt: 0.8,
                                // One return in three reaches the picture; the
                                // stage and its row see all of them. See
                                // `CloudCfg::stride`.
                                stride: 3,
                            },
                        )
                    }
                })?,
        ),
        _ => None,
    };

    // The transform stage, and the consumer of what it produces. `reduce`
    // takes the `Arc<Admission>` -- it is the first consumer in this project
    // that admits -- which is why `Admission` is built above the consumers
    // rather than below them.
    let reduce_handle = match (&reduce_q, &det_schema, n_sweeps) {
        (Some(q), Some(schema), Some(n)) => Some(
            std::thread::Builder::new()
                .name("reduce".to_string())
                .spawn({
                    let q = Arc::clone(q);
                    let admission = Arc::clone(&admission);
                    let sink = Arc::clone(&sink);
                    let ctx = Arc::clone(&ctx);
                    let cfg = ReduceCfg {
                        voxel_size: a.voxel_size_m,
                        schema: Arc::clone(schema),
                        expect: n,
                    };
                    move || reduce_thread(q, admission, sink, ctx, cfg)
                })?,
        ),
        _ => None,
    };
    // The SAME consumer as `cloud`, on the derived stream. Not a near-copy:
    // the identical function, which is only possible because `reduce` emits
    // the layout it consumed. That is the Arrow rule, demonstrated.
    let obj_viz_stream = canvas("obj-sink");
    let track_viz_stream = canvas("track");
    let state_viz_stream = canvas("state");
    let sink_viz_stream = canvas("state-sink");
    let det_viz_stream = canvas("det-cloud");
    let det_cloud_handle = match (&det_q, n_sweeps) {
        (Some(q), Some(n)) => Some(
            std::thread::Builder::new()
                .name("det-cloud".to_string())
                .spawn({
                    let q = Arc::clone(q);
                    let sink = Arc::clone(&sink);
                    let ctx = Arc::clone(&ctx);
                    move || {
                        cloud_thread(
                            q,
                            sink,
                            ctx,
                            CloudCfg {
                                edge: "det->cloud",
                                stage: "det-cloud",
                                slot: SLOT_DET_CLOUD,
                                expect: n,
                                // A derived sample MUST name the sweep it came
                                // from, and this is where that is checked.
                                expect_parent: Some(StreamId::LIDAR),
                                rec: det_viz_stream,
                                entity: entity::LIDAR_VOXELS,
                                // The same teal, more opaque, and half again
                                // as large as the sweep's points: "fewer,
                                // bigger" is the thing you see, which is what
                                // the byte chain is counting. Drawn at the
                                // sweep's size the voxels read as a thinner
                                // copy of it.
                                color: VOXEL_RGBA,
                                radius_pt: 1.2,
                                // Every voxel: this is the claim the 3D view
                                // exists to show, and there are few enough.
                                stride: 1,
                            },
                        )
                    }
                })?,
        ),
        _ => None,
    };

    // The third link, and the consumer of what it produces. `detect` takes
    // the `Arc<Admission>` for the same reason `reduce` does: it is a consumer
    // thread that admits.
    let detect_handle = match (&detect_q, &obj_schema, n_sweeps) {
        (Some(q), Some(schema), Some(n)) => Some(
            std::thread::Builder::new()
                .name("detect".to_string())
                .spawn({
                    let q = Arc::clone(q);
                    let admission = Arc::clone(&admission);
                    let sink = Arc::clone(&sink);
                    let ctx = Arc::clone(&ctx);
                    let cfg = DetectCfg {
                        voxel_size: a.voxel_size_m,
                        schema: Arc::clone(schema),
                        expect: n,
                    };
                    move || detect_thread(q, admission, sink, ctx, cfg)
                })?,
        ),
        _ => None,
    };
    // NOT `cloud_thread`, and that is the point rather than an oversight.
    // Two instances of that one function read both ends of the `reduce`
    // hand-off because a reduced cloud is still a cloud. A detection is not,
    // so it needs a consumer of its own -- which is the difference between
    // decimating the data and saying something about it, made structural.
    let obj_handle = match (&obj_q, n_sweeps) {
        (Some(q), Some(n)) => Some(
            std::thread::Builder::new()
                .name("obj-sink".to_string())
                .spawn({
                    let q = Arc::clone(q);
                    let sink = Arc::clone(&sink);
                    let ctx = Arc::clone(&ctx);
                    let rec = obj_viz_stream;
                    move || obj_thread(q, sink, ctx, ObjCfg { expect: n, rec })
                })?,
        ),
        _ => None,
    };

    // The fusion, and the two stages after it. `track` takes the
    // `Arc<Admission>` for the reason `reduce` and `detect` do, and one more
    // besides: it is the first stage with TWO inputs, so it also takes the
    // camera edge itself rather than a second `pop` loop -- see
    // `track_thread`.
    let track_handle = match (&obj_track_q, &trk_schema, n_sweeps) {
        (Some(q), Some(schema), Some(n)) => Some(
            std::thread::Builder::new()
                .name("track".to_string())
                .spawn({
                    let q = Arc::clone(q);
                    let admission = Arc::clone(&admission);
                    let sink = Arc::clone(&sink);
                    let ctx = Arc::clone(&ctx);
                    let cfg = TrackCfg {
                        schema: Arc::clone(schema),
                        voxel_size: a.voxel_size_m,
                        expect: n,
                        rec: track_viz_stream,
                        calib,
                        cam_q: cam_det_q.as_ref().map(Arc::clone),
                        cam_absent: driver.absent_in_source(),
                        pair_wait_ns: a
                            .pair_wait_ms
                            .map(|ms| i64::try_from(ms).unwrap_or(i64::MAX) * 1_000_000),
                        pair_stale_ns: i64::try_from(a.pair_stale_ms).unwrap_or(i64::MAX)
                            * 1_000_000,
                    };
                    move || track_thread(q, admission, sink, ctx, cfg)
                })?,
        ),
        _ => None,
    };
    let state_handle = match (&state_q, &ego_schema, n_sweeps) {
        (Some(q), Some(schema), Some(n)) => Some(
            std::thread::Builder::new()
                .name("state".to_string())
                .spawn({
                    let q = Arc::clone(q);
                    let admission = Arc::clone(&admission);
                    let sink = Arc::clone(&sink);
                    let ctx = Arc::clone(&ctx);
                    let cfg = StateCfg {
                        schema: Arc::clone(schema),
                        expect: n,
                        rec: state_viz_stream,
                        image_wh: width.zip(height).map(|(w, h)| (w as f32, h as f32)),
                    };
                    move || state_thread(q, admission, sink, ctx, cfg)
                })?,
        ),
        _ => None,
    };
    let state_sink_handle = match (&ego_q, n_sweeps) {
        (Some(q), Some(n)) => Some(
            std::thread::Builder::new()
                .name("state-sink".to_string())
                .spawn({
                    let q = Arc::clone(q);
                    let sink = Arc::clone(&sink);
                    let ctx = Arc::clone(&ctx);
                    let rec = sink_viz_stream;
                    move || state_sink_thread(q, sink, ctx, StateSinkCfg { expect: n, rec })
                })?,
        ),
        _ => None,
    };

    // The second producer, on a thread of its own, and both halves of that
    // matter:
    //
    // * `SLOT_VELO_DRIVER` is a THREAD-local attribution, so a velodyne
    //   driver sharing the main thread would have its sweeps counted against
    //   the camera's `bytes/frame` -- silently, and in the one number the
    //   committed record is read for.
    // * Two producers that are not concurrent do not exercise the thing this
    //   exists to exercise. `arrival_seq` is a global order across streams
    //   only if two threads really are contending for one admission mutex; if
    //   the lidar were replayed after the camera, every camera sample would
    //   take a contiguous block of `arrival_seq` and the column would mean
    //   no more than it did with one driver.
    let velo_handle = match (&velo, &velo_schema) {
        (Some(v), Some(vs)) => Some(
            std::thread::Builder::new()
                .name("velo-driver".to_string())
                .spawn({
                    let v = Arc::clone(v);
                    let vs = Arc::clone(vs);
                    let admission = Arc::clone(&admission);
                    let sink = Arc::clone(&sink);
                    let ctx = Arc::clone(&ctx);
                    let clock = clock.clone();
                    move || {
                        set_stage_slot(SLOT_VELO_DRIVER);
                        let rc = ctx.row_ctx();
                        // The gaps in the source, by their first frame, so
                        // each is named once, as an event, when the replay
                        // reaches it. Built before the first admit, so it is
                        // not in any sweep's `bytes_alloc`.
                        let gap_at: BTreeMap<u64, Gap> =
                            v.gaps().into_iter().map(|g| (g.first, g)).collect();
                        // Exactly the camera's accounting, on this thread's
                        // own counter: the delta between consecutive admits
                        // is one sweep's `read` plus its Arrow batch build.
                        let mut mark = bytes_alloc(SLOT_VELO_DRIVER);
                        let mut driver_bytes = 0u64;
                        let totals = v.run(&clock, spin, &vs, &mut |ev| {
                            let sweep_bytes = bytes_alloc(SLOT_VELO_DRIVER) - mark;
                            match ev {
                                DriverEvent::Sample(s) => {
                                    let admitted = admission.admit(s);
                                    let mut row = Evidence::driver_admitted(
                                        &rc,
                                        &admitted,
                                        "velo",
                                        "velo-driver",
                                    );
                                    row.bytes_alloc = sweep_bytes;
                                    driver_bytes += sweep_bytes;
                                    sink.send(EvRow::Evidence(row), &ctx, "velo-driver");
                                }
                                DriverEvent::Missing {
                                    seq,
                                    tov,
                                    due,
                                    reason,
                                } => {
                                    // A gap in the source is named once, as
                                    // a WARN, at the moment the replay
                                    // reaches it, ahead of its frames' rows.
                                    if reason == ABSENT_IN_SOURCE {
                                        if let Some(g) = gap_at.get(&seq) {
                                            sink.send(
                                                EvRow::Event(Event::source_gap(
                                                    &rc,
                                                    now(),
                                                    "velo-driver",
                                                    admission.admitted(),
                                                    format!(
                                                        "lidar {}: no sweep in the source ({ABSENT_IN_SOURCE}); recorded as missing inputs, and the fusion has no set for them",
                                                        g.describe()
                                                    ),
                                                )),
                                                &ctx,
                                                "velo-driver",
                                            );
                                        }
                                    }
                                    sink.send(
                                        EvRow::Evidence(Evidence::driver_missing(
                                            &rc,
                                            StreamId::LIDAR,
                                            "velo",
                                            "velo-driver",
                                            seq,
                                            tov,
                                            due,
                                            now(),
                                            reason,
                                        )),
                                        &ctx,
                                        "velo-driver",
                                    );
                                }
                            }
                            mark = bytes_alloc(SLOT_VELO_DRIVER);
                        });
                        (totals, driver_bytes)
                    }
                })?,
        ),
        _ => None,
    };

    set_stage_slot(SLOT_DRIVER);
    let rc = ctx.row_ctx();
    // Driver rows carry the frame's own allocation: the `SLOT_DRIVER` counter
    // delta between consecutive admits (two atomic loads per frame, nothing
    // else added to the frame path). The driver decodes on this thread, and
    // between the end of one admit and the start of the next it does exactly
    // one frame's work — the PNG decode and the Arrow batch build — so that is
    // what the number contains; anything else run on this thread in that
    // window would be counted too.
    let mut driver_mark = bytes_alloc(SLOT_DRIVER);
    // Built before the first admit, so it is in no frame's `bytes_alloc`.
    let cam_gap_at: BTreeMap<u64, Gap> = cam_gaps.iter().map(|g| (g.first, *g)).collect();
    let totals = driver.run(&clock, spin, &schema, &mut |ev| {
        let frame_bytes = bytes_alloc(SLOT_DRIVER) - driver_mark;
        match ev {
            DriverEvent::Sample(s) => {
                let admitted = admission.admit(s);
                let mut row = Evidence::driver_admitted(&rc, &admitted, "cam0", "driver");
                row.bytes_alloc = frame_bytes;
                sink.send(EvRow::Evidence(row), &ctx, "driver");
            }
            DriverEvent::Missing {
                seq,
                tov,
                due,
                reason,
            } => {
                if reason == ABSENT_IN_SOURCE {
                    if let Some(g) = cam_gap_at.get(&seq) {
                        sink.send(
                            EvRow::Event(Event::source_gap(
                                &rc,
                                now(),
                                "driver",
                                admission.admitted(),
                                format!(
                                    "camera {}: no frame in the source ({ABSENT_IN_SOURCE}); recorded as missing inputs, and a sweep of their numbers has no camera frame to pair with",
                                    g.describe()
                                ),
                            )),
                            &ctx,
                            "driver",
                        );
                    }
                }
                sink.send(
                    EvRow::Evidence(Evidence::driver_missing(
                        &rc,
                        StreamId::CAM0,
                        "cam0",
                        "driver",
                        seq,
                        tov,
                        due,
                        now(),
                        reason,
                    )),
                    &ctx,
                    "driver",
                );
            }
        }
        driver_mark = bytes_alloc(SLOT_DRIVER);
    });

    // Shutdown is unconditional and ordered: close the data queues, join both
    // consumers, record how the run ended, then let `Recorder::finish` close the
    // sink and join `rec`. Every join result is captured rather than `?`-ed out
    // of the middle of the sequence, so a panicked stage cannot leave the
    // recorder blocked on a queue nobody closed with `evidence.csv` truncated
    // mid-row. (A panic while a stage held a queue mutex poisons it; D9's
    // `PoisonError::into_inner` policy is what keeps the rest of this sequence
    // working.)
    //
    // With a second producer the order gains one step, and it is strictly
    // PRODUCERS, then queues, then consumers. The velodyne driver is still
    // admitting when the camera's replay returns -- the two streams are the
    // same length but not the same phase -- so its thread is joined BEFORE
    // any queue is closed. Closing `velo->cloud` first would be the obvious
    // minimal edit and it fails silently rather than loudly: every sweep
    // admitted during the drain comes back `PushOutcome::Closed`, which is
    // counted as a drop and written as a `closed` drop row, so
    // `delivered + dropped == admitted` still balances and the run reports
    // OK while the sweeps have vanished from the measurement.
    // With a CHAIN the order stops being "producers, queues, consumers" and
    // becomes strictly upstream to downstream, one rung at a time: close a
    // queue only once everything that pushes to it has stopped, then join the
    // thread that pops it, then move down.
    //
    // `det->cloud` is the rung that makes this necessary, and getting it wrong
    // fails SILENTLY rather than loudly. Closing it up here beside the others
    // -- the obvious minimal edit -- would make every cloud `reduce` admits
    // during its drain come back `PushOutcome::Closed`, which is counted as a
    // drop and written as a `closed` drop row. `delivered + dropped ==
    // admitted` would still balance, every invariant would still print OK, and
    // the derived samples would simply have vanished from the measurement. So
    // `det_q` is closed after `reduce` has been JOINED, not before.
    let velo_res = velo_handle.map(std::thread::JoinHandle::join);

    if let Some(q) = &proc_q {
        q.close();
    }
    if let Some(q) = &rerun_q {
        q.close();
    }
    // The detector's queue: its producer, the camera driver, has stopped.
    if let Some(q) = &camdet_q {
        q.close();
    }
    // Both consumers of the velodyne stream: its producer has stopped.
    if let Some(q) = &velo_q {
        q.close();
    }
    if let Some(q) = &reduce_q {
        q.close();
    }
    let proc_res = proc_handle.map(std::thread::JoinHandle::join);
    // `camdet` is still admitting onto `cam_det->track` until this join
    // returns: it can be most of a frame's inference behind the driver.
    let camdet_res = camdet_handle.map(std::thread::JoinHandle::join);
    // Both camera stages that can admit onto `cam_det->track` -- `proc`
    // without the detector, `camdet` with it -- have stopped, so it can be
    // closed, and it MUST be closed here rather than with the rest, because
    // `track` drains it with `try_pop` and can only know the producer has
    // finished from its own primary edge closing later. Closing it earlier
    // would turn every sample admitted during the producer's drain into a
    // `closed` drop row: counted, balanced, and silently absent from the
    // pairing.
    if let Some(q) = &cam_det_q {
        q.close();
    }
    let rerun_res = rerun_handle.map(std::thread::JoinHandle::join);
    let cloud_res = cloud_handle.map(std::thread::JoinHandle::join);
    // `reduce` is still admitting onto `det->cloud` until this join returns.
    let reduce_res = reduce_handle.map(std::thread::JoinHandle::join);
    // `reduce` has stopped, so BOTH of the edges it admits onto can be closed
    // -- and not one of them. `det->detect` closed a rung early is the same
    // silent failure `det->cloud` was: the clouds admitted during the drain
    // come back `PushOutcome::Closed`, get counted as drops, and every
    // invariant still prints OK while the samples have vanished.
    if let Some(q) = &det_q {
        q.close();
    }
    if let Some(q) = &detect_q {
        q.close();
    }
    let det_cloud_res = det_cloud_handle.map(std::thread::JoinHandle::join);
    // `detect` is still admitting onto `obj->sink` until this join returns,
    // one more rung of the same rule.
    let detect_res = detect_handle.map(std::thread::JoinHandle::join);
    if let Some(q) = &obj_q {
        q.close();
    }
    if let Some(q) = &obj_track_q {
        q.close();
    }
    let obj_res = obj_handle.map(std::thread::JoinHandle::join);
    // One more rung of the same rule, twice: `track` admits onto
    // `track->state` until this join returns, and `state` admits onto
    // `state->sink` until the next one does.
    let track_res = track_handle.map(std::thread::JoinHandle::join);
    if let Some(q) = &state_q {
        q.close();
    }
    let state_res = state_handle.map(std::thread::JoinHandle::join);
    if let Some(q) = &ego_q {
        q.close();
    }
    let state_sink_res = state_sink_handle.map(std::thread::JoinHandle::join);

    let panicked = if matches!(proc_res, Some(Err(_))) {
        Some("proc")
    } else if matches!(rerun_res, Some(Err(_))) {
        Some("rerun")
    } else if matches!(camdet_res, Some(Err(_))) {
        Some("camdet")
    } else if matches!(velo_res, Some(Err(_))) {
        Some("velo-driver")
    } else if matches!(cloud_res, Some(Err(_))) {
        Some("cloud")
    } else if matches!(reduce_res, Some(Err(_))) {
        Some("reduce")
    } else if matches!(det_cloud_res, Some(Err(_))) {
        Some("det-cloud")
    } else if matches!(detect_res, Some(Err(_))) {
        Some("detect")
    } else if matches!(obj_res, Some(Err(_))) {
        Some("obj-sink")
    } else if matches!(track_res, Some(Err(_))) {
        Some("track")
    } else if matches!(state_res, Some(Err(_))) {
        Some("state")
    } else if matches!(state_sink_res, Some(Err(_))) {
        Some("state-sink")
    } else {
        None
    };
    let detail = match panicked {
        Some(stage) => format!("panicked:{stage}"),
        None => "clean".to_string(),
    };
    sink.send(
        EvRow::Event(Event::shutdown(&rc, now(), admission.admitted(), &detail)),
        &ctx,
        "run",
    );
    // The viewer's queue, one rung below every stage that draws. With the
    // dashboard, the `rec` thread draws last and closes it once it has drawn
    // the shutdown event; without it, every producer has stopped already.
    // The viewer thread then drains what is queued -- waiting out a viewer
    // that is not taking data, since nothing upstream waits on it any more --
    // and writes its rows before the sink is closed below.
    if !dashboard_on {
        if let Some(v) = &viewer {
            v.close();
        }
    }
    let viewer_res = viewer_thread.map(ViewerThread::join);
    let rows = recorder.finish()?;
    let (viewer_report, stream) = match viewer_res {
        Some(r) => {
            let (report, stream) = r?;
            (Some(report), Some(stream))
        }
        None => (None, None),
    };

    let proc = match proc_res {
        Some(r) => Some(r.map_err(|_| RunError::Panicked("proc"))?),
        None => None,
    };
    let rerun = match rerun_res {
        Some(r) => Some(r.map_err(|_| RunError::Panicked("rerun"))??),
        None => None,
    };
    // The detector, reconciled against the camera driver's own rows, and its
    // batches written out: after every thread has stopped, so the file costs
    // the pipeline nothing.
    let camdet = match camdet_res {
        Some(r) => {
            let report = r.map_err(|_| RunError::Panicked("camdet"))?;
            let in_check = storage_check_2(
                &producer_ids(&rows, "driver", "cam0"),
                &report.in_by_seq,
                report.delivered,
            );
            let file = dir.join(CAM_DET_FILE);
            if let Some(schema) = &cam_det_sch {
                write_batches(&file, schema, &report.batches)?;
            }
            Some(CamDetRun {
                report,
                model_sha256: cam_det_sch
                    .as_ref()
                    .and_then(|s| s.metadata().get("model_sha256").cloned())
                    .unwrap_or_default(),
                load_ms: model_load_ms,
                period_ns: (a.rate.is_finite() && a.rate > 0.0)
                    .then(|| (median_period_ns(&driver.timestamps) as f64 / a.rate) as i64),
                in_check,
                file,
                // The detector's own: it ran, so the knobs were its.
                queue: cams.knobs(),
            })
        }
        None => None,
    };
    // The derived chain, reconciled before the lidar block it hangs off.
    // Either both halves ran or neither did; one without the other can only
    // mean this function grew a path that spawns one and not the other, and
    // it is reported as absent rather than half-published.
    let reduce_run = match (reduce_res, det_cloud_res) {
        (Some(r), Some(c)) => {
            let report = r.map_err(|_| RunError::Panicked("reduce"))?;
            let cloud = c.map_err(|_| RunError::Panicked("det-cloud"))?;
            // Both checks take their PRODUCER side from the recorded evidence
            // rather than from memory, so each is a statement about the file a
            // reviewer can read rather than about a vector this process held.
            let in_check = storage_check_2(
                &producer_ids(&rows, "velo-driver", "velo"),
                &report.in_by_seq,
                report.delivered,
            );
            let out_check = storage_check_2(
                &producer_ids(&rows, "reduce", "det"),
                &cloud.by_seq,
                report.produced,
            );
            // The third link, reconciled the same way and hanging off the same
            // producer rows. Either both halves ran or neither did.
            let detect_run = match (detect_res, obj_res) {
                (Some(dr), Some(or)) => {
                    let d = dr.map_err(|_| RunError::Panicked("detect"))?;
                    let sink = or.map_err(|_| RunError::Panicked("obj-sink"))?;
                    // `detect`'s INPUT side shares `reduce`'s producer rows
                    // with `det-cloud`: one producer, two consumers, and each
                    // consumer's proof is its own.
                    let in_check = storage_check_2(
                        &producer_ids(&rows, "reduce", "det"),
                        &d.in_by_seq,
                        report.produced,
                    );
                    let out_check = storage_check_2(
                        &producer_ids(&rows, "detect", "obj"),
                        &sink.by_seq,
                        d.produced,
                    );
                    let parent_check = parent_check(&d.out_parent, &sink.parent_by_seq);
                    // The last two links, reconciled on the same terms. All
                    // three have to be present or none is reported: one
                    // without the others can only mean this function grew a
                    // path that spawns some and not the rest.
                    let track_run = match (track_res, state_res, state_sink_res) {
                        (Some(tr), Some(sr), Some(kr)) => {
                            let t = tr.map_err(|_| RunError::Panicked("track"))?;
                            let st = sr.map_err(|_| RunError::Panicked("state"))?;
                            let sk = kr.map_err(|_| RunError::Panicked("state-sink"))?;
                            // `track`'s INPUT side shares `detect`'s producer
                            // rows with `obj-sink`: one producer, two
                            // consumers, each consumer's proof its own.
                            let in_check = storage_check_2(
                                &producer_ids(&rows, "detect", "obj"),
                                &t.in_by_seq,
                                d.produced,
                            );
                            let out_check = storage_check_2(
                                &producer_ids(&rows, "track", "track"),
                                &st.in_by_seq,
                                t.produced,
                            );
                            let state_check = storage_check_2(
                                &producer_ids(&rows, "state", "state"),
                                &sk.by_seq,
                                st.produced,
                            );
                            // `state_parent_check` FIRST: after the `let`
                            // below, the name `parent_check` is a value and
                            // the function is out of reach.
                            // Qualified: `parent_check` is already a VALUE in
                            // this scope, bound a few lines above.
                            let state_parent_check =
                                crate::run::parent_check(&st.out_parent, &sk.parent_by_seq);
                            let parent_check =
                                crate::run::parent_check(&t.out_parent, &st.in_parent_by_seq);
                            // **The camera half.** Joined against the camera
                            // driver's own recorded rows rather than against
                            // anything this process remembered.
                            // With the detector, the frame must also be one
                            // `camdet` produced detections for: `cam_seq` is
                            // the `CAM_DET` seq too, so it names both.
                            let det_ids = camdet
                                .as_ref()
                                .map(|_| producer_ids(&rows, "camdet", "cam_det"));
                            let pair_check =
                                pair_check(&rows, &t.out_cam, a.pair_stale_ms, det_ids.as_ref());
                            // The detector's batches, read where it built
                            // them: joined against `camdet`'s own producer
                            // rows. Nothing to join without the detector.
                            let cam_check = camdet.as_ref().map(|c| {
                                storage_check_2(
                                    &producer_ids(&rows, "camdet", "cam_det"),
                                    &t.cam_by_seq,
                                    c.report.produced,
                                )
                            });
                            // What the fusion produced, set by set, after
                            // every thread has stopped.
                            let fused_file = dir.join(FUSED_FILE);
                            if let Some(schema) = &trk_schema {
                                write_batches(&fused_file, schema, &t.batches)?;
                            }
                            // The instants either sensor's source left
                            // without a partner, from the evidence.
                            let gap_pairs = source_gap_pairs(
                                &rows,
                                &velo
                                    .as_ref()
                                    .map(|v| v.absent_in_source())
                                    .unwrap_or_default(),
                                &driver.absent_in_source(),
                                &t.out_cam,
                                &t.pair_absent_in_source,
                            );
                            Some(TrackRun {
                                report: t,
                                state: st,
                                sink: sk,
                                in_check,
                                out_check,
                                state_check,
                                parent_check,
                                state_parent_check,
                                pair_check,
                                cam_check,
                                detector: camdet.is_some(),
                                fused_file,
                                calib,
                                gap_pairs,
                            })
                        }
                        _ => None,
                    };
                    Some(DetectRun {
                        report: d,
                        sink,
                        in_check,
                        out_check,
                        parent_check,
                        track: track_run,
                    })
                }
                _ => None,
            };
            let parent_check = parent_check(&report.out_parent, &cloud.parent_by_seq);
            Some(ReduceRun {
                voxel_size: a.voxel_size_m,
                report,
                cloud,
                in_check,
                out_check,
                parent_check,
                detect: detect_run,
            })
        }
        _ => None,
    };
    let velo_run = match (velo.as_ref(), velo_res, cloud_res) {
        (Some(v), Some(r), Some(c)) => {
            let (totals, driver_bytes) = r.map_err(|_| RunError::Panicked("velo-driver"))?;
            let cloud = c.map_err(|_| RunError::Panicked("cloud"))?;
            let storage_check = velo_storage_check(&rows, &cloud, totals.admitted);
            Some(VeloRun {
                n_sweeps: v.len(),
                frames: v.frame_slots(),
                absent_in_source: v.absent_in_source(),
                gaps: v.gaps(),
                totals,
                driver_bytes,
                cloud,
                storage_check,
                reduce: reduce_run,
            })
        }
        // Either the lidar did not run, or one half of it did and the other
        // did not -- which can only happen if this function grew a path that
        // spawns one without the other. Treated as "no lidar" rather than
        // half-reported, so a half-wired run cannot publish a lidar block.
        _ => None,
    };
    let mut edge_reports = Vec::new();
    for (name, q) in [
        ("cam0->proc", &proc_q),
        ("cam0->rerun", &rerun_q),
        ("cam0->camdet", &camdet_q),
        ("velo->cloud", &velo_q),
        ("velo->reduce", &reduce_q),
        ("det->cloud", &det_q),
        ("det->detect", &detect_q),
        ("obj->sink", &obj_q),
        ("obj->track", &obj_track_q),
        ("cam_det->track", &cam_det_q),
        ("track->state", &state_q),
        ("state->sink", &ego_q),
    ] {
        if let Some(q) = q {
            edge_reports.push(EdgeReport {
                name,
                admitted: admission.admitted_on(name),
                queue_dropped: q.dropped(),
                blocked_ns: admission.push_blocked_ns(name),
            });
        }
    }
    // The viewer's edge, on the same terms: every drawing pushed, every one
    // not delivered, and how long each push took.
    if let Some(v) = &viewer {
        edge_reports.push(EdgeReport {
            name: VIEWER_EDGE,
            admitted: v.admitted(),
            queue_dropped: v.dropped(),
            blocked_ns: v.push_ns(),
        });
    }

    let storage_check = storage_check(&rows, proc.as_ref(), rerun.as_ref());
    Ok(RunReport {
        run_id: name,
        dir,
        n_frames: driver.frame_slots(),
        cam_on_disk: driver.len(),
        cam_absent_in_source: driver.absent_in_source(),
        camera: cams.knobs(),
        reuse_output: a.reuse_output,
        rerun_mode: rerun_mode_name,
        totals,
        admitted: admission.admitted(),
        admitted_by_stream: admission.admitted_by_stream(),
        proc,
        rerun,
        camdet,
        velo: velo_run,
        edges: edge_reports,
        sink_blocked_ns: sink.blocked_ns(),
        evidence_lost: sink.lost(),
        recorder_degraded: sink.degraded(),
        rows,
        storage_check,
        untracked_bytes: bytes_alloc(SLOT_UNTRACKED),
        stream,
        viewer: viewer_report,
        recording,
    })
}

/// Compares the driver's `storage_id` per seq with what each consumer
/// re-derived from the real buffer; reports the first mismatching seq.
fn storage_check(
    rows: &[Evidence],
    proc: Option<&ProcReport>,
    rerun: Option<&RerunReport>,
) -> StorageCheck {
    let stages: u8 = 1 + u8::from(proc.is_some()) + u8::from(rerun.is_some());
    let driver_ids: BTreeMap<u64, usize> = rows
        .iter()
        .filter(|r| r.stage == "driver" && r.outcome == Outcome::Delivered)
        .map(|r| (r.seq, r.storage_id))
        .collect();
    let proc_ids: BTreeMap<u64, usize> = proc
        .map(|p| p.by_seq.iter().copied().collect())
        .unwrap_or_default();
    let rerun_ids: BTreeMap<u64, usize> = rerun
        .map(|r| r.by_seq.iter().copied().collect())
        .unwrap_or_default();
    for (&seq, &driver) in &driver_ids {
        let p = proc_ids.get(&seq).copied();
        let r = rerun_ids.get(&seq).copied();
        if p.is_some_and(|p| p != driver) || r.is_some_and(|r| r != driver) {
            return StorageCheck::Mismatch {
                seq,
                driver,
                proc: p,
                rerun: r,
            };
        }
    }
    // A consumer row whose seq the driver never admitted is also a mismatch.
    let orphan = proc_ids
        .keys()
        .chain(rerun_ids.keys())
        .find(|seq| !driver_ids.contains_key(seq));
    if let Some(&seq) = orphan {
        return StorageCheck::Mismatch {
            seq,
            driver: 0,
            proc: proc_ids.get(&seq).copied(),
            rerun: rerun_ids.get(&seq).copied(),
        };
    }
    StorageCheck::Ok { stages }
}

/// The lidar's own `storage_id` check: velodyne driver -> `cloud`.
///
/// A separate function from [`storage_check`], not an extra argument to it,
/// and the reason is a defect the camera version would have hidden. That one
/// selects its driver rows with `stage == "driver"`; the velodyne driver must
/// use a different stage string (two producers both numbering from 0 collide
/// in a map keyed on `seq`), so every lidar row is filtered out. Plumbed in
/// there, the lidar would have compared **nothing** and returned `Ok` -- a
/// check that cannot fail, reported as proof.
///
/// Hence [`VeloStorageCheck::NotChecked`]: an empty population is its own
/// answer here, and it is not `Ok`.
fn velo_storage_check(rows: &[Evidence], cloud: &CloudReport, admitted: u64) -> VeloStorageCheck {
    storage_check_2(
        &producer_ids(rows, "velo-driver", "velo"),
        &cloud.by_seq,
        admitted,
    )
}

/// The provenance join: every derived sample that reached the far end must
/// name the sweep `reduce` built it from.
///
/// `producer` is `(derived seq -> sweep seq)` as the transform recorded it;
/// `consumer` is `(derived seq -> the parent seq that arrived)`. A derived
/// sample the producer never recorded is a mismatch rather than a skip: it
/// means something reached the consumer that this stage did not admit.
fn parent_check(producer: &[(u64, u64)], consumer: &[(u64, Option<u64>)]) -> ParentCheck {
    let want: BTreeMap<u64, u64> = producer.iter().copied().collect();
    let mut compared = 0usize;
    for &(seq, got) in consumer {
        match want.get(&seq) {
            Some(&sweep) if got == Some(sweep) => compared += 1,
            Some(&sweep) => {
                return ParentCheck::Mismatch {
                    seq,
                    want: sweep,
                    got,
                }
            }
            // No producer record for this sample at all. `want: seq` is not a
            // sweep number here, it is "the producer never admitted this".
            None => {
                return ParentCheck::Mismatch {
                    seq,
                    want: seq,
                    got,
                }
            }
        }
    }
    if compared == 0 {
        return ParentCheck::NotChecked {
            produced: producer.len(),
        };
    }
    ParentCheck::Ok { compared }
}

/// Joins every fused sample's `cam_seq` against the camera driver's own rows.
///
/// **The camera half of the provenance, and the reason it is not a lie.**
/// `Sample::parent` holds one parent and it holds the lidar one; the camera
/// frame travels as a payload column, and a payload column nothing reads back
/// is exactly what `Sample::parent` was before this project noticed: declared,
/// set to `None` everywhere, read nowhere, and a derived sample naming the
/// WRONG sweep passed every test in the workspace.
///
/// Two things are checked and the second is the one that bites. First, that
/// the named frame is one the camera driver really admitted -- the `cam0`
/// pseudo-edge's own rows, from the recorded evidence rather than from memory.
/// Second, that its INSTANT is where the set's own label says: inside that
/// sweep's range for a completed set, which is the pairing rule itself, and
/// before the range but no further before the sweep's trigger than
/// `--pair-stale-ms` for a degraded one -- the trigger riding in `out_cam`,
/// since no evidence column carries it. A fusion that paired the wrong frame
/// would pass the first check and fail the second.
///
/// A degraded set used to be held to the completed set's rule, so every run
/// that asked for stale pairs printed a MISMATCH for the first one and
/// reported `cam_pair_join_ok = false`, while the comment here said stale sets
/// were excluded. They are checked now, against the rule that produced them.
///
/// With the detector on, `detections` is `camdet`'s own producer rows by seq,
/// and a third claim is checked: the named frame is one the detector really
/// produced a `CAM_DET` batch for. `camdet` numbers its batches by frame, so
/// `cam_seq` names the detections the fusion used as well as the picture, and
/// this is the join that makes that second name a fact rather than a
/// convention.
fn pair_check(
    rows: &[Evidence],
    out_cam: &[(u64, i64, Pairing, i64)],
    stale_ms: u64,
    detections: Option<&BTreeMap<u64, usize>>,
) -> PairCheck {
    // `cam0` driver rows: frame seq -> its instant on the sensor clock.
    let frames: BTreeMap<u64, i64> = rows
        .iter()
        .filter(|r| r.stage == "driver" && r.edge == "cam0" && r.outcome == Outcome::Delivered)
        .map(|r| (r.seq, r.tov_start_ns))
        .collect();
    // `track` producer rows: fused seq -> the sweep range it inherited.
    let sweeps: BTreeMap<u64, (i64, i64)> = rows
        .iter()
        .filter(|r| r.stage == "track" && r.edge == "track" && r.outcome == Outcome::Delivered)
        .map(|r| (r.seq, (r.tov_start_ns, r.tov_end_ns)))
        .collect();
    let window_ns = i64::try_from(stale_ms)
        .unwrap_or(i64::MAX)
        .saturating_mul(1_000_000);
    let (mut checked, mut stale) = (0usize, 0usize);
    for &(seq, cam_seq, pairing, trigger) in out_cam {
        // -1 cannot occur on a produced sample: only a paired or stale set
        // produces one, and both name a frame.
        let Ok(cs) = u64::try_from(cam_seq) else {
            continue;
        };
        let Some(&t) = frames.get(&cs) else {
            return PairCheck::Unknown { seq, cam_seq };
        };
        if detections.is_some_and(|d| !d.contains_key(&cs)) {
            return PairCheck::NoDetections { seq, cam_seq };
        }
        let Some(&(start, end)) = sweeps.get(&seq) else {
            continue;
        };
        if pairing == Pairing::Stale {
            // The window's own rule, from `pair_camera`: strictly before the
            // range, and no further before the trigger than the declared
            // window -- the trigger held to the range as `pair_camera` holds it.
            let trigger = trigger.max(start).min(end);
            let in_window = t < start && trigger - t <= window_ns;
            if !in_window {
                return PairCheck::OutsideWindow { seq, cam_seq };
            }
            stale += 1;
        } else {
            if t < start || t >= end {
                return PairCheck::Outside { seq, cam_seq };
            }
            checked += 1;
        }
    }
    if checked + stale == 0 {
        return PairCheck::NotChecked {
            produced: out_cam.len(),
        };
    }
    PairCheck::Ok {
        checked,
        stale,
        window_ms: stale_ms,
    }
}

/// The instants a gap in either sensor's source left without a partner
/// ([`SourceGapPairs`]): read off the camera edge's own rows at `track` --
/// whose seq is the frame's number, the one both sensors share -- off the
/// sets' `cam_seq`, and off `track`'s own record of the sweeps it expired for
/// a camera frame the source never had. Never assumed.
fn source_gap_pairs(
    rows: &[Evidence],
    no_sweep: &[u64],
    no_frame: &[u64],
    out_cam: &[(u64, i64, Pairing, i64)],
    expired_no_frame: &[u64],
) -> SourceGapPairs {
    if no_sweep.is_empty() && no_frame.is_empty() {
        return SourceGapPairs::default();
    }
    let reached: BTreeSet<u64> = rows
        .iter()
        .filter(|r| r.edge == "cam_det->track" && r.outcome == Outcome::Delivered)
        .map(|r| r.seq)
        .collect();
    let used: BTreeSet<u64> = out_cam
        .iter()
        .filter_map(|&(_, cam_seq, _, _)| u64::try_from(cam_seq).ok())
        .collect();
    let expired: BTreeSet<u64> = expired_no_frame.iter().copied().collect();
    let of = |frames: &[u64], set: &BTreeSet<u64>| -> Vec<u64> {
        frames.iter().copied().filter(|f| set.contains(f)).collect()
    };
    SourceGapPairs {
        no_sweep: no_sweep.to_vec(),
        no_sweep_camera_reached_track: of(no_sweep, &reached),
        no_sweep_camera_used_stale: of(no_sweep, &used),
        no_frame: no_frame.to_vec(),
        no_frame_sweep_expired: of(no_frame, &expired),
    }
}

/// `seq -> storage_id` for every sample a producer recorded on its own
/// pseudo-edge.
///
/// Selected by stage **and** edge. Stage alone was enough while every producer
/// wrote one kind of row, but `reduce` writes two per sample — a consumer row
/// on `velo->reduce` carrying the INPUT's address and a producer row on `det`
/// carrying the OUTPUT's. Matching on stage alone would mix the two addresses
/// into one map and compare a derived buffer against a sweep.
fn producer_ids(rows: &[Evidence], stage: &str, edge: &str) -> BTreeMap<u64, usize> {
    rows.iter()
        .filter(|r| r.stage == stage && r.edge == edge && r.outcome == Outcome::Delivered)
        .map(|r| (r.seq, r.storage_id))
        .collect()
}

/// One producer -> one consumer, over a stream: every id the consumer
/// re-derived from the real buffer must equal the one the producer recorded
/// for that seq.
fn storage_check_2(
    producer: &BTreeMap<u64, usize>,
    consumer: &[(u64, usize)],
    admitted: u64,
) -> VeloStorageCheck {
    let mut compared = 0usize;
    for &(seq, id) in consumer {
        match producer.get(&seq) {
            Some(&driver) if driver == id => compared += 1,
            Some(&driver) => {
                return VeloStorageCheck::Mismatch {
                    seq,
                    driver,
                    consumer: Some(id),
                }
            }
            // A consumer row for a sample no producer recorded. `driver: 0`
            // is not an address here, it is "there was none".
            None => {
                return VeloStorageCheck::Mismatch {
                    seq,
                    driver: 0,
                    consumer: Some(id),
                }
            }
        }
    }
    if compared == 0 {
        return VeloStorageCheck::NotChecked {
            admitted: admitted as usize,
        };
    }
    VeloStorageCheck::Ok { compared }
}

/// Where the detector's batches go, under the run directory.
///
/// One batch per frame the detector ran on. Which frames those are is the
/// queue's business, not the model's: on a loaded host `cam0->camdet` can
/// evict a frame, and that run's file is then shorter, with a different hash.
/// Two runs are equal frame by frame, over the frames both delivered.
pub const CAM_DET_FILE: &str = "cam_det.arrows";

/// Where the fusion's sets go, under the run directory.
///
/// `cam_det.arrows` says what the camera saw; this says what the fusion did
/// with it, sweep by sweep: every track's lidar lanes and camera lanes, the
/// detections no track matched, and the provenance -- `cam_seq`,
/// `pair_outcome`, `pair_age_ns` -- that attributes a class to the frame it
/// came from. The stale-camera comparison is two of these files: the lidar
/// lanes must be identical, and every difference in the camera lanes is then
/// the camera's.
///
/// Two runs write the same file only when the same sets complete. The fusion
/// is a function of what reached it, but on a loaded host a set can expire
/// (a slow detector frame makes it `pair_late`) or a sweep be evicted on the
/// way, and the file then differs from a quiet run's. So runs are compared
/// set by set, over the sets both completed, and never by the file's hash
/// alone.
pub const FUSED_FILE: &str = "fused.arrows";

/// Writes batches in order, as one Arrow IPC stream: the detector's into
/// [`CAM_DET_FILE`], the fusion's into [`FUSED_FILE`].
///
/// The evidence rows say THAT a stage contributed and when; these files say
/// WHAT it contributed, sample by sample, in the exact bytes the next stage
/// was handed. They are what lets a reader check an answer's camera half after
/// the run, and what two runs are compared on: the detector's batches of the
/// frames both runs delivered must be byte-identical, and so must the fusion's
/// lidar lanes. Written after every thread has stopped, so they cost the
/// measured pipeline nothing.
///
/// A stage that ran and produced nothing still gets its file: a stream with
/// the stage's own `schema` and no batch. A missing file read as a run that
/// crashed before writing it, where the truth -- every set expired, say --
/// is in the summary and should be in the file too. `schema` is used only
/// then; a batch's own schema heads a file that has one, as it always has.
fn write_batches(path: &Path, schema: &SchemaRef, batches: &[RecordBatch]) -> Result<(), RunError> {
    let schema = batches
        .first()
        .map_or_else(|| schema.clone(), RecordBatch::schema);
    let file = std::io::BufWriter::new(std::fs::File::create(path)?);
    let mut w =
        StreamWriter::try_new(file, &schema).map_err(|e| RunError::Io(std::io::Error::other(e)))?;
    for b in batches {
        w.write(b)
            .map_err(|e| RunError::Io(std::io::Error::other(e)))?;
    }
    w.finish()
        .map_err(|e| RunError::Io(std::io::Error::other(e)))?;
    Ok(())
}

fn write_run_json(path: &Path, json: &RunJson<'_>) -> Result<(), RunError> {
    std::fs::write(path, serde_json::to_string_pretty(json)?)?;
    Ok(())
}

/// `git rev-parse --short=12 HEAD` of the current directory, or `"unknown"`.
fn git_sha() -> String {
    std::process::Command::new("git")
        .args(["rev-parse", "--short=12", "HEAD"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|| "unknown".to_string())
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use std::sync::Arc;

    use arrow::datatypes::Schema;
    use pipes_core::clock::{SensorTime, Tov};

    use super::*;

    const MS: i64 = 1_000_000;

    #[test]
    fn a_viewer_the_run_starts_opens_at_the_layout_s_size() {
        let opts = viewer_spawn_options(9876);
        assert_eq!(opts.port, 9876);
        assert_eq!(opts.extra_args, ["--window-size", "1600x900"]);
        // Everything else is the SDK's own: the executable from PATH, a
        // window rather than headless.
        let default = rerun::SpawnOptions::default();
        assert_eq!(opts.executable_name, default.executable_name);
        assert!(!opts.headless);
        // The run waits for the viewer itself, before its clock starts; the
        // SDK's own wait panics in a debug build and carries on in release.
        assert!(!opts.wait_for_bind);
    }

    #[test]
    fn a_stage_that_produced_nothing_still_writes_its_file() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join(FUSED_FILE);
        let schema = pipes_kitti::track::track_schema();
        write_batches(&path, &schema, &[]).unwrap();
        let f = std::fs::File::open(&path).expect("no file for a stage that ran");
        let r = arrow::ipc::reader::StreamReader::try_new(f, None).unwrap();
        assert_eq!(
            r.schema(),
            schema,
            "the empty stream is not in the stage's format"
        );
        assert_eq!(r.count(), 0, "batches appeared from nowhere");
    }

    #[test]
    fn the_fusion_sets_line_names_every_sweep_that_never_reached_track() {
        // The stalled-viewer run: 154 sweeps, `obj->track` evicted 87, and
        // the 67 that arrived became 63 completed and 4 expired sets.
        let b = BeforeTrack {
            sweeps: 154,
            reached: 67,
            skipped: 0,
            evicted_velo_reduce: 0,
            evicted_det_detect: 0,
            evicted_obj_track: 87,
            stage_errors: 0,
            absent_in_source: 0,
        };
        assert_eq!(b.never_reached(), 87);
        assert!(b.balanced());
        assert_eq!(
            b.line(),
            "NEVER REACHED track 87 of 154 sweeps = evicted on obj->track 87 + det->detect 0 \
             + velo->reduce 0 + skipped by the driver 0 + stage errors 0"
        );
        // A healthy run says so in five words.
        let ok = BeforeTrack {
            reached: 154,
            evicted_obj_track: 0,
            ..b
        };
        assert_eq!(ok.line(), "NEVER REACHED track 0 of 154 sweeps");
        // Losses that do not add up are called out, not printed as a sum.
        let off = BeforeTrack {
            evicted_obj_track: 86,
            ..b
        };
        assert!(!off.balanced());
        assert!(off
            .line()
            .ends_with("MISMATCH: those do not add up to the sweeps that never arrived"));
    }

    #[test]
    fn the_viewer_wait_returns_once_something_listens() {
        let l = std::net::TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0)).unwrap();
        let port = l.local_addr().unwrap().port();
        let mut notes = 0;
        let waited = wait_for_listener(
            LOCAL_VIEWER_HOST,
            port,
            Duration::from_secs(5),
            Duration::ZERO,
            |_| {
                notes += 1;
            },
        )
        .expect("a listening port was not seen");
        assert!(waited < Duration::from_secs(5), "{waited:?}");
        assert_eq!(
            notes, 0,
            "a wait that succeeded at once said it was waiting"
        );
    }

    #[test]
    fn the_viewer_wait_gives_up_at_its_budget_and_says_it_is_waiting() {
        // A port nothing listens on: bound by the OS, then released.
        let port = {
            let l = std::net::TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0)).unwrap();
            l.local_addr().unwrap().port()
        };
        let budget = Duration::from_millis(400);
        let mut notes = Vec::new();
        let waited = wait_for_listener(
            LOCAL_VIEWER_HOST,
            port,
            budget,
            Duration::from_millis(150),
            |w| {
                notes.push(w);
            },
        )
        .expect_err("a port nothing listens on was taken for a viewer");
        assert!(
            waited >= budget,
            "gave up after {waited:?}, before {budget:?}"
        );
        assert_eq!(
            notes.len(),
            1,
            "said it was waiting {notes:?} times, not once"
        );
        assert!(notes[0] >= Duration::from_millis(150), "{notes:?}");
        // And the error the run stops with names every way out.
        let e = RerunError::NotListening {
            port,
            pid: 7,
            waited_s: waited.as_secs_f64(),
        }
        .to_string();
        assert!(e.contains(&format!("port {port} ")), "{e}");
        assert!(e.contains("before its clock started"), "{e}");
        assert!(e.contains("python -m pip install rerun-sdk==0.38.1"), "{e}");
        assert!(e.contains("Windows Firewall"), "{e}");
        assert!(e.contains("--rerun rrd"), "{e}");
    }

    #[test]
    fn the_viewer_wait_resolves_a_host_name() {
        // `--rerun-host` names hosts, as `host.docker.internal` does, so the
        // wait must resolve one rather than take only an address.
        let l = std::net::TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0)).unwrap();
        let port = l.local_addr().unwrap().port();
        let waited = wait_for_listener(
            "localhost",
            port,
            Duration::from_secs(5),
            Duration::ZERO,
            |_| {},
        )
        .expect("a listener behind `localhost` was not seen");
        assert!(waited < Duration::from_secs(5), "{waited:?}");
    }

    #[test]
    fn a_host_that_does_not_resolve_is_nothing_listening() {
        let port = crate::cli::DEFAULT_VIEWER_PORT;
        let waited = wait_for_listener(
            "no-such-host.invalid",
            port,
            Duration::from_millis(300),
            Duration::from_secs(60),
            |_| {},
        )
        .expect_err("a host that does not resolve was taken for a viewer");
        assert!(waited >= Duration::from_millis(300), "{waited:?}");
        let e = RerunError::NoRemoteViewer {
            host: "no-such-host.invalid".to_string(),
            port,
            waited_s: waited.as_secs_f64(),
        }
        .to_string();
        assert!(e.contains(&format!("no-such-host.invalid:{port}")), "{e}");
        assert!(e.contains("before its clock started"), "{e}");
        assert!(e.contains("python -m pip install rerun-sdk==0.38.1"), "{e}");
        assert!(e.contains("--rerun rrd"), "{e}");
    }

    fn row(
        stream: StreamId,
        edge: &'static str,
        stage: &'static str,
        seq: u64,
        tov: Tov,
    ) -> Evidence {
        let ctx = RowCtx {
            run_id: "t".to_string(),
            t0_host: HostTime(0),
            epoch: 0,
        };
        let s = Sample {
            stream,
            seq,
            arrival_seq: seq,
            parent: None,
            tov,
            epoch: 0,
            due: None,
            arrival: HostTime(0),
            payload: RecordBatch::new_empty(Arc::new(Schema::empty())),
            storage_id: 0,
            decode_ns: 0,
        };
        Evidence::driver_admitted(&ctx, &s, edge, stage)
    }

    /// Three camera frames at 10, 110 and 210 ms, and the three fused
    /// samples of the sweeps `[0, 100)`, `[100, 200)` and `[200, 300)` ms.
    fn rows() -> Vec<Evidence> {
        let mut v = Vec::new();
        for k in 0..3i64 {
            v.push(row(
                StreamId::CAM0,
                "cam0",
                "driver",
                k as u64,
                Tov::Time(SensorTime((100 * k + 10) * MS)),
            ));
            v.push(row(
                StreamId::TRACKS,
                "track",
                "track",
                k as u64,
                Tov::Range {
                    start: SensorTime(100 * k * MS),
                    end: SensorTime(100 * (k + 1) * MS),
                },
            ));
        }
        v
    }

    /// A completed set is held to the pairing rule, and a degraded one to the
    /// declared window that produced it -- not to the completed set's rule,
    /// which every stale pair breaks by definition -- counted back from the
    /// sweep's trigger, the instant its pair age is measured from.
    #[test]
    fn a_stale_pair_is_held_to_its_window_and_a_clean_one_to_its_sweep() {
        use Pairing::{Paired, Stale};
        let r = rows();
        // Each sweep's trigger, halfway through its range.
        let trig = |k: i64| (100 * k + 50) * MS;
        assert_eq!(
            pair_check(
                &r,
                &[(0, 0, Paired, trig(0)), (1, 1, Paired, trig(1))],
                0,
                None
            ),
            PairCheck::Ok {
                checked: 2,
                stale: 0,
                window_ms: 0
            }
        );
        // Sweep 2 paired stale with frame 1, 140 ms before the sweep's
        // trigger: inside a 150 ms window. This is what used to print a
        // MISMATCH.
        assert_eq!(
            pair_check(
                &r,
                &[(0, 0, Paired, trig(0)), (2, 1, Stale, trig(2))],
                150,
                None
            ),
            PairCheck::Ok {
                checked: 1,
                stale: 1,
                window_ms: 150
            }
        );
        // The same pair under a 100 ms window is one the window did not make.
        // Frame 1 is only 90 ms before the sweep's START, and the window used
        // to be counted from there: it passed, and was labelled 140 ms stale.
        assert_eq!(
            pair_check(&r, &[(2, 1, Stale, trig(2))], 100, None),
            PairCheck::OutsideWindow { seq: 2, cam_seq: 1 }
        );
        // A trigger outside the range (a payload without one reads 0) is held
        // to the range, as `pair_camera` holds it: the window can narrow to
        // the range's start, never widen to every older frame.
        assert!(pair_check(&r, &[(2, 1, Stale, 0)], 100, None).passed());
        assert_eq!(
            pair_check(&r, &[(2, 1, Stale, 0)], 80, None),
            PairCheck::OutsideWindow { seq: 2, cam_seq: 1 }
        );
        // A "stale" label on the frame inside its own sweep is a lie too.
        assert_eq!(
            pair_check(&r, &[(2, 2, Stale, trig(2))], 500, None),
            PairCheck::OutsideWindow { seq: 2, cam_seq: 2 }
        );
        // And the completed set's rule still bites: the old check, unchanged.
        assert_eq!(
            pair_check(&r, &[(2, 1, Paired, trig(2))], 500, None),
            PairCheck::Outside { seq: 2, cam_seq: 1 }
        );
        assert_eq!(
            pair_check(&r, &[(0, 7, Paired, trig(0))], 0, None),
            PairCheck::Unknown { seq: 0, cam_seq: 7 }
        );
        assert!(matches!(
            pair_check(&r, &[], 0, None),
            PairCheck::NotChecked { produced: 0 }
        ));
        // With the detector on, the named frame must have detections: frame
        // 1 has a `CAM_DET` batch and frame 0 does not.
        let dets = BTreeMap::from([(1u64, 0x100usize)]);
        assert_eq!(
            pair_check(&r, &[(1, 1, Paired, trig(1))], 0, Some(&dets)),
            PairCheck::Ok {
                checked: 1,
                stale: 0,
                window_ms: 0
            }
        );
        assert_eq!(
            pair_check(&r, &[(0, 0, Paired, trig(0))], 0, Some(&dets)),
            PairCheck::NoDetections { seq: 0, cam_seq: 0 }
        );
        // The line names both populations only when there are degraded ones.
        let both = PairCheck::Ok {
            checked: 1,
            stale: 1,
            window_ms: 100,
        };
        assert!(both
            .line()
            .starts_with("fused cam_seq = a frame the camera really admitted"));
        assert!(both.line().contains(
            "at most 100 ms before its trigger, the declared stale window, on all 1 degraded"
        ));
    }

    /// The instants a gap in the source left without a partner are read off
    /// the evidence, never assumed. A camera frame of the lidar's gap that
    /// never reached `track` -- evicted on its way -- is not one that "reached
    /// track and had no sweep to pair with"; one a stale window used is named
    /// as used; and a sweep of the camera's gap is named as expired only when
    /// `track` expired it so.
    #[test]
    fn the_instants_a_gap_left_unpaired_are_read_off_the_evidence() {
        let reached = |seq: u64| {
            row(
                StreamId::CAM_DET,
                "cam_det->track",
                "track",
                seq,
                Tov::Time(SensorTime(seq as i64 * 100 * MS)),
            )
        };
        // Camera frames 2, 3 and 5 reached `track`; frame 4 did not.
        let rows = vec![reached(2), reached(3), reached(5)];
        let g = source_gap_pairs(&rows, &[3, 4], &[], &[], &[]);
        assert_eq!(g.no_sweep, vec![3, 4]);
        assert_eq!(
            g.no_sweep_camera_reached_track,
            vec![3],
            "frame 4's camera frame never reached track"
        );
        assert!(g.no_sweep_camera_used_stale.is_empty());
        // Sweep 5 paired stale with frame 4.
        let g = source_gap_pairs(&rows, &[3, 4], &[], &[(5, 4, Pairing::Stale, 0)], &[]);
        assert_eq!(g.no_sweep_camera_used_stale, vec![4]);
        // The camera's gap at 6-7: only sweep 7 reached track and expired.
        let g = source_gap_pairs(&rows, &[], &[6, 7], &[], &[7]);
        assert_eq!(
            (g.no_frame, g.no_frame_sweep_expired),
            (vec![6, 7], vec![7])
        );
        // No gap, nothing to say.
        assert_eq!(
            source_gap_pairs(&rows, &[], &[], &[], &[]),
            SourceGapPairs::default()
        );
    }

    /// `pipes run <args>` as the binary parses it.
    fn run_args(args: &[&str]) -> RunArgs {
        use clap::Parser;
        let argv = ["pipes", "run"].iter().chain(args).copied();
        match crate::cli::Cli::try_parse_from(argv).unwrap().cmd {
            crate::cli::Cmd::Run(a) => a,
            crate::cli::Cmd::Drives(_) => panic!("parsed as `drives`"),
        }
    }

    /// **The camera knobs' rule.** With the detector running, `--cap`,
    /// `--policy`, `--block-max-wait-ms` and `--consumer-delay-ms` land on
    /// `cam0->camdet` and `proc` does not run; without it, the same
    /// arguments land on `cam0->proc`. Both directions from the same
    /// argument list, so neither can pass by the knobs landing nowhere.
    #[test]
    fn the_camera_knobs_act_on_the_queue_that_feeds_the_answer() {
        let a = run_args(&[
            "--cap",
            "2",
            "--consumer-delay-ms",
            "200",
            "--policy",
            "block",
            "--block-max-wait-ms",
            "7",
        ]);
        let block = QueuePolicy::Block {
            max_wait: Duration::from_millis(7),
        };
        let asked = |edge| CameraQueue {
            edge,
            cap: 2,
            policy: block,
            delay_ms: 200,
        };

        let on = camera_queues(&a, true);
        assert_eq!(on.camdet(), Some(asked("cam0->camdet")));
        assert_eq!(on.proc(), None, "proc runs beside the detector");
        assert_eq!(on.knobs(), asked("cam0->camdet"));

        // Positive control: the same arguments without a detector reach proc.
        let off = camera_queues(&a, false);
        assert_eq!(off.camdet(), None);
        assert_eq!(off.proc(), Some(asked("cam0->proc")));
        assert_eq!(off.knobs(), asked("cam0->proc"));
    }

    /// Omitted, `--cap` is 1 on the detector's queue and 4 on `proc`'s -- each
    /// what it was before the knobs could reach the detector -- and the policy
    /// is drop-oldest on both.
    #[test]
    fn an_omitted_cap_is_one_on_the_detector_and_four_on_proc() {
        let a = run_args(&[]);
        let on = camera_queues(&a, true);
        assert_eq!(on.knobs().edge, "cam0->camdet");
        assert_eq!((on.knobs().cap, CAMDET_CAP), (1, 1));
        assert_eq!(on.knobs().policy, QueuePolicy::DropOldest);
        assert_eq!(on.knobs().delay_ms, 0);
        assert_eq!(on.proc(), None);
        let off = camera_queues(&a, false);
        assert_eq!(off.knobs().edge, "cam0->proc");
        assert_eq!((off.knobs().cap, PROC_CAP), (4, 4));
        assert_eq!(off.knobs().policy, QueuePolicy::DropOldest);
    }
}
