//! Per-run summary (spec H.2, D19) and the H.3 invariants. Everything here is
//! derived from the evidence rows the `rec` thread returned plus the
//! queue/admission counters in `RunReport`.

use std::collections::BTreeMap;

use pipes_core::evidence::{Evidence, Outcome};
use pipes_core::sample::StreamId;
use pipes_core::stats::{percentile, percentile_sorted};
use pipes_kitti::camdet::{MODEL_FILE, MODEL_NAME};
use pipes_kitti::detect::{
    detect_params_note, MIN_CLUSTER_VOXELS, PERSISTENCE_GATE_M, RANGE_LIMIT_M,
};
use pipes_kitti::fuse::FUSION_RULE;
use pipes_kitti::layout::ABSENT_IN_SOURCE;
use pipes_kitti::state::{CORRIDOR_HALF_WIDTH_M, OBJECT_BYTES};
use pipes_kitti::track::gate_note;
use pipes_kitti::voxel::VOXEL_SIZE_RATIONALE;
use serde::Serialize;

use crate::consumers::{answer_words, Extent, FuseTally, ShapeRow};
use crate::run::{BeforeTrack, RunReport, SourceGapPairs, StorageCheck, VeloStorageCheck};
use crate::viewer::{VIEWER_CAP, VIEWER_EDGE, VIEWER_POLICY};

/// What every `bytes/frame` figure means (spec I.4 wording, verbatim).
/// What one detection IS, in one sentence, recorded beside every count of
/// them.
///
/// It is a constant rather than prose in a doc comment because the number and
/// this sentence have to travel together: 177 reads as "177 objects", and it
/// is 177 connected non-ground structures of which roughly three fifths are
/// pieces of something larger. The `reduce` stage learned the same lesson with
/// `singleton_voxels`, one link up the chain.
/// What one track row is, in the same shape as [`WHAT_A_DETECTION_IS`] and
/// for the same reason: the count above it is not a count of objects, and a
/// reader who assumes it is will read the whole chain wrong.
pub const WHAT_A_TRACK_IS: &str = "one connected structure this stage has now matched in at \
     least two sweeps, never more than one sweep apart, carrying a velocity measured over \
     its own lifetime and that lifetime in seconds on the sensor clock, and -- when the \
     fusion associated it with a camera detection of the paired frame -- that detection's \
     class and confidence, for this sweep only -- \
     NOT one object: it inherits every caveat on a detection, plus association errors of \
     its own, and its velocity is in the SENSOR frame, so a parked car reads as moving at \
     the ego vehicle's speed";

/// How the answer's distance is defined, in words, **including the part of it
/// that does not hold**, because this string travels in every run's
/// `summary.json` and is the only account of the number a reader of one run
/// has.
///
/// It used to end "it can be checked by filtering the raw sweep to the
/// corridor, dropping the ground, and taking the smallest forward coordinate".
/// That is the claim `pipes_kitti::state` also made, and it is false: the
/// corridor test and the distance are two independent projections of one
/// axis-aligned box.
pub const DISTANCE_RATIONALE: &str = "lidar origin to the NEAR FACE of the object's box \
     (bbox_min.x), not to its centroid -- so it is the gap. CAVEAT: it is the near face of \
     the BOX, and the corridor test is a SEPARATE projection of that same box, so nothing \
     checks that the material at the near face is the material inside the corridor -- on a \
     wide structure the reported distance can be metres nearer than anything actually in \
     the vehicle's path. It is measured from the LIDAR and not from the bumper: the \
     velodyne sits on the roof, some way back, and that offset is in no calibration file \
     this project reads, so it is not applied rather than guessed";

pub const WHAT_A_DETECTION_IS: &str = "one connected group of non-ground voxels within the      range bound -- NOT one object: about 59% are fragments of a larger structure broken up by      sparse sampling, and a few per sweep are two objects connected components could not separate";

pub const BYTES_ALLOC_SEMANTICS: &str = "bytes requested from the allocator on this thread (allocations + realloc growth), not resident memory";

/// Nearest-rank percentiles in microseconds. `None` means the sample set was
/// empty — nothing was measured — which JSON writes as `null` and CSV as an
/// empty field. It is never a zero, because zero is a measurement this
/// pipeline really does record.
#[derive(Clone, Copy, Debug, Default, Serialize)]
pub struct Pct {
    pub p50: Option<i64>,
    pub p95: Option<i64>,
    pub p99: Option<i64>,
    pub max: Option<i64>,
}

impl Pct {
    /// Percentiles of `v`, which is **sorted in place** (once, for all four).
    pub fn of(v: &mut [i64]) -> Pct {
        v.sort_unstable();
        Pct {
            p50: percentile_sorted(v, 50.0),
            p95: percentile_sorted(v, 95.0),
            p99: percentile_sorted(v, 99.0),
            max: percentile_sorted(v, 100.0),
        }
    }
}

/// One edge: the driver pseudo-edge `cam0` or a consumer edge, with `stage`
/// naming the stage that delivers on it (`driver`, `proc`, `rerun`). The
/// admission drop rows of a consumer edge are counted here too, so
/// `delivered + dropped_* + timeout == admitted` reads off one line.
#[derive(Clone, Debug, Serialize)]
pub struct StageSummary {
    pub edge: &'static str,
    pub stage: &'static str,
    pub delivered: u64,
    pub dropped_oldest: u64,
    pub dropped_newest: u64,
    pub timeout: u64,
    pub missing: u64,
    pub queue_wait_us: Pct,
    pub measurement_age_us: Pct,
    /// D19: over Delivered rows dequeued before the driver's last admission.
    pub measurement_age_steady_us: Pct,
    pub steady_rows: u64,
    pub bytes_alloc_per_frame_mean: f64,
    /// `None` when nothing was ever pushed on this edge.
    pub push_blocked_ns_p99: Option<i64>,
}

impl StageSummary {
    pub fn dropped(&self) -> u64 {
        self.dropped_oldest + self.dropped_newest + self.timeout
    }
}

/// The frozen camera detector's block of `summary.json`; absent unless
/// `--detector on` ran it.
///
/// Its own block, on [`LidarSummary`]'s argument: every run-level figure is
/// the camera's as the 39 committed rows measured it, and the detector is a
/// consumer those rows never had.
#[derive(Clone, Debug, Serialize)]
pub struct CamDetSummary {
    /// Name, file and sha256 of the weights that ran -- the pinned file, or
    /// the run would have refused to start.
    pub model: &'static str,
    pub model_file: &'static str,
    pub model_sha256: String,
    /// What loading and optimising the model cost, before the clock started.
    pub model_load_ms: f64,
    /// Frames routed onto `cam0->camdet`, frames the stage ran on, batches it
    /// admitted, and frames it could not turn into one.
    pub admitted: u64,
    pub delivered: u64,
    pub produced: u64,
    pub errors: u64,
    /// Frames lost at admission because the stage had not finished the one
    /// before: `admitted - delivered`, each with its own drop row.
    pub dropped: u64,
    /// `delivered / admitted`, and that share of the camera's rate as
    /// replayed (`null` unpaced).
    pub delivered_fraction: Option<f64>,
    pub delivered_hz: Option<f64>,
    /// The camera's period as replayed -- the drive's median interval over
    /// `--rate` -- which is the stage's budget; `null` unpaced.
    pub period_ms: Option<f64>,
    /// Per frame, in microseconds: everything the stage did (`proc_end -
    /// proc_start`), the letterbox and tensor, and the network with its decode
    /// and NMS. Measured on a host that was not checked quiet.
    pub service_us: Pct,
    pub preprocess_us: Pct,
    pub infer_us: Pct,
    /// Frames whose service took longer than the period; `null` unpaced.
    pub frames_over_period: Option<u64>,
    pub detections: u64,
    pub detections_per_frame: Option<f64>,
    /// Detections by class name, over every batch produced.
    pub classes: BTreeMap<&'static str, u64>,
    /// Allocated looking at each frame (resize, tensor, network), of which
    /// the resize and the tensor; allocated building each batch; and the
    /// Arrow bytes carried in (the frame, read in place) and out.
    pub frame_bytes_per_frame: u64,
    pub preprocess_bytes_per_frame: u64,
    pub build_bytes_per_batch: u64,
    pub in_payload_bytes_per_frame: u64,
    pub out_payload_bytes_per_batch: Option<u64>,
    /// The frame was read where the camera driver put it.
    pub input_storage_id_equal: bool,
    /// `cam_det.arrows` under the run directory: every batch, in order, and
    /// a stream with no batch in it when the stage produced none.
    pub batches_file: Option<String>,
}

/// The lidar half of `summary.json`; absent entirely on a camera-only run.
///
/// A separate block rather than more fields on [`RunSummary`], because every
/// run-level figure up there -- `n_frames`, `admitted`, `missing`,
/// `pacing_us`, `decode_ms_p50` -- is the CAMERA's, and
/// was measured as the camera's for all 39 committed rows. Widening them to
/// "the run's" would change what those rows assert without changing a single
/// value in the file.
#[derive(Clone, Debug, Serialize)]
pub struct LidarSummary {
    /// Sweeps the drive holds (`VeloDriver::len`).
    pub n_sweeps: usize,
    /// Frame slots the replay accounted for: the sweeps, and the frames whose
    /// sweep is absent in the source. `admitted + missing == frames`.
    pub frames: usize,
    /// The frames with no sweep in the source, ascending: `[177, 178, 179,
    /// 180]` on drive 0009, empty on a drive with a sweep for every frame.
    /// Each is a `Missing` row on `velo` with reason `absent_in_source`, and
    /// is counted in `missing`.
    pub absent_in_source: Vec<u64>,
    /// Sweeps the velodyne driver admitted, and the frame slots it never
    /// produced -- skipped, unreadable, or absent in the source.
    pub admitted: u64,
    pub missing: u64,
    /// Sweeps that reached the `cloud` consumer.
    pub delivered: u64,
    /// KITTI `.bin` files are not all the same length, so these three are a
    /// real spread rather than one number repeated.
    pub points_min: Option<u32>,
    pub points_mean: Option<u64>,
    pub points_max: Option<u32>,
    /// Axis-aligned bounds over every point the consumer saw, in sensor metres.
    pub extent: Option<Extent>,
    /// **Carried**: Arrow bytes crossing `velo->cloud`, summed over delivered
    /// sweeps. The unrounded figure, because `payload_bytes_per_sweep` is an
    /// integer mean and a reader checking it against the sweeps that were
    /// actually delivered needs the total to do the arithmetic exactly.
    pub payload_bytes_total: u64,
    /// **Carried**: Arrow bytes crossing `velo->cloud` per delivered sweep.
    pub payload_bytes_per_sweep: Option<u64>,
    /// **Allocated**: bytes the velodyne replay thread requested per admitted
    /// sweep -- the read plus the Arrow wrap.
    pub driver_bytes_per_sweep: u64,
    /// **Allocated**: bytes the `cloud` stage requested per delivered sweep.
    /// Meant to be 0; it reads the shared point buffer in place.
    pub cloud_bytes_per_sweep: u64,
    /// Sweeps whose `point_count` column disagreed with the payload length.
    pub point_count_mismatch: u64,
    /// Mean `tov_trigger_ns - tov.start()` over delivered sweeps: how far into
    /// its rotation the head faced forward. ~51.6 ms on drive_0005, and the
    /// offset a fusion stage has to reconcile against a camera instant.
    pub trigger_offset_mean_ns: Option<i64>,
    /// Sweeps whose payload trigger fell outside the envelope's own range, or
    /// was absent. Both are 0 on a healthy run, and both are the kind of
    /// disagreement that no downstream timing check could find on its own.
    pub trigger_outside_range: u64,
    pub trigger_missing: u64,
    /// The lidar's own 2-stage `storage_id` proof. **False when nothing was
    /// compared**, which is a distinct state from a mismatch and is not a
    /// pass -- see `VeloStorageCheck::NotChecked`.
    pub storage_id_equal_all_stages: bool,
    /// The velodyne driver's pacing error, `arrival - due`, in µs. Its own
    /// figure: a sweep is paced to the END of its rotation, so it is not
    /// comparable with the camera's and must not be averaged into it.
    pub pacing_us: Pct,
    /// The derived chain hanging off these sweeps; absent under `--reduce off`.
    pub reduce: Option<ReduceSummary>,
}

/// The stage-to-stage hand-off, as `summary.json` records it.
///
/// This block is the deliverable of M12, and it is laid out so the byte chain
/// can be read straight down it. Two distinctions are kept sharp because the
/// project has confused them before and the whole claim turns on them:
///
/// * **carried** (`*_payload_bytes_*`) is what an edge TRANSFERS;
///   **allocated** (`*_bytes_*`) is what a stage had to REQUEST. Zero copy is
///   carried-large-with-allocated-zero; a transform is carried-falling with
///   allocated-non-zero exactly once, in the stage that transforms.
/// * **the input was not copied** (`read_bytes_per_sweep`, which is 0) is not
///   the same claim as **nothing was allocated** (`build_bytes_per_sweep`,
///   which is not 0 and must not be — producing a new Arrow result is the
///   correct behaviour for a stage that transforms data).
#[derive(Clone, Debug, Serialize)]
pub struct ReduceSummary {
    /// The grid edge, in metres. Recorded because every ratio below is
    /// meaningless without it.
    pub voxel_size_m: f32,
    /// Where that number comes from, in words, so a reader of the file never
    /// has to take it on trust or go looking for the commit that chose it.
    pub voxel_size_rationale: &'static str,
    /// Sweeps `reduce` consumed, clouds it produced, and sweeps whose result
    /// could not be built. `produced + errors == delivered`.
    pub delivered: u64,
    pub produced: u64,
    pub errors: u64,
    /// Derived clouds that reached the consumer at the far end.
    pub down_delivered: u64,
    /// **Carried**, per sample, at each link of the chain. The first is the
    /// raw sweep on `velo->reduce`; the second is the reduced cloud on
    /// `det->cloud`. These are the two numbers the work exists to put beside
    /// each other.
    pub in_payload_bytes_per_sweep: Option<u64>,
    pub out_payload_bytes_per_sample: Option<u64>,
    /// Unrounded totals, so a reader can redo the division exactly against the
    /// populations above rather than against a rounded mean.
    pub in_payload_bytes_total: u64,
    pub out_payload_bytes_total: u64,
    /// Carried in divided by carried out: the headline shrink, and the number
    /// that must be reported as measured rather than as hoped for.
    pub payload_shrink: Option<f64>,
    /// Points in and out per sample, and their ratio.
    pub in_points_per_sweep: Option<u64>,
    pub out_points_per_sample: Option<u64>,
    pub point_shrink: Option<f64>,
    /// **Allocated** by `reduce` per sweep while READING the input. 0: the
    /// sweep is borrowed in place.
    pub read_bytes_per_sweep: u64,
    /// **Allocated** by `reduce` per cloud while BUILDING the output. One
    /// buffer, and non-zero by design.
    pub build_bytes_per_sample: u64,
    /// **Allocated** by the consumer at the far end, per sample. 0, for the
    /// same reason the raw cloud's consumer reports 0.
    pub down_bytes_per_sample: u64,
    /// velodyne driver -> `reduce`: the transform read the driver's own
    /// buffer. **False when nothing was compared**, which is not a pass.
    pub input_storage_id_equal: bool,
    /// `reduce` -> the far consumer: the derived buffer was shared in turn.
    pub output_storage_id_equal: bool,
    /// Derived samples that failed to name the sweep they came from. 0, or
    /// the chain has produced a result nothing can trace to a measurement.
    pub parent_mismatch: u64,
    /// Whether every derived cloud that reached the far end named the sweep
    /// `reduce` actually built it from — the seq half of `parent`, which
    /// `parent_mismatch` above does not cover and nothing read until now.
    pub parent_seq_equal: bool,
    /// Output rows whose voxel held exactly one point, and the share of all
    /// output rows that is. The caveat that belongs beside the ratio: for
    /// those rows the centroid averages nothing and copies one input point.
    pub singleton_voxels: u64,
    pub singleton_fraction: Option<f64>,
    /// Points in the fullest voxel of the run — the bound under which the f64
    /// accumulation argument holds.
    pub max_occupancy: u32,
    /// Points discarded as non-finite, or as outside the packed grid. Both 0
    /// on real data; counted rather than silently folded into voxel (0, 0, 0).
    pub non_finite_points: u64,
    pub out_of_range_points: u64,
    /// Those two as a share of the points that came in.
    ///
    /// Kept beside `payload_shrink` because a grid that DISCARDS points and
    /// one that MERGES them report the same ratio, and only this number tells
    /// them apart. 0.0 on every run on a real drive measured so far.
    pub discarded_points_fraction: Option<f64>,
    /// How many times the reusable scratch grew, and how big it ended up.
    ///
    /// These are the honest caveat on `read_bytes_per_sweep = 0`. That 0 says
    /// the stage allocates nothing PER SWEEP; what it excludes is one scratch
    /// buffer, grown outside the measured window, which on a real drive is
    /// about the size of a single sweep. Far too large to leave implied.
    pub scratch_growths: u64,
    pub scratch_bytes: u64,
    /// The third link hanging off these clouds; absent under `--detect off`.
    pub detect: Option<DetectSummary>,
}

/// The `detect` stage, as `summary.json` records it.
///
/// The block is laid out so that **no number can be read without its caveat**.
/// `detections_per_sweep` is immediately followed by `what_a_detection_is` and
/// by the two counts that qualify it, because a detection count reads as a
/// count of objects and is not one; `ground_plane_fallbacks` sits next to
/// `ground_removed_fraction` because a fallback plane is the fixed-height
/// method the stage's docs argue against, and a run that used one has to say
/// so rather than publish a fraction that looks fitted.
#[derive(Clone, Debug, Serialize)]
pub struct DetectSummary {
    /// The bound past which the measured return spacing exceeds one voxel
    /// edge, so the adjacency this stage tests is not present in the data.
    /// A validity bound, not a crop; recorded because the count below means
    /// nothing without it.
    pub range_limit_m: f32,
    /// Smallest cluster reported, in voxels.
    pub min_cluster_voxels: u32,
    /// Where all three of this stage's numbers come from, in words, so a
    /// reader of the file never has to take them on trust.
    pub params_rationale: String,
    /// Clouds `detect` consumed, batches it produced, clouds whose detections
    /// could not be built. `produced + errors == delivered`.
    pub delivered: u64,
    pub produced: u64,
    pub errors: u64,
    /// Detection batches that reached the consumer at the far end.
    pub down_delivered: u64,
    /// **Carried**, per sample, at each link: the reduced cloud on
    /// `det->detect`, then the detections on `obj->sink`.
    pub in_payload_bytes_per_sample: Option<u64>,
    pub out_payload_bytes_per_sample: Option<u64>,
    /// Unrounded totals, so a reader can redo the division exactly.
    pub in_payload_bytes_total: u64,
    pub out_payload_bytes_total: u64,
    /// Carried in divided by carried out: this link's shrink.
    pub payload_shrink: Option<f64>,
    /// Voxels in and detections out per sample, and their ratio.
    pub in_voxels_per_sample: Option<u64>,
    pub detections_per_sweep: Option<u64>,
    pub voxel_shrink: Option<f64>,
    /// **What one of those actually is.** Not decoration: without it the
    /// number above reads as a count of objects, and about three fifths of
    /// them are pieces of something larger.
    pub what_a_detection_is: &'static str,
    /// Detections too small to be any KITTI object class, and the share of all
    /// detections that is.
    pub fragment_detections: u64,
    pub fragment_fraction: Option<f64>,
    /// Detections spanning more than any single KITTI object class: two things
    /// connected components could not separate.
    pub merged_detections: u64,
    /// Raw returns per detection over the WHOLE chain, computed at the far end
    /// from the columns the detection batch carries rather than from anything
    /// the producing stage remembered.
    pub chain_returns_per_detection: Option<f64>,
    /// **Allocated** by `detect` per cloud while READING the input. 0: the
    /// cloud is borrowed in place.
    pub read_bytes_per_sample: u64,
    /// **Allocated** by `detect` per batch while BUILDING the output. One
    /// buffer, and non-zero by design.
    pub build_bytes_per_sample: u64,
    /// **Allocated** by the consumer at the far end, per sample. 0.
    pub down_bytes_per_sample: u64,
    /// `reduce` -> `detect`: the transform read `reduce`'s own buffer.
    /// **False when nothing was compared**, which is not a pass.
    pub input_storage_id_equal: bool,
    /// `detect` -> the far consumer: the detection buffer was shared in turn.
    pub output_storage_id_equal: bool,
    /// Detection batches that failed to name the cloud they came from.
    pub parent_mismatch: u64,
    /// Whether every batch that reached the far end named the cloud `detect`
    /// actually built it from — the seq half of `parent`.
    pub parent_seq_equal: bool,
    /// In-range voxels, and the share of them the ground plane removed.
    pub in_range_voxels: u64,
    pub ground_voxels: u64,
    pub ground_removed_fraction: Option<f64>,
    /// Sweeps whose plane was fitted, and sweeps where the fit was REFUSED and
    /// fell back to a fixed height. The second is the honest one: a fallback
    /// is the fixed-z method, so a non-zero count means part of this run did
    /// the thing the design rejected.
    pub ground_plane_fits: u64,
    pub ground_plane_fallbacks: u64,
    /// Mean and maximum fitted tilt, in degrees. The numbers the flat-world
    /// assumption is read against: one plane is defensible at 2 degrees and
    /// collapses on a real hill.
    pub ground_tilt_deg_mean: Option<f64>,
    pub ground_tilt_deg_max: f64,
    /// Connected components before the minimum-size gate.
    pub clusters: u64,
    /// Voxels discarded as non-finite, as outside the packed grid, and voxels
    /// whose recovered grid index collided with another's. All three 0 on real
    /// data; the third is the checkable half of "the grid is recoverable".
    pub non_finite_voxels: u64,
    pub out_of_range_voxels: u64,
    pub key_collisions: u64,
    /// The share of detections that reappeared in the next sweep, and the
    /// control it is read against.
    ///
    /// **Uncompensated**, and `persistence_is_compensated` says so. Correcting
    /// for the vehicle's own motion needs the OXTS stream, which this pipeline
    /// does not replay; uncompensated is the weaker of the two figures
    /// measured (35.9 % against 71.8 %) and is still about 20x its control.
    /// The control is the same detections turned a quarter turn, which has
    /// **identical** spatial statistics — so it cannot be explained away by
    /// density, which is what makes it the sharper of the two controls.
    pub persistence_fraction: Option<f64>,
    pub persistence_control_fraction: Option<f64>,
    pub persistence_samples: u32,
    /// Pairs of CONSECUTIVE sweeps the check ran over. The number that says
    /// whether the fraction above is worth reading: a Drop* edge upstream can
    /// leave an unpaced run with almost no adjacent pairs.
    pub persistence_sweep_pairs: u64,
    pub persistence_gate_m: f32,
    pub persistence_is_compensated: bool,
    /// How many times the reusable scratch grew, and how big it ended up: the
    /// honest caveat on `read_bytes_per_sample = 0`, which says the stage
    /// allocates nothing PER CLOUD and not that it allocates nothing.
    pub scratch_growths: u64,
    pub scratch_bytes: u64,
    /// The last two links, when `--track on` ran them.
    pub track: Option<TrackSummary>,
}

/// The fusion and the answer: the two links that end the chain.
#[derive(Clone, Debug, Serialize)]
pub struct TrackSummary {
    /// Detection batches `track` consumed, fused samples it produced, sweeps
    /// it refused to fuse, and batches that could not be built.
    /// `produced + expired + errors == delivered`.
    pub delivered: u64,
    pub produced: u64,
    pub expired: u64,
    pub errors: u64,
    /// The architecture document's three categories, computed rather than
    /// asserted: `completed` is a pair inside the sweep's range, `degraded` a
    /// declared stale one, `expired` a sweep that produced nothing.
    pub completed: u64,
    pub degraded: u64,
    /// Why each expired set expired. A frame that had not arrived yet
    /// (`late`), one that was produced and thrown away upstream (`dropped`),
    /// and no camera stream at all (`absent`) are three different failures,
    /// and calling the second the first would be a false accusation.
    pub expired_dropped: u64,
    pub expired_late: u64,
    pub expired_absent: u64,
    /// Sweeps whose camera frame is absent in the camera's source, so the
    /// set expired with nothing to pair and nothing coming
    /// (`pair_absent_in_source`): the frames, ascending.
    pub expired_absent_in_source: Vec<u64>,
    /// The instants a gap in either sensor's source left without a partner:
    /// camera frames with no sweep, and sweeps with no camera frame.
    pub source_gaps: SourceGapPairs,
    /// Every sweep of the drive against the fusion: the sweeps above are
    /// only the ones `track` was handed, and this says how many there were
    /// in all and where the rest were lost -- skipped by the driver, evicted
    /// on `velo->reduce`, `det->detect` or `obj->track`, or failed in
    /// `reduce` or `detect`. `completed + degraded + expired + errors +
    /// (sweeps - reached) == sweeps`.
    pub sweeps_before_track: Option<BeforeTrack>,
    /// `cam_tov - tov_trigger_ns` over completed sets: **the check that the
    /// pairing found the right frame rather than merely a frame.** The
    /// timestamps say the camera fires 10.503 ms after the sweep's
    /// forward-facing trigger, sd 0.071 ms, so anything else here is a
    /// mispairing and a mispairing is a whole sweep wrong.
    pub pair_age_ms_mean: Option<f64>,
    pub pair_age_ms_min: Option<f64>,
    pub pair_age_ms_max: Option<f64>,
    /// The same camera-instant-minus-trigger over **degraded** sets, or
    /// `null` when there were none: how stale the camera half of the stale
    /// answers was, negative because the frame came before the trigger
    /// (about -92.8 ms for the previous frame on drive_0005). Apart from the
    /// three above, which cover completed sets only and read `null` on a run
    /// whose every set was degraded.
    pub degraded_pair_age_ms_mean: Option<f64>,
    pub degraded_pair_age_ms_min: Option<f64>,
    pub degraded_pair_age_ms_max: Option<f64>,
    /// Milliseconds per sweep spent waiting for a camera frame. Inside the
    /// measured window, so it is already in `measurement_age_ns`.
    pub pair_wait_ms_per_sweep: f64,
    /// Camera references popped, and any whose payload was not one.
    pub cam_delivered: u64,
    pub cam_bad_format: u64,
    /// **Carried**, per sample, at each link: the detections on `obj->track`,
    /// then the tracks on `track->state`.
    pub in_payload_bytes_per_sample: Option<u64>,
    pub out_payload_bytes_per_sample: Option<u64>,
    /// How much smaller the payload got here.
    ///
    /// **Barely above 1, and that is the finding.** Every earlier link shrank
    /// the payload because it threw data away; this one adds a velocity, an
    /// age and an image rectangle that no detection had. The chain stops
    /// compressing here and starts inferring, and a schema trimmed to make
    /// this number look better would be exactly the tuning this project
    /// forbids.
    pub payload_shrink: Option<f64>,
    /// Detections in, tracks out, per sample.
    pub in_detections_per_sample: Option<u64>,
    pub tracks_per_sample: Option<u64>,
    /// What a track record contains, in words, so the ratio above is readable.
    pub what_a_track_is: &'static str,
    /// The association gate on the last paced pair, and the interval it was
    /// computed from. The gate is a FUNCTION of the interval, so a dropped
    /// sweep widens it rather than silently retuning the tracker.
    pub gate_m: f64,
    pub gate_dt_s: f64,
    /// Where that gate comes from, term by term, including the quarter of it
    /// that is ego motion the pipeline cannot subtract.
    pub gate_rationale: String,
    /// Detections that continued an existing track, over detections seen.
    ///
    /// **Read this next to `ambiguous_fraction`.** A wide gate raises the
    /// first and the second at the same time, and the second is the price.
    pub association_rate: Option<f64>,
    /// Detections that had MORE THAN ONE live track inside the gate: how often
    /// "nearest" was a coin toss.
    pub ambiguous_fraction: Option<f64>,
    /// Tracks born, died and coasted over the run, and ids issued in total.
    pub tracks_born: u64,
    pub tracks_died: u64,
    pub tracks_coasted: u64,
    pub ids_issued: u64,
    /// Sweeps that were not `previous + 1`, so every track was discarded.
    /// **The number that attributes a collapsed tracker to an upstream
    /// eviction, or to a gap in the source (`lidar.absent_in_source`),
    /// rather than to the algorithm.**
    pub tracks_reset: u64,
    /// Mean and maximum age of the emitted tracks, in SECONDS on the sensor
    /// clock: last seen minus first seen, trigger to trigger. The velocity's
    /// quantisation noise is `0.20 m / age_s`, so this is what says how good
    /// a closing speed can be.
    ///
    /// Named with the unit because `track_age_mean` in earlier runs' files
    /// was a count of observations, and a key that changed its unit under
    /// the same name would be read against the old files without complaint.
    pub track_age_s_mean: Option<f64>,
    pub track_age_s_max: Option<f64>,
    /// Mean and maximum number of sweeps an emitted track was matched in:
    /// what the emission rule counts, and what `track_age_*` used to be. It
    /// understates a track's life by every sweep it coasted through.
    pub track_observations_mean: Option<f64>,
    pub track_observations_max: u32,
    /// Emitted tracks that were coasting -- not seen in the sweep they were
    /// emitted with -- over all emitted tracks.
    pub tracks_unseen_fraction: Option<f64>,
    /// Emitted tracks that landed inside the camera frame: what the
    /// projection actually contributed. The camera sees 81.4 deg of the
    /// velodyne's 360, so a number near a quarter is the geometry rather than
    /// a failure.
    pub tracks_in_frame_fraction: Option<f64>,
    /// Whether the run had a calibration at all.
    pub calib_loaded: bool,
    /// Bytes `track` requested per sample, split the way every stage in this
    /// chain splits it: borrowing the inputs against building the output.
    pub read_bytes_per_sample: u64,
    pub build_bytes_per_sample: u64,
    /// Zero-copy at the last three links, and the provenance joins.
    pub input_storage_id_equal: bool,
    pub output_storage_id_equal: bool,
    pub state_storage_id_equal: bool,
    pub parent_seq_equal: bool,
    pub state_parent_seq_equal: bool,
    /// **The camera half of the provenance**, joined against the camera
    /// driver's own rows: every fused sample named a frame that was really
    /// admitted, and whose instant really falls inside that sweep -- or, for
    /// a degraded set, before it and inside the declared stale window.
    pub cam_pair_join_ok: bool,
    /// The tracker's scratch, outside the measured window.
    pub scratch_growths: u64,
    pub scratch_bytes: u64,
    /// What the association did with the camera's detections.
    pub fusion: FusionSummary,
    /// The answer.
    pub state: StateSummary,
}

/// The fusion's association, for `summary.json`: the camera's detections
/// against the lidar's tracks, over completed and degraded sets apart.
#[derive(Clone, Debug, Serialize)]
pub struct FusionSummary {
    /// Whether a detector's output reached the fusion. `false` under
    /// `--detector off`, where every count below is 0 and every track is
    /// unfused.
    pub detector: bool,
    /// The rule, in one line; it has no parameter to record.
    pub rule: Option<&'static str>,
    /// Sets paired inside their sweep.
    pub completed: FuseTallySummary,
    /// Sets paired with an OLDER frame under `--pair-stale-ms`: the
    /// stale-camera experiment's result.
    pub degraded: FuseTallySummary,
    /// Every emitted track of every completed set, filed by what the fusion
    /// made of it and by the shape of its box: does the camera separate real
    /// objects from fragments?
    pub shape_completed: ShapeSummary,
    /// The fused row again, by the class the camera gave it.
    pub fused_by_class: Vec<(&'static str, ShapeRowSummary)>,
    /// Records by population at the far end, and the camera-only detections
    /// the answers carried.
    pub down_fused: u64,
    pub down_lidar_only: u64,
    pub down_unfused: u64,
    pub down_camera_only: u64,
    /// Answers whose `fused_count` disagreed with their records, and sets
    /// whose did in the fusion's own batches. Both meant to be 0.
    pub down_fused_mismatch: u64,
    pub column_mismatch: u64,
    /// The association's scratch, outside both byte windows.
    pub scratch_growths: u64,
    pub scratch_bytes: u64,
    /// `runs/<name>/fused.arrows`: every set, in order, and a stream with no
    /// batch in it when every set expired.
    pub batches_file: Option<String>,
}

/// One population of sets' association, as `summary.json` records it.
#[derive(Clone, Debug, Serialize)]
pub struct FuseTallySummary {
    pub samples: u64,
    pub detections: u64,
    /// Fresh in-frame tracks: the ones that could be fused.
    pub candidates: u64,
    pub fused: u64,
    pub camera_only: u64,
    pub contested: u64,
    pub crowded: u64,
    pub fused_per_sample: Option<f64>,
    /// Fused over candidates: how much of what the lidar holds in view the
    /// camera confirmed.
    pub fused_fraction_of_candidates: Option<f64>,
    /// Fused over detections: how much of what the camera saw the lidar
    /// confirmed.
    pub fused_fraction_of_detections: Option<f64>,
    pub iou_p10: Option<f64>,
    pub iou_p50: Option<f64>,
    pub iou_p90: Option<f64>,
    /// Fused tracks by the class the camera gave them, most first.
    pub classes: Vec<(&'static str, u64)>,
}

impl FuseTallySummary {
    fn of(f: &FuseTally) -> FuseTallySummary {
        FuseTallySummary {
            samples: f.samples,
            detections: f.detections,
            candidates: f.candidates,
            fused: f.fused,
            camera_only: f.camera_only,
            contested: f.contested,
            crowded: f.crowded,
            fused_per_sample: f.fused_per_sample(),
            fused_fraction_of_candidates: f.fused_fraction_of_candidates(),
            fused_fraction_of_detections: f.fused_fraction_of_detections(),
            iou_p10: f.iou_percentile(10.0),
            iou_p50: f.iou_percentile(50.0),
            iou_p90: f.iou_percentile(90.0),
            classes: f.class_counts(),
        }
    }
}

/// One row of the shape table, as `summary.json` records it.
#[derive(Clone, Copy, Debug, Serialize)]
pub struct ShapeRowSummary {
    pub tracks: u64,
    pub fragment: u64,
    pub object: u64,
    pub merged: u64,
    pub fragment_fraction: Option<f64>,
    pub age_s_mean: Option<f64>,
    /// Mean box extent along x, y and z, metres.
    pub extent_m_mean: Option<[f64; 3]>,
}

impl ShapeRowSummary {
    fn of(r: &ShapeRow) -> ShapeRowSummary {
        ShapeRowSummary {
            tracks: r.tracks,
            fragment: r.fragment,
            object: r.object,
            merged: r.merged,
            fragment_fraction: r.fragment_fraction(),
            age_s_mean: r.age_s_mean(),
            extent_m_mean: r.extent_m_mean(),
        }
    }
}

/// The shape table's five rows. See `consumers::ShapeTable`.
#[derive(Clone, Copy, Debug, Serialize)]
pub struct ShapeSummary {
    pub fused: ShapeRowSummary,
    pub inside_fused: ShapeRowSummary,
    pub in_frame_alone: ShapeRowSummary,
    pub coasted_in_frame: ShapeRowSummary,
    pub out_of_frame: ShapeRowSummary,
}

/// The end of the chain: the answer, and what it cost.
#[derive(Clone, Debug, Serialize)]
pub struct StateSummary {
    /// Fused samples `state` consumed and answers it produced.
    pub delivered: u64,
    pub produced: u64,
    pub errors: u64,
    /// Answers that reached the far end.
    pub down_delivered: u64,
    /// **Carried**, per sample: the tracks on `track->state`, then the answer
    /// on `state->sink`.
    pub in_payload_bytes_per_sample: Option<u64>,
    pub out_payload_bytes_per_sample: Option<u64>,
    pub payload_shrink: Option<f64>,
    /// Bytes of one answer record. The answer is one record per track --
    /// every track the fusion emitted, in the camera frame or not -- so the
    /// sample is `objects_per_answer` of these plus a fixed provenance part.
    pub object_bytes: usize,
    /// Records per answer at the far end, and how many of them were in the
    /// camera frame. Out-of-frame records are carried and flagged, not
    /// dropped, so the first is every track and the second a subset.
    pub objects_per_answer: Option<f64>,
    pub in_frame_per_answer: Option<f64>,
    /// **Raw laser returns per answer**: the whole chain, recomputed at the
    /// far end from the columns the batch carries rather than from anything
    /// the process remembered. Per answer SAMPLE -- which carries every
    /// track -- and, beside it, per record.
    pub chain_returns_per_answer: Option<f64>,
    pub chain_returns_per_object: Option<f64>,
    /// What the far end read off the records themselves: records flagged
    /// `nearest_in_path`; answers whose `has_object` column is set; answers
    /// whose flags disagree with that column (one flag when set, none when
    /// not); answers whose record count disagrees with their `source_tracks`;
    /// and records whose redundant lanes disagree. `down_flagged ==
    /// down_with_object` with the last three 0 is the run's
    /// `INVARIANT answer` line.
    pub down_flagged: u64,
    pub down_with_object: u64,
    pub down_flag_mismatch: u64,
    pub down_count_mismatch: u64,
    pub down_inconsistent_records: u64,
    /// Sweeps whose answer was "something is in the vehicle's path", and the
    /// share of sweeps that was.
    pub with_object: u64,
    pub with_object_fraction: Option<f64>,
    /// Of those, how many were also visible to the camera.
    pub with_image: u64,
    /// Distance to the nearest object in the path, metres: mean, and the
    /// extremes so a mean cannot hide them.
    pub distance_m_mean: Option<f64>,
    pub distance_m_min: Option<f32>,
    pub distance_m_max: Option<f32>,
    /// Closing speed, m/s. **Uncompensated and correctly so**: the question is
    /// how fast the gap is shrinking, and a parked car ahead of a moving
    /// vehicle is closing at the vehicle's speed.
    pub closing_mps_mean: Option<f64>,
    /// The most urgent thing the run had to say, and the sweep it said it on.
    pub min_time_to_contact_s: Option<f32>,
    pub min_time_to_contact_sweep: Option<i64>,
    /// That answer in words, led by how stale the camera was and which frame
    /// it was when its half was stale (`STALE camera 93 ms (frame 126): ...`).
    pub most_urgent: Option<String>,
    /// The corridor the answer is about: half the width of the recording
    /// vehicle, so "in the path" means the space the car would drive through.
    pub corridor_half_width_m: f32,
    /// How the distance is defined, in words, because it is the half that has
    /// to be checkable against the raw cloud.
    pub distance_rationale: &'static str,
    /// Bytes `state` requested per sample.
    pub read_bytes_per_sample: u64,
    pub build_bytes_per_sample: u64,
    /// The far end: allocated nothing, and the answers it saw were paired.
    pub down_bytes_per_sample: u64,
    pub down_paired: u64,
    /// Answers the far end saw that were built from a **declared stale**
    /// camera frame.
    ///
    /// Beside `down_paired` rather than derived from it, because they are
    /// different claims: `down_paired < down_delivered` is also what a run
    /// whose `pair_outcome` column was never written would report, while a
    /// non-zero figure here is only producible by a stale pair that kept its
    /// label through `state`'s rebuild of the batch.
    pub down_stale: u64,
}

/// The viewer's own edge, `draw->viewer`: every picture the run drew, and
/// what became of it. Present exactly when the run had a recording.
#[derive(Clone, Debug, Serialize)]
pub struct ViewerSummary {
    pub edge: &'static str,
    /// In drawings: one stage's picture of one sample.
    pub cap: usize,
    pub policy: &'static str,
    /// Drawings pushed.
    pub admitted: u64,
    /// Drawings the Rerun SDK took.
    pub delivered: u64,
    /// Drawings a push dropped, by the queue's own count.
    pub dropped: u64,
    /// The dropped drawings by the reason on their rows (`evicted`).
    pub dropped_by_reason: BTreeMap<&'static str, u64>,
    /// `log` calls the SDK refused.
    pub log_errors: u64,
}

/// `runs/<name>/summary.json`.
#[derive(Clone, Debug, Serialize)]
pub struct RunSummary {
    pub run_id: String,
    pub cap: usize,
    pub delay_ms: u64,
    pub policy: &'static str,
    /// The edge the three above describe: `cam0->camdet` when the detector
    /// ran, `cam0->proc` when it did not -- the camera consumer whose output
    /// reached the answer, which is where `--cap`, `--policy` and
    /// `--consumer-delay-ms` act.
    pub camera_queue: &'static str,
    pub rerun_mode: &'static str,
    pub reuse_output: bool,
    /// The camera's frame slots: PNGs on disk and frames absent in the source.
    pub n_frames: usize,
    /// The frames, ascending, whose PNG is absent in the camera's source;
    /// each is a `Missing` row on `cam0` with reason `absent_in_source`,
    /// counted in `missing`. Empty when the camera has a PNG for every frame.
    pub absent_in_source: Vec<u64>,
    pub admitted: u64,
    pub missing: u64,
    pub wall_s: f64,
    pub pacing_us: Pct,
    pub abs_pacing_us: Pct,
    /// `None` when no frame was admitted, so no decode was ever timed.
    pub decode_ms_p50: Option<f64>,
    pub stages: Vec<StageSummary>,
    /// Present exactly when the frozen camera detector ran.
    pub camera_detector: Option<CamDetSummary>,
    /// Present exactly when the run replayed the drive's lidar.
    pub lidar: Option<LidarSummary>,
    pub storage_id_equal_all_stages: bool,
    /// Present exactly when the run had a recording to draw on.
    pub viewer: Option<ViewerSummary>,
    pub evidence_lost: u64,
    pub recorder_degraded: bool,
    /// Resident set at the end of the run (`tasklist` on Windows), the only resident-memory figure.
    pub rss_bytes: Option<u64>,
    pub invariants_ok: bool,
    pub bytes_alloc_semantics: &'static str,
}

/// One H.3 line, printed as `<text> -> OK|FAIL`.
pub struct Invariant {
    pub text: String,
    pub ok: bool,
}

/// Sort order of the `stages` array: down the pipeline, producer before its
/// consumers, and the derived stream after the sensor it derives from.
fn edge_rank(edge: &str) -> u8 {
    match edge {
        "cam0" => 0,
        "cam0->proc" => 1,
        "cam0->rerun" => 2,
        "cam0->camdet" => 3,
        "cam_det" => 4,
        "velo" => 5,
        "velo->cloud" => 6,
        "velo->reduce" => 7,
        "det" => 8,
        "det->cloud" => 9,
        "det->detect" => 10,
        "obj" => 11,
        "obj->sink" => 12,
        "obj->track" => 13,
        "cam_det->track" => 14,
        "track" => 15,
        "track->state" => 16,
        "state" => 17,
        "state->sink" => 18,
        _ => 19,
    }
}

/// The driver pseudo-edge a consumer edge hangs off: `cam0->proc` -> `cam0`,
/// `velo->cloud` -> `velo`. A driver pseudo-edge is its own producer.
///
/// Derived from the name rather than looked up in a table, so it cannot fall
/// out of date when an edge is added. The thing it decides is not cosmetic:
/// `measurement_age_steady_us` is taken over rows dequeued before **the
/// producer's** last admission, and using the camera's cutoff for a lidar
/// edge would trim the lidar's steady window with a clock it has nothing to
/// do with.
fn producer_of(edge: &str) -> &str {
    edge.split_once("->").map_or(edge, |(p, _)| p)
}

fn stage_of(rows: &[&Evidence]) -> &'static str {
    rows.iter()
        .find(|r| r.outcome == Outcome::Delivered)
        .or(rows.first())
        .map_or("", |r| r.stage)
}

fn stage_summary(
    edge: &'static str,
    rows: &[&Evidence],
    last_arrival_ns: Option<i64>,
    blocked_ns: &[i64],
) -> StageSummary {
    let count = |o: Outcome| rows.iter().filter(|r| r.outcome == o).count() as u64;
    let delivered: Vec<&&Evidence> = rows
        .iter()
        .filter(|r| r.outcome == Outcome::Delivered)
        .collect();
    let mut wait_us: Vec<i64> = delivered
        .iter()
        .filter_map(|r| r.queue_wait_ns)
        .map(|ns| ns / 1000)
        .collect();
    let mut age_us: Vec<i64> = delivered
        .iter()
        .filter_map(|r| r.measurement_age_ns)
        .map(|ns| ns / 1000)
        .collect();
    let steady: Vec<&&&Evidence> = delivered
        .iter()
        .filter(|r| match (r.dequeued_ns, last_arrival_ns) {
            (Some(d), Some(last)) => d < last,
            _ => false,
        })
        .collect();
    let mut steady_age_us: Vec<i64> = steady
        .iter()
        .filter_map(|r| r.measurement_age_ns)
        .map(|ns| ns / 1000)
        .collect();
    let bytes_mean = if delivered.is_empty() {
        0.0
    } else {
        delivered.iter().map(|r| r.bytes_alloc as f64).sum::<f64>() / delivered.len() as f64
    };
    let mut blocked = blocked_ns.to_vec();
    StageSummary {
        edge,
        stage: stage_of(rows),
        delivered: delivered.len() as u64,
        dropped_oldest: count(Outcome::DroppedOldest),
        dropped_newest: count(Outcome::DroppedNewest),
        timeout: count(Outcome::Timeout),
        missing: count(Outcome::Missing),
        queue_wait_us: Pct::of(&mut wait_us),
        measurement_age_us: Pct::of(&mut age_us),
        measurement_age_steady_us: Pct::of(&mut steady_age_us),
        steady_rows: steady.len() as u64,
        bytes_alloc_per_frame_mean: bytes_mean,
        push_blocked_ns_p99: percentile(&mut blocked, 99.0),
    }
}

/// Builds the summary and evaluates the H.3 invariants.
pub fn summarize(report: &RunReport, rss_bytes: Option<u64>) -> (RunSummary, Vec<Invariant>) {
    let mut by_edge: BTreeMap<&'static str, Vec<&Evidence>> = BTreeMap::new();
    for r in &report.rows {
        by_edge.entry(r.edge).or_default().push(r);
    }
    // Per PRODUCER, not one number for the run. `last_arrival_ns` is the
    // steady-state cutoff every consumer edge is trimmed by, and with two
    // producers a single value means one stream's drain tail is decided by
    // the other stream's clock.
    let last_arrival: BTreeMap<&'static str, i64> = by_edge
        .iter()
        .filter(|(edge, _)| !edge.contains("->"))
        .filter_map(|(edge, rows)| {
            rows.iter()
                .filter(|r| r.outcome == Outcome::Delivered)
                .map(|r| r.arrival_ns)
                .max()
                .map(|m| (*edge, m))
        })
        .collect();
    let driver_rows: Vec<&Evidence> = by_edge.get("cam0").cloned().unwrap_or_default();
    let driver_delivered: Vec<&&Evidence> = driver_rows
        .iter()
        .filter(|r| r.outcome == Outcome::Delivered)
        .collect();

    // cam0-only, and named as such at the call sites: `pacing_us`,
    // `abs_pacing_us` and `decode_ms_p50` in `RunSummary` are the CAMERA's.
    // The lidar's own pacing lives in [`LidarSummary::pacing_us`], because
    // one unlabelled pacing figure over two sensors is a number nobody can
    // read.
    let mut pacing_us: Vec<i64> = driver_delivered
        .iter()
        .filter_map(|r| r.due_ns.map(|d| (r.arrival_ns - d) / 1000))
        .collect();
    let mut abs_pacing_us: Vec<i64> = pacing_us.iter().map(|p| p.abs()).collect();
    let mut decode_ns: Vec<i64> = driver_delivered.iter().map(|r| r.decode_ns).collect();
    let decode_ms_p50 = percentile(&mut decode_ns, 50.0).map(|ns| ns as f64 / 1e6);

    let mut edges: Vec<&'static str> = by_edge.keys().copied().collect();
    edges.sort_by_key(|e| (edge_rank(e), *e));
    let empty: Vec<i64> = Vec::new();
    let stages: Vec<StageSummary> = edges
        .iter()
        .map(|&edge| {
            // By NAME. These used to be `push_blocked_ns(0)` / `(1)` resolved
            // positionally at the call site, which silently reassigns an
            // edge's timings the moment one is inserted anywhere but the end.
            let blocked: &[i64] = report
                .edge(edge)
                .map_or(&empty, |e| e.blocked_ns.as_slice());
            stage_summary(
                edge,
                &by_edge[edge],
                last_arrival.get(producer_of(edge)).copied(),
                blocked,
            )
        })
        .collect();

    let invariants = check_invariants(report, &stages, &by_edge);
    let invariants_ok = invariants.iter().all(|i| i.ok);
    let summary = RunSummary {
        run_id: report.run_id.clone(),
        cap: report.camera.cap,
        delay_ms: report.camera.delay_ms,
        policy: report.camera.policy.name(),
        camera_queue: report.camera.edge,
        rerun_mode: report.rerun_mode,
        reuse_output: report.reuse_output,
        n_frames: report.n_frames,
        absent_in_source: report.cam_absent_in_source.clone(),
        admitted: report.totals.admitted,
        missing: report.totals.missing,
        wall_s: report.totals.wall_ns as f64 / 1e9,
        pacing_us: Pct::of(&mut pacing_us),
        abs_pacing_us: Pct::of(&mut abs_pacing_us),
        decode_ms_p50,
        stages,
        camera_detector: report.camdet.as_ref().map(|c| {
            let r = &c.report;
            let admitted = report.edge("cam0->camdet").map_or(0, |e| e.admitted);
            let us = |v: &[i64]| Pct::of(&mut v.iter().map(|ns| ns / 1000).collect::<Vec<_>>());
            let fraction = (admitted > 0).then(|| r.delivered as f64 / admitted as f64);
            CamDetSummary {
                model: MODEL_NAME,
                model_file: MODEL_FILE,
                model_sha256: c.model_sha256.clone(),
                model_load_ms: c.load_ms,
                admitted,
                delivered: r.delivered,
                produced: r.produced,
                errors: r.errors,
                dropped: report.edge("cam0->camdet").map_or(0, |e| e.queue_dropped),
                delivered_fraction: fraction,
                delivered_hz: match (fraction, c.period_ns) {
                    (Some(f), Some(p)) if p > 0 => Some(f * 1e9 / p as f64),
                    _ => None,
                },
                period_ms: c.period_ns.map(|p| p as f64 / 1e6),
                service_us: us(&r.service_ns),
                preprocess_us: us(&r.preprocess_ns),
                infer_us: us(&r.infer_ns),
                frames_over_period: c.period_ns.map(|p| r.over_period(p) as u64),
                detections: r.detections_total,
                detections_per_frame: r.detections_mean(),
                classes: r.classes().into_iter().collect(),
                frame_bytes_per_frame: r.frame_bytes_total / r.delivered.max(1),
                preprocess_bytes_per_frame: r.preprocess_bytes_total / r.delivered.max(1),
                build_bytes_per_batch: r.build_bytes_total / r.produced.max(1),
                in_payload_bytes_per_frame: r.in_payload_bytes_total / r.delivered.max(1),
                out_payload_bytes_per_batch: r.out_payload_bytes_mean(),
                input_storage_id_equal: c.in_check.passed(),
                batches_file: Some(c.file.display().to_string()),
            }
        }),
        lidar: report.velo.as_ref().map(|v| {
            let mut velo_pacing_us: Vec<i64> = by_edge
                .get("velo")
                .map(|rows| {
                    rows.iter()
                        .filter(|r| r.outcome == Outcome::Delivered)
                        .filter_map(|r| r.due_ns.map(|d| (r.arrival_ns - d) / 1000))
                        .collect()
                })
                .unwrap_or_default();
            LidarSummary {
                n_sweeps: v.n_sweeps,
                frames: v.frames,
                absent_in_source: v.absent_in_source.clone(),
                admitted: v.totals.admitted,
                missing: v.totals.missing,
                delivered: v.cloud.delivered,
                points_min: v.cloud.points_min,
                points_mean: v.cloud.points_mean(),
                points_max: v.cloud.points_max,
                extent: v.cloud.extent,
                payload_bytes_total: v.cloud.payload_bytes_total,
                payload_bytes_per_sweep: v.cloud.payload_bytes_mean(),
                driver_bytes_per_sweep: v.driver_bytes / v.totals.admitted.max(1),
                cloud_bytes_per_sweep: v.cloud.bytes_total / v.cloud.delivered.max(1),
                point_count_mismatch: v.cloud.count_mismatch,
                trigger_offset_mean_ns: v.cloud.trigger_offset_mean_ns(),
                trigger_outside_range: v.cloud.trigger_outside_range,
                trigger_missing: v.cloud.trigger_missing,
                storage_id_equal_all_stages: matches!(v.storage_check, VeloStorageCheck::Ok { .. }),
                pacing_us: Pct::of(&mut velo_pacing_us),
                reduce: v.reduce.as_ref().map(|r| {
                    let rep = &r.report;
                    ReduceSummary {
                        voxel_size_m: r.voxel_size.metres(),
                        voxel_size_rationale: VOXEL_SIZE_RATIONALE,
                        delivered: rep.delivered,
                        produced: rep.produced,
                        errors: rep.errors,
                        down_delivered: r.cloud.delivered,
                        in_payload_bytes_per_sweep: rep.in_payload_bytes_mean(),
                        out_payload_bytes_per_sample: rep.out_payload_bytes_mean(),
                        in_payload_bytes_total: rep.in_payload_bytes_total,
                        out_payload_bytes_total: rep.out_payload_bytes_total,
                        payload_shrink: rep.payload_shrink(),
                        in_points_per_sweep: (rep.delivered > 0)
                            .then(|| rep.in_points_total / rep.delivered),
                        out_points_per_sample: (rep.produced > 0)
                            .then(|| rep.out_points_total / rep.produced),
                        point_shrink: rep.point_shrink(),
                        read_bytes_per_sweep: rep.read_bytes_total / rep.delivered.max(1),
                        build_bytes_per_sample: rep.build_bytes_total / rep.produced.max(1),
                        down_bytes_per_sample: r.cloud.bytes_total / r.cloud.delivered.max(1),
                        input_storage_id_equal: r.in_check.passed(),
                        output_storage_id_equal: r.out_check.passed(),
                        parent_mismatch: r.cloud.parent_mismatch,
                        parent_seq_equal: r.parent_check.passed(),
                        singleton_voxels: rep.singleton_voxels_total,
                        singleton_fraction: rep.singleton_fraction(),
                        max_occupancy: rep.max_occupancy,
                        non_finite_points: rep.non_finite_total,
                        out_of_range_points: rep.out_of_range_total,
                        discarded_points_fraction: rep.discarded_fraction(),
                        scratch_growths: rep.scratch_growths,
                        scratch_bytes: rep.scratch_bytes,
                        detect: r.detect.as_ref().map(|d| {
                            let dr = &d.report;
                            DetectSummary {
                                range_limit_m: RANGE_LIMIT_M,
                                min_cluster_voxels: MIN_CLUSTER_VOXELS,
                                params_rationale: detect_params_note(r.voxel_size),
                                delivered: dr.delivered,
                                produced: dr.produced,
                                errors: dr.errors,
                                down_delivered: d.sink.delivered,
                                in_payload_bytes_per_sample: dr.in_payload_bytes_mean(),
                                out_payload_bytes_per_sample: dr.out_payload_bytes_mean(),
                                in_payload_bytes_total: dr.in_payload_bytes_total,
                                out_payload_bytes_total: dr.out_payload_bytes_total,
                                payload_shrink: dr.payload_shrink(),
                                in_voxels_per_sample: (dr.delivered > 0)
                                    .then(|| dr.in_voxels_total / dr.delivered),
                                detections_per_sweep: dr.detections_mean(),
                                voxel_shrink: dr.voxel_shrink(),
                                what_a_detection_is: WHAT_A_DETECTION_IS,
                                fragment_detections: dr.fragment_detections_total,
                                fragment_fraction: dr.fragment_fraction(),
                                merged_detections: dr.merged_detections_total,
                                chain_returns_per_detection: d.sink.chain_shrink(),
                                read_bytes_per_sample: dr.read_bytes_total / dr.delivered.max(1),
                                build_bytes_per_sample: dr.build_bytes_total / dr.produced.max(1),
                                down_bytes_per_sample: d.sink.bytes_total / d.sink.delivered.max(1),
                                input_storage_id_equal: d.in_check.passed(),
                                output_storage_id_equal: d.out_check.passed(),
                                parent_mismatch: d.sink.parent_mismatch,
                                parent_seq_equal: d.parent_check.passed(),
                                in_range_voxels: dr.in_range_voxels_total,
                                ground_voxels: dr.ground_voxels_total,
                                ground_removed_fraction: dr.ground_fraction(),
                                ground_plane_fits: dr.ground_fitted,
                                ground_plane_fallbacks: dr.ground_fallback,
                                ground_tilt_deg_mean: dr.tilt_deg_mean(),
                                ground_tilt_deg_max: dr.tilt_mdeg_max as f64 / 1000.0,
                                clusters: dr.clusters_total,
                                non_finite_voxels: dr.non_finite_total,
                                out_of_range_voxels: dr.out_of_range_total,
                                key_collisions: dr.key_collisions_total,
                                persistence_fraction: dr.persistence.fraction(),
                                persistence_control_fraction: dr
                                    .persistence_control_turned
                                    .fraction(),
                                persistence_samples: dr.persistence.total,
                                persistence_sweep_pairs: dr.persistence_pairs,
                                persistence_gate_m: PERSISTENCE_GATE_M,
                                // Nothing in this pipeline replays the
                                // vehicle's velocity, so this is false and the
                                // field exists to keep it from being read as
                                // the compensated figure.
                                persistence_is_compensated: false,
                                scratch_growths: dr.scratch_growths,
                                scratch_bytes: dr.scratch_bytes,
                                track: d.track.as_ref().map(|tk| {
                                    let tr = &tk.report;
                                    let st = &tk.state;
                                    TrackSummary {
                                        delivered: tr.delivered,
                                        produced: tr.produced,
                                        expired: tr.expired,
                                        errors: tr.errors,
                                        completed: tr.pair_ok,
                                        degraded: tr.pair_stale,
                                        expired_dropped: tr.pair_dropped,
                                        expired_late: tr.pair_late,
                                        expired_absent: tr.pair_absent,
                                        expired_absent_in_source: tr.pair_absent_in_source.clone(),
                                        source_gaps: tk.gap_pairs.clone(),
                                        sweeps_before_track: report.before_track(),
                                        pair_age_ms_mean: tr
                                            .pair_age_ns_mean()
                                            .map(|ns| ns as f64 / 1e6),
                                        pair_age_ms_min: tr
                                            .pair_age_ns_min
                                            .map(|ns| ns as f64 / 1e6),
                                        pair_age_ms_max: tr
                                            .pair_age_ns_max
                                            .map(|ns| ns as f64 / 1e6),
                                        degraded_pair_age_ms_mean: tr
                                            .stale_age_ns_mean()
                                            .map(|ns| ns as f64 / 1e6),
                                        degraded_pair_age_ms_min: tr
                                            .stale_age_ns_min
                                            .map(|ns| ns as f64 / 1e6),
                                        degraded_pair_age_ms_max: tr
                                            .stale_age_ns_max
                                            .map(|ns| ns as f64 / 1e6),
                                        pair_wait_ms_per_sweep: tr.wait_ns_total as f64
                                            / 1e6
                                            / tr.delivered.max(1) as f64,
                                        cam_delivered: tr.cam_delivered,
                                        cam_bad_format: tr.cam_bad_format,
                                        in_payload_bytes_per_sample: tr.in_payload_bytes_mean(),
                                        out_payload_bytes_per_sample: tr.out_payload_bytes_mean(),
                                        payload_shrink: tr.payload_shrink(),
                                        in_detections_per_sample: (tr.delivered > 0)
                                            .then(|| tr.in_detections_total / tr.delivered),
                                        tracks_per_sample: tr.tracks_mean(),
                                        what_a_track_is: WHAT_A_TRACK_IS,
                                        gate_m: tr.gate_m_last,
                                        gate_dt_s: tr.dt_s_last,
                                        gate_rationale: gate_note(
                                            tr.dt_s_last,
                                            r.voxel_size,
                                            RANGE_LIMIT_M,
                                        ),
                                        association_rate: tr.association_rate(),
                                        ambiguous_fraction: tr.ambiguous_fraction(),
                                        tracks_born: tr.born_total,
                                        tracks_died: tr.died_total,
                                        tracks_coasted: tr.coasted_total,
                                        ids_issued: tr.ids_issued,
                                        tracks_reset: tr.resets,
                                        track_age_s_mean: tr.age_s_mean(),
                                        track_age_s_max: (tr.out_tracks_total > 0)
                                            .then_some(f64::from(tr.age_s_max)),
                                        track_observations_mean: tr.observations_mean(),
                                        track_observations_max: tr.observations_max,
                                        tracks_unseen_fraction: tr.unseen_fraction(),
                                        tracks_in_frame_fraction: tr.in_frame_fraction(),
                                        calib_loaded: tk.calib.is_some(),
                                        read_bytes_per_sample: tr.read_bytes_total
                                            / tr.delivered.max(1),
                                        build_bytes_per_sample: tr.build_bytes_total
                                            / tr.produced.max(1),
                                        input_storage_id_equal: tk.in_check.passed(),
                                        output_storage_id_equal: tk.out_check.passed(),
                                        state_storage_id_equal: tk.state_check.passed(),
                                        parent_seq_equal: tk.parent_check.passed(),
                                        state_parent_seq_equal: tk.state_parent_check.passed(),
                                        cam_pair_join_ok: tk.pair_check.passed(),
                                        scratch_growths: tr.scratch_growths,
                                        scratch_bytes: tr.scratch_bytes,
                                        fusion: FusionSummary {
                                            detector: tk.detector,
                                            rule: tk.detector.then_some(FUSION_RULE),
                                            completed: FuseTallySummary::of(&tr.fuse_completed),
                                            degraded: FuseTallySummary::of(&tr.fuse_degraded),
                                            shape_completed: ShapeSummary {
                                                fused: ShapeRowSummary::of(&tr.shape.fused),
                                                inside_fused: ShapeRowSummary::of(
                                                    &tr.shape.inside_fused,
                                                ),
                                                in_frame_alone: ShapeRowSummary::of(
                                                    &tr.shape.in_frame_alone,
                                                ),
                                                coasted_in_frame: ShapeRowSummary::of(
                                                    &tr.shape.coasted_in_frame,
                                                ),
                                                out_of_frame: ShapeRowSummary::of(
                                                    &tr.shape.out_of_frame,
                                                ),
                                            },
                                            fused_by_class: tr
                                                .shape
                                                .by_class()
                                                .iter()
                                                .map(|(k, r)| (*k, ShapeRowSummary::of(r)))
                                                .collect(),
                                            down_fused: tk.sink.fused_total,
                                            down_lidar_only: tk.sink.lidar_only_total,
                                            down_unfused: tk.sink.unfused_total,
                                            down_camera_only: tk.sink.camera_only_total,
                                            down_fused_mismatch: tk.sink.fused_mismatch,
                                            column_mismatch: tr.fuse_column_mismatch,
                                            scratch_growths: tr.fuse_scratch_growths,
                                            scratch_bytes: tr.fuse_scratch_bytes,
                                            batches_file: Some(tk.fused_file.display().to_string()),
                                        },
                                        state: StateSummary {
                                            delivered: st.delivered,
                                            produced: st.produced,
                                            errors: st.errors,
                                            down_delivered: tk.sink.delivered,
                                            in_payload_bytes_per_sample: st.in_payload_bytes_mean(),
                                            out_payload_bytes_per_sample: st
                                                .out_payload_bytes_mean(),
                                            payload_shrink: st.payload_shrink(),
                                            object_bytes: OBJECT_BYTES,
                                            objects_per_answer: tk.sink.objects_mean(),
                                            in_frame_per_answer: tk.sink.in_frame_mean(),
                                            chain_returns_per_answer: tk.sink.chain_shrink(),
                                            chain_returns_per_object: tk
                                                .sink
                                                .chain_shrink_per_object(),
                                            down_flagged: tk.sink.flagged_total,
                                            down_with_object: tk.sink.with_object,
                                            down_flag_mismatch: tk.sink.flag_mismatch,
                                            down_count_mismatch: tk.sink.count_mismatch,
                                            down_inconsistent_records: tk.sink.inconsistent_total,
                                            with_object: st.with_object,
                                            with_object_fraction: st.with_object_fraction(),
                                            with_image: st.with_image,
                                            distance_m_mean: st.distance_m_mean(),
                                            distance_m_min: st.distance_m_min,
                                            distance_m_max: st.distance_m_max,
                                            closing_mps_mean: st.closing_mps_mean(),
                                            min_time_to_contact_s: st.ttc_s_min,
                                            min_time_to_contact_sweep: st
                                                .most_urgent
                                                .map(|(sweep, _, _)| sweep),
                                            most_urgent: st
                                                .most_urgent
                                                .map(|(_, a, cam)| answer_words(&a, cam)),
                                            corridor_half_width_m: CORRIDOR_HALF_WIDTH_M,
                                            distance_rationale: DISTANCE_RATIONALE,
                                            read_bytes_per_sample: st.read_bytes_total
                                                / st.delivered.max(1),
                                            build_bytes_per_sample: st.build_bytes_total
                                                / st.produced.max(1),
                                            down_bytes_per_sample: tk.sink.bytes_total
                                                / tk.sink.delivered.max(1),
                                            down_paired: tk.sink.paired,
                                            down_stale: tk.sink.stale,
                                        },
                                    }
                                }),
                            }
                        }),
                    }
                }),
            }
        }),
        storage_id_equal_all_stages: matches!(report.storage_check, StorageCheck::Ok { .. }),
        viewer: report.viewer.as_ref().map(|v| ViewerSummary {
            edge: VIEWER_EDGE,
            cap: VIEWER_CAP,
            policy: VIEWER_POLICY.name(),
            admitted: report.edge(VIEWER_EDGE).map_or(0, |e| e.admitted),
            delivered: v.delivered,
            dropped: report.edge(VIEWER_EDGE).map_or(0, |e| e.queue_dropped),
            dropped_by_reason: by_edge
                .get(VIEWER_EDGE)
                .into_iter()
                .flatten()
                .filter(|r| r.outcome != Outcome::Delivered)
                .fold(BTreeMap::new(), |mut m, r| {
                    *m.entry(r.reason).or_default() += 1;
                    m
                }),
            log_errors: v.log_errors,
        }),
        evidence_lost: report.evidence_lost,
        recorder_degraded: report.recorder_degraded,
        rss_bytes,
        invariants_ok,
        bytes_alloc_semantics: BYTES_ALLOC_SEMANTICS,
    };
    (summary, invariants)
}

/// The H.3 lines: one per driver, one reconciling the drivers with admission,
/// then per consumer edge `delivered + dropped == admitted` against the
/// queue's own counter and exactly one row per admitted sample per edge.
///
/// **Every denominator here is per-producer or per-edge, and that is the
/// change a second producer forced.** The global admission counter
/// (`report.admitted`) used to stand in for all three, which is only the same
/// number while one driver exists. With two it counts both streams, so the
/// camera's `cam0->proc delivered + dropped` -- camera frames only -- stops
/// equalling it, and the line would have FAILed on a perfectly healthy run.
/// The structure was right; only the right-hand side was wrong.
fn check_invariants(
    report: &RunReport,
    stages: &[StageSummary],
    by_edge: &BTreeMap<&'static str, Vec<&Evidence>>,
) -> Vec<Invariant> {
    let a = report.totals.admitted;
    let m = report.totals.missing;
    let n = report.n_frames as u64;
    // The camera's line keeps its exact spelling on a drive with a PNG for
    // every frame: it is pinned by the integration tests, quoted in the
    // write-up, and every one of the 39 committed rows was measured with it
    // on stdout. A gap in the camera's source adds its terms, and the same
    // check the lidar's line makes: one `absent_in_source` row per absent
    // frame, at its number, and no other.
    let cam_absent_rows = absent_rows(by_edge, "cam0");
    let cam_absent = report.cam_absent_in_source.len() as u64;
    let on_disk = report.cam_on_disk as u64;
    let mut out = vec![Invariant {
        text: if cam_absent == 0 {
            format!("INVARIANT driver admitted={a} missing={m} n_frames={n}")
        } else {
            format!(
                "INVARIANT driver admitted={a} missing={m} n_frames={n} = on disk {on_disk} + absent_in_source {cam_absent} (rows {})",
                cam_absent_rows.len()
            )
        },
        ok: a + m == n
            && n == on_disk + cam_absent
            && a == report.admitted_of(StreamId::CAM0)
            && cam_absent_rows == report.cam_absent_in_source,
    }];
    // The detector's line, in the derived producers' shape and with their
    // second half: every frame it ran on became a batch or an error, and
    // every batch took exactly one `arrival_seq` on `CAM_DET` -- which it is
    // the only producer of while it runs.
    if let Some(c) = &report.camdet {
        let (d, p, e) = (c.report.delivered, c.report.produced, c.report.errors);
        out.push(Invariant {
            text: format!("INVARIANT stage=camdet delivered={d} produced={p} errors={e}"),
            ok: p + e == d && p == report.admitted_of(StreamId::CAM_DET),
        });
    }
    if let Some(v) = &report.velo {
        let (va, vm, vn) = (v.totals.admitted, v.totals.missing, v.n_sweeps as u64);
        // Every frame slot once: the driver's own count of what it admitted
        // and what it did not must be the drive's sweeps plus the frames
        // absent in its source -- and the evidence must hold exactly one
        // `absent_in_source` row for each of those frames, at its number,
        // and no other. A replay that dropped them silently, or invented one,
        // fails here even when its counters agree with each other.
        let frames = v.frames as u64;
        let absent = v.absent_in_source.len() as u64;
        let absent_rows = absent_rows(by_edge, "velo");
        let ok = va + vm == frames
            && frames == vn + absent
            && va == report.admitted_of(StreamId::LIDAR)
            && absent_rows == v.absent_in_source;
        // A drive with a sweep for every frame keeps the line it always had.
        let text = if absent == 0 {
            format!("INVARIANT driver=velo admitted={va} missing={vm} n_sweeps={vn}")
        } else {
            format!(
                "INVARIANT driver=velo admitted={va} missing={vm} frames={frames} = n_sweeps {vn} + absent_in_source {absent} (rows {})",
                absent_rows.len()
            )
        };
        out.push(Invariant { text, ok });
        // The derived producer's line, in the drivers' shape because it IS a
        // producer — it just reads a queue instead of a disk. Every sweep it
        // was handed either became a cloud or became an error, and every cloud
        // it produced took exactly one `arrival_seq` on the derived stream.
        //
        // The second half is the one that bites. Without it the line would
        // still hold if the whole derived stream never reached admission at
        // all: `reduce` would report what it built, nothing would be admitted,
        // and the chain would be a stage talking to itself.
        if let Some(r) = &v.reduce {
            let (d, p, e) = (r.report.delivered, r.report.produced, r.report.errors);
            out.push(Invariant {
                text: format!("INVARIANT stage=reduce delivered={d} produced={p} errors={e}"),
                ok: p + e == d && p == report.admitted_of(StreamId::LIDAR_DET),
            });
            // The same line for the third link, and the same second half: it
            // is what stops the invariant holding for a stage whose output
            // never reached admission at all.
            if let Some(dt) = &r.detect {
                let (d, p, e) = (dt.report.delivered, dt.report.produced, dt.report.errors);
                out.push(Invariant {
                    text: format!("INVARIANT stage=detect delivered={d} produced={p} errors={e}"),
                    ok: p + e == d && p == report.admitted_of(StreamId::LIDAR_OBJ),
                });
                if let Some(tk) = &dt.track {
                    // The fusion's line has a THIRD term the others do not:
                    // `expired`. A stage that can refuse has to balance
                    // `produced + expired + errors == delivered`, or a refusal
                    // and a vanished sample are the same arithmetic.
                    let t = &tk.report;
                    let (d, p, x, e) = (t.delivered, t.produced, t.expired, t.errors);
                    out.push(Invariant {
                        text: format!(
                            "INVARIANT stage=track delivered={d} produced={p} expired={x} errors={e}"
                        ),
                        ok: p + x + e == d && p == report.admitted_of(StreamId::TRACKS),
                    });
                    // The architecture document's own three categories,
                    // computed from the counters rather than asserted: every
                    // fused set is completed, degraded or expired, and nothing
                    // is two of them.
                    let (c, g) = (t.pair_ok, t.pair_stale);
                    out.push(Invariant {
                        text: format!(
                            "INVARIANT stage=track completed={c} degraded={g} expired={x} delivered={d}"
                        ),
                        ok: c + g == p && c + g + x + e == d,
                    });
                    // The association's conservation, from three places: the
                    // paired frames' detection counts, the tracks' population
                    // lanes, and the `camera_only` column. Every detection
                    // went exactly one place, and the `fused_count` columns
                    // agree with the lanes they count -- in the fusion's own
                    // batches and again at the far end.
                    let (fc, fd) = (&t.fuse_completed, &t.fuse_degraded);
                    let (nd, nf, nc) = (
                        fc.detections + fd.detections,
                        fc.fused + fd.fused,
                        fc.camera_only + fd.camera_only,
                    );
                    let (cm, dm) = (t.fuse_column_mismatch, tk.sink.fused_mismatch);
                    out.push(Invariant {
                        text: format!(
                            "INVARIANT fusion detections={nd} fused={nf} camera_only={nc} column_mismatch={cm} down_fused_mismatch={dm}"
                        ),
                        ok: nd == nf + nc && cm == 0 && dm == 0,
                    });
                    let st = &tk.state;
                    let (d, p, e) = (st.delivered, st.produced, st.errors);
                    out.push(Invariant {
                        text: format!(
                            "INVARIANT stage=state delivered={d} produced={p} errors={e}"
                        ),
                        ok: p + e == d && p == report.admitted_of(StreamId::EGO),
                    });
                    // The answer's structure, read off the records at the far
                    // end: one record per track the columns say arrived, the
                    // nearest-in-path flag on exactly one record of every
                    // answer whose `has_object` is set and on none of the
                    // others, and no record whose redundant lanes disagree.
                    // The totals can balance across answers, so both are
                    // also checked per answer and the misses counted.
                    let k = &tk.sink;
                    let (o, t, f, w) = (
                        k.objects_total,
                        k.source_tracks_total,
                        k.flagged_total,
                        k.with_object,
                    );
                    let (fm, cm, i) = (k.flag_mismatch, k.count_mismatch, k.inconsistent_total);
                    out.push(Invariant {
                        text: format!(
                            "INVARIANT answer records={o} tracks={t} flagged={f} with_object={w} flag_mismatch={fm} count_mismatch={cm} inconsistent={i}"
                        ),
                        ok: o == t && f == w && fm == 0 && cm == 0 && i == 0,
                    });
                }
            }
        }
    }
    // The line that ties the producers to admission: every sample a driver
    // admitted took exactly one `arrival_seq`, and no `arrival_seq` went to
    // anything else. With one producer it restates the line above; with two
    // it is the only place the global counter is checked at all, and it is
    // what makes `arrival_seq` a claim rather than a number.
    let total: u64 = report.admitted_by_stream.values().sum();
    out.push(Invariant {
        text: format!(
            "INVARIANT admitted total={total} admission={}",
            report.admitted
        ),
        ok: total == report.admitted,
    });
    for e in &report.edges {
        let (delivered, dropped_rows) = stages
            .iter()
            .find(|s| s.edge == e.name)
            .map_or((0, 0), |s| (s.delivered, s.dropped()));
        let (edge, queue_dropped, admitted) = (e.name, e.queue_dropped, e.admitted);
        out.push(Invariant {
            text: format!(
                "INVARIANT edge={edge} delivered={delivered} dropped={queue_dropped} admitted={admitted}"
            ),
            ok: delivered + dropped_rows == admitted && dropped_rows == queue_dropped,
        });
        let rows = by_edge.get(edge).map_or(0, |v| v.len()) as u64;
        out.push(Invariant {
            text: format!("INVARIANT rows edge={edge} rows={rows} admitted={admitted}"),
            ok: rows == admitted,
        });
    }
    out
}

/// The frames a sensor driver's pseudo-edge (`cam0`, `velo`) recorded as
/// absent in the source, in the order the rows came: its `Missing` rows with
/// reason `absent_in_source`.
fn absent_rows(by_edge: &BTreeMap<&'static str, Vec<&Evidence>>, edge: &str) -> Vec<u64> {
    by_edge.get(edge).map_or_else(Vec::new, |rows| {
        rows.iter()
            .filter(|r| r.outcome == Outcome::Missing && r.reason == ABSENT_IN_SOURCE)
            .map(|r| r.seq)
            .collect()
    })
}

/// Resident set of this process via `tasklist` (Windows only); `None`
/// elsewhere or when the output cannot be parsed (D15).
pub fn rss_bytes() -> Option<u64> {
    if !cfg!(windows) {
        return None;
    }
    let out = std::process::Command::new("tasklist")
        .args([
            "/FI",
            &format!("PID eq {}", std::process::id()),
            "/FO",
            "CSV",
            "/NH",
        ])
        .output()
        .ok()?;
    parse_tasklist_csv(&String::from_utf8_lossy(&out.stdout))
}

/// `"pipes.exe","1234","Console","1","123,456 K"` → `123456 * 1024`. Keeps
/// only the digits of the last field, so locale separators do not matter.
fn parse_tasklist_csv(text: &str) -> Option<u64> {
    let line = text.lines().find(|l| l.trim_start().starts_with('"'))?;
    let last = line.rsplit("\",\"").next()?;
    let digits: String = last.chars().filter(char::is_ascii_digit).collect();
    let kb: u64 = digits.parse().ok()?;
    Some(kb * 1024)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use std::path::PathBuf;
    use std::sync::Arc;

    use arrow::array::RecordBatch;
    use arrow::datatypes::Schema;
    use pipes_core::clock::{HostTime, SensorTime, Tov};
    use pipes_core::evidence::RowCtx;
    use pipes_core::queue::QueuePolicy;
    use pipes_core::sample::{Sample, StreamId};
    use pipes_kitti::cam0::RunTotals;

    use pipes_core::evidence::Evidence;

    use super::*;
    use crate::consumers::{CloudReport, ProcReport};
    use crate::run::{EdgeReport, VeloRun};

    fn sample(seq: u64, arrival: i64) -> Sample {
        Sample {
            stream: StreamId::CAM0,
            seq,
            arrival_seq: seq,
            parent: None,
            tov: Tov::Time(SensorTime(seq as i64 * 100)),
            epoch: 0,
            due: Some(HostTime(arrival - 1_000)),
            arrival: HostTime(arrival),
            payload: RecordBatch::new_empty(Arc::new(Schema::empty())),
            storage_id: 0x10 + seq as usize,
            decode_ns: 5_000_000,
        }
    }

    /// 4 admitted frames; proc delivered 0,1,3 (seq 2 evicted), seq 3 dequeued after the last admission.
    fn report() -> RunReport {
        let ctx = RowCtx {
            run_id: "t".to_string(),
            t0_host: HostTime(0),
            epoch: 0,
        };
        let samples: Vec<Sample> = (0..4)
            .map(|i| sample(i, 100_000 * (i as i64 + 1)))
            .collect();
        let mut rows: Vec<Evidence> = samples
            .iter()
            .map(|s| Evidence::driver_admitted(&ctx, s, "cam0", "driver"))
            .collect();
        let deliver = |s: &Sample, dequeued: i64| {
            Evidence::delivered(
                &ctx,
                s,
                "cam0->proc",
                "proc",
                s.arrival,
                0,
                Some(0),
                HostTime(dequeued),
                HostTime(dequeued + 10),
                HostTime(dequeued + 20_000),
                s.storage_id,
                465_750,
            )
        };
        rows.push(deliver(&samples[0], 150_000));
        rows.push(deliver(&samples[1], 250_000));
        rows.push(Evidence::dropped(
            &ctx,
            &samples[2],
            "cam0->proc",
            "admission",
            Outcome::DroppedOldest,
            "evicted",
            samples[2].arrival,
            1,
            700,
        ));
        rows.push(deliver(&samples[3], 450_000));
        RunReport {
            run_id: "t".to_string(),
            dir: PathBuf::from("runs/t"),
            n_frames: 4,
            cam_on_disk: 4,
            cam_absent_in_source: Vec::new(),
            camera: crate::run::CameraQueue {
                edge: "cam0->proc",
                cap: 1,
                policy: QueuePolicy::DropOldest,
                delay_ms: 0,
            },
            reuse_output: false,
            rerun_mode: "off",
            totals: RunTotals {
                admitted: 4,
                missing: 0,
                wall_ns: 500_000_000,
            },
            admitted: 4,
            admitted_by_stream: BTreeMap::from([(StreamId::CAM0.0, 4)]),
            proc: Some(ProcReport {
                delivered: 3,
                storage_mismatch: 0,
                bytes_total: 3 * 465_750,
                by_seq: vec![(0, 0x10), (1, 0x11), (3, 0x13)],
                cam_refs_produced: 0,
                cam_ref_bytes_total: 0,
                cam_ref_errors: 0,
            }),
            rerun: None,
            camdet: None,
            velo: None,
            edges: vec![EdgeReport {
                name: "cam0->proc",
                admitted: 4,
                queue_dropped: 1,
                blocked_ns: vec![500, 600, 700, 800],
            }],
            sink_blocked_ns: vec![1, 2, 3],
            evidence_lost: 0,
            recorder_degraded: false,
            rows,
            storage_check: StorageCheck::Ok { stages: 2 },
            untracked_bytes: 0,
            stream: None,
            viewer: None,
            recording: crate::run::Recording::Nothing,
        }
    }

    #[test]
    fn summarize_groups_by_edge_and_applies_the_steady_rule() {
        let (s, inv) = summarize(&report(), Some(4096));
        assert_eq!(s.stages.len(), 2);
        assert_eq!((s.stages[0].edge, s.stages[0].stage), ("cam0", "driver"));
        let proc = &s.stages[1];
        assert_eq!((proc.edge, proc.stage), ("cam0->proc", "proc"));
        assert_eq!(
            (proc.delivered, proc.dropped_oldest, proc.dropped()),
            (3, 1, 1)
        );
        // last driver arrival = 400_000; seq 3 was dequeued at 450_000 -> drain tail.
        assert_eq!(proc.steady_rows, 2);
        assert_eq!(
            proc.measurement_age_us.max,
            Some((450_000 + 20_000 - 399_000) / 1000)
        );
        assert_eq!(
            proc.measurement_age_steady_us.max,
            Some((250_000 + 20_000 - 199_000) / 1000)
        );
        assert_eq!(proc.queue_wait_us.p50, Some(50));
        assert_eq!(proc.bytes_alloc_per_frame_mean, 465_750.0);
        assert_eq!(proc.push_blocked_ns_p99, Some(800));
        assert_eq!((s.pacing_us.p50, s.abs_pacing_us.max), (Some(1), Some(1)));
        assert_eq!(s.decode_ms_p50, Some(5.0));
        assert_eq!((s.admitted, s.missing, s.rss_bytes), (4, 0, Some(4096)));
        assert!(s.storage_id_equal_all_stages);
        assert!(s.invariants_ok);
        assert!(s.lidar.is_none(), "a cam0-only run must claim no lidar");
        assert_eq!(inv.len(), 4);
        assert!(inv.iter().all(|i| i.ok));
        assert_eq!(
            inv[0].text,
            "INVARIANT driver admitted=4 missing=0 n_frames=4"
        );
        // The reconciliation line: one producer, so it restates the line
        // above -- and it is the only line that touches the global counter.
        assert_eq!(inv[1].text, "INVARIANT admitted total=4 admission=4");
        assert_eq!(
            inv[2].text,
            "INVARIANT edge=cam0->proc delivered=3 dropped=1 admitted=4"
        );
        assert_eq!(
            inv[3].text,
            "INVARIANT rows edge=cam0->proc rows=4 admitted=4"
        );
        assert_eq!(s.bytes_alloc_semantics, BYTES_ALLOC_SEMANTICS);
    }

    #[test]
    fn invariants_fail_when_a_row_is_missing_or_the_counter_disagrees() {
        let mut r = report();
        r.rows.pop(); // lose seq 3's proc row
        let (s, inv) = summarize(&r, None);
        assert!(!s.invariants_ok);
        assert!(!inv[2].ok && !inv[3].ok);
        let mut r = report();
        r.edges[0].queue_dropped = 2; // queue counter disagrees with the rows
        let (_, inv) = summarize(&r, None);
        assert!(!inv[2].ok && inv[3].ok);
        // And the reconciliation line bites on its own: a per-stream count
        // that does not add up to the global one is a sample that took an
        // `arrival_seq` without being admitted by any driver, or the reverse.
        let mut r = report();
        r.admitted = 5;
        let (_, inv) = summarize(&r, None);
        assert_eq!(inv[1].text, "INVARIANT admitted total=4 admission=5");
        assert!(!inv[1].ok);
    }

    /// A sweep sample, so a two-producer report can be built without a
    /// velodyne file. `arrival_seq` is the interleaved order: cam0 takes the
    /// even ones, velo the odd ones.
    fn sweep(seq: u64, arrival: i64) -> Sample {
        Sample {
            stream: StreamId::LIDAR,
            seq,
            arrival_seq: seq * 2 + 1,
            parent: None,
            tov: Tov::Range {
                start: SensorTime(seq as i64 * 100),
                end: SensorTime(seq as i64 * 100 + 103),
            },
            epoch: 0,
            due: Some(HostTime(arrival - 1_000)),
            arrival: HostTime(arrival),
            payload: RecordBatch::new_empty(Arc::new(Schema::empty())),
            storage_id: 0xa0 + seq as usize,
            decode_ns: 1_000_000,
        }
    }

    /// The camera report with a second producer beside it: 4 frames on
    /// `cam0->proc` and 3 sweeps on `velo->cloud`, all delivered.
    ///
    /// The counts are deliberately DIFFERENT per stream (4 against 3, global
    /// admission 7). A fixture where both streams admitted the same number
    /// cannot tell a per-edge denominator from the global one, which is the
    /// single thing that had to change.
    fn two_producer_report() -> RunReport {
        let ctx = RowCtx {
            run_id: "t".to_string(),
            t0_host: HostTime(0),
            epoch: 0,
        };
        let mut r = report();
        for i in 0..4usize {
            r.rows[i].arrival_seq = Some(i as u64 * 2);
        }
        let sweeps: Vec<Sample> = (0..3).map(|i| sweep(i, 120_000 * (i as i64 + 1))).collect();
        for s in &sweeps {
            r.rows
                .push(Evidence::driver_admitted(&ctx, s, "velo", "velo-driver"));
            r.rows.push(Evidence::delivered(
                &ctx,
                s,
                "velo->cloud",
                "cloud",
                s.arrival,
                0,
                Some(0),
                HostTime(s.arrival.0 + 10),
                HostTime(s.arrival.0 + 20),
                HostTime(s.arrival.0 + 30),
                s.storage_id,
                0,
            ));
        }
        r.admitted = 7;
        r.admitted_by_stream = BTreeMap::from([(StreamId::CAM0.0, 4), (StreamId::LIDAR.0, 3)]);
        r.edges.push(EdgeReport {
            name: "velo->cloud",
            admitted: 3,
            queue_dropped: 0,
            blocked_ns: vec![10, 20, 30],
        });
        r.velo = Some(VeloRun {
            n_sweeps: 3,
            frames: 3,
            absent_in_source: Vec::new(),
            gaps: Vec::new(),
            totals: RunTotals {
                admitted: 3,
                missing: 0,
                wall_ns: 400_000_000,
            },
            driver_bytes: 3 * 1_974_352,
            cloud: CloudReport {
                delivered: 3,
                storage_mismatch: 0,
                count_mismatch: 0,
                bytes_total: 0,
                payload_bytes_total: 3 * 1_974_352,
                by_seq: sweeps.iter().map(|s| (s.seq, s.storage_id)).collect(),
                // No viewer in this fixture, so nothing was drawn and nothing
                // was spent drawing it.
                viz_bytes_total: 0,
                viz_ns_total: 0,
                viz_log_errors: 0,
                points_total: 3 * 100,
                points_min: Some(90),
                points_max: Some(110),
                extent: None,
                // 103 ns sweeps in this fixture, triggered 40 ns in.
                trigger_offset_total_ns: 3 * 40,
                trigger_outside_range: 0,
                trigger_missing: 0,
                // A sweep off the sensor has no parent, so nothing could be
                // missing and there is no seq to join on; the derived
                // stream's own provenance is in `reduce`.
                parent_mismatch: 0,
                parent_by_seq: Vec::new(),
            },
            storage_check: VeloStorageCheck::Ok { compared: 3 },
            // M11's shape: lidar, leaf consumers, no derived chain. The
            // chain's own report is built by `two_producer_report_reducing`.
            reduce: None,
        });
        r
    }

    /// The accounting with two producers, which is the whole of step 1.
    ///
    /// Against the old code every assertion on `cam0->proc` below FAILs: it
    /// compared 3 delivered + 1 dropped against the GLOBAL `admitted = 7`,
    /// which is camera frames measured against camera frames plus sweeps.
    #[test]
    fn a_second_producer_gets_its_own_denominators_and_the_camera_keeps_its_own() {
        let (s, inv) = summarize(&two_producer_report(), None);
        let texts: Vec<&str> = inv.iter().map(|i| i.text.as_str()).collect();
        assert_eq!(
            texts,
            vec![
                "INVARIANT driver admitted=4 missing=0 n_frames=4",
                "INVARIANT driver=velo admitted=3 missing=0 n_sweeps=3",
                "INVARIANT admitted total=7 admission=7",
                "INVARIANT edge=cam0->proc delivered=3 dropped=1 admitted=4",
                "INVARIANT rows edge=cam0->proc rows=4 admitted=4",
                "INVARIANT edge=velo->cloud delivered=3 dropped=0 admitted=3",
                "INVARIANT rows edge=velo->cloud rows=3 admitted=3",
            ]
        );
        assert!(
            inv.iter().all(|i| i.ok),
            "a healthy two-producer run FAILed"
        );
        assert!(s.invariants_ok);

        // The edges are discovered from the rows, in producer order.
        let edges: Vec<&str> = s.stages.iter().map(|st| st.edge).collect();
        assert_eq!(edges, vec!["cam0", "cam0->proc", "velo", "velo->cloud"]);

        // The run-level figures stayed the CAMERA's, unmoved by a second
        // stream: 4 frames, 4 admitted, and the camera's own decode p50.
        assert_eq!((s.n_frames, s.admitted, s.missing), (4, 4, 0));
        assert_eq!(s.decode_ms_p50, Some(5.0));

        let l = s.lidar.expect("a two-producer run carries a lidar block");
        assert_eq!((l.n_sweeps, l.admitted, l.delivered), (3, 3, 3));
        assert!(l.storage_id_equal_all_stages);
        // The pair the next step has to move: ~1.95 MB carried, 0 allocated.
        assert_eq!(l.payload_bytes_total, 3 * 1_974_352);
        assert_eq!(l.payload_bytes_per_sweep, Some(1_974_352));
        assert_eq!(l.cloud_bytes_per_sweep, 0);
        assert_eq!(l.driver_bytes_per_sweep, 1_974_352);
        // The lidar's pacing is its own column, not folded into the camera's.
        assert_eq!(l.pacing_us.p50, Some(1));
        // The payload's trigger, reconciled against the envelope's range.
        assert_eq!(l.trigger_offset_mean_ns, Some(40));
        assert_eq!((l.trigger_outside_range, l.trigger_missing), (0, 0));
    }

    /// The per-edge denominator, made to bite. One sweep is dropped at
    /// admission, so `velo->cloud` reads 2 delivered + 1 dropped against its
    /// own 3 -- while `cam0->proc` is untouched at 4. A global denominator
    /// cannot produce both of those lines.
    #[test]
    fn a_drop_on_one_edge_moves_only_that_edges_invariant() {
        let mut r = two_producer_report();
        // Turn the last sweep's consumer row into the pusher's drop row.
        let last = r.rows.len() - 1;
        r.rows[last].outcome = Outcome::DroppedOldest;
        r.rows[last].reason = "evicted";
        r.edges
            .iter_mut()
            .find(|e| e.name == "velo->cloud")
            .expect("velo edge")
            .queue_dropped = 1;
        if let Some(v) = r.velo.as_mut() {
            v.cloud.delivered = 2;
            v.cloud.by_seq.pop();
        }
        let (_, inv) = summarize(&r, None);
        let texts: Vec<&str> = inv.iter().map(|i| i.text.as_str()).collect();
        assert!(
            texts.contains(&"INVARIANT edge=velo->cloud delivered=2 dropped=1 admitted=3"),
            "{texts:?}"
        );
        assert!(
            texts.contains(&"INVARIANT edge=cam0->proc delivered=3 dropped=1 admitted=4"),
            "{texts:?}"
        );
        assert!(inv.iter().all(|i| i.ok), "{texts:?}");
    }

    /// A frame the source never had, as the recorder writes it: its
    /// driver's `Missing` row, with no instant.
    fn absent_row(stream: StreamId, edge: &'static str, stage: &'static str, seq: u64) -> Evidence {
        let ctx = RowCtx {
            run_id: "t".to_string(),
            t0_host: HostTime(0),
            epoch: 0,
        };
        Evidence::driver_missing(
            &ctx,
            stream,
            edge,
            stage,
            seq,
            Tov::None,
            None,
            HostTime(500_000),
            ABSENT_IN_SOURCE,
        )
    }

    /// The camera's line with a gap in its source: the frames are the PNGs
    /// on disk plus the absent ones, and there is one `absent_in_source` row
    /// at each absent frame's number. Each term bites on its own: a report
    /// whose counters agree with each other and with the rows, but whose
    /// frames are not on disk plus absent, FAILs.
    #[test]
    fn the_camera_line_needs_its_frames_to_be_on_disk_plus_absent() {
        // Frames 0-3 delivered, frame 4 absent in the source.
        let with_gap = |n_frames: usize, on_disk: usize, row_at: u64| {
            let mut r = report();
            r.rows
                .push(absent_row(StreamId::CAM0, "cam0", "driver", row_at));
            r.n_frames = n_frames;
            r.cam_on_disk = on_disk;
            r.cam_absent_in_source = vec![4];
            r.totals.missing = 1;
            summarize(&r, None).1
        };
        let inv = with_gap(5, 4, 4);
        assert_eq!(
            inv[0].text,
            "INVARIANT driver admitted=4 missing=1 n_frames=5 = on disk 4 + absent_in_source 1 (rows 1)"
        );
        assert!(inv[0].ok, "{}", inv[0].text);
        // Five frames, and five PNGs on disk beside an absent one: the
        // counters still balance (4 + 1 == 5) and the rows match, so only
        // `n == on disk + absent` can see it.
        assert!(
            !with_gap(5, 5, 4)[0].ok,
            "frames != on disk + absent passed"
        );
        // The row at another frame's number.
        assert!(
            !with_gap(5, 4, 3)[0].ok,
            "an absent row at the wrong frame passed"
        );
    }

    /// The lidar's line with a gap in its source: its counters balance the
    /// frames, AND the evidence holds one `absent_in_source` row at each
    /// absent frame's number. A report whose counters all agree but whose row
    /// says something else FAILs.
    #[test]
    fn the_lidar_line_needs_an_absent_row_at_each_absent_frame() {
        // Sweeps 0-2 delivered; frame 3 absent in the lidar's source.
        let with_gap = |reason: &'static str, row_at: u64| {
            let mut r = two_producer_report();
            let mut row = absent_row(StreamId::LIDAR, "velo", "velo-driver", row_at);
            row.reason = reason;
            r.rows.push(row);
            if let Some(v) = r.velo.as_mut() {
                v.frames = 4;
                v.absent_in_source = vec![3];
                v.totals.missing = 1;
            }
            let inv = summarize(&r, None).1;
            let line = inv
                .iter()
                .find(|i| i.text.starts_with("INVARIANT driver=velo "))
                .expect("the lidar's line");
            (line.text.clone(), line.ok)
        };
        let (text, ok) = with_gap(ABSENT_IN_SOURCE, 3);
        assert_eq!(
            text,
            "INVARIANT driver=velo admitted=3 missing=1 frames=4 = n_sweeps 3 + absent_in_source 1 (rows 1)"
        );
        assert!(ok, "{text}");
        // The same counts with the missing row a driver's skip: the counters
        // cannot tell, the rows can.
        let (text, ok) = with_gap("deadline_skipped", 3);
        assert!(
            !ok,
            "a skipped sweep passed as absent in the source: {text}"
        );
        // Or at another frame's number.
        assert!(
            !with_gap(ABSENT_IN_SOURCE, 2).1,
            "an absent row at the wrong frame passed"
        );
    }

    /// The steady-state cutoff is the PRODUCER's last admission, not the
    /// run's. The camera's last frame arrives at 400 ms-scale ticks and the
    /// lidar's last sweep earlier, so one run-wide cutoff would trim each
    /// edge with the other stream's clock.
    #[test]
    fn the_steady_window_is_cut_by_each_streams_own_last_admission() {
        let (s, _) = summarize(&two_producer_report(), None);
        let stage = |e: &str| s.stages.iter().find(|st| st.edge == e).expect(e);
        // cam0's last arrival is 400_000; seq 3 was dequeued at 450_000.
        assert_eq!(stage("cam0->proc").steady_rows, 2);
        // velo's last arrival is 360_000 and its last sweep was dequeued at
        // 360_010, after it -- so two of three sweeps are steady. Under the
        // camera's cutoff of 400_000 all three would have counted.
        assert_eq!(stage("velo->cloud").steady_rows, 2);
    }

    #[test]
    fn empty_sample_sets_report_no_measurement_not_zero() {
        // A run that admitted nothing: there is no pacing error, no queue wait
        // and no decode time to report. Before percentile() returned Option,
        // every one of these came out as a confident 0.
        let mut r = report();
        r.rows.clear();
        r.edges[0].blocked_ns.clear();
        r.admitted = 0;
        r.admitted_by_stream.clear();
        r.edges[0].admitted = 0;
        r.edges[0].queue_dropped = 0;
        if let Some(p) = &mut r.proc {
            p.delivered = 0;
        }
        r.n_frames = 0;
        r.totals = RunTotals {
            admitted: 0,
            missing: 0,
            wall_ns: 0,
        };
        let (s, _) = summarize(&r, None);
        assert_eq!(s.decode_ms_p50, None);
        assert_eq!(
            (
                s.pacing_us.p50,
                s.pacing_us.p95,
                s.pacing_us.p99,
                s.pacing_us.max
            ),
            (None, None, None, None)
        );
        assert!(s.stages.is_empty());
    }

    #[test]
    fn tasklist_parse() {
        assert_eq!(
            parse_tasklist_csv("\"pipes.exe\",\"1234\",\"Console\",\"1\",\"123,456 K\"\r\n"),
            Some(123_456 * 1024)
        );
        assert_eq!(parse_tasklist_csv("INFO: No tasks are running."), None);
    }
}
