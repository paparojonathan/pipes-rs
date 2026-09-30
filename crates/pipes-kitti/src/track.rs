//! Tracks: detections associated across sweeps, each with a stable id, an age
//! in sensor time and a velocity — **and the camera frame the sweep belongs
//! to**.
//!
//! This is where the chain stops compressing and starts inferring, and it is
//! worth being blunt about what that costs. Every earlier link made the
//! payload smaller: [`crate::voxel`] decimated a sweep 4x, [`crate::detect`]
//! turned 30,754 points into 177 structures, 62x. **This link does not shrink
//! the bytes at all** — 177 detections become about 183 emitted tracks per
//! sweep on drive_0005 (a track outlives the sweep that last saw it by one
//! sweep, so there can be more tracks than detections), each one larger than
//! a detection because it carries a velocity, an age, how long since it was
//! last seen and an image rectangle that no detection had.
//!
//! That is the honest shape of the chain and the schema is **not** trimmed to
//! improve it. Dropping the 3D box would save 24 of 104 bytes per track and
//! make the ratio look better; it would also make the answer stage guess an
//! object's lateral extent from its centroid, and "no parameter tuned to make
//! the demo look good" has to include the schema. What changes at this link is
//! not the size, it is the **kind**: a track's velocity does not exist in any
//! single sweep, and neither does the fact that it is the same object -- nor,
//! since the camera detector, what the camera says it is ([`crate::fuse`]).
//!
//! # The association gate, and why it is a function rather than a constant
//!
//! Nearest-centre association with a distance gate. The gate is **computed per
//! sweep pair from the measured interval**, not written down:
//!
//! ```text
//! gate(dt) = V_REL_MAX * dt              relative speed of two road users
//!          + YAW_RATE_MAX * dt * R       the apparent swing of a static
//!                                        object at range R under ego yaw
//!          + voxel_edge                  centroid quantisation
//! ```
//!
//! At this drive's 103.27 ms sweep and a 30 m range limit that is
//! `2.101 + 0.781 + 0.200 = 3.08 m`.
//!
//! Each term has a source and none of them came from looking at the output:
//!
//! * **`V_REL_MAX`** = an urban speed limit (50 km/h = 13.89 m/s) plus this
//!   drive's measured maximum ego speed (6.46 m/s, from its OXTS) — head-on,
//!   the worst case. 20.35 m/s.
//! * **`YAW_RATE_MAX`** = this drive's measured maximum, 0.252 rad/s. Over one
//!   sweep that is 1.49 deg, and at the detector's 30 m bound a static object
//!   swings 0.78 m through the frame. **This term is pure ego motion and it is
//!   a quarter of the gate.** The pipeline does not replay OXTS, so the
//!   tracker cannot subtract it and has to widen the gate to swallow it
//!   instead — see the section below for what that costs.
//! * **the voxel edge** = a centroid is the mean of whichever returns landed
//!   in a 0.20 m cell, and which returns those are changes between sweeps.
//!
//! **It scales with `dt`, and that is the point.** [`crate::detect`]'s
//! persistence check learned this the hard way: its first version compared
//! whatever two clouds arrived in succession, which on an unpaced run were
//! ~0.6 s apart, and reported a number about the eviction rate under the name
//! of one about the detector. A fixed gate does the same thing to a tracker —
//! it silently becomes too tight the moment a sweep is dropped. Here a longer
//! interval gets a wider gate automatically, and an interval that is not one
//! sweep long is refused outright (below).
//!
//! # Consecutive sweeps only, by sequence number, with no tolerance
//!
//! A track extends from sweep *k* to sweep *k+1* or not at all. The test is
//! `source_sweep_seq`, carried through the chain by [`crate::detect`], not a
//! threshold on `dt`: it is exact, and the alternative is the same class of
//! mistake the gate above avoids.
//!
//! When a sweep is missing — `velo->reduce` evicted it, `det->detect` evicted
//! it, or the lidar's source never had it (a sweep's number is its frame, so
//! after a gap in the source the next sweep does not follow the last one:
//! 181 after 176 on drive 0009) — **every track dies and the ids start
//! again**. Nothing coasts across the gap. That is a deliberate refusal to
//! hide an upstream loss: a tracker that interpolated through a dropped sweep
//! would produce a plausible answer from data it never had, and
//! `tracks_reset` in the report is the number that says how often the chain
//! broke.
//!
//! Within a run of consecutive sweeps a track does survive **one** unmatched
//! sweep ([`MAX_MISSES`]), and that one is not a tolerance either: 59.2 % of
//! this detector's detections are fragments of larger structures, and which
//! fragments a structure breaks into moves from sweep to sweep. One sweep of
//! memory is the smallest amount that lets a re-split object keep its
//! identity. A coasted track keeps its **last observed** position — it is not
//! propagated forward by its velocity — and lane 8 of the payload says so
//! (lane 20 says how long ago, in seconds), so a consumer that needs a fresh
//! observation can demand `misses == 0`. The answer stage does.
//!
//! # Velocity, and why it is measured over the track's whole life
//!
//! `v = (p_now - p_first) / (t_now - t_first)`, not the difference against the
//! previous sweep.
//!
//! Two-point differencing over one 103 ms interval is unusable here and the
//! arithmetic says so before any data does: a voxel centroid carries up to
//! 0.20 m of quantisation, so a per-sweep velocity carries up to
//! `0.20 / 0.103 = 1.94 m/s` of noise — comparable to the speeds being
//! measured. Over a track's whole life the same 0.20 m is divided by the whole
//! elapsed time, so the noise is `0.20 m / age_s`: 1.94 m/s for a track first
//! seen one sweep ago, 0.28 m/s at 0.72 s, 0.13 m/s at 1.55 s. That is a
//! two-point estimate over the longest baseline available and it introduces no
//! filter, no gain and no parameter. `age_s` is exactly that baseline — the
//! velocity's denominator — which is one reason it is measured the way the
//! next section says.
//!
//! It also means **a young track's velocity is nearly meaningless**, which is
//! why lane 19 carries the age beside it rather than leaving a reader to
//! assume every row is equally good.
//!
//! # Age is sensor time, first seen to last seen, on the sweep's trigger
//!
//! `age_s = last_seen - first_seen`, and `since_seen_s = now - last_seen`,
//! where every instant is a sweep's **trigger** (`tov_trigger_ns`, from the
//! velodyne's `timestamps.txt`) and "now" is the trigger of the sweep this
//! batch was built for. Integer nanoseconds on the sensor clock inside the
//! tracker, converted to f32 seconds once, when the lane is written. Never
//! host time: `clippy.toml` keeps the host clock out of this crate, and a
//! replay at `--rate 10` must report the same ages as one at `--rate 1`.
//!
//! **Why not a count of observations.** Age used to be the number of sweeps
//! a track had been matched in, and that number understates the track's life
//! whenever it misses a sweep: a track that coasts (below) keeps going
//! without an observation, so its count stops while its life does not. On
//! drive_0005 42 % of emitted rows had coasted at least once, and the worst
//! count understated its track's real span by ten sweeps, about a second.
//! A count also quietly assumes every interval is one sweep period; today a
//! gap resets the tracker (below), but if that rule were ever relaxed a count
//! would read the same across a gap of eight lost sweeps as across none. The
//! count is still carried, renamed `observations` (lane 7), because the
//! emission rule is about it: a velocity needs two observations
//! ([`MIN_OBSERVATIONS`]), however far apart.
//!
//! **Why the trigger, and not the start or the end of the sweep's range.**
//! Three instants describe a sweep; the trigger is the one:
//!
//! * **it is when the head faces forward** — `timestamps.txt` is the instant
//!   the scanner faced +x and the cameras fired ([`crate::velo`]) — so it is
//!   when anything ahead of the vehicle, and anything in the camera's view,
//!   was actually measured. The start and end of the range are the
//!   rear-facing instants, 51.6 ms either side of it on drive_0005;
//! * **it is the clock everything else here already uses**: `dt`, the gate
//!   and the velocity baseline are all trigger to trigger, and
//!   `pair_age_ns` is the camera instant minus the trigger. Measuring age on
//!   another end of the range would put two clocks in one record, and the
//!   velocity's baseline and the age beside it would disagree by up to the
//!   0.94 ms by which trigger intervals and start intervals differ;
//! * **mixing ends inflates it**: start of the first range to end of the
//!   last adds one whole sweep span (103.3 ms) to every age, so a track seen
//!   in two consecutive sweeps would read twice its real baseline.
//!
//! The limit, stated rather than corrected: a return is really measured at
//! about `trigger + bearing / omega` (the head turns ~3.5 deg/ms), so for an
//! object whose bearing changes a lot while it is tracked the age is off by
//! the change in bearing over omega — tens of milliseconds at most, and zero
//! for anything that stays at one bearing.
//!
//! # The velocity is in the sensor frame and is NOT ego-compensated
//!
//! Stated plainly because it is the difference between a number that is wrong
//! and a number that is right for a different question than you might assume.
//! A parked car ahead of a vehicle doing 5 m/s has a tracked velocity of
//! -5 m/s in this frame.
//!
//! For the answer this chain produces — *how fast is the gap between me and
//! that thing closing* — that is the correct quantity and an ego-compensated
//! one would be the wrong one. For "is that object moving in the world", it is
//! not, and this module cannot answer that, because nothing in the pipeline
//! replays OXTS. Both facts are in [`state`](crate::state)'s docs at the point
//! where the number is used.
//!
//! # Determinism
//!
//! On the same terms as [`crate::detect`]. No map — `clippy.toml` bans
//! `HashMap`/`HashSet` and the pairing uses a sorted `Vec` instead. Candidate
//! pairs are sorted by `(distance, track index, detection index)` with
//! `f32::total_cmp`, so the greedy assignment cannot depend on the order pairs
//! were generated in and cannot depend on a partial comparison of a NaN.
//! Track ids are handed out in the order unmatched detections appear, and
//! detections arrive in `detect`'s own key order, which is itself independent
//! of its input order.
//!
//! That is determinism over the same sweeps in the same order. The tracker
//! carries state from sweep to sweep, so a run that lost a sweep upstream --
//! evicted on a loaded host, say -- hands it a different sequence, and every
//! answer after the loss may differ from a quiet run's.

use std::sync::Arc;

use arrow::array::{
    Array, ArrayRef, AsArray, BooleanArray, FixedSizeListArray, Float32Array, Int64Array,
    LargeListArray, RecordBatch, StringArray, UInt32Array,
};
use arrow::buffer::{Buffer, MutableBuffer, OffsetBuffer, ScalarBuffer};
use arrow::datatypes::{DataType, Field, FieldRef, Float32Type, Int64Type, Schema, UInt32Type};

use crate::calib::{Calib, ImageBox};
use crate::camdet::DetView;
use crate::detect::DETECTION_LANES;
use crate::fuse::{build_camera_only, FusePlan, Fuser, POPULATION_FUSED, POPULATION_UNFUSED};
use crate::velo::SweepError;
use crate::voxel::VoxelSize;

/// f32 lanes in one track record. See [`TRACK_BYTES`] for the layout.
pub const TRACK_VALUES: i32 = 26;

/// [`TRACK_VALUES`] as a `usize`, for slicing. The same number twice because
/// Arrow's `FixedSizeListArray` wants an `i32` width and `as_chunks` wants a
/// `usize` const generic; derived from the other so the pair cannot drift.
pub const TRACK_LANES: usize = TRACK_VALUES as usize;

/// Bytes in one track: 26 f32 lanes, in this order.
///
/// | lanes | field | why it is here |
/// |---|---|---|
/// | 0-2 | `position` x, y, z | the centroid of the latest observation, in the lidar frame. |
/// | 3-5 | `velocity` vx, vy, vz | m/s in the **sensor** frame, over the track's whole life. Not ego-compensated — see the module docs. |
/// | 6 | `id` | stable while the track lives; never reissued within a run. |
/// | 7 | `observations` | sweeps the track was matched in, starting at 1. What the emission rule ([`MIN_OBSERVATIONS`]) counts; **not** the track's age, which it understates whenever the track coasts. |
/// | 8 | `misses` | consecutive sweeps with no observation, 0 or [`MAX_MISSES`]. `0` means "seen this sweep". |
/// | 9-11 | `bbox_min` x, y, z | the near face is what "distance to the obstacle" means, and the lateral extent is what decides whether it is in the vehicle's path. A centroid gives neither. |
/// | 12-14 | `bbox_max` x, y, z | the other corner. |
/// | 15-18 | `image_box` x0, y0, x1, y1 | where this object is in the paired camera frame, in pixels. **All four zero means it is not in frame** — a real rectangle always has positive area, so zero is unambiguous and needs no fifth lane to flag it. |
/// | 19 | `age_s` | last seen minus first seen, seconds on the sensor clock, trigger to trigger. The velocity's baseline and therefore its quality: the quantisation noise is `0.20 m / age_s`. See the module docs for why it is time and why the trigger. |
/// | 20 | `since_seen_s` | this sweep's trigger minus last seen, seconds: `0` exactly when `misses` is 0, one sweep interval when the track is coasting. How old the position in lanes 0-2 and 9-18 is. |
/// | 21 | `det_index` | **the camera half.** The index, in the paired frame's `CAM_DET` batch, of the one detection [`crate::fuse`] associated with this track, or `-1`. With `cam_seq` it names the box in `cam_det.arrows`. |
/// | 22 | `fuse_iou` | the image-space IoU of this track's box (lanes 15-18) with that detection's, above 1/2 by construction; `0` when there is none. |
/// | 23 | `class_id` | that detection's class, an index into [`crate::camdet::COCO_CLASSES`]; `0` when there is none, which is also "person" -- so it is read only on a fused track (lane 25). |
/// | 24 | `confidence` | that detection's score, above the detector's 0.3 by construction; `0` when there is none. |
/// | 25 | `population` | [`crate::fuse::POPULATION_FUSED`] with a detection, [`crate::fuse::POPULATION_LIDAR_ONLY`] when a detector's output was associated and nothing matched, [`crate::fuse::POPULATION_UNFUSED`] when there was no detector output to associate with (`--detector off`, whose frame reference carries no detections). |
///
/// **A fused track carries its distance, velocity and age from the lidar
/// (lanes 0-20) and its class and confidence from the camera (lanes 23-24),**
/// in one record, and nothing else from the camera: no class memory across
/// sweeps and no camera-derived geometry. The camera's frame is named by the
/// batch's `cam_seq`, and how stale it was by its `pair_age_ns`, so a wrong
/// class is attributable to the frame it came from. See [`crate::fuse`].
///
/// `id` and `observations` are integers in f32 lanes, exactly as
/// [`crate::detect::DETECTION_BYTES`] carries `voxel_count`: f32 holds every
/// integer below 2^24, and a whole drive issues thousands of ids, not
/// millions. A separate integer column would give the batch a second data
/// buffer and therefore a second `storage_id`, and the address-equality
/// evidence this project proves zero copy with assumes one.
///
/// The two times are **relative** seconds for the same reason. An absolute
/// instant (about 1.3e18 ns) does not fit an f32, and an i64 column is the
/// second buffer. Relative to the batch's own `tov_trigger_ns` they are exact
/// to about a microsecond — f32 rounding at 16 s — against the 0.94 ms by
/// which trigger intervals jitter; the absolute instants can be rebuilt from
/// them (`last = trigger - since_seen`, `first = last - age`) to that
/// microsecond, not to the nanosecond the tracker holds.
///
/// The five camera lanes (21-25) are in the same buffer for the same reason:
/// a fused object is ONE record with one address, so the zero-copy proof that
/// follows the tracks to `state` follows the fusion with them.
pub const TRACK_BYTES: usize = 104;

/// Lane 21 of [`TRACK_BYTES`]: the associated detection's index, or -1.
pub const LANE_DET_INDEX: usize = 21;
/// Lane 22: the association's IoU.
pub const LANE_FUSE_IOU: usize = 22;
/// Lane 23: the detection's class id.
pub const LANE_CLASS_ID: usize = 23;
/// Lane 24: the detection's confidence.
pub const LANE_CONFIDENCE: usize = 24;
/// Lane 25: the population code.
pub const LANE_POPULATION: usize = 25;

/// Value of the `track_format` column: what the 26 lanes mean.
///
/// `p3` position, `v3` velocity, `i1` id, `n1` observations, `m1` misses, `b6`
/// two box corners, `c4` the camera rectangle, `a1` age in seconds, `s1`
/// seconds since last seen, `j1` detection index, `u1` IoU, `k1` class, `q1`
/// confidence, `o1` population. Deliberately shares no spelling with
/// [`crate::detect::DETECTION_FORMAT`] or the point formats, so nothing
/// downstream can read a track as a detection or as a point — and it changed
/// when lane 7 stopped being called the age, and again when the camera lanes
/// were added, because a lane whose meaning changes under the same format
/// string is the failure the string exists to prevent.
pub const TRACK_FORMAT: &str = "trk_p3v3i1n1m1b6c4a1s1j1u1k1q1o1_f32le";

/// Relative speed two road users can close at, m/s.
///
/// An urban speed limit (50 km/h = 13.89 m/s) plus drive_0005's measured
/// maximum ego speed (6.46 m/s, from its OXTS `vf`/`vl`), head-on. It is a
/// bound on the drive this project replays and **not** a universal one: a
/// motorway drive would need a larger number here, and the gate would widen
/// with it.
pub const V_REL_MAX_MPS: f64 = 13.89 + 6.46;

/// Ego yaw rate bound, rad/s: drive_0005's measured maximum (-0.252..+0.192).
///
/// This term exists only because the pipeline does not replay OXTS. With the
/// vehicle's own rotation available it would be subtracted rather than
/// tolerated, and the gate would be about a quarter narrower.
pub const YAW_RATE_MAX_RPS: f64 = 0.252;

/// Consecutive sweeps a track survives without an observation before it dies.
///
/// **1**, and derived rather than chosen: 59.2 % of this detector's detections
/// are fragments of larger structures broken up by sparse sampling, and which
/// fragments a structure breaks into changes from sweep to sweep. One sweep is
/// the smallest memory that lets a re-split object keep its identity. It is
/// deliberately not larger — a track that coasts for several sweeps is a claim
/// about data that was never received, which is the thing this project exists
/// to make impossible rather than merely discouraged.
pub const MAX_MISSES: u32 = 1;

/// Fewest observations at which a track is emitted.
///
/// **2**, and this is arithmetic rather than a parameter: a velocity is a
/// difference between two observations, so a track with one observation has no
/// velocity to report and is not yet a claim that anything persisted. Tracks
/// below it are kept alive internally — that is how they reach 2 — and simply
/// not written to the payload.
///
/// A count and not a time, deliberately: the rule is about how many
/// measurements the velocity is made of, not how long ago the first was. It
/// used to be called the minimum *age*, when age was this count.
pub const MIN_OBSERVATIONS: u32 = 2;

/// The association gate for an interval of `dt` seconds, in metres.
///
/// See the module docs for where each term comes from. `range_limit_m` is the
/// detector's own bound, because that is the largest radius at which a
/// detection can exist and therefore the largest lever ego yaw has.
pub fn association_gate_m(dt_s: f64, voxel: VoxelSize, range_limit_m: f32) -> f64 {
    let closing = V_REL_MAX_MPS * dt_s;
    let swing = YAW_RATE_MAX_RPS * dt_s * f64::from(range_limit_m);
    closing + swing + f64::from(voxel.metres())
}

/// The gate's three terms in one line, so a run's own output can state where
/// its number came from and a reader never has to take it on trust.
pub fn gate_note(dt_s: f64, voxel: VoxelSize, range_limit_m: f32) -> String {
    format!(
        "gate = {:.3} m at dt = {:.1} ms: {:.3} m closing ({} m/s relative, an urban limit plus \
         this drive's measured max ego speed) + {:.3} m ego-yaw swing ({} rad/s measured, at the \
         detector's {} m bound -- this term is ego motion the pipeline cannot subtract because \
         oxts is not replayed) + {} m voxel quantisation",
        association_gate_m(dt_s, voxel, range_limit_m),
        dt_s * 1e3,
        V_REL_MAX_MPS * dt_s,
        V_REL_MAX_MPS,
        YAW_RATE_MAX_RPS * dt_s * f64::from(range_limit_m),
        YAW_RATE_MAX_RPS,
        range_limit_m,
        voxel.metres(),
    )
}

/// What happened when this sweep was matched to a camera frame.
///
/// The four failures are kept apart because they have different causes and a
/// reader has to be able to attribute the bad answer to the right one. Calling
/// a frame that has not arrived *yet* a drop is a false accusation, and the
/// measured case that makes it one is real: at `--detector off
/// --consumer-delay-ms 100 --cap 16` the camera path drops **nothing**, and
/// a fusion that does not wait finds no frame for any of the 154 sweeps.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Pairing {
    /// A camera instant fell inside this sweep's `Tov::Range`. A **completed**
    /// fusion set.
    Paired,
    /// No frame inside the range, and a declared stale window allowed an older
    /// one to be used instead. A **degraded** set: the sample exists and
    /// carries `pair_age_ns` saying how stale, so the threshold is applied by
    /// whoever reads it rather than baked into the data.
    Stale,
    /// The buffer held a frame before the range and a frame after it, but none
    /// inside: the frame that belonged here was produced and lost upstream.
    Dropped,
    /// Nothing newer than the range's end had arrived when the bounded wait
    /// expired. Not the same failure as [`Pairing::Dropped`]: the frame may
    /// simply be behind.
    Late,
    /// No camera frame was ever seen for this sweep and none was expected —
    /// the run has no camera stream reaching this stage at all.
    #[default]
    Absent,
    /// The camera's source has no frame for this sweep: KITTI's `image_02`
    /// skips the frame number the sweep carries (both sensors share one frame
    /// index). Not [`Pairing::Dropped`] -- nothing was produced to lose -- and
    /// not [`Pairing::Late`] -- nothing is coming, so the fusion does not wait
    /// for it. The pairing itself is still by time; the frame number only
    /// names why no instant fell inside the range.
    AbsentInSource,
}

impl Pairing {
    /// The payload / evidence spelling.
    pub fn name(self) -> &'static str {
        match self {
            Pairing::Paired => "paired",
            Pairing::Stale => "stale",
            Pairing::Dropped => "pair_dropped",
            Pairing::Late => "pair_late",
            Pairing::Absent => "pair_absent",
            Pairing::AbsentInSource => "pair_absent_in_source",
        }
    }

    /// Whether a fused sample may be produced at all. `false` is the doc's
    /// **expired** set: nothing is emitted and the evidence says why.
    pub fn produces(self) -> bool {
        matches!(self, Pairing::Paired | Pairing::Stale)
    }
}

/// The camera half of a pair: everything the fusion needs about a frame, and
/// nothing about its pixels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CamRef {
    /// The camera driver's own frame number.
    pub seq: u64,
    /// The instant the frame is valid at, ns on the sensor clock.
    pub tov_ns: i64,
    /// Image width in pixels, as the frame itself reports it.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
}

/// Arrow field of one track lane.
fn track_value_field() -> FieldRef {
    Arc::new(Field::new("item", DataType::Float32, false))
}

/// Arrow field of one track: `FixedSizeList<Float32, 21>`.
fn track_field() -> FieldRef {
    Arc::new(Field::new(
        "item",
        DataType::FixedSizeList(track_value_field(), TRACK_VALUES),
        false,
    ))
}

/// `track_count u32, track_format Utf8, tov_trigger_ns i64,
/// source_sweep_seq i64, source_detections u32, source_voxel_count u32,
/// source_point_count u32, gate_m f32, dt_s f32, cam_seq i64, cam_tov_ns i64,
/// pair_age_ns i64, pair_outcome Utf8, image_width u32, image_height u32,
/// paired bool, tracks LargeList<FixedSizeList<Float32, 26>>, fused_count
/// u32, fuse_contested u32, fuse_crowded u32, camera_only
/// LargeList<FixedSizeList<Float32, 7>>`.
///
/// The last four are the association's own statement ([`crate::fuse`]):
/// how many tracks it fused, how often a detection or a track had more than
/// one partner above the threshold, and the paired frame's detections that
/// matched no track, each as `(det_index, class_id, confidence, x0, y0, x1,
/// y1)`. Every detection of the frame is therefore in exactly one place --
/// on a track's lanes 21-25 or in `camera_only` -- and `fused_count +
/// camera_only` is the frame's detection count.
///
/// `source_sweep_seq` names the lidar parent; `cam_seq` names the camera one,
/// and it is also the `CAM_DET` sample's seq, because `camdet` numbers its
/// batches by the frame they were found in. So the one column names both the
/// frame and the detections, and `run::pair_check` joins it to both.
///
/// **The three columns that are the point of this stage** are `cam_seq`,
/// `cam_tov_ns` and `pair_age_ns`. [`pipes_core::sample::Sample::parent`] can
/// name only one parent and it names the lidar one, because that is the parent
/// whose `tov` and `due` this sample inherits; the camera parent has nowhere
/// to go in the envelope and travels here instead. That is the same place
/// `tov_trigger_ns` travels and for the same reason — the stage that needs it
/// is reading the payload anyway.
///
/// It is only honest **because these columns are joined by a test.** The
/// project has already shipped a provenance field that was written and never
/// read: `Sample::parent` was declared, set to `None` at every site, read
/// nowhere, and a derived sample naming the wrong sweep passed every test in
/// the workspace. `run::pair_check` exists so that cannot happen to the field
/// the whole fusion claim rests on.
///
/// `pair_age_ns` is a **number, not a flag**, on the same principle as
/// `queue_wait_ns`: the run records the measurement and the check computes the
/// verdict, so a reader can apply their own staleness threshold to a recording
/// made under someone else's.
pub fn track_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("track_count", DataType::UInt32, false),
        Field::new("track_format", DataType::Utf8, false),
        Field::new("tov_trigger_ns", DataType::Int64, false),
        Field::new("source_sweep_seq", DataType::Int64, false),
        Field::new("source_detections", DataType::UInt32, false),
        Field::new("source_voxel_count", DataType::UInt32, false),
        Field::new("source_point_count", DataType::UInt32, false),
        Field::new("gate_m", DataType::Float32, false),
        Field::new("dt_s", DataType::Float32, false),
        Field::new("cam_seq", DataType::Int64, false),
        Field::new("cam_tov_ns", DataType::Int64, false),
        Field::new("pair_age_ns", DataType::Int64, false),
        Field::new("pair_outcome", DataType::Utf8, false),
        Field::new("image_width", DataType::UInt32, false),
        Field::new("image_height", DataType::UInt32, false),
        Field::new("paired", DataType::Boolean, false),
        Field::new("tracks", DataType::LargeList(track_field()), false),
        Field::new("fused_count", DataType::UInt32, false),
        Field::new("fuse_contested", DataType::UInt32, false),
        Field::new("fuse_crowded", DataType::UInt32, false),
        Field::new("camera_only", crate::fuse::camera_only_data_type(), false),
    ]))
}

/// The interleaved track lanes of a track batch, shared with the buffer the
/// stage built. `chunks_exact(26)` gives one track each.
pub fn tracks_f32(batch: &RecordBatch) -> Option<&[f32]> {
    Some(
        batch
            .column_by_name("tracks")?
            .as_list_opt::<i64>()?
            .values()
            .as_fixed_size_list_opt()?
            .values()
            .as_primitive_opt::<Float32Type>()?
            .values(),
    )
}

/// Address of the shared track buffer, read back out of the finished batch:
/// the zero-copy proof, extended to this stream.
pub fn track_storage_id(batch: &RecordBatch) -> Option<usize> {
    tracks_f32(batch).map(|v| v.as_ptr() as usize)
}

/// The tracks of a batch as fixed-size records, or an empty slice.
pub fn track_rows(batch: &RecordBatch) -> &[[f32; TRACK_LANES]] {
    let lanes: &[f32] = tracks_f32(batch).unwrap_or(&[]);
    let (rows, _) = lanes.as_chunks::<TRACK_LANES>();
    rows
}

fn u32_col(batch: &RecordBatch, name: &str) -> Option<u32> {
    let c = batch
        .column_by_name(name)?
        .as_primitive_opt::<UInt32Type>()?;
    (!c.is_empty()).then(|| c.value(0))
}

fn i64_col(batch: &RecordBatch, name: &str) -> Option<i64> {
    let c = batch
        .column_by_name(name)?
        .as_primitive_opt::<Int64Type>()?;
    (!c.is_empty()).then(|| c.value(0))
}

fn f32_col(batch: &RecordBatch, name: &str) -> Option<f32> {
    let c = batch
        .column_by_name(name)?
        .as_primitive_opt::<Float32Type>()?;
    (!c.is_empty()).then(|| c.value(0))
}

fn str_col<'a>(batch: &'a RecordBatch, name: &str) -> Option<&'a str> {
    let c = batch
        .column_by_name(name)?
        .as_any()
        .downcast_ref::<StringArray>()?;
    (!c.is_empty()).then(|| c.value(0))
}

/// Tracks in the batch, from the metadata column rather than the payload
/// length, so a disagreement between the two is visible.
pub fn track_count(batch: &RecordBatch) -> Option<u32> {
    u32_col(batch, "track_count")
}

/// What the 26 lanes of a track MEAN, read back out of the batch.
pub fn track_format(batch: &RecordBatch) -> Option<&str> {
    str_col(batch, "track_format")
}

/// The sweep's trigger instant, inherited unchanged down the chain.
pub fn track_trigger_ns(batch: &RecordBatch) -> Option<i64> {
    i64_col(batch, "tov_trigger_ns")
}

/// The velodyne sweep these tracks were last observed in.
pub fn track_sweep_seq(batch: &RecordBatch) -> Option<i64> {
    i64_col(batch, "source_sweep_seq")
}

/// Detections these tracks were associated from.
pub fn source_detections(batch: &RecordBatch) -> Option<u32> {
    u32_col(batch, "source_detections")
}

/// Voxels behind those detections, carried through so every link of the chain
/// is recomputable from this batch alone.
pub fn source_voxel_count(batch: &RecordBatch) -> Option<u32> {
    u32_col(batch, "source_voxel_count")
}

/// Raw laser returns behind everything above, from the same place.
pub fn source_point_count(batch: &RecordBatch) -> Option<u32> {
    u32_col(batch, "source_point_count")
}

/// The association gate that was in force, in metres.
pub fn gate_m(batch: &RecordBatch) -> Option<f32> {
    f32_col(batch, "gate_m")
}

/// The interval the gate and the velocities were computed over, in seconds.
pub fn dt_s(batch: &RecordBatch) -> Option<f32> {
    f32_col(batch, "dt_s")
}

/// The camera frame this sweep was paired with, or `-1`.
///
/// **The camera half of the provenance.** See [`track_schema`].
pub fn cam_seq(batch: &RecordBatch) -> Option<i64> {
    i64_col(batch, "cam_seq")
}

/// That frame's instant, ns on the sensor clock, or 0 when unpaired.
pub fn cam_tov_ns(batch: &RecordBatch) -> Option<i64> {
    i64_col(batch, "cam_tov_ns")
}

/// `cam_tov_ns - tov_trigger_ns`: how far the camera instant sits from the
/// sweep's forward-facing trigger, ns. About +10.5 ms on a healthy pair.
pub fn pair_age_ns(batch: &RecordBatch) -> Option<i64> {
    i64_col(batch, "pair_age_ns")
}

/// How the pairing went, as [`Pairing::name`] spells it.
pub fn pair_outcome(batch: &RecordBatch) -> Option<&str> {
    str_col(batch, "pair_outcome")
}

/// Whether the camera frame was inside the sweep's range (as opposed to a
/// declared stale one).
pub fn paired(batch: &RecordBatch) -> Option<bool> {
    let c = batch
        .column_by_name("paired")?
        .as_any()
        .downcast_ref::<BooleanArray>()?;
    (!c.is_empty()).then(|| c.value(0))
}

/// Tracks the association fused with a detection of the paired frame: the
/// batch's own count, beside the lanes it counts, so the two can be checked
/// against each other.
pub fn fused_count(batch: &RecordBatch) -> Option<u32> {
    u32_col(batch, "fused_count")
}

/// Detections of the paired frame that had MORE THAN ONE fresh in-frame track
/// above the association's threshold.
pub fn fuse_contested(batch: &RecordBatch) -> Option<u32> {
    u32_col(batch, "fuse_contested")
}

/// Fresh in-frame tracks that had more than one detection above it -- the
/// same contest seen from the other side.
pub fn fuse_crowded(batch: &RecordBatch) -> Option<u32> {
    u32_col(batch, "fuse_crowded")
}

/// Errors from building one track batch.
#[derive(Debug)]
pub enum TrackError {
    /// The derived buffer could not be wrapped as Arrow.
    Sweep(SweepError),
    /// [`Tracker::build`] was asked for a different number of tracks than the
    /// plan counted — the two would then disagree about the payload's length.
    PlanMismatch {
        /// Tracks the plan said would be emitted.
        planned: u32,
        /// Tracks `build` found.
        got: usize,
    },
}

impl std::fmt::Display for TrackError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TrackError::Sweep(_) => write!(f, "the track batch could not be wrapped as Arrow"),
            TrackError::PlanMismatch { planned, got } => write!(
                f,
                "the plan counted {planned} emitted tracks but build found {got}"
            ),
        }
    }
}

impl From<SweepError> for TrackError {
    fn from(e: SweepError) -> Self {
        TrackError::Sweep(e)
    }
}

impl std::error::Error for TrackError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            TrackError::Sweep(e) => Some(e),
            TrackError::PlanMismatch { .. } => None,
        }
    }
}

/// One live track. Internal: the Arrow record is the 26 lanes of
/// [`TRACK_BYTES`], and this carries the extra state that produces them.
#[derive(Clone, Copy, Debug)]
struct Live {
    id: u64,
    pos: [f32; 3],
    lo: [f32; 3],
    hi: [f32; 3],
    /// The first observation's centroid: the far end of the velocity
    /// baseline. See the module docs for why the baseline is the track's
    /// whole life rather than one sweep.
    first_pos: [f32; 3],
    /// The trigger of the first sweep this track was matched in, ns on the
    /// sensor clock. With [`Live::last_ns`], the age; see the module docs for
    /// why the trigger.
    first_ns: i64,
    /// The trigger of the latest sweep this track was matched in. Not moved
    /// by a coasted sweep: it is when the track was last SEEN.
    last_ns: i64,
    /// Sweeps matched in, starting at 1: the emission rule's count, and not
    /// the age.
    observations: u32,
    misses: u32,
}

impl Live {
    /// Last seen minus first seen, ns on the sensor clock: the track's age,
    /// and the velocity's baseline.
    fn age_ns(&self) -> i64 {
        self.last_ns - self.first_ns
    }

    /// m/s over the whole life, or zero while there is no baseline.
    fn velocity(&self) -> [f32; 3] {
        let dt = self.age_ns() as f64 * 1e-9;
        if dt <= 0.0 {
            return [0.0; 3];
        }
        let mut v = [0.0f32; 3];
        for (i, out) in v.iter_mut().enumerate() {
            *out = ((f64::from(self.pos[i]) - f64::from(self.first_pos[i])) / dt) as f32;
        }
        v
    }
}

/// Nanoseconds on the sensor clock as the f32 seconds a lane carries. The one
/// conversion, so every time lane rounds the same way.
fn ns_to_lane_s(ns: i64) -> f32 {
    (ns as f64 * 1e-9) as f32
}

/// What one tracking update did, counted while it was doing it.
///
/// Every field is reported rather than asserted away, on the same terms as
/// [`crate::detect::DetectPlan`]. The ones that matter most are the ones that
/// say the tracker is *not* working: [`TrackPlan::ambiguous`] and
/// [`TrackPlan::resets`].
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct TrackPlan {
    /// Detections this update was given.
    pub source_detections: u32,
    /// Tracks alive after the update, at any age.
    pub live: u32,
    /// Tracks written to the payload: those with at least
    /// [`MIN_OBSERVATIONS`].
    pub emitted: u32,
    /// Tracks created this update, i.e. detections that matched nothing.
    pub born: u32,
    /// Tracks that ran out of [`MAX_MISSES`] and died.
    pub died: u32,
    /// Tracks that survived this update without an observation.
    pub coasted: u32,
    /// Detections associated with an existing track.
    pub matched: u32,
    /// Detections with **more than one** live track inside the gate.
    ///
    /// The tracker's own measure of how badly it is over-reaching. A gate wide
    /// enough to swallow un-subtracted ego motion is wide enough to capture
    /// the wrong object, and this counts the sweeps where it had a choice.
    /// Reported, never suppressed: nearest wins, and this says how often
    /// "nearest" was a coin toss.
    pub ambiguous: u32,
    /// Live tracks with more than one detection inside the gate — the same
    /// ambiguity seen from the other side.
    pub contested: u32,
    /// Whether the chain broke before this update: the previous sweep was not
    /// `seq - 1`, so every track was discarded and the ids started again.
    ///
    /// Counted by the caller into `tracks_reset`. It is the number that
    /// attributes a collapsed tracker to an upstream eviction rather than to
    /// the algorithm.
    pub reset: bool,
    /// Interval since the previous sweep, seconds; 0 on a reset.
    pub dt_s: f64,
    /// The gate that interval produced, metres.
    pub gate_m: f64,
}

/// Reusable scratch for tracking, so the only per-sample allocation this stage
/// makes is the output buffer itself.
///
/// The same load-bearing constraint [`crate::detect::Detector`] documents, for
/// the same reason: per-sample scratch inside the measured window would make
/// the reported figure "the stage's output plus its bookkeeping" while the
/// column still said output.
#[derive(Debug, Default)]
pub struct Tracker {
    live: Vec<Live>,
    next_id: u64,
    prev_sweep_seq: Option<i64>,
    /// The trigger of the sweep the last [`Tracker::plan`] was given: the
    /// interval to the next one is measured from it, and it is "now" for
    /// every `since_seen_s` [`Tracker::build`] writes.
    prev_ns: i64,
    /// Candidate `(distance^2, track index, detection index)` inside the gate.
    pairs: Vec<(f32, u32, u32)>,
    /// Whether each detection / track has been assigned this update.
    det_taken: Vec<bool>,
    trk_taken: Vec<bool>,
    /// Detections that found this track inside the gate, per track, and the
    /// mirror — both only for the ambiguity counters.
    det_cands: Vec<u32>,
    trk_cands: Vec<u32>,
}

impl Tracker {
    /// An empty tracker. [`Tracker::reserve`] sizes its scratch.
    pub fn new() -> Tracker {
        Tracker::default()
    }

    /// Makes room for `dets` detections against the tracks currently live.
    ///
    /// **Call this outside the measured window.** Returns whether it had to
    /// allocate, so a caller can announce the growth and a test can assert the
    /// steady state does not grow at all.
    pub fn reserve(&mut self, dets: usize) -> bool {
        let tracks = self.live.len() + dets;
        // One pair per (track, detection) in the worst case. This is the only
        // quadratic term and it is why the reserve is announced: at 177
        // detections against ~200 live tracks it is ~67,000 entries, 800 KB,
        // allocated outside the measured window.
        let pairs = tracks.saturating_mul(dets);
        let grew = self.pairs.capacity() < pairs
            || self.det_taken.capacity() < dets
            || self.trk_taken.capacity() < tracks
            || self.det_cands.capacity() < dets
            || self.trk_cands.capacity() < tracks
            || self.live.capacity() < tracks;
        if !grew {
            return false;
        }
        // **Clear before reserving.** `Vec::reserve(n)` promises capacity for
        // `len + n`, and these scratch vectors still hold the PREVIOUS sweep's
        // entries at this point -- so reserving `want - capacity` against a
        // non-empty vector asked for the wrong amount and the capacity check
        // above fired again on the next sweep. It grew 146 times in 154
        // sweeps before this, which is not a steady state and made the
        // "allocated once, outside the window" claim false in the only way
        // that matters.
        //
        // `live` is deliberately NOT cleared: it is the tracker's state, not
        // scratch, and clearing it here would silently reset every track.
        self.pairs.clear();
        self.det_taken.clear();
        self.trk_taken.clear();
        self.det_cands.clear();
        self.trk_cands.clear();
        // Geometric, not exact: `tracks` moves by a few every sweep as objects
        // come and go, and `reserve_exact` would reallocate on each of those.
        // `Voxelizer` and `Detector` can use `reserve_exact` because their
        // sizes are fixed by the input; this one's is fixed by the input AND
        // by how many tracks happen to be alive.
        self.pairs.reserve(pairs);
        self.det_taken.reserve(dets);
        self.trk_taken.reserve(tracks);
        self.det_cands.reserve(dets);
        self.trk_cands.reserve(tracks);
        self.live.reserve(tracks - self.live.len());
        true
    }

    /// Bytes the scratch currently holds.
    pub fn scratch_bytes(&self) -> usize {
        self.pairs.capacity() * std::mem::size_of::<(f32, u32, u32)>()
            + self.det_taken.capacity()
            + self.trk_taken.capacity()
            + self.det_cands.capacity() * 4
            + self.trk_cands.capacity() * 4
            + self.live.capacity() * std::mem::size_of::<Live>()
    }

    /// Tracks currently alive, at any age.
    pub fn live(&self) -> usize {
        self.live.len()
    }

    /// Ids issued so far in this run.
    pub fn ids_issued(&self) -> u64 {
        self.next_id
    }

    /// Phase 1: associate `dets` with the live tracks. **Borrows `dets` and
    /// allocates nothing** once [`Tracker::reserve`] has been called.
    ///
    /// `dets` is [`crate::detect`]'s output read as fixed-size records.
    /// `sweep_seq` is the velodyne sweep the detections came from, carried
    /// through the chain so "consecutive" is exact; `trigger_ns` is that
    /// sweep's forward-facing trigger, which is what `dt` is measured between.
    pub fn plan(
        &mut self,
        dets: &[[f32; DETECTION_LANES]],
        sweep_seq: i64,
        trigger_ns: i64,
        voxel: VoxelSize,
        range_limit_m: f32,
    ) -> TrackPlan {
        let mut plan = TrackPlan {
            source_detections: dets.len() as u32,
            ..TrackPlan::default()
        };
        // **Exact, no tolerance.** A missing sweep discards every track rather
        // than coasting across a loss the pipeline caused; see the module docs.
        let consecutive = self.prev_sweep_seq == Some(sweep_seq - 1);
        if !consecutive {
            plan.reset = !self.live.is_empty() || self.prev_sweep_seq.is_some();
            self.live.clear();
        }
        let dt = if consecutive {
            ((trigger_ns - self.prev_ns) as f64 * 1e-9).max(0.0)
        } else {
            0.0
        };
        let gate = if consecutive {
            association_gate_m(dt, voxel, range_limit_m)
        } else {
            0.0
        };
        plan.dt_s = dt;
        plan.gate_m = gate;

        self.pairs.clear();
        self.det_taken.clear();
        self.det_taken.resize(dets.len(), false);
        self.trk_taken.clear();
        self.trk_taken.resize(self.live.len(), false);
        self.det_cands.clear();
        self.det_cands.resize(dets.len(), 0);
        self.trk_cands.clear();
        self.trk_cands.resize(self.live.len(), 0);

        let gate2 = (gate * gate) as f32;
        if gate > 0.0 {
            for (ti, t) in self.live.iter().enumerate() {
                for (di, d) in dets.iter().enumerate() {
                    let dx = d[0] - t.pos[0];
                    let dy = d[1] - t.pos[1];
                    let dz = d[2] - t.pos[2];
                    let d2 = dx * dx + dy * dy + dz * dz;
                    if d2 <= gate2 {
                        self.pairs.push((d2, ti as u32, di as u32));
                        self.det_cands[di] += 1;
                        self.trk_cands[ti] += 1;
                    }
                }
            }
        }
        plan.ambiguous = self.det_cands.iter().filter(|&&n| n > 1).count() as u32;
        plan.contested = self.trk_cands.iter().filter(|&&n| n > 1).count() as u32;
        // `total_cmp` and then both indices: a total order, so the greedy pass
        // cannot depend on the order pairs were generated in and a NaN cannot
        // make the comparison partial. NaN distances are impossible here (the
        // detector rejects non-finite voxels) but "impossible" is not a sort
        // predicate.
        // `sort_unstable_by`, and the choice is load-bearing twice over. The
        // key `(distance, track index, detection index)` is UNIQUE -- no two
        // pairs share both indices -- so stability cannot change the result
        // and asking for it buys nothing. It also allocates: Rust's stable
        // sort takes a scratch buffer, and this sort runs INSIDE the stage's
        // measured window, so it was charging ~10 KB per sweep to the number
        // that is supposed to read 0 for a borrowed input.
        self.pairs
            .sort_unstable_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));

        for &(_, ti, di) in &self.pairs {
            if self.trk_taken[ti as usize] || self.det_taken[di as usize] {
                continue;
            }
            self.trk_taken[ti as usize] = true;
            self.det_taken[di as usize] = true;
            let d = &dets[di as usize];
            let t = &mut self.live[ti as usize];
            t.pos = [d[0], d[1], d[2]];
            t.lo = [d[3], d[4], d[5]];
            t.hi = [d[6], d[7], d[8]];
            t.last_ns = trigger_ns;
            t.observations += 1;
            t.misses = 0;
            plan.matched += 1;
        }

        // Unmatched tracks coast once, then die. Counted off `trk_taken`
        // rather than off a timestamp comparison: the index is what the
        // association actually decided, and a track whose `last_ns` happened
        // to equal this sweep's would otherwise be read as observed.
        {
            let taken = &self.trk_taken;
            for (i, t) in self.live.iter_mut().enumerate() {
                if !taken[i] {
                    t.misses += 1;
                }
            }
        }
        let before = self.live.len();
        self.live.retain(|t| t.misses <= MAX_MISSES);
        plan.died = (before - self.live.len()) as u32;
        plan.coasted = self.live.iter().filter(|t| t.misses > 0).count() as u32;

        for (di, d) in dets.iter().enumerate() {
            if self.det_taken[di] {
                continue;
            }
            self.live.push(Live {
                id: self.next_id,
                pos: [d[0], d[1], d[2]],
                lo: [d[3], d[4], d[5]],
                hi: [d[6], d[7], d[8]],
                first_pos: [d[0], d[1], d[2]],
                first_ns: trigger_ns,
                last_ns: trigger_ns,
                observations: 1,
                misses: 0,
            });
            self.next_id += 1;
            plan.born += 1;
        }

        self.prev_sweep_seq = Some(sweep_seq);
        self.prev_ns = trigger_ns;
        plan.live = self.live.len() as u32;
        plan.emitted = self
            .live
            .iter()
            .filter(|t| t.observations >= MIN_OBSERVATIONS)
            .count() as u32;
        plan
    }

    /// Phase 2: write the emitted tracks into one contiguous f32 buffer and
    /// wrap it as Arrow **without copying it**, with no camera detections to
    /// associate: every track's population is
    /// [`crate::fuse::POPULATION_UNFUSED`]. See [`Tracker::build_fused`].
    pub fn build(
        &self,
        plan: &TrackPlan,
        meta: &TrackMeta,
        calib: Option<&Calib>,
        schema: &Arc<Schema>,
    ) -> Result<(RecordBatch, usize, u32), TrackError> {
        let b = self.build_fused(plan, meta, calib, None, schema)?;
        Ok((b.batch, b.storage_id, b.in_frame))
    }

    /// Phase 2, with the fusion: write the emitted tracks into one contiguous
    /// f32 buffer, associate them with the paired frame's detections **in
    /// that buffer** ([`crate::fuse::Fuser::associate`] fills lanes 21-25),
    /// and wrap it as Arrow **without copying it**.
    ///
    /// `calib` projects each track's 3D box into the paired camera frame; pass
    /// `None` when the sweep has no usable pair, and every track's image
    /// rectangle is written as four zeros — the payload's own encoding of "not
    /// in frame", which needs no extra lane because a real rectangle always
    /// has area.
    ///
    /// `fusion` is the paired frame's detections and the association's
    /// scratch, or `None` when the camera half carries no detections (`proc`'s
    /// bare frame reference under `--detector off`): the tracks are then all
    /// [`crate::fuse::POPULATION_UNFUSED`] and `camera_only` is empty. The
    /// association must see the image boxes, which is why it runs here, after
    /// the projection and before the buffer is frozen, rather than on the
    /// finished batch. Call [`crate::fuse::Fuser::reserve`] outside the
    /// measured window first, or its scratch grows inside it.
    pub fn build_fused(
        &self,
        plan: &TrackPlan,
        meta: &TrackMeta,
        calib: Option<&Calib>,
        fusion: Option<(&DetView<'_>, &mut Fuser)>,
        schema: &Arc<Schema>,
    ) -> Result<Built, TrackError> {
        let emitted: Vec<&Live> = self
            .live
            .iter()
            .filter(|t| t.observations >= MIN_OBSERVATIONS)
            .collect();
        if emitted.len() != plan.emitted as usize {
            return Err(TrackError::PlanMismatch {
                planned: plan.emitted,
                got: emitted.len(),
            });
        }
        // The one allocation for the tracks, and exactly their size:
        // `from_len_zeroed` takes a layout of precisely `len`, so the bytes
        // this stage requests for them and the bytes the column carries are
        // the same number.
        let mut buf = MutableBuffer::from_len_zeroed(emitted.len() * TRACK_BYTES);
        let addr = buf.as_slice().as_ptr() as usize;
        if !addr.is_multiple_of(std::mem::align_of::<f32>()) {
            return Err(TrackError::Sweep(SweepError::Misaligned { addr }));
        }
        let mut in_frame = 0u32;
        {
            let out = buf.typed_data_mut::<f32>();
            for (w, t) in emitted.iter().enumerate() {
                let lane = &mut out[w * TRACK_LANES..][..TRACK_LANES];
                let v = t.velocity();
                lane[0..3].copy_from_slice(&t.pos);
                lane[3..6].copy_from_slice(&v);
                // Exact below 2^24; see [`TRACK_BYTES`].
                lane[6] = t.id as f32;
                lane[7] = t.observations as f32;
                lane[8] = t.misses as f32;
                lane[9..12].copy_from_slice(&t.lo);
                lane[12..15].copy_from_slice(&t.hi);
                if let Some(b) = calib.and_then(|c| c.project_box(t.lo, t.hi)) {
                    lane[15] = b.x0;
                    lane[16] = b.y0;
                    lane[17] = b.x1;
                    lane[18] = b.y1;
                    in_frame += 1;
                }
                // Both in integer ns until here, on the sensor clock; see the
                // module docs. `since_seen` is 0 exactly when the track was
                // matched this sweep, because a match sets `last_ns` to the
                // trigger that `prev_ns` also holds.
                lane[19] = ns_to_lane_s(t.age_ns());
                lane[20] = ns_to_lane_s(self.prev_ns - t.last_ns);
                // Nothing associated yet. The fusion below rewrites lanes
                // 21-25 of every track when there is anything to associate
                // with; otherwise this is the record's final word.
                lane[LANE_DET_INDEX] = -1.0;
                lane[LANE_POPULATION] = POPULATION_UNFUSED as f32;
            }
        }
        let (camera_only, fuse) = match fusion {
            Some((dets, fuser)) => {
                let (rows, _) = buf.typed_data_mut::<f32>().as_chunks_mut::<TRACK_LANES>();
                let fuse = fuser.associate(rows, dets);
                (build_camera_only(Some(dets), fuser.taken())?, fuse)
            }
            None => (build_camera_only(None, &[])?, FusePlan::default()),
        };
        let (arr, storage_id, n) = build_tracks_array(Buffer::from(buf))?;
        let columns: Vec<ArrayRef> = vec![
            Arc::new(UInt32Array::from(vec![n as u32])),
            Arc::new(StringArray::from(vec![TRACK_FORMAT])),
            Arc::new(Int64Array::from(vec![meta.trigger_ns])),
            Arc::new(Int64Array::from(vec![meta.sweep_seq])),
            Arc::new(UInt32Array::from(vec![plan.source_detections])),
            Arc::new(UInt32Array::from(vec![meta.source_voxels])),
            Arc::new(UInt32Array::from(vec![meta.source_points])),
            Arc::new(Float32Array::from(vec![plan.gate_m as f32])),
            Arc::new(Float32Array::from(vec![plan.dt_s as f32])),
            Arc::new(Int64Array::from(vec![meta
                .cam
                .map_or(-1, |c| c.seq as i64)])),
            Arc::new(Int64Array::from(vec![meta.cam.map_or(0, |c| c.tov_ns)])),
            Arc::new(Int64Array::from(vec![meta
                .cam
                .map_or(0, |c| c.tov_ns - meta.trigger_ns)])),
            Arc::new(StringArray::from(vec![meta.pairing.name()])),
            Arc::new(UInt32Array::from(vec![meta.cam.map_or(0, |c| c.width)])),
            Arc::new(UInt32Array::from(vec![meta.cam.map_or(0, |c| c.height)])),
            Arc::new(BooleanArray::from(vec![meta.pairing == Pairing::Paired])),
            arr,
            Arc::new(UInt32Array::from(vec![fuse.fused])),
            Arc::new(UInt32Array::from(vec![fuse.contested])),
            Arc::new(UInt32Array::from(vec![fuse.crowded])),
            camera_only,
        ];
        Ok(Built {
            batch: RecordBatch::try_new(Arc::clone(schema), columns).map_err(SweepError::Arrow)?,
            storage_id,
            in_frame,
            fuse,
        })
    }
}

/// What [`Tracker::build_fused`] made: the batch, the address of its track
/// buffer (the zero-copy proof), how many tracks landed in the camera frame,
/// and what the association did.
#[derive(Debug)]
pub struct Built {
    /// The one-row track batch.
    pub batch: RecordBatch,
    /// Address of the `tracks` column's buffer.
    pub storage_id: usize,
    /// Tracks with an image rectangle.
    pub in_frame: u32,
    /// The association's counts: all zero, with `ran == false`, when there
    /// was nothing to associate with.
    pub fuse: FusePlan,
}

/// Everything about one sweep that is not a track: what it came from, and
/// which camera frame it was matched to.
#[derive(Clone, Copy, Debug)]
pub struct TrackMeta {
    /// The sweep's forward-facing trigger, ns on the sensor clock.
    pub trigger_ns: i64,
    /// The velodyne sweep number, carried through the chain.
    pub sweep_seq: i64,
    /// Voxels behind this sweep's detections.
    pub source_voxels: u32,
    /// Raw laser returns behind those voxels.
    pub source_points: u32,
    /// The camera frame the sweep was paired with, if any.
    pub cam: Option<CamRef>,
    /// How the pairing went.
    pub pairing: Pairing,
}

/// The one-row `tracks` column, wrapping `buf` **without copying it**.
fn build_tracks_array(buf: Buffer) -> Result<(ArrayRef, usize, usize), SweepError> {
    let len = buf.len();
    if !len.is_multiple_of(TRACK_BYTES) {
        return Err(SweepError::NotWholePoints { got: len });
    }
    let storage_id = buf.as_ptr() as usize;
    if !storage_id.is_multiple_of(std::mem::align_of::<f32>()) {
        return Err(SweepError::Misaligned { addr: storage_id });
    }
    let n = len / TRACK_BYTES;
    let values = Float32Array::new(ScalarBuffer::<f32>::from(buf), None);
    let rec = FixedSizeListArray::try_new(
        track_value_field(),
        TRACK_VALUES,
        Arc::new(values) as ArrayRef,
        None,
    )?;
    let offsets = OffsetBuffer::<i64>::new(ScalarBuffer::<i64>::from(vec![0i64, n as i64]));
    let arr = LargeListArray::try_new(track_field(), offsets, Arc::new(rec), None)?;
    Ok((Arc::new(arr) as ArrayRef, storage_id, n))
}

/// Value of the `ref_format` column of a camera-reference batch.
///
/// **Deliberately not a detection format.** The stream it travels on is
/// [`pipes_core::sample::StreamId::CAM_DET`], whose name says "det", and this
/// string is what stops that name from becoming a claim: the payload is a
/// reference to a frame, not anything found in one.
pub const CAM_REF_FORMAT: &str = "cam_frame_ref_v1";

/// `frame_seq i64, tov_ns i64, width u32, height u32, ref_format Utf8`.
///
/// What the camera consumer finished, and nothing about the picture. A fusion
/// needs the frame's **instant** (to test containment in a sweep's range) and
/// its **bounds** (to clip a projected box); it does not need the pixels,
/// because the image rectangle of a 3D object comes from the calibration and
/// the box, not from looking. So this is a few dozen bytes that travel from
/// `proc` to `track` while the 1,397,250 B of pixels stay where they were.
///
/// It carries no shared buffer, so `Sample::storage_id` on this stream is 0
/// rather than an address: there is nothing here that could be copied, and a
/// zero-copy claim about a payload this size would be noise.
pub fn cam_ref_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("frame_seq", DataType::Int64, false),
        Field::new("tov_ns", DataType::Int64, false),
        Field::new("width", DataType::UInt32, false),
        Field::new("height", DataType::UInt32, false),
        Field::new("ref_format", DataType::Utf8, false),
    ]))
}

/// Builds the one-row camera reference `proc` hands to the fusion.
pub fn build_cam_ref_batch(
    r: CamRef,
    schema: &Arc<Schema>,
) -> Result<RecordBatch, arrow::error::ArrowError> {
    let columns: Vec<ArrayRef> = vec![
        Arc::new(Int64Array::from(vec![r.seq as i64])),
        Arc::new(Int64Array::from(vec![r.tov_ns])),
        Arc::new(UInt32Array::from(vec![r.width])),
        Arc::new(UInt32Array::from(vec![r.height])),
        Arc::new(StringArray::from(vec![CAM_REF_FORMAT])),
    ];
    RecordBatch::try_new(Arc::clone(schema), columns)
}

/// Reads a camera reference back out of its batch, or `None` if the batch is
/// not one.
///
/// Two payloads carry one: the bare reference `proc` builds
/// ([`CAM_REF_FORMAT`]) and the frozen detector's batch
/// ([`crate::camdet::CAM_DET_FORMAT`]), whose first four columns are the same
/// reference with the same names, so the pairing reads the frame's instant
/// the same way off either.
///
/// Checks `ref_format` rather than trusting the stream id, so a payload that
/// is not a frame reference cannot be read as one — the same rule
/// [`crate::detect::DETECTION_FORMAT`] exists for.
pub fn read_cam_ref(batch: &RecordBatch) -> Option<CamRef> {
    let fmt = str_col(batch, "ref_format")?;
    if fmt != CAM_REF_FORMAT && fmt != crate::camdet::CAM_DET_FORMAT {
        return None;
    }
    Some(CamRef {
        seq: u64::try_from(i64_col(batch, "frame_seq")?).ok()?,
        tov_ns: i64_col(batch, "tov_ns")?,
        width: u32_col(batch, "width")?,
        height: u32_col(batch, "height")?,
    })
}

/// The image rectangle of one track record, or `None` when it is not in frame.
///
/// The payload encodes "not in frame" as four zeros, and this is the one place
/// that knows it — so no consumer has to remember the convention or invent its
/// own sentinel.
pub fn track_image_box(t: &[f32; TRACK_LANES]) -> Option<ImageBox> {
    let b = ImageBox {
        x0: t[15],
        y0: t[16],
        x1: t[17],
        y1: t[18],
    };
    (b.w() > 0.0 && b.h() > 0.0).then_some(b)
}

/// Sweeps one track record was matched in (lane 7): the emission rule's
/// count, not its age.
pub fn track_observations(t: &[f32; TRACK_LANES]) -> u32 {
    t[7] as u32
}

/// Last seen minus first seen, seconds on the sensor clock (lane 19): the
/// track's age.
pub fn track_age_s(t: &[f32; TRACK_LANES]) -> f32 {
    t[19]
}

/// This sweep's trigger minus the track's last observation, seconds (lane
/// 20): 0 when it was seen this sweep.
pub fn track_since_seen_s(t: &[f32; TRACK_LANES]) -> f32 {
    t[20]
}

/// The population code of one track record (lane 25): see [`crate::fuse`].
pub fn track_population(t: &[f32; TRACK_LANES]) -> u32 {
    t[LANE_POPULATION] as u32
}

/// The camera detection one track record was fused with, as `(det_index,
/// iou, class_id, confidence)`, or `None` for a track that was not fused.
///
/// Read off the population lane first: lanes 21-24 of an unfused track are
/// `-1, 0, 0, 0`, and class 0 is "person", so the lanes alone cannot be
/// trusted to say "no class".
pub fn track_detection(t: &[f32; TRACK_LANES]) -> Option<(u32, f32, u32, f32)> {
    (t[LANE_POPULATION] == POPULATION_FUSED as f32 && t[LANE_DET_INDEX] >= 0.0).then(|| {
        (
            t[LANE_DET_INDEX] as u32,
            t[LANE_FUSE_IOU],
            t[LANE_CLASS_ID] as u32,
            t[LANE_CONFIDENCE],
        )
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    const V: f32 = 0.2;
    const RANGE: f32 = 30.0;
    /// 103.27 ms, drive_0005's measured mean sweep span.
    const DT_NS: i64 = 103_270_000;

    fn size() -> VoxelSize {
        VoxelSize::new(V).unwrap()
    }

    /// One detection record at `(x, y, z)` with a half-metre box round it.
    fn det(x: f32, y: f32, z: f32) -> [f32; DETECTION_LANES] {
        let mut d = [0.0f32; DETECTION_LANES];
        d[0] = x;
        d[1] = y;
        d[2] = z;
        d[3] = x - 0.5;
        d[4] = y - 0.5;
        d[5] = z - 0.5;
        d[6] = x + 0.5;
        d[7] = y + 0.5;
        d[8] = z + 0.5;
        d[9] = 10.0;
        d[10] = 0.3;
        d
    }

    fn meta(seq: i64, ns: i64) -> TrackMeta {
        TrackMeta {
            trigger_ns: ns,
            sweep_seq: seq,
            source_voxels: 0,
            source_points: 0,
            cam: None,
            pairing: Pairing::Absent,
        }
    }

    fn plan_at(tk: &mut Tracker, dets: &[[f32; DETECTION_LANES]], seq: i64) -> TrackPlan {
        tk.reserve(dets.len());
        tk.plan(dets, seq, seq * DT_NS, size(), RANGE)
    }

    /// The gate is arithmetic over stated constants, so this pins the number
    /// the module docs quote. If a term is dropped the total moves.
    #[test]
    fn the_gate_is_the_sum_of_its_three_stated_terms() {
        let dt = 0.103_27;
        let g = association_gate_m(dt, size(), RANGE);
        assert!((g - (2.101 + 0.781 + 0.200)).abs() < 0.005, "gate = {g}");
        // It scales with dt -- the property a fixed gate does not have, and the
        // one that stops an eviction from silently retuning the tracker.
        let wide = association_gate_m(dt * 6.0, size(), RANGE);
        assert!(wide > g * 5.0, "{wide} is not ~6x {g}");
        // And the ego-yaw term really is a quarter of it: the cost of not
        // replaying oxts, as a number rather than a claim.
        let no_yaw = V_REL_MAX_MPS * dt + f64::from(V);
        assert!(
            (g - no_yaw) / g > 0.2,
            "the yaw term is only {:.1}% of the gate",
            (g - no_yaw) / g * 100.0
        );
    }

    /// A static object seen in three consecutive sweeps keeps one id and
    /// reports ~zero velocity.
    #[test]
    fn a_static_object_keeps_its_id_and_has_no_velocity() {
        let mut tk = Tracker::new();
        let d = [det(10.0, 0.0, -1.0)];
        plan_at(&mut tk, &d, 0);
        plan_at(&mut tk, &d, 1);
        let p = plan_at(&mut tk, &d, 2);
        assert_eq!((p.live, p.emitted, p.born, p.matched), (1, 1, 0, 1));
        let (batch, _, _) = tk
            .build(&p, &meta(2, 2 * DT_NS), None, &track_schema())
            .unwrap();
        let rows = track_rows(&batch);
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0][6], 0.0, "the id was reissued");
        assert_eq!(track_observations(&rows[0]), 3);
        // Two intervals, first seen to last seen, and seen this sweep.
        let want = (2 * DT_NS) as f32 * 1e-9;
        assert!((track_age_s(&rows[0]) - want).abs() < 1e-6, "age");
        assert_eq!(track_since_seen_s(&rows[0]), 0.0);
        let speed = (rows[0][3].powi(2) + rows[0][4].powi(2) + rows[0][5].powi(2)).sqrt();
        assert!(speed < 1e-3, "static object moving at {speed} m/s");
    }

    /// An object closing at a known speed gets that speed back, over the
    /// track's whole life rather than the last sweep.
    #[test]
    fn a_closing_object_reports_its_speed() {
        let mut tk = Tracker::new();
        let dt = DT_NS as f32 * 1e-9;
        // 4 m/s towards the sensor.
        for k in 0..5i64 {
            let x = 20.0 - 4.0 * dt * k as f32;
            let d = [det(x, 0.0, -1.0)];
            let p = plan_at(&mut tk, &d, k);
            if k == 4 {
                let (b, _, _) = tk
                    .build(&p, &meta(k, k * DT_NS), None, &track_schema())
                    .unwrap();
                let r = track_rows(&b)[0];
                assert!((r[3] + 4.0).abs() < 0.01, "vx = {}, expected -4", r[3]);
                assert_eq!(track_observations(&r), 5);
                // The velocity's baseline and the age are one number.
                assert!((track_age_s(&r) - 4.0 * dt).abs() < 1e-6, "age");
            }
        }
    }

    /// A gap in the sweep sequence kills every track. This is the refusal to
    /// hide an upstream loss, and it has to be a test or it is a comment.
    #[test]
    fn a_missing_sweep_resets_the_tracker() {
        let mut tk = Tracker::new();
        let d = [det(10.0, 0.0, -1.0)];
        plan_at(&mut tk, &d, 0);
        plan_at(&mut tk, &d, 1);
        // Sweep 2 never arrived.
        let p = plan_at(&mut tk, &d, 3);
        assert!(p.reset, "a non-consecutive sweep did not reset the tracker");
        assert_eq!(p.emitted, 0, "a reset track was emitted with an age");
        assert_eq!(p.born, 1);
        assert_eq!(p.dt_s, 0.0, "dt across a gap is not a measurement");
        // Positive control: the SAME sequence without the gap does not reset.
        let mut ok = Tracker::new();
        plan_at(&mut ok, &d, 0);
        plan_at(&mut ok, &d, 1);
        let q = plan_at(&mut ok, &d, 2);
        assert!(!q.reset);
        assert_eq!(q.emitted, 1);
    }

    /// One unmatched sweep coasts; two kill the track.
    #[test]
    fn a_track_coasts_once_and_then_dies() {
        let mut tk = Tracker::new();
        let near = [det(10.0, 0.0, -1.0)];
        let far = [det(10.0, 25.0, -1.0)];
        plan_at(&mut tk, &near, 0);
        plan_at(&mut tk, &near, 1);
        let p = plan_at(&mut tk, &far, 2);
        assert_eq!((p.coasted, p.died), (1, 0), "the first miss killed it");
        assert_eq!(p.live, 2);
        let q = plan_at(&mut tk, &far, 3);
        assert_eq!(
            (q.coasted, q.died),
            (0, 1),
            "the second miss did not kill it"
        );
        assert_eq!(q.live, 1);
    }

    /// A coasted track keeps its LAST OBSERVED position: it is not propagated
    /// forward by its velocity, so nothing in the payload is a prediction.
    #[test]
    fn a_coasted_track_is_not_extrapolated() {
        let mut tk = Tracker::new();
        let dt = DT_NS as f32 * 1e-9;
        for k in 0..3i64 {
            let d = [det(20.0 - 4.0 * dt * k as f32, 0.0, -1.0)];
            plan_at(&mut tk, &d, k);
        }
        let last_x = 20.0 - 4.0 * dt * 2.0;
        // Sweep 3 sees nothing near it.
        let p = plan_at(&mut tk, &[det(5.0, 20.0, -1.0)], 3);
        assert_eq!(p.coasted, 1);
        let (b, _, _) = tk
            .build(&p, &meta(3, 3 * DT_NS), None, &track_schema())
            .unwrap();
        let coasted = track_rows(&b)
            .iter()
            .find(|r| r[8] > 0.0)
            .expect("no coasted track in the payload");
        assert!(
            (coasted[0] - last_x).abs() < 1e-4,
            "coasted x = {} but the last OBSERVATION was {last_x}",
            coasted[0]
        );
        assert_eq!(coasted[8], 1.0, "misses lane");
    }

    /// A one-observation track has no velocity, so it is not emitted.
    #[test]
    fn a_track_is_not_emitted_until_it_has_a_velocity() {
        let mut tk = Tracker::new();
        let p = plan_at(&mut tk, &[det(10.0, 0.0, -1.0)], 0);
        assert_eq!((p.live, p.emitted), (1, 0));
        let (b, _, _) = tk.build(&p, &meta(0, 0), None, &track_schema()).unwrap();
        assert!(track_rows(&b).is_empty());
        assert_eq!(track_count(&b), Some(0));
    }

    /// Age is first seen to last seen on the sweeps' own triggers, and a
    /// sweep the track coasted through is in it: the count of observations
    /// stops while the track's life does not. This is the case a count got
    /// wrong.
    #[test]
    fn age_is_sensor_time_and_a_coasted_sweep_is_part_of_it() {
        let mut tk = Tracker::new();
        let near = [det(10.0, 0.0, -1.0)];
        let elsewhere = [det(10.0, 25.0, -1.0)];
        plan_at(&mut tk, &near, 0);
        plan_at(&mut tk, &near, 1);
        // Sweep 2 misses it: the track coasts, and says for how long.
        let p = plan_at(&mut tk, &elsewhere, 2);
        let (b, _, _) = tk
            .build(&p, &meta(2, 2 * DT_NS), None, &track_schema())
            .unwrap();
        let coasted = *track_rows(&b)
            .iter()
            .find(|r| r[6] == 0.0)
            .expect("the coasting track was not emitted");
        assert_eq!(coasted[8], 1.0, "misses");
        let dt = DT_NS as f32 * 1e-9;
        assert!((track_since_seen_s(&coasted) - dt).abs() < 1e-6);
        assert!(
            (track_age_s(&coasted) - dt).abs() < 1e-6,
            "age moved while unseen"
        );
        // Sweep 3 sees it again: three observations over three intervals.
        let p = plan_at(&mut tk, &near, 3);
        let (b, _, _) = tk
            .build(&p, &meta(3, 3 * DT_NS), None, &track_schema())
            .unwrap();
        let r = *track_rows(&b).iter().find(|r| r[6] == 0.0).unwrap();
        assert_eq!(track_observations(&r), 3, "observations");
        assert!(
            (track_age_s(&r) - 3.0 * dt).abs() < 1e-6,
            "age {} is not the three intervals it lived; a count would say two",
            track_age_s(&r)
        );
        assert_eq!(track_since_seen_s(&r), 0.0);
    }

    /// The age is read off the instants the sweeps really carry, not a count
    /// times a nominal period: triggers 100 ms and then 150 ms apart give
    /// 0.25 s.
    #[test]
    fn age_is_read_off_the_triggers_not_a_nominal_period() {
        let mut tk = Tracker::new();
        let d = [det(10.0, 0.0, -1.0)];
        let mut last = None;
        for (seq, ns) in [(0i64, 0i64), (1, 100_000_000), (2, 250_000_000)] {
            tk.reserve(d.len());
            let p = tk.plan(&d, seq, ns, size(), RANGE);
            last = Some((p, ns));
        }
        let (p, ns) = last.unwrap();
        let (b, _, _) = tk.build(&p, &meta(2, ns), None, &track_schema()).unwrap();
        let r = track_rows(&b)[0];
        assert!(
            (track_age_s(&r) - 0.25).abs() < 1e-6,
            "age {}",
            track_age_s(&r)
        );
        assert_eq!(track_observations(&r), 3);
    }

    /// `since_seen_s` and `misses` are two readings of one fact and must never
    /// disagree: zero exactly when the track was matched this sweep.
    #[test]
    fn since_seen_is_zero_exactly_when_misses_is() {
        let mut tk = Tracker::new();
        let a = [det(10.0, 0.0, -1.0), det(16.0, 3.0, -1.0)];
        let b = [det(10.0, 0.0, -1.0), det(5.0, -20.0, -1.0)];
        let mut seen = (0, 0);
        for (k, dets) in [&a, &a, &b, &a, &b, &b].iter().enumerate() {
            let k = k as i64;
            let p = plan_at(&mut tk, *dets, k);
            let (batch, _, _) = tk
                .build(&p, &meta(k, k * DT_NS), None, &track_schema())
                .unwrap();
            for r in track_rows(&batch) {
                assert_eq!(
                    r[8] == 0.0,
                    track_since_seen_s(r) == 0.0,
                    "misses {} against since_seen {}",
                    r[8],
                    track_since_seen_s(r)
                );
                if r[8] == 0.0 {
                    seen.0 += 1;
                } else {
                    seen.1 += 1;
                }
            }
        }
        // Both halves happened, or the equality above held over one of them.
        assert!(seen.0 > 0 && seen.1 > 0, "{seen:?}");
    }

    /// Two tracks inside one detection's gate are counted, not hidden, and the
    /// nearer one wins.
    #[test]
    fn competing_associations_are_counted_and_the_nearest_wins() {
        let mut tk = Tracker::new();
        let two = [det(10.0, 0.0, -1.0), det(11.0, 0.0, -1.0)];
        plan_at(&mut tk, &two, 0);
        // One detection between them: both tracks are inside the 3.08 m gate.
        let p = plan_at(&mut tk, &[det(10.4, 0.0, -1.0)], 1);
        assert_eq!(p.ambiguous, 1, "the contest was not counted");
        assert_eq!(p.matched, 1);
        let (b, _, _) = tk
            .build(&p, &meta(1, DT_NS), None, &track_schema())
            .unwrap();
        let r = track_rows(&b);
        assert_eq!(r.len(), 1);
        assert_eq!(r[0][6], 0.0, "the FARTHER track won the association");
    }

    /// Permuting the input must not change the output bytes. The association
    /// is a sort and a greedy pass, so this is the property that says the sort
    /// is total.
    #[test]
    fn the_same_detections_in_a_different_order_give_the_same_tracks() {
        let scene = [
            det(10.0, 1.0, -1.0),
            det(14.0, -2.0, -1.0),
            det(7.5, 0.5, -0.8),
            det(22.0, 4.0, -1.2),
        ];
        let run = |order: &[usize]| -> Vec<[f32; TRACK_LANES]> {
            let mut tk = Tracker::new();
            for k in 0..3i64 {
                let d: Vec<_> = order.iter().map(|&i| scene[i]).collect();
                let p = plan_at(&mut tk, &d, k);
                if k == 2 {
                    let (b, _, _) = tk
                        .build(&p, &meta(k, k * DT_NS), None, &track_schema())
                        .unwrap();
                    return track_rows(&b).to_vec();
                }
            }
            Vec::new()
        };
        let a = run(&[0, 1, 2, 3]);
        let b = run(&[3, 1, 0, 2]);
        assert_eq!(a.len(), 4, "the fixture produced no tracks to compare");
        // Ids depend on the order detections are first seen, which is the input
        // order by construction; everything else must match once sorted by
        // position.
        let key = |r: &[f32; TRACK_LANES]| (r[0].to_bits(), r[1].to_bits());
        let mut a2: Vec<_> = a
            .iter()
            .map(|r| (key(r), r[3].to_bits(), r[7].to_bits()))
            .collect();
        let mut b2: Vec<_> = b
            .iter()
            .map(|r| (key(r), r[3].to_bits(), r[7].to_bits()))
            .collect();
        a2.sort();
        b2.sort();
        assert_eq!(a2, b2);
    }

    /// The batch carries the whole chain's provenance, so the shrink is
    /// recomputable from the sample in hand.
    #[test]
    fn the_batch_carries_the_camera_half_and_the_whole_chain() {
        let mut tk = Tracker::new();
        let d = [det(10.0, 0.0, -1.0)];
        plan_at(&mut tk, &d, 0);
        let p = plan_at(&mut tk, &d, 1);
        let m = TrackMeta {
            trigger_ns: 1_000_000_000,
            sweep_seq: 1,
            source_voxels: 30_754,
            source_points: 121_732,
            cam: Some(CamRef {
                seq: 41,
                tov_ns: 1_010_503_000,
                width: 1242,
                height: 375,
            }),
            pairing: Pairing::Paired,
        };
        let (b, sid, _) = tk.build(&p, &m, None, &track_schema()).unwrap();
        assert_eq!(cam_seq(&b), Some(41));
        assert_eq!(cam_tov_ns(&b), Some(1_010_503_000));
        assert_eq!(pair_age_ns(&b), Some(10_503_000));
        assert_eq!(pair_outcome(&b), Some("paired"));
        assert_eq!(paired(&b), Some(true));
        assert_eq!(source_detections(&b), Some(1));
        assert_eq!(source_voxel_count(&b), Some(30_754));
        assert_eq!(source_point_count(&b), Some(121_732));
        assert_eq!(track_sweep_seq(&b), Some(1));
        assert_eq!(track_format(&b), Some(TRACK_FORMAT));
        assert_eq!(track_storage_id(&b), Some(sid));
        // Unpaired writes -1, which cannot be confused with frame 0.
        let (b2, _, _) = tk
            .build(&p, &meta(1, DT_NS), None, &track_schema())
            .unwrap();
        assert_eq!(cam_seq(&b2), Some(-1));
        assert_eq!(paired(&b2), Some(false));
        assert_eq!(pair_outcome(&b2), Some("pair_absent"));
    }

    /// 21 lanes at 4 bytes is 84, and one track record is exactly that. The
    /// constant and the layout cannot drift apart silently.
    #[test]
    fn a_track_record_is_its_documented_size() {
        assert_eq!(TRACK_LANES * std::mem::size_of::<f32>(), TRACK_BYTES);
        let mut tk = Tracker::new();
        let d = [det(10.0, 0.0, -1.0), det(16.0, 3.0, -1.0)];
        plan_at(&mut tk, &d, 0);
        let p = plan_at(&mut tk, &d, 1);
        let (b, _, _) = tk
            .build(&p, &meta(1, DT_NS), None, &track_schema())
            .unwrap();
        let carried = pipes_core::sample::payload_bytes(&b);
        assert!(
            carried >= 2 * TRACK_BYTES,
            "{carried} B carried for two tracks"
        );
        // The metadata columns are a fixed cost, not a fraction of the payload.
        let mut tk2 = Tracker::new();
        let d4 = [
            det(10.0, 0.0, -1.0),
            det(16.0, 3.0, -1.0),
            det(20.0, -4.0, -1.0),
            det(24.0, 6.0, -1.0),
        ];
        plan_at(&mut tk2, &d4, 0);
        let p4 = plan_at(&mut tk2, &d4, 1);
        let (b4, _, _) = tk2
            .build(&p4, &meta(1, DT_NS), None, &track_schema())
            .unwrap();
        assert_eq!(
            pipes_core::sample::payload_bytes(&b4) - carried,
            2 * TRACK_BYTES,
            "the per-track cost is not TRACK_BYTES"
        );
    }

    /// A full-size KITTI calibration, written to a temp dir and read back
    /// the way the run reads it.
    fn kitti_calib() -> (tempfile::TempDir, Calib) {
        let dir = tempfile::tempdir().unwrap();
        crate::testing::write_calib_fixture_at(dir.path(), "2011_09_26", 1242, 375).unwrap();
        let c = Calib::load(dir.path(), "2011_09_26").unwrap();
        (dir, c)
    }

    /// A one-row `CAM_DET` batch holding `boxes`, each a car at 0.9.
    fn cam_dets(boxes: &[ImageBox]) -> RecordBatch {
        let dets: Vec<crate::camdet::Det> = boxes
            .iter()
            .enumerate()
            .map(|(i, b)| crate::camdet::Det {
                x0: b.x0,
                y0: b.y0,
                x1: b.x1,
                y1: b.y1,
                class_id: 2,
                score: 0.9,
                row: i as u32,
            })
            .collect();
        let r = CamRef {
            seq: 1,
            tov_ns: DT_NS,
            width: 1242,
            height: 375,
        };
        crate::camdet::build_cam_det_batch(
            r,
            &dets,
            &crate::camdet::cam_det_schema(crate::camdet::MODEL_SHA256),
        )
        .unwrap()
    }

    /// Two tracks ahead of the vehicle, both in frame, observed twice.
    fn two_tracks_in_frame() -> (Tracker, TrackPlan) {
        let mut tk = Tracker::new();
        let d = [det(12.0, 1.5, -0.8), det(20.0, -3.0, -0.8)];
        plan_at(&mut tk, &d, 0);
        let p = plan_at(&mut tk, &d, 1);
        assert_eq!(p.emitted, 2);
        (tk, p)
    }

    /// Without detections every track is UNFUSED -- nothing looked -- and the
    /// batch says so in its lanes and its columns.
    #[test]
    fn build_without_detections_leaves_every_track_unfused() {
        let (_d, calib) = kitti_calib();
        let (tk, p) = two_tracks_in_frame();
        let b = tk
            .build_fused(&p, &meta(1, DT_NS), Some(&calib), None, &track_schema())
            .unwrap();
        assert!(!b.fuse.ran);
        let rows = track_rows(&b.batch);
        assert_eq!(rows.len(), 2);
        for r in rows {
            assert!(
                track_image_box(r).is_some(),
                "the fixture track is not in frame"
            );
            assert_eq!(track_population(r), crate::fuse::POPULATION_UNFUSED);
            assert_eq!(track_detection(r), None);
            assert_eq!(r[LANE_DET_INDEX], -1.0);
        }
        assert_eq!(fused_count(&b.batch), Some(0));
        assert!(crate::fuse::camera_only_rows(&b.batch).is_empty());
    }

    /// With the paired frame's detections, the track whose projected box the
    /// detection covers is FUSED -- class and confidence in its own record,
    /// beside the lidar lanes, which do not move -- the other is lidar-only,
    /// and a detection nothing matched is camera-only, with no range.
    #[test]
    fn build_fused_puts_the_camera_s_class_on_the_track_it_covers() {
        let (_d, calib) = kitti_calib();
        let (tk, p) = two_tracks_in_frame();
        let plain = tk
            .build_fused(&p, &meta(1, DT_NS), Some(&calib), None, &track_schema())
            .unwrap();
        let near = track_rows(&plain.batch)[0];
        let nb = track_image_box(&near).unwrap();
        // The detector's box: the near track's own image box, a little
        // tighter; and one far from anything.
        let tight = ImageBox {
            x0: nb.x0 + 2.0,
            y0: nb.y0 + 2.0,
            x1: nb.x1 - 2.0,
            y1: nb.y1,
        };
        let lone = ImageBox {
            x0: 1.0,
            y0: 1.0,
            x1: 30.0,
            y1: 40.0,
        };
        let cb = cam_dets(&[lone, tight]);
        let view = crate::camdet::cam_det_view(&cb).unwrap();
        let mut fuser = crate::fuse::Fuser::new();
        fuser.reserve(2, 2);
        let b = tk
            .build_fused(
                &p,
                &meta(1, DT_NS),
                Some(&calib),
                Some((&view, &mut fuser)),
                &track_schema(),
            )
            .unwrap();
        assert!(b.fuse.ran);
        assert_eq!(
            (b.fuse.detections, b.fuse.fused, b.fuse.camera_only),
            (2, 1, 1)
        );
        let rows = track_rows(&b.batch);
        let (di, iou, class, conf) =
            track_detection(&rows[0]).expect("the near track was not fused");
        assert_eq!((di, class, conf), (1, 2, 0.9));
        assert!(iou > 0.5, "{iou}");
        assert_eq!(
            track_population(&rows[1]),
            crate::fuse::POPULATION_LIDAR_ONLY
        );
        // The lidar lanes are the unfused build's, bit for bit.
        let plain_rows = track_rows(&plain.batch);
        for (a, z) in rows.iter().zip(plain_rows) {
            assert_eq!(
                a[..21].iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                z[..21].iter().map(|x| x.to_bits()).collect::<Vec<_>>()
            );
        }
        assert_eq!(fused_count(&b.batch), Some(1));
        let co = crate::fuse::camera_only_rows(&b.batch);
        assert_eq!(co.len(), 1);
        assert_eq!(
            co[0][0], 0.0,
            "the lone detection is not the camera-only one"
        );
        // Still one buffer for the tracks, at the address the batch reports.
        assert_eq!(track_storage_id(&b.batch), Some(b.storage_id));
    }

    /// The "not in frame" encoding: four zeros, and the accessor is the one
    /// place that knows it.
    #[test]
    fn a_track_with_no_camera_has_no_image_box() {
        let mut t = [0.0f32; TRACK_LANES];
        assert_eq!(track_image_box(&t), None);
        t[15] = 100.0;
        t[16] = 50.0;
        t[17] = 160.0;
        t[18] = 120.0;
        let b = track_image_box(&t).expect("a real rectangle read as absent");
        assert_eq!((b.w(), b.h()), (60.0, 70.0));
    }
}
