//! The fusion's association: which of the paired camera frame's detections,
//! if any, is the same object as each lidar track -- and the three
//! populations that come out of asking.
//!
//! Before this module the camera contributed an INSTANT and nothing else: a
//! stale frame could stop a sweep from producing an answer, but no pixel ever
//! reached the answer, so no camera input could make one wrong. After it, a
//! fused object carries its distance, velocity and age from the lidar and its
//! class and confidence from the camera ([`crate::track::TRACK_BYTES`], lanes
//! 21-25), and a stale frame can now put a wrong class on a right distance.
//! That is the failure the project exists to make attributable, so this module
//! is written to make its output a function of its two inputs and nothing
//! else.
//!
//! # The rule
//!
//! A track and a detection are the same object when the track's projected
//! image box ([`crate::calib::Calib::project_box`]) and the detection's box
//! overlap with **IoU > 1/2**, tested without dividing as `3 I > A_t + A_d`
//! ([`fused_iou`]) -- the positive-test, divide-last form
//! [`crate::calib::Calib::project`] uses, so a NaN fails it.
//!
//! Only a **fresh** track (`misses == 0`, lane 8) that is **in frame** is a
//! candidate. A coasted track's box is one sweep old -- a stale input of the
//! tracker's own -- and the answer already refuses it for the same reason.
//!
//! # One half is derived, not chosen
//!
//! If a detection D has IoU > 1/2 with two tracks T1 and T2, then more than
//! half of D lies inside T1 and more than half inside T2, so T1 and T2 overlap
//! inside D. Above one half, therefore, a detection has at most one candidate
//! unless the lidar boxes themselves overlap -- and the same holds with the
//! roles swapped. The threshold is the smallest one at which the matching is
//! unique by geometry, so greedy, Hungarian and every processing order give
//! the same answer everywhere except where the inputs overlap, and those
//! cases are COUNTED ([`FusePlan::contested`], [`FusePlan::crowded`]) rather
//! than resolved silently. Below it the choice of algorithm starts to decide
//! the answer: a read-only replay of drive_0005 found 97.2 % of object-shaped
//! in-frame tracks overlapping another track somewhere in the image.
//!
//! What it costs: two boxes of one real object are not fused when one has
//! twice the other's area or more, even nested, and sooner once they are also
//! offset. The projected box is the hull of a 3D box's eight corners, so it
//! is larger than the object for anything not square-on to the camera; it
//! inherits the sweep's intra-sweep ego motion ([`crate::calib::ImageBox`]);
//! and it stops at the ground band `detect` removed. The detector's box is
//! tight and reaches the ground. An error budget built before this module
//! existed (calibration, yaw, ego translation, the ground band; read-only, on
//! drive_0005) put the worst case for a true fresh pair below one half for
//! about the smallest quarter of object-shaped tracks. Every run reports how
//! many tracks were fused and of what shape; nothing here is tuned to raise
//! it.
//!
//! # Greedy, on a total order
//!
//! Candidate pairs are sorted by IoU descending, then track id, then
//! detection index -- a unique key, so `sort_unstable_by` cannot depend on the
//! order pairs were generated in, and `total_cmp` cannot be made partial by a
//! NaN -- and taken greedily, one to one. Track id rather than row, because
//! the id is what names the object across sweeps. The detection index is the
//! detection's rank in its `CAM_DET` batch, which `camdet` writes in a total
//! order ([`crate::camdet::nms`]), so "detection 3" names the same box on
//! every run.
//!
//! # What it deliberately does not do
//!
//! * **Widen with staleness.** The 3D gate in [`crate::track`] scales with
//!   `dt` because `dt` is the real interval between two valid measurements. A
//!   camera frame older than its sweep is an input defect, and widening this
//!   test to absorb it would fuse on data the pipeline does not have. A fixed
//!   test is also what makes staleness show up where it can be counted: as a
//!   falling fused count and a falling IoU.
//! * **Remember a class.** Every sweep is associated afresh from its own
//!   pair. A class carried along a track's life would let one stale frame
//!   poison the object for the rest of its life, and would make this
//!   sweep's answer depend on camera frames it was not paired with.
//! * **Give a camera-only object a distance.** A detection with no track has
//!   a box, a class and a score, and nothing else: it travels in its own
//!   column ([`CAMERA_ONLY_BYTES`]), never as a track record with a
//!   made-up range.
//!
//! # The second camera: a class only where the two agree
//!
//! With the right camera's detections too ([`RightHalf`], [`Fuser::confirm`])
//! the left association above is a proposal rather than a verdict. Each
//! track the left camera matched is projected into the right image and
//! matched there by the same rule; it keeps [`POPULATION_FUSED`] only when a
//! right detection of the **same class** overlaps it by more than half, and
//! otherwise becomes [`POPULATION_ONE_CAMERA`]: a class one camera gave and
//! the other did not confirm, kept in lanes 21-24 for the record and trusted
//! nowhere -- the answer takes a class from a fused track only. A track the
//! left camera missed and the right one matched is one-camera too, with the
//! right detection's class and confidence in lanes 23-24 and `-1` in lane 21,
//! because the index would name a box in the wrong frame.
//!
//! The camera-only population is then held to the same standard: a left
//! detection no track matched is kept only when the right camera's unmatched
//! detections hold its stereo partner -- the same class, on the same image
//! rows (the two cameras are rectified to one plane, so a point is on the
//! same row in both), a box of about the same size, and further left in the
//! right image by a disparity a real depth allows ([`stereo_match`]). The
//! disparity is a depth: [`crate::calib::stereo_range_m`], written in the
//! row's eighth lane, so a camera-only object has a range after all -- from
//! the two cameras, never from a track it does not have. Left detections with
//! no partner are dropped from the answer and counted
//! ([`FusePlan::camera_only_dropped`]).
//!
//! # The populations
//!
//! Every lidar record and every detection of the paired frame ends in exactly
//! one population, and the code travels in lane 25 of the track record and
//! lane 16 of the answer record:
//!
//! | code | population | what it is |
//! |---|---|---|
//! | [`POPULATION_FUSED`] 1 | fused | a fresh in-frame track and a detection, one to one -- and, with the right camera, a detection of the same class in both |
//! | [`POPULATION_LIDAR_ONLY`] 2 | lidar-only | a track, when a detector's output was associated and nothing matched it -- out of frame, coasted, or in frame and unmatched |
//! | [`POPULATION_CAMERA_ONLY`] 3 | camera-only | a detection no track matched; never a record, always a `camera_only` row |
//! | [`POPULATION_ONE_CAMERA`] 4 | one-camera | with the right camera: a track one camera matched and the other did not confirm |
//! | [`POPULATION_UNFUSED`] 0 | unfused | a track, when there was no detector output at all (`--detector off`) |
//!
//! 0 and 2 are kept apart because they are different claims: "the camera
//! looked and saw nothing here" against "nothing looked".

use std::sync::Arc;

use arrow::array::{ArrayRef, AsArray, FixedSizeListArray, Float32Array, LargeListArray};
use arrow::buffer::{Buffer, MutableBuffer, OffsetBuffer, ScalarBuffer};
use arrow::datatypes::{DataType, Field, FieldRef, Float32Type};
use arrow::record_batch::RecordBatch;

use crate::calib::{Calib, ImageBox};
use crate::camdet::DetView;
use crate::track::{
    track_image_box, LANE_CLASS_ID, LANE_CONFIDENCE, LANE_DET_INDEX, LANE_FUSE_IOU,
    LANE_POPULATION, TRACK_LANES,
};
use crate::velo::SweepError;

/// Population code: no detector output to associate with.
pub const POPULATION_UNFUSED: u32 = 0;
/// Population code: a track fused with a detection.
pub const POPULATION_FUSED: u32 = 1;
/// Population code: a track no detection matched, when there were
/// detections to match.
pub const POPULATION_LIDAR_ONLY: u32 = 2;
/// Population code: a detection no track matched.
pub const POPULATION_CAMERA_ONLY: u32 = 3;
/// Population code: a track one camera matched and the other did not
/// confirm. Only with the right camera; see the module docs.
pub const POPULATION_ONE_CAMERA: u32 = 4;

/// The spelling of a population code, as the run's output and `summary.json`
/// use it.
pub fn population_name(p: u32) -> &'static str {
    match p {
        POPULATION_UNFUSED => "unfused",
        POPULATION_FUSED => "fused",
        POPULATION_LIDAR_ONLY => "lidar_only",
        POPULATION_CAMERA_ONLY => "camera_only",
        POPULATION_ONE_CAMERA => "one_camera",
        _ => "unknown",
    }
}

/// The association rule in one line, as `run.json` and `summary.json` record
/// it: there is no parameter to record, because the threshold is derived, so
/// the rule itself is the provenance.
pub const FUSION_RULE: &str = "fresh in-frame track box vs detection box, IoU > 1/2 (3 I > A_t + A_d), greedy on (IoU desc, track id, detection index), one to one";

/// The agreement and stereo rules in one line, beside [`FUSION_RULE`], for
/// `run.json` and `summary.json` when the right camera runs.
pub const STEREO_RULE: &str = "a fused track keeps its class only when a right-camera detection of the same class overlaps its right-image box by more than half, else it is one_camera; a track the left camera missed takes the right camera's match as one_camera; a camera-only detection is kept only with a stereo partner: same class, box rows within 8 px, widths within 30 %, the right box 0 to 192 px further left (depth 2 m or more), greedy on (row difference asc, left index, right index); its range is the baseline over the disparity; a sweep with one camera's frame only keeps that camera's matches as one_camera and its camera-only detections without a range; a sweep with the right frame only is answered from it";

/// f32 lanes in one `camera_only` row. See [`CAMERA_ONLY_BYTES`].
pub const CAMERA_ONLY_VALUES: i32 = 8;

/// [`CAMERA_ONLY_VALUES`] as a `usize`.
pub const CAMERA_ONLY_LANES: usize = CAMERA_ONLY_VALUES as usize;

/// Bytes in one `camera_only` row: 8 f32 lanes.
///
/// | lane | field |
/// |---|---|
/// | 0 | `det_index` -- the detection's index in the paired frame's `CAM_DET` batch |
/// | 1 | `class_id` -- an index into [`crate::camdet::COCO_CLASSES`] |
/// | 2 | `confidence` |
/// | 3-6 | `x0, y0, x1, y1` -- its box in the frame, pixels |
/// | 7 | `range_m` -- its depth from the two cameras' disparity when the right camera saw it too ([`stereo_match`]), else 0 |
///
/// No velocity or age, and no distance from one camera alone: the camera
/// measured none, and this module does not invent them. The range in lane 7
/// is a measurement the two cameras made together.
pub const CAMERA_ONLY_BYTES: usize = 32;

/// Lane 7 of a `camera_only` row: the stereo range, or 0.
pub const CAMERA_ONLY_LANE_RANGE: usize = 7;

/// The nearest an object a camera-only detection names can be, metres: the
/// largest disparity the stereo match accepts is the baseline over this.
pub const STEREO_MIN_RANGE_M: f64 = 2.0;

/// How far apart, in pixels, the top edges and the bottom edges of a stereo
/// pair may be. The two cameras are rectified to one image plane, so a
/// point is on the same row in both; the detector's box edges are not that
/// exact.
pub const STEREO_ROW_TOL_PX: f32 = 8.0;

/// The most one box of a stereo pair may be wider than the other.
pub const STEREO_WIDTH_RATIO_MAX: f32 = 1.3;

/// What the stereo rules need from the two cameras' calibration.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Stereo {
    /// `left.baseline_term - right.baseline_term`: the pixel-metres a
    /// disparity divides into a depth ([`crate::calib::stereo_range_m`]).
    pub px_m: f64,
    /// The largest disparity a real object can have: `px_m` over
    /// [`STEREO_MIN_RANGE_M`].
    pub max_disparity_px: f32,
}

impl Stereo {
    /// From the two cameras' calibration.
    pub fn of(left: &Calib, right: &Calib) -> Stereo {
        let px_m = left.baseline_term - right.baseline_term;
        Stereo {
            px_m,
            max_disparity_px: (px_m / STEREO_MIN_RANGE_M) as f32,
        }
    }

    /// The depth of a disparity, or `None` for one that is not positive.
    pub fn range_m(&self, disparity_px: f32) -> Option<f32> {
        (disparity_px > 0.0 && self.px_m > 0.0)
            .then(|| (self.px_m / f64::from(disparity_px)) as f32)
    }
}

/// The right camera's half of one sweep's association.
pub struct RightHalf<'a> {
    /// What the detector found in the right frame paired with the sweep.
    pub dets: &'a DetView<'a>,
    /// The two cameras' baseline, for the camera-only stereo match.
    pub stereo: Stereo,
}

/// Whether a left detection and a right one are one object seen from the
/// two cameras, and if so the cost of the pairing (the row and width
/// differences, pixels) and the disparity: the same class, both box rows
/// within [`STEREO_ROW_TOL_PX`], widths within [`STEREO_WIDTH_RATIO_MAX`],
/// and the right box further left by 0 to `max_disparity_px`. Positive
/// tests throughout, so a NaN refuses.
pub fn stereo_match(
    left: &DetView<'_>,
    li: usize,
    right: &DetView<'_>,
    ri: usize,
    stereo: &Stereo,
) -> Option<(f32, f32)> {
    if left.class_id[li] != right.class_id[ri] {
        return None;
    }
    let dy0 = (left.y0[li] - right.y0[ri]).abs();
    let dy1 = (left.y1[li] - right.y1[ri]).abs();
    let (wl, wr) = (left.x1[li] - left.x0[li], right.x1[ri] - right.x0[ri]);
    let rows_agree = dy0 <= STEREO_ROW_TOL_PX && dy1 <= STEREO_ROW_TOL_PX;
    let widths_agree = wl > 0.0
        && wr > 0.0
        && wl <= wr * STEREO_WIDTH_RATIO_MAX
        && wr <= wl * STEREO_WIDTH_RATIO_MAX;
    let disparity = (left.x0[li] + left.x1[li]) * 0.5 - (right.x0[ri] + right.x1[ri]) * 0.5;
    let depth_allows = disparity > 0.0 && disparity <= stereo.max_disparity_px;
    (rows_agree && widths_agree && depth_allows).then_some((dy0 + dy1 + (wl - wr).abs(), disparity))
}

/// Intersection over union of a track's image box and a detection's, when it
/// is above one half -- the association's threshold -- and `None` otherwise.
///
/// Tested as `3 I > A_t + A_d`, which is `I / (A_t + A_d - I) > 1/2` with the
/// divide taken out, and only after both boxes are shown to have positive
/// area by positive comparisons: a NaN corner makes a width NaN, fails
/// `w > 0`, and the pair is refused instead of reaching the arithmetic. The
/// IoU itself is divided out only for a pair that passed, for the sort key
/// and the record.
pub fn fused_iou(t: &ImageBox, d: &ImageBox) -> Option<f32> {
    let (tw, th) = (t.x1 - t.x0, t.y1 - t.y0);
    let (dw, dh) = (d.x1 - d.x0, d.y1 - d.y0);
    let has_area = tw > 0.0 && th > 0.0 && dw > 0.0 && dh > 0.0;
    if !has_area {
        return None;
    }
    let iw = t.x1.min(d.x1) - t.x0.max(d.x0);
    let ih = t.y1.min(d.y1) - t.y0.max(d.y0);
    let overlaps = iw > 0.0 && ih > 0.0;
    if !overlaps {
        return None;
    }
    let (inter, at, ad) = (iw * ih, tw * th, dw * dh);
    (3.0 * inter > at + ad).then(|| inter / (at + ad - inter))
}

/// What one association did, counted while it was doing it.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct FusePlan {
    /// Whether a detector's output was associated at all. `false` under
    /// `--detector off`, where every other field is 0.
    pub ran: bool,
    /// Detections in the paired frame.
    pub detections: u32,
    /// Tracks that could be fused: fresh (seen this sweep) and in frame.
    pub candidates: u32,
    /// Tracks fused, one detection each.
    pub fused: u32,
    /// Detections no track matched: `detections - fused`.
    pub camera_only: u32,
    /// Detections with MORE THAN ONE candidate track above the threshold --
    /// only possible where the lidar boxes overlap each other. Greedy on IoU
    /// decided these; the count says how often it had to.
    pub contested: u32,
    /// Candidate tracks with more than one detection above the threshold --
    /// a person on a bicycle over one lidar cluster.
    pub crowded: u32,
    /// Sum of the fused pairs' IoU, for the mean.
    pub iou_sum: f64,
    /// Whether the right camera's detections were associated too
    /// ([`Fuser::confirm`]). The six below are 0 when not.
    pub right_ran: bool,
    /// Detections in the paired right frame.
    pub right_detections: u32,
    /// Left-fused tracks the right camera confirmed, same class: these stay
    /// [`POPULATION_FUSED`], and `fused` is reduced to them.
    pub confirmed: u32,
    /// Tracks one camera matched and the other did not confirm
    /// ([`POPULATION_ONE_CAMERA`]): the left-fused tracks not confirmed, plus
    /// `right_only`.
    pub one_camera: u32,
    /// Tracks the left camera missed and the right one matched.
    pub right_only: u32,
    /// Left camera-only detections with a stereo partner, kept with a range.
    pub stereo_matched: u32,
    /// Left camera-only detections with none, dropped.
    pub camera_only_dropped: u32,
    /// Left-fused tracks the right camera did not confirm: the part of
    /// `one_camera` that came off `fused`. With [`FusePlan::demoted`], all
    /// of them.
    pub unconfirmed: u32,
    /// Whether the set was associated with one camera's frame in a run that
    /// has two ([`demote_to_one_camera`]): every match is one camera's.
    pub demoted: bool,
}

/// The second camera's pass when the sweep has no frame of it, in a run
/// that has two: one camera's match is not an agreement, so every fused
/// track becomes [`POPULATION_ONE_CAMERA`] -- keeping the class and the
/// confidence it was matched with, for the record -- and `plan` says so.
/// The camera-only detections stay as the one camera left them, with no
/// range. Allocates nothing.
pub fn demote_to_one_camera(rows: &mut [[f32; TRACK_LANES]], plan: &mut FusePlan) {
    for t in rows.iter_mut() {
        if t[LANE_POPULATION] == POPULATION_FUSED as f32 {
            t[LANE_POPULATION] = POPULATION_ONE_CAMERA as f32;
        }
    }
    plan.demoted = true;
    plan.unconfirmed = plan.fused;
    plan.one_camera = plan.fused;
    plan.fused = 0;
}

/// Reusable scratch for the association, so the only per-sample allocation
/// the fusion makes is its output.
///
/// The constraint [`crate::track::Tracker`] documents, for the same reason:
/// scratch that grows inside the measured window makes `bytes_alloc` read
/// "the output plus its bookkeeping" under the name of the output.
#[derive(Debug, Default)]
pub struct Fuser {
    /// Candidate `(IoU, track id, track row, detection index)` above the
    /// threshold.
    pairs: Vec<(f32, u32, u32, u32)>,
    /// Whether each detection has been fused this association.
    det_taken: Vec<bool>,
    /// Candidate tracks per detection, only for [`FusePlan::contested`].
    det_cands: Vec<u32>,
    /// Each track's box in the right image, filled by the caller
    /// ([`Fuser::right_boxes`]) before [`Fuser::confirm`].
    right_boxes: Vec<Option<ImageBox>>,
    /// Candidate `(IoU, track id, track row, right detection index)` above
    /// the threshold, in the right image.
    right_pairs: Vec<(f32, u32, u32, u32)>,
    /// Whether each right detection has been taken this association.
    right_taken: Vec<bool>,
    /// Whether each track row has been settled by the right pass.
    row_done: Vec<bool>,
    /// `(cost, left index, right index)` of the stereo candidates among the
    /// camera-only detections.
    stereo_pairs: Vec<(f32, u32, u32)>,
    /// Each left detection's stereo range, 0 where it has no partner.
    ranges: Vec<f32>,
    /// `(the lidar's distance to the near face, to the box's centre, the
    /// stereo range)` of every track both cameras confirmed this
    /// association: the second sensor's range beside the first's, for the
    /// run to compare. Both lidar figures, because a silhouette's disparity
    /// is nearer the object's centre than its near face.
    checks: Vec<(f32, f32, f32)>,
}

impl Fuser {
    /// An empty association scratch. [`Fuser::reserve`] sizes it.
    pub fn new() -> Fuser {
        Fuser::default()
    }

    /// Makes room for `tracks` against `dets` detections. **Call it outside
    /// the measured window.** Returns whether it had to allocate, so a caller
    /// can count the growth and a test can assert a steady state.
    pub fn reserve(&mut self, tracks: usize, dets: usize) -> bool {
        let pairs = tracks.saturating_mul(dets);
        let grew = self.pairs.capacity() < pairs
            || self.det_taken.capacity() < dets
            || self.det_cands.capacity() < dets;
        if !grew {
            return false;
        }
        // Cleared first: `reserve(n)` promises `len + n`, and these still
        // hold the last association's entries -- the trap `Tracker::reserve`
        // documents.
        self.pairs.clear();
        self.det_taken.clear();
        self.det_cands.clear();
        self.pairs.reserve(pairs);
        self.det_taken.reserve(dets);
        self.det_cands.reserve(dets);
        true
    }

    /// Makes room for the right camera's half: `tracks` rows against
    /// `right_dets` detections, and the stereo match of `dets` left
    /// detections against them. **Outside the measured window**, on
    /// [`Fuser::reserve`]'s terms, and returns whether it had to allocate.
    pub fn reserve_right(&mut self, tracks: usize, dets: usize, right_dets: usize) -> bool {
        let pairs = tracks.saturating_mul(right_dets);
        let stereo = dets.saturating_mul(right_dets);
        let grew = self.right_boxes.capacity() < tracks
            || self.right_pairs.capacity() < pairs
            || self.right_taken.capacity() < right_dets
            || self.row_done.capacity() < tracks
            || self.stereo_pairs.capacity() < stereo
            || self.ranges.capacity() < dets
            || self.checks.capacity() < tracks;
        if !grew {
            return false;
        }
        self.right_boxes.clear();
        self.right_pairs.clear();
        self.right_taken.clear();
        self.row_done.clear();
        self.stereo_pairs.clear();
        self.ranges.clear();
        self.checks.clear();
        self.right_boxes.reserve(tracks);
        self.right_pairs.reserve(pairs);
        self.right_taken.reserve(right_dets);
        self.row_done.reserve(tracks);
        self.stereo_pairs.reserve(stereo);
        self.ranges.reserve(dets);
        self.checks.reserve(tracks);
        true
    }

    /// Bytes the scratch currently holds.
    pub fn scratch_bytes(&self) -> usize {
        self.pairs.capacity() * std::mem::size_of::<(f32, u32, u32, u32)>()
            + self.det_taken.capacity()
            + self.det_cands.capacity() * 4
            + self.right_boxes.capacity() * std::mem::size_of::<Option<ImageBox>>()
            + self.right_pairs.capacity() * std::mem::size_of::<(f32, u32, u32, u32)>()
            + self.right_taken.capacity()
            + self.row_done.capacity()
            + self.stereo_pairs.capacity() * std::mem::size_of::<(f32, u32, u32)>()
            + self.ranges.capacity() * 4
            + self.checks.capacity() * std::mem::size_of::<(f32, f32, f32)>()
    }

    /// The slot for each of `n` track rows' box in the right image, cleared,
    /// for the caller to fill before [`Fuser::confirm`]. Allocates nothing
    /// once [`Fuser::reserve_right`] has been called for `n` tracks.
    pub fn right_boxes(&mut self, n: usize) -> &mut [Option<ImageBox>] {
        self.right_boxes.clear();
        self.right_boxes.resize(n, None);
        &mut self.right_boxes
    }

    /// Each left detection's stereo range after [`Fuser::confirm`]: 0 where
    /// it has no partner, which is also every detection that was fused.
    pub fn stereo_ranges(&self) -> &[f32] {
        &self.ranges
    }

    /// Whether each right detection was taken, by a track or as a stereo
    /// partner.
    pub fn right_taken(&self) -> &[bool] {
        &self.right_taken
    }

    /// `(the lidar's distance to the near face, to the box's centre, the
    /// stereo range)` of every track both cameras confirmed in the last
    /// [`Fuser::confirm`]: the disparity of the two boxes' centres, turned
    /// into a depth. Two sensors measuring one object, for the run to
    /// compare.
    pub fn stereo_checks(&self) -> &[(f32, f32, f32)] {
        &self.checks
    }

    /// The right camera's pass, after [`Fuser::associate`] and after
    /// [`Fuser::right_boxes`] has been filled: holds every left-fused track
    /// to a right detection of the same class, gives the right camera's
    /// matches to the tracks the left missed, and finds the camera-only
    /// detections' stereo partners. Rewrites lanes 21-25 where it changes
    /// its mind, and `plan` with it. See the module docs.
    ///
    /// Allocates nothing once [`Fuser::reserve_right`] has been called.
    pub fn confirm(
        &mut self,
        rows: &mut [[f32; TRACK_LANES]],
        left: &DetView<'_>,
        right: &RightHalf<'_>,
        plan: &mut FusePlan,
    ) {
        let dets = right.dets;
        let n = dets.len();
        plan.right_ran = true;
        plan.right_detections = n as u32;
        self.right_pairs.clear();
        self.right_taken.clear();
        self.right_taken.resize(n, false);
        self.row_done.clear();
        self.row_done.resize(rows.len(), false);
        self.checks.clear();
        for (ri, t) in rows.iter().enumerate() {
            if t[8] != 0.0 {
                continue;
            }
            let Some(tb) = self.right_boxes.get(ri).copied().flatten() else {
                continue;
            };
            let pop = t[LANE_POPULATION];
            let fused = pop == POPULATION_FUSED as f32;
            if !fused && pop != POPULATION_LIDAR_ONLY as f32 {
                continue;
            }
            // A right-only match is for a track the left camera MISSED, not
            // one outside the left frame: the record's image box, and the
            // picture the answer is drawn on, are the left camera's, and a
            // class on a track with no box there is a record the far end
            // refuses. The right image sees further right than the left.
            if !fused && track_image_box(t).is_none() {
                continue;
            }
            for di in 0..n {
                // A fused track's confirmation must be the same class; a
                // track the left camera missed takes any.
                if fused && dets.class_id[di] as f32 != t[LANE_CLASS_ID] {
                    continue;
                }
                let d = ImageBox {
                    x0: dets.x0[di],
                    y0: dets.y0[di],
                    x1: dets.x1[di],
                    y1: dets.y1[di],
                };
                if let Some(iou) = fused_iou(&tb, &d) {
                    self.right_pairs
                        .push((iou, t[6] as u32, ri as u32, di as u32));
                }
            }
        }
        self.right_pairs
            .sort_unstable_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)).then(a.3.cmp(&b.3)));
        for &(iou, _, ri, di) in &self.right_pairs {
            let (ri, di) = (ri as usize, di as usize);
            if self.right_taken[di] || self.row_done[ri] {
                continue;
            }
            self.right_taken[di] = true;
            self.row_done[ri] = true;
            let t = &mut rows[ri];
            if t[LANE_POPULATION] == POPULATION_FUSED as f32 {
                plan.confirmed += 1;
                // One object in both cameras: the two boxes' centres apart
                // is its disparity, and so a second range, which the run
                // reads against the lidar's distance to its near face.
                let li = t[LANE_DET_INDEX];
                if li >= 0.0 {
                    let li = li as usize;
                    if let (Some(&lx0), Some(&lx1)) = (left.x0.get(li), left.x1.get(li)) {
                        let disparity = 0.5 * ((lx0 + lx1) - (dets.x0[di] + dets.x1[di]));
                        if let Some(range) = right.stereo.range_m(disparity) {
                            self.checks.push((t[9], 0.5 * (t[9] + t[12]), range));
                        }
                    }
                }
            } else {
                // The right camera alone: its class and confidence, and no
                // index, because the index would name a box in the left frame.
                t[LANE_DET_INDEX] = -1.0;
                t[LANE_FUSE_IOU] = iou;
                t[LANE_CLASS_ID] = dets.class_id[di] as f32;
                t[LANE_CONFIDENCE] = dets.confidence[di];
                t[LANE_POPULATION] = POPULATION_ONE_CAMERA as f32;
                plan.right_only += 1;
            }
        }
        // A left match the right camera did not confirm keeps its lanes for
        // the record and loses the population that would let the answer
        // trust them.
        for (ri, t) in rows.iter_mut().enumerate() {
            if t[LANE_POPULATION] == POPULATION_FUSED as f32 && !self.row_done[ri] {
                t[LANE_POPULATION] = POPULATION_ONE_CAMERA as f32;
            }
        }
        plan.unconfirmed = plan.fused - plan.confirmed;
        plan.one_camera = plan.unconfirmed + plan.right_only;
        plan.fused = plan.confirmed;

        // The camera-only detections' stereo partners, among what neither
        // pass took on either side.
        let left_n = left.len();
        self.ranges.clear();
        self.ranges.resize(left_n, 0.0);
        self.stereo_pairs.clear();
        for li in 0..left_n {
            if self.det_taken.get(li).copied().unwrap_or(false) {
                continue;
            }
            for di in 0..n {
                if self.right_taken[di] {
                    continue;
                }
                if let Some((cost, _)) = stereo_match(left, li, dets, di, &right.stereo) {
                    self.stereo_pairs.push((cost, li as u32, di as u32));
                }
            }
        }
        self.stereo_pairs
            .sort_unstable_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));
        let mut left_unmatched = (0..left_n)
            .filter(|&li| !self.det_taken.get(li).copied().unwrap_or(false))
            .count() as u32;
        for &(_, li, di) in &self.stereo_pairs {
            let (li, di) = (li as usize, di as usize);
            if self.right_taken[di] || self.ranges[li] > 0.0 {
                continue;
            }
            let Some((_, disparity)) = stereo_match(left, li, dets, di, &right.stereo) else {
                continue;
            };
            let Some(range) = right.stereo.range_m(disparity) else {
                continue;
            };
            self.right_taken[di] = true;
            self.ranges[li] = range;
            plan.stereo_matched += 1;
            left_unmatched -= 1;
        }
        plan.camera_only_dropped = left_unmatched;
        plan.camera_only = plan.stereo_matched;
    }

    /// Whether each detection of the last association was fused: the
    /// complement is the camera-only population.
    pub fn taken(&self) -> &[bool] {
        &self.det_taken
    }

    /// Associates `dets` with `rows` and writes the result into lanes 21-25
    /// of EVERY row: the detection's index, the IoU, its class and
    /// confidence, and [`POPULATION_FUSED`] on a fused track;
    /// `-1, 0, 0, 0` and [`POPULATION_LIDAR_ONLY`] on every other.
    ///
    /// Allocates nothing once [`Fuser::reserve`] has been called for at least
    /// `rows.len()` tracks and `dets.len()` detections.
    pub fn associate(&mut self, rows: &mut [[f32; TRACK_LANES]], dets: &DetView<'_>) -> FusePlan {
        let n = dets.len();
        let mut plan = FusePlan {
            ran: true,
            detections: n as u32,
            ..FusePlan::default()
        };
        self.pairs.clear();
        self.det_taken.clear();
        self.det_taken.resize(n, false);
        self.det_cands.clear();
        self.det_cands.resize(n, 0);
        for (ri, t) in rows.iter_mut().enumerate() {
            t[LANE_DET_INDEX] = -1.0;
            t[LANE_FUSE_IOU] = 0.0;
            t[LANE_CLASS_ID] = 0.0;
            t[LANE_CONFIDENCE] = 0.0;
            t[LANE_POPULATION] = POPULATION_LIDAR_ONLY as f32;
            // Fresh and in frame, or not a candidate. See the module docs.
            if t[8] != 0.0 {
                continue;
            }
            let Some(tb) = track_image_box(t) else {
                continue;
            };
            plan.candidates += 1;
            let mut found = 0u32;
            for (di, cands) in self.det_cands.iter_mut().enumerate() {
                let d = ImageBox {
                    x0: dets.x0[di],
                    y0: dets.y0[di],
                    x1: dets.x1[di],
                    y1: dets.y1[di],
                };
                if let Some(iou) = fused_iou(&tb, &d) {
                    // Exact below 2^24, as every integer lane in this chain.
                    self.pairs.push((iou, t[6] as u32, ri as u32, di as u32));
                    *cands += 1;
                    found += 1;
                }
            }
            plan.crowded += u32::from(found > 1);
        }
        plan.contested = self.det_cands.iter().filter(|&&c| c > 1).count() as u32;
        // IoU descending, then track id, then detection index: unique, so
        // `sort_unstable_by` is deterministic and allocates nothing -- the
        // stable sort takes a scratch buffer, inside the measured window.
        self.pairs
            .sort_unstable_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)).then(a.3.cmp(&b.3)));
        for &(iou, _, ri, di) in &self.pairs {
            let t = &mut rows[ri as usize];
            let di = di as usize;
            if self.det_taken[di] || t[LANE_POPULATION] == POPULATION_FUSED as f32 {
                continue;
            }
            self.det_taken[di] = true;
            t[LANE_DET_INDEX] = di as f32;
            t[LANE_FUSE_IOU] = iou;
            t[LANE_CLASS_ID] = dets.class_id[di] as f32;
            t[LANE_CONFIDENCE] = dets.confidence[di];
            t[LANE_POPULATION] = POPULATION_FUSED as f32;
            plan.fused += 1;
            plan.iou_sum += f64::from(iou);
        }
        plan.camera_only = plan.detections - plan.fused;
        plan
    }
}

/// Arrow field of one `camera_only` lane.
fn camera_only_value_field() -> FieldRef {
    Arc::new(Field::new("item", DataType::Float32, false))
}

/// Arrow field of one `camera_only` row: `FixedSizeList<Float32, 7>`.
fn camera_only_field() -> FieldRef {
    Arc::new(Field::new(
        "item",
        DataType::FixedSizeList(camera_only_value_field(), CAMERA_ONLY_VALUES),
        false,
    ))
}

/// The type of the `camera_only` column the track and answer batches carry:
/// `LargeList<FixedSizeList<Float32, 7>>`, one list per sample.
pub fn camera_only_data_type() -> DataType {
    DataType::LargeList(camera_only_field())
}

/// The one-row `camera_only` column: every detection of `dets` that `taken`
/// says was NOT fused, in detection order, or an empty list when there are no
/// detections at all (`None`). With `ranges`, the right camera ran, and only
/// the detections with a stereo partner -- a range above 0 -- are kept, each
/// with its range in lane 7.
///
/// Its own buffer, allocated at exactly its size: a handful of rows a frame,
/// never the track buffer, so the address the zero-copy proof follows is
/// untouched by it.
pub fn build_camera_only(
    dets: Option<&DetView<'_>>,
    taken: &[bool],
    ranges: Option<&[f32]>,
) -> Result<ArrayRef, SweepError> {
    let n_dets = dets.map_or(0, |d| d.len());
    let keep = |i: usize| {
        !taken.get(i).copied().unwrap_or(false)
            && ranges.is_none_or(|r| r.get(i).copied().unwrap_or(0.0) > 0.0)
    };
    let n = (0..n_dets).filter(|&i| keep(i)).count();
    let mut buf = MutableBuffer::from_len_zeroed(n * CAMERA_ONLY_BYTES);
    if let Some(d) = dets {
        let out = buf.typed_data_mut::<f32>();
        let mut w = 0usize;
        for i in 0..d.len() {
            if !keep(i) {
                continue;
            }
            let lane = &mut out[w * CAMERA_ONLY_LANES..][..CAMERA_ONLY_LANES];
            lane[0] = i as f32;
            lane[1] = d.class_id[i] as f32;
            lane[2] = d.confidence[i];
            lane[3] = d.x0[i];
            lane[4] = d.y0[i];
            lane[5] = d.x1[i];
            lane[6] = d.y1[i];
            lane[CAMERA_ONLY_LANE_RANGE] = ranges.map_or(0.0, |r| r.get(i).copied().unwrap_or(0.0));
            w += 1;
        }
    }
    let values = Float32Array::new(ScalarBuffer::<f32>::from(Buffer::from(buf)), None);
    let rec = FixedSizeListArray::try_new(
        camera_only_value_field(),
        CAMERA_ONLY_VALUES,
        Arc::new(values) as ArrayRef,
        None,
    )?;
    let offsets = OffsetBuffer::<i64>::new(ScalarBuffer::<i64>::from(vec![0i64, n as i64]));
    let arr = LargeListArray::try_new(camera_only_field(), offsets, Arc::new(rec), None)?;
    Ok(Arc::new(arr) as ArrayRef)
}

/// The `camera_only` column of a track or answer batch, shared with whoever
/// built it: `state` hands the track batch's column on unchanged.
pub fn camera_only_column(batch: &RecordBatch) -> Option<&ArrayRef> {
    batch.column_by_name("camera_only")
}

/// The camera-only rows of a track or answer batch, or an empty slice.
pub fn camera_only_rows(batch: &RecordBatch) -> &[[f32; CAMERA_ONLY_LANES]] {
    let lanes: &[f32] = camera_only_column(batch)
        .and_then(|c| c.as_list_opt::<i64>())
        .and_then(|l| l.values().as_fixed_size_list_opt())
        .and_then(|f| f.values().as_primitive_opt::<Float32Type>())
        .map_or(&[], |p| p.values());
    let (rows, _) = lanes.as_chunks::<CAMERA_ONLY_LANES>();
    rows
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::camdet::{build_cam_det_batch, cam_det_schema, cam_det_view, Det, MODEL_SHA256};
    use crate::track::{track_detection, track_population, CamRef};
    use arrow::array::Array;

    fn b(x0: f32, y0: f32, x1: f32, y1: f32) -> ImageBox {
        ImageBox { x0, y0, x1, y1 }
    }

    /// A fresh track record with an image box, and an id.
    fn trk(id: f32, bx: ImageBox) -> [f32; TRACK_LANES] {
        let mut t = [0.0f32; TRACK_LANES];
        t[6] = id;
        t[15] = bx.x0;
        t[16] = bx.y0;
        t[17] = bx.x1;
        t[18] = bx.y1;
        t[LANE_DET_INDEX] = -1.0;
        t
    }

    fn det(bx: ImageBox, class_id: u32, score: f32, row: u32) -> Det {
        Det {
            x0: bx.x0,
            y0: bx.y0,
            x1: bx.x1,
            y1: bx.y1,
            class_id,
            score,
            row,
        }
    }

    fn batch(dets: &[Det]) -> RecordBatch {
        let r = CamRef {
            seq: 5,
            tov_ns: 6,
            width: 1242,
            height: 375,
        };
        build_cam_det_batch(r, dets, &cam_det_schema(MODEL_SHA256)).unwrap()
    }

    fn fuse(rows: &mut [[f32; TRACK_LANES]], dets: &[Det]) -> (FusePlan, Vec<bool>) {
        let bt = batch(dets);
        let v = cam_det_view(&bt).unwrap();
        let mut f = Fuser::new();
        f.reserve(rows.len(), v.len());
        let p = f.associate(rows, &v);
        (p, f.taken().to_vec())
    }

    /// The 2011_09_26 rig: 384.38 pixel-metres between the two cameras.
    fn rig() -> Stereo {
        Stereo {
            px_m: 384.38148,
            max_disparity_px: (384.38148 / STEREO_MIN_RANGE_M) as f32,
        }
    }

    /// A box as the right camera sees it: `d` pixels further left.
    fn shifted(bx: ImageBox, d: f32) -> ImageBox {
        b(bx.x0 - d, bx.y0, bx.x1 - d, bx.y1)
    }

    /// [`fuse`], then the right camera's pass with every track's right box
    /// its left box shifted by `shift` (the disparity of a track at one
    /// depth), and the camera-only column built from the result.
    fn fuse_both(
        rows: &mut [[f32; TRACK_LANES]],
        dets: &[Det],
        right: &[Det],
        shift: f32,
    ) -> (FusePlan, Vec<[f32; CAMERA_ONLY_LANES]>, Vec<bool>) {
        let lb = batch(dets);
        let lv = cam_det_view(&lb).unwrap();
        let rb = batch(right);
        let rv = cam_det_view(&rb).unwrap();
        let mut f = Fuser::new();
        f.reserve(rows.len(), lv.len());
        f.reserve_right(rows.len(), lv.len(), rv.len());
        let mut p = f.associate(rows, &lv);
        let boxes = f.right_boxes(rows.len());
        for (slot, t) in boxes.iter_mut().zip(rows.iter()) {
            *slot = track_image_box(t).map(|bx| shifted(bx, shift));
        }
        let half = RightHalf {
            dets: &rv,
            stereo: rig(),
        };
        f.confirm(rows, &lv, &half, &mut p);
        let col = build_camera_only(Some(&lv), f.taken(), Some(f.stereo_ranges())).unwrap();
        let lanes: &[f32] = col
            .as_list::<i64>()
            .values()
            .as_fixed_size_list()
            .values()
            .as_primitive::<Float32Type>()
            .values();
        let (co, _) = lanes.as_chunks::<CAMERA_ONLY_LANES>();
        (p, co.to_vec(), f.right_taken().to_vec())
    }

    /// With the right camera, a class survives only where both cameras gave
    /// it: a confirmed car stays fused, a car the right camera calls a
    /// bicycle becomes one-camera with the left's class kept for the record,
    /// and a track only the right camera matched is one-camera with the
    /// right's class and no left index.
    #[test]
    fn a_class_is_kept_only_where_the_right_camera_agrees() {
        let mut rows = vec![
            trk(7.0, b(100.0, 100.0, 200.0, 200.0)),
            trk(8.0, b(400.0, 100.0, 450.0, 200.0)),
            trk(9.0, b(800.0, 100.0, 900.0, 200.0)),
        ];
        let shift = 40.0;
        let dets = [
            det(b(105.0, 102.0, 198.0, 205.0), 2, 0.91, 0), // car, on track 7
            det(b(402.0, 98.0, 452.0, 199.0), 0, 0.74, 1),  // person, on track 8
        ];
        let right = [
            det(shifted(b(105.0, 102.0, 198.0, 205.0), shift), 2, 0.88, 0), // car on 7: agrees
            det(shifted(b(402.0, 98.0, 452.0, 199.0), shift), 1, 0.70, 1), // bicycle on 8: disagrees
            det(shifted(b(805.0, 101.0, 895.0, 199.0), shift), 2, 0.66, 2), // car on 9: right only
        ];
        let (p, co, right_taken) = fuse_both(&mut rows, &dets, &right, shift);
        assert!(p.right_ran);
        assert_eq!(
            (p.fused, p.confirmed, p.one_camera, p.right_only),
            (1, 1, 2, 1),
            "{p:?}"
        );
        assert_eq!(track_population(&rows[0]), POPULATION_FUSED);
        let (index, iou, class, conf) = track_detection(&rows[0]).unwrap();
        assert_eq!((index, class, conf), (0, 2, 0.91));
        assert!(iou > 0.5, "{iou}");
        // The left's class is kept in the lanes, and the population says it
        // was not confirmed; the reader that hands out classes refuses it.
        assert_eq!(track_population(&rows[1]), POPULATION_ONE_CAMERA);
        assert_eq!(rows[1][LANE_CLASS_ID], 0.0);
        assert_eq!(rows[1][LANE_DET_INDEX], 1.0);
        assert_eq!(track_detection(&rows[1]), None);
        // The right camera's own match: its class, its confidence, no index.
        assert_eq!(track_population(&rows[2]), POPULATION_ONE_CAMERA);
        assert_eq!(rows[2][LANE_DET_INDEX], -1.0);
        assert_eq!(rows[2][LANE_CLASS_ID], 2.0);
        assert_eq!(rows[2][LANE_CONFIDENCE], 0.66);
        // The bicycle could not confirm a person, so nothing took it.
        assert_eq!(right_taken, [true, false, true]);
        // Nothing was camera-only on either side.
        assert!(co.is_empty());
        assert_eq!(
            (p.stereo_matched, p.camera_only_dropped, p.camera_only),
            (0, 0, 0)
        );
    }

    /// A detection no track matched is kept only with a stereo partner, and
    /// then carries the range the two cameras agree on; one without is
    /// dropped and counted.
    #[test]
    fn a_camera_only_detection_needs_a_stereo_partner_and_gets_a_range() {
        let mut rows = vec![trk(7.0, b(100.0, 100.0, 200.0, 200.0))];
        let dets = [
            det(b(105.0, 102.0, 198.0, 205.0), 2, 0.91, 0), // car, on track 7
            det(b(600.0, 100.0, 650.0, 150.0), 1, 0.42, 1), // bicycle, nowhere: 10 m away
            det(b(900.0, 200.0, 960.0, 300.0), 0, 0.55, 2), // person, nowhere, no partner
        ];
        let right = [
            det(shifted(b(105.0, 102.0, 198.0, 205.0), 40.0), 2, 0.88, 0), // confirms the car
            det(b(560.0, 103.0, 612.0, 152.0), 1, 0.40, 1), // the bicycle, 38.4 px left (10 m)
            det(b(300.0, 200.0, 360.0, 300.0), 2, 0.50, 2), // a car no left detection has
        ];
        let (p, co, right_taken) = fuse_both(&mut rows, &dets, &right, 40.0);
        assert_eq!((p.fused, p.confirmed), (1, 1));
        assert_eq!(
            (p.stereo_matched, p.camera_only_dropped, p.camera_only),
            (1, 1, 1),
            "{p:?}"
        );
        assert_eq!(co.len(), 1);
        let r = &co[0];
        assert_eq!((r[0], r[1], r[2]), (1.0, 1.0, 0.42));
        assert_eq!(&r[3..7], &[600.0, 100.0, 650.0, 150.0]);
        // Centres 625 and 586: 39 px apart, 9.86 m.
        assert!(
            (r[CAMERA_ONLY_LANE_RANGE] - 384.38148 / 39.0).abs() < 1e-3,
            "{}",
            r[7]
        );
        assert_eq!(right_taken, [true, true, false]);
        // Without the right camera the same column keeps both, with no range.
        let (_, taken) = fuse(&mut rows.clone(), &dets);
        let lb = batch(&dets);
        let lv = cam_det_view(&lb).unwrap();
        let col = build_camera_only(Some(&lv), &taken, None).unwrap();
        assert_eq!(col.as_list::<i64>().value(0).len(), 2);
    }

    /// A sweep with one camera's frame in a run that has two: the matches
    /// stay in the lanes, for the record, and lose the population the
    /// answer trusts; the plan says every one of them is one camera's.
    #[test]
    fn without_the_second_frame_every_match_is_one_camera_s() {
        let mut rows = vec![
            trk(7.0, b(100.0, 100.0, 200.0, 200.0)),
            trk(8.0, b(400.0, 100.0, 500.0, 200.0)),
        ];
        let dets = [
            det(b(105.0, 102.0, 198.0, 205.0), 2, 0.91, 0),
            det(b(900.0, 200.0, 960.0, 300.0), 0, 0.55, 1),
        ];
        let (mut p, taken) = fuse(&mut rows, &dets);
        assert_eq!((p.fused, p.camera_only), (1, 1));
        demote_to_one_camera(&mut rows, &mut p);
        assert!(p.demoted);
        assert_eq!((p.fused, p.unconfirmed, p.one_camera), (0, 1, 1));
        assert_eq!(p.camera_only, 1, "the camera-only detection is kept");
        assert_eq!(track_population(&rows[0]), POPULATION_ONE_CAMERA);
        assert_eq!(
            (
                rows[0][LANE_DET_INDEX],
                rows[0][LANE_CLASS_ID],
                rows[0][LANE_CONFIDENCE]
            ),
            (0.0, 2.0, 0.91),
            "the match is kept for the record"
        );
        assert_eq!(track_detection(&rows[0]), None, "and is not a fused one");
        assert_eq!(track_population(&rows[1]), POPULATION_LIDAR_ONLY);
        // The camera-only column is the one camera's, with no range.
        let lb = batch(&dets);
        let lv = cam_det_view(&lb).unwrap();
        let col = build_camera_only(Some(&lv), &taken, None).unwrap();
        assert_eq!(col.as_list::<i64>().value(0).len(), 1);
    }

    /// Every track both cameras confirm gives the run a second range: the
    /// boxes' centres 40 px apart on a track whose near face the lidar put
    /// at 9.5 m is 9.6 m by stereo.
    #[test]
    fn a_confirmed_track_has_a_stereo_range_to_check_the_lidar_s_against() {
        let mut rows = vec![trk(7.0, b(100.0, 100.0, 200.0, 200.0))];
        rows[0][9] = 9.5;
        let dets = [det(b(105.0, 102.0, 198.0, 205.0), 2, 0.91, 0)];
        let right = [det(
            shifted(b(105.0, 102.0, 198.0, 205.0), 40.0),
            2,
            0.88,
            0,
        )];
        let lb = batch(&dets);
        let lv = cam_det_view(&lb).unwrap();
        let rb = batch(&right);
        let rv = cam_det_view(&rb).unwrap();
        let mut f = Fuser::new();
        f.reserve(rows.len(), lv.len());
        f.reserve_right(rows.len(), lv.len(), rv.len());
        let mut p = f.associate(&mut rows, &lv);
        let boxes = f.right_boxes(rows.len());
        boxes[0] = Some(shifted(b(100.0, 100.0, 200.0, 200.0), 40.0));
        let half = RightHalf {
            dets: &rv,
            stereo: rig(),
        };
        f.confirm(&mut rows, &lv, &half, &mut p);
        assert_eq!((p.confirmed, p.unconfirmed, p.demoted), (1, 0, false));
        let checks = f.stereo_checks();
        assert_eq!(checks.len(), 1);
        assert_eq!((checks[0].0, checks[0].1), (9.5, 4.75));
        assert!((checks[0].2 - 384.38148 / 40.0).abs() < 1e-3, "{checks:?}");
    }

    /// A track outside the left frame is not the right camera's to match
    /// alone: its record would carry a class and no box on the picture the
    /// answer is drawn on. One inside it, which the left camera missed, is.
    #[test]
    fn a_right_only_match_needs_the_track_in_the_left_frame() {
        let mut rows = vec![
            trk(7.0, b(100.0, 100.0, 200.0, 200.0)),
            trk(8.0, b(0.0, 0.0, 0.0, 0.0)),
        ];
        let dets: [Det; 0] = [];
        let right = [
            det(b(60.0, 102.0, 160.0, 198.0), 2, 0.9, 0),
            det(b(1100.0, 100.0, 1200.0, 200.0), 2, 0.9, 1),
        ];
        let lb = batch(&dets);
        let lv = cam_det_view(&lb).unwrap();
        let rb = batch(&right);
        let rv = cam_det_view(&rb).unwrap();
        let mut f = Fuser::new();
        f.reserve(rows.len(), lv.len());
        f.reserve_right(rows.len(), lv.len(), rv.len());
        let mut p = f.associate(&mut rows, &lv);
        let boxes = f.right_boxes(rows.len());
        boxes[0] = Some(b(60.0, 100.0, 160.0, 200.0));
        // Track 8 is beyond the left frame's right edge and inside the right
        // frame.
        boxes[1] = Some(b(1100.0, 100.0, 1200.0, 200.0));
        let half = RightHalf {
            dets: &rv,
            stereo: rig(),
        };
        f.confirm(&mut rows, &lv, &half, &mut p);
        assert_eq!((p.right_only, p.one_camera), (1, 1), "{p:?}");
        assert_eq!(track_population(&rows[0]), POPULATION_ONE_CAMERA);
        assert_eq!(track_population(&rows[1]), POPULATION_LIDAR_ONLY);
        assert_eq!(f.right_taken(), [true, false]);
    }

    /// The stereo rules, one at a time: the right box must be on the same
    /// rows, about the same width, the same class, and to the LEFT by no
    /// more than the nearest allowed depth's disparity.
    #[test]
    fn a_stereo_pair_is_the_same_rows_the_same_width_the_same_class_and_to_the_left() {
        let stereo = rig();
        let left_box = b(600.0, 100.0, 650.0, 150.0);
        let l = batch(&[det(left_box, 1, 0.5, 0)]);
        let lv = cam_det_view(&l).unwrap();
        let try_right = |d: Det| {
            let rb = batch(&[d]);
            let rv = cam_det_view(&rb).unwrap();
            stereo_match(&lv, 0, &rv, 0, &stereo)
        };
        let ok = try_right(det(b(560.0, 104.0, 612.0, 146.0), 1, 0.5, 0)).unwrap();
        assert!(
            (ok.1 - 39.0).abs() < 1e-6 && (ok.0 - (4.0 + 4.0 + 2.0)).abs() < 1e-6,
            "{ok:?}"
        );
        // A different class, a box on other rows, a box twice the width, a
        // box to the RIGHT, and one further left than 2 m allows.
        assert_eq!(
            try_right(det(b(560.0, 104.0, 612.0, 146.0), 2, 0.5, 0)),
            None
        );
        assert_eq!(
            try_right(det(b(560.0, 120.0, 612.0, 170.0), 1, 0.5, 0)),
            None
        );
        assert_eq!(
            try_right(det(b(500.0, 100.0, 612.0, 150.0), 1, 0.5, 0)),
            None
        );
        assert_eq!(
            try_right(det(b(640.0, 100.0, 690.0, 150.0), 1, 0.5, 0)),
            None
        );
        assert_eq!(
            try_right(det(b(380.0, 100.0, 430.0, 150.0), 1, 0.5, 0)),
            None
        );
        // The rig's own numbers: 54 cm at 721.5 px is 192 px at 2 m.
        let r = Stereo::of(&calib(44.85728), &calib(-339.5242));
        assert!((r.px_m - 384.38148).abs() < 1e-5);
        assert!((r.max_disparity_px - 192.19) < 0.01);
        assert_eq!(r.range_m(0.0), None);
        assert!((r.range_m(38.438_15).unwrap() - 10.0).abs() < 1e-4);
    }

    /// A calibration with only the baseline term set: what [`Stereo::of`]
    /// reads.
    fn calib(baseline_term: f64) -> Calib {
        Calib {
            cam: crate::cam0::Camera::Left,
            m: [[0.0; 4]; 3],
            width: 1242,
            height: 375,
            baseline_term,
        }
    }

    /// The threshold is one half, exactly, and the divide-free test agrees
    /// with the divided one on both sides of it.
    #[test]
    fn the_test_is_iou_above_one_half() {
        // Two 10x10 boxes offset by d in x: I = 10(10 - d), U = 100 + 10d.
        // IoU = 1/2 at d = 10/3.
        let a = b(0.0, 0.0, 10.0, 10.0);
        let just_in = b(3.3, 0.0, 13.3, 10.0);
        let just_out = b(3.4, 0.0, 13.4, 10.0);
        let iou = fused_iou(&a, &just_in).expect("IoU 0.503 was refused");
        assert!(iou > 0.5 && iou < 0.51, "{iou}");
        assert_eq!(fused_iou(&a, &just_out), None, "IoU 0.49 was fused");
        assert_eq!(fused_iou(&a, &a), Some(1.0));
        // Disjoint, touching, and empty boxes are never the same object.
        assert_eq!(fused_iou(&a, &b(20.0, 0.0, 30.0, 10.0)), None);
        assert_eq!(fused_iou(&a, &b(10.0, 0.0, 20.0, 10.0)), None);
        assert_eq!(fused_iou(&a, &b(5.0, 5.0, 5.0, 9.0)), None);
        // A NaN corner is refused, not fused: the positive tests fail on it.
        assert_eq!(fused_iou(&a, &b(f32::NAN, 0.0, 10.0, 10.0)), None);
        assert_eq!(fused_iou(&b(0.0, 0.0, f32::NAN, 10.0), &a), None);
    }

    /// One to one, each track to the detection it overlaps most, and every
    /// detection accounted for: fused on a track's lanes or camera-only.
    #[test]
    fn a_track_takes_its_detection_s_class_and_the_rest_are_counted() {
        let mut rows = vec![
            trk(7.0, b(100.0, 100.0, 200.0, 200.0)),
            trk(8.0, b(400.0, 100.0, 450.0, 200.0)),
            trk(9.0, b(800.0, 100.0, 900.0, 200.0)),
        ];
        let dets = [
            det(b(105.0, 102.0, 198.0, 205.0), 2, 0.91, 0), // car, on track 7
            det(b(402.0, 98.0, 452.0, 199.0), 0, 0.74, 1),  // person, on track 8
            det(b(600.0, 100.0, 650.0, 150.0), 1, 0.42, 2), // bicycle, nowhere
        ];
        let (p, taken) = fuse(&mut rows, &dets);
        assert_eq!(
            (p.detections, p.candidates, p.fused, p.camera_only),
            (3, 3, 2, 1)
        );
        assert_eq!((p.contested, p.crowded), (0, 0));
        assert_eq!(taken, [true, true, false]);
        let (i, iou, class, conf) = track_detection(&rows[0]).unwrap();
        assert_eq!((i, class, conf), (0, 2, 0.91));
        assert!(iou > 0.5);
        // Class 0 is a real class ("person") on a fused track...
        assert_eq!(track_detection(&rows[1]).map(|d| (d.0, d.2)), Some((1, 0)));
        // ...and an unfused track reads as no class at all, not as class 0.
        assert_eq!(track_detection(&rows[2]), None);
        assert_eq!(track_population(&rows[2]), POPULATION_LIDAR_ONLY);
        assert_eq!(rows[2][LANE_DET_INDEX], -1.0);
        // The one left over is the camera-only row, with no range.
        let co =
            build_camera_only(Some(&cam_det_view(&batch(&dets)).unwrap()), &taken, None).unwrap();
        let rb = RecordBatch::try_from_iter([("camera_only", co)]).unwrap();
        let rows_co = camera_only_rows(&rb);
        assert_eq!(rows_co.len(), 1);
        assert_eq!(
            rows_co[0],
            [2.0, 1.0, 0.42, 600.0, 100.0, 650.0, 150.0, 0.0]
        );
    }

    /// A coasted track and an out-of-frame track are never candidates: their
    /// boxes are not a measurement of this sweep. Positive control: the same
    /// track fresh is fused.
    #[test]
    fn only_a_fresh_in_frame_track_is_a_candidate() {
        let bx = b(100.0, 100.0, 200.0, 200.0);
        let d = [det(bx, 2, 0.9, 0)];
        let mut coasted = vec![trk(1.0, bx)];
        coasted[0][8] = 1.0;
        let (p, _) = fuse(&mut coasted, &d);
        assert_eq!((p.candidates, p.fused, p.camera_only), (0, 0, 1));
        assert_eq!(track_population(&coasted[0]), POPULATION_LIDAR_ONLY);
        let mut out = vec![trk(1.0, b(0.0, 0.0, 0.0, 0.0))];
        let (p, _) = fuse(&mut out, &d);
        assert_eq!((p.candidates, p.fused), (0, 0));
        let mut fresh = vec![trk(1.0, bx)];
        let (p, _) = fuse(&mut fresh, &d);
        assert_eq!(p.fused, 1, "the positive control fused nothing");
    }

    /// Where the lidar boxes overlap, two tracks can both clear one half on
    /// one detection. The higher IoU wins, the contest is counted, and the
    /// order the tracks arrive in decides nothing.
    #[test]
    fn a_contest_is_counted_and_decided_by_iou_not_by_order() {
        let near = trk(4.0, b(100.0, 100.0, 200.0, 200.0));
        let wider = trk(3.0, b(95.0, 95.0, 215.0, 210.0));
        let d = [det(b(100.0, 100.0, 202.0, 201.0), 2, 0.8, 0)];
        for order in [[near, wider], [wider, near]] {
            let mut rows = order.to_vec();
            let (p, _) = fuse(&mut rows, &d);
            assert_eq!((p.contested, p.fused), (1, 1), "{p:?}");
            let winner: Vec<f32> = rows
                .iter()
                .filter(|r| track_population(r) == POPULATION_FUSED)
                .map(|r| r[6])
                .collect();
            assert_eq!(winner, [4.0], "the tighter box did not win");
        }
        // And on an exact IoU tie the lower track id wins, in either order.
        let a = trk(12.0, b(0.0, 0.0, 10.0, 10.0));
        let c = trk(11.0, b(0.0, 0.0, 10.0, 10.0));
        let d = [det(b(0.0, 0.0, 10.0, 10.0), 2, 0.8, 0)];
        for order in [[a, c], [c, a]] {
            let mut rows = order.to_vec();
            fuse(&mut rows, &d);
            let winner: Vec<f32> = rows
                .iter()
                .filter(|r| track_population(r) == POPULATION_FUSED)
                .map(|r| r[6])
                .collect();
            assert_eq!(winner, [11.0]);
        }
    }

    /// One track over two detections -- a rider over a bicycle -- takes the
    /// better one, is counted as crowded, and leaves the other camera-only.
    #[test]
    fn a_crowded_track_takes_one_detection() {
        let mut rows = vec![trk(1.0, b(100.0, 100.0, 200.0, 220.0))];
        let d = [
            det(b(100.0, 100.0, 200.0, 200.0), 0, 0.9, 0),
            det(b(100.0, 110.0, 200.0, 220.0), 1, 0.8, 1),
        ];
        let (p, taken) = fuse(&mut rows, &d);
        assert_eq!((p.crowded, p.fused, p.camera_only), (1, 1, 1));
        assert_eq!(taken.iter().filter(|t| **t).count(), 1);
    }

    /// An empty frame is a frame: every track lidar-only, none unfused, and
    /// an empty camera-only list rather than none.
    #[test]
    fn a_frame_with_no_detections_leaves_every_track_lidar_only() {
        let mut rows = vec![trk(1.0, b(100.0, 100.0, 200.0, 200.0))];
        let (p, _) = fuse(&mut rows, &[]);
        assert!(p.ran);
        assert_eq!((p.detections, p.candidates, p.fused), (0, 1, 0));
        assert_eq!(track_population(&rows[0]), POPULATION_LIDAR_ONLY);
        let co = build_camera_only(None, &[], None).unwrap();
        assert_eq!(co.len(), 1, "one list per sample");
        let rb = RecordBatch::try_from_iter([("camera_only", co)]).unwrap();
        assert!(camera_only_rows(&rb).is_empty());
    }

    /// Reserved once, the scratch does not grow on a second association of
    /// the same size.
    #[test]
    fn the_scratch_reaches_a_steady_state() {
        let mut f = Fuser::new();
        assert!(f.reserve(10, 5));
        assert!(!f.reserve(10, 5), "a second reserve of the same size grew");
        assert!(!f.reserve(4, 2));
        assert!(f.scratch_bytes() > 0);
    }

    #[test]
    fn every_population_has_one_name() {
        let names: Vec<&str> = [
            POPULATION_UNFUSED,
            POPULATION_FUSED,
            POPULATION_LIDAR_ONLY,
            POPULATION_CAMERA_ONLY,
        ]
        .map(population_name)
        .to_vec();
        assert_eq!(names, ["unfused", "fused", "lidar_only", "camera_only"]);
        assert_eq!(population_name(9), "unknown");
        assert_eq!(CAMERA_ONLY_LANES * 4, CAMERA_ONLY_BYTES);
    }
}
