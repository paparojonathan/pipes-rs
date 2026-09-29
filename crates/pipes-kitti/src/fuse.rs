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
//! # The populations
//!
//! Every lidar record and every detection of the paired frame ends in exactly
//! one population, and the code travels in lane 25 of the track record and
//! lane 16 of the answer record:
//!
//! | code | population | what it is |
//! |---|---|---|
//! | [`POPULATION_FUSED`] 1 | fused | a fresh in-frame track and a detection, one to one |
//! | [`POPULATION_LIDAR_ONLY`] 2 | lidar-only | a track, when a detector's output was associated and nothing matched it -- out of frame, coasted, or in frame and unmatched |
//! | [`POPULATION_CAMERA_ONLY`] 3 | camera-only | a detection no track matched; never a record, always a `camera_only` row |
//! | [`POPULATION_UNFUSED`] 0 | unfused | a track, when there was no detector output at all (`--detector off`) |
//!
//! 0 and 2 are kept apart because they are different claims: "the camera
//! looked and saw nothing here" against "nothing looked".

use std::sync::Arc;

use arrow::array::{ArrayRef, AsArray, FixedSizeListArray, Float32Array, LargeListArray};
use arrow::buffer::{Buffer, MutableBuffer, OffsetBuffer, ScalarBuffer};
use arrow::datatypes::{DataType, Field, FieldRef, Float32Type};
use arrow::record_batch::RecordBatch;

use crate::calib::ImageBox;
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

/// The spelling of a population code, as the run's output and `summary.json`
/// use it.
pub fn population_name(p: u32) -> &'static str {
    match p {
        POPULATION_UNFUSED => "unfused",
        POPULATION_FUSED => "fused",
        POPULATION_LIDAR_ONLY => "lidar_only",
        POPULATION_CAMERA_ONLY => "camera_only",
        _ => "unknown",
    }
}

/// The association rule in one line, as `run.json` and `summary.json` record
/// it: there is no parameter to record, because the threshold is derived, so
/// the rule itself is the provenance.
pub const FUSION_RULE: &str = "fresh in-frame track box vs detection box, IoU > 1/2 (3 I > A_t + A_d), greedy on (IoU desc, track id, detection index), one to one";

/// f32 lanes in one `camera_only` row. See [`CAMERA_ONLY_BYTES`].
pub const CAMERA_ONLY_VALUES: i32 = 7;

/// [`CAMERA_ONLY_VALUES`] as a `usize`.
pub const CAMERA_ONLY_LANES: usize = CAMERA_ONLY_VALUES as usize;

/// Bytes in one `camera_only` row: 7 f32 lanes.
///
/// | lane | field |
/// |---|---|
/// | 0 | `det_index` -- the detection's index in the paired frame's `CAM_DET` batch |
/// | 1 | `class_id` -- an index into [`crate::camdet::COCO_CLASSES`] |
/// | 2 | `confidence` |
/// | 3-6 | `x0, y0, x1, y1` -- its box in the frame, pixels |
///
/// No distance, velocity or age: the camera measured none, and this module
/// does not invent them.
pub const CAMERA_ONLY_BYTES: usize = 28;

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

    /// Bytes the scratch currently holds.
    pub fn scratch_bytes(&self) -> usize {
        self.pairs.capacity() * std::mem::size_of::<(f32, u32, u32, u32)>()
            + self.det_taken.capacity()
            + self.det_cands.capacity() * 4
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
/// detections at all (`None`).
///
/// Its own buffer, allocated at exactly its size: a handful of rows a frame,
/// never the track buffer, so the address the zero-copy proof follows is
/// untouched by it.
pub fn build_camera_only(
    dets: Option<&DetView<'_>>,
    taken: &[bool],
) -> Result<ArrayRef, SweepError> {
    let n_dets = dets.map_or(0, |d| d.len());
    let n = (0..n_dets)
        .filter(|&i| !taken.get(i).copied().unwrap_or(false))
        .count();
    let mut buf = MutableBuffer::from_len_zeroed(n * CAMERA_ONLY_BYTES);
    if let Some(d) = dets {
        let out = buf.typed_data_mut::<f32>();
        let mut w = 0usize;
        for i in 0..d.len() {
            if taken.get(i).copied().unwrap_or(false) {
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
        let co = build_camera_only(Some(&cam_det_view(&batch(&dets)).unwrap()), &taken).unwrap();
        let rb = RecordBatch::try_from_iter([("camera_only", co)]).unwrap();
        let rows_co = camera_only_rows(&rb);
        assert_eq!(rows_co.len(), 1);
        assert_eq!(rows_co[0], [2.0, 1.0, 0.42, 600.0, 100.0, 650.0, 150.0]);
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
        let co = build_camera_only(None, &[]).unwrap();
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
