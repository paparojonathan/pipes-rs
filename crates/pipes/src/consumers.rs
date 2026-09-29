//! Every stage of a run, starting with the consumers of `cam0`: `proc` and
//! `rerun`, the two the spec named (G.4/G.5), and [`camdet_thread`], the
//! frozen detector (below). Each is `loop { pop; stamp; work; stamp; row }`,
//! re-derives `storage_id` from the real buffer through
//! `frame::cam0_storage_id` (the standing zero-copy proof) and counts its own
//! allocations through the stage slot.
//!
//! `proc`'s work is one grayscale pass per frame: one output buffer of `w*h`
//! bytes, `proc_bytes_per_frame = 465750` on a 1242x375 frame, and nothing
//! else. Cheap enough that a slow consumer has to be simulated with
//! `--consumer-delay-ms`.
//!
//! M11: [`cloud_thread`] is the lidar's leaf consumer — the second producer's
//! `proc`. It is deliberately the smallest honest one (point count and
//! bounding box) and allocates nothing per sweep, so the ~1.95 MB the velodyne
//! driver read crosses its edge without being copied.
//!
//! M12: [`reduce_thread`] is the first stage in this project that **consumes
//! an Arrow payload and produces a smaller one**, which is the architecture
//! document's step 2 and had no implementation before it. Everything until now
//! was a driver fanning out to leaf consumers; this is stage to stage.
//!
//! [`cloud_thread`] is what makes that claim checkable rather than asserted.
//! It is not specialised to the driver's sweep: the *same* consumer, byte for
//! byte, reads the raw cloud on `velo->cloud` and the derived cloud on
//! `det->cloud`, because [`reduce_thread`] produces the layout it was handed.
//! "Processing stages consume and produce the same combination" is therefore a
//! property the code demonstrates rather than a sentence in a document — if it
//! ever stopped being true, this file would need two consumers instead of one.
//!
//! M13: [`detect_thread`] is the stage the chain was for. `reduce` made the
//! payload four times smaller and it was still the same kind of thing — x, y,
//! z, reflectance, fewer of them. This one removes the ground, groups what is
//! left into connected structures, and emits 44 bytes per structure: where it
//! is, how big it is, how bright it is. It is the first stage in this project
//! whose output is smaller **because it says something more specific**, and the
//! first whose output is not the shape of its input.
//!
//! That last part is why [`obj_thread`] exists. One `cloud_thread` reads both
//! ends of the `reduce` hand-off because a reduced cloud IS a cloud; a
//! detection is not, so it gets its own consumer and its own accessors. Needing
//! a second consumer is the measurable difference between decimating data and
//! saying something about it.
//!
//! [`camdet_thread`] is the camera's equivalent, and the first stage here
//! that runs a model: a frozen YOLOX-Nano on every frame it has time for,
//! handing `track` the frame's detections as the camera's half of each pair.
//! It is also the one consumer of a shared buffer that allocates megabytes per
//! sample, because a network cannot read an RGB8 frame in place, and its
//! evidence says so rather than hiding it.
//!
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use arrow::array::RecordBatch;
use arrow::datatypes::Schema;
use pipes_core::alloc::{
    bytes_alloc, set_stage_slot, SLOT_CAMDET, SLOT_DETECT, SLOT_OBJ, SLOT_PROC, SLOT_REDUCE,
    SLOT_RERUN, SLOT_STATE, SLOT_STATE_SINK, SLOT_TRACK,
};
use pipes_core::clock::{now, HostTime};
use pipes_core::evidence::{Evidence, RowCtx};
use pipes_core::queue::{BoundedQueue, Queued};
use pipes_core::sample::{payload_bytes, Sample, StreamId};
use pipes_kitti::calib::{Calib, ImageBox};
use pipes_kitti::camdet::{
    build_cam_det_batch, cam_det_storage_id, cam_det_view, class_name, nms, preprocess, Det, Model,
    COCO_CLASSES,
};
use pipes_kitti::detect::{
    cloud_points, detect_storage_id, detect_sweep_seq, detect_trigger_ns, detection_count,
    detection_rows, detections_f32, ground_plane, persistence, shape_of,
    source_point_count as detect_source_points, source_voxel_count as detect_source_voxels,
    DetectPlan, Detector, EgoMotion, Persistence, Shape, DETECTION_LANES, PERSISTENCE_GATE_M,
    RANGE_LIMIT_M,
};
use pipes_kitti::frame::{cam0_dims, cam0_pixels, cam0_storage_id};
use pipes_kitti::fuse::{
    camera_only_column, camera_only_rows, FusePlan, Fuser, POPULATION_FUSED, POPULATION_LIDAR_ONLY,
};
use pipes_kitti::layout::ABSENT_IN_SOURCE;
use pipes_kitti::state::{
    build_state_batch, cam_seq as state_cam_seq, candidates as state_candidates,
    fused_count as state_fused_count, has_object as state_has_object,
    in_frame_count as state_in_frame_count, object_rows, pair_age_ns as state_pair_age_ns,
    pair_outcome as state_pair_outcome, read_answer, record_is_consistent,
    source_point_count as state_source_points, source_tracks as state_source_tracks,
    state_storage_id, state_sweep_seq, Object, StateAnswer, StateMeta,
};
use pipes_kitti::track::{
    build_cam_ref_batch,
    cam_seq as track_cam_seq,
    cam_tov_ns as track_cam_tov_ns,
    pair_age_ns as track_pair_age_ns,
    pair_outcome as track_pair_outcome,
    read_cam_ref,
    source_detections as track_source_detections,
    source_point_count as track_source_points,
    source_voxel_count as track_source_voxels,
    track_age_s,
    track_detection,
    track_image_box,
    track_observations,
    track_population,
    track_rows,
    track_since_seen_s,
    track_storage_id,
    track_sweep_seq,
    track_trigger_ns,
    CamRef,
    Pairing,
    TrackMeta,
    // Renamed at the import so the name says what it tracks: `proc` once had
    // a 2D pixel-region tracker of the same name in this file.
    Tracker as ObjectTracker,
    TRACK_LANES,
};
use pipes_kitti::velo::{velo_point_count, velo_storage_id, velo_trigger_ns, velo_xyzr};
use pipes_kitti::voxel::{
    source_point_count as voxel_source_points, voxel_size_m, VoxelPlan, VoxelSize, Voxelizer,
};
use rerun::archetypes::{
    AnnotationContext, Boxes2D, Boxes3D, Clear, Image, Measurements, Points2D, Points3D, Scalars,
    TextDocument, TextLog,
};
use rerun::components::{Color, FillMode, ImageBuffer, MediaType, Radius};
use rerun::datatypes::{Blob, ChannelDatatype, ColorModel, Rgba32};
use serde::Serialize;

use crate::admission::Admission;
use crate::dashboard::{
    entity, ANSWER_LEAVES, FUSED_FRACTION_LEAF, PAIRING_BAND, PAIRING_LEAVES, TRACK_COUNT_LEAVES,
    TTC_TOP_S, TTC_WARN_S,
};
use crate::record::{
    band_archetype, series_archetype, EvRow, EvidenceSink, C_ANSWER, C_ATTENTION, C_CAMERA_ONLY,
    C_FUSED, C_LIDAR_ONLY,
};
use crate::run::RunCtx;
use crate::viewer::{Canvas, Subject};

/// `proc` stage configuration.
pub struct ProcCfg {
    /// Artificial per-frame consumer delay (plain sleep after the work):
    /// `--consumer-delay-ms` when no detector runs, 0 when one does.
    pub delay_ms: u64,
    /// Grayscale into one `Vec<u8>` owned by the thread instead of a new one per frame.
    pub reuse_output: bool,
    /// Test hook (`--panic-at-frame`): panic when this frame arrives, so the
    /// shipped binary's shutdown path can be exercised. `None` in every real run.
    pub panic_at_frame: Option<u64>,
    /// `--track on --detector off` only: admit a
    /// [`pipes_kitti::track::CamRef`] for every frame this stage FINISHES,
    /// onto `cam_det->track`.
    ///
    /// **`None` otherwise**, and the emission happens after `proc_end` on
    /// every path where it is `Some`, so `proc_bytes_per_frame` is 465750
    /// without flags and 0 with `--reuse-output` either way. What it costs is
    /// reported separately as [`ProcReport::cam_ref_bytes_total`] rather than
    /// folded into that figure.
    ///
    /// The camera half of a fusion has to come from a CONSUMER rather than
    /// straight off `Admission`, and that is the single most load-bearing
    /// wiring decision in this work: a fusion taking its camera input off
    /// `Admission` would pair 154 of 154 frames on a run where the camera
    /// consumer dropped half of them, and the experiment would demonstrate
    /// nothing. Under `--detector off` that consumer is this one; under
    /// `--detector on` it is [`camdet_thread`] on its own queue. `--cap`,
    /// `--policy` and `--consumer-delay-ms` follow it (`run::camera_queues`),
    /// so either way they reach the fusion, and with the detector on this
    /// stage does not run at all.
    pub cam_ref: Option<CamRefOut>,
}

/// Where `proc` sends its per-frame camera reference.
pub struct CamRefOut {
    /// The same `Admission` every producer uses, so the reference takes an
    /// `arrival_seq` in the one global order like anything else.
    pub admission: Arc<Admission>,
    /// [`pipes_kitti::track::cam_ref_schema`], built once.
    pub schema: Arc<Schema>,
}

/// What the `proc` thread returns.
pub struct ProcReport {
    pub delivered: u64,
    pub storage_mismatch: u64,
    pub bytes_total: u64,
    /// `(seq, storage_id re-derived from the batch)` per delivered frame.
    pub by_seq: Vec<(u64, usize)>,
    /// Camera references admitted onto `cam_det->track` (`--track on
    /// --detector off` only).
    pub cam_refs_produced: u64,
    /// Bytes building them cost, measured on this stage's own slot AFTER
    /// `proc_end` and deliberately NOT added to [`ProcReport::bytes_total`] —
    /// that figure is the frame's output buffer and nothing else.
    pub cam_ref_bytes_total: u64,
    /// References that could not be built (an Arrow error). Counted, never
    /// silently skipped.
    pub cam_ref_errors: u64,
}

/// One pixel of the fixed-point RGB -> luma weighted sum (77/150/29 >> 8).
/// Extracted so the test below pins the arithmetic the stage runs.
#[inline]
fn luma(p: &[u8; 3]) -> u8 {
    ((u32::from(p[0]) * 77 + u32::from(p[1]) * 150 + u32::from(p[2]) * 29) >> 8) as u8
}

/// Class ids of the fusion's populations, as the annotation context at
/// `camera/` and `lidar/` names and colours them -- the v2 layout's own
/// numbering, so a view can filter or recolour by population without reading
/// a single colour.
const CLASS_FUSED: u16 = 1;
/// See [`CLASS_FUSED`].
const CLASS_LIDAR_ONLY: u16 = 2;
/// See [`CLASS_FUSED`].
const CLASS_CAMERA_ONLY: u16 = 3;
/// See [`CLASS_FUSED`].
const CLASS_ANSWER: u16 = 9;

/// The annotation context both pictures carry: population by class id.
///
/// Every box also carries an explicit colour, so the context names more than
/// it colours; it is what makes "fused" a word the viewer knows rather than a
/// shade of green.
fn population_classes() -> AnnotationContext {
    let rgba = |c: [u8; 3]| Rgba32::from_rgb(c[0], c[1], c[2]);
    AnnotationContext::new([
        (CLASS_FUSED, "fused", rgba(C_FUSED)),
        (CLASS_LIDAR_ONLY, "lidar-only", rgba(C_LIDAR_ONLY)),
        (CLASS_CAMERA_ONLY, "camera-only", rgba(C_CAMERA_ONLY)),
        (CLASS_ANSWER, "answer", rgba(C_ANSWER)),
    ])
}

/// Alpha of a lidar-only outline on the camera picture: translucent enough
/// that forty of them do not hide the photograph they explain.
const LIDAR_ONLY_ALPHA: u8 = 120;

/// Alpha of a fused or camera-only outline: nearly opaque, because these are
/// the few boxes a reader is meant to read.
const LABELLED_ALPHA: u8 = 230;

/// Alpha of an in-frame record that was NOT seen this sweep: fainter than a
/// fresh one, because its box is last sweep's geometry drawn over this
/// sweep's frame.
const COASTED_OUTLINE_ALPHA: u8 = 60;

/// Outline width of a lidar-only or camera-only box on the picture, in UI
/// points rather than pixels, so it stays a 1 px line at any zoom instead of
/// growing into a slab when the reader zooms in.
const TRACK_OUTLINE_PT: f32 = 0.8;

/// Outline width of a fused box: heavier than the crowd's hairline, lighter
/// than the answer's slab.
const FUSED_OUTLINE_PT: f32 = 1.5;

/// How far outside a fused box its amber stale stroke is drawn, in pixels, on
/// a frame the fusion paired stale: the green stays -- the population is
/// still the population -- and the amber says its class came from an older
/// frame than the picture under it.
const STALE_STROKE_PX: f32 = 3.0;

/// Outline width of the ONE answer box, in UI points: thick at any zoom,
/// because it is the one rectangle on the picture that is a decision.
const ANSWER_OUTLINE_PT: f32 = 3.5;

/// The answer's 3D wireframe, in UI points: the one box among two hundred
/// in the lidar view that is a decision, so the heaviest line there.
const ANSWER_3D_OUTLINE_PT: f32 = 4.0;

/// The point that carries the answer's label on the picture, in UI points:
/// a hair, so the words show and the point does not.
const ANSWER_LABEL_POINT_PT: f32 = 0.1;

/// Where the camera status line sits on the image, pixels: the top-left
/// corner, clear of the road.
const STATUS_AT: [f32; 2] = [150.0, 16.0];

/// The status line's colour on a clean pair: a quiet grey.
const C_STATUS: [u8; 3] = [200, 200, 200];

/// The camera stamp's colour on a sweep whose set expired: the failure red,
/// #E5484D, the colour of `expired` on the pairing lane.
const C_EXPIRED: [u8; 3] = [229, 72, 77];

/// The camera picture's opacity while the fusion's camera half is stale: dim
/// enough that the boxes and the amber stamp read as the subject, bright
/// enough that the scene is still recognisable.
const STALE_IMAGE_OPACITY: f32 = 0.35;

/// How the answer's object is moving, in the words its labels and the
/// headline use: `TTC 3.1 s` while it closes, `receding` while the gap grows,
/// and `not closing` for a closing speed of exactly zero, which is neither.
/// `with_speed` puts the closing speed in front of the TTC.
fn answer_motion(a: &StateAnswer, with_speed: bool) -> String {
    if a.ttc_s > 0.0 && a.closing_mps > 0.0 {
        if with_speed {
            format!("closing {:.1} m/s · TTC {:.1} s", a.closing_mps, a.ttc_s)
        } else {
            format!("TTC {:.1} s", a.ttc_s)
        }
    } else if a.closing_mps < 0.0 {
        "receding".to_string()
    } else {
        "not closing".to_string()
    }
}

/// The camera's class and confidence as a label's lead, `car 0.91 · `, or
/// nothing when the fusion gave the object no class.
fn class_lead(class: Option<(u32, f32)>) -> String {
    class.map_or_else(String::new, |(k, q)| format!("{} {q:.2} · ", class_name(k)))
}

/// The answer as a label on the picture: short enough to sit under its box.
///
/// The camera's class and confidence when the fusion gave it one, then the
/// distance and how it moves: `car 0.91 · 11.4 m · TTC 3.1 s`. Without a
/// class the closing speed takes the class's place, `11.4 m · closing 3.7
/// m/s · TTC 3.1 s`. No track id: a reader follows the gold box, not a
/// number, and the id is in `answer/line`. The full sentence
/// ([`StateAnswer::line`]) wrapped into a six-line blob over the object it
/// named, so it stays in the log.
fn answer_label(a: &StateAnswer) -> String {
    format!(
        "{}{:.1} m · {}",
        class_lead(a.class),
        a.distance_m,
        answer_motion(a, a.class.is_none())
    )
}

/// The answer's label in 3D, beside the gold wireframe: the distance and the
/// TTC only (`11.4 m · TTC 3.1 s`, `11.3 m · receding`), because the camera
/// picture above it carries the class.
fn answer_label_3d(a: &StateAnswer) -> String {
    format!("{:.1} m · {}", a.distance_m, answer_motion(a, false))
}

/// Where the answer's label point goes on a `w` x `h` picture: the middle of
/// the box's bottom edge -- the viewer draws a point's label under the point,
/// so the words sit just below the box -- kept at least [`LABEL_EDGE_PX`]
/// from either side and inside the picture top to bottom, so a box at the
/// frame's edge does not push its words off it. Unclamped when the frame's
/// size is unknown.
fn answer_label_at(b: &ImageBox, image_wh: Option<(f32, f32)>) -> [f32; 2] {
    let (x, y) = (b.x0 + 0.5 * b.w(), b.y1);
    match image_wh {
        Some((w, h)) if w > 2.0 * LABEL_EDGE_PX => [
            x.clamp(LABEL_EDGE_PX, w - LABEL_EDGE_PX),
            y.clamp(LABEL_TOP_PX, (h - LABEL_BOTTOM_PX).max(LABEL_TOP_PX)),
        ],
        _ => [x, y],
    }
}

/// Whether two point labels, each drawn centred under its anchor point, `n`
/// characters wide, would print over each other on the camera picture.
/// Approximate: [`LABEL_CHAR_PX`] a character and [`LABEL_LINE_PX`] a line,
/// in image pixels, as the viewer draws them at 1600x900.
fn labels_collide(a: [f32; 2], a_chars: usize, b: [f32; 2], b_chars: usize) -> bool {
    let half_widths = 0.5 * LABEL_CHAR_PX * (a_chars + b_chars) as f32;
    (a[0] - b[0]).abs() < half_widths && (a[1] - b[1]).abs() < LABEL_LINE_PX
}

/// A label character's width on the camera picture, in image pixels: the
/// viewer draws labels at a fixed size on screen, about 6 px a character,
/// and the 1242 px frame is 990 px wide on screen at 1600x900.
const LABEL_CHAR_PX: f32 = 7.5;

/// A label line's height on the camera picture, in image pixels, on the
/// same terms.
const LABEL_LINE_PX: f32 = 18.0;

/// How far the answer label's point is kept from the picture's left and
/// right edges, pixels: half the width of its longest label at 1600x900.
const LABEL_EDGE_PX: f32 = 190.0;
/// The highest the answer label's point may sit, pixels from the top.
const LABEL_TOP_PX: f32 = 22.0;
/// The lowest, pixels from the bottom: room for one line of label text
/// under the point.
const LABEL_BOTTOM_PX: f32 = 14.0;

/// A lidar-only object's label, shown on hover: its id, its distance and how
/// long it has been tracked -- what a reader needs to follow an object from
/// frame to frame and to judge how much its closing speed is worth.
///
/// Carried, not drawn: drive_0005 puts 43.7 records in the frame on a sweep
/// and up to 68, and with every label drawn at 1600x900 they covered the
/// photograph. The labels drawn are the fused objects' and the camera-only
/// detections' -- a handful a frame -- and the answer's.
fn object_label(o: &Object) -> String {
    format!("#{} · {:.1} m · {:.1} s", o.id, o.distance_m, o.age_s)
}

/// A fused object's label, drawn: what the camera says it is and how sure it
/// is, and what the lidar says about it -- `class confidence · distance ·
/// closing · age`, one line, the closing speed signed as the answer's label
/// signs it (positive while the gap shrinks). The class and confidence lead
/// as the v2 layout's `car 0.91` has them, and as the answer's own label
/// does: without the confidence a fused object's camera score was visible
/// only on the one object the answer flagged. Kept to the numbers and their
/// units: several of these share one strip of road on the picture, and the
/// words that once spelled out "closing" and "receding" made them overlap.
fn fused_label(o: &Object) -> String {
    let class = o.class.map_or_else(
        || "?".to_string(),
        |(k, q)| format!("{} {q:.2}", class_name(k)),
    );
    format!(
        "{class} · {:.1} m · {:.1} m/s · {:.1} s",
        o.distance_m, o.closing_mps, o.age_s
    )
}

/// A camera-only detection's label, drawn: its class and score, and the fact
/// that it has no range -- the one thing a reader must not assume it has.
fn camera_only_label(class_id: u32, confidence: f32) -> String {
    format!("{} {confidence:.2} · no range", class_name(class_id))
}

/// The camera status line: which frame the boxes on the picture were fused
/// with, and how its instant sat against the sweep's trigger -- the paired
/// frame about +10.5 ms after it, a stale one a period or more before.
fn camera_status(cam_seq: i64, pair_age_ns: i64, stale: bool) -> String {
    let ms = pair_age_ns as f64 * 1e-6;
    if stale {
        format!("STALE · cam {cam_seq} · {ms:+.0} ms")
    } else {
        format!("cam {cam_seq} · paired {ms:+.0} ms")
    }
}

/// Where a picture of one sweep sits on the recording's `seq` timeline: at
/// the SWEEP's number, read out of the payload, so every picture of one sweep
/// -- the raw cloud, the voxels, the detections, the tracks, the answer --
/// lines up with the others and with the camera frame of the same number,
/// whichever stage drew it and however many sweeps were lost upstream.
///
/// A derived sample's own `seq` is that same number: every stage numbers its
/// output by the frame it came from. The payload's is read first because it
/// is the one the batch itself carries; `own` is used only when the sample
/// names no sweep (-1).
fn sweep_timeline(sweep_seq: Option<i64>, own: u64) -> i64 {
    sweep_seq.filter(|s| *s >= 0).unwrap_or(own as i64)
}

/// How stale an answer's camera half was, in words: `STALE camera 93 ms`,
/// how long before the sweep's trigger the paired frame was taken -- the age
/// the payload's `pair_age_ns` carries and the picture's stamp shows. `None`
/// on a clean pair, the one other outcome that produces an answer.
pub fn stale_note(pair_outcome: &str, pair_age_ns: i64) -> Option<String> {
    (pair_outcome == Pairing::Stale.name())
        .then(|| format!("STALE camera {:.0} ms", -(pair_age_ns as f64) * 1e-6))
}

/// Where an answer's camera half came from: the frame, how the sweep was
/// paired with it, and how old it was. Copied beside an answer that is kept
/// for later, so its words can still say so once the batch is gone.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CamHalf {
    /// The paired frame, or -1.
    pub cam_seq: i64,
    /// `Pairing::name` of how it was paired.
    pub pair_outcome: &'static str,
    /// Its instant minus the sweep's trigger, ns: negative when stale.
    pub pair_age_ns: i64,
}

impl CamHalf {
    /// The camera half an answer batch names, off its own columns.
    pub fn of_answer(payload: &RecordBatch) -> CamHalf {
        CamHalf {
            cam_seq: state_cam_seq(payload).unwrap_or(-1),
            pair_outcome: pairing_name(state_pair_outcome(payload)),
            pair_age_ns: state_pair_age_ns(payload).unwrap_or(0),
        }
    }
}

/// An answer's sentence as the run prints it and `summary.json` keeps it:
/// the flagged record's words ([`StateAnswer::line`]), led by how stale the
/// camera was and which frame it was when its half was stale. The class in
/// that sentence is the older frame's, and a reader who has only the words
/// must be able to tell; on a clean pair it is the sentence alone.
pub fn answer_words(a: &StateAnswer, cam: CamHalf) -> String {
    match stale_note(cam.pair_outcome, cam.pair_age_ns) {
        Some(note) => format!("{note} (frame {}): {}", cam.cam_seq, a.line()),
        None => a.line(),
    }
}

/// The answer as one line of the log: the sweep it is about, the camera
/// frame that sweep was paired with -- and, when that frame was stale, how
/// stale, right beside it -- what the answer holds -- how many
/// objects, how many of them in the camera frame, how many of those the
/// camera confirmed, how many detections only the camera holds, how many in
/// the vehicle's path -- and the flagged record's sentence. The counts go before the
/// sentence because the panel cuts the end of a long line, and the end of
/// the sentence is the image rectangle, which the picture beside it shows.
///
/// Every number is read out of the state batch, the words' own source: the
/// batch's `source_sweep_seq` is the sweep the whole chain inherited, and the
/// sample's `seq` is the same number (a derived sample carries its source
/// frame's), so the line and the lanes agree.
fn answer_line(payload: &RecordBatch, a: &StateAnswer) -> String {
    let cam = CamHalf::of_answer(payload);
    let stale = stale_note(cam.pair_outcome, cam.pair_age_ns)
        .map_or_else(String::new, |note| format!(" | {note}"));
    format!(
        "sweep {} | cam frame {}{stale} | {} tracked, {} in frame, {} fused, {} camera-only, {} in path | {}",
        state_sweep_seq(payload).unwrap_or(-1),
        cam.cam_seq,
        object_rows(payload).len(),
        state_in_frame_count(payload).unwrap_or(0),
        state_fused_count(payload).unwrap_or(0),
        camera_only_rows(payload).len(),
        state_candidates(payload).unwrap_or(0),
        a.line(),
    )
}

/// Body of the `proc` thread: grayscale every frame, one counted allocation
/// per frame (`w*h` bytes) or none with `reuse_output`.
pub fn proc_thread(
    q: Arc<BoundedQueue<Arc<Sample>>>,
    sink: Arc<EvidenceSink>,
    ctx: Arc<RunCtx>,
    cfg: ProcCfg,
) -> ProcReport {
    set_stage_slot(SLOT_PROC);
    let rc = ctx.row_ctx();
    let mut report = ProcReport {
        delivered: 0,
        storage_mismatch: 0,
        bytes_total: 0,
        by_seq: Vec::with_capacity(ctx.n_frames),
        cam_refs_produced: 0,
        cam_ref_bytes_total: 0,
        cam_ref_errors: 0,
    };
    let mut out: Vec<u8> = Vec::new();
    let mut reuse_announced = false;
    while let Some(env) = q.pop() {
        let s: &Sample = &env.item;
        // Fault injection for `panic_in_proc_shuts_down_cleanly_and_exits_3`:
        // one `Option` compare per frame, never `Some` in a real run.
        if cfg.panic_at_frame == Some(s.seq) {
            panic!("injected panic at frame {}", s.seq);
        }
        let dequeued = now();
        let (w, h) = cam0_dims(&s.payload).unwrap_or((0, 0));
        let px: &[u8] = cam0_pixels(&s.payload).map_or(&[], |b| b.as_slice());
        let n = w as usize * h as usize;
        if cfg.reuse_output && out.len() != n {
            // One-time setup, outside the measured window.
            out.resize(n, 0);
            if !reuse_announced {
                println!("proc reuse buffer = {n} bytes");
                reuse_announced = true;
            }
        }
        let proc_start = now();
        let b0 = bytes_alloc(SLOT_PROC);
        if !cfg.reuse_output {
            // The one counted allocation per frame (alloc_zeroed of w*h bytes).
            out = vec![0u8; n];
        }
        let (rgb, _) = px.as_chunks::<3>();
        for (o, p) in out.iter_mut().zip(rgb) {
            *o = luma(p);
        }
        std::hint::black_box(&out);
        let bytes = bytes_alloc(SLOT_PROC) - b0;
        if cfg.delay_ms > 0 {
            std::thread::sleep(Duration::from_millis(cfg.delay_ms));
        }
        let proc_end = now();

        // Everything below this line is outside the measured window. The
        // grayscale frame itself is not drawn: it was a grey copy of the
        // camera image whose only reason was the `--proc-work` knob, and it
        // cost 11% of a recording to say "the consumer ran".

        // The camera half of the fusion, admitted as its own stream.
        // **After `proc_end`**, so the measured window is byte for byte what
        // it was before this existed, and counted on its own so the cost is
        // stated rather than hidden. Only a frame this stage actually
        // FINISHED produces one, which is what makes an induced drop here
        // propagate into the fusion instead of being invisible to it.
        if let Some(out) = &cfg.cam_ref {
            let c0 = bytes_alloc(SLOT_PROC);
            let r = CamRef {
                seq: s.seq,
                tov_ns: s.tov.start().map_or(0, |t| t.0),
                width: w,
                height: h,
            };
            match build_cam_ref_batch(r, &out.schema) {
                Ok(batch) => {
                    let derived = Sample {
                        stream: StreamId::CAM_DET,
                        seq: s.seq,
                        arrival_seq: 0,
                        parent: Some((s.stream, s.seq)),
                        tov: s.tov,
                        epoch: s.epoch,
                        due: s.due,
                        arrival: now(),
                        payload: batch,
                        // No shared buffer to name: see `cam_ref_schema`.
                        storage_id: 0,
                        decode_ns: 0,
                    };
                    let admitted = out.admission.admit(derived);
                    let mut prow = Evidence::driver_admitted(&rc, &admitted, "cam_det", "proc");
                    prow.bytes_alloc = bytes_alloc(SLOT_PROC) - c0;
                    sink.send(EvRow::Evidence(prow), &ctx, "proc");
                    report.cam_refs_produced += 1;
                }
                Err(e) => {
                    eprintln!("proc: frame {}: camera reference: {e}", s.seq);
                    report.cam_ref_errors += 1;
                }
            }
            report.cam_ref_bytes_total += bytes_alloc(SLOT_PROC) - c0;
        }

        let derived = cam0_storage_id(&s.payload);
        if derived != Some(s.storage_id) {
            report.storage_mismatch += 1;
        }
        let row = Evidence::delivered(
            &rc,
            s,
            "cam0->proc",
            "proc",
            env.enqueued,
            env.depth_at_push,
            env.depth_after_pop,
            dequeued,
            proc_start,
            proc_end,
            derived.unwrap_or(0),
            bytes,
        );
        sink.send(EvRow::Evidence(row), &ctx, "proc");
        report.delivered += 1;
        report.bytes_total += bytes;
        report.by_seq.push((s.seq, derived.unwrap_or(0)));
    }
    report
}

/// Where the `rerun` consumer sends frames. `Null` is the second consumer
/// without a viewer or file: measured, nothing written.
pub enum RerunMode {
    Off,
    Null,
    Rrd(PathBuf),
    Grpc,
}

impl RerunMode {
    pub fn name(&self) -> &'static str {
        match self {
            RerunMode::Off => "off",
            RerunMode::Null => "null",
            RerunMode::Rrd(_) => "rrd",
            RerunMode::Grpc => "grpc",
        }
    }
}

/// What the `rerun` thread returns.
pub struct RerunReport {
    pub delivered: u64,
    pub storage_mismatch: u64,
    pub bytes_total: u64,
    /// `(seq, storage_id re-derived after wrapping for Rerun)` per delivered frame.
    pub by_seq: Vec<(u64, usize)>,
}

/// The one `pip` line that puts a viewer matching this binary's SDK on PATH.
///
/// The version is pinned to the SDK the binary was built against, because
/// rerun makes no compatibility promise across versions and a mismatched
/// viewer fails in ways that look like pipeline bugs.
pub const VIEWER_INSTALL: &str = "python -m pip install rerun-sdk==0.38.1";

/// Errors of the `rerun` thread.
#[derive(Debug)]
pub enum RerunError {
    Stream(rerun::RecordingStreamError),
    Flush(String),
    /// `--rerun grpc` found no viewer listening on `port` and no `rerun`
    /// executable on PATH to start one. Its own error, rather than the SDK's
    /// wrapped one, so the message can say the one command that fixes it on
    /// this project.
    NoViewer {
        port: u16,
    },
    /// `--rerun grpc` started `rerun` from PATH and it was still not
    /// listening on `port` when the run's wait for it ran out. Raised before
    /// the clock starts: a run that replayed into a viewer that is not
    /// listening lost more than half its frames and still exited 0.
    NotListening {
        port: u16,
        /// The process `rerun::spawn` started.
        pid: u32,
        /// How long the run waited, seconds.
        waited_s: f64,
    },
}

impl std::fmt::Display for RerunError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RerunError::Stream(e) => write!(f, "rerun: {e}"),
            RerunError::Flush(e) => write!(f, "rerun flush: {e}"),
            RerunError::NoViewer { port } => write!(
                f,
                "no Rerun viewer: nothing is listening on port {port} and there is no `rerun` \
                 executable on PATH to start one. Install it with `{VIEWER_INSTALL}` (the \
                 Python package puts `rerun` on PATH), or pass `--rerun rrd` to write the \
                 recording to a file instead"
            ),
            RerunError::NotListening {
                port,
                pid,
                waited_s,
            } => write!(
                f,
                "the Rerun viewer this run started (`rerun` from PATH, pid {pid}) was not \
                 listening on port {port} after {waited_s:.0} s, so the run stopped before its \
                 clock started rather than replay into a viewer that cannot take the data. \
                 Check that `rerun --version` prints 0.38.1 (install it with \
                 `{VIEWER_INSTALL}`), answer the Windows Firewall prompt if one is open, and run \
                 again; or pass `--rerun rrd` to write the recording to a file instead"
            ),
        }
    }
}

impl std::error::Error for RerunError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            RerunError::Stream(e) => Some(e),
            RerunError::Flush(_)
            | RerunError::NoViewer { .. }
            | RerunError::NotListening { .. } => None,
        }
    }
}

impl From<rerun::RecordingStreamError> for RerunError {
    fn from(e: rerun::RecordingStreamError) -> Self {
        RerunError::Stream(e)
    }
}

/// Body of the `rerun` thread: wrap the shared pixel buffer as a Rerun
/// `Image` (no copy) on three timelines, draw it (or not — `rec` is `None`
/// under `--rerun null`), and write this stage's own evidence row. The
/// picture is queued for the viewer thread ([`crate::viewer`]), which is the
/// one that calls the SDK: the measured window is building the image, and
/// handing it over never waits.
///
/// `clear_detections` (`--detector on` with the dashboard drawing): after each
/// image, outside the measured window, clear `camera/cam_det` at that frame's
/// own instant. The detector draws its boxes at the same instant ~90 ms later,
/// and the later row wins; a frame the detector never saw keeps the clear. This
/// stage does it because it is the one that sees every frame the picture
/// shows -- without it, a frame the detector skipped would show the previous
/// frame's boxes over a newer image, a stale input drawn as a fresh one.
pub fn rerun_thread(
    q: Arc<BoundedQueue<Arc<Sample>>>,
    sink: Arc<EvidenceSink>,
    ctx: Arc<RunCtx>,
    rec: Option<Canvas>,
    clear_detections: bool,
) -> Result<RerunReport, RerunError> {
    set_stage_slot(SLOT_RERUN);
    let rc = ctx.row_ctx();
    let mut report = RerunReport {
        delivered: 0,
        storage_mismatch: 0,
        bytes_total: 0,
        by_seq: Vec::with_capacity(ctx.n_frames),
    };
    while let Some(env) = q.pop() {
        let s: &Sample = &env.item;
        let dequeued = now();
        let (w, h) = cam0_dims(&s.payload).unwrap_or((0, 0));
        let tov_start_ns = s.tov.start().map_or(0, |t| t.0);

        let proc_start = now();
        let b0 = bytes_alloc(SLOT_RERUN);
        let (ib, derived) = match cam0_pixels(&s.payload) {
            Some(buf) => {
                let ib = ImageBuffer(Blob::from(buf.clone()));
                let derived = ib.0 .0.inner().as_ptr() as usize;
                (Some(ib), Some(derived))
            }
            None => (None, None),
        };
        if let (Some(rec), Some(ib)) = (&rec, ib) {
            rec.set_timestamp_nanos_since_epoch("sensor_time", tov_start_ns);
            rec.set_duration_secs("host", (s.arrival - ctx.t0_host) as f64 * 1e-9);
            rec.set_time_sequence("seq", s.seq as i64);
            rec.log(
                entity::CAMERA_IMAGE,
                &Image::from_color_model_and_bytes(
                    ib,
                    [w, h],
                    ColorModel::RGB,
                    ChannelDatatype::U8,
                ),
            )?;
        }
        let bytes = bytes_alloc(SLOT_RERUN) - b0;
        let proc_end = now();

        // On the timelines the image was just logged at, which this canvas
        // still holds. See the function docs.
        if let (true, Some(rec)) = (clear_detections, &rec) {
            rec.log(entity::CAMERA_DET, &Clear::flat())?;
        }
        // The frame and its clear, as one drawing: taken or dropped together.
        if let Some(rec) = &rec {
            rec.send(Subject::of_sample(&rc, s));
        }
        if derived != Some(s.storage_id) {
            report.storage_mismatch += 1;
        }
        let row = Evidence::delivered(
            &rc,
            s,
            "cam0->rerun",
            "rerun",
            env.enqueued,
            env.depth_at_push,
            env.depth_after_pop,
            dequeued,
            proc_start,
            proc_end,
            derived.unwrap_or(0),
            bytes,
        );
        sink.send(EvRow::Evidence(row), &ctx, "rerun");
        report.delivered += 1;
        report.bytes_total += bytes;
        report.by_seq.push((s.seq, derived.unwrap_or(0)));
    }
    drop(rec);
    Ok(report)
}

/// Colour of the detector's own boxes on the camera picture, #9B7BD8: the
/// dashboard's "detection" purple, which no track, answer or population
/// colour uses, so a box the camera found cannot be read as one the lidar
/// tracked.
const C_CAM_DET: [u8; 3] = [155, 123, 216];

/// Outline width of a detector box, in UI points: heavier than a track's
/// hairline, lighter than the answer's slab.
const CAM_DET_OUTLINE_PT: f32 = 1.5;

/// A detector box's label: class and confidence, `car 0.91`, and nothing
/// else -- which object it is and how sure the model was is all the camera
/// knows about it.
fn cam_det_label(d: &Det) -> String {
    format!("{} {:.2}", class_name(d.class_id), d.score)
}

/// `camdet` stage configuration.
pub struct CamDetCfg {
    /// The network, loaded and checked against its pinned sha256 before the
    /// clock started (D16), and moved here: nothing else runs it.
    pub model: Model,
    /// [`pipes_kitti::camdet::cam_det_schema`], built once; its metadata
    /// carries the model's name and sha256 on every batch.
    pub schema: Arc<Schema>,
    /// Capacity hint for the per-frame vectors.
    pub expect: usize,
    /// `--dashboard` only: draw each frame's detections on the camera picture
    /// (`camera/cam_det`), after the measured window has closed. `None` on
    /// the default path, so the measured stage is byte for byte what it is
    /// without a viewer.
    pub rec: Option<Canvas>,
    /// `--consumer-delay-ms`: an artificial per-frame sleep after the batch
    /// is built and before `proc_end`, so the evidence shows it as service
    /// time, exactly as `proc`'s did. 0 unless asked for.
    pub delay_ms: u64,
}

/// What the `camdet` stage returns.
pub struct CamDetReport {
    /// Frames popped off `cam0->camdet`.
    pub delivered: u64,
    /// `CAM_DET` batches admitted onto `cam_det->track`.
    pub produced: u64,
    /// Frames the detector could not turn into a batch: a frame of the wrong
    /// size, a runtime error, an Arrow error. `produced + errors ==
    /// delivered`, and the run checks it.
    pub errors: u64,
    /// Frames whose re-derived pixel-buffer address disagreed with the
    /// envelope's.
    pub storage_mismatch: u64,
    /// `(frame seq, storage_id re-derived from the frame)`: the input half of
    /// the zero-copy proof, joined against the camera driver's own rows.
    pub in_by_seq: Vec<(u64, usize)>,
    /// `(cam_det seq, the frame seq it names as its parent)`.
    pub out_parent: Vec<(u64, u64)>,
    /// Bytes allocated turning each frame into detections -- the letterbox,
    /// the tensor, the network's own buffers, the candidates -- summed. The
    /// `cam0->camdet` row's `bytes_alloc`.
    ///
    /// **Megabytes per frame, and that is the honest number.** The frame is
    /// read in place, but a network cannot read an RGB8 buffer, so the
    /// preprocessing copies it (about 5.7 MB a frame on KITTI) and the
    /// runtime allocates its activations (about 75 MB more, measured on
    /// drive_0005). Every other consumer of a shared buffer in this run
    /// allocates next to nothing; this one does not, and says so.
    pub frame_bytes_total: u64,
    /// Of which preprocessing: the resize's intermediate, the resized image
    /// and the float tensor, three real copies of the frame's content.
    pub preprocess_bytes_total: u64,
    /// Bytes building the Arrow batch cost: the `cam_det` row's
    /// `bytes_alloc`.
    pub build_bytes_total: u64,
    /// Arrow bytes carried in (the frame) over frames delivered.
    pub in_payload_bytes_total: u64,
    /// Arrow bytes carried out (the detections) over batches produced.
    pub out_payload_bytes_total: u64,
    /// Per frame, ns: the letterbox and the tensor.
    pub preprocess_ns: Vec<i64>,
    /// Per frame, ns: the network, the decode of its output and the NMS.
    pub infer_ns: Vec<i64>,
    /// Per frame, ns: `proc_end - proc_start`, everything the stage did to a
    /// frame. Against the camera's 103 ms period this is the stage's budget.
    pub service_ns: Vec<i64>,
    /// Detections kept, over the batches produced.
    pub detections_total: u64,
    /// The same, by class id (an index into `COCO_CLASSES`).
    pub class_counts: Vec<u64>,
    /// Every batch produced, in order, for `runs/<name>/cam_det.arrows`: a
    /// clone of the admitted batch, which shares its buffers rather than
    /// copying them.
    pub batches: Vec<RecordBatch>,
    /// `--dashboard` only: bytes drawing the boxes cost, measured on this
    /// stage's own slot AFTER `proc_end`.
    pub viz_bytes_total: u64,
    /// Nanoseconds the same drawing cost.
    pub viz_ns_total: i64,
    /// Frames the viewer would not accept. The pipeline is unaffected.
    pub viz_log_errors: u64,
}

impl CamDetReport {
    /// Detections per batch produced.
    pub fn detections_mean(&self) -> Option<f64> {
        (self.produced > 0).then(|| self.detections_total as f64 / self.produced as f64)
    }

    /// Arrow bytes carried out per batch produced.
    pub fn out_payload_bytes_mean(&self) -> Option<u64> {
        (self.produced > 0).then(|| self.out_payload_bytes_total / self.produced)
    }

    /// Frames whose service took longer than `period_ns`: the frames this
    /// stage could not have kept up with on its own.
    pub fn over_period(&self, period_ns: i64) -> usize {
        self.service_ns.iter().filter(|ns| **ns > period_ns).count()
    }

    /// `(class name, count)` for every class seen, most frequent first, ties
    /// by class id.
    pub fn classes(&self) -> Vec<(&'static str, u64)> {
        let mut v: Vec<(usize, u64)> = self
            .class_counts
            .iter()
            .copied()
            .enumerate()
            .filter(|(_, c)| *c > 0)
            .collect();
        v.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        v.into_iter()
            .map(|(k, c)| (class_name(k as u32), c))
            .collect()
    }
}

/// Body of the `camdet` thread: **the frozen camera detector**, and the first
/// stage in this project whose output says what is IN the picture.
///
/// It consumes `cam0->camdet`, a fan-out of the camera frame beside
/// `cam0->rerun` and in place of `cam0->proc` (`proc` does not run beside
/// it), and admits one `CAM_DET` batch per frame it
/// finishes onto `cam_det->track`: the frame's reference (seq, instant,
/// bounds) and its detections (box, class, confidence). The sample inherits
/// the frame's `tov` and `due` and names it as its parent, so the fusion pairs
/// it by the instant the picture was taken, never by when the network got to
/// it -- a detection stamped with its own completion time would pair a stale
/// picture as a fresh one, silently.
///
/// One thread, one frame at a time, in order: the fusion's "a later frame
/// arrived, so the one I wanted was lost" inference needs the batches in frame
/// order.
///
/// # The two measured windows
///
/// As in [`detect_thread`], with the first one's meaning turned round. The
/// `cam0->camdet` row carries what looking at the frame cost, and it is NOT
/// 0: the letterbox and the tensor are copies the network needs, and its
/// activations are its own. The `cam_det` row carries the batch.
///
/// # Keeping up
///
/// At ~90 ms a frame against a 103 ms period it keeps up at the median and
/// not always at the tail. It never stalls the camera: its queue is a Drop*
/// edge by default (`--policy`; `--policy block` is the negative control), so
/// a frame it has not reached when the next arrives is evicted at admission,
/// with a drop row that says so, and the fusion then reports that sweep's
/// camera half as dropped. `--consumer-delay-ms` slows it on purpose, and
/// `--cap` sizes that queue (`run::camera_queues`).
pub fn camdet_thread(
    q: Arc<BoundedQueue<Arc<Sample>>>,
    admission: Arc<Admission>,
    sink: Arc<EvidenceSink>,
    ctx: Arc<RunCtx>,
    cfg: CamDetCfg,
) -> CamDetReport {
    set_stage_slot(SLOT_CAMDET);
    let rc = ctx.row_ctx();
    let mut report = CamDetReport {
        delivered: 0,
        produced: 0,
        errors: 0,
        storage_mismatch: 0,
        in_by_seq: Vec::with_capacity(cfg.expect),
        out_parent: Vec::with_capacity(cfg.expect),
        frame_bytes_total: 0,
        preprocess_bytes_total: 0,
        build_bytes_total: 0,
        in_payload_bytes_total: 0,
        out_payload_bytes_total: 0,
        preprocess_ns: Vec::with_capacity(cfg.expect),
        infer_ns: Vec::with_capacity(cfg.expect),
        service_ns: Vec::with_capacity(cfg.expect),
        detections_total: 0,
        class_counts: vec![0; COCO_CLASSES.len()],
        batches: Vec::with_capacity(cfg.expect),
        viz_bytes_total: 0,
        viz_ns_total: 0,
        viz_log_errors: 0,
    };
    while let Some(env) = q.pop() {
        let s: &Sample = &env.item;
        let dequeued = now();
        let (w, h) = cam0_dims(&s.payload).unwrap_or((0, 0));
        // Borrowed, not copied: the driver's pixel buffer, kept alive by the
        // `Arc<Sample>` this stage holds.
        let px: &[u8] = cam0_pixels(&s.payload).map_or(&[], |b| b.as_slice());
        let frame = CamRef {
            seq: s.seq,
            tov_ns: s.tov.start().map_or(0, |t| t.0),
            width: w,
            height: h,
        };

        let proc_start = now();
        // Window 1 -- looking at the frame. Megabytes, and meant to be.
        let b0 = bytes_alloc(SLOT_CAMDET);
        let pre = preprocess(px, w, h);
        let pre_bytes = bytes_alloc(SLOT_CAMDET) - b0;
        let t_pre = now();
        let found = pre.and_then(|(tensor, r)| cfg.model.infer(tensor, r, w, h).map(nms));
        let t_inf = now();
        let frame_bytes = bytes_alloc(SLOT_CAMDET) - b0;
        // Window 2 -- building the batch.
        let b1 = bytes_alloc(SLOT_CAMDET);
        let built = found.map_err(|e| e.to_string()).and_then(|dets| {
            build_cam_det_batch(frame, &dets, &cfg.schema)
                .map(|b| (b, dets))
                .map_err(|e| e.to_string())
        });
        let build_bytes = bytes_alloc(SLOT_CAMDET) - b1;
        // The slowed camera, when `--consumer-delay-ms` asks for it: inside
        // the window, after both byte counts, so it is service time and
        // allocates nothing that could be mistaken for the detector's.
        if cfg.delay_ms > 0 {
            std::thread::sleep(Duration::from_millis(cfg.delay_ms));
        }
        let proc_end = now();

        // Everything below is outside the measured windows.
        let in_storage = cam0_storage_id(&s.payload);
        if in_storage != Some(s.storage_id) {
            report.storage_mismatch += 1;
        }
        let row = Evidence::delivered(
            &rc,
            s,
            "cam0->camdet",
            "camdet",
            env.enqueued,
            env.depth_at_push,
            env.depth_after_pop,
            dequeued,
            proc_start,
            proc_end,
            in_storage.unwrap_or(0),
            frame_bytes,
        );
        sink.send(EvRow::Evidence(row), &ctx, "camdet");
        report.delivered += 1;
        report.frame_bytes_total += frame_bytes;
        report.preprocess_bytes_total += pre_bytes;
        report.in_payload_bytes_total += s.payload_bytes() as u64;
        report.in_by_seq.push((s.seq, in_storage.unwrap_or(0)));
        report.preprocess_ns.push(t_pre - proc_start);
        report.infer_ns.push(t_inf - t_pre);
        report.service_ns.push(proc_end - proc_start);

        let (batch, dets) = match built {
            Ok(x) => x,
            Err(e) => {
                // Counted, never silently skipped: `produced + errors ==
                // delivered` is checked at the end of the run.
                eprintln!("camdet: frame {}: {e}", s.seq);
                report.errors += 1;
                continue;
            }
        };
        report.build_bytes_total += build_bytes;
        report.detections_total += dets.len() as u64;
        for d in &dets {
            if let Some(c) = report.class_counts.get_mut(d.class_id as usize) {
                *c += 1;
            }
        }
        report.out_payload_bytes_total += payload_bytes(&batch) as u64;

        // The boxes, on the frame they were found in: the image's own three
        // timelines, so they scrub with it. After `proc_end`, on this stage's
        // own slot, and costed.
        if let Some(rec) = &cfg.rec {
            let t_viz = now();
            let v0 = bytes_alloc(SLOT_CAMDET);
            rec.set_timestamp_nanos_since_epoch("sensor_time", frame.tov_ns);
            rec.set_duration_secs("host", (s.arrival - ctx.t0_host) as f64 * 1e-9);
            rec.set_time_sequence("seq", s.seq as i64);
            let logged = if dets.is_empty() {
                // "Looked and saw nothing" is drawn as nothing.
                rec.log(entity::CAMERA_DET, &Clear::flat())
            } else {
                let color = Color::from_rgb(C_CAM_DET[0], C_CAM_DET[1], C_CAM_DET[2]);
                let (mut mins, mut sizes, mut tops, mut labels) = (
                    Vec::with_capacity(dets.len()),
                    Vec::with_capacity(dets.len()),
                    Vec::with_capacity(dets.len()),
                    Vec::with_capacity(dets.len()),
                );
                for d in &dets {
                    mins.push([d.x0, d.y0]);
                    sizes.push([d.x1 - d.x0, d.y1 - d.y0]);
                    tops.push([0.5 * (d.x0 + d.x1), d.y0]);
                    labels.push(cam_det_label(d));
                }
                // The label rides a point on the TOP edge: a box's own label
                // wraps at the box's width, and the bottom edges are where
                // the tracks and the answer put theirs.
                rec.log(
                    entity::CAMERA_DET,
                    &Boxes2D::from_mins_and_sizes(mins, sizes)
                        .with_colors([color])
                        .with_radii([Radius::new_ui_points(CAM_DET_OUTLINE_PT)])
                        .with_show_labels(false),
                )
                .and_then(|()| {
                    rec.log(
                        entity::CAMERA_DET,
                        &Points2D::new(tops)
                            .with_colors([color])
                            .with_radii([Radius::new_ui_points(0.5 * CAM_DET_OUTLINE_PT)])
                            .with_labels(labels)
                            .with_show_labels(true),
                    )
                })
            };
            if logged.is_err() {
                report.viz_log_errors += 1;
            }
            rec.send(Subject::of_sample(&rc, s));
            report.viz_bytes_total += bytes_alloc(SLOT_CAMDET) - v0;
            report.viz_ns_total += now() - t_viz;
        }

        let out_storage = cam_det_storage_id(&batch).unwrap_or(0);
        report.batches.push(batch.clone());
        let derived = Sample {
            stream: StreamId::CAM_DET,
            // The frame's own number, as the bare reference used: a gap on
            // this stream is a frame the detector never saw.
            seq: s.seq,
            arrival_seq: 0,
            parent: Some((s.stream, s.seq)),
            tov: s.tov,
            epoch: s.epoch,
            due: s.due,
            arrival: now(),
            payload: batch,
            storage_id: out_storage,
            // No decode: this sample came off a queue, not off a disk.
            decode_ns: 0,
        };
        // `cam_det->track` is a Drop* edge (`run::CHAIN_POLICY`): this cannot
        // block, so a slow fusion can never back-pressure the camera.
        let admitted = admission.admit(derived);
        let mut produced_row = Evidence::driver_admitted(&rc, &admitted, "cam_det", "camdet");
        produced_row.bytes_alloc = build_bytes;
        sink.send(EvRow::Evidence(produced_row), &ctx, "camdet");
        report.produced += 1;
        report.out_parent.push((admitted.seq, s.seq));
    }
    report
}

/// Axis-aligned bounds of a point cloud, in the sensor's own metres.
///
/// `None` for a run that saw no points at all: a cloud with no points has no
/// extent, and `[0, 0]` would read as "every point was at the origin".
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct Extent {
    pub min: [f32; 3],
    pub max: [f32; 3],
}

impl Extent {
    /// The degenerate box around one point.
    fn of(p: [f32; 3]) -> Extent {
        Extent { min: p, max: p }
    }

    /// The smallest box containing both, with `None` as the identity — the
    /// empty cloud, which has no extent rather than an extent of zero.
    ///
    /// One operation for both jobs: growing a sweep's box a point at a time,
    /// and folding each sweep's box into the run's. Two near-identical
    /// hand-rolled loops is how the per-run bounds end up disagreeing with the
    /// per-sweep ones.
    fn merge(a: Option<Extent>, b: Extent) -> Option<Extent> {
        Some(match a {
            None => b,
            Some(a) => {
                let mut out = a;
                for (o, v) in out.min.iter_mut().zip(b.min) {
                    *o = o.min(v);
                }
                for (o, v) in out.max.iter_mut().zip(b.max) {
                    *o = o.max(v);
                }
                out
            }
        })
    }
}

/// The raw sweep's colour in the viewer: the lidar's teal, #00A8B0, at alpha
/// 120, so a surface of 41k points stays see-through.
pub const SWEEP_RGBA: [u8; 4] = [0, 168, 176, 120];

/// The voxels' colour: the same family, #0096A0 at alpha 170 -- a shade
/// darker and more opaque, because each point stands for a cell.
pub const VOXEL_RGBA: [u8; 4] = [0, 150, 160, 170];

/// What the `cloud` thread returns.
pub struct CloudReport {
    pub delivered: u64,
    pub storage_mismatch: u64,
    /// Sweeps whose `point_count` column disagreed with the payload's own
    /// length. Counted rather than asserted: the two are written by the same
    /// builder, so a disagreement means the batch was rebuilt somewhere, and
    /// that is a finding rather than a crash.
    pub count_mismatch: u64,
    pub bytes_total: u64,
    /// Arrow bytes this edge CARRIED, summed over delivered sweeps
    /// ([`pipes_core::sample::payload_bytes`]).
    ///
    /// Beside `bytes_total` -- what the stage ALLOCATED -- this is the pair
    /// the next step has to move. A zero-copy hand-off reads as
    /// `payload_bytes_total` large and `bytes_total` ~0: ~1.95 MB crossed the
    /// edge and nothing was copied to make it happen. A transform stage that
    /// produces a smaller result shows up as this number falling down the
    /// chain while the allocation rises once, which is the whole claim about
    /// handing on less than you were given.
    pub payload_bytes_total: u64,
    /// `(seq, storage_id re-derived from the batch)` per delivered sweep.
    pub by_seq: Vec<(u64, usize)>,
    /// Bytes this stage allocated BUILDING THE POINT CLOUD FOR THE VIEWER,
    /// summed over the sweeps it drew. 0 without `--dashboard`, where nothing
    /// is drawn at all.
    ///
    /// Reported rather than absorbed, because it is not free and pretending it
    /// were would undercut every other number in this file. `Points3D` needs
    /// positions as `[f32; 3]` and the payload is interleaved `xyzr`, so the
    /// stride is wrong and a deinterleaving COPY is unavoidable -- there is no
    /// zero-copy path in rerun 0.38.1 (`Vec3D::to_arrow` collects its input,
    /// flattens it and collects again, so even a correctly shaped input is
    /// copied three more times inside the SDK). This is measured on the same
    /// allocator slot as the stage's own work, in a window that opens AFTER
    /// `proc_end`, so it is counted and excluded rather than counted and
    /// blamed on the pipeline.
    pub viz_bytes_total: u64,
    /// Wall-clock ns spent building and logging those clouds, summed. Entirely
    /// outside the measured window, like every picture in this file.
    pub viz_ns_total: i64,
    /// Sweeps whose cloud could not be logged. A viewer failure is not a
    /// pipeline failure, so it is counted here and never propagated.
    pub viz_log_errors: u64,
    pub points_total: u64,
    /// `None` until a sweep has been seen.
    pub points_min: Option<u32>,
    pub points_max: Option<u32>,
    pub extent: Option<Extent>,
    /// `tov_trigger_ns - tov.start()` summed over delivered sweeps: how far
    /// into its own rotation the head was facing forward.
    ///
    /// The trigger is the one instant that says what a camera frame is
    /// contemporaneous *with*, and it is the only part of a sweep's timing
    /// that travels in the **payload** rather than the envelope — `Tov::Range`
    /// has nowhere to put a third instant. Reading it back here is what makes
    /// that a fact about a running pipeline rather than about the driver's
    /// unit tests.
    pub trigger_offset_total_ns: i64,
    /// Sweeps whose payload trigger fell outside the envelope's own
    /// `Tov::Range`. Zero, or the driver paired a trigger with the wrong
    /// sweep — which no timing check downstream could catch, because each
    /// number is individually plausible.
    pub trigger_outside_range: u64,
    /// Sweeps whose payload carried no `tov_trigger_ns` column at all.
    pub trigger_missing: u64,
    /// Samples that did not carry the provenance [`CloudCfg::expect_parent`]
    /// asked for. Always 0 when no parent was expected, which is the only
    /// honest reading for a stream straight off a sensor: a driver sample has
    /// no parent, so there is nothing to be missing.
    ///
    /// **This is the STREAM half of `parent` only.** It says a sample came
    /// from the lidar; it does not say which sweep, and the seq is the half
    /// that identifies the measurement.
    pub parent_mismatch: u64,
    /// `(own seq, the seq this sample names as its parent)` per delivered
    /// sample, when a parent was expected.
    ///
    /// Recorded rather than checked here, because this thread cannot check
    /// it: knowing whether sample 7 really came from sweep 9 means knowing
    /// what the producer read, which lives in another thread's report. The
    /// run joins the two afterwards (`run::parent_check`). Until it did, the
    /// seq half of `parent` was written by `reduce` and read by nothing — a
    /// derived sample naming the WRONG sweep passed every test in the
    /// workspace, `parent_mismatch` included.
    pub parent_by_seq: Vec<(u64, Option<u64>)>,
}

/// Which edge a [`cloud_thread`] instance is reading, and what it may assume
/// about what arrives on it.
///
/// A struct rather than four positional arguments because there are now two
/// instances of this consumer and they differ only in these fields — which is
/// the point being demonstrated. If the raw and the derived cloud needed
/// different *code*, the Arrow rule would not hold.
pub struct CloudCfg {
    /// The edge, e.g. `velo->cloud`.
    pub edge: &'static str,
    /// The stage name written on every row, e.g. `cloud`.
    pub stage: &'static str,
    /// Allocator slot for this instance. Never shared with another stage: a
    /// shared counter would report the producer's output buffer as the
    /// consumer's cost, which is precisely the reading this chain exists for.
    pub slot: usize,
    /// Capacity hint for the per-sample vectors, sized outside the loop.
    pub expect: usize,
    /// The stream a sample on this edge must name as its parent, or `None` for
    /// a stream that comes off a sensor and has no parent to name.
    pub expect_parent: Option<StreamId>,
    /// Where this instance draws its cloud, or `None` to draw nothing.
    ///
    /// `None` unless `--dashboard`, exactly like `ProcCfg::rec`: on the default
    /// path this consumer has no canvas at all, so the measured stage is
    /// byte for byte what it was before the viewer existed.
    pub rec: Option<Canvas>,
    /// Entity path for this instance's cloud: `lidar/sweep` for the raw
    /// sweep, `lidar/voxels` for the reduced one. Both hang under `lidar/` so
    /// they share one origin and scrub together, on two tabs of one view.
    pub entity: &'static str,
    /// The one colour this cloud is drawn in, RGBA: the lidar's teal, a
    /// little more opaque for the voxels than for the raw sweep. One colour
    /// for the whole cloud rather than one per point: the pictures colour by
    /// PROVENANCE -- teal is the lidar, green fused, blue the camera, gold
    /// the answer -- and a per-point ramp here was the one place a colour
    /// meant something else. It is also one colour on the recording instead
    /// of 4 B a point.
    pub color: [u8; 4],
    /// Point radius, in UI points: the same size on screen at any distance,
    /// so the near road does not turn into a slab of spheres and the far
    /// points do not vanish. The voxels are drawn larger than the sweep
    /// because there are far fewer of them; drawn at the same size they read
    /// as a thinner copy of the sweep rather than as a coarser one.
    pub radius_pt: f32,
    /// Every `stride`-th point goes to the VIEWER'S COPY of the cloud; 1
    /// draws them all. This touches nothing the pipeline measures: the
    /// buffer the stage read is the full sweep, `payload_bytes` on its row
    /// is the full sweep's, and the count printed at the end of the run is
    /// every point. It is the picture that is thinned, because the viewer
    /// re-uploads a point cloud on every frame and 123k points a sweep on
    /// a tab nobody has open was 244 MB of a 555 MB recording.
    pub stride: usize,
}

impl CloudReport {
    /// Mean points per delivered sweep, or `None` when nothing was delivered.
    pub fn points_mean(&self) -> Option<u64> {
        (self.delivered > 0).then(|| self.points_total / self.delivered)
    }

    /// Mean Arrow bytes carried across the edge per delivered sweep, or `None`
    /// when nothing was delivered. `None` rather than 0, because 0 is what a
    /// stage that allocates nothing legitimately reports for `bytes_total`,
    /// and the two numbers must not be confusable.
    pub fn payload_bytes_mean(&self) -> Option<u64> {
        (self.delivered > 0).then(|| self.payload_bytes_total / self.delivered)
    }

    /// Mean trigger offset into the sweep, in ns; `None` when no delivered
    /// sweep carried one.
    pub fn trigger_offset_mean_ns(&self) -> Option<i64> {
        let n = self.delivered - self.trigger_missing;
        (n > 0).then(|| self.trigger_offset_total_ns / n as i64)
    }
}

/// Body of the `cloud` thread: a leaf consumer of a point cloud, whichever
/// stage produced it.
///
/// Deliberately the smallest honest consumer — the point count and the
/// bounding box — because its job is to prove a hand-off happened and measure
/// what it cost, not to do perception. It reads the shared point buffer **in
/// place** through [`velo_xyzr`], so its counted allocation per sample is 0.
///
/// **Two instances run, and that they can is the demonstration.** One reads
/// the driver's raw sweep on `velo->cloud`; the other reads the reduced cloud
/// `reduce` produced, on `det->cloud`. Same function, same accessors, no
/// branch on which kind of cloud arrived — because [`reduce_thread`] emits the
/// layout it consumed. Read the two instances' reported bytes against each
/// other and the byte chain is the difference.
///
/// Both point counts are read: the `point_count` metadata column and the
/// payload's own length. They are written by one builder from one buffer, so
/// a disagreement means something rebuilt the batch, and it is recorded
/// rather than assumed away.
pub fn cloud_thread(
    q: Arc<BoundedQueue<Arc<Sample>>>,
    sink: Arc<EvidenceSink>,
    ctx: Arc<RunCtx>,
    cfg: CloudCfg,
) -> CloudReport {
    set_stage_slot(cfg.slot);
    let rc = ctx.row_ctx();
    let mut report = CloudReport {
        delivered: 0,
        storage_mismatch: 0,
        count_mismatch: 0,
        bytes_total: 0,
        payload_bytes_total: 0,
        by_seq: Vec::with_capacity(cfg.expect),
        viz_bytes_total: 0,
        viz_ns_total: 0,
        viz_log_errors: 0,
        points_total: 0,
        points_min: None,
        points_max: None,
        extent: None,
        trigger_offset_total_ns: 0,
        trigger_outside_range: 0,
        trigger_missing: 0,
        parent_mismatch: 0,
        parent_by_seq: Vec::with_capacity(if cfg.expect_parent.is_some() {
            cfg.expect
        } else {
            0
        }),
    };
    while let Some(env) = q.pop() {
        let s: &Sample = &env.item;
        let dequeued = now();

        let proc_start = now();
        let b0 = bytes_alloc(cfg.slot);
        let xyzr: &[f32] = velo_xyzr(&s.payload).unwrap_or(&[]);
        let (points, _) = xyzr.as_chunks::<4>();
        let mut extent: Option<Extent> = None;
        for p in points {
            extent = Extent::merge(extent, Extent::of([p[0], p[1], p[2]]));
        }
        let n = points.len() as u32;
        std::hint::black_box(&extent);
        let bytes = bytes_alloc(cfg.slot) - b0;
        let proc_end = now();

        // Everything below is outside the measured window.
        if velo_point_count(&s.payload) != Some(n) {
            report.count_mismatch += 1;
        }
        // Provenance, read at the FAR end of the chain. `Sample::parent` was
        // declared for derived streams and, until this step, was `None` at
        // every construction site and read nowhere — a field that could have
        // been quietly wrong forever. A derived sample that reached here
        // without naming the sweep it came from would be a result no one could
        // trace back to a measurement.
        if let Some(want) = cfg.expect_parent {
            if s.parent.map(|(stream, _)| stream) != Some(want) {
                report.parent_mismatch += 1;
            }
            // The seq half, carried out to where it can be joined against
            // what the producing stage actually read.
            report
                .parent_by_seq
                .push((s.seq, s.parent.map(|(_, seq)| seq)));
        }
        // The payload's trigger against the envelope's range. Two independent
        // records of the same sweep's timing, from two different places, and
        // the only opportunity in the run to notice that they disagree.
        match (velo_trigger_ns(&s.payload), s.tov.start(), s.tov.end()) {
            (Some(t), Some(start), Some(end)) => {
                report.trigger_offset_total_ns += t - start.0;
                if t <= start.0 || t >= end.0 {
                    report.trigger_outside_range += 1;
                }
            }
            _ => report.trigger_missing += 1,
        }
        if let Some(e) = extent {
            report.extent = Extent::merge(report.extent, e);
        }
        report.points_total += u64::from(n);
        report.points_min = Some(report.points_min.map_or(n, |m| m.min(n)));
        report.points_max = Some(report.points_max.map_or(n, |m| m.max(n)));

        // The cloud itself, for the viewer. The stage read EVERY point -- the
        // count above is over all of them, and `payload_bytes` on this row
        // is the full sweep's -- and the picture carries every
        // `cfg.stride`-th one: all of them for the voxels, one in three for
        // the raw sweep, whose 123k points a sweep the viewer re-uploads on
        // every frame. The pipeline's claim is measured on the buffer, not
        // on the picture of it.
        //
        // Deliberately after `proc_end`, so the picture cannot inflate the
        // timing it exists to illustrate. It is also measured: `v0` opens a
        // SECOND window on the same allocator slot, after the stage's own
        // window has closed, so what the drawing costs is counted, reported
        // at the end of the run, and kept out of `bytes_alloc`.
        //
        // The copy is unavoidable rather than sloppy. `Points3D` wants
        // positions as `[f32; 3]`; the payload is interleaved `xyzr`, so the
        // stride is 4 where rerun needs 3 and no slice, cast or arrow view can
        // bridge that. There is no zero-copy path in rerun 0.38.1: even handed
        // a correctly shaped `&[[f32; 3]]`, `Vec3D::to_arrow` collects it,
        // flattens it and collects again. So this builds ONE deinterleaved
        // buffer (12 B/point) and hands it over with one colour and one
        // radius for the whole cloud; the SDK's own copies are the SDK's own.
        if let Some(rec) = &cfg.rec {
            let t_viz = now();
            let v0 = bytes_alloc(cfg.slot);
            rec.set_timestamp_nanos_since_epoch("sensor_time", s.tov.start().map_or(0, |t| t.0));
            rec.set_duration_secs("host", (s.arrival - ctx.t0_host) as f64 * 1e-9);
            // The sweep itself on `velo->cloud`; its parent, the sweep it
            // was reduced from, on `det->cloud`.
            let sweep = if s.stream == StreamId::LIDAR {
                Some(s.seq as i64)
            } else {
                s.parent
                    .filter(|(stream, _)| *stream == StreamId::LIDAR)
                    .map(|(_, seq)| seq as i64)
            };
            rec.set_time_sequence("seq", sweep_timeline(sweep, s.seq));
            let stride = cfg.stride.max(1);
            let drawn = points.len().div_ceil(stride);
            let mut pos: Vec<[f32; 3]> = Vec::with_capacity(drawn);
            for p in points.iter().step_by(stride) {
                pos.push([p[0], p[1], p[2]]);
            }
            // One colour and one radius for the whole cloud: rerun splats a
            // single value, so each is 1 component and not another
            // `points.len()` of them.
            let [r, g, b, a] = cfg.color;
            let cloud = Points3D::new(pos)
                .with_colors([Color::from_unmultiplied_rgba(r, g, b, a)])
                .with_radii([Radius::new_ui_points(cfg.radius_pt)]);
            if rec.log(cfg.entity, &cloud).is_err() {
                report.viz_log_errors += 1;
            }
            rec.send(Subject::of_sample(&rc, s));
            report.viz_bytes_total += bytes_alloc(cfg.slot) - v0;
            report.viz_ns_total += now() - t_viz;
        }

        let derived = velo_storage_id(&s.payload);
        if derived != Some(s.storage_id) {
            report.storage_mismatch += 1;
        }
        let row = Evidence::delivered(
            &rc,
            s,
            cfg.edge,
            cfg.stage,
            env.enqueued,
            env.depth_at_push,
            env.depth_after_pop,
            dequeued,
            proc_start,
            proc_end,
            derived.unwrap_or(0),
            bytes,
        );
        sink.send(EvRow::Evidence(row), &ctx, cfg.stage);
        report.delivered += 1;
        report.bytes_total += bytes;
        report.payload_bytes_total += s.payload_bytes() as u64;
        report.by_seq.push((s.seq, derived.unwrap_or(0)));
    }
    report
}

/// `reduce` stage configuration.
pub struct ReduceCfg {
    /// The grid edge in metres, already validated at the CLI boundary.
    pub voxel_size: VoxelSize,
    /// [`pipes_kitti::voxel::voxel_schema`], built once before the clock
    /// starts and shared by every derived batch (D16).
    pub schema: Arc<Schema>,
    /// Capacity hint for the per-sample vectors.
    pub expect: usize,
}

/// What the `reduce` thread returns: the middle link of the byte chain, from
/// both sides.
pub struct ReduceReport {
    /// Sweeps popped off `velo->reduce`.
    pub delivered: u64,
    /// Derived clouds built and admitted.
    pub produced: u64,
    /// Sweeps whose result could not be built. `produced + errors ==
    /// delivered` is the stage's own conservation law.
    pub errors: u64,
    /// Sweeps whose re-derived input address disagreed with the envelope's.
    pub storage_mismatch: u64,
    /// `(sweep seq, input storage_id re-derived here)`. Compared against the
    /// velodyne driver's own ids: the zero-copy proof, extended one stage.
    pub in_by_seq: Vec<(u64, usize)>,
    /// `(derived seq, output storage_id)` as built. Compared against what the
    /// downstream consumer re-derives.
    pub out_by_seq: Vec<(u64, usize)>,
    /// `(derived seq, the sweep seq it was built from)` per sample admitted.
    ///
    /// The producer side of the provenance join. `parent` is written here and
    /// read at the far end of the chain, so without a record of what this
    /// stage believed it was reducing there is nothing to compare the far
    /// end's answer WITH, and a wrong seq is indistinguishable from a right
    /// one.
    pub out_parent: Vec<(u64, u64)>,
    /// Bytes allocated while READING the sweep. Meant to be 0 — the whole
    /// input is borrowed — and it is the measured half of "the input was not
    /// copied".
    pub read_bytes_total: u64,
    /// Bytes allocated while BUILDING the result. Meant to be one output
    /// buffer per sweep, and it is *supposed* to be non-zero: a stage that
    /// transforms data must allocate its result.
    pub build_bytes_total: u64,
    /// Arrow bytes carried IN, over the sweeps this stage consumed.
    pub in_payload_bytes_total: u64,
    /// Arrow bytes carried OUT, over the clouds it produced. The two totals
    /// are taken over populations that differ only by `errors`, so their ratio
    /// is the shrink — measured here rather than downstream, where an eviction
    /// would silently change the denominator.
    pub out_payload_bytes_total: u64,
    /// Points in, points out.
    pub in_points_total: u64,
    pub out_points_total: u64,
    /// Points discarded as non-finite, and as outside the packed grid.
    pub non_finite_total: u64,
    pub out_of_range_total: u64,
    /// Output voxels holding exactly one point: the honesty caveat that
    /// belongs beside the ratio, because for those rows the "centroid"
    /// averages nothing and is a copy of one input point.
    pub singleton_voxels_total: u64,
    /// Points in the fullest voxel of the run.
    pub max_occupancy: u32,
    /// How many times the reusable scratch had to grow. Every growth is
    /// outside the measured window by construction; the count is here so
    /// "outside the window" is a number rather than a claim.
    pub scratch_growths: u64,
    /// Bytes the reusable scratch ended up holding.
    ///
    /// Reported because it is the honest caveat on `read_bytes_total = 0`.
    /// That 0 is not "this stage allocates nothing ever" — it is "this stage
    /// allocates nothing PER SWEEP", and what it excludes is this, once,
    /// outside the measured window. On a real drive it is about the size of
    /// one sweep, which is far too large to leave implied.
    pub scratch_bytes: u64,
}

impl ReduceReport {
    /// Arrow bytes carried in per sweep consumed; `None` when none was.
    pub fn in_payload_bytes_mean(&self) -> Option<u64> {
        (self.delivered > 0).then(|| self.in_payload_bytes_total / self.delivered)
    }

    /// Arrow bytes carried out per cloud produced; `None` when none was.
    pub fn out_payload_bytes_mean(&self) -> Option<u64> {
        (self.produced > 0).then(|| self.out_payload_bytes_total / self.produced)
    }

    /// Carried bytes in divided by carried bytes out: the headline shrink.
    ///
    /// `None` rather than a number when either side is empty. A ratio computed
    /// from nothing would be the easiest figure in this project to publish by
    /// accident.
    ///
    /// **`out_points_total > 0` is part of that, and it is the condition that
    /// bites.** An empty result still carries 164 B of Arrow scaffolding, so
    /// the bytes alone never reach zero: `--voxel-size-m 1e-9` addresses
    /// +-1 mm, puts every point of a sweep outside the packed grid, produces
    /// clouds of no points at all — and reported `11883.34x smaller`, the
    /// largest number this stage has ever printed, over a result with nothing
    /// in it. [`Self::point_shrink`] already answered `None` there; the two
    /// now agree, and [`Self::discarded_fraction`] says what happened.
    pub fn payload_shrink(&self) -> Option<f64> {
        (self.out_points_total > 0
            && self.out_payload_bytes_total > 0
            && self.delivered > 0
            && self.produced > 0)
            .then(|| {
                let per_in = self.in_payload_bytes_total as f64 / self.delivered as f64;
                let per_out = self.out_payload_bytes_total as f64 / self.produced as f64;
                per_in / per_out
            })
    }

    /// Input points per output point.
    pub fn point_shrink(&self) -> Option<f64> {
        (self.out_points_total > 0).then(|| {
            let per_in = self.in_points_total as f64 / self.delivered.max(1) as f64;
            let per_out = self.out_points_total as f64 / self.produced.max(1) as f64;
            per_in / per_out
        })
    }

    /// Share of input points the grid threw away, as a fraction.
    ///
    /// The number that has to travel **with** the shrink ratio rather than
    /// three lines below it. A reduction that merges points and one that
    /// discards them print the same headline, and only this tells them
    /// apart: at `--voxel-size-m 1e-9` every point of a KITTI sweep falls
    /// outside the packed grid and the run reported `11883.34x smaller` — a
    /// filter, reported as a reduction. 0.0 on `2011_09_26_drive_0005_sync`,
    /// the default drive, at both voxel sizes measured, so on the path anyone
    /// actually runs it annotates nothing.
    pub fn discarded_fraction(&self) -> Option<f64> {
        (self.in_points_total > 0).then(|| {
            (self.non_finite_total + self.out_of_range_total) as f64 / self.in_points_total as f64
        })
    }

    /// Share of output rows whose voxel held exactly one point, as a fraction.
    pub fn singleton_fraction(&self) -> Option<f64> {
        (self.out_points_total > 0)
            .then(|| self.singleton_voxels_total as f64 / self.out_points_total as f64)
    }
}

/// Body of the `reduce` thread: consume a sweep, produce a smaller cloud, and
/// admit it as a stream of its own.
///
/// This is the one thing the project had never done. `proc` produces a
/// `Vec<u8>` that goes nowhere; the velodyne driver fans out to leaf
/// consumers. Here a stage takes an Arrow payload off a queue, borrows it,
/// builds a *new, smaller* Arrow payload, and hands that to the next stage
/// through the same [`Admission`] the drivers use — so the derived cloud gets
/// its own `arrival_seq`, its own edge, its own evidence rows and its own
/// place in the conservation checks, exactly like a sensor stream.
///
/// # The two measured windows, and why they are two
///
/// Per sweep the stage writes **two** evidence rows and splits its allocation
/// between them, because "the input was not copied" and "nothing was
/// allocated" are different claims and the project has confused them before:
///
/// * the `velo->reduce` consumer row carries `read_bytes` — what receiving
///   and reading the sweep cost, which is **0**, because
///   [`pipes_kitti::velo::velo_xyzr`] hands back a slice into the driver's own
///   buffer and [`Voxelizer::plan`] indexes it in place;
/// * the `det` producer row carries `build_bytes` — one output buffer, which
///   is **not** 0 and must not be. A stage that genuinely transforms data
///   allocates its result; that is step 2 of the architecture, not a leak.
///
/// The scratch that makes the first number 0 is sized *before* the window
/// opens, which is the same house rule `proc_thread` follows: per-sample
/// scratch allocated inside the window would make the figure mean "the
/// stage's output plus its bookkeeping" while the column still said output.
///
/// # What the derived sample inherits, and what it does not
///
/// `tov` and `due` are the sweep's, unchanged: a derived sample's time of
/// validity is when the world was measured, never when a stage got round to
/// it, and inheriting `due` is what keeps `measurement_age_ns` meaning the age
/// of the *measurement* down the whole chain. `seq` is its own — two producers
/// numbering from 0 under one stage string collide in a map keyed on `seq`,
/// which is the defect `Evidence::driver_admitted` already documents.
/// `storage_id` is legitimately different, because the buffer legitimately is.
/// `parent` carries the sweep it came from, and the consumer at the far end
/// checks it.
pub fn reduce_thread(
    q: Arc<BoundedQueue<Arc<Sample>>>,
    admission: Arc<Admission>,
    sink: Arc<EvidenceSink>,
    ctx: Arc<RunCtx>,
    cfg: ReduceCfg,
) -> ReduceReport {
    set_stage_slot(SLOT_REDUCE);
    let rc = ctx.row_ctx();
    let mut report = ReduceReport {
        delivered: 0,
        produced: 0,
        errors: 0,
        storage_mismatch: 0,
        in_by_seq: Vec::with_capacity(cfg.expect),
        out_by_seq: Vec::with_capacity(cfg.expect),
        out_parent: Vec::with_capacity(cfg.expect),
        read_bytes_total: 0,
        build_bytes_total: 0,
        in_payload_bytes_total: 0,
        out_payload_bytes_total: 0,
        in_points_total: 0,
        out_points_total: 0,
        non_finite_total: 0,
        out_of_range_total: 0,
        singleton_voxels_total: 0,
        max_occupancy: 0,
        scratch_growths: 0,
        scratch_bytes: 0,
    };
    let mut vx = Voxelizer::new();
    let mut announced = false;
    while let Some(env) = q.pop() {
        let s: &Sample = &env.item;
        let dequeued = now();
        // Borrowed, not copied: a slice into the buffer the driver read the
        // file into, kept alive by the `Arc<Sample>` this stage is holding.
        let xyzr: &[f32] = velo_xyzr(&s.payload).unwrap_or(&[]);
        let (points, _) = xyzr.as_chunks::<4>();
        // Sized OUTSIDE the measured window, and announced once so a reader of
        // the run knows what the 0 below excludes.
        if vx.reserve(points.len()) {
            report.scratch_growths += 1;
            report.scratch_bytes = vx.scratch_bytes() as u64;
            if !announced {
                println!(
                    "reduce scratch = {} bytes for {} points, allocated outside the measured window",
                    vx.scratch_bytes(),
                    points.len()
                );
                announced = true;
            }
        }

        let proc_start = now();
        // Window 1 — reading the input. Expected to be 0.
        let b0 = bytes_alloc(SLOT_REDUCE);
        let trigger_ns = velo_trigger_ns(&s.payload).unwrap_or(0);
        let plan: VoxelPlan = vx.plan(points, cfg.voxel_size);
        let read_bytes = bytes_alloc(SLOT_REDUCE) - b0;
        // Window 2 — building the result. Expected to be one output buffer.
        let b1 = bytes_alloc(SLOT_REDUCE);
        let built = vx.build(points, &plan, cfg.voxel_size, trigger_ns, &cfg.schema);
        let build_bytes = bytes_alloc(SLOT_REDUCE) - b1;
        let proc_end = now();

        // Everything below is outside the measured windows.
        let in_storage = velo_storage_id(&s.payload);
        if in_storage != Some(s.storage_id) {
            report.storage_mismatch += 1;
        }
        let row = Evidence::delivered(
            &rc,
            s,
            "velo->reduce",
            "reduce",
            env.enqueued,
            env.depth_at_push,
            env.depth_after_pop,
            dequeued,
            proc_start,
            proc_end,
            in_storage.unwrap_or(0),
            read_bytes,
        );
        sink.send(EvRow::Evidence(row), &ctx, "reduce");
        report.delivered += 1;
        report.read_bytes_total += read_bytes;
        report.in_payload_bytes_total += s.payload_bytes() as u64;
        report.in_points_total += u64::from(plan.source_points);
        report.non_finite_total += u64::from(plan.non_finite);
        report.out_of_range_total += u64::from(plan.out_of_range);
        report.in_by_seq.push((s.seq, in_storage.unwrap_or(0)));

        let (batch, out_storage) = match built {
            Ok(x) => x,
            Err(e) => {
                // Reported and counted, never silently skipped: `produced +
                // errors == delivered` is checked at the end of the run, so a
                // sweep that vanished here cannot pass for one that was never
                // delivered.
                eprintln!("reduce: sweep {}: {e}", s.seq);
                report.errors += 1;
                continue;
            }
        };
        report.build_bytes_total += build_bytes;
        report.out_points_total += u64::from(plan.out_points);
        report.singleton_voxels_total += u64::from(plan.singleton_voxels);
        report.max_occupancy = report.max_occupancy.max(plan.max_occupancy);
        report.out_payload_bytes_total += payload_bytes(&batch) as u64;

        let derived = Sample {
            stream: StreamId::LIDAR_DET,
            // The sweep's own number, so one frame has one number on every
            // edge and stage: sweep 181 is 181 here as on `velo`, and as the
            // camera frame of the same instant. Unique on this stream, as a
            // stage's own counter was: one output at most per input, and an
            // input's seq arrives once.
            seq: s.seq,
            arrival_seq: 0,
            parent: Some((s.stream, s.seq)),
            tov: s.tov,
            epoch: s.epoch,
            due: s.due,
            arrival: now(),
            payload: batch,
            storage_id: out_storage,
            // No decode: this sample came off a queue, not off a disk. 0 is
            // the honest value, and the column is a DRIVER cost everywhere
            // else in the file.
            decode_ns: 0,
        };
        // The hand-off: a consumer thread admitting into the same `Admission`
        // the drivers use. `det->cloud` is a Drop* edge, so this cannot block
        // (see `run::DET_POLICY`) and a slow downstream stage can never
        // back-pressure the lidar, let alone the camera.
        let admitted = admission.admit(derived);
        let mut produced_row = Evidence::driver_admitted(&rc, &admitted, "det", "reduce");
        produced_row.bytes_alloc = build_bytes;
        sink.send(EvRow::Evidence(produced_row), &ctx, "reduce");
        report.produced += 1;
        report.out_by_seq.push((admitted.seq, out_storage));
        report.out_parent.push((admitted.seq, s.seq));
    }
    report
}

/// `detect` stage configuration.
pub struct DetectCfg {
    /// The grid edge the cloud upstream was built on, needed because this
    /// stage recovers that grid from the centroids. Taken from the run rather
    /// than from the batch, and checked against the batch per sample: see
    /// [`DetectReport::voxel_size_mismatch`].
    pub voxel_size: VoxelSize,
    /// [`pipes_kitti::detect::detect_schema`], built once before the clock
    /// starts and shared by every detection batch (D16).
    pub schema: Arc<Schema>,
    /// Capacity hint for the per-sample vectors.
    pub expect: usize,
}

/// What the `detect` thread returns: the third link of the byte chain, from
/// both sides.
pub struct DetectReport {
    /// Reduced clouds popped off `det->detect`.
    pub delivered: u64,
    /// Detection batches built and admitted.
    pub produced: u64,
    /// Clouds whose detections could not be built. `produced + errors ==
    /// delivered` is the stage's own conservation law.
    pub errors: u64,
    /// Clouds whose re-derived input address disagreed with the envelope's.
    pub storage_mismatch: u64,
    /// Clouds whose `voxel_size_m` column disagreed with the run's.
    ///
    /// Nothing in the pipeline can currently produce a disagreement — one CLI
    /// flag feeds both stages — which is exactly why it is counted rather than
    /// assumed. This stage's whole input contract is that the cloud sits on
    /// the grid named here; if that ever stops being true, the detections are
    /// computed on the wrong grid and every number downstream is quietly
    /// wrong, with nothing else in the run able to notice.
    pub voxel_size_mismatch: u64,
    /// `(cloud seq, input storage_id re-derived here)`, compared against what
    /// `reduce` recorded: the zero-copy proof, extended one more stage.
    pub in_by_seq: Vec<(u64, usize)>,
    /// `(detection seq, output storage_id)` as built, compared against what
    /// the consumer at the far end re-derives.
    pub out_by_seq: Vec<(u64, usize)>,
    /// `(detection seq, the cloud seq it was built from)` per sample admitted:
    /// the producer side of the provenance join.
    pub out_parent: Vec<(u64, u64)>,
    /// Bytes allocated while READING the cloud. Meant to be 0 — the whole
    /// input is borrowed — and it is the measured half of "the input was not
    /// copied".
    pub read_bytes_total: u64,
    /// Bytes allocated while BUILDING the detections. One output buffer per
    /// cloud, and *supposed* to be non-zero.
    pub build_bytes_total: u64,
    /// Arrow bytes carried IN, over the clouds consumed.
    pub in_payload_bytes_total: u64,
    /// Arrow bytes carried OUT, over the detection batches produced.
    pub out_payload_bytes_total: u64,
    /// Voxels in, detections out.
    pub in_voxels_total: u64,
    /// Detections out, over the batches produced.
    pub out_detections_total: u64,
    /// Voxels inside the stage's validity bound, and how many of those were
    /// ground.
    pub in_range_voxels_total: u64,
    /// Ground voxels removed, over the same population.
    pub ground_voxels_total: u64,
    /// Connected components found, before the minimum-size gate.
    pub clusters_total: u64,
    /// Detections smaller than any KITTI object class, and detections larger
    /// than one.
    ///
    /// The honesty pair that belongs beside the detection count, on the terms
    /// `reduce` reports `singleton_voxels` on: about 59 % of detections are
    /// fragments of larger structures and about six per sweep are two things
    /// fused, so the count is a count of connected non-ground structures and
    /// not a count of objects.
    pub fragment_detections_total: u64,
    /// Detections spanning more than any single KITTI object class.
    pub merged_detections_total: u64,
    /// Sweeps whose ground plane was fitted.
    pub ground_fitted: u64,
    /// Sweeps where the fit was refused and fell back to a fixed height.
    ///
    /// The one that matters: a fallback IS the fixed-z method this stage's
    /// docs argue against, so a run that used one has to say so rather than
    /// report a plane that was never fitted.
    pub ground_fallback: u64,
    /// Sum of the fitted planes' tilt in millidegrees, over the sweeps that
    /// were fitted. An integer so the report carries no floating accumulation;
    /// the run prints degrees.
    pub tilt_mdeg_total: i64,
    /// Largest fitted tilt seen, in millidegrees. The number the flat-world
    /// caveat is read against: this stage's single plane is defensible at
    /// 1.86 deg and is not at 5.
    pub tilt_mdeg_max: i64,
    /// Voxels discarded as non-finite.
    pub non_finite_total: u64,
    /// Voxels discarded as outside the packed grid.
    pub out_of_range_total: u64,
    /// Voxels whose recovered grid index collided with another's: the
    /// checkable half of "the grid is recoverable", 0 on real data.
    pub key_collisions_total: u64,
    /// The persistence check over consecutive sweeps, **uncompensated**, and
    /// the control it is read against. See
    /// [`pipes_kitti::detect::persistence`] and the note on
    /// [`detect_thread`] for why only the uncompensated half runs here.
    pub persistence: Persistence,
    /// The same check against the same detections turned a quarter turn: a
    /// control with identical spatial statistics.
    pub persistence_control_turned: Persistence,
    /// Pairs of CONSECUTIVE sweeps the check actually ran over.
    ///
    /// Reported because it is the number that says whether the fraction above
    /// is worth reading. `velo->reduce` is a Drop* edge, so an unpaced run can
    /// deliver a sixth of the sweeps and leave almost no adjacent pairs; the
    /// fraction over three pairs and the fraction over 153 are the same field
    /// and are not the same claim.
    pub persistence_pairs: u64,
    /// What the check cost, measured on this stage's own slot after the
    /// measured window closed, and reported rather than absorbed.
    pub check_bytes_total: u64,
    /// Nanoseconds the check cost, on the same terms.
    pub check_ns_total: i64,
    /// How many times the reusable scratch had to grow.
    pub scratch_growths: u64,
    /// Bytes the reusable scratch ended up holding.
    ///
    /// The honest caveat on `read_bytes_total = 0`: that 0 says the stage
    /// allocates nothing PER CLOUD, and this is what it excludes, once,
    /// outside the measured window.
    pub scratch_bytes: u64,
}

impl DetectReport {
    /// Arrow bytes carried in per cloud consumed; `None` when none was.
    pub fn in_payload_bytes_mean(&self) -> Option<u64> {
        (self.delivered > 0).then(|| self.in_payload_bytes_total / self.delivered)
    }

    /// Arrow bytes carried out per batch produced; `None` when none was.
    pub fn out_payload_bytes_mean(&self) -> Option<u64> {
        (self.produced > 0).then(|| self.out_payload_bytes_total / self.produced)
    }

    /// Carried bytes in divided by carried bytes out: this link's shrink.
    ///
    /// `None` rather than a number when the output is empty, and
    /// `out_detections_total > 0` is the condition that bites — for exactly
    /// the reason [`ReduceReport::payload_shrink`] documents. An empty
    /// detection batch still carries its Arrow header, so the bytes alone
    /// never reach zero, and a stage that detected nothing at all would
    /// otherwise print the largest shrink in the project over a result with
    /// nothing in it.
    pub fn payload_shrink(&self) -> Option<f64> {
        (self.out_detections_total > 0
            && self.out_payload_bytes_total > 0
            && self.delivered > 0
            && self.produced > 0)
            .then(|| {
                let per_in = self.in_payload_bytes_total as f64 / self.delivered as f64;
                let per_out = self.out_payload_bytes_total as f64 / self.produced as f64;
                per_in / per_out
            })
    }

    /// Input voxels per detection.
    pub fn voxel_shrink(&self) -> Option<f64> {
        (self.out_detections_total > 0).then(|| {
            let per_in = self.in_voxels_total as f64 / self.delivered.max(1) as f64;
            let per_out = self.out_detections_total as f64 / self.produced.max(1) as f64;
            per_in / per_out
        })
    }

    /// Detections per batch produced.
    pub fn detections_mean(&self) -> Option<u64> {
        (self.produced > 0).then(|| self.out_detections_total / self.produced)
    }

    /// Share of in-range voxels the ground plane removed.
    pub fn ground_fraction(&self) -> Option<f64> {
        (self.in_range_voxels_total > 0)
            .then(|| self.ground_voxels_total as f64 / self.in_range_voxels_total as f64)
    }

    /// Share of detections too small to be any KITTI object class — the
    /// caveat that travels with the count.
    pub fn fragment_fraction(&self) -> Option<f64> {
        (self.out_detections_total > 0)
            .then(|| self.fragment_detections_total as f64 / self.out_detections_total as f64)
    }

    /// Mean fitted ground tilt in degrees, over the sweeps that were fitted.
    pub fn tilt_deg_mean(&self) -> Option<f64> {
        (self.ground_fitted > 0)
            .then(|| self.tilt_mdeg_total as f64 / self.ground_fitted as f64 / 1000.0)
    }
}

/// Body of the `detect` thread: consume a reduced cloud, produce the objects
/// in it, and admit them as a stream of its own.
///
/// The second stage in this project to consume Arrow and produce Arrow, and
/// the first whose output is **a different shape from its input**. `reduce`
/// was deliberately built to emit the layout it consumed, which is what lets
/// one [`cloud_thread`] read both ends of that hand-off. This stage
/// deliberately does not, because a detection is not a point — so it has its
/// own consumer ([`obj_thread`]) and its own accessors, and `velo_xyzr`
/// returns `None` on its output rather than reading a bounding-box corner as a
/// position.
///
/// # The two measured windows, and why they are two
///
/// Exactly as in [`reduce_thread`], for the same reason: "the input was not
/// copied" and "nothing was allocated" are different claims. The `det->detect`
/// consumer row carries `read_bytes`, which is **0**, because the cloud is
/// indexed in place; the `obj` producer row carries `build_bytes`, which is
/// one detection buffer and must not be 0.
///
/// # The sanity check runs here, and only the weaker half of it
///
/// [`pipes_kitti::detect::persistence`] needs the PREVIOUS sweep's detections,
/// and this thread is the only place that still has them — so it runs here,
/// after `proc_end`, on this stage's own slot, with what it costs measured and
/// reported exactly like the cloud consumers' drawing.
///
/// **It is the UNCOMPENSATED number, and the run says so.** Compensating for
/// the vehicle's own motion needs the OXTS stream, which this pipeline does
/// not replay: `StreamId::OXTS` has no driver. Uncompensated is the weaker of
/// the two figures the investigation measured — 35.9 % against 71.8 % — and it
/// is still about 20x its control, which is why it is worth reporting instead
/// of skipping. The compensated figure is measured offline, against the same
/// code, by `crates/pipes/tests/detect_real_drive.rs`.
pub fn detect_thread(
    q: Arc<BoundedQueue<Arc<Sample>>>,
    admission: Arc<Admission>,
    sink: Arc<EvidenceSink>,
    ctx: Arc<RunCtx>,
    cfg: DetectCfg,
) -> DetectReport {
    set_stage_slot(SLOT_DETECT);
    let rc = ctx.row_ctx();
    let mut report = DetectReport {
        delivered: 0,
        produced: 0,
        errors: 0,
        storage_mismatch: 0,
        voxel_size_mismatch: 0,
        in_by_seq: Vec::with_capacity(cfg.expect),
        out_by_seq: Vec::with_capacity(cfg.expect),
        out_parent: Vec::with_capacity(cfg.expect),
        read_bytes_total: 0,
        build_bytes_total: 0,
        in_payload_bytes_total: 0,
        out_payload_bytes_total: 0,
        in_voxels_total: 0,
        out_detections_total: 0,
        in_range_voxels_total: 0,
        ground_voxels_total: 0,
        clusters_total: 0,
        fragment_detections_total: 0,
        merged_detections_total: 0,
        ground_fitted: 0,
        ground_fallback: 0,
        tilt_mdeg_total: 0,
        tilt_mdeg_max: 0,
        non_finite_total: 0,
        out_of_range_total: 0,
        key_collisions_total: 0,
        persistence: Persistence::default(),
        persistence_control_turned: Persistence::default(),
        persistence_pairs: 0,
        check_bytes_total: 0,
        check_ns_total: 0,
        scratch_growths: 0,
        scratch_bytes: 0,
    };
    let mut det = Detector::new();
    let mut announced = false;
    // The previous sweep's detections and its trigger, for the persistence
    // check, plus the buffer its control is built in. Both are reused across
    // sweeps and both live outside the measured windows.
    let mut prev: Vec<f32> = Vec::new();
    let mut turned: Vec<f32> = Vec::new();
    // The SWEEP the previous detections came from, as the cloud's parent
    // names it: the provenance, read rather than assumed. See the check
    // below for why only consecutive sweeps are compared.
    let mut prev_sweep_seq: Option<u64> = None;
    while let Some(env) = q.pop() {
        let s: &Sample = &env.item;
        let dequeued = now();
        // Borrowed, not copied: a slice into the buffer `reduce` built, kept
        // alive by the `Arc<Sample>` this stage is holding.
        let voxels = cloud_points(&s.payload);
        // Sized OUTSIDE the measured window, and announced once so a reader of
        // the run knows what the 0 below excludes.
        if det.reserve(voxels.len()) {
            report.scratch_growths += 1;
            report.scratch_bytes = det.scratch_bytes() as u64;
            if !announced {
                println!(
                    "detect scratch = {} bytes for {} voxels, allocated outside the measured window",
                    det.scratch_bytes(),
                    voxels.len()
                );
                announced = true;
            }
        }

        // Read BEFORE the window opens: it is one `Option` copy off the
        // envelope, not work on the payload.
        let sweep_seq = s.parent.map(|(_, seq)| seq);
        let proc_start = now();
        // Window 1 — reading the input. Expected to be 0.
        let b0 = bytes_alloc(SLOT_DETECT);
        let trigger_ns = velo_trigger_ns(&s.payload).unwrap_or(0);
        let plan: DetectPlan = det.plan(voxels, cfg.voxel_size);
        let read_bytes = bytes_alloc(SLOT_DETECT) - b0;
        // Window 2 — building the result. Expected to be one output buffer.
        let b1 = bytes_alloc(SLOT_DETECT);
        let built = det.build(
            voxels,
            &plan,
            cfg.voxel_size,
            trigger_ns,
            // The SWEEP this cloud came from, as its parent names it: the
            // tracker downstream needs to know whether two batches are
            // consecutive sweeps. The cloud's own seq is the same number;
            // the parent is the provenance, and the one checked end to end.
            // -1 when the cloud named no parent, which the `expect_parent`
            // check already counts.
            sweep_seq.map_or(-1, |q| q as i64),
            voxel_source_points(&s.payload).unwrap_or(0),
            &cfg.schema,
        );
        let build_bytes = bytes_alloc(SLOT_DETECT) - b1;
        let proc_end = now();

        // Everything below is outside the measured windows.
        let in_storage = velo_storage_id(&s.payload);
        if in_storage != Some(s.storage_id) {
            report.storage_mismatch += 1;
        }
        // The grid this stage assumed, against the grid the batch says it was
        // built on. See `DetectReport::voxel_size_mismatch`.
        if voxel_size_m(&s.payload).is_some_and(|v| v != cfg.voxel_size.metres()) {
            report.voxel_size_mismatch += 1;
        }
        let row = Evidence::delivered(
            &rc,
            s,
            "det->detect",
            "detect",
            env.enqueued,
            env.depth_at_push,
            env.depth_after_pop,
            dequeued,
            proc_start,
            proc_end,
            in_storage.unwrap_or(0),
            read_bytes,
        );
        sink.send(EvRow::Evidence(row), &ctx, "detect");
        report.delivered += 1;
        report.read_bytes_total += read_bytes;
        report.in_payload_bytes_total += s.payload_bytes() as u64;
        report.in_voxels_total += u64::from(plan.source_voxels);
        report.in_range_voxels_total += u64::from(plan.in_range_voxels);
        report.ground_voxels_total += u64::from(plan.ground_voxels);
        report.non_finite_total += u64::from(plan.non_finite);
        report.out_of_range_total += u64::from(plan.out_of_range);
        report.key_collisions_total += u64::from(plan.key_collisions);
        report.clusters_total += u64::from(plan.clusters);
        if plan.ground.fitted {
            report.ground_fitted += 1;
            let mdeg = (plan.ground.tilt_deg() * 1000.0) as i64;
            report.tilt_mdeg_total += mdeg;
            report.tilt_mdeg_max = report.tilt_mdeg_max.max(mdeg);
        } else {
            report.ground_fallback += 1;
        }
        report.in_by_seq.push((s.seq, in_storage.unwrap_or(0)));

        let (batch, out_storage) = match built {
            Ok(x) => x,
            Err(e) => {
                // Reported and counted, never silently skipped: `produced +
                // errors == delivered` is checked at the end of the run, so a
                // cloud that vanished here cannot pass for one never delivered.
                eprintln!("detect: cloud {}: {e}", s.seq);
                report.errors += 1;
                continue;
            }
        };
        report.build_bytes_total += build_bytes;
        report.out_detections_total += u64::from(plan.detections);
        report.fragment_detections_total += u64::from(plan.fragment_detections);
        report.merged_detections_total += u64::from(plan.merged_detections);
        report.out_payload_bytes_total += payload_bytes(&batch) as u64;

        // The sanity check and its control, against the previous sweep.
        // Deliberately after `proc_end` and measured on its own: it is a
        // diagnostic ABOUT the stage's output, not part of producing it, and
        // inside the window it would inflate the one number the chain is read
        // for.
        let t_check = now();
        let c0 = bytes_alloc(SLOT_DETECT);
        if let Some(lanes) = detections_f32(&batch) {
            // **Only CONSECUTIVE sweeps.** The pair has to be sweep `k` and
            // sweep `k+1`, and the test is the parent seq rather than a
            // tolerance on `dt`, because it is exact and needs no threshold.
            //
            // This is not a refinement, it is the difference between a
            // measurement and a number. `velo->reduce` is a Drop* edge, so on
            // an unpaced run most sweeps never reach this stage: the first
            // version compared whatever two clouds arrived in succession, which
            // on `--rate inf` were about 0.6 s apart, and reported 6.8 %
            // persistence -- a figure about the eviction rate wearing the name
            // of one about the detector. Skipping the non-adjacent pairs
            // reports over a smaller population and says what that population
            // was.
            let adjacent = matches!(
                (prev_sweep_seq, sweep_seq),
                (Some(p), Some(c)) if c == p + 1
            );
            // No `!prev.is_empty()` guard: a pair whose earlier sweep held
            // no detections contributes 0 to both counts and is still a pair
            // the check SAW. Counting it keeps `persistence_pairs` meaning
            // "adjacent sweep pairs", which is the number a reader needs to
            // judge the fraction, rather than "pairs that happened to have
            // something in them", which flatters it.
            if adjacent {
                // Zero velocity: this is the UNCOMPENSATED number. See the
                // function docs — compensating needs OXTS, which nothing
                // replays yet, and claiming the compensated figure here would
                // be claiming a measurement that was not made.
                let ego = EgoMotion::default();
                report.persistence_pairs += 1;
                report
                    .persistence
                    .add(persistence(&prev, lanes, ego, PERSISTENCE_GATE_M));
                // The control: the same detections, turned a quarter turn
                // about the sensor. IDENTICAL spatial statistics, so a score
                // near the real one would mean the real one was density
                // rather than structure.
                turned.clear();
                turned.extend_from_slice(lanes);
                for d in turned.as_chunks_mut::<DETECTION_LANES>().0 {
                    let (x, y) = (d[0], d[1]);
                    d[0] = -y;
                    d[1] = x;
                }
                report.persistence_control_turned.add(persistence(
                    &prev,
                    &turned,
                    ego,
                    PERSISTENCE_GATE_M,
                ));
            }
            prev.clear();
            prev.extend_from_slice(lanes);
            prev_sweep_seq = sweep_seq;
        }
        report.check_bytes_total += bytes_alloc(SLOT_DETECT) - c0;
        report.check_ns_total += now() - t_check;

        let derived = Sample {
            stream: StreamId::LIDAR_OBJ,
            // The sweep's own number, as the cloud carried it: see `reduce`.
            seq: s.seq,
            arrival_seq: 0,
            parent: Some((s.stream, s.seq)),
            tov: s.tov,
            epoch: s.epoch,
            due: s.due,
            arrival: now(),
            payload: batch,
            storage_id: out_storage,
            // No decode: this sample came off a queue, not off a disk.
            decode_ns: 0,
        };
        // The hand-off. `obj->sink` is a Drop* edge (see `run::OBJ_POLICY`),
        // so this cannot block and a slow downstream stage can never
        // back-pressure `reduce`, let alone the lidar or the camera.
        let admitted = admission.admit(derived);
        let mut produced_row = Evidence::driver_admitted(&rc, &admitted, "obj", "detect");
        produced_row.bytes_alloc = build_bytes;
        sink.send(EvRow::Evidence(produced_row), &ctx, "detect");
        report.produced += 1;
        report.out_by_seq.push((admitted.seq, out_storage));
        report.out_parent.push((admitted.seq, s.seq));
    }
    report
}

/// `obj` consumer configuration.
pub struct ObjCfg {
    /// Capacity hint for the per-sample vectors.
    pub expect: usize,
    /// `--dashboard` only: draw each sweep's detections as boxes
    /// (`lidar/detections`), after the measured window has closed. `None` on
    /// the default path, so the measured stage is byte for byte what it is
    /// without a viewer.
    pub rec: Option<Canvas>,
}

/// What the `obj` consumer returns: the far end of the chain.
pub struct ObjReport {
    /// Detection batches popped off `obj->sink`.
    pub delivered: u64,
    /// Batches whose re-derived buffer address disagreed with the envelope's.
    pub storage_mismatch: u64,
    /// Batches whose `detection_count` column disagreed with the payload.
    pub count_mismatch: u64,
    /// Bytes this stage allocated, over delivered samples. Meant to be 0.
    pub bytes_total: u64,
    /// Arrow bytes carried across `obj->sink`.
    pub payload_bytes_total: u64,
    /// Detections seen.
    pub detections_total: u64,
    /// Fewest detections in any one batch.
    pub detections_min: Option<u32>,
    /// Most detections in any one batch.
    pub detections_max: Option<u32>,
    /// Voxels behind those detections, read out of the batch's own column.
    ///
    /// This is what makes the chain checkable at the FAR end rather than only
    /// at the stage that computed it: a detection batch carries
    /// `source_voxel_count` and `source_point_count`, so this consumer can
    /// state the whole reduction — returns, to voxels, to detections — from
    /// the sample in its hand.
    pub source_voxels_total: u64,
    /// Raw returns behind those voxels, from the same place.
    pub source_points_total: u64,
    /// Samples that did not name `lidar_det` as their parent stream.
    pub parent_mismatch: u64,
    /// `(own seq, the seq this sample names as its parent)`, carried out to be
    /// joined against what `detect` recorded. The stream half above says a
    /// result came from the reduced cloud; this says which cloud.
    pub parent_by_seq: Vec<(u64, Option<u64>)>,
    /// `(seq, storage_id re-derived from the batch)` per delivered sample.
    pub by_seq: Vec<(u64, usize)>,
    /// Samples whose ground plane was a fallback rather than a fit, read back
    /// out of the payload at the far end.
    pub ground_fallback: u64,
    /// `--dashboard` only: bytes drawing the boxes cost, measured on this
    /// stage's own slot AFTER `proc_end`.
    pub viz_bytes_total: u64,
    /// Nanoseconds the same drawing cost.
    pub viz_ns_total: i64,
    /// Samples the viewer would not accept. The pipeline is unaffected.
    pub viz_log_errors: u64,
}

impl ObjReport {
    /// Mean detections per delivered batch.
    pub fn detections_mean(&self) -> Option<u64> {
        (self.delivered > 0).then(|| self.detections_total / self.delivered)
    }

    /// Mean Arrow bytes carried across the edge per delivered batch. `None`
    /// rather than 0, because 0 is what `bytes_total` legitimately reports and
    /// the two must not be confusable.
    pub fn payload_bytes_mean(&self) -> Option<u64> {
        (self.delivered > 0).then(|| self.payload_bytes_total / self.delivered)
    }

    /// Raw returns per detection over the WHOLE chain, computed from the
    /// columns the batch carries rather than from anything this process
    /// remembers.
    pub fn chain_shrink(&self) -> Option<f64> {
        (self.detections_total > 0 && self.source_points_total > 0)
            .then(|| self.source_points_total as f64 / self.detections_total as f64)
    }
}

/// Body of the `obj` thread: the leaf consumer of the detections.
///
/// Deliberately the smallest honest one, exactly as [`cloud_thread`] is for a
/// point cloud: it counts, it reads the provenance, it re-derives the buffer
/// address, and it allocates nothing. Its job is to prove the hand-off
/// happened and to measure what it cost.
///
/// It is **not** `cloud_thread`, and that is the interesting part. Two
/// instances of that one function read both ends of the `reduce` hand-off,
/// because `reduce` emits the layout it consumed. This stage's output is a
/// different shape, so it needs a consumer of its own — and the fact that it
/// does is the measurable difference between "the same data, decimated" and "a
/// different, smaller statement about the same data".
pub fn obj_thread(
    q: Arc<BoundedQueue<Arc<Sample>>>,
    sink: Arc<EvidenceSink>,
    ctx: Arc<RunCtx>,
    cfg: ObjCfg,
) -> ObjReport {
    set_stage_slot(SLOT_OBJ);
    let rc = ctx.row_ctx();
    let mut report = ObjReport {
        delivered: 0,
        storage_mismatch: 0,
        count_mismatch: 0,
        bytes_total: 0,
        payload_bytes_total: 0,
        detections_total: 0,
        detections_min: None,
        detections_max: None,
        source_voxels_total: 0,
        source_points_total: 0,
        parent_mismatch: 0,
        parent_by_seq: Vec::with_capacity(cfg.expect),
        by_seq: Vec::with_capacity(cfg.expect),
        ground_fallback: 0,
        viz_bytes_total: 0,
        viz_ns_total: 0,
        viz_log_errors: 0,
    };
    while let Some(env) = q.pop() {
        let s: &Sample = &env.item;
        let dequeued = now();

        let proc_start = now();
        let b0 = bytes_alloc(SLOT_OBJ);
        // Read in place, like every leaf consumer here: a slice into the
        // buffer `detect` built, never a copy of it.
        let lanes: &[f32] = detections_f32(&s.payload).unwrap_or(&[]);
        let mut widest = 0f32;
        for d in lanes.as_chunks::<DETECTION_LANES>().0 {
            widest = widest.max((d[6] - d[3]).max(d[7] - d[4]).max(d[8] - d[5]));
        }
        let n = (lanes.len() / DETECTION_LANES) as u32;
        std::hint::black_box(&widest);
        let bytes = bytes_alloc(SLOT_OBJ) - b0;
        let proc_end = now();

        // Everything below is outside the measured window.
        if detection_count(&s.payload) != Some(n) {
            report.count_mismatch += 1;
        }
        // Provenance, read at the far end rather than asserted upstream, in
        // both halves: which STREAM this came from, and which cloud.
        if s.parent.map(|(stream, _)| stream) != Some(StreamId::LIDAR_DET) {
            report.parent_mismatch += 1;
        }
        report
            .parent_by_seq
            .push((s.seq, s.parent.map(|(_, seq)| seq)));
        if ground_plane(&s.payload).is_some_and(|(_, fitted)| !fitted) {
            report.ground_fallback += 1;
        }
        report.source_voxels_total += u64::from(detect_source_voxels(&s.payload).unwrap_or(0));
        report.source_points_total += u64::from(detect_source_points(&s.payload).unwrap_or(0));
        report.detections_total += u64::from(n);
        report.detections_min = Some(report.detections_min.map_or(n, |m| m.min(n)));
        report.detections_max = Some(report.detections_max.map_or(n, |m| m.max(n)));

        // The boxes, for the viewer, after `proc_end` and measured on this
        // stage's own slot — the terms `cloud_thread` draws its cloud on, so
        // an illustration can never inflate the timing it illustrates.
        if let Some(rec) = &cfg.rec {
            let t_viz = now();
            let v0 = bytes_alloc(SLOT_OBJ);
            rec.set_timestamp_nanos_since_epoch("sensor_time", s.tov.start().map_or(0, |t| t.0));
            rec.set_duration_secs("host", (s.arrival - ctx.t0_host) as f64 * 1e-9);
            rec.set_time_sequence("seq", sweep_timeline(detect_sweep_seq(&s.payload), s.seq));
            let m = lanes.len() / DETECTION_LANES;
            let mut centres: Vec<[f32; 3]> = Vec::with_capacity(m);
            let mut halves: Vec<[f32; 3]> = Vec::with_capacity(m);
            for d in lanes.as_chunks::<DETECTION_LANES>().0 {
                centres.push([
                    (d[3] + d[6]) * 0.5,
                    (d[4] + d[7]) * 0.5,
                    (d[5] + d[8]) * 0.5,
                ]);
                // Half the extent, floored at half a voxel, so a three-voxel
                // detection is visible rather than a zero-size box.
                halves.push([
                    ((d[6] - d[3]) * 0.5).max(0.1),
                    ((d[7] - d[4]) * 0.5).max(0.1),
                    ((d[8] - d[5]) * 0.5).max(0.1),
                ]);
            }
            // Wireframe, so the points a box was found in stay visible
            // through it -- a filled box hides exactly what it explains --
            // and no label, because a detection has nothing to say yet.
            let boxes = Boxes3D::from_centers_and_half_sizes(centres, halves)
                .with_colors([Color::from_rgb(0x33, 0xDD, 0x88)])
                .with_fill_mode(FillMode::MajorWireframe)
                .with_show_labels(false);
            if rec.log(entity::LIDAR_DETECTIONS, &boxes).is_err() {
                report.viz_log_errors += 1;
            }
            rec.send(Subject::of_sample(&rc, s));
            report.viz_bytes_total += bytes_alloc(SLOT_OBJ) - v0;
            report.viz_ns_total += now() - t_viz;
        }

        let derived = detect_storage_id(&s.payload);
        if derived != Some(s.storage_id) {
            report.storage_mismatch += 1;
        }
        let row = Evidence::delivered(
            &rc,
            s,
            "obj->sink",
            "obj-sink",
            env.enqueued,
            env.depth_at_push,
            env.depth_after_pop,
            dequeued,
            proc_start,
            proc_end,
            derived.unwrap_or(0),
            bytes,
        );
        sink.send(EvRow::Evidence(row), &ctx, "obj-sink");
        report.delivered += 1;
        report.bytes_total += bytes;
        report.payload_bytes_total += s.payload_bytes() as u64;
        report.by_seq.push((s.seq, derived.unwrap_or(0)));
    }
    report
}

/// `track` stage configuration.
pub struct TrackCfg {
    /// [`pipes_kitti::track::track_schema`], built once.
    pub schema: Arc<Schema>,
    /// The grid the detections were found on; the tracker's gate borrows its
    /// quantisation term from it.
    pub voxel_size: VoxelSize,
    /// Capacity hint for the per-sample vectors.
    pub expect: usize,
    /// The lidar-to-camera projection, or `None` when the drive's calibration
    /// could not be read. Without it a fused sample still pairs and still
    /// carries the camera's identity; its tracks simply have no image
    /// rectangle, which the payload encodes as four zeros.
    pub calib: Option<Calib>,
    /// The `cam_det->track` edge, or `None` when no camera reaches this stage.
    pub cam_q: Option<Arc<BoundedQueue<Arc<Sample>>>>,
    /// The frames, ascending, whose camera frame is absent in the source
    /// (`Cam0Driver::absent_in_source`): a sweep of one of these numbers has
    /// no camera instant to pair with and never will, so its set expires as
    /// `pair_absent_in_source` without waiting. Empty on every drive whose
    /// camera has a frame for every frame number.
    pub cam_absent: Vec<u64>,
    /// How long to wait for a camera frame that has not arrived yet, ns.
    /// `None` means one sweep span, measured from the sweep's own `Tov::Range`
    /// rather than assumed.
    ///
    /// **This wait is a requirement, not a hack**: with the frozen detector
    /// on -- 70 to 90 ms a frame on the development host, uncertified -- the
    /// camera path delivered 154 of 154 frames at real time with zero drops,
    /// and a fusion that refuses to wait (`--pair-wait-ms 0`) expired all 154
    /// sets as `pair_late`, on runs where nothing was lost; the default wait
    /// completed all 154, at 21 to 36 ms of waiting per sweep -- the wait is
    /// the detector's time less the lidar chain's, so it moves with both. Under
    /// `--detector off` the same was
    /// measured at `--consumer-delay-ms 100 --cap 16`, and at
    /// `--consumer-delay-ms 50` the camera was mostly in
    /// time (0 or 9 of 154 expired on two runs), because the lidar chain's own
    /// latency gives it more slack than the sweep's deadline suggests.
    pub pair_wait_ns: Option<i64>,
    /// `--dashboard` only: draw this sweep's tracks, in 3D (`lidar/tracks`)
    /// and projected into the paired camera frame (`camera/tracks`, a child
    /// of the entity carrying the image, which is what puts the boxes in that
    /// image's own 2D space), after the measured window has closed. `None` on
    /// the default path, so the measured stage is byte for byte what it is
    /// without a viewer.
    pub rec: Option<Canvas>,
    /// A declared window within which an OLDER camera frame may be used, ns.
    /// 0 (the default) refuses.
    ///
    /// Counted back from the sweep's **trigger**, the instant `pair_age_ns` is
    /// measured from, so a set this window permits is never labelled staler
    /// than the window: the previous frame, 92.5 to 93.2 ms before the
    /// trigger on drive_0005, needs 94 ms or more to be admitted on every
    /// sweep. It used to count from the range's START, about half a
    /// period earlier, and `--pair-stale-ms 50` then produced sets labelled
    /// "stale 93 ms".
    ///
    /// The instructor's document is explicit that "silent stale-data reuse
    /// would undermine the whole project"; the operative word is *silent*. A
    /// stale pair is therefore possible only when a run asks for it by name,
    /// and the sample it produces carries `pair_age_ns` and a `pair_outcome`
    /// of `stale`, so the reader applies the threshold rather than the data
    /// having it baked in.
    pub pair_stale_ns: i64,
}

/// What the `track` stage returns.
pub struct TrackReport {
    /// Detection batches popped off `obj->track`.
    pub delivered: u64,
    /// Fused samples admitted onto `track->state`: the **completed** and
    /// **degraded** sets together.
    pub produced: u64,
    /// Sweeps that produced nothing because no camera frame could be paired:
    /// the **expired** sets. `produced + expired + errors == delivered`, and
    /// the run checks it.
    pub expired: u64,
    /// Batches that could not be built.
    pub errors: u64,
    /// Expired sets by cause. A frame that has not arrived yet and one that
    /// was thrown away are different failures and are counted apart.
    pub pair_dropped: u64,
    /// Waited the declared window and nothing new enough arrived.
    pub pair_late: u64,
    /// No camera stream reaches this stage at all.
    pub pair_absent: u64,
    /// The camera's source has no frame for the sweep's frame number: the
    /// frames, ascending, of the sweeps that expired so.
    pub pair_absent_in_source: Vec<u64>,
    /// Produced from a camera frame outside the sweep's range, under a
    /// declared `--pair-stale-ms`: the **degraded** sets.
    pub pair_stale: u64,
    /// Pairs that were inside the range: the **completed** sets.
    pub pair_ok: u64,
    /// Sum of `cam_tov - trigger` over completed sets, for the mean. About
    /// +10.5 ms on this dataset, and the check that the pairing found the
    /// right frame rather than merely a frame.
    pub pair_age_ns_total: i64,
    /// Smallest of the same, so a mean cannot hide a straggler.
    pub pair_age_ns_min: Option<i64>,
    /// See [`TrackReport::pair_age_ns_min`].
    pub pair_age_ns_max: Option<i64>,
    /// The same `cam_tov - trigger` over **degraded** sets: how old the
    /// camera half of every stale answer was -- negative, the frame having
    /// been taken before the sweep's trigger. Kept apart from the completed
    /// sets' figure, whose job is the mispairing check, so that a run of
    /// stale answers says how stale in its summary rather than only per set.
    pub stale_age_ns_total: i64,
    /// Smallest (most negative: stalest) of the same.
    pub stale_age_ns_min: Option<i64>,
    /// Largest (freshest) of the same.
    pub stale_age_ns_max: Option<i64>,
    /// Nanoseconds spent waiting for a camera frame, over all sweeps. Inside
    /// the measured window, so it is already in `measurement_age_ns`; reported
    /// here so it can be read as its own number too.
    pub wait_ns_total: i64,
    /// Camera references popped off `cam_det->track`.
    pub cam_delivered: u64,
    /// References popped whose payload was not a frame reference.
    pub cam_bad_format: u64,
    /// References dropped from this stage's own buffer because it was full.
    /// Meant to be 0: the buffer is sized for the whole run.
    pub cam_buffer_overflow: u64,
    /// `(seq, storage_id re-derived from the batch)` per camera sample popped
    /// off `cam_det->track`: the detector's batch was read where `camdet`
    /// built it. 0 for a bare frame reference, which names no buffer.
    pub cam_by_seq: Vec<(u64, usize)>,
    /// Bytes reading the inputs cost. Meant to be 0 — both are borrowed.
    pub read_bytes_total: u64,
    /// Bytes building the fused batches cost: one output buffer each.
    pub build_bytes_total: u64,
    /// Arrow bytes carried in on `obj->track`.
    pub in_payload_bytes_total: u64,
    /// Arrow bytes carried out on `track->state`.
    pub out_payload_bytes_total: u64,
    /// Detections seen, for the shrink at this link.
    pub in_detections_total: u64,
    /// Tracks emitted, for the same.
    pub out_tracks_total: u64,
    /// Tracks alive at any age, summed over samples — the population the
    /// emitted ones were filtered out of.
    pub live_total: u64,
    /// Ids issued over the run. Against `out_tracks_total` this is the
    /// tracker's churn: a tracker that re-identified everything every sweep
    /// would issue one id per detection per sweep.
    pub ids_issued: u64,
    /// Detections that matched an existing track.
    pub matched_total: u64,
    /// Detections that matched nothing and started a track.
    pub born_total: u64,
    /// Tracks that ran out of misses.
    pub died_total: u64,
    /// Tracks that survived a sweep without an observation.
    pub coasted_total: u64,
    /// Detections that had MORE THAN ONE track inside the gate. The tracker's
    /// own measure of how badly the gate is over-reaching.
    pub ambiguous_total: u64,
    /// The same contest counted from the tracks' side.
    pub contested_total: u64,
    /// Sweeps that were not `previous + 1`, so every track was discarded.
    /// **This is the number that attributes a collapsed tracker to an upstream
    /// eviction, or to a gap in the lidar's source (drive 0009 goes from
    /// sweep 176 to 181), rather than to the algorithm.**
    pub resets: u64,
    /// Tracks in the camera frame, over emitted tracks: what the projection
    /// actually contributed.
    pub in_frame_total: u64,
    /// Sum of the observation counts of emitted tracks: the emission rule's
    /// count, which is NOT the age.
    pub observations_total: u64,
    /// The most observations any emitted track had.
    pub observations_max: u32,
    /// Sum of the ages of emitted tracks, seconds on the sensor clock (last
    /// seen minus first seen, trigger to trigger).
    pub age_s_total: f64,
    /// The longest-lived track seen, seconds.
    pub age_s_max: f32,
    /// Emitted tracks that were NOT seen in the sweep they were emitted
    /// with: coasting on a position one sweep old, which `since_seen_s` says.
    pub unseen_total: u64,
    /// The gate in force on the last paced pair, metres.
    pub gate_m_last: f64,
    /// The interval that gate came from, seconds — both reported so the run
    /// can print its own derivation.
    pub dt_s_last: f64,
    /// `(own seq, the detections seq it names as its parent)`, joined against
    /// what the consumer downstream saw.
    pub out_parent: Vec<(u64, u64)>,
    /// `(own seq, the camera frame it paired with, how)` for completed and
    /// degraded sets: the **camera half** of the provenance, which
    /// `Sample::parent` cannot hold. Joined by `run::pair_check` against the
    /// camera's own driver rows, because a second parent written and never
    /// joined is the exact failure `Sample::parent` already had once. The
    /// pairing travels with it because the two kinds of set make different
    /// claims about the frame they name: inside the sweep, or before it and
    /// inside the declared stale window. The sweep's trigger rides with them,
    /// because that window is counted back from it and no evidence column
    /// carries it.
    pub out_cam: Vec<(u64, i64, Pairing, i64)>,
    /// `(seq, storage_id re-derived from the batch)` per sample popped.
    pub in_by_seq: Vec<(u64, usize)>,
    /// The same for what it produced.
    pub out_by_seq: Vec<(u64, usize)>,
    /// `--dashboard` only: bytes drawing the tracks cost, measured on this
    /// stage's own slot AFTER `proc_end`, exactly as `obj` measures its boxes.
    pub viz_bytes_total: u64,
    /// Nanoseconds the same drawing cost.
    pub viz_ns_total: i64,
    /// Samples the viewer would not accept. The pipeline is unaffected.
    pub viz_log_errors: u64,
    /// How often the tracker's scratch had to grow, outside the measured
    /// window.
    pub scratch_growths: u64,
    /// How big it ended up.
    pub scratch_bytes: u64,
    /// The association over the **completed** sets: each sweep's tracks
    /// against the detections of the frame inside its range.
    pub fuse_completed: FuseTally,
    /// The association over the **degraded** sets: each sweep's tracks
    /// against the detections of an OLDER frame, under `--pair-stale-ms`.
    /// Kept apart from the completed sets because it is the stale-camera
    /// experiment's result, and a mean over both would hide it.
    pub fuse_degraded: FuseTally,
    /// Sets whose `fused_count` column disagreed with the population lanes
    /// it counts. Meant to be 0.
    pub fuse_column_mismatch: u64,
    /// What shape the tracks of each population are, over the completed sets
    /// only: does the camera confirm the object-sized tracks and leave the
    /// fragments? See [`ShapeTable`].
    pub shape: ShapeTable,
    /// How often the association's scratch had to grow, outside both byte
    /// windows, and how big it ended up.
    pub fuse_scratch_growths: u64,
    pub fuse_scratch_bytes: u64,
    /// Every set produced, in order, for `runs/<name>/fused.arrows`: a clone
    /// of the admitted batch, which shares its buffers rather than copying
    /// them.
    pub batches: Vec<RecordBatch>,
}

/// What the association did over one population of fused sets.
///
/// Three counts are taken from three different places on purpose, so the run
/// can check them against each other: `detections` from the paired frame's
/// batch, `fused` off the tracks' population lanes, `camera_only` off the
/// `camera_only` column. `detections == fused + camera_only` is the
/// `INVARIANT fusion` line: every detection went exactly one place.
#[derive(Clone, Debug, Default)]
pub struct FuseTally {
    /// Sets that had a detector's output to associate with.
    pub samples: u64,
    /// Detections in their paired frames.
    pub detections: u64,
    /// Fresh in-frame tracks: those that could be fused.
    pub candidates: u64,
    /// Tracks fused, counted off lane 25.
    pub fused: u64,
    /// Detections no track matched, counted off the `camera_only` column.
    pub camera_only: u64,
    /// Detections with more than one candidate above the threshold.
    pub contested: u64,
    /// Candidates with more than one detection above it.
    pub crowded: u64,
    /// The IoU of every fused pair, for the distribution.
    pub iou: Vec<f32>,
    /// Fused tracks per class id (an index into `COCO_CLASSES`).
    pub classes: Vec<u64>,
}

impl FuseTally {
    fn add(&mut self, plan: &FusePlan, rows: &[[f32; TRACK_LANES]], camera_only: usize) {
        self.samples += 1;
        self.detections += u64::from(plan.detections);
        self.candidates += u64::from(plan.candidates);
        self.contested += u64::from(plan.contested);
        self.crowded += u64::from(plan.crowded);
        self.camera_only += camera_only as u64;
        if self.classes.is_empty() {
            self.classes = vec![0; COCO_CLASSES.len()];
        }
        for (_, iou, class, _) in rows.iter().filter_map(track_detection) {
            self.fused += 1;
            self.iou.push(iou);
            if let Some(c) = self.classes.get_mut(class as usize) {
                *c += 1;
            }
        }
    }

    /// Fused tracks over the tracks that could have been: how much of what
    /// the lidar holds in the camera's view the camera confirmed.
    pub fn fused_fraction_of_candidates(&self) -> Option<f64> {
        (self.candidates > 0).then(|| self.fused as f64 / self.candidates as f64)
    }

    /// Fused tracks over detections: how much of what the camera saw the
    /// lidar confirmed.
    pub fn fused_fraction_of_detections(&self) -> Option<f64> {
        (self.detections > 0).then(|| self.fused as f64 / self.detections as f64)
    }

    /// Fused tracks per set.
    pub fn fused_per_sample(&self) -> Option<f64> {
        (self.samples > 0).then(|| self.fused as f64 / self.samples as f64)
    }

    /// A percentile of the fused pairs' IoU, or `None` over none.
    pub fn iou_percentile(&self, p: f64) -> Option<f64> {
        if self.iou.is_empty() {
            return None;
        }
        let mut v = self.iou.clone();
        v.sort_by(f32::total_cmp);
        let i = ((p / 100.0) * (v.len() - 1) as f64).round() as usize;
        v.get(i.min(v.len() - 1)).map(|x| f64::from(*x))
    }

    /// `(class name, fused tracks)`, most frequent first, ties by class id.
    pub fn class_counts(&self) -> Vec<(&'static str, u64)> {
        let mut v: Vec<(usize, u64)> = self
            .classes
            .iter()
            .copied()
            .enumerate()
            .filter(|(_, n)| *n > 0)
            .collect();
        v.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        v.into_iter()
            .map(|(k, n)| (class_name(k as u32), n))
            .collect()
    }
}

/// One row of the shape table: how many tracks, how many of each
/// [`Shape`], how old, and how big.
#[derive(Clone, Copy, Debug, Default, Serialize)]
pub struct ShapeRow {
    /// Tracks in the row.
    pub tracks: u64,
    /// Of which fragment-shaped, object-shaped and merged, by `detect`'s own
    /// size thresholds applied to the track's box.
    pub fragment: u64,
    pub object: u64,
    pub merged: u64,
    /// Sum of their ages, seconds on the sensor clock.
    pub age_s_sum: f64,
    /// Sum of their box extents along x, y and z, metres.
    pub extent_m_sum: [f64; 3],
}

impl ShapeRow {
    fn add(&mut self, t: &[f32; TRACK_LANES]) {
        let (lo, hi) = ([t[9], t[10], t[11]], [t[12], t[13], t[14]]);
        self.tracks += 1;
        match shape_of(lo, hi) {
            Shape::Fragment => self.fragment += 1,
            Shape::Object => self.object += 1,
            Shape::Merged => self.merged += 1,
        }
        self.age_s_sum += f64::from(track_age_s(t));
        for k in 0..3 {
            self.extent_m_sum[k] += f64::from(hi[k] - lo[k]);
        }
    }

    /// Share of the row that is fragment-shaped.
    pub fn fragment_fraction(&self) -> Option<f64> {
        (self.tracks > 0).then(|| self.fragment as f64 / self.tracks as f64)
    }

    /// Mean age, seconds.
    pub fn age_s_mean(&self) -> Option<f64> {
        (self.tracks > 0).then(|| self.age_s_sum / self.tracks as f64)
    }

    /// Mean box extent along x, y and z, metres.
    pub fn extent_m_mean(&self) -> Option<[f64; 3]> {
        (self.tracks > 0).then(|| self.extent_m_sum.map(|e| e / self.tracks as f64))
    }
}

/// **Does the camera separate real objects from fragments?** Every emitted
/// track of every completed set, in exactly one row:
///
/// * `fused` -- a detection confirmed it;
/// * `inside_fused` -- lidar-only, fresh and in frame, with at least half its
///   image box inside a fused track's AND its centroid inside that track's 3D
///   box grown by one voxel edge: a piece of an object the camera confirmed.
///   The depth test is what stops an occluder in front of a car from counting;
/// * `in_frame_alone` -- lidar-only, fresh and in frame, and inside nothing
///   the camera confirmed: a wall, vegetation, a class the model does not
///   know, or a detector miss -- this table cannot say which;
/// * `coasted_in_frame` -- in frame but not seen this sweep, so never a
///   candidate;
/// * `out_of_frame` -- outside the camera's view, where the camera has
///   nothing to say.
///
/// The limit, stated: the camera is a model with its own misses, so this is
/// agreement between two sensors, not ground truth.
#[derive(Clone, Debug, Default)]
pub struct ShapeTable {
    pub fused: ShapeRow,
    pub inside_fused: ShapeRow,
    pub in_frame_alone: ShapeRow,
    pub coasted_in_frame: ShapeRow,
    pub out_of_frame: ShapeRow,
    /// The fused row again, split by the class the camera gave it: are the
    /// tracks it calls cars car-sized?
    pub fused_by_class: Vec<ShapeRow>,
}

impl ShapeTable {
    /// Files every track of one completed set. `grow_m` is the voxel edge the
    /// depth test grows a fused box by.
    fn add_sample(&mut self, rows: &[[f32; TRACK_LANES]], grow_m: f32) {
        if self.fused_by_class.is_empty() {
            self.fused_by_class = vec![ShapeRow::default(); COCO_CLASSES.len()];
        }
        for t in rows {
            let in_frame = track_image_box(t).is_some();
            if let Some((_, _, class, _)) = track_detection(t) {
                self.fused.add(t);
                if let Some(row) = self.fused_by_class.get_mut(class as usize) {
                    row.add(t);
                }
            } else if !in_frame {
                self.out_of_frame.add(t);
            } else if t[8] != 0.0 {
                self.coasted_in_frame.add(t);
            } else if inside_a_fused_object(t, rows, grow_m) {
                self.inside_fused.add(t);
            } else {
                self.in_frame_alone.add(t);
            }
        }
    }

    /// `(class name, row)` for every class the camera gave a fused track,
    /// most tracks first.
    pub fn by_class(&self) -> Vec<(&'static str, ShapeRow)> {
        let mut v: Vec<(usize, ShapeRow)> = self
            .fused_by_class
            .iter()
            .copied()
            .enumerate()
            .filter(|(_, r)| r.tracks > 0)
            .collect();
        v.sort_by(|a, b| b.1.tracks.cmp(&a.1.tracks).then(a.0.cmp(&b.0)));
        v.into_iter()
            .map(|(k, r)| (class_name(k as u32), r))
            .collect()
    }
}

/// Whether a lidar-only track is a piece of an object the camera confirmed:
/// at least half of its image box inside a fused track's image box, and its
/// centroid inside that track's 3D box grown by `grow_m` on every side.
fn inside_a_fused_object(t: &[f32; TRACK_LANES], rows: &[[f32; TRACK_LANES]], grow_m: f32) -> bool {
    let Some(tb) = track_image_box(t) else {
        return false;
    };
    let area = tb.w() * tb.h();
    rows.iter()
        .filter(|f| track_population(f) == POPULATION_FUSED)
        .any(|f| {
            let Some(fb) = track_image_box(f) else {
                return false;
            };
            let iw = tb.x1.min(fb.x1) - tb.x0.max(fb.x0);
            let ih = tb.y1.min(fb.y1) - tb.y0.max(fb.y0);
            let half_inside = iw > 0.0 && ih > 0.0 && 2.0 * iw * ih >= area;
            let in_depth = (0..3).all(|k| t[k] >= f[9 + k] - grow_m && t[k] <= f[12 + k] + grow_m);
            half_inside && in_depth
        })
}

impl TrackReport {
    /// Arrow bytes per detection batch carried in.
    pub fn in_payload_bytes_mean(&self) -> Option<u64> {
        (self.delivered > 0).then(|| self.in_payload_bytes_total / self.delivered)
    }

    /// Arrow bytes per fused sample carried out.
    pub fn out_payload_bytes_mean(&self) -> Option<u64> {
        (self.produced > 0).then(|| self.out_payload_bytes_total / self.produced)
    }

    /// How much smaller the payload got at this link, or `None` over nothing.
    ///
    /// **Expected to be barely above 1, and that is the finding rather than a
    /// disappointment.** See `pipes_kitti::track`'s module docs: this link
    /// adds information instead of removing it.
    pub fn payload_shrink(&self) -> Option<f64> {
        let (a, b) = (
            self.in_payload_bytes_mean()?,
            self.out_payload_bytes_mean()?,
        );
        (b > 0).then(|| a as f64 / b as f64)
    }

    /// Tracks per fused sample.
    pub fn tracks_mean(&self) -> Option<u64> {
        (self.produced > 0).then(|| self.out_tracks_total / self.produced)
    }

    /// Mean age of the tracks that were emitted, seconds on the sensor clock.
    /// The velocity's quantisation noise is `0.20 m / age_s`, so a low number
    /// here is a statement about how good the answer can be.
    pub fn age_s_mean(&self) -> Option<f64> {
        (self.out_tracks_total > 0).then(|| self.age_s_total / self.out_tracks_total as f64)
    }

    /// Mean observations per emitted track: what the emission rule counts.
    pub fn observations_mean(&self) -> Option<f64> {
        (self.out_tracks_total > 0)
            .then(|| self.observations_total as f64 / self.out_tracks_total as f64)
    }

    /// Share of emitted tracks that were coasting, not seen this sweep.
    pub fn unseen_fraction(&self) -> Option<f64> {
        (self.out_tracks_total > 0).then(|| self.unseen_total as f64 / self.out_tracks_total as f64)
    }

    /// Share of detections that continued an existing track.
    pub fn association_rate(&self) -> Option<f64> {
        let seen = self.matched_total + self.born_total;
        (seen > 0).then(|| self.matched_total as f64 / seen as f64)
    }

    /// Share of detections that had more than one candidate track.
    pub fn ambiguous_fraction(&self) -> Option<f64> {
        (self.in_detections_total > 0)
            .then(|| self.ambiguous_total as f64 / self.in_detections_total as f64)
    }

    /// Share of emitted tracks that landed in the camera frame.
    pub fn in_frame_fraction(&self) -> Option<f64> {
        (self.out_tracks_total > 0)
            .then(|| self.in_frame_total as f64 / self.out_tracks_total as f64)
    }

    /// Mean `cam_tov - trigger` over completed sets, ns.
    pub fn pair_age_ns_mean(&self) -> Option<i64> {
        (self.pair_ok > 0).then(|| self.pair_age_ns_total / self.pair_ok as i64)
    }

    /// Mean `cam_tov - trigger` over degraded sets, ns: negative.
    pub fn stale_age_ns_mean(&self) -> Option<i64> {
        (self.pair_stale > 0).then(|| self.stale_age_ns_total / self.pair_stale as i64)
    }
}

/// The pairing plot's numbers for one sweep, in ms from the sweep's start on
/// the sensor clock: the band the camera frame should sit in, from the
/// sweep's start (0) to its end, the trigger inside it, and the paired
/// frame's instant -- about +62 ms on drive 0005, just after the trigger, and
/// a period earlier when the fusion had to take a stale frame.
#[derive(Clone, Copy, Debug, PartialEq)]
struct PairingWindow {
    /// The sweep's length: the band spans `0..period_ms`.
    period_ms: f64,
    /// The trigger, when the payload carried one strictly inside the range.
    trigger_ms: Option<f64>,
    /// The paired camera frame's instant, when a frame was paired, clean or
    /// stale. `None` on a sweep whose set expired.
    camera_ms: Option<f64>,
}

impl PairingWindow {
    fn of(start: i64, end: i64, trigger_ns: i64, cam_tov_ns: Option<i64>) -> Self {
        let ms = |t: i64| (t - start) as f64 * 1e-6;
        PairingWindow {
            period_ms: ms(end),
            trigger_ms: (trigger_ns > start && trigger_ns < end).then(|| ms(trigger_ns)),
            camera_ms: cam_tov_ns.map(ms),
        }
    }

    /// Logs the window at the timelines already set: the band as a
    /// `Measurements` of half the period, one standard deviation of which
    /// is the other half, so it spans exactly the range; the range's two
    /// ends and the trigger as lines, because the band alone is faint on
    /// black; and the camera frame as a point. Returns how many logs the
    /// viewer refused.
    fn log(&self, rec: &Canvas) -> u64 {
        let half = 0.5 * self.period_ms;
        let at = |leaf: &str| format!("{}/{leaf}", entity::PAIRING);
        let mut refused = 0;
        let mut put = |leaf: &str, v: f64| {
            refused += u64::from(rec.log(at(leaf), &Scalars::single(v)).is_err());
        };
        put("sweep_start", 0.0);
        put("sweep_end", self.period_ms);
        if let Some(t) = self.trigger_ms {
            put("trigger", t);
        }
        if let Some(c) = self.camera_ms {
            put("camera", c);
        }
        refused += u64::from(
            rec.log(
                at(PAIRING_BAND),
                &Measurements::new([half]).with_variances([half * half]),
            )
            .is_err(),
        );
        refused
    }
}

/// Why a sweep got no answer, in the words the pictures use: what happened
/// to the camera frame it needed.
fn expiry_words(p: Pairing) -> &'static str {
    match p {
        Pairing::Dropped => "camera frame dropped",
        Pairing::Late => "camera frame late",
        Pairing::Absent => "no camera frame",
        Pairing::AbsentInSource => "no camera frame in the source",
        Pairing::Paired | Pairing::Stale => "answered",
    }
}

/// The answer's pictures of a sweep whose set expired, at the timelines
/// already set: the last answer's boxes, words and wireframe cleared off
/// both pictures,
/// the camera stamp in red saying which sweep and why (`EXPIRED · sweep 57
/// · pair_dropped`; `NO FRAME · sweep 30 · pair_absent_in_source` for a
/// frame the camera's source never had, which is a gap in the source and
/// not the fusion's failure), the headline saying there is no answer and
/// why, and the sweep's line in the answer log. Returns how many logs the
/// viewer refused.
fn log_expired(rec: &Canvas, sweep_seq: i64, pairing: Pairing) -> u64 {
    let lead = if pairing == Pairing::AbsentInSource {
        "NO FRAME"
    } else {
        "EXPIRED"
    };
    let mut refused = 0;
    let mut ok = |r: Result<(), rerun::RecordingStreamError>| refused += u64::from(r.is_err());
    for path in [
        entity::CAMERA_TRACKS,
        entity::CAMERA_ANSWER,
        entity::CAMERA_ANSWER_LABEL,
        entity::LIDAR_ANSWER,
    ] {
        ok(rec.log(path, &Clear::flat()));
    }
    ok(rec.log(
        entity::CAMERA_STATUS,
        &Points2D::new([STATUS_AT])
            .with_colors([Color::from_rgb(C_EXPIRED[0], C_EXPIRED[1], C_EXPIRED[2])])
            .with_radii([Radius::new_ui_points(0.1)])
            .with_labels([format!("{lead} · sweep {sweep_seq} · {}", pairing.name())])
            .with_show_labels(true),
    ));
    ok(rec.log(
        entity::ANSWER_HEADLINE,
        &TextDocument::new(expired_headline(sweep_seq, pairing))
            .with_media_type(MediaType::markdown()),
    ));
    ok(rec.log(
        format!("{}/line", entity::ANSWER),
        &TextLog::new(format!(
            "sweep {sweep_seq} | no answer: {} ({})",
            expiry_words(pairing),
            pairing.name()
        )),
    ));
    refused
}

/// The headline of a sweep with no answer: `## no answer for sweep 57 ·
/// camera frame dropped (pair_dropped)`.
fn expired_headline(sweep_seq: i64, pairing: Pairing) -> String {
    format!(
        "## no answer for sweep {sweep_seq}   ·   {} ({})",
        expiry_words(pairing),
        pairing.name()
    )
}

/// The headline of a frame the lidar's source has no sweep for: `## no
/// answer for frame 177 · no lidar sweep in the source (absent_in_source)`.
/// "Frame", not "sweep": there is no sweep 177.
pub(crate) fn no_sweep_headline(frame: u64) -> String {
    format!(
        "## no answer for frame {frame}   ·   no lidar sweep in the source ({ABSENT_IN_SOURCE})"
    )
}

/// The answer's pictures of a frame the lidar's source has no sweep for, on
/// the terms [`log_expired`] draws a sweep whose set expired, at the
/// timelines the caller set: the last answer's boxes and words cleared off
/// both pictures, the camera stamp in red (`NO SWEEP · frame 177 ·
/// absent_in_source`), the headline saying there is no answer and why, and
/// the frame's line in the answer log. Without it the answer of the sweep
/// before the gap stayed on screen over four newer camera frames -- an answer
/// those instants never had.
///
/// Called by the `rec` thread's dashboard, which sees the velodyne driver's
/// row for the frame -- no stage of the chain ever does -- so an error is its
/// error, like every other log it makes.
pub(crate) fn log_no_sweep(rec: &Canvas, frame: u64) -> Result<(), rerun::RecordingStreamError> {
    for path in [
        entity::CAMERA_TRACKS,
        entity::CAMERA_ANSWER,
        entity::CAMERA_ANSWER_LABEL,
        entity::LIDAR_ANSWER,
    ] {
        rec.log(path, &Clear::flat())?;
    }
    rec.log(
        entity::CAMERA_STATUS,
        &Points2D::new([STATUS_AT])
            .with_colors([Color::from_rgb(C_EXPIRED[0], C_EXPIRED[1], C_EXPIRED[2])])
            .with_radii([Radius::new_ui_points(0.1)])
            .with_labels([format!("NO SWEEP · frame {frame} · {ABSENT_IN_SOURCE}")])
            .with_show_labels(true),
    )?;
    rec.log(
        entity::ANSWER_HEADLINE,
        &TextDocument::new(no_sweep_headline(frame)).with_media_type(MediaType::markdown()),
    )?;
    rec.log(
        format!("{}/line", entity::ANSWER),
        &TextLog::new(format!(
            "frame {frame} | no answer: no lidar sweep in the source ({ABSENT_IN_SOURCE})"
        )),
    )?;
    Ok(())
}

/// The camera picture of a frame the camera's source has no PNG for, at the
/// timelines the caller set: the last frame and everything drawn on it
/// cleared, so an older picture does not stand in for an instant it does not
/// show -- the next real frame replaces the blank. With an answer, the stamp
/// says why (`NO FRAME · frame 7 · absent_in_source`); the sweep of that
/// number then expires as `pair_absent_in_source`, and `track` draws its
/// headline. Called by the `rec` thread's dashboard, on the camera driver's
/// own row for the frame.
pub(crate) fn log_no_frame(
    rec: &Canvas,
    frame: u64,
    answer: bool,
) -> Result<(), rerun::RecordingStreamError> {
    for path in [
        entity::CAMERA_IMAGE,
        entity::CAMERA_DET,
        entity::CAMERA_TRACKS,
        entity::CAMERA_ANSWER,
        entity::CAMERA_ANSWER_LABEL,
    ] {
        rec.log(path, &Clear::flat())?;
    }
    if answer {
        rec.log(
            entity::CAMERA_STATUS,
            &Points2D::new([STATUS_AT])
                .with_colors([Color::from_rgb(C_EXPIRED[0], C_EXPIRED[1], C_EXPIRED[2])])
                .with_radii([Radius::new_ui_points(0.1)])
                .with_labels([format!("NO FRAME · frame {frame} · {ABSENT_IN_SOURCE}")])
                .with_show_labels(true),
        )?;
    }
    Ok(())
}

/// The `track` stage's static styling, logged once: the pairing's series and
/// its band, and the populations by class id for everything under `lidar/`.
/// Returns how many logs the viewer refused.
fn style_track_entities(rec: &Canvas) -> u64 {
    let mut refused = 0;
    for leaf in PAIRING_LEAVES {
        if let Some(style) = series_archetype(leaf) {
            refused += u64::from(
                rec.log_static(format!("{}/{leaf}", entity::PAIRING), style.as_ref())
                    .is_err(),
            );
        }
    }
    refused += u64::from(
        rec.log_static(
            format!("{}/{PAIRING_BAND}", entity::PAIRING),
            &band_archetype("sweep window"),
        )
        .is_err(),
    );
    refused += u64::from(
        rec.log_static(entity::LIDAR_ROOT, &population_classes())
            .is_err(),
    );
    refused
}

/// Body of the `track` thread: **the fusion**.
///
/// The first stage in this project with two inputs, and the first whose output
/// can fail because of *when* an input arrived rather than because the
/// arithmetic changed. It blocks on `obj->track` — the lidar-derived edge,
/// which is the one that decides when a fused sample is due — and drains
/// `cam_det->track` without blocking, because `pop` on the wrong edge would
/// deadlock the moment that stream fell silent.
///
/// The pairing rule is containment on the sensor clock: sweep `S` with
/// `Tov::Range{start, end}` pairs with the camera frame at `Tov::Time(t)` iff
/// `start <= t < end`. Half-open, deliberately, and deliberately not the
/// strict inequality the trigger check elsewhere in this file uses: the
/// sweeps' ranges tile the timeline with a measured mean gap of 0.000 ms, so a
/// half-open interval makes "exactly one sweep contains any instant" true by
/// construction rather than by the dataset happening to cooperate. The two
/// differ only for an instant landing exactly on a boundary, which never
/// happens in these 154 frames — but an inconsistency has to be written down
/// rather than left for someone to find.
///
/// The rule has enormous margin, which matters because it means a mispairing
/// is never a near miss: a camera instant is at least 40.93 ms from the
/// nearest NEIGHBOURING sweep boundary, against an alignment jitter of 71 us.
/// Any wrong pairing is a whole sweep wrong, never a boundary case.
///
/// # The association
///
/// With the detector on, every produced set's tracks are then associated with
/// the paired frame's detections ([`pipes_kitti::fuse`]), inside the second
/// measured window, where the tracks' image boxes are built; the result
/// travels in lanes 21-25 of each track and in the batch's `camera_only`
/// column. A **degraded** set is associated with its OLDER frame's detections,
/// deliberately: that is the stale-camera failure this project exists to make
/// attributable, and the set's own `pair_outcome`, `cam_seq` and
/// `pair_age_ns` say whose detections they were. The tracker is not touched by
/// any of it -- a stale camera changes the camera lanes and nothing else,
/// which is what lets a stale run be compared with a clean one sweep by sweep.
pub fn track_thread(
    q: Arc<BoundedQueue<Arc<Sample>>>,
    admission: Arc<Admission>,
    sink: Arc<EvidenceSink>,
    ctx: Arc<RunCtx>,
    cfg: TrackCfg,
) -> TrackReport {
    set_stage_slot(SLOT_TRACK);
    let rc = ctx.row_ctx();
    let mut report = TrackReport {
        delivered: 0,
        produced: 0,
        expired: 0,
        errors: 0,
        pair_dropped: 0,
        pair_late: 0,
        pair_absent: 0,
        pair_absent_in_source: Vec::new(),
        pair_stale: 0,
        pair_ok: 0,
        pair_age_ns_total: 0,
        pair_age_ns_min: None,
        pair_age_ns_max: None,
        stale_age_ns_total: 0,
        stale_age_ns_min: None,
        stale_age_ns_max: None,
        wait_ns_total: 0,
        cam_delivered: 0,
        cam_bad_format: 0,
        cam_buffer_overflow: 0,
        cam_by_seq: Vec::with_capacity(ctx.n_frames + 1),
        read_bytes_total: 0,
        build_bytes_total: 0,
        in_payload_bytes_total: 0,
        out_payload_bytes_total: 0,
        in_detections_total: 0,
        out_tracks_total: 0,
        live_total: 0,
        ids_issued: 0,
        matched_total: 0,
        born_total: 0,
        died_total: 0,
        coasted_total: 0,
        ambiguous_total: 0,
        contested_total: 0,
        resets: 0,
        in_frame_total: 0,
        observations_total: 0,
        observations_max: 0,
        age_s_total: 0.0,
        age_s_max: 0.0,
        unseen_total: 0,
        gate_m_last: 0.0,
        dt_s_last: 0.0,
        out_parent: Vec::with_capacity(cfg.expect),
        out_cam: Vec::with_capacity(cfg.expect),
        in_by_seq: Vec::with_capacity(cfg.expect),
        out_by_seq: Vec::with_capacity(cfg.expect),
        scratch_growths: 0,
        scratch_bytes: 0,
        viz_bytes_total: 0,
        viz_ns_total: 0,
        viz_log_errors: 0,
        fuse_completed: FuseTally::default(),
        fuse_degraded: FuseTally::default(),
        fuse_column_mismatch: 0,
        shape: ShapeTable::default(),
        fuse_scratch_growths: 0,
        fuse_scratch_bytes: 0,
        batches: Vec::with_capacity(cfg.expect),
    };
    let mut tracker = ObjectTracker::new();
    // The association's scratch: reserved between the two byte windows, once
    // the paired frame's detection count is known, and counted when it grows.
    let mut fuser = Fuser::new();
    // The camera buffer. A sorted `Vec` and not a map: references arrive in
    // instant order, so pushing keeps it sorted, and `clippy.toml` bans the
    // maps whose iteration order would decide a pairing. Sized for the whole
    // run up front — 24 B per frame, 3.7 KB on this drive — so it can never
    // overflow and the drain inside the measured window never allocates.
    let mut buf = CamBuf {
        cams: Vec::with_capacity(ctx.n_frames + 1),
        pending: Vec::with_capacity(ctx.n_frames + 1),
    };
    let mut announced = false;
    // The pairing's series and the lidar's population classes are styled
    // once, on first sight, for the reason the dashboard styles on first
    // sight: an unstyled series is an anonymous line.
    let mut styled = false;
    while let Some(env) = q.pop() {
        let s: &Sample = &env.item;
        let dequeued = now();
        // Borrowed, not copied: a slice into the buffer `detect` built.
        let dets = detection_rows(&s.payload);
        // Sized OUTSIDE the measured window, and announced once.
        if tracker.reserve(dets.len()) {
            report.scratch_growths += 1;
            report.scratch_bytes = tracker.scratch_bytes() as u64;
            if !announced {
                println!(
                    "track scratch = {} bytes for {} detections against {} live tracks, allocated outside the measured window",
                    tracker.scratch_bytes(),
                    dets.len(),
                    tracker.live()
                );
                announced = true;
            }
        }

        let proc_start = now();
        // Window 1 — reading both inputs and waiting for the second. Expected
        // to allocate 0. The WAIT is deliberately inside it: it is real
        // latency the fused sample carries, and `measurement_age_ns` is
        // exactly where a reader should see it.
        let b0 = bytes_alloc(SLOT_TRACK);
        let trigger_ns = detect_trigger_ns(&s.payload).unwrap_or(0);
        let sweep_seq = detect_sweep_seq(&s.payload).unwrap_or(-1);
        let start = s.tov.start().map_or(0, |t| t.0);
        let end = s.tov.end().map_or(0, |t| t.0);
        let wait_ns = cfg.pair_wait_ns.unwrap_or(end - start).max(0);
        let t_wait = now();
        let (pairing, paired) = pair_camera(
            &cfg,
            &mut buf,
            &mut report,
            (start, end, trigger_ns),
            sweep_seq,
            wait_ns,
        );
        let cam = paired.as_ref().map(|(r, _)| *r);
        report.wait_ns_total += now() - t_wait;
        let plan = tracker.plan(dets, sweep_seq, trigger_ns, cfg.voxel_size, RANGE_LIMIT_M);
        let read_bytes = bytes_alloc(SLOT_TRACK) - b0;
        // The paired frame's detections, borrowed where `camdet` built them
        // (`None` for `proc`'s bare reference, which carries none), and the
        // association's scratch sized for them -- between the two windows, so
        // a growth is counted here and charged to neither.
        let cam_dets = paired
            .as_ref()
            .and_then(|(_, cs)| cam_det_view(&cs.payload));
        if let Some(d) = &cam_dets {
            if fuser.reserve(plan.emitted as usize, d.len()) {
                report.fuse_scratch_growths += 1;
                report.fuse_scratch_bytes = fuser.scratch_bytes() as u64;
            }
        }
        // Window 2 — building the result, when there is one to build: the
        // tracks, projected, and associated with those detections.
        let b1 = bytes_alloc(SLOT_TRACK);
        let meta = TrackMeta {
            trigger_ns,
            sweep_seq,
            source_voxels: detect_source_voxels(&s.payload).unwrap_or(0),
            source_points: detect_source_points(&s.payload).unwrap_or(0),
            cam,
            pairing,
        };
        // **The refusal.** A set that could not be paired produces no sample
        // at all; the tracker has still been advanced, because the lidar half
        // did not fail and pretending it did would break every track for a
        // camera's reason.
        let built = pairing.produces().then(|| {
            tracker.build_fused(
                &plan,
                &meta,
                cfg.calib.as_ref(),
                cam_dets.as_ref().map(|d| (d, &mut fuser)),
                &cfg.schema,
            )
        });
        let build_bytes = bytes_alloc(SLOT_TRACK) - b1;
        let proc_end = now();

        // Everything below is outside the measured windows, including the
        // camera rows this sweep's drain has been holding.
        flush_cam(&mut buf, &mut report, &sink, &ctx, &rc);
        let in_storage = detect_storage_id(&s.payload);
        let row = Evidence::delivered(
            &rc,
            s,
            "obj->track",
            "track",
            env.enqueued,
            env.depth_at_push,
            env.depth_after_pop,
            dequeued,
            proc_start,
            proc_end,
            in_storage.unwrap_or(0),
            read_bytes,
        );
        sink.send(EvRow::Evidence(row), &ctx, "track");
        report.delivered += 1;
        report.read_bytes_total += read_bytes;
        report.in_payload_bytes_total += s.payload_bytes() as u64;
        report.in_detections_total += u64::from(plan.source_detections);
        report.in_by_seq.push((s.seq, in_storage.unwrap_or(0)));
        report.live_total += u64::from(plan.live);
        report.matched_total += u64::from(plan.matched);
        report.born_total += u64::from(plan.born);
        report.died_total += u64::from(plan.died);
        report.coasted_total += u64::from(plan.coasted);
        report.ambiguous_total += u64::from(plan.ambiguous);
        report.contested_total += u64::from(plan.contested);
        report.resets += u64::from(plan.reset);
        report.ids_issued = tracker.ids_issued();
        if plan.dt_s > 0.0 {
            report.gate_m_last = plan.gate_m;
            report.dt_s_last = plan.dt_s;
        }
        match pairing {
            Pairing::Paired => {
                report.pair_ok += 1;
                if let Some(c) = cam {
                    let age = c.tov_ns - trigger_ns;
                    report.pair_age_ns_total += age;
                    report.pair_age_ns_min =
                        Some(report.pair_age_ns_min.map_or(age, |m: i64| m.min(age)));
                    report.pair_age_ns_max =
                        Some(report.pair_age_ns_max.map_or(age, |m: i64| m.max(age)));
                }
            }
            Pairing::Stale => {
                report.pair_stale += 1;
                if let Some(c) = cam {
                    let age = c.tov_ns - trigger_ns;
                    report.stale_age_ns_total += age;
                    report.stale_age_ns_min =
                        Some(report.stale_age_ns_min.map_or(age, |m: i64| m.min(age)));
                    report.stale_age_ns_max =
                        Some(report.stale_age_ns_max.map_or(age, |m: i64| m.max(age)));
                }
            }
            Pairing::Dropped => report.pair_dropped += 1,
            Pairing::Late => report.pair_late += 1,
            Pairing::Absent => report.pair_absent += 1,
            Pairing::AbsentInSource => report
                .pair_absent_in_source
                .push(u64::try_from(sweep_seq).unwrap_or(0)),
        }

        // The pairing, on the dashboard, for EVERY sweep -- one that expires
        // below still had a window, and the band has no hole where it did:
        // the sweep's range from its start to its end, its trigger, and the
        // instant of the camera frame the fusion paired with it, all in ms
        // from the sweep's start on the sensor clock. A paired frame sits
        // inside the band; a stale one below it. Numbers no evidence column
        // carries, so this stage logs them, after `proc_end` and on its own
        // slot like every picture here.
        if let Some(rec) = &cfg.rec {
            let t_viz = now();
            let v0 = bytes_alloc(SLOT_TRACK);
            rec.set_timestamp_nanos_since_epoch("sensor_time", start);
            rec.set_duration_secs("host", (s.arrival - ctx.t0_host) as f64 * 1e-9);
            rec.set_time_sequence("seq", sweep_timeline(Some(sweep_seq), s.seq));
            if !styled {
                report.viz_log_errors += style_track_entities(rec);
                styled = true;
            }
            let window = PairingWindow::of(start, end, trigger_ns, cam.map(|c| c.tov_ns));
            report.viz_log_errors += window.log(rec);
            // A set that expires produces no answer, so nothing downstream
            // draws this sweep. Without this, the last answer's boxes, words
            // and headline stayed on screen over newer frames -- an answer
            // this sweep never had. Every one of them is cleared at
            // `proc_end`, when the fusion was done with the sweep: the host
            // time the track boxes below are drawn at for a sweep that did
            // not expire, and later than anything `state` drew of the sweep
            // before, which it draws only once this stage has handed it on.
            // Cleared at the sweep's arrival instead, a previous answer
            // delayed by the pair wait would land after the clear and cover
            // it.
            if !pairing.produces() {
                rec.set_duration_secs("host", (proc_end - ctx.t0_host) as f64 * 1e-9);
                report.viz_log_errors +=
                    u64::from(rec.log(entity::LIDAR_TRACKS, &Clear::flat()).is_err());
                report.viz_log_errors += log_expired(rec, sweep_seq, pairing);
            }
            rec.send(Subject::of_sample(&rc, s));
            report.viz_bytes_total += bytes_alloc(SLOT_TRACK) - v0;
            report.viz_ns_total += now() - t_viz;
        }

        let (batch, out_storage, in_frame, fuse) = match built {
            Some(Ok(b)) => (b.batch, b.storage_id, b.in_frame, b.fuse),
            Some(Err(e)) => {
                eprintln!("track: detections {}: {e}", s.seq);
                report.errors += 1;
                continue;
            }
            None => {
                // The **expired** set. `Evidence::driver_missing` is exactly
                // the right row and needs no schema change: no sample was ever
                // admitted, so `arrival_seq` is empty and `payload_bytes` is 0,
                // and `reason` names which of the failures it was. At the
                // sweep's own number, the frame's on every other edge.
                report.expired += 1;
                sink.send(
                    EvRow::Evidence(Evidence::driver_missing(
                        &rc,
                        StreamId::TRACKS,
                        "track",
                        "track",
                        s.seq,
                        s.tov,
                        s.due,
                        now(),
                        pairing.name(),
                    )),
                    &ctx,
                    "track",
                );
                continue;
            }
        };
        report.build_bytes_total += build_bytes;
        report.out_tracks_total += u64::from(plan.emitted);
        report.in_frame_total += u64::from(in_frame);
        // The association's result, read back out of the batch it built --
        // the lanes and the column, not the plan -- and filed by how the
        // camera half was paired. Outside the measured windows.
        {
            let rows = track_rows(&batch);
            let lanes_fused = rows
                .iter()
                .filter(|t| track_population(t) == POPULATION_FUSED)
                .count() as u64;
            report.fuse_column_mismatch += u64::from(
                pipes_kitti::track::fused_count(&batch).map(u64::from) != Some(lanes_fused),
            );
            if fuse.ran {
                let camera_only = camera_only_rows(&batch).len();
                match pairing {
                    Pairing::Stale => report.fuse_degraded.add(&fuse, rows, camera_only),
                    _ => {
                        report.fuse_completed.add(&fuse, rows, camera_only);
                        report.shape.add_sample(rows, cfg.voxel_size.metres());
                    }
                }
            }
        }
        report.batches.push(batch.clone());
        report.out_payload_bytes_total += payload_bytes(&batch) as u64;
        for t in track_rows(&batch) {
            let n = track_observations(t);
            report.observations_total += u64::from(n);
            report.observations_max = report.observations_max.max(n);
            let age = track_age_s(t);
            report.age_s_total += f64::from(age);
            report.age_s_max = report.age_s_max.max(age);
            report.unseen_total += u64::from(track_since_seen_s(t) > 0.0);
        }

        // The pictures, after `proc_end` and on this stage's own slot, on the
        // terms `obj_thread` draws its boxes on: an illustration can never
        // inflate the timing it illustrates, and what it cost is printed
        // rather than hidden.
        //
        // The 3D boxes are the same geometry `lidar/detections` already
        // draws, coloured by what a detection does not have: whether the
        // camera confirmed the object. Their projection into the camera frame
        // is `state`'s to draw, from the answer's records.
        //
        // On `host` at `proc_end`, when this stage had them, like every
        // picture a stage draws of its own output -- not at the detections'
        // arrival, which is before the fusion waited for the camera. Stamped
        // there, a loaded run's 3D view showed boxes the fusion had not yet
        // made: on the overload run, sweep 144's 169 tracks at 14.925 s
        // under a headline and a camera picture still on sweep 142, whose
        // answer only came at 15.177 s.
        if let Some(rec) = &cfg.rec {
            let t_viz = now();
            let v0 = bytes_alloc(SLOT_TRACK);
            let rows = track_rows(&batch);
            rec.set_timestamp_nanos_since_epoch("sensor_time", s.tov.start().map_or(0, |t| t.0));
            rec.set_duration_secs("host", (proc_end - ctx.t0_host) as f64 * 1e-9);
            rec.set_time_sequence("seq", sweep_timeline(Some(sweep_seq), s.seq));
            let mut centres: Vec<[f32; 3]> = Vec::with_capacity(rows.len());
            let mut halves: Vec<[f32; 3]> = Vec::with_capacity(rows.len());
            let mut colors: Vec<Color> = Vec::with_capacity(rows.len());
            let mut classes: Vec<u16> = Vec::with_capacity(rows.len());
            for t in rows {
                centres.push([
                    (t[9] + t[12]) * 0.5,
                    (t[10] + t[13]) * 0.5,
                    (t[11] + t[14]) * 0.5,
                ]);
                // Floored at half a voxel, like the detection boxes, so a
                // three-voxel track is visible rather than a zero-size box.
                halves.push([
                    ((t[12] - t[9]) * 0.5).max(0.1),
                    ((t[13] - t[10]) * 0.5).max(0.1),
                    ((t[14] - t[11]) * 0.5).max(0.1),
                ]);
                // By population, not by id: green where the camera
                // confirmed the object, teal where only the lidar holds it.
                let (rgb, alpha, class) = if track_population(t) == POPULATION_FUSED {
                    (C_FUSED, LABELLED_ALPHA, CLASS_FUSED)
                } else {
                    (C_LIDAR_ONLY, 170, CLASS_LIDAR_ONLY)
                };
                colors.push(Color::from_unmultiplied_rgba(rgb[0], rgb[1], rgb[2], alpha));
                classes.push(class);
            }
            // Wireframe, so the voxels a box was built from stay visible
            // through it, and no label: two hundred boxes, most of them out
            // of the camera's view, labelled over the scene was noise. The
            // colour is PROVENANCE -- which sensors vouch for the object --
            // and the labels are on the camera picture, where the objects in
            // frame are.
            let boxes = Boxes3D::from_centers_and_half_sizes(centres, halves)
                .with_colors(colors)
                .with_class_ids(classes)
                .with_fill_mode(FillMode::MajorWireframe)
                .with_show_labels(false);
            if rec.log(entity::LIDAR_TRACKS, &boxes).is_err() {
                report.viz_log_errors += 1;
            }
            // The camera half -- these same objects projected into the paired
            // frame -- is drawn by `state` from the ANSWER's records, so the
            // picture shows what the answer says about each object (its
            // distance, its age, whether it is in frame) rather than a second
            // derivation of it here.
            rec.send(Subject::of_sample(&rc, s));
            report.viz_bytes_total += bytes_alloc(SLOT_TRACK) - v0;
            report.viz_ns_total += now() - t_viz;
        }

        let derived = Sample {
            stream: StreamId::TRACKS,
            // The sweep's own number: see `reduce`.
            seq: s.seq,
            arrival_seq: 0,
            // The LIDAR parent, because it is the one whose `tov` and `due`
            // this sample inherits. The camera parent is in the payload; see
            // `Sample::parent` and `track_schema`.
            parent: Some((s.stream, s.seq)),
            tov: s.tov,
            epoch: s.epoch,
            due: s.due,
            arrival: now(),
            payload: batch,
            storage_id: out_storage,
            decode_ns: 0,
        };
        let admitted = admission.admit(derived);
        let mut produced_row = Evidence::driver_admitted(&rc, &admitted, "track", "track");
        produced_row.bytes_alloc = build_bytes;
        // A **degraded** set is a Delivered row like any other, and without
        // this it is indistinguishable from a clean one in `evidence.csv`:
        // under `--pair-stale-ms` the staleness survived only in the payload's
        // `pair_outcome` column, so a reader with the evidence in front of
        // them could not tell which answers came from a frame that was already
        // old. `reason` is blank on a clean pair, so a non-empty one here is
        // unambiguous and costs no column.
        if pairing != Pairing::Paired {
            produced_row.reason = pairing.name();
        }
        sink.send(EvRow::Evidence(produced_row), &ctx, "track");
        report.produced += 1;
        report.out_by_seq.push((admitted.seq, out_storage));
        report.out_parent.push((admitted.seq, s.seq));
        report.out_cam.push((
            admitted.seq,
            cam.map_or(-1, |c| c.seq as i64),
            pairing,
            trigger_ns,
        ));
    }
    // The camera edge's producer stopped before this one did (the shutdown
    // order guarantees it), so anything still queued is drained here rather
    // than left unaccounted: `INVARIANT rows edge=cam_det->track` requires one
    // row per admitted sample, and a reference this stage never looked at
    // still owes one.
    drain_cam(&cfg, &mut buf, &mut report);
    flush_cam(&mut buf, &mut report, &sink, &ctx, &rc);
    report
}

/// The `track` stage's two buffers, sized once for the whole run.
///
/// `cams` is a sorted `Vec` and not a map: references arrive in instant order,
/// so pushing keeps it sorted, and `clippy.toml` bans the maps whose iteration
/// order would decide a pairing.
///
/// `pending` exists for one reason and it is a measurement reason. Draining
/// the camera edge happens INSIDE the stage's measured window -- the bounded
/// wait is real latency and belongs in `measurement_age_ns` -- but
/// `Evidence::delivered` clones the run id, so writing a row there charges the
/// allocation of the EVIDENCE to the number that is supposed to read 0 for a
/// borrowed input. It measured **10,444 B per sample** before this split,
/// which is the stage's own bookkeeping wearing the name of its input
/// handling. The envelopes are held here and their rows written after
/// `proc_end`, exactly where every other stage in this file writes its own.
struct CamBuf {
    /// Every camera sample seen, in instant order, with its reference read
    /// out once. The sample is held -- an `Arc` clone, no copy -- because the
    /// fusion needs the paired frame's DETECTIONS, not only its instant.
    cams: Vec<(CamRef, Arc<Sample>)>,
    /// Each held envelope, when it was popped, and the storage id re-derived
    /// from its payload then.
    pending: Vec<(Queued<Arc<Sample>>, HostTime, usize)>,
}

/// Pops everything waiting on `cam_det->track`. Never blocks, never allocates,
/// and writes no evidence -- see [`CamBuf`].
fn drain_cam(cfg: &TrackCfg, buf: &mut CamBuf, report: &mut TrackReport) {
    let Some(q) = &cfg.cam_q else { return };
    while let Some(env) = q.try_pop() {
        take_cam(env, buf, report);
    }
}

/// One popped camera reference: its place in the buffer, and its envelope held
/// for the row that will be written outside the measured window.
fn take_cam(env: Queued<Arc<Sample>>, buf: &mut CamBuf, report: &mut TrackReport) {
    let t = now();
    // Read, not copied: the address of the buffer the batch arrived in.
    let storage = cam_det_storage_id(&env.item.payload).unwrap_or(0);
    match read_cam_ref(&env.item.payload) {
        Some(r) => {
            if buf.cams.len() == buf.cams.capacity() {
                // Sized for the whole run, so this is unreachable rather than
                // unlikely -- counted anyway, because an unreachable case that
                // silently drops evidence is how a measurement becomes wrong.
                report.cam_buffer_overflow += 1;
                buf.cams.remove(0);
            }
            buf.cams.push((r, Arc::clone(&env.item)));
        }
        None => report.cam_bad_format += 1,
    }
    buf.pending.push((env, t, storage));
}

/// Writes the held camera rows. **Call outside the measured window.**
fn flush_cam(
    buf: &mut CamBuf,
    report: &mut TrackReport,
    sink: &Arc<EvidenceSink>,
    ctx: &Arc<RunCtx>,
    rc: &RowCtx,
) {
    for (env, t, storage) in buf.pending.drain(..) {
        // **Delivered, even when it is never paired.** It reached the stage
        // and was considered; the failure, if there is one, belongs on the
        // OUTPUT row. The alternative breaks
        // `INVARIANT rows edge=cam_det->track`.
        let row = Evidence::delivered(
            rc,
            &env.item,
            "cam_det->track",
            "track",
            env.enqueued,
            env.depth_at_push,
            env.depth_after_pop,
            t,
            t,
            t,
            storage,
            0,
        );
        sink.send(EvRow::Evidence(row), ctx, "track");
        report.cam_delivered += 1;
        report.cam_by_seq.push((env.item.seq, storage));
    }
}

/// The pairing rule, with its bounded wait and its four failures. Returns
/// the paired camera sample itself beside its reference -- an `Arc` clone,
/// so handing it over allocates nothing.
///
/// `sweep` is the sweep's range and its trigger: the range decides a
/// completed pair, and the declared stale window is counted back from the
/// trigger (see [`TrackCfg::pair_stale_ns`]). `sweep_frame` is its frame
/// number, which names why nothing is inside the range when the camera's
/// source has no frame of that number ([`Pairing::AbsentInSource`]).
fn pair_camera(
    cfg: &TrackCfg,
    buf: &mut CamBuf,
    report: &mut TrackReport,
    sweep: (i64, i64, i64),
    sweep_frame: i64,
    wait_ns: i64,
) -> (Pairing, Option<(CamRef, Arc<Sample>)>) {
    // A trigger outside the range -- only a payload without one, read as 0 --
    // is held to the range, so a missing trigger can narrow the window to the
    // range's start but never widen it to every older frame.
    let (start, end, trigger) = (sweep.0, sweep.1, sweep.2.max(sweep.0).min(sweep.1));
    let Some(q) = &cfg.cam_q else {
        return (Pairing::Absent, None);
    };
    drain_cam(cfg, buf, report);
    let cams = &buf.cams;
    // Half-open containment; see `track_thread`'s docs.
    let inside = |c: &&(CamRef, Arc<Sample>)| c.0.tov_ns >= start && c.0.tov_ns < end;
    if let Some((r, s)) = cams.iter().find(inside) {
        return (Pairing::Paired, Some((*r, Arc::clone(s))));
    }
    // A frame AFTER the window has already arrived, and frames arrive in
    // instant order, so the one that belonged inside it was produced and lost.
    // That is a different claim from "it has not arrived yet", and the bounded
    // wait below is skipped for it — waiting for a frame that is provably gone
    // would charge the fusion's latency for an upstream drop.
    let mut straddled = cams.iter().any(|c| c.0.tov_ns >= end);
    // And a frame the camera's source never had is not coming at all: the
    // wait is skipped for it too, for the same reason.
    let absent_in_source =
        u64::try_from(sweep_frame).is_ok_and(|f| cfg.cam_absent.binary_search(&f).is_ok());
    if !straddled && !absent_in_source && wait_ns > 0 {
        let deadline = now() + wait_ns;
        while let Some(env) = q.pop_by(deadline) {
            take_cam(env, buf, report);
            if let Some((r, s)) = buf.cams.iter().find(inside) {
                return (Pairing::Paired, Some((*r, Arc::clone(s))));
            }
            if buf.cams.iter().any(|c| c.0.tov_ns >= end) {
                straddled = true;
                break;
            }
        }
    }
    let cams = &buf.cams;
    // A declared stale window, or nothing: the freshest frame that is still
    // too old -- before the range, and no further before the TRIGGER than the
    // window, so its `pair_age_ns` is never staler than the window declared.
    if cfg.pair_stale_ns > 0 {
        if let Some((r, s)) = cams
            .iter()
            .filter(|c| c.0.tov_ns < start && trigger - c.0.tov_ns <= cfg.pair_stale_ns)
            .max_by_key(|c| c.0.tov_ns)
        {
            return (Pairing::Stale, Some((*r, Arc::clone(s))));
        }
    }
    if absent_in_source {
        (Pairing::AbsentInSource, None)
    } else if straddled {
        (Pairing::Dropped, None)
    } else {
        (Pairing::Late, None)
    }
}

/// `state` stage configuration.
pub struct StateCfg {
    /// [`pipes_kitti::state::state_schema`], built once.
    pub schema: Arc<Schema>,
    /// Capacity hint for the per-sample vectors.
    pub expect: usize,
    /// `--dashboard` only: draw the one object the answer flags into the
    /// paired camera frame (`camera/answer`), after the measured window has
    /// closed. A sibling of the fusion's overlay, not the same entity: one
    /// shows everything the fusion is holding and the other shows the record
    /// flagged nearest in the path, and drawing them on one path would make
    /// the flag invisible.
    pub rec: Option<Canvas>,
    /// The camera frame's width and height in pixels, when the PNG header
    /// said: the answer's label is kept inside them ([`answer_label_at`]).
    pub image_wh: Option<(f32, f32)>,
}

/// What the `state` stage returns: the end of the chain.
pub struct StateReport {
    /// Fused samples popped off `track->state`.
    pub delivered: u64,
    /// Answers admitted onto `state->sink`.
    pub produced: u64,
    /// Batches that could not be built.
    pub errors: u64,
    /// Sweeps whose answer was "something is in the vehicle's path".
    pub with_object: u64,
    /// Sweeps whose nearest object was also in the camera frame.
    pub with_image: u64,
    /// Sum of the reported distance over those sweeps, metres.
    pub distance_m_total: f64,
    /// Nearest anything ever came.
    pub distance_m_min: Option<f32>,
    /// Farthest the nearest object ever was.
    pub distance_m_max: Option<f32>,
    /// Sum of the reported closing speed over answers that had an object.
    pub closing_mps_total: f64,
    /// Sweeps where the nearest object was closing at all.
    pub closing_count: u64,
    /// Smallest time to contact seen, seconds: the most urgent thing this run
    /// had to say.
    pub ttc_s_min: Option<f32>,
    /// The sweep and the answer that produced it, kept verbatim so the run can
    /// print it in words -- with the camera half it was fused with, so those
    /// words can say when that half was stale.
    pub most_urgent: Option<(i64, StateAnswer, CamHalf)>,
    /// Bytes reading the tracks cost. Meant to be 0.
    pub read_bytes_total: u64,
    /// Bytes building the answers cost.
    pub build_bytes_total: u64,
    /// Arrow bytes carried in on `track->state`.
    pub in_payload_bytes_total: u64,
    /// Arrow bytes carried out on `state->sink`.
    pub out_payload_bytes_total: u64,
    /// Tracks the answers were built from: one record each.
    pub in_tracks_total: u64,
    /// Tracks that were in the corridor, summed.
    pub candidates_total: u64,
    /// `(own seq, the tracks seq it names as its parent)`.
    pub out_parent: Vec<(u64, u64)>,
    /// `(the fused sample's seq, the detections seq IT named)`, read off each
    /// sample as it arrives. This is the consumer half of the `track`
    /// provenance join: the producer's claim is `TrackReport::out_parent`, and
    /// a claim nothing reads back is the failure `Sample::parent` already had.
    pub in_parent_by_seq: Vec<(u64, Option<u64>)>,
    /// `(seq, storage_id re-derived from the batch)` per sample popped.
    pub in_by_seq: Vec<(u64, usize)>,
    /// The same for what it produced.
    pub out_by_seq: Vec<(u64, usize)>,
    /// `--dashboard` only: bytes drawing the answer's rectangle cost, on this
    /// stage's own slot and after `proc_end`.
    pub viz_bytes_total: u64,
    /// Nanoseconds the same drawing cost.
    pub viz_ns_total: i64,
    /// Samples the viewer would not accept.
    pub viz_log_errors: u64,
}

impl StateReport {
    /// Arrow bytes per fused sample carried in.
    pub fn in_payload_bytes_mean(&self) -> Option<u64> {
        (self.delivered > 0).then(|| self.in_payload_bytes_total / self.delivered)
    }

    /// Arrow bytes per answer carried out.
    pub fn out_payload_bytes_mean(&self) -> Option<u64> {
        (self.produced > 0).then(|| self.out_payload_bytes_total / self.produced)
    }

    /// How much smaller the payload got at the last link.
    pub fn payload_shrink(&self) -> Option<f64> {
        let (a, b) = (
            self.in_payload_bytes_mean()?,
            self.out_payload_bytes_mean()?,
        );
        (b > 0).then(|| a as f64 / b as f64)
    }

    /// Mean distance to the nearest object in the path, over the sweeps that
    /// had one.
    pub fn distance_m_mean(&self) -> Option<f64> {
        (self.with_object > 0).then(|| self.distance_m_total / self.with_object as f64)
    }

    /// Mean closing speed over the same population.
    pub fn closing_mps_mean(&self) -> Option<f64> {
        (self.with_object > 0).then(|| self.closing_mps_total / self.with_object as f64)
    }

    /// Share of sweeps that had anything in the vehicle's path.
    pub fn with_object_fraction(&self) -> Option<f64> {
        (self.delivered > 0).then(|| self.with_object as f64 / self.delivered as f64)
    }
}

/// Body of the `state` thread: **the answer**.
///
/// The last producing stage. It takes a fused sample of tracks and emits one
/// record per track: how far, how fast the gap is closing, how long that
/// leaves, how long it has been tracked and how long since it was seen, and
/// where it is in the picture -- out-of-frame records carried and flagged --
/// with the nearest fresh object in the vehicle's path flagged on its record.
/// Everything about how those are computed and chosen — the corridor's width,
/// the near face, the refusal to flag a coasted position — is in
/// `pipes_kitti::state`.
pub fn state_thread(
    q: Arc<BoundedQueue<Arc<Sample>>>,
    admission: Arc<Admission>,
    sink: Arc<EvidenceSink>,
    ctx: Arc<RunCtx>,
    cfg: StateCfg,
) -> StateReport {
    set_stage_slot(SLOT_STATE);
    let rc = ctx.row_ctx();
    let mut report = StateReport {
        delivered: 0,
        produced: 0,
        errors: 0,
        with_object: 0,
        with_image: 0,
        distance_m_total: 0.0,
        distance_m_min: None,
        distance_m_max: None,
        closing_mps_total: 0.0,
        closing_count: 0,
        ttc_s_min: None,
        most_urgent: None,
        read_bytes_total: 0,
        build_bytes_total: 0,
        in_payload_bytes_total: 0,
        out_payload_bytes_total: 0,
        in_tracks_total: 0,
        candidates_total: 0,
        out_parent: Vec::with_capacity(cfg.expect),
        in_parent_by_seq: Vec::with_capacity(cfg.expect),
        in_by_seq: Vec::with_capacity(cfg.expect),
        out_by_seq: Vec::with_capacity(cfg.expect),
        viz_bytes_total: 0,
        viz_ns_total: 0,
        viz_log_errors: 0,
    };
    // The population classes are logged once, on the first picture.
    let mut annotated = false;
    while let Some(env) = q.pop() {
        let s: &Sample = &env.item;
        let dequeued = now();

        let proc_start = now();
        // Window 1 — reading the tracks in place. Expected to be 0.
        let b0 = bytes_alloc(SLOT_STATE);
        let rows = track_rows(&s.payload);
        let answer = StateAnswer::of(rows);
        let read_bytes = bytes_alloc(SLOT_STATE) - b0;
        // Window 2 — one record per track, the nearest in the path flagged.
        let b1 = bytes_alloc(SLOT_STATE);
        let meta = StateMeta {
            trigger_ns: track_trigger_ns(&s.payload).unwrap_or(0),
            sweep_seq: track_sweep_seq(&s.payload).unwrap_or(-1),
            source_detections: track_source_detections(&s.payload).unwrap_or(0),
            source_voxels: track_source_voxels(&s.payload).unwrap_or(0),
            source_points: track_source_points(&s.payload).unwrap_or(0),
            cam_seq: track_cam_seq(&s.payload).unwrap_or(-1),
            // Carried through unchanged, so a bad answer is attributable to a
            // missing or stale camera frame at the point where the answer is
            // read.
            pair_outcome: pairing_name(track_pair_outcome(&s.payload)),
            pair_age_ns: track_pair_age_ns(&s.payload).unwrap_or(0),
        };
        // The camera's unmatched detections, handed on as the track batch
        // holds them: an `Arc` clone of the column, not a copy of it.
        let camera_only = camera_only_column(&s.payload).cloned();
        let built = build_state_batch(rows, &answer, &meta, camera_only, &cfg.schema);
        let build_bytes = bytes_alloc(SLOT_STATE) - b1;
        let proc_end = now();

        // Everything below is outside the measured windows.
        let in_storage = track_storage_id(&s.payload);
        let row = Evidence::delivered(
            &rc,
            s,
            "track->state",
            "state",
            env.enqueued,
            env.depth_at_push,
            env.depth_after_pop,
            dequeued,
            proc_start,
            proc_end,
            in_storage.unwrap_or(0),
            read_bytes,
        );
        sink.send(EvRow::Evidence(row), &ctx, "state");
        report.delivered += 1;
        report.read_bytes_total += read_bytes;
        report.in_payload_bytes_total += s.payload_bytes() as u64;
        report.in_tracks_total += rows.len() as u64;
        report.candidates_total += u64::from(answer.candidates);
        report.in_by_seq.push((s.seq, in_storage.unwrap_or(0)));
        report
            .in_parent_by_seq
            .push((s.seq, s.parent.map(|(_, seq)| seq)));
        if answer.has_object {
            report.with_object += 1;
            report.distance_m_total += f64::from(answer.distance_m);
            let d = answer.distance_m;
            report.distance_m_min = Some(report.distance_m_min.map_or(d, |m: f32| m.min(d)));
            report.distance_m_max = Some(report.distance_m_max.map_or(d, |m: f32| m.max(d)));
            report.closing_mps_total += f64::from(answer.closing_mps);
            report.with_image += u64::from(answer.image.is_some());
            if answer.ttc_s > 0.0 {
                report.closing_count += 1;
                if report.ttc_s_min.is_none_or(|m| answer.ttc_s < m) {
                    report.ttc_s_min = Some(answer.ttc_s);
                    report.most_urgent = Some((
                        meta.sweep_seq,
                        answer,
                        CamHalf {
                            cam_seq: meta.cam_seq,
                            pair_outcome: meta.pair_outcome,
                            pair_age_ns: meta.pair_age_ns,
                        },
                    ));
                }
            }
        }

        let (batch, out_storage) = match built {
            Ok(x) => x,
            Err(e) => {
                eprintln!("state: tracks {}: {e}", s.seq);
                report.errors += 1;
                continue;
            }
        };
        report.build_bytes_total += build_bytes;
        report.out_payload_bytes_total += payload_bytes(&batch) as u64;

        // The answer, on the picture: every record in the camera frame, and
        // the one flagged nearest in the path. Drawn FROM THE BATCH just
        // built, so the picture shows what the answer says about each object
        // rather than a second derivation of it. Drawn HERE rather than at the
        // far end for one concrete reason: the camera frame's own instant
        // lives in the TRACK payload (`cam_tov_ns`) and not in the answer's,
        // so this is the last stage that can stamp a rectangle on the frame it
        // belongs to instead of on the sweep that produced it, and a picture
        // is not a reason to change the answer's schema. After `proc_end`, on
        // this stage's own slot, like every other picture in this file.
        //
        // The flagged box is drawn on EVERY sample, as a box or as a `Clear`:
        // an entity that is logged only when there is an answer keeps showing
        // its last box over every frame that has none, and a gold rectangle
        // round nothing is the one thing this picture must never say. Gold on
        // a clean pair, amber on a frame the fusion declared stale, so a
        // degraded answer is visible where it is read.
        if let Some(rec) = &cfg.rec {
            let t_viz = now();
            let v0 = bytes_alloc(SLOT_STATE);
            let paired = meta.pair_outcome == Pairing::Paired.name();
            let rgb = if paired { C_ANSWER } else { C_ATTENTION };
            let color = Color::from_rgb(rgb[0], rgb[1], rgb[2]);
            // On a stale pair every label below says how stale -- the 3D
            // answer's as well as the picture's -- so amber is never the only
            // sign of it.
            let stale = !paired;
            let suffix = if stale {
                format!(" · stale {:.0} ms", -(meta.pair_age_ns as f64) * 1e-6)
            } else {
                String::new()
            };
            // In 3D first, on the sweep's timelines: the answered track's own
            // box, found among the tracks by the id the answer names.
            rec.set_timestamp_nanos_since_epoch("sensor_time", s.tov.start().map_or(0, |t| t.0));
            rec.set_duration_secs("host", (s.arrival - ctx.t0_host) as f64 * 1e-9);
            rec.set_time_sequence("seq", sweep_timeline(Some(meta.sweep_seq), s.seq));
            let named = answer
                .has_object
                .then(|| rows.iter().find(|t| t[6] as u32 == answer.object_id))
                .flatten();
            let logged = match named {
                Some(t) => rec.log(
                    entity::LIDAR_ANSWER,
                    &Boxes3D::from_centers_and_half_sizes(
                        [[
                            (t[9] + t[12]) * 0.5,
                            (t[10] + t[13]) * 0.5,
                            (t[11] + t[14]) * 0.5,
                        ]],
                        [[
                            ((t[12] - t[9]) * 0.5).max(0.1),
                            ((t[13] - t[10]) * 0.5).max(0.1),
                            ((t[14] - t[11]) * 0.5).max(0.1),
                        ]],
                    )
                    .with_colors([color])
                    .with_radii([Radius::new_ui_points(ANSWER_3D_OUTLINE_PT)])
                    .with_fill_mode(FillMode::MajorWireframe)
                    .with_labels([format!("{}{suffix}", answer_label_3d(&answer))])
                    .with_show_labels(true),
                ),
                None => rec.log(entity::LIDAR_ANSWER, &Clear::flat()),
            };
            if logged.is_err() {
                report.viz_log_errors += 1;
            }
            // Then on the picture, on the paired frame's own timelines, so a
            // box sits on the frame the fusion actually used and a pairing
            // that went to the wrong frame is something you can see instead
            // of only count. `host` is the SWEEP's arrival, set above and not
            // moved: a stale pair puts an older frame's instant on
            // `sensor_time` and `seq`, and the one timeline that has to stay
            // in arrival order is this one.
            if let (Some(cs), Some(ct)) = (track_cam_seq(&s.payload), track_cam_tov_ns(&s.payload))
            {
                rec.set_timestamp_nanos_since_epoch("sensor_time", ct);
                rec.set_time_sequence("seq", cs);
                if !annotated {
                    // The populations by class id, for everything under
                    // `camera/`.
                    if rec
                        .log_static(entity::CAMERA_ROOT, &population_classes())
                        .is_err()
                    {
                        report.viz_log_errors += 1;
                    }
                    annotated = true;
                }
                // Every record in frame and every camera-only detection, by
                // POPULATION: green where the camera confirmed the track,
                // teal where only the lidar holds it (fainter when it was not
                // seen this sweep: its box is last sweep's geometry), blue
                // where only the camera saw something. A fused box is
                // labelled with what both sensors say about it, a
                // camera-only one with its class and "no range"; a
                // lidar-only one carries its label for hover only, because
                // forty of them drawn covered the photograph. Out-of-frame
                // records have no place on a picture; they are in the
                // answer, and `answer/line` counts them.
                //
                // On a frame the fusion paired STALE, the population colours
                // stay -- greying every box, as this picture once did, erased
                // exactly the split a stale camera moves -- and each fused box
                // gains an amber stroke outside its green, and its label says
                // how stale (`suffix`, above): the class on it came from an
                // older frame than the picture under it.
                let records = object_rows(&batch);
                let camera_only = camera_only_rows(&batch);
                let n = 2 * records.len() + camera_only.len();
                let (mut mins, mut sizes, mut colors, mut radii, mut classes, mut hover) = (
                    Vec::with_capacity(n),
                    Vec::with_capacity(n),
                    Vec::with_capacity(n),
                    Vec::with_capacity(n),
                    Vec::with_capacity(n),
                    Vec::with_capacity(n),
                );
                let (mut feet, mut label_colors, mut labels) = (
                    Vec::with_capacity(n),
                    Vec::with_capacity(n),
                    Vec::with_capacity(n),
                );
                let rgba = |c: [u8; 3], a: u8| Color::from_unmultiplied_rgba(c[0], c[1], c[2], a);
                // The answer's label is the one a reader must be able to read,
                // so a drawn label that would print over it gives way: its box
                // is still drawn and still names it on hover. In a dense frame
                // the fused labels along the boxes' bottom edges buried it.
                let answer_spot = match (answer.has_object, answer.image) {
                    (true, Some(b)) => Some((
                        answer_label_at(&b, cfg.image_wh),
                        answer_label(&answer).chars().count() + suffix.chars().count(),
                    )),
                    _ => None,
                };
                let gives_way = |at: [f32; 2], label: &str| {
                    answer_spot
                        .is_some_and(|(spot, n)| labels_collide(at, label.chars().count(), spot, n))
                };
                for r in records {
                    let o = Object::from_lanes(r);
                    let Some(b) = o.image else { continue };
                    mins.push([b.x0, b.y0]);
                    sizes.push([b.w(), b.h()]);
                    if o.population == POPULATION_FUSED {
                        let label = fused_label(&o);
                        colors.push(rgba(C_FUSED, LABELLED_ALPHA));
                        radii.push(FUSED_OUTLINE_PT);
                        classes.push(CLASS_FUSED);
                        hover.push(label.clone());
                        // The answer draws its own label on its own box; a
                        // second one on the same spot would print over it.
                        let foot = [b.x0 + 0.5 * b.w(), b.y1];
                        let drawn = format!("{label}{suffix}");
                        if !o.nearest_in_path && !gives_way(foot, &drawn) {
                            feet.push(foot);
                            label_colors.push(rgba(C_FUSED, LABELLED_ALPHA));
                            labels.push(drawn);
                        }
                        if stale {
                            let k = STALE_STROKE_PX;
                            mins.push([b.x0 - k, b.y0 - k]);
                            sizes.push([b.w() + 2.0 * k, b.h() + 2.0 * k]);
                            colors.push(rgba(C_ATTENTION, LABELLED_ALPHA));
                            radii.push(TRACK_OUTLINE_PT);
                            classes.push(CLASS_FUSED);
                            hover.push(format!("{label}{suffix}"));
                        }
                    } else {
                        let alpha = if o.since_seen_s > 0.0 {
                            COASTED_OUTLINE_ALPHA
                        } else {
                            LIDAR_ONLY_ALPHA
                        };
                        colors.push(rgba(C_LIDAR_ONLY, alpha));
                        radii.push(TRACK_OUTLINE_PT);
                        classes.push(CLASS_LIDAR_ONLY);
                        hover.push(object_label(&o));
                    }
                }
                // A camera-only label rides the box's TOP edge, where the
                // detector drew its own, so it does not share the strip of
                // bottom edges with the fused labels and the answer's. No
                // stale suffix: the corner stamp says the frame is stale, and
                // a detection no track matched has no track's class to
                // corrupt.
                for c in camera_only {
                    let label = camera_only_label(c[1] as u32, c[2]);
                    mins.push([c[3], c[4]]);
                    sizes.push([c[5] - c[3], c[6] - c[4]]);
                    colors.push(rgba(C_CAMERA_ONLY, LABELLED_ALPHA));
                    radii.push(TRACK_OUTLINE_PT);
                    classes.push(CLASS_CAMERA_ONLY);
                    hover.push(label.clone());
                    let foot = [0.5 * (c[3] + c[5]), c[4]];
                    if !gives_way(foot, &label) {
                        feet.push(foot);
                        label_colors.push(rgba(C_CAMERA_ONLY, LABELLED_ALPHA));
                        labels.push(label);
                    }
                }
                // The drawn labels ride points on the boxes' edges, for the
                // reason the answer's does: the viewer wraps a box's own label
                // at the box's width.
                let logged = rec
                    .log(
                        entity::CAMERA_TRACKS,
                        &Boxes2D::from_mins_and_sizes(mins, sizes)
                            .with_colors(colors)
                            .with_radii(radii.into_iter().map(Radius::new_ui_points))
                            .with_class_ids(classes)
                            .with_labels(hover)
                            .with_show_labels(false),
                    )
                    .and_then(|()| {
                        rec.log(
                            entity::CAMERA_TRACKS,
                            &Points2D::new(feet)
                                .with_colors(label_colors)
                                .with_radii([Radius::new_ui_points(0.5 * TRACK_OUTLINE_PT)])
                                .with_labels(labels)
                                .with_show_labels(true),
                        )
                    });
                if logged.is_err() {
                    report.viz_log_errors += 1;
                }
                // Which frame all of that was fused with, and how old it was,
                // in the picture's own corner: grey on a clean pair, amber on
                // a stale one.
                let status_rgb = if stale { C_ATTENTION } else { C_STATUS };
                let logged = rec.log(
                    entity::CAMERA_STATUS,
                    &Points2D::new([STATUS_AT])
                        .with_colors([Color::from_rgb(status_rgb[0], status_rgb[1], status_rgb[2])])
                        .with_radii([Radius::new_ui_points(0.1)])
                        .with_labels([camera_status(cs, meta.pair_age_ns, stale)])
                        .with_show_labels(true),
                );
                if logged.is_err() {
                    report.viz_log_errors += 1;
                }
                // And on a stale pair the picture itself dims under the boxes
                // -- the class on them came from an older frame than the one
                // shown -- so staleness is never the amber alone. The
                // image's own opacity, set here on every answer and nowhere
                // else, so it holds until the next answer says otherwise: the
                // image consumer never sets it.
                let opacity = if stale { STALE_IMAGE_OPACITY } else { 1.0 };
                if rec
                    .log(
                        entity::CAMERA_IMAGE,
                        &Image::update_fields().with_opacity(opacity),
                    )
                    .is_err()
                {
                    report.viz_log_errors += 1;
                }
                // The flagged record: the box on `camera/answer`, and its
                // label on its own entity, a point at the middle of the box's
                // bottom edge kept on the picture ([`answer_label_at`]). The
                // point is a hair, so only its words show. A frame with no
                // answer clears both: a gold rectangle round nothing, or its
                // words over nothing, is the one thing this picture must
                // never say.
                let logged = match (answer.has_object, answer.image) {
                    (true, Some(b)) => rec
                        .log(
                            entity::CAMERA_ANSWER,
                            &Boxes2D::from_mins_and_sizes([[b.x0, b.y0]], [[b.w(), b.h()]])
                                .with_colors([color])
                                .with_radii([Radius::new_ui_points(ANSWER_OUTLINE_PT)])
                                .with_class_ids([CLASS_ANSWER])
                                .with_show_labels(false),
                        )
                        .and_then(|()| {
                            rec.log(
                                entity::CAMERA_ANSWER_LABEL,
                                &Points2D::new([answer_label_at(&b, cfg.image_wh)])
                                    .with_colors([color])
                                    .with_radii([Radius::new_ui_points(ANSWER_LABEL_POINT_PT)])
                                    .with_class_ids([CLASS_ANSWER])
                                    .with_labels([format!("{}{suffix}", answer_label(&answer))])
                                    .with_show_labels(true),
                            )
                        }),
                    _ => rec
                        .log(entity::CAMERA_ANSWER, &Clear::flat())
                        .and_then(|()| rec.log(entity::CAMERA_ANSWER_LABEL, &Clear::flat())),
                };
                if logged.is_err() {
                    report.viz_log_errors += 1;
                }
            }
            rec.send(Subject::of_sample(&rc, s));
            report.viz_bytes_total += bytes_alloc(SLOT_STATE) - v0;
            report.viz_ns_total += now() - t_viz;
        }

        let derived = Sample {
            stream: StreamId::EGO,
            // The sweep's own number: see `reduce`.
            seq: s.seq,
            arrival_seq: 0,
            parent: Some((s.stream, s.seq)),
            tov: s.tov,
            epoch: s.epoch,
            due: s.due,
            arrival: now(),
            payload: batch,
            storage_id: out_storage,
            decode_ns: 0,
        };
        let admitted = admission.admit(derived);
        let mut produced_row = Evidence::driver_admitted(&rc, &admitted, "state", "state");
        produced_row.bytes_alloc = build_bytes;
        sink.send(EvRow::Evidence(produced_row), &ctx, "state");
        report.produced += 1;
        report.out_by_seq.push((admitted.seq, out_storage));
        report.out_parent.push((admitted.seq, s.seq));
    }
    report
}

/// The `pair_outcome` string as a `&'static str`, so it can live in an
/// evidence row and in a rebuilt batch.
///
/// A lookup rather than a leak: the payload's value came from
/// [`Pairing::name`], which is already `'static`, and re-deriving it here means
/// a value that is not one of the six shows up as `pair_unknown` instead of
/// being copied through as though it were.
fn pairing_name(s: Option<&str>) -> &'static str {
    for p in [
        Pairing::Paired,
        Pairing::Stale,
        Pairing::Dropped,
        Pairing::Late,
        Pairing::Absent,
        Pairing::AbsentInSource,
    ] {
        if s == Some(p.name()) {
            return p.name();
        }
    }
    "pair_unknown"
}

/// `state-sink` configuration.
pub struct StateSinkCfg {
    /// Capacity hint for the per-sample vectors.
    pub expect: usize,
    /// `--dashboard` only: mirror the answer onto the recording under
    /// `answer/`, as numbers and as a sentence, after the measured window has
    /// closed. The one group on the dashboard that measures the WORLD rather
    /// than the pipeline.
    pub rec: Option<Canvas>,
}

/// What the `state-sink` consumer returns: the far end of the whole chain.
pub struct StateSinkReport {
    /// Answers popped off `state->sink`.
    pub delivered: u64,
    /// Batches whose re-derived buffer address disagreed with the envelope's.
    pub storage_mismatch: u64,
    /// Samples that did not name `tracks` as their parent stream.
    pub parent_mismatch: u64,
    /// Bytes this stage allocated per sample. Meant to be 0.
    pub bytes_total: u64,
    /// Arrow bytes carried across `state->sink`.
    pub payload_bytes_total: u64,
    /// Raw laser returns behind those answers, read out of the batch's own
    /// column. **This is what makes the whole chain checkable at the far end**
    /// rather than only at the stage that computed it.
    pub source_points_total: u64,
    /// Tracks behind them, from the same place.
    pub source_tracks_total: u64,
    /// Records the answers actually carried, counted off the payload: one
    /// per track, so this equals `source_tracks_total` or a track was lost
    /// between the column that counts them and the list that holds them.
    pub objects_total: u64,
    /// Of those, records in the camera frame (lane 7).
    pub in_frame_total: u64,
    /// Records flagged `nearest_in_path`, over all answers.
    pub flagged_total: u64,
    /// Answers whose `has_object` column says something is in the path.
    /// Equal to `flagged_total` exactly when every such answer flags one
    /// record and no other answer flags any.
    pub with_object: u64,
    /// Records whose redundant lanes disagree (`in_frame` against the box, a
    /// flag on a record that did not earn it): `record_is_consistent`.
    pub inconsistent_total: u64,
    /// Answers whose number of flagged records is not what their own
    /// `has_object` column says -- one when it is set, none when it is not.
    /// Per answer, because the totals above can balance across answers: one
    /// answer flagging two records and another that says it has an object
    /// flagging none sum to the same `flagged_total` as two correct ones.
    pub flag_mismatch: u64,
    /// Answers carrying a different number of records than their own
    /// `source_tracks` column says arrived, on the same per-answer terms.
    pub count_mismatch: u64,
    /// Answers whose sweep was paired with a camera frame inside its range.
    pub paired: u64,
    /// Answers built from a **declared stale** camera frame, read out of the
    /// payload at the far end.
    ///
    /// The positive control for `--pair-stale-ms`. `paired` alone is a
    /// negative — "not all of them were clean" is also what a stage that never
    /// wrote `pair_outcome` at all would report — and a run that asks for
    /// stale pairs has to be able to prove the label survived `state`'s
    /// rebuild of the batch and reached the consumer that acts on it.
    pub stale: u64,
    /// Records by population, read off lane 16 at the far end: fused with a
    /// camera detection, lidar-only while a detector ran, and unfused
    /// (`--detector off`).
    pub fused_total: u64,
    pub lidar_only_total: u64,
    pub unfused_total: u64,
    /// The camera's detections that matched no track, read off the answer's
    /// `camera_only` column: the third population, which is never a record.
    pub camera_only_total: u64,
    /// Answers whose `fused_count` column disagrees with the records it
    /// counts. Meant to be 0; per answer, for the reason `flag_mismatch` is.
    pub fused_mismatch: u64,
    /// The last answer that arrived, in words ([`answer_words`]: led by how
    /// stale the camera was when it was).
    pub last_line: Option<String>,
    /// `(own seq, the seq this sample names as its parent)`.
    pub parent_by_seq: Vec<(u64, Option<u64>)>,
    /// `(seq, storage_id re-derived from the batch)` per delivered sample.
    pub by_seq: Vec<(u64, usize)>,
    /// `--dashboard` only: bytes mirroring the answer cost, on this stage's
    /// own slot and after `proc_end`.
    pub viz_bytes_total: u64,
    /// Nanoseconds the same mirroring cost.
    pub viz_ns_total: i64,
    /// Samples the viewer would not accept.
    pub viz_log_errors: u64,
}

impl StateSinkReport {
    /// Raw laser returns per answer: the whole chain, recomputed from the
    /// columns the batch carries rather than from anything this process
    /// remembered.
    pub fn chain_shrink(&self) -> Option<f64> {
        (self.delivered > 0).then(|| self.source_points_total as f64 / self.delivered as f64)
    }

    /// Arrow bytes per answer.
    pub fn payload_bytes_mean(&self) -> Option<u64> {
        (self.delivered > 0).then(|| self.payload_bytes_total / self.delivered)
    }

    /// Records per answer: every track the fusion emitted.
    pub fn objects_mean(&self) -> Option<f64> {
        (self.delivered > 0).then(|| self.objects_total as f64 / self.delivered as f64)
    }

    /// Records per answer that are in the camera frame.
    pub fn in_frame_mean(&self) -> Option<f64> {
        (self.delivered > 0).then(|| self.in_frame_total as f64 / self.delivered as f64)
    }

    /// Raw laser returns per record: the whole chain, per object.
    pub fn chain_shrink_per_object(&self) -> Option<f64> {
        (self.objects_total > 0)
            .then(|| self.source_points_total as f64 / self.objects_total as f64)
    }
}

/// The share of the in-frame lidar tracks the camera confirmed, `fused /
/// (fused + lidar_only)`, or `None` -- a gap -- for a frame with no lidar
/// track in it, where a share of nothing is not 0.
fn fused_fraction(fused: u32, lidar_only: u32) -> Option<f64> {
    let n = fused + lidar_only;
    (n > 0).then(|| f64::from(fused) / f64::from(n))
}

/// The answer as the demo screen's headline: one markdown heading, read from
/// across a room. `## 11.4 m ahead   ·   closing 3.7 m/s   ·   TTC 3.1 s   ·
/// answer 26 ms old`; `receding` or `not closing` in place of the closing
/// speed and the TTC when the object is not closing; `no object in the
/// corridor` when nothing is flagged. The camera's class leads when the
/// fusion gave the object one, and takes the closing speed's place, as it
/// does in the answer's label (`car 0.91 · 11.4 m ahead · TTC 3.1 s · ...`):
/// a stale answer with a class and a closing speed ran past one line at
/// 1600x900 and lost its last words. The answer's age (`age_ms`, how long
/// after the sweep was due the answer reached the end of the chain) follows
/// when the run was paced, and a stale camera half ends it: `STALE camera
/// 93 ms`.
pub fn headline(a: &StateAnswer, cam: CamHalf, age_ms: Option<f64>) -> String {
    let mut parts: Vec<String> = Vec::new();
    if a.has_object {
        if let Some((k, q)) = a.class {
            parts.push(format!("{} {q:.2}", class_name(k)));
        }
        parts.push(format!("{:.1} m ahead", a.distance_m));
        if a.ttc_s > 0.0 && a.closing_mps > 0.0 {
            if a.class.is_none() {
                parts.push(format!("closing {:.1} m/s", a.closing_mps));
            }
            parts.push(format!("TTC {:.1} s", a.ttc_s));
        } else {
            parts.push(answer_motion(a, false));
        }
    } else {
        parts.push("no object in the corridor".to_string());
    }
    if let Some(ms) = age_ms {
        parts.push(format!("answer {ms:.0} ms old"));
    }
    if let Some(note) = stale_note(cam.pair_outcome, cam.pair_age_ns) {
        parts.push(note);
    }
    // Wide separators, as the layout was designed with, while the line fits
    // one heading line of the headline's view at 1600x900; narrow ones when
    // it would not -- a stale answer that is closing runs to 98 characters
    // with wide ones, and its second line, `STALE ...` included, was cut
    // off.
    let wide = parts.join("   ·   ");
    if wide.chars().count() <= HEADLINE_WIDE_MAX {
        format!("## {wide}")
    } else {
        format!("## {}", parts.join(" · "))
    }
}

/// The longest headline, in characters, drawn with wide separators: what
/// one `##` line holds across the headline's view at 1600x900, measured at
/// about 9.8 px a character over the view's 975 px.
const HEADLINE_WIDE_MAX: usize = 95;

/// The TTC plot's point for one answer: its time to contact, drawn at
/// [`TTC_TOP_S`] when it is that or more, or `None` -- a gap -- when nothing
/// is flagged or the flagged object is not closing.
fn ttc_point(a: &StateAnswer) -> Option<f64> {
    (a.has_object && a.closing_mps > 0.0 && a.ttc_s > 0.0)
        .then(|| f64::from(a.ttc_s).min(TTC_TOP_S))
}

/// Body of the `state-sink` thread: the leaf consumer of the answer.
///
/// The smallest honest one, on the terms [`obj_thread`] and [`cloud_thread`]
/// are: it counts, it reads the provenance, it re-derives the buffer address,
/// and it allocates nothing. What it adds is the end-to-end statement — raw
/// returns per answer — taken from the sample in its hand.
pub fn state_sink_thread(
    q: Arc<BoundedQueue<Arc<Sample>>>,
    sink: Arc<EvidenceSink>,
    ctx: Arc<RunCtx>,
    cfg: StateSinkCfg,
) -> StateSinkReport {
    set_stage_slot(SLOT_STATE_SINK);
    let rc = ctx.row_ctx();
    let mut report = StateSinkReport {
        delivered: 0,
        storage_mismatch: 0,
        parent_mismatch: 0,
        bytes_total: 0,
        payload_bytes_total: 0,
        source_points_total: 0,
        source_tracks_total: 0,
        objects_total: 0,
        in_frame_total: 0,
        flagged_total: 0,
        with_object: 0,
        inconsistent_total: 0,
        flag_mismatch: 0,
        count_mismatch: 0,
        paired: 0,
        stale: 0,
        fused_total: 0,
        lidar_only_total: 0,
        unfused_total: 0,
        camera_only_total: 0,
        fused_mismatch: 0,
        last_line: None,
        parent_by_seq: Vec::with_capacity(cfg.expect),
        by_seq: Vec::with_capacity(cfg.expect),
        viz_bytes_total: 0,
        viz_ns_total: 0,
        viz_log_errors: 0,
    };
    let mut styled = false;
    while let Some(env) = q.pop() {
        let s: &Sample = &env.item;
        let dequeued = now();

        let proc_start = now();
        let b0 = bytes_alloc(SLOT_STATE_SINK);
        let answer = read_answer(&s.payload);
        std::hint::black_box(&answer);
        let bytes = bytes_alloc(SLOT_STATE_SINK) - b0;
        let proc_end = now();

        // Everything below is outside the measured window.
        if s.parent.map(|(stream, _)| stream) != Some(StreamId::TRACKS) {
            report.parent_mismatch += 1;
        }
        report
            .parent_by_seq
            .push((s.seq, s.parent.map(|(_, seq)| seq)));
        report.source_points_total += u64::from(state_source_points(&s.payload).unwrap_or(0));
        report.source_tracks_total += u64::from(state_source_tracks(&s.payload).unwrap_or(0));
        // The structure of the answer, read off the payload at the far end:
        // every record counted, the flag counted per answer, and every
        // record's redundant lanes checked against each other. Outside the
        // measured window: it is the check, not the consumer's work.
        let records = object_rows(&s.payload);
        let mut flags = 0u64;
        let mut fused = 0u64;
        for r in records {
            let o = Object::from_lanes(r);
            report.in_frame_total += u64::from(r[7] == 1.0);
            flags += u64::from(o.nearest_in_path);
            report.inconsistent_total += u64::from(!record_is_consistent(r));
            match o.population {
                POPULATION_FUSED => fused += 1,
                POPULATION_LIDAR_ONLY => report.lidar_only_total += 1,
                _ => report.unfused_total += 1,
            }
        }
        report.fused_total += fused;
        report.fused_mismatch +=
            u64::from(state_fused_count(&s.payload).map(u64::from) != Some(fused));
        report.camera_only_total += camera_only_rows(&s.payload).len() as u64;
        report.objects_total += records.len() as u64;
        report.flagged_total += flags;
        // The column, not `read_answer`: that returns nothing when the column
        // and the flags disagree, which would drop the answer from BOTH sides
        // of the comparison this count exists for.
        let has_object = state_has_object(&s.payload) == Some(true);
        report.with_object += u64::from(has_object);
        report.flag_mismatch += u64::from(flags != u64::from(has_object));
        report.count_mismatch +=
            u64::from(state_source_tracks(&s.payload).map(u64::from) != Some(records.len() as u64));
        report.paired += u64::from(state_pair_outcome(&s.payload) == Some(Pairing::Paired.name()));
        report.stale += u64::from(state_pair_outcome(&s.payload) == Some(Pairing::Stale.name()));
        // Allocates, and deliberately after `proc_end`: a string is not what
        // this stage is measured for.
        if let Some(a) = answer {
            report.last_line = Some(answer_words(&a, CamHalf::of_answer(&s.payload)));
        }

        // **The end of the chain, on the dashboard.** Every other series on
        // that page measures the PIPELINE -- how late, how full, how many
        // bytes. These measure the world, and live under their own root.
        //
        // The time to contact is logged ONLY while the flagged object
        // closes. `state` refuses to spell "nothing ahead" as a distance of
        // 0, and it writes a TTC of 0 for an object that is not closing; a
        // point on the axis on those sweeps would read as a collision. So the
        // series has a GAP there, and the headline says why ("receding",
        // "no object in the corridor"). A TTC of 10 s or more is drawn at 10,
        // the plot's top edge. The distance and the closing speed are words
        // in the headline and the label, not lines: a line joined one
        // object's numbers to the next one's. The populations in the camera
        // frame are logged on every answer: every answer has them, and 0 is a
        // count.
        if let (Some(rec), Some(a)) = (&cfg.rec, answer) {
            let t_viz = now();
            let v0 = bytes_alloc(SLOT_STATE_SINK);
            rec.set_timestamp_nanos_since_epoch("sensor_time", s.tov.start().map_or(0, |t| t.0));
            rec.set_duration_secs("host", (s.arrival - ctx.t0_host) as f64 * 1e-9);
            // The sweep the answer is about, not the answer's own count: see
            // `sweep_timeline`, and `answer_line`, which names the same sweep.
            rec.set_time_sequence("seq", sweep_timeline(state_sweep_seq(&s.payload), s.seq));
            // Styled once, on the first answer, for the reason the dashboard
            // styles on first sight: an unstyled series is an anonymous line.
            if !styled {
                if let Some(style) = series_archetype(FUSED_FRACTION_LEAF) {
                    if rec
                        .log_static(entity::TRACKS_FUSED_FRACTION, style.as_ref())
                        .is_err()
                    {
                        report.viz_log_errors += 1;
                    }
                }
                for leaf in ANSWER_LEAVES {
                    if let Some(style) = series_archetype(leaf) {
                        if rec
                            .log_static(format!("{}/{leaf}", entity::ANSWER), style.as_ref())
                            .is_err()
                        {
                            report.viz_log_errors += 1;
                        }
                    }
                }
                for leaf in TRACK_COUNT_LEAVES {
                    if let Some(style) = series_archetype(leaf) {
                        if rec
                            .log_static(format!("{}/{leaf}", entity::TRACKS_COUNT), style.as_ref())
                            .is_err()
                        {
                            report.viz_log_errors += 1;
                        }
                    }
                }
                styled = true;
            }
            let put = |leaf: &str, v: f64, errs: &mut u64| {
                if rec
                    .log(format!("{}/{leaf}", entity::ANSWER), &Scalars::single(v))
                    .is_err()
                {
                    *errs += 1;
                }
            };
            if let Some(ttc) = ttc_point(&a) {
                put("ttc_s", ttc, &mut report.viz_log_errors);
            }
            put("ttc_warn", TTC_WARN_S, &mut report.viz_log_errors);
            // The fusion's populations in the frame, off the records' own
            // lanes and the answer's `camera_only` column: what the camera
            // half contributed to this answer, on every answer.
            let (mut fused, mut lidar_only) = (0u32, 0u32);
            for r in object_rows(&s.payload) {
                if r[7] == 1.0 {
                    if Object::from_lanes(r).population == POPULATION_FUSED {
                        fused += 1;
                    } else {
                        lidar_only += 1;
                    }
                }
            }
            let camera_only = camera_only_rows(&s.payload).len();
            for (leaf, v) in [
                ("fused", f64::from(fused)),
                ("lidar_only", f64::from(lidar_only)),
                ("camera_only", camera_only as f64),
            ] {
                if rec
                    .log(
                        format!("{}/{leaf}", entity::TRACKS_COUNT),
                        &Scalars::single(v),
                    )
                    .is_err()
                {
                    report.viz_log_errors += 1;
                }
            }
            // The share of the lidar's tracks in the frame the camera
            // confirmed: 0 to 1, and a gap on a frame with none in it. A
            // stale camera moves it down, which the Fusion tab shows beside
            // its cause.
            if let Some(share) = fused_fraction(fused, lidar_only) {
                if rec
                    .log(entity::TRACKS_FUSED_FRACTION, &Scalars::single(share))
                    .is_err()
                {
                    report.viz_log_errors += 1;
                }
            }
            // The answer as the demo screen's headline, in large type: what
            // is ahead, how it moves, how old the answer is -- the same age
            // the evidence row below carries -- and, when the camera half
            // was stale, how stale.
            let age_ms = s.due.map(|d| (proc_end - d) as f64 * 1e-6);
            if rec
                .log(
                    entity::ANSWER_HEADLINE,
                    &TextDocument::new(headline(&a, CamHalf::of_answer(&s.payload), age_ms))
                        .with_media_type(MediaType::markdown()),
                )
                .is_err()
            {
                report.viz_log_errors += 1;
            }
            // The same answer in words, which is the artefact a reader who has
            // never seen this project can actually judge. It names the sweep
            // and the camera frame it was fused with, so the sentence names
            // both sensors.
            if rec
                .log(
                    format!("{}/line", entity::ANSWER),
                    &TextLog::new(answer_line(&s.payload, &a)),
                )
                .is_err()
            {
                report.viz_log_errors += 1;
            }
            rec.send(Subject::of_sample(&rc, s));
            report.viz_bytes_total += bytes_alloc(SLOT_STATE_SINK) - v0;
            report.viz_ns_total += now() - t_viz;
        }

        let derived = state_storage_id(&s.payload);
        if derived != Some(s.storage_id) {
            report.storage_mismatch += 1;
        }
        let row = Evidence::delivered(
            &rc,
            s,
            "state->sink",
            "state-sink",
            env.enqueued,
            env.depth_at_push,
            env.depth_after_pop,
            dequeued,
            proc_start,
            proc_end,
            derived.unwrap_or(0),
            bytes,
        );
        sink.send(EvRow::Evidence(row), &ctx, "state-sink");
        report.delivered += 1;
        report.bytes_total += bytes;
        report.payload_bytes_total += s.payload_bytes() as u64;
        report.by_seq.push((s.seq, derived.unwrap_or(0)));
    }
    report
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn a_detector_box_says_class_and_confidence_and_is_not_an_answer_colour() {
        let mut d = Det {
            x0: 1.0,
            y0: 2.0,
            x1: 3.0,
            y1: 4.0,
            class_id: 2,
            score: 0.912,
            row: 0,
        };
        assert_eq!(cam_det_label(&d), "car 0.91");
        // The class comes from the id, not from a fixed word.
        d.class_id = 0;
        d.score = 0.3001;
        assert_eq!(cam_det_label(&d), "person 0.30");
        // The purple is none of the colours the answer or its states use, so a
        // box the camera found cannot be read as the one the answer flags.
        for c in [C_ANSWER, C_ATTENTION] {
            assert_ne!(C_CAM_DET, c);
        }
    }

    #[test]
    fn both_clouds_are_the_lidar_s_teal_and_the_voxels_are_the_heavier() {
        // Colour on the lidar pictures is provenance, like everywhere else:
        // both clouds are the lidar's, so both are its teal -- no red, green
        // above red, blue above red -- and neither is a population's or the
        // answer's colour.
        for c in [SWEEP_RGBA, VOXEL_RGBA] {
            assert!(c[0] == 0 && c[1] > 100 && c[2] > 100, "{c:?} is not teal");
            let rgb = [c[0], c[1], c[2]];
            for other in [C_FUSED, C_CAMERA_ONLY, C_ANSWER, C_ATTENTION] {
                assert_ne!(rgb, other);
            }
        }
        // The sweep's is exactly the lidar-only tracks' teal, #00A8B0.
        assert_eq!([SWEEP_RGBA[0], SWEEP_RGBA[1], SWEEP_RGBA[2]], C_LIDAR_ONLY);
        // A voxel stands for a cell, so it is drawn more opaque.
        assert!(VOXEL_RGBA[3] > SWEEP_RGBA[3]);
    }

    #[test]
    fn every_answer_lane_this_stage_draws_has_a_name_and_a_colour() {
        // `state-sink` logs its series straight onto the recording instead of
        // through the dashboard's evidence mirror, so nothing else would catch
        // a lane added here and never styled. It would ship as an anonymous
        // grey line in a recording, which is exactly the failure the styling
        // table exists to prevent.
        for leaf in ANSWER_LEAVES.iter().chain(TRACK_COUNT_LEAVES.iter()) {
            assert!(
                series_archetype(leaf).is_some(),
                "{leaf} is drawn by state-sink and has no style"
            );
        }
        // The `track` stage draws the pairing's the same way.
        for leaf in PAIRING_LEAVES {
            assert!(
                series_archetype(leaf).is_some(),
                "{leaf} is drawn by track and has no style"
            );
        }
        // The positive control on the check itself: a name that is NOT in the
        // table has to come back `None`, or the loop above passes against a
        // function that says yes to everything.
        assert!(series_archetype("answer_not_a_lane").is_none());
        // No duplicates, across both stages that log their own series: a
        // repeated leaf would overwrite its own scalar and silently halve the
        // series.
        let mut seen: Vec<&str> = ANSWER_LEAVES
            .iter()
            .chain(TRACK_COUNT_LEAVES.iter())
            .chain(PAIRING_LEAVES.iter())
            .copied()
            .collect();
        seen.sort_unstable();
        let n = seen.len();
        seen.dedup();
        assert_eq!(seen.len(), n, "a lane is drawn twice per sweep");
    }

    #[test]
    fn colour_is_population_and_no_population_is_the_answer_s() {
        // Colour on the pictures means PROVENANCE -- which sensors vouch for
        // the object -- so the three populations are three colours, none of
        // them the answer's gold or the stale amber, and the class ids the
        // boxes carry are the ones the annotation context names.
        let pops = [C_FUSED, C_LIDAR_ONLY, C_CAMERA_ONLY];
        for (i, a) in pops.iter().enumerate() {
            for b in &pops[i + 1..] {
                assert_ne!(a, b, "two populations share a colour");
            }
            assert_ne!(*a, C_ANSWER);
            assert_ne!(*a, C_ATTENTION);
        }
        // The v2 layout's palette, by value: fused #5CB278, lidar #00A8B0,
        // camera #408CD6.
        assert_eq!(C_FUSED, [0x5C, 0xB2, 0x78]);
        assert_eq!(C_LIDAR_ONLY, [0x00, 0xA8, 0xB0]);
        assert_eq!(C_CAMERA_ONLY, [0x40, 0x8C, 0xD6]);
        assert_eq!(
            [
                CLASS_FUSED,
                CLASS_LIDAR_ONLY,
                CLASS_CAMERA_ONLY,
                CLASS_ANSWER
            ],
            [1, 2, 3, 9]
        );
        // The class ids are the population codes the payload carries, so a
        // record's lane 16 and its box's class id say the same thing.
        assert_eq!(u32::from(CLASS_FUSED), POPULATION_FUSED);
        assert_eq!(u32::from(CLASS_LIDAR_ONLY), POPULATION_LIDAR_ONLY);
        assert_eq!(
            u32::from(CLASS_CAMERA_ONLY),
            pipes_kitti::fuse::POPULATION_CAMERA_ONLY
        );
    }

    #[test]
    fn a_fused_label_says_what_both_sensors_say_and_a_camera_only_one_says_no_range() {
        let o = Object {
            id: 1838,
            distance_m: 7.561,
            closing_mps: 4.23,
            age_s: 2.78,
            class: Some((2, 0.89)),
            population: POPULATION_FUSED,
            ..Object::default()
        };
        assert_eq!(fused_label(&o), "car 0.89 · 7.6 m · 4.2 m/s · 2.8 s");
        let away = Object {
            closing_mps: -0.4,
            class: Some((0, 0.7)),
            ..o
        };
        assert_eq!(fused_label(&away), "person 0.70 · 7.6 m · -0.4 m/s · 2.8 s");
        // The fused label leads exactly as the answer's does, so the one
        // object both describe reads the same camera half in both places.
        let a = StateAnswer {
            has_object: true,
            object_id: 1838,
            class: o.class,
            ..StateAnswer::default()
        };
        let lead = |s: &str| s.split(" · ").next().unwrap_or_default().to_string();
        assert_eq!(lead(&fused_label(&o)), lead(&answer_label(&a)));
        assert_eq!(camera_only_label(2, 0.624), "car 0.62 · no range");
        // The status line: which frame, and how its instant sat against the
        // sweep's trigger, positive when paired and negative when stale.
        assert_eq!(
            camera_status(76, 10_503_000, false),
            "cam 76 · paired +11 ms"
        );
        assert_eq!(
            camera_status(75, -92_790_000, true),
            "STALE · cam 75 · -93 ms"
        );
    }

    #[test]
    fn the_answer_label_is_one_short_line_with_the_numbers_that_matter() {
        // The label sits ON the picture, under the object it names, so it is
        // the distance and how the object moves -- never the sentence, which
        // wrapped into a blob that hid the object, and never the id, which a
        // reader cannot act on.
        let closing = StateAnswer {
            has_object: true,
            object_id: 597,
            age_s: 4.3,
            distance_m: 11.88,
            closing_mps: 3.69,
            ttc_s: 3.22,
            bearing_deg: 2.96,
            image: None,
            class: None,
            candidates: 2,
        };
        assert_eq!(
            answer_label(&closing),
            "11.9 m · closing 3.7 m/s · TTC 3.2 s"
        );
        // A gap that grows is `receding` -- not a TTC of 0.0 s, which reads
        // as a collision -- and a closing speed of exactly zero is neither.
        let receding = StateAnswer {
            closing_mps: -0.4,
            ttc_s: 0.0,
            ..closing
        };
        assert_eq!(answer_label(&receding), "11.9 m · receding");
        let holding = StateAnswer {
            closing_mps: 0.0,
            ttc_s: 0.0,
            ..closing
        };
        assert_eq!(answer_label(&holding), "11.9 m · not closing");
        // Fused, the camera's class and confidence lead it, in the closing
        // speed's place.
        let car = StateAnswer {
            class: Some((2, 0.914)),
            ..closing
        };
        assert_eq!(answer_label(&car), "car 0.91 · 11.9 m · TTC 3.2 s");
        assert_eq!(
            answer_label(&StateAnswer {
                class: Some((0, 0.5)),
                ..receding
            }),
            "person 0.50 · 11.9 m · receding"
        );
        // One line, no id, and short enough not to cover the box it labels.
        for a in [&closing, &receding, &holding, &car] {
            let l = answer_label(a);
            assert!(!l.contains('\n'));
            assert!(l.chars().count() <= 40, "{l}");
            assert!(!l.contains('#') && !l.contains("597"), "{l}");
            assert!(!l.contains("deg") && !l.contains("sweeps"), "{l}");
        }
        // In 3D, beside the wireframe: distance and TTC only.
        assert_eq!(answer_label_3d(&car), "11.9 m · TTC 3.2 s");
        assert_eq!(answer_label_3d(&receding), "11.9 m · receding");
    }

    #[test]
    fn the_ttc_plot_draws_a_closing_object_only_and_a_long_ttc_at_its_top_edge() {
        let a = StateAnswer {
            has_object: true,
            distance_m: 11.4,
            closing_mps: 3.7,
            ttc_s: 3.1,
            ..StateAnswer::default()
        };
        assert_eq!(ttc_point(&a), Some(f64::from(3.1f32)));
        // 40 s away is drawn at the top edge, as "10 s or more".
        let far = StateAnswer {
            closing_mps: 0.3,
            ttc_s: 40.0,
            ..a
        };
        assert_eq!(ttc_point(&far), Some(TTC_TOP_S));
        // Receding, holding distance, or nothing flagged: a gap, never a
        // point at 0, which reads as a collision.
        for gap in [
            StateAnswer {
                closing_mps: -0.4,
                ttc_s: 0.0,
                ..a
            },
            StateAnswer {
                closing_mps: 0.0,
                ttc_s: 0.0,
                ..a
            },
            StateAnswer::default(),
        ] {
            assert_eq!(ttc_point(&gap), None, "{gap:?}");
        }
    }

    #[test]
    fn the_pairing_window_is_the_sweep_s_range_and_a_stale_frame_falls_below_it() {
        // Drive 0005's shape: a 103.3 ms sweep, its trigger 51.6 ms in, and
        // the paired frame 10.5 ms after the trigger.
        let t0 = 1_317_000_000_000_000_000i64;
        let ms = 1_000_000i64;
        let (end, trig) = (t0 + 103_300_000, t0 + 51_600_000);
        let w = PairingWindow::of(t0, end, trig, Some(trig + 10_500_000));
        assert!((w.period_ms - 103.3).abs() < 1e-9, "{w:?}");
        assert!((w.trigger_ms.unwrap() - 51.6).abs() < 1e-9, "{w:?}");
        let c = w.camera_ms.unwrap();
        assert!(
            c > 0.0 && c < w.period_ms && (c - 62.1).abs() < 1e-9,
            "{w:?}"
        );
        // A stale frame, a period earlier, is below the band's start.
        let stale = PairingWindow::of(t0, end, trig, Some(trig + 10_500_000 - 103 * ms));
        assert!(stale.camera_ms.unwrap() < 0.0, "{stale:?}");
        // An expired set has no camera point, and a trigger outside the
        // range (or none, read as 0) draws no trigger line.
        let expired = PairingWindow::of(t0, end, 0, None);
        assert_eq!(expired.camera_ms, None);
        assert_eq!(expired.trigger_ms, None);
        assert_eq!(expired.period_ms, w.period_ms);
        // The camera instant is drawn as points in the camera's blue, the
        // range's ends as lines.
        for leaf in PAIRING_LEAVES {
            assert!(series_archetype(leaf).is_some(), "{leaf}");
        }
    }

    #[test]
    fn the_headline_says_what_is_ahead_how_it_moves_how_old_and_whether_stale() {
        let clean = CamHalf {
            cam_seq: 76,
            pair_outcome: Pairing::Paired.name(),
            pair_age_ns: 10_503_000,
        };
        let a = StateAnswer {
            has_object: true,
            object_id: 597,
            distance_m: 11.38,
            closing_mps: 3.69,
            ttc_s: 3.08,
            ..StateAnswer::default()
        };
        assert_eq!(
            headline(&a, clean, Some(25.6)),
            "## 11.4 m ahead   ·   closing 3.7 m/s   ·   TTC 3.1 s   ·   answer 26 ms old"
        );
        // Receding: no TTC, and never a TTC of 0.
        let away = StateAnswer {
            closing_mps: -0.4,
            ttc_s: 0.0,
            ..a
        };
        assert_eq!(
            headline(&away, clean, Some(16.0)),
            "## 11.4 m ahead   ·   receding   ·   answer 16 ms old"
        );
        // Fused, the camera's class leads; stale, the staleness ends it.
        let stale = CamHalf {
            cam_seq: 75,
            pair_outcome: Pairing::Stale.name(),
            pair_age_ns: -92_790_000,
        };
        let car = StateAnswer {
            class: Some((2, 0.914)),
            ..a
        };
        let h = headline(&car, stale, Some(20.0));
        assert_eq!(
            h,
            "## car 0.91   ·   11.4 m ahead   ·   TTC 3.1 s   ·   answer 20 ms old   ·   STALE camera 93 ms"
        );
        // Without a class the closing speed stays, and the line is too long
        // for wide separators: narrow ones keep it on one line.
        assert_eq!(
            headline(&a, stale, Some(20.0)),
            "## 11.4 m ahead · closing 3.7 m/s · TTC 3.1 s · answer 20 ms old · STALE camera 93 ms"
        );
        // Nothing flagged; and an unpaced run has no age to give.
        assert_eq!(
            headline(&StateAnswer::default(), clean, None),
            "## no object in the corridor"
        );
        // One heading line, no id.
        for h in [headline(&a, clean, Some(1.0)), h] {
            assert!(h.starts_with("## ") && !h.contains('\n') && !h.contains("597"));
        }
    }

    #[test]
    fn the_fused_share_is_of_the_lidar_tracks_in_frame_and_a_gap_when_there_are_none() {
        assert_eq!(fused_fraction(3, 37), Some(0.075));
        assert_eq!(fused_fraction(0, 40), Some(0.0));
        assert_eq!(fused_fraction(4, 0), Some(1.0));
        assert_eq!(fused_fraction(0, 0), None);
    }

    #[test]
    fn a_sweep_with_no_answer_says_so_and_why() {
        assert_eq!(
            expired_headline(57, Pairing::Dropped),
            "## no answer for sweep 57   ·   camera frame dropped (pair_dropped)"
        );
        assert_eq!(expiry_words(Pairing::Late), "camera frame late");
        assert_eq!(expiry_words(Pairing::Absent), "no camera frame");
        // Only an expired set is drawn this way.
        for p in [Pairing::Dropped, Pairing::Late, Pairing::Absent] {
            assert!(!p.produces());
        }
    }

    #[test]
    fn a_label_that_would_print_over_the_answer_s_gives_way() {
        // Drive 0005 at 15 s: the answer's label under its box, and a fused
        // neighbour's label 20 px to its left on the same strip.
        let answer = [558.0, 360.0];
        assert!(labels_collide([538.0, 362.0], 48, answer, 30));
        // A label a line lower, or well clear to the side, stays.
        assert!(!labels_collide([538.0, 380.0], 48, answer, 30));
        assert!(!labels_collide([160.0, 362.0], 20, answer, 30));
        // Symmetric.
        assert_eq!(
            labels_collide(answer, 30, [538.0, 362.0], 48),
            labels_collide([538.0, 362.0], 48, answer, 30)
        );
    }

    #[test]
    fn the_answer_label_stays_on_the_picture() {
        let at = |x0: f32, y0: f32, x1: f32, y1: f32| {
            answer_label_at(&ImageBox { x0, y0, x1, y1 }, Some((1242.0, 375.0)))
        };
        // Under the middle of the box, where there is room.
        assert_eq!(at(400.0, 100.0, 500.0, 200.0), [450.0, 200.0]);
        // A box at either side keeps its words 190 px in.
        assert_eq!(at(0.0, 100.0, 40.0, 200.0), [190.0, 200.0]);
        assert_eq!(at(1200.0, 100.0, 1242.0, 200.0), [1052.0, 200.0]);
        // A box that reaches the bottom keeps one line of text on the image.
        assert_eq!(at(400.0, 200.0, 500.0, 375.0), [450.0, 361.0]);
        // A frame of unknown size is not clamped.
        let b = ImageBox {
            x0: 0.0,
            y0: 0.0,
            x1: 10.0,
            y1: 10.0,
        };
        assert_eq!(answer_label_at(&b, None), [5.0, 10.0]);
    }

    #[test]
    fn an_object_label_is_one_short_line_with_its_id_distance_and_age() {
        // Forty of these sit on one picture, so each is the three numbers a
        // reader follows an object by and nothing else.
        let o = Object {
            id: 1838,
            distance_m: 7.561,
            age_s: 2.78,
            ..Object::default()
        };
        let l = object_label(&o);
        assert_eq!(l, "#1838 · 7.6 m · 2.8 s");
        assert!(!l.contains('\n') && l.chars().count() <= 24, "{l}");
    }

    #[test]
    fn the_answer_line_names_the_sweep_the_batch_names_not_the_state_stream_s_count() {
        // A run that lost sweeps upstream: this answer is about sweep 24,
        // paired with camera frame 72, and it is the first answer `state`
        // produced, so the sample carrying it has seq 0. The line must say
        // 24, which is what the lane beside it and the evidence rows say.
        // One fresh track dead ahead, 9.5 m to its near face, closing at
        // 1.2 m/s, first seen 0.2 s ago.
        let mut t = [0.0f32; pipes_kitti::track::TRACK_LANES];
        t[0] = 11.5;
        t[3] = -1.2;
        t[6] = 244.0;
        t[7] = 3.0;
        t[9..12].copy_from_slice(&[9.5, -0.9, -1.7]);
        t[12..15].copy_from_slice(&[13.5, 0.9, 0.0]);
        t[19] = 0.2;
        let a = StateAnswer::of(&[t]);
        assert!(a.has_object && a.object_id == 244, "{a:?}");
        let meta = StateMeta {
            trigger_ns: 1_000,
            sweep_seq: 24,
            source_detections: 7,
            source_voxels: 900,
            source_points: 120_000,
            cam_seq: 72,
            pair_outcome: Pairing::Paired.name(),
            pair_age_ns: 10_503_000,
        };
        let (batch, _) =
            build_state_batch(&[t], &a, &meta, None, &pipes_kitti::state::state_schema()).unwrap();
        let line = answer_line(&batch, &a);
        assert!(
            line.starts_with(
                "sweep 24 | cam frame 72 | 1 tracked, 0 in frame, 0 fused, 0 camera-only, 1 in path | "
            ),
            "{line}"
        );
        assert!(line.ends_with(&a.line()), "{line}");
        // A clean pair's words are the sentence alone, with no staleness in
        // either place.
        assert!(!line.contains("STALE"), "{line}");
        assert_eq!(answer_words(&a, CamHalf::of_answer(&batch)), a.line());

        // The same answer fused with the frame BEFORE the sweep's: the log
        // line says how stale right beside the frame it names, and the words
        // the run prints and `summary.json` keeps lead with it -- the class
        // in the sentence is that older frame's, and nothing but the frame
        // number used to say so.
        let stale_meta = StateMeta {
            cam_seq: 71,
            pair_outcome: Pairing::Stale.name(),
            pair_age_ns: -92_790_000,
            ..meta
        };
        let (stale, _) = build_state_batch(
            &[t],
            &a,
            &stale_meta,
            None,
            &pipes_kitti::state::state_schema(),
        )
        .unwrap();
        let line = answer_line(&stale, &a);
        assert!(
            line.starts_with("sweep 24 | cam frame 71 | STALE camera 93 ms | 1 tracked, "),
            "{line}"
        );
        assert!(line.ends_with(&a.line()), "{line}");
        let cam = CamHalf::of_answer(&stale);
        assert_eq!(
            cam,
            CamHalf {
                cam_seq: 71,
                pair_outcome: "stale",
                pair_age_ns: -92_790_000
            }
        );
        assert_eq!(
            answer_words(&a, cam),
            format!("STALE camera 93 ms (frame 71): {}", a.line())
        );
        assert_eq!(stale_note("paired", 10_503_000), None);
    }

    /// The answer about sweep 24 that `state` numbered 0 sits at 24 on the
    /// recording's `seq` timeline, beside sweep 24's cloud, and not at 0; a
    /// sample that names no sweep keeps its own number rather than -1.
    #[test]
    fn a_picture_of_a_sweep_sits_at_the_sweep_s_number() {
        assert_eq!(sweep_timeline(Some(24), 0), 24);
        assert_eq!(sweep_timeline(Some(0), 3), 0);
        assert_eq!(sweep_timeline(Some(-1), 7), 7);
        assert_eq!(sweep_timeline(None, 7), 7);
    }

    #[test]
    fn luma_is_the_fixed_point_weighted_sum() {
        assert_eq!(luma(&[0, 0, 0]), 0);
        // The weights sum to exactly 256, so white stays white after the
        // shift and the transform loses no range. Pinned here so that the
        // stage cannot quietly grayscale differently from the figures every
        // committed run was measured under.
        assert_eq!(luma(&[255, 255, 255]), 255);
        assert_eq!(luma(&[255, 0, 0]), 76);
        assert_eq!(luma(&[0, 255, 0]), 149);
        assert_eq!(luma(&[0, 0, 255]), 28);
    }
}
