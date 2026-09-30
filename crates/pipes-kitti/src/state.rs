//! The answer: **every object the chain is tracking around this vehicle —
//! how far, how fast the gap is closing, how long it has been tracked, where
//! it is in the picture — and which one of them is the nearest thing in the
//! vehicle's path.**
//!
//! The end of the chain. A sweep arrives as 1,947,870 B of laser returns;
//! [`crate::voxel`] makes it 492,231 B of the same returns decimated;
//! [`crate::detect`] makes that 7,990 B of structures; [`crate::track`] makes
//! that about 183 records of 104 B -- structures **with identities, ages and
//! velocities**, and since the camera detector, with the class the camera
//! gave the ones it could see ([`crate::fuse`]). This module makes one
//! [`OBJECT_BYTES`]-byte record per track, plus the camera's detections that
//! matched no track, plus the provenance columns that let anyone recompute
//! the whole chain from the sample in their hand.
//!
//! # Every track, not one: what that costs and why
//!
//! This stage used to answer with ONE object — the nearest thing in the
//! vehicle's path, 36 bytes — and the chain's end-to-end ratio (5,712x by
//! bytes) came from that choice: one object reported out of about 183. That
//! was a choice of question, not a property of the pipeline, and it threw
//! away everything the fusion knew about the other 182 — including whether
//! the camera could see them. So the answer is now **every track the fusion
//! emitted, each mapped onto the camera image**, and the old answer survives
//! as a flag on the one record it named ([`Object::nearest_in_path`]).
//!
//! The honest consequence: **the last link barely shrinks the payload any
//! more.** A record is smaller than a track (17 lanes against 26: the answer
//! drops the 3D box, the raw velocity and the association's detection index
//! and IoU, and keeps the quantities a consumer acts on), so the answer is
//! smaller than the tracks it came from and larger than the detections
//! before them. The chain compresses at `reduce` and at `detect`; after that
//! every link adds information.
//!
//! # What the camera adds to a record, and what it does not
//!
//! A record whose track [`crate::fuse`] associated with a detection of the
//! paired frame carries that detection's class and confidence in lanes 14 and
//! 15 and [`crate::fuse::POPULATION_FUSED`] in lane 16; everything else in it
//! is the lidar's. A detection no track matched is **not** a record: it has
//! no distance, no velocity and no age, and a record with zeros there is the
//! sentinel this module refuses. It travels in the `camera_only` column,
//! handed on from the track batch without a copy, with its box, class and
//! confidence and nothing else.
//!
//! The camera's contribution is attributable where it is read: the answer
//! carries `cam_seq` (the frame -- and the `CAM_DET` batch, which `camdet`
//! numbers by frame), `pair_outcome` and `pair_age_ns`, so a record whose
//! class came from a stale frame says so in the same batch.
//!
//! **Out-of-frame objects are carried and flagged, never dropped.** About
//! three in four tracks are outside the camera's 81 degrees; they are still
//! objects the vehicle has to know about, and a camera-side consumer tells
//! them apart by [`Object::image`] (lane 7 and the box, which a test pins
//! agreeing), not by their absence.
//!
//! # What "in front of this vehicle" means, and where the number comes from
//!
//! A track counts as being in the vehicle's path when its bounding box
//! overlaps a corridor of half-width [`CORRIDOR_HALF_WIDTH_M`] about the
//! sensor's x axis. That half-width is **half the width of the recording
//! vehicle**, a VW Passat B6 Variant at 1.82 m — not a lane width, not a
//! margin, and not a number chosen by looking at how many objects it selects.
//! The claim it supports is exactly "this object is in the space the car would
//! drive through", and widening it to a lane would make the claim weaker while
//! making the demo look busier.
//!
//! The test is on the **box**, not the centroid. A van whose centroid is 1.2 m
//! off the axis still has a metre of itself in the corridor, and a chain that
//! answered "nothing ahead" in that situation would be wrong in the one
//! direction that matters.
//!
//! # What the distance is measured from, and to
//!
//! **From the lidar origin, to the near face of the object's box**
//! (`bbox_min.x`), signed: it is the gap for anything ahead, and it is
//! negative for a box that reaches behind the lidar origin — a car alongside,
//! or one behind — where "the gap ahead" has no meaning and time to contact
//! is 0 (below).
//!
//! The near face rather than the centroid, because "the nearest obstacle is
//! 12.3 m away" is a statement about the gap, and a centroid puts a car's
//! reported distance half a car-length too far.
//!
//! ## The distance is to the BOX, and the box is not the corridor
//!
//! **Read this before quoting the number.** The two tests above are two
//! independent projections of one axis-aligned box. `in_path` asks whether
//! the box's *y* interval overlaps the corridor; `distance_m` takes the box's
//! *x* minimum. **Nothing checks that the material at that *x* is the material
//! inside the corridor**, and on a wide structure it is routinely not: a fence
//! or a row of parked cars that clips the corridor at 15 m reports the near
//! face of the whole run of it, which may be metres nearer and metres to one
//! side. An axis-aligned box cannot distinguish the two, so this is a property
//! of the schema and not a tuning question — see the note on
//! [`StateAnswer::distance_m`] for what it would take to fix.
//!
//! An earlier version of this file claimed the opposite — that filtering the
//! raw sweep to the corridor and taking the smallest forward coordinate "must
//! agree to within one voxel edge", and that
//! [`crate::track`]'s real-drive harness checked it. **It does not, and the
//! harness never did.** What the harness measures, over the 144 sweeps of
//! drive_0005 that produce an answer, is printed in full by
//! `the_answer_agrees_with_the_raw_cloud_and_not_with_a_displaced_control`:
//! the median difference is 0.000 m and the p90 is +6.635 m, the answer is
//! within 0.5 m — two and a half voxel edges — on 68.8 % of sweeps, and the
//! worst sweep the harness now prints by name is further out than that again.
//! It asserts a median under 1 m and agreement on more than half the sweeps,
//! which is what the evidence supports. The claim it replaced was checkable,
//! false, and had no test.
//!
//! **From the lidar origin, and not from the bumper.** The velodyne sits on
//! the roof, some way behind the front of the car; that offset is not in any
//! calibration file this project reads, so it is not applied rather than
//! guessed. Every distance here is therefore larger than the gap to the bumper
//! by a fixed, unmeasured amount of order a metre.
//!
//! # Closing speed: uncompensated, and that is the correct choice here
//!
//! `closing = -vx` of the track, in the sensor frame, with no ego-motion
//! correction — because the pipeline does not replay OXTS.
//!
//! For most questions that is a defect. For **this** question it is the right
//! quantity and a compensated one would be the wrong one: a parked car ahead
//! of a vehicle doing 5 m/s is closing at 5 m/s, and that is what a driver,
//! or a brake, needs to know. The number answers "how fast is the gap
//! shrinking", not "is that object moving", and this module cannot answer the
//! second.
//!
//! Its **noise** is inherited from [`crate::track`] and is `0.20 m / age_s`:
//! a voxel centroid carries 0.20 m of quantisation and the velocity is
//! measured over the track's whole life, so a track first seen one sweep ago
//! (0.103 s) has a closing speed good to about ±2 m/s and one tracked for
//! 0.72 s to about ±0.3 m/s. The age travels in every record, in seconds on
//! the sensor clock, so that is readable rather than assumed.
//!
//! Time to contact is `distance / closing` only when **both** are positive —
//! something ahead, and the gap shrinking — and 0 otherwise, rather than a
//! negative or infinite number: with every track in the answer, most records
//! are beside or behind the vehicle or not closing at all.
//!
//! # Only a track observed in this sweep can be the nearest in the path
//!
//! `misses == 0`. [`crate::track`] lets a track survive one sweep without an
//! observation so a re-split object keeps its identity, but a coasted track's
//! position is its **last observed** one, and flagging "the nearest obstacle
//! is at 12.3 m" from a position one sweep old would be reporting a
//! measurement that was not made. A coasted track is still a record — with
//! its `since_seen_s` saying how old its position is — and can still be
//! `in_path` as last observed; it cannot be the one flagged.
//!
//! # When there is nothing ahead
//!
//! [`StateAnswer::has_object`] is `false` and no record is flagged. It is a
//! separate column rather than a sentinel distance, because every sentinel
//! this project has tried — 0 for "no arrival_seq", `vec![]` for "no parent" —
//! turned into two spellings of the same thing that something eventually read
//! as a measurement.

use std::sync::Arc;

use arrow::array::{
    Array, ArrayRef, AsArray, BooleanArray, FixedSizeListArray, Float32Array, Int64Array,
    LargeListArray, RecordBatch, StringArray, UInt32Array,
};
use arrow::buffer::{Buffer, MutableBuffer, OffsetBuffer, ScalarBuffer};
use arrow::datatypes::{DataType, Field, FieldRef, Float32Type, Int64Type, Schema, UInt32Type};

use crate::calib::ImageBox;
use crate::camdet::class_name;
use crate::fuse::{
    build_camera_only, camera_only_data_type, POPULATION_FUSED, POPULATION_LIDAR_ONLY,
    POPULATION_UNFUSED,
};
use crate::track::{
    track_age_s, track_detection, track_image_box, track_population, track_since_seen_s,
    TRACK_LANES,
};
use crate::velo::SweepError;

/// f32 lanes in one answer record. See [`OBJECT_BYTES`].
pub const OBJECT_VALUES: i32 = 17;

/// [`OBJECT_VALUES`] as a `usize`.
pub const OBJECT_LANES: usize = OBJECT_VALUES as usize;

/// Bytes in one answer record — one per track — as 17 f32 lanes:
///
/// | lane | field |
/// |---|---|
/// | 0 | `object_id` — the track's stable id, so an object can be followed across sweeps |
/// | 1 | `distance_m` — lidar origin to the near face of the box, along x, signed; see the module docs |
/// | 2 | `closing_mps` — positive is approaching; uncompensated, see the module docs |
/// | 3 | `time_to_contact_s` — `distance / closing` when both are positive, else 0 |
/// | 4 | `bearing_deg` — of the centroid, 0 straight ahead, positive to the left |
/// | 5 | `age_s` — last seen minus first seen, seconds on the sensor clock; the closing speed's noise is `0.20 m / age_s` |
/// | 6 | `since_seen_s` — seconds since the track was last seen: 0 on a fresh observation, one sweep on a coasted one. How old lanes 1-4 and 8-11 are |
/// | 7 | `in_frame` — 1 when the object projects into the paired camera frame, else 0 |
/// | 8-11 | `image_box` x0, y0, x1, y1 — where it is in that frame, pixels; all zero when `in_frame` is 0 |
/// | 12 | `in_path` — 1 when its box, as last observed, is ahead and overlaps the corridor |
/// | 13 | `nearest_in_path` — 1 on exactly one record when anything fresh is in the path: the nearest such, which is the one object this stage used to answer with |
/// | 14 | `class_id` — the class of the camera detection fused with this track, an index into [`crate::camdet::COCO_CLASSES`]; 0 when not fused |
/// | 15 | `confidence` — that detection's score, above the detector's 0.3; 0 when not fused |
/// | 16 | `population` — [`crate::fuse::POPULATION_FUSED`], [`crate::fuse::POPULATION_LIDAR_ONLY`] or [`crate::fuse::POPULATION_UNFUSED`]; see [`crate::fuse`] |
///
/// **Lanes 14-15 of a record that was not fused are zero, and that is the one
/// exception to this module's rule against sentinels.** Class 0 is also a real
/// class ("person"), so the two lanes are never read alone: [`Object::class`]
/// is their one reader and it reads nothing while the confidence is 0, which a
/// fused record's never is; lane 16 says the same thing in words, and
/// [`record_is_consistent`] checks that the two agree. The lanes were written
/// as zeros before the detector existed so that filling them would not change
/// this schema, and it did not.
///
/// `in_frame` restates what the four zeros of the box already say. It is here
/// because a consumer filtering "what the camera can see" should not have to
/// know that convention, and a second spelling of one fact is only safe if the
/// two cannot disagree: [`record_is_consistent`] checks them, and the far end
/// of the chain counts every record where they do.
pub const OBJECT_BYTES: usize = 68;

/// Value of the `state_format` column: what the 17 lanes mean.
///
/// `i1` id, `d1` distance, `c1` closing, `t1` time to contact, `b1` bearing,
/// `a1` age, `s1` since seen, `f1` in frame, `r4` image rectangle, `p1` in
/// path, `n1` nearest in path, `k1` class, `q1` confidence, `o1` population.
/// It changed when the answer stopped being one object of nine lanes.
pub const STATE_FORMAT: &str = "state_i1d1c1t1b1a1s1f1r4p1n1k1q1o1_f32le";

/// Width of the KITTI recording vehicle (VW Passat B6 Variant), metres.
pub const VEHICLE_WIDTH_M: f32 = 1.82;

/// Half-width of the corridor a track must overlap to count as being in the
/// vehicle's path, metres.
///
/// Half the vehicle's own width and nothing else. See the module docs for why
/// it is not a lane width.
pub const CORRIDOR_HALF_WIDTH_M: f32 = VEHICLE_WIDTH_M / 2.0;

/// Whether a track's box, as last observed, is ahead of the sensor and
/// overlaps the corridor.
///
/// Both tests written POSITIVELY, so a NaN coordinate excludes the track
/// instead of slipping through a negated float comparison. Box overlap, not
/// centroid containment: see the module docs.
fn in_path(t: &[f32; TRACK_LANES]) -> bool {
    let (xmin, ymin, ymax) = (t[9], t[10], t[13]);
    let ahead = xmin > 0.0;
    let in_corridor = ymin <= CORRIDOR_HALF_WIDTH_M && ymax >= -CORRIDOR_HALF_WIDTH_M;
    ahead && in_corridor
}

/// `distance / closing` when something is ahead and the gap is shrinking,
/// else 0.
fn time_to_contact(distance_m: f32, closing_mps: f32) -> f32 {
    if distance_m > 0.0 && closing_mps > 0.0 {
        distance_m / closing_mps
    } else {
        0.0
    }
}

/// One record of the answer: one tracked object.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Object {
    /// The track's id.
    pub id: u32,
    /// Lidar origin to the near face of its box along x, metres, signed.
    pub distance_m: f32,
    /// Positive when the gap is shrinking, m/s.
    pub closing_mps: f32,
    /// `distance / closing` when both are positive, else 0.
    pub ttc_s: f32,
    /// Bearing of its centroid, degrees, 0 straight ahead.
    pub bearing_deg: f32,
    /// Last seen minus first seen, seconds on the sensor clock.
    pub age_s: f32,
    /// Seconds since it was last seen: 0 when seen this sweep.
    pub since_seen_s: f32,
    /// Where it is in the paired camera frame, when it is in frame at all.
    pub image: Option<ImageBox>,
    /// Its box, as last observed, is ahead and overlaps the corridor.
    pub in_path: bool,
    /// The one record the old single answer named: the nearest FRESH object
    /// in the path.
    pub nearest_in_path: bool,
    /// `(class_id, confidence)` of the camera detection fused with this
    /// track, or `None` when it was not fused; see [`OBJECT_BYTES`].
    pub class: Option<(u32, f32)>,
    /// The population code ([`crate::fuse`]): fused, lidar-only, or unfused.
    pub population: u32,
}

impl Object {
    /// The record for one track of [`crate::track::track_rows`], not flagged.
    pub fn of_track(t: &[f32; TRACK_LANES]) -> Object {
        let distance_m = t[9];
        // `-vx`: positive when the object is coming towards the sensor.
        let closing_mps = -t[3];
        Object {
            id: t[6] as u32,
            distance_m,
            closing_mps,
            ttc_s: time_to_contact(distance_m, closing_mps),
            bearing_deg: t[1].atan2(t[0]).to_degrees(),
            age_s: track_age_s(t),
            since_seen_s: track_since_seen_s(t),
            image: track_image_box(t),
            in_path: in_path(t),
            nearest_in_path: false,
            class: track_detection(t).map(|(_, _, class, confidence)| (class, confidence)),
            population: track_population(t),
        }
    }

    /// The 17 lanes, in [`OBJECT_BYTES`]' order.
    pub fn lanes(&self) -> [f32; OBJECT_LANES] {
        let flag = |b: bool| if b { 1.0 } else { 0.0 };
        let mut r = [0.0f32; OBJECT_LANES];
        // Exact below 2^24; see `crate::track::TRACK_BYTES`.
        r[0] = self.id as f32;
        r[1] = self.distance_m;
        r[2] = self.closing_mps;
        r[3] = self.ttc_s;
        r[4] = self.bearing_deg;
        r[5] = self.age_s;
        r[6] = self.since_seen_s;
        if let Some(b) = self.image {
            r[7] = 1.0;
            r[8] = b.x0;
            r[9] = b.y0;
            r[10] = b.x1;
            r[11] = b.y1;
        }
        r[12] = flag(self.in_path);
        r[13] = flag(self.nearest_in_path);
        if let Some((class, confidence)) = self.class {
            r[14] = class as f32;
            r[15] = confidence;
        }
        r[16] = self.population as f32;
        r
    }

    /// One record read back out of its lanes.
    ///
    /// The box is read the way [`crate::track::track_image_box`] reads it —
    /// four zeros are "not in frame" — and lane 7 is NOT consulted here:
    /// whether the two agree is [`record_is_consistent`]'s question, asked
    /// separately so a disagreement is counted rather than silently resolved.
    pub fn from_lanes(r: &[f32; OBJECT_LANES]) -> Object {
        let b = ImageBox {
            x0: r[8],
            y0: r[9],
            x1: r[10],
            y1: r[11],
        };
        Object {
            id: r[0] as u32,
            distance_m: r[1],
            closing_mps: r[2],
            ttc_s: r[3],
            bearing_deg: r[4],
            age_s: r[5],
            since_seen_s: r[6],
            image: (b.w() > 0.0 && b.h() > 0.0).then_some(b),
            in_path: r[12] == 1.0,
            nearest_in_path: r[13] == 1.0,
            class: (r[15] > 0.0).then_some((r[14] as u32, r[15])),
            population: r[16] as u32,
        }
    }
}

/// Whether one record's redundant lanes agree with each other: `in_frame`
/// (lane 7) with the box, a `nearest_in_path` flag only on a fresh record
/// that is in the path, and the camera's lanes with the population: a class
/// exactly on a fused record, and a fused record only where a fusion could
/// have happened -- fresh and in frame.
///
/// The far end of the chain counts the records where this is false, and the
/// run's invariants require that count to be 0.
pub fn record_is_consistent(r: &[f32; OBJECT_LANES]) -> bool {
    let o = Object::from_lanes(r);
    let in_frame_agrees = (r[7] == 1.0) == o.image.is_some() && (r[7] == 0.0 || r[7] == 1.0);
    let flag_is_earned = !o.nearest_in_path || (o.in_path && o.since_seen_s == 0.0);
    let fused = o.population == POPULATION_FUSED;
    let population_known = matches!(
        o.population,
        POPULATION_UNFUSED | POPULATION_FUSED | POPULATION_LIDAR_ONLY
    ) && r[16] == o.population as f32;
    let class_agrees = fused == o.class.is_some();
    let fusion_is_possible = !fused || (o.image.is_some() && o.since_seen_s == 0.0);
    in_frame_agrees && flag_is_earned && population_known && class_agrees && fusion_is_possible
}

/// The nearest tracked object in the vehicle's path, or the fact that there
/// is none: the one record [`Object::nearest_in_path`] flags, summarised.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct StateAnswer {
    /// Whether anything fresh is in the corridor at all.
    pub has_object: bool,
    /// The track's id.
    pub object_id: u32,
    /// Age of that track: last seen minus first seen, seconds on the sensor
    /// clock ([`crate::track::track_age_s`]). The closing speed's noise is
    /// `0.20 m / age_s`.
    ///
    /// It used to be a count of the sweeps the track was matched in, which
    /// understates a track's life by every sweep it coasted through; see
    /// [`crate::track`]'s module docs.
    pub age_s: f32,
    /// Lidar origin to the near face of its box, metres.
    ///
    /// **To the box, not to the corridor.** The box's near face and the box's
    /// overlap with the corridor are two independent projections of one
    /// axis-aligned box, and nothing here checks that the material at the near
    /// face is the material inside the corridor. Fixing it properly means
    /// carrying the corridor-restricted near face down from
    /// [`crate::detect`] — a detection would have to say where its material is
    /// as well as how far it extends, which changes the detection and track
    /// schemas and therefore every measured byte figure in this project. It is
    /// stated rather than quietly worked around. See the module docs.
    pub distance_m: f32,
    /// Positive when the gap is shrinking, m/s.
    pub closing_mps: f32,
    /// `distance / closing` when closing, else 0.
    pub ttc_s: f32,
    /// Bearing of its centroid, degrees, 0 straight ahead.
    pub bearing_deg: f32,
    /// Where it is in the paired camera frame, when it is in frame at all.
    pub image: Option<ImageBox>,
    /// `(class_id, confidence)` from the camera, when the fusion associated
    /// this object with a detection of the paired frame.
    pub class: Option<(u32, f32)>,
    /// Fresh tracks that were in the corridor, of which this is the nearest.
    pub candidates: u32,
}

impl StateAnswer {
    /// The nearest qualifying track of `tracks`, which are
    /// [`crate::track::track_rows`] of one fused sample.
    ///
    /// Qualifying means: observed in this sweep (`misses == 0`), its box in
    /// front of the sensor, and its box overlapping the corridor.
    /// Deterministic on ties, by lowest id — the rows are already in a fixed
    /// order, but "nearest" over floats can tie and iteration order must not
    /// be what decides.
    pub fn of(tracks: &[[f32; TRACK_LANES]]) -> StateAnswer {
        let mut out = StateAnswer::default();
        let mut best: Option<(f32, u32)> = None;
        for t in tracks {
            // Fresh observation only: a coasted track's position is one sweep
            // old and answering from it would report a measurement that was
            // not made.
            if t[8] != 0.0 || !in_path(t) {
                continue;
            }
            out.candidates += 1;
            let o = Object::of_track(t);
            let better = match best {
                None => true,
                Some((d, bid)) => o.distance_m < d || (o.distance_m == d && o.id < bid),
            };
            if better {
                best = Some((o.distance_m, o.id));
                out = StateAnswer {
                    candidates: out.candidates,
                    ..StateAnswer::from_object(&o)
                };
            }
        }
        out
    }

    /// The answer a flagged record gives, `candidates` aside.
    fn from_object(o: &Object) -> StateAnswer {
        StateAnswer {
            has_object: true,
            object_id: o.id,
            age_s: o.age_s,
            distance_m: o.distance_m,
            closing_mps: o.closing_mps,
            ttc_s: o.ttc_s,
            bearing_deg: o.bearing_deg,
            image: o.image,
            class: o.class,
            candidates: 0,
        }
    }

    /// The answer as one line of plain English — what the whole pipeline
    /// exists to produce.
    pub fn line(&self) -> String {
        if !self.has_object {
            return "nothing in the vehicle's path".to_string();
        }
        let ttc = if self.ttc_s > 0.0 {
            format!("{:.1} s to contact", self.ttc_s)
        } else {
            "not closing".to_string()
        };
        let img = match self.image {
            Some(b) => format!("image ({:.0},{:.0})-({:.0},{:.0})", b.x0, b.y0, b.x1, b.y1),
            None => "not in the camera frame".to_string(),
        };
        let what = match self.class {
            Some((class, confidence)) => format!(" ({} {confidence:.2})", class_name(class)),
            None => String::new(),
        };
        format!(
            "object {}{what} at {:.2} m, {:+.2} deg, closing {:.2} m/s, {ttc}, tracked for {:.2} s, {img}",
            self.object_id, self.distance_m, self.bearing_deg, self.closing_mps, self.age_s
        )
    }
}

/// Arrow field of one answer lane.
fn object_value_field() -> FieldRef {
    Arc::new(Field::new("item", DataType::Float32, false))
}

/// Arrow field of one record: `FixedSizeList<Float32, 17>`.
fn object_field() -> FieldRef {
    Arc::new(Field::new(
        "item",
        DataType::FixedSizeList(object_value_field(), OBJECT_VALUES),
        false,
    ))
}

/// `state_format Utf8, tov_trigger_ns i64, source_sweep_seq i64,
/// has_object bool, candidates u32, in_frame_count u32, source_tracks u32,
/// source_detections u32, source_voxel_count u32, source_point_count u32,
/// cam_seq i64, pair_outcome Utf8,
/// objects LargeList<FixedSizeList<Float32, 17>>, pair_age_ns i64,
/// fused_count u32, camera_only LargeList<FixedSizeList<Float32, 7>>`.
///
/// One row per sweep, as every other link of the chain, and one record per
/// track in `objects`, the way [`crate::track::track_schema`] carries its
/// tracks: one data buffer, so the address-equality proof that has followed
/// every other link follows this one too.
///
/// `has_object` says whether a record is flagged `nearest_in_path`;
/// `candidates` is how many fresh records were in the path for it to be
/// chosen from; `in_frame_count` how many records are in the camera frame;
/// `source_tracks` how many tracks arrived, which is how many records there
/// are. `source_point_count` alone is what lets the far end of the run state
/// the whole chain's ratio from the sample in its hand.
///
/// `pair_outcome`, `cam_seq` and `pair_age_ns` are carried through from
/// [`crate::track::track_schema`] unchanged, so a bad answer can be attributed
/// to a missing or stale camera frame **at the point where the answer is
/// read**, without joining back up the chain. `fused_count` is how many
/// records carry a camera class, written beside the records it counts so the
/// far end can check the two against each other; `camera_only` is the track
/// batch's own column, shared rather than copied: the paired frame's
/// detections that no track matched ([`crate::fuse::CAMERA_ONLY_BYTES`]).
pub fn state_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("state_format", DataType::Utf8, false),
        Field::new("tov_trigger_ns", DataType::Int64, false),
        Field::new("source_sweep_seq", DataType::Int64, false),
        Field::new("has_object", DataType::Boolean, false),
        Field::new("candidates", DataType::UInt32, false),
        Field::new("in_frame_count", DataType::UInt32, false),
        Field::new("source_tracks", DataType::UInt32, false),
        Field::new("source_detections", DataType::UInt32, false),
        Field::new("source_voxel_count", DataType::UInt32, false),
        Field::new("source_point_count", DataType::UInt32, false),
        Field::new("cam_seq", DataType::Int64, false),
        Field::new("pair_outcome", DataType::Utf8, false),
        Field::new("objects", DataType::LargeList(object_field()), false),
        Field::new("pair_age_ns", DataType::Int64, false),
        Field::new("fused_count", DataType::UInt32, false),
        Field::new("camera_only", camera_only_data_type(), false),
    ]))
}

/// Everything the state batch carries besides the records themselves.
#[derive(Clone, Copy, Debug)]
pub struct StateMeta {
    /// The sweep's trigger, inherited unchanged down the whole chain.
    pub trigger_ns: i64,
    /// The velodyne sweep this answer is about.
    pub sweep_seq: i64,
    /// Detections behind the tracks.
    pub source_detections: u32,
    /// Voxels behind those detections.
    pub source_voxels: u32,
    /// Raw laser returns behind everything.
    pub source_points: u32,
    /// The camera frame the sweep was paired with, or -1.
    pub cam_seq: i64,
    /// How that pairing went, as `Pairing::name` spells it.
    pub pair_outcome: &'static str,
    /// The camera frame's instant minus the sweep's trigger, ns: about +10.5
    /// ms on a clean pair, about -92.8 ms on a frame one period stale.
    pub pair_age_ns: i64,
}

/// Builds the one-row answer batch: one record per track in `tracks`, the
/// one `answer` names flagged, wrapped **without copying** the buffer.
///
/// `answer` is [`StateAnswer::of`] the same `tracks`; it is taken rather than
/// recomputed so the caller can measure choosing and building apart. The
/// record flagged is the one with its id, and ids are never reissued within a
/// run, so there is one.
///
/// `camera_only` is the track batch's own column, handed on as it is -- the
/// same buffer, not a copy -- or `None` for a sample that carried none, which
/// is written as an empty list.
///
/// Returns the batch and its `storage_id`, so the address-equality proof that
/// has followed every other link of this chain follows the last one too.
pub fn build_state_batch(
    tracks: &[[f32; TRACK_LANES]],
    answer: &StateAnswer,
    meta: &StateMeta,
    camera_only: Option<ArrayRef>,
    schema: &Arc<Schema>,
) -> Result<(RecordBatch, usize), SweepError> {
    // The one allocation for the records, and exactly their size.
    let mut buf = MutableBuffer::from_len_zeroed(tracks.len() * OBJECT_BYTES);
    let addr = buf.as_slice().as_ptr() as usize;
    if !addr.is_multiple_of(std::mem::align_of::<f32>()) {
        return Err(SweepError::Misaligned { addr });
    }
    let mut in_frame = 0u32;
    let mut fused = 0u32;
    let mut flagged = false;
    {
        let out = buf.typed_data_mut::<f32>();
        for (w, t) in tracks.iter().enumerate() {
            let mut o = Object::of_track(t);
            if answer.has_object && !flagged && o.id == answer.object_id && t[8] == 0.0 {
                o.nearest_in_path = true;
                flagged = true;
            }
            in_frame += u32::from(o.image.is_some());
            fused += u32::from(o.population == POPULATION_FUSED);
            out[w * OBJECT_LANES..][..OBJECT_LANES].copy_from_slice(&o.lanes());
        }
    }
    let buf = Buffer::from(buf);
    let storage_id = buf.as_ptr() as usize;
    if !storage_id.is_multiple_of(std::mem::align_of::<f32>()) {
        return Err(SweepError::Misaligned { addr: storage_id });
    }
    let n = tracks.len();
    let values = Float32Array::new(ScalarBuffer::<f32>::from(buf), None);
    let rec = FixedSizeListArray::try_new(
        object_value_field(),
        OBJECT_VALUES,
        Arc::new(values) as ArrayRef,
        None,
    )?;
    let offsets = OffsetBuffer::<i64>::new(ScalarBuffer::<i64>::from(vec![0i64, n as i64]));
    let objects = LargeListArray::try_new(object_field(), offsets, Arc::new(rec), None)?;
    let camera_only = match camera_only {
        Some(c) => c,
        None => build_camera_only(None, &[])?,
    };
    let columns: Vec<ArrayRef> = vec![
        Arc::new(StringArray::from(vec![STATE_FORMAT])),
        Arc::new(Int64Array::from(vec![meta.trigger_ns])),
        Arc::new(Int64Array::from(vec![meta.sweep_seq])),
        // What was WRITTEN, not what was asked for: if the named record were
        // somehow absent, the column says so and the far end counts it.
        Arc::new(BooleanArray::from(vec![flagged])),
        Arc::new(UInt32Array::from(vec![answer.candidates])),
        Arc::new(UInt32Array::from(vec![in_frame])),
        Arc::new(UInt32Array::from(vec![n as u32])),
        Arc::new(UInt32Array::from(vec![meta.source_detections])),
        Arc::new(UInt32Array::from(vec![meta.source_voxels])),
        Arc::new(UInt32Array::from(vec![meta.source_points])),
        Arc::new(Int64Array::from(vec![meta.cam_seq])),
        Arc::new(StringArray::from(vec![meta.pair_outcome])),
        Arc::new(objects),
        Arc::new(Int64Array::from(vec![meta.pair_age_ns])),
        Arc::new(UInt32Array::from(vec![fused])),
        camera_only,
    ];
    Ok((
        RecordBatch::try_new(Arc::clone(schema), columns).map_err(SweepError::Arrow)?,
        storage_id,
    ))
}

/// The interleaved answer lanes of a state batch, shared with the buffer the
/// stage built. `chunks_exact(17)` gives one record each.
pub fn objects_f32(batch: &RecordBatch) -> Option<&[f32]> {
    Some(
        batch
            .column_by_name("objects")?
            .as_list_opt::<i64>()?
            .values()
            .as_fixed_size_list_opt()?
            .values()
            .as_primitive_opt::<Float32Type>()?
            .values(),
    )
}

/// The records of a state batch, or an empty slice.
pub fn object_rows(batch: &RecordBatch) -> &[[f32; OBJECT_LANES]] {
    let lanes: &[f32] = objects_f32(batch).unwrap_or(&[]);
    let (rows, _) = lanes.as_chunks::<OBJECT_LANES>();
    rows
}

/// Address of the shared answer buffer: the zero-copy proof at the far end of
/// the chain.
pub fn state_storage_id(batch: &RecordBatch) -> Option<usize> {
    objects_f32(batch).map(|v| v.as_ptr() as usize)
}

fn u32_col(batch: &RecordBatch, name: &str) -> Option<u32> {
    let c = batch
        .column_by_name(name)?
        .as_primitive_opt::<UInt32Type>()?;
    (!c.is_empty()).then(|| c.value(0))
}

/// The answer, read back out of a state batch: the flagged record,
/// summarised, or "nothing in the path".
///
/// `None` when the batch is not an answer, or when `has_object` claims a
/// flagged record the payload does not carry — two spellings that disagree
/// are reported as unreadable rather than resolved in favour of either.
/// Allocates nothing: it walks the shared records in place.
pub fn read_answer(batch: &RecordBatch) -> Option<StateAnswer> {
    let has_object = has_object(batch)?;
    let candidates = u32_col(batch, "candidates")?;
    if !has_object {
        return Some(StateAnswer {
            candidates,
            ..StateAnswer::default()
        });
    }
    let o = object_rows(batch)
        .iter()
        .map(Object::from_lanes)
        .find(|o| o.nearest_in_path)?;
    Some(StateAnswer {
        candidates,
        ..StateAnswer::from_object(&o)
    })
}

/// Whether the batch says a record is flagged `nearest_in_path`: the column
/// as written, read without consulting the records.
pub fn has_object(batch: &RecordBatch) -> Option<bool> {
    batch
        .column_by_name("has_object")?
        .as_any()
        .downcast_ref::<BooleanArray>()
        .filter(|c| !c.is_empty())
        .map(|c| c.value(0))
}

/// Fresh records in the vehicle's path: the population the flagged record
/// was chosen from.
pub fn candidates(batch: &RecordBatch) -> Option<u32> {
    u32_col(batch, "candidates")
}

/// Records in the batch that are in the camera frame.
pub fn in_frame_count(batch: &RecordBatch) -> Option<u32> {
    u32_col(batch, "in_frame_count")
}

/// Raw laser returns behind this answer, for the end-to-end ratio.
pub fn source_point_count(batch: &RecordBatch) -> Option<u32> {
    u32_col(batch, "source_point_count")
}

/// Tracks this answer was built from: one record each.
pub fn source_tracks(batch: &RecordBatch) -> Option<u32> {
    u32_col(batch, "source_tracks")
}

/// What the 17 lanes of a record MEAN, read back out of the batch.
pub fn state_format(batch: &RecordBatch) -> Option<&str> {
    let c = batch
        .column_by_name("state_format")?
        .as_any()
        .downcast_ref::<StringArray>()?;
    (!c.is_empty()).then(|| c.value(0))
}

/// How the camera pairing went, carried through so a bad answer can be
/// attributed where it is read.
pub fn pair_outcome(batch: &RecordBatch) -> Option<&str> {
    let c = batch
        .column_by_name("pair_outcome")?
        .as_any()
        .downcast_ref::<StringArray>()?;
    (!c.is_empty()).then(|| c.value(0))
}

/// The paired camera frame's instant minus the sweep's trigger, ns: how old
/// the camera half of this answer was. Negative for a stale pair.
pub fn pair_age_ns(batch: &RecordBatch) -> Option<i64> {
    let c = batch
        .column_by_name("pair_age_ns")?
        .as_primitive_opt::<Int64Type>()?;
    (!c.is_empty()).then(|| c.value(0))
}

/// Records carrying a camera class, as the batch's own column says.
pub fn fused_count(batch: &RecordBatch) -> Option<u32> {
    u32_col(batch, "fused_count")
}

/// The camera frame this answer's sweep was paired with, or -1.
pub fn cam_seq(batch: &RecordBatch) -> Option<i64> {
    let c = batch
        .column_by_name("cam_seq")?
        .as_primitive_opt::<Int64Type>()?;
    (!c.is_empty()).then(|| c.value(0))
}

/// The sweep this answer is about.
pub fn state_sweep_seq(batch: &RecordBatch) -> Option<i64> {
    let c = batch
        .column_by_name("source_sweep_seq")?
        .as_primitive_opt::<Int64Type>()?;
    (!c.is_empty()).then(|| c.value(0))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    /// One track record: centroid, velocity, id, observations, misses, box,
    /// and an age of one 0.1 s interval per observation after the first.
    #[allow(clippy::too_many_arguments)]
    fn trk(
        id: f32,
        observations: f32,
        misses: f32,
        pos: [f32; 3],
        vel: [f32; 3],
        lo: [f32; 3],
        hi: [f32; 3],
    ) -> [f32; TRACK_LANES] {
        let mut t = [0.0f32; TRACK_LANES];
        t[0..3].copy_from_slice(&pos);
        t[3..6].copy_from_slice(&vel);
        t[6] = id;
        t[7] = observations;
        t[8] = misses;
        t[9..12].copy_from_slice(&lo);
        t[12..15].copy_from_slice(&hi);
        t[19] = (observations - 1.0) * 0.1;
        t[20] = misses * 0.1;
        t
    }

    fn car(id: f32, x: f32, y: f32, vx: f32) -> [f32; TRACK_LANES] {
        trk(
            id,
            5.0,
            0.0,
            [x + 2.0, y, -1.0],
            [vx, 0.0, 0.0],
            [x, y - 0.9, -1.7],
            [x + 4.0, y + 0.9, 0.0],
        )
    }

    fn with_image(mut t: [f32; TRACK_LANES]) -> [f32; TRACK_LANES] {
        t[15] = 500.0;
        t[16] = 150.0;
        t[17] = 700.0;
        t[18] = 300.0;
        t
    }

    fn meta() -> StateMeta {
        StateMeta {
            trigger_ns: 1_000,
            sweep_seq: 7,
            source_detections: 177,
            source_voxels: 30_754,
            source_points: 121_732,
            cam_seq: 7,
            pair_outcome: "paired",
            pair_age_ns: 10_503_000,
        }
    }

    fn build(tracks: &[[f32; TRACK_LANES]]) -> RecordBatch {
        let a = StateAnswer::of(tracks);
        build_state_batch(tracks, &a, &meta(), None, &state_schema())
            .unwrap()
            .0
    }

    /// A track the fusion associated with detection `det` of the paired
    /// frame, of `class` at `confidence`.
    fn fused(
        mut t: [f32; TRACK_LANES],
        det: f32,
        class: f32,
        confidence: f32,
    ) -> [f32; TRACK_LANES] {
        use crate::track::{
            LANE_CLASS_ID, LANE_CONFIDENCE, LANE_DET_INDEX, LANE_FUSE_IOU, LANE_POPULATION,
        };
        t[LANE_DET_INDEX] = det;
        t[LANE_FUSE_IOU] = 0.8;
        t[LANE_CLASS_ID] = class;
        t[LANE_CONFIDENCE] = confidence;
        t[LANE_POPULATION] = POPULATION_FUSED as f32;
        t
    }

    #[test]
    fn the_nearest_object_in_the_corridor_wins() {
        let tracks = [car(1.0, 20.0, 0.0, -3.0), car(2.0, 12.0, 0.0, -5.0)];
        let a = StateAnswer::of(&tracks);
        assert!(a.has_object);
        assert_eq!(a.object_id, 2);
        assert_eq!(a.distance_m, 12.0);
        assert_eq!(a.closing_mps, 5.0);
        assert!((a.ttc_s - 2.4).abs() < 1e-5, "ttc = {}", a.ttc_s);
        assert_eq!(a.candidates, 2);
    }

    /// A nearer object OUTSIDE the corridor must not win. Without this the
    /// corridor is decorative.
    #[test]
    fn an_object_beside_the_path_is_not_the_answer() {
        let tracks = [car(1.0, 20.0, 0.0, -3.0), car(2.0, 8.0, 6.0, -5.0)];
        let a = StateAnswer::of(&tracks);
        assert_eq!(
            a.object_id, 1,
            "an object 6 m to the side became the answer"
        );
        assert_eq!(a.candidates, 1);
    }

    /// The test is on the BOX, not the centroid: a wide object whose centre is
    /// outside the corridor still blocks it.
    #[test]
    fn a_wide_object_overlapping_the_corridor_counts() {
        // Centroid at y = +1.6, outside the 0.91 m half-width; box reaches
        // y = +0.4, inside it.
        let wide = trk(
            9.0,
            4.0,
            0.0,
            [11.0, 1.6, -1.0],
            [-2.0, 0.0, 0.0],
            [10.0, 0.4, -1.7],
            [12.0, 2.8, 0.0],
        );
        let a = StateAnswer::of(&[wide]);
        assert!(a.has_object, "the box overlaps the corridor and was missed");
        assert_eq!(a.distance_m, 10.0);
    }

    /// A coasted track cannot be the answer: its position is one sweep old.
    #[test]
    fn a_coasted_track_is_not_the_answer() {
        let mut stale = car(1.0, 8.0, 0.0, -5.0);
        stale[8] = 1.0;
        stale[20] = 0.1;
        let fresh = car(2.0, 20.0, 0.0, -3.0);
        let a = StateAnswer::of(&[stale, fresh]);
        assert_eq!(a.object_id, 2, "an unobserved position became the answer");
        assert_eq!(a.candidates, 1);
        // It is still a record, in the path as last observed, not flagged,
        // and saying how old its position is.
        let b = build(&[stale, fresh]);
        let rows: Vec<Object> = object_rows(&b).iter().map(Object::from_lanes).collect();
        let coasted = rows.iter().find(|o| o.id == 1).expect("the coasted record");
        assert!(coasted.in_path && !coasted.nearest_in_path);
        assert_eq!(coasted.since_seen_s, 0.1);
        // Positive control: the same pair with a fresh near track answers 1.
        let mut ok = stale;
        ok[8] = 0.0;
        ok[20] = 0.0;
        assert_eq!(StateAnswer::of(&[ok, fresh]).object_id, 1);
    }

    /// Nothing ahead is an answer, not an absence, and it is flagged rather
    /// than encoded as a distance -- and every track is still carried.
    #[test]
    fn an_empty_corridor_is_an_answer_and_still_carries_every_track() {
        let beside = car(1.0, 10.0, 8.0, -3.0);
        let a = StateAnswer::of(&[beside]);
        assert!(!a.has_object);
        assert_eq!(a.candidates, 0);
        assert_eq!(a.line(), "nothing in the vehicle's path");
        let b = build(&[beside]);
        let back = read_answer(&b).unwrap();
        assert!(!back.has_object);
        assert_eq!(back.distance_m, 0.0);
        let rows = object_rows(&b);
        assert_eq!(rows.len(), 1, "the object beside the path was dropped");
        let o = Object::from_lanes(&rows[0]);
        assert_eq!((o.id, o.in_path, o.nearest_in_path), (1, false, false));
        assert_eq!(o.distance_m, 10.0);
    }

    /// An object moving away has no time to contact, and the field says 0
    /// rather than a negative or infinite one.
    #[test]
    fn a_receding_object_has_no_time_to_contact() {
        let a = StateAnswer::of(&[car(1.0, 10.0, 0.0, 4.0)]);
        assert!(a.has_object);
        assert_eq!(a.closing_mps, -4.0);
        assert_eq!(a.ttc_s, 0.0);
        assert!(a.line().contains("not closing"), "{}", a.line());
    }

    /// Something alongside or behind that is "closing" has no time to
    /// contact either: its signed distance is not a gap ahead.
    #[test]
    fn an_object_behind_has_no_time_to_contact() {
        let behind = car(3.0, -12.0, 0.0, -4.0);
        let o = Object::of_track(&behind);
        assert!(o.distance_m < 0.0);
        assert!(o.closing_mps > 0.0);
        assert_eq!(o.ttc_s, 0.0);
        assert!(
            !o.in_path,
            "a box behind the sensor is not in the path ahead"
        );
    }

    /// Every track becomes exactly one record, in the order it arrived, and
    /// exactly one of them carries the flag -- the one `StateAnswer::of`
    /// names.
    #[test]
    fn every_track_is_a_record_and_one_is_flagged() {
        let tracks = [
            car(1.0, 20.0, 0.0, -3.0),
            with_image(car(2.0, 12.0, 0.0, -5.0)),
            car(3.0, 8.0, 6.0, -5.0),
            car(4.0, -6.0, 0.0, 1.0),
        ];
        let b = build(&tracks);
        assert_eq!(state_format(&b), Some(STATE_FORMAT));
        assert_eq!(source_tracks(&b), Some(4));
        let rows = object_rows(&b);
        assert_eq!(rows.len(), tracks.len());
        let ids: Vec<u32> = rows.iter().map(|r| Object::from_lanes(r).id).collect();
        assert_eq!(ids, [1, 2, 3, 4], "records are not in track order");
        let flagged: Vec<u32> = rows
            .iter()
            .map(Object::from_lanes)
            .filter(|o| o.nearest_in_path)
            .map(|o| o.id)
            .collect();
        assert_eq!(flagged, [StateAnswer::of(&tracks).object_id]);
        for r in rows {
            assert!(record_is_consistent(r), "{r:?}");
        }
        // The flagged record reads back as the answer, in full.
        let back = read_answer(&b).unwrap();
        assert_eq!(back, StateAnswer::of(&tracks));
        assert_eq!(back.image.map(|i| (i.x0, i.y1)), Some((500.0, 300.0)));
    }

    /// Out of frame is carried and flagged, and the flag and the box agree.
    #[test]
    fn an_out_of_frame_object_is_carried_and_flagged() {
        let b = build(&[
            with_image(car(1.0, 12.0, 0.0, -1.0)),
            car(2.0, 9.0, 7.0, -1.0),
        ]);
        assert_eq!(in_frame_count(&b), Some(1));
        let rows = object_rows(&b);
        assert_eq!(rows.len(), 2, "the object out of frame was dropped");
        assert_eq!((rows[0][7], rows[1][7]), (1.0, 0.0), "in_frame lanes");
        assert!(Object::from_lanes(&rows[0]).image.is_some());
        assert!(Object::from_lanes(&rows[1]).image.is_none());
        assert_eq!(&rows[1][8..12], &[0.0; 4]);
        // The negative control on the consistency check: a record whose flag
        // says in frame over a zero box is caught.
        let mut bad = rows[1];
        bad[7] = 1.0;
        assert!(!record_is_consistent(&bad));
        let mut unearned = rows[1];
        unearned[13] = 1.0;
        assert!(!record_is_consistent(&unearned), "a flag off the path");
    }

    /// An unfused track's camera lanes are empty and read as no class -- not
    /// as class 0, which is "person" -- and a fused track's carry its
    /// detection's class and confidence into the answer, with the population
    /// saying which is which.
    #[test]
    fn the_camera_lanes_carry_the_fused_detection_and_nothing_else() {
        let plain = car(1.0, 12.0, 0.0, -1.0);
        let b = build(&[plain]);
        let r = object_rows(&b)[0];
        assert_eq!(&r[14..17], &[0.0, 0.0, POPULATION_UNFUSED as f32]);
        let o = Object::from_lanes(&r);
        assert_eq!(o.class, None);
        assert_eq!(o.population, POPULATION_UNFUSED);
        assert_eq!(fused_count(&b), Some(0));
        // Fused: class 0 at 0.74 is a person, read as one.
        let person = fused(with_image(car(2.0, 9.0, 0.0, -2.0)), 3.0, 0.0, 0.74);
        let b = build(&[plain, person]);
        let rows = object_rows(&b);
        let o = Object::from_lanes(&rows[1]);
        assert_eq!(o.class, Some((0, 0.74)));
        assert_eq!(o.population, POPULATION_FUSED);
        assert_eq!(fused_count(&b), Some(1));
        for r in rows {
            assert!(record_is_consistent(r), "{r:?}");
        }
        // The answer names the class of the object it flags.
        let a = read_answer(&b).unwrap();
        assert_eq!(a.object_id, 2);
        assert_eq!(a.class, Some((0, 0.74)));
        assert!(a.line().contains("(person 0.74)"), "{}", a.line());
    }

    /// A class on a record the population says is not fused, a fused record
    /// with no class, a fused record out of frame or coasting, and a code no
    /// population has are all caught by the far end's check.
    #[test]
    fn a_population_that_disagrees_with_its_lanes_is_caught() {
        let good = fused(with_image(car(1.0, 12.0, 0.0, -1.0)), 0.0, 2.0, 0.9);
        let r = Object::of_track(&good).lanes();
        assert!(record_is_consistent(&r), "the positive control failed");
        let mut classed_lidar_only = r;
        classed_lidar_only[16] = POPULATION_LIDAR_ONLY as f32;
        assert!(!record_is_consistent(&classed_lidar_only));
        let mut unclassed_fused = r;
        unclassed_fused[15] = 0.0;
        assert!(!record_is_consistent(&unclassed_fused));
        let mut out_of_frame = r;
        out_of_frame[7] = 0.0;
        out_of_frame[8..12].copy_from_slice(&[0.0; 4]);
        assert!(!record_is_consistent(&out_of_frame));
        let mut coasted = r;
        coasted[6] = 0.1;
        assert!(!record_is_consistent(&coasted));
        let mut unknown = r;
        unknown[16] = 7.0;
        assert!(!record_is_consistent(&unknown));
        // Camera-only is a population, and never a record's.
        let mut camera_only = r;
        camera_only[16] = crate::fuse::POPULATION_CAMERA_ONLY as f32;
        assert!(!record_is_consistent(&camera_only));
    }

    /// The camera-only detections are the track batch's column, handed on:
    /// the answer's is the same buffer, and an answer without one carries an
    /// empty list rather than none.
    #[test]
    fn the_camera_only_detections_are_handed_on_not_copied() {
        let t = [car(1.0, 12.0, 0.0, -1.0)];
        let a = StateAnswer::of(&t);
        let empty = build(&t);
        assert!(crate::fuse::camera_only_rows(&empty).is_empty());
        assert_eq!(pair_age_ns(&empty), Some(10_503_000));
        let r = crate::track::CamRef {
            seq: 7,
            tov_ns: 0,
            width: 1242,
            height: 375,
        };
        let d = crate::camdet::Det {
            x0: 10.0,
            y0: 20.0,
            x1: 30.0,
            y1: 60.0,
            class_id: 2,
            score: 0.6,
            row: 0,
        };
        let cb = crate::camdet::build_cam_det_batch(
            r,
            &[d],
            &crate::camdet::cam_det_schema(crate::camdet::MODEL_SHA256),
        )
        .unwrap();
        let v = crate::camdet::cam_det_view(&cb).unwrap();
        let co = build_camera_only(Some(&v), &[false]).unwrap();
        let (b, _) =
            build_state_batch(&t, &a, &meta(), Some(Arc::clone(&co)), &state_schema()).unwrap();
        let rows = crate::fuse::camera_only_rows(&b);
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0], [0.0, 2.0, 0.6, 10.0, 20.0, 30.0, 60.0]);
        let src = co
            .as_list::<i64>()
            .values()
            .as_fixed_size_list()
            .values()
            .as_primitive::<Float32Type>()
            .values()
            .as_ptr();
        assert_eq!(
            rows.as_ptr() as *const f32,
            src,
            "the answer copied the column"
        );
    }

    /// Round trip through Arrow of every lane, and the age in seconds.
    #[test]
    fn a_record_round_trips_through_its_lanes() {
        let t = with_image(car(4.0, 12.25, 0.0, -5.0));
        let a = StateAnswer::of(&[t]);
        let (b, sid) = build_state_batch(&[t], &a, &meta(), None, &state_schema()).unwrap();
        assert_eq!(state_storage_id(&b), Some(sid));
        let o = Object::from_lanes(&object_rows(&b)[0]);
        let mut want = Object::of_track(&t);
        want.nearest_in_path = true;
        assert_eq!(o, want);
        assert_eq!(o.age_s, 0.4);
        let back = read_answer(&b).expect("the answer did not round trip");
        assert_eq!(back.object_id, 4);
        assert_eq!(back.distance_m, 12.25);
        assert_eq!(back.closing_mps, 5.0);
        assert_eq!(back.age_s, 0.4);
        assert_eq!(source_point_count(&b), Some(121_732));
        assert_eq!(pair_outcome(&b), Some("paired"));
        assert_eq!(cam_seq(&b), Some(7));
        assert_eq!(state_sweep_seq(&b), Some(7));
    }

    /// 17 lanes at 4 bytes is 68, one record is exactly that, and every
    /// further track costs exactly one record: the answer grows with the
    /// scene now, and its fixed part is provenance.
    #[test]
    fn a_record_is_its_documented_size_and_the_answer_grows_by_one_per_track() {
        assert_eq!(OBJECT_LANES * std::mem::size_of::<f32>(), OBJECT_BYTES);
        let two = [car(1.0, 20.0, 0.0, -3.0), car(2.0, 12.0, 0.0, -5.0)];
        let four = [
            two[0],
            two[1],
            car(3.0, 8.0, 6.0, -5.0),
            car(4.0, 30.0, -6.0, 0.0),
        ];
        let (b2, b4) = (build(&two), build(&four));
        let (c2, c4) = (
            pipes_core::sample::payload_bytes(&b2),
            pipes_core::sample::payload_bytes(&b4),
        );
        assert!(c2 >= 2 * OBJECT_BYTES, "{c2} B carried for two records");
        assert_eq!(
            c4 - c2,
            2 * OBJECT_BYTES,
            "the per-track cost is not OBJECT_BYTES"
        );
        println!("state payload_bytes = {c2} for 2 records, {c4} for 4");
        // An empty scene is still an answer, and costs only its provenance.
        let empty = build(&[]);
        assert!(object_rows(&empty).is_empty());
        assert!(!read_answer(&empty).unwrap().has_object);
    }

    /// The corridor is half the vehicle, and the vehicle is the KITTI
    /// platform. If either changes the other must.
    #[test]
    fn the_corridor_is_half_the_recording_vehicle() {
        assert_eq!(CORRIDOR_HALF_WIDTH_M * 2.0, VEHICLE_WIDTH_M);
        assert_eq!(CORRIDOR_HALF_WIDTH_M, 0.91);
    }

    /// Ties are broken by id, not by whichever row came first -- in the
    /// answer and in the flag.
    #[test]
    fn a_tie_is_broken_deterministically() {
        let a = car(7.0, 10.0, 0.5, -1.0);
        let b = car(3.0, 10.0, -0.5, -1.0);
        assert_eq!(StateAnswer::of(&[a, b]).object_id, 3);
        assert_eq!(StateAnswer::of(&[b, a]).object_id, 3);
        for order in [[a, b], [b, a]] {
            let flagged: Vec<u32> = object_rows(&build(&order))
                .iter()
                .map(Object::from_lanes)
                .filter(|o| o.nearest_in_path)
                .map(|o| o.id)
                .collect();
            assert_eq!(flagged, [3]);
        }
    }
}
