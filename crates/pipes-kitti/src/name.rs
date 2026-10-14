//! Which of the lidar detector's boxes is which lidar track: the class a
//! track takes from the lidar, beside the one the cameras give it.
//!
//! [`crate::fuse`] gives a track a class from the camera by overlap in the
//! image. This module gives it one from the lidar itself, by containment in
//! the ground plane: [`crate::pillars`] finds cars, pedestrians and cyclists
//! as boxes with a heading, in metres, in the same frame the tracks are in,
//! so no projection and no second sensor's geometry come between them.
//!
//! # The rule
//!
//! A **fresh** track (`misses == 0`, lane 8) takes a box's class when its
//! centroid (lanes 0-1) lies inside the box's footprint, the rotated
//! rectangle of [`LidarDet::contains_bev`], and the box scores at least
//! [`NAME_SCORE_MIN`]. The centroid and not the track's own box, because
//! `detect`'s boxes are axis-aligned hulls of a cluster and the network's are
//! tight, oriented and learned: a car's cluster box overlaps the pavement
//! beside it, its centroid does not. Inside several boxes -- rare after an
//! NMS that suppresses any overlap -- the highest score wins, then the
//! lowest detection index, a total order.
//!
//! # The score a box needs, and where that number comes from
//!
//! The network's own threshold is 0.1 ([`crate::pillars::SCORE_THRESHOLD`]),
//! the one it is evaluated with, where a detector is scored on recall as
//! well as precision. The `pillars` stage keeps every box at it: that is
//! what the model said, and it is all in `pillar_det.arrows`. A name on a
//! track is a different claim -- "this is a pedestrian" -- and at 0.1 most
//! of the boxes are not: on drive_0005, 5,203 of 6,959 boxes score under
//! 0.3, and 2,677 of those are "pedestrians", which on that street are poles
//! and tree trunks.
//!
//! [`NAME_SCORE_MIN`] is 0.3. **It is chosen, then checked, and this is the
//! one place that says so**, as [`crate::detect::PERSISTENCE_GATE_M`] does
//! for its own. It is the camera detector's threshold
//! ([`crate::camdet::SCORE_THRESHOLD`]), so the two sensors' names clear the
//! same bar; and the check is the other sensor: on drive_0005, of the tracks
//! a camera also named, all 352 named by a box at 0.3 or more got the same
//! kind of object from the camera, against 48 of the 58 named by a box
//! under it (and 0 of 5 for the pedestrians under it). That is agreement
//! between two sensors, which the run counts every time (`agree`, `differ`),
//! on whatever drive it replays. Accuracy is a second check, against the
//! boxes KITTI's labellers drew on the four drives
//! (`scripts/score_detectors.py`): inside the camera's view, boxes that
//! overlap no labelled object number 12.9 a sweep at the network's 0.1 and
//! 2.4 at 0.3, while the labelled vehicles found go from 94 % to 91 %. The
//! bar costs three vehicles in a hundred and removes four boxes in five
//! that no label stands under.
//!
//! One box may name several tracks, and that is information rather than a
//! conflict: `detect` breaks a car into fragments ([`crate::detect`], about
//! 59 % of its detections), and a box that names three fragments says they
//! are one car. The plan counts those boxes ([`NamePlan::shared`]) rather
//! than choosing one fragment.
//!
//! # What it does not do
//!
//! Remember a class (every sweep is named afresh from its own box, as the
//! fusion is), move a track (the lidar's distance and velocity are
//! `detect`'s and `track`'s), or make an object out of a box no track lies
//! in (counted as [`NamePlan::unused`]: a pedestrian at 40 m the clustering
//! range does not reach, or a box with nothing under it). A camera class and
//! a lidar class on one track are both kept, and whether they agree is
//! [`agrees`]'s question, answered per record and counted by the run.

use crate::camdet::COCO_CLASSES;
use crate::pillars::{LidarDet, PILLAR_DET_LANES};
use crate::track::{
    LANE_LIDAR_BOX, LANE_LIDAR_CLASS_ID, LANE_LIDAR_CONFIDENCE, LANE_LIDAR_DET_INDEX, TRACK_LANES,
};

/// The rule, in one line, for `run.json`.
pub const NAMING_RULE: &str =
    "a fresh track takes the lidar detector's class when its centroid lies \
    inside the footprint (the rotated rectangle in the ground plane) of a box scoring 0.3 or more; \
    inside several, the highest score, then the lowest detection index";

/// The least a box may score and still name a track. See the module docs:
/// chosen (the camera detector's own threshold), then checked against the
/// cameras' classes, and the network's every box is recorded either way.
pub const NAME_SCORE_MIN: f32 = 0.3;

/// What naming one sweep's tracks did.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct NamePlan {
    /// Whether there was a batch of boxes to name from at all.
    pub ran: bool,
    /// Boxes in the batch.
    pub detections: u32,
    /// Boxes scoring under [`NAME_SCORE_MIN`]: recorded, and naming nothing.
    pub below_score: u32,
    /// Fresh tracks: the ones that could be named.
    pub candidates: u32,
    /// Tracks given a class.
    pub named: u32,
    /// Boxes at the score or above that named at least one track.
    pub used: u32,
    /// Boxes at the score or above that named none.
    pub unused: u32,
    /// Boxes that named more than one track: fragments of one object.
    pub shared: u32,
    /// Tracks whose centroid lay inside more than one box.
    pub contested: u32,
    /// Tracks named, by class id.
    pub classes: [u32; 3],
}

/// Names `rows` from `dets` **in place**, writing lanes
/// [`LANE_LIDAR_DET_INDEX`], [`LANE_LIDAR_CLASS_ID`],
/// [`LANE_LIDAR_CONFIDENCE`] and the five of [`LANE_LIDAR_BOX`] of each
/// named track; the lanes of the others are left as
/// [`crate::track::Tracker::build_fused`] wrote them (-1 and zeros).
/// Allocates nothing.
pub fn name_tracks(rows: &mut [[f32; TRACK_LANES]], dets: &[[f32; PILLAR_DET_LANES]]) -> NamePlan {
    let mut plan = NamePlan {
        ran: true,
        detections: dets.len() as u32,
        ..NamePlan::default()
    };
    for t in rows.iter_mut() {
        if t[8] != 0.0 {
            continue;
        }
        plan.candidates += 1;
        let (x, y) = (t[0], t[1]);
        let mut best: Option<(usize, f32)> = None;
        let mut inside = 0u32;
        for (i, lanes) in dets.iter().enumerate() {
            let d = LidarDet::from_lanes(lanes);
            // Written positively, so a NaN score names nothing.
            let scores = d.score >= NAME_SCORE_MIN;
            if !(scores && d.contains_bev(x, y)) {
                continue;
            }
            inside += 1;
            // Score descending, then index ascending: the index order is
            // the loop's, so a tie keeps the earlier box.
            if best.is_none_or(|(_, s)| d.score > s) {
                best = Some((i, d.score));
            }
        }
        if inside > 1 {
            plan.contested += 1;
        }
        if let Some((i, score)) = best {
            let class = dets[i][7] as u32;
            t[LANE_LIDAR_DET_INDEX] = i as f32;
            t[LANE_LIDAR_CLASS_ID] = class as f32;
            t[LANE_LIDAR_CONFIDENCE] = score;
            // The box's rectangle in the ground plane, beside its name:
            // centre, length, width, heading (lanes 0, 1, 3, 4, 6 of a box).
            let b = &dets[i];
            t[LANE_LIDAR_BOX..LANE_LIDAR_BOX + 5].copy_from_slice(&[b[0], b[1], b[3], b[4], b[6]]);
            plan.named += 1;
            if let Some(c) = plan.classes.get_mut(class as usize) {
                *c += 1;
            }
        }
    }
    // The boxes' side, read back off the lanes rather than remembered: a
    // second pass over a few dozen boxes and a couple of hundred rows.
    for (i, d) in dets.iter().enumerate() {
        // Lane 8 of a box is its score.
        if d[8] < NAME_SCORE_MIN || d[8].is_nan() {
            plan.below_score += 1;
            continue;
        }
        let named = rows
            .iter()
            .filter(|t| t[LANE_LIDAR_DET_INDEX] == i as f32)
            .count();
        match named {
            0 => plan.unused += 1,
            1 => plan.used += 1,
            _ => {
                plan.used += 1;
                plan.shared += 1;
            }
        }
    }
    plan
}

/// Whether a camera class (an index into [`COCO_CLASSES`]) and a lidar
/// class (an index into [`crate::pillars::CLASSES`]) name the same kind of
/// road user. The two vocabularies differ, so this is by family: the
/// lidar's `car` is the camera's car, bus or truck; its `pedestrian` is a
/// person; its `cyclist` -- KITTI's word for a person on a bicycle or a
/// motorcycle -- is a bicycle, a motorcycle, or the person riding it, which
/// is what the camera's box most often covers.
pub fn agrees(camera_class: u32, lidar_class: u32) -> bool {
    let camera = COCO_CLASSES.get(camera_class as usize).copied();
    match lidar_class {
        0 => matches!(camera, Some("car" | "bus" | "truck")),
        1 => matches!(camera, Some("person")),
        2 => matches!(camera, Some("bicycle" | "motorcycle" | "person")),
        _ => false,
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::pillars::LidarDet;

    fn row(x: f32, y: f32, misses: f32) -> [f32; TRACK_LANES] {
        let mut t = [0.0f32; TRACK_LANES];
        t[0] = x;
        t[1] = y;
        t[8] = misses;
        t[LANE_LIDAR_DET_INDEX] = -1.0;
        t
    }

    fn det(
        x: f32,
        y: f32,
        dx: f32,
        dy: f32,
        yaw: f32,
        class: u32,
        score: f32,
    ) -> [f32; PILLAR_DET_LANES] {
        LidarDet {
            x,
            y,
            z: 0.0,
            dx,
            dy,
            dz: 1.5,
            yaw,
            class_id: class,
            score,
        }
        .lanes()
    }

    #[test]
    fn a_fresh_track_inside_a_box_takes_its_class_and_a_coasted_one_does_not() {
        let dets = [
            det(10.0, 0.0, 4.0, 2.0, 0.0, 0, 0.9),
            det(20.0, 5.0, 0.8, 0.6, 1.57, 1, 0.5),
        ];
        let mut rows = [
            row(10.5, 0.4, 0.0), // in the car
            row(9.0, -0.5, 0.0), // in the car too: a fragment
            row(10.5, 0.4, 1.0), // coasted: not named
            row(20.0, 5.0, 0.0), // the pedestrian
            row(30.0, 0.0, 0.0), // nothing there
        ];
        let plan = name_tracks(&mut rows, &dets);
        assert_eq!(plan.candidates, 4);
        assert_eq!(plan.below_score, 0);
        assert_eq!(plan.named, 3);
        assert_eq!(plan.used, 2);
        assert_eq!(plan.unused, 0);
        assert_eq!(plan.shared, 1);
        assert_eq!(plan.contested, 0);
        assert_eq!(plan.classes, [2, 1, 0]);
        assert_eq!(rows[0][LANE_LIDAR_DET_INDEX], 0.0);
        assert_eq!(rows[0][LANE_LIDAR_CLASS_ID], 0.0);
        assert_eq!(rows[0][LANE_LIDAR_CONFIDENCE], 0.9);
        // The box's rectangle rides with the name, and reads back as one.
        assert_eq!(
            crate::track::track_lidar_box(&rows[0]),
            Some(crate::path::Rect {
                x: 10.0,
                y: 0.0,
                dx: 4.0,
                dy: 2.0,
                yaw: 0.0
            })
        );
        assert_eq!(crate::track::track_lidar_box(&rows[2]), None);
        assert_eq!(rows[4][LANE_LIDAR_BOX..LANE_LIDAR_BOX + 5], [0.0; 5]);
        assert_eq!(rows[2][LANE_LIDAR_DET_INDEX], -1.0);
        assert_eq!(rows[3][LANE_LIDAR_CLASS_ID], 1.0);
        assert_eq!(rows[4][LANE_LIDAR_DET_INDEX], -1.0);
    }

    #[test]
    fn inside_two_boxes_the_higher_score_names_the_track_and_the_contest_is_counted() {
        let dets = [
            det(10.0, 0.0, 4.0, 2.0, 0.0, 0, 0.4),
            det(10.5, 0.0, 4.0, 2.0, 0.0, 2, 0.8),
        ];
        let mut rows = [row(10.2, 0.1, 0.0)];
        let plan = name_tracks(&mut rows, &dets);
        assert_eq!(plan.contested, 1);
        assert_eq!(rows[0][LANE_LIDAR_DET_INDEX], 1.0);
        assert_eq!(rows[0][LANE_LIDAR_CLASS_ID], 2.0);
        assert_eq!(plan.unused, 1);
        // Equal scores: the earlier box.
        let dets = [
            det(10.0, 0.0, 4.0, 2.0, 0.0, 0, 0.8),
            det(10.5, 0.0, 4.0, 2.0, 0.0, 2, 0.8),
        ];
        let mut rows = [row(10.2, 0.1, 0.0)];
        name_tracks(&mut rows, &dets);
        assert_eq!(rows[0][LANE_LIDAR_DET_INDEX], 0.0);
    }

    /// A box under the score names nothing, however well it sits on a
    /// track, and is counted apart from the boxes that could have named.
    #[test]
    fn a_box_under_the_score_names_nothing_and_is_counted() {
        let dets = [
            det(10.0, 0.0, 4.0, 2.0, 0.0, 1, 0.29),
            det(20.0, 0.0, 4.0, 2.0, 0.0, 0, NAME_SCORE_MIN),
            det(30.0, 0.0, 4.0, 2.0, 0.0, 0, f32::NAN),
        ];
        let mut rows = [
            row(10.0, 0.0, 0.0),
            row(20.0, 0.0, 0.0),
            row(30.0, 0.0, 0.0),
        ];
        let plan = name_tracks(&mut rows, &dets);
        assert_eq!(
            (
                plan.detections,
                plan.below_score,
                plan.named,
                plan.used,
                plan.unused
            ),
            (3, 2, 1, 1, 0)
        );
        assert_eq!(rows[0][LANE_LIDAR_DET_INDEX], -1.0);
        assert_eq!(rows[1][LANE_LIDAR_DET_INDEX], 1.0);
        assert_eq!(rows[2][LANE_LIDAR_DET_INDEX], -1.0);
        assert_eq!(plan.contested, 0);
        // The rule's one line says the number the constant holds.
        assert!(
            NAMING_RULE.contains(&format!("{NAME_SCORE_MIN} or more")),
            "{NAMING_RULE}"
        );
    }

    #[test]
    fn the_class_families_agree_across_the_two_vocabularies() {
        let coco = |name: &str| COCO_CLASSES.iter().position(|c| *c == name).unwrap() as u32;
        assert!(agrees(coco("car"), 0));
        assert!(agrees(coco("truck"), 0));
        assert!(!agrees(coco("person"), 0));
        assert!(agrees(coco("person"), 1));
        assert!(agrees(coco("bicycle"), 2));
        assert!(agrees(coco("person"), 2));
        assert!(!agrees(coco("car"), 2));
        assert!(!agrees(coco("car"), 7));
    }
}
