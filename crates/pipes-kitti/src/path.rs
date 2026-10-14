//! The path the vehicle is on, and whether a track stands on it.
//!
//! The answer used to ask whether a track's box reached into a corridor
//! **straight ahead**. A vehicle that is turning does not go straight ahead:
//! on drive_0005, which turns on 150 of its 154 sweeps at radii of 16 to 60
//! m, the nearest thing in the straight corridor and the nearest thing on the
//! road the vehicle was actually following were different objects on 127
//! sweeps, and the one the straight corridor called most urgent -- a parked
//! car 7.56 m away, 1.3 s to contact -- was beside the road, while a cyclist
//! rode ahead on it. This module is the corridor bent to the road.
//!
//! # The path is an arc, from the vehicle's own two numbers
//!
//! The GPS/IMU row the fusion already reads ([`crate::oxts`]) says how fast
//! the vehicle goes forward (`vf`) and how fast it turns (`yaw_rate`). A
//! vehicle doing both follows a circle of curvature `yaw_rate / vf` (one over
//! the radius, positive to the left), and for the second or two an answer
//! looks ahead that circle is the path: [`Path::of`]. No map, no steering
//! model, and no parameter. Two bounds keep the division honest:
//!
//! * under [`MIN_SPEED_MPS`] the path is straight. A yaw rate over a speed
//!   near zero is noise over nothing, and a vehicle that is nearly stopped
//!   has no arc worth predicting;
//! * under [`MIN_CURVATURE`] it is straight too: a radius over 10 km bends
//!   4.5 cm in the 30 m the detector sees, a quarter of a voxel.
//!
//! Without the vehicle's motion (`--oxts off`, or a row that was not there)
//! the path is [`Path::STRAIGHT`], which is the corridor as it always was.
//!
//! # What stands on it
//!
//! The corridor is the path's centre line widened by half the vehicle's own
//! width either side ([`crate::state::CORRIDOR_HALF_WIDTH_M`]): the ground
//! the vehicle will sweep. A track stands on it ([`place`]) when it is
//! wholly ahead of the sensor and
//!
//! * **its lidar-detector box overlaps the corridor**, for a track the
//!   lidar detector named ([`crate::name`]): the network's box is a tight
//!   rectangle turned to the object's heading, so "does any of it reach the
//!   corridor" is a fair question to ask it, and it is asked exactly
//!   ([`Path::reaches`]); or
//! * **its centroid lies in the corridor**, for every other track. Not its
//!   box. A cluster's box is the axis-aligned hull of whatever was
//!   connected: for a hedge or a row of parked cars along a bend that hull
//!   is a rectangle metres wider than the material in it, and it covers the
//!   road the material stands beside. The centroid is the mean of the
//!   cluster's own returns, so it is where the material is.
//!
//! Measured on drive_0005 with the lidar detector on: with this rule the
//! nearest thing on the path within 15 m was named by the lidar detector on
//! 71 of 82 sweeps and moving on 71 (the cyclist); with the straight
//! corridor and the cluster boxes it was stationary on 72 of 117 sweeps --
//! walls, hedges and parked cars the vehicle was turning past. The same
//! boxes against the curved path kept two stationary objects "on the path"
//! until they were beside the vehicle's own nose, which a vehicle that hit
//! nothing shows to be false.
//!
//! **What the centroid rule gives up, stated:** an unnamed structure wider
//! than the corridor whose middle is beside the road and whose end reaches
//! into it -- a wall across half the lane -- is not on the path. The
//! structures a road user must not hit are the ones the detectors name, and
//! those are tested by their box.
//!
//! # How far the arc holds
//!
//! The arc is what the vehicle is doing at one instant, carried on. A
//! recorded drive says what it did instead, in the GPS/IMU's later rows,
//! which the pipeline never reads. Over every frame of the three longer
//! drives on disk, this is the share whose position after so many metres
//! was within half the vehicle's width of where the arc had put it, and the
//! same for a straight line (measured by a script since removed with the
//! decision layer; docs/measurements.md, "The decision"):
//!
//! | drive | path | 5 m | 10 m | 15 m | 20 m | 30 m |
//! |---|---|---|---|---|---|---|
//! | 0005 | arc | 100 % | 79 % | 57 % | 28 % | 1 % |
//! | | straight | 95 % | 12 % | 7 % | 2 % | 0 % |
//! | 0009 | arc | 98 % | 86 % | 78 % | 70 % | 59 % |
//! | | straight | 88 % | 82 % | 79 % | 75 % | 46 % |
//! | 0013 | arc | 100 % | 100 % | 100 % | 92 % | 69 % |
//! | | straight | 100 % | 100 % | 100 % | 92 % | 56 % |
//!
//! So the arc is the road for the first ten metres, and on a street that
//! keeps turning (drive_0005) a guess past twenty: a driver unwinds a bend
//! the arc takes to go on. A path whose curvature changes at the rate it
//! has been changing -- a clothoid, fitted over the last half second -- was
//! tried on the same frames and is no better past ten metres, so the arc
//! stays: it is one division, and its error is the one a reader expects.
//! What follows from that error: a standing object well along a tight arc
//! may be beside the road the vehicle goes on to take, and still be counted
//! as in its path.
//!
//! # The distance stays the lidar's
//!
//! The gap to a track is measured along the path, to the nearest corner of
//! the track's own cluster box ([`Placed::gap_m`]): on a straight path that
//! is the box's near face, exactly the distance the answer always reported.
//! The lidar detector's box decides whether a track is on the path and what
//! it is called; it never moves a distance.

use crate::oxts::EgoRates;
use crate::track::{track_lidar_box, TRACK_LANES};

/// Forward speed under which the path is taken as straight, m/s. Walking
/// pace: below it the yaw rate's noise over the speed is a curvature of
/// nothing.
pub const MIN_SPEED_MPS: f64 = 1.0;

/// Curvature under which the path is taken as straight, 1/m: a radius over
/// 10 km, whose arc leaves the straight line by 4.5 cm at 30 m.
pub const MIN_CURVATURE: f64 = 1e-4;

/// The path the vehicle is on: an arc from the sensor's origin, tangent to
/// its forward axis.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Path {
    /// One over the radius, 1/m, positive when turning left; 0 is straight.
    curvature: f64,
}

impl Path {
    /// Straight ahead: the path without the vehicle's own motion.
    pub const STRAIGHT: Path = Path { curvature: 0.0 };

    /// The arc the vehicle is on, from its forward speed and yaw rate, or
    /// straight without them, at walking pace, or when the arc is flatter
    /// than [`MIN_CURVATURE`].
    pub fn of(ego: Option<EgoRates>) -> Path {
        match ego {
            Some(r) if r.vf >= MIN_SPEED_MPS => Path::from_curvature(r.yaw_rate / r.vf),
            _ => Path::STRAIGHT,
        }
    }

    /// A path of this curvature, 1/m: what a batch's `path_curvature`
    /// column reads back to. Straight under [`MIN_CURVATURE`], or for a
    /// value that is not a number.
    pub fn from_curvature(curvature: f64) -> Path {
        if curvature.is_finite() && curvature.abs() >= MIN_CURVATURE {
            Path { curvature }
        } else {
            Path::STRAIGHT
        }
    }

    /// One over the radius, 1/m, positive to the left; 0 when straight.
    pub fn curvature(&self) -> f64 {
        self.curvature
    }

    /// The radius, m, signed like the curvature; `None` when straight.
    pub fn radius_m(&self) -> Option<f64> {
        (self.curvature != 0.0).then(|| 1.0 / self.curvature)
    }

    /// Where a point of the ground plane is against the path: how far to
    /// the left of its centre line (negative: to the right), and how far
    /// along it (negative: behind). On a straight path those are `y` and
    /// `x`.
    pub fn locate(&self, x: f32, y: f32) -> (f32, f32) {
        let Some(r) = self.radius_m() else {
            return (y, x);
        };
        let (x, y) = (f64::from(x), f64::from(y));
        let side = r.signum();
        // The arc's centre is at (0, r); a point's distance from it against
        // the radius is its offset, and its angle round it the arc length.
        let rho = x.hypot(y - r);
        let offset = side * (r.abs() - rho);
        let along = r.abs() * x.atan2(side * (r - y));
        (offset as f32, along as f32)
    }

    /// The path's direction where it has run `along` metres, as the unit
    /// vector `(x, y)`: what a velocity is projected on to say how fast an
    /// object moves along the road. `(1, 0)` on a straight path.
    pub fn tangent(&self, along: f32) -> (f32, f32) {
        match self.radius_m() {
            None => (1.0, 0.0),
            Some(r) => {
                let theta = f64::from(along) / r.abs();
                (theta.cos() as f32, (r.signum() * theta.sin()) as f32)
            }
        }
    }

    /// The point of the ground plane `along` metres down the path and
    /// `offset` metres to the left of its centre line: [`Path::locate`]
    /// backwards, which is what drawing the corridor needs.
    pub fn point(&self, along: f32, offset: f32) -> (f32, f32) {
        let Some(r) = self.radius_m() else {
            return (along, offset);
        };
        let side = r.signum();
        let theta = f64::from(along) / r.abs();
        // Round the arc's centre at (0, r), at the radius the offset leaves.
        let rho = r.abs() - side * f64::from(offset);
        (
            (rho * theta.sin()) as f32,
            (r - side * rho * theta.cos()) as f32,
        )
    }

    /// Whether a rectangle of the ground plane -- centre, full length and
    /// width, heading -- reaches the corridor of `half_width` either side
    /// of the path. Exact: the rectangle's distances from the arc's centre
    /// run from the nearest point of it to its farthest corner, and it
    /// reaches the corridor when that run meets the corridor's own.
    pub fn reaches(&self, rect: &Rect, half_width: f32) -> bool {
        let hw = f64::from(half_width);
        let corners = rect.corners();
        let Some(r) = self.radius_m() else {
            // Straight: the corridor is the strip |y| <= hw.
            let (lo, hi) = corners
                .iter()
                .fold((f64::INFINITY, f64::NEG_INFINITY), |m, c| {
                    (m.0.min(c[1]), m.1.max(c[1]))
                });
            return lo <= hw && hi >= -hw;
        };
        // The arc's centre, in the rectangle's own frame.
        let (s, c) = f64::from(rect.yaw).sin_cos();
        let (dx, dy) = (-f64::from(rect.x), r - f64::from(rect.y));
        let (u, v) = (dx * c + dy * s, -dx * s + dy * c);
        let near = (u.abs() - 0.5 * f64::from(rect.dx))
            .max(0.0)
            .hypot((v.abs() - 0.5 * f64::from(rect.dy)).max(0.0));
        let far = corners
            .iter()
            .map(|p| p[0].hypot(p[1] - r))
            .fold(0.0f64, f64::max);
        near <= r.abs() + hw && far >= r.abs() - hw
    }
}

/// A rectangle of the ground plane: the lidar detector's box, seen from
/// above.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rect {
    /// Centre.
    pub x: f32,
    /// See `x`.
    pub y: f32,
    /// Length along its heading.
    pub dx: f32,
    /// Width across it.
    pub dy: f32,
    /// Heading, radians from +x towards +y.
    pub yaw: f32,
}

impl Rect {
    /// The four corners, in f64.
    fn corners(&self) -> [[f64; 2]; 4] {
        let (s, c) = f64::from(self.yaw).sin_cos();
        let (hx, hy) = (0.5 * f64::from(self.dx), 0.5 * f64::from(self.dy));
        let at = |u: f64, v: f64| {
            [
                f64::from(self.x) + u * c - v * s,
                f64::from(self.y) + u * s + v * c,
            ]
        };
        [at(-hx, -hy), at(hx, -hy), at(hx, hy), at(-hx, hy)]
    }
}

/// Where one track is against the path.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Placed {
    /// Wholly ahead of the sensor and on the corridor: by its lidar-detector
    /// box when it has one, by its centroid when it does not.
    pub on: bool,
    /// Distance along the path to the nearest corner of the track's own
    /// cluster box, metres, signed: the box's near face on a straight path.
    /// Negative for a box that reaches behind the sensor.
    pub gap_m: f32,
    /// The centroid's offset from the path's centre line, metres, positive
    /// to the left.
    pub offset_m: f32,
    /// The centroid's distance along the path, metres.
    pub along_m: f32,
}

/// Places one track record against the path, for a corridor of `half_width`
/// either side of it. See the module docs for the rule.
pub fn place(path: &Path, t: &[f32; TRACK_LANES], half_width: f32) -> Placed {
    let (offset_m, along_m) = path.locate(t[0], t[1]);
    // The cluster box's four corners along the path: the nearest is the gap.
    let gap_m = [(t[9], t[10]), (t[9], t[13]), (t[12], t[10]), (t[12], t[13])]
        .iter()
        .map(|&(x, y)| path.locate(x, y).1)
        .fold(f32::INFINITY, f32::min);
    // Written positively, so a NaN anywhere is "not on the path".
    let ahead = gap_m > 0.0;
    let in_corridor = match track_lidar_box(t) {
        Some(rect) => path.reaches(&rect, half_width),
        None => offset_m.abs() <= half_width,
    };
    Placed {
        on: ahead && in_corridor,
        gap_m,
        offset_m,
        along_m,
    }
}

/// How fast the gap to a track closes, m/s, positive while it shrinks: the
/// vehicle's own forward speed less the track's velocity along the path
/// where the track is. `ego_vf` is `None` when the track's velocity is
/// relative to the vehicle already, and the closing speed is then minus its
/// part along the path.
pub fn closing_mps(path: &Path, t: &[f32; TRACK_LANES], along_m: f32, ego_vf: Option<f32>) -> f32 {
    let (tx, ty) = path.tangent(along_m);
    let along_speed = t[3] * tx + t[4] * ty;
    match ego_vf {
        Some(vf) => vf - along_speed,
        None => -along_speed,
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::track::{LANE_LIDAR_BOX, LANE_LIDAR_CONFIDENCE, LANE_LIDAR_DET_INDEX, TRACK_LANES};

    const HW: f32 = 0.91;

    fn ego(vf: f64, yaw_rate: f64) -> Option<EgoRates> {
        Some(EgoRates {
            vf,
            vl: 0.0,
            yaw_rate,
        })
    }

    #[test]
    fn a_point_of_the_path_is_where_locate_finds_it() {
        for curvature in [0.0, 1.0 / 20.0, -1.0 / 20.0, 1.0 / 8.5, -1.0 / 300.0] {
            let path = Path::from_curvature(curvature);
            for along in [0.0f32, 1.0, 7.5, 12.0] {
                for offset in [0.0f32, HW, -HW, 3.0] {
                    let (x, y) = path.point(along, offset);
                    let (o, a) = path.locate(x, y);
                    assert!(
                        (o - offset).abs() < 1e-3 && (a - along).abs() < 1e-3,
                        "curvature {curvature}: ({along}, {offset}) -> ({x}, {y}) -> ({a}, {o})"
                    );
                }
            }
        }
        // Straight ahead it is the point itself; on a left turn of 20 m
        // radius a quarter of the circle ends 20 m ahead and 20 m left.
        assert_eq!(Path::STRAIGHT.point(12.0, -0.5), (12.0, -0.5));
        let (x, y) =
            Path::from_curvature(1.0 / 20.0).point(std::f32::consts::FRAC_PI_2 * 20.0, 0.0);
        assert!(
            (x - 20.0).abs() < 1e-3 && (y - 20.0).abs() < 1e-3,
            "({x}, {y})"
        );
    }

    /// A track whose centroid is at `(x, y)` with a half-metre cluster box.
    fn trk(x: f32, y: f32) -> [f32; TRACK_LANES] {
        let mut t = [0.0f32; TRACK_LANES];
        t[0] = x;
        t[1] = y;
        t[9] = x - 0.25;
        t[10] = y - 0.25;
        t[12] = x + 0.25;
        t[13] = y + 0.25;
        t[LANE_LIDAR_DET_INDEX] = -1.0;
        t
    }

    fn named(mut t: [f32; TRACK_LANES], r: Rect) -> [f32; TRACK_LANES] {
        t[LANE_LIDAR_DET_INDEX] = 0.0;
        t[LANE_LIDAR_CONFIDENCE] = 0.9;
        t[LANE_LIDAR_BOX..LANE_LIDAR_BOX + 5].copy_from_slice(&[r.x, r.y, r.dx, r.dy, r.yaw]);
        t
    }

    #[test]
    fn the_path_is_straight_without_motion_at_walking_pace_and_when_nearly_flat() {
        assert_eq!(Path::of(None), Path::STRAIGHT);
        assert_eq!(Path::of(ego(0.9, 0.2)), Path::STRAIGHT);
        assert_eq!(Path::of(ego(10.0, 0.0005)), Path::STRAIGHT);
        let left = Path::of(ego(5.0, 0.25));
        assert!((left.radius_m().unwrap() - 20.0).abs() < 1e-9);
        let right = Path::of(ego(5.0, -0.25));
        assert!((right.radius_m().unwrap() + 20.0).abs() < 1e-9);
        assert_eq!(Path::from_curvature(f64::NAN), Path::STRAIGHT);
        assert_eq!(Path::from_curvature(left.curvature()), left);
        // Straight is the corridor as it was: offset is y, along is x.
        assert_eq!(Path::STRAIGHT.locate(7.0, -2.0), (-2.0, 7.0));
        assert_eq!(Path::STRAIGHT.tangent(12.0), (1.0, 0.0));
    }

    #[test]
    fn a_point_on_the_arc_has_no_offset_and_its_arc_length_along() {
        for r in [20.0f64, -20.0, 41.5, -16.4] {
            let path = Path::from_curvature(1.0 / r);
            for theta in [0.1f64, 0.5, 1.0] {
                // The arc's own point at angle theta, and one a metre to
                // its left.
                let (x, y) = (r.abs() * theta.sin(), r * (1.0 - theta.cos()));
                let (off, along) = path.locate(x as f32, y as f32);
                assert!(off.abs() < 1e-3, "r {r} theta {theta}: offset {off}");
                assert!((f64::from(along) - r.abs() * theta).abs() < 1e-3, "{along}");
                let (tx, ty) = path.tangent(along);
                let (nx, ny) = (-f64::from(ty), f64::from(tx));
                let (off, _) = path.locate((x + nx) as f32, (y + ny) as f32);
                assert!((off - 1.0).abs() < 1e-3, "r {r}: a metre left reads {off}");
            }
        }
        // At the origin a turning path agrees with the straight one.
        let path = Path::from_curvature(1.0 / 20.0);
        let (off, along) = path.locate(0.0, 0.5);
        assert!((off - 0.5).abs() < 1e-5 && along.abs() < 1e-5);
    }

    #[test]
    fn a_centroid_in_the_corridor_is_on_the_path_and_one_beside_it_is_not() {
        // Straight: as the corridor always was, for a compact track.
        assert!(place(&Path::STRAIGHT, &trk(10.0, 0.5), HW).on);
        assert!(!place(&Path::STRAIGHT, &trk(10.0, 1.5), HW).on);
        assert!(!place(&Path::STRAIGHT, &trk(-5.0, 0.0), HW).on);
        let p = place(&Path::STRAIGHT, &trk(10.0, 0.5), HW);
        assert_eq!((p.gap_m, p.offset_m, p.along_m), (9.75, 0.5, 10.0));
        // Turning left on a 20 m radius: at 10 m of arc the road is 2.4 m
        // to the left of straight ahead. The track straight ahead is beside
        // it, and the one on the arc is on it.
        let path = Path::of(ego(5.0, 0.25));
        let theta = 0.5f64;
        let (x, y) = (
            (20.0 * theta.sin()) as f32,
            (20.0 * (1.0 - theta.cos())) as f32,
        );
        assert!(y > 2.3, "{y}");
        assert!(place(&path, &trk(x, y), HW).on);
        assert!(!place(&path, &trk(x, 0.0), HW).on);
        let p = place(&path, &trk(x, y), HW);
        assert!(
            (p.along_m - 10.0).abs() < 1e-3 && p.gap_m < p.along_m,
            "{p:?}"
        );
    }

    /// A wide cluster's box is not the test: a hedge beside the road whose
    /// hull covers the corridor is on the path only if its centroid is.
    #[test]
    fn a_wide_box_whose_centroid_is_beside_the_corridor_is_not_on_the_path() {
        let mut hedge = trk(12.0, 3.0);
        hedge[10] = -1.0; // the hull reaches across the corridor
        hedge[13] = 7.0;
        assert!(!place(&Path::STRAIGHT, &hedge, HW).on);
        // Its gap is still its own box's near face.
        assert_eq!(place(&Path::STRAIGHT, &hedge, HW).gap_m, 11.75);
    }

    /// A track the lidar detector named is tested by the detector's box:
    /// a car whose fragment's centroid is beside the corridor is on the path
    /// when the car's rectangle reaches it, and off it when turned away.
    #[test]
    fn a_named_track_is_on_the_path_when_its_lidar_box_reaches_the_corridor() {
        // A fragment at y = 1.4: its centroid is outside the corridor.
        let fragment = trk(10.0, 1.4);
        assert!(!place(&Path::STRAIGHT, &fragment, HW).on);
        // The car it is a fragment of: 4 m long, 1.8 m wide, centred 1.5 m
        // left, so its right side is at y = 0.6: inside the corridor.
        let car = Rect {
            x: 10.0,
            y: 1.5,
            dx: 4.0,
            dy: 1.8,
            yaw: 0.0,
        };
        assert!(place(&Path::STRAIGHT, &named(fragment, car), HW).on);
        // The same car a metre further left clears it.
        let clear = Rect { y: 2.5, ..car };
        assert!(!place(&Path::STRAIGHT, &named(fragment, clear), HW).on);
        // Turned across the road, its length reaches it again.
        let across = Rect {
            y: 2.5,
            yaw: std::f32::consts::FRAC_PI_2,
            ..car
        };
        assert!(place(&Path::STRAIGHT, &named(fragment, across), HW).on);
        // On a left turn of 20 m radius the corridor at x = 10 is 2.7 m to
        // the left: the car at 2.5 m is on it and the one at 0 is not.
        let path = Path::of(ego(5.0, 0.25));
        let on_road = Rect { y: 2.7, ..car };
        assert!(path.reaches(&on_road, HW));
        let straight_ahead = Rect {
            y: -0.5,
            dy: 1.0,
            dx: 2.0,
            ..car
        };
        assert!(!path.reaches(&straight_ahead, HW));
        // A box behind the sensor is never on the path, whatever its rectangle.
        assert!(
            !place(
                &Path::STRAIGHT,
                &named(
                    trk(-3.0, 0.0),
                    Rect {
                        x: -3.0,
                        y: 0.0,
                        ..car
                    }
                ),
                HW
            )
            .on
        );
    }

    #[test]
    fn the_closing_speed_is_along_the_path() {
        // Straight: the vehicle's speed less the object's forward velocity,
        // or minus a relative one, as it always was.
        let mut t = trk(10.0, 0.0);
        t[3] = 2.0;
        assert_eq!(closing_mps(&Path::STRAIGHT, &t, 10.0, Some(5.0)), 3.0);
        assert_eq!(closing_mps(&Path::STRAIGHT, &t, 10.0, None), -2.0);
        // A quarter of the way round a left turn, an object moving along
        // the road moves partly sideways in the sensor's axes: its speed
        // along the path is the whole of it.
        let path = Path::from_curvature(1.0 / 20.0);
        let along = (20.0 * std::f64::consts::FRAC_PI_4) as f32;
        let (tx, ty) = path.tangent(along);
        let mut rider = trk(14.1, 5.9);
        rider[3] = 4.0 * tx;
        rider[4] = 4.0 * ty;
        assert!((closing_mps(&path, &rider, along, Some(5.0)) - 1.0).abs() < 1e-5);
    }
}
