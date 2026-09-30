//! The lidar-to-camera projection: parsing KITTI's two calibration files,
//! collapsing them to one 3x4 matrix, and asking whether a 3D box is in frame.
//!
//! This is the geometry a fusion stage needs and nothing else. It does not
//! touch pixels: [`Calib::project_box`] turns a lidar-frame bounding box into
//! an image rectangle, which is what lets a tracked object carry "and it is
//! *there* in the picture" without any stage ever reading the picture.
//!
//! # The chain
//!
//! ```text
//! y = P_rect_02 @ [R_rect_00 0; 0 1] @ [R T; 0 1] @ (x, y, z, 1)
//! u = y0 / y2 ,  v = y1 / y2
//! ```
//!
//! `P_rect_02` is **3x4**, the two rotations are 4x4, so the product is a
//! single 3x4 [`Calib::m`] and the whole chain is one matrix-vector product
//! per point. Precomputed once at load; measured on this dataset's files it is
//!
//! ```text
//! [[ 6.09695409e+02, -7.21421597e+02, -1.25125855e+00, -1.23041806e+02],
//!  [ 1.80384202e+02,  7.64479802e+00, -7.19651474e+02, -1.01016688e+02],
//!  [ 9.99945389e-01,  1.24365378e-04,  1.04513030e-02, -2.69386912e-01]]
//! ```
//!
//! # The divide is not by `z_cam`, and the difference is not negligible
//!
//! `P_rect_02`'s fourth column is not zero — the raw-data `P_rect_0x` carries
//! the cam0-to-cam_x baseline there (`tx = 44.857`). So the denominator is
//! `z_rect + 0.002745884`, not `z_rect`. Dropping that column entirely still
//! projects every point to somewhere plausible and costs **2.91 px median,
//! 7.83 px p99** — below the threshold at which an overlay looks wrong. Only
//! a test on the matrix catches it, which is why [`Calib::m`] is public and
//! `calib_matrix_matches_the_measured_chain` pins all twelve entries.
//!
//! # Rejecting the half of the sweep that is behind the camera
//!
//! **The test is `w > 0`**, where `w` is the third component *before* the
//! divide. It is not `z_velo`, not range, and not quite `x_velo`: the `w = 0`
//! plane sits at `x_velo ~ 0.2694 m`, the camera's focal plane, 27 cm in front
//! of the lidar.
//!
//! Measured over all 154 sweeps of drive_0005 (18,746,749 points):
//!
//! | outcome | points | share |
//! |---|---|---|
//! | rejected, `w <= 0` (behind the camera plane) | 9,434,086 | 50.32 % |
//! | rejected, outside the image horizontally | 5,144,365 | 27.44 % |
//! | rejected, outside it vertically | 1,139,919 | 6.08 % |
//! | **kept** | **3,028,379** | **16.15 %** |
//!
//! Skipping the sign test does not lose points, it **invents** them. Dividing
//! by a negative `w` flips both signs, so the scene behind the car is mirrored
//! through the optical centre and lands in frame: on sweep 0, 20,172 of the
//! 63,155 points behind the camera divide to a pixel inside 1242x375, against
//! 20,478 genuine ones. One false point per real point, and in rows 125-250
//! the two are interleaved and indistinguishable by pixel position — nothing
//! but the sign of `w` separates them.
//!
//! The test is therefore done **in homogeneous coordinates, with no division
//! at all** ([`Calib::project`]): `w > 0 && 0 <= a < W*w && 0 <= b < H*w`.
//! Verified elementwise identical to divide-then-bounds on sweep 0 (the same
//! 20,478 points), and it cannot overflow — sweep 0's smallest positive `w` is
//! `5.7e-05`, which divides to pixel `(-34,609,869, 8,963,036)`, and anything
//! casting that to `i32` before bounds-checking wraps.
//!
//! # What is verified about the calibration, and what is not
//!
//! Overlays of frames 0, 60 and 120 line up: point silhouettes follow a
//! cyclist's torso and a van's outline, bollards each carry their own column,
//! and the point field stops exactly at the rooflines. Numerically, without
//! looking at an image: ground points project **below** the horizon row
//! (226.6-375.0 against 180.4) and points above the sensor project **above**
//! it (125.6-172.1), with no overlap. An edge-alignment sweep over four frames
//! puts the optimum within 1 px of zero shift, but with a broad surface — so
//! the alignment is constrained to about **±2 px**, not sub-pixel.
//!
//! ±2 px bounds the **rotation** to about ±0.15 deg (yaw moves a point 14.5 px
//! per degree). It barely constrains the **translation**: 5 cm of mount error
//! moves the median point 1-2.4 px, inside the noise floor. So the rotation is
//! verified and the translation is verified only to a few centimetres.
//!
//! **The error that dwarfs all of it is ego motion inside the sweep**, and
//! this module does not correct it. Over drive_0005's 103.27 ms rotation the
//! vehicle turns up to 1.49 deg, which is **~21.6 px** of projection error for
//! a point measured at the far end of the rotation — ten times the worst
//! calibration subtlety and ten times the verification floor. A projection
//! that mattered to a pixel would have to deskew, or restrict itself to points
//! measured near the trigger azimuth. What this module is used for here —
//! "which image rectangle is this object in" on a metre-scale box — is
//! comfortably inside that error, and [`ImageBox`] is documented as a region
//! rather than as a measurement.
//!
//! # f32 is safe
//!
//! Running the whole chain in `f32` costs a median of 0.00004 px and a maximum
//! of 0.00030 px over sweep 0's in-image points, so the projection could
//! consume the zero-copy `&[f32]` directly. It is still done in `f64` here
//! because the error grows near the `w = 0` singularity and the inputs are a
//! handful of box corners per sweep rather than 120,000 points — there is
//! nothing to save.

use std::path::{Path, PathBuf};

/// Camera-2 (left colour) is the one this project replays, so its `P_rect` and
/// `S_rect` rows are the ones read.
const CAM: &str = "02";

/// Smallest `w` a box corner may have before the box is refused as straddling
/// the camera plane, in the same units as `w` (metres along the optical axis,
/// offset by the baseline term).
///
/// Not a tuned number: `w` is a depth, `0.1 m` is inside the 27 cm gap between
/// the lidar origin and the `w = 0` plane, and no object this pipeline detects
/// can be there — [`crate::detect::RANGE_LIMIT_M`] starts at 30 m and the
/// nearest voxel of any real return is metres away. It exists so the divide in
/// [`Calib::project_box`] runs only on corners already known to be in front of
/// the camera, which is the guard that makes the pixel arithmetic total.
const W_MIN: f64 = 0.1;

/// The collapsed lidar-to-image projection for one KITTI capture date, plus
/// the rectified image size that projection is valid for.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Calib {
    /// `P_rect_02 @ R_rect_00 @ Tr_velo_to_cam`, row-major 3x4.
    ///
    /// Public because the only thing that catches a silently-wrong calibration
    /// is a test on these twelve numbers. Every plausible shortcut — omitting
    /// `R_rect_00` (6.58 px), using `R_rect_02` (4.98 px), using `P_rect_00`
    /// (2.91 px), zeroing the fourth column (2.91 px) — produces an overlay
    /// that looks right.
    pub m: [[f64; 4]; 3],
    /// Rectified image width from `S_rect_02`, in pixels.
    pub width: u32,
    /// Rectified image height from `S_rect_02`, in pixels.
    pub height: u32,
}

/// A rectangle in the rectified camera-2 image, in pixels, clipped to the
/// frame.
///
/// **A region, not a measurement.** It is the axis-aligned hull of a 3D box's
/// eight projected corners, so it is at least as large as the object and
/// larger for anything not facing the camera square-on; and it inherits the
/// ~21.6 px of intra-sweep ego motion the module docs describe. Read it as
/// "the object is in here", never as "the object is this size".
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ImageBox {
    /// Left edge, pixels, `0 <= x0 < x1 <= width`.
    pub x0: f32,
    /// Top edge, pixels, `0 <= y0 < y1 <= height`.
    pub y0: f32,
    /// Right edge, pixels.
    pub x1: f32,
    /// Bottom edge, pixels.
    pub y1: f32,
}

impl ImageBox {
    /// Width in pixels.
    pub fn w(&self) -> f32 {
        self.x1 - self.x0
    }

    /// Height in pixels.
    pub fn h(&self) -> f32 {
        self.y1 - self.y0
    }
}

/// Why a calibration could not be read, or could be read and is not a
/// calibration.
///
/// Every variant carries the **path** and not a bare filename: the two files
/// sit beside each other and both have an `R` row, so "field R malformed" is
/// ambiguous without it.
#[derive(Debug)]
pub enum CalibError {
    /// The file could not be read.
    Io {
        /// File that failed to open.
        path: PathBuf,
        /// The underlying error.
        source: std::io::Error,
    },
    /// A non-empty line had no `:` separating key from values.
    MissingColon {
        /// File the line is in.
        path: PathBuf,
        /// 1-based line number.
        line_no: usize,
        /// The line itself, so the report does not require the file.
        text: String,
    },
    /// A key the projection needs is not in the file.
    ///
    /// **The variant that matters most.** A silent default here is not a
    /// crash, it is a plausible answer: an identity for a missing `R_rect_00`
    /// costs 6.58 px and a zero for the missing fourth column of `P_rect_02`
    /// costs 2.91 px, and both would ship.
    MissingField {
        /// File that should have had it.
        path: PathBuf,
        /// The key that is missing.
        key: String,
    },
    /// A key had the wrong number of values.
    Arity {
        /// File the key is in.
        path: PathBuf,
        /// The key.
        key: String,
        /// Values the projection needs.
        expected: usize,
        /// Values the file has.
        got: usize,
    },
    /// A value was not a number.
    BadFloat {
        /// File the value is in.
        path: PathBuf,
        /// The key its row belongs to.
        key: String,
        /// Index of the offending value within the row.
        index: usize,
        /// The text that would not parse.
        text: String,
        /// The parse error.
        source: std::num::ParseFloatError,
    },
    /// A 3x3 that should be a rotation is not one, or is one but points the
    /// wrong way.
    ///
    /// The second case is the reason this variant exists. A **transposed**
    /// parse of `R` is still a proper rotation — `det` is still 1 and
    /// `R R^T - I` is still zero — so orthonormality cannot catch a
    /// row/column-major mix-up. It costs 7,552 px of median error and still
    /// leaves 10,036 points inside the image bounds, i.e. it does not fail, it
    /// produces a half-size plausible-looking result. `R @ (1,0,0)` must come
    /// out as camera-forward, and that is what [`CalibError::NotARotation::forward_z`]
    /// records.
    NotARotation {
        /// File the matrix is in.
        path: PathBuf,
        /// The key.
        key: String,
        /// Its determinant (must be within 1e-6 of 1).
        det: f64,
        /// `max |R R^T - I|` (must be under 1e-6).
        ortho_err: f64,
        /// `(R @ (1,0,0))[2]`: lidar-forward expressed in the camera frame
        /// must be camera-forward, so this must exceed 0.99. Measured
        /// row-major on this dataset: 0.99986. Transposed: -0.00062.
        forward_z: f64,
    },
    /// `S_rect_02` disagrees with the image the drive actually holds.
    ///
    /// Checked rather than assumed because KITTI resolution varies by capture
    /// date (1242x375, 1224x370, ...), which `crate::drives` already measures
    /// per drive rather than hardcoding. A constant 1242x375 would be wrong on
    /// three of the other four dates and wrong *silently*.
    ImageSize {
        /// File `S_rect_02` came from.
        path: PathBuf,
        /// What the calibration says.
        calib: (u32, u32),
        /// What the PNG says.
        drive: (u32, u32),
    },
}

impl std::fmt::Display for CalibError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CalibError::Io { path, .. } => write!(f, "{} could not be read", path.display()),
            CalibError::MissingColon {
                path,
                line_no,
                text,
            } => write!(
                f,
                "{}:{line_no}: no `:` in `{text}` (every calibration row is `key: values`)",
                path.display()
            ),
            CalibError::MissingField { path, key } => write!(
                f,
                "{} has no `{key}:` row, and there is no safe default for it",
                path.display()
            ),
            CalibError::Arity {
                path,
                key,
                expected,
                got,
            } => write!(
                f,
                "{}: `{key}` has {got} values, the projection needs {expected}",
                path.display()
            ),
            CalibError::BadFloat {
                path,
                key,
                index,
                text,
                ..
            } => write!(
                f,
                "{}: `{key}` value {index} is `{text}`, not a number",
                path.display()
            ),
            CalibError::NotARotation {
                path,
                key,
                det,
                ortho_err,
                forward_z,
            } => write!(
                f,
                "{}: `{key}` is not a forward-facing rotation (det={det}, \
                 max|RR^T-I|={ortho_err}, forward_z={forward_z}); a TRANSPOSED parse \
                 passes the first two and fails the third",
                path.display()
            ),
            CalibError::ImageSize { path, calib, drive } => write!(
                f,
                "{}: S_rect_{CAM} says {}x{} but the drive's images are {}x{}",
                path.display(),
                calib.0,
                calib.1,
                drive.0,
                drive.1
            ),
        }
    }
}

impl std::error::Error for CalibError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            CalibError::Io { source, .. } => Some(source),
            CalibError::BadFloat { source, .. } => Some(source),
            _ => None,
        }
    }
}

/// One calibration file, as `key -> raw text`.
///
/// The values stay **text** until something asks for them as numbers, so an
/// unparseable `calib_time` cannot break a projection that never needed it.
/// That is not hypothetical: `calib_time: 15-Mar-2012 11:37:16` contains
/// colons in its value, so `split(':')` gives four pieces on that line and a
/// numeric pass over every row fails on a file that is completely fine.
///
/// `BTreeMap`, not `HashMap`: `clippy.toml` bans the latter for
/// nondeterministic iteration and the reason applies directly here, because
/// this map feeds a matrix the project claims is replayable.
struct RawCalib {
    path: PathBuf,
    rows: std::collections::BTreeMap<String, String>,
}

impl RawCalib {
    /// One generic pass: `split_once(':')`, trim the key, keep the rest.
    fn read(path: &Path) -> Result<RawCalib, CalibError> {
        let text = std::fs::read_to_string(path).map_err(|e| CalibError::Io {
            path: path.to_path_buf(),
            source: e,
        })?;
        let mut rows = std::collections::BTreeMap::new();
        for (i, line) in text.lines().enumerate() {
            if line.trim().is_empty() {
                continue;
            }
            // `split_once`, never `split`: see the type docs.
            let (key, rest) = line
                .split_once(':')
                .ok_or_else(|| CalibError::MissingColon {
                    path: path.to_path_buf(),
                    line_no: i + 1,
                    text: line.to_string(),
                })?;
            rows.insert(key.trim().to_string(), rest.to_string());
        }
        Ok(RawCalib {
            path: path.to_path_buf(),
            rows,
        })
    }

    /// `key`'s values as exactly `N` floats.
    fn values<const N: usize>(&self, key: &str) -> Result<[f64; N], CalibError> {
        let raw = self.rows.get(key).ok_or_else(|| CalibError::MissingField {
            path: self.path.clone(),
            key: key.to_string(),
        })?;
        let mut out = [0.0f64; N];
        let mut n = 0usize;
        // `split_whitespace`, so a stray tab or CR is absorbed rather than
        // becoming a `BadFloat` about an invisible character.
        for (i, tok) in raw.split_whitespace().enumerate() {
            let v: f64 = tok.parse().map_err(|e| CalibError::BadFloat {
                path: self.path.clone(),
                key: key.to_string(),
                index: i,
                text: tok.to_string(),
                source: e,
            })?;
            if i < N {
                out[i] = v;
            }
            n += 1;
        }
        if n != N {
            return Err(CalibError::Arity {
                path: self.path.clone(),
                key: key.to_string(),
                expected: N,
                got: n,
            });
        }
        Ok(out)
    }

    /// A 3x3 rotation, validated: proper, orthonormal, and forward-facing.
    fn rotation(&self, key: &str) -> Result<[[f64; 3]; 3], CalibError> {
        let v = self.values::<9>(key)?;
        let r = [[v[0], v[1], v[2]], [v[3], v[4], v[5]], [v[6], v[7], v[8]]];
        let det = r[0][0] * (r[1][1] * r[2][2] - r[1][2] * r[2][1])
            - r[0][1] * (r[1][0] * r[2][2] - r[1][2] * r[2][0])
            + r[0][2] * (r[1][0] * r[2][1] - r[1][1] * r[2][0]);
        let mut ortho_err = 0.0f64;
        for i in 0..3 {
            for j in 0..3 {
                let dot: f64 = (0..3).map(|k| r[i][k] * r[j][k]).sum();
                let want = if i == j { 1.0 } else { 0.0 };
                ortho_err = ortho_err.max((dot - want).abs());
            }
        }
        // `R @ (1,0,0)` is the first COLUMN, and its z component says whether
        // lidar-forward came out as camera-forward. This is the check a
        // transposed parse fails and the two above do not.
        let forward_z = r[2][0];
        // `R_rect_00` is a rectifying rotation about the optical axis, so its
        // (1,0,0) image is camera-RIGHT and its z component is ~0. Only the
        // extrinsic is asked to face forward.
        let needs_forward = key == "R";
        if (det - 1.0).abs() > 1e-6 || ortho_err > 1e-6 || (needs_forward && forward_z < 0.99) {
            return Err(CalibError::NotARotation {
                path: self.path.clone(),
                key: key.to_string(),
                det,
                ortho_err,
                forward_z,
            });
        }
        Ok(r)
    }
}

impl Calib {
    /// Reads `<root>/<date>/calib_velo_to_cam.txt` and `calib_cam_to_cam.txt`
    /// and collapses them to [`Calib::m`].
    ///
    /// One calibration set per **capture date**, shared by every drive of that
    /// date — not one per drive.
    ///
    /// There is deliberately no `Tr_velo_to_cam:` lookup. That key belongs to
    /// the *odometry* devkit's single `calib.txt`; this raw-data format has
    /// separate `R:` and `T:` rows, and code copied from the odometry
    /// benchmark finds nothing and either errors or silently uses an identity.
    pub fn load(root: &Path, date: &str) -> Result<Calib, CalibError> {
        let dir = root.join(date);
        let v2c = RawCalib::read(&dir.join("calib_velo_to_cam.txt"))?;
        let c2c = RawCalib::read(&dir.join("calib_cam_to_cam.txt"))?;

        let r = v2c.rotation("R")?;
        let t = v2c.values::<3>("T")?;
        // `R_rect_00` for every camera, which is what KITTI's own devkit does.
        // UNVERIFIED that it is right for `image_02` specifically rather than
        // `R_rect_02`: the difference is 4.98 px, which the edge-alignment
        // test cannot resolve at its ±2 px floor. Following the devkit, not
        // having confirmed it.
        let rr = c2c.rotation("R_rect_00")?;
        let p = c2c.values::<12>(&format!("P_rect_{CAM}"))?;
        let s = c2c.values::<2>(&format!("S_rect_{CAM}"))?;

        // rt = R_rect_00 @ [R | T], a 3x4.
        let mut rt = [[0.0f64; 4]; 3];
        for (i, row) in rt.iter_mut().enumerate() {
            for (j, cell) in row.iter_mut().enumerate().take(3) {
                *cell = (0..3).map(|k| rr[i][k] * r[k][j]).sum();
            }
            row[3] = (0..3).map(|k| rr[i][k] * t[k]).sum();
        }
        // m = P_rect_02 @ [rt; 0 0 0 1]. `P_rect_02` is 3x4 and its fourth
        // column is the baseline term the perspective divide needs -- see the
        // module docs for what dropping it costs.
        let mut m = [[0.0f64; 4]; 3];
        for (i, row) in m.iter_mut().enumerate() {
            for (j, cell) in row.iter_mut().enumerate() {
                let mut acc: f64 = (0..3).map(|k| p[i * 4 + k] * rt[k][j]).sum();
                if j == 3 {
                    acc += p[i * 4 + 3];
                }
                *cell = acc;
            }
        }
        Ok(Calib {
            m,
            width: s[0] as u32,
            height: s[1] as u32,
        })
    }

    /// Fails if the rectified size this calibration is for is not the size the
    /// drive's images actually are.
    ///
    /// Taken from `S_rect_02` and cross-checked against the PNG header the
    /// drive scanner already reads, never assumed: see [`CalibError::ImageSize`].
    pub fn check_image_size(&self, path: &Path, w: u32, h: u32) -> Result<(), CalibError> {
        if (self.width, self.height) == (w, h) {
            return Ok(());
        }
        Err(CalibError::ImageSize {
            path: path.to_path_buf(),
            calib: (self.width, self.height),
            drive: (w, h),
        })
    }

    /// `m @ (x, y, z, 1)`, before any divide.
    ///
    /// Returned as `(a, b, w)` so a caller can do the frustum test in
    /// homogeneous coordinates. See the module docs for why the divide is the
    /// last thing that should happen.
    pub fn homogeneous(&self, p: [f32; 3]) -> (f64, f64, f64) {
        let (x, y, z) = (f64::from(p[0]), f64::from(p[1]), f64::from(p[2]));
        let row = |r: &[f64; 4]| r[0] * x + r[1] * y + r[2] * z + r[3];
        (row(&self.m[0]), row(&self.m[1]), row(&self.m[2]))
    }

    /// Pixel of a lidar-frame point, or `None` if it is behind the camera
    /// plane or outside the image.
    ///
    /// The bounds test runs **before** the divide, in homogeneous coordinates:
    /// `w > 0 && 0 <= a < W*w && 0 <= b < H*w`. Verified elementwise identical
    /// to divide-then-bounds over sweep 0's 20,478 in-image points, one
    /// multiply cheaper per axis than a divide, and — the part that matters —
    /// it cannot produce a NaN or an overflow, because the divide only ever
    /// runs on a point already decided to be in frame.
    pub fn project(&self, p: [f32; 3]) -> Option<[f32; 2]> {
        let (a, b, w) = self.homogeneous(p);
        let (wf, hf) = (f64::from(self.width), f64::from(self.height));
        // Written as ONE positive test rather than as negated guards, so a
        // NaN in any of the three fails it instead of passing through a `!(x >
        // y)` that a NaN satisfies. The divide below therefore only ever runs
        // on a point already decided to be in frame.
        let in_frame = w > 0.0 && a >= 0.0 && a < wf * w && b >= 0.0 && b < hf * w;
        in_frame.then(|| [(a / w) as f32, (b / w) as f32])
    }

    /// The image rectangle a lidar-frame axis-aligned box occupies, clipped to
    /// the frame, or `None` if none of it is in frame.
    ///
    /// All **eight** corners are projected, not two: a box's image hull is not
    /// the projection of its own corners taken pairwise, because perspective
    /// is not axis-aligned. A corner is what a 2D box needs and a centroid
    /// cannot give, which is why [`crate::detect::DETECTION_BYTES`] carries
    /// both corners in the first place.
    ///
    /// Refused outright — `None`, not clipped — if **any** corner is nearer
    /// than [`W_MIN`] to the camera plane. A box straddling that plane has no
    /// finite image hull (its projection runs to infinity in some direction),
    /// and clipping the corners that happen to be in front would report a
    /// confident small rectangle for an object wrapped around the camera. No
    /// detection this pipeline produces can be there.
    pub fn project_box(&self, lo: [f32; 3], hi: [f32; 3]) -> Option<ImageBox> {
        let (mut x0, mut y0) = (f64::INFINITY, f64::INFINITY);
        let (mut x1, mut y1) = (f64::NEG_INFINITY, f64::NEG_INFINITY);
        for c in 0..8u8 {
            let p = [
                if c & 1 == 0 { lo[0] } else { hi[0] },
                if c & 2 == 0 { lo[1] } else { hi[1] },
                if c & 4 == 0 { lo[2] } else { hi[2] },
            ];
            let (a, b, w) = self.homogeneous(p);
            // Positive, for the reason `project` gives: a NaN coordinate must
            // refuse the box rather than slip past a negated comparison.
            let ahead = w > W_MIN;
            if !ahead {
                return None;
            }
            let (u, v) = (a / w, b / w);
            x0 = x0.min(u);
            x1 = x1.max(u);
            y0 = y0.min(v);
            y1 = y1.max(v);
        }
        let (wf, hf) = (f64::from(self.width), f64::from(self.height));
        let cx0 = x0.max(0.0);
        let cy0 = y0.max(0.0);
        let cx1 = x1.min(wf);
        let cy1 = y1.min(hf);
        // Same rule once more: a clipped rectangle with no area, or one whose
        // corners are NaN, is not in frame.
        let has_area = cx1 > cx0 && cy1 > cy0;
        has_area.then_some(ImageBox {
            x0: cx0 as f32,
            y0: cy0 as f32,
            x1: cx1 as f32,
            y1: cy1 as f32,
        })
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use std::path::PathBuf;

    use super::*;

    /// The two files, written to a temp directory so the unit tests need no
    /// dataset. Byte-for-byte the rows this project's `2011_09_26` set holds,
    /// including `calib_time`'s colons.
    const VELO_TO_CAM: &str = "calib_time: 15-Mar-2012 11:37:16\n\
R: 7.533745e-03 -9.999714e-01 -6.166020e-04 1.480249e-02 7.280733e-04 -9.998902e-01 9.998621e-01 7.523790e-03 1.480755e-02\n\
T: -4.069766e-03 -7.631618e-02 -2.717806e-01\n\
delta_f: 0.000000e+00 0.000000e+00\n\
delta_c: 0.000000e+00 0.000000e+00\n";

    const CAM_TO_CAM: &str = "calib_time: 09-Jan-2012 13:57:47\n\
corner_dist: 9.950000e-02\n\
S_rect_00: 1.242000e+03 3.750000e+02\n\
R_rect_00: 9.999239e-01 9.837760e-03 -7.445048e-03 -9.869795e-03 9.999421e-01 -4.278459e-03 7.402527e-03 4.351614e-03 9.999631e-01\n\
S_rect_02: 1.242000e+03 3.750000e+02\n\
P_rect_02: 7.215377e+02 0.000000e+00 6.095593e+02 4.485728e+01 0.000000e+00 7.215377e+02 1.728540e+02 2.163791e-01 0.000000e+00 0.000000e+00 1.000000e+00 2.745884e-03\n";

    fn write_calib(dir: &Path, v2c: &str, c2c: &str) {
        std::fs::create_dir_all(dir.join("d")).unwrap();
        std::fs::write(dir.join("d/calib_velo_to_cam.txt"), v2c).unwrap();
        std::fs::write(dir.join("d/calib_cam_to_cam.txt"), c2c).unwrap();
    }

    /// A directory no other test shares.
    ///
    /// The process id alone was not enough and the failure was instructive:
    /// every test here runs in ONE process, `cargo test` runs them in
    /// parallel, and `good()` both removed and rewrote its directory — so two
    /// tests with the same name deleted each other's files mid-read and the
    /// matrix test failed only in a full run. The counter makes each call's
    /// directory unique regardless of the name it is given.
    fn tmp(name: &str) -> PathBuf {
        static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        let n = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let d = std::env::temp_dir().join(format!("pipes-calib-{name}-{}-{n}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        d
    }

    fn good() -> (PathBuf, Calib) {
        let d = tmp("good");
        write_calib(&d, VELO_TO_CAM, CAM_TO_CAM);
        let c = Calib::load(&d, "d").unwrap();
        (d, c)
    }

    /// All twelve entries, against the matrix the investigation measured with
    /// numpy from the same files. **This is the test that catches every
    /// shortcut the module docs list**, because each of them still produces an
    /// overlay that looks right.
    #[test]
    fn calib_matrix_matches_the_measured_chain() {
        let (_d, c) = good();
        let want = [
            [
                6.09695409e+02,
                -7.21421597e+02,
                -1.25125855e+00,
                -1.23041806e+02,
            ],
            [
                1.80384202e+02,
                7.64479802e+00,
                -7.19651474e+02,
                -1.01016688e+02,
            ],
            [
                9.99945389e-01,
                1.24365378e-04,
                1.04513030e-02,
                -2.69386912e-01,
            ],
        ];
        for (i, (got, exp)) in c.m.iter().zip(want.iter()).enumerate() {
            for (j, (g, e)) in got.iter().zip(exp.iter()).enumerate() {
                assert!(
                    (g - e).abs() <= e.abs() * 1e-6 + 1e-6,
                    "m[{i}][{j}] = {g}, expected {e}"
                );
            }
        }
        assert_eq!((c.width, c.height), (1242, 375));
    }

    /// Positive control for the test below: the forward ray really does land
    /// on the principal point, so "projects to nothing" is not this module's
    /// default answer.
    #[test]
    fn the_forward_ray_lands_on_the_principal_point() {
        let (_d, c) = good();
        // 30 m straight ahead, at the height of the optical centre.
        let px = c
            .project([30.0, 0.0, -0.07])
            .expect("forward point is in frame");
        assert!((px[0] - 609.7).abs() < 2.0, "u = {}", px[0]);
        assert!((px[1] - 180.4).abs() < 3.0, "v = {}", px[1]);
    }

    /// The whole reason the sign test exists: a point BEHIND the car divides
    /// to a pixel inside the frame, and only `w` says so.
    #[test]
    fn a_point_behind_the_camera_is_refused_although_its_pixel_is_in_frame() {
        let (_d, c) = good();
        // 20 m behind, a metre up: `w` is negative, and dividing by it mirrors
        // the point through the optical centre into the top of the image.
        let p = [-20.0f32, 0.0, 1.0];
        let (a, b, w) = c.homogeneous(p);
        assert!(w < 0.0, "w = {w}");
        let (u, v) = (a / w, b / w);
        assert!(
            (0.0..1242.0).contains(&u) && (0.0..375.0).contains(&v),
            "this test is pointless unless the fake pixel is in frame: ({u}, {v})"
        );
        assert_eq!(c.project(p), None, "a point behind the camera was accepted");
    }

    /// The homogeneous test and divide-then-bounds must select the SAME
    /// points, or the cheap one is a different filter wearing the same name.
    #[test]
    fn the_homogeneous_bounds_test_agrees_with_dividing_first() {
        let (_d, c) = good();
        let mut checked = 0u32;
        let mut kept = 0u32;
        for xi in 0..40 {
            for yi in -20..20 {
                for zi in -4..4 {
                    let p = [xi as f32 - 5.0, yi as f32 * 0.7, zi as f32 * 0.6];
                    let (a, b, w) = c.homogeneous(p);
                    let slow = if w > 0.0 {
                        let (u, v) = (a / w, b / w);
                        ((0.0..f64::from(c.width)).contains(&u)
                            && (0.0..f64::from(c.height)).contains(&v))
                        .then_some([u as f32, v as f32])
                    } else {
                        None
                    };
                    let fast = c.project(p);
                    assert_eq!(fast.is_some(), slow.is_some(), "disagreed at {p:?}");
                    checked += 1;
                    kept += u32::from(fast.is_some());
                }
            }
        }
        // Positive control: an agreement over a set where nothing was ever
        // kept would be vacuous.
        assert!(kept > 100, "only {kept} of {checked} points were in frame");
    }

    /// A box in front of the camera gets a rectangle; the same box behind it
    /// gets nothing. Both halves, because "returns None" passes trivially
    /// against a function that always does.
    #[test]
    fn project_box_frames_what_is_ahead_and_refuses_what_is_behind() {
        let (_d, c) = good();
        let b = c
            .project_box([9.0, -1.0, -1.7], [13.0, 1.0, 0.0])
            .expect("a car-sized box 10 m ahead is in frame");
        assert!(b.x0 >= 0.0 && b.y0 >= 0.0);
        assert!(b.x1 <= 1242.0 && b.y1 <= 375.0);
        assert!(b.w() > 1.0 && b.h() > 1.0, "{b:?} has no area");
        // The same box 10 m behind.
        assert_eq!(c.project_box([-13.0, -1.0, -1.7], [-9.0, 1.0, 0.0]), None);
        // And one that straddles the camera plane: refused, not clipped.
        assert_eq!(c.project_box([-2.0, -1.0, -1.7], [2.0, 1.0, 0.0]), None);
    }

    /// A nearer box of the same size must be BIGGER on the image. If
    /// `project_box` ignored depth this would not hold.
    #[test]
    fn a_nearer_box_is_larger_in_the_image() {
        let (_d, c) = good();
        let near = c.project_box([5.0, -1.0, -1.7], [7.0, 1.0, 0.0]).unwrap();
        let far = c.project_box([25.0, -1.0, -1.7], [27.0, 1.0, 0.0]).unwrap();
        assert!(
            near.h() > far.h() * 2.0,
            "near {near:?} is not much bigger than far {far:?}"
        );
    }

    /// `calib_time`'s value contains colons. A `split(':')` reader gets four
    /// pieces on that line; this asserts the file parses anyway.
    #[test]
    fn a_colon_in_a_value_does_not_break_the_reader() {
        let (_d, c) = good();
        assert_eq!(c.width, 1242);
    }

    /// A missing field must be an error, because the plausible default is the
    /// dangerous outcome: an identity for `R_rect_00` costs 6.58 px and ships.
    #[test]
    fn a_missing_field_is_an_error_naming_the_file_and_the_key() {
        let d = tmp("missing");
        let without = CAM_TO_CAM
            .lines()
            .filter(|l| !l.starts_with("R_rect_00:"))
            .collect::<Vec<_>>()
            .join("\n");
        write_calib(&d, VELO_TO_CAM, &without);
        match Calib::load(&d, "d") {
            Err(CalibError::MissingField { path, key }) => {
                assert_eq!(key, "R_rect_00");
                assert!(path.ends_with("calib_cam_to_cam.txt"), "{path:?}");
            }
            other => panic!("expected MissingField, got {other:?}"),
        }
    }

    /// The check orthonormality cannot make. A transposed `R` is still a
    /// proper rotation; only the forward test rejects it.
    #[test]
    fn a_transposed_rotation_is_rejected_although_it_is_still_a_rotation() {
        let d = tmp("transposed");
        let v: Vec<f64> = VELO_TO_CAM
            .lines()
            .find(|l| l.starts_with("R:"))
            .and_then(|l| l.split_once(':'))
            .map(|(_, r)| {
                r.split_whitespace()
                    .filter_map(|t| t.parse().ok())
                    .collect()
            })
            .unwrap();
        let t: Vec<String> = [0, 3, 6, 1, 4, 7, 2, 5, 8]
            .iter()
            .map(|&i| format!("{:e}", v[i]))
            .collect();
        let swapped = VELO_TO_CAM.replace(
            VELO_TO_CAM.lines().find(|l| l.starts_with("R:")).unwrap(),
            &format!("R: {}", t.join(" ")),
        );
        write_calib(&d, &swapped, CAM_TO_CAM);
        match Calib::load(&d, "d") {
            Err(CalibError::NotARotation {
                det,
                ortho_err,
                forward_z,
                ..
            }) => {
                // The point of the test: the two checks that do NOT catch it.
                assert!((det - 1.0).abs() < 1e-6, "det {det} should still be 1");
                assert!(ortho_err < 1e-6, "ortho_err {ortho_err} should still be 0");
                assert!(forward_z < 0.99, "forward_z {forward_z} is what catches it");
            }
            other => panic!("expected NotARotation, got {other:?}"),
        }
    }

    /// Wrong arity is an error rather than a truncation.
    #[test]
    fn a_short_row_is_an_arity_error() {
        let d = tmp("arity");
        let short = CAM_TO_CAM.replace(
            "P_rect_02: 7.215377e+02 0.000000e+00 6.095593e+02 4.485728e+01",
            "P_rect_02: 7.215377e+02 0.000000e+00 6.095593e+02",
        );
        write_calib(&d, VELO_TO_CAM, &short);
        match Calib::load(&d, "d") {
            Err(CalibError::Arity {
                key, expected, got, ..
            }) => {
                assert_eq!((key.as_str(), expected, got), ("P_rect_02", 12, 11));
            }
            other => panic!("expected Arity, got {other:?}"),
        }
    }

    /// A line with no colon at all is an error, not a skipped line.
    #[test]
    fn a_line_without_a_colon_is_an_error() {
        let d = tmp("nocolon");
        write_calib(&d, &format!("{VELO_TO_CAM}garbage line\n"), CAM_TO_CAM);
        match Calib::load(&d, "d") {
            Err(CalibError::MissingColon { line_no, text, .. }) => {
                assert_eq!(line_no, 6);
                assert_eq!(text, "garbage line");
            }
            other => panic!("expected MissingColon, got {other:?}"),
        }
    }

    /// A non-numeric value in a row the projection needs names the row and the
    /// index.
    #[test]
    fn a_bad_float_names_its_key_and_index() {
        let d = tmp("badfloat");
        let bad = VELO_TO_CAM.replace("T: -4.069766e-03", "T: nope");
        write_calib(&d, &bad, CAM_TO_CAM);
        match Calib::load(&d, "d") {
            Err(CalibError::BadFloat {
                key, index, text, ..
            }) => {
                assert_eq!((key.as_str(), index, text.as_str()), ("T", 0, "nope"));
            }
            other => panic!("expected BadFloat, got {other:?}"),
        }
    }

    /// The image size comes from `S_rect_02` and is cross-checked, never
    /// assumed: a hardcoded 1242x375 is wrong on three of KITTI's five dates.
    #[test]
    fn a_resolution_mismatch_is_an_error() {
        let (_d, c) = good();
        let p = PathBuf::from("calib_cam_to_cam.txt");
        assert!(c.check_image_size(&p, 1242, 375).is_ok());
        match c.check_image_size(&p, 1224, 370) {
            Err(CalibError::ImageSize { calib, drive, .. }) => {
                assert_eq!((calib, drive), ((1242, 375), (1224, 370)));
            }
            other => panic!("expected ImageSize, got {other:?}"),
        }
    }
}
