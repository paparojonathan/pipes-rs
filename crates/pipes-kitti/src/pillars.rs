//! The lidar's frozen detector: PointPillars, run on each raw sweep in a
//! stage of its own, `pillars`, and the `PILLAR_DET` batch its boxes travel
//! in.
//!
//! [`crate::detect`] groups a sweep into connected structures and refuses to
//! name them: a guessed label is exactly the plausible-looking wrongness that
//! module exists to refuse. This module names what a trained network can,
//! and nothing else. PointPillars (Lang et al., 2019) is a 3D detector that
//! reads the sweep's points in vertical columns (pillars), turns each column
//! into one 64-wide feature, scatters those into a bird's-eye image, and runs
//! a 2D convolutional network over it; its output is a box with a heading
//! for each **car, pedestrian and cyclist** it finds, in metres, in the
//! lidar's own frame. On a bush, a wall or a fragment of either it finds
//! nothing, which is the point: a structure the camera cannot see and this
//! network does not name is, with good probability, not a road user.
//!
//! **Frozen**, on the camera detector's terms ([`crate::camdet`]): the
//! weights are one file whose sha256 is pinned here ([`MODEL_SHA256`]) and
//! compared at load; nothing learns; the preprocessing and the decoding are
//! the model's own, not this project's. The weights are OpenPCDet's
//! `pointpillar_7728` checkpoint (KITTI, 3 classes), as NVIDIA exports it to
//! ONNX in CUDA-PointPillars ([`SOURCE_SHA256`]); that export carries one
//! TensorRT-only node, the pillar scatter, which `scripts/prepare_models.py`
//! rewrites as the standard `ScatterND` so ONNX Runtime can run the graph,
//! and makes the pillar count a free dimension. No weight changes. The
//! derived file is what `pipes` loads, and its hash is the pinned one. On
//! the GPU it loads the same graph with its weights as 16-bit floats
//! ([`MODEL_FILE_HALF`]), derived by the same script and pinned the same
//! way: the network runs in three quarters of the time and, scored against
//! KITTI's labelled boxes on the four drives, finds the same objects.
//!
//! # What the model is given, and why that is this module's job
//!
//! The network starts at its first linear layer; everything before it --
//! which points go in which pillar, and the ten numbers that describe each
//! point -- is preprocessing that NVIDIA does in a CUDA kernel and this
//! module does on the CPU, in Rust, deterministically. The numbers are the
//! model's ([`RANGE_MIN`], [`RANGE_MAX`], [`PILLAR_EDGE_M`],
//! [`POINTS_PER_PILLAR`]): the range is the front 69 m and 40 m to either
//! side, which is the region KITTI's object labels cover and the region the
//! network was trained to look at, so **one pass of it sees nothing behind
//! the car** (the next section); a pillar is 16 cm square, the whole height
//! of the range; a pillar
//! keeps its first 32 points in the sweep's own order, exactly as the
//! training-time voxeliser did (NVIDIA's GPU kernel keeps a random 32, and
//! says so); the ten features are the point, its offset from its pillar's
//! mean and its offset from its pillar's centre. Padding is zeros, which the
//! network saw in training too.
//!
//! The decoding is NVIDIA's postprocess, number for number: six anchors per
//! cell of the 248x216 output ([`ANCHOR_SIZES`], two headings each), the
//! residual box code, the two-bin heading, the sigmoid score and
//! [`SCORE_THRESHOLD`], and a bird's-eye rotated-box NMS at [`NMS_IOU`]
//! (0.01: any overlap suppresses). Checked against a numpy reference of the
//! same arithmetic on drive_0005 sweeps 0 and 140 when the stage was built:
//! on sweep 140 the network's first box is a car whose near face is 7.6 m
//! ahead, which is the object every earlier run's answer named.
//!
//! # Two views of one sweep
//!
//! The lidar turns all the way round, and half of every sweep is behind the
//! car, where a network trained on the road ahead never looks.
//! [`View::Behind`] gives the network that half **turned half round about
//! the vertical**, `(-x, -y, z)`, so what was behind is ahead and looks like
//! any road it was trained on, and turns each box it finds back
//! ([`LidarDet::turned`]). A run that asks for both views ([`View::AROUND`])
//! runs the network twice a sweep and suppresses overlaps over the two
//! answers together, so a car beside the vehicle, cut by the line between
//! the views, is one box.
//!
//! Nothing behind the car is labelled, so what the second view is worth was
//! measured another way: a parked car stays where KITTI labelled it, and
//! its box, carried through the vehicle's own motion into the sweeps after
//! the vehicle has passed it, is ground truth behind the car. Over the four
//! drives the turned view found 93 % of such cars within 15 m, 73 % between
//! 15 and 30 m and 25 % beyond that; the last two are lower bounds, a
//! parked car far behind being mostly hidden by the nearer ones. Ahead,
//! where the labels say what can be seen, the network finds 91 % of the
//! vehicles (`scripts/score_detectors.py`).
//!
//! # Determinism
//!
//! On the terms of [`crate::detect`]: no map, no randomness. Points are
//! binned by a sort on `(pillar key, point index)`, so a pillar's points are
//! in sweep order and the pillars are in key order, independent of anything
//! but the sweep itself. Candidates are sorted by `(score, view, row,
//! column, anchor)`, a total order, before the NMS. The network itself runs on ONNX
//! Runtime -- on the CPU, where it repeats to the byte, or on the GPU with
//! cuDNN's algorithms chosen by rule ([`crate::camdet::CudaOptions`]), where
//! it repeats on one machine -- and `tract` is refused: at 70 GFLOP a sweep
//! the network is a GPU's job, and one CPU thread would take seconds.
//!
//! # What it is not
//!
//! Not a tracker's input and not the answer's: the structures the chain
//! tracks are still `detect`'s, with their distances and velocities; a box
//! from here gives a track a class ([`crate::name`]) and nothing else, so a
//! wrong box can put a wrong name on a right distance and never a wrong
//! distance anywhere. Not a second opinion behind the car: the cameras look
//! ahead, so a name given there is the lidar's alone. And a run is not a
//! measurement of accuracy: it reports how many tracks it named and how
//! often the cameras agreed, which is evidence, not a score. The score is
//! `scripts/score_detectors.py`'s, against KITTI's labelled boxes.

use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, PoisonError};

use arrow::array::{
    Array, ArrayRef, AsArray, FixedSizeListArray, Float32Array, Int64Array, LargeListArray,
    RecordBatch, StringArray, UInt32Array,
};
use arrow::buffer::{Buffer, MutableBuffer, OffsetBuffer, ScalarBuffer};
use arrow::datatypes::{DataType, Field, FieldRef, Float32Type, Int64Type, Schema, UInt32Type};

use crate::camdet::{sha256_hex, Backend, CudaOptions};
use crate::onnx::ort_session;
use crate::velo::SweepError;

/// The name `--lidar-detector` runs and `run.json` records.
pub const MODEL_NAME: &str = "pointpillars-kitti";

/// The derived file, relative to the working directory: the one the `ort`
/// backend loads.
pub const MODEL_FILE: &str = "models/pointpillars-kitti.onnx";

/// sha256 of [`MODEL_FILE`].
pub const MODEL_SHA256: &str = "8a6c857492799b297d500c5e33424b9e870326069e59d122e87df4e3ff657011";

/// The same graph with its weights and inner tensors as 16-bit floats, its
/// inputs and outputs still 32-bit: the one the `cuda` backend loads.
pub const MODEL_FILE_HALF: &str = "models/pointpillars-kitti-fp16.onnx";

/// sha256 of [`MODEL_FILE_HALF`].
pub const MODEL_SHA256_HALF: &str =
    "999e30dcca49ef7f35f0b0c4ddb493ab470b23de3c3554ee91fc7ab9bd748237";

/// The file a backend loads and its pinned sha256: 16-bit weights on the
/// GPU, which runs them in three quarters of the time; 32-bit on a CPU, which
/// has no 16-bit arithmetic and would convert every tensor back.
pub fn model_for(backend: Backend) -> (&'static str, &'static str) {
    match backend {
        Backend::Cuda => (MODEL_FILE_HALF, MODEL_SHA256_HALF),
        Backend::Ort | Backend::Tract => (MODEL_FILE, MODEL_SHA256),
    }
}

/// Which half of the sweep a pass of the network reads. See the module
/// docs. Ordered, ahead first: part of the candidates' total order.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum View {
    /// The sweep as it is: the 69 m ahead of the sensor.
    #[default]
    Ahead,
    /// The sweep turned half round about the vertical: the 69 m behind.
    Behind,
}

impl View {
    /// Both views, in the order a sweep is read: all the way round.
    pub const AROUND: [View; 2] = [View::Ahead, View::Behind];

    /// The view's name in `run.json` and the batch's metadata.
    pub fn name(self) -> &'static str {
        match self {
            View::Ahead => "ahead",
            View::Behind => "behind",
        }
    }

    /// What x and y are multiplied by to turn the sweep to this view. The
    /// view ahead multiplies by one, which changes no bit.
    fn sign(self) -> f32 {
        match self {
            View::Ahead => 1.0,
            View::Behind => -1.0,
        }
    }
}

/// `views`, as the words `ahead, behind`.
pub fn views_words(views: &[View]) -> String {
    views
        .iter()
        .map(|v| v.name())
        .collect::<Vec<_>>()
        .join(", ")
}

/// NVIDIA's export, in `models/`, that the derived file is made from.
pub const SOURCE_FILE: &str = "models/pointpillar.onnx";

/// sha256 of [`SOURCE_FILE`].
pub const SOURCE_SHA256: &str = "b48eada794191301b1255a327542b772152fd9a9364b2c7c184e11e58bcbf517";

/// Parameters, millions, from the model zoo.
pub const PARAMS_M: f32 = 4.8;

/// The three classes, by id: KITTI's.
pub const CLASSES: [&str; 3] = ["car", "pedestrian", "cyclist"];

/// Class name of an id; `"unknown"` past the three.
pub fn class_name(id: u32) -> &'static str {
    CLASSES.get(id as usize).copied().unwrap_or("unknown")
}

/// The near corner of the region the network reads, metres in the lidar
/// frame: ahead of the sensor, 39.68 m to the right, 3 m below.
pub const RANGE_MIN: [f32; 3] = [0.0, -39.68, -3.0];

/// The far corner: 69.12 m ahead, 39.68 m to the left, 1 m above. A point on
/// the far face is outside (half-open), as the model's voxeliser has it.
pub const RANGE_MAX: [f32; 3] = [69.12, 39.68, 1.0];

/// A pillar's edge in x and y, metres: `(69.12 - 0) / 432 = (39.68 * 2) /
/// 496 = 0.16`.
pub const PILLAR_EDGE_M: f32 = 0.16;

/// A pillar's height: the whole range, so a pillar is a column.
pub const PILLAR_HEIGHT_M: f32 = 4.0;

/// Pillars along x (ahead) and y (left) in the grid the network reads.
pub const GRID_X: usize = 432;

/// See [`GRID_X`].
pub const GRID_Y: usize = 496;

/// Points a pillar keeps: its first 32 in the sweep's order.
pub const POINTS_PER_PILLAR: usize = 32;

/// Numbers per point the network reads: x, y, z, reflectance, the offset
/// from the pillar's mean (3) and from the pillar's centre (3).
pub const FEATURES_PER_POINT: usize = 10;

/// Pillars a sweep may hand the network, NVIDIA's own cap: a KITTI sweep
/// fills 7,000 to 13,000 in the range, so the cap is a bound rather than a
/// crop, and a sweep past it keeps its lowest keys, which is nearest the
/// right-hand edge of the range -- counted, so a run that hit it says so.
pub const MAX_PILLARS: usize = 40_000;

/// The output grid: the pillar grid at stride 2.
pub const FEATURE_X: usize = GRID_X / 2;

/// See [`FEATURE_X`].
pub const FEATURE_Y: usize = GRID_Y / 2;

/// Anchors per output cell: three classes, two headings each.
pub const ANCHORS_PER_CELL: usize = 6;

/// Anchor box length, width and height per class, metres, the model's.
pub const ANCHOR_SIZES: [[f32; 3]; 3] = [[3.9, 1.6, 1.56], [0.8, 0.6, 1.73], [1.76, 0.6, 1.73]];

/// Height of each class's anchor bottom above the lidar, metres: the box's
/// centre is half its height above that.
pub const ANCHOR_BOTTOM_M: [f32; 3] = [-1.78, -0.6, -0.6];

/// The two anchor headings, radians: along x and across it. `1.57` rather
/// than `PI / 2`, because that is the number the model was trained with.
pub const ANCHOR_YAW: [f32; 2] = [0.0, 1.57];

/// A box is kept when its best class's sigmoid score reaches this.
pub const SCORE_THRESHOLD: f32 = 0.1;

/// Bird's-eye rotated IoU at which a lower-scored box is suppressed: any
/// real overlap.
pub const NMS_IOU: f32 = 0.01;

/// A class logit under this cannot reach [`SCORE_THRESHOLD`]: the sigmoid is
/// monotone and `sigmoid(-2.3) = 0.091`, under 0.1 with room for rounding
/// (the threshold's own logit is `ln(1/9) = -2.197`). [`decode`] skips an
/// anchor whose three logits are all under it without computing a sigmoid,
/// which is 321,408 anchors less a few hundred, every sweep; the anchors
/// that pass are scored exactly as before.
const LOGIT_FLOOR: f32 = -2.3;

/// The heading decoder's offset, radians: the model's own literal, not
/// `PI / 4` (which it approximates), because the network was trained and
/// evaluated with this number and a decoder that used the exact constant
/// would be a different decoder.
#[allow(clippy::approx_constant)]
const DIR_OFFSET: f32 = 0.78539;

/// f32 lanes in one lidar detection. See [`PILLAR_DET_BYTES`].
pub const PILLAR_DET_VALUES: i32 = 9;

/// [`PILLAR_DET_VALUES`] as a `usize`.
pub const PILLAR_DET_LANES: usize = PILLAR_DET_VALUES as usize;

/// Bytes in one lidar detection: 9 f32 lanes, in this order.
///
/// | lanes | field |
/// |---|---|
/// | 0-2 | `centre` x, y, z of the box, metres in the lidar frame |
/// | 3-5 | `size` along the heading, across it, and up |
/// | 6 | `yaw` of the heading from +x towards +y, radians |
/// | 7 | `class_id`, an index into [`CLASSES`], exact in an f32 lane |
/// | 8 | `score`, in [[`SCORE_THRESHOLD`], 1] |
///
/// One f32 buffer per batch for the reason every other payload in this
/// project is one: one address, one `storage_id`, and the zero-copy proof
/// extends to this stream with no new machinery.
pub const PILLAR_DET_BYTES: usize = 36;

/// Value of the `format` column: what the 9 lanes mean. `c3` centre, `s3`
/// size, `y1` yaw, `k1` class, `q1` score. Shares no spelling with the other
/// formats, so nothing downstream can read a box as a point, a cluster or a
/// track.
pub const PILLAR_DET_FORMAT: &str = "pdet_c3s3y1k1q1_f32le";

/// One box the network found: a car, a pedestrian or a cyclist, with its
/// heading.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct LidarDet {
    /// Centre of the box, metres in the lidar frame.
    pub x: f32,
    /// See `x`.
    pub y: f32,
    /// See `x`.
    pub z: f32,
    /// Extent along the heading.
    pub dx: f32,
    /// Extent across it.
    pub dy: f32,
    /// Height.
    pub dz: f32,
    /// Heading, radians from +x towards +y.
    pub yaw: f32,
    /// Index into [`CLASSES`].
    pub class_id: u32,
    /// The best class's sigmoid score.
    pub score: f32,
}

impl LidarDet {
    /// The nine lanes, in [`PILLAR_DET_BYTES`]' order.
    pub fn lanes(&self) -> [f32; PILLAR_DET_LANES] {
        [
            self.x,
            self.y,
            self.z,
            self.dx,
            self.dy,
            self.dz,
            self.yaw,
            self.class_id as f32,
            self.score,
        ]
    }

    /// A box read back out of its lanes.
    pub fn from_lanes(l: &[f32; PILLAR_DET_LANES]) -> LidarDet {
        LidarDet {
            x: l[0],
            y: l[1],
            z: l[2],
            dx: l[3],
            dy: l[4],
            dz: l[5],
            yaw: l[6],
            class_id: l[7] as u32,
            score: l[8],
        }
    }

    /// The box's footprint in the ground plane: its four corners, counter-
    /// clockwise from the rear-right one, metres.
    pub fn footprint(&self) -> [[f32; 2]; 4] {
        let (s, c) = self.yaw.sin_cos();
        let (hx, hy) = (0.5 * self.dx, 0.5 * self.dy);
        let corner = |u: f32, v: f32| [self.x + u * c - v * s, self.y + u * s + v * c];
        [
            corner(-hx, -hy),
            corner(hx, -hy),
            corner(hx, hy),
            corner(-hx, hy),
        ]
    }

    /// Whether a point of the ground plane lies inside the footprint, tested
    /// in the box's own frame and written positively, so a NaN anywhere is
    /// outside.
    pub fn contains_bev(&self, px: f32, py: f32) -> bool {
        let (s, c) = self.yaw.sin_cos();
        let (dx, dy) = (px - self.x, py - self.y);
        let u = dx * c + dy * s;
        let v = -dx * s + dy * c;
        u.abs() <= 0.5 * self.dx && v.abs() <= 0.5 * self.dy
    }

    /// The box's nearest extent ahead of the sensor, metres: the smallest x
    /// of its footprint. The same thing a cluster's `bbox_min.x` is.
    pub fn near_face_m(&self) -> f32 {
        self.footprint()
            .iter()
            .map(|p| p[0])
            .fold(f32::INFINITY, f32::min)
    }

    /// The footprint's area, square metres.
    pub fn area_bev(&self) -> f32 {
        self.dx * self.dy
    }

    /// The same box in a frame turned half round about the vertical: a box
    /// found in the sweep turned that way, put back where it is. Its heading
    /// stays in the range the decoder gives one.
    pub fn turned(self) -> LidarDet {
        let full = std::f32::consts::TAU;
        let yaw = self.yaw + std::f32::consts::PI;
        LidarDet {
            x: -self.x,
            y: -self.y,
            yaw: if yaw >= DIR_OFFSET + full {
                yaw - full
            } else {
                yaw
            },
            ..self
        }
    }
}

/// Why a model could not be loaded or run.
#[derive(Debug)]
pub enum PillarsError {
    /// The model file is not there.
    Missing(PathBuf),
    /// The model file could not be read.
    Io(PathBuf, std::io::Error),
    /// The model file is not the pinned one.
    Sha256Mismatch {
        /// The file that was read.
        path: PathBuf,
        /// Its sha256.
        got: String,
        /// The pinned one.
        expected: &'static str,
    },
    /// `tract`: refused, see the module docs.
    Backend(Backend),
    /// The runtime refused the model or a sweep.
    Runtime(String),
    /// The network's output was not the shape the decoder reads.
    Output(String),
    /// The batch could not be built.
    Sweep(SweepError),
}

impl std::fmt::Display for PillarsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PillarsError::Missing(p) => write!(
                f,
                "lidar detector model {0} not found; `docker compose build` derives it, or run \
                 `python3 scripts/prepare_models.py {1}`",
                p.display(),
                p.file_stem().and_then(|s| s.to_str()).unwrap_or(MODEL_NAME)
            ),
            PillarsError::Io(p, e) => write!(f, "reading {}: {e}", p.display()),
            PillarsError::Sha256Mismatch {
                path,
                got,
                expected,
            } => write!(
                f,
                "{} is not the pinned model: sha256 {got}, expected {expected}",
                path.display()
            ),
            PillarsError::Backend(b) => write!(
                f,
                "the lidar detector does not run on `{}`: at 70 GFLOP a sweep the network needs \
                 ONNX Runtime (--lidar-backend ort) or the GPU (--lidar-backend cuda)",
                b.name()
            ),
            PillarsError::Runtime(s) => write!(f, "lidar detector: {s}"),
            PillarsError::Output(s) => write!(f, "lidar detector output: {s}"),
            PillarsError::Sweep(e) => write!(f, "lidar detector batch: {e}"),
        }
    }
}

impl std::error::Error for PillarsError {}

impl From<SweepError> for PillarsError {
    fn from(e: SweepError) -> Self {
        PillarsError::Sweep(e)
    }
}

/// What one sweep's pillars were: the counts a run reports beside its
/// timings, so a sweep that hit a cap or carried junk says so.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PillarPlan {
    /// Points in the sweep.
    pub source_points: u32,
    /// Points inside the range, and so in some pillar.
    pub in_range: u32,
    /// Points with a non-finite coordinate, skipped.
    pub non_finite: u32,
    /// Pillars with at least one point, before the cap.
    pub pillars_found: u32,
    /// Pillars handed to the network: `pillars_found` up to [`MAX_PILLARS`].
    pub pillars: u32,
    /// Points past the 32nd of their pillar, which the pillar does not keep.
    pub truncated_points: u32,
    /// Points in pillars past the cap, not handed to the network at all.
    pub capped_points: u32,
}

impl PillarPlan {
    /// These counts with another view's of the same sweep added. The sweep's
    /// points, and those of them that are not numbers, are the same points
    /// both times and are counted once; the rest is each view's own, a point
    /// at x = 0 exactly being in both.
    pub fn with(self, other: PillarPlan) -> PillarPlan {
        PillarPlan {
            source_points: self.source_points.max(other.source_points),
            in_range: self.in_range + other.in_range,
            non_finite: self.non_finite.max(other.non_finite),
            pillars_found: self.pillars_found + other.pillars_found,
            pillars: self.pillars + other.pillars,
            truncated_points: self.truncated_points + other.truncated_points,
            capped_points: self.capped_points + other.capped_points,
        }
    }
}

/// The pillars of one view of one sweep, in the network's input layout, and
/// the scratch that builds them. Reused across views and sweeps;
/// [`Pillarizer::reserve`] sizes it outside the measured window.
pub struct Pillarizer {
    /// `(pillar key, point index)` of every in-range point, sorted.
    cells: Vec<(u32, u32)>,
    /// `pillars x 32 x 10`, the network's `voxels` input, zero-padded.
    features: Vec<f32>,
    /// `pillars x 1`, the network's `pillar_index` input: `y * GRID_X + x`.
    index: Vec<i64>,
    /// Pillars in the two above.
    n: usize,
    /// Which way the sweep was read to make them.
    view: View,
}

impl Default for Pillarizer {
    fn default() -> Self {
        Self::new()
    }
}

impl Pillarizer {
    /// Empty scratch.
    pub fn new() -> Pillarizer {
        Pillarizer {
            cells: Vec::new(),
            features: Vec::new(),
            index: Vec::new(),
            n: 0,
            view: View::Ahead,
        }
    }

    /// Sizes the point scratch for a sweep of `points` returns. Returns
    /// whether it grew. The pillar buffers grow in [`Pillarizer::plan`] on the
    /// first sweep that needs more, which [`PillarPlan`] does not count; the
    /// stage counts its bytes in the usual windows.
    pub fn reserve(&mut self, points: usize) -> bool {
        let mut grew = false;
        if self.cells.capacity() < points {
            self.cells.reserve_exact(points - self.cells.len());
            grew = true;
        }
        grew
    }

    /// Bytes the scratch holds.
    pub fn scratch_bytes(&self) -> usize {
        self.cells.capacity() * std::mem::size_of::<(u32, u32)>()
            + self.features.capacity() * std::mem::size_of::<f32>()
            + self.index.capacity() * std::mem::size_of::<i64>()
    }

    /// Bins the sweep's points into the pillars of the view ahead and writes
    /// the network's two inputs. Deterministic: see the module docs.
    pub fn plan(&mut self, points: &[[f32; 4]]) -> PillarPlan {
        self.plan_view(points, View::Ahead)
    }

    /// [`Pillarizer::plan`] for either view: [`View::Behind`] reads every
    /// point turned half round about the vertical, `(-x, -y, z)`, and
    /// nothing else differs.
    pub fn plan_view(&mut self, points: &[[f32; 4]], view: View) -> PillarPlan {
        let turn = view.sign();
        let read = |p: &[f32; 4]| [turn * p[0], turn * p[1], p[2], p[3]];
        self.view = view;
        let mut plan = PillarPlan {
            source_points: clamp_u32(points.len()),
            ..PillarPlan::default()
        };
        self.cells.clear();
        for (i, raw) in points.iter().enumerate() {
            let p = read(raw);
            if !(p[0].is_finite() && p[1].is_finite() && p[2].is_finite() && p[3].is_finite()) {
                plan.non_finite += 1;
                continue;
            }
            // Half-open on every axis, as the model's voxeliser has it, and
            // written positively so a NaN is out.
            let inside = p[0] >= RANGE_MIN[0]
                && p[0] < RANGE_MAX[0]
                && p[1] >= RANGE_MIN[1]
                && p[1] < RANGE_MAX[1]
                && p[2] >= RANGE_MIN[2]
                && p[2] < RANGE_MAX[2];
            if !inside {
                continue;
            }
            let ix = ((p[0] - RANGE_MIN[0]) / PILLAR_EDGE_M).floor() as usize;
            let iy = ((p[1] - RANGE_MIN[1]) / PILLAR_EDGE_M).floor() as usize;
            // A point on the range's far face is excluded above, but f32
            // rounding at the edge could still floor to the grid's size.
            let (ix, iy) = (ix.min(GRID_X - 1), iy.min(GRID_Y - 1));
            self.cells.push(((iy * GRID_X + ix) as u32, i as u32));
        }
        plan.in_range = clamp_u32(self.cells.len());
        // `(key, point)` is unique, so an unstable sort cannot change the
        // result and allocates nothing.
        self.cells.sort_unstable();
        // Count the pillars first, so the two buffers are sized once.
        let mut found = 0usize;
        let mut last: Option<u32> = None;
        for &(k, _) in &self.cells {
            if last != Some(k) {
                found += 1;
                last = Some(k);
            }
        }
        plan.pillars_found = clamp_u32(found);
        let n = found.min(MAX_PILLARS);
        plan.pillars = clamp_u32(n);
        self.n = n;
        let want = n * POINTS_PER_PILLAR * FEATURES_PER_POINT;
        self.features.clear();
        self.features.resize(want, 0.0);
        self.index.clear();
        self.index.resize(n, 0);

        let mut pillar = 0usize;
        let mut start = 0usize;
        while start < self.cells.len() {
            let key = self.cells[start].0;
            let mut end = start;
            while end < self.cells.len() && self.cells[end].0 == key {
                end += 1;
            }
            if pillar >= n {
                plan.capped_points += clamp_u32(end - start);
                start = end;
                continue;
            }
            let kept = (end - start).min(POINTS_PER_PILLAR);
            plan.truncated_points += clamp_u32(end - start - kept);
            let run = &self.cells[start..start + kept];
            // The pillar's mean over the points it keeps, in f64 over a
            // pinned order, as every accumulation in the chain is.
            let mut sum = [0f64; 3];
            for &(_, pi) in run {
                let p = read(&points[pi as usize]);
                sum[0] += f64::from(p[0]);
                sum[1] += f64::from(p[1]);
                sum[2] += f64::from(p[2]);
            }
            let mean = sum.map(|s| (s / kept as f64) as f32);
            let ix = (key as usize) % GRID_X;
            let iy = (key as usize) / GRID_X;
            let centre = [
                0.5 * PILLAR_EDGE_M + ix as f32 * PILLAR_EDGE_M + RANGE_MIN[0],
                0.5 * PILLAR_EDGE_M + iy as f32 * PILLAR_EDGE_M + RANGE_MIN[1],
                0.5 * PILLAR_HEIGHT_M + RANGE_MIN[2],
            ];
            let base = pillar * POINTS_PER_PILLAR * FEATURES_PER_POINT;
            for (slot, &(_, pi)) in run.iter().enumerate() {
                let p = read(&points[pi as usize]);
                let f =
                    &mut self.features[base + slot * FEATURES_PER_POINT..][..FEATURES_PER_POINT];
                f[0] = p[0];
                f[1] = p[1];
                f[2] = p[2];
                f[3] = p[3];
                f[4] = p[0] - mean[0];
                f[5] = p[1] - mean[1];
                f[6] = p[2] - mean[2];
                f[7] = p[0] - centre[0];
                f[8] = p[1] - centre[1];
                f[9] = p[2] - centre[2];
            }
            self.index[pillar] = i64::from(key);
            pillar += 1;
            start = end;
        }
        plan
    }

    /// Pillars in the inputs below.
    pub fn pillars(&self) -> usize {
        self.n
    }

    /// Which way the sweep was read to make them.
    pub fn view(&self) -> View {
        self.view
    }

    /// The network's `voxels` input: `pillars x 32 x 10`.
    pub fn features(&self) -> &[f32] {
        &self.features[..self.n * POINTS_PER_PILLAR * FEATURES_PER_POINT]
    }

    /// The network's `pillar_index` input: `pillars x 1`.
    pub fn index(&self) -> &[i64] {
        &self.index[..self.n]
    }
}

/// One candidate before the NMS: a box, in the lidar's own frame, and which
/// view and where in its output it came from, which with the score is a
/// total order.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Candidate {
    /// The decoded box.
    pub det: LidarDet,
    /// The view the network found it in.
    pub view: View,
    /// Output row (y), column (x) and anchor, in that view.
    pub cell: (u16, u16, u8),
}

/// What the network's three outputs hold, read where the runtime left
/// them: `cls` is `248 x 216 x 6 x 3`, `boxes` `248 x 216 x 6 x 7`, `dirs`
/// `248 x 216 x 6 x 2`.
pub struct Outputs<'a> {
    /// Class logits.
    pub cls: &'a [f32],
    /// Box residuals.
    pub boxes: &'a [f32],
    /// Heading-bin logits.
    pub dirs: &'a [f32],
}

/// Decodes every anchor whose best class reaches [`SCORE_THRESHOLD`] into
/// `out`, NVIDIA's postprocess number for number, and sorts them by score
/// descending, then output position. `out` is cleared first. One view, the
/// one ahead: [`decode_view`] and [`rank`] are its two halves, which a
/// second view needs apart.
pub fn decode(o: Outputs<'_>, out: &mut Vec<Candidate>) -> Result<(), PillarsError> {
    out.clear();
    decode_view(o, View::Ahead, out)?;
    rank(out);
    Ok(())
}

/// Decodes one view's outputs and **appends** its candidates to `out`, each
/// box in the lidar's own frame whichever way the view looked: one found in
/// the sweep turned half round is turned back. Not sorted; [`rank`] does
/// that once every view is in.
pub fn decode_view(
    o: Outputs<'_>,
    view: View,
    out: &mut Vec<Candidate>,
) -> Result<(), PillarsError> {
    let cells = FEATURE_Y * FEATURE_X;
    let want = |name: &str, got: usize, per: usize| {
        if got == cells * ANCHORS_PER_CELL * per {
            Ok(())
        } else {
            Err(PillarsError::Output(format!(
                "{name} has {got} values, expected {}",
                cells * ANCHORS_PER_CELL * per
            )))
        }
    };
    want("cls_preds", o.cls.len(), 3)?;
    want("box_preds", o.boxes.len(), 7)?;
    want("dir_cls_preds", o.dirs.len(), 2)?;
    let sigmoid = |v: f32| 1.0 / (1.0 + (-v).exp());
    let x_stride = (RANGE_MAX[0] - RANGE_MIN[0]) / (FEATURE_X as f32 - 1.0);
    let y_stride = (RANGE_MAX[1] - RANGE_MIN[1]) / (FEATURE_Y as f32 - 1.0);
    for row in 0..FEATURE_Y {
        for col in 0..FEATURE_X {
            let loc = row * FEATURE_X + col;
            for a in 0..ANCHORS_PER_CELL {
                let c = &o.cls[(loc * ANCHORS_PER_CELL + a) * 3..][..3];
                // Written positively, so a NaN logit is no candidate.
                if !(c[0] >= LOGIT_FLOOR || c[1] >= LOGIT_FLOOR || c[2] >= LOGIT_FLOOR) {
                    continue;
                }
                let (mut best, mut score) = (0u32, sigmoid(c[0]));
                for (k, &v) in c.iter().enumerate().skip(1) {
                    let s = sigmoid(v);
                    if s > score {
                        score = s;
                        best = k as u32;
                    }
                }
                if score < SCORE_THRESHOLD {
                    continue;
                }
                let class = a / 2;
                let [dxa, dya, dza] = ANCHOR_SIZES[class];
                let ra = ANCHOR_YAW[a % 2];
                let xa = RANGE_MIN[0] + col as f32 * x_stride;
                let ya = RANGE_MIN[1] + row as f32 * y_stride;
                let za = 0.5 * dza + ANCHOR_BOTTOM_M[class];
                let e = &o.boxes[(loc * ANCHORS_PER_CELL + a) * 7..][..7];
                let diagonal = (dxa * dxa + dya * dya).sqrt();
                let d = &o.dirs[(loc * ANCHORS_PER_CELL + a) * 2..][..2];
                let dir_label = if d[0] > d[1] { 0.0 } else { 1.0 };
                let period = std::f32::consts::PI;
                let val = e[6] + ra - DIR_OFFSET;
                let dir_rot = val - (val / (period + 1e-8)).floor() * period;
                let det = LidarDet {
                    x: e[0] * diagonal + xa,
                    y: e[1] * diagonal + ya,
                    z: e[2] * dza + za,
                    dx: e[3].exp() * dxa,
                    dy: e[4].exp() * dya,
                    dz: e[5].exp() * dza,
                    yaw: dir_rot + DIR_OFFSET + period * dir_label,
                    class_id: best,
                    score,
                };
                out.push(Candidate {
                    det: match view {
                        View::Ahead => det,
                        View::Behind => det.turned(),
                    },
                    view,
                    cell: (row as u16, col as u16, a as u8),
                });
            }
        }
    }
    Ok(())
}

/// Sorts candidates by score descending, then view, then output position:
/// the order [`nms`] reads them in. `total_cmp` on the score and then the
/// view and the cell is a unique key, so the unstable sort cannot change the
/// result.
pub fn rank(cands: &mut [Candidate]) {
    cands.sort_unstable_by(|p, q| {
        q.det
            .score
            .total_cmp(&p.det.score)
            .then(p.view.cmp(&q.view))
            .then(p.cell.cmp(&q.cell))
    });
}

/// Bird's-eye intersection over union of two boxes' footprints, rotated as
/// they are: the intersection is the convex polygon one footprint clips the
/// other to (Sutherland-Hodgman, in f64), the union the two areas less it.
pub fn bev_iou(a: &LidarDet, b: &LidarDet) -> f32 {
    let to64 = |c: [[f32; 2]; 4]| c.map(|p| [f64::from(p[0]), f64::from(p[1])]);
    let (ca, cb) = (to64(a.footprint()), to64(b.footprint()));
    let inter = polygon_area(&clip_polygon(&ca, &cb));
    let union = f64::from(a.area_bev()) + f64::from(b.area_bev()) - inter;
    if union <= 1e-8 {
        return 0.0;
    }
    (inter / union) as f32
}

/// `subject` clipped to the convex, counter-clockwise `clip`.
fn clip_polygon(subject: &[[f64; 2]; 4], clip: &[[f64; 2]; 4]) -> Vec<[f64; 2]> {
    let inside = |p: [f64; 2], a: [f64; 2], b: [f64; 2]| {
        (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) >= 0.0
    };
    let intersect = |a: [f64; 2], b: [f64; 2], p: [f64; 2], q: [f64; 2]| -> [f64; 2] {
        let d1 = [b[0] - a[0], b[1] - a[1]];
        let d2 = [q[0] - p[0], q[1] - p[1]];
        let den = d1[0] * d2[1] - d1[1] * d2[0];
        if den.abs() < 1e-12 {
            return p;
        }
        let t = ((p[0] - a[0]) * d2[1] - (p[1] - a[1]) * d2[0]) / den;
        [a[0] + t * d1[0], a[1] + t * d1[1]]
    };
    let mut out: Vec<[f64; 2]> = subject.to_vec();
    for i in 0..4 {
        let (a, b) = (clip[i], clip[(i + 1) % 4]);
        let input = std::mem::take(&mut out);
        if input.is_empty() {
            break;
        }
        let mut s = input[input.len() - 1];
        for e in input {
            if inside(e, a, b) {
                if !inside(s, a, b) {
                    out.push(intersect(a, b, s, e));
                }
                out.push(e);
            } else if inside(s, a, b) {
                out.push(intersect(a, b, s, e));
            }
            s = e;
        }
    }
    out
}

/// Area of a simple polygon by the shoelace formula; 0 under three vertices.
fn polygon_area(poly: &[[f64; 2]]) -> f64 {
    if poly.len() < 3 {
        return 0.0;
    }
    let mut twice = 0.0;
    for i in 0..poly.len() {
        let (p, q) = (poly[i], poly[(i + 1) % poly.len()]);
        twice += p[0] * q[1] - q[0] * p[1];
    }
    twice.abs() * 0.5
}

/// Greedy non-maximum suppression over candidates already sorted by
/// [`rank`]: each is kept unless a kept box overlaps it by [`NMS_IOU`] or
/// more, NVIDIA's rule, whichever view either came from. Returns the kept
/// boxes in that order.
///
/// Two footprints further apart than the sum of their half-diagonals cannot
/// touch, so their overlap is not computed: the same answer, without the
/// polygon clipping, for nearly every pair of a sweep.
pub fn nms(cands: &[Candidate]) -> Vec<LidarDet> {
    let mut kept: Vec<LidarDet> = Vec::new();
    for c in cands {
        if kept
            .iter()
            .all(|k| !may_overlap(&c.det, k) || bev_iou(&c.det, k) < NMS_IOU)
        {
            kept.push(c.det);
        }
    }
    kept
}

/// Whether two boxes' footprints can overlap at all: their centres are no
/// further apart than the sum of their half-diagonals. Written so that a
/// NaN anywhere says "may", and the pair goes to [`bev_iou`] as before.
fn may_overlap(a: &LidarDet, b: &LidarDet) -> bool {
    let (dx, dy) = (a.x - b.x, a.y - b.y);
    let reach = 0.5 * (a.dx.hypot(a.dy) + b.dx.hypot(b.dy));
    let (apart2, reach2) = (dx * dx + dy * dy, reach * reach);
    apart2 <= reach2 || apart2.is_nan() || reach2.is_nan()
}

/// A moment inside [`LidarModel::infer_views`], handed to the caller's
/// callback so it can time the parts: this crate does not read a clock.
/// Each comes once a view, in this order; the NMS, once a sweep, is what is
/// left after the last.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Step {
    /// A view's points are binned into its pillars.
    Binned,
    /// The network has run on them and its outputs are in hand.
    Ran,
    /// The outputs are decoded into candidates.
    Decoded,
}

/// The loaded network. Built once, before the clock starts; `infer` is the
/// only thing a sweep costs.
pub struct LidarModel {
    session: Mutex<ort::session::Session>,
    /// Which runtime runs it: `ort` or `cuda`.
    pub backend: Backend,
    /// The CPU threads the `ort` backend was given; 1 on `cuda`.
    pub threads: usize,
    /// The `cuda` backend's knobs.
    pub cuda: CudaOptions,
    /// The file it was loaded from, relative to the working directory:
    /// [`model_for`] the backend.
    pub file: &'static str,
    /// sha256 of the file it was loaded from: the pinned one.
    pub sha256: String,
}

impl LidarModel {
    /// Reads the file [`model_for`] `backend` names, checks its sha256
    /// against the pinned one, builds the session on `backend` and runs one
    /// empty sweep through it, so the runtime's first-run cost is paid
    /// before the clock starts.
    pub fn load(
        backend: Backend,
        threads: usize,
        cuda: CudaOptions,
    ) -> Result<LidarModel, PillarsError> {
        if !backend.is_ort() {
            return Err(PillarsError::Backend(backend));
        }
        let (name, pinned) = model_for(backend);
        let path = Path::new(name);
        let mut file = match std::fs::File::open(path) {
            Ok(f) => f,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                return Err(PillarsError::Missing(path.to_path_buf()))
            }
            Err(e) => return Err(PillarsError::Io(path.to_path_buf(), e)),
        };
        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes)
            .map_err(|e| PillarsError::Io(path.to_path_buf(), e))?;
        let sha256 = sha256_hex(&bytes);
        if sha256 != pinned {
            return Err(PillarsError::Sha256Mismatch {
                path: path.to_path_buf(),
                got: sha256,
                expected: pinned,
            });
        }
        let threads = if backend == Backend::Ort {
            threads.max(1)
        } else {
            1
        };
        let session = ort_session(&bytes, threads, (backend == Backend::Cuda).then_some(cuda))
            .map_err(PillarsError::Runtime)?;
        let model = LidarModel {
            session: Mutex::new(session),
            backend,
            threads,
            cuda,
            file: name,
            sha256,
        };
        // The warm-up: one pillar of zeros at the grid's corner.
        let blank = Pillarizer {
            cells: Vec::new(),
            features: vec![0.0; POINTS_PER_PILLAR * FEATURES_PER_POINT],
            index: vec![0],
            n: 1,
            view: View::Ahead,
        };
        let mut scratch = Vec::new();
        model.infer(&blank, &mut scratch)?;
        Ok(model)
    }

    /// Runs the network on one view's pillars, already binned, and decodes
    /// its output **where the runtime left it**, into `scratch` (the
    /// candidates, kept between sweeps so the decode allocates only when a
    /// sweep has more than any before), then suppresses the overlaps. The
    /// boxes come back in NMS order: score descending.
    pub fn infer(
        &self,
        pillars: &Pillarizer,
        scratch: &mut Vec<Candidate>,
    ) -> Result<Vec<LidarDet>, PillarsError> {
        scratch.clear();
        self.look(pillars, scratch, &mut |_| {})?;
        rank(scratch);
        Ok(nms(scratch))
    }

    /// Every one of `views` of one sweep: for each, the pillars, the network
    /// and the decode, then one NMS over all their candidates together.
    /// `step` is called at each [`Step`] of each view, so a caller with a
    /// clock can say how long the pillars, the network, the decode and the
    /// NMS took. The plan is what was read of the sweep, the views' counts
    /// added, whether or not the network then ran.
    pub fn infer_views(
        &self,
        pillars: &mut Pillarizer,
        points: &[[f32; 4]],
        views: &[View],
        scratch: &mut Vec<Candidate>,
        step: &mut dyn FnMut(Step),
    ) -> (PillarPlan, Result<Vec<LidarDet>, PillarsError>) {
        scratch.clear();
        let mut plan = PillarPlan::default();
        for (k, &view) in views.iter().enumerate() {
            let one = pillars.plan_view(points, view);
            plan = if k == 0 { one } else { plan.with(one) };
            step(Step::Binned);
            if let Err(e) = self.look(pillars, scratch, step) {
                return (plan, Err(e));
            }
        }
        rank(scratch);
        (plan, Ok(nms(scratch)))
    }

    /// One pass of the network, over the pillars of one view, its
    /// candidates appended to `cands` in the lidar's own frame.
    fn look(
        &self,
        pillars: &Pillarizer,
        cands: &mut Vec<Candidate>,
        step: &mut dyn FnMut(Step),
    ) -> Result<(), PillarsError> {
        use ort::value::TensorRef;
        let rt = |e: ort::Error| PillarsError::Runtime(format!("ONNX Runtime: {e}"));
        let n = pillars.pillars();
        // Borrowed, not copied: the runtime reads the pillars where the
        // voxeliser wrote them.
        let voxels = TensorRef::from_array_view((
            [n, POINTS_PER_PILLAR, FEATURES_PER_POINT],
            pillars.features(),
        ))
        .map_err(rt)?;
        let index = TensorRef::from_array_view(([n, 1], pillars.index())).map_err(rt)?;
        let mut s = self.session.lock().unwrap_or_else(PoisonError::into_inner);
        let out = s
            .run(ort::inputs!["voxels" => voxels, "pillar_index" => index])
            .map_err(rt)?;
        let (_, cls) = out["cls_preds"].try_extract_tensor::<f32>().map_err(rt)?;
        let (_, boxes) = out["box_preds"].try_extract_tensor::<f32>().map_err(rt)?;
        let (_, dirs) = out["dir_cls_preds"]
            .try_extract_tensor::<f32>()
            .map_err(rt)?;
        step(Step::Ran);
        decode_view(Outputs { cls, boxes, dirs }, pillars.view(), cands)?;
        step(Step::Decoded);
        Ok(())
    }
}

/// Arrow field of one lane.
fn lane_field() -> FieldRef {
    Arc::new(Field::new("item", DataType::Float32, false))
}

/// Arrow field of one detection: `FixedSizeList<Float32, 9>`.
fn det_field() -> FieldRef {
    Arc::new(Field::new(
        "item",
        DataType::FixedSizeList(lane_field(), PILLAR_DET_VALUES),
        false,
    ))
}

/// `detection_count u32, format Utf8, tov_trigger_ns i64, source_sweep_seq
/// i64, source_point_count u32, points_in_range u32, pillars u32,
/// candidates u32, detections LargeList<FixedSizeList<Float32, 9>>`, one row
/// per sweep.
///
/// On [`crate::detect::detect_schema`]'s terms: one contiguous f32 buffer
/// for the payload, and the counts that make the stage's work recomputable
/// from the batch alone. The model's provenance -- name, sha256, runtime,
/// range, views, thresholds -- is the schema's metadata, carried by every
/// batch by reference. A sweep in which the network found nothing is still
/// a row, with an empty list: the stage looked. The counts are the views'
/// added.
pub fn pillar_det_schema(model_sha256: &str, backend: &str, views: &[View]) -> Arc<Schema> {
    let meta = [
        ("model", MODEL_NAME.to_string()),
        ("model_sha256", model_sha256.to_string()),
        ("backend", backend.to_string()),
        ("views", views_words(views)),
        (
            "input",
            format!(
                "pillars x {POINTS_PER_PILLAR} x {FEATURES_PER_POINT} over x {}..{} y {}..{} z {}..{} m, {PILLAR_EDGE_M} m pillars",
                RANGE_MIN[0], RANGE_MAX[0], RANGE_MIN[1], RANGE_MAX[1], RANGE_MIN[2], RANGE_MAX[2]
            ),
        ),
        ("score_threshold", SCORE_THRESHOLD.to_string()),
        ("nms_iou", NMS_IOU.to_string()),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_string(), v))
    .collect();
    Arc::new(
        Schema::new(vec![
            Field::new("detection_count", DataType::UInt32, false),
            Field::new("format", DataType::Utf8, false),
            Field::new("tov_trigger_ns", DataType::Int64, false),
            Field::new("source_sweep_seq", DataType::Int64, false),
            Field::new("source_point_count", DataType::UInt32, false),
            Field::new("points_in_range", DataType::UInt32, false),
            Field::new("pillars", DataType::UInt32, false),
            Field::new("candidates", DataType::UInt32, false),
            Field::new("detections", DataType::LargeList(det_field()), false),
        ])
        .with_metadata(meta),
    )
}

/// What a batch says about the sweep it came from.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PillarMeta {
    /// The sweep's forward-facing trigger, ns on the sensor clock.
    pub trigger_ns: i64,
    /// The sweep's number.
    pub sweep_seq: i64,
    /// Points in the sweep, in the range, and pillars handed to the network.
    pub source_points: u32,
    /// See `source_points`.
    pub in_range: u32,
    /// See `source_points`.
    pub pillars: u32,
    /// Anchors above the score threshold, before the NMS.
    pub candidates: u32,
}

/// Builds the one-row `PILLAR_DET` batch for a sweep and its boxes: one
/// f32 buffer, wrapped **without copying it**. Returns the batch and the
/// buffer's address, its `storage_id`.
pub fn build_pillar_det_batch(
    dets: &[LidarDet],
    meta: PillarMeta,
    schema: &Arc<Schema>,
) -> Result<(RecordBatch, usize), PillarsError> {
    let mut buf = MutableBuffer::from_len_zeroed(dets.len() * PILLAR_DET_BYTES);
    let addr = buf.as_slice().as_ptr() as usize;
    if !addr.is_multiple_of(std::mem::align_of::<f32>()) {
        return Err(PillarsError::Sweep(SweepError::Misaligned { addr }));
    }
    {
        let out = buf.typed_data_mut::<f32>();
        for (w, d) in dets.iter().enumerate() {
            out[w * PILLAR_DET_LANES..][..PILLAR_DET_LANES].copy_from_slice(&d.lanes());
        }
    }
    let buf = Buffer::from(buf);
    let len = buf.len();
    if !len.is_multiple_of(PILLAR_DET_BYTES) {
        return Err(SweepError::NotWholePoints { got: len }.into());
    }
    let storage_id = buf.as_ptr() as usize;
    let n = len / PILLAR_DET_BYTES;
    let values = Float32Array::new(ScalarBuffer::<f32>::from(buf), None);
    let det = FixedSizeListArray::try_new(
        lane_field(),
        PILLAR_DET_VALUES,
        Arc::new(values) as ArrayRef,
        None,
    )
    .map_err(SweepError::Arrow)?;
    let offsets = OffsetBuffer::<i64>::new(ScalarBuffer::<i64>::from(vec![0i64, n as i64]));
    let arr = LargeListArray::try_new(det_field(), offsets, Arc::new(det), None)
        .map_err(SweepError::Arrow)?;
    let columns: Vec<ArrayRef> = vec![
        Arc::new(UInt32Array::from(vec![clamp_u32(n)])),
        Arc::new(StringArray::from(vec![PILLAR_DET_FORMAT])),
        Arc::new(Int64Array::from(vec![meta.trigger_ns])),
        Arc::new(Int64Array::from(vec![meta.sweep_seq])),
        Arc::new(UInt32Array::from(vec![meta.source_points])),
        Arc::new(UInt32Array::from(vec![meta.in_range])),
        Arc::new(UInt32Array::from(vec![meta.pillars])),
        Arc::new(UInt32Array::from(vec![meta.candidates])),
        Arc::new(arr),
    ];
    Ok((
        RecordBatch::try_new(Arc::clone(schema), columns).map_err(SweepError::Arrow)?,
        storage_id,
    ))
}

/// The interleaved lanes of a `PILLAR_DET` batch, shared with the buffer the
/// stage built; `None` for a batch of another format.
pub fn pillar_det_f32(batch: &RecordBatch) -> Option<&[f32]> {
    let fmt = batch.column_by_name("format")?.as_string_opt::<i32>()?;
    if fmt.is_empty() || fmt.value(0) != PILLAR_DET_FORMAT {
        return None;
    }
    Some(
        batch
            .column_by_name("detections")?
            .as_list_opt::<i64>()?
            .values()
            .as_fixed_size_list_opt()?
            .values()
            .as_primitive_opt::<Float32Type>()?
            .values(),
    )
}

/// The boxes of a `PILLAR_DET` batch, one lane array each, borrowed; empty
/// for a batch of another format.
pub fn pillar_det_rows(batch: &RecordBatch) -> &[[f32; PILLAR_DET_LANES]] {
    pillar_det_f32(batch).map_or(&[], |v| v.as_chunks::<PILLAR_DET_LANES>().0)
}

/// Address of the shared buffer, read back out of the finished batch.
pub fn pillar_det_storage_id(batch: &RecordBatch) -> Option<usize> {
    pillar_det_f32(batch).map(|v| v.as_ptr() as usize)
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

/// Boxes in the batch, from its count column rather than the payload's
/// length, so a disagreement is visible.
pub fn pillar_det_count(batch: &RecordBatch) -> Option<u32> {
    u32_col(batch, "detection_count")
}

/// The sweep's trigger instant.
pub fn pillar_det_trigger_ns(batch: &RecordBatch) -> Option<i64> {
    i64_col(batch, "tov_trigger_ns")
}

/// The sweep the boxes came from.
pub fn pillar_det_sweep_seq(batch: &RecordBatch) -> Option<i64> {
    i64_col(batch, "source_sweep_seq")
}

/// Points in that sweep.
pub fn pillar_det_source_points(batch: &RecordBatch) -> Option<u32> {
    u32_col(batch, "source_point_count")
}

/// Points of it inside the range.
pub fn pillar_det_points_in_range(batch: &RecordBatch) -> Option<u32> {
    u32_col(batch, "points_in_range")
}

/// Pillars handed to the network.
pub fn pillar_det_pillars(batch: &RecordBatch) -> Option<u32> {
    u32_col(batch, "pillars")
}

/// Anchors above the score threshold, before the NMS.
pub fn pillar_det_candidates(batch: &RecordBatch) -> Option<u32> {
    u32_col(batch, "candidates")
}

fn clamp_u32(n: usize) -> u32 {
    u32::try_from(n).unwrap_or(u32::MAX)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    fn det(x: f32, y: f32, dx: f32, dy: f32, yaw: f32) -> LidarDet {
        LidarDet {
            x,
            y,
            z: 0.0,
            dx,
            dy,
            dz: 1.5,
            yaw,
            class_id: 0,
            score: 0.5,
        }
    }

    #[test]
    fn pillars_are_binned_in_key_order_and_keep_their_first_32_points_in_sweep_order() {
        // Two pillars, out of order in the sweep; the second has 34 points.
        let mut points: Vec<[f32; 4]> = Vec::new();
        for k in 0..34 {
            // All in pillar (x 9.92..10.08, y 0.32..0.48): key = 250 * 432 + 62.
            points.push([10.0 + 0.001 * k as f32, 0.40, -1.0, 0.1 * (k % 10) as f32]);
        }
        points.push([2.9, -1.0, -1.5, 0.2]); // pillar x 18 (2.88..3.04), y 241
        points.push([2.95, -1.0, -1.4, 0.3]); // the same pillar, after
        points.push([80.0, 0.0, 0.0, 0.0]); // past the range
        points.push([f32::NAN, 0.0, 0.0, 0.0]);
        let mut pz = Pillarizer::new();
        pz.reserve(points.len());
        let plan = pz.plan(&points);
        assert_eq!(plan.source_points, 38);
        assert_eq!(plan.in_range, 36);
        assert_eq!(plan.non_finite, 1);
        assert_eq!(plan.pillars, 2);
        assert_eq!(plan.truncated_points, 2);
        assert_eq!(pz.pillars(), 2);
        let key_a = (241 * GRID_X + 18) as i64;
        let key_b = (250 * GRID_X + 62) as i64;
        assert_eq!(pz.index(), &[key_a, key_b]);
        let f = pz.features();
        let per = POINTS_PER_PILLAR * FEATURES_PER_POINT;
        // Pillar A: two points, the first at x 2.9, offset from the mean -0.025.
        assert_eq!(f[0], 2.9);
        assert!((f[4] + 0.025).abs() < 1e-5, "{}", f[4]);
        // Its centre: x = 0.08 + 18 * 0.16 = 2.96, y = 0.08 + 241 * 0.16 - 39.68 = -1.04, z = -1.
        assert!((f[7] + 0.06).abs() < 1e-4, "{}", f[7]);
        assert!((f[8] - 0.04).abs() < 1e-4, "{}", f[8]);
        assert!((f[9] + 0.5).abs() < 1e-5, "{}", f[9]);
        // Its third slot is padding.
        assert!(f[2 * FEATURES_PER_POINT..3 * FEATURES_PER_POINT]
            .iter()
            .all(|v| *v == 0.0));
        // Pillar B: 32 points in sweep order, the 33rd and 34th dropped.
        assert_eq!(f[per], 10.0);
        assert!((f[per + 31 * FEATURES_PER_POINT] - 10.031).abs() < 1e-4);
        assert_eq!(f.len(), 2 * per);
        // The same sweep permuted gives the same pillars in the same order.
        let mut shuffled = points.clone();
        shuffled.rotate_left(5);
        let mut pz2 = Pillarizer::new();
        let plan2 = pz2.plan(&shuffled);
        assert_eq!(plan2.pillars, plan.pillars);
        assert_eq!(pz2.index(), pz.index());
    }

    #[test]
    fn a_point_is_inside_a_rotated_footprint_in_the_box_frame() {
        let b = det(10.0, 0.0, 4.0, 2.0, std::f32::consts::FRAC_PI_2);
        // Turned a quarter turn: 2 m along x, 4 m along y.
        assert!(b.contains_bev(10.0, 1.9));
        assert!(!b.contains_bev(10.0, 2.1));
        assert!(b.contains_bev(10.9, 0.0));
        assert!(!b.contains_bev(11.1, 0.0));
        assert!(!b.contains_bev(f32::NAN, 0.0));
        assert!((b.near_face_m() - 9.0).abs() < 1e-5);
    }

    #[test]
    fn rotated_iou_is_exact_on_aligned_boxes_and_zero_apart() {
        let a = det(0.0, 0.0, 4.0, 2.0, 0.0);
        let b = det(2.0, 0.0, 4.0, 2.0, 0.0); // half overlap: 4 / (8 + 8 - 4)
        assert!(
            (bev_iou(&a, &b) - 1.0 / 3.0).abs() < 1e-5,
            "{}",
            bev_iou(&a, &b)
        );
        assert!((bev_iou(&a, &a) - 1.0).abs() < 1e-5);
        let far = det(20.0, 0.0, 4.0, 2.0, 0.0);
        assert_eq!(bev_iou(&a, &far), 0.0);
        // A square turned 45 degrees inside a larger one: the small area.
        let big = det(0.0, 0.0, 10.0, 10.0, 0.0);
        let small = det(0.0, 0.0, 2.0, 2.0, std::f32::consts::FRAC_PI_4);
        assert!((bev_iou(&big, &small) - 4.0 / 100.0).abs() < 1e-5);
    }

    #[test]
    fn nms_keeps_the_first_of_any_overlapping_pair_and_the_decode_order_is_total() {
        let hi = Candidate {
            det: det(5.0, 0.0, 4.0, 2.0, 0.0),
            view: View::Ahead,
            cell: (1, 1, 0),
        };
        let lo = Candidate {
            det: LidarDet {
                score: 0.2,
                ..det(6.0, 0.5, 4.0, 2.0, 0.3)
            },
            view: View::Ahead,
            cell: (1, 2, 1),
        };
        let apart = Candidate {
            det: LidarDet {
                score: 0.1,
                ..det(30.0, 10.0, 0.8, 0.6, 0.0)
            },
            view: View::Ahead,
            cell: (9, 9, 2),
        };
        let kept = nms(&[hi, lo, apart]);
        assert_eq!(kept, vec![hi.det, apart.det]);
    }

    /// The view behind is the sweep turned half round: a point behind the
    /// sensor lands in the pillar its mirror image ahead lands in, with the
    /// features that image has, and nothing ahead is in it. The two views'
    /// counts add, the sweep's own points counted once.
    #[test]
    fn the_view_behind_reads_the_sweep_turned_half_round() {
        let points = [
            [10.0f32, 0.40, -1.0, 0.5],
            [-10.0, -0.40, -1.0, 0.25],
            [-30.0, 5.0, -1.0, 0.1],
            [f32::NAN, 0.0, 0.0, 0.0],
        ];
        let mut pz = Pillarizer::new();
        let ahead = pz.plan(&points);
        assert_eq!(pz.view(), View::Ahead);
        assert_eq!((ahead.in_range, ahead.pillars, ahead.non_finite), (1, 1, 1));
        let key_ahead = pz.index()[0];
        assert_eq!(&pz.features()[..4], &[10.0, 0.40, -1.0, 0.5]);

        let behind = pz.plan_view(&points, View::Behind);
        assert_eq!(pz.view(), View::Behind);
        assert_eq!((behind.in_range, behind.pillars), (2, 2));
        // (-10, -0.40) turned is (10, 0.40): the first point's pillar.
        let at = pz.index().iter().position(|&k| k == key_ahead).unwrap();
        let f = &pz.features()[at * POINTS_PER_PILLAR * FEATURES_PER_POINT..][..4];
        assert_eq!(f, &[10.0, 0.40, -1.0, 0.25]);

        let both = ahead.with(behind);
        assert_eq!(both.source_points, 4);
        assert_eq!(both.non_finite, 1);
        assert_eq!((both.in_range, both.pillars, both.pillars_found), (3, 3, 3));
        assert_eq!(views_words(&View::AROUND), "ahead, behind");
    }

    /// A box turned back is the same rectangle on the other side of the
    /// sensor, and turning it twice is where it started, heading included.
    #[test]
    fn a_box_turned_back_is_the_same_rectangle_behind_the_sensor() {
        let seen = det(12.0, -3.0, 4.0, 1.8, 0.9);
        let back = seen.turned();
        assert_eq!((back.x, back.y), (-12.0, 3.0));
        assert_eq!((back.dx, back.dy, back.dz), (seen.dx, seen.dy, seen.dz));
        for (p, q) in seen.footprint().iter().zip(back.footprint()) {
            assert!((p[0] + q[0]).abs() < 1e-4 && (p[1] + q[1]).abs() < 1e-4);
        }
        assert!(back.contains_bev(-12.5, 3.2));
        assert!(!back.contains_bev(12.5, -3.2));
        let twice = back.turned();
        assert_eq!((twice.x, twice.y), (seen.x, seen.y));
        assert!((twice.yaw - seen.yaw).abs() < 1e-5, "{}", twice.yaw);
        // A heading already past half a turn comes back inside one turn.
        let late = det(1.0, 1.0, 4.0, 1.8, 5.5).turned();
        assert!((late.yaw - (5.5 - std::f32::consts::PI)).abs() < 1e-5);
    }

    /// One NMS over both views: a car beside the vehicle, cut by the line
    /// between them, is found once in each and kept once, the better one;
    /// and on a tie the view ahead comes first, so the order is total.
    #[test]
    fn two_views_of_one_car_are_one_box() {
        let ahead = Candidate {
            det: LidarDet {
                score: 0.6,
                ..det(1.2, 2.5, 4.0, 1.8, 0.0)
            },
            view: View::Ahead,
            cell: (140, 3, 0),
        };
        let behind = Candidate {
            det: LidarDet {
                score: 0.8,
                ..det(0.4, 2.6, 4.1, 1.7, 0.0)
            },
            view: View::Behind,
            cell: (107, 1, 0),
        };
        let far_behind = Candidate {
            det: LidarDet {
                score: 0.4,
                ..det(-25.0, 0.0, 4.0, 1.8, 0.0)
            },
            view: View::Behind,
            cell: (124, 78, 0),
        };
        let mut cands = vec![ahead, far_behind, behind];
        rank(&mut cands);
        assert_eq!(cands, vec![behind, ahead, far_behind]);
        assert_eq!(nms(&cands), vec![behind.det, far_behind.det]);

        let tie = Candidate {
            view: View::Ahead,
            ..behind
        };
        let mut tied = vec![behind, tie];
        rank(&mut tied);
        assert_eq!(tied, vec![tie, behind]);
    }

    /// The NMS's shortcut is exact: boxes too far apart to touch are kept
    /// without their overlap being computed, boxes that touch are still
    /// compared, and a NaN box goes to the comparison as it always did.
    #[test]
    fn the_nms_shortcut_never_changes_which_boxes_are_kept() {
        let a = det(0.0, 0.0, 4.0, 2.0, 0.0);
        // Half-diagonals 2.236 each: 4.4 m apart can touch, 4.5 m cannot.
        assert!(may_overlap(&a, &det(4.4, 0.0, 4.0, 2.0, 0.0)));
        assert!(!may_overlap(&a, &det(4.5, 0.0, 4.0, 2.0, 0.0)));
        assert!(may_overlap(&a, &det(f32::NAN, 0.0, 4.0, 2.0, 0.0)));
        // Wherever the shortcut says "cannot", the overlap really is nothing.
        for k in 0..400 {
            let t = k as f32;
            let b = det(
                (t * 0.37).sin() * 9.0,
                (t * 0.53).cos() * 9.0,
                0.6 + (t * 0.11).sin().abs() * 4.0,
                0.5 + (t * 0.07).cos().abs() * 2.0,
                t * 0.29,
            );
            if !may_overlap(&a, &b) {
                assert_eq!(bev_iou(&a, &b), 0.0, "{b:?}");
            }
        }
    }

    /// An anchor whose logits are all under the floor is skipped, one at the
    /// threshold's own logit is scored, and a NaN logit is no candidate.
    #[test]
    fn the_decoder_skips_only_anchors_that_cannot_reach_the_score() {
        let cells = FEATURE_Y * FEATURE_X;
        let n = cells * ANCHORS_PER_CELL;
        let boxes = vec![0.0f32; n * 7];
        let dirs = vec![0.0f32; n * 2];
        let run = |logit: f32| {
            let mut cls = vec![-20.0f32; n * 3];
            cls[2] = logit;
            let mut out = Vec::new();
            decode(
                Outputs {
                    cls: &cls,
                    boxes: &boxes,
                    dirs: &dirs,
                },
                &mut out,
            )
            .unwrap();
            out
        };
        // Just above the threshold's logit: a candidate, class 2.
        let at = run(-2.19);
        assert_eq!(at.len(), 1);
        assert_eq!(at[0].det.class_id, 2);
        assert!(at[0].det.score >= SCORE_THRESHOLD);
        // Between the floor and the threshold: scored, and under it.
        assert!(run(-2.25).is_empty());
        // Under the floor, and NaN: never scored.
        assert!(run(-5.0).is_empty());
        assert!(run(f32::NAN).is_empty());
        // The floor is under the threshold's own logit, with room.
        assert!(1.0 / (1.0 + (-LOGIT_FLOOR).exp()) < SCORE_THRESHOLD - 0.005);
    }

    #[test]
    fn the_decoder_reads_one_anchor_back_to_its_anchor_box() {
        let cells = FEATURE_Y * FEATURE_X;
        let mut cls = vec![-20.0f32; cells * ANCHORS_PER_CELL * 3];
        let boxes = vec![0.0f32; cells * ANCHORS_PER_CELL * 7];
        let mut dirs = vec![0.0f32; cells * ANCHORS_PER_CELL * 2];
        // Row 100, column 50, anchor 3 (pedestrian, across): class 1 lit.
        let (row, col, a) = (100usize, 50usize, 3usize);
        let loc = row * FEATURE_X + col;
        cls[(loc * ANCHORS_PER_CELL + a) * 3 + 1] = 2.0;
        dirs[(loc * ANCHORS_PER_CELL + a) * 2 + 1] = 1.0;
        let mut out = Vec::new();
        decode(
            Outputs {
                cls: &cls,
                boxes: &boxes,
                dirs: &dirs,
            },
            &mut out,
        )
        .unwrap();
        assert_eq!(out.len(), 1);
        let c = out[0];
        assert_eq!(c.cell, (100, 50, 3));
        let d = c.det;
        assert_eq!(d.class_id, 1);
        assert!((d.score - 1.0 / (1.0 + (-2.0f32).exp())).abs() < 1e-6);
        assert!((d.x - 50.0 * 69.12 / 215.0).abs() < 1e-4, "{}", d.x);
        assert!(
            (d.y - (-39.68 + 100.0 * 79.36 / 247.0)).abs() < 1e-4,
            "{}",
            d.y
        );
        assert!((d.z - (-0.6 + 0.865)).abs() < 1e-5);
        assert!((d.dx - 0.8).abs() < 1e-6 && (d.dy - 0.6).abs() < 1e-6);
        // Heading: anchor 1.57, the second bin: 1.57 + pi, wrapped by the
        // model's rule.
        let period = std::f32::consts::PI;
        let val = 1.57 - DIR_OFFSET;
        let want = val - (val / (period + 1e-8)).floor() * period + DIR_OFFSET + period;
        assert!((d.yaw - want).abs() < 1e-5, "{} vs {want}", d.yaw);
        // A wrong-sized output is refused, not misread.
        assert!(decode(
            Outputs {
                cls: &cls[1..],
                boxes: &boxes,
                dirs: &dirs
            },
            &mut out
        )
        .is_err());
    }

    #[test]
    fn a_batch_carries_its_boxes_in_one_buffer_and_reads_them_back() {
        let schema = pillar_det_schema("abc", "ort", &View::AROUND);
        let dets = [det(1.0, 2.0, 3.0, 4.0, 0.5), det(5.0, 6.0, 7.0, 8.0, 1.5)];
        let meta = PillarMeta {
            trigger_ns: 7,
            sweep_seq: 3,
            source_points: 100,
            in_range: 50,
            pillars: 9,
            candidates: 4,
        };
        let (batch, storage) = build_pillar_det_batch(&dets, meta, &schema).unwrap();
        assert_eq!(pillar_det_storage_id(&batch), Some(storage));
        assert_eq!(pillar_det_count(&batch), Some(2));
        assert_eq!(pillar_det_sweep_seq(&batch), Some(3));
        assert_eq!(pillar_det_trigger_ns(&batch), Some(7));
        assert_eq!(pillar_det_pillars(&batch), Some(9));
        assert_eq!(pillar_det_candidates(&batch), Some(4));
        assert_eq!(pillar_det_points_in_range(&batch), Some(50));
        let rows = pillar_det_rows(&batch);
        assert_eq!(rows.len(), 2);
        assert_eq!(LidarDet::from_lanes(&rows[1]), dets[1]);
        assert_eq!(batch.schema().metadata()["model_sha256"], "abc");
        assert_eq!(batch.schema().metadata()["views"], "ahead, behind");
        // An empty sweep is still a row.
        let (empty, _) = build_pillar_det_batch(&[], meta, &schema).unwrap();
        assert_eq!(pillar_det_count(&empty), Some(0));
        assert!(pillar_det_rows(&empty).is_empty());
    }
}
