//! Detections: the first stage in this project whose output is **smaller
//! because it says something more specific**, rather than because it is the
//! same thing decimated.
//!
//! [`crate::voxel`] shrinks a sweep 4x and hands on *the same four fields* —
//! x, y, z, reflectance — averaged per 20 cm cube. Useful, honest, and it
//! answers nothing: 1.95 MB of returns becomes 492 KB of returns. This module
//! consumes that cloud and emits **structures**: ground removed, the rest
//! grouped into connected objects, each one summarised as where it is, how big
//! it is and how bright it is. The reduction is another ~60x and it is a
//! different *kind* of reduction, which is the whole point of the stage.
//!
//! # Every parameter here is derived, and here is where each comes from
//!
//! * **The ground band is one voxel edge.** It is the run's
//!   [`crate::voxel::VoxelSize`] itself, not a number of its own — 0.20 m at
//!   the default grid. A band narrower than a cell cannot classify a cell
//!   whose centroid is quantised to that cell.
//! * **The cluster gate is 3 voxels**, and it reuses the argument that chose
//!   the voxel size. `voxel.rs` picked 0.20 m so that a 0.6 m pedestrian — the
//!   smallest KITTI object class — is "at least three voxels across". A
//!   cluster of fewer than three voxels is therefore *smaller than the
//!   smallest object the grid was built to preserve*. Same number, same
//!   argument, no second parameter.
//! * **The range limit is 30 m, and it is a validity bound rather than a
//!   crop.** `voxel.rs` rejected range cropping as "a filter wearing a hat",
//!   and that rejection stands: the limit here is not the reduction —
//!   clustering is — and it was not chosen to shrink the output. It is the
//!   range past which the algorithm's *precondition* fails. Measured
//!   nearest-neighbour spacing between non-ground returns on drive_0005 sweep
//!   0, against the 0.20 m voxel edge whose adjacency connected components
//!   tests:
//!
//!   | range | median NN | p90 NN | share of NN gaps over one voxel edge |
//!   |---|---|---|---|
//!   | 0-10 m | 0.024 m | 0.056 m | 0.2 % |
//!   | 10-20 m | 0.051 m | 0.100 m | 0.9 % |
//!   | 20-30 m | 0.092 m | 0.147 m | 3.2 % |
//!   | 30-40 m | 0.132 m | 0.226 m | 15.4 % |
//!   | 60-80 m | 0.261 m | 0.535 m | 85.2 % |
//!
//!   Beyond ~30 m the returns of one object are further apart than one voxel,
//!   so the adjacency this stage tests **is not physically present in the
//!   data** and any cluster out there is an artifact of sampling. 20-30 m is
//!   the last bin whose p90 stays under the voxel edge. It is not a knee read
//!   off an output curve: detection counts scale smoothly with area (85 at
//!   20 m, 175 at 30 m, 276 at 40 m, 615 at 80 m) while the fragment ratio
//!   degrades monotonically, so nothing about the output suggests 30 m. Only
//!   the spacing table does.
//!
//!   **Unverified:** the table was measured at 0.20 m only. The bound should
//!   scale with the voxel edge, and [`RANGE_LIMIT_M`] does not — a run at
//!   `--voxel-size-m 0.5` gets a limit measured for a finer grid. It is
//!   recorded in the batch and printed by the run so the mismatch is visible
//!   rather than buried.
//!
//! # Ground removal, and the flat-world assumption it rests on
//!
//! `voxel.rs` rejected ground removal as a *reduction*, partly because "the
//! correct version (RANSAC) is randomised". Here ground removal is not a
//! reduction but a **precondition** — clusters bridge through the road surface
//! and merge into 15-27 m blobs without it — so it is back on the table. The
//! randomness objection is not: `voxel.rs` grounds this project's
//! replayability claim on the pipeline using no randomness and no map at all,
//! and a randomised detect stage would make "same sweep, same detections"
//! false. [`GroundPlane::fit`] is therefore a **deterministic least-squares
//! fit** — median-z seed, smallest eigenvector of the seed's scatter matrix by
//! cyclic Jacobi, two fixed refits against the inlier band, no sampling and no
//! convergence test.
//!
//! **This is a flat-world assumption and it will not survive a hill.** A
//! single plane cannot describe a crest or a banked corner. On drive_0005 it
//! does not have to: **this code**, run over all 154 sweeps, fits a mean tilt
//! of 1.02 deg and a maximum of 2.14 deg.
//!
//! The design note for this work said 0.84 and 1.86, from the numpy
//! investigation that preceded it. Both are right and they are not the same
//! measurement: that fit seeded from the whole cloud, this one seeds from the
//! voxels inside [`RANGE_LIMIT_M`] only, so it is fitted to a smaller and
//! nearer patch of road. The argument is unchanged — around one degree, never
//! past two — but a doc comment about this code should state this code's
//! numbers, which is the same correction `voxel.rs` had to make about its own.
//!
//! The three candidate methods (fixed z, fitted plane, per-cell
//! minimum z) agree to within 3.2 percentage points of removed voxels on
//! average here — **and that is a fact about this drive, not about the
//! methods.** The disagreement tracks tilt exactly: at 0.17 deg it is 0.1 %
//! within 10 m, at 1.84 deg it is 21.3 % and the fixed threshold misses 1,250
//! ground voxels in the near field. A 2 deg slope lifts the ground 1.05 m
//! across a 30 m radius and 5 deg lifts it 2.62 m, against a 0.20 m band, so a
//! fixed threshold on real terrain does not degrade — it collapses. That
//! extrapolation is arithmetic, not measurement; this drive cannot demonstrate
//! it, which is exactly why the fitted plane is here and why the caveat is
//! written down instead of discovered later.
//!
//! Per-cell minimum z would handle a crest and is also deterministic, but it
//! needs a cell size and a tolerance — two numbers with no derivation — and
//! measured 1.0 pp different from the plane on this drive. One plane, and the
//! assumption stated.
//!
//! # What a detection count is, and what it is not
//!
//! It is **connected non-ground structures within 30 m**. It is *not* a count
//! of objects, and the gap is wide enough that reporting the number without
//! this would mislead. Bucketing 6,865 detections of drive_0005 by size: ~11 %
//! are pedestrian-shaped, ~7 % car-or-van-shaped, and **59 % have a footprint
//! under 1.0 m and a height under 0.8 m**. Those are not ground clutter — a
//! height-above-ground gate barely moves the count (175 -> 166 at 0.5 m,
//! -> 140 at 0.8 m, from the investigation rather than from this code) — they
//! are **fragments of larger structures** broken up by
//! sparse sampling. About six per sweep go the other way and merge into
//! 6-23 m blobs; connected components cannot separate a car parked against a
//! wall, and no setting of any parameter here fixes that.
//! [`DetectPlan::fragment_detections`] and [`DetectPlan::merged_detections`]
//! count both, so the caveat travels with the number instead of living only in
//! this comment.
//!
//! # Where the numbers in this file came from
//!
//! Two places, and they are not interchangeable, so the distinction is made
//! once here rather than hedged at every figure.
//!
//! **Reproduced by this code**, on all 154 sweeps of drive_0005, by
//! `crates/pipes-kitti/tests/detect_real_drive.rs`: 177 detections per sweep
//! (148-209), 7,789 B per sweep, 59.2 % fragments, 964 merged over the drive,
//! a fitted tilt of mean 1.02 deg and max 2.14 deg with 0 fallbacks, 0
//! recovered-index collisions, and the persistence table (71.6 % compensated,
//! 35.8 % uncompensated, 1.8 % and 1.1 % for the two controls).
//!
//! **From the numpy investigation that preceded this module, and NOT
//! reproduced here**: the nearest-neighbour spacing table, the detection
//! counts at other range limits (85 at 20 m ... 615 at 80 m), and the
//! height-above-ground gate's effect on the count (175 -> 166 -> 140). They
//! are the evidence the parameters were chosen on, they are quoted as
//! measurements because they are, and nothing in this workspace re-checks
//! them. A reader deciding whether to trust 30 m should know which of those
//! two lists the argument for it is in.
//!
//! # Determinism
//!
//! By construction, in the three ways [`crate::voxel`] is. The grouping is a
//! sort and a union-find over sorted keys, not a map — `clippy.toml` bans
//! `HashMap`/`HashSet` for nondeterministic iteration and this module uses no
//! map at all. [`union`] always attaches the higher cell index to the lower,
//! so a component's root is its lexicographically smallest voxel and the
//! output order is that key ascending, independent of input order. Every
//! accumulation is f64 over a pinned order — the ground seed is sorted before
//! it is fitted, for the reason [`Detector::plan`] gives at that line — so the
//! result does not depend on which voxels were added first.
//!
//! **One honest limit on that.** The cell sort key is `(packed key, input
//! row)`, and the row half moves when the input is permuted. It only matters
//! when two input rows recover the SAME grid key, because then the two are
//! summed in whichever order the permutation left them: everything else is
//! ordered by key alone. [`DetectPlan::key_collisions`] counts exactly that
//! case, and it is 0 on every sweep measured — so the stronger property
//! ("permute the input, get identical bytes") holds in practice, while the
//! property that holds unconditionally is the one the pipeline needs: the same
//! cloud in the same order always gives the same detections.

use std::sync::Arc;

use arrow::array::{
    Array, ArrayRef, AsArray, BooleanArray, FixedSizeListArray, Float32Array, Int64Array,
    LargeListArray, RecordBatch, StringArray, UInt32Array,
};
use arrow::buffer::{Buffer, MutableBuffer, OffsetBuffer, ScalarBuffer};
use arrow::datatypes::{DataType, Field, FieldRef, Float32Type, Schema};

use crate::velo::{velo_xyzr, SweepError};
use crate::voxel::VoxelSize;

/// f32 lanes in one detection record. See [`DETECTION_BYTES`] for the layout.
pub const DETECTION_VALUES: i32 = 11;

/// [`DETECTION_VALUES`] as a `usize`, for slicing.
///
/// The same number twice because two APIs want two types: Arrow's
/// `FixedSizeListArray` takes an `i32` width, and `as_chunks` takes a `usize`
/// const generic. Derived from the other rather than written out again, so the
/// pair cannot drift.
pub const DETECTION_LANES: usize = DETECTION_VALUES as usize;

/// Bytes in one detection: 11 f32 lanes, in this order.
///
/// | lanes | field | why it is here |
/// |---|---|---|
/// | 0-2 | `centroid` x, y, z | what a tracker associates on. Stabler than the box centre, which jumps whenever a far edge of the object first returns. |
/// | 3-5 | `bbox_min` x, y, z | a camera projection needs eight corners to make a 2D box; a centroid cannot give them. |
/// | 6-8 | `bbox_max` x, y, z | the other corner. |
/// | 9 | `voxel_count` | size, and the fragment indicator: a 3-voxel detection at 25 m is a sliver of something. |
/// | 10 | `mean_reflectance` | free — it is already the 4th lane of the input — and it separates road paint and number plates from foliage. |
///
/// **`voxel_count` is an integer in an f32 lane, and that is exact rather than
/// hopeful.** f32 represents every integer below 2^24 exactly; a whole voxel
/// cloud on this dataset is ~35,000 rows, so one cluster's count cannot come
/// within three orders of magnitude of the bound. The alternative — a separate
/// `UInt32` column — would give the batch a second data buffer and therefore a
/// second `storage_id`, and the address-equality evidence this project proves
/// zero copy with assumes one.
///
/// **Deliberately absent: a raw point count.** It is not derivable. The schema
/// `reduce` emits has no per-voxel count column, so a stage consuming its
/// output cannot say how many laser returns are behind a detection — only how
/// many voxels. Adding one means changing `reduce`'s output shape, which is
/// the one thing `voxel.rs` is built not to do. **Also absent: orientation**
/// (not observable from a single sweep; it belongs to a tracker) and **a class
/// label** (there is no classifier here, and a guessed label is exactly the
/// plausible-looking wrongness `voxel.rs` exists to refuse).
pub const DETECTION_BYTES: usize = 44;

/// Value of the `detection_format` column: what the 11 lanes mean.
///
/// `c3` centroid, `b6` two box corners, `n1` voxel count, `r1` mean
/// reflectance. Deliberately shares no spelling with
/// [`crate::velo::POINT_FORMAT_XYZR_F32LE`] or its voxel variant, because a
/// detection is not a point and nothing downstream should be able to read one
/// as the other.
pub const DETECTION_FORMAT: &str = "det_c3b6n1r1_f32le";

/// Horizontal range past which voxel adjacency stops being present in the
/// data, in metres. See the module docs for the spacing table it comes from.
///
/// Horizontal (`hypot(x, y)`), not slant range: a ground-based lidar's z span
/// is a few metres against tens of metres of radius, so the two differ by
/// under 1 % at the bound, and the spacing that matters is the spacing across
/// a surface the sensor is looking at from the side.
pub const RANGE_LIMIT_M: f32 = 30.0;

/// Association gate for [`persistence`], in metres.
///
/// **Not derived, and this constant exists so that it says so in one place
/// rather than appearing as a literal at three call sites.** 0.5 m is where
/// the investigation measured. It sits above the 0.4 m an object closing at
/// 4 m/s covers in one 10 Hz interval and above the 0.2 m of quantisation a
/// voxel centroid carries, which is why it works — but it was computed from
/// neither of those, it was chosen and then checked. Every other number in
/// this module comes from a measurement or from a number already in the tree;
/// this one does not, and hiding that would be worse than the parameter.
pub const PERSISTENCE_GATE_M: f32 = 0.5;

/// Smallest cluster reported as a detection, in voxels.
///
/// Not a free parameter: `voxel.rs` chose 0.20 m so that a 0.6 m pedestrian is
/// at least three voxels across, so a cluster of fewer than three voxels is
/// smaller than the smallest object the grid was sized to preserve.
pub const MIN_CLUSTER_VOXELS: u32 = 3;

/// Where each of this stage's three numbers comes from, in one line, so a
/// run's own output can say it and a reader never has to take them on trust.
///
/// A function of the voxel size for the same reason
/// [`crate::voxel::voxel_size_note`] is: the band and the cluster gate are
/// *derived from* that edge, so printing their justification beside a
/// different edge would be printing a derivation that is arithmetically false.
pub fn detect_params_note(v: VoxelSize) -> String {
    format!(
        "ground band = {} m (one voxel edge); cluster gate = {} voxels (the width \
         the voxel edge was chosen to give a 0.6 m pedestrian); range limit = {} m \
         (past it the measured return spacing exceeds one voxel edge, so the \
         adjacency this stage tests is not in the data){}",
        v.metres(),
        MIN_CLUSTER_VOXELS,
        RANGE_LIMIT_M,
        if v.is_default() {
            ""
        } else {
            " -- NOTE: the range limit was measured at 0.20 m only and does not \
             scale with --voxel-size-m"
        }
    )
}

/// Bits per packed axis index, and so the addressable grid: signed
/// `[-2^20, 2^20)` per axis. Matches `voxel.rs` exactly, because this stage
/// recovers the grid `reduce` used rather than inventing one of its own.
const INDEX_BITS: u32 = 21;

/// Added to each signed axis index so the packed key is non-negative and key
/// order is lexicographic in `(ix, iy, iz)`.
const INDEX_BIAS: i64 = 1 << (INDEX_BITS - 1);

/// Footprint under which a detection is counted as a fragment, in metres:
/// wider than a pedestrian, so nothing under it is a whole KITTI object class.
const FRAGMENT_FOOTPRINT_M: f32 = 1.0;

/// Height under which a detection is counted as a fragment, in metres: about
/// half a pedestrian, so nothing under it is a whole KITTI object class.
const FRAGMENT_HEIGHT_M: f32 = 0.8;

/// Extent past which a detection is counted as merged, in metres. Longer than
/// any single KITTI object class, so anything over it is at least two things
/// stuck together.
const MERGED_EXTENT_M: f32 = 6.0;

/// What a box's extent alone says it can be, by the three thresholds above.
///
/// A statement about SIZE and nothing else: a `Fragment` is smaller than any
/// whole KITTI object class, a `Merged` box longer than any, and `Object` is
/// everything between -- sized like a road user, which is not the same as
/// being one. [`DetectPlan::fragment_detections`] and
/// [`DetectPlan::merged_detections`] count detections by it, and the fusion
/// asks the same question of a track's box, so "fragment-shaped" means one
/// thing everywhere in the chain.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Shape {
    /// Under [`FRAGMENT_FOOTPRINT_M`] in both horizontal axes and under
    /// [`FRAGMENT_HEIGHT_M`] tall.
    Fragment,
    /// Neither a fragment nor merged.
    Object,
    /// Over [`MERGED_EXTENT_M`] along some axis.
    Merged,
}

impl Shape {
    /// The spelling the run's output and `summary.json` use.
    pub fn name(self) -> &'static str {
        match self {
            Shape::Fragment => "fragment",
            Shape::Object => "object",
            Shape::Merged => "merged",
        }
    }
}

/// The [`Shape`] of an axis-aligned box from its two corners.
pub fn shape_of(lo: [f32; 3], hi: [f32; 3]) -> Shape {
    let ext = [hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]];
    if ext[0] < FRAGMENT_FOOTPRINT_M && ext[1] < FRAGMENT_FOOTPRINT_M && ext[2] < FRAGMENT_HEIGHT_M
    {
        Shape::Fragment
    } else if ext.iter().any(|e| *e > MERGED_EXTENT_M) {
        Shape::Merged
    } else {
        Shape::Object
    }
}

/// Arrow field of one detection lane.
fn detection_value_field() -> FieldRef {
    Arc::new(Field::new("item", DataType::Float32, false))
}

/// Arrow field of one detection: `FixedSizeList<Float32, 11>`.
fn detection_field() -> FieldRef {
    Arc::new(Field::new(
        "item",
        DataType::FixedSizeList(detection_value_field(), DETECTION_VALUES),
        false,
    ))
}

/// Arrow field of a ground-plane component.
fn ground_plane_field() -> FieldRef {
    Arc::new(Field::new("item", DataType::Float32, false))
}

/// `detection_count u32, detection_format Utf8, tov_trigger_ns i64,
/// source_sweep_seq i64, voxel_size_m f32, range_limit_m f32, min_cluster_voxels u32,
/// source_voxel_count u32, source_point_count u32,
/// ground_plane FixedSizeList<Float32, 4>, ground_fitted bool,
/// detections LargeList<FixedSizeList<Float32, 11>>`.
///
/// **Deliberately not the cloud's schema**, and that is the first time in this
/// project a stage has changed the shape of what it passes on. `reduce` was
/// built to emit exactly the layout it consumed, so one accessor reads both
/// and `cloud_thread` runs unmodified on either — the property that made the
/// Arrow rule demonstrable rather than aspirational. It is worth saying
/// plainly why it ends here: it ends because the data stopped being the same
/// kind of thing. A detection is not a point, and a schema that let
/// [`velo_xyzr`] read one as four floats would let every existing consumer
/// silently treat a bounding-box corner as a position.
///
/// What is kept from `reduce`'s design is the part that carries: one row per
/// sample, one contiguous f32 buffer for the payload (so one `storage_id`, so
/// the address-equality proof extends with no new machinery), and every
/// parameter recorded *in the batch* so a recorded sample is interpretable on
/// its own. `source_voxel_count` and `source_point_count` make **both** links
/// of the chain recomputable from this batch alone rather than from a log line
/// somebody has to still have.
///
/// **`source_sweep_seq` is here for a stage that did not exist when this
/// schema was written**, and it costs 8 bytes per sample to avoid a threshold.
/// A tracker has to know whether two batches came from CONSECUTIVE SWEEPS, and
/// nothing else in the batch says: [`pipes_core::sample::Sample::seq`] is this
/// stage's own counter, which stays consecutive across an eviction on
/// `velo->reduce`, and [`pipes_core::sample::Sample::parent`] names the reduced
/// cloud rather than the sweep behind it. The alternative was a tolerance on
/// `tov_trigger_ns`, and a tolerance is exactly what turned this module's own
/// persistence check into a number about the eviction rate. -1 when the cloud
/// carried no parent.
pub fn detect_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("detection_count", DataType::UInt32, false),
        Field::new("detection_format", DataType::Utf8, false),
        Field::new("tov_trigger_ns", DataType::Int64, false),
        Field::new("source_sweep_seq", DataType::Int64, false),
        Field::new("voxel_size_m", DataType::Float32, false),
        Field::new("range_limit_m", DataType::Float32, false),
        Field::new("min_cluster_voxels", DataType::UInt32, false),
        Field::new("source_voxel_count", DataType::UInt32, false),
        Field::new("source_point_count", DataType::UInt32, false),
        Field::new(
            "ground_plane",
            DataType::FixedSizeList(ground_plane_field(), 4),
            false,
        ),
        Field::new("ground_fitted", DataType::Boolean, false),
        Field::new("detections", DataType::LargeList(detection_field()), false),
    ]))
}

/// The interleaved detection lanes of a detect batch, shared with the buffer
/// the stage built. `chunks_exact(11)` gives one detection each.
pub fn detections_f32(batch: &RecordBatch) -> Option<&[f32]> {
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

/// Address of the shared detection buffer, read back out of the finished
/// batch: the zero-copy proof, extended to this stream.
pub fn detect_storage_id(batch: &RecordBatch) -> Option<usize> {
    detections_f32(batch).map(|v| v.as_ptr() as usize)
}

/// Detections in the batch, from the metadata column rather than the payload
/// length, so a disagreement between the two is visible.
pub fn detection_count(batch: &RecordBatch) -> Option<u32> {
    let c = batch
        .column_by_name("detection_count")?
        .as_primitive_opt::<arrow::datatypes::UInt32Type>()?;
    (!c.is_empty()).then(|| c.value(0))
}

/// What the 11 lanes of a detection MEAN, read back out of the batch.
pub fn detection_format(batch: &RecordBatch) -> Option<&str> {
    let c = batch
        .column_by_name("detection_format")?
        .as_any()
        .downcast_ref::<StringArray>()?;
    (!c.is_empty()).then(|| c.value(0))
}

/// The sweep's trigger instant, inherited unchanged from the cloud.
pub fn detect_trigger_ns(batch: &RecordBatch) -> Option<i64> {
    let c = batch
        .column_by_name("tov_trigger_ns")?
        .as_primitive_opt::<arrow::datatypes::Int64Type>()?;
    (!c.is_empty()).then(|| c.value(0))
}

/// The velodyne sweep these detections came from, or -1.
///
/// Carried so a downstream stage can tell CONSECUTIVE sweeps from merely
/// consecutive samples without a tolerance; see [`detect_schema`].
pub fn detect_sweep_seq(batch: &RecordBatch) -> Option<i64> {
    let c = batch
        .column_by_name("source_sweep_seq")?
        .as_primitive_opt::<arrow::datatypes::Int64Type>()?;
    (!c.is_empty()).then(|| c.value(0))
}

/// The ground plane these detections were separated from, as
/// `(nx, ny, nz, d)`, and whether it was fitted or fell back. See
/// [`GroundPlane`].
pub fn ground_plane(batch: &RecordBatch) -> Option<([f32; 4], bool)> {
    let l = batch
        .column_by_name("ground_plane")?
        .as_any()
        .downcast_ref::<FixedSizeListArray>()?;
    let v = l.values().as_primitive_opt::<Float32Type>()?.values();
    let fitted = batch
        .column_by_name("ground_fitted")?
        .as_any()
        .downcast_ref::<BooleanArray>()?;
    (v.len() >= 4 && !fitted.is_empty()).then(|| ([v[0], v[1], v[2], v[3]], fitted.value(0)))
}

/// Voxels these detections were found in, carried so the shrink of THIS link
/// is recomputable from the batch alone.
pub fn source_voxel_count(batch: &RecordBatch) -> Option<u32> {
    let c = batch
        .column_by_name("source_voxel_count")?
        .as_primitive_opt::<arrow::datatypes::UInt32Type>()?;
    (!c.is_empty()).then(|| c.value(0))
}

/// Raw returns behind those voxels, carried through from `reduce` so the
/// WHOLE chain is recomputable from this batch alone.
pub fn source_point_count(batch: &RecordBatch) -> Option<u32> {
    let c = batch
        .column_by_name("source_point_count")?
        .as_primitive_opt::<arrow::datatypes::UInt32Type>()?;
    (!c.is_empty()).then(|| c.value(0))
}

/// The voxel edge these detections were found on, read back out of the batch.
pub fn detect_voxel_size_m(batch: &RecordBatch) -> Option<f32> {
    let c = batch
        .column_by_name("voxel_size_m")?
        .as_primitive_opt::<Float32Type>()?;
    (!c.is_empty()).then(|| c.value(0))
}

/// The range bound that was in force, read back out of the batch.
pub fn detect_range_limit_m(batch: &RecordBatch) -> Option<f32> {
    let c = batch
        .column_by_name("range_limit_m")?
        .as_primitive_opt::<Float32Type>()?;
    (!c.is_empty()).then(|| c.value(0))
}

/// Errors from detecting on one cloud.
#[derive(Debug)]
pub enum DetectError {
    /// The derived buffer could not be wrapped as Arrow.
    Sweep(SweepError),
    /// [`Detector::build`] was handed different voxels than [`Detector::plan`]
    /// indexed. The scratch holds indices into the planned slice, so this
    /// would read the wrong voxels (or panic); it is a typed error rather than
    /// either.
    PlanMismatch {
        /// Voxels the plan was computed over.
        planned: u32,
        /// Voxels `build` was given.
        got: usize,
    },
}

impl std::fmt::Display for DetectError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DetectError::Sweep(_) => write!(f, "the detections could not be wrapped as Arrow"),
            DetectError::PlanMismatch { planned, got } => write!(
                f,
                "the plan indexed {planned} voxels but build was given {got}"
            ),
        }
    }
}

impl std::error::Error for DetectError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            DetectError::Sweep(e) => Some(e),
            DetectError::PlanMismatch { .. } => None,
        }
    }
}

impl From<SweepError> for DetectError {
    fn from(e: SweepError) -> Self {
        DetectError::Sweep(e)
    }
}

/// A ground plane, `n . x + d = 0` with `n` unit length and `n[2] >= 0`, so
/// [`GroundPlane::height`] is signed height *above* the surface.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GroundPlane {
    /// Unit normal, z component non-negative.
    pub n: [f64; 3],
    /// Offset: `n . x + d = 0` on the surface.
    pub d: f64,
    /// Whether this came from the least-squares fit, or from the fallback.
    ///
    /// `false` means the seed did not determine a plane and this is the
    /// horizontal plane through the seed's median height. Counted and
    /// reported rather than silently substituted: a fallback plane *is* the
    /// fixed-z method this module's docs argue against, so a run that used one
    /// has to say so.
    pub fitted: bool,
}

impl GroundPlane {
    /// Signed height of a point above the surface, in metres.
    pub fn height(&self, p: [f32; 3]) -> f64 {
        self.n[0] * f64::from(p[0])
            + self.n[1] * f64::from(p[1])
            + self.n[2] * f64::from(p[2])
            + self.d
    }

    /// Tilt from horizontal, in degrees. Mean 1.02, max 2.14 over
    /// drive_0005's 154 sweeps as this code fits them — the measurement the
    /// flat-world caveat in the module docs rests on.
    pub fn tilt_deg(&self) -> f64 {
        self.n[2].clamp(-1.0, 1.0).acos().to_degrees()
    }

    /// The horizontal plane at height `z`, used when the seed does not
    /// determine one.
    pub fn horizontal(z: f64) -> GroundPlane {
        GroundPlane {
            n: [0.0, 0.0, 1.0],
            d: -z,
            fitted: false,
        }
    }

    /// Deterministic least-squares ground fit over `seed`, refined against a
    /// `band`-wide inlier band.
    ///
    /// **No randomness, by construction, and that is a requirement rather than
    /// a preference.** RANSAC is the textbook answer and it is disqualified
    /// here: `voxel.rs` grounds this project's replayability claim on the
    /// pipeline using no randomness and no map, and a detect stage that
    /// sampled would make "same sweep, same detections" false.
    ///
    /// Three fixed passes, no convergence test — a convergence test makes the
    /// *number of passes* data-dependent, which is one more thing that has to
    /// be argued about before "deterministic" holds. Each pass:
    /// 1. accumulate the centred scatter matrix of the current inlier set in
    ///    f64, in two passes (mean first, then deviations) so nothing cancels;
    /// 2. take the eigenvector of its **smallest** eigenvalue — the direction
    ///    of least spread, i.e. the plane normal — through [`sym_eigen3`];
    /// 3. re-select inliers as the seed points within `band` of it.
    ///
    /// Two guards, both derived rather than tuned. A fit failing either falls
    /// back to [`GroundPlane::horizontal`] at the seed's median height:
    ///
    /// * **The normal must be nearer vertical than horizontal**, `n_z >=
    ///   1/sqrt(2)`. That is the bisector, not an angle someone liked: a
    ///   surface tilted past 45 deg is not the ground the sensor is standing
    ///   on. drive_0005's largest measured tilt is 2.14 deg, so on real data
    ///   this has 43 deg of headroom and fires on 0 of 154 sweeps.
    /// * **The seed must spread further than one band in its second
    ///   direction**, `sqrt(lambda_1 / n) > band`. A seed lying along a line
    ///   is contained by infinitely many planes and the eigenvector picked
    ///   from among them is arbitrary; this says so instead of returning one.
    ///   The threshold is the band itself, so it adds no new number.
    pub fn fit(seed: &[[f32; 3]], band: f64, median_z: f64) -> GroundPlane {
        let fallback = GroundPlane::horizontal(median_z);
        if seed.len() < 3 {
            return fallback;
        }
        let mut plane = fallback;
        for pass in 0..3 {
            // Pass 0 fits the whole seed; later passes fit the inliers of the
            // plane the previous pass produced. `move` copies the plane in, so
            // the predicate is `Copy` and can be used by both loops below.
            let current = plane;
            let keep =
                move |p: &&[f32; 3]| pass == 0 || current.height([p[0], p[1], p[2]]).abs() <= band;
            let mut n = 0u64;
            let mut mean = [0f64; 3];
            for p in seed.iter().filter(keep) {
                n += 1;
                for (m, c) in mean.iter_mut().zip(p) {
                    *m += f64::from(*c);
                }
            }
            if n < 3 {
                return plane;
            }
            for m in &mut mean {
                *m /= n as f64;
            }
            let mut scatter = [[0f64; 3]; 3];
            for p in seed.iter().filter(keep) {
                let dv = [
                    f64::from(p[0]) - mean[0],
                    f64::from(p[1]) - mean[1],
                    f64::from(p[2]) - mean[2],
                ];
                for (i, row) in scatter.iter_mut().enumerate() {
                    for (j, cell) in row.iter_mut().enumerate() {
                        *cell += dv[i] * dv[j];
                    }
                }
            }
            let (lambda, vecs) = sym_eigen3(scatter);
            // lambda[1] is the middle eigenvalue: the spread in the seed's
            // second-widest direction. Under one band it is a line, not a
            // patch, and no plane through it is determined.
            if (lambda[1] / n as f64).sqrt() <= band {
                return fallback;
            }
            let mut normal = [vecs[0][0], vecs[1][0], vecs[2][0]];
            // Sign fixed so `height` is height ABOVE the ground rather than
            // depending on which end of the eigenvector the solver returned.
            if normal[2] < 0.0 {
                for c in &mut normal {
                    *c = -*c;
                }
            }
            if normal[2] < std::f64::consts::FRAC_1_SQRT_2 {
                return fallback;
            }
            let d = -(normal[0] * mean[0] + normal[1] * mean[1] + normal[2] * mean[2]);
            plane = GroundPlane {
                n: normal,
                d,
                fitted: true,
            };
        }
        plane
    }
}

/// Eigen-decomposition of a symmetric 3x3 matrix by **cyclic Jacobi rotations
/// with a fixed sweep count**. Returns eigenvalues ascending and the matching
/// eigenvectors as columns, i.e. `vecs[row][k]`.
///
/// A fixed 12 sweeps rather than "until the off-diagonals are small", for the
/// same reason [`GroundPlane::fit`] takes a fixed three passes: a
/// data-dependent iteration count is one more thing that must be argued about
/// before a result is reproducible. Cyclic Jacobi on a 3x3 reaches machine
/// precision in three to six sweeps, so 12 is not a tolerance in disguise, it
/// is slack.
///
/// The rotation is applied as two full 3x3 products rather than the usual
/// in-place update. That is a few dozen extra multiplies per rotation on a
/// matrix this size — nothing next to indexing tens of thousands of voxels —
/// and it removes the whole class of index errors the compact form is known
/// for.
fn sym_eigen3(mut a: [[f64; 3]; 3]) -> ([f64; 3], [[f64; 3]; 3]) {
    let mut v = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    for _ in 0..12 {
        for (p, q) in [(0usize, 1usize), (0, 2), (1, 2)] {
            if a[p][q] == 0.0 {
                continue;
            }
            let theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
            // `f64::signum` is 1.0 at +0.0, so the denominator is never 0.
            let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
            let c = 1.0 / (t * t + 1.0).sqrt();
            let s = t * c;
            let mut g = [[0f64; 3]; 3];
            for (i, row) in g.iter_mut().enumerate() {
                row[i] = 1.0;
            }
            g[p][p] = c;
            g[q][q] = c;
            g[p][q] = s;
            g[q][p] = -s;
            a = mat_mul(&mat_transpose(&g), &mat_mul(&a, &g));
            v = mat_mul(&v, &g);
        }
    }
    // Three elements, so an explicit insertion sort: `sort_by` on f64 wants a
    // total order this does not have, and a comparator that lies is worse than
    // three compares written out.
    let mut order = [0usize, 1, 2];
    for i in 1..3 {
        let mut j = i;
        while j > 0 && a[order[j]][order[j]] < a[order[j - 1]][order[j - 1]] {
            order.swap(j, j - 1);
            j -= 1;
        }
    }
    let lambda = [
        a[order[0]][order[0]],
        a[order[1]][order[1]],
        a[order[2]][order[2]],
    ];
    let mut vecs = [[0f64; 3]; 3];
    for (k, &o) in order.iter().enumerate() {
        for (row, vrow) in vecs.iter_mut().zip(v.iter()) {
            row[k] = vrow[o];
        }
    }
    (lambda, vecs)
}

fn mat_mul(a: &[[f64; 3]; 3], b: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut o = [[0f64; 3]; 3];
    for (i, orow) in o.iter_mut().enumerate() {
        for (j, cell) in orow.iter_mut().enumerate() {
            for (k, brow) in b.iter().enumerate() {
                *cell += a[i][k] * brow[j];
            }
        }
    }
    o
}

fn mat_transpose(a: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut o = [[0f64; 3]; 3];
    for (i, row) in a.iter().enumerate() {
        for (j, cell) in row.iter().enumerate() {
            o[j][i] = *cell;
        }
    }
    o
}

/// What one detect pass did, counted while it was doing it.
///
/// Every field is reported rather than asserted away, on the same terms as
/// [`crate::voxel::VoxelPlan`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DetectPlan {
    /// Voxels in the input cloud.
    pub source_voxels: u32,
    /// Voxels within [`RANGE_LIMIT_M`] with finite values and a grid index in
    /// range — the population everything below is computed over.
    pub in_range_voxels: u32,
    /// In-range voxels classified as ground.
    pub ground_voxels: u32,
    /// Voxels discarded because a coordinate or the reflectance was not
    /// finite. Counted, never folded in: `f32::NAN.floor() as i64` is 0, so
    /// the alternative is every NaN voxel landing in cell (0, 0, 0).
    pub non_finite: u32,
    /// Voxels discarded because the recovered grid index fell outside the
    /// packed key's range. Impossible after `reduce`, which enforces the same
    /// bound; a typed count rather than a wrap-around.
    pub out_of_range: u32,
    /// In-range non-ground voxels whose recovered grid index collided with
    /// another's.
    ///
    /// The checkable half of "the grid is recoverable". This stage recovers
    /// `reduce`'s indices as `floor(centroid / voxel_size)` — verified against
    /// the real ones over 4,736,154 voxels of drive_0005 with 0 mismatches,
    /// though 1,497 centroids land exactly on a cell wall, so the margin can be
    /// zero and the property rests on f32 rounding rather than on a proof.
    /// Detect cannot check the original indices, which `reduce` does not emit;
    /// it CAN check that no two rows land on one index, which is what a
    /// rounding failure would look like from here. 0 on every sweep of
    /// drive_0005.
    ///
    /// **What it does not cover:** the ground and out-of-range voxels, which
    /// are dropped before the cells are built. On drive_0005 the check
    /// therefore sees about 1.76 M of the 4.74 M voxels in the drive, not all
    /// of them, and a collision among the discarded ones would not be seen —
    /// though it also could not affect a detection.
    pub key_collisions: u32,
    /// Connected components before the [`MIN_CLUSTER_VOXELS`] gate.
    pub clusters: u32,
    /// Components that passed the gate, i.e. rows in the output.
    pub detections: u32,
    /// Detections with a footprint under 1 m and a height under 0.8 m.
    ///
    /// **~59 % of them on real data**, and they are *not* ground clutter —
    /// they are fragments of larger structures broken up by sparse sampling.
    /// See the module docs. Counted so the caveat travels with the number.
    pub fragment_detections: u32,
    /// Detections spanning more than 6 m in any axis: the other failure mode,
    /// where connected components fuses a car into the wall behind it.
    pub merged_detections: u32,
    /// The ground plane used, and whether it was fitted.
    pub ground: GroundPlane,
}

impl DetectPlan {
    /// Input voxels per detection, or `None` when nothing was detected.
    ///
    /// `None` rather than a number over an empty output, for the reason
    /// [`crate::voxel::VoxelPlan::ratio`] gives: a ratio computed over nothing
    /// is the easiest figure in this project to publish by accident.
    pub fn ratio(&self) -> Option<f64> {
        (self.detections > 0).then(|| f64::from(self.source_voxels) / f64::from(self.detections))
    }

    /// Share of detections that are fragments of something larger.
    pub fn fragment_fraction(&self) -> Option<f64> {
        (self.detections > 0)
            .then(|| f64::from(self.fragment_detections) / f64::from(self.detections))
    }
}

/// One occupied, in-range, non-ground cell: its recovered grid key and the row
/// of the input cloud it came from.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
struct Cell {
    key: i64,
    row: u32,
}

/// One component's accumulator. f64 sums for the same reason `voxel.rs` uses
/// them: the result must not depend on the order the voxels were added in.
#[derive(Clone, Copy, Debug, Default)]
struct Agg {
    n: u32,
    sum: [f64; 4],
    min: [f32; 3],
    max: [f32; 3],
}

/// The 13 neighbour offsets lexicographically greater than the origin: half of
/// 26-connectivity.
///
/// Half, because union is symmetric and every cell is visited, so pairing each
/// cell with its 13 forward neighbours reaches every adjacent pair exactly
/// once. 26 and not 6: a 6-connected grid splits an object wherever its
/// surface crosses a cell only diagonally, which at 0.20 m is most of a car's
/// roof line.
const FORWARD_NEIGHBOURS: [[i64; 3]; 13] = [
    [1, -1, -1],
    [1, -1, 0],
    [1, -1, 1],
    [1, 0, -1],
    [1, 0, 0],
    [1, 0, 1],
    [1, 1, -1],
    [1, 1, 0],
    [1, 1, 1],
    [0, 1, -1],
    [0, 1, 0],
    [0, 1, 1],
    [0, 0, 1],
];

/// Reusable scratch for detection, so the only per-sample allocation this
/// stage makes is the output buffer itself.
///
/// The same load-bearing constraint [`crate::voxel::Voxelizer`] documents, for
/// the same reason: per-sample scratch allocated inside the measured window
/// would make the reported figure "the stage's output plus its bookkeeping"
/// while the column still said output. Everything this stage needs — the cell
/// list, the union-find parents, the per-component accumulators, the seed the
/// plane is fitted to and the heights its median comes from — is sized once
/// through [`Detector::reserve`], outside the window.
#[derive(Debug, Default)]
pub struct Detector {
    /// In-range non-ground cells, sorted by `(key, row)`.
    cells: Vec<Cell>,
    /// Union-find parent per entry of `cells`.
    parent: Vec<u32>,
    /// Per-component accumulator, indexed by the component's root.
    agg: Vec<Agg>,
    /// Seed for the ground fit: the in-range voxels at or below the median z.
    seed: Vec<[f32; 3]>,
    /// The in-range voxels' z values, sorted, for that median.
    zs: Vec<f32>,
}

impl Detector {
    /// An empty scratch. [`Detector::reserve`] sizes it.
    pub fn new() -> Detector {
        Detector::default()
    }

    /// Makes room for `n` input voxels, emptying the scratch first.
    ///
    /// **Call this outside the measured window.** Returns whether it had to
    /// allocate, so a caller can announce the growth and a test can assert
    /// that the steady state does not grow at all.
    pub fn reserve(&mut self, n: usize) -> bool {
        self.cells.clear();
        self.parent.clear();
        self.agg.clear();
        self.seed.clear();
        self.zs.clear();
        if self.cells.capacity() >= n
            && self.parent.capacity() >= n
            && self.agg.capacity() >= n
            && self.seed.capacity() >= n
            && self.zs.capacity() >= n
        {
            return false;
        }
        self.cells.reserve_exact(n);
        self.parent.reserve_exact(n);
        self.agg.reserve_exact(n);
        self.seed.reserve_exact(n);
        self.zs.reserve_exact(n);
        true
    }

    /// Bytes the scratch currently holds.
    pub fn scratch_bytes(&self) -> usize {
        self.cells.capacity() * std::mem::size_of::<Cell>()
            + self.parent.capacity() * std::mem::size_of::<u32>()
            + self.agg.capacity() * std::mem::size_of::<Agg>()
            + self.seed.capacity() * std::mem::size_of::<[f32; 3]>()
            + self.zs.capacity() * std::mem::size_of::<f32>()
    }

    /// Phase 1: fit the ground, drop it, group what is left. **Borrows
    /// `voxels` and allocates nothing** once [`Detector::reserve`] has been
    /// called for this size.
    ///
    /// `voxels` is the interleaved `xyzr` of a cloud — `reduce`'s output, or
    /// anything in that layout. The grid is recovered as
    /// `floor(centroid / voxel_size)`; see [`DetectPlan::key_collisions`] for
    /// what is and is not checked about that.
    pub fn plan(&mut self, voxels: &[[f32; 4]], voxel_size: VoxelSize) -> DetectPlan {
        self.cells.clear();
        self.parent.clear();
        self.agg.clear();
        self.seed.clear();
        self.zs.clear();
        let v = f64::from(voxel_size.metres());
        let band = v;
        let mut non_finite = 0u32;
        let mut out_of_range = 0u32;

        // Pass 1: the in-range population and its z values, for the median.
        // Range-limited FIRST, so the plane is fitted to the near field where
        // the returns are dense rather than to a scattering of far ones.
        for p in voxels {
            if !p.iter().all(|c| c.is_finite()) {
                non_finite += 1;
                continue;
            }
            if p[0].hypot(p[1]) > RANGE_LIMIT_M {
                continue;
            }
            if voxel_key(p, v).is_none() {
                out_of_range += 1;
                continue;
            }
            self.zs.push(p[2]);
        }
        let in_range_voxels = clamp_u32(self.zs.len());
        // `sort_unstable` on f32 needs a total order. Every value here passed
        // `is_finite`, so `total_cmp` is an ordinary comparison rather than a
        // NaN policy in disguise.
        self.zs.sort_unstable_by(f32::total_cmp);
        let median_z = match self.zs.len() {
            0 => 0.0,
            n => f64::from(self.zs[n / 2]),
        };

        // Pass 2: the seed — everything at or below the median height. Half
        // the in-range cloud by construction, which is a generous superset of
        // the ~40 % that really is ground, and the refits narrow it.
        for p in voxels {
            if !p.iter().all(|c| c.is_finite()) || p[0].hypot(p[1]) > RANGE_LIMIT_M {
                continue;
            }
            if f64::from(p[2]) <= median_z {
                self.seed.push([p[0], p[1], p[2]]);
            }
        }
        // Sorted so the fit does not depend on the order the cloud arrived
        // in. `GroundPlane::fit` accumulates a scatter matrix in f64, and f64
        // addition is not associative: the same seed summed in two orders can
        // differ in the last bits, which can move the plane by ~1e-14 m and —
        // in principle — flip the classification of a voxel sitting exactly on
        // the band. Sorting removes the argument rather than bounding it,
        // costs one sort of about half the in-range voxels, and allocates
        // nothing. `voxel.rs` pins its accumulation order the same way and for
        // the same reason.
        self.seed.sort_unstable_by(|a, b| {
            a[0].total_cmp(&b[0])
                .then(a[1].total_cmp(&b[1]))
                .then(a[2].total_cmp(&b[2]))
        });
        let ground = GroundPlane::fit(&self.seed, band, median_z);

        // Pass 3: the cells that are left.
        let mut ground_voxels = 0u32;
        for (row, p) in voxels.iter().enumerate() {
            if !p.iter().all(|c| c.is_finite()) || p[0].hypot(p[1]) > RANGE_LIMIT_M {
                continue;
            }
            let Some(key) = voxel_key(p, v) else { continue };
            // Below the plane counts as ground too. A return under the road
            // surface is not an object; it is the surface, measured low.
            if ground.height([p[0], p[1], p[2]]) < band {
                ground_voxels += 1;
                continue;
            }
            self.cells.push(Cell {
                key,
                row: row as u32,
            });
        }
        // `reduce` emits ascending voxel-key order and the recovered keys
        // preserve it — but this stage does not get to ASSUME that about its
        // input, because the binary search below is only correct on a sorted
        // slice. `(key, row)` is a total order, so no two entries compare
        // equal and there is nothing for an unstable sort to reorder.
        self.cells.sort_unstable();

        let n = self.cells.len();
        self.parent.extend(0..n as u32);
        // Equal keys are adjacent after the sort. Unioned rather than dropped,
        // so a collision merges two rows into one detection instead of losing
        // one, and counted so that it is never silent.
        let mut key_collisions = 0u32;
        for i in 1..n {
            if self.cells[i].key == self.cells[i - 1].key {
                key_collisions += 1;
                union(&mut self.parent, i as u32, (i - 1) as u32);
            }
        }
        for i in 0..n {
            let (ix, iy, iz) = unpack(self.cells[i].key);
            for off in FORWARD_NEIGHBOURS {
                let Some(nk) = pack(ix + off[0], iy + off[1], iz + off[2]) else {
                    continue;
                };
                let found = self.cells.binary_search_by_key(&nk, |c| c.key);
                if let Ok(j) = found {
                    union(&mut self.parent, i as u32, j as u32);
                }
            }
        }

        self.agg.resize(n, Agg::default());
        for i in 0..n {
            let r = find(&mut self.parent, i as u32) as usize;
            let p = voxels[self.cells[i].row as usize];
            let a = &mut self.agg[r];
            if a.n == 0 {
                a.min = [p[0], p[1], p[2]];
                a.max = [p[0], p[1], p[2]];
            } else {
                for (k, (lo, hi)) in a.min.iter_mut().zip(a.max.iter_mut()).enumerate() {
                    *lo = lo.min(p[k]);
                    *hi = hi.max(p[k]);
                }
            }
            a.n += 1;
            for (s, c) in a.sum.iter_mut().zip(&p) {
                *s += f64::from(*c);
            }
        }

        let mut clusters = 0u32;
        let mut detections = 0u32;
        let mut fragment_detections = 0u32;
        let mut merged_detections = 0u32;
        for a in &self.agg {
            if a.n == 0 {
                continue;
            }
            clusters += 1;
            if a.n < MIN_CLUSTER_VOXELS {
                continue;
            }
            detections += 1;
            match shape_of(a.min, a.max) {
                Shape::Fragment => fragment_detections += 1,
                Shape::Merged => merged_detections += 1,
                Shape::Object => {}
            }
        }

        DetectPlan {
            source_voxels: clamp_u32(voxels.len()),
            in_range_voxels,
            ground_voxels,
            non_finite,
            out_of_range,
            key_collisions,
            clusters,
            detections,
            fragment_detections,
            merged_detections,
            ground,
        }
    }

    /// Phase 2: allocate the detection buffer and fill it.
    ///
    /// This is where the stage allocates, and it **should**: producing a new
    /// Arrow result is the correct behaviour for a stage that transforms data,
    /// and it is not a violation of zero copy. What matters is that the input
    /// was never copied — phase 1 read it in place — and that this is exactly
    /// one buffer of exactly the output's size, so the measured number is the
    /// result and nothing else.
    ///
    /// Rows come out in ascending component-root order. A component's root is
    /// its lowest cell index, the cells are sorted by packed key, and the key
    /// is lexicographic in `(ix, iy, iz)` — so the output order is *each
    /// component's lexicographically smallest voxel, ascending*, which is
    /// independent of input order exactly as `reduce`'s is.
    #[allow(clippy::too_many_arguments)]
    pub fn build(
        &self,
        voxels: &[[f32; 4]],
        plan: &DetectPlan,
        voxel_size: VoxelSize,
        trigger_ns: i64,
        source_sweep_seq: i64,
        source_point_count: u32,
        schema: &Arc<Schema>,
    ) -> Result<(RecordBatch, usize), DetectError> {
        if plan.source_voxels as usize != voxels.len() {
            return Err(DetectError::PlanMismatch {
                planned: plan.source_voxels,
                got: voxels.len(),
            });
        }
        // The one allocation, and exactly the output's size: `from_len_zeroed`
        // takes a layout of precisely `len` (unlike `with_capacity`, which
        // rounds up), so the bytes this stage requests and the bytes the batch
        // carries are the same number.
        let mut buf = MutableBuffer::from_len_zeroed(plan.detections as usize * DETECTION_BYTES);
        // Checked before `typed_data_mut`, whose own alignment gate is an
        // `assert!`. It cannot fire on a `from_len_zeroed` buffer; this makes
        // that structural rather than a comment.
        let addr = buf.as_slice().as_ptr() as usize;
        if !addr.is_multiple_of(std::mem::align_of::<f32>()) {
            return Err(DetectError::Sweep(SweepError::Misaligned { addr }));
        }
        {
            let lanes = DETECTION_VALUES as usize;
            let out = buf.typed_data_mut::<f32>();
            let mut w = 0usize;
            for a in &self.agg {
                if a.n < MIN_CLUSTER_VOXELS {
                    continue;
                }
                let n = f64::from(a.n);
                let lane = &mut out[w * lanes..][..lanes];
                for (k, (lo, hi)) in a.min.iter().zip(a.max.iter()).enumerate() {
                    lane[k] = (a.sum[k] / n) as f32;
                    lane[3 + k] = *lo;
                    lane[6 + k] = *hi;
                }
                // Exact: f32 holds every integer below 2^24 and a whole voxel
                // cloud is ~35,000 rows. See [`DETECTION_BYTES`].
                lane[9] = a.n as f32;
                lane[10] = (a.sum[3] / n) as f32;
                w += 1;
            }
            debug_assert_eq!(w, plan.detections as usize);
        }
        let (det_arr, storage_id, n_det) = build_detections_array(Buffer::from(buf))?;
        let g = plan.ground;
        let plane = Float32Array::from(vec![
            g.n[0] as f32,
            g.n[1] as f32,
            g.n[2] as f32,
            g.d as f32,
        ]);
        let plane_arr =
            FixedSizeListArray::try_new(ground_plane_field(), 4, Arc::new(plane) as ArrayRef, None)
                .map_err(SweepError::Arrow)?;
        let columns: Vec<ArrayRef> = vec![
            Arc::new(UInt32Array::from(vec![clamp_u32(n_det)])),
            Arc::new(StringArray::from(vec![DETECTION_FORMAT])),
            Arc::new(Int64Array::from(vec![trigger_ns])),
            Arc::new(Int64Array::from(vec![source_sweep_seq])),
            Arc::new(Float32Array::from(vec![voxel_size.metres()])),
            Arc::new(Float32Array::from(vec![RANGE_LIMIT_M])),
            Arc::new(UInt32Array::from(vec![MIN_CLUSTER_VOXELS])),
            Arc::new(UInt32Array::from(vec![plan.source_voxels])),
            Arc::new(UInt32Array::from(vec![source_point_count])),
            Arc::new(plane_arr),
            Arc::new(BooleanArray::from(vec![g.fitted])),
            det_arr,
        ];
        Ok((
            RecordBatch::try_new(Arc::clone(schema), columns).map_err(SweepError::Arrow)?,
            storage_id,
        ))
    }
}

/// The one-row `detections` column, wrapping `buf` **without copying it**.
/// Returns the column, the `storage_id` and the detection count.
fn build_detections_array(buf: Buffer) -> Result<(ArrayRef, usize, usize), SweepError> {
    let len = buf.len();
    if !len.is_multiple_of(DETECTION_BYTES) {
        return Err(SweepError::NotWholePoints { got: len });
    }
    let storage_id = buf.as_ptr() as usize;
    // Checked before `ScalarBuffer::from`, whose own check is an `assert!`.
    if !storage_id.is_multiple_of(std::mem::align_of::<f32>()) {
        return Err(SweepError::Misaligned { addr: storage_id });
    }
    let n = len / DETECTION_BYTES;
    let values = Float32Array::new(ScalarBuffer::<f32>::from(buf), None);
    let det = FixedSizeListArray::try_new(
        detection_value_field(),
        DETECTION_VALUES,
        Arc::new(values) as ArrayRef,
        None,
    )?;
    let offsets = OffsetBuffer::<i64>::new(ScalarBuffer::<i64>::from(vec![0i64, n as i64]));
    let arr = LargeListArray::try_new(detection_field(), offsets, Arc::new(det), None)?;
    Ok((Arc::new(arr) as ArrayRef, storage_id, n))
}

/// Packed grid key of a voxel centroid, or `None` if it cannot have one.
///
/// `floor`, in f64, exactly the arithmetic `voxel.rs` performs — run on the
/// centroid instead of on the point, which is what makes the grid recoverable.
/// See [`DetectPlan::key_collisions`].
fn voxel_key(p: &[f32; 4], v: f64) -> Option<i64> {
    let ix = axis_index(p[0], v)?;
    let iy = axis_index(p[1], v)?;
    let iz = axis_index(p[2], v)?;
    pack(ix, iy, iz)
}

fn axis_index(c: f32, v: f64) -> Option<i64> {
    let q = (f64::from(c) / v).floor();
    // A positive range test, so a NaN quotient fails it rather than passing.
    (q >= -(INDEX_BIAS as f64) && q < INDEX_BIAS as f64).then_some(q as i64)
}

/// Packs signed axis indices into one `i64`; `None` outside the grid.
fn pack(ix: i64, iy: i64, iz: i64) -> Option<i64> {
    let bias = |i: i64| -> Option<i64> {
        let b = i + INDEX_BIAS;
        (0..(1i64 << INDEX_BITS)).contains(&b).then_some(b)
    };
    Some((bias(ix)? << (2 * INDEX_BITS)) | (bias(iy)? << INDEX_BITS) | bias(iz)?)
}

/// Inverse of [`pack`].
fn unpack(key: i64) -> (i64, i64, i64) {
    let mask = (1i64 << INDEX_BITS) - 1;
    (
        ((key >> (2 * INDEX_BITS)) & mask) - INDEX_BIAS,
        ((key >> INDEX_BITS) & mask) - INDEX_BIAS,
        (key & mask) - INDEX_BIAS,
    )
}

/// Union-find root of `i`, with path halving.
fn find(parent: &mut [u32], mut i: u32) -> u32 {
    while parent[i as usize] != i {
        let g = parent[parent[i as usize] as usize];
        parent[i as usize] = g;
        i = g;
    }
    i
}

/// Union by **lower index wins**, not by rank.
///
/// Deliberate: it makes each component's root its lowest cell index, which —
/// the cells being sorted by packed key — is the component's lexicographically
/// smallest voxel. The output order then falls out of the data structure
/// instead of needing a sort afterwards, and it does not depend on the order
/// the unions happened in. Union by rank would be marginally faster and would
/// leave the root arbitrary.
fn union(parent: &mut [u32], a: u32, b: u32) {
    let (ra, rb) = (find(parent, a), find(parent, b));
    if ra != rb {
        let (lo, hi) = if ra < rb { (ra, rb) } else { (rb, ra) };
        parent[hi as usize] = lo;
    }
}

/// `usize` as `u32`, saturating. Only ever reached with voxel counts, which
/// are five digits on this dataset; saturating beats wrapping because a count
/// that read as small would be believed.
fn clamp_u32(n: usize) -> u32 {
    u32::try_from(n).unwrap_or(u32::MAX)
}

/// The ego vehicle's own motion between two sweeps.
///
/// Fields 8, 9 and 19 of a KITTI `oxts/data/*.txt` row: forward and leftward
/// velocity in the vehicle frame, and yaw rate.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct EgoMotion {
    /// Forward velocity, m/s (OXTS `vf`).
    pub vf: f64,
    /// Leftward velocity, m/s (OXTS `vl`).
    pub vl: f64,
    /// Yaw rate, rad/s (OXTS `wz`).
    pub yaw_rate: f64,
    /// Seconds between the two sweeps.
    pub dt: f64,
}

impl EgoMotion {
    /// Where a point fixed to the world, observed at `t` in the vehicle frame,
    /// appears at `t + dt` in the vehicle frame.
    ///
    /// Translate by what the vehicle travelled, then rotate by minus what it
    /// turned. Planar: KITTI's lidar frame is vehicle-fixed and this drive's
    /// pitch and roll rates are noise next to its yaw rate, so z is carried
    /// through unchanged.
    pub fn carry(&self, p: [f32; 3]) -> [f32; 3] {
        let (dx, dy) = (self.vf * self.dt, self.vl * self.dt);
        let psi = -self.yaw_rate * self.dt;
        let (s, c) = psi.sin_cos();
        let (x, y) = (f64::from(p[0]) - dx, f64::from(p[1]) - dy);
        [(c * x - s * y) as f32, (s * x + c * y) as f32, p[2]]
    }
}

/// Result of the persistence check: how many detections of one sweep had a
/// counterpart in the next.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Persistence {
    /// Detections of the earlier sweep that found a match.
    pub matched: u32,
    /// Detections of the earlier sweep that were tested.
    pub total: u32,
}

impl Persistence {
    /// Matched share, or `None` when nothing was tested.
    pub fn fraction(&self) -> Option<f64> {
        (self.total > 0).then(|| f64::from(self.matched) / f64::from(self.total))
    }

    /// Adds another sweep pair's counts.
    pub fn add(&mut self, other: Persistence) {
        self.matched += other.matched;
        self.total += other.total;
    }
}

/// **The sanity check that needs no labels.** Carry each detection of `prev`
/// forward by the measured ego motion and ask whether the next sweep has one
/// within `gate_m`.
///
/// A detector emitting noise scores near zero here, and — the part that gives
/// the check teeth — **a detector emitting noise could not be improved by
/// applying the correct physical motion**. Measured over all 153 consecutive
/// pairs of drive_0005:
///
/// ```text
/// ego-compensated, real detections            71.8 %  (sd 4.6)
/// uncompensated, real detections              35.9 %
/// control: uniform-random points               1.1 %
/// control: the same detections turned 90 deg   1.8 %
/// ```
///
/// Both controls matter and the second is the stronger: rotating the
/// detections leaves their spatial statistics **identical**, so 1.8 % against
/// 71.8 % cannot be explained by density. Compensation doubling the rate is a
/// second, independent signal — a statement about the physics rather than
/// about the count.
///
/// `gate_m` is **not derived, and this is the place to say so** rather than
/// bury it. 0.5 m is where the investigation measured. It sits above the 0.4 m
/// an object closing at 4 m/s covers in one 10 Hz interval and above the 0.2 m
/// quantisation a voxel centroid carries, which is why it works — but it was
/// not computed from either.
///
/// Nearest neighbour, not a mutually exclusive assignment: this measures
/// whether the *structure* persists, and an assignment algorithm would add a
/// second thing whose failures would be read as the detector's.
pub fn persistence(prev: &[f32], next: &[f32], ego: EgoMotion, gate_m: f32) -> Persistence {
    let gate2 = f64::from(gate_m) * f64::from(gate_m);
    let mut out = Persistence::default();
    let (next, _) = next.as_chunks::<DETECTION_LANES>();
    let (prev, _) = prev.as_chunks::<DETECTION_LANES>();
    for a in prev {
        out.total += 1;
        let c = ego.carry([a[0], a[1], a[2]]);
        let hit = next.iter().any(|b| {
            let dx = f64::from(b[0]) - f64::from(c[0]);
            let dy = f64::from(b[1]) - f64::from(c[1]);
            let dz = f64::from(b[2]) - f64::from(c[2]);
            dx * dx + dy * dy + dz * dz <= gate2
        });
        out.matched += u32::from(hit);
    }
    out
}

/// The detections of a detect batch as fixed-size records — the shape
/// [`crate::track::Tracker::plan`] takes — or an empty slice.
///
/// A helper rather than open-coding `as_chunks` at each call site, for the
/// reason [`cloud_points`] gives: the remainder it returns has to be discarded
/// deliberately, and a payload whose length is not a whole number of
/// detections cannot reach here because the Arrow builder rejects it.
pub fn detection_rows(batch: &RecordBatch) -> &[[f32; DETECTION_LANES]] {
    let lanes: &[f32] = detections_f32(batch).unwrap_or(&[]);
    let (rows, _) = lanes.as_chunks::<DETECTION_LANES>();
    rows
}

/// The interleaved `xyzr` of a cloud batch as fixed-size points — the shape
/// [`Detector::plan`] takes — or an empty slice.
///
/// A helper rather than open-coding `as_chunks` at each call site, because the
/// remainder it returns has to be discarded deliberately: a payload whose
/// length is not a whole number of points cannot reach here (the Arrow builder
/// rejects it), and silently dropping a partial point somewhere else would be
/// a different decision than this one.
pub fn cloud_points(batch: &RecordBatch) -> &[[f32; 4]] {
    let xyzr: &[f32] = velo_xyzr(batch).unwrap_or(&[]);
    let (points, _) = xyzr.as_chunks::<4>();
    points
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use pipes_core::sample::payload_bytes;

    use super::*;
    use crate::velo::velo_point_count;
    use crate::voxel::{voxel_schema, Voxelizer, DEFAULT_VOXEL_SIZE_M};

    const V: f32 = DEFAULT_VOXEL_SIZE_M;
    /// Road tilt of the scenes below. Under the 1.86 deg this drive's fitted
    /// planes actually reach, so it is a tilt the real data has rather than
    /// one invented to make the fit look necessary.
    const TILT_DEG: f64 = 1.5;
    /// KITTI's velodyne sits about 1.73 m above the road.
    const Z0: f32 = -1.73;

    fn size() -> VoxelSize {
        VoxelSize::new(V).unwrap()
    }

    fn ground_z(x: f32, tilt_deg: f64) -> f32 {
        Z0 + (f64::from(x) * tilt_deg.to_radians().tan()) as f32
    }

    /// A road surface: a grid `step` apart over `+-half_x * step` by
    /// `+-half_y * step`, tilted about the y axis.
    fn ground_points(tilt_deg: f64, half_x: i32, half_y: i32, step: f32) -> Vec<[f32; 4]> {
        let mut v = Vec::with_capacity((4 * half_x * half_y) as usize);
        for i in -half_x..half_x {
            for j in -half_y..half_y {
                let (x, y) = (i as f32 * step, j as f32 * step);
                v.push([x, y, ground_z(x, tilt_deg), 0.1]);
            }
        }
        v
    }

    /// A 0.6 m square, 1.8 m tall block standing on the road at `(cx, cy)` --
    /// the pedestrian the voxel edge was sized for.
    fn box_points(cx: f32, cy: f32, tilt_deg: f64) -> Vec<[f32; 4]> {
        let base = ground_z(cx, tilt_deg);
        let mut v = Vec::with_capacity(7 * 7 * 19);
        for i in 0..=6 {
            for j in 0..=6 {
                for k in 0..=18 {
                    v.push([
                        cx + i as f32 * 0.1 - 0.3,
                        cy + j as f32 * 0.1 - 0.3,
                        base + k as f32 * 0.1,
                        0.6,
                    ]);
                }
            }
        }
        v
    }

    /// Through the REAL reduction, so these tests run against the shape
    /// `reduce` actually emits rather than a hand-made lookalike. A detect
    /// stage tested on raw points would not exercise the one thing it has to
    /// do to its input: recover the grid from the centroids.
    fn voxelize(points: &[[f32; 4]]) -> Vec<[f32; 4]> {
        let mut vx = Voxelizer::new();
        vx.reserve(points.len());
        let plan = vx.plan(points, size());
        let (b, _) = vx.build(points, &plan, size(), 0, &voxel_schema()).unwrap();
        let (out, rest) = velo_xyzr(&b).unwrap().as_chunks::<4>();
        assert!(rest.is_empty(), "a partial point came out of the reduction");
        out.to_vec()
    }

    /// A point at the centre of grid cell `(ix, iy, iz)`.
    ///
    /// Cell centres rather than round metre values, and it is not fussiness:
    /// `2.0 / 0.2` lands just under 10 in f64 while `2.2 / 0.2` lands just
    /// over 11, so points placed on round coordinates are not on the grid
    /// steps they look like they are on, and a test about ADJACENCY built from
    /// them tests the wrong cells. A centre is half a cell from either wall.
    fn cell_point(ix: i32, iy: i32, iz: i32, r: f32) -> [f32; 4] {
        [
            (ix as f32 + 0.5) * V,
            (iy as f32 + 0.5) * V,
            (iz as f32 + 0.5) * V,
            r,
        ]
    }

    fn scene(tilt_deg: f64, half_x: i32, half_y: i32, boxes: &[(f32, f32)]) -> Vec<[f32; 4]> {
        let mut pts = ground_points(tilt_deg, half_x, half_y, 0.15);
        for &(cx, cy) in boxes {
            pts.extend(box_points(cx, cy, tilt_deg));
        }
        voxelize(&pts)
    }

    fn run(voxels: &[[f32; 4]]) -> (RecordBatch, DetectPlan) {
        let mut d = Detector::new();
        d.reserve(voxels.len());
        let plan = d.plan(voxels, size());
        let (b, _) = d
            .build(voxels, &plan, size(), 4242, 17, 99_999, &detect_schema())
            .unwrap();
        (b, plan)
    }

    fn lanes(batch: &RecordBatch) -> Vec<[f32; DETECTION_LANES]> {
        detections_f32(batch).unwrap().as_chunks().0.to_vec()
    }

    /// The claim the whole stage exists for: a cloud of points becomes a
    /// handful of objects, and the transfer collapses because the OUTPUT SAYS
    /// SOMETHING ELSE rather than because it was thinned.
    #[test]
    fn a_road_with_three_things_on_it_becomes_three_detections() {
        let vox = scene(TILT_DEG, 67, 40, &[(3.0, -2.0), (3.0, 2.0), (6.0, 0.0)]);
        let (batch, plan) = run(&vox);

        assert_eq!(plan.detections, 3, "plan = {plan:?}");
        assert_eq!(
            plan.clusters, 3,
            "something other than the three blocks survived ground removal: {plan:?}"
        );
        assert_eq!(plan.key_collisions, 0);
        assert_eq!((plan.non_finite, plan.out_of_range), (0, 0));

        // Conservation: every in-range voxel is either ground or inside a
        // detection, because this scene has no sub-gate clusters. It is what
        // makes `voxel_count` an accounting of the input rather than a number
        // the stage is free to get wrong.
        let counted: u32 = lanes(&batch).iter().map(|d| d[9] as u32).sum();
        assert_eq!(
            plan.ground_voxels + counted,
            plan.in_range_voxels,
            "{} ground + {counted} in detections != {} in range",
            plan.ground_voxels,
            plan.in_range_voxels
        );

        // Each detection is the block that was put there: 0.6 m square, and
        // tall -- the bottom of the block falls inside the ground band and is
        // removed with it, which is why the height is under 1.8 m.
        for d in lanes(&batch) {
            let ext = [d[6] - d[3], d[7] - d[4], d[8] - d[5]];
            assert!(ext[0] <= 0.85 && ext[1] <= 0.85, "footprint {ext:?}");
            assert!((1.0..1.85).contains(&ext[2]), "height {ext:?}");
            // The centroid is inside its own bounding box: the cheapest
            // possible check that the two are not swapped.
            for k in 0..3 {
                assert!(d[3 + k] <= d[k] && d[k] <= d[6 + k], "centroid {d:?}");
            }
            assert!(d[9] >= MIN_CLUSTER_VOXELS as f32, "{d:?}");
            assert!((0.55..0.65).contains(&d[10]), "reflectance {d:?}");
        }

        // The byte chain's third link, in its smallest honest form.
        let n = plan.detections as usize;
        assert_eq!(detections_f32(&batch).unwrap().len(), n * 11);
        assert!(
            payload_bytes(&batch) < vox.len() * 16,
            "{} B of detections against {} B of voxels",
            payload_bytes(&batch),
            vox.len() * 16
        );
    }

    /// Ground removal is the PRECONDITION, not a nicety, and a tilted road is
    /// where a fixed `z` threshold stops working.
    ///
    /// The positive control is the second half: it counts the ground voxels
    /// sitting ABOVE `median z + one band` -- the ones a fixed threshold at
    /// the same nominal height would have kept -- and asserts there are many.
    /// Without it, "the fit removed the ground" would pass just as well on a
    /// scene where a fixed threshold would also have removed it, and the test
    /// would assert nothing about the fit at all.
    #[test]
    fn the_fitted_plane_removes_ground_a_fixed_threshold_would_keep() {
        let vox = scene(TILT_DEG, 67, 40, &[(3.0, -2.0), (3.0, 2.0), (6.0, 0.0)]);
        let (_, plan) = run(&vox);

        assert!(plan.ground.fitted, "the fit fell back: {:?}", plan.ground);
        assert!(
            (plan.ground.tilt_deg() - TILT_DEG).abs() < 0.05,
            "fitted tilt {:.3} deg against the scene's {TILT_DEG}",
            plan.ground.tilt_deg()
        );

        let mut zs: Vec<f32> = vox.iter().map(|p| p[2]).collect();
        zs.sort_unstable_by(f32::total_cmp);
        let median_z = zs[zs.len() / 2];
        let missed = vox
            .iter()
            .filter(|p| {
                p[2] >= median_z + V && plan.ground.height([p[0], p[1], p[2]]) < f64::from(V)
            })
            .count();
        assert!(
            missed > 300,
            "only {missed} voxels distinguish the fitted plane from a fixed \
             threshold on this scene, so this test does not test the fit"
        );
        // And those voxels really were removed: had they survived they would
        // have bridged the three blocks into one component through the road.
        assert_eq!(plan.clusters, 3);
    }

    /// The cluster gate, with its own positive control in the same test.
    ///
    /// A test that only showed the 2-voxel blob being rejected would pass
    /// against a stage that rejected everything, so the 3-voxel blob is put in
    /// the identical place and must be reported.
    #[test]
    fn a_cluster_smaller_than_the_object_the_grid_was_sized_for_is_not_a_detection() {
        let boxes = [(3.0, -2.0), (3.0, 2.0), (6.0, 0.0)];
        let blob = |n: usize| -> Vec<[f32; 4]> {
            (0..n)
                .map(|i| {
                    [
                        -5.0 + i as f32 * 0.25,
                        4.0,
                        ground_z(-5.0, TILT_DEG) + 1.0,
                        0.4,
                    ]
                })
                .collect()
        };
        for (n, want_clusters, want_detections) in [(2usize, 4u32, 3u32), (3, 4, 4)] {
            let mut pts = ground_points(TILT_DEG, 67, 40, 0.15);
            for &(cx, cy) in &boxes {
                pts.extend(box_points(cx, cy, TILT_DEG));
            }
            pts.extend(blob(n));
            let (_, plan) = run(&voxelize(&pts));
            assert_eq!(
                (plan.clusters, plan.detections),
                (want_clusters, want_detections),
                "a {n}-voxel blob: {plan:?}"
            );
        }
    }

    /// Same cloud, same detections -- twice over, and then under a permutation
    /// of the input rows, which is the stronger statement. See the module
    /// docs' determinism section for the one case that is not covered.
    #[test]
    fn the_same_cloud_gives_the_same_bytes_however_it_is_ordered() {
        let vox = scene(TILT_DEG, 67, 40, &[(3.0, -2.0), (3.0, 2.0), (6.0, 0.0)]);
        let (a, pa) = run(&vox);
        let (b, pb) = run(&vox);
        assert_eq!(detections_f32(&a).unwrap(), detections_f32(&b).unwrap());
        assert_eq!(pa, pb);

        let mut reversed = vox.clone();
        reversed.reverse();
        let (c, pc) = run(&reversed);
        assert_eq!(pc.key_collisions, 0, "the permutation case is not covered");
        assert_eq!(
            detections_f32(&a).unwrap(),
            detections_f32(&c).unwrap(),
            "reversing the input changed the detections"
        );
        assert_eq!(pa, pc);

        // A permutation that is not a reversal, in case a reversal happens to
        // be a symmetry of the accumulation.
        let mut strided: Vec<[f32; 4]> = Vec::with_capacity(vox.len());
        for start in 0..7 {
            strided.extend(vox.iter().skip(start).step_by(7).copied());
        }
        assert_eq!(strided.len(), vox.len());
        let (d, pd) = run(&strided);
        assert_eq!(detections_f32(&a).unwrap(), detections_f32(&d).unwrap());
        assert_eq!(pa, pd);
    }

    /// The degenerate cases the fit must refuse rather than answer.
    ///
    /// Each one returns a plane that is *defensible* if you do not look: a
    /// collinear seed has infinitely many planes through it and a solver will
    /// hand back one of them; a wall is a perfectly good plane that simply is
    /// not the ground. Both are reported as `fitted = false` and fall back to
    /// the horizontal plane, which is the honest answer.
    #[test]
    fn a_seed_that_does_not_determine_a_plane_says_so() {
        let band = f64::from(V);

        let line: Vec<[f32; 3]> = (0..200)
            .map(|i| [i as f32 * 0.1, -(i as f32) * 0.1, -1.7])
            .collect();
        let p = GroundPlane::fit(&line, band, -1.7);
        assert!(!p.fitted, "a line was accepted as a plane: {p:?}");
        assert_eq!(p.n, [0.0, 0.0, 1.0]);

        // A strip of road one band wide: spread over 8 m in x, 0.05 m in
        // y. The least-spread direction IS vertical, so the verticality guard
        // passes it -- only the degeneracy guard can refuse it, and it must,
        // because a 5 cm-wide strip of ground says nothing about the road's
        // cross-slope.
        let strip: Vec<[f32; 3]> = (0..400)
            .map(|i| [i as f32 * 0.02, (i % 3) as f32 * 0.025, -1.7])
            .collect();
        let p = GroundPlane::fit(&strip, band, -1.7);
        assert!(!p.fitted, "a one-band-wide strip determined a plane: {p:?}");

        let wall: Vec<[f32; 3]> = (0..40)
            .flat_map(|i| (0..40).map(move |j| [0.0, i as f32 * 0.2, j as f32 * 0.2]))
            .collect();
        let p = GroundPlane::fit(&wall, band, 0.0);
        assert!(!p.fitted, "a wall was accepted as the ground: {p:?}");

        assert!(!GroundPlane::fit(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], band, 0.0).fitted);

        // Positive control: a real patch IS accepted, so the three above are
        // not passing against a fit that never succeeds.
        let patch: Vec<[f32; 3]> = (0..40)
            .flat_map(|i| (0..40).map(move |j| [i as f32 * 0.2, j as f32 * 0.2, -1.7]))
            .collect();
        let p = GroundPlane::fit(&patch, band, -1.7);
        assert!(p.fitted, "{p:?}");
        assert!(p.tilt_deg() < 1e-6, "{p:?}");
    }

    /// 26-connectivity, not 6: a surface that crosses a cell only diagonally
    /// must not be cut into pieces.
    ///
    /// The diagonal chain is 26-connected and NOT 6-connected, so it is one
    /// object under the rule this stage uses and four singletons under the
    /// other -- four singletons being below the cluster gate, it would vanish
    /// entirely. The axis-aligned chain beside it is the positive control: it
    /// is one object under either rule, so a failure on the first one cannot
    /// be "this scene produces no detections".
    #[test]
    fn a_chain_that_is_only_diagonally_connected_is_one_object() {
        let mut pts = ground_points(0.0, 40, 30, 0.15);
        for i in 0..4 {
            pts.push(cell_point(-15 + i, 9 + i, -4, 0.7));
            pts.push(cell_point(10 + i, -12, -4, 0.7));
        }
        let (batch, plan) = run(&voxelize(&pts));
        assert_eq!(
            (plan.clusters, plan.detections),
            (2, 2),
            "the diagonal chain was cut up: {plan:?}"
        );
        for d in lanes(&batch) {
            assert_eq!(d[9] as u32, 4, "{d:?}");
        }
    }

    /// Detections come out in ascending order of each component's
    /// lexicographically smallest voxel, which is what makes the output an
    /// order a replay can rely on rather than whatever the union-find happened
    /// to produce.
    ///
    /// The two bars are deliberately INTERLEAVED in key order -- A starts
    /// before B and ends after it -- because the obvious scene of two
    /// well-separated blobs is ordered the same way by smallest voxel and by
    /// largest, so it cannot tell the documented rule from its opposite.
    #[test]
    fn detections_come_out_in_ascending_voxel_order() {
        let mut pts = ground_points(0.0, 40, 30, 0.15);
        // B first in the input, so a stage that emitted arrival order fails.
        for ix in 1..4 {
            pts.push(cell_point(ix, 6, -4, 0.7));
        }
        for ix in 0..5 {
            pts.push(cell_point(ix, 0, -4, 0.7));
        }
        let (batch, plan) = run(&voxelize(&pts));
        assert_eq!((plan.clusters, plan.detections), (2, 2), "{plan:?}");
        let d = lanes(&batch);
        assert_eq!(
            (d[0][9] as u32, d[1][9] as u32),
            (5, 3),
            "the 5-voxel bar starts at the lower voxel key and must come first"
        );
        let key = |x: f32, y: f32| (((x / V).floor() as i64) << 21) | ((y / V).floor() as i64);
        assert!(
            key(d[0][3], d[0][4]) < key(d[1][3], d[1][4]),
            "detections are not in ascending voxel order: {:?} then {:?}",
            &d[0][3..6],
            &d[1][3..6]
        );
    }

    /// The eigensolver, against answers that can be written down.
    #[test]
    fn the_eigensolver_returns_an_ascending_orthonormal_basis() {
        let (l, v) = sym_eigen3([[3.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 2.0]]);
        for (got, want) in l.iter().zip([1.0, 2.0, 3.0]) {
            assert!((got - want).abs() < 1e-12, "{l:?}");
        }
        // The eigenvector of the smallest eigenvalue is +-e1.
        assert!((v[1][0].abs() - 1.0).abs() < 1e-12, "{v:?}");

        // [[2,1,0],[1,2,0],[0,0,5]] has eigenvalues 1, 3, 5.
        let (l, v) = sym_eigen3([[2.0, 1.0, 0.0], [1.0, 2.0, 0.0], [0.0, 0.0, 5.0]]);
        for (got, want) in l.iter().zip([1.0, 3.0, 5.0]) {
            assert!((got - want).abs() < 1e-12, "{l:?}");
        }
        for k in 0..3 {
            let norm: f64 = (0..3).map(|r| v[r][k] * v[r][k]).sum();
            assert!(
                (norm - 1.0).abs() < 1e-12,
                "column {k} is not a unit vector"
            );
            for k2 in (k + 1)..3 {
                let dot: f64 = (0..3).map(|r| v[r][k] * v[r][k2]).sum();
                assert!(dot.abs() < 1e-12, "columns {k} and {k2} are not orthogonal");
            }
        }
    }

    /// A deterministic generator, so the random control below is a control and
    /// not a source of flakiness.
    struct Lcg(u64);

    impl Lcg {
        fn unit(&mut self) -> f64 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (self.0 >> 11) as f64 / (1u64 << 53) as f64
        }
    }

    /// **The sanity check, with the two controls that give it teeth.**
    ///
    /// The scene is fixed to the world and the vehicle drives through it, so
    /// the second sweep's detections are the first sweep's carried by a motion
    /// the test knows. Four numbers come out and only the pattern across all
    /// four means anything:
    ///
    /// * compensated by the real motion -- high;
    /// * not compensated -- lower, because a detector emitting noise could not
    ///   be *improved* by applying the correct physics, so this gap is a claim
    ///   about the world rather than about the count;
    /// * matched against the same detections turned 90 deg -- near zero, and
    ///   this is the sharp one: the control has **identical** spatial
    ///   statistics, so it cannot be explained away by density;
    /// * matched against uniform-random points over the region the stage can
    ///   report in -- near zero.
    ///
    /// The bounds asserted are the ones the real drive is measured against
    /// (>50 % and <5 %, against 71.8 % and 1.1-1.8 % there). The contrast here
    /// is sharper than on the real drive by construction: this vehicle moves
    /// further between sweeps than KITTI's does, so the uncompensated case
    /// collapses completely instead of falling to 35.9 %.
    #[test]
    fn detections_persist_under_the_real_motion_and_not_under_the_controls() {
        let ego = EgoMotion {
            vf: 6.0,
            vl: 0.0,
            yaw_rate: 0.3,
            dt: 0.1,
        };
        let world: Vec<(f32, f32)> = (0..6)
            .flat_map(|i| (0..4).map(move |j| (4.0 + i as f32 * 4.0, -6.0 + j as f32 * 4.0)))
            .collect();
        // Where the vehicle SEES each object one interval later, written out
        // here rather than obtained from `EgoMotion::carry`.
        //
        // This is not duplication for its own sake. The first version of this
        // test built the scene with `carry` and then checked `carry` against
        // it, so deleting the translation term from `carry` deleted it from
        // the fixture too and every assertion still passed -- a test that
        // could not fail, verified by planting exactly that bug. The fixture
        // and the subject have to be separate code.
        let psi = -ego.yaw_rate * ego.dt;
        let (sin_psi, cos_psi) = psi.sin_cos();
        let moved: Vec<(f32, f32)> = world
            .iter()
            .map(|&(x, y)| {
                let (x, y) = (
                    f64::from(x) - ego.vf * ego.dt,
                    f64::from(y) - ego.vl * ego.dt,
                );
                (
                    (cos_psi * x - sin_psi * y) as f32,
                    (sin_psi * x + cos_psi * y) as f32,
                )
            })
            .collect();
        // Flat road: this test is about association, and a tilt would only add
        // a second thing that could explain a failure.
        let (b0, p0) = run(&scene(0.0, 96, 60, &world));
        let (b1, p1) = run(&scene(0.0, 96, 60, &moved));
        assert_eq!((p0.detections, p1.detections), (24, 24), "{p0:?} {p1:?}");
        let (d0, d1) = (detections_f32(&b0).unwrap(), detections_f32(&b1).unwrap());

        let gate = 0.5;
        let real = persistence(d0, d1, ego, gate);
        let uncompensated = persistence(
            d0,
            d1,
            EgoMotion {
                dt: ego.dt,
                ..Default::default()
            },
            gate,
        );
        assert_eq!(real.total, 24);
        assert!(
            real.fraction().unwrap() > 0.5,
            "ego-compensated persistence {:?}",
            real.fraction()
        );
        assert!(
            uncompensated.fraction().unwrap() < real.fraction().unwrap(),
            "applying the real motion did not improve the match: {real:?} vs {uncompensated:?}"
        );

        // Control 1: the same detections, turned 90 deg about the sensor.
        // Identical spatial statistics, so a high score here would mean the
        // real number was density and not structure.
        let mut turned_lanes = d1.to_vec();
        for d in turned_lanes.as_chunks_mut::<DETECTION_LANES>().0 {
            let (x, y) = (d[0], d[1]);
            d[0] = -y;
            d[1] = x;
        }
        let turned = persistence(d0, &turned_lanes, ego, gate);

        // Control 2: uniform-random points over the disc the stage can report
        // in, at the heights the real detections occupy.
        let zs: Vec<f32> = d1
            .as_chunks::<DETECTION_LANES>()
            .0
            .iter()
            .map(|d| d[2])
            .collect();
        let (zlo, zhi) = (
            zs.iter().copied().fold(f32::INFINITY, f32::min),
            zs.iter().copied().fold(f32::NEG_INFINITY, f32::max),
        );
        let mut rng = Lcg(0x5EED);
        let mut random_lanes = vec![0f32; d1.len()];
        for d in random_lanes.as_chunks_mut::<DETECTION_LANES>().0 {
            let r = f64::from(RANGE_LIMIT_M) * rng.unit().sqrt();
            let th = std::f64::consts::TAU * rng.unit();
            d[0] = (r * th.cos()) as f32;
            d[1] = (r * th.sin()) as f32;
            d[2] = zlo + (zhi - zlo) * rng.unit() as f32;
        }
        let random = persistence(d0, &random_lanes, ego, gate);

        for (name, c) in [("turned 90 deg", turned), ("uniform random", random)] {
            assert_eq!(c.total, 24, "{name}");
            assert!(
                c.fraction().unwrap() < 0.05,
                "control `{name}` scored {:?}, so the check does not discriminate",
                c.fraction()
            );
        }
    }

    /// A detect batch is read by the detect accessors and by nothing else.
    ///
    /// The second half is the point. `reduce` was built so that the sweep's
    /// accessors read its output unchanged; this stage deliberately breaks
    /// that, because its output is not a point cloud. If `velo_xyzr` ever
    /// starts returning something here, every existing cloud consumer will
    /// silently read a bounding-box corner as a position.
    #[test]
    fn a_detect_batch_is_self_describing_and_is_not_a_point_cloud() {
        let vox = scene(TILT_DEG, 67, 40, &[(3.0, -2.0), (6.0, 0.0)]);
        let (batch, plan) = run(&vox);

        assert_eq!(detection_count(&batch), Some(plan.detections));
        assert_eq!(detection_format(&batch), Some(DETECTION_FORMAT));
        assert_eq!(detect_trigger_ns(&batch), Some(4242));
        assert_eq!(detect_voxel_size_m(&batch), Some(V));
        assert_eq!(detect_range_limit_m(&batch), Some(RANGE_LIMIT_M));
        assert_eq!(source_voxel_count(&batch), Some(plan.source_voxels));
        assert_eq!(source_point_count(&batch), Some(99_999));
        let (plane, fitted) = ground_plane(&batch).unwrap();
        assert!(fitted);
        assert!(
            (f64::from(plane[2]) - plan.ground.n[2]).abs() < 1e-6,
            "{plane:?}"
        );
        assert_eq!(
            detect_storage_id(&batch),
            Some(detections_f32(&batch).unwrap().as_ptr() as usize)
        );

        assert!(
            velo_xyzr(&batch).is_none(),
            "a detection batch reads as a point cloud, so every cloud consumer \
             will read a bounding-box corner as a position"
        );
        assert!(velo_point_count(&batch).is_none());
    }

    /// The scratch is sized once and then stops growing, which is what makes
    /// the stage's per-sample allocation its output and nothing else.
    #[test]
    fn the_scratch_stops_growing() {
        let vox = scene(TILT_DEG, 40, 30, &[(3.0, 0.0)]);
        let mut d = Detector::new();
        assert!(d.reserve(vox.len()), "the first reserve must allocate");
        let bytes = d.scratch_bytes();
        assert!(bytes > 0);
        for _ in 0..3 {
            let plan = d.plan(&vox, size());
            assert!(plan.detections > 0);
            assert!(
                !d.reserve(vox.len()),
                "the scratch grew in the steady state"
            );
            assert_eq!(d.scratch_bytes(), bytes);
        }
    }

    /// The plan indexes the cloud it was given, so building against a
    /// different one is a typed error rather than the wrong answer.
    #[test]
    fn building_against_a_different_cloud_is_refused() {
        let vox = scene(TILT_DEG, 40, 30, &[(3.0, 0.0)]);
        let mut d = Detector::new();
        d.reserve(vox.len());
        let plan = d.plan(&vox, size());
        let err = d
            .build(
                &vox[..vox.len() - 1],
                &plan,
                size(),
                0,
                -1,
                0,
                &detect_schema(),
            )
            .unwrap_err();
        match err {
            DetectError::PlanMismatch { planned, got } => {
                assert_eq!((planned as usize, got), (vox.len(), vox.len() - 1));
            }
            other => panic!("{other:?}"),
        }
    }

    /// The size classes, at their edges: under a metre in both horizontal
    /// axes and under 0.8 m tall is a fragment, over 6 m on any axis is
    /// merged, and a car is neither.
    #[test]
    fn a_box_s_shape_is_its_extent_against_the_three_thresholds() {
        let o = [0.0, 0.0, 0.0];
        assert_eq!(shape_of(o, [0.9, 0.9, 0.7]), Shape::Fragment);
        // A pedestrian: small footprint, but tall -- not a fragment.
        assert_eq!(shape_of(o, [0.6, 0.6, 1.6]), Shape::Object);
        // One horizontal axis at a metre is enough to leave the class.
        assert_eq!(shape_of(o, [1.0, 0.5, 0.5]), Shape::Object);
        assert_eq!(shape_of(o, [4.5, 1.8, 1.4]), Shape::Object);
        assert_eq!(shape_of(o, [6.0, 1.8, 1.4]), Shape::Object);
        assert_eq!(shape_of(o, [6.1, 1.8, 1.4]), Shape::Merged);
        assert_eq!(shape_of(o, [1.0, 12.0, 1.0]), Shape::Merged);
        assert_eq!(
            [Shape::Fragment, Shape::Object, Shape::Merged].map(Shape::name),
            ["fragment", "object", "merged"]
        );
    }

    /// An empty cloud reports no detections and no ratio rather than a
    /// division by nothing. `reduce` shipped the opposite once --
    /// `11883.34x smaller` over a result with nothing in it.
    #[test]
    fn an_empty_cloud_reports_no_ratio_rather_than_a_large_one() {
        let (batch, plan) = run(&[]);
        assert_eq!((plan.detections, plan.clusters), (0, 0));
        assert_eq!(plan.ratio(), None);
        assert_eq!(plan.fragment_fraction(), None);
        assert_eq!(detection_count(&batch), Some(0));
        assert!(detections_f32(&batch).unwrap().is_empty());
        assert!(!plan.ground.fitted, "a plane was fitted to nothing");
    }

    /// Everything further away than the spacing table supports is left out.
    ///
    /// The far blob sits at a FIXED 50 m rather than at `RANGE_LIMIT_M + k`,
    /// so widening the bound really does pull it in. The first version placed
    /// it relative to the constant and therefore moved with it -- and at
    /// `1e9` the coordinates stopped being representable in f32 at all, so
    /// raising the bound changed nothing and the test passed against the bug.
    #[test]
    fn nothing_past_the_validity_bound_is_reported() {
        const FAR_M: i32 = 250; // 250 cells x 0.20 m = 50 m
        assert!(
            RANGE_LIMIT_M < FAR_M as f32 * V,
            "the far blob is inside the bound, so this test proves nothing"
        );
        let near = {
            let mut p = ground_points(0.0, 40, 30, 0.15);
            for ix in 0..5 {
                p.push(cell_point(ix, 0, -4, 0.7));
            }
            voxelize(&p)
        };
        let mut with_far = near.clone();
        for ix in 0..5 {
            with_far.push(cell_point(FAR_M + ix, 0, -4, 0.7));
        }

        let a = run(&near).1;
        let b = run(&with_far).1;
        assert_eq!(
            b.source_voxels,
            a.source_voxels + 5,
            "the far blob never reached the stage, so this proves nothing"
        );
        assert_eq!(
            (b.in_range_voxels, b.clusters, b.detections),
            (a.in_range_voxels, a.clusters, a.detections),
            "voxels at 50 m reached the clusterer, {} m past the bound",
            FAR_M as f32 * V - RANGE_LIMIT_M
        );
        assert_eq!((a.clusters, a.detections), (1, 1), "{a:?}");
    }

    /// The parameter note must not travel with a size it does not justify --
    /// the same failure `voxel.rs` shipped and then pinned, one stage on.
    #[test]
    fn the_parameter_note_admits_when_the_range_bound_was_not_measured() {
        let default_note = detect_params_note(VoxelSize::default());
        let other_note = detect_params_note(VoxelSize::new(0.5).unwrap());
        for note in [&default_note, &other_note] {
            assert!(note.contains("0.6 m pedestrian"), "{note}");
            assert!(note.contains("30 m"), "{note}");
        }
        assert!(
            !default_note.contains("NOTE:"),
            "the default carries a caveat that is only true off it: {default_note}"
        );
        assert!(
            other_note.contains("measured at 0.20 m only"),
            "a non-default grid is told the range bound applies to it: {other_note}"
        );
    }
}
