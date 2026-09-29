//! The voxel-grid reduction: the project's first stage that **consumes an
//! Arrow payload and produces a smaller Arrow payload**.
//!
//! Everything before this was fan-out — one driver, two leaf consumers — and
//! the architecture document's step 2 ("pipeline stages ... create another
//! Arrow result when they actually transform the data") had no implementation
//! at all. This module is that result; [`crate::velo`]'s sweep is its input
//! and its output is the same shape, so the stage after it reads both with one
//! accessor.
//!
//! # Why a voxel grid, and not something with a better number
//!
//! Four candidates were considered. Three were rejected for reasons that are
//! about honesty rather than about the ratio:
//!
//! * **Random subsample.** Its shrink ratio is a free parameter: you dial it to
//!   whatever you wanted to report, so the number says nothing about the data.
//! * **Range / field-of-view crop.** The single biggest reduction available,
//!   and defensible for a fusion demo — but every output point is byte
//!   identical to an input point, so it *filters*, it does not "actually
//!   transform the data".
//! * **Ground-plane removal.** Real, but the correct version (RANSAC) is
//!   randomised, and the cheap fixed-`z` version is a crop wearing a hat.
//!
//! A voxel grid is the only one of the four where an output value is
//! *computed from several input values*, and — the part that matters — where
//! **the ratio is not a free parameter**. Name a voxel edge in metres and the
//! scene decides the ratio. A different scene gives a different number, which
//! is the sign that the number measures something.
//!
//! # Where 0.20 m comes from
//!
//! [`DEFAULT_VOXEL_SIZE_M`] is derived from the smallest object that has to
//! survive the reduction, not from the ratio it produces. The smallest KITTI
//! object class is a pedestrian, about 0.6 m across. One third of that is
//! 0.20 m, which puts at least three voxels across a pedestrian's width and
//! eight up its height, so the smallest thing that matters is still a
//! multi-voxel structure afterwards rather than a single cell.
//!
//! Report the ratio that falls out of that, whatever it is. Choosing 0.5 m
//! instead would roughly triple the headline number and would be exactly the
//! parameter-tuning-to-produce-a-result this project's standards forbid.
//!
//! # Two traps that were measured, not guessed
//!
//! * **`as i32` truncates toward zero; the index needs `floor`.** KITTI's x, y
//!   and z all span negative values, and truncation makes the voxel straddling
//!   each axis origin twice as wide. On drive_0005 frame 0 at 0.20 m, 119,421
//!   of 123,397 points get a different index from `(x / v) as i32` than from
//!   `(x / v).floor()`, and the occupied-voxel count differs (34,621 against
//!   33,788). See `a_voxel_boundary_is_not_at_the_origin`.
//!
//!   Those three figures are the arithmetic [`voxel_key`] actually performs —
//!   `f64::from(c) / v`. An earlier draft of this note quoted 119,317 /
//!   34,616 / 33,795, which is the same experiment run entirely in f32, to
//!   the unit; the shipped grouping is f64 and gives the numbers above. The
//!   argument is unchanged — about 119 k of 123 k points move either way —
//!   but a doc comment about this code should state this code's numbers.
//! * **f32 centroid summation is order dependent.** On the same frame, grouped
//!   the way this module groups, 10,968 of its 19,309 multi-point voxels give
//!   a different f32 sum in at least one of the four components when the
//!   points are accumulated in reverse (5,177 of them differ in x alone). The
//!   earlier 9,992 of 19,313 was again the f32-grouped version of the count.
//!   This module pins the order *and* accumulates in f64 — belt and
//!   braces, because the f64 argument is a practical one rather than a proof:
//!   the fullest voxel measured over all 154 sweeps of drive_0005 holds 395
//!   points, so f64 accumulation error lands about seven orders of magnitude
//!   below f32 rounding. (The design note for this work said 167; that was
//!   frame 0 alone. Over the whole drive it is 395, which does not change the
//!   argument but is the number this code should state.)
//!
//! # Determinism, by construction rather than by convention
//!
//! The grouping is a sort, not a map — `clippy.toml` bans `HashMap`/`HashSet`
//! for nondeterministic iteration, and a sort sidesteps the question by using
//! no map at all. The sort key is `(voxel key, input index)`, so **no two
//! entries ever compare equal** and the sorted order is a total order. That
//! makes `sort_unstable` (which allocates nothing, unlike the stable sort)
//! safe here: there are no ties for an unstable sort to reorder. Output rows
//! come out in ascending voxel-key order, which is lexicographic in
//! `(ix, iy, iz)` and independent of input order.
//!
//! **One honest caveat about that tie-break: no test in this workspace can
//! tell whether it is there.** Replacing `sort_unstable()` with
//! `sort_unstable_by_key(|e| e.0)` — which leaves the within-voxel order
//! unspecified — was tried, and every test still passed. The reason is the
//! f64 accumulation below: it makes the centroid insensitive to the order of
//! the points inside a voxel, so the one thing the tie-break protects is
//! already protected by something else. It is kept because it costs nothing
//! and because determinism should not rest on a floating-point argument alone,
//! but it is defence in depth rather than a property the tests demonstrate,
//! and claiming otherwise would be claiming coverage that does not exist.

use std::sync::Arc;

use arrow::array::{ArrayRef, Float32Array, Int64Array, RecordBatch, StringArray, UInt32Array};
use arrow::buffer::{Buffer, MutableBuffer};
use arrow::datatypes::{DataType, Field, Schema};

use crate::velo::{build_points_array, points_field, SweepError, POINT_BYTES};

/// Voxel edge in metres, derived from the smallest object that must survive:
/// one third of a 0.6 m pedestrian width. See the module docs — this number
/// comes from the scene, not from the ratio it happens to produce.
pub const DEFAULT_VOXEL_SIZE_M: f32 = 0.20;

/// Where [`DEFAULT_VOXEL_SIZE_M`] comes from, in one line, so a run's own
/// output can say it and a reader never has to take the number on trust.
///
/// It names the number it justifies. The first version did not — it read
/// "1/3 of a 0.6 m pedestrian width, …" and was printed beside whatever
/// `--voxel-size-m` was given, so `--voxel-size-m 0.5` produced the line
/// `reduce voxel = 0.5 m (1/3 of a 0.6 m pedestrian width …)`, which is
/// arithmetically false. A justification that travels with the wrong number
/// is worse than none, because it reads as a derivation.
pub const VOXEL_SIZE_RATIONALE: &str =
    "the default 0.20 m is 1/3 of a 0.6 m pedestrian width, the smallest KITTI \
     object class that must survive as a multi-voxel structure";

/// The parenthetical a run prints after its voxel edge: the rationale, and
/// whether this run is using the size that rationale is about.
///
/// See [`VOXEL_SIZE_RATIONALE`] for why this is a function of the size rather
/// than a constant printed beside it.
pub fn voxel_size_note(v: VoxelSize) -> String {
    if v.is_default() {
        VOXEL_SIZE_RATIONALE.to_string()
    } else {
        format!("set with --voxel-size-m; {VOXEL_SIZE_RATIONALE}")
    }
}

/// Value of the `point_format` column for a voxel-reduced cloud.
///
/// Deliberately **not** [`crate::velo::POINT_FORMAT_XYZR_F32LE`], although the
/// bytes are laid out identically. The fourth f32 is now a *mean* reflectance
/// over the points of a voxel, not a reflectance the sensor measured, and a
/// derived value that still calls itself a measurement is the exact shape of
/// the plausible-looking wrongness this project exists to refuse.
pub const POINT_FORMAT_XYZR_F32LE_VOXEL: &str = "xyzr_f32le_voxel";

/// Bits per packed axis index, and therefore the addressable grid: signed
/// `[-2^20, 2^20)` per axis, i.e. +-209 km at 0.20 m. Three of these fit in an
/// `i64` exactly (3 x 21 = 63), so the packed key never touches the sign bit.
const INDEX_BITS: u32 = 21;

/// Added to each signed axis index so the packed key is non-negative and the
/// key order is lexicographic in `(ix, iy, iz)`.
const INDEX_BIAS: i64 = 1 << (INDEX_BITS - 1);

/// A validated voxel edge: finite and strictly positive.
///
/// A newtype rather than a bare `f32` because every one of the invalid values
/// fails *silently* rather than loudly. `0.0` makes every division infinite and
/// every point land out of range; a negative edge flips `floor` so the grid
/// runs backwards; `NaN` makes every comparison false, so every point is
/// discarded and the stage reports an empty cloud with no error anywhere. The
/// CLI parses into this once, at the edge of the program.
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub struct VoxelSize(f32);

/// Why a voxel edge was refused.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VoxelSizeError {
    /// The value that was refused.
    pub got: f32,
}

impl std::fmt::Display for VoxelSizeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "voxel size must be a finite positive number of metres; got {}",
            self.got
        )
    }
}

impl std::error::Error for VoxelSizeError {}

impl VoxelSize {
    /// Checks `m` is finite and `> 0`.
    pub fn new(m: f32) -> Result<VoxelSize, VoxelSizeError> {
        if !m.is_finite() || m <= 0.0 {
            return Err(VoxelSizeError { got: m });
        }
        Ok(VoxelSize(m))
    }

    /// The edge in metres.
    pub fn metres(self) -> f32 {
        self.0
    }

    /// Whether this is [`DEFAULT_VOXEL_SIZE_M`].
    ///
    /// Bit equality, not a tolerance: the question is "was the default used",
    /// which is exact, and not "are two measurements close", which is what a
    /// tolerance would answer. `"0.2"` parses to the bits of the literal, so
    /// `--voxel-size-m 0.2` is the default and says so.
    pub fn is_default(self) -> bool {
        self.0.to_bits() == DEFAULT_VOXEL_SIZE_M.to_bits()
    }
}

impl Default for VoxelSize {
    fn default() -> Self {
        VoxelSize(DEFAULT_VOXEL_SIZE_M)
    }
}

/// Round-trips through [`VoxelSize::new`], which is what lets the CLI print
/// its own default and parse it back to the same value.
impl std::fmt::Display for VoxelSize {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// `point_count u32, point_format Utf8, tov_trigger_ns i64, voxel_size_m f32,
/// source_point_count u32, points LargeList<FixedSizeList<Float32, 4>>`.
///
/// The input's schema plus two columns, and the `points` column is byte-for-byte
/// the same shape, which is the whole design:
///
/// * [`crate::velo::velo_xyzr`], [`crate::velo::velo_point_count`] and
///   [`crate::velo::velo_trigger_ns`] read a reduced cloud **unchanged**. A
///   downstream stage needs one accessor for the sweep and for the result,
///   which is "consume and produce the same combination" made mechanical
///   rather than aspirational.
/// * One contiguous f32 buffer means one `storage_id`, so the address-equality
///   evidence that proves the sweep was shared extends to the derived stream
///   with no new machinery.
/// * One row per cloud keeps the per-sample metadata columns — above all
///   `tov_trigger_ns`, which is what a sync stage needs to match this cloud's
///   interval to a camera instant.
///
/// The two added columns are there so a recorded batch is interpretable on its
/// own: `voxel_size_m` carries the parameter with the data, and
/// `source_point_count` makes the reduction ratio recomputable from the batch
/// alone rather than from a log line somebody has to still have.
pub fn voxel_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("point_count", DataType::UInt32, false),
        Field::new("point_format", DataType::Utf8, false),
        Field::new("tov_trigger_ns", DataType::Int64, false),
        Field::new("voxel_size_m", DataType::Float32, false),
        Field::new("source_point_count", DataType::UInt32, false),
        Field::new("points", DataType::LargeList(points_field()), false),
    ]))
}

/// The voxel edge a reduced cloud was built with, read back out of the batch.
pub fn voxel_size_m(batch: &RecordBatch) -> Option<f32> {
    let c = batch
        .column_by_name("voxel_size_m")?
        .as_any()
        .downcast_ref::<Float32Array>()?;
    (!c.is_empty()).then(|| c.value(0))
}

/// Points the cloud was reduced FROM, so the ratio is recomputable from the
/// batch without the run that produced it.
pub fn source_point_count(batch: &RecordBatch) -> Option<u32> {
    let c = batch
        .column_by_name("source_point_count")?
        .as_any()
        .downcast_ref::<UInt32Array>()?;
    (!c.is_empty()).then(|| c.value(0))
}

/// Errors from reducing one cloud.
#[derive(Debug)]
pub enum VoxelError {
    /// The derived buffer could not be wrapped as Arrow.
    Sweep(SweepError),
    /// [`Voxelizer::build`] was handed different points than
    /// [`Voxelizer::plan`] indexed. The scratch holds indices into the planned
    /// slice, so this would read the wrong points (or panic); it is a typed
    /// error rather than either.
    PlanMismatch {
        /// Points the plan was computed over.
        planned: u32,
        /// Points `build` was given.
        got: usize,
    },
}

impl std::fmt::Display for VoxelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VoxelError::Sweep(_) => write!(f, "the reduced cloud could not be wrapped as Arrow"),
            VoxelError::PlanMismatch { planned, got } => write!(
                f,
                "the plan indexed {planned} points but build was given {got}"
            ),
        }
    }
}

impl std::error::Error for VoxelError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            VoxelError::Sweep(e) => Some(e),
            VoxelError::PlanMismatch { .. } => None,
        }
    }
}

impl From<SweepError> for VoxelError {
    fn from(e: SweepError) -> Self {
        VoxelError::Sweep(e)
    }
}

/// What one reduction did, counted while it was doing it.
///
/// Every field here is reported rather than asserted away. `singleton_voxels`
/// in particular is the honesty caveat that belongs beside the shrink ratio: a
/// voxel holding one point has a "centroid" that averages nothing and is a
/// copy of that point, and on real data that is a large minority of the output
/// rows.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct VoxelPlan {
    /// Points in the input cloud.
    pub source_points: u32,
    /// Points that got a grid index, i.e. `source_points` less the discards.
    pub indexed_points: u32,
    /// Occupied voxels, and so rows in the output cloud.
    pub out_points: u32,
    /// Points discarded because a coordinate or the reflectance was not finite.
    ///
    /// Counted, never silently folded in: `f32::NAN.floor() as i32` is `0` in
    /// Rust, so the alternative is every NaN point landing in voxel (0, 0, 0).
    pub non_finite: u32,
    /// Points discarded because a grid index fell outside the packed key's
    /// +-209 km range (at 0.20 m). Impossible for a lidar; a typed count
    /// rather than a wrap-around.
    pub out_of_range: u32,
    /// Output voxels holding exactly one point.
    pub singleton_voxels: u32,
    /// Points in the fullest voxel.
    pub max_occupancy: u32,
}

impl VoxelPlan {
    /// Input points per output point, or `None` when nothing survived.
    pub fn ratio(&self) -> Option<f64> {
        (self.out_points > 0).then(|| f64::from(self.source_points) / f64::from(self.out_points))
    }
}

/// Packed grid key of one point, or `None` if it cannot have one.
///
/// `floor`, not `as i32`. See the module docs: the two disagree on 96.7 % of a
/// real KITTI sweep, because truncation makes the voxel straddling each axis
/// origin twice as wide and KITTI's x, y and z all span negative values.
///
/// The division is done in f64. The inputs are f32 so the widening is exact,
/// and f64 keeps the quotient from landing on the wrong side of an integer
/// boundary through rounding alone.
fn voxel_key(p: &[f32; 4], v: f64) -> Option<i64> {
    let axis = |c: f32| -> Option<i64> {
        let q = (f64::from(c) / v).floor();
        // Written as a positive range test so a NaN quotient — which cannot
        // reach here, but could if the finiteness check above ever moved —
        // fails it rather than passing it.
        (q >= -(INDEX_BIAS as f64) && q < INDEX_BIAS as f64).then(|| q as i64 + INDEX_BIAS)
    };
    let ix = axis(p[0])?;
    let iy = axis(p[1])?;
    let iz = axis(p[2])?;
    Some((ix << (2 * INDEX_BITS)) | (iy << INDEX_BITS) | iz)
}

/// Reusable scratch for the reduction, so the only per-cloud allocation a
/// stage makes is the output buffer itself.
///
/// This is the load-bearing design constraint, not a micro-optimisation. The
/// house rule (see `pipes::consumers::proc_thread`) is that per-sample scratch
/// is allocated **once, outside the measured window**, or the byte figure
/// stops meaning "the stage's output". A `BTreeMap<key, accumulator>` of ~30k
/// entries allocates per cloud and cannot be pre-sized or reused, which would
/// have made the measured number "output plus map" while the column still said
/// output.
#[derive(Debug, Default)]
pub struct Voxelizer {
    /// `(packed voxel key, index into the input points)`, sorted.
    entries: Vec<(i64, u32)>,
}

impl Voxelizer {
    /// An empty scratch. [`Voxelizer::reserve`] sizes it.
    pub fn new() -> Voxelizer {
        Voxelizer {
            entries: Vec::new(),
        }
    }

    /// Makes room for `n` points, emptying the scratch first.
    ///
    /// **Call this outside the measured window.** Returns whether it had to
    /// allocate, so a caller can announce the growth and a test can assert
    /// that the steady state does not grow at all.
    pub fn reserve(&mut self, n: usize) -> bool {
        self.entries.clear();
        if self.entries.capacity() >= n {
            return false;
        }
        self.entries.reserve_exact(n);
        true
    }

    /// Bytes the scratch currently holds.
    pub fn scratch_bytes(&self) -> usize {
        self.entries.capacity() * std::mem::size_of::<(i64, u32)>()
    }

    /// Phase 1: index every point and sort. **Borrows `points` and allocates
    /// nothing** once [`Voxelizer::reserve`] has been called for this size —
    /// `sort_unstable` sorts in place, where the stable sort would allocate a
    /// scratch buffer of its own and put an allocation inside the window that
    /// is supposed to contain only the output.
    ///
    /// Sorting by `(key, index)` makes the order total, so nothing depends on
    /// which sort is used, and it puts each voxel's points in **input order**,
    /// which is what pins the accumulation order in [`Voxelizer::build`].
    pub fn plan(&mut self, points: &[[f32; 4]], voxel_size: VoxelSize) -> VoxelPlan {
        self.entries.clear();
        let v = f64::from(voxel_size.metres());
        let mut non_finite = 0u32;
        let mut out_of_range = 0u32;
        for (i, p) in points.iter().enumerate() {
            // The reflectance is checked with the coordinates: a NaN there
            // would poison the mean of an otherwise sound voxel.
            if !p.iter().all(|c| c.is_finite()) {
                non_finite += 1;
                continue;
            }
            match voxel_key(p, v) {
                Some(key) => self.entries.push((key, i as u32)),
                None => out_of_range += 1,
            }
        }
        self.entries.sort_unstable();

        let mut out_points = 0u32;
        let mut singleton_voxels = 0u32;
        let mut max_occupancy = 0u32;
        for run in self.runs() {
            let occupancy = run.len() as u32;
            out_points += 1;
            singleton_voxels += u32::from(occupancy == 1);
            max_occupancy = max_occupancy.max(occupancy);
        }
        VoxelPlan {
            source_points: clamp_u32(points.len()),
            indexed_points: clamp_u32(self.entries.len()),
            out_points,
            non_finite,
            out_of_range,
            singleton_voxels,
            max_occupancy,
        }
    }

    /// The sorted entries grouped into runs of one voxel key each.
    fn runs(&self) -> impl Iterator<Item = &[(i64, u32)]> {
        self.entries.chunk_by(|a, b| a.0 == b.0)
    }

    /// Phase 2: allocate the output cloud and fill it.
    ///
    /// This is where the stage allocates, and it **should**: producing a new
    /// Arrow result is the correct behaviour for a stage that transforms data,
    /// and it is not a violation of zero copy. What matters is that the input
    /// was never copied — phase 1 read it in place — and that this allocation
    /// is exactly one buffer of exactly the output's size, so the measured
    /// number is the result and nothing else.
    ///
    /// Centroids are accumulated in f64 in input order and rounded to f32
    /// once, at the end. Both halves matter; see the module docs.
    pub fn build(
        &self,
        points: &[[f32; 4]],
        plan: &VoxelPlan,
        voxel_size: VoxelSize,
        trigger_ns: i64,
        schema: &Arc<Schema>,
    ) -> Result<(RecordBatch, usize), VoxelError> {
        if plan.source_points as usize != points.len() {
            return Err(VoxelError::PlanMismatch {
                planned: plan.source_points,
                got: points.len(),
            });
        }
        // The one allocation, and exactly the output's size: `from_len_zeroed`
        // takes a layout of precisely `len` (unlike `with_capacity`, which
        // rounds up to 64 B), so the bytes this stage requests and the bytes
        // the batch carries are the same number. Arrow's ALIGNMENT makes the
        // 4-byte alignment `Float32Array` needs structural rather than lucky,
        // which is the same reason `velo::read_sweep` uses it.
        let mut buf = MutableBuffer::from_len_zeroed(plan.out_points as usize * POINT_BYTES);
        // Checked before `typed_data_mut`, whose own alignment gate is an
        // `assert!`. It cannot fire on a `from_len_zeroed` buffer; this makes
        // that structural instead of a comment.
        let addr = buf.as_slice().as_ptr() as usize;
        if !addr.is_multiple_of(std::mem::align_of::<f32>()) {
            return Err(VoxelError::Sweep(SweepError::Misaligned { addr }));
        }
        {
            let out = buf.typed_data_mut::<f32>();
            for (w, run) in self.runs().enumerate() {
                let mut sum = [0f64; 4];
                // Input order, pinned by the `(key, index)` sort. Reversing
                // this changes the f32 result on over half the multi-point
                // voxels of a real sweep — measured, see the module docs.
                for &(_, i) in run {
                    let p = &points[i as usize];
                    for (s, c) in sum.iter_mut().zip(p) {
                        *s += f64::from(*c);
                    }
                }
                let n = run.len() as f64;
                for (k, s) in sum.iter().enumerate() {
                    out[w * 4 + k] = (s / n) as f32;
                }
            }
        }
        let (points_arr, storage_id, n_points) = build_points_array(Buffer::from(buf))?;
        let columns: Vec<ArrayRef> = vec![
            Arc::new(UInt32Array::from(vec![clamp_u32(n_points)])),
            Arc::new(StringArray::from(vec![POINT_FORMAT_XYZR_F32LE_VOXEL])),
            Arc::new(Int64Array::from(vec![trigger_ns])),
            Arc::new(Float32Array::from(vec![voxel_size.metres()])),
            Arc::new(UInt32Array::from(vec![plan.source_points])),
            points_arr,
        ];
        Ok((
            RecordBatch::try_new(Arc::clone(schema), columns).map_err(SweepError::Arrow)?,
            storage_id,
        ))
    }
}

/// `usize` as `u32`, saturating. Only ever reached with point counts, which
/// are six digits on this dataset; saturating beats a wrapping cast because a
/// count that read as small would be believed.
fn clamp_u32(n: usize) -> u32 {
    u32::try_from(n).unwrap_or(u32::MAX)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use pipes_core::sample::payload_bytes;

    use super::*;
    use crate::velo::{
        build_velo_batch, velo_point_count, velo_point_format, velo_schema, velo_trigger_ns,
        velo_xyzr, POINT_FORMAT_XYZR_F32LE,
    };

    const V: f32 = DEFAULT_VOXEL_SIZE_M;

    fn size() -> VoxelSize {
        VoxelSize::new(V).unwrap()
    }

    /// A cloud straight out of the driver's own builder, so these tests run
    /// against the real input shape rather than a hand-made lookalike.
    fn sweep_batch(points: &[[f32; 4]], trigger_ns: i64) -> RecordBatch {
        let mut buf = MutableBuffer::from_len_zeroed(points.len() * POINT_BYTES);
        buf.typed_data_mut::<f32>()
            .copy_from_slice(&points.concat());
        build_velo_batch(Buffer::from(buf), trigger_ns, &velo_schema())
            .unwrap()
            .0
    }

    fn reduce(points: &[[f32; 4]]) -> (RecordBatch, VoxelPlan) {
        reduce_at(points, size(), 7)
    }

    fn reduce_at(points: &[[f32; 4]], v: VoxelSize, trigger_ns: i64) -> (RecordBatch, VoxelPlan) {
        let mut vx = Voxelizer::new();
        vx.reserve(points.len());
        let plan = vx.plan(points, v);
        let (batch, _) = vx
            .build(points, &plan, v, trigger_ns, &voxel_schema())
            .unwrap();
        (batch, plan)
    }

    fn out_points(batch: &RecordBatch) -> Vec<[f32; 4]> {
        let (points, rest) = velo_xyzr(batch).unwrap().as_chunks::<4>();
        assert!(rest.is_empty(), "a partial point came out of the builder");
        points.to_vec()
    }

    /// The Arrow rule, mechanically: the accessors written for the DRIVER's
    /// sweep read the derived cloud unchanged. If this ever fails, a
    /// downstream stage needs two code paths and "consume and produce the same
    /// combination" has stopped being true.
    #[test]
    fn a_reduced_cloud_is_read_by_the_sweep_accessors() {
        let pts = [[0.0, 0.0, 0.0, 1.0], [5.0, -5.0, 1.0, 0.5]];
        let sweep = sweep_batch(&pts, 1234);
        let (reduced, plan) = reduce_at(&pts, size(), 1234);

        // Same accessor, both batches, no branch on which one it is.
        for b in [&sweep, &reduced] {
            assert_eq!(velo_xyzr(b).unwrap().len(), 8, "8 f32 = 2 points");
            assert_eq!(velo_point_count(b), Some(2));
            assert_eq!(velo_trigger_ns(b), Some(1234));
        }
        assert_eq!(plan.out_points, 2, "two points 5 m apart are two voxels");

        // And the ONE thing that must differ is the declaration of what the
        // fourth f32 means. Same bytes, different claim about them.
        assert_eq!(velo_point_format(&sweep), Some(POINT_FORMAT_XYZR_F32LE));
        assert_eq!(
            velo_point_format(&reduced),
            Some(POINT_FORMAT_XYZR_F32LE_VOXEL)
        );
        // The two columns that make a recorded batch self-describing.
        assert_eq!(voxel_size_m(&reduced), Some(V));
        assert_eq!(source_point_count(&reduced), Some(2));
        assert!(voxel_size_m(&sweep).is_none(), "a sweep has no voxel size");
    }

    /// The claim the whole stage exists for, in its smallest honest form:
    /// redundant points collapse and the payload gets smaller.
    #[test]
    fn a_redundant_cloud_shrinks_and_the_payload_shrinks_with_it() {
        // 100 points inside one 0.20 m voxel, plus one far away. Two voxels.
        let mut pts: Vec<[f32; 4]> = (0..100)
            .map(|i| [0.001 * i as f32, 0.0, 0.0, 1.0])
            .collect();
        pts.push([50.0, 0.0, 0.0, 0.25]);
        let sweep = sweep_batch(&pts, 0);
        let (reduced, plan) = reduce(&pts);

        assert_eq!((plan.source_points, plan.out_points), (101, 2));
        assert_eq!(plan.ratio(), Some(50.5));
        assert_eq!(plan.singleton_voxels, 1, "the far point is alone");
        assert_eq!(plan.max_occupancy, 100);
        // The number the byte chain is made of: what the NEXT edge carries.
        // Stated as the POINT bytes, exactly, plus the direction of the whole
        // payload. The totals also carry each batch's fixed header, and the
        // reduced one's is larger by two metadata columns — at 101 points that
        // header is most of the batch, which is precisely why the exact claim
        // is about the points and the loose one about the total.
        let (before, after) = (payload_bytes(&sweep), payload_bytes(&reduced));
        assert_eq!(velo_xyzr(&sweep).unwrap().len(), 101 * 4);
        assert_eq!(velo_xyzr(&reduced).unwrap().len(), 2 * 4);
        assert!(
            after < before,
            "reduced payload {after} B against {before} B: the transfer did not shrink"
        );
        // The output is the points and a fixed header, exactly as the input is.
        assert!(after >= 2 * POINT_BYTES);

        // At a realistic point count the header stops mattering and the
        // transfer really does collapse by the reduction's own ratio. Same
        // scene, 100x denser: still two voxels.
        let mut dense: Vec<[f32; 4]> = (0..100_000)
            .map(|i| [0.000_001 * i as f32, 0.0, 0.0, 1.0])
            .collect();
        dense.push([50.0, 0.0, 0.0, 0.25]);
        let dense_out = payload_bytes(&reduce(&dense).0);
        let dense_in = payload_bytes(&sweep_batch(&dense, 0));
        assert!(
            dense_out * 1000 < dense_in,
            "{dense_out} B out of {dense_in} B: 100k redundant points did not collapse"
        );
    }

    /// Trap A, and the sharpest one in the design. `(x / v) as i32` truncates
    /// toward zero, which fuses the two voxels either side of each axis origin
    /// into one twice as wide. KITTI's x, y and z all span negative values, so
    /// this is not a corner case: it is 96.7 % of the points of a real sweep.
    ///
    /// Reintroduce the bug by replacing `.floor()` with nothing in
    /// `voxel_key`'s `axis` and this test fails: the two points below collapse
    /// into one output row.
    #[test]
    fn a_voxel_boundary_is_not_at_the_origin() {
        // -0.1 / 0.2 = -0.5. floor -> -1; truncation -> 0, same cell as +0.1.
        let pts = [[-0.1, 0.0, 0.0, 1.0], [0.1, 0.0, 0.0, 1.0]];
        let (reduced, plan) = reduce(&pts);
        assert_eq!(
            plan.out_points, 2,
            "the points either side of x = 0 landed in one voxel, so the index \
             truncates instead of flooring"
        );
        // Positive control in the same test: two points genuinely inside one
        // cell DO collapse, so the assertion above is not just "nothing ever
        // merges".
        let together = [[0.01, 0.0, 0.0, 1.0], [0.11, 0.0, 0.0, 1.0]];
        assert_eq!(reduce(&together).1.out_points, 1);
        // And the surviving rows are the two inputs, unaveraged.
        let out = out_points(&reduced);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0][0], -0.1, "row order is ascending voxel key");
        assert_eq!(out[1][0], 0.1);
    }

    /// Byte-for-byte determinism, which is the requirement as stated: same
    /// input, same output, twice.
    #[test]
    fn the_same_cloud_reduces_to_the_same_bytes_twice() {
        let pts: Vec<[f32; 4]> = (0..2000)
            .map(|i| {
                let t = i as f32 * 0.37;
                [
                    t.sin() * 9.0,
                    t.cos() * 11.0,
                    (t * 0.3).sin() * 2.0,
                    t % 1.0,
                ]
            })
            .collect();
        let a = reduce(&pts).0;
        let b = reduce(&pts).0;
        assert_eq!(
            velo_xyzr(&a).unwrap(),
            velo_xyzr(&b).unwrap(),
            "two reductions of one cloud disagree"
        );
        // Not a trivial pass: the cloud really does reduce, so there are
        // multi-point voxels whose centroids could have differed.
        let plan = reduce(&pts).1;
        assert!(plan.out_points < plan.source_points);
        assert!(plan.max_occupancy > 1);
    }

    /// Trap B, one step further out: the same points in a different order.
    ///
    /// The voxel SET must be identical — that follows from the sort. The
    /// centroids are then equal because they are accumulated in f64 and
    /// rounded once, which is a practical argument rather than a proof (see
    /// the module docs), so it is asserted here against real arithmetic rather
    /// than assumed.
    #[test]
    fn reordering_the_input_does_not_move_the_output() {
        let pts: Vec<[f32; 4]> = (0..1500)
            .map(|i| {
                let t = i as f32 * 0.11;
                [t.sin() * 3.0, t.cos() * 3.0, (t * 0.7).cos(), 0.5]
            })
            .collect();
        let mut shuffled = pts.clone();
        shuffled.reverse();
        let (a, pa) = reduce(&pts);
        let (b, pb) = reduce(&shuffled);
        assert_eq!(pa.out_points, pb.out_points);
        assert!(
            pa.max_occupancy > 2,
            "nothing was averaged, so nothing was tested"
        );
        assert_eq!(velo_xyzr(&a).unwrap(), velo_xyzr(&b).unwrap());
    }

    /// A centroid is the mean of its voxel, computed in closed form here so
    /// the assertion is checkable against arithmetic rather than against the
    /// implementation.
    #[test]
    fn a_centroid_is_the_mean_of_its_voxel() {
        // Four points in the cell [0.0, 0.2)^3, mean (0.05, 0.05, 0.05, 0.5).
        let pts = [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.1, 0.5],
            [0.1, 0.1, 0.0, 0.5],
            [0.1, 0.1, 0.1, 1.0],
        ];
        let (reduced, plan) = reduce(&pts);
        assert_eq!(plan.out_points, 1);
        let out = out_points(&reduced);
        assert_eq!(out[0], [0.05, 0.05, 0.05, 0.5]);
    }

    /// Non-finite input is discarded and counted. Silence here would put every
    /// NaN point in voxel (0, 0, 0): `f32::NAN.floor() as i32` is `0` in Rust,
    /// so the alternative is a phantom object at the sensor origin.
    #[test]
    fn non_finite_points_are_discarded_and_counted() {
        let pts = [
            [1.0, 1.0, 1.0, 1.0],
            [f32::NAN, 0.0, 0.0, 1.0],
            [0.0, f32::INFINITY, 0.0, 1.0],
            [0.0, 0.0, 0.0, f32::NAN],
        ];
        let (reduced, plan) = reduce(&pts);
        assert_eq!(plan.non_finite, 3);
        assert_eq!((plan.source_points, plan.indexed_points), (4, 1));
        assert_eq!(plan.out_points, 1);
        // And nothing landed at the origin, which is where a silent floor of
        // NaN would have put all three.
        assert_eq!(out_points(&reduced), vec![[1.0, 1.0, 1.0, 1.0]]);
    }

    /// A coordinate beyond the packed grid is discarded and counted, rather
    /// than wrapping into some other voxel.
    #[test]
    fn points_outside_the_grid_are_discarded_and_counted() {
        let pts = [[0.0, 0.0, 0.0, 1.0], [1e9, 0.0, 0.0, 1.0]];
        let (_, plan) = reduce(&pts);
        assert_eq!((plan.out_of_range, plan.out_points), (1, 1));
        // The control: the same magnitude with a voxel large enough to hold
        // it is in range, so the check is about the grid and not about 1e9.
        let big = VoxelSize::new(10_000.0).unwrap();
        assert_eq!(reduce_at(&pts, big, 0).1.out_of_range, 0);
    }

    /// An empty cloud reduces to an empty cloud, through the real builder.
    #[test]
    fn an_empty_cloud_is_not_a_special_case() {
        let (reduced, plan) = reduce(&[]);
        assert_eq!((plan.source_points, plan.out_points), (0, 0));
        assert_eq!(velo_point_count(&reduced), Some(0));
        assert!(velo_xyzr(&reduced).unwrap().is_empty());
    }

    /// The scratch is sized once and then stops growing, which is what makes
    /// the measured allocation in `build` the output and nothing else.
    #[test]
    fn the_scratch_stops_growing() {
        let mut vx = Voxelizer::new();
        assert!(vx.reserve(4096), "the first sizing must allocate");
        let bytes = vx.scratch_bytes();
        assert!(bytes >= 4096 * 12);
        for n in [1, 100, 4096] {
            assert!(!vx.reserve(n), "re-sizing to {n} allocated again");
        }
        assert_eq!(vx.scratch_bytes(), bytes);
        assert!(vx.reserve(8192), "a bigger cloud must be allowed to grow");
    }

    /// The plan indexes into the slice it was given, so building against a
    /// different one is a typed error rather than a wrong answer or a panic.
    #[test]
    fn building_against_the_wrong_points_is_an_error() {
        let pts = [[0.0, 0.0, 0.0, 1.0], [9.0, 9.0, 9.0, 1.0]];
        let mut vx = Voxelizer::new();
        vx.reserve(pts.len());
        let plan = vx.plan(&pts, size());
        let err = vx
            .build(&pts[..1], &plan, size(), 0, &voxel_schema())
            .expect_err("a short slice must be refused");
        assert!(matches!(
            err,
            VoxelError::PlanMismatch { planned: 2, got: 1 }
        ));
    }

    /// An invalid voxel edge is refused where it enters the program, because
    /// every invalid value fails silently rather than loudly further in.
    #[test]
    fn an_invalid_voxel_size_is_refused() {
        for bad in [0.0, -0.2, f32::NAN, f32::INFINITY] {
            assert!(VoxelSize::new(bad).is_err(), "{bad} was accepted");
        }
        assert_eq!(VoxelSize::new(0.05).unwrap().metres(), 0.05);
        assert_eq!(VoxelSize::default().metres(), DEFAULT_VOXEL_SIZE_M);
    }

    /// The rationale must not travel with a size it does not justify.
    ///
    /// The bug this pins was live: the constant read "1/3 of a 0.6 m
    /// pedestrian width" and the run printed it beside whatever
    /// `--voxel-size-m` said, so a 0.5 m run claimed 0.5 was a third of 0.6.
    /// Both halves are asserted — that the default's note still carries the
    /// derivation, and that a non-default note does not claim it for itself —
    /// because dropping the rationale entirely would also pass one of them.
    #[test]
    fn the_rationale_names_the_size_it_justifies() {
        assert!(VoxelSize::default().is_default());
        assert!(VoxelSize::new(DEFAULT_VOXEL_SIZE_M).unwrap().is_default());
        assert!(!VoxelSize::new(0.5).unwrap().is_default());

        let default_note = voxel_size_note(VoxelSize::default());
        let other_note = voxel_size_note(VoxelSize::new(0.5).unwrap());
        for note in [&default_note, &other_note] {
            assert!(
                note.contains("0.20 m is 1/3 of a 0.6 m pedestrian width"),
                "a run's note stopped saying where the default came from: {note}"
            );
        }
        assert!(
            default_note.starts_with("the default 0.20 m is"),
            "{default_note}"
        );
        assert!(
            other_note.starts_with("set with --voxel-size-m;"),
            "a non-default size is reported as though it were the default: {other_note}"
        );
    }

    /// The parameter is not a free knob to dial the ratio with, but it IS a
    /// parameter, and a coarser grid must reduce further on the same scene.
    /// Stated as a direction, not as a target number.
    #[test]
    fn a_coarser_grid_reduces_further() {
        let pts: Vec<[f32; 4]> = (0..1000)
            .map(|i| {
                let t = i as f32 * 0.05;
                [t.sin() * 4.0, t.cos() * 4.0, t * 0.01, 0.5]
            })
            .collect();
        let fine = reduce_at(&pts, VoxelSize::new(0.1).unwrap(), 0).1;
        let coarse = reduce_at(&pts, VoxelSize::new(0.8).unwrap(), 0).1;
        assert!(
            coarse.out_points < fine.out_points,
            "0.8 m gave {} voxels and 0.1 m gave {}",
            coarse.out_points,
            fine.out_points
        );
    }
}
