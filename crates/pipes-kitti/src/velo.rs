//! `VeloDriver`: the paced Velodyne HDL-64E replayer for KITTI
//! `velodyne_points`, and the Arrow sweep builder beside it.
//!
//! This is the project's second stream, and it is the case [`Tov::Range`] was
//! written for. A camera frame is an *instant*; a lidar sweep is an *interval*
//! — the sensor rotates for ~103 ms and every point in the file was measured at
//! a different angle, and therefore a different time, inside that window. KITTI
//! gives all three instants per sweep, in three separate files:
//!
//! | file | what it is |
//! |---|---|
//! | `timestamps_start.txt` | the sweep began |
//! | `timestamps.txt` | the **trigger**: the head faced forward, and the camera fired |
//! | `timestamps_end.txt` | the sweep completed |
//!
//! [`Tov::Range`] carries start and end. The trigger is a *third* record, in a
//! file of its own, and it means something else: where the manufacturer chose
//! to align the cameras, not the middle of the rotation -- though on
//! drive_0005 it is the midpoint it resembles, to within 0.5 ns on all 154
//! sweeps. Deriving it from the range would make that coincidence an
//! assumption, and collapsing the range to it would throw away the rotation.
//! It is the one value that says what the camera frame is contemporaneous
//! *with*, and it travels in the payload, as `tov_trigger_ns`,
//! because it is sensor data rather than a pipeline fact and because the stage
//! that will need it (a sync stage, matching sweeps to frames) is reading the
//! payload for the points anyway.
//!
//! **Deadline.** A rotating lidar cannot hand over a sweep before the rotation
//! finishes, so the driver paces to `tov.end()`, not `tov.start()`. Note that
//! [`ClockModel::due`] applied to a `Range` returns the deadline of its
//! *start* (`clock.rs`), which for drive_0005 is 51.6 ms too early. That is
//! correct for "when did this measurement begin"; it is the wrong wake-up for a
//! producer, so this driver computes its deadline from `end` explicitly.
//!
//! **Zero copy.** The bytes `read_sweep` pulls off disk are the bytes the
//! `Float32Array` points at: no memcpy, no de-interleave, no `unsafe` (this
//! crate is `#![forbid(unsafe_code)]`). See [`read_sweep`] for why the
//! destination is an arrow `MutableBuffer` rather than a `Vec<u8>`.
//!
//! **A sweep is its frame number, and the source can have gaps.** KITTI names
//! each sweep file by the frame it belongs to -- the index the camera, the
//! lidar and the GPS share -- and writes one line per frame to each of the
//! three timestamp files, line `i + 1` for frame `i`, leaving the line blank
//! in all three where the frame has no sweep ([`crate::layout`], shared with
//! the camera). `2011_09_26_drive_0009_sync` is the case on disk: its source
//! has no `0000000177.bin`-`0000000180.bin` (a real sensor fault, 413.9 ms
//! between sweep 176's end and sweep 181's start), and its three files have
//! 447 lines each, blank at lines 178-181. A sweep keeps its frame number as
//! its `seq` -- sweep 181 is 181, the number of the camera frame of the same
//! instant -- and every frame with no sweep is replayed at its own place in
//! the schedule as a [`DriverEvent::Missing`] whose reason is
//! [`ABSENT_IN_SOURCE`](crate::layout::ABSENT_IN_SOURCE) and whose time of
//! validity is [`Tov::None`]: the source never measured it, so there is no
//! instant to give it.

use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use arrow::array::{
    Array, ArrayRef, AsArray, FixedSizeListArray, Float32Array, Int64Array, LargeListArray,
    RecordBatch, StringArray, UInt32Array,
};
use arrow::buffer::{Buffer, MutableBuffer, OffsetBuffer, ScalarBuffer};
use arrow::datatypes::{DataType, Field, FieldRef, Float32Type, Schema};
use arrow::error::ArrowError;
use pipes_core::clock::{now, sleep_until, ClockModel, HostTime, SensorTime, Tov};
use pipes_core::sample::{Sample, StreamId};

use crate::driver::{median_period_ns, DriverEvent, RunTotals};
use crate::layout::{
    check_increasing, instants, replay_absent, FrameIndex, Gap, LayoutError, Slot,
};
use crate::timestamps::{parse_timestamp_lines, TimestampError};

/// The three timestamp files of a sweep, in the order a sweep has them.
const SWEEP_FILES: [&str; 3] = [
    "timestamps_start.txt",
    "timestamps.txt",
    "timestamps_end.txt",
];

// The `.bin` holds little-endian f32 and this module hands those bytes to
// arrow as native f32 without swapping them. That is only sound on a
// little-endian host, and a silently byte-swapped point cloud is exactly the
// kind of plausible-looking garbage this project exists to make impossible.
#[cfg(target_endian = "big")]
compile_error!(
    "KITTI velodyne .bin is little-endian f32; pipes-kitti::velo reinterprets \
     the file bytes in place and has no byte-swapping path"
);

/// Bytes per point: `x, y, z, reflectance`, four little-endian `f32`.
pub const POINT_BYTES: usize = 16;

/// Values per point, i.e. the `FixedSizeList` width.
pub const POINT_VALUES: i32 = 4;

/// Value of the `point_format` column for KITTI's interleaved `xyzr`.
pub const POINT_FORMAT_XYZR_F32LE: &str = "xyzr_f32le";

/// The one directory a drive's lidar stream lives in.
pub const VELODYNE_DIR: &str = "velodyne_points";

/// How many sibling directory names an error message lists before it stops.
const SIBLINGS_SHOWN: usize = 8;

/// Arrow field of one coordinate. Must match [`points_field`]'s child and the
/// schema's, or `RecordBatch::try_new` rejects the batch on data type.
pub(crate) fn point_value_field() -> FieldRef {
    Arc::new(Field::new("item", DataType::Float32, false))
}

/// Arrow field of one point: `FixedSizeList<Float32, 4>`.
///
/// `pub(crate)` because [`crate::voxel`] builds a *derived* cloud with exactly
/// this child type. That is not a convenience: a stage that consumed a sweep
/// and produced a differently-typed cloud would break the one property this
/// workspace's Arrow rule rests on — that [`velo_xyzr`] reads the input and
/// the output of a transform stage with the same accessor.
pub(crate) fn points_field() -> FieldRef {
    Arc::new(Field::new(
        "item",
        DataType::FixedSizeList(point_value_field(), POINT_VALUES),
        false,
    ))
}

/// `point_count u32, point_format Utf8, tov_trigger_ns i64,
/// points LargeList<FixedSizeList<Float32, 4>>`, all non-null. Call once per
/// run and pass the `Arc` around.
///
/// One row per **sweep**, not per point. A top-level `FixedSizeList` would
/// make `num_rows` the point count, and every accessor in this workspace that
/// reads `.value(0)` means "this sample" — so the sweep would stop being a
/// sample and the per-sweep metadata columns would have nowhere to live.
///
/// The nesting is what buys the structural check for free: KITTI's `.bin`
/// files do **not** all hold the same number of points (measured on
/// drive_0005: 154 files, 154 distinct sizes, 98,532 to 124,122 points), so
/// there is no `expected` length a builder could compare against the way
/// [`crate::frame::build_cam0_batch`] compares `width * height * 3`. The only
/// structural invariant is `len % 16 == 0`, and `FixedSizeListArray::try_new`
/// enforces exactly that, as a typed error, at the driver boundary.
pub fn velo_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("point_count", DataType::UInt32, false),
        Field::new("point_format", DataType::Utf8, false),
        Field::new("tov_trigger_ns", DataType::Int64, false),
        Field::new("points", DataType::LargeList(points_field()), false),
    ]))
}

/// Errors from wrapping one sweep's bytes as an Arrow batch.
#[derive(Debug)]
pub enum SweepError {
    /// The file length is not a whole number of 16-byte points.
    NotWholePoints {
        /// Bytes actually supplied.
        got: usize,
    },
    /// The buffer's address is not 4-byte aligned, so it cannot be viewed as
    /// `f32` without a copy.
    ///
    /// Arrow's own gate here is an `assert!` inside
    /// `ScalarBuffer::<f32>::from(Buffer)` — a panic, in a crate that denies
    /// `unwrap`. [`read_sweep`] allocates through `MutableBuffer`, which is
    /// aligned to 128 bytes by construction, so this cannot fire on the
    /// driver's own path; it exists so that a buffer arriving from anywhere
    /// else is a `Result` rather than a crash.
    Misaligned {
        /// The offending address.
        addr: usize,
    },
    /// Arrow rejected the array or the batch.
    Arrow(ArrowError),
}

impl std::fmt::Display for SweepError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SweepError::NotWholePoints { got } => write!(
                f,
                "{got} bytes is not a whole number of {POINT_BYTES}-byte points \
                 ({} left over); KITTI velodyne .bin is 4 x f32 per point and \
                 every file length is a multiple of {POINT_BYTES}",
                got % POINT_BYTES
            ),
            SweepError::Misaligned { addr } => write!(
                f,
                "point buffer at {addr:#x} is not 4-byte aligned, so it cannot \
                 be read as f32 without copying it"
            ),
            SweepError::Arrow(e) => write!(f, "arrow: {e}"),
        }
    }
}

impl std::error::Error for SweepError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            SweepError::NotWholePoints { .. } | SweepError::Misaligned { .. } => None,
            SweepError::Arrow(e) => Some(e),
        }
    }
}

impl From<ArrowError> for SweepError {
    fn from(e: ArrowError) -> Self {
        SweepError::Arrow(e)
    }
}

/// The one-row `points` column of a cloud batch, wrapping `points`
/// **without copying it**. Returns the column, the `storage_id` (address of
/// the point buffer) and the point count.
///
/// Shared by [`build_velo_batch`] and by [`crate::voxel`]'s derived-cloud
/// builder, and shared deliberately. A transform stage that hand-rolled its
/// own nesting would be free to get the child field, the list width or the
/// multiple-of-16 check subtly different, and the result would still *look*
/// like a point cloud — right up to the point where [`velo_xyzr`] returned
/// `None` on it and a consumer silently read zero points.
pub(crate) fn build_points_array(points: Buffer) -> Result<(ArrayRef, usize, usize), SweepError> {
    let len = points.len();
    if !len.is_multiple_of(POINT_BYTES) {
        return Err(SweepError::NotWholePoints { got: len });
    }
    let storage_id = points.as_ptr() as usize;
    // Checked before `ScalarBuffer::from`, whose own check is an `assert!`.
    // `ScalarBuffer::<f32>::from` also *floors* the length at `len / 4` and
    // silently discards 1-3 trailing bytes, which is why the multiple-of-16
    // check above happens on the byte length rather than on the f32 count.
    if !storage_id.is_multiple_of(std::mem::align_of::<f32>()) {
        return Err(SweepError::Misaligned { addr: storage_id });
    }
    let n_points = len / POINT_BYTES;
    let values = Float32Array::new(ScalarBuffer::<f32>::from(points), None);
    let point = FixedSizeListArray::try_new(
        point_value_field(),
        POINT_VALUES,
        Arc::new(values) as ArrayRef,
        None,
    )?;
    let offsets = OffsetBuffer::<i64>::new(ScalarBuffer::<i64>::from(vec![0i64, n_points as i64]));
    let points_arr = LargeListArray::try_new(points_field(), offsets, Arc::new(point), None)?;
    Ok((Arc::new(points_arr) as ArrayRef, storage_id, n_points))
}

/// Wraps one sweep's raw bytes as a 1-row batch **without copying them**.
/// Returns the batch and the `storage_id` (address of the point buffer).
///
/// `trigger_ns` is the sweep's `timestamps.txt` instant — the third
/// measurement the range cannot carry; see the module docs.
pub fn build_velo_batch(
    points: Buffer,
    trigger_ns: i64,
    schema: &Arc<Schema>,
) -> Result<(RecordBatch, usize), SweepError> {
    let (points_arr, storage_id, n_points) = build_points_array(points)?;
    let columns: Vec<ArrayRef> = vec![
        Arc::new(UInt32Array::from(vec![n_points as u32])),
        Arc::new(StringArray::from(vec![POINT_FORMAT_XYZR_F32LE])),
        Arc::new(Int64Array::from(vec![trigger_ns])),
        points_arr,
    ];
    Ok((
        RecordBatch::try_new(Arc::clone(schema), columns)?,
        storage_id,
    ))
}

/// The interleaved `x, y, z, reflectance` values of a velo batch, shared with
/// the file bytes the driver read. `chunks_exact(4)` gives one point each.
pub fn velo_xyzr(batch: &RecordBatch) -> Option<&[f32]> {
    Some(
        batch
            .column_by_name("points")?
            .as_list_opt::<i64>()?
            .values()
            .as_fixed_size_list_opt()?
            .values()
            .as_primitive_opt::<Float32Type>()?
            .values(),
    )
}

/// Re-derived storage id for the 3-stage proof: the address of the shared
/// point buffer, read back out of the finished batch.
pub fn velo_storage_id(batch: &RecordBatch) -> Option<usize> {
    velo_xyzr(batch).map(|v| v.as_ptr() as usize)
}

/// Points in a velo batch, from the metadata column rather than the payload
/// length, so a disagreement between the two is visible.
pub fn velo_point_count(batch: &RecordBatch) -> Option<u32> {
    let c = batch
        .column_by_name("point_count")?
        .as_primitive_opt::<arrow::datatypes::UInt32Type>()?;
    (!c.is_empty()).then(|| c.value(0))
}

/// What the four f32 of a point MEAN, read back out of the batch.
///
/// [`POINT_FORMAT_XYZR_F32LE`] for a sweep off the sensor;
/// [`crate::voxel::POINT_FORMAT_XYZR_F32LE_VOXEL`] for a cloud a transform
/// stage produced, whose fourth value is a *mean* reflectance rather than a
/// measured one. The layout is byte-identical either way, which is exactly why
/// the column has to exist: nothing downstream could tell the two apart from
/// the buffer, and silently averaging measured reflectances into a value that
/// still calls itself a measurement is the kind of plausible-looking wrongness
/// this project is built to refuse.
pub fn velo_point_format(batch: &RecordBatch) -> Option<&str> {
    let c = batch
        .column_by_name("point_format")?
        .as_any()
        .downcast_ref::<StringArray>()?;
    (!c.is_empty()).then(|| c.value(0))
}

/// The sweep's trigger instant (`timestamps.txt`) in sensor ns.
pub fn velo_trigger_ns(batch: &RecordBatch) -> Option<i64> {
    let c = batch
        .column_by_name("tov_trigger_ns")?
        .as_primitive_opt::<arrow::datatypes::Int64Type>()?;
    (!c.is_empty()).then(|| c.value(0))
}

/// Reads one `.bin` whole into an arrow buffer, with no copy afterwards.
///
/// The destination is a [`MutableBuffer`] rather than a `Vec<u8>` for one
/// reason, and it is not the cache-line argument usually given for it:
/// `Buffer::from_vec` inherits the `Vec`'s own `Layout`, which for `Vec<u8>`
/// has `align = 1`, and arrow does not realign. Viewing those bytes as `f32`
/// then depends on whatever the system allocator happened to return — true on
/// this host for all 154 files, guaranteed by nobody. `MutableBuffer` allocates
/// at arrow's `ALIGNMENT` (128 on x86_64), so the 4 bytes `Float32Array`
/// requires are structural rather than lucky.
///
/// It costs one `alloc_zeroed` of the file's length and no memcpy: the
/// kernel-to-userspace copy that `read_exact` pays is the same one
/// `std::fs::read` pays. The zeroing is then overwritten in full by
/// `read_exact`, which errors rather than returning short.
pub fn read_sweep(path: &Path) -> std::io::Result<Buffer> {
    let mut f = std::fs::File::open(path)?;
    let len = usize::try_from(f.metadata()?.len())
        .map_err(|_| std::io::Error::other(format!("{} is larger than usize", path.display())))?;
    let mut buf = MutableBuffer::from_len_zeroed(len);
    f.read_exact(buf.as_slice_mut())?;
    Ok(Buffer::from(buf))
}

/// The one filename the sweep of frame `i` can have: `{i:010}.bin`.
pub fn sweep_file_name(i: usize) -> String {
    format!("{i:010}.bin")
}

/// `<root>/<date>/<drive>/velodyne_points`.
pub fn velo_dir(root: &Path, date: &str, drive: &str) -> PathBuf {
    root.join(date).join(drive).join(VELODYNE_DIR)
}

/// Whether this drive carries a lidar stream at all.
///
/// The cheapest honest test: `velodyne_points/` either exists or it does not.
/// Many KITTI downloads are `image_02` only, so a camera-only drive is a
/// common case, not an exception, and it must stay usable.
pub fn has_velodyne(root: &Path, date: &str, drive: &str) -> bool {
    velo_dir(root, date, drive).is_dir()
}

/// Errors from opening a drive's lidar stream or replaying a sweep.
#[derive(Debug)]
pub enum VeloError {
    /// A path could not be read.
    Io {
        /// The path that failed.
        path: PathBuf,
        /// Underlying I/O error.
        source: std::io::Error,
    },
    /// The drive carries no `velodyne_points/` directory.
    ///
    /// Not a failure by itself: it is exactly what `--lidar auto` tests for.
    /// It becomes an error only under `--lidar on`, where the user said the
    /// stream must be there.
    NoVelodyne {
        /// The `velodyne_points` directory that was looked for.
        dir: PathBuf,
        /// The drive directory it sits under, when that exists.
        drive_dir: PathBuf,
        /// What *is* in the drive directory, sorted; empty when unreadable.
        siblings: Vec<String>,
    },
    /// One of the three timestamp files could not be parsed.
    Timestamps {
        /// Which file.
        path: PathBuf,
        /// Underlying error.
        source: TimestampError,
    },
    /// The three timestamp files have different numbers of lines. Each has
    /// one line per frame, blank lines included, so they must agree.
    TimestampCount {
        /// Lines in `timestamps_start.txt`.
        start: usize,
        /// Lines in `timestamps.txt`.
        trigger: usize,
        /// Lines in `timestamps_end.txt`.
        end: usize,
    },
    /// One timestamp file's line is blank where another's has a time: a
    /// frame the source has no sweep for is blank in all three.
    BlankLinesDiffer {
        /// 1-based line, the same in all three files.
        line: usize,
        /// A file whose line is blank.
        blank: &'static str,
        /// A file whose line has a time.
        filled: &'static str,
    },
    /// The sweep files and the timestamp lines are not KITTI's layout: a file
    /// not named by its frame, a line with a time and no file, a file whose
    /// line is blank or past the last, or times that do not increase. The
    /// camera's own checks, from the same code ([`LayoutError`]).
    Layout(LayoutError),
    /// A sweep's end is not after its start.
    ///
    /// A rotation takes about 103 ms; one that ends before it begins is not a
    /// sweep, and no camera instant could be paired inside it.
    EndBeforeStart {
        /// 1-based line number, in both `timestamps_start.txt` and
        /// `timestamps_end.txt`, as an editor numbers it: the frame plus one.
        line: usize,
        /// The frame that line belongs to.
        frame: u64,
    },
}

impl std::fmt::Display for VeloError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VeloError::Io { path, source } => write!(f, "{}: {source}", path.display()),
            VeloError::NoVelodyne {
                dir,
                drive_dir,
                siblings,
            } => {
                write!(
                    f,
                    "the drive {} carries no lidar stream: {} does not exist. ",
                    drive_dir.display(),
                    dir.display()
                )?;
                if siblings.is_empty() {
                    f.write_str("That drive directory holds no subdirectories at all. ")?;
                } else {
                    write!(
                        f,
                        "It holds: {}. ",
                        siblings[..siblings.len().min(SIBLINGS_SHOWN)].join(", ")
                    )?;
                }
                // Only `--lidar on` can reach this message: `auto`, which is
                // the default, replays such a drive as a camera-only one
                // without ever opening the directory. An earlier version of
                // this line called `off` the default and told the reader to
                // pass it, which is advice to set a flag they have not set.
                f.write_str(
                    "Run `cargo run --release -- drives` to see which drives have lidar, or drop \
                     `--lidar on`: the default `auto` replays this drive as a \
                     camera-only one",
                )
            }
            VeloError::Timestamps { path, source } => {
                write!(f, "{}: {source}", path.display())
            }
            VeloError::TimestampCount {
                start,
                trigger,
                end,
            } => write!(
                f,
                "velodyne_points' timestamps_start.txt ends at line {start}, \
                 timestamps.txt at line {trigger} and timestamps_end.txt at line \
                 {end}: KITTI writes one line per frame to each -- line N is \
                 frame N-1, blank where the frame has no sweep -- and a sweep \
                 needs all three of its instants, so the three must have the same \
                 number of lines, and which frame a line belongs to would \
                 otherwise be a guess. Re-extract velodyne_points from the \
                 drive's zip; `cargo run --release -- drives` checks every drive \
                 the same way"
            ),
            VeloError::BlankLinesDiffer {
                line,
                blank,
                filled,
            } => write!(
                f,
                "velodyne_points' {blank} line {line} is blank but {filled} line \
                 {line} has a time: KITTI blanks a frame's line in all three \
                 timestamp files when the frame has no sweep, and a sweep needs \
                 all three of its instants, so whether frame {} has a sweep would \
                 be a guess. Re-extract velodyne_points from the drive's zip; \
                 `cargo run --release -- drives` checks every drive the same way",
                line.saturating_sub(1)
            ),
            // The wrapped text is the whole message: it names the directory,
            // the file and what helps.
            VeloError::Layout(e) => write!(f, "{e}"),
            VeloError::EndBeforeStart { line, frame } => write!(
                f,
                "the sweep of frame {frame} (line {line} of timestamps_start.txt \
                 and timestamps_end.txt) ends before it starts: a sweep is a \
                 rotation of about 103 ms, so this line is not one, and no camera \
                 instant could be paired inside it. Re-extract velodyne_points \
                 from the drive's zip; `cargo run --release -- drives` checks \
                 every drive the same way"
            ),
        }
    }
}

impl std::error::Error for VeloError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            VeloError::Io { source, .. } => Some(source),
            VeloError::Timestamps { source, .. } => Some(source),
            // Not `Some(e)`: `Display` already prints the layout error whole,
            // and `main` would print it a second time as its cause.
            VeloError::Layout(_)
            | VeloError::NoVelodyne { .. }
            | VeloError::TimestampCount { .. }
            | VeloError::BlankLinesDiffer { .. }
            | VeloError::EndBeforeStart { .. } => None,
        }
    }
}

impl From<LayoutError> for VeloError {
    fn from(e: LayoutError) -> Self {
        VeloError::Layout(e)
    }
}

/// The velodyne replayer for one drive.
pub struct VeloDriver {
    /// KITTI data root holding `<date>/<drive>/velodyne_points`.
    pub root: PathBuf,
    /// Capture date directory, e.g. `2011_09_26`.
    pub date: String,
    /// Drive directory, e.g. `2011_09_26_drive_0005_sync`.
    pub drive: String,
    /// Which frames have a sweep: `index.frame(k)` is the frame of the `k`-th
    /// sweep, and its `seq`. Equal to `k` on a drive with a sweep for every
    /// frame; on drive 0009 it jumps from 176 to 181.
    pub index: FrameIndex,
    /// When each sweep began (`timestamps_start.txt`), by sweep index `k`:
    /// the `k`-th non-blank line, the line of frame `index.frame(k)`.
    pub start: Vec<SensorTime>,
    /// When the head faced forward and the cameras fired (`timestamps.txt`).
    /// A third record, read rather than derived from the range -- see the
    /// module docs.
    pub trigger: Vec<SensorTime>,
    /// When each sweep completed (`timestamps_end.txt`); the driver's deadline.
    pub end: Vec<SensorTime>,
}

impl VeloDriver {
    /// Opens `<root>/<date>/<drive>/velodyne_points` the way KITTI lays a
    /// synced drive out ([`crate::layout`]): every `.bin` in `data/` is named
    /// by the frame it belongs to, and each of the three timestamp files has
    /// one line per frame, line `i + 1` for frame `i`, blank in all three
    /// where the frame has no sweep. A frame whose lines are blank and whose
    /// file is absent is a gap in the source
    /// ([`VeloDriver::absent_in_source`]): recorded by the replay, not
    /// refused. On drive 0009 that is lines 178-181, frames 177-180: 443
    /// sweeps over 447 frames.
    ///
    /// Refused, each naming the real line as an editor shows it: the three
    /// files of different lengths ([`VeloError::TimestampCount`]); a line
    /// blank in one and not in another ([`VeloError::BlankLinesDiffer`]); any
    /// of the layout errors the camera is held to as well ([`LayoutError`],
    /// judged on `timestamps.txt`); a sweep that ends before it starts
    /// ([`VeloError::EndBeforeStart`]); and a start or end file whose times do
    /// not increase.
    pub fn open(root: &Path, date: &str, drive: &str) -> Result<Self, VeloError> {
        let dir = velo_dir(root, date, drive);
        if !dir.is_dir() {
            let drive_dir = root.join(date).join(drive);
            return Err(VeloError::NoVelodyne {
                siblings: crate::cam0::sibling_dirs(&drive_dir),
                drive_dir,
                dir,
            });
        }
        let read = |name: &str| -> Result<Vec<Option<SensorTime>>, VeloError> {
            let path = dir.join(name);
            parse_timestamp_lines(&path).map_err(|source| match source {
                TimestampError::Io(source) => VeloError::Io {
                    path: path.clone(),
                    source,
                },
                other => VeloError::Timestamps {
                    path: path.clone(),
                    source: other,
                },
            })
        };
        let [start, trigger, end] = [
            read(SWEEP_FILES[0])?,
            read(SWEEP_FILES[1])?,
            read(SWEEP_FILES[2])?,
        ];
        if start.len() != trigger.len() || trigger.len() != end.len() {
            return Err(VeloError::TimestampCount {
                start: start.len(),
                trigger: trigger.len(),
                end: end.len(),
            });
        }
        // The same frames blank in all three, or which instants a sweep has
        // would be a guess.
        let files = [&start, &trigger, &end];
        for i in 0..trigger.len() {
            let blank = files.map(|f| f[i].is_none());
            if let (Some(b), Some(t)) = (
                blank.iter().position(|&x| x),
                blank.iter().position(|&x| !x),
            ) {
                return Err(VeloError::BlankLinesDiffer {
                    line: i + 1,
                    blank: SWEEP_FILES[b],
                    filled: SWEEP_FILES[t],
                });
            }
        }
        // The trigger is the instant the camera fired with, so it is the file
        // the layout is judged on; the range's two ends must increase too.
        let index = FrameIndex::read(
            &dir.join("data"),
            "bin",
            &dir.join(SWEEP_FILES[1]),
            &trigger,
        )?;
        if let Some(i) = start
            .iter()
            .zip(&end)
            .position(|(s, e)| matches!((s, e), (Some(s), Some(e)) if e <= s))
        {
            return Err(VeloError::EndBeforeStart {
                line: i + 1,
                frame: i as u64,
            });
        }
        check_increasing(&dir.join(SWEEP_FILES[0]), &start)?;
        check_increasing(&dir.join(SWEEP_FILES[2]), &end)?;
        Ok(VeloDriver {
            root: root.to_path_buf(),
            date: date.to_string(),
            drive: drive.to_string(),
            index,
            start: instants(&start),
            trigger: instants(&trigger),
            end: instants(&end),
        })
    }

    /// Extends the stream to `n` frame slots, the drive's frame count, when
    /// that is more than the lidar's lines declare.
    ///
    /// KITTI numbers every sensor of a synced drive by one frame index, so a
    /// lidar whose timestamp files stop before another sensor's lack the
    /// frames after them, and those are absent in the source like any other;
    /// the lidar alone cannot see them. Never shrinks.
    pub fn with_frame_slots(mut self, n: usize) -> Self {
        self.index = self.index.with_slots(n);
        self
    }

    /// Number of sweeps in the drive: files, and non-blank lines in each
    /// timestamp file.
    pub fn len(&self) -> usize {
        self.trigger.len()
    }

    /// Whether the drive has no sweeps.
    pub fn is_empty(&self) -> bool {
        self.trigger.is_empty()
    }

    /// The frame of each sweep, ascending: the sweeps' `seq`s.
    pub fn frames(&self) -> &[u64] {
        self.index.frames()
    }

    /// Frame slots on the lidar stream: its timestamp files' lines, every
    /// sweep plus every frame whose sweep is absent in the source. 447 on
    /// drive 0009, whose 443 sweeps skip frames 177-180.
    pub fn frame_slots(&self) -> usize {
        self.index.slots()
    }

    /// The frames, ascending, that have no sweep in the source: `[177, 178,
    /// 179, 180]` on drive 0009, empty on a drive with a sweep per frame.
    pub fn absent_in_source(&self) -> Vec<u64> {
        self.index.absent()
    }

    /// The runs of consecutive frames absent in the source, with the real
    /// sweeps either side of each and the hole between them: from the end of
    /// the sweep before to the start of the sweep after.
    pub fn gaps(&self) -> Vec<Gap> {
        self.index.gaps(&self.end, &self.start)
    }

    /// `velodyne_points/data/{frame:010}.bin`, whether or not it exists.
    pub fn frame_path(&self, frame: u64) -> PathBuf {
        velo_dir(&self.root, &self.date, &self.drive)
            .join("data")
            .join(format!("{frame:010}.bin"))
    }

    /// The file of sweep `k` -- the `k`-th file on disk, named by its frame.
    /// `None` past the last sweep.
    pub fn sweep_path(&self, k: usize) -> Option<PathBuf> {
        self.index.frame(k).map(|f| self.frame_path(f))
    }

    /// Time of validity of sweep `k` (the `k`-th sweep, NOT frame `k`: the
    /// line of frame `index.frame(k)`): the interval it covers.
    pub fn tov(&self, k: usize) -> Option<Tov> {
        Some(Tov::Range {
            start: *self.start.get(k)?,
            end: *self.end.get(k)?,
        })
    }

    /// Host deadline of sweep `k`: the instant the rotation **completes**.
    ///
    /// Deliberately not `clock.due(self.tov(k))`, which would give the sweep's
    /// *start* — 51.6 ms earlier on drive_0005, a deadline the sensor could not
    /// have met because half the points did not exist yet.
    ///
    /// Read off the sweep's own line, so the hole a gap in the source leaves
    /// is in the schedule too: on drive 0009 sweep 181 is due 517.4 ms after
    /// sweep 176, not one period, because that is when the sensor finished it.
    pub fn due(&self, clock: &ClockModel, k: usize) -> Option<HostTime> {
        clock.due(Tov::Time(*self.end.get(k)?))
    }

    /// Host deadline of the slot of `frame`, a frame absent in the source:
    /// derived from the ends of the real sweeps either side of it (see
    /// [`crate::layout::replay_absent`]). `None` unpaced, or when the stream
    /// has no sweep to derive it from.
    pub fn absent_due(&self, clock: &ClockModel, frame: u64) -> Option<HostTime> {
        let at = self
            .index
            .slot_deadline_ns(&self.end, frame, median_period_ns(&self.end))?;
        clock.due(Tov::Time(SensorTime(at)))
    }

    /// The instant at which sweep `k` counts as skipped: the next sweep's
    /// deadline, or one median period past this one's.
    ///
    /// The next sweep that EXISTS: before a gap in the source that is the
    /// sweep after the gap, so the last sweep before a hole is still handed
    /// over late rather than dropped -- until something newer exists, it is
    /// the freshest measurement there is.
    fn skip_at(&self, clock: &ClockModel, k: usize, period_ns: i64) -> Option<HostTime> {
        match self.end.get(k + 1) {
            Some(&next) => clock.due(Tov::Time(next)),
            None => {
                let last = self.end.get(k)?;
                clock.due(Tov::Time(SensorTime(last.0 + period_ns)))
            }
        }
    }

    /// Replays every frame slot at the clock's rate, calling `admit` exactly
    /// once per slot: a `Sample` for a sweep, whose `seq` is its frame, or a
    /// `Missing`. Same shape as [`crate::cam0::Cam0Driver::run`]: absolute
    /// deadlines, the skip test taken both before the read and after the
    /// Arrow build, and a slow admit costing the *next* sweep rather than
    /// shifting the phase.
    ///
    /// A frame absent in the source is admitted as `Missing` with reason
    /// [`ABSENT_IN_SOURCE`](crate::layout::ABSENT_IN_SOURCE) and
    /// [`Tov::None`], once its derived deadline has passed
    /// ([`crate::layout::replay_absent`]), so the replay sits through the
    /// hole the sensor left rather than closing it: on drive 0009 nothing is
    /// handed over for the 414 ms the source recorded nothing.
    pub fn run(
        &self,
        clock: &ClockModel,
        spin_window: Duration,
        schema: &Arc<Schema>,
        admit: &mut dyn FnMut(DriverEvent),
    ) -> RunTotals {
        let t_start = now();
        let period_ns = median_period_ns(&self.end);
        let mut admitted = 0u64;
        let mut missing = 0u64;
        for slot in self.index.slot_list() {
            let (frame, i) = match slot {
                Slot::Absent { frame } => {
                    replay_absent(
                        &self.index,
                        &self.end,
                        period_ns,
                        clock,
                        spin_window,
                        frame,
                        admit,
                    );
                    missing += 1;
                    continue;
                }
                Slot::Sample { frame, k } => (frame, k),
            };
            let seq = frame;
            // `start` and `end` have one entry per sweep (`open` checks it),
            // so this cannot be `None`; if it ever were, the slot is still
            // reported rather than silently skipped.
            let Some(tov) = self.tov(i) else {
                admit(DriverEvent::Missing {
                    seq,
                    tov: Tov::None,
                    due: None,
                    reason: "read_error",
                });
                missing += 1;
                continue;
            };
            let due = self.due(clock, i);
            let skip_at = self.skip_at(clock, i, period_ns);
            let mut miss = |reason: &'static str, admit: &mut dyn FnMut(DriverEvent)| {
                admit(DriverEvent::Missing {
                    seq,
                    tov,
                    due,
                    reason,
                });
                missing += 1;
            };

            if skip_at.is_some_and(|s| now() >= s) {
                miss("deadline_skipped", admit);
                continue;
            }
            let path = self.frame_path(frame);
            let t_read = now();
            let points = match read_sweep(&path) {
                Ok(b) => b,
                Err(e) => {
                    eprintln!("velo: sweep {seq} ({}): {e}", path.display());
                    miss("read_error", admit);
                    continue;
                }
            };
            // The lidar's analogue of the camera's `decode_ns`: the cost of
            // turning a file into points. There is no decoder here, so it is
            // the read — which is the honest comparison, because it is the
            // whole of what this driver spends per sample before the wrap.
            let decode_ns = now() - t_read;
            let trigger_ns = self.trigger.get(i).map_or(0, |t| t.0);
            let (batch, storage_id) = match build_velo_batch(points, trigger_ns, schema) {
                Ok(x) => x,
                Err(e) => {
                    eprintln!("velo: sweep {seq} ({}): {e}", path.display());
                    miss("batch_error", admit);
                    continue;
                }
            };
            if skip_at.is_some_and(|s| now() >= s) {
                miss("deadline_skipped", admit);
                continue;
            }
            if let Some(d) = due {
                sleep_until(d, spin_window);
            }
            let arrival = now();
            admit(DriverEvent::Sample(Sample {
                stream: StreamId::LIDAR,
                seq,
                arrival_seq: 0,
                parent: None,
                tov,
                epoch: clock.epoch,
                due,
                arrival,
                payload: batch,
                storage_id,
                decode_ns,
            }));
            admitted += 1;
        }
        RunTotals {
            admitted,
            missing,
            wall_ns: now() - t_start,
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::layout::ABSENT_IN_SOURCE;

    /// `n` points whose x is the point index, so every value is predictable.
    fn sweep_bytes(n: usize) -> Vec<u8> {
        let mut v = Vec::with_capacity(n * POINT_BYTES);
        for i in 0..n {
            for k in 0..4 {
                v.extend_from_slice(&(i as f32 + k as f32 / 10.0).to_le_bytes());
            }
        }
        v
    }

    fn buffer_of(bytes: &[u8]) -> Buffer {
        let mut mb = MutableBuffer::from_len_zeroed(bytes.len());
        mb.as_slice_mut().copy_from_slice(bytes);
        Buffer::from(mb)
    }

    /// The lidar half of the standing zero-copy proof (`frame.rs` g2a): the
    /// address the driver records is the address the payload still hands back,
    /// three downcasts deep, and the bytes read as the points that were written.
    #[test]
    fn v1_points_ptr_equals_storage_id_and_survives_the_nesting() {
        let bytes = sweep_bytes(7);
        let buf = buffer_of(&bytes);
        let p_buf = buf.as_ptr() as usize;
        let schema = velo_schema();
        let (batch, storage_id) = build_velo_batch(buf, 1_234, &schema).unwrap();
        assert_eq!(storage_id, p_buf, "the point buffer was copied");
        assert_eq!(velo_storage_id(&batch), Some(storage_id));
        assert_eq!(batch.num_rows(), 1, "one row per sweep, not per point");
        assert_eq!(velo_point_count(&batch), Some(7));
        assert_eq!(velo_trigger_ns(&batch), Some(1_234));
        let xyzr = velo_xyzr(&batch).unwrap();
        assert_eq!(xyzr.len(), 28);
        assert_eq!(xyzr.as_ptr() as usize, storage_id);
        let p3: Vec<f32> = xyzr.as_chunks::<4>().0[3].to_vec();
        assert_eq!(p3, vec![3.0, 3.1, 3.2, 3.3]);
    }

    /// The counterpart of `frame.rs`'s g2b: batch construction is a constant
    /// cost, so the zero-copy claim does not quietly depend on sweep size.
    #[test]
    fn v2_batch_alloc_small_and_point_count_invariant() {
        // SLOT_TEST2, not SLOT_TEST, and that is load-bearing. The counters are
        // global atomics, and `frame.rs`'s `g2b_batch_alloc_small_and_resolution_invariant`
        // -- the camera's zero-copy proof -- already holds
        // SLOT_TEST. Both live in this crate's test binary, which cargo runs in
        // parallel, so sharing the slot makes each test measure the other's
        // allocations. It fails or passes on thread scheduling: measured here as
        // 1632 B for every point count from 1024 to 123,397, against 1888-19,704 B
        // of noise below that, purely from the overlap.
        use pipes_core::alloc::{bytes_alloc, set_stage_slot, SLOT_TEST2};
        let schema = velo_schema(); // hoisted: not counted
        set_stage_slot(SLOT_TEST2);
        let measure = |n: usize| -> u64 {
            let buf = buffer_of(&sweep_bytes(n)); // the payload is allocated BEFORE the window
            let before = bytes_alloc(SLOT_TEST2);
            let (batch, _) = build_velo_batch(buf, 0, &schema).unwrap();
            let after = bytes_alloc(SLOT_TEST2);
            drop(batch);
            after - before
        };
        let _warm = measure(4); // absorbs any first-call one-off
        let small = measure(16);
        let full = measure(123_397);
        println!("v2: sweep batch construction allocated small={small} B full={full} B");
        assert!(small < 4096, "batch construction allocated {small} B");
        assert_eq!(
            small, full,
            "allocation depends on point count: {small} vs {full}"
        );
    }

    /// The variable-length file is the whole reason this builder cannot take a
    /// `BadLength {{ expected, got }}` shape: there is no expected length, only
    /// the multiple-of-16 rule.
    #[test]
    fn v3_a_partial_point_is_a_typed_error_not_a_silent_truncation() {
        let schema = velo_schema();
        let mut bytes = sweep_bytes(3);
        bytes.truncate(bytes.len() - 5); // 43 bytes: two points and a fragment
        let err = build_velo_batch(buffer_of(&bytes), 0, &schema).unwrap_err();
        assert!(
            matches!(err, SweepError::NotWholePoints { got: 43 }),
            "{err:?}"
        );
        // The positive control: the untruncated buffer builds, so the check
        // above is rejecting the fragment rather than rejecting everything.
        let (ok, _) = build_velo_batch(buffer_of(&sweep_bytes(3)), 0, &schema).unwrap();
        assert_eq!(velo_point_count(&ok), Some(3));
    }

    /// `ScalarBuffer::<f32>::from` answers a misaligned buffer with a panic.
    /// The driver can never hand it one, so this proves the guard fires by
    /// constructing the misalignment on purpose — otherwise the `Misaligned`
    /// arm would be a branch no test can reach.
    #[test]
    fn v4_a_misaligned_buffer_is_a_typed_error_not_an_arrow_panic() {
        let schema = velo_schema();
        let whole = buffer_of(&sweep_bytes(3));
        // Slice one byte in: still a whole number of points in length, and
        // guaranteed odd-addressed because `whole` is 128-byte aligned.
        let skewed = whole.slice_with_length(1, POINT_BYTES * 2);
        assert_ne!(
            skewed.as_ptr() as usize % 4,
            0,
            "the slice is still aligned"
        );
        let err = build_velo_batch(skewed, 0, &schema).unwrap_err();
        assert!(matches!(err, SweepError::Misaligned { .. }), "{err:?}");
        // Positive control: the same slice taken at a 16-byte offset is
        // aligned and builds, so the guard is about alignment and not about
        // slices.
        let aligned = whole.slice_with_length(POINT_BYTES, POINT_BYTES * 2);
        let (b, _) = build_velo_batch(aligned, 0, &schema).unwrap();
        assert_eq!(velo_point_count(&b), Some(2));
    }

    /// `read_sweep`'s reason for existing: a buffer that can be viewed as f32
    /// by construction rather than by the allocator's good manners.
    #[test]
    fn v5_read_sweep_is_aligned_and_whole() {
        let dir = std::env::temp_dir().join(format!("pipes-velo-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(sweep_file_name(0));
        let bytes = sweep_bytes(5);
        std::fs::write(&path, &bytes).unwrap();
        let buf = read_sweep(&path).unwrap();
        assert_eq!(buf.len(), bytes.len());
        assert_eq!(buf.as_slice(), &bytes[..]);
        assert_eq!(
            buf.as_ptr() as usize % std::mem::align_of::<f32>(),
            0,
            "MutableBuffer is supposed to make this structural"
        );
        let (batch, _) = build_velo_batch(buf, 9, &velo_schema()).unwrap();
        assert_eq!(velo_point_count(&batch), Some(5));
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A contiguous driver: sweep `k` is frame `k`.
    fn driver(start: &[i64], trigger: &[i64], end: &[i64]) -> VeloDriver {
        VeloDriver {
            root: PathBuf::from("root"),
            date: "d".to_string(),
            drive: "drv".to_string(),
            index: FrameIndex::contiguous(start.len()),
            start: start.iter().copied().map(SensorTime).collect(),
            trigger: trigger.iter().copied().map(SensorTime).collect(),
            end: end.iter().copied().map(SensorTime).collect(),
        }
    }

    fn clock_at(t0_sensor: i64) -> ClockModel {
        ClockModel {
            epoch: 0,
            t0_sensor: SensorTime(t0_sensor),
            t0_host: HostTime(0),
            t0_wall: 0,
            rate_factor: 1.0,
            offset_uncertainty_ns: 0,
            sensor_time_source: String::new(),
        }
    }

    /// The pacing decision, pinned: the deadline is the end of the rotation,
    /// which for a real sweep is half a period after the deadline
    /// `ClockModel::due` would give the same `Tov::Range`.
    #[test]
    fn v6_the_deadline_is_the_end_of_the_sweep_not_its_start() {
        let d = driver(&[1_000, 1_100], &[1_050, 1_150], &[1_100, 1_200]);
        let clock = clock_at(1_000);
        assert_eq!(d.due(&clock, 0), Some(HostTime(100)));
        assert_eq!(d.due(&clock, 1), Some(HostTime(200)));
        assert_eq!(d.due(&clock, 2), None);
        // What the range on its own would have said — 100 ns early, the shape
        // of the 51.6 ms error on the real drive.
        assert_eq!(clock.due(d.tov(0).unwrap()), Some(HostTime(0)));
        assert_eq!(
            d.tov(0),
            Some(Tov::Range {
                start: SensorTime(1_000),
                end: SensorTime(1_100)
            })
        );
    }

    #[test]
    fn v7_skip_at_is_the_next_end_or_one_period_after_the_last() {
        let d = driver(&[1_000, 1_100], &[1_050, 1_150], &[1_100, 1_200]);
        let clock = clock_at(1_000);
        assert_eq!(d.skip_at(&clock, 0, 100), Some(HostTime(200)));
        assert_eq!(d.skip_at(&clock, 1, 100), Some(HostTime(300)));
        assert_eq!(d.skip_at(&clock, 2, 100), None);
    }

    /// The synthetic drive really is one this driver can open, and its three
    /// timestamp files really are three measurements.
    ///
    /// Without this the fixture is an assumption the integration tests rest
    /// on: they would run the binary against it, see no lidar, and report a
    /// clean camera-only run as a pass. `has_velodyne` is asserted BOTH ways
    /// here, so "the fixture has lidar" is a claim with a control.
    #[test]
    fn v9_the_synthetic_drive_opens_and_carries_three_distinct_timestamps() {
        use crate::testing::{
            fixture_sweep_points, write_velo_fixture_at, FIXTURE_DATE, FIXTURE_DRIVE,
        };
        let tmp = tempfile::TempDir::new().unwrap();
        let root = tmp.path();
        let period = 100_000_000i64;

        // The negative control first: nothing has been written yet.
        assert!(
            !has_velodyne(root, FIXTURE_DATE, FIXTURE_DRIVE),
            "has_velodyne answered yes before anything existed"
        );
        write_velo_fixture_at(root, FIXTURE_DATE, FIXTURE_DRIVE, 4, period).unwrap();
        assert!(has_velodyne(root, FIXTURE_DATE, FIXTURE_DRIVE));

        let d = VeloDriver::open(root, FIXTURE_DATE, FIXTURE_DRIVE).unwrap();
        assert_eq!(d.len(), 4);
        for i in 0..4 {
            let (st, tr, en) = (d.start[i].0, d.trigger[i].0, d.end[i].0);
            // Three separate numbers, and the trigger strictly inside the
            // range -- which is the whole reason `Tov::Range` cannot carry it.
            assert!(st < tr && tr < en, "sweep {i}: {st} {tr} {en}");
            assert_eq!(en - st, period, "sweep {i} does not span a period");
            assert_eq!(tr - st, period / 2, "sweep {i} trigger is not the midpoint");
            assert_eq!(
                d.tov(i),
                Some(Tov::Range {
                    start: SensorTime(st),
                    end: SensorTime(en)
                })
            );
        }
        // Consecutive sweeps abut: one ends where the next begins.
        assert_eq!(d.end[0], d.start[1]);

        // And the payload is readable as points, at the per-sweep count the
        // fixture claims -- so a consumer asserting on point counts is
        // asserting on data, not on a constant it also wrote.
        let schema = velo_schema();
        for i in 0..4 {
            let buf = read_sweep(&d.sweep_path(i).unwrap()).unwrap();
            let (batch, _) = build_velo_batch(buf, d.trigger[i].0, &schema).unwrap();
            assert_eq!(
                velo_point_count(&batch),
                Some(fixture_sweep_points(i) as u32)
            );
            assert_eq!(velo_trigger_ns(&batch), Some(d.trigger[i].0));
            let xyzr = velo_xyzr(&batch).unwrap();
            let pts = xyzr.as_chunks::<4>().0;
            assert_eq!(pts.len(), fixture_sweep_points(i));
            // Point j is (j, -j, i, 0.5), so the last point pins both ends.
            let last = pts[pts.len() - 1];
            assert_eq!(
                last,
                [
                    (pts.len() - 1) as f32,
                    -((pts.len() - 1) as f32),
                    i as f32,
                    0.5
                ]
            );
        }
    }

    #[test]
    fn v8_sweep_path_is_zero_padded() {
        let d = driver(&[], &[], &[]);
        assert_eq!(
            d.frame_path(7),
            Path::new("root")
                .join("d")
                .join("drv")
                .join(VELODYNE_DIR)
                .join("data")
                .join("0000000007.bin")
        );
        // No sweep 0 in a driver with no sweeps, rather than a path to a
        // file that is not there.
        assert_eq!(d.sweep_path(0), None);
    }

    // ---- gaps in the source ------------------------------------------------
    //
    // Drive 0009's shape in miniature: ten frames, no sweep for frames 3 and
    // 4, and ten lines in each timestamp file -- line i + 1 for frame i, blank
    // at lines 4 and 5.

    const GAP_FRAMES: usize = 10;
    const GAP_ABSENT: [usize; 2] = [3, 4];
    const GAP_PERIOD: i64 = 100_000_000;

    fn gapped() -> (tempfile::TempDir, PathBuf) {
        use crate::testing::{write_velo_fixture_gapped_at, FIXTURE_DATE, FIXTURE_DRIVE};
        let tmp = tempfile::TempDir::new().unwrap();
        write_velo_fixture_gapped_at(
            tmp.path(),
            FIXTURE_DATE,
            FIXTURE_DRIVE,
            GAP_FRAMES,
            &GAP_ABSENT,
            GAP_PERIOD,
        )
        .unwrap();
        let dir = velo_dir(
            tmp.path(),
            crate::testing::FIXTURE_DATE,
            crate::testing::FIXTURE_DRIVE,
        );
        (tmp, dir)
    }

    fn open_gapped(tmp: &tempfile::TempDir) -> Result<VeloDriver, VeloError> {
        VeloDriver::open(
            tmp.path(),
            crate::testing::FIXTURE_DATE,
            crate::testing::FIXTURE_DRIVE,
        )
    }

    /// Every event a replay hands to `admit`, in order: `(seq, reason, tov,
    /// due, points, trigger)`, `reason` empty on a sample.
    type Seen = (
        u64,
        &'static str,
        Tov,
        Option<HostTime>,
        Option<u32>,
        Option<i64>,
    );

    fn replay(d: &VeloDriver, clock: &ClockModel) -> (Vec<Seen>, RunTotals) {
        let schema = velo_schema();
        let mut seen: Vec<Seen> = Vec::new();
        let totals = d.run(
            clock,
            Duration::from_micros(500),
            &schema,
            &mut |ev| match ev {
                DriverEvent::Sample(s) => seen.push((
                    s.seq,
                    "",
                    s.tov,
                    s.due,
                    velo_point_count(&s.payload),
                    velo_trigger_ns(&s.payload),
                )),
                DriverEvent::Missing {
                    seq,
                    tov,
                    due,
                    reason,
                } => seen.push((seq, reason, tov, due, None, None)),
            },
        );
        (seen, totals)
    }

    fn unpaced() -> ClockModel {
        let mut c = clock_at(0);
        c.rate_factor = f64::INFINITY;
        c
    }

    /// The rule itself: line `i + 1` of each timestamp file is frame `i`,
    /// blank where the frame has no sweep, and a sweep is named by its frame.
    /// The fourth sweep is frame 5: it reads `0000000005.bin`, carries line 6
    /// of the timestamps, and its points are the ones written for frame 5.
    #[test]
    fn v10_a_gapped_drive_opens_and_line_i_plus_1_is_frame_i() {
        let (tmp, dir) = gapped();
        let d = open_gapped(&tmp).expect("KITTI's gapped layout must open");
        assert_eq!(
            d.len(),
            8,
            "eight sweeps: one per file, one per non-blank line"
        );
        assert_eq!(d.frames(), [0, 1, 2, 5, 6, 7, 8, 9]);
        assert_eq!(d.frame_slots(), GAP_FRAMES, "ten lines, ten frames");
        assert_eq!(d.absent_in_source(), vec![3, 4]);
        let text = std::fs::read_to_string(dir.join("timestamps_start.txt")).unwrap();
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines.len(), GAP_FRAMES);
        assert_eq!((lines[3], lines[4]), ("", ""), "lines 4-5 are frames 3-4");

        // The fourth sweep is frame 5's file...
        let path = d.sweep_path(3).unwrap();
        assert!(path.ends_with("0000000005.bin"), "{}", path.display());
        // ...with line 6 of each timestamp file, which is frame 5's instant.
        let base = crate::testing::FIXTURE_BASE_NS;
        assert_eq!(
            lines[5],
            crate::testing::fixture_timestamp_line_ns(5 * GAP_PERIOD)
        );
        assert_eq!(
            d.tov(3),
            Some(Tov::Range {
                start: SensorTime(base + 5 * GAP_PERIOD),
                end: SensorTime(base + 6 * GAP_PERIOD),
            })
        );
        // And the points are frame 5's: point j is (j, -j, frame, 0.5).
        let buf = read_sweep(&path).unwrap();
        let (batch, _) = build_velo_batch(buf, d.trigger[3].0, &velo_schema()).unwrap();
        let xyzr = velo_xyzr(&batch).unwrap();
        assert_eq!(
            xyzr.len() / 4,
            crate::testing::fixture_sweep_points(5),
            "the points of the file NAMED 5, not of the fourth line's number"
        );
        assert!(xyzr.as_chunks::<4>().0.iter().all(|p| p[2] == 5.0));

        // The hole, measured from the two real sweeps either side of it.
        assert_eq!(
            d.gaps(),
            vec![Gap {
                first: 3,
                last: 4,
                before: Some(2),
                after: Some(5),
                hole_ns: Some(2 * GAP_PERIOD),
            }]
        );
        assert_eq!(
            d.gaps()[0].describe(),
            "frames 3-4 (2 frames, 200.0 ms with no measurement between frame 2 and frame 5)"
        );
    }

    /// The replay: one event per frame slot, in frame order; every sweep's
    /// `seq` is its frame and it carries its own line's range and trigger;
    /// and exactly the two absent frames come out as `Missing`, with the
    /// reason that names the cause and no time of validity.
    #[test]
    fn v11_the_replay_numbers_sweeps_by_frame_and_records_the_absent_ones() {
        let (tmp, _dir) = gapped();
        let d = open_gapped(&tmp).unwrap();
        let (seen, totals) = replay(&d, &unpaced());
        let seqs: Vec<u64> = seen.iter().map(|e| e.0).collect();
        assert_eq!(seqs, (0..GAP_FRAMES as u64).collect::<Vec<_>>());

        let samples: Vec<&Seen> = seen.iter().filter(|e| e.1.is_empty()).collect();
        let missing: Vec<&Seen> = seen.iter().filter(|e| !e.1.is_empty()).collect();
        assert_eq!(
            samples.iter().map(|e| e.0).collect::<Vec<_>>(),
            d.frames(),
            "a sweep's seq is its frame number"
        );
        for (k, s) in samples.iter().enumerate() {
            let frame = s.0 as usize;
            assert_eq!(s.2, d.tov(k).unwrap(), "sweep {frame}: not its own line");
            assert_eq!(s.5, Some(d.trigger[k].0), "sweep {frame}: trigger");
            assert_eq!(
                s.4,
                Some(crate::testing::fixture_sweep_points(frame) as u32),
                "sweep {frame}: not its own file"
            );
        }
        assert_eq!(missing.len(), 2, "{missing:?}");
        for (m, want) in missing.iter().zip(GAP_ABSENT) {
            assert_eq!(m.0, want as u64);
            assert_eq!(m.1, ABSENT_IN_SOURCE);
            // Never an invented instant: the source measured nothing here.
            assert_eq!(m.2, Tov::None);
            // Unpaced, so no deadline either.
            assert_eq!(m.3, None);
        }
        assert_eq!((totals.admitted, totals.missing), (8, 2));
        assert_eq!(
            (totals.admitted + totals.missing) as usize,
            d.frame_slots(),
            "every frame slot accounted for exactly once"
        );
    }

    /// The schedule keeps the hole the sensor left: sweep 5 falls due three
    /// periods after sweep 2, as its own timestamps say, not one; and the
    /// absent frames' derived deadlines sit evenly inside it, where the
    /// sweeps would have been.
    #[test]
    fn v12_the_schedule_keeps_the_hole() {
        let (tmp, _dir) = gapped();
        let d = open_gapped(&tmp).unwrap();
        let base = crate::testing::FIXTURE_BASE_NS;
        let clock = clock_at(base);
        let p = GAP_PERIOD;
        // Sweep index 2 is frame 2, index 3 is frame 5.
        assert_eq!(d.due(&clock, 2), Some(HostTime(3 * p)));
        assert_eq!(d.due(&clock, 3), Some(HostTime(6 * p)));
        assert_eq!(d.absent_due(&clock, 3), Some(HostTime(4 * p)));
        assert_eq!(d.absent_due(&clock, 4), Some(HostTime(5 * p)));
        // At twice the rate the hole is half as long, and still a hole.
        let mut fast = clock_at(base);
        fast.rate_factor = 2.0;
        assert_eq!(
            d.due(&fast, 3).unwrap() - d.due(&fast, 2).unwrap(),
            3 * p / 2
        );
        // The last sweep before the hole counts as skipped only once the
        // sweep after it is due: until then it is the freshest there is.
        assert_eq!(d.skip_at(&clock, 2, p), Some(HostTime(6 * p)));
        // Unpaced, there is nothing to derive.
        assert_eq!(d.absent_due(&unpaced(), 3), None);
    }

    /// The replay really waits: paced, each absent frame is reported only
    /// once its derived deadline has passed, and sweep 5 is not handed over
    /// before three periods after sweep 2's deadline.
    #[test]
    fn v12b_a_paced_replay_sits_through_the_hole() {
        let (tmp, _dir) = gapped();
        let d = open_gapped(&tmp).unwrap();
        // 20x: a 10 ms period, the whole drive in 50 ms.
        let clock = ClockModel::start_now(SensorTime(crate::testing::FIXTURE_BASE_NS), 20.0);
        let schema = velo_schema();
        let mut when: Vec<(u64, &'static str, Option<HostTime>, HostTime, Tov)> = Vec::new();
        d.run(&clock, Duration::from_micros(500), &schema, &mut |ev| {
            let t = now();
            match ev {
                DriverEvent::Sample(s) => when.push((s.seq, "", s.due, t, s.tov)),
                DriverEvent::Missing {
                    seq,
                    due,
                    reason,
                    tov,
                } => when.push((seq, reason, due, t, tov)),
            }
        });
        let of = |seq: u64| when.iter().find(|w| w.0 == seq).copied().unwrap();
        for f in GAP_ABSENT {
            let (_, reason, due, t, tov) = of(f as u64);
            assert_eq!(reason, ABSENT_IN_SOURCE);
            // Paced, the slot has a deadline -- and still no instant.
            assert_eq!(tov, Tov::None, "frame {f}: an instant was invented");
            let due = due.expect("paced, so the absent frame has a derived deadline");
            assert!(
                t >= due,
                "frame {f} reported {} ns before its slot",
                due - t
            );
        }
        let (_, reason, due5, t5, _) = of(5);
        assert_eq!(reason, "", "sweep 5 was not handed over: {reason}");
        let due2 = of(2).2.unwrap();
        assert_eq!(due5.unwrap() - due2, 3 * GAP_PERIOD / 20);
        assert!(
            t5 - due2 >= 3 * GAP_PERIOD / 20,
            "sweep 5 was handed over {} ns after sweep 2's deadline: the hole was compressed",
            t5 - due2
        );
    }

    /// The positive control: a drive with a sweep for every frame is what it
    /// was -- sweep `k` is frame `k`, nothing is absent, and the replay
    /// reports no `Missing` at all.
    #[test]
    fn v14_a_contiguous_drive_is_unchanged() {
        use crate::testing::{write_velo_fixture_at, FIXTURE_DATE, FIXTURE_DRIVE};
        let tmp = tempfile::TempDir::new().unwrap();
        write_velo_fixture_at(tmp.path(), FIXTURE_DATE, FIXTURE_DRIVE, 6, GAP_PERIOD).unwrap();
        let d = VeloDriver::open(tmp.path(), FIXTURE_DATE, FIXTURE_DRIVE).unwrap();
        assert_eq!(d.frames(), [0, 1, 2, 3, 4, 5]);
        assert_eq!((d.len(), d.frame_slots()), (6, 6));
        assert!(d.absent_in_source().is_empty());
        assert!(d.gaps().is_empty());
        let (seen, totals) = replay(&d, &unpaced());
        assert!(seen.iter().all(|e| e.1.is_empty()), "{seen:?}");
        assert_eq!(
            seen.iter().map(|e| e.0).collect::<Vec<_>>(),
            vec![0, 1, 2, 3, 4, 5]
        );
        assert_eq!((totals.admitted, totals.missing), (6, 0));
    }

    /// The camera's frame count extends the stream past the last sweep:
    /// frames the camera has and the lidar does not are absent too, and are
    /// replayed as such. It never shrinks the stream.
    #[test]
    fn v15_frames_past_the_last_sweep_are_absent_too() {
        use crate::testing::{write_velo_fixture_at, FIXTURE_DATE, FIXTURE_DRIVE};
        let tmp = tempfile::TempDir::new().unwrap();
        write_velo_fixture_at(tmp.path(), FIXTURE_DATE, FIXTURE_DRIVE, 4, GAP_PERIOD).unwrap();
        let open = || VeloDriver::open(tmp.path(), FIXTURE_DATE, FIXTURE_DRIVE).unwrap();
        assert_eq!(open().with_frame_slots(2).frame_slots(), 4, "shrank");
        let d = open().with_frame_slots(6);
        assert_eq!(d.absent_in_source(), vec![4, 5]);
        assert_eq!(
            d.gaps(),
            vec![Gap {
                first: 4,
                last: 5,
                before: Some(3),
                after: None,
                hole_ns: None,
            }]
        );
        // One median period per frame past the last sweep's end.
        let clock = clock_at(crate::testing::FIXTURE_BASE_NS);
        assert_eq!(d.absent_due(&clock, 5), Some(HostTime(6 * GAP_PERIOD)));
        let (seen, totals) = replay(&d, &unpaced());
        assert_eq!(
            seen.iter().map(|e| (e.0, e.1)).collect::<Vec<_>>(),
            vec![
                (0, ""),
                (1, ""),
                (2, ""),
                (3, ""),
                (4, ABSENT_IN_SOURCE),
                (5, ABSENT_IN_SOURCE)
            ]
        );
        assert_eq!((totals.admitted, totals.missing), (4, 2));
    }

    /// A line with a time and no file, and a file whose lines are blank or
    /// past the end, are refused, each at its real line: KITTI blanks the
    /// line of a frame it has no file for.
    #[test]
    fn v13_files_and_lines_that_break_the_layout_are_refused_at_their_real_line() {
        // Frame 6's file gone, its lines kept: line 7.
        let (tmp, dir) = gapped();
        std::fs::remove_file(dir.join("data").join("0000000006.bin")).unwrap();
        let err = open_gapped(&tmp)
            .err()
            .expect("a line with a time and no file opened");
        assert!(
            matches!(
                &err,
                VeloError::Layout(LayoutError::MissingFile { line: 7, name, .. })
                    if name == "0000000006.bin"
            ),
            "{err:?}"
        );
        let text = err.to_string();
        for needle in [
            "timestamps.txt line 7 has a time",
            "0000000006.bin is not in",
            "would be a guess",
            "drives",
        ] {
            assert!(text.contains(needle), "{needle:?} not in: {text}");
        }

        // The layout with one line per EXISTING file: the gap's lines taken
        // out rather than blanked. Line 4 then has frame 5's time and names
        // frame 3, which has no file.
        let (tmp, dir) = gapped();
        for f in SWEEP_FILES {
            let text = std::fs::read_to_string(dir.join(f)).unwrap();
            let kept: String = text
                .lines()
                .filter(|l| !l.is_empty())
                .map(|l| format!("{l}\n"))
                .collect();
            std::fs::write(dir.join(f), kept).unwrap();
        }
        let err = open_gapped(&tmp).err().expect("dropped lines opened");
        assert!(
            matches!(
                &err,
                VeloError::Layout(LayoutError::MissingFile { line: 4, .. })
            ),
            "{err:?}"
        );

        // A file on a blank line: frame 3's file written into the gap.
        let (tmp, dir) = gapped();
        std::fs::write(dir.join("data").join("0000000003.bin"), [0u8; 16]).unwrap();
        let err = open_gapped(&tmp)
            .err()
            .expect("a file on a blank line opened");
        assert!(
            matches!(
                &err,
                VeloError::Layout(LayoutError::FileOnBlankLine { line: 4, .. })
            ),
            "{err:?}"
        );

        // A file past the last line.
        let (tmp, dir) = gapped();
        std::fs::write(dir.join("data").join("0000000010.bin"), [0u8; 16]).unwrap();
        let err = open_gapped(&tmp).err().expect("a file past the end opened");
        assert!(
            matches!(
                &err,
                VeloError::Layout(LayoutError::FileBeyondEnd {
                    lines: 10,
                    line: 11,
                    ..
                })
            ),
            "{err:?}"
        );
    }

    #[test]
    fn v16_timestamp_files_of_different_lengths_are_refused() {
        let (tmp, dir) = gapped();
        let path = dir.join("timestamps_end.txt");
        let text = std::fs::read_to_string(&path).unwrap();
        let short: String = text.lines().take(9).map(|l| format!("{l}\n")).collect();
        std::fs::write(&path, short).unwrap();
        let err = open_gapped(&tmp)
            .err()
            .expect("unequal timestamp files opened");
        assert!(
            matches!(
                err,
                VeloError::TimestampCount {
                    start: 10,
                    trigger: 10,
                    end: 9
                }
            ),
            "{err:?}"
        );
        assert!(
            err.to_string().contains(
                "timestamps_start.txt ends at line 10, timestamps.txt at line 10 and timestamps_end.txt at line 9"
            ),
            "{err}"
        );
    }

    /// A frame's line blank in one file and not in another: which instants
    /// its sweep has would be a guess.
    #[test]
    fn v16b_blank_lines_that_differ_between_the_three_files_are_refused() {
        let (tmp, dir) = gapped();
        let path = dir.join("timestamps_end.txt");
        let mut lines: Vec<String> = std::fs::read_to_string(&path)
            .unwrap()
            .lines()
            .map(str::to_string)
            .collect();
        // Frame 3's end line filled in: line 4.
        lines[3] = crate::testing::fixture_timestamp_line_ns(4 * GAP_PERIOD);
        std::fs::write(&path, lines.join("\n") + "\n").unwrap();
        let err = open_gapped(&tmp)
            .err()
            .expect("differing blank lines opened");
        assert!(
            matches!(
                err,
                VeloError::BlankLinesDiffer {
                    line: 4,
                    blank: "timestamps_start.txt",
                    filled: "timestamps_end.txt"
                }
            ),
            "{err:?}"
        );
        assert!(
            err.to_string().contains(
                "timestamps_start.txt line 4 is blank but timestamps_end.txt line 4 has a time"
            ),
            "{err}"
        );
    }

    /// A `.bin` not named by a zero-padded frame number has no frame, so it
    /// is refused rather than slotted somewhere. Three spellings that would
    /// each parse to a number if the check were lenient.
    #[test]
    fn v17_a_sweep_file_not_named_by_its_frame_is_refused() {
        for bad in ["sweep_a.bin", "5.bin", "0000000005.BIN"] {
            let (tmp, dir) = gapped();
            let data = dir.join("data");
            std::fs::rename(data.join("0000000005.bin"), data.join(bad)).unwrap();
            let err = open_gapped(&tmp)
                .err()
                .unwrap_or_else(|| panic!("{bad} was accepted"));
            match &err {
                VeloError::Layout(LayoutError::NotAFrameNumber { name, .. }) => {
                    assert_eq!(name, bad)
                }
                other => panic!("{bad}: {other:?}"),
            }
            let text = err.to_string();
            assert!(
                text.contains(bad) && text.contains("would be a guess"),
                "{text}"
            );
        }
    }

    #[test]
    fn v18_timestamps_that_do_not_increase_are_refused() {
        let (tmp, dir) = gapped();
        let path = dir.join("timestamps.txt");
        let mut lines: Vec<String> = std::fs::read_to_string(&path)
            .unwrap()
            .lines()
            .map(str::to_string)
            .collect();
        // Frames 6 and 7, lines 7 and 8, swapped.
        lines.swap(6, 7);
        std::fs::write(&path, lines.join("\n") + "\n").unwrap();
        let err = open_gapped(&tmp).err().expect("out-of-order lines opened");
        match &err {
            VeloError::Layout(LayoutError::NotIncreasing {
                path: p,
                line,
                prev_line,
            }) => {
                assert!(p.ends_with("timestamps.txt"), "{}", p.display());
                assert_eq!((*line, *prev_line), (8, 7));
            }
            other => panic!("{other:?}"),
        }
        assert!(
            err.to_string().contains("line 8 is not later than line 7"),
            "{err}"
        );
        // And across the gap: frame 5's line (6) set before frame 2's (3).
        let (tmp, dir) = gapped();
        let path = dir.join("timestamps.txt");
        let mut lines: Vec<String> = std::fs::read_to_string(&path)
            .unwrap()
            .lines()
            .map(str::to_string)
            .collect();
        lines[5] = crate::testing::fixture_timestamp_line_ns(GAP_PERIOD);
        std::fs::write(&path, lines.join("\n") + "\n").unwrap();
        let err = open_gapped(&tmp)
            .err()
            .expect("out of order across the gap");
        assert!(
            err.to_string().contains("line 6 is not later than line 3"),
            "{err}"
        );

        // A sweep that ends where it starts is not a sweep.
        let (tmp, dir) = gapped();
        let path = dir.join("timestamps_end.txt");
        let mut lines: Vec<String> = std::fs::read_to_string(&path)
            .unwrap()
            .lines()
            .map(str::to_string)
            .collect();
        lines[0] = crate::testing::fixture_timestamp_line_ns(0);
        std::fs::write(&path, lines.join("\n") + "\n").unwrap();
        let err = open_gapped(&tmp).err().expect("a zero-length sweep opened");
        assert!(
            matches!(err, VeloError::EndBeforeStart { line: 1, frame: 0 }),
            "{err:?}"
        );
        // After a gap the line is the frame's own: frame 5 is line 6.
        let (tmp, dir) = gapped();
        let path = dir.join("timestamps_end.txt");
        let mut lines: Vec<String> = std::fs::read_to_string(&path)
            .unwrap()
            .lines()
            .map(str::to_string)
            .collect();
        lines[5] = crate::testing::fixture_timestamp_line_ns(5 * GAP_PERIOD);
        std::fs::write(&path, lines.join("\n") + "\n").unwrap();
        let err = open_gapped(&tmp).err().expect("a zero-length sweep opened");
        assert!(
            matches!(err, VeloError::EndBeforeStart { line: 6, frame: 5 }),
            "{err:?}"
        );
        assert!(
            err.to_string()
                .contains("the sweep of frame 5 (line 6 of timestamps_start.txt"),
            "{err}"
        );
    }

    /// A lidar whose timestamp files end in blank lines declares those frames
    /// and has no sweep for them: absent, like any other gap.
    #[test]
    fn v19_trailing_blank_lines_are_frames_absent_in_the_source() {
        use crate::testing::{write_velo_fixture_gapped_at, FIXTURE_DATE, FIXTURE_DRIVE};
        let tmp = tempfile::TempDir::new().unwrap();
        write_velo_fixture_gapped_at(
            tmp.path(),
            FIXTURE_DATE,
            FIXTURE_DRIVE,
            8,
            &[6, 7],
            GAP_PERIOD,
        )
        .unwrap();
        let d = open_gapped(&tmp).expect("trailing blank lines are KITTI's layout");
        assert_eq!((d.len(), d.frame_slots()), (6, 8));
        assert_eq!(d.absent_in_source(), vec![6, 7]);
        assert_eq!(d.gaps()[0].after, None);
        let (seen, totals) = replay(&d, &unpaced());
        assert_eq!((totals.admitted, totals.missing), (6, 2));
        assert_eq!(seen[6].1, ABSENT_IN_SOURCE);
        assert_eq!(seen[7].1, ABSENT_IN_SOURCE);
    }
}
