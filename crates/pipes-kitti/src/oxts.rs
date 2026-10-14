//! `OxtsDriver`: the paced replayer for KITTI `oxts`, the GPS/IMU unit, and
//! the Arrow row builder beside it. The third sensor, and the one that says
//! what the vehicle itself was doing.
//!
//! KITTI's synced drives carry one OXTS row per frame, `oxts/data/{i:010}.txt`,
//! stamped in `oxts/timestamps.txt` the way the cameras are: on drive 0005
//! the row arrives about 17 ms after the lidar's forward-facing trigger, 35 ms
//! before the sweep ends. A row is 30 numbers (`oxts/dataformat.txt`):
//! position, attitude, velocities in the earth frame and in the vehicle's,
//! accelerations and angular rates in both, two accuracies and five status
//! words. The whole row travels, as 30 `f64`, so a reader can take any field;
//! the fusion reads three ([`EgoRates`]): forward and leftward speed and the
//! yaw rate in the vehicle frame, which is what carrying a track through the
//! vehicle's own motion needs.
//!
//! Same shape as [`crate::cam0::Cam0Driver`]: absolute deadlines from its
//! own timestamps (`Tov::Time`), every frame slot admitted exactly once as a
//! sample or a `Missing`, and a frame the source never had replayed as
//! `absent_in_source` at its own place in the schedule ([`crate::layout`]).

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use arrow::array::{
    Array, ArrayRef, AsArray, FixedSizeListArray, Float64Array, RecordBatch, StringArray,
};
use arrow::datatypes::{DataType, Field, FieldRef, Float64Type, Schema};
use arrow::error::ArrowError;
use pipes_core::clock::{now, sleep_until, ClockModel, HostTime, SensorTime, Tov};
use pipes_core::sample::{Sample, StreamId};

use crate::driver::{median_period_ns, DriverEvent, RunTotals};
use crate::layout::{instants, replay_absent, FrameIndex, Gap, LayoutError, Slot};
use crate::timestamps::{parse_timestamp_lines, TimestampError};

/// The one directory a drive's GPS/IMU stream lives in.
pub const OXTS_DIR: &str = "oxts";

/// Values in one row, in `oxts/dataformat.txt`'s order.
pub const OXTS_FIELDS: usize = 30;

/// [`OXTS_FIELDS`] as the `i32` an Arrow `FixedSizeList` wants.
pub const OXTS_VALUES: i32 = OXTS_FIELDS as i32;

/// Value of the `oxts_format` column: the 30 fields of KITTI's `dataformat.txt`,
/// in its order, as `f64`.
pub const OXTS_FORMAT: &str = "oxts_kitti30_f64";

/// Field 8: forward velocity, m/s, in the vehicle frame (`vf`).
pub const FIELD_VF: usize = 8;
/// Field 9: leftward velocity, m/s, in the vehicle frame (`vl`).
pub const FIELD_VL: usize = 9;
/// Field 22: angular rate about the vehicle's up axis, rad/s (`wu`): the yaw
/// rate in the vehicle frame. Field 19 (`wz`) is the same rate about the
/// earth's vertical; on a drive this close to level they differ by the
/// cosine of the pitch.
pub const FIELD_WU: usize = 22;

/// The three fields the fusion reads out of a row: what the vehicle itself
/// was doing.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct EgoRates {
    /// Forward velocity, m/s ([`FIELD_VF`]).
    pub vf: f64,
    /// Leftward velocity, m/s ([`FIELD_VL`]).
    pub vl: f64,
    /// Yaw rate, rad/s, positive turning left ([`FIELD_WU`]).
    pub yaw_rate: f64,
}

impl EgoRates {
    /// The three fields of one row.
    pub fn of_row(row: &[f64; OXTS_FIELDS]) -> EgoRates {
        EgoRates {
            vf: row[FIELD_VF],
            vl: row[FIELD_VL],
            yaw_rate: row[FIELD_WU],
        }
    }

    /// Ground speed, m/s: the forward and leftward velocities combined.
    pub fn speed(&self) -> f64 {
        self.vf.hypot(self.vl)
    }
}

/// `<root>/<date>/<drive>/oxts`.
pub fn oxts_dir(root: &Path, date: &str, drive: &str) -> PathBuf {
    root.join(date).join(drive).join(OXTS_DIR)
}

/// Whether this drive carries a GPS/IMU stream at all: `oxts/` either exists
/// or it does not, on the terms [`crate::velo::has_velodyne`] decides for
/// the lidar.
pub fn has_oxts(root: &Path, date: &str, drive: &str) -> bool {
    oxts_dir(root, date, drive).is_dir()
}

/// What is wrong with one `oxts/data/*.txt` row.
#[derive(Debug, PartialEq)]
pub enum RowError {
    /// Not [`OXTS_FIELDS`] values.
    FieldCount {
        /// Values found.
        got: usize,
    },
    /// A value that is not a number.
    NotANumber {
        /// Its 0-based index in the row.
        field: usize,
        /// The text.
        text: String,
    },
    /// A value that is NaN or infinite, which no field of a measured row is.
    NonFinite {
        /// Its 0-based index in the row.
        field: usize,
    },
}

impl std::fmt::Display for RowError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RowError::FieldCount { got } => {
                write!(
                    f,
                    "{got} values, not the {OXTS_FIELDS} of oxts/dataformat.txt"
                )
            }
            RowError::NotANumber { field, text } => {
                write!(f, "field {field}: `{text}` is not a number")
            }
            RowError::NonFinite { field } => write!(f, "field {field} is not finite"),
        }
    }
}

impl std::error::Error for RowError {}

/// Errors from opening a drive's GPS/IMU stream or replaying a row.
#[derive(Debug)]
pub enum OxtsError {
    /// A path could not be read.
    Io {
        /// The path that failed.
        path: PathBuf,
        /// Underlying I/O error.
        source: std::io::Error,
    },
    /// The drive exists but has no `oxts/` directory.
    NoOxts {
        /// The drive directory that was looked in.
        drive_dir: PathBuf,
        /// The directory that was looked for.
        dir: PathBuf,
        /// Directories that do exist in the drive, sorted.
        siblings: Vec<String>,
    },
    /// `timestamps.txt` could not be parsed.
    Timestamps {
        /// The file.
        path: PathBuf,
        /// What was wrong with it.
        source: TimestampError,
    },
    /// The rows and the lines of `timestamps.txt` are not KITTI's layout.
    Layout(LayoutError),
    /// One row could not be read as 30 numbers.
    Row {
        /// The file.
        path: PathBuf,
        /// What was wrong with it.
        source: RowError,
    },
    /// Arrow rejected the batch.
    Arrow(ArrowError),
}

impl std::fmt::Display for OxtsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OxtsError::Io { path, source } => write!(f, "{}: {source}", path.display()),
            OxtsError::NoOxts {
                drive_dir,
                dir,
                siblings,
            } => {
                write!(
                    f,
                    "the drive {} has no {OXTS_DIR}/ directory ({} was looked for), so it carries no GPS/IMU stream",
                    drive_dir.display(),
                    dir.display()
                )?;
                if siblings.is_empty() {
                    f.write_str("; it has no directories at all")
                } else {
                    write!(f, "; it has: {}", siblings.join(", "))
                }
            }
            OxtsError::Timestamps { path, source } => write!(f, "{}: {source}", path.display()),
            OxtsError::Layout(e) => write!(f, "{e}"),
            OxtsError::Row { path, source } => write!(f, "{}: {source}", path.display()),
            OxtsError::Arrow(e) => write!(f, "arrow: {e}"),
        }
    }
}

impl std::error::Error for OxtsError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            OxtsError::Io { source, .. } => Some(source),
            OxtsError::Timestamps { source, .. } => Some(source),
            OxtsError::Row { source, .. } => Some(source),
            OxtsError::Arrow(e) => Some(e),
            // `Display` already prints these whole.
            OxtsError::NoOxts { .. } | OxtsError::Layout(_) => None,
        }
    }
}

impl From<LayoutError> for OxtsError {
    fn from(e: LayoutError) -> Self {
        OxtsError::Layout(e)
    }
}

impl From<ArrowError> for OxtsError {
    fn from(e: ArrowError) -> Self {
        OxtsError::Arrow(e)
    }
}

/// Parses one row: [`OXTS_FIELDS`] finite numbers separated by whitespace.
pub fn parse_row(text: &str) -> Result<[f64; OXTS_FIELDS], RowError> {
    let mut row = [0.0f64; OXTS_FIELDS];
    let mut n = 0usize;
    for (i, t) in text.split_whitespace().enumerate() {
        if i < OXTS_FIELDS {
            let v: f64 = t.parse().map_err(|_| RowError::NotANumber {
                field: i,
                text: t.to_string(),
            })?;
            if !v.is_finite() {
                return Err(RowError::NonFinite { field: i });
            }
            row[i] = v;
        }
        n = i + 1;
    }
    if n != OXTS_FIELDS {
        return Err(RowError::FieldCount { got: n });
    }
    Ok(row)
}

/// Arrow field of one value of a row.
fn oxts_value_field() -> FieldRef {
    Arc::new(Field::new("item", DataType::Float64, false))
}

/// `oxts_format Utf8, fields FixedSizeList<Float64, 30>`, one row per batch.
///
/// The 30 values in one buffer, so the batch has one `storage_id` the way
/// every other payload in this workspace does, and a reader takes a field
/// by its index in `dataformat.txt` rather than by a name this crate chose.
pub fn oxts_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("oxts_format", DataType::Utf8, false),
        Field::new(
            "fields",
            DataType::FixedSizeList(oxts_value_field(), OXTS_VALUES),
            false,
        ),
    ]))
}

/// Wraps one row as a 1-row batch. Returns the batch and the `storage_id`,
/// the address of the buffer the 30 values live in.
pub fn build_oxts_batch(
    row: [f64; OXTS_FIELDS],
    schema: &Arc<Schema>,
) -> Result<(RecordBatch, usize), ArrowError> {
    let values = Float64Array::from(row.to_vec());
    let storage_id = values.values().as_ptr() as usize;
    let fields = FixedSizeListArray::try_new(
        oxts_value_field(),
        OXTS_VALUES,
        Arc::new(values) as ArrayRef,
        None,
    )?;
    let columns: Vec<ArrayRef> = vec![
        Arc::new(StringArray::from(vec![OXTS_FORMAT])),
        Arc::new(fields),
    ];
    Ok((
        RecordBatch::try_new(Arc::clone(schema), columns)?,
        storage_id,
    ))
}

/// The 30 values of an OXTS batch, shared with the buffer the driver built,
/// or `None` for a batch that is not one. Checks `oxts_format` rather than
/// trusting the stream id, on the terms every other reader here does.
pub fn oxts_row(batch: &RecordBatch) -> Option<&[f64; OXTS_FIELDS]> {
    let fmt = batch
        .column_by_name("oxts_format")?
        .as_string_opt::<i32>()?;
    if fmt.is_empty() || fmt.value(0) != OXTS_FORMAT {
        return None;
    }
    let values = batch
        .column_by_name("fields")?
        .as_fixed_size_list_opt()?
        .values()
        .as_primitive_opt::<Float64Type>()?
        .values();
    values.as_ref().try_into().ok()
}

/// The vehicle's own motion, read out of an OXTS batch.
pub fn read_ego(batch: &RecordBatch) -> Option<EgoRates> {
    oxts_row(batch).map(EgoRates::of_row)
}

/// Re-derived storage id of an OXTS batch, for the proof that a consumer
/// read the driver's buffer.
pub fn oxts_storage_id(batch: &RecordBatch) -> Option<usize> {
    oxts_row(batch).map(|r| r.as_ptr() as usize)
}

/// The GPS/IMU replayer for one drive.
pub struct OxtsDriver {
    /// KITTI data root holding `<date>/<drive>/oxts`.
    pub root: PathBuf,
    /// Capture date directory, e.g. `2011_09_26`.
    pub date: String,
    /// Drive directory, e.g. `2011_09_26_drive_0005_sync`.
    pub drive: String,
    /// One sensor timestamp per row on disk, parsed once at open: entry `k`
    /// is the `k`-th non-blank line of `timestamps.txt`, the instant of the
    /// `k`-th row, whose frame is `index.frame(k)`.
    pub timestamps: Vec<SensorTime>,
    /// Which frames have a row ([`crate::layout`]).
    pub index: FrameIndex,
}

impl OxtsDriver {
    /// Opens `<root>/<date>/<drive>/oxts` the way KITTI lays a synced drive
    /// out ([`crate::layout`]): each `data/*.txt` named by its frame, line
    /// `i + 1` of `timestamps.txt` for frame `i`, blank where the frame has
    /// no row. Every other layout is refused, naming the real line.
    pub fn open(root: &Path, date: &str, drive: &str) -> Result<Self, OxtsError> {
        let dir = oxts_dir(root, date, drive);
        if !dir.is_dir() {
            let drive_dir = root.join(date).join(drive);
            return Err(OxtsError::NoOxts {
                siblings: crate::cam0::sibling_dirs(&drive_dir),
                drive_dir,
                dir,
            });
        }
        let ts_path = dir.join("timestamps.txt");
        let lines = parse_timestamp_lines(&ts_path).map_err(|source| match source {
            TimestampError::Io(source) => OxtsError::Io {
                path: ts_path.clone(),
                source,
            },
            other => OxtsError::Timestamps {
                path: ts_path.clone(),
                source: other,
            },
        })?;
        let index = FrameIndex::read(&dir.join("data"), "txt", &ts_path, &lines)?;
        Ok(OxtsDriver {
            root: root.to_path_buf(),
            date: date.to_string(),
            drive: drive.to_string(),
            timestamps: instants(&lines),
            index,
        })
    }

    /// Extends the stream to `n` frame slots, the drive's frame count, when
    /// that is more than `timestamps.txt` declares. Never shrinks. See
    /// [`crate::cam0::Cam0Driver::with_frame_slots`].
    pub fn with_frame_slots(mut self, n: usize) -> Self {
        self.index = self.index.with_slots(n);
        self
    }

    /// Rows on disk.
    pub fn len(&self) -> usize {
        self.timestamps.len()
    }

    /// Whether the drive has no rows.
    pub fn is_empty(&self) -> bool {
        self.timestamps.is_empty()
    }

    /// Frame slots on the stream: every row, every frame whose line is
    /// blank, and any frames [`OxtsDriver::with_frame_slots`] added.
    pub fn frame_slots(&self) -> usize {
        self.index.slots()
    }

    /// The frames, ascending, that have no row in the source.
    pub fn absent_in_source(&self) -> Vec<u64> {
        self.index.absent()
    }

    /// The runs of consecutive frames absent in the source, with the real
    /// frames either side of each and the time between their instants.
    pub fn gaps(&self) -> Vec<Gap> {
        self.index.gaps(&self.timestamps, &self.timestamps)
    }

    fn dir(&self) -> PathBuf {
        oxts_dir(&self.root, &self.date, &self.drive)
    }

    /// `dir/data/{frame:010}.txt`.
    pub fn frame_path(&self, frame: u64) -> PathBuf {
        self.dir().join("data").join(format!("{frame:010}.txt"))
    }

    /// Reads and parses the row of `frame`.
    pub fn read_row(&self, frame: u64) -> Result<[f64; OXTS_FIELDS], OxtsError> {
        let path = self.frame_path(frame);
        let text = std::fs::read_to_string(&path).map_err(|source| OxtsError::Io {
            path: path.clone(),
            source,
        })?;
        parse_row(&text).map_err(|source| OxtsError::Row { path, source })
    }

    /// The instant at which the `i`-th row counts as skipped: the next row's
    /// deadline, or one period past the last one's own.
    fn skip_at(&self, clock: &ClockModel, i: usize, period_ns: i64) -> Option<HostTime> {
        match self.timestamps.get(i + 1) {
            Some(&next) => clock.due(Tov::Time(next)),
            None => {
                let last = self.timestamps.get(i)?;
                clock.due(Tov::Time(SensorTime(last.0 + period_ns)))
            }
        }
    }

    /// Replays every frame slot at the clock's rate, calling `admit` exactly
    /// once per slot (a `Sample` or a `Missing`), whose `seq` is the frame's
    /// number. Same shape as [`crate::cam0::Cam0Driver::run`].
    pub fn run(
        &self,
        clock: &ClockModel,
        spin_window: Duration,
        schema: &Arc<Schema>,
        admit: &mut dyn FnMut(DriverEvent),
    ) -> RunTotals {
        let t_start = now();
        let period_ns = median_period_ns(&self.timestamps);
        let mut admitted = 0u64;
        let mut missing = 0u64;
        for slot in self.index.slot_list() {
            let (frame, i) = match slot {
                Slot::Absent { frame } => {
                    replay_absent(
                        &self.index,
                        &self.timestamps,
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
            let Some(&ts) = self.timestamps.get(i) else {
                admit(DriverEvent::Missing {
                    seq,
                    tov: Tov::None,
                    due: None,
                    reason: "read_error",
                });
                missing += 1;
                continue;
            };
            let tov = Tov::Time(ts);
            let due = clock.due(tov);
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
            let t_read = now();
            let row = match self.read_row(frame) {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("oxts: {e}");
                    miss("read_error", admit);
                    continue;
                }
            };
            // The parse: this driver's analogue of the camera's decode.
            let decode_ns = now() - t_read;
            let (batch, storage_id) = match build_oxts_batch(row, schema) {
                Ok(x) => x,
                Err(e) => {
                    eprintln!("oxts: frame {seq}: {e}");
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
                stream: StreamId::OXTS,
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
    use crate::testing::{FixtureDrive, FIXTURE_DATE, FIXTURE_DRIVE};

    /// Drive 0005's first row, as the file spells it.
    const ROW: &str = "49.011212804408 8.4228850417969 112.83492279053 0.022447 1e-05 -1.2219096732051 -3.3256321640686 1.1384311814592 3.5147680214713 0.037625160413037 -0.03878884255623 -0.29437452763793 0.037166856911681 9.9957015129717 -0.30581030960531 -0.19635662515203 9.9942128010936 -0.017332142869546 0.024792163815438 0.14511808479348 -0.017498934149631 0.021393359392165 0.14563031426063 0.49229361157748 0.068883960397178 4 10 4 4 0";

    #[test]
    fn a_row_is_thirty_numbers_and_the_fusion_reads_three() {
        let row = parse_row(ROW).unwrap();
        assert_eq!(row[0], 49.011212804408);
        assert_eq!(row[29], 0.0);
        let ego = EgoRates::of_row(&row);
        assert_eq!(ego.vf, 3.5147680214713);
        assert_eq!(ego.vl, 0.037625160413037);
        assert_eq!(ego.yaw_rate, 0.14563031426063);
        assert!((ego.speed() - 3.5149694).abs() < 1e-6);
        // Trailing whitespace and a newline are the file's own.
        assert_eq!(parse_row(&format!("{ROW}\n")).unwrap(), row);
        assert_eq!(
            parse_row("1 2 3").unwrap_err(),
            RowError::FieldCount { got: 3 }
        );
        assert_eq!(
            parse_row(&format!("{ROW} 7")).unwrap_err(),
            RowError::FieldCount { got: 31 }
        );
        assert_eq!(
            parse_row(&ROW.replacen("8.4228850417969", "eight", 1)).unwrap_err(),
            RowError::NotANumber {
                field: 1,
                text: "eight".to_string()
            }
        );
        assert_eq!(
            parse_row(&ROW.replacen("112.83492279053", "nan", 1)).unwrap_err(),
            RowError::NonFinite { field: 2 }
        );
    }

    #[test]
    fn a_batch_carries_the_row_in_one_buffer_and_reads_back() {
        let row = parse_row(ROW).unwrap();
        let (b, sid) = build_oxts_batch(row, &oxts_schema()).unwrap();
        assert_eq!(b.num_rows(), 1);
        assert_eq!(oxts_row(&b), Some(&row));
        assert_eq!(oxts_storage_id(&b), Some(sid));
        assert_eq!(read_ego(&b), Some(EgoRates::of_row(&row)));
        // A batch that is not a row reads as none, whatever its stream.
        let other = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new(
                "oxts_format",
                DataType::Utf8,
                false,
            )])),
            vec![Arc::new(StringArray::from(vec!["something_else"])) as ArrayRef],
        )
        .unwrap();
        assert_eq!(oxts_row(&other), None);
    }

    #[test]
    fn the_driver_opens_a_fixture_and_replays_every_row() {
        let fx = FixtureDrive::with_velodyne(6, 6, 100_000_000).unwrap();
        fx.write_oxts(&[], |i| (5.0, 0.0, i as f64 / 10.0)).unwrap();
        assert!(has_oxts(fx.root(), FIXTURE_DATE, FIXTURE_DRIVE));
        let d = OxtsDriver::open(fx.root(), FIXTURE_DATE, FIXTURE_DRIVE).unwrap();
        assert_eq!((d.len(), d.frame_slots()), (6, 6));
        assert!(d.absent_in_source().is_empty());
        let row = d.read_row(3).unwrap();
        assert_eq!(EgoRates::of_row(&row).yaw_rate, 0.3);
        // Unpaced: every slot admitted, in frame order, with its row.
        let clock = ClockModel::start_now(d.timestamps[0], f64::INFINITY);
        let schema = oxts_schema();
        let mut seen = Vec::new();
        let totals = d.run(&clock, Duration::ZERO, &schema, &mut |ev| match ev {
            DriverEvent::Sample(s) => {
                assert_eq!(s.stream, StreamId::OXTS);
                let ego = read_ego(&s.payload).unwrap();
                assert_eq!(oxts_storage_id(&s.payload), Some(s.storage_id));
                seen.push((s.seq, ego.yaw_rate));
            }
            DriverEvent::Missing { seq, reason, .. } => panic!("frame {seq} missing: {reason}"),
        });
        assert_eq!((totals.admitted, totals.missing), (6, 0));
        assert_eq!(
            seen,
            (0..6).map(|i| (i, i as f64 / 10.0)).collect::<Vec<_>>()
        );
    }

    #[test]
    fn a_frame_the_source_lacks_is_a_gap_not_an_error() {
        let fx = FixtureDrive::with_velodyne(6, 6, 100_000_000).unwrap();
        fx.write_oxts(&[2, 3], |_| (1.0, 0.0, 0.0)).unwrap();
        let d = OxtsDriver::open(fx.root(), FIXTURE_DATE, FIXTURE_DRIVE).unwrap();
        assert_eq!((d.len(), d.frame_slots()), (4, 6));
        assert_eq!(d.absent_in_source(), [2, 3]);
        assert_eq!(d.gaps().len(), 1);
        let d = d.with_frame_slots(8);
        assert_eq!(d.frame_slots(), 8);
        assert_eq!(d.absent_in_source(), [2, 3, 6, 7]);
        let clock = ClockModel::start_now(d.timestamps[0], f64::INFINITY);
        let mut missing = Vec::new();
        let totals = d.run(&clock, Duration::ZERO, &oxts_schema(), &mut |ev| {
            if let DriverEvent::Missing { seq, reason, .. } = ev {
                missing.push((seq, reason));
            }
        });
        assert_eq!((totals.admitted, totals.missing), (4, 4));
        assert_eq!(
            missing,
            [2, 3, 6, 7].map(|f| (f, crate::layout::ABSENT_IN_SOURCE))
        );
    }

    #[test]
    fn a_drive_without_oxts_says_so() {
        let fx = FixtureDrive::new(3).unwrap();
        assert!(!has_oxts(fx.root(), FIXTURE_DATE, FIXTURE_DRIVE));
        let e = OxtsDriver::open(fx.root(), FIXTURE_DATE, FIXTURE_DRIVE)
            .err()
            .unwrap();
        let text = e.to_string();
        assert!(text.contains("no oxts/ directory"), "{text}");
        assert!(text.contains("image_02"), "{text}");
    }
}
