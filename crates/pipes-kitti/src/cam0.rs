//! `Cam0Driver`: the paced PNG replayer for KITTI `image_02` (design §5.2).
//! Deadlines are absolute, from the `ClockModel`, never from the previous
//! arrival, so a slow decode or a slow admit never shifts a later `due`.
//! Frame `i` is decoded in the gap before `due(i)` ("decode one ahead"), the
//! driver sleeps to `due - spin_window` then spins, and a frame whose next
//! deadline has already passed is reported `Missing` without being decoded.
//!
//! The PNGs are read the way the lidar's sweeps are ([`crate::layout`], one
//! piece of code for both): each is named by its frame, line `i + 1` of
//! `timestamps.txt` is frame `i`, and a frame whose line is blank and whose
//! PNG is absent is a gap in the source, replayed as `Missing` with reason
//! `absent_in_source` and no time of validity rather than refused.

use std::collections::BTreeSet;
use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use arrow::datatypes::Schema;
use pipes_core::clock::{now, sleep_until, ClockModel, HostTime, SensorTime, Tov};
use pipes_core::sample::{Sample, StreamId};

use crate::driver::data_names;
use crate::frame::{build_cam0_batch, FrameError};
use crate::layout::{instants, replay_absent, FrameIndex, Gap, LayoutError, Slot};
use crate::timestamps::{parse_timestamp_lines, TimestampError};

// `DEFAULT_PERIOD_NS`, `median_period_ns`, `DriverEvent` and `RunTotals` moved
// to [`crate::driver`] when the velodyne driver arrived: they describe *a*
// driver, not the camera one. Re-exported here so every
// `pipes_kitti::cam0::…` path still resolves.
pub use crate::driver::{median_period_ns, DriverEvent, RunTotals, DEFAULT_PERIOD_NS};

/// How many sibling directory names an error message lists before it stops.
const SIBLINGS_SHOWN: usize = 6;

/// One decoded frame, before it becomes an Arrow batch.
pub struct Decoded {
    /// Interleaved RGB8, `width * height * 3` bytes. This buffer becomes the
    /// Arrow pixel buffer without a copy.
    pub pixels: Vec<u8>,
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
    /// Wall cost of the decode in ns, measured on the decoding thread.
    pub decode_ns: i64,
}

/// The cam0 replayer for one drive.
pub struct Cam0Driver {
    /// KITTI data root holding `<date>/<drive>/image_02`.
    pub root: PathBuf,
    /// Capture date directory, e.g. `2011_09_26`.
    pub date: String,
    /// Drive directory, e.g. `2011_09_26_drive_0005_sync`.
    pub drive: String,
    /// One sensor timestamp per PNG on disk, parsed once at open: entry `k`
    /// is the `k`-th non-blank line of `timestamps.txt`, the instant of the
    /// `k`-th PNG, whose frame is `index.frame(k)` (and whose line is that
    /// frame plus one).
    pub timestamps: Vec<SensorTime>,
    /// Which frames have a PNG ([`crate::layout`]): every frame on a drive
    /// with no gap in its camera's source.
    pub index: FrameIndex,
}

/// Which level of `<root>/<date>/<drive>/image_02` was the first not to exist.
///
/// The four cases used to share one byte-identical message, so a typo in
/// `--drive`, a wrong date prefix on it and a drive that really is missing its camera
/// directory were indistinguishable — and none of them told the user what
/// *does* exist.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MissingLevel {
    /// The data root itself is not a directory (`--kitti-root` is wrong).
    Root,
    /// The root exists but holds no such capture date (the drive name's prefix is wrong).
    Date,
    /// The date exists but holds no such drive (`--drive` is wrong).
    Drive,
    /// The drive exists but has no `image_02/`, so it carries no cam0 stream.
    Image02,
}

/// What [`DriverError::DriveMissing`] carries, boxed out of the enum.
///
/// Seven fields inline would make `DriveMissing` the widest variant by far,
/// and every `Result<_, DriverError>` is as wide as its widest variant —
/// including the one [`Cam0Driver::decode`] returns once per frame. This is
/// the coldest path in the module, so it pays the indirection.
#[derive(Debug)]
pub struct DriveMissing {
    /// `root/date/drive/image_02`, as it was looked for.
    pub dir: PathBuf,
    /// The data root it was resolved against (`--kitti-root`).
    pub root: PathBuf,
    /// The capture date, as given (the first 10 characters of `--drive`).
    pub date: String,
    /// `--drive`, as given.
    pub drive: String,
    /// Working directory, when `root` is relative and so ambiguous alone.
    pub cwd: Option<PathBuf>,
    /// The first path component that does not exist.
    pub level: MissingLevel,
    /// Directory names that *do* exist beside the missing one, sorted; dates
    /// for [`MissingLevel::Date`], drives for [`MissingLevel::Drive`], empty
    /// otherwise.
    pub siblings: Vec<String>,
}

/// Errors from opening a drive or decoding a frame.
#[derive(Debug)]
pub enum DriverError {
    /// A path could not be read.
    Io {
        /// The path that failed.
        path: PathBuf,
        /// Underlying I/O error.
        source: std::io::Error,
    },
    /// The drive directory does not exist. Split out from [`DriverError::Io`]
    /// because, now that the data root defaults to a relative path, this is the
    /// *common* first-run failure for anyone who has not placed the dataset
    /// where the README's "Setup" says: it needs to name the directory it looked in and
    /// the override, not report `timestamps.txt: not found`.
    ///
    /// [`DriveMissing::level`] and [`DriveMissing::siblings`] say *which* of
    /// the four path components is wrong and what exists beside it, so the
    /// message ends at a next step rather than at a dead end.
    DriveMissing(Box<DriveMissing>),
    /// `timestamps.txt` could not be parsed.
    Timestamps(TimestampError),
    /// The PNGs and the lines of `timestamps.txt` are not KITTI's layout: a
    /// file not named by its frame, a line with a time and no PNG, a PNG
    /// whose line is blank or past the last, or times that do not increase.
    /// The lidar's own checks, from the same code ([`LayoutError`]).
    ///
    /// Names, not counts, are what is checked first, and that is load-bearing:
    /// a directory with the right *count* and the wrong *names* once passed
    /// `open`, the run exited 0 with every invariant OK, and the frames that
    /// were never there became `Missing{decode_error}` -- absorbed by the
    /// `admitted + missing == n_frames` identity.
    Layout(LayoutError),
    /// One frame failed to decode.
    Decode {
        /// Frame number.
        seq: u64,
        /// The PNG that failed.
        path: PathBuf,
        /// Underlying decoder error.
        source: image::ImageError,
    },
    /// The decoded pixels could not be wrapped as an Arrow batch.
    Frame(FrameError),
}

impl std::fmt::Display for DriverError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DriverError::Io { path, source } => write!(f, "{}: {source}", path.display()),
            DriverError::DriveMissing(m) => {
                let DriveMissing {
                    dir,
                    root,
                    date,
                    drive,
                    cwd,
                    level,
                    siblings,
                } = m.as_ref();
                match level {
                    MissingLevel::Root => {
                        write!(f, "no KITTI data root at {}", root.display())?;
                        if let Some(cwd) = cwd {
                            write!(f, " (relative to {})", cwd.display())?;
                        }
                        write!(f, "; it does not hold <date>/<drive>/image_02, so ")?;
                        write!(f, "{} could not be looked for. ", dir.display())?;
                    }
                    MissingLevel::Date => {
                        write!(f, "no capture date `{date}` in the data root ")?;
                        write!(f, "{}", root.display())?;
                        if let Some(cwd) = cwd {
                            write!(f, " (relative to {})", cwd.display())?;
                        }
                        f.write_str("; the first 10 characters of --drive name a directory such as 2011_09_26. ")?;
                        write_siblings(f, "dates present", siblings)?;
                    }
                    MissingLevel::Drive => {
                        write!(f, "no drive `{drive}` under {}", root.join(date).display())?;
                        if let Some(cwd) = cwd {
                            write!(f, " (relative to {})", cwd.display())?;
                        }
                        f.write_str("; the capture date exists, --drive does not. ")?;
                        write_siblings(f, "drives present", siblings)?;
                    }
                    MissingLevel::Image02 => {
                        write!(
                            f,
                            "the drive {} exists but has ",
                            root.join(date).join(drive).display()
                        )?;
                        f.write_str("no image_02/ directory, so it carries no cam0 stream. ")?;
                    }
                }
                f.write_str(
                    "Run `cargo run --release -- drives` to list every drive under the data root. ",
                )?;
                f.write_str("Pass --kitti-root <dir>, or set PIPES_KITTI_ROOT once; ")?;
                f.write_str("the default expects the dataset beside the checkout")
            }
            DriverError::Timestamps(e) => write!(f, "{e}"),
            // The wrapped text is the whole message: it names the directory,
            // the file and what helps.
            DriverError::Layout(e) => write!(f, "{e}"),
            DriverError::Decode { seq, path, source } => {
                write!(f, "frame {seq} ({}): {source}", path.display())
            }
            DriverError::Frame(e) => write!(f, "frame batch: {e}"),
        }
    }
}

impl std::error::Error for DriverError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            DriverError::Io { source, .. } => Some(source),
            DriverError::Timestamps(e) => Some(e),
            // Not `Some(e)` for `Layout`: `Display` already prints it whole,
            // and `main` would print it a second time as its cause.
            DriverError::DriveMissing(_) | DriverError::Layout(_) => None,
            DriverError::Decode { source, .. } => Some(source),
            DriverError::Frame(e) => Some(e),
        }
    }
}

impl From<FrameError> for DriverError {
    fn from(e: FrameError) -> Self {
        DriverError::Frame(e)
    }
}

impl From<LayoutError> for DriverError {
    fn from(e: LayoutError) -> Self {
        DriverError::Layout(e)
    }
}

impl Cam0Driver {
    /// `dir = root/date/drive/image_02`: parses `dir/timestamps.txt` line by
    /// line, blank lines kept, then reads `dir/data` the way KITTI lays a
    /// synced drive out ([`crate::layout`]): each PNG named by its frame,
    /// line `i + 1` for frame `i`, and a frame whose line is blank and whose
    /// PNG is absent a gap in the source that the replay records. Every other
    /// layout is refused ([`DriverError::Layout`]), naming the real line --
    /// by *name* first, not by count.
    ///
    /// The name check is the whole point. Counting accepted a directory whose
    /// PNGs were named anything at all as long as there were enough of them:
    /// `open` returned `Ok`, the run exited 0, every invariant printed OK, and
    /// the frames that did not exist came out as `Missing{decode_error}`,
    /// which `admitted + missing == n_frames` swallows without complaint.
    ///
    /// One `read_dir` of `data`: no per-frame `stat`.
    pub fn open(root: &Path, date: &str, drive: &str) -> Result<Self, DriverError> {
        let dir = root.join(date).join(drive).join("image_02");
        if !dir.is_dir() {
            return Err(drive_missing(root, date, drive, dir));
        }
        let ts_path = dir.join("timestamps.txt");
        let lines = parse_timestamp_lines(&ts_path).map_err(|e| match e {
            TimestampError::Io(source) => DriverError::Io {
                path: ts_path.clone(),
                source,
            },
            other => DriverError::Timestamps(other),
        })?;
        let index = FrameIndex::read(&dir.join("data"), "png", &ts_path, &lines)?;
        Ok(Cam0Driver {
            root: root.to_path_buf(),
            date: date.to_string(),
            drive: drive.to_string(),
            timestamps: instants(&lines),
            index,
        })
    }

    /// Extends the stream to `n` frame slots, the drive's frame count, when
    /// that is more than `timestamps.txt` declares.
    ///
    /// KITTI numbers every sensor of a synced drive by one frame index, so a
    /// camera whose timestamp file stops before another sensor's lacks the
    /// frames after it: absent in the source like any other, replayed as
    /// missing inputs, and invisible to the camera alone. Never shrinks.
    pub fn with_frame_slots(mut self, n: usize) -> Self {
        self.index = self.index.with_slots(n);
        self
    }

    /// Number of frames on disk: PNGs, and non-blank lines in
    /// `timestamps.txt`.
    pub fn len(&self) -> usize {
        self.timestamps.len()
    }

    /// Whether the drive has no frames.
    pub fn is_empty(&self) -> bool {
        self.timestamps.is_empty()
    }

    /// Frame slots on the camera stream: the lines of `timestamps.txt`, every
    /// PNG plus every frame whose line is blank, and any frames past its last
    /// line that [`Cam0Driver::with_frame_slots`] added. Equal to
    /// [`Cam0Driver::len`] on a drive with no gap in its camera's source.
    pub fn frame_slots(&self) -> usize {
        self.index.slots()
    }

    /// The frames, ascending, that have no PNG in the source.
    pub fn absent_in_source(&self) -> Vec<u64> {
        self.index.absent()
    }

    /// The runs of consecutive frames absent in the source, with the real
    /// frames either side of each and the time between their instants.
    pub fn gaps(&self) -> Vec<Gap> {
        self.index.gaps(&self.timestamps, &self.timestamps)
    }

    /// `dir/data/{frame:010}.png` of the `k`-th PNG on disk -- for `k = 0`
    /// the drive's first frame, whatever its number. `None` past the last.
    pub fn file_path(&self, k: usize) -> Option<PathBuf> {
        self.index
            .frame(k)
            .map(|f| self.frame_path(usize::try_from(f).unwrap_or(usize::MAX)))
    }

    fn dir(&self) -> PathBuf {
        self.root
            .join(&self.date)
            .join(&self.drive)
            .join("image_02")
    }

    /// `dir/data/{i:010}.png`.
    pub fn frame_path(&self, i: usize) -> PathBuf {
        self.dir().join("data").join(frame_file_name(i))
    }

    /// Decodes frame `i` -- by its number, which is its file name -- via
    /// [`decode_png`].
    pub fn decode(&self, i: usize) -> Result<Decoded, DriverError> {
        let path = self.frame_path(i);
        decode_png(&path).map_err(|source| DriverError::Decode {
            seq: i as u64,
            path,
            source,
        })
    }

    /// The instant at which the `i`-th PNG counts as skipped: the next PNG's
    /// deadline, or one period past the last one's own deadline. The next
    /// that EXISTS, so across a gap in the source the frame before it is
    /// still handed over late rather than dropped.
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
    /// number. `admit` may take a while (a blocking edge in M4): the next
    /// frame is then judged against its own absolute deadline and skipped if
    /// that has already passed. A frame absent in the source is reported as
    /// the lidar reports one ([`crate::layout::replay_absent`]).
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
            // One line per PNG (`open` checks it), so this cannot be `None`;
            // if it ever were, the slot is still reported, not skipped.
            let Some(&ts) = self.timestamps.get(i) else {
                admit(DriverEvent::Missing {
                    seq,
                    tov: Tov::None,
                    due: None,
                    reason: "decode_error",
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
            let decoded = match self.decode(usize::try_from(frame).unwrap_or(usize::MAX)) {
                Ok(d) => d,
                Err(e) => {
                    eprintln!("cam0: {e}");
                    miss("decode_error", admit);
                    continue;
                }
            };
            let decode_ns = decoded.decode_ns;
            let (batch, storage_id) =
                match build_cam0_batch(decoded.pixels, decoded.width, decoded.height, schema) {
                    Ok(x) => x,
                    Err(e) => {
                        eprintln!("cam0: frame {seq}: {e}");
                        miss("frame_error", admit);
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
                stream: StreamId::CAM0,
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

/// Decodes one PNG to interleaved RGB8 (`image::open` → `into_rgb8` →
/// `into_raw`) and measures the decode cost. The one decode path in the
/// workspace: used by [`Cam0Driver::decode`].
pub fn decode_png(path: &Path) -> Result<Decoded, image::ImageError> {
    let t = now();
    let img = image::open(path)?.into_rgb8();
    let (width, height) = img.dimensions();
    let pixels = img.into_raw();
    Ok(Decoded {
        pixels,
        width,
        height,
        decode_ns: now() - t,
    })
}

/// Reads a PNG's **header** only and returns `(width, height)`.
///
/// `decode_png` would answer the same question by decoding every pixel; this
/// reads the 8-byte signature and the IHDR chunk and stops. Listing a data
/// root full of drives is the caller that cares.
pub fn png_dimensions(path: &Path) -> Result<(u32, u32), image::ImageError> {
    image::ImageReader::open(path)
        .map_err(image::ImageError::IoError)?
        // The format from the extension is a guess; sniff the magic bytes so a
        // mis-named file fails as "not a PNG" rather than as a decode error.
        .with_guessed_format()
        .map_err(image::ImageError::IoError)?
        .into_dimensions()
}

/// The one filename frame `i` can have: `{i:010}.png`.
///
/// [`Cam0Driver::frame_path`] builds it and [`Cam0Driver::open`] requires it,
/// so the check and the later `decode` cannot disagree about what a frame is
/// called.
pub fn frame_file_name(i: usize) -> String {
    format!("{i:010}.png")
}

/// Every `*.png` file name directly inside `dir`, exactly as the filesystem
/// spells it. See [`crate::driver::data_names`].
pub(crate) fn png_names(dir: &Path) -> std::io::Result<BTreeSet<OsString>> {
    data_names(dir, "png")
}

/// Sorted names of the directories directly inside `dir`; empty if `dir`
/// cannot be read (this only ever feeds a "what exists instead?" hint).
pub(crate) fn sibling_dirs(dir: &Path) -> Vec<String> {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut names: Vec<String> = entries
        .flatten()
        .filter(|e| e.path().is_dir())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .collect();
    names.sort();
    names
}

/// Builds the [`DriverError::DriveMissing`] for a `root/date/drive/image_02`
/// that is not a directory, naming *which* component is the wrong one.
fn drive_missing(root: &Path, date: &str, drive: &str, dir: PathBuf) -> DriverError {
    let date_dir = root.join(date);
    let drive_dir = date_dir.join(drive);
    let (level, siblings) = if !root.is_dir() {
        (MissingLevel::Root, Vec::new())
    } else if !date_dir.is_dir() {
        (MissingLevel::Date, sibling_dirs(root))
    } else if !drive_dir.is_dir() {
        (MissingLevel::Drive, sibling_dirs(&date_dir))
    } else {
        (MissingLevel::Image02, Vec::new())
    };
    DriverError::DriveMissing(Box::new(DriveMissing {
        dir,
        root: root.to_path_buf(),
        date: date.to_string(),
        drive: drive.to_string(),
        cwd: if root.is_relative() {
            std::env::current_dir().ok()
        } else {
            None
        },
        level,
        siblings,
    }))
}

/// `label: a, b, c (and N more). ` — or "none at all" when the list is empty,
/// which is itself the answer to "so what *is* there?".
fn write_siblings(
    f: &mut std::fmt::Formatter<'_>,
    label: &str,
    siblings: &[String],
) -> std::fmt::Result {
    if siblings.is_empty() {
        return write!(f, "No {label} at all. ");
    }
    write!(
        f,
        "{label}: {}",
        siblings[..siblings.len().min(SIBLINGS_SHOWN)].join(", ")
    )?;
    if siblings.len() > SIBLINGS_SHOWN {
        write!(f, " (and {} more)", siblings.len() - SIBLINGS_SHOWN)?;
    }
    f.write_str(". ")
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    fn driver(timestamps: Vec<SensorTime>) -> Cam0Driver {
        Cam0Driver {
            root: PathBuf::from("root"),
            date: "d".to_string(),
            drive: "drv".to_string(),
            index: FrameIndex::contiguous(timestamps.len()),
            timestamps,
        }
    }

    #[test]
    fn frame_path_is_zero_padded() {
        let p = driver(Vec::new()).frame_path(7);
        assert_eq!(
            p,
            Path::new("root")
                .join("d")
                .join("drv")
                .join("image_02")
                .join("data")
                .join("0000000007.png")
        );
    }

    #[test]
    fn median_period() {
        let ts = [0, 100, 210, 300].map(SensorTime);
        assert_eq!(median_period_ns(&ts), 100);
        assert_eq!(median_period_ns(&ts[..1]), DEFAULT_PERIOD_NS);
    }

    #[test]
    fn skip_at_is_next_due_or_one_period_after_last() {
        let d = driver(vec![SensorTime(1_000), SensorTime(1_100)]);
        let clock = ClockModel {
            epoch: 0,
            t0_sensor: SensorTime(1_000),
            t0_host: HostTime(0),
            t0_wall: 0,
            rate_factor: 1.0,
            offset_uncertainty_ns: 0,
            sensor_time_source: String::new(),
        };
        assert_eq!(d.skip_at(&clock, 0, 100), Some(HostTime(100)));
        assert_eq!(d.skip_at(&clock, 1, 100), Some(HostTime(200)));
        assert_eq!(d.skip_at(&clock, 2, 100), None);
    }
}
