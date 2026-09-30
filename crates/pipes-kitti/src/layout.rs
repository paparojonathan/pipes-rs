//! Which frames a sensor directory holds, read the way KITTI lays out a synced
//! drive, and how a replay accounts for the frames it does not hold. One piece
//! of code for both drivers, so a gap on the camera and a gap on the lidar
//! are the same thing, measured the same way.
//!
//! **The layout.** KITTI numbers every sensor of a synced drive by one frame
//! index. Each sample file is named by its frame, `{frame:010}.<ext>`, and
//! each timestamp file has one line per frame: line `i + 1`, as an editor
//! numbers it, is frame `i`. Where the source has no sample for a frame, its
//! file is absent and its line is left **blank**. That is the raw-data
//! devkit's own statement ("Numbers in the data stream correspond to each
//! numbers in each other data stream and to line numbers in the timestamp
//! file"), and it is what KITTI's data does: `2011_09_26_drive_0009_sync`'s
//! three velodyne timestamp files have 447 lines each, blank at lines 178-181,
//! and its `data/` has no `0000000177.bin`-`0000000180.bin` -- frames
//! 177-180, 443 sweeps over 447 frames. A camera gap has never been seen in
//! KITTI, so for the camera this is the same rule applied, not a gap
//! observed.
//!
//! **The rule, exactly** ([`FrameIndex::read`]): a file `{i:010}.<ext>` exists
//! exactly when line `i + 1` is non-blank, and the non-blank lines increase.
//! The blank lines are the authority on which frames are absent; nothing is
//! inferred from the times. A trailing blank line is a frame too, declared and
//! absent. The frames past a stream's last line are for its caller to add
//! ([`FrameIndex::with_slots`]): the drive's frame count is the most any of
//! its sensors declares.
//!
//! **What a gap becomes.** A frame with a blank line and no file is *absent
//! in the source*: the driver replays it at its own place in the schedule as a
//! `Missing` with reason [`ABSENT_IN_SOURCE`] and time of validity
//! [`Tov::None`] -- the source measured nothing, so there is no instant to
//! give it -- and every sample keeps its frame number as its `seq`, so a sweep
//! and the camera frame of the same instant keep the same number everywhere.
//!
//! **What is refused** ([`LayoutError`]), each naming the real line of the
//! file as an editor shows it: a file not named by a zero-padded frame number;
//! a line with a time whose file is missing; a file whose line is blank, or
//! past the file's last line; and non-blank lines that do not increase.

use std::collections::BTreeSet;
use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::time::Duration;

use pipes_core::clock::{sleep_until, ClockModel, SensorTime, Tov};

use crate::driver::{data_names, DriverEvent};

/// The `reason` of a frame the source has no sample for: the sensor's
/// `data/` holds no file for it, and its line in the timestamp file is blank.
/// Not the pipeline's loss and not the driver's -- drive 0009's lidar lacks
/// frames 177-180 in KITTI's own release.
pub const ABSENT_IN_SOURCE: &str = "absent_in_source";

/// The frame number a data file is named by, or `None` for a name that is not
/// exactly ten ASCII digits, a dot and `ext` -- the only way KITTI names them.
///
/// Exact rather than lenient: `0000000005.PNG` or `5.png` would each parse to
/// a number, and a driver that accepted them would be guessing that the number
/// means what KITTI's names mean.
pub fn frame_number(name: &str, ext: &str) -> Option<u64> {
    let digits = name.strip_suffix(ext)?.strip_suffix('.')?;
    if digits.len() != 10 || !digits.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    digits.parse().ok()
}

/// Frame numbers as ranges, ascending: `177-180`, `3-4, 9`, or empty.
pub fn frame_ranges(frames: &[u64]) -> String {
    let mut out: Vec<String> = Vec::new();
    let mut i = 0;
    while i < frames.len() {
        let mut j = i;
        while j + 1 < frames.len() && frames[j + 1] == frames[j] + 1 {
            j += 1;
        }
        out.push(if i == j {
            frames[i].to_string()
        } else {
            format!("{}-{}", frames[i], frames[j])
        });
        i = j + 1;
    }
    out.join(", ")
}

/// Frame numbers in words: `frame 6`, `frames 177-180`, `frames 3-4, 9`.
pub fn frames_phrase(frames: &[u64]) -> String {
    match frames {
        [one] => format!("frame {one}"),
        _ => format!("frames {}", frame_ranges(frames)),
    }
}

/// The instants of a timestamp file's non-blank lines, in order: one per
/// sample, so entry `k` is the instant of the `k`-th frame that has one.
pub fn instants(lines: &[Option<SensorTime>]) -> Vec<SensorTime> {
    lines.iter().flatten().copied().collect()
}

/// One run of consecutive frames with no sample in the source.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Gap {
    /// First frame of the run.
    pub first: u64,
    /// Last frame of the run, inclusive.
    pub last: u64,
    /// The frame of the real sample just before it; `None` when the run opens
    /// the stream.
    pub before: Option<u64>,
    /// The frame of the real sample just after it; `None` when the run closes
    /// the stream.
    pub after: Option<u64>,
    /// Sensor time from the end of the sample before to the start of the one
    /// after, ns: how long the source recorded nothing. Read off the two real
    /// samples' own timestamps, so `None` unless both exist. For a sweep, end
    /// and start are the rotation's; for a camera frame, its instant.
    pub hole_ns: Option<i64>,
}

impl Gap {
    /// Frames in the run; at least one.
    pub fn frame_count(&self) -> u64 {
        self.last - self.first + 1
    }

    /// The run in words: `frames 177-180 (4 frames, 413.9 ms with no
    /// measurement between frame 176 and frame 181)`.
    pub fn describe(&self) -> String {
        let n = self.frame_count();
        let frames = if n == 1 {
            format!("frame {}", self.first)
        } else {
            format!("frames {}-{}", self.first, self.last)
        };
        let count = if n == 1 {
            "1 frame".to_string()
        } else {
            format!("{n} frames")
        };
        match (self.before, self.after, self.hole_ns) {
            (Some(b), Some(a), Some(ns)) => format!(
                "{frames} ({count}, {:.1} ms with no measurement between frame {b} and frame {a})",
                ns as f64 / 1e6
            ),
            (None, Some(a), _) => format!("{frames} ({count}, before the first sample, frame {a})"),
            (Some(b), None, _) => format!("{frames} ({count}, after the last sample, frame {b})"),
            _ => format!("{frames} ({count})"),
        }
    }
}

/// Why a sensor directory's files and timestamp lines break KITTI's layout.
/// Every variant that concerns a line names it as an editor numbers it,
/// blank lines counted.
#[derive(Debug)]
pub enum LayoutError {
    /// The data directory could not be listed.
    Io {
        /// The directory that failed.
        path: PathBuf,
        /// Underlying I/O error.
        source: std::io::Error,
    },
    /// A data file is not named by a zero-padded frame number.
    ///
    /// That number is the only thing that says which frame a file belongs to,
    /// and so which timestamp line and which sample of the other sensor. A
    /// file named anything else has no frame.
    NotAFrameNumber {
        /// The data directory that was read.
        dir: PathBuf,
        /// The first such name, in the directory's sorted order.
        name: String,
        /// The data files' extension, `png` or `bin`.
        ext: &'static str,
    },
    /// A line has a time, and the file of its frame is not in the directory.
    ///
    /// KITTI blanks the line of a frame it has no file for, so a line with a
    /// time says the frame was recorded; without its file, the time belongs
    /// to nothing. This is also what a timestamp file written with one line
    /// per EXISTING file looks like -- its lines have moved up past the gap
    /// -- and pairing it any other way would be a guess.
    MissingFile {
        /// The timestamp file.
        timestamps: PathBuf,
        /// 1-based line, as an editor numbers it: the frame plus one.
        line: usize,
        /// The data directory.
        dir: PathBuf,
        /// The file the line needs, `{frame:010}.<ext>`.
        name: String,
    },
    /// A file is in the directory, and its frame's line is blank: the source
    /// recorded no time for it.
    FileOnBlankLine {
        /// The timestamp file.
        timestamps: PathBuf,
        /// 1-based line of the file's frame.
        line: usize,
        /// The data directory.
        dir: PathBuf,
        /// The file.
        name: String,
    },
    /// A file is in the directory, and its frame's line is past the timestamp
    /// file's last line.
    FileBeyondEnd {
        /// The timestamp file.
        timestamps: PathBuf,
        /// Lines the timestamp file has.
        lines: usize,
        /// The line the file's frame would be, 1-based.
        line: usize,
        /// The data directory.
        dir: PathBuf,
        /// The file.
        name: String,
    },
    /// A non-blank line is not later than the non-blank line before it.
    ///
    /// Line `i + 1` is frame `i`'s instant and the frames were taken in
    /// order, so a line out of order means the file cannot be trusted.
    NotIncreasing {
        /// The timestamp file.
        path: PathBuf,
        /// 1-based line of the first time that is not later than the one
        /// before it.
        line: usize,
        /// 1-based line of that one: the nearest non-blank line above.
        prev_line: usize,
    },
}

/// The tail every layout error ends with: what helps.
const LAYOUT_HELP: &str = "Re-extract the sensor's directory from the drive's zip; \
     `cargo run --release -- drives` checks every drive the same way";

/// The layout every message is judged against, in the words it uses.
const KITTI_LAYOUT: &str = "KITTI writes one timestamp line per frame -- line N is \
     frame N-1 -- and leaves the line blank where the frame has no file";

impl std::fmt::Display for LayoutError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LayoutError::Io { path, source } => write!(f, "{}: {source}", path.display()),
            LayoutError::NotAFrameNumber { dir, name, ext } => write!(
                f,
                "{name} in {} is not named by a frame: KITTI names each file by \
                 its frame, ten digits and .{ext} (0000000000.{ext}, \
                 0000000001.{ext}, ...), and that number is the only link between \
                 a file, its timestamp line and the other sensor's sample of the \
                 same instant, so this file's frame would be a guess. {LAYOUT_HELP}",
                dir.display()
            ),
            LayoutError::MissingFile {
                timestamps,
                line,
                dir,
                name,
            } => write!(
                f,
                "{} line {line} has a time, but {name} is not in {}: {KITTI_LAYOUT}, \
                 so this line says frame {} was recorded, and which file its time \
                 belongs to would be a guess (a timestamp file with lines only for \
                 the files that exist reads exactly so). {LAYOUT_HELP}",
                timestamps.display(),
                dir.display(),
                line.saturating_sub(1)
            ),
            LayoutError::FileOnBlankLine {
                timestamps,
                line,
                dir,
                name,
            } => write!(
                f,
                "{name} is in {}, but its line in {}, line {line}, is blank: \
                 {KITTI_LAYOUT}, so the source has no time for frame {}, and the \
                 instant this file was measured at would be a guess. {LAYOUT_HELP}",
                dir.display(),
                timestamps.display(),
                line.saturating_sub(1)
            ),
            LayoutError::FileBeyondEnd {
                timestamps,
                lines,
                line,
                dir,
                name,
            } => write!(
                f,
                "{name} is in {}, but {} has {lines} line(s) and frame {}'s would be \
                 line {line}: {KITTI_LAYOUT}, so this file has no time, and the \
                 instant it was measured at would be a guess. {LAYOUT_HELP}",
                dir.display(),
                timestamps.display(),
                line.saturating_sub(1)
            ),
            LayoutError::NotIncreasing {
                path,
                line,
                prev_line,
            } => write!(
                f,
                "{} line {line} is not later than line {prev_line}, the line with a \
                 time before it: line N is frame N-1's instant and the frames were \
                 taken in order, so the file cannot be trusted and which instant \
                 belongs to which frame would be a guess. {LAYOUT_HELP}",
                path.display()
            ),
        }
    }
}

impl std::error::Error for LayoutError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            LayoutError::Io { source, .. } => Some(source),
            LayoutError::NotAFrameNumber { .. }
            | LayoutError::MissingFile { .. }
            | LayoutError::FileOnBlankLine { .. }
            | LayoutError::FileBeyondEnd { .. }
            | LayoutError::NotIncreasing { .. } => None,
        }
    }
}

/// Which frames one sensor directory holds, and how many frame slots its
/// stream has.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FrameIndex {
    /// The frame of each data file, ascending: `frames[k]` is the frame of
    /// the `k`-th non-blank timestamp line.
    frames: Vec<u64>,
    /// Frame slots on the stream: the timestamp file's lines, or more after
    /// [`FrameIndex::with_slots`].
    slots: u64,
}

/// One frame slot of a replay, in frame order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Slot {
    /// The frame has a sample on disk: its file, and the `k`-th non-blank
    /// timestamp line.
    Sample {
        /// The frame number: the sample's `seq`.
        frame: u64,
        /// Its index among the samples, and so among the non-blank lines.
        k: usize,
    },
    /// The frame has no sample in the source.
    Absent {
        /// The frame number.
        frame: u64,
    },
}

impl FrameIndex {
    /// Reads the data files `{frame:010}.<ext>` of `data` against `lines`,
    /// the timestamp file `timestamps` line by line (`None` for a blank line;
    /// [`crate::timestamps::parse_timestamp_lines`]), and refuses every layout
    /// that is not KITTI's (see the module docs): the first failure in frame
    /// order, named by its real line.
    ///
    /// One `read_dir` of `data`; no per-file `stat`.
    pub fn read(
        data: &Path,
        ext: &'static str,
        timestamps: &Path,
        lines: &[Option<SensorTime>],
    ) -> Result<FrameIndex, LayoutError> {
        let names: BTreeSet<OsString> =
            data_names(data, ext).map_err(|source| LayoutError::Io {
                path: data.to_path_buf(),
                source,
            })?;
        // `names` is sorted, and ten-digit names sort as their numbers do, so
        // `frames` comes out ascending; a set cannot hold one name twice, and
        // two spellings of one number cannot both pass `frame_number`, so it
        // has no repeats either.
        let mut frames: Vec<u64> = Vec::with_capacity(names.len());
        for name in &names {
            let Some(frame) = name.to_str().and_then(|n| frame_number(n, ext)) else {
                return Err(LayoutError::NotAFrameNumber {
                    dir: data.to_path_buf(),
                    name: name.to_string_lossy().into_owned(),
                    ext,
                });
            };
            frames.push(frame);
        }
        // Line by line, frame by frame: a file exactly where the line has a
        // time. Bounded by the file's own lines, so a stray file named
        // 9999999999 costs one comparison, not ten billion.
        let file = |frame: u64| format!("{frame:010}.{ext}");
        let mut next = frames.iter().peekable();
        for (i, line) in lines.iter().enumerate() {
            let frame = i as u64;
            let has_file = next.next_if(|&&f| f == frame).is_some();
            match (line, has_file) {
                (Some(_), false) => {
                    return Err(LayoutError::MissingFile {
                        timestamps: timestamps.to_path_buf(),
                        line: i + 1,
                        dir: data.to_path_buf(),
                        name: file(frame),
                    })
                }
                (None, true) => {
                    return Err(LayoutError::FileOnBlankLine {
                        timestamps: timestamps.to_path_buf(),
                        line: i + 1,
                        dir: data.to_path_buf(),
                        name: file(frame),
                    })
                }
                _ => {}
            }
        }
        if let Some(&&frame) = next.peek() {
            return Err(LayoutError::FileBeyondEnd {
                timestamps: timestamps.to_path_buf(),
                lines: lines.len(),
                line: usize::try_from(frame).map_or(usize::MAX, |f| f.saturating_add(1)),
                dir: data.to_path_buf(),
                name: file(frame),
            });
        }
        check_increasing(timestamps, lines)?;
        Ok(FrameIndex {
            frames,
            slots: lines.len() as u64,
        })
    }

    /// `n` files named `0..n`: a stream with a sample for every frame.
    pub fn contiguous(n: usize) -> FrameIndex {
        FrameIndex {
            frames: (0..n as u64).collect(),
            slots: n as u64,
        }
    }

    /// Extends the stream to `n` frame slots when that is more than its lines
    /// declare: the frames past its last line are then absent too. Never
    /// shrinks.
    pub fn with_slots(mut self, n: usize) -> FrameIndex {
        self.slots = self.slots.max(n as u64);
        self
    }

    /// The frame of each file, ascending.
    pub fn frames(&self) -> &[u64] {
        &self.frames
    }

    /// Files: samples on disk.
    pub fn len(&self) -> usize {
        self.frames.len()
    }

    /// Whether there are no files at all.
    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }

    /// Frame slots on the stream: every sample, and every frame absent in the
    /// source.
    pub fn slots(&self) -> usize {
        usize::try_from(self.slots).unwrap_or(usize::MAX)
    }

    /// The frame of file `k`.
    pub fn frame(&self, k: usize) -> Option<u64> {
        self.frames.get(k).copied()
    }

    /// The frames, ascending, with no sample in the source.
    pub fn absent(&self) -> Vec<u64> {
        self.slot_list()
            .filter_map(|s| match s {
                Slot::Absent { frame } => Some(frame),
                Slot::Sample { .. } => None,
            })
            .collect()
    }

    /// Every frame slot in order, each a sample (with its file index) or an
    /// absence.
    pub fn slot_list(&self) -> impl Iterator<Item = Slot> + '_ {
        let mut k = 0usize;
        (0..self.slots).map(move |frame| {
            if self.frames.get(k) == Some(&frame) {
                k += 1;
                Slot::Sample { frame, k: k - 1 }
            } else {
                Slot::Absent { frame }
            }
        })
    }

    /// The runs of consecutive absent frames, with the real samples either
    /// side of each. `ends[k]` and `starts[k]` are when sample `k` ended and
    /// began -- a sweep's two ends, or a camera frame's one instant twice --
    /// so a gap's hole runs from the end of the sample before it to the start
    /// of the one after.
    pub fn gaps(&self, ends: &[SensorTime], starts: &[SensorTime]) -> Vec<Gap> {
        let absent = self.absent();
        let mut out = Vec::new();
        let mut i = 0;
        while i < absent.len() {
            let mut j = i;
            while j + 1 < absent.len() && absent[j + 1] == absent[j] + 1 {
                j += 1;
            }
            let (first, last) = (absent[i], absent[j]);
            // The number of files below `first` is the index of the first
            // file above the run.
            let k = self.frames.partition_point(|&f| f < first);
            let before = k.checked_sub(1);
            let after = (k < self.frames.len()).then_some(k);
            let hole_ns = match (before, after) {
                (Some(b), Some(a)) => match (ends.get(b), starts.get(a)) {
                    (Some(&e), Some(&s)) => Some(s - e),
                    _ => None,
                },
                _ => None,
            };
            out.push(Gap {
                first,
                last,
                before: before.and_then(|b| self.frame(b)),
                after: after.and_then(|a| self.frame(a)),
                hole_ns,
            });
            i = j + 1;
        }
        out
    }

    /// The sensor-clock position of an absent frame's deadline, given each
    /// sample's own deadline (`deadlines[k]`: a sweep's end, a camera frame's
    /// instant). See [`replay_absent`] for why it is derived and what it is
    /// never used as.
    pub fn slot_deadline_ns(
        &self,
        deadlines: &[SensorTime],
        frame: u64,
        period_ns: i64,
    ) -> Option<i64> {
        let k = self.frames.partition_point(|&f| f < frame);
        let steps = |a: u64, b: u64| i64::try_from(a.abs_diff(b)).unwrap_or(i64::MAX);
        match (k.checked_sub(1), (k < self.frames.len()).then_some(k)) {
            (Some(b), Some(a)) => {
                let (fb, fa) = (self.frame(b)?, self.frame(a)?);
                let (db, da) = (deadlines.get(b)?.0, deadlines.get(a)?.0);
                // i128, so a long hole times a large frame count cannot wrap.
                let num = i128::from(da - db) * i128::from(steps(frame, fb));
                let den = i128::from(steps(fa, fb).max(1));
                Some(db + i64::try_from(num / den).ok()?)
            }
            (None, Some(a)) => {
                let fa = self.frame(a)?;
                Some(deadlines.get(a)?.0 - steps(fa, frame).saturating_mul(period_ns))
            }
            (Some(b), None) => {
                let fb = self.frame(b)?;
                Some(deadlines.get(b)?.0 + steps(frame, fb).saturating_mul(period_ns))
            }
            (None, None) => None,
        }
    }
}

/// The first non-blank line of `lines` that is not later than the non-blank
/// line before it, as the error that names both by their real line numbers.
pub fn check_increasing(path: &Path, lines: &[Option<SensorTime>]) -> Result<(), LayoutError> {
    let mut prev: Option<(usize, SensorTime)> = None;
    for (i, t) in lines.iter().enumerate() {
        let Some(t) = *t else { continue };
        if let Some((p, pt)) = prev {
            if t <= pt {
                return Err(LayoutError::NotIncreasing {
                    path: path.to_path_buf(),
                    line: i + 1,
                    prev_line: p + 1,
                });
            }
        }
        prev = Some((i, t));
    }
    Ok(())
}

/// Replays one frame absent in the source, for either driver: waits for the
/// slot's deadline, then reports it as `Missing` with reason
/// [`ABSENT_IN_SOURCE`] and time of validity [`Tov::None`].
///
/// **The deadline is derived, and only a deadline.** The source recorded
/// nothing for this frame, so there is no instant to read; the slot's
/// deadline is placed on the straight line between the deadlines of the real
/// samples either side of it, in proportion to its frame number -- frame 178
/// sits two fifths of the way from sweep 176's end to sweep 181's -- so the
/// absent frames fall due evenly through the hole the source left, and the
/// replay sits through that hole rather than closing it. Before the first
/// sample or after the last, one median period per frame from the nearest real
/// one. The sensor-clock position computed on the way is turned into a host
/// deadline and nothing else: the frame's time of validity stays
/// [`Tov::None`]. Unpaced, there is no deadline and nothing to wait for.
pub(crate) fn replay_absent(
    index: &FrameIndex,
    deadlines: &[SensorTime],
    period_ns: i64,
    clock: &ClockModel,
    spin_window: Duration,
    frame: u64,
    admit: &mut dyn FnMut(DriverEvent),
) {
    let due = index
        .slot_deadline_ns(deadlines, frame, period_ns)
        .and_then(|at| clock.due(Tov::Time(SensorTime(at))));
    if let Some(d) = due {
        sleep_until(d, spin_window);
    }
    admit(DriverEvent::Missing {
        seq: frame,
        tov: Tov::None,
        due,
        reason: ABSENT_IN_SOURCE,
    });
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    const P: i64 = 100_000_000;

    /// A directory of `{frame:010}.<ext>` files for `frames`, and KITTI's
    /// lines for them: `n_lines` lines, frame `f`'s at `f * P`, blank where
    /// `frames` has no `f`.
    fn dir_of(
        frames: &[u64],
        n_lines: usize,
        ext: &str,
    ) -> (tempfile::TempDir, Vec<Option<SensorTime>>) {
        let tmp = tempfile::TempDir::new().unwrap();
        for f in frames {
            std::fs::write(tmp.path().join(format!("{f:010}.{ext}")), b"x").unwrap();
        }
        let lines = (0..n_lines as u64)
            .map(|f| frames.contains(&f).then(|| SensorTime(f as i64 * P)))
            .collect();
        (tmp, lines)
    }

    fn read(
        tmp: &tempfile::TempDir,
        lines: &[Option<SensorTime>],
    ) -> Result<FrameIndex, LayoutError> {
        FrameIndex::read(tmp.path(), "png", Path::new("timestamps.txt"), lines)
    }

    #[test]
    fn a_file_is_named_by_its_frame_and_nothing_else() {
        assert_eq!(frame_number("0000000177.bin", "bin"), Some(177));
        assert_eq!(frame_number("0000000000.png", "png"), Some(0));
        for bad in [
            "177.bin",
            "0000000177.BIN",
            "000000177a.bin",
            "0000000177.bin.bak",
            "00000000177.bin",
            "0000000177bin",
            "0000000177.png",
        ] {
            assert_eq!(frame_number(bad, "bin"), None, "{bad}");
        }
        assert_eq!(frame_ranges(&[177, 178, 179, 180]), "177-180");
        assert_eq!(frame_ranges(&[3, 4, 9]), "3-4, 9");
        assert_eq!(frame_ranges(&[5]), "5");
        assert_eq!(frame_ranges(&[]), "");
        assert_eq!(frames_phrase(&[6]), "frame 6");
        assert_eq!(frames_phrase(&[177, 178, 179, 180]), "frames 177-180");
        assert_eq!(frames_phrase(&[3, 4, 9]), "frames 3-4, 9");
    }

    /// The rule: line `i + 1` is frame `i`, a blank line is a frame with no
    /// file, and every slot is accounted for, in frame order.
    #[test]
    fn a_blank_line_is_an_absent_frame_and_line_i_plus_1_is_frame_i() {
        let (tmp, lines) = dir_of(&[0, 1, 2, 5, 6], 7, "png");
        let ix = read(&tmp, &lines).expect("KITTI's gapped layout");
        assert_eq!(ix.frames(), [0, 1, 2, 5, 6]);
        assert_eq!((ix.len(), ix.slots()), (5, 7));
        assert_eq!(
            ix.frame(3),
            Some(5),
            "the fourth sample is the file named 5"
        );
        assert_eq!(ix.absent(), vec![3, 4]);
        let ts = instants(&lines);
        assert_eq!(ts[3], SensorTime(5 * P), "and carries line 6's time");
        assert_eq!(
            ix.slot_list().collect::<Vec<_>>(),
            vec![
                Slot::Sample { frame: 0, k: 0 },
                Slot::Sample { frame: 1, k: 1 },
                Slot::Sample { frame: 2, k: 2 },
                Slot::Absent { frame: 3 },
                Slot::Absent { frame: 4 },
                Slot::Sample { frame: 5, k: 3 },
                Slot::Sample { frame: 6, k: 4 },
            ]
        );
        // The hole runs from the sample before to the sample after.
        assert_eq!(
            ix.gaps(&ts, &ts),
            vec![Gap {
                first: 3,
                last: 4,
                before: Some(2),
                after: Some(5),
                hole_ns: Some(3 * P),
            }]
        );
        assert_eq!(
            ix.gaps(&ts, &ts)[0].describe(),
            "frames 3-4 (2 frames, 300.0 ms with no measurement between frame 2 and frame 5)"
        );
        // Frames past the last line, once the stream is told it has them,
        // are absent too; it never shrinks.
        let wider = ix.clone().with_slots(9);
        assert_eq!(wider.absent(), vec![3, 4, 7, 8]);
        assert_eq!(ix.clone().with_slots(2).slots(), 7);
        assert_eq!(wider.gaps(&ts, &ts)[1].after, None);
        // A missing first frame is a gap too, before any sample.
        let (tmp, lines) = dir_of(&[2, 3, 4], 5, "png");
        let ix = read(&tmp, &lines).unwrap();
        assert_eq!(ix.absent(), vec![0, 1]);
        let ts = instants(&lines);
        assert_eq!(ix.gaps(&ts, &ts)[0].before, None);
    }

    /// A trailing blank line is a frame the stream declares and has no file
    /// for: absent, like any other.
    #[test]
    fn trailing_blank_lines_are_declared_absent_frames() {
        let (tmp, lines) = dir_of(&[0, 1, 2], 6, "png");
        let ix = read(&tmp, &lines).unwrap();
        assert_eq!((ix.len(), ix.slots()), (3, 6));
        assert_eq!(ix.absent(), vec![3, 4, 5]);
        let ts = instants(&lines);
        assert_eq!(
            ix.gaps(&ts, &ts),
            vec![Gap {
                first: 3,
                last: 5,
                before: Some(2),
                after: None,
                hole_ns: None,
            }]
        );
    }

    /// The positive control: a file for every line is read as it always was.
    #[test]
    fn a_stream_with_a_file_per_frame_is_unchanged() {
        let (tmp, lines) = dir_of(&[0, 1, 2, 3], 4, "png");
        let ix = read(&tmp, &lines).unwrap();
        assert_eq!(ix, FrameIndex::contiguous(4));
        assert!(ix.absent().is_empty());
        let ts = instants(&lines);
        assert!(ix.gaps(&ts, &ts).is_empty());
        assert!(ix
            .slot_list()
            .all(|s| matches!(s, Slot::Sample { frame, k } if frame == k as u64)));
    }

    /// An absent frame's deadline sits where the frame would have been, on
    /// the line between the real samples either side; before the first or
    /// after the last, a period per frame from the nearest one.
    #[test]
    fn an_absent_frame_falls_due_where_it_would_have_been() {
        let (tmp, lines) = dir_of(&[1, 2, 6, 7], 8, "png");
        let ix = read(&tmp, &lines).unwrap().with_slots(9);
        let ts = instants(&lines);
        for f in [0u64, 3, 4, 5, 8] {
            assert_eq!(
                ix.slot_deadline_ns(&ts, f, P),
                Some(f as i64 * P),
                "frame {f}"
            );
        }
        // Nothing to derive from with no sample at all.
        assert_eq!(
            FrameIndex::contiguous(0)
                .with_slots(3)
                .slot_deadline_ns(&[], 1, P),
            None
        );
    }

    #[test]
    fn a_file_not_named_by_its_frame_is_refused() {
        for bad in ["thumb.png", "5.png", "0000000005.PNG"] {
            let (tmp, lines) = dir_of(&[0, 1, 2], 3, "png");
            std::fs::write(tmp.path().join(bad), b"x").unwrap();
            let err = read(&tmp, &lines)
                .err()
                .unwrap_or_else(|| panic!("{bad} accepted"));
            assert!(
                matches!(&err, LayoutError::NotAFrameNumber { name, .. } if name == bad),
                "{bad}: {err:?}"
            );
            assert!(err.to_string().contains("would be a guess"), "{err}");
        }
    }

    /// A line with a time and no file -- including the layout with one line
    /// per EXISTING file, whose lines have moved up past its gap -- is
    /// refused, naming the line as an editor shows it.
    #[test]
    fn a_line_with_a_time_and_no_file_is_refused_at_its_real_line() {
        // Frame 5's file gone, its line kept: line 6.
        let (tmp, mut lines) = dir_of(&[0, 1, 2, 3, 4, 6], 7, "png");
        lines[5] = Some(SensorTime(5 * P));
        let err = read(&tmp, &lines).expect_err("a line with no file");
        assert!(
            matches!(&err, LayoutError::MissingFile { line: 6, name, .. } if name == "0000000005.png"),
            "{err:?}"
        );
        let text = err.to_string();
        for needle in [
            "timestamps.txt line 6 has a time",
            "0000000005.png is not in",
            "frame 5 was recorded",
            "would be a guess",
        ] {
            assert!(text.contains(needle), "{needle:?} not in {text}");
        }
        // One line per EXISTING file: files 0-2 and 5-6, five lines, no
        // blanks. Line 4 is frame 3's, and frame 3 has no file.
        let (tmp, _) = dir_of(&[0, 1, 2, 5, 6], 7, "png");
        let dropped: Vec<Option<SensorTime>> =
            [0, 1, 2, 5, 6].map(|f| Some(SensorTime(f * P))).to_vec();
        let err = read(&tmp, &dropped).expect_err("lines only for the files that exist");
        assert!(
            matches!(&err, LayoutError::MissingFile { line: 4, .. }),
            "{err:?}"
        );
    }

    /// A file whose line is blank, or past the last line, has no time.
    #[test]
    fn a_file_on_a_blank_line_or_past_the_end_is_refused_at_its_real_line() {
        let (tmp, mut lines) = dir_of(&[0, 1, 2, 3], 4, "png");
        lines[2] = None;
        let err = read(&tmp, &lines).expect_err("a file on a blank line");
        assert!(
            matches!(&err, LayoutError::FileOnBlankLine { line: 3, name, .. } if name == "0000000002.png"),
            "{err:?}"
        );
        assert!(
            err.to_string()
                .contains("its line in timestamps.txt, line 3, is blank"),
            "{err}"
        );
        let (tmp, lines) = dir_of(&[0, 1, 2, 3], 3, "png");
        let err = read(&tmp, &lines).expect_err("a file past the last line");
        assert!(
            matches!(
                &err,
                LayoutError::FileBeyondEnd {
                    lines: 3,
                    line: 4,
                    ..
                }
            ),
            "{err:?}"
        );
        assert!(
            err.to_string()
                .contains("timestamps.txt has 3 line(s) and frame 3's would be line 4"),
            "{err}"
        );
        // A stray frame number far past the end is one comparison, not a walk.
        let (tmp, lines) = dir_of(&[0, 1], 2, "png");
        std::fs::write(tmp.path().join("9999999999.png"), b"x").unwrap();
        assert!(matches!(
            read(&tmp, &lines),
            Err(LayoutError::FileBeyondEnd {
                line: 10_000_000_000,
                ..
            })
        ));
    }

    /// Times that do not increase are refused, naming both lines as an editor
    /// numbers them -- blank lines between them counted.
    #[test]
    fn lines_that_do_not_increase_are_refused_at_their_real_lines() {
        let (tmp, mut lines) = dir_of(&[0, 1, 2, 3], 4, "png");
        lines.swap(1, 2);
        let err = read(&tmp, &lines).expect_err("out of order");
        assert!(
            matches!(
                err,
                LayoutError::NotIncreasing {
                    line: 3,
                    prev_line: 2,
                    ..
                }
            ),
            "{err:?}"
        );
        // Across a gap: frames 2 and 5 are lines 3 and 6.
        let (tmp, mut lines) = dir_of(&[0, 1, 2, 5, 6], 7, "png");
        lines[5] = Some(SensorTime(2 * P));
        let err = read(&tmp, &lines).expect_err("out of order across a gap");
        assert!(
            matches!(
                err,
                LayoutError::NotIncreasing {
                    line: 6,
                    prev_line: 3,
                    ..
                }
            ),
            "{err:?}"
        );
        assert!(
            err.to_string()
                .contains("timestamps.txt line 6 is not later than line 3"),
            "{err}"
        );
    }
}
