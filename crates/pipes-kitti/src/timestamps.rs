//! `parse_timestamps`: the KITTI `timestamps.txt` reader. Each line is
//! `YYYY-MM-DD HH:MM:SS.fffffffff` with no zone and is read as UTC (design §3).

use std::path::Path;

use chrono::NaiveDateTime;
use pipes_core::clock::SensorTime;

/// chrono format of one `timestamps.txt` line (`%.f` consumes the dot and 1–9 digits).
pub const KITTI_TS_FMT: &str = "%Y-%m-%d %H:%M:%S%.f";

/// Errors from reading a whole `timestamps.txt`.
#[derive(Debug)]
pub enum TimestampError {
    /// The file could not be read.
    Io(std::io::Error),
    /// A line did not match [`KITTI_TS_FMT`].
    Parse {
        /// 1-based line number.
        line_no: usize,
        /// The offending line, for the error message.
        text: String,
        /// Underlying chrono error.
        source: chrono::ParseError,
    },
    /// A line parsed but falls outside the range representable as i64
    /// nanoseconds since the Unix epoch.
    OutOfRange {
        /// 1-based line number.
        line_no: usize,
    },
}

impl std::fmt::Display for TimestampError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TimestampError::Io(e) => write!(f, "timestamps.txt: {e}"),
            TimestampError::Parse {
                line_no,
                text,
                source,
            } => write!(f, "timestamps.txt line {line_no}: {source} in {text:?}"),
            TimestampError::OutOfRange { line_no } => {
                write!(f, "timestamps.txt line {line_no}: out of i64 ns range")
            }
        }
    }
}

impl std::error::Error for TimestampError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            TimestampError::Io(e) => Some(e),
            TimestampError::Parse { source, .. } => Some(source),
            TimestampError::OutOfRange { .. } => None,
        }
    }
}

impl From<std::io::Error> for TimestampError {
    fn from(e: std::io::Error) -> Self {
        TimestampError::Io(e)
    }
}

/// Errors from parsing one line.
#[derive(Debug)]
pub enum LineError {
    /// The line did not match [`KITTI_TS_FMT`].
    Parse(chrono::ParseError),
    /// The instant is outside i64-nanosecond range.
    OutOfRange,
}

/// Parses one trimmed line as UTC and returns ns since the Unix epoch.
pub fn parse_timestamp_line(line: &str) -> Result<SensorTime, LineError> {
    NaiveDateTime::parse_from_str(line.trim(), KITTI_TS_FMT)
        .map_err(LineError::Parse)?
        .and_utc()
        .timestamp_nanos_opt()
        .map(SensorTime)
        .ok_or(LineError::OutOfRange)
}

/// Reads `path` line by line, KEEPING its blank lines: entry `i` is line
/// `i + 1` as an editor numbers it, `None` where that line is empty after
/// trim.
///
/// This is the reader a sensor directory needs. KITTI writes one timestamp
/// line per frame -- line `i + 1` is frame `i` -- and leaves the line blank
/// where the frame has no sample (drive 0009's three velodyne files have 447
/// lines each, blank at lines 178-181: frames 177-180), so a blank line is
/// data, the position of a frame the source does not have, and dropping it
/// would renumber every frame after it. [`crate::layout::FrameIndex::read`]
/// pairs these lines with the data files.
pub fn parse_timestamp_lines(path: &Path) -> Result<Vec<Option<SensorTime>>, TimestampError> {
    let text = std::fs::read_to_string(path)?;
    let mut out = Vec::new();
    for (idx, line) in text.lines().enumerate() {
        let line_no = idx + 1;
        if line.trim().is_empty() {
            out.push(None);
            continue;
        }
        let t = parse_timestamp_line(line).map_err(|e| match e {
            LineError::Parse(source) => TimestampError::Parse {
                line_no,
                text: line.to_string(),
                source,
            },
            LineError::OutOfRange => TimestampError::OutOfRange { line_no },
        })?;
        out.push(Some(t));
    }
    Ok(out)
}

/// Reads `path`; lines are numbered from 1; lines that are empty after trim
/// are skipped, so the result is every instant the file holds, in order.
///
/// Not the reader for a sensor directory, where a blank line is the place of
/// a frame with no sample and skipping it renumbers the frames after it: that
/// is [`parse_timestamp_lines`]. On a file with no blank line the two agree.
pub fn parse_timestamps(path: &Path) -> Result<Vec<SensorTime>, TimestampError> {
    Ok(parse_timestamp_lines(path)?.into_iter().flatten().collect())
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    const KNOWN: &str = "2011-09-26 13:02:25.594360375";
    const KNOWN_NS: i64 = 1_317_042_145_594_360_375;

    #[test]
    fn ts_known_line() {
        assert_eq!(parse_timestamp_line(KNOWN).unwrap(), SensorTime(KNOWN_NS));
    }

    #[test]
    fn ts_nine_digit_fraction_is_exact() {
        assert_eq!(
            parse_timestamp_line("2011-09-26 13:02:25.000000001").unwrap(),
            SensorTime(1_317_042_145_000_000_001)
        );
    }

    #[test]
    fn ts_crlf_tolerated() {
        assert_eq!(
            parse_timestamp_line(&format!("{KNOWN}\r\n")).unwrap(),
            SensorTime(KNOWN_NS)
        );
    }

    /// A blank line is kept, at its place, and a line after it keeps its
    /// number: KITTI's mark of a frame with no sample.
    #[test]
    fn ts_blank_lines_are_kept_in_place_and_numbering_is_the_file_s() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("timestamps.txt");
        std::fs::write(&path, format!("{KNOWN}\n\n  \r\n{KNOWN}\nnot a time\n")).unwrap();
        let err = parse_timestamp_lines(&path).unwrap_err();
        // Line 5 as an editor numbers it, blank lines counted.
        assert!(
            matches!(err, TimestampError::Parse { line_no: 5, .. }),
            "{err:?}"
        );
        std::fs::write(&path, format!("{KNOWN}\n\n  \r\n{KNOWN}\n\n")).unwrap();
        let t = SensorTime(KNOWN_NS);
        assert_eq!(
            parse_timestamp_lines(&path).unwrap(),
            vec![Some(t), None, None, Some(t), None],
            "a trailing blank line is a line too"
        );
        assert_eq!(parse_timestamps(&path).unwrap(), vec![t; 2]);
    }

    #[test]
    fn ts_garbage_is_parse_error() {
        assert!(matches!(
            parse_timestamp_line("not a timestamp"),
            Err(LineError::Parse(_))
        ));
    }

    #[test]
    #[ignore = "needs KITTI at PIPES_KITTI_ROOT"]
    fn ts_first_and_last_line_of_drive_0005() {
        // Derived from the crate's own location, not a machine-specific
        // absolute path: `<repo>/crates/pipes-kitti` -> three levels up is the
        // directory holding the checkout, and the dataset sits beside it
        // (README, "Setup"). Works on any clone following the documented layout, and
        // still skips cleanly where the data is absent.
        let root = std::env::var("PIPES_KITTI_ROOT").unwrap_or_else(|_| {
            concat!(env!("CARGO_MANIFEST_DIR"), "/../../../data/kitti").to_string()
        });
        let path = Path::new(&root)
            .join("2011_09_26")
            .join("2011_09_26_drive_0005_sync")
            .join("image_02")
            .join("timestamps.txt");
        let v = parse_timestamps(&path).unwrap();
        assert_eq!(v.len(), 154);
        assert_eq!(v[0], SensorTime(1_317_042_272_345_808_896));
        assert_eq!(v[153], SensorTime(1_317_042_288_145_451_520));
    }
}
