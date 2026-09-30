//! What every KITTI driver hands to `admit`, what it returns, and the one
//! directory read both drivers make (their layout is [`crate::layout`]).
//!
//! These lived in [`crate::cam0`] while cam0 was the only driver. A second
//! producer ([`crate::velo`]) makes them the *shape of a driver* rather than a
//! detail of the camera one, and a lidar module that had to say
//! `use crate::cam0::DriverEvent` would read as though the lidar were a kind of
//! camera. `cam0` re-exports every name that moved, so existing paths resolve.

use std::collections::BTreeSet;
use std::ffi::OsString;
use std::path::Path;

use pipes_core::clock::{HostTime, SensorTime, Tov};
use pipes_core::sample::Sample;
use pipes_core::stats::percentile;

/// Sample period assumed when a stream has fewer than two timestamps.
pub const DEFAULT_PERIOD_NS: i64 = 100_000_000;

/// What a driver hands to `admit` for every seq: exactly one event each.
pub enum DriverEvent {
    /// A sample was produced, wrapped and handed over on schedule.
    Sample(Sample),
    /// A sample was not produced; the run continues without resetting phase.
    Missing {
        /// Sequence number that was skipped: the frame number.
        seq: u64,
        /// Its time of validity, known from the timestamp files -- or
        /// [`Tov::None`] for a frame the source never measured
        /// (`absent_in_source`), which has no instant to give.
        tov: Tov,
        /// Its deadline, if the run is paced. Derived from the real samples
        /// either side, rather than read, for a frame absent in the source
        /// ([`crate::layout::replay_absent`]).
        due: Option<HostTime>,
        /// Why it was skipped: `deadline_skipped`, `decode_error`,
        /// `frame_error`, `read_error`, `batch_error`, or `absent_in_source`
        /// (the sensor's source has no sample for this frame).
        reason: &'static str,
    },
}

/// Counts returned by a driver's replay loop; `admitted + missing` is every
/// frame slot the driver replayed -- `len()` for the camera, `frame_slots()`
/// for the lidar.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RunTotals {
    /// Samples handed to `admit`.
    pub admitted: u64,
    /// Samples reported missing; `admitted + missing` is the sample count.
    pub missing: u64,
    /// Wall time of the replay loop in ns.
    pub wall_ns: i64,
}

/// Median inter-sample period, or [`DEFAULT_PERIOD_NS`] with fewer than two.
///
/// Public because it is also the period the dashboard's measurement-age bound
/// is written in terms of: the stream's own cadence, not a constant, so a
/// stream captured at a different rate gets the right reference line.
pub fn median_period_ns(ts: &[SensorTime]) -> i64 {
    let mut d: Vec<i64> = ts.windows(2).map(|w| w[1] - w[0]).collect();
    // `None` only when there are fewer than two samples, i.e. no gap to
    // measure; the documented default stands in for the missing measurement.
    percentile(&mut d, 50.0).unwrap_or(DEFAULT_PERIOD_NS)
}

/// Every file name directly inside `dir` whose extension is `ext`, exactly as
/// the filesystem spells it.
///
/// One directory read, so a caller can test `n` samples without `n` `stat`
/// calls. The extension match is case-insensitive (it decides what *counts* as
/// a data file); the per-sample name match in
/// [`crate::layout::frame_number`] is not, because the driver will open one
/// exact lowercase name and nothing else.
pub(crate) fn data_names(dir: &Path, ext: &str) -> std::io::Result<BTreeSet<OsString>> {
    let mut names = BTreeSet::new();
    for entry in std::fs::read_dir(dir)? {
        let path = entry?.path();
        if path
            .extension()
            .is_some_and(|x| x.eq_ignore_ascii_case(ext))
        {
            if let Some(name) = path.file_name() {
                names.insert(name.to_os_string());
            }
        }
    }
    Ok(names)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn median_period_of_a_short_stream_is_the_default() {
        let ts = [0, 100, 210, 300].map(SensorTime);
        assert_eq!(median_period_ns(&ts), 100);
        assert_eq!(median_period_ns(&ts[..1]), DEFAULT_PERIOD_NS);
        assert_eq!(median_period_ns(&[]), DEFAULT_PERIOD_NS);
    }
}
