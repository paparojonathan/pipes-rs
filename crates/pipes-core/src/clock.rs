//! Clocks (design §3). Three time types that are never subtracted across each
//! other: `SensorTime` (dataset clock), `HostTime` (process clock) and the
//! `ClockModel` that maps one onto the other for replay at a rate.
//!
//! This module is the only place in the workspace that reads `Instant` or
//! `SystemTime` (D10); everything else receives times as values.

use std::time::Duration;

use serde::{Deserialize, Serialize};

/// Nanoseconds since the Unix epoch on the dataset clock (KITTI timestamp
/// strings read as UTC).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct SensorTime(pub i64);

/// Nanoseconds since the process clock origin (the first call to [`now`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct HostTime(pub i64);

impl std::ops::Sub for HostTime {
    type Output = i64;

    fn sub(self, rhs: HostTime) -> i64 {
        self.0 - rhs.0
    }
}

impl std::ops::Sub for SensorTime {
    type Output = i64;

    fn sub(self, rhs: SensorTime) -> i64 {
        self.0 - rhs.0
    }
}

impl std::ops::Add<i64> for HostTime {
    type Output = HostTime;

    /// Saturating, so the operator is total. `ClockModel::due` can produce
    /// `i64::MAX` from a float→int cast (Rust's cast saturates), and
    /// `BoundedQueue::push` can produce it from an absurd `max_wait`; an
    /// unchecked `+` there panics in debug and — worse — wraps silently in
    /// release, turning a bad argument into a schedule of plausible-looking
    /// garbage. Callers reject those arguments at the boundary; this makes the
    /// failure impossible rather than merely unlikely.
    fn add(self, rhs: i64) -> HostTime {
        HostTime(self.0.saturating_add(rhs))
    }
}

/// Time of validity of a sample on the sensor clock.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Tov {
    /// No time of validity (a sample the sensor did not timestamp).
    None,
    /// An instant: a camera trigger, an IMU sample.
    Time(SensorTime),
    /// An interval: a lidar rotation, where `start` and `end` bracket the sweep.
    Range {
        /// First instant covered by the sample.
        start: SensorTime,
        /// Last instant covered by the sample.
        end: SensorTime,
    },
}

impl Tov {
    /// Start of validity (`Time(t)` gives `t`).
    pub fn start(self) -> Option<SensorTime> {
        match self {
            Tov::None => None,
            Tov::Time(t) => Some(t),
            Tov::Range { start, .. } => Some(start),
        }
    }

    /// End of validity (`Time(t)` gives `t`).
    pub fn end(self) -> Option<SensorTime> {
        match self {
            Tov::None => None,
            Tov::Time(t) => Some(t),
            Tov::Range { end, .. } => Some(end),
        }
    }
}

/// `ClockModel::sensor_time_source` for KITTI `timestamps.txt` read as UTC.
pub const SENSOR_TIME_SOURCE_KITTI: &str = "kitti-timestamps-txt-as-utc";

/// `due(0) = t0_host = now() + START_LEAD_NS`, so frame 0 is not born late.
pub const START_LEAD_NS: i64 = 200_000_000;

/// The sensor-to-host clock conversion for one run, written once to `run.json`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ClockModel {
    /// Reserved for clock-jump handling (phase 2); always 0 this sprint. Every
    /// sample records the epoch it was paced under, so once a backwards jump
    /// does bump this, times are never compared across one.
    pub epoch: u32,
    /// Sensor time of the run's first frame — the origin the schedule maps from.
    pub t0_sensor: SensorTime,
    /// Host time the schedule maps onto: `now()` plus [`START_LEAD_NS`], so
    /// frame 0 is not already late when the run begins.
    pub t0_host: HostTime,
    /// UTC ns since the Unix epoch, read once via [`wall_now_unix_ns`].
    pub t0_wall: i64,
    /// 1.0 = real time; `f64::INFINITY` = unpaced (`due` is `None`).
    pub rate_factor: f64,
    /// 0 for a dataset; a live sensor would fill in its own clock uncertainty.
    pub offset_uncertainty_ns: i64,
    /// How sensor time was obtained, recorded so a reader can tell what the
    /// numbers mean — [`SENSOR_TIME_SOURCE_KITTI`] for this dataset.
    pub sensor_time_source: String,
}

impl ClockModel {
    /// `t0_host = now() + START_LEAD_NS`; `t0_wall = wall_now_unix_ns()`;
    /// epoch 0; offset 0; source KITTI.
    pub fn start_now(t0_sensor: SensorTime, rate_factor: f64) -> Self {
        let t0_host = now() + START_LEAD_NS;
        let t0_wall = wall_now_unix_ns();
        ClockModel {
            epoch: 0,
            t0_sensor,
            t0_host,
            t0_wall,
            rate_factor,
            offset_uncertainty_ns: 0,
            sensor_time_source: SENSOR_TIME_SOURCE_KITTI.to_string(),
        }
    }

    /// Host deadline of `tov`. `None` if `rate_factor` is not finite or `tov`
    /// has no start. Rate 1.0 takes the exact integer path.
    pub fn due(&self, tov: Tov) -> Option<HostTime> {
        if !self.rate_factor.is_finite() {
            return None;
        }
        let delta = tov.start()? - self.t0_sensor;
        if self.rate_factor == 1.0 {
            Some(self.t0_host + delta)
        } else {
            Some(self.t0_host + (delta as f64 / self.rate_factor).round() as i64)
        }
    }
}

/// Monotonic host time. The only reader of `Instant::now` in the workspace.
#[allow(clippy::disallowed_methods)]
pub fn now() -> HostTime {
    static T0: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
    let t0 = *T0.get_or_init(std::time::Instant::now);
    HostTime(i64::try_from(t0.elapsed().as_nanos()).unwrap_or(i64::MAX))
}

/// Wall clock as UTC nanoseconds since the Unix epoch; `0` if the system
/// clock is before the epoch. The only reader of `SystemTime::now`.
#[allow(clippy::disallowed_methods)]
pub fn wall_now_unix_ns() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .ok()
        .and_then(|d| i64::try_from(d.as_nanos()).ok())
        .unwrap_or(0)
}

/// Longest single `thread::sleep` issued while pacing. The M0 sleep probe
/// characterised back-to-back `sleep(1 ms)` (p99 overshoot 1.086 ms on this
/// host); one long sleep on an otherwise idle core overshoots by far more
/// (0–27 ms measured in the first M1 dry run), so the wait is taken in
/// probe-sized steps and only the final `spin_window` is spun.
pub const MAX_SLEEP_STEP: Duration = Duration::from_millis(1);

/// Sleep in steps of at most [`MAX_SLEEP_STEP`] until `due - spin_window`,
/// then spin until `now() >= due`. Returns immediately if already past due.
pub fn sleep_until(due: HostTime, spin_window: Duration) {
    let spin = i64::try_from(spin_window.as_nanos()).unwrap_or(i64::MAX);
    let step = i64::try_from(MAX_SLEEP_STEP.as_nanos()).unwrap_or(i64::MAX);
    loop {
        let rem = due - now();
        if rem <= 0 {
            return;
        }
        if rem > spin {
            let ns = (rem - spin).min(step);
            std::thread::sleep(Duration::from_nanos(u64::try_from(ns).unwrap_or(0)));
        } else {
            std::hint::spin_loop();
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    fn model(rate_factor: f64) -> ClockModel {
        ClockModel {
            epoch: 0,
            t0_sensor: SensorTime(1_000),
            t0_host: HostTime(5),
            t0_wall: 0,
            rate_factor,
            offset_uncertainty_ns: 0,
            sensor_time_source: SENSOR_TIME_SOURCE_KITTI.to_string(),
        }
    }

    #[test]
    fn due_rate_1_is_exact() {
        let m = model(1.0);
        assert_eq!(m.due(Tov::Time(SensorTime(1_250))), Some(HostTime(255)));
        assert_eq!(m.due(Tov::None), None);
        assert_eq!(
            m.due(Tov::Range {
                start: SensorTime(1_250),
                end: SensorTime(1_300)
            }),
            Some(HostTime(255))
        );
    }

    #[test]
    fn due_inf_is_none() {
        assert_eq!(model(f64::INFINITY).due(Tov::Time(SensorTime(1_250))), None);
    }

    #[test]
    fn due_rate_10() {
        assert_eq!(
            model(10.0).due(Tov::Time(SensorTime(2_000))),
            Some(HostTime(105))
        );
    }

    #[test]
    fn now_is_monotone() {
        let a = now();
        let b = now();
        assert!(b >= a);
    }

    #[test]
    fn sleep_until_past_due_returns() {
        let t = now();
        sleep_until(HostTime(t.0 - 1), Duration::from_millis(2));
        assert!(now() - t < 1_000_000);
    }

    #[test]
    fn tov_start_end() {
        let r = Tov::Range {
            start: SensorTime(1),
            end: SensorTime(2),
        };
        assert_eq!(r.start(), Some(SensorTime(1)));
        assert_eq!(r.end(), Some(SensorTime(2)));
        assert_eq!(Tov::Time(SensorTime(7)).end(), Some(SensorTime(7)));
        assert_eq!(Tov::None.start(), None);
    }
}
