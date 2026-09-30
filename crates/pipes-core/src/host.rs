//! The host's own scheduling policy, where it would make the measured
//! pipeline slower for a reason that is not the pipeline's.
//!
//! On Windows 11 a process whose window is not in the foreground can be
//! power-throttled ("EcoQoS"): on a hybrid CPU its threads are steered to the
//! efficiency cores. `pipes run` is exactly such a process the moment the
//! viewer it spawns takes the focus, and on the development host (i7-12700H,
//! on AC, Balanced plan) that slowed every thread of it about twofold. The
//! fusion expired 62 to 104 of drive_0005's 154 sweeps on every one of
//! eleven live runs with the viewer's window open, and on the ten whose logs
//! were kept the detector's median service was 115 to 194 ms a frame
//! against the camera's 103 ms period (85 to 86 ms writing an `.rrd` with no
//! window), its letterbox step alone 11 to 19 ms against 10. Raising the
//! process's priority did not help (155 ms, 88 expired). Opting it out of
//! throttling from outside did (92 and 93 ms, none expired, two runs), and so
//! did this module on three more (81, 84 and 86 ms, none expired).
//! Uncertified. A run is a benchmark: its timings should be the hardware's,
//! not a function of which window has the focus.

/// What [`opt_out_of_power_throttling`] did.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PowerThrottling {
    /// Windows was told not to throttle this process's execution speed.
    Off,
    /// Windows refused, with this error code: a Windows older than the
    /// setting (Windows 10 1709), where there is nothing to opt out of.
    Refused(i32),
    /// Not Windows: no such policy here.
    NotApplicable,
}

impl PowerThrottling {
    /// The word `run.json` records.
    pub fn name(self) -> String {
        match self {
            PowerThrottling::Off => "off".to_string(),
            PowerThrottling::Refused(code) => format!("refused (os error {code})"),
            PowerThrottling::NotApplicable => "not applicable".to_string(),
        }
    }
}

/// Asks the OS not to throttle this process's execution speed while its
/// window is in the background, for the whole process and every thread it
/// starts later. A no-op off Windows.
pub fn opt_out_of_power_throttling() -> PowerThrottling {
    #[cfg(windows)]
    {
        match windows::opt_out() {
            Ok(()) => PowerThrottling::Off,
            Err(code) => PowerThrottling::Refused(code),
        }
    }
    #[cfg(not(windows))]
    {
        PowerThrottling::NotApplicable
    }
}

#[cfg(windows)]
mod windows {
    use std::ffi::c_void;

    /// `PROCESS_POWER_THROTTLING_STATE` (processthreadsapi.h).
    #[repr(C)]
    #[derive(Default)]
    pub(super) struct State {
        pub(super) version: u32,
        pub(super) control_mask: u32,
        pub(super) state_mask: u32,
    }

    /// `ProcessPowerThrottling` in `PROCESS_INFORMATION_CLASS`.
    const PROCESS_POWER_THROTTLING: i32 = 4;
    /// `PROCESS_POWER_THROTTLING_CURRENT_VERSION`.
    const VERSION: u32 = 1;
    /// `PROCESS_POWER_THROTTLING_EXECUTION_SPEED`: in the control mask, "this
    /// process decides"; left out of the state mask, "and it is not
    /// throttled".
    pub(super) const EXECUTION_SPEED: u32 = 0x1;

    #[link(name = "kernel32")]
    extern "system" {
        fn GetCurrentProcess() -> *mut c_void;
        fn SetProcessInformation(
            process: *mut c_void,
            class: i32,
            info: *const c_void,
            size: u32,
        ) -> i32;
        #[cfg(test)]
        fn GetProcessInformation(
            process: *mut c_void,
            class: i32,
            info: *mut c_void,
            size: u32,
        ) -> i32;
    }

    pub(super) fn opt_out() -> Result<(), i32> {
        let state = State {
            version: VERSION,
            control_mask: EXECUTION_SPEED,
            state_mask: 0,
        };
        // SAFETY: `GetCurrentProcess` returns a pseudo-handle that needs no
        // closing and is valid for the call. `info` points at a live
        // `PROCESS_POWER_THROTTLING_STATE` of exactly the size passed, which
        // is what the `ProcessPowerThrottling` class reads, and Windows only
        // reads it.
        let ok = unsafe {
            SetProcessInformation(
                GetCurrentProcess(),
                PROCESS_POWER_THROTTLING,
                (&state as *const State).cast(),
                std::mem::size_of::<State>() as u32,
            )
        };
        if ok != 0 {
            Ok(())
        } else {
            Err(std::io::Error::last_os_error().raw_os_error().unwrap_or(0))
        }
    }

    /// The process's throttling state as Windows holds it, for the test that
    /// the opt-out took.
    #[cfg(test)]
    pub(super) fn current() -> Result<State, i32> {
        let mut state = State {
            version: VERSION,
            ..State::default()
        };
        // SAFETY: as in `opt_out`, with `info` a live, writable state of the
        // size passed, which Windows fills in.
        let ok = unsafe {
            GetProcessInformation(
                GetCurrentProcess(),
                PROCESS_POWER_THROTTLING,
                (&mut state as *mut State).cast(),
                std::mem::size_of::<State>() as u32,
            )
        };
        if ok != 0 {
            Ok(state)
        } else {
            Err(std::io::Error::last_os_error().raw_os_error().unwrap_or(0))
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn the_word_run_json_records() {
        assert_eq!(PowerThrottling::Off.name(), "off");
        assert_eq!(PowerThrottling::Refused(87).name(), "refused (os error 87)");
        assert_eq!(PowerThrottling::NotApplicable.name(), "not applicable");
    }

    #[cfg(windows)]
    #[test]
    fn windows_takes_the_opt_out_and_holds_it() {
        assert_eq!(opt_out_of_power_throttling(), PowerThrottling::Off);
        // Read back: the process now decides its execution speed, and
        // decided it is not throttled.
        let s = windows::current().expect("GetProcessInformation");
        assert_eq!(
            s.control_mask & windows::EXECUTION_SPEED,
            windows::EXECUTION_SPEED
        );
        assert_eq!(s.state_mask & windows::EXECUTION_SPEED, 0);
    }

    #[cfg(not(windows))]
    #[test]
    fn elsewhere_there_is_nothing_to_opt_out_of() {
        assert_eq!(
            opt_out_of_power_throttling(),
            PowerThrottling::NotApplicable
        );
    }
}
