#![deny(clippy::unwrap_used)]
#![deny(clippy::expect_used)]
#![warn(missing_docs)]
#![deny(unsafe_op_in_unsafe_fn)]
//! Core types for the Pipes single-stream sprint: clocks, samples, bounded
//! queues, evidence rows and the counting allocator.

pub mod alloc; // M2: CountingAlloc (#[global_allocator] lives HERE, once for every binary and test)
pub mod clock; // M1: SensorTime, HostTime, Tov, ClockModel, now(), sleep_until()
pub mod evidence; // M5
pub mod host; // the OS's power throttling, opted out of so timings are the hardware's
pub mod queue; // M3
pub mod sample; // M1/M2: StreamId, Sample
pub mod stats; // M0: percentile()
