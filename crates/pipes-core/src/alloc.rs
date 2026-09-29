//! `CountingAlloc`: the per-stage byte-counting global allocator (D11).
//!
//! Every thread carries a stage slot in a `const`-initialised thread-local
//! `Cell<usize>`; every allocation adds its requested size to that slot's
//! global `AtomicU64`. Nothing in here allocates: the counters are `static`
//! arrays, the thread-local has no lazy initialiser and no destructor, and
//! `try_with` returns `Err` instead of panicking during thread teardown.
//! The `#[global_allocator]` is declared here, once, so every binary and
//! every test binary that links `pipes-core` counts.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

/// Number of per-stage counter slots, one per stage. An out-of-range
/// index is a programming error, so [`set_stage_slot`] asserts in debug builds
/// (where the tests run) and clamps into the last slot in release, where
/// mis-attributing a counter is preferable to aborting a measured run.
///
/// Was 8 while cam0 was the only stream. The lidar needed two more — a producer
/// and a consumer — and slot 7 was the only one left, so the array grew rather
/// than the velodyne driver and its consumer sharing a counter, which would
/// have made the lidar's per-stage `bytes_alloc` the sum of two stages and
/// silently unreadable.
///
/// It grew again, from 10, for the derived-cloud chain, and the argument is
/// the same one sharpened: the entire point of that chain is reading the
/// payload's size **per stage** as it falls, so `reduce` and the consumer of
/// its result sharing a counter would have destroyed exactly the reading the
/// work exists to produce. One free slot was not enough for two stages, so the
/// array grew rather than the measurement being compromised.
///
/// And again, from 12, for the `detect` stage and the consumer of ITS result,
/// for the third time for the same reason. The chain is now four links long
/// and each one's allocation has to be readable on its own, or the claim that
/// the payload shrinks while the allocation stays at zero is unreadable at the
/// link where it matters most.
///
/// And from 13, for `track` and `state` — the two stages that end the chain in
/// an answer — plus the leaf that consumes the answer. Three at once, and the
/// same argument a fourth time: the whole point of those links is that the
/// payload falls from ~8 KB to ~700 B to ~40 B while every stage's
/// `bytes_alloc` stays one output buffer, and two stages sharing a counter
/// would make exactly that reading impossible. Slot 15 was then the last one,
/// and a sixth link would have to grow this array again.
///
/// And from 16, for the camera detector ([`SLOT_CAMDET`]): the one stage whose
/// allocation is supposed to be LARGE -- a letterbox, a tensor and a network's
/// activations per frame -- and the argument turns round. Sharing a counter
/// with `proc` would hide that cost inside the grayscale pass's 465,750 B, and
/// hiding a real copy is the one thing this counter exists not to do.
///
/// And from 17, for the viewer ([`SLOT_VIEWER`]): the one thread that calls
/// the Rerun SDK, so what the SDK allocates when it takes a drawing is the
/// viewer's own number rather than a stage's.
pub const N_SLOTS: usize = 18;
/// Where allocations land for any thread that never called [`set_stage_slot`] —
/// notably Rerun's own batcher and encoder threads, which is why a viewer run
/// reports ~1.5 GB untracked while the pipeline's own stages report kilobytes.
pub const SLOT_UNTRACKED: usize = 0;
/// The replay thread: one frame's PNG decode plus its Arrow batch build,
/// measured between consecutive admits.
pub const SLOT_DRIVER: usize = 1;
/// The `proc` consumer: its grayscale output buffer, or nothing with `--reuse-output`.
pub const SLOT_PROC: usize = 2;
/// The `rerun` consumer: the `Blob`/chunk wrapper built per frame, never the pixels.
pub const SLOT_RERUN: usize = 3;
/// The `rec` thread: CSV writing and, with `--dashboard`, Rerun scalar logging.
pub const SLOT_REC: usize = 4;
/// Reserved for tests that measure their own allocation.
///
/// `pub` and not `pub(crate)` because `pipes-kitti` uses it from a
/// `#[cfg(test)]` module, which is a different crate; hidden from the docs
/// because it is a test seam, not API.
#[doc(hidden)]
pub const SLOT_TEST: usize = 5;
/// Reserved for tests that must not share a counter with [`SLOT_TEST`]. A test
/// seam, not API — see [`SLOT_TEST`].
#[doc(hidden)]
pub const SLOT_TEST2: usize = 6;
/// The velodyne replay thread: one sweep's `read` plus its Arrow batch build,
/// measured between consecutive admits exactly as [`SLOT_DRIVER`] is.
pub const SLOT_VELO_DRIVER: usize = 7;
/// The `velo` consumer: what it allocates per sweep, which is meant to be
/// nothing — it reads the shared point buffer in place.
pub const SLOT_VELO: usize = 8;
/// The `reduce` stage: the first stage in this project that both consumes and
/// produces Arrow, so its counter is read as two numbers rather than one —
/// nothing while it reads the sweep it borrowed, and one output buffer when it
/// builds the smaller cloud it hands on.
pub const SLOT_REDUCE: usize = 9;
/// The consumer of the reduced cloud: what it allocates per derived sample,
/// which is meant to be nothing for the same reason [`SLOT_VELO`] is. This is
/// the far end of the byte chain, and it is a separate counter from
/// [`SLOT_REDUCE`] because a shared one would report the producer's output
/// buffer as the consumer's cost.
pub const SLOT_DET_CLOUD: usize = 10;
/// The `detect` stage: the second stage that both consumes and produces Arrow,
/// so its counter is read as two numbers exactly as [`SLOT_REDUCE`] is —
/// nothing while it reads the reduced cloud it borrowed, and one output buffer
/// when it builds the detections.
pub const SLOT_DETECT: usize = 11;
/// The consumer of the detections: what it allocates per sample, which is
/// meant to be nothing for the same reason [`SLOT_DET_CLOUD`] is, and a
/// separate counter from [`SLOT_DETECT`] because a shared one would report the
/// producer's output buffer as the consumer's cost.
pub const SLOT_OBJ: usize = 12;
/// The `track` stage: the third stage that both consumes and produces Arrow,
/// read as two numbers exactly as [`SLOT_REDUCE`] and [`SLOT_DETECT`] are.
///
/// It is also the first stage with **two** inputs, so its "reading" window
/// covers both the detections it blocks on and the camera references it
/// drains — which is what makes a zero there a statement about the fusion
/// rather than about one of its halves.
pub const SLOT_TRACK: usize = 13;
/// The `state` stage: the last link, where ~700 B of tracks becomes a few
/// dozen bytes of answer. Its own counter and not [`SLOT_TRACK`]'s, because
/// the interesting number is that the stage which reduces the most allocates
/// the least, and a shared counter would report the sum.
pub const SLOT_STATE: usize = 14;
/// The consumer of the answer: the far end of the whole chain, meant to
/// allocate nothing for the reason [`SLOT_OBJ`] is.
pub const SLOT_STATE_SINK: usize = 15;
/// The `camdet` stage: the frozen camera detector. It reads the camera
/// frame's shared buffer in place and then, unlike every other consumer of a
/// shared buffer here, **copies** it -- into a letterboxed image and a float
/// tensor, because the network cannot read RGB8 -- and runs a network that
/// allocates its own activations. Its counter is expected to be megabytes per
/// frame, and it is its own so that figure is read rather than absorbed.
pub const SLOT_CAMDET: usize = 16;
/// The `viewer` thread: what the Rerun SDK allocates while it takes one
/// drawing (its rows, its batcher's bookkeeping). The drawings themselves are
/// built on the stages' own slots; this thread only hands them over.
pub const SLOT_VIEWER: usize = 17;

static BYTES_ALLOC: [AtomicU64; N_SLOTS] = [const { AtomicU64::new(0) }; N_SLOTS];
static COUNT_ALLOC: [AtomicU64; N_SLOTS] = [const { AtomicU64::new(0) }; N_SLOTS];

thread_local! {
    static SLOT: Cell<usize> = const { Cell::new(SLOT_UNTRACKED) };
}

/// The system allocator plus per-stage counters.
pub struct CountingAlloc;

#[global_allocator]
static GLOBAL: CountingAlloc = CountingAlloc;

#[inline]
fn record(bytes: usize) {
    let slot = SLOT.try_with(Cell::get).unwrap_or(SLOT_UNTRACKED);
    BYTES_ALLOC[slot].fetch_add(bytes as u64, Relaxed);
    COUNT_ALLOC[slot].fetch_add(1, Relaxed);
}

// SAFETY: `CountingAlloc` adds only counting to `System`. Every method forwards
// the caller's own `Layout` and pointer unchanged to the corresponding `System`
// method, so the allocator contract is exactly `System`'s, and the bookkeeping
// that follows touches only `static` atomics and a `Cell<usize>` thread-local —
// it allocates nothing and so cannot re-enter the allocator.
unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        // SAFETY: `l` is the caller's layout, forwarded unchanged; upholding
        // `GlobalAlloc::alloc`'s requirement (non-zero size) is the caller's
        // obligation and is passed straight through to `System`.
        let p = unsafe { System.alloc(l) };
        if !p.is_null() {
            record(l.size());
        }
        p
    }

    unsafe fn alloc_zeroed(&self, l: Layout) -> *mut u8 {
        // SAFETY: as in `alloc` — `l` is forwarded unchanged to `System`, which
        // has the same contract for the zeroing variant.
        let p = unsafe { System.alloc_zeroed(l) };
        if !p.is_null() {
            record(l.size());
        }
        p
    }

    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        // SAFETY: `p` was returned by one of this allocator's methods (which
        // return `System`'s own pointers) and `l` is the layout it was
        // allocated with, so `System` sees the pair it issued. Nothing is
        // recorded here: the counters measure requests, not frees.
        unsafe { System.dealloc(p, l) }
    }

    unsafe fn realloc(&self, p: *mut u8, l: Layout, n: usize) -> *mut u8 {
        // SAFETY: `p`/`l` are a pair `System` issued (see `dealloc`) and `n` is
        // the caller's new size, forwarded unchanged; `System::realloc` has the
        // same requirements this method already imposes on its caller.
        let q = unsafe { System.realloc(p, l, n) };
        if !q.is_null() && n > l.size() {
            record(n - l.size());
        }
        q
    }
}

/// Routes this thread's allocations to `slot`.
///
/// Debug builds assert `slot < N_SLOTS`; release builds clamp to `N_SLOTS - 1`.
pub fn set_stage_slot(slot: usize) {
    debug_assert!(slot < N_SLOTS);
    SLOT.with(|s| s.set(slot.min(N_SLOTS - 1)));
}

/// Bytes requested by allocations attributed to `slot` since process start.
pub fn bytes_alloc(slot: usize) -> u64 {
    BYTES_ALLOC[slot.min(N_SLOTS - 1)].load(Relaxed)
}

/// Number of allocations attributed to `slot` since process start.
pub fn count_alloc(slot: usize) -> u64 {
    COUNT_ALLOC[slot.min(N_SLOTS - 1)].load(Relaxed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alloc_1mib_moves_counter() {
        set_stage_slot(SLOT_TEST2);
        let b = bytes_alloc(SLOT_TEST2);
        let c = count_alloc(SLOT_TEST2);
        let v = vec![1u8; 1 << 20];
        std::hint::black_box(&v);
        assert!(bytes_alloc(SLOT_TEST2) - b >= 1 << 20);
        assert!(count_alloc(SLOT_TEST2) - c >= 1);
    }

    #[test]
    fn slot_is_per_thread() {
        set_stage_slot(SLOT_TEST);
        let before = bytes_alloc(SLOT_TEST);
        std::thread::spawn(|| {
            // A fresh thread starts untracked; its allocations must not land in SLOT_TEST.
            let v = vec![2u8; 1 << 16];
            std::hint::black_box(&v);
        })
        .join()
        .unwrap_or(());
        assert!(bytes_alloc(SLOT_TEST) - before < 1 << 16);
    }
}
