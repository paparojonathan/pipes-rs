//! `BoundedQueue<T>`: `Mutex<VecDeque>` + two `Condvar`s with a declared
//! overflow policy (design §4). `push`/`try_push` return the envelope that
//! was NOT delivered (the evictee under DropOldest, the new item otherwise)
//! so the CALLER can write the drop row; the queue itself never writes
//! evidence. `dropped` is counted here on every non-`Accepted` outcome, so
//! `delivered + dropped == pushed` holds by construction.
//!
//! Depth is stamped on BOTH edges of the cycle: `depth_at_push` on the way in
//! and `depth_after_pop` on the way out. With only the push edge, a queue that
//! is permanently full and a queue that is full for the instant of each push
//! produce the same series, and a drain is invisible — see
//! [`Queued::depth_after_pop`].

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::{Condvar, Mutex, MutexGuard, PoisonError};
use std::time::Duration;

use crate::clock::{now, HostTime};

/// One queued item with the stamps the evidence row needs.
#[derive(Debug)]
pub struct Queued<T> {
    /// The queued value itself.
    pub item: T,
    /// The pusher's `now` argument.
    pub enqueued: HostTime,
    /// Queue length BEFORE this push: the items already waiting when this one
    /// arrived, never counting this one.
    pub depth_at_push: u16,
    /// Queue length AFTER this item was removed by [`BoundedQueue::pop`]: the
    /// items this pop LEFT BEHIND, never counting the item in this envelope.
    /// `Some(0)` means "this pop emptied the queue", NOT "one item was there".
    ///
    /// After removal rather than before, for two reasons. It matches
    /// `depth_at_push`, which also excludes its own sample, so the two read as
    /// one occupancy series — backlog on arrival, backlog on departure. And the
    /// depth before removal is at least 1 by construction, so an emptied queue
    /// could never be seen; the drain is exactly the half of the cycle the push
    /// depth already fails to show.
    ///
    /// `None` on every envelope that did not come out of `pop`: the evictee
    /// under DropOldest and the item handed back by `Rejected`, `Timeout` or
    /// `Closed`. An eviction is not a drain — the pusher's item takes the
    /// evictee's place and the depth is unchanged — so stamping one would put a
    /// point on the drain series that no consumer ever produced.
    pub depth_after_pop: Option<u16>,
}

/// What happens when a push finds the queue full.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QueuePolicy {
    /// Evict the front of the queue to make room, and return the evictee so the
    /// pusher can record which sample was lost. Latest-wins: age stays bounded
    /// by the capacity.
    DropOldest,
    /// Refuse the new item and return it. The queue keeps the oldest entries,
    /// so age grows with capacity.
    DropNewest,
    /// Wait at most `max_wait` for room; always bounded. Forbidden on a data
    /// edge by design (it back-pressures the driver) and kept only as the
    /// sprint's negative control.
    Block {
        /// Longest a single push may wait before it gives up with `Timeout`.
        max_wait: Duration,
    },
}

impl QueuePolicy {
    /// `"drop-oldest"` | `"drop-newest"` | `"block"` (the CLI and CSV spelling).
    pub fn name(self) -> &'static str {
        match self {
            QueuePolicy::DropOldest => "drop-oldest",
            QueuePolicy::DropNewest => "drop-newest",
            QueuePolicy::Block { .. } => "block",
        }
    }
}

/// Result of a push. Every variant but `Accepted` counts as one drop.
#[derive(Debug)]
pub enum PushOutcome<T> {
    /// Stored; `depth` is the length AFTER the push.
    Accepted {
        /// Queue length after this item was stored.
        depth: u16,
    },
    /// Stored after evicting the OLD front envelope, which is returned.
    Evicted(Queued<T>),
    /// Not stored: full under DropNewest, or full under Block via `try_push`.
    Rejected(Queued<T>),
    /// Not stored: full under Block and `max_wait` elapsed.
    Timeout(Queued<T>),
    /// Not stored: the queue is closed.
    Closed(Queued<T>),
}

struct Inner<T> {
    q: VecDeque<Queued<T>>,
    closed: bool,
    /// Under the lock so it can be degraded at run time (spec G.1).
    policy: QueuePolicy,
}

/// Fixed-capacity FIFO with a declared overflow policy.
pub struct BoundedQueue<T> {
    inner: Mutex<Inner<T>>,
    not_empty: Condvar,
    not_full: Condvar,
    cap: usize,
    dropped: AtomicU64,
}

fn depth_u16(n: usize) -> u16 {
    u16::try_from(n).unwrap_or(u16::MAX)
}

impl<T> BoundedQueue<T> {
    /// Capacity is clamped to at least 1.
    pub fn new(cap: usize, policy: QueuePolicy) -> Self {
        let cap = cap.max(1);
        BoundedQueue {
            inner: Mutex::new(Inner {
                q: VecDeque::with_capacity(cap),
                closed: false,
                policy,
            }),
            not_empty: Condvar::new(),
            not_full: Condvar::new(),
            cap,
            dropped: AtomicU64::new(0),
        }
    }

    /// Capacity this queue was built with (at least 1).
    pub fn cap(&self) -> usize {
        self.cap
    }

    /// The overflow policy currently in force; it can change at run time via
    /// [`BoundedQueue::set_policy`] when the recorder degrades.
    pub fn policy(&self) -> QueuePolicy {
        self.lock().policy
    }

    /// Changes the policy for every later push and wakes pushers waiting
    /// under `Block` so they re-evaluate under the new policy.
    pub fn set_policy(&self, p: QueuePolicy) {
        self.lock().policy = p;
        self.not_full.notify_all();
    }

    /// D9 poison policy at every lock site: a panicked holder does not
    /// poison the pipeline.
    fn lock(&self) -> MutexGuard<'_, Inner<T>> {
        self.inner.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// One locked push attempt that never waits.
    fn push_locked(&self, g: &mut Inner<T>, item: T, now: HostTime) -> PushOutcome<T> {
        let depth = g.q.len();
        let env = Queued {
            item,
            enqueued: now,
            depth_at_push: depth_u16(depth),
            depth_after_pop: None,
        };
        if g.closed {
            self.dropped.fetch_add(1, Relaxed);
            return PushOutcome::Closed(env);
        }
        if depth < self.cap {
            g.q.push_back(env);
            self.not_empty.notify_one();
            return PushOutcome::Accepted {
                depth: depth_u16(depth + 1),
            };
        }
        match g.policy {
            QueuePolicy::DropOldest => {
                let old = g.q.pop_front();
                g.q.push_back(env);
                self.not_empty.notify_one();
                match old {
                    Some(old) => {
                        self.dropped.fetch_add(1, Relaxed);
                        PushOutcome::Evicted(old)
                    }
                    // Unreachable (cap >= 1 and the queue was full); nothing was dropped.
                    None => PushOutcome::Accepted {
                        depth: depth_u16(g.q.len()),
                    },
                }
            }
            QueuePolicy::DropNewest | QueuePolicy::Block { .. } => {
                self.dropped.fetch_add(1, Relaxed);
                PushOutcome::Rejected(env)
            }
        }
    }

    /// Never blocks. Full under `Block` is `Rejected`; the only variant
    /// Admission calls on data edges.
    pub fn try_push(&self, item: T, now: HostTime) -> PushOutcome<T> {
        let mut g = self.lock();
        self.push_locked(&mut g, item, now)
    }

    /// `try_push` under DropOldest/DropNewest. Under `Block { max_wait }`
    /// waits for room until an ABSOLUTE deadline taken from `clock::now()`,
    /// recomputed across spurious wakeups; `Timeout` carries the new item.
    pub fn push(&self, item: T, now_arg: HostTime) -> PushOutcome<T> {
        let mut g = self.lock();
        let max_wait = match g.policy {
            QueuePolicy::Block { max_wait } => max_wait,
            QueuePolicy::DropOldest | QueuePolicy::DropNewest => {
                return self.push_locked(&mut g, item, now_arg)
            }
        };
        let deadline = now() + i64::try_from(max_wait.as_nanos()).unwrap_or(i64::MAX);
        loop {
            let still_block = matches!(g.policy, QueuePolicy::Block { .. });
            if g.closed || g.q.len() < self.cap || !still_block {
                return self.push_locked(&mut g, item, now_arg);
            }
            let rem = deadline - now();
            if rem <= 0 {
                let depth = depth_u16(g.q.len());
                self.dropped.fetch_add(1, Relaxed);
                return PushOutcome::Timeout(Queued {
                    item,
                    enqueued: now_arg,
                    depth_at_push: depth,
                    depth_after_pop: None,
                });
            }
            let (guard, _) = self
                .not_full
                .wait_timeout(g, Duration::from_nanos(u64::try_from(rem).unwrap_or(0)))
                .unwrap_or_else(PoisonError::into_inner);
            g = guard;
        }
    }

    /// Blocks until an item is available. `None` only when closed AND drained.
    ///
    /// The returned envelope carries [`Queued::depth_after_pop`], read under the
    /// same lock that removed the item, so it is the queue's true length at the
    /// instant of the pop rather than a [`BoundedQueue::len`] snapshot taken
    /// afterwards, which another thread can have moved.
    pub fn pop(&self) -> Option<Queued<T>> {
        let mut g = self.lock();
        loop {
            if let Some(mut x) = g.q.pop_front() {
                // AFTER the removal: what this pop left behind, so a drain to
                // empty reads as 0. Still under the lock.
                x.depth_after_pop = Some(depth_u16(g.q.len()));
                self.not_full.notify_one();
                return Some(x);
            }
            if g.closed {
                return None;
            }
            g = self
                .not_empty
                .wait(g)
                .unwrap_or_else(PoisonError::into_inner);
        }
    }

    /// Takes an item if one is there **right now**; never waits, and never
    /// reports "closed" — `None` means only "nothing queued at this instant".
    ///
    /// Exists for a stage with **two** inputs. [`BoundedQueue::pop`] blocks,
    /// so a stage that blocks on one edge can never look at the other, and a
    /// stage that blocked on the wrong one would deadlock the moment the
    /// stream it chose fell silent. A two-input stage therefore blocks on its
    /// PRIMARY edge and drains the secondary with this.
    ///
    /// The distinction this deliberately does not make is closed-versus-empty.
    /// A drain loop wants to stop at both, and a caller that needs to know the
    /// producer has finished learns it from its primary edge returning `None`
    /// from `pop`, which is ordered by the shutdown sequence — not from
    /// guessing here.
    pub fn try_pop(&self) -> Option<Queued<T>> {
        let mut g = self.lock();
        let mut x = g.q.pop_front()?;
        // AFTER the removal, under the same lock, exactly as `pop` does it.
        x.depth_after_pop = Some(depth_u16(g.q.len()));
        self.not_full.notify_one();
        Some(x)
    }

    /// Waits for an item until an ABSOLUTE host deadline, then gives up.
    /// `None` means the deadline passed with nothing queued, or the queue was
    /// closed and drained.
    ///
    /// The read half of [`BoundedQueue::push`]'s `Block` path and written to
    /// the same rule: the deadline is absolute and the remaining wait is
    /// recomputed on every wakeup, because a spurious wakeup with a relative
    /// timeout restarts the clock and turns a bounded wait into an unbounded
    /// one.
    ///
    /// A fusion stage needs this and cannot be written honestly without it. A
    /// camera frame that has not arrived **yet** and one that was dropped
    /// upstream are different failures, and the only thing that separates them
    /// is having waited a stated interval before deciding. Refusing
    /// immediately reports every slow consumer as a loss; waiting forever
    /// reports every real loss as a hang.
    pub fn pop_by(&self, deadline: HostTime) -> Option<Queued<T>> {
        let mut g = self.lock();
        loop {
            if let Some(mut x) = g.q.pop_front() {
                x.depth_after_pop = Some(depth_u16(g.q.len()));
                self.not_full.notify_one();
                return Some(x);
            }
            if g.closed {
                return None;
            }
            let rem = deadline - now();
            if rem <= 0 {
                return None;
            }
            let (guard, _) = self
                .not_empty
                .wait_timeout(g, Duration::from_nanos(u64::try_from(rem).unwrap_or(0)))
                .unwrap_or_else(PoisonError::into_inner);
            g = guard;
        }
    }

    /// Shutdown protocol: later pushes are `Closed`, `pop` drains then returns `None`.
    pub fn close(&self) {
        let mut g = self.lock();
        g.closed = true;
        drop(g);
        self.not_empty.notify_all();
        self.not_full.notify_all();
    }

    /// Items currently queued. A snapshot: it may be stale the moment it returns.
    pub fn len(&self) -> usize {
        self.lock().q.len()
    }

    /// Whether the queue is momentarily empty; see [`BoundedQueue::len`].
    pub fn is_empty(&self) -> bool {
        self.lock().q.is_empty()
    }

    /// Items not delivered through this queue (every non-`Accepted` outcome).
    pub fn dropped(&self) -> u64 {
        self.dropped.load(Relaxed)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use std::sync::Arc;
    use std::time::Duration;

    use super::*;

    fn h(n: i64) -> HostTime {
        HostTime(n)
    }

    #[test]
    fn cap1_drop_oldest_is_latest_only() {
        let q = BoundedQueue::new(1, QueuePolicy::DropOldest);
        assert!(matches!(
            q.try_push(1u64, h(1)),
            PushOutcome::Accepted { depth: 1 }
        ));
        match q.try_push(2, h(2)) {
            PushOutcome::Evicted(e) => assert_eq!(e.item, 1),
            o => panic!("expected Evicted, got {o:?}"),
        }
        match q.try_push(3, h(3)) {
            PushOutcome::Evicted(e) => assert_eq!(e.item, 2),
            o => panic!("expected Evicted, got {o:?}"),
        }
        assert_eq!(q.len(), 1);
        assert_eq!(q.pop().unwrap().item, 3);
        assert_eq!(q.dropped(), 2);
    }

    #[test]
    fn drop_newest_returns_pushed_item() {
        let q = BoundedQueue::new(2, QueuePolicy::DropNewest);
        assert!(matches!(
            q.try_push(1u64, h(1)),
            PushOutcome::Accepted { depth: 1 }
        ));
        assert!(matches!(
            q.try_push(2, h(2)),
            PushOutcome::Accepted { depth: 2 }
        ));
        match q.try_push(3, h(30)) {
            PushOutcome::Rejected(e) => {
                assert_eq!(e.item, 3);
                assert_eq!(e.depth_at_push, 2);
                assert_eq!(e.enqueued, h(30));
            }
            o => panic!("expected Rejected, got {o:?}"),
        }
        assert_eq!(q.len(), 2);
        assert_eq!(q.pop().unwrap().item, 1);
        assert_eq!(q.pop().unwrap().item, 2);
        assert_eq!(q.dropped(), 1);
    }

    #[test]
    fn block_try_push_rejects_immediately() {
        let q = BoundedQueue::new(
            1,
            QueuePolicy::Block {
                max_wait: Duration::from_millis(50),
            },
        );
        assert!(matches!(
            q.push(1u64, now()),
            PushOutcome::Accepted { depth: 1 }
        ));
        let t = now();
        assert!(matches!(q.try_push(2, now()), PushOutcome::Rejected(_)));
        assert!(now() - t < 5_000_000, "try_push waited");
        assert_eq!(q.dropped(), 1);
    }

    #[test]
    fn block_times_out_after_max_wait() {
        let q = BoundedQueue::new(
            1,
            QueuePolicy::Block {
                max_wait: Duration::from_millis(50),
            },
        );
        assert!(matches!(
            q.push(1u64, now()),
            PushOutcome::Accepted { depth: 1 }
        ));
        let t = now();
        let out = q.push(2, now());
        let dt_ms = (now() - t) / 1_000_000;
        match out {
            PushOutcome::Timeout(e) => {
                assert_eq!(e.item, 2);
                assert_eq!(e.depth_at_push, 1);
                assert_eq!(e.depth_after_pop, None, "a timeout never reached the queue");
            }
            o => panic!("expected Timeout, got {o:?}"),
        }
        assert!((50..250).contains(&dt_ms), "waited {dt_ms} ms");
        assert_eq!(q.dropped(), 1);
        assert_eq!(q.len(), 1);
    }

    #[test]
    fn close_drains_then_none() {
        let q = BoundedQueue::new(4, QueuePolicy::DropOldest);
        assert!(matches!(
            q.try_push(1u64, h(1)),
            PushOutcome::Accepted { .. }
        ));
        assert!(matches!(q.try_push(2, h(2)), PushOutcome::Accepted { .. }));
        q.close();
        assert_eq!(q.pop().map(|e| e.item), Some(1));
        assert_eq!(q.pop().map(|e| e.item), Some(2));
        assert!(q.pop().is_none());
        assert!(q.pop().is_none());
        match q.try_push(3, h(9)) {
            PushOutcome::Closed(e) => assert_eq!(e.item, 3),
            o => panic!("expected Closed, got {o:?}"),
        }
        assert_eq!(q.dropped(), 1);
    }

    #[test]
    fn evicted_envelope_is_older_than_pusher() {
        let q = BoundedQueue::new(1, QueuePolicy::DropOldest);
        assert!(matches!(
            q.try_push(10u64, h(100)),
            PushOutcome::Accepted { .. }
        ));
        match q.try_push(20, h(200)) {
            PushOutcome::Evicted(e) => {
                assert_eq!(e.enqueued, h(100));
                assert!(e.enqueued < h(200));
                assert_eq!(e.depth_at_push, 0);
            }
            o => panic!("expected Evicted, got {o:?}"),
        }
        let p = q.pop().unwrap();
        assert_eq!(p.item, 20);
        assert_eq!(p.enqueued, h(200));
        assert_eq!(p.depth_at_push, 1);
        // The same envelope, the two depths: one behind it on the way in, none
        // behind it on the way out.
        assert_eq!(p.depth_after_pop, Some(0));
    }

    #[test]
    fn stress_two_threads_delivered_plus_dropped_equals_pushed() {
        let q = Arc::new(BoundedQueue::new(4, QueuePolicy::DropOldest));
        let producer = {
            let q = Arc::clone(&q);
            std::thread::spawn(move || {
                let (mut acc, mut ev) = (0u64, 0u64);
                for i in 0..10_000u64 {
                    match q.try_push(i, now()) {
                        PushOutcome::Accepted { .. } => acc += 1,
                        PushOutcome::Evicted(_) => ev += 1,
                        o => panic!("unexpected {o:?}"),
                    }
                }
                q.close();
                (acc, ev)
            })
        };
        let mut delivered = 0u64;
        let mut last: Option<u64> = None;
        while let Some(e) = q.pop() {
            // A depth read BEFORE the removal could be `cap` here; one read
            // after it never can, because this pop took an item out.
            assert!(
                e.depth_after_pop.is_some_and(|d| usize::from(d) < q.cap()),
                "pop reported {:?} on a cap-{} queue",
                e.depth_after_pop,
                q.cap()
            );
            if let Some(prev) = last {
                assert!(e.item > prev, "order broken: {prev} then {}", e.item);
            }
            last = Some(e.item);
            delivered += 1;
        }
        let (acc, ev) = producer.join().unwrap();
        assert_eq!(delivered + q.dropped(), 10_000);
        assert_eq!(q.dropped(), ev);
        assert_eq!(acc + ev, 10_000);
    }

    #[test]
    fn block_push_wakes_when_consumer_pops() {
        let q = Arc::new(BoundedQueue::new(
            1,
            QueuePolicy::Block {
                max_wait: Duration::from_millis(500),
            },
        ));
        assert!(matches!(q.push(1u64, now()), PushOutcome::Accepted { .. }));
        let consumer = {
            let q = Arc::clone(&q);
            std::thread::spawn(move || {
                std::thread::sleep(Duration::from_millis(20));
                q.pop().map(|e| e.item)
            })
        };
        let t = now();
        let out = q.push(2, now());
        let dt_ms = (now() - t) / 1_000_000;
        assert!(matches!(out, PushOutcome::Accepted { depth: 1 }), "{out:?}");
        assert!(dt_ms < 400, "push did not wake on pop: {dt_ms} ms");
        assert_eq!(consumer.join().unwrap(), Some(1));
        assert_eq!(q.dropped(), 0);
    }

    #[test]
    fn set_policy_releases_waiting_block_pusher() {
        let q = Arc::new(BoundedQueue::new(
            1,
            QueuePolicy::Block {
                max_wait: Duration::from_millis(500),
            },
        ));
        assert!(matches!(q.push(1u64, now()), PushOutcome::Accepted { .. }));
        let pusher = {
            let q = Arc::clone(&q);
            std::thread::spawn(move || q.push(2, now()))
        };
        std::thread::sleep(Duration::from_millis(20));
        let t = now();
        q.set_policy(QueuePolicy::DropOldest);
        let out = pusher.join().unwrap();
        let dt_ms = (now() - t) / 1_000_000;
        match out {
            PushOutcome::Evicted(e) => assert_eq!(e.item, 1),
            o => panic!("expected Evicted after degrade, got {o:?}"),
        }
        assert!(dt_ms < 400, "pusher did not wake on set_policy: {dt_ms} ms");
        assert_eq!(q.policy(), QueuePolicy::DropOldest);
        assert_eq!(q.pop().unwrap().item, 2);
    }

    #[test]
    fn depth_after_pop_counts_what_the_pop_left_behind() {
        // Fill to capacity, then drain it. Every pop reports the items it left,
        // so the series walks 3, 2, 1, 0 and only reaches 0 once the queue is
        // actually empty. `depth_at_push` on the same envelopes walks 0, 1, 2,
        // 3 — the two numbers are different questions about the same item, and
        // for item 1 they are 0 and 3.
        let q = BoundedQueue::new(4, QueuePolicy::DropOldest);
        for i in 1..=4u64 {
            assert!(
                matches!(q.try_push(i, h(i as i64)), PushOutcome::Accepted { .. }),
                "setup: push {i} rejected"
            );
        }
        assert_eq!(q.len(), 4, "setup: queue is not full");
        let drained: Vec<(u64, u16, Option<u16>)> = std::iter::from_fn(|| q.pop())
            .take(4)
            .map(|e| (e.item, e.depth_at_push, e.depth_after_pop))
            .collect();
        assert_eq!(
            drained,
            vec![
                (1, 0, Some(3)),
                (2, 1, Some(2)),
                (3, 2, Some(1)),
                (4, 3, Some(0)),
            ]
        );
    }

    #[test]
    fn depth_after_pop_is_none_unless_the_item_was_popped() {
        // Positive control: an envelope that DID come out of `pop` carries a
        // depth, so the `None`s below are a real distinction and not a field
        // that is never set.
        let q = BoundedQueue::new(1, QueuePolicy::DropOldest);
        assert!(matches!(
            q.try_push(1u64, h(1)),
            PushOutcome::Accepted { .. }
        ));
        assert_eq!(q.pop().unwrap().depth_after_pop, Some(0));

        // An eviction is not a drain: the new item takes the evictee's place.
        assert!(matches!(q.try_push(2, h(2)), PushOutcome::Accepted { .. }));
        match q.try_push(3, h(3)) {
            PushOutcome::Evicted(e) => {
                assert_eq!(e.item, 2);
                assert_eq!(e.depth_at_push, 0, "the evictee kept its push depth");
                assert_eq!(e.depth_after_pop, None);
            }
            o => panic!("expected Evicted, got {o:?}"),
        }

        let r = BoundedQueue::new(1, QueuePolicy::DropNewest);
        assert!(matches!(
            r.try_push(1u64, h(1)),
            PushOutcome::Accepted { .. }
        ));
        match r.try_push(2, h(2)) {
            PushOutcome::Rejected(e) => assert_eq!(e.depth_after_pop, None),
            o => panic!("expected Rejected, got {o:?}"),
        }
        r.close();
        match r.try_push(3, h(3)) {
            PushOutcome::Closed(e) => assert_eq!(e.depth_after_pop, None),
            o => panic!("expected Closed, got {o:?}"),
        }
        // ... and the queue it never entered still drains normally.
        assert_eq!(r.pop().unwrap().depth_after_pop, Some(0));
    }

    #[test]
    fn policy_names_and_cap_clamp() {
        assert_eq!(QueuePolicy::DropOldest.name(), "drop-oldest");
        assert_eq!(QueuePolicy::DropNewest.name(), "drop-newest");
        assert_eq!(
            QueuePolicy::Block {
                max_wait: Duration::ZERO
            }
            .name(),
            "block"
        );
        let q: BoundedQueue<u64> = BoundedQueue::new(0, QueuePolicy::DropNewest);
        assert_eq!(q.cap(), 1);
        assert!(q.is_empty());
    }
}
