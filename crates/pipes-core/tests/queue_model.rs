//! `BoundedQueue` against a reference model, over long deterministic operation
//! sequences.
//!
//! This is the coverage a property-test crate would have given, without the
//! dependency: proptest shrinks *values*, and the failures that threaten this
//! queue are *interleavings*, which it cannot explore. A fixed-seed LCG gives
//! the same 10_000 operations on every OS and every run, so a failure is
//! reproducible from the seed alone.

use std::collections::VecDeque;
use std::sync::Arc;

use pipes_core::clock::HostTime;
use pipes_core::queue::{BoundedQueue, PushOutcome, QueuePolicy};

/// Deterministic LCG: no dependency, same sequence on every OS and every run.
fn next(seed: &mut u64) -> u64 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *seed >> 33
}

#[test]
fn queue_model_matches_reference_over_10k_ops() {
    for cap in [1usize, 2, 7] {
        for policy in [QueuePolicy::DropOldest, QueuePolicy::DropNewest] {
            let q: BoundedQueue<u64> = BoundedQueue::new(cap, policy);
            let mut model: VecDeque<u64> = VecDeque::new();
            let mut model_dropped = 0u64;
            let mut seed = 0x5171_2345u64;
            let mut pushed = 0u64;
            let mut delivered = 0u64;

            for k in 0..10_000u64 {
                if next(&mut seed) % 10 < 6 {
                    pushed += 1;
                    let out = q.try_push(k, HostTime(k as i64));
                    match policy {
                        QueuePolicy::DropOldest => {
                            if model.len() == cap {
                                model.pop_front();
                                model_dropped += 1;
                                assert!(
                                    matches!(out, PushOutcome::Evicted(_)),
                                    "cap={cap} {policy:?} op {k}: full queue did not evict"
                                );
                            } else {
                                assert!(matches!(out, PushOutcome::Accepted { .. }));
                            }
                            model.push_back(k);
                        }
                        QueuePolicy::DropNewest => {
                            if model.len() == cap {
                                model_dropped += 1;
                                assert!(
                                    matches!(out, PushOutcome::Rejected(_)),
                                    "cap={cap} {policy:?} op {k}: full queue did not reject"
                                );
                            } else {
                                assert!(matches!(out, PushOutcome::Accepted { .. }));
                                model.push_back(k);
                            }
                        }
                        QueuePolicy::Block { .. } => unreachable!(),
                    }
                } else if !model.is_empty() {
                    // Never pop an open empty queue: `pop` blocks by design.
                    delivered += 1;
                    let got = q.pop().map(|e| e.item);
                    assert_eq!(
                        got,
                        model.pop_front(),
                        "cap={cap} {policy:?} op {k}: FIFO order diverged"
                    );
                }
                assert_eq!(q.len(), model.len(), "cap={cap} {policy:?} op {k}: length");
                assert!(q.len() <= cap, "cap={cap} {policy:?} op {k}: over capacity");
                assert_eq!(
                    q.dropped(),
                    model_dropped,
                    "cap={cap} {policy:?} op {k}: drop count"
                );
            }

            // The model must have been exercised, or "matches the reference"
            // would be a statement about two idle objects.
            assert!(
                model_dropped > 0 && delivered > 0,
                "cap={cap} {policy:?}: the sequence never filled or never drained the queue"
            );
            q.close();
            while q.pop().is_some() {
                delivered += 1;
            }
            assert_eq!(
                delivered + q.dropped(),
                pushed,
                "cap={cap} {policy:?}: delivered + dropped != pushed"
            );
        }
    }
}

#[test]
fn close_drains_exactly_once() {
    let q: BoundedQueue<u64> = BoundedQueue::new(4, QueuePolicy::DropOldest);
    for i in 0..3u64 {
        assert!(matches!(
            q.try_push(i, HostTime(i as i64)),
            PushOutcome::Accepted { .. }
        ));
    }
    q.close();
    for i in 0..3u64 {
        assert_eq!(q.pop().map(|e| e.item), Some(i), "drain order at {i}");
    }
    // Twice: a closed, drained queue keeps returning None rather than blocking
    // or handing back a phantom item.
    assert_eq!(q.pop().map(|e| e.item), None);
    assert_eq!(q.pop().map(|e| e.item), None);

    let before = q.dropped();
    assert!(matches!(
        q.try_push(99, HostTime(99)),
        PushOutcome::Closed(_)
    ));
    assert_eq!(q.dropped(), before + 1, "a Closed push is a drop");
    assert_eq!(q.len(), 0, "a Closed push must not store the item");
}

#[test]
fn len_never_exceeds_cap_under_two_threads() {
    const N: u64 = 10_000;
    let q: Arc<BoundedQueue<u64>> = Arc::new(BoundedQueue::new(4, QueuePolicy::DropOldest));
    let consumer = {
        let q = Arc::clone(&q);
        std::thread::spawn(move || {
            let mut delivered = 0u64;
            while q.pop().is_some() {
                delivered += 1;
            }
            delivered
        })
    };
    let producer = {
        let q = Arc::clone(&q);
        std::thread::spawn(move || {
            for i in 0..N {
                q.try_push(i, HostTime(i as i64));
                let len = q.len();
                assert!(len <= 4, "queue held {len} items at push {i}");
            }
            q.close();
        })
    };
    producer.join().unwrap();
    let delivered = consumer.join().unwrap();
    assert_eq!(
        delivered + q.dropped(),
        N,
        "delivered {delivered} + dropped {} != {N}",
        q.dropped()
    );
}
