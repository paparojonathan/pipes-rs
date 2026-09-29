//! Admission (design §5.3, tracker D8): one mutex assigns `arrival_seq` and
//! snapshots the edges this sample's stream travels on, is released, and only
//! then is the sample pushed to each of them. Drop* edges get `try_push`
//! (never waits); a `Block` data edge gets a bounded `push(max_wait)` on the
//! caller's (driver's) thread — the sprint's negative control (D14). Every
//! push is timed (`push_blocked_ns`) and every non-Accepted outcome becomes
//! the pusher's drop row.
//!
//! **Two producers share one `Admission`.** `admit` takes `&self` and holds
//! the mutex only for the bookkeeping, so the cam0 driver and the velodyne
//! driver can call it concurrently from their own threads: that is what makes
//! `arrival_seq` a single, global admission order across streams rather than
//! a per-stream counter, and it is the one ordering decision that serialises
//! the two producers.
//!
//! Because one `Admission` now carries more than one stream, an [`Edge`] names
//! the stream it accepts. Without that filter `admit` would push a lidar sweep
//! onto `cam0->proc`, where the consumer would read a point cloud as a camera
//! frame. Each edge therefore has its own admitted count — the number of
//! samples *routed onto it* — and that, not the global `arrival_seq`, is the
//! denominator of `delivered + dropped == admitted` (spec H.3).

use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use pipes_core::clock::now;
use pipes_core::evidence::Evidence;
use pipes_core::queue::{BoundedQueue, QueuePolicy};
use pipes_core::sample::{Sample, StreamId};

use crate::record::{EvRow, EvidenceSink};
use crate::run::RunCtx;

/// One fan-out target of one stream.
#[derive(Clone)]
pub struct Edge {
    /// e.g. `"cam0->proc"`.
    pub name: &'static str,
    /// Only samples of this stream travel here. An edge belongs to exactly one
    /// producer; a consumer that wants two streams takes two edges.
    pub stream: StreamId,
    pub queue: Arc<BoundedQueue<Arc<Sample>>>,
}

struct AdmissionInner {
    next_seq: u64,
    edges: Vec<Edge>,
    /// Samples admitted per stream, indexed by [`StreamId::0`]. The driver
    /// invariant `admitted + missing == n` is per producer, so it needs this
    /// rather than `next_seq`, which counts every stream at once.
    by_stream: std::collections::BTreeMap<u8, u64>,
}

pub struct Admission {
    inner: Mutex<AdmissionInner>,
    sink: Arc<EvidenceSink>,
    ctx: Arc<RunCtx>,
    /// Edge names, positionally aligned with [`Admission::blocked`] and
    /// [`Admission::routed`]. Lookups are by name: the indices used to be
    /// positional at the call sites, so inserting an edge anywhere but the end
    /// silently reassigned another edge's timings.
    names: Vec<&'static str>,
    /// Per edge: the duration of every push (ns).
    blocked: Vec<Mutex<Vec<i64>>>,
    /// Per edge: samples routed onto it.
    routed: Vec<AtomicU64>,
}

impl Admission {
    pub fn new(edges: Vec<Edge>, sink: Arc<EvidenceSink>, ctx: Arc<RunCtx>) -> Self {
        let names = edges.iter().map(|e| e.name).collect();
        let blocked = edges
            .iter()
            .map(|_| Mutex::new(Vec::with_capacity(ctx.n_frames)))
            .collect();
        let routed = edges.iter().map(|_| AtomicU64::new(0)).collect();
        Admission {
            inner: Mutex::new(AdmissionInner {
                next_seq: 0,
                edges,
                by_stream: std::collections::BTreeMap::new(),
            }),
            sink,
            ctx,
            names,
            blocked,
            routed,
        }
    }

    fn lock(&self) -> MutexGuard<'_, AdmissionInner> {
        self.inner.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn index_of(&self, edge: &str) -> Option<usize> {
        self.names.iter().position(|n| *n == edge)
    }

    /// Two-step admit (D8). Step 1 under the mutex: assign `arrival_seq`,
    /// count the sample against its stream, wrap, and snapshot **the edges
    /// this sample's stream travels on** — no blocking call. Step 2 with the
    /// mutex released: push to each of them, time it, and turn a non-Accepted
    /// outcome into the drop row.
    pub fn admit(&self, mut sample: Sample) -> Arc<Sample> {
        let (arc, targets) = {
            let mut g = self.lock();
            sample.arrival_seq = g.next_seq;
            g.next_seq += 1;
            *g.by_stream.entry(sample.stream.0).or_insert(0) += 1;
            let stream = sample.stream;
            let targets: Vec<(usize, Edge)> = g
                .edges
                .iter()
                .enumerate()
                .filter(|(_, e)| e.stream == stream)
                .map(|(i, e)| (i, e.clone()))
                .collect();
            for &(i, _) in &targets {
                // Under the mutex, so an edge's admitted count can never
                // disagree with the `arrival_seq` that was handed out with it.
                self.routed[i].fetch_add(1, Relaxed);
            }
            (Arc::new(sample), targets)
        };
        for (i, e) in &targets {
            let policy = e.queue.policy();
            let t0 = now();
            let out = if matches!(policy, QueuePolicy::Block { .. }) {
                e.queue.push(Arc::clone(&arc), t0)
            } else {
                e.queue.try_push(Arc::clone(&arc), t0)
            };
            let blocked = now() - t0;
            if let Some(v) = self.blocked.get(*i) {
                v.lock()
                    .unwrap_or_else(PoisonError::into_inner)
                    .push(blocked);
            }
            if let Some(row) = Evidence::from_push(
                &self.ctx.row_ctx(),
                e.name,
                "admission",
                policy,
                out,
                blocked,
            ) {
                self.sink.send(EvRow::Evidence(row), &self.ctx, "admission");
            }
        }
        arc
    }

    /// Samples admitted so far across every stream (= the next `arrival_seq`).
    pub fn admitted(&self) -> u64 {
        self.lock().next_seq
    }

    /// Every stream that admitted anything, keyed by [`StreamId::0`]. The
    /// values sum to [`Admission::admitted`]; the run checks that, and that
    /// check is the only thing that makes `arrival_seq` a claim rather than
    /// a number.
    ///
    /// A stream that admitted nothing is **absent**, not present with 0: the
    /// run's per-driver invariant needs to tell "this producer admitted no
    /// samples" from "there was no such producer", and a defaulted 0 makes
    /// those the same row.
    pub fn admitted_by_stream(&self) -> std::collections::BTreeMap<u8, u64> {
        self.lock().by_stream.clone()
    }

    /// Samples routed onto `edge`, i.e. the denominator of that edge's
    /// `delivered + dropped == admitted`. Unknown edge names give 0.
    pub fn admitted_on(&self, edge: &str) -> u64 {
        self.index_of(edge)
            .and_then(|i| self.routed.get(i))
            .map_or(0, |c| c.load(Relaxed))
    }

    /// Duration of every push on `edge` (ns), in the order the pushes
    /// completed. With one producer that is also admission order; with two it
    /// is not, because two threads can be inside step 2 at once.
    pub fn push_blocked_ns(&self, edge: &str) -> Vec<i64> {
        self.index_of(edge)
            .and_then(|i| self.blocked.get(i))
            .map(|v| v.lock().unwrap_or_else(PoisonError::into_inner).clone())
            .unwrap_or_default()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use std::sync::{Arc, Barrier};
    use std::time::Duration;

    use pipes_core::clock::{HostTime, SensorTime, Tov};
    use pipes_core::evidence::Outcome;
    use pipes_core::queue::PushOutcome;
    use pipes_core::sample::StreamId;
    use pipes_kitti::frame::{build_cam0_batch, cam0_schema};

    use super::*;

    /// A 4×2 RGB frame through the real batch builder.
    fn test_sample(seq: u64) -> Sample {
        stream_sample(StreamId::CAM0, seq)
    }

    /// The same frame, filed under an arbitrary stream, so routing can be
    /// tested without a second payload type.
    fn stream_sample(stream: StreamId, seq: u64) -> Sample {
        let (w, h) = (4u32, 2u32);
        let (batch, storage_id) =
            build_cam0_batch(vec![1u8; (w * h * 3) as usize], w, h, &cam0_schema()).unwrap();
        Sample {
            stream,
            seq,
            arrival_seq: 0,
            parent: None,
            tov: Tov::Time(SensorTime(1)),
            epoch: 0,
            due: Some(HostTime(0)),
            arrival: now(),
            payload: batch,
            storage_id,
            decode_ns: 0,
        }
    }

    fn ctx() -> Arc<RunCtx> {
        Arc::new(RunCtx {
            run_id: "test".to_string(),
            t0_host: HostTime(0),
            epoch: 0,
            n_frames: 4,
        })
    }

    #[test]
    fn strong_count_is_three_while_both_consumers_hold_frame() {
        let qa: Arc<BoundedQueue<Arc<Sample>>> =
            Arc::new(BoundedQueue::new(1, QueuePolicy::DropOldest));
        let qb: Arc<BoundedQueue<Arc<Sample>>> =
            Arc::new(BoundedQueue::new(1, QueuePolicy::DropOldest));
        let barrier = Arc::new(Barrier::new(3));
        let sample = Arc::new(test_sample(0));
        let hold = |q: Arc<BoundedQueue<Arc<Sample>>>, b: Arc<Barrier>| {
            std::thread::spawn(move || {
                let held = q.pop().unwrap().item;
                b.wait();
                b.wait();
                drop(held);
            })
        };
        let ta = hold(Arc::clone(&qa), Arc::clone(&barrier));
        let tb = hold(Arc::clone(&qb), Arc::clone(&barrier));
        assert!(matches!(
            qa.try_push(Arc::clone(&sample), now()),
            PushOutcome::Accepted { .. }
        ));
        assert!(matches!(
            qb.try_push(Arc::clone(&sample), now()),
            PushOutcome::Accepted { .. }
        ));
        barrier.wait();
        assert_eq!(Arc::strong_count(&sample), 3);
        barrier.wait();
        ta.join().unwrap();
        tb.join().unwrap();
        assert_eq!(Arc::strong_count(&sample), 1);
    }

    /// The Block edge is the sprint's negative control: it is the one policy
    /// that stalls the driver, and `max_wait` is the promise that the stall is
    /// bounded. Nobody pops here, so the second admit can only return by
    /// timing out -- forced by construction, not by timing luck.
    #[test]
    fn block_edge_times_out_and_writes_a_timeout_row() {
        let sink = Arc::new(EvidenceSink::new(64, Duration::from_millis(100)));
        let q: Arc<BoundedQueue<Arc<Sample>>> = Arc::new(BoundedQueue::new(
            1,
            QueuePolicy::Block {
                max_wait: Duration::from_millis(50),
            },
        ));
        let edges = vec![Edge {
            name: "cam0->proc",
            stream: StreamId::CAM0,
            queue: Arc::clone(&q),
        }];
        let adm = Admission::new(edges, Arc::clone(&sink), ctx());
        adm.admit(test_sample(0)); // fills the cap-1 queue

        let t0 = now();
        adm.admit(test_sample(1)); // must wait out max_wait, then give up
        let waited = now() - t0;
        assert!(
            waited >= 50_000_000,
            "returned after {waited} ns, before max_wait elapsed"
        );
        assert!(waited < 500_000_000, "waited {waited} ns, 10x max_wait");

        // A Timeout still consumed an arrival_seq: the sample was admitted,
        // it just did not get through this edge.
        assert_eq!(adm.admitted(), 2);
        assert_eq!(q.len(), 1, "the timed-out sample must not be stored");

        sink.close();
        let sq = sink.queue();
        let mut rows = Vec::new();
        while let Some(env) = sq.pop() {
            if let EvRow::Evidence(e) = env.item {
                rows.push(e);
            }
        }
        assert_eq!(rows.len(), 1, "expected exactly one drop row, got {rows:?}");
        let row = &rows[0];
        assert_eq!(
            (row.seq, row.outcome, row.reason),
            (1, Outcome::Timeout, "max_wait")
        );
        assert_eq!(row.depth_at_push, Some(1));
        let blocked = row.push_blocked_ns.unwrap();
        assert!(
            blocked >= 50_000_000,
            "push_blocked_ns = {blocked} ns, under max_wait"
        );
        assert_eq!(sink.lost(), 0);
    }

    #[test]
    fn admit_assigns_arrival_seq_and_writes_drop_rows() {
        let sink = Arc::new(EvidenceSink::new(64, Duration::from_millis(100)));
        let qa: Arc<BoundedQueue<Arc<Sample>>> =
            Arc::new(BoundedQueue::new(1, QueuePolicy::DropOldest));
        let qb: Arc<BoundedQueue<Arc<Sample>>> =
            Arc::new(BoundedQueue::new(1, QueuePolicy::DropNewest));
        let edges = vec![
            Edge {
                name: "cam0->proc",
                stream: StreamId::CAM0,
                queue: Arc::clone(&qa),
            },
            Edge {
                name: "cam0->rerun",
                stream: StreamId::CAM0,
                queue: Arc::clone(&qb),
            },
        ];
        let adm = Admission::new(edges, Arc::clone(&sink), ctx());
        let a0 = adm.admit(test_sample(0));
        let a1 = adm.admit(test_sample(1));
        assert_eq!((a0.arrival_seq, a1.arrival_seq), (0, 1));
        assert_eq!(adm.admitted(), 2);
        assert_eq!(adm.push_blocked_ns("cam0->proc").len(), 2);
        assert_eq!(adm.push_blocked_ns("cam0->rerun").len(), 2);
        assert!(adm.push_blocked_ns("velo->cloud").is_empty());
        assert_eq!(adm.admitted_on("cam0->proc"), 2);
        assert_eq!(adm.admitted_on("cam0->rerun"), 2);
        assert_eq!(
            adm.admitted_by_stream(),
            std::collections::BTreeMap::from([(StreamId::CAM0.0, 2)])
        );
        // qa (DropOldest) evicted seq 0; qb (DropNewest) rejected seq 1.
        assert_eq!(qa.pop().unwrap().item.seq, 1);
        assert_eq!(qb.pop().unwrap().item.seq, 0);
        sink.close();
        let q = sink.queue();
        let mut rows = Vec::new();
        while let Some(env) = q.pop() {
            if let EvRow::Evidence(e) = env.item {
                rows.push(e);
            }
        }
        assert_eq!(rows.len(), 2);
        let proc_row = rows.iter().find(|r| r.edge == "cam0->proc").unwrap();
        assert_eq!(
            (proc_row.seq, proc_row.outcome, proc_row.reason),
            (0, Outcome::DroppedOldest, "evicted")
        );
        assert_eq!(
            (proc_row.stage, proc_row.arrival_seq),
            ("admission", Some(0))
        );
        let rerun_row = rows.iter().find(|r| r.edge == "cam0->rerun").unwrap();
        assert_eq!(
            (rerun_row.seq, rerun_row.outcome, rerun_row.reason),
            (1, Outcome::DroppedNewest, "full:drop-newest")
        );
        assert!(rerun_row.push_blocked_ns.is_some());
        assert_eq!(sink.lost(), 0);
        assert!(!sink.degraded());
    }

    /// The routing rule, and the reason it exists: one `Admission` now carries
    /// two streams, and without the filter a ~1.95 MB lidar sweep would be
    /// pushed onto `cam0->proc`, where the consumer reads the payload as a
    /// camera frame. The positive control is in the same test — the camera
    /// frame really does reach the camera edge — so this cannot pass by
    /// nothing being routed anywhere.
    #[test]
    fn an_edge_only_carries_its_own_stream_and_counts_only_what_it_carried() {
        let sink = Arc::new(EvidenceSink::new(64, Duration::from_millis(100)));
        let cam: Arc<BoundedQueue<Arc<Sample>>> =
            Arc::new(BoundedQueue::new(8, QueuePolicy::DropOldest));
        let velo: Arc<BoundedQueue<Arc<Sample>>> =
            Arc::new(BoundedQueue::new(8, QueuePolicy::DropOldest));
        let adm = Admission::new(
            vec![
                Edge {
                    name: "cam0->proc",
                    stream: StreamId::CAM0,
                    queue: Arc::clone(&cam),
                },
                Edge {
                    name: "velo->cloud",
                    stream: StreamId::LIDAR,
                    queue: Arc::clone(&velo),
                },
            ],
            Arc::clone(&sink),
            ctx(),
        );
        adm.admit(stream_sample(StreamId::CAM0, 0));
        adm.admit(stream_sample(StreamId::LIDAR, 0));
        adm.admit(stream_sample(StreamId::LIDAR, 1));

        assert_eq!((cam.len(), velo.len()), (1, 2), "a sample crossed streams");
        assert_eq!(cam.pop().unwrap().item.stream, StreamId::CAM0);
        for _ in 0..2 {
            assert_eq!(velo.pop().unwrap().item.stream, StreamId::LIDAR);
        }
        // Per-edge admitted is what the H.3 denominator must be: the global
        // count is 3, and neither edge saw 3 samples.
        assert_eq!(adm.admitted(), 3);
        assert_eq!(adm.admitted_on("cam0->proc"), 1);
        assert_eq!(adm.admitted_on("velo->cloud"), 2);
        assert_eq!(
            adm.admitted_by_stream(),
            std::collections::BTreeMap::from([(StreamId::CAM0.0, 1), (StreamId::LIDAR.0, 2)])
        );
        assert_eq!(
            adm.admitted_by_stream().values().sum::<u64>(),
            adm.admitted(),
            "a sample took an arrival_seq without being counted against a stream"
        );
        // And the push timings followed the routing, not the admit count.
        assert_eq!(adm.push_blocked_ns("cam0->proc").len(), 1);
        assert_eq!(adm.push_blocked_ns("velo->cloud").len(), 2);
    }

    /// Two producers, one `Admission`, and the property the whole exercise
    /// rests on: `arrival_seq` is a single dense order over both streams, with
    /// no value handed out twice and none skipped. If the mutex were per
    /// stream (or absent) this would show up as duplicates.
    #[test]
    fn two_threads_admitting_share_one_dense_arrival_seq() {
        let sink = Arc::new(EvidenceSink::new(4096, Duration::from_millis(500)));
        let cam: Arc<BoundedQueue<Arc<Sample>>> =
            Arc::new(BoundedQueue::new(256, QueuePolicy::DropOldest));
        let velo: Arc<BoundedQueue<Arc<Sample>>> =
            Arc::new(BoundedQueue::new(256, QueuePolicy::DropOldest));
        let adm = Arc::new(Admission::new(
            vec![
                Edge {
                    name: "cam0->proc",
                    stream: StreamId::CAM0,
                    queue: Arc::clone(&cam),
                },
                Edge {
                    name: "velo->cloud",
                    stream: StreamId::LIDAR,
                    queue: Arc::clone(&velo),
                },
            ],
            Arc::clone(&sink),
            ctx(),
        ));
        let n = 64u64;
        let start = Arc::new(Barrier::new(2));
        let produce = |stream: StreamId| {
            let adm = Arc::clone(&adm);
            let start = Arc::clone(&start);
            std::thread::spawn(move || {
                start.wait();
                (0..n)
                    .map(|seq| adm.admit(stream_sample(stream, seq)).arrival_seq)
                    .collect::<Vec<u64>>()
            })
        };
        let a = produce(StreamId::CAM0);
        let b = produce(StreamId::LIDAR);
        let (mut ca, mut cb) = (a.join().unwrap(), b.join().unwrap());
        assert_eq!(adm.admitted(), 2 * n);
        assert_eq!(adm.admitted_on("cam0->proc"), n);
        assert_eq!(adm.admitted_on("velo->cloud"), n);
        // Each thread saw its own values strictly increasing, and together
        // they are exactly 0..2n with no repeats.
        assert!(ca.windows(2).all(|w| w[0] < w[1]));
        assert!(cb.windows(2).all(|w| w[0] < w[1]));
        ca.append(&mut cb);
        ca.sort_unstable();
        assert_eq!(ca, (0..2 * n).collect::<Vec<u64>>());
        assert_eq!(
            adm.admitted_by_stream(),
            std::collections::BTreeMap::from([(StreamId::CAM0.0, n), (StreamId::LIDAR.0, n)])
        );
        // Absent, not zero: nothing was ever admitted on this stream, and the
        // per-driver invariant has to be able to tell that from a producer
        // that ran and admitted nothing.
        assert!(!adm.admitted_by_stream().contains_key(&StreamId::OXTS.0));
    }
}
