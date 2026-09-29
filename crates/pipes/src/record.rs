//! The evidence channel (spec G.1/G.2): every stage `send`s rows into one
//! bounded queue (`evidence->rec`, cap 4096, `Block { 1 s }`) and the `rec`
//! thread drains it to `evidence.csv` / `events.csv`. On a Block timeout the
//! channel degrades to DropOldest once, records exactly one
//! `RecorderDegraded` event and counts lost rows: the pipeline never stalls
//! on its recorder for longer than `max_wait` (design §7).
//!
//! M8: with `--dashboard` the same thread also mirrors each row onto the run's
//! Rerun recording as scalars ([`Dashboard`]) — a READER of the evidence
//! stream, so the live view costs no new instrumentation anywhere else. It
//! draws on a [`Canvas`] like every stage: its drawings go through the
//! viewer's own queue, so a slow viewer can never back up into the evidence
//! channel and, through it, into every stage that writes rows.

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::Relaxed};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::Duration;

use pipes_core::alloc::{set_stage_slot, SLOT_REC};
use pipes_core::clock::now;
use pipes_core::evidence::{Event, EventKind, Evidence, Outcome};
use pipes_core::queue::{BoundedQueue, PushOutcome, QueuePolicy};
use pipes_core::sample::StreamId;
use pipes_kitti::layout::ABSENT_IN_SOURCE;
use rerun::archetypes::{
    BarChart, Clear, GraphEdges, GraphNodes, Measurements, Scalars, SeriesLines, SeriesPoints,
    StateChange, StateConfiguration, TextDocument, TextLog,
};
use rerun::components::{Color, InterpolationMode, MarkerShape, MediaType, TextLogLevel};
use rerun::{AsComponents, RecordingStream};

use crate::consumers::{log_no_frame, log_no_sweep};
use crate::dashboard::{
    admission_root, entity, lane_path, queue_lane, queue_root, send_blueprint, Mode, ABSENT_LEAF,
    ADMISSION_EDGES, ALLOC_RATIO, CHAIN_BYTES, EDGES, GRAPH_EDGES, GRAPH_NODES, HEADROOM_TOP,
    LANE_CAMERA_NAME, LANE_LIDAR_NAME, LANE_NAMES, LANE_PAIRING_NAME, STAGES,
};
use crate::run::RunCtx;
use crate::viewer::{Canvas, Subject, VIEWER_EDGE};

/// What travels on `evidence->rec`.
///
/// `Evidence` is ~300 bytes and `Event` ~90, which trips
/// `clippy::large_enum_variant`. Its suggested fix — box the large variant —
/// is the wrong trade here: every stage sends one of these per sample per
/// edge, so a `Box` would put a heap allocation on the evidence path of the
/// pipeline whose allocations this project exists to count. The cost of not
/// boxing is 4096 × ~300 B ≈ 1.2 MB in the channel's buffer, paid once,
/// before the clock starts.
#[allow(clippy::large_enum_variant)]
pub enum EvRow {
    Evidence(Evidence),
    Event(Event),
}

/// The producer side of `evidence->rec`, shared by every stage as `Arc<EvidenceSink>`.
pub struct EvidenceSink {
    q: Arc<BoundedQueue<EvRow>>,
    degraded: AtomicBool,
    lost: AtomicU64,
    blocked_ns: Mutex<Vec<i64>>,
}

impl EvidenceSink {
    pub fn new(cap: usize, max_wait: Duration) -> Self {
        EvidenceSink {
            q: Arc::new(BoundedQueue::new(cap, QueuePolicy::Block { max_wait })),
            degraded: AtomicBool::new(false),
            lost: AtomicU64::new(0),
            blocked_ns: Mutex::new(Vec::new()),
        }
    }

    /// Pushes one row: bounded wait while the channel is healthy, never a
    /// wait once degraded. `stage` names the caller for the degrade event.
    pub fn send(&self, row: EvRow, ctx: &RunCtx, stage: &'static str) {
        let arrival_seq_after = match &row {
            // A driver `Missing` row carries no admission order, so a degrade
            // event triggered by one is anchored at 0. That is imprecise, not
            // ambiguous: `Event::arrival_seq_after` is still a plain `u64`, and
            // making it `Option` too is a follow-up that also touches
            // events.csv's schema.
            EvRow::Evidence(e) => e.arrival_seq.unwrap_or(0),
            EvRow::Event(ev) => ev.arrival_seq_after,
        };
        let t0 = now();
        let out = self.q.push(row, t0);
        self.blocked_ns
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(now() - t0);
        match out {
            PushOutcome::Accepted { .. } => {}
            PushOutcome::Timeout(env) => {
                let first = !self.degraded.swap(true, Relaxed);
                if first {
                    self.q.set_policy(QueuePolicy::DropOldest);
                }
                if !matches!(
                    self.q.try_push(env.item, now()),
                    PushOutcome::Accepted { .. }
                ) {
                    self.lost.fetch_add(1, Relaxed);
                }
                if first {
                    let ev =
                        Event::recorder_degraded(&ctx.row_ctx(), now(), stage, arrival_seq_after);
                    if !matches!(
                        self.q.try_push(EvRow::Event(ev), now()),
                        PushOutcome::Accepted { .. }
                    ) {
                        self.lost.fetch_add(1, Relaxed);
                    }
                }
            }
            PushOutcome::Evicted(_) | PushOutcome::Rejected(_) | PushOutcome::Closed(_) => {
                self.lost.fetch_add(1, Relaxed);
            }
        }
    }

    pub fn close(&self) {
        self.q.close();
    }

    pub fn queue(&self) -> Arc<BoundedQueue<EvRow>> {
        Arc::clone(&self.q)
    }

    /// Rows that never reached the recorder.
    pub fn lost(&self) -> u64 {
        self.lost.load(Relaxed)
    }

    pub fn degraded(&self) -> bool {
        self.degraded.load(Relaxed)
    }

    /// Duration of every `send` so far (ns), for the `push_blocked_ns p99` line.
    pub fn blocked_ns(&self) -> Vec<i64> {
        self.blocked_ns
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }
}

// The palette groups by MEANING, not by variety, because a dozen entities in
// one plot is unreadable when every colour is arbitrary: a series is the
// colour of what it is about -- the camera's blue, the lidar's teal, the
// detection purple, the answer's gold -- anything that counts a failure is
// red, and a reference bound is thin and grey so it reads as a backdrop
// rather than a measurement.
/// The detection family, #9B7BD8: the lidar detections and tracks on the
/// byte and stage bars, and the fusion's wait for the camera's detections.
const C_DETECTION: [u8; 3] = [155, 123, 216];
/// A failure that has already happened, #E5484D -- and the time to contact,
/// the one number on the demo screen that is bad when it is small.
const C_FAIL: [u8; 3] = [229, 72, 77];
/// A band a measurement is read inside, and the auditor's reference lines:
/// mid grey. The viewer draws a band only in an opaque colour at a width of
/// at least 1.
const C_BOUND: [u8; 3] = [150, 150, 150];
/// A reference line a measurement is read against -- a capacity, a 1.0, the
/// 3 s warning, a sweep's ends, the period: a bright grey, #D2D2D2, because
/// a mid-grey line was invisible on the viewer's black.
const C_REFERENCE: [u8; 3] = [210, 210, 210];
/// Something was missed and the pipeline carried on: a lane's `skipping` or
/// `stale`, and the answer box on a frame that came from a stale pair. `pub`
/// for the box, on the terms [`C_ANSWER`] is.
pub const C_ATTENTION: [u8; 3] = [240, 170, 32];
/// Healthy: a lane's `ok` or `same`, #3C8250 -- a darker green than the
/// fused population's, so a lane of health reads as a backdrop.
const C_LANE_OK: [u8; 3] = [60, 130, 80];
/// Nothing there at all: a lane's `no sweep`, `no frame` or `source gap`, for
/// a frame a sensor's source never had (`absent_in_source`), #808080. Grey,
/// because it is neither the pipeline's health nor its failure -- nothing
/// upstream dropped it -- and a reader must not take the source's gap for a
/// queue's loss. The WARN log and the headline say why.
const C_ABSENT: [u8; 3] = [128, 128, 128];
/// Queue occupancy, measured on the way IN: the depth each push met. Green,
/// because an occupancy is neither a failure nor a timing.
const C_DEPTH_PUSH: [u8; 3] = [92, 178, 120];
/// Queue occupancy measured on the way OUT: the backlog a pop left behind.
/// A lighter voice of the same green, so the two read as one quantity's two
/// edges.
const C_DEPTH_DRAIN: [u8; 3] = [166, 217, 181];
/// The queues nobody is reading for on the fill plot: a slate that recedes
/// behind the camera's blue and the lidar's teal.
const C_NEUTRAL: [u8; 3] = [122, 138, 153];
/// The ANSWER's own colour, gold, and deliberately a colour nothing else in
/// this palette uses: the answer's box, its label and its 3D wireframe, and
/// the two lines that time it, its age in ms and over one period. Every
/// other colour here marks the rest of the PIPELINE -- how late, how full,
/// how many bytes -- or a sensor. Its distance and closing speed are words in
/// its label and the headline rather than lines on a plot.
///
/// `pub` because `consumers` draws the answer's rectangle over the camera
/// image in this colour. Exported rather than repeated: two constants that
/// have to agree are two constants that will one day not, and a test that they
/// still match is a worse guarantee than there being only one of them.
pub const C_ANSWER: [u8; 3] = [255, 208, 48];
// The fusion's three populations, in the dashboard layout's own palette
// (v2 spec: fused #5CB278, lidar #00A8B0, camera #408CD6). Colour means
// PROVENANCE here -- which sensor vouches for the object -- never identity:
// the per-id hues the tracks used to wear said nothing a reader could act on.
/// A track the camera confirmed: green, the one population both sensors see.
pub const C_FUSED: [u8; 3] = [92, 178, 120];
/// A track only the lidar holds: teal, the lidar's colour everywhere.
pub const C_LIDAR_ONLY: [u8; 3] = [0, 168, 176];
/// A detection only the camera holds: blue, the camera's colour everywhere.
pub const C_CAMERA_ONLY: [u8; 3] = [64, 140, 214];

/// How one series is marked on its plot.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mark {
    /// A line. Counts hold their value until the next sample (`step`); a
    /// continuous measurement is interpolated between samples.
    Line { step: bool },
    /// One marker per sample, never joined: a series whose consecutive
    /// samples need not describe the same thing, which a line would join into
    /// a jump that no object made.
    Points(MarkerShape),
}

/// Legend name, colour and mark of one scalar series.
struct SeriesStyle {
    /// The legend entry, complete: the quantity and its unit, and nothing
    /// about which edge it is on -- the view's title says that, and a legend
    /// of eleven "cam0->proc queue wait (ms)"-style entries was unreadable.
    label: &'static str,
    rgb: [u8; 3],
    /// The line's width, or the marker's size.
    width: f32,
    mark: Mark,
}

/// The longest legend entry a plot legend shows without eating the plot.
#[cfg(test)]
const LABEL_MAX: usize = 20;

impl SeriesStyle {
    /// Whether the series is a line that holds its value between samples.
    #[cfg(test)]
    fn step(&self) -> bool {
        self.mark == Mark::Line { step: true }
    }

    /// The static archetype that names, colours and marks one entity's
    /// series: `SeriesLines` for a line, `SeriesPoints` for markers.
    fn archetype(&self) -> Box<dyn AsComponents> {
        let color = Color::from_rgb(self.rgb[0], self.rgb[1], self.rgb[2]);
        match self.mark {
            Mark::Line { step } => Box::new(
                SeriesLines::new()
                    .with_names([self.label])
                    .with_colors([color])
                    .with_widths([self.width])
                    .with_interpolation_mode(if step {
                        InterpolationMode::StepAfter
                    } else {
                        InterpolationMode::Linear
                    }),
            ),
            Mark::Points(marker) => Box::new(
                SeriesPoints::new()
                    .with_names([self.label])
                    .with_colors([color])
                    .with_markers([marker])
                    .with_marker_sizes([self.width]),
            ),
        }
    }
}

/// Every scalar leaf the run can emit. `series_style` must answer for all of
/// them (pinned by a test), so adding a series without giving it a name and a
/// colour fails the build's tests rather than quietly producing another
/// anonymous line in the viewer.
#[cfg(test)]
const SERIES_LEAVES: [&str; 27] = [
    "drops",
    ABSENT_LEAF,
    "depth_at_push",
    "depth",
    "cap",
    "drop_events",
    "fill",
    "run_length",
    "alternating",
    "answer",
    "camera_queue",
    "camera_wait",
    "velo_to_reduce",
    "at_bound",
    "cam_service",
    "answer_age_ms",
    "period_ms",
    // Logged by the `track` stage straight onto the recording, for every
    // sweep: the pairing window is in no evidence column.
    "sweep_start",
    "sweep_end",
    "trigger",
    "camera",
    // The answer's own series. Logged by `state-sink` straight onto the
    // recording rather than mirrored off an evidence row -- an evidence row
    // carries how long a sample took and how big it was, never what it SAID --
    // so they reach this table through `series_archetype` instead of
    // `Dashboard::scalar`. They are in it for the reason everything else is: an
    // unstyled series is an anonymous line in somebody's viewer.
    "ttc_s",
    "ttc_warn",
    // The fusion's populations in the camera frame, logged by `state-sink`
    // off the answer's records and its `camera_only` column, on every answer.
    "fused",
    "lidar_only",
    "camera_only",
    // Beside them, the share of the in-frame lidar tracks that are fused.
    "fused_fraction",
];

/// The static style of a grey band -- a `Measurements` series, whose one
/// standard deviation either side of its value is drawn as a translucent band
/// -- named `name`, in ms. Opaque grey and a 1 px line: the viewer drew
/// nothing for a translucent colour at width 0.
pub fn band_archetype(name: &str) -> Measurements {
    Measurements::update_fields()
        .with_names([name])
        .with_colors([Color::from_rgb(C_BOUND[0], C_BOUND[1], C_BOUND[2])])
        .with_widths([1.0f32])
        .with_units(["ms"])
}

/// The static styling archetype of one scalar leaf, for a stage that logs its
/// own series directly onto the recording instead of through the evidence
/// mirror.
///
/// `None` for a leaf with no style, which is a bug the test over
/// [`SERIES_LEAVES`] exists to catch before it ships as an unnamed grey line.
pub fn series_archetype(leaf: &str) -> Option<Box<dyn AsComponents>> {
    series_style(leaf).map(|s| s.archetype())
}

/// How to draw one scalar leaf, or `None` for a leaf with no style yet.
fn series_style(leaf: &str) -> Option<SeriesStyle> {
    let s = |label, rgb, width, step| {
        Some(SeriesStyle {
            label,
            rgb,
            width,
            mark: Mark::Line { step },
        })
    };
    let points = |label, rgb, size, marker| {
        Some(SeriesStyle {
            label,
            rgb,
            width: size,
            mark: Mark::Points(marker),
        })
    };
    match leaf {
        "drops" => s("drops (cum)", C_FAIL, 2.0, true),
        // Beside the drops, on a sensor driver's edge: the frames its source
        // never had, in the grey of a source gap. Not a drop, so not red.
        ABSENT_LEAF => s("source gap (cum)", C_ABSENT, 2.0, true),
        // The camera queue's depth, both edges of it: the depth each push
        // met shows a queue running full, and the backlog each pop LEFT
        // BEHIND shows it draining -- 0 means that pop emptied the queue. A
        // queue full only for the instant of each push and one full all the
        // time look the same on the push side alone.
        "depth_at_push" => s("depth at push", C_DEPTH_PUSH, 2.0, true),
        "depth" => s("depth after pop", C_DEPTH_DRAIN, 1.5, true),
        "cap" => s("capacity", C_REFERENCE, 1.5, true),
        // A cross at the capacity for every frame the camera queue did not
        // deliver: the queue was full, and this is what it cost.
        "drop_events" => points("dropped (at cap)", C_FAIL, 4.0, MarkerShape::Cross),
        // Depth at push over capacity, per queue, 0 to 1: the one series
        // that shows a queue running full. Coloured by queue at logging
        // time (`fill_rgb`); this is the colour of the rest.
        "fill" => s("fill (share of cap)", C_NEUTRAL, 1.5, true),
        // The admission's run of same-stream driver admissions, against
        // the 1.0 of strict alternation.
        "run_length" => s("run length (count)", C_LIDAR_ONLY, 2.0, true),
        "alternating" => s("alternating (1)", C_REFERENCE, 1.0, true),
        // The Latency page's ratios, each a time over the bound it is read
        // against, so 1 is at the bound on every line: the answer's age in
        // the answer's gold and heaviest, the camera's in its blue, the
        // fusion's wait for the camera in the detection purple, the lidar's
        // in its teal, and the 1.0 they are read against.
        "answer" => s("answer (age/period)", C_ANSWER, 2.5, false),
        "camera_queue" => s("camera (age/bound)", C_CAMERA_ONLY, 1.5, false),
        "camera_wait" => s("cam wait (/period)", C_DETECTION, 1.5, false),
        "velo_to_reduce" => s("lidar (age/period)", C_LIDAR_ONLY, 1.5, false),
        "at_bound" => s("at bound (1)", C_REFERENCE, 1.5, true),
        // The camera stage's service over one period: past 1 it cannot keep
        // up with the camera, whatever its queue does.
        "cam_service" => s("service (/period)", C_CAMERA_ONLY, 2.0, false),
        // The chain's end, on the Latency page, in the answer's gold: how
        // long after the sweep was due its answer arrived, under the one
        // bound that matters there -- the sweep period. Under it, the answer
        // is ready before the next sweep lands.
        "answer_age_ms" => s("answer age (ms)", C_ANSWER, 2.0, false),
        "period_ms" => s("period (ms)", C_REFERENCE, 2.0, true),
        // The pairing window, in ms from the sweep's start: its two ends as
        // bright reference lines (the grey band between them alone is faint
        // on black), the trigger as a thinner one, and the paired camera
        // frame's instant as a POINT per sweep, in the camera's blue. Inside
        // the band it is contemporaneous with the sweep; below it, stale.
        "sweep_start" => s("sweep start (ms)", C_REFERENCE, 1.5, true),
        "sweep_end" => s("sweep end (ms)", C_REFERENCE, 1.5, true),
        "trigger" => s("trigger (ms)", C_REFERENCE, 1.0, true),
        "camera" => points("camera frame (ms)", C_CAMERA_ONLY, 2.5, MarkerShape::Circle),
        // Time to contact with the object the answer flags, as POINTS: the
        // flagged object changes from sweep to sweep, and a line joined one
        // object's approach to the next one's with a vertical jump no object
        // made, where points show each approach as its own diagonal falling
        // through the warning line. Logged only while the object closes, so
        // a receding object is a gap and not a TTC of 0, which reads as a
        // collision. In the warning red: it is the one number on the demo
        // screen a reader should be alarmed by.
        "ttc_s" => points("TTC (s)", C_FAIL, 2.0, MarkerShape::Circle),
        // The line those points are read against: 3 s.
        "ttc_warn" => s("TTC warning (s)", C_REFERENCE, 1.5, true),
        // Counts, so they hold their value: in the frame, how many tracks the
        // camera confirmed, how many only the lidar holds, and how many
        // detections only the camera holds. A stale camera moves the first
        // down and the other two up.
        "fused" => s("fused (count)", C_FUSED, 2.0, true),
        "lidar_only" => s("lidar-only (count)", C_LIDAR_ONLY, 2.0, true),
        "camera_only" => s("camera-only (count)", C_CAMERA_ONLY, 2.0, true),
        // The share of the lidar's in-frame tracks the camera confirmed, in
        // the fused green, held per answer like the counts it divides.
        "fused_fraction" => s("fused (share)", C_FUSED, 2.0, true),
        _ => None,
    }
}

/// The states one lane can show, with a colour each. Logged once, statically,
/// as the lane's `StateConfiguration`, so the viewer draws a green lane for
/// health and a red one for a failure instead of picking colours by first
/// sight.
struct Lane {
    values: &'static [&'static str],
    colors: &'static [[u8; 3]],
}

impl Lane {
    fn archetype(&self) -> StateConfiguration {
        StateConfiguration::new()
            .with_values(self.values.iter().copied())
            .with_labels(self.values.iter().copied())
            .with_colors(
                self.colors
                    .iter()
                    .map(|c| Color::from_rgb(c[0], c[1], c[2])),
            )
    }
}

/// How a queue lane reads: green while every sample gets through, amber
/// (`skipping`) when a delivered sample's predecessor on the edge was never
/// seen, red (`dropping`) when a sample was dropped.
const LANE_QUEUE: Lane = Lane {
    values: &["ok", "skipping", "dropping"],
    colors: &[C_LANE_OK, C_ATTENTION, C_FAIL],
};

/// How the lidar's lane reads: the queue's three states, and grey `no
/// sweep` for a frame the lidar's source never had -- the gap is the source's,
/// not the queue's, and is drawn on the lidar's own lane at its own time.
const LANE_LIDAR: Lane = Lane {
    values: &["ok", "skipping", NO_SWEEP, "dropping"],
    colors: &[C_LANE_OK, C_ATTENTION, C_ABSENT, C_FAIL],
};

/// How the camera's lane reads: the queue's three states, and grey `no
/// frame` for a frame the camera's source never had, on the same terms.
const LANE_CAMERA: Lane = Lane {
    values: &["ok", "skipping", NO_FRAME, "dropping"],
    colors: &[C_LANE_OK, C_ATTENTION, C_ABSENT, C_FAIL],
};

/// How the pairing lane reads: green while the fusion pairs every sweep with
/// its own frame, amber (`stale`) when it took an older frame, red
/// (`expired`) when a sweep's set expired and the answer lost that sweep --
/// `pair_late`, `pair_dropped` or `pair_absent`, which the WARN log names --
/// and grey `source gap` for an instant whose sweep or camera frame a sensor's
/// source never had, so the fusion had nothing to pair.
const LANE_PAIRING: Lane = Lane {
    values: &["ok", "stale", SOURCE_GAP, "expired"],
    colors: &[C_LANE_OK, C_ATTENTION, C_ABSENT, C_FAIL],
};

/// The lidar lane's word for a frame whose sweep is absent in the source.
const NO_SWEEP: &str = "no sweep";

/// The camera lane's word for a frame whose PNG is absent in the source.
const NO_FRAME: &str = "no frame";

/// The pairing lane's word for an instant a gap in either sensor's source
/// left without a partner. Not "absent", which there would read as the
/// camera stream's `pair_absent`.
const SOURCE_GAP: &str = "source gap";

/// Whether a consumer read the producer's buffer or a copy of it.
const LANE_SHARED: Lane = Lane {
    values: &["same", "copied"],
    colors: &[C_LANE_OK, C_FAIL],
};

/// Every lane entity the run can emit, for the test that no lane ships
/// without its `StateConfiguration`.
#[cfg(test)]
const LANE_PATHS: [&str; 6] = [
    "lanes/pairing",
    "lanes/camera_queue",
    "lanes/lidar_queue",
    "lanes/detect_queue",
    "lanes/fusion_queue",
    entity::STORAGE_SHARED,
];

/// How to draw one lane, by its entity path, or `None` for a path with no
/// lane style, which is the bug the test over [`LANE_PATHS`] exists to catch.
fn lane_style(path: &str) -> Option<&'static Lane> {
    if path == lane_path(LANE_PAIRING_NAME) {
        Some(&LANE_PAIRING)
    } else if path == lane_path(LANE_LIDAR_NAME) {
        Some(&LANE_LIDAR)
    } else if path == lane_path(LANE_CAMERA_NAME) {
        Some(&LANE_CAMERA)
    } else if path == entity::STORAGE_SHARED {
        Some(&LANE_SHARED)
    } else if LANE_NAMES
        .iter()
        .any(|n| *n != LANE_PAIRING_NAME && path == lane_path(n))
    {
        Some(&LANE_QUEUE)
    } else {
        None
    }
}

/// What one row says about a lane: `None` when it is clean, or the lane's
/// word for what went wrong and how bad it is (1 amber, 2 red).
type LaneEvent = Option<(&'static str, u8)>;

/// A queue lane's event for one row: `dropping` for any sample the queue did
/// not deliver, `skipping` for a delivered one whose predecessor on the edge
/// was never seen, clean otherwise.
fn queue_event(outcome: Outcome, gap: Option<f64>) -> LaneEvent {
    match outcome {
        Outcome::Delivered if gap.is_some_and(|g| g > 1.0) => Some(("skipping", 1)),
        Outcome::Delivered => None,
        _ => Some(("dropping", 2)),
    }
}

/// The pairing lane's event for the fusion's own row: `reason` is blank on a
/// clean pair, `stale` on a Delivered row paired inside `--pair-stale-ms`,
/// and names how the set expired (`pair_late`, `pair_dropped`,
/// `pair_absent`, `pair_absent_in_source`) on a Missing one -- the last a gap
/// in the camera's source, not a failure of the pipeline.
fn pairing_event(e: &Evidence) -> LaneEvent {
    match (e.outcome, e.reason) {
        (Outcome::Delivered, "") => None,
        (Outcome::Delivered, _) => Some(("stale", 1)),
        (_, "pair_absent_in_source") => Some((SOURCE_GAP, 2)),
        _ => Some(("expired", 2)),
    }
}

/// How many rows without its failure step a lane down, per edge feeding it:
/// a second of them at 10 Hz.
const LANE_CLEAR_AFTER: u32 = 10;

/// How many rows without its failure step the lane `name` down: a second's
/// worth of ITS rows, [`LANE_CLEAR_AFTER`] for each edge that feeds it. The
/// fusion's lane is fed by two, `cam_det->track` and `obj->track`, and at a
/// flat ten it cleared in half a second: on a run whose fusion input
/// dropped a sweep every second and a half it flickered between dropping
/// and skipping in half-second blocks too short to carry their words.
fn lane_clear_after(name: &str, camera_edge: Option<&str>) -> u32 {
    let edges = EDGES
        .iter()
        .filter(|e| queue_lane(e, camera_edge) == Some(name))
        .count()
        .max(1);
    LANE_CLEAR_AFTER * edges as u32
}

/// One sticky lane: a state that is logged only when it changes.
///
/// A lane that logged every sample's word shattered into grey "N states"
/// blocks wherever the words alternated -- a queue dropping every other frame
/// alternates `dropping` and `skipping` -- so the lane holds its worst recent
/// state instead. A failure as bad as the one showing, or worse, sets it
/// (and restarts its count); a lesser one does not. Once the state showing
/// has not recurred for a second's worth of the lane's rows
/// ([`lane_clear_after`]), the lane steps down to the worst lesser failure
/// of those rows, or back to `ok` if they were all clean. So one expired
/// sweep ahead of a run of stale ones reads `expired` for a second and
/// `stale` after it, not `expired` for the whole run.
#[derive(Debug)]
struct Sticky {
    word: &'static str,
    severity: u8,
    /// Rows since the state showing was last set or confirmed.
    quiet: u32,
    /// Rows of quiet that step the state down.
    clear_after: u32,
    /// The worst lesser failure seen in those rows.
    lesser: Option<(u8, &'static str)>,
    started: bool,
}

impl Default for Sticky {
    fn default() -> Self {
        Sticky::new(LANE_CLEAR_AFTER)
    }
}

impl Sticky {
    fn new(clear_after: u32) -> Self {
        Sticky {
            word: "ok",
            severity: 0,
            quiet: 0,
            clear_after,
            lesser: None,
            started: false,
        }
    }

    /// Feeds one row's event; returns the word to log when the lane's state
    /// changed, or on the lane's first row.
    fn observe(&mut self, event: LaneEvent) -> Option<&'static str> {
        let before = self.word;
        match event {
            Some((word, severity)) if severity >= self.severity => {
                self.word = word;
                self.severity = severity;
                self.quiet = 0;
                self.lesser = None;
            }
            _ => {
                if let Some((word, severity)) = event {
                    if self.lesser.is_none_or(|(s, _)| severity >= s) {
                        self.lesser = Some((severity, word));
                    }
                }
                self.quiet += 1;
                if self.severity > 0 && self.quiet >= self.clear_after {
                    let (severity, word) = self.lesser.take().unwrap_or((0, "ok"));
                    self.word = word;
                    self.severity = severity;
                    self.quiet = 0;
                }
            }
        }
        let first = !self.started;
        self.started = true;
        (first || before != self.word).then_some(self.word)
    }
}

/// The admission's run of same-stream sensor-driver admissions.
#[derive(Default)]
struct AdmissionRun {
    last: Option<&'static str>,
    run: u32,
}

impl AdmissionRun {
    /// Records one driver admission and returns how many in a row, this one
    /// included, came from the same driver.
    fn observe(&mut self, stream: &'static str) -> u32 {
        self.run = if self.last == Some(stream) {
            self.run + 1
        } else {
            1
        };
        self.last = Some(stream);
        self.run
    }
}

/// The colour of one queue's fill line, by the queue it hangs under: the
/// camera queue that feeds the answer in the camera's blue, the lidar's
/// queue into `reduce` in its teal, and every other queue in one neutral
/// slate, so the two a reader came for stand out of twelve.
fn fill_rgb(root: &str, camera_edge: Option<&str>) -> [u8; 3] {
    if camera_edge.is_some_and(|e| root == queue_root(e)) {
        C_CAMERA_ONLY
    } else if root == queue_root("velo->reduce") {
        C_LIDAR_ONLY
    } else {
        C_NEUTRAL
    }
}

/// One period on the host clock, ms, for a row: the sweep's own range on a
/// sweep-derived row, the camera's median period on a camera row (an
/// instant has no range), either divided by `--rate`. `None` on an unpaced
/// run, which has no deadlines to be late against.
fn host_period_ms(bounds: &Bounds, e: &Evidence) -> Option<f64> {
    let rate = bounds.rate?;
    let sensor_ms = if e.tov_end_ns > e.tov_start_ns {
        (e.tov_end_ns - e.tov_start_ns) as f64 * 1e-6
    } else {
        bounds.period_ms?
    };
    (rate > 0.0).then(|| sensor_ms / rate)
}

/// The answer-age plot's band: one period, 0 to P.
const PERIOD_BAND: &str = "latency/period_band";

/// One stage bar's colour, by its place in the chain: the lidar's teal for
/// the two stages that read clouds, the detection purple for the two that
/// read detections and tracks, the answer's gold for the last.
fn stage_rgb(i: usize) -> [u8; 3] {
    match i {
        0 | 1 => C_LIDAR_ONLY,
        2 | 3 => C_DETECTION,
        _ => C_ANSWER,
    }
}

/// The edges the byte table reads: the chain's five links and the lidar
/// driver's own row, the one copy.
const TABLE_EDGES: [&str; 6] = [
    "velo->reduce",
    "det->cloud",
    "obj->sink",
    "track->state",
    "state->sink",
    "velo",
];

/// Answers after which the byte table is first written, so a live viewer
/// has one to read long before the run ends.
const TABLE_AFTER_ANSWERS: u64 = 10;

/// The pipeline graph's node radius, in the graph's own units.
const GRAPH_NODE_RADIUS: f32 = 6.0;

/// One row of the byte table: a link's median carried and allocated bytes,
/// `None` where the run had no such row, and how many rows the medians are
/// over -- the samples that link delivered, which on a run that evicted is
/// fewer than the sweeps.
struct ByteLink<'a> {
    name: &'a str,
    carried: Option<u64>,
    allocated: Option<u64>,
    rows: usize,
}

/// A byte count the way the table and the graph print it: `1.97 MB`,
/// `518 kB`, `341 B`, in decimal units.
fn human_bytes(b: u64) -> String {
    let f = b as f64;
    if f >= 1e6 {
        format!("{:.2} MB", f / 1e6)
    } else if f >= 1e3 {
        format!("{:.0} kB", f / 1e3)
    } else {
        format!("{b} B")
    }
}

/// The byte chain as a markdown table -- each link, what it carried, the
/// step from the link before (a shrink as `÷ 3.8`, a growth as `× 1.4`),
/// and what the link's consumer allocated to read it -- followed by one
/// sentence about the lidar driver, the chain's one copy: `driver` is its
/// `(allocated, carried)` medians over `driver_rows` sweeps. The last
/// sentence says what the medians are over: one number when every link
/// delivered the same sweeps, each link's own count when they did not, so
/// a run that evicted does not read as a median of sweeps a link never saw.
fn byte_table(
    links: &[ByteLink],
    driver: (Option<u64>, Option<u64>),
    driver_rows: usize,
) -> String {
    let show = |b: Option<u64>| b.map_or_else(|| "-".to_string(), human_bytes);
    let mut out =
        String::from("| link | carried | step | allocated on the edge |\n|---|---:|---:|---:|\n");
    let mut prev: Option<u64> = None;
    for (i, l) in links.iter().enumerate() {
        let step = match (prev, l.carried) {
            (Some(a), Some(b)) if a > 0 && b > 0 && b <= a => {
                format!("÷ {:.1}", a as f64 / b as f64)
            }
            (Some(a), Some(b)) if a > 0 && b > a => format!("× {:.1}", b as f64 / a as f64),
            _ => String::new(),
        };
        out.push_str(&format!(
            "| {} {} | **{}** | {} | {} |\n",
            i + 1,
            l.name,
            show(l.carried),
            step,
            show(l.allocated)
        ));
        prev = l.carried;
    }
    let in_place = links.iter().all(|l| l.allocated == Some(0));
    out.push('\n');
    match driver {
        (Some(alloc), Some(carried)) if in_place => out.push_str(&format!(
            "Every link reads its input in place: the lidar driver's {} allocated for {} carried is the chain's one copy.",
            human_bytes(alloc),
            human_bytes(carried)
        )),
        (Some(alloc), Some(carried)) => out.push_str(&format!(
            "The lidar driver allocated {} for {} carried.",
            human_bytes(alloc),
            human_bytes(carried)
        )),
        _ => {}
    }
    let rows: Vec<usize> = links.iter().map(|l| l.rows).collect();
    if rows.iter().all(|&n| n == driver_rows) {
        out.push_str(&format!(" Medians of {driver_rows} sweeps."));
    } else {
        let each: Vec<String> = rows.iter().map(usize::to_string).collect();
        out.push_str(&format!(
            " Medians of each link's own rows: {} for links 1 to {}, of the driver's {driver_rows} sweeps.",
            each.join(", "),
            rows.len()
        ));
    }
    out
}

/// Allocated over carried, 0 for an edge that carried nothing.
fn alloc_ratio(allocated: u64, carried: u64) -> f64 {
    if carried == 0 {
        0.0
    } else {
        allocated as f64 / carried as f64
    }
}

/// An edge's colour on the byte charts, by who produced what it carries:
/// the lidar's teal for the sweep and the voxels, the detection purple for
/// the detections, the camera's blue for a camera frame.
fn producer_rgb(edge: &str, camera_edge: Option<&str>) -> [u8; 3] {
    if edge.starts_with("cam0") || camera_edge == Some(edge) {
        C_CAMERA_ONLY
    } else if edge.starts_with("obj") || edge.starts_with("track") {
        C_DETECTION
    } else if edge.starts_with("state") {
        C_ANSWER
    } else {
        C_LIDAR_ONLY
    }
}

/// `log10` of a byte count, with 0 bytes drawn as 0 rather than as minus
/// infinity: the bar charts span four orders of magnitude, and the viewer
/// has no log axis.
fn log10_bytes(bytes: u64) -> f64 {
    (bytes.max(1) as f64).log10()
}

/// Cumulative non-`Delivered` outcomes per edge, so the dashboard can draw a
/// step function instead of isolated points -- and, apart from them, the
/// frames a sensor's source never had.
///
/// A frame absent in the source is a `Missing` row on its driver's
/// pseudo-edge, and it is not a drop: nothing in the pipeline lost it. Counted
/// in the drops, drive 0009's four absent sweeps drew the lidar driver's red
/// drops line up to 4 on a run that lost nothing.
#[derive(Default)]
struct DropCounter(BTreeMap<&'static str, (u64, u64)>);

impl DropCounter {
    /// Records one row and returns that edge's running `(drops, absent in
    /// source)`.
    fn observe(&mut self, edge: &'static str, outcome: Outcome, reason: &str) -> (u64, u64) {
        let n = self.0.entry(edge).or_insert((0, 0));
        if outcome != Outcome::Delivered {
            if reason == ABSENT_IN_SOURCE {
                n.1 += 1;
            } else {
                n.0 += 1;
            }
        }
        *n
    }
}

/// A sensor whose source can lack a frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Sensor {
    Camera,
    Lidar,
}

/// The sensors whose frame numbers a stream's `seq` is: the sensor's own
/// stream, and the streams derived from it that carry the source frame's
/// number. A frame absent in the source of one of them is a frame the stream
/// never had to deliver, on every edge it crosses.
fn sensors_of(stream: u8) -> &'static [Sensor] {
    match StreamId(stream) {
        StreamId::CAM0 | StreamId::CAM_DET => &[Sensor::Camera],
        StreamId::LIDAR | StreamId::LIDAR_DET | StreamId::LIDAR_OBJ => &[Sensor::Lidar],
        // The fusion's output and the answer: an instant either sensor's
        // source lacks makes no set.
        StreamId::TRACKS | StreamId::EGO => &[Sensor::Camera, Sensor::Lidar],
        _ => &[],
    }
}

/// Previous *delivered* `seq` per edge.
#[derive(Default)]
struct SeqGap(BTreeMap<&'static str, u64>);

impl SeqGap {
    /// Records one delivered `seq` and returns how many frames of sequence
    /// space it advanced past the previous delivery on that edge: 1 when the
    /// consumer kept up, 3 when it never saw two frames in between.
    ///
    /// `None` for the first delivery on an edge, which has no predecessor to
    /// measure against. That row is skipped rather than logged as 1, which
    /// would claim "nothing was missed" on evidence we do not have, or as 0,
    /// which reads as a stall.
    ///
    /// The gap is computed in `f64` rather than `u64` so that a seq that did
    /// not advance shows up as a visible zero or negative rather than
    /// wrapping into a huge number.
    fn observe(&mut self, edge: &'static str, seq: u64) -> Option<f64> {
        self.0
            .insert(edge, seq)
            .map(|prev| seq as f64 - prev as f64)
    }
}

/// Constant reference lines the dashboard draws beside the measured series, so
/// a glance answers "is this number good?" instead of only "what is it?".
/// Computed in `run.rs` from the run's own arguments — the dashboard only
/// draws them and never measures anything itself.
#[derive(Clone, Debug, Default)]
pub struct Bounds {
    /// The *queueing* half of the measurement-age bound per bounded edge
    /// (`cam0->proc`, `cam0->rerun`, `cam0->camdet`), in ms: `cap × frame
    /// period`. The
    /// frame's own service time is added when the line is drawn, from
    /// `proc_end - proc_start` on that row.
    ///
    /// The pre-registered statement bounds age by `cap × period + service`, and
    /// service is measured rather than configured. An earlier version used
    /// `--consumer-delay-ms` in place of it, which was indistinguishable while
    /// `proc` was a ~0.5 ms grayscale pass. With a heavier ~25 ms workload it
    /// drew a target that a *healthy* run crossed on roughly a quarter of its
    /// frames -- the line said the pipeline was late when nothing had been
    /// missed. The artificial `--consumer-delay-ms` sleep happens between
    /// `proc_start` and `proc_end` of the stage it slows (`camdet` with the
    /// detector, `proc` without), so it is already inside the measured
    /// service and must not be added twice.
    ///
    /// Empty for an unpaced run (`--rate inf`), which has no deadlines.
    pub age_queue_ms: BTreeMap<&'static str, f64>,
    /// Each real queue's CAPACITY, in items, keyed by edge name.
    ///
    /// Drawn as a flat reference line under the same entity as that edge's two
    /// depth series, because a depth on its own does not answer the question
    /// anyone actually has. Depth 2 is idle on a 16-deep queue and saturated on
    /// a 2-deep one, and this pipeline runs both at once: `--cap` sets the
    /// camera queue that feeds the answer (`cam0->camdet` with the detector,
    /// `cam0->proc` without), while the lidar edges are fixed at [`crate::run::VELO_CAP`]
    /// and [`crate::run::DET_CAP`] precisely so the camera experiment's
    /// independent variable cannot reach them.
    ///
    /// An edge with no rows draws no line, so an entry here for a queue this
    /// run never opened (`cam0->rerun` under `--rerun off`) costs nothing and
    /// claims nothing.
    pub queue_cap: BTreeMap<&'static str, f64>,
    /// The camera frame's width and height in pixels, from frame 0's PNG
    /// header: the camera view's bounds, so the picture fills its view edge
    /// to edge from the first frame instead of fitting whatever was drawn
    /// first. `None` when the header could not be read, and the view then
    /// fits the image the viewer's own way.
    pub image_wh: Option<(f32, f32)>,
    /// One sensor period in ms on the SENSOR clock: the camera's median
    /// frame period, which on these drives is the lidar's sweep period too
    /// (both run at 10 Hz). The axes of the plots drawn in sensor time -- the
    /// pairing window -- are scaled by it. The driver's documented default
    /// period stands in on a drive with fewer than two frames; `None` only
    /// in a `Bounds` built without a drive.
    pub period_ms: Option<f64>,
    /// The camera queue whose output reaches the answer, where the camera
    /// knobs act (`run::camera_queues`): `cam0->camdet` with the detector,
    /// `cam0->proc` without. The one queue drawn in items, and the one the
    /// `camera_queue` lane watches.
    pub camera_edge: Option<&'static str>,
    /// Frames the camera driver will replay: the most any queue can drop,
    /// and so the top of the cumulative-drops plot.
    pub n_frames: Option<f64>,
    /// `--rate`, when it is finite: a sensor period divided by it is the
    /// host period the latencies are read against. `None` on an unpaced
    /// run, which has no deadlines.
    pub rate: Option<f64>,
    /// Whether the run has `velo->reduce`, the edge the lidar lane watches.
    /// Only then does a frame absent in the source mark that lane: on a run
    /// without it no row would ever step the lane back down, and `no sweep`
    /// would stand to the end of the recording.
    pub lidar_lane: bool,
    /// Whether the chain ends in an answer (`track` runs). Only then does a
    /// frame absent in the source put its own headline up and mark the
    /// pairing lane, for the same reason: a run with no answer has nothing
    /// to replace them afterwards.
    pub answer: bool,
}

/// The live dashboard (M8): mirrors evidence rows onto the run's Rerun
/// recording as scalars on the same three timelines the image uses, so a frame
/// and its timing scrub together. Constructed only with `--dashboard`.
///
/// No legend document travels with the recording: the entity names and the
/// series names have to carry their own meaning, and a reader who needs more
/// has the README ("The dashboard") rather than a panel in the viewer.
pub struct Dashboard {
    rec: Canvas,
    /// The sample of the last row drawn. An event is drawn at that row's
    /// timelines but `host` ([`Dashboard::log_event`]), and so is the byte
    /// table at the end, so their drawings are filed under it.
    last: Subject,
    /// Whether [`Dashboard::finish`] has run.
    finished: bool,
    drops: DropCounter,
    gaps: SeqGap,
    bounds: Bounds,
    /// Entity paths that already carry their static `SeriesLines` styling.
    /// The set of edges and stages is not known until rows arrive, so styling
    /// is registered on each path's first scalar rather than up front.
    styled: BTreeSet<String>,
    /// Lane entities that already carry their static `StateConfiguration`.
    lanes: BTreeSet<String>,
    /// Each sticky lane's state, by lane name ([`crate::dashboard::LANE_NAMES`]).
    sticky: BTreeMap<&'static str, Sticky>,
    /// The admission's current run of same-stream driver admissions.
    run: AdmissionRun,
    /// The latest delivered `(payload_bytes, bytes_alloc)` per edge and per
    /// producer, for the byte chain and the graph. A bar chart is one
    /// picture per sweep, and the rows that feed it arrive one edge at a
    /// time, so each is redrawn from the latest value on every edge.
    bytes: BTreeMap<&'static str, (u64, u64)>,
    /// The latest `proc_end - due` per edge, in ms, for the stage bars.
    after_due_ms: BTreeMap<&'static str, f64>,
    /// Every delivered `(payload_bytes, bytes_alloc)` on the byte table's
    /// edges, for its medians: a few hundred pairs a run.
    medians: BTreeMap<&'static str, Vec<(u64, u64)>>,
    /// Answers seen, so the byte table is first written after
    /// [`TABLE_AFTER_ANSWERS`].
    answers: u64,
    /// The storage lane's last word, so it is logged only when it changes.
    shared: Option<&'static str>,
    /// Whether the pipeline graph's edges are logged yet.
    graphed: bool,
    /// Each producer row's `storage_id`, by `(stream, seq)`, so a consumer's
    /// row can say whether it read that buffer or a copy. Grows with the run
    /// at 16 bytes a sample, like the rows the `rec` thread already keeps.
    produced: BTreeMap<(u8, u64), usize>,
    /// Frames the source has no sample for, by `(sensor, frame)`, so a
    /// queue's skip over them -- the lidar edges go from sweep 176 to 181 on
    /// drive 0009 -- is not drawn as the queue's own loss, on the sensor's
    /// own edges or on any edge of a stream that carries its frame numbers
    /// ([`sensors_of`]).
    absent: BTreeSet<(Sensor, u64)>,
}

impl Dashboard {
    /// Sends the layout first, before any row and before the clock starts,
    /// straight to the recording: a viewer the run spawned opens straight
    /// into it, and a file carries it at its head. Every row after it is
    /// drawn on `canvas`.
    pub fn new(
        rec: &RecordingStream,
        canvas: Canvas,
        bounds: Bounds,
        mode: Mode,
    ) -> Result<Self, rerun::RecordingStreamError> {
        send_blueprint(rec, mode, &bounds)?;
        Ok(Dashboard {
            rec: canvas,
            last: Subject::default(),
            finished: false,
            drops: DropCounter::default(),
            gaps: SeqGap::default(),
            bounds,
            styled: BTreeSet::new(),
            lanes: BTreeSet::new(),
            sticky: BTreeMap::new(),
            run: AdmissionRun::default(),
            bytes: BTreeMap::new(),
            after_due_ms: BTreeMap::new(),
            medians: BTreeMap::new(),
            answers: 0,
            shared: None,
            graphed: false,
            produced: BTreeMap::new(),
            absent: BTreeSet::new(),
        })
    }

    /// A frame a sensor's source has no sample for (`absent_in_source`), at
    /// the slot's own place on `host`, where the replay reported it.
    ///
    /// The sensor's own lane turns grey -- `no sweep` on the lidar's, `no
    /// frame` on the camera's -- and its pictures of the sample before are
    /// cleared, so an old cloud or frame does not stand in for an instant it
    /// does not show. For a missing sweep, with an answer, the pairing lane
    /// turns grey `source gap` too and the answer's own pictures and headline
    /// say there is none for this frame, and why ([`log_no_sweep`]); for a
    /// missing camera frame the sweep of that number reaches the fusion and
    /// expires as `pair_absent_in_source`, which marks the pairing lane and
    /// draws the headline itself. The frame's WARN row, and the run's one
    /// `SourceGap` event naming the whole gap, are logged where every drop and
    /// event is.
    fn log_absent(&mut self, e: &Evidence) -> Result<(), RecError> {
        let sensor = if e.edge == "cam0" {
            Sensor::Camera
        } else {
            Sensor::Lidar
        };
        self.absent.insert((sensor, e.seq));
        if e.edge == "cam0" {
            self.sticky(LANE_CAMERA_NAME, Some((NO_FRAME, 2)))?;
            log_no_frame(&self.rec, e.seq, self.bounds.answer)?;
            return Ok(());
        }
        if self.bounds.lidar_lane {
            self.sticky(LANE_LIDAR_NAME, Some((NO_SWEEP, 2)))?;
        }
        for path in [
            entity::LIDAR_SWEEP,
            entity::LIDAR_VOXELS,
            entity::LIDAR_DETECTIONS,
            entity::LIDAR_TRACKS,
        ] {
            self.rec.log(path, &Clear::flat())?;
        }
        if self.bounds.answer {
            self.sticky(LANE_PAIRING_NAME, Some((SOURCE_GAP, 2)))?;
            log_no_sweep(&self.rec, e.seq)?;
        }
        Ok(())
    }

    /// How many frames absent in the source a delivery on `e`'s stream
    /// stepped over since the one before it, `gap` sequence numbers back: the
    /// frames in between that the source of any sensor the stream's `seq`
    /// counts never had. By sensor, not by stream: `cam_det->track` carries
    /// the camera's frame numbers on its own stream id, and a camera frame
    /// absent in the source is as absent there as on `cam0`'s own edges.
    fn absent_between(&self, e: &Evidence, gap: f64) -> f64 {
        if gap <= 1.0 {
            return 0.0;
        }
        let prev = e.seq.saturating_sub(gap as u64);
        let frames: BTreeSet<u64> = sensors_of(e.stream)
            .iter()
            .flat_map(|&s| {
                self.absent
                    .range((s, prev + 1)..(s, e.seq))
                    .map(|&(_, f)| f)
            })
            .collect();
        frames.len() as f64
    }

    /// Logs one scalar at `<root>/<leaf>`, naming and colouring that entity
    /// the first time it is seen.
    fn scalar(&mut self, root: &str, leaf: &str, v: f64) -> Result<(), RecError> {
        let path = format!("{root}/{leaf}");
        if !self.styled.contains(&path) {
            if let Some(mut style) = series_style(leaf) {
                if leaf == "fill" {
                    style.rgb = fill_rgb(root, self.bounds.camera_edge);
                }
                self.rec
                    .log_static(path.clone(), style.archetype().as_ref())?;
            }
            self.styled.insert(path.clone());
        }
        self.rec.log(path, &Scalars::single(v))?;
        Ok(())
    }

    /// Logs one state onto a lane, configuring the lane's states and colours
    /// the first time it is seen.
    fn lane(&mut self, path: &str, state: &str) -> Result<(), RecError> {
        if !self.lanes.contains(path) {
            if let Some(lane) = lane_style(path) {
                self.rec.log_static(path.to_string(), &lane.archetype())?;
            }
            self.lanes.insert(path.to_string());
        }
        self.rec
            .log(path.to_string(), &StateChange::single(state))?;
        Ok(())
    }

    /// Logs one grey band at `path` spanning 0 to `full`: a `Measurements`
    /// of half of it whose standard deviation is that half. Styled on first
    /// sight like every series.
    fn band(&mut self, path: &str, name: &str, full: f64) -> Result<(), RecError> {
        if !self.styled.contains(path) {
            self.rec
                .log_static(path.to_string(), &band_archetype(name))?;
            self.styled.insert(path.to_string());
        }
        let half = 0.5 * full;
        self.rec.log(
            path.to_string(),
            &Measurements::new([half]).with_variances([half * half]),
        )?;
        Ok(())
    }

    /// The Latency page's ratios for one delivered queue row, each a
    /// measured time over the bound it is read against, so 1 is AT the
    /// bound on every line and they share one axis. Drawn at
    /// [`HEADROOM_TOP`] when they are that or more, so an overloaded run
    /// pins the top edge instead of leaving the plot.
    ///
    /// - the camera queue that feeds the answer: its frame's age over the
    ///   pre-registered bound, `cap x period + service`, the service being
    ///   this frame's own `proc_end - proc_start`, so a stage that does real
    ///   work is bounded by what it actually cost (and the bound is not
    ///   drawn without both timestamps -- a wrong line is worse than none);
    ///   and the service alone over one period, the detector's leading
    ///   indicator: past 1 it cannot keep up with the camera;
    /// - the lidar's queue into `reduce`: age over one period;
    /// - the fusion's camera input, `cam_det->track`: how long the camera's
    ///   half waited in its queue for the fusion, over one period.
    fn log_headroom(&mut self, e: &Evidence) -> Result<(), RecError> {
        let Some(p) = host_period_ms(&self.bounds, e) else {
            return Ok(());
        };
        let ratio = |v: f64, bound: f64| (v / bound).min(HEADROOM_TOP);
        let age_ms = e.measurement_age_ns.map(|a| a as f64 * 1e-6);
        let service_ms = e
            .proc_start_ns
            .zip(e.proc_end_ns)
            .map(|(ps, pe)| (pe - ps) as f64 * 1e-6);
        if self.bounds.camera_edge == Some(e.edge) {
            if let (Some(age), Some(svc), Some(queue_ms)) = (
                age_ms,
                service_ms,
                self.bounds.age_queue_ms.get(e.edge).copied(),
            ) {
                self.scalar(entity::HEADROOM, "camera_queue", ratio(age, queue_ms + svc))?;
            }
            if let Some(svc) = service_ms {
                self.scalar(entity::LATENCY, "cam_service", ratio(svc, p))?;
            }
        }
        match (e.edge, age_ms, e.queue_wait_ns) {
            ("velo->reduce", Some(age), _) => {
                self.scalar(entity::HEADROOM, "velo_to_reduce", ratio(age, p))?;
            }
            ("cam_det->track", _, Some(wait)) => {
                self.scalar(
                    entity::HEADROOM,
                    "camera_wait",
                    ratio(wait as f64 * 1e-6, p),
                )?;
            }
            ("state->sink", Some(age), _) => {
                self.scalar(entity::HEADROOM, "answer", ratio(age, p))?;
                self.scalar(entity::HEADROOM, "at_bound", 1.0)?;
            }
            _ => {}
        }
        Ok(())
    }

    /// Feeds one row's event to the sticky lane `name`, and logs the lane's
    /// state only when it changed (or on its first row).
    fn sticky(&mut self, name: &'static str, event: LaneEvent) -> Result<(), RecError> {
        let camera_edge = self.bounds.camera_edge;
        let changed = self
            .sticky
            .entry(name)
            .or_insert_with(|| Sticky::new(lane_clear_after(name, camera_edge)))
            .observe(event);
        if let Some(word) = changed {
            self.lane(&lane_path(name), word)?;
        }
        Ok(())
    }

    /// Logs one entity of a bar chart: `values` at `abscissa`, in one colour.
    ///
    /// A chart is several such entities under one root, one per bar,
    /// because a bar carries no label of its own and an entity has one
    /// colour. The viewer names a hovered bar by its entity path, and two
    /// series in one chart can alternate.
    fn bars(
        &self,
        path: String,
        abscissa: Vec<f64>,
        values: Vec<f64>,
        rgb: [u8; 3],
    ) -> Result<(), RecError> {
        self.rec.log(
            path,
            &BarChart::new(values)
                .with_abscissa(abscissa)
                .with_color(Color::from_rgb(rgb[0], rgb[1], rgb[2])),
        )?;
        Ok(())
    }

    /// The allocation chart's bar for one edge, from the delivered row that
    /// just arrived: what the edge's consumer allocated over what the edge
    /// carried. Each bar is its own entity at its own place, so only the bar
    /// whose row arrived is logged; the others keep their latest.
    fn log_alloc_ratio(&self, e: &Evidence) -> Result<(), RecError> {
        if let Some(i) = ALLOC_RATIO.iter().position(|(_, edge)| *edge == e.edge) {
            let (name, edge) = ALLOC_RATIO[i];
            self.bars(
                format!("{}/{name}", entity::BYTES_ALLOC_RATIO),
                vec![i as f64],
                vec![alloc_ratio(e.bytes_alloc, e.payload_bytes)],
                producer_rgb(edge, self.bounds.camera_edge),
            )?;
        }
        Ok(())
    }

    /// The pipeline graph for the answer that just arrived: its edges once,
    /// statically, then its nodes at this answer's timelines, each labelled
    /// with what its stage handed on for the latest sweep and coloured by
    /// the sensor or family it belongs to.
    fn log_graph(&mut self, e: &Evidence) -> Result<(), RecError> {
        if !self.graphed {
            self.rec.log_static(
                entity::GRAPH_PIPELINE,
                &GraphEdges::new(GRAPH_EDGES).with_directed_edges(),
            )?;
            self.graphed = true;
        }
        let carried = |edge: &str| self.bytes.get(edge).map(|b| human_bytes(b.0));
        let camera_stage = self
            .bounds
            .camera_edge
            .and_then(|e| e.split_once("->"))
            .map_or("camera", |(_, stage)| stage);
        let label = |id: &str| -> String {
            let with = |what: &str, edge: &str| match carried(edge) {
                Some(b) => format!("{what} {b}"),
                None => what.to_string(),
            };
            match id {
                "cam0" => with("camera", "cam0"),
                "velo" => with("lidar", "velo"),
                "admit" => "admission".to_string(),
                "camera" => with(camera_stage, "cam_det"),
                "reduce" => with("voxels", "det"),
                "detect" => with("dets", "obj"),
                "track" => with("tracks", "track"),
                "state" => with("state", "state"),
                _ => match e.measurement_age_ns {
                    Some(age) => format!("answer {:.0} ms", age as f64 * 1e-6),
                    None => "answer".to_string(),
                },
            }
        };
        let rgb = |id: &str| match id {
            "cam0" | "camera" => C_CAMERA_ONLY,
            "velo" | "reduce" => C_LIDAR_ONLY,
            "detect" | "track" | "state" => C_DETECTION,
            "answer" => C_ANSWER,
            _ => C_BOUND,
        };
        let nodes = GraphNodes::new(GRAPH_NODES.map(|(id, _, _)| id))
            .with_positions(GRAPH_NODES.map(|(_, x, y)| [x, y]))
            .with_labels(GRAPH_NODES.map(|(id, _, _)| label(id)))
            .with_colors(GRAPH_NODES.map(|(id, _, _)| {
                let c = rgb(id);
                Color::from_rgb(c[0], c[1], c[2])
            }))
            .with_radii([GRAPH_NODE_RADIUS; GRAPH_NODES.len()])
            .with_show_labels(true);
        self.rec.log(entity::GRAPH_PIPELINE, &nodes)?;
        Ok(())
    }

    /// The byte table, from the medians of every delivered row so far,
    /// logged statically so the last write is the one a reader sees.
    fn log_byte_table(&self) -> Result<(), RecError> {
        let median = |edge: &str, pick: fn(&(u64, u64)) -> u64| {
            self.medians.get(edge).and_then(|rows| {
                let mut v: Vec<u64> = rows.iter().map(pick).collect();
                v.sort_unstable();
                v.get(v.len() / 2).copied()
            })
        };
        let links: Vec<ByteLink> = CHAIN_BYTES
            .iter()
            .map(|(name, edge)| ByteLink {
                name: &name[2..],
                carried: median(edge, |r| r.0),
                allocated: median(edge, |r| r.1),
                rows: self.medians.get(edge).map_or(0, Vec::len),
            })
            .collect();
        let driver = (median("velo", |r| r.1), median("velo", |r| r.0));
        let driver_rows = self.medians.get("velo").map_or(0, Vec::len);
        self.rec.log_static(
            entity::BYTES_TABLE,
            &TextDocument::new(byte_table(&links, driver, driver_rows))
                .with_media_type(MediaType::markdown()),
        )?;
        Ok(())
    }

    /// At the end of the run, once every stage has stopped: the byte table
    /// from the whole run's medians. The dashboard draws last, so it then
    /// closes the viewer's queue, and the viewer thread drains it and stops.
    /// Runs once.
    fn finish(&mut self) -> Result<(), RecError> {
        if self.finished {
            return Ok(());
        }
        self.finished = true;
        let drawn = if self.answers > 0 {
            self.log_byte_table()
        } else {
            Ok(())
        };
        self.rec.send(self.last.clone());
        self.rec.close();
        drawn
    }

    /// One evidence row, drawn as one drawing of the row's sample.
    ///
    /// Not the viewer's own rows (`draw->viewer`): drawn, each would be a
    /// drawing, and each drawing another row.
    fn log_row(&mut self, e: &Evidence) -> Result<(), RecError> {
        if e.edge == VIEWER_EDGE {
            return Ok(());
        }
        let drawn = self.draw_row(e);
        self.last = Subject::of_row(e);
        self.rec.send(self.last.clone());
        drawn
    }

    /// Stamps the three timelines from the row itself, then logs whichever
    /// series that row carries. Everything a queue did is keyed by EDGE
    /// (`queues/<edge>/`); the lanes, the ratios and the bars each read the
    /// edges they are about.
    fn draw_row(&mut self, e: &Evidence) -> Result<(), RecError> {
        // A row with no time of validity -- a frame absent in the source,
        // whose `tov_*_ns` are 0 because the source measured nothing -- has
        // no place on the sensor clock, and stamped at 0 it would put a point
        // in 1970 and stretch that timeline over forty years. It is logged on
        // `host` and `seq` only.
        if e.tov_start_ns == 0 && e.tov_end_ns == 0 {
            self.rec.disable_timeline("sensor_time");
        } else {
            self.rec
                .set_timestamp_nanos_since_epoch("sensor_time", e.tov_start_ns);
        }
        self.rec
            .set_duration_secs("host", e.arrival_ns as f64 * 1e-9);
        self.rec.set_time_sequence("seq", e.seq as i64);

        let real_queue = e.edge.contains("->");
        let delivered = e.outcome == Outcome::Delivered;
        let queues = queue_root(e.edge);

        // Logged on every row so the series is a step function, not a scatter.
        // Only for the edges that can actually drop something: the real
        // queues, and the two sensor drivers' pseudo-edges, whose `Missing`
        // rows are frames the driver never produced. A derived stream's
        // producer row (`det`, `obj`, `track`, `state`, `cam_det`) is
        // structurally always Delivered, and drawing a flat zero for each of
        // them said nothing. A frame the source never had is not a drop: it
        // is counted apart, in the grey of a source gap, on its driver's edge
        // and only from the first one, so a drive with none draws nothing.
        let (drops, absent) = self.drops.observe(e.edge, e.outcome, e.reason);
        if real_queue || ADMISSION_EDGES.contains(&e.edge) {
            self.scalar(&queues, "drops", drops as f64)?;
            if absent > 0 {
                self.scalar(&queues, ABSENT_LEAF, absent as f64)?;
            }
        }

        if real_queue {
            let cap = self.bounds.queue_cap.get(e.edge).copied();
            // How full each push found the queue, as a share of its
            // capacity, on every queue: 1 is a push that met a full queue.
            if let (Some(depth), Some(cap)) = (e.depth_at_push, cap) {
                if cap > 0.0 {
                    self.scalar(&queues, "fill", f64::from(depth) / cap)?;
                }
            }
            // The camera queue that feeds the answer, in items: the depth
            // each push met, the backlog each pop LEFT BEHIND (`Some(0)` is
            // a pop that emptied it), the capacity line, on every row of the
            // edge so it spans exactly as long as the edge was alive, and a
            // cross at the capacity for every frame it did not deliver.
            if self.bounds.camera_edge == Some(e.edge) {
                if let Some(depth) = e.depth_at_push {
                    self.scalar(&queues, "depth_at_push", f64::from(depth))?;
                }
                if let Some(depth) = e.depth_after_pop {
                    self.scalar(&queues, "depth", f64::from(depth))?;
                }
                if let Some(cap) = cap {
                    self.scalar(&queues, "cap", cap)?;
                    if !delivered {
                        self.scalar(&queues, "drop_events", cap)?;
                    }
                }
            }
            // What the queue did with this sample, on its sticky lane: the
            // gap is measured only between deliveries on a real queue -- the
            // driver's `cam0` pseudo-edge has no consumer behind it to fall
            // behind. Frames the source never had are not the queue's to
            // lose: a lidar edge's 176 -> 181 on drive 0009 is one step.
            let gap = delivered
                .then(|| self.gaps.observe(e.edge, e.seq))
                .flatten()
                .map(|g| g - self.absent_between(e, g));
            if let Some(lane) = queue_lane(e.edge, self.bounds.camera_edge) {
                self.sticky(lane, queue_event(e.outcome, gap))?;
            }
        } else if delivered {
            // A producer's row: remember which buffer it handed on, so each
            // consumer's row below can say whether it read that one.
            self.produced.insert((e.stream, e.seq), e.storage_id);
            // A sensor driver's admission: how many in a row came from the
            // same driver. A flat 1 is strict alternation, the proof that
            // two producers contend for one admission. The two drivers admit
            // tens of milliseconds apart, so the rows reach this thread in
            // admission order.
            if ADMISSION_EDGES.contains(&e.edge) && e.arrival_seq.is_some() {
                let run = self.run.observe(e.edge);
                let root = admission_root();
                self.scalar(&root, "run_length", f64::from(run))?;
                self.scalar(&root, "alternating", 1.0)?;
            }
        }
        // The fusion's own row (`track`, the produced or the missing one),
        // on the pairing lane: `reason` is blank on a clean pair and names
        // the failure otherwise, which no other row carries.
        if e.edge == "track" {
            self.sticky(LANE_PAIRING_NAME, pairing_event(e))?;
        }
        // A frame a sensor's source never had: that driver's own row, the
        // only one there is for it anywhere in the pipeline.
        if !delivered && ADMISSION_EDGES.contains(&e.edge) && e.reason == ABSENT_IN_SOURCE {
            self.log_absent(e)?;
        }

        if delivered && real_queue {
            self.log_headroom(e)?;
            if let (Some(due), Some(pe)) = (e.due_ns, e.proc_end_ns) {
                self.after_due_ms.insert(e.edge, (pe - due) as f64 * 1e-6);
            }
            // Did this consumer read the buffer its producer handed on? The
            // producer's row precedes this one on the evidence channel in
            // practice (it is sent at admission, this one after `proc_end`);
            // when it does not, the lane has a gap rather than a guess.
            // Logged only when it changes, like the health lanes: a lane of
            // one word repeated per sample said nothing more.
            if let Some(&producer) = self.produced.get(&(e.stream, e.seq)) {
                let shared = if producer == e.storage_id {
                    "same"
                } else {
                    "copied"
                };
                if self.shared != Some(shared) {
                    self.shared = Some(shared);
                    self.lane(entity::STORAGE_SHARED, shared)?;
                }
            }
        }

        // Bytes are not time series -- a value that is the same on every
        // sample is a line with no information on the time axis -- so they
        // are bar charts, a table and a graph, redrawn from the latest row on
        // each of their edges.
        if delivered {
            self.bytes.insert(e.edge, (e.payload_bytes, e.bytes_alloc));
            self.log_alloc_ratio(e)?;
            if TABLE_EDGES.contains(&e.edge) {
                self.medians
                    .entry(e.edge)
                    .or_default()
                    .push((e.payload_bytes, e.bytes_alloc));
            }
        }
        if delivered && e.edge == "state->sink" {
            // The answer arrived: the chain's end, drawn four ways. The
            // byte chain and the stages, one bar per link or stage, each
            // bar its own entity at its position in the chain; the pipeline
            // graph, labelled with this sweep's bytes; and the answer's age
            // against one period.
            for (i, (name, edge)) in CHAIN_BYTES.iter().enumerate() {
                let bytes = self.bytes.get(edge).map_or(0, |b| b.0);
                self.bars(
                    format!("{}/{name}", entity::BYTES_CHAIN),
                    vec![i as f64],
                    vec![log10_bytes(bytes)],
                    stage_rgb(i),
                )?;
            }
            self.log_graph(e)?;
            self.answers += 1;
            if self.answers == TABLE_AFTER_ANSWERS {
                self.log_byte_table()?;
            }
            for (i, (stage, edge)) in STAGES.iter().enumerate() {
                let ms = self.after_due_ms.get(edge).copied().unwrap_or(0.0);
                self.bars(
                    format!("{}/{stage}", entity::LATENCY_STAGES),
                    vec![i as f64],
                    vec![ms],
                    stage_rgb(i),
                )?;
            }
            // The chain's end in ms, against one period drawn as a band.
            // The period is the sweep's own, on the host clock the answer's
            // age is measured on (so halved at `--rate 2`).
            if let (Some(age_ns), Some(p)) = (e.measurement_age_ns, host_period_ms(&self.bounds, e))
            {
                self.scalar(entity::CHAIN_END, "answer_age_ms", age_ns as f64 * 1e-6)?;
                self.scalar(entity::CHAIN_END, "period_ms", p)?;
                self.band(PERIOD_BAND, "one period", p)?;
            }
        }

        if !delivered {
            // One entity for every edge, and the only place a drop's REASON
            // appears: the cumulative count is what you scan, this is what
            // tells you why. The edge is in the text because the entity no
            // longer says it.
            self.rec.log(
                entity::LOG_DROPS,
                &TextLog::new(format!(
                    "{} seq {} {:?}{}{}",
                    e.edge,
                    e.seq,
                    e.outcome,
                    if e.reason.is_empty() { "" } else { " " },
                    e.reason
                ))
                .with_level(TextLogLevel::WARN),
            )?;
        }
        Ok(())
    }

    /// A rare run-level event (D15) on `log/events`: a clean shutdown is
    /// information, a gap in a sensor's source a warning, a degraded recorder
    /// an error.
    ///
    /// Only `host` is stamped from the event. `seq` is left where the last
    /// evidence row put it -- a per-stream frame number -- because stamping
    /// the event's `arrival_seq_after` there once stretched the `seq` axis of
    /// every plot to the admission count of the whole run. For the same
    /// reason its drawing is filed under that row's sample.
    fn log_event(&self, ev: &Event) -> Result<(), RecError> {
        let rec = &self.rec;
        rec.set_duration_secs("host", ev.host_ns as f64 * 1e-9);
        let level = match ev.kind {
            EventKind::Shutdown => TextLogLevel::INFO,
            EventKind::SourceGap => TextLogLevel::WARN,
            EventKind::RecorderDegraded => TextLogLevel::ERROR,
        };
        let drawn = rec.log(
            entity::LOG_EVENTS,
            &TextLog::new(format!("{:?} at {}: {}", ev.kind, ev.stage, ev.detail))
                .with_level(level),
        );
        rec.send(self.last.clone());
        Ok(drawn?)
    }
}

impl Drop for Dashboard {
    /// However the `rec` thread ends -- an error, a panic -- the viewer's
    /// queue is closed, so the viewer thread is never left waiting on a
    /// producer that is gone.
    fn drop(&mut self) {
        self.rec.close();
    }
}

/// How many evidence rows the `rec` thread buffers before forcing a flush, so
/// an abnormal exit loses a bounded tail rather than everything buffered.
const FLUSH_EVERY: usize = 64;

/// Column names of `events.csv`, in `Event` field order (pinned by a test).
pub const EVENT_HEADER: [&str; 6] = [
    "run_id",
    "host_ns",
    "arrival_seq_after",
    "kind",
    "stage",
    "detail",
];

/// Errors of the `rec` thread.
#[derive(Debug)]
pub enum RecError {
    Csv(csv::Error),
    Io(std::io::Error),
    Rerun(rerun::RecordingStreamError),
}

impl std::fmt::Display for RecError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RecError::Csv(e) => write!(f, "csv: {e}"),
            RecError::Io(e) => write!(f, "io: {e}"),
            RecError::Rerun(e) => write!(f, "dashboard: {e}"),
        }
    }
}

impl std::error::Error for RecError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            RecError::Csv(e) => Some(e),
            RecError::Io(e) => Some(e),
            RecError::Rerun(e) => Some(e),
        }
    }
}

impl From<csv::Error> for RecError {
    fn from(e: csv::Error) -> Self {
        RecError::Csv(e)
    }
}

impl From<std::io::Error> for RecError {
    fn from(e: std::io::Error) -> Self {
        RecError::Io(e)
    }
}

impl From<rerun::RecordingStreamError> for RecError {
    fn from(e: rerun::RecordingStreamError) -> Self {
        RecError::Rerun(e)
    }
}

/// Body of the `rec` thread: drains `q` into `<dir>/evidence.csv` and
/// `<dir>/events.csv` (headers written on the first row of each) and
/// returns the evidence rows for the summary, so nothing reads CSV back.
/// With `Some(dashboard)` each row is also logged to Rerun (M8).
pub fn rec_thread(
    q: Arc<BoundedQueue<EvRow>>,
    dir: PathBuf,
    mut dashboard: Option<Dashboard>,
) -> Result<Vec<Evidence>, RecError> {
    set_stage_slot(SLOT_REC);
    let mut w = csv::Writer::from_path(dir.join("evidence.csv"))?;
    // Events are rare and csv writes headers lazily, so an event-free run
    // would leave a 0-byte file: write the header eagerly instead.
    let mut ew = csv::WriterBuilder::new()
        .has_headers(false)
        .from_path(dir.join("events.csv"))?;
    ew.write_record(EVENT_HEADER)?;
    ew.flush()?;
    let mut rows = Vec::new();
    // Flush every `FLUSH_EVERY` rows rather than only at the end: if the
    // process is killed (Ctrl-C, SIGKILL, a panic that tears the process down
    // before this thread is joined), the `BufWriter` inside `csv::Writer` is
    // destroyed without running, and everything still buffered is lost —
    // truncating `evidence.csv` mid-row. Periodic flushing bounds that loss to
    // the last `FLUSH_EVERY` rows instead of "everything since the last 8 KiB
    // boundary", and a row is only ever written whole.
    let mut since_flush = 0usize;
    while let Some(env) = q.pop() {
        match env.item {
            EvRow::Evidence(e) => {
                w.serialize(&e)?;
                if let Some(d) = dashboard.as_mut() {
                    d.log_row(&e)?;
                }
                rows.push(e);
                since_flush += 1;
                if since_flush >= FLUSH_EVERY {
                    w.flush()?;
                    since_flush = 0;
                }
            }
            EvRow::Event(ev) => {
                ew.serialize(&ev)?;
                // Events are rare and always interesting; never leave one buffered.
                ew.flush()?;
                if let Some(d) = dashboard.as_mut() {
                    d.log_event(&ev)?;
                    // The shutdown event is sent once every stage has
                    // stopped, so nothing the dashboard draws comes after
                    // it: it finishes here, and closes the viewer's queue,
                    // while the viewer's own rows are still to come.
                    if matches!(ev.kind, EventKind::Shutdown) {
                        d.finish()?;
                    }
                }
            }
        }
    }
    if let Some(d) = dashboard.as_mut() {
        d.finish()?;
    }
    w.flush()?;
    ew.flush()?;
    Ok(rows)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use pipes_core::clock::HostTime;
    use pipes_core::evidence::RowCtx;

    use super::*;
    use crate::viewer::ViewerQueue;

    #[test]
    fn event_header_matches_the_struct() {
        let ctx = RowCtx {
            run_id: "t".to_string(),
            t0_host: HostTime(0),
            epoch: 0,
        };
        let ev = Event::recorder_degraded(&ctx, HostTime(1), "proc", 2);
        let mut w = csv::Writer::from_writer(Vec::new());
        w.serialize(&ev).unwrap();
        let text = String::from_utf8(w.into_inner().unwrap()).unwrap();
        assert_eq!(text.lines().next().unwrap(), EVENT_HEADER.join(","));
    }

    #[test]
    fn rec_thread_writes_an_events_header_even_without_events() {
        let dir = std::env::temp_dir().join(format!("pipes-rec-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let q: Arc<BoundedQueue<EvRow>> = Arc::new(BoundedQueue::new(4, QueuePolicy::DropOldest));
        q.close();
        let rows = rec_thread(Arc::clone(&q), dir.clone(), None).unwrap();
        assert!(rows.is_empty());
        let events = std::fs::read_to_string(dir.join("events.csv")).unwrap();
        assert_eq!(events.trim_end(), EVENT_HEADER.join(","));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn every_logged_series_has_a_name_and_a_colour() {
        for leaf in SERIES_LEAVES {
            let style = series_style(leaf)
                .unwrap_or_else(|| panic!("{leaf} is logged but has no SeriesLines style"));
            assert!(!style.label.is_empty(), "{leaf} has an empty legend label");
            assert!(style.width > 0.0, "{leaf} has a zero stroke width");
        }
        assert!(series_style("not_a_series").is_none());
    }

    #[test]
    fn name_fits_a_legend_and_carries_its_unit() {
        // A legend entry is read at a glance beside the plot, so it is short,
        // and it says what the number IS, so a measurement ends in its unit
        // in parentheses. The three queue counts are the exception: "depth
        // at push", "depth after pop" and "capacity" are items, and the view
        // title names the capacity they are read against.
        for leaf in SERIES_LEAVES {
            let label = series_style(leaf).unwrap().label;
            assert!(
                label.chars().count() <= LABEL_MAX,
                "{leaf}: {label:?} is longer than {LABEL_MAX} characters"
            );
            if !matches!(leaf, "depth_at_push" | "depth" | "cap") {
                assert!(
                    label.ends_with(')') && label.contains(" ("),
                    "{leaf}: {label:?} does not end in a unit"
                );
            }
        }
    }

    #[test]
    fn the_queue_depth_is_read_against_a_capacity_line_behind_it() {
        // Both edges of the camera queue's occupancy are drawn: the depth
        // each push met (the one series that shows a queue running full) and
        // the backlog each pop left (the one that shows it draining). Both
        // are the green occupancy family, the push the louder voice.
        let push = series_style("depth_at_push").unwrap();
        let pop = series_style("depth").unwrap();
        for d in [&push, &pop] {
            assert!(
                d.rgb[1] > d.rgb[0] && d.rgb[1] > d.rgb[2],
                "{:?} is not in the green occupancy family",
                d.rgb
            );
            // A count holds its value until the next sample.
            assert!(d.step());
        }
        assert_ne!(push.rgb, pop.rgb);
        assert!(push.width > pop.width);
        // The line they are read against is a reference, not a third
        // measurement: bright grey, and no heavier than the drain.
        let cap = series_style("cap").unwrap();
        assert_eq!(cap.rgb, C_REFERENCE);
        assert!(cap.width <= pop.width && cap.step());
        // A dropped frame is a red cross ON that line: points, not a line.
        let dropped = series_style("drop_events").unwrap();
        assert_eq!(dropped.rgb, C_FAIL);
        assert_eq!(dropped.mark, Mark::Points(MarkerShape::Cross));
        // And the cumulative drops are the failure colour, as a step.
        let drops = series_style("drops").unwrap();
        assert_eq!(drops.rgb, C_FAIL);
        assert!(drops.step());
    }

    #[test]
    fn the_fill_plot_picks_out_the_camera_and_the_lidar_queue() {
        let cam = Some("cam0->camdet");
        assert_eq!(fill_rgb(&queue_root("cam0->camdet"), cam), C_CAMERA_ONLY);
        assert_eq!(fill_rgb(&queue_root("velo->reduce"), cam), C_LIDAR_ONLY);
        // The same queue is slate when it is not the one feeding the answer.
        assert_eq!(fill_rgb(&queue_root("cam0->proc"), cam), C_NEUTRAL);
        assert_eq!(
            fill_rgb(&queue_root("cam0->proc"), Some("cam0->proc")),
            C_CAMERA_ONLY
        );
        assert_eq!(fill_rgb(&queue_root("obj->track"), cam), C_NEUTRAL);
    }

    #[test]
    fn the_admission_run_is_one_while_the_drivers_alternate() {
        let mut r = AdmissionRun::default();
        let runs: Vec<u32> = ["cam0", "velo", "cam0", "velo", "velo", "velo", "cam0"]
            .iter()
            .map(|s| r.observe(s))
            .collect();
        assert_eq!(runs, [1, 1, 1, 1, 2, 3, 1]);
    }

    #[test]
    fn the_time_to_contact_is_points_in_the_warning_red_against_a_grey_line() {
        // The flagged object changes from sweep to sweep, so the TTC is drawn
        // as markers: a line joined one object's approach to the next one's
        // with a jump no object made.
        let ttc = series_style("ttc_s").unwrap();
        assert_eq!(ttc.mark, Mark::Points(MarkerShape::Circle));
        // Red: the one demo number that is bad when it is small. Never the
        // answer's gold, which is the box and its words.
        assert!(
            ttc.rgb[0] > 200 && ttc.rgb[0] > 2 * ttc.rgb[1] && ttc.rgb[0] > 2 * ttc.rgb[2],
            "time to contact is not in the warning family: {:?}",
            ttc.rgb
        );
        assert_ne!(ttc.rgb, C_ANSWER);
        // The 3 s line it is read against is a reference: a grey line.
        let warn = series_style("ttc_warn").unwrap();
        assert!(matches!(warn.mark, Mark::Line { .. }));
        assert!(warn.rgb[0] == warn.rgb[1] && warn.rgb[1] == warn.rgb[2]);
        // And its archetype is the points one, not a line with a marker.
        let arch = series_archetype("ttc_s").unwrap();
        let names: Vec<String> = arch
            .as_serialized_batches()
            .iter()
            .map(|b| b.descriptor.component.to_string())
            .collect();
        assert!(
            names.iter().any(|n| n.contains("SeriesPoints")),
            "{names:?}"
        );
    }

    #[test]
    fn a_stage_that_logs_its_own_series_gets_the_same_styling_table() {
        // `series_archetype` is the door `state-sink` comes through, because
        // an evidence row carries how long a sample took and never what it
        // SAID. It must answer for exactly the leaves `series_style` does, or
        // the test above is checking a table nothing reads.
        for leaf in SERIES_LEAVES {
            assert!(
                series_archetype(leaf).is_some(),
                "{leaf} is styled for the dashboard but not for a stage that logs it itself"
            );
        }
        assert!(series_archetype("not_a_series").is_none());
    }

    #[test]
    fn a_reference_line_is_grey_flat_and_never_heavier_than_what_it_bounds() {
        // A reference line sits behind the measurement it bounds rather than
        // competing with it: the one bright grey, a step, and no heavier.
        for (reference, measured) in [
            ("cap", "depth_at_push"),
            ("at_bound", "answer"),
            ("at_bound", "camera_queue"),
            ("at_bound", "cam_service"),
            ("alternating", "run_length"),
            ("ttc_warn", "ttc_s"),
            ("sweep_start", "camera"),
            ("sweep_end", "camera"),
            ("trigger", "camera"),
            ("period_ms", "answer_age_ms"),
        ] {
            let r = series_style(reference).unwrap();
            let m = series_style(measured).unwrap();
            assert_eq!(r.rgb, C_REFERENCE, "{reference}");
            assert!(r.step(), "{reference} is not flat");
            assert!(r.width <= m.width, "{reference} is heavier than {measured}");
        }
        // The answer's own ratio is the loudest line on its plot, and it and
        // the answer's age in ms are the answer's gold, as its box is.
        let answer = series_style("answer").unwrap();
        assert_eq!(answer.rgb, C_ANSWER);
        assert_eq!(series_style("answer_age_ms").unwrap().rgb, C_ANSWER);
        for other in ["camera_queue", "camera_wait", "velo_to_reduce"] {
            assert!(series_style(other).unwrap().width < answer.width);
        }
    }

    #[test]
    fn a_period_is_the_sweep_s_own_or_the_camera_s_on_the_host_clock() {
        use pipes_core::clock::{SensorTime, Tov};
        use pipes_core::sample::StreamId;
        let ctx = RowCtx {
            run_id: "t".to_string(),
            t0_host: HostTime(0),
            epoch: 0,
        };
        let mut b = Bounds {
            period_ms: Some(103.6),
            rate: Some(1.0),
            ..Bounds::default()
        };
        let mut sweep = Evidence::driver_missing(
            &ctx,
            StreamId::LIDAR,
            "velo",
            "driver",
            3,
            Tov::Range {
                start: SensorTime(1_000_000_000),
                end: SensorTime(1_103_300_000),
            },
            Some(HostTime(0)),
            HostTime(0),
            "",
        );
        let p = host_period_ms(&b, &sweep).unwrap();
        assert!((p - 103.3).abs() < 1e-9, "{p}");
        // A camera row is an instant: the camera's median period stands in.
        sweep.tov_end_ns = sweep.tov_start_ns;
        assert_eq!(host_period_ms(&b, &sweep), Some(103.6));
        // Twice real time halves it; an unpaced run has none.
        b.rate = Some(2.0);
        assert_eq!(host_period_ms(&b, &sweep), Some(51.8));
        b.rate = None;
        assert_eq!(host_period_ms(&b, &sweep), None);
    }

    #[test]
    fn the_stage_bars_are_coloured_by_what_they_read() {
        assert_eq!(STAGES.len(), 5);
        assert_eq!(stage_rgb(0), C_LIDAR_ONLY);
        assert_eq!(stage_rgb(1), C_LIDAR_ONLY);
        assert_eq!(stage_rgb(2), C_DETECTION);
        assert_eq!(stage_rgb(3), C_DETECTION);
        assert_eq!(stage_rgb(4), C_ANSWER);
    }

    #[test]
    fn every_lane_has_its_states_and_colours() {
        for path in LANE_PATHS {
            let lane = lane_style(path).unwrap_or_else(|| panic!("{path} is a lane with no style"));
            assert_eq!(
                lane.values.len(),
                lane.colors.len(),
                "{path}: a state without a colour"
            );
            let mut seen = lane.values.to_vec();
            seen.sort_unstable();
            seen.dedup();
            assert_eq!(seen.len(), lane.values.len(), "{path}: a state twice");
        }
        // The table is the dashboard's list of lanes, and nothing else.
        for name in LANE_NAMES {
            assert!(LANE_PATHS.contains(&lane_path(name).as_str()), "{name}");
        }
        assert!(lane_style("queues/cam0_to_proc/depth").is_none());
        assert!(lane_style("answer/line").is_none());
        assert!(lane_style("lanes/not_a_lane").is_none());
        // Health is the dark green on every lane, and a loss is red.
        for lane in [
            &LANE_QUEUE,
            &LANE_LIDAR,
            &LANE_CAMERA,
            &LANE_PAIRING,
            &LANE_SHARED,
        ] {
            assert_eq!(lane.colors[0], C_LANE_OK);
            assert_eq!(lane.colors[lane.colors.len() - 1], C_FAIL);
        }
        // A gap in a source is grey on the lanes that can show one -- neither
        // health nor the pipeline's failure -- and the lidar's and camera's
        // lanes and the pairing lane are the ones that can.
        for (lane, word) in [
            (&LANE_LIDAR, NO_SWEEP),
            (&LANE_CAMERA, NO_FRAME),
            (&LANE_PAIRING, SOURCE_GAP),
        ] {
            let i = lane.values.iter().position(|v| *v == word).unwrap();
            assert_eq!(lane.colors[i], C_ABSENT, "{word}");
        }
        assert!(!LANE_QUEUE.values.contains(&NO_SWEEP));
        assert_eq!(
            lane_style("lanes/lidar_queue").map(|l| l.values),
            Some(LANE_LIDAR.values)
        );
        assert_eq!(
            lane_style("lanes/camera_queue").map(|l| l.values),
            Some(LANE_CAMERA.values)
        );
    }

    #[test]
    fn a_lane_s_words_are_states_it_can_show() {
        // A queue's row: dropped is red, a delivery after a gap amber, and
        // a delivery that follows its predecessor clean.
        for (outcome, gap, want) in [
            (Outcome::Delivered, None, None),
            (Outcome::Delivered, Some(1.0), None),
            (Outcome::Delivered, Some(3.0), Some(("skipping", 1))),
            (Outcome::DroppedOldest, None, Some(("dropping", 2))),
            (Outcome::DroppedNewest, None, Some(("dropping", 2))),
            (Outcome::Timeout, None, Some(("dropping", 2))),
            (Outcome::Missing, None, Some(("dropping", 2))),
        ] {
            let event = queue_event(outcome, gap);
            assert_eq!(event, want, "{outcome:?} {gap:?}");
            if let Some((word, _)) = event {
                assert!(LANE_QUEUE.values.contains(&word), "{word} has no colour");
            }
        }
    }

    #[test]
    fn the_pairing_lane_reads_the_fusion_s_own_row() {
        use pipes_core::clock::{SensorTime, Tov};
        use pipes_core::sample::StreamId;
        let ctx = RowCtx {
            run_id: "t".to_string(),
            t0_host: HostTime(0),
            epoch: 0,
        };
        // The expired set's row, exactly as `track_thread` writes it: every
        // way a set expires is `expired`, and the WARN log names which.
        let mut row = Evidence::driver_missing(
            &ctx,
            StreamId::TRACKS,
            "track",
            "track",
            7,
            Tov::Time(SensorTime(700)),
            Some(HostTime(690)),
            HostTime(700),
            "pair_late",
        );
        for reason in ["pair_late", "pair_dropped", "pair_absent"] {
            row.reason = reason;
            assert_eq!(pairing_event(&row), Some(("expired", 2)));
        }
        // The produced row: `reason` blank on a clean pair, `stale` under
        // `--pair-stale-ms`.
        row.outcome = Outcome::Delivered;
        row.reason = "";
        assert_eq!(pairing_event(&row), None);
        row.reason = "stale";
        assert_eq!(pairing_event(&row), Some(("stale", 1)));
        // A set that expired because the camera's SOURCE has no frame for its
        // sweep is the source's gap, grey, not a red expiry.
        row.outcome = Outcome::Missing;
        row.reason = "pair_absent_in_source";
        assert_eq!(pairing_event(&row), Some((SOURCE_GAP, 2)));
        for word in ["stale", "expired", SOURCE_GAP] {
            assert!(LANE_PAIRING.values.contains(&word));
        }
    }

    #[test]
    fn a_sticky_lane_logs_only_its_changes_and_holds_the_worst() {
        let mut lane = Sticky::default();
        // The first row logs the lane's state, even a clean one; the next
        // clean rows log nothing.
        assert_eq!(lane.observe(None), Some("ok"));
        assert_eq!(lane.observe(None), None);
        // A drop turns it red; the skip that follows each drop does not turn
        // it amber -- a queue dropping every other frame is one red block,
        // not an alternation that shatters the lane into "N states".
        assert_eq!(lane.observe(Some(("dropping", 2))), Some("dropping"));
        for _ in 0..20 {
            assert_eq!(lane.observe(Some(("skipping", 1))), None);
            assert_eq!(lane.observe(Some(("dropping", 2))), None);
        }
        // Nine clean rows are not enough; the tenth turns it green.
        for _ in 1..LANE_CLEAR_AFTER {
            assert_eq!(lane.observe(None), None);
        }
        assert_eq!(lane.observe(None), Some("ok"));
        // A lesser failure shows when nothing worse is showing, and a worse
        // one replaces it.
        assert_eq!(lane.observe(Some(("skipping", 1))), Some("skipping"));
        assert_eq!(lane.observe(Some(("dropping", 2))), Some("dropping"));
        // A worse state that stops recurring steps down to the lesser
        // failure still happening, then to ok once that stops too.
        for _ in 1..LANE_CLEAR_AFTER {
            assert_eq!(lane.observe(Some(("skipping", 1))), None);
        }
        assert_eq!(lane.observe(Some(("skipping", 1))), Some("skipping"));
        for _ in 1..LANE_CLEAR_AFTER {
            assert_eq!(lane.observe(None), None);
        }
        assert_eq!(lane.observe(None), Some("ok"));
        // The stale run's pairing lane: sweep 0 expires (it has no older
        // frame), every sweep after it is paired stale. It reads expired for
        // ten sweeps and stale after, not expired for the whole run.
        let mut pairing = Sticky::default();
        assert_eq!(pairing.observe(Some(("expired", 2))), Some("expired"));
        let words: Vec<Option<&str>> = (0..30)
            .map(|_| pairing.observe(Some(("stale", 1))))
            .collect();
        let changed: Vec<(usize, &str)> = words
            .iter()
            .enumerate()
            .filter_map(|(i, w)| w.map(|w| (i, w)))
            .collect();
        assert_eq!(changed, [(LANE_CLEAR_AFTER as usize - 1, "stale")]);
        // A lane whose first row is a failure logs the failure first.
        assert_eq!(Sticky::default().observe(Some(("stale", 1))), Some("stale"));
    }

    #[test]
    fn a_lane_steps_down_after_a_second_of_its_own_rows() {
        // One edge feeds each lane but the fusion's, which two feed: ten rows
        // are a second on the others and half of one there.
        for camera_edge in [Some("cam0->camdet"), Some("cam0->proc")] {
            for name in LANE_NAMES {
                let want = if name == "fusion_queue" { 20 } else { 10 };
                assert_eq!(lane_clear_after(name, camera_edge), want, "{name}");
            }
        }
        // And a lane given twenty holds its failure through nineteen quiet
        // rows, stepping down on the twentieth.
        let mut lane = Sticky::new(20);
        assert_eq!(lane.observe(Some(("dropping", 2))), Some("dropping"));
        for _ in 1..20 {
            assert_eq!(lane.observe(Some(("skipping", 1))), None);
        }
        assert_eq!(lane.observe(None), Some("skipping"));
    }

    #[test]
    fn the_byte_table_steps_down_the_chain_and_names_the_one_copy() {
        let link = |name, carried, allocated| ByteLink {
            name,
            carried: Some(carried),
            allocated: Some(allocated),
            rows: 154,
        };
        let links = [
            link("sweep", 1_958_000, 0),
            link("voxels", 518_000, 0),
            link("dets", 7_980, 0),
            link("tracks", 19_522, 0),
            link("answer", 12_916, 0),
        ];
        let t = byte_table(&links, (Some(1_974_352), Some(1_958_000)), 154);
        let lines: Vec<&str> = t.lines().collect();
        assert_eq!(
            lines[0],
            "| link | carried | step | allocated on the edge |"
        );
        assert_eq!(lines[2], "| 1 sweep | **1.96 MB** |  | 0 B |");
        assert_eq!(lines[3], "| 2 voxels | **518 kB** | ÷ 3.8 | 0 B |");
        assert_eq!(lines[4], "| 3 dets | **8 kB** | ÷ 64.9 | 0 B |");
        // A link that grows says so, as a product.
        assert_eq!(lines[5], "| 4 tracks | **20 kB** | × 2.4 | 0 B |");
        assert_eq!(lines[6], "| 5 answer | **13 kB** | ÷ 1.5 | 0 B |");
        assert!(
            t.ends_with("Every link reads its input in place: the lidar driver's 1.97 MB allocated for 1.96 MB carried is the chain's one copy. Medians of 154 sweeps."),
            "{t}"
        );
        // A link that allocated is not called in place.
        let mut copied = links;
        copied[1].allocated = Some(518_000);
        let t = byte_table(&copied, (Some(1_974_352), Some(1_958_000)), 154);
        assert!(!t.contains("in place"), "{t}");
        assert!(
            t.contains("| 2 voxels | **518 kB** | ÷ 3.8 | 518 kB |"),
            "{t}"
        );
        // A run whose chain lost sweeps says what each median is over,
        // rather than calling them all medians of the driver's sweeps.
        let mut evicted = copied;
        for l in &mut evicted[2..] {
            l.rows = 67;
        }
        evicted[4].rows = 63;
        let t = byte_table(&evicted, (Some(1_974_352), Some(1_958_000)), 154);
        assert!(
            t.ends_with(
                "Medians of each link's own rows: 154, 154, 67, 67, 63 for links 1 to 5, of the driver's 154 sweeps."
            ),
            "{t}"
        );
        assert!(!t.contains("Medians of 154 sweeps"), "{t}");
        // Every link of the chain is one the table has rows for.
        for (_, edge) in CHAIN_BYTES {
            assert!(TABLE_EDGES.contains(&edge), "{edge}");
        }
    }

    #[test]
    fn bytes_read_the_way_the_table_prints_them() {
        assert_eq!(human_bytes(341), "341 B");
        assert_eq!(human_bytes(12_916), "13 kB");
        assert_eq!(human_bytes(1_974_352), "1.97 MB");
        // Allocated over carried: 0 is zero-copy, and an edge that carried
        // nothing is 0, not a division by zero.
        assert_eq!(alloc_ratio(0, 1_958_000), 0.0);
        assert_eq!(alloc_ratio(465_750, 1_397_250), 465_750.0 / 1_397_250.0);
        assert_eq!(alloc_ratio(10, 0), 0.0);
        // Coloured by producer.
        assert_eq!(producer_rgb("velo", None), C_LIDAR_ONLY);
        assert_eq!(producer_rgb("det->cloud", None), C_LIDAR_ONLY);
        assert_eq!(producer_rgb("obj->sink", None), C_DETECTION);
        assert_eq!(producer_rgb("cam0->proc", None), C_CAMERA_ONLY);
    }

    #[test]
    fn bytes_are_drawn_on_a_log_scale_that_survives_zero() {
        assert_eq!(log10_bytes(0), 0.0);
        assert_eq!(log10_bytes(1), 0.0);
        assert_eq!(log10_bytes(1_000), 3.0);
        assert!((log10_bytes(1_974_352) - 6.295).abs() < 0.001);
    }

    #[test]
    fn seq_gap_skips_the_first_delivery_then_counts_frames_never_seen() {
        let mut g = SeqGap::default();
        // No predecessor: nothing to measure against.
        assert_eq!(g.observe("cam0->proc", 0), None);
        // Consecutive deliveries: the consumer kept up.
        assert_eq!(g.observe("cam0->proc", 1), Some(1.0));
        // Frames 2 and 3 were dropped ahead of the consumer.
        assert_eq!(g.observe("cam0->proc", 4), Some(3.0));
        // Edges are tracked separately, each with its own first delivery.
        assert_eq!(g.observe("cam0->rerun", 4), None);
        assert_eq!(g.observe("cam0->rerun", 5), Some(1.0));
        assert_eq!(g.observe("cam0->proc", 5), Some(1.0));
    }

    /// Every shape of row the recorder mirrors, three sweeps' worth: each
    /// sensor driver's admitted frame and a missing one; each derived
    /// stream's producer row, the fusion's clean, stale and expired; and on
    /// every queue this binary opens a delivered row, a dropped one and a
    /// delivered one after the gap. Each row carries every column the
    /// dashboard reads.
    fn every_kind_of_row() -> Vec<Evidence> {
        use crate::dashboard::{ADMISSION_EDGES, EDGES};
        use pipes_core::clock::{SensorTime, Tov};
        use pipes_core::sample::StreamId;
        let ctx = RowCtx {
            run_id: "t".to_string(),
            t0_host: HostTime(0),
            epoch: 0,
        };
        const MS: i64 = 1_000_000;
        let stream = |edge: &str| -> u8 {
            let producer = edge.split("->").next().unwrap_or(edge);
            match producer {
                "cam0" => 0,
                "velo" => 1,
                "det" => 3,
                "cam_det" => 4,
                "state" => 5,
                "track" => 6,
                _ => 7,
            }
        };
        let mut rows = Vec::new();
        let mut arrival = 0i64;
        let mut row = |edge: &'static str, seq: u64, outcome: Outcome, reason: &'static str| {
            arrival += 10 * MS;
            let start = seq as i64 * 100 * MS;
            let end = if edge.starts_with("cam0") {
                start
            } else {
                start + 103 * MS
            };
            let mut e = Evidence::driver_missing(
                &ctx,
                StreamId::LIDAR,
                edge,
                "stage",
                seq,
                Tov::Range {
                    start: SensorTime(start),
                    end: SensorTime(end.max(start + 1)),
                },
                Some(HostTime(start)),
                HostTime(arrival),
                reason,
            );
            e.tov_end_ns = end;
            e.stream = stream(edge);
            e.outcome = outcome;
            if ADMISSION_EDGES.contains(&edge) && outcome == Outcome::Delivered {
                e.arrival_seq = Some(arrival as u64);
            }
            if edge.contains("->") {
                e.depth_at_push = Some(if outcome == Outcome::Delivered { 1 } else { 4 });
            }
            if outcome == Outcome::Delivered {
                e.depth_after_pop = edge.contains("->").then_some(0);
                e.proc_start_ns = Some(arrival);
                e.proc_end_ns = Some(arrival + 5 * MS);
                e.queue_wait_ns = Some(2 * MS);
                e.measurement_age_ns = Some(40 * MS);
                e.storage_id = 1000 + seq as usize;
                e.payload_bytes = 1000;
            }
            rows.push(e);
        };
        for seq in 0..3u64 {
            for edge in ADMISSION_EDGES {
                row(edge, seq, Outcome::Delivered, "");
            }
            for edge in ["det", "obj", "cam_det", "state"] {
                row(edge, seq, Outcome::Delivered, "");
            }
            match seq {
                0 => row("track", seq, Outcome::Delivered, ""),
                1 => row("track", seq, Outcome::Delivered, "stale"),
                _ => row("track", seq, Outcome::Missing, "pair_late"),
            }
            for edge in EDGES {
                if seq == 1 {
                    row(edge, seq, Outcome::DroppedOldest, "evicted");
                } else {
                    row(edge, seq, Outcome::Delivered, "");
                }
            }
        }
        for edge in ADMISSION_EDGES {
            row(edge, 3, Outcome::Missing, "absent");
        }
        // A frame each sensor's source never had: no instant, and the
        // fusion's expiry for the camera's.
        for edge in ADMISSION_EDGES {
            row(edge, 4, Outcome::Missing, "absent_in_source");
        }
        row("track", 4, Outcome::Missing, "pair_absent_in_source");
        for e in rows.iter_mut().filter(|e| e.reason == "absent_in_source") {
            e.tov_start_ns = 0;
            e.tov_end_ns = 0;
        }
        rows
    }

    #[test]
    fn every_series_the_dashboard_logs_is_styled_and_in_a_view() {
        // Read back from what the dashboard LOGGED, not from a list kept by
        // hand beside it: a leaf added to `log_row` without a style, or
        // without a view to show it, fails here the first time it is logged.
        use crate::dashboard::{logged, send_blueprint, EDGES};
        use rerun::{RecordingStreamBuilder, StoreKind};
        for camera_edge in ["cam0->camdet", "cam0->proc"] {
            let bounds = Bounds {
                age_queue_ms: BTreeMap::from([
                    ("cam0->proc", 413.2),
                    ("cam0->rerun", 103.3),
                    ("cam0->camdet", 103.3),
                ]),
                queue_cap: EDGES
                    .iter()
                    .map(|e| (*e, if e.starts_with("cam0") { 1.0 } else { 4.0 }))
                    .collect(),
                image_wh: Some((1242.0, 375.0)),
                period_ms: Some(103.3),
                camera_edge: Some(camera_edge),
                n_frames: Some(4.0),
                rate: Some(1.0),
                lidar_lane: true,
                answer: true,
            };
            let (rec, storage) = RecordingStreamBuilder::new("pipes-test").memory().unwrap();
            let viewer = ViewerQueue::new(1 << 16);
            let mut d =
                Dashboard::new(&rec, viewer.canvas("rec"), bounds.clone(), Mode::File).unwrap();
            for e in every_kind_of_row() {
                d.log_row(&e).unwrap();
            }
            let ctx = RowCtx {
                run_id: "t".to_string(),
                t0_host: HostTime(0),
                epoch: 0,
            };
            d.log_event(&Event::recorder_degraded(&ctx, HostTime(1), "proc", 2))
                .unwrap();
            d.log_event(&Event::shutdown(&ctx, HostTime(2), 3, "done"))
                .unwrap();
            d.finish().unwrap();
            // What the viewer thread would do with every drawing queued.
            assert_eq!(viewer.dropped(), 0, "the test's queue overflowed");
            viewer.draw_queued(&rec);
            rec.flush_blocking().unwrap();
            let logged = logged(&storage, StoreKind::Recording);
            let (bp, _bp_storage) = RecordingStreamBuilder::new("pipes-test").memory().unwrap();
            let layout = send_blueprint(&bp, Mode::File, &bounds).unwrap();

            // The run under test drew every family the dashboard has.
            for want in [
                "queues/cam0_to_proc/fill",
                "queues/velo_to_reduce/drops",
                "lanes/pairing",
                "latency/headroom/answer",
                "latency/stages/5_answer",
                "bytes/chain/1_sweep",
                "bytes/table",
                "graph/pipeline",
                "log/drops",
                "log/events",
                // The gaps: each sensor's lane, and what was cleared or
                // stamped for the frame that is not there.
                "lanes/lidar_queue",
                "lanes/camera_queue",
                "queues/admission/velo/absent_in_source",
                "queues/admission/cam0/absent_in_source",
                "camera/status",
                "answer/headline",
                "lidar/sweep",
            ] {
                assert!(
                    logged.contains_key(want),
                    "{camera_edge}: nothing at {want}"
                );
            }
            let has = |comps: &BTreeSet<String>, archetype: &str| {
                comps
                    .iter()
                    .any(|c| c.starts_with(&format!("{archetype}:")))
            };
            for (path, comps) in &logged {
                if path.starts_with("__") {
                    continue; // the SDK's own recording properties
                }
                assert!(
                    layout.shows(path),
                    "{camera_edge}: {path} is logged but no view shows it"
                );
                if has(comps, "Scalars") {
                    assert!(
                        has(comps, "SeriesLines") || has(comps, "SeriesPoints"),
                        "{camera_edge}: {path} is a series with no name or colour: {comps:?}"
                    );
                    // And its leaf is in the table the legend tests walk.
                    let leaf = path.rsplit('/').next().unwrap_or(path);
                    assert!(
                        SERIES_LEAVES.contains(&leaf),
                        "{camera_edge}: {leaf} ({path}) is not in SERIES_LEAVES"
                    );
                }
                if has(comps, "Measurements") {
                    assert!(
                        comps.iter().any(|c| c.starts_with("Measurements:names")),
                        "{camera_edge}: {path} is a band with no name: {comps:?}"
                    );
                }
                if has(comps, "StateChange") {
                    assert!(
                        has(comps, "StateConfiguration"),
                        "{camera_edge}: {path} is a lane with no states' colours: {comps:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn drop_counter_counts_only_non_delivered() {
        let mut c = DropCounter::default();
        let mut drops = |edge, outcome, reason| c.observe(edge, outcome, reason).0;
        assert_eq!(drops("cam0->proc", Outcome::Delivered, ""), 0);
        assert_eq!(drops("cam0->proc", Outcome::DroppedOldest, "evicted"), 1);
        assert_eq!(drops("cam0->proc", Outcome::Delivered, ""), 1);
        assert_eq!(drops("cam0->proc", Outcome::DroppedNewest, "full"), 2);
        assert_eq!(drops("cam0->proc", Outcome::Timeout, "max_wait"), 3);
        assert_eq!(drops("cam0->proc", Outcome::Missing, "deadline_skipped"), 4);
        // Edges count separately.
        assert_eq!(drops("cam0->rerun", Outcome::Delivered, ""), 0);
        assert_eq!(drops("cam0->rerun", Outcome::DroppedOldest, "evicted"), 1);
        assert_eq!(drops("cam0->proc", Outcome::Delivered, ""), 4);
    }

    /// A frame the source never had is not a drop: it is counted apart, and
    /// the drops stay where the pipeline's own losses put them.
    #[test]
    fn a_frame_absent_in_the_source_is_not_a_drop() {
        let mut c = DropCounter::default();
        assert_eq!(c.observe("velo", Outcome::Delivered, ""), (0, 0));
        assert_eq!(
            c.observe("velo", Outcome::Missing, ABSENT_IN_SOURCE),
            (0, 1)
        );
        assert_eq!(
            c.observe("velo", Outcome::Missing, ABSENT_IN_SOURCE),
            (0, 2)
        );
        assert_eq!(
            c.observe("velo", Outcome::Missing, "deadline_skipped"),
            (1, 2)
        );
        assert_eq!(c.observe("velo", Outcome::Delivered, ""), (1, 2));
    }

    /// The seqs a stream carries are the frames of the sensors it counts,
    /// so a frame absent in a sensor's source is absent on every edge of
    /// every such stream -- the camera's on `cam_det->track` as on
    /// `cam0->camdet` -- and never on a stream of the other sensor.
    #[test]
    fn a_source_gap_is_stepped_over_on_every_stream_that_counts_its_frames() {
        use pipes_core::sample::StreamId;
        let (rec, _storage) = rerun::RecordingStreamBuilder::new("pipes-test")
            .memory()
            .unwrap();
        let viewer = ViewerQueue::new(16);
        let mut d =
            Dashboard::new(&rec, viewer.canvas("rec"), Bounds::default(), Mode::File).unwrap();
        d.absent.insert((Sensor::Camera, 8));
        d.absent.insert((Sensor::Lidar, 3));
        d.absent.insert((Sensor::Lidar, 4));
        let row = |stream: StreamId, seq: u64| {
            let mut e = Evidence::driver_missing(
                &RowCtx {
                    run_id: "t".to_string(),
                    t0_host: HostTime(0),
                    epoch: 0,
                },
                stream,
                "e",
                "s",
                seq,
                pipes_core::clock::Tov::None,
                None,
                HostTime(0),
                "",
            );
            e.stream = stream.0;
            e
        };
        // 7 -> 9 over the camera's frame 8, on the camera and on cam_det.
        assert_eq!(d.absent_between(&row(StreamId::CAM0, 9), 2.0), 1.0);
        assert_eq!(d.absent_between(&row(StreamId::CAM_DET, 9), 2.0), 1.0);
        // The same step on the lidar is the lidar's own: nothing to excuse.
        assert_eq!(d.absent_between(&row(StreamId::LIDAR, 9), 2.0), 0.0);
        // 2 -> 5 over the lidar's frames 3-4, on the lidar and every stream
        // derived from its sweeps -- and on nothing of the camera's.
        for s in [StreamId::LIDAR, StreamId::LIDAR_DET, StreamId::LIDAR_OBJ] {
            assert_eq!(d.absent_between(&row(s, 5), 3.0), 2.0, "{s:?}");
        }
        assert_eq!(d.absent_between(&row(StreamId::CAM_DET, 5), 3.0), 0.0);
        // The fusion's output lacks an instant either source lacks.
        assert_eq!(d.absent_between(&row(StreamId::TRACKS, 9), 7.0), 3.0);
        assert_eq!(d.absent_between(&row(StreamId::EGO, 9), 7.0), 3.0);
        // A consecutive delivery steps over nothing.
        assert_eq!(d.absent_between(&row(StreamId::CAM0, 8), 1.0), 0.0);
    }
}
