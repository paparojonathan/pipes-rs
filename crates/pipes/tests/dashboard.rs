//! The dashboard, end to end: what a real run LOGS onto its recording, read
//! back out of the `.rrd` it wrote, against the layout the same file carries.
//!
//! The unit tests check the layout against the entity table in
//! `dashboard.rs` and the recorder's series against its own rows; neither
//! sees what the stages draw straight onto the recording -- the answer, its
//! headline and label, the pairing window, the clouds, the boxes. This does:
//! whatever any stage logs, under whatever name, must be shown by some view
//! of the file's own layout, and every series and lane must carry the style
//! that names and colours it. A new entity, or a renamed one, fails here the
//! first time a run logs it.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use pipes_kitti::camdet::MODEL_FILE;
use pipes_kitti::testing::FixtureDrive;
use rerun::archetypes::{Points2D, StateChange, TextDocument, TextLog};
use rerun::external::re_log_encoding::DecoderApp;
use rerun::external::re_sdk_types::blueprint::archetypes::ViewContents;
use rerun::log::{Chunk, LogMsg};
use rerun::StoreKind;
use tempfile::TempDir;

mod common;
use common::{assert_ok, evidence, pipes};

/// Sweeps in the fixture, and its period: paced, so the run has deadlines
/// and draws its latencies.
const SWEEPS: usize = 8;
const PERIOD_NS: i64 = 25_000_000;

/// Frames of the gap fixture: 19 sweeps after the lidar's gap at 3-4, well
/// past the ten quiet rows after which the lidar lane steps down.
const GAP_FRAMES: usize = 24;

/// Rows without a failure after which a lane steps down (`record.rs`).
const LANE_CLEAR_AFTER: usize = 10;

// The sweeps after the lidar's gap (frames 5 on) must outnumber it.
const _: () = assert!(GAP_FRAMES - 5 > LANE_CLEAR_AFTER, "too short to step down");

/// What one `.rrd` holds: every entity of the recording with the components
/// logged on it (`Archetype:field`), and every `+ /path` expression of the
/// layout's views.
struct Rrd {
    entities: BTreeMap<String, BTreeSet<String>>,
    shown: Vec<String>,
    /// Every string the recording logged, by entity: lane states, log lines,
    /// documents and point labels -- what a reader of the viewer reads.
    texts: BTreeMap<String, Vec<String>>,
}

fn read_rrd(path: &Path) -> Rrd {
    let file = std::fs::File::open(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let msgs = DecoderApp::decode_eager(std::io::BufReader::new(file)).expect("an .rrd");
    let query = ViewContents::descriptor_query().component;
    let words = [
        StateChange::descriptor_state().component,
        TextLog::descriptor_text().component,
        TextDocument::descriptor_text().component,
        Points2D::descriptor_labels().component,
    ];
    let mut entities: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    let mut shown = Vec::new();
    let mut texts: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for msg in msgs {
        let LogMsg::ArrowMsg(store, arrow) = msg.expect("a message") else {
            continue;
        };
        let chunk = Chunk::from_arrow_msg(&arrow).expect("a chunk");
        let path = chunk.entity_path().to_string();
        let path = path.trim_start_matches('/').to_string();
        match store.kind() {
            StoreKind::Recording => {
                for w in &words {
                    for batch in chunk.iter_slices::<String>(*w) {
                        texts
                            .entry(path.clone())
                            .or_default()
                            .extend(batch.iter().map(|s| s.to_string()));
                    }
                }
                let comps = entities.entry(path).or_default();
                for d in chunk.component_descriptors() {
                    comps.insert(d.component.to_string());
                }
            }
            StoreKind::Blueprint => {
                if path.ends_with("/ViewContents") {
                    for batch in chunk.iter_slices::<String>(query) {
                        shown.extend(batch.iter().map(|s| s.to_string()));
                    }
                }
            }
        }
    }
    Rrd {
        entities,
        shown,
        texts,
    }
}

impl Rrd {
    fn shows(&self, entity: &str) -> bool {
        self.shown
            .iter()
            .filter_map(|c| c.strip_prefix("+ /"))
            .any(|c| match c.strip_suffix("/**") {
                Some(prefix) => entity == prefix || entity.starts_with(&format!("{prefix}/")),
                None => entity == c,
            })
    }
}

/// One fixture run with the whole dashboard on, written to `cam0.rrd` in a
/// fresh CWD, and what the file holds. Two camera frames against eight
/// sweeps: the first two sweeps pair and are answered, the rest expire, so
/// the run draws both an answer and a sweep without one. `detector` runs the
/// frozen camera detector, from a copy of the model in the CWD.
fn fixture_run(detector: bool) -> Rrd {
    let fx = FixtureDrive::with_velodyne(2, SWEEPS, PERIOD_NS).unwrap();
    fx.write_calib().unwrap();
    let cwd = TempDir::new().unwrap();
    if detector {
        let repo = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let model = cwd.path().join(MODEL_FILE);
        std::fs::create_dir_all(model.parent().unwrap()).unwrap();
        std::fs::copy(repo.join(MODEL_FILE), &model)
            .unwrap_or_else(|e| panic!("{}: {e}", repo.join(MODEL_FILE).display()));
    }
    let o = pipes(
        &cwd,
        &fx,
        &[
            "--name",
            "cov",
            "--rerun",
            "rrd",
            "--dashboard",
            "on",
            "--track",
            "on",
            "--detector",
            if detector { "on" } else { "off" },
            "--rate",
            "1.0",
            "--cap",
            "16",
            "--pair-wait-ms",
            "200",
        ],
    );
    assert_ok(&o);
    let rrd = read_rrd(&cwd.path().join("runs/cov/cam0.rrd"));
    assert!(!rrd.shown.is_empty(), "the file carries no layout");
    rrd
}

/// Every entity `rrd` logged is shown by a view of its own layout, and
/// every series and lane carries its style. `want` must be among them, so
/// the check is over something.
fn assert_shown_and_styled(rrd: &Rrd, want: &[&str]) {
    for w in want {
        assert!(
            rrd.entities.contains_key(*w),
            "nothing at {w}; logged: {:?}",
            rrd.entities.keys().collect::<Vec<_>>()
        );
    }
    let has = |comps: &BTreeSet<String>, archetype: &str| {
        comps
            .iter()
            .any(|c| c.starts_with(&format!("{archetype}:")))
    };
    for (path, comps) in &rrd.entities {
        // The SDK's own recording properties, and the two annotation
        // contexts, which are not drawn: they name the classes of what is.
        if path.starts_with("__") || comps.iter().all(|c| c.starts_with("AnnotationContext:")) {
            continue;
        }
        assert!(rrd.shows(path), "{path} is logged but no view shows it");
        if has(comps, "Scalars") {
            assert!(
                has(comps, "SeriesLines") || has(comps, "SeriesPoints"),
                "{path} is a series with no name or colour: {comps:?}"
            );
        }
        if has(comps, "StateChange") {
            assert!(
                has(comps, "StateConfiguration"),
                "{path} is a lane with no states' colours: {comps:?}"
            );
        }
    }
}

/// Every family the dashboard draws, the stages' own pictures and the
/// recorder's mirror alike.
const EVERY_FAMILY: [&str; 19] = [
    "camera/image",
    "camera/status",
    "answer/headline",
    "answer/line",
    "answer/ttc_warn",
    "pairing/camera",
    "pairing/window",
    "tracks/count/lidar_only",
    "lidar/voxels",
    "lidar/tracks",
    "lidar/answer",
    "lanes/pairing",
    "pairing/1_camera_done",
    "pairing/3_detect_done",
    "after_sweep/sweep_end",
    "after_sweep/2_reduce_done",
    "latency/table",
    "bytes/table",
    "graph/pipeline",
];

#[test]
fn every_entity_a_run_logs_is_shown_and_every_series_is_styled() {
    assert_shown_and_styled(&fixture_run(false), &EVERY_FAMILY);
}

/// The same with the camera detector, whose stage draws its own boxes -- and
/// runs in `proc`'s place, so the stage table names it and never `proc`.
#[test]
#[ignore = "needs models/yolox_nano.onnx (scripts/fetch_model.ps1)"]
fn with_the_detector_every_entity_is_shown_and_styled_too() {
    let mut want: Vec<&str> = EVERY_FAMILY.to_vec();
    want.push("camera/cam_det");
    let rrd = fixture_run(true);
    assert_shown_and_styled(&rrd, &want);
    let table = rrd.texts.get("latency/table").cloned().unwrap_or_default();
    let last = table.last().map(String::as_str).unwrap_or_default();
    assert!(last.contains("| camdet |"), "{last}");
    assert!(
        !last.contains("| proc |"),
        "proc ran beside the detector: {last}"
    );
}

/// Without it, the camera's half of each pair is `proc`'s, and the stage
/// table and the camera bar say so: `proc` is the camera stage, and the
/// table's every line is a stage the run had.
#[test]
fn without_the_detector_the_camera_stage_is_proc() {
    let rrd = fixture_run(false);
    let table = rrd.texts.get("latency/table").cloned().unwrap_or_default();
    let last = table.last().map(String::as_str).unwrap_or_default();
    assert!(last.contains("| proc |"), "{last}");
    assert!(!last.contains("| camdet |"), "{last}");
    for stage in ["cam0", "velo", "reduce", "detect", "track", "state"] {
        assert!(last.contains(&format!("| {stage} |")), "{stage}: {last}");
    }
    let bytes = rrd.texts.get("bytes/table").cloned().unwrap_or_default();
    let last = bytes.last().map(String::as_str).unwrap_or_default();
    assert!(last.contains("| 1 sweep |"), "{last}");
    assert!(last.contains("**built**"), "{last}");
}

/// A gap in either sensor's source is SHOWN, not only counted: the sensor's
/// own lane turns grey at the gap (`no sweep`, `no frame`), the pairing lane
/// says `source gap`, the answer's headline and the camera's stamp say there
/// is no answer for that frame and why, and the log carries one WARN event
/// per gap naming its frames, beside a WARN row per frame. And everything
/// drawn for it is shown and styled, like every other entity.
///
/// Long enough after the lidar's gap for its lane to step down from `no
/// sweep` (a lane clears after [`LANE_CLEAR_AFTER`] quiet rows), so what it
/// steps down TO is on the recording: `ok`, not the queue's own `skipping`.
/// And no lane downstream of either gap -- the camera's queue, the lidar's,
/// the detector's, the fusion's two inputs -- ever reads `skipping` for it.
#[test]
fn a_gap_in_either_source_is_shown_on_the_dashboard() {
    let fx = FixtureDrive::with_gaps(GAP_FRAMES, &[9], &[3, 4], PERIOD_NS).unwrap();
    fx.write_calib().unwrap();
    let cwd = TempDir::new().unwrap();
    let o = pipes(
        &cwd,
        &fx,
        &[
            "--name",
            "gaps",
            "--rerun",
            "rrd",
            "--dashboard",
            "on",
            "--track",
            "on",
            "--rate",
            "1.0",
            "--cap",
            "16",
        ],
    );
    assert_ok(&o);
    let rrd = read_rrd(&cwd.path().join("runs/gaps/cam0.rrd"));
    assert_shown_and_styled(
        &rrd,
        &[
            "lanes/lidar_queue",
            "lanes/camera_queue",
            "lanes/pairing",
            "log/events",
            "log/drops",
            "answer/headline",
            "camera/status",
        ],
    );
    let said = |entity: &str, needle: &str| {
        let texts = rrd.texts.get(entity).map(Vec::as_slice).unwrap_or(&[]);
        assert!(
            texts.iter().any(|t| t.contains(needle)),
            "{entity} never said {needle:?}: {texts:?}"
        );
    };
    said("lanes/lidar_queue", "no sweep");
    said("lanes/camera_queue", "no frame");
    said("lanes/pairing", "source gap");
    said("answer/headline", "## no answer for frame 3");
    said(
        "answer/headline",
        "no camera frame in the source (pair_absent_in_source)",
    );
    said("camera/status", "NO SWEEP · frame 3 · absent_in_source");
    said("camera/status", "NO FRAME · frame 9 · absent_in_source");
    said("log/events", "SourceGap");
    said("log/events", "lidar frames 3-4");
    said("log/events", "camera frame 9");
    said("log/drops", "velo seq 3 Missing absent_in_source");
    said("log/drops", "cam0 seq 9 Missing absent_in_source");
    // The positive control for the grey: a lane that turned grey is one that
    // was otherwise green, not one that never said anything else.
    said("lanes/lidar_queue", "ok");
    // And the source's gap is not drawn as the queue's loss: `velo->reduce`
    // goes from sweep 2 to sweep 5, and unless something really lost a sweep
    // in front of `reduce` the lidar lane steps down from `no sweep` to `ok`,
    // never to `skipping`.
    let csv = evidence(&cwd, "gaps");
    // The pipeline's own losses anywhere in front of a lane: a queue's drop,
    // or a driver's skip. The fusion's expiries are its output, not a lane's
    // input, and a frame absent in the source is the source's.
    let losses: Vec<String> = csv
        .rows
        .iter()
        .filter(|r| {
            csv.col(r, "outcome") != "Delivered"
                && csv.col(r, "reason") != "absent_in_source"
                && csv.col(r, "edge") != "track"
        })
        .map(|r| {
            format!(
                "{} seq {} {}",
                csv.col(r, "edge"),
                csv.col(r, "seq"),
                csv.col(r, "reason")
            )
        })
        .collect();
    let lane = |name: &str| {
        rrd.texts
            .get(&format!("lanes/{name}"))
            .cloned()
            .unwrap_or_default()
    };
    if losses.is_empty() {
        // Every lane either gap reaches: the camera's frame 9 crosses
        // `cam0->proc` and `cam_det->track`; the lidar's frames 3-4 cross
        // `velo->reduce`, `det->detect` and `obj->track`.
        for name in [
            "camera_queue",
            "lidar_queue",
            "detect_queue",
            "fusion_queue",
        ] {
            let states = lane(name);
            assert!(!states.is_empty(), "lanes/{name} said nothing");
            assert!(
                !states.iter().any(|t| t == "skipping" || t == "dropping"),
                "a gap in the source was drawn as lanes/{name}'s own loss: {states:?}"
            );
        }
        let lidar = lane("lidar_queue");
        let after = lidar.iter().skip_while(|t| t.as_str() != "no sweep").nth(1);
        assert_eq!(
            after.map(String::as_str),
            Some("ok"),
            "the lidar lane after `no sweep`: {lidar:?}"
        );
    } else {
        eprintln!("the pipeline lost samples, so the lanes may skip: {losses:?}");
    }
}
