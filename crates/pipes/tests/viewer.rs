//! The viewer, from the pipeline's side: `--rerun grpc` when there is no
//! viewer to stream to, and a viewer that stops taking data mid-run.
//!
//! ## No viewer
//!
//! The run starts the viewer itself from PATH, so the failure a fresh machine
//! can hit is "no `rerun` executable". That failure has to name the one
//! install line, and it has to happen before a frame is replayed, not at the
//! flush at the end of the run.
//!
//! The viewer is kept off the test by giving the binary an empty `PATH`. That
//! alone is not enough: the SDK joins whatever is listening on its port
//! before it looks at PATH at all, and a viewer left open from an earlier
//! `pipes run` listens on the default 9876 for as long as its window is up.
//! Under that condition a test on the default port would stream a fixture
//! run into the leftover viewer and observe nothing, and an earlier version
//! of this test did exactly that: it printed a note and returned, which
//! cargo counts as a pass. So the run under test gets a port of its own,
//! one nothing listens on, through the hidden `--rerun-port`.

use std::net::{Ipv4Addr, TcpListener};

use pipes_kitti::testing::FixtureDrive;
use tempfile::TempDir;

mod common;
use common::{assert_ok, command, evidence, pipes, stderr, stdout, summary, Csv};

/// A port nothing is listening on: bound to 0 by the OS and released again.
/// The window between the release and the binary's own probe is a few
/// milliseconds on an otherwise idle loopback, and a collision would fail
/// the test loudly (the run would join the stranger), not pass it.
fn free_port() -> u16 {
    let l = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).expect("bind port 0");
    l.local_addr().expect("local addr").port()
}

#[test]
fn grpc_without_a_viewer_on_path_fails_first_and_names_the_install_line() {
    let fx = FixtureDrive::new(4).unwrap();
    let cwd = TempDir::new().unwrap();
    let port = free_port().to_string();
    let o = command(
        &cwd,
        &fx,
        &["--name", "novw", "--rerun", "grpc", "--rerun-port", &port],
    )
    .env("PATH", "")
    .output()
    .expect("spawn pipes");
    assert_eq!(
        o.status.code(),
        Some(1),
        "--- stdout ---\n{}\n--- stderr ---\n{}",
        stdout(&o),
        stderr(&o)
    );
    let err = stderr(&o);
    assert!(
        err.contains("python -m pip install rerun-sdk==0.38.1"),
        "the error does not say what to install:\n{err}"
    );
    assert!(err.contains("--rerun rrd"), "no way out is offered:\n{err}");
    // The port it looked on, not a hard-coded 9876.
    assert!(
        err.contains(&format!("port {port} ")),
        "the error does not name the port it looked on ({port}):\n{err}"
    );
    // Failed before the run, not after it: no frame was replayed and nothing
    // vouches for the directory.
    assert!(
        !stdout(&o).contains("driver admitted="),
        "the run went ahead without a sink:\n{}",
        stdout(&o)
    );
    assert!(!common::run_dir(&cwd, "novw").join("_COMPLETE").exists());
}

// ## A viewer that stops taking data
//
// The Rerun SDK does not drop: once its buffers are full, `log` blocks until
// the viewer takes more. When the stages called `log` themselves, a viewer
// frozen for 10 s blocked them and their queues evicted a fifth of drive
// 0005's sweeps. Now only the viewer thread calls it, and the stages' drawings
// wait on the viewer's own queue, `draw->viewer`, which drops its oldest.
//
// The hidden `--viewer-delay-ms` freezes the run's `.rrd` for that long from
// the moment the clock starts, and cuts the SDK's buffers from 100 MiB to 256
// KiB so the stall reaches `log` within a few of the fixture's samples, as a
// frozen viewer's reaches it within about six seconds of drive 0005.

/// Sweeps, and camera frames, in the fixture: paced, two seconds of them.
const N: usize = 80;
const PERIOD_NS: i64 = 25_000_000;
/// Longer than the SDK's cut-down buffers and the viewer's queue can hold at
/// the fixture's rate of drawing.
const FREEZE_MS: &str = "1500";
/// `viewer::VIEWER_CAP`: the positive control must draw more than this, or
/// the frozen run could not overflow the queue.
const VIEWER_CAP: usize = 256;

/// Every row off the viewer's edge that is not a delivery -- a queue's drop,
/// a driver's missing frame, an expired set -- as `edge seq outcome reason`.
fn pipeline_losses(ev: &Csv) -> Vec<String> {
    ev.rows
        .iter()
        .filter(|r| ev.col(r, "edge") != "draw->viewer" && ev.col(r, "outcome") != "Delivered")
        .map(|r| {
            format!(
                "{} seq {} {} {}",
                ev.col(r, "edge"),
                ev.col(r, "seq"),
                ev.col(r, "outcome"),
                ev.col(r, "reason")
            )
        })
        .collect()
}

#[test]
fn a_frozen_viewer_loses_drawings_on_its_own_edge_and_nowhere_else() {
    let fx = FixtureDrive::with_velodyne(N, N, PERIOD_NS).unwrap();
    fx.write_calib().unwrap();
    let cwd = TempDir::new().unwrap();
    let run = |name: &str, delay_ms: &str| {
        let o = pipes(
            &cwd,
            &fx,
            &[
                "--name",
                name,
                "--rerun",
                "rrd",
                "--dashboard",
                "on",
                "--track",
                "on",
                "--cap",
                "16",
                "--rate",
                "1.0",
                "--viewer-delay-ms",
                delay_ms,
            ],
        );
        assert_ok(&o);
        o
    };

    // ---- positive control: a viewer that keeps up loses nothing ----------
    run("thawed", "0");
    let ev = evidence(&cwd, "thawed");
    assert_eq!(
        pipeline_losses(&ev),
        Vec::<String>::new(),
        "the fixture is too fast for this host"
    );
    let drawings = ev.with("edge", "draw->viewer");
    assert!(
        drawings.len() > 2 * VIEWER_CAP,
        "{} drawings: too few to overflow the viewer's queue",
        drawings.len()
    );
    let lost: Vec<&&Vec<String>> = drawings
        .iter()
        .filter(|r| ev.col(r, "outcome") != "Delivered")
        .collect();
    assert!(
        lost.is_empty(),
        "a viewer that kept up lost {} drawings",
        lost.len()
    );

    // ---- frozen: the drawings are lost, and only the drawings ------------
    let o = run("frozen", FREEZE_MS);
    let ev = evidence(&cwd, "frozen");
    assert_eq!(
        pipeline_losses(&ev),
        Vec::<String>::new(),
        "a frozen viewer cost the pipeline samples"
    );
    let drawings = ev.with("edge", "draw->viewer");
    let lost: Vec<&&Vec<String>> = drawings
        .iter()
        .filter(|r| ev.col(r, "outcome") != "Delivered")
        .collect();
    assert!(
        !lost.is_empty(),
        "the freeze dropped no drawing, so it tested nothing"
    );
    for r in &lost {
        assert_eq!(
            (ev.col(r, "outcome").as_str(), ev.col(r, "reason").as_str()),
            ("DroppedOldest", "evicted")
        );
    }
    // Every drawing has one row, and the queue's own count agrees with them.
    let line = stdout(&o)
        .lines()
        .find(|l| l.starts_with("INVARIANT edge=draw->viewer "))
        .map(str::to_string)
        .unwrap_or_else(|| {
            panic!(
                "no viewer invariant:
{}",
                stdout(&o)
            )
        });
    assert!(line.ends_with("-> OK"), "{line}");
    assert!(
        line.contains(&format!(
            " dropped={} admitted={}",
            lost.len(),
            drawings.len()
        )),
        "{line}: {} rows, {} of them lost",
        drawings.len(),
        lost.len()
    );
    let s = summary(&cwd, "frozen");
    assert_eq!(
        s["viewer"]["dropped"],
        serde_json::json!(lost.len()),
        "{}",
        s["viewer"]
    );
    assert_eq!(
        s["viewer"]["dropped_by_reason"]["evicted"],
        serde_json::json!(lost.len())
    );
}
