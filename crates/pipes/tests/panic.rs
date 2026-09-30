//! A panic in a stage thread must not deadlock the run, must not leave a
//! truncated evidence file behind, and must not be mistaken for a completed
//! run by anything downstream.
//!
//! This is the only named proof of the ordered-shutdown path in `run::execute`
//! (`cli.rs` and `consumers.rs` both cite this test by name), and
//! `--panic-at-frame` exists solely so it can be written against the binary
//! that actually ships rather than a `#[cfg(feature)]` variant of it.
//!
//! Before that shutdown was made unconditional, `proc_handle.join()` returning
//! `Err` took a `?` out of the middle of the sequence: `sink.close()` was never
//! called, the `rec` thread blocked forever in `q.pop()`, its `csv::Writer` was
//! never flushed, and `evidence.csv` was left cut off at the last 8 KiB buffer
//! boundary. Every assertion below is a piece of that failure.

use std::io::Read;
use std::process::{Output, Stdio};
use std::time::Duration;

use pipes_kitti::testing::FixtureDrive;
use tempfile::TempDir;

mod common;
use common::{command, run_dir, stderr, stdout, Csv, EVENT_HEADER, EVIDENCE_HEADER};

/// Poll interval and poll count: a deadlock must not hang CI forever, so the
/// run is bounded at roughly 30 s. The bound is approximate on purpose --
/// nothing here measures time, it only refuses to wait unboundedly.
const TICK: Duration = Duration::from_millis(10);
const TICKS: u32 = 3_000;

/// Runs the binary with a deadline, and **kills it** if the deadline passes.
///
/// Killing matters: a deadlocked child inherits nothing but still holds its
/// pipes and its CWD open, so leaving it running keeps the cargo invocation
/// itself alive long after the test has reported its failure — 600 s in the
/// run that proved this test catches the regression — and on Windows blocks
/// the `TempDir` from being removed.
fn run_bounded(cwd: &TempDir, fx: &FixtureDrive, args: &[&str]) -> Output {
    // `spawn` inherits stdio where `output` pipes it, so without these the
    // child's panic message would land on the test harness's own stderr and
    // `Output::stderr` would be empty -- the assertions below would then be
    // checking a blank string.
    let mut child = command(cwd, fx, args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn pipes");
    // Drained on their own threads, so a chatty run cannot block on a full
    // pipe and be mistaken for a deadlock.
    let drain = |mut pipe: Option<Box<dyn Read + Send>>| {
        std::thread::spawn(move || {
            let mut buf = Vec::new();
            if let Some(p) = pipe.as_mut() {
                let _ = p.read_to_end(&mut buf);
            }
            buf
        })
    };
    let out_t = drain(
        child
            .stdout
            .take()
            .map(|p| Box::new(p) as Box<dyn Read + Send>),
    );
    let err_t = drain(
        child
            .stderr
            .take()
            .map(|p| Box::new(p) as Box<dyn Read + Send>),
    );

    let mut status = None;
    for _ in 0..TICKS {
        match child.try_wait().expect("try_wait") {
            Some(s) => {
                status = Some(s);
                break;
            }
            None => std::thread::sleep(TICK),
        }
    }
    let Some(status) = status else {
        let _ = child.kill();
        let _ = child.wait();
        panic!(
            "pipes did not exit within about {:?} -- the shutdown deadlocked",
            TICK * TICKS
        );
    };
    Output {
        status,
        stdout: out_t.join().unwrap_or_default(),
        stderr: err_t.join().unwrap_or_default(),
    }
}

#[test]
fn panic_in_proc_shuts_down_cleanly_and_exits_3() {
    // `--rate 1` with the fixture's 100 ms period: frame 1 is reached in about
    // 300 ms. Pacing matters here -- at a high rate the targeted frame can be
    // dropped by admission before `proc` ever sees it, and the hook then
    // silently does nothing, which reads as a broken test rather than a
    // skipped one. `--cap 16` for 8 frames removes drops entirely, and the
    // assertions on stderr below prove the panic really fired.
    let fx = FixtureDrive::new(8).unwrap();
    let cwd = TempDir::new().unwrap();
    let o = run_bounded(
        &cwd,
        &fx,
        &[
            "--name",
            "panicrun",
            "--rate",
            "1",
            "--cap",
            "16",
            "--panic-at-frame",
            "1",
        ],
    );
    let err = stderr(&o);

    // 3, not 2 (an invariant FAILed) and not 1 (any other error).
    assert_eq!(
        o.status.code(),
        Some(3),
        "--- stdout ---\n{}\n--- stderr ---\n{err}",
        stdout(&o)
    );
    // The panic actually fired, rather than the run failing for some other
    // reason that also happens to exit non-zero.
    assert!(
        err.contains("injected panic at frame 1"),
        "the injected panic never reached stderr:\n{err}"
    );
    // The stage's own panic message reaches stderr through the default hook,
    // and `RunError::Panicked` names the stage.
    assert!(
        err.contains("panicked") && err.contains("proc"),
        "stderr does not name the panicking stage:\n{err}"
    );

    let dir = run_dir(&cwd, "panicrun");

    // Evidence written before the panic is flushed and readable. `Csv::read`
    // fails on any row whose width differs from the header, which is exactly
    // what a writer killed mid-row leaves behind.
    let ev_path = dir.join("evidence.csv");
    let ev_text = std::fs::read_to_string(&ev_path).unwrap();
    assert_eq!(ev_text.lines().next().unwrap(), EVIDENCE_HEADER);
    assert!(
        ev_text.ends_with('\n'),
        "evidence.csv was cut off mid-row: it does not end with a newline"
    );
    let ev = Csv::read(&ev_path);
    assert!(
        !ev.rows.is_empty(),
        "no evidence survived the panic at all; the rec thread never flushed"
    );

    // events.csv keeps its header and records how the run ended.
    let events = Csv::read(&dir.join("events.csv"));
    assert_eq!(events.header.join(","), EVENT_HEADER);
    let shutdown: Vec<_> = events
        .rows
        .iter()
        .filter(|r| events.col(r, "kind") == "Shutdown")
        .collect();
    assert_eq!(
        shutdown.len(),
        1,
        "expected one Shutdown event: {events:?}",
        events = events.rows
    );
    assert_eq!(events.col(shutdown[0], "detail"), "panicked:proc");
    assert_eq!(events.col(shutdown[0], "stage"), "run");

    // A panicked run produced no result, so nothing downstream may treat it as
    // one: no completion marker and no summary.
    assert!(
        !dir.join("_COMPLETE").exists(),
        "_COMPLETE vouches for a run that never finished"
    );
    assert!(
        !dir.join("summary.json").exists(),
        "summary.json from a run that produced no summary"
    );
}

#[test]
fn panic_free_run_writes_the_marker_and_a_clean_shutdown() {
    // The positive control. Without it, every assertion above would also pass
    // against a binary that never writes `_COMPLETE`, never writes
    // `summary.json` and never records a Shutdown event at all.
    let fx = FixtureDrive::new(8).unwrap();
    let cwd = TempDir::new().unwrap();
    let o = run_bounded(&cwd, &fx, &["--name", "clean", "--cap", "16"]);
    assert!(
        o.status.success(),
        "exit {:?}\n{}\n{}",
        o.status.code(),
        stdout(&o),
        stderr(&o)
    );
    let dir = run_dir(&cwd, "clean");
    assert!(dir.join("_COMPLETE").is_file());
    assert!(dir.join("summary.json").is_file());
    let events = Csv::read(&dir.join("events.csv"));
    let shutdown: Vec<_> = events
        .rows
        .iter()
        .filter(|r| events.col(r, "kind") == "Shutdown")
        .collect();
    assert_eq!(shutdown.len(), 1);
    assert_eq!(events.col(shutdown[0], "detail"), "clean");
}
