//! Shared harness for the `crates/pipes` integration tests.
//!
//! `crates/pipes` is binary-only, so there is nothing to `use pipes::…`; the
//! pipeline is tested by running the real binary through
//! `env!("CARGO_BIN_EXE_pipes")`. Every run gets its own `TempDir` as its CWD,
//! so `runs/` lands there and the tests are safe in parallel.
//!
//! `tests/common/` is a subdirectory, so cargo does not build it as a test
//! target of its own; each test file pulls it in with `mod common;`. Each of
//! those files uses a different part of it, hence the blanket `dead_code`
//! allow — an unused helper here is not a defect, it is another file's helper.

#![allow(dead_code)]

use std::path::Path;
use std::process::{Command, Output};

use pipes_kitti::testing::{FixtureDrive, FIXTURE_DRIVE};
use tempfile::TempDir;

/// The 26 columns of `evidence.csv`, in `Evidence` field order.
pub const EVIDENCE_HEADER: &str = "run_id,epoch,arrival_seq,stream,seq,edge,stage,tov_start_ns,tov_end_ns,due_ns,arrival_ns,enqueued_ns,dequeued_ns,proc_start_ns,proc_end_ns,queue_wait_ns,measurement_age_ns,outcome,reason,depth_at_push,depth_after_pop,storage_id,bytes_alloc,payload_bytes,decode_ns,push_blocked_ns";

/// The 6 columns of `events.csv`, in `Event` field order (`record::EVENT_HEADER`).
pub const EVENT_HEADER: &str = "run_id,host_ns,arrival_seq_after,kind,stage,detail";

/// The argument list for one `pipes run` against `fx`, with its CWD in `cwd`.
///
/// The binary's own defaults show everything (`--rerun grpc --dashboard on
/// --track on --detector on`); the tests want the plain pipeline. `--rate inf`
/// finishes in milliseconds; `--rerun null` keeps the second consumer (and its
/// evidence rows and `storage_id` re-derivation) while opening no sink at all;
/// `--dashboard off` draws nothing; `--track off` keeps the chain at the
/// detections, so a lidar fixture without a calibration still runs; and
/// `--detector off` fuses with `proc`'s frame reference, because a test's CWD
/// is a temp dir with no `models/` in it, and CI has no model at all. A flag
/// the caller supplies *replaces* the default rather than joining it: clap
/// rejects a repeated flag outright, so passing both would make an overriding
/// test exit 2 on an argument error and read as a pipeline failure.
pub fn run_args<'a>(fx_root: &'a str, args: &[&'a str]) -> Vec<&'a str> {
    let mut all: Vec<&str> = vec!["run", "--kitti-root", fx_root, "--drive", FIXTURE_DRIVE];
    for (flag, default) in [
        ("--rate", "inf"),
        ("--rerun", "null"),
        ("--dashboard", "off"),
        ("--track", "off"),
        ("--detector", "off"),
    ] {
        if !args.contains(&flag) {
            all.push(flag);
            all.push(default);
        }
    }
    all.extend_from_slice(args);
    all
}

/// A `Command` for the real binary, ready to spawn or run.
pub fn command(cwd: &TempDir, fx: &FixtureDrive, args: &[&str]) -> Command {
    let root = fx.root().display().to_string();
    let mut c = Command::new(env!("CARGO_BIN_EXE_pipes"));
    c.current_dir(cwd.path())
        // The developer's own dataset must not leak in through the environment.
        .env_remove("PIPES_KITTI_ROOT")
        .args(run_args(&root, args));
    c
}

/// Runs the binary to completion.
pub fn pipes(cwd: &TempDir, fx: &FixtureDrive, args: &[&str]) -> Output {
    command(cwd, fx, args).output().expect("spawn pipes")
}

pub fn stdout(o: &Output) -> String {
    String::from_utf8_lossy(&o.stdout).into_owned()
}

pub fn stderr(o: &Output) -> String {
    String::from_utf8_lossy(&o.stderr).into_owned()
}

/// One whole line of stdout equals `line`.
pub fn has(o: &Output, line: &str) -> bool {
    stdout(o).lines().any(|l| l == line)
}

pub fn assert_line(o: &Output, line: &str) {
    assert!(
        has(o, line),
        "missing stdout line:\n  {line}\n--- stdout ---\n{}\n--- stderr ---\n{}",
        stdout(o),
        stderr(o)
    );
}

pub fn assert_ok(o: &Output) {
    assert!(
        o.status.success(),
        "exit {:?} (2 = an invariant FAILed, 3 = a stage panicked)\n--- stdout ---\n{}\n--- stderr ---\n{}",
        o.status.code(),
        stdout(o),
        stderr(o)
    );
}

/// The first stdout line starting with `prefix`.
pub fn line_with<'a>(text: &'a str, prefix: &str) -> &'a str {
    text.lines()
        .find(|l| l.starts_with(prefix))
        .unwrap_or_else(|| panic!("no line starting {prefix:?} in:\n{text}"))
}

/// `key=<number>` out of a whitespace-separated line.
pub fn field(line: &str, key: &str) -> i64 {
    line.split_whitespace()
        .find_map(|t| t.strip_prefix(key))
        .unwrap_or_else(|| panic!("no {key} in {line}"))
        .parse()
        .unwrap_or_else(|e| panic!("{key} in {line}: {e}"))
}

/// A parsed CSV with named column access. Every row is checked against the
/// header width, so a ragged file fails here instead of silently shifting a
/// later assertion onto the wrong column.
pub struct Csv {
    pub header: Vec<String>,
    pub rows: Vec<Vec<String>>,
}

impl Csv {
    pub fn read(path: &Path) -> Csv {
        let text =
            std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        let mut r = csv::Reader::from_reader(text.as_bytes());
        let header: Vec<String> = r
            .headers()
            .expect("csv header")
            .iter()
            .map(str::to_string)
            .collect();
        let rows: Vec<Vec<String>> = r
            .records()
            .map(|rec| {
                let rec = rec.unwrap_or_else(|e| panic!("{}: {e}", path.display()));
                assert_eq!(
                    rec.len(),
                    header.len(),
                    "{}: row of {} fields against a {}-column header",
                    path.display(),
                    rec.len(),
                    header.len()
                );
                rec.iter().map(str::to_string).collect()
            })
            .collect();
        Csv { header, rows }
    }

    pub fn col(&self, row: &[String], name: &str) -> String {
        let i = self
            .header
            .iter()
            .position(|h| h == name)
            .unwrap_or_else(|| panic!("no column {name} in {:?}", self.header));
        row[i].clone()
    }

    pub fn with(&self, name: &str, value: &str) -> Vec<&Vec<String>> {
        self.rows
            .iter()
            .filter(|r| self.col(r, name) == value)
            .collect()
    }
}

pub fn run_dir(cwd: &TempDir, name: &str) -> std::path::PathBuf {
    cwd.path().join("runs").join(name)
}

pub fn evidence(cwd: &TempDir, name: &str) -> Csv {
    Csv::read(&run_dir(cwd, name).join("evidence.csv"))
}

/// A run's `runs/<name>/run.json`, parsed.
pub fn run_json(cwd: &TempDir, name: &str) -> serde_json::Value {
    let path = run_dir(cwd, name).join("run.json");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    serde_json::from_str(&text).expect("run.json is not valid JSON")
}

pub fn summary(cwd: &TempDir, name: &str) -> serde_json::Value {
    let path = run_dir(cwd, name).join("summary.json");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    serde_json::from_str(&text).expect("summary.json is not valid JSON")
}

/// One edge's entry of a run's `summary.json` `stages` array, by name.
pub fn stage(s: &serde_json::Value, edge: &str) -> serde_json::Value {
    s["stages"]
        .as_array()
        .expect("stages array")
        .iter()
        .find(|st| st["edge"] == serde_json::json!(edge))
        .unwrap_or_else(|| panic!("no {edge} stage in {s}"))
        .clone()
}

/// The `cam0->proc` entry of a run's `summary.json` `stages` array.
pub fn proc_stage(s: &serde_json::Value) -> serde_json::Value {
    stage(s, "cam0->proc")
}

/// Samples an edge's queue threw away, all three ways it can. Every consumer
/// edge in the binary balances `delivered + dropped == admitted`, and every
/// edge of the derived chain is a fixed-cap DropOldest queue that may
/// legitimately evict on a loaded host -- so a test that wants "what the far
/// end saw" against "what the stage produced" reads this rather than assuming
/// it is 0.
///
/// 0 for an edge with no entry. `stages` is built from the evidence rows, so
/// an edge nothing travelled -- `track->state` on a run where every set
/// expired -- has no entry at all, and no rows is no drops.
pub fn dropped_on(s: &serde_json::Value, edge: &str) -> u64 {
    s["stages"]
        .as_array()
        .expect("stages array")
        .iter()
        .find(|st| st["edge"] == serde_json::json!(edge))
        .map_or(0, |st| {
            u64_of(st, "dropped_oldest") + u64_of(st, "dropped_newest") + u64_of(st, "timeout")
        })
}

/// A `u64` field of a `summary.json` object, by name.
pub fn u64_of(v: &serde_json::Value, key: &str) -> u64 {
    v[key]
        .as_u64()
        .unwrap_or_else(|| panic!("{key} is {} , not a number", v[key]))
}
