//! What `--help` offers: two commands, and the eleven flags of `run` a reader
//! needs to see the pipeline or make it fail on purpose. The experiment's
//! knobs are still accepted (`hide = true`), and that is pinned too, because
//! a hidden flag that stopped parsing would fail every test that uses it in
//! a way that reads as a pipeline failure.

use std::process::Command;

fn help(args: &[&str]) -> String {
    let o = Command::new(env!("CARGO_BIN_EXE_pipes"))
        .args(args)
        .output()
        .expect("spawn pipes");
    assert!(o.status.success(), "{args:?} exited {:?}", o.status.code());
    String::from_utf8_lossy(&o.stdout).into_owned()
}

/// The `--long` names on the option lines of a help page, in order.
fn long_flags(help: &str) -> Vec<String> {
    help.lines()
        .filter_map(|l| {
            let l = l.trim_start();
            let l = l.strip_prefix("-h, ").unwrap_or(l);
            let rest = l.strip_prefix("--")?;
            let end = rest
                .find(|c: char| !(c.is_ascii_alphanumeric() || c == '-'))
                .unwrap_or(rest.len());
            Some(rest[..end].to_string())
        })
        .collect()
}

#[test]
fn the_top_level_help_offers_run_and_drives_and_nothing_else() {
    let h = help(&["--help"]);
    let commands: Vec<&str> = h
        .lines()
        .skip_while(|l| !l.starts_with("Commands:"))
        .skip(1)
        .take_while(|l| !l.trim().is_empty())
        .filter_map(|l| l.split_whitespace().next())
        .collect();
    assert_eq!(commands, ["drives", "run"], "{h}");
}

#[test]
fn run_help_offers_the_eleven_core_flags_and_no_knob() {
    let h = help(&["run", "--help"]);
    let mut flags = long_flags(&h);
    flags.retain(|f| f != "help");
    let mut want = vec![
        "kitti-root",
        "drive",
        "name",
        "cap",
        "consumer-delay-ms",
        "track",
        "detector",
        "pair-wait-ms",
        "pair-stale-ms",
        "rerun",
        "dashboard",
    ];
    want.sort_unstable();
    flags.sort_unstable();
    assert_eq!(flags, want, "{h}");
    // The two sinks that only measure are accepted but not offered. The
    // short page puts the values on the flag's own line.
    let short = help(&["run", "-h"]);
    assert!(
        short.contains("[possible values: rrd, grpc]"),
        "--rerun offers more than rrd|grpc:\n{short}"
    );
    assert!(!h.contains("- null"), "{h}");
    // And the defaults `--help` states are the ones a plain `run` uses.
    for line in [
        "[default: grpc]",
        "[default: on]", // --track, --detector and --dashboard
    ] {
        assert!(h.contains(line), "missing {line} in:\n{h}");
    }
    assert!(
        h.matches("[default: on]").count() == 3,
        "expected --track, --detector and --dashboard all to default to on:\n{h}"
    );
    // The detector's help names the download, in a form PowerShell runs.
    assert!(
        h.contains(r"scripts\fetch_model.ps1"),
        "--detector does not say where its model comes from:\n{h}"
    );
}

/// One flag's paragraph of `run --help`, whitespace collapsed, so a phrase
/// the page wrapped across two lines still matches.
fn flag_help(help: &str, flag: &str) -> String {
    let opening = |l: &str| {
        let l = l.trim_start();
        l.strip_prefix("-h, ").unwrap_or(l).starts_with("--")
    };
    let mut lines = help.lines().skip_while(|l| {
        !(opening(l)
            && l.trim_start()
                .split(|c: char| c.is_whitespace() || c == '=' || c == '<')
                .next()
                == Some(flag))
    });
    let first = lines.next().unwrap_or_default();
    std::iter::once(first)
        .chain(lines.take_while(|l| !opening(l)))
        .flat_map(str::split_whitespace)
        .collect::<Vec<_>>()
        .join(" ")
}

/// `--cap` and `--consumer-delay-ms` act on the camera stage whose output
/// reaches the answer -- `camdet` with the detector, `proc` without it -- and
/// their help says so, where a reader meets them: which queue, which stage,
/// and what an omitted `--cap` is on each. The caveat they used to carry,
/// that they reached the answer only under `--detector off`, is gone with the
/// behaviour it described.
#[test]
fn the_camera_knobs_say_which_queue_they_act_on() {
    let h = help(&["run", "--help"]);
    let cap = flag_help(&h, "--cap");
    assert!(cap.starts_with("--cap"), "no paragraph for --cap:\n{h}");
    for phrase in [
        "`cam0->camdet`, the detector's, when the detector runs",
        "`cam0->proc` when it does not",
        "Omitted, it is 1 on `cam0->camdet` and 4 on `cam0->proc`",
    ] {
        assert!(cap.contains(phrase), "--cap lacks {phrase:?}: {cap}");
    }
    let delay = flag_help(&h, "--consumer-delay-ms");
    assert!(delay.starts_with("--consumer-delay-ms"), "{h}");
    assert!(
        delay.contains("`camdet` when the detector runs, `proc` when it does not"),
        "--consumer-delay-ms does not say which stage it slows: {delay}"
    );
    assert!(
        !h.contains("only under `--detector off`"),
        "the old caveat is still on the page:\n{h}"
    );
    // Positive control for the paragraph cut: the phrase that opens `--cap`'s
    // paragraph is not in the detector's, so the cut stopped at the next flag.
    let det = flag_help(&h, "--detector");
    assert!(det.starts_with("--detector"), "{h}");
    assert!(!det.contains("Omitted, it is 1"), "{det}");
}

#[test]
fn the_hidden_knobs_still_parse() {
    // `--help` after the knobs: clap parses everything before it, so an
    // unknown flag would exit 2 here instead of printing the page.
    let h = help(&[
        "run",
        "--rate",
        "inf",
        "--policy",
        "block",
        "--block-max-wait-ms",
        "1",
        "--reuse-output",
        "--lidar",
        "off",
        "--reduce",
        "off",
        "--detect",
        "off",
        "--voxel-size-m",
        "0.5",
        "--rerun",
        "null",
        "--rerun-port",
        "9877",
        "--rerun-host",
        "host.docker.internal",
        "--panic-at-frame",
        "3",
        "--viewer-delay-ms",
        "1",
        "--help",
    ]);
    let offered = long_flags(&h);
    assert!(offered.iter().any(|f| f == "track"), "{h}");
    for hidden in [
        "--rate",
        "--policy",
        "--block-max-wait-ms",
        "--reuse-output",
        "--lidar",
        "--reduce",
        "--detect",
        "--voxel-size-m",
        "--rerun-port",
        "--rerun-host",
        "--panic-at-frame",
        "--viewer-delay-ms",
    ] {
        // By whole flag name: `--detect` is a prefix of the visible
        // `--detector`, so a substring test cannot tell them apart.
        let name = hidden.trim_start_matches("--");
        assert!(
            !offered.iter().any(|f| f == name),
            "{hidden} is on the help page:\n{h}"
        );
        assert!(
            !h.lines()
                .flat_map(|l| l.split(|c: char| c.is_whitespace() || c == '`'))
                .any(|w| w == hidden),
            "{hidden} is mentioned on the help page:\n{h}"
        );
    }
    // Positive control for the whole-name test: a visible flag that one of
    // the hidden names is a prefix of is found by it.
    assert!(offered.iter().any(|f| f == "detector"), "{h}");
}
