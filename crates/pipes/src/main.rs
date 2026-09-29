#![deny(clippy::unwrap_used)]
#![deny(clippy::expect_used)]
#![forbid(unsafe_code)]
//! `pipes`: single-stream zero-copy pipeline experiments (sprint 1).

mod admission; // M4 two-step admit (D8)
mod cli; // clap types
mod consumers; // M4 proc + rerun threads
mod dashboard; // the recording's entity tree
mod drives; // `pipes drives`: what the data root holds
mod record; // M4 evidence channel + rec thread
mod run; // M4 full mode
mod summary; // M5 summary, invariants
mod viewer; // the viewer's own queue and thread: the only caller of the SDK's `log`

use clap::Parser;

fn main() {
    let result = match cli::Cli::parse().cmd {
        cli::Cmd::Drives(a) => drives::main(a),
        cli::Cmd::Run(a) => run::main(a),
    };
    let Err(e) = result else { return };

    // `main` returning `Result` prints the error with `Debug`, which for a
    // nested error is a wall of struct syntax: `Driver(Io { path: "...",
    // source: Os { code: 3, .. } })`. Print the `Display` chain instead — each
    // layer names itself and `source()` supplies the next — so the common
    // first-run failure reads as a sentence and ends at the real cause.
    eprintln!("error: {e}");
    let mut src = e.source();
    while let Some(s) = src {
        eprintln!("  caused by: {s}");
        src = s.source();
    }

    // A stage thread panicked: the run produced no result, so exit 3 — distinct
    // from 2 (the run completed but an invariant FAILed, exited at the check)
    // and from 1 (any other error). The stage's own panic message already
    // reached stderr through the default hook.
    let code = match e.downcast_ref::<run::RunError>() {
        Some(run::RunError::Panicked(_)) => 3,
        _ => 1,
    };
    std::process::exit(code);
}
