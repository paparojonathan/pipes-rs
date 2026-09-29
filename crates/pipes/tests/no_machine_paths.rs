//! Guard: no source file under `crates/` may contain an absolute path into
//! somebody's home directory.
//!
//! Two such paths shipped in this repo — the `--kitti-root` default and a test
//! fallback — and neither was noticed for eight milestones, because they work
//! perfectly on the machine that wrote them and fail everywhere else with a
//! "path not found" that names the dataset rather than the real problem. They
//! come back the moment someone debugs with a literal path, so this is a test
//! rather than a review habit.
//!
//! The needles are assembled at run time from fragments. If they appeared as
//! literals here, this file would match itself and the guard would fail on its
//! own source — and, worse, anyone fixing that by skipping this file would
//! create a hole. `matcher_detects_a_planted_path` exists for the opposite
//! reason: a search that silently matches nothing is indistinguishable from a
//! clean tree, which is exactly how the first grep for these paths came back
//! empty while both literals were sitting in the source.

use std::path::{Path, PathBuf};

/// Absolute home-directory prefixes, built so they never appear verbatim here.
fn needles() -> Vec<String> {
    let user = ["U", "sers"].concat();
    vec![
        format!("C:\\{user}"),
        format!("C:/{user}"),
        format!("/{user}/"),
        ["/h", "ome/"].concat(),
    ]
}

fn crates_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crates/pipes has a parent")
        .to_path_buf()
}

fn rust_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let entries = std::fs::read_dir(dir).unwrap_or_else(|e| panic!("read {}: {e}", dir.display()));
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            // `target/` can appear as a per-crate build directory.
            if path.file_name().is_some_and(|n| n == "target") {
                continue;
            }
            rust_files(&path, out);
        } else if path.extension().is_some_and(|e| e == "rs") {
            out.push(path);
        }
    }
}

#[test]
fn no_absolute_home_paths_under_crates() {
    let dir = crates_dir();
    let mut files = Vec::new();
    rust_files(&dir, &mut files);
    assert!(
        files.len() > 5,
        "only {} .rs files found under {} — the walk is broken, and a guard \
         that scans nothing passes forever",
        files.len(),
        dir.display()
    );

    let needles = needles();
    let mut hits = Vec::new();
    for file in &files {
        let text = match std::fs::read_to_string(file) {
            Ok(t) => t,
            Err(e) => panic!("read {}: {e}", file.display()),
        };
        for (lineno, line) in text.lines().enumerate() {
            for needle in &needles {
                if line.contains(needle.as_str()) {
                    hits.push(format!("{}:{}", file.display(), lineno + 1));
                }
            }
        }
    }
    assert!(
        hits.is_empty(),
        "absolute home-directory path(s) in source — they work only on the \
         machine that wrote them. Use a path relative to `CARGO_MANIFEST_DIR`, \
         or read `PIPES_KITTI_ROOT`:\n  {}",
        hits.join("\n  ")
    );
}

#[test]
fn matcher_detects_a_planted_path() {
    let planted = format!(
        "let p = r\"C:\\{}\\someone\\data\";",
        ["U", "sers"].concat()
    );
    let needles = needles();
    assert!(
        needles.iter().any(|n| planted.contains(n.as_str())),
        "the matcher missed a planted path, so a clean result from \
         `no_absolute_home_paths_under_crates` would prove nothing"
    );
    assert!(
        !needles
            .iter()
            .any(|n| "let p = \"../data/kitti\";".contains(n.as_str())),
        "the matcher fires on a relative path, so it would block the fix"
    );
}
