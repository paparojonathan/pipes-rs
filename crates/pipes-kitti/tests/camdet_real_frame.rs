//! The frozen camera detector on real KITTI frames.
//!
//! **Not in the gates**, on the terms the other real-drive tests are: it needs
//! `2011_09_26_drive_0005_sync` on disk and the model fetched by
//! `scripts/fetch_model.ps1`, neither of which CI has. Run it with
//!
//! ```text
//! cargo test --release -p pipes-kitti --test camdet_real_frame -- --ignored --nocapture
//! ```
//!
//! It fails loudly, naming the path it looked at, when either is missing.
//! The second test walks all 154 frames twice, about half a minute in
//! release.
//!
//! The reference boxes are the ones the model's reference implementation
//! (ONNX Runtime, cv2 resize) found on frame 0 at this input size, recorded
//! when the detector was chosen; the Rust preprocessing uses a different
//! bilinear resize, so a box is matched by class and overlap, not bytes.

use std::path::PathBuf;

use arrow::array::RecordBatch;
use arrow::ipc::writer::StreamWriter;
use pipes_core::clock::now;
use pipes_kitti::camdet::{
    build_cam_det_batch, cam_det_schema, class_name, detect_frame, iou, Det, Model, MODEL_FILE,
};
use pipes_kitti::track::CamRef;

fn kitti_root() -> PathBuf {
    PathBuf::from(std::env::var("PIPES_KITTI_ROOT").unwrap_or_else(|_| {
        concat!(env!("CARGO_MANIFEST_DIR"), "/../../../data/kitti").to_string()
    }))
}

fn model_path() -> PathBuf {
    PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/../..")).join(MODEL_FILE)
}

fn frame(i: usize) -> (Vec<u8>, u32, u32) {
    let p = kitti_root()
        .join("2011_09_26/2011_09_26_drive_0005_sync/image_02/data")
        .join(format!("{i:010}.png"));
    let img = image::open(&p)
        .unwrap_or_else(|e| panic!("{}: {e}", p.display()))
        .to_rgb8();
    let (w, h) = img.dimensions();
    (img.into_raw(), w, h)
}

fn reference(class_id: u32, b: [f32; 4]) -> Det {
    Det {
        x0: b[0],
        y0: b[1],
        x1: b[2],
        y1: b[3],
        class_id,
        score: 1.0,
        row: 0,
    }
}

#[test]
#[ignore = "needs drive_0005 and models/yolox_nano.onnx"]
fn frame_0_finds_the_reference_objects_and_repeats_bit_for_bit() {
    let path = model_path();
    let model = Model::load(&path).unwrap_or_else(|e| panic!("{e}"));
    let (rgb, w, h) = frame(0);
    assert_eq!((w, h), (1242, 375));

    let t = now();
    let dets = detect_frame(&model, &rgb, w, h).unwrap_or_else(|e| panic!("{e}"));
    println!(
        "frame 0: {:.1} ms (uncertified)",
        (now().0 - t.0) as f64 * 1e-6
    );
    for d in &dets {
        println!(
            "  {} {:.3} [{:.0},{:.0},{:.0},{:.0}]",
            class_name(d.class_id),
            d.score,
            d.x0,
            d.y0,
            d.x1,
            d.y1
        );
    }

    // person, person, bicycle, car: the reference's four principal objects.
    for r in [
        reference(0, [1108.0, 173.0, 1197.0, 314.0]),
        reference(0, [775.0, 166.0, 880.0, 361.0]),
        reference(1, [783.0, 267.0, 884.0, 374.0]),
        reference(2, [294.0, 167.0, 451.0, 285.0]),
    ] {
        assert!(
            dets.iter()
                .any(|d| d.class_id == r.class_id && iou(d, &r) > 0.7),
            "no {} near {:?}",
            class_name(r.class_id),
            [r.x0, r.y0, r.x1, r.y1]
        );
    }
    // Sorted by score, highest first.
    assert!(dets.windows(2).all(|p| p[0].score >= p[1].score));

    // Frozen: the same frame gives the same detections, to the bit.
    let again = detect_frame(&model, &rgb, w, h).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(
        dets.iter()
            .map(|d| [d.x0, d.y0, d.x1, d.y1, d.score].map(f32::to_bits))
            .collect::<Vec<_>>(),
        again
            .iter()
            .map(|d| [d.x0, d.y0, d.x1, d.y1, d.score].map(f32::to_bits))
            .collect::<Vec<_>>()
    );

    // The negative control: a flat grey frame of the same size has nothing
    // in it, and the detector says so.
    let grey = vec![128u8; rgb.len()];
    let none = detect_frame(&model, &grey, w, h).unwrap_or_else(|e| panic!("{e}"));
    assert!(none.is_empty(), "{none:?}");
}

/// A batch as the bytes an Arrow IPC stream would carry: the whole payload,
/// schema metadata included, so "the same" means the same on disk.
fn ipc_bytes(b: &RecordBatch) -> Vec<u8> {
    let mut out = Vec::new();
    let mut w = StreamWriter::try_new(&mut out, &b.schema()).unwrap_or_else(|e| panic!("{e}"));
    w.write(b).unwrap_or_else(|e| panic!("{e}"));
    w.finish().unwrap_or_else(|e| panic!("{e}"));
    drop(w);
    out
}

/// **Frozen, over the whole drive:** two models loaded separately -- what two
/// runs of `pipes run` do -- turn every frame of drive_0005 into the SAME
/// `CAM_DET` bytes. Also prints what the detector found and what each frame
/// cost, which are measurements for the run's report rather than assertions.
///
/// The positive control is that the comparison can fail: one detection's
/// score moved by one ulp makes different bytes.
#[test]
#[ignore = "needs drive_0005 and models/yolox_nano.onnx"]
fn every_frame_of_drive_0005_gives_the_same_bytes_from_two_loads_of_the_model() {
    let a = Model::load(&model_path()).unwrap_or_else(|e| panic!("{e}"));
    let b = Model::load(&model_path()).unwrap_or_else(|e| panic!("{e}"));
    let schema = cam_det_schema(&a.sha256);
    let n =
        std::fs::read_dir(kitti_root().join("2011_09_26/2011_09_26_drive_0005_sync/image_02/data"))
            .unwrap_or_else(|e| panic!("{e}"))
            .count();
    assert!(n > 100, "drive_0005 has {n} frames");
    let mut ms: Vec<f64> = Vec::with_capacity(n);
    let mut counts = [0u64; 80];
    let mut total = 0usize;
    let mut control_done = false;
    for i in 0..n {
        let (rgb, w, h) = frame(i);
        let r = CamRef {
            seq: i as u64,
            tov_ns: i as i64,
            width: w,
            height: h,
        };
        let t = now();
        let da = detect_frame(&a, &rgb, w, h).unwrap_or_else(|e| panic!("{e}"));
        ms.push((now().0 - t.0) as f64 * 1e-6);
        let db = detect_frame(&b, &rgb, w, h).unwrap_or_else(|e| panic!("{e}"));
        let ba = build_cam_det_batch(r, &da, &schema).unwrap_or_else(|e| panic!("{e}"));
        let bb = build_cam_det_batch(r, &db, &schema).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(
            ipc_bytes(&ba),
            ipc_bytes(&bb),
            "frame {i}: two loads of the model disagree"
        );
        if !control_done && !da.is_empty() {
            let mut moved = da.clone();
            moved[0].score = f32::from_bits(moved[0].score.to_bits() + 1);
            let bm = build_cam_det_batch(r, &moved, &schema).unwrap_or_else(|e| panic!("{e}"));
            assert_ne!(ipc_bytes(&ba), ipc_bytes(&bm), "the comparison is blind");
            control_done = true;
        }
        total += da.len();
        for d in &da {
            counts[d.class_id as usize] += 1;
        }
    }
    assert!(control_done, "no frame had a detection to perturb");
    ms.sort_by(f64::total_cmp);
    let q = |p: f64| ms[((ms.len() - 1) as f64 * p).round() as usize];
    println!(
        "{n} frames, {total} detections ({:.1} per frame); detect_frame ms p50 {:.1} p90 {:.1} max {:.1} (uncertified)",
        total as f64 / n as f64,
        q(0.5),
        q(0.9),
        q(1.0)
    );
    let mut by_count: Vec<(u64, usize)> = counts
        .iter()
        .enumerate()
        .filter(|(_, c)| **c > 0)
        .map(|(k, c)| (*c, k))
        .collect();
    by_count.sort_by(|x, y| y.0.cmp(&x.0).then(x.1.cmp(&y.1)));
    for (c, k) in by_count {
        println!("  {} {c}", class_name(k as u32));
    }
}
