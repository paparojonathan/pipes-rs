//! The detect stage measured against the real drive, with the ego motion the
//! pipeline cannot yet replay.
//!
//! **This does not run in the gates**, and that is deliberate rather than a
//! gap being hidden. It needs `2011_09_26_drive_0005_sync` on disk, which CI
//! does not have, and it takes tens of seconds. `cargo test -p pipes-kitti
//! --test detect_real_drive -- --ignored --nocapture` runs it and prints
//! everything the write-up quotes.
//!
//! It exists because there is one number the pipeline **cannot** produce, and
//! it is the one that makes the sanity check convincing. `pipes run` reports
//! the UNCOMPENSATED persistence, because correcting for the vehicle's own
//! motion needs `oxts/`, and `StreamId::OXTS` has no driver. Uncompensated is
//! about 36 % against a 2 % control — real, but the sharper statement is that
//! applying the **measured** physical motion roughly doubles it, because a
//! detector emitting noise could not be improved by being told how the car
//! moved. That statement needs the oxts rows, so it is made here, against the
//! same shipped code the pipeline runs.
//!
//! Everything under `#[ignore]` is a claim nothing checks by default, so this
//! file is written to fail loudly rather than skip: if the drive is not where
//! it is expected, it says so with the path it looked at.

use std::path::{Path, PathBuf};

use pipes_kitti::detect::{
    cloud_points, detect_schema, detections_f32, persistence, DetectPlan, Detector, EgoMotion,
    Persistence, DETECTION_LANES, PERSISTENCE_GATE_M, RANGE_LIMIT_M,
};
use pipes_kitti::velo::{build_velo_batch, read_sweep, sweep_file_name, velo_dir, velo_schema};
use pipes_kitti::voxel::{voxel_schema, VoxelSize, Voxelizer};

const DATE: &str = "2011_09_26";
const DRIVE: &str = "2011_09_26_drive_0005_sync";
const N_SWEEPS: usize = 154;

/// Where the dataset is, derived from the crate's own location rather than
/// from a machine-specific absolute path — the same rule
/// `pipes_kitti::timestamps`'s real-data test follows.
fn kitti_root() -> PathBuf {
    PathBuf::from(std::env::var("PIPES_KITTI_ROOT").unwrap_or_else(|_| {
        concat!(env!("CARGO_MANIFEST_DIR"), "/../../../data/kitti").to_string()
    }))
}

/// One `oxts/data/*.txt` row's motion fields.
///
/// A local parser rather than a driver, because this file is a measurement
/// harness and not a pipeline stage: shipping an `OxtsDriver` to satisfy one
/// ignored test would be scope this work does not have. The three indices are
/// KITTI's own, 0-based: 8 `vf` forward, 9 `vl` leftward, 19 `wz` yaw rate.
///
/// `wz` is the yaw rate in the EARTH frame; the vehicle-frame one is field 22
/// (`wu`). On a drive this close to level the two differ by the cosine of the
/// pitch, well under a part in a thousand, and 19 is the field the design note
/// named — so it is the one used, and the difference is noted rather than
/// silently swapped.
fn oxts_motion(path: &Path) -> EgoMotion {
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let f: Vec<f64> = text
        .split_whitespace()
        .map(|t| t.parse().unwrap_or_else(|e| panic!("{path:?}: `{t}`: {e}")))
        .collect();
    assert!(f.len() > 22, "{}: {} fields", path.display(), f.len());
    EgoMotion {
        vf: f[8],
        vl: f[9],
        yaw_rate: f[19],
        dt: 0.0,
    }
}

/// Deterministic generator for the uniform-random control, so a re-run of this
/// harness gives the same control figure.
struct Lcg(u64);

impl Lcg {
    fn unit(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
}

fn pct(p: Persistence) -> f64 {
    p.fraction().unwrap_or(f64::NAN) * 100.0
}

/// The whole chain on the real drive, and the sanity check with the ego motion
/// the pipeline has no driver for.
///
/// Four numbers come out and only the pattern across all four means anything:
/// compensated high, uncompensated lower, both controls near zero. The
/// controls are the half that gives it teeth — especially the rotated one,
/// which has **identical** spatial statistics to the real detections, so a
/// high score there would mean the real figure was density rather than
/// structure.
#[test]
#[ignore = "needs KITTI at PIPES_KITTI_ROOT"]
fn ego_compensated_persistence_beats_its_controls_on_drive_0005() {
    let root = kitti_root();
    let dir = velo_dir(&root, DATE, DRIVE);
    assert!(
        dir.is_dir(),
        "no velodyne data at {} -- set PIPES_KITTI_ROOT",
        dir.display()
    );
    let ts = pipes_kitti::timestamps::parse_timestamps(&dir.join("timestamps.txt"))
        .unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(ts.len(), N_SWEEPS);

    let v = VoxelSize::default();
    let (vs, ds, ss) = (voxel_schema(), detect_schema(), velo_schema());
    let mut vx = Voxelizer::new();
    let mut det = Detector::new();

    // Per sweep: the detection lanes, and the plan that produced them.
    let mut lanes: Vec<Vec<f32>> = Vec::with_capacity(N_SWEEPS);
    let mut plans: Vec<DetectPlan> = Vec::with_capacity(N_SWEEPS);
    let (mut raw_bytes, mut vox_bytes, mut det_bytes) = (0u64, 0u64, 0u64);
    for (i, trigger) in ts.iter().enumerate() {
        let buf = read_sweep(&dir.join("data").join(sweep_file_name(i)))
            .unwrap_or_else(|e| panic!("sweep {i}: {e}"));
        raw_bytes += buf.len() as u64;
        let (sweep, _) = build_velo_batch(buf, trigger.0, &ss).unwrap_or_else(|e| panic!("{e}"));
        let points = cloud_points(&sweep);
        vx.reserve(points.len());
        let vplan = vx.plan(points, v);
        let (cloud, _) = vx
            .build(points, &vplan, v, trigger.0, &vs)
            .unwrap_or_else(|e| panic!("{e}"));
        vox_bytes += (vplan.out_points as u64) * 16;

        let voxels = cloud_points(&cloud);
        det.reserve(voxels.len());
        let dplan = det.plan(voxels, v);
        let (batch, _) = det
            .build(
                voxels,
                &dplan,
                v,
                trigger.0,
                i as i64,
                vplan.source_points,
                &ds,
            )
            .unwrap_or_else(|e| panic!("{e}"));
        det_bytes += u64::from(dplan.detections) * 44;
        lanes.push(detections_f32(&batch).unwrap_or_default().to_vec());
        plans.push(dplan);
    }

    // ---- what each stage produced, as the write-up quotes it --------------
    let n = N_SWEEPS as u64;
    let dets: u64 = plans.iter().map(|p| u64::from(p.detections)).sum();
    let (lo, hi) = plans.iter().fold((u32::MAX, 0), |(lo, hi), p| {
        (lo.min(p.detections), hi.max(p.detections))
    });
    println!("sweeps            {N_SWEEPS}");
    println!("raw bytes/sweep   {}", raw_bytes / n);
    println!("voxel bytes/sweep {}", vox_bytes / n);
    println!("det bytes/sweep   {}", det_bytes / n);
    println!("detections/sweep  mean {} min {lo} max {hi}", dets / n);
    println!(
        "fragments         {:.1}%   merged {}",
        plans
            .iter()
            .map(|p| u64::from(p.fragment_detections))
            .sum::<u64>() as f64
            / dets as f64
            * 100.0,
        plans
            .iter()
            .map(|p| u64::from(p.merged_detections))
            .sum::<u64>()
    );
    let tilts: Vec<f64> = plans.iter().map(|p| p.ground.tilt_deg()).collect();
    println!(
        "ground tilt       mean {:.2} deg  max {:.2} deg  fallbacks {}",
        tilts.iter().sum::<f64>() / tilts.len() as f64,
        tilts.iter().copied().fold(0.0, f64::max),
        plans.iter().filter(|p| !p.ground.fitted).count()
    );

    // The regression net on the real data: these hold for the whole drive and
    // each one is a different way the stage could have gone quietly wrong.
    assert_eq!(
        plans.iter().map(|p| p.key_collisions).sum::<u32>(),
        0,
        "two voxels recovered the same grid index, so `floor(centroid / v)` \
         no longer recovers reduce's grid"
    );
    assert_eq!(
        plans.iter().filter(|p| !p.ground.fitted).count(),
        0,
        "the ground fit fell back to a fixed height on a drive this flat"
    );
    assert!(
        tilts.iter().copied().fold(0.0, f64::max) < 5.0,
        "the fitted ground tilt left the range the single-plane assumption is \
         defensible in"
    );
    assert!(
        (150..=210).contains(&(dets / n)),
        "detections/sweep = {} is outside the measured band",
        dets / n
    );

    // ---- the sanity check, and its two controls ---------------------------
    let mut real = Persistence::default();
    let mut uncompensated = Persistence::default();
    let mut turned_ctl = Persistence::default();
    let mut random_ctl = Persistence::default();
    let mut rng = Lcg(0x5EED);
    let mut turned: Vec<f32> = Vec::new();
    let mut random: Vec<f32> = Vec::new();
    for i in 0..N_SWEEPS - 1 {
        let path = root
            .join(DATE)
            .join(DRIVE)
            .join("oxts")
            .join("data")
            .join(format!("{i:010}.txt"));
        let dt = (ts[i + 1].0 - ts[i].0) as f64 * 1e-9;
        assert!((0.05..0.2).contains(&dt), "sweep {i}: dt = {dt}");
        let ego = EgoMotion {
            dt,
            ..oxts_motion(&path)
        };
        let (a, b) = (&lanes[i], &lanes[i + 1]);
        real.add(persistence(a, b, ego, PERSISTENCE_GATE_M));
        // The same pair with the vehicle held still: the number the pipeline
        // itself reports, and the one the measured motion has to improve on.
        uncompensated.add(persistence(
            a,
            b,
            EgoMotion {
                dt,
                ..Default::default()
            },
            PERSISTENCE_GATE_M,
        ));

        // Control 1: the next sweep's detections, turned a quarter turn about
        // the sensor. Identical spatial statistics.
        turned.clear();
        turned.extend_from_slice(b);
        for d in turned.as_chunks_mut::<DETECTION_LANES>().0 {
            let (x, y) = (d[0], d[1]);
            d[0] = -y;
            d[1] = x;
        }
        turned_ctl.add(persistence(a, &turned, ego, PERSISTENCE_GATE_M));

        // Control 2: uniform-random points over the disc the stage can report
        // in, at the heights the real detections occupy, and the same number
        // of them.
        let zs: Vec<f32> = b
            .as_chunks::<DETECTION_LANES>()
            .0
            .iter()
            .map(|d| d[2])
            .collect();
        let zlo = zs.iter().copied().fold(f32::INFINITY, f32::min);
        let zhi = zs.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        random.clear();
        random.resize(b.len(), 0.0);
        for d in random.as_chunks_mut::<DETECTION_LANES>().0 {
            let r = f64::from(RANGE_LIMIT_M) * rng.unit().sqrt();
            let th = std::f64::consts::TAU * rng.unit();
            d[0] = (r * th.cos()) as f32;
            d[1] = (r * th.sin()) as f32;
            d[2] = zlo + (zhi - zlo) * rng.unit() as f32;
        }
        random_ctl.add(persistence(a, &random, ego, PERSISTENCE_GATE_M));
    }

    println!(
        "\npersistence over {} pairs, gate {PERSISTENCE_GATE_M} m",
        N_SWEEPS - 1
    );
    println!("  ego-compensated, real          {:.1}%", pct(real));
    println!(
        "  uncompensated, real            {:.1}%",
        pct(uncompensated)
    );
    println!("  CONTROL turned 90 deg          {:.1}%", pct(turned_ctl));
    println!("  CONTROL uniform-random         {:.1}%", pct(random_ctl));

    let r = real.fraction().unwrap_or(0.0);
    let u = uncompensated.fraction().unwrap_or(0.0);
    // The population is every detection of sweeps 0..152, so a check that
    // quietly skipped pairs would show up here rather than as a better number.
    let last = u64::from(plans[N_SWEEPS - 1].detections);
    assert_eq!(
        u64::from(real.total),
        dets - last,
        "the check did not run over every detection it should have"
    );
    assert!(r > 0.5, "ego-compensated persistence {r:.3}");
    // The second, independent signal: applying the MEASURED motion has to
    // improve the match substantially. A detector emitting noise could not be
    // improved by being told how the car moved.
    assert!(
        r > u * 1.5,
        "applying the measured ego motion barely moved the match rate:          {r:.3} against {u:.3} uncompensated"
    );
    for (name, c) in [
        ("turned 90 deg", turned_ctl),
        ("uniform random", random_ctl),
    ] {
        let f = c.fraction().unwrap_or(1.0);
        assert!(
            f < 0.05,
            "control `{name}` scored {f:.3}, so the check does not discriminate"
        );
    }
}
