//! The fusion and the answer, measured against the real drive — **and the
//! answer checked against the raw cloud it came from**.
//!
//! **This does not run in the gates**, on the same terms as
//! `detect_real_drive`: it needs `2011_09_26_drive_0005_sync` on disk, which
//! CI does not have, and it takes tens of seconds. Run it with
//! `cargo test -p pipes-kitti --test track_real_drive -- --ignored --nocapture`.
//!
//! It exists for one thing the pipeline cannot do for itself. `pipes run`
//! reports "the nearest object in the vehicle's path is 12.11 m away on
//! average"; nothing in that run checks the number against anything. Here the
//! same claim is recomputed **from the raw laser returns**, by a completely
//! different route — no voxels, no clustering, no tracking, just "filter the
//! sweep to the corridor, drop the ground, take the smallest forward
//! coordinate" — and the two are compared sweep by sweep.
//!
//! What that check shares with the pipeline, stated so the independence is not
//! overclaimed: the **fitted ground plane**, because a ground rule of its own
//! would be testing two ground rules against each other rather than testing
//! the answer. Everything else is independent — the grid, the clustering, the
//! minimum cluster size, the association, the corridor test on boxes rather
//! than points, and the choice of which track is nearest.
//!
//! And a **control**, because an agreement with nothing to disagree with is
//! not evidence: the same raw check run down a corridor displaced 6 m to the
//! side must NOT agree with the answer.
//!
//! # What this does NOT establish
//!
//! It is a **comparison, reported as percentiles**, not a tolerance test.
//! `pipes_kitti::state` used to claim this harness checked the answer "to
//! within one voxel edge"; it never did, and the claim has been removed rather
//! than the harness tightened, because the evidence does not support it. What
//! is asserted below is a median under 1 m, agreement within 0.5 m on more
//! than half the sweeps, a non-empty presence window, and a control that
//! disagrees. What is measured and merely printed is the rest of the
//! distribution — including the **worst sweep in each direction, by name**,
//! because the near-side one is a real defect: the answer's `distance_m` is
//! the near face of a box while the corridor test is a separate projection of
//! that same box, so a wide structure can be selected on material that is not
//! in the corridor at all. A percentile hides that; a named sweep does not.
//!
//! # What the first version of this check got wrong, and what it found
//!
//! It asked for the nearest non-ground **return**, and reported that the
//! answer was a median of **+8.09 m** further away than that. The answer was
//! right and the check was wrong, and the reason is worth keeping because it
//! is a fact about this dataset rather than about this code.
//!
//! Every sweep of drive_0005 contains **one or two returns at x ~ 2.50 m,
//! y ~ +-0.47 m, z ~ -0.91 m, with reflectance 0.00**, in the same place every
//! time. That is 0.82 m above the road surface, 2.5 m directly in front of the
//! sensor, and it does not move over 154 sweeps: it is **the recording
//! vehicle's own bonnet**. A check that takes the nearest return finds the car
//! it is mounted on.
//!
//! `detect` does not report it, and not by accident: it is one or two returns
//! and [`pipes_kitti::detect::MIN_CLUSTER_VOXELS`] is three. So the naive
//! check was measuring the distance between "something reflected" and "a
//! structure large enough to be an object" — which is exactly the rule the
//! detector exists to apply.
//!
//! The check below therefore applies the **same** rule to the raw returns,
//! built from the same two numbers and no new ones: at least
//! `MIN_CLUSTER_VOXELS` non-ground returns within `MIN_CLUSTER_VOXELS` voxel
//! edges of one another along x. It uses no grid, no connected components and
//! no tracking, so it is still an independent route to the answer.

use std::path::{Path, PathBuf};

use pipes_kitti::calib::Calib;
use pipes_kitti::detect::{
    cloud_points, detect_schema, detection_rows, DetectPlan, Detector, GroundPlane,
    MIN_CLUSTER_VOXELS, RANGE_LIMIT_M,
};
use pipes_kitti::state::{
    build_state_batch, object_rows, read_answer, state_schema, StateAnswer, StateMeta,
    CORRIDOR_HALF_WIDTH_M, OBJECT_BYTES,
};
use pipes_kitti::track::{
    association_gate_m, track_age_s, track_observations, track_rows, track_schema, CamRef, Pairing,
    TrackMeta, Tracker, TRACK_BYTES,
};
use pipes_kitti::velo::{build_velo_batch, read_sweep, sweep_file_name, velo_dir, velo_schema};
use pipes_kitti::voxel::{voxel_schema, VoxelSize, Voxelizer};

const DATE: &str = "2011_09_26";
const DRIVE: &str = "2011_09_26_drive_0005_sync";
const N_SWEEPS: usize = 154;

/// Height above the fitted plane at which a raw return stops being road, in
/// metres.
///
/// One voxel edge, which is the band `detect` itself classifies ground with —
/// so the raw check and the pipeline disagree about the ground by nothing,
/// and any disagreement in the answer is about the clustering and the
/// tracking, which is what is being checked.
const GROUND_BAND_M: f64 = 0.20;

/// Lateral displacement of the control corridor, metres: far enough to be a
/// different part of the scene, near enough to still have returns in it.
const CONTROL_OFFSET_M: f32 = 6.0;

/// Length of the x window a raw "structure" must fill, metres.
///
/// `MIN_CLUSTER_VOXELS * 0.20 m` — the span the detector's own minimum cluster
/// occupies. Not a new parameter: the same two numbers the pipeline already
/// uses, applied to raw returns instead of to voxels.
const STRUCTURE_M: f32 = MIN_CLUSTER_VOXELS as f32 * 0.20;

fn kitti_root() -> PathBuf {
    PathBuf::from(std::env::var("PIPES_KITTI_ROOT").unwrap_or_else(|_| {
        concat!(env!("CARGO_MANIFEST_DIR"), "/../../../data/kitti").to_string()
    }))
}

/// The nearest non-ground raw return in a corridor, straight off the sweep.
///
/// **This is the independent half of the check.** It is deliberately the
/// dumbest possible implementation of the same question: no grid, no
/// clustering, no minimum size, no tracking. `y_offset` displaces the corridor
/// sideways for the control.
fn raw_nearest(points: &[[f32; 4]], ground: GroundPlane, y_offset: f32) -> Option<f32> {
    let mut best: Option<f32> = None;
    for p in points {
        if !p.iter().all(|c| c.is_finite()) {
            continue;
        }
        let (x, y) = (p[0], p[1] - y_offset);
        if x <= 0.0 || y.abs() > CORRIDOR_HALF_WIDTH_M {
            continue;
        }
        if x.hypot(p[1]) > RANGE_LIMIT_M {
            continue;
        }
        if ground.height([p[0], p[1], p[2]]) <= GROUND_BAND_M {
            continue;
        }
        best = Some(best.map_or(x, |b: f32| b.min(x)));
    }
    best
}

/// Every non-ground raw return in the corridor, as `x`, ascending.
fn corridor_xs(points: &[[f32; 4]], ground: GroundPlane, y_offset: f32) -> Vec<f32> {
    let mut xs: Vec<f32> = points
        .iter()
        .filter(|p| p.iter().all(|c| c.is_finite()))
        .filter(|p| {
            let (x, y) = (p[0], p[1] - y_offset);
            x > 0.0
                && y.abs() <= CORRIDOR_HALF_WIDTH_M
                && x.hypot(p[1]) <= RANGE_LIMIT_M
                && ground.height([p[0], p[1], p[2]]) > GROUND_BAND_M
        })
        .map(|p| p[0])
        .collect();
    xs.sort_by(f32::total_cmp);
    xs
}

/// The nearest raw **structure** in the corridor: the smallest `x` at which
/// at least [`MIN_CLUSTER_VOXELS`] non-ground returns fall inside a window of
/// [`STRUCTURE_M`].
///
/// **This is the independent half of the check.** No grid, no connected
/// components, no minimum-size gate borrowed from the detector's code — a
/// sliding window over sorted x, applying the detector's own RULE with none of
/// its machinery. See the module docs for why the nearer "any single return"
/// version measured the recording vehicle's bonnet.
fn raw_nearest_structure(points: &[[f32; 4]], ground: GroundPlane, y_offset: f32) -> Option<f32> {
    let xs = corridor_xs(points, ground, y_offset);
    let need = MIN_CLUSTER_VOXELS as usize;
    for (i, &x) in xs.iter().enumerate() {
        let j = i + need - 1;
        if j >= xs.len() {
            break;
        }
        if xs[j] - x <= STRUCTURE_M {
            return Some(x);
        }
    }
    None
}

/// Non-ground raw returns within `w` metres of `x0`, inside the reported
/// object's own lateral extent: **the presence check**.
///
/// An invented distance has nothing at it. The y band is the TRACK's, widened
/// by `w`, and not the corridor's, because the answer selects a track whose
/// BOX overlaps the corridor -- a wide object can qualify while its nearest
/// returns sit outside the 1.82 m the car would drive through. Checking the
/// corridor instead found 0 returns on some sweeps, which was the check being
/// narrower than the claim rather than the claim being wrong.
fn returns_in_object(
    points: &[[f32; 4]],
    ground: GroundPlane,
    x0: f32,
    w: f32,
    ylo: f32,
    yhi: f32,
) -> usize {
    points
        .iter()
        .filter(|p| p.iter().all(|c| c.is_finite()))
        .filter(|p| {
            (p[0] - x0).abs() <= w
                && p[1] >= ylo - w
                && p[1] <= yhi + w
                && ground.height([p[0], p[1], p[2]]) > GROUND_BAND_M
        })
        .count()
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(f64::total_cmp);
    if v.is_empty() {
        return f64::NAN;
    }
    v[v.len() / 2]
}

/// The whole chain on the real drive, the fusion against the real camera
/// timestamps, and the answer against the raw cloud.
#[test]
#[ignore = "needs KITTI at PIPES_KITTI_ROOT"]
fn the_answer_agrees_with_the_raw_cloud_and_not_with_a_displaced_control() {
    let root = kitti_root();
    let dir = velo_dir(&root, DATE, DRIVE);
    assert!(
        dir.is_dir(),
        "no velodyne data at {} -- set PIPES_KITTI_ROOT",
        dir.display()
    );
    let calib = Calib::load(&root, DATE).unwrap_or_else(|e| panic!("{e}"));

    // The three instants a sweep has, and the camera's one. Read here rather
    // than taken on trust: the pairing rule is a claim about these files.
    let ts = |name: &str| {
        pipes_kitti::timestamps::parse_timestamps(&dir.join(name)).unwrap_or_else(|e| panic!("{e}"))
    };
    let starts = ts("timestamps_start.txt");
    let triggers = ts("timestamps.txt");
    let ends = ts("timestamps_end.txt");
    let cam_dir: &Path = &root.join(DATE).join(DRIVE).join("image_02");
    let cams = pipes_kitti::timestamps::parse_timestamps(&cam_dir.join("timestamps.txt"))
        .unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(starts.len(), N_SWEEPS);
    assert_eq!(cams.len(), N_SWEEPS);

    let v = VoxelSize::default();
    let (vs, ds, ts_schema, ss) = (
        voxel_schema(),
        detect_schema(),
        track_schema(),
        velo_schema(),
    );
    let mut vx = Voxelizer::new();
    let mut det = Detector::new();
    let mut tk = Tracker::new();

    // Per sweep: the answer, the raw check, and the control.
    let mut agree: Vec<f64> = Vec::with_capacity(N_SWEEPS);
    // `(sweep, answer, corridor structure)` for every sweep in `agree`, so the
    // WORST one can be named. Percentiles alone hide the failure mode this
    // check exists to expose: the answer's distance is the near face of a BOX
    // and the corridor test is a separate projection of the same box, so a wide
    // structure can be selected on material that is not in the corridor at all.
    // A p90 is compatible with that being rare; a named worst sweep is not.
    let mut agree_at: Vec<(usize, f32, f32)> = Vec::with_capacity(N_SWEEPS);
    let mut control: Vec<f64> = Vec::with_capacity(N_SWEEPS);
    // The naive "nearest return" figure, kept because it is the one that found
    // the ego vehicle's bonnet, and its size is the detector's minimum-cluster
    // rule expressed in metres.
    let mut bonnet: Vec<f64> = Vec::with_capacity(N_SWEEPS);
    let mut present: Vec<usize> = Vec::with_capacity(N_SWEEPS);
    let mut present_ctl: Vec<usize> = Vec::with_capacity(N_SWEEPS);
    let mut answers: Vec<(usize, StateAnswer, Option<f32>, Option<f32>)> =
        Vec::with_capacity(N_SWEEPS);
    let mut plans: Vec<DetectPlan> = Vec::with_capacity(N_SWEEPS);
    let (mut paired, mut pair_age_ns) = (0usize, 0i64);
    let (mut emitted_total, mut ambiguous_total, mut in_frame_total) = (0u64, 0u64, 0u64);
    let (mut det_bytes, mut trk_bytes, mut ans_bytes) = (0u64, 0u64, 0u64);
    // The raw basis of the byte chain, summed from the returns THEMSELVES.
    // It used to be `source_voxels * 16 * 4` -- the voxel count scaled by the
    // reduction's own nominal 4x -- which is a number derived from the answer
    // it was meant to check. It read 1,968,256 B/sweep against the pipeline's
    // measured 1,947,870, 1.05 % high, so every ratio printed below it was
    // optimistic by that much. Points times 16 is what the file contains.
    let mut raw_bytes = 0u64;
    // The same quantity by an independent route: the voxelizer's own count of
    // the returns it consumed, which is what travels down the chain as
    // `source_point_count`. The two are asserted equal below, and that is the
    // assertion the old formula would have failed.
    let mut source_points_total = 0u64;
    // Ages in seconds on the sensor clock, and the observation counts that
    // used to stand in for them. Their disagreement is the coasting a count
    // hid: `coasted_rows` are emitted rows whose life in sweeps is longer
    // than their observations, and `worst_gap` is the largest difference.
    let (mut observations_total, mut age_s_total, mut age_s_max) = (0u64, 0f64, 0f32);
    let (mut coasted_rows, mut worst_gap) = (0u64, 0i64);
    let period_s = (triggers[N_SWEEPS - 1].0 - triggers[0].0) as f64 * 1e-9 / (N_SWEEPS - 1) as f64;

    for i in 0..N_SWEEPS {
        let buf = read_sweep(&dir.join("data").join(sweep_file_name(i)))
            .unwrap_or_else(|e| panic!("sweep {i}: {e}"));
        let (sweep, _) =
            build_velo_batch(buf, triggers[i].0, &ss).unwrap_or_else(|e| panic!("{e}"));
        let points = cloud_points(&sweep);
        raw_bytes += points.len() as u64 * 16;
        vx.reserve(points.len());
        let vplan = vx.plan(points, v);
        let (cloud, _) = vx
            .build(points, &vplan, v, triggers[i].0, &vs)
            .unwrap_or_else(|e| panic!("{e}"));

        source_points_total += u64::from(vplan.source_points);
        let voxels = cloud_points(&cloud);
        det.reserve(voxels.len());
        let dplan = det.plan(voxels, v);
        let (dbatch, _) = det
            .build(
                voxels,
                &dplan,
                v,
                triggers[i].0,
                i as i64,
                vplan.source_points,
                &ds,
            )
            .unwrap_or_else(|e| panic!("{e}"));
        det_bytes += u64::from(dplan.detections) * 44;
        plans.push(dplan);

        // The pairing rule, against the real timestamps: half-open
        // containment of the camera instant in the sweep's range.
        let (start, end, cam_t) = (starts[i].0, ends[i].0, cams[i].0);
        let inside = start <= cam_t && cam_t < end;
        assert!(
            inside,
            "sweep {i}: the camera instant {cam_t} is not inside [{start}, {end})"
        );
        paired += 1;
        pair_age_ns += cam_t - triggers[i].0;

        let dets = detection_rows(&dbatch);
        tk.reserve(dets.len());
        let tplan = tk.plan(dets, i as i64, triggers[i].0, v, RANGE_LIMIT_M);
        ambiguous_total += u64::from(tplan.ambiguous);
        let meta = TrackMeta {
            trigger_ns: triggers[i].0,
            sweep_seq: i as i64,
            source_voxels: dplan.source_voxels,
            source_points: vplan.source_points,
            cam: Some(CamRef {
                seq: i as u64,
                tov_ns: cam_t,
                width: calib.width,
                height: calib.height,
            }),
            pairing: Pairing::Paired,
        };
        let (tbatch, _, in_frame) = tk
            .build(&tplan, &meta, Some(&calib), &ts_schema)
            .unwrap_or_else(|e| panic!("{e}"));
        emitted_total += u64::from(tplan.emitted);
        in_frame_total += u64::from(in_frame);
        trk_bytes += u64::from(tplan.emitted) * TRACK_BYTES as u64;

        let rows = track_rows(&tbatch);
        for r in rows {
            let (n, age) = (track_observations(r), track_age_s(r));
            observations_total += u64::from(n);
            age_s_total += f64::from(age);
            age_s_max = age_s_max.max(age);
            // Sweeps the track has lived through, first seen to last seen,
            // against the sweeps it was seen in.
            let lived = (f64::from(age) / period_s).round() as i64 + 1;
            let gap = lived - i64::from(n);
            coasted_rows += u64::from(gap > 0);
            worst_gap = worst_gap.max(gap);
        }
        let answer = StateAnswer::of(rows);
        // The answer as the pipeline builds it: every track a record, the
        // nearest in the path flagged. The flagged record must BE the answer
        // this harness checks against the raw cloud below, on every sweep,
        // or the check is about a number the pipeline no longer reports.
        let smeta = StateMeta {
            trigger_ns: triggers[i].0,
            sweep_seq: i as i64,
            source_detections: dplan.detections,
            source_voxels: dplan.source_voxels,
            source_points: vplan.source_points,
            cam_seq: i as i64,
            pair_outcome: Pairing::Paired.name(),
            pair_age_ns: cam_t - triggers[i].0,
        };
        let (sbatch, _) = build_state_batch(rows, &answer, &smeta, None, &state_schema())
            .unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(
            read_answer(&sbatch),
            Some(answer),
            "sweep {i}: the flagged record is not the answer"
        );
        assert_eq!(
            object_rows(&sbatch).len(),
            rows.len(),
            "sweep {i}: a track was dropped"
        );
        ans_bytes += (object_rows(&sbatch).len() * OBJECT_BYTES) as u64;

        // ---- the independent check, and its two controls ------------------
        let raw = raw_nearest_structure(points, dplan.ground, 0.0);
        let ctl = raw_nearest_structure(points, dplan.ground, CONTROL_OFFSET_M);
        let bare = raw_nearest(points, dplan.ground, 0.0);
        if let (true, Some(r)) = (answer.has_object, raw) {
            agree.push(f64::from(answer.distance_m) - f64::from(r));
            agree_at.push((i, answer.distance_m, r));
            // The presence check: there must BE something at the reported
            // distance. A window of one voxel edge either side, because that
            // is the quantisation a centroid carries.
            // The reported object's own lateral extent, read off the track
            // the answer names.
            let (ylo, yhi) = rows
                .iter()
                .find(|r| r[6] as u32 == answer.object_id)
                .map_or((-CORRIDOR_HALF_WIDTH_M, CORRIDOR_HALF_WIDTH_M), |r| {
                    (r[10], r[13])
                });
            present.push(returns_in_object(
                points,
                dplan.ground,
                answer.distance_m,
                0.20,
                ylo,
                yhi,
            ));
            // The same window, the same distance, a band 6 m to the side. If
            // this were as full as the one above, the presence check would be
            // counting the scene rather than the object.
            present_ctl.push(returns_in_object(
                points,
                dplan.ground,
                answer.distance_m,
                0.20,
                ylo + CONTROL_OFFSET_M,
                yhi + CONTROL_OFFSET_M,
            ));
        }
        if let (true, Some(c)) = (answer.has_object, ctl) {
            control.push(f64::from(answer.distance_m) - f64::from(c));
        }
        if let (true, Some(b)) = (answer.has_object, bare) {
            bonnet.push(f64::from(answer.distance_m) - f64::from(b));
        }
        answers.push((i, answer, raw, ctl));
    }

    // ---- what the chain produced ------------------------------------------
    let n = N_SWEEPS as u64;
    let dets_total: u64 = plans.iter().map(|p| u64::from(p.detections)).sum();
    println!("\n---- the chain, per sweep, over {N_SWEEPS} sweeps of {DRIVE} ----");
    println!("detections  {:>8}", dets_total / n);
    println!(
        "tracks      {:>8}  (emitted, >= 2 observations)",
        emitted_total / n
    );
    println!(
        "bytes       raw {:>9}  detections {:>6}  tracks {:>6}  answer {:>6}  (records only, no provenance)",
        raw_bytes / n,
        det_bytes / n,
        trk_bytes / n,
        ans_bytes / n
    );
    // **The basis of every ratio on the line above.** It used to be
    // `source_voxels * 16 * 4` -- the voxel count scaled by the reduction's own
    // nominal 4x -- which reported 1,968,256 B/sweep against a true 1,947,714,
    // 1.05 % high, and made the end-to-end ratio optimistic by the same amount.
    // It was a number derived from the answer it was meant to check, which is
    // the class of mistake this whole harness exists to catch, so the fix gets
    // an assertion and not just a corrected line. `points.len()` is what the
    // `.bin` file contains; `vplan.source_points` is what the voxelizer counted
    // and what travels down the chain as `source_point_count`. They are
    // different routes to the same number and the ratio is only honest if they
    // agree.
    assert_eq!(
        raw_bytes,
        source_points_total * 16,
        "the byte chain's raw basis is not the raw returns"
    );
    println!(
        "gate        {:.3} m at dt = {:.1} ms",
        association_gate_m(0.10327, v, RANGE_LIMIT_M),
        103.27
    );
    println!(
        "ambiguous   {:.1}% of detections had more than one track inside the gate",
        ambiguous_total as f64 / dets_total as f64 * 100.0
    );
    let span_s = (triggers[N_SWEEPS - 1].0 - triggers[0].0) as f64 * 1e-9;
    println!(
        "track age   mean {:.3} s, max {:.4} s (sensor time, first seen to last seen; the drive spans {span_s:.4} s) | observations mean {:.1}",
        age_s_total / emitted_total as f64,
        age_s_max,
        observations_total as f64 / emitted_total as f64
    );
    println!(
        "            {coasted_rows} of {emitted_total} emitted rows ({:.2}%) coasted at least once, so a count of observations understated their age; the worst by {worst_gap} sweeps ({:.2} s)",
        coasted_rows as f64 / emitted_total as f64 * 100.0,
        worst_gap as f64 * period_s
    );
    // An age is a span of this drive's own triggers, so it cannot exceed the
    // drive. The f32 lane rounds to about a microsecond at 16 s.
    assert!(
        f64::from(age_s_max) <= span_s + 1e-5,
        "a track is older ({age_s_max} s) than the drive ({span_s} s)"
    );
    println!(
        "in frame    {:.1}% of emitted tracks projected inside the camera image",
        in_frame_total as f64 / emitted_total as f64 * 100.0
    );
    println!(
        "pairing     {paired}/{N_SWEEPS} camera instants inside their sweep, mean offset from the trigger {:.3} ms",
        pair_age_ns as f64 / paired as f64 / 1e6
    );
    assert_eq!(paired, N_SWEEPS);

    // ---- the verification --------------------------------------------------
    assert!(
        agree.len() > N_SWEEPS / 2,
        "only {} of {N_SWEEPS} sweeps had both an answer and a raw return to check it against",
        agree.len()
    );
    let (mut a, mut c) = (agree.clone(), control.clone());
    let (med, med_ctl) = (median(&mut a), median(&mut c));
    let within_half: usize = agree.iter().filter(|d| d.abs() <= 0.5).count();
    let pct = |v: &[f64], q: f64| -> f64 {
        let mut w = v.to_vec();
        w.sort_by(f64::total_cmp);
        if w.is_empty() {
            return f64::NAN;
        }
        w[((w.len() - 1) as f64 * q / 100.0).round() as usize]
    };
    let ctl_within_half: usize = control.iter().filter(|d| d.abs() <= 0.5).count();
    println!("\n---- the answer against the raw cloud ----");
    println!(
        "corridor      (answer - raw structure) over {} sweeps: p10 {:+.3}  median {med:+.3}  p90 {:+.3} m; within 0.5 m on {within_half} ({:.1}%)",
        agree.len(),
        pct(&agree, 10.0),
        pct(&agree, 90.0),
        within_half as f64 / agree.len() as f64 * 100.0
    );
    println!(
        "              a NEGATIVE difference means the answer is nearer than any return strictly inside the corridor: the answer selects a track whose BOX overlaps it, so a wide object counts before its returns do"
    );
    // Named, not summarised. The p10 above is compatible with the near face
    // being a metre off; the two extremes say how wrong `distance_m` can be
    // when the box overlaps the corridor somewhere other than at its near
    // face, which is the limitation `pipes_kitti::state`'s docs now carry.
    let worst = |cmp: fn(f32, f32) -> bool| -> Option<(usize, f32, f32)> {
        agree_at
            .iter()
            .copied()
            .reduce(|a, b| if cmp(b.1 - b.2, a.1 - a.2) { b } else { a })
    };
    if let Some((i, a, r)) = worst(|x, y| x < y) {
        println!(
            "WORST near    sweep {i}: answer {a:.2} m against the nearest corridor structure at {r:.3} m, {:+.3} m -- the answer is NEARER than anything strictly in the corridor, because the box it names overlaps the corridor somewhere other than at its near face",
            a - r
        );
    }
    if let Some((i, a, r)) = worst(|x, y| x > y) {
        println!(
            "WORST far     sweep {i}: answer {a:.2} m against the nearest corridor structure at {r:.3} m, {:+.3} m -- the corridor holds a structure the detector did not report as a track, or reported at a greater range",
            a - r
        );
    }
    println!(
        "CONTROL +{CONTROL_OFFSET_M} m median (answer - raw structure) = {med_ctl:+.3} m over {} sweeps; within 0.5 m on {ctl_within_half} ({:.1}%)",
        control.len(),
        ctl_within_half as f64 / control.len().max(1) as f64 * 100.0
    );
    let mut b = bonnet.clone();
    println!(
        "CONTROL any single return: median (answer - nearest RETURN) = {:+.3} m -- that gap is the recording vehicle's own bonnet at x ~ 2.50 m, one or two returns per sweep, which is below the detector's {MIN_CLUSTER_VOXELS}-voxel minimum and correctly not an obstacle",
        median(&mut b)
    );
    let mut pr: Vec<f64> = present.iter().map(|n| *n as f64).collect();
    let mut pc: Vec<f64> = present_ctl.iter().map(|n| *n as f64).collect();
    let (pr_med, pc_med) = (median(&mut pr), median(&mut pc));
    println!(
        "presence      median {pr_med:.0} raw non-ground returns within 0.20 m of the reported distance, in the reported object's own lateral band (min {}, and NEVER 0 -- no answer names a distance with nothing at it).
              This is NOT a corridor check and must not be read as one: the band is the TRACK's and can lie entirely outside the 1.82 m the car would drive through. It rules out an INVENTED distance, not a distance measured off the wrong part of a wide object",
        present.iter().copied().min().unwrap_or(0)
    );
    println!(
        "CONTROL +{CONTROL_OFFSET_M} m same x, band displaced sideways: median {pc_med:.0} returns"
    );

    // One sweep, in full, so the check is reproducible by hand.
    let (i, ans, raw, ctl) = answers
        .iter()
        .find(|(_, a, r, _)| a.has_object && r.is_some())
        .copied()
        .expect("no sweep had both an answer and a raw return");
    println!("\n---- sweep {i}, in full: the answer checked by hand ----");
    println!("answer            {}", ans.line());
    println!(
        "raw structure     {:.3} m  (>= {MIN_CLUSTER_VOXELS} non-ground returns within {STRUCTURE_M:.2} m of x, in the corridor)",
        raw.unwrap_or(f32::NAN)
    );
    println!(
        "difference        {:+.3} m (answer - raw)",
        f64::from(ans.distance_m) - f64::from(raw.unwrap_or(f32::NAN))
    );
    println!(
        "CONTROL +{CONTROL_OFFSET_M} m      {:.3} m  -- a different part of the scene entirely",
        ctl.unwrap_or(f32::NAN)
    );

    // The claims. The answer's distance is the smallest x of a voxel CENTROID
    // in a cluster of at least three voxels; a centroid sits inside its cell,
    // so the answer should be at or slightly beyond the nearest raw structure,
    // never far short of it.
    assert!(
        med.abs() <= 1.0,
        "the answer disagrees with the raw cloud by a median of {med:+.3} m"
    );
    // And there is really something there. A reported distance with nothing
    // at it is the failure mode a median cannot catch.
    // Never ZERO is the claim, not "many": the near face of a cluster is a
    // voxel that can hold a single return, so a reported distance legitimately
    // has as few as one return exactly at it. What it can never have is none,
    // which is what an invented distance would have.
    assert!(
        present.iter().all(|n| *n > 0),
        "an answer reported a distance with NO raw return at it: min {:?}",
        present.iter().min()
    );
    // And the count is about the object rather than about the scene.
    assert!(
        pr_med > pc_med * 2.0,
        "the presence window finds about as much 6 m to the side ({pc_med}) as at the object ({pr_med}), so it is counting the scene"
    );
    assert!(
        within_half as f64 / agree.len() as f64 > 0.5,
        "the answer is within half a metre of the raw nearest return on only {:.1}% of sweeps",
        within_half as f64 / agree.len() as f64 * 100.0
    );
    // The control has to FAIL, or the agreement above is not evidence.
    assert!(
        med_ctl.abs() > med.abs() * 2.0 + 0.5,
        "the displaced control agrees about as well ({med_ctl:+.3} m) as the real corridor ({med:+.3} m), so the agreement says nothing about the corridor"
    );
}
