//! `pipes bench-detector`: the camera detector alone, timed over the first
//! frames of a drive, one model on one backend per invocation.
//!
//! What a run's `camdet` stage measures, without the run: the PNGs are
//! decoded first, outside the timing, and each frame then costs the
//! letterbox, the network and the decode of its output, timed apart, then
//! NMS. The frames are run twice, and the second pass's detections are
//! compared with the first's byte for byte, so the line says whether the
//! backend gives the same answer twice on this machine -- which the default
//! does everywhere and the GPU is held to on one machine.

use std::path::Path;
use std::time::Instant;

use pipes_core::stats::percentile;
use pipes_kitti::cam0::{camera_dir, Cam0Driver, Camera};
use pipes_kitti::camdet::{nms, sha256_hex, Backend, Det, Model};
use pipes_kitti::pillars::{
    class_name, views_words, LidarDet, LidarModel, Pillarizer, Step, CLASSES, MODEL_NAME, PARAMS_M,
};
use pipes_kitti::velo::{
    build_velo_batch, read_sweep, velo_dir, velo_schema, velo_xyzr, VeloDriver,
};

use crate::cli::{BenchArgs, BenchLidarArgs};

type BoxError = Box<dyn std::error::Error + Send + Sync>;

/// One frame's timings, ns.
struct FrameTimes {
    preprocess: i64,
    infer: i64,
    nms: i64,
}

/// Runs the benchmark and prints its lines; the last is the one-line
/// summary a reader quotes.
pub fn main(a: BenchArgs) -> Result<(), BoxError> {
    let cam = match a.camera.as_str() {
        "left" => Camera::Left,
        "right" => Camera::Right,
        other => return Err(format!("--camera must be left or right, not {other}").into()),
    };
    let date = &a.drive[..10];
    let driver = Cam0Driver::open_camera(&a.kitti_root, date, &a.drive, cam)?;
    let n = a.frames.min(driver.len()).max(1);
    let spec = a.model.spec();
    let backend = a.backend.backend();
    let t = Instant::now();
    let cuda = a.cuda_search.options(a.tf32);
    let model = Model::load_with(Path::new(spec.file), spec, backend, a.threads, cuda)?;
    let load_ms = t.elapsed().as_secs_f64() * 1e3;
    println!(
        "bench-detector: {} ({} M parameters, input {}) on {}{}, sha256 {}, loaded in {load_ms:.0} ms; {n} frames of {} ({})",
        spec.name,
        spec.params_m,
        spec.input_words(),
        backend.name(),
        match backend {
            Backend::Ort => format!(" with {} threads", model.threads),
            Backend::Cuda => format!(
                " ({} algorithm search, TF32 {})",
                cuda.search.name(),
                if cuda.tf32 { "on" } else { "off" }
            ),
            Backend::Tract => String::new(),
        },
        model.sha256,
        a.drive,
        camera_dir(&a.kitti_root, date, &a.drive, cam).display()
    );
    // The frames, decoded up front: the disk is not what is timed.
    let mut frames = Vec::with_capacity(n);
    for k in 0..n {
        let d = driver.decode(k)?;
        frames.push((d.pixels, d.width, d.height));
    }
    // Two passes: the first warms the runtime on its first two frames
    // (which are timed, and reported apart), the second says whether the
    // answers repeat.
    let mut first: Vec<Vec<Det>> = Vec::with_capacity(n);
    let mut times: Vec<FrameTimes> = Vec::with_capacity(n);
    for (px, w, h) in &frames {
        let t0 = Instant::now();
        let (tensor, r) = model.preprocess(px, *w, *h)?;
        let t1 = Instant::now();
        let raw = model.infer(tensor, r, *w, *h)?;
        let t2 = Instant::now();
        let dets = nms(raw);
        let t3 = Instant::now();
        times.push(FrameTimes {
            preprocess: (t1 - t0).as_nanos() as i64,
            infer: (t2 - t1).as_nanos() as i64,
            nms: (t3 - t2).as_nanos() as i64,
        });
        first.push(dets);
    }
    let mut differ: Option<usize> = None;
    for (k, (px, w, h)) in frames.iter().enumerate() {
        let (tensor, r) = model.preprocess(px, *w, *h)?;
        let dets = nms(model.infer(tensor, r, *w, *h)?);
        if !same(&dets, &first[k]) {
            differ = Some(k);
            break;
        }
    }
    let ms = |v: Vec<i64>, p: f64| -> String {
        let mut v = v;
        percentile(&mut v, p)
            .map_or_else(|| "-".to_string(), |ns| format!("{:.1}", ns as f64 / 1e6))
    };
    // The first two frames are the runtime's warm-up on every backend (tract
    // allocates its buffers, ONNX Runtime its arenas, CUDA its kernels), so
    // the steady figures leave them out, and the warm-up is said.
    let warm: Vec<f64> = times
        .iter()
        .take(2)
        .map(|t| (t.preprocess + t.infer + t.nms) as f64 / 1e6)
        .collect();
    let steady: Vec<&FrameTimes> = times.iter().skip(2.min(times.len() - 1)).collect();
    let total: Vec<i64> = steady
        .iter()
        .map(|t| t.preprocess + t.infer + t.nms)
        .collect();
    let dets_total: usize = first.iter().map(Vec::len).sum();
    // The detections' own hash, so two invocations -- two processes, two
    // sessions, two algorithm choices -- can be compared by their lines,
    // which the two passes inside one process cannot show.
    let mut bytes = Vec::with_capacity(dets_total * 24);
    for d in first.iter().flatten() {
        for v in [d.x0, d.y0, d.x1, d.y1, d.score] {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        bytes.extend_from_slice(&d.class_id.to_le_bytes());
    }
    let digest = sha256_hex(&bytes);
    println!(
        "frames {n}: warm-up {}; then per frame ms: total p50 {} p90 {} max {} (letterbox p50 {}, network+decode p50 {}, NMS p50 {}); detections {} ({:.1} per frame), sha256 {}; repeatable: {}",
        warm.iter().map(|w| format!("{w:.0}")).collect::<Vec<_>>().join(" + "),
        ms(total.clone(), 50.0),
        ms(total.clone(), 90.0),
        ms(total, 100.0),
        ms(steady.iter().map(|t| t.preprocess).collect(), 50.0),
        ms(steady.iter().map(|t| t.infer).collect(), 50.0),
        ms(steady.iter().map(|t| t.nms).collect(), 50.0),
        dets_total,
        dets_total as f64 / n as f64,
        &digest[..16],
        match differ {
            None => "yes, the same bytes on both passes".to_string(),
            Some(k) => format!("NO, frame {k} differs between the passes"),
        }
    );
    Ok(())
}

/// Runs the lidar benchmark and prints its lines, on [`main`]'s terms: the
/// sweeps are read first, outside the timing, and each then costs the
/// pillars (binning and the ten features, on the CPU) and the network with
/// its decode and NMS, timed apart; two passes, compared byte for byte.
pub fn lidar(a: BenchLidarArgs) -> Result<(), BoxError> {
    let date = &a.drive[..10];
    let driver = VeloDriver::open(&a.kitti_root, date, &a.drive)?;
    let n = a.sweeps.min(driver.len()).max(1);
    let backend = a.backend.backend();
    let cuda = a.cuda_search.options(a.tf32);
    let t = Instant::now();
    let model = LidarModel::load(backend, a.threads, cuda)?;
    let load_ms = t.elapsed().as_secs_f64() * 1e3;
    let views = a.views.views(backend);
    println!(
        "bench-lidar: {MODEL_NAME} ({file}, {PARAMS_M} M parameters) on {}{}, sha256 {}, loaded in {load_ms:.0} ms; looking {looks}; {n} sweeps of {} ({})",
        backend.name(),
        match backend {
            Backend::Ort => format!(" with {} threads", model.threads),
            Backend::Cuda => format!(
                " ({} algorithm search, TF32 {})",
                cuda.search.name(),
                if cuda.tf32 { "on" } else { "off" }
            ),
            Backend::Tract => String::new(),
        },
        model.sha256,
        a.drive,
        velo_dir(&a.kitti_root, date, &a.drive).display(),
        file = model.file,
        looks = views_words(views),
    );
    // The sweeps, read up front: the disk is not what is timed.
    let schema = velo_schema();
    let mut sweeps = Vec::with_capacity(n);
    for k in 0..n {
        let path = driver
            .sweep_path(k)
            .ok_or_else(|| format!("sweep {k} of {} has no file", a.drive))?;
        let (batch, _) = build_velo_batch(read_sweep(&path)?, 0, &schema)?;
        sweeps.push(batch);
    }
    let points = |b: &arrow::array::RecordBatch| -> Vec<[f32; 4]> {
        velo_xyzr(b).unwrap_or(&[]).as_chunks::<4>().0.to_vec()
    };
    let clouds: Vec<Vec<[f32; 4]>> = sweeps.iter().map(points).collect();
    let mut pz = Pillarizer::new();
    let mut scratch = Vec::new();
    let mut first: Vec<Vec<LidarDet>> = Vec::with_capacity(n);
    // `(pillars, network + decode + NMS)` ns per sweep, the second in its
    // three parts, and the pillar count.
    let mut times: Vec<(i64, i64)> = Vec::with_capacity(n);
    let mut parts: Vec<[i64; 3]> = Vec::with_capacity(n);
    let mut pillars_total = 0usize;
    let started = Instant::now();
    for (k, pts) in clouds.iter().enumerate() {
        pz.reserve(pts.len());
        // Paced, when asked: each sweep at its own instant, the processor
        // and the GPU idle in between, as a run leaves them.
        if a.period_ms > 0 {
            let due = started + std::time::Duration::from_millis(a.period_ms * k as u64);
            std::thread::sleep(due.saturating_duration_since(Instant::now()));
        }
        let t0 = Instant::now();
        let mut last = t0;
        // Pillars, network, decode, the views' added: ns.
        let mut spent = [0i64; 3];
        let (plan, dets) = model.infer_views(&mut pz, pts, views, &mut scratch, &mut |step| {
            let t = Instant::now();
            let k = match step {
                Step::Binned => 0,
                Step::Ran => 1,
                Step::Decoded => 2,
            };
            spent[k] += (t - last).as_nanos() as i64;
            last = t;
        });
        let dets = dets?;
        let t2 = Instant::now();
        times.push((spent[0], (t2 - t0).as_nanos() as i64 - spent[0]));
        parts.push([spent[1], spent[2], (t2 - last).as_nanos() as i64]);
        pillars_total += plan.pillars as usize;
        first.push(dets);
    }
    let mut differ: Option<usize> = None;
    for (k, pts) in clouds.iter().enumerate() {
        let dets = model
            .infer_views(&mut pz, pts, views, &mut scratch, &mut |_| {})
            .1?;
        if !same_lidar(&dets, &first[k]) {
            differ = Some(k);
            break;
        }
    }
    let ms = |v: Vec<i64>, p: f64| -> String {
        let mut v = v;
        percentile(&mut v, p)
            .map_or_else(|| "-".to_string(), |ns| format!("{:.1}", ns as f64 / 1e6))
    };
    let warm: Vec<f64> = times
        .iter()
        .take(2)
        .map(|t| (t.0 + t.1) as f64 / 1e6)
        .collect();
    let steady: Vec<&(i64, i64)> = times.iter().skip(2.min(times.len() - 1)).collect();
    let steady_parts: Vec<&[i64; 3]> = parts.iter().skip(2.min(parts.len() - 1)).collect();
    let total: Vec<i64> = steady.iter().map(|t| t.0 + t.1).collect();
    let dets_total: usize = first.iter().map(Vec::len).sum();
    let mut by_class = [0usize; 3];
    let mut bytes = Vec::with_capacity(dets_total * 36);
    for d in first.iter().flatten() {
        if let Some(c) = by_class.get_mut(d.class_id as usize) {
            *c += 1;
        }
        for v in d.lanes() {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
    }
    let digest = sha256_hex(&bytes);
    println!(
        "sweeps {n}{}: warm-up {}; then per sweep ms: total p50 {} p90 {} max {} (pillars p50 {}, network p50 {}, decode p50 {}, NMS p50 {}); {:.0} pillars a sweep; detections {} ({:.1} per sweep: {}), sha256 {}; repeatable: {}",
        if a.period_ms > 0 {
            format!(", one every {} ms", a.period_ms)
        } else {
            ", back to back".to_string()
        },
        warm.iter().map(|w| format!("{w:.0}")).collect::<Vec<_>>().join(" + "),
        ms(total.clone(), 50.0),
        ms(total.clone(), 90.0),
        ms(total, 100.0),
        ms(steady.iter().map(|t| t.0).collect(), 50.0),
        ms(steady_parts.iter().map(|p| p[0]).collect(), 50.0),
        ms(steady_parts.iter().map(|p| p[1]).collect(), 50.0),
        ms(steady_parts.iter().map(|p| p[2]).collect(), 50.0),
        pillars_total as f64 / n as f64,
        dets_total,
        dets_total as f64 / n as f64,
        CLASSES
            .iter()
            .enumerate()
            .map(|(k, _)| format!("{} {}", by_class[k], class_name(k as u32)))
            .collect::<Vec<_>>()
            .join(", "),
        &digest[..16],
        match differ {
            None => "yes, the same bytes on both passes".to_string(),
            Some(k) => format!("NO, sweep {k} differs between the passes"),
        }
    );
    // One sweep's boxes, when asked for: what the numbers above are made of.
    if let Some(k) = a.show {
        match first.get(k) {
            Some(dets) => {
                println!("sweep {k}: {} boxes", dets.len());
                for d in dets {
                    println!(
                        "  {:<10} {:.4}  centre ({:7.3}, {:7.3}, {:6.3}) m  size {:.2} x {:.2} x {:.2} m  heading {:6.1} deg  near face {:6.2} m",
                        class_name(d.class_id),
                        d.score,
                        d.x,
                        d.y,
                        d.z,
                        d.dx,
                        d.dy,
                        d.dz,
                        // The network's heading is not wrapped; a reader's is.
                        d.yaw.to_degrees().rem_euclid(360.0),
                        d.near_face_m()
                    );
                }
            }
            None => println!("sweep {k}: not among the {n} sweeps timed (--sweeps)"),
        }
    }
    Ok(())
}

/// Whether two passes' boxes of one sweep are the same bytes.
fn same_lidar(a: &[LidarDet], b: &[LidarDet]) -> bool {
    a.len() == b.len()
        && a.iter().zip(b).all(|(x, y)| {
            x.lanes()
                .iter()
                .zip(y.lanes())
                .all(|(p, q)| p.to_bits() == q.to_bits())
        })
}

/// Whether two passes' detections of one frame are the same bytes.
fn same(a: &[Det], b: &[Det]) -> bool {
    a.len() == b.len()
        && a.iter().zip(b).all(|(x, y)| {
            x.x0.to_bits() == y.x0.to_bits()
                && x.y0.to_bits() == y.y0.to_bits()
                && x.x1.to_bits() == y.x1.to_bits()
                && x.y1.to_bits() == y.y1.to_bits()
                && x.class_id == y.class_id
                && x.score.to_bits() == y.score.to_bits()
        })
}
