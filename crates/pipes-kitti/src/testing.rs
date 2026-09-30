//! A synthetic KITTI drive on disk, so the pipeline can be tested without the
//! 646 MB dataset.
//!
//! Compiled only under `cfg(test)` or the `testing` feature, so a normal build
//! of this crate contains none of it. Frames are 8x4 RGB8 with content derived
//! from the frame index, which makes every byte of a decoded frame predictable
//! and lets a test assert on pixels rather than on "it decoded something".

use std::path::{Path, PathBuf};

use tempfile::TempDir;

/// Capture-date directory of the fixture drive, as KITTI spells it.
pub const FIXTURE_DATE: &str = "2011_09_26";
/// Drive directory of the fixture, matching the real drive the sprint used.
pub const FIXTURE_DRIVE: &str = "2011_09_26_drive_0005_sync";
/// Frame width in pixels.
pub const FIXTURE_W: u32 = 8;
/// Frame height in pixels.
pub const FIXTURE_H: u32 = 4;
/// 2011-09-26 00:00:00 UTC in ns since the Unix epoch.
pub const FIXTURE_BASE_NS: i64 = 1_316_995_200_000_000_000;
/// 100 ms: frame i is exactly FIXTURE_BASE_NS + i * 100_000_000.
pub const FIXTURE_PERIOD_NS: i64 = 100_000_000;

/// Frame i, pixel (x, y) = (R, G, B) = (i as u8, x as u8, y as u8), row-major,
/// interleaved. Length is always `FIXTURE_W * FIXTURE_H * 3` = 96.
pub fn fixture_pixels(i: usize) -> Vec<u8> {
    let mut px = Vec::with_capacity((FIXTURE_W * FIXTURE_H * 3) as usize);
    for y in 0..FIXTURE_H {
        for x in 0..FIXTURE_W {
            px.push(i as u8);
            px.push(x as u8);
            px.push(y as u8);
        }
    }
    px
}

/// `FIXTURE_BASE_NS + i as i64 * period_ns`.
pub fn fixture_timestamp_ns(i: usize, period_ns: i64) -> i64 {
    FIXTURE_BASE_NS + i as i64 * period_ns
}

/// One `timestamps.txt` line: `2011-09-26 HH:MM:SS.fffffffff` (UTC, 9 fraction
/// digits). Hour and minute are literal because the fixture never leaves the
/// first minute; [`write_fixture`] asserts that in debug builds.
pub fn fixture_timestamp_line(i: usize, period_ns: i64) -> String {
    fixture_timestamp_line_ns(i as i64 * period_ns)
}

/// [`fixture_timestamp_line`] for an offset that is not a whole number of
/// periods, which the lidar needs: a sweep's start, trigger and end are three
/// distinct instants inside one period and only the first of them lands on a
/// period boundary.
pub fn fixture_timestamp_line_ns(total_ns: i64) -> String {
    format!(
        "2011-09-26 00:00:{:02}.{:09}",
        total_ns / 1_000_000_000,
        total_ns % 1_000_000_000
    )
}

/// Writes `<root>/<date>/<drive>/image_02/{data/{i:010}.png, timestamps.txt}`.
///
/// PNGs are 8x4 RGB8; `timestamps.txt` is LF-terminated, one line per frame,
/// with no trailing blank line. Creates every parent directory.
pub fn write_fixture(root: &Path, n_frames: usize, period_ns: i64) -> std::io::Result<()> {
    write_fixture_at(root, FIXTURE_DATE, FIXTURE_DRIVE, n_frames, period_ns)
}

/// [`write_fixture`] under a chosen `<date>/<drive>`, so a test can build a
/// data root holding more than the one drive — which is what a listing has to
/// be tested against, and what a single hard-coded pair cannot produce.
pub fn write_fixture_at(
    root: &Path,
    date: &str,
    drive: &str,
    n_frames: usize,
    period_ns: i64,
) -> std::io::Result<()> {
    debug_assert!(
        n_frames as i64 * period_ns < 60_000_000_000,
        "fixture_timestamp_line hard-codes 00:00, so the drive must stay inside the first minute"
    );
    let image_dir = root.join(date).join(drive).join("image_02");
    let data_dir = image_dir.join("data");
    std::fs::create_dir_all(&data_dir)?;
    for i in 0..n_frames {
        let img = image::RgbImage::from_raw(FIXTURE_W, FIXTURE_H, fixture_pixels(i)).ok_or_else(
            || {
                std::io::Error::other(format!(
                    "fixture_pixels({i}) is not {FIXTURE_W}x{FIXTURE_H}"
                ))
            },
        )?;
        img.save(data_dir.join(format!("{i:010}.png")))
            .map_err(std::io::Error::other)?;
    }
    let mut text = String::new();
    for i in 0..n_frames {
        text.push_str(&fixture_timestamp_line(i, period_ns));
        text.push('\n');
    }
    std::fs::write(image_dir.join("timestamps.txt"), text)
}

/// A camera whose source lacks the frames of `absent`, laid out the way
/// KITTI lays out drive 0009's lidar gap (`crate::layout`): `n_frames` lines
/// in `timestamps.txt`, line `i + 1` for frame `i` at that frame's own
/// instant, BLANK for each frame of `absent`, whose PNG is not written -- so
/// the absent frames leave a hole in time as well as in the names. An absent
/// frame at the end is a trailing blank line: declared, and absent. Writes the
/// whole drive with [`write_fixture_at`], then takes the absent frames' PNGs
/// out and blanks their lines.
pub fn write_fixture_gapped_at(
    root: &Path,
    date: &str,
    drive: &str,
    n_frames: usize,
    absent: &[usize],
    period_ns: i64,
) -> std::io::Result<()> {
    write_fixture_at(root, date, drive, n_frames, period_ns)?;
    let image_dir = root.join(date).join(drive).join("image_02");
    let mut text = String::new();
    for i in 0..n_frames {
        if absent.contains(&i) {
            std::fs::remove_file(image_dir.join("data").join(format!("{i:010}.png")))?;
        } else {
            text.push_str(&fixture_timestamp_line(i, period_ns));
        }
        text.push('\n');
    }
    std::fs::write(image_dir.join("timestamps.txt"), text)
}

/// Points in fixture sweep `i`: `FIXTURE_SWEEP_POINTS_BASE + i`.
///
/// Deliberately a different count per sweep, because the real thing is: KITTI
/// `.bin` files are all different lengths (drive_0005: 154 files, 154 distinct
/// sizes), so a fixture with one fixed count cannot catch a consumer that
/// reads the `point_count` column instead of the payload, or the reverse.
pub const FIXTURE_SWEEP_POINTS_BASE: usize = 4;

/// Bytes per point in the fixture, matching `velo::POINT_BYTES`.
const FIXTURE_POINT_BYTES: usize = 16;

/// Points in fixture sweep `i`.
pub fn fixture_sweep_points(i: usize) -> usize {
    FIXTURE_SWEEP_POINTS_BASE + i
}

/// How many points [`write_velo_fixture_at`] puts in sweep `i`.
///
/// A function rather than a count because the two things a lidar fixture is
/// needed for want opposite sizes. Most tests want sweeps small enough to
/// state in closed form. A test about SHUTDOWN ORDER wants the opposite: a
/// consumer slow enough that it is certainly still working when its producer
/// stops, or the race it exists to catch never happens and the test passes
/// against the bug.
pub type SweepSize = fn(usize) -> usize;

/// Sweep `i` as the little-endian `xyzr` f32 bytes KITTI stores.
///
/// Point `j` is `(x, y, z, r) = (j, -j, i, 0.5)`: the extent over one sweep is
/// then exactly `x in [0, n-1]`, `y in [-(n-1), 0]`, `z = i`, which is a
/// bounding box a test can state in closed form instead of copying out of the
/// implementation it is checking.
pub fn fixture_sweep_bytes(i: usize) -> Vec<u8> {
    fixture_sweep_bytes_of(i, fixture_sweep_points(i))
}

/// [`fixture_sweep_bytes`] with the point count chosen by the caller, so a
/// test that needs a *slow* consumer can have one without changing the
/// geometry every other test states in closed form.
pub fn fixture_sweep_bytes_of(i: usize, n: usize) -> Vec<u8> {
    let mut v = Vec::with_capacity(n * FIXTURE_POINT_BYTES);
    for j in 0..n {
        for x in [j as f32, -(j as f32), i as f32, 0.5] {
            v.extend_from_slice(&x.to_le_bytes());
        }
    }
    v
}

/// Writes `<root>/<date>/calib_velo_to_cam.txt` and `calib_cam_to_cam.txt`.
///
/// The **real** rotation, translation, rectification and projection of KITTI's
/// `2011_09_26` set, byte for byte — including `calib_time`'s colons, which a
/// `split(':')` reader gets four pieces from. Only `S_rect_02` is the
/// fixture's own, because the fixture's frames are 8x4 and
/// `Calib::check_image_size` cross-checks it against the PNG header rather
/// than assuming a resolution.
///
/// Real numbers rather than an identity on purpose: a fixture calibration that
/// was the identity would let a projection bug through, and the whole point of
/// `calib.rs`'s validations is that a WRONG calibration still produces
/// plausible-looking output.
pub fn write_calib_fixture_at(root: &Path, date: &str, w: u32, h: u32) -> std::io::Result<()> {
    let dir = root.join(date);
    std::fs::create_dir_all(&dir)?;
    std::fs::write(
        dir.join("calib_velo_to_cam.txt"),
        "calib_time: 15-Mar-2012 11:37:16\n\
         R: 7.533745e-03 -9.999714e-01 -6.166020e-04 1.480249e-02 7.280733e-04 -9.998902e-01 \
         9.998621e-01 7.523790e-03 1.480755e-02\n\
         T: -4.069766e-03 -7.631618e-02 -2.717806e-01\n",
    )?;
    std::fs::write(
        dir.join("calib_cam_to_cam.txt"),
        format!(
            "calib_time: 09-Jan-2012 13:57:47\n\
             R_rect_00: 9.999239e-01 9.837760e-03 -7.445048e-03 -9.869795e-03 9.999421e-01 \
             -4.278459e-03 7.402527e-03 4.351614e-03 9.999631e-01\n\
             S_rect_02: {w:e} {h:e}\n\
             P_rect_02: 7.215377e+02 0.000000e+00 6.095593e+02 4.485728e+01 0.000000e+00 \
             7.215377e+02 1.728540e+02 2.163791e-01 0.000000e+00 0.000000e+00 1.000000e+00 \
             2.745884e-03\n",
            w = f64::from(w),
            h = f64::from(h)
        ),
    )?;
    Ok(())
}

/// Writes `<root>/<date>/<drive>/velodyne_points/{data/{i:010}.bin,
/// timestamps_start.txt, timestamps.txt, timestamps_end.txt}`.
///
/// Sweep `i` spans a whole period -- `start = i * period`, `end = (i+1) *
/// period` -- with the trigger at its MIDPOINT, which is the shape of a
/// continuously rotating head. The three files are three independent
/// measurements in KITTI and they are three here too: a fixture that derived
/// the trigger from the range would make a pipeline that did the same look
/// correct.
pub fn write_velo_fixture_at(
    root: &Path,
    date: &str,
    drive: &str,
    n_sweeps: usize,
    period_ns: i64,
) -> std::io::Result<()> {
    write_velo_fixture_sized_at(root, date, drive, n_sweeps, period_ns, fixture_sweep_points)
}

/// [`write_velo_fixture_at`] with the sweep size chosen by the caller; see
/// [`SweepSize`] for why a fixture needs that knob at all.
pub fn write_velo_fixture_sized_at(
    root: &Path,
    date: &str,
    drive: &str,
    n_sweeps: usize,
    period_ns: i64,
    points: SweepSize,
) -> std::io::Result<()> {
    let frames: Vec<usize> = (0..n_sweeps).collect();
    write_velo_frames_at(root, date, drive, &frames, n_sweeps, period_ns, points)
}

/// A lidar whose source lacks the sweeps of `absent`, laid out the way KITTI
/// lays out drive 0009: `n_frames` lines in each of the three timestamp
/// files, line `i + 1` for frame `i`, BLANK in all three for each frame of
/// `absent`, and a `.bin` named by its frame for every frame NOT in `absent`.
/// An absent frame at the end is a trailing blank line: declared, and absent.
///
/// Every sweep keeps its frame's place on the clock (`start = frame *
/// period`), so the frames that are absent leave a real hole in time, as the
/// sensor fault did.
pub fn write_velo_fixture_gapped_at(
    root: &Path,
    date: &str,
    drive: &str,
    n_frames: usize,
    absent: &[usize],
    period_ns: i64,
) -> std::io::Result<()> {
    let frames: Vec<usize> = (0..n_frames).filter(|f| !absent.contains(f)).collect();
    write_velo_frames_at(
        root,
        date,
        drive,
        &frames,
        n_frames,
        period_ns,
        fixture_sweep_points,
    )
}

/// Writes a sweep file for each frame of `frames` (ascending) and `n_lines`
/// lines to each of the three timestamp files: frame `f`'s at line `f + 1`,
/// blank for a frame not in `frames`. Sweep `f` spans `f * period .. (f + 1)
/// * period` with its trigger at the midpoint.
fn write_velo_frames_at(
    root: &Path,
    date: &str,
    drive: &str,
    frames: &[usize],
    n_lines: usize,
    period_ns: i64,
    points: SweepSize,
) -> std::io::Result<()> {
    debug_assert!(
        frames
            .last()
            .is_none_or(|&f| (f as i64 + 2) * period_ns < 60_000_000_000),
        "fixture_timestamp_line_ns hard-codes 00:00, and the last sweep ENDS one period past its start"
    );
    let velo_dir = root.join(date).join(drive).join("velodyne_points");
    let data_dir = velo_dir.join("data");
    std::fs::create_dir_all(&data_dir)?;
    for &i in frames {
        std::fs::write(
            data_dir.join(format!("{i:010}.bin")),
            fixture_sweep_bytes_of(i, points(i)),
        )?;
    }
    let lines = |f: &dyn Fn(i64) -> i64| -> String {
        let mut text = String::new();
        for i in 0..n_lines {
            if frames.contains(&i) {
                text.push_str(&fixture_timestamp_line_ns(f(i as i64 * period_ns)));
            }
            text.push('\n');
        }
        text
    };
    std::fs::write(velo_dir.join("timestamps_start.txt"), lines(&|t| t))?;
    std::fs::write(
        velo_dir.join("timestamps.txt"),
        lines(&|t| t + period_ns / 2),
    )?;
    std::fs::write(
        velo_dir.join("timestamps_end.txt"),
        lines(&|t| t + period_ns),
    )?;
    Ok(())
}

/// A fixture drive in a [`TempDir`] that deletes itself on drop.
pub struct FixtureDrive {
    tmp: TempDir,
    /// Frames written.
    pub n_frames: usize,
    /// Sweeps written; `0` when the drive carries no `velodyne_points/` at
    /// all, the shape of an `image_02`-only download.
    pub n_sweeps: usize,
    /// Nominal gap between consecutive frame timestamps, in ns.
    pub period_ns: i64,
}

impl FixtureDrive {
    /// `n_frames` frames at [`FIXTURE_PERIOD_NS`].
    pub fn new(n_frames: usize) -> std::io::Result<FixtureDrive> {
        FixtureDrive::with_period(n_frames, FIXTURE_PERIOD_NS)
    }

    /// `n_frames` frames spaced `period_ns` apart, and no lidar.
    pub fn with_period(n_frames: usize, period_ns: i64) -> std::io::Result<FixtureDrive> {
        let tmp = TempDir::new()?;
        write_fixture(tmp.path(), n_frames, period_ns)?;
        Ok(FixtureDrive {
            tmp,
            n_frames,
            n_sweeps: 0,
            period_ns,
        })
    }

    /// The same drive with `velodyne_points/` beside `image_02/`: the second
    /// producer, on the same period as the camera.
    ///
    /// Camera-only stays the DEFAULT constructor, deliberately. Many KITTI
    /// downloads are `image_02` alone, so a fixture that always had a lidar
    /// would stop that common case from being the one that is tested.
    pub fn with_velodyne(
        n_frames: usize,
        n_sweeps: usize,
        period_ns: i64,
    ) -> std::io::Result<FixtureDrive> {
        FixtureDrive::with_velodyne_sized(n_frames, n_sweeps, period_ns, fixture_sweep_points)
    }

    /// [`FixtureDrive::with_velodyne`] with the sweep size chosen by the
    /// caller.
    ///
    /// For one job only, and it is a real one: a test about SHUTDOWN ORDER
    /// needs the consumer of the sweeps to be certainly still working when its
    /// producer stops. With the default four-to-eleven-point sweeps every
    /// consumer in this workspace drains faster than the driver can read, the
    /// race never happens, and a test written against it passes just as
    /// happily against the bug. Measured: it did.
    pub fn with_velodyne_sized(
        n_frames: usize,
        n_sweeps: usize,
        period_ns: i64,
        points: SweepSize,
    ) -> std::io::Result<FixtureDrive> {
        let mut fx = FixtureDrive::with_period(n_frames, period_ns)?;
        write_velo_fixture_sized_at(
            fx.root(),
            FIXTURE_DATE,
            FIXTURE_DRIVE,
            n_sweeps,
            period_ns,
            points,
        )?;
        fx.n_sweeps = n_sweeps;
        Ok(fx)
    }

    /// `n_frames` camera frames and a lidar whose source has no sweep for the
    /// frames in `absent`: drive 0009's shape, in miniature. See
    /// [`write_velo_fixture_gapped_at`].
    pub fn with_velodyne_gapped(
        n_frames: usize,
        absent: &[usize],
        period_ns: i64,
    ) -> std::io::Result<FixtureDrive> {
        let mut fx = FixtureDrive::with_period(n_frames, period_ns)?;
        write_velo_fixture_gapped_at(
            fx.root(),
            FIXTURE_DATE,
            FIXTURE_DRIVE,
            n_frames,
            absent,
            period_ns,
        )?;
        fx.n_sweeps = (0..n_frames).filter(|f| !absent.contains(f)).count();
        Ok(fx)
    }

    /// `n_frames` frame numbers with a gap in either sensor's source, or in
    /// both: no PNG for the frames of `camera_absent`, no sweep for those of
    /// `lidar_absent`, each stream laid out as KITTI lays out a gap. With both
    /// empty it is [`FixtureDrive::with_velodyne`] with a sweep per frame.
    pub fn with_gaps(
        n_frames: usize,
        camera_absent: &[usize],
        lidar_absent: &[usize],
        period_ns: i64,
    ) -> std::io::Result<FixtureDrive> {
        let mut fx = FixtureDrive::with_velodyne_gapped(n_frames, lidar_absent, period_ns)?;
        write_fixture_gapped_at(
            fx.root(),
            FIXTURE_DATE,
            FIXTURE_DRIVE,
            n_frames,
            camera_absent,
            period_ns,
        )?;
        fx.n_frames = n_frames - camera_absent.len();
        Ok(fx)
    }

    /// Writes this drive's date-level calibration, sized to the fixture's own
    /// frames. `--track on` needs it and fails loudly without it, which is a
    /// case a test asserts rather than a gap.
    pub fn write_calib(&self) -> std::io::Result<()> {
        write_calib_fixture_at(self.root(), FIXTURE_DATE, FIXTURE_W, FIXTURE_H)
    }

    /// `<root>/<date>/<drive>/velodyne_points`, whether or not it exists.
    pub fn velo_dir(&self) -> PathBuf {
        self.root()
            .join(FIXTURE_DATE)
            .join(FIXTURE_DRIVE)
            .join("velodyne_points")
    }

    /// The `--kitti-root` value; absolute, because the binary under test runs
    /// with its CWD somewhere else entirely.
    pub fn root(&self) -> &Path {
        self.tmp.path()
    }

    /// `<root>/<date>/<drive>/image_02`.
    pub fn image_dir(&self) -> PathBuf {
        self.root()
            .join(FIXTURE_DATE)
            .join(FIXTURE_DRIVE)
            .join("image_02")
    }

    /// `<image_dir>/data/{i:010}.png`, whether or not it exists.
    pub fn frame_path(&self, i: usize) -> PathBuf {
        self.image_dir().join("data").join(format!("{i:010}.png"))
    }

    /// Renames `data/<from>` to `data/<to>`; reversible by swapping them.
    ///
    /// The PNG *count* does not change, which is the entire point: a drive
    /// whose frames are named anything at all is the shape of a partial
    /// download or an interrupted unzip, and it is exactly what a count-based
    /// check cannot see.
    pub fn rename_in_data(&self, from: &str, to: &str) -> std::io::Result<()> {
        let data = self.image_dir().join("data");
        std::fs::rename(data.join(from), data.join(to))
    }

    /// Timestamp of frame `i` on this drive's period.
    pub fn timestamp_ns(&self, i: usize) -> i64 {
        fixture_timestamp_ns(i, self.period_ns)
    }
}
