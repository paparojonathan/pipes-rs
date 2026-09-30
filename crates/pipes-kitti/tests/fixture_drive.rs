//! The synthetic drive is a real KITTI drive as far as this crate is
//! concerned: same layout, same timestamps format, same PNGs. If it were not,
//! every test built on it would prove nothing, so these tests check the
//! fixture itself against the production readers before anything else uses it.
//!
//! This is its own crate root, so it inherits none of the library's
//! `#![deny(clippy::unwrap_used)]`.

use std::time::Duration;

use pipes_core::clock::{ClockModel, HostTime, SensorTime, Tov};
use pipes_kitti::cam0::{
    decode_png, Cam0Driver, DriverError, DriverEvent, MissingLevel, RunTotals,
};
use pipes_kitti::frame::{cam0_dims, cam0_schema, cam0_storage_id};
use pipes_kitti::layout::{LayoutError, ABSENT_IN_SOURCE};
use pipes_kitti::testing::{
    fixture_pixels, fixture_timestamp_ns, write_fixture_gapped_at, FixtureDrive, FIXTURE_DATE,
    FIXTURE_DRIVE, FIXTURE_PERIOD_NS,
};
use pipes_kitti::timestamps::parse_timestamps;

#[test]
fn fixture_layout_matches_kitti() {
    let fx = FixtureDrive::new(16).unwrap();
    assert!(fx.image_dir().join("timestamps.txt").is_file());
    assert!(fx.frame_path(0).ends_with("data/0000000000.png"));
    assert!(fx.frame_path(15).ends_with("data/0000000015.png"));
    assert!(fx.frame_path(0).is_file());
    assert!(fx.frame_path(15).is_file());
    let pngs = std::fs::read_dir(fx.image_dir().join("data"))
        .unwrap()
        .filter_map(Result::ok)
        .filter(|e| e.path().extension().is_some_and(|x| x == "png"))
        .count();
    assert_eq!(pngs, 16);
    // A frame that was never written must not exist, or "is_file" above would
    // be measuring the directory walk rather than the writer.
    assert!(!fx.frame_path(16).is_file());
}

#[test]
fn fixture_timestamps_parse_to_exact_ns() {
    let fx = FixtureDrive::new(16).unwrap();
    let v = parse_timestamps(&fx.image_dir().join("timestamps.txt")).unwrap();
    assert_eq!(v.len(), 16);
    assert_eq!(v[0], SensorTime(1_316_995_200_000_000_000));
    assert_eq!(v[1], SensorTime(1_316_995_200_100_000_000));
    assert_eq!(v[15], SensorTime(1_316_995_201_500_000_000));
    for i in 0..15 {
        assert_eq!(v[i + 1] - v[i], 100_000_000, "gap after frame {i}");
    }
}

#[test]
fn fixture_png_round_trips_through_decode_png() {
    let fx = FixtureDrive::new(16).unwrap();
    let d = decode_png(&fx.frame_path(3)).unwrap();
    assert_eq!((d.width, d.height), (8, 4));
    assert_eq!(d.pixels.len(), 96);
    assert_eq!(d.pixels, fixture_pixels(3));
    assert!(d.decode_ns >= 0);
    // Frame 3 and frame 4 must differ, or "pixels == fixture_pixels(3)" would
    // hold for a writer that ignored the frame index.
    assert_ne!(d.pixels, fixture_pixels(4));
    assert_eq!(&d.pixels[0..6], &[3, 0, 0, 3, 1, 0]);
    assert_eq!(&d.pixels[93..96], &[3, 7, 3]);
}

#[test]
fn driver_open_counts_frames() {
    let fx = FixtureDrive::new(16).unwrap();
    let d = Cam0Driver::open(fx.root(), FIXTURE_DATE, FIXTURE_DRIVE).unwrap();
    assert_eq!(d.len(), 16);
    assert!(!d.is_empty());
    assert_eq!(d.timestamps[0].0, 1_316_995_200_000_000_000);
}

/// A PNG deleted with its timestamp line left behind. KITTI blanks the line
/// of a frame it has no file for, so a line with a time and no PNG is not a
/// gap in the source -- that is a blank line and no file, which the driver
/// replays -- and `open` refuses it, naming the line as an editor shows it
/// and the file it needs.
#[test]
fn driver_open_rejects_a_deleted_frame() {
    let fx = FixtureDrive::new(16).unwrap();
    std::fs::remove_file(fx.frame_path(15)).unwrap();
    let Err(err) = Cam0Driver::open(fx.root(), FIXTURE_DATE, FIXTURE_DRIVE) else {
        panic!("open accepted a drive with 15 PNGs and 16 timestamps");
    };
    let DriverError::Layout(LayoutError::MissingFile { line, name, .. }) = &err else {
        panic!("expected a line with no file, got {err:?}");
    };
    assert_eq!((*line, name.as_str()), (16, "0000000015.png"));
    assert!(
        err.to_string()
            .contains("timestamps.txt line 16 has a time, but 0000000015.png is not in"),
        "the message must name the line and the file: {err}"
    );
}

/// P4. A directory with the right PNG *count* and the wrong *names* used to
/// pass `open`: the run exited 0, every invariant printed OK, and the frames
/// that were never there became `Missing{decode_error}`, which
/// `admitted + missing == n_frames` absorbs without a murmur. The wrong answer
/// was a silent one, which is the worst kind.
///
/// The accept / reject / accept sandwich is the point. A test that only showed
/// the broken fixture being refused would pass just as well against an `open`
/// that refused everything, and would never have been observed failing for the
/// reason it claims.
#[test]
fn driver_open_rejects_right_count_wrong_names() {
    let fx = FixtureDrive::new(16).unwrap();
    assert!(
        Cam0Driver::open(fx.root(), FIXTURE_DATE, FIXTURE_DRIVE).is_ok(),
        "positive control: the untouched fixture must open"
    );

    // Renamed, not deleted: 16 PNGs before, 16 PNGs after.
    fx.rename_in_data("0000000015.png", "frame_0015.png")
        .unwrap();
    let pngs = std::fs::read_dir(fx.image_dir().join("data"))
        .unwrap()
        .filter_map(Result::ok)
        .filter(|e| e.path().extension().is_some_and(|x| x == "png"))
        .count();
    assert_eq!(pngs, 16, "the rename must not change the PNG count");

    let Err(err) = Cam0Driver::open(fx.root(), FIXTURE_DATE, FIXTURE_DRIVE) else {
        panic!("open accepted 16 PNGs of which one is not a frame of this drive");
    };
    // Refused by the NAME, before any count is taken: the file has no frame.
    let DriverError::Layout(LayoutError::NotAFrameNumber { name, .. }) = &err else {
        panic!("expected a name that is not a frame, got {err:?}");
    };
    assert_eq!(name, "frame_0015.png");
    assert!(err.to_string().contains("frame_0015.png"), "{err}");

    // Positive control the other way round: put the name back and the same 16
    // files open cleanly, so the rejection was caused by the name and by
    // nothing else this test did.
    fx.rename_in_data("frame_0015.png", "0000000015.png")
        .unwrap();
    assert!(
        Cam0Driver::open(fx.root(), FIXTURE_DATE, FIXTURE_DRIVE).is_ok(),
        "restoring the filename must restore the drive"
    );
}

#[test]
fn driver_open_rejects_a_stray_png() {
    let fx = FixtureDrive::new(16).unwrap();
    // A stray file that is not a frame at all is refused by its name.
    let stray = fx.image_dir().join("data").join("thumb.png");
    std::fs::copy(fx.frame_path(0), &stray).unwrap();
    let Err(err) = Cam0Driver::open(fx.root(), FIXTURE_DATE, FIXTURE_DRIVE) else {
        panic!("open accepted a PNG that is not named by a frame");
    };
    assert!(
        matches!(&err, DriverError::Layout(LayoutError::NotAFrameNumber { name, .. }) if name == "thumb.png"),
        "{err:?}"
    );
    std::fs::remove_file(&stray).unwrap();
    // Every frame present and correctly named, and one frame-named file past
    // the last line: a file with no time.
    std::fs::copy(
        fx.frame_path(0),
        fx.image_dir().join("data").join("0000000016.png"),
    )
    .unwrap();
    let Err(err) = Cam0Driver::open(fx.root(), FIXTURE_DATE, FIXTURE_DRIVE) else {
        panic!("open accepted 17 PNGs against 16 timestamps");
    };
    let DriverError::Layout(LayoutError::FileBeyondEnd {
        dir, lines, line, ..
    }) = &err
    else {
        panic!("expected a file past the last line, got {err:?}");
    };
    assert_eq!((*lines, *line), (16, 17));
    // The message used to name a hard-coded "image_02/data" and no drive at
    // all, so with several scenes in play it could not say which one failed.
    assert_eq!(dir, &fx.image_dir().join("data"));
    let msg = err.to_string();
    assert!(
        msg.contains(&fx.image_dir().join("data").display().to_string()),
        "the message must name the real directory: {msg}"
    );
    assert!(
        msg.contains(FIXTURE_DRIVE),
        "the message must name the failing drive: {msg}"
    );
}

#[test]
fn driver_open_reports_a_missing_drive() {
    let fx = FixtureDrive::new(1).unwrap();
    let Err(err) = Cam0Driver::open(fx.root(), FIXTURE_DATE, "no_such_drive") else {
        panic!("open accepted a drive directory that does not exist");
    };
    // `DriveMissing`, not `Io`: the spec was written before that variant was
    // split out to name the directory and the `--kitti-root` override.
    assert!(
        matches!(&err, DriverError::DriveMissing(m) if m.level == MissingLevel::Drive),
        "expected DriveMissing at the Drive level, got {err:?}"
    );
    assert!(err.to_string().contains("no_such_drive"));
    // The whole value of knowing the level: the message can list what *is*
    // there, which turns a dead end into a next step.
    assert!(
        err.to_string().contains(FIXTURE_DRIVE),
        "the message must list the drive that does exist: {err}"
    );
}

/// One byte-identical sentence covered a typo'd `--drive`, a typo'd date prefix,
/// a wrong `--kitti-root` and a drive with no `image_02/`, so it told the user
/// nothing about which of the four they had got wrong. The assertions are on
/// the rendered text as well as the variant, because the text is what the user
/// reads.
#[test]
fn driver_open_distinguishes_the_four_missing_levels() {
    let fx = FixtureDrive::new(1).unwrap();
    let level =
        |root: &std::path::Path, date: &str, drive: &str| match Cam0Driver::open(root, date, drive)
        {
            Err(DriverError::DriveMissing(m)) => m.level,
            Err(e) => panic!("expected DriveMissing for {date}/{drive}, got {e:?}"),
            Ok(_) => panic!("open accepted {date}/{drive}"),
        };
    let message =
        |root: &std::path::Path, date: &str, drive: &str| match Cam0Driver::open(root, date, drive)
        {
            Err(e) => e.to_string(),
            Ok(_) => panic!("open accepted {date}/{drive}"),
        };

    // A drive directory that exists and carries no camera: the case that used
    // to read exactly like a typo.
    std::fs::create_dir_all(fx.root().join(FIXTURE_DATE).join("bare_drive")).unwrap();
    let absent_root = fx.root().join("no_such_root");

    assert_eq!(
        level(&absent_root, FIXTURE_DATE, FIXTURE_DRIVE),
        MissingLevel::Root
    );
    assert_eq!(
        level(fx.root(), "1999_01_01", FIXTURE_DRIVE),
        MissingLevel::Date
    );
    assert_eq!(
        level(fx.root(), FIXTURE_DATE, "no_such_drive"),
        MissingLevel::Drive
    );
    assert_eq!(
        level(fx.root(), FIXTURE_DATE, "bare_drive"),
        MissingLevel::Image02
    );

    let texts = [
        message(&absent_root, FIXTURE_DATE, FIXTURE_DRIVE),
        message(fx.root(), "1999_01_01", FIXTURE_DRIVE),
        message(fx.root(), FIXTURE_DATE, "no_such_drive"),
        message(fx.root(), FIXTURE_DATE, "bare_drive"),
    ];
    for (i, a) in texts.iter().enumerate() {
        for b in &texts[i + 1..] {
            assert_ne!(a, b, "two of the four failures print the same sentence");
        }
    }
    assert!(texts[1].contains("1999_01_01"), "{}", texts[1]);
    assert!(
        texts[1].contains(FIXTURE_DATE),
        "the date that does exist must be listed: {}",
        texts[1]
    );
    assert!(texts[3].contains("image_02"), "{}", texts[3]);
}

#[test]
fn driver_run_unpaced_admits_every_frame() {
    let fx = FixtureDrive::new(16).unwrap();
    let d = Cam0Driver::open(fx.root(), FIXTURE_DATE, FIXTURE_DRIVE).unwrap();
    // rate_factor = inf => ClockModel::due is None => skip_at is None => the
    // skip branch cannot fire. admitted == 16 holds by construction, not by
    // this machine happening to be fast enough.
    let clock = ClockModel::start_now(d.timestamps[0], f64::INFINITY);
    let schema = cam0_schema();
    let mut events: Vec<DriverEvent> = Vec::new();
    let mut collect = |e: DriverEvent| events.push(e);
    let totals = d.run(&clock, Duration::ZERO, &schema, &mut collect);
    assert_eq!(
        totals,
        RunTotals {
            admitted: 16,
            missing: 0,
            wall_ns: totals.wall_ns
        }
    );
    assert!(totals.wall_ns >= 0);
    assert_eq!(events.len(), 16);
    for (i, ev) in events.iter().enumerate() {
        let DriverEvent::Sample(s) = ev else {
            panic!("frame {i} was Missing in an unpaced run");
        };
        assert_eq!(s.seq, i as u64);
        assert_eq!(s.due, None, "unpaced frames carry no deadline");
        assert_eq!(
            s.tov,
            Tov::Time(SensorTime(fixture_timestamp_ns(i, FIXTURE_PERIOD_NS)))
        );
        assert_eq!(cam0_storage_id(&s.payload), Some(s.storage_id));
        assert_eq!(cam0_dims(&s.payload), Some((8, 4)));
    }
}

#[test]
fn driver_run_paced_keeps_absolute_deadlines() {
    // 8 frames 10 ms apart at rate 10 => 1 ms deadlines: ~200 ms of start lead
    // plus 8 ms of replay.
    let fx = FixtureDrive::with_period(8, 10_000_000).unwrap();
    let d = Cam0Driver::open(fx.root(), FIXTURE_DATE, FIXTURE_DRIVE).unwrap();
    let clock = ClockModel::start_now(d.timestamps[0], 10.0);
    let schema = cam0_schema();
    let mut events: Vec<DriverEvent> = Vec::new();
    let mut collect = |e: DriverEvent| events.push(e);
    let totals = d.run(&clock, Duration::from_millis(1), &schema, &mut collect);
    // Conservation, never `missing == 0`: a runner that stalls past a deadline
    // is exactly what Missing{deadline_skipped} is for.
    assert_eq!(totals.admitted + totals.missing, 8);
    assert_eq!(events.len(), 8);

    let admitted: Vec<_> = events
        .iter()
        .filter_map(|e| match e {
            DriverEvent::Sample(s) => Some(s),
            DriverEvent::Missing { .. } => None,
        })
        .collect();
    assert_eq!(admitted.len() as u64, totals.admitted);
    let Some(first) = admitted.first() else {
        panic!("a paced run at rate 10 admitted nothing at all");
    };
    let due0 = first.due.unwrap();
    let tov0 = first.tov.start().unwrap();
    for s in &admitted {
        let due = s.due.unwrap();
        let tov = s.tov.start().unwrap();
        // Absolute deadlines: phase is never reset, so the deadline spacing is
        // the sensor spacing divided by the rate, exactly.
        assert_eq!(
            due - due0,
            (tov - tov0) / 10,
            "frame {} drifted off its absolute deadline",
            s.seq
        );
    }
}

// ---- gaps in the camera's source --------------------------------------------
//
// The camera read as the lidar is (`pipes_kitti::layout`): ten frame numbers,
// ten lines in `timestamps.txt`, and no PNG for frames 3 and 4, whose lines
// (4 and 5) are blank.

const GAP_FRAMES: usize = 10;
const GAP_ABSENT: [usize; 2] = [3, 4];

fn gapped_camera() -> FixtureDrive {
    let fx = FixtureDrive::new(GAP_FRAMES).unwrap();
    write_fixture_gapped_at(
        fx.root(),
        FIXTURE_DATE,
        FIXTURE_DRIVE,
        GAP_FRAMES,
        &GAP_ABSENT,
        FIXTURE_PERIOD_NS,
    )
    .unwrap();
    fx
}

/// The rule, on the camera: line `i + 1` is frame `i`, blank where the frame
/// has no PNG, a frame keeps its number, and frames 3-4 are absent in the
/// source.
#[test]
fn a_camera_gap_opens_and_line_i_plus_1_is_frame_i() {
    let fx = gapped_camera();
    let text = std::fs::read_to_string(fx.image_dir().join("timestamps.txt")).unwrap();
    let lines: Vec<&str> = text.lines().collect();
    assert_eq!(lines.len(), GAP_FRAMES);
    assert_eq!((lines[3], lines[4]), ("", ""), "lines 4-5 are frames 3-4");
    let d = Cam0Driver::open(fx.root(), FIXTURE_DATE, FIXTURE_DRIVE)
        .expect("KITTI's gapped layout must open on the camera too");
    assert_eq!((d.len(), d.frame_slots()), (8, GAP_FRAMES));
    assert_eq!(d.absent_in_source(), vec![3, 4]);
    assert_eq!(
        d.index.frame(3),
        Some(5),
        "the fourth PNG is the one named 5"
    );
    assert_eq!(
        d.timestamps[3],
        SensorTime(fixture_timestamp_ns(5, FIXTURE_PERIOD_NS)),
        "and carries frame 5's instant"
    );
    assert!(d.file_path(3).unwrap().ends_with("data/0000000005.png"));
    let gaps = d.gaps();
    assert_eq!(gaps.len(), 1);
    assert_eq!((gaps[0].first, gaps[0].last), (3, 4));
    assert_eq!(gaps[0].hole_ns, Some(3 * FIXTURE_PERIOD_NS));
}

/// The replay: one event per frame slot, in frame order; every frame's
/// `seq` is its number and its pixels are its own PNG's; the two absent
/// frames come out as `Missing` with the reason that names the cause and no
/// time of validity.
#[test]
fn a_camera_gap_is_replayed_as_missing_inputs_with_no_invented_instant() {
    let fx = gapped_camera();
    let d = Cam0Driver::open(fx.root(), FIXTURE_DATE, FIXTURE_DRIVE).unwrap();
    let clock = ClockModel::start_now(d.timestamps[0], f64::INFINITY);
    let schema = cam0_schema();
    let mut events: Vec<DriverEvent> = Vec::new();
    let totals = d.run(&clock, Duration::ZERO, &schema, &mut |e| events.push(e));
    assert_eq!((totals.admitted, totals.missing), (8, 2));
    assert_eq!(events.len(), GAP_FRAMES, "one event per frame slot");
    for (i, ev) in events.iter().enumerate() {
        match ev {
            DriverEvent::Sample(s) => {
                assert!(!GAP_ABSENT.contains(&i), "frame {i} was invented");
                assert_eq!(s.seq, i as u64);
                assert_eq!(
                    s.tov,
                    Tov::Time(SensorTime(fixture_timestamp_ns(i, FIXTURE_PERIOD_NS)))
                );
                // Pixel (0,0)'s red is the frame number the PNG was written for.
                assert_eq!(
                    pipes_kitti::frame::cam0_pixels(&s.payload).map(|p| p[0]),
                    Some(i as u8),
                    "frame {i} carries another frame's pixels"
                );
            }
            DriverEvent::Missing {
                seq,
                tov,
                due,
                reason,
            } => {
                assert!(GAP_ABSENT.contains(&i), "frame {i} was lost: {reason}");
                assert_eq!(*seq, i as u64);
                assert_eq!(*reason, ABSENT_IN_SOURCE);
                assert_eq!(*tov, Tov::None, "an instant was invented for frame {i}");
                assert_eq!(*due, None, "unpaced");
            }
        }
    }
}

/// Paced, the replay sits through the hole: each absent frame is reported
/// once its derived deadline has passed, and frame 5 is not handed over
/// until three periods after frame 2's deadline.
#[test]
fn a_paced_camera_replay_keeps_the_hole() {
    let fx = FixtureDrive::with_period(GAP_FRAMES, 10_000_000).unwrap();
    write_fixture_gapped_at(
        fx.root(),
        FIXTURE_DATE,
        FIXTURE_DRIVE,
        GAP_FRAMES,
        &GAP_ABSENT,
        10_000_000,
    )
    .unwrap();
    let d = Cam0Driver::open(fx.root(), FIXTURE_DATE, FIXTURE_DRIVE).unwrap();
    // Rate 1 at a 10 ms period: the drive in 100 ms.
    let clock = ClockModel::start_now(d.timestamps[0], 1.0);
    let schema = cam0_schema();
    let mut seen: Vec<(u64, &'static str, Option<HostTime>, HostTime, Tov)> = Vec::new();
    d.run(&clock, Duration::from_micros(500), &schema, &mut |e| {
        let t = pipes_core::clock::now();
        match e {
            DriverEvent::Sample(s) => seen.push((s.seq, "", s.due, t, s.tov)),
            DriverEvent::Missing {
                seq,
                due,
                reason,
                tov,
            } => seen.push((seq, reason, due, t, tov)),
        }
    });
    let of = |f: u64| *seen.iter().find(|w| w.0 == f).unwrap();
    for f in GAP_ABSENT {
        let (_, reason, due, t, tov) = of(f as u64);
        assert_eq!(reason, ABSENT_IN_SOURCE);
        // Paced, the slot has a deadline -- and still no instant.
        assert_eq!(tov, Tov::None, "frame {f}: an instant was invented");
        let due = due.expect("paced, so a derived deadline");
        assert!(t >= due, "frame {f} reported before its slot");
    }
    let (_, reason, due5, t5, _) = of(5);
    assert_eq!(reason, "", "frame 5 was not handed over: {reason}");
    let due2 = of(2).2.unwrap();
    assert_eq!(due5.unwrap() - due2, 30_000_000);
    assert!(t5 - due2 >= 30_000_000, "the hole was compressed");
}

/// A PNG renamed to another frame's number keeps the count and the names
/// valid; the lines are what tell. Frame 5's line has a time and its PNG is
/// gone, so the drive is refused at line 6 -- where a real gap, whose line is
/// blank, opens.
#[test]
fn a_renamed_png_the_timestamps_contradict_is_refused() {
    let fx = FixtureDrive::new(8).unwrap();
    fx.rename_in_data("0000000005.png", "0000000009.png")
        .unwrap();
    let Err(err) = Cam0Driver::open(fx.root(), FIXTURE_DATE, FIXTURE_DRIVE) else {
        panic!("open accepted a PNG whose name its timestamp contradicts");
    };
    assert!(
        matches!(
            &err,
            DriverError::Layout(LayoutError::MissingFile { line: 6, name, .. })
                if name == "0000000005.png"
        ),
        "{err:?}"
    );
    // Positive control: the name restored, the drive opens with no gap.
    fx.rename_in_data("0000000009.png", "0000000005.png")
        .unwrap();
    let d = Cam0Driver::open(fx.root(), FIXTURE_DATE, FIXTURE_DRIVE).unwrap();
    assert!(d.absent_in_source().is_empty());
    assert_eq!(d.frame_slots(), d.len());
}

/// A camera whose `timestamps.txt` ends in blank lines declares those frames
/// and has no PNG for them: absent in the source, replayed as missing inputs.
#[test]
fn trailing_blank_lines_are_camera_frames_absent_in_the_source() {
    let fx = FixtureDrive::new(GAP_FRAMES).unwrap();
    write_fixture_gapped_at(
        fx.root(),
        FIXTURE_DATE,
        FIXTURE_DRIVE,
        GAP_FRAMES,
        &[8, 9],
        FIXTURE_PERIOD_NS,
    )
    .unwrap();
    let text = std::fs::read_to_string(fx.image_dir().join("timestamps.txt")).unwrap();
    assert!(
        text.ends_with("\n\n\n"),
        "two trailing blank lines: {text:?}"
    );
    let d = Cam0Driver::open(fx.root(), FIXTURE_DATE, FIXTURE_DRIVE)
        .expect("trailing blank lines are KITTI's layout");
    assert_eq!((d.len(), d.frame_slots()), (8, GAP_FRAMES));
    assert_eq!(d.absent_in_source(), vec![8, 9]);
    let clock = ClockModel::start_now(d.timestamps[0], f64::INFINITY);
    let mut events: Vec<(u64, &'static str)> = Vec::new();
    let totals = d.run(&clock, Duration::ZERO, &cam0_schema(), &mut |e| {
        events.push(match e {
            DriverEvent::Sample(s) => (s.seq, ""),
            DriverEvent::Missing { seq, reason, .. } => (seq, reason),
        })
    });
    assert_eq!((totals.admitted, totals.missing), (8, 2));
    assert_eq!(
        &events[8..],
        &[(8, ABSENT_IN_SOURCE), (9, ABSENT_IN_SOURCE)]
    );
}

/// A camera that stops before the drive does is extended to the drive's
/// frames: the ones past its last line are absent in its source and replayed
/// as such. It never shrinks.
#[test]
fn a_camera_extended_to_the_drive_replays_its_missing_tail() {
    let fx = FixtureDrive::new(8).unwrap();
    let open = || Cam0Driver::open(fx.root(), FIXTURE_DATE, FIXTURE_DRIVE).unwrap();
    assert_eq!(open().with_frame_slots(4).frame_slots(), 8, "shrank");
    let d = open().with_frame_slots(GAP_FRAMES);
    assert_eq!((d.len(), d.frame_slots()), (8, GAP_FRAMES));
    assert_eq!(d.absent_in_source(), vec![8, 9]);
    let gaps = d.gaps();
    assert_eq!((gaps[0].first, gaps[0].last, gaps[0].after), (8, 9, None));
    let clock = ClockModel::start_now(d.timestamps[0], f64::INFINITY);
    let mut events: Vec<(u64, &'static str)> = Vec::new();
    let totals = d.run(&clock, Duration::ZERO, &cam0_schema(), &mut |e| {
        events.push(match e {
            DriverEvent::Sample(s) => (s.seq, ""),
            DriverEvent::Missing { seq, reason, .. } => (seq, reason),
        })
    });
    assert_eq!((totals.admitted, totals.missing), (8, 2));
    assert_eq!(
        &events[8..],
        &[(8, ABSENT_IN_SOURCE), (9, ABSENT_IN_SOURCE)]
    );
}
