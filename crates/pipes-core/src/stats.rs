//! Small statistics helpers for the evidence summary.

/// Nearest-rank percentile of `v`, which **must already be sorted ascending**.
/// `p` is in percent (0..=100): `k = ceil(p / 100 * n)`, result
/// `v[k.clamp(1, n) - 1]`.
///
/// Returns `None` for an empty slice. There is no percentile of no samples,
/// and a caller that has none must say so rather than report a number it never
/// measured — `0` is a value this pipeline genuinely records (a queue wait of
/// zero, a pacing error of zero), so it cannot double as "no data".
///
/// Taking several percentiles of one sample set means sorting once and calling
/// this repeatedly; [`percentile`] is the sort-then-ask shorthand for one.
pub fn percentile_sorted(v: &[i64], p: f64) -> Option<i64> {
    let n = v.len();
    if n == 0 {
        return None;
    }
    debug_assert!(
        v.is_sorted(),
        "percentile_sorted was handed an unsorted slice; use percentile()"
    );
    let k = (p / 100.0 * n as f64).ceil() as usize;
    v.get(k.clamp(1, n) - 1).copied()
}

/// Sorts `v` ascending **in place**, then takes [`percentile_sorted`]. The
/// `&mut` in the signature is the whole warning: this reorders the caller's
/// slice, so nothing downstream may rely on its original order.
///
/// Returns `None` for an empty slice, for the reason given on
/// [`percentile_sorted`].
pub fn percentile(v: &mut [i64], p: f64) -> Option<i64> {
    v.sort_unstable();
    percentile_sorted(v, p)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::{percentile, percentile_sorted};

    #[test]
    fn nearest_rank_1_to_100() {
        let mut v: Vec<i64> = (1..=100).collect();
        assert_eq!(percentile(&mut v, 50.0), Some(50));
        assert_eq!(percentile(&mut v, 99.0), Some(99));
        assert_eq!(percentile(&mut v, 100.0), Some(100));
    }

    #[test]
    fn empty_is_none_not_zero() {
        let mut v: Vec<i64> = Vec::new();
        assert_eq!(percentile(&mut v, 50.0), None);
        assert_eq!(percentile_sorted(&v, 50.0), None);
        // The point of the change: a real measurement of zero is Some(0), so a
        // caller can tell "nothing was measured" from "zero was measured".
        assert_eq!(percentile(&mut [0i64], 50.0), Some(0));
    }

    #[test]
    fn sorts_unsorted_input() {
        let mut v = vec![5, 1, 4, 2, 3];
        assert_eq!(percentile(&mut v, 0.0), Some(1));
        assert_eq!(percentile(&mut v, 100.0), Some(5));
        assert_eq!(v, vec![1, 2, 3, 4, 5], "percentile sorts in place");
    }

    #[test]
    fn sorted_variant_leaves_the_slice_alone() {
        let v = vec![1, 2, 3, 4, 5];
        assert_eq!(percentile_sorted(&v, 50.0), Some(3));
        assert_eq!(percentile_sorted(&v, 100.0), Some(5));
        assert_eq!(v, vec![1, 2, 3, 4, 5]);
    }
}
