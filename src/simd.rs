//! SIMD-accelerated `ln_sum_exp` over contiguous slices.
//!
//! The generic [`LogSumExp`](crate::LogSumExp) consumes any iterator and runs a
//! serial, single-pass online reduction. That shape is inherently scalar: it
//! has a loop-carried dependency (the running max can change on any element)
//! and never sees the data as a contiguous block.
//!
//! When the data is already a `&[f64]`/`&[f32]`, we can do better with the
//! classic two-pass formulation — find the max, then sum `exp(x - max)` — both
//! of which vectorize cleanly. The expensive part is `exp`, which [`wide`]
//! evaluates a lane at a time (four `f64` or eight `f32`), lowering to AVX on
//! x86 and NEON on aarch64 when the target enables those features.
//!
//! Whether this is actually faster is target-dependent: it pays off on x86 with
//! wide native lanes (AVX2/AVX-512), but on targets with a fast scalar `exp` and
//! only 128-bit lanes (notably Apple Silicon) the scalar path can win. Benchmark
//! on your target — and build with `-C target-cpu=native` — before relying on it.
//!
//! Non-finite maxima (`±inf`, or `NaN` reaching the max lane) are rare and have
//! fiddly semantics, so those cases delegate to the scalar implementation,
//! which is the source of truth.

use crate::LogSumExp;
use wide::{f32x8, f64x4};

/// SIMD-accelerated [`LogSumExp`](crate::LogSumExp) for slices.
///
/// Enabled by the `simd` feature. Build with a SIMD-capable target (for
/// example `RUSTFLAGS="-C target-cpu=native"`) for the widest codegen.
///
/// # Examples
///
/// ```
/// use logaddexp::SimdLogSumExp;
/// let values = [1.0_f64, 2.0, 4.0];
/// values.as_slice().ln_sum_exp_simd();
/// ```
pub trait SimdLogSumExp {
    /// The result of the computation.
    type Output;

    /// Compute the log of the sum of exponentials of a slice using SIMD.
    ///
    /// Produces the same value as [`LogSumExp::ln_sum_exp`] over the same
    /// elements.
    #[must_use]
    fn ln_sum_exp_simd(self) -> Self::Output;
}

macro_rules! impl_simd_log_sum_exp {
    ($scalar:ty, $vector:ty, $lanes:expr) => {
        impl SimdLogSumExp for &[$scalar] {
            type Output = $scalar;

            fn ln_sum_exp_simd(self) -> $scalar {
                let mut chunks = self.chunks_exact($lanes);

                // pass 1: max
                let mut acc_max = <$vector>::splat(<$scalar>::NEG_INFINITY);
                for chunk in chunks.by_ref() {
                    acc_max = acc_max.max(<$vector>::from(chunk));
                }
                let mut max = <$scalar>::NEG_INFINITY;
                for lane in acc_max.to_array() {
                    if lane > max {
                        max = lane;
                    }
                }
                for &val in chunks.remainder() {
                    if val > max {
                        max = val;
                    }
                }

                // The shifted sum below assumes a finite pivot: with a non-finite
                // max, `x - max` produces `NaN`s whose handling differs from the
                // scalar contract, so fall back to the source of truth.
                if !max.is_finite() {
                    return self.iter().copied().ln_sum_exp();
                }

                // pass 2: sum of exp(x - max)
                let splat_max = <$vector>::splat(max);
                let mut acc_sum = <$vector>::splat(0.0);
                let mut chunks = self.chunks_exact($lanes);
                for chunk in chunks.by_ref() {
                    acc_sum += (<$vector>::from(chunk) - splat_max).exp();
                }
                let mut sum = acc_sum.reduce_add();
                for &val in chunks.remainder() {
                    sum += (val - max).exp();
                }

                sum.ln() + max
            }
        }
    };
}

impl_simd_log_sum_exp!(f64, f64x4, 4);
impl_simd_log_sum_exp!(f32, f32x8, 8);

#[cfg(test)]
mod tests {
    // Several asserts compare against representable special values (the
    // infinities), where `float_cmp` is the intended comparison.
    #![allow(clippy::float_cmp)]

    use super::SimdLogSumExp;
    use crate::LogSumExp;

    fn assert_close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() <= 1e-8 + 1e-9 * expected.abs(),
            "left: {actual:?}, right: {expected:?}",
        );
    }

    #[test]
    fn matches_scalar_for_varied_lengths() {
        // Cover lengths around and across the lane boundary so the chunked
        // body and the scalar remainder are both exercised.
        for len in 0..40 {
            let values: Vec<f64> = (0..len)
                .map(|index| {
                    let step = f64::from(index);
                    if index % 3 == 0 {
                        -step * 0.25
                    } else {
                        step * 0.5
                    }
                })
                .collect();
            let actual = values.as_slice().ln_sum_exp_simd();
            let expected = values.iter().copied().ln_sum_exp();
            if expected.is_finite() {
                assert_close(actual, expected);
            } else {
                assert_eq!(actual, expected);
            }
        }
    }

    #[test]
    fn stable_for_large_values() {
        let values = [1000.0_f64; 9];
        let actual = values.as_slice().ln_sum_exp_simd();
        assert_close(actual, 1000.0 + 9_f64.ln());
        assert!(actual.is_finite());
    }

    #[test]
    fn matches_scalar_edge_cases() {
        let empty: [f64; 0] = [];
        assert_eq!(empty.as_slice().ln_sum_exp_simd(), f64::NEG_INFINITY,);
        assert_eq!(
            [f64::NEG_INFINITY; 5].as_slice().ln_sum_exp_simd(),
            f64::NEG_INFINITY,
        );
        assert_eq!(
            [f64::INFINITY; 5].as_slice().ln_sum_exp_simd(),
            f64::INFINITY,
        );
        assert_eq!(
            [f64::NEG_INFINITY, f64::INFINITY, 0.0]
                .as_slice()
                .ln_sum_exp_simd(),
            f64::INFINITY,
        );
        assert!([f64::NAN, 1.0, 2.0].as_slice().ln_sum_exp_simd().is_nan());
        assert!(
            [f64::INFINITY, f64::NAN]
                .as_slice()
                .ln_sum_exp_simd()
                .is_nan()
        );
    }

    #[test]
    fn f32_matches_scalar() {
        let values = [0.5_f32, 1.5, -2.0, 3.0, 0.0, 4.0, -1.0, 2.5, 1.0];
        let actual = values.as_slice().ln_sum_exp_simd();
        let expected = values.iter().copied().ln_sum_exp();
        assert!((actual - expected).abs() <= 1e-5 * expected.abs() + 1e-6);
    }
}
