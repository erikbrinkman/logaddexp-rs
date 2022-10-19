//! Crate for computing [`ln_add_exp`][LogAddExp::ln_add_exp] and
//! [`ln_sum_exp`][LogSumExp::ln_sum_exp]
//!
//! These functions allow more numerically stable implementations for this order of operations than
//! doing them naively. They come in handy when doing computation in log-space.
//!
//! # Examples
//!
//! If you have a large number in log space, you can add one to it by doing
//! ```
//! use logaddexp::LogAddExp;
//!
//! let ln_large_number: f64 = // ...
//! # 100.0;
//! let ln_large_number_1p = ln_large_number.ln_add_exp(0.0);
//! ```
//!
//! You can use [LogSumExp] to handle an [Iterator] of floats.
//!
//! ```
//! use logaddexp::LogSumExp;
//!
//! (1..100).into_iter().map(|v| v as f64).ln_sum_exp();
//! ```
#![warn(missing_docs)]

use num_traits::{Float, FloatConst, Zero};
use std::ops::Add;

/// A trait for computing ln_add_exp
pub trait LogAddExp<Rhs = Self> {
    /// The result of the computation
    type Output;

    /// Compute the log of the addition of the exponentials
    ///
    /// This computes the same value value as `(self.exp() + other.exp()).ln()` but in a more
    /// numerically stable way then computing it using that formula.
    ///
    /// # Examples
    ///
    /// ```
    /// use logaddexp::LogAddExp;
    /// 100_f64.ln().ln_add_exp(0.0); // 101_f64.ln()
    /// ```
    fn ln_add_exp(self, other: Rhs) -> Self::Output;
}

impl<T> LogAddExp for T
where
    T: Float + FloatConst,
{
    type Output = T;

    fn ln_add_exp(self, other: Self) -> Self {
        if self == other {
            self + Self::LN_2()
        } else {
            let diff = self - other;
            if diff.is_nan() {
                diff
            } else if diff > Self::zero() {
                self + (-diff).exp().ln_1p()
            } else {
                other + diff.exp().ln_1p()
            }
        }
    }
}

impl<'a, T> LogAddExp<&'a T> for T
where
    T: Float + FloatConst,
{
    type Output = T;

    fn ln_add_exp(self, other: &'a Self) -> T {
        self.ln_add_exp(*other)
    }
}

/// A trait for computing ln_sum_exp
pub trait LogSumExp {
    /// The result of the computation
    type Output;

    /// Compute the log of the sum of exponentials
    ///
    /// This computes the same value value as `self.map(|v| v.exp()).sum().ln()` but in a more
    /// numerically stable way then computing it using that formula. This is also slightly more
    /// stable then doing `self.reduce(|a, b| a.ln_add_exp(b))`.
    ///
    /// # Examples
    ///
    /// ```
    /// use logaddexp::LogSumExp;
    /// [1.0, 2.0, 4.0].into_iter().ln_sum_exp();
    /// ```
    fn ln_sum_exp(self) -> Self::Output;
}

impl<T> LogSumExp for T
where
    T: Iterator + Clone,
    T::Item: Float + FloatConst,
{
    type Output = T::Item;

    fn ln_sum_exp(self) -> Self::Output {
        if let Some(max) = self.clone().reduce(Self::Output::max) {
            if max.is_nan() {
                max
            } else {
                let sum = self
                    .map(|val| (val - max).exp())
                    .reduce(Self::Output::add)
                    .unwrap_or_else(Self::Output::zero);
                sum.ln() + max
            }
        } else {
            Self::Output::neg_infinity()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{LogAddExp, LogSumExp};

    macro_rules! assert_close {
        ($a:expr, $b:expr, rtol = $rtol:expr, atol = $atol:expr) => {{
            let a = $a;
            let b = $b;
            assert!(
                (a - b).abs() <= $atol + $rtol * b.abs(),
                "assertion failed: `(left !== right)`\n  left: `{:?}`,\n right: `{:?}`",
                a,
                b,
            );
        }};
        ($a:expr, $b:expr, atol = $atol:expr, rtol = $rtol:expr) => {
            assert_close!($a, $b, rol = $rtol, atol = $atol);
        };
        ($a:expr, $b:expr, rtol = $rtol:expr) => {
            assert_close!($a, $b, rtol = $rtol, atol = 1e-8);
        };
        ($a:expr, $b:expr, atol = $atol:expr) => {
            assert_close!($a, $b, atol = $atol, rtol = 1e-5);
        };
        ($a:expr, $b:expr) => {
            assert_close!($a, $b, rtol = 1e-5);
        };
    }

    #[test]
    fn test_ln_add_exp() {
        assert_close!(f64::ln_add_exp(1.0, 1.0), 1.0 + 2_f64.ln());
        assert_close!(1.0.ln_add_exp(2.0), (1_f64.exp() + 2_f64.exp()).ln());
        assert_close!(f64::ln_add_exp(0.0, &0.0), 2_f64.ln());
        assert_close!(2_f64.ln().ln_add_exp(&0.0), 3_f64.ln());
        assert!(f64::NAN.ln_add_exp(&1.0).is_nan());
        assert!(1.0.ln_add_exp(f64::NAN).is_nan());
        assert_eq!(f64::INFINITY.ln_add_exp(&0.0), f64::INFINITY);
        assert_eq!(1.0.ln_add_exp(f64::INFINITY), f64::INFINITY);
        assert_eq!(f64::INFINITY.ln_add_exp(f64::INFINITY), f64::INFINITY);
        assert_eq!(f64::NEG_INFINITY.ln_add_exp(f64::INFINITY), f64::INFINITY);
        assert_eq!(f64::INFINITY.ln_add_exp(f64::NEG_INFINITY), f64::INFINITY);
        assert_eq!(
            f64::NEG_INFINITY.ln_add_exp(f64::NEG_INFINITY),
            f64::NEG_INFINITY
        );
    }

    #[test]
    fn test_ln_sum_exp() {
        let raw = (1..10).into_iter().map(|n| (n as f64).ln());

        let binary = raw.clone().reduce(f64::ln_add_exp).unwrap();
        let expected: u64 = (1..10).sum();
        assert_close!(binary, (expected as f64).ln());

        let actual = raw.ln_sum_exp();
        assert_close!(actual, binary);

        assert_eq!(<[f64; 0]>::into_iter([]).ln_sum_exp(), f64::NEG_INFINITY);

        assert!([f64::NAN, 1.0].into_iter().ln_sum_exp().is_nan());
    }
}
