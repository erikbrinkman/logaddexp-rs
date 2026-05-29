#![feature(test)]
extern crate test;

use logaddexp::{LogAddExp, LogSumExp};
use test::{Bencher, black_box};

/// A spread of finite values in log space whose magnitude grows and whose sign
/// alternates, so that neither operand trivially dominates.
fn sample_values(count: usize) -> Vec<f64> {
    let mut values = Vec::with_capacity(count);
    let mut step = 0.0_f64;
    for index in 0..count {
        let magnitude = step * 0.5;
        let signed = if index % 2 == 0 {
            magnitude
        } else {
            -magnitude
        };
        values.push(signed);
        step += 1.0;
    }
    values
}

#[bench]
fn bench_ln_add_exp(bencher: &mut Bencher) {
    let values = sample_values(1024);
    bencher.iter(|| {
        let mut acc = f64::NEG_INFINITY;
        for &value in &values {
            acc = black_box(acc).ln_add_exp(black_box(value));
        }
        acc
    });
}

#[bench]
fn bench_ln_add_exp_naive(bencher: &mut Bencher) {
    let values = sample_values(1024);
    bencher.iter(|| {
        let mut acc = f64::NEG_INFINITY;
        for &value in &values {
            acc = (black_box(acc).exp() + black_box(value).exp()).ln();
        }
        acc
    });
}

#[bench]
fn bench_ln_sum_exp(bencher: &mut Bencher) {
    let values = sample_values(1024);
    bencher.iter(|| black_box(values.iter().copied()).ln_sum_exp());
}

#[bench]
fn bench_ln_sum_exp_naive(bencher: &mut Bencher) {
    let values = sample_values(1024);
    bencher.iter(|| {
        black_box(&values)
            .iter()
            .map(|value| value.exp())
            .sum::<f64>()
            .ln()
    });
}

#[cfg(feature = "simd")]
#[bench]
fn bench_ln_sum_exp_simd(bencher: &mut Bencher) {
    use logaddexp::SimdLogSumExp;
    let values = sample_values(1024);
    bencher.iter(|| black_box(values.as_slice()).ln_sum_exp_simd());
}
