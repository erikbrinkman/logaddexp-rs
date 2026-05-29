logaddexp-rs
============

[![crates.io](https://img.shields.io/crates/v/logaddexp)](https://crates.io/crates/logaddexp)
[![docs](https://docs.rs/logaddexp/badge.svg)](https://docs.rs/logaddexp)
[![license](https://img.shields.io/github/license/erikbrinkman/logaddexp-rs)](LICENSE)
[![build](https://github.com/erikbrinkman/logaddexp-rs/actions/workflows/rust.yml/badge.svg)](https://github.com/erikbrinkman/logaddexp-rs/actions/workflows/rust.yml)

Stable implementations of logaddexp and logsumexp in rust. Computing
`log(sum_i(exp(v_i)))` naively can easily overflow. This crate provides
implementations for two values (`ln_add_exp`) and many values (`ln_sum_exp`)
that are more stable (less prone to overflow) than the naive computation.

Usage
-----

Run

```sh
$ cargo add logaddexp
```

Then import the trait you want to use and call the function on the appropriate types

```rs
use logaddexp::LogAddExp;

f64::ln_add_exp(..., ...);
```

```rs
use logaddexp::LogSumExp;

[...].into_iter().ln_sum_exp();
```

Features
--------

- `simd` (off by default) — adds a `SimdLogSumExp` trait with a slice
  specialized `ln_sum_exp_simd` for `&[f64]`/`&[f32]`, backed by the
  [`wide`](https://crates.io/crates/wide) crate's vectorized `exp`. The benefit
  is target dependent: it pays off on x86 with wide native lanes (AVX2/AVX-512)
  but can lose to a fast scalar `exp` on 128-bit-lane targets such as Apple
  Silicon, so benchmark on your target and build with
  `RUSTFLAGS="-C target-cpu=native"`.
