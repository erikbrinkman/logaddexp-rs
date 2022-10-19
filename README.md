logaddexp-rs
============

Stable implementations of logaddexp and logsumexp in rust. Computing
`log(sum_i(exp(v_i)))` for more than one value can esily result in overflow.
This crate provies implementations for two (ln_add_exp) and many (ln_sum_exp)
that are more stable (less prone to overfloe) than doing that computation
naively.

Usage
-----

Run

```
$ cargo add logadexp
```

Then import the trait you want to use and call the function on the appropriate types

```
use logaddexp::LogAddExp;

f64::ln_add_exp(..., ...);
```

```
use logaddexp::LogSumExp;

[...].into_iter().ln_sum_exp();
```
