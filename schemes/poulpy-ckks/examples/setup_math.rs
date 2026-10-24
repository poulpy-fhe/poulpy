//! Compare portable CKKS setup math with the target's general Quad math routing.
#![allow(clippy::disallowed_methods)] // Deliberately benchmark the platform route.

use std::{hint::black_box, time::Instant};

use num_traits::{Float, FromPrimitive};
use poulpy_ckks::{Quad, numerics::CKKSFloat};

fn median_ns(mut f: impl FnMut(Quad) -> Quad) -> f64 {
    let inputs: Vec<_> = (0..4096)
        .map(|i| Quad::from_f64(0.125 + i as f64 / 4096.0).unwrap())
        .collect();
    for &x in &inputs {
        black_box(f(black_box(x)));
    }
    let mut samples: Vec<_> = (0..7)
        .map(|_| {
            let start = Instant::now();
            for &x in &inputs {
                black_box(f(black_box(x)));
            }
            start.elapsed().as_secs_f64() * 1e9 / inputs.len() as f64
        })
        .collect();
    samples.sort_by(f64::total_cmp);
    samples[samples.len() / 2]
}

fn main() {
    println!(
        "{} {}, libquadmath feature: {}",
        std::env::consts::ARCH,
        std::env::consts::OS,
        cfg!(feature = "libquadmath")
    );
    println!("Median of 7 warmed batches, 4096 inputs in [0.125, 1.125). Record CPU, rustc, flags and Cargo.lock with results.");
    let exponent = Quad::from_f64(1.25).unwrap();
    let cos_platform = median_ns(Float::cos);
    let cos_portable = median_ns(CKKSFloat::ckks_cos);
    let pow_platform = median_ns(|x| x.powf(black_box(exponent)));
    let pow_portable = median_ns(|x| x.ckks_powf(black_box(exponent)));
    println!(
        "cos: Quad {cos_platform:.1} ns, CKKS {cos_portable:.1} ns, ratio {:.2}",
        cos_portable / cos_platform
    );
    println!(
        "powf: Quad {pow_platform:.1} ns, CKKS {pow_portable:.1} ns, ratio {:.2}",
        pow_portable / pow_platform
    );
    println!("These operation ratios do not measure whole CKKS setup. Both routes may use the same portable implementation.");
}
