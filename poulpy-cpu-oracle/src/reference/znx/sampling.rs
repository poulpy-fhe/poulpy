#[cfg(feature = "enable-core")]
use rand_distr::{Distribution, Normal};

use poulpy_hal::source::Source;

pub fn znx_fill_uniform_ref(base2k: usize, res: &mut [i64], source: &mut Source) {
    let pow2k: u64 = 1 << base2k;
    let mask: u64 = pow2k - 1;
    let pow2k_half: i64 = (pow2k >> 1) as i64;
    res.iter_mut()
        .for_each(|xi| *xi = (source.next_u64n(pow2k, mask) as i64) - pow2k_half)
}

#[cfg(feature = "enable-core")]
pub fn znx_add_normal_f64_ref(res: &mut [i64], sigma: f64, bound: f64, shift: u32, source: &mut Source) {
    let normal: Normal<f64> = Normal::new(0.0, sigma).unwrap();
    res.iter_mut().for_each(|xi| {
        let mut dist_f64: f64 = normal.sample(source);
        while dist_f64.abs() > bound {
            dist_f64 = normal.sample(source)
        }
        *xi += (dist_f64.round() as i64) << shift
    })
}
