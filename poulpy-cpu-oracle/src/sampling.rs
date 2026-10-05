//! The `poulpy-core` sampling extension point, on host buffers.

use poulpy_core::{Distribution, NoiseInfos, oep::SamplingImpl};
use poulpy_hal::{
    layouts::{Module, ScalarZnxBackendMut, VecZnxBackendMut, VecZnxBigBackendMut, ZnxViewMut},
    source::Source,
};
use rand_distr::{Distribution as _, Normal};

use crate::{
    ScalarZnxFill,
    backend::Oracle,
    family::{DFTFamily, Int},
    ring::OracleRing,
};

/// Adds rounded Gaussian noise, rejected above `bound`, to the limb holding
/// precision `k`, scaled to that precision.
fn add_normal<T: Int>(limb: &mut [T], base2k: usize, k: usize, sigma: f64, bound: f64, seed: [u8; 32]) {
    assert!((bound.log2().ceil() as i64) < 64, "invalid bound: ceil(log2(bound)) > 63");
    let shift = (k.div_ceil(base2k) * base2k - k) as u32;
    let normal = Normal::new(0.0, sigma).unwrap();
    let mut source = Source::new(seed);
    for x in limb {
        let mut e: f64 = normal.sample(&mut source);
        while e.abs() > bound {
            e = normal.sample(&mut source);
        }
        *x = x.add(T::from(e.round() as i64).shl(shift));
    }
}

unsafe impl<F: DFTFamily, R: OracleRing> SamplingImpl for Oracle<F, R> {
    fn scalar_znx_fill_distribution(
        _module: &Module<Self>,
        res: &mut ScalarZnxBackendMut<'_, Self>,
        res_col: usize,
        dist: Distribution,
        seed: [u8; 32],
    ) {
        let mut source = Source::new(seed);
        match dist {
            Distribution::TernaryFixed(hw) => res.fill_ternary_hw(res_col, hw, &mut source),
            Distribution::TernaryProb(prob) => res.fill_ternary_prob(res_col, prob, &mut source),
            Distribution::BinaryFixed(hw) => res.fill_binary_hw(res_col, hw, &mut source),
            Distribution::BinaryProb(prob) => res.fill_binary_prob(res_col, prob, &mut source),
            Distribution::BinaryBlock(block_size) => res.fill_binary_block(res_col, block_size, &mut source),
            Distribution::ZERO => res.at_mut(res_col, 0).fill(0),
            Distribution::NONE | Distribution::ENCAPSULATED(_) => {
                panic!("scalar_znx_fill_distribution: {dist:?} is not a sampleable distribution")
            }
        }
    }

    fn vec_znx_add_normal(
        _module: &Module<Self>,
        base2k: usize,
        res: &mut VecZnxBackendMut<'_, Self>,
        res_col: usize,
        noise: NoiseInfos,
        seed: [u8; 32],
    ) {
        let limb = noise.k.div_ceil(base2k) - 1;
        add_normal(res.at_mut(res_col, limb), base2k, noise.k, noise.sigma, noise.bound, seed);
    }

    fn vec_znx_big_add_normal(
        _module: &Module<Self>,
        base2k: usize,
        res: &mut VecZnxBigBackendMut<'_, Self>,
        res_col: usize,
        noise: NoiseInfos,
        seed: [u8; 32],
    ) {
        let limb = noise.k.div_ceil(base2k) - 1;
        add_normal(res.at_mut(res_col, limb), base2k, noise.k, noise.sigma, noise.bound, seed);
    }
}
