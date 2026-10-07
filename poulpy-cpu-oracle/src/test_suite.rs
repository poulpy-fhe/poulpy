//! Oracles that copy the backend under test's draws, for comparing
//! independently implemented encryption paths.

use std::marker::PhantomData;

use poulpy_hal::layouts::{ConjugateInvariant, Standard};

use crate::{DFTFamily, Fft64, Ntt4x30, Oracle};

/// The transform family `F`, whose oracle samples by copying the draws of the
/// current [`poulpy_core::test_suite::parity::controlled_sampling::with_backend_samples`]
/// scope. Its arithmetic is `F`'s; drawing outside such a scope panics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ControlledSampling<F>(PhantomData<F>);

impl<F: DFTFamily> DFTFamily for ControlledSampling<F> {
    type Dft = F::Dft;
    type Big = F::Big;
    type Table = F::Table;

    fn table(n: usize) -> Self::Table {
        F::table(n)
    }
    fn forward(table: &Self::Table, res: &mut [Self::Dft], a: &[i64]) {
        F::forward(table, res, a)
    }
    fn inverse(table: &Self::Table, res: &mut [Self::Big], a: &[Self::Dft]) {
        F::inverse(table, res, a)
    }
    fn dft_embed(res: &mut [Self::Dft], a: &[Self::Dft]) {
        F::dft_embed(res, a)
    }
    fn dft_add(a: Self::Dft, b: Self::Dft) -> Self::Dft {
        F::dft_add(a, b)
    }
    fn dft_neg(a: Self::Dft) -> Self::Dft {
        F::dft_neg(a)
    }
    fn mul_acc(res: &mut [Self::Dft], a: &[Self::Dft], b: &[Self::Dft]) {
        F::mul_acc(res, a, b)
    }
    fn mul_assign(res: &mut [Self::Dft], a: &[Self::Dft]) {
        F::mul_assign(res, a)
    }
    fn dft_automorphism(p: i64, res: &mut [Self::Dft], a: &[Self::Dft]) {
        F::dft_automorphism(p, res, a)
    }
    fn ci_expand(res: &mut [Self::Dft], a: &[Self::Dft]) {
        F::ci_expand(res, a)
    }
    fn ci_mul_acc(res: &mut [Self::Dft], a: &[Self::Dft], b: &[Self::Dft]) {
        F::ci_mul_acc(res, a, b)
    }
    fn ci_mul_assign(res: &mut [Self::Dft], a: &[Self::Dft]) {
        F::ci_mul_assign(res, a)
    }
}

/// [`crate::FFT64Oracle`] with controlled sampling.
pub type ControlledSamplingFFT64Oracle<R = Standard> = Oracle<ControlledSampling<Fft64>, R>;

/// [`crate::NTT4x30Oracle`] with controlled sampling.
pub type ControlledSamplingNTT4x30Oracle<R = Standard> = Oracle<ControlledSampling<Ntt4x30>, R>;

/// [`crate::FFT64CIOracle`] with controlled sampling.
pub type ControlledSamplingFFT64CIOracle = ControlledSamplingFFT64Oracle<ConjugateInvariant>;
