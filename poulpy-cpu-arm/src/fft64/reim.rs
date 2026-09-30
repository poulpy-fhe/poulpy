//! Real/imaginary interleaved FFT primitives for [`FFT64Neon`](super::FFT64Neon).

#[cfg(not(target_arch = "aarch64"))]
use poulpy_cpu_portable::kernels::fft64::reim::{fft_portable, ifft_portable};
use poulpy_cpu_portable::kernels::fft64::{
    convolution::I64Ops,
    reim::{ReimArith, ReimFFTExecute, ReimFFTTable, ReimIFFTTable},
    reim4::{Reim4BlkMatVec, Reim4Convolution},
};
use poulpy_hal::api::{NegacyclicFFT, NegacyclicFFTNew};

use super::FFT64Neon;
use poulpy_hal::layouts::Ring;

/// Precomputed twiddle-factor tables for the negacyclic reim FFT and IFFT,
/// dispatching to NEON-accelerated kernels on AArch64 and the portable
/// portable kernels otherwise.
/// Wraps [`ReimFFTTable`] and [`ReimIFFTTable`] into a single object that
/// implements [`NegacyclicFFT`], suitable for use as the transform provider
/// in the CPU CKKS encoding implementation.
pub struct FFT64NeonReimTable {
    fft: ReimFFTTable<f64>,
    ifft: ReimIFFTTable<f64>,
}

impl NegacyclicFFT<f64> for FFT64NeonReimTable {
    fn m(&self) -> usize {
        self.fft.m()
    }

    fn fft(&self, data: &mut [f64]) {
        ReimFFTNeon::reim_dft_execute(&self.fft, data);
    }

    fn ifft(&self, data: &mut [f64]) {
        ReimIFFTNeon::reim_dft_execute(&self.ifft, data);
    }
}

impl NegacyclicFFTNew<f64> for FFT64NeonReimTable {
    fn new(m: usize) -> Self {
        Self {
            fft: ReimFFTTable::new(m),
            ifft: ReimIFFTTable::new(m),
        }
    }
}

/// Negacyclic transform for CKKS encoding in `f64`: the NEON kernels on the
/// correctly rounded twiddles of
/// [`EncodingFFTTable`](poulpy_cpu_portable::ckks_encoding::EncodingFFTTable),
/// byte identical to every other CPU backend.
#[cfg(feature = "enable-ckks")]
pub struct FFT64NeonEncodingTable(poulpy_cpu_portable::ckks_encoding::EncodingFFTTable<f64>);

#[cfg(feature = "enable-ckks")]
impl NegacyclicFFT<f64> for FFT64NeonEncodingTable {
    fn m(&self) -> usize {
        self.0.m()
    }

    fn fft(&self, data: &mut [f64]) {
        ReimFFTNeon::reim_dft_execute(self.0.forward(), data);
    }

    fn ifft(&self, data: &mut [f64]) {
        ReimIFFTNeon::reim_dft_execute(self.0.inverse(), data);
    }
}

#[cfg(feature = "enable-ckks")]
impl NegacyclicFFTNew<f64> for FFT64NeonEncodingTable {
    fn new(m: usize) -> Self {
        Self(NegacyclicFFTNew::new(m))
    }
}

pub struct ReimFFTNeon;

impl ReimFFTExecute<ReimFFTTable<f64>, f64> for ReimFFTNeon {
    #[inline(always)]
    fn reim_dft_execute(table: &ReimFFTTable<f64>, data: &mut [f64]) {
        #[cfg(target_arch = "aarch64")]
        {
            crate::neon::fft::fft_neon(table.m(), table.omg(), data);
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            fft_portable(table.m(), table.omg(), data);
        }
    }
}

pub struct ReimIFFTNeon;

impl ReimFFTExecute<ReimIFFTTable<f64>, f64> for ReimIFFTNeon {
    #[inline(always)]
    fn reim_dft_execute(table: &ReimIFFTTable<f64>, data: &mut [f64]) {
        #[cfg(target_arch = "aarch64")]
        {
            crate::neon::fft::ifft_neon(table.m(), table.omg(), data);
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            ifft_portable(table.m(), table.omg(), data);
        }
    }
}

#[cfg(target_arch = "aarch64")]
impl<R: Ring> ReimFFTExecute<ReimFFTTable<f64>, f64> for FFT64Neon<R> {
    fn reim_dft_execute(table: &ReimFFTTable<f64>, data: &mut [f64]) {
        crate::neon::fft::fft_neon(table.m(), table.omg(), data);
    }
}

#[cfg(not(target_arch = "aarch64"))]
impl<R: Ring> ReimFFTExecute<ReimFFTTable<f64>, f64> for FFT64Neon<R> {
    fn reim_dft_execute(table: &ReimFFTTable<f64>, data: &mut [f64]) {
        fft_portable(table.m(), table.omg(), data);
    }
}

#[cfg(target_arch = "aarch64")]
impl<R: Ring> ReimFFTExecute<ReimIFFTTable<f64>, f64> for FFT64Neon<R> {
    fn reim_dft_execute(table: &ReimIFFTTable<f64>, data: &mut [f64]) {
        crate::neon::fft::ifft_neon(table.m(), table.omg(), data);
    }
}

#[cfg(not(target_arch = "aarch64"))]
impl<R: Ring> ReimFFTExecute<ReimIFFTTable<f64>, f64> for FFT64Neon<R> {
    fn reim_dft_execute(table: &ReimIFFTTable<f64>, data: &mut [f64]) {
        ifft_portable(table.m(), table.omg(), data);
    }
}

#[cfg(target_arch = "aarch64")]
impl<R: Ring> ReimArith for FFT64Neon<R> {
    // reim_add / reim_add_assign: defer to the portable autovec impl. The
    // hand-NEON loop (see neon::reim_arith::reim_add_neon) is memory-bandwidth
    // bound at large n; the autovec reference is as fast or faster.
    #[inline(always)]
    fn reim_add(res: &mut [f64], a: &[f64], b: &[f64]) {
        poulpy_cpu_portable::kernels::fft64::reim::reim_add_portable(res, a, b);
    }
    #[inline(always)]
    fn reim_add_assign(res: &mut [f64], a: &[f64]) {
        poulpy_cpu_portable::kernels::fft64::reim::reim_add_assign_portable(res, a);
    }
    #[inline(always)]
    fn reim_sub(res: &mut [f64], a: &[f64], b: &[f64]) {
        crate::neon::reim_arith::reim_sub_neon(res, a, b);
    }
    #[inline(always)]
    fn reim_sub_assign(res: &mut [f64], a: &[f64]) {
        crate::neon::reim_arith::reim_sub_assign_neon(res, a);
    }
    #[inline(always)]
    fn reim_sub_negate_assign(res: &mut [f64], a: &[f64]) {
        crate::neon::reim_arith::reim_sub_negate_assign_neon(res, a);
    }
    #[inline(always)]
    fn reim_negate(res: &mut [f64], a: &[f64]) {
        crate::neon::reim_arith::reim_negate_neon(res, a);
    }
    #[inline(always)]
    fn reim_negate_assign(res: &mut [f64]) {
        crate::neon::reim_arith::reim_negate_assign_neon(res);
    }
    #[inline(always)]
    fn reim_mul(res: &mut [f64], a: &[f64], b: &[f64]) {
        crate::neon::reim_arith::reim_mul_neon(res, a, b);
    }
    #[inline(always)]
    fn reim_mul_assign(res: &mut [f64], a: &[f64]) {
        crate::neon::reim_arith::reim_mul_assign_neon(res, a);
    }
    #[inline(always)]
    fn reim_addmul(res: &mut [f64], a: &[f64], b: &[f64]) {
        crate::neon::reim_arith::reim_addmul_neon(res, a, b);
    }
    #[inline(always)]
    fn reim_from_znx(res: &mut [f64], a: &[i64]) {
        crate::neon::reim_arith::reim_from_znx_i64_bnd50_neon(res, a);
    }
    #[inline(always)]
    fn reim_to_znx(res: &mut [i64], divisor: f64, a: &[f64]) {
        crate::neon::reim_arith::reim_to_znx_i64_bnd63_neon(res, divisor, a);
    }
    #[inline(always)]
    fn reim_to_znx_assign(res: &mut [f64], divisor: f64) {
        crate::neon::reim_arith::reim_to_znx_i64_assign_bnd63_neon(res, divisor);
    }
}

#[cfg(not(target_arch = "aarch64"))]
impl<R: Ring> ReimArith for FFT64Neon<R> {}

#[cfg(target_arch = "aarch64")]
impl<R: Ring> Reim4BlkMatVec for FFT64Neon<R> {
    #[inline(always)]
    fn reim4_extract_1blk_contiguous(m: usize, rows: usize, blk: usize, dst: &mut [f64], src: &[f64]) {
        crate::neon::reim4_arith::reim4_extract_1blk_contiguous_neon(m, rows, blk, dst, src);
    }
    #[inline(always)]
    fn reim4_save_1blk_contiguous(m: usize, rows: usize, blk: usize, dst: &mut [f64], src: &[f64]) {
        crate::neon::reim4_arith::reim4_save_1blk_contiguous_neon(m, rows, blk, dst, src);
    }
    #[inline(always)]
    fn reim4_save_1blk<const OVERWRITE: bool>(m: usize, blk: usize, dst: &mut [f64], src: &[f64]) {
        crate::neon::reim4_arith::reim4_save_1blk_neon::<OVERWRITE>(m, blk, dst, src);
    }
    #[inline(always)]
    fn reim4_save_2blks<const OVERWRITE: bool>(m: usize, blk: usize, dst: &mut [f64], src: &[f64]) {
        crate::neon::reim4_arith::reim4_save_2blks_neon::<OVERWRITE>(m, blk, dst, src);
    }
    #[inline(always)]
    fn reim4_mat1col_prod(nrows: usize, dst: &mut [f64], u: &[f64], v: &[f64]) {
        crate::neon::reim4_arith::reim4_mat1col_prod_neon(nrows, dst, u, v);
    }
    #[inline(always)]
    fn reim4_mat2cols_prod(nrows: usize, dst: &mut [f64], u: &[f64], v: &[f64]) {
        crate::neon::reim4_arith::reim4_mat2cols_prod_neon(nrows, dst, u, v);
    }
    #[inline(always)]
    fn reim4_mat2cols_2ndcol_prod(nrows: usize, dst: &mut [f64], u: &[f64], v: &[f64]) {
        crate::neon::reim4_arith::reim4_mat2cols_2ndcol_prod_neon(nrows, dst, u, v);
    }
    #[inline(always)]
    fn reim4_real_mat1col_prod(nrows: usize, dst: &mut [f64], u: &[f64], v: &[f64]) {
        unsafe { crate::neon::reim4_arith::reim4_real_mat_prod_neon::<1, 8>(nrows, dst, u, v, 0) }
    }
    #[inline(always)]
    fn reim4_real_mat2cols_prod(nrows: usize, dst: &mut [f64], u: &[f64], v: &[f64]) {
        unsafe { crate::neon::reim4_arith::reim4_real_mat_prod_neon::<2, 16>(nrows, dst, u, v, 0) }
    }
    #[inline(always)]
    fn reim4_real_mat2cols_2ndcol_prod(nrows: usize, dst: &mut [f64], u: &[f64], v: &[f64]) {
        unsafe { crate::neon::reim4_arith::reim4_real_mat_prod_neon::<1, 16>(nrows, dst, u, v, 8) }
    }
}

#[cfg(not(target_arch = "aarch64"))]
impl<R: Ring> Reim4BlkMatVec for FFT64Neon<R> {}

#[cfg(target_arch = "aarch64")]
impl<R: Ring> Reim4Convolution for FFT64Neon<R> {
    #[inline(always)]
    fn reim4_convolution_1coeff(k: usize, dst: &mut [f64; 8], a: &[f64], a_size: usize, b: &[f64], b_size: usize) {
        crate::neon::reim4_conv::reim4_convolution_1coeff_neon(k, dst, a, a_size, b, b_size);
    }
    #[inline(always)]
    fn reim4_convolution_2coeffs(k: usize, dst: &mut [f64; 16], a: &[f64], a_size: usize, b: &[f64], b_size: usize) {
        crate::neon::reim4_conv::reim4_convolution_2coeffs_neon(k, dst, a, a_size, b, b_size);
    }
    #[inline(always)]
    fn reim4_real_convolution_1coeff(k: usize, dst: &mut [f64; 8], a: &[f64], a_size: usize, b: &[f64], b_size: usize) {
        unsafe { crate::neon::reim4_conv::reim4_real_convolution_1coeff_neon(k, dst, a, a_size, b, b_size) }
    }
    #[inline(always)]
    fn reim4_real_convolution_2coeffs(k: usize, dst: &mut [f64; 16], a: &[f64], a_size: usize, b: &[f64], b_size: usize) {
        let (lo, hi) = dst.split_at_mut(8);
        Self::reim4_real_convolution_1coeff(k, lo.try_into().unwrap(), a, a_size, b, b_size);
        Self::reim4_real_convolution_1coeff(k + 1, hi.try_into().unwrap(), a, a_size, b, b_size);
    }
    #[inline(always)]
    fn reim4_convolution_by_real_const_1coeff(k: usize, dst: &mut [f64; 8], a: &[f64], a_size: usize, b: &[f64]) {
        crate::neon::reim4_conv::reim4_convolution_by_real_const_1coeff_neon(k, dst, a, a_size, b);
    }
    #[inline(always)]
    fn reim4_convolution_by_real_const_2coeffs(k: usize, dst: &mut [f64; 16], a: &[f64], a_size: usize, b: &[f64]) {
        crate::neon::reim4_conv::reim4_convolution_by_real_const_2coeffs_neon(k, dst, a, a_size, b);
    }
}

#[cfg(not(target_arch = "aarch64"))]
impl<R: Ring> Reim4Convolution for FFT64Neon<R> {}

#[cfg(target_arch = "aarch64")]
impl<R: Ring> I64Ops for FFT64Neon<R> {
    #[inline(always)]
    fn i64_extract_1blk_contiguous(n: usize, offset: usize, rows: usize, blk: usize, dst: &mut [i64], src: &[i64]) {
        crate::neon::conv_i64::i64_extract_1blk_contiguous_neon(n, offset, rows, blk, dst, src);
    }
    #[inline(always)]
    fn i64_save_1blk_contiguous(n: usize, offset: usize, rows: usize, blk: usize, dst: &mut [i64], src: &[i64]) {
        crate::neon::conv_i64::i64_save_1blk_contiguous_neon(n, offset, rows, blk, dst, src);
    }
    #[inline(always)]
    fn i64_convolution_by_const_1coeff(k: usize, dst: &mut [i64; 8], a: &[i64], a_size: usize, b: &[i64]) {
        crate::neon::conv_i64::i64_convolution_by_const_1coeff_neon(k, dst, a, a_size, b);
    }
    #[inline(always)]
    fn i64_convolution_by_const_2coeffs(k: usize, dst: &mut [i64; 16], a: &[i64], a_size: usize, b: &[i64]) {
        crate::neon::conv_i64::i64_convolution_by_const_2coeffs_neon(k, dst, a, a_size, b);
    }
}

#[cfg(not(target_arch = "aarch64"))]
impl<R: Ring> I64Ops for FFT64Neon<R> {}

impl<R: Ring> poulpy_cpu_portable::hal_defaults::BigWordHadamardProduct for FFT64Neon<R> {
    #[inline(always)]
    fn big_word_hadamard_product(res: &mut [i64], a: &[i64], b: &[i64]) {
        <Self as I64Ops>::i64_hadamard_product(res, a, b)
    }
}

#[cfg(test)]
mod contract_tests {
    use super::*;
    use poulpy_hal::api::NegacyclicFFTNew;

    #[test]
    fn raw_transform_matches_contract() {
        for m in [16, 32, 64, 128, 256] {
            let table = FFT64NeonReimTable::new(m);
            poulpy_hal::test_suite::reim::test_negacyclic_fft(&table);
        }
    }
}

#[cfg(all(test, feature = "enable-ckks"))]
mod encoding_tests {
    use poulpy_cpu_portable::ckks_encoding::EncodingFFTTable;
    use poulpy_hal::{api::NegacyclicFFTNew, test_suite::reim::test_negacyclic_fft_bit_exact};

    use super::FFT64NeonEncodingTable;

    #[test]
    fn encoding_transform_matches_portable() {
        for log_m in 0..=15 {
            let m = 1 << log_m;
            test_negacyclic_fft_bit_exact::<f64, _, _>(&FFT64NeonEncodingTable::new(m), &EncodingFFTTable::<f64>::new(m));
        }
    }
}
