//! Real/imaginary interleaved FFT primitives for [`FFT64Neon`](super::FFT64Neon).

#[cfg(not(target_arch = "aarch64"))]
use poulpy_cpu_ref::reference::fft64::reim::{fft_ref, ifft_ref};
use poulpy_cpu_ref::reference::fft64::{
    convolution::I64Ops,
    reim::{ReimArith, ReimFFTExecute, ReimFFTTable, ReimIFFTTable},
    reim4::{Reim4BlkMatVec, Reim4Convolution},
};
use poulpy_hal::api::{NegacyclicFFT, NegacyclicFFTNew};

use super::FFT64Neon;

/// Precomputed twiddle-factor tables for the negacyclic reim FFT and IFFT,
/// dispatching to NEON-accelerated kernels on AArch64 and the portable
/// reference kernels otherwise.
/// Wraps [`ReimFFTTable`] and [`ReimIFFTTable`] into a single object that
/// implements [`NegacyclicFFT`], suitable for use as the transform provider
/// in a CKKS [`poulpy_ckks::encoding::Encoder`].
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
            fft_ref(table.m(), table.omg(), data);
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
            ifft_ref(table.m(), table.omg(), data);
        }
    }
}

#[cfg(target_arch = "aarch64")]
impl ReimFFTExecute<ReimFFTTable<f64>, f64> for FFT64Neon {
    fn reim_dft_execute(table: &ReimFFTTable<f64>, data: &mut [f64]) {
        crate::neon::fft::fft_neon(table.m(), table.omg(), data);
    }
}

#[cfg(not(target_arch = "aarch64"))]
impl ReimFFTExecute<ReimFFTTable<f64>, f64> for FFT64Neon {
    fn reim_dft_execute(table: &ReimFFTTable<f64>, data: &mut [f64]) {
        fft_ref(table.m(), table.omg(), data);
    }
}

#[cfg(target_arch = "aarch64")]
impl ReimFFTExecute<ReimIFFTTable<f64>, f64> for FFT64Neon {
    fn reim_dft_execute(table: &ReimIFFTTable<f64>, data: &mut [f64]) {
        crate::neon::fft::ifft_neon(table.m(), table.omg(), data);
    }
}

#[cfg(not(target_arch = "aarch64"))]
impl ReimFFTExecute<ReimIFFTTable<f64>, f64> for FFT64Neon {
    fn reim_dft_execute(table: &ReimIFFTTable<f64>, data: &mut [f64]) {
        ifft_ref(table.m(), table.omg(), data);
    }
}

#[cfg(target_arch = "aarch64")]
impl ReimArith for FFT64Neon {
    // reim_add / reim_add_assign: defer to the portable autovec impl. The
    // hand-NEON loop (see neon::reim_arith::reim_add_neon) is memory-bandwidth
    // bound at large n; the autovec reference is as fast or faster.
    #[inline(always)]
    fn reim_add(res: &mut [f64], a: &[f64], b: &[f64]) {
        poulpy_cpu_ref::reference::fft64::reim::reim_add_ref(res, a, b);
    }
    #[inline(always)]
    fn reim_add_assign(res: &mut [f64], a: &[f64]) {
        poulpy_cpu_ref::reference::fft64::reim::reim_add_assign_ref(res, a);
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
    fn reim_from_znx_masked(res: &mut [f64], a: &[i64], mask: i64) {
        crate::neon::reim_arith::reim_from_znx_i64_masked_bnd50_neon(res, a, mask);
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
impl ReimArith for FFT64Neon {}

#[cfg(target_arch = "aarch64")]
impl Reim4BlkMatVec for FFT64Neon {
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
}

#[cfg(not(target_arch = "aarch64"))]
impl Reim4BlkMatVec for FFT64Neon {}

#[cfg(target_arch = "aarch64")]
impl Reim4Convolution for FFT64Neon {
    #[inline(always)]
    fn reim4_convolution_1coeff(k: usize, dst: &mut [f64; 8], a: &[f64], a_size: usize, b: &[f64], b_size: usize) {
        crate::neon::reim4_conv::reim4_convolution_1coeff_neon(k, dst, a, a_size, b, b_size);
    }
    #[inline(always)]
    fn reim4_convolution_2coeffs(k: usize, dst: &mut [f64; 16], a: &[f64], a_size: usize, b: &[f64], b_size: usize) {
        crate::neon::reim4_conv::reim4_convolution_2coeffs_neon(k, dst, a, a_size, b, b_size);
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
impl Reim4Convolution for FFT64Neon {}

#[cfg(target_arch = "aarch64")]
impl I64Ops for FFT64Neon {
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
impl I64Ops for FFT64Neon {}

impl poulpy_cpu_ref::hal_defaults::ScalarBigHadamardProduct for FFT64Neon {
    #[inline(always)]
    fn scalar_big_hadamard_product(res: &mut [i64], a: &[i64], b: &[i64]) {
        <Self as I64Ops>::i64_hadamard_product(res, a, b)
    }
}
