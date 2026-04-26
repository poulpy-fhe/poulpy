use std::ptr::NonNull;

use poulpy_cpu_ref::reference::{
    fft64::{
        convolution::I64Ops,
        module::{FFT64HandleFactory, FFTHandleProvider},
        reim::{ReimArith, ReimFFTExecute, ReimFFTTable, ReimIFFTTable, reim_copy_ref, reim_zero_ref},
        reim4::{Reim4BlkMatVec, Reim4Convolution},
    },
    znx::{
        ZnxAdd, ZnxAddAssign, ZnxAutomorphism, ZnxCopy, ZnxExtractDigitAddMul, ZnxMulAddPowerOfTwo, ZnxMulPowerOfTwo,
        ZnxMulPowerOfTwoAssign, ZnxNegate, ZnxNegateAssign, ZnxNormalizeDigit, ZnxNormalizeFinalStep,
        ZnxNormalizeFinalStepAssign, ZnxNormalizeFinalStepSub, ZnxNormalizeFirstStep, ZnxNormalizeFirstStepAssign,
        ZnxNormalizeFirstStepCarryOnly, ZnxNormalizeMiddleStep, ZnxNormalizeMiddleStepAssign, ZnxNormalizeMiddleStepCarryOnly,
        ZnxNormalizeMiddleStepSub, ZnxRotate, ZnxSub, ZnxSubAssign, ZnxSubNegateAssign, ZnxSwitchRing, ZnxZero, znx_copy_ref,
        znx_rotate, znx_zero_ref,
    },
};
use poulpy_hal::{alloc_aligned, assert_alignment, layouts::Backend};

use crate::{
    FFT64Avx,
    fft64::{
        convolution::{
            i64_convolution_by_const_1coeff_avx, i64_convolution_by_real_const_2coeffs_avx, i64_extract_1blk_contiguous_avx,
            i64_save_1blk_contiguous_avx,
        },
        reim::{
            ReimFFTAvx, ReimIFFTAvx, reim_add_assign_avx2_fma, reim_add_avx2_fma, reim_addmul_avx2_fma,
            reim_from_znx_i64_bnd50_fma, reim_from_znx_i64_masked_bnd50_fma, reim_mul_assign_avx2_fma, reim_mul_avx2_fma,
            reim_negate_assign_avx2_fma, reim_negate_avx2_fma, reim_sub_assign_avx2_fma, reim_sub_avx2_fma,
            reim_sub_negate_assign_avx2_fma, reim_to_znx_i64_assign_bnd63_avx2_fma, reim_to_znx_i64_bnd63_avx2_fma,
        },
        reim4::{
            reim4_convolution_1coeff_avx, reim4_convolution_2coeffs_avx, reim4_convolution_by_real_const_1coeff_avx,
            reim4_convolution_by_real_const_2coeffs_avx, reim4_extract_1blk_from_reim_contiguous_avx,
            reim4_save_1blk_to_reim_avx, reim4_save_1blk_to_reim_contiguous_avx, reim4_save_2blk_to_reim_avx,
            reim4_vec_mat1col_product_avx, reim4_vec_mat2cols_2ndcol_product_avx, reim4_vec_mat2cols_product_avx,
        },
    },
    znx_avx::{
        znx_add_assign_avx, znx_add_avx, znx_automorphism_avx, znx_extract_digit_addmul_avx, znx_mul_add_power_of_two_avx,
        znx_mul_power_of_two_assign_avx, znx_mul_power_of_two_avx, znx_negate_assign_avx, znx_negate_avx,
        znx_normalize_digit_avx, znx_normalize_final_step_assign_avx, znx_normalize_final_step_avx,
        znx_normalize_final_step_sub_avx, znx_normalize_first_step_assign_avx, znx_normalize_first_step_avx,
        znx_normalize_first_step_carry_only_avx, znx_normalize_middle_step_assign_avx, znx_normalize_middle_step_avx,
        znx_normalize_middle_step_carry_only_avx, znx_normalize_middle_step_sub_avx, znx_sub_assign_avx, znx_sub_avx,
        znx_sub_negate_assign_avx, znx_switch_ring_avx,
    },
};

/// Backend-specific handle storing precomputed FFT/IFFT twiddle factors.
///
/// This structure is allocated once during [`Module::new()`](poulpy_hal::layouts::Module::new)
/// and persists for the lifetime of the module. It contains precomputed complex roots of unity
/// (twiddle factors) required for efficient FFT and inverse FFT operations on ring elements
/// of degree `n`.
///
/// # Memory layout
///
/// - **Alignment**: Natural alignment for `f64` arrays (8 bytes).
/// - **Size**: `O(n)` storage for `n`-degree polynomial ring (twiddle tables scale linearly).
/// - **Ownership**: Managed via `Box` and leaked to obtain a stable `NonNull` pointer stored in `Module`.
///
/// # Thread safety
///
/// Twiddle tables are **immutable** after construction, making this type safe to share across threads
/// via `&Module<FFT64Avx>`. The `Module` type enforces `Send + Sync` bounds.
///
/// # Destruction
///
/// The handle is destroyed via [`Backend::destroy()`](poulpy_hal::layouts::Backend::destroy)
/// when the module is dropped, which reconstructs the `Box` from the raw pointer and drops it.
#[repr(C)]
pub struct FFT64AvxHandle {
    table_fft: ReimFFTTable<f64>,
    table_ifft: ReimIFFTTable<f64>,
}

impl Backend for FFT64Avx {
    type ScalarPrep = f64;
    type ScalarBig = i64;
    type OwnedBuf = Vec<u8>;
    type Handle = FFT64AvxHandle;
    fn alloc_bytes(len: usize) -> Self::OwnedBuf {
        alloc_aligned::<u8>(len)
    }
    fn from_bytes(bytes: Vec<u8>) -> Self::OwnedBuf {
        assert_alignment(bytes.as_ptr());
        bytes
    }
    unsafe fn destroy(handle: NonNull<Self::Handle>) {
        unsafe {
            drop(Box::from_raw(handle.as_ptr()));
        }
    }
}

/// # Safety
///
/// The returned handle must be fully initialized for `n`.
unsafe impl FFT64HandleFactory for FFT64AvxHandle {
    fn create_fft64_handle(n: usize) -> Self {
        FFT64AvxHandle {
            table_fft: ReimFFTTable::new(n >> 1),
            table_ifft: ReimIFFTTable::new(n >> 1),
        }
    }

    fn assert_fft64_runtime_support() {
        if !std::arch::is_x86_feature_detected!("avx")
            || !std::arch::is_x86_feature_detected!("avx2")
            || !std::arch::is_x86_feature_detected!("fma")
        {
            panic!("arch must support avx2, avx and fma")
        }
    }
}

unsafe impl FFTHandleProvider<f64> for FFT64AvxHandle {
    fn get_fft_table(&self) -> &ReimFFTTable<f64> {
        &self.table_fft
    }

    fn get_ifft_table(&self) -> &ReimIFFTTable<f64> {
        &self.table_ifft
    }
}

impl ZnxAdd for FFT64Avx {
    #[inline(always)]
    fn znx_add(res: &mut [i64], a: &[i64], b: &[i64]) {
        unsafe {
            znx_add_avx(res, a, b);
        }
    }
}

impl ZnxAddAssign for FFT64Avx {
    #[inline(always)]
    fn znx_add_assign(res: &mut [i64], a: &[i64]) {
        unsafe {
            znx_add_assign_avx(res, a);
        }
    }
}

impl ZnxSub for FFT64Avx {
    #[inline(always)]
    fn znx_sub(res: &mut [i64], a: &[i64], b: &[i64]) {
        unsafe {
            znx_sub_avx(res, a, b);
        }
    }
}

impl ZnxSubAssign for FFT64Avx {
    #[inline(always)]
    fn znx_sub_assign(res: &mut [i64], a: &[i64]) {
        unsafe {
            znx_sub_assign_avx(res, a);
        }
    }
}

impl ZnxSubNegateAssign for FFT64Avx {
    #[inline(always)]
    fn znx_sub_negate_assign(res: &mut [i64], a: &[i64]) {
        unsafe {
            znx_sub_negate_assign_avx(res, a);
        }
    }
}

impl ZnxAutomorphism for FFT64Avx {
    #[inline(always)]
    fn znx_automorphism(p: i64, res: &mut [i64], a: &[i64]) {
        unsafe {
            znx_automorphism_avx(p, res, a);
        }
    }
}

impl ZnxCopy for FFT64Avx {
    #[inline(always)]
    fn znx_copy(res: &mut [i64], a: &[i64]) {
        znx_copy_ref(res, a);
    }
}

impl ZnxNegate for FFT64Avx {
    #[inline(always)]
    fn znx_negate(res: &mut [i64], src: &[i64]) {
        unsafe {
            znx_negate_avx(res, src);
        }
    }
}

impl ZnxNegateAssign for FFT64Avx {
    #[inline(always)]
    fn znx_negate_assign(res: &mut [i64]) {
        unsafe {
            znx_negate_assign_avx(res);
        }
    }
}

impl ZnxMulAddPowerOfTwo for FFT64Avx {
    #[inline(always)]
    fn znx_muladd_power_of_two(k: i64, res: &mut [i64], a: &[i64]) {
        unsafe {
            znx_mul_add_power_of_two_avx(k, res, a);
        }
    }
}

impl ZnxMulPowerOfTwo for FFT64Avx {
    #[inline(always)]
    fn znx_mul_power_of_two(k: i64, res: &mut [i64], a: &[i64]) {
        unsafe {
            znx_mul_power_of_two_avx(k, res, a);
        }
    }
}

impl ZnxMulPowerOfTwoAssign for FFT64Avx {
    #[inline(always)]
    fn znx_mul_power_of_two_assign(k: i64, res: &mut [i64]) {
        unsafe {
            znx_mul_power_of_two_assign_avx(k, res);
        }
    }
}

impl ZnxRotate for FFT64Avx {
    #[inline(always)]
    fn znx_rotate(p: i64, res: &mut [i64], src: &[i64]) {
        znx_rotate::<Self>(p, res, src);
    }
}

impl ZnxZero for FFT64Avx {
    #[inline(always)]
    fn znx_zero(res: &mut [i64]) {
        znx_zero_ref(res);
    }
}

impl ZnxSwitchRing for FFT64Avx {
    #[inline(always)]
    fn znx_switch_ring(res: &mut [i64], a: &[i64]) {
        unsafe {
            znx_switch_ring_avx(res, a);
        }
    }
}

impl ZnxNormalizeFirstStep for FFT64Avx {
    #[inline(always)]
    fn znx_normalize_first_step<const OVERWRITE: bool>(base2k: usize, lsh: usize, x: &mut [i64], a: &[i64], carry: &mut [i64]) {
        unsafe {
            znx_normalize_first_step_avx::<OVERWRITE>(base2k, lsh, x, a, carry);
        }
    }
}

impl ZnxNormalizeMiddleStep for FFT64Avx {
    #[inline(always)]
    fn znx_normalize_middle_step<const OVERWRITE: bool>(base2k: usize, lsh: usize, x: &mut [i64], a: &[i64], carry: &mut [i64]) {
        unsafe {
            znx_normalize_middle_step_avx::<OVERWRITE>(base2k, lsh, x, a, carry);
        }
    }
}

impl ZnxNormalizeFinalStep for FFT64Avx {
    #[inline(always)]
    fn znx_normalize_final_step<const OVERWRITE: bool>(base2k: usize, lsh: usize, x: &mut [i64], a: &[i64], carry: &mut [i64]) {
        unsafe {
            znx_normalize_final_step_avx::<OVERWRITE>(base2k, lsh, x, a, carry);
        }
    }
}

impl ZnxNormalizeMiddleStepSub for FFT64Avx {
    #[inline(always)]
    fn znx_normalize_middle_step_sub(base2k: usize, lsh: usize, x: &mut [i64], a: &[i64], carry: &mut [i64]) {
        unsafe {
            znx_normalize_middle_step_sub_avx(base2k, lsh, x, a, carry);
        }
    }
}

impl ZnxNormalizeFinalStepSub for FFT64Avx {
    #[inline(always)]
    fn znx_normalize_final_step_sub(base2k: usize, lsh: usize, x: &mut [i64], a: &[i64], carry: &mut [i64]) {
        unsafe {
            znx_normalize_final_step_sub_avx(base2k, lsh, x, a, carry);
        }
    }
}

impl ZnxNormalizeFinalStepAssign for FFT64Avx {
    #[inline(always)]
    fn znx_normalize_final_step_assign(base2k: usize, lsh: usize, x: &mut [i64], carry: &mut [i64]) {
        unsafe {
            znx_normalize_final_step_assign_avx(base2k, lsh, x, carry);
        }
    }
}

impl ZnxNormalizeFirstStepCarryOnly for FFT64Avx {
    #[inline(always)]
    fn znx_normalize_first_step_carry_only(base2k: usize, lsh: usize, x: &[i64], carry: &mut [i64]) {
        unsafe {
            znx_normalize_first_step_carry_only_avx(base2k, lsh, x, carry);
        }
    }
}

impl ZnxNormalizeFirstStepAssign for FFT64Avx {
    #[inline(always)]
    fn znx_normalize_first_step_assign(base2k: usize, lsh: usize, x: &mut [i64], carry: &mut [i64]) {
        unsafe {
            znx_normalize_first_step_assign_avx(base2k, lsh, x, carry);
        }
    }
}

impl ZnxNormalizeMiddleStepCarryOnly for FFT64Avx {
    #[inline(always)]
    fn znx_normalize_middle_step_carry_only(base2k: usize, lsh: usize, x: &[i64], carry: &mut [i64]) {
        unsafe {
            znx_normalize_middle_step_carry_only_avx(base2k, lsh, x, carry);
        }
    }
}

impl ZnxNormalizeMiddleStepAssign for FFT64Avx {
    #[inline(always)]
    fn znx_normalize_middle_step_assign(base2k: usize, lsh: usize, x: &mut [i64], carry: &mut [i64]) {
        unsafe {
            znx_normalize_middle_step_assign_avx(base2k, lsh, x, carry);
        }
    }
}

impl ZnxExtractDigitAddMul for FFT64Avx {
    #[inline(always)]
    fn znx_extract_digit_addmul(base2k: usize, lsh: usize, res: &mut [i64], src: &mut [i64]) {
        unsafe {
            znx_extract_digit_addmul_avx(base2k, lsh, res, src);
        }
    }
}

impl ZnxNormalizeDigit for FFT64Avx {
    #[inline(always)]
    fn znx_normalize_digit(base2k: usize, res: &mut [i64], src: &mut [i64]) {
        unsafe {
            znx_normalize_digit_avx(base2k, res, src);
        }
    }
}

impl ReimFFTExecute<ReimFFTTable<f64>, f64> for FFT64Avx {
    #[inline(always)]
    fn reim_dft_execute(table: &ReimFFTTable<f64>, data: &mut [f64]) {
        ReimFFTAvx::reim_dft_execute(table, data);
    }
}

impl ReimFFTExecute<ReimIFFTTable<f64>, f64> for FFT64Avx {
    #[inline(always)]
    fn reim_dft_execute(table: &ReimIFFTTable<f64>, data: &mut [f64]) {
        ReimIFFTAvx::reim_dft_execute(table, data);
    }
}

impl ReimArith for FFT64Avx {
    #[inline(always)]
    fn reim_from_znx(res: &mut [f64], a: &[i64]) {
        unsafe { reim_from_znx_i64_bnd50_fma(res, a) }
    }

    #[inline(always)]
    fn reim_from_znx_masked(res: &mut [f64], a: &[i64], mask: i64) {
        unsafe { reim_from_znx_i64_masked_bnd50_fma(res, a, mask) }
    }

    #[inline(always)]
    fn reim_to_znx(res: &mut [i64], divisor: f64, a: &[f64]) {
        unsafe { reim_to_znx_i64_bnd63_avx2_fma(res, divisor, a) }
    }

    #[inline(always)]
    fn reim_to_znx_assign(res: &mut [f64], divisor: f64) {
        unsafe { reim_to_znx_i64_assign_bnd63_avx2_fma(res, divisor) }
    }

    #[inline(always)]
    fn reim_add(res: &mut [f64], a: &[f64], b: &[f64]) {
        unsafe { reim_add_avx2_fma(res, a, b) }
    }

    #[inline(always)]
    fn reim_add_assign(res: &mut [f64], a: &[f64]) {
        unsafe { reim_add_assign_avx2_fma(res, a) }
    }

    #[inline(always)]
    fn reim_sub(res: &mut [f64], a: &[f64], b: &[f64]) {
        unsafe { reim_sub_avx2_fma(res, a, b) }
    }

    #[inline(always)]
    fn reim_sub_assign(res: &mut [f64], a: &[f64]) {
        unsafe { reim_sub_assign_avx2_fma(res, a) }
    }

    #[inline(always)]
    fn reim_sub_negate_assign(res: &mut [f64], a: &[f64]) {
        unsafe { reim_sub_negate_assign_avx2_fma(res, a) }
    }

    #[inline(always)]
    fn reim_negate(res: &mut [f64], a: &[f64]) {
        unsafe { reim_negate_avx2_fma(res, a) }
    }

    #[inline(always)]
    fn reim_negate_assign(res: &mut [f64]) {
        unsafe { reim_negate_assign_avx2_fma(res) }
    }

    #[inline(always)]
    fn reim_mul(res: &mut [f64], a: &[f64], b: &[f64]) {
        unsafe { reim_mul_avx2_fma(res, a, b) }
    }

    #[inline(always)]
    fn reim_mul_assign(res: &mut [f64], a: &[f64]) {
        unsafe { reim_mul_assign_avx2_fma(res, a) }
    }

    #[inline(always)]
    fn reim_addmul(res: &mut [f64], a: &[f64], b: &[f64]) {
        unsafe { reim_addmul_avx2_fma(res, a, b) }
    }

    #[inline(always)]
    fn reim_copy(res: &mut [f64], a: &[f64]) {
        reim_copy_ref(res, a)
    }

    #[inline(always)]
    fn reim_zero(res: &mut [f64]) {
        reim_zero_ref(res)
    }
}

impl Reim4BlkMatVec for FFT64Avx {
    #[inline(always)]
    fn reim4_extract_1blk_contiguous(m: usize, rows: usize, blk: usize, dst: &mut [f64], src: &[f64]) {
        unsafe { reim4_extract_1blk_from_reim_contiguous_avx(m, rows, blk, dst, src) }
    }

    #[inline(always)]
    fn reim4_save_1blk_contiguous(m: usize, rows: usize, blk: usize, dst: &mut [f64], src: &[f64]) {
        unsafe { reim4_save_1blk_to_reim_contiguous_avx(m, rows, blk, dst, src) }
    }

    #[inline(always)]
    fn reim4_save_1blk<const OVERWRITE: bool>(m: usize, blk: usize, dst: &mut [f64], src: &[f64]) {
        unsafe { reim4_save_1blk_to_reim_avx::<OVERWRITE>(m, blk, dst, src) }
    }

    #[inline(always)]
    fn reim4_save_2blks<const OVERWRITE: bool>(m: usize, blk: usize, dst: &mut [f64], src: &[f64]) {
        unsafe { reim4_save_2blk_to_reim_avx::<OVERWRITE>(m, blk, dst, src) }
    }

    #[inline(always)]
    fn reim4_mat1col_prod(nrows: usize, dst: &mut [f64], u: &[f64], v: &[f64]) {
        unsafe { reim4_vec_mat1col_product_avx(nrows, dst, u, v) }
    }

    #[inline(always)]
    fn reim4_mat2cols_prod(nrows: usize, dst: &mut [f64], u: &[f64], v: &[f64]) {
        unsafe { reim4_vec_mat2cols_product_avx(nrows, dst, u, v) }
    }

    #[inline(always)]
    fn reim4_mat2cols_2ndcol_prod(nrows: usize, dst: &mut [f64], u: &[f64], v: &[f64]) {
        unsafe { reim4_vec_mat2cols_2ndcol_product_avx(nrows, dst, u, v) }
    }
}

impl Reim4Convolution for FFT64Avx {
    #[inline(always)]
    fn reim4_convolution_1coeff(k: usize, dst: &mut [f64; 8], a: &[f64], a_size: usize, b: &[f64], b_size: usize) {
        unsafe { reim4_convolution_1coeff_avx(k, dst, a, a_size, b, b_size) }
    }

    #[inline(always)]
    fn reim4_convolution_2coeffs(k: usize, dst: &mut [f64; 16], a: &[f64], a_size: usize, b: &[f64], b_size: usize) {
        unsafe { reim4_convolution_2coeffs_avx(k, dst, a, a_size, b, b_size) }
    }

    #[inline(always)]
    fn reim4_convolution_by_real_const_1coeff(k: usize, dst: &mut [f64; 8], a: &[f64], a_size: usize, b: &[f64]) {
        unsafe { reim4_convolution_by_real_const_1coeff_avx(k, dst, a, a_size, b) }
    }

    #[inline(always)]
    fn reim4_convolution_by_real_const_2coeffs(k: usize, dst: &mut [f64; 16], a: &[f64], a_size: usize, b: &[f64]) {
        unsafe { reim4_convolution_by_real_const_2coeffs_avx(k, dst, a, a_size, b) }
    }
}

impl I64Ops for FFT64Avx {
    #[inline(always)]
    fn i64_extract_1blk_contiguous(n: usize, offset: usize, rows: usize, blk: usize, dst: &mut [i64], src: &[i64]) {
        unsafe { i64_extract_1blk_contiguous_avx(n, offset, rows, blk, dst, src) }
    }

    #[inline(always)]
    fn i64_save_1blk_contiguous(n: usize, offset: usize, rows: usize, blk: usize, dst: &mut [i64], src: &[i64]) {
        unsafe { i64_save_1blk_contiguous_avx(n, offset, rows, blk, dst, src) }
    }

    #[inline(always)]
    fn i64_convolution_by_const_1coeff(k: usize, dst: &mut [i64; 8], a: &[i64], a_size: usize, b: &[i64]) {
        unsafe { i64_convolution_by_const_1coeff_avx(k, dst, a, a_size, b) }
    }

    #[inline(always)]
    fn i64_convolution_by_const_2coeffs(k: usize, dst: &mut [i64; 16], a: &[i64], a_size: usize, b: &[i64]) {
        unsafe { i64_convolution_by_real_const_2coeffs_avx(k, dst, a, a_size, b) }
    }
}
