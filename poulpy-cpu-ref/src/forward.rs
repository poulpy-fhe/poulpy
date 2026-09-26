//! Kernel forwarding for backends that reuse another backend's kernels.
//!
//! A conjugate-invariant backend keeps its own module, handle and
//! registrations, and forwards every ring-independent kernel to the standard
//! backend it mirrors: `forward_*_kernels!(Target => Base)` implements each
//! kernel trait for `Target` by calling `Base`'s implementation.

/// Forwards the `Znx*` kernels and [`I64NormalizeOps`](crate::reference::normalization::I64NormalizeOps) to `$base`.
#[macro_export]
macro_rules! forward_znx_kernels {
    ($be:ty => $base:ty) => {
        impl $crate::reference::znx::ZnxAdd for $be {
            #[inline(always)]
            fn znx_add(res: &mut [i64], a: &[i64], b: &[i64]) {
                <$base as $crate::reference::znx::ZnxAdd>::znx_add(res, a, b)
            }
        }
        impl $crate::reference::znx::ZnxAddAssign for $be {
            #[inline(always)]
            fn znx_add_assign(res: &mut [i64], a: &[i64]) {
                <$base as $crate::reference::znx::ZnxAddAssign>::znx_add_assign(res, a)
            }
        }
        impl $crate::reference::znx::ZnxSub for $be {
            #[inline(always)]
            fn znx_sub(res: &mut [i64], a: &[i64], b: &[i64]) {
                <$base as $crate::reference::znx::ZnxSub>::znx_sub(res, a, b)
            }
        }
        impl $crate::reference::znx::ZnxSubAssign for $be {
            #[inline(always)]
            fn znx_sub_assign(res: &mut [i64], a: &[i64]) {
                <$base as $crate::reference::znx::ZnxSubAssign>::znx_sub_assign(res, a)
            }
        }
        impl $crate::reference::znx::ZnxSubNegateAssign for $be {
            #[inline(always)]
            fn znx_sub_negate_assign(res: &mut [i64], a: &[i64]) {
                <$base as $crate::reference::znx::ZnxSubNegateAssign>::znx_sub_negate_assign(res, a)
            }
        }
        impl $crate::reference::znx::ZnxAutomorphism for $be {
            #[inline(always)]
            fn znx_automorphism(p: i64, res: &mut [i64], a: &[i64]) {
                <$base as $crate::reference::znx::ZnxAutomorphism>::znx_automorphism(p, res, a)
            }
        }
        impl $crate::reference::znx::ZnxAutomorphismRotate for $be {
            #[inline(always)]
            fn znx_automorphism_rotate(p: i64, k: i64, res: &mut [i64], a: &[i64]) {
                <$base as $crate::reference::znx::ZnxAutomorphismRotate>::znx_automorphism_rotate(p, k, res, a)
            }
        }
        impl $crate::reference::znx::ZnxCopy for $be {
            #[inline(always)]
            fn znx_copy(res: &mut [i64], a: &[i64]) {
                <$base as $crate::reference::znx::ZnxCopy>::znx_copy(res, a)
            }
        }
        impl $crate::reference::znx::ZnxNegate for $be {
            #[inline(always)]
            fn znx_negate(res: &mut [i64], src: &[i64]) {
                <$base as $crate::reference::znx::ZnxNegate>::znx_negate(res, src)
            }
        }
        impl $crate::reference::znx::ZnxNegateAssign for $be {
            #[inline(always)]
            fn znx_negate_assign(res: &mut [i64]) {
                <$base as $crate::reference::znx::ZnxNegateAssign>::znx_negate_assign(res)
            }
        }
        impl $crate::reference::znx::ZnxRotate for $be {
            #[inline(always)]
            fn znx_rotate(p: i64, res: &mut [i64], src: &[i64]) {
                <$base as $crate::reference::znx::ZnxRotate>::znx_rotate(p, res, src)
            }
        }
        impl $crate::reference::znx::ZnxZero for $be {
            #[inline(always)]
            fn znx_zero(res: &mut [i64]) {
                <$base as $crate::reference::znx::ZnxZero>::znx_zero(res)
            }
        }
        impl $crate::reference::znx::ZnxMulPowerOfTwo for $be {
            #[inline(always)]
            fn znx_mul_power_of_two(k: i64, res: &mut [i64], a: &[i64]) {
                <$base as $crate::reference::znx::ZnxMulPowerOfTwo>::znx_mul_power_of_two(k, res, a)
            }
        }
        impl $crate::reference::znx::ZnxMulAddPowerOfTwo for $be {
            #[inline(always)]
            fn znx_muladd_power_of_two(k: i64, res: &mut [i64], a: &[i64]) {
                <$base as $crate::reference::znx::ZnxMulAddPowerOfTwo>::znx_muladd_power_of_two(k, res, a)
            }
        }
        impl $crate::reference::znx::ZnxMulPowerOfTwoAssign for $be {
            #[inline(always)]
            fn znx_mul_power_of_two_assign(k: i64, res: &mut [i64]) {
                <$base as $crate::reference::znx::ZnxMulPowerOfTwoAssign>::znx_mul_power_of_two_assign(k, res)
            }
        }
        impl $crate::reference::znx::ZnxSwitchRing for $be {
            #[inline(always)]
            fn znx_switch_ring(res: &mut [i64], a: &[i64]) {
                <$base as $crate::reference::znx::ZnxSwitchRing>::znx_switch_ring(res, a)
            }
        }
        impl $crate::reference::znx::ZnxNormalizeFirstStep for $be {
            #[inline(always)]
            fn znx_normalize_first_step<const OVERWRITE: bool>(
                base2k: usize,
                lsh: usize,
                x: &mut [i64],
                a: &[i64],
                carry: &mut [i64],
            ) {
                <$base as $crate::reference::znx::ZnxNormalizeFirstStep>::znx_normalize_first_step::<OVERWRITE>(
                    base2k, lsh, x, a, carry,
                )
            }
        }
        impl $crate::reference::znx::ZnxNormalizeMiddleStep for $be {
            #[inline(always)]
            fn znx_normalize_middle_step<const OVERWRITE: bool>(
                base2k: usize,
                lsh: usize,
                x: &mut [i64],
                a: &[i64],
                carry: &mut [i64],
            ) {
                <$base as $crate::reference::znx::ZnxNormalizeMiddleStep>::znx_normalize_middle_step::<OVERWRITE>(
                    base2k, lsh, x, a, carry,
                )
            }
        }
        impl $crate::reference::znx::ZnxNormalizeFinalStep for $be {
            #[inline(always)]
            fn znx_normalize_final_step<const OVERWRITE: bool>(
                base2k: usize,
                lsh: usize,
                x: &mut [i64],
                a: &[i64],
                carry: &mut [i64],
            ) {
                <$base as $crate::reference::znx::ZnxNormalizeFinalStep>::znx_normalize_final_step::<OVERWRITE>(
                    base2k, lsh, x, a, carry,
                )
            }
        }
        impl $crate::reference::znx::ZnxNormalizeFirstStepCarryOnly for $be {
            #[inline(always)]
            fn znx_normalize_first_step_carry_only(base2k: usize, lsh: usize, x: &[i64], carry: &mut [i64]) {
                <$base as $crate::reference::znx::ZnxNormalizeFirstStepCarryOnly>::znx_normalize_first_step_carry_only(
                    base2k, lsh, x, carry,
                )
            }
        }
        impl $crate::reference::znx::ZnxNormalizeFirstStepAssign for $be {
            #[inline(always)]
            fn znx_normalize_first_step_assign(base2k: usize, lsh: usize, x: &mut [i64], carry: &mut [i64]) {
                <$base as $crate::reference::znx::ZnxNormalizeFirstStepAssign>::znx_normalize_first_step_assign(
                    base2k, lsh, x, carry,
                )
            }
        }
        impl $crate::reference::znx::ZnxNormalizeMiddleStepCarryOnly for $be {
            #[inline(always)]
            fn znx_normalize_middle_step_carry_only(base2k: usize, lsh: usize, x: &[i64], carry: &mut [i64]) {
                <$base as $crate::reference::znx::ZnxNormalizeMiddleStepCarryOnly>::znx_normalize_middle_step_carry_only(
                    base2k, lsh, x, carry,
                )
            }
        }
        impl $crate::reference::znx::ZnxNormalizeMiddleStepAssign for $be {
            #[inline(always)]
            fn znx_normalize_middle_step_assign(base2k: usize, lsh: usize, x: &mut [i64], carry: &mut [i64]) {
                <$base as $crate::reference::znx::ZnxNormalizeMiddleStepAssign>::znx_normalize_middle_step_assign(
                    base2k, lsh, x, carry,
                )
            }
        }
        impl $crate::reference::znx::ZnxNormalizeFinalStepAssign for $be {
            #[inline(always)]
            fn znx_normalize_final_step_assign(base2k: usize, lsh: usize, x: &mut [i64], carry: &mut [i64]) {
                <$base as $crate::reference::znx::ZnxNormalizeFinalStepAssign>::znx_normalize_final_step_assign(
                    base2k, lsh, x, carry,
                )
            }
        }
        impl $crate::reference::znx::ZnxExtractDigitAddMul for $be {
            #[inline(always)]
            fn znx_extract_digit_addmul(base2k: usize, lsh: usize, res: &mut [i64], src: &mut [i64]) {
                <$base as $crate::reference::znx::ZnxExtractDigitAddMul>::znx_extract_digit_addmul(base2k, lsh, res, src)
            }
        }
        impl $crate::reference::znx::ZnxNormalizeDigit for $be {
            #[inline(always)]
            fn znx_normalize_digit(base2k: usize, res: &mut [i64], src: &mut [i64]) {
                <$base as $crate::reference::znx::ZnxNormalizeDigit>::znx_normalize_digit(base2k, res, src)
            }
        }
        impl $crate::reference::normalization::I64NormalizeOps for $be {
            #[inline(always)]
            fn znx_normalize_floor<const CARRY_IN: bool, const ROUND: bool>(
                base2k: usize,
                lsh: usize,
                a: &[i64],
                carry: &mut [i64],
            ) {
                <$base as $crate::reference::normalization::I64NormalizeOps>::znx_normalize_floor::<CARRY_IN, ROUND>(
                    base2k, lsh, a, carry,
                )
            }
            #[inline(always)]
            fn znx_normalize_round<const CARRY_IN: bool, const PAD: bool>(
                base2k: usize,
                lsh: usize,
                padding: usize,
                res: &mut [i64],
                a: &[i64],
                carry: &mut [i64],
            ) {
                <$base as $crate::reference::normalization::I64NormalizeOps>::znx_normalize_round::<CARRY_IN, PAD>(
                    base2k, lsh, padding, res, a, carry,
                )
            }
            #[inline(always)]
            fn znx_normalize_round_assign<const CARRY_IN: bool>(
                base2k: usize,
                lsh: usize,
                padding: usize,
                res: &mut [i64],
                carry: &mut [i64],
            ) {
                <$base as $crate::reference::normalization::I64NormalizeOps>::znx_normalize_round_assign::<CARRY_IN>(
                    base2k, lsh, padding, res, carry,
                )
            }
            #[inline(always)]
            fn znx_extract_digit_mul(base2k: usize, lsh: usize, res: &mut [i64], src: &mut [i64]) {
                <$base as $crate::reference::normalization::I64NormalizeOps>::znx_extract_digit_mul(base2k, lsh, res, src)
            }
            #[inline(always)]
            fn znx_extract_digit_addmul_normalize<const OVERWRITE: bool>(
                base2k: usize,
                lsh: usize,
                res_base2k: usize,
                res: &mut [i64],
                src: &mut [i64],
                carry: &mut [i64],
            ) {
                <$base as $crate::reference::normalization::I64NormalizeOps>::znx_extract_digit_addmul_normalize::<OVERWRITE>(
                    base2k, lsh, res_base2k, res, src, carry,
                )
            }
        }
    };
}

/// Forwards the FFT64 kernels (reim execution and arithmetic, reim4 blocks and
/// convolutions, `i64` block ops, big-word Hadamard product) to `$base`.
#[macro_export]
macro_rules! forward_fft64_kernels {
    ($be:ty => $base:ty) => {
        impl $crate::reference::fft64::reim::ReimFFTExecute<$crate::reference::fft64::reim::ReimFFTTable<f64>, f64> for $be {
            #[inline(always)]
            fn reim_dft_execute(table: &$crate::reference::fft64::reim::ReimFFTTable<f64>, data: &mut [f64]) {
                <$base as $crate::reference::fft64::reim::ReimFFTExecute<
                    $crate::reference::fft64::reim::ReimFFTTable<f64>,
                    f64,
                >>::reim_dft_execute(table, data)
            }
        }
        impl $crate::reference::fft64::reim::ReimFFTExecute<$crate::reference::fft64::reim::ReimIFFTTable<f64>, f64> for $be {
            #[inline(always)]
            fn reim_dft_execute(table: &$crate::reference::fft64::reim::ReimIFFTTable<f64>, data: &mut [f64]) {
                <$base as $crate::reference::fft64::reim::ReimFFTExecute<
                    $crate::reference::fft64::reim::ReimIFFTTable<f64>,
                    f64,
                >>::reim_dft_execute(table, data)
            }
        }
        impl $crate::reference::fft64::reim::ReimArith for $be {
            #[inline(always)]
            fn reim_from_znx(res: &mut [f64], a: &[i64]) {
                <$base as $crate::reference::fft64::reim::ReimArith>::reim_from_znx(res, a)
            }
            #[inline(always)]
            fn reim_to_znx(res: &mut [i64], divisor: f64, a: &[f64]) {
                <$base as $crate::reference::fft64::reim::ReimArith>::reim_to_znx(res, divisor, a)
            }
            #[inline(always)]
            fn reim_to_znx_assign(res: &mut [f64], divisor: f64) {
                <$base as $crate::reference::fft64::reim::ReimArith>::reim_to_znx_assign(res, divisor)
            }
            #[inline(always)]
            fn reim_add(res: &mut [f64], a: &[f64], b: &[f64]) {
                <$base as $crate::reference::fft64::reim::ReimArith>::reim_add(res, a, b)
            }
            #[inline(always)]
            fn reim_add_assign(res: &mut [f64], a: &[f64]) {
                <$base as $crate::reference::fft64::reim::ReimArith>::reim_add_assign(res, a)
            }
            #[inline(always)]
            fn reim_sub(res: &mut [f64], a: &[f64], b: &[f64]) {
                <$base as $crate::reference::fft64::reim::ReimArith>::reim_sub(res, a, b)
            }
            #[inline(always)]
            fn reim_sub_assign(res: &mut [f64], a: &[f64]) {
                <$base as $crate::reference::fft64::reim::ReimArith>::reim_sub_assign(res, a)
            }
            #[inline(always)]
            fn reim_sub_negate_assign(res: &mut [f64], a: &[f64]) {
                <$base as $crate::reference::fft64::reim::ReimArith>::reim_sub_negate_assign(res, a)
            }
            #[inline(always)]
            fn reim_negate(res: &mut [f64], a: &[f64]) {
                <$base as $crate::reference::fft64::reim::ReimArith>::reim_negate(res, a)
            }
            #[inline(always)]
            fn reim_negate_assign(res: &mut [f64]) {
                <$base as $crate::reference::fft64::reim::ReimArith>::reim_negate_assign(res)
            }
            #[inline(always)]
            fn reim_mul(res: &mut [f64], a: &[f64], b: &[f64]) {
                <$base as $crate::reference::fft64::reim::ReimArith>::reim_mul(res, a, b)
            }
            #[inline(always)]
            fn reim_mul_assign(res: &mut [f64], a: &[f64]) {
                <$base as $crate::reference::fft64::reim::ReimArith>::reim_mul_assign(res, a)
            }
            #[inline(always)]
            fn reim_real_mul(res: &mut [f64], a: &[f64], b: &[f64]) {
                <$base as $crate::reference::fft64::reim::ReimArith>::reim_real_mul(res, a, b)
            }
            #[inline(always)]
            fn reim_real_mul_assign(res: &mut [f64], a: &[f64]) {
                <$base as $crate::reference::fft64::reim::ReimArith>::reim_real_mul_assign(res, a)
            }
            #[inline(always)]
            fn reim_real_addmul(res: &mut [f64], a: &[f64], b: &[f64]) {
                <$base as $crate::reference::fft64::reim::ReimArith>::reim_real_addmul(res, a, b)
            }
            #[inline(always)]
            fn reim_addmul(res: &mut [f64], a: &[f64], b: &[f64]) {
                <$base as $crate::reference::fft64::reim::ReimArith>::reim_addmul(res, a, b)
            }
            #[inline(always)]
            fn reim_copy(res: &mut [f64], a: &[f64]) {
                <$base as $crate::reference::fft64::reim::ReimArith>::reim_copy(res, a)
            }
            #[inline(always)]
            fn reim_zero(res: &mut [f64]) {
                <$base as $crate::reference::fft64::reim::ReimArith>::reim_zero(res)
            }
        }
        impl $crate::reference::fft64::reim4::Reim4BlkMatVec for $be {
            #[inline(always)]
            fn reim4_extract_1blk_contiguous(m: usize, rows: usize, blk: usize, dst: &mut [f64], src: &[f64]) {
                <$base as $crate::reference::fft64::reim4::Reim4BlkMatVec>::reim4_extract_1blk_contiguous(m, rows, blk, dst, src)
            }
            #[inline(always)]
            fn reim4_save_1blk_contiguous(m: usize, rows: usize, blk: usize, dst: &mut [f64], src: &[f64]) {
                <$base as $crate::reference::fft64::reim4::Reim4BlkMatVec>::reim4_save_1blk_contiguous(m, rows, blk, dst, src)
            }
            #[inline(always)]
            fn reim4_save_1blk<const OVERWRITE: bool>(m: usize, blk: usize, dst: &mut [f64], src: &[f64]) {
                <$base as $crate::reference::fft64::reim4::Reim4BlkMatVec>::reim4_save_1blk::<OVERWRITE>(m, blk, dst, src)
            }
            #[inline(always)]
            fn reim4_save_2blks<const OVERWRITE: bool>(m: usize, blk: usize, dst: &mut [f64], src: &[f64]) {
                <$base as $crate::reference::fft64::reim4::Reim4BlkMatVec>::reim4_save_2blks::<OVERWRITE>(m, blk, dst, src)
            }
            #[inline(always)]
            fn reim4_mat1col_prod(nrows: usize, dst: &mut [f64], u: &[f64], v: &[f64]) {
                <$base as $crate::reference::fft64::reim4::Reim4BlkMatVec>::reim4_mat1col_prod(nrows, dst, u, v)
            }
            #[inline(always)]
            fn reim4_mat2cols_prod(nrows: usize, dst: &mut [f64], u: &[f64], v: &[f64]) {
                <$base as $crate::reference::fft64::reim4::Reim4BlkMatVec>::reim4_mat2cols_prod(nrows, dst, u, v)
            }
            #[inline(always)]
            fn reim4_mat2cols_2ndcol_prod(nrows: usize, dst: &mut [f64], u: &[f64], v: &[f64]) {
                <$base as $crate::reference::fft64::reim4::Reim4BlkMatVec>::reim4_mat2cols_2ndcol_prod(nrows, dst, u, v)
            }
            #[inline(always)]
            fn reim4_real_mat1col_prod(nrows: usize, dst: &mut [f64], u: &[f64], v: &[f64]) {
                <$base as $crate::reference::fft64::reim4::Reim4BlkMatVec>::reim4_real_mat1col_prod(nrows, dst, u, v)
            }
            #[inline(always)]
            fn reim4_real_mat2cols_prod(nrows: usize, dst: &mut [f64], u: &[f64], v: &[f64]) {
                <$base as $crate::reference::fft64::reim4::Reim4BlkMatVec>::reim4_real_mat2cols_prod(nrows, dst, u, v)
            }
            #[inline(always)]
            fn reim4_real_mat2cols_2ndcol_prod(nrows: usize, dst: &mut [f64], u: &[f64], v: &[f64]) {
                <$base as $crate::reference::fft64::reim4::Reim4BlkMatVec>::reim4_real_mat2cols_2ndcol_prod(nrows, dst, u, v)
            }
        }
        impl $crate::reference::fft64::reim4::Reim4Convolution for $be {
            #[inline(always)]
            fn reim4_convolution_1coeff(k: usize, dst: &mut [f64; 8], a: &[f64], a_size: usize, b: &[f64], b_size: usize) {
                <$base as $crate::reference::fft64::reim4::Reim4Convolution>::reim4_convolution_1coeff(
                    k, dst, a, a_size, b, b_size,
                )
            }
            #[inline(always)]
            fn reim4_convolution_2coeffs(k: usize, dst: &mut [f64; 16], a: &[f64], a_size: usize, b: &[f64], b_size: usize) {
                <$base as $crate::reference::fft64::reim4::Reim4Convolution>::reim4_convolution_2coeffs(
                    k, dst, a, a_size, b, b_size,
                )
            }
            #[inline(always)]
            fn reim4_real_convolution_1coeff(k: usize, dst: &mut [f64; 8], a: &[f64], a_size: usize, b: &[f64], b_size: usize) {
                <$base as $crate::reference::fft64::reim4::Reim4Convolution>::reim4_real_convolution_1coeff(
                    k, dst, a, a_size, b, b_size,
                )
            }
            #[inline(always)]
            fn reim4_real_convolution_2coeffs(k: usize, dst: &mut [f64; 16], a: &[f64], a_size: usize, b: &[f64], b_size: usize) {
                <$base as $crate::reference::fft64::reim4::Reim4Convolution>::reim4_real_convolution_2coeffs(
                    k, dst, a, a_size, b, b_size,
                )
            }
            #[inline(always)]
            fn reim4_convolution(
                dst: &mut [f64],
                dst_size: usize,
                offset: usize,
                a: &[f64],
                a_size: usize,
                b: &[f64],
                b_size: usize,
            ) {
                <$base as $crate::reference::fft64::reim4::Reim4Convolution>::reim4_convolution(
                    dst, dst_size, offset, a, a_size, b, b_size,
                )
            }
            #[inline(always)]
            fn reim4_real_convolution(
                dst: &mut [f64],
                dst_size: usize,
                offset: usize,
                a: &[f64],
                a_size: usize,
                b: &[f64],
                b_size: usize,
            ) {
                <$base as $crate::reference::fft64::reim4::Reim4Convolution>::reim4_real_convolution(
                    dst, dst_size, offset, a, a_size, b, b_size,
                )
            }
            #[inline(always)]
            fn reim4_convolution_apply(
                m: usize,
                min_size: usize,
                offset: usize,
                dst: &mut [f64],
                dst_stride: usize,
                a: &[f64],
                a_size: usize,
                b: &[f64],
                b_size: usize,
                b_log_gap: usize,
                tmp: &mut [f64],
            ) where
                Self: $crate::reference::fft64::reim4::Reim4BlkMatVec + Sized,
            {
                <$base as $crate::reference::fft64::reim4::Reim4Convolution>::reim4_convolution_apply(
                    m, min_size, offset, dst, dst_stride, a, a_size, b, b_size, b_log_gap, tmp,
                )
            }
            #[inline(always)]
            fn reim4_real_convolution_apply(
                m: usize,
                min_size: usize,
                offset: usize,
                dst: &mut [f64],
                dst_stride: usize,
                a: &[f64],
                a_size: usize,
                b: &[f64],
                b_size: usize,
                b_log_gap: usize,
                tmp: &mut [f64],
            ) where
                Self: $crate::reference::fft64::reim4::Reim4BlkMatVec + Sized,
            {
                <$base as $crate::reference::fft64::reim4::Reim4Convolution>::reim4_real_convolution_apply(
                    m, min_size, offset, dst, dst_stride, a, a_size, b, b_size, b_log_gap, tmp,
                )
            }
            #[inline(always)]
            fn reim4_convolution_apply_accumulate(
                m: usize,
                min_size: usize,
                offset: usize,
                dst: &mut [f64],
                dst_stride: usize,
                a: &[f64],
                a_size: usize,
                b: &[f64],
                b_size: usize,
                b_log_gap: usize,
                tmp: &mut [f64],
            ) where
                Self: $crate::reference::fft64::reim4::Reim4BlkMatVec + Sized,
            {
                <$base as $crate::reference::fft64::reim4::Reim4Convolution>::reim4_convolution_apply_accumulate(
                    m, min_size, offset, dst, dst_stride, a, a_size, b, b_size, b_log_gap, tmp,
                )
            }
            #[inline(always)]
            fn reim4_real_convolution_apply_accumulate(
                m: usize,
                min_size: usize,
                offset: usize,
                dst: &mut [f64],
                dst_stride: usize,
                a: &[f64],
                a_size: usize,
                b: &[f64],
                b_size: usize,
                b_log_gap: usize,
                tmp: &mut [f64],
            ) where
                Self: $crate::reference::fft64::reim4::Reim4BlkMatVec + Sized,
            {
                <$base as $crate::reference::fft64::reim4::Reim4Convolution>::reim4_real_convolution_apply_accumulate(
                    m, min_size, offset, dst, dst_stride, a, a_size, b, b_size, b_log_gap, tmp,
                )
            }
            #[inline(always)]
            fn reim4_convolution_pairwise_apply(
                m: usize,
                min_size: usize,
                offset: usize,
                dst: &mut [f64],
                dst_stride: usize,
                a0: &[f64],
                a1: &[f64],
                a_size: usize,
                b0: &[f64],
                b1: &[f64],
                b_size: usize,
                b_log_gap: usize,
                tmp: &mut [f64],
            ) where
                Self: $crate::reference::fft64::reim4::Reim4BlkMatVec + $crate::reference::fft64::reim::ReimArith + Sized,
            {
                <$base as $crate::reference::fft64::reim4::Reim4Convolution>::reim4_convolution_pairwise_apply(
                    m, min_size, offset, dst, dst_stride, a0, a1, a_size, b0, b1, b_size, b_log_gap, tmp,
                )
            }
            #[inline(always)]
            fn reim4_real_convolution_pairwise_apply(
                m: usize,
                min_size: usize,
                offset: usize,
                dst: &mut [f64],
                dst_stride: usize,
                a0: &[f64],
                a1: &[f64],
                a_size: usize,
                b0: &[f64],
                b1: &[f64],
                b_size: usize,
                b_log_gap: usize,
                tmp: &mut [f64],
            ) where
                Self: $crate::reference::fft64::reim4::Reim4BlkMatVec + $crate::reference::fft64::reim::ReimArith + Sized,
            {
                <$base as $crate::reference::fft64::reim4::Reim4Convolution>::reim4_real_convolution_pairwise_apply(
                    m, min_size, offset, dst, dst_stride, a0, a1, a_size, b0, b1, b_size, b_log_gap, tmp,
                )
            }
            #[inline(always)]
            fn reim4_convolution_by_real_const_1coeff(k: usize, dst: &mut [f64; 8], a: &[f64], a_size: usize, b: &[f64]) {
                <$base as $crate::reference::fft64::reim4::Reim4Convolution>::reim4_convolution_by_real_const_1coeff(
                    k, dst, a, a_size, b,
                )
            }
            #[inline(always)]
            fn reim4_convolution_by_real_const_2coeffs(k: usize, dst: &mut [f64; 16], a: &[f64], a_size: usize, b: &[f64]) {
                <$base as $crate::reference::fft64::reim4::Reim4Convolution>::reim4_convolution_by_real_const_2coeffs(
                    k, dst, a, a_size, b,
                )
            }
            #[inline(always)]
            fn reim4_convolution_by_real_const(
                dst: &mut [f64],
                dst_size: usize,
                offset: usize,
                a: &[f64],
                a_size: usize,
                b: &[f64],
            ) {
                <$base as $crate::reference::fft64::reim4::Reim4Convolution>::reim4_convolution_by_real_const(
                    dst, dst_size, offset, a, a_size, b,
                )
            }
        }
        impl $crate::reference::fft64::convolution::I64Ops for $be {
            #[inline(always)]
            fn i64_hadamard_product(res: &mut [i64], a: &[i64], b: &[i64]) {
                <$base as $crate::reference::fft64::convolution::I64Ops>::i64_hadamard_product(res, a, b)
            }
            #[inline(always)]
            fn i64_extract_1blk_contiguous(n: usize, offset: usize, rows: usize, blk: usize, dst: &mut [i64], src: &[i64]) {
                <$base as $crate::reference::fft64::convolution::I64Ops>::i64_extract_1blk_contiguous(
                    n, offset, rows, blk, dst, src,
                )
            }
            #[inline(always)]
            fn i64_save_1blk_contiguous(n: usize, offset: usize, rows: usize, blk: usize, dst: &mut [i64], src: &[i64]) {
                <$base as $crate::reference::fft64::convolution::I64Ops>::i64_save_1blk_contiguous(n, offset, rows, blk, dst, src)
            }
            #[inline(always)]
            fn i64_convolution_by_const_1coeff(k: usize, dst: &mut [i64; 8], a: &[i64], a_size: usize, b: &[i64]) {
                <$base as $crate::reference::fft64::convolution::I64Ops>::i64_convolution_by_const_1coeff(k, dst, a, a_size, b)
            }
            #[inline(always)]
            fn i64_convolution_by_const_2coeffs(k: usize, dst: &mut [i64; 16], a: &[i64], a_size: usize, b: &[i64]) {
                <$base as $crate::reference::fft64::convolution::I64Ops>::i64_convolution_by_const_2coeffs(k, dst, a, a_size, b)
            }
            #[inline(always)]
            fn i64_convolution_by_const(dst: &mut [i64], dst_size: usize, offset: usize, a: &[i64], a_size: usize, b: &[i64]) {
                <$base as $crate::reference::fft64::convolution::I64Ops>::i64_convolution_by_const(
                    dst, dst_size, offset, a, a_size, b,
                )
            }
        }
        impl $crate::hal_defaults::BigWordHadamardProduct for $be {
            #[inline(always)]
            fn big_word_hadamard_product(res: &mut [<$be as $crate::layouts::Backend>::BigWord], a: &[i64], b: &[i64]) {
                <$base as $crate::hal_defaults::BigWordHadamardProduct>::big_word_hadamard_product(res, a, b)
            }
        }
    };
}

/// Forwards the ring- and prime-independent NTT-domain arithmetic (add, sub,
/// negate, zero, copy) to `$base`.
#[macro_export]
macro_rules! forward_ntt_arith_kernels {
    ($be:ty => $base:ty) => {
        impl $crate::reference::ntt4x30::NttAdd for $be {
            #[inline(always)]
            fn ntt_add(res: &mut [u64], a: &[u64], b: &[u64]) {
                <$base as $crate::reference::ntt4x30::NttAdd>::ntt_add(res, a, b)
            }
        }
        impl $crate::reference::ntt4x30::NttAddAssign for $be {
            #[inline(always)]
            fn ntt_add_assign(res: &mut [u64], a: &[u64]) {
                <$base as $crate::reference::ntt4x30::NttAddAssign>::ntt_add_assign(res, a)
            }
        }
        impl $crate::reference::ntt4x30::NttSub for $be {
            #[inline(always)]
            fn ntt_sub(res: &mut [u64], a: &[u64], b: &[u64]) {
                <$base as $crate::reference::ntt4x30::NttSub>::ntt_sub(res, a, b)
            }
        }
        impl $crate::reference::ntt4x30::NttSubAssign for $be {
            #[inline(always)]
            fn ntt_sub_assign(res: &mut [u64], a: &[u64]) {
                <$base as $crate::reference::ntt4x30::NttSubAssign>::ntt_sub_assign(res, a)
            }
        }
        impl $crate::reference::ntt4x30::NttSubNegateAssign for $be {
            #[inline(always)]
            fn ntt_sub_negate_assign(res: &mut [u64], a: &[u64]) {
                <$base as $crate::reference::ntt4x30::NttSubNegateAssign>::ntt_sub_negate_assign(res, a)
            }
        }
        impl $crate::reference::ntt4x30::NttNegate for $be {
            #[inline(always)]
            fn ntt_negate(res: &mut [u64], a: &[u64]) {
                <$base as $crate::reference::ntt4x30::NttNegate>::ntt_negate(res, a)
            }
        }
        impl $crate::reference::ntt4x30::NttNegateAssign for $be {
            #[inline(always)]
            fn ntt_negate_assign(res: &mut [u64]) {
                <$base as $crate::reference::ntt4x30::NttNegateAssign>::ntt_negate_assign(res)
            }
        }
        impl $crate::reference::ntt4x30::NttZero for $be {
            #[inline(always)]
            fn ntt_zero(res: &mut [u64]) {
                <$base as $crate::reference::ntt4x30::NttZero>::ntt_zero(res)
            }
        }
        impl $crate::reference::ntt4x30::NttCopy for $be {
            #[inline(always)]
            fn ntt_copy(res: &mut [u64], a: &[u64]) {
                <$base as $crate::reference::ntt4x30::NttCopy>::ntt_copy(res, a)
            }
        }
    };
}

/// Forwards the Primes30 NTT kernels (domain conversion, multiply-accumulate,
/// packing) to `$base`. `NttDFTExecute` is not forwarded: the transform
/// depends on the ring.
#[macro_export]
macro_rules! forward_ntt4x30_kernels {
    ($be:ty => $base:ty) => {
        impl $crate::reference::ntt4x30::NttFromZnx64 for $be {
            #[inline(always)]
            fn ntt_from_znx64(res: &mut [u64], a: &[i64]) {
                <$base as $crate::reference::ntt4x30::NttFromZnx64>::ntt_from_znx64(res, a)
            }
        }
        impl $crate::reference::ntt4x30::NttToZnx128 for $be {
            #[inline(always)]
            fn ntt_to_znx128(res: &mut [i128], divisor_is_n: usize, a: &[u64]) {
                <$base as $crate::reference::ntt4x30::NttToZnx128>::ntt_to_znx128(res, divisor_is_n, a)
            }
        }
        impl $crate::reference::ntt4x30::NttMulBbb for $be {
            #[inline(always)]
            fn ntt_mul_bbb(
                meta: &$crate::reference::ntt4x30::mat_vec::BbbMeta<$crate::reference::ntt4x30::primes::Primes30>,
                ell: usize,
                res: &mut [u64],
                a: &[u64],
                b: &[u64],
            ) {
                <$base as $crate::reference::ntt4x30::NttMulBbb>::ntt_mul_bbb(meta, ell, res, a, b)
            }
        }
        impl $crate::reference::ntt4x30::NttMulBbc for $be {
            #[inline(always)]
            fn ntt_mul_bbc(
                meta: &$crate::reference::ntt4x30::mat_vec::BbcMeta<$crate::reference::ntt4x30::primes::Primes30>,
                ell: usize,
                res: &mut [u64],
                ntt_coeff: &[u32],
                prepared: &[u32],
            ) {
                <$base as $crate::reference::ntt4x30::NttMulBbc>::ntt_mul_bbc(meta, ell, res, ntt_coeff, prepared)
            }
        }
        impl $crate::reference::ntt4x30::NttCFromB for $be {
            #[inline(always)]
            fn ntt_c_from_b(n: usize, res: &mut [u32], a: &[u64]) {
                <$base as $crate::reference::ntt4x30::NttCFromB>::ntt_c_from_b(n, res, a)
            }
        }
        impl $crate::reference::ntt4x30::NttMulBbc1ColX2 for $be {
            #[inline(always)]
            fn ntt_mul_bbc_1col_x2(
                meta: &$crate::reference::ntt4x30::mat_vec::BbcMeta<$crate::reference::ntt4x30::primes::Primes30>,
                ell: usize,
                res: &mut [u64],
                a: &[u32],
                b: &[u32],
            ) {
                <$base as $crate::reference::ntt4x30::NttMulBbc1ColX2>::ntt_mul_bbc_1col_x2(meta, ell, res, a, b)
            }
            #[inline(always)]
            fn ntt_mul_bbc_tile4_x2(
                meta: &$crate::reference::ntt4x30::mat_vec::BbcMeta<$crate::reference::ntt4x30::primes::Primes30>,
                len: usize,
                res: &mut [u64],
                a: &[u32],
                b: &[u32],
            ) {
                <$base as $crate::reference::ntt4x30::NttMulBbc1ColX2>::ntt_mul_bbc_tile4_x2(meta, len, res, a, b)
            }
        }
        impl $crate::reference::ntt4x30::NttMulBbc2ColsX2 for $be {
            #[inline(always)]
            fn ntt_mul_bbc_2cols_x2(
                meta: &$crate::reference::ntt4x30::mat_vec::BbcMeta<$crate::reference::ntt4x30::primes::Primes30>,
                ell: usize,
                res: &mut [u64],
                a: &[u32],
                b: &[u32],
            ) {
                <$base as $crate::reference::ntt4x30::NttMulBbc2ColsX2>::ntt_mul_bbc_2cols_x2(meta, ell, res, a, b)
            }
        }
        impl $crate::reference::ntt4x30::NttExtract1BlkContiguous for $be {
            #[inline(always)]
            fn ntt_extract_1blk_contiguous(n: usize, row_max: usize, blk: usize, dst: &mut [u64], src: &[u64]) {
                <$base as $crate::reference::ntt4x30::NttExtract1BlkContiguous>::ntt_extract_1blk_contiguous(
                    n, row_max, blk, dst, src,
                )
            }
        }
        impl $crate::reference::ntt4x30::NttPackLeft1BlkX2 for $be {
            #[inline(always)]
            fn ntt_pack_left_1blk_x2(dst: &mut [u32], a: &[u64], row_count: usize, row_stride: usize, blk: usize) {
                <$base as $crate::reference::ntt4x30::NttPackLeft1BlkX2>::ntt_pack_left_1blk_x2(
                    dst, a, row_count, row_stride, blk,
                )
            }
        }
        impl $crate::reference::ntt4x30::NttPackRight1BlkX2 for $be {
            #[inline(always)]
            fn ntt_pack_right_1blk_x2(dst: &mut [u32], a: &[u32], row_count: usize, row_stride: usize, blk: usize) {
                <$base as $crate::reference::ntt4x30::NttPackRight1BlkX2>::ntt_pack_right_1blk_x2(
                    dst, a, row_count, row_stride, blk,
                )
            }
        }
        impl $crate::reference::ntt4x30::NttPairwisePackLeft1BlkX2 for $be {
            #[inline(always)]
            fn ntt_pairwise_pack_left_1blk_x2(
                dst: &mut [u32],
                a: &[u64],
                b: &[u64],
                row_count: usize,
                row_stride: usize,
                blk: usize,
            ) {
                <$base as $crate::reference::ntt4x30::NttPairwisePackLeft1BlkX2>::ntt_pairwise_pack_left_1blk_x2(
                    dst, a, b, row_count, row_stride, blk,
                )
            }
        }
        impl $crate::reference::ntt4x30::NttPairwisePackRight1BlkX2 for $be {
            #[inline(always)]
            fn ntt_pairwise_pack_right_1blk_x2(
                dst: &mut [u32],
                a: &[u32],
                b: &[u32],
                row_count: usize,
                row_stride: usize,
                blk: usize,
            ) {
                <$base as $crate::reference::ntt4x30::NttPairwisePackRight1BlkX2>::ntt_pairwise_pack_right_1blk_x2(
                    dst, a, b, row_count, row_stride, blk,
                )
            }
        }
    };
}

/// Forwards the `i128` big-coefficient kernels and the big-word Hadamard
/// product to `$base`.
#[macro_export]
macro_rules! forward_i128_kernels {
    ($be:ty => $base:ty) => {
        impl $crate::reference::ntt4x30::vec_znx_big::I128BigOps for $be {
            #[inline(always)]
            fn i128_hadamard_product_i64(res: &mut [i128], a: &[i64], b: &[i64]) {
                <$base as $crate::reference::ntt4x30::vec_znx_big::I128BigOps>::i128_hadamard_product_i64(res, a, b)
            }
            #[inline(always)]
            fn i128_add(res: &mut [i128], a: &[i128], b: &[i128]) {
                <$base as $crate::reference::ntt4x30::vec_znx_big::I128BigOps>::i128_add(res, a, b)
            }
            #[inline(always)]
            fn i128_add_assign(res: &mut [i128], a: &[i128]) {
                <$base as $crate::reference::ntt4x30::vec_znx_big::I128BigOps>::i128_add_assign(res, a)
            }
            #[inline(always)]
            fn i128_add_small(res: &mut [i128], a: &[i128], b: &[i64]) {
                <$base as $crate::reference::ntt4x30::vec_znx_big::I128BigOps>::i128_add_small(res, a, b)
            }
            #[inline(always)]
            fn i128_add_small_assign(res: &mut [i128], a: &[i64]) {
                <$base as $crate::reference::ntt4x30::vec_znx_big::I128BigOps>::i128_add_small_assign(res, a)
            }
            #[inline(always)]
            fn i128_sub(res: &mut [i128], a: &[i128], b: &[i128]) {
                <$base as $crate::reference::ntt4x30::vec_znx_big::I128BigOps>::i128_sub(res, a, b)
            }
            #[inline(always)]
            fn i128_sub_assign(res: &mut [i128], a: &[i128]) {
                <$base as $crate::reference::ntt4x30::vec_znx_big::I128BigOps>::i128_sub_assign(res, a)
            }
            #[inline(always)]
            fn i128_sub_negate_assign(res: &mut [i128], a: &[i128]) {
                <$base as $crate::reference::ntt4x30::vec_znx_big::I128BigOps>::i128_sub_negate_assign(res, a)
            }
            #[inline(always)]
            fn i128_sub_small_a(res: &mut [i128], a: &[i64], b: &[i128]) {
                <$base as $crate::reference::ntt4x30::vec_znx_big::I128BigOps>::i128_sub_small_a(res, a, b)
            }
            #[inline(always)]
            fn i128_sub_small_b(res: &mut [i128], a: &[i128], b: &[i64]) {
                <$base as $crate::reference::ntt4x30::vec_znx_big::I128BigOps>::i128_sub_small_b(res, a, b)
            }
            #[inline(always)]
            fn i128_sub_small_assign(res: &mut [i128], a: &[i64]) {
                <$base as $crate::reference::ntt4x30::vec_znx_big::I128BigOps>::i128_sub_small_assign(res, a)
            }
            #[inline(always)]
            fn i128_sub_small_negate_assign(res: &mut [i128], a: &[i64]) {
                <$base as $crate::reference::ntt4x30::vec_znx_big::I128BigOps>::i128_sub_small_negate_assign(res, a)
            }
            #[inline(always)]
            fn i128_negate(res: &mut [i128], a: &[i128]) {
                <$base as $crate::reference::ntt4x30::vec_znx_big::I128BigOps>::i128_negate(res, a)
            }
            #[inline(always)]
            fn i128_negate_assign(res: &mut [i128]) {
                <$base as $crate::reference::ntt4x30::vec_znx_big::I128BigOps>::i128_negate_assign(res)
            }
            #[inline(always)]
            fn i128_neg_from_small(res: &mut [i128], a: &[i64]) {
                <$base as $crate::reference::ntt4x30::vec_znx_big::I128BigOps>::i128_neg_from_small(res, a)
            }
            #[inline(always)]
            fn i128_from_small(res: &mut [i128], a: &[i64]) {
                <$base as $crate::reference::ntt4x30::vec_znx_big::I128BigOps>::i128_from_small(res, a)
            }
        }
        impl $crate::reference::ntt4x30::vec_znx_big::I128NormalizeOps for $be {
            #[inline(always)]
            fn nfc_add_small_carry(carry: &mut [i128], a: &[i64]) {
                <$base as $crate::reference::ntt4x30::vec_znx_big::I128NormalizeOps>::nfc_add_small_carry(carry, a)
            }
            #[inline(always)]
            fn znx_extract_digit_addmul_i128(base2k: usize, lsh: usize, res: &mut [i64], src: &mut [i128]) {
                <$base as $crate::reference::ntt4x30::vec_znx_big::I128NormalizeOps>::znx_extract_digit_addmul_i128(
                    base2k, lsh, res, src,
                )
            }
            #[inline(always)]
            fn nfc_normalize_floor<const CARRY_IN: bool, const ROUND: bool>(
                base2k: usize,
                lsh: usize,
                a: &[i128],
                carry: &mut [i128],
            ) {
                <$base as $crate::reference::ntt4x30::vec_znx_big::I128NormalizeOps>::nfc_normalize_floor::<CARRY_IN, ROUND>(
                    base2k, lsh, a, carry,
                )
            }
            #[inline(always)]
            fn nfc_normalize_round<const CARRY_IN: bool, const PAD: bool>(
                base2k: usize,
                lsh: usize,
                padding: usize,
                res: &mut [i64],
                a: &[i128],
                carry: &mut [i128],
            ) {
                <$base as $crate::reference::ntt4x30::vec_znx_big::I128NormalizeOps>::nfc_normalize_round::<CARRY_IN, PAD>(
                    base2k, lsh, padding, res, a, carry,
                )
            }
            const FUSE_NORMALIZE: bool = <$base as $crate::reference::ntt4x30::vec_znx_big::I128NormalizeOps>::FUSE_NORMALIZE;
            #[inline(always)]
            fn znx_extract_digit_mul_i128(base2k: usize, lsh: usize, res: &mut [i64], src: &mut [i128]) {
                <$base as $crate::reference::ntt4x30::vec_znx_big::I128NormalizeOps>::znx_extract_digit_mul_i128(
                    base2k, lsh, res, src,
                )
            }
            #[inline(always)]
            fn znx_extract_digit_addmul_normalize_i128<const OVERWRITE: bool>(
                base2k: usize,
                lsh: usize,
                res_base2k: usize,
                res: &mut [i64],
                src: &mut [i128],
                carry: &mut [i128],
            ) {
                <$base as $crate::reference::ntt4x30::vec_znx_big::I128NormalizeOps>::znx_extract_digit_addmul_normalize_i128::<
                    OVERWRITE,
                >(base2k, lsh, res_base2k, res, src, carry)
            }
            #[inline(always)]
            fn nfc_middle_step(base2k: usize, lsh: usize, res: &mut [i64], a: &[i128], carry: &mut [i128]) {
                <$base as $crate::reference::ntt4x30::vec_znx_big::I128NormalizeOps>::nfc_middle_step(base2k, lsh, res, a, carry)
            }
            #[inline(always)]
            fn nfc_middle_step_into<O: $crate::reference::ntt4x30::vec_znx_big::AssignOp>(
                base2k: usize,
                lsh: usize,
                res: &mut [i64],
                a: &[i128],
                carry: &mut [i128],
            ) {
                <$base as $crate::reference::ntt4x30::vec_znx_big::I128NormalizeOps>::nfc_middle_step_into::<O>(
                    base2k, lsh, res, a, carry,
                )
            }
            #[inline(always)]
            fn nfc_middle_step_assign(base2k: usize, lsh: usize, res: &mut [i64], carry: &mut [i128]) {
                <$base as $crate::reference::ntt4x30::vec_znx_big::I128NormalizeOps>::nfc_middle_step_assign(
                    base2k, lsh, res, carry,
                )
            }
            #[inline(always)]
            fn nfc_final_step_assign(base2k: usize, lsh: usize, res: &mut [i64], carry: &mut [i128]) {
                <$base as $crate::reference::ntt4x30::vec_znx_big::I128NormalizeOps>::nfc_final_step_assign(
                    base2k, lsh, res, carry,
                )
            }
            #[inline(always)]
            fn nfc_final_step_into<O: $crate::reference::ntt4x30::vec_znx_big::AssignOp>(
                base2k: usize,
                lsh: usize,
                res: &mut [i64],
                carry: &mut [i128],
            ) {
                <$base as $crate::reference::ntt4x30::vec_znx_big::I128NormalizeOps>::nfc_final_step_into::<O>(
                    base2k, lsh, res, carry,
                )
            }
        }
        impl $crate::hal_defaults::BigWordHadamardProduct for $be {
            #[inline(always)]
            fn big_word_hadamard_product(res: &mut [<$be as $crate::layouts::Backend>::BigWord], a: &[i64], b: &[i64]) {
                <$base as $crate::hal_defaults::BigWordHadamardProduct>::big_word_hadamard_product(res, a, b)
            }
        }
    };
}

/// Forwards every [`Backend`](poulpy_hal::layouts::Backend) item to `$base`
/// except `Handle`, `Ring`, `destroy`, and `CYCLOTOMIC_ORDER_FACTOR`, which
/// derives from `Ring`. Used inside the target's `impl Backend` block, which
/// declares `Handle`, `Ring` and `destroy` itself. Unlike `impl_backend_from!`,
/// this declares no layout compatibility between the two backends: they serve
/// different rings.
#[macro_export]
macro_rules! forward_backend_storage {
    ($base:ty) => {
        const MIN_DEGREE: usize = <$base as $crate::layouts::Backend>::MIN_DEGREE;
        const DFT_LIMBS_CONTIGUOUS: bool = <$base as $crate::layouts::Backend>::DFT_LIMBS_CONTIGUOUS;
        const SCRATCH_ALIGN: usize = <$base as $crate::layouts::Backend>::SCRATCH_ALIGN;

        type TaskExecutor = <$base as $crate::layouts::Backend>::TaskExecutor;
        type ZnxWord = <$base as $crate::layouts::Backend>::ZnxWord;
        type BigWord = <$base as $crate::layouts::Backend>::BigWord;
        type DftWord = <$base as $crate::layouts::Backend>::DftWord;
        type OwnedBuf = <$base as $crate::layouts::Backend>::OwnedBuf;
        type BufRef<'a> = <$base as $crate::layouts::Backend>::BufRef<'a>;
        type BufMut<'a> = <$base as $crate::layouts::Backend>::BufMut<'a>;
        type Location = <$base as $crate::layouts::Backend>::Location;

        fn alloc_bytes(len: usize) -> Self::OwnedBuf {
            <$base as $crate::layouts::Backend>::alloc_bytes(len)
        }

        fn alloc_zeroed_bytes(len: usize) -> Self::OwnedBuf {
            <$base as $crate::layouts::Backend>::alloc_zeroed_bytes(len)
        }

        fn from_host_bytes(bytes: &[u8]) -> Self::OwnedBuf {
            <$base as $crate::layouts::Backend>::from_host_bytes(bytes)
        }

        fn from_bytes(bytes: Vec<u8>) -> Self::OwnedBuf {
            <$base as $crate::layouts::Backend>::from_bytes(bytes)
        }

        fn to_host_bytes(buf: &Self::OwnedBuf) -> Vec<u8> {
            <$base as $crate::layouts::Backend>::to_host_bytes(buf)
        }

        fn copy_to_host(buf: &Self::OwnedBuf, dst: &mut [u8]) {
            <$base as $crate::layouts::Backend>::copy_to_host(buf, dst)
        }

        fn copy_from_host(buf: &mut Self::OwnedBuf, src: &[u8]) {
            <$base as $crate::layouts::Backend>::copy_from_host(buf, src)
        }

        fn copy_view_to_host(buf: &Self::BufRef<'_>, dst: &mut [u8]) {
            <$base as $crate::layouts::Backend>::copy_view_to_host(buf, dst)
        }

        fn copy_host_to_view(buf: &mut Self::BufMut<'_>, src: &[u8]) {
            <$base as $crate::layouts::Backend>::copy_host_to_view(buf, src)
        }

        fn len_bytes(buf: &Self::OwnedBuf) -> usize {
            <$base as $crate::layouts::Backend>::len_bytes(buf)
        }

        fn len_bytes_ref(buf: &Self::BufRef<'_>) -> usize {
            <$base as $crate::layouts::Backend>::len_bytes_ref(buf)
        }

        fn len_bytes_mut(buf: &Self::BufMut<'_>) -> usize {
            <$base as $crate::layouts::Backend>::len_bytes_mut(buf)
        }

        fn view(buf: &Self::OwnedBuf) -> Self::BufRef<'_> {
            <$base as $crate::layouts::Backend>::view(buf)
        }

        fn view_ref<'a, 'b>(buf: &'a Self::BufRef<'b>) -> Self::BufRef<'a>
        where
            Self: 'b,
        {
            <$base as $crate::layouts::Backend>::view_ref(buf)
        }

        fn view_ref_mut<'a, 'b>(buf: &'a Self::BufMut<'b>) -> Self::BufRef<'a>
        where
            Self: 'b,
        {
            <$base as $crate::layouts::Backend>::view_ref_mut(buf)
        }

        fn view_mut_ref<'a, 'b>(buf: &'a mut Self::BufMut<'b>) -> Self::BufMut<'a>
        where
            Self: 'b,
        {
            <$base as $crate::layouts::Backend>::view_mut_ref(buf)
        }

        fn view_mut(buf: &mut Self::OwnedBuf) -> Self::BufMut<'_> {
            <$base as $crate::layouts::Backend>::view_mut(buf)
        }

        fn region(buf: &Self::OwnedBuf, offset: usize, len: usize) -> Self::BufRef<'_> {
            <$base as $crate::layouts::Backend>::region(buf, offset, len)
        }

        fn region_mut(buf: &mut Self::OwnedBuf, offset: usize, len: usize) -> Self::BufMut<'_> {
            <$base as $crate::layouts::Backend>::region_mut(buf, offset, len)
        }

        fn region_ref<'a, 'b>(buf: &'a Self::BufRef<'b>, offset: usize, len: usize) -> Self::BufRef<'a>
        where
            Self: 'b,
        {
            <$base as $crate::layouts::Backend>::region_ref(buf, offset, len)
        }

        fn region_ref_mut<'a, 'b>(buf: &'a Self::BufMut<'b>, offset: usize, len: usize) -> Self::BufRef<'a>
        where
            Self: 'b,
        {
            <$base as $crate::layouts::Backend>::region_ref_mut(buf, offset, len)
        }

        fn region_mut_ref<'a, 'b>(buf: &'a mut Self::BufMut<'b>, offset: usize, len: usize) -> Self::BufMut<'a>
        where
            Self: 'b,
        {
            <$base as $crate::layouts::Backend>::region_mut_ref(buf, offset, len)
        }

        fn size_of_znx_word() -> usize {
            <$base as $crate::layouts::Backend>::size_of_znx_word()
        }

        fn size_of_big_word() -> usize {
            <$base as $crate::layouts::Backend>::size_of_big_word()
        }

        fn size_of_dft_word() -> usize {
            <$base as $crate::layouts::Backend>::size_of_dft_word()
        }

        fn scratch_aligned(len: usize) -> usize {
            <$base as $crate::layouts::Backend>::scratch_aligned(len)
        }

        fn bytes_of_vec_znx(n: usize, cols: usize, size: usize) -> usize {
            <$base as $crate::layouts::Backend>::bytes_of_vec_znx(n, cols, size)
        }

        fn bytes_of_scalar_znx(n: usize, cols: usize) -> usize {
            <$base as $crate::layouts::Backend>::bytes_of_scalar_znx(n, cols)
        }

        fn bytes_of_mat_znx(n: usize, rows: usize, cols_in: usize, cols_out: usize, size: usize) -> usize {
            <$base as $crate::layouts::Backend>::bytes_of_mat_znx(n, rows, cols_in, cols_out, size)
        }

        fn bytes_of_vec_znx_dft(n: usize, cols: usize, size: usize) -> usize {
            <$base as $crate::layouts::Backend>::bytes_of_vec_znx_dft(n, cols, size)
        }

        fn bytes_of_vec_znx_big(n: usize, cols: usize, size: usize) -> usize {
            <$base as $crate::layouts::Backend>::bytes_of_vec_znx_big(n, cols, size)
        }

        fn bytes_of_svp_ppol(n: usize, cols: usize, hint: $crate::layouts::PrepareHint) -> usize {
            <$base as $crate::layouts::Backend>::bytes_of_svp_ppol(n, cols, hint)
        }

        fn bytes_of_vmp_pmat(
            n: usize,
            rows: usize,
            cols_in: usize,
            cols_out: usize,
            size: usize,
            hint: $crate::layouts::PrepareHint,
        ) -> usize {
            <$base as $crate::layouts::Backend>::bytes_of_vmp_pmat(n, rows, cols_in, cols_out, size, hint)
        }

        fn bytes_of_cnv_pvec_left(n: usize, cols: usize, size: usize, hint: $crate::layouts::PrepareHint) -> usize {
            <$base as $crate::layouts::Backend>::bytes_of_cnv_pvec_left(n, cols, size, hint)
        }

        fn bytes_of_cnv_pvec_right(n: usize, cols: usize, size: usize, hint: $crate::layouts::PrepareHint) -> usize {
            <$base as $crate::layouts::Backend>::bytes_of_cnv_pvec_right(n, cols, size, hint)
        }
    };
}
