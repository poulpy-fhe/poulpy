//! Backend extension points for coefficient-domain [`VecZnx`](poulpy_hal::layouts::VecZnx) operations.

use std::mem::size_of;

use crate::reference::vec_znx::{
    vec_znx_add_const_assign, vec_znx_add_const_into, vec_znx_add_into, vec_znx_add_normal_ref, vec_znx_add_scalar_assign,
    vec_znx_add_scalar_into, vec_znx_automorphism, vec_znx_automorphism_assign, vec_znx_automorphism_assign_tmp_bytes,
    vec_znx_automorphism_rotate, vec_znx_copy, vec_znx_extract_coeff, vec_znx_fill_normal_ref, vec_znx_fill_uniform_ref,
    vec_znx_lsh, vec_znx_lsh_add_coeff_to_coeff, vec_znx_lsh_assign, vec_znx_lsh_coeff, vec_znx_lsh_sub,
    vec_znx_lsh_sub_coeff_to_coeff, vec_znx_lsh_tmp_bytes, vec_znx_merge_rings, vec_znx_merge_rings_tmp_bytes,
    vec_znx_mul_xp_minus_one, vec_znx_mul_xp_minus_one_assign, vec_znx_mul_xp_minus_one_assign_tmp_bytes, vec_znx_negate,
    vec_znx_negate_assign, vec_znx_normalize, vec_znx_normalize_assign, vec_znx_normalize_coeff, vec_znx_normalize_coeff_assign,
    vec_znx_normalize_tmp_bytes, vec_znx_rotate, vec_znx_rotate_assign, vec_znx_rotate_assign_tmp_bytes, vec_znx_rsh,
    vec_znx_rsh_add_coeff_into, vec_znx_rsh_assign, vec_znx_rsh_coeff, vec_znx_rsh_sub, vec_znx_rsh_sub_coeff_into,
    vec_znx_rsh_tmp_bytes, vec_znx_split_ring, vec_znx_split_ring_tmp_bytes, vec_znx_sub, vec_znx_sub_assign,
    vec_znx_sub_negate_assign, vec_znx_sub_scalar, vec_znx_sub_scalar_assign, vec_znx_switch_ring, vec_znx_transpose,
    vec_znx_zero,
};
use crate::reference::znx::{
    ZnxAdd, ZnxAddAssign, ZnxAutomorphism, ZnxAutomorphismRotate, ZnxCopy, ZnxExtractDigitAddMul, ZnxMulPowerOfTwoAssign,
    ZnxNegate, ZnxNegateAssign, ZnxNormalizeDigit, ZnxNormalizeFinalStep, ZnxNormalizeFinalStepAssign, ZnxNormalizeFinalStepSub,
    ZnxNormalizeFirstStep, ZnxNormalizeFirstStepAssign, ZnxNormalizeFirstStepCarryOnly, ZnxNormalizeMiddleStep,
    ZnxNormalizeMiddleStepAssign, ZnxNormalizeMiddleStepCarryOnly, ZnxNormalizeMiddleStepSub, ZnxRotate, ZnxSub, ZnxSubAssign,
    ZnxSubNegateAssign, ZnxSwitchRing, ZnxZero,
};
use crate::reference::{fft64::convolution::I64Ops, ntt4x30::I128BigOps};
use poulpy_hal::{
    api::HostBufMut,
    layouts::{
        Backend, HostDataMut, Module, NoiseInfos, ScalarZnxBackendMut, ScalarZnxBackendRef, ScratchArena, VecZnxBackendMut,
        VecZnxBackendRef, VecZnxBigBackendMut, ZnxView, ZnxViewMut,
    },
    source::Source,
};

#[inline]
fn take_host_typed<'a, BE, T>(arena: ScratchArena<'a, BE>, len: usize) -> (&'a mut [T], ScratchArena<'a, BE>)
where
    BE: Backend + 'a,
    BE::BufMut<'a>: HostBufMut<'a>,
    T: Copy,
{
    debug_assert!(
        BE::SCRATCH_ALIGN.is_multiple_of(std::mem::align_of::<T>()),
        "B::SCRATCH_ALIGN ({}) must be a multiple of align_of::<T>() ({})",
        BE::SCRATCH_ALIGN,
        std::mem::align_of::<T>()
    );
    let (buf, arena) = arena.take_region(len * std::mem::size_of::<T>());
    let bytes: &'a mut [u8] = buf.into_bytes();
    let slice = unsafe { std::slice::from_raw_parts_mut(bytes.as_mut_ptr() as *mut T, len) };
    (slice, arena)
}

#[doc(hidden)]
pub trait ScalarBigHadamardProduct: Backend {
    fn scalar_big_hadamard_product(res: &mut [Self::ScalarBig], a: &[i64], b: &[i64]);
}

impl ScalarBigHadamardProduct for crate::FFT64Ref {
    #[inline(always)]
    fn scalar_big_hadamard_product(res: &mut [i64], a: &[i64], b: &[i64]) {
        Self::i64_hadamard_product(res, a, b)
    }
}

impl ScalarBigHadamardProduct for crate::NTT4x30Ref {
    #[inline(always)]
    fn scalar_big_hadamard_product(res: &mut [i128], a: &[i64], b: &[i64]) {
        Self::i128_hadamard_product_i64(res, a, b)
    }
}

#[doc(hidden)]
pub trait HalVecZnxDefault<BE: Backend>: Backend
where
    BE::OwnedBuf: poulpy_hal::layouts::HostDataMut,
{
    fn scalar_znx_fill_ternary_hw_backend_default(
        _module: &Module<BE>,
        res: &mut ScalarZnxBackendMut<'_, BE>,
        res_col: usize,
        hw: usize,
        seed: [u8; 32],
    ) where
        for<'x> BE::BufMut<'x>: HostDataMut,
    {
        let mut source = Source::new(seed);
        res.fill_ternary_hw(res_col, hw, &mut source);
    }

    fn scalar_znx_fill_ternary_prob_backend_default(
        _module: &Module<BE>,
        res: &mut ScalarZnxBackendMut<'_, BE>,
        res_col: usize,
        prob: f64,
        seed: [u8; 32],
    ) where
        for<'x> BE::BufMut<'x>: HostDataMut,
    {
        let mut source = Source::new(seed);
        res.fill_ternary_prob(res_col, prob, &mut source);
    }

    fn scalar_znx_fill_binary_hw_backend_default(
        _module: &Module<BE>,
        res: &mut ScalarZnxBackendMut<'_, BE>,
        res_col: usize,
        hw: usize,
        seed: [u8; 32],
    ) where
        for<'x> BE::BufMut<'x>: HostDataMut,
    {
        let mut source = Source::new(seed);
        res.fill_binary_hw(res_col, hw, &mut source);
    }

    fn scalar_znx_fill_binary_prob_backend_default(
        _module: &Module<BE>,
        res: &mut ScalarZnxBackendMut<'_, BE>,
        res_col: usize,
        prob: f64,
        seed: [u8; 32],
    ) where
        for<'x> BE::BufMut<'x>: HostDataMut,
    {
        let mut source = Source::new(seed);
        res.fill_binary_prob(res_col, prob, &mut source);
    }

    fn scalar_znx_fill_binary_block_backend_default(
        _module: &Module<BE>,
        res: &mut ScalarZnxBackendMut<'_, BE>,
        res_col: usize,
        block_size: usize,
        seed: [u8; 32],
    ) where
        for<'x> BE::BufMut<'x>: HostDataMut,
    {
        let mut source = Source::new(seed);
        res.fill_binary_block(res_col, block_size, &mut source);
    }

    fn vec_znx_zero_backend_default(_module: &Module<BE>, res: &mut VecZnxBackendMut<'_, BE>, res_col: usize)
    where
        BE: ZnxZero,
        for<'x> BE::BufMut<'x>: HostDataMut,
    {
        vec_znx_zero::<BE>(res, res_col);
    }

    fn vec_znx_hadamard_product_scalar_znx_backend_default(
        _module: &Module<BE>,
        res: &mut VecZnxBigBackendMut<'_, BE>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, BE>,
        a_col: usize,
        b: &ScalarZnxBackendRef<'_, BE>,
        b_col: usize,
    ) where
        BE: ScalarBigHadamardProduct,
        for<'x> BE::BufMut<'x>: HostDataMut,
        for<'x> BE::BufRef<'x>: poulpy_hal::layouts::HostDataRef,
    {
        assert_eq!(res.n(), a.n());
        assert_eq!(res.n(), b.n());
        assert!(res.size() <= a.size());

        for limb in 0..res.size() {
            let res_slice = res.at_mut(res_col, limb);
            let a_slice = a.at(a_col, limb);
            let b_slice = b.at(b_col, 0);

            BE::scalar_big_hadamard_product(res_slice, a_slice, b_slice);
        }
    }

    fn vec_znx_normalize_tmp_bytes_backend_default(module: &Module<BE>) -> usize {
        vec_znx_normalize_tmp_bytes(module.n())
    }

    #[allow(clippy::too_many_arguments)]
    fn vec_znx_normalize_backend_default(
        module: &Module<BE>,
        res: &mut VecZnxBackendMut<'_, BE>,
        res_base2k: usize,
        res_offset: i64,
        res_col: usize,
        a: &VecZnxBackendRef<'_, BE>,
        a_base2k: usize,
        a_col: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        BE: ZnxZero
            + ZnxCopy
            + ZnxAddAssign
            + ZnxMulPowerOfTwoAssign
            + ZnxNormalizeFirstStepCarryOnly
            + ZnxNormalizeMiddleStepCarryOnly
            + ZnxNormalizeMiddleStep
            + ZnxNormalizeFinalStep
            + ZnxNormalizeFirstStep
            + ZnxExtractDigitAddMul
            + ZnxNormalizeMiddleStepAssign
            + ZnxNormalizeFinalStepAssign
            + ZnxNormalizeDigit,
        for<'x> BE::BufMut<'x>: HostDataMut,
        for<'x> BE::BufRef<'x>: poulpy_hal::layouts::HostDataRef,
        for<'x> BE::BufMut<'x>: HostBufMut<'x>,
    {
        let byte_count = vec_znx_normalize_tmp_bytes(module.n());
        assert!(
            byte_count.is_multiple_of(size_of::<i64>()),
            "Scratch buffer size {} must be divisible by {}",
            byte_count,
            size_of::<i64>()
        );
        let (carry, _) = take_host_typed::<BE, i64>(scratch.borrow(), byte_count / size_of::<i64>());
        vec_znx_normalize::<BE>(res, res_base2k, res_offset, res_col, a, a_base2k, a_col, carry);
    }

    fn vec_znx_normalize_assign_backend_default(
        module: &Module<BE>,
        base2k: usize,
        res: &mut VecZnxBackendMut<'_, BE>,
        res_col: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        BE: ZnxNormalizeFirstStepAssign + ZnxNormalizeMiddleStepAssign + ZnxNormalizeFinalStepAssign,
        for<'x> BE::BufMut<'x>: HostDataMut,
        for<'x> BE::BufMut<'x>: HostBufMut<'x>,
    {
        let byte_count = vec_znx_normalize_tmp_bytes(module.n());
        assert!(
            byte_count.is_multiple_of(size_of::<i64>()),
            "Scratch buffer size {} must be divisible by {}",
            byte_count,
            size_of::<i64>()
        );
        let (carry, _) = take_host_typed::<BE, i64>(scratch.borrow(), byte_count / size_of::<i64>());
        vec_znx_normalize_assign::<BE>(base2k, res, res_col, carry);
    }

    fn vec_znx_normalize_coeff_assign_backend_default(
        _module: &Module<BE>,
        base2k: usize,
        res: &mut VecZnxBackendMut<'_, BE>,
        res_col: usize,
        res_coeff: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        BE: ZnxNormalizeFirstStepAssign + ZnxNormalizeMiddleStepAssign + ZnxNormalizeFinalStepAssign,
        for<'x> BE::BufMut<'x>: HostDataMut,
        for<'x> BE::BufMut<'x>: HostBufMut<'x>,
    {
        let (carry, _) = take_host_typed::<BE, i64>(scratch.borrow(), 1);
        vec_znx_normalize_coeff_assign::<BE>(base2k, res, res_col, res_coeff, carry);
    }

    #[allow(clippy::too_many_arguments)]
    fn vec_znx_normalize_coeff_backend_default(
        _module: &Module<BE>,
        res: &mut VecZnxBackendMut<'_, BE>,
        res_base2k: usize,
        res_offset: i64,
        res_col: usize,
        a: &VecZnxBackendRef<'_, BE>,
        a_base2k: usize,
        a_col: usize,
        a_coeff: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        BE: ZnxZero
            + ZnxCopy
            + ZnxAddAssign
            + ZnxMulPowerOfTwoAssign
            + ZnxNormalizeFirstStepCarryOnly
            + ZnxNormalizeMiddleStepCarryOnly
            + ZnxNormalizeMiddleStep
            + ZnxNormalizeFinalStep
            + ZnxNormalizeFirstStep
            + ZnxExtractDigitAddMul
            + ZnxNormalizeMiddleStepAssign
            + ZnxNormalizeFinalStepAssign
            + ZnxNormalizeDigit,
        for<'x> BE::BufMut<'x>: HostDataMut,
        for<'x> BE::BufRef<'x>: poulpy_hal::layouts::HostDataRef,
        for<'x> BE::BufMut<'x>: HostBufMut<'x>,
    {
        let (carry, _) = take_host_typed::<BE, i64>(scratch.borrow(), 3);
        vec_znx_normalize_coeff::<BE>(res, res_base2k, res_offset, res_col, a, a_base2k, a_col, a_coeff, carry);
    }

    fn vec_znx_add_into_backend_default<'a>(
        _module: &Module<BE>,
        res: &mut VecZnxBackendMut<'_, BE>,
        res_col: usize,
        a: &VecZnxBackendRef<'a, BE>,
        a_col: usize,
        b: &VecZnxBackendRef<'a, BE>,
        b_col: usize,
    ) where
        BE: ZnxAdd + ZnxCopy + ZnxZero,
        for<'x> BE::BufMut<'x>: HostDataMut,
        for<'x> BE::BufRef<'x>: PartialEq + Eq + Sized + Default + AsRef<[u8]> + Sync,
    {
        vec_znx_add_into::<BE>(res, res_col, a, a_col, b, b_col);
    }

    fn vec_znx_add_assign_backend_default(
        _module: &Module<BE>,
        res: &mut VecZnxBackendMut<'_, BE>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, BE>,
        a_col: usize,
    ) where
        BE: ZnxAddAssign,
        for<'x> BE::BufMut<'x>: HostDataMut,
        for<'x> BE::BufRef<'x>: poulpy_hal::layouts::HostDataRef,
    {
        #[cfg(debug_assertions)]
        {
            assert_eq!(a.n(), res.n());
        }

        let sum_size: usize = a.size().min(res.size());

        for j in 0..sum_size {
            BE::znx_add_assign(res.at_mut(res_col, j), a.at(a_col, j));
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn vec_znx_add_const_into_backend_default<'a>(
        _module: &Module<BE>,
        res: &mut VecZnxBackendMut<'_, BE>,
        res_col: usize,
        a: &VecZnxBackendRef<'a, BE>,
        a_col: usize,
        cnst: &VecZnxBackendRef<'a, BE>,
        cnst_col: usize,
        cnst_coeff: usize,
        res_limb: usize,
        res_coeff: usize,
    ) where
        BE: ZnxCopy + ZnxZero,
        for<'x> BE::BufMut<'x>: HostDataMut,
        for<'x> BE::BufRef<'x>: poulpy_hal::layouts::HostDataRef,
    {
        vec_znx_add_const_into::<BE>(res, res_col, a, a_col, cnst, cnst_col, cnst_coeff, res_limb, res_coeff);
    }

    fn vec_znx_add_const_assign_backend_default(
        _module: &Module<BE>,
        res: &mut VecZnxBackendMut<'_, BE>,
        res_col: usize,
        cnst: &VecZnxBackendRef<'_, BE>,
        cnst_col: usize,
        cnst_coeff: usize,
        res_limb: usize,
        res_coeff: usize,
    ) where
        for<'x> BE::BufMut<'x>: HostDataMut,
        for<'x> BE::BufRef<'x>: poulpy_hal::layouts::HostDataRef,
    {
        vec_znx_add_const_assign::<BE>(res, res_col, cnst, cnst_col, cnst_coeff, res_limb, res_coeff);
    }

    fn vec_znx_extract_coeff_backend_default(
        _module: &Module<BE>,
        res: &mut VecZnxBackendMut<'_, BE>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, BE>,
        a_col: usize,
        a_coeff: usize,
    ) where
        for<'x> BE::BufMut<'x>: HostDataMut,
        for<'x> BE::BufRef<'x>: poulpy_hal::layouts::HostDataRef,
    {
        vec_znx_extract_coeff::<BE>(res, res_col, a, a_col, a_coeff);
    }

    #[allow(clippy::too_many_arguments)]
    fn vec_znx_add_scalar_into_backend_default<'a>(
        _module: &Module<BE>,
        res: &mut VecZnxBackendMut<'_, BE>,
        res_col: usize,
        a: &ScalarZnxBackendRef<'a, BE>,
        a_col: usize,
        b: &VecZnxBackendRef<'a, BE>,
        b_col: usize,
        b_limb: usize,
    ) where
        BE: ZnxAdd + ZnxCopy + ZnxZero,
        for<'x> BE::BufMut<'x>: HostDataMut,
        for<'x> BE::BufRef<'x>: poulpy_hal::layouts::HostDataRef,
    {
        vec_znx_add_scalar_into::<BE>(res, res_col, a, a_col, b, b_col, b_limb);
    }

    fn vec_znx_add_scalar_assign_backend_default(
        _module: &Module<BE>,
        res: &mut VecZnxBackendMut<'_, BE>,
        res_col: usize,
        res_limb: usize,
        a: &ScalarZnxBackendRef<'_, BE>,
        a_col: usize,
    ) where
        BE: ZnxAddAssign,
        for<'x> BE::BufMut<'x>: HostDataMut,
        for<'x> BE::BufRef<'x>: poulpy_hal::layouts::HostDataRef,
    {
        vec_znx_add_scalar_assign::<BE>(res, res_col, res_limb, a, a_col);
    }

    fn vec_znx_sub_backend_default<'a>(
        _module: &Module<BE>,
        res: &mut VecZnxBackendMut<'_, BE>,
        res_col: usize,
        a: &VecZnxBackendRef<'a, BE>,
        a_col: usize,
        b: &VecZnxBackendRef<'a, BE>,
        b_col: usize,
    ) where
        BE: ZnxSub + ZnxNegate + ZnxZero + ZnxCopy,
        for<'x> BE::BufMut<'x>: HostDataMut,
        for<'x> BE::BufRef<'x>: poulpy_hal::layouts::HostDataRef,
    {
        vec_znx_sub::<BE>(res, res_col, a, a_col, b, b_col);
    }

    fn vec_znx_sub_assign_backend_default(
        _module: &Module<BE>,
        res: &mut VecZnxBackendMut<'_, BE>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, BE>,
        a_col: usize,
    ) where
        BE: ZnxSubAssign,
        for<'x> BE::BufMut<'x>: HostDataMut,
        for<'x> BE::BufRef<'x>: poulpy_hal::layouts::HostDataRef,
    {
        vec_znx_sub_assign::<BE>(res, res_col, a, a_col);
    }

    fn vec_znx_sub_negate_assign_backend_default(
        _module: &Module<BE>,
        res: &mut VecZnxBackendMut<'_, BE>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, BE>,
        a_col: usize,
    ) where
        BE: ZnxSubNegateAssign + ZnxNegateAssign,
        for<'x> BE::BufMut<'x>: HostDataMut,
        for<'x> BE::BufRef<'x>: poulpy_hal::layouts::HostDataRef,
    {
        vec_znx_sub_negate_assign::<BE>(res, res_col, a, a_col);
    }

    #[allow(clippy::too_many_arguments)]
    fn vec_znx_sub_scalar_backend_default<'a>(
        _module: &Module<BE>,
        res: &mut VecZnxBackendMut<'_, BE>,
        res_col: usize,
        a: &ScalarZnxBackendRef<'a, BE>,
        a_col: usize,
        b: &VecZnxBackendRef<'a, BE>,
        b_col: usize,
        b_limb: usize,
    ) where
        BE: ZnxSub + ZnxZero,
        for<'x> BE::BufMut<'x>: HostDataMut,
        for<'x> BE::BufRef<'x>: poulpy_hal::layouts::HostDataRef,
    {
        vec_znx_sub_scalar::<BE>(res, res_col, a, a_col, b, b_col, b_limb);
    }

    fn vec_znx_sub_scalar_assign_backend_default(
        _module: &Module<BE>,
        res: &mut VecZnxBackendMut<'_, BE>,
        res_col: usize,
        res_limb: usize,
        a: &ScalarZnxBackendRef<'_, BE>,
        a_col: usize,
    ) where
        BE: ZnxSubAssign,
        for<'x> BE::BufMut<'x>: HostDataMut,
        for<'x> BE::BufRef<'x>: poulpy_hal::layouts::HostDataRef,
    {
        vec_znx_sub_scalar_assign::<BE>(res, res_col, res_limb, a, a_col);
    }

    fn vec_znx_negate_backend_default(
        _module: &Module<BE>,
        res: &mut VecZnxBackendMut<'_, BE>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, BE>,
        a_col: usize,
    ) where
        BE: ZnxNegate + ZnxZero,
        for<'x> BE::BufMut<'x>: HostDataMut,
        for<'x> BE::BufRef<'x>: poulpy_hal::layouts::HostDataRef,
    {
        vec_znx_negate::<BE>(res, res_col, a, a_col);
    }

    fn vec_znx_negate_assign_backend_default(_module: &Module<BE>, res: &mut VecZnxBackendMut<'_, BE>, res_col: usize)
    where
        BE: ZnxNegateAssign,
        for<'x> BE::BufMut<'x>: HostDataMut,
    {
        vec_znx_negate_assign::<BE>(res, res_col);
    }

    fn vec_znx_rsh_tmp_bytes_backend_default(module: &Module<BE>) -> usize {
        vec_znx_rsh_tmp_bytes(module.n())
    }

    #[allow(clippy::too_many_arguments)]
    fn vec_znx_rsh_backend_default(
        module: &Module<BE>,
        base2k: usize,
        k: usize,
        res: &mut VecZnxBackendMut<'_, BE>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, BE>,
        a_col: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        BE: ZnxZero
            + ZnxCopy
            + ZnxNormalizeFirstStepCarryOnly
            + ZnxNormalizeMiddleStepCarryOnly
            + ZnxNormalizeFirstStep
            + ZnxNormalizeMiddleStep
            + ZnxNormalizeMiddleStepAssign
            + ZnxNormalizeFirstStepAssign
            + ZnxNormalizeFinalStepAssign,
        for<'x> BE::BufMut<'x>: HostDataMut,
        for<'x> BE::BufRef<'x>: poulpy_hal::layouts::HostDataRef,
        for<'x> BE::BufMut<'x>: HostBufMut<'x>,
    {
        let (carry, _) = take_host_typed::<BE, i64>(scratch.borrow(), vec_znx_rsh_tmp_bytes(module.n()) / size_of::<i64>());
        vec_znx_rsh::<BE, true>(base2k, k, res, res_col, a, a_col, carry);
    }

    #[allow(clippy::too_many_arguments)]
    fn vec_znx_rsh_coeff_backend_default(
        _module: &Module<BE>,
        base2k: usize,
        k: usize,
        res: &mut VecZnxBackendMut<'_, BE>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, BE>,
        a_col: usize,
        a_coeff: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        BE: ZnxZero
            + ZnxCopy
            + ZnxNormalizeFirstStepCarryOnly
            + ZnxNormalizeMiddleStepCarryOnly
            + ZnxNormalizeFirstStep
            + ZnxNormalizeMiddleStep
            + ZnxNormalizeMiddleStepAssign
            + ZnxNormalizeFirstStepAssign
            + ZnxNormalizeFinalStepAssign,
        for<'x> BE::BufMut<'x>: HostDataMut,
        for<'x> BE::BufRef<'x>: poulpy_hal::layouts::HostDataRef,
        for<'x> BE::BufMut<'x>: HostBufMut<'x>,
    {
        let (carry, _) = take_host_typed::<BE, i64>(scratch.borrow(), 1);
        vec_znx_rsh_coeff::<BE, true>(base2k, k, res, res_col, a, a_col, a_coeff, carry);
    }

    #[allow(clippy::too_many_arguments)]
    fn vec_znx_rsh_add_into_backend_default(
        module: &Module<BE>,
        base2k: usize,
        k: usize,
        res: &mut VecZnxBackendMut<'_, BE>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, BE>,
        a_col: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        BE: ZnxZero
            + ZnxCopy
            + ZnxNormalizeFirstStepCarryOnly
            + ZnxNormalizeMiddleStepCarryOnly
            + ZnxNormalizeFirstStep
            + ZnxNormalizeMiddleStep
            + ZnxNormalizeMiddleStepAssign
            + ZnxNormalizeFirstStepAssign
            + ZnxNormalizeFinalStepAssign,
        for<'x> BE::BufMut<'x>: HostDataMut,
        for<'x> BE::BufRef<'x>: poulpy_hal::layouts::HostDataRef,
        for<'x> BE::BufMut<'x>: HostBufMut<'x>,
    {
        let (carry, _) = take_host_typed::<BE, i64>(scratch.borrow(), vec_znx_rsh_tmp_bytes(module.n()) / size_of::<i64>());
        vec_znx_rsh::<BE, false>(base2k, k, res, res_col, a, a_col, carry);
    }

    #[allow(clippy::too_many_arguments)]
    fn vec_znx_rsh_add_coeff_into_backend_default(
        _module: &Module<BE>,
        base2k: usize,
        k: usize,
        res: &mut VecZnxBackendMut<'_, BE>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, BE>,
        a_col: usize,
        a_coeff: usize,
        res_coeff: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        BE: ZnxZero
            + ZnxCopy
            + ZnxNormalizeFirstStepCarryOnly
            + ZnxNormalizeMiddleStepCarryOnly
            + ZnxNormalizeFirstStep
            + ZnxNormalizeMiddleStep
            + ZnxNormalizeMiddleStepAssign
            + ZnxNormalizeFirstStepAssign
            + ZnxNormalizeFinalStepAssign,
        for<'x> BE::BufMut<'x>: HostDataMut,
        for<'x> BE::BufRef<'x>: poulpy_hal::layouts::HostDataRef,
        for<'x> BE::BufMut<'x>: HostBufMut<'x>,
    {
        let (carry, _) = take_host_typed::<BE, i64>(scratch.borrow(), 1);
        vec_znx_rsh_add_coeff_into::<BE>(base2k, k, res, res_col, a, a_col, a_coeff, res_coeff, carry);
    }

    #[allow(clippy::too_many_arguments)]
    fn vec_znx_rsh_sub_coeff_into_backend_default(
        _module: &Module<BE>,
        base2k: usize,
        k: usize,
        res: &mut VecZnxBackendMut<'_, BE>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, BE>,
        a_col: usize,
        a_coeff: usize,
        res_coeff: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        BE: ZnxZero
            + ZnxCopy
            + ZnxNormalizeFirstStepCarryOnly
            + ZnxNormalizeMiddleStepCarryOnly
            + ZnxNormalizeFirstStep
            + ZnxNormalizeMiddleStep
            + ZnxNormalizeMiddleStepSub
            + ZnxNormalizeFinalStepSub,
        for<'x> BE::BufMut<'x>: HostDataMut,
        for<'x> BE::BufRef<'x>: poulpy_hal::layouts::HostDataRef,
        for<'x> BE::BufMut<'x>: HostBufMut<'x>,
    {
        let (carry, _) = take_host_typed::<BE, i64>(scratch.borrow(), 1);
        vec_znx_rsh_sub_coeff_into::<BE>(base2k, k, res, res_col, a, a_col, a_coeff, res_coeff, carry);
    }

    fn vec_znx_lsh_tmp_bytes_backend_default(module: &Module<BE>) -> usize {
        vec_znx_lsh_tmp_bytes(module.n())
    }

    #[allow(clippy::too_many_arguments)]
    fn vec_znx_lsh_backend_default(
        module: &Module<BE>,
        base2k: usize,
        k: usize,
        res: &mut VecZnxBackendMut<'_, BE>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, BE>,
        a_col: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        BE: ZnxZero
            + ZnxNormalizeFirstStep
            + ZnxNormalizeMiddleStep
            + ZnxCopy
            + ZnxNormalizeFinalStep
            + ZnxNormalizeFirstStepCarryOnly
            + ZnxNormalizeMiddleStepCarryOnly,
        for<'x> BE::BufMut<'x>: HostDataMut,
        for<'x> BE::BufRef<'x>: poulpy_hal::layouts::HostDataRef,
        for<'x> BE::BufMut<'x>: HostBufMut<'x>,
    {
        let (carry, _) = take_host_typed::<BE, i64>(scratch.borrow(), vec_znx_lsh_tmp_bytes(module.n()) / size_of::<i64>());
        vec_znx_lsh::<BE, true>(base2k, k, res, res_col, a, a_col, carry);
    }

    #[allow(clippy::too_many_arguments)]
    fn vec_znx_lsh_coeff_backend_default(
        _module: &Module<BE>,
        base2k: usize,
        k: usize,
        res: &mut VecZnxBackendMut<'_, BE>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, BE>,
        a_col: usize,
        a_coeff: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        BE: ZnxZero
            + ZnxNormalizeFirstStep
            + ZnxNormalizeMiddleStep
            + ZnxCopy
            + ZnxNormalizeFinalStep
            + ZnxNormalizeFirstStepCarryOnly
            + ZnxNormalizeMiddleStepCarryOnly,
        for<'x> BE::BufMut<'x>: HostDataMut,
        for<'x> BE::BufRef<'x>: poulpy_hal::layouts::HostDataRef,
        for<'x> BE::BufMut<'x>: HostBufMut<'x>,
    {
        let (carry, _) = take_host_typed::<BE, i64>(scratch.borrow(), 1);
        vec_znx_lsh_coeff::<BE, true>(base2k, k, res, res_col, a, a_col, a_coeff, carry);
    }

    #[allow(clippy::too_many_arguments)]
    fn vec_znx_lsh_add_into_backend_default(
        module: &Module<BE>,
        base2k: usize,
        k: usize,
        res: &mut VecZnxBackendMut<'_, BE>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, BE>,
        a_col: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        BE: ZnxZero
            + ZnxNormalizeFirstStep
            + ZnxNormalizeMiddleStep
            + ZnxCopy
            + ZnxNormalizeFinalStep
            + ZnxNormalizeFirstStepCarryOnly
            + ZnxNormalizeMiddleStepCarryOnly,
        for<'x> BE::BufMut<'x>: HostDataMut,
        for<'x> BE::BufRef<'x>: poulpy_hal::layouts::HostDataRef,
        for<'x> BE::BufMut<'x>: HostBufMut<'x>,
    {
        let (carry, _) = take_host_typed::<BE, i64>(scratch.borrow(), vec_znx_lsh_tmp_bytes(module.n()) / size_of::<i64>());
        vec_znx_lsh::<BE, false>(base2k, k, res, res_col, a, a_col, carry);
    }

    #[allow(clippy::too_many_arguments)]
    fn vec_znx_lsh_add_coeff_into_backend_default(
        _module: &Module<BE>,
        base2k: usize,
        k: usize,
        res: &mut VecZnxBackendMut<'_, BE>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, BE>,
        a_col: usize,
        a_coeff: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        BE: ZnxZero
            + ZnxNormalizeFirstStep
            + ZnxNormalizeMiddleStep
            + ZnxCopy
            + ZnxNormalizeFinalStep
            + ZnxNormalizeFirstStepCarryOnly
            + ZnxNormalizeMiddleStepCarryOnly,
        for<'x> BE::BufMut<'x>: HostDataMut,
        for<'x> BE::BufRef<'x>: poulpy_hal::layouts::HostDataRef,
        for<'x> BE::BufMut<'x>: HostBufMut<'x>,
    {
        let (carry, _) = take_host_typed::<BE, i64>(scratch.borrow(), 1);
        vec_znx_lsh_coeff::<BE, false>(base2k, k, res, res_col, a, a_col, a_coeff, carry);
    }

    #[allow(clippy::too_many_arguments)]
    fn vec_znx_lsh_add_coeff_to_coeff_backend_default<'s, 'r, 'a>(
        _module: &Module<BE>,
        base2k: usize,
        k: usize,
        res: &mut VecZnxBackendMut<'r, BE>,
        res_col: usize,
        a: &VecZnxBackendRef<'a, BE>,
        a_col: usize,
        a_coeff: usize,
        res_coeff: usize,
        scratch: &'s mut ScratchArena<'s, BE>,
    ) where
        BE: 's,
        BE: ZnxZero
            + ZnxNormalizeFirstStep
            + ZnxNormalizeMiddleStep
            + ZnxNormalizeFinalStep
            + ZnxNormalizeFirstStepCarryOnly
            + ZnxNormalizeMiddleStepCarryOnly,
        BE::BufMut<'r>: HostDataMut,
        BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
        BE::BufMut<'s>: HostBufMut<'s>,
    {
        let (carry, _) = take_host_typed::<BE, i64>(scratch.borrow(), 1);
        vec_znx_lsh_add_coeff_to_coeff::<BE>(base2k, k, res, res_col, a, a_col, a_coeff, res_coeff, carry);
    }

    #[allow(clippy::too_many_arguments)]
    fn vec_znx_lsh_sub_coeff_to_coeff_backend_default<'s, 'r, 'a>(
        _module: &Module<BE>,
        base2k: usize,
        k: usize,
        res: &mut VecZnxBackendMut<'r, BE>,
        res_col: usize,
        a: &VecZnxBackendRef<'a, BE>,
        a_col: usize,
        a_coeff: usize,
        res_coeff: usize,
        scratch: &'s mut ScratchArena<'s, BE>,
    ) where
        BE: 's,
        BE: ZnxZero
            + ZnxNormalizeFirstStepCarryOnly
            + ZnxNormalizeMiddleStepSub
            + ZnxNormalizeFinalStepSub
            + ZnxNormalizeMiddleStepCarryOnly,
        BE::BufMut<'r>: HostDataMut,
        BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
        BE::BufMut<'s>: HostBufMut<'s>,
    {
        let (carry, _) = take_host_typed::<BE, i64>(scratch.borrow(), 1);
        vec_znx_lsh_sub_coeff_to_coeff::<BE>(base2k, k, res, res_col, a, a_col, a_coeff, res_coeff, carry);
    }

    #[allow(clippy::too_many_arguments)]
    fn vec_znx_lsh_sub_backend_default<'s, 'r, 'a>(
        module: &Module<BE>,
        base2k: usize,
        k: usize,
        res: &mut VecZnxBackendMut<'_, BE>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, BE>,
        a_col: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        BE: ZnxZero
            + ZnxNormalizeFirstStepCarryOnly
            + ZnxNormalizeMiddleStepSub
            + ZnxNormalizeFinalStepSub
            + ZnxNormalizeMiddleStepCarryOnly,
        for<'x> BE::BufMut<'x>: HostDataMut,
        for<'x> BE::BufRef<'x>: poulpy_hal::layouts::HostDataRef,
        for<'x> BE::BufMut<'x>: HostBufMut<'x>,
    {
        let (carry, _) = take_host_typed::<BE, i64>(scratch.borrow(), vec_znx_lsh_tmp_bytes(module.n()) / size_of::<i64>());
        vec_znx_lsh_sub::<BE>(base2k, k, res, res_col, a, a_col, carry);
    }

    #[allow(clippy::too_many_arguments)]
    fn vec_znx_rsh_sub_backend_default(
        module: &Module<BE>,
        base2k: usize,
        k: usize,
        res: &mut VecZnxBackendMut<'_, BE>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, BE>,
        a_col: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        BE: ZnxZero
            + ZnxCopy
            + ZnxNormalizeFirstStepCarryOnly
            + ZnxNormalizeMiddleStepCarryOnly
            + ZnxNormalizeMiddleStepSub
            + ZnxNormalizeMiddleStepAssign
            + ZnxNormalizeFirstStepAssign
            + ZnxNormalizeFinalStepAssign,
        for<'x> BE::BufMut<'x>: HostDataMut,
        for<'x> BE::BufRef<'x>: poulpy_hal::layouts::HostDataRef,
        for<'x> BE::BufMut<'x>: HostBufMut<'x>,
    {
        let (carry, _) = take_host_typed::<BE, i64>(scratch.borrow(), vec_znx_rsh_tmp_bytes(module.n()) / size_of::<i64>());
        vec_znx_rsh_sub::<BE>(base2k, k, res, res_col, a, a_col, carry);
    }

    fn vec_znx_rsh_assign_backend_default(
        module: &Module<BE>,
        base2k: usize,
        k: usize,
        res: &mut VecZnxBackendMut<'_, BE>,
        res_col: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        BE: ZnxZero
            + ZnxCopy
            + ZnxNormalizeFirstStepCarryOnly
            + ZnxNormalizeMiddleStepCarryOnly
            + ZnxNormalizeMiddleStep
            + ZnxNormalizeMiddleStepAssign
            + ZnxNormalizeFirstStepAssign
            + ZnxNormalizeFinalStepAssign,
        for<'x> BE::BufMut<'x>: HostDataMut,
        for<'x> BE::BufMut<'x>: HostBufMut<'x>,
    {
        let (carry, _) = take_host_typed::<BE, i64>(scratch.borrow(), vec_znx_rsh_tmp_bytes(module.n()) / size_of::<i64>());
        vec_znx_rsh_assign::<BE>(base2k, k, res, res_col, carry);
    }

    fn vec_znx_lsh_assign_backend_default(
        module: &Module<BE>,
        base2k: usize,
        k: usize,
        res: &mut VecZnxBackendMut<'_, BE>,
        res_col: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        BE: ZnxZero + ZnxCopy + ZnxNormalizeFirstStepAssign + ZnxNormalizeMiddleStepAssign + ZnxNormalizeFinalStepAssign,
        for<'x> BE::BufMut<'x>: HostDataMut,
        for<'x> BE::BufMut<'x>: HostBufMut<'x>,
    {
        let (carry, _) = take_host_typed::<BE, i64>(scratch.borrow(), vec_znx_lsh_tmp_bytes(module.n()) / size_of::<i64>());
        vec_znx_lsh_assign::<BE>(base2k, k, res, res_col, carry);
    }

    fn vec_znx_rotate_backend_default(
        _module: &Module<BE>,
        p: i64,
        res: &mut VecZnxBackendMut<'_, BE>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, BE>,
        a_col: usize,
    ) where
        BE: ZnxRotate + ZnxZero,
        for<'x> BE::BufMut<'x>: HostDataMut,
        for<'x> BE::BufRef<'x>: poulpy_hal::layouts::HostDataRef,
    {
        vec_znx_rotate::<BE>(p, res, res_col, a, a_col);
    }

    fn vec_znx_rotate_assign_tmp_bytes_backend_default(module: &Module<BE>) -> usize {
        vec_znx_rotate_assign_tmp_bytes(module.n())
    }

    fn vec_znx_rotate_assign_backend_default(
        module: &Module<BE>,
        p: i64,
        res: &mut VecZnxBackendMut<'_, BE>,
        res_col: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        BE: ZnxRotate + ZnxCopy,
        for<'x> BE::BufMut<'x>: HostDataMut,
        for<'x> BE::BufMut<'x>: HostBufMut<'x>,
    {
        let (tmp, _) = take_host_typed::<BE, i64>(
            scratch.borrow(),
            vec_znx_rotate_assign_tmp_bytes(module.n()) / size_of::<i64>(),
        );
        vec_znx_rotate_assign::<BE>(p, res, res_col, tmp);
    }

    fn vec_znx_automorphism_backend_default(
        _module: &Module<BE>,
        p: i64,
        res: &mut VecZnxBackendMut<'_, BE>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, BE>,
        a_col: usize,
    ) where
        BE: ZnxAutomorphism + ZnxZero,
        for<'x> BE::BufMut<'x>: HostDataMut,
        for<'x> BE::BufRef<'x>: poulpy_hal::layouts::HostDataRef,
    {
        vec_znx_automorphism::<BE>(p, res, res_col, a, a_col);
    }

    fn vec_znx_automorphism_assign_tmp_bytes_backend_default(module: &Module<BE>) -> usize {
        vec_znx_automorphism_assign_tmp_bytes(module.n())
    }

    fn vec_znx_automorphism_assign_backend_default(
        module: &Module<BE>,
        p: i64,
        res: &mut VecZnxBackendMut<'_, BE>,
        res_col: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        BE: ZnxAutomorphism + ZnxCopy,
        for<'x> BE::BufMut<'x>: HostDataMut,
        for<'x> BE::BufMut<'x>: HostBufMut<'x>,
    {
        let (tmp, _) = take_host_typed::<BE, i64>(
            scratch.borrow(),
            vec_znx_automorphism_assign_tmp_bytes(module.n()) / size_of::<i64>(),
        );
        vec_znx_automorphism_assign::<BE>(p, res, res_col, tmp);
    }

    #[allow(clippy::too_many_arguments)]
    fn vec_znx_automorphism_rotate_backend_default(
        _module: &Module<BE>,
        p: i64,
        k: i64,
        res: &mut VecZnxBackendMut<'_, BE>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, BE>,
        a_col: usize,
    ) where
        BE: ZnxAutomorphismRotate + ZnxZero,
        for<'x> BE::BufMut<'x>: HostDataMut,
        for<'x> BE::BufRef<'x>: poulpy_hal::layouts::HostDataRef,
    {
        vec_znx_automorphism_rotate::<BE>(p, k, res, res_col, a, a_col);
    }

    fn vec_znx_mul_xp_minus_one_backend_default(
        _module: &Module<BE>,
        p: i64,
        res: &mut VecZnxBackendMut<'_, BE>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, BE>,
        a_col: usize,
    ) where
        BE: ZnxRotate + ZnxZero + ZnxSubAssign,
        for<'x> BE::BufMut<'x>: HostDataMut,
        for<'x> BE::BufRef<'x>: poulpy_hal::layouts::HostDataRef,
    {
        vec_znx_mul_xp_minus_one::<BE>(p, res, res_col, a, a_col);
    }

    fn vec_znx_mul_xp_minus_one_assign_tmp_bytes_backend_default(module: &Module<BE>) -> usize {
        vec_znx_mul_xp_minus_one_assign_tmp_bytes(module.n())
    }

    fn vec_znx_mul_xp_minus_one_assign_backend_default(
        module: &Module<BE>,
        p: i64,
        res: &mut VecZnxBackendMut<'_, BE>,
        res_col: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        BE: ZnxRotate + ZnxNegate + ZnxSubNegateAssign,
        for<'x> BE::BufMut<'x>: HostDataMut,
        for<'x> BE::BufMut<'x>: HostBufMut<'x>,
    {
        let (tmp, _) = take_host_typed::<BE, i64>(
            scratch.borrow(),
            vec_znx_mul_xp_minus_one_assign_tmp_bytes(module.n()) / size_of::<i64>(),
        );
        vec_znx_mul_xp_minus_one_assign::<BE>(p, res, res_col, tmp);
    }

    fn vec_znx_split_ring_tmp_bytes_backend_default(module: &Module<BE>) -> usize {
        vec_znx_split_ring_tmp_bytes(module.n())
    }

    fn vec_znx_split_ring_backend_default(
        module: &Module<BE>,
        res: &mut [VecZnxBackendMut<'_, BE>],
        res_col: usize,
        a: &VecZnxBackendRef<'_, BE>,
        a_col: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        BE: ZnxSwitchRing + ZnxRotate + ZnxZero,
        for<'x> BE::BufMut<'x>: HostDataMut,
        for<'x> BE::BufRef<'x>: poulpy_hal::layouts::HostDataRef,
        for<'x> BE::BufMut<'x>: HostBufMut<'x>,
    {
        let (tmp, _) = take_host_typed::<BE, i64>(scratch.borrow(), vec_znx_split_ring_tmp_bytes(module.n()) / size_of::<i64>());
        vec_znx_split_ring::<BE>(res, res_col, a, a_col, tmp);
    }

    fn vec_znx_merge_rings_tmp_bytes_backend_default(module: &Module<BE>) -> usize {
        vec_znx_merge_rings_tmp_bytes(module.n())
    }

    fn vec_znx_merge_rings_backend_default(
        module: &Module<BE>,
        res: &mut VecZnxBackendMut<'_, BE>,
        res_col: usize,
        a: &[VecZnxBackendRef<'_, BE>],
        a_col: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        BE: ZnxCopy + ZnxSwitchRing + ZnxRotate + ZnxZero,
        for<'x> BE::BufMut<'x>: HostDataMut,
        for<'x> BE::BufRef<'x>: poulpy_hal::layouts::HostDataRef,
        for<'x> BE::BufMut<'x>: HostBufMut<'x>,
    {
        let (tmp, _) = take_host_typed::<BE, i64>(scratch.borrow(), vec_znx_merge_rings_tmp_bytes(module.n()) / size_of::<i64>());
        vec_znx_merge_rings::<BE>(res, res_col, a, a_col, tmp);
    }

    fn vec_znx_switch_ring_backend_default(
        _module: &Module<BE>,
        res: &mut VecZnxBackendMut<'_, BE>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, BE>,
        a_col: usize,
    ) where
        BE: ZnxCopy + ZnxSwitchRing + ZnxZero,
        for<'x> BE::BufMut<'x>: HostDataMut,
        for<'x> BE::BufRef<'x>: poulpy_hal::layouts::HostDataRef,
    {
        vec_znx_switch_ring::<BE>(res, res_col, a, a_col);
    }

    fn vec_znx_copy_backend_default(
        _module: &Module<BE>,
        res: &mut VecZnxBackendMut<'_, BE>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, BE>,
        a_col: usize,
    ) where
        BE: ZnxCopy + ZnxZero,
        for<'x> BE::BufMut<'x>: HostDataMut,
        for<'x> BE::BufRef<'x>: poulpy_hal::layouts::HostDataRef,
    {
        vec_znx_copy::<BE>(res, res_col, a, a_col);
    }

    fn vec_znx_transpose_backend_default(_module: &Module<BE>, res: &mut VecZnxBackendMut<'_, BE>, a: &VecZnxBackendRef<'_, BE>)
    where
        for<'x> BE::BufMut<'x>: HostDataMut,
        for<'x> BE::BufRef<'x>: poulpy_hal::layouts::HostDataRef,
    {
        vec_znx_transpose::<BE>(res, a);
    }

    #[allow(clippy::too_many_arguments)]
    fn vec_znx_copy_range_backend_default(
        _module: &Module<BE>,
        res: &mut VecZnxBackendMut<'_, BE>,
        res_col: usize,
        res_limb: usize,
        res_offset: usize,
        a: &VecZnxBackendRef<'_, BE>,
        a_col: usize,
        a_limb: usize,
        a_offset: usize,
        len: usize,
    ) where
        BE: ZnxCopy,
        for<'x> BE::BufMut<'x>: HostDataMut,
        for<'x> BE::BufRef<'x>: poulpy_hal::layouts::HostDataRef,
    {
        crate::reference::vec_znx::vec_znx_copy_range::<BE>(res, res_col, res_limb, res_offset, a, a_col, a_limb, a_offset, len);
    }

    fn vec_znx_fill_uniform_backend_default(
        _module: &Module<BE>,
        base2k: usize,
        res: &mut VecZnxBackendMut<'_, BE>,
        res_col: usize,
        seed: [u8; 32],
    ) where
        for<'x> BE::BufMut<'x>: HostDataMut,
    {
        let mut source = Source::new(seed);
        vec_znx_fill_uniform_ref::<BE>(base2k, res, res_col, &mut source);
    }

    fn vec_znx_fill_normal_backend_default(
        _module: &Module<BE>,
        res_base2k: usize,
        res: &mut VecZnxBackendMut<'_, BE>,
        res_col: usize,
        noise_infos: NoiseInfos,
        seed: [u8; 32],
    ) where
        for<'x> BE::BufMut<'x>: HostDataMut,
    {
        let mut source = Source::new(seed);
        vec_znx_fill_normal_ref::<BE>(res_base2k, res, res_col, noise_infos, &mut source);
    }

    fn vec_znx_add_normal_backend_default(
        _module: &Module<BE>,
        res_base2k: usize,
        res: &mut VecZnxBackendMut<'_, BE>,
        res_col: usize,
        noise_infos: NoiseInfos,
        seed: [u8; 32],
    ) where
        for<'x> BE::BufMut<'x>: HostDataMut,
    {
        let mut source = Source::new(seed);
        vec_znx_add_normal_ref::<BE>(res_base2k, res, res_col, noise_infos, &mut source);
    }
}

impl<BE: Backend> HalVecZnxDefault<BE> for BE where BE::OwnedBuf: poulpy_hal::layouts::HostDataMut {}
