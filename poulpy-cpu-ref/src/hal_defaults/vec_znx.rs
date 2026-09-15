//! Backend extension points for coefficient-domain [`VecZnx`](poulpy_hal::layouts::VecZnx) operations.

use std::mem::size_of;

use crate::reference::vec_znx::{
    vec_znx_add, vec_znx_automorphism, vec_znx_automorphism_assign, vec_znx_automorphism_assign_tmp_bytes, vec_znx_copy,
    vec_znx_fill_uniform_ref, vec_znx_lsh_assign, vec_znx_lsh_assign_carry_bytes, vec_znx_mul_xp_minus_one_assign,
    vec_znx_mul_xp_minus_one_assign_tmp_bytes, vec_znx_negate, vec_znx_negate_assign, vec_znx_normalize,
    vec_znx_normalize_assign, vec_znx_normalize_tmp_bytes, vec_znx_rotate, vec_znx_rotate_assign,
    vec_znx_rotate_assign_tmp_bytes, vec_znx_sub, vec_znx_sub_assign, vec_znx_sub_negate_assign, vec_znx_switch_ring,
    vec_znx_zero,
};
use crate::reference::znx::{
    I64NormalizeOps, ZnxAdd, ZnxAddAssign, ZnxAutomorphism, ZnxCopy, ZnxMulPowerOfTwoAssign, ZnxNegate, ZnxNegateAssign,
    ZnxNormalizeDigit, ZnxNormalizeFinalStep, ZnxNormalizeFinalStepAssign, ZnxNormalizeFirstStep, ZnxNormalizeFirstStepAssign,
    ZnxNormalizeFirstStepCarryOnly, ZnxNormalizeMiddleStep, ZnxNormalizeMiddleStepAssign, ZnxNormalizeMiddleStepCarryOnly,
    ZnxRotate, ZnxSub, ZnxSubAssign, ZnxSubNegateAssign, ZnxSwitchRing, ZnxZero,
};
use crate::reference::{fft64::convolution::I64Ops, ntt4x30::I128BigOps};
use poulpy_hal::{
    api::HostBufMut,
    layouts::{Backend, HostDataMut, Module, ScratchArena, VecZnxBackendMut, VecZnxBackendRef, ZnxView, ZnxViewMut},
    source::Source,
};

#[inline]
fn take_host_typed<'a, BE, T>(arena: ScratchArena<'a, BE>, len: usize) -> (&'a mut [T], ScratchArena<'a, BE>)
where
    BE: Backend<ZnxWord = i64> + 'a,
    BE::BufMut<'a>: HostBufMut<'a>,
    T: Copy,
{
    assert!(
        BE::SCRATCH_ALIGN.is_multiple_of(std::mem::align_of::<T>()),
        "B::SCRATCH_ALIGN ({}) must be a multiple of align_of::<T>() ({})",
        BE::SCRATCH_ALIGN,
        std::mem::align_of::<T>()
    );
    let byte_len = len
        .checked_mul(std::mem::size_of::<T>())
        .expect("typed scratch byte size overflows usize");
    let (buf, arena) = arena.take_region(byte_len);
    let bytes: &'a mut [u8] = buf.into_bytes();
    assert!(
        (bytes.as_mut_ptr() as usize).is_multiple_of(std::mem::align_of::<T>()),
        "scratch region is not aligned to align_of::<T>() = {}",
        std::mem::align_of::<T>()
    );
    let slice = unsafe { std::slice::from_raw_parts_mut(bytes.as_mut_ptr() as *mut T, len) };
    (slice, arena)
}

#[doc(hidden)]
pub trait BigWordHadamardProduct: Backend<ZnxWord = i64> {
    fn big_word_hadamard_product(res: &mut [Self::BigWord], a: &[i64], b: &[i64]);
}

impl BigWordHadamardProduct for crate::FFT64Ref {
    #[inline(always)]
    fn big_word_hadamard_product(res: &mut [i64], a: &[i64], b: &[i64]) {
        Self::i64_hadamard_product(res, a, b)
    }
}

impl BigWordHadamardProduct for crate::NTT4x30Ref {
    #[inline(always)]
    fn big_word_hadamard_product(res: &mut [i128], a: &[i64], b: &[i64]) {
        Self::i128_hadamard_product_i64(res, a, b)
    }
}

#[doc(hidden)]
pub trait HalVecZnxDefault: Backend<ZnxWord = i64>
where
    Self::OwnedBuf: poulpy_hal::layouts::HostDataMut,
{
    fn vec_znx_zero_default(_module: &Module<Self>, res: &mut VecZnxBackendMut<'_, Self>, res_col: usize)
    where
        Self: ZnxZero,
        for<'x> Self::BufMut<'x>: HostDataMut,
    {
        vec_znx_zero::<Self>(res, res_col);
    }

    fn vec_znx_normalize_tmp_bytes_default(module: &Module<Self>) -> usize {
        vec_znx_normalize_tmp_bytes(module.n())
    }

    #[allow(clippy::too_many_arguments)]
    fn vec_znx_normalize_default(
        module: &Module<Self>,
        res: &mut VecZnxBackendMut<'_, Self>,
        res_base2k: usize,
        res_k: usize,
        res_offset: i64,
        res_col: usize,
        a: &VecZnxBackendRef<'_, Self>,
        a_base2k: usize,
        a_col: usize,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        Self: ZnxZero
            + ZnxCopy
            + ZnxAddAssign
            + ZnxMulPowerOfTwoAssign
            + ZnxNormalizeFirstStepCarryOnly
            + ZnxNormalizeMiddleStepCarryOnly
            + ZnxNormalizeMiddleStep
            + ZnxNormalizeFinalStep
            + ZnxNormalizeFirstStep
            + I64NormalizeOps
            + ZnxNormalizeMiddleStepAssign
            + ZnxNormalizeFinalStepAssign
            + ZnxNormalizeDigit,
        for<'x> Self::BufMut<'x>: HostDataMut,
        for<'x> Self::BufRef<'x>: poulpy_hal::layouts::HostDataRef,
        for<'x> Self::BufMut<'x>: HostBufMut<'x>,
    {
        let byte_count = vec_znx_normalize_tmp_bytes(module.n());
        assert!(
            byte_count.is_multiple_of(size_of::<i64>()),
            "Scratch buffer size {} must be divisible by {}",
            byte_count,
            size_of::<i64>()
        );
        let (carry, _) = take_host_typed::<Self, i64>(scratch.borrow(), byte_count / size_of::<i64>());
        vec_znx_normalize::<Self>(res, res_base2k, res_k, res_offset, res_col, a, a_base2k, a_col, carry);
    }

    #[allow(clippy::too_many_arguments)]
    fn vec_znx_normalize_assign_default(
        module: &Module<Self>,
        base2k: usize,
        k: usize,
        res_offset: i64,
        res: &mut VecZnxBackendMut<'_, Self>,
        res_col: usize,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        Self: I64NormalizeOps
            + ZnxZero
            + ZnxNormalizeFirstStepCarryOnly
            + ZnxNormalizeMiddleStepCarryOnly
            + ZnxNormalizeMiddleStep
            + ZnxNormalizeFirstStepAssign
            + ZnxNormalizeMiddleStepAssign
            + ZnxNormalizeFinalStepAssign,
        for<'x> Self::BufMut<'x>: HostDataMut,
        for<'x> Self::BufMut<'x>: HostBufMut<'x>,
    {
        let byte_count = vec_znx_normalize_tmp_bytes(module.n());
        assert!(
            byte_count.is_multiple_of(size_of::<i64>()),
            "Scratch buffer size {} must be divisible by {}",
            byte_count,
            size_of::<i64>()
        );
        let (carry, _) = take_host_typed::<Self, i64>(scratch.borrow(), byte_count / size_of::<i64>());
        vec_znx_normalize_assign::<Self>(base2k, k, res_offset, res, res_col, carry);
    }

    fn vec_znx_add_default<'a>(
        _module: &Module<Self>,
        res: &mut VecZnxBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxBackendRef<'a, Self>,
        a_col: usize,
        b: &VecZnxBackendRef<'a, Self>,
        b_col: usize,
    ) where
        Self: ZnxAdd + ZnxCopy + ZnxZero,
        for<'x> Self::BufMut<'x>: HostDataMut,
        for<'x> Self::BufRef<'x>: PartialEq + Eq + Sized + Default + AsRef<[u8]> + Sync,
    {
        vec_znx_add::<Self>(res, res_col, a, a_col, b, b_col);
    }

    fn vec_znx_add_assign_default(
        _module: &Module<Self>,
        res: &mut VecZnxBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, Self>,
        a_col: usize,
    ) where
        Self: ZnxAddAssign,
        for<'x> Self::BufMut<'x>: HostDataMut,
        for<'x> Self::BufRef<'x>: poulpy_hal::layouts::HostDataRef,
    {
        {
            assert_eq!(a.n(), res.n());
        }

        let sum_size: usize = a.size().min(res.size());

        for j in 0..sum_size {
            Self::znx_add_assign(res.at_mut(res_col, j), a.at(a_col, j));
        }
    }

    fn vec_znx_sub_default<'a>(
        _module: &Module<Self>,
        res: &mut VecZnxBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxBackendRef<'a, Self>,
        a_col: usize,
        b: &VecZnxBackendRef<'a, Self>,
        b_col: usize,
    ) where
        Self: ZnxSub + ZnxNegate + ZnxZero + ZnxCopy,
        for<'x> Self::BufMut<'x>: HostDataMut,
        for<'x> Self::BufRef<'x>: poulpy_hal::layouts::HostDataRef,
    {
        vec_znx_sub::<Self>(res, res_col, a, a_col, b, b_col);
    }

    fn vec_znx_sub_assign_default(
        _module: &Module<Self>,
        res: &mut VecZnxBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, Self>,
        a_col: usize,
    ) where
        Self: ZnxSubAssign,
        for<'x> Self::BufMut<'x>: HostDataMut,
        for<'x> Self::BufRef<'x>: poulpy_hal::layouts::HostDataRef,
    {
        vec_znx_sub_assign::<Self>(res, res_col, a, a_col);
    }

    fn vec_znx_sub_negate_assign_default(
        _module: &Module<Self>,
        res: &mut VecZnxBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, Self>,
        a_col: usize,
    ) where
        Self: ZnxSubNegateAssign + ZnxNegateAssign,
        for<'x> Self::BufMut<'x>: HostDataMut,
        for<'x> Self::BufRef<'x>: poulpy_hal::layouts::HostDataRef,
    {
        vec_znx_sub_negate_assign::<Self>(res, res_col, a, a_col);
    }

    fn vec_znx_negate_default(
        _module: &Module<Self>,
        res: &mut VecZnxBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, Self>,
        a_col: usize,
    ) where
        Self: ZnxNegate + ZnxZero,
        for<'x> Self::BufMut<'x>: HostDataMut,
        for<'x> Self::BufRef<'x>: poulpy_hal::layouts::HostDataRef,
    {
        vec_znx_negate::<Self>(res, res_col, a, a_col);
    }

    fn vec_znx_negate_assign_default(_module: &Module<Self>, res: &mut VecZnxBackendMut<'_, Self>, res_col: usize)
    where
        Self: ZnxNegateAssign,
        for<'x> Self::BufMut<'x>: HostDataMut,
    {
        vec_znx_negate_assign::<Self>(res, res_col);
    }

    /// CPU override of [`poulpy_hal::oep::vec_znx_lsh_assign_derived`]: the
    /// fused in-place kernel, bit-exact with the default.
    fn vec_znx_lsh_assign_default(
        module: &Module<Self>,
        base2k: usize,
        k: usize,
        res: &mut VecZnxBackendMut<'_, Self>,
        res_col: usize,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        Self: ZnxZero + ZnxCopy + ZnxNormalizeFirstStepAssign + ZnxNormalizeMiddleStepAssign + ZnxNormalizeFinalStepAssign,
        for<'x> Self::BufMut<'x>: HostDataMut,
        for<'x> Self::BufMut<'x>: HostBufMut<'x>,
    {
        let (carry, _) = take_host_typed::<Self, i64>(
            scratch.borrow(),
            vec_znx_lsh_assign_carry_bytes(module.n()) / size_of::<i64>(),
        );
        vec_znx_lsh_assign::<Self>(base2k, k, res, res_col, carry);
    }

    fn vec_znx_rotate_default(
        _module: &Module<Self>,
        p: i64,
        res: &mut VecZnxBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, Self>,
        a_col: usize,
    ) where
        Self: ZnxRotate + ZnxZero,
        for<'x> Self::BufMut<'x>: HostDataMut,
        for<'x> Self::BufRef<'x>: poulpy_hal::layouts::HostDataRef,
    {
        vec_znx_rotate::<Self>(p, res, res_col, a, a_col);
    }

    fn vec_znx_rotate_assign_tmp_bytes_default(module: &Module<Self>) -> usize {
        vec_znx_rotate_assign_tmp_bytes(module.n())
    }

    fn vec_znx_rotate_assign_default(
        module: &Module<Self>,
        p: i64,
        res: &mut VecZnxBackendMut<'_, Self>,
        res_col: usize,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        Self: ZnxRotate + ZnxCopy,
        for<'x> Self::BufMut<'x>: HostDataMut,
        for<'x> Self::BufMut<'x>: HostBufMut<'x>,
    {
        let (tmp, _) = take_host_typed::<Self, i64>(
            scratch.borrow(),
            vec_znx_rotate_assign_tmp_bytes(module.n()) / size_of::<i64>(),
        );
        vec_znx_rotate_assign::<Self>(p, res, res_col, tmp);
    }

    fn vec_znx_automorphism_default(
        _module: &Module<Self>,
        p: i64,
        res: &mut VecZnxBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, Self>,
        a_col: usize,
    ) where
        Self: ZnxAutomorphism + ZnxZero,
        for<'x> Self::BufMut<'x>: HostDataMut,
        for<'x> Self::BufRef<'x>: poulpy_hal::layouts::HostDataRef,
    {
        vec_znx_automorphism::<Self>(p, res, res_col, a, a_col);
    }

    fn vec_znx_automorphism_assign_tmp_bytes_default(module: &Module<Self>) -> usize {
        vec_znx_automorphism_assign_tmp_bytes(module.n())
    }

    fn vec_znx_automorphism_assign_default(
        module: &Module<Self>,
        p: i64,
        res: &mut VecZnxBackendMut<'_, Self>,
        res_col: usize,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        Self: ZnxAutomorphism + ZnxCopy,
        for<'x> Self::BufMut<'x>: HostDataMut,
        for<'x> Self::BufMut<'x>: HostBufMut<'x>,
    {
        let (tmp, _) = take_host_typed::<Self, i64>(
            scratch.borrow(),
            vec_znx_automorphism_assign_tmp_bytes(module.n()) / size_of::<i64>(),
        );
        vec_znx_automorphism_assign::<Self>(p, res, res_col, tmp);
    }

    fn vec_znx_mul_xp_minus_one_assign_tmp_bytes_default(module: &Module<Self>) -> usize {
        vec_znx_mul_xp_minus_one_assign_tmp_bytes(module.n())
    }

    /// CPU override of [`poulpy_hal::oep::vec_znx_mul_xp_minus_one_assign_derived`]:
    /// a per-limb rotate through the one-limb temporary that
    /// [`Self::vec_znx_mul_xp_minus_one_assign_tmp_bytes_default`] reports,
    /// bit-exact with the default.
    fn vec_znx_mul_xp_minus_one_assign_default(
        module: &Module<Self>,
        p: i64,
        res: &mut VecZnxBackendMut<'_, Self>,
        res_col: usize,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        Self: ZnxRotate + ZnxNegate + ZnxSubNegateAssign,
        for<'x> Self::BufMut<'x>: HostDataMut,
        for<'x> Self::BufMut<'x>: HostBufMut<'x>,
    {
        let (tmp, _) = take_host_typed::<Self, i64>(
            scratch.borrow(),
            vec_znx_mul_xp_minus_one_assign_tmp_bytes(module.n()) / size_of::<i64>(),
        );
        vec_znx_mul_xp_minus_one_assign::<Self>(p, res, res_col, tmp);
    }

    fn vec_znx_switch_ring_default(
        _module: &Module<Self>,
        res: &mut VecZnxBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, Self>,
        a_col: usize,
    ) where
        Self: ZnxCopy + ZnxSwitchRing + ZnxZero,
        for<'x> Self::BufMut<'x>: HostDataMut,
        for<'x> Self::BufRef<'x>: poulpy_hal::layouts::HostDataRef,
    {
        vec_znx_switch_ring::<Self>(res, res_col, a, a_col);
    }

    fn vec_znx_copy_default(
        _module: &Module<Self>,
        res: &mut VecZnxBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, Self>,
        a_col: usize,
    ) where
        Self: ZnxCopy + ZnxZero,
        for<'x> Self::BufMut<'x>: HostDataMut,
        for<'x> Self::BufRef<'x>: poulpy_hal::layouts::HostDataRef,
    {
        vec_znx_copy::<Self>(res, res_col, a, a_col);
    }

    fn vec_znx_fill_uniform_default(
        _module: &Module<Self>,
        base2k: usize,
        k: usize,
        res: &mut VecZnxBackendMut<'_, Self>,
        res_col: usize,
        seed: [u8; 32],
    ) where
        for<'x> Self::BufMut<'x>: HostDataMut,
    {
        let mut source = Source::new(seed);
        vec_znx_fill_uniform_ref::<Self>(base2k, k, res, res_col, &mut source);
    }
}

impl<BE: Backend<ZnxWord = i64>> HalVecZnxDefault for BE where BE::OwnedBuf: poulpy_hal::layouts::HostDataMut {}
