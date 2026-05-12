//! Backend extension points for extended-precision [`VecZnxBig`](poulpy_hal::layouts::VecZnxBig) operations.

#![allow(clippy::too_many_arguments)]

use std::mem::size_of;

use crate::reference::{
    fft64::vec_znx_big::{
        vec_znx_big_add_assign as fft64_vec_znx_big_add_assign, vec_znx_big_add_into as fft64_vec_znx_big_add_into,
        vec_znx_big_add_normal_ref as fft64_vec_znx_big_add_normal_ref,
        vec_znx_big_add_small_assign as fft64_vec_znx_big_add_small_assign,
        vec_znx_big_add_small_into as fft64_vec_znx_big_add_small_into,
        vec_znx_big_automorphism as fft64_vec_znx_big_automorphism,
        vec_znx_big_automorphism_assign as fft64_vec_znx_big_automorphism_assign,
        vec_znx_big_automorphism_assign_tmp_bytes as fft64_vec_znx_big_automorphism_assign_tmp_bytes,
        vec_znx_big_negate as fft64_vec_znx_big_negate, vec_znx_big_negate_assign as fft64_vec_znx_big_negate_assign,
        vec_znx_big_normalize as fft64_vec_znx_big_normalize,
        vec_znx_big_normalize_tmp_bytes as fft64_vec_znx_big_normalize_tmp_bytes, vec_znx_big_sub as fft64_vec_znx_big_sub,
        vec_znx_big_sub_assign as fft64_vec_znx_big_sub_assign,
        vec_znx_big_sub_negate_assign as fft64_vec_znx_big_sub_negate_assign,
        vec_znx_big_sub_small_a as fft64_vec_znx_big_sub_small_a,
        vec_znx_big_sub_small_a_assign as fft64_vec_znx_big_sub_small_a_assign,
        vec_znx_big_sub_small_b as fft64_vec_znx_big_sub_small_b,
        vec_znx_big_sub_small_b_assign as fft64_vec_znx_big_sub_small_b_assign,
    },
    ntt120::vec_znx_big::{
        I128BigOps, I128NormalizeOps, ntt120_vec_znx_big_add_assign, ntt120_vec_znx_big_add_into,
        ntt120_vec_znx_big_add_normal_ref, ntt120_vec_znx_big_add_small_assign, ntt120_vec_znx_big_add_small_into,
        ntt120_vec_znx_big_automorphism, ntt120_vec_znx_big_automorphism_assign,
        ntt120_vec_znx_big_automorphism_assign_tmp_bytes, ntt120_vec_znx_big_from_small, ntt120_vec_znx_big_negate,
        ntt120_vec_znx_big_negate_assign, ntt120_vec_znx_big_normalize, ntt120_vec_znx_big_normalize_add_assign,
        ntt120_vec_znx_big_normalize_sub_assign, ntt120_vec_znx_big_normalize_tmp_bytes, ntt120_vec_znx_big_sub,
        ntt120_vec_znx_big_sub_assign, ntt120_vec_znx_big_sub_negate_assign, ntt120_vec_znx_big_sub_small_a,
        ntt120_vec_znx_big_sub_small_assign, ntt120_vec_znx_big_sub_small_b, ntt120_vec_znx_big_sub_small_negate_assign,
    },
    znx::{
        ZnxAdd, ZnxAddAssign, ZnxAutomorphism, ZnxCopy, ZnxExtractDigitAddMul, ZnxMulPowerOfTwoAssign, ZnxNegate,
        ZnxNegateAssign, ZnxNormalizeDigit, ZnxNormalizeFinalStep, ZnxNormalizeFinalStepAssign, ZnxNormalizeFirstStep,
        ZnxNormalizeFirstStepCarryOnly, ZnxNormalizeMiddleStep, ZnxNormalizeMiddleStepAssign, ZnxNormalizeMiddleStepCarryOnly,
        ZnxSub, ZnxSubAssign, ZnxSubNegateAssign, ZnxZero, znx_copy_ref, znx_zero_ref,
    },
};
use poulpy_hal::{
    api::HostBufMut,
    layouts::{
        Backend, HostDataMut, HostDataRef, Module, NoiseInfos, ScratchArena, VecZnx, VecZnxBackendRef, VecZnxBigToBackendMut,
        VecZnxBigToBackendRef, VecZnxToBackendMut, ZnxView, ZnxViewMut,
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

#[inline]
fn vec_znx_backend_ref_as_host_ref<'a, 'b, BE>(a: &'a VecZnx<BE::BufRef<'b>>) -> VecZnx<&'a [u8]>
where
    BE: Backend + 'b,
    BE::BufRef<'b>: AsRef<[u8]>,
{
    VecZnx::from_data_with_max_size(a.data.as_ref(), a.n(), a.cols(), a.size(), a.max_size())
}

#[doc(hidden)]
pub trait FFT64VecZnxBigDefault<BE: Backend>: Backend
where
    BE::OwnedBuf: poulpy_hal::layouts::HostDataMut,
    for<'a> BE::BufMut<'a>: HostDataMut,
    for<'a> BE::BufRef<'a>: HostDataRef,
{
    fn vec_znx_big_from_small_default<R>(res: &mut R, res_col: usize, a: &VecZnxBackendRef<'_, BE>, a_col: usize)
    where
        BE: Backend<ScalarBig = i64>,
        for<'a> BE::BufMut<'a>: HostDataMut,
        for<'a> BE::BufRef<'a>: AsRef<[u8]>,
        R: VecZnxBigToBackendMut<BE>,
    {
        let mut res = res.to_backend_mut();
        let a: VecZnx<&[u8]> = vec_znx_backend_ref_as_host_ref::<BE>(a);

        let res_size = res.size();
        let a_size = a.size();
        let min_size = res_size.min(a_size);

        for j in 0..min_size {
            znx_copy_ref(res.at_mut(res_col, j), a.at(a_col, j));
        }

        for j in min_size..res_size {
            znx_zero_ref(res.at_mut(res_col, j));
        }
    }

    fn vec_znx_big_add_normal_default<R>(
        _module: &Module<BE>,
        res_base2k: usize,
        res: &mut R,
        res_col: usize,
        noise_infos: NoiseInfos,
        source: &mut Source,
    ) where
        BE: Backend<ScalarBig = i64>,
        R: VecZnxBigToBackendMut<BE>,
    {
        fft64_vec_znx_big_add_normal_ref(res_base2k, res, res_col, noise_infos, source);
    }

    fn vec_znx_big_add_normal_seed_default<R>(
        module: &Module<BE>,
        res_base2k: usize,
        res: &mut R,
        res_col: usize,
        noise_infos: NoiseInfos,
        seed: [u8; 32],
    ) where
        BE: Backend<ScalarBig = i64>,
        R: VecZnxBigToBackendMut<BE>,
    {
        let mut source = Source::new(seed);
        Self::vec_znx_big_add_normal_default(module, res_base2k, res, res_col, noise_infos, &mut source);
    }

    fn vec_znx_big_add_into_default<R, A, C>(
        _module: &Module<BE>,
        res: &mut R,
        res_col: usize,
        a: &A,
        a_col: usize,
        b: &C,
        b_col: usize,
    ) where
        BE: Backend<ScalarBig = i64> + ZnxAdd + ZnxCopy + ZnxZero,
        for<'a> BE::BufRef<'a>: AsRef<[u8]>,
        R: VecZnxBigToBackendMut<BE>,
        A: VecZnxBigToBackendRef<BE>,
        C: VecZnxBigToBackendRef<BE>,
    {
        fft64_vec_znx_big_add_into(res, res_col, a, a_col, b, b_col);
    }

    fn vec_znx_big_add_assign_default<R, A>(_module: &Module<BE>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        BE: Backend<ScalarBig = i64> + ZnxAddAssign,
        R: VecZnxBigToBackendMut<BE>,
        A: VecZnxBigToBackendRef<BE>,
    {
        fft64_vec_znx_big_add_assign(res, res_col, a, a_col);
    }

    fn vec_znx_big_add_small_into_default<R, A>(
        _module: &Module<BE>,
        res: &mut R,
        res_col: usize,
        a: &A,
        a_col: usize,
        b: &VecZnxBackendRef<'_, BE>,
        b_col: usize,
    ) where
        BE: Backend<ScalarBig = i64> + ZnxAdd + ZnxCopy + ZnxZero,
        for<'a> BE::BufRef<'a>: AsRef<[u8]>,
        R: VecZnxBigToBackendMut<BE>,
        A: VecZnxBigToBackendRef<BE>,
    {
        fft64_vec_znx_big_add_small_into(res, res_col, a, a_col, &b, b_col);
    }

    fn vec_znx_big_add_small_assign_default<R>(
        _module: &Module<BE>,
        res: &mut R,
        res_col: usize,
        a: &VecZnxBackendRef<'_, BE>,
        a_col: usize,
    ) where
        BE: Backend<ScalarBig = i64> + ZnxAddAssign,
        for<'a> BE::BufRef<'a>: AsRef<[u8]>,
        R: VecZnxBigToBackendMut<BE>,
    {
        fft64_vec_znx_big_add_small_assign(res, res_col, &a, a_col);
    }

    fn vec_znx_big_sub_default<R, A, C>(
        _module: &Module<BE>,
        res: &mut R,
        res_col: usize,
        a: &A,
        a_col: usize,
        b: &C,
        b_col: usize,
    ) where
        BE: Backend<ScalarBig = i64> + ZnxSub + ZnxNegate + ZnxZero + ZnxCopy,
        R: VecZnxBigToBackendMut<BE>,
        A: VecZnxBigToBackendRef<BE>,
        C: VecZnxBigToBackendRef<BE>,
    {
        fft64_vec_znx_big_sub(res, res_col, a, a_col, b, b_col);
    }

    fn vec_znx_big_sub_assign_default<R, A>(_module: &Module<BE>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        BE: Backend<ScalarBig = i64> + ZnxSubAssign,
        R: VecZnxBigToBackendMut<BE>,
        A: VecZnxBigToBackendRef<BE>,
    {
        fft64_vec_znx_big_sub_assign(res, res_col, a, a_col);
    }

    fn vec_znx_big_sub_negate_assign_default<R, A>(_module: &Module<BE>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        BE: Backend<ScalarBig = i64> + ZnxSubNegateAssign + ZnxNegateAssign,
        R: VecZnxBigToBackendMut<BE>,
        A: VecZnxBigToBackendRef<BE>,
    {
        fft64_vec_znx_big_sub_negate_assign(res, res_col, a, a_col);
    }

    fn vec_znx_big_sub_small_a_default<R, C>(
        _module: &Module<BE>,
        res: &mut R,
        res_col: usize,
        a: &VecZnxBackendRef<'_, BE>,
        a_col: usize,
        b: &C,
        b_col: usize,
    ) where
        BE: Backend<ScalarBig = i64> + ZnxSub + ZnxNegate + ZnxZero + ZnxCopy,
        for<'a> BE::BufRef<'a>: AsRef<[u8]>,
        R: VecZnxBigToBackendMut<BE>,
        C: VecZnxBigToBackendRef<BE>,
    {
        fft64_vec_znx_big_sub_small_a(res, res_col, &a, a_col, b, b_col);
    }

    fn vec_znx_big_sub_small_assign_default<R>(
        _module: &Module<BE>,
        res: &mut R,
        res_col: usize,
        a: &VecZnxBackendRef<'_, BE>,
        a_col: usize,
    ) where
        BE: Backend<ScalarBig = i64> + ZnxSubAssign,
        for<'a> BE::BufRef<'a>: AsRef<[u8]>,
        R: VecZnxBigToBackendMut<BE>,
    {
        fft64_vec_znx_big_sub_small_a_assign(res, res_col, &a, a_col);
    }

    fn vec_znx_big_sub_small_b_default<R, A>(
        _module: &Module<BE>,
        res: &mut R,
        res_col: usize,
        a: &A,
        a_col: usize,
        b: &VecZnxBackendRef<'_, BE>,
        b_col: usize,
    ) where
        BE: Backend<ScalarBig = i64> + ZnxSub + ZnxNegate + ZnxZero + ZnxCopy,
        for<'a> BE::BufRef<'a>: AsRef<[u8]>,
        R: VecZnxBigToBackendMut<BE>,
        A: VecZnxBigToBackendRef<BE>,
    {
        fft64_vec_znx_big_sub_small_b(res, res_col, a, a_col, &b, b_col);
    }

    fn vec_znx_big_sub_small_negate_assign_default<R>(
        _module: &Module<BE>,
        res: &mut R,
        res_col: usize,
        a: &VecZnxBackendRef<'_, BE>,
        a_col: usize,
    ) where
        BE: Backend<ScalarBig = i64> + ZnxSubNegateAssign + ZnxNegateAssign,
        for<'a> BE::BufRef<'a>: AsRef<[u8]>,
        R: VecZnxBigToBackendMut<BE>,
    {
        fft64_vec_znx_big_sub_small_b_assign(res, res_col, &a, a_col);
    }

    fn vec_znx_big_negate_default<R, A>(_module: &Module<BE>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        BE: Backend<ScalarBig = i64> + ZnxNegate + ZnxZero,
        R: VecZnxBigToBackendMut<BE>,
        A: VecZnxBigToBackendRef<BE>,
    {
        fft64_vec_znx_big_negate(res, res_col, a, a_col);
    }

    fn vec_znx_big_negate_assign_default<R>(_module: &Module<BE>, res: &mut R, res_col: usize)
    where
        BE: Backend<ScalarBig = i64> + ZnxNegateAssign,
        R: VecZnxBigToBackendMut<BE>,
    {
        fft64_vec_znx_big_negate_assign(res, res_col);
    }

    fn vec_znx_big_normalize_tmp_bytes_default(module: &Module<BE>) -> usize
    where
        BE: Backend<ScalarBig = i64>,
    {
        fft64_vec_znx_big_normalize_tmp_bytes(module.n())
    }

    fn vec_znx_big_normalize_default<'s, R, A>(
        module: &Module<BE>,
        res: &mut R,
        res_base2k: usize,
        res_offset: i64,
        res_col: usize,
        a: &A,
        a_base2k: usize,
        a_col: usize,
        scratch: &'s mut ScratchArena<'s, BE>,
    ) where
        BE: Backend<ScalarBig = i64>
            + ZnxZero
            + ZnxCopy
            + ZnxAddAssign
            + ZnxMulPowerOfTwoAssign
            + ZnxNormalizeFirstStepCarryOnly
            + ZnxNormalizeMiddleStepCarryOnly
            + ZnxNormalizeMiddleStep
            + ZnxNormalizeFinalStep
            + ZnxNormalizeFirstStep
            + ZnxExtractDigitAddMul
            + ZnxNormalizeDigit
            + ZnxNormalizeMiddleStepAssign
            + ZnxNormalizeFinalStepAssign,
        BE::BufMut<'s>: HostBufMut<'s>,
        R: VecZnxToBackendMut<BE>,
        A: VecZnxBigToBackendRef<BE>,
    {
        let (carry, _) = take_host_typed::<BE, i64>(
            scratch.borrow(),
            fft64_vec_znx_big_normalize_tmp_bytes(module.n()) / size_of::<i64>(),
        );
        fft64_vec_znx_big_normalize(res, res_base2k, res_offset, res_col, a, a_base2k, a_col, carry);
    }

    fn vec_znx_big_automorphism_default<R, A>(_module: &Module<BE>, k: i64, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        BE: Backend<ScalarBig = i64> + ZnxAutomorphism + ZnxZero,
        R: VecZnxBigToBackendMut<BE>,
        A: VecZnxBigToBackendRef<BE>,
    {
        fft64_vec_znx_big_automorphism(k, res, res_col, a, a_col);
    }

    fn vec_znx_big_automorphism_assign_tmp_bytes_default(module: &Module<BE>) -> usize
    where
        BE: Backend<ScalarBig = i64>,
    {
        fft64_vec_znx_big_automorphism_assign_tmp_bytes(module.n())
    }

    fn vec_znx_big_automorphism_assign_default<'s, R>(
        module: &Module<BE>,
        k: i64,
        res: &mut R,
        res_col: usize,
        scratch: &'s mut ScratchArena<'s, BE>,
    ) where
        BE: Backend<ScalarBig = i64> + ZnxAutomorphism + ZnxCopy,
        BE::BufMut<'s>: HostBufMut<'s>,
        R: VecZnxBigToBackendMut<BE>,
    {
        let (tmp, _) = take_host_typed::<BE, i64>(
            scratch.borrow(),
            fft64_vec_znx_big_automorphism_assign_tmp_bytes(module.n()) / size_of::<i64>(),
        );
        fft64_vec_znx_big_automorphism_assign(k, res, res_col, tmp);
    }
}

impl<BE: Backend> FFT64VecZnxBigDefault<BE> for BE
where
    BE::OwnedBuf: poulpy_hal::layouts::HostDataMut,
    for<'a> BE::BufMut<'a>: HostDataMut,
    for<'a> BE::BufRef<'a>: HostDataRef,
{
}

#[doc(hidden)]
pub trait NTT120VecZnxBigDefault<BE: Backend>: Backend
where
    BE::OwnedBuf: poulpy_hal::layouts::HostDataMut,
    for<'a> BE::BufMut<'a>: HostDataMut,
    for<'a> BE::BufRef<'a>: HostDataRef,
{
    fn vec_znx_big_from_small_default<R>(res: &mut R, res_col: usize, a: &VecZnxBackendRef<'_, BE>, a_col: usize)
    where
        BE: Backend<ScalarBig = i128> + I128BigOps,
        for<'a> BE::BufRef<'a>: AsRef<[u8]>,
        R: VecZnxBigToBackendMut<BE>,
    {
        let a = vec_znx_backend_ref_as_host_ref::<BE>(a);
        ntt120_vec_znx_big_from_small(res, res_col, &a, a_col);
    }

    fn vec_znx_big_add_normal_default<R>(
        _module: &Module<BE>,
        res_base2k: usize,
        res: &mut R,
        res_col: usize,
        noise_infos: NoiseInfos,
        source: &mut Source,
    ) where
        BE: Backend<ScalarBig = i128>,
        R: VecZnxBigToBackendMut<BE>,
    {
        ntt120_vec_znx_big_add_normal_ref(res_base2k, res, res_col, noise_infos, source);
    }

    fn vec_znx_big_add_normal_seed_default<R>(
        module: &Module<BE>,
        res_base2k: usize,
        res: &mut R,
        res_col: usize,
        noise_infos: NoiseInfos,
        seed: [u8; 32],
    ) where
        BE: Backend<ScalarBig = i128>,
        R: VecZnxBigToBackendMut<BE>,
    {
        let mut source = Source::new(seed);
        Self::vec_znx_big_add_normal_default(module, res_base2k, res, res_col, noise_infos, &mut source);
    }

    fn vec_znx_big_add_into_default<R, A, C>(
        _module: &Module<BE>,
        res: &mut R,
        res_col: usize,
        a: &A,
        a_col: usize,
        b: &C,
        b_col: usize,
    ) where
        BE: Backend<ScalarBig = i128> + I128BigOps,
        for<'a> BE::BufRef<'a>: AsRef<[u8]>,
        R: VecZnxBigToBackendMut<BE>,
        A: VecZnxBigToBackendRef<BE>,
        C: VecZnxBigToBackendRef<BE>,
    {
        ntt120_vec_znx_big_add_into(res, res_col, a, a_col, b, b_col);
    }

    fn vec_znx_big_add_assign_default<R, A>(_module: &Module<BE>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        BE: Backend<ScalarBig = i128> + I128BigOps,
        R: VecZnxBigToBackendMut<BE>,
        A: VecZnxBigToBackendRef<BE>,
    {
        ntt120_vec_znx_big_add_assign(res, res_col, a, a_col);
    }

    fn vec_znx_big_add_small_into_default<R, A>(
        _module: &Module<BE>,
        res: &mut R,
        res_col: usize,
        a: &A,
        a_col: usize,
        b: &VecZnxBackendRef<'_, BE>,
        b_col: usize,
    ) where
        BE: Backend<ScalarBig = i128> + I128BigOps,
        for<'a> BE::BufRef<'a>: AsRef<[u8]>,
        R: VecZnxBigToBackendMut<BE>,
        A: VecZnxBigToBackendRef<BE>,
    {
        let b = vec_znx_backend_ref_as_host_ref::<BE>(b);
        ntt120_vec_znx_big_add_small_into(res, res_col, a, a_col, &b, b_col);
    }

    fn vec_znx_big_add_small_assign_default<R>(
        _module: &Module<BE>,
        res: &mut R,
        res_col: usize,
        a: &VecZnxBackendRef<'_, BE>,
        a_col: usize,
    ) where
        BE: Backend<ScalarBig = i128> + I128BigOps,
        for<'a> BE::BufRef<'a>: AsRef<[u8]>,
        R: VecZnxBigToBackendMut<BE>,
    {
        let a = vec_znx_backend_ref_as_host_ref::<BE>(a);
        ntt120_vec_znx_big_add_small_assign(res, res_col, &a, a_col);
    }

    fn vec_znx_big_sub_default<R, A, C>(
        _module: &Module<BE>,
        res: &mut R,
        res_col: usize,
        a: &A,
        a_col: usize,
        b: &C,
        b_col: usize,
    ) where
        BE: Backend<ScalarBig = i128> + I128BigOps,
        R: VecZnxBigToBackendMut<BE>,
        A: VecZnxBigToBackendRef<BE>,
        C: VecZnxBigToBackendRef<BE>,
    {
        ntt120_vec_znx_big_sub(res, res_col, a, a_col, b, b_col);
    }

    fn vec_znx_big_sub_assign_default<R, A>(_module: &Module<BE>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        BE: Backend<ScalarBig = i128> + I128BigOps,
        R: VecZnxBigToBackendMut<BE>,
        A: VecZnxBigToBackendRef<BE>,
    {
        ntt120_vec_znx_big_sub_assign(res, res_col, a, a_col);
    }

    fn vec_znx_big_sub_negate_assign_default<R, A>(_module: &Module<BE>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        BE: Backend<ScalarBig = i128> + I128BigOps,
        R: VecZnxBigToBackendMut<BE>,
        A: VecZnxBigToBackendRef<BE>,
    {
        ntt120_vec_znx_big_sub_negate_assign(res, res_col, a, a_col);
    }

    fn vec_znx_big_sub_small_a_default<R, C>(
        _module: &Module<BE>,
        res: &mut R,
        res_col: usize,
        a: &VecZnxBackendRef<'_, BE>,
        a_col: usize,
        b: &C,
        b_col: usize,
    ) where
        BE: Backend<ScalarBig = i128> + I128BigOps,
        for<'a> BE::BufRef<'a>: AsRef<[u8]>,
        R: VecZnxBigToBackendMut<BE>,
        C: VecZnxBigToBackendRef<BE>,
    {
        let a = vec_znx_backend_ref_as_host_ref::<BE>(a);
        ntt120_vec_znx_big_sub_small_a(res, res_col, &a, a_col, b, b_col);
    }

    fn vec_znx_big_sub_small_assign_default<R>(
        _module: &Module<BE>,
        res: &mut R,
        res_col: usize,
        a: &VecZnxBackendRef<'_, BE>,
        a_col: usize,
    ) where
        BE: Backend<ScalarBig = i128> + I128BigOps,
        for<'a> BE::BufRef<'a>: AsRef<[u8]>,
        R: VecZnxBigToBackendMut<BE>,
    {
        let a = vec_znx_backend_ref_as_host_ref::<BE>(a);
        ntt120_vec_znx_big_sub_small_assign(res, res_col, &a, a_col);
    }

    fn vec_znx_big_sub_small_b_default<R, A>(
        _module: &Module<BE>,
        res: &mut R,
        res_col: usize,
        a: &A,
        a_col: usize,
        b: &VecZnxBackendRef<'_, BE>,
        b_col: usize,
    ) where
        BE: Backend<ScalarBig = i128> + I128BigOps,
        for<'a> BE::BufRef<'a>: AsRef<[u8]>,
        R: VecZnxBigToBackendMut<BE>,
        A: VecZnxBigToBackendRef<BE>,
    {
        let b = vec_znx_backend_ref_as_host_ref::<BE>(b);
        ntt120_vec_znx_big_sub_small_b(res, res_col, a, a_col, &b, b_col);
    }

    fn vec_znx_big_sub_small_negate_assign_default<R>(
        _module: &Module<BE>,
        res: &mut R,
        res_col: usize,
        a: &VecZnxBackendRef<'_, BE>,
        a_col: usize,
    ) where
        BE: Backend<ScalarBig = i128> + I128BigOps,
        for<'a> BE::BufRef<'a>: AsRef<[u8]>,
        R: VecZnxBigToBackendMut<BE>,
    {
        let a = vec_znx_backend_ref_as_host_ref::<BE>(a);
        ntt120_vec_znx_big_sub_small_negate_assign(res, res_col, &a, a_col);
    }

    fn vec_znx_big_negate_default<R, A>(_module: &Module<BE>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        BE: Backend<ScalarBig = i128> + I128BigOps,
        R: VecZnxBigToBackendMut<BE>,
        A: VecZnxBigToBackendRef<BE>,
    {
        ntt120_vec_znx_big_negate(res, res_col, a, a_col);
    }

    fn vec_znx_big_negate_assign_default<R>(_module: &Module<BE>, res: &mut R, res_col: usize)
    where
        BE: Backend<ScalarBig = i128> + I128BigOps,
        R: VecZnxBigToBackendMut<BE>,
    {
        ntt120_vec_znx_big_negate_assign(res, res_col);
    }

    fn vec_znx_big_normalize_tmp_bytes_default(module: &Module<BE>) -> usize
    where
        BE: Backend<ScalarBig = i128> + I128NormalizeOps,
    {
        ntt120_vec_znx_big_normalize_tmp_bytes(module.n())
    }

    fn vec_znx_big_normalize_default<'s, R, A>(
        module: &Module<BE>,
        res: &mut R,
        res_base2k: usize,
        res_offset: i64,
        res_col: usize,
        a: &A,
        a_base2k: usize,
        a_col: usize,
        scratch: &'s mut ScratchArena<'s, BE>,
    ) where
        BE: Backend<ScalarBig = i128> + I128NormalizeOps,
        BE::BufMut<'s>: HostBufMut<'s>,
        R: VecZnxToBackendMut<BE>,
        A: VecZnxBigToBackendRef<BE>,
    {
        let (carry, _) = take_host_typed::<BE, i128>(
            scratch.borrow(),
            ntt120_vec_znx_big_normalize_tmp_bytes(module.n()) / size_of::<i128>(),
        );
        ntt120_vec_znx_big_normalize(res, res_base2k, res_offset, res_col, a, a_base2k, a_col, carry);
    }

    fn vec_znx_big_normalize_add_assign_default<'s, R, A>(
        module: &Module<BE>,
        res: &mut R,
        res_base2k: usize,
        res_offset: i64,
        res_col: usize,
        a: &A,
        a_base2k: usize,
        a_col: usize,
        scratch: &'s mut ScratchArena<'s, BE>,
    ) where
        BE: Backend<ScalarBig = i128> + I128NormalizeOps,
        BE::BufMut<'s>: HostBufMut<'s>,
        R: VecZnxToBackendMut<BE>,
        A: VecZnxBigToBackendRef<BE>,
    {
        let (carry, _) = take_host_typed::<BE, i128>(
            scratch.borrow(),
            ntt120_vec_znx_big_normalize_tmp_bytes(module.n()) / size_of::<i128>(),
        );
        ntt120_vec_znx_big_normalize_add_assign(res, res_base2k, res_offset, res_col, a, a_base2k, a_col, carry);
    }

    fn vec_znx_big_normalize_sub_assign_default<'s, R, A>(
        module: &Module<BE>,
        res: &mut R,
        res_base2k: usize,
        res_offset: i64,
        res_col: usize,
        a: &A,
        a_base2k: usize,
        a_col: usize,
        scratch: &'s mut ScratchArena<'s, BE>,
    ) where
        BE: Backend<ScalarBig = i128> + I128NormalizeOps,
        BE::BufMut<'s>: HostBufMut<'s>,
        R: VecZnxToBackendMut<BE>,
        A: VecZnxBigToBackendRef<BE>,
    {
        let (carry, _) = take_host_typed::<BE, i128>(
            scratch.borrow(),
            ntt120_vec_znx_big_normalize_tmp_bytes(module.n()) / size_of::<i128>(),
        );
        ntt120_vec_znx_big_normalize_sub_assign(res, res_base2k, res_offset, res_col, a, a_base2k, a_col, carry);
    }

    fn vec_znx_big_automorphism_default<R, A>(_module: &Module<BE>, k: i64, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        BE: Backend<ScalarBig = i128> + I128BigOps,
        R: VecZnxBigToBackendMut<BE>,
        A: VecZnxBigToBackendRef<BE>,
    {
        ntt120_vec_znx_big_automorphism(k, res, res_col, a, a_col);
    }

    fn vec_znx_big_automorphism_assign_tmp_bytes_default(module: &Module<BE>) -> usize
    where
        BE: Backend<ScalarBig = i128> + I128BigOps,
    {
        ntt120_vec_znx_big_automorphism_assign_tmp_bytes(module.n())
    }

    fn vec_znx_big_automorphism_assign_default<'s, R>(
        module: &Module<BE>,
        k: i64,
        res: &mut R,
        res_col: usize,
        scratch: &'s mut ScratchArena<'s, BE>,
    ) where
        BE: Backend<ScalarBig = i128> + I128BigOps,
        BE::BufMut<'s>: HostBufMut<'s>,
        R: VecZnxBigToBackendMut<BE>,
    {
        let (tmp, _) = take_host_typed::<BE, i128>(
            scratch.borrow(),
            ntt120_vec_znx_big_automorphism_assign_tmp_bytes(module.n()) / size_of::<i128>(),
        );
        ntt120_vec_znx_big_automorphism_assign(k, res, res_col, tmp);
    }
}

impl<BE: Backend> NTT120VecZnxBigDefault<BE> for BE
where
    BE::OwnedBuf: poulpy_hal::layouts::HostDataMut,
    for<'a> BE::BufMut<'a>: HostDataMut,
    for<'a> BE::BufRef<'a>: HostDataRef,
{
}
