use std::collections::HashMap;

use poulpy_hal::layouts::{Backend, DataMut, DataRef, Module, Scratch};

use crate::{
    ScratchTakeCore,
    glwe_packer::{GLWEPacker, GLWEPackerOpsDefault},
    glwe_packing::GLWEPackingDefault,
    glwe_trace::GLWETraceDefault,
    layouts::{
        GGLWEInfos, GGLWEPreparedToRef, GGSWToMut, GGSWToRef, GLWE, GLWEAutomorphismKeyHelper, GLWEInfos, GLWEPlaintext,
        GLWETensor, GLWETensorKeyPrepared, GLWEToMut, GLWEToRef, GetGaloisElement,
    },
    operations::{
        GGSWRotateDefault, GLWEMulConstDefault, GLWEMulPlainDefault, GLWEMulXpMinusOneDefault, GLWENormalizeDefault,
        GLWERotateDefault, GLWEShiftDefault, GLWETensoringDefault,
    },
};

#[doc(hidden)]
pub trait CoreOperationsDefaults<BE: Backend>: Backend {
    fn glwe_mul_const_tmp_bytes_default<R, A>(module: &Module<BE>, res: &R, a: &A, b_size: usize) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos;

    fn glwe_mul_const_default<R, A>(
        module: &Module<BE>,
        cnv_offset: usize,
        res: &mut GLWE<R>,
        a: &GLWE<A>,
        b: &[i64],
        scratch: &mut Scratch<BE>,
    ) where
        R: DataMut,
        A: DataRef;

    fn glwe_mul_const_assign_default<R>(
        module: &Module<BE>,
        cnv_offset: usize,
        res: &mut GLWE<R>,
        b: &[i64],
        scratch: &mut Scratch<BE>,
    ) where
        R: DataMut;

    fn glwe_mul_plain_tmp_bytes_default<R, A, B>(module: &Module<BE>, res: &R, a: &A, b: &B) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        B: GLWEInfos;

    #[allow(clippy::too_many_arguments)]
    fn glwe_mul_plain_default<R, A, B>(
        module: &Module<BE>,
        cnv_offset: usize,
        res: &mut GLWE<R>,
        a: &GLWE<A>,
        a_effective_k: usize,
        b: &GLWEPlaintext<B>,
        b_effective_k: usize,
        scratch: &mut Scratch<BE>,
    ) where
        R: DataMut,
        A: DataRef,
        B: DataRef;

    #[allow(clippy::too_many_arguments)]
    fn glwe_tensor_apply_add_assign_default<R, A, B>(
        module: &Module<BE>,
        cnv_offset: usize,
        res: &mut GLWETensor<R>,
        a: &GLWE<A>,
        a_effective_k: usize,
        b: &GLWE<B>,
        b_effective_k: usize,
        scratch: &mut Scratch<BE>,
    ) where
        R: DataMut,
        A: DataRef,
        B: DataRef;

    #[allow(clippy::too_many_arguments)]
    fn glwe_mul_plain_assign_default<R, A>(
        module: &Module<BE>,
        cnv_offset: usize,
        res: &mut GLWE<R>,
        res_effective_k: usize,
        a: &GLWEPlaintext<A>,
        a_effective_k: usize,
        scratch: &mut Scratch<BE>,
    ) where
        R: DataMut,
        A: DataRef;

    fn glwe_tensor_apply_tmp_bytes_default<R, A, B>(module: &Module<BE>, res: &R, a: &A, b: &B) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        B: GLWEInfos;

    fn glwe_tensor_square_apply_tmp_bytes_default<R, A>(module: &Module<BE>, res: &R, a: &A) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos;

    #[allow(clippy::too_many_arguments)]
    fn glwe_tensor_apply_default<R, A, B>(
        module: &Module<BE>,
        cnv_offset: usize,
        res: &mut GLWETensor<R>,
        a: &GLWE<A>,
        a_effective_k: usize,
        b: &GLWE<B>,
        b_effective_k: usize,
        scratch: &mut Scratch<BE>,
    ) where
        R: DataMut,
        A: DataRef,
        B: DataRef;

    #[allow(clippy::too_many_arguments)]
    fn glwe_tensor_square_apply_default<R, A>(
        module: &Module<BE>,
        cnv_offset: usize,
        res: &mut GLWETensor<R>,
        a: &GLWE<A>,
        a_effective_k: usize,
        scratch: &mut Scratch<BE>,
    ) where
        R: DataMut,
        A: DataRef;

    fn glwe_tensor_relinearize_default<R, A, B>(
        module: &Module<BE>,
        res: &mut GLWE<R>,
        a: &GLWETensor<A>,
        tsk: &GLWETensorKeyPrepared<B, BE>,
        tsk_size: usize,
        scratch: &mut Scratch<BE>,
    ) where
        R: DataMut,
        A: DataRef,
        B: DataRef;

    fn glwe_tensor_relinearize_tmp_bytes_default<R, A, B>(module: &Module<BE>, res: &R, a: &A, tsk: &B) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        B: GGLWEInfos;

    fn glwe_rotate_tmp_bytes_default(module: &Module<BE>) -> usize;

    fn glwe_rotate_default<R, A>(module: &Module<BE>, k: i64, res: &mut R, a: &A)
    where
        R: GLWEToMut,
        A: GLWEToRef;

    fn glwe_rotate_assign_default<R>(module: &Module<BE>, k: i64, res: &mut R, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut,
        Scratch<BE>: ScratchTakeCore<BE>;

    fn ggsw_rotate_tmp_bytes_default(module: &Module<BE>) -> usize;

    fn ggsw_rotate_default<R, A>(module: &Module<BE>, k: i64, res: &mut R, a: &A)
    where
        R: GGSWToMut,
        A: GGSWToRef;

    fn ggsw_rotate_assign_default<R>(module: &Module<BE>, k: i64, res: &mut R, scratch: &mut Scratch<BE>)
    where
        R: GGSWToMut,
        Scratch<BE>: ScratchTakeCore<BE> + poulpy_hal::api::ScratchAvailable;

    fn glwe_mul_xp_minus_one_default<R, A>(module: &Module<BE>, k: i64, res: &mut R, a: &A)
    where
        R: GLWEToMut,
        A: GLWEToRef;

    fn glwe_mul_xp_minus_one_assign_default<R>(module: &Module<BE>, k: i64, res: &mut R, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut;

    fn glwe_shift_tmp_bytes_default(module: &Module<BE>) -> usize;

    fn glwe_rsh_default<R>(module: &Module<BE>, k: usize, res: &mut R, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut,
        Scratch<BE>: ScratchTakeCore<BE>;

    fn glwe_lsh_assign_default<R>(module: &Module<BE>, res: &mut R, k: usize, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut,
        Scratch<BE>: ScratchTakeCore<BE>;

    fn glwe_lsh_default<R, A>(module: &Module<BE>, res: &mut R, a: &A, k: usize, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut,
        A: GLWEToRef,
        Scratch<BE>: ScratchTakeCore<BE>;

    fn glwe_lsh_add_default<R, A>(module: &Module<BE>, res: &mut R, a: &A, k: usize, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut,
        A: GLWEToRef,
        Scratch<BE>: ScratchTakeCore<BE>;

    fn glwe_lsh_sub_default<R, A>(module: &Module<BE>, res: &mut R, a: &A, k: usize, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut,
        A: GLWEToRef,
        Scratch<BE>: ScratchTakeCore<BE>;

    fn glwe_normalize_tmp_bytes_default(module: &Module<BE>) -> usize;

    fn glwe_maybe_cross_normalize_to_ref_default<'a, A>(
        module: &Module<BE>,
        glwe: &'a A,
        target_base2k: usize,
        tmp_slot: &'a mut Option<GLWE<&'a mut [u8]>>,
        scratch: &'a mut Scratch<BE>,
    ) -> (GLWE<&'a [u8]>, &'a mut Scratch<BE>)
    where
        A: GLWEToRef + GLWEInfos,
        Scratch<BE>: ScratchTakeCore<BE>;

    fn glwe_maybe_cross_normalize_to_mut_default<'a, A>(
        module: &Module<BE>,
        glwe: &'a mut A,
        target_base2k: usize,
        tmp_slot: &'a mut Option<GLWE<&'a mut [u8]>>,
        scratch: &'a mut Scratch<BE>,
    ) -> (GLWE<&'a mut [u8]>, &'a mut Scratch<BE>)
    where
        A: GLWEToMut + GLWEInfos,
        Scratch<BE>: ScratchTakeCore<BE>;

    fn glwe_normalize_default<R, A>(module: &Module<BE>, res: &mut R, a: &A, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut,
        A: GLWEToRef,
        Scratch<BE>: ScratchTakeCore<BE>;

    fn glwe_normalize_assign_default<R>(module: &Module<BE>, res: &mut R, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut,
        Scratch<BE>: ScratchTakeCore<BE>;

    fn glwe_trace_galois_elements_default(module: &Module<BE>) -> Vec<i64>;

    fn glwe_trace_tmp_bytes_default<R, A, K>(module: &Module<BE>, res_infos: &R, a_infos: &A, key_infos: &K) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        K: GGLWEInfos;

    fn glwe_trace_default<R, A, K, H>(module: &Module<BE>, res: &mut R, skip: usize, a: &A, keys: &H, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut + GLWEInfos,
        A: GLWEToRef + GLWEInfos,
        K: GGLWEPreparedToRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>;

    fn glwe_trace_assign_default<R, K, H>(module: &Module<BE>, res: &mut R, skip: usize, keys: &H, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut,
        K: GGLWEPreparedToRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>;

    fn glwe_pack_galois_elements_default(module: &Module<BE>) -> Vec<i64>;

    fn glwe_pack_tmp_bytes_default<R, K>(module: &Module<BE>, res: &R, key: &K) -> usize
    where
        R: GLWEInfos,
        K: GGLWEInfos;

    fn glwe_pack_default<R, A, K, H>(
        module: &Module<BE>,
        res: &mut R,
        a: HashMap<usize, &mut A>,
        log_gap_out: usize,
        keys: &H,
        scratch: &mut Scratch<BE>,
    ) where
        R: GLWEToMut + GLWEInfos,
        A: GLWEToMut + GLWEInfos,
        K: GGLWEPreparedToRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>;

    fn packer_add_default<A, K, H>(
        module: &Module<BE>,
        packer: &mut GLWEPacker,
        a: Option<&A>,
        i: usize,
        auto_keys: &H,
        scratch: &mut Scratch<BE>,
    ) where
        A: GLWEToRef + GLWEInfos,
        K: GGLWEPreparedToRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>,
        Scratch<BE>: ScratchTakeCore<BE>;
}

impl<BE: Backend> CoreOperationsDefaults<BE> for BE
where
    Module<BE>: GLWEMulConstDefault<BE>
        + GLWEMulPlainDefault<BE>
        + GLWETensoringDefault<BE>
        + GLWERotateDefault<BE>
        + GGSWRotateDefault<BE>
        + GLWEMulXpMinusOneDefault<BE>
        + GLWEShiftDefault<BE>
        + GLWENormalizeDefault<BE>
        + GLWETraceDefault<BE>
        + GLWEPackingDefault<BE>
        + GLWEPackerOpsDefault<BE>,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    fn glwe_mul_const_tmp_bytes_default<R, A>(module: &Module<BE>, res: &R, a: &A, b_size: usize) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
    {
        <Module<BE> as GLWEMulConstDefault<BE>>::glwe_mul_const_tmp_bytes(module, res, a, b_size)
    }

    fn glwe_mul_const_default<R, A>(
        module: &Module<BE>,
        cnv_offset: usize,
        res: &mut GLWE<R>,
        a: &GLWE<A>,
        b: &[i64],
        scratch: &mut Scratch<BE>,
    ) where
        R: DataMut,
        A: DataRef,
    {
        <Module<BE> as GLWEMulConstDefault<BE>>::glwe_mul_const(module, cnv_offset, res, a, b, scratch)
    }

    fn glwe_mul_const_assign_default<R>(
        module: &Module<BE>,
        cnv_offset: usize,
        res: &mut GLWE<R>,
        b: &[i64],
        scratch: &mut Scratch<BE>,
    ) where
        R: DataMut,
    {
        <Module<BE> as GLWEMulConstDefault<BE>>::glwe_mul_const_assign(module, cnv_offset, res, b, scratch)
    }

    fn glwe_mul_plain_tmp_bytes_default<R, A, B>(module: &Module<BE>, res: &R, a: &A, b: &B) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        B: GLWEInfos,
    {
        <Module<BE> as GLWEMulPlainDefault<BE>>::glwe_mul_plain_tmp_bytes(module, res, a, b)
    }

    fn glwe_mul_plain_default<R, A, B>(
        module: &Module<BE>,
        cnv_offset: usize,
        res: &mut GLWE<R>,
        a: &GLWE<A>,
        a_effective_k: usize,
        b: &GLWEPlaintext<B>,
        b_effective_k: usize,
        scratch: &mut Scratch<BE>,
    ) where
        R: DataMut,
        A: DataRef,
        B: DataRef,
    {
        <Module<BE> as GLWEMulPlainDefault<BE>>::glwe_mul_plain(
            module,
            cnv_offset,
            res,
            a,
            a_effective_k,
            b,
            b_effective_k,
            scratch,
        )
    }

    fn glwe_mul_plain_assign_default<R, A>(
        module: &Module<BE>,
        cnv_offset: usize,
        res: &mut GLWE<R>,
        res_effective_k: usize,
        a: &GLWEPlaintext<A>,
        a_effective_k: usize,
        scratch: &mut Scratch<BE>,
    ) where
        R: DataMut,
        A: DataRef,
    {
        <Module<BE> as GLWEMulPlainDefault<BE>>::glwe_mul_plain_assign(
            module,
            cnv_offset,
            res,
            res_effective_k,
            a,
            a_effective_k,
            scratch,
        )
    }

    fn glwe_tensor_apply_tmp_bytes_default<R, A, B>(module: &Module<BE>, res: &R, a: &A, b: &B) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        B: GLWEInfos,
    {
        <Module<BE> as GLWETensoringDefault<BE>>::glwe_tensor_apply_tmp_bytes(module, res, a, b)
    }

    fn glwe_tensor_square_apply_tmp_bytes_default<R, A>(module: &Module<BE>, res: &R, a: &A) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
    {
        <Module<BE> as GLWETensoringDefault<BE>>::glwe_tensor_square_apply_tmp_bytes(module, res, a)
    }

    fn glwe_tensor_apply_default<R, A, B>(
        module: &Module<BE>,
        cnv_offset: usize,
        res: &mut GLWETensor<R>,
        a: &GLWE<A>,
        a_effective_k: usize,
        b: &GLWE<B>,
        b_effective_k: usize,
        scratch: &mut Scratch<BE>,
    ) where
        R: DataMut,
        A: DataRef,
        B: DataRef,
    {
        <Module<BE> as GLWETensoringDefault<BE>>::glwe_tensor_apply(
            module,
            cnv_offset,
            res,
            a,
            a_effective_k,
            b,
            b_effective_k,
            scratch,
        )
    }

    fn glwe_tensor_apply_add_assign_default<R, A, B>(
        module: &Module<BE>,
        cnv_offset: usize,
        res: &mut GLWETensor<R>,
        a: &GLWE<A>,
        a_effective_k: usize,
        b: &GLWE<B>,
        b_effective_k: usize,
        scratch: &mut Scratch<BE>,
    ) where
        R: DataMut,
        A: DataRef,
        B: DataRef,
    {
        <Module<BE> as GLWETensoringDefault<BE>>::glwe_tensor_apply_add_assign(
            module,
            cnv_offset,
            res,
            a,
            a_effective_k,
            b,
            b_effective_k,
            scratch,
        )
    }

    fn glwe_tensor_square_apply_default<R, A>(
        module: &Module<BE>,
        cnv_offset: usize,
        res: &mut GLWETensor<R>,
        a: &GLWE<A>,
        a_effective_k: usize,
        scratch: &mut Scratch<BE>,
    ) where
        R: DataMut,
        A: DataRef,
    {
        <Module<BE> as GLWETensoringDefault<BE>>::glwe_tensor_square_apply(module, cnv_offset, res, a, a_effective_k, scratch)
    }

    fn glwe_tensor_relinearize_default<R, A, B>(
        module: &Module<BE>,
        res: &mut GLWE<R>,
        a: &GLWETensor<A>,
        tsk: &GLWETensorKeyPrepared<B, BE>,
        tsk_size: usize,
        scratch: &mut Scratch<BE>,
    ) where
        R: DataMut,
        A: DataRef,
        B: DataRef,
    {
        <Module<BE> as GLWETensoringDefault<BE>>::glwe_tensor_relinearize(module, res, a, tsk, tsk_size, scratch)
    }

    fn glwe_tensor_relinearize_tmp_bytes_default<R, A, B>(module: &Module<BE>, res: &R, a: &A, tsk: &B) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        B: GGLWEInfos,
    {
        <Module<BE> as GLWETensoringDefault<BE>>::glwe_tensor_relinearize_tmp_bytes(module, res, a, tsk)
    }

    fn glwe_rotate_tmp_bytes_default(module: &Module<BE>) -> usize {
        <Module<BE> as GLWERotateDefault<BE>>::glwe_rotate_tmp_bytes(module)
    }

    fn glwe_rotate_default<R, A>(module: &Module<BE>, k: i64, res: &mut R, a: &A)
    where
        R: GLWEToMut,
        A: GLWEToRef,
    {
        <Module<BE> as GLWERotateDefault<BE>>::glwe_rotate(module, k, res, a)
    }

    fn glwe_rotate_assign_default<R>(module: &Module<BE>, k: i64, res: &mut R, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        <Module<BE> as GLWERotateDefault<BE>>::glwe_rotate_assign(module, k, res, scratch)
    }

    fn ggsw_rotate_tmp_bytes_default(module: &Module<BE>) -> usize {
        <Module<BE> as GGSWRotateDefault<BE>>::ggsw_rotate_tmp_bytes_default(module)
    }

    fn ggsw_rotate_default<R, A>(module: &Module<BE>, k: i64, res: &mut R, a: &A)
    where
        R: GGSWToMut,
        A: GGSWToRef,
    {
        <Module<BE> as GGSWRotateDefault<BE>>::ggsw_rotate_default(module, k, res, a)
    }

    fn ggsw_rotate_assign_default<R>(module: &Module<BE>, k: i64, res: &mut R, scratch: &mut Scratch<BE>)
    where
        R: GGSWToMut,
        Scratch<BE>: ScratchTakeCore<BE> + poulpy_hal::api::ScratchAvailable,
    {
        <Module<BE> as GGSWRotateDefault<BE>>::ggsw_rotate_assign_default(module, k, res, scratch)
    }

    fn glwe_mul_xp_minus_one_default<R, A>(module: &Module<BE>, k: i64, res: &mut R, a: &A)
    where
        R: GLWEToMut,
        A: GLWEToRef,
    {
        <Module<BE> as GLWEMulXpMinusOneDefault<BE>>::glwe_mul_xp_minus_one(module, k, res, a)
    }

    fn glwe_mul_xp_minus_one_assign_default<R>(module: &Module<BE>, k: i64, res: &mut R, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut,
    {
        <Module<BE> as GLWEMulXpMinusOneDefault<BE>>::glwe_mul_xp_minus_one_assign(module, k, res, scratch)
    }

    fn glwe_shift_tmp_bytes_default(module: &Module<BE>) -> usize {
        <Module<BE> as GLWEShiftDefault<BE>>::glwe_shift_tmp_bytes(module)
    }

    fn glwe_rsh_default<R>(module: &Module<BE>, k: usize, res: &mut R, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        <Module<BE> as GLWEShiftDefault<BE>>::glwe_rsh(module, k, res, scratch)
    }

    fn glwe_lsh_assign_default<R>(module: &Module<BE>, res: &mut R, k: usize, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        <Module<BE> as GLWEShiftDefault<BE>>::glwe_lsh_assign(module, res, k, scratch)
    }

    fn glwe_lsh_default<R, A>(module: &Module<BE>, res: &mut R, a: &A, k: usize, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut,
        A: GLWEToRef,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        <Module<BE> as GLWEShiftDefault<BE>>::glwe_lsh(module, res, a, k, scratch)
    }

    fn glwe_lsh_add_default<R, A>(module: &Module<BE>, res: &mut R, a: &A, k: usize, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut,
        A: GLWEToRef,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        <Module<BE> as GLWEShiftDefault<BE>>::glwe_lsh_add(module, res, a, k, scratch)
    }

    fn glwe_lsh_sub_default<R, A>(module: &Module<BE>, res: &mut R, a: &A, k: usize, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut,
        A: GLWEToRef,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        <Module<BE> as GLWEShiftDefault<BE>>::glwe_lsh_sub(module, res, a, k, scratch)
    }

    fn glwe_normalize_tmp_bytes_default(module: &Module<BE>) -> usize {
        <Module<BE> as GLWENormalizeDefault<BE>>::glwe_normalize_tmp_bytes(module)
    }

    fn glwe_maybe_cross_normalize_to_ref_default<'a, A>(
        module: &Module<BE>,
        glwe: &'a A,
        target_base2k: usize,
        tmp_slot: &'a mut Option<GLWE<&'a mut [u8]>>,
        scratch: &'a mut Scratch<BE>,
    ) -> (GLWE<&'a [u8]>, &'a mut Scratch<BE>)
    where
        A: GLWEToRef + GLWEInfos,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        <Module<BE> as GLWENormalizeDefault<BE>>::glwe_maybe_cross_normalize_to_ref(
            module,
            glwe,
            target_base2k,
            tmp_slot,
            scratch,
        )
    }

    fn glwe_maybe_cross_normalize_to_mut_default<'a, A>(
        module: &Module<BE>,
        glwe: &'a mut A,
        target_base2k: usize,
        tmp_slot: &'a mut Option<GLWE<&'a mut [u8]>>,
        scratch: &'a mut Scratch<BE>,
    ) -> (GLWE<&'a mut [u8]>, &'a mut Scratch<BE>)
    where
        A: GLWEToMut + GLWEInfos,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        <Module<BE> as GLWENormalizeDefault<BE>>::glwe_maybe_cross_normalize_to_mut(
            module,
            glwe,
            target_base2k,
            tmp_slot,
            scratch,
        )
    }

    fn glwe_normalize_default<R, A>(module: &Module<BE>, res: &mut R, a: &A, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut,
        A: GLWEToRef,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        <Module<BE> as GLWENormalizeDefault<BE>>::glwe_normalize(module, res, a, scratch)
    }

    fn glwe_normalize_assign_default<R>(module: &Module<BE>, res: &mut R, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        <Module<BE> as GLWENormalizeDefault<BE>>::glwe_normalize_assign(module, res, scratch)
    }

    fn glwe_trace_galois_elements_default(module: &Module<BE>) -> Vec<i64> {
        <Module<BE> as GLWETraceDefault<BE>>::glwe_trace_galois_elements_default(module)
    }

    fn glwe_trace_tmp_bytes_default<R, A, K>(module: &Module<BE>, res_infos: &R, a_infos: &A, key_infos: &K) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        K: GGLWEInfos,
    {
        <Module<BE> as GLWETraceDefault<BE>>::glwe_trace_tmp_bytes_default(module, res_infos, a_infos, key_infos)
    }

    fn glwe_trace_default<R, A, K, H>(module: &Module<BE>, res: &mut R, skip: usize, a: &A, keys: &H, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut + GLWEInfos,
        A: GLWEToRef + GLWEInfos,
        K: GGLWEPreparedToRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>,
    {
        <Module<BE> as GLWETraceDefault<BE>>::glwe_trace_default(module, res, skip, a, keys, scratch)
    }

    fn glwe_trace_assign_default<R, K, H>(module: &Module<BE>, res: &mut R, skip: usize, keys: &H, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut,
        K: GGLWEPreparedToRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>,
    {
        <Module<BE> as GLWETraceDefault<BE>>::glwe_trace_assign_default(module, res, skip, keys, scratch)
    }

    fn glwe_pack_galois_elements_default(module: &Module<BE>) -> Vec<i64> {
        <Module<BE> as GLWEPackingDefault<BE>>::glwe_pack_galois_elements_default(module)
    }

    fn glwe_pack_tmp_bytes_default<R, K>(module: &Module<BE>, res: &R, key: &K) -> usize
    where
        R: GLWEInfos,
        K: GGLWEInfos,
    {
        <Module<BE> as GLWEPackingDefault<BE>>::glwe_pack_tmp_bytes_default(module, res, key)
    }

    fn glwe_pack_default<R, A, K, H>(
        module: &Module<BE>,
        res: &mut R,
        a: HashMap<usize, &mut A>,
        log_gap_out: usize,
        keys: &H,
        scratch: &mut Scratch<BE>,
    ) where
        R: GLWEToMut + GLWEInfos,
        A: GLWEToMut + GLWEInfos,
        K: GGLWEPreparedToRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>,
    {
        <Module<BE> as GLWEPackingDefault<BE>>::glwe_pack_default(module, res, a, log_gap_out, keys, scratch)
    }

    fn packer_add_default<A, K, H>(
        module: &Module<BE>,
        packer: &mut GLWEPacker,
        a: Option<&A>,
        i: usize,
        auto_keys: &H,
        scratch: &mut Scratch<BE>,
    ) where
        A: GLWEToRef + GLWEInfos,
        K: GGLWEPreparedToRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        <Module<BE> as GLWEPackerOpsDefault<BE>>::packer_add_default(module, packer, a, i, auto_keys, scratch)
    }
}

#[macro_export]
macro_rules! impl_core_operations_default_methods {
    ($be:ty) => {
        fn glwe_mul_const_tmp_bytes<R, A>(module: &poulpy_hal::layouts::Module<$be>, res: &R, a: &A, b_size: usize) -> usize
        where
            R: $crate::layouts::GLWEInfos,
            A: $crate::layouts::GLWEInfos,
        {
            <$be as $crate::oep::CoreOperationsDefaults<$be>>::glwe_mul_const_tmp_bytes_default(module, res, a, b_size)
        }

        fn glwe_mul_const<R, A>(
            module: &poulpy_hal::layouts::Module<$be>,
            cnv_offset: usize,
            res: &mut $crate::layouts::GLWE<R>,
            a: &$crate::layouts::GLWE<A>,
            b: &[i64],
            scratch: &mut poulpy_hal::layouts::Scratch<$be>,
        ) where
            R: poulpy_hal::layouts::DataMut,
            A: poulpy_hal::layouts::DataRef,
        {
            <$be as $crate::oep::CoreOperationsDefaults<$be>>::glwe_mul_const_default(module, cnv_offset, res, a, b, scratch)
        }

        fn glwe_mul_const_assign<R>(
            module: &poulpy_hal::layouts::Module<$be>,
            cnv_offset: usize,
            res: &mut $crate::layouts::GLWE<R>,
            b: &[i64],
            scratch: &mut poulpy_hal::layouts::Scratch<$be>,
        ) where
            R: poulpy_hal::layouts::DataMut,
        {
            <$be as $crate::oep::CoreOperationsDefaults<$be>>::glwe_mul_const_assign_default(module, cnv_offset, res, b, scratch)
        }

        fn glwe_mul_plain_tmp_bytes<R, A, B>(module: &poulpy_hal::layouts::Module<$be>, res: &R, a: &A, b: &B) -> usize
        where
            R: $crate::layouts::GLWEInfos,
            A: $crate::layouts::GLWEInfos,
            B: $crate::layouts::GLWEInfos,
        {
            <$be as $crate::oep::CoreOperationsDefaults<$be>>::glwe_mul_plain_tmp_bytes_default(module, res, a, b)
        }

        fn glwe_mul_plain<R, A, B>(
            module: &poulpy_hal::layouts::Module<$be>,
            cnv_offset: usize,
            res: &mut $crate::layouts::GLWE<R>,
            a: &$crate::layouts::GLWE<A>,
            a_effective_k: usize,
            b: &$crate::layouts::GLWEPlaintext<B>,
            b_effective_k: usize,
            scratch: &mut poulpy_hal::layouts::Scratch<$be>,
        ) where
            R: poulpy_hal::layouts::DataMut,
            A: poulpy_hal::layouts::DataRef,
            B: poulpy_hal::layouts::DataRef,
        {
            <$be as $crate::oep::CoreOperationsDefaults<$be>>::glwe_mul_plain_default(
                module,
                cnv_offset,
                res,
                a,
                a_effective_k,
                b,
                b_effective_k,
                scratch,
            )
        }

        fn glwe_mul_plain_assign<R, A>(
            module: &poulpy_hal::layouts::Module<$be>,
            cnv_offset: usize,
            res: &mut $crate::layouts::GLWE<R>,
            res_effective_k: usize,
            a: &$crate::layouts::GLWEPlaintext<A>,
            a_effective_k: usize,
            scratch: &mut poulpy_hal::layouts::Scratch<$be>,
        ) where
            R: poulpy_hal::layouts::DataMut,
            A: poulpy_hal::layouts::DataRef,
        {
            <$be as $crate::oep::CoreOperationsDefaults<$be>>::glwe_mul_plain_assign_default(
                module,
                cnv_offset,
                res,
                res_effective_k,
                a,
                a_effective_k,
                scratch,
            )
        }

        fn glwe_tensor_apply_tmp_bytes<R, A, B>(module: &poulpy_hal::layouts::Module<$be>, res: &R, a: &A, b: &B) -> usize
        where
            R: $crate::layouts::GLWEInfos,
            A: $crate::layouts::GLWEInfos,
            B: $crate::layouts::GLWEInfos,
        {
            <$be as $crate::oep::CoreOperationsDefaults<$be>>::glwe_tensor_apply_tmp_bytes_default(module, res, a, b)
        }

        fn glwe_tensor_square_apply_tmp_bytes<R, A>(module: &poulpy_hal::layouts::Module<$be>, res: &R, a: &A) -> usize
        where
            R: $crate::layouts::GLWEInfos,
            A: $crate::layouts::GLWEInfos,
        {
            <$be as $crate::oep::CoreOperationsDefaults<$be>>::glwe_tensor_square_apply_tmp_bytes_default(module, res, a)
        }

        fn glwe_tensor_apply<R, A, B>(
            module: &poulpy_hal::layouts::Module<$be>,
            cnv_offset: usize,
            res: &mut $crate::layouts::GLWETensor<R>,
            a: &$crate::layouts::GLWE<A>,
            a_effective_k: usize,
            b: &$crate::layouts::GLWE<B>,
            b_effective_k: usize,
            scratch: &mut poulpy_hal::layouts::Scratch<$be>,
        ) where
            R: poulpy_hal::layouts::DataMut,
            A: poulpy_hal::layouts::DataRef,
            B: poulpy_hal::layouts::DataRef,
        {
            <$be as $crate::oep::CoreOperationsDefaults<$be>>::glwe_tensor_apply_default(
                module,
                cnv_offset,
                res,
                a,
                a_effective_k,
                b,
                b_effective_k,
                scratch,
            )
        }

        fn glwe_tensor_apply_add_assign<R, A, B>(
            module: &poulpy_hal::layouts::Module<$be>,
            cnv_offset: usize,
            res: &mut $crate::layouts::GLWETensor<R>,
            a: &$crate::layouts::GLWE<A>,
            a_effective_k: usize,
            b: &$crate::layouts::GLWE<B>,
            b_effective_k: usize,
            scratch: &mut poulpy_hal::layouts::Scratch<$be>,
        ) where
            R: poulpy_hal::layouts::DataMut,
            A: poulpy_hal::layouts::DataRef,
            B: poulpy_hal::layouts::DataRef,
        {
            <$be as $crate::oep::CoreOperationsDefaults<$be>>::glwe_tensor_apply_add_assign_default(
                module,
                cnv_offset,
                res,
                a,
                a_effective_k,
                b,
                b_effective_k,
                scratch,
            )
        }

        fn glwe_tensor_square_apply<R, A>(
            module: &poulpy_hal::layouts::Module<$be>,
            cnv_offset: usize,
            res: &mut $crate::layouts::GLWETensor<R>,
            a: &$crate::layouts::GLWE<A>,
            a_effective_k: usize,
            scratch: &mut poulpy_hal::layouts::Scratch<$be>,
        ) where
            R: poulpy_hal::layouts::DataMut,
            A: poulpy_hal::layouts::DataRef,
        {
            <$be as $crate::oep::CoreOperationsDefaults<$be>>::glwe_tensor_square_apply_default(
                module,
                cnv_offset,
                res,
                a,
                a_effective_k,
                scratch,
            )
        }

        fn glwe_tensor_relinearize<R, A, B>(
            module: &poulpy_hal::layouts::Module<$be>,
            res: &mut $crate::layouts::GLWE<R>,
            a: &$crate::layouts::GLWETensor<A>,
            tsk: &$crate::layouts::GLWETensorKeyPrepared<B, $be>,
            tsk_size: usize,
            scratch: &mut poulpy_hal::layouts::Scratch<$be>,
        ) where
            R: poulpy_hal::layouts::DataMut,
            A: poulpy_hal::layouts::DataRef,
            B: poulpy_hal::layouts::DataRef,
        {
            <$be as $crate::oep::CoreOperationsDefaults<$be>>::glwe_tensor_relinearize_default(
                module, res, a, tsk, tsk_size, scratch,
            )
        }

        fn glwe_tensor_relinearize_tmp_bytes<R, A, B>(module: &poulpy_hal::layouts::Module<$be>, res: &R, a: &A, tsk: &B) -> usize
        where
            R: $crate::layouts::GLWEInfos,
            A: $crate::layouts::GLWEInfos,
            B: $crate::layouts::GGLWEInfos,
        {
            <$be as $crate::oep::CoreOperationsDefaults<$be>>::glwe_tensor_relinearize_tmp_bytes_default(module, res, a, tsk)
        }

        fn glwe_rotate_tmp_bytes(module: &poulpy_hal::layouts::Module<$be>) -> usize {
            <$be as $crate::oep::CoreOperationsDefaults<$be>>::glwe_rotate_tmp_bytes_default(module)
        }

        fn glwe_rotate<R, A>(module: &poulpy_hal::layouts::Module<$be>, k: i64, res: &mut R, a: &A)
        where
            R: $crate::layouts::GLWEToMut,
            A: $crate::layouts::GLWEToRef,
        {
            <$be as $crate::oep::CoreOperationsDefaults<$be>>::glwe_rotate_default(module, k, res, a)
        }

        fn glwe_rotate_assign<R>(
            module: &poulpy_hal::layouts::Module<$be>,
            k: i64,
            res: &mut R,
            scratch: &mut poulpy_hal::layouts::Scratch<$be>,
        ) where
            R: $crate::layouts::GLWEToMut,
            poulpy_hal::layouts::Scratch<$be>: $crate::ScratchTakeCore<$be>,
        {
            <$be as $crate::oep::CoreOperationsDefaults<$be>>::glwe_rotate_assign_default(module, k, res, scratch)
        }

        fn ggsw_rotate_tmp_bytes(module: &poulpy_hal::layouts::Module<$be>) -> usize {
            <$be as $crate::oep::CoreOperationsDefaults<$be>>::ggsw_rotate_tmp_bytes_default(module)
        }

        fn ggsw_rotate<R, A>(module: &poulpy_hal::layouts::Module<$be>, k: i64, res: &mut R, a: &A)
        where
            R: $crate::layouts::GGSWToMut,
            A: $crate::layouts::GGSWToRef,
        {
            <$be as $crate::oep::CoreOperationsDefaults<$be>>::ggsw_rotate_default(module, k, res, a)
        }

        fn ggsw_rotate_assign<R>(
            module: &poulpy_hal::layouts::Module<$be>,
            k: i64,
            res: &mut R,
            scratch: &mut poulpy_hal::layouts::Scratch<$be>,
        ) where
            R: $crate::layouts::GGSWToMut,
            poulpy_hal::layouts::Scratch<$be>: $crate::ScratchTakeCore<$be> + poulpy_hal::api::ScratchAvailable,
        {
            <$be as $crate::oep::CoreOperationsDefaults<$be>>::ggsw_rotate_assign_default(module, k, res, scratch)
        }

        fn glwe_mul_xp_minus_one<R, A>(module: &poulpy_hal::layouts::Module<$be>, k: i64, res: &mut R, a: &A)
        where
            R: $crate::layouts::GLWEToMut,
            A: $crate::layouts::GLWEToRef,
        {
            <$be as $crate::oep::CoreOperationsDefaults<$be>>::glwe_mul_xp_minus_one_default(module, k, res, a)
        }

        fn glwe_mul_xp_minus_one_assign<R>(
            module: &poulpy_hal::layouts::Module<$be>,
            k: i64,
            res: &mut R,
            scratch: &mut poulpy_hal::layouts::Scratch<$be>,
        ) where
            R: $crate::layouts::GLWEToMut,
        {
            <$be as $crate::oep::CoreOperationsDefaults<$be>>::glwe_mul_xp_minus_one_assign_default(module, k, res, scratch)
        }

        fn glwe_shift_tmp_bytes(module: &poulpy_hal::layouts::Module<$be>) -> usize {
            <$be as $crate::oep::CoreOperationsDefaults<$be>>::glwe_shift_tmp_bytes_default(module)
        }

        fn glwe_rsh<R>(
            module: &poulpy_hal::layouts::Module<$be>,
            k: usize,
            res: &mut R,
            scratch: &mut poulpy_hal::layouts::Scratch<$be>,
        ) where
            R: $crate::layouts::GLWEToMut,
            poulpy_hal::layouts::Scratch<$be>: $crate::ScratchTakeCore<$be>,
        {
            <$be as $crate::oep::CoreOperationsDefaults<$be>>::glwe_rsh_default(module, k, res, scratch)
        }

        fn glwe_lsh_assign<R>(
            module: &poulpy_hal::layouts::Module<$be>,
            res: &mut R,
            k: usize,
            scratch: &mut poulpy_hal::layouts::Scratch<$be>,
        ) where
            R: $crate::layouts::GLWEToMut,
            poulpy_hal::layouts::Scratch<$be>: $crate::ScratchTakeCore<$be>,
        {
            <$be as $crate::oep::CoreOperationsDefaults<$be>>::glwe_lsh_assign_default(module, res, k, scratch)
        }

        fn glwe_lsh<R, A>(
            module: &poulpy_hal::layouts::Module<$be>,
            res: &mut R,
            a: &A,
            k: usize,
            scratch: &mut poulpy_hal::layouts::Scratch<$be>,
        ) where
            R: $crate::layouts::GLWEToMut,
            A: $crate::layouts::GLWEToRef,
            poulpy_hal::layouts::Scratch<$be>: $crate::ScratchTakeCore<$be>,
        {
            <$be as $crate::oep::CoreOperationsDefaults<$be>>::glwe_lsh_default(module, res, a, k, scratch)
        }

        fn glwe_lsh_add<R, A>(
            module: &poulpy_hal::layouts::Module<$be>,
            res: &mut R,
            a: &A,
            k: usize,
            scratch: &mut poulpy_hal::layouts::Scratch<$be>,
        ) where
            R: $crate::layouts::GLWEToMut,
            A: $crate::layouts::GLWEToRef,
            poulpy_hal::layouts::Scratch<$be>: $crate::ScratchTakeCore<$be>,
        {
            <$be as $crate::oep::CoreOperationsDefaults<$be>>::glwe_lsh_add_default(module, res, a, k, scratch)
        }

        fn glwe_lsh_sub<R, A>(
            module: &poulpy_hal::layouts::Module<$be>,
            res: &mut R,
            a: &A,
            k: usize,
            scratch: &mut poulpy_hal::layouts::Scratch<$be>,
        ) where
            R: $crate::layouts::GLWEToMut,
            A: $crate::layouts::GLWEToRef,
            poulpy_hal::layouts::Scratch<$be>: $crate::ScratchTakeCore<$be>,
        {
            <$be as $crate::oep::CoreOperationsDefaults<$be>>::glwe_lsh_sub_default(module, res, a, k, scratch)
        }

        fn glwe_normalize_tmp_bytes(module: &poulpy_hal::layouts::Module<$be>) -> usize {
            <$be as $crate::oep::CoreOperationsDefaults<$be>>::glwe_normalize_tmp_bytes_default(module)
        }

        fn glwe_maybe_cross_normalize_to_ref<'a, A>(
            module: &poulpy_hal::layouts::Module<$be>,
            glwe: &'a A,
            target_base2k: usize,
            tmp_slot: &'a mut Option<$crate::layouts::GLWE<&'a mut [u8]>>,
            scratch: &'a mut poulpy_hal::layouts::Scratch<$be>,
        ) -> (
            $crate::layouts::GLWE<&'a [u8]>,
            &'a mut poulpy_hal::layouts::Scratch<$be>,
        )
        where
            A: $crate::layouts::GLWEToRef + $crate::layouts::GLWEInfos,
            poulpy_hal::layouts::Scratch<$be>: $crate::ScratchTakeCore<$be>,
        {
            <$be as $crate::oep::CoreOperationsDefaults<$be>>::glwe_maybe_cross_normalize_to_ref_default(
                module,
                glwe,
                target_base2k,
                tmp_slot,
                scratch,
            )
        }

        fn glwe_maybe_cross_normalize_to_mut<'a, A>(
            module: &poulpy_hal::layouts::Module<$be>,
            glwe: &'a mut A,
            target_base2k: usize,
            tmp_slot: &'a mut Option<$crate::layouts::GLWE<&'a mut [u8]>>,
            scratch: &'a mut poulpy_hal::layouts::Scratch<$be>,
        ) -> (
            $crate::layouts::GLWE<&'a mut [u8]>,
            &'a mut poulpy_hal::layouts::Scratch<$be>,
        )
        where
            A: $crate::layouts::GLWEToMut + $crate::layouts::GLWEInfos,
            poulpy_hal::layouts::Scratch<$be>: $crate::ScratchTakeCore<$be>,
        {
            <$be as $crate::oep::CoreOperationsDefaults<$be>>::glwe_maybe_cross_normalize_to_mut_default(
                module,
                glwe,
                target_base2k,
                tmp_slot,
                scratch,
            )
        }

        fn glwe_normalize<R, A>(
            module: &poulpy_hal::layouts::Module<$be>,
            res: &mut R,
            a: &A,
            scratch: &mut poulpy_hal::layouts::Scratch<$be>,
        ) where
            R: $crate::layouts::GLWEToMut,
            A: $crate::layouts::GLWEToRef,
            poulpy_hal::layouts::Scratch<$be>: $crate::ScratchTakeCore<$be>,
        {
            <$be as $crate::oep::CoreOperationsDefaults<$be>>::glwe_normalize_default(module, res, a, scratch)
        }

        fn glwe_normalize_assign<R>(
            module: &poulpy_hal::layouts::Module<$be>,
            res: &mut R,
            scratch: &mut poulpy_hal::layouts::Scratch<$be>,
        ) where
            R: $crate::layouts::GLWEToMut,
            poulpy_hal::layouts::Scratch<$be>: $crate::ScratchTakeCore<$be>,
        {
            <$be as $crate::oep::CoreOperationsDefaults<$be>>::glwe_normalize_assign_default(module, res, scratch)
        }

        fn glwe_trace_galois_elements(module: &poulpy_hal::layouts::Module<$be>) -> Vec<i64> {
            <$be as $crate::oep::CoreOperationsDefaults<$be>>::glwe_trace_galois_elements_default(module)
        }

        fn glwe_trace_tmp_bytes<R, A, K>(
            module: &poulpy_hal::layouts::Module<$be>,
            res_infos: &R,
            a_infos: &A,
            key_infos: &K,
        ) -> usize
        where
            R: $crate::layouts::GLWEInfos,
            A: $crate::layouts::GLWEInfos,
            K: $crate::layouts::GGLWEInfos,
        {
            <$be as $crate::oep::CoreOperationsDefaults<$be>>::glwe_trace_tmp_bytes_default(module, res_infos, a_infos, key_infos)
        }

        fn glwe_trace<R, A, K, H>(
            module: &poulpy_hal::layouts::Module<$be>,
            res: &mut R,
            skip: usize,
            a: &A,
            keys: &H,
            scratch: &mut poulpy_hal::layouts::Scratch<$be>,
        ) where
            R: $crate::layouts::GLWEToMut + $crate::layouts::GLWEInfos,
            A: $crate::layouts::GLWEToRef + $crate::layouts::GLWEInfos,
            K: $crate::layouts::GGLWEPreparedToRef<$be> + $crate::layouts::GetGaloisElement + $crate::layouts::GGLWEInfos,
            H: $crate::layouts::GLWEAutomorphismKeyHelper<K, $be>,
        {
            <$be as $crate::oep::CoreOperationsDefaults<$be>>::glwe_trace_default(module, res, skip, a, keys, scratch)
        }

        fn glwe_trace_assign<R, K, H>(
            module: &poulpy_hal::layouts::Module<$be>,
            res: &mut R,
            skip: usize,
            keys: &H,
            scratch: &mut poulpy_hal::layouts::Scratch<$be>,
        ) where
            R: $crate::layouts::GLWEToMut,
            K: $crate::layouts::GGLWEPreparedToRef<$be> + $crate::layouts::GetGaloisElement + $crate::layouts::GGLWEInfos,
            H: $crate::layouts::GLWEAutomorphismKeyHelper<K, $be>,
        {
            <$be as $crate::oep::CoreOperationsDefaults<$be>>::glwe_trace_assign_default(module, res, skip, keys, scratch)
        }

        fn glwe_pack_galois_elements(module: &poulpy_hal::layouts::Module<$be>) -> Vec<i64> {
            <$be as $crate::oep::CoreOperationsDefaults<$be>>::glwe_pack_galois_elements_default(module)
        }

        fn glwe_pack_tmp_bytes<R, K>(module: &poulpy_hal::layouts::Module<$be>, res: &R, key: &K) -> usize
        where
            R: $crate::layouts::GLWEInfos,
            K: $crate::layouts::GGLWEInfos,
        {
            <$be as $crate::oep::CoreOperationsDefaults<$be>>::glwe_pack_tmp_bytes_default(module, res, key)
        }

        fn glwe_pack<R, A, K, H>(
            module: &poulpy_hal::layouts::Module<$be>,
            res: &mut R,
            a: std::collections::HashMap<usize, &mut A>,
            log_gap_out: usize,
            keys: &H,
            scratch: &mut poulpy_hal::layouts::Scratch<$be>,
        ) where
            R: $crate::layouts::GLWEToMut + $crate::layouts::GLWEInfos,
            A: $crate::layouts::GLWEToMut + $crate::layouts::GLWEInfos,
            K: $crate::layouts::GGLWEPreparedToRef<$be> + $crate::layouts::GetGaloisElement + $crate::layouts::GGLWEInfos,
            H: $crate::layouts::GLWEAutomorphismKeyHelper<K, $be>,
        {
            <$be as $crate::oep::CoreOperationsDefaults<$be>>::glwe_pack_default(module, res, a, log_gap_out, keys, scratch)
        }

        fn packer_add<A, K, H>(
            module: &poulpy_hal::layouts::Module<$be>,
            packer: &mut $crate::GLWEPacker,
            a: Option<&A>,
            i: usize,
            auto_keys: &H,
            scratch: &mut poulpy_hal::layouts::Scratch<$be>,
        ) where
            A: $crate::layouts::GLWEToRef + $crate::layouts::GLWEInfos,
            K: $crate::layouts::GGLWEPreparedToRef<$be> + $crate::layouts::GetGaloisElement + $crate::layouts::GGLWEInfos,
            H: $crate::layouts::GLWEAutomorphismKeyHelper<K, $be>,
            poulpy_hal::layouts::Scratch<$be>: $crate::ScratchTakeCore<$be>,
        {
            <$be as $crate::oep::CoreOperationsDefaults<$be>>::packer_add_default(module, packer, a, i, auto_keys, scratch)
        }
    };
}
