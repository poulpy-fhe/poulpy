use std::collections::HashMap;

use poulpy_hal::layouts::{Backend, DataMut, DataRef, Module, Scratch};

use crate::{
    api::{
        GGSWRotate, GLWEMulConst, GLWEMulPlain, GLWEMulXpMinusOne, GLWENormalize, GLWEPackerOps, GLWEPacking, GLWERotate,
        GLWEShift, GLWETensoring, GLWETrace,
    },
    glwe_packer::GLWEPackerOpsDefault,
    glwe_packing::GLWEPackingDefault,
    glwe_trace::GLWETraceDefault,
    layouts::{
        GGLWEInfos, GGLWEPreparedToRef, GGSWToMut, GGSWToRef, GLWE, GLWEAutomorphismKeyHelper, GLWEInfos, GLWEPlaintext,
        GLWETensor, GLWETensorKeyPrepared, GLWEToMut, GLWEToRef, GetGaloisElement,
    },
    oep::CoreImpl,
    operations::{
        GGSWRotateDefault, GLWEMulConstDefault, GLWEMulPlainDefault, GLWEMulXpMinusOneDefault, GLWENormalizeDefault,
        GLWERotateDefault, GLWEShiftDefault, GLWETensoringDefault,
    },
};

impl<BE> GLWEMulConst<BE> for Module<BE>
where
    BE: Backend + CoreImpl<BE>,
    Module<BE>: GLWEMulConstDefault<BE>,
{
    fn glwe_mul_const_tmp_bytes<R, A>(&self, res: &R, a: &A, b_size: usize) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
    {
        BE::glwe_mul_const_tmp_bytes(self, res, a, b_size)
    }

    fn glwe_mul_const<R, A>(&self, cnv_offset: usize, res: &mut GLWE<R>, a: &GLWE<A>, b: &[i64], scratch: &mut Scratch<BE>)
    where
        R: DataMut,
        A: DataRef,
    {
        BE::glwe_mul_const(self, cnv_offset, res, a, b, scratch)
    }

    fn glwe_mul_const_assign<R>(&self, cnv_offset: usize, res: &mut GLWE<R>, b: &[i64], scratch: &mut Scratch<BE>)
    where
        R: DataMut,
    {
        BE::glwe_mul_const_assign(self, cnv_offset, res, b, scratch)
    }
}

impl<BE> GLWEMulPlain<BE> for Module<BE>
where
    BE: Backend + CoreImpl<BE>,
    Module<BE>: GLWEMulPlainDefault<BE>,
{
    fn glwe_mul_plain_tmp_bytes<R, A, B>(&self, res: &R, a: &A, b: &B) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        B: GLWEInfos,
    {
        BE::glwe_mul_plain_tmp_bytes(self, res, a, b)
    }

    fn glwe_mul_plain<R, A, B>(
        &self,
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
        BE::glwe_mul_plain(self, cnv_offset, res, a, a_effective_k, b, b_effective_k, scratch)
    }

    fn glwe_mul_plain_assign<R, A>(
        &self,
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
        BE::glwe_mul_plain_assign(self, cnv_offset, res, res_effective_k, a, a_effective_k, scratch)
    }
}

impl<BE> GLWETensoring<BE> for Module<BE>
where
    BE: Backend + CoreImpl<BE>,
    Module<BE>: GLWETensoringDefault<BE>,
{
    fn glwe_tensor_apply_tmp_bytes<R, A, B>(&self, res: &R, a: &A, b: &B) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        B: GLWEInfos,
    {
        BE::glwe_tensor_apply_tmp_bytes(self, res, a, b)
    }

    fn glwe_tensor_square_apply_tmp_bytes<R, A>(&self, res: &R, a: &A) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
    {
        BE::glwe_tensor_square_apply_tmp_bytes(self, res, a)
    }

    fn glwe_tensor_apply<R, A, B>(
        &self,
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
        BE::glwe_tensor_apply(self, cnv_offset, res, a, a_effective_k, b, b_effective_k, scratch)
    }

    fn glwe_tensor_apply_add_assign<R, A, B>(
        &self,
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
        BE::glwe_tensor_apply_add_assign(self, cnv_offset, res, a, a_effective_k, b, b_effective_k, scratch)
    }

    fn glwe_tensor_square_apply<R, A>(
        &self,
        cnv_offset: usize,
        res: &mut GLWETensor<R>,
        a: &GLWE<A>,
        a_effective_k: usize,
        scratch: &mut Scratch<BE>,
    ) where
        R: DataMut,
        A: DataRef,
    {
        BE::glwe_tensor_square_apply(self, cnv_offset, res, a, a_effective_k, scratch)
    }

    fn glwe_tensor_relinearize<R, A, B>(
        &self,
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
        BE::glwe_tensor_relinearize(self, res, a, tsk, tsk_size, scratch)
    }

    fn glwe_tensor_relinearize_tmp_bytes<R, A, B>(&self, res: &R, a: &A, tsk: &B) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        B: GGLWEInfos,
    {
        BE::glwe_tensor_relinearize_tmp_bytes(self, res, a, tsk)
    }
}

impl<BE> GLWERotate<BE> for Module<BE>
where
    BE: Backend + CoreImpl<BE>,
    Module<BE>: GLWERotateDefault<BE>,
{
    fn glwe_rotate_tmp_bytes(&self) -> usize {
        BE::glwe_rotate_tmp_bytes(self)
    }

    fn glwe_rotate<R, A>(&self, k: i64, res: &mut R, a: &A)
    where
        R: GLWEToMut,
        A: GLWEToRef,
    {
        BE::glwe_rotate(self, k, res, a)
    }

    fn glwe_rotate_assign<R>(&self, k: i64, res: &mut R, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut,
        Scratch<BE>: crate::ScratchTakeCore<BE>,
    {
        BE::glwe_rotate_assign(self, k, res, scratch)
    }
}

impl<BE> GGSWRotate<BE> for Module<BE>
where
    BE: Backend + CoreImpl<BE>,
    Module<BE>: GGSWRotateDefault<BE>,
{
    fn ggsw_rotate_tmp_bytes(&self) -> usize {
        BE::ggsw_rotate_tmp_bytes(self)
    }

    fn ggsw_rotate<R, A>(&self, k: i64, res: &mut R, a: &A)
    where
        R: GGSWToMut,
        A: GGSWToRef,
    {
        BE::ggsw_rotate(self, k, res, a)
    }

    fn ggsw_rotate_assign<R>(&self, k: i64, res: &mut R, scratch: &mut Scratch<BE>)
    where
        R: GGSWToMut,
        Scratch<BE>: crate::ScratchTakeCore<BE> + poulpy_hal::api::ScratchAvailable,
    {
        BE::ggsw_rotate_assign(self, k, res, scratch)
    }
}

impl<BE> GLWEMulXpMinusOne<BE> for Module<BE>
where
    BE: Backend + CoreImpl<BE>,
    Module<BE>: GLWEMulXpMinusOneDefault<BE>,
{
    fn glwe_mul_xp_minus_one<R, A>(&self, k: i64, res: &mut R, a: &A)
    where
        R: GLWEToMut,
        A: GLWEToRef,
    {
        BE::glwe_mul_xp_minus_one(self, k, res, a)
    }

    fn glwe_mul_xp_minus_one_assign<R>(&self, k: i64, res: &mut R, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut,
    {
        BE::glwe_mul_xp_minus_one_assign(self, k, res, scratch)
    }
}

impl<BE> GLWEShift<BE> for Module<BE>
where
    BE: Backend + CoreImpl<BE>,
    Module<BE>: GLWEShiftDefault<BE>,
{
    fn glwe_shift_tmp_bytes(&self) -> usize {
        BE::glwe_shift_tmp_bytes(self)
    }

    fn glwe_rsh<R>(&self, k: usize, res: &mut R, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut,
        Scratch<BE>: crate::ScratchTakeCore<BE>,
    {
        BE::glwe_rsh(self, k, res, scratch)
    }

    fn glwe_lsh_assign<R>(&self, res: &mut R, k: usize, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut,
        Scratch<BE>: crate::ScratchTakeCore<BE>,
    {
        BE::glwe_lsh_assign(self, res, k, scratch)
    }

    fn glwe_lsh<R, A>(&self, res: &mut R, a: &A, k: usize, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut,
        A: GLWEToRef,
        Scratch<BE>: crate::ScratchTakeCore<BE>,
    {
        BE::glwe_lsh(self, res, a, k, scratch)
    }

    fn glwe_lsh_add<R, A>(&self, res: &mut R, a: &A, k: usize, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut,
        A: GLWEToRef,
        Scratch<BE>: crate::ScratchTakeCore<BE>,
    {
        BE::glwe_lsh_add(self, res, a, k, scratch)
    }

    fn glwe_lsh_sub<R, A>(&self, res: &mut R, a: &A, k: usize, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut,
        A: GLWEToRef,
        Scratch<BE>: crate::ScratchTakeCore<BE>,
    {
        BE::glwe_lsh_sub(self, res, a, k, scratch)
    }
}

impl<BE> GLWENormalize<BE> for Module<BE>
where
    BE: Backend + CoreImpl<BE>,
    Module<BE>: GLWENormalizeDefault<BE>,
{
    fn glwe_normalize_tmp_bytes(&self) -> usize {
        BE::glwe_normalize_tmp_bytes(self)
    }

    fn glwe_maybe_cross_normalize_to_ref<'a, A>(
        &self,
        glwe: &'a A,
        target_base2k: usize,
        tmp_slot: &'a mut Option<GLWE<&'a mut [u8]>>,
        scratch: &'a mut Scratch<BE>,
    ) -> (GLWE<&'a [u8]>, &'a mut Scratch<BE>)
    where
        A: GLWEToRef + GLWEInfos,
        Scratch<BE>: crate::ScratchTakeCore<BE>,
    {
        BE::glwe_maybe_cross_normalize_to_ref(self, glwe, target_base2k, tmp_slot, scratch)
    }

    fn glwe_maybe_cross_normalize_to_mut<'a, A>(
        &self,
        glwe: &'a mut A,
        target_base2k: usize,
        tmp_slot: &'a mut Option<GLWE<&'a mut [u8]>>,
        scratch: &'a mut Scratch<BE>,
    ) -> (GLWE<&'a mut [u8]>, &'a mut Scratch<BE>)
    where
        A: GLWEToMut + GLWEInfos,
        Scratch<BE>: crate::ScratchTakeCore<BE>,
    {
        BE::glwe_maybe_cross_normalize_to_mut(self, glwe, target_base2k, tmp_slot, scratch)
    }

    fn glwe_normalize<R, A>(&self, res: &mut R, a: &A, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut,
        A: GLWEToRef,
        Scratch<BE>: crate::ScratchTakeCore<BE>,
    {
        BE::glwe_normalize(self, res, a, scratch)
    }

    fn glwe_normalize_assign<R>(&self, res: &mut R, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut,
        Scratch<BE>: crate::ScratchTakeCore<BE>,
    {
        BE::glwe_normalize_assign(self, res, scratch)
    }
}

impl<BE> GLWETrace<BE> for Module<BE>
where
    BE: Backend + CoreImpl<BE>,
    Module<BE>: GLWETraceDefault<BE>,
{
    fn glwe_trace_galois_elements(&self) -> Vec<i64> {
        BE::glwe_trace_galois_elements(self)
    }

    fn glwe_trace_tmp_bytes<R, A, K>(&self, res_infos: &R, a_infos: &A, key_infos: &K) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        K: GGLWEInfos,
    {
        BE::glwe_trace_tmp_bytes(self, res_infos, a_infos, key_infos)
    }

    fn glwe_trace<R, A, K, H>(&self, res: &mut R, skip: usize, a: &A, keys: &H, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut + GLWEInfos,
        A: GLWEToRef + GLWEInfos,
        K: GGLWEPreparedToRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>,
    {
        BE::glwe_trace(self, res, skip, a, keys, scratch)
    }

    fn glwe_trace_assign<R, K, H>(&self, res: &mut R, skip: usize, keys: &H, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut,
        K: GGLWEPreparedToRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>,
    {
        BE::glwe_trace_assign(self, res, skip, keys, scratch)
    }
}

impl<BE> GLWEPacking<BE> for Module<BE>
where
    BE: Backend + CoreImpl<BE>,
    Module<BE>: GLWEPackingDefault<BE>,
{
    fn glwe_pack_galois_elements(&self) -> Vec<i64> {
        BE::glwe_pack_galois_elements(self)
    }

    fn glwe_pack_tmp_bytes<R, K>(&self, res: &R, key: &K) -> usize
    where
        R: GLWEInfos,
        K: GGLWEInfos,
    {
        BE::glwe_pack_tmp_bytes(self, res, key)
    }

    fn glwe_pack<R, A, K, H>(
        &self,
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
        BE::glwe_pack(self, res, a, log_gap_out, keys, scratch)
    }
}

impl<BE> GLWEPackerOps<BE> for Module<BE>
where
    BE: Backend + CoreImpl<BE>,
    Module<BE>: GLWEPackerOpsDefault<BE>,
{
    fn packer_add<A, K, H>(
        &self,
        packer: &mut crate::GLWEPacker,
        a: Option<&A>,
        i: usize,
        auto_keys: &H,
        scratch: &mut Scratch<BE>,
    ) where
        A: GLWEToRef + GLWEInfos,
        K: GGLWEPreparedToRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>,
        Scratch<BE>: crate::ScratchTakeCore<BE>,
    {
        BE::packer_add(self, packer, a, i, auto_keys, scratch)
    }
}
