//! Prepared GLWE linear-transform evaluators.
//!
//! The prepared path follows docs/lt_bsgs.md §6: hoisted baby rotations,
//! DFT-domain inner products, lazy giant rotations, and one final BIG
//! normalization when the backend/key layout permits it.

use poulpy_hal::{
    api::{
        CnvPVecAlloc, Convolution, VecZnxAutomorphismAssignBackend, VecZnxBigAddAssign, VecZnxBigAddSmallAssign, VecZnxBigAlloc,
        VecZnxBigAutomorphismAssign, VecZnxBigAutomorphismAssignTmpBytes, VecZnxBigBytesOf, VecZnxBigFromSmallBackend,
        VecZnxBigNormalize, VecZnxCopyBackend, VecZnxDftApply, VecZnxDftBytesOf, VecZnxDftZero, VecZnxIdftApply,
        VecZnxIdftApplyTmpA, VecZnxIdftApplyTmpBytes,
    },
    layouts::{Backend, Module, ScratchArena},
};

use crate::{
    GLWEAdd, GLWEAutomorphism, GLWECopy, GLWEMulPlain,
    default::{
        keyswitching::GGLWEProductDefault,
        linear_transformation::{
            baby_steps::{glwe_prepare_baby_rotations, glwe_prepare_baby_rotations_tmp_bytes},
            inner_product::glwe_accumulate_prepared_inner_product_big_tmp_bytes,
            lazy::{glwe_lazy_giant_automorphism_from_big_tmp_bytes, glwe_lazy_giant_automorphism_tmp_bytes},
            ops::GLWELinearTransformOps,
            prepared_giants::glwe_prepared_linear_transform_with_babies as glwe_prepared_linear_transform_with_babies_impl,
        },
    },
    layouts::{
        GGLWEInfos, GLWEAutomorphismKeyHelper, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, GetGaloisElement, ModuleCoreAlloc,
        prepared::GGLWEPreparedToBackendRef,
    },
};

use super::{GLWELinearTransform, GLWEPreparedLinearTransform};

impl<BE: Backend> GLWELinearTransformOps<BE> for Module<BE>
where
    Module<BE>: GLWEAutomorphism<BE>
        + GLWEMulPlain<BE>
        + GLWEAdd<BE>
        + GLWECopy<BE>
        + CnvPVecAlloc<BE>
        + Convolution<BE>
        + GGLWEProductDefault<BE>
        + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf>
        + VecZnxAutomorphismAssignBackend<BE>
        + VecZnxBigAddAssign<BE>
        + VecZnxBigAddSmallAssign<BE>
        + VecZnxBigAlloc<BE>
        + VecZnxBigAutomorphismAssign<BE>
        + VecZnxBigAutomorphismAssignTmpBytes
        + VecZnxBigBytesOf
        + VecZnxBigFromSmallBackend<BE>
        + VecZnxBigNormalize<BE>
        + VecZnxCopyBackend<BE>
        + VecZnxDftApply<BE>
        + VecZnxDftBytesOf
        + VecZnxDftZero<BE>
        + VecZnxIdftApply<BE>
        + VecZnxIdftApplyTmpA<BE>
        + VecZnxIdftApplyTmpBytes,
{
    fn glwe_prepared_linear_transform_tmp_bytes<R, A, B, K>(&self, res: &R, a: &A, pt: &B, key: &K) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        B: GLWEInfos,
        K: GGLWEInfos,
    {
        // The prepared evaluator carries a (r+1)-column BIG `prod_big` over the
        // giant-step loop and a (r+1)-column BIG `rot_big` per non-zero giant
        // step; the inner-product hoists its DFT scratch outside the column
        // loop; the lazy giant variant needs its own DFT/SMALL workspace.
        let cols = a.rank().as_usize() + 1;
        let cnv_offset_hi = pt.size().saturating_sub(1);
        let prod_size = a.size() + pt.size() - cnv_offset_hi;
        let inner = glwe_accumulate_prepared_inner_product_big_tmp_bytes(self, cnv_offset_hi, a.size(), pt.size());
        let prod_big = self.bytes_of_vec_znx_big(cols, prod_size);
        let rot_big = self.bytes_of_vec_znx_big(cols, key.size());
        let prepare_right = self.cnv_prepare_right_tmp_bytes(pt.size(), pt.size());
        let lazy_big =
            glwe_lazy_giant_automorphism_from_big_tmp_bytes::<BE, _, _>(self, a.rank().as_usize(), prod_size, key, key.size());

        self.glwe_automorphism_tmp_bytes(res, a, key)
            .max(self.glwe_mul_plain_tmp_bytes(res, a, pt))
            .max(prepare_right)
            .max(glwe_prepare_baby_rotations_tmp_bytes::<BE, _, _, _>(self, a, key))
            .max(glwe_lazy_giant_automorphism_tmp_bytes::<BE, _, _, _>(
                self,
                res,
                key,
                key.size(),
            ))
            .max(prod_big + rot_big + inner.max(lazy_big))
    }

    fn glwe_prepare_baby_rotations_tmp_bytes<A, K>(&self, a: &A, key: &K) -> usize
    where
        A: GLWEInfos,
        K: GGLWEInfos,
    {
        glwe_prepare_baby_rotations_tmp_bytes::<BE, _, _, _>(self, a, key)
    }

    fn glwe_prepare_baby_rotations<A, H, K>(
        &self,
        baby_steps: &[i64],
        a: &A,
        a_effective_k: usize,
        key_size: usize,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> super::GLWEPreparedBabyRotations<BE>
    where
        A: GLWEToBackendRef<BE> + GLWEInfos,
        K: GetGaloisElement + GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>,
    {
        glwe_prepare_baby_rotations(self, baby_steps, a, a_effective_k, key_size, keys, scratch)
    }

    fn glwe_prepared_linear_transform<R, P, H, K>(
        &self,
        res: &mut R,
        lt: &GLWELinearTransform<P>,
        prepared: &GLWEPreparedLinearTransform<BE>,
        babies: &super::GLWEPreparedBabyRotations<BE>,
        cnv_offset: usize,
        key_size: usize,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        P: GLWEToBackendRef<BE> + GLWEInfos,
        K: GetGaloisElement + GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>,
    {
        assert!(
            prepared.giant_steps.iter().any(|gs| !gs.diagonals.is_empty()),
            "linear transformation has no non-empty giant steps"
        );

        glwe_prepared_linear_transform_with_babies_impl(self, res, lt, prepared, babies, cnv_offset, key_size, keys, scratch);
    }
}
