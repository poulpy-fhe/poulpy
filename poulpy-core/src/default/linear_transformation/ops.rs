//! Public GLWE linear-transform operation traits.
//!
//! These traits are the scheme-agnostic API used by CKKS. The evaluator consumes
//! prepared transforms and prepared baby rotations, matching the optimized path
//! described in docs/lt_bsgs.md §6.

use poulpy_hal::layouts::{Backend, ScratchArena};

use crate::layouts::{
    GGLWEInfos, GLWEAutomorphismKeyHelper, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, GetGaloisElement,
    prepared::GGLWEPreparedToBackendRef,
};

use super::{GLWELinearTransform, GLWEPreparedBabyRotations, GLWEPreparedLinearTransform};

/// GLWE-level evaluation of a [`GLWEPreparedLinearTransform`].
pub trait GLWELinearTransformOps<BE: Backend> {
    /// Scratch bytes required by prepared linear-transform evaluation and the
    /// usual one-shot baby-rotation preparation around it.
    fn glwe_prepared_linear_transform_tmp_bytes<R, A, B, K>(&self, res: &R, a: &A, pt: &B, key: &K) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        B: GLWEInfos,
        K: GGLWEInfos;

    /// Scratch bytes required by [`Self::glwe_prepare_baby_rotations`].
    fn glwe_prepare_baby_rotations_tmp_bytes<A, K>(&self, _a: &A, _key: &K) -> usize
    where
        A: GLWEInfos,
        K: GGLWEInfos,
    {
        panic!("GLWELinearTransformOps::glwe_prepare_baby_rotations_tmp_bytes is not implemented for this module")
    }

    /// Computes and prepares the requested baby-step rotations of `a`.
    ///
    /// The returned cache stores only final `CnvPVecL` baby steps. Intermediate
    /// rotated GLWE ciphertexts are scratch-backed.
    #[allow(clippy::too_many_arguments)]
    fn glwe_prepare_baby_rotations<A, H, K>(
        &self,
        _baby_steps: &[i64],
        _a: &A,
        _a_effective_k: usize,
        _key_size: usize,
        _keys: &H,
        _scratch: &mut ScratchArena<'_, BE>,
    ) -> GLWEPreparedBabyRotations<BE>
    where
        A: GLWEToBackendRef<BE> + GLWEInfos,
        K: GetGaloisElement + GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>,
    {
        panic!("GLWELinearTransformOps::glwe_prepare_baby_rotations is not implemented for this module")
    }

    /// Computes `res = M(a)` using a precomputed baby-step cache for `a`.
    ///
    /// `babies` may contain exactly `lt.baby_steps` or a superset such as the
    /// union of baby rotations needed by several transforms.
    #[allow(clippy::too_many_arguments)]
    fn glwe_prepared_linear_transform<R, P, H, K>(
        &self,
        res: &mut R,
        lt: &GLWELinearTransform<P>,
        prepared: &GLWEPreparedLinearTransform<BE>,
        babies: &GLWEPreparedBabyRotations<BE>,
        cnv_offset: usize,
        key_size: usize,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        P: GLWEToBackendRef<BE> + GLWEInfos,
        K: GetGaloisElement + GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>;
}
