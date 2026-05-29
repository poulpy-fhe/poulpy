//! CKKS-facing API for the homomorphic linear transformation (a matrix–vector
//! product over the slots), evaluated with the baby-step / giant-step (BSGS)
//! decomposition.
//!
//! The transformation itself is the scheme-agnostic GLWE-level engine in
//! [`poulpy_core`] (`GLWELinearTransform` / `GLWELinearTransformOps`); this layer
//! re-exports the data type under CKKS names and adds the scale-aware entry point.
//! All `log_delta` / `log_budget` accounting lives here — the core engine only
//! receives base2k-level alignment integers. See
//! [`docs/lt_bsgs.md`](https://github.com/poulpy-fhe/poulpy/blob/main/docs/lt_bsgs.md).

use anyhow::Result;
use poulpy_core::layouts::{
    GGLWEInfos, GGLWEPreparedToBackendRef, GLWEAutomorphismKeyHelper, GLWEToBackendMut, GLWEToBackendRef, GetGaloisElement,
    prepared::GLWEAutomorphismKeyPreparedToBackendRef,
};
use poulpy_hal::layouts::{Backend, ScratchArena};

use crate::{CKKSCtBounds, SetCKKSInfos};

pub use poulpy_core::{
    GLWELinearTransform as LinearTransformation, GLWELinearTransformDiagonal as Diagonal,
    GLWELinearTransformGiantStep as GiantStep, GLWELinearTransformIndex as LinearTransformationIndex,
    GLWEPreparedBabyRotations as PreparedBabyRotations, GLWEPreparedBabyStepHelper as PreparedBabyStepHelper,
    GLWEPreparedLinearTransform as PreparedLinearTransformation, GLWEPreparedLinearTransformGiantStep as PreparedGiantStep,
    LinearTransformationStrategy, bsgs_index, linear_transform_index, normalize_linear_transform_diagonal,
    optimal_bsgs_giant_step,
};

/// Homomorphic evaluation of a [`LinearTransformation`] on a CKKS ciphertext.
pub trait LinearTransformationOps<BE: Backend> {
    /// Scratch bytes required by [`Self::ckks_prepare_linear_transformation`].
    fn ckks_prepare_linear_transformation_tmp_bytes<P>(&self, lt: &LinearTransformation<P>) -> usize
    where
        P: CKKSCtBounds;

    /// Scratch bytes required by [`Self::ckks_eval_linear_transformation_into`].
    fn ckks_eval_linear_transformation_tmp_bytes<C, K>(&self, ct: &C, key: &K) -> usize
    where
        C: CKKSCtBounds,
        K: GGLWEInfos;

    /// Scratch bytes required by [`Self::ckks_prepare_baby_rotations`].
    fn ckks_prepare_baby_rotations_tmp_bytes<C, K>(&self, ct: &C, key: &K) -> usize
    where
        C: CKKSCtBounds,
        K: GGLWEInfos;

    /// Prepares `lt` into a reusable right-operand cache for repeated evaluation.
    fn ckks_prepare_linear_transformation<P>(
        &self,
        lt: &LinearTransformation<P>,
        prepared: &mut PreparedLinearTransformation<BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        P: GLWEToBackendRef<BE> + CKKSCtBounds;

    /// Computes and prepares the requested baby-step rotations of `src`.
    ///
    /// This is the reusable left-operand cache consumed by the prepared BSGS
    /// inner sums. Only the final `CnvPVecL` baby steps are owned by the return
    /// value; intermediate rotated ciphertexts are scratch-backed.
    fn ckks_prepare_baby_rotations<Src, H, K>(
        &self,
        baby_steps: &[i64],
        src: &Src,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<PreparedBabyRotations<BE>>
    where
        Src: GLWEToBackendRef<BE> + CKKSCtBounds,
        K: GLWEAutomorphismKeyPreparedToBackendRef<BE> + GGLWEPreparedToBackendRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>;

    /// Computes `dst = M · src`, where `M` is the linear map encoded by `lt`.
    ///
    /// `keys` must contain an automorphism key, keyed by rotation amount, for every
    /// rotation returned by [`LinearTransformation::required_rotations`].
    ///
    /// This borrowed one-shot API prepares the transform internally. For
    /// repeated evaluation, prepare the transform once with
    /// [`Self::ckks_prepare_linear_transformation`] and call
    /// [`Self::ckks_eval_prepared_linear_transformation_into`].
    fn ckks_eval_linear_transformation_into<Dst, Src, P, H, K>(
        &self,
        dst: &mut Dst,
        src: &Src,
        lt: &LinearTransformation<P>,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + CKKSCtBounds,
        P: GLWEToBackendRef<BE> + CKKSCtBounds,
        K: GLWEAutomorphismKeyPreparedToBackendRef<BE> + GGLWEPreparedToBackendRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>;

    /// Computes `dst = M · src` with `lt` and its prepared cache.
    fn ckks_eval_prepared_linear_transformation_into<Dst, Src, P, H, K>(
        &self,
        dst: &mut Dst,
        src: &Src,
        lt: &LinearTransformation<P>,
        prepared: &PreparedLinearTransformation<BE>,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + CKKSCtBounds,
        P: GLWEToBackendRef<BE> + CKKSCtBounds,
        K: GLWEAutomorphismKeyPreparedToBackendRef<BE> + GGLWEPreparedToBackendRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>;

    /// Computes `dst = M · src` with `lt`, its prepared cache, and a
    /// precomputed baby-step cache for `src`.
    ///
    /// `src` is used for CKKS scale/budget metadata; the ciphertext left
    /// operands consumed by the BSGS inner sums come from `babies`.
    fn ckks_eval_prepared_linear_transformation_with_babies_into<Dst, Src, P, H, K>(
        &self,
        dst: &mut Dst,
        src: &Src,
        lt: &LinearTransformation<P>,
        prepared: &PreparedLinearTransformation<BE>,
        babies: &PreparedBabyRotations<BE>,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + CKKSCtBounds,
        P: GLWEToBackendRef<BE> + CKKSCtBounds,
        K: GLWEAutomorphismKeyPreparedToBackendRef<BE> + GGLWEPreparedToBackendRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>;

    /// Computes `dst = M · dst`.
    fn ckks_eval_linear_transformation_assign<Dst, P, H, K>(
        &self,
        dst: &mut Dst,
        lt: &LinearTransformation<P>,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        P: GLWEToBackendRef<BE> + CKKSCtBounds,
        K: GLWEAutomorphismKeyPreparedToBackendRef<BE> + GGLWEPreparedToBackendRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>;

    /// Computes `dst = M · dst` with `lt` and its prepared cache.
    fn ckks_eval_prepared_linear_transformation_assign<Dst, P, H, K>(
        &self,
        dst: &mut Dst,
        lt: &LinearTransformation<P>,
        prepared: &PreparedLinearTransformation<BE>,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        P: GLWEToBackendRef<BE> + CKKSCtBounds,
        K: GLWEAutomorphismKeyPreparedToBackendRef<BE> + GGLWEPreparedToBackendRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>;

    /// Computes `dsts[i] = transforms[i] · src` for several prepared transform caches.
    fn ckks_eval_many_prepared_linear_transformations_into<Dst, Src, P, H, K>(
        &self,
        dsts: &mut [Dst],
        src: &Src,
        transforms: &[LinearTransformation<P>],
        prepared_transforms: &[PreparedLinearTransformation<BE>],
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + CKKSCtBounds,
        P: GLWEToBackendRef<BE> + CKKSCtBounds,
        K: GLWEAutomorphismKeyPreparedToBackendRef<BE> + GGLWEPreparedToBackendRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>;

    /// Computes `dst = M_n(...M_1(M_0(src)))` with prepared transform caches,
    /// using the normal per-step CKKS linear-transform scale handling.
    fn ckks_eval_sequential_prepared_linear_transformations_into<Dst, Src, P, H, K>(
        &self,
        dst: &mut Dst,
        src: &Src,
        transforms: &[LinearTransformation<P>],
        prepared_transforms: &[PreparedLinearTransformation<BE>],
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + CKKSCtBounds,
        P: GLWEToBackendRef<BE> + CKKSCtBounds,
        K: GLWEAutomorphismKeyPreparedToBackendRef<BE> + GGLWEPreparedToBackendRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>;
}
