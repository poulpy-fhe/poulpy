#![allow(clippy::too_many_arguments)]

use poulpy_hal::layouts::{Backend, ScratchArena, VecZnxDftBackendMut, VecZnxDftBackendRef};

use crate::layouts::{
    GGLWEInfos, GGLWEToBackendMut, GGLWEToBackendRef, GGSWInfos, GGSWToBackendMut, GGSWToBackendRef, GLWEBigToBackendMut,
    GLWEBigToBackendRef, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, LWEInfos, LWEToBackendMut, LWEToBackendRef,
    prepared::{GGLWEPreparedToBackendRef, GGLWEToGGSWKeyPreparedToBackendRef},
};

pub trait GLWEKeyswitch<BE: Backend> {
    fn glwe_keyswitch_tmp_bytes<R, A, B>(&self, res_infos: &R, a_infos: &A, key_infos: &B) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        B: GGLWEInfos;

    fn glwe_keyswitch<R, A, K>(&self, res: &mut R, a: &A, key: &K, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        A: GLWEToBackendRef<BE> + GLWEInfos,
        K: GGLWEPreparedToBackendRef<BE> + GGLWEInfos;

    fn glwe_keyswitch_assign<R, K>(&self, res: &mut R, key: &K, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        K: GGLWEPreparedToBackendRef<BE> + GGLWEInfos;
}

/// Keyswitch that stops in the big domain, leaving normalization to the caller.
///
/// Additive: the ergonomic [`GLWEKeyswitch`] is implemented as this stage
/// followed by [`GLWEFinalizeBig`], so the two cannot drift apart.
///
/// Carve the destination with
/// [`glwe_keyswitch_big_layout`](crate::layouts::glwe_keyswitch_big_layout),
/// which is pure layout arithmetic and therefore not part of this override
/// surface.
pub trait GLWEKeyswitchIntoBig<BE: Backend> {
    fn glwe_keyswitch_into_big_tmp_bytes<R, A, K>(&self, res_infos: &R, a_infos: &A, key_infos: &K) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        K: GGLWEInfos;

    fn glwe_keyswitch_into_big<R, A, K>(&self, res_big: &mut R, a: &A, key: &K, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GLWEBigToBackendMut<BE> + GLWEInfos,
        A: GLWEToBackendRef<BE> + GLWEInfos,
        K: GGLWEPreparedToBackendRef<BE> + GGLWEInfos;

    /// Transforms `a`'s mask columns into `res`, which must hold `a.rank()`
    /// columns. Call once, then feed `res` to [`Self::glwe_keyswitch_from_mask_into_big`]
    /// for each key: that is what makes a hoisted evaluation cheaper than one
    /// keyswitch per key.
    fn glwe_mask_dft_apply<A>(&self, res: &mut VecZnxDftBackendMut<'_, BE>, a: &A)
    where
        A: GLWEToBackendRef<BE> + GLWEInfos;

    fn glwe_keyswitch_from_mask_into_big_tmp_bytes<R, A, K>(&self, res_infos: &R, a_infos: &A, key_infos: &K) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        K: GGLWEInfos;

    /// [`Self::glwe_keyswitch_into_big`] from an already-transformed mask.
    fn glwe_keyswitch_from_mask_into_big<R, A, K>(
        &self,
        res_big: &mut R,
        mask_dft: &VecZnxDftBackendRef<'_, BE>,
        a: &A,
        key: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEBigToBackendMut<BE> + GLWEInfos,
        A: GLWEToBackendRef<BE> + GLWEInfos,
        K: GGLWEPreparedToBackendRef<BE> + GGLWEInfos;
}

/// Carries a big-domain accumulator back to the coefficient domain, at the
/// destination's own precision.
///
/// This is the single point where an accumulator becomes a precision claim: the
/// target precision is read from `res`, never from `a_big`.
pub trait GLWEFinalizeBig<BE: Backend> {
    fn glwe_finalize_big_tmp_bytes(&self) -> usize;

    fn glwe_finalize_big_into<R, A>(&self, res: &mut R, a_big: &A, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        A: GLWEBigToBackendRef<BE> + GLWEInfos;
}

pub trait GGLWEKeyswitch<BE: Backend> {
    fn gglwe_keyswitch_tmp_bytes<R, A, K>(&self, res_infos: &R, a_infos: &A, key_infos: &K) -> usize
    where
        R: GGLWEInfos,
        A: GGLWEInfos,
        K: GGLWEInfos;

    fn gglwe_keyswitch<R, A, B>(&self, res: &mut R, a: &A, b: &B, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GGLWEToBackendMut<BE> + GGLWEInfos,
        A: GGLWEToBackendRef<BE> + GGLWEInfos,
        B: GGLWEPreparedToBackendRef<BE> + GGLWEInfos;

    fn gglwe_keyswitch_assign<R, A>(&self, res: &mut R, a: &A, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GGLWEToBackendMut<BE> + GGLWEInfos,
        A: GGLWEPreparedToBackendRef<BE> + GGLWEInfos;
}

pub trait GGSWKeyswitch<BE: Backend> {
    fn ggsw_keyswitch_tmp_bytes<R, A, K, T>(&self, res_infos: &R, a_infos: &A, key_infos: &K, tsk_infos: &T) -> usize
    where
        R: GGSWInfos,
        A: GGSWInfos,
        K: GGLWEInfos,
        T: GGLWEInfos;

    fn ggsw_keyswitch<R, A, K, T>(&self, res: &mut R, a: &A, key: &K, tsk: &T, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GGSWToBackendMut<BE> + GGSWInfos,
        A: GGSWToBackendRef<BE> + GGSWInfos,
        K: GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
        T: GGLWEToGGSWKeyPreparedToBackendRef<BE> + GGLWEInfos;

    fn ggsw_keyswitch_assign<R, K, T>(&self, res: &mut R, key: &K, tsk: &T, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GGSWToBackendMut<BE> + GGSWInfos,
        K: GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
        T: GGLWEToGGSWKeyPreparedToBackendRef<BE> + GGLWEInfos;
}

pub trait LWEKeyswitch<BE: Backend> {
    fn lwe_keyswitch_tmp_bytes<R, A, K>(&self, res_infos: &R, a_infos: &A, key_infos: &K) -> usize
    where
        R: LWEInfos,
        A: LWEInfos,
        K: GGLWEInfos;

    fn lwe_keyswitch<R, A, K>(&self, res: &mut R, a: &A, ksk: &K, scratch: &mut ScratchArena<'_, BE>)
    where
        R: LWEToBackendMut<BE> + LWEInfos,
        A: LWEToBackendRef<BE> + LWEInfos,
        K: GGLWEPreparedToBackendRef<BE> + GGLWEInfos;
}
