#![allow(clippy::too_many_arguments)]

use poulpy_hal::{
    api::VecZnxDftAddAssign,
    layouts::{Backend, ScratchArena, VecZnxDftBackendMut},
};

use crate::layouts::{
    GGLWEInfos, GGLWEToBackendMut, GGLWEToBackendRef, GGSWInfos, GGSWToBackendMut, GGSWToBackendRef, GLWEInfos, GLWEToBackendMut,
    GLWEToBackendRef, LWEInfos, LWEToBackendMut, LWEToBackendRef,
    prepared::{GGLWEPreparedBackendRef, GGLWEToGGSWKeyPreparedBackendRef},
};

pub trait GLWEKeyswitch<BE: Backend> {
    fn glwe_keyswitch_tmp_bytes<R, A, B>(&self, res_infos: &R, a_infos: &A, key_infos: &B) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        B: GGLWEInfos;

    fn glwe_keyswitch<R, A>(&self, res: &mut R, a: &A, key: &GGLWEPreparedBackendRef<'_, BE>, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        A: GLWEToBackendRef<BE> + GLWEInfos;

    fn glwe_keyswitch_assign<R>(&self, res: &mut R, key: &GGLWEPreparedBackendRef<'_, BE>, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GLWEToBackendMut<BE> + GLWEInfos;

    /// Scratch of [`Self::glwe_keyswitch_dft`] for an accumulator of `res_size` limbs.
    fn glwe_keyswitch_dft_tmp_bytes<A, K>(&self, res_size: usize, a_infos: &A, key_infos: &K) -> usize
    where
        A: GLWEInfos,
        K: GGLWEInfos;

    /// The gadget product of a key switch of `a = (b, a_1, ..., a_rank)`, in the DFT
    /// domain over the limbs of `res`: `res = Σ_i decomp(a_i) · key_i`, the mask
    /// columns `a_i` decomposed and multiplied by the rows of the key, which is the
    /// whole switch but for the body `b`. Several products summed with
    /// [`Self::glwe_keyswitch_dft_add`] are finished by one `vec_znx_idft_normalize`
    /// per column, with the sum of their bodies, each lowered by `key.k_offset()`, as
    /// the addend of column 0. `a` is at the key radix; a key with a row offset takes
    /// an input of at most its `max_input_k`.
    fn glwe_keyswitch_dft<A>(
        &self,
        res: &mut VecZnxDftBackendMut<'_, BE>,
        a: &A,
        key: &GGLWEPreparedBackendRef<'_, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        A: GLWEToBackendRef<BE> + GLWEInfos;

    /// Scratch of [`Self::glwe_keyswitch_dft_add`] for an accumulator of `res_size`
    /// limbs: a product of that size and the scratch of [`Self::glwe_keyswitch_dft`].
    fn glwe_keyswitch_dft_add_tmp_bytes<A, K>(&self, res_size: usize, a_infos: &A, key_infos: &K) -> usize
    where
        A: GLWEInfos,
        K: GGLWEInfos;

    /// `res += Σ_i decomp(a_i) · key_i`, the product of [`Self::glwe_keyswitch_dft`]
    /// added to `res`.
    fn glwe_keyswitch_dft_add<A>(
        &self,
        res: &mut VecZnxDftBackendMut<'_, BE>,
        a: &A,
        key: &GGLWEPreparedBackendRef<'_, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        A: GLWEToBackendRef<BE> + GLWEInfos,
        Self: VecZnxDftAddAssign<BE>;
}

pub trait GGLWEKeyswitch<BE: Backend> {
    fn gglwe_keyswitch_tmp_bytes<R, A, K>(&self, res_infos: &R, a_infos: &A, key_infos: &K) -> usize
    where
        R: GGLWEInfos,
        A: GGLWEInfos,
        K: GGLWEInfos;

    fn gglwe_keyswitch<R, A>(
        &self,
        res: &mut R,
        a: &A,
        key: &GGLWEPreparedBackendRef<'_, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GGLWEToBackendMut<BE> + GGLWEInfos,
        A: GGLWEToBackendRef<BE> + GGLWEInfos;

    fn gglwe_keyswitch_assign<R>(&self, res: &mut R, key: &GGLWEPreparedBackendRef<'_, BE>, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GGLWEToBackendMut<BE> + GGLWEInfos;
}

pub trait GGSWKeyswitch<BE: Backend> {
    fn ggsw_keyswitch_tmp_bytes<R, A, K, T>(&self, res_infos: &R, a_infos: &A, key_infos: &K, tsk_infos: &T) -> usize
    where
        R: GGSWInfos,
        A: GGSWInfos,
        K: GGLWEInfos,
        T: GGLWEInfos;

    fn ggsw_keyswitch<R, A>(
        &self,
        res: &mut R,
        a: &A,
        key: &GGLWEPreparedBackendRef<'_, BE>,
        tsk: &GGLWEToGGSWKeyPreparedBackendRef<'_, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GGSWToBackendMut<BE> + GGSWInfos,
        A: GGSWToBackendRef<BE> + GGSWInfos;

    fn ggsw_keyswitch_assign<R>(
        &self,
        res: &mut R,
        key: &GGLWEPreparedBackendRef<'_, BE>,
        tsk: &GGLWEToGGSWKeyPreparedBackendRef<'_, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GGSWToBackendMut<BE> + GGSWInfos;
}

pub trait LWEKeyswitch<BE: Backend> {
    fn lwe_keyswitch_tmp_bytes<R, A, K>(&self, res_infos: &R, a_infos: &A, key_infos: &K) -> usize
    where
        R: LWEInfos,
        A: LWEInfos,
        K: GGLWEInfos;

    fn lwe_keyswitch<R, A>(&self, res: &mut R, a: &A, ksk: &GGLWEPreparedBackendRef<'_, BE>, scratch: &mut ScratchArena<'_, BE>)
    where
        R: LWEToBackendMut<BE> + LWEInfos,
        A: LWEToBackendRef<BE> + LWEInfos;
}
