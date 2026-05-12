use poulpy_hal::{
    api::{ModuleN, ScratchArenaTakeBasic, SvpPPolBytesOf, VmpPMatBytesOf},
    layouts::{Backend, ScratchArena},
};

use crate::{
    dist::Distribution,
    layouts::{
        Degree, GGLWE, GGLWEInfos, GGLWEPreparedViewMut, GGLWEViewMut, GGSW, GGSWInfos, GGSWPreparedViewMut, GGSWViewMut, GLWE,
        GLWEInfos, GLWEPlaintext, GLWEPlaintextViewMut, GLWESecret, GLWESecretPreparedViewMut, GLWESecretTensor,
        GLWESecretTensorViewMut, GLWESecretViewMut, GLWETensor, GLWETensorViewMut, GLWEViewMut, LWE, LWEInfos, LWEPlaintext,
        LWEPlaintextViewMut, LWEViewMut, Rank,
        prepared::{GGLWEPrepared, GGSWPrepared, GLWESecretPrepared},
    },
};

/// Backend-native arena allocation for core ciphertext/key layouts.
///
/// Returns backend-native borrows (`B::BufMut<'a>`) carved from a [`ScratchArena`].
pub trait ScratchArenaTakeCore<'a, B: Backend>: ScratchArenaTakeBasic<'a, B> + Sized {
    /// Allocates an [`LWE`] ciphertext from scratch space.
    fn take_lwe_scratch<A>(self, infos: &A) -> (LWEViewMut<'a, B>, Self)
    where
        B: 'a,
        A: LWEInfos,
    {
        let (data, scratch) = self.take_vec_znx_scratch(infos.n().into(), 1, infos.size());
        (
            LWEViewMut::from_inner(LWE {
                base2k: infos.base2k(),
                data: data.into_inner(),
            }),
            scratch,
        )
    }

    /// Allocates an [`LWEPlaintext`] from scratch space.
    fn take_lwe_plaintext_scratch<A>(self, infos: &A) -> (LWEPlaintextViewMut<'a, B>, Self)
    where
        B: 'a,
        A: LWEInfos,
    {
        let (data, scratch) = self.take_vec_znx_scratch(1, 1, infos.size());
        (
            LWEPlaintextViewMut::from_inner(LWEPlaintext {
                base2k: infos.base2k(),
                data: data.into_inner(),
            }),
            scratch,
        )
    }

    /// Allocates a [`GLWE`] ciphertext from scratch space.
    fn take_glwe_scratch<A>(self, infos: &A) -> (GLWEViewMut<'a, B>, Self)
    where
        B: 'a,
        A: GLWEInfos,
    {
        let (data, scratch) = self.take_vec_znx_scratch(infos.n().into(), (infos.rank() + 1).into(), infos.size());
        (
            GLWEViewMut::from_inner(GLWE {
                base2k: infos.base2k(),
                data: data.into_inner(),
            }),
            scratch,
        )
    }

    /// Allocates a `Vec` of `size` [`GLWE`] ciphertexts from scratch space.
    fn take_glwe_slice_scratch<A>(self, size: usize, infos: &A) -> (Vec<GLWEViewMut<'a, B>>, Self)
    where
        B: 'a,
        A: GLWEInfos,
    {
        let mut scratch: Self = self;
        let mut cts: Vec<GLWEViewMut<'a, B>> = Vec::with_capacity(size);
        for _ in 0..size {
            let (ct, new_scratch) = scratch.take_glwe_scratch(infos);
            scratch = new_scratch;
            cts.push(ct);
        }
        (cts, scratch)
    }

    /// Allocates a [`GLWETensor`] from scratch space.
    fn take_glwe_tensor_scratch<A>(self, infos: &A) -> (GLWETensorViewMut<'a, B>, Self)
    where
        B: 'a,
        A: GLWEInfos,
    {
        let cols: usize = infos.rank().as_usize() + 1;
        let pairs: usize = (((cols + 1) * cols) >> 1).max(1);
        let (data, scratch) = self.take_vec_znx_scratch(infos.n().into(), pairs, infos.size());
        (
            GLWETensorViewMut::from_inner(GLWETensor {
                base2k: infos.base2k(),
                rank: infos.rank(),
                data: data.into_inner(),
            }),
            scratch,
        )
    }

    /// Allocates a [`GLWEPlaintext`] from scratch space.
    fn take_glwe_plaintext_scratch<A>(self, infos: &A) -> (GLWEPlaintextViewMut<'a, B>, Self)
    where
        B: 'a,
        A: GLWEInfos,
    {
        let (data, scratch) = self.take_vec_znx_scratch(infos.n().into(), 1, infos.size());
        (
            GLWEPlaintextViewMut::from_inner(GLWEPlaintext {
                base2k: infos.base2k(),
                data: data.into_inner(),
            }),
            scratch,
        )
    }

    /// Allocates a [`GLWESecretPrepared`] (DFT-domain secret key) from scratch space.
    fn take_glwe_secret_prepared_scratch<M>(self, module: &M, rank: Rank) -> (GLWESecretPreparedViewMut<'a, B>, Self)
    where
        B: 'a,
        M: ModuleN + SvpPPolBytesOf,
    {
        let (data, scratch) = self.take_svp_ppol_scratch(module, rank.into());
        (
            GLWESecretPreparedViewMut::from_inner(GLWESecretPrepared {
                data: data.into_inner(),
                dist: Distribution::NONE,
            }),
            scratch,
        )
    }

    /// Allocates a [`GLWESecret`] from scratch space.
    fn take_glwe_secret_scratch(self, n: Degree, rank: Rank) -> (GLWESecretViewMut<'a, B>, Self)
    where
        B: 'a,
    {
        let (data, scratch) = self.take_scalar_znx_scratch(n.into(), rank.into());
        (
            GLWESecretViewMut::from_inner(GLWESecret {
                data: data.into_inner(),
                dist: Distribution::NONE,
            }),
            scratch,
        )
    }

    /// Allocates a [`GLWESecretTensor`] from scratch space.
    fn take_glwe_secret_tensor_scratch(self, n: Degree, rank: Rank) -> (GLWESecretTensorViewMut<'a, B>, Self)
    where
        B: 'a,
    {
        let (data, scratch) = self.take_scalar_znx_scratch(n.into(), GLWESecretTensor::pairs(rank.into()));
        (
            GLWESecretTensorViewMut::from_inner(GLWESecretTensor {
                data: data.into_inner(),
                rank,
                dist: Distribution::NONE,
            }),
            scratch,
        )
    }

    /// Allocates a [`GGLWE`] ciphertext from scratch space.
    fn take_gglwe_scratch<A>(self, infos: &A) -> (GGLWEViewMut<'a, B>, Self)
    where
        B: 'a,
        A: GGLWEInfos,
    {
        let (data, scratch) = self.take_mat_znx_scratch(
            infos.n().into(),
            infos.dnum().0.div_ceil(infos.dsize().0) as usize,
            infos.rank_in().into(),
            (infos.rank_out() + 1).into(),
            infos.size(),
        );
        (
            GGLWEViewMut::from_inner(GGLWE {
                base2k: infos.base2k(),
                dsize: infos.dsize(),
                data: data.into_inner(),
            }),
            scratch,
        )
    }

    /// Allocates a [`GGLWEPrepared`] (DFT-domain GGLWE) from scratch space.
    fn take_gglwe_prepared_scratch<A, M>(self, module: &M, infos: &A) -> (GGLWEPreparedViewMut<'a, B>, Self)
    where
        B: 'a,
        A: GGLWEInfos,
        M: ModuleN + VmpPMatBytesOf,
    {
        assert_eq!(module.n() as u32, infos.n());
        let (data, scratch) = self.take_vmp_pmat_scratch(
            module,
            infos.dnum().into(),
            infos.rank_in().into(),
            (infos.rank_out() + 1).into(),
            infos.size(),
        );
        (
            GGLWEPreparedViewMut::from_inner(GGLWEPrepared {
                base2k: infos.base2k(),
                dsize: infos.dsize(),
                data: data.into_inner(),
            }),
            scratch,
        )
    }

    /// Allocates a [`GGSW`] ciphertext from scratch space.
    fn take_ggsw_scratch<A>(self, infos: &A) -> (GGSWViewMut<'a, B>, Self)
    where
        B: 'a,
        A: GGSWInfos,
    {
        let (data, scratch) = self.take_mat_znx_scratch(
            infos.n().into(),
            infos.dnum().into(),
            (infos.rank() + 1).into(),
            (infos.rank() + 1).into(),
            infos.size(),
        );
        (
            GGSWViewMut::from_inner(GGSW {
                base2k: infos.base2k(),
                dsize: infos.dsize(),
                data: data.into_inner(),
            }),
            scratch,
        )
    }

    /// Allocates a [`GGSWPrepared`] (DFT-domain GGSW) from scratch space.
    fn take_ggsw_prepared_scratch<A, M>(self, module: &M, infos: &A) -> (GGSWPreparedViewMut<'a, B>, Self)
    where
        B: 'a,
        A: GGSWInfos,
        M: ModuleN + VmpPMatBytesOf,
    {
        assert_eq!(module.n() as u32, infos.n());
        let (data, scratch) = self.take_vmp_pmat_scratch(
            module,
            infos.dnum().into(),
            (infos.rank() + 1).into(),
            (infos.rank() + 1).into(),
            infos.size(),
        );
        (
            GGSWPreparedViewMut::from_inner(GGSWPrepared {
                base2k: infos.base2k(),
                dsize: infos.dsize(),
                data: data.into_inner(),
            }),
            scratch,
        )
    }
}

impl<'a, B: Backend> ScratchArenaTakeCore<'a, B> for ScratchArena<'a, B> {}
