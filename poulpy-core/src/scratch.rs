use std::collections::{BTreeMap, btree_map::Entry};
use std::marker::PhantomData;

use poulpy_hal::{
    api::{
        ModuleN, ScratchArenaTakeCnvPVecL, ScratchArenaTakeMatZnx, ScratchArenaTakeScalarZnx, ScratchArenaTakeSvpPPol,
        ScratchArenaTakeVecZnx, ScratchArenaTakeVmpPMat, VmpPMatBytesOf,
    },
    layouts::{Backend, CnvPVecLViewMut, PrepareHint, ScratchArena},
};

use crate::layouts::operand_degree;
use crate::{
    dist::Distribution,
    layouts::{
        Degree, GGLWE, GGLWEInfos, GGLWEPreparedViewMut, GGLWEViewMut, GGSW, GGSWInfos, GGSWPreparedViewMut, GGSWViewMut, GLWE,
        GLWEInfos, GLWEPlaintext, GLWEPlaintextViewMut, GLWESecret, GLWESecretPreparedViewMut, GLWESecretTensor,
        GLWESecretTensorViewMut, GLWESecretViewMut, GLWETensor, GLWETensorViewMut, GLWEViewMut, LWE, LWEInfos, LWEPlaintext,
        LWEPlaintextViewMut, LWEViewMut, LinearTransformationPlan, Rank,
        prepared::{GGLWEPrepared, GGSWPrepared, GLWESecretPrepared, LinearTransformationBabySteps},
    },
};

/// Backend-native arena allocation for core ciphertext/key layouts.
///
/// Returns backend-native borrows (`B::BufMut<'a>`) carved from a [`ScratchArena`].
pub trait ScratchArenaTakeGLWELinearTransformationBabySteps<'a, B: Backend>: ScratchArenaTakeCnvPVecL<'a, B> + Sized {
    /// Carves the baby-step cache of the transformation schedule `plan` (a transformation's
    /// [`index`](crate::layouts::LinearTransformation::index)) for a source of the layout `a`, in
    /// [`GLWELinearTransformationBabyStepsBytesOf::glwe_linear_transformation_baby_steps_bytes_of`](crate::GLWELinearTransformationBabyStepsBytesOf::glwe_linear_transformation_baby_steps_bytes_of)
    /// bytes. Its operands are arbitrary until a preparation fills them.
    ///
    /// # Panics
    ///
    /// When the module does not serve the degree of `a`, or the arena is below that size.
    fn take_glwe_linear_transformation_baby_steps_scratch<M, A>(
        self,
        module: &M,
        plan: &LinearTransformationPlan,
        a: &A,
    ) -> (LinearTransformationBabySteps<B, CnvPVecLViewMut<'a, B>>, Self)
    where
        B: 'a,
        M: ModuleN,
        A: GLWEInfos,
    {
        let n: usize = operand_degree(module.n(), &[a.n()]);
        let (cols, size) = (a.rank().as_usize() + 1, a.size());
        let mut values = BTreeMap::new();
        let mut scratch = self;
        for &rot in &plan.baby_steps {
            if let Entry::Vacant(entry) = values.entry(rot) {
                let (operand, rest) = scratch.take_cnv_pvec_left_scratch(n, cols, size, PrepareHint::Reuse);
                entry.insert(operand);
                scratch = rest;
            }
        }
        (
            LinearTransformationBabySteps {
                values,
                rotations: plan.rotations,
                k: a.k(),
                _backend: PhantomData,
            },
            scratch,
        )
    }
}

/// Carves [`LWE`] layouts from a [`ScratchArena`].
pub trait ScratchArenaTakeLWE<'a, B: Backend>: ScratchArenaTakeVecZnx<'a, B> + Sized {
    /// Allocates an [`LWE`] ciphertext from scratch space.
    fn take_lwe_scratch<A>(self, infos: &A) -> (LWEViewMut<'a, B>, Self)
    where
        B: 'a,
        A: LWEInfos,
    {
        let (body, scratch_1) = self.take_vec_znx_scratch(1, 1, infos.size());
        let (mask, scratch) = scratch_1.take_vec_znx_scratch(infos.n().into(), 1, infos.size());
        (
            LWEViewMut::from_inner(LWE {
                noise: None,
                base2k: infos.base2k(),
                k: infos.k(),
                body: body.into_inner(),
                mask: mask.into_inner(),
            }),
            scratch,
        )
    }
}

/// Carves [`LWEPlaintext`] layouts from a [`ScratchArena`].
pub trait ScratchArenaTakeLWEPlaintext<'a, B: Backend>: ScratchArenaTakeVecZnx<'a, B> + Sized {
    /// Allocates an [`LWEPlaintext`] from scratch space.
    fn take_lwe_plaintext_scratch<A>(self, infos: &A) -> (LWEPlaintextViewMut<'a, B>, Self)
    where
        B: 'a,
        A: LWEInfos,
    {
        let (data, scratch) = self.take_vec_znx_scratch(1, 1, infos.size());
        (
            LWEPlaintextViewMut::from_inner(LWEPlaintext {
                k: infos.k(),
                base2k: infos.base2k(),
                data: data.into_inner(),
            }),
            scratch,
        )
    }
}

/// Carves [`GLWE`] layouts from a [`ScratchArena`].
pub trait ScratchArenaTakeGLWE<'a, B: Backend>: ScratchArenaTakeVecZnx<'a, B> + Sized {
    /// Allocates a [`GLWE`] ciphertext from scratch space.
    fn take_glwe_scratch<A>(self, infos: &A) -> (GLWEViewMut<'a, B>, Self)
    where
        B: 'a,
        A: GLWEInfos,
    {
        let (data, scratch) = self.take_vec_znx_scratch(infos.n().into(), (infos.rank() + 1).into(), infos.size());
        (
            GLWEViewMut::from_inner(GLWE {
                noise: None,
                k: infos.k(),
                base2k: infos.base2k(),
                canonical: false,
                plaintext_meta: None,
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
}

/// Carves [`GLWETensor`] layouts from a [`ScratchArena`].
pub trait ScratchArenaTakeGLWETensor<'a, B: Backend>: ScratchArenaTakeVecZnx<'a, B> + Sized {
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
                k: infos.k(),
                base2k: infos.base2k(),
                rank: infos.rank(),
                data: data.into_inner(),
            }),
            scratch,
        )
    }
}

/// Carves [`GLWEPlaintext`] layouts from a [`ScratchArena`].
pub trait ScratchArenaTakeGLWEPlaintext<'a, B: Backend>: ScratchArenaTakeVecZnx<'a, B> + Sized {
    /// Allocates a [`GLWEPlaintext`] from scratch space.
    fn take_glwe_plaintext_scratch<A>(self, infos: &A) -> (GLWEPlaintextViewMut<'a, B>, Self)
    where
        B: 'a,
        A: GLWEInfos,
    {
        let (data, scratch) = self.take_vec_znx_scratch(infos.n().into(), 1, infos.size());
        (
            GLWEPlaintextViewMut::from_inner(GLWEPlaintext {
                k: infos.k(),
                plaintext_meta: None,
                base2k: infos.base2k(),
                data: data.into_inner(),
            }),
            scratch,
        )
    }
}

/// Carves [`GLWESecretPrepared`] layouts from a [`ScratchArena`].
pub trait ScratchArenaTakeGLWESecretPrepared<'a, B: Backend>: ScratchArenaTakeSvpPPol<'a, B> + Sized {
    /// Allocates a [`GLWESecretPrepared`] (DFT-domain secret key) from scratch space.
    fn take_glwe_secret_prepared_scratch(self, n: Degree, rank: Rank) -> (GLWESecretPreparedViewMut<'a, B>, Self)
    where
        B: 'a,
    {
        let (data, scratch) = self.take_svp_ppol_scratch(n.into(), rank.into(), PrepareHint::Reuse);
        (
            GLWESecretPreparedViewMut::from_inner(GLWESecretPrepared {
                data: data.into_inner(),
                dist: Distribution::NONE,
            }),
            scratch,
        )
    }
}

/// Carves [`GLWESecret`] layouts from a [`ScratchArena`].
pub trait ScratchArenaTakeGLWESecret<'a, B: Backend>: ScratchArenaTakeScalarZnx<'a, B> + Sized {
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
}

/// Carves [`GLWESecretTensor`] layouts from a [`ScratchArena`].
pub trait ScratchArenaTakeGLWESecretTensor<'a, B: Backend>: ScratchArenaTakeScalarZnx<'a, B> + Sized {
    /// Allocates a [`GLWESecretTensor`] from scratch space.
    fn take_glwe_secret_tensor_scratch(self, n: Degree, rank: Rank) -> (GLWESecretTensorViewMut<'a, B>, Self)
    where
        B: 'a,
    {
        let (data, scratch) = self.take_scalar_znx_scratch(n.into(), crate::layouts::pairs(rank.into()));
        (
            GLWESecretTensorViewMut::from_inner(GLWESecretTensor {
                data: data.into_inner(),
                rank,
                dist: Distribution::NONE,
            }),
            scratch,
        )
    }
}

/// Carves [`GGLWE`] layouts from a [`ScratchArena`].
pub trait ScratchArenaTakeGGLWE<'a, B: Backend>: ScratchArenaTakeMatZnx<'a, B> + Sized {
    /// Allocates a [`GGLWE`] ciphertext from scratch space.
    fn take_gglwe_scratch<A>(self, infos: &A) -> (GGLWEViewMut<'a, B>, Self)
    where
        B: 'a,
        A: GGLWEInfos,
    {
        // The shape of an owned GGLWE of these infos: a row per digit.
        let (data, scratch) = self.take_mat_znx_scratch(
            infos.n().into(),
            infos.dnum().into(),
            infos.rank_in().into(),
            (infos.rank_out() + 1).into(),
            infos.size(),
        );
        (
            GGLWEViewMut::from_inner(GGLWE {
                noise: None,
                k_aux: infos.k_aux(),
                base2k: infos.base2k(),
                k_digit: infos.k_digit(),
                k_offset: infos.k_offset(),
                data: data.into_inner(),
            }),
            scratch,
        )
    }
}

/// Carves [`GGLWEPrepared`] layouts from a [`ScratchArena`].
pub trait ScratchArenaTakeGGLWEPrepared<'a, B: Backend>: ScratchArenaTakeVmpPMat<'a, B> + Sized {
    /// Allocates a [`GGLWEPrepared`] (DFT-domain GGLWE) from scratch space.
    fn take_gglwe_prepared_scratch<A, M>(self, module: &M, infos: &A) -> (GGLWEPreparedViewMut<'a, B>, Self)
    where
        B: 'a,
        A: GGLWEInfos,
        M: ModuleN + VmpPMatBytesOf,
    {
        let n: usize = operand_degree(module.n(), &[infos.n()]);
        let (data, scratch) = self.take_vmp_pmat_scratch(
            n,
            infos.dnum().into(),
            infos.rank_in().into(),
            (infos.rank_out() + 1).into(),
            infos.size(),
            PrepareHint::Reuse,
        );
        (
            GGLWEPreparedViewMut::from_inner(GGLWEPrepared {
                noise: None,
                k_aux: infos.k_aux(),
                base2k: infos.base2k(),
                k_digit: infos.k_digit(),
                k_offset: infos.k_offset(),
                dnum: infos.dnum(),
                stride: 1,
                data: data.into_inner(),
            }),
            scratch,
        )
    }
}

/// Carves [`GGSW`] layouts from a [`ScratchArena`].
pub trait ScratchArenaTakeGGSW<'a, B: Backend>: ScratchArenaTakeMatZnx<'a, B> + Sized {
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
                noise: None,
                k_aux: infos.k_aux(),
                base2k: infos.base2k(),
                k_digit: infos.k_digit(),
                data: data.into_inner(),
            }),
            scratch,
        )
    }
}

/// Carves [`GGSWPrepared`] layouts from a [`ScratchArena`].
pub trait ScratchArenaTakeGGSWPrepared<'a, B: Backend>: ScratchArenaTakeVmpPMat<'a, B> + Sized {
    /// Allocates a [`GGSWPrepared`] (DFT-domain GGSW) from scratch space.
    fn take_ggsw_prepared_scratch<A, M>(self, module: &M, infos: &A) -> (GGSWPreparedViewMut<'a, B>, Self)
    where
        B: 'a,
        A: GGSWInfos,
        M: ModuleN + VmpPMatBytesOf,
    {
        let n: usize = operand_degree(module.n(), &[infos.n()]);
        let (data, scratch) = self.take_vmp_pmat_scratch(
            n,
            infos.dnum().into(),
            (infos.rank() + 1).into(),
            (infos.rank() + 1).into(),
            infos.size(),
            PrepareHint::Reuse,
        );
        (
            GGSWPreparedViewMut::from_inner(GGSWPrepared {
                noise: None,
                k_aux: infos.k_aux(),
                base2k: infos.base2k(),
                k_digit: infos.k_digit(),
                data: data.into_inner(),
            }),
            scratch,
        )
    }
}

impl<'a, B: Backend> ScratchArenaTakeGLWELinearTransformationBabySteps<'a, B> for ScratchArena<'a, B> {}
impl<'a, B: Backend> ScratchArenaTakeLWE<'a, B> for ScratchArena<'a, B> {}
impl<'a, B: Backend> ScratchArenaTakeLWEPlaintext<'a, B> for ScratchArena<'a, B> {}
impl<'a, B: Backend> ScratchArenaTakeGLWE<'a, B> for ScratchArena<'a, B> {}
impl<'a, B: Backend> ScratchArenaTakeGLWETensor<'a, B> for ScratchArena<'a, B> {}
impl<'a, B: Backend> ScratchArenaTakeGLWEPlaintext<'a, B> for ScratchArena<'a, B> {}
impl<'a, B: Backend> ScratchArenaTakeGLWESecretPrepared<'a, B> for ScratchArena<'a, B> {}
impl<'a, B: Backend> ScratchArenaTakeGLWESecret<'a, B> for ScratchArena<'a, B> {}
impl<'a, B: Backend> ScratchArenaTakeGLWESecretTensor<'a, B> for ScratchArena<'a, B> {}
impl<'a, B: Backend> ScratchArenaTakeGGLWE<'a, B> for ScratchArena<'a, B> {}
impl<'a, B: Backend> ScratchArenaTakeGGLWEPrepared<'a, B> for ScratchArena<'a, B> {}
impl<'a, B: Backend> ScratchArenaTakeGGSW<'a, B> for ScratchArena<'a, B> {}
impl<'a, B: Backend> ScratchArenaTakeGGSWPrepared<'a, B> for ScratchArena<'a, B> {}

#[cfg(test)]
mod tests {
    use std::marker::PhantomData;

    use poulpy_hal::layouts::{Backend, DataViewMut, Module, ScratchOwned};

    use super::*;
    use crate::{
        layouts::{Base2K, Degree, Dnum, Dsize, GGSWLayout, GLWELayout, Rank, TorusPrecision},
        test_backend::HostVecBackend,
    };

    type TestBackend = HostVecBackend<0>;

    #[test]
    fn scratch_arena_take_core_returns_disjoint_backend_regions() {
        let module: Module<TestBackend> = Module::new(64);
        let glwe_infos = GLWELayout {
            n: Degree(64),
            base2k: Base2K(8),
            k: TorusPrecision(24),
            rank: Rank(2),
        };
        let ggsw_infos = GGSWLayout {
            n: Degree(64),
            base2k: Base2K(8),
            dnum: Dnum(2),
            k_aux: TorusPrecision(24 - 2 * 8),
            rank: Rank(1),
            k_digit: crate::layouts::k_digit_of(Base2K(8), Dsize(1)),
        };

        let mut scratch: ScratchOwned<TestBackend> = ScratchOwned {
            data: TestBackend::alloc_bytes(1 << 15),
            _phantom: PhantomData,
        };
        let arena = scratch.arena();
        let available_before = arena.available();

        let (mut glwe, arena) = arena.take_glwe_scratch(&glwe_infos);
        let glwe_bytes = glwe.data_mut().data_mut();
        glwe_bytes[0] = 0x5a;
        let glwe_ptr = glwe_bytes.as_mut_ptr() as usize;
        let glwe_len = glwe_bytes.len();

        let (mut prepared, arena) = arena.take_ggsw_prepared_scratch(&module, &ggsw_infos);
        let prepared_bytes = prepared.data.data_mut();
        prepared_bytes[0] = 0xa5;
        let prepared_ptr = prepared_bytes.as_mut_ptr() as usize;
        let prepared_len = prepared_bytes.len();

        assert!(arena.available() < available_before);
        assert!(glwe_ptr + glwe_len <= prepared_ptr || prepared_ptr + prepared_len <= glwe_ptr);
    }
}
