//! Byte sizes of core layouts, routed through the backend.
//!
//! Unlike the operation families, this API needs no backend extension point.
//! A layout's byte size is fully determined by `poulpy-hal`'s [`Backend`]
//! sizing methods, so bounding on those lets `poulpy-core` implement the trait
//! directly for [`Module`] instead of dispatching through `oep`/`delegates`.

use poulpy_hal::layouts::{Backend, Module, PrepareHint};

use crate::layouts::{
    Base2K, Degree, GGLWEInfos, GGSWInfos, GLWEInfos, LWEInfos, LinearTransformationPlan, Rank, TorusPrecision, key_size, pairs,
};

/// Byte sizes of core layouts, routed through the backend.
///
/// Scratch sizing must go through this trait rather than through the static
/// `Type::bytes_of` constructors. Those compute the size from the *host* layout
/// (`VecZnx::<AlignedBuf, W>::bytes_of`), which silently bypasses a backend's
/// [`Backend::vec_znx_bytes_of`] / [`Backend::scalar_znx_bytes_of`] overrides.
/// A device backend that pads or aligns differently from the host would
/// otherwise have its override ignored by every core scratch computation.
///
/// The static constructors remain correct for genuinely host-owned buffers,
/// where naming `AlignedBuf` is the truth rather than a placeholder.
///
/// Sizing is keyed on the storage *domain*, never on the coefficient word: one
/// word maps to several layouts (`DftWord` covers `VecZnxDft`, `SvpPPol`,
/// `VmpPMat` and both `CnvPVec` forms), and several domains may share one word
/// (a backend may use `i64` for both `ZnxWord` and `BigWord`). The DFT-domain
/// counterpart of this trait is
/// [`GLWEPreparedFactory::glwe_prepared_bytes_of`](crate::api::GLWEPreparedFactory::glwe_prepared_bytes_of).
pub trait GLWEBytesOf<BE: Backend> {
    /// Byte size of a [`GLWE`](crate::layouts::GLWE) over this backend.
    fn glwe_bytes_of(&self, n: Degree, base2k: Base2K, k: TorusPrecision, rank: Rank) -> usize;

    /// Byte size of a [`GLWE`](crate::layouts::GLWE) described by `infos`.
    fn glwe_bytes_of_from_infos<A: GLWEInfos>(&self, infos: &A) -> usize {
        self.glwe_bytes_of(infos.n(), infos.base2k(), infos.k(), infos.rank())
    }
}

impl<B: Backend> GLWEBytesOf<B> for Module<B> {
    fn glwe_bytes_of(&self, n: Degree, base2k: Base2K, k: TorusPrecision, rank: Rank) -> usize {
        B::vec_znx_bytes_of(n.into(), (rank + 1).into(), k.0.div_ceil(base2k.0) as usize)
    }
}

/// Byte sizes of [`GLWEPlaintext`](crate::layouts::GLWEPlaintext) layouts over this backend.
pub trait GLWEPlaintextBytesOf<BE: Backend> {
    /// Byte size of a [`GLWEPlaintext`](crate::layouts::GLWEPlaintext) described by `infos`.
    ///
    /// Sized to `infos.size()` so key infos (with auxiliary limbs) reserve the
    /// full width; for non-key infos `size() == ceil(k/base2k)`.
    fn glwe_plaintext_bytes_of_from_infos<A: GLWEInfos>(&self, infos: &A) -> usize;
}

impl<B: Backend> GLWEPlaintextBytesOf<B> for Module<B> {
    fn glwe_plaintext_bytes_of_from_infos<A: GLWEInfos>(&self, infos: &A) -> usize {
        B::vec_znx_bytes_of(infos.n().into(), 1, infos.size())
    }
}

/// Byte sizes of [`GLWESecret`](crate::layouts::GLWESecret) layouts over this backend.
pub trait GLWESecretBytesOf<BE: Backend> {
    /// Byte size of a [`GLWESecret`](crate::layouts::GLWESecret) over this backend.
    fn glwe_secret_bytes_of(&self, n: Degree, rank: Rank) -> usize;

    /// Byte size of a [`GLWESecret`](crate::layouts::GLWESecret) described by `infos`.
    fn glwe_secret_bytes_of_from_infos<A: GLWEInfos>(&self, infos: &A) -> usize {
        self.glwe_secret_bytes_of(infos.n(), infos.rank())
    }
}

impl<B: Backend> GLWESecretBytesOf<B> for Module<B> {
    fn glwe_secret_bytes_of(&self, n: Degree, rank: Rank) -> usize {
        B::scalar_znx_bytes_of(n.into(), rank.into())
    }
}

/// Byte sizes of [`GLWESecretTensor`](crate::layouts::GLWESecretTensor) layouts over this backend.
pub trait GLWESecretTensorBytesOf<BE: Backend> {
    /// Byte size of a [`GLWESecretTensor`](crate::layouts::GLWESecretTensor) described by `infos`.
    fn glwe_secret_tensor_bytes_of_from_infos<A: GLWEInfos>(&self, infos: &A) -> usize;
}

impl<B: Backend> GLWESecretTensorBytesOf<B> for Module<B> {
    fn glwe_secret_tensor_bytes_of_from_infos<A: GLWEInfos>(&self, infos: &A) -> usize {
        B::scalar_znx_bytes_of(infos.n().into(), pairs(infos.rank().into()))
    }
}

/// Byte sizes of [`GLWETensor`](crate::layouts::GLWETensor) layouts over this backend.
pub trait GLWETensorBytesOf<BE: Backend> {
    /// Byte size of a [`GLWETensor`](crate::layouts::GLWETensor) described by `infos`.
    fn glwe_tensor_bytes_of_from_infos<A: GLWEInfos>(&self, infos: &A) -> usize;
}

impl<B: Backend> GLWETensorBytesOf<B> for Module<B> {
    fn glwe_tensor_bytes_of_from_infos<A: GLWEInfos>(&self, infos: &A) -> usize {
        let cols: usize = infos.rank().as_usize() + 1;
        let pairs: usize = (((cols + 1) * cols) >> 1).max(1);
        B::vec_znx_bytes_of(infos.n().into(), pairs, infos.k().0.div_ceil(infos.base2k().0) as usize)
    }
}

/// Byte sizes of [`LWE`](crate::layouts::LWE) layouts over this backend.
pub trait LWEBytesOf<BE: Backend> {
    /// Byte size of an [`LWE`](crate::layouts::LWE) described by `infos`.
    fn lwe_bytes_of_from_infos<A: LWEInfos>(&self, infos: &A) -> usize;
}

impl<B: Backend> LWEBytesOf<B> for Module<B> {
    fn lwe_bytes_of_from_infos<A: LWEInfos>(&self, infos: &A) -> usize {
        let size: usize = infos.k().0.div_ceil(infos.base2k().0) as usize;
        // An LWE is two separate buffers (body then mask). `ScratchArena::take_region`
        // aligns every carve to `SCRATCH_ALIGN`, so the body's contribution has to be
        // rounded up or the reservation under-counts the gap before the mask.
        let body: usize = B::vec_znx_bytes_of(1, 1, size).next_multiple_of(B::SCRATCH_ALIGN);
        body + B::vec_znx_bytes_of(infos.n().as_usize(), 1, size)
    }
}

/// Byte sizes of [`GGLWE`](crate::layouts::GGLWE) layouts over this backend.
pub trait GGLWEBytesOf<BE: Backend> {
    /// Byte size of a [`GGLWE`](crate::layouts::GGLWE) described by `infos`.
    fn gglwe_bytes_of_from_infos<A: GGLWEInfos>(&self, infos: &A) -> usize;
}

impl<B: Backend> GGLWEBytesOf<B> for Module<B> {
    fn gglwe_bytes_of_from_infos<A: GGLWEInfos>(&self, infos: &A) -> usize {
        // The limbs of an owned GGLWE of these infos, its row offset included.
        let size: usize =
            crate::layouts::offset_key_size(infos.base2k(), infos.k_offset(), infos.dnum(), infos.k_digit(), infos.k_aux());
        B::mat_znx_bytes_of(
            infos.n().into(),
            infos.dnum().into(),
            infos.rank_in().into(),
            (infos.rank_out() + 1).into(),
            size,
        )
    }
}

/// Byte sizes of [`GGSW`](crate::layouts::GGSW) layouts over this backend.
pub trait GGSWBytesOf<BE: Backend> {
    /// Byte size of a [`GGSW`](crate::layouts::GGSW) described by `infos`.
    fn ggsw_bytes_of_from_infos<A: GGSWInfos>(&self, infos: &A) -> usize;
}

impl<B: Backend> GGSWBytesOf<B> for Module<B> {
    fn ggsw_bytes_of_from_infos<A: GGSWInfos>(&self, infos: &A) -> usize {
        let size: usize = key_size(infos.base2k(), infos.dnum(), infos.k_digit(), infos.k_aux());
        B::mat_znx_bytes_of(
            infos.n().into(),
            infos.dnum().into(),
            (infos.rank() + 1).into(),
            (infos.rank() + 1).into(),
            size,
        )
    }
}

/// Byte size of a linear transformation's baby-step cache over this backend.
pub trait GLWELinearTransformationBabyStepsBytesOf<BE: Backend> {
    /// Byte size of the baby-step cache of the transformation schedule `plan` for a source of the layout `a`, as
    /// [`ScratchArenaTakeGLWELinearTransformationBabySteps::take_glwe_linear_transformation_baby_steps_scratch`](crate::ScratchArenaTakeGLWELinearTransformationBabySteps::take_glwe_linear_transformation_baby_steps_scratch)
    /// carves it: one aligned operand per baby rotation.
    ///
    /// # Panics
    ///
    /// When the size overflows `usize`.
    fn glwe_linear_transformation_baby_steps_bytes_of<A: GLWEInfos>(&self, plan: &LinearTransformationPlan, a: &A) -> usize;
}

impl<B: Backend> GLWELinearTransformationBabyStepsBytesOf<B> for Module<B> {
    fn glwe_linear_transformation_baby_steps_bytes_of<A: GLWEInfos>(&self, plan: &LinearTransformationPlan, a: &A) -> usize {
        let operand = B::cnv_pvec_left_bytes_of(a.n().into(), (a.rank() + 1).into(), a.size(), PrepareHint::Reuse);
        B::scratch_aligned(operand)
            .checked_mul(plan.baby_steps.len())
            .expect("baby-step cache byte size overflows usize")
    }
}
