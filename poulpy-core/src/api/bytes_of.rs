//! Byte sizes of core layouts, routed through the backend.
//!
//! Unlike the operation families, this API needs no backend extension point.
//! A layout's byte size is fully determined by `poulpy-hal`'s [`Backend`]
//! sizing methods, so bounding on those lets `poulpy-core` implement the trait
//! directly for [`Module`] instead of dispatching through `oep`/`delegates`.

use poulpy_hal::layouts::{Backend, Module};

use crate::layouts::{Base2K, Degree, GGLWEInfos, GGSWInfos, GLWEInfos, LWEInfos, Rank, TorusPrecision, key_size, pairs};

/// Byte sizes of core layouts, routed through the backend.
///
/// Scratch sizing must go through this trait rather than through the static
/// `Type::bytes_of` constructors. Those compute the size from the *host* layout
/// (`VecZnx::<Vec<u8>, W>::bytes_of`), which silently bypasses a backend's
/// [`Backend::bytes_of_vec_znx`] / [`Backend::bytes_of_scalar_znx`] overrides.
/// A device backend that pads or aligns differently from the host would
/// otherwise have its override ignored by every core scratch computation.
///
/// The static constructors remain correct for genuinely host-owned buffers,
/// where naming `Vec<u8>` is the truth rather than a placeholder.
///
/// Sizing is keyed on the storage *domain*, never on the coefficient word: one
/// word maps to several layouts (`DftWord` covers `VecZnxDft`, `SvpPPol`,
/// `VmpPMat` and both `CnvPVec` forms), and several domains may share one word
/// (FFT64Ref declares `ZnxWord == BigWord == i64`). The DFT-domain counterpart
/// of this trait is
/// [`GLWEPreparedFactory::glwe_prepared_bytes_of`](crate::layouts::GLWEPreparedFactory::glwe_prepared_bytes_of).
pub trait GLWEBytesOf<BE: Backend> {
    /// Byte size of a [`GLWE`](crate::layouts::GLWE) over this backend.
    fn glwe_bytes_of(&self, n: Degree, base2k: Base2K, k: TorusPrecision, rank: Rank) -> usize;

    /// Byte size of a [`GLWE`](crate::layouts::GLWE) described by `infos`.
    fn glwe_bytes_of_from_infos<A: GLWEInfos>(&self, infos: &A) -> usize {
        self.glwe_bytes_of(infos.n(), infos.base2k(), infos.k(), infos.rank())
    }

    /// Byte size of a [`GLWEPlaintext`](crate::layouts::GLWEPlaintext) described by `infos`.
    ///
    /// Sized to `infos.size()` so key infos (with auxiliary limbs) reserve the
    /// full width; for non-key infos `size() == ceil(k/base2k)`.
    fn glwe_plaintext_bytes_of_from_infos<A: GLWEInfos>(&self, infos: &A) -> usize;

    /// Byte size of a [`GLWESecret`](crate::layouts::GLWESecret) over this backend.
    fn glwe_secret_bytes_of(&self, n: Degree, rank: Rank) -> usize;

    /// Byte size of a [`GLWESecret`](crate::layouts::GLWESecret) described by `infos`.
    fn glwe_secret_bytes_of_from_infos<A: GLWEInfos>(&self, infos: &A) -> usize {
        self.glwe_secret_bytes_of(infos.n(), infos.rank())
    }

    /// Byte size of a [`GLWESecretTensor`](crate::layouts::GLWESecretTensor) described by `infos`.
    fn glwe_secret_tensor_bytes_of_from_infos<A: GLWEInfos>(&self, infos: &A) -> usize;

    /// Byte size of a [`GLWETensor`](crate::layouts::GLWETensor) described by `infos`.
    fn glwe_tensor_bytes_of_from_infos<A: GLWEInfos>(&self, infos: &A) -> usize;

    /// Byte size of an [`LWE`](crate::layouts::LWE) described by `infos`.
    fn lwe_bytes_of_from_infos<A: LWEInfos>(&self, infos: &A) -> usize;

    /// Byte size of a [`GGLWE`](crate::layouts::GGLWE) described by `infos`.
    fn gglwe_bytes_of_from_infos<A: GGLWEInfos>(&self, infos: &A) -> usize;

    /// Byte size of a [`GGSW`](crate::layouts::GGSW) described by `infos`.
    fn ggsw_bytes_of_from_infos<A: GGSWInfos>(&self, infos: &A) -> usize;
}

impl<B: Backend> GLWEBytesOf<B> for Module<B> {
    fn glwe_bytes_of(&self, n: Degree, base2k: Base2K, k: TorusPrecision, rank: Rank) -> usize {
        B::bytes_of_vec_znx(n.into(), (rank + 1).into(), k.0.div_ceil(base2k.0) as usize)
    }

    fn glwe_plaintext_bytes_of_from_infos<A: GLWEInfos>(&self, infos: &A) -> usize {
        B::bytes_of_vec_znx(infos.n().into(), 1, infos.size())
    }

    fn glwe_secret_bytes_of(&self, n: Degree, rank: Rank) -> usize {
        B::bytes_of_scalar_znx(n.into(), rank.into())
    }

    fn glwe_secret_tensor_bytes_of_from_infos<A: GLWEInfos>(&self, infos: &A) -> usize {
        B::bytes_of_scalar_znx(infos.n().into(), pairs(infos.rank().into()))
    }

    fn glwe_tensor_bytes_of_from_infos<A: GLWEInfos>(&self, infos: &A) -> usize {
        let cols: usize = infos.rank().as_usize() + 1;
        let pairs: usize = (((cols + 1) * cols) >> 1).max(1);
        B::bytes_of_vec_znx(infos.n().into(), pairs, infos.k().0.div_ceil(infos.base2k().0) as usize)
    }

    fn lwe_bytes_of_from_infos<A: LWEInfos>(&self, infos: &A) -> usize {
        let size: usize = infos.k().0.div_ceil(infos.base2k().0) as usize;
        B::bytes_of_vec_znx(1, 1, size) + B::bytes_of_vec_znx(infos.n().as_usize(), 1, size)
    }

    fn gglwe_bytes_of_from_infos<A: GGLWEInfos>(&self, infos: &A) -> usize {
        let size: usize = key_size(infos.base2k(), infos.dnum(), infos.dsize(), infos.k_aux());
        B::bytes_of_mat_znx(
            infos.n().into(),
            infos.dnum().into(),
            infos.rank_in().into(),
            (infos.rank_out() + 1).into(),
            size,
        )
    }

    fn ggsw_bytes_of_from_infos<A: GGSWInfos>(&self, infos: &A) -> usize {
        let size: usize = key_size(infos.base2k(), infos.dnum(), infos.dsize(), infos.k_aux());
        B::bytes_of_mat_znx(
            infos.n().into(),
            infos.dnum().into(),
            (infos.rank() + 1).into(),
            (infos.rank() + 1).into(),
            size,
        )
    }
}
