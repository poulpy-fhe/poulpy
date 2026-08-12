use poulpy_core::layouts::{Base2K, Degree, GLWEInfos, GLWEPlaintextLayout, GetDegree, ModuleCoreAlloc, Rank, TorusPrecision};
use poulpy_hal::layouts::{Backend, Module};

use crate::{CKKSInfos, CKKSMeta, SetCKKSInfos};

use super::{CKKSCiphertext, CKKSCiphertextOwned, CKKSPlaintext, CKKSPlaintextOwned};

/// CKKS container allocation on a backend module.
///
/// Every method is default-bodied over the [`ModuleCoreAlloc`] supertrait, so
/// the blanket impl for `Module<BE>` is empty: the whole constructor matrix is
/// two primitive shapes (ciphertext with explicit rank, plaintext with explicit
/// degree) plus thin conveniences over them.
pub trait CKKSModuleAlloc<BE: Backend>: ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = BE::ZnxWord> {
    /// Allocates a ciphertext with `infos`' layout **and** its CKKS metadata
    /// (`log_delta`, `log_sparsity`), mirroring
    /// [`Self::ckks_plaintext_alloc_from_infos`]. Use
    /// [`Self::ckks_ciphertext_alloc_from_glwe_infos`] when only a GLWE layout
    /// is available (fresh default metadata).
    fn ckks_ciphertext_alloc_from_infos<A>(&self, infos: &A) -> CKKSCiphertextOwned<BE>
    where
        A: GLWEInfos + CKKSInfos,
    {
        CKKSCiphertext::from_inner(self.glwe_alloc_from_infos(infos), infos.meta())
    }

    /// Allocates a default-meta ciphertext from a bare GLWE layout; the name
    /// makes the metadata drop explicit.
    fn ckks_ciphertext_alloc_from_glwe_infos<A>(&self, infos: &A) -> CKKSCiphertextOwned<BE>
    where
        A: GLWEInfos,
    {
        CKKSCiphertext::from_inner(self.glwe_alloc_from_infos(infos), CKKSMeta::default())
    }

    /// Allocates a default-meta ciphertext of the given `rank`.
    fn ckks_ciphertext_alloc_with_rank(&self, base2k: Base2K, k: TorusPrecision, rank: Rank) -> CKKSCiphertextOwned<BE> {
        CKKSCiphertext::from_inner(self.glwe_alloc(base2k, k, rank), CKKSMeta::default())
    }

    /// Rank-1 convenience over [`Self::ckks_ciphertext_alloc_with_rank`].
    fn ckks_ciphertext_alloc(&self, base2k: Base2K, k: TorusPrecision) -> CKKSCiphertextOwned<BE> {
        self.ckks_ciphertext_alloc_with_rank(base2k, k, Rank(1))
    }

    fn ckks_plaintext_alloc_from_infos<A>(&self, infos: &A) -> CKKSPlaintextOwned<BE>
    where
        A: CKKSInfos,
    {
        let mut pt = self.ckks_plaintext_alloc(infos.n(), infos.base2k(), infos.k());
        pt.set_meta(infos.meta());
        pt
    }

    /// Allocates a default-meta plaintext sized to `k` over `base2k`. The semantic
    /// [`CKKSMeta`] is not needed to size the buffer — set it afterwards with
    /// [`SetCKKSInfos::set_meta`] (the `_from_infos` variants do this for you).
    fn ckks_plaintext_alloc(&self, n: Degree, base2k: Base2K, k: TorusPrecision) -> CKKSPlaintextOwned<BE> {
        // `k` is the effective torus width (`log_delta + log_budget`); the buffer
        // auto-sizes to `ceil(k / base2k)` limbs, so the integer-poly storage spans
        // `max_k` while `k` records the meaningful precision. The semantic meta is
        // independent of sizing and defaults here; callers set it via `set_meta`.
        CKKSPlaintext::from_inner(
            self.glwe_plaintext_alloc_from_infos(&GLWEPlaintextLayout { n, base2k, k }),
            CKKSMeta::default(),
        )
    }

    /// Coefficient-count convenience over [`Self::ckks_plaintext_alloc`].
    fn ckks_pt_coeffs_alloc(&self, coeff_count: usize, base2k: Base2K, k: TorusPrecision) -> CKKSPlaintextOwned<BE> {
        self.ckks_plaintext_alloc(coeff_count.into(), base2k, k)
    }

    /// Full-ring-degree convenience over [`Self::ckks_plaintext_alloc`].
    fn ckks_pt_vec_alloc(&self, base2k: Base2K, k: TorusPrecision) -> CKKSPlaintextOwned<BE>
    where
        Self: GetDegree,
    {
        self.ckks_plaintext_alloc(self.ring_degree(), base2k, k)
    }
}

impl<BE: Backend> CKKSModuleAlloc<BE> for Module<BE> where
    Module<BE>: ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = BE::ZnxWord>
{
}
