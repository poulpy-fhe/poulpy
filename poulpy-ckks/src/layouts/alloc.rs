use poulpy_core::layouts::{
    Base2K, Degree, GLWEInfos, GLWEPlaintextLayout, GetDegree, LWEInfos, ModuleCoreAlloc, Rank, TorusPrecision,
};
use poulpy_hal::layouts::{Backend, Module};

use crate::{CKKSInfos, CKKSMeta, SetCKKSInfos};

use super::{CKKSCiphertext, CKKSPlaintext};

pub trait CKKSModuleAlloc<BE: Backend>: ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf> {
    fn ckks_ciphertext_alloc_from_infos<A>(&self, infos: &A) -> CKKSCiphertext<BE::OwnedBuf>
    where
        A: GLWEInfos;

    fn ckks_ciphertext_alloc(&self, base2k: Base2K, k: TorusPrecision) -> CKKSCiphertext<BE::OwnedBuf>;

    fn ckks_pt_vec_alloc_from_infos<A>(&self, infos: &A) -> CKKSPlaintext<BE::OwnedBuf>
    where
        A: GLWEInfos + CKKSInfos;

    fn ckks_plaintext_alloc_from_infos<A>(&self, infos: &A) -> CKKSPlaintext<BE::OwnedBuf>
    where
        A: LWEInfos + CKKSInfos;

    /// Allocates a default-meta plaintext sized to `k` over `base2k`. The semantic
    /// [`CKKSMeta`] is not needed to size the buffer — set it afterwards with
    /// [`SetCKKSInfos::set_meta`] (the `_from_infos` variants do this for you).
    fn ckks_plaintext_alloc(&self, n: Degree, base2k: Base2K, k: TorusPrecision) -> CKKSPlaintext<BE::OwnedBuf>;

    fn ckks_pt_coeffs_alloc(&self, coeff_count: usize, base2k: Base2K, k: TorusPrecision) -> CKKSPlaintext<BE::OwnedBuf> {
        self.ckks_plaintext_alloc(coeff_count.into(), base2k, k)
    }

    fn ckks_pt_vec_alloc(&self, base2k: Base2K, k: TorusPrecision) -> CKKSPlaintext<BE::OwnedBuf>;
}

impl<BE: Backend> CKKSModuleAlloc<BE> for Module<BE>
where
    Module<BE>: ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf>,
{
    fn ckks_ciphertext_alloc_from_infos<A>(&self, infos: &A) -> CKKSCiphertext<BE::OwnedBuf>
    where
        A: GLWEInfos,
    {
        CKKSCiphertext::from_inner(self.glwe_alloc_from_infos(infos), CKKSMeta::default())
    }

    fn ckks_ciphertext_alloc(&self, base2k: Base2K, k: TorusPrecision) -> CKKSCiphertext<BE::OwnedBuf> {
        CKKSCiphertext::from_inner(self.glwe_alloc(base2k, k, Rank(1)), CKKSMeta::default())
    }

    fn ckks_pt_vec_alloc_from_infos<A>(&self, infos: &A) -> CKKSPlaintext<BE::OwnedBuf>
    where
        A: GLWEInfos + CKKSInfos,
    {
        self.ckks_plaintext_alloc_from_infos(infos)
    }

    fn ckks_plaintext_alloc_from_infos<A>(&self, infos: &A) -> CKKSPlaintext<BE::OwnedBuf>
    where
        A: LWEInfos + CKKSInfos,
    {
        let mut pt = self.ckks_plaintext_alloc(infos.n(), infos.base2k(), infos.k());
        pt.set_meta(infos.meta());
        pt
    }

    fn ckks_plaintext_alloc(&self, n: Degree, base2k: Base2K, k: TorusPrecision) -> CKKSPlaintext<BE::OwnedBuf> {
        // `k` is the effective torus width (`log_delta + log_budget`); the buffer
        // auto-sizes to `ceil(k / base2k)` limbs, so the integer-poly storage spans
        // `max_k` while `k` records the meaningful precision. The semantic meta is
        // independent of sizing and defaults here; callers set it via `set_meta`.
        CKKSPlaintext::from_inner(
            self.glwe_plaintext_alloc_from_infos(&GLWEPlaintextLayout { n, base2k, k }),
            CKKSMeta::default(),
        )
    }

    fn ckks_pt_vec_alloc(&self, base2k: Base2K, k: TorusPrecision) -> CKKSPlaintext<BE::OwnedBuf> {
        self.ckks_plaintext_alloc(self.ring_degree(), base2k, k)
    }
}
