use poulpy_core::layouts::{
    Base2K, Degree, GLWEInfos, GLWEPlaintextLayout, GetDegree, LWEInfos, ModuleCoreAlloc, Rank, TorusPrecision,
};
use poulpy_hal::layouts::{Backend, Module};

use crate::{CKKSInfos, CKKSMeta};

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

    fn ckks_plaintext_alloc(&self, n: Degree, base2k: Base2K, meta: CKKSMeta) -> CKKSPlaintext<BE::OwnedBuf>;

    fn ckks_pt_coeffs_alloc(&self, coeff_count: usize, base2k: Base2K, meta: CKKSMeta) -> CKKSPlaintext<BE::OwnedBuf> {
        self.ckks_plaintext_alloc(coeff_count.into(), base2k, meta)
    }

    fn ckks_pt_vec_alloc(&self, base2k: Base2K, meta: CKKSMeta) -> CKKSPlaintext<BE::OwnedBuf>;
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
        self.ckks_plaintext_alloc(infos.n(), infos.base2k(), infos.meta())
    }

    fn ckks_plaintext_alloc(&self, n: Degree, base2k: Base2K, meta: CKKSMeta) -> CKKSPlaintext<BE::OwnedBuf> {
        CKKSPlaintext::from_inner(
            self.glwe_plaintext_alloc_from_infos(&GLWEPlaintextLayout {
                n,
                base2k,
                k: meta.min_k(base2k),
            }),
            meta,
        )
    }

    fn ckks_pt_vec_alloc(&self, base2k: Base2K, meta: CKKSMeta) -> CKKSPlaintext<BE::OwnedBuf> {
        self.ckks_plaintext_alloc(self.ring_degree(), base2k, meta)
    }
}
