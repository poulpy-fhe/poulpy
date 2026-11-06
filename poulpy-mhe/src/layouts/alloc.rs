use poulpy_core::layouts::{
    Base2K, Degree, Dnum, GGLWEAlloc, GGLWECompressedAlloc, GGLWEInfos, GGSWInfos, GLWEAlloc, GLWECompressedAlloc, GLWEInfos,
    GLWEPublicKeyCompressedAlloc, ModuleCoreAlloc, ModuleCoreCompressedAlloc, Rank, TorusPrecision,
};
use poulpy_hal::{
    api::ModuleN,
    layouts::{Backend, Module},
};

use crate::layouts::{
    MHEGGLWEPat, MHEGGLWEPatCompressed, MHEGGLWEPatCompressedOwned, MHEGGLWEPatOwned, MHEGGSWShare, MHEGGSWShareOwned,
    MHEGLWEAutomorphismKeyShare, MHEGLWEAutomorphismKeyShareOwned, MHEGLWEEncToShareShare, MHEGLWEEncToShareShareOwned,
    MHEGLWEPatCompressed, MHEGLWEPatCompressedOwned, MHEGLWEPrivateKeyswitchShare, MHEGLWEPrivateKeyswitchShareOwned,
    MHEGLWEPublicKeyShare, MHEGLWEPublicKeyShareOwned, MHEGLWEPublicKeyswitchShare, MHEGLWEPublicKeyswitchShareOwned,
    MHEGLWEShareToEncShare, MHEGLWEShareToEncShareOwned, MHEGLWESwitchingKeyShare, MHEGLWESwitchingKeyShareOwned,
    MHEGLWETensorKeyShare, MHEGLWETensorKeyShareOwned, ggsw_share_part_layout,
};

fn assert_degree(module_n: usize, infos_n: Degree) {
    assert!(
        infos_n.as_usize() == module_n,
        "invalid layout: degree differs from the module's"
    );
}

/// Allocation of [`MHEGLWEPatCompressed`] on a backend module.
///
/// Every method is default-bodied over the core allocation supertraits, so the
/// blanket impl for `Module<BE>` is empty. A fresh PAT or share is zero; share
/// metadata starts as in core (degrees `0`, Galois element `0`, distribution
/// `NONE`). Every `*_alloc_from_infos` panics unless the infos degree is the
/// module's.
pub trait MHEGLWEPatCompressedAlloc<BE: Backend>:
    ModuleN + ModuleCoreCompressedAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = BE::ZnxWord> + GLWECompressedAlloc
{
    fn mhe_glwe_pat_compressed_alloc_from_infos<A: GLWEInfos>(&self, infos: &A) -> MHEGLWEPatCompressedOwned<BE> {
        assert_degree(self.n(), infos.n());
        MHEGLWEPatCompressed {
            inner: self.glwe_compressed_alloc_from_infos(infos),
        }
    }

    fn mhe_glwe_pat_compressed_alloc(&self, base2k: Base2K, k: TorusPrecision, rank: Rank) -> MHEGLWEPatCompressedOwned<BE> {
        MHEGLWEPatCompressed {
            inner: self.glwe_compressed_alloc(base2k, k, rank),
        }
    }
}

impl<BE: Backend> MHEGLWEPatCompressedAlloc<BE> for Module<BE> {}

/// Allocation of [`MHEGGLWEPatCompressed`] on a backend module, with the contract of [`MHEGLWEPatCompressedAlloc`].
pub trait MHEGGLWEPatCompressedAlloc<BE: Backend>:
    ModuleN + ModuleCoreCompressedAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = BE::ZnxWord> + GGLWECompressedAlloc
{
    fn mhe_gglwe_pat_compressed_alloc_from_infos<A: GGLWEInfos>(&self, infos: &A) -> MHEGGLWEPatCompressedOwned<BE> {
        assert_degree(self.n(), infos.n());
        MHEGGLWEPatCompressed {
            inner: self.gglwe_compressed_alloc_from_infos(infos),
        }
    }

    fn mhe_gglwe_pat_compressed_alloc(
        &self,
        base2k: Base2K,
        dnum: Dnum,
        k_digit: TorusPrecision,
        k_aux: TorusPrecision,
        rank_in: Rank,
        rank_out: Rank,
    ) -> MHEGGLWEPatCompressedOwned<BE> {
        MHEGGLWEPatCompressed {
            inner: self.gglwe_compressed_alloc(base2k, dnum, k_digit, k_aux, rank_in, rank_out),
        }
    }
}

impl<BE: Backend> MHEGGLWEPatCompressedAlloc<BE> for Module<BE> {}

/// Allocation of [`MHEGGLWEPat`] on a backend module, with the contract of [`MHEGLWEPatCompressedAlloc`].
pub trait MHEGGLWEPatAlloc<BE: Backend>:
    ModuleN + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = BE::ZnxWord> + GGLWEAlloc
{
    fn mhe_gglwe_pat_alloc_from_infos<A: GGLWEInfos>(&self, infos: &A) -> MHEGGLWEPatOwned<BE> {
        assert_degree(self.n(), infos.n());
        MHEGGLWEPat {
            inner: self.gglwe_alloc_from_infos(infos),
        }
    }

    fn mhe_gglwe_pat_alloc(
        &self,
        base2k: Base2K,
        dnum: Dnum,
        k_digit: TorusPrecision,
        k_aux: TorusPrecision,
        rank_in: Rank,
        rank_out: Rank,
    ) -> MHEGGLWEPatOwned<BE> {
        MHEGGLWEPat {
            inner: self.gglwe_alloc(base2k, dnum, k_digit, k_aux, rank_in, rank_out),
        }
    }
}

impl<BE: Backend> MHEGGLWEPatAlloc<BE> for Module<BE> {}

/// Allocation of [`MHEGLWEPublicKeyShare`] on a backend module, with the contract of [`MHEGLWEPatCompressedAlloc`].
pub trait MHEGLWEPublicKeyShareAlloc<BE: Backend>:
    ModuleN + ModuleCoreCompressedAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = BE::ZnxWord> + GLWEPublicKeyCompressedAlloc
{
    fn mhe_glwe_public_key_share_alloc_from_infos<A: GLWEInfos>(&self, infos: &A) -> MHEGLWEPublicKeyShareOwned<BE> {
        assert_degree(self.n(), infos.n());
        MHEGLWEPublicKeyShare {
            key: self.glwe_public_key_compressed_alloc_from_infos(infos),
        }
    }

    fn mhe_glwe_public_key_share_alloc(&self, base2k: Base2K, k: TorusPrecision, rank: Rank) -> MHEGLWEPublicKeyShareOwned<BE> {
        MHEGLWEPublicKeyShare {
            key: self.glwe_public_key_compressed_alloc(base2k, k, rank),
        }
    }
}

impl<BE: Backend> MHEGLWEPublicKeyShareAlloc<BE> for Module<BE> {}

/// Allocation of [`MHEGLWESwitchingKeyShare`] on a backend module, with the contract of [`MHEGLWEPatCompressedAlloc`].
pub trait MHEGLWESwitchingKeyShareAlloc<BE: Backend>: ModuleN + MHEGGLWEPatCompressedAlloc<BE> {
    fn mhe_glwe_switching_key_share_alloc_from_infos<A: GGLWEInfos>(&self, infos: &A) -> MHEGLWESwitchingKeyShareOwned<BE> {
        MHEGLWESwitchingKeyShare {
            key: self.mhe_gglwe_pat_compressed_alloc_from_infos(infos),
            input_degree: Degree(0),
            output_degree: Degree(0),
        }
    }

    fn mhe_glwe_switching_key_share_alloc(
        &self,
        base2k: Base2K,
        dnum: Dnum,
        k_digit: TorusPrecision,
        k_aux: TorusPrecision,
        rank_in: Rank,
        rank_out: Rank,
    ) -> MHEGLWESwitchingKeyShareOwned<BE> {
        MHEGLWESwitchingKeyShare {
            key: self.mhe_gglwe_pat_compressed_alloc(base2k, dnum, k_digit, k_aux, rank_in, rank_out),
            input_degree: Degree(0),
            output_degree: Degree(0),
        }
    }
}

impl<BE: Backend> MHEGLWESwitchingKeyShareAlloc<BE> for Module<BE> {}

/// Allocation of [`MHEGLWEAutomorphismKeyShare`] on a backend module, with the contract of [`MHEGLWEPatCompressedAlloc`].
pub trait MHEGLWEAutomorphismKeyShareAlloc<BE: Backend>: ModuleN + MHEGGLWEPatCompressedAlloc<BE> {
    fn mhe_glwe_automorphism_key_share_alloc_from_infos<A: GGLWEInfos>(&self, infos: &A) -> MHEGLWEAutomorphismKeyShareOwned<BE> {
        MHEGLWEAutomorphismKeyShare {
            key: self.mhe_gglwe_pat_compressed_alloc_from_infos(infos),
            p: 0,
        }
    }

    fn mhe_glwe_automorphism_key_share_alloc(
        &self,
        base2k: Base2K,
        dnum: Dnum,
        k_digit: TorusPrecision,
        k_aux: TorusPrecision,
        rank: Rank,
    ) -> MHEGLWEAutomorphismKeyShareOwned<BE> {
        MHEGLWEAutomorphismKeyShare {
            key: self.mhe_gglwe_pat_compressed_alloc(base2k, dnum, k_digit, k_aux, rank, rank),
            p: 0,
        }
    }
}

impl<BE: Backend> MHEGLWEAutomorphismKeyShareAlloc<BE> for Module<BE> {}

/// Allocation of [`MHEGLWETensorKeyShare`] on a backend module, with the contract of [`MHEGLWEPatCompressedAlloc`].
pub trait MHEGLWETensorKeyShareAlloc<BE: Backend>: ModuleN + MHEGGLWEPatAlloc<BE> {
    fn mhe_glwe_tensor_key_share_alloc_from_infos<A: GGLWEInfos>(&self, infos: &A) -> MHEGLWETensorKeyShareOwned<BE> {
        MHEGLWETensorKeyShare {
            key: self.mhe_gglwe_pat_alloc_from_infos(infos),
        }
    }

    fn mhe_glwe_tensor_key_share_alloc(
        &self,
        base2k: Base2K,
        dnum: Dnum,
        k_digit: TorusPrecision,
        k_aux: TorusPrecision,
        rank: Rank,
    ) -> MHEGLWETensorKeyShareOwned<BE> {
        let pairs = Rank(rank.0 * (rank.0 + 1) / 2);
        MHEGLWETensorKeyShare {
            key: self.mhe_gglwe_pat_alloc(base2k, dnum, k_digit, k_aux, pairs, rank),
        }
    }
}

impl<BE: Backend> MHEGLWETensorKeyShareAlloc<BE> for Module<BE> {}

/// Allocation of [`MHEGGSWShare`] on a backend module, with the contract of [`MHEGLWEPatCompressedAlloc`].
pub trait MHEGGSWShareAlloc<BE: Backend>: ModuleN + MHEGGLWEPatCompressedAlloc<BE> {
    fn mhe_ggsw_share_alloc_from_infos<A: GGSWInfos>(&self, infos: &A) -> MHEGGSWShareOwned<BE> {
        let circ = || -> Vec<MHEGGLWEPatCompressedOwned<BE>> {
            (0..infos.rank().as_usize())
                .map(|_| {
                    self.mhe_gglwe_pat_compressed_alloc_from_infos(&ggsw_share_part_layout(infos, infos.rank(), infos.rank()))
                })
                .collect()
        };
        MHEGGSWShare {
            col0: self.mhe_gglwe_pat_compressed_alloc_from_infos(&ggsw_share_part_layout(infos, Rank(1), infos.rank())),
            circ_u: circ(),
            circ_s: circ(),
        }
    }

    fn mhe_ggsw_share_alloc(
        &self,
        base2k: Base2K,
        dnum: Dnum,
        k_digit: TorusPrecision,
        k_aux: TorusPrecision,
        rank: Rank,
    ) -> MHEGGSWShareOwned<BE> {
        let circ = || -> Vec<MHEGGLWEPatCompressedOwned<BE>> {
            (0..rank.as_usize())
                .map(|_| self.mhe_gglwe_pat_compressed_alloc(base2k, dnum, k_digit, k_aux, rank, rank))
                .collect()
        };
        MHEGGSWShare {
            col0: self.mhe_gglwe_pat_compressed_alloc(base2k, dnum, k_digit, k_aux, Rank(1), rank),
            circ_u: circ(),
            circ_s: circ(),
        }
    }
}

impl<BE: Backend> MHEGGSWShareAlloc<BE> for Module<BE> {}

/// Allocation of [`MHEGLWEPrivateKeyswitchShare`] on a backend module, with the contract of [`MHEGLWEPatCompressedAlloc`].
pub trait MHEGLWEPrivateKeyswitchShareAlloc<BE: Backend>:
    ModuleN + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = BE::ZnxWord> + GLWEAlloc
{
    /// A key switching share of `infos`, the ciphertext layout, at rank 0.
    fn mhe_glwe_private_keyswitch_share_alloc_from_infos<A: GLWEInfos>(
        &self,
        infos: &A,
    ) -> MHEGLWEPrivateKeyswitchShareOwned<BE> {
        assert_degree(self.n(), infos.n());
        self.mhe_glwe_private_keyswitch_share_alloc(infos.base2k(), infos.k())
    }

    fn mhe_glwe_private_keyswitch_share_alloc(&self, base2k: Base2K, k: TorusPrecision) -> MHEGLWEPrivateKeyswitchShareOwned<BE> {
        MHEGLWEPrivateKeyswitchShare {
            inner: self.glwe_alloc(base2k, k, Rank(0)),
        }
    }
}

impl<BE: Backend> MHEGLWEPrivateKeyswitchShareAlloc<BE> for Module<BE> {}

/// Allocation of [`MHEGLWEPublicKeyswitchShare`] on a backend module, with the contract of [`MHEGLWEPatCompressedAlloc`].
pub trait MHEGLWEPublicKeyswitchShareAlloc<BE: Backend>:
    ModuleN + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = BE::ZnxWord> + GLWEAlloc
{
    /// A public key switching share of `infos`, the share layout.
    fn mhe_glwe_public_keyswitch_share_alloc_from_infos<A: GLWEInfos>(&self, infos: &A) -> MHEGLWEPublicKeyswitchShareOwned<BE> {
        assert_degree(self.n(), infos.n());
        MHEGLWEPublicKeyswitchShare {
            inner: self.glwe_alloc_from_infos(infos),
        }
    }

    fn mhe_glwe_public_keyswitch_share_alloc(
        &self,
        base2k: Base2K,
        k: TorusPrecision,
        rank: Rank,
    ) -> MHEGLWEPublicKeyswitchShareOwned<BE> {
        MHEGLWEPublicKeyswitchShare {
            inner: self.glwe_alloc(base2k, k, rank),
        }
    }
}

impl<BE: Backend> MHEGLWEPublicKeyswitchShareAlloc<BE> for Module<BE> {}

/// Allocation of [`MHEGLWEEncToShareShare`] on a backend module, with the contract of [`MHEGLWEPatCompressedAlloc`].
pub trait MHEGLWEEncToShareShareAlloc<BE: Backend>:
    ModuleN + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = BE::ZnxWord> + GLWEAlloc
{
    /// An encryption-to-shares public share of `infos`, the ciphertext layout, at rank 0.
    fn mhe_glwe_enc_to_share_share_alloc_from_infos<A: GLWEInfos>(&self, infos: &A) -> MHEGLWEEncToShareShareOwned<BE> {
        assert_degree(self.n(), infos.n());
        self.mhe_glwe_enc_to_share_share_alloc(infos.base2k(), infos.k())
    }

    fn mhe_glwe_enc_to_share_share_alloc(&self, base2k: Base2K, k: TorusPrecision) -> MHEGLWEEncToShareShareOwned<BE> {
        MHEGLWEEncToShareShare {
            inner: self.glwe_alloc(base2k, k, Rank(0)),
        }
    }
}

impl<BE: Backend> MHEGLWEEncToShareShareAlloc<BE> for Module<BE> {}

/// Allocation of [`MHEGLWEShareToEncShare`] on a backend module, with the contract of [`MHEGLWEPatCompressedAlloc`].
pub trait MHEGLWEShareToEncShareAlloc<BE: Backend>: ModuleN + MHEGLWEPatCompressedAlloc<BE> {
    /// A shares-to-encryption share of `infos`, the output layout.
    fn mhe_glwe_share_to_enc_share_alloc_from_infos<A: GLWEInfos>(&self, infos: &A) -> MHEGLWEShareToEncShareOwned<BE> {
        MHEGLWEShareToEncShare {
            inner: self.mhe_glwe_pat_compressed_alloc_from_infos(infos),
        }
    }

    fn mhe_glwe_share_to_enc_share_alloc(
        &self,
        base2k: Base2K,
        k: TorusPrecision,
        rank: Rank,
    ) -> MHEGLWEShareToEncShareOwned<BE> {
        MHEGLWEShareToEncShare {
            inner: self.mhe_glwe_pat_compressed_alloc(base2k, k, rank),
        }
    }
}

impl<BE: Backend> MHEGLWEShareToEncShareAlloc<BE> for Module<BE> {}
