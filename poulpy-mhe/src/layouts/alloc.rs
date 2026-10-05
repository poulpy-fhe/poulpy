use poulpy_core::layouts::{
    Base2K, Degree, Dnum, Dsize, GGLWEInfos, GGSWInfos, GLWEInfos, ModuleCoreAlloc, ModuleCoreCompressedAlloc, Rank,
    TorusPrecision,
};
use poulpy_hal::layouts::{Backend, Module};

use crate::layouts::{
    CKKSRefreshShare, CKKSRefreshShareOwned, GGLWEPat, GGLWEPatCompressed, GGLWEPatCompressedOwned, GGLWEPatOwned, GGSWShare,
    GGSWShareOwned, GLWEAutomorphismKeyShare, GLWEAutomorphismKeyShareOwned, GLWEEncToShareShare, GLWEEncToShareShareOwned,
    GLWEPatCompressed, GLWEPatCompressedOwned, GLWEPrivateKeyswitchShare, GLWEPrivateKeyswitchShareOwned, GLWEPublicKeyShare,
    GLWEPublicKeyShareOwned, GLWEPublicKeyswitchShare, GLWEPublicKeyswitchShareOwned, GLWEShareToEncShare,
    GLWEShareToEncShareOwned, GLWESwitchingKeyShare, GLWESwitchingKeyShareOwned, GLWETensorKeyShare, GLWETensorKeyShareOwned,
    ggsw_share_part_layout,
};

/// PAT and share allocation on a backend module.
///
/// Every method is default-bodied over the core allocation supertraits, so the
/// blanket impl for `Module<BE>` is empty. A fresh PAT or share is zero; share
/// metadata starts as in core (degrees `0`, Galois element `0`, distribution
/// `NONE`).
pub trait MHEModuleAlloc<BE: Backend>:
    ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = BE::ZnxWord>
    + ModuleCoreCompressedAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = BE::ZnxWord>
{
    fn glwe_pat_compressed_alloc_from_infos<A: GLWEInfos>(&self, infos: &A) -> GLWEPatCompressedOwned<BE> {
        GLWEPatCompressed {
            inner: self.glwe_compressed_alloc_from_infos(infos),
        }
    }

    fn glwe_pat_compressed_alloc(&self, base2k: Base2K, k: TorusPrecision, rank: Rank) -> GLWEPatCompressedOwned<BE> {
        GLWEPatCompressed {
            inner: self.glwe_compressed_alloc(base2k, k, rank),
        }
    }

    fn gglwe_pat_compressed_alloc_from_infos<A: GGLWEInfos>(&self, infos: &A) -> GGLWEPatCompressedOwned<BE> {
        GGLWEPatCompressed {
            inner: self.gglwe_compressed_alloc_from_infos(infos),
        }
    }

    fn gglwe_pat_compressed_alloc(
        &self,
        base2k: Base2K,
        dnum: Dnum,
        dsize: Dsize,
        k_aux: TorusPrecision,
        rank_in: Rank,
        rank_out: Rank,
    ) -> GGLWEPatCompressedOwned<BE> {
        GGLWEPatCompressed {
            inner: self.gglwe_compressed_alloc(base2k, dnum, dsize, k_aux, rank_in, rank_out),
        }
    }

    fn gglwe_pat_alloc_from_infos<A: GGLWEInfos>(&self, infos: &A) -> GGLWEPatOwned<BE> {
        GGLWEPat {
            inner: self.gglwe_alloc_from_infos(infos),
        }
    }

    fn gglwe_pat_alloc(
        &self,
        base2k: Base2K,
        dnum: Dnum,
        dsize: Dsize,
        k_aux: TorusPrecision,
        rank_in: Rank,
        rank_out: Rank,
    ) -> GGLWEPatOwned<BE> {
        GGLWEPat {
            inner: self.gglwe_alloc(base2k, dnum, dsize, k_aux, rank_in, rank_out),
        }
    }

    fn glwe_public_key_share_alloc_from_infos<A: GLWEInfos>(&self, infos: &A) -> GLWEPublicKeyShareOwned<BE> {
        GLWEPublicKeyShare {
            key: self.glwe_public_key_compressed_alloc_from_infos(infos),
        }
    }

    fn glwe_public_key_share_alloc(&self, base2k: Base2K, k: TorusPrecision, rank: Rank) -> GLWEPublicKeyShareOwned<BE> {
        GLWEPublicKeyShare {
            key: self.glwe_public_key_compressed_alloc(base2k, k, rank),
        }
    }

    fn glwe_switching_key_share_alloc_from_infos<A: GGLWEInfos>(&self, infos: &A) -> GLWESwitchingKeyShareOwned<BE> {
        GLWESwitchingKeyShare {
            key: self.gglwe_pat_compressed_alloc_from_infos(infos),
            input_degree: Degree(0),
            output_degree: Degree(0),
        }
    }

    fn glwe_switching_key_share_alloc(
        &self,
        base2k: Base2K,
        dnum: Dnum,
        dsize: Dsize,
        k_aux: TorusPrecision,
        rank_in: Rank,
        rank_out: Rank,
    ) -> GLWESwitchingKeyShareOwned<BE> {
        GLWESwitchingKeyShare {
            key: self.gglwe_pat_compressed_alloc(base2k, dnum, dsize, k_aux, rank_in, rank_out),
            input_degree: Degree(0),
            output_degree: Degree(0),
        }
    }

    fn glwe_automorphism_key_share_alloc_from_infos<A: GGLWEInfos>(&self, infos: &A) -> GLWEAutomorphismKeyShareOwned<BE> {
        GLWEAutomorphismKeyShare {
            key: self.gglwe_pat_compressed_alloc_from_infos(infos),
            p: 0,
        }
    }

    fn glwe_automorphism_key_share_alloc(
        &self,
        base2k: Base2K,
        dnum: Dnum,
        dsize: Dsize,
        k_aux: TorusPrecision,
        rank: Rank,
    ) -> GLWEAutomorphismKeyShareOwned<BE> {
        GLWEAutomorphismKeyShare {
            key: self.gglwe_pat_compressed_alloc(base2k, dnum, dsize, k_aux, rank, rank),
            p: 0,
        }
    }

    fn glwe_tensor_key_share_alloc_from_infos<A: GGLWEInfos>(&self, infos: &A) -> GLWETensorKeyShareOwned<BE> {
        GLWETensorKeyShare {
            key: self.gglwe_pat_alloc_from_infos(infos),
        }
    }

    fn glwe_tensor_key_share_alloc(
        &self,
        base2k: Base2K,
        dnum: Dnum,
        dsize: Dsize,
        k_aux: TorusPrecision,
        rank: Rank,
    ) -> GLWETensorKeyShareOwned<BE> {
        let pairs = Rank(rank.0 * (rank.0 + 1) / 2);
        GLWETensorKeyShare {
            key: self.gglwe_pat_alloc(base2k, dnum, dsize, k_aux, pairs, rank),
        }
    }

    fn ggsw_share_alloc_from_infos<A: GGSWInfos>(&self, infos: &A) -> GGSWShareOwned<BE> {
        let circ = || -> Vec<GGLWEPatCompressedOwned<BE>> {
            (0..infos.rank().as_usize())
                .map(|_| self.gglwe_pat_compressed_alloc_from_infos(&ggsw_share_part_layout(infos, infos.rank(), infos.rank())))
                .collect()
        };
        GGSWShare {
            col0: self.gglwe_pat_compressed_alloc_from_infos(&ggsw_share_part_layout(infos, Rank(1), infos.rank())),
            circ_u: circ(),
            circ_s: circ(),
        }
    }

    fn ggsw_share_alloc(
        &self,
        base2k: Base2K,
        dnum: Dnum,
        dsize: Dsize,
        k_aux: TorusPrecision,
        rank: Rank,
    ) -> GGSWShareOwned<BE> {
        let circ = || -> Vec<GGLWEPatCompressedOwned<BE>> {
            (0..rank.as_usize())
                .map(|_| self.gglwe_pat_compressed_alloc(base2k, dnum, dsize, k_aux, rank, rank))
                .collect()
        };
        GGSWShare {
            col0: self.gglwe_pat_compressed_alloc(base2k, dnum, dsize, k_aux, Rank(1), rank),
            circ_u: circ(),
            circ_s: circ(),
        }
    }

    /// A key switching share of `infos`, the ciphertext layout, at rank 0.
    fn glwe_private_keyswitch_share_alloc_from_infos<A: GLWEInfos>(&self, infos: &A) -> GLWEPrivateKeyswitchShareOwned<BE> {
        self.glwe_private_keyswitch_share_alloc(infos.base2k(), infos.k())
    }

    fn glwe_private_keyswitch_share_alloc(&self, base2k: Base2K, k: TorusPrecision) -> GLWEPrivateKeyswitchShareOwned<BE> {
        GLWEPrivateKeyswitchShare {
            inner: self.glwe_alloc(base2k, k, Rank(0)),
        }
    }

    /// A public key switching share of `infos`, the share layout.
    fn glwe_public_keyswitch_share_alloc_from_infos<A: GLWEInfos>(&self, infos: &A) -> GLWEPublicKeyswitchShareOwned<BE> {
        GLWEPublicKeyswitchShare {
            inner: self.glwe_alloc_from_infos(infos),
        }
    }

    fn glwe_public_keyswitch_share_alloc(
        &self,
        base2k: Base2K,
        k: TorusPrecision,
        rank: Rank,
    ) -> GLWEPublicKeyswitchShareOwned<BE> {
        GLWEPublicKeyswitchShare {
            inner: self.glwe_alloc(base2k, k, rank),
        }
    }

    /// An encryption-to-shares public share of `infos`, the ciphertext layout, at rank 0.
    fn glwe_enc_to_share_share_alloc_from_infos<A: GLWEInfos>(&self, infos: &A) -> GLWEEncToShareShareOwned<BE> {
        self.glwe_enc_to_share_share_alloc(infos.base2k(), infos.k())
    }

    fn glwe_enc_to_share_share_alloc(&self, base2k: Base2K, k: TorusPrecision) -> GLWEEncToShareShareOwned<BE> {
        GLWEEncToShareShare {
            inner: self.glwe_alloc(base2k, k, Rank(0)),
        }
    }

    /// A shares-to-encryption share of `infos`, the output layout.
    fn glwe_share_to_enc_share_alloc_from_infos<A: GLWEInfos>(&self, infos: &A) -> GLWEShareToEncShareOwned<BE> {
        GLWEShareToEncShare {
            inner: self.glwe_pat_compressed_alloc_from_infos(infos),
        }
    }

    fn glwe_share_to_enc_share_alloc(&self, base2k: Base2K, k: TorusPrecision, rank: Rank) -> GLWEShareToEncShareOwned<BE> {
        GLWEShareToEncShare {
            inner: self.glwe_pat_compressed_alloc(base2k, k, rank),
        }
    }

    /// A CKKS refresh share of a ciphertext at `ct_infos` into an output at `res_infos`.
    fn ckks_refresh_share_alloc_from_infos<A: GLWEInfos, B: GLWEInfos>(
        &self,
        ct_infos: &A,
        res_infos: &B,
    ) -> CKKSRefreshShareOwned<BE> {
        CKKSRefreshShare {
            e2s: self.glwe_enc_to_share_share_alloc_from_infos(ct_infos),
            s2e: self.glwe_share_to_enc_share_alloc_from_infos(res_infos),
        }
    }
}

impl<BE: Backend> MHEModuleAlloc<BE> for Module<BE> where
    Self: ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = BE::ZnxWord>
        + ModuleCoreCompressedAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = BE::ZnxWord>
{
}
