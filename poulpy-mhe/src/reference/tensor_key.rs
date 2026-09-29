use poulpy_core::{
    Distribution, EncryptionInfos, GLWEEncryptPk, GLWENormalize, GetDistribution,
    layouts::{GGLWEInfos, GGLWEToBackendMut, GLWEInfos, GLWEPublicKeyPreparedToBackendRef, GLWESecretToBackendRef, LWEInfos},
};
use poulpy_hal::{
    api::VecZnxAddScalarAssign,
    layouts::{Backend, Module, ScratchArena},
    source::Source,
};

use crate::layouts::GLWETensorKeyShareOwned;

pub trait GLWETensorKeyMHEProtocolReference<BE: Backend> {
    fn mhe_glwe_tensor_key_share_gen_tmp_bytes_reference<A, B>(&self, res_infos: &A, pk_infos: &B) -> usize
    where
        A: GGLWEInfos,
        B: GLWEInfos;

    #[allow(clippy::too_many_arguments)]
    fn mhe_glwe_tensor_key_share_gen_reference<S, K, E>(
        &self,
        res: &mut GLWETensorKeyShareOwned<BE>,
        sk: &S,
        pk: &K,
        enc_infos: &E,
        source_xu: &mut Source,
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        S: GLWESecretToBackendRef<BE> + GLWEInfos,
        K: GLWEPublicKeyPreparedToBackendRef<BE> + GLWEInfos,
        E: EncryptionInfos;
}

impl<BE: Backend> GLWETensorKeyMHEProtocolReference<BE> for Module<BE>
where
    Self: GLWEEncryptPk<BE> + GLWENormalize<BE> + VecZnxAddScalarAssign<BE>,
{
    fn mhe_glwe_tensor_key_share_gen_tmp_bytes_reference<A, B>(&self, res_infos: &A, pk_infos: &B) -> usize
    where
        A: GGLWEInfos,
        B: GLWEInfos,
    {
        assert!(
            res_infos.n().as_usize() == self.n(),
            "invalid layout: degree differs from the module's"
        );
        self.glwe_encrypt_pk_tmp_bytes(res_infos, pk_infos)
            .max(self.glwe_normalize_tmp_bytes())
    }

    fn mhe_glwe_tensor_key_share_gen_reference<S, K, E>(
        &self,
        res: &mut GLWETensorKeyShareOwned<BE>,
        sk: &S,
        pk: &K,
        enc_infos: &E,
        source_xu: &mut Source,
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        S: GLWESecretToBackendRef<BE> + GLWEInfos,
        K: GLWEPublicKeyPreparedToBackendRef<BE> + GLWEInfos,
        E: EncryptionInfos,
    {
        assert!(
            res.n().as_usize() == self.n(),
            "invalid share: degree differs from the module's"
        );
        let rank: usize = res.rank_out().into();
        assert!(
            res.rank_in().as_usize() == rank * (rank + 1) / 2,
            "invalid share: tensor key layout does not match its rank"
        );
        assert!(
            sk.rank().as_usize() == rank,
            "invalid share: secret rank differs from the key's"
        );
        assert!(sk.n() == res.n(), "invalid share: secret degree differs from the key's");
        assert!(
            pk.rank().as_usize() == rank,
            "invalid share: public key rank differs from the key's"
        );
        assert!(pk.n() == res.n(), "invalid share: public key degree differs from the key's");
        assert!(
            pk.base2k() == res.base2k(),
            "invalid share: public key radix differs from the key's"
        );
        assert!(pk.k() >= res.k(), "invalid share: public key less precise than the share");
        assert!(
            !matches!(
                GLWEPublicKeyPreparedToBackendRef::to_backend_ref(pk).dist(),
                Distribution::NONE | Distribution::ENCAPSULATED(_)
            ),
            "invalid share: public key needs a samplable distribution"
        );
        let (dnum, dsize): (usize, usize) = (res.dnum().into(), res.dsize().into());
        let sk = sk.to_backend_ref();
        let mut res_be = GGLWEToBackendMut::<BE>::to_backend_mut(&mut res.key);
        for row in 0..dnum {
            for a in 0..rank {
                for b in a..rank {
                    let mut entry = res_be.at_view_mut(row, a * rank + b - a * (a + 1) / 2);
                    self.glwe_encrypt_zero_pk(&mut entry, pk, enc_infos, source_xu, source_xe, scratch);
                    // Mask column 1 + a meets S_a at decryption: s_b there sums to S_a * S_b.
                    self.vec_znx_add_scalar_assign(entry.data_mut(), 1 + a, (dsize - 1) + row * dsize, sk.data(), b);
                    self.glwe_normalize_assign(&mut entry, scratch);
                }
            }
        }
    }
}
