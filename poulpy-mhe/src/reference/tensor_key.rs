use poulpy_core::{
    EncryptionInfos, GLWEEncryptPk, GLWENormalize, GetDistribution,
    layouts::{GGLWEInfos, GGLWEToBackendMut, GLWEInfos, GLWEPreparedToBackendRef, GLWESecretToBackendRef, LWEInfos},
};
use poulpy_hal::{
    api::VecZnxAddScalarAssign,
    layouts::{Backend, Module, ScratchArena},
    source::Source,
};

use crate::layouts::GGLWEPatOwned;

pub trait GLWETensorKeyShareReference<BE: Backend> {
    fn glwe_tensor_key_share_tmp_bytes_reference<A, B>(&self, res_infos: &A, pk_infos: &B) -> usize
    where
        A: GGLWEInfos,
        B: GLWEInfos;

    #[allow(clippy::too_many_arguments)]
    fn glwe_tensor_key_share_reference<S, K, E>(
        &self,
        res: &mut GGLWEPatOwned<BE>,
        sk: &S,
        pk: &K,
        enc_infos: &E,
        source_xu: &mut Source,
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        S: GLWESecretToBackendRef<BE> + GLWEInfos,
        K: GLWEPreparedToBackendRef<BE> + GetDistribution + GLWEInfos,
        E: EncryptionInfos;
}

impl<BE: Backend> GLWETensorKeyShareReference<BE> for Module<BE>
where
    Self: GLWEEncryptPk<BE> + GLWENormalize<BE> + VecZnxAddScalarAssign<BE>,
{
    fn glwe_tensor_key_share_tmp_bytes_reference<A, B>(&self, res_infos: &A, pk_infos: &B) -> usize
    where
        A: GGLWEInfos,
        B: GLWEInfos,
    {
        self.glwe_encrypt_pk_tmp_bytes(res_infos)
            .max(self.glwe_encrypt_pk_tmp_bytes(pk_infos))
            .max(self.glwe_normalize_tmp_bytes())
    }

    fn glwe_tensor_key_share_reference<S, K, E>(
        &self,
        res: &mut GGLWEPatOwned<BE>,
        sk: &S,
        pk: &K,
        enc_infos: &E,
        source_xu: &mut Source,
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        S: GLWESecretToBackendRef<BE> + GLWEInfos,
        K: GLWEPreparedToBackendRef<BE> + GetDistribution + GLWEInfos,
        E: EncryptionInfos,
    {
        let rank: usize = res.rank_out().into();
        assert!(
            res.rank_in().as_usize() == rank * (rank + 1) / 2,
            "invalid share: tensor key layout does not match its rank"
        );
        assert!(
            sk.rank().as_usize() == rank,
            "invalid share: secret rank differs from the key's"
        );
        assert!(
            pk.rank().as_usize() == rank,
            "invalid share: public key rank differs from the key's"
        );
        assert!(
            pk.size() >= res.size(),
            "invalid share: public key less precise than the share"
        );
        let (dnum, dsize): (usize, usize) = (res.dnum().into(), res.dsize().into());
        {
            let sk = sk.to_backend_ref();
            let mut res_be = <GGLWEPatOwned<BE> as GGLWEToBackendMut<BE>>::to_backend_mut(res);
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
        res.canonical = true;
    }
}
