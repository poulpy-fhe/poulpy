use poulpy_core::{
    Distribution, GLWEEncryptPk, GetDistribution, ScratchArenaTakeCore,
    api::GLWEBytesOf,
    layouts::{GGLWEInfos, GGLWEToBackendMut, GLWEInfos, GLWEPublicKeyPreparedToBackendRef, GLWESecretToBackendRef, LWEInfos},
};
use poulpy_hal::{
    api::{VecZnxAddScalarAssign, VecZnxZero},
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
    fn mhe_glwe_tensor_key_share_gen_reference<S, K>(
        &self,
        res: &mut GLWETensorKeyShareOwned<BE>,
        sk: &S,
        pk: &K,
        source_xu: &mut Source,
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        S: GLWESecretToBackendRef<BE> + GLWEInfos,
        K: GLWEPublicKeyPreparedToBackendRef<BE> + GLWEInfos;
}

impl<BE: Backend> GLWETensorKeyMHEProtocolReference<BE> for Module<BE>
where
    Self: GLWEEncryptPk<BE> + GLWEBytesOf<BE> + VecZnxZero<BE> + VecZnxAddScalarAssign<BE>,
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
        BE::scratch_aligned(self.glwe_plaintext_bytes_of_from_infos(res_infos))
            + self.glwe_encrypt_pk_tmp_bytes(res_infos, pk_infos)
    }

    fn mhe_glwe_tensor_key_share_gen_reference<S, K>(
        &self,
        res: &mut GLWETensorKeyShareOwned<BE>,
        sk: &S,
        pk: &K,
        source_xu: &mut Source,
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        S: GLWESecretToBackendRef<BE> + GLWEInfos,
        K: GLWEPublicKeyPreparedToBackendRef<BE> + GLWEInfos,
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
        super::assert_public_key_distribution::<BE, _>(pk);
        let (dnum, dsize): (usize, usize) = (res.dnum().into(), res.dsize().into());
        let tmp_bytes: usize = self.mhe_glwe_tensor_key_share_gen_tmp_bytes_reference(&*res, pk);
        let mut metadata = None;
        {
            let sk = sk.to_backend_ref();
            let (mut pt, mut scratch_1) = scratch.borrow().take_glwe_plaintext_scratch(&*res);
            let mut res_be = GGLWEToBackendMut::<BE>::to_backend_mut(&mut res.key);
            for row in 0..dnum {
                for a in 0..rank {
                    for b in a..rank {
                        self.vec_znx_zero(pt.data_mut(), 0);
                        self.vec_znx_add_scalar_assign(pt.data_mut(), 0, (dsize - 1) + row * dsize, sk.data(), b);
                        // Mask column 1 + a meets S_a at decryption: s_b there sums to S_a * S_b.
                        let mut cell = res_be.at_view_mut(row, a * rank + b - a * (a + 1) / 2);
                        self.glwe_encrypt_pk_at_col(&mut cell, &pt, 1 + a, pk, source_xu, source_xe, &mut scratch_1.borrow());
                        metadata = cell.encryption_metadata();
                    }
                }
            }
        }
        GGLWEToBackendMut::<BE>::set_encryption_metadata(res, metadata);
        scratch.wipe(tmp_bytes);
    }
}
