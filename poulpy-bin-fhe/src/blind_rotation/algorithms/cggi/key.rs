use poulpy_hal::{
    layouts::{Backend, HostDataMut, Module, ScalarZnx, ScratchArena, ZnxView, ZnxViewMut},
    source::Source,
};

use poulpy_core::{
    Distribution, EncryptionInfos, GGSWEncryptSk, GetDistribution, ScratchArenaTakeCore,
    layouts::{GGSWInfos, GLWEInfos, GLWESecretPreparedToBackendRef, LWEInfos, LWESecretToBackendRef},
};

use crate::blind_rotation::{BlindRotationKey, BlindRotationKeyEncryptSk, CGGI};

impl<BE: Backend + 'static> BlindRotationKeyEncryptSk<CGGI, BE> for Module<BE>
where
    Self: GGSWEncryptSk<BE>,
    // TODO(device): this implementation is still host-backed because the
    // plaintext staging buffer is built via host-visible scalar views.
    BE::OwnedBuf: HostDataMut,
    for<'a> BE::BufMut<'a>: HostDataMut,
    for<'a> BE: Backend<BufMut<'a> = &'a mut [u8], BufRef<'a> = &'a [u8]>,
{
    fn blind_rotation_key_encrypt_sk_tmp_bytes<A: GGSWInfos>(&self, infos: &A) -> usize {
        self.ggsw_encrypt_sk_tmp_bytes(infos)
    }

    fn blind_rotation_key_encrypt_sk<'s, S0, S1, E>(
        &self,
        res: &mut BlindRotationKey<BE::OwnedBuf, CGGI>,
        sk_glwe: &S0,
        sk_lwe: &S1,
        enc_infos: &E,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        S0: GLWESecretPreparedToBackendRef<BE> + GLWEInfos,
        E: EncryptionInfos,
        S1: LWESecretToBackendRef<BE> + LWEInfos + GetDistribution,
        ScratchArena<'s, BE>: ScratchArenaTakeCore<'s, BE>,
        BE: 's,
    {
        assert_eq!(res.keys.len() as u32, sk_lwe.n());
        assert!(sk_glwe.n() <= self.n() as u32);
        assert_eq!(sk_glwe.rank(), res.keys[0].rank());

        match sk_lwe.dist() {
            Distribution::BinaryBlock(_) | Distribution::BinaryFixed(_) | Distribution::BinaryProb(_) | Distribution::ZERO => {}
            _ => {
                panic!("invalid GLWESecret distribution: must be BinaryBlock, BinaryFixed or BinaryProb (or ZERO for debugging)")
            }
        }

        {
            let sk_lwe = sk_lwe.to_backend_ref();

            res.dist = sk_lwe.dist();

            let mut pt: ScalarZnx<BE::OwnedBuf> = self.scalar_znx_alloc(1);
            let sk_ref = sk_lwe.data();

            for (i, ggsw) in res.keys.iter_mut().enumerate() {
                pt.at_mut(0, 0)[0] = sk_ref.at(0, 0)[i];
                let mut scratch_iter = scratch.borrow();
                self.ggsw_encrypt_sk(ggsw, &pt, sk_glwe, enc_infos, source_xe, source_xa, &mut scratch_iter);
            }
        }
    }
}
