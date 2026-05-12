use std::marker::PhantomData;

use poulpy_core::{
    Distribution, EncryptionInfos, GGSWCompressedEncryptSk, GetDistribution, ScratchArenaTakeCore,
    layouts::{
        GGSWCompressed, GGSWInfos, GLWEInfos, GLWESecretPreparedToBackendRef, LWEInfos, LWESecretToBackendRef,
        ModuleCoreCompressedAlloc,
    },
};
use poulpy_hal::{
    api::ModuleN,
    layouts::{Backend, HostDataMut, HostDataRef, Module, ScalarZnx, ScratchArena, ZnxView, ZnxViewMut},
    source::Source,
};

use crate::blind_rotation::{
    BlindRotationKeyCompressed, BlindRotationKeyCompressedEncryptSk, BlindRotationKeyCompressedFactory, BlindRotationKeyInfos,
    CGGI,
};

impl<D: HostDataRef> BlindRotationKeyCompressedFactory<CGGI> for BlindRotationKeyCompressed<D, CGGI> {
    fn blind_rotation_key_compressed_alloc<M, A>(module: &M, infos: &A) -> BlindRotationKeyCompressed<Vec<u8>, CGGI>
    where
        M: ModuleCoreCompressedAlloc + ModuleN,
        A: BlindRotationKeyInfos,
    {
        #[cfg(debug_assertions)]
        {
            assert_eq!(module.n(), infos.n_glwe().as_usize());
        }
        let mut data: Vec<GGSWCompressed<Vec<u8>>> = Vec::with_capacity(infos.n_lwe().into());
        (0..infos.n_lwe().as_usize()).for_each(|_| data.push(module.ggsw_compressed_alloc_from_infos(infos)));
        BlindRotationKeyCompressed {
            keys: data,
            dist: Distribution::NONE,
            _phantom: PhantomData,
        }
    }
}

impl<BE: Backend + 'static> BlindRotationKeyCompressedEncryptSk<BE, CGGI> for Module<BE>
where
    Self: GGSWCompressedEncryptSk<BE>,
    // TODO(device): this implementation is still host-backed because the
    // compressed key path also stages plaintext limbs through host views.
    BE::OwnedBuf: HostDataMut,
    for<'a> BE::BufMut<'a>: HostDataMut,
    for<'a> BE: Backend<BufMut<'a> = &'a mut [u8], BufRef<'a> = &'a [u8]>,
{
    fn blind_rotation_key_compressed_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGSWInfos,
    {
        self.ggsw_compressed_encrypt_sk_tmp_bytes(infos)
    }

    fn blind_rotation_key_compressed_encrypt_sk<'s, S0, S1, E>(
        &self,
        res: &mut BlindRotationKeyCompressed<BE::OwnedBuf, CGGI>,
        sk_glwe: &S0,
        sk_lwe: &S1,
        seed_xa: [u8; 32],
        enc_infos: &E,
        source_xe: &mut Source,
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

            let mut source_xa: Source = Source::new(seed_xa);

            res.dist = sk_lwe.dist();

            let mut pt: ScalarZnx<BE::OwnedBuf> = self.scalar_znx_alloc(1);
            let sk_ref = sk_lwe.data();

            for (i, ggsw) in res.keys.iter_mut().enumerate() {
                pt.at_mut(0, 0)[0] = sk_ref.at(0, 0)[i];
                let mut scratch_iter = scratch.borrow();
                self.ggsw_compressed_encrypt_sk(
                    ggsw,
                    &pt,
                    sk_glwe,
                    source_xa.new_seed(),
                    enc_infos,
                    source_xe,
                    &mut scratch_iter,
                );
            }
        }
    }
}
