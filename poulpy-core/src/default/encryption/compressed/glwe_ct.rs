#![allow(clippy::too_many_arguments)]

use poulpy_hal::{
    layouts::{Backend, Module, ScratchArena},
    source::Source,
};

use crate::{
    EncryptionInfos, ScratchArenaTakeCore,
    encryption::{GLWEEncryptSk, GLWEEncryptSkInternal},
    layouts::{
        GLWECompressedSeedMut, GLWEInfos, GLWEToBackendRef, LWEInfos, compressed::GLWECompressedToBackendMut,
        prepared::GLWESecretPreparedToBackendRef,
    },
};

#[doc(hidden)]
pub trait GLWECompressedEncryptSkDefault<BE: Backend> {
    fn glwe_compressed_encrypt_sk_tmp_bytes_default<A>(&self, infos: &A) -> usize
    where
        A: GLWEInfos;

    fn glwe_compressed_encrypt_sk_default<'s, R, P, S, E>(
        &self,
        res: &'s mut R,
        pt: &P,
        sk: &S,
        seed_xa: [u8; 32],
        enc_infos: &E,
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        R: GLWECompressedToBackendMut<BE> + GLWECompressedSeedMut,
        P: GLWEToBackendRef<BE>,
        E: EncryptionInfos,
        S: GLWESecretPreparedToBackendRef<BE>,
        BE: 's,
        ScratchArena<'s, BE>: ScratchArenaTakeCore<'s, BE>;
}

impl<BE: Backend> GLWECompressedEncryptSkDefault<BE> for Module<BE>
where
    Self: GLWEEncryptSkInternal<BE> + GLWEEncryptSk<BE>,
{
    fn glwe_compressed_encrypt_sk_tmp_bytes_default<A>(&self, infos: &A) -> usize
    where
        A: GLWEInfos,
    {
        assert_eq!(self.n() as u32, infos.n());
        let lvl_0: usize = self.glwe_encrypt_sk_tmp_bytes(infos);
        lvl_0
    }

    #[allow(clippy::too_many_arguments)]
    fn glwe_compressed_encrypt_sk_default<'s, R, P, S, E>(
        &self,
        res: &'s mut R,
        pt: &P,
        sk: &S,
        seed_xa: [u8; 32],
        enc_infos: &E,
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        R: GLWECompressedToBackendMut<BE> + GLWECompressedSeedMut,
        P: GLWEToBackendRef<BE>,
        E: EncryptionInfos,
        S: GLWESecretPreparedToBackendRef<BE>,
        BE: 's,
        ScratchArena<'s, BE>: ScratchArenaTakeCore<'s, BE>,
    {
        res.seed_mut().copy_from_slice(&seed_xa);

        {
            let mut res_backend = res.to_backend_mut();
            let mut source_xa: Source = Source::new(seed_xa);
            let cols: usize = (res_backend.rank() + 1).into();
            assert!(
                scratch.available() >= self.glwe_compressed_encrypt_sk_tmp_bytes_default(&res_backend),
                "scratch.available(): {} < GLWECompressedEncryptSk::glwe_compressed_encrypt_sk_tmp_bytes: {}",
                scratch.available(),
                self.glwe_compressed_encrypt_sk_tmp_bytes_default(&res_backend)
            );

            self.glwe_encrypt_sk_internal(
                res_backend.base2k().into(),
                &mut res_backend.data,
                cols,
                true,
                Some((pt.to_backend_ref(), 0)),
                sk,
                enc_infos,
                source_xe,
                &mut source_xa,
                scratch,
            );
        }
    }
}
