#![allow(clippy::too_many_arguments)]

use poulpy_core::{
    Distribution, EncryptionInfos, GGSWCompressedEncryptSk, GetDistribution,
    layouts::{GGSWInfos, GLWEInfos, GLWESecretPreparedToBackendRef, LWEInfos, LWESecretToBackendRef},
};
use poulpy_hal::{
    layouts::{Backend, Module, ScalarZnx, ScratchArena},
    source::Source,
};

use crate::blind_rotation::{BlindRotationKeyCompressed, CGGI};

/// Canonical lower-layer composition for `blind_rotation_key_compressed_encrypt_sk_tmp_bytes`.
pub fn blind_rotation_key_compressed_encrypt_sk_tmp_bytes_ref<BE, A>(module: &Module<BE>, infos: &A) -> usize
where
    A: GGSWInfos,
    BE: Backend<ZnxWord = i64> + 'static,
    Module<BE>: GGSWCompressedEncryptSk<BE>,
{
    module.ggsw_compressed_encrypt_sk_tmp_bytes(infos)
}

/// Canonical lower-layer composition for `blind_rotation_key_compressed_encrypt_sk`.
pub fn blind_rotation_key_compressed_encrypt_sk_ref<BE, S0, S1, E>(
    module: &Module<BE>,
    res: &mut BlindRotationKeyCompressed<BE::OwnedBuf, CGGI, BE::ZnxWord>,
    sk_glwe: &S0,
    sk_lwe: &S1,
    seed_xa: [u8; 32],
    enc_infos: &E,
    source_xe: &mut Source,
    scratch: &mut ScratchArena<'_, BE>,
) where
    S0: GLWESecretPreparedToBackendRef<BE> + GLWEInfos,
    E: EncryptionInfos,
    S1: LWESecretToBackendRef<BE> + LWEInfos + GetDistribution,
    BE: Backend<ZnxWord = i64> + 'static,
    Module<BE>: GGSWCompressedEncryptSk<BE>,
{
    assert_eq!(res.keys.len() as u32, sk_lwe.n());
    assert!(sk_glwe.n() <= module.n() as u32);
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

        res.dist = *sk_lwe.dist();

        let mut pt: ScalarZnx<BE::OwnedBuf, BE::ZnxWord> = module.scalar_znx_alloc(module.n(), 1);
        let sk_ref = sk_lwe.data();
        let mut sk_host = vec![0u8; BE::bytes_of_scalar_znx(sk_ref.n(), sk_ref.cols())];
        BE::copy_view_to_host(&BE::region_ref(&sk_ref.data, 0, sk_host.len()), &mut sk_host);
        let mut pt_host = vec![0u8; BE::bytes_of_scalar_znx(pt.n(), pt.cols())];

        for (i, ggsw) in res.keys.iter_mut().enumerate() {
            let word_bytes = core::mem::size_of::<i64>();
            pt_host[..word_bytes].copy_from_slice(&sk_host[i * word_bytes..(i + 1) * word_bytes]);
            BE::copy_from_host(&mut pt.data, &pt_host);
            let mut scratch_iter = scratch.borrow();
            module.ggsw_compressed_encrypt_sk(
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
