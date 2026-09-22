#![allow(clippy::too_many_arguments)]
use std::mem::size_of;

use poulpy_hal::{
    layouts::{Backend, Module, ScalarZnx, ScratchArena},
    source::Source,
};

use poulpy_core::{
    Distribution, EncryptionInfos, GGSWEncryptSk, GetDistribution,
    layouts::{GGSWInfos, GLWEInfos, GLWESecretPreparedToBackendRef, LWEInfos, LWESecretToBackendRef},
};

use crate::blind_rotation::{BlindRotationKey, CGGI};

/// Canonical lower-layer composition for `blind_rotation_key_encrypt_sk_tmp_bytes`.
pub fn blind_rotation_key_encrypt_sk_tmp_bytes_ref<BE, A: GGSWInfos>(module: &Module<BE>, infos: &A) -> usize
where
    BE: Backend<ZnxWord = i64> + 'static,
    Module<BE>: GGSWEncryptSk<BE>,
{
    module.ggsw_encrypt_sk_tmp_bytes(infos)
}

/// Canonical lower-layer composition for `blind_rotation_key_encrypt_sk`.
pub fn blind_rotation_key_encrypt_sk_ref<BE, S0, S1, E>(
    module: &Module<BE>,
    res: &mut BlindRotationKey<BE::OwnedBuf, CGGI, BE::ZnxWord>,
    sk_glwe: &S0,
    sk_lwe: &S1,
    enc_infos: &E,
    source_xe: &mut Source,
    source_xa: &mut Source,
    scratch: &mut ScratchArena<'_, BE>,
) where
    S0: GLWESecretPreparedToBackendRef<BE> + GLWEInfos,
    E: EncryptionInfos,
    S1: LWESecretToBackendRef<BE> + LWEInfos + GetDistribution,
    BE: Backend<ZnxWord = i64> + 'static,
    Module<BE>: GGSWEncryptSk<BE>,
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

        res.dist = *sk_lwe.dist();

        let mut pt: ScalarZnx<BE::OwnedBuf, BE::ZnxWord> = module.scalar_znx_alloc(module.n(), 1);
        let sk_ref = sk_lwe.data();
        let mut sk_host = vec![0u8; BE::bytes_of_scalar_znx(sk_ref.n(), sk_ref.cols())];
        BE::copy_view_to_host(&BE::region_ref(&sk_ref.data, 0, sk_host.len()), &mut sk_host);
        let mut pt_host = vec![0u8; BE::bytes_of_scalar_znx(pt.n(), pt.cols())];

        for (i, ggsw) in res.keys.iter_mut().enumerate() {
            pt_host[..size_of::<i64>()].copy_from_slice(&sk_host[i * size_of::<i64>()..(i + 1) * size_of::<i64>()]);
            BE::copy_from_host(&mut pt.data, &pt_host);
            let mut scratch_iter = scratch.borrow();
            module.ggsw_encrypt_sk(ggsw, &pt, sk_glwe, enc_infos, source_xe, source_xa, &mut scratch_iter);
        }
    }
}
