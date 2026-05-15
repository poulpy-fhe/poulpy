use poulpy_hal::{
    api::{ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxNormalize},
    layouts::{Module, ScratchOwned, ZnxView},
    source::Source,
    test_suite::{TestParams, vec_znx_backend_mut, vec_znx_backend_ref},
};

use crate::{
    EncryptionLayout, LWEDecrypt, LWEEncryptSk, LWEKeyswitch, LWESwitchingKeyEncrypt, ScratchArenaTakeCore,
    layouts::{
        LWE, LWEInfos, LWELayout, LWEPlaintext, LWESecret, LWESwitchingKey, LWESwitchingKeyLayout,
        LWESwitchingKeyPreparedFactory, ModuleCoreAlloc, prepared::LWESwitchingKeyPrepared,
    },
};

pub fn test_lwe_keyswitch<BE: crate::test_suite::TestBackend>(params: &TestParams, module: &Module<BE>)
where
    BE::OwnedBuf: poulpy_hal::layouts::HostDataMut,
    for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: LWEKeyswitch<BE>
        + LWESwitchingKeyEncrypt<BE>
        + LWEEncryptSk<BE>
        + LWESwitchingKeyPreparedFactory<BE>
        + LWEDecrypt<BE>
        + VecZnxNormalize<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    for<'a> poulpy_hal::layouts::ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
{
    let n: usize = module.n();
    let base2k: usize = params.base2k;
    let in_base2k: usize = base2k - 1;
    let out_base2k: usize = base2k - 2;
    let key_base2k: usize = base2k;

    let n_lwe_in: usize = module.n() >> 1;
    let n_lwe_out: usize = module.n() >> 1;
    let k_lwe_ct: usize = 4 * base2k + 1;
    let k_lwe_pt: usize = 8;

    let k_ksk: usize = k_lwe_ct + key_base2k;
    let dnum: usize = k_lwe_ct.div_ceil(key_base2k);

    let mut source_xs: Source = Source::new([0u8; 32]);
    let mut source_xa: Source = Source::new([0u8; 32]);
    let mut source_xe: Source = Source::new([0u8; 32]);

    let key_apply_infos = EncryptionLayout::new_from_default_sigma(LWESwitchingKeyLayout {
        n: n.into(),
        base2k: key_base2k.into(),
        k: k_ksk.into(),
        dnum: dnum.into(),
    })
    .unwrap();

    let lwe_in_infos = EncryptionLayout::new_from_default_sigma(LWELayout {
        n: n_lwe_in.into(),
        base2k: in_base2k.into(),
        k: k_lwe_ct.into(),
    })
    .unwrap();

    let lwe_out_infos: LWELayout = LWELayout {
        n: n_lwe_out.into(),
        k: k_lwe_ct.into(),
        base2k: out_base2k.into(),
    };

    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
        (module).lwe_switching_key_encrypt_sk_tmp_bytes(&key_apply_infos)
            | module.lwe_keyswitch_tmp_bytes(&lwe_out_infos, &lwe_in_infos, &key_apply_infos),
    );

    let mut sk_lwe_in: LWESecret<Vec<u8>> = module.lwe_secret_alloc(n_lwe_in.into());
    sk_lwe_in.fill_ternary_prob(0.5, &mut source_xs);

    let mut sk_lwe_out: LWESecret<Vec<u8>> = module.lwe_secret_alloc(n_lwe_out.into());
    sk_lwe_out.fill_ternary_prob(0.5, &mut source_xs);

    let data: i64 = 17;

    let mut lwe_pt_in: LWEPlaintext<Vec<u8>> = module.lwe_plaintext_alloc(in_base2k.into(), k_lwe_pt.into());
    lwe_pt_in.encode_i64(data, k_lwe_pt.into());

    let mut lwe_ct_in: LWE<Vec<u8>> = module.lwe_alloc_from_infos(&lwe_in_infos);
    module.lwe_encrypt_sk(
        &mut lwe_ct_in,
        &lwe_pt_in,
        &sk_lwe_in,
        &lwe_in_infos,
        &mut source_xe,
        &mut source_xa,
        &mut scratch.borrow(),
    );

    let mut ksk: LWESwitchingKey<Vec<u8>> = module.lwe_switching_key_alloc_from_infos(&key_apply_infos);

    module.lwe_switching_key_encrypt_sk(
        &mut ksk,
        &sk_lwe_in,
        &sk_lwe_out,
        &key_apply_infos,
        &mut source_xe,
        &mut source_xa,
        &mut crate::test_suite::scratch_host_arena(&mut scratch),
    );

    let mut lwe_ct_out: LWE<Vec<u8>> = module.lwe_alloc_from_infos(&lwe_out_infos);

    let mut ksk_prepared: LWESwitchingKeyPrepared<BE::OwnedBuf, BE> = module.lwe_switching_key_prepared_alloc_from_infos(&ksk);
    module.lwe_switching_key_prepare(&mut ksk_prepared, &ksk, &mut scratch.borrow());

    module.lwe_keyswitch(
        &mut lwe_ct_out,
        &lwe_ct_in,
        &ksk_prepared,
        ksk_prepared.size(),
        &mut scratch.borrow(),
    );

    let mut lwe_pt_out: LWEPlaintext<Vec<u8>> = module.lwe_plaintext_alloc_from_infos(&lwe_out_infos);
    module.lwe_decrypt(&lwe_ct_out, &mut lwe_pt_out, &sk_lwe_out, &mut scratch.borrow());

    let mut lwe_pt_want: LWEPlaintext<Vec<u8>> = module.lwe_plaintext_alloc_from_infos(&lwe_out_infos);
    module.vec_znx_normalize(
        &mut vec_znx_backend_mut::<BE>(&mut lwe_pt_want.data),
        out_base2k,
        0,
        0,
        &vec_znx_backend_ref::<BE>(&lwe_pt_in.data),
        in_base2k,
        0,
        &mut scratch.borrow(),
    );

    assert_eq!(lwe_pt_want.data.at(0, 0)[0], lwe_pt_out.data.at(0, 0)[0]);
}
