//! Secret-derived diagnostic temporaries are erased without touching caller data.

use poulpy_hal::{
    api::{ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Backend, HostDataMut, HostDataRef, Module, ScalarZnx, ScratchArena, ScratchOwned, Stats, ZnxViewMut},
    source::Source,
    test_suite::TestParams,
};

use crate::{
    EncryptionLayout, GGLWEEncryptSk, GGLWENoise, GGSWEncryptSk, GGSWNoise, GLWEEncryptSk, GLWENoise,
    layouts::{GGLWELayout, GGSWLayout, GLWELayout, GLWESecretPreparedFactory, GLWESecretSampling, LWEInfos, ModuleCoreAlloc},
};

fn assert_noise_wipes_own_prefix<BE: Backend>(bytes: usize, mut op: impl FnMut(&mut ScratchArena<'_, BE>) -> Stats) {
    let mut expected = None;
    crate::test_suite::assert_wipes_scratch::<BE>(bytes, |scratch| expected = Some(op(scratch)));
    let expected = expected.unwrap();
    assert!(expected.std().is_finite() && expected.std() > 0.0);

    let guard = BE::SCRATCH_ALIGN;
    let mut scratch = ScratchOwned::<BE> {
        data: BE::from_host_bytes(&vec![0xA5; guard + bytes + guard]),
        _phantom: std::marker::PhantomData,
    };
    let (_, mut arena) = scratch.arena().split_at(guard);
    let available = arena.available();
    let actual = op(&mut arena);
    assert_eq!(
        arena.available(),
        available,
        "noise diagnostics must not consume the caller's arena"
    );
    assert_eq!(actual.std(), expected.std(), "scratch contents changed the noise statistics");
    assert_eq!(actual.max(), expected.max(), "scratch contents changed the noise statistics");

    let data = BE::to_host_bytes(&scratch.data);
    assert!(
        data[..guard].iter().all(|&byte| byte == 0xA5),
        "noise diagnostics erased the caller's prefix"
    );
    assert!(
        data[guard..guard + bytes].iter().all(|&byte| byte == 0),
        "noise diagnostics retained scratch data"
    );
    assert!(
        data[guard + bytes..].iter().all(|&byte| byte == 0xA5),
        "noise diagnostics erased beyond their scratch budget"
    );
}

pub fn test_noise_wipes_scratch<BE: super::TestBackend>(params: &TestParams, module: &Module<BE>)
where
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
    Module<BE>: GLWESecretPreparedFactory<BE>
        + GLWEEncryptSk<BE>
        + GGLWEEncryptSk<BE>
        + GGSWEncryptSk<BE>
        + GLWENoise<BE>
        + GGLWENoise<BE>
        + GGSWNoise<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let base2k = params.base2k;
    let ggsw_layout = GGSWLayout {
        n: module.n().into(),
        base2k: base2k.into(),
        rank: 2_usize.into(),
        dnum: 2_usize.into(),
        dsize: 2_usize.into(),
        k_aux: (2 * base2k + module.log_n()).into(),
    };
    let gglwe_layout = GGLWELayout {
        n: ggsw_layout.n,
        base2k: ggsw_layout.base2k,
        rank_in: 2_usize.into(),
        rank_out: ggsw_layout.rank,
        dnum: ggsw_layout.dnum,
        dsize: ggsw_layout.dsize,
        k_aux: ggsw_layout.k_aux,
        stride: 1,
    };
    let glwe_layout = GLWELayout {
        n: ggsw_layout.n,
        base2k: ggsw_layout.base2k,
        rank: ggsw_layout.rank,
        k: ggsw_layout.k(),
    };
    let mut secret = module.glwe_secret_alloc(ggsw_layout.rank);
    module.glwe_secret_fill_ternary_prob(&mut secret, 0.5, &mut Source::new([31; 32]));
    let mut prepared = module.glwe_secret_prepared_alloc(ggsw_layout.rank);
    module.glwe_secret_prepare(&mut prepared, &secret);

    let mut scratch = ScratchOwned::<BE>::alloc(
        module
            .glwe_encrypt_sk_tmp_bytes(&glwe_layout)
            .max(module.gglwe_encrypt_sk_tmp_bytes(&gglwe_layout))
            .max(module.ggsw_encrypt_sk_tmp_bytes(&ggsw_layout)),
    );
    let mut errors = Source::new([32; 32]);
    let mut masks = Source::new([33; 32]);
    let mut plaintext = module.glwe_plaintext_alloc_from_infos(&glwe_layout);
    plaintext.data.at_mut(0, 0)[0] = 1;
    let mut glwe = module.glwe_alloc_from_infos(&glwe_layout);
    module.glwe_encrypt_sk(
        &mut glwe,
        &plaintext,
        &prepared,
        &EncryptionLayout::new_from_default_sigma(glwe_layout).unwrap(),
        &mut errors,
        &mut masks,
        &mut scratch.borrow(),
    );
    for canonical in [true, false] {
        glwe.canonical = canonical;
        assert_noise_wipes_own_prefix::<BE>(module.glwe_noise_tmp_bytes(&glwe), |scratch| {
            module.glwe_noise(&glwe, &plaintext, &prepared, scratch)
        });
    }

    let mut messages: ScalarZnx<BE::OwnedBuf, BE::ZnxWord> = module.scalar_znx_alloc(module.n(), 2);
    messages.at_mut(0, 0)[0] = 1;
    messages.at_mut(1, 0)[1] = -1;
    let mut gglwe = module.gglwe_alloc_from_infos(&gglwe_layout);
    module.gglwe_encrypt_sk(
        &mut gglwe,
        &messages,
        &prepared,
        &EncryptionLayout::new_from_default_sigma(gglwe_layout).unwrap(),
        &mut errors,
        &mut masks,
        &mut scratch.borrow(),
    );
    for row in 0..2 {
        for col in 0..2 {
            assert_noise_wipes_own_prefix::<BE>(module.gglwe_noise_tmp_bytes(&gglwe), |scratch| {
                module.gglwe_noise(&gglwe, row, col, &messages.to_ref(), &prepared, scratch)
            });
        }
    }

    let mut message: ScalarZnx<BE::OwnedBuf, BE::ZnxWord> = module.scalar_znx_alloc(module.n(), 1);
    message.at_mut(0, 0)[1] = 1;
    let mut ggsw = module.ggsw_alloc_from_infos(&ggsw_layout);
    module.ggsw_encrypt_sk(
        &mut ggsw,
        &message,
        &prepared,
        &EncryptionLayout::new_from_default_sigma(ggsw_layout).unwrap(),
        &mut errors,
        &mut masks,
        &mut scratch.borrow(),
    );
    for row in 0..2 {
        for col in 0..3 {
            assert_noise_wipes_own_prefix::<BE>(module.ggsw_noise_tmp_bytes(&ggsw), |scratch| {
                module.ggsw_noise(&ggsw, row, col, &message.to_ref(), &prepared, scratch)
            });
        }
    }
}
