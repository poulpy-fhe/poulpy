use std::{
    marker::PhantomData,
    panic::{AssertUnwindSafe, catch_unwind},
};

use poulpy_hal::{
    api::{ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{DataViewMut, Module, ScratchArena, ScratchOwned},
    source::Source,
    test_suite::TestParams,
};

use crate::{
    ComponentNoise, Distribution, GLWECIKeyEncryptSk, GetDistributionMut,
    layouts::{GGLWELayout, GLWECIKey, GLWESecret, GLWESecretLayout, ModuleCoreAlloc},
    oep::GLWECIConversionImpl,
    test_suite::{decryption::panic_message, poisoned_scratch},
};

/// A refused conjugate-invariant key encryption: the key degree, the degree and rank of `sk_ci`, the degree, rank and
/// law of `sk`, and the message.
type Case = (usize, usize, usize, usize, usize, Distribution, &'static str);

type Secret<BE> = GLWESecret<<BE as poulpy_hal::layouts::Backend>::OwnedBuf, <BE as poulpy_hal::layouts::Backend>::ZnxWord>;

/// Encrypting an embed key from a conjugate-invariant secret of more than half the key degree or of a rank other than
/// the key's input rank, or to a secret wider than the key, of a rank other than its output rank or of the `NONE` law,
/// is refused with the key, the sources and the scratch as they were.
pub fn test_glwe_ci_embed_key_encrypt_sk_rejects_mismatched_secrets<BE>(params: &TestParams, module: &Module<BE>)
where
    BE: crate::test_suite::noise::TestBackend + GLWECIConversionImpl,
    BE::OwnedBuf: poulpy_hal::layouts::HostDataMut,
    for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: GLWECIKeyEncryptSk<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let n = module.n();
    let law = Distribution::TernaryProb(0.5);
    let cases: [Case; 5] = [
        (
            n,
            n,
            1,
            n,
            2,
            law,
            "conjugate-invariant key encryption: embedded secret degree exceeds the key's",
        ),
        (
            n,
            n / 2,
            2,
            n,
            2,
            law,
            "conjugate-invariant key encryption: conjugate-invariant secret rank differs from the key's",
        ),
        (
            n / 2,
            n / 4,
            1,
            n,
            2,
            law,
            "conjugate-invariant key encryption: secret degree exceeds the key's",
        ),
        (
            n,
            n / 2,
            1,
            n,
            1,
            law,
            "conjugate-invariant key encryption: secret rank differs from the key's",
        ),
        (
            n,
            n / 2,
            1,
            n,
            2,
            Distribution::NONE,
            "conjugate-invariant key encryption: secret distribution is NONE",
        ),
    ];
    assert_ci_key_refusals(
        params,
        module,
        &cases,
        |res: &mut GLWECIKey<_, _, crate::layouts::CIEmbed>, sk_ci, sk, source_xe, source_xa, scratch| {
            module.glwe_ci_embed_key_encrypt_sk(res, sk_ci, sk, source_xe, source_xa, scratch)
        },
    );
}

/// Encrypting a trace key from a secret wider than the key or of a rank other than its input rank, to a
/// conjugate-invariant secret of more than half the key degree or of a rank other than the key's output rank, is
/// refused with the key, the sources and the scratch as they were.
pub fn test_glwe_ci_trace_key_encrypt_sk_rejects_mismatched_secrets<BE>(params: &TestParams, module: &Module<BE>)
where
    BE: crate::test_suite::noise::TestBackend + GLWECIConversionImpl,
    BE::OwnedBuf: poulpy_hal::layouts::HostDataMut,
    for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: GLWECIKeyEncryptSk<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let n = module.n();
    let law = Distribution::TernaryProb(0.5);
    let cases: [Case; 4] = [
        (
            n,
            n,
            2,
            n,
            1,
            law,
            "conjugate-invariant key encryption: embedded secret degree exceeds the key's",
        ),
        (
            n,
            n / 2,
            1,
            n,
            1,
            law,
            "conjugate-invariant key encryption: conjugate-invariant secret rank differs from the key's",
        ),
        (
            n / 2,
            n / 4,
            2,
            n,
            1,
            law,
            "conjugate-invariant key encryption: secret degree exceeds the key's",
        ),
        (
            n,
            n / 2,
            2,
            n,
            2,
            law,
            "conjugate-invariant key encryption: secret rank differs from the key's",
        ),
    ];
    assert_ci_key_refusals(
        params,
        module,
        &cases,
        |res: &mut GLWECIKey<_, _, crate::layouts::CITrace>, sk_ci, sk, source_xe, source_xa, scratch| {
            module.glwe_ci_trace_key_encrypt_sk(res, sk_ci, sk, source_xe, source_xa, scratch)
        },
    );
}

/// Runs each case on a key of input rank 1 and output rank 2 holding other content, on an empty arena and on exactly
/// the query's poisoned bytes, and asserts the full message, then the key, both sources and the arena unchanged.
fn assert_ci_key_refusals<BE, M>(
    params: &TestParams,
    module: &Module<BE>,
    cases: &[Case],
    encrypt: impl Fn(
        &mut GLWECIKey<BE::OwnedBuf, BE::ZnxWord, M>,
        &Secret<BE>,
        &Secret<BE>,
        &mut Source,
        &mut Source,
        &mut ScratchArena<'_, BE>,
    ),
) where
    BE: crate::test_suite::noise::TestBackend + GLWECIConversionImpl,
    BE::OwnedBuf: poulpy_hal::layouts::HostDataMut,
    for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: GLWECIKeyEncryptSk<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    M: PartialEq + Clone,
{
    let base2k = params.base2k;
    for &(key_n, ci_n, ci_rank, sk_n, sk_rank, dist, message) in cases {
        let layout = GGLWELayout {
            n: key_n.into(),
            base2k: base2k.into(),
            dnum: 2_usize.into(),
            k_digit: base2k.into(),
            k_aux: base2k.into(),
            k_offset: 0_usize.into(),
            rank_in: 1_usize.into(),
            rank_out: 2_usize.into(),
            stride: 1,
        };
        let sk_ci: Secret<BE> = module.glwe_secret_alloc_from_infos(&GLWESecretLayout {
            n: ci_n.into(),
            rank: ci_rank.into(),
        });
        let mut sk: Secret<BE> = module.glwe_secret_alloc_from_infos(&GLWESecretLayout {
            n: sk_n.into(),
            rank: sk_rank.into(),
        });
        *sk.dist_mut() = dist;
        for bytes in [0, module.glwe_ci_key_encrypt_sk_tmp_bytes(&layout)] {
            let mut res: GLWECIKey<BE::OwnedBuf, BE::ZnxWord, M> =
                GLWECIKey(module.glwe_switching_key_alloc_from_infos(&layout), PhantomData);
            res.0
                .key
                .data
                .data_mut()
                .iter_mut()
                .enumerate()
                .for_each(|(i, x)| *x = (i as u8).wrapping_mul(29).wrapping_add(11));
            res.0.key.noise = Some(ComponentNoise::from_secret_at(
                Distribution::TernaryProb(0.5),
                layout.k_aux,
                2,
            ));
            let before = res.clone();
            let mut scratch: ScratchOwned<BE> = poisoned_scratch::<BE>(bytes);
            let arena = BE::to_host_bytes(&scratch.data);
            let (mut source_xe, mut source_xa) = (Source::new([7u8; 32]), Source::new([8u8; 32]));
            let err = catch_unwind(AssertUnwindSafe(|| {
                encrypt(&mut res, &sk_ci, &sk, &mut source_xe, &mut source_xa, &mut scratch.borrow())
            }))
            .expect_err("the call was accepted");
            assert_eq!(panic_message(&*err), message);
            assert!(res == before, "a refused call wrote into its destination");
            assert!(
                BE::to_host_bytes(&scratch.data) == arena,
                "a refused call wrote into its scratch"
            );
            assert_eq!(
                (source_xe.next_u128(), source_xa.next_u128()),
                (Source::new([7u8; 32]).next_u128(), Source::new([8u8; 32]).next_u128()),
                "a refused call drew from its sources"
            );
        }
    }
}
