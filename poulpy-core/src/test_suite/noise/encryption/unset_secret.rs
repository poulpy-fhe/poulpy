use poulpy_hal::{
    api::{ScalarZnxAlloc, ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Module, ScratchArena, ScratchOwned},
    source::Source,
    test_suite::TestParams,
};

use crate::{
    GGLWECompressedEncryptSk, GGLWEToGGSWKeyCompressedEncryptSk, GGLWEToGGSWKeyEncryptSk, GGSWCompressedEncryptSk, GGSWEncryptSk,
    GLWEAutomorphismKeyCompressedEncryptSk, GLWEAutomorphismKeyEncryptSk, GLWECompressedEncryptSk, GLWEEncryptSk,
    GLWESwitchingKeyCompressedEncryptSk, GLWESwitchingKeyEncryptSk, GLWETensorKeyCompressedEncryptSk, GLWETensorKeyEncryptSk,
    GLWEToLWEKeyEncryptSk, LWEEncryptSk, LWESwitchingKeyEncryptSk, LWEToGLWEKeyEncryptSk,
    api::GLWESecretPreparedFactory,
    layouts::{
        Base2K, Degree, Dnum, GGLWECompressedAlloc, GGLWELayout, GGLWEToGGSWKeyAlloc, GGLWEToGGSWKeyCompressedAlloc, GGSWAlloc,
        GGSWCompressedAlloc, GGSWLayout, GLWEAlloc, GLWEAutomorphismKeyAlloc, GLWEAutomorphismKeyCompressedAlloc,
        GLWECompressedAlloc, GLWELayout, GLWEPlaintextAlloc, GLWESecret, GLWESecretAlloc, GLWESecretLayout,
        GLWESwitchingKeyAlloc, GLWESwitchingKeyCompressedAlloc, GLWETensorKeyAlloc, GLWETensorKeyCompressedAlloc,
        GLWEToLWEKeyAlloc, LWEAlloc, LWEPlaintextAlloc, LWESecret, LWESecretAlloc, LWESwitchingKeyAlloc, LWEToGLWEKeyAlloc, Rank,
        TorusPrecision, prepared::GLWESecretPrepared,
    },
    test_suite::{poisoned_scratch, refusal::assert_refusal},
};

const GLWE_NONE: &str = "glwe secret distribution is NONE (have you prepared the key?)";

type Secret<BE> = GLWESecret<<BE as poulpy_hal::layouts::Backend>::OwnedBuf, <BE as poulpy_hal::layouts::Backend>::ZnxWord>;
type Prepared<BE> = GLWESecretPrepared<<BE as poulpy_hal::layouts::Backend>::OwnedBuf, BE>;

/// Runs `encrypt` on `res` on an empty arena and asserts the full message, `res` and both sources unchanged.
fn assert_unset_secret_refused<BE, T>(
    message: &str,
    res: &mut T,
    encrypt: impl FnOnce(&mut T, &mut Source, &mut Source, &mut ScratchArena<'_, BE>),
) where
    BE: poulpy_hal::layouts::Backend,
    T: Clone + PartialEq,
{
    let mut scratch = poisoned_scratch::<BE>(0);
    let (mut source_xe, mut source_xa) = (Source::new([7u8; 32]), Source::new([8u8; 32]));
    assert_refusal(
        message,
        res,
        |res| res.clone(),
        |res| encrypt(res, &mut source_xe, &mut source_xa, &mut scratch.borrow()),
    );
    assert_eq!(
        (source_xe.next_u128(), source_xa.next_u128()),
        (Source::new([7u8; 32]).next_u128(), Source::new([8u8; 32]).next_u128()),
        "a refused call drew from its sources"
    );
}

fn glwe_layout<BE: poulpy_hal::layouts::Backend>(params: &TestParams, module: &Module<BE>, rank: usize) -> GLWELayout {
    GLWELayout {
        n: module.n().into(),
        base2k: params.base2k.into(),
        k: (2 * params.base2k).into(),
        rank: rank.into(),
    }
}

fn gglwe_layout<BE: poulpy_hal::layouts::Backend>(
    params: &TestParams,
    module: &Module<BE>,
    rank_in: usize,
    rank_out: usize,
) -> GGLWELayout {
    let base2k: Base2K = params.base2k.into();
    GGLWELayout {
        n: module.n().into(),
        base2k,
        dnum: Dnum(2),
        k_digit: TorusPrecision::from(base2k),
        k_aux: TorusPrecision::from(base2k),
        k_offset: TorusPrecision(0),
        rank_in: Rank(rank_in as u32),
        rank_out: Rank(rank_out as u32),
        stride: 1,
    }
}

fn ggsw_layout<BE: poulpy_hal::layouts::Backend>(params: &TestParams, module: &Module<BE>) -> GGSWLayout {
    let base2k: Base2K = params.base2k.into();
    GGSWLayout {
        n: module.n().into(),
        base2k,
        dnum: Dnum(2),
        k_digit: TorusPrecision::from(base2k),
        k_aux: TorusPrecision::from(base2k),
        rank: Rank(1),
    }
}

fn unset_secret<BE: crate::test_suite::noise::TestBackend>(module: &Module<BE>, rank: usize) -> Secret<BE>
where
    BE::OwnedBuf: poulpy_hal::layouts::HostDataMut,
    for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
{
    module.glwe_secret_alloc_from_infos(&GLWESecretLayout {
        n: module.n().into(),
        rank: rank.into(),
    })
}

/// Encrypting an LWE under a secret that was never sampled is refused before any draw or carve.
pub fn test_lwe_encrypt_sk_rejects_unset_secret<BE: crate::test_suite::noise::TestBackend>(
    params: &TestParams,
    module: &Module<BE>,
) where
    BE::OwnedBuf: poulpy_hal::layouts::HostDataMut,
    for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: LWEEncryptSk<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let (n, base2k, k) = (
        Degree(module.n() as u32),
        Base2K(params.base2k as u32),
        TorusPrecision(2 * params.base2k as u32),
    );
    let mut res = module.lwe_alloc(n, base2k, k);
    let pt = module.lwe_plaintext_alloc(base2k, k);
    let sk: LWESecret<BE::OwnedBuf, BE::ZnxWord> = module.lwe_secret_alloc(n);
    assert_unset_secret_refused(
        "lwe secret distribution is NONE (have you prepared the key?)",
        &mut res,
        |res, xe, xa, scratch| module.lwe_encrypt_sk(res, &pt, &sk, xe, xa, scratch),
    );
}

/// Encrypting a GLWE, a zero or a GLWE with a given mask under a secret that was never sampled is refused before any
/// draw or carve.
pub fn test_glwe_encrypt_sk_rejects_unset_secret<BE: crate::test_suite::noise::TestBackend>(
    params: &TestParams,
    module: &Module<BE>,
) where
    BE::OwnedBuf: poulpy_hal::layouts::HostDataMut,
    for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: GLWEEncryptSk<BE> + GLWESecretPreparedFactory<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let layout = glwe_layout(params, module, 1);
    let mut res = module.glwe_alloc_from_infos(&layout);
    let pt = module.glwe_plaintext_alloc_from_infos(&layout);
    let sk: Prepared<BE> = module.glwe_secret_prepared_alloc(Rank(1));
    assert_unset_secret_refused(GLWE_NONE, &mut res, |res, xe, xa, scratch| {
        module.glwe_encrypt_sk(res, &pt, &sk, xe, xa, scratch)
    });
    assert_unset_secret_refused(GLWE_NONE, &mut res, |res, xe, xa, scratch| {
        module.glwe_encrypt_zero_sk(res, &sk, xe, xa, scratch)
    });
    assert_unset_secret_refused(GLWE_NONE, &mut res, |res, xe, _, scratch| {
        module.glwe_encrypt_sk_with_mask(res, &pt, &sk, xe, scratch)
    });
}

/// Encrypting a compressed GLWE or zero under a secret that was never sampled is refused before any draw or carve.
pub fn test_glwe_compressed_encrypt_sk_rejects_unset_secret<BE: crate::test_suite::noise::TestBackend>(
    params: &TestParams,
    module: &Module<BE>,
) where
    BE::OwnedBuf: poulpy_hal::layouts::HostDataMut,
    for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: GLWECompressedEncryptSk<BE> + GLWESecretPreparedFactory<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let layout = glwe_layout(params, module, 1);
    let mut res = module.glwe_compressed_alloc_from_infos(&layout);
    let pt = module.glwe_plaintext_alloc_from_infos(&layout);
    let sk: Prepared<BE> = module.glwe_secret_prepared_alloc(Rank(1));
    assert_unset_secret_refused(GLWE_NONE, &mut res, |res, xe, _, scratch| {
        module.glwe_compressed_encrypt_sk(res, &pt, &sk, [1u8; 32], xe, scratch)
    });
    assert_unset_secret_refused(GLWE_NONE, &mut res, |res, xe, _, scratch| {
        module.glwe_compressed_encrypt_zero_sk(res, &sk, [1u8; 32], xe, scratch)
    });
}

/// Encrypting a compressed GGLWE under a secret that was never sampled is refused before any draw or carve.
pub fn test_gglwe_compressed_encrypt_sk_rejects_unset_secret<BE: crate::test_suite::noise::TestBackend>(
    params: &TestParams,
    module: &Module<BE>,
) where
    BE::OwnedBuf: poulpy_hal::layouts::HostDataMut,
    for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: GGLWECompressedEncryptSk<BE> + GLWESecretPreparedFactory<BE> + ScalarZnxAlloc<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let mut res = module.gglwe_compressed_alloc_from_infos(&gglwe_layout(params, module, 1, 1));
    let pt = module.scalar_znx_alloc(module.n(), 1);
    let sk: Prepared<BE> = module.glwe_secret_prepared_alloc(Rank(1));
    assert_unset_secret_refused(GLWE_NONE, &mut res, |res, xe, _, scratch| {
        module.gglwe_compressed_encrypt_sk(res, &pt, &sk, [1u8; 32], xe, scratch)
    });
}

/// Encrypting a GGSW, plain or compressed, under a secret that was never sampled is refused before any draw or carve.
pub fn test_ggsw_encrypt_sk_rejects_unset_secret<BE: crate::test_suite::noise::TestBackend>(
    params: &TestParams,
    module: &Module<BE>,
) where
    BE::OwnedBuf: poulpy_hal::layouts::HostDataMut,
    for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: GGSWEncryptSk<BE> + GGSWCompressedEncryptSk<BE> + GLWESecretPreparedFactory<BE> + ScalarZnxAlloc<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let layout = ggsw_layout(params, module);
    let pt = module.scalar_znx_alloc(module.n(), 1);
    let sk: Prepared<BE> = module.glwe_secret_prepared_alloc(Rank(1));
    let mut res = module.ggsw_alloc_from_infos(&layout);
    assert_unset_secret_refused(GLWE_NONE, &mut res, |res, xe, xa, scratch| {
        module.ggsw_encrypt_sk(res, &pt, &sk, xe, xa, scratch)
    });
    let mut res = module.ggsw_compressed_alloc_from_infos(&layout);
    assert_unset_secret_refused(GLWE_NONE, &mut res, |res, xe, _, scratch| {
        module.ggsw_compressed_encrypt_sk(res, &pt, &sk, [1u8; 32], xe, scratch)
    });
}

/// Encrypting a switching key, plain or compressed, to an output secret that was never sampled is refused before any
/// draw or carve.
pub fn test_glwe_switching_key_encrypt_sk_rejects_unset_secret<BE: crate::test_suite::noise::TestBackend>(
    params: &TestParams,
    module: &Module<BE>,
) where
    BE::OwnedBuf: poulpy_hal::layouts::HostDataMut,
    for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: GLWESwitchingKeyEncryptSk<BE> + GLWESwitchingKeyCompressedEncryptSk<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let layout = gglwe_layout(params, module, 1, 1);
    let (sk_in, sk_out) = (unset_secret(module, 1), unset_secret(module, 1));
    let mut res = module.glwe_switching_key_alloc_from_infos(&layout);
    assert_unset_secret_refused(GLWE_NONE, &mut res, |res, xe, xa, scratch| {
        module.glwe_switching_key_encrypt_sk(res, &sk_in, &sk_out, xe, xa, scratch)
    });
    let mut res = module.glwe_switching_key_compressed_alloc_from_infos(&layout);
    assert_unset_secret_refused(GLWE_NONE, &mut res, |res, xe, _, scratch| {
        module.glwe_switching_key_compressed_encrypt_sk(res, &sk_in, &sk_out, [1u8; 32], xe, scratch)
    });
}

/// Encrypting an automorphism key, plain or compressed, under a secret that was never sampled is refused before any
/// draw or carve.
pub fn test_glwe_automorphism_key_encrypt_sk_rejects_unset_secret<BE: crate::test_suite::noise::TestBackend>(
    params: &TestParams,
    module: &Module<BE>,
) where
    BE::OwnedBuf: poulpy_hal::layouts::HostDataMut,
    for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: GLWEAutomorphismKeyEncryptSk<BE> + GLWEAutomorphismKeyCompressedEncryptSk<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let layout = gglwe_layout(params, module, 1, 1);
    let sk = unset_secret(module, 1);
    let mut res = module.glwe_automorphism_key_alloc_from_infos(&layout);
    assert_unset_secret_refused(GLWE_NONE, &mut res, |res, xe, xa, scratch| {
        module.glwe_automorphism_key_encrypt_sk(res, 5, &sk, xe, xa, scratch)
    });
    let mut res = module.glwe_automorphism_key_compressed_alloc_from_infos(&layout);
    assert_unset_secret_refused(GLWE_NONE, &mut res, |res, xe, _, scratch| {
        module.glwe_automorphism_key_compressed_encrypt_sk(res, 5, &sk, [1u8; 32], xe, scratch)
    });
}

/// Encrypting a tensor key or a GGLWE-to-GGSW key, plain or compressed, under a secret that was never sampled is
/// refused before any draw or carve.
pub fn test_glwe_tensor_key_encrypt_sk_rejects_unset_secret<BE: crate::test_suite::noise::TestBackend>(
    params: &TestParams,
    module: &Module<BE>,
) where
    BE::OwnedBuf: poulpy_hal::layouts::HostDataMut,
    for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: GLWETensorKeyEncryptSk<BE>
        + GLWETensorKeyCompressedEncryptSk<BE>
        + GGLWEToGGSWKeyEncryptSk<BE>
        + GGLWEToGGSWKeyCompressedEncryptSk<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let sk = unset_secret(module, 1);
    let layout = gglwe_layout(params, module, 1, 1);
    let mut res = module.glwe_tensor_key_alloc_from_infos(&layout);
    assert_unset_secret_refused(GLWE_NONE, &mut res, |res, xe, xa, scratch| {
        module.glwe_tensor_key_encrypt_sk(res, &sk, xe, xa, scratch)
    });
    let mut res = module.glwe_tensor_key_compressed_alloc_from_infos(&layout);
    assert_unset_secret_refused(GLWE_NONE, &mut res, |res, xe, _, scratch| {
        module.glwe_tensor_key_compressed_encrypt_sk(res, &sk, [1u8; 32], xe, scratch)
    });
    let mut res = module.gglwe_to_ggsw_key_alloc_from_infos(&layout);
    assert_unset_secret_refused(GLWE_NONE, &mut res, |res, xe, xa, scratch| {
        module.gglwe_to_ggsw_key_encrypt_sk(res, &sk, xe, xa, scratch)
    });
    let mut res = module.gglwe_to_ggsw_key_compressed_alloc_from_infos(&layout);
    assert_unset_secret_refused(GLWE_NONE, &mut res, |res, xe, _, scratch| {
        module.gglwe_to_ggsw_key_compressed_encrypt_sk(res, &sk, [1u8; 32], xe, scratch)
    });
}

/// Encrypting a GLWE-to-LWE, LWE switching or LWE-to-GLWE key to an output secret that was never sampled is refused
/// before any draw or carve.
pub fn test_lwe_switching_key_encrypt_sk_rejects_unset_secret<BE: crate::test_suite::noise::TestBackend>(
    params: &TestParams,
    module: &Module<BE>,
) where
    BE::OwnedBuf: poulpy_hal::layouts::HostDataMut,
    for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>:
        GLWEToLWEKeyEncryptSk<BE> + LWESwitchingKeyEncryptSk<BE> + LWEToGLWEKeyEncryptSk<BE> + GLWESecretPreparedFactory<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let layout = gglwe_layout(params, module, 1, 1);
    let sk_glwe = unset_secret(module, 1);
    let sk_glwe_prepared: Prepared<BE> = module.glwe_secret_prepared_alloc(Rank(1));
    let n = Degree(module.n() as u32);
    let sk_lwe_in: LWESecret<BE::OwnedBuf, BE::ZnxWord> = module.lwe_secret_alloc(n);
    let sk_lwe_out: LWESecret<BE::OwnedBuf, BE::ZnxWord> = module.lwe_secret_alloc(n);
    let mut res = module.glwe_to_lwe_key_alloc_from_infos(&layout);
    assert_unset_secret_refused(GLWE_NONE, &mut res, |res, xe, xa, scratch| {
        module.glwe_to_lwe_key_encrypt_sk(res, &sk_lwe_out, &sk_glwe, xe, xa, scratch)
    });
    let mut res = module.lwe_switching_key_alloc_from_infos(&layout);
    assert_unset_secret_refused(GLWE_NONE, &mut res, |res, xe, xa, scratch| {
        module.lwe_switching_key_encrypt_sk(res, &sk_lwe_in, &sk_lwe_out, xe, xa, scratch)
    });
    let mut res = module.lwe_to_glwe_key_alloc_from_infos(&layout);
    assert_unset_secret_refused(GLWE_NONE, &mut res, |res, xe, xa, scratch| {
        module.lwe_to_glwe_key_encrypt_sk(res, &sk_lwe_in, &sk_glwe_prepared, xe, xa, scratch)
    });
}
