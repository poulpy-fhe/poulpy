//! Degree switch expressed by composing HAL operations, and the conjugate-invariant key
//! encryptions composed from the secret embedding and the selected switching-key encryption.
use poulpy_hal::{
    layouts::{Backend, Data, Module, ScratchArena},
    source::Source,
};

use crate::{
    Distribution, GetDistribution, ScratchArenaTakeCore,
    api::GLWEBytesOf,
    layouts::{
        Degree, GGLWEInfos, GGLWEToBackendMut, GLWECIEmbedKey, GLWECITraceKey, GLWEInfos, GLWESecretToBackendRef,
        GLWEToBackendMut, GLWEToBackendRef, LWEInfos, Rank,
    },
    oep::GLWECIConversionImpl,
};

/// Writes each column of `a` into `res` at the degree of `res`: `X -> X^(N_res / N_a)`
/// when it is larger, the coefficients at multiples of the ratio when it is smaller.
/// `res` keeps the canonical flag of `a`.
pub(crate) fn glwe_switch_degree_derived<BE, R, A>(module: &Module<BE>, res: &mut R, a: &A)
where
    BE: GLWECIConversionImpl,
    R: GLWEToBackendMut<BE> + GLWEInfos,
    A: GLWEToBackendRef<BE> + GLWEInfos,
{
    assert_eq!(res.rank(), a.rank());
    assert_eq!(res.base2k(), a.base2k());
    res.set_canonical(a.is_canonical());
    let mut res = res.to_backend_mut();
    let a = a.to_backend_ref();
    for col in 0..res.rank().as_usize() + 1 {
        BE::vec_znx_switch_ring(module, &mut res.data, col, &a.data, col);
    }
}

/// Refuses, before `sk_ci` is embedded into the scratch, what the nested embedding and switching-key encryption refuse.
fn assert_ci_key_secrets<BE, K, S1, S2>(key: &K, sk_ci: &S1, sk: &S2, ci_rank: Rank, sk_rank: Rank)
where
    BE: Backend,
    K: GGLWEInfos,
    S1: GLWESecretToBackendRef<BE>,
    S2: GLWEInfos,
{
    let sk_ci = sk_ci.to_backend_ref();
    assert!(
        2 * sk_ci.n().as_usize() <= key.n().as_usize(),
        "conjugate-invariant key encryption: embedded secret degree exceeds the key's"
    );
    assert!(
        sk_ci.rank() == ci_rank,
        "conjugate-invariant key encryption: conjugate-invariant secret rank differs from the key's"
    );
    assert!(
        sk.n().as_usize() <= key.n().as_usize(),
        "conjugate-invariant key encryption: secret degree exceeds the key's"
    );
    assert!(
        sk.rank() == sk_rank,
        "conjugate-invariant key encryption: secret rank differs from the key's"
    );
}

pub(crate) fn glwe_ci_key_encrypt_sk_tmp_bytes_derived<BE, A>(module: &Module<BE>, infos: &A) -> usize
where
    BE: GLWECIConversionImpl,
    A: GGLWEInfos,
{
    let sk_embedded: usize = module.glwe_secret_bytes_of(infos.n(), infos.rank_in().max(infos.rank_out()));
    BE::scratch_aligned(sk_embedded) + BE::glwe_switching_key_encrypt_sk_tmp_bytes(module, infos)
}

pub(crate) fn glwe_ci_embed_key_encrypt_sk_derived<BE, D, S1, S2>(
    module: &Module<BE>,
    res: &mut GLWECIEmbedKey<D, BE::ZnxWord>,
    sk_ci: &S1,
    sk: &S2,
    source_xe: &mut Source,
    source_xa: &mut Source,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: GLWECIConversionImpl,
    D: Data,
    GLWECIEmbedKey<D, BE::ZnxWord>: GGLWEToBackendMut<BE>,
    S1: GLWESecretToBackendRef<BE>,
    S2: GLWESecretToBackendRef<BE> + GetDistribution + GLWEInfos,
{
    assert_ci_key_secrets::<BE, _, _, _>(res, sk_ci, sk, res.rank_in(), res.rank_out());
    assert!(
        *sk.dist() != Distribution::NONE,
        "conjugate-invariant key encryption: secret distribution is NONE"
    );
    let tmp_bytes: usize = glwe_ci_key_encrypt_sk_tmp_bytes_derived(module, res);
    assert!(
        scratch.available() >= tmp_bytes,
        "insufficient scratch for conjugate-invariant key encryption"
    );
    {
        let sk_ci_ref = sk_ci.to_backend_ref();
        let n: Degree = (2 * sk_ci_ref.n().as_usize()).into();
        let (mut sk_embedded, mut scratch_1) = scratch.borrow().take_glwe_secret_scratch(n, sk_ci_ref.rank());
        BE::glwe_secret_ci_embed(module, &mut sk_embedded, sk_ci);
        BE::glwe_switching_key_encrypt_sk(module, res, &sk_embedded, sk, source_xe, source_xa, &mut scratch_1);
    }
    scratch.wipe(tmp_bytes);
}

pub(crate) fn glwe_ci_trace_key_encrypt_sk_derived<BE, D, S1, S2>(
    module: &Module<BE>,
    res: &mut GLWECITraceKey<D, BE::ZnxWord>,
    sk_ci: &S1,
    sk: &S2,
    source_xe: &mut Source,
    source_xa: &mut Source,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: GLWECIConversionImpl,
    D: Data,
    GLWECITraceKey<D, BE::ZnxWord>: GGLWEToBackendMut<BE>,
    S1: GLWESecretToBackendRef<BE>,
    S2: GLWESecretToBackendRef<BE> + GetDistribution + GLWEInfos,
{
    assert_ci_key_secrets::<BE, _, _, _>(res, sk_ci, sk, res.rank_out(), res.rank_in());
    let tmp_bytes: usize = glwe_ci_key_encrypt_sk_tmp_bytes_derived(module, res);
    assert!(
        scratch.available() >= tmp_bytes,
        "insufficient scratch for conjugate-invariant key encryption"
    );
    {
        let sk_ci_ref = sk_ci.to_backend_ref();
        let n: Degree = (2 * sk_ci_ref.n().as_usize()).into();
        let (mut sk_embedded, mut scratch_1) = scratch.borrow().take_glwe_secret_scratch(n, sk_ci_ref.rank());
        BE::glwe_secret_ci_embed(module, &mut sk_embedded, sk_ci);
        BE::glwe_switching_key_encrypt_sk(module, res, sk, &sk_embedded, source_xe, source_xa, &mut scratch_1);
    }
    scratch.wipe(tmp_bytes);
}
