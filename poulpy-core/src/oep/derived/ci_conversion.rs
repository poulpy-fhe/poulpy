//! Degree switch expressed by composing HAL operations, and the conjugate-invariant key
//! encryptions composed from the secret embedding and the selected switching-key encryption.
use poulpy_hal::{
    api::VecZnxSwitchRing,
    layouts::{Backend, Data, Module, ScratchArena},
    source::Source,
};

use crate::{
    GetDistribution,
    layouts::{
        GGLWEInfos, GGLWEToBackendMut, GLWECIEmbedKey, GLWECITraceKey, GLWEInfos, GLWESecretToBackendRef, GLWEToBackendMut,
        GLWEToBackendRef, LWEInfos,
    },
    oep::GLWECIConversionImpl,
};

/// Writes each column of `a` into `res` at the degree of `res`: `X -> X^(N_res / N_a)`
/// when it is larger, the coefficients at multiples of the ratio when it is smaller.
/// `res` keeps the canonical flag of `a`.
pub(crate) fn glwe_switch_degree_derived<BE, M, R, A>(module: &M, res: &mut R, a: &A)
where
    BE: Backend,
    M: VecZnxSwitchRing<BE>,
    R: GLWEToBackendMut<BE> + GLWEInfos,
    A: GLWEToBackendRef<BE> + GLWEInfos,
{
    res.set_canonical(a.is_canonical());
    let mut res = res.to_backend_mut();
    let a = a.to_backend_ref();
    assert_eq!(res.rank(), a.rank());
    assert_eq!(res.base2k(), a.base2k());
    for col in 0..res.rank().as_usize() + 1 {
        module.vec_znx_switch_ring(&mut res.data, col, &a.data, col);
    }
}

pub(crate) fn glwe_ci_key_encrypt_sk_tmp_bytes_derived<BE, A>(module: &Module<BE>, infos: &A) -> usize
where
    BE: GLWECIConversionImpl,
    A: GGLWEInfos,
{
    BE::glwe_switching_key_encrypt_sk_tmp_bytes(module, infos)
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
    let sk_embedded = BE::glwe_secret_ci_embed(module, sk_ci);
    BE::glwe_switching_key_encrypt_sk(module, res, &sk_embedded, sk, source_xe, source_xa, scratch);
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
    let sk_embedded = BE::glwe_secret_ci_embed(module, sk_ci);
    BE::glwe_switching_key_encrypt_sk(module, res, sk, &sk_embedded, source_xe, source_xa, scratch);
}
