use poulpy_hal::{
    layouts::{Backend, Data, ScratchArena},
    source::Source,
};

use crate::{
    GetDistribution,
    layouts::{
        GGLWEInfos, GGLWEToBackendMut, GLWECIEmbedKey, GLWECITraceKey, GLWEInfos, GLWESecretToBackendRef, GLWEToBackendMut,
        GLWEToBackendRef,
    },
};

/// Embedding of a conjugate-invariant GLWE of degree `N` into the standard ring of
/// degree `2N`, on a standard module of degree at least `2N`.
///
/// The result decrypts under the embedded conjugate-invariant secret
/// ([`GLWESecretCIEmbed`](crate::layouts::GLWESecretCIEmbed)).
pub trait GLWECIEmbed<BE: Backend> {
    /// Writes `a_0 + Σ a_i (X^i + X^-i)` of every column of `a` into `res`, at
    /// the radix of `a`. The digits are exact but not renormalized, so `res` is
    /// flagged non-canonical.
    fn glwe_ci_embed<R, A>(&self, res: &mut R, a: &A)
    where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        A: GLWEToBackendRef<BE> + GLWEInfos;
}

/// Relative trace of a standard GLWE of degree `2N` onto the conjugate-invariant ring of
/// degree `N`, on a standard module of degree at least `2N`.
///
/// `a` must decrypt under the embedded conjugate-invariant secret; the result
/// decrypts under the conjugate-invariant secret to `m(X) + m(X^-1)`.
pub trait GLWECITrace<BE: Backend> {
    fn glwe_ci_trace_tmp_bytes<R, A>(&self, res_infos: &R, a_infos: &A) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos;

    /// Writes the compressed `a(X) + a(X^-1)` of every column of `a` into `res`,
    /// normalized to the layout of `res`.
    fn glwe_ci_trace<R, A>(&self, res: &mut R, a: &A, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        A: GLWEToBackendRef<BE> + GLWEInfos;
}

/// Encryption of the switching keys between the embedding of a conjugate-invariant secret `sk_ci`
/// of degree `N` and a standard secret `sk` of degree `2N`, on a standard module of degree at least
/// `2N`; `sk_ci` is embedded internally.
#[allow(clippy::too_many_arguments)]
pub trait GLWECIKeyEncryptSk<BE: Backend> {
    fn glwe_ci_key_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos;

    /// Encrypts the switch from the embedded `sk_ci` to `sk`.
    fn glwe_ci_embed_key_encrypt_sk<D, S1, S2>(
        &self,
        res: &mut GLWECIEmbedKey<D, BE::ZnxWord>,
        sk_ci: &S1,
        sk: &S2,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        D: Data,
        GLWECIEmbedKey<D, BE::ZnxWord>: GGLWEToBackendMut<BE>,
        S1: GLWESecretToBackendRef<BE>,
        S2: GLWESecretToBackendRef<BE> + GetDistribution + GLWEInfos;

    /// Encrypts the switch from `sk` to the embedded `sk_ci`.
    fn glwe_ci_trace_key_encrypt_sk<D, S1, S2>(
        &self,
        res: &mut GLWECITraceKey<D, BE::ZnxWord>,
        sk_ci: &S1,
        sk: &S2,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        D: Data,
        GLWECITraceKey<D, BE::ZnxWord>: GGLWEToBackendMut<BE>,
        S1: GLWESecretToBackendRef<BE>,
        S2: GLWESecretToBackendRef<BE> + GetDistribution + GLWEInfos;
}
