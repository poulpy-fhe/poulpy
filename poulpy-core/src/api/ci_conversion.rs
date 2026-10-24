use poulpy_hal::{
    layouts::{Backend, Data, ScratchArena},
    source::Source,
};

use crate::{
    GetDistribution, GetDistributionMut,
    layouts::{
        GGLWEInfos, GGLWEToBackendMut, GLWECIEmbedKey, GLWECITraceKey, GLWEInfos, GLWESecretToBackendMut, GLWESecretToBackendRef,
        GLWEToBackendMut, GLWEToBackendRef,
    },
};

/// Embedding of a conjugate-invariant GLWE of degree `N` into the standard ring of
/// degree `2N`, on a standard module of degree at least `2N`.
///
/// The result decrypts under the embedded conjugate-invariant secret
/// ([`GLWESecretCIEmbed`]).
pub trait GLWECIEmbed<BE: Backend> {
    /// Writes `a_0 + Σ a_i (X^i + X^-i)` of every column of `a` into `res`, at
    /// the radix of `a`. The digits are exact but not renormalized, so `res` is
    /// flagged non-canonical.
    fn glwe_ci_embed<R, A>(&self, res: &mut R, a: &A)
    where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        A: GLWEToBackendRef<BE> + GLWEInfos;
}

/// Embedding of a conjugate-invariant secret of degree `N` into the standard secret of degree `2N`,
/// on a standard module of degree at least `2N`.
pub trait GLWESecretCIEmbed<BE: Backend> {
    /// Writes into `res` the embedding of the conjugate-invariant secret `src` of degree `N`:
    /// the standard secret of degree `2N` under which embedded ciphertexts decrypt; the module
    /// degree is at least `2N`.
    ///
    /// `res` is tagged [`Distribution::ENCAPSULATED`](crate::Distribution::ENCAPSULATED):
    /// its coefficients mirror the source, so it only backs the ring-switching keys.
    ///
    /// # Panics
    /// Before writing `res`, when its degree is not `2N` or exceeds the module degree, or its
    /// rank differs from that of `src`.
    fn glwe_secret_ci_embed<R, S>(&self, res: &mut R, src: &S)
    where
        R: GLWESecretToBackendMut<BE> + GetDistributionMut,
        S: GLWESecretToBackendRef<BE>;
}

/// Switch of a GLWE between ring degrees that divide one another, on a module of
/// degree at least the larger, as from `N` to the degree `2N` of the conjugate-invariant maps.
pub trait GLWESwitchDegree<BE: Backend> {
    /// Writes each column of `a` into `res` at the degree of `res`: `X -> X^(N_res / N_a)`
    /// when it is larger, the coefficients at multiples of the ratio when it is smaller.
    /// `res` keeps the canonical flag of `a`.
    fn glwe_switch_degree<R, A>(&self, res: &mut R, a: &A)
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
/// `2N`; `sk_ci` is embedded into the scratch, which is wiped over
/// [`glwe_ci_key_encrypt_sk_tmp_bytes`](Self::glwe_ci_key_encrypt_sk_tmp_bytes) before returning.
#[allow(clippy::too_many_arguments)]
pub trait GLWECIKeyEncryptSk<BE: Backend> {
    /// Scratch bytes of [`Self::glwe_ci_embed_key_encrypt_sk`] and
    /// [`Self::glwe_ci_trace_key_encrypt_sk`] for keys of layout `infos`.
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
