use crate::CKKSResult as Result;
use poulpy_core::layouts::{
    GGLWEInfos, GGLWEPreparedToBackendRef, GLWE, GLWECIEmbedKeyPrepared, GLWECITraceKeyPrepared, GLWEInfos, GLWEToBackendMut,
    GLWEToBackendRef,
};
use poulpy_hal::layouts::{Backend, ConjugateInvariant, Data, ScratchArena};

use crate::{CKKSCtBounds, SetCKKSInfos, layouts::CKKSCiphertext};

/// The embedding of conjugate-invariant ciphertexts of degree `N` into standard
/// ciphertexts of degree `2N` and the relative trace back, on a standard module
/// of degree `2N`.
///
/// Each map switches between the conjugate-invariant secret and a standard secret
/// with its key, [`GLWECIEmbedKey`](poulpy_core::layouts::GLWECIEmbedKey) or
/// [`GLWECITraceKey`](poulpy_core::layouts::GLWECITraceKey), encrypted by
/// [`GLWECIKeyEncryptSk`](poulpy_core::GLWECIKeyEncryptSk). Not implemented on
/// conjugate-invariant backends.
///
/// # Metadata
///
/// ```text
/// embed: log_delta_out = src.log_delta, k_out = src.k
/// trace: log_delta_out = src.log_delta, k_out = src.k − 1   (dst holds Re(m))
/// ```
///
/// The trace doubles the real part and halves it by normalizing at `k − 1`, which
/// spends one bit of budget. Both keep the sparsity, set the slot kind to
/// `SlotsKind::Real` and leave canonical outputs.
pub trait CKKSCIRingMapOps<BE: Backend> {
    fn ckks_ci_embed_tmp_bytes<R, K>(&self, res_infos: &R, key_infos: &K) -> usize
    where
        R: GLWEInfos,
        K: GGLWEInfos;

    /// Embeds `src` into `dst`, at the radix of `src`, and switches it to the standard secret.
    fn ckks_ci_embed<Dst, D, K>(
        &self,
        dst: &mut Dst,
        src: &CKKSCiphertext<D, BE::ZnxWord, ConjugateInvariant>,
        key: &GLWECIEmbedKeyPrepared<K, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        D: Data,
        GLWE<D, BE::ZnxWord>: GLWEToBackendRef<BE>,
        K: Data,
        GLWECIEmbedKeyPrepared<K, BE>: GGLWEPreparedToBackendRef<BE>;

    fn ckks_ci_trace_tmp_bytes<R, A, K>(&self, res_infos: &R, a_infos: &A, key_infos: &K) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        K: GGLWEInfos;

    /// Switches `src` to the embedded conjugate-invariant secret and writes its relative trace into `dst`.
    fn ckks_ci_trace<D, Src, K>(
        &self,
        dst: &mut CKKSCiphertext<D, BE::ZnxWord, ConjugateInvariant>,
        src: &Src,
        key: &GLWECITraceKeyPrepared<K, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        D: Data,
        GLWE<D, BE::ZnxWord>: GLWEToBackendMut<BE>,
        Src: GLWEToBackendRef<BE> + CKKSCtBounds,
        K: Data,
        GLWECITraceKeyPrepared<K, BE>: GGLWEPreparedToBackendRef<BE>;
}
