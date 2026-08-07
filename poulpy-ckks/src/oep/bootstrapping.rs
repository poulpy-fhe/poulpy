use crate::CKKSResult as Result;
use crate::default::bootstrapping::CKKSBootstrappingOpsDefault;

use poulpy_core::{
    GLWECopy, GLWEKeyswitch, GLWEShift,
    layouts::{BSGSMeta, GGLWEInfos, GLWEInfos, GLWETensorKeyPrepared, SetBSGSMeta, prepared::GLWETensorKeyPreparedToBackendRef},
};
use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use crate::{
    CKKSCtBounds, CKKSInfos, GLWEToBackendMut, GLWEToBackendRef, SetCKKSInfos,
    api::{
        CKKSAddOps, CKKSAffineOps, CKKSAllOpsTmpBytes, CKKSConjugateOps, CKKSCopyOps, CKKSDFTOps, CKKSEvalModOps, CKKSImagOps,
        CKKSMulOps, CKKSPolynomialEvaluationOps, CKKSPow2Ops, CKKSSubOps,
    },
    layouts::{BootstrappingContext, BootstrappingKeys, BootstrappingKeysLayout, CKKSCiphertext, CKKSPlaintext, EncodedLut},
};

/// Backend override hook for [`CKKSBootstrappingOps`](crate::api::CKKSBootstrappingOps).
///
/// The blanket impl below forwards to the backend-generic reference in
/// [`CKKSBootstrappingOpsDefault`]; a backend may instead provide a specialized
/// implementation by implementing this trait directly.
///
/// # Safety
///
/// Implementations must satisfy the contracts of all trait methods, including
/// any HAL-level invariants (alignment, layout, scratch sizing) implied by the
/// associated method signatures.
pub unsafe trait CKKSBootstrappingImpl<BE: Backend>: Backend {
    fn ckks_mod_up_tmp_bytes_impl(module: &Module<BE>) -> usize;

    /// See [`CKKSBootstrappingOps::ckks_bootstrap_tmp_bytes`](crate::api::CKKSBootstrappingOps::ckks_bootstrap_tmp_bytes).
    fn ckks_bootstrap_tmp_bytes_impl<C1, C2, F>(
        module: &Module<BE>,
        ct_out: &C1,
        ct_in: &C2,
        ctx: &BootstrappingContext<BE, F>,
        keys_layout: &BootstrappingKeysLayout,
    ) -> usize
    where
        C1: CKKSCtBounds,
        C2: CKKSCtBounds;

    fn ckks_functional_bootstrap_tmp_bytes_impl<C1, C2, F>(
        module: &Module<BE>,
        ct_out: &C1,
        ct_in: &C2,
        ctx: &BootstrappingContext<BE, F>,
        lut: &EncodedLut<CKKSPlaintext<BE::OwnedBuf>>,
        keys_layout: &BootstrappingKeysLayout,
    ) -> usize
    where
        C1: CKKSCtBounds,
        C2: CKKSCtBounds;

    fn ckks_mod_up_into_impl<Dst, Src>(
        module: &Module<BE>,
        dst: &mut Dst,
        src: &Src,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSInfos + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + GLWEInfos + CKKSInfos;

    /// See [`CKKSBootstrappingOps::ckks_bootstrap`](crate::api::CKKSBootstrappingOps::ckks_bootstrap).
    fn ckks_bootstrap_impl<F, K>(
        module: &Module<BE>,
        ct_out: &mut CKKSCiphertext<BE::OwnedBuf>,
        ct_in: &CKKSCiphertext<BE::OwnedBuf>,
        ctx: &BootstrappingContext<BE, F>,
        keys: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        K: BootstrappingKeys<BE, TensorKey = GLWETensorKeyPrepared<BE::OwnedBuf, BE>>;

    fn ckks_bootstrap_real<F, K>(
        module: &Module<BE>,
        ct_out: &mut CKKSCiphertext<BE::OwnedBuf>,
        ct_in: &CKKSCiphertext<BE::OwnedBuf>,
        ctx: &BootstrappingContext<BE, F>,
        keys: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        K: BootstrappingKeys<BE, TensorKey = GLWETensorKeyPrepared<BE::OwnedBuf, BE>>;

    #[allow(clippy::too_many_arguments)]
    fn ckks_functional_bootstrap<F, K>(
        module: &Module<BE>,
        ct_out: &mut CKKSCiphertext<BE::OwnedBuf>,
        ct_in: &CKKSCiphertext<BE::OwnedBuf>,
        ctx: &BootstrappingContext<BE, F>,
        lut: &EncodedLut<CKKSPlaintext<BE::OwnedBuf>>,
        keys: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        K: BootstrappingKeys<BE, TensorKey = GLWETensorKeyPrepared<BE::OwnedBuf, BE>>;

    #[allow(clippy::too_many_arguments)]
    fn ckks_functional_bootstrap_real<F, K>(
        module: &Module<BE>,
        ct_out: &mut CKKSCiphertext<BE::OwnedBuf>,
        ct_in: &CKKSCiphertext<BE::OwnedBuf>,
        ctx: &BootstrappingContext<BE, F>,
        lut: &EncodedLut<CKKSPlaintext<BE::OwnedBuf>>,
        keys: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        K: BootstrappingKeys<BE, TensorKey = GLWETensorKeyPrepared<BE::OwnedBuf, BE>>;

    #[allow(clippy::too_many_arguments)]
    fn ckks_functional_bootstrap_multi<F, K>(
        module: &Module<BE>,
        ct_outs: &mut [CKKSCiphertext<BE::OwnedBuf>],
        ct_in: &CKKSCiphertext<BE::OwnedBuf>,
        ctx: &BootstrappingContext<BE, F>,
        luts: &[EncodedLut<CKKSPlaintext<BE::OwnedBuf>>],
        keys: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        K: BootstrappingKeys<BE, TensorKey = GLWETensorKeyPrepared<BE::OwnedBuf, BE>>;
}

unsafe impl<BE: Backend> CKKSBootstrappingImpl<BE> for BE
where
    Module<BE>: CKKSBootstrappingOpsDefault<BE>
        + GLWECopy<BE>
        + GLWEShift<BE>
        + GLWEKeyswitch<BE>
        + CKKSCopyOps<BE>
        + CKKSPow2Ops<BE>
        + CKKSAddOps<BE>
        + CKKSSubOps<BE>
        + CKKSConjugateOps<BE>
        + CKKSImagOps<BE>
        + CKKSDFTOps<BE>
        + CKKSEvalModOps<BE>
        + CKKSAllOpsTmpBytes<BE>
        + CKKSMulOps<BE>
        + CKKSAffineOps<BE>
        + CKKSPolynomialEvaluationOps<BE>,
    CKKSCiphertext<BE::OwnedBuf>:
        GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta + BSGSMeta,
    GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEInfos,
{
    fn ckks_mod_up_tmp_bytes_impl(module: &Module<BE>) -> usize {
        module.ckks_mod_up_tmp_bytes_default()
    }

    fn ckks_bootstrap_tmp_bytes_impl<C1, C2, F>(
        module: &Module<BE>,
        ct_out: &C1,
        ct_in: &C2,
        ctx: &BootstrappingContext<BE, F>,
        keys_layout: &BootstrappingKeysLayout,
    ) -> usize
    where
        C1: CKKSCtBounds,
        C2: CKKSCtBounds,
    {
        module.ckks_bootstrap_tmp_bytes_default(ct_out, ct_in, ctx, keys_layout)
    }

    fn ckks_functional_bootstrap_tmp_bytes_impl<C1, C2, F>(
        module: &Module<BE>,
        ct_out: &C1,
        ct_in: &C2,
        ctx: &BootstrappingContext<BE, F>,
        lut: &EncodedLut<CKKSPlaintext<BE::OwnedBuf>>,
        keys_layout: &BootstrappingKeysLayout,
    ) -> usize
    where
        C1: CKKSCtBounds,
        C2: CKKSCtBounds,
    {
        module.ckks_functional_bootstrap_tmp_bytes_default(ct_out, ct_in, ctx, lut, keys_layout)
    }

    fn ckks_mod_up_into_impl<Dst, Src>(
        module: &Module<BE>,
        dst: &mut Dst,
        src: &Src,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSInfos + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + GLWEInfos + CKKSInfos,
    {
        module.ckks_mod_up_into_default(dst, src, scratch)
    }

    fn ckks_bootstrap_impl<F, K>(
        module: &Module<BE>,
        ct_out: &mut CKKSCiphertext<BE::OwnedBuf>,
        ct_in: &CKKSCiphertext<BE::OwnedBuf>,
        ctx: &BootstrappingContext<BE, F>,
        keys: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        K: BootstrappingKeys<BE, TensorKey = GLWETensorKeyPrepared<BE::OwnedBuf, BE>>,
    {
        module.ckks_bootstrap_default(ct_out, ct_in, ctx, keys, scratch)
    }

    fn ckks_bootstrap_real<F, K>(
        module: &Module<BE>,
        ct_out: &mut CKKSCiphertext<BE::OwnedBuf>,
        ct_in: &CKKSCiphertext<BE::OwnedBuf>,
        ctx: &BootstrappingContext<BE, F>,
        keys: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        K: BootstrappingKeys<BE, TensorKey = GLWETensorKeyPrepared<BE::OwnedBuf, BE>>,
    {
        module.ckks_bootstrap_real_default(ct_out, ct_in, ctx, keys, scratch)
    }

    fn ckks_functional_bootstrap<F, K>(
        module: &Module<BE>,
        ct_out: &mut CKKSCiphertext<BE::OwnedBuf>,
        ct_in: &CKKSCiphertext<BE::OwnedBuf>,
        ctx: &BootstrappingContext<BE, F>,
        lut: &EncodedLut<CKKSPlaintext<BE::OwnedBuf>>,
        keys: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        K: BootstrappingKeys<BE, TensorKey = GLWETensorKeyPrepared<BE::OwnedBuf, BE>>,
    {
        crate::default::bootstrapping::ckks_functional_bootstrap_default(module, ct_out, ct_in, ctx, lut, keys, scratch)
    }

    fn ckks_functional_bootstrap_real<F, K>(
        module: &Module<BE>,
        ct_out: &mut CKKSCiphertext<BE::OwnedBuf>,
        ct_in: &CKKSCiphertext<BE::OwnedBuf>,
        ctx: &BootstrappingContext<BE, F>,
        lut: &EncodedLut<CKKSPlaintext<BE::OwnedBuf>>,
        keys: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        K: BootstrappingKeys<BE, TensorKey = GLWETensorKeyPrepared<BE::OwnedBuf, BE>>,
    {
        crate::default::bootstrapping::ckks_functional_bootstrap_real_default(module, ct_out, ct_in, ctx, lut, keys, scratch)
    }

    fn ckks_functional_bootstrap_multi<F, K>(
        module: &Module<BE>,
        ct_outs: &mut [CKKSCiphertext<BE::OwnedBuf>],
        ct_in: &CKKSCiphertext<BE::OwnedBuf>,
        ctx: &BootstrappingContext<BE, F>,
        luts: &[EncodedLut<CKKSPlaintext<BE::OwnedBuf>>],
        keys: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        K: BootstrappingKeys<BE, TensorKey = GLWETensorKeyPrepared<BE::OwnedBuf, BE>>,
    {
        crate::default::bootstrapping::ckks_functional_bootstrap_multi_default(module, ct_outs, ct_in, ctx, luts, keys, scratch)
    }
}

#[macro_export]
macro_rules! impl_ckks_bootstrapping_defaults {
    ($be:ty) => {
        impl $crate::default::bootstrapping::CKKSBootstrappingOpsDefault<$be> for ::poulpy_hal::layouts::Module<$be> {}
    };
}
pub use crate::impl_ckks_bootstrapping_defaults;
