//! Reference implementation of the PaCo coefficient encoding
//! ([`CKKSPaCoCoeffEncodingImpl`](poulpy_ckks::oep::CKKSPaCoCoeffEncodingImpl)).
//!
//! The scheme-defining ciphertext → four-plaintexts transformation lives in
//! `poulpy-ckks` ([`paco_coeff_encodings_host`]) and dispatches all value-level
//! work through the backend's CKKS encoding operations
//! ([`CKKSEncodingImpl`]); precomputed
//! encoding material is the module's own memoized state. This module only
//! stages generic inputs (e.g. scratch-carved views) into an owned buffer
//! before their residues are read. Invoke
//! [`impl_ckks_paco_coeff_encoding`](crate::impl_ckks_paco_coeff_encoding)
//! after installing the backend's CKKS encoding implementation; a backend
//! with a fused native kernel implements the OEP directly instead.

use anyhow::{Result, ensure};
use poulpy_ckks::{
    CKKSCtBounds,
    api::{CKKSEncodingOps, PaCoScalar},
    encoding::paco_coeff_encodings_host,
    layouts::{CKKSModuleAlloc, CKKSPlaintextOwned, PaCoPlan},
    oep::CKKSEncodingImpl,
};
use poulpy_core::{
    GLWECopy,
    layouts::{Base2K, GLWEToBackendRef},
};
use poulpy_hal::{
    api::ModuleN,
    layouts::{Backend, HostDataRef, Module},
};

/// Stages a generic input (e.g. a scratch-carved view) into one owned copy of
/// the small modulus-q ciphertext, then runs the scheme-defining
/// [`paco_coeff_encodings_host`] with the prepared encoding material.
///
/// Returns an error for incompatible degree/rank/radix metadata,
/// non-representable coefficients, or a failed encoding.
pub fn paco_coeff_encodings_staged<BE, F, Src>(
    module: &Module<BE>,
    ct: &Src,
    plan: &PaCoPlan,
    base2k: Base2K,
) -> Result<[CKKSPlaintextOwned<BE>; 4]>
where
    BE: Backend<ZnxWord = i64> + CKKSEncodingImpl<BE, F>,
    BE::OwnedBuf: HostDataRef,
    Module<BE>: ModuleN + CKKSModuleAlloc<BE> + CKKSEncodingOps<BE, F> + GLWECopy<BE>,
    F: PaCoScalar,
    Src: GLWEToBackendRef<BE> + CKKSCtBounds,
{
    ensure!(
        module.n() == plan.n(),
        "PaCo module degree {} does not match plan degree {}",
        module.n(),
        plan.n()
    );
    let mut owned = module.ckks_ciphertext_alloc_from_infos(ct);
    module.glwe_copy(&mut owned, ct);
    owned.set_meta_checked(ct.meta())?;
    paco_coeff_encodings_host::<BE, _, F>(module, &owned, plan, base2k)
}

/// Opts a CPU backend into the reference PaCo coefficient encoding: the
/// staged scheme routine [`paco_coeff_encodings_staged`] over the backend's
/// CKKS encoding operations. Requires the backend's
/// CKKS encoding implementation.
#[macro_export]
macro_rules! impl_ckks_paco_coeff_encoding {
    ($be:ty) => {
        unsafe impl ::poulpy_ckks::oep::CKKSPaCoCoeffEncodingImpl<$be> for $be {
            fn ckks_paco_coeff_encodings_tmp_bytes_impl<F>(
                _module: &::poulpy_hal::layouts::Module<$be>,
                _plan: &::poulpy_ckks::layouts::PaCoPlan,
            ) -> ::poulpy_ckks::CKKSResult<usize>
            where
                F: ::poulpy_ckks::api::PaCoScalar,
                $be: ::poulpy_ckks::oep::CKKSEncodingImpl<$be, F>,
            {
                // The reference path uses ordinary host vectors. Native
                // device implementations report their arena workspace here.
                Ok(0)
            }

            fn ckks_paco_coeff_encodings_impl<F, Src>(
                module: &::poulpy_hal::layouts::Module<$be>,
                ct: &Src,
                plan: &::poulpy_ckks::layouts::PaCoPlan,
                base2k: ::poulpy_core::layouts::Base2K,
                _scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
            ) -> ::poulpy_ckks::CKKSResult<
                [::poulpy_ckks::layouts::CKKSPlaintext<
                    <$be as ::poulpy_hal::layouts::Backend>::OwnedBuf,
                    <$be as ::poulpy_hal::layouts::Backend>::ZnxWord,
                >; 4],
            >
            where
                F: ::poulpy_ckks::api::PaCoScalar,
                $be: ::poulpy_ckks::oep::CKKSEncodingImpl<$be, F>,
                Src: ::poulpy_core::layouts::GLWEToBackendRef<$be> + ::poulpy_ckks::CKKSCtBounds,
            {
                $crate::ckks_paco::paco_coeff_encodings_staged::<$be, F, Src>(module, ct, plan, base2k)
                    .map_err(::poulpy_ckks::CKKSError::from)
            }
        }
    };
}
