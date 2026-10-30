//! Reference implementation of the PaCo coefficient encoding
//! ([`CKKSPaCoCoeffEncodingImpl`](poulpy_ckks::oep::CKKSPaCoCoeffEncodingImpl)).
//!
//! The scheme-defining ciphertext → four-plaintexts transformation lives in
//! `poulpy-ckks` ([`paco_coeff_encodings_host`]) and dispatches all value-level
//! work through the backend's CKKS encoding operations
//! ([`CKKSEncodingImpl`]); precomputed
//! encoding material is the module's own memoized state. This module only
//! reads generic inputs (e.g. scratch-carved views) in place through their
//! host-addressable backend views. Invoke
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
use poulpy_core::layouts::{Base2K, GLWEToBackendRef};
use poulpy_hal::{
    api::ModuleN,
    layouts::{Backend, HostDataRef, Module, Standard},
};

/// Runs the scheme-defining [`paco_coeff_encodings_host`] with the prepared
/// encoding material on the small modulus-q ciphertext `ct`, owned or a view
/// such as a scratch carve, read in place.
///
/// Returns an error for incompatible degree/rank/radix metadata, a scale
/// above the ciphertext's width, non-representable coefficients, or a failed
/// encoding.
pub fn paco_coeff_encodings_staged<BE, F, Src>(
    module: &Module<BE>,
    base2k: Base2K,
    ct: &Src,
    plan: &PaCoPlan,
) -> Result<[CKKSPlaintextOwned<BE>; 4]>
where
    BE: Backend<ZnxWord = i64, Ring = Standard> + CKKSEncodingImpl<F>,
    for<'x> BE::BufRef<'x>: HostDataRef,
    Module<BE>: ModuleN + CKKSModuleAlloc<BE> + CKKSEncodingOps<BE, F>,
    F: PaCoScalar,
    Src: GLWEToBackendRef<BE> + CKKSCtBounds,
{
    ensure!(
        module.n() == plan.n(),
        "PaCo module degree {} does not match plan degree {}",
        module.n(),
        plan.n()
    );
    ensure!(
        ct.log_delta() <= ct.k().as_usize(),
        "PaCo ciphertext scale {} exceeds its width {}",
        ct.log_delta(),
        ct.k()
    );
    paco_coeff_encodings_host::<BE, _, F>(module, base2k, &ct.to_backend_ref(), plan)
}

/// Opts a CPU backend into the reference PaCo coefficient encoding: the
/// staged scheme routine [`paco_coeff_encodings_staged`] over the backend's
/// CKKS encoding operations. Requires the backend's
/// CKKS encoding implementation.
#[macro_export]
macro_rules! impl_ckks_paco_coeff_encoding {
    ($be:ty) => {
        unsafe impl ::poulpy_ckks::oep::CKKSPaCoCoeffEncodingImpl for $be {
            fn ckks_paco_coeff_encodings_tmp_bytes<F>(
                _module: &::poulpy_hal::layouts::Module<$be>,
                _context: &::poulpy_ckks::layouts::PaCoContext<$be, F>,
            ) -> ::poulpy_ckks::CKKSResult<usize>
            where
                F: ::poulpy_ckks::api::PaCoScalar,
                $be: ::poulpy_ckks::oep::CKKSEncodingImpl<F>,
            {
                // The reference path uses ordinary host vectors. Native
                // device implementations report their arena workspace here.
                Ok(0)
            }

            fn ckks_paco_coeff_encodings<F, Src>(
                module: &::poulpy_hal::layouts::Module<$be>,
                ciphertext: &Src,
                context: &::poulpy_ckks::layouts::PaCoContext<$be, F>,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
            ) -> ::poulpy_ckks::CKKSResult<[::poulpy_ckks::layouts::CKKSPlaintextOwned<$be>; 4]>
            where
                F: ::poulpy_ckks::api::PaCoScalar,
                $be: ::poulpy_ckks::oep::CKKSEncodingImpl<F>,
                Src: ::poulpy_core::layouts::GLWEToBackendRef<$be> + ::poulpy_ckks::CKKSCtBounds,
            {
                let _ = scratch;
                $crate::ckks_paco::paco_coeff_encodings_staged::<$be, F, Src>(
                    module,
                    context.base2k(),
                    ciphertext,
                    context.plan(),
                )
                .map_err(::poulpy_ckks::CKKSError::from)
            }
        }
    };
}
