//! Reference implementation of the SHIP coefficient encoding
//! ([`CKKSShipCoeffEncodingImpl`](poulpy_ckks::oep::CKKSShipCoeffEncodingImpl)).
//!
//! The scheme-defining transformation lives in `poulpy-ckks`
//! ([`ship_coeff_encodings_host`]); this module only reads generic inputs
//! (e.g. scratch-carved views) in place through their host-addressable backend
//! views. Invoke [`impl_ckks_ship_coeff_encoding`](crate::impl_ckks_ship_coeff_encoding)
//! after installing the backend's CKKS encoding implementation; a backend
//! with a fused native kernel implements the OEP directly instead.

use anyhow::{Result, ensure};
use poulpy_ckks::{
    CKKSCtBounds,
    api::{CKKSEncodingOps, ShipScalar},
    encoding::ship_coeff_encodings_host,
    layouts::{CKKSModuleAlloc, ShipCoeffEncodings, ShipPlan},
    oep::CKKSEncodingImpl,
};
use poulpy_core::layouts::{Base2K, GLWEToBackendRef};
use poulpy_hal::{
    api::ModuleN,
    layouts::{Backend, HostDataRef, Module, Standard},
};

/// Runs the scheme-defining [`ship_coeff_encodings_host`] on the one-limb
/// bottom ciphertext `ct`, owned or a view such as a scratch carve, read in
/// place.
pub fn ship_coeff_encodings_staged<BE, F, Src>(
    module: &Module<BE>,
    base2k: Base2K,
    complex: bool,
    ct: &Src,
    plan: &ShipPlan,
) -> Result<ShipCoeffEncodings<BE::OwnedBuf, BE::ZnxWord>>
where
    BE: Backend<ZnxWord = i64, Ring = Standard> + CKKSEncodingImpl<F>,
    for<'x> BE::BufRef<'x>: HostDataRef,
    Module<BE>: ModuleN + CKKSModuleAlloc<BE> + CKKSEncodingOps<BE, F>,
    F: ShipScalar,
    Src: GLWEToBackendRef<BE> + CKKSCtBounds,
{
    ensure!(
        module.n() == plan.n(),
        "SHIP module degree {} does not match plan degree {}",
        module.n(),
        plan.n()
    );
    ensure!(
        ct.log_delta() <= ct.k().as_usize(),
        "SHIP ciphertext scale {} exceeds its width {}",
        ct.log_delta(),
        ct.k()
    );
    ship_coeff_encodings_host::<BE, _, F>(module, base2k, complex, &ct.to_backend_ref(), plan)
}

/// Opts a CPU backend into the reference SHIP coefficient encoding. Requires
/// the backend's CKKS encoding implementation.
#[macro_export]
macro_rules! impl_ckks_ship_coeff_encoding {
    ($be:ty) => {
        unsafe impl ::poulpy_ckks::oep::CKKSShipCoeffEncodingImpl for $be {
            fn ckks_ship_coeff_encodings_tmp_bytes<F>(
                _module: &::poulpy_hal::layouts::Module<$be>,
                _base2k: ::poulpy_core::layouts::Base2K,
                _complex: bool,
                _plan: &::poulpy_ckks::layouts::ShipPlan,
            ) -> ::poulpy_ckks::CKKSResult<usize>
            where
                F: ::poulpy_ckks::api::ShipScalar,
                $be: ::poulpy_ckks::oep::CKKSEncodingImpl<F>,
            {
                // The reference path uses ordinary host vectors. Native
                // device implementations report their arena workspace here.
                Ok(0)
            }

            fn ckks_ship_coeff_encodings<F, Src>(
                module: &::poulpy_hal::layouts::Module<$be>,
                base2k: ::poulpy_core::layouts::Base2K,
                complex: bool,
                ciphertext: &Src,
                plan: &::poulpy_ckks::layouts::ShipPlan,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
            ) -> ::poulpy_ckks::CKKSResult<
                ::poulpy_ckks::layouts::ShipCoeffEncodings<
                    <$be as ::poulpy_hal::layouts::Backend>::OwnedBuf,
                    <$be as ::poulpy_hal::layouts::Backend>::ZnxWord,
                >,
            >
            where
                F: ::poulpy_ckks::api::ShipScalar,
                $be: ::poulpy_ckks::oep::CKKSEncodingImpl<F>,
                Src: ::poulpy_core::layouts::GLWEToBackendRef<$be> + ::poulpy_ckks::CKKSCtBounds,
            {
                let _ = scratch;
                $crate::ckks_ship::ship_coeff_encodings_staged::<$be, F, Src>(module, base2k, complex, ciphertext, plan)
                    .map_err(::poulpy_ckks::CKKSError::from)
            }
        }
    };
}
