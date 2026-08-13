//! Reference implementation of the SHIP coefficient encoding
//! ([`CKKSShipCoeffEncodingImpl`](poulpy_ckks::oep::CKKSShipCoeffEncodingImpl)).
//!
//! The scheme-defining transformation lives in `poulpy-ckks`
//! ([`ship_coeff_encodings_host`]); this module only stages generic inputs
//! (e.g. scratch-carved views) into an owned buffer before their residues are
//! read. Invoke [`impl_ckks_ship_coeff_encoding`](crate::impl_ckks_ship_coeff_encoding)
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
use poulpy_core::{
    GLWECopy,
    layouts::{Base2K, GLWEToBackendRef},
};
use poulpy_hal::{
    api::ModuleN,
    layouts::{Backend, HostDataRef, Module},
};

/// Stages a generic input (e.g. a scratch-carved view) into one owned copy of
/// the one-limb bottom ciphertext, then runs the scheme-defining
/// [`ship_coeff_encodings_host`].
pub fn ship_coeff_encodings_staged<BE, F, Src>(
    module: &Module<BE>,
    ct: &Src,
    plan: &ShipPlan,
    base2k: Base2K,
    complex: bool,
) -> Result<ShipCoeffEncodings<BE::OwnedBuf, BE::ZnxWord>>
where
    BE: Backend<ZnxWord = i64> + CKKSEncodingImpl<BE, F>,
    BE::OwnedBuf: HostDataRef,
    Module<BE>: ModuleN + CKKSModuleAlloc<BE> + CKKSEncodingOps<BE, F> + GLWECopy<BE>,
    F: ShipScalar,
    Src: GLWEToBackendRef<BE> + CKKSCtBounds,
{
    ensure!(
        module.n() == plan.n(),
        "SHIP module degree {} does not match plan degree {}",
        module.n(),
        plan.n()
    );
    let mut owned = module.ckks_ciphertext_alloc_from_infos(ct);
    module.glwe_copy(&mut owned, ct);
    owned.set_meta_checked(ct.meta())?;
    ship_coeff_encodings_host::<BE, _, F>(module, &owned, plan, base2k, complex)
}

/// Opts a CPU backend into the reference SHIP coefficient encoding. Requires
/// the backend's CKKS encoding implementation.
#[macro_export]
macro_rules! impl_ckks_ship_coeff_encoding {
    ($be:ty) => {
        unsafe impl ::poulpy_ckks::oep::CKKSShipCoeffEncodingImpl<$be> for $be {
            fn ckks_ship_coeff_encodings_tmp_bytes_impl<F>(
                _module: &::poulpy_hal::layouts::Module<$be>,
                _plan: &::poulpy_ckks::layouts::ShipPlan,
                _base2k: ::poulpy_core::layouts::Base2K,
                _complex: bool,
            ) -> ::poulpy_ckks::CKKSResult<usize>
            where
                F: ::poulpy_ckks::api::ShipScalar,
                $be: ::poulpy_ckks::oep::CKKSEncodingImpl<$be, F>,
            {
                // The reference path uses ordinary host vectors. Native
                // device implementations report their arena workspace here.
                Ok(0)
            }

            fn ckks_ship_coeff_encodings_impl<F, Src>(
                module: &::poulpy_hal::layouts::Module<$be>,
                ct: &Src,
                plan: &::poulpy_ckks::layouts::ShipPlan,
                base2k: ::poulpy_core::layouts::Base2K,
                complex: bool,
                _scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
            ) -> ::poulpy_ckks::CKKSResult<
                ::poulpy_ckks::layouts::ShipCoeffEncodings<
                    <$be as ::poulpy_hal::layouts::Backend>::OwnedBuf,
                    <$be as ::poulpy_hal::layouts::Backend>::ZnxWord,
                >,
            >
            where
                F: ::poulpy_ckks::api::ShipScalar,
                $be: ::poulpy_ckks::oep::CKKSEncodingImpl<$be, F>,
                Src: ::poulpy_core::layouts::GLWEToBackendRef<$be> + ::poulpy_ckks::CKKSCtBounds,
            {
                $crate::ckks_ship::ship_coeff_encodings_staged::<$be, F, Src>(module, ct, plan, base2k, complex)
                    .map_err(::poulpy_ckks::CKKSError::from)
            }
        }
    };
}
