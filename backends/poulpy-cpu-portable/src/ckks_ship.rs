//! Reference implementation of the SHIP coefficient encoding
//! ([`CKKSShipCoeffEncodingImpl`](poulpy_ckks::oep::CKKSShipCoeffEncodingImpl)).
//!
//! The scheme-defining transformation lives in `poulpy-ckks`
//! ([`ship_coeff_encodings_host`]); this module only reads generic inputs
//! (e.g. scratch-carved views) through their host-addressable backend views. Invoke [`impl_ckks_ship_coeff_encoding`](crate::impl_ckks_ship_coeff_encoding)
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
    GLWEBytesOf, GLWENormalize, ScratchArenaTakeCore,
    layouts::{Base2K, GLWELayout, GLWEToBackendRef, LWEInfos, Rank},
};
use poulpy_hal::{
    api::ModuleN,
    layouts::{Backend, HostDataRef, Module, ScratchArena, Standard},
};

/// The one-limb rank-1 layout of a SHIP bottom ciphertext of `plan` at `base2k`.
fn bottom_layout(base2k: Base2K, plan: &ShipPlan) -> GLWELayout {
    GLWELayout {
        n: plan.n().into(),
        base2k,
        k: base2k.as_usize().into(),
        rank: Rank(1),
    }
}

/// Scratch of [`ship_coeff_encodings_staged`]: the rounded copy of a bottom ciphertext beside the normalization.
pub fn ship_coeff_encodings_staged_tmp_bytes<BE>(module: &Module<BE>, base2k: Base2K, plan: &ShipPlan) -> usize
where
    BE: Backend,
    Module<BE>: GLWEBytesOf<BE> + GLWENormalize<BE>,
{
    BE::scratch_aligned(module.glwe_bytes_of_from_infos(&bottom_layout(base2k, plan))) + module.glwe_normalize_tmp_bytes()
}

/// Runs the scheme-defining [`ship_coeff_encodings_host`] on the one-limb
/// bottom ciphertext `ct`, owned or a view such as a scratch carve, as copied
/// into its own layout: limbs past its width of a non-canonical `ct` are
/// rounded in, into a copy carved from `scratch`, and any other `ct` is read
/// in place.
pub fn ship_coeff_encodings_staged<BE, F, Src>(
    module: &Module<BE>,
    base2k: Base2K,
    complex: bool,
    ct: &Src,
    plan: &ShipPlan,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<ShipCoeffEncodings<BE::OwnedBuf, BE::ZnxWord>>
where
    BE: Backend<ZnxWord = i64, Ring = Standard> + CKKSEncodingImpl<F>,
    for<'x> BE::BufRef<'x>: HostDataRef,
    Module<BE>: ModuleN + CKKSModuleAlloc<BE> + CKKSEncodingOps<BE, F> + GLWEBytesOf<BE> + GLWENormalize<BE>,
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
    let layout = ct.glwe_layout();
    let ct_ref = ct.to_backend_ref();
    if ct_ref.is_canonical() || ct_ref.max_size() <= layout.max_size() {
        return ship_coeff_encodings_host::<BE, _, F>(module, base2k, complex, &ct_ref, plan);
    }
    // The copy is carved only at the layout the query counts, which the encoding requires anyway.
    ensure!(
        layout == bottom_layout(base2k, plan),
        "SHIP ciphertext layout {layout:?} is not the one-limb rank-1 layout at base2k {base2k}"
    );
    ensure!(
        scratch.available() >= ship_coeff_encodings_staged_tmp_bytes(module, base2k, plan),
        "SHIP coefficient encoding: scratch below its query"
    );
    let (mut rounded, mut scratch) = scratch.borrow().take_glwe_scratch(&layout);
    module.glwe_normalize(&mut rounded, ct, &mut scratch);
    ship_coeff_encodings_host::<BE, _, F>(module, base2k, complex, &rounded.to_backend_ref(), plan)
}

/// Opts a CPU backend into the reference SHIP coefficient encoding. Requires
/// the backend's CKKS encoding implementation.
#[macro_export]
macro_rules! impl_ckks_ship_coeff_encoding {
    ($be:ty) => {
        unsafe impl ::poulpy_ckks::oep::CKKSShipCoeffEncodingImpl for $be {
            fn ckks_ship_coeff_encodings_tmp_bytes<F>(
                module: &::poulpy_hal::layouts::Module<$be>,
                base2k: ::poulpy_core::layouts::Base2K,
                _complex: bool,
                plan: &::poulpy_ckks::layouts::ShipPlan,
            ) -> ::poulpy_ckks::CKKSResult<usize>
            where
                F: ::poulpy_ckks::api::ShipScalar,
                $be: ::poulpy_ckks::oep::CKKSEncodingImpl<F>,
            {
                Ok($crate::ckks_ship::ship_coeff_encodings_staged_tmp_bytes(
                    module, base2k, plan,
                ))
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
                $crate::ckks_ship::ship_coeff_encodings_staged::<$be, F, Src>(module, base2k, complex, ciphertext, plan, scratch)
                    .map_err(::poulpy_ckks::CKKSError::from)
            }
        }
    };
}
