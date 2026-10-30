//! Reference implementation of the PaCo coefficient encoding
//! ([`CKKSPaCoCoeffEncodingImpl`](poulpy_ckks::oep::CKKSPaCoCoeffEncodingImpl)).
//!
//! The scheme-defining ciphertext → four-plaintexts transformation lives in
//! `poulpy-ckks` ([`paco_coeff_encodings_host`]) and dispatches all value-level
//! work through the backend's CKKS encoding operations
//! ([`CKKSEncodingImpl`]); precomputed
//! encoding material is the module's own memoized state. This module only
//! reads generic inputs (e.g. scratch-carved views) through their
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
use poulpy_core::{
    GLWEBytesOf, GLWENormalize, ScratchArenaTakeCore,
    layouts::{Base2K, GLWEInfos, GLWEToBackendRef, LWEInfos},
};
use poulpy_hal::{
    api::ModuleN,
    layouts::{Backend, HostDataRef, Module, ScratchArena, Standard},
};

/// Scratch of [`paco_coeff_encodings_staged`] for an input of the layout `ct`: its rounded copy beside the
/// normalization.
pub fn paco_coeff_encodings_staged_tmp_bytes<BE, A>(module: &Module<BE>, ct: &A) -> usize
where
    BE: Backend,
    Module<BE>: GLWEBytesOf<BE> + GLWENormalize<BE>,
    A: GLWEInfos,
{
    BE::scratch_aligned(module.glwe_bytes_of_from_infos(&ct.glwe_layout())) + module.glwe_normalize_tmp_bytes()
}

/// Runs the scheme-defining [`paco_coeff_encodings_host`] with the prepared
/// encoding material on the small modulus-q ciphertext `ct`, owned or a view
/// such as a scratch carve, as copied into its own layout: limbs past its
/// width of a non-canonical `ct` are rounded in, into a copy carved from
/// `scratch`, and any other `ct` is read in place.
///
/// Returns an error for incompatible degree/rank/radix metadata, a scale
/// above the ciphertext's width, an arena below
/// [`paco_coeff_encodings_staged_tmp_bytes`] for a copy, non-representable
/// coefficients, or a failed encoding.
pub fn paco_coeff_encodings_staged<BE, F, Src>(
    module: &Module<BE>,
    base2k: Base2K,
    ct: &Src,
    plan: &PaCoPlan,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<[CKKSPlaintextOwned<BE>; 4]>
where
    BE: Backend<ZnxWord = i64, Ring = Standard> + CKKSEncodingImpl<F>,
    for<'x> BE::BufRef<'x>: HostDataRef,
    Module<BE>: ModuleN + CKKSModuleAlloc<BE> + CKKSEncodingOps<BE, F> + GLWEBytesOf<BE> + GLWENormalize<BE>,
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
        ct.n().as_usize() == plan.n(),
        "PaCo ciphertext degree {} does not match plan degree {}",
        ct.n(),
        plan.n()
    );
    ensure!(
        ct.log_delta() <= ct.k().as_usize(),
        "PaCo ciphertext scale {} exceeds its width {}",
        ct.log_delta(),
        ct.k()
    );
    let layout = ct.glwe_layout();
    let ct_ref = ct.to_backend_ref();
    if ct_ref.is_canonical() || ct_ref.max_size() <= layout.max_size() {
        return paco_coeff_encodings_host::<BE, _, F>(module, base2k, &ct_ref, plan);
    }
    ensure!(
        scratch.available() >= paco_coeff_encodings_staged_tmp_bytes(module, ct),
        "PaCo coefficient encoding: scratch below its query"
    );
    let (mut rounded, mut scratch) = scratch.borrow().take_glwe_scratch(&layout);
    module.glwe_normalize(&mut rounded, ct, &mut scratch);
    paco_coeff_encodings_host::<BE, _, F>(module, base2k, &rounded.to_backend_ref(), plan)
}

/// Opts a CPU backend into the reference PaCo coefficient encoding: the
/// staged scheme routine [`paco_coeff_encodings_staged`] over the backend's
/// CKKS encoding operations. Requires the backend's
/// CKKS encoding implementation.
#[macro_export]
macro_rules! impl_ckks_paco_coeff_encoding {
    ($be:ty) => {
        unsafe impl ::poulpy_ckks::oep::CKKSPaCoCoeffEncodingImpl for $be {
            fn ckks_paco_coeff_encodings_tmp_bytes<F, A>(
                module: &::poulpy_hal::layouts::Module<$be>,
                _context: &::poulpy_ckks::layouts::PaCoContext<$be, F>,
                input: &A,
            ) -> ::poulpy_ckks::CKKSResult<usize>
            where
                F: ::poulpy_ckks::api::PaCoScalar,
                $be: ::poulpy_ckks::oep::CKKSEncodingImpl<F>,
                A: ::poulpy_core::layouts::GLWEInfos,
            {
                Ok($crate::ckks_paco::paco_coeff_encodings_staged_tmp_bytes(module, input))
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
                $crate::ckks_paco::paco_coeff_encodings_staged::<$be, F, Src>(
                    module,
                    context.base2k(),
                    ciphertext,
                    context.plan(),
                    scratch,
                )
                .map_err(::poulpy_ckks::CKKSError::from)
            }
        }
    };
}
