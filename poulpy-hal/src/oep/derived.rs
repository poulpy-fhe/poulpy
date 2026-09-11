//! Shared default bodies for the derived HAL operations.
//!
//! Every function here is the canonical decomposition of one derived
//! operation in terms of *other* OEP methods of the same backend. The OEP
//! trait method's default body is a one-line call into the matching
//! `<op>_derived` function, and its `_tmp_bytes` default calls
//! `<op>_tmp_bytes_derived`.
//!
//! Free functions in this module are generic over the implementing type `S`
//! and the backend `BE`; a trait default body passes `Self` for `S`, while a
//! test that wants to run the decomposition directly on a concrete backend
//! passes that backend for both `S` and `BE`. `S`'s only bound is the OEP
//! trait of the op being decomposed (e.g. `S: HalVmpImpl<BE>`); the
//! supertrait chain on `S` supplies whichever other families the
//! decomposition needs (e.g. `HalVecZnxDftImpl<BE>`), so every OEP call
//! inside is dispatched as `<S as HalXImpl<BE>>::method(...)`, never
//! `<BE as HalXImpl<BE>>::...`.
//!
//! Rules (spec section 5) every function in this module obeys:
//!
//! - only OEP methods of `S`/`BE`, backend-native views, [`ScratchArena`]
//!   carving and `Backend::bytes_of_*`;
//! - no `HostData*` bound, no family kernel trait, no knowledge of a prepared
//!   representation;
//! - `<op>_tmp_bytes_derived` reports exactly the scratch `<op>_derived` takes.
//!
//! These functions are `pub` so that `poulpy_hal::test_suite` can call the
//! decomposition directly and pin it against a hand-built oracle, which is
//! how the per-op parity tests are written. They are not part of the public
//! HAL surface; call the api traits instead.

#![allow(clippy::too_many_arguments)]

use crate::{
    api::ScratchArenaTakeBasic,
    layouts::{
        Backend, MatZnxInfos, Module, ScratchArena, VecZnxBackendRef, VecZnxDftToBackendMut, VecZnxDftToBackendRef, VecZnxInfos,
        VmpPMatBackendRef, ZnxInfos,
    },
    oep::{HalVecZnxDftImpl, HalVmpImpl},
};

/// Scratch for [`vmp_apply_dft_derived`]: one `VecZnxDft` for the transformed
/// input plus whatever `vmp_apply_dft_to_dft` needs.
#[doc(hidden)]
pub fn vmp_apply_dft_tmp_bytes_derived<S, BE>(
    module: &Module<BE>,
    res_size: usize,
    a_size: usize,
    b_rows: usize,
    b_cols_in: usize,
    b_cols_out: usize,
    b_size: usize,
) -> usize
where
    S: HalVmpImpl<BE>,
    BE: Backend,
{
    let a_dft_size = a_size.min(b_rows);
    BE::bytes_of_vec_znx_dft(module.n(), b_cols_in, a_dft_size)
        + <S as HalVmpImpl<BE>>::vmp_apply_dft_to_dft_tmp_bytes(
            module, res_size, a_dft_size, b_rows, b_cols_in, b_cols_out, b_size,
        )
}

/// `res = dft(a) * pmat`: transform the `min(a.size(), pmat.rows())` limbs the
/// matrix consumes into scratch, aligning `a`'s trailing columns with
/// `pmat.cols_in()` and zeroing the leading ones, then apply in the DFT domain.
#[doc(hidden)]
pub fn vmp_apply_dft_derived<S, BE, R>(
    module: &Module<BE>,
    res: &mut R,
    a: &VecZnxBackendRef<'_, BE>,
    b: &VmpPMatBackendRef<'_, BE>,
    scratch: &mut ScratchArena<'_, BE>,
) where
    S: HalVmpImpl<BE>,
    BE: Backend,
    R: VecZnxDftToBackendMut<BE>,
{
    let a_cols: usize = VecZnxInfos::cols(a);
    let a_size: usize = ZnxInfos::size(a);
    let b_rows: usize = MatZnxInfos::rows(b);
    let cols_to_copy: usize = a_cols.min(MatZnxInfos::cols_in(b));
    let a_start_col: usize = a_cols - cols_to_copy;
    let a_dft_size: usize = a_size.min(b_rows);
    let offset: usize = MatZnxInfos::cols_in(b) - cols_to_copy;

    let (mut a_dft, mut scratch) =
        ScratchArenaTakeBasic::take_vec_znx_dft_scratch(scratch.borrow(), module, MatZnxInfos::cols_in(b), a_dft_size);

    for j in 0..offset {
        <S as HalVecZnxDftImpl<BE>>::vec_znx_dft_zero(module, &mut a_dft, j);
    }

    for j in 0..cols_to_copy {
        <S as HalVecZnxDftImpl<BE>>::vec_znx_dft_apply(module, 1, 0, &mut a_dft, offset + j, a, a_start_col + j);
    }

    let mut res_ref = res.to_backend_mut();
    <S as HalVmpImpl<BE>>::vmp_apply_dft_to_dft(module, &mut res_ref, &a_dft.to_backend_ref(), b, 0, &mut scratch);
}
