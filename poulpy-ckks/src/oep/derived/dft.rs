//! Format compositions over the selected CKKS DFT and arithmetic contracts.
use crate::api::{
    CKKSAddOps, CKKSConjugateOps, CKKSCopyOps, CKKSDFTOps, CKKSImagOps, CKKSRotateOps, CKKSSubOps, LtDiagonalScale,
};
use crate::layouts::{CKKSModuleAlloc, DFTMatrix, Decode, Encode, Repack, Split, Standard};
use crate::{CKKSCtBounds, CKKSResult as Result, SetCKKSInfos, SlotsKind};
use poulpy_core::layouts::{GLWEToBackendMut, GLWEToBackendRef, GetAutomorphismKey, IntPolyInfos, LinearTransformation};
use poulpy_core::reference::linear_transformation::DiagonalProd;
use poulpy_hal::layouts::{Backend, Module, ScratchArena};
/// Homomorphic encoding (CoeffsToSlots), `Standard` format: evaluates the Encode
/// (IDFT) matrix in place. `dft.literal.kind` must be
/// [`DFTType::Encode`](crate::layouts::DFTType::Encode) and the
/// format [`crate::layouts::DFTOutputFormat::Standard`] (the real/imag-splitting formats are a later
/// increment).
pub(crate) fn ckks_coeffs_to_slots_assign<BE, P, Dst, H>(
    module: &Module<BE>,
    ct: &mut Dst,
    dft: &DFTMatrix<BE, Encode, Standard, LinearTransformation<P>>,
    keys: &H,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: Backend,
    P: DiagonalProd<BE> + LtDiagonalScale + IntPolyInfos,
    Module<BE>: crate::api::CKKSDFTOps<BE>,
    Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    H: GetAutomorphismKey<BE>,
{
    module.ckks_dft_evaluate_assign(ct, dft, keys, scratch)
}

/// Homomorphic decoding (SlotsToCoeffs), `Standard` format: evaluates the Decode
/// (DFT) matrix in place. `dft.literal.kind` must be
/// [`DFTType::Decode`](crate::layouts::DFTType::Decode).
pub(crate) fn ckks_slots_to_coeffs_assign<BE, P, Dst, H>(
    module: &Module<BE>,
    ct: &mut Dst,
    dft: &DFTMatrix<BE, Decode, Standard, LinearTransformation<P>>,
    keys: &H,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: Backend,
    P: DiagonalProd<BE> + LtDiagonalScale + IntPolyInfos,
    Module<BE>: crate::api::CKKSDFTOps<BE>,
    Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    H: GetAutomorphismKey<BE>,
{
    module.ckks_dft_evaluate_assign(ct, dft, keys, scratch)
}

/// `CoeffsToSlots` with the real and imaginary parts returned in two separate
/// real-vector ciphertexts (`DFTOutputFormat::SplitRealAndImag`, dense packing).
///
/// Evaluates the Encode matrix, then splits `z` into `2·Re(z)` and `2·Im(z)` with
/// a conjugation
/// (`z + z̄` and `−i·(z − z̄)`); the matrix's `1/(2·slots)` scaling makes the net
/// result `Re` / `Im` of the slot DFT. `conj_key` is the conjugation automorphism
/// key (Galois element `−1`). On return, `ct_real` holds the real parts and
/// `ct_imag` the imaginary parts. Consumes `ct_in` by reference (copied).
#[allow(clippy::too_many_arguments)]
pub(crate) fn ckks_coeffs_to_slots_split<BE, P, Dst, Src, H>(
    module: &Module<BE>,
    ct_real: &mut Dst,
    ct_imag: &mut Dst,
    ct_in: &Src,
    dft: &DFTMatrix<BE, Encode, Split, LinearTransformation<P>>,
    keys: &H,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: Backend,
    P: DiagonalProd<BE> + LtDiagonalScale + IntPolyInfos,
    Module<BE>: crate::api::CKKSDFTOps<BE>
        + CKKSModuleAlloc<BE>
        + CKKSCopyOps<BE>
        + CKKSConjugateOps<BE>
        + CKKSAddOps<BE>
        + CKKSSubOps<BE>
        + CKKSImagOps<BE>,
    Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    Src: GLWEToBackendRef<BE> + CKKSCtBounds,
    H: GetAutomorphismKey<BE>,
{
    // ct_real := z = Encode(ct_in).
    module.ckks_copy(ct_real, ct_in, scratch)?;
    module.ckks_dft_evaluate_assign(ct_real, dft, keys, scratch)?;

    // ct_imag := conj(z).
    module.ckks_conjugate_into(ct_imag, ct_real, keys, scratch)?;

    // tmp := z − conj(z); ct_real := z + conj(z) = 2·Re(z); ct_imag := −i·tmp = 2·Im(z).
    let mut tmp = module.ckks_ciphertext_alloc_from_infos(ct_real);
    module.ckks_sub_into(&mut tmp, ct_real, ct_imag, scratch)?;
    module.ckks_add_assign(ct_real, ct_imag, scratch)?;
    module.ckks_div_i_into(ct_imag, &tmp, scratch)?;
    // Both halves are the (real) coefficients of the input polynomial.
    ct_real.set_slots(SlotsKind::Real);
    ct_imag.set_slots(SlotsKind::Real);
    Ok(())
}

/// `SlotsToCoeffs` from real/imaginary parts in two ciphertexts
/// (`DFTOutputFormat::SplitRealAndImag`, dense packing).
///
/// Combines `ct_real + i·ct_imag`, then evaluates the Decode matrix. Writes the
/// result into `op_out`.
pub(crate) fn ckks_slots_to_coeffs_split<BE, P, Dst, Src, H>(
    module: &Module<BE>,
    op_out: &mut Dst,
    ct_real: &Src,
    ct_imag: &Src,
    dft: &DFTMatrix<BE, Decode, Split, LinearTransformation<P>>,
    keys: &H,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: Backend,
    P: DiagonalProd<BE> + LtDiagonalScale + IntPolyInfos,
    Module<BE>: crate::api::CKKSDFTOps<BE> + CKKSAddOps<BE> + CKKSImagOps<BE>,
    Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    Src: GLWEToBackendRef<BE> + CKKSCtBounds,
    H: GetAutomorphismKey<BE>,
{
    // op_out := ct_real + i·ct_imag, then Decode.
    module.ckks_mul_i_into(op_out, ct_imag, scratch)?;
    module.ckks_add_assign(op_out, ct_real, scratch)?;
    module.ckks_dft_evaluate_assign(op_out, dft, keys, scratch)
}

/// Sparse `CoeffsToSlots` with the imaginary part repacked into the right half of
/// a single ciphertext (`DFTOutputFormat::RepackImagAsReal`, `log_slots < log_max_slots`).
///
/// Evaluates the Encode matrix, splits `z` into `2·Re(z)` / `2·Im(z)`, then rotates
/// the imaginary part by `slots` and adds it into the real part so the result holds
/// `Re` in the left `slots` and `Im` in the right `slots` of each `2·slots` period.
/// The live slot count doubles, so `ct_out.log_sparsity` is decremented by one.
#[allow(clippy::too_many_arguments)]
pub(crate) fn ckks_coeffs_to_slots_repack<BE, P, Dst, Src, H>(
    module: &Module<BE>,
    ct_out: &mut Dst,
    ct_in: &Src,
    dft: &DFTMatrix<BE, Encode, Repack, LinearTransformation<P>>,
    keys: &H,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: Backend,
    P: DiagonalProd<BE> + LtDiagonalScale + IntPolyInfos,
    Module<BE>: crate::api::CKKSDFTOps<BE>
        + CKKSModuleAlloc<BE>
        + CKKSCopyOps<BE>
        + CKKSConjugateOps<BE>
        + CKKSAddOps<BE>
        + CKKSSubOps<BE>
        + CKKSImagOps<BE>
        + CKKSRotateOps<BE>,
    Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    Src: GLWEToBackendRef<BE> + CKKSCtBounds,
    H: GetAutomorphismKey<BE>,
{
    let slots = 1i64 << dft.plan().log_slots();

    // ct_out := z = Encode(ct_in).
    module.ckks_copy(ct_out, ct_in, scratch)?;
    module.ckks_dft_evaluate_assign(ct_out, dft, keys, scratch)?;

    // conj := conj(z); imag := −i·(z − conj) = 2·Im(z); ct_out := z + conj = 2·Re(z).
    let mut conj = module.ckks_ciphertext_alloc_from_infos(ct_out);
    module.ckks_conjugate_into(&mut conj, ct_out, keys, scratch)?;
    let mut imag = module.ckks_ciphertext_alloc_from_infos(ct_out);
    module.ckks_sub_into(&mut imag, ct_out, &conj, scratch)?;
    module.ckks_div_i_assign(&mut imag, scratch)?;
    module.ckks_add_assign(ct_out, &conj, scratch)?;

    // Repack: rotate Im by `slots` and add into Re → [Re | Im] per 2·slots period.
    module.ckks_rotate_assign(&mut imag, slots, keys, scratch)?;
    module.ckks_add_assign(ct_out, &imag, scratch)?;

    // The repack doubles the live slot count.
    let log_sparsity = ct_in.log_sparsity().saturating_sub(1);
    ct_out.set_log_sparsity(log_sparsity);
    // `[Re | Im]` packs the input polynomial's (real) coefficients.
    ct_out.set_slots(SlotsKind::Real);
    Ok(())
}

/// Sparse `SlotsToCoeffs` from a single repacked ciphertext (the inverse of
/// [`ckks_coeffs_to_slots_repack`]; `DFTOutputFormat::RepackImagAsReal`, sparse).
///
/// The Decode matrices already embed the repack matrix that recombines the
/// `[Re | Im]` real packing into the complex form, so this is just an in-place
/// evaluation. The live slot count halves, so `op_out.log_sparsity` is incremented
/// by one.
pub(crate) fn ckks_slots_to_coeffs_repack<BE, P, Dst, Src, H>(
    module: &Module<BE>,
    op_out: &mut Dst,
    ct_in: &Src,
    dft: &DFTMatrix<BE, Decode, Repack, LinearTransformation<P>>,
    keys: &H,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: Backend,
    P: DiagonalProd<BE> + LtDiagonalScale + IntPolyInfos,
    Module<BE>: crate::api::CKKSDFTOps<BE> + CKKSCopyOps<BE>,
    Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    Src: GLWEToBackendRef<BE> + CKKSCtBounds,
    H: GetAutomorphismKey<BE>,
{
    module.ckks_copy(op_out, ct_in, scratch)?;
    module.ckks_dft_evaluate_assign(op_out, dft, keys, scratch)?;

    // The repack-decode halves the live slot count.
    op_out.set_log_sparsity(ct_in.log_sparsity() + 1);
    Ok(())
}
