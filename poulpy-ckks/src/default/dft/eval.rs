//! Homomorphic (I)DFT generation and evaluation.
//!
//! Builds the prepared factor operands of a [`DFTMatrix`] from a
//! [`DFTPlan`] (encode each generated [`crate::layouts::ComplexDiagonals`] factor into a
//! CKKS linear transformation, then prepare its right operand), and evaluates the
//! transform by chaining one prepared linear transformation per factor.
//!
//! Scale accounting (see the homomorphic-DFT scale accounting in
//! [`docs/bootstrapping.md`](https://github.com/poulpy-fhe/poulpy/blob/main/docs/bootstrapping.md)). poulpy's torus plaintext-multiply
//! already realigns the result to the **input** `log_delta` via its `cnv_offset` —
//! i.e. the rescale is implicit in the linear-transform eval. So the homomorphic
//! (I)DFT is simply *one prepared linear transformation per factor, chained*, with
//! **no** explicit rescale between them. Each factor consumes `factor_meta.log_delta`
//! bits of `log_budget`; the whole transform consumes [`DFTMatrix::consumed_bits`],
//! which the input ciphertext must have available.
//!
//! `Standard` format goes through `coeffs_to_slots` / `slots_to_coeffs`;
//! `SplitRealAndImag` (real/imag returned in two ciphertexts) goes through
//! `coeffs_to_slots_split` / `slots_to_coeffs_split`; the sparse `RepackImagAsReal`
//! packing (imag packed into the right half) through `coeffs_to_slots_repack` /
//! `slots_to_coeffs_repack`.

use crate::CKKSAtkBounds;
use crate::{CKKSResult as Result, ckks_ensure};
use poulpy_core::layouts::IntPolyInfos;
use poulpy_core::{
    default::linear_transformation::DiagonalProd,
    layouts::{
        Base2K, GLWEAutomorphismKeyHelper, GLWEToBackendMut, GLWEToBackendRef, LinearTransformation, LinearTransformationStrategy,
    },
};
use poulpy_hal::{
    api::CnvPVecAlloc,
    layouts::{Backend, Module, ScratchArena},
};

use crate::SlotsKind;
use crate::{
    CKKSCompositionError, CKKSCtBounds, SetCKKSInfos,
    api::{
        CKKSAddOps, CKKSConjugateOps, CKKSCopyOps, CKKSEncodingOps, CKKSEncodingScalar, CKKSImagOps, CKKSLinearTransformationOps,
        CKKSRotateOps, CKKSSubOps, LinearTransformationBabySteps, LinearTransformationPrepared, LtDiagonalScale,
    },
    default::dft::matrices::{DftScalar, gen_dft_matrices},
    layouts::{
        CKKSModuleAlloc, CKKSPlaintextOwned, DFTMatrix, DFTMatrixFactors, DFTMatrixPrepared, DFTOutputFormat, DFTPlan, Decode,
        DftDirection, DftFormat, Encode, Repack, ScratchArenaTakeCKKS, Split, Standard,
    },
    oep::CKKSEncodingImpl,
};

/// One unprepared factor: the diagonals of a single DFT factor matrix encoded
/// as a CKKS linear transformation (plaintext diagonals).
type DftFactorLt<BE> = LinearTransformation<CKKSPlaintextOwned<BE>>;

fn checked_dft_log_slots(literal: &DFTPlan, log_n: usize) -> Result<usize> {
    // Shape validity (including a non-overflowing layer sum) is guaranteed by
    // `DFTPlan::new`; only the plan-vs-ring clause remains to check here.
    let log_slots = literal.log_slots();
    ckks_ensure!(
        log_slots < log_n,
        "DFT has 2^{log_slots} slots, but a degree-2^{log_n} CKKS ring supports at most 2^{} slots",
        log_n.saturating_sub(1),
    );
    Ok(log_slots)
}

/// Resolves the plan and generates + encodes each factor into an (unprepared)
/// CKKS linear transformation (the plaintext factors of an unprepared
/// [`DFTMatrix`]); [`ckks_prepare_dft_matrix`] later prepares each factor into a
/// `CnvPVec` to produce the resident [`DFTMatrixPrepared`].
#[allow(clippy::too_many_arguments)]
fn dft_factor_lts<BE, F>(
    module: &Module<BE>,
    base2k: Base2K,
    literal: &DFTPlan,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<(DFTPlan, Vec<DftFactorLt<BE>>)>
where
    BE: Backend + CKKSEncodingImpl<BE, F>,
    Module<BE>: CnvPVecAlloc<BE> + CKKSLinearTransformationOps<BE> + CKKSModuleAlloc<BE> + CKKSEncodingOps<BE, F>,
    F: DftScalar + CKKSEncodingScalar,
{
    checked_dft_log_slots(literal, module.log_n())?;

    // Resolve the plan: sparsity is decided against the ring degree, then a dense
    // `RepackImagAsReal` is canonicalized to `SplitRealAndImag` (the two coincide)
    // so the stored `format` is canonical and `plan.is_sparse()` is exact. The
    // per-factor scale is carried by `literal.coeffs_meta`.
    let sparse = literal.log_slots() < module.log_n().saturating_sub(1) && literal.format == DFTOutputFormat::RepackImagAsReal;
    let mut plan = literal.clone();
    if literal.format == DFTOutputFormat::RepackImagAsReal && !sparse {
        plan.format = DFTOutputFormat::SplitRealAndImag;
    }

    let factors_cd = gen_dft_matrices::<F>(&plan, module.log_n());

    // Each factor's BSGS width is taken from the plan's schedule; the library
    // applies no implicit optimum. `giant_step == 1` is the direct schedule.
    let lts = factors_cd
        .iter()
        .zip(&plan.schedule.steps)
        .map(|(cd, step)| {
            let strategy = LinearTransformationStrategy::Bsgs {
                giant_step: step.giant_step,
            };
            crate::default::ckks_encode_linear_transformation_from_diagonals::<BE, F>(
                module,
                base2k,
                plan.coeffs_meta,
                cd,
                strategy,
                false,
                scratch,
            )
        })
        .collect::<core::result::Result<Vec<_>, ::anyhow::Error>>()
        .map_err(crate::CKKSError::from)?;

    Ok((plan, lts))
}

/// Prepares an unprepared (host plaintext) [`DFTMatrix`] into its resident
/// convolution-domain form [`DFTMatrixPrepared`], preserving the `Dir`/`Fmt`
/// type-state.
///
/// Each factor's plaintext diagonals are prepared into a `CnvPVec` right operand
/// (resident), so repeated evaluation no longer materializes them per factor. The
/// resolved plan is preserved; only the factor storage changes. Build the input
/// via [`ckks_new_dft_matrix`].
pub fn ckks_prepare_dft_matrix<Dir, Fmt, BE, P>(
    module: &Module<BE>,
    dft: &DFTMatrix<BE, Dir, Fmt, LinearTransformation<P>>,
    scratch: &mut ScratchArena<'_, BE>,
) -> DFTMatrixPrepared<BE, Dir, Fmt>
where
    BE: Backend,
    Module<BE>: CnvPVecAlloc<BE> + CKKSLinearTransformationOps<BE>,
    P: GLWEToBackendRef<BE> + IntPolyInfos + CKKSCtBounds + DiagonalProd<BE>,
{
    let inner = dft.inner();
    let plan = inner.plan.clone();

    let mut factors = Vec::with_capacity(inner.factors.len());
    for lt in &inner.factors {
        let first_pt = lt.first_diagonal_plaintext().expect("dft factor has no diagonals");
        let mut prepared = LinearTransformationPrepared::<BE>::alloc_prepared_from_index(module, &lt.index(), first_pt);
        module.ckks_prepare_linear_transformation_rhs(&mut prepared, lt, scratch);
        factors.push(prepared);
    }

    // Same direction/format as the input; only `R` changes.
    DFTMatrix::from_factors(DFTMatrixFactors::new(plan, factors))
}

/// Builds the backend-owned, unprepared homomorphic (I)DFT for direction `Dir` and
/// output format `Fmt`, described by `literal`.
///
/// The `Dir`/`Fmt` type parameters are authoritative: they set the transform
/// direction and the requested output format (`literal.kind` / `literal.format`
/// are ignored, like `factor_log_delta`). Each factor matrix is encoded into a
/// CKKS linear transformation whose diagonals are kept as plaintexts (the default
/// [`DFTMatrix`]); evaluating it materializes each diagonal on the fly per factor.
/// Promote it to the resident form with [`ckks_prepare_dft_matrix`].
///
/// Returns an error when the plan is invalid, a generated diagonal cannot be
/// encoded at the requested metadata, or `Fmt = Repack` is requested with
/// dense parameters (`log_slots ≥ log_n − 1`), in which case the repack
/// collapses to `Split` and the caller must build with `Split` instead. The
/// latter is the single runtime format resolution; every downstream eval is
/// type-checked.
#[allow(clippy::too_many_arguments)]
pub fn ckks_new_dft_matrix<Dir, Fmt, BE, F>(
    module: &Module<BE>,
    base2k: Base2K,
    literal: &DFTPlan,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<DFTMatrix<BE, Dir, Fmt>>
where
    Dir: DftDirection,
    Fmt: DftFormat,
    BE: Backend + CKKSEncodingImpl<BE, F>,
    Module<BE>: CnvPVecAlloc<BE> + CKKSLinearTransformationOps<BE> + CKKSModuleAlloc<BE> + CKKSEncodingOps<BE, F>,
    F: DftScalar + CKKSEncodingScalar,
{
    // The markers are the source of truth for direction and requested format.
    let mut literal = literal.clone();
    literal.kind = Dir::KIND;
    literal.format = Fmt::FORMAT;

    let (plan, lts) = dft_factor_lts::<BE, F>(module, base2k, &literal, scratch)?;

    // The dense-`RepackImagAsReal` → `Split` canonicalization is the only runtime
    // format resolution; if it changed the requested format, the request was
    // unrepresentable for these parameters.
    if plan.format != Fmt::FORMAT {
        return Err(CKKSCompositionError::DftMatrixMismatch {
            op: "ckks_new_dft_matrix",
            expected: "sparse Repack (RepackImagAsReal)",
            got: "dense parameters — build with the `Split` format instead".to_string(),
        }
        .into());
    }

    Ok(DFTMatrix::from_factors(DFTMatrixFactors::new(plan, lts)))
}

impl<BE: Backend, Dir, Fmt: DftFormat, P> DFTMatrix<BE, Dir, Fmt, LinearTransformation<P>> {
    /// The distinct Galois elements whose automorphism keys evaluating this
    /// transform requires (the union over all factors, plus the `slots` repack
    /// rotation for the sparse path).
    pub fn galois_elements(&self, cyclotomic_order: i64) -> Vec<i64> {
        let mut set: std::collections::BTreeSet<i64> = std::collections::BTreeSet::new();
        for f in self.factors() {
            set.extend(f.galois_elements(cyclotomic_order));
        }
        if self.is_sparse() {
            set.insert(poulpy_hal::layouts::galois_element(
                1i64 << self.plan().log_slots(),
                cyclotomic_order,
            ));
        }
        // Defensive: drop the identity element if present.
        set.remove(&poulpy_hal::layouts::galois_element(0, cyclotomic_order));
        set.into_iter().collect()
    }
}

/// Evaluates the homomorphic (I)DFT `dft` in place on `ct`.
///
/// Chains one linear transformation per factor — generic over the diagonal
/// representation `P` (resident `PreparedDiagonal` or streamed plaintext),
/// dispatched by the single unified
/// [`ckks_eval_linear_transformation_assign`](CKKSLinearTransformationOps::ckks_eval_linear_transformation_assign)
/// — preparing the baby rotations of the running ciphertext each time. No
/// explicit rescale is needed (the plaintext-multiply realigns to the input
/// `log_delta`, see the module docs). The input `ct.log_budget()` must be at
/// least `dft.consumed_bits()`.
///
/// The chain alternates between `ct` and a single scratch ciphertext, so no
/// factor materializes an intermediate copy; an odd factor count evaluates its
/// first factor in place to land the result back in `ct`.
pub fn ckks_dft_evaluate_assign<BE, Dir, Fmt, P, Dst, H, K>(
    module: &Module<BE>,
    ct: &mut Dst,
    dft: &DFTMatrix<BE, Dir, Fmt, LinearTransformation<P>>,
    keys: &H,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: Backend,
    P: DiagonalProd<BE> + LtDiagonalScale + IntPolyInfos,
    Module<BE>: CKKSLinearTransformationOps<BE> + CnvPVecAlloc<BE>,
    Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    K: CKKSAtkBounds<BE>,
    H: GLWEAutomorphismKeyHelper<K, BE>,
{
    // Ping-pong: the factors alternate between `ct` and one scratch ciphertext
    // through the `_into` eval, so only the (odd-count) leading factor pays for
    // an intermediate copy instead of every factor doing so.
    let factors = dft.factors();
    let (head, rest) = factors.split_at(factors.len() % 2);
    for factor in head {
        eval_factor_assign(module, ct, factor, keys, scratch)?;
    }
    if rest.is_empty() {
        return Ok(());
    }
    scratch.scope(|scratch| {
        let (mut tmp, mut scratch) = scratch.take_ckks_ciphertext_like_scratch(ct);
        for pair in rest.chunks(2) {
            eval_factor_into(module, &mut tmp, ct, &pair[0], keys, &mut scratch)?;
            eval_factor_into(module, ct, &tmp, &pair[1], keys, &mut scratch)?;
        }
        Ok(())
    })
}

/// Evaluates a single homomorphic-DFT factor from `src` into `dst`:
/// (re)allocates and prepares the baby rotations of `src`, then applies the
/// unified linear-transformation eval. `P` only decides how the factor's RHS is
/// materialized inside the eval (resident vs streamed).
fn eval_factor_into<BE, P, Dst, Src, H, K>(
    module: &Module<BE>,
    dst: &mut Dst,
    src: &Src,
    factor: &LinearTransformation<P>,
    keys: &H,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: Backend,
    P: DiagonalProd<BE> + LtDiagonalScale + IntPolyInfos,
    Module<BE>: CKKSLinearTransformationOps<BE> + CnvPVecAlloc<BE>,
    Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
    Src: GLWEToBackendRef<BE> + CKKSCtBounds,
    K: CKKSAtkBounds<BE>,
    H: GLWEAutomorphismKeyHelper<K, BE>,
{
    // `dst` is the previous factor's buffer: its logical metadata and width still
    // describe that factor's output. Realign it with `src` before anything reads
    // `dst.k()` (the eval parameters) or sizes work off it.
    dst.set_meta(src.meta());
    dst.set_k(src.k());
    let mut babies = LinearTransformationBabySteps::alloc(module, factor.baby_steps(), src);
    module.ckks_prepare_linear_transformation_baby_steps(&mut babies, src, keys, scratch)?;
    module.ckks_eval_linear_transformation_into(dst, src, &babies, factor, keys, scratch)
}

/// [`eval_factor_into`] in place, for the leading factor of an odd-length chain.
fn eval_factor_assign<BE, P, Dst, H, K>(
    module: &Module<BE>,
    running: &mut Dst,
    factor: &LinearTransformation<P>,
    keys: &H,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: Backend,
    P: DiagonalProd<BE> + LtDiagonalScale + IntPolyInfos,
    Module<BE>: CKKSLinearTransformationOps<BE> + CnvPVecAlloc<BE>,
    Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    K: CKKSAtkBounds<BE>,
    H: GLWEAutomorphismKeyHelper<K, BE>,
{
    let mut babies = LinearTransformationBabySteps::alloc(module, factor.baby_steps(), running);
    module.ckks_prepare_linear_transformation_baby_steps(&mut babies, running, keys, scratch)?;
    module.ckks_eval_linear_transformation_assign(running, &babies, factor, keys, scratch)
}

/// Homomorphic encoding (CoeffsToSlots), `Standard` format: evaluates the Encode
/// (IDFT) matrix in place. `dft.literal.kind` must be
/// [`DFTType::Encode`](crate::layouts::DFTType::Encode) and the
/// format [`DFTOutputFormat::Standard`] (the real/imag-splitting formats are a later
/// increment).
pub fn ckks_coeffs_to_slots_assign<BE, P, Dst, H, K>(
    module: &Module<BE>,
    ct: &mut Dst,
    dft: &DFTMatrix<BE, Encode, Standard, LinearTransformation<P>>,
    keys: &H,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: Backend,
    P: DiagonalProd<BE> + LtDiagonalScale + IntPolyInfos,
    Module<BE>: CKKSLinearTransformationOps<BE> + CnvPVecAlloc<BE>,
    Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    K: CKKSAtkBounds<BE>,
    H: GLWEAutomorphismKeyHelper<K, BE>,
{
    ckks_dft_evaluate_assign(module, ct, dft, keys, scratch)
}

/// Homomorphic decoding (SlotsToCoeffs), `Standard` format: evaluates the Decode
/// (DFT) matrix in place. `dft.literal.kind` must be
/// [`DFTType::Decode`](crate::layouts::DFTType::Decode).
pub fn ckks_slots_to_coeffs_assign<BE, P, Dst, H, K>(
    module: &Module<BE>,
    ct: &mut Dst,
    dft: &DFTMatrix<BE, Decode, Standard, LinearTransformation<P>>,
    keys: &H,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: Backend,
    P: DiagonalProd<BE> + LtDiagonalScale + IntPolyInfos,
    Module<BE>: CKKSLinearTransformationOps<BE> + CnvPVecAlloc<BE>,
    Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    K: CKKSAtkBounds<BE>,
    H: GLWEAutomorphismKeyHelper<K, BE>,
{
    ckks_dft_evaluate_assign(module, ct, dft, keys, scratch)
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
pub fn ckks_coeffs_to_slots_split<BE, P, Dst, Src, H, K>(
    module: &Module<BE>,
    ct_real: &mut Dst,
    ct_imag: &mut Dst,
    ct_in: &Src,
    dft: &DFTMatrix<BE, Encode, Split, LinearTransformation<P>>,
    keys: &H,
    conj_key: &K,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: Backend,
    P: DiagonalProd<BE> + LtDiagonalScale + IntPolyInfos,
    Module<BE>: CKKSLinearTransformationOps<BE>
        + CnvPVecAlloc<BE>
        + CKKSModuleAlloc<BE>
        + CKKSCopyOps<BE>
        + CKKSConjugateOps<BE>
        + CKKSAddOps<BE>
        + CKKSSubOps<BE>
        + CKKSImagOps<BE>,
    Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    Src: GLWEToBackendRef<BE> + CKKSCtBounds,
    K: CKKSAtkBounds<BE>,
    H: GLWEAutomorphismKeyHelper<K, BE>,
{
    // ct_real := z = Encode(ct_in).
    module.ckks_copy(ct_real, ct_in, scratch)?;
    ckks_dft_evaluate_assign(module, ct_real, dft, keys, scratch)?;

    // ct_imag := conj(z).
    module.ckks_conjugate_into(ct_imag, ct_real, conj_key, scratch)?;

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
pub fn ckks_slots_to_coeffs_split<BE, P, Dst, Src, H, K>(
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
    Module<BE>: CKKSLinearTransformationOps<BE> + CnvPVecAlloc<BE> + CKKSAddOps<BE> + CKKSImagOps<BE>,
    Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    Src: GLWEToBackendRef<BE> + CKKSCtBounds,
    K: CKKSAtkBounds<BE>,
    H: GLWEAutomorphismKeyHelper<K, BE>,
{
    // op_out := ct_real + i·ct_imag, then Decode.
    module.ckks_mul_i_into(op_out, ct_imag, scratch)?;
    module.ckks_add_assign(op_out, ct_real, scratch)?;
    ckks_dft_evaluate_assign(module, op_out, dft, keys, scratch)
}

/// Sparse `CoeffsToSlots` with the imaginary part repacked into the right half of
/// a single ciphertext (`DFTOutputFormat::RepackImagAsReal`, `log_slots < log_max_slots`).
///
/// Evaluates the Encode matrix, splits `z` into `2·Re(z)` / `2·Im(z)`, then rotates
/// the imaginary part by `slots` and adds it into the real part so the result holds
/// `Re` in the left `slots` and `Im` in the right `slots` of each `2·slots` period.
/// The live slot count doubles, so `ct_out.log_sparsity` is decremented by one.
#[allow(clippy::too_many_arguments)]
pub fn ckks_coeffs_to_slots_repack<BE, P, Dst, Src, H, K>(
    module: &Module<BE>,
    ct_out: &mut Dst,
    ct_in: &Src,
    dft: &DFTMatrix<BE, Encode, Repack, LinearTransformation<P>>,
    keys: &H,
    conj_key: &K,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: Backend,
    P: DiagonalProd<BE> + LtDiagonalScale + IntPolyInfos,
    Module<BE>: CKKSLinearTransformationOps<BE>
        + CnvPVecAlloc<BE>
        + CKKSModuleAlloc<BE>
        + CKKSCopyOps<BE>
        + CKKSConjugateOps<BE>
        + CKKSAddOps<BE>
        + CKKSSubOps<BE>
        + CKKSImagOps<BE>
        + CKKSRotateOps<BE>,
    Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    Src: GLWEToBackendRef<BE> + CKKSCtBounds,
    K: CKKSAtkBounds<BE>,
    H: GLWEAutomorphismKeyHelper<K, BE>,
{
    let slots = 1i64 << dft.plan().log_slots();

    // ct_out := z = Encode(ct_in).
    module.ckks_copy(ct_out, ct_in, scratch)?;
    ckks_dft_evaluate_assign(module, ct_out, dft, keys, scratch)?;

    // conj := conj(z); imag := −i·(z − conj) = 2·Im(z); ct_out := z + conj = 2·Re(z).
    let mut conj = module.ckks_ciphertext_alloc_from_infos(ct_out);
    module.ckks_conjugate_into(&mut conj, ct_out, conj_key, scratch)?;
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
pub fn ckks_slots_to_coeffs_repack<BE, P, Dst, Src, H, K>(
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
    Module<BE>: CKKSLinearTransformationOps<BE> + CnvPVecAlloc<BE> + CKKSCopyOps<BE>,
    Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    Src: GLWEToBackendRef<BE> + CKKSCtBounds,
    K: CKKSAtkBounds<BE>,
    H: GLWEAutomorphismKeyHelper<K, BE>,
{
    module.ckks_copy(op_out, ct_in, scratch)?;
    ckks_dft_evaluate_assign(module, op_out, dft, keys, scratch)?;

    // The repack-decode halves the live slot count.
    op_out.set_log_sparsity(ct_in.log_sparsity() + 1);
    Ok(())
}

#[cfg(test)]
mod validation_tests {
    use crate::layouts::{DFTOutputFormat, DFTType};

    use super::*;

    fn plan(factorization_depth: Vec<usize>) -> anyhow::Result<DFTPlan> {
        DFTPlan::new(
            DFTType::Encode,
            factorization_depth.into_iter().map(|d| (d, 1)).collect::<Vec<_>>(),
            DFTOutputFormat::Standard,
            crate::CoeffsMeta::from_delta_budget(0, 0),
        )
    }

    #[test]
    fn malformed_dft_dimensions_are_rejected() {
        // An overflowing layer sum never yields a plan: `DFTPlan::new` rejects it.
        assert!(plan(vec![usize::MAX, 1]).is_err());

        // A valid plan can still be too wide for a given ring.
        let error =
            checked_dft_log_slots(&plan(vec![4]).unwrap(), 4).expect_err("a DFT cannot expose more slots than its CKKS ring");
        assert!(error.to_string().contains("supports at most"), "unexpected error: {error:#}");
    }
}
