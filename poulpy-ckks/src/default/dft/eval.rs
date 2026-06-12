//! Homomorphic (I)DFT generation and evaluation.
//!
//! Builds the prepared factor operands of a [`DFTMatrix`] from a
//! [`DFTMatrixLiteral`] (encode each generated [`ComplexDiagonals`] factor into a
//! CKKS linear transformation, then prepare its right operand), and evaluates the
//! transform by chaining one prepared linear transformation per factor.
//!
//! Scale accounting (see `docs/ckks_dft.md` §6). poulpy's torus plaintext-multiply
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

use anyhow::Result;
use poulpy_core::layouts::{
    Base2K, GGLWEInfos, GGLWEPreparedToBackendRef, GLWEAutomorphismKeyHelper, GLWEToBackendMut, GLWEToBackendRef,
    GetGaloisElement, LinearTransformationStrategy, prepared::GLWEAutomorphismKeyPreparedToBackendRef,
};
use poulpy_hal::{
    api::{CnvPVecAlloc, ModuleNew, NegacyclicFFT},
    layouts::{Backend, HostBytesBackend, Module, ScratchArena, TransferFrom},
};

use crate::{
    CKKSCtBounds, CKKSMeta, SetCKKSInfos,
    api::{
        CKKSAddOps, CKKSConjugateOps, CKKSCopyOps, CKKSImagOps, CKKSRotateOps, CKKSSubOps, LinearTransformationOps,
        PreparedLinearTransformationLhs, PreparedLinearTransformationRhs,
    },
    default::dft::matrices::gen_dft_matrices,
    encoding::reim::Encoder,
    layouts::{CKKSModuleAlloc, CKKSPlaintextVecHostCodec, DFTFormat, DFTMatrix, DFTMatrixLiteral, DFTType},
};

/// Builds the prepared homomorphic (I)DFT described by `literal`.
///
/// Each generated factor matrix is encoded into a CKKS linear transformation at
/// scale `factor_meta.log_delta` and prepared into its right operand. Evaluating
/// the resulting [`DFTMatrix`] consumes `factor_meta.log_delta` bits of
/// `log_budget` per factor.
#[allow(clippy::too_many_arguments)]
pub fn ckks_new_dft_matrix<BE, E>(
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
    encoder: &Encoder<E>,
    base2k: Base2K,
    factor_meta: CKKSMeta,
    literal: &DFTMatrixLiteral,
    strategy: LinearTransformationStrategy,
    scratch: &mut ScratchArena<'_, BE>,
) -> DFTMatrix<BE>
where
    BE: Backend + TransferFrom<HostBytesBackend>,
    Module<BE>: CnvPVecAlloc<BE> + LinearTransformationOps<BE> + CKKSModuleAlloc<BE>,
    Module<HostBytesBackend>: ModuleNew<HostBytesBackend> + CKKSModuleAlloc<HostBytesBackend>,
    E: NegacyclicFFT<f64> + poulpy_hal::api::NegacyclicFFTNew<f64>,
    crate::layouts::CKKSPlaintext<Vec<u8>>: CKKSPlaintextVecHostCodec<f64>,
{
    literal.check().expect("invalid DFTMatrixLiteral");
    let factors_cd = gen_dft_matrices::<f64>(literal, module.log_n());

    // Sparse repack: the diagonals live in the `dslots = 2·slots` sub-ring and are
    // gap-encoded into the degree-`N` plaintext via a `dslots`-slot encoder; the
    // homomorphic rotations stay in the small slot space.
    let sparse = literal.log_slots < module.log_n().saturating_sub(1) && literal.format == DFTFormat::RepackImagAsReal;
    let dslots = 2usize << literal.log_slots; // 2·slots
    let sparse_encoder = sparse.then(|| Encoder::<E>::new::<f64>(dslots).expect("dslots is a power of two"));
    let enc_ref = sparse_encoder.as_ref().unwrap_or(encoder);

    let mut factors = Vec::with_capacity(factors_cd.len());
    for cd in &factors_cd {
        // Encode this factor's complex diagonals into a CKKS linear transformation.
        let lt = crate::default::ckks_encode_linear_transformation_from_diagonals::<BE, f64, E>(
            module,
            host_module,
            enc_ref,
            base2k,
            factor_meta,
            cd,
            strategy,
            false,
            sparse,
        );

        // Allocate + populate the prepared right operand from the encoded factor.
        let first_pt = lt
            .giant_steps
            .iter()
            .flat_map(|gs| gs.diagonals.iter())
            .map(|d| &d.plaintext)
            .next()
            .expect("dft factor has no diagonals");
        let mut prepared = PreparedLinearTransformationRhs::alloc_from_index(module, &lt.index(), first_pt);
        module.ckks_prepare_linear_transformation_rhs(&mut prepared, &lt, scratch);
        factors.push(prepared);
    }

    DFTMatrix {
        literal: literal.clone(),
        factors,
        factor_log_delta: factor_meta.log_delta,
        sparse,
    }
}

/// Evaluates the homomorphic (I)DFT `dft` in place on `ct`.
///
/// Chains one prepared linear transformation per factor, preparing the baby
/// rotations of the running ciphertext each time. No explicit rescale is needed
/// (the plaintext-multiply realigns to the input `log_delta`, see the module
/// docs). The input `ct.log_budget()` must be at least `dft.consumed_bits()`.
pub fn ckks_dft_evaluate_assign<BE, Dst, H, K>(
    module: &Module<BE>,
    ct: &mut Dst,
    dft: &DFTMatrix<BE>,
    keys: &H,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: Backend,
    Module<BE>: LinearTransformationOps<BE> + CnvPVecAlloc<BE>,
    Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    K: GLWEAutomorphismKeyPreparedToBackendRef<BE> + GGLWEPreparedToBackendRef<BE> + GetGaloisElement + GGLWEInfos,
    H: GLWEAutomorphismKeyHelper<K, BE>,
{
    for factor in &dft.factors {
        let mut babies = PreparedLinearTransformationLhs::alloc(module, factor.baby_steps(), ct);
        module.ckks_prepare_linear_transformation_lhs(&mut babies, ct, keys, scratch)?;
        module.ckks_eval_prepared_linear_transformation_assign(ct, factor, &babies, keys, scratch)?;
    }
    Ok(())
}

/// Homomorphic encoding (CoeffsToSlots), `Standard` format: evaluates the Encode
/// (IDFT) matrix in place. `dft.literal.kind` must be [`DFTType::Encode`] and the
/// format [`DFTFormat::Standard`] (the real/imag-splitting formats are a later
/// increment).
pub fn ckks_coeffs_to_slots_assign<BE, Dst, H, K>(
    module: &Module<BE>,
    ct: &mut Dst,
    dft: &DFTMatrix<BE>,
    keys: &H,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: Backend,
    Module<BE>: LinearTransformationOps<BE> + CnvPVecAlloc<BE>,
    Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    K: GLWEAutomorphismKeyPreparedToBackendRef<BE> + GGLWEPreparedToBackendRef<BE> + GetGaloisElement + GGLWEInfos,
    H: GLWEAutomorphismKeyHelper<K, BE>,
{
    assert_eq!(dft.literal.kind, DFTType::Encode, "coeffs_to_slots requires an Encode matrix");
    assert_eq!(
        dft.literal.format,
        DFTFormat::Standard,
        "only the Standard format is wired through coeffs_to_slots so far"
    );
    ckks_dft_evaluate_assign(module, ct, dft, keys, scratch)
}

/// Homomorphic decoding (SlotsToCoeffs), `Standard` format: evaluates the Decode
/// (DFT) matrix in place. `dft.literal.kind` must be [`DFTType::Decode`].
pub fn ckks_slots_to_coeffs_assign<BE, Dst, H, K>(
    module: &Module<BE>,
    ct: &mut Dst,
    dft: &DFTMatrix<BE>,
    keys: &H,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: Backend,
    Module<BE>: LinearTransformationOps<BE> + CnvPVecAlloc<BE>,
    Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    K: GLWEAutomorphismKeyPreparedToBackendRef<BE> + GGLWEPreparedToBackendRef<BE> + GetGaloisElement + GGLWEInfos,
    H: GLWEAutomorphismKeyHelper<K, BE>,
{
    assert_eq!(dft.literal.kind, DFTType::Decode, "slots_to_coeffs requires a Decode matrix");
    assert_eq!(
        dft.literal.format,
        DFTFormat::Standard,
        "only the Standard format is wired through slots_to_coeffs so far"
    );
    ckks_dft_evaluate_assign(module, ct, dft, keys, scratch)
}

/// `CoeffsToSlots` with the real and imaginary parts returned in two separate
/// real-vector ciphertexts (`DFTFormat::SplitRealAndImag`, dense packing).
///
/// Evaluates the Encode matrix, then splits `z` into `2·Re(z)` and `2·Im(z)` with
/// a conjugation
/// (`z + z̄` and `−i·(z − z̄)`); the matrix's `1/(2·slots)` scaling makes the net
/// result `Re` / `Im` of the slot DFT. `conj_key` is the conjugation automorphism
/// key (Galois element `−1`). On return, `ct_real` holds the real parts and
/// `ct_imag` the imaginary parts. Consumes `ct_in` by reference (copied).
#[allow(clippy::too_many_arguments)]
pub fn ckks_coeffs_to_slots_split<BE, Dst, Src, H, K>(
    module: &Module<BE>,
    ct_real: &mut Dst,
    ct_imag: &mut Dst,
    ct_in: &Src,
    dft: &DFTMatrix<BE>,
    keys: &H,
    conj_key: &K,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: Backend,
    Module<BE>: LinearTransformationOps<BE>
        + CnvPVecAlloc<BE>
        + CKKSModuleAlloc<BE>
        + CKKSCopyOps<BE>
        + CKKSConjugateOps<BE>
        + CKKSAddOps<BE>
        + CKKSSubOps<BE>
        + CKKSImagOps<BE>,
    Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    Src: GLWEToBackendRef<BE> + CKKSCtBounds,
    K: GLWEAutomorphismKeyPreparedToBackendRef<BE> + GGLWEPreparedToBackendRef<BE> + GetGaloisElement + GGLWEInfos,
    H: GLWEAutomorphismKeyHelper<K, BE>,
{
    assert_eq!(dft.literal.kind, DFTType::Encode, "coeffs_to_slots requires an Encode matrix");
    assert!(
        matches!(dft.literal.format, DFTFormat::SplitRealAndImag | DFTFormat::RepackImagAsReal),
        "coeffs_to_slots_split requires a Split/Repack-format matrix"
    );

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
    Ok(())
}

/// `SlotsToCoeffs` from real/imaginary parts in two ciphertexts
/// (`DFTFormat::SplitRealAndImag`, dense packing).
///
/// Combines `ct_real + i·ct_imag`, then evaluates the Decode matrix. Writes the
/// result into `op_out`.
pub fn ckks_slots_to_coeffs_split<BE, Dst, Src, H, K>(
    module: &Module<BE>,
    op_out: &mut Dst,
    ct_real: &Src,
    ct_imag: &Src,
    dft: &DFTMatrix<BE>,
    keys: &H,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: Backend,
    Module<BE>: LinearTransformationOps<BE> + CnvPVecAlloc<BE> + CKKSAddOps<BE> + CKKSImagOps<BE>,
    Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    Src: GLWEToBackendRef<BE> + CKKSCtBounds,
    K: GLWEAutomorphismKeyPreparedToBackendRef<BE> + GGLWEPreparedToBackendRef<BE> + GetGaloisElement + GGLWEInfos,
    H: GLWEAutomorphismKeyHelper<K, BE>,
{
    assert_eq!(dft.literal.kind, DFTType::Decode, "slots_to_coeffs requires a Decode matrix");

    // op_out := ct_real + i·ct_imag, then Decode.
    module.ckks_mul_i_into(op_out, ct_imag, scratch)?;
    module.ckks_add_assign(op_out, ct_real, scratch)?;
    ckks_dft_evaluate_assign(module, op_out, dft, keys, scratch)
}

/// Sparse `CoeffsToSlots` with the imaginary part repacked into the right half of
/// a single ciphertext (`DFTFormat::RepackImagAsReal`, `log_slots < log_max_slots`).
///
/// Evaluates the Encode matrix, splits `z` into `2·Re(z)` / `2·Im(z)`, then rotates
/// the imaginary part by `slots` and adds it into the real part so the result holds
/// `Re` in the left `slots` and `Im` in the right `slots` of each `2·slots` period.
/// The live slot count doubles, so `ct_out.log_sparsity` is decremented by one.
#[allow(clippy::too_many_arguments)]
pub fn ckks_coeffs_to_slots_repack<BE, Dst, Src, H, K>(
    module: &Module<BE>,
    ct_out: &mut Dst,
    ct_in: &Src,
    dft: &DFTMatrix<BE>,
    keys: &H,
    conj_key: &K,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: Backend,
    Module<BE>: LinearTransformationOps<BE>
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
    K: GLWEAutomorphismKeyPreparedToBackendRef<BE> + GGLWEPreparedToBackendRef<BE> + GetGaloisElement + GGLWEInfos,
    H: GLWEAutomorphismKeyHelper<K, BE>,
{
    assert_eq!(dft.literal.kind, DFTType::Encode, "coeffs_to_slots requires an Encode matrix");
    assert!(
        dft.is_sparse(),
        "coeffs_to_slots_repack requires a sparse RepackImagAsReal matrix"
    );
    let slots = 1i64 << dft.literal.log_slots;

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
    Ok(())
}

/// Sparse `SlotsToCoeffs` from a single repacked ciphertext (the inverse of
/// [`ckks_coeffs_to_slots_repack`]; `DFTFormat::RepackImagAsReal`, sparse).
///
/// The Decode matrices already embed the repack matrix that recombines the
/// `[Re | Im]` real packing into the complex form, so this is just an in-place
/// evaluation. The live slot count halves, so `op_out.log_sparsity` is incremented
/// by one.
pub fn ckks_slots_to_coeffs_repack<BE, Dst, Src, H, K>(
    module: &Module<BE>,
    op_out: &mut Dst,
    ct_in: &Src,
    dft: &DFTMatrix<BE>,
    keys: &H,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: Backend,
    Module<BE>: LinearTransformationOps<BE> + CnvPVecAlloc<BE> + CKKSCopyOps<BE>,
    Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    Src: GLWEToBackendRef<BE> + CKKSCtBounds,
    K: GLWEAutomorphismKeyPreparedToBackendRef<BE> + GGLWEPreparedToBackendRef<BE> + GetGaloisElement + GGLWEInfos,
    H: GLWEAutomorphismKeyHelper<K, BE>,
{
    assert_eq!(dft.literal.kind, DFTType::Decode, "slots_to_coeffs requires a Decode matrix");
    assert!(
        dft.is_sparse(),
        "slots_to_coeffs_repack requires a sparse RepackImagAsReal matrix"
    );

    module.ckks_copy(op_out, ct_in, scratch)?;
    ckks_dft_evaluate_assign(module, op_out, dft, keys, scratch)?;

    // The repack-decode halves the live slot count.
    op_out.set_log_sparsity(ct_in.log_sparsity() + 1);
    Ok(())
}

/// `Module`-method surface over the free functions above.
impl<BE: Backend> crate::api::DFTOps<BE> for Module<BE>
where
    Module<BE>: LinearTransformationOps<BE>
        + CnvPVecAlloc<BE>
        + CKKSModuleAlloc<BE>
        + CKKSCopyOps<BE>
        + CKKSConjugateOps<BE>
        + CKKSAddOps<BE>
        + CKKSSubOps<BE>
        + CKKSImagOps<BE>
        + CKKSRotateOps<BE>,
{
    fn ckks_new_dft_matrix<E>(
        &self,
        host_module: &Module<HostBytesBackend>,
        encoder: &Encoder<E>,
        base2k: Base2K,
        factor_meta: CKKSMeta,
        literal: &DFTMatrixLiteral,
        strategy: LinearTransformationStrategy,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> DFTMatrix<BE>
    where
        BE: TransferFrom<HostBytesBackend>,
        Module<HostBytesBackend>: ModuleNew<HostBytesBackend> + CKKSModuleAlloc<HostBytesBackend>,
        E: NegacyclicFFT<f64> + poulpy_hal::api::NegacyclicFFTNew<f64>,
        crate::layouts::CKKSPlaintext<Vec<u8>>: CKKSPlaintextVecHostCodec<f64>,
    {
        ckks_new_dft_matrix(self, host_module, encoder, base2k, factor_meta, literal, strategy, scratch)
    }

    fn ckks_dft_evaluate_assign<Dst, H, K>(
        &self,
        ct: &mut Dst,
        dft: &DFTMatrix<BE>,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        K: GLWEAutomorphismKeyPreparedToBackendRef<BE> + GGLWEPreparedToBackendRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>,
    {
        ckks_dft_evaluate_assign(self, ct, dft, keys, scratch)
    }

    fn ckks_coeffs_to_slots<Dst, H, K>(
        &self,
        ct: &mut Dst,
        dft: &DFTMatrix<BE>,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        K: GLWEAutomorphismKeyPreparedToBackendRef<BE> + GGLWEPreparedToBackendRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>,
    {
        ckks_coeffs_to_slots_assign(self, ct, dft, keys, scratch)
    }

    fn ckks_slots_to_coeffs<Dst, H, K>(
        &self,
        ct: &mut Dst,
        dft: &DFTMatrix<BE>,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        K: GLWEAutomorphismKeyPreparedToBackendRef<BE> + GGLWEPreparedToBackendRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>,
    {
        ckks_slots_to_coeffs_assign(self, ct, dft, keys, scratch)
    }

    fn ckks_coeffs_to_slots_split<Dst, Src, H, K>(
        &self,
        ct_real: &mut Dst,
        ct_imag: &mut Dst,
        ct_in: &Src,
        dft: &DFTMatrix<BE>,
        keys: &H,
        conj_key: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + CKKSCtBounds,
        K: GLWEAutomorphismKeyPreparedToBackendRef<BE> + GGLWEPreparedToBackendRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>,
    {
        ckks_coeffs_to_slots_split(self, ct_real, ct_imag, ct_in, dft, keys, conj_key, scratch)
    }

    fn ckks_slots_to_coeffs_split<Dst, Src, H, K>(
        &self,
        op_out: &mut Dst,
        ct_real: &Src,
        ct_imag: &Src,
        dft: &DFTMatrix<BE>,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + CKKSCtBounds,
        K: GLWEAutomorphismKeyPreparedToBackendRef<BE> + GGLWEPreparedToBackendRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>,
    {
        ckks_slots_to_coeffs_split(self, op_out, ct_real, ct_imag, dft, keys, scratch)
    }

    fn ckks_coeffs_to_slots_repack<Dst, Src, H, K>(
        &self,
        ct_out: &mut Dst,
        ct_in: &Src,
        dft: &DFTMatrix<BE>,
        keys: &H,
        conj_key: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + CKKSCtBounds,
        K: GLWEAutomorphismKeyPreparedToBackendRef<BE> + GGLWEPreparedToBackendRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>,
    {
        ckks_coeffs_to_slots_repack(self, ct_out, ct_in, dft, keys, conj_key, scratch)
    }

    fn ckks_slots_to_coeffs_repack<Dst, Src, H, K>(
        &self,
        op_out: &mut Dst,
        ct_in: &Src,
        dft: &DFTMatrix<BE>,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + CKKSCtBounds,
        K: GLWEAutomorphismKeyPreparedToBackendRef<BE> + GGLWEPreparedToBackendRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>,
    {
        ckks_slots_to_coeffs_repack(self, op_out, ct_in, dft, keys, scratch)
    }
}
