//! Production helper: build a [`LinearTransformation<CKKSPlaintext>`] from a
//! raw [`ComplexDiagonals`] map.
//!
//! Performs BSGS pre-rotation + giant-step bucketing inside the
//! scheme-agnostic [`ComplexDiagonals::build_transform`] and only does the
//! per-diagonal backend-native encode here. Callers supply *raw* matrix
//! diagonals — `diag_i[j] = M[j][(j+i) mod cols]` — with no manual
//! `rot(diag, -n1·j)` pre-rotation needed.

use anyhow::{Context, Result, ensure};
use poulpy_core::layouts::{
    Base2K, DiagonalArithmetic, GLWELayout, GLWEPlaintextInfos, GLWEPlaintextMeta, LinearTransformationLayout,
    LinearTransformationStrategy, Rank,
};
use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use crate::api::CKKSModuleInfos;
use crate::{
    CKKSCompositionError, CKKSLayout, CKKSResult, CoeffsMeta,
    api::{CKKSEncodingHostOps, CKKSEncodingOps, CKKSEncodingScalar, LinearTransformation},
    ckks_ensure,
    layouts::{CKKSModuleAlloc, CKKSPlaintextOwned, ComplexDiagonals, slots_degree},
};

fn validate_compile_inputs<F>(
    full_slots: usize,
    diagonals: &ComplexDiagonals<F>,
    strategy: LinearTransformationStrategy,
) -> Result<()> {
    let slots = diagonals.re.slots();
    ensure!(
        slots == diagonals.im.slots(),
        "complex diagonal maps have different slot counts: real {slots}, imaginary {}",
        diagonals.im.slots(),
    );
    ensure!(
        !diagonals.indexes().is_empty(),
        "linear transformation must contain at least one diagonal"
    );
    ensure!(
        slots <= full_slots && full_slots.is_multiple_of(slots),
        "linear-transformation slot count {slots} must divide the ring's {full_slots} CKKS slots",
    );
    if let LinearTransformationStrategy::Bsgs { giant_step } = strategy {
        ensure!(giant_step > 0, "linear-transformation giant step must be non-zero");
    }
    Ok(())
}

/// Encodes the complex linear transformation represented by `diagonals` into a
/// [`LinearTransformation<CKKSPlaintext>`] ready for the
/// [`CKKSLinearTransformationOps`](crate::api::CKKSLinearTransformationOps) pipeline.
///
/// Matrix coefficients are generated as host slices, but their slot transform
/// and plaintext mapping are performed by `module`'s CKKS encoding backend.
/// A device backend therefore uploads the slices into its arena and runs its
/// native IFFT and coefficient codec without constructing a host module.
///
/// `strategy` selects the BSGS schedule (`Auto` / `Bsgs { giant_step }` /
/// `Direct`). The resulting transform reports `lt.galois_elements(cyclotomic_order)` so the
/// caller knows which Galois keys to generate.
///
/// `transpose` selects the orientation of the matrix-vector product applied to
/// the encrypted input `a`:
/// - `false`: encode `B` as-is — engine produces `B·a` (matrix on the left).
/// - `true`:  encode `Bᵀ` via [`ComplexDiagonals::transpose`] — engine produces
///   `Bᵀ·a = a·B` (matrix on the right of the row vector `a`).
///
/// # Errors
///
/// Returns an error when the CKKS layout is inconsistent, a diagonal cannot
/// be represented at the requested scale, the diagonal map or BSGS schedule
/// is malformed, the backend cannot encode a generated diagonal, or the
/// caller-provided arena is too small. A zero torus width, `coeffs_meta.k = 0`,
/// is [`CKKSCompositionError::InvalidOperand`], returned before anything is
/// allocated or written.
///
/// # Migration
///
/// This compilation helper is now fallible; callers of the previous
/// infallible signature must propagate or otherwise handle its [`Result`].
/// This removes the former panic on malformed layouts and encoding failures.
#[allow(
    clippy::too_many_arguments,
    reason = "linear-transformation compilation needs the encoding layout, matrix, schedule, and arena"
)]
pub fn ckks_encode_linear_transformation_from_diagonals<BE, F>(
    module: &Module<BE>,
    base2k: Base2K,
    coeffs_meta: CoeffsMeta,
    diagonals: &ComplexDiagonals<F>,
    strategy: LinearTransformationStrategy,
    transpose: bool,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<LinearTransformation<CKKSPlaintextOwned<BE>>>
where
    BE: Backend,
    Module<BE>: CKKSModuleAlloc<BE> + CKKSEncodingOps<BE, F>,
    F: DiagonalArithmetic + CKKSEncodingScalar,
{
    coeffs_meta.ensure_width("ckks_encode_linear_transformation_from_diagonals")?;
    ensure!(
        (1..=63).contains(&base2k.as_usize()),
        "linear-transformation base2k must be in [1, 63], got {base2k}",
    );
    validate_compile_inputs(module.ckks_max_slots(), diagonals, strategy)?;

    let owned_transposed;
    let cd: &ComplexDiagonals<F> = if transpose {
        let mut t = diagonals.clone();
        t.transpose();
        owned_transposed = t;
        &owned_transposed
    } else {
        diagonals
    };
    let layout = LinearTransformationLayout {
        indexes: cd.indexes(),
        slots: cd.slots(),
        strategy,
    };
    let infos = CKKSLayout {
        glwe_layout: GLWELayout {
            n: slots_degree::<BE, _>(module, cd.period()),
            base2k,
            k: coeffs_meta.k,
            rank: Rank(0),
        },
        meta: coeffs_meta.meta,
    };
    let mut lt = module.ckks_linear_transformation_alloc_from_infos(&layout, &infos);
    ckks_encode_linear_transformation_into(module, &mut lt, coeffs_meta.meta, cd, strategy, scratch)
        .context("cannot encode a linear-transformation diagonal")?;
    Ok(lt)
}

/// Encodes `diagonals` into `res`, allocated by
/// [`CKKSModuleAlloc::ckks_linear_transformation_alloc_from_infos`] for their layout under `strategy`: every diagonal at
/// `meta`, in the schedule's order, each plaintext overwritten, so encoding again into the same transform refills it.
///
/// # Errors
///
/// The errors of [`ckks_encode_linear_transformation_from_diagonals`] for malformed `diagonals` or `strategy`, and
/// [`CKKSCompositionError::InvalidOperand`] when the schedule of `res` is not the one they give, before anything is
/// written; then the encoding's errors, after which the content of `res` is unspecified.
pub fn ckks_encode_linear_transformation_into<BE, M, F>(
    module: &M,
    res: &mut LinearTransformation<CKKSPlaintextOwned<BE>>,
    meta: GLWEPlaintextMeta,
    diagonals: &ComplexDiagonals<F>,
    strategy: LinearTransformationStrategy,
    scratch: &mut ScratchArena<'_, BE>,
) -> CKKSResult<()>
where
    BE: Backend,
    M: CKKSModuleInfos + CKKSEncodingOps<BE, F>,
    F: DiagonalArithmetic + CKKSEncodingScalar,
{
    validate_compile_inputs(module.ckks_max_slots(), diagonals, strategy)?;
    let plan = LinearTransformationLayout {
        indexes: diagonals.indexes(),
        slots: diagonals.slots(),
        strategy,
    }
    .index();
    let same_schedule = res.baby_steps == plan.baby_steps
        && res.giant_steps.len() == plan.giant_steps.len()
        && res
            .giant_steps
            .iter()
            .zip(plan.giant_steps.iter().zip(&plan.index))
            .all(|(gs, (&rot, babies))| gs.rot == rot && gs.diagonals.iter().map(|d| d.baby).eq(babies.iter().copied()));
    if !same_schedule {
        return Err(CKKSCompositionError::InvalidOperand {
            op: "ckks_encode_linear_transformation_into",
            reason: "the transform's schedule is not the diagonals'".into(),
        }
        .into());
    }
    let complex = diagonals.im.indexes().iter().any(|&index| {
        diagonals
            .im
            .get(index)
            .is_some_and(|values| values.iter().any(|value| !value.is_zero()))
    });
    let mut targets = res.giant_steps.iter_mut().flat_map(|gs| &mut gs.diagonals);
    let mut result = Ok(());
    // The builder visits the diagonals in the schedule's order, the one checked above.
    diagonals.build_transform(strategy, |re, im| {
        if let (Ok(()), Some(dst)) = (&result, targets.next()) {
            result = encode_diagonal(module, &mut dst.plaintext, meta, complex, re, im, scratch);
        }
    });
    result
}

/// Encodes `re + i im` into `pt` at `meta`.
fn encode_diagonal<BE, M, F>(
    module: &M,
    pt: &mut CKKSPlaintextOwned<BE>,
    meta: GLWEPlaintextMeta,
    complex: bool,
    re: &[F],
    im: &[F],
    scratch: &mut ScratchArena<'_, BE>,
) -> CKKSResult<()>
where
    BE: Backend,
    M: CKKSEncodingOps<BE, F>,
    F: CKKSEncodingScalar,
{
    pt.set_meta_checked(meta)?;
    module.ckks_encode_reim_into(pt, re, im, scratch)?;
    // Real slots after encoding complex diagonals mean the imaginary parts were dropped.
    ckks_ensure!(
        !complex || !pt.slots().is_real(),
        "complex linear-transformation diagonals cannot be encoded into real slots"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use poulpy_core::layouts::Diagonals;

    use super::*;

    #[test]
    fn malformed_diagonal_layout_is_rejected_before_schedule_construction() {
        let mut re = Diagonals::new(4);
        re.set(0, vec![1.0; 4]);
        let mismatched = ComplexDiagonals {
            re,
            im: Diagonals::new(8),
        };
        assert!(validate_compile_inputs(8, &mismatched, LinearTransformationStrategy::Direct).is_err());

        let mut re = Diagonals::new(4);
        re.set(0, vec![1.0; 4]);
        let valid = ComplexDiagonals::new(re, Diagonals::new(4));
        assert!(validate_compile_inputs(8, &valid, LinearTransformationStrategy::Bsgs { giant_step: 0 },).is_err());
    }
}
