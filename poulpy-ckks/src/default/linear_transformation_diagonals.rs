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
    Base2K, DiagonalArithmetic, LinearTransformationDiagonal, LinearTransformationGiantStep, LinearTransformationStrategy,
};
use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use crate::{
    CoeffsMeta,
    api::{CKKSEncodingHostOps, CKKSEncodingOps, CKKSEncodingScalar, LinearTransformation},
    layouts::{CKKSModuleAlloc, ComplexDiagonals, plaintext::CKKSPlaintext},
};

fn validate_compile_inputs<F>(
    module_n: usize,
    base2k: Base2K,
    diagonals: &ComplexDiagonals<F>,
    strategy: LinearTransformationStrategy,
) -> Result<()> {
    ensure!(
        (1..=63).contains(&base2k.as_usize()),
        "linear-transformation base2k must be in [1, 63], got {base2k}",
    );
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
    let full_slots = module_n / 2;
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
/// caller-provided arena is too small.
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
) -> Result<LinearTransformation<CKKSPlaintext<BE::OwnedBuf>>>
where
    BE: Backend,
    Module<BE>: CKKSModuleAlloc<BE> + CKKSEncodingOps<BE, F>,
    F: DiagonalArithmetic + CKKSEncodingScalar,
{
    validate_compile_inputs(module.n(), base2k, diagonals, strategy)?;

    encode_linear_transformation_from_diagonals(module, base2k, coeffs_meta, diagonals, strategy, transpose, scratch)
}

/// Shared implementation of the public fallible compilation path.
#[allow(
    clippy::too_many_arguments,
    reason = "fallible compilation mirrors the established linear-transformation helper's explicit inputs"
)]
fn encode_linear_transformation_from_diagonals<BE, F>(
    module: &Module<BE>,
    base2k: Base2K,
    coeffs_meta: CoeffsMeta,
    diagonals: &ComplexDiagonals<F>,
    strategy: LinearTransformationStrategy,
    transpose: bool,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<LinearTransformation<CKKSPlaintext<BE::OwnedBuf>>>
where
    BE: Backend,
    Module<BE>: CKKSModuleAlloc<BE> + CKKSEncodingOps<BE, F>,
    F: DiagonalArithmetic + CKKSEncodingScalar,
{
    let owned_transposed;
    let cd: &ComplexDiagonals<F> = if transpose {
        let mut t = diagonals.clone();
        t.transpose();
        owned_transposed = t;
        &owned_transposed
    } else {
        diagonals
    };
    let encoded = cd.build_transform(strategy, |pre_re, pre_im| -> Result<_> {
        let mut pt = module.ckks_pt_vec_alloc(base2k, coeffs_meta.k);
        pt.set_meta_checked(coeffs_meta.meta)?;
        module
            .ckks_encode_reim_into(&mut pt, pre_re, pre_im, scratch)
            .context("cannot encode a linear-transformation diagonal")?;
        Ok(pt)
    });

    let giant_steps = encoded
        .giant_steps
        .into_iter()
        .map(|step| {
            let diagonals = step
                .diagonals
                .into_iter()
                .map(|diagonal| {
                    Ok(LinearTransformationDiagonal {
                        baby: diagonal.baby,
                        plaintext: diagonal.plaintext?,
                    })
                })
                .collect::<Result<_>>()?;
            Ok(LinearTransformationGiantStep {
                rot: step.rot,
                diagonals,
            })
        })
        .collect::<Result<_>>()?;
    Ok(LinearTransformation {
        baby_steps: encoded.baby_steps,
        giant_steps,
    })
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
        assert!(validate_compile_inputs(8, Base2K(2), &mismatched, LinearTransformationStrategy::Direct).is_err());

        let mut re = Diagonals::new(4);
        re.set(0, vec![1.0; 4]);
        let valid = ComplexDiagonals::new(re, Diagonals::new(4));
        assert!(validate_compile_inputs(8, Base2K(2), &valid, LinearTransformationStrategy::Bsgs { giant_step: 0 },).is_err());
    }
}
