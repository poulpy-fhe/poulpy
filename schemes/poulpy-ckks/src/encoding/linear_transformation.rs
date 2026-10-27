//! Production helper: build a [`LinearTransformation<CKKSPlaintext>`] from a
//! raw [`ComplexDiagonals`] map.
//!
//! Performs BSGS pre-rotation + giant-step bucketing inside the
//! scheme-agnostic [`ComplexDiagonals::build_transform`] and only does the
//! per-diagonal backend-native encode here. Callers supply *raw* matrix
//! diagonals — `diag_i[j] = M[j][(j+i) mod cols]` — with no manual
//! `rot(diag, -n1·j)` pre-rotation needed.

use anyhow::{Context, Result};
use poulpy_core::layouts::{
    Base2K, Degree, DiagonalArithmetic, GLWELayout, GLWEPlaintextMeta, GetDegree, LWEInfos, LinearTransformationLayout,
    LinearTransformationStrategy, Rank,
};
use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use crate::api::CKKSModuleInfos;
use crate::{
    CKKSCompositionError, CKKSLayout, CKKSResult, CoeffsMeta,
    api::{CKKSEncodingHostOps, CKKSEncodingOps, CKKSEncodingScalar, LinearTransformation},
    ckks_ensure_operand,
    layouts::{CKKSModuleAlloc, CKKSPlaintextOwned, ComplexDiagonals},
};

fn validate_compile_inputs<F>(
    op: &'static str,
    full_slots: usize,
    diagonals: &ComplexDiagonals<F>,
    strategy: LinearTransformationStrategy,
) -> CKKSResult<()> {
    let slots = diagonals.re.slots();
    ckks_ensure_operand!(
        op,
        slots == diagonals.im.slots(),
        "complex diagonal maps have different slot counts: real {slots}, imaginary {}",
        diagonals.im.slots(),
    );
    ckks_ensure_operand!(
        op,
        !diagonals.indexes().is_empty(),
        "linear transformation must contain at least one diagonal"
    );
    ckks_ensure_operand!(
        op,
        slots <= full_slots && full_slots.is_multiple_of(slots),
        "linear-transformation slot count {slots} must divide the ring's {full_slots} CKKS slots",
    );
    if let LinearTransformationStrategy::Bsgs { giant_step } | LinearTransformationStrategy::BsgsPowersOfTwo { giant_step } =
        strategy
    {
        ckks_ensure_operand!(op, giant_step > 0, "linear-transformation giant step must be non-zero");
    }
    Ok(())
}

/// `InvalidOperand` for `op` when `diagonals` has a nonzero imaginary part but encodes into real slots: at a `meta` of
/// real slots, or into plaintexts of degree `n` on a ring whose degree-`n` plaintexts hold `n` real slots.
fn ensure_complex_fits<M: CKKSModuleInfos, F: CKKSEncodingScalar>(
    op: &'static str,
    module: &M,
    meta: GLWEPlaintextMeta,
    n: usize,
    diagonals: &ComplexDiagonals<F>,
) -> CKKSResult<()> {
    let complex = diagonals.im.indexes().iter().any(|&index| {
        diagonals
            .im
            .get(index)
            .is_some_and(|values| values.iter().any(|value| !value.is_zero()))
    });
    let real_slots = meta.slots.is_real() || module.ckks_slots_of_degree(n) == n;
    ckks_ensure_operand!(
        op,
        !(complex && real_slots),
        "complex linear-transformation diagonals cannot be encoded into real slots"
    );
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
/// `scratch` holds [`ckks_encode_linear_transformation_tmp_bytes`] of `diagonals.period()`.
///
/// # Errors
///
/// [`CKKSCompositionError::InvalidOperand`], before anything is allocated or
/// written, for a zero torus width (`coeffs_meta.k = 0`), a radix outside
/// `1..=63`, a malformed diagonal map or BSGS schedule, or a nonzero imaginary
/// part encoded into real slots (`coeffs_meta` of real slots, or the
/// conjugate-invariant ring). Then an error when a diagonal cannot be
/// represented at the requested scale, the backend cannot encode a generated
/// diagonal, or the caller-provided arena is too small.
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
    const OP: &str = "ckks_encode_linear_transformation_from_diagonals";
    coeffs_meta.ensure_width(OP)?;
    if !(1..=63).contains(&base2k.as_usize()) {
        return Err(CKKSCompositionError::InvalidOperand {
            op: OP,
            reason: format!("linear-transformation base2k must be in [1, 63], got {base2k}"),
        }
        .into());
    }
    validate_compile_inputs(OP, module.ckks_max_slots(), diagonals, strategy)?;
    let n = period_degree::<BE, _>(module, diagonals.period());
    ensure_complex_fits(OP, module, coeffs_meta.meta, n.as_usize(), diagonals)?;

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
            n,
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

/// Scratch bytes [`ckks_encode_linear_transformation_into`] and [`ckks_encode_linear_transformation_from_diagonals`]
/// need for a map of period at most `period`: the slot encoding of one diagonal, which fills its plaintext, so a period
/// below the backend's minimum degree counts the slots of that degree.
pub fn ckks_encode_linear_transformation_tmp_bytes<BE, M, F>(module: &M, period: usize) -> usize
where
    BE: Backend,
    M: CKKSModuleInfos + GetDegree + CKKSEncodingHostOps<BE, F>,
    F: CKKSEncodingScalar,
{
    let slots = module.ckks_slots_of_degree(period_degree::<BE, M>(module, period).as_usize());
    <M as CKKSEncodingHostOps<BE, F>>::ckks_reim_tmp_bytes(module, slots)
}

/// Encodes `diagonals` into `res`, allocated by
/// [`CKKSModuleAlloc::ckks_linear_transformation_alloc_from_infos`] for their layout under `strategy`: every diagonal at
/// `meta`, in the schedule's order, each plaintext overwritten, so encoding again into the same transform refills it.
/// `scratch` holds [`ckks_encode_linear_transformation_tmp_bytes`] of the period of `diagonals`.
///
/// # Errors
///
/// [`CKKSCompositionError::InvalidOperand`], before anything is written, for malformed `diagonals` or `strategy`, a
/// schedule of `res` other than the one they give, or a nonzero imaginary part encoded into real slots; then the
/// encoding's errors, after which the content of `res` is unspecified.
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
    const OP: &str = "ckks_encode_linear_transformation_into";
    validate_compile_inputs(OP, module.ckks_max_slots(), diagonals, strategy)?;
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
            op: OP,
            reason: "the transform's schedule is not the diagonals'".into(),
        }
        .into());
    }
    if let Some(pt) = res.first_diagonal_plaintext() {
        ensure_complex_fits(OP, module, meta, pt.n().as_usize(), diagonals)?;
    }
    let mut targets = res.giant_steps.iter_mut().flat_map(|gs| &mut gs.diagonals);
    let mut result = Ok(());
    // The builder visits the diagonals in the schedule's order, the one checked above.
    diagonals.build_transform(strategy, |re, im| {
        if let (Ok(()), Some(dst)) = (&result, targets.next()) {
            result = encode_diagonal(module, &mut dst.plaintext, meta, re, im, scratch);
        }
    });
    result
}

/// Encodes `re + i im`, one period, into `pt` at `meta`, repeated over the slots of `pt`.
fn encode_diagonal<BE, M, F>(
    module: &M,
    pt: &mut CKKSPlaintextOwned<BE>,
    meta: GLWEPlaintextMeta,
    re: &[F],
    im: &[F],
    scratch: &mut ScratchArena<'_, BE>,
) -> CKKSResult<()>
where
    BE: Backend,
    M: CKKSModuleInfos + CKKSEncodingOps<BE, F>,
    F: CKKSEncodingScalar,
{
    pt.set_meta_checked(meta)?;
    // A period below the backend's minimum degree fills fewer slots than the plaintext holds.
    let reps = module.ckks_slots_of_degree(pt.n().as_usize()) / re.len();
    if reps > 1 {
        module.ckks_encode_reim_into(pt, &re.repeat(reps), &im.repeat(reps), scratch)?;
    } else {
        module.ckks_encode_reim_into(pt, re, im, scratch)?;
    }
    Ok(())
}

/// The degree of the diagonals of a map of period `period`: `period * N / ckks_max_slots` (`2 * period` on the
/// standard ring, `period` on the invariant ring), raised to the backend's minimum degree and capped at the ring degree.
fn period_degree<BE: Backend, M: CKKSModuleInfos + GetDegree>(module: &M, period: usize) -> Degree {
    assert!(period.is_power_of_two(), "a map's period is a power of two");
    let ring = module.ring_degree().as_usize();
    period
        .checked_mul(ring / module.ckks_max_slots())
        .expect("slot count overflow")
        .max(BE::MIN_DEGREE)
        .min(ring)
        .into()
}

#[cfg(test)]
mod tests {
    use poulpy_core::layouts::Diagonals;

    use super::*;

    #[test]
    fn malformed_diagonal_layout_is_rejected_before_schedule_construction() {
        let refused = |result: CKKSResult<()>, reason: &str| {
            assert_eq!(
                result.unwrap_err().composition(),
                Some(&CKKSCompositionError::InvalidOperand {
                    op: "test",
                    reason: reason.into(),
                })
            );
        };
        let mut re = Diagonals::new(4);
        re.set(0, vec![1.0; 4]);
        let mismatched = ComplexDiagonals {
            re,
            im: Diagonals::new(8),
        };
        refused(
            validate_compile_inputs("test", 8, &mismatched, LinearTransformationStrategy::Direct),
            "complex diagonal maps have different slot counts: real 4, imaginary 8",
        );

        let empty = ComplexDiagonals::new(Diagonals::<f64>::new(4), Diagonals::new(4));
        refused(
            validate_compile_inputs("test", 8, &empty, LinearTransformationStrategy::Direct),
            "linear transformation must contain at least one diagonal",
        );

        let mut re = Diagonals::new(4);
        re.set(0, vec![1.0; 4]);
        let valid = ComplexDiagonals::new(re, Diagonals::new(4));
        refused(
            validate_compile_inputs("test", 2, &valid, LinearTransformationStrategy::Direct),
            "linear-transformation slot count 4 must divide the ring's 2 CKKS slots",
        );
        for strategy in [
            LinearTransformationStrategy::Bsgs { giant_step: 0 },
            LinearTransformationStrategy::BsgsPowersOfTwo { giant_step: 0 },
        ] {
            refused(
                validate_compile_inputs("test", 8, &valid, strategy),
                "linear-transformation giant step must be non-zero",
            );
        }
    }
}
