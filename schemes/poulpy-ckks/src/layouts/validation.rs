//! Generic key- and operand-layout validation shared by the bootstrapping pipelines and the linear transformations.
//!
//! These helpers check the structural invariants (degree, rank, radix, decomposition shape, storage capacity, backend-view consistency) that any gadget key or runtime ciphertext operand must satisfy before entering the arithmetic primitives, independent of which pipeline (PaCo or classic bootstrapping) consumes them, and the uniformity of a linear transformation's diagonals.
//! Pipeline-specific validation (e.g. PaCo's plan-against-material checks) stays with its pipeline; this module is the reusable bottom layer.

use anyhow::{Context, Result, ensure};
use poulpy_core::{
    LinearTransformationPrepared,
    layouts::{Base2K, GGLWEInfos, GLWEInfos, GLWEToBackendRef, LWEInfos, LinearTransformation},
};
use poulpy_hal::layouts::Backend;

use crate::{CKKSCompositionError, SlotsKind};

/// `InvalidOperand` for `op` unless `prepared` was allocated for `lt`'s layout: the same schedule, every diagonal of
/// both at one degree, radix, rank, width and limb count, and `lt`'s baby steps led by the identity rotation.
pub(crate) fn ensure_prepared_transformation_layout<BE: Backend, P: GLWEInfos>(
    op: &'static str,
    prepared: &LinearTransformationPrepared<BE>,
    lt: &LinearTransformation<P>,
) -> crate::CKKSResult<()> {
    let invalid = |reason: &str| -> crate::CKKSError {
        CKKSCompositionError::InvalidOperand {
            op,
            reason: reason.into(),
        }
        .into()
    };
    if prepared.index() != lt.index() {
        return Err(invalid("the transformation's schedule differs from the prepared one's"));
    }
    let Some(first) = prepared.first_diagonal_plaintext() else {
        return Err(invalid("the prepared transformation has no diagonal"));
    };
    let layout = (first.n(), first.base2k(), first.rank(), first.k(), first.max_size());
    let prepared_layouts = prepared.giant_steps.iter().flat_map(|gs| &gs.diagonals).map(|d| {
        (
            d.plaintext.n(),
            d.plaintext.base2k(),
            d.plaintext.rank(),
            d.plaintext.k(),
            d.plaintext.max_size(),
        )
    });
    let source_layouts = lt.giant_steps.iter().flat_map(|gs| &gs.diagonals).map(|d| {
        (
            d.plaintext.n(),
            d.plaintext.base2k(),
            d.plaintext.rank(),
            d.plaintext.k(),
            d.plaintext.max_size(),
        )
    });
    if prepared_layouts.chain(source_layouts).any(|diagonal| diagonal != layout) {
        return Err(invalid("a diagonal's layout differs from the prepared transformation's"));
    }
    if lt.baby_steps.first().is_some_and(|&baby| baby != 0) {
        return Err(invalid("the baby steps do not start with the identity rotation"));
    }
    Ok(())
}

/// `InvalidOperand` for `op` unless every diagonal of `lt` shares the first one's scale, storage width and slot kind,
/// the `(log_scale, max_k, slots)` that `meta` reads.
///
/// The evaluation derives one `cnv_offset` and the result metadata from the first diagonal alone, and the preparation
/// stashes the first diagonal's scale and slot kind for all, so a hand-built transform with another scale on a later
/// diagonal would mis-scale its contribution and one with another slot kind would mislabel the result.
pub(crate) fn ensure_uniform_diagonals<P>(
    op: &'static str,
    lt: &LinearTransformation<P>,
    meta: impl Fn(&P) -> (usize, usize, SlotsKind),
) -> crate::CKKSResult<()> {
    let Some(first) = lt.first_diagonal_plaintext() else {
        return Ok(());
    };
    let (log_scale, max_k, slots) = meta(first);
    for gs in &lt.giant_steps {
        for diag in &gs.diagonals {
            let (diag_log_scale, diag_max_k, diag_slots) = meta(&diag.plaintext);
            if (diag_log_scale, diag_max_k, diag_slots) != (log_scale, max_k, slots) {
                return Err(CKKSCompositionError::InvalidOperand {
                    op,
                    reason: format!(
                        "linear transformation diagonals are not uniform: diagonal (giant rot {}, baby {}) has (log_scale {diag_log_scale}, max_k {diag_max_k}, {diag_slots:?}) but the first diagonal has ({log_scale}, {max_k}, {slots:?})",
                        gs.rot, diag.baby,
                    ),
                }
                .into());
            }
        }
    }
    Ok(())
}

/// Checks degree/rank/radix compatibility shared by all rank-1 gadget keys.
pub(crate) fn validate_gadget_key<K: GGLWEInfos + ?Sized>(
    name: &str,
    key: &K,
    n: usize,
    base2k: Base2K,
    working_size: usize,
) -> Result<()> {
    ensure!(
        key.n().as_usize() == n,
        "{name} degree {} does not match the expected degree {n}",
        key.n()
    );
    ensure!(
        key.rank_in().as_usize() == 1 && key.rank_out().as_usize() == 1,
        "{name} must have rank_in=rank_out=1, got rank_in={} and rank_out={}",
        key.rank_in(),
        key.rank_out(),
    );
    ensure!(
        key.base2k() == base2k,
        "{name} base2k {} does not match the expected base2k {base2k}",
        key.base2k(),
    );
    ensure!(key.dnum().as_usize() > 0, "{name} must have at least one decomposition digit");
    ensure!(key.k_digit().as_usize() > 0, "{name} decomposition digits must be non-empty");
    ensure!(
        key.k().as_usize() > key.k_digit().as_usize(),
        "{name} decomposition digit of {} bits must be narrower than the {}-bit key",
        key.k_digit(),
        key.k(),
    );
    let gadget_k = key
        .dnum()
        .as_usize()
        .checked_mul(key.k_digit().as_usize())
        .with_context(|| format!("{name} decomposition size overflows usize"))?;
    ensure!(
        gadget_k <= key.k().as_usize(),
        "{name} decomposition covers {gadget_k} bits, exceeding its {}-bit key",
        key.k(),
    );
    let required_size = working_size
        .checked_add(key.dsize().as_usize())
        .with_context(|| format!("{name} required working width overflows usize"))?;
    ensure!(
        key.max_size() >= required_size,
        "{name} stores {} limbs, but operations require at least {required_size} ({working_size} working + {} guard)",
        key.max_size(),
        key.dsize(),
    );
    validate_storage_capacity(name, key)
}

/// Validates a custom key manager's reported gadget-key metadata against the
/// backend-native view that the arithmetic primitives actually consume.
///
/// Built-in prepared key wrappers forward these fields directly. Checking
/// both representations keeps safe key-manager extension points (e.g.
/// [`PaCoKeys`](crate::layouts::PaCoKeys)) from turning inconsistent
/// lazy-store metadata into assertion failures below the CKKS composition
/// layer.
pub(crate) fn validate_gadget_backend_view<R, V>(
    name: &str,
    reported: &R,
    view: &V,
    n: usize,
    base2k: Base2K,
    working_size: usize,
) -> Result<()>
where
    R: GGLWEInfos + ?Sized,
    V: GGLWEInfos + ?Sized,
{
    validate_gadget_key(name, reported, n, base2k, working_size)?;
    let view_name = format!("{name} backend view");
    validate_gadget_key(&view_name, view, n, base2k, working_size)?;
    ensure!(
        reported.n() == view.n()
            && reported.base2k() == view.base2k()
            && reported.k() == view.k()
            && reported.max_size() == view.max_size()
            && reported.rank_in() == view.rank_in()
            && reported.rank_out() == view.rank_out()
            && reported.dnum() == view.dnum()
            && reported.k_digit() == view.k_digit(),
        "{name} reported layout does not match its backend-native view",
    );
    Ok(())
}

/// Rejects metadata that claims more torus precision than the backing
/// allocation can hold. Active-limb sufficiency (a backend view truncated
/// below the metadata) is checked separately by
/// [`validate_backend_storage_capacity`].
pub(crate) fn validate_storage_capacity<K: LWEInfos + ?Sized>(name: &str, key: &K) -> Result<()> {
    let base2k = key.base2k().as_usize();
    ensure!((1..=63).contains(&base2k), "{name} base2k must be in [1, 63], got {base2k}");
    let capacity = key
        .max_size()
        .checked_mul(base2k)
        .with_context(|| format!("{name} storage capacity overflows usize"))?;
    ensure!(
        key.k().as_usize() <= capacity,
        "{name} torus width {} exceeds its {capacity}-bit storage capacity",
        key.k(),
    );
    Ok(())
}

/// Validates both allocated capacity and active backend limbs for a runtime
/// ciphertext-like operand. This catches truncated or malformed custom key
/// stores whose metadata still advertises more precision than their active
/// view exposes.
pub(crate) fn validate_backend_storage_capacity<BE, K>(name: &str, key: &K) -> Result<()>
where
    BE: Backend,
    K: GLWEInfos + GLWEToBackendRef<BE>,
{
    validate_storage_capacity(name, key)?;
    let view = key.to_backend_ref();
    ensure!(
        key.n() == view.n()
            && key.base2k() == view.base2k()
            && key.k() == view.k()
            && key.max_size() == view.max_size()
            && key.rank() == view.rank(),
        "{name} reported layout does not match its backend-native view",
    );
    let active = view.data().size();
    ensure!(
        active >= key.size(),
        "{name} exposes {active} active limbs, but its torus width requires {}",
        key.size(),
    );
    Ok(())
}
