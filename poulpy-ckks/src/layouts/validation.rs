//! Generic key- and operand-layout validation shared by the bootstrapping pipelines.
//!
//! These helpers check the structural invariants (degree, rank, radix, decomposition shape, storage capacity, backend-view consistency) that any gadget key or runtime ciphertext operand must satisfy before entering the arithmetic primitives, independent of which pipeline (PaCo or classic bootstrapping) consumes them.
//! Pipeline-specific validation (e.g. PaCo's plan-against-material checks) stays with its pipeline; this module is the reusable bottom layer.

use anyhow::{Context, Result, ensure};
use poulpy_core::layouts::{Base2K, GGLWEInfos, GLWEInfos, GLWEToBackendRef, LWEInfos};
use poulpy_hal::layouts::Backend;

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
    ensure!(key.dsize().as_usize() > 0, "{name} decomposition digits must be non-empty");
    let size = key.k().as_usize().div_ceil(base2k.as_usize());
    ensure!(
        size > key.dsize().as_usize(),
        "{name} decomposition digit size {} must be smaller than the {size}-limb key width",
        key.dsize(),
    );
    let decomposition_limbs = key
        .dnum()
        .as_usize()
        .checked_mul(key.dsize().as_usize())
        .with_context(|| format!("{name} decomposition size overflows usize"))?;
    ensure!(
        decomposition_limbs <= size,
        "{name} decomposition uses {decomposition_limbs} limbs, exceeding its {size}-limb key width",
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
            && reported.dsize() == view.dsize(),
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
