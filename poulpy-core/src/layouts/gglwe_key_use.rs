//! Level-policy-selected reuse of GGLWE keys through whole-row containment.
//!
//! A key with a fine decomposition contains coarser keys as whole rows: for
//! `s = D / dsize`, effective digit `i` is physical row `(i + 1) * s - 1`.
//! [`GGLWEKeyUsePolicy`] picks the effective `dsize` from the input precision,
//! [`resolve_gglwe_key_use`] turns a physical key plus that `dsize` into the
//! logical layout and row geometry a product must use, and
//! [`GGLWEKeyRegistry`] compiles `(function, size) -> physical key` once.

use std::{collections::HashMap, hash::Hash, num::NonZeroUsize};

use crate::{
    error::{CoreError, Result},
    layouts::{Base2K, Dnum, Dsize, GGLWEInfos, GGLWELayout, LWEInfos, TorusPrecision},
};

pub(crate) fn err(op: &'static str, detail: String) -> CoreError {
    CoreError::GGLWEKeyUse { op, detail }
}

/// Gadget scalars of a physical key, widened out of the saturating newtypes so
/// every product below is exact.
struct Gadget {
    base2k: usize,
    dsize: usize,
    dnum: usize,
    k_aux: usize,
    /// `dnum * dsize * base2k + k_aux`: the total torus precision the key spans.
    total_k: usize,
}

/// `a * b`, or an overflow error naming the product.
fn mul(a: usize, b: usize, what: &str, op: &'static str) -> Result<usize> {
    a.checked_mul(b)
        .ok_or_else(|| err(op, format!("{what}: {a} * {b} overflows usize")))
}

/// `a + b`, or an overflow error naming the sum.
fn add(a: usize, b: usize, what: &str, op: &'static str) -> Result<usize> {
    a.checked_add(b)
        .ok_or_else(|| err(op, format!("{what}: {a} + {b} overflows usize")))
}

fn gadget<P: GGLWEInfos>(key: &P, op: &'static str) -> Result<Gadget> {
    let (base2k, dsize) = (key.base2k().as_usize(), key.dsize().as_usize());
    let (dnum, k_aux) = (key.dnum().as_usize(), key.k_aux().as_usize());
    if base2k == 0 || dsize == 0 {
        return Err(err(op, format!("base2k={base2k} and dsize={dsize} must both be non-zero")));
    }
    let digit: usize = mul(dsize, base2k, "digit", op)?;
    if k_aux < digit {
        return Err(err(
            op,
            format!("k_aux={k_aux} does not cover one gadget digit of {digit} bits"),
        ));
    }
    let total_k: usize = add(mul(dnum, digit, "stored precision", op)?, k_aux, "total_k", op)?;
    narrow(total_k, "total precision", op)?;
    Ok(Gadget {
        base2k,
        dsize,
        dnum,
        k_aux,
        total_k,
    })
}

/// Narrows back into the `u32`-backed layout newtypes.
fn narrow(v: usize, what: &str, op: &'static str) -> Result<u32> {
    u32::try_from(v).map_err(|_| err(op, format!("{what}={v} exceeds u32")))
}

/// Tie-break tuple shared by registration and dispatch selection.
type PhysicalOrder = (u128, u32, u32, u32, u32, usize, usize);

/// Backend-independent matrix material, widened so adversarial public layout
/// metadata cannot wrap the deterministic registry ordering.
fn material<P: GGLWEInfos>(key: &P, op: &'static str) -> Result<u128> {
    let cols_out: u128 = u128::from(key.rank_out().as_u32())
        .checked_add(1)
        .ok_or_else(|| err(op, "output column count overflows u128".to_string()))?;
    [
        u128::from(key.dnum().as_u32()),
        u128::from(key.rank_in().as_u32()),
        cols_out,
        key.max_size() as u128,
    ]
    .into_iter()
    .try_fold(1u128, u128::checked_mul)
    .ok_or_else(|| err(op, "GGLWE material score overflows u128".to_string()))
}

fn physical_order<P: GGLWEInfos>(key: &P, index: usize, op: &'static str) -> Result<PhysicalOrder> {
    Ok((
        material(key, op)?,
        key.k().as_u32(),
        key.dsize().as_u32(),
        key.dnum().as_u32(),
        key.k_aux().as_u32(),
        key.max_size(),
        index,
    ))
}

/// Whether `requested` is exactly a whole-row subset of `parent`.
///
/// Requires matching degree, `base2k` and ranks, a `dsize` that is a multiple
/// of the parent's, enough physical rows, and enough residual precision to keep
/// the requested auxiliary guard. Callers check cryptographic domain and
/// functional identity separately.
pub fn gglwe_is_whole_row_subset<P: GGLWEInfos, Q: GGLWEInfos>(parent: &P, requested: &Q) -> Result<bool> {
    const OP: &str = "gglwe_is_whole_row_subset";
    let p: Gadget = gadget(parent, OP)?;
    let q: Gadget = gadget(requested, OP)?;
    if parent.n() != requested.n()
        || p.base2k != q.base2k
        || parent.rank_in() != requested.rank_in()
        || parent.rank_out() != requested.rank_out()
        || !q.dsize.is_multiple_of(p.dsize)
    {
        return Ok(false);
    }
    let s: usize = q.dsize / p.dsize;
    let consumed: usize = mul(q.dnum, mul(q.dsize, p.base2k, "requested digit", OP)?, "consumed", OP)?;
    let rows: usize = mul(s, q.dnum, "requested rows", OP)?;
    Ok(rows <= p.dnum && p.total_k >= consumed && p.total_k - consumed >= q.k_aux)
}

/// Immutable `size -> dsize` table: the effective decomposition is a hard
/// requirement of the input precision, never a preference.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GGLWEKeyUsePolicy {
    base2k: Base2K,
    /// Indexed exactly by `size = ceil(k / base2k)`; entry zero is the
    /// zero-precision convention.
    dsize_by_size: Box<[Dsize]>,
}

impl GGLWEKeyUsePolicy {
    pub fn new(base2k: Base2K, dsize_by_size: Box<[Dsize]>) -> Result<Self> {
        const OP: &str = "GGLWEKeyUsePolicy::new";
        if base2k.as_u32() == 0 {
            return Err(err(OP, "base2k must be non-zero".to_string()));
        }
        if dsize_by_size.is_empty() {
            return Err(err(OP, "table must declare at least size 0".to_string()));
        }
        if let Some(i) = dsize_by_size.iter().position(|d| d.as_u32() == 0) {
            return Err(err(OP, format!("dsize_by_size[{i}] is zero")));
        }
        Ok(Self { base2k, dsize_by_size })
    }

    pub fn base2k(&self) -> Base2K {
        self.base2k
    }

    /// Number of declared sizes, i.e. one past the largest indexable size.
    pub fn sizes(&self) -> usize {
        self.dsize_by_size.len()
    }

    /// Effective `dsize` for exact precision `k`. Out of range is an error;
    /// there is no nearest-size fallback.
    pub fn dsize_for_k(&self, k: TorusPrecision) -> Result<Dsize> {
        let size: usize = k.div_ceil(self.base2k) as usize;
        self.dsize_by_size.get(size).copied().ok_or_else(|| {
            err(
                "GGLWEKeyUsePolicy::dsize_for_k",
                format!("k={k} maps to size={size}, outside the {} declared sizes", self.sizes()),
            )
        })
    }
}

/// One key use resolved for an exact input precision.
///
/// Sizing and execution both read this, never the physical `dnum`, `dsize` or
/// `k_aux`, which a [`GGLWEKeyUse`](super::GGLWEKeyUse) wrapper still forwards.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GGLWEUse {
    /// Zero precision: no row is active, so no key material is read and no
    /// product runs. Not an error.
    Empty,
    Active(GGLWEActiveUse),
}

/// A positive-precision use: which physical rows are read, and the logical
/// layout the product and its sizing are carried out in.
///
/// Every field is private and there is no public constructor: a value of this
/// type can only come out of [`resolve_gglwe_key_use`], so the relation between
/// the logical layout, the row map, the readable prefix and the stored
/// decomposition holds by construction. A hand-built use claiming `dsize = 1`
/// over a key that stores `dsize = 8` would otherwise reach the dense
/// specialization and read the whole stored matrix.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GGLWEActiveUse {
    /// Exact input precision this use was resolved for.
    input_k: TorusPrecision,
    /// `(dnum, dsize, k_aux) = (r_active, D, a_eff)`, physical degree/base2k/ranks.
    logical_layout: GGLWELayout,
    /// Logical row `i` is physical row `first_physical_row + i * physical_row_step`.
    first_physical_row: usize,
    physical_row_step: NonZeroUsize,
    /// Readable limb prefix within each selected physical polynomial.
    logical_work_size: usize,
    /// Rows the physical key stores.
    physical_rows: usize,
    /// Storage pitch: limbs each physical polynomial occupies.
    physical_size: usize,
    /// The stored decomposition the rows were selected out of. A prepared key
    /// is paired with a bound only if all three agree, so two keys of the same
    /// shape but different radix, digit or guard cannot be swapped.
    physical_base2k: Base2K,
    physical_dsize: Dsize,
    physical_k_aux: TorusPrecision,
    /// Whether the key has enough digits for the whole input precision. A partial
    /// raw-key use is valid, while registry/helper selection requires coverage.
    covers_input: bool,
}

impl GGLWEActiveUse {
    /// Exact input precision this use was resolved for.
    pub fn input_k(&self) -> TorusPrecision {
        self.input_k
    }

    /// The layout the product and its sizing are carried out in.
    pub fn logical_layout(&self) -> &GGLWELayout {
        &self.logical_layout
    }

    /// Logical row `i` is physical row `first_physical_row() + i * physical_row_step()`.
    pub fn first_physical_row(&self) -> usize {
        self.first_physical_row
    }

    pub fn physical_row_step(&self) -> NonZeroUsize {
        self.physical_row_step
    }

    /// Readable limb prefix within each selected physical polynomial.
    pub fn logical_work_size(&self) -> usize {
        self.logical_work_size
    }

    /// Rows the physical key stores.
    pub fn physical_rows(&self) -> usize {
        self.physical_rows
    }

    /// Storage pitch: limbs each physical polynomial occupies.
    pub fn physical_size(&self) -> usize {
        self.physical_size
    }

    /// The stored decomposition the rows were selected out of.
    pub fn physical_gadget(&self) -> (Base2K, Dsize, TorusPrecision) {
        (self.physical_base2k, self.physical_dsize, self.physical_k_aux)
    }

    /// Whether the key has enough digits for the whole input precision.
    pub fn covers_input(&self) -> bool {
        self.covers_input
    }

    /// Limbs the product input must carry.
    pub fn input_size(&self) -> usize {
        self.input_k.div_ceil(self.logical_layout.base2k()) as usize
    }

    /// Whether the use covers the whole stored key, so a dense kernel reads it
    /// as-is with no row map and no prefix.
    pub fn is_dense(&self) -> bool {
        self.physical_row_step.get() == 1
            && self.logical_layout.dnum().as_usize() == self.physical_rows
            && self.logical_work_size == self.physical_size
    }
}

impl GGLWEUse {
    /// The active geometry, or `None` at zero precision.
    pub fn active(&self) -> Option<&GGLWEActiveUse> {
        match self {
            Self::Empty => None,
            Self::Active(use_) => Some(use_),
        }
    }
}

/// Resolves a key for one exact input precision.
///
/// The default resolves [`GGLWEInfos::effective_dsize`], so a stored key binds
/// through its own decomposition and a
/// [`WithEffectiveDsize::with_dsize`](super::WithEffectiveDsize::with_dsize)
/// wrapper binds through the one it requests. Binding must happen before the
/// wrapper is converted to a concrete prepared reference, since that conversion
/// carries the physical `dsize`.
pub trait GGLWEBind: GGLWEInfos {
    fn bind_for(&self, input_k: TorusPrecision) -> Result<GGLWEUse>
    where
        Self: Sized,
    {
        let effective_dsize: Dsize = self.effective_dsize();
        match resolve_gglwe_key_use(self, input_k, effective_dsize)? {
            Some(use_) => Ok(use_),
            None => Err(err(
                "bind_for",
                format!("key cannot realize dsize={effective_dsize} at input_k={input_k}"),
            )),
        }
    }

    /// [`Self::bind_for`], refusing a key that cannot decompose the whole input
    /// precision. Use this for registry/helper selection that must not drop digits.
    fn bind_covering_for(&self, input_k: TorusPrecision) -> Result<GGLWEUse>
    where
        Self: Sized,
    {
        let use_ = self.bind_for(input_k)?;
        if let GGLWEUse::Active(active) = &use_
            && !active.covers_input
        {
            let digit: usize = mul(
                active.logical_layout.dsize().as_usize(),
                self.base2k().as_usize(),
                "digit",
                "bind_covering_for",
            )?;
            return Err(err(
                "bind_covering_for",
                format!(
                    "key has {} digit(s) of {digit} bits, short of the {} input_k={input_k} needs",
                    active.logical_layout.dnum(),
                    input_k.as_usize().div_ceil(digit),
                ),
            ));
        }
        Ok(use_)
    }
}

impl<T: GGLWEInfos> GGLWEBind for T {}

/// Resolves how `physical` realizes `effective_dsize` at exact precision
/// `input_k`.
///
/// `Err` is an invalid layout or an out-of-range conversion; `Ok(None)` means
/// this physical key cannot realize that decomposition at that precision.
pub(crate) fn resolve_gglwe_key_use<P: GGLWEInfos>(
    physical: &P,
    input_k: TorusPrecision,
    effective_dsize: Dsize,
) -> Result<Option<GGLWEUse>> {
    const OP: &str = "resolve_gglwe_key_use";
    let p: Gadget = gadget(physical, OP)?;
    let d: usize = effective_dsize.as_usize();
    if d == 0 {
        return Err(err(OP, "effective_dsize must be non-zero".to_string()));
    }
    // Zero precision reads no row and therefore does not require the policy's
    // conventional dsize to be realizable by the physical decomposition. Keep
    // validating the physical gadget and the non-zero effective dsize above,
    // but do not reject an otherwise valid key on row geometry that is unused.
    if input_k.as_usize() == 0 {
        return Ok(Some(GGLWEUse::Empty));
    }
    if !d.is_multiple_of(p.dsize) {
        return Ok(None);
    }
    let s: usize = d / p.dsize;
    let digit: usize = mul(d, p.base2k, "effective digit", OP)?;
    if p.total_k < digit {
        return Ok(None);
    }
    // The stored decomposition uses every stored row and its exact `k_aux`.
    // Coarsening must reserve a full coarse digit of padding, so the capacity
    // rule below applies only when rows are actually being skipped.
    let (r_eff, a_eff): (usize, usize) = if s == 1 {
        (p.dnum, p.k_aux)
    } else {
        // Largest complete effective decomposition rows and padding both allow;
        // the remaining precision stays as auxiliary padding, never dropped.
        // `total_k >= digit` was checked above, so the quotient is at least one.
        let r: usize = (p.dnum / s).min((p.total_k / digit) - 1);
        (r, p.total_k - mul(r, digit, "coarse precision", OP)?)
    };
    let requested: usize = input_k.as_usize().div_ceil(digit);
    if r_eff == 0 {
        return Ok(None);
    }
    // Raw-key execution may intentionally omit disposable low/auxiliary digits;
    // registry and helper selection reject that case through `covers_input`.
    let covers_input: bool = requested <= r_eff;
    let r_active: usize = requested.min(r_eff);

    let logical_layout = GGLWELayout {
        n: physical.n(),
        base2k: physical.base2k(),
        dnum: Dnum(narrow(r_active, "effective dnum", OP)?),
        k_aux: TorusPrecision(narrow(a_eff, "effective k_aux", OP)?),
        rank_in: physical.rank_in(),
        rank_out: physical.rank_out(),
        dsize: effective_dsize,
    };
    let logical_work_size: usize = add(
        mul(r_active, d, "effective work rows", OP)?,
        a_eff.div_ceil(p.base2k),
        "logical work size",
        OP,
    )?;
    debug_assert_eq!(logical_layout.max_size(), logical_work_size);
    debug_assert_eq!(logical_layout.size(), logical_work_size);

    let step: NonZeroUsize = NonZeroUsize::new(s).expect("row step is a positive quotient");
    Ok(Some(GGLWEUse::Active(GGLWEActiveUse {
        input_k,
        logical_layout,
        first_physical_row: step.get() - 1,
        physical_row_step: step,
        logical_work_size,
        physical_rows: p.dnum,
        physical_size: physical.max_size(),
        physical_base2k: physical.base2k(),
        physical_dsize: physical.dsize(),
        physical_k_aux: physical.k_aux(),
        covers_input,
    })))
}

/// Opaque, stable identifier of a registered physical key.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GGLWEPhysicalKeyId(usize);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GGLWERegisterOutcome {
    Inserted(GGLWEPhysicalKeyId),
    Reused(GGLWEPhysicalKeyId),
}

/// Registration phase: physical keys only, in stable insertion order.
pub struct GGLWEKeyRegistryBuilder<Id, K> {
    entries: Vec<(Id, K)>,
}

impl<Id, K> Default for GGLWEKeyRegistryBuilder<Id, K> {
    fn default() -> Self {
        Self { entries: Vec::new() }
    }
}

impl<Id: Clone + Eq + Hash, K: GGLWEInfos> GGLWEKeyRegistryBuilder<Id, K> {
    pub fn new() -> Self {
        Self::default()
    }

    /// Smallest already-registered key of `id` that contains `requested` as
    /// whole rows.
    pub fn find_subsuming<Q: GGLWEInfos>(&self, id: &Id, requested: &Q) -> Result<Option<GGLWEPhysicalKeyId>> {
        let mut best: Option<(PhysicalOrder, usize)> = None;
        for (i, (entry_id, key)) in self.entries.iter().enumerate() {
            if entry_id != id || !gglwe_is_whole_row_subset(key, requested)? {
                continue;
            }
            let order: PhysicalOrder = physical_order(key, i, "GGLWEKeyRegistryBuilder::find_subsuming")?;
            if best.as_ref().is_none_or(|(b, _)| order < *b) {
                best = Some((order, i));
            }
        }
        Ok(best.map(|(_, i)| GGLWEPhysicalKeyId(i)))
    }

    /// Registers `key` under `id`, reusing a subsuming parent when one exists.
    pub fn register(&mut self, id: Id, key: K) -> Result<GGLWERegisterOutcome> {
        const OP: &str = "GGLWEKeyRegistryBuilder::register";
        gadget(&key, OP)?;
        if let Some((_, first)) = self.entries.iter().find(|(entry_id, _)| *entry_id == id)
            && (first.n() != key.n()
                || first.base2k() != key.base2k()
                || first.rank_in() != key.rank_in()
                || first.rank_out() != key.rank_out())
        {
            return Err(err(
                OP,
                format!(
                    "function fixes (n, base2k, rank_in, rank_out) = ({}, {}, {}, {}), got ({}, {}, {}, {})",
                    first.n(),
                    first.base2k(),
                    first.rank_in(),
                    first.rank_out(),
                    key.n(),
                    key.base2k(),
                    key.rank_in(),
                    key.rank_out()
                ),
            ));
        }
        if let Some(existing) = self.find_subsuming(&id, &key)? {
            return Ok(GGLWERegisterOutcome::Reused(existing));
        }
        self.entries.push((id, key));
        Ok(GGLWERegisterOutcome::Inserted(GGLWEPhysicalKeyId(self.entries.len() - 1)))
    }

    /// Compiles `(function, size) -> physical key` for every declared size.
    pub fn finish(self, policy: GGLWEKeyUsePolicy) -> Result<GGLWEKeyRegistry<Id, K>> {
        const OP: &str = "GGLWEKeyRegistryBuilder::finish";
        for (_, key) in &self.entries {
            if key.base2k() != policy.base2k() {
                return Err(err(
                    OP,
                    format!("key base2k={} does not match policy base2k={}", key.base2k(), policy.base2k()),
                ));
            }
        }

        let mut dispatch: HashMap<Id, Box<[Option<usize>]>> = HashMap::new();
        for (id, _) in &self.entries {
            if dispatch.contains_key(id) {
                continue;
            }
            let mut row: Vec<Option<usize>> = Vec::with_capacity(policy.sizes());
            for size in 0..policy.sizes() {
                row.push(
                    match size
                        .checked_mul(policy.base2k().as_usize())
                        .and_then(|k| u32::try_from(k).ok())
                    {
                        // No `k` has this size, so the cell is unreachable.
                        None => None,
                        Some(k) => self.select(id, TorusPrecision(k), policy.dsize_by_size[size])?,
                    },
                );
            }
            dispatch.insert(id.clone(), row.into_boxed_slice());
        }

        Ok(GGLWEKeyRegistry {
            keys: self.entries.into_iter().map(|(_, key)| key).collect(),
            dispatch,
            policy,
        })
    }

    /// Least-material key of `id` able to realize `d` at exact precision `k`.
    fn select(&self, id: &Id, k: TorusPrecision, d: Dsize) -> Result<Option<usize>> {
        let mut best: Option<((u128, PhysicalOrder), usize)> = None;
        for (i, (entry_id, key)) in self.entries.iter().enumerate() {
            if entry_id != id {
                continue;
            }
            let Some(use_) = resolve_gglwe_key_use(key, k, d)? else {
                continue;
            };
            // Dispatch only picks keys wide enough for the whole precision.
            if use_.active().is_some_and(|a| !a.covers_input) {
                continue;
            }
            // Zero precision reads nothing, so every candidate ties at zero.
            let active_material: u128 = use_
                .active()
                .map_or(Ok(0), |a| material(a.logical_layout(), "GGLWEKeyRegistryBuilder::select"))?;
            let order: (u128, PhysicalOrder) = (active_material, physical_order(key, i, "GGLWEKeyRegistryBuilder::select")?);
            if best.as_ref().is_none_or(|(b, _)| order < *b) {
                best = Some((order, i));
            }
        }
        Ok(best.map(|(_, i)| i))
    }
}

/// Immutable policy plus compiled dispatch over physical keys.
pub struct GGLWEKeyRegistry<Id, K> {
    /// Indexed by [`GGLWEPhysicalKeyId`]; registration order is serialization order.
    keys: Vec<K>,
    dispatch: HashMap<Id, Box<[Option<usize>]>>,
    policy: GGLWEKeyUsePolicy,
}

impl<Id: Clone + Eq + Hash, K: GGLWEInfos> GGLWEKeyRegistry<Id, K> {
    pub fn policy(&self) -> &GGLWEKeyUsePolicy {
        &self.policy
    }

    /// Physical key and effective `dsize` for `id` at exact precision `k`.
    pub fn key_for(&self, id: &Id, k: TorusPrecision) -> Result<(&K, Dsize)> {
        const OP: &str = "GGLWEKeyRegistry::key_for";
        let d: Dsize = self.policy.dsize_for_k(k)?;
        let size: usize = k.div_ceil(self.policy.base2k()) as usize;
        let row: &[Option<usize>] = self
            .dispatch
            .get(id)
            .ok_or_else(|| err(OP, format!("no key registered for this function (k={k})")))?;
        let index: usize = row[size].ok_or_else(|| err(OP, format!("no key covers k={k} at dsize={d} (size={size})")))?;
        let key: &K = &self.keys[index];
        let covers: bool = match resolve_gglwe_key_use(key, k, d)? {
            None => false,
            Some(use_) => use_.active().is_none_or(|a| a.covers_input),
        };
        if !covers {
            return Err(err(OP, format!("dispatched key cannot realize dsize={d} at k={k}")));
        }
        Ok((key, d))
    }

    /// Rebuilds the registry over mapped values, preserving policy, physical
    /// ids and dispatch. Mapped keys must keep the exact source infos.
    pub fn try_map_values<K2: GGLWEInfos>(
        &self,
        mut map: impl FnMut(GGLWEPhysicalKeyId, &K) -> Result<K2>,
    ) -> Result<GGLWEKeyRegistry<Id, K2>> {
        const OP: &str = "GGLWEKeyRegistry::try_map_values";
        let mut keys: Vec<K2> = Vec::with_capacity(self.keys.len());
        for (i, key) in self.keys.iter().enumerate() {
            let mapped: K2 = map(GGLWEPhysicalKeyId(i), key)?;
            if mapped.gglwe_layout() != key.gglwe_layout() {
                return Err(err(OP, format!("mapped key {i} does not have the source physical infos")));
            }
            keys.push(mapped);
        }
        Ok(GGLWEKeyRegistry {
            keys,
            dispatch: self.dispatch.clone(),
            policy: self.policy.clone(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layouts::{Degree, Rank, WithEffectiveDsize, key_work_size};

    const N: Degree = Degree(1024);
    const K: u32 = 8;
    /// Normal padding: one full gadget digit plus `log2(n)`.
    fn normal_aux(dsize: u32) -> TorusPrecision {
        TorusPrecision(dsize * K + N.log2() as u32)
    }

    fn layout(dsize: u32, dnum: u32, k_aux: TorusPrecision) -> GGLWELayout {
        GGLWELayout {
            n: N,
            base2k: Base2K(K),
            dnum: Dnum(dnum),
            k_aux,
            rank_in: Rank(1),
            rank_out: Rank(1),
            dsize: Dsize(dsize),
        }
    }

    fn normal(dsize: u32, dnum: u32) -> GGLWELayout {
        layout(dsize, dnum, normal_aux(dsize))
    }

    fn policy(dsizes: &[u32]) -> GGLWEKeyUsePolicy {
        GGLWEKeyUsePolicy::new(Base2K(K), dsizes.iter().map(|d| Dsize(*d)).collect()).unwrap()
    }

    /// A coarsened use is sized from its resolved `k_aux`, not the physical one
    /// the `with_dsize` wrapper still forwards.
    ///
    /// `dsize=8, dnum=3, k_aux=8K+g` at `D=16` and `input=16K` resolves to
    /// `k_aux=16K+g` and 33 work limbs; reading the wrapper's physical `k_aux`
    /// gives 25 and under-sizes every accumulated product built on it.
    #[test]
    fn coarsened_use_is_sized_from_its_resolved_aux() {
        // `g = 1` so the guard costs one limb, matching the request's arithmetic.
        let physical: GGLWELayout = layout(8, 3, TorusPrecision(8 * K + 1));
        let input_k: TorusPrecision = TorusPrecision(16 * K);
        let effective: Dsize = Dsize(16);

        let use_: GGLWEActiveUse = resolve_gglwe_key_use(&physical, input_k, effective)
            .expect("valid layout")
            .expect("the parent realizes the coarsening")
            .active()
            .copied()
            .expect("positive precision");

        assert_eq!(use_.logical_layout.k_aux, TorusPrecision(16 * K + 1));
        assert_eq!(use_.logical_work_size, 33);
        assert_eq!(
            key_work_size(
                physical.base2k(),
                input_k,
                use_.logical_layout.dsize(),
                use_.logical_layout.k_aux()
            ),
            use_.logical_work_size
        );

        // What the wrapper alone would have produced: the effective `dsize` with
        // the physical `k_aux`.
        let wrapper = physical.with_dsize(effective);
        assert_eq!(
            key_work_size(wrapper.base2k(), input_k, wrapper.effective_dsize(), wrapper.k_aux()),
            25,
            "the regression this pins is the wrapper's physical k_aux reaching a sizing helper"
        );
    }

    // Acceptance 1.
    #[test]
    fn whole_row_containment() {
        assert!(gglwe_is_whole_row_subset(&normal(8, 3), &normal(16, 1)).unwrap());
        assert!(!gglwe_is_whole_row_subset(&normal(8, 2), &normal(16, 1)).unwrap());
        // dsize must be a whole multiple, and ranks must match.
        assert!(!gglwe_is_whole_row_subset(&normal(8, 8), &normal(12, 1)).unwrap());
        let mut wide: GGLWELayout = normal(16, 1);
        wide.rank_in = Rank(2);
        assert!(!gglwe_is_whole_row_subset(&normal(8, 3), &wide).unwrap());
    }

    // Acceptance 2: non-minimal padding survives both containment and resolution.
    #[test]
    fn non_minimal_padding_is_preserved() {
        let parent: GGLWELayout = normal(8, 4);
        let requested: GGLWELayout = layout(16, 1, TorusPrecision(24 * K + N.log2() as u32));
        assert!(gglwe_is_whole_row_subset(&parent, &requested).unwrap());

        let use_: GGLWEActiveUse = resolve_gglwe_key_use(&parent, TorusPrecision(16 * K), Dsize(16))
            .unwrap()
            .unwrap()
            .active()
            .copied()
            .expect("positive precision");
        assert_eq!(use_.logical_layout.k_aux, requested.k_aux);
        assert_eq!(use_.logical_layout.dnum, Dnum(1));
        assert_eq!(use_.first_physical_row, 1);
    }

    // Acceptance 3.
    #[test]
    fn dense_policy_indexing() {
        let p: GGLWEKeyUsePolicy = policy(&[1, 1, 2, 2, 4]);
        assert_eq!(p.dsize_for_k(TorusPrecision(0)).unwrap(), Dsize(1));
        assert_eq!(p.dsize_for_k(TorusPrecision(2 * K)).unwrap(), Dsize(2));
        // No nearest-size fallback past the declared table.
        assert_eq!(p.dsize_for_k(TorusPrecision(3 * K)).unwrap(), Dsize(2));
        assert_eq!(p.dsize_for_k(TorusPrecision(4 * K)).unwrap(), Dsize(4));
        assert!(p.dsize_for_k(TorusPrecision(4 * K + 1)).is_err());
        assert!(GGLWEKeyUsePolicy::new(Base2K(K), Box::new([])).is_err());
        assert!(GGLWEKeyUsePolicy::new(Base2K(K), Box::new([Dsize(1), Dsize(0)])).is_err());
    }

    // Acceptance 3: zero precision resolves with no row to read.
    #[test]
    fn zero_precision_selects_no_row() {
        let use_: GGLWEUse = resolve_gglwe_key_use(&normal(8, 3), TorusPrecision(0), Dsize(16))
            .unwrap()
            .unwrap();
        assert_eq!(use_, GGLWEUse::Empty, "zero precision is a valid use, not an error");
        assert!(use_.active().is_none());
    }

    #[test]
    fn registry_zero_precision_ignores_conventional_dsize_geometry() {
        let mut builder: GGLWEKeyRegistryBuilder<(), GGLWELayout> = GGLWEKeyRegistryBuilder::new();
        builder.register((), normal(8, 4)).unwrap();

        // Entry zero is only a convention. It need not divide the physical
        // key's dsize because a zero-precision use reads no rows.
        let registry = builder.finish(policy(&[1, 8])).unwrap();
        let (key, dsize) = registry.key_for(&(), TorusPrecision(0)).unwrap();
        assert_eq!(key.dsize(), Dsize(8));
        assert_eq!(dsize, Dsize(1));
        assert_eq!(key.with_dsize(dsize).bind_for(TorusPrecision(0)).unwrap(), GGLWEUse::Empty);
    }

    // Acceptance 4: a native-dsize use is a step-1 identity over r_active rows.
    #[test]
    fn native_dsize_reads_active_rows_only() {
        let parent: GGLWELayout = normal(8, 4);
        let use_: GGLWEActiveUse = resolve_gglwe_key_use(&parent, TorusPrecision(16 * K), Dsize(8))
            .unwrap()
            .unwrap()
            .active()
            .copied()
            .expect("positive precision");
        assert_eq!(use_.physical_row_step.get(), 1);
        assert_eq!(use_.first_physical_row, 0);
        // Native, but not dense: fewer active rows and a shorter prefix.
        assert!(!use_.is_dense());
        // Two active digits out of four capacity rows.
        assert_eq!(use_.logical_layout.dnum, Dnum(2));
        assert_eq!(use_.logical_layout.dsize, parent.dsize);
        assert!(use_.logical_work_size < parent.max_size());
    }

    // Acceptance 5 and 6: selected rows are 1, then 1 and 3.
    #[test]
    fn coarsened_rows_are_odd_indexed() {
        let rows = |dnum: u32, k: u32| -> Vec<usize> {
            let use_: GGLWEActiveUse = resolve_gglwe_key_use(&normal(8, dnum), TorusPrecision(k * K), Dsize(16))
                .unwrap()
                .unwrap()
                .active()
                .copied()
                .expect("positive precision");
            (0..use_.logical_layout.dnum.as_usize())
                .map(|i| use_.first_physical_row + i * use_.physical_row_step.get())
                .collect()
        };
        assert_eq!(rows(3, 16), vec![1]);
        assert_eq!(rows(6, 32), vec![1, 3]);
        // Row capacity is there but padding is not: one digit short. A raw-key
        // product may intentionally use the available prefix, while registry or
        // helper selection must require complete coverage.
        let physical = normal(8, 4);
        let short: GGLWEActiveUse = resolve_gglwe_key_use(&physical, TorusPrecision(32 * K), Dsize(16))
            .unwrap()
            .unwrap()
            .active()
            .copied()
            .expect("positive precision");
        assert!(!short.covers_input);
        assert_eq!(short.logical_layout.dnum, Dnum(1));
        let wrapped = physical.with_dsize(Dsize(16));
        assert!(wrapped.bind_for(TorusPrecision(32 * K)).is_ok());
        assert!(wrapped.bind_covering_for(TorusPrecision(32 * K)).is_err());
    }

    // Acceptance 10.
    #[test]
    fn registry_dedup_dispatch_and_ties() {
        let mut builder: GGLWEKeyRegistryBuilder<i64, GGLWELayout> = GGLWEKeyRegistryBuilder::new();
        assert_eq!(
            builder.register(5, normal(8, 4)).unwrap(),
            GGLWERegisterOutcome::Inserted(GGLWEPhysicalKeyId(0))
        );
        // Subsumed as whole rows: reused, not stored.
        assert_eq!(
            builder.register(5, normal(16, 1)).unwrap(),
            GGLWERegisterOutcome::Reused(GGLWEPhysicalKeyId(0))
        );
        // Not a whole-row coarsening of dsize 8: coexists.
        assert_eq!(
            builder.register(5, normal(3, 4)).unwrap(),
            GGLWERegisterOutcome::Inserted(GGLWEPhysicalKeyId(1))
        );
        // A different function never reuses another function's key.
        assert_eq!(
            builder.register(7, normal(8, 4)).unwrap(),
            GGLWERegisterOutcome::Inserted(GGLWEPhysicalKeyId(2))
        );
        // Ranks are fixed per function.
        let mut wide: GGLWELayout = normal(8, 4);
        wide.rank_in = Rank(2);
        assert!(builder.register(5, wide).is_err());

        let registry: GGLWEKeyRegistry<i64, GGLWELayout> = builder.finish(policy(&[1, 1, 8, 8, 16])).unwrap();
        // size 2 wants dsize 8: the dsize-8 parent, at one active digit.
        let (key, d) = registry.key_for(&5, TorusPrecision(2 * K)).unwrap();
        assert_eq!((key.dsize, d), (Dsize(8), Dsize(8)));
        // size 4 wants dsize 16: the same parent, coarsened.
        let (key, d) = registry.key_for(&5, TorusPrecision(4 * K)).unwrap();
        assert_eq!((key.dsize, d), (Dsize(8), Dsize(16)));
        // Unregistered function, and a size no key covers.
        assert!(registry.key_for(&9, TorusPrecision(2 * K)).is_err());
        assert!(registry.key_for(&5, TorusPrecision(5 * K)).is_err());
    }

    // Acceptance 10: selection is by score, not by registration order. Both keys
    // coarsen to dsize 6; the second holds less unique material for this size.
    #[test]
    fn registry_dispatch_selects_least_material() {
        let mut builder: GGLWEKeyRegistryBuilder<(), GGLWELayout> = GGLWEKeyRegistryBuilder::new();
        builder.register((), normal(3, 20)).unwrap();
        builder.register((), normal(2, 30)).unwrap();
        let registry: GGLWEKeyRegistry<(), GGLWELayout> = builder.finish(policy(&[1, 1, 1, 1, 1, 1, 6])).unwrap();
        let (key, d) = registry.key_for(&(), TorusPrecision(6 * K)).unwrap();
        assert_eq!((key.dsize, d), (Dsize(2), Dsize(6)));
    }

    // Acceptance 10: a subsumed smaller-aux registration loses its own aux choice.
    #[test]
    fn subsumed_registration_loses_its_aux_choice() {
        let mut builder: GGLWEKeyRegistryBuilder<(), GGLWELayout> = GGLWEKeyRegistryBuilder::new();
        builder.register((), normal(8, 4)).unwrap();
        builder.register((), layout(16, 1, normal_aux(16))).unwrap();
        let registry: GGLWEKeyRegistry<(), GGLWELayout> = builder.finish(policy(&[1, 1, 8, 8, 16])).unwrap();
        let (key, _) = registry.key_for(&(), TorusPrecision(4 * K)).unwrap();
        // The parent's padding, not the subsumed registration's.
        assert_eq!(key.k_aux, normal_aux(8));
    }

    // Acceptance 11.
    #[test]
    fn try_map_values_preserves_dispatch() {
        let mut builder: GGLWEKeyRegistryBuilder<i64, GGLWELayout> = GGLWEKeyRegistryBuilder::new();
        builder.register(5, normal(8, 4)).unwrap();
        builder.register(5, normal(3, 4)).unwrap();
        let registry: GGLWEKeyRegistry<i64, GGLWELayout> = builder.finish(policy(&[1, 1, 8, 8, 16])).unwrap();

        let mut seen: Vec<usize> = Vec::new();
        let mapped: GGLWEKeyRegistry<i64, GGLWELayout> = registry
            .try_map_values(|id, key| {
                seen.push(id.0);
                Ok(*key)
            })
            .unwrap();
        assert_eq!(seen, vec![0, 1]);
        assert_eq!(mapped.dispatch, registry.dispatch);

        // A mapped value must keep the exact physical infos.
        assert!(registry.try_map_values(|_, _| Ok(normal(1, 1))).is_err());
    }

    #[test]
    fn registration_rejects_total_precision_outside_the_layout_domain() {
        let invalid = layout(1, u32::MAX, TorusPrecision(K));
        let mut builder: GGLWEKeyRegistryBuilder<(), GGLWELayout> = GGLWEKeyRegistryBuilder::new();
        assert!(builder.register((), invalid).is_err());
        assert!(gglwe_is_whole_row_subset(&invalid, &normal(1, 1)).is_err());
    }
}
