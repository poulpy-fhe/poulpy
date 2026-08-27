# Poulpy API request: level-policy-selected prepared GGLWE reuse

## Scope

Target Poulpy `main`. This is generic GGLWE infrastructure. It must:

1. detect when a requested key is contained as whole rows of an existing key;
2. let prepared-key product kernels read only the selected physical rows and
   useful limb prefixes;
3. choose an effective decomposition from an explicit, level-aware
   `size -> dsize` policy; and
4. use the smallest compatible physical key for the requested function and exact
   input precision `k`.

At runtime, physical keys are complete backend-prepared `GGLWEPrepared`
objects. Helpers return a reference to that complete physical key together
with the effective `dsize`; the product backend receives the complete
`VmpPMat` byte view but addresses only the selected rows and limb prefixes.
This request concerns prepared-memory reads by the product kernel. It does not
add cold-storage paging, selective H2D transfer, or on-the-fly preparation.

There is no projected-key type, persistent public resolved-use object,
row/partial-row view, or public row selector. The client owns parameter
selection and any total-modulus cap; the registry does not enforce one.

## 1. Whole-row containment

For a physical parent `P`, define:

~~~text
K = base2k,  d_p = P.dsize,  r_p = P.dnum,  a_p = P.k_aux
T_p = r_p * d_p * K + a_p
~~~

A requested physical layout `Q = (d_q, r_q, a_q)` is contained iff degree,
`base2k`, input rank, and output rank match and:

~~~text
d_q % d_p == 0
s = d_q / d_p
s * r_q <= r_p
T_p - r_q * d_q * K >= a_q
physical_row(i) = (i + 1) * s - 1
~~~

All arithmetic is checked and both layouts must satisfy Poulpy's invariants.
The registry separately checks cryptographic domain and functional identity.

~~~rust
pub fn gglwe_is_whole_row_subset<P: GGLWEInfos, Q: GGLWEInfos>(
    parent: &P,
    requested: &Q,
) -> Result<bool>;
~~~

With normal padding, `(dsize=8,dnum=3) -> (16,1)` succeeds using physical
row `1`; `(8,2) -> (16,1)` fails because it does not retain one complete
`16K` auxiliary digit.

Registration containment concerns an exact requested physical layout. Runtime
selection below instead asks whether a physical key can implement a
policy-selected effective `dsize` at precision `k`.

## 2. Dense level policy

Replace `MaxDsize`/`MinDsize` with an immutable dense policy:

~~~rust
pub struct GGLWEKeyUsePolicy {
    base2k: Base2K,
    // Indexed exactly by size = ceil(k / base2k).
    // Entry zero explicitly follows Poulpy's zero-precision convention.
    dsize_by_size: Box<[Dsize]>,
}

impl GGLWEKeyUsePolicy {
    pub fn new(base2k: Base2K, dsize_by_size: Box<[Dsize]>) -> Result<Self>;
    pub fn base2k(&self) -> Base2K;
    pub fn dsize_for_k(&self, k: TorusPrecision) -> Result<Dsize>;
}
~~~

For a request:

~~~text
size = ceil(k / policy.base2k)
D    = policy.dsize_by_size[size]
~~~

The policy result `D` is a hard requirement, not a preference. An
out-of-range size, invalid entry, or missing compatible key is an error; there
is no nearest-size or alternative-`dsize` fallback. Exact `k`, rather than
only `size`, continues to drive coverage, work-size, and output calculations.

A policy and its helper cover one `base2k`. Registries reject physical keys
with another radix. Policies may be shared by automorphism, relinearization,
and other GGLWE helpers in the same cryptographic domain.

## 3. Resolving one physical key

For policy-selected `D`, a physical key is a candidate iff `D` is a
whole-row coarsening of its native decomposition and it retains a complete
effective auxiliary digit:

~~~text
D % d_p == 0
s = D / d_p

r_from_rows    = floor(r_p / s)
r_from_padding = floor((T_p - D * K) / (D * K))
r_eff          = min(r_from_rows, r_from_padding)
a_eff          = T_p - r_eff * D * K

r_active = ceil(k / (D * K))
candidate iff r_active <= r_eff
logical_work_size = r_active * D + ceil(a_eff / K)
                  = key_work_size(K, k, D, a_eff)

physical_row(i) = (i + 1) * s - 1
~~~

The subtraction is checked; `T_p < D*K` is not a candidate. This general
formula also handles non-minimal physical `k_aux`.

The shared internal resolver returns effective metadata, not a key view:

~~~rust
pub(crate) struct ResolvedGGLWEUse {
    // Logical layout used for product math. Its dnum is r_active.
    pub(crate) logical_layout: GGLWELayout,
    // None iff r_active == 0.
    pub(crate) first_physical_row: Option<usize>,
    pub(crate) physical_row_step: NonZeroUsize,
    pub(crate) logical_work_size: usize,
}

pub(crate) fn resolve_gglwe_key_use<P: GGLWEInfos>(
    physical: &P,
    input_k: TorusPrecision,
    effective_dsize: Dsize,
) -> Result<Option<ResolvedGGLWEUse>>;
~~~

`logical_layout` has `(dnum, dsize, k_aux) = (r_active, D, a_eff)`.
It copies the physical degree, `base2k`, and ranks, and satisfies
`logical_layout.size() == logical_layout.max_size() == logical_work_size`.
All conversions into `GGLWELayout` scalar fields and all intermediate
arithmetic are checked.
For a positive-precision use,
`first_physical_row = Some(row_step - 1)`. `Err` means invalid layout or
checked-arithmetic failure; `None` means that physical key cannot realize `D`
at `k`. Product, scratch, and output calculations consume this logical
metadata and must not accidentally read the physical key's native
`dnum`/`dsize`/`k_aux` as the effective values.

The resolver preserves the physical parent's full available precision when it
derives `a_eff`: it first takes the largest complete effective decomposition
allowed by rows and padding, then keeps the remaining precision as auxiliary
padding. It does not silently discard guard bits to minimize the suffix.
The `dsize`-only policy does not independently select `k_aux`; `a_eff` is
determined by the chosen physical parent. Because subset registration retains
no logical alias, registering a subsumed smaller-aux layout does not create a
second runtime auxiliary choice.

For `k == 0`, follow Poulpy's zero-precision convention: resolution and lookup
remain deterministic and `first_physical_row` is `None`. Branch before row
bounds checks or OEP dispatch; do not construct a zero-row compact key, read
the key, or launch a product kernel. Preserve each operation's empty-product
semantics: overwrite zeroes its result, accumulation leaves its accumulator
unchanged, and fused paths still execute any non-product body or epilogue.
Their `tmp_bytes` methods take the identical branch and include no
selected-product scratch.

For fixed functional identity, `D`, and `k`, choose the lexicographic
minimum of:

~~~text
U = r_active * rank_in * (rank_out + 1) * logical_work_size
P = physical_dnum * rank_in * (rank_out + 1) * physical_key.max_size()

tie order: U, P, physical key.k(), dsize, dnum, k_aux, max_size,
           stable physical-key ID
~~~

`U` is a backend-independent unique-material score, not literal traffic.
Compute `U` and `P` with checked `u128` arithmetic (or reject overflow before
comparison). Stable ordering must never depend on hash-map iteration.

The public helper result is only:

~~~rust
(&K, Dsize)
~~~

`K` is the complete physical prepared key. The returned `Dsize` may differ
from its native `dsize`; resolution, scratch, and execution must all use the
same returned value.

## 4. Prepared-key product seam

Runtime keys remain ordinary complete `GGLWEPrepared<D, BE>` objects. Existing
prepared wrapper types expose their physical key through
`GGLWEPreparedToBackendRef<BE>`; do not add a parallel source type,
projected-key type, or row view. Runtime never inverse-transforms or re-prepares
a selected use.

For a positive-precision result from section 3, the product may read only:

~~~text
logical rows = r_active
logical dsize = D
logical k_aux = a_eff
logical row width = logical_work_size

source row(i) = first_physical_row + i * row_step
              = (i + 1) * row_step - 1,  0 <= i < r_active
source limbs  = [0, logical_work_size)
~~~

This also applies when `D == d_p`: `r_eff` is a coverage bound, not a read
count. The complete prepared allocation may remain resident, but residency
does not authorize streaming its unused rows or suffixes.

Add matching scratch and execution methods equivalent to:

~~~rust
fn gglwe_product_dft_selected_tmp_bytes<K: GGLWEInfos>(
    &self,
    res_size: usize,
    input_size: usize,
    input_k: TorusPrecision,
    key: &K,
    effective_dsize: Dsize,
) -> usize;

fn gglwe_product_dft_selected(
    &self,
    res: &mut VecZnxDftBackendMut<'_, BE>,
    input: &VecZnxDftBackendRef<'_, BE>,
    input_k: TorusPrecision,
    key: &GGLWEPreparedBackendRef<'_, BE>,
    effective_dsize: Dsize,
    term_count: usize,
    scratch: &mut ScratchArena<'_, BE>,
);
~~~

Registry/helper construction validates selected uses. These methods call the
same resolver and use the same `ResolvedGGLWEUse`. Invalid direct calls must
fail explicitly and never fall back to the physical key's native decomposition.
Keep scratch and execution consistently infallible precondition APIs, as today,
or make both return `Result`.

Extend `GGLWEProductDigitsStridedImpl`, or add a sibling OEP, whose run and
`tmp_bytes` hooks receive these scalar values from the resolved use:

~~~text
effective_dsize, active_dnum, first_physical_row, physical_row_step,
logical_work_size, physical_pmat_rows, physical_pmat_size
~~~

Execution also receives `product_limbs` and the complete physical
`VmpPMatBackendRef`. The backend computes addresses from the full physical
shape and its own prepared layout. Physical size is the source row pitch;
`logical_work_size` is the per-row limb bound. There is no portable flat
prepared-layout formula and no generic `VmpPMat` row view.

Before OEP dispatch, validate the concrete prepared object against its physical
metadata: `key.data().size() == key.max_size()`, the `VmpPMat` degree, rows,
and columns match the resolved physical layout, and the backing span covers the
backend-authoritative `bytes_of_vmp_pmat` result. All row, column, pitch, and
byte-offset arithmetic is checked (or widened before comparison/conversion) so
malformed restored keys cannot turn valid resolver math into out-of-bounds addressing.

For positive precision, require:

~~~text
input_size = input.size() = ceil(input_k / key.base2k())
physical_row(i) = first_physical_row + i * physical_row_step
physical_flat_row(i, c) = physical_row(i) * physical_cols_in + c
digit_size(di) = ((input_size + di) / effective_dsize).min(active_dnum)
~~~

Logical metadata drives decomposition, output sizing, product terms, and
scratch. Physical metadata drives only bounds and addressing. This rule also
covers `effective_dsize == 1`; it must not route through a dense VMP path that
reads every physical row.

The selected product kernel must not dereference an unselected physical row or
a limb at or beyond `logical_work_size`. The target is reduced prepared-key
HBM/global-memory traffic, not reduced allocation size or selective host/device
transfer.

A backend may internally gather the selected prefixes into a temporary dense
prepared matrix and call its existing product, but that is an implementation
fallback, not a public materialization API or registry object. It must still
read only the selected prefixes and must not launch once per row.

CPU is the reference implementation. CUDA must preserve fused, pair/quad, and
cross-rotation batching by addressing selected rows inside batched kernels.
A correctness oracle materializes a dense key from the same selected prepared
parent bytes; do not compare bit-for-bit with an independently encrypted key,
whose masks and noise are randomized.

## 5. Registry construction and dispatch

Use a registration phase followed by immutable finalization:

~~~rust
pub struct GGLWEPhysicalKeyId(/* opaque */);

pub enum GGLWERegisterOutcome {
    Inserted(GGLWEPhysicalKeyId),
    Reused(GGLWEPhysicalKeyId),
}

pub struct GGLWEKeyRegistryBuilder<Id, K> { /* physical keys only */ }
pub struct GGLWEKeyRegistry<Id, K> { /* policy + immutable dispatch */ }

impl<Id: Clone + Eq + Hash, K: GGLWEInfos>
    GGLWEKeyRegistryBuilder<Id, K>
{
    pub fn find_subsuming<Q: GGLWEInfos>(
        &self,
        id: &Id,
        requested: &Q,
    ) -> Result<Option<GGLWEPhysicalKeyId>>;

    pub fn register(
        &mut self,
        id: Id,
        key: K,
    ) -> Result<GGLWERegisterOutcome>;

    pub fn finish(
        self,
        policy: GGLWEKeyUsePolicy,
    ) -> Result<GGLWEKeyRegistry<Id, K>>;
}

impl<Id: Clone + Eq + Hash, K: GGLWEInfos> GGLWEKeyRegistry<Id, K> {
    pub fn key_for(
        &self,
        id: &Id,
        k: TorusPrecision,
    ) -> Result<(&K, Dsize)>;

    pub fn try_map_values<K2: GGLWEInfos>(
        &self,
        map: impl FnMut(GGLWEPhysicalKeyId, &K) -> Result<K2>,
    ) -> Result<GGLWEKeyRegistry<Id, K2>>;
}
~~~

For each functional ID, degree, `base2k`, and input/output ranks are fixed;
`dsize`, `dnum`, and `k_aux` may differ. A registry covers one
cryptographic key domain, while `Id` identifies the function within it.

Registration validates layouts, reuses a subsuming parent without retaining a
logical alias, and retains non-subset decompositions. If several existing
parents subsume a registration request, choose the lexicographic minimum of:

~~~text
P, key.k(), dsize, dnum, k_aux, max_size,
stable physical-key ID
~~~

Finalization evaluates the policy for every table size and compiles:

~~~text
(function ID, size) -> physical key ID
~~~

Cells without a compatible physical key remain explicit errors. Higher-level
plans validate the cells they will use. Lookup is deterministic and O(1);
exact `k` is still checked against the dispatched key.

`try_map_values` preserves the policy, physical IDs, and dispatch table and
requires mapped keys to have exactly the source physical infos. This lets a
layout registry and its complete prepared-key registry compile identical
dispatch. Projected rows or effective logical layouts are never registry
values; execution derives them from the physical ID and shared resolver.

## 6. Automorphism and relinearization integration

Helpers own their policy; callers supply only function and exact precision:

~~~rust
pub trait GLWEAutomorphismKeyHelper<K, BE: Backend> {
    fn get_automorphism_key_for(
        &self,
        p: i64,
        k: TorusPrecision,
    ) -> Result<(&K, Dsize)>;
}

pub trait GLWEAutomorphismKeyLayoutHelper<L: GGLWEInfos> {
    fn get_automorphism_key_layout_for(
        &self,
        p: i64,
        k: TorusPrecision,
    ) -> Result<(&L, Dsize)>;
}

pub trait GLWERelinearizationKeyHelper<K, BE: Backend> {
    fn get_relinearization_key_for(
        &self,
        k: TorusPrecision,
    ) -> Result<(&K, Dsize)>;
}

pub trait GLWERelinearizationKeyLayoutHelper<L: GGLWEInfos> {
    fn get_relinearization_key_layout_for(
        &self,
        k: TorusPrecision,
    ) -> Result<(&L, Dsize)>;
}
~~~

Single-key adapters and `GGLWEKeyRegistry<i64, K>` or
`GGLWEKeyRegistry<(), K>` implement the corresponding helpers. A single-key
adapter still applies the policy and may use that key through whole-row
coarsening; it does not silently substitute its native `dsize`.

The current assumption that every automorphism key has one common physical
`GGLWELayout` must not be used for level-aware scratch sizing. Layout-only
planning resolves the actual rotations used and takes the maximum required
scratch. Prepared-key and layout helpers preserve physical IDs; selected row
geometry is resolved only when the operation executes.

Evaluation keysets expose helpers rather than one fixed RLK:

~~~rust
type RelinearizationKey: GGLWEInfos
    + GGLWEPreparedToBackendRef<BE>
    + GLWETensorKeyPreparedToBackendRef<BE>;

type RelinearizationKeys:
    GLWERelinearizationKeyHelper<Self::RelinearizationKey, BE>;

fn relinearization_keys(&self) -> &Self::RelinearizationKeys;
~~~

Standalone relinearization queries the exact tensor-input precision. CKKS
queries are:

~~~text
ckks_mul_into             max(a.k(), b.k())
ckks_mul_assign           max(dst.k(), a.k())
ckks_mul_prepared_assign  max(dst.k(), prepared.k())
ckks_square_into          a.k()
ckks_square_assign        dst.k()
~~~

For a DFT, resolve `D` once from each factor's input `k` and pin it for all
baby- and giant-step rotations in that factor. Physical parents may differ by
rotation, but the effective decomposition may not drift within the factor.
Scratch takes the maximum over the exact rotations used. This preserves shared
decomposition, hoisting, batching, and fusion.

EvalMod resolves the RLK immediately before each ciphertext multiplication or
square, so different levels may select different `D` and physical keys.

Thread the helpers through tensor/relinearize operations and scratch,
BSGS/polynomials, CKKS arithmetic/composites, EvalMod/LUT, bootstrap/PaCo
keysets, delegates, CPU, and CUDA. An execution registry holds complete
physical keys in the prepared domain; selection never requires a
coefficient-domain source or per-use preparation. Serialization may retain
Poulpy's existing canonical whole-physical-key representation. Any one-time
preparation after loading finishes before the execution registry is exposed
and is not part of selected use. If prepared bytes are persisted directly,
serialize a stable backend/prepared-layout/version fingerprint and reject a
mismatch. Serialize physical keys in stable physical-ID order, preserve those
IDs when restoring them, and rebuild dispatch deterministically. Do not
serialize selected rows, effective layouts, projected keys, or logical
aliases.
Missing coverage reports the function and exact precision without fallback.

## Acceptance checklist

1. Positive `(8,3)->(16,1)` and negative `(8,2)->(16,1)` containment
   tests pass.
2. Non-minimal padding is preserved, including
   `(8,4,8K+g)->(16,1,24K+g)`.
3. Dense policy indexing covers every declared size and never performs
   nearest-size fallback. The declared zero-precision entry performs no key
   read and launches no product kernel; overwrite, accumulate, and fused
   empty-product results still match their specified semantics.
4. A native-`dsize` use reads only `r_active` rows and their
   `logical_work_size` prefixes, not every capacity row.
5. For a prepared `dsize=8` key coarsened to `D=16`, one active digit
   addresses only physical row `1`. Device-side address instrumentation and
   poisoned rows `0` and `2` prove that they are not read.
6. With a sufficiently large parent and two active `D=16` digits, only
   physical rows `1` and `3` are addressed.
7. The direct selected product is bit-for-bit identical to a dense compact
   copy made from the same selected prepared-parent bytes. A freshly encrypted
   comparison key is tested only semantically unless its randomness is coupled.
8. Resolver, scratch, CPU, CUDA, and OEP agree on the logical layout, selected
   rows, active rows, and logical work size. Malformed prepared shapes and
   overflowing address calculations are rejected before OEP dispatch.
9. A debug key-load address trace contains no address outside selected ranges.
   Profiled prepared-key global-memory traffic tracks selected material `U`,
   rather than the physical key's full `dnum * max_size`.
10. Registry tests cover validation, subset deduplication, non-overlapping
   coexistence, the documented loss of a subsumed smaller-aux choice, compiled
   dispatch, missing cells, and deterministic ties.
11. `try_map_values` makes layout scratch and prepared execution select the
    same physical ID and exact physical metadata.
12. Automorphism/RLK helpers return `(&physical_key, effective_dsize)` and use
    exact `k`.
13. Every DFT factor pins one effective `dsize`; EvalMod may change RLK and
    `dsize` between multiplications.
14. Mixed-precision CUDA pair/quad and cross-rotation batching survives row
    skipping, issues no per-row product launches, and matches CPU.
15. Runtime key use starts from complete prepared keys and never re-prepares or
    inverse-transforms selected rows.
16. Bootstrap, PaCo, CKKS arithmetic, polynomials, EvalMod, and LUT compile
    against helper-based RLKs rather than a fixed tensor key.
17. The helper and registry contain no configurable total-modulus cap.
