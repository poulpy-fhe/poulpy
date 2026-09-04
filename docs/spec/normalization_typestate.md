# Coefficient Typestate, Canonicality, and Buffer Provenance

> **Status:** Proposed long-term design and migration plan
>
> **Scope:** `poulpy-hal`, `poulpy-core`, `poulpy-ckks`,
> `poulpy-bin-fhe`, backend crates, examples, tests, and benchmarks
>
> **Compatibility:** Deliberate public type/API break. The established
> normalization operation family is preserved.
>
> **Related contract:**
> [`poulpy-hal/docs/backend_safety_contract.md`](../../poulpy-hal/docs/backend_safety_contract.md)

This plan replaces the current normalization label with an enforceable ownership and
buffer-provenance model, and adds an independent canonicality axis for coefficient
representations. The objective is narrow and testable: outside a small unsafe backend
boundary, safe Rust must not be able to retain a `Normalized` or `Canonical` owner after
mutating its storage in a way that invalidates that label.

The design deliberately does not introduce a new normalization API. The existing
module-dispatched `vec_znx_normalize*`, `vec_znx_big_normalize*`, fused IDFT-normalize,
and `glwe_normalize*` methods keep their current names, receiver, argument order,
scratch arguments, and return types. Normalization operations keep their `()` return;
scratch-size queries keep their `usize` return. Only state bounds change to carry the
new generics.

---

## 1. Decisions fixed by this plan

The following are requirements, not open design questions.

1. Normalization remains on `Module` and remains infallible. In particular, this plan
   does not add a generic `normalize_consume` operation and does not replace the current
   scratch-explicit APIs.
2. No normalization, canonicalization, or typestate-validation operation is implemented
   as an inherent method on `VecZnx`, `GLWE`, `CKKSCiphertext`, a view, or a scratch
   wrapper. Public computation and state changes are dispatched through `Module`.
3. Canonicalization has exactly two public call shapes:

   ```rust,ignore
   module.make_canonical(&mut out, &input);        // returns ()
   let out = module.make_canonical_consume(input); // returns output directly
   ```

   There is no public canonicalization scan, checked-reference wrapper, assign variant,
   rounding variant, truncation variant, failure enum, or scratch-size query.
4. Normalization and canonicalization do not return `Result`. They are total for every
   initialized value admitted by their respective typed domain and the existing scratch contract.
   Invalid coefficient contents are not a recoverable error case.
5. There is no public validation-based typestate promotion. Arbitrary initialized bytes
   obtain stronger guarantees by running the enforcing transformation, not by scanning
   and conditionally relabelling them.
6. Structural parsing, I/O, allocation, parameter construction, and malformed-shape
   errors are outside this rule. They may continue to return their existing errors.
   They must not manufacture an arithmetic typestate on success without an enforcing
   write or a trusted producer contract.
7. An unchanged `*_normalize_assign(&mut T) -> ()` cannot strengthen `T`'s Rust type.
   Assign normalization changes bytes in place and leaves the static state conservative.
   Typed recovery from `Unnormalized` uses an existing out-of-place normalization method
   and a destination already typed to accept normalized output.
8. The existing fused name `vec_znx_idft_normalize_consume` is retained. Its `consume`
   suffix describes clobbering the DFT operand; it is not a general coefficient-owner
   typestate transition.

These decisions supersede earlier sketches that proposed receiver-side `.normalize(...)`,
a new one-argument normalization operation, or fallible proof/certification APIs.

---

## 2. Why the current design is unsound as a semantic API

### 2.1 Borrowed relabelling leaves the authoritative owner stale

Today the state parameter is attached to `VecZnx<D, W, S>`, and relabelling is available
for every `D`. That includes backend mutable-borrow buffers. This shape is therefore
legal in the current
[`VecZnx` layout](../../poulpy-hal/src/layouts/vec_znx.rs):

```rust,ignore
let mut owner: VecZnxOwned<_, Normalized> = /* ... */;

{
    let mut borrowed = owner.to_backend_mut().into_unnormalized();
    module.carry_producing_op(&mut borrowed, /* ... */);
}

// The owner still says Normalized although its digits may contain carries.
module.vec_znx_dft_apply(/* ... */, &owner.to_backend_ref(), /* ... */);
```

Weakening the borrow from `Normalized` to `Unnormalized` is locally truthful. The bug is
that the borrow is not the authoritative state label. Dropping it cannot update the
owner's type. This is the accidental channel that must be closed completely.

### 2.2 Safe raw mutation has the same effect

The current public `data` fields and methods such as `data_mut`, `raw_mut`, `at_mut`,
`from_data_like`, mutable scalar projection, and in-place `ReaderFrom` permit arbitrary
writes while preserving `S`. Deleting the unused `as_scalar_znx_mut` closes only one
entry point; it does not fix the class of defect.

Backends do need raw storage. The long-term boundary must distinguish a safe typed
application view from a trusted kernel capability instead of documenting all mutable
bytes as an informal exception.

### 2.3 Construction, scratch, and aggregates can forge labels

The current raw constructors label caller-provided bytes as normalized, reused scratch
is exposed through normalized views before initialization, and aggregates such as
`MatZnx`, `GGLWE`, and `GGSW` can create default-normalized child views without carrying
the parent's state. Each path independently defeats the advertised guarantee.

### 2.4 The invariants depend on runtime representation metadata

Normalization is relative to `base2k`. Canonicality is relative to both `base2k` and the
represented coefficient precision. Mutating `base2k` or `k` while retaining the marker
invalidates the proof even if the bytes do not change. The target type therefore carries
private immutable representation context along with its marker.

---

## 3. Semantic model

### 3.1 Orthogonal state axes

Use one sealed coefficient-state parameter with these legal forms:

```rust,ignore
pub struct Unwritten;
pub struct Raw;
pub struct Coeff<N: Normalization, C: Canonicality>(/* private */);

pub struct Normalized;
pub struct Unnormalized;

pub struct Canonical;
pub struct NonCanonical;
```

`NonCanonical` means “canonicality is not proven,” not “at least one padding bit is
known to be non-zero.” Likewise, `Unnormalized` is conservative: a value whose bytes
happen to be normalized may still carry that marker.

The state meanings are:

| State | Readable as coefficients | Arithmetic meaning |
|---|---:|---|
| `Unwritten` | No | Scratch/output storage whose complete logical contents have not been initialized |
| `Raw` | Yes, as words/bytes | Initialized storage with no arithmetic or canonicality claim |
| `Coeff<Unnormalized, C>` | Yes | An initialized limb representation accepted by total normalization; `C` describes its padding |
| `Coeff<Normalized, C>` | Yes | Every live digit satisfies the backend's documented normalized bound for the stored radix; `C` describes its padding |

Prepared, DFT, big-accumulator, and standalone scalar objects keep domain-specific states.
They must not be silently treated as `Coeff<N, C>`. Standalone scalar-bound certification
is outside this plan and requires a separate total Module API design if it is added; a
scalar must never inherit `Normalized` merely by being viewed as a one-limb `VecZnx`.

The public layout may expose `N` and `C` directly or wrap them in `Coeff<N, C>` behind
aliases. The implementation choice should minimize churn, but operation signatures and
compile-fail tests must make both axes visible. The recommended representation is one
`S` parameter containing `Coeff<N, C>` because existing backend conversion traits
already expose one associated `State`.

### 3.2 Normalized

`Normalized` means that each live signed limb digit is inside the exact interval assumed
by the backend's DFT/convolution preparation kernels for the value's immutable
`base2k`. Phase 0 must freeze the inclusive/exclusive endpoints from the reference
normalizer and use the same wording in HAL, Core, and backend contracts.

Normalization is total over every value admitted by the existing normalization
operation's arithmetic domain. `Raw` is not passed directly to normalization because it
lacks an arithmetic interpretation/context; a structural binding or full writer first
produces `Coeff<Unnormalized, NonCanonical>`. That binding examines no coefficient
predicate and has no coefficient-content failure. No carry-headroom certificate is a
precondition to calling normalization.

`Coeff<Unnormalized, C>` nevertheless carries a private conservative bound/headroom
certificate for *subsequent carry-producing operations*. Each such operation must
reserve and update its own headroom before mutation, use wider intermediates, or require
an earlier normalization. Exhausted headroom may prevent another carry operation, but
it never prevents or makes normalization fail.

### 3.3 Canonical

Let:

- `b = base2k`, with `0 < b <=` the backend coefficient-word width;
- `k = represented_k`, the precision governing this particular coefficient value;
- `L = ceil(k / b)`, the number of live limbs;
- limb `L - 1` be the least-significant live, or bottom, limb;
- `r = k % b`;
- `p = 0` when `r == 0`, otherwise `p = b - r`.

`Coeff<N, Canonical>` means that the low `p` bits of every coefficient word in the
bottom live limb are zero. When `p == 0`, every arithmetic representation is canonical
without changing bytes. `k == 0` cannot occur: every context is constructed with
`represented_k > 0`, so `L >= 1` and the bottom live limb always exists.

For a bottom-limb word `d`, canonicalization is the bit projection

```text
P_p(d) = from_bits(to_bits(d) & (!0 << p)).
```

The mask uses the unsigned bit representation of the backend word and then converts the
bits back without a numeric cast. The operation is applied to every coefficient of every
bottom live limb covered by the layout. All other live words are copied unchanged.
Inactive allocation capacity is not part of this canonicality guarantee and is preserved
by the out-of-place operation; a wire format that serializes inactive capacity must state
its own zeroing rule.

This definition fixes the semantics:

- canonicalization clears padding; it does not scan, reject, or choose nearest rounding;
- it is deterministic and idempotent;
- out-of-place and consuming forms are bit-identical apart from storage identity;
- for `p == 0`, the consuming form is a metadata-only state transition and the
  out-of-place form is an exact copy;
- it preserves `Normalized`, because clearing low bits of a signed normalized digit
  remains inside the normalized interval;
- it preserves the conservative `Unnormalized` marker for unnormalized input, while
  recomputing that root's private digit bound/headroom because a negative word may move
  downward by as much as `2^p - 1`;
- it is value-preserving under the declared `k` abstraction because the modified bits
  are, by definition, padding.

Existing decoding paths that use dirty padding bits to round a value must be classified
as raw/legacy decoding or changed to the canonical interpretation. Such rounding is not
part of `make_canonical`. The current
[`msb_mask_bottom_limb`](../../poulpy-core/src/default/operations/glwe.rs) helper is the
reference for bottom-limb orientation, but PR 0 must give the projection its own HAL
reference implementation rather than coupling it to a Core consumer.

### 3.4 Representation context

Every arithmetic root carries a private, immutable context sufficient to interpret both
markers:

```rust,ignore
struct CoeffContext {
    n: usize,
    cols: usize,
    live_limbs: usize,
    capacity_limbs: usize,
    base2k: usize,
    represented_k: usize,
    // backend/domain identity as required
}
```

An unnormalized root also carries private conservative arithmetic evidence, for example:

```rust,ignore
struct CarryCert {
    min_digit: i128,
    max_digit: i128,
    remaining_headroom: u32,
}
```

The concrete representation may use a cheaper symbolic bound, and it must not force
host metadata traffic for device values. Its sole purpose is to decide whether another
carry-producing operation is sound and to update that decision conservatively. The
normalizer neither requires nor rejects on this certificate. Structural binding of raw
words may start with zero additional headroom; normalization is still always available.

Names differ by layout: a `GLWE` uses its effective `k`, while some plaintext or encoded
objects use `encoded_k` or full physical precision. The owning aggregate decides which
field is `represented_k`; child coefficient views borrow that decision rather than
recomputing or defaulting it.

`base2k`, `represented_k`, live width, and capacity cannot be mutated through public
setters on a typed arithmetic value. A representation-changing module operation writes
into a destination with the target context and declares the resulting state. Merely
editing metadata yields `Raw`, never a preserved proof.

### 3.5 State weakening and compatibility

Define sealed proof-weakening relations:

```text
Normalized   fits Normalized and Unnormalized
Unnormalized fits Unnormalized only

Canonical    fits Canonical and NonCanonical
NonCanonical fits NonCanonical only
```

The product relation is used by copy and state-preserving operations. It permits a
stronger input to be written to a weaker destination without relabelling a borrow.
There is no public generic state-relabel method on layouts or views.

### 3.6 Normative invariants

The implementation is complete only if all of these hold:

1. A borrow has exactly the root's `S` and immutable representation context.
2. Ordinary `Ref` and `Mut` views cannot change `S`, even in the weakening direction.
3. Only an authoritative owner or scratch root may be consumed into a different state.
4. No safe mutable byte/word access exists for `Coeff<Normalized, _>` or
   `Coeff<_, Canonical>`.
5. Raw construction and deserialization produce `Raw`; dirty scratch produces
   `Unwritten`.
6. A zeroed full allocation may directly produce `Coeff<Normalized, Canonical>` because
   zero satisfies both invariants for every valid context.
7. A full writer/kernel may publish a strong state only when its sealed postcondition
   covers every logical word and all relevant metadata.
8. Every aggregate and subview propagates both axes; no default state is inferred while
   projecting a child.
9. Normalization methods retain their established module API and are direct/infallible.
10. Canonicalization is exposed only through the two `Module` calls in §5.
11. An output-parameter operation preserves the state already carried by its destination
    for the duration of the call; it never relies on relabelling that destination after
    the call.
12. A backend that accepts a safe typed operation is trusted to satisfy its declared
    per-write and final postconditions. Violating them is an unsafe backend defect.
13. Every arithmetic and normalization view is clamped to `live_limbs`. Inactive capacity
    is never allowed to participate in a carry chain or influence a live coefficient.
14. A mutable child that can produce carries holds a reservation borrowed from the
    authoritative root's headroom certificate. Parallel reservations are disjoint or
    conservatively joined before the root is used again.

---

## 4. Nominal storage and borrowing model

### 4.1 Public roles

Do not infer ownership from a generic data parameter such as `D`. Introduce nominal
roles over a private representation:

```rust,ignore
pub struct VecZnxOwned<B: Backend, S: CoefficientState> { /* private */ }
pub struct VecZnxRef<'a, B: Backend, S: CoefficientState> { /* private */ }
pub struct VecZnxMut<'a, B: Backend, S: CoefficientState> { /* private */ }
pub struct VecZnxScratch<'a, B: Backend, S: CoefficientState> { /* private */ }
```

Use the same pattern for `GLWE` and state-bearing aggregates. Internally the four roles
may share a `VecZnxRepr<D, W>`; the distinction must remain nominal at the public and
sealed-trait boundary.

- `Owned` owns both storage and the authoritative state label.
- `Ref` is read-only and inherits the root state.
- `Mut` has exclusive byte access at the Rust level but is not the authoritative label;
  it inherits the root state and cannot relabel.
- `Scratch` is the authoritative root of its arena region for its lifetime. It may be
  consumed between states, unlike an ordinary subview.

A short-lived `DataOwned` bound is acceptable as an intermediate containment patch, but
it is not the target. Backend-owned buffers, mapped device buffers, borrows, and scratch
regions cannot be classified reliably from the shape of `D` alone.

### 4.2 Borrowing

Borrowing is state invariant:

```rust,ignore
fn as_ref(&self) -> VecZnxRef<'_, B, S>;
fn as_mut(&mut self) -> VecZnxMut<'_, B, S>;
```

Neither returned type implements a public state-changing conversion. A mutable child of
a canonical aggregate is canonical and can be passed only to operations that preserve or
re-establish canonicality before each write. A carry-producing operation cannot accept a
`Mut<Normalized, _>` merely by weakening the view.

Subview methods for a matrix, GGLWE, GGSW, ciphertext, key, or packed vector return
`Ref<..., S>`/`Mut<..., S>` with the same `S` and a borrowed child context. Any aggregate
whose children use different `represented_k` values must store those contexts explicitly.

### 4.3 What may remain inherent

Layout types may retain passive, non-mutating accessors such as `n()`, `size()`, `k()`,
`base2k()`, and state-preserving borrows. Allocation, computation, normalization,
canonicalization, full writes, metadata conversions, and state transitions are module
operations. In particular, remove the inherent normalization implementations currently
on:

- `VecZnx<..., Unnormalized>` in
  [`poulpy-hal/src/layouts/vec_znx.rs`](../../poulpy-hal/src/layouts/vec_znx.rs);
- `VecZnxViewMut<..., Unnormalized>` in
  [`poulpy-hal/src/layouts/scratch_views.rs`](../../poulpy-hal/src/layouts/scratch_views.rs);
- `GLWE<..., Unnormalized>` in
  [`poulpy-core/src/layouts/glwe.rs`](../../poulpy-core/src/layouts/glwe.rs);
- `GLWEViewMut<..., Unnormalized>` in
  [`poulpy-core/src/layouts/scratch_views.rs`](../../poulpy-core/src/layouts/scratch_views.rs);
- `CKKSCiphertext<..., Unnormalized>` in
  [`poulpy-ckks/src/layouts/ciphertext.rs`](../../poulpy-ckks/src/layouts/ciphertext.rs);
- the crate-private mutable CKKS normalization wrapper.

There is no inherent `make_canonical`, no free public alias for it, and no extension trait
implemented on a layout receiver.

### 4.4 Output parameters and panic safety

An output parameter already has a Rust state before a module method begins. A method that
accepts `&mut Coeff<Normalized, Canonical>` must keep every observable write inside both
invariants; it cannot fill dirty bytes and repair them just before return. This rule makes
the owner safe even if a reference backend panics between stores.

For operations whose natural intermediate violates the destination state, use one of:

1. an `Unwritten` output builder that is not readable until a sealed full-write commit;
2. a separate `Unnormalized`/`NonCanonical` scratch root followed by an existing
   out-of-place normalizer and, when needed, an explicit `make_canonical*` call;
3. a kernel that computes through wider/private temporaries and stores only values that
   satisfy the destination state.

The old pattern—borrow a normalized owner as unnormalized, write carries, normalize the
borrow, and let it drop—is forbidden even if every current caller remembers the final
pass.

---

## 5. Public module API

### 5.1 Module-only rule

Safe application-facing operations are traits implemented for `Module<B>`. Layout types
are operands, never computational receivers. Backend free functions may exist behind the
OEP boundary, but they are not re-exported as public application API.

### 5.2 Preserve the normalization family

The following public surface is retained:

| Layer | Existing methods retained |
|---|---|
| HAL | `vec_znx_normalize_tmp_bytes`, `vec_znx_normalize`, `vec_znx_normalize_coeff_backend`, `vec_znx_normalize_assign_backend`, `vec_znx_normalize_coeff_assign_backend` |
| HAL big | `vec_znx_big_normalize_tmp_bytes`, `vec_znx_big_normalize` |
| HAL fused DFT | `vec_znx_idft_normalize_consume_tmp_bytes`, `vec_znx_idft_normalize_consume` |
| Core | `glwe_normalize_tmp_bytes`, `glwe_normalize`, `glwe_normalize_assign` |

CKKS and BinFHE do not gain duplicate normalization traits. They use the Core/HAL
operations as they do today.

The current declarations are in
[`poulpy-hal/src/api/vec_znx.rs`](../../poulpy-hal/src/api/vec_znx.rs),
[`poulpy-hal/src/api/vec_znx_big.rs`](../../poulpy-hal/src/api/vec_znx_big.rs),
[`poulpy-hal/src/api/vec_znx_dft.rs`](../../poulpy-hal/src/api/vec_znx_dft.rs), and
[`poulpy-core/src/api/operations.rs`](../../poulpy-core/src/api/operations.rs). These
files are the source of truth for the API-shape tests in PR 0.

The same shape-preservation rule applies to the public backend normalization hooks in
[`poulpy-hal/src/oep/hal_impl.rs`](../../poulpy-hal/src/oep/hal_impl.rs) and
[`poulpy-core/src/oep/operations.rs`](../../poulpy-core/src/oep/operations.rs): method
names, argument order, scratch, and returns remain stable. Their sealed state bounds and
the state-bearing buffer capability types may gain the new generics.

For example, the Core call shape remains:

```rust,ignore
pub trait GLWENormalize<BE: Backend> {
    fn glwe_normalize_tmp_bytes(&self) -> usize;

    fn glwe_normalize<R, A>(
        &self,
        res: &mut R,
        a: &A,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE>,
        A: GLWEToBackendRef<BE>;

    fn glwe_normalize_assign<R>(
        &self,
        res: &mut R,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE>;
}
```

And the HAL column operation keeps its existing parameters and order:

```rust,ignore
fn vec_znx_normalize(
    &self,
    res: &mut VecZnxBackendMut<'_, B, SRes>,
    res_base2k: usize,
    res_offset: i64,
    res_col: usize,
    a: &VecZnxBackendRef<'_, B, SIn>,
    a_base2k: usize,
    a_col: usize,
    scratch: &mut ScratchArena<'_, B>,
);
```

`SIn`/`SRes` above illustrate the only intended signature evolution: sealed state bounds
replace `impl NormalizationState` so both normalization and canonicality are tracked.
No parameter disappears, moves into `Module`, or changes order, and the return remains
`()`. The same rule applies to coefficient, assign, big, and fused variants.

### 5.3 Normalization state effects

Add sealed relations expressing the state already guaranteed by each unchanged
normalization operation and whether that state fits the destination. Conceptually:

```rust,ignore
NormalizeEffect<Operation, A::State>: FitsIn<R::State>
```

Its contract is:

- the produced live digits are normalized for the destination radix;
- the accepted source domain remains the domain of the existing numeric operation,
  extended only by the new normalization/canonicality generics;
- every out-of-place, coefficient, big-to-coefficient, and fused normalization output is
  typed `Coeff<Normalized, NonCanonical>` (or a weaker destination state)
  conservatively, even when particular bytes happen to have zero padding;
- normalization does not mask padding and never establishes `Canonical`;
- normalization reads exactly the declared live limbs; inactive capacity is ignored and
  cannot feed a carry into the bottom live limb;
- an out-of-place destination carrying `Canonical` is not accepted; the caller
  normalizes into a `NonCanonical` destination and then calls `make_canonical` or
  `make_canonical_consume`;
- a weaker destination may accept stronger produced bytes through `FitsIn` without any
  relabelling;
- no data-dependent branch returns an error.

This lets an existing out-of-place call provide typed recovery:

```rust,ignore
let mut normalized = module.glwe_alloc_from_infos(/* Normalized, NonCanonical */);
module.glwe_normalize(&mut normalized, &unnormalized, &mut scratch);
```

The allocator or output builder determines `normalized`'s state; the call does not
change its type. If the next consumer requires canonical padding, compose explicitly:

```rust,ignore
let normalized = module.make_canonical_consume(normalized);
```

The assign variants are the sole normalization exception on the canonicality axis
because they do not change storage or representation context:

- assigning into `Coeff<Normalized, C>` preserves that state;
- assigning into `Coeff<Unnormalized, C>` leaves the owner statically unnormalized even
  though its bytes are now normalized;
- canonicality is preserved only because assign normalization uses the same immutable
  representation context and the reference identity in §6.2; it is not created or
  repaired.

Therefore `glwe_normalize_assign` and `vec_znx_normalize_assign_backend` are byte
operations, not typestate promotions. A zero-copy `Unnormalized -> Normalized` owner
promotion would require an additional consuming API, which is intentionally not part of
this design.

### 5.4 The only canonicalization API

Expose the two forms through sealed operation traits implemented by `Module`:

```rust,ignore
pub trait MakeCanonical<BE: Backend>: private::Sealed {
    fn make_canonical<Out, In>(&self, out: &mut Out, input: &In)
    where
        Out: MakeCanonicalOutput<BE>,
        In: MakeCanonicalInput<
            BE,
            Normalization = Out::Normalization,
            Layout = Out::Layout,
            ContextBrand = Out::ContextBrand,
        >;
}

pub trait MakeCanonicalConsume<BE: Backend, Input>: private::Sealed {
    type Output;

    fn make_canonical_consume(&self, input: Input) -> Self::Output;
}
```

The exact supporting trait names are implementation details. The public calls and their
semantics are fixed:

```rust,ignore
module.make_canonical(&mut out, &input);
let output = module.make_canonical_consume(input);
```

Both operations:

- accept `Coeff<N, C>` for either normalization marker and either canonicality marker;
- produce/write `Coeff<N, Canonical>` and preserve `N` exactly;
- use the input/target representation context described in §3.4;
- apply `P_p` directly, with no validation scan and no rounding;
- have no scratch argument and perform no allocation or host/device transfer;
- return directly and have no failure type.

For `N = Unnormalized`, both forms update the private carry certificate to cover the
projected bounds. The consuming form replaces the root certificate after applying
`P_p`. The out-of-place form installs a conservative join of the old destination bound
and projected input bound before its first write, then may narrow it after completion.
This metadata update is deterministic and cannot reject the input.

The out-of-place operands have an identical representation context: layout family, word
format, radix, `represented_k`, live width, capacity, and module/backend domain. The
recommended implementation gives values built for the same context a non-forgeable
context brand and requires that brand in the sealed input/output traits. During a
transition period, any runtime equality assertion is a structural programmer precondition,
not a coefficient-content error or `Result`. The backend regions must not overlap.

The output is already typed `Coeff<N, Canonical>`, so the kernel stores masked bottom
words directly. It never copies a dirty bottom word and repairs it in a second pass.

The consuming form is the sole public in-place canonicality promotion. It is implemented
only for authoritative `Owned` and `Scratch` roots and returns the same storage role and
layout with `C = Canonical`. It is not implemented for `Ref`, ordinary `Mut`, or a child
view: state-changing consumes are root-only, and a child proof cannot strengthen the
aggregate root even though leaving a conservative parent label would not itself be
false.

`Raw`, `Unwritten`, scalar, DFT, big, and prepared objects are outside the public
canonicalization trait bounds. Raw coefficient input first goes through a structural
binding/full-write path that establishes `Coeff<Unnormalized, NonCanonical>`, then an
existing normalization operation. Canonicality is established only by a trusted
canonical producer or one of the two methods above.

### 5.5 No error-bearing validation or certification API

There is no public operation that scans arbitrary bytes and returns a conditionally
relabelled owner/reference. Such an API either needs a negative result or becomes
unsound; neither is needed when enforcing transformations are total.

Consequently this plan contains no normalization failure, canonicality failure,
validation failure, checked canonical reference, or padding-conversion error. It also
contains no public `try_*` form for these state changes.

Tests and debug backend builds may use internal observational predicates that return a
boolean or report. Those helpers never alter typestate and are not part of the stable
application API. If public diagnostics are later requested, they require a separate API
proposal and must remain observational rather than returning a stronger typed value.

### 5.6 Transition matrix

| Source | Module operation | Destination/effect |
|---|---|---|
| zeroed `Unwritten` output | sealed full-write commit | `Coeff<Normalized, Canonical>` |
| initialized `Raw` | structural binding/full write | `Coeff<Unnormalized, NonCanonical>` with a conservative carry certificate; no coefficient-content failure |
| `Coeff<Unnormalized, C>` | existing out-of-place normalize | `Coeff<Normalized, NonCanonical>` (or a weaker destination state) |
| `Coeff<N, C>` | existing assign normalize | static state remains `Coeff<N, C>` |
| authoritative `Coeff<Normalized, C>` root | consuming weakening (`into_unnormalized`) | same root/storage as `Coeff<Unnormalized, C>` with a conservative carry certificate; bytes unchanged |
| authoritative `Coeff<N, Canonical>` root | consuming weakening (`forget_canonical`) | same root/storage as `Coeff<N, NonCanonical>`; bytes unchanged |
| `Coeff<N, C>` | `make_canonical(out, input)` | writes `Coeff<N, Canonical>` |
| authoritative `Coeff<N, C>` root | `make_canonical_consume(input)` | same root/storage as `Coeff<N, Canonical>` |
| any arithmetic state | immutable borrow | same state and context |
| any arithmetic state | mutable borrow | same state and context; no relabel |
| any initialized state | raw-state consuming downgrade, if retained | authoritative `Raw` root; never a borrowed relabel |
| any state | metadata/precision conversion | separately typed output chosen by the module operation |

No row promotes a borrowed state. No row requires a coefficient-content error path.
The two consuming weakenings are the only state-weakening operations; they exist on
authoritative owner and scratch roots only, never on `Ref`, `Mut`, or aggregate
children. They are how an existing normalized value becomes a legal carry-producing
destination (for example, accumulating onto an already-populated ciphertext).

---

## 6. Operation integration

### 6.1 Destination state is a write contract

Every mutating operation must say which invariant it requires of its inputs and which
invariant it maintains in its destination. Output parameters do not acquire a state at
the end of the call; they already have one, and the implementation must respect it while
writing.

Use three patterns:

| Operation class | Type rule |
|---|---|
| State-preserving copy/permutation | every input state must `FitIn` the destination state under the same representation context |
| Carry-producing arithmetic | destination normalization is `Unnormalized`; canonicality is derived from the operation's exact algebra |
| State-producing full write | destination is an `Unwritten` slot or already-valid state that the kernel maintains per store; the sealed operation declares the postcondition |

Do not use `impl NormalizationState` as a catch-all once the new algebra exists. Each
trait must mention the appropriate sealed relation so a new marker cannot accidentally
be accepted by an old blanket implementation.

A carry-producing mutable child additionally receives a reservation derived from its
authoritative root's `CarryCert`. A successful operation spends/updates that reservation;
parallel children partition the budget and join conservative bounds before returning
control to the root. None of this evidence is consulted by normalization.

### 6.2 Canonicality effects

Canonical words form a grid of multiples of `2^p` in the bottom limb. This supports
several mechanically provable rules, but the audit must still be per operation:

| Operation | Canonicality rule |
|---|---|
| Exact copy into the same context | Preserves `C` |
| Coefficient/column permutation, rotation, automorphism | Preserves `C` when it does not change the representation context |
| Addition or subtraction | Produces `Canonical` only when every contributing bottom limb is canonical under the same `p`; otherwise `NonCanonical` |
| Negation | Preserves `Canonical` for a fixed context |
| In-place assign normalization in a fixed context | Preserves `C`; it does not repair dirty padding or promote `N` |
| Any out-of-place/coefficient/big/fused normalization | Conservatively produces `NonCanonical`; it never repairs padding |
| Radix, offset, precision, or live-width conversion | Produces `NonCanonical` |
| Sampling/encoding/full zero | Producer-specific; claim `Canonical` only if every bottom word is masked |
| Raw overwrite/deserialization | `Raw` |
| `make_canonical*` | Always produces `Canonical`, preserves `N` |

For a fixed representation context, canonicalization and normalization must commute:

```text
normalize(P_p(x)) == P_p(normalize(x)).
```

This identity receives a reference proof and property tests and justifies canonicality
preservation for fixed-context assign normalization. Out-of-place, cross-radix, and
offset normalization remain typed `NonCanonical`; callers invoke one of the two
canonicalization methods explicitly.

### 6.3 Consumer requirements

Consumers are classified independently on both axes:

- DFT/convolution entry points that assume bounded digits require `Normalized`.
- Any consumer that treats `k` as the effective precision either requires `Canonical`
  or explicitly defines a mask-on-read compatibility path.
- Physical bytewise equality/hash may accept any initialized state because it compares
  storage, not semantic values.
- A unique *semantic coefficient encoding* requires at least
  `Coeff<Normalized, Canonical>` plus format-specific normalized-endpoint, top-overflow,
  metadata, and inactive-capacity rules. `Canonical` alone proves only bottom padding.
- Operations that are mathematically insensitive to padding may accept either
  canonicality marker; their documentation must say so.
- Standalone scalar preparation remains outside this coefficient-state design and must
  not inherit a `Normalized` proof merely because a scalar can be viewed as a one-limb
  `VecZnx`.

The final API should prefer canonical-only precision-sensitive consumers. During
migration, existing defensive masks may remain, but they cannot be cited as proof that a
dirty value is canonical and they should be removed from hot paths only after parity and
benchmark gates pass.

### 6.4 Replacing current in-place fusion patterns

The production sites that currently relabel a normalized mutable borrow fall into three
groups.

**Scratch accumulator.** Take an authoritative `Unwritten` scratch root, initialize it
as `Coeff<Unnormalized, C>`, perform carry-producing operations, then use the existing
out-of-place normalization API to write the final typed destination.

```rust,ignore
let mut accum = module.take_glwe_scratch_unwritten(/* ... */);
let mut accum = module.initialize_accumulator(accum, /* ... */);
module.glwe_add_assign(&mut accum, &term, &mut scratch);

let mut out = module.glwe_alloc_from_infos(/* ... */);
module.glwe_normalize(&mut out, &accum, &mut scratch);
```

**Caller-provided normalized destination.** Compute in separate unnormalized scratch and
normalize into the destination. If computation panics, the caller's old destination
continues to satisfy its type. The final normalizer writes a destination typed
`Coeff<Normalized, NonCanonical>` and makes only normalized stores; canonicalization, if
required, is the next explicit module call.

**Full writer.** Sampling, key generation, decoding, or a fused backend operation that
completely determines every word uses a sealed `Unwritten` builder. It commits directly
to the state its postcondition proves. It does not borrow a pre-labelled owner and
relabel the borrow.

These replacements intentionally do not call `*_normalize_assign` and then cast. Assign
normalization remains useful for algorithms whose root is already conservatively typed,
but it is not an exit from the unnormalized typestate.

### 6.5 Full-write protocol

A full-write producer receives a non-readable capability rather than `&mut [u8]`:

```rust,ignore
#[doc(hidden)]
pub struct OutputSlot<'a, B, Layout, ProducedState> {
    /* private storage, shape, context, completion state */
}
```

Like the kernel capability in §9.1, `OutputSlot` is public but doc-hidden and
non-constructible outside `poulpy-hal`: full-write producers (samplers, key generation,
fused kernels) live in sibling backend crates and must be able to receive it, while
safe scheme code can neither construct one nor read through it before commit.

The protocol must ensure:

1. no safe read before complete initialization;
2. exactly the promised logical region is written;
3. a panic drops or returns the region as `Unwritten`, never as `ProducedState`;
4. commit is available only to the sealed module/backend implementation;
5. a partial column/entry writer cannot commit the entire aggregate;
6. device completion is observed before the strong state becomes usable.

Debug builds should track written ranges or columns. Release builds may use a zero-cost
linear builder when the operation structure statically covers the full layout.

### 6.6 Narrowed working widths

Working-width narrowing keeps its current runtime behavior. Operations today accept
views whose working size is smaller than the allocation (the `with_size` helpers,
min-size kernel conventions, leveled operation windows, and gadget-product limb
windows), and this plan does not change those semantics: the rewrite adds compile-time
state bounds only. A narrowed view remains an ordinary state-preserving borrow whose
working size is part of its shape, exactly as today, and the `live_limbs` clamp of
invariant 13 is realized through that existing size convention rather than a new
runtime mechanism.

One axis is conservative: a narrowed view carries `NonCanonical` unless the narrowing
preserves the value's bottom live limb, because the padding predicate is relative to
the represented precision and a shorter window has a different bottom limb. The
normalization marker is preserved by narrowing, since every retained limb digit keeps
its bound.

PR 0 must verify this compatibility explicitly: inventory every narrowing site and
confirm that the state rules in this document leave its runtime behavior byte-identical.
If any rule here is found to conflict with an existing narrowing path, the conflict is
resolved by amending this design first; the migration must not silently change
working-width semantics.

---

## 7. Raw access, construction, and serialization

### 7.1 Safe raw boundary

Make coefficient storage fields private. Safe mutable byte/word access is available only
on `Raw` roots and `Unwritten` output slots. Read-only byte access may be exposed for
arithmetic states if it cannot be converted back into mutable storage and the wire
semantics are documented.

The following current routes must be removed, restricted, or moved behind the backend
boundary:

- public `data` fields on state-bearing coefficient layouts;
- `DataViewMut::data_mut` from an arithmetic state;
- `ZnxViewMut::{raw_mut, at_mut}` through application-facing traits;
- state-preserving `from_data_like`/mutable mapping;
- mutable scalar projection from `VecZnx` (the currently unused
  `as_scalar_znx_mut` should be deleted);
- `ReaderFrom` implementations that overwrite an arithmetic state in place;
- aggregate `data_mut`/`at_mut` methods that synthesize a default child state.

If an application must edit bytes, a module operation consumes the authoritative root
to `Raw` or allocates a `Raw` output. There is no borrowed raw escape followed by an
implicit restoration of the old label.

Once a `Raw` root already has a structurally valid word layout and immutable coefficient
context, a sealed Module binding may consume it as
`Coeff<Unnormalized, NonCanonical>` without changing bytes. This step is total and
installs a conservative certificate with no additional carry headroom. It interprets
initialized words; it does not validate a requested numeric bound, normalize, or make
the value canonical.

### 7.2 Construction

Classify constructors by what they prove:

| Construction path | Initial state |
|---|---|
| Zeroed allocation from valid infos | `Coeff<Normalized, Canonical>` |
| Uninitialized/reused arena region | `Unwritten` |
| Arbitrary initialized bytes or backend buffer | `Raw` |
| Exact deep clone/transfer | Same state and representation context |
| Trusted sampler/encoder/full kernel | Declared `Coeff<N, C>` postcondition |
| Shape-only wrapper over bytes | `Raw`, never a default arithmetic state |

Because zero satisfies every arithmetic weakening of
`Coeff<Normalized, Canonical>`, module allocators may return a caller-selected state that
is inferred from the destination use. During compatibility migration, the canonicality
default is `NonCanonical`, so existing out-of-place normalization calls do not
accidentally acquire a canonical destination. Code that needs the strongest zero proof
requests it explicitly.

Unsafe `from_raw_parts` remains possible for backend implementors, but its safety
contract includes shape, initialization, state, context, and aliasing. Safe public
constructors cannot take a caller-chosen `N` or `C` marker.

### 7.3 Decoding and encoding

Separate structural decoding from coefficient-state enforcement:

1. The reader validates byte length, version, dimensions, and parameter encodings using
   its normal I/O/format error type.
2. Successful decoding yields `Raw` initialized storage with immutable representation
   context.
3. A total Module structural binding/full writer produces
   `Coeff<Unnormalized, NonCanonical>` without inspecting coefficient contents.
4. The caller uses an existing module normalization operation to write a
   `Coeff<Normalized, NonCanonical>` destination.
5. If canonical padding is required, the caller explicitly uses one of the two
   `make_canonical` forms.

This sequence has no coefficient-validation `Result`. A malformed stream may still be a
decode error; arbitrary coefficient words in a structurally valid stream are handled by
the total transforms.

Writers that promise a unique semantic coefficient encoding require
`Coeff<Normalized, Canonical>` and a format-specific readiness contract covering the
normalized endpoint/top-overflow convention, metadata, and inactive capacity. The
generic `Canonical` marker alone does not certify unique bytes. Legacy formats that
preserve dirty padding must accept `Raw`/`NonCanonical` explicitly and cannot advertise
canonical padding.

### 7.4 Copies and backend transfer

An exact copy or host/device transfer preserves `N` and `C` only if it also preserves the
representation context bit-for-bit. A transfer that changes word format, radix, active
width, or precision is a conversion operation with an explicit destination state.

No transfer implementation may reconstruct a state-bearing layout by calling an
unchecked default-state constructor. The state and context travel together through a
sealed conversion trait.

---

## 8. Scratch and aggregate propagation

### 8.1 Scratch lifecycle

Every arena take starts as `Scratch<Unwritten>`, regardless of what bytes happen to be in
the reused region. Valid progressions are:

```text
Unwritten scratch --full write--> Coeff<N, C> scratch
Unwritten scratch --raw fill----> Raw scratch
Coeff<N, C> scratch --make_canonical_consume--> Coeff<N, Canonical> scratch
```

Normalization of scratch follows the same unchanged APIs as owned storage. Assign
normalization does not promote its static state. An out-of-place normalizer may write a
separate scratch destination already typed with the required state.

Ordinary subviews cut from a scratch root are still non-authoritative `Mut` values and
cannot consume/relabel independently. Only the root token may commit a whole-layout
state transition.

### 8.2 Partial initialization

Current algorithms frequently fill a scratch object one column, row, or key entry at a
time. Support that without lying about initialization:

- the root remains `Unwritten` while entry capabilities are issued;
- each entry capability is affine and tied to a root completion tracker;
- an entry's strong state is joined into the aggregate state;
- commit requires every logical entry exactly once;
- if any entry is `Unnormalized` or `NonCanonical`, the whole aggregate uses the weaker
  marker;
- inactive capacity is tracked separately from logical initialization.

For simple fixed loops, a sealed `write_all_columns` module helper can hide the tracker.
It is an internal construction mechanism, not another normalization or canonicalization
API.

### 8.3 Aggregates in scope

Propagate `S` through every coefficient-domain container and its views, including at
least:

- `MatZnx` and its row/column/entry views;
- `GLWE`, `GGLWE`, `GGSW`, LWE bodies/masks, and tensor layouts;
- GLWE plaintexts, ciphertexts, public keys, switching keys, automorphism keys, tensor
  keys, with compressed forms handled by the expansion rule below;
- CKKS ciphertext wrappers and evaluation/key aggregates;
- BinFHE/FHE integer containers that splice or expose GLWE children;
- seed-expanded and decompressed layouts before preparation.

Prepared/DFT-domain layouts need not carry coefficient `N/C` once conversion has
consumed the coefficient input, but their constructors must require the correct input
state. Converting back to coefficients declares a fresh coefficient state.

### 8.4 Aggregate canonicality

An aggregate is `Canonical` only when every logical coefficient child is canonical under
that child's represented precision. A single noncanonical child weakens the whole root.

`make_canonical(out, input)` traverses the complete logical aggregate and applies each
child's context. `make_canonical_consume(input)` may promote only the complete
authoritative aggregate root; consuming one child view cannot promote its parent. If an
algorithm needs only one child canonical, it writes that child into a separately owned
canonical destination.

Compressed/seeded objects are excluded from both generic `make_canonical` input domains:
their implicit expanded coefficients cannot be projected in place while preserving the
same storage/layout. Expansion/decompression is a full-write module operation into an
uncompressed destination. Its output state is determined by the stored explicit terms,
the versioned expander, and their existing bytes; if that output is noncanonical, the
caller then invokes one of the two canonicalization methods on the uncompressed value.
If a particular compressed format proves that every possible expansion is canonical,
that fact belongs to its format-specific output contract, not a generic in-place
canonicalization implementation.

---

## 9. Backend/OEP contract

### 9.1 Narrow unsafe capability

Backend kernels receive a public-but-doc-hidden, nonconstructible capability that
exposes raw regions together with a declared input/output contract. Sibling backend
crates can name it in OEP implementations, but safe scheme code cannot construct it or
extract an unrestricted mutable slice.

Each unsafe kernel contract records:

- accepted input state and representation context;
- any root carry reservation consumed and the conservative bound returned;
- state guaranteed for each destination region;
- whether it is a full write or state-preserving update;
- initialized/live/capacity ranges;
- permitted aliasing and overlap;
- scratch size/alignment;
- behavior under unwind or device-launch failure;
- synchronization point at which the output state becomes usable.

The storage provider itself is also a trusted boundary. `Backend::OwnedBuf` must denote
authoritative storage (or a documented reference-counted copy-on-write equivalent),
deep-copy operations must not retain writable aliases, and the `BufRef`/`BufMut` GATs
must uphold Rust aliasing for every lifetime. Express this as a sealed unsafe backend
contract and test it independently of numeric kernels. The raw capability can be
public/doc-hidden so sibling backend crates can receive it, but its fields and safe
constructors remain private to HAL; safe scheme code cannot manufacture one.

The safe `Module` delegate performs structural preflight, creates the capability, calls
the backend, and exposes only the state promised by the trait.

### 9.2 Relabelling OEP

Remove the generic safe `SetNormalizationState` escape hatch in the final design.
During migration, both strengthening and weakening methods on borrowed buffers are
`unsafe` and crate-private. The weakening direction is numerically safe but still
dangerous on a borrow because subsequent writes can invalidate the parent.

A fused kernel should not “set” a state after arbitrary mutation. Its OEP trait declares
the produced state, and the safe delegate provides an appropriate output slot or
already-valid destination. This makes the proof local and greppable.

### 9.3 Total normalizer contract

Every backend normalization implementation must agree with the reference implementation
for every initialized input word pattern in its supported word/radix domain. It returns
directly and has no input-data failure case. The existing tmp-byte query remains the
source of its scratch requirement.

The audit must specifically cover:

- the exact signed normalized interval;
- top-limb overflow/discard semantics;
- same-radix and cross-radix paths;
- positive and negative `res_offset`;
- coefficient-only and assign variants;
- big-to-coefficient normalization;
- fused IDFT normalization;
- declared canonicality effects: fixed-context preservation and conservative
  context-changing output;
- strict clamping to `live_limbs`, with adversarial inactive capacity unable to
  contribute carries;
- partial-column writes and unwind safety.

### 9.4 Canonical kernel contract

Backends implement two internal primitives corresponding to the two public calls:

1. disjoint streaming copy with `P_p` on bottom words;
2. authoritative in-place `P_p` for the consuming form.

They require no scratch allocation. For device storage they execute device-locally and
must not cause an implicit host round trip. The safe return occurs only once the backend
completion model makes the state guarantee true.

At `p == 0`, the in-place primitive does no data work. The out-of-place primitive still
performs the required exact copy.

---

## 10. Migration strategy

The migration is deliberately vertical. Each pull request leaves the workspace with a
coherent safety story for the paths it converts; compatibility shims are removed rather
than allowed to become permanent alternate APIs.

### PR 0 — Freeze contracts and inventory the workspace

Status: delivered in [normalization_inventory.md](normalization_inventory.md) (frozen
contracts, site inventory, baselines, deny-list ratchet); the performance/scratch/binary
baselines there are pending dedicated hardware and gate PR 1.

Deliverables:

- write reference definitions for normalized bounds and `P_p`;
- record the exact existing normalization signatures and add API-shape tests;
- inventory all `into_unnormalized`, inherent `.normalize`, public raw mutation,
  unchecked constructors, `ReaderFrom`, scratch-take, aggregate-child, and metadata
  setter sites;
- classify every normalization path as out-of-place, assign, coefficient, big, or fused;
- classify every padding-sensitive consumer and existing `msb_mask_bottom_limb` call;
- inventory every working-width narrowing site and confirm the state rules leave its
  runtime behavior byte-identical (§6.6), amending this design first on any conflict;
- capture correctness, scratch-size, binary-size, and performance baselines;
- add a temporary deny-list CI grep so new bypasses do not land during migration.

Exit gate: reviewers agree on the numeric bounds, bottom-limb orientation, meaning of
`represented_k` for every root layout, and the exact list of normalization APIs that
must remain source-shaped.

### PR 1 — Introduce the state algebra without changing kernels

Status: delivered in `poulpy-hal/src/layouts/coeff_state.rs` (sealed algebra, relations,
compatibility aliases, `CoeffContext`, `CarryCert`, reference `P_p` with property
tests). The backend-ref/mut traits reach the new algebra through the
`NormalizationState::AsCoeff` bridge rather than a new required associated type on every
impl; the explicit state/context associated types land with the nominal roots in PR 2 to
avoid double churn (sanctioned by §3.1's churn-minimization license). The existing
normalization where-clauses needed no adaptation because roots still carry
`NormalizationState`; kernels and behavior are untouched.

Deliverables:

- add sealed `CoefficientState`, `Normalization`, `Canonicality`, and product `FitsIn`
  relations;
- introduce `Unwritten`, `Raw`, `Coeff<N, C>`, and compatibility aliases;
- add private immutable representation context;
- define the private `CarryCert`, child-reservation, and parallel-join rules, explicitly
  excluding the certificate from normalization preconditions;
- extend backend-ref/mut traits with the new state/context associated types;
- adapt the existing normalization where-clauses only; preserve all call arguments and
  return types;
- default conservatively where a temporary compatibility default is unavoidable.

Exit gate: all existing normalization call-shape tests pass, no new normalization method
or failure type exists, and normalization output remains byte-identical to the baseline.

### PR 2 — Nominal HAL roots and private raw storage

Status: partially delivered. Done: `as_scalar_znx_mut` deleted; `from_data_like`/
`map_data_mut` moved to backend-only OEP reborrow functions; the root state parameter
switched to `CoefficientState` (`CoeffNormalized` default) across the workspace with
`impl ArithmeticState` op parameters and `CoeffFitsIn` bounds, byte-identical behavior.
Also done: `VecZnx`'s storage field is private (access via `DataView::data`,
`DataViewMut::data_mut`, consuming `into_data`); `DataViewMut`/`ZnxViewMut` are gated to
`Coeff<Unnormalized, NonCanonical>` (invariant 4) with the sealed unsafe kernel
capability `oep::vec_znx_kernel_words_mut` for backend kernels and harnesses; root
transitions require the `DataOwned` storage marker so mutable borrows cannot be
relabelled (invariants 2–3 under the §4.1 containment patch), with the authoritative
arena view wrappers keeping their transitions; and the compile-fail exit-gate doctests
pass (no safe mutable bytes on normalized roots, no borrow relabel, no manufactured
proof). The inventoried §6.4-B sites ride the ratcheted transitional
`*_borrowed_carry_view` bridges until PR 5's scratch transactions remove them.
Remaining: nominal owned/ref/mut/scratch roles; `Raw`-producing raw constructors
(deferred toward PR 7's reader migration; `from_data_unnormalized` provides
weakest-label ingestion meanwhile); strongest-state zeroed allocators (deferred to
PR 4, which relaxes destination bounds).

Deliverables:

- introduce nominal owned/ref/mut/scratch roles for `VecZnx`;
- make its storage private;
- make ordinary borrows state-invariant;
- route safe mutable bytes exclusively through `Raw`/`Unwritten`;
- delete `as_scalar_znx_mut` and close `from_data_like`/mapping loopholes;
- make raw constructors produce `Raw` and zeroed module allocators produce the strongest
  valid state;
- add the internal unsafe kernel capability.

Exit gate: compile-fail tests demonstrate that a mutable borrow cannot be relabelled and
that a normalized or canonical root cannot expose safe mutable bytes.

### PR 3 — Remove receiver normalization and migrate HAL callers

Deliverables:

- remove inherent normalization from `VecZnx` and HAL scratch views;
- preserve and rebind the established HAL normalization traits to the new state algebra;
- make assign normalization explicitly state-preserving/conservative;
- convert HAL tests and reference helpers to typed destinations or output builders;
- convert scratch accumulators to authoritative `Unwritten`/`Unnormalized` roots;
- remove or quarantine `SetNormalizationState`.

Exit gate: HAL and CPU reference backends pass parity tests; no safe code can repair a
stale owner by normalizing a relabelled borrow.

### PR 4 — Implement canonicality end to end in HAL

Deliverables:

- implement the reference `P_p` projection;
- add only `Module::make_canonical(out, input)` and
  `Module::make_canonical_consume(input)` publicly;
- implement copy-mask and in-place-mask kernels for each backend;
- propagate `C` through HAL arithmetic, copies, transforms, and scratch;
- update unnormalized carry certificates after both canonical projections;
- classify which unchanged normalization paths preserve `C` and make every other path
  produce `NonCanonical`;
- add internal diagnostic predicates for tests only.

Exit gate: both canonical APIs are direct/infallible, no third public canonicalization
entry point exists, and property/parity/performance tests in §§11–12 pass.

### PR 5 — Core aggregates and normalization migration

Deliverables:

- propagate state/context through `GLWE`, matrices, GGLWE/GGSW, LWE components,
  plaintexts, keys, compressed layouts, and all subviews;
- remove inherent normalization from `GLWE` and Core scratch views;
- preserve `GLWENormalize` exactly apart from new bounds;
- migrate packing, encryption, linear-transformation, noise, tensor, trace, and
  keyswitching call sites;
- replace normalized-destination borrow relabelling with scratch transactions and
  out-of-place `glwe_normalize`;
- thread root-owned carry reservations through mutable and parallel child operations;
- annotate precision-sensitive consumers with explicit `Canonical` requirements.

Exit gate: no aggregate projection manufactures a default state, and all Core public
typed paths maintain both axes under panic-injection tests.

### PR 6 — CKKS and BinFHE migration

Deliverables:

- propagate state through `CKKSCiphertext` and FHE integer/ciphertext wrappers;
- remove CKKS receiver normalization and its mutable write wrapper;
- do not add scheme-specific normalization methods; use Core/HAL module operations;
- migrate packing, evaluation, ciphertext splices, key generation, examples, and tests;
- classify effective `k` versus encoded/full precision for every canonical context;
- update decoders that previously rounded from dirty padding.

Exit gate: searches find no production borrow-relabel-normalize pattern, and the usual
scheme test suites pass with canonical padding assertions enabled.

### PR 7 — Serialization, metadata, and backend completion

Deliverables:

- make readers construct `Raw` rather than overwrite arithmetic states;
- make unique semantic writers require `Coeff<Normalized, Canonical>` plus a
  format-readiness bound covering endpoint, top, metadata, and capacity conventions;
- replace public `set_k`/`set_base2k` on typed roots with module conversions or immutable
  construction;
- migrate non-reference backends and device synchronization contracts;
- complete unsafe-kernel audit records;
- remove temporary compatibility defaults and shims.

Exit gate: all ingress routes start at `Raw`/`Unwritten` or a reviewed trusted producer,
and all egress routes declare whether canonical bytes are required.

### PR 8 — Public cutover and cleanup

Deliverables:

- remove obsolete relabel traits, old constructors, receiver methods, and deprecated raw
  APIs;
- turn bypass-search CI warnings into hard failures;
- publish migration notes with before/after examples;
- run the complete test, fuzz, cross-target, documentation, and benchmark matrix;
- update the backend safety contract and crate-level docs.

Exit gate: the Definition of Done in §14 is satisfied and no compatibility shim can
recreate the original stale-owner path.

### 10.1 Suggested sequencing and ownership

PRs 0–2 should have one HAL owner because state algebra, context, and nominal borrowing
must land together. PR 4 can proceed in parallel with the Core call-site classification
once PR 2 stabilizes. Core, CKKS/BinFHE, and non-reference backend migrations can then be
split by crate, but the public cutover waits for every backend.

Expect the change to touch most normalization-bearing files. A coarse current grep finds
roughly 17 HAL, 69 Core, 99 CKKS, 10 BinFHE, and 31 CPU-reference Rust files mentioning
the existing normalization vocabulary; these are planning signals, not precise edit
counts. Refresh the inventory in PR 0 and track each site in a checked migration table.

---

## 11. Verification plan

### 11.1 Compile-fail/UI tests

Add tests proving that safe code cannot:

- relabel `VecZnxMut`, `GLWEMut`, aggregate child views, or ordinary scratch subviews;
- call a carry-producing operation with a normalized destination borrow;
- mutate raw bytes/words through a normalized or canonical root;
- deserialize directly into `Coeff<Normalized, _>` or `Coeff<_, Canonical>`;
- call a normalized-only DFT/preparation primitive with `Unnormalized` or `Raw`;
- call a canonical-only precision consumer with `NonCanonical` or `Raw`;
- call `make_canonical_consume` on `Ref`, `Mut`, or an aggregate child;
- call `make_canonical` on `Raw`, `Unwritten`, compressed, DFT, big, scalar, or prepared
  storage;
- invoke `.normalize(...)` or `.make_canonical(...)` on a layout value;
- treat any normalization or canonicalization call as a `Result` or apply `?`;
- construct state markers or state-bearing views from public raw parts;
- implement sealed state/effect traits outside their defining crate.

Include the original exploit as a UI test. It must fail at the attempted borrowed
relabel, before any DFT call is considered.

### 11.2 Compile-pass/API-shape tests

Pin representative calls for every existing normalization method. The tests must verify:

- method names and `Module` receiver are unchanged;
- `base2k`, offsets, columns, addends, and scratch retain their current order;
- tmp-byte queries retain their current parameters and `usize` return;
- normalization operations retain their `()` return;
- the fused DFT method retains its existing `consume` name and semantics;
- no CKKS/BinFHE duplicate normalization surface is needed;
- assign calls compile for conservative states but do not promote their owner type;
- out-of-place normalization writes a normalized/noncanonical (or weaker) destination
  and rejects a canonical destination at compile time;
- the two canonicalization calls compile for both normalization markers and both input
  canonicality markers;
- the consuming canonical form preserves ownership role and normalization marker.

### 11.3 Canonicality property tests

For all supported `base2k`, representative `k`, shapes, backends, and signed words:

- every output bottom word has its low `p` bits zero;
- no other live word changes;
- inactive capacity follows the documented copy/preserve rule;
- `p == 0` is byte identity;
- applying canonicalization twice is byte identity after the first application;
- out-of-place and consuming forms match bit-for-bit;
- normalized input remains within the exact normalized interval;
- unnormalized input remains accepted for later normalization;
- projected unnormalized bounds/headroom conservatively cover every output word;
- a following carry-producing operation uses the projected certificate rather than the
  destination's pre-call certificate;
- fixed-context normalization and `P_p` commute;
- negative two's-complement edge cases match the unsigned-mask reference;
- aggregate traversal covers every child and uses each child's `represented_k`;
- reference, FFT64, NTT, and device implementations agree.

Generate adversarial bottom words with every padding-bit pattern, including signed
minimum/maximum values and values immediately around digit-bound endpoints.

### 11.4 Normalization tests

Extend the existing normalization suites rather than replacing them:

- exhaust reduced word/radix domains and fuzz full-width words;
- cover same/cross radix, offsets, coefficient-only, assign, big, and fused IDFT paths;
- assert all admitted input contents complete without a data-dependent error;
- verify normalized output bounds for every column/coefficient;
- verify top overflow semantics against the reference implementation;
- verify fixed-context assign normalization preserves canonical input without changing
  its existing byte semantics;
- verify noncanonical and context-changing inputs are never strengthened to canonical;
- verify assign normalization changes bytes but does not change the static source type;
- test typed recovery through the existing out-of-place API.
- fill inactive capacity with adversarial values and prove normalization neither reads it
  into the carry chain nor changes its declared live result.

### 11.5 Scratch, panic, and alias tests

- Fill arena memory with adversarial non-zero patterns before every scratch take.
- Prove `Unwritten` cannot be read or passed to arithmetic consumers.
- exhaust root/child/parallel carry reservations and prove another carry operation cannot
  mutate until the algorithm restores headroom, while normalization remains available.
- Inject a panic after each logical write in reference kernels; every surviving typed
  destination must still satisfy its state.
- Abort/drop partial output builders and verify no strong state escapes.
- Exercise disjoint adjacent regions and reject/forbid overlapping out/input buffers.
- For async backends, delay completion and prove the strong output cannot be borrowed
  before the completion boundary.

### 11.6 Serialization and fuzzing

- Fuzz shapes, lengths, versions, `base2k`, `represented_k`, and coefficient bytes.
- Structural failures remain decoder errors; structurally valid arbitrary words yield
  `Raw`, bind to `Coeff<Unnormalized, NonCanonical>`, normalize through an existing
  operation, and only then enter either canonicalization method, without a
  coefficient-state error.
- Round-trip format-ready normalized/canonical values byte-for-byte for unique formats.
- Confirm legacy dirty-padding decoding is explicitly separated and cannot produce a
  canonical typed value.
- Add source audits/CI greps for forbidden public fields, borrowed relabels, unchecked
  state constructors, receiver normalization, and raw mutation on arithmetic states.

### 11.7 CI lanes

Run the normal workspace format, lint, unit, integration, documentation, and feature
matrices, plus:

- compile-fail tests on stable Rust;
- Miri for owner/view/output-slot transitions where supported;
- reference-versus-optimized backend parity;
- release-mode overflow/fuzz runs;
- device backend tests where available;
- the existing aarch64 cross-check;
- serialization compatibility fixtures;
- criterion or equivalent performance baselines from §12.

---

## 12. Performance gates

The type redesign should be zero-cost for ordinary borrows and state-preserving
operations. Measure rather than assume this.

### 12.1 Normalization

The preserved APIs keep their current caller-provided scratch and tmp-byte queries.
Benchmark before/after for each normalization family and backend. Generic state checks
must compile away, and existing normalization kernels must remain byte-identical.

Migration from a borrowed in-place accumulator to a separate scratch accumulator can
increase peak scratch and memory traffic. Record, per hot call site:

- old and new scratch bytes;
- number of normalization passes;
- allocation count;
- bytes copied;
- host/device transfers;
- latency and throughput.

No hidden allocation or transfer is permitted inside normalization. A regression above
the project's normal noise threshold (suggested starting gate: 2% on stable hot
benchmarks) requires an explicit design review rather than weakening the typestate.

### 12.2 Canonicalization

Benchmark the one streaming copy-mask pass and the in-place consuming pass. Required
properties:

- zero scratch and zero allocation;
- no host/device transfer;
- `p == 0` consuming form performs no data work;
- `p == 0` out-of-place form is no slower than the corresponding exact copy beyond
  benchmark noise;
- no preliminary validation scan;
- trusted full writers may produce canonical bytes directly when that is already part of
  their documented output contract; normalization does not gain hidden masking work.

### 12.3 Compile-time and binary-size gates

The product state algebra can increase monomorphization. Track clean/check build time,
incremental build time, and binary size for representative binaries. Prefer sealed
effect traits and shared backend implementations over duplicating kernels for all four
`N/C` combinations.

---

## 13. Risks and mitigations

| Risk | Consequence | Mitigation |
|---|---|---|
| Assign normalization is mistaken for promotion | An unnormalized owner is passed to a normalized-only consumer | Encode the conservative rule in docs/UI tests; use out-of-place normalization for typed recovery |
| Canonical output is filled dirty then masked | Panic exposes a falsely canonical destination | Store masked bottom words directly; panic-inject after each store |
| Runtime `k`/`base2k` changes behind the marker | Both proofs refer to the wrong representation | Private immutable context; module conversion instead of setters |
| Aggregate child defaults its state | Strong child view is forged | Parent-associated state/context on every projection; no defaults in constructors |
| Safe backend view leaks raw mutation | Original bypass survives under another trait | Crate-private unsafe capability and public API surface audit |
| Cross-radix normalization is assumed to preserve canonicality | Dirty target padding enters precision-sensitive code | Produce `NonCanonical`, then call `make_canonical*`; cross-radix property tests |
| Signed mask semantics differ by backend | Negative values or bounds diverge | Unsigned bit projection reference and edge-case parity tests |
| Dirty scratch is read before full initialization | Nondeterminism or false state | `Unwritten` roots, affine builders, dirty-arena tests |
| Inactive capacity enters a carry chain | Dirty tail changes a live canonical/normalized value | Clamp every arithmetic view to `live_limbs`; adversarial-tail parity tests |
| Child/parallel carry update leaves a stale root bound | Later arithmetic overflows despite a valid-looking certificate | Root reservations, conservative joins, unwind tests |
| Canonical projection leaves stale headroom | Masking a negative word widens its magnitude before later arithmetic | Recompute/project bounds in both canonical forms before exposing output |
| Legacy decoder rounded padding | Observable behavior changes | Classify as raw legacy behavior or migrate explicitly; compatibility fixtures |
| Scratch transaction or explicit canonicalization increases hot-path cost | Performance regression | Inventory early, reuse arenas, let trusted producers emit canonical output, enforce benchmark gates |
| State generics explode compile time | Slow builds and large binaries | Product state wrapper, sealed relations, shared monomorphic kernel core |
| Async backend returns before proof is true | Typed value races incomplete device work | Completion token/synchronization in safe delegate contract |
| Backend buffer types retain writable aliases | Nominal layout roles do not represent real authority | Sealed unsafe storage-provider contract; deep-copy and alias tests |
| Structural errors are confused with coefficient-state errors | Pressure to reintroduce fallible transforms | Keep decode/parameter errors separate; transforms remain total |
| Transitional unsafe relabel survives cutover | Permanent escape hatch | Deprecation deadline, deny-list grep, final OEP audit |

---

## 14. Definition of Done

### Type system and ownership

- [ ] Coefficient layouts use sealed `Unwritten`, `Raw`, and `Coeff<N, C>` states.
- [ ] `Normalized` and `Canonical` are tied to immutable per-value representation
      context.
- [ ] Owned, borrowed, mutable, and scratch roots are nominally distinct.
- [ ] Ordinary borrows and aggregate child views cannot relabel either axis.
- [ ] Every coefficient-domain aggregate propagates the complete state and child context.
- [ ] Fresh reused scratch is `Unwritten`, and arbitrary bytes are `Raw`.
- [ ] Private carry certificates are root-owned; child/parallel writes reserve and join
      headroom conservatively, and normalization never depends on that headroom.
- [ ] Arithmetic and normalization views are clamped to `live_limbs`; inactive capacity
      cannot influence a live carry chain.

### Normalization API

- [ ] The existing HAL, big, fused IDFT, and Core normalization method names, receiver,
      argument order, scratch arguments, normalization-operation `()` returns, and
      tmp-byte-query `usize` returns are retained.
- [ ] Only state/generic bounds change.
- [ ] No generic replacement normalization API is added.
- [ ] No inherent normalization implementation remains on a layout or view.
- [ ] Normalization accepts every initialized coefficient input in its declared domain
      and has no coefficient-content failure type or `Result`.
- [ ] Assign normalization is documented and tested as non-promoting.
- [ ] Out-of-place/coefficient/big/fused normalization produces `NonCanonical`; it does
      not mask padding or accept a canonical destination.
- [ ] Typed `Unnormalized -> Normalized` recovery uses the existing out-of-place API and
      a suitable destination.

### Canonicality API

- [ ] `module.make_canonical(&mut out, &input)` exists and returns `()`.
- [ ] `module.make_canonical_consume(input)` exists and returns its output directly.
- [ ] These are the only public canonicalization operations.
- [ ] Both implement the exact `P_p` projection, preserve `N`, and produce `Canonical`.
- [ ] Neither takes scratch, allocates, transfers to host, scans first, rounds, or returns
      an error.
- [ ] The consuming form is implemented only for authoritative owner/scratch roots.
- [ ] Both forms update unnormalized bound/headroom evidence for the signed bit
      projection before the result can feed another carry operation.
- [ ] `p == 0`, signed edges, idempotence, and normalization commutation are tested.

### Validation and errors

- [ ] No public validation/certification operation promotes coefficient typestate.
- [ ] No normalization/canonicality/typestate-validation failure enum or checked wrapper
      remains.
- [ ] Internal diagnostic scans are observational only and cannot affect typestate.
- [ ] Structural decode/I/O/parameter errors remain clearly separated from coefficient
      transforms.

### Raw and backend boundary

- [ ] State-bearing storage fields are private.
- [ ] Safe mutable raw access is limited to `Raw`/`Unwritten` authoritative storage.
- [ ] `as_scalar_znx_mut` is removed.
- [ ] In-place readers do not overwrite arithmetic states.
- [ ] Unsafe kernel contracts name input/output states, context, coverage, aliasing,
      unwind, scratch, and synchronization.
- [ ] The backend storage-provider contract proves authoritative ownership, deep-copy
      behavior, and `BufRef`/`BufMut` aliasing for every lifetime.
- [ ] The generic safe relabel OEP is gone.

### Integration and quality

- [ ] All production borrow-relabel sites are migrated.
- [ ] DFT and every bound-sensitive consumer require the correct normalization state.
- [ ] Every precision-sensitive consumer handles canonicality explicitly; every unique
      semantic byte format additionally requires normalized and format-specific readiness.
- [ ] Compile-fail tests include the original exploit and every raw/aggregate variant.
- [ ] Reference/optimized/device parity, dirty-scratch, panic, fuzz, serialization, and
      cross-target suites pass.
- [ ] Performance, scratch, compile-time, and binary-size gates pass or have an explicit
      approved exception.
- [ ] Public migration notes and the backend safety contract match the shipped API.

---

## 15. Rejected alternatives

### Gate relabelling with `DataOwned`

This is useful as a containment patch but not a durable ownership model. Generic backend
buffers and scratch regions do not communicate authoritative ownership through a
portable marker bound, and aggregate subviews can still retain stale parents.

### Keep raw mutation as a documented safe trust boundary

That changes the guarantee from compile-time enforcement to convention. Backends need a
trust boundary; application code does not need the same safe capability.

### Automatically normalize on mutable-borrow drop

Drop has no scratch parameter, cannot report or control backend completion, and is hard
to make panic-safe. It also hides a potentially expensive pass in lifetime mechanics.

### Add a new consuming normalization operation

A by-value operation would enable zero-copy typestate promotion, but it would replace or
duplicate the established normalization surface. The requirement is to retain the
current module APIs; out-of-place normalization is the explicit typed recovery path.

### Treat assign normalization as a proof promotion

Rust cannot change the generic parameter of an object through `&mut T`. Relabelling a
temporary mutable view recreates the original bug. The assign operation therefore keeps
the static state conservative.

### Make canonicality a strongest linear state

Canonicality and normalization are independent. All four combinations are meaningful,
so one linear `Raw -> Unnormalized -> Normalized -> Canonical` ladder is incorrect.

### Make normalization always imply canonicality

Normalization controls signed digit bounds; canonicality controls precision padding.
They commute for a fixed context but neither logically implies the other. Normalization
may preserve an existing canonical proof, but only the two `make_canonical` operations
and trusted canonical producers establish one.

### Validate and relabel without changing bytes

A scan can discover that a particular value already satisfies an invariant, but an
infallible scan cannot promote arbitrary dirty input. Error-bearing proof promotion is
unnecessary because normalization and bit projection enforce the properties directly.

### Keep only masks at consumers

Repeated masks may defend individual operations but do not let the type state whether
bottom padding is zero. Canonicality is useful even though unique semantic serialization
also requires normalization and format-specific endpoint/top/capacity rules.

---

## 16. Immediate implementation checklist

Before editing public Rust types, complete these concrete tasks in order:

1. Add API-shape tests around every method listed in §5.2.
2. Freeze the exact normalized interval and confirm total behavior for supported word
   patterns across reference and optimized normalizers.
3. Make a checked inventory of production borrow relabels and assign each one a scratch,
   output-builder, or direct-write migration.
4. Define `represented_k` for every coefficient root and all child views.
5. Add a standalone reference implementation and exhaustive reduced-domain tests for
   `P_p`.
6. Land the sealed product-state algebra and immutable context without changing bytes.
7. Land nominal HAL roles and close public raw mutation.
8. Remove receiver normalization while migrating call sites to the existing module
   family.
9. Add the two canonicalization operations and backend kernels.
10. Propagate the model upward through Core, CKKS, BinFHE, serialization, and every
    backend.
11. Remove compatibility shims only after source audits and compile-fail tests prove the
    stale-owner path is gone.

The plan is ready for PR 0 once the normalized interval and the per-layout
`represented_k` mapping are recorded. Neither item changes the API decisions above.
