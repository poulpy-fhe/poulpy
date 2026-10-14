# Implementing BLAS operations

Public matrix calls follow `api → delegates → oep`. `Module<BE>` implements
`BlasMatInterOps`, `BlasMatIntraOps` and `BlasMatMulOps` for every backend that
implements the matching `*Impl` trait: each call runs its validator, then
forwards to the backend's method. The BLAS crate imports no backend crate:
each backend crate wires its own types.

## Reference implementations

A **reference** implementation defines a matrix operation by composing CKKS
and core operations on the module. The reference traits are implemented for
every `Module<BE>` and are public, so an override can forward to them. They
call `poulpy-ckks` and `poulpy-core` through their public APIs, so a backend's
CKKS and core overrides remain effective inside them.

| OEP family | Operations | Reference | Macro |
|---|---|---|---|
| `BlasMatInterImpl` | `blas_rotate`, `blas_sum` | `BlasMatInterReference`, [`reference/inter.rs`](../src/reference/inter.rs) | `impl_blas_inter_reference!` |
| `BlasMatIntraImpl` | the map hook `blas_map_blocks_impl`; `blas_transpose`, `blas_rotate_rows` and `blas_rotate_cols` are provided over it | `BlasMatIntraReference` (the hook), [`reference/intra.rs`](../src/reference/intra.rs) | `impl_blas_intra_reference!` |
| `BlasMatMulImpl` | `blas_matmul_prepare_left`, `blas_matmul_prepare_right`, `blas_matmul_prepared`, `blas_matmul_pt`, `blas_matmul_pt_left`; `blas_matmul` is provided over the preparations and the prepared product | `BlasMatMulReference`, [`reference/matmul.rs`](../src/reference/matmul.rs) | `impl_blas_matmul_reference!` |
| `BlasJklsImpl` | the JKLS kernel `blas_jkls_block_impl`, one output block of the prepared product | `BlasMatMulReference::blas_jkls_block_reference`, [`reference/matmul.rs`](../src/reference/matmul.rs) | `impl_blas_jkls_reference!` |

Each operation's `_tmp_bytes` query belongs to the same family.
`impl_blas_reference_full!` expands to the four family macros. `BlasImpl`
aggregates the four families as a test bound; it has a blanket implementation
and is never implemented directly.

The provided methods, in the crate-private [`oep/derived.rs`](../src/oep/derived.rs),
call the backend's selected methods: the transpose and both shifts call its
map hook, and `blas_matmul` its preparations and prepared product. The
reference's cold preparations likewise call the backend's map hook, and the
reference prepared product calls it for the terms of a hot operand, then its
JKLS kernel. So `impl_blas_matmul_reference!` needs the intra and JKLS
families, and an override of the hook or of the kernel reaches every operation
built on it.

## Grids and keys

`res` is a `BlasGridMut<BE>` and every ciphertext input a `BlasGrid<BE>`,
which expose the blocks of a grid as a row-major slice. The prepared product
reads prepared operands instead ([Products](#products)), and the products by
public matrices also read a `BlasPtOperand`. A grid's blocks number exactly
`rows x cols`, at least one, and every operation rejects a grid that breaks
this. `BlasCiphertext` over owned storage implements both traits on every
backend. Borrowed views are not grids: a `BufRef` view is `Sync`, but its
blocks are not `GLWEToBackendRef`.

Rotation keys come as `atk` and the tensor key as `tsk`, two separate stores:
the rotations, the sums, the maps, the preparations and the products by public
matrices take `atk` alone, and the products both. The prepared product looks
up rotation keys only for a hot operand, whose terms it evaluates: of two cold
operands it reads `tsk` alone. The traits' `Send` and `Sync` bounds leave an
implementation its own choice of concurrency ([Parallelism](#parallelism)).

## Batches

A grid records its batch, `BlasGridInfos::mat_slots`: the number of matrices
its blocks hold. `BlasCiphertext::from_blocks` records the batch the blocks'
label proves, and a grid without a record (`mat_slots()` is `None`) is read at
that batch. Every operation rejects a grid whose label does not allow its
batch: with `S` slots, degree `n` and label `s`, the batch is a power of two
between `(S >> s) / d^2` and `S / d^2`, and `s` is at most `log2 n`. The API
stamps the record of `res` once the implementation returns `Ok`: products
record the larger batch of their operands, every other operation its input's.
An implementation writes blocks only, and reads an input's batch with
`poulpy_blas::layouts::grid_batch`, as the references do: its record, or for a
grid without one the batch its label proves. A preparation records that batch
on its operand.

Encryption labels a block from its plaintext's degree, so a batch of `count`
matrices comes from compact plaintexts of `count d^2` slots. Compact degrees
are raised to the backend's `MIN_DEGREE` (8 by default, 16 on AVX-512 FFT64),
and a raised degree records a larger batch. A compact batch therefore records
`count` on every backend only when its degree, `2 count d^2` on the standard
ring and `count d^2` on the conjugate-invariant one, reaches every target's
`MIN_DEGREE`; otherwise encrypt it dense.

## Keys

Callers generate rotation keys from the per-operation sets of `BlasPlan`,
which take the ring's cyclotomic `order`, and from `sum_galois_elements`. Every
operation looks its keys up at fixed widths, for the maps' scale `p` and a
public operand's scale `p_B`:

| Operation | Rotation keys | Widths |
|---|---|---|
| `blas_rotate` by `r` | `rotate_galois_elements(order, r)` | `a.k` |
| `blas_sum` | `sum_galois_elements(module, a)` | `min(res.k, a.k)` |
| `blas_transpose` | `transpose_galois_elements(order)` | baby steps at `a.k`, giant steps at `min(res.k, a.k - p)` |
| `blas_rotate_rows`, `blas_rotate_cols` by `k` | `rotate_rows_galois_elements(order, k)`, `rotate_cols_galois_elements(order, k)` | as the transpose; none for a shift by a multiple of `d`, a copy |
| `blas_matmul_prepare_left`, `blas_matmul_prepare_right`, `blas_matmul`, and `blas_matmul_prepared` of a hot operand | `matmul_galois_elements(order)` | sigma's (tau's) baby steps at `a.k` and giant steps at `a.k - p`; the shifts' giant steps at `a.k - 2p` |
| `blas_matmul_pt`, `blas_matmul_pt_left` | `matmul_pt_galois_elements(order)`, `matmul_pt_left_galois_elements(order)` | baby steps at `a.k`, giant steps at `min(res.k, a.k - p_B)` |

The sets list the rotations of the reference's maps as the plan schedules
them: shift maps `Direct`, the others at the cost-optimal BSGS giant step.
`galois_elements(order)` covers every map of a context, so every operation's
set but `blas_rotate`'s and `blas_sum`'s. The preparations check the keys of
their terms under either hint, and the prepared product those of a hot
operand's under its own context. The tensor key is needed if and only if a
pipeline calls `blas_matmul` or `blas_matmul_prepared`, at `max(w_a, w_b)`, the
wider of the operands' term widths `a.k - 2p` and `b.k - 2p`. The
[pipeline example](../../poulpy-cpu-portable/examples/blas_pipeline.rs) plans
its keys this way, with one rotation-key layout covering its widest input and a
tensor key covering the terms, and prints the width each operation consumes.
Every key has the rank of the blocks it switches.

These sets and widths bind implementations: `GetAutomorphismKey` and
`GetTensorKey` receive the width, so a store may answer per width. An
implementation looks up only keys, at widths, that the reference looks up, so
a store holding its operation's set alone suffices. Parity records every
lookup of both backends to check this.

## Wiring a backend crate

A backend crate of this workspace wires BLAS behind its own `enable-blas`
feature:

```toml
[features]
enable-blas = ["dep:poulpy-blas", "enable-ckks"]

[dependencies]
poulpy-blas = {workspace = true, optional = true}
```

`enable-blas` implies the crate's own `enable-ckks`, which wires its CKKS
families; BLAS needs nothing from another backend crate. A `blas_impl.rs`
module (`blas.rs` in the oracle), declared under `enable-blas`, selects the
reference for each type, with the same `#[cfg]` gates as `ckks_impl.rs`:

```rust
::poulpy_blas::impl_blas_reference_full!(crate::FFT64Avx);
#[cfg(feature = "enable-rayon")]
::poulpy_blas::impl_blas_reference_full!(crate::FFT64AvxRayon);
```

To replace a family, replace the type's `impl_blas_reference_full!` line with
the macros of the three families it keeps, and write the replaced family's
`*Impl` implementation directly, forwarding the methods it does not change to
their reference routines. The expansion of each family macro, in
[`oep`](../src/oep/), spells out every signature and its forwarding call. A
family macro and a handwritten implementation of the same trait are mutually
exclusive. The family traits are `unsafe`: their `# Safety` sections state what
an implementation must reproduce, which
[What an override must preserve](#what-an-override-must-preserve) gathers.
Three replacements cover most needs:

- **The map hook**, for a backend that evaluates maps its own way. It covers
  the transpose, both shifts, the cold preparations and the prepared product's
  hot sides, so `blas_matmul` too:

  ```rust
  ::poulpy_blas::impl_blas_inter_reference!(MyBackend);
  ::poulpy_blas::impl_blas_matmul_reference!(MyBackend);
  ::poulpy_blas::impl_blas_jkls_reference!(MyBackend);
  unsafe impl poulpy_blas::oep::BlasMatIntraImpl for MyBackend { /* the hook and its query */ }
  ```

- **The JKLS kernel**, a fused kernel behind the backend's own `enable-blas`.
  Both `blas_matmul` and `blas_matmul_prepared` reach it:

  ```rust
  ::poulpy_blas::impl_blas_inter_reference!(MyBackend);
  ::poulpy_blas::impl_blas_intra_reference!(MyBackend);
  ::poulpy_blas::impl_blas_matmul_reference!(MyBackend);
  unsafe impl poulpy_blas::oep::BlasJklsImpl for MyBackend { /* the kernel and its query */ }
  ```

- **The products**, for prepared operands in the backend's own layout, hot or
  cold. The inter, intra and JKLS families keep their macros, and
  `BlasMatMulImpl` is written by hand: it sets both operand types, implements
  both preparations and the prepared product, and forwards `blas_matmul_pt`
  and `blas_matmul_pt_left` to `BlasMatMulReference`, while `blas_matmul` stays
  provided. The JKLS kernel reads poulpy-ckks's prepared terms, so a backend
  that holds its terms in another representation replaces this family rather
  than the kernel.

A backend may also override a provided method, for example the shifts, to
chunk their heap grid, or `blas_matmul`, to stream its operands; it then
reproduces the circuit the provided body fixes. A Rayon backend writes
nothing: its kernels already run in parallel inside each operation
([Parallelism](#parallelism)).

## What an override must preserve

Parity is bit-exact, so an override computes the reference's circuit: every
output block equals the reference's in its digits and in its CKKS metadata,
`k`, radix, rank, `log_delta`, `log_sparsity`, slot kind, canonical flag and
noise tag, as the `# Metadata` sections of `BlasMatInterOps`,
`BlasMatIntraOps` and `BlasMatMulOps` state them. The `# Safety` section of
each family states what an implementation reproduces; the sections below
gather them, family by family, and anything they do not list as free changes
bits. Every family also follows [Keys](#keys), [Scratch](#scratch) and
[Parallelism](#parallelism).

### Validation and errors

Each API call first runs its validator, `validate_<op>` in
`poulpy_blas::reference`, from metadata alone, then the backend's method, then
stamps the batch record of `res`; a preparation returns its operand instead.
The validators check this crate's rules and fail in this order:

1. `CKKSCompositionError::InvalidOperand` for an operand they reject: a grid
   at another `log_d`, whose blocks do not tile `rows x cols` or do not share
   one layout, whose `log_delta` exceeds its `k`, or whose label does not allow
   its batch record; a shape or inner dimension that does not match; blocks of
   another degree than the module's, or inputs of another radix or rank than
   `res`; a context or public operand encoded for another slot count or radix;
   public matrices that do not divide the batch; a prepared operand without a
   batch record, or a hot one at another `log_d` than the product's context
   or under a context encoded for another slot count or radix; and a product
   at a radix above 62, whose digits cannot hold two lazy terms of the JKLS
   kernel's two-pass circuit. A hot operand's scale is not checked: the
   product spends that of its context's maps.
2. Key by key, the rotation keys first, each at the width the operation
   looks it up at ([Keys](#keys)): `MissingAutomorphismKey`, or
   `MissingRelinearizationKey` for the tensor key, when the store lacks it,
   and `InvalidOperand` when it is of another rank than the blocks it
   switches, on which poulpy-core's key switch would panic. A preparation's
   input has no `res` to match, so this is where its rank is checked.

They check no budget. poulpy-ckks raises the budget errors,
`MultiplicationPrecisionUnderflow` and `InsufficientHomomorphicCapacity`,
inside the implementation, from the rotations, the copies, the maps, the
products by public matrices and the JKLS kernel's dot product, before any write
to `res`: every block of a grid shares one layout, so the first block's call
fails before any block is written. A preparation writes only the operand it
returns. A hot operand's terms are evaluated by the product, so its budget
error comes from the product, the left side's before the right side's, and
before the kernel's; a hot preparation checks no budget. So `blas_matmul`,
whose operands are hot, meets its factors' budget errors before the kernel's.

A rejected call therefore leaves `res`, its batch record and the inputs
untouched, whichever backend runs it. An implementation runs only on inputs
its validator accepted and needs no checks of its own. It returns `Ok` on them,
or the reference's budget error before any write to `res`; any other `Err` is a
backend fault, a validator gap or a device failure, and `res` is then
unspecified. It never stamps a batch record: the API does, after `Ok`, with
the batch the validator computed. Since keys are checked first, an input that
poulpy-ckks rejects for its budget may ask for a key that poulpy-ckks would not
look up for it: a store holding exactly the keys in use then reports
`MissingAutomorphismKey` instead of the budget error.

### Rotations and sums

`BlasMatInterImpl` has no hook: an override reproduces its two operations
block by block.

- `blas_rotate` by `r` is poulpy-ckks's `ckks_rotate_into` of each block of `a`
  by `r d^2` slots into the matching block of `res`: a `ckks_copy` with a
  cleared noise tag when `r d^2` is a multiple of the slot count.
- `blas_sum` copies each block of `a` into `res` with `ckks_copy` and clears
  its noise tag, then, for each `i < log2(mat_slots)` in turn, rotates the
  block by `2^i d^2` slots into a block of `res`'s layout and adds that with
  `ckks_add_assign`.

**Free:** the order of the blocks.

### The map hook

`BlasMatIntraImpl` requires the map hook `blas_map_blocks_impl` and its query,
and provides the transpose and both shifts over them, poulpy-ckks's copy and
its addition.

**The hook** is called with at least one map `lts`, sources `src` that share
one layout, and `src.len() * lts.len()` blocks of `dst` that share another, on
inputs the calling operation's validator accepted. It writes block
`t * lts.len() + m` of `dst` with map `lts[m]` evaluated from `src[t]` by
poulpy-ckks's `ckks_eval_linear_transformation_into`, then stamps it with the
label (`log_sparsity`) of `src[t]`. It looks up each map's baby-step keys at
the width of `src` and its giant-step keys at `min(dst.k, src.k - p)`. Within
the scratch its query reports for `dst`, `src`, the source count, the maps and
every key it looks up, it returns `Ok`, or poulpy-ckks's budget error before
any write to `dst`; with less scratch it panics before any write. **Free:** the
order of the blocks; one baby-step cache per source across its maps, as the
reference keeps; batching key switches; normalizing a source read through a
DFT.

**The provided methods** fix the rest of the circuit, which an override of one
of them reproduces:

- `blas_transpose` is one hook call, `res_ji = transpose(a_ij)`, its sources
  in the order of `res`'s blocks.
- `blas_rotate_rows` and `blas_rotate_cols` by `k` write `a` shifted by
  `k mod m d = q d + k'` along the shifted block axis, of extent `m`:
  - for `k' = 0`, each block of `res` is a `ckks_copy` of the block `q` steps
    along, with a cleared noise tag: no hook call and no key;
  - for `m = 1`, one hook call evaluates psi^k' (phi^k') into `res`;
  - otherwise, one hook call evaluates both half maps of every block of `a`
    into a heap grid of two blocks per block of `a`, at `res`'s block layout,
    and each block of `res` is the `ckks_add_into` of the first half map of
    the block `q` steps along and the second of the block `q + 1` steps along.

  The shift maps are `Direct`, as the plan records. The heap grid is the
  provided shifts' memory ceiling; an override may chunk it.

The reference cold preparations call the hook too, and the reference prepared
product for the terms of a hot operand ([Products](#products)), so a backend
that overrides the hook alone evaluates every map of this crate but those of
the products by public matrices.

### Products

**Prepared operands.** `MatMulLeft` and `MatMulRight`, which the API names
`BlasMatMulLeft<BE>` and `BlasMatMulRight<BE>`, may hold any representation,
such as device memory, contiguous or DFT-domain terms, and only the
preparations make one, under a `PrepareHint`:

- **hot**, `OneShot`, prepared for one product: the reference relabels its
  input, a verbatim copy of every block, through no map, key or transform. The
  product evaluates its terms under the product's own context, whatever the
  context the operand was prepared under, at whatever scale: the maps spend
  their scale from the operand's budget, and the product stamps by the ct x ct
  rule, at the smaller scale;
- **cold**, `Reuse`, prepared for many products: every term, each converted
  once to the transform domain by poulpy-ckks's `ckks_prepare_left` or
  `ckks_prepare_right`, under the preparation's context.

Through `BlasMatMulOperandInfos`, an operand prepared from `a` reports its
hint, `a`'s block layout as its source, `a`'s dimensions, `Some` of `a`'s
batch, `grid_batch::<BE::Ring>(a)`, and as its block layout `a`'s when hot and
the layout `poulpy_blas::reference::blas_matmul_term_layout(a, ctx)` gives
when cold; `blas_matmul_operand_layout(a, ctx, hint)` gives these infos before
preparing. The prepared product's validator, the batch the API records and
the provided `blas_matmul` query read nothing else, and parity compares these
infos, never the storage. The prepared product of two operands, of any hints,
equals the reference's on the reference's operands of the same sources and
hints under the same contexts. The preparations' validators check the
rotation keys of the terms under either hint; a cold preparation looks them
up, and the prepared product looks up those of each hot side's terms, and the
tensor key. The reference's operand, `BlasMatMulOperand<BE, P>`, holds either
the copied blocks or the terms, `P` being `CKKSPreparedLeft<BE>` on the left
and `CKKSPreparedRight<BE>` on the right.

**Fixed:**

- The terms. Each source block `a_il` has `d` terms, all at the width
  `w = a.k - 2p`: term `0` is sigma of `a_il` evaluated into a block of `a`'s
  layout, which leaves it at `a.k - p`, then narrowed exactly to `w` by
  `ckks_copy`, which charges the shift to the budget; term `k` is phi^k of that
  sigma result into a block of width `w`, through the `Direct` shift maps the
  plan records. On the right, tau and psi^k. Each term is converted once, on
  its `ceil(w / base2k)` limbs. The terms depend on neither `res` nor the other
  factor: the kernel's offset absorbs a narrower `res`. The reference makes
  them block row after block row, with two hook calls per row, sigma over its
  blocks, then every shift of every sigma result, from one baby-step cache per
  result.
- The bits, whatever the hints: the prepared product equals the reference's
  on the reference's operands of the same sources and hints under the same
  contexts, so an operand prepared under the product's context gives the same
  products under either hint, and `blas_matmul` is the prepared product of two
  hot operands.
- One JKLS kernel per block `res_ij`, over its `K d` pairs of terms
  `(a_il[k], b_lj[k])`, `l`-major: ascending `l`, then ascending `k` within
  each `l`, the order the one-pass kernel sums them in, part of the bits on FFT
  ([The JKLS kernel](#the-jkls-kernel)).
- The errors and their order: the validator's, then poulpy-ckks's budget
  errors before any write to `res`, the left terms', the right terms', then the
  first block's ct x ct rule. A hot side's come from the product, so a hot
  preparation of an input short of its budget returns `Ok`.
- The products by public matrices share their giant steps across the inner
  index: block `res_ij` is poulpy-ckks's
  `ckks_eval_linear_transformation_sum_into` of its `K` terms in ascending `l`,
  `(a_il, b_lj)` for `blas_matmul_pt` and `(a_lj, b_il)` for
  `blas_matmul_pt_left`, `res = B A` from a `BlasPtOperand<BE, Left>` that
  `BlasPtOperandUnprepared::encode_left` encodes and `prepare` prepares, then
  stamped with `a`'s label.
  Each giant step adds the products of every term in the transform domain, in
  that order, before its one rotation; an FFT backend rounds that sum, so the
  order is part of the bits, and the radix must hold `K` times a giant step's
  products (`Module::max_base2k`). The reference caches the baby steps of each
  block of `a` once for its row of `res` (its column, on the left).

**Free:** the representation of an operand under either hint: a hot operand
may hold its source and evaluate its terms in the product, as the reference
does, or prepare them eagerly, as long as it reports `OneShot` and its source,
returns its budget error from the product, and gives the terms of the
product's context, recomputing them when that context is not the one they came
from; the order of the blocks; which side the product holds and which it
streams; materializing or streaming the terms; caching them or their
convolution forms; sharing baby-step caches; batching key switches.

`blas_matmul` is provided as both hot preparations, then the prepared product,
so it reaches the backend's methods. For an `m x K` by `K x n` product, the
reference prepared product holds the terms of its hot sides for the call,
`K (m + n) d` of them in the backend's transform-domain form for
`blas_matmul`; streaming them is free. Hoisting the shifts is the one reserved
variant: it changes the bits and the key widths, so it would arrive as a
`BlasPlan` variant, not as an override.

### The JKLS kernel

`BlasJklsImpl::blas_jkls_block_impl` computes one block of the prepared
product, `res = sum_t lhs[t] rhs[t]`, over terms prepared once by
poulpy-ckks's `ckks_prepare_left` and `ckks_prepare_right`, which give their
convolution operands through `cnv()` and report the terms they were prepared
from: the kernel transforms no operand. The reference is poulpy-ckks's
`ckks_dot_product_prepared`, which is `ckks_dot_product_ct` of those terms, bit
for bit, for pairs that share one layout per side. It is called with
`lhs.len() == rhs.len() >= 1` pairs, every `lhs` prepared from terms of one
layout and every `rhs` from terms of another, at a radix of 62 at most, after
the validator found the tensor key at the tensor width `max(lhs.k, rhs.k)`, of
their rank.

**Fixed:**

- The circuit. It is one-pass where poulpy-ckks's `ckks_mul_into` takes its
  one-pass product: rank 1, one `base2k` across `res`, the terms and the tensor
  key, and an offset of at least `base2k`. It is two-pass otherwise, as at
  rank 2.
- One-pass: poulpy-core's `glwe_mul_prepared_relinearize_sum` over the pairs,
  in slice order, at the offset poulpy-ckks's ct x ct rule gives for `res` and
  the width it stamps. Every pair's products are added in the transform domain,
  in that order, and only the key-switched column of the sum comes back to
  digits before the one final normalization. An FFT backend adds in floating
  point, so the order is part of the bits. The radix must hold `lhs.len()`
  times the products of one pair (`Module::max_base2k`), as `blas_matmul`
  states.
- Two-pass: each term is one poulpy-core `glwe_tensor_apply_prepared` of
  `(lhs[t], rhs[t])` at that offset, rounded on its own at the tensor width.
  Summing unrounded products, as `cnv_apply_dft_sum` would, changes bits. The
  terms are summed exactly, and the sum is normalized at that width before the
  relinearization reads it.
- Either way, `res` is stamped with the rule's budget and scale, the smallest
  label and the joined slot kind, and relinearized once, with the tensor key at
  the tensor width. `res` comes back canonical, without a noise tag.

**Free:** streaming the pairs; caching other forms of the operands. On the
two-pass circuit, also the order and grouping of the terms, computed in turn or
concurrently, and normalizing partial sums, or leaving them unnormalized until
the DFT that reads the sum. On the one-pass circuit, the order and grouping of
the pairs are free only on an NTT backend, whose transform-domain additions are
exact.

**Normalization.** On the two-pass circuit, partial sums may stay unnormalized
until the next DFT. A normalization is exact carry propagation that preserves
the value, so where it happens never changes the bits: the only obligation is
the reference's bits. The reference normalizes once, right before the
relinearization. Like any lazy addition, the sum must fit a 64-bit digit, which
the caller's parameters ensure: signed terms grow slowly, like an Irwin–Hall
sum, and radixes in practice are 54 or less. The one-pass circuit adds no
digits: its products accumulate in the transform domain, and the radix rule
above bounds them.

Within the scratch its query reports for `res`, `lhs[0]`, `rhs[0]`, the pair
count and the key, it returns `Ok`, or poulpy-ckks's budget error before any
write to `res`; with less scratch it panics before any write. The reference
prepared product calls it once per block of `res`, in row-major order, on one
scratch.

## Scratch

Every method runs within the scratch its own `_tmp_bytes_impl` reports for the
layouts and keys it receives, which may differ from the reference's: parity
gives each backend exactly the bytes its own query reports, poisoned and
guarded, so an underestimate fails there. The references assert that bound on
entry and panic before any write when it falls short, for a call that looks up
a key.

The queries read layouts (`BlasGridInfos`, `CKKSCtBounds`, `GGLWEInfos`), never
ciphertext data: the prepared product's query receives the prepared operands,
or, before preparing, the layouts `blas_matmul_operand_layout` gives, and the
context, under which it sizes a hot side's terms, so it reads nothing but their
infos. The provided and reference queries call the selected implementations'
queries: the provided transpose the hook's; the provided shifts the hook's, the
copy's and the addition's; a reference hot preparation the copy's; a reference
cold preparation the hook's through sigma (tau) and through the shifts over a
block row, the copy's, and poulpy-ckks's `ckks_prepare_tmp_bytes` for a term;
the reference prepared product the kernel's, and for each hot side the cold
preparation's of its source; and the provided `blas_matmul` the largest of
both hot preparations' queries and the prepared product's on the layouts they
give. An override of the hook or of the kernel that needs other scratch
reports it in its own query, and every operation over it follows.

The references run their steps one after another on one scratch, so their
queries report the largest single step, on every executor
([Parallelism](#parallelism)).
Temporaries that outlive a hook call live on the heap, not in scratch: the
copies of a hot preparation; the sigma results and terms of a cold preparation,
or of a hot side in the prepared product, each coefficient term dropped once
converted; and the heap grid of a shift across blocks. A query may panic on
layouts its validator rejects, as its `# Panics` section states.

## Parallelism

Concurrency is intra-ciphertext: it lives inside the backend's kernels, within
one ciphertext operation. A Rayon backend parallelizes there, as poulpy-core's
baby steps do, and BLAS never splits work across ciphertexts: each reference
runs its blocks, sources and kernel calls in order, on one scratch, the first
error stopping it, and its query reports the largest single step
([Scratch](#scratch)).

An override may choose its own level instead, never both: a backend whose
kernels are serial, a device queue for example, may run the blocks of an
operation concurrently, as long as its bits and its error are the reference's
and its query reserves what its own schedule uses. An implementation cannot add
bounds its trait lacks, so the traits carry the ones that choice needs: input
blocks and key stores are `Sync`, output blocks `Send`, the hook's sources
`Sync` and its destinations `Send`, the kernel's prepared pairs and key
`Sync`, and the prepared operands `Send + Sync`.

## Testing a backend

Enable `test-utils` on a `poulpy-blas` dev-dependency. It exports three
registration macros.

`blas_backend_test_suite!` registers the functional tests for one backend,
scalar, encoder and parameter set. They encrypt, run every operation, and check
the decrypted values and the output metadata against cleartext expectations,
along with the input guards, key widths, and the plan's key sets against the
encoded maps; each operation runs on the keys of its own set alone, and the
preparations and the prepared product under every hint and mix of hot and cold
operands. The transpose, the shifts through both half maps, the preparations,
the products and the products by public matrices also run with maps or public
matrices a quarter of the parameter set's scale below and above it, which they
consume in place of their inputs' scale, so the parameter set's `k` must hold
two maps at the larger scale and a product. Hot operands prepared under one
context serve the products under all three, and a hot operand beside a cold
one prepared under another context gives the product of their terms. Through
those maps, the products also multiply factors at
two scales, in either order, one an eighth of the scale lower and either as
much narrower, so that their terms keep one budget, or as wide, so that its
terms have the larger budget: crossed scales, where the larger budget plus the
larger scale, from which the ct x ct rule takes its offset, exceeds the wider
term's width. `test_blas_key_widths` runs with
the maps and the public matrices a quarter below that scale, where a key width
taken from the inputs' scale in place of `p` or `p_B`, or capped by it, finds no
key. They read coefficients on the host and use the standard ring, so they need
a host backend of the standard ring. An optional `log_d = …` sets the blocks to
`2^log_d x 2^log_d`, 2 by default. The ring degree `N` is free within two
bounds: the blocks fit the `N / 2` slots, `d^2 <= N / 2` (`N >= 32` at the
default), and the guards encode on a module of degree `2N`, which the backend
must support (`N <= 2^16` at most). `test_blas_context` sweeps `log_d` 1 to 3
and skips with a message those whose blocks do not fit;
`test_blas_batch_record`, below four matrix slots, and the operand at another
dimension in the guards of the products by public matrices, where only `2 x 2`
blocks fit, are skipped with a message too. `poulpy-cpu-oracle` runs them on
`FFT64Oracle` at `log_d` 1, 2 and 3 (the last at `N = 128`, where one block
fills the 64 slots), at `log_d` 1 also at `N = 16`, whose 8 slots hold no
`4 x 4` block, and on `NTT4x30Oracle` in `f64`, in `Quad` and at rank 2,
each also at radix 54 (`N = 128`), the largest in practice.
`test_blas_matmul_long_sums`, a `1 x 8` grid times an `8 x 1` grid of
distinct blocks, runs on its own ring, `N = 32`.

`blas_conjugate_invariant_test_suite!(mod m, backend = …, params = …)` registers
the test `conjugate_invariant` for a backend of the conjugate-invariant ring,
whose `N` slots are real. It encrypts real matrices of `4 x 4` blocks with the
module's own encoder, runs every operation once, each on exactly the bytes its
query reports, and checks the decrypted values and the output metadata, as the
functional tests do, and that every output, the prepared operands included, has
`Real` slots. Below `N = 16`, whose slots hold no such block, it is skipped
with a message.

`blas_parity_test_suite!` compares `backend_test` with `backend_ref`:

```rust
poulpy_blas::blas_parity_test_suite! {
    mod blas_parity_fft64portable_f64,
    backend_ref = poulpy_cpu_oracle::FFT64Oracle,
    backend_test = crate::FFT64Portable,
    scalar = f64,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 12, ..poulpy_ckks::test_suite::BASE19_PARAMS_F64 },
}
```

It registers the tests `inter`, `intra` and `matmul`, each on its own pair of
modules of degree `params.n`. Both backends run on identical fixtures and keys,
and the operations are called through the API, which forwards to their `*Impl`
methods. Every call, encodings and preparations included, gets exactly the
bytes its query reports, poisoned and guarded. Every output is compared: its
batch record, and every block bit for bit, digits and CKKS metadata. Prepared
operands, made under either hint, are compared through their infos alone,
`(log_d, rows, cols, block_layout, mat_slots)`, which must also equal their
input's batch and, as the layout, their input's when hot and its term layout
when cold, beside their hint and their input's layout as their source; the
prepared product of every mix of hot and cold operands runs on one pair per
scale of the maps into `res` of three widths, and on pairs at two scales into
`res` of one, and must equal `blas_matmul` each time. Each call looks its keys
up in a store of its
operation's set alone, from `BlasPlan` or `sum_galois_elements`, which records
every lookup: the tested backend may look up only keys, at widths, that the
reference looked up in the same call. No call may write its inputs. The calls
reach every branch of the circuit: identity rotations, one matrix slot, the
copies of row and column shifts by `d` and `2d` along three blocks, one-row and
one-column grids, `res` wider and narrower than the result, factors one limb
apart in both orders, lazy inputs on real slots, sparse unless one matrix fills
them, copies of lazy inputs, public products of one matrix and of one per matrix
slot, maps and public matrices at scales below and above the operands', factors
at two scales, in either order, through those maps, whose terms keep one budget
or give the lower scale the larger budget (crossed scales, where a kernel that
takes its offset from the wider term's width changes the bits), and a long
product, `1 x 32` by `32 x 1` distinct blocks (`1 x 16` by `16 x 1` above
radix 52), at every registration, which a kernel that drops or repeats a term
of a long sum fails. Each
family's parity also runs calls that poulpy-ckks rejects for their budget,
among them the prepared product of a hot operand one bit short of the `2p` its
terms spend: both backends return the same error and leave `res` and the inputs
unchanged. Storage
and ring are free: the bounds are `BlasParityBackend` (`Backend<ZnxWord = i64>`
and every BLAS family) and `BlasParityModule`. Rank and `dsize` follow `params`,
as in CKKS parity.

The optional `shapes` argument, a `BlasParityShapes`, sets the swept block
dimensions `log_d` (default `[1, 2]`). A `log_d` below 1, or whose `d^2`
exceeds the slot count, is skipped with a message, and the test fails when none
remains. A backend that lacks a block dimension restricts the sweep instead of
dropping the suite:

```rust
    shapes = poulpy_blas::test_suite::parity::BlasParityShapes { log_d: vec![2] },
```

`log_d` is the only shapes axis: rank and `dsize` come from `params`, and one
macro registers all three families, so none can be dropped silently.

In this workspace the oracle runs the functional tests and parity between its
FFT64 and NTT4x30 families, standard and conjugate-invariant, in both
directions, and every conjugate-invariant type, the oracle's included, runs the
conjugate-invariant suite. Every other backend type registers parity against
the oracle of its family: `NTT3x42` types against `NTT4x30Oracle`,
conjugate-invariant types against the conjugate-invariant oracles, and Rayon
types against the oracle rather than their serial sibling. The NTT types among
them also register parity at radix 54, the largest in practice: at `N = 64` on
the standard ring, and at `N = 32` on the conjugate-invariant ring, whose
products are as long as the standard ring's at `N = 64`. The FFT64 types of the
standard ring also register parity at `N = 32`, where one `4 x 4` block fills
the 16 slots.
[`blas_tests.rs`](../../poulpy-cpu-portable/src/tests/blas_tests.rs) in
`poulpy-cpu-portable` shows the registrations, and
[`blas_overrides.rs`](../../poulpy-cpu-portable/src/tests/blas_overrides.rs)
overrides the map hook and the JKLS kernel, checks parity with the oracle
through both, and checks that no rejected call enters either. Commands and CI
coverage belong in the [repository README](../../README.md#testing-a-backend).

## Attestation

Parity is bit-exact, so an override must compute the reference's circuit. An
algorithm that is only numerically close fails it.

An override ships with its parity test passing against an attested backend.
Attestation is transitive ([CONTRIBUTING](../../CONTRIBUTING.md#coding-conventions)):
`poulpy-cpu-oracle` runs the reference directly, and a backend that passes
parity against an attested one can serve as the reference for the next, for the
same operations and parameter ranges. There are three routes:

- **Inside the workspace**, against the oracle of the backend's family, from a
  `poulpy-cpu-oracle` dev-dependency with `enable-blas`.
- **Outside the workspace, against the oracle.** The oracle is not published,
  so it comes as a git dev-dependency. It takes the other poulpy crates from its
  own workspace, so the crate takes every poulpy crate from that one revision,
  or redirects its crates.io ones there with `[patch.crates-io]`:

  ```toml
  [dependencies]
  poulpy-hal = {git = "https://github.com/poulpy-fhe/poulpy", rev = "<rev>"}
  poulpy-core = {git = "https://github.com/poulpy-fhe/poulpy", rev = "<rev>"}
  poulpy-ckks = {git = "https://github.com/poulpy-fhe/poulpy", rev = "<rev>"}
  poulpy-blas = {git = "https://github.com/poulpy-fhe/poulpy", rev = "<rev>"}

  [dev-dependencies]
  poulpy-blas = {git = "https://github.com/poulpy-fhe/poulpy", rev = "<rev>", features = ["test-utils"]}
  poulpy-cpu-oracle = {git = "https://github.com/poulpy-fhe/poulpy", rev = "<rev>", features = ["enable-blas"]}
  ```

- **Outside the workspace, from crates.io**, against the published
  `poulpy-cpu-portable` with `enable-blas`, which is attested against the oracle
  here.

## Backends outside the workspace

Everything above applies to a backend crate outside this workspace,
closed-source ones included. It defines its own backend type, implements the
HAL, core and CKKS families for it, and may then take any BLAS family from its
macro or override it. Three Rust rules constrain its wiring:

- **It wires only its own types.** Implementing a BLAS trait for another
  crate's type, a CPU backend of this workspace included, breaks the orphan
  rule (E0117). The CPU backends wire theirs behind `enable-blas`.
- **One source.** `poulpy-hal`, `poulpy-core`, `poulpy-ckks`, `poulpy-blas` and
  the comparison backend come from one source and revision. Two copies of a
  crate define distinct types and traits, so a crates.io copy beside a git one
  does not compile: trait bounds fail (E0277) and the module lacks the expected
  methods (E0599), which rustc traces to multiple versions of a crate in the
  dependency graph.
- **Direct dependency names.** The wiring macros expand to `::poulpy_hal`,
  `::poulpy_core` and `::poulpy_ckks`, and the registration macros to
  `::poulpy_hal`, so the crate that invokes them depends on `poulpy-hal`,
  `poulpy-core` and `poulpy-ckks` directly, under those names (E0433
  otherwise). The parameter sets a registration passes as `params` live in
  `poulpy_ckks::test_suite`, which the `test-utils` feature of `poulpy-blas`
  enables.
