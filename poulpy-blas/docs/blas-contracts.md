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
reference preparations likewise call the backend's map hook, and the reference
prepared product its JKLS kernel. So `impl_blas_matmul_reference!` needs the
intra and JKLS families, and an override of the hook or of the kernel reaches
every operation built on it.

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
matrices take `atk` alone, the prepared product `tsk` alone, and `blas_matmul`
both. The traits' `Send` and `Sync` bounds let an implementation spread blocks
over the backend's task executor ([Parallelism](#parallelism)).

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
| `blas_matmul_prepare_left`, `blas_matmul_prepare_right`, `blas_matmul` | `matmul_galois_elements(order)` | sigma's (tau's) baby steps at `a.k` and giant steps at `a.k - p`; the shifts' giant steps at `a.k - 2p` |
| `blas_matmul_pt`, `blas_matmul_pt_left` | `matmul_pt_galois_elements(order)`, `matmul_pt_left_galois_elements(order)` | baby steps at `a.k`, giant steps at `min(res.k, a.k - p_B)` |

The sets list the rotations of the reference's maps as the plan schedules
them: shift maps `Direct`, the others at the cost-optimal BSGS giant step.
`galois_elements(order)` covers every map of a context, so every operation's
set but `blas_rotate`'s and `blas_sum`'s. The tensor key is needed if and only
if a pipeline calls `blas_matmul` or `blas_matmul_prepared`, at `max(w_a, w_b)`,
the wider of the operands' term widths `a.k - 2p` and `b.k - 2p`. The
[pipeline example](../../poulpy-cpu-portable/examples/blas_pipeline.rs) plans
its keys this way, with one rotation-key layout covering its widest input and a
tensor key covering the terms, and prints the width each operation consumes.

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
  the transpose, both shifts and both preparations, so `blas_matmul` too:

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

- **The products**, for prepared operands in the backend's own layout. The
  inter, intra and JKLS families keep their macros, and `BlasMatMulImpl` is
  written by hand: it sets both operand types, implements both preparations
  and the prepared product, and forwards `blas_matmul_pt` and
  `blas_matmul_pt_left` to `BlasMatMulReference`, while `blas_matmul` stays
  provided.

A backend may also override a provided method, for example the shifts, to
chunk their heap grid, or `blas_matmul`, to stream its operands; it then
reproduces the circuit the provided body fixes. A Rayon backend writes
nothing: the references already split their blocks over the backend's task
executor ([Parallelism](#parallelism)).

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
   batch record; and lazy sums a digit cannot hold: more `blas_sum` terms,
   `log2(mat_slots) + 1`, than `2^(63 - base2k)`, or a product at a radix
   above 62.
2. `MissingAutomorphismKey` for a rotation key missing at the width the
   operation looks it up at ([Keys](#keys)), then `MissingRelinearizationKey`
   for the tensor key.

They check no budget. poulpy-ckks raises the budget errors,
`MultiplicationPrecisionUnderflow` and `InsufficientHomomorphicCapacity`,
inside the implementation, from the rotations, the copies, the maps, the
products by public matrices and the JKLS kernel's dot product, before any write
to `res`: every block of a grid shares one layout, so the first block's call
fails before any block is written, and each parallel task fails before it
writes its own block. A preparation writes only the operand it returns, and
`blas_matmul` meets the preparations' errors before the product's.

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

**Free:** the order of the blocks and their split over workers.

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
order of the blocks and their split over workers; one baby-step cache per
source across its maps, as the reference keeps; batching key switches;
normalizing a source read through a DFT.

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

The reference preparations call the hook too ([Products](#products)), so a
backend that overrides the hook alone evaluates every map of this crate but
those of the products by public matrices.

### Products

**Prepared operands.** `MatMulLeft` and `MatMulRight`, which the API names
`BlasMatMulLeft<BE>` and `BlasMatMulRight<BE>`, may hold any representation,
such as device memory, contiguous or DFT-domain terms, and only the
preparations make one. Through `BlasGridInfos`, an operand prepared from `a`
under `ctx` reports `a`'s dimensions, the layout
`poulpy_blas::reference::blas_matmul_term_layout(a, ctx)` gives, and `Some` of
`a`'s batch, `grid_batch::<BE::Ring>(a)`: the prepared product's validator,
the batch the API records and the provided `blas_matmul` query read nothing
else, and parity compares these infos, never the storage. The prepared product
of two operands equals the reference's on the reference's operands of the same
sources. The preparations look up rotation keys only, the prepared product
the tensor key only. The reference's operand, `BlasMatMulOperand<BE, Left>`
(`Right`), holds the `d` terms of every block, which `term(row, col, k)`
returns.

**Fixed:**

- The terms. Each source block `a_il` has `d` terms, all at the width
  `w = a.k - 2p`: term `0` is sigma of `a_il` evaluated into a block of `a`'s
  layout, which leaves it at `a.k - p`, then narrowed exactly to `w` by
  `ckks_copy`, which charges the shift to the budget; term `k` is phi^k of that
  sigma result into a block of width `w`, through the `Direct` shift maps the
  plan records. On the right, tau and psi^k. The terms depend on neither `res`
  nor the other factor: the kernel's offset absorbs a narrower `res`. The
  reference makes them with two hook calls, sigma over every block, then every
  shift of every sigma result, from one baby-step cache per result.
- One JKLS kernel per block `res_ij`, over its `K d` pairs of terms
  `(a_il[k], b_lj[k])` ([The JKLS kernel](#the-jkls-kernel)).
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

**Free:** the order and split of the blocks over workers; materializing or
streaming the terms; caching them or their convolution forms; sharing
baby-step caches; batching key switches.

`blas_matmul` is provided as both preparations, then the prepared product, so
it reaches the backend's methods; for an `m x K` by `K x n` product it holds
`K (m + n) d` heap terms, and a streaming variant overrides it. Hoisting the
shifts is the one reserved variant: it changes the bits and the key widths, so
it would arrive as a `BlasPlan` variant, not as an override.

### The JKLS kernel

`BlasJklsImpl::blas_jkls_block_impl` computes one block of the prepared
product, `res = sum_t lhs[t] rhs[t]`, as poulpy-ckks's `ckks_dot_product_ct`
does for pairs that share one layout per side, which is the reference. It is
called with `lhs.len() == rhs.len() >= 1` pairs, every `lhs` of one layout and
every `rhs` of another, at a radix of 62 at most, after the validator found the
tensor key at the tensor width `max(lhs.k, rhs.k)`.

**Fixed:**

- Each term is one core tensor product of `(lhs[t], rhs[t])` at the offset
  poulpy-ckks's ct x ct rule gives for `res`, rounded on its own at the tensor
  width. Summing unrounded products, as `cnv_apply_dft_sum` would, changes
  bits.
- The terms are summed exactly and normalized at that width; `res` is stamped
  with the rule's budget and scale, the smallest label and the joined slot
  kind, then relinearized once, with the tensor key at that width. `res` comes
  back canonical, without a noise tag.

**Free:** the order, grouping and split of the terms over workers,
materialized or streamed; extra exact normalizations of the sum; caching the
operands or their convolution forms; normalizing an operand read through a
DFT.

**Normalization.** Whatever the grouping, no accumulator, partial sums
included, holds more than `2^(63 - base2k)` terms since its last normalization,
a normalized sum counting as one: partial sums are normalized before they are
combined unless their counts fit together. Parity cannot check this rule. A
canonical digit is at most `2^(base2k - 1)` in magnitude, so a 64-bit digit
holds `2^(64 - base2k)` terms, twice the bound, and random terms add up far
below it: a kernel that skips every normalization but the last still matches
the reference on every registered case. The rule is an obligation of the
`# Safety` section, met by construction.

Within the scratch its query reports for `res`, `lhs[0]`, `rhs[0]`, the pair
count and the key, it returns `Ok`, or poulpy-ckks's budget error before any
write to `res`; with less scratch it panics before any write. The reference
prepared product calls it once per block of `res`, concurrently on a parallel
executor, each call on its own block and scratch slice.

## Scratch

Every method runs within the scratch its own `_tmp_bytes_impl` reports for the
layouts and keys it receives, which may differ from the reference's: parity
gives each backend exactly the bytes its own query reports, poisoned and
guarded, so an underestimate fails there. The references assert that bound on
entry and panic before any write when it falls short, for a call that looks up
a key.

The queries read layouts (`BlasGridInfos`, `CKKSCtBounds`, `GGLWEInfos`), never
ciphertext data: the prepared product's query receives the prepared operands,
or, before preparing, the layouts `blas_matmul_term_layout` gives, so it reads
nothing but their infos. The provided and reference queries call the selected
implementations' queries: the provided transpose the hook's; the provided
shifts the hook's, the copy's and the addition's; the reference preparations
the hook's through sigma (tau) and through the shifts, and the copy's; the
reference prepared product the kernel's; and the provided `blas_matmul` the
largest of both preparations' queries and the prepared product's on the term
layouts. An override of the hook or of the kernel that needs other scratch
reports it in its own query, and every operation over it follows.

On a parallel executor the reference queries reserve one slice per worker, up
to four, and keep the serial size otherwise ([Parallelism](#parallelism)).
Temporaries that outlive a hook call live on the heap, not in scratch: the
sigma results and terms of a preparation, and the heap grid of a shift across
blocks. A query may panic on layouts its validator rejects, as its `# Panics`
section states.

## Parallelism

Executor tasks are `Fn + Send + Sync`, and an implementation cannot add bounds
its trait lacks, so the traits carry them: input blocks and key stores are
`Sync`, output blocks `Send`, the hook's sources `Sync` and its destinations
`Send`, the kernel's pairs and key `Sync`, and the prepared operands
`Send + Sync`.

The references split their work over up to four workers of `BE::TaskExecutor`,
each on its own scratch slice: the rotation and the sum by block; the hook by
runs of consecutive sources, each worker with its own baby-step cache; the
prepared product by block of `res`, one kernel call each; and the products by
public matrices by runs of block rows (block columns, on the left), each worker
with its own caches. On failure they return the error of the lowest failing
task, as a serial loop would; a serial executor runs the tasks in order and
stops at the first error. The sum's first copies, the shifts' copies and
additions and the preparations' narrowing copies run serially.

The split is free. A product with fewer output blocks, rows or columns than
workers leaves workers idle in the reference; an override may split finer, for
example one block's JKLS terms over workers, as long as its bits and its error
are the reference's, and its query reserves what its own split uses.

## Testing a backend

Enable `test-utils` on a `poulpy-blas` dev-dependency. It exports three
registration macros.

`blas_backend_test_suite!` registers the functional tests for one backend,
scalar, encoder and parameter set. They encrypt, run every operation, and check
the decrypted values and the output metadata against cleartext expectations,
along with the input guards, key widths, and the plan's key sets against the
encoded maps; each operation runs on the keys of its own set alone. They read
coefficients on the host and use the standard ring, so they need a host backend
of the standard ring. An optional `log_d = …` sets the blocks to
`2^log_d x 2^log_d`, 2 by default. `poulpy-cpu-oracle` runs them on
`FFT64Oracle` at `log_d` 1, 2 and 3 (the last at `N = 128`, where one block
fills the 64 slots), and on `NTT4x30Oracle` in `f64`, in `Quad` and at rank 2,
each also at radix 57 (`N = 128`). `test_blas_matmul_long_sums`, sums at least
three times longer than the `2^(63 - base2k)` terms the JKLS kernel sums
between normalizations, runs only for a suite registered at `base2k >= 57`;
every other suite skips it with a message.

`blas_conjugate_invariant_test_suite!(mod m, backend = …, params = …)` registers
the test `conjugate_invariant` for a backend of the conjugate-invariant ring,
whose `N` slots are real. It encrypts real matrices of `4 x 4` blocks with the
module's own encoder, runs every operation once, each on exactly the bytes its
query reports, and checks the decrypted values and the output metadata, as the
functional tests do, and that every output, the prepared operands included, has
`Real` slots.

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
operands are compared through their infos alone, `(log_d, rows, cols,
block_layout, mat_slots)`, which must also equal their input's term layout and
batch; the prepared product runs on one pair into `res` of three widths and
must equal `blas_matmul` each time. Each call looks its keys up in a store of
its operation's set alone, from `BlasPlan` or `sum_galois_elements`, which
records every lookup: the tested backend may look up only keys, at widths, that
the reference looked up in the same call. No call may write its inputs. The
calls reach every branch of the circuit: identity rotations, one matrix slot,
the copies of row and column shifts by `d` and `2d` along three blocks,
one-row and one-column grids, `res` wider and narrower than the result, factors
one limb apart in both orders, lazy inputs on real slots, sparse unless one
matrix fills them, copies of lazy inputs, public products of one matrix and of
one per matrix slot, and, where some `K <= 32` reaches it, a `1 x K` by `K x 1`
product of more than the `2^(63 - base2k)` terms the kernel sums between
normalizations. Each family's parity also runs calls that poulpy-ckks rejects
for their budget: both backends return the same error and leave `res` and the
inputs unchanged. Storage and ring are free: the bounds are
`BlasParityBackend` (`Backend<ZnxWord = i64>` and every BLAS family) and
`BlasParityModule`. Rank and `dsize` follow `params`, as in CKKS parity.

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

**The radix rule.** The long product runs only where some `K <= 32` sums
more than `2^(63 - base2k)` terms: from radix 57 at `log_d` 2 (58 at `log_d`
1), far above the radix an FFT64 transform holds, which stops near 23. An
NTT-family backend therefore registers parity with one parameter set at
`base2k >= 57`, `log_d` 2 swept, on each ring it supports. That registration
pins that the kernel accepts a sum past the bound and returns the reference's
bits; it does not show that the kernel normalizes, which stays a `# Safety`
obligation ([The JKLS kernel](#the-jkls-kernel)).

In this workspace the oracle runs the functional tests and parity between its
FFT64 and NTT4x30 families, standard and conjugate-invariant, in both
directions, and every conjugate-invariant type, the oracle's included, runs the
conjugate-invariant suite. Every other backend type registers parity against
the oracle of its family: `NTT3x42` types against `NTT4x30Oracle`,
conjugate-invariant types against the conjugate-invariant oracles, and Rayon
types against the oracle rather than their serial sibling. The NTT types among
them also register parity at radix 57: at `N = 64` on the standard ring, and at
`N = 32` on the conjugate-invariant ring, whose products are as long as the
standard ring's at `N = 64`, since the cross-modulus `NTT3x42` parity needs
NTT4x30's modulus to hold every radix-57 product. The FFT64 types of the
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
