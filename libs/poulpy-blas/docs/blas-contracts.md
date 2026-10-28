# Implementing BLAS operations

Public matrix calls follow the [layering contract](../../../docs/guides/layering-contract.md): `Module<BE>` implements
`BlasMatInterOps`, `BlasMatIntraOps` and `BlasMatMulOps`, and `BlasEncodingOps`,
`BlasPtOperandPreparedFactory` and `BlasContextPreparedFactory`, for every
backend that implements the matching `*Impl` trait, and each OEP method of an
operation runs its validator first. The BLAS
crate imports no backend crate: each backend crate wires its own types.

This file holds the rules every family shares. What an implementation of one
family must reproduce, and what it may change, is the doc of each method and
hook of that family's trait (its **Fixed** and **Free** lists), the trait's
`# Safety` holding what the methods share; the output metadata and the errors of
each operation are on its API method. Parity is bit-exact, so an override
computes the reference's circuit: anything a method's doc does not list as free
changes bits.

## Families

| OEP family | Required | Provided |
|---|---|---|
| `BlasMatInterImpl` | `blas_rotate`, `blas_sum` | |
| `BlasMatIntraImpl` | the map hook `blas_map_blocks`, `blas_rotate_rows`, `blas_rotate_cols`, `blas_sum_rows`, `blas_sum_cols` | `blas_transpose`, over the hook |
| `BlasMatMulImpl` | `blas_matmul`, `blas_matmul_prepare_left`, `blas_matmul_prepare_right`, `blas_matmul_prepared`, `blas_matmul_pt_right`, `blas_matmul_pt_left` | |
| `BlasJklsImpl` | the JKLS kernel `blas_jkls_block`, one output block of the prepared product | |
| `BlasEncodingImpl` | `blas_pt_operand_encode`, `blas_context_encode`, `blas_pt_operand_prepare`, `blas_context_prepare` | |

Each operation's `_tmp_bytes` query belongs to the same family. The encodings
take poulpy-ckks's encoding scalar `F` as a method parameter bounded by
`Self: CKKSEncodingImpl<F>`, and the allocations of contexts and public
operands, prepared or not, stay direct on the module. `BlasImpl`
aggregates the five families as a test bound; it has a blanket implementation
and is never implemented directly.

The provided transpose, and the references of the shifts, of the reductions
and of the products, call the backend's selected map hook, and the references
of the products also its JKLS kernel. So an override of the hook or of the
kernel reaches every operation built on it. The provided body is crate-private,
`blas_transpose_derived` with its query `blas_transpose_tmp_bytes_derived`, in
the family's file under `oep::derived`.

The prepared operands of the products, `BlasMatMulLeft<BE>` and
`BlasMatMulRight<BE>`, are `poulpy_blas::layouts::BlasMatMulOperand`s of the
backend's poulpy-ckks storage, a type the backend does not choose.
`BlasMatMulOperandPreparedFactory` allocates them; an implementation of the
preparations and of the prepared product reads and fills their storage through
`blocks`, `terms` and their `_mut` forms, and records a preparation with
`set_prepared`.

## Reference implementations

The references compose `poulpy-ckks` and `poulpy-core` through their public
APIs, so a backend's CKKS and core overrides remain effective inside them.

| OEP family | Reference | Macro |
|---|---|---|
| `BlasMatInterImpl` | `BlasMatInterReference`, [`reference/inter.rs`](../src/reference/inter.rs) | `impl_blas_mat_inter_reference_full!` |
| `BlasMatIntraImpl` | `BlasMatIntraReference` (the hook, the shifts and the reductions), [`reference/intra.rs`](../src/reference/intra.rs) | `impl_blas_mat_intra_reference_full!` |
| `BlasMatMulImpl` | `BlasMatMulReference`, [`reference/matmul.rs`](../src/reference/matmul.rs) | `impl_blas_mat_mul_reference_full!` |
| `BlasJklsImpl` | `BlasJklsReference` (the kernel), [`reference/jkls.rs`](../src/reference/jkls.rs) | `impl_blas_jkls_reference_full!` |
| `BlasEncodingImpl` | `BlasEncodingReference`, [`reference/encoding.rs`](../src/reference/encoding.rs) | `impl_blas_encoding_reference_full!` |

`impl_blas_reference_full!` expands to the five family macros. The reference's
cold preparations call the backend's map hook, and the reference prepared
product, like the reference `blas_matmul`, calls it for the terms of a hot
operand, then its JKLS kernel. So `impl_blas_mat_mul_reference_full!` needs the intra
and JKLS families. The reference `blas_matmul` is the reference prepared
product of two hot operands, bit for bit, on the grids read in place, without
the preparations' copies.

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
module, declared under `enable-blas`, selects the reference for each type, with
the same `#[cfg]` gates as `ckks_impl.rs`:

```rust
::poulpy_blas::impl_blas_reference_full!(crate::MyBackend);
```

To replace a family, replace the type's `impl_blas_reference_full!` line with
the macros of the four families it keeps, as the [layering contract](../../../docs/guides/layering-contract.md)
states. The expansion of each family macro, in [`oep`](../src/oep/), spells out
every signature and its forwarding call, and the docs of each family's methods
and hooks state what an implementation must reproduce. Three replacements cover most needs:

- **The map hook**, for a backend that evaluates maps its own way. It covers
  the transpose, both shifts, both reductions, the cold preparations and the hot
  sides of the products, `blas_matmul` included:

  ```rust
  ::poulpy_blas::impl_blas_mat_inter_reference_full!(MyBackend);
  ::poulpy_blas::impl_blas_mat_mul_reference_full!(MyBackend);
  ::poulpy_blas::impl_blas_jkls_reference_full!(MyBackend);
  ::poulpy_blas::impl_blas_encoding_reference_full!(MyBackend);
  unsafe impl poulpy_blas::oep::BlasMatIntraImpl for MyBackend {
      /* the hook and its query; the shifts, the reductions and their queries forwarded to BlasMatIntraReference */
  }
  ```

- **The JKLS kernel**, a fused kernel behind the backend's own `enable-blas`.
  Both `blas_matmul` and `blas_matmul_prepared` reach it:

  ```rust
  ::poulpy_blas::impl_blas_mat_inter_reference_full!(MyBackend);
  ::poulpy_blas::impl_blas_mat_intra_reference_full!(MyBackend);
  ::poulpy_blas::impl_blas_mat_mul_reference_full!(MyBackend);
  ::poulpy_blas::impl_blas_encoding_reference_full!(MyBackend);
  unsafe impl poulpy_blas::oep::BlasJklsImpl for MyBackend { /* the kernel and its query */ }
  ```

- **The products**, in the backend's own evaluation. The inter, intra, JKLS
  and encoding families keep their macros, and `BlasMatMulImpl` is written by hand: it
  implements the methods it replaces, each running its validator and stamping
  the batch as the reference does, and forwards the others to
  `BlasMatMulReference`, `blas_matmul` to `blas_matmul_reference` as the macro
  does. Preparations and a prepared product read and fill the operands through
  `BlasMatMulOperand::{blocks, terms}`, their `_mut` forms and `set_prepared`.

A backend may also override a shift, for example to chunk its heap grid, in a
hand-written `BlasMatIntraImpl`; it then reproduces the circuit the reference
fixes.

## Grids and keys

`res` is a `BlasGridMut<BE>` and every ciphertext input a `BlasGrid<BE>`,
which expose the blocks of a grid as a row-major slice. The prepared product
reads prepared operands instead, and the products by public matrices also read
a `BlasPtOperand`. A grid's blocks number exactly `rows x cols`, at least one,
and every operation rejects a grid that breaks this. `BlasCiphertext` over
owned storage implements both traits on every backend. Borrowed views are not
grids: a `BufRef` view is `Sync`, but its blocks are not `GLWEToBackendRef`.

Rotation keys come as `keys` and the tensor key as `tsk`, two separate stores:
the rotations, the sums, the maps, the preparations and the products by public
matrices take `keys` alone, and the products both. The prepared product looks
up rotation keys only for a hot operand, whose terms it evaluates: of two cold
operands it reads `tsk` alone. The traits' `Send` and `Sync` bounds leave an
implementation its own choice of concurrency ([Parallelism](#parallelism)).

## Batches

A grid records its batch, `BlasGridInfos::mat_slots`: the number of matrices
its blocks hold. `BlasCiphertext::from_blocks` records the batch the blocks'
label proves, and a grid without a record (`mat_slots()` is `None`) is read at
that batch. Every operation rejects a grid whose label does not allow its
batch: with `S` slots, degree `n` and label `s`, the batch is a power of two
between `(S >> s) / d^2` and `S / d^2`, and `s` is at most `log2 n`. The
implementation stamps the record of `res` once it succeeds, with the batch its
validator returns (`BlasGridMut::set_mat_slots`): products record the larger
batch of their operands, every other operation its input's. It reads an input's
batch with `poulpy_blas::layouts::grid_batch`: its record, or for a grid
without one the batch its label proves. A preparation records that batch on
its operand.

Encryption labels a block from its plaintext's degree, so a batch of `count`
matrices comes from plaintexts of `count d^2` slots, of degree `2 count d^2`
on the standard ring and `count d^2` on the conjugate-invariant one. A
plaintext's degree is at least the backend's `MIN_DEGREE`, which some backends
set above the default of 8, and a degree raised to it records a larger batch.
Such a batch therefore records `count` on every backend only when its degree
reaches every target's `MIN_DEGREE`; otherwise encrypt it at the ring degree.

## Keys

Callers generate rotation keys from the per-operation sets of `BlasPlan`,
which take the ring's cyclotomic `order`, and from `sum_galois_elements`. Every
operation looks its keys up at fixed widths, for the maps' scale `p` and a
public operand's scale `p_B`:

| Operation | Rotation keys | Widths |
|---|---|---|
| `blas_rotate` by `r` | `rotate_galois_elements(order, r)` | `min(res.k, a.k)` |
| `blas_sum` | `sum_galois_elements(order, a)` | `min(res.k, a.k)` |
| `blas_transpose` | `transpose_galois_elements(order)` | baby steps at `a.k`, giant steps at `min(res.k, a.k - p)` |
| `blas_rotate_rows`, `blas_rotate_cols` by `k` | `rotate_rows_galois_elements(order, k)`, `rotate_cols_galois_elements(order, k)` | as the transpose; none for a shift by a multiple of `d`, a copy; a row shift of a replicated grid one block high, a rotation by `k d`, at `min(res.k, a.k)` |
| `blas_sum_rows`, `blas_sum_cols` | `sum_rows_galois_elements(order)`, `sum_cols_galois_elements(order)` | the tree's rotations at `a.k`, the broadcast's at `min(res.k, a.k - p)`; along the rows of a replicated grid, the tree's at `min(res.k, a.k)` and no broadcast |
| `blas_matmul_prepare_left`, `blas_matmul_prepare_right`, `blas_matmul`, and `blas_matmul_prepared` of a hot operand | `matmul_galois_elements(order)` | sigma's (tau's) baby steps at `a.k` and giant steps at `a.k - p`; the shifts' giant steps at `a.k - 2p`, those of a replicated right factor `b` at `b.k - p` |
| `blas_matmul_pt_right`, `blas_matmul_pt_left` | `matmul_pt_right_galois_elements(order)`, `matmul_pt_left_galois_elements(order)` of the operand's plan | under `Diagonal`, the blocks' baby steps at `a.k`, giant steps at `min(res.k, a.k - p_B)`; under `Jkls`, the first map's baby steps at `a.k`, giant steps at `a.k - p_B`, then the blocks' baby steps at `a.k - p_B`, giant steps at `min(res.k, a.k - 2 p_B)`; under `JklsStored` and `JklsGalois`, the first map's as under `Jkls`, then the context's masks', `Direct`, at `a.k - p_B - p` |

A plan also carries a `BlasPtAlgorithm` through `BlasPlan::with_pt_algorithm`,
which the public operands encoded from it follow. `Diagonal`, the default, maps
each block once, one scale. `Jkls` maps every block of the encrypted grid by
sigma (tau) once, then by one map per block, two scales. `JklsStored` and
`JklsGalois` put JKLS's masks on the ciphertext: that first map, then the
context's masks of its results, phi^k on the right and the column halves of
psi^k on the left, then one sum of products by prepared public plaintexts
(poulpy-ckks's `ckks_dot_product_prepared_pt`), three scales; `JklsStored`
stores `d` plaintexts per block on the right and `2d - 1` on the left,
`JklsGalois` one, read through the Galois elements of their rotations. On the
right these two take one public matrix. Each path is its own circuit. The
products take the context, which only the last two read; a context encodes the
column halves of psi^k only when its plan selects one of them, and a left
product under these paths needs such a context. The first two look up the same
rotation keys; the last two add their masks', which their plan's sets list, at
the widths of the last row.

A left product of a replicated input by an operand of one public matrix on more
than one matrix slot takes the operand's replicated-input block maps
(`BlasPtOperand::block_replicated`), `d` diagonals scheduled at `left_mul`'s
giant step, so they rotate within `matmul_pt_left_galois_elements` and the
table's widths hold: about 22 key switches per block instead of 30 at
`d = 128`.

The sets list the rotations of the reference's maps as the plan schedules them:
shift maps `Direct`, the others at the cost-optimal BSGS giant step. The maps
of a replicated right factor rotate within `matmul_galois_elements`:
`BlasContext::tau_rep` at tau's giant step, and `BlasContext::rotate_rows_rep(k)`
by `k d`, `Direct`, one of psi^k's rotations, as is the rotation of a row
shift of a replicated grid one block high. The reductions rotate by `2^i d`
(`2^i`) and its opposite, rotations of psi^(2^i) and psi^(d - 2^i) (phi^(2^i)
and phi^(d - 2^i)), and their projections take no key. `galois_elements(order)`
covers every map of a context, so every operation's set but `blas_rotate`'s and
`blas_sum`'s. The preparations check the keys of their terms under either hint,
and the prepared product those of a hot operand's under its own context. The
tensor key is needed if and only if a pipeline calls `blas_matmul` or
`blas_matmul_prepared`, at `max(w_a, w_b)`, the wider of the operands' term
widths `a.k - 2p` and `b.k - 2p`, or `b.k - p` for a replicated right factor.
Every key has the rank of the blocks it switches.

These sets and widths bind implementations: `GetAutomorphismKey` and
`GetTensorKey` receive the width, so a store may answer per width. An
implementation looks up only keys, at widths, that the reference looks up, so
a store holding its operation's set alone suffices.

### Key-use queries

Each operation has a query in `poulpy_blas::layouts` that lists the keys it
looks up: `blas_matmul_key_uses`,
`blas_matmul_prepare_left_key_uses`, `blas_matmul_prepare_right_key_uses`,
`blas_matmul_prepared_key_uses`, `blas_matmul_pt_right_key_uses`,
`blas_matmul_pt_left_key_uses`, `blas_transpose_key_uses`,
`blas_rotate_rows_key_uses`, `blas_rotate_cols_key_uses`,
`blas_rotate_key_uses`, `blas_sum_key_uses`, `blas_sum_rows_key_uses` and
`blas_sum_cols_key_uses`. A
query takes the grids, operands and context of the operation's `_tmp_bytes`
query, without the key layouts. It leaves out what the keys do not depend on:
the result grid of a product of two encrypted grids, and the `PrepareHint` of a
preparation, which looks up the same keys under either. A rotation and a shift
take the amount `r` or `k`, which selects their maps. A query returns a
`poulpy_blas::layouts::BlasKeyUses`:

- `rotations`, the `(Galois element, width)` pairs the operation looks up,
  sorted and without repeats, the identity (element 1), which takes no key,
  left out;
- `tensor`, the width of the tensor key, `None` where the operation does not
  relinearize.

The validator of an operation and its query enumerate the lookups with one
function, so they list the same keys at the same widths; the validator checks
each against the stores, in the enumeration's order, and the query collects
them.

The output-layout queries next to them (`blas_matmul_output_layout`,
`blas_matmul_prepared_output_layout`, `blas_matmul_pt_right_output_layout`,
`blas_matmul_pt_left_output_layout`, `blas_transpose_output_layout`,
`blas_rotate_rows_output_layout`, `blas_rotate_cols_output_layout`,
`blas_rotate_output_layout`, `blas_sum_output_layout`, and in
`poulpy_blas::layouts` `blas_sum_rows_output_layout` and
`blas_sum_cols_output_layout`) take the input layouts
and the allocated `res` and return the `BlasLayout` the call stamps on `res`,
whose block gives `k`, `log_delta` and label; the record is the validator's.
They read a map's effect through the same function the preparations and staged
blocks use, so a client plans every intermediate grid, and with it every key
width, before it encrypts or runs anything.

## Validation and errors

Each implementation first runs its validator, the public `validate_<op>` in
`poulpy_blas::layouts`, from metadata alone, before any write, and after `Ok`
stamps the batch record of `res` with the batch the validator returns; a
preparation records its source's batch on its operand instead. The API
dispatches to the backend without a check, so an override runs the same
`validate_<op>` and stamps the batch as the reference does, and returns the
reference's errors, in the reference's order. The validators check this
crate's rules and fail in this order:

1. `CKKSCompositionError::InvalidOperand` for an operand they reject: a grid
   at another `log_d`, whose blocks do not tile `rows x cols` or do not share
   one layout, whose `log_delta` exceeds its `k`, or whose label does not allow
   its batch record; a shape or inner dimension that does not match; blocks of
   another degree than the module's, or inputs of another radix or rank than
   `res`; a context or public operand encoded for another slot count or radix,
   or a context wider than the blocks its maps read, which an intra operation
   checks only on a path that evaluates a map: a shift by a multiple of `d`,
   the row shift of a replicated grid one block high and the row sum of a
   replicated grid check the slot count alone; public matrices that do not
   divide the batch; under `JklsStored` and `JklsGalois`, more than one public
   matrix on the right, and a context at
   another `log_d`, encoded for another slot count or radix, wider than the
   input or, on the left, without their masks; a prepared operand without a
   batch record, or a hot one at another `log_d` than the product's context
   or under a context encoded for another slot count or radix; and a product
   at a radix above 62, where two canonical tensor digits can overflow an
   `i64`, as poulpy-ckks's prepared dot product rejects it. A hot operand's
   scale is not checked: the product spends that of its context's maps. No
   radix is rejected for the length of a lazy sum, which normalizes instead
   ([Lazy sums](#lazy-sums)).
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
fills. A hot operand's terms are evaluated by the product, so its budget error
comes from the product, the left side's before the right side's, and before
the kernel's; a hot preparation checks no budget. So `blas_matmul`, whose
operands are hot, meets its factors' budget errors before the kernel's.

A rejected call therefore leaves `res`, its batch record and the inputs
untouched, whichever backend runs it. An implementation needs no checks beyond
its validator. On inputs it accepted, it returns `Ok`, or the reference's
budget error before any write to `res`; any other `Err` is a backend fault, a
validator gap or a device failure, and `res` is then unspecified. It stamps the
batch record only after `Ok`, with the batch the validator computed. Since keys
are checked first, an input that poulpy-ckks rejects for its budget may ask for
a key that poulpy-ckks would not look up for it: a store holding exactly the
keys in use then reports `MissingAutomorphismKey` instead of the budget error.

## Lazy sums

The reductions sum the blocks of a line with poulpy-ckks's
`ckks_add_many`, which normalizes the sum in place after every
`t / w - 1` additions, so that it holds `t` canonical digits per coefficient at
most, for `w` the digits a block holds, 1 on a line of canonical blocks and 2
otherwise, and `t`
poulpy-core's `max_lazy_terms` over the sum's `n (rank + 1) size` coefficients,
the larger of two counts:

- the worst case, `2^(63 - base2k)` digits of magnitude at most
  `2^(base2k - 1)`, whose sum stays in the normalizer's domain
  `[-2^62, 2^62]`;
- the average case of `docs/guides/base2k-failure-probability.md`: the most
  independent centered uniform digits, of deviation `2^base2k / sqrt(12)`,
  whose sum leaves `[-2^62, 2^62]` with probability at most `2^-128` over
  those coefficients, by the Mills-ratio bound on `erfc` and a union bound.

The average case binds from 56 bits down: at `n = 2^16`, rank 1 and 20 limbs,
52 bits hold 62526 terms where the worst case allows 2048, and 62 bits hold 2.
There two flag-clear blocks already exceed the bound, so such a line is summed
block by block, each flag-clear block first normalized into a scratch block and
the sum normalized before each addition. As everywhere in base2k arithmetic,
the inputs are taken as independent: correlated inputs, such as thousands of
copies of one block, can overflow. A flag-clear input block holds at most two
canonical digits per coefficient, the sum of two canonical values, as a shift's
half-map sum or one `ckks_add_into` of canonical blocks leaves it: a caller
obligation, since a sum of flag-clear blocks or a `ckks_add_many` normalized
less often holds more. None of this is checked.

The trees of the reductions and `blas_sum` are poulpy-ckks's inner sums, and
the broadcasts its replicates, sums rather than means, over poulpy-core's
summed trace, whose automorphism at each level leaves the sum canonical, so
their results are canonical.

## Scratch

Every method runs within the scratch its own `_tmp_bytes` reports for the
layouts and keys it receives, which may differ from the reference's; the
reference asserts its bound on entry and panics before any write when it falls
short, for a call that looks up a key. An override that needs other scratch
reports it in its own query, and every operation over it follows: the provided
transpose query and the reference queries call the selected implementations'
queries, and the reference
`blas_matmul` query is the largest of both reference hot preparations' queries
and the reference prepared product's on the layouts they give.

The queries read layouts (`BlasGridInfos`, `CKKSCtBounds`, `GGLWEInfos`), never
ciphertext data: the prepared product's query receives the prepared operands,
or, before preparing, the layouts `blas_matmul_left_operand_layout` and
`blas_matmul_right_operand_layout` give, and the context, under which it sizes
a hot side's terms. A query may panic on layouts its validator rejects, as its
`# Panics` section states.

Steps run one after another on one scratch, so a query reports the largest
single step. Temporaries that outlive a hook call live on the heap, not in
scratch: the copies of a hot preparation, the terms a preparation or a product
makes before converting them, the blocks a JKLS product by public matrices
stages through its first map, and the heap grid of a shift across blocks.

The reference products make every term once and convert it once, which fixes
the transform count at its minimum, and keep a term only while it serves. With
`T = K d` terms in a unit, a block row of `a` or a block column of `b`, of `La`
and `Lb` limbs, for an `m x K` by `K x n` product:

- two cold operands: no term;
- one hot side: one unit of its terms at a time, beside its `K` sigma (tau)
  results while the unit is made;
- two hot sides, and `blas_matmul`: one side is held, `a` when
  `(m - 1) La <= (n - 1) Lb`, the fewer limbs for the held terms plus one unit
  of the other side, and the other streams, a unit at a time; with `b` held,
  `a`'s first unit is made first, so the left terms' budget errors still lead.

The heap of `blas_matmul` beyond its inputs and `res` is then the held side's
terms plus one unit, whatever the extent of the streamed side; the held terms
are in the backend's transform-domain form, one to four times their coefficient
size depending on its transform. A cold operand holds `d` such terms per block
of its grid.

Under `JklsStored` and `JklsGalois`, a product by public matrices makes its
terms as a hot side's: a unit at a time, a block row of `a` on the right and
a block column on the left, each of its `K d` (left `K (2d - 1)`) terms
converted once under `OneShot` for every block of `res` the unit writes.
Each unit is allocated when made, its working blocks dropped as its terms are
converted, and the unit dropped after its blocks; nothing is reused across
units. Beside `res` and the operands the heap holds one unit of prepared
terms, in the backend's transform-domain form, and while the unit is made its
`K` first-map results and its unconverted terms, whatever the number of units.

## Parallelism

Concurrency is intra-ciphertext: it lives inside the backend's kernels, within
one ciphertext operation, and BLAS never splits work across ciphertexts. Each
reference runs its blocks, sources and kernel calls in order, on one scratch,
the first error stopping it, and its query reports the largest single step
([Scratch](#scratch)). A backend whose kernels already run in parallel writes
nothing for BLAS.

An override may choose its own level instead, never both: a backend whose
kernels are serial, a device queue for example, may run the blocks of an
operation concurrently, as long as its bits and its error are the reference's
and its query reserves what its own schedule uses. An implementation cannot add
bounds its trait lacks, so the traits carry the ones that choice needs: input
blocks and key stores are `Sync`, output blocks `Send`, the hook's sources
`Sync` and its destinations `Send`, the kernel's prepared pairs and key
`Sync`, and the prepared operands `Send + Sync`.

Calls with distinct `res` and `scratch` may run concurrently over a shared
module, context, public operands, prepared operands and key stores: BLAS spawns
no thread and holds no shared mutable state, and the types they share are
`Send` and `Sync`. An override keeps no state across calls, so disjoint
concurrent calls stay valid. The parity test `concurrency` runs a
`blas_matmul`, a `blas_matmul_prepared` of cold operands and a
`blas_matmul_pt_right`, each twice, at once over one module, context, key
store, public operand and pair of prepared operands, and requires each result
bit for bit equal to the same call run alone.

## Testing a backend

Enable `test-utils` on a `poulpy-blas` dev-dependency. It exports four
registration macros.

`blas_backend_test_suite!` registers the functional tests for one backend,
scalar, encoder and parameter set. They encrypt, run every operation, and check
the decrypted values and the output metadata against cleartext expectations,
along with the input guards, key widths, and the plan's key sets against the
encoded maps; each operation runs on the keys of its own set alone, and the
preparations and the prepared product under every hint and mix of hot and cold
operands. The transpose, the shifts through both half maps, the preparations,
the products and the products by public matrices, under `Diagonal` and `Jkls`,
also run with maps or public matrices a quarter of the parameter set's scale
below and above it, which they consume in place of their inputs' scale, so the
parameter set's `k` must hold two maps at the larger scale and a product. Hot
operands prepared under one context serve the products under all three, and a
hot operand beside a cold one prepared under another context gives the product
of their terms. Through those maps, the products also multiply factors at two
scales, in either order, one an eighth of the scale lower and either as much
narrower, so that their terms keep one budget, or as wide, so that its terms
have the larger budget: crossed scales, where the larger budget plus the larger
scale, from which the ct x ct rule takes its offset, exceeds the wider term's
width. The masked products by public matrices, `JklsStored` and `JklsGalois`,
run under a context holding their masks at the parameter set's scale, three
scales deep, on the right of one public matrix and on the left of one, two and
every public matrix per matrix slot, and their guards reject a right operand of
two public matrices on more than one matrix slot, a product on either side
under a context of masks encoded on a module of another degree, at another
radix, at another dimension or wider than its input, and a left product under a
context without masks. `test_blas_key_widths` runs with the maps and the public matrices a
quarter below that scale, where a key width taken from the inputs' scale in
place of `p` or `p_B`, or capped by it, finds no key. They read coefficients on
the host and use the standard ring, so they need a host backend of the standard
ring. An optional `log_d = …` sets the blocks to `2^log_d x 2^log_d`, 2 by
default. The ring degree `N` is free within two bounds: the blocks fit the
`N / 2` slots, `d^2 <= N / 2` (`N >= 32` at the default), and the guards encode
on a module of degree `2N`, which the backend must support (`N <= 2^16` at
most). `test_blas_batch_record`, below four matrix slots, and, in the guards of
the products by public matrices, the operand at another dimension, where only
`2 x 2` blocks fit, and the relabelled grid, on one matrix slot, are skipped
with a message. `test_blas_matmul_long_sums`, a `1 x 8` grid times an `8 x 1`
grid of distinct blocks, runs on its own ring, `N = 32`, so a registration
that sets `log_d` skips it with a message: its default registration runs it.

`blas_conjugate_invariant_test_suite!(mod m, backend = …, params = …)`
registers the test `conjugate_invariant` for a backend of the
conjugate-invariant ring, whose `N` slots are real. It encrypts real matrices
of `4 x 4` blocks with the module's own encoder, runs every operation once,
each on exactly the bytes its query reports, the products also by a right
factor whose compact encryption proves it holds one matrix and the products by
public matrices under every algorithm, the masked ones on the right of one
public matrix, and checks the decrypted values and the output metadata, as the
functional tests do, and that every output, the prepared operands included, has
`Real` slots. Below `N = 16`, whose slots hold no such block, it is skipped
with a message.

`blas_parity_test_suite!` compares `backend_test` with `backend_ref`, an
attested backend:

```rust
poulpy_blas::blas_parity_test_suite! {
    mod blas_parity_my_backend_f64,
    backend_ref = attested::Backend,
    backend_test = crate::MyBackend,
    scalar = f64,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 12, ..poulpy_ckks::test_suite::BASE19_PARAMS_F64 },
}
```

It registers the tests `inter`, `intra` and `matmul`, each on its own
pair of modules of degree `params.n`, and `concurrency`, on the tested
backend's module alone ([Parallelism](#parallelism)). Both backends
run on identical fixtures and keys, and the operations are called
through the API, which forwards to their `*Impl` methods. Every call,
encodings and preparations included, gets exactly the
bytes its query reports, poisoned and guarded. Every output is compared: its
batch record, and every block bit for bit, digits and CKKS metadata. Prepared
operands, made under either hint, are compared through their infos alone,
`(log_d, rows, cols, block_layout, mat_slots)`, which must also equal their
input's batch and, as the layout, their input's when hot and its term layout
when cold, beside their hint and their input's layout as their source; the
prepared product of every mix of hot and cold operands runs on one pair per
scale of the maps into `res` of three widths, and on pairs at two scales into
`res` of one, and must equal `blas_matmul` each time. Each call looks its keys
up in a store of its operation's set alone, from `BlasPlan` or
`sum_galois_elements`, which records every lookup: the tested backend may look
up only keys, at widths, that the reference looked up in the same call. No call
may write its inputs. The calls reach every branch of the circuit: identity
rotations, one matrix slot, the copies of row and column shifts by `d` and `2d`
along three blocks, one-row and one-column grids, `res` wider and narrower than
the result, factors one limb apart in both orders, lazy inputs on real slots,
sparse unless one matrix fills them, copies of lazy inputs, a right factor
holding one matrix, public products, under every algorithm, of one matrix and
of one per matrix slot, the masked ones on the right of one matrix only, under
a context holding their masks, maps and public matrices at scales below and
above the operands', factors at two scales, in either order, through those
maps, whose terms keep one budget or give the lower scale the larger budget
(crossed scales, where a kernel that takes its offset from the wider term's
width changes the bits), and a long product, `1 x 32` by `32 x 1` distinct
blocks (`1 x 16` by `16 x 1` above radix 52), at every registration, which a
kernel that drops or repeats a term of a long sum fails. Each family's parity also
runs calls that poulpy-ckks rejects for their budget, among them the prepared
product of a hot operand one bit short of the `2p` its terms spend, the product
by a right factor of budget `3p - 1` holding distinct matrices, which one
holding one matrix passes, and the JKLS products by public matrices one bit
short of the first map's scale, then of both maps': both backends return the
same error and leave `res` and the inputs unchanged. Storage and ring are free:
the bounds are `BlasParityBackend` (`Backend<ZnxWord = i64>` and every BLAS
family) and `BlasParityModule`. Rank and `dsize` follow `params`, as in CKKS
parity.

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

`blas_max_base2k_test_suite!` registers the test `max_base2k` for one backend,
scalar and parameter set: `blas_matmul_max_base2k`,
`blas_matmul_pt_right_max_base2k` and `blas_matmul_pt_left_max_base2k` on
`4 x 4` blocks on 32 slots (`n = 64`, or `n = 32` on the conjugate-invariant
ring), under every algorithm, against counts made by hand, at every failure
target from 1 to 1000 bits. It needs a backend that implements `MaxBase2k`.

The registrations of this workspace, and the commands and CI coverage, are in
the [repository README](../../../README.md#testing-a-backend).

## Attestation

Parity is bit-exact, so an override must compute the reference's circuit. An
algorithm that is only numerically close fails it.

An override ships with its parity test passing against an attested backend.
Attestation is transitive ([CONTRIBUTING](../../../CONTRIBUTING.md#coding-conventions)):
a backend that passes parity against an attested one can serve as the reference
for the next, for the same operations and parameter ranges. The repository
README lists the attested backends a crate outside the workspace can compare
against.

## Backends outside the workspace

Everything above applies to a backend crate outside this workspace,
closed-source ones included. It defines its own backend type, implements the
HAL, core and CKKS families for it, and may then take any BLAS family from its
macro or override it. Three Rust rules constrain its wiring:

- **It wires only its own types.** Implementing a BLAS trait for another
  crate's type, a backend of this workspace included, breaks the orphan rule
  (E0117).
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
