# Implementing BLAS operations

Public matrix calls follow `api → delegates → oep`. `Module<BE>` implements
`BlasMatInterOps`, `BlasMatIntraOps` and `BlasMatMulOps` for every backend that
implements the matching `*Impl` trait: each call forwards to the backend's
method without a check, and the method runs its validator first. The BLAS
crate imports no backend crate: each backend crate wires its own types.

This file holds the rules every family shares. What an implementation of one
family must reproduce, and what it may change, is the `# Safety` section of
that family's trait; the output metadata and the errors of each operation are
on its API method. Parity is bit-exact, so an override computes the
reference's circuit: anything a `# Safety` section does not list as free
changes bits.

## Families

| OEP family | Required | Provided |
|---|---|---|
| `BlasMatInterImpl` | `blas_rotate`, `blas_sum` | |
| `BlasMatIntraImpl` | the map hook `blas_map_blocks_impl` | `blas_transpose`, `blas_rotate_rows`, `blas_rotate_cols`, over the hook |
| `BlasMatMulImpl` | `blas_matmul_prepare_left`, `blas_matmul_prepare_right`, `blas_matmul_prepared`, `blas_matmul_pt_right`, `blas_matmul_pt_left` | `blas_matmul`, over both hot preparations and the prepared product |
| `BlasJklsImpl` | the JKLS kernel `blas_jkls_block_impl`, one output block of the prepared product | |

Each operation's `_tmp_bytes` query belongs to the same family. `BlasImpl`
aggregates the four families as a test bound; it has a blanket implementation
and is never implemented directly.

The provided methods call the backend's selected methods: the transpose and
both shifts call its map hook, and `blas_matmul` its preparations and prepared
product. So an override of the hook or of the kernel reaches every operation
built on it. The provided bodies are public in `poulpy_blas::oep::derived`, as
`blas_transpose_derived`, `blas_rotate_rows_derived`,
`blas_rotate_cols_derived` and `blas_matmul_derived` with their
`_tmp_bytes_derived` queries, so an override of a provided method can fall
back to the provided body, for both the call and its query.

The prepared operands of the products, `BlasMatMulLeft<BE>` and
`BlasMatMulRight<BE>`, are `poulpy_blas::layouts::BlasMatMulOperand`s of the
backend's poulpy-ckks storage, a type the backend does not choose.
`BlasMatMulOperandPreparedFactory` allocates them; an implementation of the
preparations and of the prepared product reads and fills their storage through
`blocks`, `terms` and their `_mut` forms, and records a preparation with
`set_prepared`.

## Reference implementations

A **reference** implementation defines a matrix operation by composing CKKS
and core operations on the module. The reference traits are implemented for
every `Module<BE>` and are public, so an override can forward to them. They
call `poulpy-ckks` and `poulpy-core` through their public APIs, so a backend's
CKKS and core overrides remain effective inside them.

| OEP family | Reference | Macro |
|---|---|---|
| `BlasMatInterImpl` | `BlasMatInterReference`, [`reference/inter.rs`](../src/reference/inter.rs) | `impl_blas_inter_reference!` |
| `BlasMatIntraImpl` | `BlasMatIntraReference` (the hook), [`reference/intra.rs`](../src/reference/intra.rs) | `impl_blas_intra_reference!` |
| `BlasMatMulImpl` | `BlasMatMulReference`, [`reference/matmul.rs`](../src/reference/matmul.rs) | `impl_blas_matmul_reference!` |
| `BlasJklsImpl` | `BlasMatMulReference::blas_jkls_block_reference`, [`reference/matmul.rs`](../src/reference/matmul.rs) | `impl_blas_jkls_reference!` |

`impl_blas_reference_full!` expands to the four family macros. The reference's
cold preparations call the backend's map hook, and the reference prepared
product, like the reference `blas_matmul`, calls it for the terms of a hot
operand, then its JKLS kernel. So `impl_blas_matmul_reference!` needs the intra
and JKLS families. It overrides the provided `blas_matmul` with
`BlasMatMulReference::blas_matmul_reference`, the same product on the grids
read in place, without the preparations' copies, under the provided query.

## Grids and keys

`res` is a `BlasGridMut<BE>` and every ciphertext input a `BlasGrid<BE>`,
which expose the blocks of a grid as a row-major slice. The prepared product
reads prepared operands instead, and the products by public matrices also read
a `BlasPtOperand`. A grid's blocks number exactly `rows x cols`, at least one,
and every operation rejects a grid that breaks this. `BlasCiphertext` over
owned storage implements both traits on every backend. Borrowed views are not
grids: a `BufRef` view is `Sync`, but its blocks are not `GLWEToBackendRef`.

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
| `blas_rotate` by `r` | `rotate_galois_elements(order, r)` | `a.k` |
| `blas_sum` | `sum_galois_elements(order, a)` | `min(res.k, a.k)` |
| `blas_transpose` | `transpose_galois_elements(order)` | baby steps at `a.k`, giant steps at `min(res.k, a.k - p)` |
| `blas_rotate_rows`, `blas_rotate_cols` by `k` | `rotate_rows_galois_elements(order, k)`, `rotate_cols_galois_elements(order, k)` | as the transpose; none for a shift by a multiple of `d`, a copy |
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
by `k d`, `Direct`, one of psi^k's rotations. `galois_elements(order)` covers
every map of a context, so every operation's set but `blas_rotate`'s and
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

Each operation has a query in `poulpy_blas::reference` that lists the keys it
looks up: `blas_matmul_key_uses`, `blas_matmul_prepare_left_key_uses`,
`blas_matmul_prepare_right_key_uses`, `blas_matmul_prepared_key_uses`,
`blas_matmul_pt_right_key_uses`, `blas_matmul_pt_left_key_uses`,
`blas_transpose_key_uses`, `blas_rotate_rows_key_uses`,
`blas_rotate_cols_key_uses`, `blas_rotate_key_uses` and `blas_sum_key_uses`. A
query takes the grids, operands and context of the operation's `_tmp_bytes`
query, without the key layouts. It leaves out what the keys do not depend on:
the result grid of a product of two encrypted grids, and the `PrepareHint` of a
preparation, which looks up the same keys under either. A rotation and a shift
take the amount `r` or `k`, which selects their maps. A query returns a
`BlasKeyUses`:

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
`blas_rotate_output_layout`, `blas_sum_output_layout`) take the input layouts
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
   `res`; a context or public operand encoded for another slot count or radix;
   public matrices that do not divide the batch; under `JklsStored` and
   `JklsGalois`, more than one public matrix on the right, and a context at
   another `log_d`, encoded for another slot count or radix, wider than the
   input or, on the left, without their masks; a prepared operand without a
   batch record, or a hot one at another `log_d` than the product's context
   or under a context encoded for another slot count or radix; and a product
   at a radix above 62, whose digits cannot hold two lazy terms of the JKLS
   kernel's two-pass circuit, or a sum at a radix whose digits cannot hold the
   `log2(mat_slots) + 1` lazy terms it adds. A hot operand's scale is not
   checked: the product spends that of its context's maps.
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

## Scratch

Every method runs within the scratch its own `_tmp_bytes_impl` reports for the
layouts and keys it receives, which may differ from the reference's; the
reference asserts its bound on entry and panics before any write when it falls
short, for a call that looks up a key. An override that needs other scratch
reports it in its own query, and every operation over it follows: the provided
queries call the selected implementations' queries, the provided `blas_matmul`
the largest of both hot preparations' queries and the prepared product's on
the layouts they give.

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
concurrent calls stay valid.
