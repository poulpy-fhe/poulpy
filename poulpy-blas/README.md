# poulpy-blas

Encrypted matrix arithmetic on [poulpy-ckks](https://github.com/poulpy-fhe/poulpy).

A matrix is a `rows x cols` grid of `d x d` blocks, one CKKS ciphertext per block. A ciphertext with `S` slots holds
`S / d^2` row-major blocks, one per matrix slot, so every operation acts on a batch of matrices at once. The grid records
its batch, up to `S / d^2` matrices repeated over the slots, from its blocks' label (see `BlasCiphertext::from_blocks`).
`layouts::grid::{split_into_blocks, join_blocks}` move matrices, read and written through element accessors, to and
from the slot vectors of the blocks.

Intra operations apply one map to every matrix of the batch: `blas_transpose`, `blas_rotate_rows`, `blas_rotate_cols`.
Inter operations move whole matrices across the batch: `blas_rotate`, `blas_sum`. Products multiply every matrix of the
batch: `blas_matmul` (JKLS, ciphertext x ciphertext), `blas_matmul_prepare_left`, `blas_matmul_prepare_right` and
`blas_matmul_prepared` (the same product from prepared operands, bit for bit under the same context), `blas_matmul_pt_right`
(ciphertext x public matrices, shared or one per matrix slot) and `blas_matmul_pt_left` (public matrices x ciphertext).
A public operand is encoded from a plan whose `BlasPtAlgorithm` picks the diagonal products (one map per block, one
scale, the default) or the JKLS ones (sigma or tau of every block of the encrypted grid, once, then one map per block,
two scales); both look up the same rotation keys. Against the diagonal products, JKLS costs a second scale, one more map
per block of the encrypted grid and that map's noise, and the operand holds that map; for one public matrix on the
right, its block maps take up to `d` times the memory. A left operand of one public matrix on more than one matrix
slot holds a second set of block maps for a replicated input, `(3d - 1) d^2` slot values per block instead of
`(2d - 1) d^2`, which the product takes when its input's label proves one matrix.
A preparation takes a `PrepareHint`: hot, `OneShot`, for one product, a relabel of its input whose terms the product
evaluates under its own context; cold, `Reuse`, for many, every term under the preparation's context, converted once to
the transform domain. `blas_matmul` multiplies two hot operands. The reference products make a hot side's terms once, a
block row or column at a time; of two hot sides they hold one side's terms and stream the other's, so the heap of
`blas_matmul` does not grow with the streamed side. A right factor whose label proves it holds one matrix, repeated over
the slots, takes its psi^k as rotations at no scale, so it may arrive one map scale narrower than the left factor:
encrypt it from a plaintext of `d^2` slots whose degree reaches the backend's `MIN_DEGREE`, since a degree-`N` one at
label 0 proves one matrix only when one fills the slots.

`reference::{blas_matmul_max_base2k, blas_matmul_pt_right_max_base2k, blas_matmul_pt_left_max_base2k}` return the
largest `base2k` that `Module::max_base2k` allows for every lazy sum of a product, from the grids' layouts, the context
and the public operand alone.

Calls with distinct `res` and `scratch` may run concurrently over a shared module, context, public operands, prepared
operands and key stores: BLAS spawns no thread and holds no shared mutable state, its types are `Send` and `Sync`, and an
override keeps no state across calls ([Parallelism](docs/blas-contracts.md#parallelism)).

A `BlasPlan` fixes the block dimension and how each block map is evaluated. The maps are encoded once from it, at a
caller-chosen precision, into a `BlasContext`, and public matrices into a `BlasPtOperand`. Operations take the context
and `Sync` key stores: rotation keys through poulpy-core's `GetAutomorphismKey` (`atk`), the tensor key through
`GetTensorKey` (`tsk`). The preparations take only `atk`, and `blas_matmul` and `blas_matmul_prepared` both: the latter
looks up the rotation keys of a hot operand's terms, and of two cold operands the tensor key alone. Every operation
takes any grid implementing `BlasGrid` (`BlasGridMut` for the result). A prepared operand, a `BlasMatMulOperand`, holds
poulpy-ckks ciphertexts or prepared terms in the backend's storage, and reports its hint and its input's dimensions,
batch and layout (`BlasMatMulOperandInfos`). Slot-wise additions, products and rescaling are poulpy-ckks operations
applied to each block.

The plan lists the rotation keys each operation looks up, from the ring's cyclotomic order alone:
`transpose_galois_elements`, `rotate_rows_galois_elements(k)`, `rotate_cols_galois_elements(k)`,
`rotate_galois_elements(r)`, `matmul_galois_elements` (the products and both preparations) and
`matmul_pt_right_galois_elements` and `matmul_pt_left_galois_elements` (any public matrices, on the right or on the left);
`galois_elements` covers every map of a context, and
`sum_galois_elements` lists `blas_sum`'s for a grid's batch. A client therefore plans its keys with no module, context
or public matrix: it generates the union of its pipeline's sets, and a tensor key only if it calls `blas_matmul` or
`blas_matmul_prepared`. The server checks each grid's batch, `mat_slots()`, against the one the summed keys were
planned for.

Each operation also has a key-use query in `poulpy_blas::reference` (`blas_matmul_key_uses`, ...) that lists each key
it looks up and the width it looks it up at: `(Galois element, width)` pairs and the tensor key's width. The output-layout
queries (`blas_matmul_output_layout`, ...) give the layout each operation stamps on `res` from its inputs' layouts, so a
client plans every intermediate grid, and with it every key, before it encrypts or runs anything. Every query takes a
context or a public operand, or their layouts (`BlasContextLayout`, `BlasPtOperandLayout`), and reads only the shape of
their maps, so the plan needs no map encoded; the radix queries name their backend, `blas_matmul_max_base2k::<BE>`.

## Lifecycle

poulpy-blas follows poulpy-core's layouts: a standard form, which a client builds and a server stores, and a prepared
form, the backend's representation, which operations read.

| Standard | Prepared | Layout and infos |
|---|---|---|
| `BlasCiphertext` | none | `BlasLayout`, `BlasGridInfos` |
| `BlasPtOperandUnprepared<BE, S>` | `BlasPtOperand<BE, S>` | `BlasPtOperandLayout<S>`, `BlasPtOperandInfos` |
| `BlasContextUnprepared<BE>` | `BlasContext<BE>` | `BlasContextLayout`, `BlasContextInfos` |
| none | `BlasMatMulLeft<BE>`, `BlasMatMulRight<BE>` | `BlasMatMulOperandLayout`, `BlasMatMulOperandInfos` |

Each is allocated on a module from its layout (`BlasModuleAlloc`, `BlasPtOperandPreparedFactory`,
`BlasContextPreparedFactory`), sized by the matching `bytes_of` (`BlasBytesOf` and the factories), and filled in place:
encoded into (`BlasEncodingOps`), then prepared into. A public operand takes its matrices at encoding, and its side is
its type parameter `S`, `Right` or `Left`. Encoding and preparing again into the same allocations refills them, with no
reallocation of their storage:

```rust,ignore
let mut ctx = module.blas_context_alloc(plan, base2k, coeffs_meta);
let mut prepared = module.blas_context_prepared_alloc_from_infos(&ctx);
let mut scratch = ScratchOwned::<BE>::alloc(
    module.blas_context_encode_tmp_bytes::<f64, _>(&ctx).max(module.blas_context_prepare_tmp_bytes(&ctx)),
);
module.blas_context_encode_into::<f64>(&mut ctx, &mut scratch.borrow())?;
module.blas_context_prepare(&mut prepared, &ctx, &mut scratch.borrow());

let mut b = module.blas_pt_operand_alloc::<Right>(plan, rows, cols, count, base2k, coeffs_meta);
module.blas_pt_operand_encode_into(&mut b, |m, row, col| entry(m, row, col), &mut scratch.borrow())?;
```

An allocation panics on a layout whose blocks do not fit the module's slots, whose count of public matrices does not
divide the matrix slots, whose maps have no torus width, or whose `base2k` is outside `1..=63`.

A prepared operand of a product is allocated from the infos `blas_matmul_left_operand_layout`
(`blas_matmul_right_operand_layout`) gives for a grid, a context and a hint, by `blas_matmul_left_alloc_from_infos`
(`blas_matmul_right_alloc_from_infos`), sized by `blas_matmul_left_bytes_of_from_infos` (`_right_`), and filled by
`blas_matmul_prepare_left` (`_right`), under that hint.

`BlasCiphertext` implements `WriterTo` and `ReaderFrom`, little-endian, with no magic or version bytes: a header of its
dimensions and its blocks' `n`, `base2k`, `k` and rank, then its blocks, row-major. A read fills a grid allocated from
the writer's layout and rejects another layout with `InvalidData` before touching it; `write_to` rejects a grid holding
a flag-clear block with `InvalidInput`, writing nothing. Public operands and contexts are not serialized: encode them
again, then prepare them.

## Tests and backend integration

poulpy-blas imports no backend. Backend crates select the reference compositions with
`poulpy_blas::impl_blas_reference_full!(Backend)`. To override a family, a backend wires the other families with
`impl_blas_inter_reference!`, `impl_blas_intra_reference!`, `impl_blas_matmul_reference!` or `impl_blas_jkls_reference!`
and implements that family's `oep` trait by hand, forwarding the methods it keeps to `reference`. A backend crate
outside this workspace does the same for its own types. It takes `poulpy-hal`, `poulpy-core`, `poulpy-ckks` and
`poulpy-blas` from one poulpy source, as direct dependencies under those names, because the macros expand to
`::poulpy_hal`, `::poulpy_core` and `::poulpy_ckks`. Like the reference, an implementation runs its operation's
validator, the public `layouts::validate_*`, before any write and stamps the batch record of `res` after `Ok`: the API
dispatches without a check. A backend does not choose the type of the prepared operands, `BlasMatMulOperand`: it
supplies their storage through its `Backend`, as poulpy-hal's prepared layouts do, and holds its terms in its own
representation through its poulpy-ckks preparations.

```sh
cargo test -p poulpy-blas
```

See the [repository README](../README.md) for commands and CI coverage, and
[Implementing BLAS operations](docs/blas-contracts.md) for the override contract.
