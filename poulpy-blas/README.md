# poulpy-blas

Encrypted matrix arithmetic on [poulpy-ckks](https://github.com/poulpy-fhe/poulpy).

A matrix is a `rows x cols` grid of `d x d` blocks, one CKKS ciphertext per block. A ciphertext with `S` slots holds
`S / d^2` row-major blocks, one per matrix of a batch, so every operation acts on that many matrices at once.
`layouts::grid::{split_into_blocks, join_blocks}` move matrices, read and written through element accessors, to and
from the slot vectors of the blocks.

Intra operations apply one map to every matrix of the batch: `blas_matmul` (JKLS, ciphertext x ciphertext),
`blas_matmul_prepare_left`, `blas_matmul_prepare_right` and `blas_matmul_prepared` (the same product from transformed
operands reused across calls), `blas_matmul_pt` (ciphertext x public matrices, shared or one per matrix slot),
`blas_transpose`, `blas_rotate_rows`, `blas_rotate_cols`. Inter operations move whole matrices across the batch:
`blas_rotate`, `blas_sum`.

The block maps are encoded once, at a caller-chosen precision, into a `BlasContext`, and public matrices into a
`BlasPtOperand`. Operations take the context and any key store implementing poulpy-core's `GetAutomorphismKey` (and
`GetTensorKey` for the ciphertext x ciphertext products). Slot-wise additions, products and rescaling are poulpy-ckks
operations applied to each block.

## Tests and backend integration

poulpy-blas imports no backend. Backend crates select the reference compositions with
`poulpy_blas::impl_blas_reference_full!(Backend)`. To override a family, a backend wires the other families with
`impl_blas_inter_reference!`, `impl_blas_intra_reference!` or `impl_blas_matmul_reference!` and implements that
family's `oep` trait by hand, forwarding the methods it keeps to `reference`. `poulpy-cpu-oracle` and the CPU backends
do so behind their `enable-blas` feature. A backend crate outside this workspace does the same for its own types. It
takes `poulpy-hal`, `poulpy-core`, `poulpy-ckks` and `poulpy-blas` from one poulpy source, as direct dependencies
under those names, because the macros expand to `::poulpy_hal`, `::poulpy_core` and `::poulpy_ckks`.

```sh
cargo test -p poulpy-blas
```

The `test-utils` feature exposes the functional suite, registered with `blas_backend_test_suite!` and run by
`poulpy-cpu-oracle`, and the parity suite, registered with `blas_parity_test_suite!`. Parity compares a tested backend
with a reference backend bit for bit, at every block dimension that the optional `shapes` argument
(`test_suite::parity::BlasParityShapes`, `log_d` 1 and 2 by default) selects. Every override ships with that parity
test passing against an attested backend: the oracle inside this workspace; outside it, either the oracle taken from
the same git revision as every other poulpy crate, or the published `poulpy-cpu-portable` with `enable-blas`.

See the [repository README](../README.md) for commands and CI coverage, and
[Implementing BLAS operations](docs/blas-contracts.md) for wiring, registration and attestation.
