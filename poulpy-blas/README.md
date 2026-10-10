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

Backend crates wire poulpy-blas behind their own `enable-blas` feature, selecting the reference compositions with
`poulpy_blas::impl_blas_reference_full!(Backend)`.
