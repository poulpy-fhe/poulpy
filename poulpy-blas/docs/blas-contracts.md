# Implementing BLAS operations

Public matrix calls follow `api → delegates → oep`. `Module<BE>` implements
`BlasMatInterOps`, `BlasMatIntraOps` and `BlasMatMulOps` for every backend that
implements the matching `*Impl` trait, and forwards each call to it. The BLAS
crate imports no backend crate: each backend crate wires its own types.

## Reference implementations

A **reference** implementation defines a matrix operation by composing CKKS
and core operations on the module. The reference traits are implemented for
every `Module<BE>` and are public, so an override can forward to them. They
call `poulpy-ckks` and `poulpy-core` through their public APIs, so a backend's
CKKS and core overrides remain effective inside them.

| OEP family | Operations | Reference | Macro |
|---|---|---|---|
| `BlasMatInterImpl` | `blas_rotate`, `blas_sum` | `BlasMatInterReference`, [`reference/inter.rs`](../src/reference/inter.rs) | `impl_blas_inter_reference!` |
| `BlasMatIntraImpl` | `blas_transpose`, `blas_rotate_rows`, `blas_rotate_cols`, provided over the map hook `blas_map_blocks` | `BlasMatIntraReference` (the hook), [`reference/intra.rs`](../src/reference/intra.rs) | `impl_blas_intra_reference!` |
| `BlasMatMulImpl` | `blas_matmul_prepare_left`, `blas_matmul_prepare_right`, `blas_matmul_prepared`, `blas_matmul_pt`, and `blas_matmul` provided over the preparations and the prepared product | `BlasMatMulReference`, [`reference/matmul.rs`](../src/reference/matmul.rs) | `impl_blas_matmul_reference!` |

Each operation's `_tmp_bytes` query belongs to the same family.
`impl_blas_reference_full!` expands to the three family macros. `BlasImpl`
aggregates the three families as a test bound; it has a blanket implementation
and is never implemented directly.

The API checks every operation before it reaches the implementation, from
metadata alone, with the `validate_*` functions of `poulpy_blas::reference`:
shapes, layouts, batch records, the context and every key at its width. A
rejected operand returns `CKKSCompositionError::InvalidOperand`, a missing key
`MissingAutomorphismKey` or `MissingRelinearizationKey`, and none changes
`res`. An implementation therefore runs only on accepted inputs and returns
`Ok` on them, except for the reference's own errors:

- poulpy-ckks's budget error, which the reference raises before any write;
- until the terms of a prepared operand share one width, two errors of the
  matrix products. The prepared product's validator reads the operands' infos
  alone, which report the widest term, so the reference checks the tensor key
  at each narrower term's width before any write and returns
  `MissingRelinearizationKey` for one `tsk` lacks. And a budget error that only
  the products of the narrower terms meet comes after the first terms are
  written: `res` is then partly written.

## Grids and keys

Every operation takes `res` as a `BlasGridMut<BE>` and its inputs as
`BlasGrid<BE>`, which expose the blocks of a grid as a row-major slice. Input
blocks are `Sync`, output blocks `Send`, and the key stores are `Sync`, so an
implementation may spread blocks over the backend's task executor. Rotation
keys come as `atk` and the tensor key as `tsk`, two separate stores: the
rotations, the maps, the preparations and the product by public matrices take
`atk` alone, the prepared product `tsk` alone, and `blas_matmul` both. A grid's
blocks number exactly `rows x cols`, at least one, and every operation rejects
a grid that breaks this.
`BlasCiphertext` over owned storage implements both traits on every backend.
Borrowed views are not grids: a `BufRef` view is `Sync`, but its blocks are not
`GLWEToBackendRef`.

## Batches

A grid records its batch, `BlasGridInfos::mat_slots`: the number of matrices
its blocks hold. `BlasCiphertext::from_blocks` records the batch the blocks'
label proves, and a grid without a record (`mat_slots()` is `None`) is read at
that batch. Every operation rejects a grid whose label does not allow its
batch. The API stamps the record of `res` once the implementation returns
`Ok`: products record the larger batch of their operands, every other operation
its input's. An implementation writes blocks only, and reads an input's batch
with `poulpy_blas::layouts::grid_batch`, as the references do: its record, or
for a grid without one the batch its label proves. A preparation records that
batch on its operand.

Encryption labels a block from its plaintext's degree, so a batch of `count`
matrices comes from compact plaintexts of `count d^2` slots. Compact degrees
are raised to the backend's `MIN_DEGREE` (8 by default, 16 on AVX-512 FFT64),
and a raised degree records a larger batch. A compact batch therefore records
`count` on every backend only when its degree, `2 count d^2` on the standard
ring and `count d^2` on the conjugate-invariant one, reaches every target's
`MIN_DEGREE`; otherwise encrypt it dense.

## Keys

Callers generate rotation keys from the per-operation sets of `BlasPlan`
(`transpose_galois_elements`, `rotate_rows_galois_elements`, …) and from
`sum_galois_elements`, which list the rotations of the reference's maps as the
plan schedules them: shift maps `Direct`, the others at the cost-optimal BSGS
giant step. An implementation looks up no rotation key outside its operation's
set, so a store holding that set alone suffices.

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
the macros of the families it keeps, and write the replaced family's `*Impl`
implementation directly, forwarding the unchanged methods to their reference
routines. The other families retain their own wiring. For example, a backend
that replaces matrix products keeps:

```rust
::poulpy_blas::impl_blas_inter_reference!(MyBackend);
::poulpy_blas::impl_blas_intra_reference!(MyBackend);
```

and implements `poulpy_blas::oep::BlasMatMulImpl` for `MyBackend` by hand. The
expansion of `impl_blas_matmul_reference!` in [`oep/matmul.rs`](../src/oep/matmul.rs)
spells out every signature and its forwarding call.

The prepared operands are the family's associated types `MatMulLeft` and
`MatMulRight` (`BlasMatMulLeft<BE>`, `BlasMatMulRight<BE>`), in any
representation the backend chooses (device memory, contiguous or DFT-domain
terms), and only `blas_matmul_prepare_left` and `blas_matmul_prepare_right` make
one. Through `BlasGridInfos`, an operand prepared from `a` reports `a`'s
dimensions, the layout `poulpy_blas::reference::blas_matmul_term_layout(a, ctx)`
gives, and `Some` of `a`'s batch, `poulpy_blas::layouts::grid_batch(a)`, which
also covers an input grid without a record: the prepared product's validator,
the batch the API records and the provided `blas_matmul` query read nothing
else, and parity compares these infos, never the storage. The reference's
operand, `BlasMatMulOperand<BE, Left>` (`Right`), holds the `d` terms of every
block the product multiplies, which `term(row, col, k)` returns. A backend with
its own layout sets both types, implements the preparations and the prepared
product, and forwards `blas_matmul_pt` to `BlasMatMulReference`; `blas_matmul`
is provided as both preparations then the prepared product, so it reaches the
backend's methods.

A family macro and a handwritten implementation of the same trait are mutually
exclusive. The family traits are `unsafe`; their `# Safety` sections state what
an implementation must reproduce.

`BlasMatIntraImpl` requires only the map hook, `blas_map_blocks_impl`, and its
query; the transpose and both shifts are provided over it, and over
poulpy-ckks's copy and addition. A backend that replaces the hook keeps the
other two families' macros and writes those two methods, as the expansion of
`impl_blas_intra_reference!` in [`oep/intra.rs`](../src/oep/intra.rs) does:

```rust
::poulpy_blas::impl_blas_inter_reference!(MyBackend);
::poulpy_blas::impl_blas_matmul_reference!(MyBackend);
unsafe impl poulpy_blas::oep::BlasMatIntraImpl for MyBackend { /* the hook and its query */ }
```

The hook writes block `t * lts.len() + m` of `dst` with map `m` of source `t`,
stamped with that source's label, in any order; the maps of one source may
share its baby-step cache. The transpose is one hook call. A shift along one
block row or column is one call with psi^k' (phi^k'). A shift across blocks is
one call with both half maps over every block of `a`, into a heap grid of
`2 m n` blocks at `res`'s block layout, then `m n` additions into `res`: the
provided shifts hold that whole grid, and a backend that needs less memory
overrides them to chunk it.

The rotation, the sum and the hook split their blocks over up to four workers
of the backend's `TaskExecutor`, each on its own scratch slice (and, in the
hook, its own baby-step cache), so a Rayon backend writes nothing. Their
queries reserve one slice per worker on a parallel executor and keep the
serial size otherwise. On failure they return the error of the lowest block, as
a serial loop would. Copies and additions run serially.

## Testing a backend

Enable `test-utils` on a `poulpy-blas` dev-dependency. It exports two
registration macros.

`blas_backend_test_suite!` registers the functional tests for one backend,
scalar, encoder and parameter set. They encrypt, run every operation, and check
the decrypted values and the output metadata against cleartext expectations,
along with the input guards, key widths, and the plan's key sets against the
encoded maps; each operation runs on the keys of its own set alone. They read
coefficients on the host and use the standard ring, so they need a host backend
of the standard ring. An optional `log_d = …` sets the blocks to
`2^log_d x 2^log_d`, 2 by default. `poulpy-cpu-oracle` runs them for
`FFT64Oracle` and `NTT4x30Oracle`: at `log_d` 1, 2 and 3 (the last at `N = 128`,
where one block fills a ciphertext), in `f64` and `Quad`, and at rank 2.

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
batch; the prepared product runs twice on one pair and must equal
`blas_matmul` both times. Inter and
intra parity also run calls that poulpy-ckks rejects for their budget: both
backends return the same error and leave `res` and the inputs unchanged.
Storage and ring are free: the bounds are `BlasParityBackend`
(`Backend<ZnxWord = i64>` and every BLAS family) and `BlasParityModule`. Rank
and `dsize` follow `params`, as in CKKS parity.

The optional `shapes` argument, a `BlasParityShapes`, sets the swept block
dimensions `log_d` (default `[1, 2]`). A `log_d` below 1, or whose `d^2` reaches
the slot count, is skipped with a message, and the test fails when none remains.
A backend that lacks a block dimension restricts the sweep instead of dropping
the suite:

```rust
    shapes = poulpy_blas::test_suite::parity::BlasParityShapes { log_d: vec![2] },
```

`log_d` is the only shapes axis: rank and `dsize` come from `params`, and one
macro registers all three families, so none can be dropped silently.

In this workspace the oracle runs the functional tests and parity between its
FFT64 and NTT4x30 families, standard and conjugate-invariant, in both
directions. Every other backend type registers parity against
the oracle of its family: `NTT3x42` types against `NTT4x30Oracle`,
conjugate-invariant types against the conjugate-invariant oracles, and Rayon
types against the oracle rather than their serial sibling.
[`blas_tests.rs`](../../poulpy-cpu-portable/src/tests/blas_tests.rs) in
`poulpy-cpu-portable` shows the registrations; commands and CI coverage belong
in the [repository README](../../README.md#testing-a-backend).

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
closed-source ones included. It may override any family, and three Rust rules
constrain its wiring:

- **It wires only its own types.** Implementing a BLAS trait for another
  crate's type breaks the orphan rule (E0117). The CPU backends wire theirs
  behind `enable-blas`.
- **One source.** `poulpy-hal`, `poulpy-core`, `poulpy-ckks`, `poulpy-blas` and
  the comparison backend come from one source and revision. Two copies of a
  crate define distinct types and traits, so a crates.io copy beside a git one
  leaves the module without the expected methods (E0599).
- **Direct dependency names.** The wiring and registration macros expand to
  `::poulpy_hal`, `::poulpy_core` and `::poulpy_ckks`, so the crate that invokes
  them depends on `poulpy-hal`, `poulpy-core` and `poulpy-ckks` directly, under
  those names (E0433 otherwise).
