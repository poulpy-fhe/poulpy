# poulpy-bin-fhe

Binary and gate-level FHE built on `poulpy-core` and `poulpy-hal`: blind
rotation, circuit bootstrapping, and BDD-based encrypted integer arithmetic.
The crate depends on no concrete backend.

## User-facing API

Import operations from `poulpy_bin_fhe::api::blind_rotation`,
`poulpy_bin_fhe::api::circuit_bootstrapping`, or `poulpy_bin_fhe::api::bdd`.
All traits are also re-exported by `poulpy_bin_fhe::api` and the crate root.
Every operation follows `API -> delegate -> OEP`, including LUT construction
and compressed-key allocation. Backends explicitly select their implementations.

- [Blind rotation](src/api/blind_rotation.rs) exposes LUT evaluation, standard
  and compressed key encryption, decompression, preparation, and LUT helpers.
- [Circuit bootstrapping](src/api/circuit_bootstrapping.rs) exposes constant
  and exponent execution, reusable plans, key encryption, and preparation.

Each execution trait includes its scratch queries. Use the selected module's
query for the actual inputs and output before calling the matching operation.
The API module documentation includes generic examples; existing key convenience
methods and import paths remain available.

## Backend integration

Public traits in `api` delegate to explicitly implemented `oep::*Impl`
contracts. Algorithms built from core/HAL operations live in `reference`;
simple compositions of binary-FHE operations are crate-private derived defaults.
The CGGI algorithm marker is independent of backend and scheduling selection.

Implement the required operation families in the backend crate, or select the
provided implementations with `impl_bin_fhe_reference_full!(MyBackend)`.
A backend can instead select individual families and replace an operation,
including its scratch query. Delegates add no reference implementation bounds.

See [operation contracts and migration](docs/bin-fhe-contracts.md) for the
family map, scratch rules, host boundaries, and parity requirements. Concrete
build commands and runnable examples belong in the [repository README](../README.md)
and backend crates.

## Validation

Register `bin_fhe_parity_test_suite!` beside each backend's scheme opt-in,
passing both `backend_ref` and `backend_test`. Either can be a previously
validated implementation; no fixed comparison backend is required. Shared
coefficient inputs and raw keys are transferred identically, prepared
independently, and checked for exact coefficient and metadata parity. Tests
use each implementation's advertised scratch with poisoned guard regions.

The separate `bin_fhe_backend_test_suite!` checks mathematical correctness and
complete encrypted arithmetic workflows. Host fixture requirements of that
suite do not constrain the operation contracts or paired execution interface.

`bin_fhe_reference_test_suite!` separately checks the callable key-encryption
references against selected operations using one backend's sampler on both
paths. It requires the reference bodies' lower-layer capabilities; replacement
backends can use the paired suite without these extra requirements.
