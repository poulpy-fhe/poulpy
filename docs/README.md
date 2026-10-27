# 🐙 Poulpy documentation

The [`guides/`](guides) folder holds the guides and design notes, [`spec/`](spec) the implementation specifications, and [`img/`](img) the diagrams. The contract documents of each crate live in its own `docs/` folder.

- [Getting Started](guides/getting-started.md) — a map of the codebase: what each crate contains and where to look, how the layers are organized, how to build, test, and benchmark, and how the parameters in the code relate to the usual FHE
  notation.
- [Backends](guides/backends.md) — the FFT and NTT arithmetic families, the currently available subfamilies (`FFT64`, `NTT4x30`, `NTT3x42`), the available backend types, and how to choose one.
- [Performance](guides/performance.md) — choosing the arithmetic family, the backend, `base2k` and the thread count, with measurements for each decision and a diagnostic for finding the thread knee on your own machine.
- [Failure estimates for base-2^K arithmetic](guides/base2k-failure-probability.md), the failure model behind `Module::max_base2k`: the probability that a sum of products fails the NTT's centered reconstruction or the FFT's rounding, on the negacyclic and conjugate-invariant rings, how the radix is selected from it, and the guard limbs and headroom it implies.
- [Grafting vs. the Bivariate Representation](guides/grafting-vs-bivariate.md) — how Poulpy's bivariate base-`2^K` representation compares to RNS Grafting for bit-granular scale and modulus management.
- [HAL model and contracts](../poulpy-hal/src/api/mod.rs), the rustdoc of `poulpy_hal::api` (`cargo doc -p poulpy-hal --open`): the value model, canonical form, limb rule, mutation and exactness classes, the degree-embedding rule and the contract block every HAL operation carries, with the derived operations and the backend implementation order in [`poulpy_hal::oep`](../poulpy-hal/src/oep/mod.rs) and the window views in [`poulpy_hal::layouts`](../poulpy-hal/src/layouts/mod.rs).

- [Layering contract](guides/layering-contract.md), the API, delegate, OEP, reference and derived layers every library crate follows, opt-in selection and what an override reproduces.

- [Implementing a core backend](../poulpy-core/docs/core-contracts.md), reference algorithms, derived defaults, backend overrides, scratch contracts and parity tests.

## CKKS

- [Implementing CKKS operations](../schemes/poulpy-ckks/docs/ckks-contracts.md), explicit reference wiring, derived defaults, backend overrides, scratch, and caller-selected parity.

- [Polynomial Evaluation](guides/polynomial_evaluation.md) — evaluating a polynomial on encrypted slots with the Baby-Step Giant-Step method, the two split strategies, the supported polynomial flavors, and the modulus consumed per degree.
- [Linear Transformations](guides/linear_transformation.md) — the homomorphic matrix-vector product over the slots (`CoeffsToSlots` / `SlotsToCoeffs`) via the Baby-Step Giant-Step diagonal method, with hoisting, lazy normalization, and the cost in key-switches.
- [Bootstrapping](guides/bootstrapping.md) — refreshing a ciphertext's homomorphic budget through the `ModUp` → `CoeffsToSlots` → `EvalMod` → `SlotsToCoeffs` pipeline, the scale and budget accounting through each stage, and the standard and EvalRound+ variants.
- [PaCo Bootstrapping](guides/paco.md) — refreshing selected coefficient classes with PaCo, including validated plans/keys, direct and encapsulated modes, bounded parallel evaluation, and metadata accounting.
- [Replicated Slot Layout](guides/replicated.md) — the small-batch scheme of ePrint 2026/2513: a batch replicated across the ring with shaped rounding, its leveled operations, conversions to and from CKKS, bootstrapping, and the presets.
- [SHIP Bootstrapping](guides/ship.md) — the shallow SHIP half bootstrap: sparse-secret encapsulation, hoisted mux blind rotations, and the product tree, with validated plans, integrated key generation, and real/complex entry points.

## BLAS

- [Implementing BLAS operations](../libs/poulpy-blas/docs/blas-contracts.md), the rules every BLAS family shares: batches, key sets and widths, validation and error order, scratch and parallelism.

## libm

- [libm API guide](../libs/poulpy-libm/docs/api.md), the supported functions with their intervals and circuit costs.
- [Custom approximations](../libs/poulpy-libm/docs/custom-approximation.md), fitting and evaluating a function outside the libm scope with the approximation API of `poulpy-ckks`.

## Specifications

The [`spec/`](spec) folder holds lower-level implementation specifications and code walkthroughs.

- [BSGS Linear Transformation — Specification](spec/lt_bsgs.md) — the implementation specification of the homomorphic slot-domain linear transformation (`CoeffsToSlots` / `SlotsToCoeffs`) via the baby-step / giant-step decomposition, with hoisting and lazy normalization.
- [BSGS Linear Transformation — Implementation Walkthrough](spec/lt_bsgs_impl.md) — a file-by-file guided tour of the reference implementation that realizes the specification.
- [PaCo DFT Convention](spec/paco_dft_convention.md) — the generator-5 packing convention used by the PaCo implementation and its relationship to the paper's reference convention.
