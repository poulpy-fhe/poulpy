# 🐙 Poulpy documentation

- [Getting Started](getting-started.md) — a map of the codebase: what each crate contains and where to look, how the layers are organized, how to build, test, and benchmark, and how the parameters in the code relate to the usual FHE
  notation.
- [Backends](backends.md) — the FFT and NTT arithmetic families, the currently available subfamilies (`FFT64`, `NTT4x30`, `NTT3x42`), the available backend types, and how to choose one.
- [Grafting vs. the Bivariate Representation](grafting-vs-bivariate.md) — how Poulpy's bivariate base-`2^K` representation compares to RNS Grafting for bit-granular scale and modulus management.

## CKKS

- [Polynomial Evaluation](polynomial_evaluation.md) — evaluating a polynomial on encrypted slots with the Baby-Step Giant-Step method, the two split strategies, the supported polynomial flavors, and the modulus consumed per degree.
- [Linear Transformations](linear_transformation.md) — the homomorphic matrix-vector product over the slots (`CoeffsToSlots` / `SlotsToCoeffs`) via the Baby-Step Giant-Step diagonal method, with hoisting, lazy normalization, and the cost in key-switches.
- [Bootstrapping](bootstrapping.md) — refreshing a ciphertext's homomorphic budget through the `ModUp` → `CoeffsToSlots` → `EvalMod` → `SlotsToCoeffs` pipeline, the scale and budget accounting through each stage, and the standard and EvalRound+ variants.
- [PaCo Bootstrapping](paco.md) — refreshing selected coefficient classes with PaCo, including validated plans/keys, direct and encapsulated modes, bounded parallel evaluation, and metadata accounting.
- [SHIP Bootstrapping](ship.md) — the shallow SHIP half bootstrap: sparse-secret encapsulation, hoisted mux blind rotations, and the product tree, with validated plans, integrated key generation, and real/complex entry points.

## Specifications

The [`spec/`](spec) folder holds lower-level implementation specifications and code walkthroughs.

- [BSGS Linear Transformation — Specification](spec/lt_bsgs.md) — the implementation specification of the homomorphic slot-domain linear transformation (`CoeffsToSlots` / `SlotsToCoeffs`) via the baby-step / giant-step decomposition, with hoisting and lazy normalization.
- [BSGS Linear Transformation — Implementation Walkthrough](spec/lt_bsgs_impl.md) — a file-by-file guided tour of the reference implementation that realizes the specification.
- [PaCo DFT Convention](spec/paco_dft_convention.md) — the generator-5 packing convention used by the PaCo implementation and its relationship to the paper's reference convention.
