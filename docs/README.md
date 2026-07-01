# 🐙 Poulpy documentation

- [Getting Started](getting-started.md) — a map of the codebase: what each crate contains and where to look, how the layers are organized, how to build, test, and benchmark, and how the parameters in the code relate to the usual FHE
  notation.
- [Backends](backends.md) — the three arithmetic families (`FFT64`, `NTT120`, `NTT126`), the available backend types, and how to choose one.
- [Grafting vs. the Bivariate Representation](grafting-vs-bivariate.md) — how Poulpy's bivariate base-`2^K` representation compares to RNS Grafting for bit-granular scale and modulus management.

## CKKS

- [Polynomial Evaluation](polynomial_evaluation.md) — evaluating a polynomial on encrypted slots with the Baby-Step Giant-Step method, the two split strategies, the supported polynomial flavors, and the modulus consumed per degree.
- [Linear Transformations](linear_transformation.md) — the homomorphic matrix-vector product over the slots (`CoeffsToSlots` / `SlotsToCoeffs`) via the Baby-Step Giant-Step diagonal method, with hoisting, lazy normalization, and the cost in key-switches.
- [Bootstrapping](bootstrapping.md) — refreshing a ciphertext's homomorphic budget through the `ModUp` → `CoeffsToSlots` → `EvalMod` → `SlotsToCoeffs` pipeline, the scale and budget accounting through each stage, and the standard and EvalRound+ variants.
