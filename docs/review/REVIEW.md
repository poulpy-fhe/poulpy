# Poulpy review guidelines

Entry point for reviewing a poulpy diff: the one file a review needs to be given. It routes the reviewer to the rules that apply, which live next to it in `docs/review/`: `workspace.md` holds the rules shared by every crate, and one file per crate holds the rest. Apply the selected rules to the diff and to the callers it affects; generic Rust correctness, safety and API review still applies on top.

## 1. How to review with these files

- **Procedure.**
  1. Read `workspace.md` in full: every review applies it.
  2. Pick the crate files from section 2, by the paths the diff touches and by what it changes.
  3. In each picked file, read the sections section 3 lists for that file, the sections section 4 selects for the kinds of change the diff makes (a diff usually has several), and the topics of section 5 the diff touches. A topic row applies whenever the diff touches the topic, even when no path picked its file.
  4. Read a picked file in full when the change fits no kind, or when the diff adds a crate, a layer, an OEP family or a registration macro.
- **Kinds.** Classify by what the diff does, not by the files it touches: a pure rename is the names row alone, and a rename that also changes behaviour adds the rows of that behaviour; a fix is the defect-fix row plus the rows of what it fixes.
- **Reading a section.** Sections are numbered headings (`## 5.`, `### 5.1.`), and a section includes its subsections. List them with `grep -n '^#' <file>` and read from a heading to the next heading of the same or a higher level.
- **Precedence.** Current owner instructions come first, then these files, then the contract sources (`docs/guides/layering-contract.md`, `poulpy-hal/src/api/mod.rs` and the per-operation contract blocks of `poulpy-hal/src/api`, the crate `poulpy-*/docs/*-contracts.md`, `poulpy-hal/docs/backend_safety_contract.md`), then READMEs and `docs/`, then existing code. Check an apparent conflict against the reviewed revision and against any intentional contract change in the diff.
- **Deviations are not precedents.** Existing code and docs break some of these rules. Never cite or copy a deviation, never ask a PR to fix one in code it does not touch, and flag every new instance a diff adds, including a copy of an existing deviating pattern into new code.
- **Verify before claiming.** Check each finding against the code, including the backend path a call actually selects, its overrides and its tests. A rule is a reason to look, not proof of a defect.
- **Actionable findings.** A finding names the changed code, the rule it breaks (file and section) and a concrete affected case or cost. Keep demonstrated defects apart from questions and from unmeasured optimizations, and stay within the PR's scope.

## 2. Files by path and by change

| The diff | Picks |
|---|---|
| touches `poulpy-hal/` | `hal.md` |
| touches `poulpy-core/` | `core.md` |
| touches `schemes/poulpy-ckks/` | `ckks.md` |
| touches `schemes/poulpy-bin-fhe/` | `bin-fhe.md` |
| touches `poulpy-mhe/` | `mhe.md` |
| touches `libs/poulpy-blas/` | `blas.md` |
| touches `backends/poulpy-cpu-*/` (portable, rayon, avx, avx512, arm, oracle) | `backends.md` and `hal.md` (also its sections 6, 7, 9 and 11), plus the file of every crate whose OEP trait or test suite the change implements, overrides or registers, and of the crate an example under `backends/poulpy-cpu-portable/examples` demonstrates |
| adds or changes an OEP trait, OEP method, opt-in or registration macro, or public test suite, in any crate | `backends.md` (the backend side must follow) |
| touches `tools/poulpy-bench/` or any `benches/` directory, or makes a performance claim (PR description, CHANGELOG, README, `docs/guides/performance.md`, gate constant) | `bench.md` |
| touches `.github/` | these sections alone, picking no file: `hal.md` section 11, `ckks.md` sections 15.1 and 15.4, `backends.md` sections 11 and 13.7, `bench.md` section 4 |
| touches `docs/guides/layering-contract.md` | these sections alone, picking no file: `hal.md` sections 1, 4 and 5, `core.md` sections 1 and 2, `ckks.md` sections 1 and 2, `bin-fhe.md` sections 1 and 2, `mhe.md` sections 1 and 2, `blas.md` sections 1 and 2, `backends.md` sections 3 and 4 |
| touches a crate's manifest, README or `docs/` directory | that crate's file |
| touches `docs/guides/bootstrapping.md` | `ckks.md` |
| touches `docs/guides/base2k-failure-probability.md` | these sections alone, picking no file: `hal.md` section 6, `ckks.md` section 7, `blas.md` section 7, `backends.md` section 5.7 |

## 3. Sections read whenever a file is picked

| File | Sections |
|---|---|
| `hal.md` | 1, 2, 10 |
| `core.md` | 1, 2 |
| `ckks.md` | 1, 2 |
| `bin-fhe.md` | 1, 2 |
| `mhe.md` | 1, 2 |
| `blas.md` | 0, 1, 2 |
| `backends.md` | 0, 1, 2, 3 |
| `bench.md` | 1, 2, 3, 4, 5 |

## 4. Sections by kind of change

Each cell lists the sections to read in that file for that kind of change, on top of section 3; `-` means none. Read only the columns of the picked files; `all` means the whole file.

| Kind of change | `hal.md` | `core.md` | `ckks.md` | `bin-fhe.md` | `mhe.md` | `blas.md` | `backends.md` |
|---|---|---|---|---|---|---|---|
| New or changed operation: API or OEP method, reference or derived body, its query | 3, 4, 5, 6, 7, 8, 9, 11 | 3, 4, 5, 6, 7, 8, 9, 13 | 3, 4, 5, 6, 7, 8, 15 | 3, 8, 9, 10, 11 | 3, 4, 5, 7, 8, 9 | 3, 4, 5, 6, 7, 8, 9, 10, 11 | 4, 8, 9, 13.2, 13.5 |
| Backend override, kernel or backend registration | 3.2, 5, 6, 7, 8, 9, 11 | 3, 7, 8, 9, 10, 13.1 | 3, 5, 6, 7, 8, 10, 15.1 | 3, 4, 5, 8, 9, 11 | 4, 5, 7, 9 | 3, 4, 5, 6, 7, 8, 9 | all |
| Backend contract: a `Backend` constant, capability flag, sizing default or layout marker | 6, 7, 8, 11 | - | - | - | - | - | 5.5, 5.7, 13 |
| Setup: plans, contexts, presets, key sets, LUTs | - | - | 3, 4, 5, 9, 10, 12, 15.3 | 4, 7 | - | 4, 6, 9 | - |
| Scratch: a query, a carve, an arena, per-worker scratch | 3.2, 5, 6, 8 | 4, 5 | 4, 13 | 3, 5 | 3.5 | 8 | 5.4, 8 |
| Validation, asserts, panics, errors | 3.2, 9 | 3, 6, 9 | 3 | 9 | 4 | 3 | 5.1 |
| Normalization, canonical flag, rounding, DFT and IDFT work | 3.2, 9 | 7, 8.3 | 7, 8 | 10 | 3.3, 7 | 5, 7 | 5.3, 5.6, 6 |
| Metadata: noise, plaintext, scheme or share metadata | - | 7, 9, 12 | 6 | 10 | 3.4 | 5 | - |
| Keys, encryption, sampling, seeds, secrets | 9 | 5, 10, 11 | 5, 14 | 4 | 5 | 4 | 5.1, 5.8 |
| Layouts and types: a new layout, field, tier, view, allocation, prepared or compressed form | 6, 7 | 6 | 4, 14 | 4 | 7 | 9 | 5.7 |
| Serialization | 7 | 12 | 14 | 4.1 | 6 | 9 | - |
| Degree, ring, Galois elements, modules | 3.2, 6, 9 | 5 | 5 | 4, 9 | 4 | 4, 5 | 5.5 |
| Names, signatures, renames | 3, 5, 9, 11 | 3, 13 | 3, 15 | 11 | 9 | 3, 11 | 4, 5.1, 13 |
| Docs, comments, contract blocks | 3 | - | 3, 6 | - | 8 | 10 | 12 |
| Visibility, features, manifests, dependencies, examples | - | - | 10, 14 | 4 | - | 10 | 11, 12 |
| Tests and fixtures | 11 | 13 | 15 | 11 | 9 | 11 | 13 |
| Defect fix | 11 | 13.4 | 15.3 | 11 | 9 | 11 | 13 |
| CI workflows and scripts | 11 | - | 15.1, 15.4 | - | - | - | 11, 13.7 |
| Performance: fused paths, gates, algorithm choices, benches | 5, 9 | 8 | 8 | 5 | - | 6 | 10, 11 |
| Unsafe code, raw pointers, parallelism | 7, 9, 11 | 4 | 4 | 5 | - | 8 | 5.2, 8 |

## 5. Topics

| File | Section | Topic |
|---|---|---|
| `hal.md` | 3 | contract blocks |
| `hal.md` | 4 | delegates |
| `hal.md` | 9 | what every backend kernel must guarantee |
| `core.md` | 8.2 | gadget products |
| `core.md` | 8.3 | tensor products and their stages, also in ckks dot products and blas JKLS products |
| `core.md` | 8.4 | linear transformations |
| `core.md` | 10 | keys, key sources and gadget decomposition, also key sources implemented in bin-fhe |
| `core.md` | 11 | encryption, sampling and secrets |
| `core.md` | 12 | transfers between backends and decompression |
| `ckks.md` | 9 | plaintexts and encoding |
| `ckks.md` | 10 | numerics, encoding transforms, codecs and setup constants, also the oracle's CKKS roots |
| `ckks.md` | 11 | linear transformations, diagonals and the homomorphic DFT (plans, matrices, generators, schedules) |
| `ckks.md` | 12 | bootstrapping, folding, PaCo and SHIP |
| `ckks.md` | 13 | polynomial evaluation, EvalMod and approximation |
| `ckks.md` | 15.4 | setup-math pins and determinism fixtures |
| `bin-fhe.md` | 1.2 | convenience methods and stateful helpers |
| `bin-fhe.md` | 4 | lookup tables: construction, drift and direction |
| `bin-fhe.md` | 5 | blind-rotation algorithms and scheduling |
| `bin-fhe.md` | 6 | BDD arithmetic and generated circuits |
| `bin-fhe.md` | 7 | circuit bootstrapping |
| `bin-fhe.md` | 8 | host data versus execution |
| `mhe.md` | 3 | gen, aggregate and finalize |
| `mhe.md` | 5 | seeds, floods and the threat model |
| `blas.md` | 6 | algorithm paths |
| `blas.md` | 7 | lazy sums, radix and normalization |
| `blas.md` | 8 | heap memory, long-lived temporaries and streamed units |
| `backends.md` | 5.9 | CKKS encoding transforms and tables |
| `backends.md` | 6 | the DFT slot law and block layouts |
| `backends.md` | 7 | SIMD crates |
| `backends.md` | 8 | Rayon twins |
| `backends.md` | 9 | the oracle |
| `backends.md` | 10 | fast-path gates and tuning constants |
| `backends.md` | 13.3 | Rayon twin and degree-floor test coverage |
| `backends.md` | 13.6 | oracle self-tests and fixtures |
