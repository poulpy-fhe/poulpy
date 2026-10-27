# Poulpy review guidelines

Entry point for reviewing a poulpy diff: the one file a review needs to be given. The rules live next to it in `docs/review/`: `workspace.md` holds the rules shared by every crate and one file per crate adds its own. Apply them to the diff and to the callers it affects; generic Rust correctness, safety and API review still applies on top.

## Files to read

Read `workspace.md`, then each file the diff selects, all in full:

- `hal.md`: `poulpy-hal/` or a backend crate.
- `core.md`: `poulpy-core/`.
- `ckks.md`: `schemes/poulpy-ckks/`.
- `bin-fhe.md`: `schemes/poulpy-bin-fhe/`.
- `mhe.md`: `poulpy-mhe/`.
- `blas.md`: `libs/poulpy-blas/`.
- `backends.md`: `backends/poulpy-cpu-*/` (portable, rayon, avx, avx512, arm, oracle), `.github/`, or a new or changed OEP trait or method, opt-in or registration macro or public test suite in any crate.
- `gpu.md`: `backends/poulpy-gpu-portable/`, `.github/`, or a new or changed OEP trait or method, opt-in or registration macro or public test suite in any crate.
- `bench.md`: `tools/poulpy-bench/`, a `benches/` directory, or a performance claim (PR description, CHANGELOG, README, `docs/guides/performance.md`, a gate constant).

Backend code also selects the file of each crate whose OEP traits or test suites it implements, overrides or registers, and an example under `backends/poulpy-cpu-portable/examples` the file of the crate it demonstrates. A guide under `docs/guides/` selects the files of the crates it documents (`layering-contract.md`: `hal.md` and `core.md`).

## How to review

- **Precedence.** Current owner instructions come first, then these files, then the contract sources (`docs/guides/layering-contract.md`, `poulpy-hal/src/api/mod.rs` and the per-operation contract blocks of `poulpy-hal/src/api`, the crate `poulpy-*/docs/*-contracts.md`, `poulpy-hal/docs/backend_safety_contract.md`), then READMEs and `docs/`, then existing code. Check an apparent conflict against the reviewed revision and against any intentional contract change in the diff.
- **Deviations are not precedents.** Existing code and docs break some of these rules. Never cite or copy a deviation, never ask a PR to fix one in code it does not touch, and flag every new instance a diff adds, including a copy of an existing deviating pattern into new code.
- **Verify before claiming.** Check each finding against the code, including the backend path a call actually selects, its overrides and its tests. A rule is a reason to look, not proof of a defect.
- **Actionable findings.** A finding names the changed code, the rule it breaks (file and section) and a concrete affected case or cost. Keep demonstrated defects apart from questions and from unmeasured optimizations, and stay within the PR's scope.
