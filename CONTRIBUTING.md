# Contributing to Poulpy

The Poulpy team encourages contributions. We welcome bug fixes, documentation, tests, performance work, and new features, from users and researchers alike. Implementations of new arithmetic backends, and of schemes built on the existing ones, are especially welcome.

## Sync with the team first

We are open to all external contributions, but please talk to us **before** writing a substantial patch.

Poulpy is a layered library: `poulpy-hal` (arithmetic and layouts), `poulpy-core` (RLWE machinery), the scheme crates, and the backend crates. A change often has a natural home in a different layer than the one where the problem surfaced, and a new operation usually has to be introduced through the override extension points (`oep`) so that backends can specialize it. A short exchange up front is much cheaper than a redesign in review.

Small, self-evident changes (typos, documentation, an obvious bug with an obvious fix) can go straight to a pull request.

## Communication

Most discussion happens on [GitHub issues](https://github.com/poulpy-fhe/poulpy/issues). For design questions and quicker exchanges, join our [Telegram group](https://t.me/+uy7_HADsdN1jNmU1).

For anything better suited to a direct exchange, reach the maintainer at [jeanphilippe.bossuat@gmail.com](mailto:jeanphilippe.bossuat@gmail.com). Security vulnerabilities are the exception and follow [SECURITY.md](./SECURITY.md).

## Reporting bugs and requesting features

File bugs at [https://github.com/poulpy-fhe/poulpy/issues](https://github.com/poulpy-fhe/poulpy/issues). A useful report names the affected crate, the backend and parameters (ring degree, `base2k`, `k`, `rank`, `dsize`), and the smallest reproduction you can manage.

Feature requests belong there too. Please make sure the feature is self-contained, easy to define, and generic enough to serve more than one use case, and describe the use case you have in mind.

**Do not open a public issue for a security vulnerability.** Follow [SECURITY.md](./SECURITY.md) instead.

## Branches and pull requests

- Contributors with write access work on a topic branch named `username/short-title`, or better `username/issuenumber-short-title`. Everyone else forks and opens a pull request from the fork.
- Keep a pull request limited to a single concern.
- Push as many commits as needed until the CI is green and every review comment is resolved.
- Once the reviews are positive and the CI is green, the PR will be merged.

### Before you push

The toolchain is pinned in [`rust-toolchain.toml`](./rust-toolchain.toml), so `cargo` picks the right nightly on its own. CI runs six jobs (portable, AVX, AVX-512, NEON, and two macOS smoke jobs); the backend jobs skip themselves when the runner lacks the instruction set. The feature sets they use are at the top of [`.github/workflows/ci.yml`](./.github/workflows/ci.yml), and the portable one is the minimum to run locally:

```sh
PORTABLE_FEATURES="poulpy-core/enable-core poulpy-cpu-ref/enable-core poulpy-cpu-ref/enable-ckks poulpy-bin-fhe/enable-bin-fhe"

cargo fmt --all --check
cargo clippy --workspace --all-targets --features "$PORTABLE_FEATURES" -- -D warnings
cargo test --workspace --features "$PORTABLE_FEATURES"
```

If you touch an accelerated backend, run its lane too, with `RUSTFLAGS="-C target-cpu=native"` and that backend's feature set.

## Coding conventions

- Format with `cargo fmt --all`; the settings live in [`rustfmt.toml`](./rustfmt.toml). Clippy must be clean under `-D warnings`.
- Public API items carry doc comments. Prefer documenting the contract, in particular anything a backend override has to reproduce.
- Respect the four-layer split. `api` declares the user-facing trait, `oep` the override extension point, `delegates` the forwarding, and `default` the portable body. A new operation family that a backend might want to specialize goes through `oep`, not straight into `default`.
- A backend does not re-implement the test suites: `poulpy-hal` and `poulpy-core` ship theirs as generic functions, instantiated through `backend_test_suite!`, `cross_backend_test_suite!`, `core_backend_test_suite!` and `core_parity_test_suite!`. See [Testing a Backend](./README.md#testing-a-backend). A backend with a narrow envelope (for example not supporting rank > 1 for CKKS) restricts the sweep through `ParityShapes` rather than dropping the suite.
- Non-trivial logic must come with its own tests. 
- Correctness against the noise model belongs in the `noise` suite, agreement with the reference backend in the `parity` suite.

## Changelog, versions and history

Every user-visible change gets an entry under `## [Unreleased]` in [CHANGELOG.md](./CHANGELOG.md), in the section of the crate it affects. Prefix an entry that breaks an API with `**Breaking:**`.

The workspace crates share one version, bumped together in a single release commit that also cuts `[Unreleased]` into a dated section. Releases are tagged on `main`.
