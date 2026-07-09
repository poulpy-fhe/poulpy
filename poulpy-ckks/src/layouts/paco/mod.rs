//! PaCo plans, compiled contexts, structured-secret specifications, and key
//! containers.
//!
//! PaCo implements the CKKS bootstrapping algorithm from "PaCo:
//! Bootstrapping for CKKS via Partial CoeffToSlot" (Coron & Seuré,
//! [ePrint 2025/886](https://eprint.iacr.org/2025/886), ASIACRYPT 2025).
//! The DFT/packing conventions are the crate's **official** ones — the
//! generator-5 embedding of [`gen_dft_matrices`](crate::default::dft), block
//! coefficients in natural slot order — not the paper reference
//! implementation's variant-DFT + extended bit-reversal: each `2C`-wide slot
//! block is an element of `ℂ[Z]/(Z^{2C} − i)`, packed in the clear by a
//! per-block small FFT and inverted homomorphically by the same generator's
//! Encode factorization (see `crate::default::paco::lt`).
//!
//! The invariant-bearing types live here with the rest of the CKKS data
//! structures; the cleartext circle embedding and coefficient encodings live
//! in `crate::encoding::paco`; the backend-generic circuits, factor
//! generation, and drivers live in [`crate::default::paco`]; evaluation is
//! exposed through [`CKKSPaCoOps`](crate::api::CKKSPaCoOps). The
//! implementation submodules stay crate-private so helper functions and
//! orchestration details do not become a second public API.
//!
//! Naming maps to the paper as follows (`N = 2^log_n` the ring degree,
//! `q = 2^log_q` the input modulus):
//!
//! | Paper | Here |
//! |-------|------|
//! | `h` (secret Hamming weight) | [`PaCoPlan::h`] |
//! | `C` (coefficients bootstrapped per run) | [`PaCoPlan::c`] |
//! | `B = N/(4h)`, `k = B/C`, `n = 2hC` | [`PaCoPlan::b`], [`PaCoPlan::k`], [`PaCoPlan::slots`] |
//! | packing / partial CoeffToSlot / SlotToCoeff′ | `secret::pack_chunk`, `lt::paco_c2s_factors`, `lt::paco_stc_factors` |
//! | Algorithm 1 (skGen) | [`PaCoSecretSpec::sample`] |
//! | Algorithm 2 (bskGen, pre-encryption σ_t) | [`PaCoSecretSpec::sigma_slots_reim`] |
//! | Algorithm 3 (getCoeffEnc) | [`CKKSPaCoOps::ckks_paco_coeff_encodings`](crate::api::CKKSPaCoOps::ckks_paco_coeff_encodings) |
//! | Algorithm 4 (seqPaCo) | private bootstrap implementation; independent test-suite oracle |
//! | Algorithm 5 (parallelPaCo) | private parallel implementation; exercised by the PaCo test suite |

use poulpy_hal::layouts::{Backend, Module, ScratchOwned};

pub(crate) mod context;
pub(crate) mod keyset;
pub(crate) mod plan;
pub(crate) mod secret;

pub use context::PaCoContext;
pub use keyset::{PaCoKeyParameters, PaCoKeySet, PaCoKeySetParts, PaCoKeys, PaCoKeysPrepared, PaCoKeysPreparedParts};
pub use plan::{PaCoDFTPlan, PaCoPlan, PaCoSlotOrder};
pub use secret::PaCoSecretSpec;

/// Reusable execution context for one background PaCo worker.
///
/// A worker owns a backend module handle and its scratch arena. Parallel PaCo
/// borrows a caller-provided slice of workers, validates every worker it uses,
/// and returns the resources to the caller after the scoped threads join. This
/// keeps module initialization and scratch allocation out of repeated
/// bootstrap calls.
pub struct PaCoWorker<BE: Backend> {
    module: Module<BE>,
    scratch: ScratchOwned<BE>,
}

impl<BE: Backend> PaCoWorker<BE> {
    /// Creates a reusable worker context.
    ///
    /// Ring-degree and scratch-capacity compatibility are checked against the
    /// concrete PaCo call before any output is modified.
    pub fn new(module: Module<BE>, scratch: ScratchOwned<BE>) -> Self {
        Self { module, scratch }
    }

    /// Backend module used by this worker.
    ///
    /// This accessor is useful for querying the backend-specific scratch bound
    /// before constructing a replacement worker arena.
    pub fn module(&self) -> &Module<BE> {
        &self.module
    }

    /// Decomposes the worker into its reusable module and scratch arena.
    pub fn into_parts(self) -> (Module<BE>, ScratchOwned<BE>) {
        (self.module, self.scratch)
    }

    pub(crate) fn parts_mut(&mut self) -> (&Module<BE>, &mut ScratchOwned<BE>) {
        (&self.module, &mut self.scratch)
    }
}
