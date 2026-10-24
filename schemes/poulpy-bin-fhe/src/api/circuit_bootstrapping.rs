//! User-facing circuit bootstrapping operations.
//!
//! These traits provide key encryption and preparation, execution with constant
//! or exponent encoding, and reusable execution plans. Each operation keeps its
//! scratch-space query alongside the corresponding execution method.
//!
//! Prepare a constant-encoding plan, then allocate the workspace selected for
//! that plan before executing it. The plan and workspace can be reused for
//! inputs with the same parameters.
//!
//! ```
//! use poulpy_bin_fhe::{
//!     api::circuit_bootstrapping::CircuitBootstrappingExecute, blind_rotation::CGGI,
//!     circuit_bootstrapping::CircuitBootstrappingKeyPrepared,
//! };
//! use poulpy_core::layouts::{
//!     GGSWAtViewMut, GGSWAtViewRef, GGSWInfos, GGSWToBackendMut, LWEInfos, LWEToBackendRef,
//! };
//! use poulpy_hal::{
//!     api::ScratchOwnedAlloc,
//!     layouts::{Backend, ScratchOwned},
//! };
//!
//! fn bootstrap<BE, M, R, L>(
//!     module: &M,
//!     output: &mut R,
//!     input: &L,
//!     key: &CircuitBootstrappingKeyPrepared<BE::OwnedBuf, CGGI, BE>,
//!     log_domain: usize,
//!     extension_factor: usize,
//! ) where
//!     BE: Backend,
//!     M: CircuitBootstrappingExecute<CGGI, BE>,
//!     R: GGSWToBackendMut<BE> + GGSWAtViewRef<BE> + GGSWAtViewMut<BE> + GGSWInfos,
//!     L: LWEToBackendRef<BE> + LWEInfos,
//!     ScratchOwned<BE>: ScratchOwnedAlloc<BE>,
//! {
//!     let plan = module.circuit_bootstrapping_prepare_to_constant(
//!         output,
//!         key,
//!         log_domain,
//!         extension_factor,
//!     );
//!     let bytes = module.circuit_bootstrapping_execute_prepared_tmp_bytes(&plan.layout(), key);
//!     let mut scratch = ScratchOwned::<BE>::alloc(bytes);
//!     module.circuit_bootstrapping_execute_prepared(
//!         output,
//!         input,
//!         key,
//!         &plan,
//!         &mut scratch.arena(),
//!     );
//! }
//! # fn main() {}
//! ```

mod circuit_bootstrapping_execute;
mod circuit_bootstrapping_key_encrypt_sk;
mod circuit_bootstrapping_key_prepared_factory;

pub use circuit_bootstrapping_execute::CircuitBootstrappingExecute;
pub use circuit_bootstrapping_key_encrypt_sk::CircuitBootstrappingKeyEncryptSk;
pub use circuit_bootstrapping_key_prepared_factory::CircuitBootstrappingKeyPreparedFactory;
