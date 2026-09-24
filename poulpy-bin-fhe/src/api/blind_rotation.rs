//! User-facing blind-rotation execution, key lifecycle, and lookup-table helpers.
//!
//! Import these traits to call operations on a backend module. Every trait,
//! including compressed-key allocation and LUT construction, dispatches through
//! the selected backend's [`crate::oep`] implementation. Operations that take
//! caller workspace keep their scratch queries with their execution methods.
//!
//! [`BlindRotationKeyPrepared`](crate::blind_rotation::BlindRotationKeyPrepared)
//! also exposes convenience methods using the same operation traits.
//!
//! # Evaluate a lookup table
//!
//! Given a prepared key and a populated LUT, query workspace for their actual
//! block and extension sizes before evaluating an encrypted index:
//!
//! ```
//! use poulpy_bin_fhe::{
//!     api::blind_rotation::BlindRotationExecute,
//!     blind_rotation::{BlindRotationKeyPrepared, CGGI, LookupTable},
//! };
//! use poulpy_core::layouts::{GLWEInfos, GLWEToBackendMut, LWEInfos, LWEToBackendRef};
//! use poulpy_hal::{
//!     api::ScratchOwnedAlloc,
//!     layouts::{Backend, ScratchOwned},
//! };
//!
//! fn evaluate<BE, M, R, L>(
//!     module: &M,
//!     output: &mut R,
//!     input: &L,
//!     lut: &LookupTable<BE::OwnedBuf, BE::ZnxWord>,
//!     key: &BlindRotationKeyPrepared<BE::OwnedBuf, CGGI, BE>,
//! ) where
//!     BE: Backend,
//!     M: BlindRotationExecute<CGGI, BE>,
//!     R: GLWEToBackendMut<BE> + GLWEInfos,
//!     L: LWEToBackendRef<BE> + LWEInfos,
//!     ScratchOwned<BE>: ScratchOwnedAlloc<BE>,
//! {
//!     let bytes = module.blind_rotation_execute_tmp_bytes(
//!         key.block_size(),
//!         lut.extension_factor(),
//!         output,
//!         key,
//!     );
//!     let mut scratch = ScratchOwned::<BE>::alloc(bytes);
//!     module.blind_rotation_execute(output, input, lut, key, &mut scratch.arena());
//! }
//! # fn main() {}
//! ```

mod blind_rotation_execute;
mod blind_rotation_key_compressed_encrypt_sk;
mod blind_rotation_key_compressed_factory;
mod blind_rotation_key_decompress;
mod blind_rotation_key_encrypt_sk;
mod blind_rotation_key_prepared_factory;
mod blind_rotation_mod_switch;
mod lookup_table_factory;

pub use blind_rotation_execute::BlindRotationExecute;
pub use blind_rotation_key_compressed_encrypt_sk::BlindRotationKeyCompressedEncryptSk;
pub use blind_rotation_key_compressed_factory::BlindRotationKeyCompressedFactory;
pub use blind_rotation_key_decompress::BlindRotationKeyDecompress;
pub use blind_rotation_key_encrypt_sk::BlindRotationKeyEncryptSk;
pub use blind_rotation_key_prepared_factory::BlindRotationKeyPreparedFactory;
pub use blind_rotation_mod_switch::BlindRotationModSwitch;
pub use lookup_table_factory::LookupTableFactory;
