/// Backend-level key-encryption trait for [`BlindRotationKeyCompressed`](crate::blind_rotation::BlindRotationKeyCompressed).
///
/// Equivalent to `BlindRotationKeyEncryptSk` but produces the seed-compressed
/// form of the key.  Instead of a mutable `source_xa`, callers supply a fixed
/// 32-byte `seed_xa`; all per-GGSW mask seeds are derived deterministically
/// from this root seed via `Source::new_seed`.
///
/// # Panics
///
/// Panics if the LWE secret distribution is not a supported binary type.
pub use crate::api::BlindRotationKeyCompressedEncryptSk;
