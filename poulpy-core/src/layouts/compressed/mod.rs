//! Seed-compressed ciphertext and key layouts.
//!
//! Compressed variants store only the *body* component of a
//! ciphertext or key; the pseudorandom mask is regenerated
//! deterministically from a 32-byte PRNG seed. This typically
//! halves (rank-1) the serialised size at the cost of a
//! decompression step before use.
//!
//! Every compressed type has a `decompress` / `GLWEDecompress`
//! method that expands it back into the corresponding standard
//! layout.

macro_rules! impl_gglwe_compressed_to_backend_for_field {
    ($ty:ty, $field:tt, $inner:ty) => {
        impl<BE: Backend> GGLWECompressedToBackendRef<BE> for $ty {
            fn to_backend_ref(&self) -> GGLWECompressedBackendRef<'_, BE> {
                <$inner as GGLWECompressedToBackendRef<BE>>::to_backend_ref(&self.$field)
            }
        }

        impl<BE: Backend> GGLWECompressedToBackendMut<BE> for $ty {
            fn to_backend_mut(&mut self) -> GGLWECompressedBackendMut<'_, BE> {
                <$inner as GGLWECompressedToBackendMut<BE>>::to_backend_mut(&mut self.$field)
            }
        }
    };
}

mod gglwe;
mod gglwe_to_ggsw_key;
mod ggsw;
mod glwe;
mod glwe_automorphism_key;
mod glwe_switching_key;
mod glwe_tensor_key;
mod glwe_to_lwe_key;
mod lwe;
mod lwe_switching_key;
mod lwe_to_glwe_key;

pub use gglwe::*;
pub use gglwe_to_ggsw_key::*;
pub use ggsw::*;
pub use glwe::*;
pub use glwe_automorphism_key::*;
pub use glwe_switching_key::*;
pub use glwe_tensor_key::*;
pub use glwe_to_lwe_key::*;
pub use lwe::*;
pub use lwe_switching_key::*;
pub use lwe_to_glwe_key::*;
