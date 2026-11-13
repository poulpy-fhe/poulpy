//! Keccak-f[1600] on eight encrypted states at once with `poulpy-ckks`, at 128-bit security. The README explains the
//! design.

pub mod bootstrapping;
pub mod client;
pub mod keccak;
pub mod layout;
pub mod params;
pub mod server;

/// The client's backend.
pub type Host = poulpy_cpu_portable::NTT4x30Portable;

/// The server's backend: the client's, or the one of the feature of its name.
#[cfg(not(any(
    feature = "portable-rayon",
    feature = "neon",
    feature = "avx",
    feature = "avx512",
    feature = "cuda",
    feature = "hip",
    feature = "metal",
    feature = "vulkan"
)))]
pub type ServerBackend = poulpy_cpu_portable::NTT4x30Portable;
#[cfg(feature = "portable-rayon")]
pub type ServerBackend = poulpy_cpu_rayon::NTT4x30PortableRayon;
#[cfg(feature = "neon")]
pub type ServerBackend = poulpy_cpu_arm::NTT4x30NeonRayon;
#[cfg(feature = "avx")]
pub type ServerBackend = poulpy_cpu_avx::NTT4x30AvxRayon;
#[cfg(feature = "avx512")]
pub type ServerBackend = poulpy_cpu_avx512::NTT4x30Avx512Rayon;
#[cfg(feature = "cuda")]
pub type ServerBackend = poulpy_gpu_portable::NTT4x30Cuda;
#[cfg(feature = "hip")]
pub type ServerBackend = poulpy_gpu_portable::NTT4x30Hip;
#[cfg(any(feature = "metal", feature = "vulkan"))]
pub type ServerBackend = poulpy_gpu_portable::NTT4x30Wgpu;

/// A ciphertext of the client.
pub type Ct = poulpy_ckks::layouts::CKKSCiphertextOwned<Host>;
/// A ciphertext of the server.
pub type ServerCt = poulpy_ckks::layouts::CKKSCiphertextOwned<ServerBackend>;
