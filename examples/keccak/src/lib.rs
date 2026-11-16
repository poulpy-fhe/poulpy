//! Keccak-f[1600] on eight encrypted states at once with `poulpy-ckks`, at 128-bit security. The README explains the
//! design.

mod bootstrapping;
pub mod client;
pub mod keccak;
pub mod layout;
pub mod params;
pub mod server;

/// The client's backend.
pub type Host = poulpy_cpu_portable::NTT4x30Portable;

/// The server's backend: the client's, or the one of the `enable-` feature of its name. With several, the most specific
/// wins: a GPU over a CPU, IFMA over AVX-512 over AVX.
#[cfg(not(any(
    feature = "enable-portable-rayon",
    feature = "enable-neon",
    feature = "enable-avx",
    feature = "enable-avx512",
    feature = "enable-ifma",
    feature = "enable-cuda",
    feature = "enable-hip",
    feature = "enable-metal",
    feature = "enable-vulkan"
)))]
pub type ServerBackend = poulpy_cpu_portable::NTT4x30Portable;
#[cfg(all(
    feature = "enable-portable-rayon",
    not(any(
        feature = "enable-neon",
        feature = "enable-avx",
        feature = "enable-avx512",
        feature = "enable-ifma",
        feature = "enable-cuda",
        feature = "enable-hip",
        feature = "enable-metal",
        feature = "enable-vulkan"
    ))
))]
pub type ServerBackend = poulpy_cpu_rayon::NTT4x30PortableRayon;
#[cfg(all(
    feature = "enable-neon",
    not(any(
        feature = "enable-avx",
        feature = "enable-avx512",
        feature = "enable-ifma",
        feature = "enable-cuda",
        feature = "enable-hip",
        feature = "enable-metal",
        feature = "enable-vulkan"
    ))
))]
pub type ServerBackend = poulpy_cpu_arm::NTT4x30NeonRayon;
#[cfg(all(
    feature = "enable-avx",
    not(any(
        feature = "enable-avx512",
        feature = "enable-ifma",
        feature = "enable-cuda",
        feature = "enable-hip",
        feature = "enable-metal",
        feature = "enable-vulkan"
    ))
))]
pub type ServerBackend = poulpy_cpu_avx::NTT4x30AvxRayon;
#[cfg(all(
    feature = "enable-avx512",
    not(any(
        feature = "enable-ifma",
        feature = "enable-cuda",
        feature = "enable-hip",
        feature = "enable-metal",
        feature = "enable-vulkan"
    ))
))]
pub type ServerBackend = poulpy_cpu_avx512::NTT4x30Avx512Rayon;
#[cfg(all(
    feature = "enable-ifma",
    not(any(
        feature = "enable-cuda",
        feature = "enable-hip",
        feature = "enable-metal",
        feature = "enable-vulkan"
    ))
))]
pub type ServerBackend = poulpy_cpu_avx512::NTT3x42IfmaRayon;
#[cfg(all(
    any(feature = "enable-metal", feature = "enable-vulkan"),
    not(any(feature = "enable-cuda", feature = "enable-hip"))
))]
pub type ServerBackend = poulpy_gpu_portable::NTT4x30Wgpu;
#[cfg(all(feature = "enable-hip", not(feature = "enable-cuda")))]
pub type ServerBackend = poulpy_gpu_portable::NTT4x30Hip;
#[cfg(feature = "enable-cuda")]
pub type ServerBackend = poulpy_gpu_portable::NTT4x30Cuda;

/// A ciphertext of the client.
pub type Ct = poulpy_ckks::layouts::CKKSCiphertextOwned<Host>;
/// A ciphertext of the server.
pub type ServerCt = poulpy_ckks::layouts::CKKSCiphertextOwned<ServerBackend>;
