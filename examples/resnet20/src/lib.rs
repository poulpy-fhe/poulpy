//! ResNet-20 on CIFAR-10 with poulpy-ckks and S2C-first bootstrapping.
//!
//! [`Params`] derives the bootstrap, a [`Client`] holds the secret, generates the keys and encrypts an image, and a
//! [`Server`] runs the network on it; [`model`] is the network in the clear. Every convolution runs in the last factor
//! of the SlotsToCoeffs of a bootstrap: [`CoeffLayout`] says where a tensor sits in the coefficients, `kernel` turns a
//! convolution into the polynomials of that factor and `merged` evaluates it.

mod client;
mod kernel;
mod layout;
mod merged;
pub mod model;
mod params;
mod relu;
mod server;

use poulpy_ckks::layouts::{BootstrappingContextPrepared, BootstrappingKeySet, CKKSCiphertextOwned, CKKSPlaintextOwned};
use poulpy_cpu_portable::NTT4x30Portable;
use poulpy_hal::layouts::Backend;

pub use client::Client;
pub use layout::CoeffLayout;
pub use params::Params;
#[cfg(feature = "enable-cuda")]
pub use server::RecordedInference;
pub use server::{Probe, ProbePoint, Server, Timings};

/// The client's backend, which holds the secret.
pub type Host = NTT4x30Portable;
/// The server's backend: the CPU, or the GPU with the `enable-cuda` feature.
#[cfg(not(feature = "enable-cuda"))]
pub type ServerBackend = NTT4x30Portable;
#[cfg(feature = "enable-cuda")]
pub type ServerBackend = poulpy_gpu_portable::NTT4x30Cuda;

/// The bootstrapping keys, as the client generates them.
pub type KeySet = BootstrappingKeySet<<Host as Backend>::OwnedBuf, i64>;

type Ct = CKKSCiphertextOwned<ServerBackend>;
type Pt = CKKSPlaintextOwned<ServerBackend>;
type Context = BootstrappingContextPrepared<ServerBackend, f64>;
