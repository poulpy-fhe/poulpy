//! Word types: the byte-layout contracts of polynomial coefficient domains.
//!
//! A word type names a **complete byte-layout convention**: element packing,
//! lane count, prime set, ordering, and reduction form. **Type equality means
//! buffer interchangeability** — two backends whose layouts project the same
//! word type promise byte-identical buffer conventions and may exchange
//! buffers of that domain without conversion. A backend deviating in *any*
//! aspect of the convention must mint a new word type, even if the byte size
//! matches.
//!
//! Word types are a sizing + identity contract, not necessarily an element
//! view: [`Backend::bytes_of_vmp_pmat`](crate::layouts::Backend::bytes_of_vmp_pmat)
//! and friends remain authoritative for total buffer sizes and may be
//! overridden by backends whose packed representation diverges from
//! `n * cols * size * size_of::<Word>()`.

use std::fmt::{Debug, Display};

use bytemuck::Pod;
use rand_distr::num_traits::Zero;

/// Coefficient-domain word of [`VecZnx`](crate::layouts::VecZnx),
/// [`ScalarZnx`](crate::layouts::ScalarZnx) and
/// [`MatZnx`](crate::layouts::MatZnx): a signed machine integer holding one
/// base-2^k limb coefficient.
pub trait ZnxWord: Pod + Copy + Zero + Display + Debug + Send + Sync + PartialEq + 'static {
    /// Bit width of the word (64 for `i64`).
    const BITS: usize;
}

impl ZnxWord for i64 {
    const BITS: usize = 64;
}

/// Extended-precision (big) word of [`VecZnxBig`](crate::layouts::VecZnxBig):
/// holds un-normalized accumulator coefficients.
pub trait BigWord: Pod + Copy + Zero + Display + Debug + Send + Sync + PartialEq + 'static {}

impl BigWord for i64 {}
impl BigWord for i128 {}

/// DFT-domain (prepared) word of [`VecZnxDft`](crate::layouts::VecZnxDft),
/// [`SvpPPol`](crate::layouts::SvpPPol), [`VmpPMat`](crate::layouts::VmpPMat)
/// and the convolution prepared vectors.
///
/// Implementors range from plain elements (`f64` for split-complex FFT
/// backends) to packed CRT-lane blocks. There is deliberately no `Eq`/`Hash`
/// bound so that `f64` qualifies.
pub trait DftWord: Pod + Copy + Zero + Display + Debug + Send + Sync + PartialEq + 'static {}

/// Split-complex FFT representation over `f64` (spqlios ordering).
impl DftWord for f64 {}

/// Host-side placeholder word used by
/// [`HostBytesBackend`](crate::layouts::HostBytesBackend).
impl DftWord for i64 {}
