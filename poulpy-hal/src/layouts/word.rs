//! Word types: the byte-layout contracts of polynomial coefficient domains.
//!
//! A word type names a **byte-layout convention**: element packing, lane
//! count, prime set, ordering, and reduction form. The word alone does NOT
//! grant cross-backend interchangeability — DFT/big-domain containers are
//! additionally keyed by the producing backend, and buffers move between
//! backends only through the explicit `into_backend` re-tag, guarded by the
//! per-container `*LayoutCompatible` markers (see
//! [`VecZnxDftLayoutCompatible`](crate::layouts::VecZnxDftLayoutCompatible)
//! and siblings). Sharing a word is necessary but not sufficient for such a
//! marker: the NTT4x30 backends share `Q120bScalar` yet pack `VmpPMat`
//! differently, which is precisely why interchangeability is opt-in per
//! container family.
//!
//! Word types are a sizing + identity contract, not necessarily an element
//! view: [`Backend::bytes_of_vmp_pmat`](crate::layouts::Backend::bytes_of_vmp_pmat)
//! and friends remain authoritative for total buffer sizes and may be
//! overridden by backends whose packed representation diverges from
//! `n * cols * size * size_of::<Word>()`.
//!
//! **Alignment:** a buffer backing a word-keyed container must be aligned to
//! `align_of::<W>()`. Every allocation path in this crate goes through
//! [`alloc_aligned`](crate::alloc_aligned) (aligned to
//! [`DEFAULTALIGN`](crate::DEFAULTALIGN), which exceeds the alignment of all
//! word types); the requirement exists for external `Data` providers and is
//! checked by a `debug_assert` in
//! [`ZnxView::as_ptr`](crate::layouts::ZnxView::as_ptr).

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

    /// Builds a word from a signed integer, truncating to [`Self::BITS`].
    ///
    /// This is what lets coefficient sampling (`FillUniform`, the ternary and
    /// binary secret fills) be written once against any word instead of being
    /// pinned to `i64`. Callers are responsible for supplying a value that
    /// already fits the word; truncation is defined but not meaningful.
    fn from_i64(value: i64) -> Self;
}

impl ZnxWord for i64 {
    const BITS: usize = 64;

    #[inline(always)]
    fn from_i64(value: i64) -> Self {
        value
    }
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
pub trait DftWord: Pod + Copy + Zero + Display + Debug + Send + Sync + PartialEq + 'static {
    /// Whether transform-domain arithmetic is exact within its documented
    /// operand bounds. CRT/NTT words are exact; floating-point FFT words are
    /// approximate.
    const IS_EXACT: bool = false;
}

/// Split-complex FFT representation over `f64` (spqlios ordering).
impl DftWord for f64 {}

/// Host-side placeholder word used by
/// [`HostBytesBackend`](crate::layouts::HostBytesBackend).
impl DftWord for i64 {}
