//! Open Extension Points (OEP) for backend crates.
//!
//! This module defines the `unsafe` backend extension layer as a set of
//! per-family traits. Backend crates implement only the families they own.
//!
//! All extension points in this module are `unsafe` because implementations
//! must uphold the backend safety contract.

mod hal_impl;

pub use hal_impl::*;

use crate::layouts::{Backend, Data, NormalizationState, Normalized, Unnormalized, VecZnx, VecZnxViewMut, ZnxWord};

/// Backend-implementor extension point: sets a container's
/// [`NormalizationState`] directly.
///
/// The safe layer only ever relaxes a label (`into_unnormalized`, `into_state`)
/// or earns [`Normalized`] back through a real pass (`normalize`); scheme code
/// has no other transition. A fused backend kernel, however, may prove the
/// digit bound by construction (e.g. an accumulator of `k` normalized limbs
/// stays within the DFT precision budget) and legitimately skip the pass before
/// feeding a `Normalized`-typed primitive. [`Self::set_normalized`] is that
/// override. The trait lives in `oep` because, like every extension point here,
/// it is reserved for backend implementations: it must not appear outside a
/// backend crate.
///
/// Implemented for every state-carrying container ([`VecZnx`],
/// [`VecZnxViewMut`], and the `GLWE`/`CKKSCiphertext` families through their
/// crates' `oep` modules).
pub trait SetNormalizationState: Sized {
    /// `Self` with its state replaced by `T`.
    type WithState<T: NormalizationState>;

    /// Relabels `self` as [`Unnormalized`].
    ///
    /// Safe: normalized digits are valid unnormalized digits, so this only
    /// relaxes the label (the trait-generic twin of `into_unnormalized`).
    fn set_unnormalized(self) -> Self::WithState<Unnormalized>;

    /// Relabels `self` as [`Normalized`] with no normalization pass.
    ///
    /// # Safety
    ///
    /// The caller asserts that every digit already satisfies the bound the
    /// consuming kernels rely on. This is a semantic contract, not a
    /// memory-safety one, but it is `unsafe` for the same reason the OEP traits
    /// are: violating it silently corrupts results (DFT precision overflow).
    unsafe fn set_normalized(self) -> Self::WithState<Normalized>;
}

impl<D: Data, W: ZnxWord, S: NormalizationState> SetNormalizationState for VecZnx<D, W, S> {
    type WithState<T: NormalizationState> = VecZnx<D, W, T>;

    fn set_unnormalized(self) -> VecZnx<D, W, Unnormalized> {
        self.relabel_unchecked()
    }

    unsafe fn set_normalized(self) -> VecZnx<D, W, Normalized> {
        self.relabel_unchecked()
    }
}

impl<'a, B: Backend + 'a, S: NormalizationState> SetNormalizationState for VecZnxViewMut<'a, B, S> {
    type WithState<T: NormalizationState> = VecZnxViewMut<'a, B, T>;

    fn set_unnormalized(self) -> VecZnxViewMut<'a, B, Unnormalized> {
        VecZnxViewMut::from_inner(self.into_inner().relabel_unchecked())
    }

    unsafe fn set_normalized(self) -> VecZnxViewMut<'a, B, Normalized> {
        VecZnxViewMut::from_inner(self.into_inner().relabel_unchecked())
    }
}

/// Backend-implementor extension point: re-wraps `data` with the shape and
/// [`NormalizationState`] of `a`.
///
/// For kernel plumbing that re-expresses the *same digits* through different
/// storage: a delegating backend (e.g. a Rayon wrapper) reborrowing a view as
/// its base backend's view type, or a host-slice view of a backend buffer. The
/// state travels with the value it is read from, so `data` must be (a reborrow
/// of) the storage of `a` itself or a byte-for-byte copy of its digits. Like
/// every extension point here it is reserved for backend implementations and
/// must not appear in scheme code.
pub fn vec_znx_from_data_like<D: Data, D2: Data, W: ZnxWord, S: NormalizationState>(
    a: &VecZnx<D, W, S>,
    data: D2,
) -> VecZnx<D2, W, S> {
    VecZnx::from_data_with_state(data, a.n(), a.cols(), a.size())
}

/// Mutable sibling of [`vec_znx_from_data_like`]: re-wraps storage extracted
/// from `a`'s own data (a reborrow, an inner buffer) under the same shape and
/// [`NormalizationState`]. Backend implementations only.
pub fn vec_znx_map_data_mut<'a, D: Data, D2: Data, W: ZnxWord, S: NormalizationState>(
    a: &'a mut VecZnx<D, W, S>,
    f: impl FnOnce(&'a mut D) -> D2,
) -> VecZnx<D2, W, S> {
    let (n, cols, size) = (a.n(), a.cols(), a.size());
    VecZnx::from_data_with_state(f(&mut a.data), n, cols, size)
}
