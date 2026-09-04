//! Open Extension Points (OEP) for backend crates.
//!
//! This module defines the `unsafe` backend extension layer as a set of
//! per-family traits. Backend crates implement only the families they own.
//!
//! All extension points in this module are `unsafe` because implementations
//! must uphold the backend safety contract.

mod hal_impl;

pub use hal_impl::*;

use crate::layouts::{Backend, CoeffNormalized, CoeffUnnormalized, CoefficientState, Data, VecZnx, VecZnxViewMut, ZnxWord};

/// Backend-implementor extension point: sets a container's
/// [`CoefficientState`] directly.
///
/// The safe layer only ever relaxes a label (`into_unnormalized`, `into_state`)
/// or earns [`CoeffNormalized`] back through a real pass (`normalize`); scheme code
/// has no other transition. A fused backend kernel, however, may prove the
/// digit bound by construction (e.g. an accumulator of `k` normalized limbs
/// stays within the DFT precision budget) and legitimately skip the pass before
/// feeding a `CoeffNormalized`-typed primitive. [`Self::set_normalized`] is that
/// override. The trait lives in `oep` because, like every extension point here,
/// it is reserved for backend implementations: it must not appear outside a
/// backend crate.
///
/// Implemented for every state-carrying container ([`VecZnx`],
/// [`VecZnxViewMut`], and the `GLWE`/`CKKSCiphertext` families through their
/// crates' `oep` modules).
pub trait SetNormalizationState: Sized {
    /// `Self` with its state replaced by `T`.
    type WithState<T: CoefficientState>;

    /// Relabels `self` as [`CoeffUnnormalized`].
    ///
    /// Safe: normalized digits are valid unnormalized digits, so this only
    /// relaxes the label (the trait-generic twin of `into_unnormalized`).
    fn set_unnormalized(self) -> Self::WithState<CoeffUnnormalized>;

    /// Relabels `self` as [`CoeffNormalized`] with no normalization pass.
    ///
    /// # Safety
    ///
    /// The caller asserts that every digit already satisfies the bound the
    /// consuming kernels rely on. This is a semantic contract, not a
    /// memory-safety one, but it is `unsafe` for the same reason the OEP traits
    /// are: violating it silently corrupts results (DFT precision overflow).
    unsafe fn set_normalized(self) -> Self::WithState<CoeffNormalized>;
}

impl<D: Data, W: ZnxWord, S: CoefficientState> SetNormalizationState for VecZnx<D, W, S> {
    type WithState<T: CoefficientState> = VecZnx<D, W, T>;

    fn set_unnormalized(self) -> VecZnx<D, W, CoeffUnnormalized> {
        self.relabel_unchecked()
    }

    unsafe fn set_normalized(self) -> VecZnx<D, W, CoeffNormalized> {
        self.relabel_unchecked()
    }
}

impl<'a, B: Backend + 'a, S: CoefficientState> SetNormalizationState for VecZnxViewMut<'a, B, S> {
    type WithState<T: CoefficientState> = VecZnxViewMut<'a, B, T>;

    fn set_unnormalized(self) -> VecZnxViewMut<'a, B, CoeffUnnormalized> {
        VecZnxViewMut::from_inner(self.into_inner().relabel_unchecked())
    }

    unsafe fn set_normalized(self) -> VecZnxViewMut<'a, B, CoeffNormalized> {
        VecZnxViewMut::from_inner(self.into_inner().relabel_unchecked())
    }
}

/// Backend-implementor extension point: re-wraps a container's storage while keeping
/// its shape and [`CoefficientState`].
///
/// For kernel plumbing that re-expresses the *same digits* through different storage:
/// a delegating backend (e.g. a Rayon wrapper) reborrowing a view as its base
/// backend's view type, or a host-slice view of a backend buffer. The state travels
/// with the value it is read from, so the new storage must be (a reborrow of) the
/// receiver's own storage or a byte-for-byte copy of its digits. Like every extension
/// point here it is reserved for backend implementations and must not appear in
/// scheme code.
#[allow(clippy::wrong_self_convention)] // `from_data_like` is the established reborrow name in this workspace
pub trait ReborrowData: Sized {
    /// The storage type of the receiver.
    type Data;
    /// The receiver re-wrapped over storage `D2`.
    type WithData<D2: Data>;

    /// Wraps `data` with the receiver's shape and state ([`Self::from_data_like`]).
    fn from_data_like<D2: Data>(&self, data: D2) -> Self::WithData<D2>;

    /// Mutable sibling of [`Self::from_data_like`]: re-wraps storage extracted from
    /// the receiver's own data (a reborrow, an inner buffer) under the same shape and
    /// state.
    fn map_data_mut<'a, D2: Data>(&'a mut self, f: impl FnOnce(&'a mut Self::Data) -> D2) -> Self::WithData<D2>;
}

impl<D: Data, W: ZnxWord, S: CoefficientState> ReborrowData for VecZnx<D, W, S> {
    type Data = D;
    type WithData<D2: Data> = VecZnx<D2, W, S>;

    fn from_data_like<D2: Data>(&self, data: D2) -> VecZnx<D2, W, S> {
        VecZnx::from_data_with_state(data, self.n(), self.cols(), self.size())
    }

    fn map_data_mut<'a, D2: Data>(&'a mut self, f: impl FnOnce(&'a mut D) -> D2) -> VecZnx<D2, W, S> {
        let (n, cols, size) = (self.n(), self.cols(), self.size());
        VecZnx::from_data_with_state(f(&mut self.data), n, cols, size)
    }
}

/// Sealed kernel capability (spec §9.1): a state-erased mutable host view over typed
/// storage. Safe mutable word access ([`crate::layouts::ZnxViewMut`] /
/// [`crate::layouts::DataViewMut`]) exists only for the weakest arithmetic state, so a
/// kernel that writes a destination under a declared postcondition takes this erased
/// view at its entry point and works through the ordinary view methods on it.
pub trait KernelWordsMut {
    /// The state-erased mutable host view over the receiver's storage.
    type Erased<'a>
    where
        Self: 'a;

    /// Returns the erased view.
    ///
    /// # Safety
    ///
    /// The receiver keeps its state label while its bytes change through the returned
    /// view. The caller (a backend kernel or its test harness) asserts that by the
    /// time the borrow ends, every word it wrote satisfies the invariant of the
    /// receiver's original state label: the operation's declared per-write/final
    /// postcondition covers the region it touched (spec invariants 7 and 12).
    /// Violating it silently corrupts results.
    unsafe fn kernel_words_mut(&mut self) -> Self::Erased<'_>;
}

impl<D: crate::layouts::HostDataMut, W: ZnxWord, S: CoefficientState> KernelWordsMut for VecZnx<D, W, S> {
    type Erased<'a>
        = VecZnx<&'a mut [u8], W, CoeffUnnormalized>
    where
        Self: 'a;

    unsafe fn kernel_words_mut(&mut self) -> VecZnx<&mut [u8], W, CoeffUnnormalized> {
        let (n, cols, size) = (self.n(), self.cols(), self.size());
        VecZnx::from_data_with_state(self.data.as_mut(), n, cols, size)
    }
}
