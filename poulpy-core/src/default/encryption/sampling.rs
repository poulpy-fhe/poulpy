//! Host-side sampling of long-lived secret keys, and the Gaussian noise descriptor.
//!
//! Secret keys are drawn on the host from a [`Source`](poulpy_hal::source::Source)
//! with the `ScalarZnx::fill_*` methods and uploaded once with
//! [`Backend::copy_host_to_view`]. Anything sampled during encryption never
//! takes that path: it goes through [`SamplingImpl`](crate::oep::SamplingImpl)
//! and is sampled in place by the backend — the ephemeral secret of public-key
//! encryption as [`ScalarZnxFillDistribution`](crate::ScalarZnxFillDistribution),
//! the Gaussian noise as [`VecZnxAddNormal`](crate::VecZnxAddNormal) and
//! [`VecZnxBigAddNormal`](crate::VecZnxBigAddNormal), so that encryption never
//! round-trips through the host. [`NoiseInfos`] is the descriptor the two noise
//! operations take.

use anyhow::Result;
use poulpy_hal::layouts::{Backend, DataView, DataViewMut, HostBytesBackend, ScalarZnx, ScalarZnxBackendMut, ZnxWord};

/// Parameters of the discrete Gaussian error added at torus precision `2^-k`.
#[derive(Clone, Copy, Debug)]
pub struct NoiseInfos {
    pub k: usize,
    pub sigma: f64,
    pub bound: f64,
}

impl NoiseInfos {
    pub fn new(k: usize, sigma: f64, bound: f64) -> Result<Self> {
        anyhow::ensure!(sigma.is_sign_positive(), "sigma must be positive");
        anyhow::ensure!(sigma >= 1.0, "sigma must be greater or equal to 1");
        anyhow::ensure!(bound >= sigma, "bound: {bound} must be greater or equal to sigma: {sigma}");
        Ok(Self { k, sigma, bound })
    }

    /// Target limb and the number of unused low bits it holds.
    pub fn target_limb_and_shift(&self, base2k: usize) -> (usize, u32) {
        let limb: usize = self.k.div_ceil(base2k) - 1;
        (limb, ((limb + 1) * base2k - self.k) as u32)
    }
}

/// Zeroed host `ScalarZnx` with the word type of the backend it will be uploaded to.
pub fn scalar_znx_host_zeroed<W: ZnxWord>(n: usize, cols: usize) -> ScalarZnx<Vec<u8>, W> {
    ScalarZnx::from_data(
        HostBytesBackend::alloc_zeroed_bytes(ScalarZnx::<Vec<u8>, W>::bytes_of(n, cols)),
        n,
        cols,
    )
}

/// Uploads a host `ScalarZnx` into a backend view of the same `(n, cols)`.
pub fn upload_scalar_znx<BE: Backend>(view: &mut ScalarZnxBackendMut<'_, BE>, host: &ScalarZnx<Vec<u8>, BE::ZnxWord>) {
    assert_eq!(
        (view.n(), view.cols()),
        (host.n(), host.cols()),
        "upload_scalar_znx: shape mismatch"
    );
    // Host allocations and owned backend buffers are padded to the allocator's
    // alignment, so neither side is exactly `bytes_of` long; `copy_host_to_view`
    // wants equal lengths. Copy exactly the shape's bytes on both sides.
    let bytes = BE::bytes_of_scalar_znx(view.n(), view.cols());
    assert!(
        host.data().len() >= bytes,
        "upload_scalar_znx: host buffer shorter than its shape"
    );
    let mut dst = BE::region_mut_ref(view.data_mut(), 0, bytes);
    BE::copy_host_to_view(&mut dst, &host.data()[..bytes]);
}
