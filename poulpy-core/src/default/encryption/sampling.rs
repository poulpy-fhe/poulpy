//! Host-side secret-key sampling.
//!
//! Secrets are drawn on the host from a [`Source`](poulpy_hal::source::Source)
//! with the `ScalarZnx::fill_*` methods and uploaded with
//! [`Backend::copy_host_to_view`]. Gaussian noise is not sampled here: it is
//! a HAL operation (`vec_znx_add_normal_source`, `vec_znx_big_add_normal`) so
//! that a backend samples it in place and encryption never round-trips
//! through the host.

use poulpy_hal::layouts::{Backend, DataView, DataViewMut, HostBytesBackend, ScalarZnx, ScalarZnxBackendMut, ZnxWord};

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
