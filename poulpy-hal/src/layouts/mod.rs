//! Data layout types and trait definitions for the hardware abstraction layer.
//!
//! This module aggregates all layout-related types and re-exports them from
//! their respective sub-modules, including convolution kernels, matrix and
//! vector representations over polynomial rings, serialization support,
//! statistical utilities, and scratch-space management.
//!
//! It also defines the shared storage trait aliases used throughout the crate.
//! `Data` models backend-owned storage in the abstract, while
//! `HostDataRef`/`HostDataMut` capture host-byte-readable buffers for the
//! portions of the API that still require direct byte access.

mod convolution;
mod encoding;
mod mat_znx;
mod module;
mod scalar_znx;
mod scratch;
mod scratch_views;
mod serialization;
mod stats;
mod svp_ppol;
mod vec_znx;
mod vec_znx_big;
mod vec_znx_dft;
mod vmp_pmat;
mod znx_base;

pub use convolution::*;
pub use mat_znx::*;
pub use module::*;
pub use scalar_znx::*;
pub use scratch::*;
pub use scratch_views::*;
pub use serialization::*;
pub use stats::*;
pub use svp_ppol::*;
pub use vec_znx::*;
pub use vec_znx_big::*;
pub use vec_znx_dft::*;
pub use vmp_pmat::*;
pub use znx_base::*;

use anyhow::Result;
use std::ptr::NonNull;

use crate::oep::HalModuleImpl;

/// Base trait alias for all data containers.
///
/// Requires equality comparison ([`PartialEq`], [`Eq`]), a known size at
/// compile time ([`Sized`]), and a default value ([`Default`]). Every
/// layout type that holds raw data must satisfy at least this bound.
pub trait Data = PartialEq + Eq + Sized + Default;

/// Trait alias for read-only host-byte-accessible containers.
///
/// Extends [`Data`] with byte-level shared access via [`AsRef<[u8]>`] and
/// thread-safe sharing via [`Sync`]. Types satisfying this bound can be
/// borrowed immutably and read across threads.
pub trait HostDataRef = Data + AsRef<[u8]> + Sync;

/// Trait alias for mutable host-byte-accessible containers.
///
/// Extends [`HostDataRef`] with byte-level mutable access via [`AsMut<[u8]>`]
/// and cross-thread transfer via [`Send`]. Types satisfying this bound
/// support in-place modification and can be moved between threads.
pub trait HostDataMut = HostDataRef + AsMut<[u8]> + Send;

mod private {
    pub trait Sealed {}
}

/// Sealed trait identifying the residency of a [`Backend`]'s buffers.
///
/// Implemented only by [`Host`] and [`Device`]. Each [`Backend`] declares
/// its residency via its [`Backend::Location`] associated type, which lets
/// generic code discriminate host- and device-resident backends at the
/// type level.
pub trait Location: private::Sealed {}

/// Marker type for host-resident buffers.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Host;

/// Marker type for device-resident buffers.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Device;

impl private::Sealed for Host {}
impl private::Sealed for Device {}
impl Location for Host {}
impl Location for Device {}

/// Convenience marker for host-resident backends.
pub trait HostBackend: Backend<Location = Host> {}
impl<BE: Backend<Location = Host>> HostBackend for BE {}

/// Convenience marker for host-resident backends whose borrowed views are directly readable and writable as host bytes.
pub trait HostVisibleBackend: HostBackend
where
    for<'a> Self::BufRef<'a>: AsRef<[u8]>,
    for<'a> Self::BufMut<'a>: AsRef<[u8]> + AsMut<[u8]>,
{
}

impl<BE> HostVisibleBackend for BE
where
    BE: HostBackend,
    for<'a> BE::BufRef<'a>: AsRef<[u8]>,
    for<'a> BE::BufMut<'a>: AsRef<[u8]> + AsMut<[u8]>,
{
}

/// Minimal host-resident backend used as the default backend adapter for
/// host-visible byte-slice views in generic helper code.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct HostBytesBackend;

impl Backend for HostBytesBackend {
    type ScalarBig = i128;
    type ScalarPrep = i64;
    type OwnedBuf = Vec<u8>;
    type BufRef<'a> = &'a [u8];
    type BufMut<'a> = &'a mut [u8];
    type Handle = ();
    type Location = Host;

    fn alloc_bytes(len: usize) -> Self::OwnedBuf {
        crate::alloc_aligned::<u8>(len)
    }

    fn alloc_zeroed_bytes(len: usize) -> Self::OwnedBuf {
        crate::alloc_aligned::<u8>(len)
    }

    fn from_host_bytes(bytes: &[u8]) -> Self::OwnedBuf {
        let mut out = crate::alloc_aligned::<u8>(bytes.len());
        out.copy_from_slice(bytes);
        out
    }

    fn from_bytes(bytes: Vec<u8>) -> Self::OwnedBuf {
        if crate::is_aligned(bytes.as_ptr()) {
            bytes
        } else {
            let mut out = crate::alloc_aligned::<u8>(bytes.len());
            out.copy_from_slice(&bytes);
            out
        }
    }

    fn to_host_bytes(buf: &Self::OwnedBuf) -> Vec<u8> {
        buf.clone()
    }

    fn copy_to_host(buf: &Self::OwnedBuf, dst: &mut [u8]) {
        assert!(
            buf.len() >= dst.len(),
            "backend buffer length {} is smaller than destination host slice length {}",
            buf.len(),
            dst.len()
        );
        dst.copy_from_slice(&buf[..dst.len()]);
    }

    fn copy_from_host(buf: &mut Self::OwnedBuf, src: &[u8]) {
        assert!(
            buf.len() >= src.len(),
            "backend buffer length {} is smaller than source host slice length {}",
            buf.len(),
            src.len()
        );
        let src_len = src.len();
        buf[..src_len].copy_from_slice(src);
        buf[src_len..].fill(0);
    }

    fn len_bytes(buf: &Self::OwnedBuf) -> usize {
        buf.len()
    }

    fn view(buf: &Self::OwnedBuf) -> Self::BufRef<'_> {
        buf.as_slice()
    }

    fn view_ref<'a, 'b>(buf: &'a Self::BufRef<'b>) -> Self::BufRef<'a>
    where
        Self: 'b,
    {
        buf
    }

    fn view_ref_mut<'a, 'b>(buf: &'a Self::BufMut<'b>) -> Self::BufRef<'a>
    where
        Self: 'b,
    {
        buf
    }

    fn view_mut_ref<'a, 'b>(buf: &'a mut Self::BufMut<'b>) -> Self::BufMut<'a>
    where
        Self: 'b,
    {
        buf
    }

    fn view_mut(buf: &mut Self::OwnedBuf) -> Self::BufMut<'_> {
        buf.as_mut_slice()
    }

    fn region(buf: &Self::OwnedBuf, offset: usize, len: usize) -> Self::BufRef<'_> {
        &buf[offset..offset + len]
    }

    fn region_mut(buf: &mut Self::OwnedBuf, offset: usize, len: usize) -> Self::BufMut<'_> {
        &mut buf[offset..offset + len]
    }

    fn region_ref<'a, 'b>(buf: &'a Self::BufRef<'b>, offset: usize, len: usize) -> Self::BufRef<'a>
    where
        Self: 'b,
    {
        &buf[offset..offset + len]
    }

    fn region_ref_mut<'a, 'b>(buf: &'a Self::BufMut<'b>, offset: usize, len: usize) -> Self::BufRef<'a>
    where
        Self: 'b,
    {
        &buf[offset..offset + len]
    }

    fn region_mut_ref<'a, 'b>(buf: &'a mut Self::BufMut<'b>, offset: usize, len: usize) -> Self::BufMut<'a>
    where
        Self: 'b,
    {
        &mut buf[offset..offset + len]
    }

    unsafe fn destroy(_handle: NonNull<Self::Handle>) {}
}

unsafe impl HalModuleImpl<HostBytesBackend> for HostBytesBackend {
    fn new(n: u64) -> crate::layouts::Module<Self> {
        assert!(n.is_power_of_two(), "n must be a power of two, got {n}");
        unsafe { crate::layouts::Module::from_nonnull(NonNull::dangling(), n) }
    }
}

/// Convenience marker for device-resident backends.
pub trait DeviceBackend: Backend<Location = Device> {}
impl<BE: Backend<Location = Device>> DeviceBackend for BE {}

/// Deep-clone a borrowed layout into a fully owned variant.
///
/// Unlike the standard [`Clone`] trait, `ToOwnedDeep` is intended for
/// types that may borrow their underlying storage. Calling
/// [`to_owned_deep`](ToOwnedDeep::to_owned_deep) produces an independent
/// copy whose lifetime is not tied to the original.
pub trait ToOwnedDeep {
    type Owned;
    fn to_owned_deep(&self) -> Self::Owned;
}

/// Compute a `u64` hash digest of a layout's contents.
///
/// Provides a lightweight fingerprint suitable for fast equality checks
/// and debugging. This is **not** cryptographically secure; it is a
/// convenience mechanism for detecting whether two values hold identical
/// data without performing a full byte-by-byte comparison.
pub trait DigestU64 {
    fn digest_u64(&self) -> u64;
}

/// Backend-owned byte buffer type alias.
pub type OwnedBuf<BE> = <BE as Backend>::OwnedBuf;

/// Cross-backend buffer transfer into the destination backend `Self`.
///
/// This is intentionally destination-owned so the canonical public API can
/// hang off `Module<To>` as `upload_*` / `download_*`.
///
/// Each concrete backend pair must provide an explicit impl. Two restricted
/// blankets are provided for [`HostBytesBackend`] so that test/bench helpers
/// that use it as a staging type continue to work without boilerplate:
/// - any host `Vec<u8>` backend → `HostBytesBackend`
/// - `HostBytesBackend` → any host `Vec<u8>` backend
///
/// All other backend-to-backend transfers (e.g. `FFT64Ref` ↔ `NTT120Ref`,
/// `FFT64Ref` → `FFT64Avx`) must be implemented explicitly in the respective
/// backend crates.
pub trait TransferFrom<From: Backend>: Backend {
    /// Transfers a buffer owned by `From` into `Self`.
    fn transfer_buf(src: &From::OwnedBuf) -> Self::OwnedBuf;
}

impl<T: Backend<Location = Host, OwnedBuf = Vec<u8>>> TransferFrom<HostBytesBackend> for T {
    fn transfer_buf(src: &Vec<u8>) -> Self::OwnedBuf {
        T::from_host_bytes(src)
    }
}

/// Implement a backend marker by forwarding all storage- and handle-level
/// behavior to an existing backend.
///
/// This is useful for proof or delegating backends that want to remain a
/// distinct backend type while reusing the same owned buffer, borrowed views,
/// scalar types, and handle representation as a source backend.
#[macro_export]
macro_rules! impl_backend_from {
    ($be:ty, $from:ty) => {
        impl poulpy_hal::layouts::Backend for $be {
            type ScalarBig = <$from as poulpy_hal::layouts::Backend>::ScalarBig;
            type ScalarPrep = <$from as poulpy_hal::layouts::Backend>::ScalarPrep;
            type OwnedBuf = <$from as poulpy_hal::layouts::Backend>::OwnedBuf;
            type BufRef<'a> = <$from as poulpy_hal::layouts::Backend>::BufRef<'a>;
            type BufMut<'a> = <$from as poulpy_hal::layouts::Backend>::BufMut<'a>;
            type Handle = <$from as poulpy_hal::layouts::Backend>::Handle;
            type Location = <$from as poulpy_hal::layouts::Backend>::Location;

            fn alloc_bytes(len: usize) -> Self::OwnedBuf {
                <$from as poulpy_hal::layouts::Backend>::alloc_bytes(len)
            }

            fn alloc_zeroed_bytes(len: usize) -> Self::OwnedBuf {
                <$from as poulpy_hal::layouts::Backend>::alloc_zeroed_bytes(len)
            }

            fn from_host_bytes(bytes: &[u8]) -> Self::OwnedBuf {
                <$from as poulpy_hal::layouts::Backend>::from_host_bytes(bytes)
            }

            fn from_bytes(bytes: Vec<u8>) -> Self::OwnedBuf {
                <$from as poulpy_hal::layouts::Backend>::from_bytes(bytes)
            }

            fn to_host_bytes(buf: &Self::OwnedBuf) -> Vec<u8> {
                <$from as poulpy_hal::layouts::Backend>::to_host_bytes(buf)
            }

            fn copy_to_host(buf: &Self::OwnedBuf, dst: &mut [u8]) {
                <$from as poulpy_hal::layouts::Backend>::copy_to_host(buf, dst)
            }

            fn copy_from_host(buf: &mut Self::OwnedBuf, src: &[u8]) {
                <$from as poulpy_hal::layouts::Backend>::copy_from_host(buf, src)
            }

            fn len_bytes(buf: &Self::OwnedBuf) -> usize {
                <$from as poulpy_hal::layouts::Backend>::len_bytes(buf)
            }

            fn view(buf: &Self::OwnedBuf) -> Self::BufRef<'_> {
                <$from as poulpy_hal::layouts::Backend>::view(buf)
            }

            fn view_ref<'a, 'b>(buf: &'a Self::BufRef<'b>) -> Self::BufRef<'a>
            where
                Self: 'b,
            {
                <$from as poulpy_hal::layouts::Backend>::view_ref(buf)
            }

            fn view_ref_mut<'a, 'b>(buf: &'a Self::BufMut<'b>) -> Self::BufRef<'a>
            where
                Self: 'b,
            {
                <$from as poulpy_hal::layouts::Backend>::view_ref_mut(buf)
            }

            fn view_mut_ref<'a, 'b>(buf: &'a mut Self::BufMut<'b>) -> Self::BufMut<'a>
            where
                Self: 'b,
            {
                <$from as poulpy_hal::layouts::Backend>::view_mut_ref(buf)
            }

            fn view_mut(buf: &mut Self::OwnedBuf) -> Self::BufMut<'_> {
                <$from as poulpy_hal::layouts::Backend>::view_mut(buf)
            }

            fn region(buf: &Self::OwnedBuf, offset: usize, len: usize) -> Self::BufRef<'_> {
                <$from as poulpy_hal::layouts::Backend>::region(buf, offset, len)
            }

            fn region_mut(buf: &mut Self::OwnedBuf, offset: usize, len: usize) -> Self::BufMut<'_> {
                <$from as poulpy_hal::layouts::Backend>::region_mut(buf, offset, len)
            }

            fn region_ref<'a, 'b>(buf: &'a Self::BufRef<'b>, offset: usize, len: usize) -> Self::BufRef<'a>
            where
                Self: 'b,
            {
                <$from as poulpy_hal::layouts::Backend>::region_ref(buf, offset, len)
            }

            fn region_ref_mut<'a, 'b>(buf: &'a Self::BufMut<'b>, offset: usize, len: usize) -> Self::BufRef<'a>
            where
                Self: 'b,
            {
                <$from as poulpy_hal::layouts::Backend>::region_ref_mut(buf, offset, len)
            }

            fn region_mut_ref<'a, 'b>(buf: &'a mut Self::BufMut<'b>, offset: usize, len: usize) -> Self::BufMut<'a>
            where
                Self: 'b,
            {
                <$from as poulpy_hal::layouts::Backend>::region_mut_ref(buf, offset, len)
            }

            unsafe fn destroy(handle: std::ptr::NonNull<Self::Handle>) {
                <$from as poulpy_hal::layouts::Backend>::destroy(handle)
            }
        }
    };
}

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

    pub fn target_limb_and_scale(&self, base2k: usize) -> (usize, f64) {
        let limb: usize = self.k.div_ceil(base2k) - 1;
        let scale: f64 = (((limb + 1) * base2k - self.k) as f64).exp2();
        (limb, scale)
    }
}
