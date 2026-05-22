//! Backend handle and module initialisation for [`NTT126Ifma`](super::NTT126Ifma).
//!
//! - [`NTT126IfmaHandle`]: the opaque handle stored inside a `Module<NTT126Ifma>`,
//!   holding precomputed NTT and iNTT twiddle-factor tables and multiply-accumulate metadata.
//! - The [`Backend`] trait implementation, which defines scalar types and the
//!   handle destruction path.
//! - [`module_new`]: constructor used by the OEP `HalImpl::new` shim.

use std::ptr::NonNull;

use crate::NTT126Ifma;
use crate::ntt126_ifma::{
    bbc_meta::Bbc126IfmaMeta,
    primes::Primes42,
    tables::{Ntt126IfmaTable, Ntt126IfmaTableInv},
};
use poulpy_cpu_ref::reference::ntt120::types::Q120bScalar;
use poulpy_hal::{
    alloc_aligned, assert_alignment,
    layouts::{Backend, Module},
};

/// Opaque handle for the [`NTT126Ifma`](super::NTT126Ifma) backend.
///
/// Holds precomputed twiddle-factor tables for the forward NTT and inverse NTT
/// of size `n`, and the lazy-accumulation metadata for IFMA prep-format
/// products.
///
/// This struct is heap-allocated during module creation and freed when the
/// `Module<NTT126Ifma>` is dropped (via [`Backend::destroy`]).
#[repr(C)]
pub struct NTT126IfmaHandle {
    pub(crate) table_ntt: Ntt126IfmaTable<Primes42>,
    pub(crate) table_intt: Ntt126IfmaTableInv<Primes42>,
    pub(crate) meta_bbc: Bbc126IfmaMeta<Primes42>,
}

impl Backend for NTT126Ifma {
    type ScalarPrep = Q120bScalar;
    type ScalarBig = i128;
    type OwnedBuf = Vec<u8>;
    type BufRef<'a> = &'a [u8];
    type BufMut<'a> = &'a mut [u8];
    type Handle = NTT126IfmaHandle;
    type Location = poulpy_hal::layouts::Host;
    fn alloc_bytes(len: usize) -> Self::OwnedBuf {
        alloc_aligned::<u8>(len)
    }
    fn from_host_bytes(bytes: &[u8]) -> Self::OwnedBuf {
        let mut buf = alloc_aligned::<u8>(bytes.len());
        buf.copy_from_slice(bytes);
        buf
    }
    fn from_bytes(bytes: Vec<u8>) -> Self::OwnedBuf {
        assert_alignment(bytes.as_ptr());
        bytes
    }
    fn to_host_bytes(buf: &Self::OwnedBuf) -> Vec<u8> {
        buf.clone()
    }
    fn copy_to_host(buf: &Self::OwnedBuf, dst: &mut [u8]) {
        assert!(buf.len() >= dst.len());
        dst.copy_from_slice(&buf[..dst.len()]);
    }
    fn copy_from_host(buf: &mut Self::OwnedBuf, src: &[u8]) {
        assert!(buf.len() >= src.len());
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
        &buf[..]
    }
    fn view_mut_ref<'a, 'b>(buf: &'a mut Self::BufMut<'b>) -> Self::BufMut<'a>
    where
        Self: 'b,
    {
        &mut buf[..]
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

    fn bytes_of_vmp_pmat(n: usize, rows: usize, cols_in: usize, cols_out: usize, size: usize) -> usize {
        // Prime-major layout: 3 planes (one per CRT prime), no padding lane.
        // Per coefficient: 3 × u64 instead of the default 4 × u64.
        n * rows * cols_in * cols_out * size * 3 * size_of::<u64>()
    }

    unsafe fn destroy(handle: NonNull<Self::Handle>) {
        unsafe {
            drop(Box::from_raw(handle.as_ptr()));
        }
    }
}

/// Borrow the backend handle stored inside the module.
///
/// # Safety considerations
///
/// `Module<NTT126Ifma>` owns a `NonNull<NTT126IfmaHandle>` pointing to a
/// fully-initialised, heap-allocated handle (set up by [`module_new`]).
/// The borrow lives for `&Module<NTT126Ifma>` and is sound under the
/// no-aliasing assumption documented on `Module`.
#[inline(always)]
pub(crate) fn handle(module: &Module<NTT126Ifma>) -> &NTT126IfmaHandle {
    unsafe { &*module.ptr() }
}

/// Verify that the host CPU supports the AVX-512-IFMA family at runtime.
///
/// # Panics
///
/// Panics if any required feature is missing.
fn assert_runtime_support() {
    #[cfg(target_arch = "x86_64")]
    {
        if !std::arch::is_x86_feature_detected!("avx512f") {
            panic!("NTT126Ifma requires x86_64 with AVX512-F support");
        }
        if !std::arch::is_x86_feature_detected!("avx512ifma") {
            panic!("NTT126Ifma requires x86_64 with AVX512-IFMA support");
        }
        if !std::arch::is_x86_feature_detected!("avx512vl") {
            panic!("NTT126Ifma requires x86_64 with AVX512-VL support");
        }
        if !std::arch::is_x86_feature_detected!("bmi2") {
            panic!("NTT126Ifma requires x86_64 with BMI2 support");
        }
        if !std::arch::is_x86_feature_detected!("adx") {
            panic!("NTT126Ifma requires x86_64 with ADX support");
        }
    }

    #[cfg(not(target_arch = "x86_64"))]
    panic!("NTT126Ifma requires x86_64 with AVX512-F + AVX512-IFMA + AVX512-VL + BMI2 + ADX support");
}

/// Allocate a fully-initialised `Module<NTT126Ifma>` of ring dimension `n`.
///
/// Verifies AVX-512-IFMA availability at runtime, then heap-allocates a
/// [`NTT126IfmaHandle`] containing the forward / inverse NTT tables and the
/// BBC metadata.
pub(crate) fn module_new(n: u64) -> Module<NTT126Ifma> {
    assert_runtime_support();
    assert!(n >= 8, "NTT126Ifma requires n >= 8, got {n}");
    let handle = NTT126IfmaHandle {
        table_ntt: Ntt126IfmaTable::new(n as usize),
        table_intt: Ntt126IfmaTableInv::new(n as usize),
        meta_bbc: Bbc126IfmaMeta::new(),
    };
    let ptr: NonNull<NTT126IfmaHandle> = NonNull::from(Box::leak(Box::new(handle)));
    unsafe { Module::from_nonnull(ptr, n) }
}
