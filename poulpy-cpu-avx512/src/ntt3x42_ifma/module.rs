//! Backend handle and module initialisation for [`NTT3x42Ifma`](super::NTT3x42Ifma).
//!
//! - [`NTT3x42IfmaHandle`]: the opaque handle stored inside a `Module<NTT3x42Ifma>`,
//!   holding precomputed NTT and iNTT twiddle-factor tables and multiply-accumulate metadata.
//! - The [`Backend`] trait implementation, which defines scalar types and the
//!   handle destruction path.
//! - [`module_new`]: constructor used by the OEP `HalImpl::new` shim.

use std::ptr::NonNull;

use crate::NTT3x42Ifma;
use crate::ntt3x42_ifma::{
    bbc_meta::Bbc126IfmaMeta,
    primes::Primes42,
    tables::{Ntt3x42IfmaTable, Ntt3x42IfmaTableInv},
    types::Q126Scalar,
};
use poulpy_hal::{
    alloc_aligned, assert_alignment,
    layouts::{Backend, Module},
};

/// Opaque handle for the [`NTT3x42Ifma`](super::NTT3x42Ifma) backend.
///
/// Holds precomputed twiddle-factor tables for the forward NTT and inverse NTT
/// of size `n`, and the lazy-accumulation metadata for IFMA prep-format
/// products.
///
/// This struct is heap-allocated during module creation and freed when the
/// `Module<NTT3x42Ifma>` is dropped (via [`Backend::destroy`]).
#[repr(C)]
pub struct NTT3x42IfmaHandle {
    pub(crate) table_ntt: Ntt3x42IfmaTable<Primes42>,
    pub(crate) table_intt: Ntt3x42IfmaTableInv<Primes42>,
    pub(crate) meta_bbc: Bbc126IfmaMeta<Primes42>,
    table_cache: ::poulpy_cpu_ref::table_cache::ModuleTableCache,
}

impl Backend for NTT3x42Ifma {
    type DftWord = Q126Scalar;
    type ZnxWord = i64;
    type BigWord = i128;
    type OwnedBuf = Vec<u8>;
    type BufRef<'a> = &'a [u8];
    type BufMut<'a> = &'a mut [u8];
    type Handle = NTT3x42IfmaHandle;
    type Location = poulpy_hal::layouts::Host;
    fn alloc_bytes(len: usize) -> Self::OwnedBuf {
        alloc_aligned::<u8>(len)
    }
    fn alloc_zeroed_bytes(len: usize) -> Self::OwnedBuf {
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
    fn copy_view_to_host(buf: &Self::BufRef<'_>, dst: &mut [u8]) {
        assert_eq!(buf.len(), dst.len());
        dst.copy_from_slice(buf);
    }
    fn copy_host_to_view(buf: &mut Self::BufMut<'_>, src: &[u8]) {
        assert_eq!(buf.len(), src.len());
        buf.copy_from_slice(src);
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

    fn bytes_of_svp_ppol(n: usize, cols: usize) -> usize {
        // Three canonical residues followed by their three Harvey quotients.
        [n, cols, 6, size_of::<u64>()]
            .into_iter()
            .try_fold(1usize, usize::checked_mul)
            .expect("IFMA SvpPPol byte size overflows usize")
    }

    fn bytes_of_vec_znx_dft(n: usize, cols: usize, size: usize) -> usize {
        [n, cols, size, 2, size_of::<u64>()]
            .into_iter()
            .try_fold(1usize, usize::checked_mul)
            .expect("IFMA VecZnxDft byte size overflows usize")
    }

    fn bytes_of_vmp_pmat(n: usize, rows: usize, cols_in: usize, cols_out: usize, size: usize) -> usize {
        // Packed prime-major layout: the three 42-bit CRT residues per
        // coefficient are packed into 2 × u64 (126 of 128 bits), unpacked
        // in registers by the apply kernel.
        [n, rows, cols_in, cols_out, size, 2, size_of::<u64>()]
            .into_iter()
            .try_fold(1usize, usize::checked_mul)
            .expect("IFMA VmpPMat byte size overflows usize")
    }

    fn bytes_of_cnv_pvec_left(n: usize, cols: usize, size: usize) -> usize {
        [n, cols, size, 2, size_of::<u64>()]
            .into_iter()
            .try_fold(1usize, usize::checked_mul)
            .expect("IFMA CnvPVecL byte size overflows usize")
    }

    fn bytes_of_cnv_pvec_right(n: usize, cols: usize, size: usize) -> usize {
        [n, cols, size, 2, size_of::<u64>()]
            .into_iter()
            .try_fold(1usize, usize::checked_mul)
            .expect("IFMA CnvPVecR byte size overflows usize")
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
/// `Module<NTT3x42Ifma>` owns a `NonNull<NTT3x42IfmaHandle>` pointing to a
/// fully-initialised, heap-allocated handle (set up by [`module_new`]).
/// The borrow lives for `&Module<NTT3x42Ifma>` and is sound under the
/// no-aliasing assumption documented on `Module`.
#[inline(always)]
pub(crate) fn handle(module: &Module<NTT3x42Ifma>) -> &NTT3x42IfmaHandle {
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
            panic!("NTT3x42Ifma requires x86_64 with AVX512-F support");
        }
        if !std::arch::is_x86_feature_detected!("avx512ifma") {
            panic!("NTT3x42Ifma requires x86_64 with AVX512-IFMA support");
        }
        if !std::arch::is_x86_feature_detected!("avx512vl") {
            panic!("NTT3x42Ifma requires x86_64 with AVX512-VL support");
        }
    }

    #[cfg(not(target_arch = "x86_64"))]
    panic!("NTT3x42Ifma requires x86_64 with AVX512-F + AVX512-IFMA + AVX512-VL support");
}

/// Allocate a fully-initialised `Module<NTT3x42Ifma>` of ring dimension `n`.
///
/// Verifies AVX-512-IFMA availability at runtime, then heap-allocates a
/// [`NTT3x42IfmaHandle`] containing the forward / inverse NTT tables and the
/// BBC metadata.
pub(crate) fn module_new(n: u64) -> Module<NTT3x42Ifma> {
    assert_runtime_support();
    assert!(n >= 8, "NTT3x42Ifma requires n >= 8, got {n}");
    let handle = NTT3x42IfmaHandle {
        table_cache: Default::default(),
        table_ntt: Ntt3x42IfmaTable::new(n as usize),
        table_intt: Ntt3x42IfmaTableInv::new(n as usize),
        meta_bbc: Bbc126IfmaMeta::new(),
    };
    let ptr: NonNull<NTT3x42IfmaHandle> = NonNull::from(Box::leak(Box::new(handle)));
    unsafe { Module::from_nonnull(ptr, n) }
}

unsafe impl ::poulpy_cpu_ref::table_cache::ModuleTableCacheProvider for NTT3x42IfmaHandle {
    fn module_plan_cache(&self) -> &::poulpy_cpu_ref::table_cache::ModuleTableCache {
        &self.table_cache
    }
}
