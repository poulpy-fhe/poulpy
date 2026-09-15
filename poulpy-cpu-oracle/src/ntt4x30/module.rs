//! NTT backend storage and transform tables.

use std::ptr::NonNull;

use poulpy_hal::{
    alloc_aligned, assert_alignment,
    layouts::{Backend, Host},
};

use crate::reference::ntt4x30::{
    primes::Primes30,
    types::Q120bScalar,
    vec_znx_dft::{NttHandleFactory, NttHandleProvider, NttPlan, NttPlanSet},
};

use crate::NTT4x30Oracle;

/// Transform plans and cached scheme tables owned by a module.
#[repr(C)]
pub struct NTT4x30OracleHandle {
    ring_plans: NttPlanSet<Primes30>,
    table_cache: crate::table_cache::ModuleTableCache,
}

impl poulpy_hal::execution::ScratchWorkers for NTT4x30Oracle {}

impl Backend for NTT4x30Oracle {
    const DFT_IS_EXACT: bool = true;

    type TaskExecutor = poulpy_hal::execution::SerialTaskExecutor;
    type DftWord = Q120bScalar;
    type ZnxWord = i64;
    type BigWord = i128;
    type OwnedBuf = Vec<u8>;
    type BufRef<'a> = &'a [u8];
    type BufMut<'a> = &'a mut [u8];
    type Handle = NTT4x30OracleHandle;
    type Location = Host;
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

    fn len_bytes_ref(buf: &Self::BufRef<'_>) -> usize {
        buf.len()
    }

    fn len_bytes_mut(buf: &Self::BufMut<'_>) -> usize {
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

    unsafe fn destroy(handle: NonNull<Self::Handle>) {
        unsafe {
            drop(Box::from_raw(handle.as_ptr()));
        }
    }
}

/// # Safety
///
/// The returned handle must be fully initialized for `n`.
unsafe impl NttHandleFactory for NTT4x30OracleHandle {
    fn create_ntt_handle(n: usize) -> Self {
        NTT4x30OracleHandle {
            table_cache: Default::default(),
            ring_plans: NttPlanSet::new(n),
        }
    }
}

/// # Safety
///
/// The returned references are valid for the lifetime of `&self`.
/// All fields are fully initialised by the [`NttHandleFactory`] impl above.
unsafe impl NttHandleProvider for NTT4x30OracleHandle {
    fn get_ntt_plan(&self, n: usize) -> &NttPlan<Primes30> {
        self.ring_plans.for_ring(n)
    }
}

unsafe impl crate::table_cache::ModuleTableCacheProvider for NTT4x30OracleHandle {
    fn module_plan_cache(&self) -> &crate::table_cache::ModuleTableCache {
        &self.table_cache
    }
}
