//! Backend handle and module initialisation for [`FFT64Neon`](super::FFT64Neon).

use crate::FFT64NeonBackend;
use poulpy_cpu_ref::ring::CpuRing;

use std::ptr::NonNull;

use poulpy_cpu_ref::reference::fft64::module::{FFT64HandleFactory, FFT64Plan, FFT64PlanSet, FFTHandleProvider};
use poulpy_hal::{
    AlignedBuf, alloc_aligned,
    layouts::{Backend, Host},
};

/// Opaque handle for the [`FFT64Neon`](super::FFT64Neon) backend.
/// Holds precomputed twiddle-factor tables for the forward FFT and inverse FFT
/// of size `m = n / 2`, where `n` is the ring dimension passed to
/// [`Module::new`](poulpy_hal::api::ModuleNew::new).
#[repr(C)]
pub struct FFT64NeonHandle<R: CpuRing = poulpy_cpu_ref::ring::Standard> {
    ring_plans: FFT64PlanSet<f64, R>,
    table_cache: ::poulpy_cpu_ref::table_cache::ModuleTableCache,
}

impl<R: CpuRing> poulpy_hal::execution::ScratchWorkers for FFT64NeonBackend<R> {}

impl<R: CpuRing> Backend for FFT64NeonBackend<R> {
    const MAX_BASE2K: usize = <poulpy_cpu_ref::FFT64Ref as Backend>::MAX_BASE2K;
    const DFT_LIMBS_CONTIGUOUS: bool = true;

    type TaskExecutor = poulpy_hal::execution::SerialTaskExecutor;
    type DftWord = f64;
    type ZnxWord = i64;
    type BigWord = i64;
    type OwnedBuf = AlignedBuf;
    type BufRef<'a> = &'a [u8];
    type BufMut<'a> = &'a mut [u8];
    type Handle = FFT64NeonHandle<R>;
    type Location = Host;
    const CYCLOTOMIC_ORDER_FACTOR: i64 = if R::IS_CI { 4 } else { 2 };

    fn alloc_bytes(len: usize) -> Self::OwnedBuf {
        alloc_aligned::<u8>(len)
    }
    fn from_host_bytes(bytes: &[u8]) -> Self::OwnedBuf {
        AlignedBuf::from(bytes)
    }
    fn to_host_bytes(buf: &Self::OwnedBuf) -> Vec<u8> {
        buf.to_vec()
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
        assert!(buf.len() >= dst.len());
        dst.copy_from_slice(&buf[..dst.len()]);
    }
    fn copy_host_to_view(buf: &mut Self::BufMut<'_>, src: &[u8]) {
        assert!(buf.len() >= src.len());
        let src_len = src.len();
        buf[..src_len].copy_from_slice(src);
        buf[src_len..].fill(0);
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
/// The returned handle must be fully initialized for `n`.
/// NEON/ASIMD is part of the AArch64 baseline; the runtime check is a no-op.
unsafe impl<R: CpuRing> FFT64HandleFactory for FFT64NeonHandle<R> {
    fn create_fft64_handle(n: usize) -> Self {
        FFT64NeonHandle::<R> {
            table_cache: Default::default(),
            ring_plans: FFT64PlanSet::new(n),
        }
    }
}

unsafe impl<R: CpuRing> FFTHandleProvider<f64> for FFT64NeonHandle<R> {
    type Ring = R;
    fn get_fft_plan(&self, n: usize) -> &FFT64Plan<f64, R> {
        self.ring_plans.for_ring(n)
    }
}

unsafe impl<R: CpuRing> ::poulpy_cpu_ref::table_cache::ModuleTableCacheProvider for FFT64NeonHandle<R> {
    fn module_plan_cache(&self) -> &::poulpy_cpu_ref::table_cache::ModuleTableCache {
        &self.table_cache
    }
}
