//! The backend marker and its storage contract.

use std::{marker::PhantomData, ptr::NonNull};

use poulpy_cpu_portable::kernels::ntt4x30::primes::Primes30;
use poulpy_hal::layouts::{Backend, CrtWord, Device, Ring, Standard};

use crate::{
    module::GpuHandle,
    runtime::{GpuBuf, GpuMut, GpuRef, GpuRuntime},
};

/// NTT4x30 on the CubeCL runtime `RT`, over the ring `R`.
///
/// Coefficient-domain limbs are `i64`, big values `i128`, and a DFT-domain
/// word is the four 30-bit residues of `Primes30`, 16 bytes per coefficient.
pub struct NTT4x30Gpu<RT: GpuRuntime, R: Ring = Standard>(PhantomData<fn() -> (RT, R)>);

impl<RT: GpuRuntime, R: Ring> PartialEq for NTT4x30Gpu<RT, R> {
    fn eq(&self, _: &Self) -> bool {
        true
    }
}

impl<RT: GpuRuntime, R: Ring> Eq for NTT4x30Gpu<RT, R> {}

impl<RT: GpuRuntime, R: Ring> Backend for NTT4x30Gpu<RT, R> {
    const DFT_LIMBS_CONTIGUOUS: bool = true;

    type TaskExecutor = poulpy_hal::execution::SerialTaskExecutor;
    type Ring = R;
    type ZnxWord = i64;
    type BigWord = i128;
    type DftWord = CrtWord<Primes30, u32>;
    type OwnedBuf = GpuBuf<RT>;
    type BufRef<'a> = GpuRef<'a, RT>;
    type BufMut<'a> = GpuMut<'a, RT>;
    type Handle = GpuHandle<RT>;
    type Location = Device;

    fn alloc_bytes(len: usize) -> Self::OwnedBuf {
        GpuBuf::zeroed(len)
    }

    fn alloc_zeroed_bytes(len: usize) -> Self::OwnedBuf {
        GpuBuf::zeroed(len)
    }

    fn from_host_bytes(bytes: &[u8]) -> Self::OwnedBuf {
        GpuBuf::from_host(bytes)
    }

    fn to_host_bytes(buf: &Self::OwnedBuf) -> Vec<u8> {
        buf.to_host()
    }

    fn copy_to_host(buf: &Self::OwnedBuf, dst: &mut [u8]) {
        buf.view().read_prefix(dst);
    }

    fn copy_from_host(buf: &mut Self::OwnedBuf, src: &[u8]) {
        buf.write_prefix(src);
    }

    fn copy_view_to_host(buf: &Self::BufRef<'_>, dst: &mut [u8]) {
        buf.read_prefix(dst);
    }

    fn copy_host_to_view(buf: &mut Self::BufMut<'_>, src: &[u8]) {
        buf.write_prefix(src);
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
        buf.view()
    }

    fn view_ref<'a, 'b>(buf: &'a Self::BufRef<'b>) -> Self::BufRef<'a>
    where
        Self: 'b,
    {
        buf.region(0, buf.len())
    }

    fn view_ref_mut<'a, 'b>(buf: &'a Self::BufMut<'b>) -> Self::BufRef<'a>
    where
        Self: 'b,
    {
        buf.region(0, buf.len())
    }

    fn view_mut_ref<'a, 'b>(buf: &'a mut Self::BufMut<'b>) -> Self::BufMut<'a>
    where
        Self: 'b,
    {
        let len = buf.len();
        buf.region_mut(0, len)
    }

    fn view_mut(buf: &mut Self::OwnedBuf) -> Self::BufMut<'_> {
        buf.view_mut()
    }

    fn region(buf: &Self::OwnedBuf, offset: usize, len: usize) -> Self::BufRef<'_> {
        buf.region(offset, len)
    }

    fn region_mut(buf: &mut Self::OwnedBuf, offset: usize, len: usize) -> Self::BufMut<'_> {
        buf.region_mut(offset, len)
    }

    fn region_ref<'a, 'b>(buf: &'a Self::BufRef<'b>, offset: usize, len: usize) -> Self::BufRef<'a>
    where
        Self: 'b,
    {
        buf.region(offset, len)
    }

    fn region_ref_mut<'a, 'b>(buf: &'a Self::BufMut<'b>, offset: usize, len: usize) -> Self::BufRef<'a>
    where
        Self: 'b,
    {
        buf.region(offset, len)
    }

    fn region_mut_ref<'a, 'b>(buf: &'a mut Self::BufMut<'b>, offset: usize, len: usize) -> Self::BufMut<'a>
    where
        Self: 'b,
    {
        buf.region_mut(offset, len)
    }

    unsafe fn destroy(handle: NonNull<Self::Handle>) {
        // SAFETY: the caller passes the pointer `HalModuleImpl::new` made from a `Box`, once.
        unsafe {
            drop(Box::from_raw(handle.as_ptr()));
        }
    }
}
