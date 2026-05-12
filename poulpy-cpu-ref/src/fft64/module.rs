//! Backend handle and module initialization for [`FFT64Ref`](crate::FFT64Ref).
//!
//! This module defines:
//!
//! - [`FFT64RefHandle`]: the opaque handle stored inside a `Module<FFT64Ref>`,
//!   holding precomputed FFT and IFFT twiddle-factor tables.
//! - The [`Backend`] trait implementation, which defines scalar types and the
//!   handle destruction path.
//! - The [`FFT64HandleFactory`] implementation, which builds the handle stored
//!   inside the `Module`.
//! - The shared [`FFT64ModuleHandle`](crate::reference::fft64::module::FFT64ModuleHandle)
//!   trait from `poulpy-hal`, which provides typed access to the FFT tables from
//!   a `Module<FFT64Ref>` and other FFT64-family backends.

use std::ptr::NonNull;

use poulpy_hal::{
    alloc_aligned, assert_alignment,
    layouts::{Backend, Host},
};

use crate::reference::fft64::{
    module::{FFT64HandleFactory, FFTHandleProvider},
    reim::{ReimFFTTable, ReimIFFTTable},
};

use super::FFT64Ref;

/// Opaque handle for the [`FFT64Ref`](crate::FFT64Ref) backend.
///
/// Holds precomputed twiddle-factor tables for the forward FFT and inverse FFT
/// of size `m = n / 2`, where `n` is the ring dimension passed to
/// [`Module::new`](poulpy_hal::api::ModuleNew::new).
///
/// This struct is heap-allocated during module creation and freed when the
/// `Module<FFT64Ref>` is dropped (via [`Backend::destroy`]).
#[repr(C)]
pub struct FFT64RefHandle {
    table_fft: ReimFFTTable<f64>,
    table_ifft: ReimIFFTTable<f64>,
}

impl Backend for FFT64Ref {
    type ScalarPrep = f64;
    type ScalarBig = i64;
    type OwnedBuf = Vec<u8>;
    type BufRef<'a> = &'a [u8];
    type BufMut<'a> = &'a mut [u8];
    type Handle = FFT64RefHandle;
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
    unsafe fn destroy(handle: NonNull<Self::Handle>) {
        unsafe {
            drop(Box::from_raw(handle.as_ptr()));
        }
    }
}

/// # Safety
///
/// The returned handle must be fully initialized for `n`.
unsafe impl FFT64HandleFactory for FFT64RefHandle {
    fn create_fft64_handle(n: usize) -> Self {
        FFT64RefHandle {
            table_fft: ReimFFTTable::new(n >> 1),
            table_ifft: ReimIFFTTable::new(n >> 1),
        }
    }
}

unsafe impl FFTHandleProvider<f64> for FFT64RefHandle {
    fn get_fft_table(&self) -> &ReimFFTTable<f64> {
        &self.table_fft
    }

    fn get_ifft_table(&self) -> &ReimIFFTTable<f64> {
        &self.table_ifft
    }
}
