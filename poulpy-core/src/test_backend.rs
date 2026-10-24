//! Host-resident backends over plain `Vec<u8>`, distinct types per `ID`, for tests that need two backends.

use std::ptr::NonNull;

use poulpy_hal::{
    layouts::{Backend, Host, Module},
    oep::HalModuleImpl,
};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub(crate) struct HostVecBackend<const ID: u8>;

pub(crate) type SrcBackend = HostVecBackend<0>;
pub(crate) type DstBackend = HostVecBackend<1>;

impl<const ID: u8> Backend for HostVecBackend<ID> {
    type TaskExecutor = poulpy_hal::execution::SerialTaskExecutor;
    type Ring = poulpy_hal::layouts::Standard;
    type ZnxWord = i64;
    type BigWord = i64;
    type DftWord = f64;
    type OwnedBuf = Vec<u8>;
    type BufRef<'a> = &'a [u8];
    type BufMut<'a> = &'a mut [u8];
    type Handle = ();
    type Location = Host;

    fn alloc_bytes(len: usize) -> Self::OwnedBuf {
        vec![0; len]
    }

    fn from_host_bytes(bytes: &[u8]) -> Self::OwnedBuf {
        bytes.to_vec()
    }

    fn to_host_bytes(buf: &Self::OwnedBuf) -> Vec<u8> {
        buf.clone()
    }

    fn copy_to_host(buf: &Self::OwnedBuf, dst: &mut [u8]) {
        assert!(buf.len() >= dst.len(), "test backend buffer shorter than destination");
        dst.copy_from_slice(&buf[..dst.len()]);
    }

    fn copy_from_host(buf: &mut Self::OwnedBuf, src: &[u8]) {
        assert!(buf.len() >= src.len(), "test backend buffer shorter than source");
        let src_len = src.len();
        buf[..src_len].copy_from_slice(src);
        buf[src_len..].fill(0);
    }
    fn copy_view_to_host(buf: &Self::BufRef<'_>, dst: &mut [u8]) {
        assert!(buf.len() >= dst.len(), "test backend view shorter than destination");
        dst.copy_from_slice(&buf[..dst.len()]);
    }
    fn copy_host_to_view(buf: &mut Self::BufMut<'_>, src: &[u8]) {
        assert!(buf.len() >= src.len(), "test backend view shorter than source");
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

    unsafe fn destroy(_: NonNull<Self::Handle>) {}
}

unsafe impl<const ID: u8> HalModuleImpl for HostVecBackend<ID> {
    fn new(n: u64) -> Module<HostVecBackend<ID>> {
        assert!(n.is_power_of_two(), "n must be a power of two, got {n}");
        unsafe { Module::from_nonnull(NonNull::dangling(), n) }
    }
}
