use std::ptr::NonNull;

use poulpy_hal::{
    layouts::{Backend, DataViewMut, Host, Module},
    oep::HalModuleImpl,
};

use crate::{
    api::ModuleTransfer,
    layouts::{Base2K, Dnum, Dsize, GGLWE, GLWE, ModuleCoreAlloc, Rank, TorusPrecision},
};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct SrcBackend;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct DstBackend;

fn host_alloc(len: usize) -> Vec<u8> {
    vec![0; len]
}

impl Backend for SrcBackend {
    type ScalarBig = i64;
    type ScalarPrep = f64;
    type OwnedBuf = Vec<u8>;
    type BufRef<'a> = &'a [u8];
    type BufMut<'a> = &'a mut [u8];
    type Handle = ();
    type Location = Host;

    fn alloc_bytes(len: usize) -> Self::OwnedBuf {
        host_alloc(len)
    }

    fn from_host_bytes(bytes: &[u8]) -> Self::OwnedBuf {
        bytes.to_vec()
    }

    fn from_bytes(bytes: Vec<u8>) -> Self::OwnedBuf {
        bytes
    }

    fn to_host_bytes(buf: &Self::OwnedBuf) -> Vec<u8> {
        buf.clone()
    }

    fn copy_to_host(buf: &Self::OwnedBuf, dst: &mut [u8]) {
        dst.copy_from_slice(buf);
    }

    fn copy_from_host(buf: &mut Self::OwnedBuf, src: &[u8]) {
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

unsafe impl HalModuleImpl<SrcBackend> for SrcBackend {
    fn new(n: u64) -> Module<SrcBackend> {
        assert!(n.is_power_of_two(), "n must be a power of two, got {n}");
        unsafe { Module::from_nonnull(NonNull::dangling(), n) }
    }
}

impl Backend for DstBackend {
    type ScalarBig = i64;
    type ScalarPrep = f64;
    type OwnedBuf = Vec<u8>;
    type BufRef<'a> = &'a [u8];
    type BufMut<'a> = &'a mut [u8];
    type Handle = ();
    type Location = Host;

    fn alloc_bytes(len: usize) -> Self::OwnedBuf {
        host_alloc(len)
    }

    fn from_host_bytes(bytes: &[u8]) -> Self::OwnedBuf {
        bytes.to_vec()
    }

    fn from_bytes(bytes: Vec<u8>) -> Self::OwnedBuf {
        bytes
    }

    fn to_host_bytes(buf: &Self::OwnedBuf) -> Vec<u8> {
        buf.clone()
    }

    fn copy_to_host(buf: &Self::OwnedBuf, dst: &mut [u8]) {
        dst.copy_from_slice(buf);
    }

    fn copy_from_host(buf: &mut Self::OwnedBuf, src: &[u8]) {
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

unsafe impl HalModuleImpl<DstBackend> for DstBackend {
    fn new(n: u64) -> Module<DstBackend> {
        assert!(n.is_power_of_two(), "n must be a power of two, got {n}");
        unsafe { Module::from_nonnull(NonNull::dangling(), n) }
    }
}

impl poulpy_hal::layouts::TransferFrom<SrcBackend> for SrcBackend {
    fn transfer_buf(src: &Vec<u8>) -> Vec<u8> {
        src.clone()
    }
}
impl poulpy_hal::layouts::TransferFrom<DstBackend> for DstBackend {
    fn transfer_buf(src: &Vec<u8>) -> Vec<u8> {
        src.clone()
    }
}
impl poulpy_hal::layouts::TransferFrom<SrcBackend> for DstBackend {
    fn transfer_buf(src: &Vec<u8>) -> Vec<u8> {
        src.clone()
    }
}
impl poulpy_hal::layouts::TransferFrom<DstBackend> for SrcBackend {
    fn transfer_buf(src: &Vec<u8>) -> Vec<u8> {
        src.clone()
    }
}

fn fill_bytes(buf: &mut [u8]) {
    for (i, byte) in buf.iter_mut().enumerate() {
        *byte = (i as u8).wrapping_mul(17).wrapping_add(3);
    }
}

#[test]
fn module_transfer_glwe_roundtrip() {
    let src_module: Module<SrcBackend> = Module::new(64);
    let dst_module: Module<DstBackend> = Module::new(64);
    let mut src: GLWE<Vec<u8>> = src_module.glwe_alloc(Base2K(12), TorusPrecision(33), Rank(2));
    fill_bytes(&mut src.data.data);

    let uploaded = dst_module.upload_glwe::<SrcBackend>(&src);
    let downloaded = src_module.download_glwe::<DstBackend>(&uploaded);
    let via_wrapper = src.to_backend::<SrcBackend, DstBackend>(&dst_module);

    assert_eq!(uploaded, via_wrapper);
    assert_eq!(downloaded, src);
}

#[test]
fn module_transfer_gglwe_roundtrip() {
    let src_module: Module<SrcBackend> = Module::new(64);
    let dst_module: Module<DstBackend> = Module::new(64);
    let mut src: GGLWE<Vec<u8>> = src_module.gglwe_alloc(Base2K(12), TorusPrecision(33), Rank(1), Rank(2), Dnum(3), Dsize(1));
    fill_bytes(src.data.data_mut());

    let uploaded = dst_module.upload_gglwe::<SrcBackend>(&src);
    let downloaded = src_module.download_gglwe::<DstBackend>(&uploaded);
    let via_wrapper = src.to_backend::<SrcBackend, DstBackend>(&dst_module);

    assert_eq!(uploaded, via_wrapper);
    assert_eq!(downloaded, src);
}
