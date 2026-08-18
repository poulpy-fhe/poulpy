use std::ptr::NonNull;

use poulpy_hal::{
    layouts::{Backend, DataViewMut, Host, Module},
    oep::HalModuleImpl,
};

use crate::{
    api::TransferInto,
    layouts::{Base2K, Dnum, Dsize, GGLWE, GLWE, ModuleCoreAlloc, Rank, TorusPrecision},
};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
struct SrcBackend;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
struct DstBackend;

fn host_alloc(len: usize) -> Vec<u8> {
    vec![0; len]
}

impl Backend for SrcBackend {
    type ZnxWord = i64;
    type BigWord = i64;
    type DftWord = f64;
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
    fn copy_view_to_host(buf: &Self::BufRef<'_>, dst: &mut [u8]) {
        dst.copy_from_slice(buf);
    }
    fn copy_host_to_view(buf: &mut Self::BufMut<'_>, src: &[u8]) {
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
    type ZnxWord = i64;
    type BigWord = i64;
    type DftWord = f64;
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
    fn copy_view_to_host(buf: &Self::BufRef<'_>, dst: &mut [u8]) {
        dst.copy_from_slice(buf);
    }
    fn copy_host_to_view(buf: &mut Self::BufMut<'_>, src: &[u8]) {
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

fn fill_bytes(buf: &mut [u8]) {
    for (i, byte) in buf.iter_mut().enumerate() {
        *byte = (i as u8).wrapping_mul(17).wrapping_add(3);
    }
}

#[test]
fn module_transfer_glwe_roundtrip() {
    let src_module: Module<SrcBackend> = Module::new(64);
    let dst_module: Module<DstBackend> = Module::new(64);
    let mut src: GLWE<<SrcBackend as Backend>::OwnedBuf, <SrcBackend as Backend>::ZnxWord> =
        src_module.glwe_alloc(Base2K(12), TorusPrecision(33), Rank(2));
    fill_bytes(&mut src.data.data);

    let mut uploaded = dst_module.glwe_alloc_from_infos(&src);
    src.transfer_into(&mut uploaded);
    let mut downloaded = src_module.glwe_alloc_from_infos(&src);
    uploaded.transfer_into(&mut downloaded);

    assert_eq!(downloaded, src);
}

#[test]
fn module_transfer_gglwe_roundtrip() {
    let src_module: Module<SrcBackend> = Module::new(64);
    let dst_module: Module<DstBackend> = Module::new(64);
    let mut src: GGLWE<<SrcBackend as Backend>::OwnedBuf, <SrcBackend as Backend>::ZnxWord> =
        src_module.gglwe_alloc(Base2K(12), Dnum(3), Dsize(1), TorusPrecision(12 + 6), Rank(1), Rank(2));
    fill_bytes(src.data.data_mut());

    let mut uploaded = dst_module.gglwe_alloc_from_infos(&src);
    src.transfer_into(&mut uploaded);
    let mut downloaded = src_module.gglwe_alloc_from_infos(&src);
    uploaded.transfer_into(&mut downloaded);

    assert_eq!(downloaded, src);
}

/// `transfer_buf_into` writes into a destination the caller already owns, so a
/// loop can hoist the allocation out. One copy whenever either side is
/// host-visible.
#[test]
fn transfer_buf_into_reuses_destination() {
    use poulpy_hal::layouts::transfer_buf_into;

    let len = 256usize;
    let mut a: Vec<u8> = <SrcBackend as Backend>::alloc_bytes(len);
    let mut b: Vec<u8> = <SrcBackend as Backend>::alloc_bytes(len);
    fill_bytes(&mut a);
    b.iter_mut().enumerate().for_each(|(i, x)| *x = (i as u8).wrapping_mul(31));

    let mut dst: Vec<u8> = <DstBackend as Backend>::alloc_bytes(len);

    transfer_buf_into(&a, &mut dst);
    assert_eq!(dst, a);

    // Same destination, different source: no bytes of `a` survive.
    transfer_buf_into(&b, &mut dst);
    assert_eq!(dst, b);
    assert_ne!(dst, a);
}

/// The size check lives in the shared move, so no implementor can skip it.
#[test]
#[should_panic(expected = "transfer_buf_into: source is 256 bytes, destination is 128")]
fn transfer_buf_into_rejects_size_mismatch() {
    use poulpy_hal::layouts::transfer_buf_into;

    let src: Vec<u8> = <SrcBackend as Backend>::alloc_bytes(256);
    let mut dst: Vec<u8> = <DstBackend as Backend>::alloc_bytes(128);
    transfer_buf_into(&src, &mut dst);
}

/// A layout move checks the whole shape, not just the byte count.
#[test]
#[should_panic(expected = "transfer_into: GLWE k")]
fn transfer_into_rejects_shape_mismatch() {
    use crate::api::TransferInto;

    let src_module: Module<SrcBackend> = Module::new(64);
    let dst_module: Module<DstBackend> = Module::new(64);
    let src = src_module.glwe_alloc(Base2K(12), TorusPrecision(24), Rank(1));
    let mut dst = dst_module.glwe_alloc(Base2K(12), TorusPrecision(36), Rank(1));
    src.transfer_into(&mut dst);
}
