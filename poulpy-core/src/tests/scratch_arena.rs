use std::{marker::PhantomData, ptr::NonNull};

use poulpy_hal::{
    api::ScratchArenaTakeBasic,
    layouts::{Backend, DataViewMut, Host, Module, ScratchOwned, VecZnx},
    oep::HalModuleImpl,
};

use crate::{
    layouts::{Base2K, Degree, Dnum, Dsize, GGSWLayout, GLWELayout, Rank, TorusPrecision},
    scratch::ScratchArenaTakeCore,
};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct TestBackend;

impl Backend for TestBackend {
    type ScalarBig = i64;
    type ScalarPrep = f64;
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

unsafe impl HalModuleImpl<TestBackend> for TestBackend {
    fn new(n: u64) -> Module<TestBackend> {
        assert!(n.is_power_of_two(), "n must be a power of two, got {n}");
        unsafe { Module::from_nonnull(NonNull::dangling(), n) }
    }
}

#[test]
fn scratch_arena_take_core_returns_disjoint_backend_regions() {
    let module: Module<TestBackend> = Module::new(64);
    let glwe_infos = GLWELayout {
        n: Degree(64),
        base2k: Base2K(8),
        k: TorusPrecision(24),
        rank: Rank(2),
    };
    let ggsw_infos = GGSWLayout {
        n: Degree(64),
        base2k: Base2K(8),
        k: TorusPrecision(24),
        rank: Rank(1),
        dnum: Dnum(2),
        dsize: Dsize(1),
    };

    let mut scratch: ScratchOwned<TestBackend> = ScratchOwned {
        data: TestBackend::alloc_bytes(1 << 15),
        _phantom: PhantomData,
    };
    let arena = scratch.arena();
    let available_before = arena.available();

    let (mut glwe, arena) = arena.take_glwe_scratch(&glwe_infos);
    let glwe_bytes = glwe.data_mut().data_mut();
    glwe_bytes[0] = 0x5a;
    let glwe_ptr = glwe_bytes.as_mut_ptr() as usize;
    let glwe_len = glwe_bytes.len();

    let (mut prepared, arena) = arena.take_ggsw_prepared_scratch(&module, &ggsw_infos);
    let prepared_bytes = prepared.data.data_mut();
    prepared_bytes[0] = 0xa5;
    let prepared_ptr = prepared_bytes.as_mut_ptr() as usize;
    let prepared_len = prepared_bytes.len();

    assert!(arena.available() < available_before);
    assert!(glwe_ptr + glwe_len <= prepared_ptr || prepared_ptr + prepared_len <= glwe_ptr);
}

#[test]
fn scratch_arena_split_yields_independent_chunks() {
    let mut scratch: ScratchOwned<TestBackend> = ScratchOwned {
        data: TestBackend::alloc_bytes(1 << 12),
        _phantom: PhantomData,
    };
    let arena = scratch.arena();
    let chunk_len = VecZnx::bytes_of(64, 2, 2);
    let (chunks, rem) = arena.split(2, chunk_len);

    assert!(rem.available() < (1 << 12));
    assert_eq!(chunks.len(), 2);

    let mut it = chunks.into_iter();
    let (mut lhs, _) = it.next().expect("missing first scratch chunk").take_vec_znx_scratch(64, 2, 2);
    let (mut rhs, _) = it
        .next()
        .expect("missing second scratch chunk")
        .take_vec_znx_scratch(64, 2, 2);

    let lhs_bytes = lhs.data_mut();
    let rhs_bytes = rhs.data_mut();
    lhs_bytes[0] = 1;
    rhs_bytes[0] = 2;

    let lhs_ptr = lhs_bytes.as_mut_ptr() as usize;
    let rhs_ptr = rhs_bytes.as_mut_ptr() as usize;
    assert!(lhs_ptr + lhs_bytes.len() <= rhs_ptr || rhs_ptr + rhs_bytes.len() <= lhs_ptr);
}
