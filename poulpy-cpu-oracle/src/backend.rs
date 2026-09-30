//! The oracle backend type, its module handle and its host byte storage.

use std::{marker::PhantomData, ptr::NonNull};

use poulpy_hal::{
    AlignedBuf, alloc_aligned,
    layouts::{Backend, Host, Module, Standard},
    oep::HalModuleImpl,
};

use crate::{family::Family, fft::Fft64, ntt::Ntt4x30};

/// Scalar correctness oracle over the transform family `F`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Oracle<F: Family>(PhantomData<F>);

/// Oracle using a scalar `f64` FFT.
pub type FFT64Oracle = Oracle<Fft64>;

/// Oracle using a scalar NTT over four 30-bit primes.
pub type NTT4x30Oracle = Oracle<Ntt4x30>;

/// Transform tables for every degree up to the module degree, by `log2(n)`.
pub struct Handle<F: Family> {
    tables: Vec<F::Table>,
}

/// The transform tables of `module` for degree `n`.
pub(crate) fn table<F: Family>(module: &Module<Oracle<F>>, n: usize) -> &F::Table {
    let handle: &Handle<F> = unsafe { &*module.ptr() };
    assert!(
        n.is_power_of_two() && (n.ilog2() as usize) < handle.tables.len(),
        "degree {n} is not served by the module"
    );
    &handle.tables[n.ilog2() as usize]
}

unsafe impl<F: Family> HalModuleImpl for Oracle<F> {
    fn new(n: u64) -> Module<Self> {
        assert!(n.is_power_of_two(), "module degree must be a power of two, got {n}");
        let tables = (0..=n.ilog2()).map(|log_n| F::table(1 << log_n)).collect();
        let ptr = NonNull::from(Box::leak(Box::new(Handle::<F> { tables })));
        unsafe { Module::from_nonnull(ptr, n) }
    }
}

impl<F: Family> poulpy_hal::execution::ScratchWorkers for Oracle<F> {}

impl<F: Family> Backend for Oracle<F> {
    const DFT_LIMBS_CONTIGUOUS: bool = true;

    type TaskExecutor = poulpy_hal::execution::SerialTaskExecutor;
    type Ring = Standard;
    type DftWord = F::Dft;
    type ZnxWord = i64;
    type BigWord = F::Big;
    type OwnedBuf = AlignedBuf;
    type BufRef<'a> = &'a [u8];
    type BufMut<'a> = &'a mut [u8];
    type Handle = Handle<F>;
    type Location = Host;

    fn alloc_bytes(len: usize) -> AlignedBuf {
        alloc_aligned::<u8>(len)
    }
    fn from_host_bytes(bytes: &[u8]) -> AlignedBuf {
        AlignedBuf::from(bytes)
    }
    fn to_host_bytes(buf: &AlignedBuf) -> Vec<u8> {
        buf.to_vec()
    }
    fn copy_to_host(buf: &AlignedBuf, dst: &mut [u8]) {
        dst.copy_from_slice(&buf[..dst.len()]);
    }
    fn copy_from_host(buf: &mut AlignedBuf, src: &[u8]) {
        buf[..src.len()].copy_from_slice(src);
        buf[src.len()..].fill(0);
    }
    fn copy_view_to_host(buf: &&[u8], dst: &mut [u8]) {
        dst.copy_from_slice(&buf[..dst.len()]);
    }
    fn copy_host_to_view(buf: &mut &mut [u8], src: &[u8]) {
        buf[..src.len()].copy_from_slice(src);
        buf[src.len()..].fill(0);
    }
    fn len_bytes(buf: &AlignedBuf) -> usize {
        buf.len()
    }
    fn len_bytes_ref(buf: &&[u8]) -> usize {
        buf.len()
    }
    fn len_bytes_mut(buf: &&mut [u8]) -> usize {
        buf.len()
    }
    fn view(buf: &AlignedBuf) -> &[u8] {
        buf.as_slice()
    }
    fn view_ref<'a, 'b>(buf: &'a &'b [u8]) -> &'a [u8]
    where
        Self: 'b,
    {
        buf
    }
    fn view_ref_mut<'a, 'b>(buf: &'a &'b mut [u8]) -> &'a [u8]
    where
        Self: 'b,
    {
        buf
    }
    fn view_mut_ref<'a, 'b>(buf: &'a mut &'b mut [u8]) -> &'a mut [u8]
    where
        Self: 'b,
    {
        buf
    }
    fn view_mut(buf: &mut AlignedBuf) -> &mut [u8] {
        buf.as_mut_slice()
    }
    fn region(buf: &AlignedBuf, offset: usize, len: usize) -> &[u8] {
        &buf[offset..offset + len]
    }
    fn region_mut(buf: &mut AlignedBuf, offset: usize, len: usize) -> &mut [u8] {
        &mut buf[offset..offset + len]
    }
    fn region_ref<'a, 'b>(buf: &'a &'b [u8], offset: usize, len: usize) -> &'a [u8]
    where
        Self: 'b,
    {
        &buf[offset..offset + len]
    }
    fn region_ref_mut<'a, 'b>(buf: &'a &'b mut [u8], offset: usize, len: usize) -> &'a [u8]
    where
        Self: 'b,
    {
        &buf[offset..offset + len]
    }
    fn region_mut_ref<'a, 'b>(buf: &'a mut &'b mut [u8], offset: usize, len: usize) -> &'a mut [u8]
    where
        Self: 'b,
    {
        &mut buf[offset..offset + len]
    }
    unsafe fn destroy(handle: NonNull<Handle<F>>) {
        drop(unsafe { Box::from_raw(handle.as_ptr()) });
    }
}
