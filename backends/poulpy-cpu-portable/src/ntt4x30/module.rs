//! Backend handle and module initialisation for [`NTT4x30Portable`](super::NTT4x30Portable).
//!
//! This module defines:
//!
//! - [`NTT4x30PortableHandle`]: the opaque handle stored inside a `Module<NTT4x30Portable>`
//!   (and, over the conjugate invariant ring, `Module<NTT4x30CIPortable>`),
//!   holding the tables of the packed NTT.
//! - The [`Backend`] trait implementation, which defines scalar types and the
//!   handle destruction path.
//! - The [`NttHandleFactory`] implementation, which builds the handle stored
//!   inside the `Module`.

use std::marker::PhantomData;
use std::ptr::NonNull;

use poulpy_hal::{
    AlignedBuf, alloc_aligned,
    layouts::{Backend, Host},
};

use crate::kernels::ntt4x30::{primes::Primes30, vec_znx_dft::NttHandleFactory};
use poulpy_hal::layouts::{CrtWord, Ring, Standard};

use super::NTT4x30Portable;
use super::ntt32::{MIN_N, Ntt32Table};

/// Opaque handle for the NTT4x30 portable backends over ring `R`
/// ([`NTT4x30Portable`](super::NTT4x30Portable) and [`NTT4x30CIPortable`](crate::NTT4x30CIPortable)).
///
/// Holds the tables of the packed NTT of every degree the module serves.
///
/// This struct is heap-allocated during module creation and freed when the
/// `Module` is dropped (via [`Backend::destroy`]).
#[repr(C)]
pub struct NTT4x30PortableHandle<R: Ring = Standard> {
    _ring: PhantomData<R>,
    table_cache: crate::table_cache::ModuleTableCache,
    /// Tables of the packed NTT, indexed by `log2(n)`.
    packed: Vec<Option<Ntt32Table>>,
}

impl<R: Ring> NTT4x30PortableHandle<R> {
    /// Tables of the packed NTT of degree `n`.
    ///
    /// Panics below the smallest degree the kernels accept, which is the smallest degree of the backend.
    #[inline]
    pub(crate) fn packed_table(&self, n: usize) -> &Ntt32Table {
        self.packed
            .get(n.trailing_zeros() as usize)
            .and_then(Option::as_ref)
            .expect("no packed NTT table for this degree")
    }
}

impl<R: Ring> poulpy_hal::execution::ScratchWorkers for NTT4x30Portable<R> {}

impl<R: Ring> poulpy_hal::layouts::MaxBase2k for NTT4x30Portable<R> {
    fn max_base2k(n: usize, products: usize, failure_bits: usize, squaring: bool) -> usize {
        poulpy_hal::layouts::max_base2k_ntt::<Self>(
            <Primes30 as poulpy_hal::layouts::PrimeSet>::LOG_Q_PRODUCT,
            n,
            products,
            failure_bits,
            squaring,
        )
    }
}

impl<R: Ring> Backend for NTT4x30Portable<R> {
    const DFT_LIMBS_CONTIGUOUS: bool = true;

    type TaskExecutor = poulpy_hal::execution::SerialTaskExecutor;
    type Ring = R;
    type DftWord = CrtWord<Primes30, u32>;
    type ZnxWord = i64;
    type BigWord = i128;
    type OwnedBuf = AlignedBuf;
    type BufRef<'a> = &'a [u8];
    type BufMut<'a> = &'a mut [u8];
    type Handle = NTT4x30PortableHandle<R>;
    type Location = Host;

    fn alloc_bytes(len: usize) -> Self::OwnedBuf {
        alloc_aligned::<u8>(len)
    }
    fn alloc_zeroed_bytes(len: usize) -> Self::OwnedBuf {
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
///
/// The returned handle must be fully initialized for `n`.
unsafe impl<R: Ring> NttHandleFactory for NTT4x30PortableHandle<R> {
    fn create_ntt_handle(n: usize) -> Self {
        NTT4x30PortableHandle {
            _ring: PhantomData,
            table_cache: Default::default(),
            packed: (0..=n.ilog2())
                .map(|log_n| {
                    let n = 1usize << log_n;
                    (n >= MIN_N).then(|| Ntt32Table::new(n, R::CYCLOTOMIC_ORDER_FACTOR == 4))
                })
                .collect(),
        }
    }
}

unsafe impl<R: Ring> crate::table_cache::ModuleTableCacheProvider for NTT4x30PortableHandle<R> {
    fn module_plan_cache(&self) -> &crate::table_cache::ModuleTableCache {
        &self.table_cache
    }
}
