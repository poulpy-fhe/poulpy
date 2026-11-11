//! Backend handle and module initialisation for [`NTT4x30Portable`](super::NTT4x30Portable).
//!
//! This module defines:
//!
//! - [`NTT4x30PortableHandle`]: the opaque handle stored inside a `Module<NTT4x30Portable>`
//!   holding the tables of the packed NTT of both rings, the conjugate-invariant ones built on first use.
//! - The [`Backend`] trait implementation, which defines scalar types and the
//!   handle destruction path.
//! - The [`NttHandleFactory`] implementation, which builds the handle stored
//!   inside the `Module`.

use std::ptr::NonNull;

use poulpy_hal::{
    AlignedBuf, alloc_aligned,
    layouts::{Backend, Host},
};

use crate::kernels::ntt4x30::{primes::Primes30, vec_znx_dft::NttHandleFactory};
use crate::kernels::{PerRing, RingFamily};
use poulpy_hal::layouts::{ConjugateInvariant, CrtWord, Ring, Standard};

use super::NTT4x30Portable;
use super::ntt32::{MIN_N, Ntt32Table};

/// Opaque handle of [`NTT4x30Portable`](super::NTT4x30Portable), holding the tables of both rings.
///
/// Holds the tables of the packed NTT of every degree the module serves.
///
/// This struct is heap-allocated during module creation and freed when the
/// `Module` is dropped (via [`Backend::destroy`]).
#[repr(C)]
pub struct NTT4x30PortableHandle {
    table_cache: crate::table_cache::ModuleTableCache,
    /// Tables of the packed NTT of each ring, indexed by `log2(n)`.
    packed: PerRing<PackedTables>,
}

/// The packed NTT tables of each ring, indexed by `log2(n)`.
struct PackedTables;

impl RingFamily for PackedTables {
    type Of<R: Ring> = Vec<Option<Ntt32Table>>;
}

fn packed_tables<Rg: Ring>(n: usize) -> Vec<Option<Ntt32Table>> {
    (0..=n.ilog2())
        .map(|log_n| {
            let n = 1usize << log_n;
            (n >= MIN_N).then(|| Ntt32Table::new(n, Rg::CYCLOTOMIC_ORDER_FACTOR == 4))
        })
        .collect()
}

impl NTT4x30PortableHandle {
    /// Tables of the packed NTT of degree `n` on ring `Rg`.
    ///
    /// Panics below the smallest degree the kernels accept, which is the smallest degree of the backend.
    #[inline]
    pub(crate) fn packed_table<Rg: Ring>(&self, n: usize) -> &Ntt32Table {
        self.packed
            .on_ring::<Rg>()
            .get(n.trailing_zeros() as usize)
            .and_then(Option::as_ref)
            .expect("no packed NTT table for this degree")
    }
}

impl poulpy_hal::execution::ScratchWorkers for NTT4x30Portable {}

impl poulpy_hal::layouts::MaxBase2k for NTT4x30Portable {
    fn max_base2k<Rg: Ring>(n: usize, products: usize, failure_bits: usize, squaring: bool) -> usize {
        poulpy_hal::layouts::max_base2k_ntt::<Self, Rg>(
            <Primes30 as poulpy_hal::layouts::PrimeSet>::LOG_Q_PRODUCT,
            n,
            products,
            failure_bits,
            squaring,
        )
    }

    fn max_base2k_squares<Rg: Ring>(n: usize, products: usize, squares: usize, failure_bits: usize) -> usize {
        poulpy_hal::layouts::max_base2k_ntt_squares::<Self, Rg>(
            <Primes30 as poulpy_hal::layouts::PrimeSet>::LOG_Q_PRODUCT,
            n,
            products,
            squares,
            failure_bits,
        )
    }
}

impl Backend for NTT4x30Portable {
    const DFT_LIMBS_CONTIGUOUS: bool = true;

    type TaskExecutor = poulpy_hal::execution::SerialTaskExecutor;
    type DftWord = CrtWord<Primes30, u32>;
    type ZnxWord = i64;
    type BigWord = i128;
    type OwnedBuf = AlignedBuf;
    type BufRef<'a> = &'a [u8];
    type BufMut<'a> = &'a mut [u8];
    type Handle = NTT4x30PortableHandle;
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
        buf.region_mut(offset, len)
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
unsafe impl NttHandleFactory for NTT4x30PortableHandle {
    fn create_ntt_handle(n: usize) -> Self {
        NTT4x30PortableHandle {
            table_cache: Default::default(),
            packed: PerRing::new(packed_tables::<Standard>(n), n, packed_tables::<ConjugateInvariant>),
        }
    }
}

unsafe impl crate::table_cache::ModuleTableCacheProvider for NTT4x30PortableHandle {
    fn module_plan_cache(&self) -> &crate::table_cache::ModuleTableCache {
        &self.table_cache
    }
}
