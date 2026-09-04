use std::{marker::PhantomData, ptr::NonNull};

use crate::layouts::{Data, DataOwned, Location, MatZnx, ScalarZnx, VecZnx, checked_product};
use crate::{
    GALOISGENERATOR,
    api::{ModuleLogN, ModuleN},
};

/// Core trait that every backend (CPU, GPU, FPGA, ...) must implement.
///
/// Defines the word types used for the coefficient domain (`ZnxWord`),
/// DFT-domain (`DftWord`) and extended-precision (`BigWord`)
/// representations, as well as the opaque `Handle` type that holds
/// backend-specific precomputed state (e.g. FFT twiddle factors).
///
/// # Safety
///
/// [`destroy`](Backend::destroy) is called during [`Module`] drop and must
/// correctly deallocate the handle without double-free.
#[allow(clippy::missing_safety_doc)]
pub trait Backend: Sized + Sync + Send + PartialEq + Eq {
    /// Whether this backend's transform-domain arithmetic is exact within its
    /// documented operand bounds.
    ///
    /// This is an arithmetic property of the backend implementation, not of
    /// the [`DftWord`](crate::layouts::DftWord) byte-layout marker.
    const DFT_IS_EXACT: bool = false;

    /// Task executor selected by this backend.
    type TaskExecutor: crate::execution::TaskExecutor;
    /// Word type for coefficient-domain (small) polynomial representations.
    type ZnxWord: crate::layouts::ZnxWord;
    /// Word type for extended-precision (big) polynomial representations.
    type BigWord: crate::layouts::BigWord;
    /// Word type for DFT-domain (prepared) polynomial representations.
    type DftWord: crate::layouts::DftWord;
    /// Owned backend storage for layouts and scratch.
    ///
    /// This buffer may be host-resident or device-resident. It is intentionally
    /// no longer required to expose direct host byte slices.
    type OwnedBuf: Data + DataOwned + Send + Sync;
    /// Shared borrowed view into backend-owned storage.
    type BufRef<'a>: Data + Sync
    where
        Self: 'a;
    /// Mutable borrowed view into backend-owned storage.
    type BufMut<'a>: Data + Send
    where
        Self: 'a;
    /// Opaque backend handle type (e.g. precomputed FFT twiddle factors).
    type Handle: 'static;
    /// Residency of this backend's buffers — [`Host`](crate::layouts::Host)
    /// or [`Device`](crate::layouts::Device).
    type Location: Location;
    /// Allocates a backend-owned byte buffer of `len` bytes.
    fn alloc_bytes(len: usize) -> Self::OwnedBuf;
    /// Allocates a zero-initialized backend-owned byte buffer of `len` bytes.
    ///
    /// Backends may override this with a device-native implementation
    /// (e.g. `cudaMemset`-backed allocation). The default implementation
    /// falls back to allocating first and then zero-filling through the
    /// existing host upload path.
    fn alloc_zeroed_bytes(len: usize) -> Self::OwnedBuf {
        let mut buf = Self::alloc_bytes(len);
        let zeros = vec![0u8; len];
        Self::copy_from_host(&mut buf, &zeros);
        buf
    }
    /// Uploads or copies host bytes into backend-owned storage.
    fn from_host_bytes(bytes: &[u8]) -> Self::OwnedBuf;
    /// Wraps/Uploads a host-owned byte buffer into backend-owned storage.
    ///
    /// Backends may override this for a zero-copy fast path when the input is
    /// already in a compatible host representation.
    fn from_bytes(bytes: Vec<u8>) -> Self::OwnedBuf;
    /// Copies the contents of a backend-owned buffer into a fresh host `Vec<u8>`.
    ///
    /// For host backends this is typically a simple clone of the underlying
    /// storage; for device backends it performs a device-to-host download.
    fn to_host_bytes(buf: &Self::OwnedBuf) -> Vec<u8>;
    /// Copies the contents of a backend-owned buffer into a host byte slice.
    ///
    /// `dst.len()` must equal the byte length of `buf`.
    fn copy_to_host(buf: &Self::OwnedBuf, dst: &mut [u8]);
    /// Copies a host byte slice into a backend-owned buffer.
    ///
    /// `src.len()` must equal the byte length of `buf`.
    fn copy_from_host(buf: &mut Self::OwnedBuf, src: &[u8]);
    /// Copies a backend-native borrowed view into a host byte slice.
    ///
    /// Unlike [`Self::copy_to_host`], this accepts a view carved from an
    /// arena. Device backends should implement it with a device-to-host copy
    /// from the view's native address.
    fn copy_view_to_host(buf: &Self::BufRef<'_>, dst: &mut [u8]);
    /// Copies a host byte slice into a backend-native mutable borrowed view.
    ///
    /// Unlike [`Self::copy_from_host`], this accepts a view carved from an
    /// arena. Device backends should implement it with a host-to-device copy
    /// to the view's native address.
    fn copy_host_to_view(buf: &mut Self::BufMut<'_>, src: &[u8]);
    /// Returns the number of bytes stored in a backend-owned buffer.
    fn len_bytes(buf: &Self::OwnedBuf) -> usize;
    /// Returns the number of bytes spanned by a shared borrowed view.
    ///
    /// Views are the unit a transfer addresses, so their extent has to be
    /// legible without an owned buffer in hand.
    fn len_bytes_ref(buf: &Self::BufRef<'_>) -> usize;
    /// Returns the number of bytes spanned by a mutable borrowed view.
    fn len_bytes_mut(buf: &Self::BufMut<'_>) -> usize;
    /// Borrows a shared backend-native view over an owned buffer.
    fn view(buf: &Self::OwnedBuf) -> Self::BufRef<'_>;
    /// Reborrows an existing shared backend-native view.
    fn view_ref<'a, 'b>(buf: &'a Self::BufRef<'b>) -> Self::BufRef<'a>
    where
        Self: 'b;
    /// Reborrows a mutable backend-native view as a shared backend-native view.
    fn view_ref_mut<'a, 'b>(buf: &'a Self::BufMut<'b>) -> Self::BufRef<'a>
    where
        Self: 'b;
    /// Reborrows an existing mutable backend-native view.
    fn view_mut_ref<'a, 'b>(buf: &'a mut Self::BufMut<'b>) -> Self::BufMut<'a>
    where
        Self: 'b;
    /// Borrows a mutable backend-native view over an owned buffer.
    fn view_mut(buf: &mut Self::OwnedBuf) -> Self::BufMut<'_>;
    /// Borrows a shared sub-region of an owned buffer.
    fn region(buf: &Self::OwnedBuf, offset: usize, len: usize) -> Self::BufRef<'_>;
    /// Borrows a mutable sub-region of an owned buffer.
    fn region_mut(buf: &mut Self::OwnedBuf, offset: usize, len: usize) -> Self::BufMut<'_>;
    /// Reborrows a shared sub-region of an existing shared backend-native view.
    fn region_ref<'a, 'b>(buf: &'a Self::BufRef<'b>, offset: usize, len: usize) -> Self::BufRef<'a>
    where
        Self: 'b;
    /// Reborrows a shared sub-region of an existing mutable backend-native view.
    fn region_ref_mut<'a, 'b>(buf: &'a Self::BufMut<'b>, offset: usize, len: usize) -> Self::BufRef<'a>
    where
        Self: 'b;
    /// Reborrows a mutable sub-region of an existing mutable backend-native view.
    fn region_mut_ref<'a, 'b>(buf: &'a mut Self::BufMut<'b>, offset: usize, len: usize) -> Self::BufMut<'a>
    where
        Self: 'b;
    /// Bytes size of `ZnxWord`.
    fn size_of_znx_word() -> usize {
        size_of::<Self::ZnxWord>()
    }
    /// Bytes size of `BigWord`.
    fn size_of_big_word() -> usize {
        size_of::<Self::BigWord>()
    }
    /// Bytes size of `DftWord`.
    fn size_of_dft_word() -> usize {
        size_of::<Self::DftWord>()
    }

    /// Required alignment (in bytes) for scratch-arena carved regions.
    ///
    /// Default to 64 (one CPU cache line). Device backends should override this
    /// to match their native memory alignment requirement (e.g. 128 for CUDA,
    /// 256 for ROCm). `ScratchArena::align_up` uses this constant so that
    /// carved regions satisfy both alignment and SIMD requirements.
    const SCRATCH_ALIGN: usize = 64;

    /// Byte size of a [`crate::layouts::VecZnx`] buffer.
    fn bytes_of_vec_znx(n: usize, cols: usize, size: usize) -> usize {
        checked_product(&[n, cols, size, Self::size_of_znx_word()], "VecZnx byte size")
    }
    /// Byte size of a [`crate::layouts::ScalarZnx`] buffer.
    fn bytes_of_scalar_znx(n: usize, cols: usize) -> usize {
        checked_product(&[n, cols, Self::size_of_znx_word()], "ScalarZnx byte size")
    }
    /// Byte size of a [`crate::layouts::MatZnx`] buffer.
    fn bytes_of_mat_znx(n: usize, rows: usize, cols_in: usize, cols_out: usize, size: usize) -> usize {
        checked_product(
            &[rows, cols_in, Self::bytes_of_vec_znx(n, cols_out, size)],
            "MatZnx byte size",
        )
    }
    /// Byte size of a [`crate::layouts::VecZnxDft`] buffer.
    fn bytes_of_vec_znx_dft(n: usize, cols: usize, size: usize) -> usize {
        checked_product(&[n, cols, size, Self::size_of_dft_word()], "VecZnxDft byte size")
    }
    /// Byte size of a [`crate::layouts::VecZnxBig`] buffer.
    fn bytes_of_vec_znx_big(n: usize, cols: usize, size: usize) -> usize {
        checked_product(&[n, cols, size, Self::size_of_big_word()], "VecZnxBig byte size")
    }
    /// Byte size of a [`crate::layouts::SvpPPol`] buffer.
    fn bytes_of_svp_ppol(n: usize, cols: usize) -> usize {
        checked_product(&[n, cols, Self::size_of_dft_word()], "SvpPPol byte size")
    }
    /// Byte size of a [`crate::layouts::VmpPMat`] buffer.
    fn bytes_of_vmp_pmat(n: usize, rows: usize, cols_in: usize, cols_out: usize, size: usize) -> usize {
        checked_product(
            &[n, rows, cols_in, cols_out, size, Self::size_of_dft_word()],
            "VmpPMat byte size",
        )
    }
    /// Byte size of a [`crate::layouts::CnvPVecL`] buffer.
    fn bytes_of_cnv_pvec_left(n: usize, cols: usize, size: usize) -> usize {
        checked_product(&[n, cols, size, Self::size_of_dft_word()], "CnvPVecL byte size")
    }
    /// Byte size of a [`crate::layouts::CnvPVecR`] buffer.
    fn bytes_of_cnv_pvec_right(n: usize, cols: usize, size: usize) -> usize {
        checked_product(&[n, cols, size, Self::size_of_dft_word()], "CnvPVecR byte size")
    }
    /// Deallocates a backend handle.
    ///
    /// # Safety
    ///
    /// `handle` must be a valid, non-dangling pointer that was previously
    /// returned by the backend's allocation routine. Must not be called
    /// more than once on the same handle.
    unsafe fn destroy(handle: NonNull<Self::Handle>);
}

/// Primary entry point for all polynomial operations over `Z[X]/(X^N + 1)`.
///
/// A `Module` pairs a maximum ring degree `N` (always a power of two) with a
/// backend-specific handle that holds any required precomputed state. All
/// [`api`](crate::api) trait methods are dispatched through this type.
/// Existing fixed-ring operations use the maximum degree; dimension-aware
/// operations may select any supported power-of-two degree from the handle.
///
/// The module **owns** its handle; dropping the `Module` calls
/// [`Backend::destroy`].
#[repr(C)]
pub struct Module<B: Backend> {
    ptr: NonNull<B::Handle>,
    n: u64,
    _marker: PhantomData<B>,
}

unsafe impl<B: Backend> Sync for Module<B> {}
unsafe impl<B: Backend> Send for Module<B> {}

impl<B: Backend> Module<B> {
    /// Creates a backend module for ring degree `N`.
    #[inline]
    pub fn new(n: u64) -> Self
    where
        Self: crate::api::ModuleNew<B>,
    {
        crate::api::ModuleNew::new(n)
    }

    /// Creates a module from a [`NonNull`] backend handle.
    ///
    /// # Safety
    ///
    /// `ptr` must point to a valid, fully initialized backend handle whose
    /// lifetime is transferred to this `Module` (it will be destroyed on drop).
    #[allow(clippy::missing_safety_doc)]
    #[inline]
    pub unsafe fn from_nonnull(ptr: NonNull<B::Handle>, n: u64) -> Self {
        assert!(n.is_power_of_two(), "n must be a power of two, got {n}");
        Self {
            ptr,
            n,
            _marker: PhantomData,
        }
    }

    /// Construct from a raw pointer managed elsewhere.
    /// SAFETY: `ptr` must be non-null and remain valid for the lifetime of this Module.
    #[inline]
    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn from_raw_parts(ptr: *mut B::Handle, n: u64) -> Self {
        assert!(n.is_power_of_two(), "n must be a power of two, got {n}");
        Self {
            ptr: NonNull::new(ptr).expect("null module ptr"),
            n,
            _marker: PhantomData,
        }
    }

    /// Returns the raw pointer to the backend handle.
    #[allow(clippy::missing_safety_doc)]
    #[inline]
    pub unsafe fn ptr(&self) -> *mut <B as Backend>::Handle {
        self.ptr.as_ptr()
    }

    /// Returns the maximum supported ring degree `N`.
    #[inline]
    pub fn n(&self) -> usize {
        self.n as usize
    }

    /// Explicit alias for [`Self::n`] when treating the module as a
    /// multi-ring execution context.
    #[inline]
    pub fn max_n(&self) -> usize {
        self.n()
    }

    /// Allocates a zero-initialized backend-owned [`ScalarZnx`].
    #[inline]
    pub fn scalar_znx_alloc(&self, cols: usize) -> ScalarZnx<B::OwnedBuf, B::ZnxWord> {
        let n = self.n();
        let len = B::bytes_of_scalar_znx(n, cols);
        let bytes = B::alloc_zeroed_bytes(len);
        ScalarZnx::from_data(bytes, n, cols)
    }

    /// Allocates a zero-initialized backend-owned [`VecZnx`].
    #[inline]
    pub fn vec_znx_alloc(&self, cols: usize, size: usize) -> VecZnx<B::OwnedBuf, B::ZnxWord> {
        let n = self.n();
        let len = self.bytes_of_vec_znx_n(n, cols, size);
        let bytes = B::alloc_zeroed_bytes(len);
        VecZnx::from_data(bytes, n, cols, size)
    }

    /// Returns the byte size of a [`VecZnx`] with this module's ring degree.
    #[inline]
    pub fn bytes_of_vec_znx(&self, cols: usize, size: usize) -> usize {
        self.bytes_of_vec_znx_n(self.n(), cols, size)
    }

    /// Returns the byte size of a [`VecZnx`] with an explicit coefficient degree.
    #[inline]
    pub fn bytes_of_vec_znx_n(&self, n: usize, cols: usize, size: usize) -> usize {
        B::bytes_of_vec_znx(n, cols, size)
    }

    /// Allocates a zero-initialized backend-owned [`MatZnx`].
    #[inline]
    pub fn mat_znx_alloc(&self, rows: usize, cols_in: usize, cols_out: usize, size: usize) -> MatZnx<B::OwnedBuf, B::ZnxWord> {
        let n = self.n();
        let len = B::bytes_of_mat_znx(n, rows, cols_in, cols_out, size);
        let bytes = B::alloc_zeroed_bytes(len);
        MatZnx::from_data(bytes, n, rows, cols_in, cols_out, size)
    }

    /// Returns the raw pointer to the backend handle.
    #[inline]
    pub fn as_mut_ptr(&self) -> *mut B::Handle {
        self.ptr.as_ptr()
    }

    /// Returns `log2(N)`.
    #[inline]
    pub fn log_n(&self) -> usize {
        (usize::BITS - (self.n() - 1).leading_zeros()) as _
    }

    /// Reinterprets this `Module<B>` as a `Module<Other>` sharing the same
    /// backend `Handle` type.
    ///
    /// This is a zero-cost view used to forward API calls to a compatible
    /// source backend without rebuilding the handle.
    #[inline]
    pub fn reinterpret<Other>(&self) -> &Module<Other>
    where
        Other: Backend<Handle = B::Handle>,
    {
        // Safety: Module is #[repr(C)] and only contains an optional NonNull<Handle>,
        // a u64, and a ZST PhantomData. When `Handle` matches, the layout is identical.
        unsafe { &*(self as *const Self as *const Module<Other>) }
    }

    /// Mutable version of [`Module::reinterpret`].
    #[inline]
    pub fn reinterpret_mut<Other>(&mut self) -> &mut Module<Other>
    where
        Other: Backend<Handle = B::Handle>,
    {
        // Safety: see Module::reinterpret.
        unsafe { &mut *(self as *mut Self as *mut Module<Other>) }
    }
}

/// Returns the cyclotomic order `2N` for the ring `Z[X]/(X^N + 1)`.
pub trait CyclotomicOrder
where
    Self: ModuleN,
{
    /// Returns `2N`, the order of the cyclotomic polynomial `X^N + 1`.
    fn cyclotomic_order(&self) -> i64 {
        (self.n() << 1) as _
    }
}

impl<BE: Backend> ModuleLogN for Module<BE> where Self: ModuleN {}

impl<BE: Backend> CyclotomicOrder for Module<BE> where Self: ModuleN {}

/// Computes [`GALOISGENERATOR`]`^|generator| * sign(generator) mod cyclotomic_order`.
///
/// Returns `1` when `generator == 0`.
///
/// # Panics (debug)
///
/// Debug-asserts that `cyclotomic_order` is a positive power of two.
#[inline(always)]
pub fn galois_element(generator: i64, cyclotomic_order: i64) -> i64 {
    debug_assert!(
        cyclotomic_order > 0 && (cyclotomic_order as u64).is_power_of_two(),
        "cyclotomic_order must be a power of two, got {cyclotomic_order}"
    );

    if generator == 0 {
        return 1;
    }

    let g_exp: u64 = mod_exp_u64(GALOISGENERATOR, generator.unsigned_abs() as usize) & (cyclotomic_order - 1) as u64;
    g_exp as i64 * generator.signum()
}

/// Maps a set of slot rotations to the distinct Galois elements whose
/// automorphism keys realize them: drops the identity (`0`) rotation, applies
/// [`galois_element`], and returns the result sorted and de-duplicated.
///
/// Shared by the linear-transformation / DFT layers, which all need "the Galois
/// keys required by these rotations" and would otherwise each re-spell the
/// filter/map/sort/dedup.
pub fn galois_elements_from_rotations(rotations: impl IntoIterator<Item = i64>, cyclotomic_order: i64) -> Vec<i64> {
    let mut gal_els: Vec<i64> = rotations
        .into_iter()
        .filter(|&rot| rot != 0)
        .map(|rot| galois_element(rot, cyclotomic_order))
        .collect();
    gal_els.sort_unstable();
    gal_els.dedup();
    gal_els
}

/// Galois group operations on the cyclotomic ring `Z[X]/(X^N + 1)`.
///
/// The Galois group `(Z/2NZ)*` acts on polynomials via the automorphisms
/// `X -> X^k` for odd `k`. This trait provides methods to compute
/// Galois elements and their inverses from a signed generator exponent.
pub trait GaloisElement
where
    Self: CyclotomicOrder,
{
    /// Returns [`GALOISGENERATOR`]`^|generator| * sign(generator) mod 2N`.
    fn galois_element(&self, generator: i64) -> i64 {
        galois_element(generator, self.cyclotomic_order())
    }

    /// Returns the inverse of `gal_el` in the Galois group `(Z/2NZ)*`.
    ///
    /// # Panics
    ///
    /// Panics if `gal_el == 0`.
    fn galois_element_inv(&self, gal_el: i64) -> i64 {
        if gal_el == 0 {
            panic!("cannot invert 0")
        }

        let g_exp: u64 =
            mod_exp_u64(gal_el.unsigned_abs(), (self.cyclotomic_order() - 1) as usize) & (self.cyclotomic_order() - 1) as u64;
        g_exp as i64 * gal_el.signum()
    }
}

impl<BE: Backend> GaloisElement for Module<BE> where Self: CyclotomicOrder {}

impl<B: Backend> Drop for Module<B> {
    fn drop(&mut self) {
        unsafe { B::destroy(self.ptr) }
    }
}

/// Computes `x^e mod 2^64` using square-and-multiply with wrapping arithmetic.
pub fn mod_exp_u64(x: u64, e: usize) -> u64 {
    let mut y: u64 = 1;
    let mut x_pow: u64 = x;
    let mut exp = e;
    while exp > 0 {
        if exp & 1 == 1 {
            y = y.wrapping_mul(x_pow);
        }
        x_pow = x_pow.wrapping_mul(x_pow);
        exp >>= 1;
    }
    y
}
