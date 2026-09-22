use std::{marker::PhantomData, ptr::NonNull};

use crate::layouts::{Data, Location, MatZnx, PrepareHint, ScalarZnx, VecZnx, checked_product, vec_znx_alloc_zeroed};
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
    /// Smallest ring degree every kernel of this backend accepts.
    ///
    /// A module built at degree `N` serves every power-of-two degree `n` with
    /// `MIN_DEGREE <= n <= N`. A backend whose kernels process a fixed number
    /// of coefficients per step raises it to the smallest degree they handle.
    const MIN_DEGREE: usize = 8;

    /// Opts into the split-complex FFT64 stochastic roundoff model used by
    /// [`Module::max_base2k`]. Sharing an `f64` word alone does not opt in.
    ///
    /// Applies to radix-2 FFT stages, ordinary complex products and sequential
    /// accumulation, followed by one inverse FFT and exact power-of-two scaling.
    /// Assumes centered independent arithmetic roundoff with variance `2^-106/3`
    /// and mean squared complex twiddle error at most `2 * 2^-106`, treating
    /// their propagated contributions as uncorrelated. The model covers both
    /// separate multiply/add and two-FMA accumulator updates.
    /// This models the error distribution; it is not a worst-case guarantee.
    const FFT64_ERROR_MODEL: bool = false;

    /// Whether a DFT vector stores each limb as one contiguous block containing
    /// every column, and a range of those blocks is itself a valid DFT vector.
    /// Within a limb, columns are contiguous blocks in column order, each of
    /// size `bytes_of_vec_znx_dft(n, 1, 1)`. This also permits indexed host
    /// zeroing through `VecZnxDft::zero_at`.
    ///
    /// Opting in requires `bytes_of_vec_znx_dft(n, cols, size)` to equal
    /// `size * bytes_of_vec_znx_dft(n, cols, 1)`, including zero at `size == 0`.
    /// Blocks have no size-dependent headers, strides or padding between them.
    /// This permits [`VecZnxDftBackendMut::with_limb_range_mut`](crate::layouts::VecZnxDftBackendMut::with_limb_range_mut)
    /// to reborrow a partial range without copying or changing its representation.
    ///
    /// The default makes no such promise. Other layouts support whole-buffer
    /// views and backend-defined operations, but cannot use partial-range views
    /// or generic indexed zeroing. Reference compositions that require partial
    /// views must opt into this capability explicitly; such backends must supply
    /// their own overrides instead of opting into those reference bodies.
    const DFT_LIMBS_CONTIGUOUS: bool = false;

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
    /// Host-resident backends use [`AlignedBuf`](crate::AlignedBuf).
    type OwnedBuf: Data + Send + Sync;
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
    ///
    /// The buffer may be longer than `bytes`; the extra bytes are zero.
    fn from_host_bytes(bytes: &[u8]) -> Self::OwnedBuf;
    /// Copies a host-owned byte buffer into backend-owned storage.
    ///
    /// The buffer may be longer than `bytes`; the extra bytes are zero. The
    /// default body copies through [`Self::from_host_bytes`].
    fn from_bytes(bytes: Vec<u8>) -> Self::OwnedBuf {
        Self::from_host_bytes(&bytes)
    }
    /// Copies the contents of a backend-owned buffer into a fresh host `Vec<u8>`.
    ///
    /// For host backends this is typically a simple clone of the underlying
    /// storage; for device backends it performs a device-to-host download.
    fn to_host_bytes(buf: &Self::OwnedBuf) -> Vec<u8>;
    /// Copies the contents of a backend-owned buffer into a host byte slice.
    ///
    /// `dst.len()` is at most the byte length of `buf`; the first `dst.len()`
    /// bytes are copied.
    fn copy_to_host(buf: &Self::OwnedBuf, dst: &mut [u8]);
    /// Copies a host byte slice into a backend-owned buffer.
    ///
    /// `src.len()` is at most the byte length of `buf`; the bytes past
    /// `src.len()` are zeroed.
    fn copy_from_host(buf: &mut Self::OwnedBuf, src: &[u8]);
    /// Copies a backend-native borrowed view into a host byte slice.
    ///
    /// Unlike [`Self::copy_to_host`], this accepts a view carved from an
    /// arena. Device backends should implement it with a device-to-host copy
    /// from the view's native address.
    ///
    /// `dst.len()` is at most the byte length of `buf`; the first `dst.len()`
    /// bytes are copied.
    fn copy_view_to_host(buf: &Self::BufRef<'_>, dst: &mut [u8]);
    /// Copies a host byte slice into a backend-native mutable borrowed view.
    ///
    /// Unlike [`Self::copy_from_host`], this accepts a view carved from an
    /// arena. Device backends should implement it with a host-to-device copy
    /// to the view's native address.
    ///
    /// `src.len()` is at most the byte length of `buf`; the bytes past
    /// `src.len()` are zeroed.
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

    /// `len` rounded up to the next multiple of [`Self::SCRATCH_ALIGN`].
    ///
    /// Every region carved from a [`ScratchArena`](crate::layouts::ScratchArena)
    /// starts at an aligned offset, so a `_tmp_bytes` formula that adds the
    /// scratch of a nested call to a temporary of its own has to round the
    /// temporary with this first. Without it the sum is short by up to
    /// `SCRATCH_ALIGN - 1` bytes whenever the temporary is not itself a
    /// multiple of the alignment, and the nested take panics.
    fn scratch_aligned(len: usize) -> usize {
        let rem = len % Self::SCRATCH_ALIGN;
        if rem == 0 {
            len
        } else {
            len.checked_add(Self::SCRATCH_ALIGN - rem)
                .expect("scratch alignment overflows usize")
        }
    }

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
    /// Byte size of a [`crate::layouts::SvpPPol`] buffer in the representation `hint` selects.
    fn bytes_of_svp_ppol(n: usize, cols: usize, _hint: PrepareHint) -> usize {
        checked_product(&[n, cols, Self::size_of_dft_word()], "SvpPPol byte size")
    }
    /// Byte size of a [`crate::layouts::VmpPMat`] buffer in the representation `hint` selects.
    fn bytes_of_vmp_pmat(n: usize, rows: usize, cols_in: usize, cols_out: usize, size: usize, _hint: PrepareHint) -> usize {
        checked_product(
            &[n, rows, cols_in, cols_out, size, Self::size_of_dft_word()],
            "VmpPMat byte size",
        )
    }
    /// Byte size of a [`crate::layouts::CnvPVecL`] buffer in the representation `hint` selects.
    fn bytes_of_cnv_pvec_left(n: usize, cols: usize, size: usize, _hint: PrepareHint) -> usize {
        checked_product(&[n, cols, size, Self::size_of_dft_word()], "CnvPVecL byte size")
    }
    /// Byte size of a [`crate::layouts::CnvPVecR`] buffer in the representation `hint` selects.
    fn bytes_of_cnv_pvec_right(n: usize, cols: usize, size: usize, _hint: PrepareHint) -> usize {
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
/// Every operation serves operands of any power-of-two degree `n` with
/// `B::MIN_DEGREE <= n <= N`, read from the operands; the handle holds the
/// transform tables of every such degree. Allocation takes the degree
/// explicitly.
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
    /// Selects a limb radix for a uniform-input Gaussian failure model.
    ///
    /// `products` is the number of polynomial products accumulated into one
    /// output polynomial of degree `n`. `failure_bits` requests an estimated
    /// probability at most `2^(-failure_bits)` that any coefficient fails CRT
    /// reconstruction or FFT integer rounding. Both arguments must be positive.
    ///
    /// The model assumes independent, centered uniform input coefficients in
    /// `[-2^(b-1), 2^(b-1)]`, neglecting integer endpoint corrections. For NTT,
    /// `sigma = 2^(2*b) * sqrt(n * products) / 12` and the threshold is `Q/2`.
    /// For backends opting into [`Backend::FFT64_ERROR_MODEL`], the rounding
    /// error model is `sigma_e = 2^(2*b-53) * sqrt(n*d*R) / 12`, where
    /// `d = products` and
    /// `R = 5*(log2(n)-1) + max(2/3 + (d+1)/6 - 1/(3*d), (d+1/2)/3)`.
    /// Its threshold is `1/2`; it includes transforms, products, and rounding
    /// of sequential partial sums before one inverse transform.
    ///
    /// Both use the Gaussian envelope `erfc(x) <= exp(-x*x)` and a union bound
    /// over `n` output coefficients. This is a parameter estimate under the
    /// stated model, not an arbitrary-input guarantee or a certified far-tail
    /// bound. Correlated operands and accumulated inputs require a suitable
    /// model of their own.
    ///
    /// Returns the largest radix up to 62 satisfying that envelope, or
    /// `Some(0)` if no positive radix does. Returns `None` for backends with
    /// neither CRT modulus metadata nor an enabled FFT64 error model.
    ///
    /// This maximum does not reserve coefficient-word headroom for additions
    /// and subtractions outside the DFT domain. Choose a smaller working radix
    /// when needed to keep every intermediate in range, including repeated
    /// accumulations before normalization and any carry-propagation bounds.
    ///
    /// For `m` output polynomials, add `ceil(log2(m))` to `failure_bits` to
    /// allocate the failure budget by a union bound. This function can be
    /// evaluated in a constant expression without constructing a module.
    ///
    /// # Panics
    ///
    /// Panics if `n` is not a power of two, is below [`Backend::MIN_DEGREE`],
    /// or if `products` or `failure_bits` is zero.
    #[inline]
    pub const fn max_base2k(n: usize, products: usize, failure_bits: usize) -> Option<usize> {
        assert!(n.is_power_of_two(), "n must be a power of two");
        assert!(n >= B::MIN_DEGREE, "n is below the backend's minimum degree");
        assert!(products > 0, "products must be positive");
        assert!(failure_bits > 0, "failure_bits must be positive");
        match <B::DftWord as crate::layouts::DftWord>::LOG_CRT_MODULUS {
            Some(log_q) => Some(super::base2k::max_base2k_ntt(log_q, n, products, failure_bits)),
            None if B::FFT64_ERROR_MODEL => Some(super::base2k::max_base2k_fft64(n, products, failure_bits)),
            None => None,
        }
    }

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

    /// Returns the largest ring degree `N` the module serves.
    #[inline]
    pub fn n(&self) -> usize {
        self.n as usize
    }

    /// Same as [`Self::n`].
    #[inline]
    pub fn max_n(&self) -> usize {
        self.n()
    }

    /// Allocates a zero-initialized backend-owned [`ScalarZnx`] of degree `n`.
    #[inline]
    pub fn scalar_znx_alloc(&self, n: usize, cols: usize) -> ScalarZnx<B::OwnedBuf, B::ZnxWord> {
        let len = B::bytes_of_scalar_znx(n, cols);
        let bytes = B::alloc_zeroed_bytes(len);
        ScalarZnx::from_data(bytes, n, cols)
    }

    /// Allocates a zero-initialized backend-owned [`VecZnx`] of degree `n`.
    #[inline]
    pub fn vec_znx_alloc(&self, n: usize, cols: usize, size: usize) -> VecZnx<B::OwnedBuf, B::ZnxWord> {
        vec_znx_alloc_zeroed::<B>(n, cols, size)
    }

    /// Returns the byte size of a [`VecZnx`] of degree `n`.
    #[inline]
    pub fn bytes_of_vec_znx(&self, n: usize, cols: usize, size: usize) -> usize {
        B::bytes_of_vec_znx(n, cols, size)
    }

    /// Allocates a zero-initialized backend-owned [`MatZnx`] of degree `n`.
    #[inline]
    pub fn mat_znx_alloc(
        &self,
        n: usize,
        rows: usize,
        cols_in: usize,
        cols_out: usize,
        size: usize,
    ) -> MatZnx<B::OwnedBuf, B::ZnxWord> {
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

/// Asserts that a module of degree `module_n` serves operands of degree `n`.
///
/// `n` must be a power of two with `BE::MIN_DEGREE <= n <= module_n`. Every
/// kernel calls this once at entry with the degree it read from its operands.
#[inline]
pub fn check_degree<BE: Backend>(module_n: usize, n: usize) {
    assert!(
        n.is_power_of_two() && n >= BE::MIN_DEGREE && n <= module_n,
        "degree {n} is not served by the module: it must be a power of two between {} and {module_n}",
        BE::MIN_DEGREE
    );
}

/// Computes [`GALOISGENERATOR`]`^|generator| * sign(generator) mod cyclotomic_order`.
///
/// Returns `1` when `generator == 0`.
///
/// # Panics
///
/// Asserts that `cyclotomic_order` is a positive power of two.
#[inline(always)]
pub fn galois_element(generator: i64, cyclotomic_order: i64) -> i64 {
    assert!(
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

#[cfg(test)]
mod degree_tests {
    use super::check_degree;
    use crate::layouts::HostBytesBackend;

    #[test]
    fn check_degree_accepts_powers_of_two_between_floor_and_module() {
        check_degree::<HostBytesBackend>(256, 256);
        check_degree::<HostBytesBackend>(256, 8);
        check_degree::<HostBytesBackend>(256, 64);
    }

    #[test]
    #[should_panic(expected = "degree 512 is not served by the module")]
    fn check_degree_rejects_above_module() {
        check_degree::<HostBytesBackend>(256, 512);
    }

    #[test]
    #[should_panic(expected = "degree 4 is not served by the module")]
    fn check_degree_rejects_below_floor() {
        check_degree::<HostBytesBackend>(256, 4);
    }

    #[test]
    #[should_panic(expected = "degree 24 is not served by the module")]
    fn check_degree_rejects_non_power_of_two() {
        check_degree::<HostBytesBackend>(256, 24);
    }
}
