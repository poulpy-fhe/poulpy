use std::{
    fmt,
    hash::{DefaultHasher, Hasher},
    marker::PhantomData,
};

use crate::{
    alloc_aligned,
    api::VecZnxNormalizeAssignBackend,
    layouts::{
        ArithmeticState, Backend, CoeffFitsIn, CoeffNormalized, CoeffUnnormalized, CoefficientState, Data, DataOwned, DataView,
        DataViewMut, DigestU64, FillUniform, HostDataMut, HostDataRef, ReaderFrom, ScalarZnx, ScratchArena, ToOwnedDeep,
        VecZnxInfos, WriterTo, ZnxInfos, ZnxView, ZnxWord, ZnxZero,
    },
    source::Source,
};

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use rand::Rng;

/// A vector of polynomials in `Z[X]/(X^N + 1)` with limb-decomposed
/// (base-2^k) representation.
///
/// This is the central data type of the crate. Each `VecZnx` contains
/// `cols` independent polynomial columns, each decomposed into `size`
/// limbs of `N` coefficients. Coefficients are [`ZnxWord`] values; the
/// word is always supplied by the backend via [`Backend::ZnxWord`] and has
/// no default, so the coefficient domain cannot silently decouple from it.
///
/// **Memory layout:** limb-major, column-minor. Limb `j` of column `i`
/// starts at scalar offset `N * (j * cols + i)`.
///
/// The type parameter `D` controls ownership: `Vec<u8>` for owned,
/// `&[u8]` for shared borrows, `&mut [u8]` for mutable borrows.
/// The type parameter `W` names the coefficient word (byte-layout
/// contract) of the buffer. The type parameter `S` is the
/// [`NormalizationState`] of the digits, [`Normalized`] by default; see that
/// trait for which operations produce, preserve, or require each state.
///
/// **Invariant:** `size` is both the working width and the allocated width, and
/// is fixed at construction. Operating on a narrower width is done through a
/// borrowed view (see [`vec_znx_backend_mut_with_size`]), never by mutating the
/// owner.
#[repr(C)]
#[derive(PartialEq, Eq, Clone, Copy, Hash, Debug, Default)]
pub struct VecZnxShape {
    n: usize,
    cols: usize,
    size: usize,
}

impl VecZnxShape {
    pub const fn new(n: usize, cols: usize, size: usize) -> Self {
        Self { n, cols, size }
    }

    pub const fn n(self) -> usize {
        self.n
    }

    pub const fn cols(self) -> usize {
        self.cols
    }

    pub const fn size(self) -> usize {
        self.size
    }

    /// Narrows the working width. Views can only ever shrink.
    pub(crate) const fn with_size(self, size: usize) -> Self {
        assert!(size <= self.size);
        Self { size, ..self }
    }
}

mod sealed {
    pub trait Sealed {}
}

/// Type-level carry-normalization state of a [`VecZnx`].
///
/// A limb vector is *normalized* when every digit `d` of every limb satisfies
/// `|d| <= 2^(base2k - 1)`, i.e. each limb fits within `base2k` bits. This is
/// the bound every DFT-domain primitive (`vec_znx_dft_apply`, VMP, SVP,
/// convolution) assumes; feeding it wider digits silently corrupts the result.
///
/// The state is tracked at the type level so that the compiler rejects
/// un-normalized inputs to those primitives:
///
/// * carry-producing ops (`add`, `sub`, `mul_xp_minus_one`, `add_normal`,
///   `lsh_add`, ...) only write into [`Unnormalized`] destinations,
/// * bound-preserving ops (`copy`, `negate`, `rotate`, `automorphism`,
///   `switch_ring`, ...) map a state into any state it [`FitsIn`],
/// * `normalize`, `big_normalize`, `lsh`, `rsh` produce normalized digits and
///   accept any destination state,
/// * DFT-domain primitives only read [`Normalized`] inputs.
///
/// Both markers are sealed: only [`Normalized`] and [`Unnormalized`] exist.
///
/// Relabelling [`Normalized`] as [`Unnormalized`] is always sound and free
/// ([`VecZnx::into_unnormalized`], [`VecZnx::into_state`]). The **only** way
/// back is a normalization pass: [`VecZnx::normalize`] (and the scratch-view
/// twin `VecZnxViewMut::normalize`) consume the [`Unnormalized`] value, run the
/// backend's normalization op over every column and return it relabelled. No
/// public constructor, conversion or method turns an [`Unnormalized`] value
/// into a [`Normalized`] one without that pass: the state field is private,
/// [`VecZnx::from_data`] and every allocator produce [`Normalized`] values, and
/// the relabel used internally by `normalize` is crate-private. The single
/// sanctioned exception is the backend-implementor extension point
/// [`crate::oep::SetNormalizationState`]: a fused backend kernel that
/// guarantees the digit bound by construction may set the state without the
/// pass, under an `unsafe` contract and only inside a backend crate.
///
/// State changes are by value: borrowed backend views inherit the state of
/// their owner, so a `&mut VecZnx<_, _, Normalized>` cannot accumulate carries
/// without first being relabelled. Relabelling a *borrowed view* leaves the
/// owner's label untouched; the digits behind a [`Normalized`] owner must then
/// be normalized in place (`vec_znx_normalize_assign`) before the borrow ends.
pub trait NormalizationState:
    sealed::Sealed + Copy + Clone + Default + fmt::Debug + PartialEq + Eq + std::hash::Hash + Send + Sync + 'static
{
    /// Conservative image of this legacy state in the coefficient-state algebra of
    /// [`crate::layouts::coeff_state`]: `Normalized` maps to
    /// `Coeff<Normalized, NonCanonical>` and `Unnormalized` to
    /// `Coeff<Unnormalized, NonCanonical>`, because today's roots prove nothing about
    /// canonical padding. Migration PRs use this mapping to bridge `type State` bounds
    /// onto [`crate::layouts::CoefficientState`] until roots carry the new parameter
    /// directly.
    type AsCoeff: crate::layouts::CoefficientState;
}

/// Marker: every limb digit fits within `base2k` bits. See [`NormalizationState`].
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq, Hash)]
pub struct Normalized;

/// Marker: limb digits may hold un-propagated carries wider than `base2k` bits. See [`NormalizationState`].
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq, Hash)]
pub struct Unnormalized;

impl sealed::Sealed for Normalized {}
impl sealed::Sealed for Unnormalized {}
impl NormalizationState for Normalized {
    type AsCoeff = crate::layouts::CoeffNormalized;
}
impl NormalizationState for Unnormalized {
    type AsCoeff = crate::layouts::CoeffUnnormalized;
}

/// Normalization axis marker of the coefficient-state algebra (spec §3.1): the sealed
/// subset of [`NormalizationState`] usable as the `N` parameter of
/// [`crate::layouts::Coeff`]. Implemented exactly by [`Normalized`] and
/// [`Unnormalized`].
pub trait Normalization: NormalizationState {}

impl Normalization for Normalized {}
impl Normalization for Unnormalized {}

/// `Self` digits are valid `S` digits.
///
/// [`Normalized`] fits in both states, [`Unnormalized`] only in itself. This is
/// the relation bound-preserving ops use to type their input against their
/// destination: `res: VecZnx<_, _, S>, a: VecZnx<_, _, impl FitsIn<S>>`.
pub trait FitsIn<S: NormalizationState>: NormalizationState {}

impl FitsIn<Normalized> for Normalized {}
/// Every state fits in [`Unnormalized`]; a blanket impl so that generic
/// `S: NormalizationState` inputs are accepted by unnormalized destinations.
impl<S: NormalizationState> FitsIn<Unnormalized> for S {}

#[repr(C)]
#[derive(PartialEq, Eq, Clone, Copy, Hash)]
pub struct VecZnx<D: Data, W: ZnxWord, S: CoefficientState = CoeffNormalized> {
    pub(crate) data: D,
    shape: VecZnxShape,
    _phantom: PhantomData<(W, S)>,
}

impl<D: Data + DataOwned, W: ZnxWord, S: CoefficientState> VecZnx<D, W, S> {
    /// Relabels this vector as [`Unnormalized`].
    ///
    /// Free and always sound: normalized digits are valid unnormalized digits.
    /// This is the entry point of a fusion loop; [`Self::normalize`] is its exit.
    ///
    /// Relabelling a *borrowed view* (rather than the owner) leaves the owner's
    /// label untouched, so a caller doing that must normalize the view in place
    /// before the borrow ends.
    pub fn into_unnormalized(self) -> VecZnx<D, W, CoeffUnnormalized> {
        VecZnx {
            data: self.data,
            shape: self.shape,
            _phantom: PhantomData,
        }
    }

    /// Relabels this vector into any state its current state [`FitsIn`].
    pub fn into_state<T: CoefficientState>(self) -> VecZnx<D, W, T>
    where
        S: CoeffFitsIn<T>,
    {
        VecZnx {
            data: self.data,
            shape: self.shape,
            _phantom: PhantomData,
        }
    }
}

impl<D: Data, W: ZnxWord, S: CoefficientState> VecZnx<D, W, S> {
    /// Relabels to an arbitrary state with no normalization pass.
    ///
    /// Crate-private on purpose: this is the only state relabel in the
    /// workspace. Its callers are [`Self::normalize`] and
    /// `VecZnxViewMut::normalize` (right after an in-place normalization pass)
    /// and the backend-implementor extension point
    /// [`crate::oep::SetNormalizationState`].
    pub(crate) fn relabel_unchecked<T: CoefficientState>(self) -> VecZnx<D, W, T> {
        VecZnx {
            data: self.data,
            shape: self.shape,
            _phantom: PhantomData,
        }
    }
}

impl<D: Data + DataOwned, W: ZnxWord> VecZnx<D, W, CoeffUnnormalized> {
    /// Propagates carries through every column and returns the vector relabelled as [`Normalized`].
    ///
    /// This is the only public path from [`Unnormalized`] to [`Normalized`].
    /// Only the top limb discards overflow. `scratch` must hold at least
    /// `vec_znx_normalize_tmp_bytes` bytes.
    pub fn normalize<M, B>(self, module: &M, base2k: usize, scratch: &mut ScratchArena<'_, B>) -> VecZnx<D, W, CoeffNormalized>
    where
        B: Backend<ZnxWord = W>,
        M: VecZnxNormalizeAssignBackend<B> + ?Sized,
        Self: VecZnxToBackendMut<B, State = CoeffUnnormalized>,
    {
        let mut me = self;
        {
            let mut view = me.to_backend_mut();
            for col in 0..view.cols() {
                module.vec_znx_normalize_assign_backend(base2k, &mut view, col, scratch);
            }
        }
        me.relabel_unchecked()
    }
}

impl<D: HostDataRef, W: ZnxWord, S: CoefficientState> VecZnx<D, W, S> {
    /// Returns a read-only [`ScalarZnx`] view of a single limb of a single column.
    pub fn as_scalar_znx_ref(&self, col: usize, limb: usize) -> ScalarZnx<&[u8], W> {
        ScalarZnx::from_data(bytemuck::cast_slice(self.at(col, limb)), self.n(), 1)
    }
}

impl<D: Data + Default, W: ZnxWord, S: CoefficientState> Default for VecZnx<D, W, S> {
    fn default() -> Self {
        Self {
            data: D::default(),
            shape: VecZnxShape::default(),
            _phantom: PhantomData,
        }
    }
}

impl<D: HostDataRef, W: ZnxWord, S: CoefficientState> DigestU64 for VecZnx<D, W, S> {
    fn digest_u64(&self) -> u64 {
        let mut h: DefaultHasher = DefaultHasher::new();
        h.write(self.data.as_ref());
        h.write_usize(self.n());
        h.write_usize(self.cols());
        h.write_usize(self.size());
        h.finish()
    }
}

impl<D: HostDataRef, W: ZnxWord, S: CoefficientState> ToOwnedDeep for VecZnx<D, W, S> {
    type Owned = VecZnx<Vec<u8>, W, S>;
    fn to_owned_deep(&self) -> Self::Owned {
        VecZnx {
            data: self.data.as_ref().to_vec(),
            shape: self.shape,
            _phantom: PhantomData,
        }
    }
}

impl<D: Data, W: ZnxWord, S: CoefficientState> VecZnx<D, W, S> {
    /// Rebuilds this backend-owned vector as a host-owned [`VecZnx<Vec<u8>>`].
    pub fn to_host_owned<BE>(&self) -> VecZnx<Vec<u8>, W, S>
    where
        BE: Backend<OwnedBuf = D>,
    {
        let shape = self.shape();
        VecZnx::from_data_with_state(
            crate::layouts::HostBytesBackend::from_bytes(BE::to_host_bytes(&self.data)),
            shape.n(),
            shape.cols(),
            shape.size(),
        )
    }

    /// Formats this backend-owned vector through the existing host [`fmt::Display`] implementation.
    pub fn display_host<BE>(&self) -> String
    where
        BE: Backend<OwnedBuf = D>,
    {
        self.to_host_owned::<BE>().to_string()
    }
}

impl<D: HostDataRef, W: ZnxWord, S: CoefficientState> fmt::Debug for VecZnx<D, W, S> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{self}")
    }
}

impl<D: Data, W: ZnxWord, S: CoefficientState> ZnxInfos for VecZnx<D, W, S> {
    fn n(&self) -> usize {
        self.shape.n()
    }

    fn size(&self) -> usize {
        self.shape.size()
    }

    fn poly_count(&self) -> usize {
        crate::layouts::checked_product(&[self.cols(), self.size()], "polynomial count")
    }
}

impl<D: Data, W: ZnxWord, S: CoefficientState> VecZnxInfos for VecZnx<D, W, S> {
    fn cols(&self) -> usize {
        self.shape.cols()
    }
}

impl<D: Data, W: ZnxWord, S: CoefficientState> DataView for VecZnx<D, W, S> {
    type D = D;
    fn data(&self) -> &Self::D {
        &self.data
    }
}

impl<D: Data, W: ZnxWord, S: CoefficientState> VecZnx<D, W, S> {
    /// Consumes the vector and returns its raw storage.
    ///
    /// The state label dies with the value: the extracted storage carries no claim, and
    /// re-wrapping it goes back through [`VecZnx::from_data`], the raw-ingestion trust
    /// boundary. Used by owning aggregates (LWE, plaintexts) that re-shape their
    /// storage between layouts.
    pub fn into_data(self) -> D {
        self.data
    }

    /// Mutable storage access for transfer and view plumbing (spec §7.4).
    ///
    /// The caller asserts the access does not change what the label claims: a
    /// byte-for-byte transfer of a value with the same state and representation
    /// (upload/download, host staging), or construction of an equally-typed view whose
    /// subsequent mutation is governed by that view's own state gate. It must not be
    /// used to edit digits: the typed-mutation route is [`DataViewMut`] (weakest state
    /// only) and kernels use [`crate::oep::vec_znx_kernel_words_mut`]. Every call site
    /// is counted by the migration deny-list ratchet.
    pub fn transfer_data_mut(&mut self) -> &mut D {
        &mut self.data
    }
}

/// Safe mutable storage access exists only for the weakest arithmetic state
/// (spec invariant 4): mutating bytes behind a `Coeff<Normalized, _>` or
/// `Coeff<_, Canonical>` label would silently invalidate the proof the label
/// carries. Kernels that write under a declared postcondition use the sealed
/// capability [`crate::oep::vec_znx_kernel_words_mut`] instead; via the
/// [`ZnxViewMut`] blanket impl this same gate governs `at_mut`/`raw_mut`.
impl<D: Data, W: ZnxWord> DataViewMut for VecZnx<D, W, CoeffUnnormalized> {
    fn data_mut(&mut self) -> &mut Self::D {
        &mut self.data
    }
}

impl<D: HostDataRef, W: ZnxWord, S: CoefficientState> ZnxView for VecZnx<D, W, S> {
    type Scalar = W;
}

impl<D: Data, W: ZnxWord, S: CoefficientState> VecZnx<D, W, S> {
    pub fn n(&self) -> usize {
        self.shape.n()
    }

    pub fn cols(&self) -> usize {
        self.shape.cols()
    }

    pub fn size(&self) -> usize {
        self.shape.size()
    }

    pub fn shape(&self) -> VecZnxShape {
        self.shape
    }
}

impl<D: Data, W: ZnxWord, S: CoefficientState> VecZnx<D, W, S> {
    /// Returns the scratch space (in bytes) required by right-shift operations.
    pub fn rsh_tmp_bytes(n: usize) -> usize {
        n * size_of::<W>()
    }
}

/// Zeroing is available for every state: zero words satisfy the normalized bound and
/// every canonical-padding rule, so the label's claim survives (spec invariant 6).
/// Implemented on the private storage directly because the [`DataViewMut`] route is
/// gated to the weakest state.
impl<D: HostDataMut, W: ZnxWord, S: CoefficientState> ZnxZero for VecZnx<D, W, S> {
    fn zero(&mut self) {
        let span = self.n() * self.cols() * self.size();
        let bytes: &mut [u8] = self.data.as_mut();
        bytemuck::cast_slice_mut::<u8, W>(&mut bytes[..span * size_of::<W>()]).fill(W::zero())
    }
    fn zero_at(&mut self, i: usize, j: usize) {
        assert!(i < self.cols() && j < self.size());
        let n = self.n();
        let offset = (j * self.cols() + i) * n;
        let span = n * self.cols() * self.size();
        let bytes: &mut [u8] = self.data.as_mut();
        bytemuck::cast_slice_mut::<u8, W>(&mut bytes[..span * size_of::<W>()])[offset..offset + n].fill(W::zero());
    }
}

impl<D: Data, W: ZnxWord, S: CoefficientState> VecZnx<D, W, S> {
    /// Returns the number of bytes required: `n * cols * size * size_of::<W>()`.
    pub fn bytes_of(n: usize, cols: usize, size: usize) -> usize {
        crate::layouts::checked_product(&[n, cols, size, size_of::<W>()], "VecZnx byte size")
    }
}

impl<W: ZnxWord> VecZnx<Vec<u8>, W> {
    /// Allocates a zero-initialized `VecZnx` aligned to [`DEFAULTALIGN`](crate::DEFAULTALIGN).
    pub(crate) fn alloc(n: usize, cols: usize, size: usize) -> Self {
        let data: Vec<u8> = alloc_aligned::<u8>(Self::bytes_of(n, cols, size));
        Self {
            data,
            shape: VecZnxShape::new(n, cols, size),
            _phantom: PhantomData,
        }
    }

    /// Wraps an existing byte buffer into a `VecZnx`.
    ///
    /// # Panics
    ///
    /// Panics if the buffer length does not equal `bytes_of(n, cols, size)` or
    /// the buffer is not aligned to [`DEFAULTALIGN`](crate::DEFAULTALIGN).
    pub fn from_bytes(n: usize, cols: usize, size: usize, bytes: impl Into<Vec<u8>>) -> Self {
        let data: Vec<u8> = bytes.into();
        assert!(
            data.len() == Self::bytes_of(n, cols, size),
            "from_bytes: data.len()={} != bytes_of({}, {}, {})={}",
            data.len(),
            n,
            cols,
            size,
            Self::bytes_of(n, cols, size)
        );
        crate::assert_alignment(data.as_ptr());
        Self {
            data,
            shape: VecZnxShape::new(n, cols, size),
            _phantom: PhantomData,
        }
    }
}

impl<D: Data, W: ZnxWord, S: CoefficientState> VecZnx<D, W, S> {
    /// Rebuilds a vector around new storage while keeping the state of the value it was copied
    /// or borrowed from (transfers, host views). Crate-private so that no external path can pick
    /// an arbitrary state for raw data; see [`VecZnx::from_data`].
    pub(crate) fn from_data_with_state(data: D, n: usize, cols: usize, size: usize) -> Self {
        Self {
            data,
            shape: VecZnxShape::new(n, cols, size),
            _phantom: PhantomData,
        }
    }
}

impl<D: Data, W: ZnxWord> VecZnx<D, W, CoeffNormalized> {
    /// Constructs a `VecZnx` from raw parts without validation.
    ///
    /// Raw ingestion is a trust boundary (like deserialization): the result is
    /// labelled [`Normalized`]. Call [`Self::into_unnormalized`] on it to obtain a
    /// carry-accumulating destination; there is no way to build an
    /// [`Unnormalized`] value and later claim it normalized without
    /// [`Self::normalize`].
    pub fn from_data(data: D, n: usize, cols: usize, size: usize) -> Self {
        Self {
            data,
            shape: VecZnxShape::new(n, cols, size),
            _phantom: PhantomData,
        }
    }
}

impl<D: Data, W: ZnxWord> VecZnx<D, W, CoeffUnnormalized> {
    /// Constructs a `VecZnx` from raw parts under the weakest arithmetic label.
    ///
    /// Unlike [`VecZnx::from_data`] (the trust boundary that claims
    /// [`CoeffNormalized`]), this claims nothing beyond "readable digits", so it is
    /// safe for any storage: the only way to a stronger label from here is a real
    /// normalization pass. Used for reborrowing big-accumulator storage as an
    /// unnormalized coefficient view inside kernels.
    pub fn from_data_unnormalized(data: D, n: usize, cols: usize, size: usize) -> Self {
        Self::from_data_with_state(data, n, cols, size)
    }
}

impl<D: HostDataRef, W: ZnxWord, S: CoefficientState> fmt::Display for VecZnx<D, W, S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "VecZnx(n={}, cols={}, size={})", self.n(), self.cols(), self.size())?;

        for col in 0..self.cols() {
            writeln!(f, "Column {col}:")?;
            for size in 0..self.size() {
                let coeffs = self.at(col, size);
                write!(f, "  Size {size}: [")?;

                let max_show = 16;
                let show_count = coeffs.len().min(max_show);

                for (i, &coeff) in coeffs.iter().take(show_count).enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{coeff}")?;
                }

                if coeffs.len() > max_show {
                    write!(f, ", ... ({} more)", coeffs.len() - max_show)?;
                }

                writeln!(f, "]")?;
            }
        }
        Ok(())
    }
}

impl<D: HostDataMut, W: ZnxWord, S: CoefficientState> FillUniform for VecZnx<D, W, S> {
    fn fill_uniform(&mut self, log_bound: usize, source: &mut Source) {
        assert!(log_bound != 0, "invalid log_bound, cannot be zero");
        assert!(
            log_bound <= W::BITS,
            "log_bound {log_bound} exceeds the {}-bit coefficient word",
            W::BITS
        );
        if log_bound == W::BITS {
            source.fill_bytes(self.data.as_mut());
            return;
        }
        let mask: u64 = (1u64 << log_bound) - 1;
        let shift: usize = 64 - log_bound;
        let span = self.n() * self.cols() * self.size();
        let bytes: &mut [u8] = self.data.as_mut();
        for x in bytemuck::cast_slice_mut::<u8, W>(&mut bytes[..span * size_of::<W>()]).iter_mut() {
            let r = source.next_u64() & mask;
            *x = W::from_i64(((r << shift) as i64) >> shift);
        }
    }
}

/// Owned `VecZnx` backed by a `Vec<u8>`.
pub type VecZnxOwned<W, S = CoeffNormalized> = VecZnx<Vec<u8>, W, S>;
/// Mutably borrowed `VecZnx`.
pub type VecZnxMut<'a, W, S = CoeffNormalized> = VecZnx<&'a mut [u8], W, S>;
/// Immutably borrowed `VecZnx`.
pub type VecZnxRef<'a, W, S = CoeffNormalized> = VecZnx<&'a [u8], W, S>;
/// Shared backend-native borrow of a `VecZnx`.
pub type VecZnxBackendRef<'a, B, S = CoeffNormalized> = VecZnx<<B as Backend>::BufRef<'a>, <B as Backend>::ZnxWord, S>;
/// Mutable backend-native borrow of a `VecZnx`.
pub type VecZnxBackendMut<'a, B, S = CoeffNormalized> = VecZnx<<B as Backend>::BufMut<'a>, <B as Backend>::ZnxWord, S>;

/// Returns a shared backend-native scalar view into a backend-owned `VecZnx`.
pub trait VecZnxAsScalarBackendRef<B: Backend> {
    fn as_scalar_znx_backend_ref(&self, col: usize, limb: usize) -> ScalarZnx<B::BufRef<'_>, B::ZnxWord>;
}

impl<B: Backend, S: CoefficientState> VecZnxAsScalarBackendRef<B> for VecZnx<B::OwnedBuf, B::ZnxWord, S> {
    fn as_scalar_znx_backend_ref(&self, col: usize, limb: usize) -> ScalarZnx<B::BufRef<'_>, B::ZnxWord> {
        assert!(limb < self.size(), "size: {limb} >= {}", self.size());
        assert!(col < self.cols(), "cols: {col} >= {}", self.cols());
        let start: usize = limb
            .checked_mul(self.cols())
            .and_then(|x| x.checked_add(col))
            .and_then(|x| x.checked_mul(self.n()))
            .and_then(|x| x.checked_mul(B::size_of_znx_word()))
            .expect("VecZnx scalar backend view offset overflows usize");
        let len: usize = self
            .n()
            .checked_mul(B::size_of_znx_word())
            .expect("VecZnx scalar backend view length overflows usize");
        ScalarZnx::from_data(B::region(&self.data, start, len), self.n(), 1)
    }
}

/// Returns a mutable backend-native scalar view into a backend-owned `VecZnx`.
pub trait VecZnxAsScalarBackendMut<B: Backend> {
    fn as_scalar_znx_backend_mut(&mut self, col: usize, limb: usize) -> ScalarZnx<B::BufMut<'_>, B::ZnxWord>;
}

impl<B: Backend, S: CoefficientState> VecZnxAsScalarBackendMut<B> for VecZnx<B::OwnedBuf, B::ZnxWord, S> {
    fn as_scalar_znx_backend_mut(&mut self, col: usize, limb: usize) -> ScalarZnx<B::BufMut<'_>, B::ZnxWord> {
        let n = self.n();
        assert!(limb < self.size(), "size: {limb} >= {}", self.size());
        assert!(col < self.cols(), "cols: {col} >= {}", self.cols());
        let start: usize = limb
            .checked_mul(self.cols())
            .and_then(|x| x.checked_add(col))
            .and_then(|x| x.checked_mul(n))
            .and_then(|x| x.checked_mul(B::size_of_znx_word()))
            .expect("VecZnx scalar backend view offset overflows usize");
        let len: usize = n
            .checked_mul(B::size_of_znx_word())
            .expect("VecZnx scalar backend view length overflows usize");
        ScalarZnx::from_data(B::region_mut(&mut self.data, start, len), n, 1)
    }
}

/// Borrow a backend-owned `VecZnx` using the backend's native view type.
///
/// The view inherits the owner's [`NormalizationState`] through
/// [`Self::State`]: bounds such as `T: VecZnxToBackendRef<B, State = Normalized>`
/// are how an op restricts the state of what it reads.
pub trait VecZnxToBackendRef<B: Backend = crate::layouts::HostBytesBackend> {
    type State: ArithmeticState;
    fn to_backend_ref(&self) -> VecZnxBackendRef<'_, B, Self::State>;
}

impl<B: Backend, S: ArithmeticState> VecZnxToBackendRef<B> for VecZnx<B::OwnedBuf, B::ZnxWord, S> {
    type State = S;
    fn to_backend_ref(&self) -> VecZnxBackendRef<'_, B, S> {
        VecZnx {
            data: B::view(&self.data),
            shape: self.shape,
            _phantom: PhantomData,
        }
    }
}

impl<'b, B: Backend + 'b, S: ArithmeticState> VecZnxToBackendRef<B> for &VecZnx<B::BufRef<'b>, B::ZnxWord, S> {
    type State = S;
    fn to_backend_ref(&self) -> VecZnxBackendRef<'_, B, S> {
        vec_znx_backend_ref_from_ref::<B, S>(self)
    }
}

impl<S: ArithmeticState> VecZnxToBackendRef<crate::layouts::HostBytesBackend> for VecZnx<&mut [u8], i64, S> {
    type State = S;
    fn to_backend_ref(&self) -> VecZnxBackendRef<'_, crate::layouts::HostBytesBackend, S> {
        VecZnx {
            data: self.data,
            shape: self.shape,
            _phantom: PhantomData,
        }
    }
}

impl<S: ArithmeticState> VecZnxToBackendRef<crate::layouts::HostBytesBackend> for VecZnx<&[u8], i64, S> {
    type State = S;
    fn to_backend_ref(&self) -> VecZnxBackendRef<'_, crate::layouts::HostBytesBackend, S> {
        VecZnx {
            data: self.data,
            shape: self.shape,
            _phantom: PhantomData,
        }
    }
}

/// Reborrow an already backend-borrowed `VecZnx` as a shared backend-native view.
pub trait VecZnxReborrowBackendRef<B: Backend = crate::layouts::HostBytesBackend> {
    type State: ArithmeticState;
    fn reborrow_backend_ref(&self) -> VecZnxBackendRef<'_, B, Self::State>;
}

pub fn vec_znx_backend_ref_from_ref<'a, 'b, B: Backend + 'b, S: CoefficientState>(
    vec: &'a VecZnx<B::BufRef<'b>, B::ZnxWord, S>,
) -> VecZnxBackendRef<'a, B, S> {
    VecZnx {
        data: B::view_ref(&vec.data),
        shape: vec.shape,
        _phantom: PhantomData,
    }
}

pub fn vec_znx_backend_ref_from_mut<'a, 'b, B: Backend + 'b, S: CoefficientState>(
    vec: &'a VecZnx<B::BufMut<'b>, B::ZnxWord, S>,
) -> VecZnxBackendRef<'a, B, S> {
    VecZnx {
        data: B::view_ref_mut(&vec.data),
        shape: vec.shape,
        _phantom: PhantomData,
    }
}

impl<'b, B: Backend + 'b, S: ArithmeticState> VecZnxReborrowBackendRef<B> for VecZnx<B::BufMut<'b>, B::ZnxWord, S> {
    type State = S;
    fn reborrow_backend_ref(&self) -> VecZnxBackendRef<'_, B, S> {
        vec_znx_backend_ref_from_mut::<B, S>(self)
    }
}

/// Mutably borrow a backend-owned `VecZnx` using the backend's native view type.
///
/// The view inherits the owner's [`NormalizationState`] through
/// [`Self::State`]; a carry-producing op therefore demands
/// `R: VecZnxToBackendMut<B, State = Unnormalized>` for its destination.
pub trait VecZnxToBackendMut<B: Backend = crate::layouts::HostBytesBackend> {
    type State: ArithmeticState;
    fn to_backend_mut(&mut self) -> VecZnxBackendMut<'_, B, Self::State>;
}

impl<B: Backend, S: ArithmeticState> VecZnxToBackendMut<B> for VecZnx<B::OwnedBuf, B::ZnxWord, S> {
    type State = S;
    fn to_backend_mut(&mut self) -> VecZnxBackendMut<'_, B, S> {
        VecZnx {
            data: B::view_mut(&mut self.data),
            shape: self.shape,
            _phantom: PhantomData,
        }
    }
}

impl<'b, B: Backend + 'b, S: ArithmeticState> VecZnxToBackendMut<B> for &mut VecZnx<B::BufMut<'b>, B::ZnxWord, S> {
    type State = S;
    fn to_backend_mut(&mut self) -> VecZnxBackendMut<'_, B, S> {
        vec_znx_backend_mut_from_mut::<B, S>(self)
    }
}

impl<S: ArithmeticState> VecZnxToBackendMut<crate::layouts::HostBytesBackend> for VecZnx<&mut [u8], i64, S> {
    type State = S;
    fn to_backend_mut(&mut self) -> VecZnxBackendMut<'_, crate::layouts::HostBytesBackend, S> {
        VecZnx {
            data: self.data,
            shape: self.shape,
            _phantom: PhantomData,
        }
    }
}

/// Reborrow an already backend-borrowed `VecZnx` as a mutable backend-native view.
pub trait VecZnxReborrowBackendMut<B: Backend = crate::layouts::HostBytesBackend> {
    type State: ArithmeticState;
    fn reborrow_backend_mut(&mut self) -> VecZnxBackendMut<'_, B, Self::State>;
}

pub fn vec_znx_host_backend_ref<D: HostDataRef, S: CoefficientState>(
    vec: &VecZnx<D, i64, S>,
) -> VecZnxBackendRef<'_, crate::layouts::HostBytesBackend, S> {
    VecZnx {
        data: vec.data.as_ref(),
        shape: vec.shape,
        _phantom: PhantomData,
    }
}

pub fn vec_znx_host_backend_mut<D: HostDataMut, S: CoefficientState>(
    vec: &mut VecZnx<D, i64, S>,
) -> VecZnxBackendMut<'_, crate::layouts::HostBytesBackend, S> {
    VecZnx {
        data: vec.data.as_mut(),
        shape: vec.shape,
        _phantom: PhantomData,
    }
}

pub fn vec_znx_backend_mut_from_mut<'a, 'b, B: Backend + 'b, S: CoefficientState>(
    vec: &'a mut VecZnx<B::BufMut<'b>, B::ZnxWord, S>,
) -> VecZnxBackendMut<'a, B, S> {
    VecZnx {
        data: B::view_mut_ref(&mut vec.data),
        shape: vec.shape,
        _phantom: PhantomData,
    }
}

impl<'b, B: Backend + 'b, S: ArithmeticState> VecZnxReborrowBackendMut<B> for VecZnx<B::BufMut<'b>, B::ZnxWord, S> {
    type State = S;
    fn reborrow_backend_mut(&mut self) -> VecZnxBackendMut<'_, B, S> {
        vec_znx_backend_mut_from_mut::<B, S>(self)
    }
}

/// Borrows a backend-owned `VecZnx` as the backend's native shared view.
///
/// [`VecZnxToBackendRef`] is keyed on `B::OwnedBuf`, a projection the compiler
/// cannot invert, so `vec.to_backend_ref()` cannot infer `B` the way the
/// backend-keyed containers (`VecZnxDft`, `VecZnxBig`) can. This names the
/// backend once, as a turbofish, instead of spelling the qualified path.
pub fn vec_znx_backend_ref<'a, B: Backend, S: ArithmeticState>(
    vec: &'a VecZnx<B::OwnedBuf, B::ZnxWord, S>,
) -> VecZnxBackendRef<'a, B, S> {
    <VecZnx<B::OwnedBuf, B::ZnxWord, S> as VecZnxToBackendRef<B>>::to_backend_ref(vec)
}

/// Reborrows an existing native mutable view for a shorter lifetime.
///
/// Same inference problem as [`vec_znx_backend_mut`]: the impl is keyed on
/// `B::BufMut`, so the backend is named once here instead of at each call.
pub fn vec_znx_reborrow_backend_mut<'a, B: Backend, S: ArithmeticState>(
    vec: &'a mut VecZnxBackendMut<'_, B, S>,
) -> VecZnxBackendMut<'a, B, S> {
    <VecZnx<B::BufMut<'_>, B::ZnxWord, S> as VecZnxReborrowBackendMut<B>>::reborrow_backend_mut(vec)
}

/// Borrows a backend-owned `VecZnx` as the backend's native mutable view.
///
/// See [`vec_znx_backend_ref`] for why this exists rather than a bare
/// `to_backend_mut()`.
pub fn vec_znx_backend_mut<'a, B: Backend, S: ArithmeticState>(
    vec: &'a mut VecZnx<B::OwnedBuf, B::ZnxWord, S>,
) -> VecZnxBackendMut<'a, B, S> {
    <VecZnx<B::OwnedBuf, B::ZnxWord, S> as VecZnxToBackendMut<B>>::to_backend_mut(vec)
}

/// Weakened *read-only* use of a stronger-labelled value: reads its digits under the
/// weakest arithmetic label. Always sound when the result is only read (a shared view
/// cannot mutate, so no owner label can go stale); used by consumers typed against
/// unnormalized inputs. Implemented for the state-carrying containers; the relabel
/// needs no backend parameter, so plain method syntax works everywhere.
pub trait WeakenBackendRef: Sized {
    /// `Self` relabelled to the weakest arithmetic state.
    type Weakened;
    /// Relabels the view; the caller asserts it will only read through the result.
    fn weaken_backend_ref(self) -> Self::Weakened;
}

impl<D: Data, W: ZnxWord, S: ArithmeticState> WeakenBackendRef for VecZnx<D, W, S> {
    type Weakened = VecZnx<D, W, CoeffUnnormalized>;
    fn weaken_backend_ref(self) -> VecZnx<D, W, CoeffUnnormalized> {
        self.relabel_unchecked()
    }
}

/// TRANSITIONAL containment bridge (spec §4.1; removed by the PR 5 scratch-transaction
/// migration): relabels a mutable *borrowed* view as unnormalized so a carry-producing
/// op can write through it while the owner keeps its label.
///
/// The caller contract is the documented §2.1 pattern under containment: the write must
/// either provably preserve the owner's digit bound, or the caller must re-normalize
/// the same storage in place (`vec_znx_normalize_assign`) before the borrow ends and
/// the owner is next read. Every call site is counted by the migration deny-list
/// ratchet and inventoried as a §6.4-B site.
pub trait BorrowedCarryView: Sized {
    /// `Self` relabelled to the weakest arithmetic state, for a contracted carry write.
    type Weakened;
    /// Relabels the view under the containment contract above.
    fn borrowed_carry_view(self) -> Self::Weakened;
}

impl<D: Data, W: ZnxWord, S: ArithmeticState> BorrowedCarryView for VecZnx<D, W, S> {
    type Weakened = VecZnx<D, W, CoeffUnnormalized>;
    fn borrowed_carry_view(self) -> VecZnx<D, W, CoeffUnnormalized> {
        self.relabel_unchecked()
    }
}

/// Narrows a mutable backend view to a smaller working size.
///
/// The returned view addresses the same allocation, but HAL kernels see `size`
/// as the active limb count. This is the only way to operate on fewer limbs
/// than a `VecZnx` was allocated with: the owner's own size never changes.
///
/// Free-standing rather than inherent because `VecZnxBackendMut<'_, B>` reaches
/// `B` only through associated types, which leaves `B` unconstrained in an
/// `impl` header.
///
/// # Panics
///
/// Panics if `size > vec.size()`.
pub fn vec_znx_backend_mut_with_size<'a, B: Backend, S: CoefficientState>(
    vec: VecZnxBackendMut<'a, B, S>,
    size: usize,
) -> VecZnxBackendMut<'a, B, S> {
    VecZnx {
        data: vec.data,
        shape: vec.shape.with_size(size),
        _phantom: PhantomData,
    }
}

impl<D: HostDataMut, W: ZnxWord, S: CoefficientState> ReaderFrom for VecZnx<D, W, S> {
    fn read_from<R: std::io::Read>(&mut self, reader: &mut R) -> std::io::Result<()> {
        // Read into temporaries first to avoid leaving self in an inconsistent state on error.
        let new_n: usize = reader.read_u64::<LittleEndian>()? as usize;
        let new_cols: usize = reader.read_u64::<LittleEndian>()? as usize;
        let new_size: usize = reader.read_u64::<LittleEndian>()? as usize;
        let len: usize = reader.read_u64::<LittleEndian>()? as usize;

        // Validate metadata consistency: n * cols * size * sizeof(W) must match data length.
        let expected_len: usize =
            crate::layouts::checked_product(&[new_n, new_cols, new_size, size_of::<W>()], "VecZnx serialized byte size");
        if expected_len != len {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "VecZnx metadata inconsistent: n={new_n} * cols={new_cols} * size={new_size} * {} = {expected_len} != data len={len}",
                    size_of::<W>()
                ),
            ));
        }

        let buf: &mut [u8] = self.data.as_mut();
        if buf.len() < len {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("VecZnx buffer too small: self.data.len()={} < read len={len}", buf.len()),
            ));
        }
        reader.read_exact(&mut buf[..len])?;

        // Only commit metadata after successful read.
        self.shape = VecZnxShape::new(new_n, new_cols, new_size);
        Ok(())
    }
}

impl<D: HostDataRef, W: ZnxWord, S: CoefficientState> WriterTo for VecZnx<D, W, S> {
    fn write_to<Wr: std::io::Write>(&self, writer: &mut Wr) -> std::io::Result<()> {
        writer.write_u64::<LittleEndian>(self.n() as u64)?;
        writer.write_u64::<LittleEndian>(self.cols() as u64)?;
        writer.write_u64::<LittleEndian>(self.size() as u64)?;
        let coeff_bytes: usize = crate::layouts::checked_product(
            &[self.n(), self.cols(), self.size(), size_of::<W>()],
            "VecZnx logical byte size",
        );
        let buf: &[u8] = self.data.as_ref();
        if buf.len() < coeff_bytes {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "VecZnx buffer too small: self.data.len()={} < coeff_bytes={coeff_bytes}",
                    buf.len()
                ),
            ));
        }
        writer.write_u64::<LittleEndian>(coeff_bytes as u64)?;
        writer.write_all(&buf[..coeff_bytes])?;
        Ok(())
    }
}
