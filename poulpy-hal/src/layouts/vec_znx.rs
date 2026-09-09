use std::{
    fmt,
    hash::{DefaultHasher, Hasher},
    marker::PhantomData,
};

use crate::{
    alloc_aligned,
    layouts::{
        Backend, Data, DataView, DataViewMut, DigestU64, FillUniform, HostDataMut, HostDataRef, ReaderFrom, ScalarZnx,
        ToOwnedDeep, VecZnxInfos, WriterTo, ZnxInfos, ZnxView, ZnxViewMut, ZnxWord, ZnxZero,
    },
    source::Source,
};

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use rand::Rng;

/// Geometry of a vector-shaped container plus an optional window onto it.
///
/// The dense buffer holds `cols` columns of limbs of `n_full` coefficients,
/// limb-major, column-minor: limb `j` of column `i` starts at scalar offset
/// `(j * cols + i) * n_full`. A window restricts what a view sees:
/// coefficients `coeff_offset..coeff_offset + n` of limbs
/// `limb_offset, limb_offset + limb_step, ...` (`size` of them). A freshly
/// constructed shape is dense (`is_dense()`), and every window is an affine
/// restriction of the box, so windows compose.
#[repr(C)]
#[derive(PartialEq, Eq, Clone, Copy, Hash, Debug)]
pub struct VecZnxShape {
    n_full: usize,
    cols: usize,
    coeff_offset: usize,
    n: usize,
    limb_offset: usize,
    limb_step: usize,
    size: usize,
}

impl Default for VecZnxShape {
    fn default() -> Self {
        Self::new(0, 0, 0)
    }
}

impl VecZnxShape {
    /// Dense shape: `n` coefficients per limb, `cols` columns, `size` limbs.
    pub const fn new(n: usize, cols: usize, size: usize) -> Self {
        Self {
            n_full: n,
            cols,
            coeff_offset: 0,
            n,
            limb_offset: 0,
            limb_step: 1,
            size,
        }
    }

    /// Visible coefficients per limb.
    pub const fn n(self) -> usize {
        self.n
    }
    pub const fn cols(self) -> usize {
        self.cols
    }
    /// Visible limb count.
    pub const fn size(self) -> usize {
        self.size
    }
    /// Coefficients per limb in the dense buffer.
    pub const fn n_full(self) -> usize {
        self.n_full
    }
    pub const fn coeff_offset(self) -> usize {
        self.coeff_offset
    }
    pub const fn limb_offset(self) -> usize {
        self.limb_offset
    }
    pub const fn limb_step(self) -> usize {
        self.limb_step
    }

    /// True when the view is a contiguous prefix of the dense buffer.
    pub const fn is_dense(self) -> bool {
        self.coeff_offset == 0 && self.n == self.n_full && self.limb_offset == 0 && self.limb_step == 1
    }

    /// Index of the `(col, limb)` block in the dense buffer, in blocks of `n_full` scalars.
    pub const fn block_index(self, col: usize, limb: usize) -> usize {
        (self.limb_offset + limb * self.limb_step) * self.cols + col
    }

    /// Scalar offset of visible coefficient 0 of the `(col, limb)` block.
    pub const fn scalar_offset(self, col: usize, limb: usize) -> usize {
        self.block_index(col, limb) * self.n_full + self.coeff_offset
    }

    /// Restricts the view to coefficients `offset..offset + len` of every visible limb.
    ///
    /// # Panics
    ///
    /// Panics if `len == 0` or `offset + len > self.n()`.
    pub const fn window_coeffs(self, offset: usize, len: usize) -> Self {
        assert!(len >= 1, "window_coeffs: len must be >= 1");
        assert!(offset + len <= self.n, "window_coeffs: offset + len exceeds visible n");
        Self {
            coeff_offset: self.coeff_offset + offset,
            n: len,
            ..self
        }
    }

    /// Restricts the view to visible limbs `offset, offset + step, ...` (`count` of them).
    ///
    /// # Panics
    ///
    /// Panics if `step == 0`, `count == 0`, or the last selected limb is outside the view.
    pub const fn window_limbs(self, offset: usize, step: usize, count: usize) -> Self {
        assert!(step >= 1, "window_limbs: step must be >= 1");
        assert!(count >= 1, "window_limbs: count must be >= 1");
        assert!(
            offset + (count - 1) * step < self.size,
            "window_limbs: last limb exceeds visible size"
        );
        Self {
            limb_offset: self.limb_offset + offset * self.limb_step,
            limb_step: self.limb_step * step,
            size: count,
            ..self
        }
    }

    /// Narrows the working width. Views can only ever shrink.
    pub(crate) const fn with_size(self, size: usize) -> Self {
        assert!(size <= self.size);
        Self { size, ..self }
    }
}

/// A vector of polynomials in `Z[X]/(X^N + 1)` with limb-decomposed
/// (base-2^k) representation.
///
/// This is the central data type of the crate. Each `VecZnx` contains
/// `cols` independent polynomial columns, each decomposed into `size`
/// limbs of `N` coefficients. Coefficients are [`ZnxWord`] values; the
/// word is always supplied by the backend via [`Backend::ZnxWord`] and has
/// no default, so the coefficient domain cannot silently decouple from it.
///
/// **Memory layout:** see [`VecZnxShape`].
///
/// The type parameter `D` controls ownership: `Vec<u8>` for owned,
/// `&[u8]` for shared borrows, `&mut [u8]` for mutable borrows.
/// The type parameter `W` names the coefficient word (byte-layout
/// contract) of the buffer.
///
/// **Invariant:** `size` is both the working width and the allocated width, and
/// is fixed at construction. Operating on a narrower width is done through a
/// borrowed view (see [`vec_znx_backend_mut_with_size`]), never by mutating the
/// owner.
///
/// See [layouts](crate::layouts#value-model) for the container value model
/// this type implements, and [windows](crate::layouts#windows) for the
/// semantics of `window_coeffs`/`window_limbs`.
#[repr(C)]
#[derive(PartialEq, Eq, Clone, Hash)]
pub struct VecZnx<D: Data, W: ZnxWord> {
    data: D,
    shape: VecZnxShape,
    pub _phantom: PhantomData<W>,
}

impl<D: HostDataRef, W: ZnxWord> VecZnx<D, W> {
    /// Returns a read-only [`ScalarZnx`] view of a single limb of a single column.
    pub fn as_scalar_znx_ref(&self, col: usize, limb: usize) -> ScalarZnx<&[u8], W> {
        ScalarZnx::from_data(bytemuck::cast_slice(self.at(col, limb)), self.n(), 1)
    }
}

impl<D: HostDataMut, W: ZnxWord> VecZnx<D, W> {
    /// Returns a mutable [`ScalarZnx`] view of a single limb of a single column.
    pub fn as_scalar_znx_mut(&mut self, col: usize, limb: usize) -> ScalarZnx<&mut [u8], W> {
        let n = self.n();
        ScalarZnx::from_data(bytemuck::cast_slice_mut(self.at_mut(col, limb)), n, 1)
    }
}

impl<D: Data + Default, W: ZnxWord> Default for VecZnx<D, W> {
    fn default() -> Self {
        Self {
            data: D::default(),
            shape: VecZnxShape::default(),
            _phantom: PhantomData,
        }
    }
}

impl<D: HostDataRef, W: ZnxWord> DigestU64 for VecZnx<D, W> {
    fn digest_u64(&self) -> u64 {
        crate::layouts::assert_dense(self, "VecZnx::digest_u64");
        let mut h: DefaultHasher = DefaultHasher::new();
        h.write(self.data.as_ref());
        h.write_usize(self.n());
        h.write_usize(self.cols());
        h.write_usize(self.size());
        h.finish()
    }
}

impl<D: HostDataRef, W: ZnxWord> ToOwnedDeep for VecZnx<D, W> {
    type Owned = VecZnx<Vec<u8>, W>;
    fn to_owned_deep(&self) -> Self::Owned {
        crate::layouts::assert_dense(self, "VecZnx::to_owned_deep");
        VecZnx {
            data: self.data.as_ref().to_vec(),
            shape: self.shape,
            _phantom: PhantomData,
        }
    }
}

impl<D: Data, W: ZnxWord> VecZnx<D, W> {
    /// Rebuilds this backend-owned vector as a host-owned [`VecZnx<Vec<u8>>`].
    pub fn to_host_owned<BE>(&self) -> VecZnx<Vec<u8>, W>
    where
        BE: Backend<OwnedBuf = D>,
    {
        crate::layouts::assert_dense(self, "VecZnx::to_host_owned");
        let shape = self.shape();
        VecZnx::from_shape(
            crate::layouts::HostBytesBackend::from_bytes(BE::to_host_bytes(&self.data)),
            shape,
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

impl<D: HostDataRef, W: ZnxWord> fmt::Debug for VecZnx<D, W> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{self}")
    }
}

impl<D: Data, W: ZnxWord> ZnxInfos for VecZnx<D, W> {
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

impl<D: Data, W: ZnxWord> VecZnxInfos for VecZnx<D, W> {
    fn cols(&self) -> usize {
        self.shape.cols()
    }
    fn n_full(&self) -> usize {
        self.shape.n_full()
    }
    fn coeff_offset(&self) -> usize {
        self.shape.coeff_offset()
    }
    fn limb_offset(&self) -> usize {
        self.shape.limb_offset()
    }
    fn limb_step(&self) -> usize {
        self.shape.limb_step()
    }
}

impl<D: Data, W: ZnxWord> DataView for VecZnx<D, W> {
    type D = D;
    fn data(&self) -> &Self::D {
        &self.data
    }
}

impl<D: Data, W: ZnxWord> DataViewMut for VecZnx<D, W> {
    fn data_mut(&mut self) -> &mut Self::D {
        &mut self.data
    }
}

impl<D: HostDataRef, W: ZnxWord> ZnxView for VecZnx<D, W> {
    type Scalar = W;
}

impl<D: Data, W: ZnxWord> VecZnx<D, W> {
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

    pub fn data(&self) -> &D {
        &self.data
    }

    pub fn data_mut(&mut self) -> &mut D {
        &mut self.data
    }

    pub fn into_data(self) -> D {
        crate::layouts::assert_dense(&self, "VecZnx::into_data");
        self.data
    }
}

impl<D: Data, W: ZnxWord> VecZnx<D, W> {
    /// Returns the scratch space (in bytes) required by right-shift operations.
    pub fn rsh_tmp_bytes(n: usize) -> usize {
        n * size_of::<W>()
    }
}

impl<D: HostDataMut, W: ZnxWord> ZnxZero for VecZnx<D, W> {
    fn zero(&mut self) {
        if self.is_dense() {
            self.raw_mut().fill(W::zero());
            return;
        }
        for j in 0..self.size() {
            for i in 0..self.cols() {
                self.at_mut(i, j).fill(W::zero());
            }
        }
    }
    fn zero_at(&mut self, i: usize, j: usize) {
        self.at_mut(i, j).fill(W::zero());
    }
}

impl<D: Data, W: ZnxWord> VecZnx<D, W> {
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

impl<D: Data, W: ZnxWord> VecZnx<D, W> {
    /// Constructs a `VecZnx` from raw parts without validation.
    pub fn from_data(data: D, n: usize, cols: usize, size: usize) -> Self {
        Self {
            data,
            shape: VecZnxShape::new(n, cols, size),
            _phantom: PhantomData,
        }
    }
}

impl<D: Data, W: ZnxWord> VecZnx<D, W> {
    /// Wraps `data` with an explicit shape, windowed or dense. No validation;
    /// element access is bounds-checked against the buffer.
    pub fn from_shape(data: D, shape: VecZnxShape) -> Self {
        Self {
            data,
            shape,
            _phantom: PhantomData,
        }
    }

    /// Re-tags this container with `shape`, keeping the buffer.
    pub fn with_shape(self, shape: VecZnxShape) -> Self {
        Self {
            data: self.data,
            shape,
            _phantom: PhantomData,
        }
    }

    /// View of coefficients `offset..offset + len` of every visible limb. See [`VecZnxShape::window_coeffs`].
    pub fn window_coeffs(self, offset: usize, len: usize) -> Self {
        let shape = self.shape.window_coeffs(offset, len);
        self.with_shape(shape)
    }

    /// View of visible limbs `offset, offset + step, ...` (`count` of them). See [`VecZnxShape::window_limbs`].
    pub fn window_limbs(self, offset: usize, step: usize, count: usize) -> Self {
        let shape = self.shape.window_limbs(offset, step, count);
        self.with_shape(shape)
    }
}

impl<D: HostDataRef, W: ZnxWord> fmt::Display for VecZnx<D, W> {
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

impl<D: HostDataMut, W: ZnxWord> FillUniform for VecZnx<D, W> {
    fn fill_uniform(&mut self, log_bound: usize, source: &mut Source) {
        crate::layouts::assert_dense(self, "VecZnx::fill_uniform");
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
        for x in self.raw_mut().iter_mut() {
            let r = source.next_u64() & mask;
            *x = W::from_i64(((r << shift) as i64) >> shift);
        }
    }
}

/// Owned `VecZnx` backed by a `Vec<u8>`.
pub type VecZnxOwned<W> = VecZnx<Vec<u8>, W>;
/// Mutably borrowed `VecZnx`.
pub type VecZnxMut<'a, W> = VecZnx<&'a mut [u8], W>;
/// Immutably borrowed `VecZnx`.
pub type VecZnxRef<'a, W> = VecZnx<&'a [u8], W>;
/// Shared backend-native borrow of a `VecZnx`.
pub type VecZnxBackendRef<'a, B> = VecZnx<<B as Backend>::BufRef<'a>, <B as Backend>::ZnxWord>;
/// Mutable backend-native borrow of a `VecZnx`.
pub type VecZnxBackendMut<'a, B> = VecZnx<<B as Backend>::BufMut<'a>, <B as Backend>::ZnxWord>;

/// Allocates a zero-initialized backend-owned `VecZnx`.
pub fn vec_znx_alloc_zeroed<B: Backend>(n: usize, cols: usize, size: usize) -> VecZnx<B::OwnedBuf, B::ZnxWord> {
    VecZnx {
        data: B::alloc_zeroed_bytes(B::bytes_of_vec_znx(n, cols, size)),
        shape: VecZnxShape::new(n, cols, size),
        _phantom: PhantomData,
    }
}

/// Returns a shared backend-native scalar view into a backend-owned `VecZnx`.
pub trait VecZnxAsScalarBackendRef<B: Backend> {
    fn as_scalar_znx_backend_ref(&self, col: usize, limb: usize) -> ScalarZnx<B::BufRef<'_>, B::ZnxWord>;
}

impl<B: Backend> VecZnxAsScalarBackendRef<B> for VecZnx<B::OwnedBuf, B::ZnxWord> {
    fn as_scalar_znx_backend_ref(&self, col: usize, limb: usize) -> ScalarZnx<B::BufRef<'_>, B::ZnxWord> {
        assert!(limb < self.size(), "size: {limb} >= {}", self.size());
        assert!(col < self.cols(), "cols: {col} >= {}", self.cols());
        let start: usize = self
            .shape
            .scalar_offset(col, limb)
            .checked_mul(B::size_of_znx_word())
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

impl<B: Backend> VecZnxAsScalarBackendMut<B> for VecZnx<B::OwnedBuf, B::ZnxWord> {
    fn as_scalar_znx_backend_mut(&mut self, col: usize, limb: usize) -> ScalarZnx<B::BufMut<'_>, B::ZnxWord> {
        let n = self.n();
        assert!(limb < self.size(), "size: {limb} >= {}", self.size());
        assert!(col < self.cols(), "cols: {col} >= {}", self.cols());
        let start: usize = self
            .shape
            .scalar_offset(col, limb)
            .checked_mul(B::size_of_znx_word())
            .expect("VecZnx scalar backend view offset overflows usize");
        let len: usize = n
            .checked_mul(B::size_of_znx_word())
            .expect("VecZnx scalar backend view length overflows usize");
        ScalarZnx::from_data(B::region_mut(&mut self.data, start, len), n, 1)
    }
}

/// Borrow a backend-owned `VecZnx` using the backend's native view type.
pub trait VecZnxToBackendRef<B: Backend = crate::layouts::HostBytesBackend> {
    fn to_backend_ref(&self) -> VecZnxBackendRef<'_, B>;
}

impl<B: Backend> VecZnxToBackendRef<B> for VecZnx<B::OwnedBuf, B::ZnxWord> {
    fn to_backend_ref(&self) -> VecZnxBackendRef<'_, B> {
        VecZnx {
            data: B::view(&self.data),
            shape: self.shape,
            _phantom: PhantomData,
        }
    }
}

impl<'b, B: Backend + 'b> VecZnxToBackendRef<B> for &VecZnx<B::BufRef<'b>, B::ZnxWord> {
    fn to_backend_ref(&self) -> VecZnxBackendRef<'_, B> {
        vec_znx_backend_ref_from_ref::<B>(self)
    }
}

impl VecZnxToBackendRef<crate::layouts::HostBytesBackend> for VecZnx<&mut [u8], i64> {
    fn to_backend_ref(&self) -> VecZnxBackendRef<'_, crate::layouts::HostBytesBackend> {
        VecZnx {
            data: self.data,
            shape: self.shape,
            _phantom: PhantomData,
        }
    }
}

impl VecZnxToBackendRef<crate::layouts::HostBytesBackend> for VecZnx<&[u8], i64> {
    fn to_backend_ref(&self) -> VecZnxBackendRef<'_, crate::layouts::HostBytesBackend> {
        VecZnx {
            data: self.data,
            shape: self.shape,
            _phantom: PhantomData,
        }
    }
}

/// Reborrow an already backend-borrowed `VecZnx` as a shared backend-native view.
pub trait VecZnxReborrowBackendRef<B: Backend = crate::layouts::HostBytesBackend> {
    fn reborrow_backend_ref(&self) -> VecZnxBackendRef<'_, B>;
}

pub fn vec_znx_backend_ref_from_ref<'a, 'b, B: Backend + 'b>(
    vec: &'a VecZnx<B::BufRef<'b>, B::ZnxWord>,
) -> VecZnxBackendRef<'a, B> {
    VecZnx {
        data: B::view_ref(&vec.data),
        shape: vec.shape,
        _phantom: PhantomData,
    }
}

pub fn vec_znx_backend_ref_from_mut<'a, 'b, B: Backend + 'b>(
    vec: &'a VecZnx<B::BufMut<'b>, B::ZnxWord>,
) -> VecZnxBackendRef<'a, B> {
    VecZnx {
        data: B::view_ref_mut(&vec.data),
        shape: vec.shape,
        _phantom: PhantomData,
    }
}

impl<'b, B: Backend + 'b> VecZnxReborrowBackendRef<B> for VecZnx<B::BufMut<'b>, B::ZnxWord> {
    fn reborrow_backend_ref(&self) -> VecZnxBackendRef<'_, B> {
        vec_znx_backend_ref_from_mut::<B>(self)
    }
}

/// Mutably borrow a backend-owned `VecZnx` using the backend's native view type.
pub trait VecZnxToBackendMut<B: Backend = crate::layouts::HostBytesBackend> {
    fn to_backend_mut(&mut self) -> VecZnxBackendMut<'_, B>;
}

impl<B: Backend> VecZnxToBackendMut<B> for VecZnx<B::OwnedBuf, B::ZnxWord> {
    fn to_backend_mut(&mut self) -> VecZnxBackendMut<'_, B> {
        VecZnx {
            data: B::view_mut(&mut self.data),
            shape: self.shape,
            _phantom: PhantomData,
        }
    }
}

impl<'b, B: Backend + 'b> VecZnxToBackendMut<B> for &mut VecZnx<B::BufMut<'b>, B::ZnxWord> {
    fn to_backend_mut(&mut self) -> VecZnxBackendMut<'_, B> {
        vec_znx_backend_mut_from_mut::<B>(self)
    }
}

impl VecZnxToBackendMut<crate::layouts::HostBytesBackend> for VecZnx<&mut [u8], i64> {
    fn to_backend_mut(&mut self) -> VecZnxBackendMut<'_, crate::layouts::HostBytesBackend> {
        VecZnx {
            data: self.data,
            shape: self.shape,
            _phantom: PhantomData,
        }
    }
}

/// Reborrow an already backend-borrowed `VecZnx` as a mutable backend-native view.
pub trait VecZnxReborrowBackendMut<B: Backend = crate::layouts::HostBytesBackend> {
    fn reborrow_backend_mut(&mut self) -> VecZnxBackendMut<'_, B>;
}

pub fn vec_znx_host_backend_ref<D: HostDataRef>(vec: &VecZnx<D, i64>) -> VecZnxBackendRef<'_, crate::layouts::HostBytesBackend> {
    VecZnx {
        data: vec.data.as_ref(),
        shape: vec.shape,
        _phantom: PhantomData,
    }
}

pub fn vec_znx_host_backend_mut<D: HostDataMut>(
    vec: &mut VecZnx<D, i64>,
) -> VecZnxBackendMut<'_, crate::layouts::HostBytesBackend> {
    VecZnx {
        data: vec.data.as_mut(),
        shape: vec.shape,
        _phantom: PhantomData,
    }
}

pub fn vec_znx_backend_mut_from_mut<'a, 'b, B: Backend + 'b>(
    vec: &'a mut VecZnx<B::BufMut<'b>, B::ZnxWord>,
) -> VecZnxBackendMut<'a, B> {
    VecZnx {
        data: B::view_mut_ref(&mut vec.data),
        shape: vec.shape,
        _phantom: PhantomData,
    }
}

impl<'b, B: Backend + 'b> VecZnxReborrowBackendMut<B> for VecZnx<B::BufMut<'b>, B::ZnxWord> {
    fn reborrow_backend_mut(&mut self) -> VecZnxBackendMut<'_, B> {
        vec_znx_backend_mut_from_mut::<B>(self)
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
/// Borrows a backend-owned `VecZnx` as the backend's native shared view.
///
/// [`VecZnxToBackendRef`] is keyed on `B::OwnedBuf`, a projection the compiler
/// cannot invert, so `vec.to_backend_ref()` cannot infer `B` the way the
/// backend-keyed containers (`VecZnxDft`, `VecZnxBig`) can. This names the
/// backend once, as a turbofish, instead of spelling the qualified path.
pub fn vec_znx_backend_ref<'a, B: Backend>(vec: &'a VecZnx<B::OwnedBuf, B::ZnxWord>) -> VecZnxBackendRef<'a, B> {
    <VecZnx<B::OwnedBuf, B::ZnxWord> as VecZnxToBackendRef<B>>::to_backend_ref(vec)
}

/// Reborrows an existing native mutable view for a shorter lifetime.
///
/// Same inference problem as [`vec_znx_backend_mut`]: the impl is keyed on
/// `B::BufMut`, so the backend is named once here instead of at each call.
pub fn vec_znx_reborrow_backend_mut<'a, B: Backend>(vec: &'a mut VecZnxBackendMut<'_, B>) -> VecZnxBackendMut<'a, B> {
    <VecZnx<B::BufMut<'_>, B::ZnxWord> as VecZnxReborrowBackendMut<B>>::reborrow_backend_mut(vec)
}

/// Borrows a backend-owned `VecZnx` as the backend's native mutable view.
///
/// See [`vec_znx_backend_ref`] for why this exists rather than a bare
/// `to_backend_mut()`.
pub fn vec_znx_backend_mut<'a, B: Backend>(vec: &'a mut VecZnx<B::OwnedBuf, B::ZnxWord>) -> VecZnxBackendMut<'a, B> {
    <VecZnx<B::OwnedBuf, B::ZnxWord> as VecZnxToBackendMut<B>>::to_backend_mut(vec)
}

/// # Panics
///
/// Panics if `size > vec.size()`.
pub fn vec_znx_backend_mut_with_size<'a, B: Backend>(vec: VecZnxBackendMut<'a, B>, size: usize) -> VecZnxBackendMut<'a, B> {
    VecZnx {
        data: vec.data,
        shape: vec.shape.with_size(size),
        _phantom: PhantomData,
    }
}

impl<D: HostDataMut, W: ZnxWord> ReaderFrom for VecZnx<D, W> {
    fn read_from<R: std::io::Read>(&mut self, reader: &mut R) -> std::io::Result<()> {
        crate::layouts::assert_dense(self, "VecZnx::read_from");
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

impl<D: HostDataRef, W: ZnxWord> WriterTo for VecZnx<D, W> {
    fn write_to<Wr: std::io::Write>(&self, writer: &mut Wr) -> std::io::Result<()> {
        crate::layouts::assert_dense(self, "VecZnx::write_to");
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

#[cfg(test)]
mod window_shape_tests {
    use super::VecZnxShape;

    #[test]
    fn dense_shape_matches_legacy_formula() {
        let s = VecZnxShape::new(16, 3, 4);
        assert!(s.is_dense());
        for limb in 0..4 {
            for col in 0..3 {
                assert_eq!(s.scalar_offset(col, limb), (limb * 3 + col) * 16);
            }
        }
    }

    #[test]
    fn coefficient_window_offsets_and_compose() {
        let s = VecZnxShape::new(16, 3, 4).window_coeffs(5, 8);
        assert!(!s.is_dense());
        assert_eq!(s.n(), 8);
        assert_eq!(s.n_full(), 16);
        assert_eq!(s.scalar_offset(1, 2), (2 * 3 + 1) * 16 + 5);
        let t = s.window_coeffs(2, 3);
        assert_eq!(t.n(), 3);
        assert_eq!(t.coeff_offset(), 7);
    }

    #[test]
    fn limb_window_offsets_and_compose() {
        let s = VecZnxShape::new(16, 3, 8).window_limbs(1, 2, 3);
        assert_eq!(s.size(), 3);
        assert_eq!(s.scalar_offset(0, 2), ((1 + 2 * 2) * 3) * 16);
        let t = s.window_limbs(1, 2, 1);
        assert_eq!(t.limb_offset(), 3);
        assert_eq!(t.limb_step(), 4);
        assert_eq!(t.size(), 1);
    }

    #[test]
    #[should_panic]
    fn coefficient_window_out_of_range_panics() {
        let _ = VecZnxShape::new(16, 1, 1).window_coeffs(10, 8);
    }

    #[test]
    #[should_panic]
    fn limb_window_out_of_range_panics() {
        let _ = VecZnxShape::new(16, 1, 4).window_limbs(2, 2, 2);
    }

    use crate::layouts::{VecZnx, ZnxView, ZnxViewMut, ZnxZero};

    fn ramp(n: usize, cols: usize, size: usize) -> VecZnx<Vec<u8>, i64> {
        let mut v = VecZnx::<Vec<u8>, i64>::alloc(n, cols, size);
        for j in 0..size {
            for i in 0..cols {
                for (k, x) in v.at_mut(i, j).iter_mut().enumerate() {
                    *x = (j * 1000 + i * 100 + k) as i64;
                }
            }
        }
        v
    }

    #[test]
    fn window_view_reads_the_expected_elements() {
        let v = ramp(8, 2, 4);
        let shape = v.shape().window_coeffs(3, 2).window_limbs(1, 2, 2);
        let w = VecZnx::<&[u8], i64>::from_shape(v.data().as_slice(), shape);
        assert_eq!(w.at(1, 0), &[1103, 1104]);
        assert_eq!(w.at(0, 1), &[3003, 3004]);
    }

    #[test]
    fn zero_on_window_touches_only_the_window() {
        let mut v = ramp(8, 2, 4);
        let before = v.clone();
        let shape = before.shape().window_coeffs(3, 2).window_limbs(1, 2, 2);
        {
            let mut w = VecZnx::<&mut [u8], i64>::from_shape(v.data_mut().as_mut_slice(), shape);
            w.zero();
        }
        for j in 0..4 {
            for i in 0..2 {
                for k in 0..8 {
                    let inside = (j == 1 || j == 3) && (3..5).contains(&k);
                    let expected = if inside { 0 } else { before.at(i, j)[k] };
                    assert_eq!(v.at(i, j)[k], expected, "col {i} limb {j} coeff {k}");
                }
            }
        }
    }

    #[test]
    #[should_panic(expected = "ZnxView::raw")]
    fn raw_on_window_panics() {
        let v = ramp(8, 1, 1);
        let w = VecZnx::<&[u8], i64>::from_shape(v.data().as_slice(), v.shape().window_coeffs(1, 2));
        let _ = w.raw();
    }
}
