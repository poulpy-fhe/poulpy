use std::{
    fmt,
    hash::{DefaultHasher, Hasher},
    marker::PhantomData,
};

use crate::{
    alloc_aligned,
    layouts::{
        Backend, Data, DataView, DataViewMut, DigestU64, FillUniform, HostDataMut, HostDataRef, ReaderFrom, ScalarZnx,
        ToOwnedDeep, WriterTo, ZnxInfos, ZnxView, ZnxViewMut, ZnxWord, ZnxZero,
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
/// contract) of the buffer.
///
/// **Invariant:** `size <= max_size`. The `max_size` field records the
/// allocated capacity; `size` can be reduced without reallocation.
#[repr(C)]
#[derive(PartialEq, Eq, Clone, Copy, Hash, Debug, Default)]
pub struct VecZnxShape {
    n: usize,
    cols: usize,
    size: usize,
    max_size: usize,
}

impl VecZnxShape {
    pub const fn new(n: usize, cols: usize, size: usize, max_size: usize) -> Self {
        assert!(size <= max_size);
        Self { n, cols, size, max_size }
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

    pub const fn max_size(self) -> usize {
        self.max_size
    }

    pub(crate) const fn with_size(self, size: usize) -> Self {
        assert!(size <= self.max_size);
        Self { size, ..self }
    }
}

#[repr(C)]
#[derive(PartialEq, Eq, Clone, Copy, Hash)]
pub struct VecZnx<D: Data, W: ZnxWord> {
    pub data: D,
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
        let mut h: DefaultHasher = DefaultHasher::new();
        h.write(self.data.as_ref());
        h.write_usize(self.n());
        h.write_usize(self.cols());
        h.write_usize(self.size());
        h.write_usize(self.max_size());
        h.finish()
    }
}

impl<D: HostDataRef, W: ZnxWord> ToOwnedDeep for VecZnx<D, W> {
    type Owned = VecZnx<Vec<u8>, W>;
    fn to_owned_deep(&self) -> Self::Owned {
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
        let shape = self.shape();
        VecZnx::from_data_with_max_size(
            crate::layouts::HostBytesBackend::from_bytes(BE::to_host_bytes(&self.data)),
            shape.n(),
            shape.cols(),
            shape.size(),
            shape.max_size(),
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
    fn cols(&self) -> usize {
        self.shape.cols()
    }

    fn rows(&self) -> usize {
        1
    }

    fn n(&self) -> usize {
        self.shape.n()
    }

    fn size(&self) -> usize {
        self.shape.size()
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

    /// Returns the allocated limb capacity.
    pub fn max_size(&self) -> usize {
        self.shape.max_size()
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
        self.raw_mut().fill(W::zero())
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
    /// Sets `max_size = size`.
    pub(crate) fn alloc(n: usize, cols: usize, size: usize) -> Self {
        let data: Vec<u8> = alloc_aligned::<u8>(Self::bytes_of(n, cols, size));
        Self {
            data,
            shape: VecZnxShape::new(n, cols, size, size),
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
            shape: VecZnxShape::new(n, cols, size, size),
            _phantom: PhantomData,
        }
    }
}

impl<D: Data, W: ZnxWord> VecZnx<D, W> {
    /// Constructs a `VecZnx` from raw parts without validation.
    /// Sets `max_size = size`.
    pub fn from_data(data: D, n: usize, cols: usize, size: usize) -> Self {
        Self {
            data,
            shape: VecZnxShape::new(n, cols, size, size),
            _phantom: PhantomData,
        }
    }

    /// Constructs a `VecZnx` from raw parts, preserving both `size` and `max_size`.
    ///
    /// Used by cross-backend transfer to rebuild a layout over a fresh
    /// buffer without shrinking its capacity.
    pub fn from_data_with_max_size(data: D, n: usize, cols: usize, size: usize, max_size: usize) -> Self {
        Self {
            data,
            shape: VecZnxShape::new(n, cols, size, max_size),
            _phantom: PhantomData,
        }
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

/// Returns a shared backend-native scalar view into a backend-owned `VecZnx`.
pub trait VecZnxAsScalarBackendRef<B: Backend> {
    fn as_scalar_znx_backend_ref(&self, col: usize, limb: usize) -> ScalarZnx<B::BufRef<'_>, B::ZnxWord>;
}

impl<B: Backend> VecZnxAsScalarBackendRef<B> for VecZnx<B::OwnedBuf, B::ZnxWord> {
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

impl<B: Backend> VecZnxAsScalarBackendMut<B> for VecZnx<B::OwnedBuf, B::ZnxWord> {
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

impl<D: HostDataMut, W: ZnxWord> ReaderFrom for VecZnx<D, W> {
    fn read_from<R: std::io::Read>(&mut self, reader: &mut R) -> std::io::Result<()> {
        // Read into temporaries first to avoid leaving self in an inconsistent state on error.
        let new_n: usize = reader.read_u64::<LittleEndian>()? as usize;
        let new_cols: usize = reader.read_u64::<LittleEndian>()? as usize;
        let new_size: usize = reader.read_u64::<LittleEndian>()? as usize;
        let new_max_size: usize = reader.read_u64::<LittleEndian>()? as usize;
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
        self.shape = VecZnxShape::new(new_n, new_cols, new_size, new_max_size);
        Ok(())
    }
}

impl<D: HostDataRef, W: ZnxWord> WriterTo for VecZnx<D, W> {
    fn write_to<Wr: std::io::Write>(&self, writer: &mut Wr) -> std::io::Result<()> {
        writer.write_u64::<LittleEndian>(self.n() as u64)?;
        writer.write_u64::<LittleEndian>(self.cols() as u64)?;
        writer.write_u64::<LittleEndian>(self.size() as u64)?;
        writer.write_u64::<LittleEndian>(self.max_size() as u64)?;
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
