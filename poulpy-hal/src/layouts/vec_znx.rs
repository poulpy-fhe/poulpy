use std::{
    fmt,
    hash::{DefaultHasher, Hasher},
};

use crate::{
    alloc_aligned,
    layouts::{
        Backend, Data, DataView, DataViewMut, DigestU64, FillUniform, HostDataMut, HostDataRef, ReaderFrom, ScalarZnx,
        ToOwnedDeep, WriterTo, ZnxInfos, ZnxView, ZnxViewMut, ZnxZero,
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
/// limbs of `N` coefficients. Coefficients are `i64` values.
///
/// **Memory layout:** limb-major, column-minor. Limb `j` of column `i`
/// starts at scalar offset `N * (j * cols + i)`.
///
/// The type parameter `D` controls ownership: `Vec<u8>` for owned,
/// `&[u8]` for shared borrows, `&mut [u8]` for mutable borrows.
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

    pub const fn with_size(self, size: usize) -> Self {
        assert!(size <= self.max_size);
        Self { size, ..self }
    }
}

#[repr(C)]
#[derive(PartialEq, Eq, Clone, Copy, Hash)]
pub struct VecZnx<D: Data> {
    pub data: D,
    shape: VecZnxShape,
}

impl<D: HostDataRef> VecZnx<D> {
    /// Returns a read-only [`ScalarZnx`] view of a single limb of a single column.
    pub fn as_scalar_znx_ref(&self, col: usize, limb: usize) -> ScalarZnx<&[u8]> {
        ScalarZnx::from_data(bytemuck::cast_slice(self.at(col, limb)), self.n(), 1)
    }
}

impl<D: HostDataMut> VecZnx<D> {
    /// Returns a mutable [`ScalarZnx`] view of a single limb of a single column.
    pub fn as_scalar_znx_mut(&mut self, col: usize, limb: usize) -> ScalarZnx<&mut [u8]> {
        let n = self.n();
        ScalarZnx::from_data(bytemuck::cast_slice_mut(self.at_mut(col, limb)), n, 1)
    }
}

impl<D: Data + Default> Default for VecZnx<D> {
    fn default() -> Self {
        Self {
            data: D::default(),
            shape: VecZnxShape::default(),
        }
    }
}

impl<D: HostDataRef> DigestU64 for VecZnx<D> {
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

impl<D: HostDataRef> ToOwnedDeep for VecZnx<D> {
    type Owned = VecZnx<Vec<u8>>;
    fn to_owned_deep(&self) -> Self::Owned {
        VecZnx {
            data: self.data.as_ref().to_vec(),
            shape: self.shape,
        }
    }
}

impl<D: Data> VecZnx<D> {
    /// Rebuilds this backend-owned vector as a host-owned [`VecZnx<Vec<u8>>`].
    pub fn to_host_owned<BE>(&self) -> VecZnx<Vec<u8>>
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

impl<D: HostDataRef> fmt::Debug for VecZnx<D> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{self}")
    }
}

impl<D: Data> ZnxInfos for VecZnx<D> {
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

impl<D: Data> DataView for VecZnx<D> {
    type D = D;
    fn data(&self) -> &Self::D {
        &self.data
    }
}

impl<D: Data> DataViewMut for VecZnx<D> {
    fn data_mut(&mut self) -> &mut Self::D {
        &mut self.data
    }
}

impl<D: HostDataRef> ZnxView for VecZnx<D> {
    type Scalar = i64;
}

impl<D: Data> VecZnx<D> {
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

    pub fn with_size(mut self, size: usize) -> Self {
        assert!(size <= self.max_size());
        self.shape = self.shape.with_size(size);
        self
    }

    /// Returns the allocated limb capacity.
    pub fn max_size(&self) -> usize {
        self.shape.max_size()
    }
}

impl<D: Data> VecZnx<D> {
    /// Sets the active limb count.
    ///
    /// # Panics
    ///
    /// Panics if `size > max_size`.
    pub fn set_size(&mut self, size: usize) {
        self.shape = self.shape.with_size(size);
    }
}

impl VecZnx<Vec<u8>> {
    /// Returns the scratch space (in bytes) required by right-shift operations.
    pub fn rsh_tmp_bytes(n: usize) -> usize {
        n * size_of::<i64>()
    }

    /// Reallocates the backing buffer so capacity matches the `new_size` limb count.
    pub fn reallocate_limbs(&mut self, new_size: usize) {
        if self.size() == new_size {
            return;
        }

        let mut compact: Self = Self::alloc(self.n(), self.cols(), new_size);
        let copy_len = compact.raw().len().min(self.raw().len());
        compact.raw_mut()[..copy_len].copy_from_slice(&self.raw()[..copy_len]);
        *self = compact;
    }
}

impl<D: HostDataMut> ZnxZero for VecZnx<D> {
    fn zero(&mut self) {
        self.raw_mut().fill(0)
    }
    fn zero_at(&mut self, i: usize, j: usize) {
        self.at_mut(i, j).fill(0);
    }
}

impl VecZnx<Vec<u8>> {
    /// Returns the number of bytes required: `n * cols * size * 8`.
    pub fn bytes_of(n: usize, cols: usize, size: usize) -> usize {
        n * cols * size * size_of::<i64>()
    }

    /// Allocates a zero-initialized `VecZnx` aligned to [`DEFAULTALIGN`](crate::DEFAULTALIGN).
    /// Sets `max_size = size`.
    pub(crate) fn alloc(n: usize, cols: usize, size: usize) -> Self {
        let data: Vec<u8> = alloc_aligned::<u8>(Self::bytes_of(n, cols, size));
        Self {
            data,
            shape: VecZnxShape::new(n, cols, size, size),
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
        }
    }
}

impl<D: Data> VecZnx<D> {
    /// Constructs a `VecZnx` from raw parts without validation.
    /// Sets `max_size = size`.
    pub fn from_data(data: D, n: usize, cols: usize, size: usize) -> Self {
        Self {
            data,
            shape: VecZnxShape::new(n, cols, size, size),
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
        }
    }
}

impl<D: HostDataRef> fmt::Display for VecZnx<D> {
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

impl<D: HostDataMut> FillUniform for VecZnx<D> {
    fn fill_uniform(&mut self, log_bound: usize, source: &mut Source) {
        match log_bound {
            64 => source.fill_bytes(self.data.as_mut()),
            0 => panic!("invalid log_bound, cannot be zero"),
            _ => {
                let mask: u64 = (1u64 << log_bound) - 1;
                for x in self.raw_mut().iter_mut() {
                    let r = source.next_u64() & mask;
                    *x = ((r << (64 - log_bound)) as i64) >> (64 - log_bound);
                }
            }
        }
    }
}

/// Owned `VecZnx` backed by a `Vec<u8>`.
pub type VecZnxOwned = VecZnx<Vec<u8>>;
/// Mutably borrowed `VecZnx`.
pub type VecZnxMut<'a> = VecZnx<&'a mut [u8]>;
/// Immutably borrowed `VecZnx`.
pub type VecZnxRef<'a> = VecZnx<&'a [u8]>;
/// Shared backend-native borrow of a `VecZnx`.
pub type VecZnxBackendRef<'a, B> = VecZnx<<B as Backend>::BufRef<'a>>;
/// Mutable backend-native borrow of a `VecZnx`.
pub type VecZnxBackendMut<'a, B> = VecZnx<<B as Backend>::BufMut<'a>>;

/// Returns a shared backend-native scalar view into a backend-owned `VecZnx`.
pub trait VecZnxAsScalarBackendRef<B: Backend> {
    fn as_scalar_znx_backend_ref(&self, col: usize, limb: usize) -> ScalarZnx<B::BufRef<'_>>;
}

impl<B: Backend> VecZnxAsScalarBackendRef<B> for VecZnx<B::OwnedBuf> {
    fn as_scalar_znx_backend_ref(&self, col: usize, limb: usize) -> ScalarZnx<B::BufRef<'_>> {
        #[cfg(debug_assertions)]
        {
            assert!(limb < self.size(), "size: {limb} >= {}", self.size());
            assert!(col < self.cols(), "cols: {col} >= {}", self.cols());
        }
        let start: usize = (limb * self.cols() + col) * self.n() * size_of::<i64>();
        let len: usize = self.n() * size_of::<i64>();
        ScalarZnx::from_data(B::region(&self.data, start, len), self.n(), 1)
    }
}

/// Returns a mutable backend-native scalar view into a backend-owned `VecZnx`.
pub trait VecZnxAsScalarBackendMut<B: Backend> {
    fn as_scalar_znx_backend_mut(&mut self, col: usize, limb: usize) -> ScalarZnx<B::BufMut<'_>>;
}

impl<B: Backend> VecZnxAsScalarBackendMut<B> for VecZnx<B::OwnedBuf> {
    fn as_scalar_znx_backend_mut(&mut self, col: usize, limb: usize) -> ScalarZnx<B::BufMut<'_>> {
        #[cfg(debug_assertions)]
        {
            assert!(limb < self.size(), "size: {limb} >= {}", self.size());
            assert!(col < self.cols(), "cols: {col} >= {}", self.cols());
        }
        let n = self.n();
        let start: usize = (limb * self.cols() + col) * n * size_of::<i64>();
        let len: usize = n * size_of::<i64>();
        ScalarZnx::from_data(B::region_mut(&mut self.data, start, len), n, 1)
    }
}

/// Borrow a backend-owned `VecZnx` using the backend's native view type.
pub trait VecZnxToBackendRef<B: Backend = crate::layouts::HostBytesBackend> {
    fn to_backend_ref(&self) -> VecZnxBackendRef<'_, B>;
}

impl<B: Backend> VecZnxToBackendRef<B> for VecZnx<B::OwnedBuf> {
    fn to_backend_ref(&self) -> VecZnxBackendRef<'_, B> {
        VecZnx {
            data: B::view(&self.data),
            shape: self.shape,
        }
    }
}

impl<'b, B: Backend + 'b> VecZnxToBackendRef<B> for &VecZnx<B::BufRef<'b>> {
    fn to_backend_ref(&self) -> VecZnxBackendRef<'_, B> {
        vec_znx_backend_ref_from_ref::<B>(self)
    }
}

impl VecZnxToBackendRef<crate::layouts::HostBytesBackend> for VecZnx<&mut [u8]> {
    fn to_backend_ref(&self) -> VecZnxBackendRef<'_, crate::layouts::HostBytesBackend> {
        VecZnx {
            data: self.data,
            shape: self.shape,
        }
    }
}

impl VecZnxToBackendRef<crate::layouts::HostBytesBackend> for VecZnx<&[u8]> {
    fn to_backend_ref(&self) -> VecZnxBackendRef<'_, crate::layouts::HostBytesBackend> {
        VecZnx {
            data: self.data,
            shape: self.shape,
        }
    }
}

/// Reborrow an already backend-borrowed `VecZnx` as a shared backend-native view.
pub trait VecZnxReborrowBackendRef<B: Backend = crate::layouts::HostBytesBackend> {
    fn reborrow_backend_ref(&self) -> VecZnxBackendRef<'_, B>;
}

pub fn vec_znx_backend_ref_from_ref<'a, 'b, B: Backend + 'b>(vec: &'a VecZnx<B::BufRef<'b>>) -> VecZnxBackendRef<'a, B> {
    VecZnx {
        data: B::view_ref(&vec.data),
        shape: vec.shape,
    }
}

pub fn vec_znx_backend_ref_from_mut<'a, 'b, B: Backend + 'b>(vec: &'a VecZnx<B::BufMut<'b>>) -> VecZnxBackendRef<'a, B> {
    VecZnx {
        data: B::view_ref_mut(&vec.data),
        shape: vec.shape,
    }
}

impl<'b, B: Backend + 'b> VecZnxReborrowBackendRef<B> for VecZnx<B::BufMut<'b>> {
    fn reborrow_backend_ref(&self) -> VecZnxBackendRef<'_, B> {
        vec_znx_backend_ref_from_mut::<B>(self)
    }
}

/// Mutably borrow a backend-owned `VecZnx` using the backend's native view type.
pub trait VecZnxToBackendMut<B: Backend = crate::layouts::HostBytesBackend> {
    fn to_backend_mut(&mut self) -> VecZnxBackendMut<'_, B>;
}

impl<B: Backend> VecZnxToBackendMut<B> for VecZnx<B::OwnedBuf> {
    fn to_backend_mut(&mut self) -> VecZnxBackendMut<'_, B> {
        VecZnx {
            data: B::view_mut(&mut self.data),
            shape: self.shape,
        }
    }
}

impl<'b, B: Backend + 'b> VecZnxToBackendMut<B> for &mut VecZnx<B::BufMut<'b>> {
    fn to_backend_mut(&mut self) -> VecZnxBackendMut<'_, B> {
        vec_znx_backend_mut_from_mut::<B>(self)
    }
}

impl VecZnxToBackendMut<crate::layouts::HostBytesBackend> for VecZnx<&mut [u8]> {
    fn to_backend_mut(&mut self) -> VecZnxBackendMut<'_, crate::layouts::HostBytesBackend> {
        VecZnx {
            data: self.data,
            shape: self.shape,
        }
    }
}

/// Reborrow an already backend-borrowed `VecZnx` as a mutable backend-native view.
pub trait VecZnxReborrowBackendMut<B: Backend = crate::layouts::HostBytesBackend> {
    fn reborrow_backend_mut(&mut self) -> VecZnxBackendMut<'_, B>;
}

pub fn vec_znx_host_backend_ref<D: HostDataRef>(vec: &VecZnx<D>) -> VecZnxBackendRef<'_, crate::layouts::HostBytesBackend> {
    VecZnx {
        data: vec.data.as_ref(),
        shape: vec.shape,
    }
}

pub fn vec_znx_host_backend_mut<D: HostDataMut>(vec: &mut VecZnx<D>) -> VecZnxBackendMut<'_, crate::layouts::HostBytesBackend> {
    VecZnx {
        data: vec.data.as_mut(),
        shape: vec.shape,
    }
}

pub fn vec_znx_backend_mut_from_mut<'a, 'b, B: Backend + 'b>(vec: &'a mut VecZnx<B::BufMut<'b>>) -> VecZnxBackendMut<'a, B> {
    VecZnx {
        data: B::view_mut_ref(&mut vec.data),
        shape: vec.shape,
    }
}

impl<'b, B: Backend + 'b> VecZnxReborrowBackendMut<B> for VecZnx<B::BufMut<'b>> {
    fn reborrow_backend_mut(&mut self) -> VecZnxBackendMut<'_, B> {
        vec_znx_backend_mut_from_mut::<B>(self)
    }
}

impl<D: HostDataMut> ReaderFrom for VecZnx<D> {
    fn read_from<R: std::io::Read>(&mut self, reader: &mut R) -> std::io::Result<()> {
        // Read into temporaries first to avoid leaving self in an inconsistent state on error.
        let new_n: usize = reader.read_u64::<LittleEndian>()? as usize;
        let new_cols: usize = reader.read_u64::<LittleEndian>()? as usize;
        let new_size: usize = reader.read_u64::<LittleEndian>()? as usize;
        let new_max_size: usize = reader.read_u64::<LittleEndian>()? as usize;
        let len: usize = reader.read_u64::<LittleEndian>()? as usize;

        // Validate metadata consistency: n * cols * size * sizeof(i64) must match data length.
        let expected_len: usize = new_n * new_cols * new_size * size_of::<i64>();
        if expected_len != len {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "VecZnx metadata inconsistent: n={new_n} * cols={new_cols} * size={new_size} * 8 = {expected_len} != data len={len}"
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

impl<D: HostDataRef> WriterTo for VecZnx<D> {
    fn write_to<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<()> {
        writer.write_u64::<LittleEndian>(self.n() as u64)?;
        writer.write_u64::<LittleEndian>(self.cols() as u64)?;
        writer.write_u64::<LittleEndian>(self.size() as u64)?;
        writer.write_u64::<LittleEndian>(self.max_size() as u64)?;
        let coeff_bytes: usize = self.n() * self.cols() * self.size() * size_of::<i64>();
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
