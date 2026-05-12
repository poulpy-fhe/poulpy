use crate::{
    alloc_aligned,
    layouts::{
        Backend, Data, DataView, DataViewMut, DigestU64, FillUniform, HostDataMut, HostDataRef, ReaderFrom, ToOwnedDeep, VecZnx,
        WriterTo, ZnxInfos, ZnxView, ZnxViewMut, ZnxZero,
    },
    source::Source,
};
use std::{
    fmt,
    hash::{DefaultHasher, Hasher},
};

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use rand::Rng;

#[repr(C)]
#[derive(PartialEq, Eq, Clone, Copy, Hash, Debug, Default)]
pub struct MatZnxShape {
    n: usize,
    size: usize,
    rows: usize,
    cols_in: usize,
    cols_out: usize,
}

impl MatZnxShape {
    pub const fn new(n: usize, rows: usize, cols_in: usize, cols_out: usize, size: usize) -> Self {
        Self {
            n,
            size,
            rows,
            cols_in,
            cols_out,
        }
    }

    pub const fn n(self) -> usize {
        self.n
    }

    pub const fn size(self) -> usize {
        self.size
    }

    pub const fn rows(self) -> usize {
        self.rows
    }

    pub const fn cols_in(self) -> usize {
        self.cols_in
    }

    pub const fn cols_out(self) -> usize {
        self.cols_out
    }
}

/// Matrix of polynomials in `Z[X]/(X^N + 1)`.
///
/// A `MatZnx` has `rows` rows, each containing `cols_in` entries.
/// Each entry is itself a [`VecZnx`] with `cols_out` columns and `size` limbs.
/// This gives a total of `rows * cols_in * cols_out * size` small polynomials.
///
/// Used primarily as the plaintext input to [`VmpPrepare`](crate::api::VmpPrepare),
/// which converts a `MatZnx` into a prepared [`VmpPMat`](crate::layouts::VmpPMat)
/// for vector-matrix products.
#[repr(C)]
#[derive(PartialEq, Eq, Clone, Hash)]
pub struct MatZnx<D: Data> {
    data: D,
    shape: MatZnxShape,
}

impl<D: HostDataRef> DigestU64 for MatZnx<D> {
    fn digest_u64(&self) -> u64 {
        let mut h: DefaultHasher = DefaultHasher::new();
        h.write(self.data.as_ref());
        h.write_usize(self.n());
        h.write_usize(self.size());
        h.write_usize(self.rows());
        h.write_usize(self.cols_in());
        h.write_usize(self.cols_out());
        h.finish()
    }
}

impl<D: HostDataRef> ToOwnedDeep for MatZnx<D> {
    type Owned = MatZnx<Vec<u8>>;
    fn to_owned_deep(&self) -> Self::Owned {
        MatZnx {
            data: self.data.as_ref().to_vec(),
            shape: self.shape,
        }
    }
}

impl<D: HostDataRef> fmt::Debug for MatZnx<D> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{self}")
    }
}

impl<D: Data> ZnxInfos for MatZnx<D> {
    fn cols(&self) -> usize {
        self.shape.cols_in()
    }

    fn rows(&self) -> usize {
        self.shape.rows()
    }

    fn n(&self) -> usize {
        self.shape.n()
    }

    fn size(&self) -> usize {
        self.shape.size()
    }

    fn poly_count(&self) -> usize {
        self.rows() * self.cols_in() * self.cols_out() * self.size()
    }
}

impl<D: Data> DataView for MatZnx<D> {
    type D = D;
    fn data(&self) -> &Self::D {
        &self.data
    }
}

impl<D: Data> DataViewMut for MatZnx<D> {
    fn data_mut(&mut self) -> &mut Self::D {
        &mut self.data
    }
}

impl<D: HostDataRef> ZnxView for MatZnx<D> {
    type Scalar = i64;
}

impl<D: Data> MatZnx<D> {
    pub fn shape(&self) -> MatZnxShape {
        self.shape
    }

    pub fn n(&self) -> usize {
        self.shape.n()
    }

    pub fn rows(&self) -> usize {
        self.shape.rows()
    }

    pub fn size(&self) -> usize {
        self.shape.size()
    }

    /// Returns the number of input columns (first matrix dimension after rows).
    pub fn cols_in(&self) -> usize {
        self.shape.cols_in()
    }

    /// Returns the number of output columns (the column count of each inner [`VecZnx`]).
    pub fn cols_out(&self) -> usize {
        self.shape.cols_out()
    }

    /// Consumes the `MatZnx` and returns its backing data.
    pub fn into_data(self) -> D {
        self.data
    }
}

impl MatZnx<Vec<u8>> {
    /// Returns the number of bytes required to store the matrix.
    pub fn bytes_of(n: usize, rows: usize, cols_in: usize, cols_out: usize, size: usize) -> usize {
        rows * cols_in * VecZnx::<Vec<u8>>::bytes_of(n, cols_out, size)
    }

    /// Allocates a zero-initialized `MatZnx` aligned to [`DEFAULTALIGN`](crate::DEFAULTALIGN).
    pub(crate) fn alloc(n: usize, rows: usize, cols_in: usize, cols_out: usize, size: usize) -> Self {
        let data: Vec<u8> = alloc_aligned(Self::bytes_of(n, rows, cols_in, cols_out, size));
        Self {
            data,
            shape: MatZnxShape::new(n, rows, cols_in, cols_out, size),
        }
    }

    pub fn from_bytes(n: usize, rows: usize, cols_in: usize, cols_out: usize, size: usize, bytes: impl Into<Vec<u8>>) -> Self {
        let data: Vec<u8> = bytes.into();
        assert!(data.len() == Self::bytes_of(n, rows, cols_in, cols_out, size));
        crate::assert_alignment(data.as_ptr());
        Self {
            data,
            shape: MatZnxShape::new(n, rows, cols_in, cols_out, size),
        }
    }
}

impl<D: HostDataRef> MatZnx<D> {
    /// Returns a shared [`VecZnx`] view of the entry at `(row, col)`.
    ///
    /// # Panics (debug)
    ///
    /// Debug-asserts that `row < rows` and `col < cols_in`.
    pub fn at(&self, row: usize, col: usize) -> VecZnx<&[u8]> {
        #[cfg(debug_assertions)]
        {
            assert!(row < self.rows(), "rows: {} >= {}", row, self.rows());
            assert!(col < self.cols_in(), "cols: {} >= {}", col, self.cols_in());
        }

        let self_ref = MatZnx {
            data: self.data.as_ref(),
            shape: self.shape,
        };
        let nb_bytes: usize = VecZnx::<Vec<u8>>::bytes_of(self.n(), self.cols_out(), self.size());
        let start: usize = nb_bytes * self.cols() * row + col * nb_bytes;
        let end: usize = start + nb_bytes;

        VecZnx::from_data(&self_ref.data[start..end], self.n(), self.cols_out(), self.size())
    }
}

impl<D: HostDataMut> MatZnx<D> {
    /// Returns a mutable [`VecZnx`] view of the entry at `(row, col)`.
    ///
    /// # Panics (debug)
    ///
    /// Debug-asserts that `row < rows` and `col < cols_in`.
    pub fn at_mut(&mut self, row: usize, col: usize) -> VecZnx<&mut [u8]> {
        #[cfg(debug_assertions)]
        {
            assert!(row < self.rows(), "rows: {} >= {}", row, self.rows());
            assert!(col < self.cols_in(), "cols: {} >= {}", col, self.cols_in());
        }

        let n: usize = self.n();
        let rows: usize = self.rows();
        let cols_out: usize = self.cols_out();
        let cols_in: usize = self.cols_in();
        let size: usize = self.size();

        let self_ref = MatZnx {
            data: self.data.as_mut(),
            shape: MatZnxShape::new(n, rows, cols_in, cols_out, size),
        };
        let nb_bytes: usize = VecZnx::<Vec<u8>>::bytes_of(n, cols_out, size);
        let start: usize = nb_bytes * cols_in * row + col * nb_bytes;
        let end: usize = start + nb_bytes;

        VecZnx::from_data(&mut self_ref.data[start..end], n, cols_out, size)
    }
}

/// Returns a shared backend-native entry view of a backend-owned `MatZnx`.
pub trait MatZnxAtBackendRef<B: Backend> {
    fn at_backend(&self, row: usize, col: usize) -> VecZnx<B::BufRef<'_>>;
}

impl<B: Backend> MatZnxAtBackendRef<B> for MatZnx<B::OwnedBuf> {
    fn at_backend(&self, row: usize, col: usize) -> VecZnx<B::BufRef<'_>> {
        #[cfg(debug_assertions)]
        {
            assert!(row < self.rows(), "rows: {} >= {}", row, self.rows());
            assert!(col < self.cols_in(), "cols: {} >= {}", col, self.cols_in());
        }

        let nb_bytes: usize = VecZnx::<Vec<u8>>::bytes_of(self.n(), self.cols_out(), self.size());
        let start: usize = nb_bytes * self.cols() * row + col * nb_bytes;
        let end: usize = start + nb_bytes;

        VecZnx::from_data(
            B::region(&self.data, start, end - start),
            self.n(),
            self.cols_out(),
            self.size(),
        )
    }
}

pub fn mat_znx_at_backend_ref_from_ref<'a, 'b, B: Backend + 'b>(
    mat: &'a MatZnx<B::BufRef<'b>>,
    row: usize,
    col: usize,
) -> VecZnx<B::BufRef<'a>> {
    #[cfg(debug_assertions)]
    {
        assert!(row < mat.rows(), "rows: {} >= {}", row, mat.rows());
        assert!(col < mat.cols_in(), "cols: {} >= {}", col, mat.cols_in());
    }

    let nb_bytes: usize = VecZnx::<Vec<u8>>::bytes_of(mat.n(), mat.cols_out(), mat.size());
    let start: usize = nb_bytes * mat.cols() * row + col * nb_bytes;
    let end: usize = start + nb_bytes;

    VecZnx::from_data(
        B::region_ref(&mat.data, start, end - start),
        mat.n(),
        mat.cols_out(),
        mat.size(),
    )
}

pub fn mat_znx_at_backend_ref_from_mut<'a, 'b, B: Backend + 'b>(
    mat: &'a MatZnx<B::BufMut<'b>>,
    row: usize,
    col: usize,
) -> VecZnx<B::BufRef<'a>> {
    #[cfg(debug_assertions)]
    {
        assert!(row < mat.rows(), "rows: {} >= {}", row, mat.rows());
        assert!(col < mat.cols_in(), "cols: {} >= {}", col, mat.cols_in());
    }

    let nb_bytes: usize = VecZnx::<Vec<u8>>::bytes_of(mat.n(), mat.cols_out(), mat.size());
    let start: usize = nb_bytes * mat.cols() * row + col * nb_bytes;
    let end: usize = start + nb_bytes;

    VecZnx::from_data(
        B::region_ref_mut(&mat.data, start, end - start),
        mat.n(),
        mat.cols_out(),
        mat.size(),
    )
}

/// Returns a mutable backend-native entry view of a backend-owned `MatZnx`.
pub trait MatZnxAtBackendMut<B: Backend> {
    fn at_backend_mut(&mut self, row: usize, col: usize) -> VecZnx<B::BufMut<'_>>;
}

impl<B: Backend> MatZnxAtBackendMut<B> for MatZnx<B::OwnedBuf> {
    fn at_backend_mut(&mut self, row: usize, col: usize) -> VecZnx<B::BufMut<'_>> {
        #[cfg(debug_assertions)]
        {
            assert!(row < self.rows(), "rows: {} >= {}", row, self.rows());
            assert!(col < self.cols_in(), "cols: {} >= {}", col, self.cols_in());
        }

        let n: usize = self.n();
        let cols_out: usize = self.cols_out();
        let cols_in: usize = self.cols_in();
        let size: usize = self.size();
        let nb_bytes: usize = VecZnx::<Vec<u8>>::bytes_of(n, cols_out, size);
        let start: usize = nb_bytes * cols_in * row + col * nb_bytes;
        let end: usize = start + nb_bytes;

        VecZnx::from_data(B::region_mut(&mut self.data, start, end - start), n, cols_out, size)
    }
}

pub fn mat_znx_at_backend_mut_from_mut<'a, 'b, B: Backend + 'b>(
    mat: &'a mut MatZnx<B::BufMut<'b>>,
    row: usize,
    col: usize,
) -> VecZnx<B::BufMut<'a>> {
    #[cfg(debug_assertions)]
    {
        assert!(row < mat.rows(), "rows: {} >= {}", row, mat.rows());
        assert!(col < mat.cols_in(), "cols: {} >= {}", col, mat.cols_in());
    }

    let n: usize = mat.n();
    let cols_out: usize = mat.cols_out();
    let cols_in: usize = mat.cols_in();
    let size: usize = mat.size();
    let nb_bytes: usize = VecZnx::<Vec<u8>>::bytes_of(n, cols_out, size);
    let start: usize = nb_bytes * cols_in * row + col * nb_bytes;
    let end: usize = start + nb_bytes;

    VecZnx::from_data(B::region_mut_ref(&mut mat.data, start, end - start), n, cols_out, size)
}

impl<D: HostDataMut> FillUniform for MatZnx<D> {
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

/// Owned `MatZnx` backed by a `Vec<u8>`.
pub type MatZnxOwned = MatZnx<Vec<u8>>;
/// Mutably borrowed `MatZnx`.
pub type MatZnxMut<'a> = MatZnx<&'a mut [u8]>;
/// Immutably borrowed `MatZnx`.
pub type MatZnxRef<'a> = MatZnx<&'a [u8]>;
/// Shared backend-native borrow of a `MatZnx`.
pub type MatZnxBackendRef<'a, B> = MatZnx<<B as Backend>::BufRef<'a>>;
/// Mutable backend-native borrow of a `MatZnx`.
pub type MatZnxBackendMut<'a, B> = MatZnx<<B as Backend>::BufMut<'a>>;

/// Borrow a backend-owned `MatZnx` using the backend's native view type.
pub trait MatZnxToBackendRef<B: Backend> {
    fn to_backend_ref(&self) -> MatZnxBackendRef<'_, B>;
}

impl<B: Backend> MatZnxToBackendRef<B> for MatZnx<B::OwnedBuf> {
    fn to_backend_ref(&self) -> MatZnxBackendRef<'_, B> {
        MatZnx {
            data: B::view(&self.data),
            shape: self.shape,
        }
    }
}

impl<'b, B: Backend + 'b> MatZnxToBackendRef<B> for &MatZnx<B::BufRef<'b>> {
    fn to_backend_ref(&self) -> MatZnxBackendRef<'_, B> {
        mat_znx_backend_ref_from_ref::<B>(self)
    }
}

impl<'b, B: Backend + 'b> MatZnxToBackendRef<B> for &mut MatZnx<B::BufMut<'b>> {
    fn to_backend_ref(&self) -> MatZnxBackendRef<'_, B> {
        mat_znx_backend_ref_from_mut::<B>(self)
    }
}

pub fn mat_znx_backend_ref_from_ref<'a, 'b, B: Backend + 'b>(mat: &'a MatZnx<B::BufRef<'b>>) -> MatZnxBackendRef<'a, B> {
    MatZnx {
        data: B::view_ref(&mat.data),
        shape: mat.shape,
    }
}

pub fn mat_znx_backend_ref_from_mut<'a, 'b, B: Backend + 'b>(mat: &'a MatZnx<B::BufMut<'b>>) -> MatZnxBackendRef<'a, B> {
    MatZnx {
        data: B::view_ref_mut(&mat.data),
        shape: mat.shape,
    }
}

/// Mutably borrow a backend-owned `MatZnx` using the backend's native view type.
pub trait MatZnxToBackendMut<B: Backend> {
    fn to_backend_mut(&mut self) -> MatZnxBackendMut<'_, B>;
}

impl<B: Backend> MatZnxToBackendMut<B> for MatZnx<B::OwnedBuf> {
    fn to_backend_mut(&mut self) -> MatZnxBackendMut<'_, B> {
        MatZnx {
            data: B::view_mut(&mut self.data),
            shape: self.shape,
        }
    }
}

impl<'b, B: Backend + 'b> MatZnxToBackendMut<B> for &mut MatZnx<B::BufMut<'b>> {
    fn to_backend_mut(&mut self) -> MatZnxBackendMut<'_, B> {
        mat_znx_backend_mut_from_mut::<B>(self)
    }
}

pub fn mat_znx_backend_mut_from_mut<'a, 'b, B: Backend + 'b>(mat: &'a mut MatZnx<B::BufMut<'b>>) -> MatZnxBackendMut<'a, B> {
    MatZnx {
        data: B::view_mut_ref(&mut mat.data),
        shape: mat.shape,
    }
}

impl<D: Data> MatZnx<D> {
    pub fn from_data(data: D, n: usize, rows: usize, cols_in: usize, cols_out: usize, size: usize) -> Self {
        Self {
            data,
            shape: MatZnxShape::new(n, rows, cols_in, cols_out, size),
        }
    }
}

impl<D: HostDataMut> ReaderFrom for MatZnx<D> {
    fn read_from<R: std::io::Read>(&mut self, reader: &mut R) -> std::io::Result<()> {
        let new_n: usize = reader.read_u64::<LittleEndian>()? as usize;
        let new_size: usize = reader.read_u64::<LittleEndian>()? as usize;
        let new_rows: usize = reader.read_u64::<LittleEndian>()? as usize;
        let new_cols_in: usize = reader.read_u64::<LittleEndian>()? as usize;
        let new_cols_out: usize = reader.read_u64::<LittleEndian>()? as usize;
        let len: usize = reader.read_u64::<LittleEndian>()? as usize;

        let expected_len: usize = new_rows * new_cols_in * new_n * new_cols_out * new_size * size_of::<i64>();
        if expected_len != len {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "MatZnx metadata inconsistent: rows={new_rows} * cols_in={new_cols_in} * n={new_n} * cols_out={new_cols_out} * size={new_size} * 8 = {expected_len} != data len={len}"
                ),
            ));
        }

        let buf: &mut [u8] = self.data.as_mut();
        if buf.len() < len {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("MatZnx buffer too small: self.data.len()={} < read len={len}", buf.len()),
            ));
        }
        reader.read_exact(&mut buf[..len])?;

        self.shape = MatZnxShape::new(new_n, new_rows, new_cols_in, new_cols_out, new_size);
        Ok(())
    }
}

impl<D: HostDataRef> WriterTo for MatZnx<D> {
    fn write_to<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<()> {
        writer.write_u64::<LittleEndian>(self.n() as u64)?;
        writer.write_u64::<LittleEndian>(self.size() as u64)?;
        writer.write_u64::<LittleEndian>(self.rows() as u64)?;
        writer.write_u64::<LittleEndian>(self.cols_in() as u64)?;
        writer.write_u64::<LittleEndian>(self.cols_out() as u64)?;
        let logical_len: usize = MatZnx::<Vec<u8>>::bytes_of(self.n(), self.rows(), self.cols_in(), self.cols_out(), self.size());
        let buf: &[u8] = self.data.as_ref();
        if buf.len() < logical_len {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "MatZnx buffer too small: self.data.len()={} < logical_len={logical_len}",
                    buf.len()
                ),
            ));
        }
        writer.write_u64::<LittleEndian>(logical_len as u64)?;
        writer.write_all(&buf[..logical_len])?;
        Ok(())
    }
}

impl<D: HostDataRef> fmt::Display for MatZnx<D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "MatZnx(n={}, rows={}, cols_in={}, cols_out={}, size={})",
            self.n(),
            self.rows(),
            self.cols_in(),
            self.cols_out(),
            self.size()
        )?;

        for row_i in 0..self.rows() {
            writeln!(f, "Row {row_i}:")?;
            for col_i in 0..self.cols_in() {
                writeln!(f, "cols_in {col_i}:")?;
                writeln!(f, "{}:", self.at(row_i, col_i))?;
            }
        }
        Ok(())
    }
}

impl<D: HostDataMut> ZnxZero for MatZnx<D> {
    fn zero(&mut self) {
        self.raw_mut().fill(0)
    }

    fn zero_at(&mut self, i: usize, j: usize) {
        self.at_mut(i, j).zero();
    }
}
