use crate::{
    alloc_aligned,
    layouts::{
        Backend, Data, DataView, DataViewMut, DigestU64, FillUniform, HostDataMut, HostDataRef, MatZnxInfos, ReaderFrom,
        ToOwnedDeep, VecZnx, WriterTo, ZnxInfos, ZnxWord, ZnxZero,
    },
    source::Source,
};
use std::{
    fmt,
    hash::{DefaultHasher, Hasher},
    marker::PhantomData,
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
/// The type parameter `W` names the coefficient word (byte-layout contract)
/// of the buffer.
///
/// Used primarily as the plaintext input to [`VmpPrepare`](crate::api::VmpPrepare),
/// which converts a `MatZnx` into a prepared [`VmpPMat`](crate::layouts::VmpPMat)
/// for vector-matrix products.
#[repr(C)]
#[derive(PartialEq, Eq, Clone, Hash)]
pub struct MatZnx<D: Data, W: ZnxWord> {
    data: D,
    shape: MatZnxShape,
    pub _phantom: PhantomData<W>,
}

impl<D: HostDataRef, W: ZnxWord> DigestU64 for MatZnx<D, W> {
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

impl<D: HostDataRef, W: ZnxWord> ToOwnedDeep for MatZnx<D, W> {
    type Owned = MatZnx<Vec<u8>, W>;
    fn to_owned_deep(&self) -> Self::Owned {
        MatZnx {
            data: self.data.as_ref().to_vec(),
            shape: self.shape,
            _phantom: PhantomData,
        }
    }
}

impl<D: HostDataRef, W: ZnxWord> fmt::Debug for MatZnx<D, W> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{self}")
    }
}

impl<D: Data, W: ZnxWord> ZnxInfos for MatZnx<D, W> {
    fn n(&self) -> usize {
        self.shape.n()
    }

    fn size(&self) -> usize {
        self.shape.size()
    }

    fn poly_count(&self) -> usize {
        crate::layouts::checked_product(
            &[self.rows(), self.cols_in(), self.cols_out(), self.size()],
            "MatZnx polynomial count",
        )
    }
}

impl<D: Data, W: ZnxWord> MatZnxInfos for MatZnx<D, W> {
    fn rows(&self) -> usize {
        self.shape.rows()
    }

    fn cols_in(&self) -> usize {
        self.shape.cols_in()
    }

    fn cols_out(&self) -> usize {
        self.shape.cols_out()
    }
}

impl<D: Data, W: ZnxWord> DataView for MatZnx<D, W> {
    type D = D;
    fn data(&self) -> &Self::D {
        &self.data
    }
}

impl<D: Data, W: ZnxWord> DataViewMut for MatZnx<D, W> {
    fn data_mut(&mut self) -> &mut Self::D {
        &mut self.data
    }
}

impl<D: HostDataRef, W: ZnxWord> MatZnx<D, W> {
    /// Returns the whole element view as a scalar slice.
    ///
    /// A matrix container has no flat `(col, limb)` indexing, so it exposes the
    /// buffer rather than implementing [`ZnxView`]. Use [`Self::at`] to address
    /// an individual entry.
    pub fn raw(&self) -> &[W] {
        let span: usize = crate::layouts::element_view_span(self);
        crate::layouts::raw_scalars(self.data.as_ref(), span)
    }
}

impl<D: HostDataMut, W: ZnxWord> MatZnx<D, W> {
    /// Mutable counterpart of [`Self::raw`].
    pub fn raw_mut(&mut self) -> &mut [W] {
        let span: usize = crate::layouts::element_view_span(self);
        crate::layouts::raw_scalars_mut(self.data.as_mut(), span)
    }
}

impl<D: Data, W: ZnxWord> MatZnx<D, W> {
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

    /// Returns the byte size of one `(row, col)` entry: an inner [`VecZnx`]
    /// with `cols_out` columns and `size` limbs of `W` words.
    fn entry_bytes(&self) -> usize {
        crate::layouts::checked_product(
            &[self.n(), self.cols_out(), self.size(), size_of::<W>()],
            "MatZnx entry byte size",
        )
    }
}

impl<D: Data, W: ZnxWord> MatZnx<D, W> {
    /// Returns the number of bytes required to store the matrix.
    pub fn bytes_of(n: usize, rows: usize, cols_in: usize, cols_out: usize, size: usize) -> usize {
        crate::layouts::checked_product(
            &[rows, cols_in, VecZnx::<Vec<u8>, W>::bytes_of(n, cols_out, size)],
            "MatZnx byte size",
        )
    }
}

impl<W: ZnxWord> MatZnx<Vec<u8>, W> {
    /// Allocates a zero-initialized `MatZnx` aligned to [`DEFAULTALIGN`](crate::DEFAULTALIGN).
    pub(crate) fn alloc(n: usize, rows: usize, cols_in: usize, cols_out: usize, size: usize) -> Self {
        let data: Vec<u8> = alloc_aligned(Self::bytes_of(n, rows, cols_in, cols_out, size));
        Self {
            data,
            shape: MatZnxShape::new(n, rows, cols_in, cols_out, size),
            _phantom: PhantomData,
        }
    }

    pub fn from_bytes(n: usize, rows: usize, cols_in: usize, cols_out: usize, size: usize, bytes: impl Into<Vec<u8>>) -> Self {
        let data: Vec<u8> = bytes.into();
        assert!(data.len() == Self::bytes_of(n, rows, cols_in, cols_out, size));
        crate::assert_alignment(data.as_ptr());
        Self {
            data,
            shape: MatZnxShape::new(n, rows, cols_in, cols_out, size),
            _phantom: PhantomData,
        }
    }
}

impl<D: HostDataRef, W: ZnxWord> MatZnx<D, W> {
    /// Returns a shared [`VecZnx`] view of the entry at `(row, col)`.
    ///
    /// # Panics
    ///
    /// Panics if `row >= rows` or `col >= cols_in`.
    pub fn at(&self, row: usize, col: usize) -> VecZnx<&[u8], W> {
        assert!(row < self.rows(), "rows: {} >= {}", row, self.rows());
        assert!(col < self.cols_in(), "cols: {} >= {}", col, self.cols_in());

        let nb_bytes: usize = self.entry_bytes();
        let start: usize = nb_bytes
            .checked_mul(self.cols_in())
            .and_then(|x| x.checked_mul(row))
            .and_then(|x| col.checked_mul(nb_bytes).and_then(|y| x.checked_add(y)))
            .expect("MatZnx entry offset overflows usize");
        let end: usize = start.checked_add(nb_bytes).expect("MatZnx entry end overflows usize");

        VecZnx::from_data(&self.data.as_ref()[start..end], self.n(), self.cols_out(), self.size())
    }
}

impl<D: HostDataMut, W: ZnxWord> MatZnx<D, W> {
    /// Returns a mutable [`VecZnx`] view of the entry at `(row, col)`.
    ///
    /// # Panics
    ///
    /// Panics if `row >= rows` or `col >= cols_in`.
    pub fn at_mut(&mut self, row: usize, col: usize) -> VecZnx<&mut [u8], W> {
        assert!(row < self.rows(), "rows: {} >= {}", row, self.rows());
        assert!(col < self.cols_in(), "cols: {} >= {}", col, self.cols_in());

        let n: usize = self.n();
        let cols_out: usize = self.cols_out();
        let cols_in: usize = self.cols_in();
        let size: usize = self.size();

        let nb_bytes: usize = self.entry_bytes();
        let start: usize = nb_bytes
            .checked_mul(cols_in)
            .and_then(|x| x.checked_mul(row))
            .and_then(|x| col.checked_mul(nb_bytes).and_then(|y| x.checked_add(y)))
            .expect("MatZnx entry offset overflows usize");
        let end: usize = start.checked_add(nb_bytes).expect("MatZnx entry end overflows usize");

        VecZnx::from_data(&mut self.data.as_mut()[start..end], n, cols_out, size)
    }
}

/// Returns a shared backend-native entry view of a backend-owned `MatZnx`.
pub trait MatZnxAtBackendRef<B: Backend> {
    fn at_backend(&self, row: usize, col: usize) -> VecZnx<B::BufRef<'_>, B::ZnxWord>;
}

impl<B: Backend> MatZnxAtBackendRef<B> for MatZnx<B::OwnedBuf, B::ZnxWord> {
    fn at_backend(&self, row: usize, col: usize) -> VecZnx<B::BufRef<'_>, B::ZnxWord> {
        assert!(row < self.rows(), "rows: {} >= {}", row, self.rows());
        assert!(col < self.cols_in(), "cols: {} >= {}", col, self.cols_in());

        let nb_bytes: usize = B::bytes_of_vec_znx(self.n(), self.cols_out(), self.size());
        let start: usize = nb_bytes
            .checked_mul(self.cols_in())
            .and_then(|x| x.checked_mul(row))
            .and_then(|x| col.checked_mul(nb_bytes).and_then(|y| x.checked_add(y)))
            .expect("MatZnx backend entry offset overflows usize");
        let end: usize = start.checked_add(nb_bytes).expect("MatZnx backend entry end overflows usize");

        VecZnx::from_data(
            B::region(&self.data, start, end - start),
            self.n(),
            self.cols_out(),
            self.size(),
        )
    }
}

pub fn mat_znx_at_backend_ref_from_ref<'a, 'b, B: Backend + 'b>(
    mat: &'a MatZnx<B::BufRef<'b>, B::ZnxWord>,
    row: usize,
    col: usize,
) -> VecZnx<B::BufRef<'a>, B::ZnxWord> {
    assert!(row < mat.rows(), "rows: {} >= {}", row, mat.rows());
    assert!(col < mat.cols_in(), "cols: {} >= {}", col, mat.cols_in());

    let nb_bytes: usize = B::bytes_of_vec_znx(mat.n(), mat.cols_out(), mat.size());
    let start: usize = nb_bytes
        .checked_mul(mat.cols_in())
        .and_then(|x| x.checked_mul(row))
        .and_then(|x| col.checked_mul(nb_bytes).and_then(|y| x.checked_add(y)))
        .expect("MatZnx backend entry offset overflows usize");
    let end: usize = start.checked_add(nb_bytes).expect("MatZnx backend entry end overflows usize");

    VecZnx::from_data(
        B::region_ref(&mat.data, start, end - start),
        mat.n(),
        mat.cols_out(),
        mat.size(),
    )
}

pub fn mat_znx_at_backend_ref_from_mut<'a, 'b, B: Backend + 'b>(
    mat: &'a MatZnx<B::BufMut<'b>, B::ZnxWord>,
    row: usize,
    col: usize,
) -> VecZnx<B::BufRef<'a>, B::ZnxWord> {
    assert!(row < mat.rows(), "rows: {} >= {}", row, mat.rows());
    assert!(col < mat.cols_in(), "cols: {} >= {}", col, mat.cols_in());

    let nb_bytes: usize = B::bytes_of_vec_znx(mat.n(), mat.cols_out(), mat.size());
    let start: usize = nb_bytes
        .checked_mul(mat.cols_in())
        .and_then(|x| x.checked_mul(row))
        .and_then(|x| col.checked_mul(nb_bytes).and_then(|y| x.checked_add(y)))
        .expect("MatZnx backend entry offset overflows usize");
    let end: usize = start.checked_add(nb_bytes).expect("MatZnx backend entry end overflows usize");

    VecZnx::from_data(
        B::region_ref_mut(&mat.data, start, end - start),
        mat.n(),
        mat.cols_out(),
        mat.size(),
    )
}

/// Returns a mutable backend-native entry view of a backend-owned `MatZnx`.
pub trait MatZnxAtBackendMut<B: Backend> {
    fn at_backend_mut(&mut self, row: usize, col: usize) -> VecZnx<B::BufMut<'_>, B::ZnxWord>;
}

impl<B: Backend> MatZnxAtBackendMut<B> for MatZnx<B::OwnedBuf, B::ZnxWord> {
    fn at_backend_mut(&mut self, row: usize, col: usize) -> VecZnx<B::BufMut<'_>, B::ZnxWord> {
        assert!(row < self.rows(), "rows: {} >= {}", row, self.rows());
        assert!(col < self.cols_in(), "cols: {} >= {}", col, self.cols_in());

        let n: usize = self.n();
        let cols_out: usize = self.cols_out();
        let cols_in: usize = self.cols_in();
        let size: usize = self.size();
        let nb_bytes: usize = B::bytes_of_vec_znx(n, cols_out, size);
        let start: usize = nb_bytes
            .checked_mul(cols_in)
            .and_then(|x| x.checked_mul(row))
            .and_then(|x| col.checked_mul(nb_bytes).and_then(|y| x.checked_add(y)))
            .expect("MatZnx backend entry offset overflows usize");
        let end: usize = start.checked_add(nb_bytes).expect("MatZnx backend entry end overflows usize");

        VecZnx::from_data(B::region_mut(&mut self.data, start, end - start), n, cols_out, size)
    }
}

pub fn mat_znx_at_backend_mut_from_mut<'a, 'b, B: Backend + 'b>(
    mat: &'a mut MatZnx<B::BufMut<'b>, B::ZnxWord>,
    row: usize,
    col: usize,
) -> VecZnx<B::BufMut<'a>, B::ZnxWord> {
    assert!(row < mat.rows(), "rows: {} >= {}", row, mat.rows());
    assert!(col < mat.cols_in(), "cols: {} >= {}", col, mat.cols_in());

    let n: usize = mat.n();
    let cols_out: usize = mat.cols_out();
    let cols_in: usize = mat.cols_in();
    let size: usize = mat.size();
    let nb_bytes: usize = B::bytes_of_vec_znx(n, cols_out, size);
    let start: usize = nb_bytes
        .checked_mul(cols_in)
        .and_then(|x| x.checked_mul(row))
        .and_then(|x| col.checked_mul(nb_bytes).and_then(|y| x.checked_add(y)))
        .expect("MatZnx backend entry offset overflows usize");
    let end: usize = start.checked_add(nb_bytes).expect("MatZnx backend entry end overflows usize");

    VecZnx::from_data(B::region_mut_ref(&mut mat.data, start, end - start), n, cols_out, size)
}

impl<D: HostDataMut, W: ZnxWord> FillUniform for MatZnx<D, W> {
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

/// Owned `MatZnx` backed by a `Vec<u8>`.
pub type MatZnxOwned<W> = MatZnx<Vec<u8>, W>;
/// Mutably borrowed `MatZnx`.
pub type MatZnxMut<'a, W> = MatZnx<&'a mut [u8], W>;
/// Immutably borrowed `MatZnx`.
pub type MatZnxRef<'a, W> = MatZnx<&'a [u8], W>;
/// Shared backend-native borrow of a `MatZnx`.
pub type MatZnxBackendRef<'a, B> = MatZnx<<B as Backend>::BufRef<'a>, <B as Backend>::ZnxWord>;
/// Mutable backend-native borrow of a `MatZnx`.
pub type MatZnxBackendMut<'a, B> = MatZnx<<B as Backend>::BufMut<'a>, <B as Backend>::ZnxWord>;

/// Borrow a backend-owned `MatZnx` using the backend's native view type.
pub trait MatZnxToBackendRef<B: Backend> {
    fn to_backend_ref(&self) -> MatZnxBackendRef<'_, B>;
}

impl<B: Backend> MatZnxToBackendRef<B> for MatZnx<B::OwnedBuf, B::ZnxWord> {
    fn to_backend_ref(&self) -> MatZnxBackendRef<'_, B> {
        MatZnx {
            data: B::view(&self.data),
            shape: self.shape,
            _phantom: PhantomData,
        }
    }
}

impl<'b, B: Backend + 'b> MatZnxToBackendRef<B> for &MatZnx<B::BufRef<'b>, B::ZnxWord> {
    fn to_backend_ref(&self) -> MatZnxBackendRef<'_, B> {
        mat_znx_backend_ref_from_ref::<B>(self)
    }
}

impl<'b, B: Backend + 'b> MatZnxToBackendRef<B> for &mut MatZnx<B::BufMut<'b>, B::ZnxWord> {
    fn to_backend_ref(&self) -> MatZnxBackendRef<'_, B> {
        mat_znx_backend_ref_from_mut::<B>(self)
    }
}

pub fn mat_znx_backend_ref_from_ref<'a, 'b, B: Backend + 'b>(
    mat: &'a MatZnx<B::BufRef<'b>, B::ZnxWord>,
) -> MatZnxBackendRef<'a, B> {
    MatZnx {
        data: B::view_ref(&mat.data),
        shape: mat.shape,
        _phantom: PhantomData,
    }
}

pub fn mat_znx_backend_ref_from_mut<'a, 'b, B: Backend + 'b>(
    mat: &'a MatZnx<B::BufMut<'b>, B::ZnxWord>,
) -> MatZnxBackendRef<'a, B> {
    MatZnx {
        data: B::view_ref_mut(&mat.data),
        shape: mat.shape,
        _phantom: PhantomData,
    }
}

/// Mutably borrow a backend-owned `MatZnx` using the backend's native view type.
pub trait MatZnxToBackendMut<B: Backend> {
    fn to_backend_mut(&mut self) -> MatZnxBackendMut<'_, B>;
}

impl<B: Backend> MatZnxToBackendMut<B> for MatZnx<B::OwnedBuf, B::ZnxWord> {
    fn to_backend_mut(&mut self) -> MatZnxBackendMut<'_, B> {
        MatZnx {
            data: B::view_mut(&mut self.data),
            shape: self.shape,
            _phantom: PhantomData,
        }
    }
}

impl<'b, B: Backend + 'b> MatZnxToBackendMut<B> for &mut MatZnx<B::BufMut<'b>, B::ZnxWord> {
    fn to_backend_mut(&mut self) -> MatZnxBackendMut<'_, B> {
        mat_znx_backend_mut_from_mut::<B>(self)
    }
}

pub fn mat_znx_backend_mut_from_mut<'a, 'b, B: Backend + 'b>(
    mat: &'a mut MatZnx<B::BufMut<'b>, B::ZnxWord>,
) -> MatZnxBackendMut<'a, B> {
    MatZnx {
        data: B::view_mut_ref(&mut mat.data),
        shape: mat.shape,
        _phantom: PhantomData,
    }
}

impl<D: Data, W: ZnxWord> MatZnx<D, W> {
    pub fn from_data(data: D, n: usize, rows: usize, cols_in: usize, cols_out: usize, size: usize) -> Self {
        Self {
            data,
            shape: MatZnxShape::new(n, rows, cols_in, cols_out, size),
            _phantom: PhantomData,
        }
    }
}

impl<D: HostDataMut, W: ZnxWord> ReaderFrom for MatZnx<D, W> {
    fn read_from<R: std::io::Read>(&mut self, reader: &mut R) -> std::io::Result<()> {
        let new_n: usize = reader.read_u64::<LittleEndian>()? as usize;
        let new_size: usize = reader.read_u64::<LittleEndian>()? as usize;
        let new_rows: usize = reader.read_u64::<LittleEndian>()? as usize;
        let new_cols_in: usize = reader.read_u64::<LittleEndian>()? as usize;
        let new_cols_out: usize = reader.read_u64::<LittleEndian>()? as usize;
        let len: usize = reader.read_u64::<LittleEndian>()? as usize;

        let expected_len: usize = crate::layouts::checked_product(
            &[new_rows, new_cols_in, new_n, new_cols_out, new_size, size_of::<W>()],
            "MatZnx serialized byte size",
        );
        if expected_len != len {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "MatZnx metadata inconsistent: rows={new_rows} * cols_in={new_cols_in} * n={new_n} * cols_out={new_cols_out} * size={new_size} * {} = {expected_len} != data len={len}",
                    size_of::<W>()
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

impl<D: HostDataRef, W: ZnxWord> WriterTo for MatZnx<D, W> {
    fn write_to<Wr: std::io::Write>(&self, writer: &mut Wr) -> std::io::Result<()> {
        writer.write_u64::<LittleEndian>(self.n() as u64)?;
        writer.write_u64::<LittleEndian>(self.size() as u64)?;
        writer.write_u64::<LittleEndian>(self.rows() as u64)?;
        writer.write_u64::<LittleEndian>(self.cols_in() as u64)?;
        writer.write_u64::<LittleEndian>(self.cols_out() as u64)?;
        let logical_len: usize =
            crate::layouts::checked_product(&[self.rows(), self.cols_in(), self.entry_bytes()], "MatZnx logical byte size");
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

impl<D: HostDataRef, W: ZnxWord> fmt::Display for MatZnx<D, W> {
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

impl<D: HostDataMut, W: ZnxWord> ZnxZero for MatZnx<D, W> {
    fn zero(&mut self) {
        self.raw_mut().fill(W::zero())
    }

    fn zero_at(&mut self, i: usize, j: usize) {
        self.at_mut(i, j).zero();
    }
}
