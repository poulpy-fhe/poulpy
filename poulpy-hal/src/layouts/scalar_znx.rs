use std::{
    fmt,
    hash::{DefaultHasher, Hasher},
    marker::PhantomData,
};

use crate::{
    AlignedBuf, alloc_aligned,
    layouts::{
        Backend, Data, DataView, DataViewMut, DigestU64, HostDataMut, HostDataRef, ReaderFrom, Ring, Standard, ToOwnedDeep,
        VecZnx, VecZnxBackendMut, VecZnxBackendRef, VecZnxInfos, WriterTo, ZnxInfos, ZnxView, ZnxViewMut, ZnxWord, ZnxZero,
        read_ring_tag, read_usize_fields, serialized_byte_len, write_ring_tag,
    },
};

#[repr(C)]
#[derive(PartialEq, Eq, Debug, Clone, Copy, Hash, Default)]
pub struct ScalarZnxShape {
    n: usize,
    cols: usize,
}

impl ScalarZnxShape {
    pub const fn new(n: usize, cols: usize) -> Self {
        Self { n, cols }
    }

    pub const fn n(self) -> usize {
        self.n
    }

    pub const fn cols(self) -> usize {
        self.cols
    }
}

impl<D: Data, W: ZnxWord, Rg: Ring> ScalarZnx<D, W, Rg> {
    pub fn n(&self) -> usize {
        self.shape.n()
    }

    pub fn cols(&self) -> usize {
        self.shape.cols()
    }

    pub fn shape(&self) -> ScalarZnxShape {
        self.shape
    }
}

/// A single-limb polynomial vector of the ring `Rg`.
///
/// `ScalarZnx` is a specialization of [`VecZnx`] with exactly one limb
/// (`size == 1`). It is the primary type for plaintext polynomials,
/// secret keys, and other single-precision ring elements.
///
/// The type parameter `D` controls ownership: `AlignedBuf` for owned,
/// `&[u8]` for shared borrows, `&mut [u8]` for mutable borrows.
/// The type parameter `W` names the coefficient word (byte-layout
/// contract) of the buffer, and `Rg` the ring of the polynomials (see
/// [layouts](crate::layouts#value-model)).
#[repr(C)]
#[derive(PartialEq, Eq, Clone, Hash)]
pub struct ScalarZnx<D: Data, W: ZnxWord, Rg: Ring = Standard> {
    pub data: D,
    shape: ScalarZnxShape,
    pub _phantom: PhantomData<(W, Rg)>,
}

/// Shape and the first 16 coefficients of each column, as `VecZnx` prints.
impl<D: HostDataRef, W: ZnxWord, Rg: Ring> fmt::Debug for ScalarZnx<D, W, Rg> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ScalarZnx(n={}, cols={})", self.n(), self.cols())?;
        for col in 0..self.cols() {
            let coeffs = self.at(col, 0);
            write!(f, " column {col}: {:?}", &coeffs[..coeffs.len().min(16)])?;
        }
        Ok(())
    }
}

impl<D: HostDataRef, W: ZnxWord, Rg: Ring> DigestU64 for ScalarZnx<D, W, Rg> {
    fn digest_u64(&self) -> u64 {
        let mut h: DefaultHasher = DefaultHasher::new();
        h.write(self.data.as_ref());
        h.write_usize(self.n());
        h.write_usize(self.cols());
        h.finish()
    }
}

impl<D: HostDataRef, W: ZnxWord, Rg: Ring> ToOwnedDeep for ScalarZnx<D, W, Rg> {
    type Owned = ScalarZnx<AlignedBuf, W, Rg>;
    fn to_owned_deep(&self) -> Self::Owned {
        ScalarZnx {
            data: AlignedBuf::from(self.data.as_ref()),
            shape: self.shape,
            _phantom: PhantomData,
        }
    }
}

impl<D: Data, W: ZnxWord, Rg: Ring> ZnxInfos for ScalarZnx<D, W, Rg> {
    fn n(&self) -> usize {
        self.shape.n()
    }

    fn size(&self) -> usize {
        1
    }

    fn poly_count(&self) -> usize {
        crate::layouts::checked_product(&[self.cols(), self.size()], "polynomial count")
    }
}

impl<D: Data, W: ZnxWord, Rg: Ring> VecZnxInfos for ScalarZnx<D, W, Rg> {
    fn cols(&self) -> usize {
        self.shape.cols()
    }
}

impl<D: Data, W: ZnxWord, Rg: Ring> DataView for ScalarZnx<D, W, Rg> {
    type D = D;
    fn data(&self) -> &Self::D {
        &self.data
    }
}

impl<D: Data, W: ZnxWord, Rg: Ring> DataViewMut for ScalarZnx<D, W, Rg> {
    fn data_mut(&mut self) -> &mut Self::D {
        &mut self.data
    }
}

impl<D: HostDataRef, W: ZnxWord, Rg: Ring> ZnxView for ScalarZnx<D, W, Rg> {
    type Scalar = W;
}

impl<D: Data, W: ZnxWord, Rg: Ring> ScalarZnx<D, W, Rg> {
    /// Returns the number of bytes required to store a `ScalarZnx` with
    /// ring degree `n` and `cols` columns: `n * cols * size_of::<W>()`.
    pub fn bytes_of(n: usize, cols: usize) -> usize {
        crate::layouts::checked_product(&[n, cols, size_of::<W>()], "ScalarZnx byte size")
    }
}

impl<W: ZnxWord, Rg: Ring> ScalarZnx<AlignedBuf, W, Rg> {
    /// Allocates a zero-initialized `ScalarZnx` aligned to [`DEFAULTALIGN`](crate::DEFAULTALIGN).
    pub(crate) fn alloc(n: usize, cols: usize) -> Self {
        let data: AlignedBuf = alloc_aligned::<u8>(Self::bytes_of(n, cols));
        Self {
            data,
            shape: ScalarZnxShape::new(n, cols),
            _phantom: PhantomData,
        }
    }

    /// Wraps an existing byte buffer into a `ScalarZnx`.
    ///
    /// # Panics
    ///
    /// Panics if the buffer length does not equal `bytes_of(n, cols)`; a `Vec<u8>`
    /// argument is first copied into storage padded to a 64-byte multiple, so its padded
    /// length is what is compared.
    pub fn from_bytes(n: usize, cols: usize, bytes: impl Into<AlignedBuf>) -> Self {
        let data: AlignedBuf = bytes.into();
        assert!(data.len() == crate::layouts::padded_bytes(Self::bytes_of(n, cols)));
        Self {
            data,
            shape: ScalarZnxShape::new(n, cols),
            _phantom: PhantomData,
        }
    }
}

impl<D: HostDataMut, W: ZnxWord, Rg: Ring> ZnxZero for ScalarZnx<D, W, Rg> {
    fn zero(&mut self) {
        self.raw_mut().fill(W::zero())
    }
    fn zero_at(&mut self, i: usize, j: usize) {
        self.at_mut(i, j).fill(W::zero());
    }
}

/// Owned `ScalarZnx` backed by an `AlignedBuf`.
pub type ScalarZnxOwned<W, Rg = Standard> = ScalarZnx<AlignedBuf, W, Rg>;
/// Shared backend-native borrow of a `ScalarZnx`.
pub type ScalarZnxBackendRef<'a, B, Rg = Standard> = ScalarZnx<<B as Backend>::BufRef<'a>, <B as Backend>::ZnxWord, Rg>;
/// Mutable backend-native borrow of a `ScalarZnx`.
pub type ScalarZnxBackendMut<'a, B, Rg = Standard> = ScalarZnx<<B as Backend>::BufMut<'a>, <B as Backend>::ZnxWord, Rg>;

impl<D: Data, W: ZnxWord, Rg: Ring> ScalarZnx<D, W, Rg> {
    /// Constructs a `ScalarZnx` from raw parts without validation.
    pub fn from_data(data: D, n: usize, cols: usize) -> Self {
        Self {
            data,
            shape: ScalarZnxShape::new(n, cols),
            _phantom: PhantomData,
        }
    }
}

/// Borrow a backend-owned `ScalarZnx` using the backend's native view type.
pub trait ScalarZnxToBackendRef<B: Backend, Rg: Ring = Standard> {
    fn to_backend_ref(&self) -> ScalarZnxBackendRef<'_, B, Rg>;
}

impl<B: Backend, Rg: Ring> ScalarZnxToBackendRef<B, Rg> for ScalarZnx<B::OwnedBuf, B::ZnxWord, Rg> {
    fn to_backend_ref(&self) -> ScalarZnxBackendRef<'_, B, Rg> {
        ScalarZnx {
            data: B::view(&self.data),
            shape: self.shape,
            _phantom: PhantomData,
        }
    }
}

impl<B: Backend, Rg: Ring> ScalarZnxToBackendRef<B, Rg> for &ScalarZnx<B::BufRef<'_>, B::ZnxWord, Rg> {
    fn to_backend_ref(&self) -> ScalarZnxBackendRef<'_, B, Rg> {
        ScalarZnx {
            data: B::view_ref(&self.data),
            shape: self.shape,
            _phantom: PhantomData,
        }
    }
}

impl<B: Backend, Rg: Ring> ScalarZnxToBackendRef<B, Rg> for &mut ScalarZnx<B::BufMut<'_>, B::ZnxWord, Rg> {
    fn to_backend_ref(&self) -> ScalarZnxBackendRef<'_, B, Rg> {
        scalar_znx_backend_ref_from_mut::<B, Rg>(self)
    }
}

/// Mutably borrow a backend-owned `ScalarZnx` using the backend's native view type.
pub trait ScalarZnxToBackendMut<B: Backend, Rg: Ring = Standard> {
    fn to_backend_mut(&mut self) -> ScalarZnxBackendMut<'_, B, Rg>;
}

impl<B: Backend, Rg: Ring> ScalarZnxToBackendMut<B, Rg> for ScalarZnx<B::OwnedBuf, B::ZnxWord, Rg> {
    fn to_backend_mut(&mut self) -> ScalarZnxBackendMut<'_, B, Rg> {
        ScalarZnx {
            data: B::view_mut(&mut self.data),
            shape: self.shape,
            _phantom: PhantomData,
        }
    }
}

impl<B: Backend, Rg: Ring> ScalarZnxToBackendMut<B, Rg> for &mut ScalarZnx<B::BufMut<'_>, B::ZnxWord, Rg> {
    fn to_backend_mut(&mut self) -> ScalarZnxBackendMut<'_, B, Rg> {
        scalar_znx_backend_mut_from_mut::<B, Rg>(self)
    }
}

fn scalar_znx_backend_ref_from_mut<'a, 'b, B: Backend + 'b, Rg: Ring>(
    scalar: &'a ScalarZnx<B::BufMut<'b>, B::ZnxWord, Rg>,
) -> ScalarZnxBackendRef<'a, B, Rg> {
    ScalarZnx {
        data: B::view_ref_mut(&scalar.data),
        shape: scalar.shape,
        _phantom: PhantomData,
    }
}

fn scalar_znx_backend_mut_from_mut<'a, 'b, B: Backend + 'b, Rg: Ring>(
    scalar: &'a mut ScalarZnx<B::BufMut<'b>, B::ZnxWord, Rg>,
) -> ScalarZnxBackendMut<'a, B, Rg> {
    ScalarZnx {
        data: B::view_mut_ref(&mut scalar.data),
        shape: scalar.shape,
        _phantom: PhantomData,
    }
}

impl<D: HostDataRef, W: ZnxWord, Rg: Ring> ScalarZnx<D, W, Rg> {
    /// Borrow a host-visible `ScalarZnx` as a shared byte-slice view.
    pub fn to_ref(&self) -> ScalarZnx<&[u8], W, Rg> {
        ScalarZnx {
            data: self.data.as_ref(),
            shape: self.shape,
            _phantom: PhantomData,
        }
    }
}

impl<D: HostDataRef, W: ZnxWord, Rg: Ring> ScalarZnx<D, W, Rg> {
    /// Views this `ScalarZnx` as a [`VecZnx`] with `size == 1`.
    pub fn as_vec_znx(&self) -> VecZnx<&[u8], W, Rg> {
        VecZnx::from_data(self.data.as_ref(), self.n(), self.cols(), 1)
    }
}

/// Views a backend-owned `ScalarZnx` as a backend-native [`VecZnx`] with `size == 1`.
pub trait ScalarZnxAsVecZnxBackendRef<B: Backend, Rg: Ring = Standard> {
    fn as_vec_znx_backend(&self) -> VecZnx<B::BufRef<'_>, B::ZnxWord, Rg>;
}

impl<B: Backend, Rg: Ring> ScalarZnxAsVecZnxBackendRef<B, Rg> for ScalarZnx<B::OwnedBuf, B::ZnxWord, Rg> {
    fn as_vec_znx_backend(&self) -> VecZnx<B::BufRef<'_>, B::ZnxWord, Rg> {
        VecZnx::from_data(B::view(&self.data), self.n(), self.cols(), 1)
    }
}

pub fn scalar_znx_as_vec_znx_backend_ref_from_ref<'a, 'b, B: Backend + 'b, Rg: Ring>(
    scalar: &'a ScalarZnx<B::BufRef<'b>, B::ZnxWord, Rg>,
) -> VecZnxBackendRef<'a, B, Rg> {
    VecZnx::from_data(B::view_ref(&scalar.data), scalar.n(), scalar.cols(), 1)
}

pub fn scalar_znx_as_vec_znx_backend_ref_from_mut<'a, 'b, B: Backend + 'b, Rg: Ring>(
    scalar: &'a ScalarZnx<B::BufMut<'b>, B::ZnxWord, Rg>,
) -> VecZnxBackendRef<'a, B, Rg> {
    VecZnx::from_data(B::view_ref_mut(&scalar.data), scalar.n(), scalar.cols(), 1)
}

impl<D: HostDataMut, W: ZnxWord, Rg: Ring> ScalarZnx<D, W, Rg> {
    /// Mutably views this `ScalarZnx` as a [`VecZnx`] with `size == 1`.
    pub fn as_vec_znx_mut(&mut self) -> VecZnx<&mut [u8], W, Rg> {
        let shape = self.shape();
        VecZnx::from_data(self.data.as_mut(), shape.n(), shape.cols(), 1)
    }
}

/// Mutably views a backend-owned `ScalarZnx` as a backend-native [`VecZnx`] with `size == 1`.
pub trait ScalarZnxAsVecZnxBackendMut<B: Backend, Rg: Ring = Standard> {
    fn as_vec_znx_backend_mut(&mut self) -> VecZnx<B::BufMut<'_>, B::ZnxWord, Rg>;
}

impl<B: Backend, Rg: Ring> ScalarZnxAsVecZnxBackendMut<B, Rg> for ScalarZnx<B::OwnedBuf, B::ZnxWord, Rg> {
    fn as_vec_znx_backend_mut(&mut self) -> VecZnx<B::BufMut<'_>, B::ZnxWord, Rg> {
        let shape = self.shape();
        VecZnx::from_data(B::view_mut(&mut self.data), shape.n(), shape.cols(), 1)
    }
}

pub fn scalar_znx_as_vec_znx_backend_mut_from_mut<'a, 'b, B: Backend + 'b, Rg: Ring>(
    scalar: &'a mut ScalarZnx<B::BufMut<'b>, B::ZnxWord, Rg>,
) -> VecZnxBackendMut<'a, B, Rg> {
    let shape = scalar.shape();
    VecZnx::from_data(B::view_mut_ref(&mut scalar.data), shape.n(), shape.cols(), 1)
}

use byteorder::{LittleEndian, WriteBytesExt};

impl<D: HostDataMut, W: ZnxWord, Rg: Ring> ReaderFrom for ScalarZnx<D, W, Rg> {
    /// Reads the format of [`WriterTo::write_to`] into this scalar, which takes the stream's shape. A stream of another
    /// ring, whose shape and byte length disagree, or larger than this scalar's buffer is
    /// [`std::io::ErrorKind::InvalidData`], refused before its coefficients.
    fn read_from<R: std::io::Read>(&mut self, reader: &mut R) -> std::io::Result<()> {
        read_ring_tag::<Rg, R>(reader)?;
        let [new_n, new_cols, len] = read_usize_fields::<3, R>(reader)?;

        let expected_len: usize = serialized_byte_len::<W>(&[new_n, new_cols])?;
        if expected_len != len {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "ScalarZnx metadata inconsistent: n={new_n} * cols={new_cols} * {} = {expected_len} != data len={len}",
                    size_of::<W>()
                ),
            ));
        }

        let buf: &mut [u8] = self.data.as_mut();
        if buf.len() < len {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("ScalarZnx buffer too small: self.data.len()={} < read len={len}", buf.len()),
            ));
        }
        reader.read_exact(&mut buf[..len])?;

        self.shape = ScalarZnxShape::new(new_n, new_cols);
        Ok(())
    }
}

impl<D: HostDataRef, W: ZnxWord, Rg: Ring> WriterTo for ScalarZnx<D, W, Rg> {
    /// Writes the ring's [`Ring::WIRE_TAG`] (one byte), then `n`, `cols` and the byte length `n * cols * size_of::<W>()`
    /// as little-endian `u64`, then the coefficients in buffer order.
    fn write_to<Wr: std::io::Write>(&self, writer: &mut Wr) -> std::io::Result<()> {
        let coeff_bytes =
            crate::layouts::checked_product(&[self.n(), self.cols(), size_of::<W>()], "ScalarZnx logical byte size");
        let buf: &[u8] = self.data.as_ref();
        if buf.len() < coeff_bytes {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "ScalarZnx buffer too small: self.data.len()={} < coeff_bytes={coeff_bytes}",
                    buf.len()
                ),
            ));
        }
        write_ring_tag::<Rg, Wr>(writer)?;
        writer.write_u64::<LittleEndian>(self.n() as u64)?;
        writer.write_u64::<LittleEndian>(self.cols() as u64)?;
        writer.write_u64::<LittleEndian>(coeff_bytes as u64)?;
        writer.write_all(&buf[..coeff_bytes])?;
        Ok(())
    }
}
