use std::{
    fmt,
    hash::{DefaultHasher, Hasher},
};

use crate::{
    alloc_aligned,
    layouts::{
        Data, DataMut, DataRef, DataView, DataViewMut, DigestU64, FillUniform, ReaderFrom, ScalarZnx, ToOwnedDeep, WriterTo,
        ZnxInfos, ZnxView, ZnxViewMut, ZnxZero,
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
#[derive(PartialEq, Eq, Clone, Copy, Hash)]
pub struct VecZnx<D: Data> {
    pub data: D,
    pub n: usize,
    pub cols: usize,
    pub size: usize,
    pub max_size: usize,
}

impl<D: DataRef> VecZnx<D> {
    /// Returns a read-only [`ScalarZnx`] view of a single limb of a single column.
    pub fn as_scalar_znx_ref(&self, col: usize, limb: usize) -> ScalarZnx<&[u8]> {
        ScalarZnx {
            data: bytemuck::cast_slice(self.at(col, limb)),
            n: self.n,
            cols: 1,
        }
    }
}

impl<D: DataMut> VecZnx<D> {
    /// Returns a mutable [`ScalarZnx`] view of a single limb of a single column.
    pub fn as_scalar_znx_mut(&mut self, col: usize, limb: usize) -> ScalarZnx<&mut [u8]> {
        ScalarZnx {
            n: self.n,
            cols: 1,
            data: bytemuck::cast_slice_mut(self.at_mut(col, limb)),
        }
    }
}

impl<D: Data + Default> Default for VecZnx<D> {
    fn default() -> Self {
        Self {
            data: D::default(),
            n: 0,
            cols: 0,
            size: 0,
            max_size: 0,
        }
    }
}

impl<D: DataRef> DigestU64 for VecZnx<D> {
    fn digest_u64(&self) -> u64 {
        let mut h: DefaultHasher = DefaultHasher::new();
        h.write(self.data.as_ref());
        h.write_usize(self.n);
        h.write_usize(self.cols);
        h.write_usize(self.size);
        h.write_usize(self.max_size);
        h.finish()
    }
}

impl<D: DataRef> ToOwnedDeep for VecZnx<D> {
    type Owned = VecZnx<Vec<u8>>;
    fn to_owned_deep(&self) -> Self::Owned {
        VecZnx {
            data: self.data.as_ref().to_vec(),
            n: self.n,
            cols: self.cols,
            size: self.size,
            max_size: self.max_size,
        }
    }
}

impl<D: DataRef> fmt::Debug for VecZnx<D> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{self}")
    }
}

impl<D: Data> ZnxInfos for VecZnx<D> {
    fn cols(&self) -> usize {
        self.cols
    }

    fn rows(&self) -> usize {
        1
    }

    fn n(&self) -> usize {
        self.n
    }

    fn size(&self) -> usize {
        self.size
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

impl<D: DataRef> ZnxView for VecZnx<D> {
    type Scalar = i64;
}

impl<D: Data> VecZnx<D> {
    /// Returns the allocated limb capacity.
    pub fn max_size(&self) -> usize {
        self.max_size
    }
}

impl<D: DataMut> VecZnx<D> {
    /// Sets the active limb count.
    ///
    /// # Panics
    ///
    /// Panics if `size > max_size`.
    pub fn set_size(&mut self, size: usize) {
        assert!(size <= self.max_size);
        self.size = size;
    }
}

impl VecZnx<Vec<u8>> {
    /// Returns the scratch space (in bytes) required by right-shift operations.
    pub fn rsh_tmp_bytes(n: usize) -> usize {
        n * size_of::<i64>()
    }

    /// Reallocates the backing buffer so capacity matches the `new_size` limb count.
    pub fn reallocate_limbs(&mut self, new_size: usize) {
        if self.size == new_size {
            return;
        }

        let mut compact: Self = Self::alloc(self.n, self.cols, new_size);
        let copy_len = compact.raw_mut().len().min(self.raw().len());
        compact.raw_mut()[..copy_len].copy_from_slice(&self.raw()[..copy_len]);
        *self = compact;
    }
}

impl<D: DataMut> ZnxZero for VecZnx<D> {
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
    pub fn alloc(n: usize, cols: usize, size: usize) -> Self {
        let data: Vec<u8> = alloc_aligned::<u8>(Self::bytes_of(n, cols, size));
        Self {
            data,
            n,
            cols,
            size,
            max_size: size,
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
            n,
            cols,
            size,
            max_size: size,
        }
    }
}

impl<D: Data> VecZnx<D> {
    /// Constructs a `VecZnx` from raw parts without validation.
    /// Sets `max_size = size`.
    pub fn from_data(data: D, n: usize, cols: usize, size: usize) -> Self {
        Self {
            data,
            n,
            cols,
            size,
            max_size: size,
        }
    }
}

impl<D: DataRef> fmt::Display for VecZnx<D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "VecZnx(n={}, cols={}, size={})", self.n, self.cols, self.size)?;

        for col in 0..self.cols {
            writeln!(f, "Column {col}:")?;
            for size in 0..self.size {
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

impl<D: DataMut> FillUniform for VecZnx<D> {
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

/// Borrow a `VecZnx` as a shared reference view.
pub trait VecZnxToRef {
    fn to_ref(&self) -> VecZnx<&[u8]>;
}

impl<D: DataRef> VecZnxToRef for VecZnx<D> {
    fn to_ref(&self) -> VecZnx<&[u8]> {
        VecZnx {
            data: self.data.as_ref(),
            n: self.n,
            cols: self.cols,
            size: self.size,
            max_size: self.max_size,
        }
    }
}

/// Borrow a `VecZnx` as a mutable reference view.
pub trait VecZnxToMut {
    fn to_mut(&mut self) -> VecZnx<&mut [u8]>;
}

impl<D: DataMut> VecZnxToMut for VecZnx<D> {
    fn to_mut(&mut self) -> VecZnx<&mut [u8]> {
        VecZnx {
            data: self.data.as_mut(),
            n: self.n,
            cols: self.cols,
            size: self.size,
            max_size: self.max_size,
        }
    }
}

impl<D: DataMut> ReaderFrom for VecZnx<D> {
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
        self.n = new_n;
        self.cols = new_cols;
        self.size = new_size;
        self.max_size = new_max_size;
        Ok(())
    }
}

impl<D: DataRef> WriterTo for VecZnx<D> {
    fn write_to<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<()> {
        writer.write_u64::<LittleEndian>(self.n as u64)?;
        writer.write_u64::<LittleEndian>(self.cols as u64)?;
        writer.write_u64::<LittleEndian>(self.size as u64)?;
        writer.write_u64::<LittleEndian>(self.max_size as u64)?;
        let coeff_bytes: usize = self.n * self.cols * self.size * size_of::<i64>();
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
