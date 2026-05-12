use std::hash::{DefaultHasher, Hasher};

use rand::seq::SliceRandom;
use rand_core::Rng;
use rand_distr::{Distribution, weighted::WeightedIndex};

use crate::{
    alloc_aligned,
    layouts::{
        Backend, Data, DataView, DataViewMut, DigestU64, FillUniform, HostDataMut, HostDataRef, ReaderFrom, ToOwnedDeep, VecZnx,
        VecZnxBackendMut, VecZnxBackendRef, WriterTo, ZnxInfos, ZnxView, ZnxViewMut, ZnxZero,
    },
    source::Source,
};

/// A single-limb polynomial vector in `Z[X]/(X^N + 1)`.
///
/// `ScalarZnx` is a specialization of [`VecZnx`] with exactly one limb
/// (`size == 1`). It is the primary type for plaintext polynomials,
/// secret keys, and other single-precision ring elements.
///
/// The type parameter `D` controls ownership: `Vec<u8>` for owned,
/// `&[u8]` for shared borrows, `&mut [u8]` for mutable borrows.
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

impl<D: Data> ScalarZnx<D> {
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

#[repr(C)]
#[derive(PartialEq, Eq, Debug, Clone, Hash)]
pub struct ScalarZnx<D: Data> {
    pub data: D,
    shape: ScalarZnxShape,
}

impl<D: HostDataRef> DigestU64 for ScalarZnx<D> {
    fn digest_u64(&self) -> u64 {
        let mut h: DefaultHasher = DefaultHasher::new();
        h.write(self.data.as_ref());
        h.write_usize(self.n());
        h.write_usize(self.cols());
        h.finish()
    }
}

impl<D: HostDataRef> ToOwnedDeep for ScalarZnx<D> {
    type Owned = ScalarZnx<Vec<u8>>;
    fn to_owned_deep(&self) -> Self::Owned {
        ScalarZnx {
            data: self.data.as_ref().to_vec(),
            shape: self.shape,
        }
    }
}

impl<D: Data> ZnxInfos for ScalarZnx<D> {
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
        1
    }
}

impl<D: Data> DataView for ScalarZnx<D> {
    type D = D;
    fn data(&self) -> &Self::D {
        &self.data
    }
}

impl<D: Data> DataViewMut for ScalarZnx<D> {
    fn data_mut(&mut self) -> &mut Self::D {
        &mut self.data
    }
}

impl<D: HostDataRef> ZnxView for ScalarZnx<D> {
    type Scalar = i64;
}

impl<D: HostDataMut> ScalarZnx<D> {
    /// Fills column `col` with ternary values `{-1, 0, 1}` where each
    /// non-zero entry appears with total probability `prob` (split equally
    /// between `-1` and `+1`).
    pub fn fill_ternary_prob(&mut self, col: usize, prob: f64, source: &mut Source) {
        let choices: [i64; 3] = [-1, 0, 1];
        let weights: [f64; 3] = [prob / 2.0, 1.0 - prob, prob / 2.0];
        let dist: WeightedIndex<f64> = WeightedIndex::new(weights).unwrap();
        self.at_mut(col, 0)
            .iter_mut()
            .for_each(|x: &mut i64| *x = choices[dist.sample(source)]);
    }

    /// Fills column `col` with exactly `hw` non-zero ternary values `{-1, +1}`
    /// at uniformly random positions; the remaining `N - hw` coefficients are zero.
    ///
    /// # Panics
    ///
    /// Panics if `hw > N`.
    pub fn fill_ternary_hw(&mut self, col: usize, hw: usize, source: &mut Source) {
        assert!(hw <= self.n());
        // Zero-initialize before setting non-zero entries, since shuffle will
        // mix positions and we need indices hw..n to be zero.
        self.at_mut(col, 0).fill(0);
        self.at_mut(col, 0)[..hw]
            .iter_mut()
            .for_each(|x: &mut i64| *x = (((source.next_u32() & 1) as i64) << 1) - 1);
        self.at_mut(col, 0).shuffle(source);
    }

    /// Fills column `col` with binary values `{0, 1}` where each entry is `1`
    /// with probability `prob`.
    pub fn fill_binary_prob(&mut self, col: usize, prob: f64, source: &mut Source) {
        let choices: [i64; 2] = [0, 1];
        let weights: [f64; 2] = [1.0 - prob, prob];
        let dist: WeightedIndex<f64> = WeightedIndex::new(weights).unwrap();
        self.at_mut(col, 0)
            .iter_mut()
            .for_each(|x: &mut i64| *x = choices[dist.sample(source)]);
    }

    /// Fills column `col` with exactly `hw` ones at uniformly random positions;
    /// the remaining `N - hw` coefficients are zero.
    ///
    /// # Panics
    ///
    /// Panics if `hw > N`.
    pub fn fill_binary_hw(&mut self, col: usize, hw: usize, source: &mut Source) {
        assert!(hw <= self.n());
        // Zero-initialize before setting non-zero entries, since shuffle will
        // mix positions and we need indices hw..n to be zero.
        self.at_mut(col, 0).fill(0);
        self.at_mut(col, 0)[..hw]
            .iter_mut()
            .for_each(|x: &mut i64| *x = (source.next_u32() & 1) as i64);
        self.at_mut(col, 0).shuffle(source);
    }

    /// Fills column `col` with a block-sparse binary pattern: the polynomial is
    /// partitioned into blocks of `block_size` coefficients, and each block
    /// independently receives at most one `1` at a uniformly random position
    /// (or no `1` at all with probability `1 / (block_size + 1)`).
    ///
    /// # Panics
    ///
    /// Panics if `N` is not a multiple of `block_size`.
    pub fn fill_binary_block(&mut self, col: usize, block_size: usize, source: &mut Source) {
        assert!(self.n().is_multiple_of(block_size));
        // Zero-initialize: each block gets at most one non-zero entry.
        self.at_mut(col, 0).fill(0);
        let max_idx: u64 = (block_size + 1) as u64;
        let mask_idx: u64 = (1 << ((u64::BITS - max_idx.leading_zeros()) as u64)) - 1;
        for block in self.at_mut(col, 0).chunks_mut(block_size) {
            let idx: usize = source.next_u64n(max_idx, mask_idx) as usize;
            if idx != block_size {
                block[idx] = 1;
            }
        }
    }
}

impl ScalarZnx<Vec<u8>> {
    /// Returns the number of bytes required to store a `ScalarZnx` with
    /// ring degree `n` and `cols` columns: `n * cols * 8`.
    pub fn bytes_of(n: usize, cols: usize) -> usize {
        n * cols * size_of::<i64>()
    }

    /// Allocates a zero-initialized `ScalarZnx` aligned to [`DEFAULTALIGN`](crate::DEFAULTALIGN).
    pub(crate) fn alloc(n: usize, cols: usize) -> Self {
        let data: Vec<u8> = alloc_aligned::<u8>(Self::bytes_of(n, cols));
        Self {
            data,
            shape: ScalarZnxShape::new(n, cols),
        }
    }

    /// Wraps an existing byte buffer into a `ScalarZnx`.
    ///
    /// # Panics
    ///
    /// Panics if the buffer length does not equal `bytes_of(n, cols)` or
    /// the buffer is not aligned to [`DEFAULTALIGN`](crate::DEFAULTALIGN).
    pub fn from_bytes(n: usize, cols: usize, bytes: impl Into<Vec<u8>>) -> Self {
        let data: Vec<u8> = bytes.into();
        assert!(data.len() == Self::bytes_of(n, cols));
        crate::assert_alignment(data.as_ptr());
        Self {
            data,
            shape: ScalarZnxShape::new(n, cols),
        }
    }
}

impl<D: HostDataMut> ZnxZero for ScalarZnx<D> {
    fn zero(&mut self) {
        self.raw_mut().fill(0)
    }
    fn zero_at(&mut self, i: usize, j: usize) {
        self.at_mut(i, j).fill(0);
    }
}

impl<D: HostDataMut> FillUniform for ScalarZnx<D> {
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

/// Owned `ScalarZnx` backed by a `Vec<u8>`.
pub type ScalarZnxOwned = ScalarZnx<Vec<u8>>;
/// Shared backend-native borrow of a `ScalarZnx`.
pub type ScalarZnxBackendRef<'a, B> = ScalarZnx<<B as Backend>::BufRef<'a>>;
/// Mutable backend-native borrow of a `ScalarZnx`.
pub type ScalarZnxBackendMut<'a, B> = ScalarZnx<<B as Backend>::BufMut<'a>>;

impl<D: Data> ScalarZnx<D> {
    /// Constructs a `ScalarZnx` from raw parts without validation.
    pub fn from_data(data: D, n: usize, cols: usize) -> Self {
        Self {
            data,
            shape: ScalarZnxShape::new(n, cols),
        }
    }
}

/// Borrow a backend-owned `ScalarZnx` using the backend's native view type.
pub trait ScalarZnxToBackendRef<B: Backend> {
    fn to_backend_ref(&self) -> ScalarZnxBackendRef<'_, B>;
}

impl<B: Backend> ScalarZnxToBackendRef<B> for ScalarZnx<B::OwnedBuf> {
    fn to_backend_ref(&self) -> ScalarZnxBackendRef<'_, B> {
        ScalarZnx {
            data: B::view(&self.data),
            shape: self.shape,
        }
    }
}

impl<'b, B: Backend + 'b> ScalarZnxToBackendRef<B> for &ScalarZnx<B::BufRef<'b>> {
    fn to_backend_ref(&self) -> ScalarZnxBackendRef<'_, B> {
        ScalarZnx {
            data: B::view_ref(&self.data),
            shape: self.shape,
        }
    }
}

impl<'b, B: Backend + 'b> ScalarZnxToBackendRef<B> for &mut ScalarZnx<B::BufMut<'b>> {
    fn to_backend_ref(&self) -> ScalarZnxBackendRef<'_, B> {
        scalar_znx_backend_ref_from_mut::<B>(self)
    }
}

/// Mutably borrow a backend-owned `ScalarZnx` using the backend's native view type.
pub trait ScalarZnxToBackendMut<B: Backend> {
    fn to_backend_mut(&mut self) -> ScalarZnxBackendMut<'_, B>;
}

impl<B: Backend> ScalarZnxToBackendMut<B> for ScalarZnx<B::OwnedBuf> {
    fn to_backend_mut(&mut self) -> ScalarZnxBackendMut<'_, B> {
        ScalarZnx {
            data: B::view_mut(&mut self.data),
            shape: self.shape,
        }
    }
}

impl<'b, B: Backend + 'b> ScalarZnxToBackendMut<B> for &mut ScalarZnx<B::BufMut<'b>> {
    fn to_backend_mut(&mut self) -> ScalarZnxBackendMut<'_, B> {
        scalar_znx_backend_mut_from_mut::<B>(self)
    }
}

fn scalar_znx_backend_ref_from_mut<'a, 'b, B: Backend + 'b>(scalar: &'a ScalarZnx<B::BufMut<'b>>) -> ScalarZnxBackendRef<'a, B> {
    ScalarZnx {
        data: B::view_ref_mut(&scalar.data),
        shape: scalar.shape,
    }
}

fn scalar_znx_backend_mut_from_mut<'a, 'b, B: Backend + 'b>(
    scalar: &'a mut ScalarZnx<B::BufMut<'b>>,
) -> ScalarZnxBackendMut<'a, B> {
    ScalarZnx {
        data: B::view_mut_ref(&mut scalar.data),
        shape: scalar.shape,
    }
}

impl<D: HostDataRef> ScalarZnx<D> {
    /// Borrow a host-visible `ScalarZnx` as a shared byte-slice view.
    pub fn to_ref(&self) -> ScalarZnx<&[u8]> {
        ScalarZnx {
            data: self.data.as_ref(),
            shape: self.shape,
        }
    }
}

impl<D: HostDataRef> ScalarZnx<D> {
    /// Views this `ScalarZnx` as a [`VecZnx`] with `size == 1`.
    pub fn as_vec_znx(&self) -> VecZnx<&[u8]> {
        VecZnx::from_data(self.data.as_ref(), self.n(), self.cols(), 1)
    }
}

/// Views a backend-owned `ScalarZnx` as a backend-native [`VecZnx`] with `size == 1`.
pub trait ScalarZnxAsVecZnxBackendRef<B: Backend> {
    fn as_vec_znx_backend(&self) -> VecZnx<B::BufRef<'_>>;
}

impl<B: Backend> ScalarZnxAsVecZnxBackendRef<B> for ScalarZnx<B::OwnedBuf> {
    fn as_vec_znx_backend(&self) -> VecZnx<B::BufRef<'_>> {
        VecZnx::from_data(B::view(&self.data), self.n(), self.cols(), 1)
    }
}

pub fn scalar_znx_as_vec_znx_backend_ref_from_ref<'a, 'b, B: Backend + 'b>(
    scalar: &'a ScalarZnx<B::BufRef<'b>>,
) -> VecZnxBackendRef<'a, B> {
    VecZnx::from_data(B::view_ref(&scalar.data), scalar.n(), scalar.cols(), 1)
}

pub fn scalar_znx_as_vec_znx_backend_ref_from_mut<'a, 'b, B: Backend + 'b>(
    scalar: &'a ScalarZnx<B::BufMut<'b>>,
) -> VecZnxBackendRef<'a, B> {
    VecZnx::from_data(B::view_ref_mut(&scalar.data), scalar.n(), scalar.cols(), 1)
}

impl<D: HostDataMut> ScalarZnx<D> {
    /// Mutably views this `ScalarZnx` as a [`VecZnx`] with `size == 1`.
    pub fn as_vec_znx_mut(&mut self) -> VecZnx<&mut [u8]> {
        let shape = self.shape();
        VecZnx::from_data(self.data.as_mut(), shape.n(), shape.cols(), 1)
    }
}

/// Mutably views a backend-owned `ScalarZnx` as a backend-native [`VecZnx`] with `size == 1`.
pub trait ScalarZnxAsVecZnxBackendMut<B: Backend> {
    fn as_vec_znx_backend_mut(&mut self) -> VecZnx<B::BufMut<'_>>;
}

impl<B: Backend> ScalarZnxAsVecZnxBackendMut<B> for ScalarZnx<B::OwnedBuf> {
    fn as_vec_znx_backend_mut(&mut self) -> VecZnx<B::BufMut<'_>> {
        let shape = self.shape();
        VecZnx::from_data(B::view_mut(&mut self.data), shape.n(), shape.cols(), 1)
    }
}

pub fn scalar_znx_as_vec_znx_backend_mut_from_mut<'a, 'b, B: Backend + 'b>(
    scalar: &'a mut ScalarZnx<B::BufMut<'b>>,
) -> VecZnxBackendMut<'a, B> {
    let shape = scalar.shape();
    VecZnx::from_data(B::view_mut_ref(&mut scalar.data), shape.n(), shape.cols(), 1)
}

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};

impl<D: HostDataMut> ReaderFrom for ScalarZnx<D> {
    fn read_from<R: std::io::Read>(&mut self, reader: &mut R) -> std::io::Result<()> {
        let new_n: usize = reader.read_u64::<LittleEndian>()? as usize;
        let new_cols: usize = reader.read_u64::<LittleEndian>()? as usize;
        let len: usize = reader.read_u64::<LittleEndian>()? as usize;

        let expected_len: usize = new_n * new_cols * size_of::<i64>();
        if expected_len != len {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("ScalarZnx metadata inconsistent: n={new_n} * cols={new_cols} * 8 = {expected_len} != data len={len}"),
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

impl<D: HostDataRef> WriterTo for ScalarZnx<D> {
    fn write_to<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<()> {
        writer.write_u64::<LittleEndian>(self.n() as u64)?;
        writer.write_u64::<LittleEndian>(self.cols() as u64)?;
        let coeff_bytes = self.n() * self.cols() * size_of::<i64>();
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
        writer.write_u64::<LittleEndian>(coeff_bytes as u64)?;
        writer.write_all(&buf[..coeff_bytes])?;
        Ok(())
    }
}
