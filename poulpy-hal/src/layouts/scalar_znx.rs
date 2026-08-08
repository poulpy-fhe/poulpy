use std::{
    hash::{DefaultHasher, Hasher},
    marker::PhantomData,
};

use rand::seq::SliceRandom;
use rand_core::Rng;
use rand_distr::{Distribution, weighted::WeightedIndex};

use crate::{
    alloc_aligned,
    layouts::{
        Backend, Data, DataView, DataViewMut, DigestU64, FillUniform, HostDataMut, HostDataRef, ReaderFrom, ToOwnedDeep, VecZnx,
        VecZnxBackendMut, VecZnxBackendRef, VecZnxInfos, WriterTo, ZnxInfos, ZnxView, ZnxViewMut, ZnxWord, ZnxZero,
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
/// The type parameter `W` names the coefficient word (byte-layout
/// contract) of the buffer.
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

impl<D: Data, W: ZnxWord> ScalarZnx<D, W> {
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
pub struct ScalarZnx<D: Data, W: ZnxWord> {
    pub data: D,
    shape: ScalarZnxShape,
    pub _phantom: PhantomData<W>,
}

impl<D: HostDataRef, W: ZnxWord> DigestU64 for ScalarZnx<D, W> {
    fn digest_u64(&self) -> u64 {
        let mut h: DefaultHasher = DefaultHasher::new();
        h.write(self.data.as_ref());
        h.write_usize(self.n());
        h.write_usize(self.cols());
        h.finish()
    }
}

impl<D: HostDataRef, W: ZnxWord> ToOwnedDeep for ScalarZnx<D, W> {
    type Owned = ScalarZnx<Vec<u8>, W>;
    fn to_owned_deep(&self) -> Self::Owned {
        ScalarZnx {
            data: self.data.as_ref().to_vec(),
            shape: self.shape,
            _phantom: PhantomData,
        }
    }
}

impl<D: Data, W: ZnxWord> ZnxInfos for ScalarZnx<D, W> {
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

impl<D: Data, W: ZnxWord> VecZnxInfos for ScalarZnx<D, W> {
    fn cols(&self) -> usize {
        self.shape.cols()
    }
}

impl<D: Data, W: ZnxWord> DataView for ScalarZnx<D, W> {
    type D = D;
    fn data(&self) -> &Self::D {
        &self.data
    }
}

impl<D: Data, W: ZnxWord> DataViewMut for ScalarZnx<D, W> {
    fn data_mut(&mut self) -> &mut Self::D {
        &mut self.data
    }
}

impl<D: HostDataRef, W: ZnxWord> ZnxView for ScalarZnx<D, W> {
    type Scalar = W;
}

impl<D: HostDataMut, W: ZnxWord> ScalarZnx<D, W> {
    /// Fills column `col` with ternary values `{-1, 0, 1}` where each
    /// non-zero entry appears with total probability `prob` (split equally
    /// between `-1` and `+1`).
    pub fn fill_ternary_prob(&mut self, col: usize, prob: f64, source: &mut Source) {
        let choices: [W; 3] = [W::from_i64(-1), W::zero(), W::from_i64(1)];
        let weights: [f64; 3] = [prob / 2.0, 1.0 - prob, prob / 2.0];
        let dist: WeightedIndex<f64> = WeightedIndex::new(weights).unwrap();
        self.at_mut(col, 0)
            .iter_mut()
            .for_each(|x: &mut W| *x = choices[dist.sample(source)]);
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
        self.at_mut(col, 0).fill(W::zero());
        self.at_mut(col, 0)[..hw]
            .iter_mut()
            .for_each(|x: &mut W| *x = W::from_i64((((source.next_u32() & 1) as i64) << 1) - 1));
        self.at_mut(col, 0).shuffle(source);
    }

    /// Fills column `col` with binary values `{0, 1}` where each entry is `1`
    /// with probability `prob`.
    pub fn fill_binary_prob(&mut self, col: usize, prob: f64, source: &mut Source) {
        let choices: [W; 2] = [W::zero(), W::from_i64(1)];
        let weights: [f64; 2] = [1.0 - prob, prob];
        let dist: WeightedIndex<f64> = WeightedIndex::new(weights).unwrap();
        self.at_mut(col, 0)
            .iter_mut()
            .for_each(|x: &mut W| *x = choices[dist.sample(source)]);
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
        self.at_mut(col, 0).fill(W::zero());
        self.at_mut(col, 0)[..hw]
            .iter_mut()
            .for_each(|x: &mut W| *x = W::from_i64((source.next_u32() & 1) as i64));
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
        self.at_mut(col, 0).fill(W::zero());
        let max_idx: u64 = (block_size + 1) as u64;
        let mask_idx: u64 = (1 << ((u64::BITS - max_idx.leading_zeros()) as u64)) - 1;
        for block in self.at_mut(col, 0).chunks_mut(block_size) {
            let idx: usize = source.next_u64n(max_idx, mask_idx) as usize;
            if idx != block_size {
                block[idx] = W::from_i64(1);
            }
        }
    }
}

impl<D: Data, W: ZnxWord> ScalarZnx<D, W> {
    /// Returns the number of bytes required to store a `ScalarZnx` with
    /// ring degree `n` and `cols` columns: `n * cols * size_of::<W>()`.
    pub fn bytes_of(n: usize, cols: usize) -> usize {
        crate::layouts::checked_product(&[n, cols, size_of::<W>()], "ScalarZnx byte size")
    }
}

impl<W: ZnxWord> ScalarZnx<Vec<u8>, W> {
    /// Allocates a zero-initialized `ScalarZnx` aligned to [`DEFAULTALIGN`](crate::DEFAULTALIGN).
    pub(crate) fn alloc(n: usize, cols: usize) -> Self {
        let data: Vec<u8> = alloc_aligned::<u8>(Self::bytes_of(n, cols));
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
    /// Panics if the buffer length does not equal `bytes_of(n, cols)` or
    /// the buffer is not aligned to [`DEFAULTALIGN`](crate::DEFAULTALIGN).
    pub fn from_bytes(n: usize, cols: usize, bytes: impl Into<Vec<u8>>) -> Self {
        let data: Vec<u8> = bytes.into();
        assert!(data.len() == Self::bytes_of(n, cols));
        crate::assert_alignment(data.as_ptr());
        Self {
            data,
            shape: ScalarZnxShape::new(n, cols),
            _phantom: PhantomData,
        }
    }
}

impl<D: HostDataMut, W: ZnxWord> ZnxZero for ScalarZnx<D, W> {
    fn zero(&mut self) {
        self.raw_mut().fill(W::zero())
    }
    fn zero_at(&mut self, i: usize, j: usize) {
        self.at_mut(i, j).fill(W::zero());
    }
}

impl<D: HostDataMut, W: ZnxWord> FillUniform for ScalarZnx<D, W> {
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

/// Owned `ScalarZnx` backed by a `Vec<u8>`.
pub type ScalarZnxOwned<W> = ScalarZnx<Vec<u8>, W>;
/// Shared backend-native borrow of a `ScalarZnx`.
pub type ScalarZnxBackendRef<'a, B> = ScalarZnx<<B as Backend>::BufRef<'a>, <B as Backend>::ZnxWord>;
/// Mutable backend-native borrow of a `ScalarZnx`.
pub type ScalarZnxBackendMut<'a, B> = ScalarZnx<<B as Backend>::BufMut<'a>, <B as Backend>::ZnxWord>;

impl<D: Data, W: ZnxWord> ScalarZnx<D, W> {
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
pub trait ScalarZnxToBackendRef<B: Backend> {
    fn to_backend_ref(&self) -> ScalarZnxBackendRef<'_, B>;
}

impl<B: Backend> ScalarZnxToBackendRef<B> for ScalarZnx<B::OwnedBuf, B::ZnxWord> {
    fn to_backend_ref(&self) -> ScalarZnxBackendRef<'_, B> {
        ScalarZnx {
            data: B::view(&self.data),
            shape: self.shape,
            _phantom: PhantomData,
        }
    }
}

impl<'b, B: Backend + 'b> ScalarZnxToBackendRef<B> for &ScalarZnx<B::BufRef<'b>, B::ZnxWord> {
    fn to_backend_ref(&self) -> ScalarZnxBackendRef<'_, B> {
        ScalarZnx {
            data: B::view_ref(&self.data),
            shape: self.shape,
            _phantom: PhantomData,
        }
    }
}

impl<'b, B: Backend + 'b> ScalarZnxToBackendRef<B> for &mut ScalarZnx<B::BufMut<'b>, B::ZnxWord> {
    fn to_backend_ref(&self) -> ScalarZnxBackendRef<'_, B> {
        scalar_znx_backend_ref_from_mut::<B>(self)
    }
}

/// Mutably borrow a backend-owned `ScalarZnx` using the backend's native view type.
pub trait ScalarZnxToBackendMut<B: Backend> {
    fn to_backend_mut(&mut self) -> ScalarZnxBackendMut<'_, B>;
}

impl<B: Backend> ScalarZnxToBackendMut<B> for ScalarZnx<B::OwnedBuf, B::ZnxWord> {
    fn to_backend_mut(&mut self) -> ScalarZnxBackendMut<'_, B> {
        ScalarZnx {
            data: B::view_mut(&mut self.data),
            shape: self.shape,
            _phantom: PhantomData,
        }
    }
}

impl<'b, B: Backend + 'b> ScalarZnxToBackendMut<B> for &mut ScalarZnx<B::BufMut<'b>, B::ZnxWord> {
    fn to_backend_mut(&mut self) -> ScalarZnxBackendMut<'_, B> {
        scalar_znx_backend_mut_from_mut::<B>(self)
    }
}

fn scalar_znx_backend_ref_from_mut<'a, 'b, B: Backend + 'b>(
    scalar: &'a ScalarZnx<B::BufMut<'b>, B::ZnxWord>,
) -> ScalarZnxBackendRef<'a, B> {
    ScalarZnx {
        data: B::view_ref_mut(&scalar.data),
        shape: scalar.shape,
        _phantom: PhantomData,
    }
}

fn scalar_znx_backend_mut_from_mut<'a, 'b, B: Backend + 'b>(
    scalar: &'a mut ScalarZnx<B::BufMut<'b>, B::ZnxWord>,
) -> ScalarZnxBackendMut<'a, B> {
    ScalarZnx {
        data: B::view_mut_ref(&mut scalar.data),
        shape: scalar.shape,
        _phantom: PhantomData,
    }
}

impl<D: HostDataRef, W: ZnxWord> ScalarZnx<D, W> {
    /// Borrow a host-visible `ScalarZnx` as a shared byte-slice view.
    pub fn to_ref(&self) -> ScalarZnx<&[u8], W> {
        ScalarZnx {
            data: self.data.as_ref(),
            shape: self.shape,
            _phantom: PhantomData,
        }
    }
}

impl<D: HostDataRef, W: ZnxWord> ScalarZnx<D, W> {
    /// Views this `ScalarZnx` as a [`VecZnx`] with `size == 1`.
    pub fn as_vec_znx(&self) -> VecZnx<&[u8], W> {
        VecZnx::from_data(self.data.as_ref(), self.n(), self.cols(), 1)
    }
}

/// Views a backend-owned `ScalarZnx` as a backend-native [`VecZnx`] with `size == 1`.
pub trait ScalarZnxAsVecZnxBackendRef<B: Backend> {
    fn as_vec_znx_backend(&self) -> VecZnx<B::BufRef<'_>, B::ZnxWord>;
}

impl<B: Backend> ScalarZnxAsVecZnxBackendRef<B> for ScalarZnx<B::OwnedBuf, B::ZnxWord> {
    fn as_vec_znx_backend(&self) -> VecZnx<B::BufRef<'_>, B::ZnxWord> {
        VecZnx::from_data(B::view(&self.data), self.n(), self.cols(), 1)
    }
}

pub fn scalar_znx_as_vec_znx_backend_ref_from_ref<'a, 'b, B: Backend + 'b>(
    scalar: &'a ScalarZnx<B::BufRef<'b>, B::ZnxWord>,
) -> VecZnxBackendRef<'a, B> {
    VecZnx::from_data(B::view_ref(&scalar.data), scalar.n(), scalar.cols(), 1)
}

pub fn scalar_znx_as_vec_znx_backend_ref_from_mut<'a, 'b, B: Backend + 'b>(
    scalar: &'a ScalarZnx<B::BufMut<'b>, B::ZnxWord>,
) -> VecZnxBackendRef<'a, B> {
    VecZnx::from_data(B::view_ref_mut(&scalar.data), scalar.n(), scalar.cols(), 1)
}

impl<D: HostDataMut, W: ZnxWord> ScalarZnx<D, W> {
    /// Mutably views this `ScalarZnx` as a [`VecZnx`] with `size == 1`.
    pub fn as_vec_znx_mut(&mut self) -> VecZnx<&mut [u8], W> {
        let shape = self.shape();
        VecZnx::from_data(self.data.as_mut(), shape.n(), shape.cols(), 1)
    }
}

/// Mutably views a backend-owned `ScalarZnx` as a backend-native [`VecZnx`] with `size == 1`.
pub trait ScalarZnxAsVecZnxBackendMut<B: Backend> {
    fn as_vec_znx_backend_mut(&mut self) -> VecZnx<B::BufMut<'_>, B::ZnxWord>;
}

impl<B: Backend> ScalarZnxAsVecZnxBackendMut<B> for ScalarZnx<B::OwnedBuf, B::ZnxWord> {
    fn as_vec_znx_backend_mut(&mut self) -> VecZnx<B::BufMut<'_>, B::ZnxWord> {
        let shape = self.shape();
        VecZnx::from_data(B::view_mut(&mut self.data), shape.n(), shape.cols(), 1)
    }
}

pub fn scalar_znx_as_vec_znx_backend_mut_from_mut<'a, 'b, B: Backend + 'b>(
    scalar: &'a mut ScalarZnx<B::BufMut<'b>, B::ZnxWord>,
) -> VecZnxBackendMut<'a, B> {
    let shape = scalar.shape();
    VecZnx::from_data(B::view_mut_ref(&mut scalar.data), shape.n(), shape.cols(), 1)
}

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};

impl<D: HostDataMut, W: ZnxWord> ReaderFrom for ScalarZnx<D, W> {
    fn read_from<R: std::io::Read>(&mut self, reader: &mut R) -> std::io::Result<()> {
        let new_n: usize = reader.read_u64::<LittleEndian>()? as usize;
        let new_cols: usize = reader.read_u64::<LittleEndian>()? as usize;
        let len: usize = reader.read_u64::<LittleEndian>()? as usize;

        let expected_len: usize =
            crate::layouts::checked_product(&[new_n, new_cols, size_of::<W>()], "ScalarZnx serialized byte size");
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

impl<D: HostDataRef, W: ZnxWord> WriterTo for ScalarZnx<D, W> {
    fn write_to<Wr: std::io::Write>(&self, writer: &mut Wr) -> std::io::Result<()> {
        writer.write_u64::<LittleEndian>(self.n() as u64)?;
        writer.write_u64::<LittleEndian>(self.cols() as u64)?;
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
        writer.write_u64::<LittleEndian>(coeff_bytes as u64)?;
        writer.write_all(&buf[..coeff_bytes])?;
        Ok(())
    }
}
