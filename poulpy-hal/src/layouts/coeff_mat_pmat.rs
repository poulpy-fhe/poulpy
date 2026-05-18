use std::{
    hash::{DefaultHasher, Hasher},
    marker::PhantomData,
};

use crate::layouts::{Backend, Data, DataView, DataViewMut, DigestU64, HostDataRef};

#[repr(C)]
#[derive(PartialEq, Eq, Clone, Copy, Hash, Debug, Default)]
pub struct CoeffMatPMatShape {
    n: usize,
    size: usize,
    rows: usize,
    cols_in: usize,
    cols_out: usize,
}

impl CoeffMatPMatShape {
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

/// Coefficient-domain prepared matrix for coefficient-matrix products.
///
/// This layout borrows the useful paired-output-column/block packing shape from
/// `VmpPMat`, but the logical entries are coefficients inside a small
/// polynomial, not small polynomials. It deliberately does not switch
/// coefficients to the `X^N + 1` DFT/NTT domain.
///
/// Logical column `c = limb * cols_out + out_col` is packed in pairs of output
/// columns and blocks of eight coefficient lanes.
#[repr(C)]
#[derive(PartialEq, Eq, Hash)]
pub struct CoeffMatPMat<D: Data, B: Backend> {
    data: D,
    shape: CoeffMatPMatShape,
    _phantom: PhantomData<B>,
}

impl<D: HostDataRef, B: Backend> DigestU64 for CoeffMatPMat<D, B> {
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

impl<D: Data, B: Backend> DataView for CoeffMatPMat<D, B> {
    type D = D;
    fn data(&self) -> &Self::D {
        &self.data
    }
}

impl<D: Data, B: Backend> DataViewMut for CoeffMatPMat<D, B> {
    fn data_mut(&mut self) -> &mut Self::D {
        &mut self.data
    }
}

impl<D: Data, B: Backend> CoeffMatPMat<D, B> {
    pub fn shape(&self) -> CoeffMatPMatShape {
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

    pub fn cols_in(&self) -> usize {
        self.shape.cols_in()
    }

    pub fn cols_out(&self) -> usize {
        self.shape.cols_out()
    }

    pub fn poly_count(&self) -> usize {
        self.rows() * self.cols_in() * self.cols_out() * self.size()
    }

    pub fn bytes_of(n: usize, rows: usize, cols_in: usize, cols_out: usize, size: usize) -> usize {
        n * rows * cols_in * cols_out * size * size_of::<i64>()
    }

    pub fn from_data(data: D, n: usize, rows: usize, cols_in: usize, cols_out: usize, size: usize) -> Self {
        Self {
            data,
            shape: CoeffMatPMatShape::new(n, rows, cols_in, cols_out, size),
            _phantom: PhantomData,
        }
    }

    pub fn packed_offset(&self, row: usize, col: usize, coeff: usize) -> usize {
        assert!(row < self.rows(), "CoeffMatPMat row out of bounds");
        assert!(coeff < self.n(), "CoeffMatPMat coefficient out of bounds");
        let ncols = self.cols_out() * self.size();
        assert!(col < ncols, "CoeffMatPMat column out of bounds");
        let nrows = self.rows() * self.cols_in();
        let block = coeff / 8;
        let lane = coeff % 8;
        let block_stride = nrows * ncols * 8;
        let row_i = row;

        if col == ncols - 1 && !ncols.is_multiple_of(2) {
            col * nrows * 8 + row_i * 8 + block * block_stride + lane
        } else {
            (col / 2) * (nrows * 16) + row_i * 16 + (col % 2) * 8 + block * block_stride + lane
        }
    }
}

impl<D: HostDataRef, B: Backend> CoeffMatPMat<D, B> {
    pub fn raw(&self) -> &[i64] {
        unsafe { std::slice::from_raw_parts(self.data.as_ref().as_ptr() as *const i64, self.n() * self.poly_count()) }
    }

    pub fn at_packed(&self, row: usize, col: usize, coeff: usize) -> i64 {
        self.raw()[self.packed_offset(row, col, coeff)]
    }
}

impl<D: crate::layouts::HostDataMut, B: Backend> CoeffMatPMat<D, B> {
    pub fn raw_mut(&mut self) -> &mut [i64] {
        unsafe { std::slice::from_raw_parts_mut(self.data.as_mut().as_mut_ptr() as *mut i64, self.n() * self.poly_count()) }
    }

    pub fn set_packed(&mut self, row: usize, col: usize, coeff: usize, value: i64) {
        let offset = self.packed_offset(row, col, coeff);
        self.raw_mut()[offset] = value;
    }
}

impl<B: Backend> CoeffMatPMat<<B as Backend>::OwnedBuf, B> {
    pub fn alloc(n: usize, rows: usize, cols_in: usize, cols_out: usize, size: usize) -> Self {
        let data = B::alloc_zeroed_bytes(Self::bytes_of(n, rows, cols_in, cols_out, size));
        Self {
            data,
            shape: CoeffMatPMatShape::new(n, rows, cols_in, cols_out, size),
            _phantom: PhantomData,
        }
    }
}

pub type CoeffMatPMatOwned<B> = CoeffMatPMat<<B as Backend>::OwnedBuf, B>;
pub type CoeffMatPMatBackendRef<'a, B> = CoeffMatPMat<<B as Backend>::BufRef<'a>, B>;
pub type CoeffMatPMatBackendMut<'a, B> = CoeffMatPMat<<B as Backend>::BufMut<'a>, B>;

pub trait CoeffMatPMatToBackendRef<B: Backend> {
    fn to_backend_ref(&self) -> CoeffMatPMatBackendRef<'_, B>;
}

impl<B: Backend> CoeffMatPMatToBackendRef<B> for CoeffMatPMat<B::OwnedBuf, B> {
    fn to_backend_ref(&self) -> CoeffMatPMatBackendRef<'_, B> {
        CoeffMatPMat {
            data: B::view(&self.data),
            shape: self.shape,
            _phantom: PhantomData,
        }
    }
}

impl<'b, B: Backend + 'b> CoeffMatPMat<B::BufMut<'b>, B> {
    pub fn to_backend_ref(&self) -> CoeffMatPMatBackendRef<'_, B> {
        CoeffMatPMat {
            data: B::view_ref_mut(&self.data),
            shape: self.shape,
            _phantom: PhantomData,
        }
    }
}

pub trait CoeffMatPMatToBackendMut<B: Backend>: CoeffMatPMatToBackendRef<B> {
    fn to_backend_mut(&mut self) -> CoeffMatPMatBackendMut<'_, B>;
}

impl<B: Backend> CoeffMatPMatToBackendMut<B> for CoeffMatPMat<B::OwnedBuf, B> {
    fn to_backend_mut(&mut self) -> CoeffMatPMatBackendMut<'_, B> {
        CoeffMatPMat {
            data: B::view_mut(&mut self.data),
            shape: self.shape,
            _phantom: PhantomData,
        }
    }
}
