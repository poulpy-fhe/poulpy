use std::{
    hash::{DefaultHasher, Hasher},
    marker::PhantomData,
};

use crate::layouts::{Backend, Data, DataView, DataViewMut, DigestU64, HostDataRef, ZnxInfos, ZnxView};

#[repr(C)]
#[derive(PartialEq, Eq, Clone, Copy, Hash, Debug, Default)]
pub struct VmpPMatShape {
    n: usize,
    size: usize,
    rows: usize,
    cols_in: usize,
    cols_out: usize,
}

impl VmpPMatShape {
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

/// Prepared (DFT-domain) polynomial matrix for vector-matrix products.
///
/// A `VmpPMat` stores a matrix of `rows * cols_in` entries, where each
/// entry is a [`VecZnxDft`](crate::layouts::VecZnxDft) of `cols_out`
/// columns and `size` limbs, all in the backend's prepared representation.
///
/// Used as the right operand in
/// [`VmpApplyDftToDft`](crate::api::VmpApplyDftToDft). Create via
/// [`VmpPrepare`](crate::api::VmpPrepare) from a coefficient-domain
/// [`MatZnx`](crate::layouts::MatZnx).
///
/// Ring degree `n` is always a power of two, so each prepared polynomial's DFT
/// coefficient count matches vector lane widths relative to buffer alignment.
#[repr(C)]
#[derive(PartialEq, Eq, Hash)]
pub struct VmpPMat<D: Data, B: Backend> {
    data: D,
    shape: VmpPMatShape,
    _phantom: PhantomData<B>,
}

impl<D: HostDataRef, B: Backend> DigestU64 for VmpPMat<D, B> {
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

impl<D: HostDataRef, B: Backend> ZnxView for VmpPMat<D, B> {
    type Scalar = B::ScalarPrep;
}

impl<D: Data, B: Backend> ZnxInfos for VmpPMat<D, B> {
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
        self.rows() * self.cols_in() * self.size() * self.cols_out()
    }
}

impl<D: Data, B: Backend> DataView for VmpPMat<D, B> {
    type D = D;
    fn data(&self) -> &Self::D {
        &self.data
    }
}

impl<D: Data, B: Backend> DataViewMut for VmpPMat<D, B> {
    fn data_mut(&mut self) -> &mut Self::D {
        &mut self.data
    }
}

impl<D: Data, B: Backend> VmpPMat<D, B> {
    pub fn shape(&self) -> VmpPMatShape {
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

    /// Returns the number of input columns.
    pub fn cols_in(&self) -> usize {
        self.shape.cols_in()
    }

    /// Returns the number of output columns.
    pub fn cols_out(&self) -> usize {
        self.shape.cols_out()
    }
}

impl<B: Backend> VmpPMat<<B as Backend>::OwnedBuf, B> {
    pub fn alloc(n: usize, rows: usize, cols_in: usize, cols_out: usize, size: usize) -> Self {
        let data: <B as Backend>::OwnedBuf = B::alloc_zeroed_bytes(B::bytes_of_vmp_pmat(n, rows, cols_in, cols_out, size));
        Self {
            data,
            shape: VmpPMatShape::new(n, rows, cols_in, cols_out, size),
            _phantom: PhantomData,
        }
    }
}

/// Owned `VmpPMat` backed by a backend-owned buffer.
pub type VmpPMatOwned<B> = VmpPMat<<B as Backend>::OwnedBuf, B>;
/// Immutably borrowed `VmpPMat`.
pub type VmpPMatRef<'a, B> = VmpPMat<&'a [u8], B>;
/// Shared backend-native borrow of a `VmpPMat`.
pub type VmpPMatBackendRef<'a, B> = VmpPMat<<B as Backend>::BufRef<'a>, B>;
/// Mutable backend-native borrow of a `VmpPMat`.
pub type VmpPMatBackendMut<'a, B> = VmpPMat<<B as Backend>::BufMut<'a>, B>;

/// Reborrow an immutable backend-native `VmpPMat` view as a shared backend-native view.
pub fn vmp_pmat_backend_ref_from_ref<'a, 'b, B: Backend + 'b>(pmat: &'a VmpPMat<B::BufRef<'b>, B>) -> VmpPMatBackendRef<'a, B> {
    VmpPMat {
        data: B::view_ref(&pmat.data),
        shape: pmat.shape,
        _phantom: PhantomData,
    }
}

/// Reborrow a mutable backend-native `VmpPMat` view as a shared backend-native view.
pub fn vmp_pmat_backend_ref_from_mut<'a, B: Backend>(pmat: &'a VmpPMatBackendMut<'a, B>) -> VmpPMatBackendRef<'a, B> {
    VmpPMat {
        data: B::view_ref_mut(&pmat.data),
        shape: pmat.shape,
        _phantom: PhantomData,
    }
}

pub fn vmp_pmat_backend_mut_from_mut<'a, 'b, B: Backend + 'b>(
    pmat: &'a mut VmpPMatBackendMut<'b, B>,
) -> VmpPMatBackendMut<'a, B> {
    VmpPMat {
        data: B::view_mut_ref(&mut pmat.data),
        shape: pmat.shape,
        _phantom: PhantomData,
    }
}

/// Borrow a backend-owned `VmpPMat` using the backend's native view type.
pub trait VmpPMatToBackendRef<B: Backend> {
    fn to_backend_ref(&self) -> VmpPMatBackendRef<'_, B>;
}

impl<B: Backend> VmpPMatToBackendRef<B> for VmpPMat<B::OwnedBuf, B> {
    fn to_backend_ref(&self) -> VmpPMatBackendRef<'_, B> {
        VmpPMat {
            data: B::view(&self.data),
            shape: self.shape,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<'b, B: Backend + 'b> VmpPMatToBackendRef<B> for &VmpPMat<B::BufRef<'b>, B> {
    fn to_backend_ref(&self) -> VmpPMatBackendRef<'_, B> {
        VmpPMat {
            data: B::view_ref(&self.data),
            shape: self.shape,
            _phantom: PhantomData,
        }
    }
}

/// Reborrow an already backend-borrowed `VmpPMat` as a shared backend-native view.
pub trait VmpPMatReborrowBackendRef<B: Backend> {
    fn reborrow_backend_ref(&self) -> VmpPMatBackendRef<'_, B>;
}

impl<'b, B: Backend + 'b> VmpPMatReborrowBackendRef<B> for VmpPMat<B::BufMut<'b>, B> {
    fn reborrow_backend_ref(&self) -> VmpPMatBackendRef<'_, B> {
        VmpPMat {
            data: B::view_ref_mut(&self.data),
            shape: self.shape,
            _phantom: std::marker::PhantomData,
        }
    }
}

/// Mutably borrow a backend-owned `VmpPMat` using the backend's native view type.
pub trait VmpPMatToBackendMut<B: Backend> {
    fn to_backend_mut(&mut self) -> VmpPMatBackendMut<'_, B>;
}

impl<B: Backend> VmpPMatToBackendMut<B> for VmpPMat<B::OwnedBuf, B> {
    fn to_backend_mut(&mut self) -> VmpPMatBackendMut<'_, B> {
        VmpPMat {
            data: B::view_mut(&mut self.data),
            shape: self.shape,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<'b, B: Backend + 'b> VmpPMatToBackendMut<B> for &mut VmpPMat<B::BufMut<'b>, B> {
    fn to_backend_mut(&mut self) -> VmpPMatBackendMut<'_, B> {
        vmp_pmat_backend_mut_from_mut::<B>(self)
    }
}

/// Reborrow an already backend-borrowed `VmpPMat` as a mutable backend-native view.
pub trait VmpPMatReborrowBackendMut<B: Backend> {
    fn reborrow_backend_mut(&mut self) -> VmpPMatBackendMut<'_, B>;
}

impl<'b, B: Backend + 'b> VmpPMatReborrowBackendMut<B> for VmpPMat<B::BufMut<'b>, B> {
    fn reborrow_backend_mut(&mut self) -> VmpPMatBackendMut<'_, B> {
        vmp_pmat_backend_mut_from_mut::<B>(self)
    }
}

impl<D: Data, B: Backend> VmpPMat<D, B> {
    pub fn from_data(data: D, n: usize, rows: usize, cols_in: usize, cols_out: usize, size: usize) -> Self {
        Self {
            data,
            shape: VmpPMatShape::new(n, rows, cols_in, cols_out, size),
            _phantom: PhantomData,
        }
    }
}
