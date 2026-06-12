use std::marker::PhantomData;

use crate::layouts::{Backend, Data, DataView, DataViewMut, HostDataRef, ZnxInfos, ZnxView};

#[repr(C)]
#[derive(PartialEq, Eq, Clone, Copy, Hash, Debug, Default)]
pub struct CnvPVecShape {
    n: usize,
    size: usize,
    cols: usize,
}

impl CnvPVecShape {
    pub const fn new(n: usize, cols: usize, size: usize) -> Self {
        Self { n, size, cols }
    }

    pub const fn n(self) -> usize {
        self.n
    }

    pub const fn size(self) -> usize {
        self.size
    }

    pub const fn cols(self) -> usize {
        self.cols
    }
}

/// Prepared right operand for bivariate convolution.
///
/// Holds a polynomial vector in the backend's prepared representation,
/// ready to be used as the right operand of
/// [`Convolution::cnv_apply_dft`](crate::api::Convolution::cnv_apply_dft).
/// Created via [`Convolution::cnv_prepare_right`](crate::api::Convolution::cnv_prepare_right).
pub struct CnvPVecR<D: Data, BE: Backend> {
    data: D,
    shape: CnvPVecShape,
    _phantom: PhantomData<BE>,
}

impl<D: Data, BE: Backend> ZnxInfos for CnvPVecR<D, BE> {
    fn cols(&self) -> usize {
        self.shape.cols()
    }

    fn n(&self) -> usize {
        self.shape.n()
    }

    fn rows(&self) -> usize {
        1
    }

    fn size(&self) -> usize {
        self.shape.size()
    }
}

impl<D: Data, BE: Backend> DataView for CnvPVecR<D, BE> {
    type D = D;
    fn data(&self) -> &Self::D {
        &self.data
    }
}

impl<D: Data, B: Backend> DataViewMut for CnvPVecR<D, B> {
    fn data_mut(&mut self) -> &mut Self::D {
        &mut self.data
    }
}

impl<D: HostDataRef, BE: Backend> ZnxView for CnvPVecR<D, BE> {
    type Scalar = BE::ScalarPrep;
}

impl<D: Data, BE: Backend> CnvPVecR<D, BE> {
    pub fn shape(&self) -> CnvPVecShape {
        self.shape
    }

    pub fn n(&self) -> usize {
        self.shape.n()
    }

    pub fn cols(&self) -> usize {
        self.shape.cols()
    }

    pub fn size(&self) -> usize {
        self.shape.size()
    }
}

impl<B: Backend> CnvPVecR<B::OwnedBuf, B> {
    pub fn alloc(n: usize, cols: usize, size: usize) -> Self {
        let data: B::OwnedBuf = B::alloc_zeroed_bytes(B::bytes_of_cnv_pvec_right(n, cols, size));
        Self {
            data,
            shape: CnvPVecShape::new(n, cols, size),
            _phantom: PhantomData,
        }
    }

    pub fn from_bytes(n: usize, cols: usize, size: usize, bytes: impl Into<Vec<u8>>) -> Self {
        let data: Vec<u8> = bytes.into();
        assert!(data.len() == B::bytes_of_cnv_pvec_right(n, cols, size));
        let data: B::OwnedBuf = B::from_host_bytes(&data);
        Self {
            data,
            shape: CnvPVecShape::new(n, cols, size),
            _phantom: PhantomData,
        }
    }
}

impl<D: Data, B: Backend> CnvPVecR<D, B> {
    pub fn from_data(data: D, n: usize, cols: usize, size: usize) -> Self {
        Self {
            data,
            shape: CnvPVecShape::new(n, cols, size),
            _phantom: PhantomData,
        }
    }
}

/// Prepared left operand for bivariate convolution.
///
/// Holds a polynomial vector in the backend's prepared representation,
/// ready to be used as the left operand of
/// [`Convolution::cnv_apply_dft`](crate::api::Convolution::cnv_apply_dft).
/// Created via [`Convolution::cnv_prepare_left`](crate::api::Convolution::cnv_prepare_left).
pub struct CnvPVecL<D: Data, BE: Backend> {
    data: D,
    shape: CnvPVecShape,
    _phantom: PhantomData<BE>,
}

impl<D: Data, BE: Backend> ZnxInfos for CnvPVecL<D, BE> {
    fn cols(&self) -> usize {
        self.shape.cols()
    }

    fn n(&self) -> usize {
        self.shape.n()
    }

    fn rows(&self) -> usize {
        1
    }

    fn size(&self) -> usize {
        self.shape.size()
    }
}

impl<D: Data, BE: Backend> DataView for CnvPVecL<D, BE> {
    type D = D;
    fn data(&self) -> &Self::D {
        &self.data
    }
}

impl<D: Data, B: Backend> DataViewMut for CnvPVecL<D, B> {
    fn data_mut(&mut self) -> &mut Self::D {
        &mut self.data
    }
}

impl<D: HostDataRef, BE: Backend> ZnxView for CnvPVecL<D, BE> {
    type Scalar = BE::ScalarPrep;
}

impl<D: Data, BE: Backend> CnvPVecL<D, BE> {
    pub fn shape(&self) -> CnvPVecShape {
        self.shape
    }

    pub fn n(&self) -> usize {
        self.shape.n()
    }

    pub fn cols(&self) -> usize {
        self.shape.cols()
    }

    pub fn size(&self) -> usize {
        self.shape.size()
    }
}

impl<B: Backend> CnvPVecL<B::OwnedBuf, B> {
    pub fn alloc(n: usize, cols: usize, size: usize) -> Self {
        let data: B::OwnedBuf = B::alloc_zeroed_bytes(B::bytes_of_cnv_pvec_left(n, cols, size));
        Self {
            data,
            shape: CnvPVecShape::new(n, cols, size),
            _phantom: PhantomData,
        }
    }

    pub fn from_bytes(n: usize, cols: usize, size: usize, bytes: impl Into<Vec<u8>>) -> Self {
        let data: Vec<u8> = bytes.into();
        assert!(data.len() == B::bytes_of_cnv_pvec_left(n, cols, size));
        let data: B::OwnedBuf = B::from_host_bytes(&data);
        Self {
            data,
            shape: CnvPVecShape::new(n, cols, size),
            _phantom: PhantomData,
        }
    }
}

impl<D: Data, B: Backend> CnvPVecL<D, B> {
    pub fn from_data(data: D, n: usize, cols: usize, size: usize) -> Self {
        Self {
            data,
            shape: CnvPVecShape::new(n, cols, size),
            _phantom: PhantomData,
        }
    }
}

/// Borrow a `CnvPVecR` as a shared reference view.
pub type CnvPVecRBackendRef<'a, B> = CnvPVecR<<B as Backend>::BufRef<'a>, B>;
pub type CnvPVecRBackendMut<'a, B> = CnvPVecR<<B as Backend>::BufMut<'a>, B>;
pub type CnvPVecLBackendRef<'a, B> = CnvPVecL<<B as Backend>::BufRef<'a>, B>;
pub type CnvPVecLBackendMut<'a, B> = CnvPVecL<<B as Backend>::BufMut<'a>, B>;

/// Borrow a backend-owned `CnvPVecR` using the backend's native view type.
pub trait CnvPVecRToBackendRef<BE: Backend> {
    fn to_backend_ref(&self) -> CnvPVecRBackendRef<'_, BE>;
}

impl<BE: Backend> CnvPVecRToBackendRef<BE> for CnvPVecR<BE::OwnedBuf, BE> {
    fn to_backend_ref(&self) -> CnvPVecRBackendRef<'_, BE> {
        CnvPVecR {
            data: BE::view(&self.data),
            shape: self.shape,
            _phantom: self._phantom,
        }
    }
}

/// Reborrow an already backend-borrowed `CnvPVecR` as a shared backend-native view.
pub trait CnvPVecRReborrowBackendRef<BE: Backend> {
    fn reborrow_backend_ref(&self) -> CnvPVecRBackendRef<'_, BE>;
}

impl<'b, BE: Backend + 'b> CnvPVecRReborrowBackendRef<BE> for CnvPVecR<BE::BufMut<'b>, BE> {
    fn reborrow_backend_ref(&self) -> CnvPVecRBackendRef<'_, BE> {
        CnvPVecR {
            data: BE::view_ref_mut(&self.data),
            shape: self.shape,
            _phantom: self._phantom,
        }
    }
}

/// Mutably borrow a backend-owned `CnvPVecR` using the backend's native view type.
pub trait CnvPVecRToBackendMut<BE: Backend> {
    fn to_backend_mut(&mut self) -> CnvPVecRBackendMut<'_, BE>;
}

impl<BE: Backend> CnvPVecRToBackendMut<BE> for CnvPVecR<BE::OwnedBuf, BE> {
    fn to_backend_mut(&mut self) -> CnvPVecRBackendMut<'_, BE> {
        CnvPVecR {
            data: BE::view_mut(&mut self.data),
            shape: self.shape,
            _phantom: self._phantom,
        }
    }
}

/// Reborrow an already backend-borrowed `CnvPVecR` as a mutable backend-native view.
pub trait CnvPVecRReborrowBackendMut<BE: Backend> {
    fn reborrow_backend_mut(&mut self) -> CnvPVecRBackendMut<'_, BE>;
}

impl<'b, BE: Backend + 'b> CnvPVecRReborrowBackendMut<BE> for CnvPVecR<BE::BufMut<'b>, BE> {
    fn reborrow_backend_mut(&mut self) -> CnvPVecRBackendMut<'_, BE> {
        CnvPVecR {
            data: BE::view_mut_ref(&mut self.data),
            shape: self.shape,
            _phantom: self._phantom,
        }
    }
}

/// Borrow a `CnvPVecL` as a shared reference view.
pub trait CnvPVecLToBackendRef<BE: Backend> {
    fn to_backend_ref(&self) -> CnvPVecLBackendRef<'_, BE>;
}

impl<BE: Backend> CnvPVecLToBackendRef<BE> for CnvPVecL<BE::OwnedBuf, BE> {
    fn to_backend_ref(&self) -> CnvPVecLBackendRef<'_, BE> {
        CnvPVecL {
            data: BE::view(&self.data),
            shape: self.shape,
            _phantom: self._phantom,
        }
    }
}

/// Reborrow an already backend-borrowed `CnvPVecL` as a shared backend-native view.
pub trait CnvPVecLReborrowBackendRef<BE: Backend> {
    fn reborrow_backend_ref(&self) -> CnvPVecLBackendRef<'_, BE>;
}

impl<'b, BE: Backend + 'b> CnvPVecLReborrowBackendRef<BE> for CnvPVecL<BE::BufMut<'b>, BE> {
    fn reborrow_backend_ref(&self) -> CnvPVecLBackendRef<'_, BE> {
        CnvPVecL {
            data: BE::view_ref_mut(&self.data),
            shape: self.shape,
            _phantom: self._phantom,
        }
    }
}

/// Mutably borrow a backend-owned `CnvPVecL` using the backend's native view type.
pub trait CnvPVecLToBackendMut<BE: Backend> {
    fn to_backend_mut(&mut self) -> CnvPVecLBackendMut<'_, BE>;
}

impl<BE: Backend> CnvPVecLToBackendMut<BE> for CnvPVecL<BE::OwnedBuf, BE> {
    fn to_backend_mut(&mut self) -> CnvPVecLBackendMut<'_, BE> {
        CnvPVecL {
            data: BE::view_mut(&mut self.data),
            shape: self.shape,
            _phantom: self._phantom,
        }
    }
}

/// Reborrow an already backend-borrowed `CnvPVecL` as a mutable backend-native view.
pub trait CnvPVecLReborrowBackendMut<BE: Backend> {
    fn reborrow_backend_mut(&mut self) -> CnvPVecLBackendMut<'_, BE>;
}

impl<'b, BE: Backend + 'b> CnvPVecLReborrowBackendMut<BE> for CnvPVecL<BE::BufMut<'b>, BE> {
    fn reborrow_backend_mut(&mut self) -> CnvPVecLBackendMut<'_, BE> {
        CnvPVecL {
            data: BE::view_mut_ref(&mut self.data),
            shape: self.shape,
            _phantom: self._phantom,
        }
    }
}
