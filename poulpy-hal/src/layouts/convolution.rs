use std::marker::PhantomData;

use crate::layouts::{Backend, Data, DataView, DataViewMut, DftWord, HostDataRef, ZnxInfos, ZnxView};

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
/// Holds a polynomial vector in the prepared representation named by the
/// [`DftWord`] type `W`, ready to be used as the right operand of
/// [`Convolution::cnv_apply_dft`](crate::api::Convolution::cnv_apply_dft).
/// Created via [`Convolution::cnv_prepare_right`](crate::api::Convolution::cnv_prepare_right).
pub struct CnvPVecR<D: Data, W: DftWord, B: Backend<DftWord = W>> {
    data: D,
    shape: CnvPVecShape,
    _phantom: PhantomData<(W, B)>,
}

impl<D: Data, W: DftWord, B: Backend<DftWord = W>> ZnxInfos for CnvPVecR<D, W, B> {
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

impl<D: Data, W: DftWord, B: Backend<DftWord = W>> DataView for CnvPVecR<D, W, B> {
    type D = D;
    fn data(&self) -> &Self::D {
        &self.data
    }
}

impl<D: Data, W: DftWord, B: Backend<DftWord = W>> DataViewMut for CnvPVecR<D, W, B> {
    fn data_mut(&mut self) -> &mut Self::D {
        &mut self.data
    }
}

impl<D: HostDataRef, W: DftWord, B: Backend<DftWord = W>> ZnxView for CnvPVecR<D, W, B> {
    type Scalar = W;
}

impl<D: Data, W: DftWord, B: Backend<DftWord = W>> CnvPVecR<D, W, B> {
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

impl<D: Data, W: DftWord, B: Backend<DftWord = W>> CnvPVecR<D, W, B> {
    /// Allocates a zero-initialized backend-owned `CnvPVecR`.
    pub fn alloc(n: usize, cols: usize, size: usize) -> CnvPVecR<B::OwnedBuf, W, B>
    where
        B: Backend<OwnedBuf = D>,
    {
        let data: B::OwnedBuf = B::alloc_zeroed_bytes(B::bytes_of_cnv_pvec_right(n, cols, size));
        CnvPVecR {
            data,
            shape: CnvPVecShape::new(n, cols, size),
            _phantom: PhantomData,
        }
    }

    /// Uploads a host byte buffer into a backend-owned `CnvPVecR`.
    ///
    /// # Panics
    ///
    /// Panics if the buffer length does not equal `B::bytes_of_cnv_pvec_right(n, cols, size)`.
    pub fn from_bytes(n: usize, cols: usize, size: usize, bytes: impl Into<Vec<u8>>) -> CnvPVecR<B::OwnedBuf, W, B>
    where
        B: Backend<OwnedBuf = D>,
    {
        let data: Vec<u8> = bytes.into();
        assert!(data.len() == B::bytes_of_cnv_pvec_right(n, cols, size));
        let data: B::OwnedBuf = B::from_host_bytes(&data);
        CnvPVecR {
            data,
            shape: CnvPVecShape::new(n, cols, size),
            _phantom: PhantomData,
        }
    }
}

impl<D: Data, W: DftWord, B: Backend<DftWord = W>> CnvPVecR<D, W, B> {
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
/// Holds a polynomial vector in the prepared representation named by the
/// [`DftWord`] type `W`, ready to be used as the left operand of
/// [`Convolution::cnv_apply_dft`](crate::api::Convolution::cnv_apply_dft).
/// Created via [`Convolution::cnv_prepare_left`](crate::api::Convolution::cnv_prepare_left).
pub struct CnvPVecL<D: Data, W: DftWord, B: Backend<DftWord = W>> {
    data: D,
    shape: CnvPVecShape,
    _phantom: PhantomData<(W, B)>,
}

impl<D: Data, W: DftWord, B: Backend<DftWord = W>> ZnxInfos for CnvPVecL<D, W, B> {
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

impl<D: Data, W: DftWord, B: Backend<DftWord = W>> DataView for CnvPVecL<D, W, B> {
    type D = D;
    fn data(&self) -> &Self::D {
        &self.data
    }
}

impl<D: Data, W: DftWord, B: Backend<DftWord = W>> DataViewMut for CnvPVecL<D, W, B> {
    fn data_mut(&mut self) -> &mut Self::D {
        &mut self.data
    }
}

impl<D: HostDataRef, W: DftWord, B: Backend<DftWord = W>> ZnxView for CnvPVecL<D, W, B> {
    type Scalar = W;
}

impl<D: Data, W: DftWord, B: Backend<DftWord = W>> CnvPVecL<D, W, B> {
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

impl<D: Data, W: DftWord, B: Backend<DftWord = W>> CnvPVecL<D, W, B> {
    /// Allocates a zero-initialized backend-owned `CnvPVecL`.
    pub fn alloc(n: usize, cols: usize, size: usize) -> CnvPVecL<B::OwnedBuf, W, B>
    where
        B: Backend<OwnedBuf = D>,
    {
        let data: B::OwnedBuf = B::alloc_zeroed_bytes(B::bytes_of_cnv_pvec_left(n, cols, size));
        CnvPVecL {
            data,
            shape: CnvPVecShape::new(n, cols, size),
            _phantom: PhantomData,
        }
    }

    /// Uploads a host byte buffer into a backend-owned `CnvPVecL`.
    ///
    /// # Panics
    ///
    /// Panics if the buffer length does not equal `B::bytes_of_cnv_pvec_left(n, cols, size)`.
    pub fn from_bytes(n: usize, cols: usize, size: usize, bytes: impl Into<Vec<u8>>) -> CnvPVecL<B::OwnedBuf, W, B>
    where
        B: Backend<OwnedBuf = D>,
    {
        let data: Vec<u8> = bytes.into();
        assert!(data.len() == B::bytes_of_cnv_pvec_left(n, cols, size));
        let data: B::OwnedBuf = B::from_host_bytes(&data);
        CnvPVecL {
            data,
            shape: CnvPVecShape::new(n, cols, size),
            _phantom: PhantomData,
        }
    }
}

impl<D: Data, W: DftWord, B: Backend<DftWord = W>> CnvPVecL<D, W, B> {
    pub fn from_data(data: D, n: usize, cols: usize, size: usize) -> Self {
        Self {
            data,
            shape: CnvPVecShape::new(n, cols, size),
            _phantom: PhantomData,
        }
    }
}

/// One `(left, right)` operand pair of a fused convolution accumulation.
///
/// Consumed by [`Convolution::cnv_accumulate_dft`](crate::api::Convolution::cnv_accumulate_dft),
/// which overwrites the destination column with the sum of the bivariate
/// convolutions of all terms.
pub struct CnvDftAccTerm<'a, BE: Backend + 'a> {
    /// Prepared left operand.
    pub a: CnvPVecLBackendRef<'a, BE>,
    /// Column of `a` to convolve.
    pub a_col: usize,
    /// Prepared right operand.
    pub b: CnvPVecRBackendRef<'a, BE>,
    /// Column of `b` to convolve.
    pub b_col: usize,
}

/// Borrow a `CnvPVecR` as a shared reference view.
/// Owned `CnvPVecR` backed by a backend-owned buffer.
pub type CnvPVecROwned<B> = CnvPVecR<<B as Backend>::OwnedBuf, <B as Backend>::DftWord, B>;
/// Owned `CnvPVecL` backed by a backend-owned buffer.
pub type CnvPVecLOwned<B> = CnvPVecL<<B as Backend>::OwnedBuf, <B as Backend>::DftWord, B>;
pub type CnvPVecRBackendRef<'a, B> = CnvPVecR<<B as Backend>::BufRef<'a>, <B as Backend>::DftWord, B>;
pub type CnvPVecRBackendMut<'a, B> = CnvPVecR<<B as Backend>::BufMut<'a>, <B as Backend>::DftWord, B>;
pub type CnvPVecLBackendRef<'a, B> = CnvPVecL<<B as Backend>::BufRef<'a>, <B as Backend>::DftWord, B>;
pub type CnvPVecLBackendMut<'a, B> = CnvPVecL<<B as Backend>::BufMut<'a>, <B as Backend>::DftWord, B>;

/// Borrow a backend-owned `CnvPVecR` using the backend's native view type.
pub trait CnvPVecRToBackendRef<BE: Backend> {
    fn to_backend_ref(&self) -> CnvPVecRBackendRef<'_, BE>;
}

impl<BE: Backend> CnvPVecRToBackendRef<BE> for CnvPVecR<BE::OwnedBuf, BE::DftWord, BE> {
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

impl<'b, BE: Backend + 'b> CnvPVecRReborrowBackendRef<BE> for CnvPVecR<BE::BufMut<'b>, BE::DftWord, BE> {
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

impl<BE: Backend> CnvPVecRToBackendMut<BE> for CnvPVecR<BE::OwnedBuf, BE::DftWord, BE> {
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

impl<'b, BE: Backend + 'b> CnvPVecRReborrowBackendMut<BE> for CnvPVecR<BE::BufMut<'b>, BE::DftWord, BE> {
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

impl<BE: Backend> CnvPVecLToBackendRef<BE> for CnvPVecL<BE::OwnedBuf, BE::DftWord, BE> {
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

impl<'b, BE: Backend + 'b> CnvPVecLReborrowBackendRef<BE> for CnvPVecL<BE::BufMut<'b>, BE::DftWord, BE> {
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

impl<BE: Backend> CnvPVecLToBackendMut<BE> for CnvPVecL<BE::OwnedBuf, BE::DftWord, BE> {
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

impl<'b, BE: Backend + 'b> CnvPVecLReborrowBackendMut<BE> for CnvPVecL<BE::BufMut<'b>, BE::DftWord, BE> {
    fn reborrow_backend_mut(&mut self) -> CnvPVecLBackendMut<'_, BE> {
        CnvPVecL {
            data: BE::view_mut_ref(&mut self.data),
            shape: self.shape,
            _phantom: self._phantom,
        }
    }
}

impl<D: Data, W: DftWord, B: Backend<DftWord = W>> CnvPVecL<D, W, B> {
    /// Zero-copy re-tag of this container to a layout-compatible backend `B2`.
    ///
    /// The buffer moves as-is; only the type tag changes. Requires the
    /// [`CnvPVecLayoutCompatible`](crate::layouts::CnvPVecLayoutCompatible) marker declared by the backend
    /// pair. `D` is kept, so for further backend-native use `B2`'s buffer
    /// types must match `D` (true for all current CPU backends).
    pub fn into_backend<B2>(self) -> CnvPVecL<D, W, B2>
    where
        B2: Backend<DftWord = W>,
        B: crate::layouts::CnvPVecLayoutCompatible<B2>,
    {
        let shape = self.shape;
        assert_eq!(
            B::bytes_of_cnv_pvec_left(shape.n(), shape.cols(), shape.size()),
            B2::bytes_of_cnv_pvec_left(shape.n(), shape.cols(), shape.size()),
            "into_backend: byte sizes diverge despite declared layout compatibility"
        );
        CnvPVecL {
            data: self.data,
            shape,
            _phantom: PhantomData,
        }
    }
}

impl<D: Data, W: DftWord, B: Backend<DftWord = W>> CnvPVecR<D, W, B> {
    /// Zero-copy re-tag of this container to a layout-compatible backend `B2`.
    ///
    /// The buffer moves as-is; only the type tag changes. Requires the
    /// [`CnvPVecLayoutCompatible`](crate::layouts::CnvPVecLayoutCompatible) marker declared by the backend
    /// pair. `D` is kept, so for further backend-native use `B2`'s buffer
    /// types must match `D` (true for all current CPU backends).
    pub fn into_backend<B2>(self) -> CnvPVecR<D, W, B2>
    where
        B2: Backend<DftWord = W>,
        B: crate::layouts::CnvPVecLayoutCompatible<B2>,
    {
        let shape = self.shape;
        assert_eq!(
            B::bytes_of_cnv_pvec_right(shape.n(), shape.cols(), shape.size()),
            B2::bytes_of_cnv_pvec_right(shape.n(), shape.cols(), shape.size()),
            "into_backend: byte sizes diverge despite declared layout compatibility"
        );
        CnvPVecR {
            data: self.data,
            shape,
            _phantom: PhantomData,
        }
    }
}
