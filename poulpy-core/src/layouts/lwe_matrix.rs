use std::{marker::PhantomData, mem::size_of};

use poulpy_hal::layouts::{Backend, Data, HostDataRef, ScalarZnx, VecZnx, VecZnxToBackendMut, VecZnxToBackendRef};

use crate::layouts::{Base2K, Degree, LWEInfos, SetLWEInfos, TorusPrecision};

/// Shape metadata for a packed matrix of LWE ciphertexts.
pub trait LWEMatrixInfos: LWEInfos {
    /// Number of active LWE rows packed on the coefficient axis.
    fn rows(&self) -> usize;
}

/// Plain-data shape for a packed matrix of LWE ciphertexts.
#[derive(PartialEq, Eq, Copy, Clone, Debug)]
pub struct LWEMatrixLayout {
    pub rows: usize,
    pub n: Degree,
    pub base2k: Base2K,
    pub k: TorusPrecision,
}

impl LWEInfos for LWEMatrixLayout {
    fn base2k(&self) -> Base2K {
        self.base2k
    }

    fn n(&self) -> Degree {
        self.n
    }

    fn size(&self) -> usize {
        self.k.as_usize().div_ceil(self.base2k.as_usize())
    }
}

impl LWEMatrixInfos for LWEMatrixLayout {
    fn rows(&self) -> usize {
        self.rows
    }
}

/// Packed matrix of LWE ciphertexts, stored as `(A, b)`.
///
/// `body[row]` is `b_row`; `mask[col][row]` is `A[row, col]`.
#[derive(PartialEq, Eq, Clone)]
pub struct LWEMatrix<D: Data> {
    pub(crate) body: VecZnx<D>,
    pub(crate) mask: VecZnx<D>,
    pub(crate) base2k: Base2K,
}

pub type LWEMatrixBackendRef<'a, BE> = LWEMatrix<<BE as Backend>::BufRef<'a>>;
pub type LWEMatrixBackendMut<'a, BE> = LWEMatrix<<BE as Backend>::BufMut<'a>>;

impl<D: Data> LWEInfos for LWEMatrix<D> {
    fn n(&self) -> Degree {
        Degree(self.mask.cols() as u32)
    }

    fn base2k(&self) -> Base2K {
        self.base2k
    }

    fn size(&self) -> usize {
        self.body.size()
    }
}

impl<D: Data> LWEMatrixInfos for LWEMatrix<D> {
    fn rows(&self) -> usize {
        self.body.n()
    }
}

impl<D: Data> SetLWEInfos for LWEMatrix<D> {
    fn set_base2k(&mut self, base2k: Base2K) {
        self.base2k = base2k;
    }
}

impl<D: Data> LWEMatrix<D> {
    pub fn body(&self) -> &VecZnx<D> {
        &self.body
    }

    pub fn body_mut(&mut self) -> &mut VecZnx<D> {
        &mut self.body
    }

    pub fn mask(&self) -> &VecZnx<D> {
        &self.mask
    }

    pub fn mask_mut(&mut self) -> &mut VecZnx<D> {
        &mut self.mask
    }
}

impl<D: Data> LWEMatrix<D> {
    pub fn reinterpret<To>(self) -> LWEMatrix<To::OwnedBuf>
    where
        To: Backend<OwnedBuf = D>,
    {
        let body_shape = self.body.shape();
        let mask_shape = self.mask.shape();
        LWEMatrix {
            body: VecZnx::from_data_with_max_size(
                self.body.data,
                body_shape.n(),
                body_shape.cols(),
                body_shape.size(),
                body_shape.max_size(),
            ),
            mask: VecZnx::from_data_with_max_size(
                self.mask.data,
                mask_shape.n(),
                mask_shape.cols(),
                mask_shape.size(),
                mask_shape.max_size(),
            ),
            base2k: self.base2k,
        }
    }
}

impl<D: HostDataRef> LWEMatrix<D> {
    pub fn to_host_owned<BE>(&self) -> LWEMatrix<Vec<u8>>
    where
        BE: Backend<OwnedBuf = D>,
    {
        LWEMatrix {
            body: self.body.to_host_owned::<BE>(),
            mask: self.mask.to_host_owned::<BE>(),
            base2k: self.base2k,
        }
    }
}

pub trait LWEMatrixToBackendRef<BE: Backend> {
    fn to_backend_ref(&self) -> LWEMatrixBackendRef<'_, BE>;
}

impl<BE: Backend> LWEMatrixToBackendRef<BE> for LWEMatrix<BE::OwnedBuf> {
    fn to_backend_ref(&self) -> LWEMatrixBackendRef<'_, BE> {
        LWEMatrix {
            body: <VecZnx<BE::OwnedBuf> as VecZnxToBackendRef<BE>>::to_backend_ref(&self.body),
            mask: <VecZnx<BE::OwnedBuf> as VecZnxToBackendRef<BE>>::to_backend_ref(&self.mask),
            base2k: self.base2k,
        }
    }
}

pub trait LWEMatrixToBackendMut<BE: Backend>: LWEMatrixToBackendRef<BE> {
    fn to_backend_mut(&mut self) -> LWEMatrixBackendMut<'_, BE>;
}

impl<BE: Backend> LWEMatrixToBackendMut<BE> for LWEMatrix<BE::OwnedBuf> {
    fn to_backend_mut(&mut self) -> LWEMatrixBackendMut<'_, BE> {
        LWEMatrix {
            body: <VecZnx<BE::OwnedBuf> as VecZnxToBackendMut<BE>>::to_backend_mut(&mut self.body),
            mask: <VecZnx<BE::OwnedBuf> as VecZnxToBackendMut<BE>>::to_backend_mut(&mut self.mask),
            base2k: self.base2k,
        }
    }
}

mod coeff_bound_private {
    pub trait Sealed {}
    impl Sealed for i8 {}
    impl Sealed for i16 {}
    impl Sealed for i32 {}
    impl Sealed for i64 {}
}

/// Compile-time bound on the magnitude of a [`CoeffMatrix`] entry.
///
/// Implemented only for `i8`/`i16`/`i32`/`i64`; the type is the guarantee
/// (no runtime range check). `WIDTH` is the SIMD piece width selected for the
/// per-limb dot kernel.
pub trait CoeffBound: coeff_bound_private::Sealed + Copy + 'static {
    const WIDTH: u32;
}
impl CoeffBound for i8 {
    const WIDTH: u32 = 8;
}
impl CoeffBound for i16 {
    const WIDTH: u32 = 16;
}
impl CoeffBound for i32 {
    const WIDTH: u32 = 32;
}
impl CoeffBound for i64 {
    const WIDTH: u32 = 64;
}

/// Shape metadata for a plain coefficient matrix used to transform [`LWEMatrix`].
pub trait CoeffMatrixInfos: LWEInfos {
    /// Compile-time entry bound (selects the GEMM kernel).
    type Bound: CoeffBound;
    /// Number of active output rows.
    fn rows_out(&self) -> usize;
}

#[derive(PartialEq, Eq, Copy, Clone, Debug)]
pub struct CoeffMatrixLayout {
    pub n: Degree,
    pub rows_out: usize,
    pub base2k: Base2K,
    pub k: TorusPrecision,
}

impl LWEInfos for CoeffMatrixLayout {
    fn base2k(&self) -> Base2K {
        self.base2k
    }

    fn n(&self) -> Degree {
        self.n
    }

    fn size(&self) -> usize {
        self.k.as_usize().div_ceil(self.base2k.as_usize())
    }
}

impl CoeffMatrixInfos for CoeffMatrixLayout {
    type Bound = i64;
    fn rows_out(&self) -> usize {
        self.rows_out
    }
}

/// Plain coefficient matrix `U`, with a compile-time entry bound `BU`.
///
/// Column `out` stores the coefficients of output row `out`, i.e.
/// `data[out][in] = U[out, in]`. `BU` declares the magnitude bound of the
/// entries (default `i64` = unconstrained); it selects the GEMM kernel at
/// monomorphization with no runtime check.
#[derive(PartialEq, Eq, Clone)]
pub struct CoeffMatrix<D: Data, BU: CoeffBound = i64> {
    pub(crate) data: VecZnx<D>,
    pub(crate) base2k: Base2K,
    pub(crate) _bound: PhantomData<BU>,
}

pub type CoeffMatrixBackendRef<'a, BE> = CoeffMatrix<<BE as Backend>::BufRef<'a>>;
pub type CoeffMatrixBackendMut<'a, BE> = CoeffMatrix<<BE as Backend>::BufMut<'a>>;

impl<D: Data, BU: CoeffBound> LWEInfos for CoeffMatrix<D, BU> {
    fn n(&self) -> Degree {
        Degree(self.data.n() as u32)
    }

    fn base2k(&self) -> Base2K {
        self.base2k
    }

    fn size(&self) -> usize {
        self.data.size()
    }
}

impl<D: Data, BU: CoeffBound> CoeffMatrixInfos for CoeffMatrix<D, BU> {
    type Bound = BU;
    fn rows_out(&self) -> usize {
        self.data.cols()
    }
}

impl<D: Data, BU: CoeffBound> CoeffMatrix<D, BU> {
    pub fn data(&self) -> &VecZnx<D> {
        &self.data
    }

    pub fn data_mut(&mut self) -> &mut VecZnx<D> {
        &mut self.data
    }
}

pub trait CoeffMatrixToBackendRef<BE: Backend> {
    fn to_backend_ref(&self) -> CoeffMatrixBackendRef<'_, BE>;
}

impl<BE: Backend, BU: CoeffBound> CoeffMatrixToBackendRef<BE> for CoeffMatrix<BE::OwnedBuf, BU> {
    fn to_backend_ref(&self) -> CoeffMatrixBackendRef<'_, BE> {
        CoeffMatrix {
            data: <VecZnx<BE::OwnedBuf> as VecZnxToBackendRef<BE>>::to_backend_ref(&self.data),
            base2k: self.base2k,
            _bound: PhantomData,
        }
    }
}

pub trait CoeffMatrixToBackendMut<BE: Backend>: CoeffMatrixToBackendRef<BE> {
    fn to_backend_mut(&mut self) -> CoeffMatrixBackendMut<'_, BE>;
}

impl<BE: Backend, BU: CoeffBound> CoeffMatrixToBackendMut<BE> for CoeffMatrix<BE::OwnedBuf, BU> {
    fn to_backend_mut(&mut self) -> CoeffMatrixBackendMut<'_, BE> {
        CoeffMatrix {
            data: <VecZnx<BE::OwnedBuf> as VecZnxToBackendMut<BE>>::to_backend_mut(&mut self.data),
            base2k: self.base2k,
            _bound: PhantomData,
        }
    }
}

pub fn coeff_matrix_scalar_backend_ref<'a, 'b, BE: Backend + 'b>(
    matrix: &'a CoeffMatrix<BE::BufRef<'b>>,
    col: usize,
    limb: usize,
) -> ScalarZnx<BE::BufRef<'a>> {
    assert!(col < matrix.data.cols(), "CoeffMatrix col out of bounds");
    assert!(limb < matrix.data.size(), "CoeffMatrix limb out of bounds");
    let start = (limb * matrix.data.cols() + col) * matrix.data.n() * size_of::<i64>();
    let len = matrix.data.n() * size_of::<i64>();
    ScalarZnx::from_data(BE::region_ref(&matrix.data.data, start, len), matrix.data.n(), 1)
}
