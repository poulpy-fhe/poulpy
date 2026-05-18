use std::mem::size_of;

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

/// Shape metadata for a plain coefficient matrix used to transform [`LWEMatrix`].
pub trait CoeffMatrixInfos: LWEInfos {
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
    fn rows_out(&self) -> usize {
        self.rows_out
    }
}

/// Plain coefficient matrix `U`.
///
/// Column `out` stores the coefficients of output row `out`, i.e.
/// `data[out][in] = U[out, in]`.
#[derive(PartialEq, Eq, Clone)]
pub struct CoeffMatrix<D: Data> {
    pub(crate) data: VecZnx<D>,
    pub(crate) base2k: Base2K,
}

pub type CoeffMatrixBackendRef<'a, BE> = CoeffMatrix<<BE as Backend>::BufRef<'a>>;
pub type CoeffMatrixBackendMut<'a, BE> = CoeffMatrix<<BE as Backend>::BufMut<'a>>;

impl<D: Data> LWEInfos for CoeffMatrix<D> {
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

impl<D: Data> CoeffMatrixInfos for CoeffMatrix<D> {
    fn rows_out(&self) -> usize {
        self.data.cols()
    }
}

impl<D: Data> CoeffMatrix<D> {
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

impl<BE: Backend> CoeffMatrixToBackendRef<BE> for CoeffMatrix<BE::OwnedBuf> {
    fn to_backend_ref(&self) -> CoeffMatrixBackendRef<'_, BE> {
        CoeffMatrix {
            data: <VecZnx<BE::OwnedBuf> as VecZnxToBackendRef<BE>>::to_backend_ref(&self.data),
            base2k: self.base2k,
        }
    }
}

pub trait CoeffMatrixToBackendMut<BE: Backend>: CoeffMatrixToBackendRef<BE> {
    fn to_backend_mut(&mut self) -> CoeffMatrixBackendMut<'_, BE>;
}

impl<BE: Backend> CoeffMatrixToBackendMut<BE> for CoeffMatrix<BE::OwnedBuf> {
    fn to_backend_mut(&mut self) -> CoeffMatrixBackendMut<'_, BE> {
        CoeffMatrix {
            data: <VecZnx<BE::OwnedBuf> as VecZnxToBackendMut<BE>>::to_backend_mut(&mut self.data),
            base2k: self.base2k,
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
