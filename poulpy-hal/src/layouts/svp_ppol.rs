use std::{
    fmt,
    hash::{DefaultHasher, Hasher},
    marker::PhantomData,
};

use crate::layouts::{Backend, Data, DataView, DataViewMut, DftWord, DigestU64, HostDataRef, ScalarZnxShape, ZnxInfos, ZnxView};

/// Prepared (DFT-domain) scalar polynomial for scalar-vector products.
///
/// An `SvpPPol` holds a single polynomial in a prepared representation
/// named by its [`DftWord`] type `W`. It is used as the left operand in
/// [`SvpApplyDft`](crate::api::SvpApplyDft) to efficiently multiply a
/// scalar polynomial by each column of a [`VecZnxDft`](crate::layouts::VecZnxDft).
///
/// Create via [`SvpPrepare`](crate::api::SvpPrepare) from a
/// coefficient-domain [`ScalarZnx`](crate::layouts::ScalarZnx).
///
/// Ring degree `n` is always a power of two, so the DFT-domain layout has a
/// coefficient count that matches vector lane widths relative to buffer alignment.
#[repr(C)]
#[derive(PartialEq, Eq, Hash)]
pub struct SvpPPol<D: Data, W: DftWord> {
    pub data: D,
    shape: ScalarZnxShape,
    pub _phantom: PhantomData<W>,
}

impl<D: HostDataRef, W: DftWord> DigestU64 for SvpPPol<D, W> {
    fn digest_u64(&self) -> u64 {
        let mut h: DefaultHasher = DefaultHasher::new();
        h.write(self.data.as_ref());
        h.write_usize(self.n());
        h.write_usize(self.cols());
        h.finish()
    }
}

impl<D: HostDataRef, W: DftWord> ZnxView for SvpPPol<D, W> {
    type Scalar = W;
}

impl<D: Data, W: DftWord> ZnxInfos for SvpPPol<D, W> {
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

impl<D: Data, W: DftWord> SvpPPol<D, W> {
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

impl<D: Data, W: DftWord> DataView for SvpPPol<D, W> {
    type D = D;
    fn data(&self) -> &Self::D {
        &self.data
    }
}

impl<D: Data, W: DftWord> DataViewMut for SvpPPol<D, W> {
    fn data_mut(&mut self) -> &mut Self::D {
        &mut self.data
    }
}

impl<D: Data, W: DftWord> SvpPPol<D, W> {
    /// Allocates a zero-initialized backend-owned `SvpPPol`.
    pub fn alloc<B>(n: usize, cols: usize) -> SvpPPolOwned<B>
    where
        B: Backend<DftWord = W, OwnedBuf = D>,
    {
        let data: <B as Backend>::OwnedBuf = B::alloc_zeroed_bytes(B::bytes_of_svp_ppol(n, cols));
        SvpPPol {
            data,
            shape: ScalarZnxShape::new(n, cols),
            _phantom: PhantomData,
        }
    }
}

impl<D: Data, W: DftWord> SvpPPol<D, W> {
    /// Borrows this backend-owned `SvpPPol` using the backend's native view type.
    ///
    /// Ergonomic inherent form of [`SvpPPolToBackendRef`]: the backend is
    /// supplied explicitly (`x.to_backend_ref::<B>()`) since it is not
    /// recoverable from the word-keyed type alone.
    pub fn to_backend_ref<B>(&self) -> SvpPPolBackendRef<'_, B>
    where
        B: Backend<OwnedBuf = D, DftWord = W>,
    {
        SvpPPolToBackendRef::<B>::to_backend_ref(self)
    }

    /// Mutably borrows this backend-owned `SvpPPol` using the backend's native view type.
    ///
    /// Ergonomic inherent form of [`SvpPPolToBackendMut`]; see
    /// [`Self::to_backend_ref`].
    pub fn to_backend_mut<B>(&mut self) -> SvpPPolBackendMut<'_, B>
    where
        B: Backend<OwnedBuf = D, DftWord = W>,
    {
        SvpPPolToBackendMut::<B>::to_backend_mut(self)
    }
}

/// Owned `SvpPPol` backed by a backend-owned buffer.
pub type SvpPPolOwned<B> = SvpPPol<<B as Backend>::OwnedBuf, <B as Backend>::DftWord>;
/// Shared backend-native borrow of an `SvpPPol`.
pub type SvpPPolBackendRef<'a, B> = SvpPPol<<B as Backend>::BufRef<'a>, <B as Backend>::DftWord>;
/// Mutable backend-native borrow of an `SvpPPol`.
pub type SvpPPolBackendMut<'a, B> = SvpPPol<<B as Backend>::BufMut<'a>, <B as Backend>::DftWord>;

/// Reborrow a mutable backend-native `SvpPPol` view as a shared backend-native view.
pub fn svp_ppol_backend_ref_from_mut<'a, 'b, B: Backend>(ppol: &'a SvpPPolBackendMut<'b, B>) -> SvpPPolBackendRef<'a, B> {
    SvpPPol {
        data: B::view_ref_mut(&ppol.data),
        shape: ppol.shape,
        _phantom: PhantomData,
    }
}

/// Borrow a backend-owned `SvpPPol` using the backend's native view type.
pub trait SvpPPolToBackendRef<B: Backend> {
    fn to_backend_ref(&self) -> SvpPPolBackendRef<'_, B>;
}

impl<B: Backend> SvpPPolToBackendRef<B> for SvpPPol<B::OwnedBuf, B::DftWord> {
    fn to_backend_ref(&self) -> SvpPPolBackendRef<'_, B> {
        SvpPPol {
            data: B::view(&self.data),
            shape: self.shape,
            _phantom: PhantomData,
        }
    }
}

impl<'b, B: Backend + 'b> SvpPPolToBackendRef<B> for &SvpPPol<B::BufRef<'b>, B::DftWord> {
    fn to_backend_ref(&self) -> SvpPPolBackendRef<'_, B> {
        SvpPPol {
            data: B::view_ref(&self.data),
            shape: self.shape,
            _phantom: PhantomData,
        }
    }
}

/// Reborrow an already backend-borrowed `SvpPPol` as a shared backend-native view.
pub trait SvpPPolReborrowBackendRef<B: Backend> {
    fn reborrow_backend_ref(&self) -> SvpPPolBackendRef<'_, B>;
}

impl<'b, B: Backend + 'b> SvpPPolReborrowBackendRef<B> for SvpPPol<B::BufMut<'b>, B::DftWord> {
    fn reborrow_backend_ref(&self) -> SvpPPolBackendRef<'_, B> {
        svp_ppol_backend_ref_from_mut::<B>(self)
    }
}

/// Mutably borrow a backend-owned `SvpPPol` using the backend's native view type.
pub trait SvpPPolToBackendMut<B: Backend> {
    fn to_backend_mut(&mut self) -> SvpPPolBackendMut<'_, B>;
}

impl<B: Backend> SvpPPolToBackendMut<B> for SvpPPol<B::OwnedBuf, B::DftWord> {
    fn to_backend_mut(&mut self) -> SvpPPolBackendMut<'_, B> {
        SvpPPol {
            data: B::view_mut(&mut self.data),
            shape: self.shape,
            _phantom: PhantomData,
        }
    }
}

impl<'b, B: Backend + 'b> SvpPPolToBackendMut<B> for &mut SvpPPol<B::BufMut<'b>, B::DftWord> {
    fn to_backend_mut(&mut self) -> SvpPPolBackendMut<'_, B> {
        SvpPPol {
            data: B::view_mut_ref(&mut self.data),
            shape: self.shape,
            _phantom: PhantomData,
        }
    }
}

/// Reborrow an already backend-borrowed `SvpPPol` as a mutable backend-native view.
pub trait SvpPPolReborrowBackendMut<B: Backend> {
    fn reborrow_backend_mut(&mut self) -> SvpPPolBackendMut<'_, B>;
}

impl<'b, B: Backend + 'b> SvpPPolReborrowBackendMut<B> for SvpPPol<B::BufMut<'b>, B::DftWord> {
    fn reborrow_backend_mut(&mut self) -> SvpPPolBackendMut<'_, B> {
        SvpPPol {
            data: B::view_mut_ref(&mut self.data),
            shape: self.shape,
            _phantom: PhantomData,
        }
    }
}

impl<D: Data, W: DftWord> SvpPPol<D, W> {
    pub fn from_data(data: D, n: usize, cols: usize) -> Self {
        Self {
            data,
            shape: ScalarZnxShape::new(n, cols),
            _phantom: PhantomData,
        }
    }
}

impl<D: HostDataRef, W: DftWord> fmt::Display for SvpPPol<D, W> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "SvpPPol(n={}, cols={})", self.n(), self.cols())?;

        for col in 0..self.cols() {
            writeln!(f, "Column {col}:")?;
            let coeffs = self.at(col, 0);
            write!(f, "[")?;

            let max_show = 100;
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
        Ok(())
    }
}
