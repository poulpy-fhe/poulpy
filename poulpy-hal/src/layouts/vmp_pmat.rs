use std::{
    hash::{DefaultHasher, Hasher},
    marker::PhantomData,
};

use crate::layouts::{Backend, Data, DataView, DataViewMut, DftWord, DigestU64, HostDataRef, ZnxInfos, ZnxView};

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
/// columns and `size` limbs, all in the prepared representation named by
/// the [`DftWord`] type `W`.
///
/// Used as the right operand in
/// [`VmpApplyDftToDft`](crate::api::VmpApplyDftToDft). Create via
/// [`VmpPrepare`](crate::api::VmpPrepare) from a coefficient-domain
/// [`MatZnx`](crate::layouts::MatZnx).
///
/// Note that a backend may pack this matrix more compactly than
/// `size_of::<W>()` per coefficient; [`Backend::bytes_of_vmp_pmat`] is
/// authoritative for the buffer size.
///
/// Ring degree `n` is always a power of two, so each prepared polynomial's DFT
/// coefficient count matches vector lane widths relative to buffer alignment.
#[repr(C)]
#[derive(PartialEq, Eq, Hash)]
pub struct VmpPMat<D: Data, W: DftWord, B: Backend<DftWord = W>> {
    data: D,
    shape: VmpPMatShape,
    _phantom: PhantomData<(W, B)>,
}

impl<D: HostDataRef, W: DftWord, B: Backend<DftWord = W>> DigestU64 for VmpPMat<D, W, B> {
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

impl<D: HostDataRef, W: DftWord, B: Backend<DftWord = W>> ZnxView for VmpPMat<D, W, B> {
    type Scalar = W;
}

impl<D: Data, W: DftWord, B: Backend<DftWord = W>> ZnxInfos for VmpPMat<D, W, B> {
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

impl<D: Data, W: DftWord, B: Backend<DftWord = W>> DataView for VmpPMat<D, W, B> {
    type D = D;
    fn data(&self) -> &Self::D {
        &self.data
    }
}

impl<D: Data, W: DftWord, B: Backend<DftWord = W>> DataViewMut for VmpPMat<D, W, B> {
    fn data_mut(&mut self) -> &mut Self::D {
        &mut self.data
    }
}

impl<D: Data, W: DftWord, B: Backend<DftWord = W>> VmpPMat<D, W, B> {
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

impl<D: Data, W: DftWord, B: Backend<DftWord = W>> VmpPMat<D, W, B> {
    /// Allocates a zero-initialized backend-owned `VmpPMat`.
    pub fn alloc(n: usize, rows: usize, cols_in: usize, cols_out: usize, size: usize) -> VmpPMatOwned<B>
    where
        B: Backend<OwnedBuf = D>,
    {
        let data: <B as Backend>::OwnedBuf = B::alloc_zeroed_bytes(B::bytes_of_vmp_pmat(n, rows, cols_in, cols_out, size));
        VmpPMat {
            data,
            shape: VmpPMatShape::new(n, rows, cols_in, cols_out, size),
            _phantom: PhantomData,
        }
    }
}

/// Owned `VmpPMat` backed by a backend-owned buffer.
pub type VmpPMatOwned<B> = VmpPMat<<B as Backend>::OwnedBuf, <B as Backend>::DftWord, B>;
/// Immutably borrowed `VmpPMat`.
pub type VmpPMatRef<'a, B> = VmpPMat<&'a [u8], <B as Backend>::DftWord, B>;
/// Shared backend-native borrow of a `VmpPMat`.
pub type VmpPMatBackendRef<'a, B> = VmpPMat<<B as Backend>::BufRef<'a>, <B as Backend>::DftWord, B>;
/// Mutable backend-native borrow of a `VmpPMat`.
pub type VmpPMatBackendMut<'a, B> = VmpPMat<<B as Backend>::BufMut<'a>, <B as Backend>::DftWord, B>;

/// Reborrow an immutable backend-native `VmpPMat` view as a shared backend-native view.
pub fn vmp_pmat_backend_ref_from_ref<'a, 'b, B: Backend + 'b>(
    pmat: &'a VmpPMat<B::BufRef<'b>, B::DftWord, B>,
) -> VmpPMatBackendRef<'a, B> {
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

impl<B: Backend> VmpPMatToBackendRef<B> for VmpPMat<B::OwnedBuf, B::DftWord, B> {
    fn to_backend_ref(&self) -> VmpPMatBackendRef<'_, B> {
        VmpPMat {
            data: B::view(&self.data),
            shape: self.shape,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<'b, B: Backend + 'b> VmpPMatToBackendRef<B> for &VmpPMat<B::BufRef<'b>, B::DftWord, B> {
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

impl<'b, B: Backend + 'b> VmpPMatReborrowBackendRef<B> for VmpPMat<B::BufMut<'b>, B::DftWord, B> {
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

impl<B: Backend> VmpPMatToBackendMut<B> for VmpPMat<B::OwnedBuf, B::DftWord, B> {
    fn to_backend_mut(&mut self) -> VmpPMatBackendMut<'_, B> {
        VmpPMat {
            data: B::view_mut(&mut self.data),
            shape: self.shape,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<'b, B: Backend + 'b> VmpPMatToBackendMut<B> for &mut VmpPMat<B::BufMut<'b>, B::DftWord, B> {
    fn to_backend_mut(&mut self) -> VmpPMatBackendMut<'_, B> {
        vmp_pmat_backend_mut_from_mut::<B>(self)
    }
}

/// Reborrow an already backend-borrowed `VmpPMat` as a mutable backend-native view.
pub trait VmpPMatReborrowBackendMut<B: Backend> {
    fn reborrow_backend_mut(&mut self) -> VmpPMatBackendMut<'_, B>;
}

impl<'b, B: Backend + 'b> VmpPMatReborrowBackendMut<B> for VmpPMat<B::BufMut<'b>, B::DftWord, B> {
    fn reborrow_backend_mut(&mut self) -> VmpPMatBackendMut<'_, B> {
        vmp_pmat_backend_mut_from_mut::<B>(self)
    }
}

impl<D: Data, W: DftWord, B: Backend<DftWord = W>> VmpPMat<D, W, B> {
    pub fn from_data(data: D, n: usize, rows: usize, cols_in: usize, cols_out: usize, size: usize) -> Self {
        Self {
            data,
            shape: VmpPMatShape::new(n, rows, cols_in, cols_out, size),
            _phantom: PhantomData,
        }
    }
}

impl<D: Data, W: DftWord, B: Backend<DftWord = W>> VmpPMat<D, W, B> {
    /// Zero-copy re-tag of this container to a layout-compatible backend `B2`.
    ///
    /// The buffer moves as-is; only the type tag changes. Requires the
    /// [`VmpPMatLayoutCompatible`](crate::layouts::VmpPMatLayoutCompatible) marker declared by the backend
    /// pair. `D` is kept, so for further backend-native use `B2`'s buffer
    /// types must match `D` (true for all current CPU backends).
    pub fn into_backend<B2>(self) -> VmpPMat<D, W, B2>
    where
        B2: Backend<DftWord = W>,
        B: crate::layouts::VmpPMatLayoutCompatible<B2>,
    {
        let shape = self.shape;
        assert_eq!(
            B::bytes_of_vmp_pmat(shape.n(), shape.rows(), shape.cols_in(), shape.cols_out(), shape.size()),
            B2::bytes_of_vmp_pmat(shape.n(), shape.rows(), shape.cols_in(), shape.cols_out(), shape.size()),
            "into_backend: byte sizes diverge despite declared layout compatibility"
        );
        VmpPMat {
            data: self.data,
            shape,
            _phantom: PhantomData,
        }
    }
}
