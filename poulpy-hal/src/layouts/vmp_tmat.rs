use std::{
    hash::{DefaultHasher, Hasher},
    marker::PhantomData,
};

use crate::layouts::{Backend, Data, DataView, DataViewMut, DftWord, DigestU64, HostDataMut, HostDataRef, MatZnxInfos, ZnxInfos};

#[repr(C)]
#[derive(PartialEq, Eq, Clone, Copy, Hash, Debug, Default)]
pub struct VmpTMatShape {
    n: usize,
    size: usize,
    rows: usize,
    cols_in: usize,
    cols_out: usize,
}

impl VmpTMatShape {
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

/// Transformed (hot-prep) polynomial matrix for vector-matrix products.
///
/// A `VmpTMat` holds the prepared form of a `rows * cols_in` by `cols_out`
/// matrix of `size`-limb polynomials, in the representation named by the
/// [`DftWord`] type `W`.
///
/// The internal arrangement is entirely backend-defined: a backend is free to
/// interleave, transpose or repack, and nothing outside that backend's own
/// kernels may assume an entry layout. [`Backend::bytes_of_vmp_tmat`] is
/// authoritative for the buffer size, and cross-backend reinterpretation is
/// gated on [`VmpTMatLayoutCompatible`](crate::layouts::VmpTMatLayoutCompatible).
///
/// `VmpTMat` is the cheap-to-build prepared form, meant for short reuse;
/// [`VmpPMat`](crate::layouts::VmpPMat) is the packed form, more expensive to
/// build but optimized for amortized repeated apply. The two are distinct types
/// even where a backend gives them the same physical storage shape.
///
/// Create via [`VmpPrepareTMat`](crate::api::VmpPrepareTMat) from a
/// coefficient-domain [`MatZnx`](crate::layouts::MatZnx), then consume through
/// the `vmp_apply_tmat_*` family.
#[repr(C)]
pub struct VmpTMat<D: Data, W: DftWord, B: Backend<DftWord = W>> {
    data: D,
    shape: VmpTMatShape,
    _phantom: PhantomData<(W, B)>,
}

// Equality (and hashing, where provided) is defined directly on the
// representation: same shape, same buffer bytes. No `W`/`B` value is ever
// compared, so no bound on them is needed — in particular `Eq` holds even
// for non-`Eq` words like `f64` (byte equality is a total equivalence).
impl<D: Data, W: DftWord, B: Backend<DftWord = W>> PartialEq for VmpTMat<D, W, B> {
    fn eq(&self, other: &Self) -> bool {
        self.shape == other.shape && self.data == other.data
    }
}

impl<D: Data, W: DftWord, B: Backend<DftWord = W>> Eq for VmpTMat<D, W, B> {}

impl<D: Data + std::hash::Hash, W: DftWord, B: Backend<DftWord = W>> std::hash::Hash for VmpTMat<D, W, B> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.shape.hash(state);
        self.data.hash(state);
    }
}

impl<D: HostDataRef, W: DftWord, B: Backend<DftWord = W>> DigestU64 for VmpTMat<D, W, B> {
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

impl<D: HostDataRef, W: DftWord, B: Backend<DftWord = W>> VmpTMat<D, W, B> {
    /// Returns the whole element view as a scalar slice.
    ///
    /// The prepared matrix is packed in a backend-defined order with no flat
    /// `(col, limb)` indexing, so it exposes the buffer rather than
    /// implementing [`ZnxView`](crate::layouts::ZnxView).
    pub fn raw(&self) -> &[W] {
        let span: usize = crate::layouts::element_view_span(self);
        crate::layouts::raw_scalars(self.data.as_ref(), span)
    }
}

impl<D: HostDataMut, W: DftWord, B: Backend<DftWord = W>> VmpTMat<D, W, B> {
    /// Mutable counterpart of [`Self::raw`].
    pub fn raw_mut(&mut self) -> &mut [W] {
        let span: usize = crate::layouts::element_view_span(self);
        crate::layouts::raw_scalars_mut(self.data.as_mut(), span)
    }
}

impl<D: Data, W: DftWord, B: Backend<DftWord = W>> ZnxInfos for VmpTMat<D, W, B> {
    fn n(&self) -> usize {
        self.shape.n()
    }

    fn size(&self) -> usize {
        self.shape.size()
    }

    fn poly_count(&self) -> usize {
        crate::layouts::checked_product(
            &[self.rows(), self.cols_in(), self.size(), self.cols_out()],
            "VmpTMat polynomial count",
        )
    }
}

impl<D: Data, W: DftWord, B: Backend<DftWord = W>> MatZnxInfos for VmpTMat<D, W, B> {
    fn rows(&self) -> usize {
        self.shape.rows()
    }

    fn cols_in(&self) -> usize {
        self.shape.cols_in()
    }

    fn cols_out(&self) -> usize {
        self.shape.cols_out()
    }
}

impl<D: Data, W: DftWord, B: Backend<DftWord = W>> DataView for VmpTMat<D, W, B> {
    type D = D;
    fn data(&self) -> &Self::D {
        &self.data
    }
}

impl<D: Data, W: DftWord, B: Backend<DftWord = W>> DataViewMut for VmpTMat<D, W, B> {
    fn data_mut(&mut self) -> &mut Self::D {
        &mut self.data
    }
}

impl<D: Data, W: DftWord, B: Backend<DftWord = W>> VmpTMat<D, W, B> {
    pub fn shape(&self) -> VmpTMatShape {
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

impl<D: Data, W: DftWord, B: Backend<DftWord = W>> VmpTMat<D, W, B> {
    /// Allocates a zero-initialized backend-owned `VmpTMat`.
    pub fn alloc(n: usize, rows: usize, cols_in: usize, cols_out: usize, size: usize) -> VmpTMatOwned<B>
    where
        B: Backend<OwnedBuf = D>,
    {
        let data: <B as Backend>::OwnedBuf = B::alloc_zeroed_bytes(B::bytes_of_vmp_tmat(n, rows, cols_in, cols_out, size));
        VmpTMat {
            data,
            shape: VmpTMatShape::new(n, rows, cols_in, cols_out, size),
            _phantom: PhantomData,
        }
    }
}

/// Owned `VmpTMat` backed by a backend-owned buffer.
pub type VmpTMatOwned<B> = VmpTMat<<B as Backend>::OwnedBuf, <B as Backend>::DftWord, B>;
/// Immutably borrowed `VmpTMat`.
pub type VmpTMatRef<'a, B> = VmpTMat<&'a [u8], <B as Backend>::DftWord, B>;
/// Shared backend-native borrow of a `VmpTMat`.
pub type VmpTMatBackendRef<'a, B> = VmpTMat<<B as Backend>::BufRef<'a>, <B as Backend>::DftWord, B>;
/// Mutable backend-native borrow of a `VmpTMat`.
pub type VmpTMatBackendMut<'a, B> = VmpTMat<<B as Backend>::BufMut<'a>, <B as Backend>::DftWord, B>;

/// Reborrow an immutable backend-native `VmpTMat` view as a shared backend-native view.
pub fn vmp_tmat_backend_ref_from_ref<'a, 'b, B: Backend + 'b>(
    pmat: &'a VmpTMat<B::BufRef<'b>, B::DftWord, B>,
) -> VmpTMatBackendRef<'a, B> {
    VmpTMat {
        data: B::view_ref(&pmat.data),
        shape: pmat.shape,
        _phantom: PhantomData,
    }
}

/// Reborrow a mutable backend-native `VmpTMat` view as a shared backend-native view.
pub fn vmp_tmat_backend_ref_from_mut<'a, B: Backend>(pmat: &'a VmpTMatBackendMut<'a, B>) -> VmpTMatBackendRef<'a, B> {
    VmpTMat {
        data: B::view_ref_mut(&pmat.data),
        shape: pmat.shape,
        _phantom: PhantomData,
    }
}

pub fn vmp_tmat_backend_mut_from_mut<'a, 'b, B: Backend + 'b>(
    pmat: &'a mut VmpTMatBackendMut<'b, B>,
) -> VmpTMatBackendMut<'a, B> {
    VmpTMat {
        data: B::view_mut_ref(&mut pmat.data),
        shape: pmat.shape,
        _phantom: PhantomData,
    }
}

/// Borrow a backend-owned `VmpTMat` using the backend's native view type.
pub trait VmpTMatToBackendRef<B: Backend> {
    fn to_backend_ref(&self) -> VmpTMatBackendRef<'_, B>;
}

impl<B: Backend> VmpTMatToBackendRef<B> for VmpTMat<B::OwnedBuf, B::DftWord, B> {
    fn to_backend_ref(&self) -> VmpTMatBackendRef<'_, B> {
        VmpTMat {
            data: B::view(&self.data),
            shape: self.shape,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<'b, B: Backend + 'b> VmpTMatToBackendRef<B> for &VmpTMat<B::BufRef<'b>, B::DftWord, B> {
    fn to_backend_ref(&self) -> VmpTMatBackendRef<'_, B> {
        VmpTMat {
            data: B::view_ref(&self.data),
            shape: self.shape,
            _phantom: PhantomData,
        }
    }
}

/// Reborrow an already backend-borrowed `VmpTMat` as a shared backend-native view.
pub trait VmpTMatReborrowBackendRef<B: Backend> {
    fn reborrow_backend_ref(&self) -> VmpTMatBackendRef<'_, B>;
}

impl<'b, B: Backend + 'b> VmpTMatReborrowBackendRef<B> for VmpTMat<B::BufMut<'b>, B::DftWord, B> {
    fn reborrow_backend_ref(&self) -> VmpTMatBackendRef<'_, B> {
        VmpTMat {
            data: B::view_ref_mut(&self.data),
            shape: self.shape,
            _phantom: std::marker::PhantomData,
        }
    }
}

/// Mutably borrow a backend-owned `VmpTMat` using the backend's native view type.
pub trait VmpTMatToBackendMut<B: Backend> {
    fn to_backend_mut(&mut self) -> VmpTMatBackendMut<'_, B>;
}

impl<B: Backend> VmpTMatToBackendMut<B> for VmpTMat<B::OwnedBuf, B::DftWord, B> {
    fn to_backend_mut(&mut self) -> VmpTMatBackendMut<'_, B> {
        VmpTMat {
            data: B::view_mut(&mut self.data),
            shape: self.shape,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<'b, B: Backend + 'b> VmpTMatToBackendMut<B> for &mut VmpTMat<B::BufMut<'b>, B::DftWord, B> {
    fn to_backend_mut(&mut self) -> VmpTMatBackendMut<'_, B> {
        vmp_tmat_backend_mut_from_mut::<B>(self)
    }
}

/// Reborrow an already backend-borrowed `VmpTMat` as a mutable backend-native view.
pub trait VmpTMatReborrowBackendMut<B: Backend> {
    fn reborrow_backend_mut(&mut self) -> VmpTMatBackendMut<'_, B>;
}

impl<'b, B: Backend + 'b> VmpTMatReborrowBackendMut<B> for VmpTMat<B::BufMut<'b>, B::DftWord, B> {
    fn reborrow_backend_mut(&mut self) -> VmpTMatBackendMut<'_, B> {
        vmp_tmat_backend_mut_from_mut::<B>(self)
    }
}

impl<D: Data, W: DftWord, B: Backend<DftWord = W>> VmpTMat<D, W, B> {
    pub fn from_data(data: D, n: usize, rows: usize, cols_in: usize, cols_out: usize, size: usize) -> Self {
        Self {
            data,
            shape: VmpTMatShape::new(n, rows, cols_in, cols_out, size),
            _phantom: PhantomData,
        }
    }
}

impl<D: Data, W: DftWord, B: Backend<DftWord = W>> VmpTMat<D, W, B> {
    /// Zero-copy re-tag of this container to a layout-compatible backend `B2`.
    ///
    /// The buffer moves as-is; only the type tag changes. Requires the
    /// [`VmpTMatLayoutCompatible`](crate::layouts::VmpTMatLayoutCompatible) marker declared by the backend
    /// pair. `D` is kept, so for further backend-native use `B2`'s buffer
    /// types must match `D` (true for all current CPU backends).
    pub fn into_backend<B2>(self) -> VmpTMat<D, W, B2>
    where
        B2: Backend<DftWord = W>,
        B: crate::layouts::VmpTMatLayoutCompatible<B2>,
    {
        let shape = self.shape;
        assert_eq!(
            B::bytes_of_vmp_tmat(shape.n(), shape.rows(), shape.cols_in(), shape.cols_out(), shape.size()),
            B2::bytes_of_vmp_tmat(shape.n(), shape.rows(), shape.cols_in(), shape.cols_out(), shape.size()),
            "into_backend: byte sizes diverge despite declared layout compatibility"
        );
        VmpTMat {
            data: self.data,
            shape,
            _phantom: PhantomData,
        }
    }
}
