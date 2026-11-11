use std::{
    hash::{DefaultHasher, Hasher},
    marker::PhantomData,
};

use crate::layouts::{
    Backend, Data, DataView, DataViewMut, DftWord, DigestU64, HostDataMut, HostDataRef, MatZnxInfos, PrepareHint, Ring, Standard,
    ZnxInfos,
};

#[repr(C)]
#[derive(PartialEq, Eq, Clone, Copy, Hash, Debug, Default)]
pub struct VmpPMatShape {
    n: usize,
    size: usize,
    rows: usize,
    cols_in: usize,
    cols_out: usize,
    hint: PrepareHint,
}

impl VmpPMatShape {
    pub const fn new(n: usize, rows: usize, cols_in: usize, cols_out: usize, size: usize, hint: PrepareHint) -> Self {
        Self {
            n,
            size,
            rows,
            cols_in,
            cols_out,
            hint,
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

    pub const fn hint(self) -> PrepareHint {
        self.hint
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
/// [`VmpPMatPrepare`](crate::api::VmpPMatPrepare) from a coefficient-domain
/// [`MatZnx`](crate::layouts::MatZnx).
///
/// Note that a backend may pack this matrix more compactly than
/// `size_of::<W>()` per coefficient; [`Backend::vmp_pmat_bytes_of`] is
/// authoritative for the buffer size.
///
/// Ring degree `n` is always a power of two, so each prepared polynomial's DFT
/// coefficient count matches vector lane widths relative to buffer alignment.
///
/// Denotes `prep(M)` in the [layouts value model](crate::layouts#value-model):
/// an opaque value fixed only by the equations `vmp_apply` satisfies, in the
/// representation the [`PrepareHint`] chosen at allocation selected (see
/// [`VmpPMatShape::hint`]).
#[repr(C)]
pub struct VmpPMat<D: Data, W: DftWord, B: Backend<DftWord = W>, Rg: Ring = Standard> {
    data: D,
    shape: VmpPMatShape,
    _phantom: PhantomData<(W, B, Rg)>,
}

// Equality (and hashing, where provided) is defined directly on the
// representation: same shape, same buffer bytes. No `W`/`B` value is ever
// compared, so no bound on them is needed — in particular `Eq` holds even
// for non-`Eq` words like `f64` (byte equality is a total equivalence).
impl<D: Data, W: DftWord, B: Backend<DftWord = W>, Rg: Ring> PartialEq for VmpPMat<D, W, B, Rg> {
    fn eq(&self, other: &Self) -> bool {
        self.shape == other.shape && self.data == other.data
    }
}

impl<D: Data, W: DftWord, B: Backend<DftWord = W>, Rg: Ring> Eq for VmpPMat<D, W, B, Rg> {}

/// The shape only: the representation is opaque.
impl<D: Data, W: DftWord, B: Backend<DftWord = W>, Rg: Ring> std::fmt::Debug for VmpPMat<D, W, B, Rg> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VmpPMat").field("shape", &self.shape).finish()
    }
}

impl<D: Data + std::hash::Hash, W: DftWord, B: Backend<DftWord = W>, Rg: Ring> std::hash::Hash for VmpPMat<D, W, B, Rg> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.shape.hash(state);
        self.data.hash(state);
    }
}

impl<D: HostDataRef, W: DftWord, B: Backend<DftWord = W>, Rg: Ring> DigestU64 for VmpPMat<D, W, B, Rg> {
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

impl<D: HostDataRef, W: DftWord, B: Backend<DftWord = W>, Rg: Ring> VmpPMat<D, W, B, Rg> {
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

impl<D: HostDataMut, W: DftWord, B: Backend<DftWord = W>, Rg: Ring> VmpPMat<D, W, B, Rg> {
    /// Mutable counterpart of [`Self::raw`].
    pub fn raw_mut(&mut self) -> &mut [W] {
        let span: usize = crate::layouts::element_view_span(self);
        crate::layouts::raw_scalars_mut(self.data.as_mut(), span)
    }
}

impl<D: Data, W: DftWord, B: Backend<DftWord = W>, Rg: Ring> ZnxInfos for VmpPMat<D, W, B, Rg> {
    fn n(&self) -> usize {
        self.shape.n()
    }

    fn size(&self) -> usize {
        self.shape.size()
    }

    fn poly_count(&self) -> usize {
        crate::layouts::checked_product(
            &[self.rows(), self.cols_in(), self.size(), self.cols_out()],
            "VmpPMat polynomial count",
        )
    }
}

impl<D: Data, W: DftWord, B: Backend<DftWord = W>, Rg: Ring> MatZnxInfos for VmpPMat<D, W, B, Rg> {
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

impl<D: Data, W: DftWord, B: Backend<DftWord = W>, Rg: Ring> DataView for VmpPMat<D, W, B, Rg> {
    type D = D;
    fn data(&self) -> &Self::D {
        &self.data
    }
}

impl<D: Data, W: DftWord, B: Backend<DftWord = W>, Rg: Ring> DataViewMut for VmpPMat<D, W, B, Rg> {
    fn data_mut(&mut self) -> &mut Self::D {
        &mut self.data
    }
}

impl<D: Data, W: DftWord, B: Backend<DftWord = W>, Rg: Ring> VmpPMat<D, W, B, Rg> {
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

    pub fn hint(&self) -> PrepareHint {
        self.shape.hint()
    }
}

impl<D: Data, W: DftWord, B: Backend<DftWord = W>, Rg: Ring> VmpPMat<D, W, B, Rg> {
    /// Allocates a zero-initialized backend-owned `VmpPMat`.
    pub fn alloc(n: usize, rows: usize, cols_in: usize, cols_out: usize, size: usize, hint: PrepareHint) -> VmpPMatOwned<B, Rg>
    where
        B: Backend<OwnedBuf = D>,
    {
        let data: <B as Backend>::OwnedBuf = B::alloc_zeroed_bytes(B::vmp_pmat_bytes_of(n, rows, cols_in, cols_out, size, hint));
        VmpPMat {
            data,
            shape: VmpPMatShape::new(n, rows, cols_in, cols_out, size, hint),
            _phantom: PhantomData,
        }
    }
}

/// Owned `VmpPMat` backed by a backend-owned buffer.
pub type VmpPMatOwned<B, Rg = Standard> = VmpPMat<<B as Backend>::OwnedBuf, <B as Backend>::DftWord, B, Rg>;
/// Immutably borrowed `VmpPMat`.
pub type VmpPMatRef<'a, B, Rg = Standard> = VmpPMat<&'a [u8], <B as Backend>::DftWord, B, Rg>;
/// Shared backend-native borrow of a `VmpPMat`.
pub type VmpPMatBackendRef<'a, B, Rg = Standard> = VmpPMat<<B as Backend>::BufRef<'a>, <B as Backend>::DftWord, B, Rg>;
/// Mutable backend-native borrow of a `VmpPMat`.
pub type VmpPMatBackendMut<'a, B, Rg = Standard> = VmpPMat<<B as Backend>::BufMut<'a>, <B as Backend>::DftWord, B, Rg>;

/// Reborrow an immutable backend-native `VmpPMat` view as a shared backend-native view.
pub fn vmp_pmat_backend_ref_from_ref<'a, 'b, B: Backend + 'b, Rg: Ring>(
    pmat: &'a VmpPMat<B::BufRef<'b>, B::DftWord, B, Rg>,
) -> VmpPMatBackendRef<'a, B, Rg> {
    VmpPMat {
        data: B::view_ref(&pmat.data),
        shape: pmat.shape,
        _phantom: PhantomData,
    }
}

/// Reborrow a mutable backend-native `VmpPMat` view as a shared backend-native view.
pub fn vmp_pmat_backend_ref_from_mut<'a, B: Backend, Rg: Ring>(
    pmat: &'a VmpPMatBackendMut<'a, B, Rg>,
) -> VmpPMatBackendRef<'a, B, Rg> {
    VmpPMat {
        data: B::view_ref_mut(&pmat.data),
        shape: pmat.shape,
        _phantom: PhantomData,
    }
}

pub fn vmp_pmat_backend_mut_from_mut<'a, 'b, B: Backend + 'b, Rg: Ring>(
    pmat: &'a mut VmpPMatBackendMut<'b, B, Rg>,
) -> VmpPMatBackendMut<'a, B, Rg> {
    VmpPMat {
        data: B::view_mut_ref(&mut pmat.data),
        shape: pmat.shape,
        _phantom: PhantomData,
    }
}

/// Borrow a backend-owned `VmpPMat` using the backend's native view type.
pub trait VmpPMatToBackendRef<B: Backend, Rg: Ring = Standard> {
    fn to_backend_ref(&self) -> VmpPMatBackendRef<'_, B, Rg>;
}

impl<B: Backend, Rg: Ring> VmpPMatToBackendRef<B, Rg> for VmpPMat<B::OwnedBuf, B::DftWord, B, Rg> {
    fn to_backend_ref(&self) -> VmpPMatBackendRef<'_, B, Rg> {
        VmpPMat {
            data: B::view(&self.data),
            shape: self.shape,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<B: Backend, Rg: Ring> VmpPMatToBackendRef<B, Rg> for &VmpPMat<B::BufRef<'_>, B::DftWord, B, Rg> {
    fn to_backend_ref(&self) -> VmpPMatBackendRef<'_, B, Rg> {
        VmpPMat {
            data: B::view_ref(&self.data),
            shape: self.shape,
            _phantom: PhantomData,
        }
    }
}

/// Reborrow an already backend-borrowed `VmpPMat` as a shared backend-native view.
pub trait VmpPMatReborrowBackendRef<B: Backend, Rg: Ring = Standard> {
    fn reborrow_backend_ref(&self) -> VmpPMatBackendRef<'_, B, Rg>;
}

impl<B: Backend, Rg: Ring> VmpPMatReborrowBackendRef<B, Rg> for VmpPMat<B::BufMut<'_>, B::DftWord, B, Rg> {
    fn reborrow_backend_ref(&self) -> VmpPMatBackendRef<'_, B, Rg> {
        VmpPMat {
            data: B::view_ref_mut(&self.data),
            shape: self.shape,
            _phantom: std::marker::PhantomData,
        }
    }
}

/// Mutably borrow a backend-owned `VmpPMat` using the backend's native view type.
pub trait VmpPMatToBackendMut<B: Backend, Rg: Ring = Standard> {
    fn to_backend_mut(&mut self) -> VmpPMatBackendMut<'_, B, Rg>;
}

impl<B: Backend, Rg: Ring> VmpPMatToBackendMut<B, Rg> for VmpPMat<B::OwnedBuf, B::DftWord, B, Rg> {
    fn to_backend_mut(&mut self) -> VmpPMatBackendMut<'_, B, Rg> {
        VmpPMat {
            data: B::view_mut(&mut self.data),
            shape: self.shape,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<B: Backend, Rg: Ring> VmpPMatToBackendMut<B, Rg> for &mut VmpPMat<B::BufMut<'_>, B::DftWord, B, Rg> {
    fn to_backend_mut(&mut self) -> VmpPMatBackendMut<'_, B, Rg> {
        vmp_pmat_backend_mut_from_mut::<B, Rg>(self)
    }
}

/// Reborrow an already backend-borrowed `VmpPMat` as a mutable backend-native view.
pub trait VmpPMatReborrowBackendMut<B: Backend, Rg: Ring = Standard> {
    fn reborrow_backend_mut(&mut self) -> VmpPMatBackendMut<'_, B, Rg>;
}

impl<B: Backend, Rg: Ring> VmpPMatReborrowBackendMut<B, Rg> for VmpPMat<B::BufMut<'_>, B::DftWord, B, Rg> {
    fn reborrow_backend_mut(&mut self) -> VmpPMatBackendMut<'_, B, Rg> {
        vmp_pmat_backend_mut_from_mut::<B, Rg>(self)
    }
}

impl<D: Data, W: DftWord, B: Backend<DftWord = W>, Rg: Ring> VmpPMat<D, W, B, Rg> {
    pub fn from_data(data: D, n: usize, rows: usize, cols_in: usize, cols_out: usize, size: usize, hint: PrepareHint) -> Self {
        Self {
            data,
            shape: VmpPMatShape::new(n, rows, cols_in, cols_out, size, hint),
            _phantom: PhantomData,
        }
    }
}

impl<D: Data, W: DftWord, B: Backend<DftWord = W>, Rg: Ring> VmpPMat<D, W, B, Rg> {
    /// Zero-copy re-tag of this container to a layout-compatible backend `B2`.
    ///
    /// The buffer moves as-is; only the type tag changes. Requires the
    /// [`VmpPMatLayoutCompatible`](crate::layouts::VmpPMatLayoutCompatible) marker declared by the backend
    /// pair. `D` is kept, so for further backend-native use `B2`'s buffer
    /// types must match `D` (true for every backend in the workspace).
    pub fn into_backend<B2>(self) -> VmpPMat<D, W, B2, Rg>
    where
        B2: Backend<DftWord = W>,
        B: crate::layouts::VmpPMatLayoutCompatible<B2>,
    {
        let shape = self.shape;
        assert_eq!(
            B::vmp_pmat_bytes_of(
                shape.n(),
                shape.rows(),
                shape.cols_in(),
                shape.cols_out(),
                shape.size(),
                shape.hint()
            ),
            B2::vmp_pmat_bytes_of(
                shape.n(),
                shape.rows(),
                shape.cols_in(),
                shape.cols_out(),
                shape.size(),
                shape.hint()
            ),
            "into_backend: byte sizes diverge despite declared layout compatibility"
        );
        VmpPMat {
            data: self.data,
            shape,
            _phantom: PhantomData,
        }
    }
}
