use std::marker::PhantomData;

use crate::AlignedBuf;
use crate::layouts::{
    Backend, Data, DataView, DataViewMut, DftWord, HostDataRef, PrepareHint, Ring, Standard, VecZnxInfos, ZnxInfos, ZnxView,
};

#[repr(C)]
#[derive(PartialEq, Eq, Clone, Copy, Hash, Debug, Default)]
pub struct CnvPVecShape {
    n: usize,
    size: usize,
    cols: usize,
    hint: PrepareHint,
}

impl CnvPVecShape {
    pub const fn new(n: usize, cols: usize, size: usize, hint: PrepareHint) -> Self {
        Self { n, size, cols, hint }
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

    pub const fn hint(self) -> PrepareHint {
        self.hint
    }
}

/// Prepared right operand for bivariate convolution.
///
/// Holds a polynomial vector in the prepared representation named by the
/// [`DftWord`] type `W`, ready to be used as the right operand of
/// [`CnvOps::cnv_apply_dft`](crate::api::CnvOps::cnv_apply_dft).
/// Created via [`CnvOps::cnv_pvec_right_prepare`](crate::api::CnvOps::cnv_pvec_right_prepare).
///
/// A prepared value in the [layouts value model](crate::layouts#value-model),
/// opaque in the representation the [`PrepareHint`] chosen at allocation
/// selected (see [`CnvPVecShape::hint`]).
pub struct CnvPVecR<D: Data, W: DftWord, B: Backend<DftWord = W>, Rg: Ring = Standard> {
    data: D,
    shape: CnvPVecShape,
    _phantom: PhantomData<(W, B, Rg)>,
}

// Same shape, same buffer bytes, as for `VmpPMat`.
impl<D: Data, W: DftWord, B: Backend<DftWord = W>, Rg: Ring> PartialEq for CnvPVecR<D, W, B, Rg> {
    fn eq(&self, other: &Self) -> bool {
        self.shape == other.shape && self.data == other.data
    }
}

impl<D: Data, W: DftWord, B: Backend<DftWord = W>, Rg: Ring> Eq for CnvPVecR<D, W, B, Rg> {}

/// The shape only: the representation is opaque.
impl<D: Data, W: DftWord, B: Backend<DftWord = W>, Rg: Ring> std::fmt::Debug for CnvPVecR<D, W, B, Rg> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CnvPVecR").field("shape", &self.shape).finish()
    }
}

impl<D: Data, W: DftWord, B: Backend<DftWord = W>, Rg: Ring> ZnxInfos for CnvPVecR<D, W, B, Rg> {
    fn n(&self) -> usize {
        self.shape.n()
    }

    fn size(&self) -> usize {
        self.shape.size()
    }

    fn poly_count(&self) -> usize {
        crate::layouts::checked_product(&[self.cols(), self.size()], "polynomial count")
    }
}

impl<D: Data, W: DftWord, B: Backend<DftWord = W>, Rg: Ring> VecZnxInfos for CnvPVecR<D, W, B, Rg> {
    fn cols(&self) -> usize {
        self.shape.cols()
    }
}

impl<D: Data, W: DftWord, B: Backend<DftWord = W>, Rg: Ring> DataView for CnvPVecR<D, W, B, Rg> {
    type D = D;
    fn data(&self) -> &Self::D {
        &self.data
    }
}

impl<D: Data, W: DftWord, B: Backend<DftWord = W>, Rg: Ring> DataViewMut for CnvPVecR<D, W, B, Rg> {
    fn data_mut(&mut self) -> &mut Self::D {
        &mut self.data
    }
}

impl<D: HostDataRef, W: DftWord, B: Backend<DftWord = W>, Rg: Ring> ZnxView for CnvPVecR<D, W, B, Rg> {
    type Scalar = W;
}

impl<D: Data, W: DftWord, B: Backend<DftWord = W>, Rg: Ring> CnvPVecR<D, W, B, Rg> {
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

    pub fn hint(&self) -> PrepareHint {
        self.shape.hint()
    }
}

impl<D: Data, W: DftWord, B: Backend<DftWord = W>, Rg: Ring> CnvPVecR<D, W, B, Rg> {
    /// Allocates a zero-initialized backend-owned `CnvPVecR`.
    pub fn alloc(n: usize, cols: usize, size: usize, hint: PrepareHint) -> CnvPVecR<B::OwnedBuf, W, B, Rg>
    where
        B: Backend<OwnedBuf = D>,
    {
        let data: B::OwnedBuf = B::alloc_zeroed_bytes(B::cnv_pvec_right_bytes_of(n, cols, size, hint));
        CnvPVecR {
            data,
            shape: CnvPVecShape::new(n, cols, size, hint),
            _phantom: PhantomData,
        }
    }

    /// Uploads a host byte buffer into a backend-owned `CnvPVecR`.
    ///
    /// # Panics
    ///
    /// Panics if the buffer length does not equal `B::cnv_pvec_right_bytes_of(n, cols, size, hint)`;
    /// a `Vec<u8>` argument is first copied into storage padded to a 64-byte multiple, so
    /// its padded length is what is compared.
    pub fn from_bytes(
        n: usize,
        cols: usize,
        size: usize,
        hint: PrepareHint,
        bytes: impl Into<AlignedBuf>,
    ) -> CnvPVecR<B::OwnedBuf, W, B, Rg>
    where
        B: Backend<OwnedBuf = D>,
    {
        let data: AlignedBuf = bytes.into();
        assert!(data.len() == crate::layouts::padded_bytes(B::cnv_pvec_right_bytes_of(n, cols, size, hint)));
        let data: B::OwnedBuf = B::from_host_bytes(&data);
        CnvPVecR {
            data,
            shape: CnvPVecShape::new(n, cols, size, hint),
            _phantom: PhantomData,
        }
    }
}

impl<D: Data, W: DftWord, B: Backend<DftWord = W>, Rg: Ring> CnvPVecR<D, W, B, Rg> {
    pub fn from_data(data: D, n: usize, cols: usize, size: usize, hint: PrepareHint) -> Self {
        Self {
            data,
            shape: CnvPVecShape::new(n, cols, size, hint),
            _phantom: PhantomData,
        }
    }
}

/// Prepared left operand for bivariate convolution.
///
/// Holds a polynomial vector in the prepared representation named by the
/// [`DftWord`] type `W`, ready to be used as the left operand of
/// [`CnvOps::cnv_apply_dft`](crate::api::CnvOps::cnv_apply_dft).
/// Created via [`CnvOps::cnv_pvec_left_prepare`](crate::api::CnvOps::cnv_pvec_left_prepare).
///
/// A prepared value in the [layouts value model](crate::layouts#value-model),
/// opaque in the representation the [`PrepareHint`] chosen at allocation
/// selected (see [`CnvPVecShape::hint`]).
pub struct CnvPVecL<D: Data, W: DftWord, B: Backend<DftWord = W>, Rg: Ring = Standard> {
    data: D,
    shape: CnvPVecShape,
    _phantom: PhantomData<(W, B, Rg)>,
}

// Same shape, same buffer bytes, as for `VmpPMat`.
impl<D: Data, W: DftWord, B: Backend<DftWord = W>, Rg: Ring> PartialEq for CnvPVecL<D, W, B, Rg> {
    fn eq(&self, other: &Self) -> bool {
        self.shape == other.shape && self.data == other.data
    }
}

impl<D: Data, W: DftWord, B: Backend<DftWord = W>, Rg: Ring> Eq for CnvPVecL<D, W, B, Rg> {}

/// The shape only: the representation is opaque.
impl<D: Data, W: DftWord, B: Backend<DftWord = W>, Rg: Ring> std::fmt::Debug for CnvPVecL<D, W, B, Rg> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CnvPVecL").field("shape", &self.shape).finish()
    }
}

impl<D: Data, W: DftWord, B: Backend<DftWord = W>, Rg: Ring> ZnxInfos for CnvPVecL<D, W, B, Rg> {
    fn n(&self) -> usize {
        self.shape.n()
    }

    fn size(&self) -> usize {
        self.shape.size()
    }

    fn poly_count(&self) -> usize {
        crate::layouts::checked_product(&[self.cols(), self.size()], "polynomial count")
    }
}

impl<D: Data, W: DftWord, B: Backend<DftWord = W>, Rg: Ring> VecZnxInfos for CnvPVecL<D, W, B, Rg> {
    fn cols(&self) -> usize {
        self.shape.cols()
    }
}

impl<D: Data, W: DftWord, B: Backend<DftWord = W>, Rg: Ring> DataView for CnvPVecL<D, W, B, Rg> {
    type D = D;
    fn data(&self) -> &Self::D {
        &self.data
    }
}

impl<D: Data, W: DftWord, B: Backend<DftWord = W>, Rg: Ring> DataViewMut for CnvPVecL<D, W, B, Rg> {
    fn data_mut(&mut self) -> &mut Self::D {
        &mut self.data
    }
}

impl<D: HostDataRef, W: DftWord, B: Backend<DftWord = W>, Rg: Ring> ZnxView for CnvPVecL<D, W, B, Rg> {
    type Scalar = W;
}

impl<D: Data, W: DftWord, B: Backend<DftWord = W>, Rg: Ring> CnvPVecL<D, W, B, Rg> {
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

    pub fn hint(&self) -> PrepareHint {
        self.shape.hint()
    }
}

impl<D: Data, W: DftWord, B: Backend<DftWord = W>, Rg: Ring> CnvPVecL<D, W, B, Rg> {
    /// Allocates a zero-initialized backend-owned `CnvPVecL`.
    pub fn alloc(n: usize, cols: usize, size: usize, hint: PrepareHint) -> CnvPVecL<B::OwnedBuf, W, B, Rg>
    where
        B: Backend<OwnedBuf = D>,
    {
        let data: B::OwnedBuf = B::alloc_zeroed_bytes(B::cnv_pvec_left_bytes_of(n, cols, size, hint));
        CnvPVecL {
            data,
            shape: CnvPVecShape::new(n, cols, size, hint),
            _phantom: PhantomData,
        }
    }

    /// Uploads a host byte buffer into a backend-owned `CnvPVecL`.
    ///
    /// # Panics
    ///
    /// Panics if the buffer length does not equal `B::cnv_pvec_left_bytes_of(n, cols, size, hint)`;
    /// a `Vec<u8>` argument is first copied into storage padded to a 64-byte multiple, so
    /// its padded length is what is compared.
    pub fn from_bytes(
        n: usize,
        cols: usize,
        size: usize,
        hint: PrepareHint,
        bytes: impl Into<AlignedBuf>,
    ) -> CnvPVecL<B::OwnedBuf, W, B, Rg>
    where
        B: Backend<OwnedBuf = D>,
    {
        let data: AlignedBuf = bytes.into();
        assert!(data.len() == crate::layouts::padded_bytes(B::cnv_pvec_left_bytes_of(n, cols, size, hint)));
        let data: B::OwnedBuf = B::from_host_bytes(&data);
        CnvPVecL {
            data,
            shape: CnvPVecShape::new(n, cols, size, hint),
            _phantom: PhantomData,
        }
    }
}

impl<D: Data, W: DftWord, B: Backend<DftWord = W>, Rg: Ring> CnvPVecL<D, W, B, Rg> {
    pub fn from_data(data: D, n: usize, cols: usize, size: usize, hint: PrepareHint) -> Self {
        Self {
            data,
            shape: CnvPVecShape::new(n, cols, size, hint),
            _phantom: PhantomData,
        }
    }
}

/// One `(left, right)` operand pair of a fused convolution accumulation.
///
/// Consumed by [`CnvOps::cnv_apply_dft_sum`](crate::api::CnvOps::cnv_apply_dft_sum),
/// which overwrites the destination column with the sum of the bivariate
/// convolutions of all terms.
pub struct CnvDftAccTerm<'a, BE: Backend + 'a, Rg: Ring = Standard> {
    /// Prepared left operand.
    pub a: CnvPVecLBackendRef<'a, BE, Rg>,
    /// Column of `a` to convolve.
    pub a_col: usize,
    /// Prepared right operand.
    pub b: CnvPVecRBackendRef<'a, BE, Rg>,
    /// Column of `b` to convolve.
    pub b_col: usize,
    /// Galois element applied to `b`, odd; 1 for `b` itself.
    pub b_galois: i64,
}

/// Asserts that a convolution Galois element is odd.
#[inline]
pub fn check_galois(b_galois: i64) {
    assert!(b_galois & 1 == 1, "convolution: b_galois must be odd");
}

/// Asserts that one product is in the domain of `cnv_apply_dft` for a destination of degree `n`: `a` of degree `n`,
/// both columns in range, `b` of degree `n` or of a power-of-two degree dividing it and at least `BE::MIN_DEGREE`,
/// and an odd `b_galois`.
pub fn check_cnv_dft_term<BE: Backend, Rg: Ring>(
    n: usize,
    a: &CnvPVecLBackendRef<'_, BE, Rg>,
    a_col: usize,
    b: &CnvPVecRBackendRef<'_, BE, Rg>,
    b_col: usize,
    b_galois: i64,
) {
    check_galois(b_galois);
    assert!(a.n() == n, "convolution: left operand degree differs from the destination's");
    assert!(a_col < a.cols(), "convolution: left operand column out of range");
    assert!(b_col < b.cols(), "convolution: right operand column out of range");
    let b_n = b.n();
    assert!(
        b_n == n || (b_n.is_power_of_two() && b_n >= BE::MIN_DEGREE && b_n < n && n.is_multiple_of(b_n)),
        "convolution: right operand degree does not embed into the destination's"
    );
}

/// Asserts [`check_cnv_dft_term`] on every term, so that a sum refuses before its first write.
pub fn check_cnv_dft_terms<BE: Backend, Rg: Ring>(n: usize, terms: &[CnvDftAccTerm<'_, BE, Rg>]) {
    for term in terms {
        check_cnv_dft_term::<BE, Rg>(n, &term.a, term.a_col, &term.b, term.b_col, term.b_galois);
    }
}

/// Borrow a `CnvPVecR` as a shared reference view.
/// Owned `CnvPVecR` backed by a backend-owned buffer.
pub type CnvPVecROwned<B, Rg = Standard> = CnvPVecR<<B as Backend>::OwnedBuf, <B as Backend>::DftWord, B, Rg>;
/// Owned `CnvPVecL` backed by a backend-owned buffer.
pub type CnvPVecLOwned<B, Rg = Standard> = CnvPVecL<<B as Backend>::OwnedBuf, <B as Backend>::DftWord, B, Rg>;
pub type CnvPVecRBackendRef<'a, B, Rg = Standard> = CnvPVecR<<B as Backend>::BufRef<'a>, <B as Backend>::DftWord, B, Rg>;
pub type CnvPVecRBackendMut<'a, B, Rg = Standard> = CnvPVecR<<B as Backend>::BufMut<'a>, <B as Backend>::DftWord, B, Rg>;
pub type CnvPVecLBackendRef<'a, B, Rg = Standard> = CnvPVecL<<B as Backend>::BufRef<'a>, <B as Backend>::DftWord, B, Rg>;
pub type CnvPVecLBackendMut<'a, B, Rg = Standard> = CnvPVecL<<B as Backend>::BufMut<'a>, <B as Backend>::DftWord, B, Rg>;

/// Borrow a backend-owned `CnvPVecR` using the backend's native view type.
pub trait CnvPVecRToBackendRef<BE: Backend, Rg: Ring = Standard> {
    fn to_backend_ref(&self) -> CnvPVecRBackendRef<'_, BE, Rg>;
}

impl<BE: Backend, Rg: Ring> CnvPVecRToBackendRef<BE, Rg> for CnvPVecR<BE::OwnedBuf, BE::DftWord, BE, Rg> {
    fn to_backend_ref(&self) -> CnvPVecRBackendRef<'_, BE, Rg> {
        CnvPVecR {
            data: BE::view(&self.data),
            shape: self.shape,
            _phantom: self._phantom,
        }
    }
}

/// Reborrow an already backend-borrowed `CnvPVecR` as a shared backend-native view.
pub trait CnvPVecRReborrowBackendRef<BE: Backend, Rg: Ring = Standard> {
    fn reborrow_backend_ref(&self) -> CnvPVecRBackendRef<'_, BE, Rg>;
}

impl<BE: Backend, Rg: Ring> CnvPVecRReborrowBackendRef<BE, Rg> for CnvPVecR<BE::BufMut<'_>, BE::DftWord, BE, Rg> {
    fn reborrow_backend_ref(&self) -> CnvPVecRBackendRef<'_, BE, Rg> {
        CnvPVecR {
            data: BE::view_ref_mut(&self.data),
            shape: self.shape,
            _phantom: self._phantom,
        }
    }
}

/// Mutably borrow a backend-owned `CnvPVecR` using the backend's native view type.
pub trait CnvPVecRToBackendMut<BE: Backend, Rg: Ring = Standard> {
    fn to_backend_mut(&mut self) -> CnvPVecRBackendMut<'_, BE, Rg>;
}

impl<BE: Backend, Rg: Ring> CnvPVecRToBackendMut<BE, Rg> for CnvPVecR<BE::OwnedBuf, BE::DftWord, BE, Rg> {
    fn to_backend_mut(&mut self) -> CnvPVecRBackendMut<'_, BE, Rg> {
        CnvPVecR {
            data: BE::view_mut(&mut self.data),
            shape: self.shape,
            _phantom: self._phantom,
        }
    }
}

/// Reborrow an already backend-borrowed `CnvPVecR` as a mutable backend-native view.
pub trait CnvPVecRReborrowBackendMut<BE: Backend, Rg: Ring = Standard> {
    fn reborrow_backend_mut(&mut self) -> CnvPVecRBackendMut<'_, BE, Rg>;
}

impl<BE: Backend, Rg: Ring> CnvPVecRReborrowBackendMut<BE, Rg> for CnvPVecR<BE::BufMut<'_>, BE::DftWord, BE, Rg> {
    fn reborrow_backend_mut(&mut self) -> CnvPVecRBackendMut<'_, BE, Rg> {
        CnvPVecR {
            data: BE::view_mut_ref(&mut self.data),
            shape: self.shape,
            _phantom: self._phantom,
        }
    }
}

/// Borrow a `CnvPVecL` as a shared reference view.
pub trait CnvPVecLToBackendRef<BE: Backend, Rg: Ring = Standard> {
    fn to_backend_ref(&self) -> CnvPVecLBackendRef<'_, BE, Rg>;
}

impl<BE: Backend, Rg: Ring> CnvPVecLToBackendRef<BE, Rg> for CnvPVecL<BE::OwnedBuf, BE::DftWord, BE, Rg> {
    fn to_backend_ref(&self) -> CnvPVecLBackendRef<'_, BE, Rg> {
        CnvPVecL {
            data: BE::view(&self.data),
            shape: self.shape,
            _phantom: self._phantom,
        }
    }
}

/// Reborrow an already backend-borrowed `CnvPVecL` as a shared backend-native view.
pub trait CnvPVecLReborrowBackendRef<BE: Backend, Rg: Ring = Standard> {
    fn reborrow_backend_ref(&self) -> CnvPVecLBackendRef<'_, BE, Rg>;
}

impl<BE: Backend, Rg: Ring> CnvPVecLReborrowBackendRef<BE, Rg> for CnvPVecL<BE::BufMut<'_>, BE::DftWord, BE, Rg> {
    fn reborrow_backend_ref(&self) -> CnvPVecLBackendRef<'_, BE, Rg> {
        CnvPVecL {
            data: BE::view_ref_mut(&self.data),
            shape: self.shape,
            _phantom: self._phantom,
        }
    }
}

/// Mutably borrow a backend-owned `CnvPVecL` using the backend's native view type.
pub trait CnvPVecLToBackendMut<BE: Backend, Rg: Ring = Standard> {
    fn to_backend_mut(&mut self) -> CnvPVecLBackendMut<'_, BE, Rg>;
}

impl<BE: Backend, Rg: Ring> CnvPVecLToBackendMut<BE, Rg> for CnvPVecL<BE::OwnedBuf, BE::DftWord, BE, Rg> {
    fn to_backend_mut(&mut self) -> CnvPVecLBackendMut<'_, BE, Rg> {
        CnvPVecL {
            data: BE::view_mut(&mut self.data),
            shape: self.shape,
            _phantom: self._phantom,
        }
    }
}

/// Reborrow an already backend-borrowed `CnvPVecL` as a mutable backend-native view.
pub trait CnvPVecLReborrowBackendMut<BE: Backend, Rg: Ring = Standard> {
    fn reborrow_backend_mut(&mut self) -> CnvPVecLBackendMut<'_, BE, Rg>;
}

impl<BE: Backend, Rg: Ring> CnvPVecLReborrowBackendMut<BE, Rg> for CnvPVecL<BE::BufMut<'_>, BE::DftWord, BE, Rg> {
    fn reborrow_backend_mut(&mut self) -> CnvPVecLBackendMut<'_, BE, Rg> {
        CnvPVecL {
            data: BE::view_mut_ref(&mut self.data),
            shape: self.shape,
            _phantom: self._phantom,
        }
    }
}

impl<D: Data, W: DftWord, B: Backend<DftWord = W>, Rg: Ring> CnvPVecL<D, W, B, Rg> {
    /// Zero-copy re-tag of this container to a layout-compatible backend `B2`.
    ///
    /// The buffer moves as-is; only the type tag changes. Requires the
    /// [`CnvPVecLayoutCompatible`](crate::layouts::CnvPVecLayoutCompatible) marker declared by the backend
    /// pair. `D` is kept, so for further backend-native use `B2`'s buffer
    /// types must match `D` (true for every backend in the workspace).
    pub fn into_backend<B2>(self) -> CnvPVecL<D, W, B2, Rg>
    where
        B2: Backend<DftWord = W>,
        B: crate::layouts::CnvPVecLayoutCompatible<B2>,
    {
        let shape = self.shape;
        assert_eq!(
            B::cnv_pvec_left_bytes_of(shape.n(), shape.cols(), shape.size(), shape.hint()),
            B2::cnv_pvec_left_bytes_of(shape.n(), shape.cols(), shape.size(), shape.hint()),
            "into_backend: byte sizes diverge despite declared layout compatibility"
        );
        CnvPVecL {
            data: self.data,
            shape,
            _phantom: PhantomData,
        }
    }
}

impl<D: Data, W: DftWord, B: Backend<DftWord = W>, Rg: Ring> CnvPVecR<D, W, B, Rg> {
    /// Zero-copy re-tag of this container to a layout-compatible backend `B2`.
    ///
    /// The buffer moves as-is; only the type tag changes. Requires the
    /// [`CnvPVecLayoutCompatible`](crate::layouts::CnvPVecLayoutCompatible) marker declared by the backend
    /// pair. `D` is kept, so for further backend-native use `B2`'s buffer
    /// types must match `D` (true for every backend in the workspace).
    pub fn into_backend<B2>(self) -> CnvPVecR<D, W, B2, Rg>
    where
        B2: Backend<DftWord = W>,
        B: crate::layouts::CnvPVecLayoutCompatible<B2>,
    {
        let shape = self.shape;
        assert_eq!(
            B::cnv_pvec_right_bytes_of(shape.n(), shape.cols(), shape.size(), shape.hint()),
            B2::cnv_pvec_right_bytes_of(shape.n(), shape.cols(), shape.size(), shape.hint()),
            "into_backend: byte sizes diverge despite declared layout compatibility"
        );
        CnvPVecR {
            data: self.data,
            shape,
            _phantom: PhantomData,
        }
    }
}
