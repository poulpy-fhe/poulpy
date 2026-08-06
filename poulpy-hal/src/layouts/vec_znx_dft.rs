use std::{
    fmt,
    hash::{DefaultHasher, Hasher},
    marker::PhantomData,
};

use crate::layouts::{
    Backend, Data, DataView, DataViewMut, DftWord, DigestU64, HostDataMut, HostDataRef, VecZnxBig, VecZnxShape, ZnxInfos,
    ZnxView, ZnxViewMut, ZnxZero,
};

/// Polynomial vector in DFT (evaluation) domain.
///
/// `VecZnxDft` has the same structural shape as [`VecZnx`](crate::layouts::VecZnx)
/// but stores coefficients as [`DftWord`] values in the frequency domain
/// rather than [`ZnxWord`](crate::layouts::ZnxWord) values in the coefficient
/// domain.
///
/// The word type `W` names the byte-layout convention of the buffer (see
/// [`DftWord`]); the backend `B` pins which implementation produced it.
/// Containers of different backends are distinct types even when their words
/// match — cross-backend movement is explicit, via the zero-copy re-tag
/// guarded by the layout-compatibility markers or via a transfer/re-prepare.
///
/// Multiplication and scalar-vector/vector-matrix products are performed
/// in this representation to exploit FFT-based convolution. Use
/// [`VecZnxDftApply`](crate::api::VecZnxDftApply) /
/// [`VecZnxIdftApply`](crate::api::VecZnxIdftApply) to convert
/// between coefficient and DFT domains.
#[repr(C)]
#[derive(PartialEq)]
pub struct VecZnxDft<D: Data, W: DftWord, B: Backend<DftWord = W>> {
    pub data: D,
    shape: VecZnxShape,
    pub _phantom: PhantomData<(W, B)>,
}

// Equality is byte equality on `data` (plus the shape); it never compares a
// `W` value, so it is an equivalence relation even for non-`Eq` words like
// `f64`. A derived `Eq` would spuriously demand `W: Eq`.
impl<D: Data, W: DftWord, B: Backend<DftWord = W>> Eq for VecZnxDft<D, W, B> {}

impl<D: HostDataRef, W: DftWord, B: Backend<DftWord = W>> DigestU64 for VecZnxDft<D, W, B> {
    fn digest_u64(&self) -> u64 {
        let mut h: DefaultHasher = DefaultHasher::new();
        h.write(self.data.as_ref());
        h.write_usize(self.n());
        h.write_usize(self.cols());
        h.write_usize(self.size());
        h.write_usize(self.max_size());
        h.finish()
    }
}

impl<D: HostDataRef, W: DftWord, B: Backend<DftWord = W>> ZnxView for VecZnxDft<D, W, B> {
    type Scalar = W;
}

impl<D: Data, W: DftWord, B: Backend<DftWord = W>> VecZnxDft<D, W, B> {
    pub fn n(&self) -> usize {
        self.shape.n()
    }

    pub fn cols(&self) -> usize {
        self.shape.cols()
    }

    pub fn size(&self) -> usize {
        self.shape.size()
    }

    /// Reinterprets this DFT vector as a [`VecZnxBig`], consuming `self`.
    ///
    /// This is a zero-copy conversion that changes only the type tag; the
    /// underlying data buffer is moved as-is. The backend `B` declares both
    /// words, which guarantees the buffer is large enough for the big-domain
    /// interpretation.
    pub fn into_big(self) -> VecZnxBig<D, B::BigWord, B> {
        let shape = self.shape;
        assert!(
            B::bytes_of_vec_znx_big(shape.n(), shape.cols(), shape.size())
                <= B::bytes_of_vec_znx_dft(shape.n(), shape.cols(), shape.size()),
            "into_big: big-domain buffer would exceed the DFT-domain allocation"
        );
        VecZnxBig::<D, B::BigWord, B>::from_data(self.data, shape.n(), shape.cols(), shape.size())
    }
}

impl<D: Data, W: DftWord, B: Backend<DftWord = W>> ZnxInfos for VecZnxDft<D, W, B> {
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
        self.shape.size()
    }
}

impl<D: Data, W: DftWord, B: Backend<DftWord = W>> DataView for VecZnxDft<D, W, B> {
    type D = D;
    fn data(&self) -> &Self::D {
        &self.data
    }
}

impl<D: Data, W: DftWord, B: Backend<DftWord = W>> DataViewMut for VecZnxDft<D, W, B> {
    fn data_mut(&mut self) -> &mut Self::D {
        &mut self.data
    }
}

impl<D: Data, W: DftWord, B: Backend<DftWord = W>> VecZnxDft<D, W, B> {
    pub fn shape(&self) -> VecZnxShape {
        self.shape
    }

    pub fn max_size(&self) -> usize {
        self.shape.max_size()
    }

    pub fn with_size(mut self, size: usize) -> Self {
        assert!(size <= self.max_size());
        self.shape = self.shape.with_size(size);
        self
    }
}

impl<D: Data, W: DftWord, B: Backend<DftWord = W>> VecZnxDft<D, W, B> {
    /// Sets the active limb count.
    ///
    /// # Panics
    ///
    /// Panics if `size > max_size`.
    pub fn set_size(&mut self, size: usize) {
        self.shape = self.shape.with_size(size)
    }
}

impl<D: HostDataMut, W: DftWord, B: Backend<DftWord = W>> ZnxZero for VecZnxDft<D, W, B> {
    fn zero(&mut self) {
        self.raw_mut().fill(W::zero())
    }
    fn zero_at(&mut self, i: usize, j: usize) {
        self.at_mut(i, j).fill(W::zero());
    }
}

impl<B: Backend> VecZnxDft<B::OwnedBuf, B::DftWord, B> {
    /// Allocates a zero-initialized backend-owned `VecZnxDft`.
    pub fn alloc(n: usize, cols: usize, size: usize) -> VecZnxDftOwned<B> {
        let data: <B as Backend>::OwnedBuf = B::alloc_zeroed_bytes(B::bytes_of_vec_znx_dft(n, cols, size));
        VecZnxDft {
            data,
            shape: VecZnxShape::new(n, cols, size, size),
            _phantom: PhantomData,
        }
    }

    /// Uploads a host byte buffer into a backend-owned `VecZnxDft`.
    ///
    /// # Panics
    ///
    /// Panics if the buffer length does not equal `B::bytes_of_vec_znx_dft(n, cols, size)`.
    pub fn from_bytes(n: usize, cols: usize, size: usize, bytes: impl Into<Vec<u8>>) -> VecZnxDftOwned<B> {
        let data: Vec<u8> = bytes.into();
        assert!(data.len() == B::bytes_of_vec_znx_dft(n, cols, size));
        let data: <B as Backend>::OwnedBuf = B::from_host_bytes(&data);
        VecZnxDft {
            data,
            shape: VecZnxShape::new(n, cols, size, size),
            _phantom: PhantomData,
        }
    }
}

/// Owned `VecZnxDft` backed by a backend-owned buffer.
pub type VecZnxDftOwned<B> = VecZnxDft<<B as Backend>::OwnedBuf, <B as Backend>::DftWord, B>;
/// Shared backend-native borrow of a `VecZnxDft`.
pub type VecZnxDftBackendRef<'a, B> = VecZnxDft<<B as Backend>::BufRef<'a>, <B as Backend>::DftWord, B>;
/// Mutable backend-native borrow of a `VecZnxDft`.
pub type VecZnxDftBackendMut<'a, B> = VecZnxDft<<B as Backend>::BufMut<'a>, <B as Backend>::DftWord, B>;

/// Reborrow a mutable backend-native `VecZnxDft` view as a shared backend-native view.
pub fn vec_znx_dft_backend_ref_from_mut<'a, 'b, B: Backend + 'b>(
    vec: &'a VecZnxDftBackendMut<'b, B>,
) -> VecZnxDftBackendRef<'a, B> {
    VecZnxDft {
        data: B::view_ref_mut(&vec.data),
        shape: vec.shape,
        _phantom: PhantomData,
    }
}

pub fn vec_znx_dft_backend_mut_from_mut<'a, 'b, B: Backend + 'b>(
    vec: &'a mut VecZnxDftBackendMut<'b, B>,
) -> VecZnxDftBackendMut<'a, B> {
    VecZnxDft {
        data: B::view_mut_ref(&mut vec.data),
        shape: vec.shape,
        _phantom: PhantomData,
    }
}

impl<D: Data, W: DftWord, B: Backend<DftWord = W>> VecZnxDft<D, W, B> {
    /// Constructs a `VecZnxDft` from raw parts without validation.
    pub fn from_data(data: D, n: usize, cols: usize, size: usize) -> Self {
        Self {
            data,
            shape: VecZnxShape::new(n, cols, size, size),
            _phantom: PhantomData,
        }
    }

    pub fn from_data_with_max_size(data: D, n: usize, cols: usize, size: usize, max_size: usize) -> Self {
        Self {
            data,
            shape: VecZnxShape::new(n, cols, size, max_size),
            _phantom: PhantomData,
        }
    }
}

/// Borrow a backend-owned `VecZnxDft` using the backend's native view type.
pub trait VecZnxDftToBackendRef<B: Backend> {
    fn to_backend_ref(&self) -> VecZnxDftBackendRef<'_, B>;
}

impl<B: Backend> VecZnxDftToBackendRef<B> for VecZnxDft<B::OwnedBuf, B::DftWord, B> {
    fn to_backend_ref(&self) -> VecZnxDftBackendRef<'_, B> {
        VecZnxDft {
            data: B::view(&self.data),
            shape: self.shape,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<'b, B: Backend + 'b> VecZnxDftToBackendRef<B> for &VecZnxDft<B::BufRef<'b>, B::DftWord, B> {
    fn to_backend_ref(&self) -> VecZnxDftBackendRef<'_, B> {
        VecZnxDft {
            data: B::view_ref(&self.data),
            shape: self.shape,
            _phantom: std::marker::PhantomData,
        }
    }
}

/// Reborrow an already backend-borrowed `VecZnxDft` as a shared backend-native view.
pub trait VecZnxDftReborrowBackendRef<B: Backend> {
    fn reborrow_backend_ref(&self) -> VecZnxDftBackendRef<'_, B>;
}

impl<'b, B: Backend + 'b> VecZnxDftReborrowBackendRef<B> for VecZnxDft<B::BufMut<'b>, B::DftWord, B> {
    fn reborrow_backend_ref(&self) -> VecZnxDftBackendRef<'_, B> {
        vec_znx_dft_backend_ref_from_mut::<B>(self)
    }
}

/// Mutably borrow a backend-owned `VecZnxDft` using the backend's native view type.
pub trait VecZnxDftToBackendMut<B: Backend> {
    fn to_backend_mut(&mut self) -> VecZnxDftBackendMut<'_, B>;
}

impl<B: Backend> VecZnxDftToBackendMut<B> for VecZnxDft<B::OwnedBuf, B::DftWord, B> {
    fn to_backend_mut(&mut self) -> VecZnxDftBackendMut<'_, B> {
        VecZnxDft {
            data: B::view_mut(&mut self.data),
            shape: self.shape,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<'b, B: Backend + 'b> VecZnxDftToBackendMut<B> for &mut VecZnxDft<B::BufMut<'b>, B::DftWord, B> {
    fn to_backend_mut(&mut self) -> VecZnxDftBackendMut<'_, B> {
        vec_znx_dft_backend_mut_from_mut::<B>(self)
    }
}

/// Reborrow an already backend-borrowed `VecZnxDft` as a mutable backend-native view.
pub trait VecZnxDftReborrowBackendMut<B: Backend> {
    fn reborrow_backend_mut(&mut self) -> VecZnxDftBackendMut<'_, B>;
}

impl<'b, B: Backend + 'b> VecZnxDftReborrowBackendMut<B> for VecZnxDft<B::BufMut<'b>, B::DftWord, B> {
    fn reborrow_backend_mut(&mut self) -> VecZnxDftBackendMut<'_, B> {
        vec_znx_dft_backend_mut_from_mut::<B>(self)
    }
}

impl<D: HostDataRef, W: DftWord, B: Backend<DftWord = W>> fmt::Display for VecZnxDft<D, W, B> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "VecZnxDft(n={}, cols={}, size={})", self.n(), self.cols(), self.size())?;

        for col in 0..self.cols() {
            writeln!(f, "Column {col}:")?;
            for size in 0..self.size() {
                let coeffs = self.at(col, size);
                write!(f, "  Size {size}: [")?;

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
        }
        Ok(())
    }
}

impl<D: Data, W: DftWord, B: Backend<DftWord = W>> VecZnxDft<D, W, B> {
    /// Zero-copy re-tag of this container to a layout-compatible backend `B2`.
    ///
    /// The buffer moves as-is; only the type tag changes. Requires the
    /// [`VecZnxDftLayoutCompatible`](crate::layouts::VecZnxDftLayoutCompatible) marker declared by the backend
    /// pair. `D` is kept, so for further backend-native use `B2`'s buffer
    /// types must match `D` (true for all current CPU backends).
    pub fn into_backend<B2>(self) -> VecZnxDft<D, W, B2>
    where
        B2: Backend<DftWord = W>,
        B: crate::layouts::VecZnxDftLayoutCompatible<B2>,
    {
        let shape = self.shape;
        assert_eq!(
            B::bytes_of_vec_znx_dft(shape.n(), shape.cols(), shape.max_size()),
            B2::bytes_of_vec_znx_dft(shape.n(), shape.cols(), shape.max_size()),
            "into_backend: byte sizes diverge despite declared layout compatibility"
        );
        VecZnxDft {
            data: self.data,
            shape,
            _phantom: PhantomData,
        }
    }
}
