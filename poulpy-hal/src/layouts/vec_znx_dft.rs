use std::{
    fmt,
    hash::{DefaultHasher, Hasher},
    marker::PhantomData,
};

use crate::layouts::{
    Backend, Data, DataView, DataViewMut, DftWord, DigestU64, HostDataMut, HostDataRef, VecZnxBig, VecZnxInfos, VecZnxShape,
    ZnxInfos, ZnxView, ZnxViewMut, ZnxZero,
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
pub struct VecZnxDft<D: Data, W: DftWord, B: Backend<DftWord = W>> {
    pub data: D,
    shape: VecZnxShape,
    pub _phantom: PhantomData<(W, B)>,
}

// Equality (and hashing, where provided) is defined directly on the
// representation: same shape, same buffer bytes. No `W`/`B` value is ever
// compared, so no bound on them is needed — in particular `Eq` holds even
// for non-`Eq` words like `f64` (byte equality is a total equivalence).
impl<D: Data, W: DftWord, B: Backend<DftWord = W>> PartialEq for VecZnxDft<D, W, B> {
    fn eq(&self, other: &Self) -> bool {
        self.shape == other.shape && self.data == other.data
    }
}

impl<D: Data, W: DftWord, B: Backend<DftWord = W>> Eq for VecZnxDft<D, W, B> {}

impl<D: HostDataRef, W: DftWord, B: Backend<DftWord = W>> DigestU64 for VecZnxDft<D, W, B> {
    fn digest_u64(&self) -> u64 {
        let mut h: DefaultHasher = DefaultHasher::new();
        h.write(self.data.as_ref());
        h.write_usize(self.n());
        h.write_usize(self.cols());
        h.write_usize(self.size());
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

impl<D: Data, W: DftWord, B: Backend<DftWord = W>> VecZnxInfos for VecZnxDft<D, W, B> {
    fn cols(&self) -> usize {
        self.shape.cols()
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

    /// Whether the backend's physical representation can be viewed as `W`
    /// elements. Packed backends may use `W` only as an identity marker.
    fn has_element_view(&self) -> bool {
        let element_bytes = crate::layouts::element_view_span(self)
            .checked_mul(size_of::<W>())
            .expect("VecZnxDft element-view byte size overflows usize");
        element_bytes == B::bytes_of_vec_znx_dft(self.n(), self.cols(), self.size())
    }
}

impl<'b, B: Backend + 'b> VecZnxDftBackendMut<'b, B> {
    /// Reborrows this buffer as a mutable view with a temporary compute size.
    ///
    /// The returned view addresses the same allocation, but HAL
    /// kernels see `size` as the active limb count. Dropping the view leaves
    /// `self`'s metadata unchanged.
    ///
    /// # Panics
    ///
    /// Panics if `size > self.size()`.
    pub fn with_size_mut(&mut self, size: usize) -> VecZnxDftBackendMut<'_, B> {
        VecZnxDft {
            data: B::view_mut_ref(&mut self.data),
            shape: self.shape.with_size(size),
            _phantom: PhantomData,
        }
    }
}

impl<D: HostDataMut, W: DftWord, B: Backend<DftWord = W>> ZnxZero for VecZnxDft<D, W, B> {
    fn zero(&mut self) {
        if self.has_element_view() {
            self.raw_mut().fill(W::zero());
            return;
        }

        let byte_len = B::bytes_of_vec_znx_dft(self.n(), self.cols(), self.size());
        let data = self.data.as_mut();
        assert!(
            byte_len <= data.len(),
            "VecZnxDft backend representation ({byte_len} bytes) exceeds the {}-byte buffer",
            data.len()
        );
        data[..byte_len].fill(0);
    }

    fn zero_at(&mut self, i: usize, j: usize) {
        if self.has_element_view() {
            self.at_mut(i, j).fill(W::zero());
            return;
        }

        assert!(i < self.cols(), "cols: {} >= self.cols(): {}", i, self.cols());
        assert!(j < self.size(), "size: {} >= self.size(): {}", j, self.size());

        let block_bytes = B::bytes_of_vec_znx_dft(self.n(), 1, 1);
        let byte_len = crate::layouts::checked_product(&[block_bytes, self.cols(), self.size()], "VecZnxDft packed byte size");
        assert_eq!(
            byte_len,
            B::bytes_of_vec_znx_dft(self.n(), self.cols(), self.size()),
            "VecZnxDft backend representation is not block-linear"
        );

        let block = j
            .checked_mul(self.cols())
            .and_then(|x| x.checked_add(i))
            .expect("VecZnxDft packed block index overflows usize");
        let offset = block
            .checked_mul(block_bytes)
            .expect("VecZnxDft packed block offset overflows usize");
        let end = offset
            .checked_add(block_bytes)
            .expect("VecZnxDft packed block end overflows usize");
        let data = self.data.as_mut();
        assert!(
            end <= data.len(),
            "VecZnxDft packed block ({i}, {j}) exceeds the {}-byte buffer",
            data.len()
        );
        data[offset..end].fill(0);
    }
}

impl<B: Backend> VecZnxDft<B::OwnedBuf, B::DftWord, B> {
    /// Allocates a zero-initialized backend-owned `VecZnxDft`.
    pub fn alloc(n: usize, cols: usize, size: usize) -> VecZnxDftOwned<B> {
        let data: <B as Backend>::OwnedBuf = B::alloc_zeroed_bytes(B::bytes_of_vec_znx_dft(n, cols, size));
        VecZnxDft {
            data,
            shape: VecZnxShape::new(n, cols, size),
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
            shape: VecZnxShape::new(n, cols, size),
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
            shape: VecZnxShape::new(n, cols, size),
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

        if !self.has_element_view() {
            return writeln!(
                f,
                "  <backend-packed representation: {} bytes>",
                B::bytes_of_vec_znx_dft(self.n(), self.cols(), self.size())
            );
        }

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
            B::bytes_of_vec_znx_dft(shape.n(), shape.cols(), shape.size()),
            B2::bytes_of_vec_znx_dft(shape.n(), shape.cols(), shape.size()),
            "into_backend: byte sizes diverge despite declared layout compatibility"
        );
        VecZnxDft {
            data: self.data,
            shape,
            _phantom: PhantomData,
        }
    }
}
