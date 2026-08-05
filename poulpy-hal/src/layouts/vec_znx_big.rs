use std::{
    hash::{DefaultHasher, Hasher},
    marker::PhantomData,
};

use std::fmt;

use crate::layouts::{
    Backend, BigWord, Data, DataView, DataViewMut, DigestU64, HostDataMut, HostDataRef, VecZnxShape, ZnxInfos, ZnxView,
    ZnxViewMut, ZnxZero,
};

/// Extended-precision polynomial vector used as a result accumulator.
///
/// `VecZnxBig` has the same structural shape as [`VecZnx`](crate::layouts::VecZnx)
/// (`cols` columns, `size` limbs, ring degree `N`) but uses a [`BigWord`]
/// as its coefficient type instead of a [`ZnxWord`](crate::layouts::ZnxWord).
/// The wider scalar type allows lossless accumulation of intermediate
/// products before normalization back to coefficient-domain limbs.
///
/// The word type `W` names the byte-layout convention of the buffer; two
/// buffers with the same `W` are interchangeable regardless of which backend
/// produced them (see [`BigWord`]).
#[repr(C)]
#[derive(PartialEq, Eq, Hash)]
pub struct VecZnxBig<D: Data, W: BigWord, B: Backend<BigWord = W>> {
    pub data: D,
    shape: VecZnxShape,
    pub _phantom: PhantomData<(W, B)>,
}

impl<D: HostDataRef, W: BigWord, B: Backend<BigWord = W>> DigestU64 for VecZnxBig<D, W, B> {
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

impl<D: HostDataRef, W: BigWord, B: Backend<BigWord = W>> ZnxView for VecZnxBig<D, W, B> {
    type Scalar = W;
}

impl<D: Data, W: BigWord, B: Backend<BigWord = W>> ZnxInfos for VecZnxBig<D, W, B> {
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

impl<D: Data, W: BigWord, B: Backend<BigWord = W>> DataView for VecZnxBig<D, W, B> {
    type D = D;
    fn data(&self) -> &Self::D {
        &self.data
    }
}

impl<D: Data, W: BigWord, B: Backend<BigWord = W>> DataViewMut for VecZnxBig<D, W, B> {
    fn data_mut(&mut self) -> &mut Self::D {
        &mut self.data
    }
}

impl<D: Data, W: BigWord, B: Backend<BigWord = W>> VecZnxBig<D, W, B> {
    pub fn n(&self) -> usize {
        self.shape.n()
    }

    pub fn cols(&self) -> usize {
        self.shape.cols()
    }

    pub fn size(&self) -> usize {
        self.shape.size()
    }

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

    pub fn set_size(&mut self, size: usize) {
        self.shape = self.shape.with_size(size);
    }
}

impl<D: HostDataMut, W: BigWord, B: Backend<BigWord = W>> ZnxZero for VecZnxBig<D, W, B> {
    fn zero(&mut self) {
        self.raw_mut().fill(W::zero())
    }
    fn zero_at(&mut self, i: usize, j: usize) {
        self.at_mut(i, j).fill(W::zero());
    }
}

impl<D: Data, W: BigWord, B: Backend<BigWord = W>> VecZnxBig<D, W, B> {
    /// Allocates a zero-initialized backend-owned `VecZnxBig`.
    pub(crate) fn alloc(n: usize, cols: usize, size: usize) -> VecZnxBigOwned<B>
    where
        B: Backend<OwnedBuf = D>,
    {
        let data: <B as Backend>::OwnedBuf = B::alloc_zeroed_bytes(B::bytes_of_vec_znx_big(n, cols, size));
        VecZnxBig {
            data,
            shape: VecZnxShape::new(n, cols, size, size),
            _phantom: PhantomData,
        }
    }

    /// Uploads a host byte buffer into a backend-owned `VecZnxBig`.
    ///
    /// # Panics
    ///
    /// Panics if the buffer length does not equal `B::bytes_of_vec_znx_big(n, cols, size)`.
    pub fn from_bytes(n: usize, cols: usize, size: usize, bytes: impl Into<Vec<u8>>) -> VecZnxBigOwned<B>
    where
        B: Backend<OwnedBuf = D>,
    {
        let data: Vec<u8> = bytes.into();
        assert!(data.len() == B::bytes_of_vec_znx_big(n, cols, size));
        let data: <B as Backend>::OwnedBuf = B::from_host_bytes(&data);
        VecZnxBig {
            data,
            shape: VecZnxShape::new(n, cols, size, size),
            _phantom: PhantomData,
        }
    }
}

impl<D: Data, W: BigWord, B: Backend<BigWord = W>> VecZnxBig<D, W, B> {
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

/// Owned `VecZnxBig` backed by a backend-owned buffer.
pub type VecZnxBigOwned<B> = VecZnxBig<<B as Backend>::OwnedBuf, <B as Backend>::BigWord, B>;
/// Shared backend-native borrow of a `VecZnxBig`.
pub type VecZnxBigBackendRef<'a, B> = VecZnxBig<<B as Backend>::BufRef<'a>, <B as Backend>::BigWord, B>;
/// Mutable backend-native borrow of a `VecZnxBig`.
pub type VecZnxBigBackendMut<'a, B> = VecZnxBig<<B as Backend>::BufMut<'a>, <B as Backend>::BigWord, B>;

/// Reborrow a mutable backend-native `VecZnxBig` view as a shared backend-native view.
pub fn vec_znx_big_backend_ref_from_mut<'a, 'b, B: Backend + 'b>(
    vec: &'a VecZnxBigBackendMut<'b, B>,
) -> VecZnxBigBackendRef<'a, B> {
    VecZnxBig {
        data: B::view_ref_mut(&vec.data),
        shape: vec.shape,
        _phantom: PhantomData,
    }
}

/// Borrow a backend-owned `VecZnxBig` using the backend's native view type.
pub trait VecZnxBigToBackendRef<B: Backend> {
    fn to_backend_ref(&self) -> VecZnxBigBackendRef<'_, B>;
}

impl<B: Backend> VecZnxBigToBackendRef<B> for VecZnxBig<B::OwnedBuf, B::BigWord, B> {
    fn to_backend_ref(&self) -> VecZnxBigBackendRef<'_, B> {
        VecZnxBig {
            data: B::view(&self.data),
            shape: self.shape,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<'b, B: Backend + 'b> VecZnxBigToBackendRef<B> for &VecZnxBig<B::BufRef<'b>, B::BigWord, B> {
    fn to_backend_ref(&self) -> VecZnxBigBackendRef<'_, B> {
        VecZnxBig {
            data: B::view_ref(&self.data),
            shape: self.shape,
            _phantom: std::marker::PhantomData,
        }
    }
}

/// Reborrow an already backend-borrowed `VecZnxBig` as a shared backend-native view.
pub trait VecZnxBigReborrowBackendRef<B: Backend> {
    fn reborrow_backend_ref(&self) -> VecZnxBigBackendRef<'_, B>;
}

impl<'b, B: Backend + 'b> VecZnxBigReborrowBackendRef<B> for VecZnxBig<B::BufMut<'b>, B::BigWord, B> {
    fn reborrow_backend_ref(&self) -> VecZnxBigBackendRef<'_, B> {
        vec_znx_big_backend_ref_from_mut::<B>(self)
    }
}

/// Mutably borrow a backend-owned `VecZnxBig` using the backend's native view type.
pub trait VecZnxBigToBackendMut<B: Backend> {
    fn to_backend_mut(&mut self) -> VecZnxBigBackendMut<'_, B>;
}

impl<B: Backend> VecZnxBigToBackendMut<B> for VecZnxBig<B::OwnedBuf, B::BigWord, B> {
    fn to_backend_mut(&mut self) -> VecZnxBigBackendMut<'_, B> {
        VecZnxBig {
            data: B::view_mut(&mut self.data),
            shape: self.shape,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<'b, B: Backend + 'b> VecZnxBigToBackendMut<B> for &mut VecZnxBig<B::BufMut<'b>, B::BigWord, B> {
    fn to_backend_mut(&mut self) -> VecZnxBigBackendMut<'_, B> {
        VecZnxBig {
            data: B::view_mut_ref(&mut self.data),
            shape: self.shape,
            _phantom: std::marker::PhantomData,
        }
    }
}

/// Reborrow an already backend-borrowed `VecZnxBig` as a mutable backend-native view.
pub trait VecZnxBigReborrowBackendMut<B: Backend> {
    fn reborrow_backend_mut(&mut self) -> VecZnxBigBackendMut<'_, B>;
}

impl<'b, B: Backend + 'b> VecZnxBigReborrowBackendMut<B> for VecZnxBig<B::BufMut<'b>, B::BigWord, B> {
    fn reborrow_backend_mut(&mut self) -> VecZnxBigBackendMut<'_, B> {
        VecZnxBig {
            data: B::view_mut_ref(&mut self.data),
            shape: self.shape,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<D: HostDataRef, W: BigWord, B: Backend<BigWord = W>> fmt::Display for VecZnxBig<D, W, B> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "VecZnxBig(n={}, cols={}, size={})", self.n(), self.cols(), self.size())?;

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

impl<D: Data, W: BigWord, B: Backend<BigWord = W>> VecZnxBig<D, W, B> {
    /// Zero-copy re-tag of this container to a layout-compatible backend `B2`.
    ///
    /// The buffer moves as-is; only the type tag changes. Requires the
    /// [`VecZnxBigLayoutCompatible`](crate::layouts::VecZnxBigLayoutCompatible) marker declared by the backend
    /// pair. `D` is kept, so for further backend-native use `B2`'s buffer
    /// types must match `D` (true for all current CPU backends).
    pub fn into_backend<B2>(self) -> VecZnxBig<D, W, B2>
    where
        B2: Backend<BigWord = W>,
        B: crate::layouts::VecZnxBigLayoutCompatible<B2>,
    {
        let shape = self.shape;
        assert_eq!(
            B::bytes_of_vec_znx_big(shape.n(), shape.cols(), shape.max_size()),
            B2::bytes_of_vec_znx_big(shape.n(), shape.cols(), shape.max_size()),
            "into_backend: byte sizes diverge despite declared layout compatibility"
        );
        VecZnxBig {
            data: self.data,
            shape,
            _phantom: PhantomData,
        }
    }
}
