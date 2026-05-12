use std::{
    hash::{DefaultHasher, Hasher},
    marker::PhantomData,
};

use rand_distr::num_traits::Zero;
use std::fmt;

use crate::layouts::{
    Backend, Data, DataView, DataViewMut, DigestU64, HostDataMut, HostDataRef, VecZnxShape, ZnxInfos, ZnxView, ZnxViewMut,
    ZnxZero,
};

/// Extended-precision polynomial vector used as a result accumulator.
///
/// `VecZnxBig` has the same structural shape as [`VecZnx`](crate::layouts::VecZnx)
/// (`cols` columns, `size` limbs, ring degree `N`) but uses
/// [`Backend::ScalarBig`] as its coefficient type instead of `i64`.
/// The wider scalar type allows lossless accumulation of intermediate
/// products before normalization back to `i64` limbs.
///
/// The exact scalar width and memory layout are backend-specific.
#[repr(C)]
#[derive(PartialEq, Eq, Hash)]
pub struct VecZnxBig<D: Data, B: Backend> {
    pub data: D,
    shape: VecZnxShape,
    pub _phantom: PhantomData<B>,
}

impl<D: HostDataRef, B: Backend> DigestU64 for VecZnxBig<D, B> {
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

impl<D: HostDataRef, B: Backend> ZnxView for VecZnxBig<D, B> {
    type Scalar = B::ScalarBig;
}

impl<D: Data, B: Backend> ZnxInfos for VecZnxBig<D, B> {
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

impl<D: Data, B: Backend> DataView for VecZnxBig<D, B> {
    type D = D;
    fn data(&self) -> &Self::D {
        &self.data
    }
}

impl<D: Data, B: Backend> DataViewMut for VecZnxBig<D, B> {
    fn data_mut(&mut self) -> &mut Self::D {
        &mut self.data
    }
}

impl<D: Data, B: Backend> VecZnxBig<D, B> {
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

impl<D: HostDataMut, B: Backend> ZnxZero for VecZnxBig<D, B>
where
    Self: ZnxViewMut,
    <Self as ZnxView>::Scalar: Zero + Copy,
{
    fn zero(&mut self) {
        self.raw_mut().fill(<Self as ZnxView>::Scalar::zero())
    }
    fn zero_at(&mut self, i: usize, j: usize) {
        self.at_mut(i, j).fill(<Self as ZnxView>::Scalar::zero());
    }
}

impl<B: Backend> VecZnxBig<<B as Backend>::OwnedBuf, B> {
    pub(crate) fn alloc(n: usize, cols: usize, size: usize) -> Self {
        let data: <B as Backend>::OwnedBuf = B::alloc_zeroed_bytes(B::bytes_of_vec_znx_big(n, cols, size));
        Self {
            data,
            shape: VecZnxShape::new(n, cols, size, size),
            _phantom: PhantomData,
        }
    }

    pub fn from_bytes(n: usize, cols: usize, size: usize, bytes: impl Into<Vec<u8>>) -> Self {
        let data: Vec<u8> = bytes.into();
        assert!(data.len() == B::bytes_of_vec_znx_big(n, cols, size));
        let data: <B as Backend>::OwnedBuf = B::from_host_bytes(&data);
        Self {
            data,
            shape: VecZnxShape::new(n, cols, size, size),
            _phantom: PhantomData,
        }
    }
}

impl<D: Data, B: Backend> VecZnxBig<D, B> {
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
pub type VecZnxBigOwned<B> = VecZnxBig<<B as Backend>::OwnedBuf, B>;
/// Shared backend-native borrow of a `VecZnxBig`.
pub type VecZnxBigBackendRef<'a, B> = VecZnxBig<<B as Backend>::BufRef<'a>, B>;
/// Mutable backend-native borrow of a `VecZnxBig`.
pub type VecZnxBigBackendMut<'a, B> = VecZnxBig<<B as Backend>::BufMut<'a>, B>;

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

impl<B: Backend> VecZnxBigToBackendRef<B> for VecZnxBig<B::OwnedBuf, B> {
    fn to_backend_ref(&self) -> VecZnxBigBackendRef<'_, B> {
        VecZnxBig {
            data: B::view(&self.data),
            shape: self.shape,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<'b, B: Backend + 'b> VecZnxBigToBackendRef<B> for &VecZnxBig<B::BufRef<'b>, B> {
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

impl<'b, B: Backend + 'b> VecZnxBigReborrowBackendRef<B> for VecZnxBig<B::BufMut<'b>, B> {
    fn reborrow_backend_ref(&self) -> VecZnxBigBackendRef<'_, B> {
        vec_znx_big_backend_ref_from_mut::<B>(self)
    }
}

/// Mutably borrow a backend-owned `VecZnxBig` using the backend's native view type.
pub trait VecZnxBigToBackendMut<B: Backend> {
    fn to_backend_mut(&mut self) -> VecZnxBigBackendMut<'_, B>;
}

impl<B: Backend> VecZnxBigToBackendMut<B> for VecZnxBig<B::OwnedBuf, B> {
    fn to_backend_mut(&mut self) -> VecZnxBigBackendMut<'_, B> {
        VecZnxBig {
            data: B::view_mut(&mut self.data),
            shape: self.shape,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<'b, B: Backend + 'b> VecZnxBigToBackendMut<B> for &mut VecZnxBig<B::BufMut<'b>, B> {
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

impl<'b, B: Backend + 'b> VecZnxBigReborrowBackendMut<B> for VecZnxBig<B::BufMut<'b>, B> {
    fn reborrow_backend_mut(&mut self) -> VecZnxBigBackendMut<'_, B> {
        VecZnxBig {
            data: B::view_mut_ref(&mut self.data),
            shape: self.shape,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<D: HostDataRef, B: Backend> fmt::Display for VecZnxBig<D, B> {
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
