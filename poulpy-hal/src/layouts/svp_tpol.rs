use std::{
    fmt,
    hash::{DefaultHasher, Hasher},
    marker::PhantomData,
};

use crate::layouts::{
    Backend, Data, DataView, DataViewMut, DftWord, DigestU64, HostDataRef, ScalarZnxShape, VecZnxInfos, ZnxInfos, ZnxView,
};

/// Transformed (hot-prep) scalar polynomial for scalar-vector products.
///
/// `SvpTPol` is the cheap-to-build prepared form, meant for short reuse or
/// one-shot use; [`SvpPPol`](crate::layouts::SvpPPol) is the packed form,
/// more expensive to build but optimized for amortized repeated apply. The
/// two are distinct types even where a backend gives them the same physical
/// storage shape.
///
/// Create via [`SvpPrepareTPol`](crate::api::SvpPrepareTPol) from a
/// coefficient-domain [`ScalarZnx`](crate::layouts::ScalarZnx), then consume
/// through the `svp_apply_tpol_*` family.
///
/// Ring degree `n` is always a power of two, so the DFT-domain layout has a
/// coefficient count that matches vector lane widths relative to buffer alignment.
#[repr(C)]
pub struct SvpTPol<D: Data, W: DftWord, B: Backend<DftWord = W>> {
    pub data: D,
    shape: ScalarZnxShape,
    pub _phantom: PhantomData<(W, B)>,
}

// Equality (and hashing, where provided) is defined directly on the
// representation: same shape, same buffer bytes. No `W`/`B` value is ever
// compared, so no bound on them is needed — in particular `Eq` holds even
// for non-`Eq` words like `f64` (byte equality is a total equivalence).
impl<D: Data, W: DftWord, B: Backend<DftWord = W>> PartialEq for SvpTPol<D, W, B> {
    fn eq(&self, other: &Self) -> bool {
        self.shape == other.shape && self.data == other.data
    }
}

impl<D: Data, W: DftWord, B: Backend<DftWord = W>> Eq for SvpTPol<D, W, B> {}

impl<D: Data + std::hash::Hash, W: DftWord, B: Backend<DftWord = W>> std::hash::Hash for SvpTPol<D, W, B> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.shape.hash(state);
        self.data.hash(state);
    }
}

impl<D: HostDataRef, W: DftWord, B: Backend<DftWord = W>> DigestU64 for SvpTPol<D, W, B> {
    fn digest_u64(&self) -> u64 {
        let mut h: DefaultHasher = DefaultHasher::new();
        h.write(self.data.as_ref());
        h.write_usize(self.n());
        h.write_usize(self.cols());
        h.finish()
    }
}

impl<D: HostDataRef, W: DftWord, B: Backend<DftWord = W>> ZnxView for SvpTPol<D, W, B> {
    type Scalar = W;
}

impl<D: Data, W: DftWord, B: Backend<DftWord = W>> ZnxInfos for SvpTPol<D, W, B> {
    fn n(&self) -> usize {
        self.shape.n()
    }

    fn size(&self) -> usize {
        1
    }

    fn poly_count(&self) -> usize {
        crate::layouts::checked_product(&[self.cols(), self.size()], "polynomial count")
    }
}

impl<D: Data, W: DftWord, B: Backend<DftWord = W>> VecZnxInfos for SvpTPol<D, W, B> {
    fn cols(&self) -> usize {
        self.shape.cols()
    }
}

impl<D: Data, W: DftWord, B: Backend<DftWord = W>> SvpTPol<D, W, B> {
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

impl<D: Data, W: DftWord, B: Backend<DftWord = W>> DataView for SvpTPol<D, W, B> {
    type D = D;
    fn data(&self) -> &Self::D {
        &self.data
    }
}

impl<D: Data, W: DftWord, B: Backend<DftWord = W>> DataViewMut for SvpTPol<D, W, B> {
    fn data_mut(&mut self) -> &mut Self::D {
        &mut self.data
    }
}

impl<D: Data, W: DftWord, B: Backend<DftWord = W>> SvpTPol<D, W, B> {
    /// Allocates a zero-initialized backend-owned `SvpTPol`.
    pub fn alloc(n: usize, cols: usize) -> SvpTPolOwned<B>
    where
        B: Backend<OwnedBuf = D>,
    {
        let data: <B as Backend>::OwnedBuf = B::alloc_zeroed_bytes(B::bytes_of_svp_tpol(n, cols));
        SvpTPol {
            data,
            shape: ScalarZnxShape::new(n, cols),
            _phantom: PhantomData,
        }
    }
}

/// Owned `SvpTPol` backed by a backend-owned buffer.
pub type SvpTPolOwned<B> = SvpTPol<<B as Backend>::OwnedBuf, <B as Backend>::DftWord, B>;
/// Shared backend-native borrow of an `SvpTPol`.
pub type SvpTPolBackendRef<'a, B> = SvpTPol<<B as Backend>::BufRef<'a>, <B as Backend>::DftWord, B>;
/// Mutable backend-native borrow of an `SvpTPol`.
pub type SvpTPolBackendMut<'a, B> = SvpTPol<<B as Backend>::BufMut<'a>, <B as Backend>::DftWord, B>;

/// Reborrow a mutable backend-native `SvpTPol` view as a shared backend-native view.
pub fn svp_tpol_backend_ref_from_mut<'a, 'b, B: Backend>(ppol: &'a SvpTPolBackendMut<'b, B>) -> SvpTPolBackendRef<'a, B> {
    SvpTPol {
        data: B::view_ref_mut(&ppol.data),
        shape: ppol.shape,
        _phantom: PhantomData,
    }
}

/// Borrow a backend-owned `SvpTPol` using the backend's native view type.
pub trait SvpTPolToBackendRef<B: Backend> {
    fn to_backend_ref(&self) -> SvpTPolBackendRef<'_, B>;
}

impl<B: Backend> SvpTPolToBackendRef<B> for SvpTPol<B::OwnedBuf, B::DftWord, B> {
    fn to_backend_ref(&self) -> SvpTPolBackendRef<'_, B> {
        SvpTPol {
            data: B::view(&self.data),
            shape: self.shape,
            _phantom: PhantomData,
        }
    }
}

impl<'b, B: Backend + 'b> SvpTPolToBackendRef<B> for &SvpTPol<B::BufRef<'b>, B::DftWord, B> {
    fn to_backend_ref(&self) -> SvpTPolBackendRef<'_, B> {
        SvpTPol {
            data: B::view_ref(&self.data),
            shape: self.shape,
            _phantom: PhantomData,
        }
    }
}

/// Reborrow an already backend-borrowed `SvpTPol` as a shared backend-native view.
pub trait SvpTPolReborrowBackendRef<B: Backend> {
    fn reborrow_backend_ref(&self) -> SvpTPolBackendRef<'_, B>;
}

impl<'b, B: Backend + 'b> SvpTPolReborrowBackendRef<B> for SvpTPol<B::BufMut<'b>, B::DftWord, B> {
    fn reborrow_backend_ref(&self) -> SvpTPolBackendRef<'_, B> {
        svp_tpol_backend_ref_from_mut::<B>(self)
    }
}

/// Mutably borrow a backend-owned `SvpTPol` using the backend's native view type.
pub trait SvpTPolToBackendMut<B: Backend> {
    fn to_backend_mut(&mut self) -> SvpTPolBackendMut<'_, B>;
}

impl<B: Backend> SvpTPolToBackendMut<B> for SvpTPol<B::OwnedBuf, B::DftWord, B> {
    fn to_backend_mut(&mut self) -> SvpTPolBackendMut<'_, B> {
        SvpTPol {
            data: B::view_mut(&mut self.data),
            shape: self.shape,
            _phantom: PhantomData,
        }
    }
}

impl<'b, B: Backend + 'b> SvpTPolToBackendMut<B> for &mut SvpTPol<B::BufMut<'b>, B::DftWord, B> {
    fn to_backend_mut(&mut self) -> SvpTPolBackendMut<'_, B> {
        SvpTPol {
            data: B::view_mut_ref(&mut self.data),
            shape: self.shape,
            _phantom: PhantomData,
        }
    }
}

/// Reborrow an already backend-borrowed `SvpTPol` as a mutable backend-native view.
pub trait SvpTPolReborrowBackendMut<B: Backend> {
    fn reborrow_backend_mut(&mut self) -> SvpTPolBackendMut<'_, B>;
}

impl<'b, B: Backend + 'b> SvpTPolReborrowBackendMut<B> for SvpTPol<B::BufMut<'b>, B::DftWord, B> {
    fn reborrow_backend_mut(&mut self) -> SvpTPolBackendMut<'_, B> {
        SvpTPol {
            data: B::view_mut_ref(&mut self.data),
            shape: self.shape,
            _phantom: PhantomData,
        }
    }
}

impl<D: Data, W: DftWord, B: Backend<DftWord = W>> SvpTPol<D, W, B> {
    pub fn from_data(data: D, n: usize, cols: usize) -> Self {
        Self {
            data,
            shape: ScalarZnxShape::new(n, cols),
            _phantom: PhantomData,
        }
    }
}

impl<D: HostDataRef, W: DftWord, B: Backend<DftWord = W>> fmt::Display for SvpTPol<D, W, B> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "SvpTPol(n={}, cols={})", self.n(), self.cols())?;

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

impl<D: Data, W: DftWord, B: Backend<DftWord = W>> SvpTPol<D, W, B> {
    /// Zero-copy re-tag of this container to a layout-compatible backend `B2`.
    ///
    /// The buffer moves as-is; only the type tag changes. Requires the
    /// [`SvpTPolLayoutCompatible`](crate::layouts::SvpTPolLayoutCompatible) marker declared by the backend
    /// pair. `D` is kept, so for further backend-native use `B2`'s buffer
    /// types must match `D` (true for all current CPU backends).
    pub fn into_backend<B2>(self) -> SvpTPol<D, W, B2>
    where
        B2: Backend<DftWord = W>,
        B: crate::layouts::SvpTPolLayoutCompatible<B2>,
    {
        let shape = self.shape;
        assert_eq!(
            B::bytes_of_svp_tpol(shape.n(), shape.cols()),
            B2::bytes_of_svp_tpol(shape.n(), shape.cols()),
            "into_backend: byte sizes diverge despite declared layout compatibility"
        );
        SvpTPol {
            data: self.data,
            shape,
            _phantom: PhantomData,
        }
    }
}
