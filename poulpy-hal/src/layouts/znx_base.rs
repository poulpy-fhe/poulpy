use std::fmt::{Debug, Display};

use crate::{
    layouts::{Data, HostDataMut, HostDataRef},
    source::Source,
};
use bytemuck::Pod;
use rand_distr::num_traits::Zero;

/// Shape shared by every polynomial container, vector- or matrix-shaped.
///
/// Deliberately holds only the dimensions that mean the same thing for both
/// families. Column counts do not: a vector container has one flat `cols`,
/// while a matrix container has `cols_in` and `cols_out`. Those live on
/// [`VecZnxInfos`] and [`MatZnxInfos`] respectively, so a caller cannot ask a
/// `VmpPMat` for "its" column count and silently get `cols_in`.
pub trait ZnxInfos {
    /// Returns the ring degree `N` of the polynomials in `Z[X]/(X^N + 1)`.
    fn n(&self) -> usize;

    /// Returns the base two logarithm of the ring dimension of the polynomials.
    fn log_n(&self) -> usize {
        (usize::BITS - (self.n() - 1).leading_zeros()) as _
    }

    /// Returns the number of limbs per polynomial.
    fn size(&self) -> usize;

    /// Returns the total number of small polynomials in the element view.
    ///
    /// Required rather than defaulted: the product differs per family, and a
    /// default here would silently under-count a matrix container.
    fn poly_count(&self) -> usize;
}

/// Shape of a vector-shaped container: one row of `cols` polynomial columns.
///
/// Coefficients are limb-major, column-minor, which is what makes the
/// `(col, limb)` indexing of [`ZnxView`] well defined.
pub trait VecZnxInfos: ZnxInfos {
    /// Returns the number of polynomial columns.
    fn cols(&self) -> usize;
}

/// Shape of a matrix-shaped container: `rows` x `cols_in` blocks, each holding
/// `cols_out` polynomial columns.
pub trait MatZnxInfos: ZnxInfos {
    /// Returns the number of rows.
    fn rows(&self) -> usize;

    /// Returns the number of input columns.
    fn cols_in(&self) -> usize;

    /// Returns the number of output columns.
    fn cols_out(&self) -> usize;
}

/// Reinterprets a byte buffer as `span` scalars.
///
/// # Panics
///
/// Panics if the buffer is misaligned for `S` or shorter than `span` scalars.
pub(crate) fn raw_scalars<S: Pod>(data: &[u8], span: usize) -> &[S] {
    let ptr: *const u8 = data.as_ptr();
    assert!(
        (ptr as usize).is_multiple_of(align_of::<S>()),
        "buffer not aligned to align_of::<Scalar>() = {}",
        align_of::<S>()
    );
    assert!(
        span.checked_mul(size_of::<S>())
            .expect("element view byte size overflows usize")
            <= data.len(),
        "element view ({} scalars of {} bytes) exceeds the {}-byte buffer: this container has no element view for its word type",
        span,
        size_of::<S>(),
        data.len()
    );
    unsafe { std::slice::from_raw_parts(ptr as *const S, span) }
}

/// Mutable counterpart of [`raw_scalars`].
pub(crate) fn raw_scalars_mut<S: Pod>(data: &mut [u8], span: usize) -> &mut [S] {
    let len: usize = data.len();
    let ptr: *mut u8 = data.as_mut_ptr();
    assert!(
        (ptr as usize).is_multiple_of(align_of::<S>()),
        "buffer not aligned to align_of::<Scalar>() = {}",
        align_of::<S>()
    );
    assert!(
        span.checked_mul(size_of::<S>())
            .expect("element view byte size overflows usize")
            <= len,
        "element view ({} scalars of {} bytes) exceeds the {}-byte buffer: this container has no element view for its word type",
        span,
        size_of::<S>(),
        len
    );
    unsafe { std::slice::from_raw_parts_mut(ptr as *mut S, span) }
}

/// Scalar span of a container's element view: `n * poly_count`.
pub(crate) fn element_view_span<T: ZnxInfos + ?Sized>(infos: &T) -> usize {
    infos
        .n()
        .checked_mul(infos.poly_count())
        .expect("element view scalar count overflows usize")
}

/// Read-only access to the underlying data container of a layout type.
pub trait DataView {
    type D: Data;
    fn data(&self) -> &Self::D;
}

/// Mutable access to the underlying data container of a layout type.
pub trait DataViewMut: DataView {
    fn data_mut(&mut self) -> &mut Self::D;
}

/// Read-only view into a polynomial container's coefficient data.
///
/// Coefficients are stored in a **limb-major, column-minor** layout.
/// For a container with `cols` columns and `size` limbs, limb `j` of
/// column `i` starts at scalar offset `n * (j * cols + i)`.
///
/// The associated `Scalar` type is the container's word type `W`
/// (`i64` by default for coefficient-domain types, the backend-declared
/// `DftWord`/`BigWord` for DFT/big representations).
pub trait ZnxView: VecZnxInfos + DataView<D: HostDataRef> {
    type Scalar: Copy + Zero + Display + Debug + Pod;

    /// Rejects generic element access when a packed backend has no dense [`Self::Scalar`] view.
    #[doc(hidden)]
    fn validate_element_view(&self) {}

    /// Returns a non-mutable pointer to the underlying coefficients array.
    fn as_ptr(&self) -> *const Self::Scalar {
        self.validate_element_view();
        let ptr: *const u8 = self.data().as_ref().as_ptr();
        assert!(
            (ptr as usize).is_multiple_of(align_of::<Self::Scalar>()),
            "buffer not aligned to align_of::<Scalar>() = {}",
            align_of::<Self::Scalar>()
        );
        ptr as *const Self::Scalar
    }

    /// Returns a non-mutable reference to the entire underlying coefficient array.
    ///
    /// # Panics
    ///
    /// Panics if the buffer is smaller than the element view (`n * poly_count`
    /// scalars), which happens when the backend sizes this container below its
    /// word type's element view (the word is then a sizing/identity token only).
    fn raw(&self) -> &[Self::Scalar] {
        self.validate_element_view();
        raw_scalars(self.data().as_ref(), element_view_span(self))
    }

    /// Returns a non-mutable pointer starting at the j-th small polynomial of the i-th column.
    fn at_ptr(&self, i: usize, j: usize) -> *const Self::Scalar {
        self.validate_element_view();
        assert!(i < self.cols(), "cols: {} >= self.cols(): {}", i, self.cols());
        assert!(j < self.size(), "size: {} >= self.size(): {}", j, self.size());
        let offset: usize = j
            .checked_mul(self.cols())
            .and_then(|x| x.checked_add(i))
            .and_then(|x| x.checked_mul(self.n()))
            .expect("element view offset overflows usize");
        assert!(
            offset
                .checked_add(self.n())
                .and_then(|x| x.checked_mul(size_of::<Self::Scalar>()))
                .expect("element view byte size overflows usize")
                <= self.data().as_ref().len(),
            "element view of block ({}, {}) exceeds the {}-byte buffer: this container has no element view for its word type",
            i,
            j,
            self.data().as_ref().len()
        );
        unsafe { self.as_ptr().add(offset) }
    }

    /// Returns non-mutable reference to the (i, j)-th small polynomial.
    fn at(&self, i: usize, j: usize) -> &[Self::Scalar] {
        unsafe { std::slice::from_raw_parts(self.at_ptr(i, j), self.n()) }
    }
}

/// Mutable view into a polynomial container's coefficient data.
///
/// Extends [`ZnxView`] with mutable pointer and slice accessors.
pub trait ZnxViewMut: ZnxView + DataViewMut<D: HostDataMut> {
    /// Returns a mutable pointer to the underlying coefficients array.
    fn as_mut_ptr(&mut self) -> *mut Self::Scalar {
        self.validate_element_view();
        let ptr: *mut u8 = self.data_mut().as_mut().as_mut_ptr();
        assert!(
            (ptr as usize).is_multiple_of(align_of::<Self::Scalar>()),
            "buffer not aligned to align_of::<Scalar>() = {}",
            align_of::<Self::Scalar>()
        );
        ptr as *mut Self::Scalar
    }

    /// Returns a mutable reference to the entire underlying coefficient array.
    ///
    /// # Panics
    ///
    /// Panics if the buffer is smaller than the element view (see [`ZnxView::raw`]).
    fn raw_mut(&mut self) -> &mut [Self::Scalar] {
        self.validate_element_view();
        let span: usize = element_view_span(self);
        raw_scalars_mut(self.data_mut().as_mut(), span)
    }

    /// Returns a mutable pointer starting at the j-th small polynomial of the i-th column.
    fn at_mut_ptr(&mut self, i: usize, j: usize) -> *mut Self::Scalar {
        self.validate_element_view();
        assert!(i < self.cols(), "cols: {} >= self.cols(): {}", i, self.cols());
        assert!(j < self.size(), "size: {} >= self.size(): {}", j, self.size());
        let offset: usize = j
            .checked_mul(self.cols())
            .and_then(|x| x.checked_add(i))
            .and_then(|x| x.checked_mul(self.n()))
            .expect("element view offset overflows usize");
        assert!(
            offset
                .checked_add(self.n())
                .and_then(|x| x.checked_mul(size_of::<Self::Scalar>()))
                .expect("element view byte size overflows usize")
                <= self.data().as_ref().len(),
            "element view of block ({}, {}) exceeds the {}-byte buffer: this container has no element view for its word type",
            i,
            j,
            self.data().as_ref().len()
        );
        unsafe { self.as_mut_ptr().add(offset) }
    }

    /// Returns mutable reference to the (i, j)-th small polynomial.
    fn at_mut(&mut self, i: usize, j: usize) -> &mut [Self::Scalar] {
        unsafe { std::slice::from_raw_parts_mut(self.at_mut_ptr(i, j), self.n()) }
    }
}

// Note: Cannot provide blanket impl of ZnxView because Scalar is not known.
impl<T> ZnxViewMut for T where T: ZnxView + DataViewMut<D: HostDataMut> {}

/// Zero-fill operations for polynomial containers.
pub trait ZnxZero
where
    Self: Sized,
{
    /// Sets all coefficients across all columns and limbs to zero.
    fn zero(&mut self);
    /// Sets all coefficients of limb `j` of column `i` to zero.
    fn zero_at(&mut self, i: usize, j: usize);
}

/// Fill a polynomial container with uniformly distributed random coefficients.
pub trait FillUniform {
    /// Fills all coefficients with values drawn uniformly from
    /// `[-2^(log_bound-1), 2^(log_bound-1))`.
    ///
    /// When `log_bound == 64`, all 64 bits are used (full `i64` range).
    ///
    /// # Panics
    ///
    /// Panics if `log_bound == 0`.
    fn fill_uniform(&mut self, log_bound: usize, source: &mut Source);
}
