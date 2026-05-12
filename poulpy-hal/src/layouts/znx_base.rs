use std::fmt::{Debug, Display};

use crate::{
    layouts::{Data, HostDataMut, HostDataRef},
    source::Source,
};
use bytemuck::Pod;
use rand_distr::num_traits::Zero;

/// Metadata trait providing the shape of a polynomial container.
///
/// Every layout type in this crate implements `ZnxInfos` to expose its
/// ring degree, row/column counts, and limb count.
pub trait ZnxInfos {
    /// Returns the ring degree `N` of the polynomials in `Z[X]/(X^N + 1)`.
    fn n(&self) -> usize;

    /// Returns the base two logarithm of the ring dimension of the polynomials.
    fn log_n(&self) -> usize {
        (usize::BITS - (self.n() - 1).leading_zeros()) as _
    }

    /// Returns the number of rows.
    fn rows(&self) -> usize;

    /// Returns the number of polynomials in each row.
    fn cols(&self) -> usize;

    /// Returns the number of limbs per polynomial.
    fn size(&self) -> usize;

    /// Returns the total number of small polynomials.
    fn poly_count(&self) -> usize {
        self.rows() * self.cols() * self.size()
    }
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
/// The associated `Scalar` type is `i64` for coefficient-domain types
/// and a backend-specific type for DFT/big representations.
pub trait ZnxView: ZnxInfos + DataView<D: HostDataRef> {
    type Scalar: Copy + Zero + Display + Debug + Pod;

    /// Returns a non-mutable pointer to the underlying coefficients array.
    fn as_ptr(&self) -> *const Self::Scalar {
        self.data().as_ref().as_ptr() as *const Self::Scalar
    }

    /// Returns a non-mutable reference to the entire underlying coefficient array.
    fn raw(&self) -> &[Self::Scalar] {
        unsafe { std::slice::from_raw_parts(self.as_ptr(), self.n() * self.poly_count()) }
    }

    /// Returns a non-mutable pointer starting at the j-th small polynomial of the i-th column.
    fn at_ptr(&self, i: usize, j: usize) -> *const Self::Scalar {
        assert!(i < self.cols(), "cols: {} >= self.cols(): {}", i, self.cols());
        assert!(j < self.size(), "size: {} >= self.size(): {}", j, self.size());
        let offset: usize = self.n() * (j * self.cols() + i);
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
        self.data_mut().as_mut().as_mut_ptr() as *mut Self::Scalar
    }

    /// Returns a mutable reference to the entire underlying coefficient array.
    fn raw_mut(&mut self) -> &mut [Self::Scalar] {
        unsafe { std::slice::from_raw_parts_mut(self.as_mut_ptr(), self.n() * self.poly_count()) }
    }

    /// Returns a mutable pointer starting at the j-th small polynomial of the i-th column.
    fn at_mut_ptr(&mut self, i: usize, j: usize) -> *mut Self::Scalar {
        assert!(i < self.cols(), "cols: {} >= self.cols(): {}", i, self.cols());
        assert!(j < self.size(), "size: {} >= self.size(): {}", j, self.size());
        let offset: usize = self.n() * (j * self.cols() + i);
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
