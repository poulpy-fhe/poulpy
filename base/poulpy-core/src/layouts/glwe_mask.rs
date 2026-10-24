use poulpy_hal::layouts::{Backend, Data, VecZnx, VecZnxToBackendRef, ZnxWord};

use crate::layouts::{Base2K, Degree, GLWE, GLWEInfos, LWEInfos, Rank, TorusPrecision};

/// The `rank` mask polynomials of a [`GLWE`] ciphertext, without its body: what a
/// party that only computes on the mask, such as a key-switch share, receives.
///
/// Mask polynomial `j` is column `offset + j` of the [`VecZnx`]: an allocated mask
/// stores them from column 0, and [`GLWEMaskToBackendRef`] views the mask of a
/// [`GLWE`] in place, from its column 1. The canonical flag follows the [`GLWE`]'s.
pub struct GLWEMask<D: Data, W: ZnxWord> {
    pub(crate) data: VecZnx<D, W>,
    pub(crate) offset: usize,
    pub(crate) k: TorusPrecision,
    pub(crate) base2k: Base2K,
    pub(crate) canonical: bool,
}

pub type GLWEMaskBackendRef<'a, BE> = GLWEMask<<BE as Backend>::BufRef<'a>, <BE as Backend>::ZnxWord>;

impl<D: Data, W: ZnxWord> GLWEMask<D, W> {
    /// The underlying [`VecZnx`], whose column [`Self::col`]`(j)` holds mask polynomial `j`.
    pub fn data(&self) -> &VecZnx<D, W> {
        &self.data
    }

    pub fn data_mut(&mut self) -> &mut VecZnx<D, W> {
        &mut self.data
    }

    /// The column of [`Self::data`] holding mask polynomial `j`.
    pub fn col(&self, j: usize) -> usize {
        self.offset + j
    }

    pub fn is_canonical(&self) -> bool {
        self.canonical
    }

    /// For data written directly into the limbs.
    pub fn set_canonical(&mut self, canonical: bool) {
        self.canonical = canonical
    }
}

impl<D: Data, W: ZnxWord> LWEInfos for GLWEMask<D, W> {
    fn base2k(&self) -> Base2K {
        self.base2k
    }

    fn n(&self) -> Degree {
        Degree(self.data.n() as u32)
    }

    fn max_size(&self) -> usize {
        self.data.size()
    }

    fn k(&self) -> TorusPrecision {
        self.k
    }
}

impl<D: Data, W: ZnxWord> GLWEInfos for GLWEMask<D, W> {
    fn rank(&self) -> Rank {
        Rank((self.data.cols() - self.offset) as u32)
    }
}

/// Backend view of a GLWE mask: of a [`GLWEMask`], or of the mask of a [`GLWE`].
pub trait GLWEMaskToBackendRef<BE: Backend> {
    fn to_mask_backend_ref(&self) -> GLWEMaskBackendRef<'_, BE>;
}

impl<BE: Backend, D: Data> GLWEMaskToBackendRef<BE> for GLWEMask<D, BE::ZnxWord>
where
    VecZnx<D, BE::ZnxWord>: VecZnxToBackendRef<BE>,
{
    fn to_mask_backend_ref(&self) -> GLWEMaskBackendRef<'_, BE> {
        GLWEMask {
            data: self.data.to_backend_ref(),
            offset: self.offset,
            k: self.k,
            base2k: self.base2k,
            canonical: self.canonical,
        }
    }
}

impl<BE: Backend, D: Data> GLWEMaskToBackendRef<BE> for GLWE<D, BE::ZnxWord>
where
    VecZnx<D, BE::ZnxWord>: VecZnxToBackendRef<BE>,
{
    fn to_mask_backend_ref(&self) -> GLWEMaskBackendRef<'_, BE> {
        GLWEMask {
            data: self.data.to_backend_ref(),
            offset: 1,
            k: self.k,
            base2k: self.base2k,
            canonical: self.canonical,
        }
    }
}
