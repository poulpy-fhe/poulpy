use poulpy_core::layouts::{Base2K, Degree, GLWE, GLWEInfos, LWEInfos, Rank, TorusPrecision};
use poulpy_hal::layouts::{Backend, Data, VecZnx, ZnxWord};

pub type GLWEWideSecretOwned<BE> = GLWEWideSecret<<BE as Backend>::OwnedBuf, <BE as Backend>::ZnxWord>;

/// An additive share of a secret whose coefficients are integers modulo
/// `2^k`, stored as the torus polynomials `s / 2^k`, one column per rank
/// index, in base `2^base2k`.
#[derive(Clone)]
pub struct GLWEWideSecret<D: Data, W: ZnxWord> {
    pub(crate) inner: GLWE<D, W>,
}

impl<D: Data, W: ZnxWord> GLWEWideSecret<D, W> {
    pub fn n(&self) -> Degree {
        self.inner.n()
    }

    pub fn base2k(&self) -> Base2K {
        self.inner.base2k()
    }

    pub fn k(&self) -> TorusPrecision {
        self.inner.k()
    }

    pub fn rank(&self) -> Rank {
        self.inner.rank() + 1
    }

    pub fn data(&self) -> &VecZnx<D, W> {
        self.inner.data()
    }
}
