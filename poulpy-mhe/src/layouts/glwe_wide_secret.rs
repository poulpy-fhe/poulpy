use poulpy_core::layouts::{Base2K, Degree, GLWE, GLWEInfos, LWEInfos, Rank, TorusPrecision, prepared::GLWESecretPrepared};
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

pub type GLWEWideSecretPreparedOwned<BE> = GLWEWideSecretPrepared<<BE as Backend>::OwnedBuf, BE>;

/// A [`GLWEWideSecret`] split into its base-`2^base2k` digits, each prepared
/// as a small secret: `digits[l]` has weight `2^(k - (l + 1) * base2k)`, the
/// last one weight 1.
pub struct GLWEWideSecretPrepared<D: Data, BE: Backend> {
    pub(crate) digits: Vec<GLWESecretPrepared<D, BE>>,
    pub(crate) base2k: Base2K,
    pub(crate) k: TorusPrecision,
}

impl<D: Data, BE: Backend> GLWEWideSecretPrepared<D, BE> {
    pub fn base2k(&self) -> Base2K {
        self.base2k
    }

    pub fn k(&self) -> TorusPrecision {
        self.k
    }

    pub fn rank(&self) -> Rank {
        self.digits[0].rank()
    }
}
