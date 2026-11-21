use poulpy_hal::layouts::ZnxWord;
use poulpy_hal::layouts::{
    Backend, Data, HostDataMut, Ring, ScalarZnx, ScalarZnxToBackendMut, ScalarZnxToBackendRef, Standard, ZnxView, ZnxViewMut,
};
use poulpy_hal::{AlignedBuf, alloc_aligned};

use crate::{
    GetDistribution, GetDistributionMut,
    dist::Distribution,
    layouts::{Base2K, Degree, GLWEInfos, LWEInfos, Rank},
};

#[derive(PartialEq, Eq, Copy, Clone, Debug)]
pub struct GLWESecretLayout {
    pub n: Degree,
    pub rank: Rank,
}

impl LWEInfos for GLWESecretLayout {
    fn base2k(&self) -> Base2K {
        Base2K(0)
    }

    fn n(&self) -> Degree {
        self.n
    }

    fn max_size(&self) -> usize {
        unimplemented!("this method is not defined for secrets)")
    }

    fn k(&self) -> super::TorusPrecision {
        unimplemented!("this method is not defined for secrets")
    }
}
impl GLWEInfos for GLWESecretLayout {
    fn rank(&self) -> Rank {
        self.rank
    }
}

#[derive(PartialEq, Eq, Clone)]
pub struct GLWESecret<D: Data, W: ZnxWord, Rg: Ring = Standard> {
    pub(crate) data: ScalarZnx<D, W, Rg>,
    /// Distribution the base secret was sampled from, shared by all `rank`
    /// polynomial components. See [`Distribution`] for how this tag is
    /// propagated: it survives automorphisms, LWE/GLWE conversions, DFT
    /// preparation and backend transfers, and is *not* redefined for
    /// derived products such as [`GLWESecretTensor`](super::GLWESecretTensor).
    pub(crate) dist: Distribution,
}

pub type GLWESecretBackendRef<'a, BE, Rg = Standard> = GLWESecret<<BE as Backend>::BufRef<'a>, <BE as Backend>::ZnxWord, Rg>;
pub type GLWESecretBackendMut<'a, BE, Rg = Standard> = GLWESecret<<BE as Backend>::BufMut<'a>, <BE as Backend>::ZnxWord, Rg>;

impl<D: Data, W: ZnxWord, Rg: Ring> LWEInfos for GLWESecret<D, W, Rg> {
    fn base2k(&self) -> Base2K {
        Base2K(0)
    }

    fn n(&self) -> Degree {
        Degree(self.data.n() as u32)
    }

    fn max_size(&self) -> usize {
        1
    }

    fn k(&self) -> super::TorusPrecision {
        unimplemented!("this method is not defined for secrets")
    }
}

impl<D: Data, W: ZnxWord, Rg: Ring> GetDistribution for GLWESecret<D, W, Rg> {
    fn dist(&self) -> &Distribution {
        &self.dist
    }
}

impl<D: Data, W: ZnxWord, Rg: Ring> GetDistributionMut for GLWESecret<D, W, Rg> {
    fn dist_mut(&mut self) -> &mut Distribution {
        &mut self.dist
    }
}

impl<D: Data, W: ZnxWord, Rg: Ring> GLWEInfos for GLWESecret<D, W, Rg> {
    fn rank(&self) -> Rank {
        Rank(self.data.cols() as u32)
    }
}

impl<D: Data, W: ZnxWord, Rg: Ring> GLWESecret<D, W, Rg> {
    pub fn data(&self) -> &ScalarZnx<D, W, Rg> {
        &self.data
    }

    pub fn data_mut(&mut self) -> &mut ScalarZnx<D, W, Rg> {
        &mut self.data
    }

    /// Zero-cost rename when both backends share the same `OwnedBuf`.
    pub fn reinterpret<To>(self) -> GLWESecret<To::OwnedBuf, To::ZnxWord, Rg>
    where
        To: Backend<OwnedBuf = D, ZnxWord = W>,
    {
        let n = self.data.n();
        let cols = self.data.cols();
        let data = self.data.data;
        GLWESecret {
            data: ScalarZnx::from_data(data, n, cols),
            dist: self.dist,
        }
    }
}

#[expect(
    dead_code,
    reason = "host-owned constructors are kept for serialization and host-only staging"
)]
impl<W: ZnxWord, Rg: Ring> GLWESecret<AlignedBuf, W, Rg> {
    pub(crate) fn alloc_from_infos<A>(infos: &A) -> Self
    where
        A: GLWEInfos,
    {
        Self::alloc(infos.n(), infos.rank())
    }

    pub(crate) fn alloc(n: Degree, rank: Rank) -> Self {
        GLWESecret {
            data: ScalarZnx::from_data(
                alloc_aligned::<u8>(ScalarZnx::<AlignedBuf, W, Rg>::bytes_of(n.into(), rank.into())),
                n.into(),
                rank.into(),
            ),
            dist: Distribution::NONE,
        }
    }

    pub fn bytes_of_from_infos<A>(infos: &A) -> usize
    where
        A: GLWEInfos,
    {
        Self::bytes_of(infos.n(), infos.rank())
    }

    pub fn bytes_of(n: Degree, rank: Rank) -> usize {
        ScalarZnx::<AlignedBuf, W, Rg>::bytes_of(n.into(), rank.into())
    }
}

impl<D: HostDataMut, W: ZnxWord, Rg: Ring> GLWESecret<D, W, Rg> {
    /// Sets column `col` to the caller-provided binary `{0, 1}` coefficient
    /// vector, for structured binary secrets whose positions are chosen by the
    /// caller (e.g. the PaCo bootstrapping key).
    ///
    /// The distribution tag is shared by all columns.
    /// It is [`Distribution::BinaryFixed`] of their Hamming weight when every column has the same one, and [`Distribution::NONE`] otherwise, which encryption refuses.
    ///
    /// `coeffs` is taken as `i64` for caller convenience rather than as the
    /// layout's own word; the assert restricts the values to `{0, 1}`, so
    /// converting them into any [`ZnxWord`] is exact.
    pub fn fill_binary_coeffs(&mut self, col: usize, coeffs: &[i64]) {
        assert!(coeffs.iter().all(|&x| x == 0 || x == 1), "coefficients must be binary");
        let dst = self.data.at_mut(col, 0);
        assert_eq!(dst.len(), coeffs.len(), "coefficient length must equal the ring degree");
        dst.iter_mut().zip(coeffs.iter()).for_each(|(d, &c)| *d = W::from_i64(c));
        let weight = |col: usize| self.data.at(col, 0).iter().filter(|&&x| x != W::zero()).count();
        let h = weight(col);
        self.dist = if (0..self.data.cols()).all(|c| weight(c) == h) {
            Distribution::BinaryFixed(h)
        } else {
            Distribution::NONE
        };
    }
}

pub trait GLWESecretToBackendMut<BE: Backend, Rg: Ring = Standard>: GLWESecretToBackendRef<BE, Rg> {
    fn to_backend_mut(&mut self) -> GLWESecretBackendMut<'_, BE, Rg>;
}

impl<BE: Backend, Rg: Ring> GLWESecretToBackendMut<BE, Rg> for GLWESecret<BE::OwnedBuf, BE::ZnxWord, Rg> {
    fn to_backend_mut(&mut self) -> GLWESecretBackendMut<'_, BE, Rg> {
        GLWESecret {
            dist: self.dist,
            data: <ScalarZnx<BE::OwnedBuf, BE::ZnxWord, Rg> as ScalarZnxToBackendMut<BE, Rg>>::to_backend_mut(&mut self.data),
        }
    }
}

pub trait GLWESecretToBackendRef<BE: Backend, Rg: Ring = Standard> {
    fn to_backend_ref(&self) -> GLWESecretBackendRef<'_, BE, Rg>;
}

impl<BE: Backend, Rg: Ring> GLWESecretToBackendRef<BE, Rg> for GLWESecret<BE::OwnedBuf, BE::ZnxWord, Rg> {
    fn to_backend_ref(&self) -> GLWESecretBackendRef<'_, BE, Rg> {
        GLWESecret {
            data: <ScalarZnx<BE::OwnedBuf, BE::ZnxWord, Rg> as ScalarZnxToBackendRef<BE, Rg>>::to_backend_ref(&self.data),
            dist: self.dist,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn binary_coeffs_are_tagged_only_when_every_column_shares_a_weight() {
        let mut sk = GLWESecret::<AlignedBuf, i64, Standard>::alloc(Degree(8), Rank(2));
        sk.fill_binary_coeffs(0, &[1, 1, 0, 0, 0, 0, 0, 0]);
        sk.fill_binary_coeffs(1, &[1, 1, 1, 1, 1, 0, 0, 0]);
        assert!(matches!(sk.dist(), Distribution::NONE));
        // A light column is not hidden behind the weight written last.
        sk.fill_binary_coeffs(1, &[0, 0, 0, 1, 1, 0, 0, 0]);
        assert!(matches!(sk.dist(), Distribution::BinaryFixed(2)));
    }
}
