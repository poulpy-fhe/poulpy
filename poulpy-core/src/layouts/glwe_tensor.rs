use poulpy_hal::layouts::{Backend, Data, HostDataRef, Ring, Standard, VecZnx, VecZnxToBackendMut, VecZnxToBackendRef};
use poulpy_hal::{AlignedBuf, alloc_aligned};

use crate::layouts::{
    Base2K, Degree, GLWE, GLWEBackendMut, GLWEBackendRef, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, LWEInfos, Rank,
    SetBase2k, TorusPrecision,
};
use poulpy_hal::layouts::ZnxWord;
use std::fmt;

#[derive(PartialEq, Eq, Clone)]
pub struct GLWETensor<D: Data, W: ZnxWord, Rg: Ring = Standard> {
    pub(crate) data: VecZnx<D, W, Rg>,
    pub(crate) k: TorusPrecision,
    pub(crate) base2k: Base2K,
    pub(crate) rank: Rank,
}

pub type GLWETensorBackendRef<'a, BE, Rg = Standard> = GLWETensor<<BE as Backend>::BufRef<'a>, <BE as Backend>::ZnxWord, Rg>;
pub type GLWETensorBackendMut<'a, BE, Rg = Standard> = GLWETensor<<BE as Backend>::BufMut<'a>, <BE as Backend>::ZnxWord, Rg>;

impl<D: Data, W: ZnxWord, Rg: Ring> SetBase2k for GLWETensor<D, W, Rg> {
    fn set_base2k(&mut self, base2k: Base2K) {
        self.base2k = base2k
    }
}

impl<D: Data, W: ZnxWord, Rg: Ring> GLWETensor<D, W, Rg> {
    pub fn data(&self) -> &VecZnx<D, W, Rg> {
        &self.data
    }
}

impl<D: Data, W: ZnxWord, Rg: Ring> GLWETensor<D, W, Rg> {
    pub fn data_mut(&mut self) -> &mut VecZnx<D, W, Rg> {
        &mut self.data
    }
}

impl<D: Data, W: ZnxWord, Rg: Ring> LWEInfos for GLWETensor<D, W, Rg> {
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

impl<D: Data, W: ZnxWord, Rg: Ring> GLWEInfos for GLWETensor<D, W, Rg> {
    ///NOTE: self.rank() != self.to_ref().rank() if self is of type [GLWETensor]
    fn rank(&self) -> Rank {
        self.rank
    }
}

impl<D: HostDataRef, W: ZnxWord, Rg: Ring> fmt::Debug for GLWETensor<D, W, Rg> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{self}")
    }
}

impl<D: HostDataRef, W: ZnxWord, Rg: Ring> fmt::Display for GLWETensor<D, W, Rg> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "GLWETensor: base2k={} k={}: {}", self.base2k().0, self.k().0, self.data)
    }
}

#[expect(
    dead_code,
    reason = "host-owned constructors are kept for serialization and host-only staging"
)]
impl<W: ZnxWord, Rg: Ring> GLWETensor<AlignedBuf, W, Rg> {
    pub(crate) fn alloc_from_infos<A>(infos: &A) -> Self
    where
        A: GLWEInfos,
    {
        Self::alloc(infos.n(), infos.base2k(), infos.k(), infos.rank())
    }

    pub(crate) fn alloc(n: Degree, base2k: Base2K, k: TorusPrecision, rank: Rank) -> Self {
        let cols: usize = rank.as_usize() + 1;
        let pairs: usize = (((cols + 1) * cols) >> 1).max(1);
        let size: usize = k.0.div_ceil(base2k.0) as usize;
        GLWETensor {
            data: VecZnx::from_data(
                alloc_aligned::<u8>(VecZnx::<AlignedBuf, W, Rg>::bytes_of(n.into(), pairs, size)),
                n.into(),
                pairs,
                size,
            ),
            base2k,
            rank,
            k,
        }
    }

    pub fn bytes_of_from_infos<A>(infos: &A) -> usize
    where
        A: GLWEInfos,
    {
        Self::bytes_of(infos.n(), infos.base2k(), infos.k(), infos.rank())
    }

    pub fn bytes_of(n: Degree, base2k: Base2K, k: TorusPrecision, rank: Rank) -> usize {
        let cols: usize = rank.as_usize() + 1;
        let pairs: usize = (((cols + 1) * cols) >> 1).max(1);
        VecZnx::<AlignedBuf, W, Rg>::bytes_of(n.into(), pairs, k.0.div_ceil(base2k.0) as usize)
    }
}

impl<BE: Backend, D: Data, Rg: Ring> GLWEToBackendRef<BE, Rg> for GLWETensor<D, BE::ZnxWord, Rg>
where
    VecZnx<D, BE::ZnxWord, Rg>: VecZnxToBackendRef<BE, Rg>,
{
    fn to_backend_ref(&self) -> GLWEBackendRef<'_, BE, Rg> {
        GLWE {
            noise: None,
            base2k: self.base2k,
            k: self.k,
            canonical: true,
            plaintext_meta: None,
            data: self.data.to_backend_ref(),
        }
    }
}

impl<BE: Backend, D: Data, Rg: Ring> GLWEToBackendRef<BE, Rg> for &GLWETensor<D, BE::ZnxWord, Rg>
where
    VecZnx<D, BE::ZnxWord, Rg>: VecZnxToBackendRef<BE, Rg>,
{
    fn to_backend_ref(&self) -> GLWEBackendRef<'_, BE, Rg> {
        GLWE {
            noise: None,
            base2k: self.base2k,
            k: self.k,
            canonical: true,
            plaintext_meta: None,
            data: self.data.to_backend_ref(),
        }
    }
}

impl<BE: Backend, D: Data, Rg: Ring> GLWEToBackendMut<BE, Rg> for GLWETensor<D, BE::ZnxWord, Rg>
where
    VecZnx<D, BE::ZnxWord, Rg>: VecZnxToBackendRef<BE, Rg> + VecZnxToBackendMut<BE, Rg>,
{
    fn set_noise(&mut self, _: Option<crate::ComponentNoise>) {}

    fn to_backend_mut(&mut self) -> GLWEBackendMut<'_, BE, Rg> {
        GLWE {
            noise: None,
            base2k: self.base2k,
            k: self.k,
            canonical: true,
            plaintext_meta: None,
            data: self.data.to_backend_mut(),
        }
    }

    fn set_canonical(&mut self, _canonical: bool) {}
}

impl<BE: Backend, Rg: Ring> GLWEToBackendRef<BE, Rg> for &mut GLWETensor<BE::BufMut<'_>, BE::ZnxWord, Rg> {
    fn to_backend_ref(&self) -> GLWEBackendRef<'_, BE, Rg> {
        GLWE {
            noise: None,
            base2k: self.base2k,
            k: self.k,
            canonical: true,
            plaintext_meta: None,
            data: poulpy_hal::layouts::vec_znx_backend_ref_from_mut::<BE, Rg>(&self.data),
        }
    }
}

impl<BE: Backend, Rg: Ring> GLWEToBackendMut<BE, Rg> for &mut GLWETensor<BE::BufMut<'_>, BE::ZnxWord, Rg> {
    fn set_noise(&mut self, _: Option<crate::ComponentNoise>) {}

    fn to_backend_mut(&mut self) -> GLWEBackendMut<'_, BE, Rg> {
        GLWE {
            noise: None,
            base2k: self.base2k,
            k: self.k,
            canonical: true,
            plaintext_meta: None,
            data: poulpy_hal::layouts::vec_znx_backend_mut_from_mut::<BE, Rg>(&mut self.data),
        }
    }

    fn set_canonical(&mut self, _canonical: bool) {}
}

impl<D: Data, W: ZnxWord, Rg: Ring> GLWETensor<D, W, Rg> {
    pub(crate) fn record_noise(&mut self, _: Option<crate::ComponentNoise>) {}
}
