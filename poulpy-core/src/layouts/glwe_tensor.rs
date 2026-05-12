use poulpy_hal::{
    layouts::{Backend, Data, FillUniform, HostDataMut, HostDataRef, VecZnx, VecZnxToBackendMut, VecZnxToBackendRef},
    source::Source,
};

use crate::layouts::{
    Base2K, Degree, GLWE, GLWEBackendMut, GLWEBackendRef, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, LWEInfos, Rank,
    SetLWEInfos, TorusPrecision,
};
use std::fmt;

#[derive(PartialEq, Eq, Clone)]
pub struct GLWETensor<D: Data> {
    pub(crate) data: VecZnx<D>,
    pub(crate) base2k: Base2K,
    pub(crate) rank: Rank,
}

pub type GLWETensorBackendRef<'a, BE> = GLWETensor<<BE as Backend>::BufRef<'a>>;
pub type GLWETensorBackendMut<'a, BE> = GLWETensor<<BE as Backend>::BufMut<'a>>;

impl<D: HostDataMut> SetLWEInfos for GLWETensor<D> {
    fn set_base2k(&mut self, base2k: Base2K) {
        self.base2k = base2k
    }
}

impl<D: HostDataRef> GLWETensor<D> {
    pub fn data(&self) -> &VecZnx<D> {
        &self.data
    }
}

impl<D: HostDataMut> GLWETensor<D> {
    pub fn data_mut(&mut self) -> &mut VecZnx<D> {
        &mut self.data
    }
}

impl<D: Data> LWEInfos for GLWETensor<D> {
    fn base2k(&self) -> Base2K {
        self.base2k
    }

    fn n(&self) -> Degree {
        Degree(self.data.n() as u32)
    }

    fn size(&self) -> usize {
        self.data.size()
    }
}

impl<D: Data> LWEInfos for &mut GLWETensor<D> {
    fn base2k(&self) -> Base2K {
        self.base2k
    }

    fn n(&self) -> Degree {
        Degree(self.data.n() as u32)
    }

    fn size(&self) -> usize {
        self.data.size()
    }
}

impl<D: Data> GLWEInfos for GLWETensor<D> {
    ///NOTE: self.rank() != self.to_ref().rank() if self is of type [GLWETensor]
    fn rank(&self) -> Rank {
        self.rank
    }
}

impl<D: Data> GLWEInfos for &mut GLWETensor<D> {
    fn rank(&self) -> Rank {
        self.rank
    }
}

impl<D: HostDataRef> fmt::Debug for GLWETensor<D> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{self}")
    }
}

impl<D: HostDataRef> fmt::Display for GLWETensor<D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "GLWETensor: base2k={} k={}: {}",
            self.base2k().0,
            self.max_k().0,
            self.data
        )
    }
}

impl<D: HostDataMut> FillUniform for GLWETensor<D> {
    fn fill_uniform(&mut self, log_bound: usize, source: &mut Source) {
        self.data.fill_uniform(log_bound, source);
    }
}

#[expect(
    dead_code,
    reason = "host-owned constructors are kept for serialization and host-only staging"
)]
impl GLWETensor<Vec<u8>> {
    pub(crate) fn alloc_from_infos<A>(infos: &A) -> Self
    where
        A: GLWEInfos,
    {
        Self::alloc(infos.n(), infos.base2k(), infos.max_k(), infos.rank())
    }

    pub(crate) fn alloc(n: Degree, base2k: Base2K, k: TorusPrecision, rank: Rank) -> Self {
        let cols: usize = rank.as_usize() + 1;
        let pairs: usize = (((cols + 1) * cols) >> 1).max(1);
        let size: usize = k.0.div_ceil(base2k.0) as usize;
        GLWETensor {
            data: VecZnx::from_data(
                poulpy_hal::layouts::HostBytesBackend::alloc_bytes(VecZnx::<Vec<u8>>::bytes_of(n.into(), pairs, size)),
                n.into(),
                pairs,
                size,
            ),
            base2k,
            rank,
        }
    }

    pub fn bytes_of_from_infos<A>(infos: &A) -> usize
    where
        A: GLWEInfos,
    {
        Self::bytes_of(infos.n(), infos.base2k(), infos.max_k(), infos.rank())
    }

    pub fn bytes_of(n: Degree, base2k: Base2K, k: TorusPrecision, rank: Rank) -> usize {
        let cols: usize = rank.as_usize() + 1;
        let pairs: usize = (((cols + 1) * cols) >> 1).max(1);
        VecZnx::bytes_of(n.into(), pairs, k.0.div_ceil(base2k.0) as usize)
    }
}

impl<BE: Backend, D: Data> GLWEToBackendRef<BE> for GLWETensor<D>
where
    VecZnx<D>: VecZnxToBackendRef<BE>,
{
    fn to_backend_ref(&self) -> GLWEBackendRef<'_, BE> {
        GLWE {
            base2k: self.base2k,
            data: self.data.to_backend_ref(),
        }
    }
}

impl<BE: Backend, D: Data> GLWEToBackendRef<BE> for &GLWETensor<D>
where
    VecZnx<D>: VecZnxToBackendRef<BE>,
{
    fn to_backend_ref(&self) -> GLWEBackendRef<'_, BE> {
        GLWE {
            base2k: self.base2k,
            data: self.data.to_backend_ref(),
        }
    }
}

impl<BE: Backend, D: Data> GLWEToBackendMut<BE> for GLWETensor<D>
where
    VecZnx<D>: VecZnxToBackendRef<BE> + VecZnxToBackendMut<BE>,
{
    fn to_backend_mut(&mut self) -> GLWEBackendMut<'_, BE> {
        GLWE {
            base2k: self.base2k,
            data: self.data.to_backend_mut(),
        }
    }
}

impl<'b, BE: Backend + 'b> GLWEToBackendRef<BE> for &mut GLWETensor<BE::BufMut<'b>> {
    fn to_backend_ref(&self) -> GLWEBackendRef<'_, BE> {
        GLWE {
            base2k: self.base2k,
            data: poulpy_hal::layouts::vec_znx_backend_ref_from_mut::<BE>(&self.data),
        }
    }
}

impl<'b, BE: Backend + 'b> GLWEToBackendMut<BE> for &mut GLWETensor<BE::BufMut<'b>> {
    fn to_backend_mut(&mut self) -> GLWEBackendMut<'_, BE> {
        GLWE {
            base2k: self.base2k,
            data: poulpy_hal::layouts::vec_znx_backend_mut_from_mut::<BE>(&mut self.data),
        }
    }
}
