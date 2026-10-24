use poulpy_hal::layouts::VecZnxDftToBackendMut;
use poulpy_hal::layouts::VecZnxDftToBackendRef;
use poulpy_hal::{
    api::{VecZnxDftAlloc, VecZnxDftBytesOf},
    layouts::{Backend, Data, ScratchArena, VecZnxDft},
};

use crate::layouts::{Base2K, Degree, GLWEInfos, GLWEToBackendRef, GetDegree, LWEInfos, Rank, TorusPrecision};
use crate::layouts::{GLWELayout, operand_degree};

/// DFT-domain (prepared) variant of [`GLWE`](crate::layouts::GLWE).
///
/// Stores polynomials in the frequency domain of the backend's DFT/NTT
/// transform, enabling O(N log N) polynomial multiplication.
/// Tied to a specific backend via `B: Backend`.
#[derive(PartialEq)]
pub struct GLWEPrepared<D: Data, B: Backend> {
    pub(crate) noise: Option<crate::ComponentNoise>,
    pub(crate) data: VecZnxDft<D, B::DftWord, B>,
    pub(crate) k: TorusPrecision,
    pub(crate) base2k: Base2K,
}

pub type GLWEPreparedBackendRef<'a, B> = GLWEPrepared<<B as Backend>::BufRef<'a>, B>;
pub type GLWEPreparedBackendMut<'a, B> = GLWEPrepared<<B as Backend>::BufMut<'a>, B>;

impl<D: Data, B: Backend> LWEInfos for GLWEPrepared<D, B> {
    fn noise(&self) -> Option<crate::ComponentNoise> {
        self.noise.clone()
    }

    fn base2k(&self) -> Base2K {
        self.base2k
    }

    fn max_size(&self) -> usize {
        self.data.size()
    }

    fn n(&self) -> Degree {
        Degree(self.data.n() as u32)
    }

    fn k(&self) -> TorusPrecision {
        self.k
    }
}

impl<D: Data, B: Backend> GLWEInfos for GLWEPrepared<D, B> {
    fn rank(&self) -> Rank {
        Rank(self.data.cols() as u32 - 1)
    }
}

/// Trait for allocating and preparing DFT-domain GLWE ciphertexts.
pub trait GLWEPreparedFactory<B: Backend>
where
    Self: GetDegree + VecZnxDftAlloc<B> + VecZnxDftBytesOf,
{
    /// Allocates a new prepared GLWE with the given parameters.
    fn glwe_prepared_alloc(&self, base2k: Base2K, k: TorusPrecision, rank: Rank) -> GLWEPrepared<B::OwnedBuf, B> {
        self.glwe_prepared_alloc_from_infos(&GLWELayout {
            n: self.ring_degree(),
            base2k,
            k,
            rank,
        })
    }

    fn glwe_prepared_alloc_from_infos<A>(&self, infos: &A) -> GLWEPrepared<B::OwnedBuf, B>
    where
        A: GLWEInfos,
    {
        let n: usize = operand_degree(self.ring_degree().as_usize(), &[infos.n()]);
        GLWEPrepared {
            noise: None,
            data: self.vec_znx_dft_alloc(n, (infos.rank() + 1).into(), infos.size()),
            base2k: infos.base2k(),
            k: infos.k(),
        }
    }

    fn glwe_prepared_bytes_of(&self, base2k: Base2K, k: TorusPrecision, rank: Rank) -> usize {
        self.glwe_prepared_bytes_of_from_infos(&GLWELayout {
            n: self.ring_degree(),
            base2k,
            k,
            rank,
        })
    }

    fn glwe_prepared_bytes_of_from_infos<A>(&self, infos: &A) -> usize
    where
        A: GLWEInfos,
    {
        let n: usize = operand_degree(self.ring_degree().as_usize(), &[infos.n()]);
        self.bytes_of_vec_znx_dft(n, (infos.rank() + 1).into(), infos.size())
    }

    fn glwe_prepare_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GLWEInfos;

    fn glwe_prepare<R, O>(&self, res: &mut R, other: &O, scratch: &mut ScratchArena<'_, B>)
    where
        R: GLWEPreparedToBackendMut<B>,
        O: GLWEToBackendRef<B> + GLWEInfos;
}

// module-only API: allocation/size helpers are provided by `GLWEPreparedFactory` on `Module`.

// module-only API: preparation is provided by `GLWEPreparedFactory` on `Module`.

pub trait GLWEPreparedToBackendRef<B: Backend> {
    fn to_backend_ref(&self) -> GLWEPreparedBackendRef<'_, B>;
}

impl<B: Backend> GLWEPreparedToBackendRef<B> for GLWEPrepared<B::OwnedBuf, B> {
    fn to_backend_ref(&self) -> GLWEPreparedBackendRef<'_, B> {
        GLWEPrepared {
            noise: crate::layouts::LWEInfos::noise(&self),
            data: self.data.to_backend_ref(),
            base2k: self.base2k,
            k: self.k,
        }
    }
}

pub trait GLWEPreparedToBackendMut<B: Backend> {
    /// Backend hook for recording or propagating component noise metadata.
    fn set_noise(&mut self, metadata: Option<crate::ComponentNoise>);

    /// Borrows coefficients mutably and clears the owner's component noise metadata.
    fn to_backend_mut(&mut self) -> GLWEPreparedBackendMut<'_, B>;
}

impl<B: Backend> GLWEPreparedToBackendMut<B> for GLWEPrepared<B::OwnedBuf, B> {
    fn set_noise(&mut self, metadata: Option<crate::ComponentNoise>) {
        self.noise = crate::layouts::checked_noise(metadata, crate::layouts::GLWEInfos::rank(self).as_usize() + 1);
    }

    fn to_backend_mut(&mut self) -> GLWEPreparedBackendMut<'_, B> {
        self.noise = None;
        GLWEPrepared {
            noise: None,
            data: self.data.to_backend_mut(),
            base2k: self.base2k,
            k: self.k,
        }
    }
}
