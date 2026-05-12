use poulpy_hal::{
    api::{VecZnxDftAlloc, VecZnxDftApply, VecZnxDftBytesOf},
    layouts::{
        Backend, Data, Module, VecZnxDft, VecZnxDftReborrowBackendRef, VecZnxDftToBackendMut, VecZnxDftToBackendRef,
        vec_znx_dft_backend_mut_from_mut,
    },
};

use crate::layouts::{Base2K, Degree, GLWEInfos, GLWEToBackendRef, GetDegree, LWEInfos, Rank, TorusPrecision};

/// DFT-domain (prepared) variant of [`GLWE`].
///
/// Stores polynomials in the frequency domain of the backend's DFT/NTT
/// transform, enabling O(N log N) polynomial multiplication.
/// Tied to a specific backend via `B: Backend`.
#[derive(PartialEq, Eq)]
pub struct GLWEPrepared<D: Data, B: Backend> {
    pub(crate) data: VecZnxDft<D, B>,
    pub(crate) base2k: Base2K,
}

pub type GLWEPreparedBackendRef<'a, B> = GLWEPrepared<<B as Backend>::BufRef<'a>, B>;
pub type GLWEPreparedBackendMut<'a, B> = GLWEPrepared<<B as Backend>::BufMut<'a>, B>;

impl<D: Data, B: Backend> LWEInfos for GLWEPrepared<D, B> {
    fn base2k(&self) -> Base2K {
        self.base2k
    }

    fn size(&self) -> usize {
        self.data.size()
    }

    fn n(&self) -> Degree {
        Degree(self.data.n() as u32)
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
    Self: GetDegree + VecZnxDftAlloc<B> + VecZnxDftBytesOf + VecZnxDftApply<B>,
{
    /// Allocates a new prepared GLWE with the given parameters.
    fn glwe_prepared_alloc(&self, base2k: Base2K, k: TorusPrecision, rank: Rank) -> GLWEPrepared<B::OwnedBuf, B> {
        GLWEPrepared {
            data: self.vec_znx_dft_alloc((rank + 1).into(), k.0.div_ceil(base2k.0) as usize),
            base2k,
        }
    }

    fn glwe_prepared_alloc_from_infos<A>(&self, infos: &A) -> GLWEPrepared<B::OwnedBuf, B>
    where
        A: GLWEInfos,
    {
        self.glwe_prepared_alloc(infos.base2k(), infos.max_k(), infos.rank())
    }

    fn glwe_prepared_bytes_of(&self, base2k: Base2K, k: TorusPrecision, rank: Rank) -> usize {
        self.bytes_of_vec_znx_dft((rank + 1).into(), k.0.div_ceil(base2k.0) as usize)
    }

    fn glwe_prepared_bytes_of_from_infos<A>(&self, infos: &A) -> usize
    where
        A: GLWEInfos,
    {
        self.glwe_prepared_bytes_of(infos.base2k(), infos.max_k(), infos.rank())
    }

    fn glwe_prepare<R, O>(&self, res: &mut R, other: &O)
    where
        R: GLWEPreparedToBackendMut<B>,
        O: GLWEToBackendRef<B> + GLWEInfos,
    {
        let mut res = res.to_backend_mut();
        let other = other.to_backend_ref();

        assert_eq!(res.n(), self.ring_degree());
        assert_eq!(other.n(), self.ring_degree());
        assert_eq!(res.size(), other.size());
        assert_eq!(res.max_k(), other.max_k());
        assert_eq!(res.base2k(), other.base2k());

        for i in 0..(res.rank() + 1).into() {
            self.vec_znx_dft_apply(1, 0, &mut res.data, i, &other.data, i);
        }
    }
}

impl<B: Backend> GLWEPreparedFactory<B> for Module<B> where Self: VecZnxDftAlloc<B> + VecZnxDftBytesOf + VecZnxDftApply<B> {}

// module-only API: allocation/size helpers are provided by `GLWEPreparedFactory` on `Module`.

// module-only API: preparation is provided by `GLWEPreparedFactory` on `Module`.

pub trait GLWEPreparedToBackendRef<B: Backend> {
    fn to_backend_ref(&self) -> GLWEPreparedBackendRef<'_, B>;
}

impl<B: Backend> GLWEPreparedToBackendRef<B> for GLWEPrepared<B::OwnedBuf, B> {
    fn to_backend_ref(&self) -> GLWEPreparedBackendRef<'_, B> {
        GLWEPrepared {
            data: self.data.to_backend_ref(),
            base2k: self.base2k,
        }
    }
}

impl<'b, B: Backend + 'b> GLWEPreparedToBackendRef<B> for &GLWEPrepared<B::BufRef<'b>, B> {
    fn to_backend_ref(&self) -> GLWEPreparedBackendRef<'_, B> {
        GLWEPrepared {
            data: VecZnxDft::from_data_with_max_size(
                B::view_ref(&self.data.data),
                self.data.n(),
                self.data.cols(),
                self.data.size(),
                self.data.max_size(),
            ),
            base2k: self.base2k,
        }
    }
}

impl<'b, B: Backend + 'b> GLWEPreparedToBackendRef<B> for &mut GLWEPrepared<B::BufMut<'b>, B> {
    fn to_backend_ref(&self) -> GLWEPreparedBackendRef<'_, B> {
        GLWEPrepared {
            data: self.data.reborrow_backend_ref(),
            base2k: self.base2k,
        }
    }
}

pub trait GLWEPreparedToBackendMut<B: Backend> {
    fn to_backend_mut(&mut self) -> GLWEPreparedBackendMut<'_, B>;
}

impl<B: Backend> GLWEPreparedToBackendMut<B> for GLWEPrepared<B::OwnedBuf, B> {
    fn to_backend_mut(&mut self) -> GLWEPreparedBackendMut<'_, B> {
        GLWEPrepared {
            data: self.data.to_backend_mut(),
            base2k: self.base2k,
        }
    }
}

impl<'b, B: Backend + 'b> GLWEPreparedToBackendMut<B> for &mut GLWEPrepared<B::BufMut<'b>, B> {
    fn to_backend_mut(&mut self) -> GLWEPreparedBackendMut<'_, B> {
        GLWEPrepared {
            data: vec_znx_dft_backend_mut_from_mut::<B>(&mut self.data),
            base2k: self.base2k,
        }
    }
}
