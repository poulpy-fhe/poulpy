use poulpy_hal::layouts::VecZnxDftToBackendMut;
use poulpy_hal::layouts::VecZnxDftToBackendRef;
use poulpy_hal::{
    api::{VecZnxDftAlloc, VecZnxDftApply, VecZnxDftBytesOf},
    layouts::{Backend, Data, Module, ScratchArena, VecZnxDft},
};

use crate::layouts::{GLWELayout, operand_degree};
use crate::{
    GLWEBytesOf, GLWENormalize, ScratchArenaTakeCore,
    layouts::{Base2K, Degree, GLWEInfos, GLWEToBackendRef, GetDegree, LWEInfos, Rank, TorusPrecision},
};

/// DFT-domain (prepared) variant of [`GLWE`](crate::layouts::GLWE).
///
/// Stores polynomials in the frequency domain of the backend's DFT/NTT
/// transform, enabling O(N log N) polynomial multiplication.
/// Tied to a specific backend via `B: Backend`.
#[derive(PartialEq)]
pub struct GLWEPrepared<D: Data, B: Backend> {
    pub(crate) metadata: Option<crate::EncryptionMetadata>,
    pub(crate) data: VecZnxDft<D, B::DftWord, B>,
    pub(crate) k: TorusPrecision,
    pub(crate) base2k: Base2K,
}

pub type GLWEPreparedBackendRef<'a, B> = GLWEPrepared<<B as Backend>::BufRef<'a>, B>;
pub type GLWEPreparedBackendMut<'a, B> = GLWEPrepared<<B as Backend>::BufMut<'a>, B>;

impl<D: Data, B: Backend> LWEInfos for GLWEPrepared<D, B> {
    fn encryption_metadata(&self) -> Option<crate::EncryptionMetadata> {
        self.metadata
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
    Self: GetDegree + VecZnxDftAlloc<B> + VecZnxDftBytesOf + VecZnxDftApply<B> + GLWENormalize<B> + GLWEBytesOf<B>,
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
            metadata: None,
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
        A: GLWEInfos,
    {
        B::scratch_aligned(self.glwe_bytes_of_from_infos(infos)) + self.glwe_normalize_tmp_bytes()
    }

    fn glwe_prepare<R, O>(&self, res: &mut R, other: &O, scratch: &mut ScratchArena<'_, B>)
    where
        R: GLWEPreparedToBackendMut<B>,
        O: GLWEToBackendRef<B> + GLWEInfos,
    {
        res.set_encryption_metadata(other.to_backend_ref().encryption_metadata());
        let (mut other_tmp, mut scratch) = scratch.borrow().take_glwe_scratch(other);
        let other = if other.is_canonical() {
            other.to_backend_ref()
        } else {
            self.glwe_normalize(&mut other_tmp, other, &mut scratch.borrow());
            other_tmp.to_backend_ref()
        };
        let mut res = res.to_backend_mut();

        operand_degree(self.ring_degree().as_usize(), &[res.n(), other.n()]);
        assert_eq!(res.size(), other.size());
        assert_eq!(res.k(), other.k());
        assert_eq!(res.base2k(), other.base2k());

        for i in 0..(res.rank() + 1).into() {
            self.vec_znx_dft_apply(1, 0, &mut res.data, i, &other.data, i);
        }
    }
}

impl<B: Backend> GLWEPreparedFactory<B> for Module<B> where
    Self: VecZnxDftAlloc<B> + VecZnxDftBytesOf + VecZnxDftApply<B> + GLWENormalize<B>
{
}

// module-only API: allocation/size helpers are provided by `GLWEPreparedFactory` on `Module`.

// module-only API: preparation is provided by `GLWEPreparedFactory` on `Module`.

pub trait GLWEPreparedToBackendRef<B: Backend> {
    fn to_backend_ref(&self) -> GLWEPreparedBackendRef<'_, B>;
}

impl<B: Backend> GLWEPreparedToBackendRef<B> for GLWEPrepared<B::OwnedBuf, B> {
    fn to_backend_ref(&self) -> GLWEPreparedBackendRef<'_, B> {
        GLWEPrepared {
            metadata: crate::layouts::LWEInfos::encryption_metadata(&self),
            data: self.data.to_backend_ref(),
            base2k: self.base2k,
            k: self.k,
        }
    }
}

pub trait GLWEPreparedToBackendMut<B: Backend> {
    /// Backend hook for recording or propagating derived encryption provenance.
    fn set_encryption_metadata(&mut self, metadata: Option<crate::EncryptionMetadata>);

    /// Borrows coefficients and copies the current layout and provenance metadata.
    /// Metadata changed on the returned view is local to that view. Operations
    /// that update the owner must call its `set_encryption_metadata` hook.
    fn to_backend_mut(&mut self) -> GLWEPreparedBackendMut<'_, B>;
}

impl<B: Backend> GLWEPreparedToBackendMut<B> for GLWEPrepared<B::OwnedBuf, B> {
    fn set_encryption_metadata(&mut self, metadata: Option<crate::EncryptionMetadata>) {
        self.metadata = metadata;
    }

    fn to_backend_mut(&mut self) -> GLWEPreparedBackendMut<'_, B> {
        GLWEPrepared {
            metadata: crate::layouts::LWEInfos::encryption_metadata(&self),
            data: self.data.to_backend_mut(),
            base2k: self.base2k,
            k: self.k,
        }
    }
}
