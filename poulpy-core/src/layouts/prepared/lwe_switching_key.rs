use poulpy_hal::layouts::{Backend, Data, Ring, Standard};

use crate::layouts::{
    Base2K, Degree, Dnum, GGLWEInfos, GGLWEPrepared, GGLWEPreparedBackendRef, GGLWEPreparedToBackendMut,
    GGLWEPreparedToBackendRef, GLWEInfos, GLWESwitchingKeyDegreesMut, LWEInfos, Rank, TorusPrecision,
    prepared::{GLWESwitchingKeyPrepared, GLWESwitchingKeyPreparedToBackendMut, GLWESwitchingKeyPreparedToBackendRef},
};

/// DFT-domain (prepared) variant of an LWE switching key.
///
/// A newtype wrapper around [`GLWESwitchingKeyPrepared`] for LWE key-switching.
/// Tied to a specific backend via `B: Backend`.
#[derive(PartialEq)]
pub struct LWESwitchingKeyPrepared<D: Data, B: Backend, Rg: Ring = Standard>(pub(crate) GLWESwitchingKeyPrepared<D, B, Rg>);

impl<D: Data, B: Backend, Rg: Ring> LWEInfos for LWESwitchingKeyPrepared<D, B, Rg> {
    fn noise(&self) -> Option<crate::ComponentNoise> {
        self.0.noise()
    }

    fn base2k(&self) -> Base2K {
        self.0.base2k()
    }

    fn n(&self) -> Degree {
        self.0.n()
    }

    fn max_size(&self) -> usize {
        self.0.max_size()
    }

    fn k(&self) -> TorusPrecision {
        self.0.k()
    }
}
impl<D: Data, B: Backend, Rg: Ring> GLWEInfos for LWESwitchingKeyPrepared<D, B, Rg> {
    fn rank(&self) -> Rank {
        self.rank_out()
    }
}

impl<D: Data, B: Backend, Rg: Ring> GGLWEInfos for LWESwitchingKeyPrepared<D, B, Rg> {
    fn k_aux(&self) -> TorusPrecision {
        self.0.k_aux()
    }

    fn k_digit(&self) -> TorusPrecision {
        self.0.k_digit()
    }

    fn k_offset(&self) -> TorusPrecision {
        self.0.k_offset()
    }

    fn rank_in(&self) -> Rank {
        self.0.rank_in()
    }

    fn rank_out(&self) -> Rank {
        self.0.rank_out()
    }

    fn dnum(&self) -> Dnum {
        self.0.dnum()
    }

    fn stride(&self) -> usize {
        self.0.stride()
    }
}

impl<D: Data, B: Backend, Rg: Ring> GGLWEPreparedToBackendMut<B, Rg> for LWESwitchingKeyPrepared<D, B, Rg>
where
    GGLWEPrepared<D, B, Rg>: GGLWEPreparedToBackendMut<B, Rg>,
{
    fn set_noise(&mut self, metadata: Option<crate::ComponentNoise>) {
        <_ as GGLWEPreparedToBackendMut<B, Rg>>::set_noise(&mut self.0.key, metadata);
    }

    fn to_backend_mut(&mut self) -> crate::layouts::GGLWEPreparedBackendMut<'_, B, Rg> {
        self.0.key.to_backend_mut()
    }
}

impl<D: Data, B: Backend, Rg: Ring> GLWESwitchingKeyDegreesMut for LWESwitchingKeyPrepared<D, B, Rg> {
    fn input_degree(&mut self) -> &mut Degree {
        &mut self.0.input_degree
    }

    fn output_degree(&mut self) -> &mut Degree {
        &mut self.0.output_degree
    }
}

pub type LWESwitchingKeyPreparedBackendRef<'a, B, Rg = Standard> = LWESwitchingKeyPrepared<<B as Backend>::BufRef<'a>, B, Rg>;
pub type LWESwitchingKeyPreparedBackendMut<'a, B, Rg = Standard> = LWESwitchingKeyPrepared<<B as Backend>::BufMut<'a>, B, Rg>;

pub trait LWESwitchingKeyPreparedToBackendRef<B: Backend, Rg: Ring = Standard> {
    fn to_backend_ref(&self) -> LWESwitchingKeyPreparedBackendRef<'_, B, Rg>;
}

impl<D: Data, B: Backend, Rg: Ring> LWESwitchingKeyPreparedToBackendRef<B, Rg> for LWESwitchingKeyPrepared<D, B, Rg>
where
    GLWESwitchingKeyPrepared<D, B, Rg>: GLWESwitchingKeyPreparedToBackendRef<B, Rg>,
{
    fn to_backend_ref(&self) -> LWESwitchingKeyPreparedBackendRef<'_, B, Rg> {
        LWESwitchingKeyPrepared(self.0.to_backend_ref())
    }
}

impl<D: Data, B: Backend, Rg: Ring> GGLWEPreparedToBackendRef<B, Rg> for LWESwitchingKeyPrepared<D, B, Rg>
where
    GGLWEPrepared<D, B, Rg>: GGLWEPreparedToBackendRef<B, Rg>,
{
    fn to_backend_ref(&self) -> GGLWEPreparedBackendRef<'_, B, Rg> {
        self.0.key.to_backend_ref()
    }
}

pub trait LWESwitchingKeyPreparedToBackendMut<B: Backend, Rg: Ring = Standard> {
    /// Borrows coefficients mutably and clears the owner's component noise metadata.
    fn to_backend_mut(&mut self) -> LWESwitchingKeyPreparedBackendMut<'_, B, Rg>;
}

impl<D: Data, B: Backend, Rg: Ring> LWESwitchingKeyPreparedToBackendMut<B, Rg> for LWESwitchingKeyPrepared<D, B, Rg>
where
    GLWESwitchingKeyPrepared<D, B, Rg>: GLWESwitchingKeyPreparedToBackendMut<B, Rg>,
{
    fn to_backend_mut(&mut self) -> LWESwitchingKeyPreparedBackendMut<'_, B, Rg> {
        LWESwitchingKeyPrepared(self.0.to_backend_mut())
    }
}
