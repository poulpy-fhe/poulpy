use poulpy_hal::layouts::{Backend, Data, Ring, Standard};

use crate::layouts::{
    Base2K, Degree, Dnum, GGLWEInfos, GGLWEPrepared, GGLWEPreparedBackendRef, GGLWEPreparedToBackendMut,
    GGLWEPreparedToBackendRef, GLWEInfos, GLWESwitchingKeyDegrees, GLWESwitchingKeyDegreesMut, LWEInfos, Rank, TorusPrecision,
    prepared::{GLWESwitchingKeyPrepared, GLWESwitchingKeyPreparedToBackendMut, GLWESwitchingKeyPreparedToBackendRef},
};

/// DFT-domain (prepared) variant of a GLWE-to-LWE conversion key.
///
/// A newtype wrapper around [`GLWESwitchingKeyPrepared`] for converting
/// GLWE to LWE. Tied to a specific backend via `B: Backend`.
#[derive(PartialEq)]
pub struct GLWEToLWEKeyPrepared<D: Data, B: Backend, Rg: Ring = Standard>(pub(crate) GLWESwitchingKeyPrepared<D, B, Rg>);

impl<D: Data, B: Backend, Rg: Ring> LWEInfos for GLWEToLWEKeyPrepared<D, B, Rg> {
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

impl<D: Data, B: Backend, Rg: Ring> GLWEInfos for GLWEToLWEKeyPrepared<D, B, Rg> {
    fn rank(&self) -> Rank {
        self.rank_out()
    }
}

impl<D: Data, B: Backend, Rg: Ring> GGLWEInfos for GLWEToLWEKeyPrepared<D, B, Rg> {
    fn k_aux(&self) -> TorusPrecision {
        self.0.k_aux()
    }

    fn rank_in(&self) -> Rank {
        self.0.rank_in()
    }

    fn k_digit(&self) -> TorusPrecision {
        self.0.k_digit()
    }

    fn k_offset(&self) -> TorusPrecision {
        self.0.k_offset()
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

impl<D: Data, B: Backend, Rg: Ring> GGLWEPreparedToBackendMut<B, Rg> for GLWEToLWEKeyPrepared<D, B, Rg>
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

impl<D: Data, B: Backend, Rg: Ring> GLWESwitchingKeyDegreesMut for GLWEToLWEKeyPrepared<D, B, Rg> {
    fn input_degree(&mut self) -> &mut Degree {
        &mut self.0.input_degree
    }

    fn output_degree(&mut self) -> &mut Degree {
        &mut self.0.output_degree
    }
}

impl<D: Data, B: Backend, Rg: Ring> GLWESwitchingKeyDegrees for GLWEToLWEKeyPrepared<D, B, Rg> {
    fn input_degree(&self) -> &Degree {
        &self.0.input_degree
    }

    fn output_degree(&self) -> &Degree {
        &self.0.output_degree
    }
}

pub type GLWEToLWEKeyPreparedBackendRef<'a, B, Rg = Standard> = GLWEToLWEKeyPrepared<<B as Backend>::BufRef<'a>, B, Rg>;
pub type GLWEToLWEKeyPreparedBackendMut<'a, B, Rg = Standard> = GLWEToLWEKeyPrepared<<B as Backend>::BufMut<'a>, B, Rg>;

pub trait GLWEToLWEKeyPreparedToBackendRef<B: Backend, Rg: Ring = Standard> {
    fn to_backend_ref(&self) -> GLWEToLWEKeyPreparedBackendRef<'_, B, Rg>;
}

impl<D: Data, B: Backend, Rg: Ring> GLWEToLWEKeyPreparedToBackendRef<B, Rg> for GLWEToLWEKeyPrepared<D, B, Rg>
where
    GLWESwitchingKeyPrepared<D, B, Rg>: GLWESwitchingKeyPreparedToBackendRef<B, Rg>,
{
    fn to_backend_ref(&self) -> GLWEToLWEKeyPreparedBackendRef<'_, B, Rg> {
        GLWEToLWEKeyPrepared(self.0.to_backend_ref())
    }
}

impl<D: Data, B: Backend, Rg: Ring> GGLWEPreparedToBackendRef<B, Rg> for GLWEToLWEKeyPrepared<D, B, Rg>
where
    GGLWEPrepared<D, B, Rg>: GGLWEPreparedToBackendRef<B, Rg>,
{
    fn to_backend_ref(&self) -> GGLWEPreparedBackendRef<'_, B, Rg> {
        self.0.key.to_backend_ref()
    }
}

pub trait GLWEToLWEKeyPreparedToBackendMut<B: Backend, Rg: Ring = Standard> {
    /// Borrows coefficients mutably and clears the owner's component noise metadata.
    fn to_backend_mut(&mut self) -> GLWEToLWEKeyPreparedBackendMut<'_, B, Rg>;
}

impl<D: Data, B: Backend, Rg: Ring> GLWEToLWEKeyPreparedToBackendMut<B, Rg> for GLWEToLWEKeyPrepared<D, B, Rg>
where
    GLWESwitchingKeyPrepared<D, B, Rg>: GLWESwitchingKeyPreparedToBackendMut<B, Rg>,
{
    fn to_backend_mut(&mut self) -> GLWEToLWEKeyPreparedBackendMut<'_, B, Rg> {
        GLWEToLWEKeyPrepared(self.0.to_backend_mut())
    }
}
