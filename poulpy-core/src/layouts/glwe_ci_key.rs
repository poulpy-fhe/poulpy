use std::{fmt, marker::PhantomData};

use poulpy_hal::layouts::{Backend, Data, HostDataMut, HostDataRef, ReaderFrom, Ring, Standard, WriterTo, ZnxWord};

use crate::layouts::{
    Base2K, Degree, Dnum, GGLWEBackendMut, GGLWEBackendRef, GGLWEInfos, GGLWEToBackendMut, GGLWEToBackendRef, GLWEInfos,
    GLWESwitchingKey, GLWESwitchingKeyDegrees, GLWESwitchingKeyDegreesMut, LWEInfos, Rank, TorusPrecision,
};

/// Direction of a [`GLWECIEmbedKey`]: from the embedded conjugate-invariant secret to a standard secret.
#[derive(PartialEq, Eq, Clone, Copy, Debug)]
pub struct CIEmbed;

/// Direction of a [`GLWECITraceKey`]: from a standard secret to the embedded conjugate-invariant secret.
#[derive(PartialEq, Eq, Clone, Copy, Debug)]
pub struct CITrace;

/// A [`GLWESwitchingKey`] of degree `2N` between the embedding of a conjugate-invariant secret of
/// degree `N` and a standard secret, typed by the map it serves.
#[derive(PartialEq, Eq, Clone)]
pub struct GLWECIKey<D: Data, W: ZnxWord, M, Rg: Ring = Standard>(
    pub(crate) GLWESwitchingKey<D, W, Rg>,
    pub(crate) PhantomData<M>,
);

/// Switches an embedded conjugate-invariant GLWE to the standard secret.
pub type GLWECIEmbedKey<D, W, Rg = Standard> = GLWECIKey<D, W, CIEmbed, Rg>;

/// Switches a standard GLWE to the embedded conjugate-invariant secret, ahead of its trace.
pub type GLWECITraceKey<D, W, Rg = Standard> = GLWECIKey<D, W, CITrace, Rg>;

impl<D: Data, W: ZnxWord, M, Rg: Ring> LWEInfos for GLWECIKey<D, W, M, Rg> {
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

impl<D: Data, W: ZnxWord, M, Rg: Ring> GLWEInfos for GLWECIKey<D, W, M, Rg> {
    fn rank(&self) -> Rank {
        self.rank_out()
    }
}

impl<D: Data, W: ZnxWord, M, Rg: Ring> GGLWEInfos for GLWECIKey<D, W, M, Rg> {
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
}

impl<D: HostDataRef, W: ZnxWord, M, Rg: Ring> fmt::Debug for GLWECIKey<D, W, M, Rg> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{self}")
    }
}

impl<D: HostDataRef, W: ZnxWord, M, Rg: Ring> fmt::Display for GLWECIKey<D, W, M, Rg> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "(GLWECIKey) {}", self.0)
    }
}

impl<D: HostDataMut, W: ZnxWord, M, Rg: Ring> ReaderFrom for GLWECIKey<D, W, M, Rg> {
    fn read_from<R: std::io::Read>(&mut self, reader: &mut R) -> std::io::Result<()> {
        self.0.read_from(reader)
    }
}

impl<D: HostDataRef, W: ZnxWord, M, Rg: Ring> WriterTo for GLWECIKey<D, W, M, Rg> {
    fn write_to<Wr: std::io::Write>(&self, writer: &mut Wr) -> std::io::Result<()> {
        self.0.write_to(writer)
    }
}

impl<BE: Backend, D: Data, M, Rg: Ring> GGLWEToBackendRef<BE, Rg> for GLWECIKey<D, BE::ZnxWord, M, Rg>
where
    GLWESwitchingKey<D, BE::ZnxWord, Rg>: GGLWEToBackendRef<BE, Rg>,
{
    fn to_backend_ref(&self) -> GGLWEBackendRef<'_, BE, Rg> {
        self.0.to_backend_ref()
    }
}

impl<BE: Backend, D: Data, M, Rg: Ring> GGLWEToBackendMut<BE, Rg> for GLWECIKey<D, BE::ZnxWord, M, Rg>
where
    GLWESwitchingKey<D, BE::ZnxWord, Rg>: GGLWEToBackendMut<BE, Rg>,
{
    fn set_noise(&mut self, metadata: Option<crate::ComponentNoise>) {
        <_ as GGLWEToBackendMut<BE, Rg>>::set_noise(&mut self.0, metadata);
    }

    fn to_backend_mut(&mut self) -> GGLWEBackendMut<'_, BE, Rg> {
        self.0.to_backend_mut()
    }
}

impl<D: Data, W: ZnxWord, M, Rg: Ring> GLWESwitchingKeyDegreesMut for GLWECIKey<D, W, M, Rg> {
    fn input_degree(&mut self) -> &mut Degree {
        &mut self.0.input_degree
    }

    fn output_degree(&mut self) -> &mut Degree {
        &mut self.0.output_degree
    }
}

impl<D: Data, W: ZnxWord, M, Rg: Ring> GLWESwitchingKeyDegrees for GLWECIKey<D, W, M, Rg> {
    fn input_degree(&self) -> &Degree {
        &self.0.input_degree
    }

    fn output_degree(&self) -> &Degree {
        &self.0.output_degree
    }
}
