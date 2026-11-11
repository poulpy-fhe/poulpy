use std::fmt;

use poulpy_core::layouts::{
    Base2K, Degree, Dnum, GGLWEInfos, GLWEInfos, GetGaloisElement, LWEInfos, Rank, SetGaloisElement, TorusPrecision,
    compressed::{
        GGLWECompressedBackendMut, GGLWECompressedBackendRef, GGLWECompressedSeed, GGLWECompressedSeedMut,
        GGLWECompressedToBackendMut, GGLWECompressedToBackendRef,
    },
};
use poulpy_hal::layouts::{
    Backend, Data, HostDataMut, HostDataRef, ReaderFrom, Ring, Standard, WriterTo, ZnxWord, is_galois_element,
};

use crate::layouts::MHEGGLWEPatCompressed;

pub type MHEGLWEAutomorphismKeyShareOwned<BE> = MHEGLWEAutomorphismKeyShare<<BE as Backend>::OwnedBuf, <BE as Backend>::ZnxWord>;

/// One party's share of a collective GLWE automorphism key: a
/// [`MHEGGLWEPatCompressed`] with the Galois element `p`.
///
/// Serializes as core's `GLWEAutomorphismKeyCompressed`.
#[derive(PartialEq, Eq, Clone)]
pub struct MHEGLWEAutomorphismKeyShare<D: Data, W: ZnxWord> {
    pub(crate) key: MHEGGLWEPatCompressed<D, W>,
    pub(crate) p: i64,
}

impl<D: Data, W: ZnxWord> GetGaloisElement for MHEGLWEAutomorphismKeyShare<D, W> {
    fn p(&self) -> i64 {
        self.p
    }
}

impl<D: Data, W: ZnxWord> SetGaloisElement for MHEGLWEAutomorphismKeyShare<D, W> {
    fn set_p(&mut self, p: i64) {
        self.p = p
    }
}

impl<D: Data, W: ZnxWord> LWEInfos for MHEGLWEAutomorphismKeyShare<D, W> {
    fn noise(&self) -> Option<poulpy_core::ComponentNoise> {
        self.key.noise()
    }

    fn n(&self) -> Degree {
        self.key.n()
    }

    fn k(&self) -> TorusPrecision {
        self.key.k()
    }

    fn base2k(&self) -> Base2K {
        self.key.base2k()
    }

    fn max_size(&self) -> usize {
        self.key.max_size()
    }
}

impl<D: Data, W: ZnxWord> GLWEInfos for MHEGLWEAutomorphismKeyShare<D, W> {
    fn rank(&self) -> Rank {
        self.key.rank()
    }
}

impl<D: Data, W: ZnxWord> GGLWEInfos for MHEGLWEAutomorphismKeyShare<D, W> {
    fn k_aux(&self) -> TorusPrecision {
        self.key.k_aux()
    }

    fn dnum(&self) -> Dnum {
        self.key.dnum()
    }

    fn k_digit(&self) -> TorusPrecision {
        self.key.k_digit()
    }

    fn k_offset(&self) -> TorusPrecision {
        self.key.k_offset()
    }

    fn rank_in(&self) -> Rank {
        self.key.rank_in()
    }

    fn rank_out(&self) -> Rank {
        self.key.rank_out()
    }
}

impl<D: Data, W: ZnxWord> GGLWECompressedSeed for MHEGLWEAutomorphismKeyShare<D, W> {
    fn seed(&self) -> &Vec<[u8; 32]> {
        self.key.seed()
    }
}

impl<D: Data, W: ZnxWord> GGLWECompressedSeedMut for MHEGLWEAutomorphismKeyShare<D, W> {
    fn seed_mut(&mut self) -> &mut Vec<[u8; 32]> {
        self.key.seed_mut()
    }
}

impl<BE: Backend, D: Data> GGLWECompressedToBackendRef<BE> for MHEGLWEAutomorphismKeyShare<D, BE::ZnxWord>
where
    MHEGGLWEPatCompressed<D, BE::ZnxWord>: GGLWECompressedToBackendRef<BE>,
{
    fn to_backend_ref(&self) -> GGLWECompressedBackendRef<'_, BE> {
        self.key.to_backend_ref()
    }
}

impl<BE: Backend, D: Data> GGLWECompressedToBackendMut<BE> for MHEGLWEAutomorphismKeyShare<D, BE::ZnxWord>
where
    MHEGGLWEPatCompressed<D, BE::ZnxWord>: GGLWECompressedToBackendMut<BE>,
{
    fn set_noise(&mut self, metadata: Option<poulpy_core::ComponentNoise>) {
        GGLWECompressedToBackendMut::<BE>::set_noise(&mut self.key, metadata);
    }

    fn to_backend_mut(&mut self) -> GGLWECompressedBackendMut<'_, BE> {
        self.key.to_backend_mut()
    }
}

impl<D: HostDataRef, W: ZnxWord> fmt::Debug for MHEGLWEAutomorphismKeyShare<D, W> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{self}")
    }
}

impl<D: HostDataRef, W: ZnxWord> fmt::Display for MHEGLWEAutomorphismKeyShare<D, W> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "(MHEGLWEAutomorphismKeyShare: p={}) {}", self.p, self.key)
    }
}

impl<D: HostDataMut, W: ZnxWord> ReaderFrom for MHEGLWEAutomorphismKeyShare<D, W> {
    /// Reads the format of [`WriterTo::write_to`]: a stream whose Galois element is not one of the standard ring at the
    /// stream's degree is [`std::io::ErrorKind::InvalidData`], the Galois element left unchanged and the noise metadata
    /// cleared.
    fn read_from<R: std::io::Read>(&mut self, reader: &mut R) -> std::io::Result<()> {
        let mut p = [0u8; 8];
        reader.read_exact(&mut p)?;
        let p = i64::from_le_bytes(p);
        self.key.read_from(reader)?;
        if !is_galois_element(p, galois_order(self.key.n())) {
            self.key.inner.clear_noise();
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "the Galois element is not odd and below the ring's cyclotomic order",
            ));
        }
        self.p = p;
        Ok(())
    }
}

impl<D: HostDataRef, W: ZnxWord> WriterTo for MHEGLWEAutomorphismKeyShare<D, W> {
    /// The Galois element `p` (`i64`), then the key share. A `p` even or not below the standard ring's cyclotomic order
    /// is [`std::io::ErrorKind::InvalidInput`], writing nothing.
    fn write_to<Wr: std::io::Write>(&self, writer: &mut Wr) -> std::io::Result<()> {
        if !is_galois_element(self.p, galois_order(self.key.n())) {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "the Galois element is not odd and below the ring's cyclotomic order",
            ));
        }
        writer.write_all(&self.p.to_le_bytes())?;
        self.key.write_to(writer)
    }
}

/// Cyclotomic order of the standard ring of degree `n`, the bound of a share's Galois element.
fn galois_order(n: Degree) -> i64 {
    Standard::CYCLOTOMIC_ORDER_FACTOR * i64::from(n.0)
}

#[deprecated(note = "renamed MHEGLWEAutomorphismKeyShare")]
pub type GLWEAutomorphismKeyShare<D, W> = MHEGLWEAutomorphismKeyShare<D, W>;

#[deprecated(note = "renamed MHEGLWEAutomorphismKeyShareOwned")]
pub type GLWEAutomorphismKeyShareOwned<BE> = MHEGLWEAutomorphismKeyShareOwned<BE>;
