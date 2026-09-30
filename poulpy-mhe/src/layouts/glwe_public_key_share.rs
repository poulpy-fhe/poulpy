use std::fmt;

use poulpy_core::{
    Distribution, GetDistribution, GetDistributionMut,
    layouts::{Base2K, Degree, GLWEInfos, LWEInfos, Rank, TorusPrecision},
};
use poulpy_hal::layouts::{Backend, Data, HostDataMut, HostDataRef, ReaderFrom, WriterTo, ZnxWord};

use crate::layouts::GLWEPatCompressed;

pub type GLWEPublicKeyShareOwned<BE> = GLWEPublicKeyShare<<BE as Backend>::OwnedBuf, <BE as Backend>::ZnxWord>;

/// One party's share of the collective public key: a [`GLWEPatCompressed`]
/// per key entry, `rank` entries in all, each under its own seed, and the
/// distribution of the secret it was generated with.
///
/// Serializes as the distribution, then its entries in order.
#[derive(Clone)]
pub struct GLWEPublicKeyShare<D: Data, W: ZnxWord> {
    pub(crate) entries: Vec<GLWEPatCompressed<D, W>>,
    pub(crate) dist: Distribution,
}

impl<D: Data, W: ZnxWord> GLWEPublicKeyShare<D, W> {
    /// The key entries, one per rank.
    pub fn entries(&self) -> &[GLWEPatCompressed<D, W>] {
        &self.entries
    }

    pub fn entries_mut(&mut self) -> &mut [GLWEPatCompressed<D, W>] {
        &mut self.entries
    }
}

impl<D: Data, W: ZnxWord> PartialEq for GLWEPublicKeyShare<D, W>
where
    GLWEPatCompressed<D, W>: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.entries == other.entries && self.dist == other.dist
    }
}

impl<D: Data, W: ZnxWord> Eq for GLWEPublicKeyShare<D, W> where GLWEPatCompressed<D, W>: Eq {}

impl<D: Data, W: ZnxWord> LWEInfos for GLWEPublicKeyShare<D, W> {
    fn n(&self) -> Degree {
        self.entries[0].n()
    }

    fn k(&self) -> TorusPrecision {
        self.entries[0].k()
    }

    fn base2k(&self) -> Base2K {
        self.entries[0].base2k()
    }

    fn max_size(&self) -> usize {
        self.entries[0].max_size()
    }
}

impl<D: Data, W: ZnxWord> GLWEInfos for GLWEPublicKeyShare<D, W> {
    fn rank(&self) -> Rank {
        self.entries[0].rank()
    }
}

impl<D: Data, W: ZnxWord> GetDistribution for GLWEPublicKeyShare<D, W> {
    fn dist(&self) -> &Distribution {
        &self.dist
    }
}

impl<D: Data, W: ZnxWord> GetDistributionMut for GLWEPublicKeyShare<D, W> {
    fn dist_mut(&mut self) -> &mut Distribution {
        &mut self.dist
    }
}

impl<D: HostDataRef, W: ZnxWord> fmt::Debug for GLWEPublicKeyShare<D, W> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{self}")
    }
}

impl<D: HostDataRef, W: ZnxWord> fmt::Display for GLWEPublicKeyShare<D, W> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "GLWEPublicKeyShare: dist={:?}", self.dist)?;
        for entry in &self.entries {
            writeln!(f, "{entry}")?;
        }
        Ok(())
    }
}

impl<D: HostDataMut, W: ZnxWord> ReaderFrom for GLWEPublicKeyShare<D, W> {
    fn read_from<R: std::io::Read>(&mut self, reader: &mut R) -> std::io::Result<()> {
        self.dist = Distribution::read_from(reader)?;
        self.entries.iter_mut().try_for_each(|entry| entry.read_from(reader))?;
        // An entry's size does not depend on its rank, so a share of another rank reads without error.
        if self.entries.iter().any(|entry| entry.rank().as_usize() != self.entries.len()) {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "invalid share: entry rank differs from the share's",
            ));
        }
        Ok(())
    }
}

impl<D: HostDataRef, W: ZnxWord> WriterTo for GLWEPublicKeyShare<D, W> {
    fn write_to<Wr: std::io::Write>(&self, writer: &mut Wr) -> std::io::Result<()> {
        self.dist.write_to(writer)?;
        self.entries.iter().try_for_each(|entry| entry.write_to(writer))
    }
}
