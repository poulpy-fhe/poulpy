use std::fmt;

use poulpy_core::layouts::{
    Base2K, Degree, Dnum, Dsize, GGLWEInfos, GGLWELayout, GGSWInfos, GLWEInfos, LWEInfos, Rank, TorusPrecision,
};
use poulpy_hal::layouts::{Backend, Data, HostDataMut, HostDataRef, ReaderFrom, WriterTo, ZnxWord};

use crate::layouts::GGLWEPatCompressed;

/// The `GGLWELayout` of a `GGSWShare` part: `col0` at `rank_in = 1`,
/// `circ_u`/`circ_s` at `rank_in = rank`, all at `rank_out = infos.rank()`.
pub(crate) fn ggsw_share_part_layout<A: GGSWInfos>(infos: &A, rank_in: Rank, rank_out: Rank) -> GGLWELayout {
    GGLWELayout {
        n: infos.n(),
        base2k: infos.base2k(),
        dnum: infos.dnum(),
        k_aux: infos.k_aux(),
        rank_in,
        rank_out,
        dsize: infos.dsize(),
        stride: 1,
    }
}

pub type GGSWShareOwned<BE> = GGSWShare<<BE as Backend>::OwnedBuf, <BE as Backend>::ZnxWord>;

/// One party's share of a collective GGSW of rank `r`, over seeded GGLWE PATs.
///
/// `col0` transcribes column 0, a GGLWE of the message (`rank_in = 1`,
/// `rank_out = r`). For every column `j >= 1`, `circ_u[j - 1]` and
/// `circ_s[j - 1]` (`rank_in = rank_out = r`) hold the two halves of the
/// circular product over one `r x r` mask matrix per gadget row: entry `i` of
/// `circ_s` is over row `i`, drawn from its seed, and entry `l` of `circ_u` over
/// column `l`. Both carry the `circ_s` seeds.
///
/// Serializes as `col0`, then every `circ_u`, then every `circ_s`.
#[derive(PartialEq, Eq, Clone)]
pub struct GGSWShare<D: Data, W: ZnxWord> {
    pub(crate) col0: GGLWEPatCompressed<D, W>,
    pub(crate) circ_u: Vec<GGLWEPatCompressed<D, W>>,
    pub(crate) circ_s: Vec<GGLWEPatCompressed<D, W>>,
}

impl<D: Data, W: ZnxWord> GGSWShare<D, W> {
    /// Iterates over the `1 + 2 * r` parts in serialization order: `col0`,
    /// then `circ_u[0..r]`, then `circ_s[0..r]`, where `r = self.rank()`.
    /// The circular parts at index `j - 1` belong to GGSW column `j`.
    ///
    /// Backends can read each part's bodies through
    /// [`GGLWECompressedToBackendRef`](poulpy_core::layouts::GGLWECompressedToBackendRef)
    /// and its seeds through [`GGLWECompressedSeed`](poulpy_core::layouts::GGLWECompressedSeed).
    pub fn parts(&self) -> impl Iterator<Item = &GGLWEPatCompressed<D, W>> {
        std::iter::once(&self.col0).chain(&self.circ_u).chain(&self.circ_s)
    }

    /// Mutably iterates over the parts in the same order as [`Self::parts`].
    ///
    /// Backends can write each part's bodies through
    /// [`GGLWECompressedToBackendMut`](poulpy_core::layouts::GGLWECompressedToBackendMut)
    /// and its seeds through [`GGLWECompressedSeedMut`](poulpy_core::layouts::GGLWECompressedSeedMut).
    /// Update seeds on the part itself: backend views hold cloned seeds.
    /// Callers must preserve the parts' layouts and seed counts, and keep
    /// the seeds of each `circ_u[j]`/`circ_s[j]` pair equal.
    pub fn parts_mut(&mut self) -> impl Iterator<Item = &mut GGLWEPatCompressed<D, W>> {
        std::iter::once(&mut self.col0)
            .chain(&mut self.circ_u)
            .chain(&mut self.circ_s)
    }
}

impl<D: Data, W: ZnxWord> LWEInfos for GGSWShare<D, W> {
    fn n(&self) -> Degree {
        self.col0.n()
    }

    fn k(&self) -> TorusPrecision {
        self.col0.k()
    }

    fn base2k(&self) -> Base2K {
        self.col0.base2k()
    }

    fn max_size(&self) -> usize {
        self.col0.max_size()
    }
}

impl<D: Data, W: ZnxWord> GLWEInfos for GGSWShare<D, W> {
    fn rank(&self) -> Rank {
        self.col0.rank_out()
    }
}

impl<D: Data, W: ZnxWord> GGSWInfos for GGSWShare<D, W> {
    fn k_aux(&self) -> TorusPrecision {
        self.col0.k_aux()
    }

    fn dnum(&self) -> Dnum {
        self.col0.dnum()
    }

    fn dsize(&self) -> Dsize {
        self.col0.dsize()
    }
}

impl<D: HostDataRef, W: ZnxWord> fmt::Debug for GGSWShare<D, W> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{self}")
    }
}

impl<D: HostDataRef, W: ZnxWord> fmt::Display for GGSWShare<D, W> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "(GGSWShare: col0) {}", self.col0)?;
        for (u, s) in self.circ_u.iter().zip(&self.circ_s) {
            write!(f, " (circ_u) {u} (circ_s) {s}")?;
        }
        Ok(())
    }
}

impl<D: HostDataMut, W: ZnxWord> ReaderFrom for GGSWShare<D, W> {
    /// Fails with [`std::io::ErrorKind::InvalidData`] on a share whose parts do
    /// not match one GGSW of the allocation's rank.
    fn read_from<R: std::io::Read>(&mut self, reader: &mut R) -> std::io::Result<()> {
        self.parts_mut().try_for_each(|part| part.read_from(reader))?;
        // A compressed part's size does not depend on its output rank, so a share of another rank reads without error.
        let (col0, circ) = (
            ggsw_share_part_layout(self, Rank(1), self.rank()),
            ggsw_share_part_layout(self, self.rank(), self.rank()),
        );
        if self.circ_u.len() != self.rank().as_usize()
            || self.col0.gglwe_layout() != col0
            || self.circ_u.iter().chain(&self.circ_s).any(|part| part.gglwe_layout() != circ)
        {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "invalid share: parts do not form one GGSW",
            ));
        }
        Ok(())
    }
}

impl<D: HostDataRef, W: ZnxWord> WriterTo for GGSWShare<D, W> {
    fn write_to<Wr: std::io::Write>(&self, writer: &mut Wr) -> std::io::Result<()> {
        self.parts().try_for_each(|part| part.write_to(writer))
    }
}
