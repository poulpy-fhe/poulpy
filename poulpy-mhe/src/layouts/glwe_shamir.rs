use std::fmt;

use poulpy_core::layouts::{Base2K, Degree, GLWE, LWEInfos, Rank, TorusPrecision};
use poulpy_hal::layouts::{Backend, Data, HostDataMut, HostDataRef, ReaderFrom, WriterTo, ZnxWord};

/// Layout of a Shamir sharing of a rank-`rank` secret over the Galois ring
/// `GR(2^k, gr_degree)` with threshold `threshold`: every GR component is a
/// torus polynomial of precision `k` in base `2^base2k`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GLWEShamirLayout {
    pub n: Degree,
    pub base2k: Base2K,
    pub k: TorusPrecision,
    pub rank: Rank,
    pub gr_degree: usize,
    pub threshold: usize,
}

pub type GLWEShamirPolynomialOwned<BE> = GLWEShamirPolynomial<<BE as Backend>::OwnedBuf, <BE as Backend>::ZnxWord>;

/// A party's Shamir polynomial, kept local: `threshold` GR-valued
/// coefficients, the constant term (the secret) first, each `rank * gr_degree`
/// columns of `inner`.
#[derive(Clone)]
pub struct GLWEShamirPolynomial<D: Data, W: ZnxWord> {
    pub(crate) inner: GLWE<D, W>,
    pub(crate) rank: Rank,
    pub(crate) gr_degree: usize,
    pub(crate) threshold: usize,
}

impl<D: Data, W: ZnxWord> GLWEShamirPolynomial<D, W> {
    pub fn layout(&self) -> GLWEShamirLayout {
        shamir_layout(&self.inner, self.rank, self.gr_degree, self.threshold)
    }
}

pub type GLWEShamirShareOwned<BE> = GLWEShamirShare<<BE as Backend>::OwnedBuf, <BE as Backend>::ZnxWord>;

/// A t-out-of-N Shamir share: the evaluation of Shamir polynomials at a
/// party's point, `rank * gr_degree` columns of `inner`. Shares are secret and
/// travel over private channels.
#[derive(Clone)]
pub struct GLWEShamirShare<D: Data, W: ZnxWord> {
    pub(crate) inner: GLWE<D, W>,
    pub(crate) rank: Rank,
    pub(crate) gr_degree: usize,
    pub(crate) threshold: usize,
}

impl<D: Data, W: ZnxWord> GLWEShamirShare<D, W> {
    pub fn layout(&self) -> GLWEShamirLayout {
        shamir_layout(&self.inner, self.rank, self.gr_degree, self.threshold)
    }
}

fn shamir_layout<D: Data, W: ZnxWord>(inner: &GLWE<D, W>, rank: Rank, gr_degree: usize, threshold: usize) -> GLWEShamirLayout {
    GLWEShamirLayout {
        n: inner.n(),
        base2k: inner.base2k(),
        k: inner.k(),
        rank,
        gr_degree,
        threshold,
    }
}

impl<D: Data, W: ZnxWord> PartialEq for GLWEShamirShare<D, W>
where
    GLWE<D, W>: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.inner == other.inner
            && self.rank == other.rank
            && self.gr_degree == other.gr_degree
            && self.threshold == other.threshold
    }
}

impl<D: Data, W: ZnxWord> Eq for GLWEShamirShare<D, W> where GLWE<D, W>: Eq {}

impl<D: Data, W: ZnxWord> fmt::Debug for GLWEShamirShare<D, W> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "GLWEShamirShare: {:?}", self.layout())
    }
}

impl<D: HostDataMut, W: ZnxWord> ReaderFrom for GLWEShamirShare<D, W> {
    fn read_from<R: std::io::Read>(&mut self, reader: &mut R) -> std::io::Result<()> {
        let mut threshold = [0u8; 4];
        reader.read_exact(&mut threshold)?;
        self.threshold = u32::from_le_bytes(threshold) as usize;
        self.inner.read_from(reader)
    }
}

impl<D: HostDataRef, W: ZnxWord> WriterTo for GLWEShamirShare<D, W> {
    fn write_to<Wr: std::io::Write>(&self, writer: &mut Wr) -> std::io::Result<()> {
        writer.write_all(&(self.threshold as u32).to_le_bytes())?;
        self.inner.write_to(writer)
    }
}
