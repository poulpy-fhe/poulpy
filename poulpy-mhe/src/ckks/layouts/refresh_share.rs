use std::fmt;

use poulpy_hal::layouts::{Backend, Data, HostDataMut, HostDataRef, ReaderFrom, WriterTo, ZnxWord};

use crate::layouts::{GLWEEncToShareShare, GLWEShareToEncShare};

pub type CKKSRefreshShareOwned<BE> = CKKSRefreshShare<<BE as Backend>::OwnedBuf, <BE as Backend>::ZnxWord>;

/// One party's share of a collective CKKS refresh: the encryption-to-shares part and
/// the shares-to-encryption part built from the party's integers `M_i`.
///
/// Serializes as its encryption-to-shares part, then its shares-to-encryption part.
#[derive(Clone)]
pub struct CKKSRefreshShare<D: Data, W: ZnxWord> {
    pub(crate) e2s: GLWEEncToShareShare<D, W>,
    pub(crate) s2e: GLWEShareToEncShare<D, W>,
}

impl<D: Data, W: ZnxWord> CKKSRefreshShare<D, W> {
    pub fn e2s(&self) -> &GLWEEncToShareShare<D, W> {
        &self.e2s
    }

    pub fn e2s_mut(&mut self) -> &mut GLWEEncToShareShare<D, W> {
        &mut self.e2s
    }

    pub fn s2e(&self) -> &GLWEShareToEncShare<D, W> {
        &self.s2e
    }

    pub fn s2e_mut(&mut self) -> &mut GLWEShareToEncShare<D, W> {
        &mut self.s2e
    }
}

impl<D: Data, W: ZnxWord> PartialEq for CKKSRefreshShare<D, W>
where
    GLWEEncToShareShare<D, W>: PartialEq,
    GLWEShareToEncShare<D, W>: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.e2s == other.e2s && self.s2e == other.s2e
    }
}

impl<D: Data, W: ZnxWord> Eq for CKKSRefreshShare<D, W>
where
    GLWEEncToShareShare<D, W>: Eq,
    GLWEShareToEncShare<D, W>: Eq,
{
}

impl<D: HostDataRef, W: ZnxWord> fmt::Debug for CKKSRefreshShare<D, W> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "CKKSRefreshShare: {:?} {:?}", self.e2s, self.s2e)
    }
}

impl<D: HostDataMut, W: ZnxWord> ReaderFrom for CKKSRefreshShare<D, W> {
    fn read_from<R: std::io::Read>(&mut self, reader: &mut R) -> std::io::Result<()> {
        self.e2s.read_from(reader)?;
        self.s2e.read_from(reader)
    }
}

impl<D: HostDataRef, W: ZnxWord> WriterTo for CKKSRefreshShare<D, W> {
    fn write_to<Wr: std::io::Write>(&self, writer: &mut Wr) -> std::io::Result<()> {
        self.e2s.write_to(writer)?;
        self.s2e.write_to(writer)
    }
}
