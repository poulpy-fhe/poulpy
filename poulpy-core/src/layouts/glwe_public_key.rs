use std::fmt;

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use poulpy_hal::AlignedBuf;
use poulpy_hal::layouts::{
    Backend, Data, HostDataMut, HostDataRef, MatZnx, MatZnxToBackendMut, MatZnxToBackendRef, ReaderFrom, WriterTo, ZnxWord,
    mat_znx_at_backend_mut_from_mut, mat_znx_at_backend_ref_from_ref,
};

use crate::{
    GetDistribution, GetDistributionMut,
    dist::Distribution,
    layouts::{Base2K, Degree, GLWE, GLWEInfos, GLWEViewMut, GLWEViewRef, LWEInfos, Rank, TorusPrecision},
};

/// GLWE public key of rank `r`: `r` encryptions of zero under the secret `s`,
/// `pk_l = (b_l, a_{l,1}, .., a_{l,r})` with `b_l = -Sum_j a_{l,j} s_j + e_l`,
/// one per ephemeral of a public-key encryption.
///
/// Stored as a GGLWE is: a matrix of one row, `r` input and `r + 1` output
/// columns, entry `l` at input column `l`. The key carries one canonical flag
/// for all its entries.
pub struct GLWEPublicKey<D: Data, W: ZnxWord> {
    pub(crate) data: MatZnx<D, W>,
    pub(crate) base2k: Base2K,
    pub(crate) k: TorusPrecision,
    pub(crate) canonical: bool,
    pub(crate) dist: Distribution,
}

impl<D: Data, W: ZnxWord> PartialEq for GLWEPublicKey<D, W>
where
    MatZnx<D, W>: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.data == other.data && self.base2k == other.base2k && self.k == other.k && self.dist == other.dist
    }
}

impl<D: Data, W: ZnxWord> Eq for GLWEPublicKey<D, W> where MatZnx<D, W>: Eq {}

impl<D: Data, W: ZnxWord> GLWEPublicKey<D, W> {
    pub fn is_canonical(&self) -> bool {
        self.canonical
    }
}

impl<D: HostDataRef, W: ZnxWord> GLWEPublicKey<D, W> {
    /// Entry `l` is the encryption of zero paired with the ephemeral `u_l`.
    pub fn entry(&self, l: usize) -> GLWE<&[u8], W> {
        GLWE {
            data: self.data.at(0, l),
            base2k: self.base2k,
            k: self.k,
            canonical: self.canonical,
        }
    }
}

impl<D: HostDataMut, W: ZnxWord> GLWEPublicKey<D, W> {
    /// The view reports the key's canonical flag and drops a change to it:
    /// set it on the key with [`GLWEPublicKeyToBackendMut::set_canonical`].
    pub fn entry_mut(&mut self, l: usize) -> GLWE<&mut [u8], W> {
        let (base2k, k, canonical) = (self.base2k, self.k, self.canonical);
        GLWE {
            data: self.data.at_mut(0, l),
            base2k,
            k,
            canonical,
        }
    }
}

/// Backend view of entry `l` of a borrowed public key.
pub fn glwe_public_key_entry_view<'a, BE: Backend>(pk: &'a GLWEPublicKeyBackendRef<'_, BE>, l: usize) -> GLWEViewRef<'a, BE> {
    GLWEViewRef::from_inner(GLWE {
        data: mat_znx_at_backend_ref_from_ref::<BE>(&pk.data, 0, l),
        base2k: pk.base2k,
        k: pk.k,
        canonical: pk.canonical,
    })
}

/// Mutable backend view of entry `l`, with the same flag rule as [`GLWEPublicKey::entry_mut`].
pub fn glwe_public_key_entry_view_mut<'a, BE: Backend>(
    pk: &'a mut GLWEPublicKeyBackendMut<'_, BE>,
    l: usize,
) -> GLWEViewMut<'a, BE> {
    let (base2k, k, canonical) = (pk.base2k, pk.k, pk.canonical);
    GLWEViewMut::from_inner(GLWE {
        data: mat_znx_at_backend_mut_from_mut::<BE>(&mut pk.data, 0, l),
        base2k,
        k,
        canonical,
    })
}

impl<D: Data, W: ZnxWord> GetDistributionMut for GLWEPublicKey<D, W> {
    fn dist_mut(&mut self) -> &mut Distribution {
        &mut self.dist
    }
}

impl<D: Data, W: ZnxWord> GetDistribution for GLWEPublicKey<D, W> {
    fn dist(&self) -> &Distribution {
        &self.dist
    }
}

#[derive(PartialEq, Eq, Copy, Clone, Debug)]
pub struct GLWEPublicKeyLayout {
    pub n: Degree,
    pub base2k: Base2K,
    pub k: TorusPrecision,
    pub rank: Rank,
}

impl<D: Data, W: ZnxWord> LWEInfos for GLWEPublicKey<D, W> {
    fn base2k(&self) -> Base2K {
        self.base2k
    }

    fn n(&self) -> Degree {
        Degree(self.data.n() as u32)
    }

    fn max_size(&self) -> usize {
        self.data.size()
    }

    fn k(&self) -> TorusPrecision {
        self.k
    }
}

impl<D: Data, W: ZnxWord> GLWEInfos for GLWEPublicKey<D, W> {
    fn rank(&self) -> Rank {
        Rank(self.data.cols_in() as u32)
    }
}

impl LWEInfos for GLWEPublicKeyLayout {
    fn base2k(&self) -> Base2K {
        self.base2k
    }

    fn n(&self) -> Degree {
        self.n
    }

    fn max_size(&self) -> usize {
        self.k.div_ceil(self.base2k) as usize
    }

    fn k(&self) -> TorusPrecision {
        self.k
    }
}

impl GLWEInfos for GLWEPublicKeyLayout {
    fn rank(&self) -> Rank {
        self.rank
    }
}

#[expect(
    dead_code,
    reason = "host-owned constructors are kept for serialization and host-only staging"
)]
impl<W: ZnxWord> GLWEPublicKey<AlignedBuf, W> {
    pub(crate) fn alloc_from_infos<A>(infos: &A) -> Self
    where
        A: GLWEInfos,
    {
        Self::alloc(infos.n(), infos.base2k(), infos.k(), infos.rank())
    }

    pub(crate) fn alloc(n: Degree, base2k: Base2K, k: TorusPrecision, rank: Rank) -> Self {
        assert!(rank.as_usize() >= 1, "invalid public key: rank must be at least 1");
        let (rows, cols_in, cols_out, size) = (1, rank.as_usize(), (rank + 1).as_usize(), k.0.div_ceil(base2k.0) as usize);
        GLWEPublicKey {
            data: MatZnx::from_data(
                poulpy_hal::layouts::HostBytesBackend::alloc_bytes(MatZnx::<AlignedBuf, W>::bytes_of(
                    n.into(),
                    rows,
                    cols_in,
                    cols_out,
                    size,
                )),
                n.into(),
                rows,
                cols_in,
                cols_out,
                size,
            ),
            base2k,
            k,
            canonical: true,
            dist: Distribution::NONE,
        }
    }

    pub fn bytes_of_from_infos<A>(infos: &A) -> usize
    where
        A: GLWEInfos,
    {
        Self::bytes_of(infos.n(), infos.base2k(), infos.k(), infos.rank())
    }

    pub fn bytes_of(n: Degree, base2k: Base2K, k: TorusPrecision, rank: Rank) -> usize {
        MatZnx::<AlignedBuf, W>::bytes_of(n.into(), 1, rank.into(), (rank + 1).into(), k.0.div_ceil(base2k.0) as usize)
    }
}

impl<D: HostDataRef, W: ZnxWord> fmt::Debug for GLWEPublicKey<D, W> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GLWEPublicKey")
            .field("base2k", &self.base2k)
            .field("k", &self.k)
            .field("canonical", &self.canonical)
            .field("dist", &self.dist)
            .field("data", &self.data)
            .finish()
    }
}

impl<D: HostDataMut, W: ZnxWord> ReaderFrom for GLWEPublicKey<D, W> {
    fn read_from<R: std::io::Read>(&mut self, reader: &mut R) -> std::io::Result<()> {
        self.dist = Distribution::read_from(reader)?;
        self.base2k = Base2K(reader.read_u32::<LittleEndian>()?);
        self.k = TorusPrecision(reader.read_u32::<LittleEndian>()?);
        self.data.read_from(reader)?;
        let m = &self.data;
        if self.base2k.0 == 0
            || m.rows() != 1
            || m.cols_in() == 0
            || m.cols_out() != m.cols_in() + 1
            || m.size() != self.k.0.div_ceil(self.base2k.0) as usize
        {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "invalid public key: not one row of rank encryptions of zero at its precision",
            ));
        }
        self.canonical = true;
        Ok(())
    }
}

impl<D: HostDataRef, W: ZnxWord> WriterTo for GLWEPublicKey<D, W> {
    /// Fails with [`std::io::ErrorKind::InvalidInput`], writing nothing, when the
    /// canonical flag is clear: normalize first.
    fn write_to<Wr: std::io::Write>(&self, writer: &mut Wr) -> std::io::Result<()> {
        if !self.canonical {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "public key is not canonical: normalize it before serializing",
            ));
        }
        self.dist.write_to(writer)?;
        writer.write_u32::<LittleEndian>(self.base2k.0)?;
        writer.write_u32::<LittleEndian>(self.k.0)?;
        self.data.write_to(writer)
    }
}

pub type GLWEPublicKeyBackendRef<'a, BE> = GLWEPublicKey<<BE as Backend>::BufRef<'a>, <BE as Backend>::ZnxWord>;
pub type GLWEPublicKeyBackendMut<'a, BE> = GLWEPublicKey<<BE as Backend>::BufMut<'a>, <BE as Backend>::ZnxWord>;

pub trait GLWEPublicKeyToBackendRef<BE: Backend> {
    fn to_backend_ref(&self) -> GLWEPublicKeyBackendRef<'_, BE>;
}

impl<BE: Backend, D: Data> GLWEPublicKeyToBackendRef<BE> for GLWEPublicKey<D, BE::ZnxWord>
where
    MatZnx<D, BE::ZnxWord>: MatZnxToBackendRef<BE>,
{
    fn to_backend_ref(&self) -> GLWEPublicKeyBackendRef<'_, BE> {
        GLWEPublicKey {
            data: self.data.to_backend_ref(),
            base2k: self.base2k,
            k: self.k,
            canonical: self.canonical,
            dist: self.dist,
        }
    }
}

pub trait GLWEPublicKeyToBackendMut<BE: Backend> {
    fn to_backend_mut(&mut self) -> GLWEPublicKeyBackendMut<'_, BE>;

    /// Sets the key's canonical flag; a flag set on the view returned by
    /// [`Self::to_backend_mut`] is lost.
    fn set_canonical(&mut self, canonical: bool);
}

impl<BE: Backend, D: Data> GLWEPublicKeyToBackendMut<BE> for GLWEPublicKey<D, BE::ZnxWord>
where
    MatZnx<D, BE::ZnxWord>: MatZnxToBackendMut<BE>,
{
    fn to_backend_mut(&mut self) -> GLWEPublicKeyBackendMut<'_, BE> {
        GLWEPublicKey {
            data: self.data.to_backend_mut(),
            base2k: self.base2k,
            k: self.k,
            canonical: self.canonical,
            dist: self.dist,
        }
    }

    fn set_canonical(&mut self, canonical: bool) {
        self.canonical = canonical
    }
}
