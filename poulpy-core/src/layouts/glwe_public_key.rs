use std::fmt;

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use poulpy_hal::AlignedBuf;
use poulpy_hal::layouts::{
    Backend, Data, HostDataMut, HostDataRef, MatZnx, MatZnxAtBackendMut, MatZnxToBackendMut, MatZnxToBackendRef, ReaderFrom,
    WriterTo, ZnxWord, mat_znx_at_backend_mut_from_mut, mat_znx_at_backend_ref_from_ref,
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
/// columns, entry `l` at input column `l`. Its entries are canonical: a writer
/// through `at_mut` or `at_view_mut` must leave them so.
pub struct GLWEPublicKey<D: Data, W: ZnxWord> {
    pub(crate) data: MatZnx<D, W>,
    pub(crate) base2k: Base2K,
    pub(crate) k: TorusPrecision,
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

impl<D: HostDataRef, W: ZnxWord> GLWEPublicKey<D, W> {
    /// Entry `l` is the encryption of zero paired with the ephemeral `u_l`.
    pub fn at(&self, l: usize) -> GLWE<&[u8], W> {
        GLWE {
            data: self.data.at(0, l),
            base2k: self.base2k,
            k: self.k,
            canonical: true,
        }
    }
}

impl<D: HostDataMut, W: ZnxWord> GLWEPublicKey<D, W> {
    pub fn at_mut(&mut self, l: usize) -> GLWE<&mut [u8], W> {
        GLWE {
            data: self.data.at_mut(0, l),
            base2k: self.base2k,
            k: self.k,
            canonical: true,
        }
    }
}

/// Backend view of entry `l` of an owned public key.
pub trait GLWEPublicKeyAtViewMut<BE: Backend> {
    fn at_view_mut(&mut self, l: usize) -> GLWEViewMut<'_, BE>;
}

impl<BE: Backend> GLWEPublicKeyAtViewMut<BE> for GLWEPublicKey<BE::OwnedBuf, BE::ZnxWord> {
    fn at_view_mut(&mut self, l: usize) -> GLWEViewMut<'_, BE> {
        GLWEViewMut::from_inner(GLWE {
            data: MatZnxAtBackendMut::<BE>::at_backend_mut(&mut self.data, 0, l),
            base2k: self.base2k,
            k: self.k,
            canonical: true,
        })
    }
}

/// Backend view of entry `l` of a borrowed public key.
pub fn glwe_public_key_at_view<'a, BE: Backend>(pk: &'a GLWEPublicKeyBackendRef<'_, BE>, l: usize) -> GLWEViewRef<'a, BE> {
    GLWEViewRef::from_inner(GLWE {
        data: mat_znx_at_backend_ref_from_ref::<BE>(&pk.data, 0, l),
        base2k: pk.base2k,
        k: pk.k,
        canonical: true,
    })
}

pub(crate) fn glwe_public_key_at_view_mut<'a, BE: Backend>(
    pk: &'a mut GLWEPublicKeyBackendMut<'_, BE>,
    l: usize,
) -> GLWEViewMut<'a, BE> {
    GLWEViewMut::from_inner(GLWE {
        data: mat_znx_at_backend_mut_from_mut::<BE>(&mut pk.data, 0, l),
        base2k: pk.base2k,
        k: pk.k,
        canonical: true,
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
                <poulpy_hal::layouts::HostBytesBackend>::alloc_bytes(MatZnx::<AlignedBuf, W>::bytes_of(
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
            .field("dist", &self.dist)
            .field("data", &self.data)
            .finish()
    }
}

impl<D: HostDataMut, W: ZnxWord> ReaderFrom for GLWEPublicKey<D, W> {
    /// Fails with [`std::io::ErrorKind::InvalidData`], leaving the key's layout
    /// unchanged, on a stream that is not one row of `r >= 1` encryptions of
    /// zero of a nonzero degree at its precision.
    fn read_from<R: std::io::Read>(&mut self, reader: &mut R) -> std::io::Result<()> {
        let dist = Distribution::read_from(reader)?;
        let base2k = Base2K(reader.read_u32::<LittleEndian>()?);
        let k = TorusPrecision(reader.read_u32::<LittleEndian>()?);
        // The matrix header (n, size, rows, cols_in, cols_out) is checked before the matrix overwrites the key.
        let mut header = [0u8; 40];
        reader.read_exact(&mut header)?;
        let field = |i: usize| u64::from_le_bytes(header[8 * i..8 * i + 8].try_into().unwrap());
        let (n, size, rows, cols_in, cols_out) = (field(0), field(1), field(2), field(3), field(4));
        if base2k.0 == 0
            || n == 0
            || rows != 1
            || cols_in == 0
            || cols_in.checked_add(1) != Some(cols_out)
            || size != u64::from(k.0.div_ceil(base2k.0))
        {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "invalid public key: not one row of rank encryptions of zero at its precision",
            ));
        }
        self.data.read_from(&mut std::io::Read::chain(header.as_slice(), reader))?;
        self.dist = dist;
        self.base2k = base2k;
        self.k = k;
        Ok(())
    }
}

impl<D: HostDataRef, W: ZnxWord> WriterTo for GLWEPublicKey<D, W> {
    fn write_to<Wr: std::io::Write>(&self, writer: &mut Wr) -> std::io::Result<()> {
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
            dist: self.dist,
        }
    }
}

pub trait GLWEPublicKeyToBackendMut<BE: Backend> {
    fn to_backend_mut(&mut self) -> GLWEPublicKeyBackendMut<'_, BE>;
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
            dist: self.dist,
        }
    }
}
