use std::{
    fmt,
    ops::{Deref, DerefMut},
};

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use poulpy_hal::AlignedBuf;
use poulpy_hal::layouts::{
    Backend, Data, HostDataMut, HostDataRef, MatZnx, MatZnxToBackendMut, MatZnxToBackendRef, Module, ReaderFrom, VecZnx,
    WriterTo, ZnxWord, mat_znx_at_backend_mut_from_mut, mat_znx_at_backend_ref_from_mut, mat_znx_at_backend_ref_from_ref,
    mat_znx_backend_mut_from_mut, mat_znx_backend_ref_from_mut, mat_znx_backend_ref_from_ref,
};

use crate::{
    GetDistribution, GetDistributionMut,
    dist::Distribution,
    layouts::{
        Base2K, Degree, GLWEInfos, GLWEPublicKeyAtViewMut, GLWEPublicKeyToBackendMut, LWEInfos, Rank, TorusPrecision,
        compressed::{GLWECompressed, GLWECompressedViewMut, GLWECompressedViewRef, GLWEDecompress},
    },
};

/// Seed-compressed [`GLWEPublicKey`](crate::layouts::GLWEPublicKey) of rank `r`:
/// the bodies of its `r` encryptions of zero, entry `l` at input column `l` of
/// a one-row matrix, each mask regenerated from its own seed.
pub struct GLWEPublicKeyCompressed<D: Data, W: ZnxWord> {
    pub(crate) data: MatZnx<D, W>,
    pub(crate) base2k: Base2K,
    pub(crate) k: TorusPrecision,
    pub(crate) seed: Vec<[u8; 32]>,
    pub(crate) dist: Distribution,
}

impl<D: Data, W: ZnxWord> PartialEq for GLWEPublicKeyCompressed<D, W>
where
    MatZnx<D, W>: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.data == other.data
            && self.base2k == other.base2k
            && self.k == other.k
            && self.seed == other.seed
            && self.dist == other.dist
    }
}

impl<D: Data, W: ZnxWord> Eq for GLWEPublicKeyCompressed<D, W> where MatZnx<D, W>: Eq {}

impl<D: Data + Clone, W: ZnxWord> Clone for GLWEPublicKeyCompressed<D, W> {
    fn clone(&self) -> Self {
        Self {
            data: self.data.clone(),
            base2k: self.base2k,
            k: self.k,
            seed: self.seed.clone(),
            dist: self.dist,
        }
    }
}

impl<D: Data, W: ZnxWord> GLWEPublicKeyCompressed<D, W> {
    pub fn data(&self) -> &MatZnx<D, W> {
        &self.data
    }

    pub fn data_mut(&mut self) -> &mut MatZnx<D, W> {
        &mut self.data
    }
}

/// Read access to the entry seeds of a compressed public key, one per entry.
pub trait GLWEPublicKeyCompressedSeed {
    fn seed(&self) -> &[[u8; 32]];
}

/// Mutable access to the entry seeds of a compressed public key, one per entry.
pub trait GLWEPublicKeyCompressedSeedMut {
    fn seed_mut(&mut self) -> &mut [[u8; 32]];
}

impl<D: Data, W: ZnxWord> GLWEPublicKeyCompressedSeed for GLWEPublicKeyCompressed<D, W> {
    fn seed(&self) -> &[[u8; 32]] {
        &self.seed
    }
}

impl<D: Data, W: ZnxWord> GLWEPublicKeyCompressedSeedMut for GLWEPublicKeyCompressed<D, W> {
    fn seed_mut(&mut self) -> &mut [[u8; 32]] {
        &mut self.seed
    }
}

impl<D: Data, W: ZnxWord> GetDistribution for GLWEPublicKeyCompressed<D, W> {
    fn dist(&self) -> &Distribution {
        &self.dist
    }
}

impl<D: Data, W: ZnxWord> GetDistributionMut for GLWEPublicKeyCompressed<D, W> {
    fn dist_mut(&mut self) -> &mut Distribution {
        &mut self.dist
    }
}

impl<D: Data, W: ZnxWord> LWEInfos for GLWEPublicKeyCompressed<D, W> {
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

impl<D: Data, W: ZnxWord> GLWEInfos for GLWEPublicKeyCompressed<D, W> {
    fn rank(&self) -> Rank {
        Rank(self.data.cols_in() as u32)
    }
}

impl<D: Data, W: ZnxWord> GLWEPublicKeyCompressed<D, W> {
    pub(crate) fn alloc_from_infos<B: Backend<OwnedBuf = D, ZnxWord = W>, A>(infos: &A) -> Self
    where
        A: GLWEInfos,
    {
        Self::alloc::<B>(infos.n(), infos.base2k(), infos.k(), infos.rank())
    }

    pub(crate) fn alloc<B: Backend<OwnedBuf = D, ZnxWord = W>>(n: Degree, base2k: Base2K, k: TorusPrecision, rank: Rank) -> Self {
        assert!(rank.as_usize() >= 1, "invalid public key: rank must be at least 1");
        let (rank, size) = (rank.as_usize(), k.0.div_ceil(base2k.0) as usize);
        GLWEPublicKeyCompressed {
            data: MatZnx::from_data(
                B::alloc_zeroed_bytes(B::bytes_of_mat_znx(n.into(), 1, rank, 1, size)),
                n.into(),
                1,
                rank,
                1,
                size,
            ),
            base2k,
            k,
            seed: vec![[0u8; 32]; rank],
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
        MatZnx::<AlignedBuf, W>::bytes_of(n.into(), 1, rank.into(), 1, k.0.div_ceil(base2k.0) as usize)
    }
}

impl<D: HostDataRef, W: ZnxWord> fmt::Debug for GLWEPublicKeyCompressed<D, W> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GLWEPublicKeyCompressed")
            .field("base2k", &self.base2k)
            .field("k", &self.k)
            .field("dist", &self.dist)
            .field("seed", &self.seed)
            .field("data", &self.data)
            .finish()
    }
}

impl<D: HostDataMut, W: ZnxWord> ReaderFrom for GLWEPublicKeyCompressed<D, W> {
    /// Fails with [`std::io::ErrorKind::InvalidData`], leaving the key
    /// unchanged, on a stream that is not one row of `r >= 1` bodies of a
    /// nonzero degree at its precision with one seed per body.
    fn read_from<R: std::io::Read>(&mut self, reader: &mut R) -> std::io::Result<()> {
        let dist = Distribution::read_from(reader)?;
        let base2k = Base2K(reader.read_u32::<LittleEndian>()?);
        let k = TorusPrecision(reader.read_u32::<LittleEndian>()?);
        let seeds = u64::from(reader.read_u32::<LittleEndian>()?);
        // The matrix header (n, size, rows, cols_in, cols_out) is checked before the matrix overwrites the key.
        let mut header = [0u8; 40];
        reader.read_exact(&mut header)?;
        let field = |i: usize| u64::from_le_bytes(header[8 * i..8 * i + 8].try_into().unwrap());
        let (n, size, rows, cols_in, cols_out) = (field(0), field(1), field(2), field(3), field(4));
        if base2k.0 == 0
            || n == 0
            || rows != 1
            || cols_in == 0
            || cols_out != 1
            || seeds != cols_in
            || size != u64::from(k.0.div_ceil(base2k.0))
        {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "invalid compressed public key: not one row of rank seeded bodies at its precision",
            ));
        }
        self.data
            .read_from(&mut std::io::Read::chain(header.as_slice(), &mut *reader))?;
        let mut seed = vec![[0u8; 32]; seeds as usize];
        for s in &mut seed {
            reader.read_exact(s)?;
        }
        self.dist = dist;
        self.base2k = base2k;
        self.k = k;
        self.seed = seed;
        Ok(())
    }
}

impl<D: HostDataRef, W: ZnxWord> WriterTo for GLWEPublicKeyCompressed<D, W> {
    fn write_to<Wr: std::io::Write>(&self, writer: &mut Wr) -> std::io::Result<()> {
        self.dist.write_to(writer)?;
        writer.write_u32::<LittleEndian>(self.base2k.0)?;
        writer.write_u32::<LittleEndian>(self.k.0)?;
        writer.write_u32::<LittleEndian>(self.seed.len() as u32)?;
        self.data.write_to(writer)?;
        self.seed.iter().try_for_each(|s| writer.write_all(s))
    }
}

pub struct GLWEPublicKeyCompressedBackendRef<'a, BE: Backend + 'a> {
    inner: GLWEPublicKeyCompressed<BE::BufRef<'a>, BE::ZnxWord>,
}

impl<'a, BE: Backend + 'a> GLWEPublicKeyCompressedBackendRef<'a, BE> {
    pub fn from_inner(inner: GLWEPublicKeyCompressed<BE::BufRef<'a>, BE::ZnxWord>) -> Self {
        Self { inner }
    }

    pub fn into_inner(self) -> GLWEPublicKeyCompressed<BE::BufRef<'a>, BE::ZnxWord> {
        self.inner
    }

    /// Entry `l`, with its seed.
    pub fn at_view(&self, l: usize) -> GLWECompressedViewRef<'_, BE> {
        let pk = &self.inner;
        GLWECompressedViewRef::from_inner(entry(mat_znx_at_backend_ref_from_ref::<BE>(&pk.data, 0, l), pk, l))
    }
}

impl<'a, BE: Backend + 'a> Deref for GLWEPublicKeyCompressedBackendRef<'a, BE> {
    type Target = GLWEPublicKeyCompressed<BE::BufRef<'a>, BE::ZnxWord>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

pub struct GLWEPublicKeyCompressedBackendMut<'a, BE: Backend + 'a> {
    inner: GLWEPublicKeyCompressed<BE::BufMut<'a>, BE::ZnxWord>,
}

impl<'a, BE: Backend + 'a> GLWEPublicKeyCompressedBackendMut<'a, BE> {
    pub fn from_inner(inner: GLWEPublicKeyCompressed<BE::BufMut<'a>, BE::ZnxWord>) -> Self {
        Self { inner }
    }

    pub fn into_inner(self) -> GLWEPublicKeyCompressed<BE::BufMut<'a>, BE::ZnxWord> {
        self.inner
    }

    /// Entry `l`, with its seed.
    pub fn at_view(&self, l: usize) -> GLWECompressedViewRef<'_, BE> {
        let pk = &self.inner;
        GLWECompressedViewRef::from_inner(entry(mat_znx_at_backend_ref_from_mut::<BE>(&pk.data, 0, l), pk, l))
    }

    /// Entry `l`, with a copy of its seed: seeds are written through
    /// [`GLWEPublicKeyCompressedSeedMut`] on the key.
    pub fn at_view_mut(&mut self, l: usize) -> GLWECompressedViewMut<'_, BE> {
        let (base2k, k, rank, seed) = (self.inner.base2k, self.inner.k, self.inner.rank(), self.inner.seed[l]);
        GLWECompressedViewMut::from_inner(GLWECompressed {
            data: mat_znx_at_backend_mut_from_mut::<BE>(&mut self.inner.data, 0, l),
            base2k,
            k,
            rank,
            seed,
        })
    }
}

impl<'a, BE: Backend + 'a> Deref for GLWEPublicKeyCompressedBackendMut<'a, BE> {
    type Target = GLWEPublicKeyCompressed<BE::BufMut<'a>, BE::ZnxWord>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<BE: Backend> DerefMut for GLWEPublicKeyCompressedBackendMut<'_, BE> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

fn entry<D: Data, E: Data, W: ZnxWord>(data: VecZnx<D, W>, pk: &GLWEPublicKeyCompressed<E, W>, l: usize) -> GLWECompressed<D, W> {
    GLWECompressed {
        data,
        base2k: pk.base2k,
        k: pk.k,
        rank: pk.rank(),
        seed: pk.seed[l],
    }
}

macro_rules! impl_compressed_public_key_infos_for_inner {
    ($ty:ident) => {
        impl<BE: Backend> LWEInfos for $ty<'_, BE> {
            fn base2k(&self) -> Base2K {
                self.inner.base2k()
            }

            fn n(&self) -> Degree {
                self.inner.n()
            }

            fn max_size(&self) -> usize {
                self.inner.max_size()
            }

            fn k(&self) -> TorusPrecision {
                self.inner.k()
            }
        }

        impl<BE: Backend> GLWEInfos for $ty<'_, BE> {
            fn rank(&self) -> Rank {
                self.inner.rank()
            }
        }

        impl<BE: Backend> GetDistribution for $ty<'_, BE> {
            fn dist(&self) -> &Distribution {
                self.inner.dist()
            }
        }

        impl<BE: Backend> GLWEPublicKeyCompressedSeed for $ty<'_, BE> {
            fn seed(&self) -> &[[u8; 32]] {
                self.inner.seed()
            }
        }
    };
}

impl_compressed_public_key_infos_for_inner!(GLWEPublicKeyCompressedBackendRef);
impl_compressed_public_key_infos_for_inner!(GLWEPublicKeyCompressedBackendMut);

impl<BE: Backend> GetDistributionMut for GLWEPublicKeyCompressedBackendMut<'_, BE> {
    fn dist_mut(&mut self) -> &mut Distribution {
        self.inner.dist_mut()
    }
}

impl<BE: Backend> GLWEPublicKeyCompressedSeedMut for GLWEPublicKeyCompressedBackendMut<'_, BE> {
    fn seed_mut(&mut self) -> &mut [[u8; 32]] {
        self.inner.seed_mut()
    }
}

pub trait GLWEPublicKeyCompressedToBackendRef<BE: Backend> {
    fn to_backend_ref(&self) -> GLWEPublicKeyCompressedBackendRef<'_, BE>;
}

pub trait GLWEPublicKeyCompressedToBackendMut<BE: Backend>: GLWEPublicKeyCompressedToBackendRef<BE> {
    fn to_backend_mut(&mut self) -> GLWEPublicKeyCompressedBackendMut<'_, BE>;
}

impl<BE: Backend, D: Data> GLWEPublicKeyCompressedToBackendRef<BE> for GLWEPublicKeyCompressed<D, BE::ZnxWord>
where
    MatZnx<D, BE::ZnxWord>: MatZnxToBackendRef<BE>,
{
    fn to_backend_ref(&self) -> GLWEPublicKeyCompressedBackendRef<'_, BE> {
        GLWEPublicKeyCompressedBackendRef::from_inner(GLWEPublicKeyCompressed {
            data: self.data.to_backend_ref(),
            base2k: self.base2k,
            k: self.k,
            seed: self.seed.clone(),
            dist: self.dist,
        })
    }
}

impl<BE: Backend, D: Data> GLWEPublicKeyCompressedToBackendMut<BE> for GLWEPublicKeyCompressed<D, BE::ZnxWord>
where
    MatZnx<D, BE::ZnxWord>: MatZnxToBackendRef<BE> + MatZnxToBackendMut<BE>,
{
    fn to_backend_mut(&mut self) -> GLWEPublicKeyCompressedBackendMut<'_, BE> {
        GLWEPublicKeyCompressedBackendMut::from_inner(GLWEPublicKeyCompressed {
            data: self.data.to_backend_mut(),
            base2k: self.base2k,
            k: self.k,
            seed: self.seed.clone(),
            dist: self.dist,
        })
    }
}

impl<BE: Backend> GLWEPublicKeyCompressedToBackendRef<BE> for GLWEPublicKeyCompressedBackendRef<'_, BE> {
    fn to_backend_ref(&self) -> GLWEPublicKeyCompressedBackendRef<'_, BE> {
        GLWEPublicKeyCompressedBackendRef::from_inner(GLWEPublicKeyCompressed {
            data: mat_znx_backend_ref_from_ref::<BE>(&self.inner.data),
            base2k: self.inner.base2k,
            k: self.inner.k,
            seed: self.inner.seed.clone(),
            dist: self.inner.dist,
        })
    }
}

impl<BE: Backend> GLWEPublicKeyCompressedToBackendRef<BE> for GLWEPublicKeyCompressedBackendMut<'_, BE> {
    fn to_backend_ref(&self) -> GLWEPublicKeyCompressedBackendRef<'_, BE> {
        GLWEPublicKeyCompressedBackendRef::from_inner(GLWEPublicKeyCompressed {
            data: mat_znx_backend_ref_from_mut::<BE>(&self.inner.data),
            base2k: self.inner.base2k,
            k: self.inner.k,
            seed: self.inner.seed.clone(),
            dist: self.inner.dist,
        })
    }
}

impl<BE: Backend> GLWEPublicKeyCompressedToBackendMut<BE> for GLWEPublicKeyCompressedBackendMut<'_, BE> {
    fn to_backend_mut(&mut self) -> GLWEPublicKeyCompressedBackendMut<'_, BE> {
        GLWEPublicKeyCompressedBackendMut::from_inner(GLWEPublicKeyCompressed {
            data: mat_znx_backend_mut_from_mut::<BE>(&mut self.inner.data),
            base2k: self.inner.base2k,
            k: self.inner.k,
            seed: self.inner.seed.clone(),
            dist: self.inner.dist,
        })
    }
}

/// Decompresses a [`GLWEPublicKeyCompressed`] into a
/// [`GLWEPublicKey`](crate::layouts::GLWEPublicKey) of the same layout.
pub trait GLWEPublicKeyDecompress
where
    Self: GLWEDecompress,
{
    /// Copies each body, regenerates its mask from its seed, and copies the
    /// distribution.
    fn decompress_glwe_public_key<R, O>(&self, res: &mut R, other: &O)
    where
        R: GLWEPublicKeyToBackendMut<Self::Backend> + GetDistributionMut + GLWEInfos,
        O: GLWEPublicKeyCompressedToBackendRef<Self::Backend> + GetDistribution + GLWEInfos,
    {
        assert!(
            res.glwe_layout() == other.glwe_layout(),
            "invalid decompression: layouts differ"
        );
        {
            let mut res = res.to_backend_mut();
            let other = other.to_backend_ref();
            for l in 0..other.rank().as_usize() {
                self.decompress_glwe(&mut res.at_view_mut(l), &other.at_view(l));
            }
        }
        *res.dist_mut() = *other.dist();
    }
}

impl<B: Backend> GLWEPublicKeyDecompress for Module<B> where Self: GLWEDecompress {}

#[cfg(test)]
mod tests {
    use poulpy_hal::layouts::HostBytesBackend;

    use super::*;

    type Key = GLWEPublicKeyCompressed<AlignedBuf, i64>;

    fn key(rank: u32) -> Key {
        Key::alloc::<HostBytesBackend>(Degree(8), Base2K(12), TorusPrecision(30), Rank(rank))
    }

    #[test]
    fn borrows_forward_the_key_traits() {
        // Deref does not satisfy trait bounds: each borrow implements them itself.
        fn key<
            BE: Backend,
            T: GLWEInfos + GetDistribution + GLWEPublicKeyCompressedSeed + GLWEPublicKeyCompressedToBackendRef<BE>,
        >() {
        }
        fn key_mut<
            BE: Backend,
            T: GetDistributionMut + GLWEPublicKeyCompressedSeedMut + GLWEPublicKeyCompressedToBackendMut<BE>,
        >() {
        }
        fn assert_for<'a, BE: Backend + 'a>() {
            key::<BE, GLWEPublicKeyCompressedBackendRef<'a, BE>>();
            key::<BE, GLWEPublicKeyCompressedBackendMut<'a, BE>>();
            key_mut::<BE, GLWEPublicKeyCompressedBackendMut<'a, BE>>();
        }
        assert_for::<HostBytesBackend>();
    }

    #[test]
    fn read_round_trips_and_rejects_another_rank() {
        let mut want = key(2);
        want.seed_mut()[1] = [7u8; 32];
        want.dist = Distribution::TernaryProb(0.5);
        let mut bytes = Vec::new();
        want.write_to(&mut bytes).unwrap();

        let mut have = key(2);
        have.read_from(&mut bytes.as_slice()).unwrap();
        assert!(have == want);

        let mut smaller = key(1);
        assert!(smaller.read_from(&mut bytes.as_slice()).is_err());
        assert_eq!(smaller.rank(), Rank(1));
    }
}
