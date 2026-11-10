use std::{
    fmt,
    ops::{Deref, DerefMut},
};

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use poulpy_hal::AlignedBuf;
use poulpy_hal::layouts::{
    Backend, Data, HostDataMut, HostDataRef, MatZnx, MatZnxToBackendMut, MatZnxToBackendRef, ReaderFrom, Ring, Standard, VecZnx,
    WriterTo, ZnxWord, mat_znx_at_backend_mut_from_mut, mat_znx_at_backend_ref_from_mut, mat_znx_at_backend_ref_from_ref,
    mat_znx_backend_mut_from_mut, mat_znx_backend_ref_from_mut, mat_znx_backend_ref_from_ref,
};

use crate::{
    GetDistribution, GetDistributionMut,
    dist::Distribution,
    layouts::{
        Base2K, Degree, GLWEInfos, LWEInfos, Rank, TorusPrecision,
        compressed::{GLWECompressed, GLWECompressedViewMut, GLWECompressedViewRef},
    },
};

/// Seed-compressed [`GLWEPublicKey`](crate::layouts::GLWEPublicKey) of rank `r`:
/// the bodies of its `r` encryptions of zero, entry `l` at input column `l` of
/// a one-row matrix, each mask regenerated from its own seed.
pub struct GLWEPublicKeyCompressed<D: Data, W: ZnxWord, Rg: Ring = Standard> {
    pub(crate) noise: Option<crate::ComponentNoise>,
    pub(crate) data: MatZnx<D, W, Rg>,
    pub(crate) base2k: Base2K,
    pub(crate) k: TorusPrecision,
    pub(crate) seed: Vec<[u8; 32]>,
    pub(crate) dist: Distribution,
}

impl<D: Data, W: ZnxWord, Rg: Ring> PartialEq for GLWEPublicKeyCompressed<D, W, Rg>
where
    MatZnx<D, W, Rg>: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.noise == other.noise
            && self.data == other.data
            && self.base2k == other.base2k
            && self.k == other.k
            && self.seed == other.seed
            && self.dist == other.dist
    }
}

impl<D: Data, W: ZnxWord, Rg: Ring> Eq for GLWEPublicKeyCompressed<D, W, Rg> where MatZnx<D, W, Rg>: Eq {}

impl<D: Data + Clone, W: ZnxWord, Rg: Ring> Clone for GLWEPublicKeyCompressed<D, W, Rg> {
    fn clone(&self) -> Self {
        Self {
            noise: self.noise.clone(),
            data: self.data.clone(),
            base2k: self.base2k,
            k: self.k,
            seed: self.seed.clone(),
            dist: self.dist,
        }
    }
}

impl<D: Data, W: ZnxWord, Rg: Ring> GLWEPublicKeyCompressed<D, W, Rg> {
    pub fn data(&self) -> &MatZnx<D, W, Rg> {
        &self.data
    }

    pub fn data_mut(&mut self) -> &mut MatZnx<D, W, Rg> {
        self.noise = None;
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

impl<D: Data, W: ZnxWord, Rg: Ring> GLWEPublicKeyCompressedSeed for GLWEPublicKeyCompressed<D, W, Rg> {
    fn seed(&self) -> &[[u8; 32]] {
        &self.seed
    }
}

impl<D: Data, W: ZnxWord, Rg: Ring> GLWEPublicKeyCompressedSeedMut for GLWEPublicKeyCompressed<D, W, Rg> {
    fn seed_mut(&mut self) -> &mut [[u8; 32]] {
        &mut self.seed
    }
}

impl<D: Data, W: ZnxWord, Rg: Ring> GetDistribution for GLWEPublicKeyCompressed<D, W, Rg> {
    fn dist(&self) -> &Distribution {
        &self.dist
    }
}

impl<D: Data, W: ZnxWord, Rg: Ring> GetDistributionMut for GLWEPublicKeyCompressed<D, W, Rg> {
    fn dist_mut(&mut self) -> &mut Distribution {
        &mut self.dist
    }
}

impl<D: Data, W: ZnxWord, Rg: Ring> LWEInfos for GLWEPublicKeyCompressed<D, W, Rg> {
    fn noise(&self) -> Option<crate::ComponentNoise> {
        self.noise.clone()
    }

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

impl<D: Data, W: ZnxWord, Rg: Ring> GLWEInfos for GLWEPublicKeyCompressed<D, W, Rg> {
    fn rank(&self) -> Rank {
        Rank(self.data.cols_in() as u32)
    }
}

impl<D: Data, W: ZnxWord, Rg: Ring> GLWEPublicKeyCompressed<D, W, Rg> {
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
            noise: None,
            data: MatZnx::from_data(
                B::alloc_zeroed_bytes(B::mat_znx_bytes_of(n.into(), 1, rank, 1, size)),
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
        MatZnx::<AlignedBuf, W, Rg>::bytes_of(n.into(), 1, rank.into(), 1, k.0.div_ceil(base2k.0) as usize)
    }
}

impl<D: HostDataRef, W: ZnxWord, Rg: Ring> fmt::Debug for GLWEPublicKeyCompressed<D, W, Rg> {
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

impl<D: HostDataMut, W: ZnxWord, Rg: Ring> ReaderFrom for GLWEPublicKeyCompressed<D, W, Rg> {
    /// Rejects an invalid shape with [`std::io::ErrorKind::InvalidData`] before
    /// changing the layout. Every read attempt clears the previous metadata.
    /// The stream must contain one row of `r >= 1` bodies at the destination's
    /// rank, with a nonzero degree and one seed per body.
    fn read_from<R: std::io::Read>(&mut self, reader: &mut R) -> std::io::Result<()> {
        self.noise = None;
        let metadata = crate::ComponentNoise::read_optional(reader, self.data.cols_in() + 1)?;
        let dist = Distribution::read_from(reader)?;
        let base2k = Base2K(reader.read_u32::<LittleEndian>()?);
        let k = TorusPrecision(reader.read_u32::<LittleEndian>()?);
        let seeds = u64::from(reader.read_u32::<LittleEndian>()?);
        // The matrix header (n, size, rows, cols_in, cols_out, past the ring tag HAL checks) is checked before the key changes.
        let mut header = [0u8; 41];
        reader.read_exact(&mut header)?;
        let field = |i: usize| u64::from_le_bytes(header[1 + 8 * i..9 + 8 * i].try_into().unwrap());
        let (n, size, rows, cols_in, cols_out) = (field(0), field(1), field(2), field(3), field(4));
        if base2k.0 == 0
            || n == 0
            || rows != 1
            || cols_in == 0
            || cols_in != self.data.cols_in() as u64
            || cols_out != 1
            || seeds != cols_in
            || size != u64::from(k.0.div_ceil(base2k.0))
        {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "invalid compressed public key: not one row of rank seeded bodies at its precision",
            ));
        }
        let components = usize::try_from(cols_in + 1)
            .map_err(|_| std::io::Error::new(std::io::ErrorKind::InvalidData, "noise component count is too large"))?;
        crate::layouts::validate_noise_components(metadata.as_ref(), components)?;
        let entries = self.data.cols_in();
        crate::layouts::read_mat_znx_with_shape(
            &mut self.data,
            &mut std::io::Read::chain(header.as_slice(), &mut *reader),
            Some(1),
            Some(entries),
            1,
        )?;
        let mut seed = vec![[0u8; 32]; seeds as usize];
        for s in &mut seed {
            reader.read_exact(s)?;
        }
        self.noise = metadata;
        self.dist = dist;
        self.base2k = base2k;
        self.k = k;
        self.seed = seed;
        Ok(())
    }
}

impl<D: HostDataRef, W: ZnxWord, Rg: Ring> WriterTo for GLWEPublicKeyCompressed<D, W, Rg> {
    fn write_to<Wr: std::io::Write>(&self, writer: &mut Wr) -> std::io::Result<()> {
        self.dist.validate_wire()?;
        crate::layouts::validate_noise_components(self.noise.as_ref(), self.data.cols_in() + 1)?;
        crate::ComponentNoise::write_optional(self.noise.as_ref(), writer)?;
        self.dist.write_to(writer)?;
        writer.write_u32::<LittleEndian>(self.base2k.0)?;
        writer.write_u32::<LittleEndian>(self.k.0)?;
        writer.write_u32::<LittleEndian>(self.seed.len() as u32)?;
        self.data.write_to(writer)?;
        self.seed.iter().try_for_each(|s| writer.write_all(s))
    }
}

pub struct GLWEPublicKeyCompressedBackendRef<'a, BE: Backend + 'a, Rg: Ring = Standard> {
    inner: GLWEPublicKeyCompressed<BE::BufRef<'a>, BE::ZnxWord, Rg>,
}

impl<'a, BE: Backend + 'a, Rg: Ring> GLWEPublicKeyCompressedBackendRef<'a, BE, Rg> {
    pub fn from_inner(inner: GLWEPublicKeyCompressed<BE::BufRef<'a>, BE::ZnxWord, Rg>) -> Self {
        Self { inner }
    }

    pub fn into_inner(self) -> GLWEPublicKeyCompressed<BE::BufRef<'a>, BE::ZnxWord, Rg> {
        self.inner
    }

    /// Entry `l`, with its seed.
    pub fn at_view(&self, l: usize) -> GLWECompressedViewRef<'_, BE, Rg> {
        let pk = &self.inner;
        GLWECompressedViewRef::from_inner(entry(mat_znx_at_backend_ref_from_ref::<BE, Rg>(&pk.data, 0, l), pk, l))
    }
}

impl<'a, BE: Backend + 'a, Rg: Ring> Deref for GLWEPublicKeyCompressedBackendRef<'a, BE, Rg> {
    type Target = GLWEPublicKeyCompressed<BE::BufRef<'a>, BE::ZnxWord, Rg>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

pub struct GLWEPublicKeyCompressedBackendMut<'a, BE: Backend + 'a, Rg: Ring = Standard> {
    inner: GLWEPublicKeyCompressed<BE::BufMut<'a>, BE::ZnxWord, Rg>,
}

impl<'a, BE: Backend + 'a, Rg: Ring> GLWEPublicKeyCompressedBackendMut<'a, BE, Rg> {
    pub fn from_inner(inner: GLWEPublicKeyCompressed<BE::BufMut<'a>, BE::ZnxWord, Rg>) -> Self {
        Self { inner }
    }

    pub fn into_inner(self) -> GLWEPublicKeyCompressed<BE::BufMut<'a>, BE::ZnxWord, Rg> {
        self.inner
    }

    /// Entry `l`, with its seed.
    pub fn at_view(&self, l: usize) -> GLWECompressedViewRef<'_, BE, Rg> {
        let pk = &self.inner;
        GLWECompressedViewRef::from_inner(entry(mat_znx_at_backend_ref_from_mut::<BE, Rg>(&pk.data, 0, l), pk, l))
    }

    /// Entry `l`, with a copy of its seed: seeds are written through
    /// [`GLWEPublicKeyCompressedSeedMut`] on the key.
    pub fn at_view_mut(&mut self, l: usize) -> GLWECompressedViewMut<'_, BE, Rg> {
        let (base2k, k, rank, seed) = (self.inner.base2k, self.inner.k, self.inner.rank(), self.inner.seed[l]);
        self.inner.noise = None;
        GLWECompressedViewMut::from_inner(GLWECompressed {
            noise: None,
            data: mat_znx_at_backend_mut_from_mut::<BE, Rg>(&mut self.inner.data, 0, l),
            base2k,
            k,
            rank,
            seed,
            plaintext_meta: None,
        })
    }
}

impl<'a, BE: Backend + 'a, Rg: Ring> Deref for GLWEPublicKeyCompressedBackendMut<'a, BE, Rg> {
    type Target = GLWEPublicKeyCompressed<BE::BufMut<'a>, BE::ZnxWord, Rg>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<BE: Backend, Rg: Ring> DerefMut for GLWEPublicKeyCompressedBackendMut<'_, BE, Rg> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

fn entry<D: Data, E: Data, W: ZnxWord, Rg: Ring>(
    data: VecZnx<D, W, Rg>,
    pk: &GLWEPublicKeyCompressed<E, W, Rg>,
    l: usize,
) -> GLWECompressed<D, W, Rg> {
    GLWECompressed {
        noise: pk.noise.clone(),
        data,
        base2k: pk.base2k,
        k: pk.k,
        rank: pk.rank(),
        seed: pk.seed[l],
        plaintext_meta: None,
    }
}

macro_rules! impl_compressed_public_key_infos_for_inner {
    ($ty:ident) => {
        impl<BE: Backend, Rg: Ring> LWEInfos for $ty<'_, BE, Rg> {
            fn noise(&self) -> Option<crate::ComponentNoise> {
                self.inner.noise.clone()
            }

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

        impl<BE: Backend, Rg: Ring> GLWEInfos for $ty<'_, BE, Rg> {
            fn rank(&self) -> Rank {
                self.inner.rank()
            }
        }

        impl<BE: Backend, Rg: Ring> GetDistribution for $ty<'_, BE, Rg> {
            fn dist(&self) -> &Distribution {
                self.inner.dist()
            }
        }

        impl<BE: Backend, Rg: Ring> GLWEPublicKeyCompressedSeed for $ty<'_, BE, Rg> {
            fn seed(&self) -> &[[u8; 32]] {
                self.inner.seed()
            }
        }
    };
}

impl_compressed_public_key_infos_for_inner!(GLWEPublicKeyCompressedBackendRef);
impl_compressed_public_key_infos_for_inner!(GLWEPublicKeyCompressedBackendMut);

impl<BE: Backend, Rg: Ring> GetDistributionMut for GLWEPublicKeyCompressedBackendMut<'_, BE, Rg> {
    fn dist_mut(&mut self) -> &mut Distribution {
        self.inner.dist_mut()
    }
}

impl<BE: Backend, Rg: Ring> GLWEPublicKeyCompressedSeedMut for GLWEPublicKeyCompressedBackendMut<'_, BE, Rg> {
    fn seed_mut(&mut self) -> &mut [[u8; 32]] {
        self.inner.seed_mut()
    }
}

pub trait GLWEPublicKeyCompressedToBackendRef<BE: Backend, Rg: Ring = Standard> {
    fn to_backend_ref(&self) -> GLWEPublicKeyCompressedBackendRef<'_, BE, Rg>;
}

pub trait GLWEPublicKeyCompressedToBackendMut<BE: Backend, Rg: Ring = Standard>:
    GLWEPublicKeyCompressedToBackendRef<BE, Rg>
{
    /// Records component noise metadata on this key.
    fn set_noise(&mut self, metadata: Option<crate::ComponentNoise>);

    /// Borrows coefficients mutably and clears the owner's component noise metadata.
    fn to_backend_mut(&mut self) -> GLWEPublicKeyCompressedBackendMut<'_, BE, Rg>;
}

impl<BE: Backend, D: Data, Rg: Ring> GLWEPublicKeyCompressedToBackendRef<BE, Rg> for GLWEPublicKeyCompressed<D, BE::ZnxWord, Rg>
where
    MatZnx<D, BE::ZnxWord, Rg>: MatZnxToBackendRef<BE, Rg>,
{
    fn to_backend_ref(&self) -> GLWEPublicKeyCompressedBackendRef<'_, BE, Rg> {
        GLWEPublicKeyCompressedBackendRef::from_inner(GLWEPublicKeyCompressed {
            noise: self.noise.clone(),
            data: self.data.to_backend_ref(),
            base2k: self.base2k,
            k: self.k,
            seed: self.seed.clone(),
            dist: self.dist,
        })
    }
}

impl<BE: Backend, D: Data, Rg: Ring> GLWEPublicKeyCompressedToBackendMut<BE, Rg> for GLWEPublicKeyCompressed<D, BE::ZnxWord, Rg>
where
    MatZnx<D, BE::ZnxWord, Rg>: MatZnxToBackendRef<BE, Rg> + MatZnxToBackendMut<BE, Rg>,
{
    fn set_noise(&mut self, metadata: Option<crate::ComponentNoise>) {
        self.noise = crate::layouts::checked_noise(metadata, crate::layouts::GLWEInfos::rank(self).as_usize() + 1);
    }

    fn to_backend_mut(&mut self) -> GLWEPublicKeyCompressedBackendMut<'_, BE, Rg> {
        self.noise = None;
        GLWEPublicKeyCompressedBackendMut::from_inner(GLWEPublicKeyCompressed {
            noise: None,
            data: self.data.to_backend_mut(),
            base2k: self.base2k,
            k: self.k,
            seed: self.seed.clone(),
            dist: self.dist,
        })
    }
}

impl<BE: Backend, Rg: Ring> GLWEPublicKeyCompressedToBackendRef<BE, Rg> for GLWEPublicKeyCompressedBackendRef<'_, BE, Rg> {
    fn to_backend_ref(&self) -> GLWEPublicKeyCompressedBackendRef<'_, BE, Rg> {
        GLWEPublicKeyCompressedBackendRef::from_inner(GLWEPublicKeyCompressed {
            noise: self.inner.noise.clone(),
            data: mat_znx_backend_ref_from_ref::<BE, Rg>(&self.inner.data),
            base2k: self.inner.base2k,
            k: self.inner.k,
            seed: self.inner.seed.clone(),
            dist: self.inner.dist,
        })
    }
}

impl<BE: Backend, Rg: Ring> GLWEPublicKeyCompressedToBackendRef<BE, Rg> for GLWEPublicKeyCompressedBackendMut<'_, BE, Rg> {
    fn to_backend_ref(&self) -> GLWEPublicKeyCompressedBackendRef<'_, BE, Rg> {
        GLWEPublicKeyCompressedBackendRef::from_inner(GLWEPublicKeyCompressed {
            noise: self.inner.noise.clone(),
            data: mat_znx_backend_ref_from_mut::<BE, Rg>(&self.inner.data),
            base2k: self.inner.base2k,
            k: self.inner.k,
            seed: self.inner.seed.clone(),
            dist: self.inner.dist,
        })
    }
}

impl<BE: Backend, Rg: Ring> GLWEPublicKeyCompressedToBackendMut<BE, Rg> for GLWEPublicKeyCompressedBackendMut<'_, BE, Rg> {
    fn set_noise(&mut self, metadata: Option<crate::ComponentNoise>) {
        self.inner.noise = crate::layouts::checked_noise(metadata, crate::layouts::GLWEInfos::rank(self).as_usize() + 1);
    }

    fn to_backend_mut(&mut self) -> GLWEPublicKeyCompressedBackendMut<'_, BE, Rg> {
        self.inner.noise = None;
        GLWEPublicKeyCompressedBackendMut::from_inner(GLWEPublicKeyCompressed {
            noise: None,
            data: mat_znx_backend_mut_from_mut::<BE, Rg>(&mut self.inner.data),
            base2k: self.inner.base2k,
            k: self.inner.k,
            seed: self.inner.seed.clone(),
            dist: self.inner.dist,
        })
    }
}

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
            T: GLWEInfos + GetDistribution + GLWEPublicKeyCompressedSeed + GLWEPublicKeyCompressedToBackendRef<BE, Rg>,
            Rg: Ring,
        >() {
        }
        fn key_mut<
            BE: Backend,
            T: GetDistributionMut + GLWEPublicKeyCompressedSeedMut + GLWEPublicKeyCompressedToBackendMut<BE, Rg>,
            Rg: Ring,
        >() {
        }
        fn assert_for<'a, BE: Backend + 'a>() {
            key::<BE, GLWEPublicKeyCompressedBackendRef<'a, BE>, Standard>();
            key::<BE, GLWEPublicKeyCompressedBackendMut<'a, BE>, Standard>();
            key_mut::<BE, GLWEPublicKeyCompressedBackendMut<'a, BE>, Standard>();
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
