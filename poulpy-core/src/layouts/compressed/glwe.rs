use poulpy_hal::{
    api::{VecZnxCopyBackend, VecZnxFillUniformSourceBackend},
    layouts::{
        Backend, Data, FillUniform, HostDataMut, HostDataRef, Module, ReaderFrom, VecZnx, VecZnxToBackendMut, VecZnxToBackendRef,
        WriterTo,
    },
    source::Source,
};

use crate::layouts::{Base2K, Degree, GLWEInfos, GLWEToBackendMut, GetDegree, LWEInfos, Rank, SetLWEInfos, TorusPrecision};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::fmt;
use std::ops::{Deref, DerefMut};

/// Seed-compressed GLWE ciphertext layout.
///
/// Stores only the body component of a [`GLWE`] ciphertext; the mask
/// polynomials are regenerated deterministically from a 32-byte PRNG
/// seed during decompression. This reduces the serialized size by a
/// factor proportional to the rank.
#[derive(PartialEq, Eq, Clone)]
pub struct GLWECompressed<D: Data> {
    pub(crate) data: VecZnx<D>,
    pub(crate) base2k: Base2K,
    pub(crate) rank: Rank,
    pub(crate) seed: [u8; 32],
}

pub type GLWECompressedBackendRef<'a, BE> = GLWECompressed<<BE as Backend>::BufRef<'a>>;
pub type GLWECompressedBackendMut<'a, BE> = GLWECompressed<<BE as Backend>::BufMut<'a>>;

pub struct GLWECompressedViewRef<'a, BE: Backend + 'a> {
    inner: GLWECompressedBackendRef<'a, BE>,
}

pub struct GLWECompressedViewMut<'a, BE: Backend + 'a> {
    inner: GLWECompressedBackendMut<'a, BE>,
}

impl<'a, BE: Backend + 'a> GLWECompressedViewRef<'a, BE> {
    pub fn from_inner(inner: GLWECompressedBackendRef<'a, BE>) -> Self {
        Self { inner }
    }

    pub fn into_inner(self) -> GLWECompressedBackendRef<'a, BE> {
        self.inner
    }
}

impl<'a, BE: Backend + 'a> GLWECompressedViewMut<'a, BE> {
    pub fn from_inner(inner: GLWECompressedBackendMut<'a, BE>) -> Self {
        Self { inner }
    }

    pub fn into_inner(self) -> GLWECompressedBackendMut<'a, BE> {
        self.inner
    }
}

impl<'a, BE: Backend + 'a> Deref for GLWECompressedViewRef<'a, BE> {
    type Target = GLWECompressedBackendRef<'a, BE>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<'a, BE: Backend + 'a> Deref for GLWECompressedViewMut<'a, BE> {
    type Target = GLWECompressedBackendMut<'a, BE>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<'a, BE: Backend + 'a> DerefMut for GLWECompressedViewMut<'a, BE> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

impl<'a, BE: Backend + 'a> LWEInfos for GLWECompressedViewRef<'a, BE> {
    fn base2k(&self) -> Base2K {
        self.inner.base2k()
    }

    fn size(&self) -> usize {
        self.inner.size()
    }

    fn n(&self) -> Degree {
        self.inner.n()
    }
}

impl<'a, BE: Backend + 'a> LWEInfos for GLWECompressedViewMut<'a, BE> {
    fn base2k(&self) -> Base2K {
        self.inner.base2k()
    }

    fn size(&self) -> usize {
        self.inner.size()
    }

    fn n(&self) -> Degree {
        self.inner.n()
    }
}

impl<'a, BE: Backend + 'a> GLWEInfos for GLWECompressedViewRef<'a, BE> {
    fn rank(&self) -> Rank {
        self.inner.rank()
    }
}

impl<'a, BE: Backend + 'a> GLWEInfos for GLWECompressedViewMut<'a, BE> {
    fn rank(&self) -> Rank {
        self.inner.rank()
    }
}

/// Provides mutable access to the PRNG seed of a compressed GLWE.
pub trait GLWECompressedSeedMut {
    /// Returns a mutable reference to the 32-byte PRNG seed.
    fn seed_mut(&mut self) -> &mut [u8; 32];
}

impl<D: Data> GLWECompressedSeedMut for GLWECompressed<D> {
    fn seed_mut(&mut self) -> &mut [u8; 32] {
        &mut self.seed
    }
}

/// Provides read access to the PRNG seed of a compressed GLWE.
pub trait GLWECompressedSeed {
    /// Returns a reference to the 32-byte PRNG seed.
    fn seed(&self) -> &[u8; 32];
}

impl<D: HostDataRef> GLWECompressedSeed for GLWECompressed<D> {
    fn seed(&self) -> &[u8; 32] {
        &self.seed
    }
}

impl<D: Data> LWEInfos for GLWECompressed<D> {
    fn base2k(&self) -> Base2K {
        self.base2k
    }

    fn size(&self) -> usize {
        self.data.size()
    }

    fn n(&self) -> Degree {
        Degree(self.data.n() as u32)
    }
}
impl<D: Data> GLWEInfos for GLWECompressed<D> {
    fn rank(&self) -> Rank {
        self.rank
    }
}

impl<D: Data> LWEInfos for &GLWECompressed<D> {
    fn n(&self) -> Degree {
        (*self).n()
    }

    fn base2k(&self) -> Base2K {
        (*self).base2k()
    }

    fn size(&self) -> usize {
        (*self).size()
    }
}

impl<D: Data> GLWEInfos for &GLWECompressed<D> {
    fn rank(&self) -> Rank {
        (*self).rank()
    }
}

impl<D: HostDataRef> fmt::Debug for GLWECompressed<D> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{self}")
    }
}

impl<D: HostDataRef> fmt::Display for GLWECompressed<D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "GLWECompressed: base2k={} k={} rank={} seed={:?}: {}",
            self.base2k(),
            self.max_k(),
            self.rank(),
            self.seed,
            self.data
        )
    }
}

impl<D: HostDataMut> FillUniform for GLWECompressed<D> {
    fn fill_uniform(&mut self, log_bound: usize, source: &mut Source) {
        self.data.fill_uniform(log_bound, source);
    }
}

impl GLWECompressed<Vec<u8>> {
    /// Allocates a new compressed GLWE by copying parameters from an existing info provider.
    pub(crate) fn alloc_from_infos<A>(infos: &A) -> Self
    where
        A: GLWEInfos,
    {
        Self::alloc(infos.n(), infos.base2k(), infos.max_k(), infos.rank())
    }

    /// Allocates a new compressed GLWE with the given parameters.
    ///
    /// The underlying `VecZnx` is sized to hold one column of
    /// `ceil(k / base2k)` limbs at ring degree `n`.
    pub(crate) fn alloc(n: Degree, base2k: Base2K, k: TorusPrecision, rank: Rank) -> Self {
        let size: usize = k.0.div_ceil(base2k.0) as usize;
        GLWECompressed {
            data: VecZnx::from_data(
                poulpy_hal::layouts::HostBytesBackend::alloc_bytes(VecZnx::<Vec<u8>>::bytes_of(n.into(), 1, size)),
                n.into(),
                1,
                size,
            ),
            base2k,
            rank,
            seed: [0u8; 32],
        }
    }

    /// Returns the serialized byte size by copying parameters from an existing info provider.
    pub fn bytes_of_from_infos<A>(infos: &A) -> usize
    where
        A: GLWEInfos,
    {
        Self::bytes_of(infos.n(), infos.base2k(), infos.max_k())
    }

    /// Returns the serialized byte size for a compressed GLWE with the given parameters.
    pub fn bytes_of(n: Degree, base2k: Base2K, k: TorusPrecision) -> usize {
        VecZnx::bytes_of(n.into(), 1, k.0.div_ceil(base2k.0) as usize)
    }
}

/// Deserializes the metadata (k, base2k, rank, seed) followed by the body data.
impl<D: HostDataMut> ReaderFrom for GLWECompressed<D> {
    fn read_from<R: std::io::Read>(&mut self, reader: &mut R) -> std::io::Result<()> {
        self.base2k = Base2K(reader.read_u32::<LittleEndian>()?);
        self.rank = Rank(reader.read_u32::<LittleEndian>()?);
        reader.read_exact(&mut self.seed)?;
        self.data.read_from(reader)
    }
}

/// Serializes the metadata (k, base2k, rank, seed) followed by the body data.
impl<D: HostDataRef> WriterTo for GLWECompressed<D> {
    fn write_to<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<()> {
        writer.write_u32::<LittleEndian>(self.base2k.into())?;
        writer.write_u32::<LittleEndian>(self.rank.into())?;
        writer.write_all(&self.seed)?;
        self.data.write_to(writer)
    }
}

/// Trait for decompressing a [`GLWECompressed`] into a standard [`GLWE`].
///
/// Copies the body from the compressed ciphertext and regenerates
/// the mask polynomials from the stored PRNG seed.
pub trait GLWEDecompress
where
    Self: GetDegree + VecZnxFillUniformSourceBackend<Self::Backend> + VecZnxCopyBackend<Self::Backend>,
{
    type Backend: Backend;

    /// Decompresses `other` into `res` by copying the body and regenerating the mask.
    fn decompress_glwe<R, O>(&self, res: &mut R, other: &O)
    where
        R: GLWEToBackendMut<Self::Backend> + SetLWEInfos,
        O: GLWECompressedToBackendRef<Self::Backend> + GLWEInfos,
    {
        let other = other.to_backend_ref();
        {
            let res = &mut res.to_backend_mut();
            assert_eq!(
                res.n(),
                self.ring_degree(),
                "invalid receiver: res.n()={} != other.n()={}",
                res.n(),
                self.ring_degree()
            );

            assert_eq!(res.glwe_layout(), other.glwe_layout());

            let mut source: Source = Source::new(other.seed);

            self.vec_znx_copy_backend(&mut res.data, 0, &other.data, 0);
            (1..(other.rank() + 1).into()).for_each(|i| {
                self.vec_znx_fill_uniform_source_backend(other.base2k.into(), &mut res.data, i, &mut source);
            });
        }

        res.set_base2k(other.base2k());
    }
}

impl<B: Backend> GLWEDecompress for Module<B>
where
    Self: GetDegree + VecZnxFillUniformSourceBackend<B> + VecZnxCopyBackend<B>,
{
    type Backend = B;
}

// module-only API: decompression is provided by `GLWEDecompress` on `Module`.

pub trait GLWECompressedToBackendRef<BE: Backend> {
    fn to_backend_ref(&self) -> GLWECompressedBackendRef<'_, BE>;
}

impl<BE: Backend> GLWECompressedToBackendRef<BE> for GLWECompressed<BE::OwnedBuf> {
    fn to_backend_ref(&self) -> GLWECompressedBackendRef<'_, BE> {
        GLWECompressed {
            seed: self.seed,
            base2k: self.base2k,
            rank: self.rank,
            data: <VecZnx<BE::OwnedBuf> as VecZnxToBackendRef<BE>>::to_backend_ref(&self.data),
        }
    }
}

impl<'a, BE: Backend + 'a> GLWECompressedToBackendRef<BE> for GLWECompressedViewRef<'a, BE> {
    fn to_backend_ref(&self) -> GLWECompressedBackendRef<'_, BE> {
        GLWECompressed {
            seed: self.inner.seed,
            base2k: self.inner.base2k,
            rank: self.inner.rank,
            data: poulpy_hal::layouts::vec_znx_backend_ref_from_ref::<BE>(&self.inner.data),
        }
    }
}

impl<'a, BE: Backend + 'a> GLWECompressedToBackendRef<BE> for GLWECompressedViewMut<'a, BE> {
    fn to_backend_ref(&self) -> GLWECompressedBackendRef<'_, BE> {
        GLWECompressed {
            seed: self.inner.seed,
            base2k: self.inner.base2k,
            rank: self.inner.rank,
            data: poulpy_hal::layouts::vec_znx_backend_ref_from_mut::<BE>(&self.inner.data),
        }
    }
}

pub trait GLWECompressedToBackendMut<BE: Backend>: GLWECompressedToBackendRef<BE> {
    fn to_backend_mut(&mut self) -> GLWECompressedBackendMut<'_, BE>;
}

impl<BE: Backend> GLWECompressedToBackendMut<BE> for GLWECompressed<BE::OwnedBuf> {
    fn to_backend_mut(&mut self) -> GLWECompressedBackendMut<'_, BE> {
        GLWECompressed {
            seed: self.seed,
            base2k: self.base2k,
            rank: self.rank,
            data: <VecZnx<BE::OwnedBuf> as VecZnxToBackendMut<BE>>::to_backend_mut(&mut self.data),
        }
    }
}
