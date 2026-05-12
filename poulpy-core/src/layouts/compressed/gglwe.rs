use poulpy_hal::{
    layouts::{
        Backend, Data, FillUniform, HostDataMut, HostDataRef, MatZnx, MatZnxToBackendMut, MatZnxToBackendRef, Module, ReaderFrom,
        WriterTo, mat_znx_at_backend_mut_from_mut, mat_znx_at_backend_ref_from_ref, mat_znx_backend_mut_from_mut,
        mat_znx_backend_ref_from_mut,
    },
    source::Source,
};

use crate::layouts::{
    Base2K, Degree, Dnum, Dsize, GGLWEInfos, GGLWEToBackendMut, GLWEInfos, LWEInfos, Rank, TorusPrecision,
    compressed::{GLWECompressed, GLWECompressedBackendMut, GLWECompressedViewMut, GLWECompressedViewRef, GLWEDecompress},
};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::{
    fmt,
    ops::{Deref, DerefMut},
};

/// Seed-compressed GGLWE (gadget GLWE) ciphertext layout.
///
/// Stores only the body components of a [`GGLWE`] ciphertext matrix;
/// the mask polynomials are regenerated deterministically from 32-byte
/// PRNG seeds during decompression.
#[derive(PartialEq, Eq, Clone)]
pub struct GGLWECompressed<D: Data> {
    pub(crate) data: MatZnx<D>,
    pub(crate) base2k: Base2K,
    pub(crate) k: TorusPrecision,
    pub(crate) rank_out: Rank,
    pub(crate) dsize: Dsize,
    pub(crate) seed: Vec<[u8; 32]>,
}

pub struct GGLWECompressedBackendRef<'a, BE: Backend + 'a> {
    inner: GGLWECompressed<BE::BufRef<'a>>,
}

impl<'a, BE: Backend + 'a> GGLWECompressedBackendRef<'a, BE> {
    pub fn from_inner(inner: GGLWECompressed<BE::BufRef<'a>>) -> Self {
        Self { inner }
    }

    pub fn into_inner(self) -> GGLWECompressed<BE::BufRef<'a>> {
        self.inner
    }

    pub fn at_view(&self, row: usize, col: usize) -> GLWECompressedViewRef<'_, BE> {
        GLWECompressedViewRef::from_inner(gglwe_compressed_at_backend_ref_from_ref::<BE>(&self.inner, row, col))
    }
}

impl<'a, BE: Backend + 'a> Deref for GGLWECompressedBackendRef<'a, BE> {
    type Target = GGLWECompressed<BE::BufRef<'a>>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

pub struct GGLWECompressedBackendMut<'a, BE: Backend + 'a> {
    inner: GGLWECompressed<BE::BufMut<'a>>,
}

impl<'a, BE: Backend + 'a> GGLWECompressedBackendMut<'a, BE> {
    pub fn from_inner(inner: GGLWECompressed<BE::BufMut<'a>>) -> Self {
        Self { inner }
    }

    pub fn into_inner(self) -> GGLWECompressed<BE::BufMut<'a>> {
        self.inner
    }

    pub fn at_view_mut(&mut self, row: usize, col: usize) -> GLWECompressedViewMut<'_, BE> {
        GLWECompressedViewMut::from_inner(gglwe_compressed_at_backend_mut_from_mut::<BE>(&mut self.inner, row, col))
    }
}

impl<'a, BE: Backend + 'a> Deref for GGLWECompressedBackendMut<'a, BE> {
    type Target = GGLWECompressed<BE::BufMut<'a>>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<'a, BE: Backend + 'a> DerefMut for GGLWECompressedBackendMut<'a, BE> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

impl_gglwe_infos_for_inner!(GGLWECompressedBackendRef<'a, BE>, ['a, BE: Backend + 'a]; inner);
impl_gglwe_infos_for_inner!(GGLWECompressedBackendMut<'a, BE>, ['a, BE: Backend + 'a]; inner);

impl<'a, BE: Backend + 'a> GGLWECompressedSeedMut for GGLWECompressedBackendMut<'a, BE> {
    fn seed_mut(&mut self) -> &mut Vec<[u8; 32]> {
        &mut self.inner.seed
    }
}

impl<'a, BE: Backend + 'a> GGLWECompressedToBackendRef<BE> for GGLWECompressedBackendRef<'a, BE> {
    fn to_backend_ref(&self) -> GGLWECompressedBackendRef<'_, BE> {
        GGLWECompressedBackendRef::from_inner(GGLWECompressed {
            k: self.inner.k,
            base2k: self.inner.base2k,
            dsize: self.inner.dsize,
            seed: self.inner.seed.clone(),
            rank_out: self.inner.rank_out,
            data: poulpy_hal::layouts::mat_znx_backend_ref_from_ref::<BE>(&self.inner.data),
        })
    }
}

impl<'a, BE: Backend + 'a> GGLWECompressedToBackendRef<BE> for GGLWECompressedBackendMut<'a, BE> {
    fn to_backend_ref(&self) -> GGLWECompressedBackendRef<'_, BE> {
        GGLWECompressedBackendRef::from_inner(GGLWECompressed {
            k: self.inner.k,
            base2k: self.inner.base2k,
            dsize: self.inner.dsize,
            seed: self.inner.seed.clone(),
            rank_out: self.inner.rank_out,
            data: mat_znx_backend_ref_from_mut::<BE>(&self.inner.data),
        })
    }
}

impl<'a, BE: Backend + 'a> GGLWECompressedToBackendMut<BE> for GGLWECompressedBackendMut<'a, BE> {
    fn to_backend_mut(&mut self) -> GGLWECompressedBackendMut<'_, BE> {
        GGLWECompressedBackendMut::from_inner(GGLWECompressed {
            k: self.inner.k,
            base2k: self.inner.base2k,
            dsize: self.inner.dsize,
            seed: self.inner.seed.clone(),
            rank_out: self.inner.rank_out,
            data: mat_znx_backend_mut_from_mut::<BE>(&mut self.inner.data),
        })
    }
}

/// Provides mutable access to the PRNG seeds of a compressed GGLWE.
pub trait GGLWECompressedSeedMut {
    /// Returns a mutable reference to the vector of 32-byte PRNG seeds.
    fn seed_mut(&mut self) -> &mut Vec<[u8; 32]>;
}

impl<D: Data> GGLWECompressedSeedMut for GGLWECompressed<D> {
    fn seed_mut(&mut self) -> &mut Vec<[u8; 32]> {
        &mut self.seed
    }
}

impl<D: Data> GGLWECompressedSeedMut for &mut GGLWECompressed<D> {
    fn seed_mut(&mut self) -> &mut Vec<[u8; 32]> {
        &mut self.seed
    }
}

/// Provides read access to the PRNG seeds of a compressed GGLWE.
pub trait GGLWECompressedSeed {
    /// Returns a reference to the vector of 32-byte PRNG seeds.
    fn seed(&self) -> &Vec<[u8; 32]>;
}

impl<D: HostDataRef> GGLWECompressedSeed for GGLWECompressed<D> {
    fn seed(&self) -> &Vec<[u8; 32]> {
        &self.seed
    }
}
impl<D: Data> LWEInfos for GGLWECompressed<D> {
    fn n(&self) -> Degree {
        Degree(self.data.n() as u32)
    }

    fn base2k(&self) -> Base2K {
        self.base2k
    }

    fn size(&self) -> usize {
        self.data.size()
    }
}
impl<D: Data> GLWEInfos for GGLWECompressed<D> {
    fn rank(&self) -> Rank {
        self.rank_out()
    }
}

impl<D: Data> GGLWEInfos for GGLWECompressed<D> {
    fn rank_in(&self) -> Rank {
        Rank(self.data.cols_in() as u32)
    }

    fn rank_out(&self) -> Rank {
        self.rank_out
    }

    fn dsize(&self) -> Dsize {
        self.dsize
    }

    fn dnum(&self) -> Dnum {
        Dnum(self.data.rows() as u32)
    }
}

impl<D: Data> LWEInfos for &GGLWECompressed<D> {
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

impl<D: Data> GLWEInfos for &GGLWECompressed<D> {
    fn rank(&self) -> Rank {
        (*self).rank()
    }
}

impl<D: Data> GGLWEInfos for &GGLWECompressed<D> {
    fn rank_in(&self) -> Rank {
        (*self).rank_in()
    }

    fn rank_out(&self) -> Rank {
        (*self).rank_out()
    }

    fn dsize(&self) -> Dsize {
        (*self).dsize()
    }

    fn dnum(&self) -> Dnum {
        (*self).dnum()
    }
}

impl<D: HostDataRef> fmt::Debug for GGLWECompressed<D> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{self}")
    }
}

impl<D: HostDataMut> FillUniform for GGLWECompressed<D> {
    fn fill_uniform(&mut self, log_bound: usize, source: &mut Source) {
        self.data.fill_uniform(log_bound, source);
    }
}

impl<D: HostDataRef> fmt::Display for GGLWECompressed<D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "(GGLWECompressed: base2k={} k={} dsize={}) {}",
            self.base2k.0, self.k.0, self.dsize.0, self.data
        )
    }
}

impl GGLWECompressed<Vec<u8>> {
    /// Allocates a new compressed GGLWE by copying parameters from an existing info provider.
    pub(crate) fn alloc_from_infos<A>(infos: &A) -> Self
    where
        A: GGLWEInfos,
    {
        Self::alloc(
            infos.n(),
            infos.base2k(),
            infos.max_k(),
            infos.rank_in(),
            infos.rank_out(),
            infos.dnum(),
            infos.dsize(),
        )
    }

    /// Allocates a new compressed GGLWE with the given parameters.
    pub(crate) fn alloc(
        n: Degree,
        base2k: Base2K,
        k: TorusPrecision,
        rank_in: Rank,
        rank_out: Rank,
        dnum: Dnum,
        dsize: Dsize,
    ) -> Self {
        let size: usize = k.0.div_ceil(base2k.0) as usize;
        debug_assert!(
            size as u32 > dsize.0,
            "invalid gglwe: ceil(k/base2k): {size} <= dsize: {}",
            dsize.0
        );

        assert!(
            dnum.0 * dsize.0 <= size as u32,
            "invalid gglwe: dnum: {} * dsize:{} > ceil(k/base2k): {size}",
            dnum.0,
            dsize.0,
        );

        GGLWECompressed {
            data: MatZnx::from_data(
                poulpy_hal::layouts::HostBytesBackend::alloc_bytes(MatZnx::<Vec<u8>>::bytes_of(
                    n.into(),
                    dnum.into(),
                    rank_in.into(),
                    1,
                    size,
                )),
                n.into(),
                dnum.into(),
                rank_in.into(),
                1,
                size,
            ),
            k,
            base2k,
            dsize,
            rank_out,
            seed: vec![[0u8; 32]; (dnum.0 * rank_in.0) as usize],
        }
    }

    /// Returns the serialized byte size by copying parameters from an existing info provider.
    pub fn bytes_of_from_infos<A>(infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        Self::bytes_of(
            infos.n(),
            infos.base2k(),
            infos.max_k(),
            infos.rank_in(),
            infos.dnum(),
            infos.dsize(),
        )
    }

    /// Returns the serialized byte size for a compressed GGLWE with the given parameters.
    pub fn bytes_of(n: Degree, base2k: Base2K, k: TorusPrecision, rank_in: Rank, dnum: Dnum, dsize: Dsize) -> usize {
        let size: usize = k.0.div_ceil(base2k.0) as usize;
        debug_assert!(
            size as u32 > dsize.0,
            "invalid gglwe: ceil(k/base2k): {size} <= dsize: {}",
            dsize.0
        );

        assert!(
            dnum.0 * dsize.0 <= size as u32,
            "invalid gglwe: dnum: {} * dsize:{} > ceil(k/base2k): {size}",
            dnum.0,
            dsize.0,
        );

        MatZnx::bytes_of(n.into(), dnum.into(), rank_in.into(), 1, k.0.div_ceil(base2k.0) as usize)
    }
}

impl<D: HostDataMut> ReaderFrom for GGLWECompressed<D> {
    fn read_from<R: std::io::Read>(&mut self, reader: &mut R) -> std::io::Result<()> {
        self.k = TorusPrecision(reader.read_u32::<LittleEndian>()?);
        self.base2k = Base2K(reader.read_u32::<LittleEndian>()?);
        self.dsize = Dsize(reader.read_u32::<LittleEndian>()?);
        self.rank_out = Rank(reader.read_u32::<LittleEndian>()?);
        let seed_len: u32 = reader.read_u32::<LittleEndian>()?;
        self.seed = vec![[0u8; 32]; seed_len as usize];
        for s in &mut self.seed {
            reader.read_exact(s)?;
        }
        self.data.read_from(reader)
    }
}

impl<D: HostDataRef> WriterTo for GGLWECompressed<D> {
    fn write_to<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<()> {
        writer.write_u32::<LittleEndian>(self.k.into())?;
        writer.write_u32::<LittleEndian>(self.base2k.into())?;
        writer.write_u32::<LittleEndian>(self.dsize.into())?;
        writer.write_u32::<LittleEndian>(self.rank_out.into())?;
        writer.write_u32::<LittleEndian>(self.seed.len() as u32)?;
        for s in &self.seed {
            writer.write_all(s)?;
        }
        self.data.write_to(writer)
    }
}

/// Trait for decompressing a [`GGLWECompressed`] into a standard [`GGLWE`].
///
/// Iterates over every (row, column) entry, decompressing each
/// compressed GLWE row individually via [`GLWEDecompress`].
pub trait GGLWEDecompress
where
    Self: GLWEDecompress,
{
    /// Decompresses `other` into `res`.
    fn decompress_gglwe<R, O>(&self, res: &mut R, other: &O)
    where
        R: GGLWEToBackendMut<Self::Backend> + GGLWEInfos,
        O: GGLWECompressedToBackendRef<Self::Backend> + GGLWEInfos,
    {
        let mut res = res.to_backend_mut();
        let other = other.to_backend_ref();

        assert_eq!(res.dsize(), other.dsize());
        assert!(res.dnum() <= other.dnum());

        let rank_in: usize = res.rank_in().into();
        let dnum: usize = res.dnum().into();
        for col_i in 0..rank_in {
            for row_i in 0..dnum {
                let mut dst = res.at_view_mut(row_i, col_i);
                let src = other.at_view(row_i, col_i);
                self.decompress_glwe(&mut dst, &src);
            }
        }
    }
}

impl<B: Backend> GGLWEDecompress for Module<B> where Self: GLWEDecompress {}

// module-only API: decompression is provided by `GGLWEDecompress` on `Module`.

pub trait GGLWECompressedToBackendRef<BE: Backend> {
    fn to_backend_ref(&self) -> GGLWECompressedBackendRef<'_, BE>;
}

impl<BE: Backend> GGLWECompressedToBackendRef<BE> for GGLWECompressed<BE::OwnedBuf> {
    fn to_backend_ref(&self) -> GGLWECompressedBackendRef<'_, BE> {
        GGLWECompressedBackendRef::from_inner(GGLWECompressed {
            k: self.max_k(),
            base2k: self.base2k(),
            dsize: self.dsize(),
            seed: self.seed.clone(),
            rank_out: self.rank_out,
            data: <MatZnx<BE::OwnedBuf> as MatZnxToBackendRef<BE>>::to_backend_ref(&self.data),
        })
    }
}

impl<'b, BE: Backend + 'b> GGLWECompressedToBackendRef<BE> for &GGLWECompressed<BE::BufRef<'b>> {
    fn to_backend_ref(&self) -> GGLWECompressedBackendRef<'_, BE> {
        GGLWECompressedBackendRef::from_inner(GGLWECompressed {
            k: self.max_k(),
            base2k: self.base2k(),
            dsize: self.dsize(),
            seed: self.seed.clone(),
            rank_out: self.rank_out,
            data: poulpy_hal::layouts::mat_znx_backend_ref_from_ref::<BE>(&self.data),
        })
    }
}

impl<'b, BE: Backend + 'b> GGLWECompressedToBackendRef<BE> for &mut GGLWECompressed<BE::BufMut<'b>> {
    fn to_backend_ref(&self) -> GGLWECompressedBackendRef<'_, BE> {
        GGLWECompressedBackendRef::from_inner(GGLWECompressed {
            k: self.max_k(),
            base2k: self.base2k(),
            dsize: self.dsize(),
            seed: self.seed.clone(),
            rank_out: self.rank_out,
            data: mat_znx_backend_ref_from_mut::<BE>(&self.data),
        })
    }
}

pub trait GGLWECompressedToBackendMut<BE: Backend>: GGLWECompressedToBackendRef<BE> {
    fn to_backend_mut(&mut self) -> GGLWECompressedBackendMut<'_, BE>;
}

impl<BE: Backend> GGLWECompressedToBackendMut<BE> for GGLWECompressed<BE::OwnedBuf> {
    fn to_backend_mut(&mut self) -> GGLWECompressedBackendMut<'_, BE> {
        GGLWECompressedBackendMut::from_inner(GGLWECompressed {
            k: self.max_k(),
            base2k: self.base2k(),
            dsize: self.dsize(),
            seed: self.seed.clone(),
            rank_out: self.rank_out,
            data: <MatZnx<BE::OwnedBuf> as MatZnxToBackendMut<BE>>::to_backend_mut(&mut self.data),
        })
    }
}

impl<'b, BE: Backend + 'b> GGLWECompressedToBackendMut<BE> for &mut GGLWECompressed<BE::BufMut<'b>> {
    fn to_backend_mut(&mut self) -> GGLWECompressedBackendMut<'_, BE> {
        GGLWECompressedBackendMut::from_inner(GGLWECompressed {
            k: self.max_k(),
            base2k: self.base2k(),
            dsize: self.dsize(),
            seed: self.seed.clone(),
            rank_out: self.rank_out,
            data: mat_znx_backend_mut_from_mut::<BE>(&mut self.data),
        })
    }
}

fn gglwe_compressed_at_backend_mut_from_mut<'a, 'b, BE: Backend>(
    gglwe: &'a mut GGLWECompressed<BE::BufMut<'b>>,
    row: usize,
    col: usize,
) -> GLWECompressedBackendMut<'a, BE> {
    let rank_in: usize = gglwe.rank_in().into();
    GLWECompressed {
        base2k: gglwe.base2k,
        rank: gglwe.rank_out,
        data: mat_znx_at_backend_mut_from_mut::<BE>(&mut gglwe.data, row, col),
        seed: gglwe.seed[rank_in * row + col],
    }
}

fn gglwe_compressed_at_backend_ref_from_ref<'a, 'b, BE: Backend>(
    gglwe: &'a GGLWECompressed<BE::BufRef<'b>>,
    row: usize,
    col: usize,
) -> crate::layouts::compressed::GLWECompressedBackendRef<'a, BE> {
    let rank_in: usize = gglwe.rank_in().into();
    GLWECompressed {
        base2k: gglwe.base2k,
        rank: gglwe.rank_out,
        data: mat_znx_at_backend_ref_from_ref::<BE>(&gglwe.data, row, col),
        seed: gglwe.seed[rank_in * row + col],
    }
}
