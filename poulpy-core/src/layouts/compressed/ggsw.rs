use poulpy_hal::{
    layouts::{
        Backend, Data, FillUniform, HostDataMut, HostDataRef, MatZnx, MatZnxToBackendMut, MatZnxToBackendRef, Module, ReaderFrom,
        WriterTo, mat_znx_at_backend_mut_from_mut, mat_znx_at_backend_ref_from_ref, mat_znx_backend_mut_from_mut,
        mat_znx_backend_ref_from_mut,
    },
    source::Source,
};

use crate::layouts::{
    Base2K, Degree, Dnum, Dsize, GGSWInfos, GGSWToBackendMut, GLWEInfos, LWEInfos, Rank, TorusPrecision,
    compressed::{
        GLWECompressed, GLWECompressedBackendMut, GLWECompressedBackendRef, GLWECompressedViewMut, GLWECompressedViewRef,
        GLWEDecompress,
    },
};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use poulpy_hal::layouts::ZnxWord;
use std::{
    fmt,
    ops::{Deref, DerefMut},
};

/// Seed-compressed GGSW (gadget GSW) ciphertext layout.
///
/// Stores only the body components of a [`GGSW`] ciphertext; the mask
/// polynomials are regenerated deterministically from 32-byte PRNG
/// seeds during decompression.
#[derive(PartialEq, Eq, Clone)]
pub struct GGSWCompressed<D: Data, W: ZnxWord> {
    pub(crate) data: MatZnx<D, W>,
    pub(crate) k_aux: TorusPrecision,
    pub(crate) base2k: Base2K,
    pub(crate) dsize: Dsize,
    pub(crate) rank: Rank,
    pub(crate) seed: Vec<[u8; 32]>,
}

pub struct GGSWCompressedBackendRef<'a, BE: Backend + 'a> {
    inner: GGSWCompressed<BE::BufRef<'a>, BE::ZnxWord>,
}

impl<'a, BE: Backend + 'a> GGSWCompressedBackendRef<'a, BE> {
    pub fn from_inner(inner: GGSWCompressed<BE::BufRef<'a>, BE::ZnxWord>) -> Self {
        Self { inner }
    }

    pub fn into_inner(self) -> GGSWCompressed<BE::BufRef<'a>, BE::ZnxWord> {
        self.inner
    }

    pub fn at_view(&self, row: usize, col: usize) -> GLWECompressedViewRef<'_, BE> {
        GLWECompressedViewRef::from_inner(ggsw_compressed_at_backend_ref_from_ref::<BE>(&self.inner, row, col))
    }
}

impl<'a, BE: Backend + 'a> Deref for GGSWCompressedBackendRef<'a, BE> {
    type Target = GGSWCompressed<BE::BufRef<'a>, BE::ZnxWord>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

pub struct GGSWCompressedBackendMut<'a, BE: Backend + 'a> {
    inner: GGSWCompressed<BE::BufMut<'a>, BE::ZnxWord>,
}

impl<'a, BE: Backend + 'a> GGSWCompressedBackendMut<'a, BE> {
    pub fn from_inner(inner: GGSWCompressed<BE::BufMut<'a>, BE::ZnxWord>) -> Self {
        Self { inner }
    }

    pub fn into_inner(self) -> GGSWCompressed<BE::BufMut<'a>, BE::ZnxWord> {
        self.inner
    }

    pub fn at_view_mut(&mut self, row: usize, col: usize) -> GLWECompressedViewMut<'_, BE> {
        GLWECompressedViewMut::from_inner(ggsw_compressed_at_backend_mut_from_mut::<BE>(&mut self.inner, row, col))
    }
}

impl<'a, BE: Backend + 'a> Deref for GGSWCompressedBackendMut<'a, BE> {
    type Target = GGSWCompressed<BE::BufMut<'a>, BE::ZnxWord>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<'a, BE: Backend + 'a> DerefMut for GGSWCompressedBackendMut<'a, BE> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

impl<'a, BE: Backend + 'a> LWEInfos for GGSWCompressedBackendRef<'a, BE> {
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

impl<'a, BE: Backend + 'a> GLWEInfos for GGSWCompressedBackendRef<'a, BE> {
    fn rank(&self) -> Rank {
        self.inner.rank()
    }
}

impl<'a, BE: Backend + 'a> GGSWInfos for GGSWCompressedBackendRef<'a, BE> {
    fn k_aux(&self) -> TorusPrecision {
        self.inner.k_aux()
    }

    fn dnum(&self) -> Dnum {
        self.inner.dnum()
    }

    fn dsize(&self) -> Dsize {
        self.inner.dsize()
    }
}

impl<'a, BE: Backend + 'a> LWEInfos for GGSWCompressedBackendMut<'a, BE> {
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

impl<'a, BE: Backend + 'a> GLWEInfos for GGSWCompressedBackendMut<'a, BE> {
    fn rank(&self) -> Rank {
        self.inner.rank()
    }
}

impl<'a, BE: Backend + 'a> GGSWInfos for GGSWCompressedBackendMut<'a, BE> {
    fn k_aux(&self) -> TorusPrecision {
        self.inner.k_aux()
    }

    fn dnum(&self) -> Dnum {
        self.inner.dnum()
    }

    fn dsize(&self) -> Dsize {
        self.inner.dsize()
    }
}

impl<'a, BE: Backend + 'a> GGSWCompressedSeedMut for GGSWCompressedBackendMut<'a, BE> {
    fn seed_mut(&mut self) -> &mut Vec<[u8; 32]> {
        &mut self.inner.seed
    }
}

/// Provides mutable access to the PRNG seeds of a compressed GGSW.
pub trait GGSWCompressedSeedMut {
    /// Returns a mutable reference to the vector of 32-byte PRNG seeds.
    fn seed_mut(&mut self) -> &mut Vec<[u8; 32]>;
}

impl<D: Data, W: ZnxWord> GGSWCompressedSeedMut for GGSWCompressed<D, W> {
    fn seed_mut(&mut self) -> &mut Vec<[u8; 32]> {
        &mut self.seed
    }
}

/// Provides read access to the PRNG seeds of a compressed GGSW.
pub trait GGSWCompressedSeed {
    /// Returns a reference to the vector of 32-byte PRNG seeds.
    fn seed(&self) -> &Vec<[u8; 32]>;
}

impl<D: HostDataRef, W: ZnxWord> GGSWCompressedSeed for GGSWCompressed<D, W> {
    fn seed(&self) -> &Vec<[u8; 32]> {
        &self.seed
    }
}

impl<D: Data, W: ZnxWord> LWEInfos for GGSWCompressed<D, W> {
    fn n(&self) -> Degree {
        Degree(self.data.n() as u32)
    }

    fn base2k(&self) -> Base2K {
        self.base2k
    }

    fn max_size(&self) -> usize {
        crate::layouts::key_size(self.base2k, self.dnum(), self.dsize, self.k_aux)
    }

    fn k(&self) -> TorusPrecision {
        crate::layouts::key_k(self.base2k, self.dnum(), self.dsize, self.k_aux)
    }
}
impl<D: Data, W: ZnxWord> GLWEInfos for GGSWCompressed<D, W> {
    fn rank(&self) -> Rank {
        self.rank
    }
}

impl<D: Data, W: ZnxWord> GGSWInfos for GGSWCompressed<D, W> {
    fn k_aux(&self) -> TorusPrecision {
        self.k_aux
    }

    fn dsize(&self) -> Dsize {
        self.dsize
    }

    fn dnum(&self) -> Dnum {
        Dnum(self.data.rows() as u32)
    }
}

impl<D: HostDataRef, W: ZnxWord> fmt::Debug for GGSWCompressed<D, W> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.data)
    }
}

impl<D: HostDataRef, W: ZnxWord> fmt::Display for GGSWCompressed<D, W> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "(GGSWCompressed: base2k={} k={} dsize={}) {}",
            self.base2k,
            self.k(),
            self.dsize,
            self.data
        )
    }
}

impl<D: HostDataMut, W: ZnxWord> FillUniform for GGSWCompressed<D, W> {
    fn fill_uniform(&mut self, log_bound: usize, source: &mut Source) {
        self.data.fill_uniform(log_bound, source);
    }
}

impl<D: Data, W: ZnxWord> GGSWCompressed<D, W> {
    /// Allocates a new compressed GGSW by copying parameters from an existing info provider.
    pub(crate) fn alloc_from_infos<B: Backend<OwnedBuf = D, ZnxWord = W>, A>(infos: &A) -> Self
    where
        A: GGSWInfos,
    {
        Self::alloc::<B>(
            infos.n(),
            infos.base2k(),
            infos.dnum(),
            infos.dsize(),
            infos.k_aux(),
            infos.rank(),
        )
    }

    /// Allocates a new compressed GGSW with the given parameters.
    pub(crate) fn alloc<B: Backend<OwnedBuf = D, ZnxWord = W>>(
        n: Degree,
        base2k: Base2K,
        dnum: Dnum,
        dsize: Dsize,
        k_aux: TorusPrecision,
        rank: Rank,
    ) -> Self {
        let size: usize = crate::layouts::key_size(base2k, dnum, dsize, k_aux);

        GGSWCompressed {
            data: MatZnx::from_data(
                B::alloc_zeroed_bytes(B::bytes_of_mat_znx(n.into(), dnum.into(), (rank + 1).into(), 1, size)),
                n.into(),
                dnum.into(),
                (rank + 1).into(),
                1,
                size,
            ),
            k_aux,
            base2k,
            dsize,
            rank,
            seed: vec![[0u8; 32]; dnum.as_usize() * (rank.as_usize() + 1)],
        }
    }

    /// Returns the serialized byte size by copying parameters from an existing info provider.
    pub fn bytes_of_from_infos<A>(infos: &A) -> usize
    where
        A: GGSWInfos,
    {
        Self::bytes_of(
            infos.n(),
            infos.base2k(),
            infos.dnum(),
            infos.dsize(),
            infos.k_aux(),
            infos.rank(),
        )
    }

    /// Returns the serialized byte size for a compressed GGSW with the given parameters.
    pub fn bytes_of(n: Degree, base2k: Base2K, dnum: Dnum, dsize: Dsize, k_aux: TorusPrecision, rank: Rank) -> usize {
        let size: usize = crate::layouts::key_size(base2k, dnum, dsize, k_aux);

        MatZnx::<Vec<u8>, W>::bytes_of(n.into(), dnum.into(), (rank + 1).into(), 1, size)
    }
}

impl<D: HostDataRef, W: ZnxWord> GGSWCompressed<D, W> {
    /// Returns an immutably-borrowed compressed GLWE at the given row and column.
    pub fn at(&self, row: usize, col: usize) -> GLWECompressed<&[u8], W> {
        let rank: usize = self.rank().into();
        GLWECompressed {
            data: self.data.at(row, col),
            k: self.k(),
            base2k: self.base2k,
            rank: self.rank,
            seed: self.seed[row * (rank + 1) + col],
        }
    }
}

impl<D: HostDataMut, W: ZnxWord> GGSWCompressed<D, W> {
    /// Returns a mutably-borrowed compressed GLWE at the given row and column.
    pub fn at_mut(&mut self, row: usize, col: usize) -> GLWECompressed<&mut [u8], W> {
        let rank: usize = self.rank().into();
        let k = self.k();
        let seed = self.seed[row * (rank + 1) + col];
        GLWECompressed {
            data: self.data.at_mut(row, col),
            k,
            base2k: self.base2k,
            rank: self.rank,
            seed,
        }
    }
}

impl<D: HostDataMut, W: ZnxWord> ReaderFrom for GGSWCompressed<D, W> {
    fn read_from<R: std::io::Read>(&mut self, reader: &mut R) -> std::io::Result<()> {
        self.k_aux = TorusPrecision(reader.read_u32::<LittleEndian>()?);
        self.base2k = Base2K(reader.read_u32::<LittleEndian>()?);
        self.dsize = Dsize(reader.read_u32::<LittleEndian>()?);
        self.rank = Rank(reader.read_u32::<LittleEndian>()?);
        let seed_len: usize = reader.read_u32::<LittleEndian>()? as usize;
        self.seed = vec![[0u8; 32]; seed_len];
        for s in &mut self.seed {
            reader.read_exact(s)?;
        }
        self.data.read_from(reader)
    }
}

impl<D: HostDataRef, W: ZnxWord> WriterTo for GGSWCompressed<D, W> {
    fn write_to<Wr: std::io::Write>(&self, writer: &mut Wr) -> std::io::Result<()> {
        writer.write_u32::<LittleEndian>(self.k_aux.into())?;
        writer.write_u32::<LittleEndian>(self.base2k.into())?;
        writer.write_u32::<LittleEndian>(self.dsize.into())?;
        writer.write_u32::<LittleEndian>(self.rank.into())?;
        writer.write_u32::<LittleEndian>(self.seed.len() as u32)?;
        for s in &self.seed {
            writer.write_all(s)?;
        }
        self.data.write_to(writer)
    }
}

/// Trait for decompressing a [`GGSWCompressed`] into a standard [`GGSW`].
///
/// Iterates over every (row, column) entry, decompressing each
/// compressed GLWE individually via [`GLWEDecompress`].
pub trait GGSWDecompress
where
    Self: GLWEDecompress,
{
    /// Decompresses `other` into `res`.
    fn decompress_ggsw<R, O>(&self, res: &mut R, other: &O)
    where
        R: GGSWToBackendMut<Self::Backend> + GGSWInfos,
        O: GGSWCompressedToBackendRef<Self::Backend> + GGSWInfos,
    {
        let mut res = res.to_backend_mut();
        let other = other.to_backend_ref();

        assert_eq!(res.rank(), other.rank());
        let dnum: usize = res.dnum().into();
        let rank: usize = res.rank().into();

        for row_i in 0..dnum {
            for col_j in 0..rank + 1 {
                let mut dst = res.at_view_mut(row_i, col_j);
                let src = other.at_view(row_i, col_j);
                self.decompress_glwe(&mut dst, &src);
            }
        }
    }
}

impl<B: Backend> GGSWDecompress for Module<B> where Self: GLWEDecompress {}

// module-only API: decompression is provided by `GGSWDecompress` on `Module`.

pub trait GGSWCompressedToBackendRef<BE: Backend> {
    fn to_backend_ref(&self) -> GGSWCompressedBackendRef<'_, BE>;
}

impl<BE: Backend> GGSWCompressedToBackendRef<BE> for GGSWCompressed<BE::OwnedBuf, BE::ZnxWord> {
    fn to_backend_ref(&self) -> GGSWCompressedBackendRef<'_, BE> {
        GGSWCompressedBackendRef::from_inner(GGSWCompressed {
            k_aux: self.k_aux(),
            base2k: self.base2k(),
            dsize: self.dsize(),
            rank: self.rank(),
            seed: self.seed.clone(),
            data: <MatZnx<BE::OwnedBuf, BE::ZnxWord> as MatZnxToBackendRef<BE>>::to_backend_ref(&self.data),
        })
    }
}

impl<'b, BE: Backend + 'b> GGSWCompressedToBackendRef<BE> for &GGSWCompressed<BE::BufRef<'b>, BE::ZnxWord> {
    fn to_backend_ref(&self) -> GGSWCompressedBackendRef<'_, BE> {
        GGSWCompressedBackendRef::from_inner(GGSWCompressed {
            k_aux: self.k_aux(),
            base2k: self.base2k(),
            dsize: self.dsize(),
            rank: self.rank(),
            seed: self.seed.clone(),
            data: poulpy_hal::layouts::mat_znx_backend_ref_from_ref::<BE>(&self.data),
        })
    }
}

impl<'b, BE: Backend + 'b> GGSWCompressedToBackendRef<BE> for &mut GGSWCompressed<BE::BufMut<'b>, BE::ZnxWord> {
    fn to_backend_ref(&self) -> GGSWCompressedBackendRef<'_, BE> {
        GGSWCompressedBackendRef::from_inner(GGSWCompressed {
            k_aux: self.k_aux(),
            base2k: self.base2k(),
            dsize: self.dsize(),
            rank: self.rank(),
            seed: self.seed.clone(),
            data: mat_znx_backend_ref_from_mut::<BE>(&self.data),
        })
    }
}

pub trait GGSWCompressedToBackendMut<BE: Backend>: GGSWCompressedToBackendRef<BE> {
    fn to_backend_mut(&mut self) -> GGSWCompressedBackendMut<'_, BE>;
}

impl<BE: Backend> GGSWCompressedToBackendMut<BE> for GGSWCompressed<BE::OwnedBuf, BE::ZnxWord> {
    fn to_backend_mut(&mut self) -> GGSWCompressedBackendMut<'_, BE> {
        GGSWCompressedBackendMut::from_inner(GGSWCompressed {
            k_aux: self.k_aux(),
            base2k: self.base2k(),
            dsize: self.dsize(),
            rank: self.rank(),
            seed: self.seed.clone(),
            data: <MatZnx<BE::OwnedBuf, BE::ZnxWord> as MatZnxToBackendMut<BE>>::to_backend_mut(&mut self.data),
        })
    }
}

impl<'b, BE: Backend + 'b> GGSWCompressedToBackendMut<BE> for &mut GGSWCompressed<BE::BufMut<'b>, BE::ZnxWord> {
    fn to_backend_mut(&mut self) -> GGSWCompressedBackendMut<'_, BE> {
        GGSWCompressedBackendMut::from_inner(GGSWCompressed {
            k_aux: self.k_aux(),
            base2k: self.base2k(),
            dsize: self.dsize(),
            rank: self.rank(),
            seed: self.seed.clone(),
            data: mat_znx_backend_mut_from_mut::<BE>(&mut self.data),
        })
    }
}

fn ggsw_compressed_at_backend_mut_from_mut<'a, 'b, BE: Backend>(
    ggsw: &'a mut GGSWCompressed<BE::BufMut<'b>, BE::ZnxWord>,
    row: usize,
    col: usize,
) -> GLWECompressedBackendMut<'a, BE> {
    let rank: usize = ggsw.rank().into();
    let k = ggsw.k();
    let seed = ggsw.seed[row * (rank + 1) + col];
    let base2k = ggsw.base2k;
    let rank_field = ggsw.rank;
    GLWECompressed {
        data: mat_znx_at_backend_mut_from_mut::<BE>(&mut ggsw.data, row, col),
        k,
        base2k,
        rank: rank_field,
        seed,
    }
}

fn ggsw_compressed_at_backend_ref_from_ref<'a, 'b, BE: Backend>(
    ggsw: &'a GGSWCompressed<BE::BufRef<'b>, BE::ZnxWord>,
    row: usize,
    col: usize,
) -> GLWECompressedBackendRef<'a, BE> {
    let rank: usize = ggsw.rank().into();
    GLWECompressed {
        data: mat_znx_at_backend_ref_from_ref::<BE>(&ggsw.data, row, col),
        k: ggsw.k(),
        base2k: ggsw.base2k,
        rank: ggsw.rank,
        seed: ggsw.seed[row * (rank + 1) + col],
    }
}
