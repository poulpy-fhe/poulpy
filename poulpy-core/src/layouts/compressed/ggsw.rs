use poulpy_hal::AlignedBuf;
use poulpy_hal::layouts::{
    Backend, Data, HostDataMut, HostDataRef, MatZnx, MatZnxToBackendMut, MatZnxToBackendRef, ReaderFrom, Ring, Standard,
    WriterTo, mat_znx_at_backend_mut_from_mut, mat_znx_at_backend_ref_from_ref, mat_znx_backend_mut_from_mut,
    mat_znx_backend_ref_from_mut,
};

use crate::layouts::{
    Base2K, Degree, Dnum, GGSWInfos, GLWEInfos, LWEInfos, Rank, TorusPrecision,
    compressed::{
        GLWECompressed, GLWECompressedBackendMut, GLWECompressedBackendRef, GLWECompressedViewMut, GLWECompressedViewRef,
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
/// Stores only the body components of a [`GGSW`](crate::layouts::GGSW) ciphertext; the mask
/// polynomials are regenerated deterministically from 32-byte PRNG
/// seeds during decompression.
#[derive(PartialEq, Eq, Clone)]
pub struct GGSWCompressed<D: Data, W: ZnxWord, Rg: Ring = Standard> {
    pub(crate) noise: Option<crate::ComponentNoise>,
    pub(crate) data: MatZnx<D, W, Rg>,
    pub(crate) k_aux: TorusPrecision,
    pub(crate) base2k: Base2K,
    pub(crate) k_digit: TorusPrecision,
    pub(crate) rank: Rank,
    pub(crate) seed: Vec<[u8; 32]>,
}

pub struct GGSWCompressedBackendRef<'a, BE: Backend + 'a, Rg: Ring = Standard> {
    inner: GGSWCompressed<BE::BufRef<'a>, BE::ZnxWord, Rg>,
}

impl<'a, BE: Backend + 'a, Rg: Ring> GGSWCompressedBackendRef<'a, BE, Rg> {
    pub fn from_inner(inner: GGSWCompressed<BE::BufRef<'a>, BE::ZnxWord, Rg>) -> Self {
        Self { inner }
    }

    pub fn into_inner(self) -> GGSWCompressed<BE::BufRef<'a>, BE::ZnxWord, Rg> {
        self.inner
    }

    pub fn at_view(&self, row: usize, col: usize) -> GLWECompressedViewRef<'_, BE, Rg> {
        GLWECompressedViewRef::from_inner(ggsw_compressed_at_backend_ref_from_ref::<BE, Rg>(&self.inner, row, col))
    }
}

impl<'a, BE: Backend + 'a, Rg: Ring> Deref for GGSWCompressedBackendRef<'a, BE, Rg> {
    type Target = GGSWCompressed<BE::BufRef<'a>, BE::ZnxWord, Rg>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

pub struct GGSWCompressedBackendMut<'a, BE: Backend + 'a, Rg: Ring = Standard> {
    inner: GGSWCompressed<BE::BufMut<'a>, BE::ZnxWord, Rg>,
}

impl<'a, BE: Backend + 'a, Rg: Ring> GGSWCompressedBackendMut<'a, BE, Rg> {
    pub fn from_inner(inner: GGSWCompressed<BE::BufMut<'a>, BE::ZnxWord, Rg>) -> Self {
        Self { inner }
    }

    pub fn into_inner(self) -> GGSWCompressed<BE::BufMut<'a>, BE::ZnxWord, Rg> {
        self.inner
    }

    pub fn at_view_mut(&mut self, row: usize, col: usize) -> GLWECompressedViewMut<'_, BE, Rg> {
        GLWECompressedViewMut::from_inner(ggsw_compressed_at_backend_mut_from_mut::<BE, Rg>(&mut self.inner, row, col))
    }
}

impl<'a, BE: Backend + 'a, Rg: Ring> Deref for GGSWCompressedBackendMut<'a, BE, Rg> {
    type Target = GGSWCompressed<BE::BufMut<'a>, BE::ZnxWord, Rg>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<BE: Backend, Rg: Ring> DerefMut for GGSWCompressedBackendMut<'_, BE, Rg> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

impl<BE: Backend, Rg: Ring> LWEInfos for GGSWCompressedBackendRef<'_, BE, Rg> {
    fn noise(&self) -> Option<crate::ComponentNoise> {
        self.inner.noise()
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

impl<BE: Backend, Rg: Ring> GLWEInfos for GGSWCompressedBackendRef<'_, BE, Rg> {
    fn rank(&self) -> Rank {
        self.inner.rank()
    }
}

impl<BE: Backend, Rg: Ring> GGSWInfos for GGSWCompressedBackendRef<'_, BE, Rg> {
    fn k_aux(&self) -> TorusPrecision {
        self.inner.k_aux()
    }

    fn dnum(&self) -> Dnum {
        self.inner.dnum()
    }

    fn k_digit(&self) -> TorusPrecision {
        self.inner.k_digit()
    }
}

impl<BE: Backend, Rg: Ring> LWEInfos for GGSWCompressedBackendMut<'_, BE, Rg> {
    fn noise(&self) -> Option<crate::ComponentNoise> {
        self.inner.noise()
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

impl<BE: Backend, Rg: Ring> GLWEInfos for GGSWCompressedBackendMut<'_, BE, Rg> {
    fn rank(&self) -> Rank {
        self.inner.rank()
    }
}

impl<BE: Backend, Rg: Ring> GGSWInfos for GGSWCompressedBackendMut<'_, BE, Rg> {
    fn k_aux(&self) -> TorusPrecision {
        self.inner.k_aux()
    }

    fn dnum(&self) -> Dnum {
        self.inner.dnum()
    }

    fn k_digit(&self) -> TorusPrecision {
        self.inner.k_digit()
    }
}

impl<BE: Backend, Rg: Ring> GGSWCompressedSeedMut for GGSWCompressedBackendMut<'_, BE, Rg> {
    fn seed_mut(&mut self) -> &mut Vec<[u8; 32]> {
        &mut self.inner.seed
    }
}

/// Provides mutable access to the PRNG seeds of a compressed GGSW.
pub trait GGSWCompressedSeedMut {
    /// Returns a mutable reference to the vector of 32-byte PRNG seeds.
    fn seed_mut(&mut self) -> &mut Vec<[u8; 32]>;
}

impl<D: Data, W: ZnxWord, Rg: Ring> GGSWCompressedSeedMut for GGSWCompressed<D, W, Rg> {
    fn seed_mut(&mut self) -> &mut Vec<[u8; 32]> {
        &mut self.seed
    }
}

/// Provides read access to the PRNG seeds of a compressed GGSW.
pub trait GGSWCompressedSeed {
    /// Returns a reference to the vector of 32-byte PRNG seeds.
    fn seed(&self) -> &Vec<[u8; 32]>;
}

impl<D: Data, W: ZnxWord, Rg: Ring> GGSWCompressedSeed for GGSWCompressed<D, W, Rg> {
    fn seed(&self) -> &Vec<[u8; 32]> {
        &self.seed
    }
}

impl<D: Data, W: ZnxWord, Rg: Ring> LWEInfos for GGSWCompressed<D, W, Rg> {
    fn noise(&self) -> Option<crate::ComponentNoise> {
        self.noise.clone()
    }

    fn n(&self) -> Degree {
        Degree(self.data.n() as u32)
    }

    fn base2k(&self) -> Base2K {
        self.base2k
    }

    fn max_size(&self) -> usize {
        crate::layouts::key_size(self.base2k, self.dnum(), self.k_digit, self.k_aux)
    }

    fn k(&self) -> TorusPrecision {
        crate::layouts::key_k(self.dnum(), self.k_digit, self.k_aux)
    }
}
impl<D: Data, W: ZnxWord, Rg: Ring> GLWEInfos for GGSWCompressed<D, W, Rg> {
    fn rank(&self) -> Rank {
        self.rank
    }
}

impl<D: Data, W: ZnxWord, Rg: Ring> GGSWInfos for GGSWCompressed<D, W, Rg> {
    fn k_aux(&self) -> TorusPrecision {
        self.k_aux
    }

    fn k_digit(&self) -> TorusPrecision {
        self.k_digit
    }

    fn dnum(&self) -> Dnum {
        Dnum(self.data.rows() as u32)
    }
}

impl<D: HostDataRef, W: ZnxWord, Rg: Ring> fmt::Debug for GGSWCompressed<D, W, Rg> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.data)
    }
}

impl<D: HostDataRef, W: ZnxWord, Rg: Ring> fmt::Display for GGSWCompressed<D, W, Rg> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "(GGSWCompressed: base2k={} k={} k_digit={}) {}",
            self.base2k,
            self.k(),
            self.k_digit,
            self.data
        )
    }
}

impl<D: Data, W: ZnxWord, Rg: Ring> GGSWCompressed<D, W, Rg> {
    /// Allocates a new compressed GGSW by copying parameters from an existing info provider.
    pub(crate) fn alloc_from_infos<B: Backend<OwnedBuf = D, ZnxWord = W>, A>(infos: &A) -> Self
    where
        A: GGSWInfos,
    {
        Self::alloc::<B>(
            infos.n(),
            infos.base2k(),
            infos.dnum(),
            infos.k_digit(),
            infos.k_aux(),
            infos.rank(),
        )
    }

    /// Allocates a new compressed GGSW with the given parameters.
    pub(crate) fn alloc<B: Backend<OwnedBuf = D, ZnxWord = W>>(
        n: Degree,
        base2k: Base2K,
        dnum: Dnum,
        k_digit: TorusPrecision,
        k_aux: TorusPrecision,
        rank: Rank,
    ) -> Self {
        let size: usize = crate::layouts::key_size(base2k, dnum, k_digit, k_aux);

        GGSWCompressed {
            noise: None,
            data: MatZnx::from_data(
                B::alloc_zeroed_bytes(B::mat_znx_bytes_of(n.into(), dnum.into(), (rank + 1).into(), 1, size)),
                n.into(),
                dnum.into(),
                (rank + 1).into(),
                1,
                size,
            ),
            k_aux,
            base2k,
            k_digit,
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
            infos.k_digit(),
            infos.k_aux(),
            infos.rank(),
        )
    }

    /// Returns the serialized byte size for a compressed GGSW with the given parameters.
    pub fn bytes_of(n: Degree, base2k: Base2K, dnum: Dnum, k_digit: TorusPrecision, k_aux: TorusPrecision, rank: Rank) -> usize {
        let size: usize = crate::layouts::key_size(base2k, dnum, k_digit, k_aux);

        MatZnx::<AlignedBuf, W, Rg>::bytes_of(n.into(), dnum.into(), (rank + 1).into(), 1, size)
    }
}

impl<D: HostDataRef, W: ZnxWord, Rg: Ring> GGSWCompressed<D, W, Rg> {
    /// Returns an immutably-borrowed compressed GLWE at the given row and column.
    pub fn at(&self, row: usize, col: usize) -> GLWECompressed<&[u8], W, Rg> {
        let rank: usize = self.rank().into();
        GLWECompressed {
            noise: crate::layouts::LWEInfos::noise(&self),
            data: self.data.at(row, col),
            k: self.k(),
            base2k: self.base2k,
            rank: self.rank,
            seed: self.seed[row * (rank + 1) + col],
            plaintext_meta: None,
        }
    }
}

impl<D: HostDataMut, W: ZnxWord, Rg: Ring> GGSWCompressed<D, W, Rg> {
    /// Returns a mutably-borrowed compressed GLWE at the given row and column.
    pub fn at_mut(&mut self, row: usize, col: usize) -> GLWECompressed<&mut [u8], W, Rg> {
        let rank: usize = self.rank().into();
        let k = self.k();
        let seed = self.seed[row * (rank + 1) + col];
        self.noise = None;
        GLWECompressed {
            noise: None,
            data: self.data.at_mut(row, col),
            k,
            base2k: self.base2k,
            rank: self.rank,
            seed,
            plaintext_meta: None,
        }
    }
}

impl<D: HostDataMut, W: ZnxWord, Rg: Ring> ReaderFrom for GGSWCompressed<D, W, Rg> {
    fn read_from<R: std::io::Read>(&mut self, reader: &mut R) -> std::io::Result<()> {
        self.noise = None;
        let noise = crate::ComponentNoise::read_optional(reader, self.rank.as_usize() + 1)?;
        let k_aux = TorusPrecision(reader.read_u32::<LittleEndian>()?);
        let base2k = Base2K(reader.read_u32::<LittleEndian>()?);
        let k_digit = TorusPrecision(reader.read_u32::<LittleEndian>()?);
        let rank = Rank(reader.read_u32::<LittleEndian>()?);
        if rank != self.rank {
            return Err(crate::layouts::invalid_serialized_shape());
        }
        let seed_len: usize = reader.read_u32::<LittleEndian>()? as usize;
        if seed_len != self.seed.len() {
            return Err(crate::layouts::invalid_serialized_shape());
        }
        let mut seed = vec![[0u8; 32]; seed_len];
        for s in &mut seed {
            reader.read_exact(s)?;
        }
        let (rows, cols_in) = (self.data.rows(), self.data.cols_in());
        crate::layouts::read_mat_znx_with_shape(&mut self.data, reader, Some(rows), Some(cols_in), 1)?;
        crate::layouts::validate_noise_components(noise.as_ref(), self.rank.as_usize() + 1)?;
        self.k_aux = k_aux;
        self.base2k = base2k;
        self.k_digit = k_digit;
        self.seed = seed;
        self.noise = noise;
        Ok(())
    }
}

impl<D: HostDataRef, W: ZnxWord, Rg: Ring> WriterTo for GGSWCompressed<D, W, Rg> {
    fn write_to<Wr: std::io::Write>(&self, writer: &mut Wr) -> std::io::Result<()> {
        crate::layouts::validate_noise_components(self.noise.as_ref(), self.rank.as_usize() + 1)?;
        crate::ComponentNoise::write_optional(self.noise.as_ref(), writer)?;
        writer.write_u32::<LittleEndian>(self.k_aux.into())?;
        writer.write_u32::<LittleEndian>(self.base2k.into())?;
        writer.write_u32::<LittleEndian>(self.k_digit.into())?;
        writer.write_u32::<LittleEndian>(self.rank.into())?;
        writer.write_u32::<LittleEndian>(self.seed.len() as u32)?;
        for s in &self.seed {
            writer.write_all(s)?;
        }
        self.data.write_to(writer)
    }
}

pub trait GGSWCompressedToBackendRef<BE: Backend, Rg: Ring = Standard> {
    fn to_backend_ref(&self) -> GGSWCompressedBackendRef<'_, BE, Rg>;
}

impl<BE: Backend, Rg: Ring> GGSWCompressedToBackendRef<BE, Rg> for GGSWCompressed<BE::OwnedBuf, BE::ZnxWord, Rg> {
    fn to_backend_ref(&self) -> GGSWCompressedBackendRef<'_, BE, Rg> {
        GGSWCompressedBackendRef::from_inner(GGSWCompressed {
            noise: crate::layouts::LWEInfos::noise(&self),
            k_aux: self.k_aux(),
            base2k: self.base2k(),
            k_digit: self.k_digit(),
            rank: self.rank(),
            seed: self.seed.clone(),
            data: <MatZnx<BE::OwnedBuf, BE::ZnxWord, Rg> as MatZnxToBackendRef<BE, Rg>>::to_backend_ref(&self.data),
        })
    }
}

impl<BE: Backend, Rg: Ring> GGSWCompressedToBackendRef<BE, Rg> for &GGSWCompressed<BE::BufRef<'_>, BE::ZnxWord, Rg> {
    fn to_backend_ref(&self) -> GGSWCompressedBackendRef<'_, BE, Rg> {
        GGSWCompressedBackendRef::from_inner(GGSWCompressed {
            noise: crate::layouts::LWEInfos::noise(&self),
            k_aux: self.k_aux(),
            base2k: self.base2k(),
            k_digit: self.k_digit(),
            rank: self.rank(),
            seed: self.seed.clone(),
            data: poulpy_hal::layouts::mat_znx_backend_ref_from_ref::<BE, Rg>(&self.data),
        })
    }
}

impl<BE: Backend, Rg: Ring> GGSWCompressedToBackendRef<BE, Rg> for &mut GGSWCompressed<BE::BufMut<'_>, BE::ZnxWord, Rg> {
    fn to_backend_ref(&self) -> GGSWCompressedBackendRef<'_, BE, Rg> {
        GGSWCompressedBackendRef::from_inner(GGSWCompressed {
            noise: crate::layouts::LWEInfos::noise(&self),
            k_aux: self.k_aux(),
            base2k: self.base2k(),
            k_digit: self.k_digit(),
            rank: self.rank(),
            seed: self.seed.clone(),
            data: mat_znx_backend_ref_from_mut::<BE, Rg>(&self.data),
        })
    }
}

pub trait GGSWCompressedToBackendMut<BE: Backend, Rg: Ring = Standard>: GGSWCompressedToBackendRef<BE, Rg> {
    /// Backend hook for recording or propagating component noise metadata.
    fn set_noise(&mut self, metadata: Option<crate::ComponentNoise>);

    /// Borrows coefficients mutably and clears the owner's component noise metadata.
    fn to_backend_mut(&mut self) -> GGSWCompressedBackendMut<'_, BE, Rg>;
}

impl<BE: Backend, Rg: Ring> GGSWCompressedToBackendMut<BE, Rg> for GGSWCompressed<BE::OwnedBuf, BE::ZnxWord, Rg> {
    fn set_noise(&mut self, metadata: Option<crate::ComponentNoise>) {
        self.noise = crate::layouts::checked_noise(metadata, crate::layouts::GLWEInfos::rank(self).as_usize() + 1);
    }

    fn to_backend_mut(&mut self) -> GGSWCompressedBackendMut<'_, BE, Rg> {
        self.noise = None;
        GGSWCompressedBackendMut::from_inner(GGSWCompressed {
            noise: None,
            k_aux: self.k_aux(),
            base2k: self.base2k(),
            k_digit: self.k_digit(),
            rank: self.rank(),
            seed: self.seed.clone(),
            data: <MatZnx<BE::OwnedBuf, BE::ZnxWord, Rg> as MatZnxToBackendMut<BE, Rg>>::to_backend_mut(&mut self.data),
        })
    }
}

impl<BE: Backend, Rg: Ring> GGSWCompressedToBackendMut<BE, Rg> for &mut GGSWCompressed<BE::BufMut<'_>, BE::ZnxWord, Rg> {
    fn set_noise(&mut self, metadata: Option<crate::ComponentNoise>) {
        self.noise = crate::layouts::checked_noise(metadata, crate::layouts::GLWEInfos::rank(self).as_usize() + 1);
    }

    fn to_backend_mut(&mut self) -> GGSWCompressedBackendMut<'_, BE, Rg> {
        self.noise = None;
        GGSWCompressedBackendMut::from_inner(GGSWCompressed {
            noise: None,
            k_aux: self.k_aux(),
            base2k: self.base2k(),
            k_digit: self.k_digit(),
            rank: self.rank(),
            seed: self.seed.clone(),
            data: mat_znx_backend_mut_from_mut::<BE, Rg>(&mut self.data),
        })
    }
}

fn ggsw_compressed_at_backend_mut_from_mut<'a, 'b, BE: Backend, Rg: Ring>(
    ggsw: &'a mut GGSWCompressed<BE::BufMut<'b>, BE::ZnxWord, Rg>,
    row: usize,
    col: usize,
) -> GLWECompressedBackendMut<'a, BE, Rg> {
    let rank: usize = ggsw.rank().into();
    let k = ggsw.k();
    let seed = ggsw.seed[row * (rank + 1) + col];
    let base2k = ggsw.base2k;
    let rank_field = ggsw.rank;
    ggsw.noise = None;
    GLWECompressed {
        noise: None,
        data: mat_znx_at_backend_mut_from_mut::<BE, Rg>(&mut ggsw.data, row, col),
        k,
        base2k,
        rank: rank_field,
        seed,
        plaintext_meta: None,
    }
}

fn ggsw_compressed_at_backend_ref_from_ref<'a, 'b, BE: Backend, Rg: Ring>(
    ggsw: &'a GGSWCompressed<BE::BufRef<'b>, BE::ZnxWord, Rg>,
    row: usize,
    col: usize,
) -> GLWECompressedBackendRef<'a, BE, Rg> {
    let rank: usize = ggsw.rank().into();
    GLWECompressed {
        noise: crate::layouts::LWEInfos::noise(&ggsw),
        data: mat_znx_at_backend_ref_from_ref::<BE, Rg>(&ggsw.data, row, col),
        k: ggsw.k(),
        base2k: ggsw.base2k,
        rank: ggsw.rank,
        seed: ggsw.seed[row * (rank + 1) + col],
        plaintext_meta: None,
    }
}
