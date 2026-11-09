use poulpy_hal::AlignedBuf;
use poulpy_hal::layouts::{Backend, Data, HostDataMut, HostDataRef, ReaderFrom, Ring, Standard, WriterTo};

use crate::layouts::{
    Base2K, Degree, Dnum, GGLWECompressed, GGLWEInfos, GLWEInfos, LWEInfos, Rank, TorusPrecision,
    compressed::{
        GGLWECompressedBackendMut, GGLWECompressedBackendRef, GGLWECompressedToBackendMut, GGLWECompressedToBackendRef,
    },
};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use poulpy_hal::layouts::ZnxWord;

use std::{
    fmt,
    ops::{Deref, DerefMut},
};

/// Seed-compressed GGLWE-to-GGSW key-switching key layout.
///
/// A vector of [`GGLWECompressed`] entries, one per rank element,
/// used for GGLWE-to-GGSW conversion. The mask of each GGLWE is
/// regenerated from its PRNG seed during decompression.
#[derive(PartialEq, Eq, Clone)]
pub struct GGLWEToGGSWKeyCompressed<D: Data, W: ZnxWord, Rg: Ring = Standard> {
    pub(crate) keys: Vec<GGLWECompressed<D, W, Rg>>,
}

pub struct GGLWEToGGSWKeyCompressedBackendRef<'a, BE: Backend + 'a, Rg: Ring = Standard> {
    inner: GGLWEToGGSWKeyCompressed<BE::BufRef<'a>, BE::ZnxWord, Rg>,
}

impl<'a, BE: Backend + 'a, Rg: Ring> GGLWEToGGSWKeyCompressedBackendRef<'a, BE, Rg> {
    pub fn from_inner(inner: GGLWEToGGSWKeyCompressed<BE::BufRef<'a>, BE::ZnxWord, Rg>) -> Self {
        Self { inner }
    }

    pub fn into_inner(self) -> GGLWEToGGSWKeyCompressed<BE::BufRef<'a>, BE::ZnxWord, Rg> {
        self.inner
    }

    pub fn at_view(&self, i: usize) -> GGLWECompressedBackendRef<'_, BE, Rg> {
        assert!((i as u32) < self.rank());
        let key_i = &self.inner.keys[i];
        GGLWECompressedBackendRef::from_inner(GGLWECompressed {
            noise: key_i.noise.clone(),
            k_aux: key_i.k_aux,
            base2k: key_i.base2k,
            k_digit: key_i.k_digit,
            k_offset: key_i.k_offset,
            seed: key_i.seed.clone(),
            rank_out: key_i.rank_out,
            data: poulpy_hal::layouts::mat_znx_backend_ref_from_ref::<BE, Rg>(&key_i.data),
        })
    }
}

impl<'a, BE: Backend + 'a, Rg: Ring> Deref for GGLWEToGGSWKeyCompressedBackendRef<'a, BE, Rg> {
    type Target = GGLWEToGGSWKeyCompressed<BE::BufRef<'a>, BE::ZnxWord, Rg>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

pub struct GGLWEToGGSWKeyCompressedBackendMut<'a, BE: Backend + 'a, Rg: Ring = Standard> {
    inner: GGLWEToGGSWKeyCompressed<BE::BufMut<'a>, BE::ZnxWord, Rg>,
}

impl<'a, BE: Backend + 'a, Rg: Ring> GGLWEToGGSWKeyCompressedBackendMut<'a, BE, Rg> {
    pub fn from_inner(inner: GGLWEToGGSWKeyCompressed<BE::BufMut<'a>, BE::ZnxWord, Rg>) -> Self {
        Self { inner }
    }

    pub fn into_inner(self) -> GGLWEToGGSWKeyCompressed<BE::BufMut<'a>, BE::ZnxWord, Rg> {
        self.inner
    }

    pub fn at_view(&self, i: usize) -> GGLWECompressedBackendRef<'_, BE, Rg> {
        assert!((i as u32) < self.rank());
        let key_i = &self.inner.keys[i];
        GGLWECompressedBackendRef::from_inner(GGLWECompressed {
            noise: key_i.noise.clone(),
            k_aux: key_i.k_aux,
            base2k: key_i.base2k,
            k_digit: key_i.k_digit,
            k_offset: key_i.k_offset,
            seed: key_i.seed.clone(),
            rank_out: key_i.rank_out,
            data: poulpy_hal::layouts::mat_znx_backend_ref_from_mut::<BE, Rg>(&key_i.data),
        })
    }

    pub fn at_view_mut(&mut self, i: usize) -> GGLWECompressedBackendMut<'_, BE, Rg> {
        assert!((i as u32) < self.rank());
        let key_i = &mut self.inner.keys[i];
        key_i.noise = None;
        GGLWECompressedBackendMut::from_inner(GGLWECompressed {
            noise: None,
            k_aux: key_i.k_aux,
            base2k: key_i.base2k,
            k_digit: key_i.k_digit,
            k_offset: key_i.k_offset,
            seed: key_i.seed.clone(),
            rank_out: key_i.rank_out,
            data: poulpy_hal::layouts::mat_znx_backend_mut_from_mut::<BE, Rg>(&mut key_i.data),
        })
    }
}

impl<'a, BE: Backend + 'a, Rg: Ring> Deref for GGLWEToGGSWKeyCompressedBackendMut<'a, BE, Rg> {
    type Target = GGLWEToGGSWKeyCompressed<BE::BufMut<'a>, BE::ZnxWord, Rg>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<BE: Backend, Rg: Ring> DerefMut for GGLWEToGGSWKeyCompressedBackendMut<'_, BE, Rg> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

impl_gglwe_infos_for_inner!(GGLWEToGGSWKeyCompressedBackendRef<'a, BE, Rg>, ['a, BE: Backend + 'a, Rg: Ring]; inner);
impl_gglwe_infos_for_inner!(GGLWEToGGSWKeyCompressedBackendMut<'a, BE, Rg>, ['a, BE: Backend + 'a, Rg: Ring]; inner);

impl<D: Data, W: ZnxWord, Rg: Ring> LWEInfos for GGLWEToGGSWKeyCompressed<D, W, Rg> {
    fn noise(&self) -> Option<crate::ComponentNoise> {
        crate::layouts::common_noise(self.keys.iter().map(crate::layouts::LWEInfos::noise))
    }

    fn n(&self) -> Degree {
        self.keys[0].n()
    }

    fn base2k(&self) -> Base2K {
        self.keys[0].base2k()
    }

    fn max_size(&self) -> usize {
        self.keys[0].max_size()
    }

    fn k(&self) -> TorusPrecision {
        self.keys[0].k()
    }
}

impl<D: Data, W: ZnxWord, Rg: Ring> GLWEInfos for GGLWEToGGSWKeyCompressed<D, W, Rg> {
    fn rank(&self) -> Rank {
        self.keys[0].rank_out()
    }
}

impl<D: Data, W: ZnxWord, Rg: Ring> GGLWEInfos for GGLWEToGGSWKeyCompressed<D, W, Rg> {
    fn k_aux(&self) -> TorusPrecision {
        self.keys[0].k_aux()
    }

    fn rank_in(&self) -> Rank {
        self.rank_out()
    }

    fn rank_out(&self) -> Rank {
        self.keys[0].rank_out()
    }

    fn k_digit(&self) -> TorusPrecision {
        self.keys[0].k_digit()
    }

    fn k_offset(&self) -> TorusPrecision {
        self.keys[0].k_offset()
    }

    fn dnum(&self) -> Dnum {
        self.keys[0].dnum()
    }
}

impl<D: HostDataRef, W: ZnxWord, Rg: Ring> fmt::Debug for GGLWEToGGSWKeyCompressed<D, W, Rg> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{self}")
    }
}

impl<D: HostDataRef, W: ZnxWord, Rg: Ring> fmt::Display for GGLWEToGGSWKeyCompressed<D, W, Rg> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "(GGLWEToGGSWKeyCompressed)",)?;
        for (i, key) in self.keys.iter().enumerate() {
            write!(f, "{i}: {key}")?;
        }
        Ok(())
    }
}

impl<D: Data, W: ZnxWord, Rg: Ring> GGLWEToGGSWKeyCompressed<D, W, Rg> {
    /// Allocates a new compressed GGLWE-to-GGSW key by copying parameters from an existing info provider.
    pub(crate) fn alloc_from_infos<B: Backend<OwnedBuf = D, ZnxWord = W>, A>(infos: &A) -> Self
    where
        A: GGLWEInfos,
    {
        assert_eq!(
            infos.rank_in(),
            infos.rank_out(),
            "rank_in != rank_out is not supported for GGLWEToGGSWKeyCompressed"
        );
        Self::alloc::<B>(
            infos.n(),
            infos.base2k(),
            infos.dnum(),
            infos.k_digit(),
            infos.k_aux(),
            infos.rank(),
        )
    }

    /// Allocates a new compressed GGLWE-to-GGSW key with the given parameters.
    pub(crate) fn alloc<B: Backend<OwnedBuf = D, ZnxWord = W>>(
        n: Degree,
        base2k: Base2K,
        dnum: Dnum,
        k_digit: TorusPrecision,
        k_aux: TorusPrecision,
        rank: Rank,
    ) -> Self {
        GGLWEToGGSWKeyCompressed {
            keys: (0..rank.as_usize())
                .map(|_| GGLWECompressed::alloc::<B>(n, base2k, dnum, k_digit, k_aux, rank, rank))
                .collect(),
        }
    }

    /// Returns the serialized byte size by copying parameters from an existing info provider.
    pub fn bytes_of_from_infos<A>(infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        assert_eq!(
            infos.rank_in(),
            infos.rank_out(),
            "rank_in != rank_out is not supported for GGLWEToGGSWKeyCompressed"
        );
        Self::bytes_of(
            infos.n(),
            infos.base2k(),
            infos.dnum(),
            infos.k_digit(),
            infos.k_aux(),
            infos.rank(),
        )
    }

    /// Returns the serialized byte size for a compressed GGLWE-to-GGSW key with the given parameters.
    pub fn bytes_of(n: Degree, base2k: Base2K, dnum: Dnum, k_digit: TorusPrecision, k_aux: TorusPrecision, rank: Rank) -> usize {
        rank.as_usize() * GGLWECompressed::<AlignedBuf, W, Rg>::bytes_of(n, base2k, dnum, k_digit, k_aux, rank)
    }
}

impl<D: Data, W: ZnxWord, Rg: Ring> GGLWEToGGSWKeyCompressed<D, W, Rg> {
    // Returns a mutable reference to GGLWE_{s}([s[i]*s[0], s[i]*s[1], ..., s[i]*s[rank]])
    pub fn at_mut(&mut self, i: usize) -> &mut GGLWECompressed<D, W, Rg> {
        assert!((i as u32) < self.rank());
        &mut self.keys[i]
    }
}

impl<D: Data, W: ZnxWord, Rg: Ring> GGLWEToGGSWKeyCompressed<D, W, Rg> {
    // Returns a reference to GGLWE_{s}(s[i] * s[j])
    pub fn at(&self, i: usize) -> &GGLWECompressed<D, W, Rg> {
        assert!((i as u32) < self.rank());
        &self.keys[i]
    }
}

impl<D: HostDataMut, W: ZnxWord, Rg: Ring> ReaderFrom for GGLWEToGGSWKeyCompressed<D, W, Rg> {
    fn read_from<R: std::io::Read>(&mut self, reader: &mut R) -> std::io::Result<()> {
        let len: usize = reader.read_u64::<LittleEndian>()? as usize;
        if self.keys.len() != len {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("self.keys.len()={} != read len={}", self.keys.len(), len),
            ));
        }
        for key in &mut self.keys {
            key.read_from(reader)?;
        }
        Ok(())
    }
}

impl<D: HostDataRef, W: ZnxWord, Rg: Ring> WriterTo for GGLWEToGGSWKeyCompressed<D, W, Rg> {
    fn write_to<Wr: std::io::Write>(&self, writer: &mut Wr) -> std::io::Result<()> {
        writer.write_u64::<LittleEndian>(self.keys.len() as u64)?;
        for key in &self.keys {
            key.write_to(writer)?;
        }
        Ok(())
    }
}

pub trait GGLWEToGGSWKeyCompressedToBackendRef<BE: Backend, Rg: Ring = Standard> {
    fn to_backend_ref(&self) -> GGLWEToGGSWKeyCompressedBackendRef<'_, BE, Rg>;
}

impl<BE: Backend, Rg: Ring> GGLWEToGGSWKeyCompressedToBackendRef<BE, Rg>
    for GGLWEToGGSWKeyCompressed<BE::OwnedBuf, BE::ZnxWord, Rg>
{
    fn to_backend_ref(&self) -> GGLWEToGGSWKeyCompressedBackendRef<'_, BE, Rg> {
        GGLWEToGGSWKeyCompressedBackendRef::from_inner(GGLWEToGGSWKeyCompressed {
            keys: self
                .keys
                .iter()
                .map(|key| GGLWECompressedToBackendRef::<BE, Rg>::to_backend_ref(key).into_inner())
                .collect(),
        })
    }
}

pub trait GGLWEToGGSWKeyCompressedToBackendMut<BE: Backend, Rg: Ring = Standard>:
    GGLWEToGGSWKeyCompressedToBackendRef<BE, Rg>
{
    /// Backend hook for recording or propagating component noise metadata.
    fn set_noise(&mut self, metadata: Option<crate::ComponentNoise>);

    /// Borrows coefficients mutably and clears the owner's component noise metadata.
    fn to_backend_mut(&mut self) -> GGLWEToGGSWKeyCompressedBackendMut<'_, BE, Rg>;
}

impl<BE: Backend, Rg: Ring> GGLWEToGGSWKeyCompressedToBackendMut<BE, Rg>
    for GGLWEToGGSWKeyCompressed<BE::OwnedBuf, BE::ZnxWord, Rg>
{
    fn set_noise(&mut self, metadata: Option<crate::ComponentNoise>) {
        for key in &mut self.keys {
            key.noise = crate::layouts::checked_noise(metadata.clone(), crate::layouts::GLWEInfos::rank(key).as_usize() + 1);
        }
    }

    fn to_backend_mut(&mut self) -> GGLWEToGGSWKeyCompressedBackendMut<'_, BE, Rg> {
        GGLWEToGGSWKeyCompressedBackendMut::from_inner(GGLWEToGGSWKeyCompressed {
            keys: self
                .keys
                .iter_mut()
                .map(|key| GGLWECompressedToBackendMut::<BE, Rg>::to_backend_mut(key).into_inner())
                .collect(),
        })
    }
}
