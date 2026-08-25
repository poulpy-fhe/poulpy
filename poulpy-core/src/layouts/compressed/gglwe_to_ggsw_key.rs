use poulpy_hal::{
    api::{ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Backend, Data, FillUniform, HostDataMut, HostDataRef, ReaderFrom, ScratchOwned, WriterTo},
    source::Source,
};

use crate::layouts::{
    Base2K, Degree, Dnum, Dsize, GGLWECompressed, GGLWEDecompress, GGLWEInfos, GGLWEToGGSWKeyToBackendMut, GLWEInfos, LWEInfos,
    Rank, TorusPrecision,
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
pub struct GGLWEToGGSWKeyCompressed<D: Data, W: ZnxWord> {
    pub(crate) keys: Vec<GGLWECompressed<D, W>>,
}

pub struct GGLWEToGGSWKeyCompressedBackendRef<'a, BE: Backend + 'a> {
    inner: GGLWEToGGSWKeyCompressed<BE::BufRef<'a>, BE::ZnxWord>,
}

impl<'a, BE: Backend + 'a> GGLWEToGGSWKeyCompressedBackendRef<'a, BE> {
    pub fn from_inner(inner: GGLWEToGGSWKeyCompressed<BE::BufRef<'a>, BE::ZnxWord>) -> Self {
        Self { inner }
    }

    pub fn into_inner(self) -> GGLWEToGGSWKeyCompressed<BE::BufRef<'a>, BE::ZnxWord> {
        self.inner
    }

    pub fn at_view(&self, i: usize) -> GGLWECompressedBackendRef<'_, BE> {
        assert!((i as u32) < self.rank());
        let key_i = &self.inner.keys[i];
        GGLWECompressedBackendRef::from_inner(GGLWECompressed {
            k_aux: key_i.k_aux,
            base2k: key_i.base2k,
            dsize: key_i.dsize,
            seed: key_i.seed.clone(),
            rank_out: key_i.rank_out,
            data: poulpy_hal::layouts::mat_znx_backend_ref_from_ref::<BE>(&key_i.data),
        })
    }
}

impl<'a, BE: Backend + 'a> Deref for GGLWEToGGSWKeyCompressedBackendRef<'a, BE> {
    type Target = GGLWEToGGSWKeyCompressed<BE::BufRef<'a>, BE::ZnxWord>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

pub struct GGLWEToGGSWKeyCompressedBackendMut<'a, BE: Backend + 'a> {
    inner: GGLWEToGGSWKeyCompressed<BE::BufMut<'a>, BE::ZnxWord>,
}

impl<'a, BE: Backend + 'a> GGLWEToGGSWKeyCompressedBackendMut<'a, BE> {
    pub fn from_inner(inner: GGLWEToGGSWKeyCompressed<BE::BufMut<'a>, BE::ZnxWord>) -> Self {
        Self { inner }
    }

    pub fn into_inner(self) -> GGLWEToGGSWKeyCompressed<BE::BufMut<'a>, BE::ZnxWord> {
        self.inner
    }

    pub fn at_view(&self, i: usize) -> GGLWECompressedBackendRef<'_, BE> {
        assert!((i as u32) < self.rank());
        let key_i = &self.inner.keys[i];
        GGLWECompressedBackendRef::from_inner(GGLWECompressed {
            k_aux: key_i.k_aux,
            base2k: key_i.base2k,
            dsize: key_i.dsize,
            seed: key_i.seed.clone(),
            rank_out: key_i.rank_out,
            data: poulpy_hal::layouts::mat_znx_backend_ref_from_mut::<BE>(&key_i.data),
        })
    }

    pub fn at_view_mut(&mut self, i: usize) -> GGLWECompressedBackendMut<'_, BE> {
        assert!((i as u32) < self.rank());
        let key_i = &mut self.inner.keys[i];
        GGLWECompressedBackendMut::from_inner(GGLWECompressed {
            k_aux: key_i.k_aux,
            base2k: key_i.base2k,
            dsize: key_i.dsize,
            seed: key_i.seed.clone(),
            rank_out: key_i.rank_out,
            data: poulpy_hal::layouts::mat_znx_backend_mut_from_mut::<BE>(&mut key_i.data),
        })
    }
}

impl<'a, BE: Backend + 'a> Deref for GGLWEToGGSWKeyCompressedBackendMut<'a, BE> {
    type Target = GGLWEToGGSWKeyCompressed<BE::BufMut<'a>, BE::ZnxWord>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<'a, BE: Backend + 'a> DerefMut for GGLWEToGGSWKeyCompressedBackendMut<'a, BE> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

impl_gglwe_infos_for_inner!(GGLWEToGGSWKeyCompressedBackendRef<'a, BE>, ['a, BE: Backend + 'a]; inner);
impl_gglwe_infos_for_inner!(GGLWEToGGSWKeyCompressedBackendMut<'a, BE>, ['a, BE: Backend + 'a]; inner);

impl<D: Data, W: ZnxWord> LWEInfos for GGLWEToGGSWKeyCompressed<D, W> {
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

impl<D: Data, W: ZnxWord> GLWEInfos for GGLWEToGGSWKeyCompressed<D, W> {
    fn rank(&self) -> Rank {
        self.keys[0].rank_out()
    }
}

impl<D: Data, W: ZnxWord> GGLWEInfos for GGLWEToGGSWKeyCompressed<D, W> {
    fn k_aux(&self) -> TorusPrecision {
        self.keys[0].k_aux()
    }

    fn rank_in(&self) -> Rank {
        self.rank_out()
    }

    fn rank_out(&self) -> Rank {
        self.keys[0].rank_out()
    }

    fn dsize(&self) -> Dsize {
        self.keys[0].dsize()
    }

    fn dnum(&self) -> Dnum {
        self.keys[0].dnum()
    }
}

impl<D: HostDataRef, W: ZnxWord> fmt::Debug for GGLWEToGGSWKeyCompressed<D, W> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{self}")
    }
}

impl<D: HostDataMut, W: ZnxWord> FillUniform for GGLWEToGGSWKeyCompressed<D, W> {
    fn fill_uniform(&mut self, log_bound: usize, source: &mut Source) {
        self.keys
            .iter_mut()
            .for_each(|key: &mut GGLWECompressed<D, W>| key.fill_uniform(log_bound, source))
    }
}

impl<D: HostDataRef, W: ZnxWord> fmt::Display for GGLWEToGGSWKeyCompressed<D, W> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "(GGLWEToGGSWKeyCompressed)",)?;
        for (i, key) in self.keys.iter().enumerate() {
            write!(f, "{i}: {key}")?;
        }
        Ok(())
    }
}

impl<D: Data, W: ZnxWord> GGLWEToGGSWKeyCompressed<D, W> {
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
            infos.dsize(),
            infos.k_aux(),
            infos.rank(),
        )
    }

    /// Allocates a new compressed GGLWE-to-GGSW key with the given parameters.
    pub(crate) fn alloc<B: Backend<OwnedBuf = D, ZnxWord = W>>(
        n: Degree,
        base2k: Base2K,
        dnum: Dnum,
        dsize: Dsize,
        k_aux: TorusPrecision,
        rank: Rank,
    ) -> Self {
        GGLWEToGGSWKeyCompressed {
            keys: (0..rank.as_usize())
                .map(|_| GGLWECompressed::alloc::<B>(n, base2k, dnum, dsize, k_aux, rank, rank))
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
            infos.dsize(),
            infos.k_aux(),
            infos.rank(),
        )
    }

    /// Returns the serialized byte size for a compressed GGLWE-to-GGSW key with the given parameters.
    pub fn bytes_of(n: Degree, base2k: Base2K, dnum: Dnum, dsize: Dsize, k_aux: TorusPrecision, rank: Rank) -> usize {
        rank.as_usize() * GGLWECompressed::<Vec<u8>, W>::bytes_of(n, base2k, dnum, dsize, k_aux, rank)
    }
}

impl<D: HostDataMut, W: ZnxWord> GGLWEToGGSWKeyCompressed<D, W> {
    // Returns a mutable reference to GGLWE_{s}([s[i]*s[0], s[i]*s[1], ..., s[i]*s[rank]])
    pub fn at_mut(&mut self, i: usize) -> &mut GGLWECompressed<D, W> {
        assert!((i as u32) < self.rank());
        &mut self.keys[i]
    }
}

impl<D: HostDataRef, W: ZnxWord> GGLWEToGGSWKeyCompressed<D, W> {
    // Returns a reference to GGLWE_{s}(s[i] * s[j])
    pub fn at(&self, i: usize) -> &GGLWECompressed<D, W> {
        assert!((i as u32) < self.rank());
        &self.keys[i]
    }
}

impl<D: HostDataMut, W: ZnxWord> ReaderFrom for GGLWEToGGSWKeyCompressed<D, W> {
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

impl<D: HostDataRef, W: ZnxWord> WriterTo for GGLWEToGGSWKeyCompressed<D, W> {
    fn write_to<Wr: std::io::Write>(&self, writer: &mut Wr) -> std::io::Result<()> {
        writer.write_u64::<LittleEndian>(self.keys.len() as u64)?;
        for key in &self.keys {
            key.write_to(writer)?;
        }
        Ok(())
    }
}

/// Trait for decompressing a [`GGLWEToGGSWKeyCompressed`] into a standard [`GGLWEToGGSWKey`].
pub trait GGLWEToGGSWKeyDecompress
where
    Self: GGLWEDecompress,
{
    /// Decompresses `other` into `res` by decompressing each GGLWE entry.
    fn decompress_gglwe_to_ggsw_key<R, O>(&self, res: &mut R, other: &O)
    where
        R: GGLWEToGGSWKeyToBackendMut<Self::Backend>,
        O: GGLWEToGGSWKeyCompressedToBackendRef<Self::Backend>,
    {
        let mut res = res.to_backend_mut();
        let other = other.to_backend_ref();
        assert_eq!(res.keys.len(), other.keys.len());
        let mut scratch = ScratchOwned::<Self::Backend>::alloc(self.decompress_glwe_tmp_bytes());
        for i in 0..res.keys.len() {
            let mut a = res.at_view_mut(i);
            let b = other.at_view(i);
            assert_eq!(a.dsize(), b.dsize());
            assert!(a.dnum() <= b.dnum());

            let rank_in: usize = a.rank_in().into();
            let dnum: usize = a.dnum().into();
            for col_i in 0..rank_in {
                for row_i in 0..dnum {
                    let mut dst = a.at_view_mut(row_i, col_i);
                    let src = b.at_view(row_i, col_i);
                    self.decompress_glwe_with_scratch(&mut dst, &src, &mut scratch.borrow());
                }
            }
        }
    }
}

// module-only API: decompression is provided by `GGLWEToGGSWKeyDecompress` on `Module`.

pub trait GGLWEToGGSWKeyCompressedToBackendRef<BE: Backend> {
    fn to_backend_ref(&self) -> GGLWEToGGSWKeyCompressedBackendRef<'_, BE>;
}

impl<BE: Backend> GGLWEToGGSWKeyCompressedToBackendRef<BE> for GGLWEToGGSWKeyCompressed<BE::OwnedBuf, BE::ZnxWord> {
    fn to_backend_ref(&self) -> GGLWEToGGSWKeyCompressedBackendRef<'_, BE> {
        GGLWEToGGSWKeyCompressedBackendRef::from_inner(GGLWEToGGSWKeyCompressed {
            keys: self
                .keys
                .iter()
                .map(|key| GGLWECompressedToBackendRef::<BE>::to_backend_ref(key).into_inner())
                .collect(),
        })
    }
}

pub trait GGLWEToGGSWKeyCompressedToBackendMut<BE: Backend>: GGLWEToGGSWKeyCompressedToBackendRef<BE> {
    fn to_backend_mut(&mut self) -> GGLWEToGGSWKeyCompressedBackendMut<'_, BE>;
}

impl<BE: Backend> GGLWEToGGSWKeyCompressedToBackendMut<BE> for GGLWEToGGSWKeyCompressed<BE::OwnedBuf, BE::ZnxWord> {
    fn to_backend_mut(&mut self) -> GGLWEToGGSWKeyCompressedBackendMut<'_, BE> {
        GGLWEToGGSWKeyCompressedBackendMut::from_inner(GGLWEToGGSWKeyCompressed {
            keys: self
                .keys
                .iter_mut()
                .map(|key| GGLWECompressedToBackendMut::<BE>::to_backend_mut(key).into_inner())
                .collect(),
        })
    }
}
