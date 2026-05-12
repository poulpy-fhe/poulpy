use poulpy_hal::{
    layouts::{Backend, Data, FillUniform, HostDataMut, HostDataRef, ReaderFrom, WriterTo},
    source::Source,
};

use crate::{
    DeclaredK,
    layouts::{
        Base2K, Degree, Dnum, Dsize, GGLWE, GGLWEBackendMut, GGLWEInfos, GGLWEToBackendMut, GGLWEToBackendRef, GLWEInfos,
        LWEInfos, Rank, TorusPrecision,
    },
};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};

use std::{
    fmt,
    ops::{Deref, DerefMut},
};

#[derive(PartialEq, Eq, Copy, Clone, Debug)]
pub struct GGLWEToGGSWKeyLayout {
    pub n: Degree,
    pub base2k: Base2K,
    pub k: TorusPrecision,
    pub rank: Rank,
    pub dnum: Dnum,
    pub dsize: Dsize,
}

impl DeclaredK for GGLWEToGGSWKeyLayout {
    fn k(&self) -> TorusPrecision {
        self.k
    }
}

#[derive(PartialEq, Eq, Clone)]
pub struct GGLWEToGGSWKey<D: Data> {
    pub(crate) keys: Vec<GGLWE<D>>,
}

pub struct GGLWEToGGSWKeyBackendRef<'a, BE: Backend + 'a> {
    inner: GGLWEToGGSWKey<BE::BufRef<'a>>,
}

impl<'a, BE: Backend + 'a> GGLWEToGGSWKeyBackendRef<'a, BE> {
    pub fn from_inner(inner: GGLWEToGGSWKey<BE::BufRef<'a>>) -> Self {
        Self { inner }
    }

    pub fn into_inner(self) -> GGLWEToGGSWKey<BE::BufRef<'a>> {
        self.inner
    }

    pub fn at_view(&self, i: usize) -> crate::layouts::GGLWEBackendRef<'_, BE> {
        assert!((i as u32) < self.rank());
        let key_i = &self.inner.keys[i];
        crate::layouts::GGLWEBackendRef::from_inner(GGLWE {
            base2k: key_i.base2k,
            dsize: key_i.dsize,
            data: poulpy_hal::layouts::mat_znx_backend_ref_from_ref::<BE>(&key_i.data),
        })
    }
}

impl<'a, BE: Backend + 'a> Deref for GGLWEToGGSWKeyBackendRef<'a, BE> {
    type Target = GGLWEToGGSWKey<BE::BufRef<'a>>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

pub struct GGLWEToGGSWKeyBackendMut<'a, BE: Backend + 'a> {
    inner: GGLWEToGGSWKey<BE::BufMut<'a>>,
}

impl<'a, BE: Backend + 'a> GGLWEToGGSWKeyBackendMut<'a, BE> {
    pub fn from_inner(inner: GGLWEToGGSWKey<BE::BufMut<'a>>) -> Self {
        Self { inner }
    }

    pub fn into_inner(self) -> GGLWEToGGSWKey<BE::BufMut<'a>> {
        self.inner
    }

    pub fn at_view(&self, i: usize) -> crate::layouts::GGLWEBackendRef<'_, BE> {
        assert!((i as u32) < self.rank());
        let key_i = &self.inner.keys[i];
        crate::layouts::GGLWEBackendRef::from_inner(GGLWE {
            base2k: key_i.base2k,
            dsize: key_i.dsize,
            data: poulpy_hal::layouts::mat_znx_backend_ref_from_mut::<BE>(&key_i.data),
        })
    }

    pub fn at_view_mut(&mut self, i: usize) -> GGLWEBackendMut<'_, BE> {
        assert!((i as u32) < self.rank());
        let key_i = &mut self.inner.keys[i];
        GGLWEBackendMut::from_inner(GGLWE {
            base2k: key_i.base2k,
            dsize: key_i.dsize,
            data: poulpy_hal::layouts::mat_znx_backend_mut_from_mut::<BE>(&mut key_i.data),
        })
    }
}

impl<'a, BE: Backend + 'a> Deref for GGLWEToGGSWKeyBackendMut<'a, BE> {
    type Target = GGLWEToGGSWKey<BE::BufMut<'a>>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<'a, BE: Backend + 'a> DerefMut for GGLWEToGGSWKeyBackendMut<'a, BE> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

impl_gglwe_infos_for_inner!(GGLWEToGGSWKeyBackendRef<'a, BE>, ['a, BE: Backend + 'a]; inner);
impl_gglwe_infos_for_inner!(GGLWEToGGSWKeyBackendMut<'a, BE>, ['a, BE: Backend + 'a]; inner);

impl<D: Data> LWEInfos for GGLWEToGGSWKey<D> {
    fn n(&self) -> Degree {
        self.keys[0].n()
    }

    fn base2k(&self) -> Base2K {
        self.keys[0].base2k()
    }

    fn size(&self) -> usize {
        self.keys[0].size()
    }
}

impl<D: Data> GLWEInfos for GGLWEToGGSWKey<D> {
    fn rank(&self) -> Rank {
        self.keys[0].rank_out()
    }
}

impl<D: Data> GGLWEInfos for GGLWEToGGSWKey<D> {
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

impl LWEInfos for GGLWEToGGSWKeyLayout {
    fn n(&self) -> Degree {
        self.n
    }

    fn base2k(&self) -> Base2K {
        self.base2k
    }

    fn size(&self) -> usize {
        self.k.as_usize().div_ceil(self.base2k.as_usize())
    }
}

impl GLWEInfos for GGLWEToGGSWKeyLayout {
    fn rank(&self) -> Rank {
        self.rank_out()
    }
}

impl GGLWEInfos for GGLWEToGGSWKeyLayout {
    fn rank_in(&self) -> Rank {
        self.rank
    }

    fn dsize(&self) -> Dsize {
        self.dsize
    }

    fn rank_out(&self) -> Rank {
        self.rank
    }

    fn dnum(&self) -> Dnum {
        self.dnum
    }
}

impl<D: HostDataRef> fmt::Debug for GGLWEToGGSWKey<D> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{self}")
    }
}

impl<D: HostDataMut> FillUniform for GGLWEToGGSWKey<D> {
    fn fill_uniform(&mut self, log_bound: usize, source: &mut Source) {
        self.keys
            .iter_mut()
            .for_each(|key: &mut GGLWE<D>| key.fill_uniform(log_bound, source))
    }
}

impl<D: HostDataRef> fmt::Display for GGLWEToGGSWKey<D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "(GGLWEToGGSWKey)",)?;
        for (i, key) in self.keys.iter().enumerate() {
            write!(f, "{i}: {key}")?;
        }
        Ok(())
    }
}

#[expect(
    dead_code,
    reason = "host-owned constructors are kept for serialization and host-only staging"
)]
impl GGLWEToGGSWKey<Vec<u8>> {
    pub(crate) fn alloc_from_infos<A>(infos: &A) -> Self
    where
        A: GGLWEInfos,
    {
        assert_eq!(
            infos.rank_in(),
            infos.rank_out(),
            "rank_in != rank_out is not supported for GGLWEToGGSWKey"
        );
        Self::alloc(
            infos.n(),
            infos.base2k(),
            infos.max_k(),
            infos.rank(),
            infos.dnum(),
            infos.dsize(),
        )
    }

    pub(crate) fn alloc(n: Degree, base2k: Base2K, k: TorusPrecision, rank: Rank, dnum: Dnum, dsize: Dsize) -> Self {
        GGLWEToGGSWKey {
            keys: (0..rank.as_usize())
                .map(|_| GGLWE::alloc(n, base2k, k, rank, rank, dnum, dsize))
                .collect(),
        }
    }

    pub fn bytes_of_from_infos<A>(infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        assert_eq!(
            infos.rank_in(),
            infos.rank_out(),
            "rank_in != rank_out is not supported for GGLWEToGGSWKey"
        );
        Self::bytes_of(
            infos.n(),
            infos.base2k(),
            infos.max_k(),
            infos.rank(),
            infos.dnum(),
            infos.dsize(),
        )
    }

    pub fn bytes_of(n: Degree, base2k: Base2K, k: TorusPrecision, rank: Rank, dnum: Dnum, dsize: Dsize) -> usize {
        rank.as_usize() * GGLWE::bytes_of(n, base2k, k, rank, rank, dnum, dsize)
    }
}

impl<D: HostDataMut> GGLWEToGGSWKey<D> {
    // Returns a mutable reference to GGLWE_{s}([s[i]*s[0], s[i]*s[1], ..., s[i]*s[rank]])
    pub fn at_mut(&mut self, i: usize) -> &mut GGLWE<D> {
        assert!((i as u32) < self.rank());
        &mut self.keys[i]
    }
}

impl<D: HostDataRef> GGLWEToGGSWKey<D> {
    // Returns a reference to GGLWE_{s}(s[i] * s[j])
    pub fn at(&self, i: usize) -> &GGLWE<D> {
        assert!((i as u32) < self.rank());
        &self.keys[i]
    }
}

impl<D: HostDataMut> ReaderFrom for GGLWEToGGSWKey<D> {
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

impl<D: HostDataRef> WriterTo for GGLWEToGGSWKey<D> {
    fn write_to<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<()> {
        writer.write_u64::<LittleEndian>(self.keys.len() as u64)?;
        for key in &self.keys {
            key.write_to(writer)?;
        }
        Ok(())
    }
}

pub trait GGLWEToGGSWKeyToBackendRef<BE: Backend> {
    fn to_backend_ref(&self) -> GGLWEToGGSWKeyBackendRef<'_, BE>;
}

impl<BE: Backend, D: Data> GGLWEToGGSWKeyToBackendRef<BE> for GGLWEToGGSWKey<D>
where
    GGLWE<D>: GGLWEToBackendRef<BE>,
{
    fn to_backend_ref(&self) -> GGLWEToGGSWKeyBackendRef<'_, BE> {
        GGLWEToGGSWKeyBackendRef::from_inner(GGLWEToGGSWKey {
            keys: self
                .keys
                .iter()
                .map(|key| GGLWEToBackendRef::<BE>::to_backend_ref(key).into_inner())
                .collect(),
        })
    }
}

pub trait GGLWEToGGSWKeyToBackendMut<BE: Backend>: GGLWEToGGSWKeyToBackendRef<BE> {
    fn to_backend_mut(&mut self) -> GGLWEToGGSWKeyBackendMut<'_, BE>;
}

impl<BE: Backend, D: Data> GGLWEToGGSWKeyToBackendMut<BE> for GGLWEToGGSWKey<D>
where
    GGLWE<D>: GGLWEToBackendMut<BE>,
{
    fn to_backend_mut(&mut self) -> GGLWEToGGSWKeyBackendMut<'_, BE> {
        GGLWEToGGSWKeyBackendMut::from_inner(GGLWEToGGSWKey {
            keys: self
                .keys
                .iter_mut()
                .map(|key| GGLWEToBackendMut::<BE>::to_backend_mut(key).into_inner())
                .collect(),
        })
    }
}

impl<BE: Backend> GGLWEToGGSWKeyToBackendRef<BE> for &mut GGLWEToGGSWKey<BE::OwnedBuf> {
    fn to_backend_ref(&self) -> GGLWEToGGSWKeyBackendRef<'_, BE> {
        <GGLWEToGGSWKey<BE::OwnedBuf> as GGLWEToGGSWKeyToBackendRef<BE>>::to_backend_ref(self)
    }
}

impl<BE: Backend> GGLWEToGGSWKeyToBackendMut<BE> for &mut GGLWEToGGSWKey<BE::OwnedBuf> {
    fn to_backend_mut(&mut self) -> GGLWEToGGSWKeyBackendMut<'_, BE> {
        <GGLWEToGGSWKey<BE::OwnedBuf> as GGLWEToGGSWKeyToBackendMut<BE>>::to_backend_mut(self)
    }
}
