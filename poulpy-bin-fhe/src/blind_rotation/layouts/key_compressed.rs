use poulpy_hal::{
    api::ModuleN,
    layouts::{Data, FillUniform, HostDataMut, HostDataRef, ReaderFrom, WriterTo},
    source::Source,
};

use std::{fmt, marker::PhantomData};

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use poulpy_core::{
    Distribution,
    layouts::{Base2K, Degree, Dsize, GGSWInfos, GLWEInfos, LWEInfos, ModuleCoreCompressedAlloc, compressed::GGSWCompressed},
};

use crate::blind_rotation::{BlindRotationAlgo, BlindRotationKeyInfos};

/// Seed-compressed form of a blind rotation bootstrapping key.
///
/// Each GGSW element stores only the body component; the mask (the `A`
/// polynomial) is deterministically regenerated from a per-key seed during
/// preparation, halving the serialised size relative to `BlindRotationKey`.
///
/// ## Trade-offs vs. Standard Key
///
/// - **Storage**: Roughly half the size of the standard form.
/// - **Preparation time**: Slower because masks must be regenerated from
///   the seed on every call to the prepare step.
/// - **On-line evaluation**: Identical to the standard form once prepared.
///
/// ## Invariants
///
/// - `keys.len() == n_lwe`.
/// - `dist` records the LWE secret distribution; `Distribution::NONE` before
///   encryption.
///
/// ## Serialisation
///
/// Implements [`ReaderFrom`] and [`WriterTo`].  The binary format is identical
/// in structure to `BlindRotationKey` but each element is `GGSWCompressed`.
#[derive(Clone)]
pub struct BlindRotationKeyCompressed<D: Data, BRT: BlindRotationAlgo> {
    pub(crate) keys: Vec<GGSWCompressed<D>>,
    pub(crate) dist: Distribution,
    pub(crate) _phantom: PhantomData<BRT>,
}

/// Algorithm-specific factory for allocating a [`BlindRotationKeyCompressed`].
pub trait BlindRotationKeyCompressedFactory<BRA: BlindRotationAlgo> {
    fn blind_rotation_key_compressed_alloc<M, A>(module: &M, infos: &A) -> BlindRotationKeyCompressed<Vec<u8>, BRA>
    where
        M: ModuleCoreCompressedAlloc + ModuleN,
        A: BlindRotationKeyInfos;
}

impl<BRA: BlindRotationAlgo> BlindRotationKeyCompressed<Vec<u8>, BRA>
where
    Self: BlindRotationKeyCompressedFactory<BRA>,
{
    pub fn alloc<M, A>(module: &M, infos: &A) -> BlindRotationKeyCompressed<Vec<u8>, BRA>
    where
        M: ModuleCoreCompressedAlloc + ModuleN,
        A: BlindRotationKeyInfos,
    {
        Self::blind_rotation_key_compressed_alloc(module, infos)
    }
}

impl<D: HostDataRef, BRT: BlindRotationAlgo> fmt::Debug for BlindRotationKeyCompressed<D, BRT> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{self}")
    }
}

impl<D: Data, BRT: BlindRotationAlgo> PartialEq for BlindRotationKeyCompressed<D, BRT> {
    fn eq(&self, other: &Self) -> bool {
        if self.keys.len() != other.keys.len() {
            return false;
        }
        for (a, b) in self.keys.iter().zip(other.keys.iter()) {
            if a != b {
                return false;
            }
        }
        self.dist == other.dist && self._phantom == other._phantom
    }
}

impl<D: Data, BRT: BlindRotationAlgo> Eq for BlindRotationKeyCompressed<D, BRT> {}

impl<D: HostDataRef, BRT: BlindRotationAlgo> fmt::Display for BlindRotationKeyCompressed<D, BRT> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, key) in self.keys.iter().enumerate() {
            write!(f, "key[{i}]: {key}")?;
        }
        writeln!(f, "{:?}", self.dist)
    }
}

impl<D: HostDataMut, BRT: BlindRotationAlgo> FillUniform for BlindRotationKeyCompressed<D, BRT> {
    fn fill_uniform(&mut self, log_bound: usize, source: &mut Source) {
        self.keys.iter_mut().for_each(|key| key.fill_uniform(log_bound, source));
    }
}

impl<D: HostDataMut, BRT: BlindRotationAlgo> ReaderFrom for BlindRotationKeyCompressed<D, BRT> {
    fn read_from<R: std::io::Read>(&mut self, reader: &mut R) -> std::io::Result<()> {
        self.dist = Distribution::read_from(reader)?;
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

impl<D: HostDataRef, BRT: BlindRotationAlgo> WriterTo for BlindRotationKeyCompressed<D, BRT> {
    fn write_to<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<()> {
        match self.dist.write_to(writer) {
            Ok(()) => {}
            Err(e) => return Err(e),
        }
        writer.write_u64::<LittleEndian>(self.keys.len() as u64)?;
        for key in &self.keys {
            key.write_to(writer)?;
        }
        Ok(())
    }
}

impl<D: HostDataRef, BRA: BlindRotationAlgo> BlindRotationKeyInfos for BlindRotationKeyCompressed<D, BRA> {
    fn n_glwe(&self) -> Degree {
        self.n()
    }

    fn n_lwe(&self) -> Degree {
        Degree(self.keys.len() as u32)
    }
}

impl<D: HostDataRef, BRA: BlindRotationAlgo> LWEInfos for BlindRotationKeyCompressed<D, BRA> {
    fn n(&self) -> Degree {
        self.keys[0].n()
    }

    fn size(&self) -> usize {
        self.keys[0].size()
    }

    fn base2k(&self) -> Base2K {
        self.keys[0].base2k()
    }
}

impl<D: HostDataRef, BRA: BlindRotationAlgo> GLWEInfos for BlindRotationKeyCompressed<D, BRA> {
    fn rank(&self) -> poulpy_core::layouts::Rank {
        self.keys[0].rank()
    }
}

impl<D: HostDataRef, BRA: BlindRotationAlgo> GGSWInfos for BlindRotationKeyCompressed<D, BRA> {
    fn dnum(&self) -> poulpy_core::layouts::Dnum {
        self.keys[0].dnum()
    }

    fn dsize(&self) -> poulpy_core::layouts::Dsize {
        Dsize(1)
    }
}

impl<D: HostDataRef, BRA: BlindRotationAlgo> BlindRotationKeyCompressed<D, BRA> {
    #[allow(dead_code)]
    pub(crate) fn block_size(&self) -> usize {
        match self.dist {
            Distribution::BinaryBlock(value) => value,
            _ => 1,
        }
    }
}
