use poulpy_hal::AlignedBuf;
use poulpy_hal::layouts::{Backend, Data, HostDataMut, HostDataRef, ReaderFrom, WriterTo, ZnxWord};

use std::{fmt, marker::PhantomData};

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use poulpy_core::{
    Distribution,
    layouts::{Base2K, Degree, Dsize, GGSWInfos, GLWEInfos, LWEInfos, compressed::GGSWCompressed},
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
pub struct BlindRotationKeyCompressed<D: Data, BRT: BlindRotationAlgo, W: ZnxWord> {
    pub(crate) keys: Vec<GGSWCompressed<D, W>>,
    pub(crate) dist: Distribution,
    pub(crate) _phantom: PhantomData<BRT>,
}

pub use crate::api::blind_rotation::BlindRotationKeyCompressedFactory;

impl<BRA: BlindRotationAlgo> BlindRotationKeyCompressed<AlignedBuf, BRA, i64> {
    /// Allocates a compressed key through the module's selected factory.
    pub fn alloc<M, A, BE>(module: &M, infos: &A) -> BlindRotationKeyCompressed<BE::OwnedBuf, BRA, BE::ZnxWord>
    where
        BE: Backend,
        M: BlindRotationKeyCompressedFactory<BRA, BE>,
        A: BlindRotationKeyInfos,
    {
        module.blind_rotation_key_compressed_alloc(infos)
    }
}

impl<D: HostDataRef, BRT: BlindRotationAlgo, W: ZnxWord> fmt::Debug for BlindRotationKeyCompressed<D, BRT, W> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{self}")
    }
}

impl<D: Data, BRT: BlindRotationAlgo, W: ZnxWord> PartialEq for BlindRotationKeyCompressed<D, BRT, W> {
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

impl<D: Data, BRT: BlindRotationAlgo, W: ZnxWord> Eq for BlindRotationKeyCompressed<D, BRT, W> {}

impl<D: HostDataRef, BRT: BlindRotationAlgo, W: ZnxWord> fmt::Display for BlindRotationKeyCompressed<D, BRT, W> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, key) in self.keys.iter().enumerate() {
            write!(f, "key[{i}]: {key}")?;
        }
        writeln!(f, "{:?}", self.dist)
    }
}

impl<D: HostDataMut, BRT: BlindRotationAlgo, W: ZnxWord> ReaderFrom for BlindRotationKeyCompressed<D, BRT, W> {
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

impl<D: HostDataRef, BRT: BlindRotationAlgo, W: ZnxWord> WriterTo for BlindRotationKeyCompressed<D, BRT, W> {
    fn write_to<Wr: std::io::Write>(&self, writer: &mut Wr) -> std::io::Result<()> {
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

impl<D: Data, BRA: BlindRotationAlgo, W: ZnxWord> BlindRotationKeyInfos for BlindRotationKeyCompressed<D, BRA, W> {
    fn n_glwe(&self) -> Degree {
        self.n()
    }

    fn n_lwe(&self) -> Degree {
        Degree(self.keys.len() as u32)
    }
}

impl<D: Data, BRA: BlindRotationAlgo, W: ZnxWord> LWEInfos for BlindRotationKeyCompressed<D, BRA, W> {
    fn n(&self) -> Degree {
        self.keys[0].n()
    }

    fn max_size(&self) -> usize {
        self.keys[0].max_size()
    }

    fn base2k(&self) -> Base2K {
        self.keys[0].base2k()
    }

    fn k(&self) -> poulpy_core::layouts::TorusPrecision {
        self.keys[0].k()
    }
}

impl<D: Data, BRA: BlindRotationAlgo, W: ZnxWord> GLWEInfos for BlindRotationKeyCompressed<D, BRA, W> {
    fn rank(&self) -> poulpy_core::layouts::Rank {
        self.keys[0].rank()
    }
}

impl<D: Data, BRA: BlindRotationAlgo, W: ZnxWord> GGSWInfos for BlindRotationKeyCompressed<D, BRA, W> {
    fn k_aux(&self) -> poulpy_core::layouts::TorusPrecision {
        self.keys[0].k_aux()
    }

    fn dnum(&self) -> poulpy_core::layouts::Dnum {
        self.keys[0].dnum()
    }

    fn dsize(&self) -> poulpy_core::layouts::Dsize {
        Dsize(1)
    }
}

impl<D: Data, BRA: BlindRotationAlgo, W: ZnxWord> BlindRotationKeyCompressed<D, BRA, W> {
    #[allow(dead_code)]
    pub(crate) fn block_size(&self) -> usize {
        match self.dist {
            Distribution::BinaryBlock(value) => value,
            _ => 1,
        }
    }
}

impl<D: Data, BRA: BlindRotationAlgo, W: ZnxWord> BlindRotationKeyCompressed<D, BRA, W> {
    /// Constructs a key from coefficient-domain elements and its secret distribution.
    pub fn from_parts(keys: Vec<GGSWCompressed<D, W>>, distribution: Distribution) -> Self {
        assert!(!keys.is_empty());
        Self {
            keys,
            dist: distribution,
            _phantom: PhantomData,
        }
    }
    /// Coefficient-domain key elements.
    pub fn keys(&self) -> &[GGSWCompressed<D, W>] {
        &self.keys
    }
    /// Mutable coefficient-domain key elements for backend implementations.
    pub fn keys_mut(&mut self) -> &mut [GGSWCompressed<D, W>] {
        &mut self.keys
    }
    /// Secret distribution carried by the key.
    pub fn distribution(&self) -> Distribution {
        self.dist
    }
    /// Updates distribution metadata after encryption or decompression.
    pub fn set_distribution(&mut self, distribution: Distribution) {
        self.dist = distribution;
    }
}

impl<D: Data, BRA: BlindRotationAlgo, W: ZnxWord> BlindRotationKeyCompressed<D, BRA, W> {
    /// Decompresses this key through the selected backend operation.
    pub fn decompress_into<BE, M>(
        &self,
        module: &M,
        output: &mut crate::blind_rotation::BlindRotationKey<D, BRA, W>,
        scratch: &mut poulpy_hal::layouts::ScratchArena<'_, BE>,
    ) where
        BE: poulpy_hal::layouts::Backend<OwnedBuf = D, ZnxWord = W>,
        M: crate::api::BlindRotationKeyDecompress<BRA, BE>,
    {
        module.blind_rotation_key_decompress(output, self, scratch);
    }
}
