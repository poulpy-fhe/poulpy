use poulpy_hal::{
    api::ModuleN,
    layouts::{CopyFromHost, CopyToHost, Data, FillUniform, HostDataMut, HostDataRef, ReaderFrom, WriterTo, ZnxWord},
    source::Source,
};

use std::{fmt, marker::PhantomData};

use poulpy_core::{
    Distribution, EncryptionLayout, TransferInto,
    layouts::{Base2K, Degree, Dnum, Dsize, GGSW, GGSWInfos, GLWEInfos, LWEInfos, ModuleCoreAlloc, Rank, TorusPrecision},
};

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};

use crate::blind_rotation::BlindRotationAlgo;

/// Plain-old-data descriptor for all dimensional parameters of a blind
/// rotation key.
///
/// This struct aggregates the dimensions needed to allocate and interpret a
/// [`BlindRotationKey`] without requiring access to the actual key data.  It
/// can be constructed manually or extracted from an existing key via
/// [`BlindRotationKeyInfos`].
///
/// # Fields
///
/// - `n_glwe`: Polynomial degree of the GLWE / GGSW ciphertext components.
/// - `n_lwe`: Number of LWE ciphertext dimensions; equals the number of GGSW
///   ciphertexts stored in the key.
/// - `base2k`: Decomposition base (bits per limb).
/// - `dnum`: Number of gadget decomposition digits (`dsize` fixed to 1); the
///   gadget precision is `dnum * dsize * base2k` and the full precision is
///   `key_k(base2k, dnum, Dsize(1), k_aux)`.
/// - `k_aux`: Auxiliary guard precision (torus bits) below the gadget region.
/// - `rank`: GLWE rank (0 for plain LWE, ≥ 1 for GLWE / Module-LWE).
#[derive(PartialEq, Eq, Copy, Clone, Debug)]
pub struct BlindRotationKeyLayout {
    pub n_glwe: Degree,
    pub n_lwe: Degree,
    pub base2k: Base2K,
    pub dnum: Dnum,
    pub k_aux: TorusPrecision,
    pub rank: Rank,
}

impl BlindRotationKeyInfos for BlindRotationKeyLayout {
    fn n_glwe(&self) -> Degree {
        self.n_glwe
    }

    fn n_lwe(&self) -> Degree {
        self.n_lwe
    }
}

impl BlindRotationKeyInfos for EncryptionLayout<BlindRotationKeyLayout> {
    fn n_glwe(&self) -> Degree {
        self.layout.n_glwe()
    }

    fn n_lwe(&self) -> Degree {
        self.layout.n_lwe()
    }
}

impl GGSWInfos for BlindRotationKeyLayout {
    fn k_aux(&self) -> TorusPrecision {
        self.k_aux
    }

    fn dsize(&self) -> Dsize {
        Dsize(1)
    }

    fn dnum(&self) -> Dnum {
        self.dnum
    }
}

impl GLWEInfos for BlindRotationKeyLayout {
    fn rank(&self) -> Rank {
        self.rank
    }
}

impl LWEInfos for BlindRotationKeyLayout {
    fn base2k(&self) -> Base2K {
        self.base2k
    }

    fn n(&self) -> Degree {
        self.n_glwe
    }

    fn max_size(&self) -> usize {
        poulpy_core::layouts::key_size(self.base2k, self.dnum, Dsize(1), self.k_aux)
    }

    fn k(&self) -> TorusPrecision {
        poulpy_core::layouts::key_k(self.base2k, self.dnum, Dsize(1), self.k_aux)
    }
}

/// Accessor trait for blind-rotation key dimensions.
///
/// Provides `n_glwe` and `n_lwe` on top of the [`GGSWInfos`] accessors
/// common to all GGSW-based key types.  Implemented by
/// [`BlindRotationKeyLayout`], [`BlindRotationKey`], and
/// `BlindRotationKeyPrepared`.
pub trait BlindRotationKeyInfos
where
    Self: GGSWInfos,
{
    /// Polynomial degree of the GLWE ring used for the GGSW ciphertexts.
    fn n_glwe(&self) -> Degree;
    /// Number of LWE dimensions; equals the number of GGSW elements in the key.
    fn n_lwe(&self) -> Degree;
}

/// Allocation trait for bootstrapping keys.
pub trait BlindRotationKeyAlloc {
    /// Allocates an uninitialised (zero-filled) key from a dimension descriptor.
    fn alloc<A>(infos: &A) -> Self
    where
        A: BlindRotationKeyInfos;
}

/// Standard (un-prepared) blind rotation bootstrapping key.
///
/// Stores one GGSW ciphertext per LWE coefficient encrypting the corresponding
/// secret-key bit (or block of bits for the `BinaryBlock` distribution).  The
/// key also records the distribution of the LWE secret key so the correct
/// execution path can be selected at evaluation time.
///
/// ## Key Lifecycle
///
/// 1. Allocate with [`BlindRotationKey::alloc`].
/// 2. Fill with [`BlindRotationKey::encrypt_sk`].
/// 3. Prepare for evaluation with `BlindRotationKeyPrepared::prepare`.
///
/// ## Serialisation
///
/// Implements [`ReaderFrom`] and [`WriterTo`] for little-endian binary I/O.
/// The serialised format prefixes the distribution tag and the key-element
/// count before the individual GGSW payloads.
///
/// ## Invariants
///
/// - `keys.len() == n_lwe`.
/// - `dist` is set to the distribution of the LWE secret after `encrypt_sk`;
///   it is `Distribution::NONE` in a freshly allocated key.
#[derive(Clone)]
pub struct BlindRotationKey<D: Data, BRT: BlindRotationAlgo, W: ZnxWord> {
    pub(crate) keys: Vec<GGSW<D, W>>,
    pub(crate) dist: Distribution,
    pub(crate) _phantom: PhantomData<BRT>,
}

impl<D: Data, BRA: BlindRotationAlgo, W: ZnxWord> BlindRotationKey<D, BRA, W> {
    pub fn alloc<M, A>(module: &M, infos: &A) -> BlindRotationKey<D, BRA, W>
    where
        M: ModuleCoreAlloc<OwnedBuf = D, ZnxWord = W> + ModuleN,
        A: BlindRotationKeyInfos,
    {
        BRA::alloc_key(module, infos)
    }
}

impl<D: HostDataRef, BRT: BlindRotationAlgo, W: ZnxWord> fmt::Debug for BlindRotationKey<D, BRT, W> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{self}")
    }
}

impl<D: Data, BRT: BlindRotationAlgo, W: ZnxWord> PartialEq for BlindRotationKey<D, BRT, W> {
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

impl<D: Data, BRT: BlindRotationAlgo, W: ZnxWord> Eq for BlindRotationKey<D, BRT, W> {}

impl<D1, D2, BRT, W> TransferInto<BlindRotationKey<D2, BRT, W>> for BlindRotationKey<D1, BRT, W>
where
    D1: Data + CopyToHost,
    D2: Data + CopyFromHost,
    BRT: BlindRotationAlgo,
    W: ZnxWord,
{
    fn transfer_into(&self, dst: &mut BlindRotationKey<D2, BRT, W>) {
        assert_eq!(self.keys.len(), dst.keys.len());
        for (src, dst) in self.keys.iter().zip(&mut dst.keys) {
            src.transfer_into(dst);
        }
        dst.dist = self.dist;
    }
}

impl<D: HostDataRef, BRT: BlindRotationAlgo, W: ZnxWord> fmt::Display for BlindRotationKey<D, BRT, W> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, key) in self.keys.iter().enumerate() {
            write!(f, "key[{i}]: {key}")?;
        }
        writeln!(f, "{:?}", self.dist)
    }
}

impl<D: HostDataMut, BRT: BlindRotationAlgo, W: ZnxWord> FillUniform for BlindRotationKey<D, BRT, W> {
    fn fill_uniform(&mut self, log_bound: usize, source: &mut Source) {
        self.keys.iter_mut().for_each(|key| key.fill_uniform(log_bound, source));
    }
}

impl<D: HostDataMut, BRT: BlindRotationAlgo, W: ZnxWord> ReaderFrom for BlindRotationKey<D, BRT, W> {
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

impl<D: HostDataRef, BRT: BlindRotationAlgo, W: ZnxWord> WriterTo for BlindRotationKey<D, BRT, W> {
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

impl<D: Data, BRT: BlindRotationAlgo, W: ZnxWord> BlindRotationKeyInfos for BlindRotationKey<D, BRT, W> {
    fn n_glwe(&self) -> Degree {
        self.n()
    }

    fn n_lwe(&self) -> Degree {
        Degree(self.keys.len() as u32)
    }
}

impl<D: Data, BRT: BlindRotationAlgo, W: ZnxWord> BlindRotationKey<D, BRT, W> {
    pub fn block_size(&self) -> usize {
        match self.dist {
            Distribution::BinaryBlock(value) => value,
            _ => 1,
        }
    }
}

impl<D: Data, BRT: BlindRotationAlgo, W: ZnxWord> LWEInfos for BlindRotationKey<D, BRT, W> {
    fn base2k(&self) -> Base2K {
        self.keys[0].base2k()
    }

    fn n(&self) -> Degree {
        self.keys[0].n()
    }

    fn max_size(&self) -> usize {
        self.keys[0].max_size()
    }

    fn k(&self) -> TorusPrecision {
        self.keys[0].k()
    }
}

impl<D: Data, BRT: BlindRotationAlgo, W: ZnxWord> GLWEInfos for BlindRotationKey<D, BRT, W> {
    fn rank(&self) -> Rank {
        self.keys[0].rank()
    }
}
impl<D: Data, BRT: BlindRotationAlgo, W: ZnxWord> GGSWInfos for BlindRotationKey<D, BRT, W> {
    fn k_aux(&self) -> TorusPrecision {
        self.keys[0].k_aux()
    }

    fn dsize(&self) -> poulpy_core::layouts::Dsize {
        Dsize(1)
    }

    fn dnum(&self) -> Dnum {
        self.keys[0].dnum()
    }
}
