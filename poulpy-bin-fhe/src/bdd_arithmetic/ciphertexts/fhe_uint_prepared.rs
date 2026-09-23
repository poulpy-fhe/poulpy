pub use crate::api::{FheUintPrepare, FheUintPreparedEncryptSk};
use poulpy_hal::AlignedBuf;
use std::marker::PhantomData;

use poulpy_core::layouts::{
    Base2K, Dnum, Dsize, GGSWInfos, GGSWPreparedFactory, GLWEInfos, LWEInfos, ModuleCoreAlloc, Rank, TorusPrecision,
    prepared::{GGSWPrepared, GGSWPreparedBackendMut},
};
use poulpy_core::layouts::{GGSWPreparedToBackendMut, GetAutomorphismKey};
use poulpy_core::{EncryptionInfos, GLWECopy, GLWEDecrypt, GLWEPacking};

use poulpy_core::layouts::GLWESecretPreparedToBackendRef;
use poulpy_hal::api::ModuleLogN;
use poulpy_hal::layouts::{Backend, Data, HostBackend, HostDataRef, Module};

use poulpy_hal::{layouts::ScratchArena, source::Source};

use crate::bdd_arithmetic::{BDDKey, BDDKeyHelper, BDDKeyInfos, BDDKeyPrepared, BDDKeyPreparedFactory, BitSize, FheUint, ToBits};
use crate::bdd_arithmetic::{Cmux, FromBits, UnsignedInteger};
use crate::blind_rotation::BlindRotationAlgo;
use poulpy_core::GLWEBytesOf;

/// A DFT-prepared FHE ciphertext encoding each bit of a [`UnsignedInteger`]
/// as a separate GGSW ciphertext.
///
/// Unlike [`FheUint`], where all bits share a single GLWE polynomial, each bit
/// of an `FheUintPrepared` is stored as a full GGSW matrix in the DFT domain,
/// making it immediately usable as a CMux selector without any additional
/// forward transform.
///
/// ## Invariants
///
/// - `bits.len() == T::BITS`.
/// - All GGSW entries share the same `n`, `base2k`, `k`, `dnum`, `dsize`, and
///   `rank` parameters.
///
/// ## Lifecycle
///
/// 1. Allocate with [`FheUintPrepared::alloc`] or [`FheUintPrepared::alloc_from_infos`].
/// 2. Populate from plaintext with [`FheUintPrepared::encrypt_sk`], or derive
///    from a packed [`FheUint`] with [`FheUintPrepared::prepare`].
/// 3. Use as input to BDD circuit evaluation (`ExecuteBDDCircuit`).
///
/// ## Thread Safety
///
/// `FheUintPrepared<&[u8], T, BE>` is `Sync`; multiple evaluation threads may
/// access separate bits concurrently through [`GetGGSWBit`].
pub struct FheUintPrepared<D: Data, T: UnsignedInteger, B: Backend> {
    pub(crate) bits: Vec<GGSWPrepared<D, B>>,
    pub(crate) _phantom: PhantomData<T>,
}

impl<T: UnsignedInteger, BE: Backend> FheUintPreparedFactory<T, BE> for Module<BE> where Self: Sized + GGSWPreparedFactory<BE> {}

/// Read-only access to individual DFT-domain GGSW bit-ciphertexts.
///
/// Implemented by [`FheUintPrepared`] and by the internal `FheUintHelper`
/// used during two-word BDD evaluation.  Required by `ExecuteBDDCircuit`
/// and `GLWEBlindRotation`.
pub trait GetGGSWBit<BE: Backend>: Sync {
    /// Returns a shared reference view of the GGSW ciphertext for bit `bit`.
    ///
    /// # Panics
    ///
    /// Panics if `bit >= self.bit_size()`.
    fn get_bit(&self, bit: usize) -> &GGSWPrepared<BE::OwnedBuf, BE>;
}

impl<T: UnsignedInteger, BE: Backend<ZnxWord = i64>> GetGGSWBit<BE> for FheUintPrepared<BE::OwnedBuf, T, BE> {
    fn get_bit(&self, bit: usize) -> &GGSWPrepared<BE::OwnedBuf, BE> {
        assert!(
            bit < self.bits.len(),
            "bit index {bit} out of bounds, len={}",
            self.bits.len()
        );
        &self.bits[bit]
    }
}

/// Mutable access to individual DFT-domain GGSW bit-ciphertexts.
///
/// Used during [`FheUintPrepared::prepare`] and its multi-thread variant to
/// write bootstrapped GGSW output into the prepared ciphertext in parallel.
pub trait GetGGSWBitMut<T: UnsignedInteger, BE: Backend> {
    /// Returns a mutable view of the GGSW ciphertext for bit `bit`.
    ///
    /// # Panics
    ///
    /// Panics if `bit >= self.bit_size()`.
    fn get_bit(&mut self, bit: usize) -> GGSWPreparedBackendMut<'_, BE>;
    /// Returns mutable views of `count` consecutive GGSW ciphertexts starting
    /// at `start`.
    ///
    /// # Panics
    ///
    /// Panics if `start + count > self.bit_size()`.
    fn get_bits(&mut self, start: usize, count: usize) -> Vec<GGSWPreparedBackendMut<'_, BE>>;
}

impl<D: Data, T: UnsignedInteger, BE: Backend> GetGGSWBitMut<T, BE> for FheUintPrepared<D, T, BE>
where
    GGSWPrepared<D, BE>: GGSWPreparedToBackendMut<BE>,
{
    fn get_bit(&mut self, bit: usize) -> GGSWPreparedBackendMut<'_, BE> {
        assert!(
            bit < self.bits.len(),
            "bit index {bit} out of bounds, len={}",
            self.bits.len()
        );
        self.bits[bit].to_backend_mut()
    }
    fn get_bits(&mut self, start: usize, count: usize) -> Vec<GGSWPreparedBackendMut<'_, BE>> {
        assert!(start + count <= self.bits.len());
        self.bits[start..start + count]
            .iter_mut()
            .map(|bit| bit.to_backend_mut())
            .collect()
    }
}

impl<D: Data, T: UnsignedInteger, BE: Backend> BitSize for FheUintPrepared<D, T, BE> {
    fn bit_size(&self) -> usize {
        T::BITS as usize
    }
}

pub trait FheUintPreparedFactory<T: UnsignedInteger, BE: Backend>
where
    Self: Sized + GGSWPreparedFactory<BE>,
{
    fn alloc_fhe_uint_prepared(
        &self,
        base2k: Base2K,
        dnum: Dnum,
        dsize: Dsize,
        k_aux: TorusPrecision,
        rank: Rank,
    ) -> FheUintPrepared<BE::OwnedBuf, T, BE> {
        FheUintPrepared {
            bits: (0..T::BITS)
                .map(|_| self.ggsw_prepared_alloc(base2k, dnum, dsize, k_aux, rank))
                .collect(),
            _phantom: PhantomData,
        }
    }

    fn alloc_fhe_uint_prepared_from_infos<A>(&self, infos: &A) -> FheUintPrepared<BE::OwnedBuf, T, BE>
    where
        A: GGSWInfos,
    {
        self.alloc_fhe_uint_prepared(infos.base2k(), infos.dnum(), infos.dsize(), infos.k_aux(), infos.rank())
    }
}

impl<T: UnsignedInteger, BE: Backend> FheUintPrepared<BE::OwnedBuf, T, BE> {
    pub fn alloc_from_infos<A, M>(module: &M, infos: &A) -> Self
    where
        A: GGSWInfos,
        M: FheUintPreparedFactory<T, BE>,
    {
        module.alloc_fhe_uint_prepared_from_infos(infos)
    }

    pub fn alloc<M>(module: &M, base2k: Base2K, dnum: Dnum, dsize: Dsize, k_aux: TorusPrecision, rank: Rank) -> Self
    where
        M: FheUintPreparedFactory<T, BE>,
    {
        module.alloc_fhe_uint_prepared(base2k, dnum, dsize, k_aux, rank)
    }
}

impl<T: UnsignedInteger + ToBits, BE: Backend<ZnxWord = i64>> FheUintPrepared<BE::OwnedBuf, T, BE> {
    #[allow(clippy::too_many_arguments)]
    pub fn encrypt_sk<M, S, E>(
        &mut self,
        module: &M,
        value: T,
        sk: &S,
        enc_infos: &E,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        S: GLWESecretPreparedToBackendRef<BE> + GLWEInfos,
        M: FheUintPreparedEncryptSk<T, BE>,
        E: EncryptionInfos,
    {
        module.fhe_uint_prepared_encrypt_sk(self, value, sk, enc_infos, source_xe, source_xa, scratch);
    }
}

impl<T: UnsignedInteger + FromBits, BE: Backend<OwnedBuf = AlignedBuf, ZnxWord = i64> + HostBackend>
    FheUintPrepared<BE::OwnedBuf, T, BE>
where
    BE::OwnedBuf: HostDataRef,
{
    pub fn decrypt<M, S, H>(&self, module: &M, sk: &S, keys: &H, scratch: &mut ScratchArena<'_, BE>) -> T
    where
        M: ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = BE::ZnxWord>
            + ModuleLogN
            + GLWEDecrypt<BE>
            + Cmux<BE>
            + GLWEPacking<BE>
            + GLWECopy<BE>
            + GLWEBytesOf<BE>,
        S: GLWESecretPreparedToBackendRef<BE> + GLWEInfos,
        H: GetAutomorphismKey<BE>,
    {
        let mut tmp: FheUint<BE::OwnedBuf, T, BE::ZnxWord> = FheUint::alloc_from_infos(module, self);
        let mut scratch_1 = scratch.borrow();
        tmp.from_fhe_uint_prepared(module, self, keys, &mut scratch_1);
        tmp.decrypt(module, sk, &mut scratch_1)
    }
}

impl<D: Data, T: UnsignedInteger, B: Backend> LWEInfos for FheUintPrepared<D, T, B> {
    fn base2k(&self) -> poulpy_core::layouts::Base2K {
        self.bits[0].base2k()
    }

    fn max_size(&self) -> usize {
        self.bits[0].max_size()
    }

    fn n(&self) -> poulpy_core::layouts::Degree {
        self.bits[0].n()
    }

    fn k(&self) -> TorusPrecision {
        self.bits[0].k()
    }
}

impl<D: Data, T: UnsignedInteger, B: Backend> GLWEInfos for FheUintPrepared<D, T, B> {
    fn rank(&self) -> poulpy_core::layouts::Rank {
        self.bits[0].rank()
    }
}

impl<D: Data, T: UnsignedInteger, B: Backend> GGSWInfos for FheUintPrepared<D, T, B> {
    fn k_aux(&self) -> TorusPrecision {
        self.bits[0].k_aux()
    }

    fn dsize(&self) -> poulpy_core::layouts::Dsize {
        self.bits[0].dsize()
    }

    fn dnum(&self) -> poulpy_core::layouts::Dnum {
        self.bits[0].dnum()
    }
}

impl<BRA: BlindRotationAlgo, BE: Backend<ZnxWord = i64>> BDDKeyPrepared<BE::OwnedBuf, BRA, BE> {
    pub fn prepare<M>(&mut self, module: &M, other: &BDDKey<BE::OwnedBuf, BRA, BE::ZnxWord>, scratch: &mut ScratchArena<'_, BE>)
    where
        M: BDDKeyPreparedFactory<BRA, BE>,
    {
        module.prepare_bdd_key(self, other, scratch);
    }
}

/// Backend-level factory for bootstrapping a packed [`FheUint`] into
/// a [`FheUintPrepared`].
///
/// For each bit of the input word, extracts an LWE ciphertext from the packed
/// GLWE, applies the circuit bootstrapping pipeline (blind rotation + trace +
/// key-switch), and DFT-prepares the resulting GGSW in-place.
///
/// The `_custom` and `_multi_thread` variants allow partial updates (only a
/// contiguous range of bits) and backend-selected parallel execution,
/// respectively. Serial backends execute these methods serially.
impl<T: UnsignedInteger, BE: Backend<ZnxWord = i64>> FheUintPrepared<BE::OwnedBuf, T, BE> {
    pub fn prepare<BRA, M, K>(
        &mut self,
        module: &M,
        other: &FheUint<BE::OwnedBuf, T, BE::ZnxWord>,
        key: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        BRA: BlindRotationAlgo,
        K: BDDKeyHelper<BE::OwnedBuf, BRA, BE> + BDDKeyInfos,
        M: FheUintPrepare<BRA, BE>,
    {
        module.fhe_uint_prepare(self, other, key, scratch);
    }
    pub fn prepare_custom<BRA, M, K>(
        &mut self,
        module: &M,
        other: &FheUint<BE::OwnedBuf, T, BE::ZnxWord>,
        bit_start: usize,
        bit_end: usize,
        key: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        BRA: BlindRotationAlgo,
        K: BDDKeyHelper<BE::OwnedBuf, BRA, BE> + BDDKeyInfos,
        M: FheUintPrepare<BRA, BE>,
    {
        assert!(bit_start <= bit_end);
        module.fhe_uint_prepare_custom(self, other, bit_start, bit_end - bit_start, key, scratch);
    }

    #[allow(clippy::too_many_arguments)]
    pub fn prepare_custom_multi_thread<BRA, M, K>(
        &mut self,
        threads: usize,
        module: &M,
        other: &FheUint<BE::OwnedBuf, T, BE::ZnxWord>,
        bit_start: usize,
        bit_end: usize,
        key: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        BRA: BlindRotationAlgo,
        K: BDDKeyHelper<BE::OwnedBuf, BRA, BE> + BDDKeyInfos,
        M: FheUintPrepare<BRA, BE>,
    {
        assert!(bit_start <= bit_end);
        module.fhe_uint_prepare_custom_multi_thread(threads, self, other, bit_start, bit_end - bit_start, key, scratch);
    }
}
