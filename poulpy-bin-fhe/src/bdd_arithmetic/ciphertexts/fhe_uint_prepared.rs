use std::marker::PhantomData;

use poulpy_core::layouts::{
    Base2K, Dnum, Dsize, GGSWInfos, GGSWPreparedFactory, GLWEInfos, LWEInfos, ModuleCoreAlloc, Rank, TorusPrecision,
    prepared::{GGSWPrepared, GGSWPreparedBackendMut},
};
use poulpy_core::layouts::{
    GGLWEInfos, GGLWEPreparedToBackendRef, GGSW, GGSWLayout, GGSWPreparedToBackendMut, GLWEAutomorphismKeyHelper,
    GetGaloisElement, LWE,
};
use poulpy_core::{EncryptionInfos, GLWECopy, GLWEDecrypt, GLWEKeyswitch, GLWEPacking, LWEFromGLWE};

use poulpy_core::{GGSWEncryptSk, ScratchArenaTakeCore, layouts::GLWESecretPreparedToBackendRef};
use poulpy_hal::api::{ModuleLogN, ScratchArenaTakeBasic};
use poulpy_hal::layouts::{Backend, Data, HostBackend, HostDataMut, HostDataRef, Module, ScalarZnx};

use poulpy_hal::{api::ModuleN, layouts::ScratchArena, source::Source};

use crate::bdd_arithmetic::{BDDKey, BDDKeyHelper, BDDKeyInfos, BDDKeyPrepared, BDDKeyPreparedFactory, BitSize, FheUint, ToBits};
use crate::bdd_arithmetic::{Cmux, FromBits, UnsignedInteger};
use crate::blind_rotation::BlindRotationAlgo;
use crate::circuit_bootstrapping::{CircuitBootstrappingExecute, CircuitBootstrappingKeyInfos};

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
pub trait GetGGSWBit<BE: Backend<OwnedBuf = Vec<u8>>>: Sync {
    /// Returns a shared reference view of the GGSW ciphertext for bit `bit`.
    ///
    /// # Panics
    ///
    /// Panics if `bit >= self.bit_size()`.
    fn get_bit(&self, bit: usize) -> &GGSWPrepared<BE::OwnedBuf, BE>;
}

impl<T: UnsignedInteger, BE: Backend<OwnedBuf = Vec<u8>>> GetGGSWBit<BE> for FheUintPrepared<BE::OwnedBuf, T, BE> {
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

/// Backend-level factory for allocating [`FheUintPrepared`] values.
///
/// Implemented for `Module<BE>` when the backend supports DFT-domain GGSW
/// preparation.  Callers should use the convenience methods on
/// `FheUintPrepared` rather than calling these directly.
pub trait FheUintPreparedFactory<T: UnsignedInteger, BE: Backend>
where
    Self: Sized + GGSWPreparedFactory<BE>,
{
    fn alloc_fhe_uint_prepared(
        &self,
        base2k: Base2K,
        k: TorusPrecision,
        dnum: Dnum,
        dsize: Dsize,
        rank: Rank,
    ) -> FheUintPrepared<BE::OwnedBuf, T, BE> {
        FheUintPrepared {
            bits: (0..T::BITS)
                .map(|_| self.ggsw_prepared_alloc(base2k, k, dnum, dsize, rank))
                .collect(),
            _phantom: PhantomData,
        }
    }

    fn alloc_fhe_uint_prepared_from_infos<A>(&self, infos: &A) -> FheUintPrepared<BE::OwnedBuf, T, BE>
    where
        A: GGSWInfos,
    {
        self.alloc_fhe_uint_prepared(infos.base2k(), infos.max_k(), infos.dnum(), infos.dsize(), infos.rank())
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

    pub fn alloc<M>(module: &M, base2k: Base2K, k: TorusPrecision, dnum: Dnum, dsize: Dsize, rank: Rank) -> Self
    where
        M: FheUintPreparedFactory<T, BE>,
    {
        module.alloc_fhe_uint_prepared(base2k, k, dnum, dsize, rank)
    }
}

impl<T: UnsignedInteger + ToBits, BE: Backend<OwnedBuf = Vec<u8>> + HostBackend> FheUintPreparedEncryptSk<T, BE> for Module<BE> where
    Self: Sized + ModuleN + GGSWEncryptSk<BE> + GGSWPreparedFactory<BE> + ModuleCoreAlloc<OwnedBuf = Vec<u8>>
{
}

/// Backend-level factory for directly encrypting a plaintext value into a
/// [`FheUintPrepared`] without first creating an [`FheUint`].
///
/// Useful in testing and debugging scenarios where the packed-GLWE intermediate
/// form is not needed.  Each bit is encrypted independently as a constant GGSW
/// and then immediately DFT-prepared in place.
pub trait FheUintPreparedEncryptSk<T: UnsignedInteger + ToBits, BE: Backend<OwnedBuf = Vec<u8>> + HostBackend>
where
    Self: Sized + ModuleN + GGSWEncryptSk<BE> + GGSWPreparedFactory<BE> + ModuleCoreAlloc<OwnedBuf = Vec<u8>>,
{
    #[allow(clippy::too_many_arguments)]
    fn fhe_uint_prepared_encrypt_sk<S, E>(
        &self,
        res: &mut FheUintPrepared<BE::OwnedBuf, T, BE>,
        value: T,
        sk: &S,
        enc_infos: &E,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        S: GLWESecretPreparedToBackendRef<BE> + GLWEInfos,
        E: EncryptionInfos,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeBasic<'a, BE>,
        for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    {
        use poulpy_hal::layouts::ZnxZero;

        assert!(self.n().is_multiple_of(T::BITS as usize));
        assert_eq!(res.n(), self.n() as u32);
        assert_eq!(sk.n(), self.n() as u32);

        let mut tmp_ggsw: GGSW<BE::OwnedBuf> = self.ggsw_alloc_from_infos(res);
        let (mut pt, mut scratch_1) = scratch.borrow().take_scalar_znx_scratch(self.n(), 1);
        pt.zero();

        for i in 0..T::BITS as usize {
            use poulpy_hal::layouts::ZnxViewMut;
            pt.at_mut(0, 0)[0] = value.bit(i) as i64;
            let pt_ref = pt.to_ref();
            let pt_backend = ScalarZnx::from_data(BE::from_host_bytes(pt_ref.data), pt_ref.n(), pt_ref.cols());
            let mut scratch_bit = scratch_1.borrow();
            self.ggsw_encrypt_sk(
                &mut tmp_ggsw,
                &pt_backend,
                sk,
                enc_infos,
                source_xe,
                source_xa,
                &mut scratch_bit,
            );
            self.ggsw_prepare(&mut res.bits[i], &tmp_ggsw, &mut scratch_bit);
        }
    }
}

impl<T: UnsignedInteger + ToBits, BE: Backend<OwnedBuf = Vec<u8>>> FheUintPrepared<BE::OwnedBuf, T, BE> {
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
        BE: HostBackend,
        S: GLWESecretPreparedToBackendRef<BE> + GLWEInfos,
        M: FheUintPreparedEncryptSk<T, BE>,
        E: EncryptionInfos,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeBasic<'a, BE>,
        for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    {
        module.fhe_uint_prepared_encrypt_sk(self, value, sk, enc_infos, source_xe, source_xa, scratch);
    }
}

impl<T: UnsignedInteger + FromBits, BE: Backend<OwnedBuf = Vec<u8>> + HostBackend> FheUintPrepared<BE::OwnedBuf, T, BE>
where
    BE::OwnedBuf: HostDataRef,
{
    pub fn decrypt<M, S, H, K>(&self, module: &M, sk: &S, keys: &H, scratch: &mut ScratchArena<'_, BE>) -> T
    where
        M: ModuleCoreAlloc<OwnedBuf = Vec<u8>> + ModuleLogN + GLWEDecrypt<BE> + Cmux<BE> + GLWEPacking<BE> + GLWECopy<BE>,
        S: GLWESecretPreparedToBackendRef<BE> + GLWEInfos,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
        K: GGLWEPreparedToBackendRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>,
        BE::OwnedBuf: HostDataRef,
        BE: 'static,
        for<'a> BE::BufMut<'a>: HostDataMut,
        for<'a> BE: Backend<BufMut<'a> = &'a mut [u8], BufRef<'a> = &'a [u8]>,
    {
        let mut tmp: FheUint<Vec<u8>, T> = FheUint::alloc_from_infos(module, self);
        let mut scratch_1 = scratch.borrow();
        tmp.from_fhe_uint_prepared(module, self, keys, &mut scratch_1);
        tmp.decrypt(module, sk, &mut scratch_1)
    }
}

impl<D: HostDataRef, T: UnsignedInteger, B: Backend> LWEInfos for FheUintPrepared<D, T, B> {
    fn base2k(&self) -> poulpy_core::layouts::Base2K {
        self.bits[0].base2k()
    }

    fn size(&self) -> usize {
        self.bits[0].size()
    }

    fn n(&self) -> poulpy_core::layouts::Degree {
        self.bits[0].n()
    }
}

impl<D: HostDataRef, T: UnsignedInteger, B: Backend> GLWEInfos for FheUintPrepared<D, T, B> {
    fn rank(&self) -> poulpy_core::layouts::Rank {
        self.bits[0].rank()
    }
}

impl<D: HostDataRef, T: UnsignedInteger, B: Backend> GGSWInfos for FheUintPrepared<D, T, B> {
    fn dsize(&self) -> poulpy_core::layouts::Dsize {
        self.bits[0].dsize()
    }

    fn dnum(&self) -> poulpy_core::layouts::Dnum {
        self.bits[0].dnum()
    }
}

impl<BRA: BlindRotationAlgo, BE: Backend<OwnedBuf = Vec<u8>>> BDDKeyPrepared<BE::OwnedBuf, BRA, BE> {
    pub fn prepare<'s, M>(&mut self, module: &M, other: &BDDKey<BE::OwnedBuf, BRA>, scratch: &mut ScratchArena<'s, BE>)
    where
        M: BDDKeyPreparedFactory<BRA, BE>,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
        BE: 's,
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
/// contiguous range of bits) and parallel execution across OS threads,
/// respectively.
pub trait FheUintPrepare<BRA: BlindRotationAlgo, BE: Backend<OwnedBuf = Vec<u8>>> {
    /// Returns the minimum scratch-space size in bytes required per thread for
    /// [`fhe_uint_prepare`][Self::fhe_uint_prepare].
    fn fhe_uint_prepare_tmp_bytes<R, A, B>(
        &self,
        block_size: usize,
        extension_factor: usize,
        res_infos: &R,
        bits_infos: &A,
        bdd_infos: &B,
    ) -> usize
    where
        R: GGSWInfos,
        A: GLWEInfos,
        B: BDDKeyInfos;
    fn fhe_uint_prepare<K, T: UnsignedInteger>(
        &self,
        res: &mut FheUintPrepared<BE::OwnedBuf, T, BE>,
        bits: &FheUint<BE::OwnedBuf, T>,
        key: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        K: BDDKeyHelper<BE::OwnedBuf, BRA, BE> + BDDKeyInfos,
    {
        self.fhe_uint_prepare_custom(res, bits, 0, T::BITS as usize, key, scratch);
    }
    fn fhe_uint_prepare_custom<K, T: UnsignedInteger>(
        &self,
        res: &mut FheUintPrepared<BE::OwnedBuf, T, BE>,
        bits: &FheUint<BE::OwnedBuf, T>,
        bit_start: usize,
        bit_count: usize,
        key: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        K: BDDKeyHelper<BE::OwnedBuf, BRA, BE> + BDDKeyInfos,
    {
        self.fhe_uint_prepare_custom_multi_thread(1, res, bits, bit_start, bit_count, key, scratch)
    }
    #[allow(clippy::too_many_arguments)]
    fn fhe_uint_prepare_custom_multi_thread<K, T: UnsignedInteger>(
        &self,
        threads: usize,
        res: &mut FheUintPrepared<BE::OwnedBuf, T, BE>,
        bits: &FheUint<BE::OwnedBuf, T>,
        bit_start: usize,
        bit_count: usize,
        key: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        K: BDDKeyHelper<BE::OwnedBuf, BRA, BE> + BDDKeyInfos;
}

impl<BRA: BlindRotationAlgo, BE: Backend> FheUintPrepare<BRA, BE> for Module<BE>
where
    Self: LWEFromGLWE<BE> + GLWEKeyswitch<BE> + CircuitBootstrappingExecute<BRA, BE> + GGSWPreparedFactory<BE>,
    for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
    BE: Backend<OwnedBuf = Vec<u8>>,
    BE::OwnedBuf: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
{
    fn fhe_uint_prepare_tmp_bytes<R, A, B>(
        &self,
        block_size: usize,
        extension_factor: usize,
        res_infos: &R,
        bits_infos: &A,
        bdd_infos: &B,
    ) -> usize
    where
        R: GGSWInfos,
        A: GLWEInfos,
        B: BDDKeyInfos,
    {
        self.circuit_bootstrapping_execute_tmp_bytes(block_size, extension_factor, res_infos, &bdd_infos.cbt_infos())
            + GGSW::bytes_of_from_infos(res_infos)
            + LWE::bytes_of_from_infos(bits_infos)
    }

    fn fhe_uint_prepare_custom_multi_thread<K, T: UnsignedInteger>(
        &self,
        _threads: usize,
        res: &mut FheUintPrepared<BE::OwnedBuf, T, BE>,
        bits: &FheUint<BE::OwnedBuf, T>,
        bit_start: usize,
        bit_count: usize,
        key: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        K: BDDKeyHelper<BE::OwnedBuf, BRA, BE> + BDDKeyInfos,
    {
        let bit_end = bit_start + bit_count;
        let (cbt, ks_glwe, ks_lwe) = key.get_cbt_key();

        assert!(bit_end <= T::BITS as usize);

        let scratch_thread_size = self.fhe_uint_prepare_tmp_bytes(cbt.block_size(), 1, res, bits, key);

        assert!(
            scratch.available() >= scratch_thread_size,
            "scratch.available():{} < scratch_thread_size:{scratch_thread_size}",
            scratch.available()
        );

        let ggsw_infos: &GGSWLayout = &res.ggsw_layout();
        let scratch_local = scratch.borrow();
        let mut tmp_ggsw: GGSW<Vec<u8>> = self.ggsw_alloc_from_infos(ggsw_infos);
        let mut tmp_lwe: LWE<Vec<u8>> = self.lwe_alloc_from_infos(bits);
        let mut scratch_1 = scratch_local;

        for bit in bit_start..bit_end {
            let mut scratch_bit = scratch_1.borrow();
            bits.get_bit_lwe(self, bit, &mut tmp_lwe, ks_glwe, ks_lwe, &mut scratch_bit);
            cbt.execute_to_constant(self, &mut tmp_ggsw, &tmp_lwe, 1, 1, &mut scratch_bit);
            self.ggsw_prepare(&mut res.bits[bit], &tmp_ggsw, &mut scratch_bit);
        }

        for i in 0..bit_start {
            self.ggsw_zero(&mut res.bits[i]);
        }

        for i in bit_end..T::BITS as usize {
            self.ggsw_zero(&mut res.bits[i]);
        }
    }
}

impl<T: UnsignedInteger, BE: Backend<OwnedBuf = Vec<u8>>> FheUintPrepared<BE::OwnedBuf, T, BE> {
    pub fn prepare<BRA, M, K>(
        &mut self,
        module: &M,
        other: &FheUint<BE::OwnedBuf, T>,
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
        other: &FheUint<BE::OwnedBuf, T>,
        bit_start: usize,
        bit_end: usize,
        key: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        BRA: BlindRotationAlgo,
        K: BDDKeyHelper<BE::OwnedBuf, BRA, BE> + BDDKeyInfos,
        M: FheUintPrepare<BRA, BE>,
    {
        module.fhe_uint_prepare_custom(self, other, bit_start, bit_end, key, scratch);
    }

    #[allow(clippy::too_many_arguments)]
    pub fn prepare_custom_multi_thread<BRA, M, K>(
        &mut self,
        threads: usize,
        module: &M,
        other: &FheUint<BE::OwnedBuf, T>,
        bit_start: usize,
        bit_end: usize,
        key: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        BRA: BlindRotationAlgo,
        K: BDDKeyHelper<BE::OwnedBuf, BRA, BE> + BDDKeyInfos,
        M: FheUintPrepare<BRA, BE>,
    {
        module.fhe_uint_prepare_custom_multi_thread(threads, self, other, bit_start, bit_end, key, scratch);
    }
}
