use std::marker::PhantomData;

use crate::bdd_arithmetic::{BDDKeyPrepared, FheUint, FheUintPrepareDebug, ToBits};
use crate::{
    bdd_arithmetic::UnsignedInteger, blind_rotation::BlindRotationAlgo, circuit_bootstrapping::CircuitBootstrappingExecute,
};
use poulpy_core::GGSWNoise;

use poulpy_core::layouts::{Base2K, Dnum, Dsize, ModuleCoreAlloc, Rank, TorusPrecision};
use poulpy_core::layouts::{GGSW, GLWESecretPreparedToBackendRef, LWE};
use poulpy_core::{
    GLWEKeyswitch, LWEFromGLWE,
    layouts::{GGSWInfos, GGSWPreparedFactory, GLWEInfos, LWEInfos},
};

use poulpy_hal::api::ModuleN;
use poulpy_hal::layouts::{
    Backend, Data, HostBackend, HostBytesBackend, HostDataMut, HostDataRef, Module, ScalarZnx, ScalarZnxToBackendRef,
    ScratchArena, Stats, ZnxWord, ZnxZero,
};

/// A debug variant of `FheUintPrepared` that stores per-bit GGSW ciphertexts
/// in standard (non-DFT) form.
///
/// Compared to `FheUintPrepared`, this variant cannot be used as a CMux
/// selector directly, but it allows noise measurement via
/// [`FheUintPreparedDebug::noise`] without a forward DFT transform.
///
/// ## Usage
///
/// Intended for testing and parameter validation.  Construct with
/// [`FheUintPreparedDebug::alloc`] and populate with
/// [`FheUintPreparedDebug::prepare`].
pub struct FheUintPreparedDebug<D: Data, T: UnsignedInteger, W: ZnxWord> {
    pub(crate) bits: Vec<GGSW<D, W>>,
    pub(crate) _phantom: PhantomData<T>,
}

impl<D: Data, T: UnsignedInteger, W: ZnxWord> FheUintPreparedDebug<D, T, W> {
    pub fn alloc_from_infos<A, M>(module: &M, infos: &A) -> Self
    where
        M: ModuleN + ModuleCoreAlloc<OwnedBuf = D, ZnxWord = W>,
        A: GGSWInfos,
    {
        Self::alloc(
            module,
            infos.base2k(),
            infos.dnum(),
            infos.dsize(),
            infos.k_aux(),
            infos.rank(),
        )
    }

    pub fn alloc<M>(module: &M, base2k: Base2K, dnum: Dnum, dsize: Dsize, k_aux: TorusPrecision, rank: Rank) -> Self
    where
        M: ModuleN + ModuleCoreAlloc<OwnedBuf = D, ZnxWord = W>,
    {
        Self {
            bits: (0..T::BITS)
                .map(|_| module.ggsw_alloc(base2k, dnum, dsize, k_aux, rank))
                .collect(),
            _phantom: PhantomData,
        }
    }
}

impl<D: HostDataRef, T: UnsignedInteger, W: ZnxWord> LWEInfos for FheUintPreparedDebug<D, T, W> {
    fn base2k(&self) -> poulpy_core::layouts::Base2K {
        self.bits[0].base2k()
    }

    fn n(&self) -> poulpy_core::layouts::Degree {
        self.bits[0].n()
    }

    fn max_size(&self) -> usize {
        self.bits[0].max_size()
    }

    fn k(&self) -> TorusPrecision {
        self.bits[0].k()
    }
}

impl<D: HostDataRef, T: UnsignedInteger, W: ZnxWord> GLWEInfos for FheUintPreparedDebug<D, T, W> {
    fn rank(&self) -> poulpy_core::layouts::Rank {
        self.bits[0].rank()
    }
}

impl<D: HostDataRef, T: UnsignedInteger, W: ZnxWord> GGSWInfos for FheUintPreparedDebug<D, T, W> {
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

impl<T: UnsignedInteger + ToBits> FheUintPreparedDebug<Vec<u8>, T, i64> {
    pub fn noise<S, M, BE>(
        &self,
        module: &M,
        row: usize,
        col: usize,
        want: T,
        sk: &S,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Vec<Stats>
    where
        S: GLWESecretPreparedToBackendRef<BE>,
        M: GGSWNoise<BE>,
        BE: Backend<OwnedBuf = Vec<u8>, ZnxWord = i64> + HostBackend,
        for<'a> BE::BufRef<'a>: HostDataRef,
        for<'a> BE::BufMut<'a>: AsMut<[u8]> + AsRef<[u8]> + Sync,
    {
        let mut stats = Vec::new();
        for (i, ggsw) in self.bits.iter().enumerate() {
            use poulpy_hal::layouts::ZnxViewMut;
            let mut pt_want: ScalarZnx<BE::OwnedBuf, BE::ZnxWord> = ScalarZnx::from_data(
                HostBytesBackend::alloc_bytes(ScalarZnx::<Vec<u8>, i64>::bytes_of(usize::from(self.n()), 1)),
                usize::from(self.n()),
                1,
            );
            pt_want.zero();
            pt_want.at_mut(0, 0)[0] = want.bit(i) as i64;
            let mut scratch_bit = scratch.borrow();
            let pt_ref =
                <ScalarZnx<BE::OwnedBuf, BE::ZnxWord> as ScalarZnxToBackendRef<HostBytesBackend>>::to_backend_ref(&pt_want);
            stats.push(ggsw.noise(module, row, col, &pt_ref, sk, &mut scratch_bit));
        }
        stats
    }
}

impl<BRA: BlindRotationAlgo, BE: Backend + HostBackend, T: UnsignedInteger> FheUintPrepareDebug<BRA, T, BE> for Module<BE>
where
    Self: ModuleN + LWEFromGLWE<BE> + GLWEKeyswitch<BE> + CircuitBootstrappingExecute<BRA, BE> + GGSWPreparedFactory<BE>,
    BE: Backend<OwnedBuf: HostDataMut + HostDataRef, ZnxWord = i64>,
    BE::OwnedBuf: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
{
    fn fhe_uint_debug_prepare(
        &self,
        res: &mut FheUintPreparedDebug<BE::OwnedBuf, T, BE::ZnxWord>,
        bits: &FheUint<BE::OwnedBuf, T, BE::ZnxWord>,
        key: &BDDKeyPrepared<BE::OwnedBuf, BRA, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) {
        let mut tmp_lwe: LWE<BE::OwnedBuf, BE::ZnxWord> = self.lwe_alloc_from_infos(bits);
        let mut scratch_1 = scratch.borrow();
        for (bit, dst) in res.bits.iter_mut().enumerate() {
            let mut scratch_bit = scratch_1.borrow();
            bits.get_bit_lwe(self, bit, &mut tmp_lwe, key.ks_glwe.as_ref(), &key.ks_lwe, &mut scratch_bit);
            key.cbt.execute_to_constant(self, dst, &tmp_lwe, 1, 1, &mut scratch_bit);
        }
    }
}

impl<T: UnsignedInteger> FheUintPreparedDebug<Vec<u8>, T, i64> {
    pub fn prepare<BRA, M, BE>(
        &mut self,
        module: &M,
        other: &FheUint<BE::OwnedBuf, T, BE::ZnxWord>,
        key: &BDDKeyPrepared<BE::OwnedBuf, BRA, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        BRA: BlindRotationAlgo,
        BE: Backend<OwnedBuf = Vec<u8>, ZnxWord = i64> + HostBackend,
        M: FheUintPrepareDebug<BRA, T, BE>,
    {
        module.fhe_uint_debug_prepare(self, other, key, scratch);
    }
}
