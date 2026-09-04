use poulpy_core::layouts::prepared::GGLWEPreparedBackendRef;
use poulpy_core::{
    EncryptionInfos, GLWEAdd, GLWECopy, GLWEDecrypt, GLWEEncryptSk, GLWEKeyswitch, GLWENoise, GLWENormalize, GLWEPacking,
    GLWERotate, GLWESub, GLWETrace, LWEFromGLWE, ScratchArenaTakeCore, TransferInto,
    layouts::{
        Base2K, GGLWEInfos, GLWE, GLWEInfos, GLWEPlaintext, GLWEPlaintextLayout, GLWESecretPreparedToBackendRef,
        GLWEToBackendMut, GLWEToBackendRef, GetAutomorphismKey, LWEInfos, LWEToBackendMut, ModuleCoreAlloc, Rank, TorusPrecision,
    },
};
use poulpy_hal::layouts::CoeffNormalized;
use poulpy_hal::layouts::ZnxWord;
use poulpy_hal::{
    api::{ModuleLogN, ModuleN, VecZnxNormalizeAssignBackend},
    layouts::{Backend, CopyFromHost, CopyToHost, Data, HostBackend, HostDataMut, HostDataRef, ScratchArena, Stats},
    source::Source,
};
use std::{collections::HashMap, marker::PhantomData};

use crate::bdd_arithmetic::{Cmux, FheUintPrepared, FromBits, GetGGSWBit, ToBits, UnsignedInteger};
use poulpy_core::GLWEBytesOf;
use poulpy_core::layouts::prepared::GGSWPreparedToBackendRef;
use poulpy_hal::layouts::BorrowedCarryView;

/// A packed GLWE ciphertext encrypting the bits of a [`UnsignedInteger`].
///
/// All `T::BITS` bits of the plaintext integer are stored in the coefficient
/// slots of a single GLWE polynomial using the interleaved layout defined by
/// [`UnsignedInteger::bit_index`].  This layout allows individual bits or
/// whole bytes to be extracted via a single rotate-and-trace operation.
///
/// ## Lifecycle
///
/// 1. Allocate with [`FheUint::alloc`] or [`FheUint::alloc_from_infos`].
/// 2. Encrypt with [`FheUint::encrypt_sk`].
/// 3. Call `FheUintPrepared::prepare` to convert
///    each bit into a GGSW ciphertext ready for CMux-based circuit evaluation.
/// 4. After BDD evaluation, fresh result bits are packed back into a new
///    `FheUint` with [`FheUint::pack`].
///
/// ## Thread Safety
///
/// `FheUint<&[u8], T>` is `Sync`; shared references can be passed to multiple
/// evaluation threads simultaneously.
pub struct FheUint<D: Data, T: UnsignedInteger, W: ZnxWord> {
    pub(crate) bits: GLWE<D, W>,
    pub(crate) _phantom: PhantomData<T>,
}

impl<D: Data, T: UnsignedInteger, W: ZnxWord> FheUint<D, T, W> {
    pub fn alloc_from_infos<M, A>(module: &M, infos: &A) -> Self
    where
        M: ModuleCoreAlloc<OwnedBuf = D, ZnxWord = W> + ModuleN,
        A: GLWEInfos,
    {
        #[cfg(debug_assertions)]
        {
            assert_eq!(module.n(), infos.n().as_usize());
        }

        Self::alloc(module, infos.base2k(), infos.k(), infos.rank())
    }

    pub fn alloc<M>(module: &M, base2k: Base2K, k: TorusPrecision, rank: Rank) -> Self
    where
        M: ModuleCoreAlloc<OwnedBuf = D, ZnxWord = W> + ModuleN,
    {
        Self {
            bits: module.glwe_alloc(base2k, k, rank),
            _phantom: PhantomData,
        }
    }
}

impl<D1, D2, T, W> TransferInto<FheUint<D2, T, W>> for FheUint<D1, T, W>
where
    D1: Data + CopyToHost,
    D2: Data + CopyFromHost,
    T: UnsignedInteger,
    W: ZnxWord,
{
    fn transfer_into(&self, dst: &mut FheUint<D2, T, W>) {
        self.bits.transfer_into(&mut dst.bits);
    }
}

impl<'a, T: UnsignedInteger> FheUint<&'a mut [u8], T, i64> {
    pub fn from_glwe_to_mut<G>(glwe: &'a mut G) -> Self
    where
        G: GLWEToBackendMut<poulpy_hal::layouts::HostBytesBackend, State = CoeffNormalized>,
    {
        FheUint {
            bits: glwe.to_backend_mut(),
            _phantom: PhantomData,
        }
    }
}

impl<'a, T: UnsignedInteger> FheUint<&'a [u8], T, i64> {
    pub fn from_glwe_to_ref<G>(glwe: &'a G) -> Self
    where
        G: GLWEToBackendRef<poulpy_hal::layouts::HostBytesBackend, State = CoeffNormalized>,
    {
        FheUint {
            bits: glwe.to_backend_ref(),
            _phantom: PhantomData,
        }
    }
}

impl<D: Data, T: UnsignedInteger, W: ZnxWord> LWEInfos for FheUint<D, T, W> {
    fn base2k(&self) -> poulpy_core::layouts::Base2K {
        self.bits.base2k()
    }

    fn max_size(&self) -> usize {
        self.bits.max_size()
    }

    fn n(&self) -> poulpy_core::layouts::Degree {
        self.bits.n()
    }

    fn k(&self) -> TorusPrecision {
        self.bits.k()
    }
}

impl<D: Data, T: UnsignedInteger, W: ZnxWord> GLWEInfos for FheUint<D, T, W> {
    fn rank(&self) -> poulpy_core::layouts::Rank {
        self.bits.rank()
    }
}

impl<D: HostDataMut, T: UnsignedInteger + ToBits> FheUint<D, T, i64> {
    #[allow(clippy::too_many_arguments)]
    pub fn encrypt_sk<S, M, E, BE>(
        &mut self,
        module: &M,
        data: T,
        sk_glwe: &S,
        enc_infos: &E,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        BE: Backend<OwnedBuf: HostDataMut + HostDataRef, ZnxWord = i64>,
        GLWE<D, i64>: GLWEToBackendMut<BE, State = CoeffNormalized>,
        S: GLWESecretPreparedToBackendRef<BE> + GLWEInfos,
        M: GLWEBytesOf<BE> + ModuleLogN + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = BE::ZnxWord> + GLWEEncryptSk<BE>,
        E: EncryptionInfos,
        for<'a> BE::BufMut<'a>: HostDataMut,
    {
        #[cfg(debug_assertions)]
        {
            assert!(module.n().is_multiple_of(T::BITS as usize));
            assert_eq!(self.n(), module.n() as u32);
            assert_eq!(sk_glwe.n(), module.n() as u32);
        }

        let mut data_bits: Vec<i64> = vec![0i64; module.n()];

        let log_gap: usize = module.log_n() - T::LOG_BITS as usize;

        // Interleaves bytes
        for i in 0..T::BITS as usize {
            data_bits[T::bit_index(i) << log_gap] = data.bit(i) as i64
        }

        let pt_infos = GLWEPlaintextLayout {
            n: self.n(),
            base2k: self.base2k(),
            k: 2_usize.into(),
        };

        let mut pt = module.glwe_plaintext_alloc_from_infos(&pt_infos);

        pt.encode_vec_i64(&data_bits, TorusPrecision(2));
        module.glwe_encrypt_sk(&mut self.bits, &pt, sk_glwe, enc_infos, source_xe, source_xa, scratch);
    }

    pub fn encrypt_sk_tmp_bytes<M, BE: Backend>(&self, module: &M) -> usize
    where
        M: GLWEBytesOf<BE> + ModuleLogN + GLWEEncryptSk<BE>,
    {
        let pt_infos = GLWEPlaintextLayout {
            n: self.n(),
            base2k: self.base2k(),
            k: 2_usize.into(),
        };
        module.glwe_plaintext_bytes_of_from_infos(&pt_infos) + module.glwe_encrypt_sk_tmp_bytes(self)
    }
}

impl<D: HostDataRef, T: UnsignedInteger + FromBits> FheUint<D, T, i64> {
    pub fn noise<S, M, BE>(&self, module: &M, want: u32, sk: &S, scratch: &mut ScratchArena<'_, BE>) -> Stats
    where
        BE: Backend<OwnedBuf: HostDataMut + HostDataRef, ZnxWord = i64> + HostBackend,
        Self: GLWEToBackendRef<BE, State = CoeffNormalized>,
        S: GLWESecretPreparedToBackendRef<BE> + GLWEInfos,
        M: GLWEBytesOf<BE>
            + ModuleLogN
            + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = BE::ZnxWord>
            + GLWEDecrypt<BE>
            + GLWENoise<BE>,
        for<'a> BE::BufRef<'a>: HostDataRef,
        for<'a> BE::BufMut<'a>: HostDataMut,
    {
        #[cfg(debug_assertions)]
        {
            assert!(module.n().is_multiple_of(T::BITS as usize));
            assert_eq!(self.n(), module.n() as u32);
            assert_eq!(sk.n(), module.n() as u32);
        }

        let pt_infos = GLWEPlaintextLayout {
            n: self.n(),
            base2k: self.base2k(),
            k: 2_usize.into(),
        };
        let mut pt: GLWEPlaintext<BE::OwnedBuf, BE::ZnxWord> = module.glwe_plaintext_alloc_from_infos(&pt_infos);
        let mut data_bits = vec![0i64; module.n()];
        let log_gap: usize = module.log_n() - T::LOG_BITS as usize;
        for i in 0..T::BITS as usize {
            data_bits[T::bit_index(i) << log_gap] = want.bit(i) as i64
        }
        pt.encode_vec_i64(&data_bits, TorusPrecision(2));
        let mut scratch_1 = scratch.borrow();
        module.glwe_noise(self, &pt, sk, &mut scratch_1)
    }

    pub fn decrypt<S, M, BE>(&self, module: &M, sk_glwe: &S, scratch: &mut ScratchArena<'_, BE>) -> T
    where
        BE: Backend<OwnedBuf: HostDataMut + HostDataRef, ZnxWord = i64> + HostBackend,
        Self: GLWEToBackendRef<BE, State = CoeffNormalized>,
        S: GLWESecretPreparedToBackendRef<BE> + GLWEInfos,
        M: GLWEBytesOf<BE> + ModuleLogN + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = BE::ZnxWord> + GLWEDecrypt<BE>,
        for<'a> BE::BufMut<'a>: HostDataMut,
    {
        #[cfg(debug_assertions)]
        {
            assert!(module.n().is_multiple_of(T::BITS as usize));
            assert_eq!(self.n(), module.n() as u32);
            assert_eq!(sk_glwe.n(), module.n() as u32);
        }

        let pt_infos = GLWEPlaintextLayout {
            n: self.n(),
            base2k: self.base2k(),
            k: 1_usize.into(),
        };

        // TODO(device): this decrypt helper still stages the plaintext in a
        // host-owned buffer because backend-mut plaintext scratch views are
        // not yet accepted end-to-end here.
        let mut pt: GLWEPlaintext<BE::OwnedBuf, BE::ZnxWord> = module.glwe_plaintext_alloc_from_infos(&pt_infos);
        let mut scratch_1 = scratch.borrow();
        module.glwe_decrypt(self, &mut pt, sk_glwe, &mut scratch_1);

        let mut data_bits: Vec<i64> = vec![0i64; module.n()];
        pt.decode_vec_i64(&mut data_bits, TorusPrecision(2));

        let mut bits: Vec<u8> = vec![0u8; T::BITS as usize];

        let log_gap: usize = module.log_n() - T::LOG_BITS as usize;

        // Retrives from interleaved bytes
        for i in 0..T::BITS as usize {
            bits[i] = data_bits[T::bit_index(i) << log_gap] as u8
        }

        T::from_bits(&bits)
    }

    pub fn decrypt_tmp_bytes<M, BE: Backend>(&self, module: &M) -> usize
    where
        M: GLWEBytesOf<BE> + ModuleLogN + GLWEDecrypt<BE>,
    {
        let pt_infos = GLWEPlaintextLayout {
            n: self.n(),
            base2k: self.base2k(),
            k: 1_usize.into(),
        };
        module.glwe_plaintext_bytes_of_from_infos(&pt_infos) + module.glwe_decrypt_tmp_bytes(self)
    }
}

impl<D: Data, T: UnsignedInteger> FheUint<D, T, i64> {
    /// Packs `Vec<GLWE(bit[i])>` into [`FheUint`].
    pub fn pack<G, M, H, BE>(&mut self, module: &M, mut bits: Vec<G>, keys: &H, scratch: &mut ScratchArena<'_, BE>)
    where
        BE: Backend<OwnedBuf = D, ZnxWord = i64>,
        G: GLWEToBackendMut<BE, State = CoeffNormalized> + GLWEInfos,
        M: GLWEBytesOf<BE> + ModuleLogN + GLWEPacking<BE> + GLWECopy<BE>,
        H: GetAutomorphismKey<BE>,
        GLWE<D, BE::ZnxWord>: GLWEToBackendMut<BE, State = CoeffNormalized>,
    {
        // Repacks the GLWE ciphertexts bits
        let log_gap: usize = module.log_n() - T::LOG_BITS as usize;
        let mut cts: HashMap<usize, &mut G> = HashMap::new();
        for (i, ct) in bits.iter_mut().enumerate().take(T::BITS as usize) {
            cts.insert(T::bit_index(i) << log_gap, ct);
        }

        module.glwe_pack(&mut self.bits, cts, log_gap, keys, scratch);
    }

    #[allow(clippy::too_many_arguments)]
    // Self <- ((a.rotate_right(dst<<4) & 0xFFFF_0000) | (b.rotate_right(src<<4) & 0x0000_FFFF)).rotate_left(dst<<4);
    pub fn splice_u16<A, B, H, M, BE>(
        &mut self,
        module: &M,
        dst: usize,
        src: usize,
        a: &A,
        b: &B,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        BE: Backend<OwnedBuf = D, ZnxWord = i64>,
        Self: GLWEToBackendMut<BE, State = CoeffNormalized>,
        A: GLWEToBackendRef<BE, State = CoeffNormalized> + GLWEInfos,
        B: GLWEToBackendRef<BE, State = CoeffNormalized> + GLWEInfos,
        H: GetAutomorphismKey<BE>,
        M: GLWEBytesOf<BE>
            + ModuleLogN
            + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = BE::ZnxWord>
            + GLWERotate<BE>
            + GLWETrace<BE>
            + GLWESub<BE>
            + GLWEAdd<BE>
            + GLWECopy<BE>
            + GLWENormalize<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeBDD<'a, T, BE>,
    {
        assert!(dst < (T::BITS >> 4) as usize);
        assert!(src < (T::BITS >> 4) as usize);

        let mut tmp: FheUint<BE::OwnedBuf, T, BE::ZnxWord> = FheUint::alloc_from_infos(module, self);
        let mut scratch_1 = scratch.borrow();
        tmp.splice_u8(module, dst << 1, src << 1, a, b, keys, &mut scratch_1);
        self.splice_u8(module, (dst << 1) + 1, (src << 1) + 1, &tmp, b, keys, &mut scratch_1);
    }

    #[allow(clippy::too_many_arguments)]
    // Self <- ((a.rotate_right(dst<<3) & 0xFFFF_FF00) | (b.rotate_right(src<<3) & 0x0000_00FF)).rotate_left(dst<<3);
    pub fn splice_u8<A, B, H, M, BE>(
        &mut self,
        module: &M,
        dst: usize,
        src: usize,
        a: &A,
        b: &B,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        BE: Backend<OwnedBuf = D, ZnxWord = i64>,
        Self: GLWEToBackendMut<BE, State = CoeffNormalized>,
        A: GLWEToBackendRef<BE, State = CoeffNormalized> + GLWEInfos,
        B: GLWEToBackendRef<BE, State = CoeffNormalized> + GLWEInfos,
        H: GetAutomorphismKey<BE>,
        M: GLWEBytesOf<BE>
            + ModuleLogN
            + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = BE::ZnxWord>
            + GLWERotate<BE>
            + GLWETrace<BE>
            + GLWESub<BE>
            + GLWEAdd<BE>
            + GLWECopy<BE>
            + GLWENormalize<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeBDD<'a, T, BE>,
    {
        assert!(dst < (T::BITS >> 3) as usize);
        assert!(src < (T::BITS >> 3) as usize);

        // 1) Zero the byte receiver
        let log_gap: usize = module.log_n() - T::LOG_BITS as usize;
        let trace_start = (T::LOG_BITS - T::LOG_BYTES) as usize;
        let rot: i64 = (T::bit_index(dst << 3) << log_gap) as i64;

        module.glwe_copy(self, a);

        self.zero_byte(module, dst, keys, scratch);

        // Isolate the byte to transfer from a
        let mut tmp_fhe_uint_byte: FheUint<BE::OwnedBuf, T, BE::ZnxWord> = FheUint::alloc_from_infos(module, b);
        let mut scratch_1 = scratch.borrow();

        // Move a[byte_a] into a[dst]
        module.glwe_rotate(-((T::bit_index(src << 3) << log_gap) as i64), &mut tmp_fhe_uint_byte, b);

        // Zeroes all other bytes
        module.glwe_trace_assign(&mut tmp_fhe_uint_byte, trace_start, keys, &mut scratch_1);

        // Moves back self[0] to self[byte_tg]
        module.glwe_rotate_assign(rot, &mut tmp_fhe_uint_byte, &mut scratch_1);

        // Add self[0] += a[0], then propagate the carries so `self` keeps its CoeffNormalized label.
        {
            let mut acc = GLWEToBackendMut::<BE>::to_backend_mut(self).borrowed_carry_view();
            module.glwe_add_assign(&mut &mut acc, &tmp_fhe_uint_byte);
        }
        module.glwe_normalize_assign(self, &mut scratch_1);
    }
}

impl<BE: Backend, D: Data, T: UnsignedInteger> GLWEToBackendRef<BE> for FheUint<D, T, BE::ZnxWord>
where
    GLWE<D, BE::ZnxWord>: GLWEToBackendRef<BE, State = CoeffNormalized>,
{
    type State = CoeffNormalized;
    fn to_backend_ref(&self) -> GLWE<<BE as Backend>::BufRef<'_>, <BE as Backend>::ZnxWord> {
        self.bits.to_backend_ref()
    }
}

impl<BE: Backend, D: Data, T: UnsignedInteger> GLWEToBackendMut<BE> for FheUint<D, T, BE::ZnxWord>
where
    GLWE<D, BE::ZnxWord>: GLWEToBackendMut<BE, State = CoeffNormalized>,
{
    fn to_backend_mut(&mut self) -> GLWE<<BE as Backend>::BufMut<'_>, <BE as Backend>::ZnxWord> {
        self.bits.to_backend_mut()
    }
}

#[doc(hidden)]
pub trait ScratchArenaTakeBDD<'a, T: UnsignedInteger, BE: Backend>
where
    Self: ScratchArenaTakeCore<'a, BE>,
{
    /// Carves a temporary [`FheUint`] from the scratch arena.
    ///
    /// Returns the temporary and the remaining scratch space.
    #[allow(dead_code)]
    fn take_fhe_uint<A>(self, infos: &A) -> (FheUint<BE::BufMut<'a>, T, BE::ZnxWord>, Self)
    where
        A: GLWEInfos,
    {
        let (glwe, scratch) = self.take_glwe_scratch(infos);
        (
            FheUint {
                bits: glwe.into_inner(),
                _phantom: PhantomData,
            },
            scratch,
        )
    }
}

impl<'a, T: UnsignedInteger, BE: Backend> ScratchArenaTakeBDD<'a, T, BE> for ScratchArena<'a, BE> where
    Self: ScratchArenaTakeCore<'a, BE>
{
}

impl<D: Data, T: UnsignedInteger, W: ZnxWord> FheUint<D, T, W> {
    pub fn get_bit_lwe<R, M, BE>(
        &self,
        module: &M,
        bit: usize,
        res: &mut R,
        ks_glwe: Option<&GGLWEPreparedBackendRef<'_, BE>>,
        ks_lwe: &GGLWEPreparedBackendRef<'_, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        BE: Backend<OwnedBuf = D, ZnxWord = i64>,
        R: LWEToBackendMut<BE> + LWEInfos,
        Self: GLWEToBackendRef<BE, State = CoeffNormalized>,
        M: GLWEBytesOf<BE>
            + ModuleLogN
            + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = BE::ZnxWord>
            + LWEFromGLWE<BE>
            + GLWEKeyswitch<BE>,
    {
        let log_gap: usize = module.log_n() - T::LOG_BITS as usize;
        if let Some(ks_glwe) = ks_glwe {
            let mut res_tmp: GLWE<BE::OwnedBuf, BE::ZnxWord> =
                module.glwe_alloc(ks_glwe.base2k(), ks_glwe.k(), ks_glwe.rank_out());
            let mut scratch_1 = scratch.borrow();
            {
                let mut scratch_op = scratch_1.borrow();
                module.glwe_keyswitch(&mut res_tmp, self, ks_glwe, &mut scratch_op);
            }
            let mut scratch_op = scratch_1.borrow();
            module.lwe_from_glwe(res, &res_tmp, T::bit_index(bit) << log_gap, ks_lwe, &mut scratch_op);
        } else {
            module.lwe_from_glwe(res, self, T::bit_index(bit) << log_gap, ks_lwe, scratch);
        }
    }

    pub fn get_bit_glwe<R, M, H, BE>(&self, module: &M, bit: usize, res: &mut R, keys: &H, scratch: &mut ScratchArena<'_, BE>)
    where
        BE: Backend,
        R: GLWEToBackendMut<BE, State = CoeffNormalized> + GLWEInfos,
        Self: GLWEToBackendRef<BE, State = CoeffNormalized>,
        M: GLWEBytesOf<BE> + ModuleLogN + GLWERotate<BE> + GLWETrace<BE>,
        H: GetAutomorphismKey<BE>,
    {
        let log_gap: usize = module.log_n() - T::LOG_BITS as usize;
        let rot = (T::bit_index(bit) << log_gap) as i64;
        module.glwe_rotate(-rot, res, self);
        module.glwe_trace_assign(res, 0, keys, scratch);
    }

    pub fn get_byte<R, M, H, BE>(&self, module: &M, byte: usize, res: &mut R, keys: &H, scratch: &mut ScratchArena<'_, BE>)
    where
        BE: Backend,
        R: GLWEToBackendMut<BE, State = CoeffNormalized> + GLWEInfos,
        Self: GLWEToBackendRef<BE, State = CoeffNormalized>,
        M: GLWEBytesOf<BE> + ModuleLogN + GLWERotate<BE> + GLWETrace<BE>,
        H: GetAutomorphismKey<BE>,
    {
        let log_gap: usize = module.log_n() - T::LOG_BITS as usize;
        let trace_start = (T::LOG_BITS - T::LOG_BYTES) as usize;
        let rot = (T::bit_index(byte << 3) << log_gap) as i64;
        module.glwe_rotate(-rot, res, self);
        module.glwe_trace_assign(res, trace_start, keys, scratch);
    }
}

impl<T: UnsignedInteger> FheUint<Vec<u8>, T, i64> {
    pub fn from_fhe_uint_prepared<M, H, BE>(
        &mut self,
        module: &M,
        other: &FheUintPrepared<BE::OwnedBuf, T, BE>,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        BE: Backend<OwnedBuf = Vec<u8>, ZnxWord = i64> + 'static,
        M: GLWEBytesOf<BE>
            + Cmux<BE>
            + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = BE::ZnxWord>
            + ModuleLogN
            + GLWEPacking<BE>
            + GLWECopy<BE>,
        GLWE<Vec<u8>, BE::ZnxWord>: GLWEToBackendMut<BE, State = CoeffNormalized>,
        Self: GLWEToBackendMut<BE, State = CoeffNormalized>,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeBDD<'a, T, BE>,
        H: GetAutomorphismKey<BE>,
        for<'a> BE::BufMut<'a>: HostDataMut,
        for<'a> BE: Backend<BufMut<'a> = &'a mut [u8], BufRef<'a> = &'a [u8]>,
    {
        let zero: GLWE<BE::OwnedBuf, BE::ZnxWord> = module.glwe_alloc_from_infos(self);
        let mut one: GLWE<BE::OwnedBuf, BE::ZnxWord> = module.glwe_alloc_from_infos(self);
        one.data_mut().encode_coeff_i64(self.base2k().into(), 0, 2, 0, 1);

        // TODO(device): this conversion still expands prepared bits into
        // host-owned temporary GLWEs before packing.
        let mut out_bits: Vec<GLWE<BE::OwnedBuf, BE::ZnxWord>> =
            (0..T::BITS as usize).map(|_| module.glwe_alloc_from_infos(self)).collect();
        let mut scratch_1 = scratch.borrow();

        for (i, bits) in out_bits.iter_mut().enumerate().take(T::BITS as usize) {
            module.cmux(bits, &one, &zero, &other.get_bit(i).to_backend_ref(), &mut scratch_1.borrow());
        }

        self.pack(module, out_bits, keys, &mut scratch_1);
    }
}

impl<D: Data, T: UnsignedInteger> FheUint<D, T, i64> {
    pub fn zero_byte<M, H, BE>(&mut self, module: &M, byte: usize, keys: &H, scratch: &mut ScratchArena<'_, BE>)
    where
        BE: Backend<OwnedBuf = D, ZnxWord = i64>,
        Self: GLWEToBackendMut<BE, State = CoeffNormalized>,
        H: GetAutomorphismKey<BE>,
        M: GLWEBytesOf<BE>
            + ModuleLogN
            + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = BE::ZnxWord>
            + GLWERotate<BE>
            + GLWETrace<BE>
            + GLWESub<BE>
            + GLWEAdd<BE>
            + GLWECopy<BE>
            + GLWENormalize<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeBDD<'a, T, BE>,
    {
        let log_gap: usize = module.log_n() - T::LOG_BITS as usize;
        let trace_start = (T::LOG_BITS - T::LOG_BYTES) as usize;
        let rot: i64 = (T::bit_index(byte << 3) << log_gap) as i64;

        // Move a to self and align byte
        module.glwe_rotate_assign(-rot, self, scratch);

        // Stores this byte (everything else zeroed) into tmp_trace
        let mut tmp_trace: GLWE<BE::OwnedBuf, BE::ZnxWord> = module.glwe_alloc_from_infos(self);
        module.glwe_trace(&mut tmp_trace, trace_start, self, keys, scratch);

        // Subtracts to self to zero it, then propagate the carries so `self` keeps its CoeffNormalized label.
        {
            let mut acc = GLWEToBackendMut::<BE>::to_backend_mut(self).borrowed_carry_view();
            module.glwe_sub_assign(&mut &mut acc, &tmp_trace);
        }
        module.glwe_normalize_assign(self, scratch);

        // Move a to self and align byte
        module.glwe_rotate_assign(rot, self, scratch);
    }

    pub fn sext<M, H, BE>(&mut self, module: &M, byte: usize, keys: &H, scratch: &mut ScratchArena<'_, BE>)
    where
        Self: GLWEToBackendRef<BE, State = CoeffNormalized>,
        Self: GLWEToBackendMut<BE, State = CoeffNormalized>,
        H: GetAutomorphismKey<BE>,
        D: poulpy_hal::layouts::DataOwned,
        BE: Backend<OwnedBuf = D, ZnxWord = i64>,
        M: GLWEBytesOf<BE>
            + ModuleLogN
            + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = BE::ZnxWord>
            + GLWERotate<BE>
            + GLWETrace<BE>
            + GLWEAdd<BE>
            + GLWESub<BE>
            + GLWECopy<BE>
            + GLWENormalize<BE>
            + VecZnxNormalizeAssignBackend<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeBDD<'a, T, BE>,
    {
        assert!(byte < (1 << T::LOG_BYTES));

        let log_gap: usize = module.log_n() - T::LOG_BITS as usize;
        let rot: i64 = (T::bit_index((byte << 3) + 7) << log_gap) as i64;

        let mut sext: GLWE<BE::OwnedBuf, BE::ZnxWord> = module.glwe_alloc_from_infos(self);
        let mut scratch_1 = scratch.borrow();

        // Extract MSB
        module.glwe_rotate(-rot, &mut sext, self);
        module.glwe_trace_assign(&mut sext, 0, keys, &mut scratch_1.borrow());

        // Replicates MSB in byte: accumulate the rotations, then normalize once.
        let mut sext = sext.into_unnormalized();
        for i in 0..3 {
            let mut tmp = module.glwe_alloc_from_infos(&sext).into_unnormalized();
            module.glwe_rotate(((1 << T::LOG_BYTES) << log_gap) << i, &mut tmp, &sext);
            module.glwe_add_assign(&mut sext, &tmp);
        }
        let sext = sext.normalize(module, &mut scratch_1.borrow());

        // Splice sext
        let mut tmp: FheUint<BE::OwnedBuf, T, BE::ZnxWord> = FheUint::alloc_from_infos(module, self);
        let mut current: GLWE<BE::OwnedBuf, BE::ZnxWord> = module.glwe_alloc_from_infos(self);
        module.glwe_copy(&mut current, self);
        for i in (byte + 1)..(1 << T::LOG_BYTES) as usize {
            tmp.splice_u8(module, i, 0, &current, &sext, keys, &mut scratch_1);
            module.glwe_copy(&mut current, &tmp.bits);
        }
        module.glwe_copy(self, &current);
    }
}
