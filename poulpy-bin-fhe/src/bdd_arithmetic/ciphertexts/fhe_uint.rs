use poulpy_core::{
    EncryptionInfos, GLWEAdd, GLWECopy, GLWEDecrypt, GLWEEncryptSk, GLWEKeyswitch, GLWENoise, GLWEPacking, GLWERotate, GLWESub,
    GLWETrace, LWEFromGLWE, ScratchArenaTakeCore,
    layouts::{
        Base2K, GGLWEInfos, GGLWEPreparedToBackendRef, GLWE, GLWEAutomorphismKeyHelper, GLWEInfos, GLWEPlaintext,
        GLWEPlaintextLayout, GLWESecretPreparedToBackendRef, GLWEToBackendMut, GLWEToBackendRef, GetGaloisElement, LWEInfos,
        LWEToBackendMut, ModuleCoreAlloc, Rank, TorusPrecision,
    },
};
use poulpy_hal::{
    api::{ModuleLogN, ModuleN},
    layouts::{Backend, Data, HostBackend, HostDataMut, HostDataRef, ScratchArena, Stats},
    source::Source,
};
use std::{collections::HashMap, marker::PhantomData};

use crate::bdd_arithmetic::{Cmux, FheUintPrepared, FromBits, GetGGSWBit, ToBits, UnsignedInteger};

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
pub struct FheUint<D: Data, T: UnsignedInteger> {
    pub(crate) bits: GLWE<D>,
    pub(crate) _phantom: PhantomData<T>,
}

impl<D: Data, T: UnsignedInteger> FheUint<D, T> {
    pub fn alloc_from_infos<M, A>(module: &M, infos: &A) -> Self
    where
        M: ModuleCoreAlloc<OwnedBuf = D> + ModuleN,
        A: GLWEInfos,
    {
        #[cfg(debug_assertions)]
        {
            assert_eq!(module.n(), infos.n().as_usize());
        }

        Self::alloc(module, infos.base2k(), infos.max_k(), infos.rank())
    }

    pub fn alloc<M>(module: &M, base2k: Base2K, k: TorusPrecision, rank: Rank) -> Self
    where
        M: ModuleCoreAlloc<OwnedBuf = D> + ModuleN,
    {
        Self {
            bits: module.glwe_alloc(base2k, k, rank),
            _phantom: PhantomData,
        }
    }
}

impl<'a, T: UnsignedInteger> FheUint<&'a mut [u8], T> {
    pub fn from_glwe_to_mut<G>(glwe: &'a mut G) -> Self
    where
        G: GLWEToBackendMut<poulpy_hal::layouts::HostBytesBackend>,
    {
        FheUint {
            bits: glwe.to_backend_mut(),
            _phantom: PhantomData,
        }
    }
}

impl<'a, T: UnsignedInteger> FheUint<&'a [u8], T> {
    pub fn from_glwe_to_ref<G>(glwe: &'a G) -> Self
    where
        G: GLWEToBackendRef<poulpy_hal::layouts::HostBytesBackend>,
    {
        FheUint {
            bits: glwe.to_backend_ref(),
            _phantom: PhantomData,
        }
    }
}

impl<D: HostDataRef, T: UnsignedInteger> LWEInfos for FheUint<D, T> {
    fn base2k(&self) -> poulpy_core::layouts::Base2K {
        self.bits.base2k()
    }

    fn size(&self) -> usize {
        self.bits.size()
    }

    fn n(&self) -> poulpy_core::layouts::Degree {
        self.bits.n()
    }
}

impl<D: HostDataRef, T: UnsignedInteger> GLWEInfos for FheUint<D, T> {
    fn rank(&self) -> poulpy_core::layouts::Rank {
        self.bits.rank()
    }
}

impl<D: HostDataMut, T: UnsignedInteger + ToBits> FheUint<D, T> {
    #[allow(clippy::too_many_arguments)]
    pub fn encrypt_sk<'s, S, M, E, BE>(
        &mut self,
        module: &M,
        data: T,
        sk_glwe: &S,
        enc_infos: &E,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        BE: Backend<OwnedBuf = Vec<u8>> + 's,
        GLWE<D>: GLWEToBackendMut<BE>,
        S: GLWESecretPreparedToBackendRef<BE> + GLWEInfos,
        M: ModuleLogN + ModuleCoreAlloc<OwnedBuf = Vec<u8>> + GLWEEncryptSk<BE>,
        E: EncryptionInfos,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
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
        M: ModuleLogN + GLWEEncryptSk<BE>,
    {
        let pt_infos = GLWEPlaintextLayout {
            n: self.n(),
            base2k: self.base2k(),
            k: 2_usize.into(),
        };
        GLWEPlaintext::<Vec<u8>>::bytes_of_from_infos(&pt_infos) + module.glwe_encrypt_sk_tmp_bytes(self)
    }
}

impl<D: HostDataRef, T: UnsignedInteger + FromBits> FheUint<D, T> {
    pub fn noise<S, M, BE>(&self, module: &M, want: u32, sk: &S, scratch: &mut ScratchArena<'_, BE>) -> Stats
    where
        BE: Backend<OwnedBuf = Vec<u8>> + HostBackend,
        Self: GLWEToBackendRef<BE>,
        S: GLWESecretPreparedToBackendRef<BE> + GLWEInfos,
        M: ModuleLogN + ModuleCoreAlloc<OwnedBuf = Vec<u8>> + GLWEDecrypt<BE> + GLWENoise<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
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
        let mut pt: GLWEPlaintext<Vec<u8>> = module.glwe_plaintext_alloc_from_infos(&pt_infos);
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
        BE: Backend<OwnedBuf = Vec<u8>> + HostBackend,
        Self: GLWEToBackendRef<BE>,
        S: GLWESecretPreparedToBackendRef<BE> + GLWEInfos,
        M: ModuleLogN + ModuleCoreAlloc<OwnedBuf = Vec<u8>> + GLWEDecrypt<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
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
        let mut pt: GLWEPlaintext<Vec<u8>> = module.glwe_plaintext_alloc_from_infos(&pt_infos);
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
        M: ModuleLogN + GLWEDecrypt<BE>,
    {
        let pt_infos = GLWEPlaintextLayout {
            n: self.n(),
            base2k: self.base2k(),
            k: 1_usize.into(),
        };
        GLWEPlaintext::<Vec<u8>>::bytes_of_from_infos(&pt_infos) + module.glwe_decrypt_tmp_bytes(self)
    }
}

impl<D: HostDataMut, T: UnsignedInteger> FheUint<D, T> {
    /// Packs `Vec<GLWE(bit[i])>` into [`FheUint`].
    pub fn pack<'s, G, M, K, H, BE>(&mut self, module: &M, mut bits: Vec<G>, keys: &H, scratch: &mut ScratchArena<'s, BE>)
    where
        BE: Backend<OwnedBuf = Vec<u8>> + 's,
        G: GLWEToBackendMut<BE> + GLWEInfos,
        M: ModuleLogN + GLWEPacking<BE> + GLWECopy<BE>,
        K: GGLWEPreparedToBackendRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>,
        GLWE<D>: GLWEToBackendMut<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
        for<'a> BE::BufMut<'a>: HostDataMut,
    {
        // Repacks the GLWE ciphertexts bits
        let log_gap: usize = module.log_n() - T::LOG_BITS as usize;
        let mut cts: HashMap<usize, &mut G> = HashMap::new();
        for (i, ct) in bits.iter_mut().enumerate().take(T::BITS as usize) {
            cts.insert(T::bit_index(i) << log_gap, ct);
        }

        module.glwe_pack(
            &mut self.bits,
            cts,
            log_gap,
            keys,
            keys.automorphism_key_infos().size(),
            scratch,
        );
    }

    #[allow(clippy::too_many_arguments)]
    // Self <- ((a.rotate_right(dst<<4) & 0xFFFF_0000) | (b.rotate_right(src<<4) & 0x0000_FFFF)).rotate_left(dst<<4);
    pub fn splice_u16<A, B, H, K, M, BE>(
        &mut self,
        module: &M,
        dst: usize,
        src: usize,
        a: &A,
        b: &B,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        BE: Backend<OwnedBuf = Vec<u8>>,
        Self: GLWEToBackendMut<BE>,
        A: GLWEToBackendRef<BE> + GLWEInfos,
        B: GLWEToBackendRef<BE> + GLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>,
        K: GGLWEPreparedToBackendRef<BE> + GGLWEInfos + GetGaloisElement,
        M: ModuleLogN
            + ModuleCoreAlloc<OwnedBuf = Vec<u8>>
            + GLWERotate<BE>
            + GLWETrace<BE>
            + GLWESub<BE>
            + GLWEAdd<BE>
            + GLWECopy<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeBDD<'a, T, BE>,
        for<'a> BE::BufMut<'a>: HostDataMut,
    {
        assert!(dst < (T::BITS >> 4) as usize);
        assert!(src < (T::BITS >> 4) as usize);

        let mut tmp: FheUint<BE::OwnedBuf, T> = FheUint::alloc_from_infos(module, self);
        let mut scratch_1 = scratch.borrow();
        tmp.splice_u8(module, dst << 1, src << 1, a, b, keys, &mut scratch_1);
        self.splice_u8(module, (dst << 1) + 1, (src << 1) + 1, &tmp, b, keys, &mut scratch_1);
    }

    #[allow(clippy::too_many_arguments)]
    // Self <- ((a.rotate_right(dst<<3) & 0xFFFF_FF00) | (b.rotate_right(src<<3) & 0x0000_00FF)).rotate_left(dst<<3);
    pub fn splice_u8<'s, A, B, H, K, M, BE>(
        &mut self,
        module: &M,
        dst: usize,
        src: usize,
        a: &A,
        b: &B,
        keys: &H,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        BE: Backend<OwnedBuf = Vec<u8>> + 's,
        Self: GLWEToBackendMut<BE>,
        A: GLWEToBackendRef<BE> + GLWEInfos,
        B: GLWEToBackendRef<BE> + GLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>,
        K: GGLWEPreparedToBackendRef<BE> + GGLWEInfos + GetGaloisElement,
        M: ModuleLogN
            + ModuleCoreAlloc<OwnedBuf = Vec<u8>>
            + GLWERotate<BE>
            + GLWETrace<BE>
            + GLWESub<BE>
            + GLWEAdd<BE>
            + GLWECopy<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeBDD<'a, T, BE>,
        for<'a> BE::BufMut<'a>: HostDataMut,
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
        // TODO(device): this byte splice still relies on a host-owned
        // temporary packed ciphertext to satisfy the backend-native rotate API.
        let mut tmp_fhe_uint_byte: FheUint<BE::OwnedBuf, T> = FheUint::alloc_from_infos(module, b);
        let mut scratch_1 = scratch.borrow();

        // Move a[byte_a] into a[dst]
        module.glwe_rotate(-((T::bit_index(src << 3) << log_gap) as i64), &mut tmp_fhe_uint_byte, b);

        // Zeroes all other bytes
        module.glwe_trace_assign(
            &mut tmp_fhe_uint_byte,
            trace_start,
            keys,
            keys.automorphism_key_infos().size(),
            &mut scratch_1,
        );

        // Moves back self[0] to self[byte_tg]
        module.glwe_rotate_assign(rot, &mut tmp_fhe_uint_byte, &mut scratch_1);

        // Add self[0] += a[0]
        module.glwe_add_assign(self, &tmp_fhe_uint_byte);
    }
}

impl<BE: Backend, D: Data, T: UnsignedInteger> GLWEToBackendRef<BE> for FheUint<D, T>
where
    GLWE<D>: GLWEToBackendRef<BE>,
{
    fn to_backend_ref(&self) -> GLWE<<BE as Backend>::BufRef<'_>> {
        self.bits.to_backend_ref()
    }
}

impl<BE: Backend, D: Data, T: UnsignedInteger> GLWEToBackendMut<BE> for FheUint<D, T>
where
    GLWE<D>: GLWEToBackendMut<BE>,
{
    fn to_backend_mut(&mut self) -> GLWE<<BE as Backend>::BufMut<'_>> {
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
    fn take_fhe_uint<A>(self, infos: &A) -> (FheUint<BE::BufMut<'a>, T>, Self)
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

impl<D: HostDataRef, T: UnsignedInteger> FheUint<D, T> {
    pub fn get_bit_lwe<'s, R, KGLWE, KLWE, M, BE>(
        &self,
        module: &M,
        bit: usize,
        res: &mut R,
        ks_glwe: Option<&KGLWE>,
        ks_lwe: &KLWE,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        BE: Backend<OwnedBuf = Vec<u8>> + 's,
        R: LWEToBackendMut<BE> + LWEInfos,
        Self: GLWEToBackendRef<BE>,
        KGLWE: GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
        KLWE: GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
        M: ModuleLogN + ModuleCoreAlloc<OwnedBuf = Vec<u8>> + LWEFromGLWE<BE> + GLWEKeyswitch<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
        for<'a> BE::BufMut<'a>: HostDataMut,
    {
        let log_gap: usize = module.log_n() - T::LOG_BITS as usize;
        if let Some(ks_glwe) = ks_glwe {
            // TODO(device): this extraction path still stages the rank-1 GLWE
            // in host-owned memory for compatibility with the current core API.
            let mut res_tmp: GLWE<BE::OwnedBuf> = module.glwe_alloc(ks_glwe.base2k(), ks_glwe.max_k(), ks_glwe.rank_out());
            let mut scratch_1 = scratch.borrow();
            {
                let mut scratch_op = scratch_1.borrow();
                module.glwe_keyswitch(&mut res_tmp, self, ks_glwe, ks_glwe.size(), &mut scratch_op);
            }
            let mut scratch_op = scratch_1.borrow();
            module.lwe_from_glwe(
                res,
                &res_tmp,
                T::bit_index(bit) << log_gap,
                ks_lwe,
                ks_lwe.size(),
                &mut scratch_op,
            );
        } else {
            module.lwe_from_glwe(res, self, T::bit_index(bit) << log_gap, ks_lwe, ks_lwe.size(), scratch);
        }
    }

    pub fn get_bit_glwe<'s, R, K, M, H, BE>(
        &self,
        module: &M,
        bit: usize,
        res: &mut R,
        keys: &H,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        BE: Backend + 's,
        R: GLWEToBackendMut<BE> + GLWEInfos,
        Self: GLWEToBackendRef<BE>,
        K: GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
        M: ModuleLogN + GLWERotate<BE> + GLWETrace<BE>,
        H: GLWEAutomorphismKeyHelper<K, BE>,
        K: GGLWEPreparedToBackendRef<BE> + GGLWEInfos + GetGaloisElement,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
        for<'a> BE::BufMut<'a>: HostDataMut,
    {
        let log_gap: usize = module.log_n() - T::LOG_BITS as usize;
        let rot = (T::bit_index(bit) << log_gap) as i64;
        module.glwe_rotate(-rot, res, self);
        module.glwe_trace_assign(res, 0, keys, keys.automorphism_key_infos().size(), scratch);
    }

    pub fn get_byte<'s, R, K, M, H, BE>(&self, module: &M, byte: usize, res: &mut R, keys: &H, scratch: &mut ScratchArena<'s, BE>)
    where
        BE: Backend + 's,
        R: GLWEToBackendMut<BE> + GLWEInfos,
        Self: GLWEToBackendRef<BE>,
        K: GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
        M: ModuleLogN + GLWERotate<BE> + GLWETrace<BE>,
        H: GLWEAutomorphismKeyHelper<K, BE>,
        K: GGLWEPreparedToBackendRef<BE> + GGLWEInfos + GetGaloisElement,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
        for<'a> BE::BufMut<'a>: HostDataMut,
    {
        let log_gap: usize = module.log_n() - T::LOG_BITS as usize;
        let trace_start = (T::LOG_BITS - T::LOG_BYTES) as usize;
        let rot = (T::bit_index(byte << 3) << log_gap) as i64;
        module.glwe_rotate(-rot, res, self);
        module.glwe_trace_assign(res, trace_start, keys, keys.automorphism_key_infos().size(), scratch);
    }
}

impl<D: HostDataMut, T: UnsignedInteger> FheUint<D, T> {
    pub fn from_fhe_uint_prepared<M, H, K, BE>(
        &mut self,
        module: &M,
        other: &FheUintPrepared<BE::OwnedBuf, T, BE>,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        BE: Backend<OwnedBuf = Vec<u8>> + 'static,
        M: Cmux<BE> + ModuleCoreAlloc<OwnedBuf = Vec<u8>> + ModuleLogN + GLWEPacking<BE> + GLWECopy<BE>,
        GLWE<D>: GLWEToBackendMut<BE>,
        Self: GLWEToBackendMut<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeBDD<'a, T, BE>,
        K: GGLWEPreparedToBackendRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>,
        for<'a> BE::BufMut<'a>: HostDataMut,
        for<'a> BE: Backend<BufMut<'a> = &'a mut [u8], BufRef<'a> = &'a [u8]>,
    {
        let zero: GLWE<BE::OwnedBuf> = module.glwe_alloc_from_infos(self);
        let mut one: GLWE<BE::OwnedBuf> = module.glwe_alloc_from_infos(self);
        one.data_mut().encode_coeff_i64(self.base2k().into(), 0, 2, 0, 1);

        // TODO(device): this conversion still expands prepared bits into
        // host-owned temporary GLWEs before packing.
        let mut out_bits: Vec<GLWE<BE::OwnedBuf>> = (0..T::BITS as usize).map(|_| module.glwe_alloc_from_infos(self)).collect();
        let mut scratch_1 = scratch.borrow();

        for (i, bits) in out_bits.iter_mut().enumerate().take(T::BITS as usize) {
            module.cmux(bits, &one, &zero, other.get_bit(i), &mut scratch_1.borrow());
        }

        self.pack(module, out_bits, keys, &mut scratch_1);
    }

    pub fn zero_byte<'s, M, K, H, BE>(&mut self, module: &M, byte: usize, keys: &H, scratch: &mut ScratchArena<'s, BE>)
    where
        BE: Backend<OwnedBuf = Vec<u8>> + 's,
        Self: GLWEToBackendMut<BE>,
        H: GLWEAutomorphismKeyHelper<K, BE>,
        K: GGLWEPreparedToBackendRef<BE> + GGLWEInfos + GetGaloisElement,
        M: ModuleLogN
            + ModuleCoreAlloc<OwnedBuf = Vec<u8>>
            + GLWERotate<BE>
            + GLWETrace<BE>
            + GLWESub<BE>
            + GLWEAdd<BE>
            + GLWECopy<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeBDD<'a, T, BE>,
        for<'a> BE::BufMut<'a>: HostDataMut,
    {
        let log_gap: usize = module.log_n() - T::LOG_BITS as usize;
        let trace_start = (T::LOG_BITS - T::LOG_BYTES) as usize;
        let rot: i64 = (T::bit_index(byte << 3) << log_gap) as i64;

        // Move a to self and align byte
        module.glwe_rotate_assign(-rot, self, scratch);

        // Stores this byte (everything else zeroed) into tmp_trace
        let mut tmp_trace: GLWE<BE::OwnedBuf> = module.glwe_alloc_from_infos(self);
        module.glwe_trace(
            &mut tmp_trace,
            trace_start,
            self,
            keys,
            keys.automorphism_key_infos().size(),
            scratch,
        );

        // Subtracts to self to zero it
        module.glwe_sub_assign(self, &tmp_trace);

        // Move a to self and align byte
        module.glwe_rotate_assign(rot, self, scratch);
    }

    pub fn sext<'s, M, H, K, BE>(&mut self, module: &M, byte: usize, keys: &H, scratch: &mut ScratchArena<'s, BE>)
    where
        Self: GLWEToBackendRef<BE>,
        Self: GLWEToBackendMut<BE>,
        H: GLWEAutomorphismKeyHelper<K, BE>,
        K: GGLWEPreparedToBackendRef<BE> + GGLWEInfos + GetGaloisElement,
        BE: Backend<OwnedBuf = Vec<u8>>,
        M: ModuleLogN
            + ModuleCoreAlloc<OwnedBuf = Vec<u8>>
            + GLWERotate<BE>
            + GLWETrace<BE>
            + GLWEAdd<BE>
            + GLWESub<BE>
            + GLWECopy<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeBDD<'a, T, BE>,
        for<'a> BE::BufMut<'a>: HostDataMut,
        BE: 's,
    {
        assert!(byte < (1 << T::LOG_BYTES));

        let log_gap: usize = module.log_n() - T::LOG_BITS as usize;
        let rot: i64 = (T::bit_index((byte << 3) + 7) << log_gap) as i64;

        let mut sext: GLWE<BE::OwnedBuf> = module.glwe_alloc_from_infos(self);
        let mut scratch_1 = scratch.borrow();

        // Extract MSB
        module.glwe_rotate(-rot, &mut sext, self);
        module.glwe_trace_assign(
            &mut sext,
            0,
            keys,
            keys.automorphism_key_infos().size(),
            &mut scratch_1.borrow(),
        );

        // Replicates MSB in byte
        for i in 0..3 {
            let mut tmp: GLWE<BE::OwnedBuf> = module.glwe_alloc_from_infos(self);
            module.glwe_rotate(((1 << T::LOG_BYTES) << log_gap) << i, &mut tmp, &sext);
            module.glwe_add_assign(&mut sext, &tmp);
        }

        // Splice sext
        let mut tmp: FheUint<BE::OwnedBuf, T> = FheUint::alloc_from_infos(module, self);
        let mut current: GLWE<BE::OwnedBuf> = module.glwe_alloc_from_infos(self);
        module.glwe_copy(&mut current, self);
        for i in (byte + 1)..(1 << T::LOG_BYTES) as usize {
            tmp.splice_u8(module, i, 0, &current, &sext, keys, &mut scratch_1);
            module.glwe_copy(&mut current, &tmp.bits);
        }
        module.glwe_copy(self, &current);
    }
}
