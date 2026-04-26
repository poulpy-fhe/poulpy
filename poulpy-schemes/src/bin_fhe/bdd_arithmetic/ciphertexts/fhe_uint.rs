use poulpy_core::{
    EncryptionInfos, GLWEAdd, GLWECopy, GLWEDecrypt, GLWEEncryptSk, GLWEKeyswitch, GLWENoise, GLWEPacking, GLWERotate, GLWESub,
    GLWETrace, LWEFromGLWE, ScratchTakeCore,
    layouts::{
        Base2K, Degree, GGLWEInfos, GGLWEPreparedToRef, GLWE, GLWEAutomorphismKeyHelper, GLWEInfos, GLWELayout, GLWEPlaintext,
        GLWEPlaintextLayout, GLWESecretPreparedToRef, GLWEToMut, GLWEToRef, GetGaloisElement, LWEInfos, LWEToMut, Rank,
        TorusPrecision,
    },
};
use poulpy_hal::{
    api::ModuleLogN,
    layouts::{Backend, Data, DataMut, DataRef, Scratch, Stats},
    source::Source,
};
use std::{collections::HashMap, marker::PhantomData};

use crate::bin_fhe::bdd_arithmetic::{Cmux, FheUintPrepared, FromBits, GetGGSWBit, ToBits, UnsignedInteger};

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

impl<T: UnsignedInteger> FheUint<Vec<u8>, T> {
    pub fn alloc_from_infos<A>(infos: &A) -> Self
    where
        A: GLWEInfos,
    {
        Self::alloc(infos.n(), infos.base2k(), infos.max_k(), infos.rank())
    }

    pub fn alloc(n: Degree, base2k: Base2K, k: TorusPrecision, rank: Rank) -> Self {
        Self {
            bits: GLWE::alloc(n, base2k, k, rank),
            _phantom: PhantomData,
        }
    }
}

impl<'a, T: UnsignedInteger> FheUint<&'a mut [u8], T> {
    pub fn from_glwe_to_mut<G>(glwe: &'a mut G) -> Self
    where
        G: GLWEToMut,
    {
        FheUint {
            bits: glwe.to_mut(),
            _phantom: PhantomData,
        }
    }
}

impl<'a, T: UnsignedInteger> FheUint<&'a [u8], T> {
    pub fn from_glwe_to_ref<G>(glwe: &'a G) -> Self
    where
        G: GLWEToRef,
    {
        FheUint {
            bits: glwe.to_ref(),
            _phantom: PhantomData,
        }
    }
}

impl<D: DataRef, T: UnsignedInteger> LWEInfos for FheUint<D, T> {
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

impl<D: DataRef, T: UnsignedInteger> GLWEInfos for FheUint<D, T> {
    fn rank(&self) -> poulpy_core::layouts::Rank {
        self.bits.rank()
    }
}

impl<D: DataMut, T: UnsignedInteger + ToBits> FheUint<D, T> {
    #[allow(clippy::too_many_arguments)]
    pub fn encrypt_sk<S, M, E, BE: Backend>(
        &mut self,
        module: &M,
        data: T,
        sk_glwe: &S,
        enc_infos: &E,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut Scratch<BE>,
    ) where
        S: GLWESecretPreparedToRef<BE> + GLWEInfos,
        M: ModuleLogN + GLWEEncryptSk<BE>,
        E: EncryptionInfos,
        Scratch<BE>: ScratchTakeCore<BE>,
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

        let (mut pt, scratch_1) = scratch.take_glwe_plaintext(&pt_infos);

        pt.encode_vec_i64(&data_bits, TorusPrecision(2));
        module.glwe_encrypt_sk(&mut self.bits, &pt, sk_glwe, enc_infos, source_xe, source_xa, scratch_1);
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

impl<D: DataRef, T: UnsignedInteger + FromBits> FheUint<D, T> {
    pub fn noise<S, M, BE: Backend>(&self, module: &M, want: u32, sk: &S, scratch: &mut Scratch<BE>) -> Stats
    where
        S: GLWESecretPreparedToRef<BE> + GLWEInfos,
        M: ModuleLogN + GLWEDecrypt<BE> + GLWENoise<BE>,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        #[cfg(debug_assertions)]
        {
            assert!(module.n().is_multiple_of(T::BITS as usize));
            assert_eq!(self.n(), module.n() as u32);
            assert_eq!(sk.n(), module.n() as u32);
        }

        let (mut pt, scratch_1) = scratch.take_glwe_plaintext(self);
        let mut data_bits = vec![0i64; module.n()];
        let log_gap: usize = module.log_n() - T::LOG_BITS as usize;
        for i in 0..T::BITS as usize {
            data_bits[T::bit_index(i) << log_gap] = want.bit(i) as i64
        }
        pt.encode_vec_i64(&data_bits, TorusPrecision(2));
        module.glwe_noise(&self.bits, &pt, sk, scratch_1)
    }

    pub fn decrypt<S, M, BE: Backend>(&self, module: &M, sk_glwe: &S, scratch: &mut Scratch<BE>) -> T
    where
        S: GLWESecretPreparedToRef<BE> + GLWEInfos,
        M: ModuleLogN + GLWEDecrypt<BE>,
        Scratch<BE>: ScratchTakeCore<BE>,
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

        let (mut pt, scratch_1) = scratch.take_glwe_plaintext(&pt_infos);

        module.glwe_decrypt(&self.bits, &mut pt, sk_glwe, scratch_1);

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

impl<D: DataMut, T: UnsignedInteger> FheUint<D, T> {
    /// Packs `Vec<GLWE(bit[i])>` into [`FheUint`].
    pub fn pack<G, M, K, H, BE: Backend>(&mut self, module: &M, mut bits: Vec<G>, keys: &H, scratch: &mut Scratch<BE>)
    where
        G: GLWEToMut + GLWEInfos,
        M: ModuleLogN + GLWEPacking<BE> + GLWECopy,
        K: GGLWEPreparedToRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>,
        Scratch<BE>: ScratchTakeCore<BE>,
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
    pub fn splice_u16<A, B, H, K, M, BE: Backend>(
        &mut self,
        module: &M,
        dst: usize,
        src: usize,
        a: &A,
        b: &B,
        keys: &H,
        scratch: &mut Scratch<BE>,
    ) where
        A: GLWEToRef + GLWEInfos,
        B: GLWEToRef + GLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>,
        K: GGLWEPreparedToRef<BE> + GGLWEInfos + GetGaloisElement,
        M: ModuleLogN + GLWERotate<BE> + GLWETrace<BE> + GLWESub + GLWEAdd + GLWECopy,
        Scratch<BE>: ScratchTakeBDD<T, BE>,
    {
        assert!(dst < (T::BITS >> 4) as usize);
        assert!(src < (T::BITS >> 4) as usize);

        let (mut tmp, scratch_1) = scratch.take_fhe_uint(self);
        tmp.splice_u8(module, dst << 1, src << 1, a, b, keys, scratch_1);
        self.splice_u8(module, (dst << 1) + 1, (src << 1) + 1, &tmp, b, keys, scratch_1);
    }

    #[allow(clippy::too_many_arguments)]
    // Self <- ((a.rotate_right(dst<<3) & 0xFFFF_FF00) | (b.rotate_right(src<<3) & 0x0000_00FF)).rotate_left(dst<<3);
    pub fn splice_u8<A, B, H, K, M, BE: Backend>(
        &mut self,
        module: &M,
        dst: usize,
        src: usize,
        a: &A,
        b: &B,
        keys: &H,
        scratch: &mut Scratch<BE>,
    ) where
        A: GLWEToRef + GLWEInfos,
        B: GLWEToRef + GLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>,
        K: GGLWEPreparedToRef<BE> + GGLWEInfos + GetGaloisElement,
        M: ModuleLogN + GLWERotate<BE> + GLWETrace<BE> + GLWESub + GLWEAdd + GLWECopy,
        Scratch<BE>: ScratchTakeBDD<T, BE>,
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
        let (mut tmp_fhe_uint_byte, scratch_1) = scratch.take_fhe_uint(b);

        // Move a[byte_a] into a[dst]
        module.glwe_rotate(-((T::bit_index(src << 3) << log_gap) as i64), &mut tmp_fhe_uint_byte, b);

        // Zeroes all other bytes
        module.glwe_trace_assign(&mut tmp_fhe_uint_byte, trace_start, keys, scratch_1);

        // Moves back self[0] to self[byte_tg]
        module.glwe_rotate_assign(rot, &mut tmp_fhe_uint_byte, scratch_1);

        // Add self[0] += a[0]
        module.glwe_add_assign(&mut self.bits, &tmp_fhe_uint_byte);
    }
}

impl<D: DataMut, T: UnsignedInteger> GLWEToMut for FheUint<D, T> {
    fn to_mut(&mut self) -> GLWE<&mut [u8]> {
        self.bits.to_mut()
    }
}

/// Extension of `ScratchTakeCore` that adds allocation of temporary
/// [`FheUint`] values from the scratch arena.
///
/// Implemented for `Scratch<BE>` whenever `Scratch<BE>: ScratchTakeCore<BE>`.
/// Callers use this to obtain short-lived `FheUint` temporaries on the hot
/// path without heap allocation.
pub trait ScratchTakeBDD<T: UnsignedInteger, BE: Backend>
where
    Self: ScratchTakeCore<BE>,
{
    /// Carves a temporary [`FheUint`] from the scratch arena.
    ///
    /// Returns the temporary and the remaining scratch space.
    fn take_fhe_uint<A>(&mut self, infos: &A) -> (FheUint<&mut [u8], T>, &mut Self)
    where
        A: GLWEInfos,
    {
        let (glwe, scratch) = self.take_glwe(infos);
        (
            FheUint {
                bits: glwe,
                _phantom: PhantomData,
            },
            scratch,
        )
    }
}

impl<T: UnsignedInteger, BE: Backend> ScratchTakeBDD<T, BE> for Scratch<BE> where Self: ScratchTakeCore<BE> {}

impl<D: DataRef, T: UnsignedInteger> FheUint<D, T> {
    pub fn get_bit_lwe<R, KGLWE, KLWE, M, BE: Backend>(
        &self,
        module: &M,
        bit: usize,
        res: &mut R,
        ks_glwe: Option<&KGLWE>,
        ks_lwe: &KLWE,
        scratch: &mut Scratch<BE>,
    ) where
        R: LWEToMut,
        KGLWE: GGLWEPreparedToRef<BE> + GGLWEInfos,
        KLWE: GGLWEPreparedToRef<BE> + GGLWEInfos,
        M: ModuleLogN + LWEFromGLWE<BE> + GLWERotate<BE> + GLWEKeyswitch<BE>,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        let log_gap: usize = module.log_n() - T::LOG_BITS as usize;
        if let Some(ks_glwe) = ks_glwe {
            let (mut res_tmp, scratch_1) = scratch.take_glwe(&GLWELayout {
                n: self.n(),
                base2k: ks_lwe.base2k(),
                k: ks_lwe.max_k().min(self.max_k()),
                rank: ks_lwe.rank_out(),
            });
            module.glwe_keyswitch(&mut res_tmp, self, ks_glwe, scratch_1);
            let mut res_lwe = res.to_mut();
            module.lwe_from_glwe(&mut res_lwe, &res_tmp, T::bit_index(bit) << log_gap, ks_lwe, scratch_1);
        } else {
            let mut res_lwe = res.to_mut();
            module.lwe_from_glwe(&mut res_lwe, self, T::bit_index(bit) << log_gap, ks_lwe, scratch);
        }
    }

    pub fn get_bit_glwe<R, K, M, H, BE: Backend>(&self, module: &M, bit: usize, res: &mut R, keys: &H, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut,
        K: GGLWEPreparedToRef<BE> + GGLWEInfos,
        M: ModuleLogN + GLWERotate<BE> + GLWETrace<BE>,
        H: GLWEAutomorphismKeyHelper<K, BE>,
        K: GGLWEPreparedToRef<BE> + GGLWEInfos + GetGaloisElement,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        let log_gap: usize = module.log_n() - T::LOG_BITS as usize;
        let rot = (T::bit_index(bit) << log_gap) as i64;
        module.glwe_rotate(-rot, res, self);
        module.glwe_trace_assign(res, 0, keys, scratch);
    }

    pub fn get_byte<R, K, M, H, BE: Backend>(&self, module: &M, byte: usize, res: &mut R, keys: &H, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut,
        K: GGLWEPreparedToRef<BE> + GGLWEInfos,
        M: ModuleLogN + GLWERotate<BE> + GLWETrace<BE>,
        H: GLWEAutomorphismKeyHelper<K, BE>,
        K: GGLWEPreparedToRef<BE> + GGLWEInfos + GetGaloisElement,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        let log_gap: usize = module.log_n() - T::LOG_BITS as usize;
        let trace_start = (T::LOG_BITS - T::LOG_BYTES) as usize;
        let rot = (T::bit_index(byte << 3) << log_gap) as i64;
        module.glwe_rotate(-rot, res, self);
        module.glwe_trace_assign(res, trace_start, keys, scratch);
    }
}

impl<D: DataRef, T: UnsignedInteger> GLWEToRef for FheUint<D, T> {
    fn to_ref(&self) -> GLWE<&[u8]> {
        self.bits.to_ref()
    }
}

impl<D: DataMut, T: UnsignedInteger> FheUint<D, T> {
    pub fn from_fhe_uint_prepared<M, DR, H, K, BE: Backend>(
        &mut self,
        module: &M,
        other: &FheUintPrepared<DR, T, BE>,
        keys: &H,
        scratch: &mut Scratch<BE>,
    ) where
        DR: DataRef,
        M: Cmux<BE> + ModuleLogN + GLWEPacking<BE> + GLWECopy,
        Scratch<BE>: ScratchTakeCore<BE>,
        K: GGLWEPreparedToRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>,
    {
        let zero: GLWE<Vec<u8>> = GLWE::alloc_from_infos(self);
        let mut one: GLWE<Vec<u8>> = GLWE::alloc_from_infos(self);
        one.data_mut().encode_coeff_i64(self.base2k().into(), 0, 2, 0, 1);

        let (mut out_bits, scratch_1) = scratch.take_glwe_slice(T::BITS as usize, self);

        for (i, bits) in out_bits.iter_mut().enumerate().take(T::BITS as usize) {
            module.cmux(bits, &one, &zero, &other.get_bit(i), scratch_1);
        }

        self.pack(module, out_bits, keys, scratch_1);
    }

    pub fn zero_byte<M, K, H, BE: Backend>(&mut self, module: &M, byte: usize, keys: &H, scratch: &mut Scratch<BE>)
    where
        H: GLWEAutomorphismKeyHelper<K, BE>,
        K: GGLWEPreparedToRef<BE> + GGLWEInfos + GetGaloisElement,
        M: ModuleLogN + GLWERotate<BE> + GLWETrace<BE> + GLWESub + GLWEAdd + GLWECopy,
        Scratch<BE>: ScratchTakeBDD<T, BE>,
    {
        let log_gap: usize = module.log_n() - T::LOG_BITS as usize;
        let trace_start = (T::LOG_BITS - T::LOG_BYTES) as usize;
        let rot: i64 = (T::bit_index(byte << 3) << log_gap) as i64;

        // Move a to self and align byte
        module.glwe_rotate_assign(-rot, &mut self.bits, scratch);

        // Stores this byte (everything else zeroed) into tmp_trace
        let (mut tmp_trace, scratch_1) = scratch.take_glwe(self);
        module.glwe_trace(&mut tmp_trace, trace_start, self, keys, scratch_1);

        // Subtracts to self to zero it
        module.glwe_sub_assign(&mut self.bits, &tmp_trace);

        // Move a to self and align byte
        module.glwe_rotate_assign(rot, &mut self.bits, scratch);
    }

    pub fn sext<M, H, K, BE>(&mut self, module: &M, byte: usize, keys: &H, scratch: &mut Scratch<BE>)
    where
        M:,
        H: GLWEAutomorphismKeyHelper<K, BE>,
        K: GGLWEPreparedToRef<BE> + GGLWEInfos + GetGaloisElement,
        BE: Backend,
        M: ModuleLogN + GLWERotate<BE> + GLWETrace<BE> + GLWEAdd + GLWESub + GLWECopy,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        assert!(byte < (1 << T::LOG_BYTES));

        let log_gap: usize = module.log_n() - T::LOG_BITS as usize;
        let rot: i64 = (T::bit_index((byte << 3) + 7) << log_gap) as i64;

        let (mut sext, scratch_1) = scratch.take_glwe(self);

        // Extract MSB
        module.glwe_rotate(-rot, &mut sext, &self.bits);
        module.glwe_trace_assign(&mut sext, 0, keys, scratch_1);

        // Replicates MSB in byte
        for i in 0..3 {
            let (mut tmp, _) = scratch_1.take_glwe(self);
            module.glwe_rotate(((1 << T::LOG_BYTES) << log_gap) << i, &mut tmp, &sext);
            module.glwe_add_assign(&mut sext, &tmp);
        }

        // Splice sext
        let (mut tmp, scratch_2) = scratch_1.take_glwe(self);
        for i in (byte + 1)..(1 << T::LOG_BYTES) as usize {
            FheUint::<&mut [u8], T>::from_glwe_to_mut(&mut tmp).splice_u8(module, i, 0, &self.bits, &sext, keys, scratch_2);
            module.glwe_copy(&mut self.bits, &tmp);
        }
    }
}
