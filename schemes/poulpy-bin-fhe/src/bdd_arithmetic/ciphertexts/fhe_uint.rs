use poulpy_core::layouts::prepared::GGLWEPreparedBackendRef;
use poulpy_core::{
    GLWEAdd, GLWECopy, GLWEDecrypt, GLWEEncryptSk, GLWEKeyswitch, GLWENoise, GLWEPacking, GLWERotate, GLWESub, GLWETrace,
    LWEFromGLWE, ScratchArenaTakeCore, TransferInto,
    layouts::{
        Base2K, GGLWEInfos, GGSWInfos, GLWE, GLWEInfos, GLWELayout, GLWEPlaintext, GLWEPlaintextLayout,
        GLWESecretPreparedToBackendRef, GLWEToBackendMut, GLWEToBackendRef, GetAutomorphismKey, LWEInfos, LWEToBackendMut,
        ModuleCoreAlloc, Rank, SetK, TorusPrecision,
    },
};
use poulpy_hal::AlignedBuf;
use poulpy_hal::layouts::ZnxWord;
use poulpy_hal::{
    api::{ModuleLogN, ModuleN},
    layouts::{Backend, CopyFromHost, CopyToHost, Data, HostBackend, HostDataMut, HostDataRef, ScratchArena, Stats},
    source::Source,
};
use std::{collections::HashMap, marker::PhantomData, ops::Range};

use crate::bdd_arithmetic::{Cmux, FheUintPrepared, FromBits, GetGGSWBit, ToBits, UnsignedInteger};
use poulpy_core::GLWEBytesOf;
use poulpy_core::layouts::prepared::GGSWPreparedToBackendRef;

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

impl<D: Data, T: UnsignedInteger, W: ZnxWord> LWEInfos for FheUint<D, T, W> {
    fn noise(&self) -> Option<poulpy_core::ComponentNoise> {
        self.bits.noise()
    }

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
    pub fn encrypt_sk<S, M, BE>(
        &mut self,
        module: &M,
        data: T,
        sk_glwe: &S,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        BE: Backend<OwnedBuf: HostDataMut + HostDataRef, ZnxWord = i64>,
        GLWE<D, i64>: GLWEToBackendMut<BE>,
        S: GLWESecretPreparedToBackendRef<BE> + GLWEInfos,
        M: GLWEBytesOf<BE> + ModuleLogN + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = BE::ZnxWord> + GLWEEncryptSk<BE>,
    {
        #[cfg(debug_assertions)]
        {
            assert!(module.n().is_multiple_of(T::BITS as usize));
            assert!(
                self.n() == module.n() as u32,
                "the ciphertext degree differs from the module degree"
            );
            assert!(
                sk_glwe.n() == module.n() as u32,
                "the secret degree differs from the module degree"
            );
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
        module.glwe_encrypt_sk(&mut self.bits, &pt, sk_glwe, source_xe, source_xa, scratch);
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
    pub fn noise_stats<S, M, BE>(&self, module: &M, want: u32, sk: &S, scratch: &mut ScratchArena<'_, BE>) -> Stats
    where
        BE: Backend<OwnedBuf: HostDataMut + HostDataRef, ZnxWord = i64> + HostBackend,
        Self: GLWEToBackendRef<BE>,
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
            assert!(
                self.n() == module.n() as u32,
                "the ciphertext degree differs from the module degree"
            );
            assert!(
                sk.n() == module.n() as u32,
                "the secret degree differs from the module degree"
            );
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
        Self: GLWEToBackendRef<BE>,
        S: GLWESecretPreparedToBackendRef<BE> + GLWEInfos,
        M: GLWEBytesOf<BE> + ModuleLogN + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = BE::ZnxWord> + GLWEDecrypt<BE>,
    {
        #[cfg(debug_assertions)]
        {
            assert!(module.n().is_multiple_of(T::BITS as usize));
            assert!(
                self.n() == module.n() as u32,
                "the ciphertext degree differs from the module degree"
            );
            assert!(
                sk_glwe.n() == module.n() as u32,
                "the secret degree differs from the module degree"
            );
        }

        let pt_infos = GLWEPlaintextLayout {
            n: self.n(),
            base2k: self.base2k(),
            k: 2_usize.into(),
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
            k: 2_usize.into(),
        };
        module.glwe_plaintext_bytes_of_from_infos(&pt_infos) + module.glwe_decrypt_tmp_bytes(self)
    }
}

impl<D: Data, T: UnsignedInteger> FheUint<D, T, i64> {
    /// Packs `Vec<GLWE(bit[i])>` into [`FheUint`].
    pub fn pack<G, M, H, BE>(&mut self, module: &M, mut bits: Vec<G>, keys: &H, scratch: &mut ScratchArena<'_, BE>)
    where
        BE: Backend<OwnedBuf = D, ZnxWord = i64>,
        G: GLWEToBackendMut<BE> + GLWEInfos,
        M: GLWEBytesOf<BE> + ModuleLogN + GLWEPacking<BE> + GLWECopy<BE>,
        H: GetAutomorphismKey<BE>,
        GLWE<D, BE::ZnxWord>: GLWEToBackendMut<BE>,
    {
        // Repacks the GLWE ciphertexts bits
        let log_gap: usize = module.log_n() - T::LOG_BITS as usize;
        let mut cts: HashMap<usize, &mut G> = HashMap::new();
        for (i, ct) in bits.iter_mut().enumerate().take(T::BITS as usize) {
            cts.insert(T::bit_index(i) << log_gap, ct);
        }

        module.glwe_pack(&mut self.bits, cts, log_gap, keys, scratch);
    }

    /// Scratch bytes of [`FheUint::splice_u16`] for operands `a` and `b` into `self`: its two byte splices run one after the other.
    ///
    /// # Panics
    ///
    /// If `keys` lacks a key the trace reads.
    pub fn splice_u16_tmp_bytes<A, B, H, M, BE>(&self, module: &M, a_infos: &A, b_infos: &B, keys: &H) -> usize
    where
        BE: Backend,
        A: GLWEInfos,
        B: GLWEInfos,
        H: GetAutomorphismKey<BE>,
        M: GLWECopy<BE> + GLWERotate<BE> + GLWETrace<BE>,
    {
        // The second splice reads the first's result, which has self's layout.
        self.splice_u8_tmp_bytes(module, a_infos, b_infos, keys)
            .max(self.splice_u8_tmp_bytes(module, self, b_infos, keys))
    }

    /// Scratch bytes of [`FheUint::splice_u8`] for operands `a` and `b` into `self`: the copy of `a`, the byte zeroing and the isolation of `b`'s byte run one after the other.
    ///
    /// # Panics
    ///
    /// If `keys` lacks a key the trace reads.
    pub fn splice_u8_tmp_bytes<A, B, H, M, BE>(&self, module: &M, a_infos: &A, b_infos: &B, keys: &H) -> usize
    where
        BE: Backend,
        A: GLWEInfos,
        B: GLWEInfos,
        H: GetAutomorphismKey<BE>,
        M: GLWECopy<BE> + GLWERotate<BE> + GLWETrace<BE>,
    {
        module
            .glwe_copy_tmp_bytes(self, a_infos)
            .max(self.zero_byte_tmp_bytes(module, keys))
            .max(trace_tmp_bytes(module, byte_levels::<T>(b_infos), b_infos, b_infos, keys))
            .max(module.glwe_rotate_tmp_bytes())
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
        Self: GLWEToBackendMut<BE>,
        A: GLWEToBackendRef<BE> + GLWEInfos,
        B: GLWEToBackendRef<BE> + GLWEInfos,
        H: GetAutomorphismKey<BE>,
        M: GLWEBytesOf<BE>
            + ModuleLogN
            + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = BE::ZnxWord>
            + GLWERotate<BE>
            + GLWETrace<BE>
            + GLWESub<BE>
            + GLWEAdd<BE>
            + GLWECopy<BE>,
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
        Self: GLWEToBackendMut<BE>,
        A: GLWEToBackendRef<BE> + GLWEInfos,
        B: GLWEToBackendRef<BE> + GLWEInfos,
        H: GetAutomorphismKey<BE>,
        M: GLWEBytesOf<BE>
            + ModuleLogN
            + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = BE::ZnxWord>
            + GLWERotate<BE>
            + GLWETrace<BE>
            + GLWESub<BE>
            + GLWEAdd<BE>
            + GLWECopy<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeBDD<'a, T, BE>,
    {
        assert!(dst < (T::BITS >> 3) as usize);
        assert!(src < (T::BITS >> 3) as usize);

        // 1) Zero the byte receiver
        let log_gap: usize = module.log_n() - T::LOG_BITS as usize;
        let trace_start = (T::LOG_BITS - T::LOG_BYTES) as usize;
        let rot: i64 = (T::bit_index(dst << 3) << log_gap) as i64;

        module.glwe_copy(self, a, scratch);

        self.zero_byte(module, dst, keys, scratch);

        // Isolate the byte to transfer from a
        let mut tmp_fhe_uint_byte: FheUint<BE::OwnedBuf, T, BE::ZnxWord> = FheUint::alloc_from_infos(module, b);
        let mut scratch_1 = scratch.borrow();

        // Move a[byte_a] into a[dst]
        module.glwe_rotate(-((T::bit_index(src << 3) << log_gap) as i64), &mut tmp_fhe_uint_byte, b);

        // Zeroes all other bytes
        let log_n = tmp_fhe_uint_byte.log_n();
        module.glwe_trace_assign(&mut tmp_fhe_uint_byte, trace_start..log_n, false, true, keys, &mut scratch_1);

        // Moves back self[0] to self[byte_tg]
        module.glwe_rotate_assign(rot, &mut tmp_fhe_uint_byte, &mut scratch_1);

        // Add self[0] += a[0]
        module.glwe_add_assign(self, &tmp_fhe_uint_byte);
    }
}

/// The levels a byte isolation traces.
fn byte_levels<T: UnsignedInteger>(infos: &impl GLWEInfos) -> Range<usize> {
    (T::LOG_BITS - T::LOG_BYTES) as usize..infos.log_n()
}

/// Scratch bytes of tracing `a_infos` over `levels` into `res_infos`, sized from the key of the first rotation as the trace reads it.
fn trace_tmp_bytes<M, R, A, H, BE>(module: &M, levels: Range<usize>, res_infos: &R, a_infos: &A, keys: &H) -> usize
where
    BE: Backend,
    R: GLWEInfos,
    A: GLWEInfos,
    H: GetAutomorphismKey<BE>,
    M: GLWETrace<BE>,
{
    let first = module
        .glwe_trace_galois_elements(res_infos.n(), levels, false)
        .first()
        .copied()
        .unwrap_or(-1);
    let atk = keys
        .get_automorphism_key(first, res_infos.k().max(a_infos.k()))
        .unwrap_or_else(|e| panic!("trace rotation {first}: {e}"));
    module.glwe_trace_tmp_bytes(res_infos, a_infos, &atk)
}

impl<BE: Backend, D: Data, T: UnsignedInteger> GLWEToBackendRef<BE> for FheUint<D, T, BE::ZnxWord>
where
    GLWE<D, BE::ZnxWord>: GLWEToBackendRef<BE>,
{
    fn to_backend_ref(&self) -> GLWE<<BE as Backend>::BufRef<'_>, <BE as Backend>::ZnxWord> {
        self.bits.to_backend_ref()
    }
}

impl<BE: Backend, D: Data, T: UnsignedInteger> GLWEToBackendMut<BE> for FheUint<D, T, BE::ZnxWord>
where
    GLWE<D, BE::ZnxWord>: GLWEToBackendMut<BE>,
{
    fn to_backend_mut(&mut self) -> GLWE<<BE as Backend>::BufMut<'_>, <BE as Backend>::ZnxWord> {
        self.bits.to_backend_mut()
    }

    fn set_canonical(&mut self, canonical: bool) {
        self.bits.set_canonical(canonical)
    }

    fn set_noise(&mut self, metadata: Option<poulpy_core::ComponentNoise>) {
        GLWEToBackendMut::<BE>::set_noise(&mut self.bits, metadata);
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

/// The `k` a switch reads: each switch takes its input at the precision its key decomposes exactly, the bits below being dropped by narrowing, not by the key.
fn max_input_k<BE: Backend>(key: &GGLWEPreparedBackendRef<'_, BE>) -> TorusPrecision {
    key.gglwe_layout().max_input_k()
}

/// The precision `a` is narrowed to before [`FheUint::get_bit_lwe`] switches it, by the first key it meets.
fn extraction_input_k<BE: Backend>(
    a: &impl GLWEInfos,
    ks_glwe: Option<&GGLWEPreparedBackendRef<'_, BE>>,
    ks_lwe: &GGLWEPreparedBackendRef<'_, BE>,
) -> TorusPrecision {
    a.k().min(ks_glwe.map_or(max_input_k(ks_lwe), max_input_k))
}

impl<D: Data, T: UnsignedInteger, W: ZnxWord> FheUint<D, T, W> {
    /// Scratch bytes of [`FheUint::get_bit_lwe`] into `res`: the key switch of the narrowed input to the layout of `ks_glwe`, when given, then the extraction at the precision of `ks_lwe`, one after the other.
    pub fn get_bit_lwe_tmp_bytes<R, M, BE>(
        &self,
        module: &M,
        res_infos: &R,
        ks_glwe: Option<&GGLWEPreparedBackendRef<'_, BE>>,
        ks_lwe: &GGLWEPreparedBackendRef<'_, BE>,
    ) -> usize
    where
        BE: Backend,
        R: LWEInfos,
        M: GLWEKeyswitch<BE> + LWEFromGLWE<BE>,
    {
        let mut a = self.glwe_layout();
        a.k = extraction_input_k(self, ks_glwe, ks_lwe);
        match ks_glwe {
            Some(ks_glwe) => {
                let switched = GLWELayout {
                    n: a.n,
                    base2k: ks_glwe.base2k(),
                    k: ks_glwe.k().min(max_input_k(ks_lwe)),
                    rank: ks_glwe.rank_out(),
                };
                module
                    .glwe_keyswitch_tmp_bytes(&switched, &a, ks_glwe)
                    .max(module.lwe_from_glwe_tmp_bytes(res_infos, &switched, ks_lwe))
            }
            None => module.lwe_from_glwe_tmp_bytes(res_infos, &a, ks_lwe),
        }
    }

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
        Self: GLWEToBackendRef<BE>,
        M: GLWEBytesOf<BE>
            + ModuleLogN
            + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = BE::ZnxWord>
            + LWEFromGLWE<BE>
            + GLWEKeyswitch<BE>,
    {
        let log_gap: usize = module.log_n() - T::LOG_BITS as usize;
        let mut a = GLWEToBackendRef::<BE>::to_backend_ref(self);
        a.set_k(extraction_input_k(self, ks_glwe, ks_lwe));
        // The keyswitch reads a lazy sum as is, which its margin tolerates.
        a.set_canonical(true);
        if let Some(ks_glwe) = ks_glwe {
            let mut res_tmp: GLWE<BE::OwnedBuf, BE::ZnxWord> =
                module.glwe_alloc(ks_glwe.base2k(), ks_glwe.k().min(max_input_k(ks_lwe)), ks_glwe.rank_out());
            let mut scratch_1 = scratch.borrow();
            {
                let mut scratch_op = scratch_1.borrow();
                module.glwe_keyswitch(&mut res_tmp, &&a, ks_glwe, &mut scratch_op);
            }
            let mut scratch_op = scratch_1.borrow();
            module.lwe_from_glwe(res, &res_tmp, T::bit_index(bit) << log_gap, ks_lwe, &mut scratch_op);
        } else {
            module.lwe_from_glwe(res, &&a, T::bit_index(bit) << log_gap, ks_lwe, scratch);
        }
    }

    /// Scratch bytes of [`FheUint::get_bit_glwe`] into `res`: the trace of `res`.
    ///
    /// # Panics
    ///
    /// If `keys` lacks a key the trace reads.
    pub fn get_bit_glwe_tmp_bytes<R, M, H, BE>(&self, module: &M, res_infos: &R, keys: &H) -> usize
    where
        BE: Backend,
        R: GLWEInfos,
        H: GetAutomorphismKey<BE>,
        M: GLWETrace<BE>,
    {
        trace_tmp_bytes(module, 0..res_infos.log_n(), res_infos, res_infos, keys)
    }

    pub fn get_bit_glwe<R, M, H, BE>(&self, module: &M, bit: usize, res: &mut R, keys: &H, scratch: &mut ScratchArena<'_, BE>)
    where
        BE: Backend,
        R: GLWEToBackendMut<BE> + GLWEInfos,
        Self: GLWEToBackendRef<BE>,
        M: GLWEBytesOf<BE> + ModuleLogN + GLWERotate<BE> + GLWETrace<BE>,
        H: GetAutomorphismKey<BE>,
    {
        let log_gap: usize = module.log_n() - T::LOG_BITS as usize;
        let rot = (T::bit_index(bit) << log_gap) as i64;
        module.glwe_rotate(-rot, res, self);
        let log_n = res.log_n();
        module.glwe_trace_assign(res, 0..log_n, false, true, keys, scratch);
    }

    /// Scratch bytes of [`FheUint::get_byte`] into `res`: the trace of `res` over the byte levels.
    ///
    /// # Panics
    ///
    /// If `keys` lacks a key the trace reads.
    pub fn get_byte_tmp_bytes<R, M, H, BE>(&self, module: &M, res_infos: &R, keys: &H) -> usize
    where
        BE: Backend,
        R: GLWEInfos,
        H: GetAutomorphismKey<BE>,
        M: GLWETrace<BE>,
    {
        trace_tmp_bytes(module, byte_levels::<T>(res_infos), res_infos, res_infos, keys)
    }

    pub fn get_byte<R, M, H, BE>(&self, module: &M, byte: usize, res: &mut R, keys: &H, scratch: &mut ScratchArena<'_, BE>)
    where
        BE: Backend,
        R: GLWEToBackendMut<BE> + GLWEInfos,
        Self: GLWEToBackendRef<BE>,
        M: GLWEBytesOf<BE> + ModuleLogN + GLWERotate<BE> + GLWETrace<BE>,
        H: GetAutomorphismKey<BE>,
    {
        let log_gap: usize = module.log_n() - T::LOG_BITS as usize;
        let trace_start = (T::LOG_BITS - T::LOG_BYTES) as usize;
        let rot = (T::bit_index(byte << 3) << log_gap) as i64;
        module.glwe_rotate(-rot, res, self);
        let log_n = res.log_n();
        module.glwe_trace_assign(res, trace_start..log_n, false, true, keys, scratch);
    }
}

impl<T: UnsignedInteger> FheUint<AlignedBuf, T, i64> {
    /// Scratch bytes of [`FheUint::from_fhe_uint_prepared`] from bits laid out as `bit_infos`: the bit selections, then the packing, one after the other.
    ///
    /// # Panics
    ///
    /// If `keys` lacks the key of the packing trace.
    pub fn from_fhe_uint_prepared_tmp_bytes<A, H, M, BE>(&self, module: &M, bit_infos: &A, keys: &H) -> usize
    where
        BE: Backend,
        A: GGSWInfos,
        H: GetAutomorphismKey<BE>,
        M: Cmux<BE> + GLWEPacking<BE>,
    {
        let atk = keys
            .get_automorphism_key(-1, self.k())
            .unwrap_or_else(|e| panic!("packing trace -1: {e}"));
        module
            .cmux_tmp_bytes(self, self, bit_infos)
            .max(module.glwe_pack_tmp_bytes(self, self, &atk))
    }

    pub fn from_fhe_uint_prepared<M, H, BE>(
        &mut self,
        module: &M,
        other: &FheUintPrepared<BE::OwnedBuf, T, BE>,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        BE: Backend<OwnedBuf = AlignedBuf, ZnxWord = i64>,
        M: GLWEBytesOf<BE>
            + Cmux<BE>
            + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = BE::ZnxWord>
            + ModuleLogN
            + GLWEPacking<BE>
            + GLWECopy<BE>,
        GLWE<AlignedBuf, BE::ZnxWord>: GLWEToBackendMut<BE>,
        Self: GLWEToBackendMut<BE>,
        H: GetAutomorphismKey<BE>,
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
    /// Scratch bytes of [`FheUint::zero_byte`]: the rotations and the trace of `self`.
    ///
    /// # Panics
    ///
    /// If `keys` lacks a key the trace reads.
    pub fn zero_byte_tmp_bytes<M, H, BE>(&self, module: &M, keys: &H) -> usize
    where
        BE: Backend,
        H: GetAutomorphismKey<BE>,
        M: GLWERotate<BE> + GLWETrace<BE>,
    {
        module
            .glwe_rotate_tmp_bytes()
            .max(trace_tmp_bytes(module, byte_levels::<T>(self), self, self, keys))
    }

    pub fn zero_byte<M, H, BE>(&mut self, module: &M, byte: usize, keys: &H, scratch: &mut ScratchArena<'_, BE>)
    where
        BE: Backend<OwnedBuf = D, ZnxWord = i64>,
        Self: GLWEToBackendMut<BE>,
        H: GetAutomorphismKey<BE>,
        M: GLWEBytesOf<BE>
            + ModuleLogN
            + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = BE::ZnxWord>
            + GLWERotate<BE>
            + GLWETrace<BE>
            + GLWESub<BE>
            + GLWEAdd<BE>
            + GLWECopy<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeBDD<'a, T, BE>,
    {
        let log_gap: usize = module.log_n() - T::LOG_BITS as usize;
        let trace_start = (T::LOG_BITS - T::LOG_BYTES) as usize;
        let rot: i64 = (T::bit_index(byte << 3) << log_gap) as i64;

        // Move a to self and align byte
        module.glwe_rotate_assign(-rot, self, scratch);

        // Stores this byte (everything else zeroed) into tmp_trace
        let mut tmp_trace: GLWE<BE::OwnedBuf, BE::ZnxWord> = module.glwe_alloc_from_infos(self);
        let log_n = tmp_trace.log_n();
        module.glwe_trace(&mut tmp_trace, trace_start..log_n, false, true, self, keys, scratch);

        // Subtracts to self to zero it
        module.glwe_sub_assign(self, &tmp_trace);

        // Move a to self and align byte
        module.glwe_rotate_assign(rot, self, scratch);
    }

    /// Scratch bytes of [`FheUint::sext`]: the trace of the sign bit, then the byte splices and the copies, one after the other.
    ///
    /// # Panics
    ///
    /// If `keys` lacks a key the trace reads.
    pub fn sext_tmp_bytes<H, M, BE>(&self, module: &M, keys: &H) -> usize
    where
        BE: Backend,
        H: GetAutomorphismKey<BE>,
        M: GLWECopy<BE> + GLWERotate<BE> + GLWETrace<BE>,
    {
        trace_tmp_bytes(module, 0..self.log_n(), self, self, keys)
            .max(module.glwe_copy_tmp_bytes(self, self))
            .max(self.splice_u8_tmp_bytes(module, self, self, keys))
    }

    pub fn sext<M, H, BE>(&mut self, module: &M, byte: usize, keys: &H, scratch: &mut ScratchArena<'_, BE>)
    where
        Self: GLWEToBackendRef<BE>,
        Self: GLWEToBackendMut<BE>,
        H: GetAutomorphismKey<BE>,
        BE: Backend<OwnedBuf = D, ZnxWord = i64>,
        M: GLWEBytesOf<BE>
            + ModuleLogN
            + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = BE::ZnxWord>
            + GLWERotate<BE>
            + GLWETrace<BE>
            + GLWEAdd<BE>
            + GLWESub<BE>
            + GLWECopy<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeBDD<'a, T, BE>,
    {
        assert!(byte < (1 << T::LOG_BYTES));

        let log_gap: usize = module.log_n() - T::LOG_BITS as usize;
        let rot: i64 = (T::bit_index((byte << 3) + 7) << log_gap) as i64;

        let mut sext: GLWE<BE::OwnedBuf, BE::ZnxWord> = module.glwe_alloc_from_infos(self);
        let mut scratch_1 = scratch.borrow();

        // Extract MSB
        module.glwe_rotate(-rot, &mut sext, self);
        let log_n = sext.log_n();
        module.glwe_trace_assign(&mut sext, 0..log_n, false, true, keys, &mut scratch_1.borrow());

        // Replicates MSB in byte
        for i in 0..3 {
            let mut tmp: GLWE<BE::OwnedBuf, BE::ZnxWord> = module.glwe_alloc_from_infos(self);
            module.glwe_rotate(((1 << T::LOG_BYTES) << log_gap) << i, &mut tmp, &sext);
            module.glwe_add_assign(&mut sext, &tmp);
        }

        // Splice sext
        let mut tmp: FheUint<BE::OwnedBuf, T, BE::ZnxWord> = FheUint::alloc_from_infos(module, self);
        let mut current: GLWE<BE::OwnedBuf, BE::ZnxWord> = module.glwe_alloc_from_infos(self);
        module.glwe_copy(&mut current, self, &mut scratch_1);
        for i in (byte + 1)..(1 << T::LOG_BYTES) as usize {
            tmp.splice_u8(module, i, 0, &current, &sext, keys, &mut scratch_1);
            module.glwe_copy(&mut current, &tmp.bits, &mut scratch_1);
        }
        module.glwe_copy(self, &current, &mut scratch_1);
        self.set_noise(None);
    }
}
