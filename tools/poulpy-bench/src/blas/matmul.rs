//! Runners for the `poulpy-blas` operations on one `d x d` block per operand.
//!
//! A runner builds its whole point once, outside the timed loop: the module, the secret key, the context or the public
//! operand, only the rotation keys its operation looks up (plus the tensor key of a ciphertext by ciphertext product),
//! the encrypted inputs and the scratch its `*_tmp_bytes` reports. It checks that `base2k` fits the lazy sums of the
//! operation, calls the operation once, decrypts the result and asserts every entry within [`error_bound`] of the
//! cleartext, then times that same call. A point builder takes a client module beside the evaluated one, which makes
//! the keys and the ciphertexts and decrypts, the keys and ciphertexts moving between them: the runners pass the
//! evaluated module itself, a GPU module takes a host client.
//!
//! Entries of the encrypted matrices lie in `[-1, 1)`, those of the second ciphertext factor and of the public matrices
//! in `[-1/d, 1/d)`, so that the products stay in `[-1, 1)`. A public matrix is one matrix shared by every matrix slot.

use std::collections::{BTreeSet, HashMap};

use criterion::{Bencher, measurement::Measurement};
use poulpy_blas::{
    api::{
        BlasContextPreparedFactory, BlasDiagonalCiphertextPreparedFactory, BlasEncodingOps, BlasMatIntraOps, BlasMatMulOps,
        BlasMatVecOperandPreparedFactory, BlasPtOperandPreparedFactory,
    },
    layouts::{
        BlasCiphertext, BlasCiphertextOwned, BlasContextPrepared, BlasGridInfos, BlasModuleAlloc, BlasPlan, BlasPtAlgorithm,
        BlasPtOperandPrepared, BlasPtSide, Left, Right, blas_matmul_max_base2k, blas_matmul_pt_left_max_base2k,
        blas_matmul_pt_right_max_base2k,
        grid::{join_blocks, split_into_blocks},
        replicated,
    },
    oep::{BlasMatIntraImpl, BlasMatMulImpl},
};
use poulpy_ckks::{
    CKKSInfos, CKKSLayout, CoeffsMeta, GLWEPlaintextMeta, Scale, SetCKKSInfos, SlotsKind,
    api::{
        CKKSAllOpsTmpBytes, CKKSDecryptOps, CKKSEncodingHostOps, CKKSEncodingOps, CKKSEncryptOps, CKKSLinearTransformationOps,
        CKKSModuleInfos, CKKSMulOps,
    },
    layouts::{CKKSCiphertextOwned, CKKSModuleAlloc},
    test_suite::helpers::TestContextSharedModule,
};
use poulpy_core::{
    GLWEAutomorphismKeyEncryptSk, GLWETensorKeyEncryptSk,
    api::{
        GLWEAutomorphismKeyPreparedFactory, GLWESecretPreparedFactory, GLWESecretSampling, GLWETensorKeyPreparedFactory,
        TransferInto,
    },
    layouts::{
        BackendGLWESecret, Base2K, Degree, Dnum, Dsize, GGLWELayout, GLWEAutomorphismKeyAlloc, GLWEAutomorphismKeyLayout,
        GLWELayout, GLWESecretAlloc, GLWETensorKeyAlloc, GLWETensorKeyLayout, LWEInfos, ModuleCoreAlloc, Rank, SetK,
        TorusPrecision, k_digit_of,
        prepared::{GLWEAutomorphismKeyPrepared, GLWESecretPrepared, GLWETensorKeyPrepared},
    },
};
use poulpy_hal::{
    api::{CnvPVecAlloc, ModuleNew, ModuleSynchronize, ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Backend, ConjugateInvariant, CyclotomicOrder, HostStaged, MaxBase2k, Module, Ring, ScratchOwned, Standard},
    source::Source,
};

use crate::{Call, backend_name, blas::params::BlasBenchParams, timed};

/// Secret-key Hamming weight.
const HW: usize = 192;
/// Bits above the scale that an encoded entry, map or public matrix keeps for a magnitude of at most one.
pub(super) const HEADROOM: usize = 4;
/// Integer bits kept when decrypting, for the sums of `d` products of magnitude at most one.
const DECODE_HEADROOM: usize = 24;
/// The failure target of the radix check, in bits.
pub(super) const FAILURE_BITS: usize = 128;

pub(super) fn base2k(bp: &BlasBenchParams) -> Base2K {
    Base2K(bp.base2k as u32)
}

/// The slots of a block on ring `R`: complex on the standard ring, real on the conjugate-invariant one.
fn slots_kind<R: Ring>() -> SlotsKind {
    if R::CYCLOTOMIC_ORDER_FACTOR == ConjugateInvariant::CYCLOTOMIC_ORDER_FACTOR {
        SlotsKind::Real
    } else {
        SlotsKind::Complex
    }
}

pub(super) fn block_layout<R: Ring>(bp: &BlasBenchParams) -> CKKSLayout {
    CKKSLayout {
        glwe_layout: GLWELayout {
            n: Degree(bp.n as u32),
            base2k: base2k(bp),
            k: TorusPrecision(bp.k as u32),
            rank: Rank(1),
        },
        meta: GLWEPlaintextMeta {
            scale: Scale::Log(bp.log_delta),
            slots: slots_kind::<R>(),
            log_sparsity: 0,
        },
    }
}

// Not `key_dnum_k_aux`: its one-digit guard leaves sqrt(dnum n) quanta of key-switching noise, which the checks would show.
fn key_shape(bp: &BlasBenchParams) -> (TorusPrecision, Dnum, TorusPrecision) {
    let k_digit = k_digit_of(base2k(bp), Dsize(bp.dsize as u32));
    let dnum = GGLWELayout::dnum_for_input(TorusPrecision(bp.k as u32), k_digit);
    (k_digit, dnum, TorusPrecision(k_digit.0 + bp.n.ilog2()))
}

pub(super) fn atk_layout(bp: &BlasBenchParams) -> GLWEAutomorphismKeyLayout {
    let (k_digit, dnum, k_aux) = key_shape(bp);
    GLWEAutomorphismKeyLayout {
        n: Degree(bp.n as u32),
        base2k: base2k(bp),
        k_aux,
        rank: Rank(1),
        k_digit,
        dnum,
    }
}

pub(super) fn tsk_layout(bp: &BlasBenchParams) -> GLWETensorKeyLayout {
    let (k_digit, dnum, k_aux) = key_shape(bp);
    GLWETensorKeyLayout {
        n: Degree(bp.n as u32),
        base2k: base2k(bp),
        k_aux,
        rank: Rank(1),
        k_digit,
        dnum,
    }
}

/// Deterministic pseudo-random value in `[-1, 1)` of entry `(r, c)` of matrix `s` in stream `tag`, a splitmix64 hash.
pub(crate) fn entry(tag: u64, s: usize, r: usize, c: usize) -> f64 {
    let mut x = tag ^ ((s as u64) << 40) ^ ((r as u64) << 20) ^ c as u64;
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
    x ^= x >> 31;
    (x >> 11) as f64 / (1u64 << 52) as f64 - 1.0
}

/// The encrypted matrices.
pub(super) fn a_entry(s: usize, r: usize, c: usize) -> f64 {
    entry(1, s, r, c)
}

/// The second ciphertext factor, scaled so that the products stay in `[-1, 1)`.
pub(super) fn b_entry(d: usize) -> impl Fn(usize, usize, usize) -> f64 {
    move |s, r, c| entry(2, s, r, c) / d as f64
}

/// The public matrix, scaled alike.
fn p_entry(d: usize) -> impl Fn(usize, usize, usize) -> f64 {
    move |s, r, c| entry(3, s, r, c) / d as f64
}

/// The largest error a result on ring `R` may show against the cleartext: `2^7 sqrt(2 s d) 2^-log_delta` for the `s`
/// slots of a block, `2^7 sqrt(n d) 2^-log_delta` on the standard ring, a precision of `log_delta - log2(n d) / 2 - 7`
/// bits.
///
/// A fresh encryption, and each key switch through the rounding of its output by a secret of Hamming weight 192, leave a
/// few quanta (`2^-log_delta`) of noise per coefficient, `u = sqrt(s) 2^-log_delta` on a slot: `sqrt(n / 2)` on the
/// standard ring, `sqrt(n)` on the conjugate-invariant one. A product sums `d` terms of an entry of magnitude below one by
/// such noise: the maxima measured over the whole sweep are at most `26 sqrt(d) u`, a seventh of the bound, which is
/// `181 sqrt(d) u`. A radix that spoils whole limbs, or a key of another secret, lands far above it.
pub fn error_bound<R: Ring>(bp: &BlasBenchParams) -> f64 {
    (7.0 - bp.log_delta as f64).exp2() * ((2 * bp.slots_on::<R>() * bp.d()) as f64).sqrt()
}

/// Prints the error of a runner's first call and asserts it within [`error_bound`].
pub(super) fn verify<BE: Backend>(op: &str, bp: &BlasBenchParams, max_err: f64, max_base2k: Option<usize>, keys: usize) {
    verify_within::<BE>(op, bp, max_err, error_bound::<BE::Ring>(bp), max_base2k, keys);
}

/// [`verify`] against `bound`.
pub(super) fn verify_within<BE>(
    op: &str,
    bp: &BlasBenchParams,
    max_err: f64,
    bound: f64,
    max_base2k: Option<usize>,
    keys: usize,
) {
    println!(
        "PRECISION backend={} op={op} params={bp} max_err={max_err:.3e} ({:.1} bits) bound={bound:.3e} ({:.1} bits) max_base2k={} keys={keys}",
        backend_name::<BE>(),
        -max_err.log2(),
        -bound.log2(),
        max_base2k.map_or_else(|| "-".to_string(), |max| max.to_string()),
    );
    assert!(
        max_err <= bound,
        "{op}: the first call deviates from the cleartext beyond the error bound"
    );
}

/// What the first call of a point showed, for a caller that reports it.
pub struct BlasCheck {
    /// The largest distance of a decrypted entry to the cleartext.
    pub max_err: f64,
    /// The [`error_bound`] of the point, which a runner asserts `max_err` within.
    pub bound: f64,
    /// The radix the lazy sums of a product allow at 128 failure bits, `None` for the other operations.
    pub max_base2k: Option<usize>,
    /// The rotation keys the point holds.
    pub keys: usize,
    /// The budget the operation consumed, from its input to its result.
    pub consumed_bits: usize,
    /// The budget the result keeps.
    pub output_log_budget: usize,
    /// The layout of the rotation keys, and of the tensor key of a ciphertext by ciphertext product.
    pub key_layout: GLWEAutomorphismKeyLayout,
}

/// A built point, for a caller that times it outside Criterion.
pub struct BlasPoint {
    /// One call of the operation, ended by a synchronization of the module.
    pub call: Call,
    /// The check of the first call.
    pub check: BlasCheck,
}

impl BlasPoint {
    /// Prints the check, as the Criterion runners report it, asserts the error within the bound ([`verify_within`])
    /// and returns the call.
    pub(super) fn reported<BE>(self, op: &str, bp: &BlasBenchParams) -> Call {
        let check = &self.check;
        verify_within::<BE>(op, bp, check.max_err, check.bound, check.max_base2k, check.keys);
        self.call
    }
}

/// The budget of the first block of a grid.
pub(super) fn log_budget<BE: Backend>(ct: &BlasCiphertextOwned<BE>) -> usize {
    ct.blocks()[0].log_budget()
}

/// Records the check of a point's first call, the consumed budget being what `res` holds less than the input's
/// `input_log_budget`.
pub(super) fn check<BE: Backend>(
    bp: &BlasBenchParams,
    max_err: f64,
    max_base2k: Option<usize>,
    keys: usize,
    input_log_budget: usize,
    res: &BlasCiphertextOwned<BE>,
) -> BlasCheck {
    BlasCheck {
        max_err,
        bound: error_bound::<BE::Ring>(bp),
        max_base2k,
        keys,
        consumed_bits: input_log_budget - log_budget::<BE>(res),
        output_log_budget: log_budget::<BE>(res),
        key_layout: atk_layout(bp),
    }
}

/// What a point asks of the evaluated module beyond the operation itself: allocation, the encoding of the maps and
/// their preparation, the preparation of the keys the client made, and the synchronization that ends a timed call.
pub trait BlasEvalModule<BE: Backend>:
    ModuleNew<BE>
    + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = BE::ZnxWord>
    + CKKSModuleAlloc<BE>
    + CKKSModuleInfos
    + CyclotomicOrder
    + GLWEAutomorphismKeyPreparedFactory<BE>
    + GLWETensorKeyPreparedFactory<BE>
    + CKKSEncodingOps<BE, f64>
    + BlasEncodingOps<BE, f64>
    + BlasContextPreparedFactory<BE>
    + BlasPtOperandPreparedFactory<BE>
    + BlasMatVecOperandPreparedFactory<BE>
    + BlasDiagonalCiphertextPreparedFactory<BE>
    + CKKSLinearTransformationOps<BE>
    + CKKSMulOps<BE>
    + CnvPVecAlloc<BE>
    + ModuleSynchronize<BE>
{
}

impl<BE: Backend, M> BlasEvalModule<BE> for M where
    M: ModuleNew<BE>
        + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = BE::ZnxWord>
        + CKKSModuleAlloc<BE>
        + CKKSModuleInfos
        + CyclotomicOrder
        + GLWEAutomorphismKeyPreparedFactory<BE>
        + GLWETensorKeyPreparedFactory<BE>
        + CKKSEncodingOps<BE, f64>
        + BlasEncodingOps<BE, f64>
        + BlasContextPreparedFactory<BE>
        + BlasPtOperandPreparedFactory<BE>
        + BlasMatVecOperandPreparedFactory<BE>
        + BlasDiagonalCiphertextPreparedFactory<BE>
        + CKKSLinearTransformationOps<BE>
        + CKKSMulOps<BE>
        + CnvPVecAlloc<BE>
        + ModuleSynchronize<BE>
{
}

/// What a point asks of the client module: the key generation, encryption and decryption of poulpy-ckks.
pub trait BlasClientModule<BC: Backend>: TestContextSharedModule<BC> + CKKSEncodingOps<BC, f64> {}

impl<BC: Backend, M> BlasClientModule<BC> for M where M: TestContextSharedModule<BC> + CKKSEncodingOps<BC, f64> {}

/// A module that evaluates and is its own client, what a runner asks of a module.
pub trait BlasBenchModule<BE: Backend>: BlasEvalModule<BE> + BlasClientModule<BE> {}

impl<BE: Backend, M> BlasBenchModule<BE> for M where M: BlasEvalModule<BE> + BlasClientModule<BE> {}

/// The two sides of a point: the evaluated module, and the client module with the secret key and one arena for key
/// generation, encoding, encryption and decryption.
pub(super) struct Client<BE: Backend, BC: Backend> {
    pub(super) module: Module<BE>,
    client: Module<BC>,
    sk_raw: BackendGLWESecret<BC>,
    sk: GLWESecretPrepared<BC::OwnedBuf, BC>,
    arena: ScratchOwned<BC>,
    /// The evaluated module's arena for preparing the moved keys.
    prepare: ScratchOwned<BE>,
    xe: Source,
    xa: Source,
}

impl<BE, BC> Client<BE, BC>
where
    BE: Backend + HostStaged + 'static,
    BC: Backend<Ring = BE::Ring> + HostStaged + 'static,
    Module<BE>: BlasEvalModule<BE>,
    Module<BC>: BlasClientModule<BC>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    ScratchOwned<BC>: ScratchOwnedAlloc<BC> + ScratchOwnedBorrow<BC>,
{
    pub(super) fn new(bp: &BlasBenchParams) -> Self {
        let module = Module::<BE>::new(bp.n as u64);
        let client = Module::<BC>::new(bp.n as u64);
        let layout = block_layout::<BE::Ring>(bp);
        let mut xs = Source::new([0u8; 32]);
        let mut sk_raw = client.glwe_secret_alloc_from_infos(&layout);
        client.glwe_secret_fill_ternary_hw(&mut sk_raw, HW, &mut xs);
        let mut sk = client.glwe_secret_prepared_alloc_from_infos(&layout);
        client.glwe_secret_prepare(&mut sk, &sk_raw);
        let bytes = client
            .ckks_all_ops_with_atk_tmp_bytes(&layout, &layout, &tsk_layout(bp), &atk_layout(bp))
            .max(<Module<BC> as CKKSEncodingHostOps<BC, f64>>::ckks_reim_tmp_bytes(
                &client,
                bp.slots_on::<BE::Ring>(),
            ));
        let prepare = module
            .glwe_tensor_key_prepare_tmp_bytes(&tsk_layout(bp))
            .max(module.glwe_automorphism_key_prepare_tmp_bytes(&atk_layout(bp)));
        Self {
            module,
            client,
            sk_raw,
            sk,
            arena: ScratchOwned::alloc(bytes),
            prepare: ScratchOwned::alloc(prepare),
            xe: Source::new([2u8; 32]),
            xa: Source::new([1u8; 32]),
        }
    }

    /// The prepared maps of the context of `plan` at scale `p`.
    pub(super) fn context(&self, bp: &BlasBenchParams, plan: BlasPlan) -> BlasContextPrepared<BE> {
        let module = &self.module;
        let meta = CoeffsMeta::from_delta_budget(bp.p, HEADROOM);
        let mut ctx = module.blas_context_alloc(plan, base2k(bp), meta);
        let mut prepared = module.blas_context_prepared_alloc_from_infos(&ctx);
        let mut arena = ScratchOwned::<BE>::alloc(
            module
                .blas_context_encode_tmp_bytes(&ctx)
                .max(module.blas_context_prepare_tmp_bytes(&ctx)),
        );
        module.blas_context_encode(&mut ctx, &mut arena.borrow()).unwrap();
        module.blas_context_prepare(&mut prepared, &ctx, &mut arena.borrow());
        prepared
    }

    /// The prepared public operand of `rows x cols` blocks of `entry` on side `S` of the encrypted grid, one matrix
    /// shared by every matrix slot.
    pub(super) fn public<S: BlasPtSide>(
        &self,
        bp: &BlasBenchParams,
        algorithm: BlasPtAlgorithm,
        rows: usize,
        cols: usize,
        entry: impl Fn(usize, usize, usize) -> f64,
    ) -> BlasPtOperandPrepared<BE, S> {
        let module = &self.module;
        let plan = BlasPlan::new(bp.log_d).with_pt_algorithm(algorithm);
        let meta = CoeffsMeta::from_delta_budget(bp.p, HEADROOM);
        let mut op = module.blas_pt_operand_alloc::<S>(plan, rows, cols, 1, base2k(bp), meta);
        let mut prepared = module.blas_pt_operand_prepared_alloc_from_infos(&op);
        let mut arena = ScratchOwned::<BE>::alloc(
            module
                .blas_pt_operand_encode_tmp_bytes(&op)
                .max(module.blas_pt_operand_prepare_tmp_bytes(&op)),
        );
        module.blas_pt_operand_encode(&mut op, entry, &mut arena.borrow()).unwrap();
        module.blas_pt_operand_prepare(&mut prepared, &op, &mut arena.borrow());
        prepared
    }

    /// One rotation key for each Galois element, and no other, made on the client and prepared on the evaluated module.
    pub(super) fn rotation_keys(
        &mut self,
        bp: &BlasBenchParams,
        elements: Vec<i64>,
    ) -> HashMap<i64, GLWEAutomorphismKeyPrepared<BE::OwnedBuf, BE>> {
        let layout = atk_layout(bp);
        elements
            .into_iter()
            .collect::<BTreeSet<_>>()
            .into_iter()
            .map(|p| {
                let mut atk = self.client.glwe_automorphism_key_alloc_from_infos(&layout);
                self.client.glwe_automorphism_key_encrypt_sk(
                    &mut atk,
                    p,
                    &self.sk_raw,
                    &mut self.xe,
                    &mut self.xa,
                    &mut self.arena.borrow(),
                );
                let mut moved = self.module.glwe_automorphism_key_alloc_from_infos(&layout);
                atk.transfer_into(&mut moved);
                let mut prepared = self.module.glwe_automorphism_key_prepared_alloc_from_infos(&layout);
                self.module
                    .glwe_automorphism_key_prepare(&mut prepared, &moved, &mut self.prepare.borrow());
                (p, prepared)
            })
            .collect()
    }

    /// The tensor key, made on the client and prepared on the evaluated module.
    pub(super) fn tensor_key(&mut self, bp: &BlasBenchParams) -> GLWETensorKeyPrepared<BE::OwnedBuf, BE> {
        let layout = tsk_layout(bp);
        let mut tsk = self.client.glwe_tensor_key_alloc_from_infos(&layout);
        self.client
            .glwe_tensor_key_encrypt_sk(&mut tsk, &self.sk_raw, &mut self.xe, &mut self.xa, &mut self.arena.borrow());
        let mut moved = self.module.glwe_tensor_key_alloc_from_infos(&layout);
        tsk.transfer_into(&mut moved);
        let mut prepared = self.module.glwe_tensor_key_prepared_alloc_from_infos(&layout);
        self.module
            .glwe_tensor_key_prepare(&mut prepared, &moved, &mut self.prepare.borrow());
        prepared
    }

    /// A `rows x cols` grid holding `matrices` matrices of `entry`, each block from a plaintext of `matrices d^2` slots,
    /// repeated up to the backend's minimum degree: the blocks' label proves their number, or the larger batch that
    /// degree holds. Each block is encrypted on the client and moved to the evaluated module.
    pub(super) fn encrypt(
        &mut self,
        bp: &BlasBenchParams,
        rows: usize,
        cols: usize,
        matrices: usize,
        entry: impl Fn(usize, usize, usize) -> f64,
    ) -> BlasCiphertextOwned<BE> {
        let slots = matrices << (2 * bp.log_d);
        let held = slots.max(self.module.ckks_slots_of_degree(BE::MIN_DEGREE));
        let mut pt = self
            .client
            .ckks_pt_vec_alloc(held, base2k(bp), TorusPrecision((bp.log_delta + HEADROOM) as u32));
        pt.set_meta(block_layout::<BE::Ring>(bp).meta);
        let im = vec![0.0; held];
        let blocks = split_into_blocks(rows, cols, bp.log_d, slots, matrices, entry)
            .into_iter()
            .map(|values| {
                self.client
                    .ckks_encode_reim_into(&mut pt, &values.repeat(held / slots), &im, &mut self.arena.borrow())
                    .unwrap();
                let mut ct = self.client.ckks_ciphertext_alloc(base2k(bp), TorusPrecision(bp.k as u32));
                self.client
                    .ckks_encrypt_sk(&mut ct, &pt, &self.sk, &mut self.xe, &mut self.xa, &mut self.arena.borrow())
                    .unwrap();
                let mut moved = self.module.ckks_ciphertext_alloc(base2k(bp), TorusPrecision(bp.k as u32));
                (*ct).transfer_into(&mut *moved);
                moved.set_meta(ct.meta());
                moved
            })
            .collect();
        BlasCiphertext::from_blocks(blocks, bp.log_d, rows, cols).unwrap()
    }

    /// A result grid of `rows x cols` blocks at the width of a fresh ciphertext.
    pub(super) fn result(&self, bp: &BlasBenchParams, rows: usize, cols: usize) -> BlasCiphertextOwned<BE> {
        self.module
            .blas_ciphertext_alloc(bp.log_d, rows, cols, base2k(bp), TorusPrecision(bp.k as u32))
    }

    /// A ciphertext of the slot values `re`, every slot of the ring, encrypted on the client as [`Self::encrypt`]'s blocks
    /// and moved to the evaluated module.
    pub(super) fn encrypt_slots(&mut self, bp: &BlasBenchParams, re: &[f64]) -> CKKSCiphertextOwned<BE> {
        let slots = bp.slots_on::<BE::Ring>();
        let mut pt = self
            .client
            .ckks_pt_vec_alloc(slots, base2k(bp), TorusPrecision((bp.log_delta + HEADROOM) as u32));
        pt.set_meta(block_layout::<BE::Ring>(bp).meta);
        self.client
            .ckks_encode_reim_into(&mut pt, re, &vec![0.0; slots], &mut self.arena.borrow())
            .unwrap();
        let mut ct = self.client.ckks_ciphertext_alloc(base2k(bp), TorusPrecision(bp.k as u32));
        self.client
            .ckks_encrypt_sk(&mut ct, &pt, &self.sk, &mut self.xe, &mut self.xa, &mut self.arena.borrow())
            .unwrap();
        let mut moved = self.module.ckks_ciphertext_alloc(base2k(bp), TorusPrecision(bp.k as u32));
        (*ct).transfer_into(&mut *moved);
        moved.set_meta(ct.meta());
        moved
    }

    /// The real and imaginary slot values of `moved`, moved back to the client and decrypted there.
    pub(super) fn decrypt_slots(&mut self, bp: &BlasBenchParams, moved: &CKKSCiphertextOwned<BE>) -> (Vec<f64>, Vec<f64>) {
        let slots = bp.slots_on::<BE::Ring>();
        // The allocation keeps the fresh width, the operation narrowed `k`.
        let mut ct = self.client.ckks_ciphertext_alloc(base2k(bp), TorusPrecision(bp.k as u32));
        ct.set_k(moved.k());
        (**moved).transfer_into(&mut *ct);
        ct.set_meta(moved.meta());
        let log_delta = ct.log_delta();
        let mut pt = self
            .client
            .ckks_pt_vec_alloc(slots, base2k(bp), TorusPrecision((log_delta + DECODE_HEADROOM) as u32));
        pt.set_meta(GLWEPlaintextMeta {
            scale: Scale::Log(log_delta),
            slots: slots_kind::<BE::Ring>(),
            log_sparsity: 0,
        });
        self.client
            .ckks_decrypt(&mut pt, &ct, &self.sk, &mut self.arena.borrow())
            .unwrap();
        let (mut re, mut im) = (vec![0.0; slots], vec![0.0; slots]);
        self.client
            .ckks_decode_reim_into(&pt, &mut re, &mut im, &mut self.arena.borrow())
            .unwrap();
        (re, im)
    }

    /// The largest distance of the decrypted grid `res` to `want(s, r, c)`, entry `(r, c)` of matrix `s`, over the
    /// matrices of the batch, and of its imaginary part to zero. Each block is moved back to the client and decrypted there.
    pub(super) fn max_error(
        &mut self,
        bp: &BlasBenchParams,
        res: &BlasCiphertextOwned<BE>,
        want: impl Fn(usize, usize, usize) -> f64,
    ) -> f64 {
        let (re, im): (Vec<_>, Vec<_>) = res.blocks().iter().map(|block| self.decrypt_slots(bp, block)).unzip();
        let (rows, cols) = (res.rows(), res.cols());
        let mut err = 0.0f64;
        let batch = bp.batch_on::<BE::Ring>();
        join_blocks(&re, rows, cols, bp.log_d, batch, |s, r, c, have| {
            err = err.max((have - want(s, r, c)).abs())
        });
        join_blocks(&im, rows, cols, bp.log_d, batch, |_, _, _, have| err = err.max(have.abs()));
        err
    }
}

/// The point of `blas_transpose` of a grid of `batch` matrices.
pub fn point_blas_transpose<BE, BC>(bp: &BlasBenchParams) -> BlasPoint
where
    BE: Backend + HostStaged + 'static + BlasMatIntraImpl,
    BC: Backend<Ring = BE::Ring> + HostStaged + 'static,
    Module<BE>: BlasEvalModule<BE>,
    Module<BC>: BlasClientModule<BC>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    ScratchOwned<BC>: ScratchOwnedAlloc<BC> + ScratchOwnedBorrow<BC>,
{
    let mut client = Client::<BE, BC>::new(bp);
    let ctx = client.context(bp, BlasPlan::new(bp.log_d));
    let ct_a = client.encrypt(bp, 1, 1, bp.batch_on::<BE::Ring>(), a_entry);
    let order = client.module.cyclotomic_order();
    let atks = client.rotation_keys(bp, BlasPlan::new(bp.log_d).transpose_galois_elements(order));
    let mut res = client.result(bp, 1, 1);
    let mut scratch = ScratchOwned::<BE>::alloc(client.module.blas_transpose_tmp_bytes(&res, &ct_a, &ctx, &atk_layout(bp)));
    client
        .module
        .blas_transpose(&mut res, &ct_a, &ctx, &atks, &mut scratch.borrow())
        .unwrap();
    let max_err = client.max_error(bp, &res, |s, r, c| a_entry(s, c, r));
    let check = check::<BE>(bp, max_err, None, atks.len(), log_budget::<BE>(&ct_a), &res);
    let module = client.module;
    BlasPoint {
        call: Box::new(move || {
            module
                .blas_transpose(&mut res, &ct_a, &ctx, &atks, &mut scratch.borrow())
                .unwrap();
            module.synchronize();
        }),
        check,
    }
}

/// `blas_transpose` of a grid of `batch` matrices.
pub fn runner_blas_transpose<BE, M>(bencher: &mut Bencher<'_, M>, bp: &BlasBenchParams)
where
    BE: Backend<Ring = Standard> + HostStaged + 'static + BlasMatIntraImpl,
    Module<BE>: BlasBenchModule<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    M: Measurement,
{
    timed::<BE, M, _>(bencher, "blas_transpose", bp, || {
        point_blas_transpose::<BE, BE>(bp).reported::<BE>("blas_transpose", bp)
    });
}

/// The point of `blas_matmul` of `bp.batch()` matrices by as many distinct matrices on the right (`dense`), or by one
/// matrix encrypted from a plaintext of `d^2` slots, which proves it, so that the right factor takes the replicated maps.
pub fn point_blas_matmul<BE, BC>(bp: &BlasBenchParams, dense: bool) -> BlasPoint
where
    BE: Backend + HostStaged + 'static + BlasMatMulImpl + MaxBase2k,
    BC: Backend<Ring = BE::Ring> + HostStaged + 'static,
    Module<BE>: BlasEvalModule<BE>,
    Module<BC>: BlasClientModule<BC>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    ScratchOwned<BC>: ScratchOwnedAlloc<BC> + ScratchOwnedBorrow<BC>,
{
    let mut client = Client::<BE, BC>::new(bp);
    let ctx = client.context(bp, BlasPlan::new(bp.log_d));
    let right = if dense { bp.batch_on::<BE::Ring>() } else { 1 };
    let ct_a = client.encrypt(bp, 1, 1, bp.batch_on::<BE::Ring>(), a_entry);
    let ct_b = client.encrypt(bp, 1, 1, right, b_entry(bp.d()));
    assert!(
        replicated::<BE::Ring>(&ct_b) != dense,
        "blas_matmul: the right factor is not the one the runner names"
    );
    let max_base2k = blas_matmul_max_base2k::<BE>(&ct_a, &ct_b, &ctx, FAILURE_BITS);
    assert!(
        max_base2k >= bp.base2k,
        "blas_matmul: base2k exceeds the radix its lazy sums allow at 128 failure bits"
    );
    let order = client.module.cyclotomic_order();
    let atks = client.rotation_keys(bp, BlasPlan::new(bp.log_d).matmul_galois_elements(order));
    let tsk = client.tensor_key(bp);
    let mut res = client.result(bp, 1, 1);
    let mut scratch = ScratchOwned::<BE>::alloc(client.module.blas_matmul_tmp_bytes(
        &res,
        &ct_a,
        &ct_b,
        &ctx,
        &tsk_layout(bp),
        &atk_layout(bp),
    ));
    client
        .module
        .blas_matmul(&mut res, &ct_a, &ct_b, &ctx, &tsk, &atks, &mut scratch.borrow())
        .unwrap();
    let b = b_entry(bp.d());
    let max_err = client.max_error(bp, &res, |s, r, c| {
        (0..bp.d()).map(|l| a_entry(s, r, l) * b(s % right, l, c)).sum()
    });
    let check = check::<BE>(bp, max_err, Some(max_base2k), atks.len(), log_budget::<BE>(&ct_a), &res);
    let module = client.module;
    BlasPoint {
        call: Box::new(move || {
            module
                .blas_matmul(&mut res, &ct_a, &ct_b, &ctx, &tsk, &atks, &mut scratch.borrow())
                .unwrap();
            module.synchronize();
        }),
        check,
    }
}

fn matmul<BE, M>(bencher: &mut Bencher<'_, M>, bp: &BlasBenchParams, op: &'static str, dense: bool)
where
    BE: Backend<Ring = Standard> + HostStaged + 'static + BlasMatMulImpl + MaxBase2k,
    Module<BE>: BlasBenchModule<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    M: Measurement,
{
    timed::<BE, M, _>(bencher, op, bp, || {
        point_blas_matmul::<BE, BE>(bp, dense).reported::<BE>(op, bp)
    });
}

/// `blas_matmul` with a dense right factor, distinct matrices per matrix slot: needs `bp.batch() > 1`.
pub fn runner_blas_matmul<BE, M>(bencher: &mut Bencher<'_, M>, bp: &BlasBenchParams)
where
    BE: Backend<Ring = Standard> + HostStaged + 'static + BlasMatMulImpl + MaxBase2k,
    Module<BE>: BlasBenchModule<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    M: Measurement,
{
    matmul::<BE, M>(bencher, bp, "blas_matmul", true);
}

/// `blas_matmul` with a replicated right factor, one matrix in every matrix slot.
pub fn runner_blas_matmul_replicated<BE, M>(bencher: &mut Bencher<'_, M>, bp: &BlasBenchParams)
where
    BE: Backend<Ring = Standard> + HostStaged + 'static + BlasMatMulImpl + MaxBase2k,
    Module<BE>: BlasBenchModule<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    M: Measurement,
{
    matmul::<BE, M>(bencher, bp, "blas_matmul_replicated", false);
}

/// The point of `blas_matmul_pt_right` by the public matrix under `algorithm`.
pub fn point_blas_matmul_pt_right<BE, BC>(bp: &BlasBenchParams, algorithm: BlasPtAlgorithm) -> BlasPoint
where
    BE: Backend + HostStaged + 'static + BlasMatMulImpl + MaxBase2k,
    BC: Backend<Ring = BE::Ring> + HostStaged + 'static,
    Module<BE>: BlasEvalModule<BE>,
    Module<BC>: BlasClientModule<BC>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    ScratchOwned<BC>: ScratchOwnedAlloc<BC> + ScratchOwnedBorrow<BC>,
{
    let mut client = Client::<BE, BC>::new(bp);
    let pt = client.public::<Right>(bp, algorithm, 1, 1, p_entry(bp.d()));
    let ctx = client.context(bp, BlasPlan::new(bp.log_d));
    let ct_a = client.encrypt(bp, 1, 1, bp.batch_on::<BE::Ring>(), a_entry);
    let max_base2k = blas_matmul_pt_right_max_base2k::<BE>(&ct_a, &pt, &ctx, FAILURE_BITS);
    assert!(
        max_base2k >= bp.base2k,
        "blas_matmul_pt_right: base2k exceeds the radix its lazy sums allow at 128 failure bits"
    );
    let order = client.module.cyclotomic_order();
    let atks = client.rotation_keys(bp, BlasPlan::new(bp.log_d).matmul_pt_right_galois_elements(order));
    let mut res = client.result(bp, 1, 1);
    let bytes = client
        .module
        .blas_matmul_pt_right_tmp_bytes(&res, &ct_a, &pt, &ctx, &atk_layout(bp));
    let mut scratch = ScratchOwned::<BE>::alloc(bytes);
    client
        .module
        .blas_matmul_pt_right(&mut res, &ct_a, &pt, &ctx, &atks, &mut scratch.borrow())
        .unwrap();
    let p = p_entry(bp.d());
    let max_err = client.max_error(bp, &res, |s, r, c| (0..bp.d()).map(|l| a_entry(s, r, l) * p(0, l, c)).sum());
    let check = check::<BE>(bp, max_err, Some(max_base2k), atks.len(), log_budget::<BE>(&ct_a), &res);
    let module = client.module;
    BlasPoint {
        call: Box::new(move || {
            module
                .blas_matmul_pt_right(&mut res, &ct_a, &pt, &ctx, &atks, &mut scratch.borrow())
                .unwrap();
            module.synchronize();
        }),
        check,
    }
}

fn pt_right<BE, M>(bencher: &mut Bencher<'_, M>, bp: &BlasBenchParams, op: &'static str, algorithm: BlasPtAlgorithm)
where
    BE: Backend<Ring = Standard> + HostStaged + 'static + BlasMatMulImpl + MaxBase2k,
    Module<BE>: BlasBenchModule<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    M: Measurement,
{
    timed::<BE, M, _>(bencher, op, bp, || {
        point_blas_matmul_pt_right::<BE, BE>(bp, algorithm).reported::<BE>(op, bp)
    });
}

/// `blas_matmul_pt_right` under `Diagonal`.
pub fn runner_blas_matmul_pt_right_diagonal<BE, M>(bencher: &mut Bencher<'_, M>, bp: &BlasBenchParams)
where
    BE: Backend<Ring = Standard> + HostStaged + 'static + BlasMatMulImpl + MaxBase2k,
    Module<BE>: BlasBenchModule<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    M: Measurement,
{
    pt_right::<BE, M>(bencher, bp, "blas_matmul_pt_right_diagonal", BlasPtAlgorithm::Diagonal);
}

/// `blas_matmul_pt_right` under `Jkls`.
pub fn runner_blas_matmul_pt_right_jkls<BE, M>(bencher: &mut Bencher<'_, M>, bp: &BlasBenchParams)
where
    BE: Backend<Ring = Standard> + HostStaged + 'static + BlasMatMulImpl + MaxBase2k,
    Module<BE>: BlasBenchModule<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    M: Measurement,
{
    pt_right::<BE, M>(bencher, bp, "blas_matmul_pt_right_jkls", BlasPtAlgorithm::Jkls);
}

/// `blas_matmul_pt_left` by the public matrix under `algorithm`.
fn pt_left<BE, M>(bencher: &mut Bencher<'_, M>, bp: &BlasBenchParams, op: &'static str, algorithm: BlasPtAlgorithm)
where
    BE: Backend<Ring = Standard> + HostStaged + 'static + BlasMatMulImpl + MaxBase2k,
    Module<BE>: BlasBenchModule<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    M: Measurement,
{
    timed::<BE, M, _>(bencher, op, bp, || {
        let mut client = Client::<BE, BE>::new(bp);
        let pt = client.public::<Left>(bp, algorithm, 1, 1, p_entry(bp.d()));
        let ctx = client.context(bp, BlasPlan::new(bp.log_d));
        let ct_a = client.encrypt(bp, 1, 1, bp.batch(), a_entry);
        let max_base2k = blas_matmul_pt_left_max_base2k::<BE>(&pt, &ct_a, &ctx, FAILURE_BITS);
        assert!(
            max_base2k >= bp.base2k,
            "blas_matmul_pt_left: base2k exceeds the radix its lazy sums allow at 128 failure bits"
        );
        let order = client.module.cyclotomic_order();
        let atks = client.rotation_keys(bp, BlasPlan::new(bp.log_d).matmul_pt_left_galois_elements(order));
        let mut res = client.result(bp, 1, 1);
        let bytes = client
            .module
            .blas_matmul_pt_left_tmp_bytes(&res, &pt, &ct_a, &ctx, &atk_layout(bp));
        let mut scratch = ScratchOwned::<BE>::alloc(bytes);
        client
            .module
            .blas_matmul_pt_left(&mut res, &pt, &ct_a, &ctx, &atks, &mut scratch.borrow())
            .unwrap();
        let p = p_entry(bp.d());
        let max_err = client.max_error(bp, &res, |s, r, c| (0..bp.d()).map(|l| p(0, r, l) * a_entry(s, l, c)).sum());
        verify::<BE>(op, bp, max_err, Some(max_base2k), atks.len());
        let module = client.module;
        Box::new(move || {
            module
                .blas_matmul_pt_left(&mut res, &pt, &ct_a, &ctx, &atks, &mut scratch.borrow())
                .unwrap();
            module.synchronize();
        })
    });
}

/// `blas_matmul_pt_left` under `Diagonal`.
pub fn runner_blas_matmul_pt_left_diagonal<BE, M>(bencher: &mut Bencher<'_, M>, bp: &BlasBenchParams)
where
    BE: Backend<Ring = Standard> + HostStaged + 'static + BlasMatMulImpl + MaxBase2k,
    Module<BE>: BlasBenchModule<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    M: Measurement,
{
    pt_left::<BE, M>(bencher, bp, "blas_matmul_pt_left_diagonal", BlasPtAlgorithm::Diagonal);
}

/// `blas_matmul_pt_left` under `Jkls`.
pub fn runner_blas_matmul_pt_left_jkls<BE, M>(bencher: &mut Bencher<'_, M>, bp: &BlasBenchParams)
where
    BE: Backend<Ring = Standard> + HostStaged + 'static + BlasMatMulImpl + MaxBase2k,
    Module<BE>: BlasBenchModule<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    M: Measurement,
{
    pt_left::<BE, M>(bencher, bp, "blas_matmul_pt_left_jkls", BlasPtAlgorithm::Jkls);
}
