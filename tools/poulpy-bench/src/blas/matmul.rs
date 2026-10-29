//! Runners for the `poulpy-blas` operations on one `d x d` block per operand.
//!
//! A runner builds its whole point once, outside the timed loop: the module, the secret key, the context or the public
//! operand, only the rotation keys its operation looks up (plus the tensor key of a ciphertext by ciphertext product),
//! the encrypted inputs and the scratch its `*_tmp_bytes` reports. A point builder takes a client module beside the
//! evaluated one, which makes the keys and the ciphertexts and decrypts, the keys and ciphertexts moving between them:
//! the runners pass the evaluated module itself, a GPU module takes a host client. It checks that `base2k` fits the lazy sums of the
//! operation, calls the operation once, decrypts the result and asserts every entry within [`error_bound`] of the
//! cleartext, then times that same call.
//!
//! Entries of the encrypted matrices lie in `[-1, 1)`, those of the second ciphertext factor and of the public matrices
//! in `[-1/d, 1/d)`, so that the products stay in `[-1, 1)`. A public matrix is one matrix shared by every matrix slot.

use std::collections::{BTreeSet, HashMap};

use criterion::{Bencher, measurement::Measurement};
use poulpy_blas::{
    api::{BlasMatIntraOps, BlasMatMulOps},
    layouts::{
        BlasCiphertext, BlasCiphertextOwned, BlasContext, BlasContextPreparedFactory, BlasEncodingOps, BlasModuleAlloc, BlasPlan,
        BlasPtAlgorithm, BlasPtOperand, BlasPtOperandPreparedFactory, BlasPtSide, Left, Right,
        grid::{join_blocks, split_into_blocks},
        replicated,
    },
    oep::{BlasMatIntraImpl, BlasMatMulImpl},
    reference::{blas_matmul_max_base2k, blas_matmul_pt_left_max_base2k, blas_matmul_pt_right_max_base2k},
};
use poulpy_ckks::{
    CKKSInfos, CKKSLayout, CoeffsMeta, GLWEPlaintextMeta, Scale, SetCKKSInfos, SlotsKind,
    api::{
        CKKSAllOpsTmpBytes, CKKSDecryptOps, CKKSEncodingHostOps, CKKSEncodingOps, CKKSEncryptOps, CKKSLinearTransformationOps,
        CKKSModuleInfos, CKKSMulOps,
    },
    layouts::CKKSModuleAlloc,
    test_suite::helpers::TestContextSharedModule,
};
use poulpy_core::{
    GLWEAutomorphismKeyEncryptSk, GLWETensorKeyEncryptSk,
    api::{
        GLWEAutomorphismKeyPreparedFactory, GLWESecretPreparedFactory, GLWESecretSampling, GLWETensorKeyPreparedFactory,
        TransferInto,
    },
    layouts::{
        BackendGLWESecret, Base2K, Degree, Dnum, Dsize, GGLWELayout, GLWEAutomorphismKeyLayout, GLWELayout, GLWETensorKeyLayout,
        LWEInfos, ModuleCoreAlloc, Rank, SetK, TorusPrecision, k_digit_of,
        prepared::{GLWEAutomorphismKeyPrepared, GLWESecretPrepared, GLWETensorKeyPrepared},
    },
};
use poulpy_hal::{
    api::{CnvPVecAlloc, ModuleNew, ModuleSynchronize, ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Backend, CyclotomicOrder, HostStaged, MaxBase2k, Module, ScratchOwned, Standard},
    source::Source,
};

use crate::{Call, backend_name, blas::params::BlasBenchParams, timed};

/// Secret-key Hamming weight.
const HW: usize = 192;
/// Bits above the scale that an encoded entry, map or public matrix keeps for a magnitude of at most one.
const HEADROOM: usize = 4;
/// Integer bits kept when decrypting, for the sums of `d` products of magnitude at most one.
const DECODE_HEADROOM: usize = 24;
/// The failure target of the radix check, in bits.
const FAILURE_BITS: usize = 128;

fn base2k(bp: &BlasBenchParams) -> Base2K {
    Base2K(bp.base2k as u32)
}

fn block_layout(bp: &BlasBenchParams) -> CKKSLayout {
    CKKSLayout {
        glwe_layout: GLWELayout {
            n: Degree(bp.n as u32),
            base2k: base2k(bp),
            k: TorusPrecision(bp.k as u32),
            rank: Rank(1),
        },
        meta: GLWEPlaintextMeta {
            scale: Scale::Log(bp.log_delta),
            slots: SlotsKind::Complex,
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

pub fn atk_layout(bp: &BlasBenchParams) -> GLWEAutomorphismKeyLayout {
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

pub fn tsk_layout(bp: &BlasBenchParams) -> GLWETensorKeyLayout {
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
fn entry(tag: u64, s: usize, r: usize, c: usize) -> f64 {
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

/// The largest error a result may show against the cleartext: `2^7 sqrt(n d) 2^-log_delta`, a precision of
/// `log_delta - log2(n d) / 2 - 7` bits.
///
/// A fresh encryption, and each key switch through the rounding of its output by a secret of Hamming weight 192, leave a
/// few quanta (`2^-log_delta`) of noise per coefficient, `u = sqrt(n / 2) 2^-log_delta` on a slot. A product sums `d`
/// terms of an entry of magnitude below one by such noise: the maxima measured over the whole sweep are at most
/// `26 sqrt(d) u`, a seventh of the bound, which is `181 sqrt(d) u`. A radix that spoils whole limbs, or a key of another
/// secret, lands far above it.
pub fn error_bound(bp: &BlasBenchParams) -> f64 {
    (7.0 - bp.log_delta as f64).exp2() * ((bp.n * bp.d()) as f64).sqrt()
}

/// What the first call of a point showed: its error against the cleartext and the bound it met, the radix the lazy sums
/// of its operation allow, the rotation keys it holds, the budget its operation consumed and the budget its result keeps.
pub struct BlasCheck {
    pub max_err: f64,
    pub bound: f64,
    pub max_base2k: Option<usize>,
    pub keys: usize,
    pub consumed_bits: usize,
    pub output_log_budget: usize,
}

/// A built point: one call of its operation, ended by a synchronization of the module, and the check of its first call.
pub struct BlasPoint {
    pub call: Call,
    pub check: BlasCheck,
}

impl BlasPoint {
    /// Prints the check, as the Criterion runners report it, and returns the call.
    pub(super) fn reported<BE>(self, op: &str, bp: &BlasBenchParams) -> Call {
        let check = &self.check;
        println!(
            "PRECISION backend={} op={op} params={bp} max_err={:.3e} ({:.1} bits) bound={:.3e} ({:.1} bits) max_base2k={} keys={}",
            backend_name::<BE>(),
            check.max_err,
            -check.max_err.log2(),
            check.bound,
            -check.bound.log2(),
            check.max_base2k.map_or_else(|| "-".to_string(), |max| max.to_string()),
            check.keys,
        );
        self.call
    }
}

/// The budget of the one block of a grid.
pub(super) fn log_budget<BE: Backend>(ct: &BlasCiphertextOwned<BE>) -> usize {
    ct.blocks()[0].log_budget()
}

/// Asserts the error of a point's first call within [`error_bound`] and records its check, the consumed budget being
/// what `res` holds less than the input's `input_log_budget`.
pub(super) fn verify<BE: Backend>(
    op: &str,
    bp: &BlasBenchParams,
    max_err: f64,
    max_base2k: Option<usize>,
    keys: usize,
    input_log_budget: usize,
    res: &BlasCiphertextOwned<BE>,
) -> BlasCheck {
    let bound = error_bound(bp);
    assert!(
        max_err <= bound,
        "{op}: the first call deviates from the cleartext beyond the error bound"
    );
    BlasCheck {
        max_err,
        bound,
        max_base2k,
        keys,
        consumed_bits: input_log_budget - log_budget::<BE>(res),
        output_log_budget: log_budget::<BE>(res),
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
    BE: Backend<Ring = Standard> + HostStaged + 'static,
    BC: Backend<Ring = Standard> + HostStaged + 'static,
    Module<BE>: BlasEvalModule<BE>,
    Module<BC>: BlasClientModule<BC>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    ScratchOwned<BC>: ScratchOwnedAlloc<BC> + ScratchOwnedBorrow<BC>,
{
    pub(super) fn new(bp: &BlasBenchParams) -> Self {
        let module = Module::<BE>::new(bp.n as u64);
        let client = Module::<BC>::new(bp.n as u64);
        let layout = block_layout(bp);
        let mut xs = Source::new([0u8; 32]);
        let mut sk_raw = client.glwe_secret_alloc_from_infos(&layout);
        client.glwe_secret_fill_ternary_hw(&mut sk_raw, HW, &mut xs);
        let mut sk = client.glwe_secret_prepared_alloc_from_infos(&layout);
        client.glwe_secret_prepare(&mut sk, &sk_raw);
        let bytes = client
            .ckks_all_ops_with_atk_tmp_bytes(&layout, &layout, &tsk_layout(bp), &atk_layout(bp))
            .max(<Module<BC> as CKKSEncodingHostOps<BC, f64>>::ckks_reim_tmp_bytes(
                &client,
                bp.slots(),
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

    /// The prepared maps of the context at scale `p`.
    pub(super) fn context(&self, bp: &BlasBenchParams) -> BlasContext<BE> {
        let module = &self.module;
        let meta = CoeffsMeta::from_delta_budget(bp.p, HEADROOM);
        let mut ctx = module.blas_context_alloc(BlasPlan::new(bp.log_d), base2k(bp), meta);
        let mut prepared = module.blas_context_prepared_alloc_from_infos(&ctx);
        let mut arena = ScratchOwned::<BE>::alloc(
            module
                .blas_context_encode_tmp_bytes::<f64, _>(&ctx)
                .max(module.blas_context_prepare_tmp_bytes(&ctx)),
        );
        module.blas_context_encode_into::<f64>(&mut ctx, &mut arena.borrow()).unwrap();
        module.blas_context_prepare(&mut prepared, &ctx, &mut arena.borrow());
        prepared
    }

    /// The prepared public matrix on side `S` of the encrypted grid, one matrix shared by every matrix slot.
    fn public<S: BlasPtSide>(&self, bp: &BlasBenchParams, algorithm: BlasPtAlgorithm) -> BlasPtOperand<BE, S> {
        let module = &self.module;
        let plan = BlasPlan::new(bp.log_d).with_pt_algorithm(algorithm);
        let meta = CoeffsMeta::from_delta_budget(bp.p, HEADROOM);
        let mut op = module.blas_pt_operand_alloc::<S>(plan, 1, 1, 1, base2k(bp), meta);
        let mut prepared = module.blas_pt_operand_prepared_alloc_from_infos(&op);
        let mut arena = ScratchOwned::<BE>::alloc(
            module
                .blas_pt_operand_encode_tmp_bytes::<f64, _>(&op)
                .max(module.blas_pt_operand_prepare_tmp_bytes(&op)),
        );
        module
            .blas_pt_operand_encode_into(&mut op, p_entry(bp.d()), &mut arena.borrow())
            .unwrap();
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
    fn tensor_key(&mut self, bp: &BlasBenchParams) -> GLWETensorKeyPrepared<BE::OwnedBuf, BE> {
        let layout = tsk_layout(bp);
        let mut tsk = self.client.glwe_tensor_key_alloc_from_infos(&layout);
        self.client
            .glwe_tensor_key_encrypt_sk(&mut tsk, &self.sk_raw, &mut self.xe, &mut self.xa, &mut self.arena.borrow());
        let mut moved = self.module.glwe_tensor_key_alloc_from_infos(&layout);
        tsk.transfer_into(&mut moved);
        let mut prepared = self.module.alloc_tensor_key_prepared_from_infos(&layout);
        self.module
            .glwe_tensor_key_prepare(&mut prepared, &moved, &mut self.prepare.borrow());
        prepared
    }

    /// One block holding `matrices` matrices of `entry`, from a plaintext of `matrices d^2` slots, repeated up to the
    /// backend's minimum degree: the block's label proves their number, or the larger batch that degree holds. It is
    /// encrypted on the client and moved to the evaluated module.
    pub(super) fn encrypt(
        &mut self,
        bp: &BlasBenchParams,
        matrices: usize,
        entry: impl Fn(usize, usize, usize) -> f64,
    ) -> BlasCiphertextOwned<BE> {
        let slots = matrices << (2 * bp.log_d);
        let held = slots.max(self.module.ckks_slots_of_degree(BE::MIN_DEGREE));
        let values = split_into_blocks(1, 1, bp.log_d, slots, matrices, entry)
            .remove(0)
            .repeat(held / slots);
        let mut pt = self
            .client
            .ckks_pt_vec_alloc(held, base2k(bp), TorusPrecision((bp.log_delta + HEADROOM) as u32));
        pt.set_meta(block_layout(bp).meta);
        self.client
            .ckks_encode_reim_into(&mut pt, &values, &vec![0.0; held], &mut self.arena.borrow())
            .unwrap();
        let mut ct = self.client.ckks_ciphertext_alloc(base2k(bp), TorusPrecision(bp.k as u32));
        self.client
            .ckks_encrypt_sk(&mut ct, &pt, &self.sk, &mut self.xe, &mut self.xa, &mut self.arena.borrow())
            .unwrap();
        let mut moved = self.module.ckks_ciphertext_alloc(base2k(bp), TorusPrecision(bp.k as u32));
        (*ct).transfer_into(&mut *moved);
        moved.set_meta(ct.meta());
        BlasCiphertext::from_blocks(vec![moved], bp.log_d, 1, 1).unwrap()
    }

    /// A result grid of one block at the width of a fresh ciphertext.
    pub(super) fn result(&self, bp: &BlasBenchParams) -> BlasCiphertextOwned<BE> {
        self.module
            .blas_ciphertext_alloc(bp.log_d, 1, 1, base2k(bp), TorusPrecision(bp.k as u32))
    }

    /// The largest distance of the decrypted block of `res` to `want(s, r, c)`, entry `(r, c)` of matrix `s`, over the
    /// matrices of the batch, and of its imaginary part to zero. The block is moved back to the client and decrypted there.
    pub(super) fn max_error(
        &mut self,
        bp: &BlasBenchParams,
        res: &BlasCiphertextOwned<BE>,
        want: impl Fn(usize, usize, usize) -> f64,
    ) -> f64 {
        // The allocation keeps the fresh width, the operation narrowed `k`.
        let mut block = self.client.ckks_ciphertext_alloc(base2k(bp), TorusPrecision(bp.k as u32));
        block.set_k(res.blocks()[0].k());
        (*res.blocks()[0]).transfer_into(&mut *block);
        block.set_meta(res.blocks()[0].meta());
        let log_delta = block.log_delta();
        let mut pt = self
            .client
            .ckks_pt_vec_alloc(bp.slots(), base2k(bp), TorusPrecision((log_delta + DECODE_HEADROOM) as u32));
        pt.set_meta(GLWEPlaintextMeta {
            scale: Scale::Log(log_delta),
            slots: SlotsKind::Complex,
            log_sparsity: 0,
        });
        self.client
            .ckks_decrypt(&mut pt, &block, &self.sk, &mut self.arena.borrow())
            .unwrap();
        let (mut re, mut im) = (vec![0.0; bp.slots()], vec![0.0; bp.slots()]);
        self.client
            .ckks_decode_reim_into(&pt, &mut re, &mut im, &mut self.arena.borrow())
            .unwrap();
        let mut err = 0.0f64;
        join_blocks(&[re], 1, 1, bp.log_d, bp.batch(), |s, r, c, have| {
            err = err.max((have - want(s, r, c)).abs())
        });
        join_blocks(&[im], 1, 1, bp.log_d, bp.batch(), |_, _, _, have| err = err.max(have.abs()));
        err
    }
}

/// The point of `blas_transpose` of a grid of `batch` matrices.
pub fn point_blas_transpose<BE, BC>(bp: &BlasBenchParams) -> BlasPoint
where
    BE: Backend<Ring = Standard> + HostStaged + 'static + BlasMatIntraImpl,
    BC: Backend<Ring = Standard> + HostStaged + 'static,
    Module<BE>: BlasEvalModule<BE>,
    Module<BC>: BlasClientModule<BC>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    ScratchOwned<BC>: ScratchOwnedAlloc<BC> + ScratchOwnedBorrow<BC>,
{
    let mut client = Client::<BE, BC>::new(bp);
    let ctx = client.context(bp);
    let ct_a = client.encrypt(bp, bp.batch(), a_entry);
    let order = client.module.cyclotomic_order();
    let atks = client.rotation_keys(bp, BlasPlan::new(bp.log_d).transpose_galois_elements(order));
    let mut res = client.result(bp);
    let mut scratch = ScratchOwned::<BE>::alloc(client.module.blas_transpose_tmp_bytes(&res, &ct_a, &ctx, &atk_layout(bp)));
    client
        .module
        .blas_transpose(&mut res, &ct_a, &ctx, &atks, &mut scratch.borrow())
        .unwrap();
    let max_err = client.max_error(bp, &res, |s, r, c| a_entry(s, c, r));
    let check = verify::<BE>("blas_transpose", bp, max_err, None, atks.len(), log_budget::<BE>(&ct_a), &res);
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
/// matrix whose compact encryption proves it, so that the right factor takes the replicated maps.
pub fn point_blas_matmul<BE, BC>(bp: &BlasBenchParams, dense: bool) -> BlasPoint
where
    BE: Backend<Ring = Standard> + HostStaged + 'static + BlasMatMulImpl + MaxBase2k,
    BC: Backend<Ring = Standard> + HostStaged + 'static,
    Module<BE>: BlasEvalModule<BE>,
    Module<BC>: BlasClientModule<BC>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    ScratchOwned<BC>: ScratchOwnedAlloc<BC> + ScratchOwnedBorrow<BC>,
{
    let mut client = Client::<BE, BC>::new(bp);
    let ctx = client.context(bp);
    let right = if dense { bp.batch() } else { 1 };
    let ct_a = client.encrypt(bp, bp.batch(), a_entry);
    let ct_b = client.encrypt(bp, right, b_entry(bp.d()));
    assert!(
        replicated::<Standard>(&ct_b) != dense,
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
    let mut res = client.result(bp);
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
    let check = verify::<BE>(
        "blas_matmul",
        bp,
        max_err,
        Some(max_base2k),
        atks.len(),
        log_budget::<BE>(&ct_a),
        &res,
    );
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
    BE: Backend<Ring = Standard> + HostStaged + 'static + BlasMatMulImpl + MaxBase2k,
    BC: Backend<Ring = Standard> + HostStaged + 'static,
    Module<BE>: BlasEvalModule<BE>,
    Module<BC>: BlasClientModule<BC>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    ScratchOwned<BC>: ScratchOwnedAlloc<BC> + ScratchOwnedBorrow<BC>,
{
    let mut client = Client::<BE, BC>::new(bp);
    let pt = client.public::<Right>(bp, algorithm);
    let ctx = client.context(bp);
    let ct_a = client.encrypt(bp, bp.batch(), a_entry);
    let max_base2k = blas_matmul_pt_right_max_base2k::<BE>(&ct_a, &pt, &ctx, FAILURE_BITS);
    assert!(
        max_base2k >= bp.base2k,
        "blas_matmul_pt_right: base2k exceeds the radix its lazy sums allow at 128 failure bits"
    );
    let order = client.module.cyclotomic_order();
    let atks = client.rotation_keys(bp, BlasPlan::new(bp.log_d).matmul_pt_right_galois_elements(order));
    let mut res = client.result(bp);
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
    let check = verify::<BE>(
        "blas_matmul_pt_right",
        bp,
        max_err,
        Some(max_base2k),
        atks.len(),
        log_budget::<BE>(&ct_a),
        &res,
    );
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
        let pt = client.public::<Left>(bp, algorithm);
        let ctx = client.context(bp);
        let ct_a = client.encrypt(bp, bp.batch(), a_entry);
        let max_base2k = blas_matmul_pt_left_max_base2k::<BE>(&pt, &ct_a, &ctx, FAILURE_BITS);
        assert!(
            max_base2k >= bp.base2k,
            "blas_matmul_pt_left: base2k exceeds the radix its lazy sums allow at 128 failure bits"
        );
        let order = client.module.cyclotomic_order();
        let atks = client.rotation_keys(bp, BlasPlan::new(bp.log_d).matmul_pt_left_galois_elements(order));
        let mut res = client.result(bp);
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
        let check = verify::<BE>(op, bp, max_err, Some(max_base2k), atks.len(), log_budget::<BE>(&ct_a), &res);
        let module = client.module;
        BlasPoint {
            call: Box::new(move || {
                module
                    .blas_matmul_pt_left(&mut res, &pt, &ct_a, &ctx, &atks, &mut scratch.borrow())
                    .unwrap();
                module.synchronize();
            }),
            check,
        }
        .reported::<BE>(op, bp)
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
