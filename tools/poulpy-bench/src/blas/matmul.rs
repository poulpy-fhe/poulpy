//! Runners for the `poulpy-blas` operations on one `d x d` block per operand.
//!
//! A runner builds its whole point once, outside the timed loop: the module, the secret key, the context or the public
//! operand, only the rotation keys its operation looks up (plus the tensor key of a ciphertext by ciphertext product),
//! the encrypted inputs and the scratch its `*_tmp_bytes` reports. It checks that `base2k` fits the lazy sums of the
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
        CKKSModuleInfos,
    },
    layouts::CKKSModuleAlloc,
    test_suite::helpers::TestContextSharedModule,
};
use poulpy_core::{
    GLWEAutomorphismKeyEncryptSk, GLWETensorKeyEncryptSk,
    api::{GLWEAutomorphismKeyPreparedFactory, GLWESecretPreparedFactory, GLWESecretSampling, GLWETensorKeyPreparedFactory},
    layouts::{
        BackendGLWESecret, Base2K, Degree, Dnum, Dsize, GGLWELayout, GLWEAutomorphismKeyLayout, GLWELayout, GLWETensorKeyLayout,
        ModuleCoreAlloc, Rank, TorusPrecision, k_digit_of,
        prepared::{GLWEAutomorphismKeyPrepared, GLWESecretPrepared, GLWETensorKeyPrepared},
    },
};
use poulpy_hal::{
    api::{CnvPVecAlloc, ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Backend, CyclotomicOrder, MaxBase2k, Module, ScratchOwned, Standard},
    source::Source,
};

use crate::{backend_name, blas::params::BlasBenchParams, timed};

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

fn atk_layout(bp: &BlasBenchParams) -> GLWEAutomorphismKeyLayout {
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

fn tsk_layout(bp: &BlasBenchParams) -> GLWETensorKeyLayout {
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
fn a_entry(s: usize, r: usize, c: usize) -> f64 {
    entry(1, s, r, c)
}

/// The second ciphertext factor, scaled so that the products stay in `[-1, 1)`.
fn b_entry(d: usize) -> impl Fn(usize, usize, usize) -> f64 {
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

/// Prints the error of a runner's first call and asserts it within [`error_bound`].
fn verify<BE>(op: &str, bp: &BlasBenchParams, max_err: f64, max_base2k: Option<usize>, keys: usize) {
    let bound = error_bound(bp);
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

/// What a runner asks of a module beyond the operation itself: the key generation, encryption and decryption of
/// poulpy-ckks, the encoding of the maps, and their preparation.
pub trait BlasBenchModule<BE: Backend>:
    TestContextSharedModule<BE> + CKKSEncodingOps<BE, f64> + CKKSLinearTransformationOps<BE> + CnvPVecAlloc<BE>
{
}

impl<BE: Backend, M> BlasBenchModule<BE> for M where
    M: TestContextSharedModule<BE> + CKKSEncodingOps<BE, f64> + CKKSLinearTransformationOps<BE> + CnvPVecAlloc<BE>
{
}

/// The client side of a point: the module, the secret key, and one arena for key generation, encoding, encryption and
/// decryption.
struct Client<BE: Backend> {
    module: Module<BE>,
    sk_raw: BackendGLWESecret<BE>,
    sk: GLWESecretPrepared<BE::OwnedBuf, BE>,
    arena: ScratchOwned<BE>,
    xe: Source,
    xa: Source,
}

impl<BE> Client<BE>
where
    BE: Backend<Ring = Standard> + 'static,
    Module<BE>: BlasBenchModule<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    fn new(bp: &BlasBenchParams) -> Self {
        let module = Module::<BE>::new(bp.n as u64);
        let layout = block_layout(bp);
        let mut xs = Source::new([0u8; 32]);
        let mut sk_raw = module.glwe_secret_alloc_from_infos(&layout);
        module.glwe_secret_fill_ternary_hw(&mut sk_raw, HW, &mut xs);
        let mut sk = module.glwe_secret_prepared_alloc_from_infos(&layout);
        module.glwe_secret_prepare(&mut sk, &sk_raw);
        let bytes = module
            .ckks_all_ops_with_atk_tmp_bytes(&layout, &tsk_layout(bp), &atk_layout(bp), &layout)
            .max(<Module<BE> as CKKSEncodingHostOps<BE, f64>>::ckks_reim_tmp_bytes(
                &module,
                bp.slots(),
            ));
        Self {
            module,
            sk_raw,
            sk,
            arena: ScratchOwned::alloc(bytes),
            xe: Source::new([2u8; 32]),
            xa: Source::new([1u8; 32]),
        }
    }

    /// The prepared maps of the context at scale `p`.
    fn context(&self, bp: &BlasBenchParams) -> BlasContext<BE> {
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

    /// One prepared rotation key for each Galois element, and no other.
    fn rotation_keys(
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
                let mut atk = self.module.glwe_automorphism_key_alloc_from_infos(&layout);
                self.module.glwe_automorphism_key_encrypt_sk(
                    &mut atk,
                    p,
                    &self.sk_raw,
                    &mut self.xe,
                    &mut self.xa,
                    &mut self.arena.borrow(),
                );
                let mut prepared = self.module.glwe_automorphism_key_prepared_alloc_from_infos(&layout);
                self.module
                    .glwe_automorphism_key_prepare(&mut prepared, &atk, &mut self.arena.borrow());
                (p, prepared)
            })
            .collect()
    }

    fn tensor_key(&mut self, bp: &BlasBenchParams) -> GLWETensorKeyPrepared<BE::OwnedBuf, BE> {
        let layout = tsk_layout(bp);
        let mut tsk = self.module.glwe_tensor_key_alloc_from_infos(&layout);
        self.module
            .glwe_tensor_key_encrypt_sk(&mut tsk, &self.sk_raw, &mut self.xe, &mut self.xa, &mut self.arena.borrow());
        let mut prepared = self.module.alloc_tensor_key_prepared_from_infos(&layout);
        self.module.prepare_tensor_key(&mut prepared, &tsk, &mut self.arena.borrow());
        prepared
    }

    /// One block holding `matrices` matrices of `entry`, from a plaintext of `matrices d^2` slots, repeated up to the
    /// backend's minimum degree: the block's label proves their number, or the larger batch that degree holds.
    fn encrypt(
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
            .module
            .ckks_pt_vec_alloc(held, base2k(bp), TorusPrecision((bp.log_delta + HEADROOM) as u32));
        pt.set_meta(block_layout(bp).meta);
        self.module
            .ckks_encode_reim_into(&mut pt, &values, &vec![0.0; held], &mut self.arena.borrow())
            .unwrap();
        let mut ct = self.module.ckks_ciphertext_alloc(base2k(bp), TorusPrecision(bp.k as u32));
        self.module
            .ckks_encrypt_sk(&mut ct, &pt, &self.sk, &mut self.xe, &mut self.xa, &mut self.arena.borrow())
            .unwrap();
        BlasCiphertext::from_blocks(vec![ct], bp.log_d, 1, 1).unwrap()
    }

    /// A result grid of one block at the width of a fresh ciphertext.
    fn result(&self, bp: &BlasBenchParams) -> BlasCiphertextOwned<BE> {
        self.module
            .blas_ciphertext_alloc(bp.log_d, 1, 1, base2k(bp), TorusPrecision(bp.k as u32))
    }

    /// The largest distance of the decrypted block of `res` to `want(s, r, c)`, entry `(r, c)` of matrix `s`, over the
    /// matrices of the batch, and of its imaginary part to zero.
    fn max_error(
        &mut self,
        bp: &BlasBenchParams,
        res: &BlasCiphertextOwned<BE>,
        want: impl Fn(usize, usize, usize) -> f64,
    ) -> f64 {
        let block = &res.blocks()[0];
        let log_delta = block.log_delta();
        let mut pt = self
            .module
            .ckks_pt_vec_alloc(bp.slots(), base2k(bp), TorusPrecision((log_delta + DECODE_HEADROOM) as u32));
        pt.set_meta(GLWEPlaintextMeta {
            scale: Scale::Log(log_delta),
            slots: SlotsKind::Complex,
            log_sparsity: 0,
        });
        self.module
            .ckks_decrypt(&mut pt, block, &self.sk, &mut self.arena.borrow())
            .unwrap();
        let (mut re, mut im) = (vec![0.0; bp.slots()], vec![0.0; bp.slots()]);
        self.module
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

/// `blas_transpose` of a grid of `batch` matrices.
pub fn runner_blas_transpose<BE, M>(bencher: &mut Bencher<'_, M>, bp: &BlasBenchParams)
where
    BE: Backend<Ring = Standard> + 'static + BlasMatIntraImpl,
    Module<BE>: BlasBenchModule<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    M: Measurement,
{
    timed::<BE, M, _>(bencher, "blas_transpose", bp, || {
        let mut client = Client::<BE>::new(bp);
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
        verify::<BE>("blas_transpose", bp, max_err, None, atks.len());
        let module = client.module;
        Box::new(move || {
            module
                .blas_transpose(&mut res, &ct_a, &ctx, &atks, &mut scratch.borrow())
                .unwrap()
        })
    });
}

/// `blas_matmul` of `bp.batch()` matrices by as many distinct matrices on the right (`dense`), or by one matrix whose
/// compact encryption proves it, so that the right factor takes the replicated maps.
fn matmul<BE, M>(bencher: &mut Bencher<'_, M>, bp: &BlasBenchParams, op: &'static str, dense: bool)
where
    BE: Backend<Ring = Standard> + 'static + BlasMatMulImpl + MaxBase2k,
    Module<BE>: BlasBenchModule<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    M: Measurement,
{
    timed::<BE, M, _>(bencher, op, bp, || {
        let mut client = Client::<BE>::new(bp);
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
        verify::<BE>(op, bp, max_err, Some(max_base2k), atks.len());
        let module = client.module;
        Box::new(move || {
            module
                .blas_matmul(&mut res, &ct_a, &ct_b, &ctx, &tsk, &atks, &mut scratch.borrow())
                .unwrap()
        })
    });
}

/// `blas_matmul` with a dense right factor, distinct matrices per matrix slot: needs `bp.batch() > 1`.
pub fn runner_blas_matmul<BE, M>(bencher: &mut Bencher<'_, M>, bp: &BlasBenchParams)
where
    BE: Backend<Ring = Standard> + 'static + BlasMatMulImpl + MaxBase2k,
    Module<BE>: BlasBenchModule<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    M: Measurement,
{
    matmul::<BE, M>(bencher, bp, "blas_matmul", true);
}

/// `blas_matmul` with a replicated right factor, one matrix in every matrix slot.
pub fn runner_blas_matmul_replicated<BE, M>(bencher: &mut Bencher<'_, M>, bp: &BlasBenchParams)
where
    BE: Backend<Ring = Standard> + 'static + BlasMatMulImpl + MaxBase2k,
    Module<BE>: BlasBenchModule<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    M: Measurement,
{
    matmul::<BE, M>(bencher, bp, "blas_matmul_replicated", false);
}

/// `blas_matmul_pt_right` by the public matrix under `algorithm`.
fn pt_right<BE, M>(bencher: &mut Bencher<'_, M>, bp: &BlasBenchParams, op: &'static str, algorithm: BlasPtAlgorithm)
where
    BE: Backend<Ring = Standard> + 'static + BlasMatMulImpl + MaxBase2k,
    Module<BE>: BlasBenchModule<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    M: Measurement,
{
    timed::<BE, M, _>(bencher, op, bp, || {
        let mut client = Client::<BE>::new(bp);
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
        verify::<BE>(op, bp, max_err, Some(max_base2k), atks.len());
        let module = client.module;
        Box::new(move || {
            module
                .blas_matmul_pt_right(&mut res, &ct_a, &pt, &ctx, &atks, &mut scratch.borrow())
                .unwrap()
        })
    });
}

/// `blas_matmul_pt_right` under `Diagonal`.
pub fn runner_blas_matmul_pt_right_diagonal<BE, M>(bencher: &mut Bencher<'_, M>, bp: &BlasBenchParams)
where
    BE: Backend<Ring = Standard> + 'static + BlasMatMulImpl + MaxBase2k,
    Module<BE>: BlasBenchModule<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    M: Measurement,
{
    pt_right::<BE, M>(bencher, bp, "blas_matmul_pt_right_diagonal", BlasPtAlgorithm::Diagonal);
}

/// `blas_matmul_pt_right` under `Jkls`.
pub fn runner_blas_matmul_pt_right_jkls<BE, M>(bencher: &mut Bencher<'_, M>, bp: &BlasBenchParams)
where
    BE: Backend<Ring = Standard> + 'static + BlasMatMulImpl + MaxBase2k,
    Module<BE>: BlasBenchModule<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    M: Measurement,
{
    pt_right::<BE, M>(bencher, bp, "blas_matmul_pt_right_jkls", BlasPtAlgorithm::Jkls);
}

/// `blas_matmul_pt_left` by the public matrix under `algorithm`.
fn pt_left<BE, M>(bencher: &mut Bencher<'_, M>, bp: &BlasBenchParams, op: &'static str, algorithm: BlasPtAlgorithm)
where
    BE: Backend<Ring = Standard> + 'static + BlasMatMulImpl + MaxBase2k,
    Module<BE>: BlasBenchModule<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    M: Measurement,
{
    timed::<BE, M, _>(bencher, op, bp, || {
        let mut client = Client::<BE>::new(bp);
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
        verify::<BE>(op, bp, max_err, Some(max_base2k), atks.len());
        let module = client.module;
        Box::new(move || {
            module
                .blas_matmul_pt_left(&mut res, &pt, &ct_a, &ctx, &atks, &mut scratch.borrow())
                .unwrap()
        })
    });
}

/// `blas_matmul_pt_left` under `Diagonal`.
pub fn runner_blas_matmul_pt_left_diagonal<BE, M>(bencher: &mut Bencher<'_, M>, bp: &BlasBenchParams)
where
    BE: Backend<Ring = Standard> + 'static + BlasMatMulImpl + MaxBase2k,
    Module<BE>: BlasBenchModule<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    M: Measurement,
{
    pt_left::<BE, M>(bencher, bp, "blas_matmul_pt_left_diagonal", BlasPtAlgorithm::Diagonal);
}

/// `blas_matmul_pt_left` under `Jkls`.
pub fn runner_blas_matmul_pt_left_jkls<BE, M>(bencher: &mut Bencher<'_, M>, bp: &BlasBenchParams)
where
    BE: Backend<Ring = Standard> + 'static + BlasMatMulImpl + MaxBase2k,
    Module<BE>: BlasBenchModule<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    M: Measurement,
{
    pt_left::<BE, M>(bencher, bp, "blas_matmul_pt_left_jkls", BlasPtAlgorithm::Jkls);
}
