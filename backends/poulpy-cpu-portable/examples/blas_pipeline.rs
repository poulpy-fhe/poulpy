//! Encrypted matrix pipeline with poulpy-blas: plan the keys from the operations a pipeline calls, run it on a batch
//! of encrypted matrices, and check the result in the clear.
//!
//! ```bash
//! cargo run -p poulpy-cpu-portable --features enable-blas --example blas_pipeline
//! ```
//!
//! For a batch of 8 pairs of encrypted 8 x 8 matrices `(A_s, B_s)`, a public 8 x 4 matrix `P` shared by the batch and
//! a public 4 x 8 matrix `Q_s` per pair, computes `G = Σ_s Q_s (A_s B_s)^T P`:
//!
//! 1. client: secret key
//! 2. the plan's maps and the public matrices are encoded, then the output-layout queries give the layout of every
//!    intermediate grid from the input layouts alone, and the key-use queries on them list the `(Galois element,
//!    width)` pairs and the tensor key width the pipeline looks up; the client generates each key once, at the widest
//!    width it is looked up at
//! 3. client: encrypt `A` and `B` as grids of 4 x 4 blocks, one ciphertext per block holding that block of every
//!    matrix of the batch
//! 4. server: run `blas_matmul`, `blas_transpose`, `blas_matmul_pt_right`, `blas_matmul_pt_left` and `blas_sum`,
//!    printing the torus width each one consumes, and check the grids carry the planned layouts
//! 5. client: decrypt and compare against the cleartext result
//!
//! `libs/poulpy-blas/docs/blas-contracts.md` lists the key sets and the widths each operation looks its keys up at.

use std::collections::{BTreeMap, HashMap};

use anyhow::Result;
use poulpy_blas::{
    api::{BlasMatInterOps, BlasMatIntraOps, BlasMatMulOps},
    layouts::{
        BlasCiphertext, BlasCiphertextOwned, BlasContextPreparedFactory, BlasEncodingOps, BlasGridInfos, BlasLayout,
        BlasModuleAlloc, BlasPlan, BlasPtOperandPreparedFactory, Left, Right,
        grid::{join_blocks, split_into_blocks},
    },
    reference::{
        BlasKeyUses, blas_matmul_key_uses, blas_matmul_output_layout, blas_matmul_pt_left_key_uses,
        blas_matmul_pt_left_output_layout, blas_matmul_pt_right_key_uses, blas_matmul_pt_right_output_layout, blas_sum_key_uses,
        blas_sum_output_layout, blas_transpose_key_uses, blas_transpose_output_layout,
    },
};
use poulpy_ckks::prelude::*;
use poulpy_core::{
    GLWEAutomorphismKeyEncryptSk, GLWETensorKeyEncryptSk,
    api::{GLWEAutomorphismKeyPreparedFactory, GLWESecretPreparedFactory, GLWESecretSampling, GLWETensorKeyPreparedFactory},
    layouts::{
        Base2K, Degree, GGLWELayout, GLWEAutomorphismKeyLayout, GLWELayout, GLWETensorKeyLayout, LWEInfos, ModuleCoreAlloc, Rank,
        TorusPrecision,
        prepared::{GLWEAutomorphismKeyPrepared, GLWETensorKeyPrepared},
    },
};
use poulpy_cpu_portable::FFT64Portable;
use poulpy_hal::{
    api::{ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Backend, Module, ScratchArena, ScratchOwned},
    source::Source,
};

type BackendImpl = FFT64Portable;
type Arena = ScratchOwned<BackendImpl>;

const N: usize = 256;
/// Complex slot count.
const SLOTS: usize = N / 2;
/// Blocks of `D x D`, `D = 2^LOG_D`.
const LOG_D: usize = 2;
const D: usize = 1 << LOG_D;
/// Matrices per batch: every ciphertext holds one block of each.
const BATCH: usize = SLOTS / (D * D);
const BASE2K: usize = 19;
/// Ciphertext torus width, 12 limbs of 19 bits: the pipeline consumes 180, leaving the scale and 18 bits above it.
const K: usize = 228;
/// Scale of the encrypted entries.
const LOG_DELTA: usize = 30;
/// Scale `p` of the plan's maps and `p_B` of the public matrices: each map consumes its scale.
const LOG_P: usize = 30;
const LOG_P_B: usize = 30;
/// Secret-key Hamming weight.
const HW: usize = 192;

/// Layout of a fresh block: width `K`, entries at scale `2^LOG_DELTA`.
const CT: CKKSLayout = CKKSLayout {
    glwe_layout: GLWELayout {
        n: Degree(N as u32),
        base2k: Base2K(BASE2K as u32),
        k: TorusPrecision(K as u32),
        rank: Rank(1),
    },
    meta: GLWEPlaintextMeta {
        scale: Scale::Log(LOG_DELTA),
        slots: SlotsKind::Complex,
        log_sparsity: 0,
    },
};

/// A rotation key for inputs of width up to `width`: digits covering it, a guard of one digit and `log2 n` bits.
fn atk_at(width: usize) -> GLWEAutomorphismKeyLayout {
    GLWEAutomorphismKeyLayout {
        n: N.into(),
        base2k: BASE2K.into(),
        dnum: GGLWELayout::dnum_for_input(width.into(), BASE2K.into()),
        k_aux: (BASE2K + N.ilog2() as usize).into(),
        rank: Rank(1),
        k_digit: BASE2K.into(),
    }
}

/// A tensor key for terms of width up to `width`, as [`atk_at`].
fn tsk_at(width: usize) -> GLWETensorKeyLayout {
    GLWETensorKeyLayout {
        n: N.into(),
        base2k: BASE2K.into(),
        dnum: GGLWELayout::dnum_for_input(width.into(), BASE2K.into()),
        k_aux: (BASE2K + N.ilog2() as usize).into(),
        rank: Rank(1),
        k_digit: BASE2K.into(),
    }
}

/// Rotation keys, one layout for every operation at full width: a fresh block at `K`.
fn atk_layout() -> GLWEAutomorphismKeyLayout {
    atk_at(K)
}

fn tsk_layout() -> GLWETensorKeyLayout {
    tsk_at(K)
}

type Atk = GLWEAutomorphismKeyPrepared<<BackendImpl as Backend>::OwnedBuf, BackendImpl>;
type Tsk = GLWETensorKeyPrepared<<BackendImpl as Backend>::OwnedBuf, BackendImpl>;

/// Entry `(r, c)` of the encrypted `A_s` and `B_s`, in `[-1, 1]`, and of the public `P` and `Q_s`, in `[-0.5, 0.5]`.
fn a(s: usize, r: usize, c: usize) -> f64 {
    ((s + 3 * r + 5 * c) % 9) as f64 / 4.0 - 1.0
}
fn b(s: usize, r: usize, c: usize) -> f64 {
    ((2 * s + r + 7 * c) % 9) as f64 / 4.0 - 1.0
}
fn p(r: usize, c: usize) -> f64 {
    ((r * c + 1) % 5) as f64 / 4.0 - 0.5
}
fn q(s: usize, r: usize, c: usize) -> f64 {
    ((s + 2 * r + c) % 5) as f64 / 4.0 - 0.5
}

/// Prints the torus width an operation consumed from `input` into `res`.
fn report(op: &str, input: &impl BlasGridInfos, res: &impl BlasGridInfos) {
    let (from, to) = (input.block_layout().k().as_usize(), res.block_layout().k().as_usize());
    println!("{op:<20} k {from:>3} -> {to:>3}: {:>2} bits", from - to);
}

fn main() -> Result<()> {
    // ── 1. client: secret key ───────────────────────────────────────────────
    let module = Module::<BackendImpl>::new(N as u64);
    let mut source_xs = Source::new([0u8; 32]);
    let mut source_xa = Source::new([1u8; 32]);
    let mut source_xe = Source::new([2u8; 32]);

    let mut sk_raw = module.glwe_secret_alloc_from_infos(&CT);
    module.glwe_secret_fill_ternary_hw(&mut sk_raw, HW, &mut source_xs);
    let mut sk = module.glwe_secret_prepared_alloc_from_infos(&CT);
    module.glwe_secret_prepare(&mut sk, &sk_raw);

    // One arena for key generation, encryption and decryption.
    let mut scratch = Arena::alloc(module.ckks_all_ops_with_atk_tmp_bytes(&CT, &tsk_layout(), &atk_layout(), &CT));

    let gen_atk = |p: i64, width: usize, xe: &mut Source, xa: &mut Source, scratch: &mut ScratchArena<'_, BackendImpl>| -> Atk {
        let layout = atk_at(width);
        let mut atk = module.glwe_automorphism_key_alloc_from_infos(&layout);
        module.glwe_automorphism_key_encrypt_sk(&mut atk, p, &sk_raw, xe, xa, scratch);
        let mut prepared = module.glwe_automorphism_key_prepared_alloc_from_infos(&layout);
        module.glwe_automorphism_key_prepare(&mut prepared, &atk, scratch);
        prepared
    };
    // The tensor key, a separate store: only the product of two encrypted grids relinearizes.
    let gen_tsk = |width: usize, xe: &mut Source, xa: &mut Source, scratch: &mut ScratchArena<'_, BackendImpl>| -> Tsk {
        let layout = tsk_at(width);
        let mut tsk_raw = module.glwe_tensor_key_alloc_from_infos(&layout);
        module.glwe_tensor_key_encrypt_sk(&mut tsk_raw, &sk_raw, xe, xa, scratch);
        let mut tsk = module.alloc_tensor_key_prepared_from_infos(&layout);
        module.prepare_tensor_key(&mut tsk, &tsk_raw, scratch);
        tsk
    };

    // ── 2. server: maps and public matrices, which fix every layout ─────────
    // The plan's maps at scale p, and the public matrices at scale p_B, each with a few bits of headroom for entries
    // of magnitude at most 1: P on the right of a grid of 2 block columns, shared by the batch (count 1), and Q_s on the
    // left of a grid of 2 block rows, one per matrix (count BATCH).
    let maps = CoeffsMeta::from_delta_budget(LOG_P, 4);
    let public = CoeffsMeta::from_delta_budget(LOG_P_B, 4);
    let plan = BlasPlan::new(LOG_D);
    let mut ctx = module.blas_context_alloc(plan, BASE2K.into(), maps);
    let mut pt_p = module.blas_pt_operand_alloc::<Right>(plan, 2, 1, 1, BASE2K.into(), public);
    let mut pt_q = module.blas_pt_operand_alloc::<Left>(plan, 1, 2, BATCH, BASE2K.into(), public);
    let mut arena = Arena::alloc(
        module
            .blas_context_encode_tmp_bytes::<f64, _>(&ctx)
            .max(module.blas_pt_operand_encode_tmp_bytes::<f64, _>(&pt_q))
            .max(module.blas_context_prepare_tmp_bytes(&ctx))
            .max(module.blas_pt_operand_prepare_tmp_bytes(&pt_p))
            .max(module.blas_pt_operand_prepare_tmp_bytes(&pt_q)),
    );
    module.blas_context_encode_into::<f64>(&mut ctx, &mut arena.borrow())?;
    module.blas_pt_operand_encode_into(&mut pt_p, |_, r, c| p(r, c), &mut arena.borrow())?;
    module.blas_pt_operand_encode_into(&mut pt_q, q, &mut arena.borrow())?;
    let (mut ctx_prepared, mut pt_p_prepared, mut pt_q_prepared) = (
        module.blas_context_prepared_alloc_from_infos(&ctx),
        module.blas_pt_operand_prepared_alloc_from_infos(&pt_p),
        module.blas_pt_operand_prepared_alloc_from_infos(&pt_q),
    );
    module.blas_context_prepare(&mut ctx_prepared, &ctx, &mut arena.borrow());
    module.blas_pt_operand_prepare(&mut pt_p_prepared, &pt_p, &mut arena.borrow());
    module.blas_pt_operand_prepare(&mut pt_q_prepared, &pt_q, &mut arena.borrow());
    let (ctx, pt_p, pt_q) = (ctx_prepared, pt_p_prepared, pt_q_prepared);

    // ── 2b. client: every layout and key, from the input layouts alone ─────
    // A result grid is allocated at K; each operation's output layout follows from its inputs' layouts and `res`'s.
    let grid = |rows: usize, cols: usize| BlasLayout::new(CT, LOG_D, rows, cols);
    let (a_l, b_l) = (grid(2, 2), grid(2, 2));
    let c_l = blas_matmul_output_layout(&grid(2, 2), &a_l, &b_l, &ctx);
    let c_t_l = blas_transpose_output_layout(&grid(2, 2), &c_l, &ctx);
    let e_l = blas_matmul_pt_right_output_layout(&grid(2, 1), &c_t_l, &pt_p, &ctx);
    let f_l = blas_matmul_pt_left_output_layout(&grid(1, 1), &pt_q, &e_l, &ctx);
    let g_l = blas_sum_output_layout(&grid(1, 1), &f_l);
    let uses: Vec<BlasKeyUses> = vec![
        blas_matmul_key_uses(&module, &a_l, &b_l, &ctx),
        blas_transpose_key_uses(&module, &grid(2, 2), &c_l, &ctx),
        blas_matmul_pt_right_key_uses(&module, &grid(2, 1), &c_t_l, &pt_p, &ctx),
        blas_matmul_pt_left_key_uses(&module, &grid(1, 1), &pt_q, &e_l, &ctx),
        blas_sum_key_uses(&module, &grid(1, 1), &f_l),
    ];
    // A key serves every narrower look-up of its element, so each element needs one key, at its widest.
    let mut widest: BTreeMap<i64, usize> = BTreeMap::new();
    for &(p, k) in uses.iter().flat_map(|u| &u.rotations) {
        let w = widest.entry(p).or_default();
        *w = (*w).max(k.as_usize());
    }
    let tsk_width = uses
        .iter()
        .filter_map(|u| u.tensor)
        .max()
        .expect("blas_matmul looks up the tensor key")
        .as_usize();
    println!(
        "keys: {} rotation keys, each at the widest width it is looked up at, and a tensor key at width {tsk_width}",
        widest.len()
    );
    let atks: HashMap<i64, Atk> = widest
        .iter()
        .map(|(&p, &width)| (p, gen_atk(p, width, &mut source_xe, &mut source_xa, &mut scratch.borrow())))
        .collect();
    let tsk = gen_tsk(tsk_width, &mut source_xe, &mut source_xa, &mut scratch.borrow());

    // ── 3. client: encrypt A and B, one ciphertext per block ────────────────
    let zeros = vec![0.0f64; SLOTS];
    let mut encrypt =
        |rows: usize, cols: usize, entry: fn(usize, usize, usize) -> f64| -> Result<BlasCiphertextOwned<BackendImpl>> {
            let blocks = split_into_blocks(rows, cols, LOG_D, SLOTS, BATCH, entry)
                .iter()
                .map(|re| {
                    let mut pt = module.ckks_pt_vec_alloc(SLOTS, CT.base2k(), CT.k());
                    pt.set_meta(CT.meta);
                    module.ckks_encode_reim_into(&mut pt, re, &zeros, &mut scratch.borrow())?;
                    let mut ct = module.ckks_ciphertext_alloc_from_infos(&CT);
                    module.ckks_encrypt_sk(&mut ct, &pt, &sk, &mut source_xe, &mut source_xa, &mut scratch.borrow())?;
                    Ok(ct)
                })
                .collect::<Result<_>>()?;
            // Plaintexts of every slot, so the grid records the full batch.
            BlasCiphertext::from_blocks(blocks, LOG_D, rows, cols)
        };
    let ct_a = encrypt(2, 2, a)?;
    let ct_b = encrypt(2, 2, b)?;

    // ── 4. server: run the pipeline ─────────────────────────────────────────
    // Each result is a fresh grid at K, and each operation's scratch is its query's.
    let alloc = |rows: usize, cols: usize| module.blas_ciphertext_alloc(LOG_D, rows, cols, BASE2K.into(), K.into());
    let (atk, tsk_infos) = (atk_layout(), tsk_layout());

    // The pipeline, returning the grids `C`, `C^T`, `E`, `F` and `G`.
    let run = |tsk: &Tsk, atks: &HashMap<i64, Atk>| -> Result<[BlasCiphertextOwned<BackendImpl>; 5]> {
        // C = A B, 8 x 8: consumes 2p + log_delta.
        let mut c = alloc(2, 2);
        let bytes = module.blas_matmul_tmp_bytes(&c, &ct_a, &ct_b, &ctx, &tsk_infos, &atk);
        module.blas_matmul(&mut c, &ct_a, &ct_b, &ctx, tsk, atks, &mut Arena::alloc(bytes).borrow())?;
        report("blas_matmul", &ct_a, &c);

        // C^T: consumes p.
        let mut c_t = alloc(2, 2);
        let bytes = module.blas_transpose_tmp_bytes(&c_t, &c, &ctx, &atk);
        module.blas_transpose(&mut c_t, &c, &ctx, atks, &mut Arena::alloc(bytes).borrow())?;
        report("blas_transpose", &c, &c_t);

        // E = C^T P, 8 x 4: consumes p_B.
        let mut e = alloc(2, 1);
        let bytes = module.blas_matmul_pt_right_tmp_bytes(&e, &c_t, &pt_p, &ctx, &atk);
        module.blas_matmul_pt_right(&mut e, &c_t, &pt_p, &ctx, atks, &mut Arena::alloc(bytes).borrow())?;
        report("blas_matmul_pt_right", &c_t, &e);

        // F = Q E, 4 x 4: consumes p_B.
        let mut f = alloc(1, 1);
        let bytes = module.blas_matmul_pt_left_tmp_bytes(&f, &pt_q, &e, &ctx, &atk);
        module.blas_matmul_pt_left(&mut f, &pt_q, &e, &ctx, atks, &mut Arena::alloc(bytes).borrow())?;
        report("blas_matmul_pt_left", &e, &f);

        // G = Σ_s F_s in every matrix slot: consumes nothing. Its keys cover the batch they were planned for.
        assert_eq!(f.mat_slots(), Some(BATCH), "the sum's keys were planned for another batch");
        let mut g = alloc(1, 1);
        let bytes = module.blas_sum_tmp_bytes(&g, &f, &atk);
        module.blas_sum(&mut g, &f, atks, &mut Arena::alloc(bytes).borrow())?;
        report("blas_sum", &f, &g);
        Ok([c, c_t, e, f, g])
    };

    let [c, c_t, e, f, g] = run(&tsk, &atks)?;
    for (name, grid, planned) in [
        ("C", &c, &c_l),
        ("C^T", &c_t, &c_t_l),
        ("E", &e, &e_l),
        ("F", &f, &f_l),
        ("G", &g, &g_l),
    ] {
        assert_eq!(
            &BlasLayout::new(grid.block_layout(), LOG_D, grid.rows(), grid.cols()),
            planned,
            "planned layout of {name}"
        );
    }

    // ── 5. client: decrypt and compare with the cleartext ───────────────────
    let blocks = g
        .blocks()
        .iter()
        .map(|block| -> Result<Vec<f64>> {
            let mut pt = module.ckks_plaintext_alloc_from_infos(block);
            module.ckks_decrypt(&mut pt, block, &sk, &mut scratch.borrow())?;
            let (mut re, mut im) = (vec![0.0f64; SLOTS], vec![0.0f64; SLOTS]);
            module.ckks_decode_reim_into(&pt, &mut re, &mut im, &mut scratch.borrow())?;
            Ok(re)
        })
        .collect::<Result<Vec<_>>>()?;

    // G(r, c) = Σ_s Σ_l Σ_m Q_s(r, l) (A_s B_s)(m, l) P(m, c), for l and m over the 8 rows of C.
    let ab = |s: usize, m: usize, l: usize| (0..2 * D).map(|t| a(s, m, t) * b(s, t, l)).sum::<f64>();
    let want = |r: usize, c: usize| {
        let mut sum = 0.0;
        for s in 0..BATCH {
            for l in 0..2 * D {
                for m in 0..2 * D {
                    sum += q(s, r, l) * ab(s, m, l) * p(m, c);
                }
            }
        }
        sum
    };
    let mut max_err = 0.0f64;
    join_blocks(&blocks, 1, 1, LOG_D, BATCH, |_, r, c, have| {
        max_err = max_err.max((want(r, c) - have).abs())
    });
    println!("max |want - have| = {max_err:.3e} over the {BATCH} matrix slots");
    assert!(max_err < 1e-4, "homomorphic result diverges: {max_err:.3e}");
    println!("ok: G = Σ_s Q_s (A_s B_s)^T P verified");
    Ok(())
}
