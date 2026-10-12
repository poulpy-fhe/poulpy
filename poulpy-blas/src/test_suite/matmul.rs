//! JKLS matrix products of whole grids.

use std::{collections::HashMap, sync::Mutex};

use poulpy_ckks::{
    CKKSInfos, GLWEPlaintextInfos, SetGLWEPlaintextInfos,
    api::{CKKSEncodingOps, CKKSLinearTransformationOps, LinearTransformationBabySteps, LinearTransformationPrepared},
    layouts::CKKSCiphertextOwned,
    test_suite::{
        CKKSTestParams,
        helpers::{
            TestContextBackend, TestContextModule, TestScalar, alloc_scratch, assert_decrypt_precision_at_log_delta,
            gen_sk_with_raw, gen_tsk,
        },
        parity::helpers::Snapshot,
        reference_encoder::ReferenceEncoder,
    },
};
use poulpy_core::layouts::{
    DiagonalArithmetic, GetAutomorphismKey, LWEInfos, TorusPrecision, prepared::GLWEAutomorphismKeyPreparedBackendRef,
};
use poulpy_hal::{
    api::{CnvPVecAlloc, NegacyclicFFT, NegacyclicFFTNew, ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{
        Backend, CyclotomicOrder, HostBytesBackend, HostDataMut, HostDataRef, Module, ScratchArena, ScratchOwned, Standard,
    },
};

use super::helpers::{
    Grid, Misshapen,
    Rejection::{InvalidOperand, MissingKey, MissingTensorKey},
    TestKeys, alloc_grid, assert_grid_decrypt, assert_guard, assert_rejected, assert_short_scratch, assert_short_scratch_panics,
    blas_context, context_and_keys, encrypt_grid, foreign_contexts, foreign_layouts, grid_snapshot, map_galois_elements,
    mapped_layout, product_layout, rejected_grids, relabel, rotation_keys,
};
use crate::{
    api::{BlasMatIntraOps, BlasMatMulOps},
    layouts::{BlasCiphertext, BlasCiphertextOwned, BlasGridInfos, BlasPlan, BlasPtOperand, BlasPtOperandUnprepared},
    oep::{BlasImpl, BlasMatMulImpl},
    reference::{Left, blas_matmul_term_layout, max_lazy_terms, validate_blas_matmul_prepared},
};

/// `dec(matmul(enc(A), enc(B)))` is `A B` for every matrix of the batch, for a `2 x 3` grid times a `3 x 2` grid, with
/// the JKLS product's metadata and canonical blocks, and `blas_matmul` equals `blas_matmul_prepare_left`,
/// `blas_matmul_prepare_right`, then `blas_matmul_prepared`, bit for bit, all with the rotation keys of the plan's
/// product set alone, which both preparations take.
pub fn test_blas_matmul<BE, F, E>(
    params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
    log_d: usize,
) where
    BE: TestContextBackend<Ring = Standard> + BlasMatMulImpl,
    Module<BE>: TestContextModule<BE> + CKKSEncodingOps<BE, F> + CKKSLinearTransformationOps<BE> + CnvPVecAlloc<BE>,
    F: TestScalar + DiagonalArithmetic,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
{
    let slots = params.n / 2;
    let encoder = ReferenceEncoder::<E>::new(slots).unwrap();
    let (sk_raw, sk) = gen_sk_with_raw(&params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, module);
    let ctx = blas_context::<BE, F>(&params, module, log_d, &mut scratch.borrow());
    // The keys of the products' set alone, which covers both preparations and the prepared product.
    let elements = ctx.plan().matmul_galois_elements(module.cyclotomic_order());
    let keys = TestKeys::<BE> {
        atks: rotation_keys(&params, module, &sk_raw, elements, &mut scratch.borrow()),
        tsk: Some(gen_tsk(&params, module, &sk_raw, &mut scratch.borrow())),
    };
    let (m, inner, n) = (2, 3, 2);
    let lanes = slots >> (2 * log_d);
    let (a, b) = (Grid::model(log_d, m, inner, lanes, 7), Grid::model(log_d, inner, n, lanes, 8));
    let ct_a = encrypt_grid::<BE, F, E>(
        &params,
        module,
        host_module,
        &encoder,
        &sk,
        params.k,
        0,
        &a,
        &mut scratch.borrow(),
    );
    let ct_b = encrypt_grid::<BE, F, E>(
        &params,
        module,
        host_module,
        &encoder,
        &sk,
        params.k,
        0,
        &b,
        &mut scratch.borrow(),
    );
    let (tsk_layout, atk_layout) = (params.tsk_layout(), params.atk_layout());
    let mut res = alloc_grid(&params, module, log_d, m, n, params.k);
    let bytes = module.blas_matmul_tmp_bytes(&res, &ct_a, &ct_b, &ctx, &tsk_layout, &atk_layout);
    module
        .blas_matmul(
            &mut res,
            &ct_a,
            &ct_b,
            &ctx,
            &keys,
            &keys,
            &mut ScratchOwned::<BE>::alloc(bytes).borrow(),
        )
        .unwrap();
    assert_grid_decrypt::<BE, F, E>(
        "matmul",
        &params,
        module,
        &encoder,
        &res,
        &sk,
        &a.matmul(&b),
        &product_layout(&ct_a, &ct_b, m, n, params.k, params.prec().log_delta()),
        &mut scratch.borrow(),
    );
    // One relinearization leaves every block normalized.
    assert!(
        res.blocks().iter().all(|block| block.is_canonical()),
        "matmul: a block is not canonical"
    );

    let bytes = module.blas_matmul_prepare_left_tmp_bytes(&ct_a, &ctx, &atk_layout);
    let left = module
        .blas_matmul_prepare_left(&ct_a, &ctx, &keys, &mut ScratchOwned::<BE>::alloc(bytes).borrow())
        .unwrap();
    let bytes = module.blas_matmul_prepare_right_tmp_bytes(&ct_b, &ctx, &atk_layout);
    let right = module
        .blas_matmul_prepare_right(&ct_b, &ctx, &keys, &mut ScratchOwned::<BE>::alloc(bytes).borrow())
        .unwrap();
    let mut prepared = alloc_grid(&params, module, log_d, m, n, params.k);
    let bytes = module.blas_matmul_prepared_tmp_bytes(&prepared, &left, &right, &tsk_layout);
    module
        .blas_matmul_prepared(
            &mut prepared,
            &left,
            &right,
            &keys,
            &mut ScratchOwned::<BE>::alloc(bytes).borrow(),
        )
        .unwrap();
    assert_eq!(grid_snapshot::<BE, _>(&prepared), grid_snapshot::<BE, _>(&res));
}

/// The public matrices of a `rows x cols` operand grid, entries in `[-0.5, 0.5]`.
pub(crate) fn public_entry(m: usize, r: usize, c: usize) -> f64 {
    ((m * 7 + r * 5 + c * 3) % 9) as f64 / 8.0 - 0.5
}

/// `keys`, counting the rotation-key lookups made through it, by width.
struct CountingKeys<'a, K> {
    keys: &'a K,
    lookups: Mutex<HashMap<usize, usize>>,
}

impl<'a, K> CountingKeys<'a, K> {
    fn new(keys: &'a K) -> Self {
        Self {
            keys,
            lookups: Mutex::default(),
        }
    }

    /// The lookups at width `k`.
    fn at(&self, k: usize) -> usize {
        self.lookups.lock().unwrap().get(&k).copied().unwrap_or(0)
    }
}

impl<BE: Backend, K: GetAutomorphismKey<BE>> GetAutomorphismKey<BE> for CountingKeys<'_, K> {
    fn lookup_automorphism_key(
        &self,
        p: i64,
        k: TorusPrecision,
    ) -> poulpy_core::Result<GLWEAutomorphismKeyPreparedBackendRef<'_, BE>> {
        *self.lookups.lock().unwrap().entry(k.as_usize()).or_default() += 1;
        self.keys.lookup_automorphism_key(p, k)
    }
}

/// Rotating giant steps of `lt` that hold diagonals: the key switches of one evaluation.
fn rotating_giant_steps<BE: Backend>(lt: &LinearTransformationPrepared<BE>) -> usize {
    lt.giant_steps
        .iter()
        .filter(|gs| gs.rot != 0 && !gs.diagonals.is_empty())
        .count()
}

/// The blocks a product by public matrices writes into `res`, recomputed with poulpy-ckks: block `(i, j)` is the summed
/// linear transformation of the `inner` terms `term(i, j, l)`, a source block and a map, in ascending `l`, each source
/// with its own baby-step cache, then takes its sources' label.
fn summed_terms<'t, BE>(
    module: &Module<BE>,
    mut res: BlasCiphertextOwned<BE>,
    inner: usize,
    term: impl Fn(usize, usize, usize) -> (&'t CKKSCiphertextOwned<BE>, &'t LinearTransformationPrepared<BE>),
    keys: &impl GetAutomorphismKey<BE>,
    scratch: &mut ScratchArena<'_, BE>,
) -> Vec<Snapshot>
where
    BE: TestContextBackend + 't,
    Module<BE>: CKKSLinearTransformationOps<BE> + CnvPVecAlloc<BE>,
{
    let cols = res.cols();
    for (t, dst) in res.blocks_mut().iter_mut().enumerate() {
        let terms: Vec<_> = (0..inner).map(|l| term(t / cols, t % cols, l)).collect();
        let caches: Vec<_> = terms
            .iter()
            .map(|&(src, lt)| {
                let mut cache = LinearTransformationBabySteps::alloc(module, lt.baby_steps(), src);
                module
                    .ckks_prepare_linear_transformation_baby_steps(&mut cache, src, keys, scratch)
                    .unwrap();
                cache
            })
            .collect();
        let summed: Vec<_> = terms
            .iter()
            .zip(&caches)
            .map(|(&(src, lt), cache)| (src, cache, lt))
            .collect();
        module
            .ckks_eval_linear_transformation_sum_into(dst, &summed, keys, scratch)
            .unwrap();
        dst.set_log_sparsity(terms[0].0.log_sparsity());
    }
    grid_snapshot::<BE, _>(&res)
}

/// `dec(matmul_pt(enc(A), B))` is `A B_(s mod count)` in matrix slot `s`, for a `2 x 3` grid times a `3 x 2` operand
/// with one, two, and one public matrix per matrix slot, of these counts those that divide the matrix slots, all with
/// the keys of the plan's set alone, at the metadata [`mapped_layout`] gives, its blocks canonical. Each block of `res`
/// is poulpy-ckks's summed linear transformation of its three terms in ascending inner index, bit for bit, also at
/// radix 26, where an FFT backend's transform-domain sums round, so that the order of the terms shows in the bits. A
/// counting key store sees the giant steps' keys looked up for one key switch per giant step and block of `res`.
pub fn test_blas_matmul_pt<BE, F, E>(
    params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
    log_d: usize,
) where
    BE: TestContextBackend<Ring = Standard> + BlasMatMulImpl,
    Module<BE>: TestContextModule<BE> + CKKSEncodingOps<BE, F> + CKKSLinearTransformationOps<BE> + CnvPVecAlloc<BE>,
    F: TestScalar + DiagonalArithmetic,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
{
    let slots = params.n / 2;
    let lanes = slots >> (2 * log_d);
    let encoder = ReferenceEncoder::<E>::new(slots).unwrap();
    let (m, inner, n) = (2, 3, 2);
    let encode = |params: &CKKSTestParams, count: usize, scratch: &mut ScratchArena<'_, BE>| {
        BlasPtOperandUnprepared::encode::<F>(
            module,
            log_d,
            inner,
            n,
            count,
            |m, r, c| F::from_f64(public_entry(m, r, c)).unwrap(),
            params.base2k.into(),
            params.prec().into(),
            scratch,
        )
        .unwrap()
        .prepare(module, scratch)
    };
    // The keys of the plan's set alone, whatever the operand's count.
    let elements = BlasPlan::new(log_d).matmul_pt_galois_elements(module.cyclotomic_order());
    let a = Grid::model(log_d, m, inner, lanes, 9);

    {
        let (sk_raw, sk) = gen_sk_with_raw(&params, module, host_module, [0u8; 32]);
        let mut scratch = alloc_scratch(&params, module);
        let mut counts: Vec<usize> = [1, 2, lanes].into_iter().filter(|c| lanes.is_multiple_of(*c)).collect();
        counts.dedup();
        let keys = TestKeys::<BE> {
            atks: rotation_keys(&params, module, &sk_raw, elements.iter().copied(), &mut scratch.borrow()),
            tsk: None,
        };
        let ct = encrypt_grid::<BE, F, E>(
            &params,
            module,
            host_module,
            &encoder,
            &sk,
            params.k,
            0,
            &a,
            &mut scratch.borrow(),
        );
        for count in counts {
            let b = encode(&params, count, &mut scratch.borrow());
            // `res` one limb wider than `a` for one public matrix.
            let res_k = if count == 1 { params.k + params.base2k } else { params.k };
            let label = format!("matmul_pt count={count} res_k={res_k}");
            let mut res = alloc_grid(&params, module, log_d, m, n, res_k);
            let bytes = module.blas_matmul_pt_tmp_bytes(&res, &ct, &b, &params.atk_layout());
            let mut op_scratch = ScratchOwned::<BE>::alloc(bytes);
            let counting = CountingKeys::new(&keys);
            module
                .blas_matmul_pt(&mut res, &ct, &b, &counting, &mut op_scratch.borrow())
                .unwrap();
            assert_grid_decrypt::<BE, F, E>(
                &label,
                &params,
                module,
                &encoder,
                &res,
                &sk,
                &a.matmul_public(n, count, public_entry),
                &mapped_layout(&ct, m, n, res_k, params.prec().log_delta()),
                &mut scratch.borrow(),
            );
            // The giant loop normalizes each block once.
            assert!(
                res.blocks().iter().all(|block| block.is_canonical()),
                "{label}: a block is not canonical"
            );
            // The giant steps' keys, at their width: the validator and the reference's scratch check look up those of
            // `b`'s `K n` maps, poulpy-ckks's summed transformation checks those of its `K` terms for each block of
            // `res`, then its giant loop looks each one up once per block, for its key switch. That is `m n g` key
            // switches for `g` rotating giant steps, where one transformation per term would take `K m n g`.
            let g = rotating_giant_steps(b.block(0, 0));
            let giant_k = res.block_layout().k().as_usize();
            assert_eq!(
                counting.at(giant_k),
                (2 * inner * n + inner * m * n + m * n) * g,
                "{label}: giant-step key lookups"
            );
            let want = summed_terms(
                module,
                alloc_grid(&params, module, log_d, m, n, res_k),
                inner,
                |i, j, l| (ct.block(i, l), b.block(l, j)),
                &keys,
                &mut op_scratch.borrow(),
            );
            assert_eq!(grid_snapshot::<BE, _>(&res), want, "{label}: summed terms");
        }
    }

    // At radix 26, where an FFT backend rounds the transform-domain sum of a giant step's products, so that summing the
    // terms in another order changes the bits.
    let params = CKKSTestParams {
        base2k: 26,
        k: 8 * 26,
        ..params
    };
    let (sk_raw, sk) = gen_sk_with_raw(&params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, module);
    let keys = TestKeys::<BE> {
        atks: rotation_keys(&params, module, &sk_raw, elements, &mut scratch.borrow()),
        tsk: None,
    };
    let ct = encrypt_grid::<BE, F, E>(
        &params,
        module,
        host_module,
        &encoder,
        &sk,
        params.k,
        0,
        &a,
        &mut scratch.borrow(),
    );
    let b = encode(&params, 1, &mut scratch.borrow());
    let mut res = alloc_grid(&params, module, log_d, m, n, params.k);
    let mut op_scratch = ScratchOwned::<BE>::alloc(module.blas_matmul_pt_tmp_bytes(&res, &ct, &b, &params.atk_layout()));
    module
        .blas_matmul_pt(&mut res, &ct, &b, &keys, &mut op_scratch.borrow())
        .unwrap();
    let want = summed_terms(
        module,
        alloc_grid(&params, module, log_d, m, n, params.k),
        inner,
        |i, j, l| (ct.block(i, l), b.block(l, j)),
        &keys,
        &mut op_scratch.borrow(),
    );
    assert_eq!(grid_snapshot::<BE, _>(&res), want, "matmul_pt at radix 26: summed terms");
}

/// An inner-dimension or `res` shape mismatch, more public matrices than a sparse input records, a copy relabelled past
/// `log2` of the ring degree, an operand encoded on a module of another degree, at another radix or at another
/// dimension, every grid [`rejected_grids`] builds, as `a` or as `res`, a `res` of another degree, radix or rank
/// ([`foreign_layouts`]), and a missing rotation key are rejected before `res` changes; a product one scratch byte short
/// of its query panics before `res` changes; an operand whose block count overflows `usize` does not encode.
pub fn test_blas_matmul_pt_guards<BE, F, E>(
    params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
    log_d: usize,
) where
    BE: TestContextBackend<Ring = Standard> + BlasMatMulImpl,
    Module<BE>: TestContextModule<BE> + CKKSEncodingOps<BE, F> + CKKSLinearTransformationOps<BE> + CnvPVecAlloc<BE>,
    F: TestScalar + DiagonalArithmetic,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
{
    let slots = params.n / 2;
    let lanes = slots >> (2 * log_d);
    let encoder = ReferenceEncoder::<E>::new(slots).unwrap();
    let (sk_raw, sk) = gen_sk_with_raw(&params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, module);
    let encode = |module: &Module<BE>, log_d: usize, count: usize, base2k: usize, scratch: &mut ScratchArena<'_, BE>| {
        BlasPtOperandUnprepared::encode::<F>(
            module,
            log_d,
            2,
            1,
            count,
            |m, r, c| F::from_f64(public_entry(m, r, c)).unwrap(),
            base2k.into(),
            params.prec().into(),
            scratch,
        )
        .unwrap()
        .prepare(module, scratch)
    };
    let b = encode(module, log_d, lanes, params.base2k, &mut scratch.borrow());
    // One public matrix, which divides the matrix slots of any module and of a grid read at any dimension.
    let one = encode(module, log_d, 1, params.base2k, &mut scratch.borrow());
    // At the previous dimension, whose smaller matrices fit wherever a block does, or the next one from 2 x 2 blocks.
    let other_log_d = if log_d > 1 { log_d - 1 } else { log_d + 1 };
    let other = encode(module, other_log_d, 1, params.base2k, &mut scratch.borrow());
    let double = Module::<BE>::new(2 * params.n as u64);
    let foreign = [
        ("degree", encode(&double, log_d, 1, params.base2k, &mut scratch.borrow())),
        ("base2k", encode(module, log_d, 1, params.base2k - 1, &mut scratch.borrow())),
    ];
    // Keys for the other operands too, so no missing key stops `blas_matmul_pt` from evaluating them.
    let blocks = [&b, &one, &other]
        .into_iter()
        .chain(foreign.iter().map(|(_, b)| b))
        .flat_map(|b| (0..b.rows()).flat_map(move |l| (0..b.cols()).map(move |j| b.block(l, j))));
    let elements = map_galois_elements(blocks, module.cyclotomic_order());
    let keys = TestKeys::<BE> {
        atks: rotation_keys(&params, module, &sk_raw, elements, &mut scratch.borrow()),
        tsk: None,
    };
    let encrypt = |rows: usize, cols: usize, seed: u64, scratch: &mut ScratchArena<'_, BE>| {
        encrypt_grid::<BE, F, E>(
            &params,
            module,
            host_module,
            &encoder,
            &sk,
            params.k,
            0,
            &Grid::model(log_d, rows, cols, lanes, seed),
            scratch,
        )
    };
    let ct = encrypt(1, 2, 11, &mut scratch.borrow());
    let narrow = encrypt(1, 3, 12, &mut scratch.borrow());
    let sparse = BlasCiphertext::from_blocks(relabel(&ct, |meta| meta.log_sparsity = 1).into_blocks(), log_d, 1, 2).unwrap();
    let sparse_64 = relabel(&ct, |meta| meta.log_sparsity = 64);
    let sparse_65 = relabel(&ct, |meta| meta.log_sparsity = 65);
    let mut op_scratch = ScratchOwned::<BE>::alloc(module.blas_matmul_pt_tmp_bytes(&ct, &ct, &b, &params.atk_layout()));

    // `narrow` has 3 block columns against 2 block rows of `b`; `res` must be 1 x 1; `sparse` records 4 matrices;
    // `sparse_64` and `sparse_65` are labelled past `log2` of the ring degree.
    for (label, rows, cols, a) in [
        ("inner", 1, 1, &narrow),
        ("shape", 1, 2, &ct),
        ("count", 1, 1, &sparse),
        ("sparsity 64", 1, 1, &sparse_64),
        ("sparsity 65", 1, 1, &sparse_65),
    ] {
        let mut res = alloc_grid(&params, module, log_d, rows, cols, params.k);
        assert_guard::<BE, _>(label, "blas_matmul_pt", InvalidOperand, &mut res, |res| {
            module.blas_matmul_pt(res, a, &b, &keys, &mut op_scratch.borrow())
        });
    }
    for (label, b) in &foreign {
        let mut res = alloc_grid(&params, module, log_d, 1, 1, params.k);
        assert_guard::<BE, _>(label, "blas_matmul_pt", InvalidOperand, &mut res, |res| {
            module.blas_matmul_pt(res, &ct, b, &keys, &mut op_scratch.borrow())
        });
    }
    // An operand encoded at another dimension, which `ct` and `res` are not read at.
    let mut res = alloc_grid(&params, module, log_d, 1, 1, params.k);
    assert_guard::<BE, _>("operand dimension", "blas_matmul_pt", InvalidOperand, &mut res, |res| {
        module.blas_matmul_pt(res, &ct, &other, &keys, &mut op_scratch.borrow())
    });

    // Each rejected grid as `a` or as `res` of a 2 x 2 grid times one public matrix.
    let square = encrypt(2, 2, 13, &mut scratch.borrow());
    let res_2x1 = alloc_grid(&params, module, log_d, 2, 1, params.k);
    for ((label, a), (_, mut bad)) in rejected_grids(module, &square)
        .into_iter()
        .zip(rejected_grids(module, &res_2x1))
    {
        let mut res = res_2x1.clone();
        assert_guard::<BE, _>(&format!("{label} a"), "blas_matmul_pt", InvalidOperand, &mut res, |res| {
            module.blas_matmul_pt(res, &a, &one, &keys, &mut op_scratch.borrow())
        });
        assert_guard::<BE, _>(&format!("{label} res"), "blas_matmul_pt", InvalidOperand, &mut bad, |res| {
            module.blas_matmul_pt(res, &square, &one, &keys, &mut op_scratch.borrow())
        });
    }
    for (label, mut bad) in foreign_layouts(module, &res_2x1) {
        assert_guard::<BE, _>(&format!("{label} res"), "blas_matmul_pt", InvalidOperand, &mut bad, |res| {
            module.blas_matmul_pt(res, &square, &one, &keys, &mut op_scratch.borrow())
        });
    }

    let mut res = alloc_grid(&params, module, log_d, 1, 1, params.k);
    let no_keys = TestKeys::<BE> {
        atks: HashMap::new(),
        tsk: None,
    };
    assert_guard::<BE, _>("no keys", "blas_matmul_pt", MissingKey, &mut res, |res| {
        module.blas_matmul_pt(res, &ct, &b, &no_keys, &mut op_scratch.borrow())
    });

    let bytes = module.blas_matmul_pt_tmp_bytes(&res, &ct, &b, &params.atk_layout());
    assert_short_scratch::<BE, _>("blas_matmul_pt", &mut res, bytes, |res, s| {
        module.blas_matmul_pt(res, &ct, &b, &keys, s)
    });

    // (2^62 + 1) x 4 blocks overflow a usize on 64 bits.
    let overflow = BlasPtOperandUnprepared::encode::<F>(
        module,
        log_d,
        usize::MAX / 4 + 2,
        4,
        lanes,
        |m, r, c| F::from_f64(public_entry(m, r, c)).unwrap(),
        params.base2k.into(),
        params.prec().into(),
        &mut scratch.borrow(),
    );
    assert!(overflow.is_err());
}

/// `dec(matmul_pt_left(B, enc(A)))` is `B_(s mod count) A` in matrix slot `s`, for a `2 x 3` operand times a `3 x 2`
/// grid with one, two and eight public matrices per matrix slot, of these counts those that divide the matrix slots,
/// all with the keys of the plan's left set alone, at the metadata [`mapped_layout`] gives, its blocks canonical. Each
/// block of `res` is poulpy-ckks's summed linear transformation of its three terms in ascending inner index, bit for
/// bit, also at radix 26, where an FFT backend's transform-domain sums round. The product spends `B`'s scale alone: the
/// same product through transposes, `transpose(blas_matmul_pt(transpose(A), B^T))`, gives the same values two context
/// scales narrower.
pub fn test_blas_matmul_pt_left<BE, F, E>(
    params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
    log_d: usize,
) where
    BE: TestContextBackend<Ring = Standard> + BlasImpl,
    Module<BE>: TestContextModule<BE> + CKKSEncodingOps<BE, F> + CKKSLinearTransformationOps<BE> + CnvPVecAlloc<BE>,
    F: TestScalar + DiagonalArithmetic,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
{
    let slots = params.n / 2;
    let lanes = slots >> (2 * log_d);
    let encoder = ReferenceEncoder::<E>::new(slots).unwrap();
    let (m, inner, n) = (2, 3, 2);
    let encode = |params: &CKKSTestParams, count: usize, scratch: &mut ScratchArena<'_, BE>| -> BlasPtOperand<BE, Left> {
        BlasPtOperandUnprepared::encode_left::<F>(
            module,
            log_d,
            m,
            inner,
            count,
            |m, r, c| F::from_f64(public_entry(m, r, c)).unwrap(),
            params.base2k.into(),
            params.prec().into(),
            scratch,
        )
        .unwrap()
        .prepare(module, scratch)
    };
    let (order, plan) = (module.cyclotomic_order(), BlasPlan::new(log_d));
    // The keys of the plan's left set alone, whatever the operand's count.
    let elements = plan.matmul_pt_left_galois_elements(order);
    let a = Grid::model(log_d, inner, n, lanes, 10);

    {
        let (sk_raw, sk) = gen_sk_with_raw(&params, module, host_module, [0u8; 32]);
        let mut scratch = alloc_scratch(&params, module);
        let keys = TestKeys::<BE> {
            atks: rotation_keys(&params, module, &sk_raw, elements.iter().copied(), &mut scratch.borrow()),
            tsk: None,
        };
        let ct = encrypt_grid::<BE, F, E>(
            &params,
            module,
            host_module,
            &encoder,
            &sk,
            params.k,
            0,
            &a,
            &mut scratch.borrow(),
        );
        // `B`'s scale, which the left product spends, and the context's, which each transpose spends.
        let (p_b, p) = (params.prec().log_delta(), params.prec().log_delta());
        for count in [1, 2, 8].into_iter().filter(|c| lanes.is_multiple_of(*c)) {
            let b = encode(&params, count, &mut scratch.borrow());
            let label = format!("matmul_pt_left count={count}");
            let mut res = alloc_grid(&params, module, log_d, m, n, params.k);
            let bytes = module.blas_matmul_pt_left_tmp_bytes(&res, &b, &ct, &params.atk_layout());
            let mut op_scratch = ScratchOwned::<BE>::alloc(bytes);
            module
                .blas_matmul_pt_left(&mut res, &b, &ct, &keys, &mut op_scratch.borrow())
                .unwrap();
            assert_grid_decrypt::<BE, F, E>(
                &label,
                &params,
                module,
                &encoder,
                &res,
                &sk,
                &a.public_matmul(m, count, public_entry),
                &mapped_layout(&ct, m, n, params.k, p_b),
                &mut scratch.borrow(),
            );
            assert!(
                res.blocks().iter().all(|block| block.is_canonical()),
                "{label}: a block is not canonical"
            );
            let want = summed_terms(
                module,
                alloc_grid(&params, module, log_d, m, n, params.k),
                inner,
                |i, j, l| (ct.block(l, j), b.block(i, l)),
                &keys,
                &mut op_scratch.borrow(),
            );
            assert_eq!(grid_snapshot::<BE, _>(&res), want, "{label}: summed terms");
        }

        // `transpose(blas_matmul_pt(transpose(A), B^T))`, with `B^T` of `inner x m` blocks on the right.
        let ctx = blas_context::<BE, F>(&params, module, log_d, &mut scratch.borrow());
        let route = plan
            .transpose_galois_elements(order)
            .into_iter()
            .chain(plan.matmul_pt_galois_elements(order));
        let keys = TestKeys::<BE> {
            atks: rotation_keys(&params, module, &sk_raw, route, &mut scratch.borrow()),
            tsk: None,
        };
        let bt = BlasPtOperandUnprepared::encode::<F>(
            module,
            log_d,
            inner,
            m,
            1,
            |s, r, c| F::from_f64(public_entry(s, c, r)).unwrap(),
            params.base2k.into(),
            params.prec().into(),
            &mut scratch.borrow(),
        )
        .unwrap()
        .prepare(module, &mut scratch.borrow());
        let atk = params.atk_layout();
        let (mut at, mut atb, mut res) = (
            alloc_grid(&params, module, log_d, n, inner, params.k),
            alloc_grid(&params, module, log_d, n, m, params.k),
            alloc_grid(&params, module, log_d, m, n, params.k),
        );
        let bytes = module
            .blas_transpose_tmp_bytes(&at, &ct, &ctx, &atk)
            .max(module.blas_matmul_pt_tmp_bytes(&atb, &at, &bt, &atk))
            .max(module.blas_transpose_tmp_bytes(&res, &atb, &ctx, &atk));
        let mut op_scratch = ScratchOwned::<BE>::alloc(bytes);
        module
            .blas_transpose(&mut at, &ct, &ctx, &keys, &mut op_scratch.borrow())
            .unwrap();
        module
            .blas_matmul_pt(&mut atb, &at, &bt, &keys, &mut op_scratch.borrow())
            .unwrap();
        module
            .blas_transpose(&mut res, &atb, &ctx, &keys, &mut op_scratch.borrow())
            .unwrap();
        let layout = mapped_layout(
            &mapped_layout(&mapped_layout(&ct, n, inner, params.k, p), n, m, params.k, p_b),
            m,
            n,
            params.k,
            p,
        );
        assert_eq!(
            (
                layout.block.k().as_usize(),
                mapped_layout(&ct, m, n, params.k, p_b).block.k().as_usize()
            ),
            (params.k - 2 * p - p_b, params.k - p_b),
            "the transposes spend two context scales"
        );
        assert_grid_decrypt::<BE, F, E>(
            "matmul_pt_left through transposes",
            &params,
            module,
            &encoder,
            &res,
            &sk,
            &a.public_matmul(m, 1, public_entry),
            &layout,
            &mut scratch.borrow(),
        );
    }

    // At radix 26, where an FFT backend rounds the transform-domain sum of a giant step's products, so that summing the
    // terms in another order changes the bits.
    let params = CKKSTestParams {
        base2k: 26,
        k: 8 * 26,
        ..params
    };
    let (sk_raw, sk) = gen_sk_with_raw(&params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, module);
    let keys = TestKeys::<BE> {
        atks: rotation_keys(&params, module, &sk_raw, elements, &mut scratch.borrow()),
        tsk: None,
    };
    let ct = encrypt_grid::<BE, F, E>(
        &params,
        module,
        host_module,
        &encoder,
        &sk,
        params.k,
        0,
        &a,
        &mut scratch.borrow(),
    );
    let b = encode(&params, 1, &mut scratch.borrow());
    let mut res = alloc_grid(&params, module, log_d, m, n, params.k);
    let bytes = module.blas_matmul_pt_left_tmp_bytes(&res, &b, &ct, &params.atk_layout());
    let mut op_scratch = ScratchOwned::<BE>::alloc(bytes);
    module
        .blas_matmul_pt_left(&mut res, &b, &ct, &keys, &mut op_scratch.borrow())
        .unwrap();
    let want = summed_terms(
        module,
        alloc_grid(&params, module, log_d, m, n, params.k),
        inner,
        |i, j, l| (ct.block(l, j), b.block(i, l)),
        &keys,
        &mut op_scratch.borrow(),
    );
    assert_eq!(grid_snapshot::<BE, _>(&res), want, "matmul_pt_left at radix 26: summed terms");
}

/// The guards of [`test_blas_matmul_pt_guards`] for `B A`, through `blas_matmul_pt_left`: an inner-dimension or `res`
/// shape mismatch, more public matrices than a sparse input records, a copy relabelled past `log2` of the ring degree,
/// an operand encoded on a module of another degree, at another radix or at another dimension, every grid
/// [`rejected_grids`] builds, as `a` or as `res`, a `res` of another degree, radix or rank ([`foreign_layouts`]), and a
/// missing rotation key are rejected before `res` changes; a product one scratch byte short of its query panics before
/// `res` changes; an operand whose block count overflows `usize` does not encode.
pub fn test_blas_matmul_pt_left_guards<BE, F, E>(
    params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
    log_d: usize,
) where
    BE: TestContextBackend<Ring = Standard> + BlasMatMulImpl,
    Module<BE>: TestContextModule<BE> + CKKSEncodingOps<BE, F> + CKKSLinearTransformationOps<BE> + CnvPVecAlloc<BE>,
    F: TestScalar + DiagonalArithmetic,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
{
    let op = "blas_matmul_pt_left";
    let slots = params.n / 2;
    let lanes = slots >> (2 * log_d);
    let encoder = ReferenceEncoder::<E>::new(slots).unwrap();
    let (sk_raw, sk) = gen_sk_with_raw(&params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, module);
    // `1 x 2` blocks of public matrices: their products take `a` of two block rows.
    let encode = |module: &Module<BE>, log_d: usize, count: usize, base2k: usize, scratch: &mut ScratchArena<'_, BE>| {
        BlasPtOperandUnprepared::encode_left::<F>(
            module,
            log_d,
            1,
            2,
            count,
            |m, r, c| F::from_f64(public_entry(m, r, c)).unwrap(),
            base2k.into(),
            params.prec().into(),
            scratch,
        )
        .unwrap()
        .prepare(module, scratch)
    };
    let b = encode(module, log_d, lanes, params.base2k, &mut scratch.borrow());
    // One public matrix, which divides the matrix slots of any module and of a grid read at any dimension.
    let one = encode(module, log_d, 1, params.base2k, &mut scratch.borrow());
    // At the previous dimension, whose smaller matrices fit wherever a block does, or the next one from 2 x 2 blocks.
    let other_log_d = if log_d > 1 { log_d - 1 } else { log_d + 1 };
    let other = encode(module, other_log_d, 1, params.base2k, &mut scratch.borrow());
    let double = Module::<BE>::new(2 * params.n as u64);
    let foreign = [
        ("degree", encode(&double, log_d, 1, params.base2k, &mut scratch.borrow())),
        ("base2k", encode(module, log_d, 1, params.base2k - 1, &mut scratch.borrow())),
    ];
    // Keys for the other operands too, so no missing key stops `blas_matmul_pt_left` from evaluating them.
    let blocks = [&b, &one, &other]
        .into_iter()
        .chain(foreign.iter().map(|(_, b)| b))
        .flat_map(|b| (0..b.rows()).flat_map(move |i| (0..b.cols()).map(move |l| b.block(i, l))));
    let elements = map_galois_elements(blocks, module.cyclotomic_order());
    let keys = TestKeys::<BE> {
        atks: rotation_keys(&params, module, &sk_raw, elements, &mut scratch.borrow()),
        tsk: None,
    };
    let encrypt = |rows: usize, cols: usize, seed: u64, scratch: &mut ScratchArena<'_, BE>| {
        encrypt_grid::<BE, F, E>(
            &params,
            module,
            host_module,
            &encoder,
            &sk,
            params.k,
            0,
            &Grid::model(log_d, rows, cols, lanes, seed),
            scratch,
        )
    };
    let ct = encrypt(2, 1, 11, &mut scratch.borrow());
    let narrow = encrypt(3, 1, 12, &mut scratch.borrow());
    let sparse = BlasCiphertext::from_blocks(relabel(&ct, |meta| meta.log_sparsity = 1).into_blocks(), log_d, 2, 1).unwrap();
    let sparse_64 = relabel(&ct, |meta| meta.log_sparsity = 64);
    let sparse_65 = relabel(&ct, |meta| meta.log_sparsity = 65);
    let square = encrypt(2, 2, 13, &mut scratch.borrow());
    let res_1x2 = alloc_grid(&params, module, log_d, 1, 2, params.k);
    let mut op_scratch =
        ScratchOwned::<BE>::alloc(module.blas_matmul_pt_left_tmp_bytes(&res_1x2, &b, &square, &params.atk_layout()));

    // `narrow` has 3 block rows against 2 block columns of `b`; `res` must be 1 x 1; `sparse` records half the matrices
    // `b` holds; `sparse_64` and `sparse_65` are labelled past `log2` of the ring degree.
    for (label, rows, cols, a) in [
        ("inner", 1, 1, &narrow),
        ("shape", 1, 2, &ct),
        ("count", 1, 1, &sparse),
        ("sparsity 64", 1, 1, &sparse_64),
        ("sparsity 65", 1, 1, &sparse_65),
    ] {
        let mut res = alloc_grid(&params, module, log_d, rows, cols, params.k);
        assert_guard::<BE, _>(label, op, InvalidOperand, &mut res, |res| {
            module.blas_matmul_pt_left(res, &b, a, &keys, &mut op_scratch.borrow())
        });
    }
    for (label, b) in &foreign {
        let mut res = alloc_grid(&params, module, log_d, 1, 1, params.k);
        assert_guard::<BE, _>(label, op, InvalidOperand, &mut res, |res| {
            module.blas_matmul_pt_left(res, b, &ct, &keys, &mut op_scratch.borrow())
        });
    }
    // An operand encoded at another dimension, which `ct` and `res` are not read at.
    let mut res = alloc_grid(&params, module, log_d, 1, 1, params.k);
    assert_guard::<BE, _>("operand dimension", op, InvalidOperand, &mut res, |res| {
        module.blas_matmul_pt_left(res, &other, &ct, &keys, &mut op_scratch.borrow())
    });

    // Each rejected grid as `a` or as `res` of one public matrix times a 2 x 2 grid.
    for ((label, a), (_, mut bad)) in rejected_grids(module, &square)
        .into_iter()
        .zip(rejected_grids(module, &res_1x2))
    {
        let mut res = res_1x2.clone();
        assert_guard::<BE, _>(&format!("{label} a"), op, InvalidOperand, &mut res, |res| {
            module.blas_matmul_pt_left(res, &one, &a, &keys, &mut op_scratch.borrow())
        });
        assert_guard::<BE, _>(&format!("{label} res"), op, InvalidOperand, &mut bad, |res| {
            module.blas_matmul_pt_left(res, &one, &square, &keys, &mut op_scratch.borrow())
        });
    }
    for (label, mut bad) in foreign_layouts(module, &res_1x2) {
        assert_guard::<BE, _>(&format!("{label} res"), op, InvalidOperand, &mut bad, |res| {
            module.blas_matmul_pt_left(res, &one, &square, &keys, &mut op_scratch.borrow())
        });
    }

    let mut res = alloc_grid(&params, module, log_d, 1, 1, params.k);
    let no_keys = TestKeys::<BE> {
        atks: HashMap::new(),
        tsk: None,
    };
    assert_guard::<BE, _>("no keys", op, MissingKey, &mut res, |res| {
        module.blas_matmul_pt_left(res, &b, &ct, &no_keys, &mut op_scratch.borrow())
    });

    let bytes = module.blas_matmul_pt_left_tmp_bytes(&res, &b, &ct, &params.atk_layout());
    assert_short_scratch::<BE, _>(op, &mut res, bytes, |res, s| {
        module.blas_matmul_pt_left(res, &b, &ct, &keys, s)
    });

    // (2^62 + 1) x 4 blocks overflow a usize on 64 bits.
    let overflow = BlasPtOperandUnprepared::encode_left::<F>(
        module,
        log_d,
        usize::MAX / 4 + 2,
        4,
        lanes,
        |m, r, c| F::from_f64(public_entry(m, r, c)).unwrap(),
        params.base2k.into(),
        params.prec().into(),
        &mut scratch.borrow(),
    );
    assert!(overflow.is_err());
}

/// Rejected before `res` changes, by the product and, of these cases, by those that meet them, the preparations and the
/// prepared product: every grid [`rejected_grids`] builds, in place of a factor or of `res`, or as the input of a
/// preparation; a `res` shape mismatch; a `res`, one factor or one prepared operand of another degree, radix or rank,
/// and an input of a preparation of another degree or radix ([`foreign_layouts`]); a copy relabelled past `log2` of the
/// ring degree; a context encoded on a module of another degree or at another radix; a factor at a radix other than the
/// context's; an inner-dimension mismatch; a radix above 62, whose digits cannot hold the JKLS kernel's two lazy terms;
/// and a missing rotation or tensor key. The prepared product's validator also rejects operand infos without a batch
/// record, at another dimension, without a block, or whose label does not allow their record. The preparations take no
/// tensor key and the prepared product no rotation key. Each of these operations one scratch byte short of its query
/// panics before `res` changes.
pub fn test_blas_matmul_guards<BE, F, E>(
    params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
    log_d: usize,
) where
    BE: TestContextBackend<Ring = Standard> + BlasMatMulImpl,
    Module<BE>: TestContextModule<BE> + CKKSEncodingOps<BE, F> + CKKSLinearTransformationOps<BE> + CnvPVecAlloc<BE>,
    F: TestScalar + DiagonalArithmetic,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
{
    let slots = params.n / 2;
    let lanes = slots >> (2 * log_d);
    let encoder = ReferenceEncoder::<E>::new(slots).unwrap();
    let (sk_raw, sk) = gen_sk_with_raw(&params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, module);
    let (ctx, mut atks) = context_and_keys::<BE, F>(&params, module, log_d, &sk_raw, &mut scratch.borrow());
    let foreign = foreign_contexts::<BE, F>(&params, module, log_d, &sk_raw, &mut atks, &mut scratch.borrow());
    let keys = TestKeys::<BE> {
        atks,
        tsk: Some(gen_tsk(&params, module, &sk_raw, &mut scratch.borrow())),
    };
    let ct = encrypt_grid::<BE, F, E>(
        &params,
        module,
        host_module,
        &encoder,
        &sk,
        params.k,
        0,
        &Grid::model(log_d, 1, 2, lanes, 10),
        &mut scratch.borrow(),
    );
    let column = BlasCiphertext::from_blocks(ct.clone().into_blocks(), log_d, 2, 1).unwrap();
    let alloc = |rows: usize, cols: usize| alloc_grid(&params, module, log_d, rows, cols, params.k);
    let (tsk_layout, atk_layout) = (params.tsk_layout(), params.atk_layout());
    // `column ct`, the widest product below, which covers both preparations.
    let bytes = module.blas_matmul_tmp_bytes(&alloc(2, 2), &column, &ct, &ctx, &tsk_layout, &atk_layout);
    let mut op_scratch = ScratchOwned::<BE>::alloc(bytes);

    // `ct` is 1 x 2 and `column` 2 x 1: `ct ct` has a mismatched inner dimension, `ct column` is 1 x 1, not 1 x 2.
    for (label, mut res, b) in [("inner", alloc(1, 2), &ct), ("shape", alloc(1, 2), &column)] {
        assert_guard::<BE, _>(label, "blas_matmul", InvalidOperand, &mut res, |res| {
            module.blas_matmul(res, &ct, b, &ctx, &keys, &keys, &mut op_scratch.borrow())
        });
    }
    let left = module
        .blas_matmul_prepare_left(&ct, &ctx, &keys, &mut op_scratch.borrow())
        .unwrap();
    let left_column = module
        .blas_matmul_prepare_left(&column, &ctx, &keys, &mut op_scratch.borrow())
        .unwrap();
    let right = module
        .blas_matmul_prepare_right(&column, &ctx, &keys, &mut op_scratch.borrow())
        .unwrap();
    let row_right = module
        .blas_matmul_prepare_right(&ct, &ctx, &keys, &mut op_scratch.borrow())
        .unwrap();
    let mut wide = alloc(1, 2);
    for (label, right) in [("inner", &row_right), ("shape", &right)] {
        assert_guard::<BE, _>(label, "blas_matmul_prepared", InvalidOperand, &mut wide, |res| {
            module.blas_matmul_prepared(res, &left, right, &keys, &mut op_scratch.borrow())
        });
    }

    // Each rejected grid as a factor of `column ct`, a 2 x 2 product, and as the input of either preparation.
    for (label, bad) in rejected_grids(module, &column) {
        let label = &format!("{label} column");
        let mut res = alloc(2, 2);
        assert_guard::<BE, _>(label, "blas_matmul", InvalidOperand, &mut res, |res| {
            module.blas_matmul(res, &bad, &ct, &ctx, &keys, &keys, &mut op_scratch.borrow())
        });
        let left = module.blas_matmul_prepare_left(&bad, &ctx, &keys, &mut op_scratch.borrow());
        assert_rejected(label, "blas_matmul_prepare_left", InvalidOperand, left);
        let right = module.blas_matmul_prepare_right(&bad, &ctx, &keys, &mut op_scratch.borrow());
        assert_rejected(label, "blas_matmul_prepare_right", InvalidOperand, right);
    }
    for (label, bad) in rejected_grids(module, &ct) {
        let label = &format!("{label} row");
        let mut res = alloc(2, 2);
        assert_guard::<BE, _>(label, "blas_matmul", InvalidOperand, &mut res, |res| {
            module.blas_matmul(res, &column, &bad, &ctx, &keys, &keys, &mut op_scratch.borrow())
        });
    }
    // Each rejected grid, and each grid of another degree, radix or rank, as `res` of `column ct` and of its prepared
    // form.
    let bad_res = rejected_grids(module, &alloc(2, 2))
        .into_iter()
        .chain(foreign_layouts(module, &alloc(2, 2)));
    for (label, mut bad) in bad_res {
        let label = &format!("{label} res");
        assert_guard::<BE, _>(label, "blas_matmul", InvalidOperand, &mut bad, |res| {
            module.blas_matmul(res, &column, &ct, &ctx, &keys, &keys, &mut op_scratch.borrow())
        });
        assert_guard::<BE, _>(label, "blas_matmul_prepared", InvalidOperand, &mut bad, |res| {
            module.blas_matmul_prepared(res, &left_column, &row_right, &keys, &mut op_scratch.borrow())
        });
    }
    // Each grid of another degree, radix or rank as one factor of `column ct` beside a valid one, and, read through its
    // infos, as one prepared operand: each factor is compared with `res` on its own.
    for ((label, bad_column), (_, bad_row)) in foreign_layouts(module, &column).into_iter().zip(foreign_layouts(module, &ct)) {
        for (side, a, b) in [("a", &bad_column, &ct), ("b", &column, &bad_row)] {
            let label = &format!("{label} {side}");
            let mut res = alloc(2, 2);
            assert_guard::<BE, _>(label, "blas_matmul", InvalidOperand, &mut res, |res| {
                module.blas_matmul(res, a, b, &ctx, &keys, &keys, &mut op_scratch.borrow())
            });
        }
        let res = alloc(2, 2);
        let infos = validate_blas_matmul_prepared(module, &res, &bad_column, &row_right, &keys);
        assert_rejected(&format!("{label} a infos"), "blas_matmul_prepared", InvalidOperand, infos);
        let infos = validate_blas_matmul_prepared(module, &res, &left_column, &bad_row, &keys);
        assert_rejected(&format!("{label} b infos"), "blas_matmul_prepared", InvalidOperand, infos);
    }

    // Copies relabelled past `log2` of the ring degree, which keep recording the dense batch.
    for log_sparsity in [64, 65] {
        let label = &format!("sparsity {log_sparsity}");
        let sparse_row = relabel(&ct, |meta| meta.log_sparsity = log_sparsity);
        let sparse_column = relabel(&column, |meta| meta.log_sparsity = log_sparsity);
        let mut res = alloc(1, 1);
        assert_guard::<BE, _>(label, "blas_matmul", InvalidOperand, &mut res, |res| {
            module.blas_matmul(res, &sparse_row, &column, &ctx, &keys, &keys, &mut op_scratch.borrow())
        });
        let left = module.blas_matmul_prepare_left(&sparse_row, &ctx, &keys, &mut op_scratch.borrow());
        assert_rejected(label, "blas_matmul_prepare_left", InvalidOperand, left);
        let right = module.blas_matmul_prepare_right(&sparse_column, &ctx, &keys, &mut op_scratch.borrow());
        assert_rejected(label, "blas_matmul_prepare_right", InvalidOperand, right);
    }

    for (label, ctx) in &foreign {
        let mut res = alloc(1, 1);
        assert_guard::<BE, _>(label, "blas_matmul", InvalidOperand, &mut res, |res| {
            module.blas_matmul(res, &ct, &column, ctx, &keys, &keys, &mut op_scratch.borrow())
        });
        let left = module.blas_matmul_prepare_left(&ct, ctx, &keys, &mut op_scratch.borrow());
        assert_rejected(label, "blas_matmul_prepare_left", InvalidOperand, left);
        let right = module.blas_matmul_prepare_right(&column, ctx, &keys, &mut op_scratch.borrow());
        assert_rejected(label, "blas_matmul_prepare_right", InvalidOperand, right);
    }

    // One factor at a radix other than the context's, the other at the context's.
    let below = CKKSTestParams {
        base2k: params.base2k - 1,
        ..params
    };
    let radix = |rows: usize, cols: usize| alloc_grid(&below, module, log_d, rows, cols, params.k);
    for (label, a, b) in [("radix a", radix(1, 2), column.clone()), ("radix b", ct.clone(), radix(2, 1))] {
        let mut res = alloc(1, 1);
        assert_guard::<BE, _>(label, "blas_matmul", InvalidOperand, &mut res, |res| {
            module.blas_matmul(res, &a, &b, &ctx, &keys, &keys, &mut op_scratch.borrow())
        });
    }
    let left_radix = module.blas_matmul_prepare_left(&radix(1, 2), &ctx, &keys, &mut op_scratch.borrow());
    assert_rejected("radix", "blas_matmul_prepare_left", InvalidOperand, left_radix);
    let right_radix = module.blas_matmul_prepare_right(&radix(2, 1), &ctx, &keys, &mut op_scratch.borrow());
    assert_rejected("radix", "blas_matmul_prepare_right", InvalidOperand, right_radix);
    // A grid of another degree or radix, alone: no `res` to compare it with.
    for (label, bad) in foreign_layouts(module, &ct).into_iter().filter(|(label, _)| *label != "rank") {
        let left = module.blas_matmul_prepare_left(&bad, &ctx, &keys, &mut op_scratch.borrow());
        assert_rejected(label, "blas_matmul_prepare_left", InvalidOperand, left);
        let right = module.blas_matmul_prepare_right(&bad, &ctx, &keys, &mut op_scratch.borrow());
        assert_rejected(label, "blas_matmul_prepare_right", InvalidOperand, right);
    }

    // `res` at radix 63: its digits hold one lazy term, short of the two the JKLS kernel sums between normalizations. Its
    // radix is not the factors' either, so the reason tells the two checks apart.
    let radix_63 = CKKSTestParams { base2k: 63, ..params };
    let mut res = alloc_grid(&radix_63, module, log_d, 1, 1, params.k);
    assert_guard::<BE, _>("lazy sums", "blas_matmul", InvalidOperand, &mut res, |res| {
        module.blas_matmul(res, &ct, &column, &ctx, &keys, &keys, &mut op_scratch.borrow())
    });
    assert_guard::<BE, _>("lazy sums", "blas_matmul_prepared", InvalidOperand, &mut res, |res| {
        module.blas_matmul_prepared(res, &left, &right, &keys, &mut op_scratch.borrow())
    });
    let errors = [
        module
            .blas_matmul(&mut res, &ct, &column, &ctx, &keys, &keys, &mut op_scratch.borrow())
            .unwrap_err(),
        module
            .blas_matmul_prepared(&mut res, &left, &right, &keys, &mut op_scratch.borrow())
            .unwrap_err(),
    ];
    for e in errors {
        assert!(e.to_string().contains("lazy terms"), "lazy sums: {e}");
    }

    // Operand infos the prepared product's validator rejects, beside a valid operand: without a batch record (term
    // layouts), at another dimension, with a label that does not allow the record or a scale wider than the blocks
    // (the recorded grids `rejected_grids` builds, read through their infos; an uneven grid reports only its first
    // block's layout), and without a block.
    let op = "blas_matmul_prepared";
    let res = alloc(1, 1);
    let unrecorded = validate_blas_matmul_prepared(module, &res, &blas_matmul_term_layout(&ct, &ctx), &right, &keys);
    assert_rejected("no record a", op, InvalidOperand, unrecorded);
    let unrecorded = validate_blas_matmul_prepared(module, &res, &left, &blas_matmul_term_layout(&column, &ctx), &keys);
    assert_rejected("no record b", op, InvalidOperand, unrecorded);
    for (label, bad) in rejected_grids(module, &column)
        .into_iter()
        .filter(|(label, _)| *label != "uneven")
    {
        let infos = validate_blas_matmul_prepared(module, &res, &left, &bad, &keys);
        assert_rejected(&format!("{label} infos"), op, InvalidOperand, infos);
    }
    let (no_cols, no_rows) = (
        Misshapen {
            grid: ct.clone(),
            rows: 1,
            cols: 0,
        },
        Misshapen {
            grid: column.clone(),
            rows: 0,
            cols: 1,
        },
    );
    let empty = validate_blas_matmul_prepared(module, &res, &no_cols, &no_rows, &keys);
    assert_rejected("no block", op, InvalidOperand, empty);

    // Each operation one scratch byte short of its query.
    let mut res = alloc(1, 1);
    let bytes = module.blas_matmul_tmp_bytes(&res, &ct, &column, &ctx, &tsk_layout, &atk_layout);
    assert_short_scratch::<BE, _>("blas_matmul", &mut res, bytes, |res, s| {
        module.blas_matmul(res, &ct, &column, &ctx, &keys, &keys, s)
    });
    let bytes = module.blas_matmul_prepared_tmp_bytes(&res, &left, &right, &tsk_layout);
    assert_short_scratch::<BE, _>("blas_matmul_prepared", &mut res, bytes, |res, s| {
        module.blas_matmul_prepared(res, &left, &right, &keys, s)
    });
    let bytes = module.blas_matmul_prepare_left_tmp_bytes(&ct, &ctx, &atk_layout);
    assert_short_scratch_panics::<BE, _>("blas_matmul_prepare_left", bytes, |s| {
        module.blas_matmul_prepare_left(&ct, &ctx, &keys, s)
    });
    let bytes = module.blas_matmul_prepare_right_tmp_bytes(&column, &ctx, &atk_layout);
    assert_short_scratch_panics::<BE, _>("blas_matmul_prepare_right", bytes, |s| {
        module.blas_matmul_prepare_right(&column, &ctx, &keys, s)
    });

    // Valid grids, a key store without rotation keys or without a tensor key.
    let TestKeys { atks, tsk } = keys;
    let no_atks = TestKeys::<BE> {
        atks: HashMap::new(),
        tsk,
    };
    let no_tsk = TestKeys::<BE> { atks, tsk: None };
    let mut res = alloc(1, 1);
    assert_guard::<BE, _>("no rotation keys", "blas_matmul", MissingKey, &mut res, |res| {
        module.blas_matmul(res, &ct, &column, &ctx, &no_atks, &no_atks, &mut op_scratch.borrow())
    });
    assert_guard::<BE, _>("no tensor key", "blas_matmul", MissingTensorKey, &mut res, |res| {
        module.blas_matmul(res, &ct, &column, &ctx, &no_tsk, &no_tsk, &mut op_scratch.borrow())
    });
    assert_guard::<BE, _>("no tensor key", "blas_matmul_prepared", MissingTensorKey, &mut res, |res| {
        module.blas_matmul_prepared(res, &left, &right, &no_tsk, &mut op_scratch.borrow())
    });
    let left = module.blas_matmul_prepare_left(&ct, &ctx, &no_atks, &mut op_scratch.borrow());
    assert_rejected("no rotation keys", "blas_matmul_prepare_left", MissingKey, left);
    let right = module.blas_matmul_prepare_right(&column, &ctx, &no_atks, &mut op_scratch.borrow());
    assert_rejected("no rotation keys", "blas_matmul_prepare_right", MissingKey, right);
    let left = module
        .blas_matmul_prepare_left(&ct, &ctx, &no_tsk, &mut op_scratch.borrow())
        .unwrap();
    let right = module
        .blas_matmul_prepare_right(&column, &ctx, &no_tsk, &mut op_scratch.borrow())
        .unwrap();
    module
        .blas_matmul_prepared(&mut res, &left, &right, &no_atks, &mut op_scratch.borrow())
        .unwrap();
}

/// Lazy sums at least three times longer than the `2^(63 - base2k)` terms the JKLS kernel sums between normalizations:
/// `dec(matmul(enc(A), enc(B)))` is `A B` for a `1 x K` grid times a `K x 1` grid, whose `K d` tensor products the JKLS
/// kernel normalizes as it sums them, and `blas_matmul_prepared` equals `blas_matmul` bit for bit. Every block is a copy
/// of one ciphertext, so the terms' digits add up coherently, and so does their noise. The product by public matrices
/// sums no coefficient-domain terms: its giant steps add their products in the transform domain, whose radix rule
/// `blas_matmul_pt` documents.
///
/// The bound is in reach only at a high radix: 64 terms at radix 57, `2^40` at FFT64's, which stops near 23. The test
/// therefore runs, on its own ring `N = 32`, only for a suite registered at `base2k >= 57`, and every other suite skips
/// it with a message: an NTT-family backend registers a parameter set at `base2k >= 57`.
pub fn test_blas_matmul_long_sums<BE, F, E>(params: CKKSTestParams, _module: &Module<BE>, _host_module: &Module<HostBytesBackend>)
where
    BE: TestContextBackend<Ring = Standard> + BlasMatMulImpl,
    Module<BE>: TestContextModule<BE> + CKKSEncodingOps<BE, F> + CKKSLinearTransformationOps<BE> + CnvPVecAlloc<BE>,
    F: TestScalar + DiagonalArithmetic,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
{
    if params.base2k < 57 {
        eprintln!(
            "test_blas_matmul_long_sums skips base2k {}: it needs base2k >= 57, which NTT-family suites register",
            params.base2k
        );
        return;
    }
    let params = CKKSTestParams { n: 32, hw: 16, ..params };
    let module = &Module::<BE>::new(params.n as u64);
    let host_module = &Module::<HostBytesBackend>::new(params.n as u64);
    // One 4 x 4 block fills the 16 slots, whatever the suite's dimension.
    let (log_d, d) = (2, 4);
    let (slots, bound) = (params.n / 2, max_lazy_terms(params.base2k));
    let lanes = slots / (d * d);
    let encoder = ReferenceEncoder::<E>::new(slots).unwrap();
    let (sk_raw, sk) = gen_sk_with_raw(&params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, module);
    let (ctx, atks) = context_and_keys::<BE, F>(&params, module, log_d, &sk_raw, &mut scratch.borrow());
    let tsk = gen_tsk(&params, module, &sk_raw, &mut scratch.borrow());
    let keys = TestKeys::<BE> { atks, tsk: Some(tsk) };
    let (tsk_layout, atk_layout) = (params.tsk_layout(), params.atk_layout());
    let (a0, b0) = (Grid::model(log_d, 1, 1, lanes, 7), Grid::model(log_d, 1, 1, lanes, 8));
    let ct_a0 = encrypt_grid::<BE, F, E>(
        &params,
        module,
        host_module,
        &encoder,
        &sk,
        params.k,
        0,
        &a0,
        &mut scratch.borrow(),
    );
    let ct_b0 = encrypt_grid::<BE, F, E>(
        &params,
        module,
        host_module,
        &encoder,
        &sk,
        params.k,
        0,
        &b0,
        &mut scratch.borrow(),
    );
    let repeat = |g: &BlasCiphertextOwned<BE>, rows: usize, cols: usize| {
        BlasCiphertext::from_blocks(vec![g.block(0, 0).clone(); rows * cols], log_d, rows, cols).unwrap()
    };
    // `copies` equal terms repeat their noise as often: `log2(copies)` bits of precision go.
    let check = |label: &str, res: &BlasCiphertextOwned<BE>, want: &Grid, copies: usize, scratch: &mut ScratchArena<'_, BE>| {
        let (block, (re, im)) = (res.block(0, 0), &want.blocks(slots)[0]);
        let to_f = |v: &[f64]| v.iter().map(|&x| F::from_f64(x).unwrap()).collect::<Vec<F>>();
        let log_delta = block.log_delta() - copies.next_power_of_two().ilog2() as usize;
        assert_decrypt_precision_at_log_delta::<BE, F, E>(
            label,
            &params,
            module,
            &encoder,
            block,
            &sk,
            &to_f(re),
            &to_f(im),
            log_delta,
            scratch,
        );
    };

    // `d` terms per inner block: `K d` is at least three times the bound.
    let inner = (3 * bound).div_ceil(d);
    let a = Grid::from_fn(log_d, 1, inner, lanes, |s, r, c| a0.at(s, r, c % d));
    let b = Grid::from_fn(log_d, inner, 1, lanes, |s, r, c| b0.at(s, r % d, c));
    let (ct_a, ct_b) = (repeat(&ct_a0, 1, inner), repeat(&ct_b0, inner, 1));
    let mut res = alloc_grid(&params, module, log_d, 1, 1, params.k);
    let bytes = module.blas_matmul_tmp_bytes(&res, &ct_a, &ct_b, &ctx, &tsk_layout, &atk_layout);
    module
        .blas_matmul(
            &mut res,
            &ct_a,
            &ct_b,
            &ctx,
            &keys,
            &keys,
            &mut ScratchOwned::<BE>::alloc(bytes).borrow(),
        )
        .unwrap();
    check("matmul", &res, &a.matmul(&b), inner, &mut scratch.borrow());

    let bytes = module
        .blas_matmul_prepare_left_tmp_bytes(&ct_a, &ctx, &atk_layout)
        .max(module.blas_matmul_prepare_right_tmp_bytes(&ct_b, &ctx, &atk_layout));
    let mut prepare_scratch = ScratchOwned::<BE>::alloc(bytes);
    let left = module
        .blas_matmul_prepare_left(&ct_a, &ctx, &keys, &mut prepare_scratch.borrow())
        .unwrap();
    let right = module
        .blas_matmul_prepare_right(&ct_b, &ctx, &keys, &mut prepare_scratch.borrow())
        .unwrap();
    let mut prepared = alloc_grid(&params, module, log_d, 1, 1, params.k);
    let bytes = module.blas_matmul_prepared_tmp_bytes(&prepared, &left, &right, &tsk_layout);
    module
        .blas_matmul_prepared(
            &mut prepared,
            &left,
            &right,
            &keys,
            &mut ScratchOwned::<BE>::alloc(bytes).borrow(),
        )
        .unwrap();
    assert_eq!(grid_snapshot::<BE, _>(&prepared), grid_snapshot::<BE, _>(&res));
}
