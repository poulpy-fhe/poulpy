//! JKLS matrix products of whole grids.

use std::{
    collections::{BTreeSet, HashMap},
    sync::Mutex,
};

use poulpy_ckks::{
    CKKSInfos, CKKSResult, CoeffsMeta, GLWEPlaintextInfos, Scale, SetGLWEPlaintextInfos,
    api::{
        CKKSCopyOps, CKKSEncodingOps, CKKSLinearTransformationOps, CKKSRotateOps, LinearTransformationBabySteps,
        LinearTransformationPrepared,
    },
    layouts::{CKKSCiphertextOwned, CKKSModuleAlloc, CKKSPreparedLeft, CKKSPreparedRight},
    test_suite::{
        CKKSTestParams,
        helpers::{
            TestContextBackend, TestContextModule, TestScalar, alloc_scratch, assert_decrypt_precision_at_log_delta,
            gen_sk_with_raw, gen_tsk,
        },
        parity::helpers::{Snapshot, snapshot, with_scratch},
        reference_encoder::ReferenceEncoder,
    },
};
use poulpy_core::{
    api::GLWETensorKeyPreparedFactory,
    layouts::{
        BackendGLWESecret, DiagonalArithmetic, GLWETensorKeyLayout, GLWEToBackendMut, GetAutomorphismKey, LWEInfos, Rank,
        TorusPrecision,
        prepared::{GLWEAutomorphismKeyPreparedBackendRef, GLWESecretPrepared},
    },
};
use poulpy_hal::{
    AlignedBuf,
    api::{CnvPVecAlloc, NegacyclicFFT, NegacyclicFFTNew, ScratchOwnedBorrow},
    layouts::{
        Backend, CyclotomicOrder, DataView, HostBytesBackend, HostDataMut, HostDataRef, Module, PrepareHint,
        PrepareHint::{OneShot, Reuse},
        ScratchArena, ScratchOwned, Standard, ZnxViewMut,
    },
};

use super::helpers::{
    Grid, HINTS, MASKED_PT_ALGORITHMS, MIXES, Misshapen, OperandInfos, PT_ALGORITHMS,
    Rejection::{self, Budget, InvalidOperand, MissingKey, MissingTensorKey},
    StrictKeys, TestKeys, alloc_grid, assert_grid_decrypt, assert_guard, assert_panics, assert_rejected, assert_short_scratch,
    assert_short_scratch_panics, at_scale, blas_context, context_and_keys, corner, encoded_context, encrypt_grid,
    foreign_contexts, foreign_layouts, grid_snapshot, map_galois_elements, mapped_layout, matmul_left_operand,
    matmul_right_operand, operand_infos, other_scales, poisoned_scratch, product_layout, pt_operand, rejected_grids, relabel,
    rotation_keys, to_f, want_operand_infos,
};
use crate::{
    api::{BlasEncodingOps, BlasMatIntraOps, BlasMatMulOps},
    layouts::{
        BlasCiphertext, BlasCiphertextOwned, BlasContext, BlasContextInfos, BlasGridInfos, BlasKeyUses, BlasLayout,
        BlasMatMulLeft, BlasMatMulOperand, BlasMatMulOperandLayout, BlasMatMulOperandPreparedFactory, BlasMatMulRight,
        BlasModuleAlloc, BlasPlan, BlasPtAlgorithm, BlasPtOperand, BlasPtOperandInfos, Left, Right,
        blas_matmul_left_operand_layout, blas_matmul_pt_left_key_uses, blas_matmul_pt_left_output_layout,
        blas_matmul_pt_right_key_uses, blas_matmul_pt_right_output_layout, blas_matmul_right_operand_layout,
        ciphertext::layout_of, planning::pt_label, replicated, validate::MASKLESS, validate_blas_matmul_prepared,
    },
    oep::{BlasEncodingImpl, BlasImpl, BlasMatMulImpl},
    reference::BlasMatMulReference,
};

/// `dec(matmul(enc(A), enc(B)))` is `A B` for every matrix of the batch, for a `2 x 3` grid times a `3 x 2` grid, with
/// the JKLS product's metadata and canonical blocks, and `blas_matmul` equals `blas_matmul_prepared` of every mix of hot
/// and cold operands, bit for bit, all with the rotation keys of the plan's product set alone. Through maps at the
/// factors' scale and at scales below and above it ([`other_scales`]), and through the latter also of factors at two
/// scales, in either order, one an eighth of the scale lower and either as much narrower, so that the terms of both
/// keep one budget, or as wide, so that its terms have the larger budget (crossed scales): each cold operand reports
/// its input's dimensions and batch and its terms' layout, its input's `2p` narrower for the maps' scale `p`, `p` for a
/// right factor holding one matrix, each hot one its input's layout, both their hint and their input's layout as their
/// source, and the product follows from the terms, at the smaller of the factors' scales. The hot operands are prepared
/// once, under the maps at the factors' scale, and serve the products under every context, which defines their terms;
/// those of `A` and `B` come from copies, dropped or overwritten once prepared. One cold operand of `B` under each
/// context multiplies every left factor there, hot and cold. Beside `B` cold under the maps below the factors' scale,
/// the hot `A` gives, under the maps above it, `A` cold under those times that `B`, which decrypts to `A B` at the
/// ct x ct rule's metadata on the terms.
pub fn test_blas_matmul<BE, F, E>(
    params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
    log_d: usize,
) where
    BE: TestContextBackend<Ring = Standard> + BlasMatMulImpl + BlasEncodingImpl,
    Module<BE>: TestContextModule<BE>
        + CKKSEncodingOps<BE, F>
        + BlasEncodingOps<BE, F>
        + CKKSLinearTransformationOps<BE>
        + CnvPVecAlloc<BE>,
    F: TestScalar + DiagonalArithmetic,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
{
    let slots = params.n / 2;
    let encoder = ReferenceEncoder::<E>::new(slots).unwrap();
    let (sk_raw, sk) = gen_sk_with_raw(&params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, module);
    // The keys of the products' set alone, which covers both preparations and the prepared product, at any scale.
    let elements = BlasPlan::new(log_d).matmul_galois_elements(module.cyclotomic_order());
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
    // `A` and `B` again an eighth of their scale lower, so that either times the other at `delta` multiplies factors at
    // two scales: as much narrower, so that the terms of both keep one budget, and as wide, so that the terms at the
    // lower scale have the larger budget (crossed scales), where the larger budget plus the larger scale, from which
    // the product takes its offset, lies `gap` bits above the wider term's width. The lower scale lies strictly between
    // `delta` and the lower of `other_scales`.
    let delta = params.prec().log_delta();
    let gap = delta / 8;
    let [[ct_a_low, ct_b_low], [ct_a_crossed, ct_b_crossed]] = [params.k - gap, params.k].map(|k| {
        [&a, &b].map(|grid| {
            encrypt_grid::<BE, F, E>(
                &at_scale(&params, delta - gap),
                module,
                host_module,
                &encoder,
                &sk,
                k,
                0,
                grid,
                &mut scratch.borrow(),
            )
        })
    });
    let (tsk_layout, atk_layout) = (params.tsk_layout(), params.atk_layout());
    let scales: Vec<usize> = [delta].into_iter().chain(other_scales(&params)).collect();
    let contexts: Vec<_> = scales
        .iter()
        .map(|&p| blas_context::<BE, F>(&at_scale(&params, p), module, log_d))
        .collect();
    // Each preparation and product on exactly the scratch its query reports.
    let prepare_left = |x: &BlasCiphertextOwned<BE>, ctx: &BlasContext<BE>, hint| {
        let bytes = module.blas_matmul_prepare_left_tmp_bytes(x, ctx, &atk_layout, hint);
        matmul_left_operand(module, x, ctx, &keys, hint, &mut poisoned_scratch::<BE>(bytes).borrow()).unwrap()
    };
    let prepare_right = |y: &BlasCiphertextOwned<BE>, ctx: &BlasContext<BE>, hint| {
        let bytes = module.blas_matmul_prepare_right_tmp_bytes(y, ctx, &atk_layout, hint);
        matmul_right_operand(module, y, ctx, &keys, hint, &mut poisoned_scratch::<BE>(bytes).borrow()).unwrap()
    };
    let product = |left: &BlasMatMulLeft<BE>, right: &BlasMatMulRight<BE>, ctx: &BlasContext<BE>| {
        let mut res = alloc_grid(&params, module, log_d, m, n, params.k);
        let bytes = module.blas_matmul_prepared_tmp_bytes(&res, left, right, ctx, &tsk_layout, &atk_layout);
        module
            .blas_matmul_prepared(
                &mut res,
                left,
                right,
                ctx,
                &keys,
                &keys,
                &mut poisoned_scratch::<BE>(bytes).borrow(),
            )
            .unwrap();
        res
    };
    // Hot operands of every factor, prepared once under the context at the factors' scale: the product evaluates their
    // terms under its own, so they serve the products under every context. Those of `A` and `B` come from copies,
    // dropped or overwritten once prepared, which the operands, owning their storage, do not see.
    let (lefts, rights) = ([&ct_a, &ct_a_low, &ct_a_crossed], [&ct_b, &ct_b_low, &ct_b_crossed]);
    let hot_a = prepare_left(&ct_a.clone(), &contexts[0], OneShot);
    let mut copy = ct_b.clone();
    let hot_b = prepare_right(&copy, &contexts[0], OneShot);
    for (block, other) in copy.blocks_mut().iter_mut().zip(ct_b_low.blocks()) {
        *block = other.clone();
    }
    let hot_lefts = [
        hot_a,
        prepare_left(lefts[1], &contexts[0], OneShot),
        prepare_left(lefts[2], &contexts[0], OneShot),
    ];
    let hot_rights = [
        hot_b,
        prepare_right(rights[1], &contexts[0], OneShot),
        prepare_right(rights[2], &contexts[0], OneShot),
    ];
    for (&p, ctx) in scales.iter().zip(&contexts) {
        let maps = at_scale(&params, p);
        // Factors at one scale; through maps at a scale apart from both, also at two scales, in either order, crossed
        // or not, every one cold under `ctx`: the cold `B` multiplies `A`, `A` low and `A` crossed, hot and cold.
        let (pairs, used): (&[(usize, usize)], usize) = if p == delta {
            (&[(0, 0)], 1)
        } else {
            (&[(0, 0), (0, 1), (1, 0), (0, 2), (2, 0)], 3)
        };
        let cold_lefts: Vec<_> = lefts[..used].iter().map(|x| prepare_left(x, ctx, Reuse)).collect();
        let cold_rights: Vec<_> = rights[..used].iter().map(|y| prepare_right(y, ctx, Reuse)).collect();
        let cold_b = operand_infos(&cold_rights[0]);
        for &(i, j) in pairs {
            let (x, y) = (lefts[i], rights[j]);
            let (lx, ly) = (x.block_layout(), y.block_layout());
            let label = format!(
                "matmul p={p} k_a={} delta_a={} k_b={} delta_b={}",
                lx.k().as_usize(),
                lx.log_delta(),
                ly.k().as_usize(),
                ly.log_delta()
            );
            let mut res = alloc_grid(&params, module, log_d, m, n, params.k);
            let bytes = module.blas_matmul_tmp_bytes(&res, x, y, ctx, &tsk_layout, &atk_layout);
            module
                .blas_matmul(&mut res, x, y, ctx, &keys, &keys, &mut poisoned_scratch::<BE>(bytes).borrow())
                .unwrap();
            assert_grid_decrypt::<BE, F, E>(
                &label,
                &maps,
                module,
                &encoder,
                &res,
                &sk,
                &a.matmul(&b),
                &product_layout::<BE::Ring>(x, y, m, n, params.k, p),
                &mut scratch.borrow(),
            );
            // One relinearization leaves every block normalized.
            assert!(
                res.blocks().iter().all(|block| block.is_canonical()),
                "{label}: a block is not canonical"
            );

            // Each operand reports its hint, its input's batch and block layout as its source, and as its block layout
            // its input's when hot and its terms', two maps of scale `p` below its input's width, one on a right factor
            // holding one matrix, when cold.
            for (side, ct, (hot, cold)) in [
                ("left", x, (operand_infos(&hot_lefts[i]), operand_infos(&cold_lefts[i]))),
                ("right", y, (operand_infos(&hot_rights[j]), operand_infos(&cold_rights[j]))),
            ] {
                let scale = if side == "right" && replicated::<BE::Ring>(ct) {
                    p
                } else {
                    2 * p
                };
                let terms = mapped_layout(ct, ct.rows(), ct.cols(), params.k, scale);
                assert_eq!(
                    hot,
                    want_operand_infos(ct, OneShot, terms),
                    "{label}: hot {side} operand infos"
                );
                assert_eq!(
                    cold,
                    want_operand_infos(ct, Reuse, terms),
                    "{label}: cold {side} operand infos"
                );
            }
            // Every mix of hot and cold operands is `blas_matmul`, bit for bit.
            for (hl, hr) in MIXES {
                let left = if hl == OneShot { &hot_lefts[i] } else { &cold_lefts[i] };
                let right = if hr == OneShot { &hot_rights[j] } else { &cold_rights[j] };
                assert_eq!(
                    grid_snapshot::<BE, _>(&product(left, right, ctx)),
                    grid_snapshot::<BE, _>(&res),
                    "{label}: {hl:?} x {hr:?}"
                );
            }
        }
        assert_eq!(
            operand_infos(&cold_rights[0]),
            cold_b,
            "p={p}: the products changed the cold B"
        );
    }

    // `A` copied into blocks one limb wider, `k < max_k`: the queries and the products read the blocks' widths, so
    // each runs on exactly the scratch its query reports, and gives `A B`'s bits under every mix of hot and cold
    // operands.
    let ctx = &contexts[0];
    let want = grid_snapshot::<BE, _>(&product(&hot_lefts[0], &hot_rights[0], ctx));
    let mut wide_a = alloc_grid(&params, module, log_d, m, inner, params.k + params.base2k);
    for (dst, src) in wide_a.blocks_mut().iter_mut().zip(ct_a.blocks()) {
        module.ckks_copy(dst, src, &mut scratch.borrow()).unwrap();
    }
    let mut res = alloc_grid(&params, module, log_d, m, n, params.k);
    let bytes = module.blas_matmul_tmp_bytes(&res, &wide_a, &ct_b, ctx, &tsk_layout, &atk_layout);
    module
        .blas_matmul(
            &mut res,
            &wide_a,
            &ct_b,
            ctx,
            &keys,
            &keys,
            &mut poisoned_scratch::<BE>(bytes).borrow(),
        )
        .unwrap();
    assert_eq!(grid_snapshot::<BE, _>(&res), want, "matmul k < max_k");
    for (hl, hr) in MIXES {
        let (left, right) = (prepare_left(&wide_a, ctx, hl), prepare_right(&ct_b, ctx, hr));
        assert_eq!(
            grid_snapshot::<BE, _>(&product(&left, &right, ctx)),
            want,
            "matmul k < max_k: {hl:?} x {hr:?}"
        );
    }

    // A hot side takes the terms of the product's context beside a cold side prepared under another: each side's maps
    // spend their own scale, and the product stamps by the ct x ct rule on the terms. Under the maps above the factors'
    // scale, the hot `A` times `B` cold under those below is `A` cold under the former times that `B`, which decrypts to
    // `A B` at the rule's metadata.
    let (low, high) = (&contexts[1], &contexts[2]);
    let (cold_a, cold_b) = (prepare_left(&ct_a, high, Reuse), prepare_right(&ct_b, low, Reuse));
    let want = product(&cold_a, &cold_b, high);
    assert_grid_decrypt::<BE, F, E>(
        "matmul across contexts",
        &at_scale(&params, scales[1]),
        module,
        &encoder,
        &want,
        &sk,
        &a.matmul(&b),
        &product_layout::<BE::Ring>(&cold_a, &cold_b, m, n, params.k, 0),
        &mut scratch.borrow(),
    );
    assert_eq!(
        grid_snapshot::<BE, _>(&product(&hot_lefts[0], &cold_b, high)),
        grid_snapshot::<BE, _>(&want),
        "matmul across contexts: hot A x cold B"
    );
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
/// with its own baby-step cache, then takes the product's joint label, `label`: `pt_label` of the input and the operand's
/// count, `min(a.log_sparsity, log2(S / (count d^2)))`, not its sources'.
fn summed_terms<'t, BE>(
    module: &Module<BE>,
    mut res: BlasCiphertextOwned<BE>,
    inner: usize,
    label: usize,
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
        let (srcs, lts): (Vec<_>, Vec<_>) = (0..inner).map(|l| term(t / cols, t % cols, l)).unzip();
        let caches: Vec<_> = srcs
            .iter()
            .zip(&lts)
            .map(|(&src, lt)| {
                let mut cache = LinearTransformationBabySteps::alloc(module, lt.baby_steps(), src);
                module
                    .ckks_prepare_linear_transformation_baby_steps(&mut cache, src, keys, scratch)
                    .unwrap();
                cache
            })
            .collect();
        let babies: Vec<_> = caches.iter().collect();
        module
            .ckks_eval_linear_transformation_sum_into(
                dst,
                &srcs,
                &babies,
                &lts,
                &poulpy_core::layouts::AutomorphismKeysOnly(keys),
                scratch,
            )
            .unwrap();
        dst.set_log_sparsity(label);
    }
    grid_snapshot::<BE, _>(&res)
}

/// The grid a product by public matrices of `a` through `b` sums the terms of: `a` itself, or under JKLS every block of
/// `a` through `b`'s first map, at scale `p_b`, into a block of the width it leaves, labelled as its source.
fn staged<BE, S>(
    params: &CKKSTestParams,
    module: &Module<BE>,
    a: &BlasCiphertextOwned<BE>,
    b: &BlasPtOperand<BE, S>,
    p_b: usize,
    keys: &impl GetAutomorphismKey<BE>,
    scratch: &mut ScratchArena<'_, BE>,
) -> BlasCiphertextOwned<BE>
where
    BE: TestContextBackend,
    Module<BE>: CKKSLinearTransformationOps<BE> + CnvPVecAlloc<BE> + CKKSModuleAlloc<BE>,
{
    let Some(first) = b.first() else {
        return a.clone();
    };
    let k = a.block_layout().k().as_usize() - p_b;
    let mut s = alloc_grid(params, module, a.log_d(), a.rows(), a.cols(), k);
    for (dst, src) in s.blocks_mut().iter_mut().zip(a.blocks()) {
        module
            .ckks_eval_linear_transformation_self_into(
                dst,
                src,
                first,
                &poulpy_core::layouts::AutomorphismKeysOnly(keys),
                scratch,
            )
            .unwrap();
        dst.set_log_sparsity(src.log_sparsity());
    }
    s
}

/// `dec(matmul_pt_right(enc(A), B))` is `A B_(s mod count)` in matrix slot `s`, for a `2 x 3` grid times a `3 x 2` operand
/// with one, two, and one public matrix per matrix slot, of these counts those that divide the matrix slots, at `A`'s
/// scale, and with one public matrix at scales below and above it ([`other_scales`]), under `Diagonal` and `Jkls`, all
/// with the keys of the plan's set alone, at the metadata [`mapped_layout`] gives for `B`'s scale, twice it under JKLS,
/// its blocks canonical. Each block of `res` is poulpy-ckks's summed linear transformation of its three terms in
/// ascending inner index, under JKLS of the results of `B`'s first map, bit for bit, also at radix 26, where an FFT
/// backend's transform-domain sums round, so that the order of the terms shows in the bits. A counting key store sees
/// the giant steps' keys looked up for one key switch per giant step and block of `res`.
pub fn test_blas_matmul_pt_right<BE, F, E>(
    params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
    log_d: usize,
) where
    BE: TestContextBackend<Ring = Standard> + BlasMatMulImpl + BlasEncodingImpl,
    Module<BE>: TestContextModule<BE>
        + CKKSEncodingOps<BE, F>
        + BlasEncodingOps<BE, F>
        + CKKSLinearTransformationOps<BE>
        + CnvPVecAlloc<BE>,
    F: TestScalar + DiagonalArithmetic,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
{
    let slots = params.n / 2;
    let lanes = slots >> (2 * log_d);
    let ctx = blas_context::<BE, F>(&params, module, log_d);
    let encoder = ReferenceEncoder::<E>::new(slots).unwrap();
    let (m, inner, n) = (2, 3, 2);
    let encode = |params: &CKKSTestParams, algorithm: BlasPtAlgorithm, count: usize| {
        pt_operand::<BE, F, Right>(
            module,
            BlasPlan::new(log_d).with_pt_algorithm(algorithm),
            inner,
            n,
            count,
            |m, r, c| F::from_f64(public_entry(m, r, c)).unwrap(),
            params.base2k.into(),
            params.prec().into(),
        )
    };
    // The keys of the plan's set alone, whatever the operand's count.
    let elements = BlasPlan::new(log_d).matmul_pt_right_galois_elements(module.cyclotomic_order());
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
        // At `a`'s scale, then one public matrix at scales below and above it.
        let delta = params.prec().log_delta();
        let cases: Vec<(usize, usize)> = counts
            .iter()
            .map(|&count| (count, delta))
            .chain(other_scales(&params).map(|p_b| (1, p_b)))
            .collect();
        for algorithm in PT_ALGORITHMS {
            for &(count, p_b) in &cases {
                let stages = if algorithm == BlasPtAlgorithm::Jkls { 2 } else { 1 };
                let public = at_scale(&params, p_b);
                let b = encode(&public, algorithm, count);
                // `res` one limb wider than `a` for one public matrix.
                let res_k = if count == 1 { params.k + params.base2k } else { params.k };
                let label = format!("matmul_pt_right {algorithm:?} count={count} p_B={p_b} res_k={res_k}");
                let mut res = alloc_grid(&params, module, log_d, m, n, res_k);
                let bytes = module.blas_matmul_pt_right_tmp_bytes(&res, &ct, &b, &ctx, &params.atk_layout());
                let mut op_scratch = poisoned_scratch::<BE>(bytes);
                let counting = CountingKeys::new(&keys);
                module
                    .blas_matmul_pt_right(&mut res, &ct, &b, &ctx, &counting, &mut op_scratch.borrow())
                    .unwrap();
                assert_grid_decrypt::<BE, F, E>(
                    &label,
                    &public,
                    module,
                    &encoder,
                    &res,
                    &sk,
                    &a.matmul_public(n, count, public_entry),
                    &mapped_layout(&ct, m, n, res_k, stages * p_b),
                    &mut scratch.borrow(),
                );
                // The giant loop normalizes each block once.
                assert!(
                    res.blocks().iter().all(|block| block.is_canonical()),
                    "{label}: a block is not canonical"
                );
                // The giant steps' keys, at their width: the validator and the reference's scratch check look up those
                // of `b`'s `K n` maps, poulpy-ckks's summed transformation checks those of its `K` terms for each block
                // of `res`, then its giant loop looks each one up once per block, for its key switch. That is `m n g`
                // key switches for `g` rotating giant steps, where one transformation per term would take `K m n g`.
                let g = rotating_giant_steps(b.block(0, 0));
                let giant_k = res.block_layout().k().as_usize();
                assert_eq!(
                    counting.at(giant_k),
                    (2 * inner * n + inner * m * n + m * n) * g,
                    "{label}: giant-step key lookups"
                );
                let src = staged(&params, module, &ct, &b, p_b, &keys, &mut scratch.borrow());
                let want = summed_terms(
                    module,
                    alloc_grid(&params, module, log_d, m, n, res_k),
                    inner,
                    pt_label(b.slots(), &ct, b.count()),
                    |i, j, l| (src.block(i, l), b.block(l, j)),
                    &keys,
                    &mut op_scratch.borrow(),
                );
                assert_eq!(grid_snapshot::<BE, _>(&res), want, "{label}: summed terms");
            }
        }
    }

    // At radix 26, where an FFT backend rounds the transform-domain sum of a giant step's products, so that summing the
    // terms in another order changes the bits.
    let params = CKKSTestParams {
        base2k: 26,
        // Room for a JKLS product's two scales above the inputs' at the largest test scale, 80 bits.
        k: 10 * 26,
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
    for algorithm in PT_ALGORITHMS {
        let b = encode(&params, algorithm, 1);
        let mut res = alloc_grid(&params, module, log_d, m, n, params.k);
        let mut op_scratch =
            poisoned_scratch::<BE>(module.blas_matmul_pt_right_tmp_bytes(&res, &ct, &b, &ctx, &params.atk_layout()));
        module
            .blas_matmul_pt_right(&mut res, &ct, &b, &ctx, &keys, &mut op_scratch.borrow())
            .unwrap();
        let src = staged(
            &params,
            module,
            &ct,
            &b,
            params.prec().log_delta(),
            &keys,
            &mut scratch.borrow(),
        );
        let want = summed_terms(
            module,
            alloc_grid(&params, module, log_d, m, n, params.k),
            inner,
            pt_label(b.slots(), &ct, b.count()),
            |i, j, l| (src.block(i, l), b.block(l, j)),
            &keys,
            &mut op_scratch.borrow(),
        );
        assert_eq!(
            grid_snapshot::<BE, _>(&res),
            want,
            "matmul_pt_right {algorithm:?} at radix 26: summed terms"
        );
    }
}

/// A `1 x 1` grid one bit narrower than `log_delta`, into which poulpy-ckks rejects the budget of a product of blocks at
/// that scale. Where the slots hold two matrices or more, it records half of them, where a product of dense inputs
/// records them all, so that a record stamped on failure shows.
fn res_below_scale<BE>(params: &CKKSTestParams, module: &Module<BE>, log_d: usize, log_delta: usize) -> BlasCiphertextOwned<BE>
where
    BE: TestContextBackend<Ring = Standard>,
    Module<BE>: CKKSModuleAlloc<BE>,
{
    let res = alloc_grid(params, module, log_d, 1, 1, log_delta - 1);
    if (params.n / 2) >> (2 * log_d) < 2 {
        return res;
    }
    BlasCiphertext::from_blocks(relabel(&res, |meta| meta.log_sparsity = 1).into_blocks(), log_d, 1, 1).unwrap()
}

/// The block dimension beside `log_d` for an operand at another dimension: the previous one, whose smaller matrices fit
/// wherever a block does, or the next one from 2 x 2 blocks, if it fits the `slots`; otherwise `test` skips that case
/// with a message.
fn other_log_d(test: &str, log_d: usize, slots: usize) -> Option<usize> {
    let other = if log_d > 1 { log_d - 1 } else { log_d + 1 };
    if 2 * other > slots.ilog2() as usize {
        eprintln!("{test} skips an operand at log_d {other}: it needs d^2 <= {slots} slots");
        return None;
    }
    Some(other)
}

/// An inner-dimension or `res` shape mismatch, more public matrices than a sparse input records, a copy relabelled past
/// `log2` of the ring degree, an operand encoded on a module of another degree, at another radix or at another
/// dimension, every grid [`rejected_grids`] builds, as `a` or as `res`, a `res` of another degree, radix or rank
/// ([`foreign_layouts`]), a missing rotation key, and `a` and `res` both of another rank than the keys are rejected
/// before `res` changes, as are a `res` narrower than the scale of `a`, and under JKLS an `a` one bit short of `B`'s
/// scale, then of twice it, by poulpy-ckks's budget check, which leaves `a` unchanged too; a product one scratch byte
/// short of its query, under `Diagonal` and `Jkls`, panics before `res` changes; an operand whose block count overflows
/// `usize` panics at its allocation. A grid holding one matrix, relabelled in place to one matrix over its dense
/// record, times two public matrices gives, under `Diagonal` and `Jkls`, blocks labelled for two matrices, not
/// [`replicated`]. Beside `2 x 2` blocks on fewer than 16 slots, where no other dimension fits, the operand at another
/// dimension is skipped with a message, and on one matrix slot the relabelled grid.
pub fn test_blas_matmul_pt_right_guards<BE, F, E>(
    params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
    log_d: usize,
) where
    BE: TestContextBackend<Ring = Standard> + BlasMatMulImpl + BlasEncodingImpl,
    Module<BE>: TestContextModule<BE>
        + CKKSEncodingOps<BE, F>
        + BlasEncodingOps<BE, F>
        + CKKSLinearTransformationOps<BE>
        + CnvPVecAlloc<BE>,
    F: TestScalar + DiagonalArithmetic,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
{
    let slots = params.n / 2;
    let lanes = slots >> (2 * log_d);
    let ctx = blas_context::<BE, F>(&params, module, log_d);
    let encoder = ReferenceEncoder::<E>::new(slots).unwrap();
    let (sk_raw, sk) = gen_sk_with_raw(&params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, module);
    let encode = |module: &Module<BE>, log_d: usize, count: usize, base2k: usize| {
        pt_operand::<BE, F, Right>(
            module,
            BlasPlan::new(log_d),
            2,
            1,
            count,
            |m, r, c| F::from_f64(public_entry(m, r, c)).unwrap(),
            base2k.into(),
            params.prec().into(),
        )
    };
    let b = encode(module, log_d, lanes, params.base2k);
    // One public matrix, which divides the matrix slots of any module and of a grid read at any dimension.
    let one = encode(module, log_d, 1, params.base2k);
    let other =
        other_log_d("test_blas_matmul_pt_right_guards", log_d, slots).map(|other| encode(module, other, 1, params.base2k));
    let double = Module::<BE>::new(2 * params.n as u64);
    let foreign = [
        ("degree", encode(&double, log_d, 1, params.base2k)),
        ("base2k", encode(module, log_d, 1, params.base2k - 1)),
    ];
    // Keys for the other operands too, so no missing key stops `blas_matmul_pt_right` from evaluating them.
    let blocks = [&b, &one]
        .into_iter()
        .chain(&other)
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
    let mut op_scratch = poisoned_scratch::<BE>(module.blas_matmul_pt_right_tmp_bytes(&ct, &ct, &b, &ctx, &params.atk_layout()));

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
        assert_guard::<BE, _>(label, "blas_matmul_pt_right", InvalidOperand, &mut res, |res| {
            module.blas_matmul_pt_right(res, a, &b, &ctx, &keys, &mut op_scratch.borrow())
        });
    }
    for (label, b) in &foreign {
        let mut res = alloc_grid(&params, module, log_d, 1, 1, params.k);
        assert_guard::<BE, _>(label, "blas_matmul_pt_right", InvalidOperand, &mut res, |res| {
            module.blas_matmul_pt_right(res, &ct, b, &ctx, &keys, &mut op_scratch.borrow())
        });
    }
    // An operand encoded at another dimension, which `ct` and `res` are not read at.
    if let Some(other) = &other {
        let mut res = alloc_grid(&params, module, log_d, 1, 1, params.k);
        assert_guard::<BE, _>("operand dimension", "blas_matmul_pt_right", InvalidOperand, &mut res, |res| {
            module.blas_matmul_pt_right(res, &ct, other, &ctx, &keys, &mut op_scratch.borrow())
        });
    }

    // Each rejected grid as `a` or as `res` of a 2 x 2 grid times one public matrix.
    let square = encrypt(2, 2, 13, &mut scratch.borrow());
    let res_2x1 = alloc_grid(&params, module, log_d, 2, 1, params.k);
    for ((label, a), (_, mut bad)) in rejected_grids(module, &square)
        .into_iter()
        .zip(rejected_grids(module, &res_2x1))
    {
        let mut res = res_2x1.clone();
        assert_guard::<BE, _>(
            &format!("{label} a"),
            "blas_matmul_pt_right",
            InvalidOperand,
            &mut res,
            |res| module.blas_matmul_pt_right(res, &a, &one, &ctx, &keys, &mut op_scratch.borrow()),
        );
        assert_guard::<BE, _>(
            &format!("{label} res"),
            "blas_matmul_pt_right",
            InvalidOperand,
            &mut bad,
            |res| module.blas_matmul_pt_right(res, &square, &one, &ctx, &keys, &mut op_scratch.borrow()),
        );
    }
    for (label, mut bad) in foreign_layouts(module, &res_2x1) {
        assert_guard::<BE, _>(
            &format!("{label} res"),
            "blas_matmul_pt_right",
            InvalidOperand,
            &mut bad,
            |res| module.blas_matmul_pt_right(res, &square, &one, &ctx, &keys, &mut op_scratch.borrow()),
        );
    }

    let mut res = alloc_grid(&params, module, log_d, 1, 1, params.k);
    let no_keys = TestKeys::<BE> {
        atks: HashMap::new(),
        tsk: None,
    };
    assert_guard::<BE, _>("no keys", "blas_matmul_pt_right", MissingKey, &mut res, |res| {
        module.blas_matmul_pt_right(res, &ct, &b, &ctx, &no_keys, &mut op_scratch.borrow())
    });

    // `a` and `res` both of another rank than the keys, which the first key of `b`'s maps rejects.
    let ([.., (_, rank_a)], [.., (_, mut rank_res)]) = (foreign_layouts(module, &ct), foreign_layouts(module, &res));
    assert_guard::<BE, _>("key rank", "blas_matmul_pt_right", InvalidOperand, &mut rank_res, |res| {
        module.blas_matmul_pt_right(res, &rank_a, &b, &ctx, &keys, &mut op_scratch.borrow())
    });

    // `res` narrower than the scale of `a`: poulpy-ckks rejects the budget of the first block's summed transformation
    // inside the implementation, before any write.
    let input = grid_snapshot::<BE, _>(&ct);
    let mut narrow_res = res_below_scale(&params, module, log_d, ct.block_layout().log_delta());
    assert_guard::<BE, _>(
        "budget",
        "ckks_eval_linear_transformation_sum_into",
        Budget,
        &mut narrow_res,
        |res| module.blas_matmul_pt_right(res, &ct, &b, &ctx, &keys, &mut op_scratch.borrow()),
    );
    assert!(grid_snapshot::<BE, _>(&ct) == input, "budget: blas_matmul_pt_right changed a");

    let jkls = pt_operand::<BE, F, Right>(
        module,
        BlasPlan::new(log_d).with_pt_algorithm(BlasPtAlgorithm::Jkls),
        2,
        1,
        1,
        |m, r, c| F::from_f64(public_entry(m, r, c)).unwrap(),
        params.base2k.into(),
        params.prec().into(),
    );
    let p_b = params.prec().log_delta();
    for (label, budget) in [("jkls first map", p_b - 1), ("jkls second map", 2 * p_b - 1)] {
        let short = encrypt_grid::<BE, F, E>(
            &params,
            module,
            host_module,
            &encoder,
            &sk,
            p_b + budget,
            0,
            &Grid::model(log_d, 1, 2, lanes, 14),
            &mut scratch.borrow(),
        );
        let input = grid_snapshot::<BE, _>(&short);
        let mut res = alloc_grid(&params, module, log_d, 1, 1, params.k);
        let bytes = module.blas_matmul_pt_right_tmp_bytes(&res, &short, &jkls, &ctx, &params.atk_layout());
        assert_guard::<BE, _>(
            label,
            if label == "jkls second map" {
                "ckks_eval_linear_transformation_sum_into"
            } else {
                "ckks_eval_linear_transformation_into"
            },
            Budget,
            &mut res,
            |res| module.blas_matmul_pt_right(res, &short, &jkls, &ctx, &keys, &mut poisoned_scratch::<BE>(bytes).borrow()),
        );
        assert!(
            grid_snapshot::<BE, _>(&short) == input,
            "{label}: blas_matmul_pt_right changed a"
        );
    }
    let bytes = module.blas_matmul_pt_right_tmp_bytes(&res, &ct, &jkls, &ctx, &params.atk_layout());
    assert_short_scratch::<BE, _>("blas_matmul_pt_right jkls", &mut res, bytes, |res, s| {
        module.blas_matmul_pt_right(res, &ct, &jkls, &ctx, &keys, s)
    });

    let bytes = module.blas_matmul_pt_right_tmp_bytes(&res, &ct, &b, &ctx, &params.atk_layout());
    assert_short_scratch::<BE, _>("blas_matmul_pt_right", &mut res, bytes, |res, s| {
        module.blas_matmul_pt_right(res, &ct, &b, &ctx, &keys, s)
    });

    // (2^62 + 1) x 4 blocks overflow a usize on 64 bits.
    assert_panics(
        "overflow",
        "the operand's grid has no block or more than a usize counts",
        || {
            module.blas_pt_operand_alloc::<Right>(
                BlasPlan::new(log_d),
                usize::MAX / 4 + 2,
                4,
                lanes,
                params.base2k.into(),
                params.prec().into(),
            )
        },
    );

    if lanes < 2 {
        eprintln!("test_blas_matmul_pt_right_guards skips the relabelled grid: it needs 2 d^2 <= {slots} slots");
        return;
    }
    let one_label = lanes.ilog2() as usize;
    let dense = encrypt_grid::<BE, F, E>(
        &params,
        module,
        host_module,
        &encoder,
        &sk,
        params.k,
        0,
        &Grid::model(log_d, 1, 2, 1, 15),
        &mut scratch.borrow(),
    );
    let relabelled = relabel(&dense, |meta| meta.log_sparsity = one_label);
    let two_label = (lanes / 2).ilog2() as usize;
    for algorithm in PT_ALGORITHMS {
        let pair = pt_operand::<BE, F, Right>(
            module,
            BlasPlan::new(log_d).with_pt_algorithm(algorithm),
            2,
            1,
            2,
            |m, r, c| F::from_f64(public_entry(m, r, c)).unwrap(),
            params.base2k.into(),
            params.prec().into(),
        );
        let mut res = alloc_grid(&params, module, log_d, 1, 1, params.k);
        let bytes = module.blas_matmul_pt_right_tmp_bytes(&res, &relabelled, &pair, &ctx, &params.atk_layout());
        module
            .blas_matmul_pt_right(
                &mut res,
                &relabelled,
                &pair,
                &ctx,
                &keys,
                &mut poisoned_scratch::<BE>(bytes).borrow(),
            )
            .unwrap();
        let label = format!("relabelled one matrix times two, {algorithm:?}");
        assert!(
            res.blocks()
                .iter()
                .all(|block| block.log_sparsity() == two_label.min(one_label)),
            "{label}: label"
        );
        assert!(!replicated::<BE::Ring>(&res), "{label}: replicated");
    }
}

/// `dec(matmul_pt_left(B, enc(A)))` is `B_(s mod count) A` in matrix slot `s`, for a `2 x 3` operand times a `3 x 2`
/// grid with one, two and eight public matrices per matrix slot, of these counts those that divide the matrix slots,
/// at `A`'s scale, and with one public matrix at scales below and above it ([`other_scales`]), under `Diagonal` and
/// `Jkls`, all with the keys of the plan's left set alone, at the metadata [`mapped_layout`] gives for `B`'s scale,
/// twice it under JKLS, its blocks canonical. Each block of `res` is poulpy-ckks's summed linear transformation of its
/// three terms in ascending inner index, under JKLS of the results of `B`'s first map, bit for bit, also at radix 26,
/// where an FFT backend's transform-domain sums round. The diagonal product spends `B`'s scale alone: the same product
/// through transposes, `transpose(blas_matmul_pt_right(transpose(A), B^T))`, with the context's maps at one of these
/// other scales and `B^T` at the other, gives the same values two context scales narrower.
pub fn test_blas_matmul_pt_left<BE, F, E>(
    params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
    log_d: usize,
) where
    BE: TestContextBackend<Ring = Standard> + BlasImpl,
    Module<BE>: TestContextModule<BE>
        + CKKSEncodingOps<BE, F>
        + BlasEncodingOps<BE, F>
        + CKKSLinearTransformationOps<BE>
        + CnvPVecAlloc<BE>,
    F: TestScalar + DiagonalArithmetic,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
{
    let slots = params.n / 2;
    let lanes = slots >> (2 * log_d);
    let ctx = blas_context::<BE, F>(&params, module, log_d);
    let encoder = ReferenceEncoder::<E>::new(slots).unwrap();
    let (m, inner, n) = (2, 3, 2);
    let encode = |params: &CKKSTestParams, algorithm: BlasPtAlgorithm, count: usize| -> BlasPtOperand<BE, Left> {
        pt_operand::<BE, F, Left>(
            module,
            BlasPlan::new(log_d).with_pt_algorithm(algorithm),
            m,
            inner,
            count,
            |m, r, c| F::from_f64(public_entry(m, r, c)).unwrap(),
            params.base2k.into(),
            params.prec().into(),
        )
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
        // At `a`'s scale, then one public matrix at scales below and above it.
        let delta = params.prec().log_delta();
        let cases: Vec<(usize, usize)> = [1, 2, 8]
            .into_iter()
            .filter(|c| lanes.is_multiple_of(*c))
            .map(|count| (count, delta))
            .chain(other_scales(&params).map(|p_b| (1, p_b)))
            .collect();
        for algorithm in PT_ALGORITHMS {
            for &(count, p_b) in &cases {
                let stages = if algorithm == BlasPtAlgorithm::Jkls { 2 } else { 1 };
                let public = at_scale(&params, p_b);
                let b = encode(&public, algorithm, count);
                let label = format!("matmul_pt_left {algorithm:?} count={count} p_B={p_b}");
                let mut res = alloc_grid(&params, module, log_d, m, n, params.k);
                let bytes = module.blas_matmul_pt_left_tmp_bytes(&res, &b, &ct, &ctx, &params.atk_layout());
                let mut op_scratch = poisoned_scratch::<BE>(bytes);
                module
                    .blas_matmul_pt_left(&mut res, &b, &ct, &ctx, &keys, &mut op_scratch.borrow())
                    .unwrap();
                assert_grid_decrypt::<BE, F, E>(
                    &label,
                    &public,
                    module,
                    &encoder,
                    &res,
                    &sk,
                    &a.public_matmul(m, count, public_entry),
                    &mapped_layout(&ct, m, n, params.k, stages * p_b),
                    &mut scratch.borrow(),
                );
                assert!(
                    res.blocks().iter().all(|block| block.is_canonical()),
                    "{label}: a block is not canonical"
                );
                let src = staged(&params, module, &ct, &b, p_b, &keys, &mut scratch.borrow());
                let want = summed_terms(
                    module,
                    alloc_grid(&params, module, log_d, m, n, params.k),
                    inner,
                    pt_label(b.slots(), &ct, b.count()),
                    |i, j, l| (src.block(l, j), b.block(i, l)),
                    &keys,
                    &mut op_scratch.borrow(),
                );
                assert_eq!(grid_snapshot::<BE, _>(&res), want, "{label}: summed terms");
            }
        }

        // `transpose(blas_matmul_pt_right(transpose(A), B^T))`, with `B^T` of `inner x m` blocks on the right, the context's
        // maps at one scale other than `a`'s and `B^T` at the other: the left product spends `B`'s scale, `p_b`, the
        // transposes the context's, `p`, twice more.
        let [p, p_b] = other_scales(&params);
        let ctx = blas_context::<BE, F>(&at_scale(&params, p), module, log_d);
        let route = plan
            .transpose_galois_elements(order)
            .into_iter()
            .chain(plan.matmul_pt_right_galois_elements(order));
        let keys = TestKeys::<BE> {
            atks: rotation_keys(&params, module, &sk_raw, route, &mut scratch.borrow()),
            tsk: None,
        };
        let bt = pt_operand::<BE, F, Right>(
            module,
            BlasPlan::new(log_d),
            inner,
            m,
            1,
            |s, r, c| F::from_f64(public_entry(s, c, r)).unwrap(),
            params.base2k.into(),
            at_scale(&params, p_b).prec().into(),
        );
        let atk = params.atk_layout();
        let (mut at, mut atb, mut res) = (
            alloc_grid(&params, module, log_d, n, inner, params.k),
            alloc_grid(&params, module, log_d, n, m, params.k),
            alloc_grid(&params, module, log_d, m, n, params.k),
        );
        let bytes = module
            .blas_transpose_tmp_bytes(&at, &ct, &ctx, &atk)
            .max(module.blas_matmul_pt_right_tmp_bytes(&atb, &at, &bt, &ctx, &atk))
            .max(module.blas_transpose_tmp_bytes(&res, &atb, &ctx, &atk));
        let mut op_scratch = poisoned_scratch::<BE>(bytes);
        module
            .blas_transpose(&mut at, &ct, &ctx, &keys, &mut op_scratch.borrow())
            .unwrap();
        module
            .blas_matmul_pt_right(&mut atb, &at, &bt, &ctx, &keys, &mut op_scratch.borrow())
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
            &at_scale(&params, p.min(p_b)),
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
        // Room for a JKLS product's two scales above the inputs' at the largest test scale, 80 bits.
        k: 10 * 26,
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
    for algorithm in PT_ALGORITHMS {
        let b = encode(&params, algorithm, 1);
        let mut res = alloc_grid(&params, module, log_d, m, n, params.k);
        let bytes = module.blas_matmul_pt_left_tmp_bytes(&res, &b, &ct, &ctx, &params.atk_layout());
        let mut op_scratch = poisoned_scratch::<BE>(bytes);
        module
            .blas_matmul_pt_left(&mut res, &b, &ct, &ctx, &keys, &mut op_scratch.borrow())
            .unwrap();
        let src = staged(
            &params,
            module,
            &ct,
            &b,
            params.prec().log_delta(),
            &keys,
            &mut scratch.borrow(),
        );
        let want = summed_terms(
            module,
            alloc_grid(&params, module, log_d, m, n, params.k),
            inner,
            pt_label(b.slots(), &ct, b.count()),
            |i, j, l| (src.block(l, j), b.block(i, l)),
            &keys,
            &mut op_scratch.borrow(),
        );
        assert_eq!(
            grid_snapshot::<BE, _>(&res),
            want,
            "matmul_pt_left {algorithm:?} at radix 26: summed terms"
        );
    }
}

/// `blas_matmul_pt_left` of a replicated input, a dense grid of copies of one matrix labelled `log2` of the matrix slots,
/// by a `Diagonal` operand of one public matrix takes the operand's replicated-input block maps
/// ([`BlasPtOperand::block_replicated`]): it decrypts to `B A` with the keys of the plan's left set alone, equals the
/// summed linear transformation of those maps bit for bit and stamps a replicated result. The same operand on a grid of
/// distinct matrices, and operands of two matrices or of one per matrix slot, `Jkls` operands and one matrix slot hold
/// no such maps or do not take them; the first of these equals the summed transformation of the `2d - 1` maps. Skipped,
/// with a message, on one matrix slot.
pub fn test_blas_matmul_pt_left_replicated<BE, F, E>(
    params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
    log_d: usize,
) where
    BE: TestContextBackend<Ring = Standard> + BlasImpl,
    Module<BE>: TestContextModule<BE>
        + CKKSEncodingOps<BE, F>
        + BlasEncodingOps<BE, F>
        + CKKSLinearTransformationOps<BE>
        + CnvPVecAlloc<BE>,
    F: TestScalar + DiagonalArithmetic,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
{
    let slots = params.n / 2;
    let lanes = slots >> (2 * log_d);
    if lanes < 2 {
        eprintln!("test_blas_matmul_pt_left_replicated skips log_d {log_d}: it needs 2 d^2 <= {slots} slots");
        return;
    }
    let ctx = blas_context::<BE, F>(&params, module, log_d);
    let encoder = ReferenceEncoder::<E>::new(slots).unwrap();
    let (m, inner, n) = (2, 3, 2);
    let encode = |algorithm: BlasPtAlgorithm, count: usize| -> BlasPtOperand<BE, Left> {
        pt_operand::<BE, F, Left>(
            module,
            BlasPlan::new(log_d).with_pt_algorithm(algorithm),
            m,
            inner,
            count,
            |m, r, c| F::from_f64(public_entry(m, r, c)).unwrap(),
            params.base2k.into(),
            params.prec().into(),
        )
    };
    let (sk_raw, sk) = gen_sk_with_raw(&params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, module);
    let keys = TestKeys::<BE> {
        atks: rotation_keys(
            &params,
            module,
            &sk_raw,
            BlasPlan::new(log_d).matmul_pt_left_galois_elements(module.cyclotomic_order()),
            &mut scratch.borrow(),
        ),
        tsk: None,
    };
    let one = Grid::model(log_d, inner, n, 1, 20);
    let spread = Grid::from_fn(log_d, inner, n, lanes, |_, r, c| one.at(0, r, c));
    let distinct = Grid::model(log_d, inner, n, lanes, 21);
    let encrypt = |grid: &Grid, log_sparsity: usize, scratch: &mut ScratchArena<'_, BE>| {
        encrypt_grid::<BE, F, E>(
            &params,
            module,
            host_module,
            &encoder,
            &sk,
            params.k,
            log_sparsity,
            grid,
            scratch,
        )
    };
    let ct_one = encrypt(&spread, lanes.ilog2() as usize, &mut scratch.borrow());
    let ct_distinct = encrypt(&distinct, 0, &mut scratch.borrow());
    assert!(replicated::<BE::Ring>(&ct_one), "the fixture holds one matrix");
    assert!(!replicated::<BE::Ring>(&ct_distinct), "the fixture holds several matrices");

    let b = encode(BlasPtAlgorithm::Diagonal, 1);
    assert!(
        (0..m).all(|i| (0..inner).all(|l| b.block_replicated(i, l).is_some())),
        "a Diagonal operand of one matrix on {lanes} matrix slots holds the replicated maps"
    );
    for (label, held) in [
        ("two matrices", encode(BlasPtAlgorithm::Diagonal, 2)),
        ("one matrix per matrix slot", encode(BlasPtAlgorithm::Diagonal, lanes)),
        ("Jkls", encode(BlasPtAlgorithm::Jkls, 1)),
    ] {
        assert!(held.block_replicated(0, 0).is_none(), "{label}: no replicated maps");
    }

    let run = |ct: &BlasCiphertextOwned<BE>, scratch: &mut ScratchArena<'_, BE>| {
        let mut res = alloc_grid(&params, module, log_d, m, n, params.k);
        let bytes = module.blas_matmul_pt_left_tmp_bytes(&res, &b, ct, &ctx, &params.atk_layout());
        module
            .blas_matmul_pt_left(&mut res, &b, ct, &ctx, &keys, &mut poisoned_scratch::<BE>(bytes).borrow())
            .unwrap();
        let want = summed_terms(
            module,
            alloc_grid(&params, module, log_d, m, n, params.k),
            inner,
            pt_label(b.slots(), ct, b.count()),
            |i, j, l| {
                let lt = if replicated::<BE::Ring>(ct) {
                    b.block_replicated(i, l).unwrap()
                } else {
                    b.block(i, l)
                };
                (ct.block(l, j), lt)
            },
            &keys,
            scratch,
        );
        (res, want)
    };
    let p = params.prec().log_delta();
    let (res, want) = run(&ct_one, &mut scratch.borrow());
    assert_grid_decrypt::<BE, F, E>(
        "matmul_pt_left of a replicated input",
        &params,
        module,
        &encoder,
        &res,
        &sk,
        &spread.public_matmul(m, 1, public_entry),
        &mapped_layout(&ct_one, m, n, params.k, p),
        &mut scratch.borrow(),
    );
    assert_eq!(grid_snapshot::<BE, _>(&res), want, "replicated input: summed terms");
    assert!(replicated::<BE::Ring>(&res), "replicated input: the result is replicated");

    let (res, want) = run(&ct_distinct, &mut scratch.borrow());
    assert_grid_decrypt::<BE, F, E>(
        "matmul_pt_left of distinct matrices",
        &params,
        module,
        &encoder,
        &res,
        &sk,
        &distinct.public_matmul(m, 1, public_entry),
        &mapped_layout(&ct_distinct, m, n, params.k, p),
        &mut scratch.borrow(),
    );
    assert_eq!(grid_snapshot::<BE, _>(&res), want, "distinct matrices: summed terms");
}

/// The guards of [`test_blas_matmul_pt_right_guards`] for `B A`, through `blas_matmul_pt_left`: an inner-dimension or `res`
/// shape mismatch, more public matrices than a sparse input records, a copy relabelled past `log2` of the ring degree,
/// an operand encoded on a module of another degree, at another radix or at another dimension, every grid
/// [`rejected_grids`] builds, as `a` or as `res`, a `res` of another degree, radix or rank ([`foreign_layouts`]), a
/// missing rotation key, and `a` and `res` both of another rank than the keys are rejected before `res` changes, as are
/// a `res` narrower than the scale of `a`, and under JKLS an `a` one bit short of `B`'s scale, then of twice it, by
/// poulpy-ckks's budget check, which leaves `a` unchanged too; a product one scratch byte short of its query, under
/// `Diagonal` and `Jkls`, panics before `res` changes; an operand whose block count overflows `usize` panics at its
/// allocation; the relabelled grid of one matrix times two public matrices gives blocks labelled for two, not
/// [`replicated`]. Beside `2 x 2` blocks on fewer than 16 slots, where no other dimension fits, the operand at another
/// dimension is skipped with a message, and on one matrix slot the relabelled grid.
pub fn test_blas_matmul_pt_left_guards<BE, F, E>(
    params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
    log_d: usize,
) where
    BE: TestContextBackend<Ring = Standard> + BlasMatMulImpl + BlasEncodingImpl,
    Module<BE>: TestContextModule<BE>
        + CKKSEncodingOps<BE, F>
        + BlasEncodingOps<BE, F>
        + CKKSLinearTransformationOps<BE>
        + CnvPVecAlloc<BE>,
    F: TestScalar + DiagonalArithmetic,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
{
    let op = "blas_matmul_pt_left";
    let slots = params.n / 2;
    let lanes = slots >> (2 * log_d);
    let ctx = blas_context::<BE, F>(&params, module, log_d);
    let encoder = ReferenceEncoder::<E>::new(slots).unwrap();
    let (sk_raw, sk) = gen_sk_with_raw(&params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, module);
    // `1 x 2` blocks of public matrices: their products take `a` of two block rows.
    let encode = |module: &Module<BE>, log_d: usize, count: usize, base2k: usize| {
        pt_operand::<BE, F, Left>(
            module,
            BlasPlan::new(log_d),
            1,
            2,
            count,
            |m, r, c| F::from_f64(public_entry(m, r, c)).unwrap(),
            base2k.into(),
            params.prec().into(),
        )
    };
    let b = encode(module, log_d, lanes, params.base2k);
    // One public matrix, which divides the matrix slots of any module and of a grid read at any dimension.
    let one = encode(module, log_d, 1, params.base2k);
    let other = other_log_d("test_blas_matmul_pt_left_guards", log_d, slots).map(|other| encode(module, other, 1, params.base2k));
    let double = Module::<BE>::new(2 * params.n as u64);
    let foreign = [
        ("degree", encode(&double, log_d, 1, params.base2k)),
        ("base2k", encode(module, log_d, 1, params.base2k - 1)),
    ];
    // Keys for the other operands too, so no missing key stops `blas_matmul_pt_left` from evaluating them.
    let blocks = [&b, &one]
        .into_iter()
        .chain(&other)
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
        poisoned_scratch::<BE>(module.blas_matmul_pt_left_tmp_bytes(&res_1x2, &b, &square, &ctx, &params.atk_layout()));

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
            module.blas_matmul_pt_left(res, &b, a, &ctx, &keys, &mut op_scratch.borrow())
        });
    }
    for (label, b) in &foreign {
        let mut res = alloc_grid(&params, module, log_d, 1, 1, params.k);
        assert_guard::<BE, _>(label, op, InvalidOperand, &mut res, |res| {
            module.blas_matmul_pt_left(res, b, &ct, &ctx, &keys, &mut op_scratch.borrow())
        });
    }
    // An operand encoded at another dimension, which `ct` and `res` are not read at.
    if let Some(other) = &other {
        let mut res = alloc_grid(&params, module, log_d, 1, 1, params.k);
        assert_guard::<BE, _>("operand dimension", op, InvalidOperand, &mut res, |res| {
            module.blas_matmul_pt_left(res, other, &ct, &ctx, &keys, &mut op_scratch.borrow())
        });
    }

    // Each rejected grid as `a` or as `res` of one public matrix times a 2 x 2 grid.
    for ((label, a), (_, mut bad)) in rejected_grids(module, &square)
        .into_iter()
        .zip(rejected_grids(module, &res_1x2))
    {
        let mut res = res_1x2.clone();
        assert_guard::<BE, _>(&format!("{label} a"), op, InvalidOperand, &mut res, |res| {
            module.blas_matmul_pt_left(res, &one, &a, &ctx, &keys, &mut op_scratch.borrow())
        });
        assert_guard::<BE, _>(&format!("{label} res"), op, InvalidOperand, &mut bad, |res| {
            module.blas_matmul_pt_left(res, &one, &square, &ctx, &keys, &mut op_scratch.borrow())
        });
    }
    for (label, mut bad) in foreign_layouts(module, &res_1x2) {
        assert_guard::<BE, _>(&format!("{label} res"), op, InvalidOperand, &mut bad, |res| {
            module.blas_matmul_pt_left(res, &one, &square, &ctx, &keys, &mut op_scratch.borrow())
        });
    }

    let mut res = alloc_grid(&params, module, log_d, 1, 1, params.k);
    let no_keys = TestKeys::<BE> {
        atks: HashMap::new(),
        tsk: None,
    };
    assert_guard::<BE, _>("no keys", op, MissingKey, &mut res, |res| {
        module.blas_matmul_pt_left(res, &b, &ct, &ctx, &no_keys, &mut op_scratch.borrow())
    });

    // `a` and `res` both of another rank than the keys, which the first key of `b`'s maps rejects.
    let ([.., (_, rank_a)], [.., (_, mut rank_res)]) = (foreign_layouts(module, &ct), foreign_layouts(module, &res));
    assert_guard::<BE, _>("key rank", op, InvalidOperand, &mut rank_res, |res| {
        module.blas_matmul_pt_left(res, &b, &rank_a, &ctx, &keys, &mut op_scratch.borrow())
    });

    // `res` narrower than the scale of `a`: poulpy-ckks rejects the budget of the first block's summed transformation
    // inside the implementation, before any write.
    let input = grid_snapshot::<BE, _>(&ct);
    let mut narrow_res = res_below_scale(&params, module, log_d, ct.block_layout().log_delta());
    assert_guard::<BE, _>(
        "budget",
        "ckks_eval_linear_transformation_sum_into",
        Budget,
        &mut narrow_res,
        |res| module.blas_matmul_pt_left(res, &b, &ct, &ctx, &keys, &mut op_scratch.borrow()),
    );
    assert!(grid_snapshot::<BE, _>(&ct) == input, "budget: blas_matmul_pt_left changed a");

    let jkls = pt_operand::<BE, F, Left>(
        module,
        BlasPlan::new(log_d).with_pt_algorithm(BlasPtAlgorithm::Jkls),
        1,
        2,
        1,
        |m, r, c| F::from_f64(public_entry(m, r, c)).unwrap(),
        params.base2k.into(),
        params.prec().into(),
    );
    let p_b = params.prec().log_delta();
    for (label, budget) in [("jkls first map", p_b - 1), ("jkls second map", 2 * p_b - 1)] {
        let short = encrypt_grid::<BE, F, E>(
            &params,
            module,
            host_module,
            &encoder,
            &sk,
            p_b + budget,
            0,
            &Grid::model(log_d, 2, 1, lanes, 14),
            &mut scratch.borrow(),
        );
        let input = grid_snapshot::<BE, _>(&short);
        let mut res = alloc_grid(&params, module, log_d, 1, 1, params.k);
        let bytes = module.blas_matmul_pt_left_tmp_bytes(&res, &jkls, &short, &ctx, &params.atk_layout());
        assert_guard::<BE, _>(
            label,
            if label == "jkls second map" {
                "ckks_eval_linear_transformation_sum_into"
            } else {
                "ckks_eval_linear_transformation_into"
            },
            Budget,
            &mut res,
            |res| module.blas_matmul_pt_left(res, &jkls, &short, &ctx, &keys, &mut poisoned_scratch::<BE>(bytes).borrow()),
        );
        assert!(
            grid_snapshot::<BE, _>(&short) == input,
            "{label}: blas_matmul_pt_left changed a"
        );
    }
    let bytes = module.blas_matmul_pt_left_tmp_bytes(&res, &jkls, &ct, &ctx, &params.atk_layout());
    assert_short_scratch::<BE, _>("blas_matmul_pt_left jkls", &mut res, bytes, |res, s| {
        module.blas_matmul_pt_left(res, &jkls, &ct, &ctx, &keys, s)
    });

    let bytes = module.blas_matmul_pt_left_tmp_bytes(&res, &b, &ct, &ctx, &params.atk_layout());
    assert_short_scratch::<BE, _>(op, &mut res, bytes, |res, s| {
        module.blas_matmul_pt_left(res, &b, &ct, &ctx, &keys, s)
    });

    // (2^62 + 1) x 4 blocks overflow a usize on 64 bits.
    assert_panics(
        "overflow",
        "the operand's grid has no block or more than a usize counts",
        || {
            module.blas_pt_operand_alloc::<Left>(
                BlasPlan::new(log_d),
                usize::MAX / 4 + 2,
                4,
                lanes,
                params.base2k.into(),
                params.prec().into(),
            )
        },
    );

    if lanes < 2 {
        eprintln!("test_blas_matmul_pt_left_guards skips the relabelled grid: it needs 2 d^2 <= {slots} slots");
        return;
    }
    let one_label = lanes.ilog2() as usize;
    let dense = encrypt_grid::<BE, F, E>(
        &params,
        module,
        host_module,
        &encoder,
        &sk,
        params.k,
        0,
        &Grid::model(log_d, 2, 1, 1, 15),
        &mut scratch.borrow(),
    );
    let relabelled = relabel(&dense, |meta| meta.log_sparsity = one_label);
    let two_label = (lanes / 2).ilog2() as usize;
    for algorithm in PT_ALGORITHMS {
        let pair = pt_operand::<BE, F, Left>(
            module,
            BlasPlan::new(log_d).with_pt_algorithm(algorithm),
            1,
            2,
            2,
            |m, r, c| F::from_f64(public_entry(m, r, c)).unwrap(),
            params.base2k.into(),
            params.prec().into(),
        );
        let mut res = alloc_grid(&params, module, log_d, 1, 1, params.k);
        let bytes = module.blas_matmul_pt_left_tmp_bytes(&res, &pair, &relabelled, &ctx, &params.atk_layout());
        module
            .blas_matmul_pt_left(
                &mut res,
                &pair,
                &relabelled,
                &ctx,
                &keys,
                &mut poisoned_scratch::<BE>(bytes).borrow(),
            )
            .unwrap();
        let label = format!("relabelled one matrix times two, {algorithm:?}");
        assert!(
            res.blocks()
                .iter()
                .all(|block| block.log_sparsity() == two_label.min(one_label)),
            "{label}: label"
        );
        assert!(!replicated::<BE::Ring>(&res), "{label}: replicated");
    }
}

/// What both masked tests share, at `params.prec()`: the encoder, the secret, a scratch, a context holding the masks,
/// each masked path's plan, and one store of the keys of their sets on both sides alone.
struct MaskedFixture<BE: Backend, E> {
    encoder: ReferenceEncoder<E>,
    sk_raw: BackendGLWESecret<BE>,
    sk: GLWESecretPrepared<BE::OwnedBuf, BE>,
    scratch: ScratchOwned<BE>,
    ctx: BlasContext<BE>,
    plans: [BlasPlan; 2],
    keys: TestKeys<BE>,
}

/// The [`MaskedFixture`] of `2^log_d x 2^log_d` blocks on `module`.
fn masked_fixture<BE, F, E>(
    params: &CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
    log_d: usize,
) -> MaskedFixture<BE, E>
where
    BE: TestContextBackend<Ring = Standard> + BlasEncodingImpl,
    Module<BE>: TestContextModule<BE> + CKKSEncodingOps<BE, F> + BlasEncodingOps<BE, F> + CKKSLinearTransformationOps<BE>,
    F: TestScalar + DiagonalArithmetic,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
{
    let (sk_raw, sk) = gen_sk_with_raw(params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(params, module);
    let (base2k, meta, order) = (params.base2k.into(), params.prec().into(), module.cyclotomic_order());
    let masked = BlasPlan::new(log_d).with_pt_algorithm(BlasPtAlgorithm::JklsGalois);
    let ctx = encoded_context::<BE, F>(module, masked, base2k, meta);
    let plans = MASKED_PT_ALGORITHMS.map(|algorithm| BlasPlan::new(log_d).with_pt_algorithm(algorithm));
    let elements: BTreeSet<i64> = plans
        .iter()
        .flat_map(|plan| {
            [
                plan.matmul_pt_right_galois_elements(order),
                plan.matmul_pt_left_galois_elements(order),
            ]
        })
        .flatten()
        .collect();
    let atks = rotation_keys(params, module, &sk_raw, elements, &mut scratch.borrow());
    MaskedFixture {
        encoder: ReferenceEncoder::<E>::new(params.n / 2).unwrap(),
        sk_raw,
        sk,
        scratch,
        ctx,
        plans,
        keys: TestKeys { atks, tsk: None },
    }
}

/// Asserts `res`, written by a masked product under `keys`, decrypts to `want` at `layout`, which the product's
/// output-layout query gave as `queried`, every block canonical, and that `call` under a store holding only the keys its
/// key-use query listed, `uses`, writes it into a fresh grid bit for bit.
fn assert_masked_product<BE, F, E>(
    label: &str,
    params: &CKKSTestParams,
    module: &Module<BE>,
    encoder: &ReferenceEncoder<E>,
    sk: &GLWESecretPrepared<BE::OwnedBuf, BE>,
    keys: &TestKeys<BE>,
    res: &BlasCiphertextOwned<BE>,
    want: &Grid,
    (layout, queried, uses): (BlasLayout, BlasLayout, BlasKeyUses),
    call: impl FnOnce(&mut BlasCiphertextOwned<BE>, &StrictKeys<'_, BE>) -> CKKSResult<()>,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: TestContextBackend<Ring = Standard>,
    Module<BE>: TestContextModule<BE> + CKKSEncodingOps<BE, F>,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
{
    assert_eq!(queried, layout, "{label}: output-layout query");
    assert_grid_decrypt::<BE, F, E>(label, params, module, encoder, res, sk, want, &layout, scratch);
    assert!(
        res.blocks().iter().all(|block| block.is_canonical()),
        "{label}: a block is not canonical"
    );
    let strict = keys.at_widths(uses.rotations.iter().map(|&(p, k)| (p, k.as_usize())), std::iter::empty());
    let mut again = alloc_grid(params, module, res.log_d(), res.rows(), res.cols(), params.k);
    call(&mut again, &strict).unwrap();
    assert!(
        grid_snapshot::<BE, _>(&again) == grid_snapshot::<BE, _>(res),
        "{label}: the result under the queried keys differs"
    );
}

/// The masked paths, `JklsStored` and `JklsGalois`, under a context holding their masks: `dec(matmul_pt_right(enc(A), B))`
/// is `A B` for a `2 x 3` grid times a `3 x 2` operand of one public matrix, and `dec(matmul_pt_left(B, enc(A)))` is
/// `B_(s mod count) A` for a `2 x 3` operand times a `3 x 2` grid, for one, two and every public matrix per matrix slot,
/// of these counts those that divide the matrix slots, all at `A`'s scale, with the keys of each plan's sets alone.
/// A prepared operand holds its first map and its terms as plaintexts, `d` (`2d - 1` on the left) per block under
/// `JklsStored`, one under `JklsGalois`. Each result is canonical, at the metadata of three scales, `2 p_B + p`
/// ([`mapped_layout`]), which its output-layout query gives, and the same call under a store holding only the keys its
/// key-use query lists gives it bit for bit.
pub fn test_blas_matmul_pt_masked<BE, F, E>(
    params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
    log_d: usize,
) where
    BE: TestContextBackend<Ring = Standard> + BlasMatMulImpl + BlasEncodingImpl,
    Module<BE>: TestContextModule<BE>
        + CKKSEncodingOps<BE, F>
        + BlasEncodingOps<BE, F>
        + CKKSLinearTransformationOps<BE>
        + CnvPVecAlloc<BE>,
    F: TestScalar + DiagonalArithmetic,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
{
    let lanes = (params.n / 2) >> (2 * log_d);
    let d = 1usize << log_d;
    let (m, inner, n) = (2, 3, 2);
    let MaskedFixture {
        encoder,
        sk,
        mut scratch,
        ctx,
        plans,
        keys,
        ..
    } = masked_fixture::<BE, F, E>(&params, module, host_module, log_d);
    let (base2k, meta, delta) = (params.base2k.into(), params.prec().into(), params.prec().log_delta());
    let (a, a_left) = (
        Grid::model(log_d, m, inner, lanes, 9),
        Grid::model(log_d, inner, n, lanes, 11),
    );
    let encrypt = |grid: &Grid, scratch: &mut ScratchArena<'_, BE>| {
        encrypt_grid::<BE, F, E>(&params, module, host_module, &encoder, &sk, params.k, 0, grid, scratch)
    };
    let (ct, ct_left) = (encrypt(&a, &mut scratch.borrow()), encrypt(&a_left, &mut scratch.borrow()));
    let mut counts: Vec<usize> = [1, 2, lanes].into_iter().filter(|c| lanes.is_multiple_of(*c)).collect();
    counts.dedup();
    let atk = params.atk_layout();
    let entry = |m: usize, r: usize, c: usize| F::from_f64(public_entry(m, r, c)).unwrap();
    for plan in &plans {
        let algorithm = plan.pt_algorithm();
        let (right_terms, left_terms) = match algorithm {
            BlasPtAlgorithm::JklsStored => (d, 2 * d - 1),
            _ => (1, 1),
        };

        let b = pt_operand::<BE, F, Right>(module, *plan, inner, n, 1, entry, base2k, meta);
        let label = format!("matmul_pt_right {algorithm:?}");
        assert_eq!(
            (b.maps().count(), b.block_plaintexts(inner - 1, n - 1).len()),
            (1, right_terms),
            "{label}: prepared terms"
        );
        let mut res = alloc_grid(&params, module, log_d, m, n, params.k);
        let bytes = module.blas_matmul_pt_right_tmp_bytes(&res, &ct, &b, &ctx, &atk);
        let layouts = (
            mapped_layout(&ct, m, n, params.k, 3 * delta),
            blas_matmul_pt_right_output_layout(&res, &ct, &b, &ctx),
            blas_matmul_pt_right_key_uses(module, &res, &ct, &b, &ctx),
        );
        module
            .blas_matmul_pt_right(&mut res, &ct, &b, &ctx, &keys, &mut poisoned_scratch::<BE>(bytes).borrow())
            .unwrap();
        assert_masked_product::<BE, F, E>(
            &label,
            &params,
            module,
            &encoder,
            &sk,
            &keys,
            &res,
            &a.matmul_public(n, 1, public_entry),
            layouts,
            |res, keys| module.blas_matmul_pt_right(res, &ct, &b, &ctx, keys, &mut poisoned_scratch::<BE>(bytes).borrow()),
            &mut scratch.borrow(),
        );

        for &count in &counts {
            let b = pt_operand::<BE, F, Left>(module, *plan, m, inner, count, entry, base2k, meta);
            let label = format!("matmul_pt_left {algorithm:?} count={count}");
            assert_eq!(
                (b.maps().count(), b.block_plaintexts(m - 1, inner - 1).len()),
                (1, left_terms),
                "{label}: prepared terms"
            );
            let mut res = alloc_grid(&params, module, log_d, m, n, params.k);
            let bytes = module.blas_matmul_pt_left_tmp_bytes(&res, &b, &ct_left, &ctx, &atk);
            let layouts = (
                mapped_layout(&ct_left, m, n, params.k, 3 * delta),
                blas_matmul_pt_left_output_layout(&res, &b, &ct_left, &ctx),
                blas_matmul_pt_left_key_uses(module, &res, &b, &ct_left, &ctx),
            );
            module
                .blas_matmul_pt_left(
                    &mut res,
                    &b,
                    &ct_left,
                    &ctx,
                    &keys,
                    &mut poisoned_scratch::<BE>(bytes).borrow(),
                )
                .unwrap();
            assert_masked_product::<BE, F, E>(
                &label,
                &params,
                module,
                &encoder,
                &sk,
                &keys,
                &res,
                &a_left.public_matmul(m, count, public_entry),
                layouts,
                |res, keys| {
                    module.blas_matmul_pt_left(res, &b, &ct_left, &ctx, keys, &mut poisoned_scratch::<BE>(bytes).borrow())
                },
                &mut scratch.borrow(),
            );
        }
    }
}

/// The masked paths reject, before `res` changes: on the right, an operand of two public matrices on more than one
/// matrix slot, whose rotations by multiples of `d` would read the next matrix slot; on either side, a context of masks
/// encoded on a module of another degree, at another radix, at another dimension or wider than the input; on the left,
/// a context whose plan holds no column halves of psi^k, whose scratch query panics with a static message. Each runs on
/// exactly its scratch query: one byte short panics before `res` changes. On one matrix slot, which holds one public
/// matrix, the right count guard is skipped with a message, as is the context at another dimension where none fits.
pub fn test_blas_matmul_pt_masked_guards<BE, F, E>(
    params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
    log_d: usize,
) where
    BE: TestContextBackend<Ring = Standard> + BlasMatMulImpl + BlasEncodingImpl,
    Module<BE>: TestContextModule<BE>
        + CKKSEncodingOps<BE, F>
        + BlasEncodingOps<BE, F>
        + CKKSLinearTransformationOps<BE>
        + CnvPVecAlloc<BE>,
    F: TestScalar + DiagonalArithmetic,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
{
    let slots = params.n / 2;
    let lanes = slots >> (2 * log_d);
    let MaskedFixture {
        encoder,
        sk_raw,
        sk,
        mut scratch,
        ctx,
        plans,
        mut keys,
    } = masked_fixture::<BE, F, E>(&params, module, host_module, log_d);
    let (base2k, meta, atk) = (params.base2k.into(), params.prec().into(), params.atk_layout());
    let plain = blas_context::<BE, F>(&params, module, log_d);
    // Every foreign context holds the masks, so its mismatch alone rejects it on either side.
    let masked = ctx.plan();
    let mut foreign: Vec<_> =
        foreign_contexts::<BE, F>(&params, module, masked, &sk_raw, &mut keys.atks, &mut scratch.borrow()).into();
    if let Some(other) = other_log_d("test_blas_matmul_pt_masked_guards", log_d, slots) {
        let plan = BlasPlan::new(other).with_pt_algorithm(masked.pt_algorithm());
        foreign.push(("dimension", encoded_context::<BE, F>(module, plan, base2k, meta)));
    }
    let wide = CoeffsMeta {
        k: (params.k + 1).into(),
        ..params.prec().into()
    };
    foreign.push(("width", encoded_context::<BE, F>(module, masked, base2k, wide)));
    let a = Grid::model(log_d, 1, 1, lanes, 3);
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
    let entry = |m: usize, r: usize, c: usize| F::from_f64(public_entry(m, r, c)).unwrap();
    if lanes == 1 {
        eprintln!("test_blas_matmul_pt_masked_guards: one matrix slot holds one public matrix, the right count guard is skipped");
    }
    for plan in &plans {
        let algorithm = plan.pt_algorithm();
        let mut res = alloc_grid(&params, module, log_d, 1, 1, params.k);
        if lanes > 1 {
            let two = pt_operand::<BE, F, Right>(module, *plan, 1, 1, 2, entry, base2k, meta);
            let bytes = module.blas_matmul_pt_right_tmp_bytes(&res, &ct, &two, &ctx, &atk);
            assert_guard::<BE, _>(
                &format!("{algorithm:?} on the right of two public matrices"),
                "blas_matmul_pt_right",
                InvalidOperand,
                &mut res,
                |res| module.blas_matmul_pt_right(res, &ct, &two, &ctx, &keys, &mut poisoned_scratch::<BE>(bytes).borrow()),
            );
        }
        let left = pt_operand::<BE, F, Left>(module, *plan, 1, 1, 1, entry, base2k, meta);
        let bytes = module.blas_matmul_pt_left_tmp_bytes(&res, &left, &ct, &ctx, &atk);
        assert_guard::<BE, _>(
            &format!("{algorithm:?} on the left under a context without masks"),
            "blas_matmul_pt_left",
            InvalidOperand,
            &mut res,
            |res| module.blas_matmul_pt_left(res, &left, &ct, &plain, &keys, &mut poisoned_scratch::<BE>(bytes).borrow()),
        );
        assert_panics(
            &format!("{algorithm:?}: the scratch query under a context without masks"),
            MASKLESS,
            || module.blas_matmul_pt_left_tmp_bytes(&res, &left, &ct, &plain, &atk),
        );
        assert_short_scratch::<BE, _>("blas_matmul_pt_left", &mut res, bytes, |res, s| {
            module.blas_matmul_pt_left(res, &left, &ct, &ctx, &keys, s)
        });
        let right = pt_operand::<BE, F, Right>(module, *plan, 1, 1, 1, entry, base2k, meta);
        let right_bytes = module.blas_matmul_pt_right_tmp_bytes(&res, &ct, &right, &ctx, &atk);
        for (label, foreign) in &foreign {
            assert_guard::<BE, _>(
                &format!("{algorithm:?} on the right under a context of another {label}"),
                "blas_matmul_pt_right",
                InvalidOperand,
                &mut res,
                |res| {
                    module.blas_matmul_pt_right(
                        res,
                        &ct,
                        &right,
                        foreign,
                        &keys,
                        &mut poisoned_scratch::<BE>(right_bytes).borrow(),
                    )
                },
            );
            assert_guard::<BE, _>(
                &format!("{algorithm:?} on the left under a context of another {label}"),
                "blas_matmul_pt_left",
                InvalidOperand,
                &mut res,
                |res| module.blas_matmul_pt_left(res, &left, &ct, foreign, &keys, &mut poisoned_scratch::<BE>(bytes).borrow()),
            );
        }
        assert_short_scratch::<BE, _>("blas_matmul_pt_right", &mut res, right_bytes, |res, s| {
            module.blas_matmul_pt_right(res, &ct, &right, &ctx, &keys, s)
        });
    }
}

/// Rejected before `res` changes, by the product and, of these cases, by those that meet them, the preparations under
/// either hint and the prepared product of every mix of hot and cold operands: every grid [`rejected_grids`] builds, in
/// place of a factor or of `res`, or as the input of a preparation; a `res` shape mismatch; a `res`, one factor or one
/// prepared operand of another degree, radix or rank, and an input of a preparation of another degree, radix or rank
/// ([`foreign_layouts`]), whose rank its keys reject; factors and `res` all of another rank than the rotation keys; a
/// copy relabelled past `log2` of the ring degree; a context encoded on a module of another degree or at another radix,
/// or at another dimension, which the prepared product of a hot operand rejects too, and that of two cold operands
/// ignores; a factor at a radix other than the context's; an inner-dimension mismatch; a radix above 62, where two
/// canonical tensor digits can overflow an `i64`; a missing rotation or tensor key; and a tensor key
/// of another rank than the blocks. The prepared product's validator also rejects operand infos of either hint without
/// a batch record, at another dimension, without a block, or whose label does not allow their record, and operands and
/// `res` all of another rank than the keys. poulpy-ckks's budget check rejects a `res` narrower than the factors' scale
/// in the product and in every mix of the prepared product before `res` or a factor changes, and an input one bit short
/// of what its maps spend, `2p`, or `p` for a right input holding one matrix, in a cold preparation, as a factor of the
/// product, and in the prepared product of its hot operand, which its preparation accepts, before it or `res` changes,
/// while a cold preparation accepts a right input holding one matrix at a budget in `[p, 2p)`; of two short factors,
/// the left one's error, whichever side the product holds. The preparations take no tensor key, the prepared product of
/// cold operands no rotation key, and that of a hot one the rotation keys of its terms. Each of these operations, under
/// every hint and mix, runs on exactly the scratch its query reports, and one byte short panics before `res` changes.
pub fn test_blas_matmul_guards<BE, F, E>(
    params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
    log_d: usize,
) where
    BE: TestContextBackend<Ring = Standard> + BlasMatMulImpl + BlasEncodingImpl,
    Module<BE>: TestContextModule<BE>
        + CKKSEncodingOps<BE, F>
        + BlasEncodingOps<BE, F>
        + CKKSLinearTransformationOps<BE>
        + CnvPVecAlloc<BE>,
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
    let mut foreign: Vec<_> = foreign_contexts::<BE, F>(
        &params,
        module,
        BlasPlan::new(log_d),
        &sk_raw,
        &mut atks,
        &mut scratch.borrow(),
    )
    .into();
    if let Some(other) = other_log_d("test_blas_matmul_guards", log_d, slots) {
        foreign.push(("dimension", blas_context::<BE, F>(&params, module, other)));
    }
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
    // `column ct`, the widest product below, which covers both preparations under either hint.
    let bytes = module.blas_matmul_tmp_bytes(&alloc(2, 2), &column, &ct, &ctx, &tsk_layout, &atk_layout);
    let mut op_scratch = poisoned_scratch::<BE>(bytes);

    // `ct` is 1 x 2 and `column` 2 x 1: `ct ct` has a mismatched inner dimension, `ct column` is 1 x 1, not 1 x 2.
    for (label, mut res, b) in [("inner", alloc(1, 2), &ct), ("shape", alloc(1, 2), &column)] {
        assert_guard::<BE, _>(label, "blas_matmul", InvalidOperand, &mut res, |res| {
            module.blas_matmul(res, &ct, b, &ctx, &keys, &keys, &mut op_scratch.borrow())
        });
    }
    // Both preparations of `ct` and `column`, by hint: hot at `side(OneShot)`, cold at `side(Reuse)`.
    let side = |hint: PrepareHint| usize::from(hint == Reuse);
    let left = HINTS.map(|hint| matmul_left_operand(module, &ct, &ctx, &keys, hint, &mut op_scratch.borrow()).unwrap());
    let left_column =
        HINTS.map(|hint| matmul_left_operand(module, &column, &ctx, &keys, hint, &mut op_scratch.borrow()).unwrap());
    let right = HINTS.map(|hint| matmul_right_operand(module, &column, &ctx, &keys, hint, &mut op_scratch.borrow()).unwrap());
    let row_right = HINTS.map(|hint| matmul_right_operand(module, &ct, &ctx, &keys, hint, &mut op_scratch.borrow()).unwrap());
    let mut wide = alloc(1, 2);
    for (hl, hr) in MIXES {
        for (label, right) in [("inner", &row_right[side(hr)]), ("shape", &right[side(hr)])] {
            let label = &format!("{label} {hl:?} x {hr:?}");
            assert_guard::<BE, _>(label, "blas_matmul_prepared", InvalidOperand, &mut wide, |res| {
                module.blas_matmul_prepared(res, &left[side(hl)], right, &ctx, &keys, &keys, &mut op_scratch.borrow())
            });
        }
    }

    // Each rejected grid as a factor of `column ct`, a 2 x 2 product, and as the input of either preparation.
    for (label, bad) in rejected_grids(module, &column) {
        let label = &format!("{label} column");
        let mut res = alloc(2, 2);
        assert_guard::<BE, _>(label, "blas_matmul", InvalidOperand, &mut res, |res| {
            module.blas_matmul(res, &bad, &ct, &ctx, &keys, &keys, &mut op_scratch.borrow())
        });
        for hint in HINTS {
            let label = &format!("{label} {hint:?}");
            let left = matmul_left_operand(module, &bad, &ctx, &keys, hint, &mut op_scratch.borrow());
            assert_rejected(label, "blas_matmul_prepare_left", InvalidOperand, left);
            let right = matmul_right_operand(module, &bad, &ctx, &keys, hint, &mut op_scratch.borrow());
            assert_rejected(label, "blas_matmul_prepare_right", InvalidOperand, right);
        }
    }
    for (label, bad) in rejected_grids(module, &ct) {
        let label = &format!("{label} row");
        let mut res = alloc(2, 2);
        assert_guard::<BE, _>(label, "blas_matmul", InvalidOperand, &mut res, |res| {
            module.blas_matmul(res, &column, &bad, &ctx, &keys, &keys, &mut op_scratch.borrow())
        });
    }
    // Each rejected grid, and each grid of another degree, radix or rank, as `res` of `column ct` and of its prepared
    // form, of every mix.
    let bad_res = rejected_grids(module, &alloc(2, 2))
        .into_iter()
        .chain(foreign_layouts(module, &alloc(2, 2)));
    for (label, mut bad) in bad_res {
        let label = &format!("{label} res");
        assert_guard::<BE, _>(label, "blas_matmul", InvalidOperand, &mut bad, |res| {
            module.blas_matmul(res, &column, &ct, &ctx, &keys, &keys, &mut op_scratch.borrow())
        });
        for (hl, hr) in MIXES {
            let (l, r) = (&left_column[side(hl)], &row_right[side(hr)]);
            assert_guard::<BE, _>(
                &format!("{label} {hl:?} x {hr:?}"),
                "blas_matmul_prepared",
                InvalidOperand,
                &mut bad,
                |res| module.blas_matmul_prepared(res, l, r, &ctx, &keys, &keys, &mut op_scratch.borrow()),
            );
        }
    }
    // Each grid of another degree, radix or rank as one factor of `column ct` beside a valid one, and, read through its
    // infos, as one prepared operand of either hint: each factor is compared with `res` on its own.
    for ((label, bad_column), (_, bad_row)) in foreign_layouts(module, &column).into_iter().zip(foreign_layouts(module, &ct)) {
        for (side, a, b) in [("a", &bad_column, &ct), ("b", &column, &bad_row)] {
            let label = &format!("{label} {side}");
            let mut res = alloc(2, 2);
            assert_guard::<BE, _>(label, "blas_matmul", InvalidOperand, &mut res, |res| {
                module.blas_matmul(res, a, b, &ctx, &keys, &keys, &mut op_scratch.borrow())
            });
        }
        let res = alloc(2, 2);
        for hint in HINTS {
            let a = OperandInfos(&bad_column, hint);
            let infos = validate_blas_matmul_prepared(module, &res, &a, &row_right[side(Reuse)], &ctx, &keys, &keys);
            assert_rejected(
                &format!("{label} a {hint:?} infos"),
                "blas_matmul_prepared",
                InvalidOperand,
                infos,
            );
            let b = OperandInfos(&bad_row, hint);
            let infos = validate_blas_matmul_prepared(module, &res, &left_column[side(Reuse)], &b, &ctx, &keys, &keys);
            assert_rejected(
                &format!("{label} b {hint:?} infos"),
                "blas_matmul_prepared",
                InvalidOperand,
                infos,
            );
        }
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
        for hint in HINTS {
            let label = &format!("{label} {hint:?}");
            let left = matmul_left_operand(module, &sparse_row, &ctx, &keys, hint, &mut op_scratch.borrow());
            assert_rejected(label, "blas_matmul_prepare_left", InvalidOperand, left);
            let right = matmul_right_operand(module, &sparse_column, &ctx, &keys, hint, &mut op_scratch.borrow());
            assert_rejected(label, "blas_matmul_prepare_right", InvalidOperand, right);
        }
    }

    // Contexts of another degree, radix or dimension. The prepared product evaluates a hot side's terms under its
    // context, which must read them; two cold operands bring their terms and read none.
    for (label, ctx) in &foreign {
        let mut res = alloc(1, 1);
        assert_guard::<BE, _>(label, "blas_matmul", InvalidOperand, &mut res, |res| {
            module.blas_matmul(res, &ct, &column, ctx, &keys, &keys, &mut op_scratch.borrow())
        });
        for hint in HINTS {
            let label = &format!("{label} {hint:?}");
            let left = matmul_left_operand(module, &ct, ctx, &keys, hint, &mut op_scratch.borrow());
            assert_rejected(label, "blas_matmul_prepare_left", InvalidOperand, left);
            let right = matmul_right_operand(module, &column, ctx, &keys, hint, &mut op_scratch.borrow());
            assert_rejected(label, "blas_matmul_prepare_right", InvalidOperand, right);
        }
        for (hl, hr) in MIXES {
            let (l, r) = (&left[side(hl)], &right[side(hr)]);
            let mut call = |res: &mut BlasCiphertextOwned<BE>| {
                module.blas_matmul_prepared(res, l, r, ctx, &keys, &keys, &mut op_scratch.borrow())
            };
            let label = &format!("{label} context {hl:?} x {hr:?}");
            if (hl, hr) == (Reuse, Reuse) {
                call(&mut res).unwrap_or_else(|e| panic!("{label}: {e}"));
            } else {
                assert_guard::<BE, _>(label, "blas_matmul_prepared", InvalidOperand, &mut res, call);
            }
        }
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
    for hint in HINTS {
        let label = &format!("radix {hint:?}");
        let left_radix = matmul_left_operand(module, &radix(1, 2), &ctx, &keys, hint, &mut op_scratch.borrow());
        assert_rejected(label, "blas_matmul_prepare_left", InvalidOperand, left_radix);
        let right_radix = matmul_right_operand(module, &radix(2, 1), &ctx, &keys, hint, &mut op_scratch.borrow());
        assert_rejected(label, "blas_matmul_prepare_right", InvalidOperand, right_radix);
        // A grid of another degree, radix or rank, alone: no `res` to compare it with, so the rank is compared with the
        // keys'.
        for (label, bad) in foreign_layouts(module, &ct) {
            let label = &format!("{label} {hint:?}");
            let left = matmul_left_operand(module, &bad, &ctx, &keys, hint, &mut op_scratch.borrow());
            assert_rejected(label, "blas_matmul_prepare_left", InvalidOperand, left);
            let right = matmul_right_operand(module, &bad, &ctx, &keys, hint, &mut op_scratch.borrow());
            assert_rejected(label, "blas_matmul_prepare_right", InvalidOperand, right);
        }
    }

    // `res` at radix 63, above poulpy-ckks's prepared dot product's 62; the radix check, which `validate.rs` pins, comes
    // before the comparison with the factors' radix.
    let radix_63 = CKKSTestParams { base2k: 63, ..params };
    let mut res = alloc_grid(&radix_63, module, log_d, 1, 1, params.k);
    assert_guard::<BE, _>("radix 63", "blas_matmul", InvalidOperand, &mut res, |res| {
        module.blas_matmul(res, &ct, &column, &ctx, &keys, &keys, &mut op_scratch.borrow())
    });
    for (hl, hr) in MIXES {
        let (l, r) = (&left[side(hl)], &right[side(hr)]);
        let label = &format!("radix 63 {hl:?} x {hr:?}");
        assert_guard::<BE, _>(label, "blas_matmul_prepared", InvalidOperand, &mut res, |res| {
            module.blas_matmul_prepared(res, l, r, &ctx, &keys, &keys, &mut op_scratch.borrow())
        });
    }

    // Operand infos the prepared product's validator rejects, beside a valid operand, under either hint: without a
    // batch record (the layouts `blas_matmul_left_operand_layout` and `blas_matmul_right_operand_layout` give), at
    // another dimension, with a label that does not allow the record or a scale wider than the blocks (the recorded
    // grids `rejected_grids` builds, read through their infos; an uneven grid reports only its first block's layout),
    // and without a block.
    let op = "blas_matmul_prepared";
    let res = alloc(1, 1);
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
    let (cold_left, cold_right) = (&left[side(Reuse)], &right[side(Reuse)]);
    for hint in HINTS {
        let unrecorded = blas_matmul_left_operand_layout(&ct, &ctx, hint);
        let infos = validate_blas_matmul_prepared(module, &res, &unrecorded, cold_right, &ctx, &keys, &keys);
        assert_rejected(&format!("no record a {hint:?}"), op, InvalidOperand, infos);
        let unrecorded = blas_matmul_right_operand_layout(&column, &ctx, hint);
        let infos = validate_blas_matmul_prepared(module, &res, cold_left, &unrecorded, &ctx, &keys, &keys);
        assert_rejected(&format!("no record b {hint:?}"), op, InvalidOperand, infos);
        for (label, bad) in rejected_grids(module, &column)
            .into_iter()
            .filter(|(label, _)| *label != "uneven")
        {
            let infos = validate_blas_matmul_prepared(module, &res, cold_left, &OperandInfos(&bad, hint), &ctx, &keys, &keys);
            assert_rejected(&format!("{label} {hint:?} infos"), op, InvalidOperand, infos);
        }
        let (a, b) = (OperandInfos(&no_cols, hint), OperandInfos(&no_rows, hint));
        let empty = validate_blas_matmul_prepared(module, &res, &a, &b, &ctx, &keys, &keys);
        assert_rejected(&format!("no block {hint:?}"), op, InvalidOperand, empty);
    }

    // Factors and `res` all one rank above the keys, beside a tensor key of their rank, so that only the rotation keys'
    // rank differs: the product meets it at its first rotation key. Read as operand infos beside the keys' tensor key,
    // they meet the prepared product's validator at the first key it looks up: of a hot side's terms, or the tensor key.
    // Then that tensor key beside valid factors and `res`: both products meet it, the product past its rotation keys,
    // the prepared product past those of its hot sides. No case uses the tensor key, so an allocated key serves.
    let rank_tsk = module.alloc_tensor_key_prepared_from_infos(&GLWETensorKeyLayout {
        rank: Rank(params.rank as u32 + 1),
        ..params.tsk_layout()
    });
    let ([.., (_, rank_row)], [.., (_, rank_column)]) = (foreign_layouts(module, &ct), foreign_layouts(module, &column));
    let [.., (_, mut rank_res)] = foreign_layouts(module, &res);
    assert_guard::<BE, _>("key rank", "blas_matmul", InvalidOperand, &mut rank_res, |res| {
        module.blas_matmul(res, &rank_row, &rank_column, &ctx, &rank_tsk, &keys, &mut op_scratch.borrow())
    });
    for (hl, hr) in MIXES {
        let (a, b) = (OperandInfos(&rank_row, hl), OperandInfos(&rank_column, hr));
        let infos = validate_blas_matmul_prepared(module, &rank_res, &a, &b, &ctx, &keys, &keys);
        assert_rejected(&format!("key rank {hl:?} x {hr:?} infos"), op, InvalidOperand, infos);
    }
    let mut res = alloc(1, 1);
    assert_guard::<BE, _>("tensor key rank", "blas_matmul", InvalidOperand, &mut res, |res| {
        module.blas_matmul(res, &ct, &column, &ctx, &rank_tsk, &keys, &mut op_scratch.borrow())
    });
    for (hl, hr) in MIXES {
        let (l, r) = (&left[side(hl)], &right[side(hr)]);
        assert_guard::<BE, _>(
            &format!("tensor key rank {hl:?} x {hr:?}"),
            op,
            InvalidOperand,
            &mut res,
            |res| module.blas_matmul_prepared(res, l, r, &ctx, &rank_tsk, &keys, &mut op_scratch.borrow()),
        );
    }

    // `res` narrower than the factors' scale: poulpy-ckks rejects the budget of the first block's JKLS kernel inside the
    // implementation, after the terms of the hot sides, before any write.
    let factors = [&ct, &column].map(grid_snapshot::<BE, _>);
    let infos = || (left.each_ref().map(operand_infos), right.each_ref().map(operand_infos));
    let operands = infos();
    let mut narrow_res = res_below_scale(&params, module, log_d, ct.block_layout().log_delta());
    assert_guard::<BE, _>("budget", "ckks_dot_product_prepared", Budget, &mut narrow_res, |res| {
        module.blas_matmul(res, &ct, &column, &ctx, &keys, &keys, &mut op_scratch.borrow())
    });
    for (hl, hr) in MIXES {
        let (l, r) = (&left[side(hl)], &right[side(hr)]);
        assert_guard::<BE, _>(
            &format!("budget {hl:?} x {hr:?}"),
            "ckks_dot_product_prepared",
            Budget,
            &mut narrow_res,
            |res| module.blas_matmul_prepared(res, l, r, &ctx, &keys, &keys, &mut op_scratch.borrow()),
        );
    }
    assert!(
        [&ct, &column].map(grid_snapshot::<BE, _>) == factors,
        "budget: blas_matmul changed a factor"
    );
    // The prepared operands are opaque: their infos are what a test reads.
    assert!(infos() == operands, "budget: blas_matmul_prepared changed an operand");
    // One bit short of what the terms spend, `2p` or `p` on a replicated right input: the overdrawing map rejects it.
    let p = params.prec().log_delta();
    let mut short_by = |spend: usize| {
        encrypt_grid::<BE, F, E>(
            &params,
            module,
            host_module,
            &encoder,
            &sk,
            ct.block_layout().log_delta() + spend - 1,
            0,
            &Grid::model(log_d, 1, 2, lanes, 10),
            &mut scratch.borrow(),
        )
    };
    let short = short_by(2 * p);
    let mut short_right = short.clone();
    if replicated::<BE::Ring>(&short_right) {
        // Its terms spend `p`, so a budget in `[p, 2p)` passes.
        matmul_right_operand(module, &short_right, &ctx, &keys, Reuse, &mut op_scratch.borrow()).unwrap();
        short_right = short_by(p);
    }
    let inputs = [&short, &short_right].map(grid_snapshot::<BE, _>);
    let left_short = matmul_left_operand(module, &short, &ctx, &keys, Reuse, &mut op_scratch.borrow());
    assert_rejected("budget", "ckks_eval_linear_transformation_into", Budget, left_short);
    let right_short = matmul_right_operand(module, &short_right, &ctx, &keys, Reuse, &mut op_scratch.borrow());
    assert_rejected("budget", "ckks_eval_linear_transformation_into", Budget, right_short);
    assert_guard::<BE, _>(
        "budget factor",
        "ckks_eval_linear_transformation_into",
        Budget,
        &mut res,
        |res| module.blas_matmul(res, &short, &column, &ctx, &keys, &keys, &mut op_scratch.borrow()),
    );
    let hot_short = matmul_left_operand(module, &short, &ctx, &keys, OneShot, &mut op_scratch.borrow()).unwrap();
    assert_guard::<BE, _>(
        "budget hot left",
        "ckks_eval_linear_transformation_into",
        Budget,
        &mut res,
        |res| module.blas_matmul_prepared(res, &hot_short, cold_right, &ctx, &keys, &keys, &mut op_scratch.borrow()),
    );
    // `short_right` is 1 x 2: on the right, it multiplies `column`'s 2 x 1 operand into 2 x 2.
    let hot_short = matmul_right_operand(module, &short_right, &ctx, &keys, OneShot, &mut op_scratch.borrow()).unwrap();
    assert_guard::<BE, _>(
        "budget hot right",
        "ckks_eval_linear_transformation_into",
        Budget,
        &mut alloc(2, 2),
        |res| {
            let left = &left_column[side(Reuse)];
            module.blas_matmul_prepared(res, left, &hot_short, &ctx, &keys, &keys, &mut op_scratch.borrow())
        },
    );
    assert!(
        [&short, &short_right].map(grid_snapshot::<BE, _>) == inputs,
        "budget: a preparation changed its input"
    );
    // Both factors short, `A` by one bit and `B` by two bits: whichever side the product holds, `A`'s terms come first,
    // so `blas_matmul` and the prepared product of hot operands return the error `A` alone gives, not `B`'s, before
    // any write. `1 x 2` by `2 x 1` holds `A`; `2 x 1` by `1 x 1` holds `B`.
    let short_b = encrypt_grid::<BE, F, E>(
        &params,
        module,
        host_module,
        &encoder,
        &sk,
        ct.block_layout().log_delta() + 2 * p - 2,
        0,
        &Grid::model(log_d, 1, 2, lanes, 11),
        &mut scratch.borrow(),
    );
    let as_column = |g: &BlasCiphertextOwned<BE>| BlasCiphertext::from_blocks(g.clone().into_blocks(), log_d, 2, 1).unwrap();
    let (short_column, short_b_column) = (as_column(&short), as_column(&short_b));
    let (ct_block, short_b_block) = (corner::<BE>(&ct, ct.rows(), 1), corner::<BE>(&short_b, short_b.rows(), 1));
    let error = |result: &Result<(), poulpy_ckks::CKKSError>| result.as_ref().err().and_then(|e| e.composition().cloned());
    for (label, short_a, short_b, a, b) in [
        ("A held", &short, &short_b_column, &ct, &column),
        ("B held", &short_column, &short_b_block, &column, &ct_block),
    ] {
        let mut res = alloc(short_a.rows(), short_b.cols());
        let mut matmul = |x: &BlasCiphertextOwned<BE>, y: &BlasCiphertextOwned<BE>, res: &mut BlasCiphertextOwned<BE>| {
            module.blas_matmul(res, x, y, &ctx, &keys, &keys, &mut op_scratch.borrow())
        };
        let left_error = error(&matmul(short_a, b, &mut res));
        let right_error = error(&matmul(a, short_b, &mut res));
        assert!(
            left_error.is_some() && left_error != right_error,
            "{label}: one error for both sides"
        );
        let mut both = None;
        assert_guard::<BE, _>(
            &format!("budget {label}"),
            "ckks_eval_linear_transformation_into",
            Budget,
            &mut res,
            |res| {
                let result = matmul(short_a, short_b, res);
                both = error(&result);
                result
            },
        );
        assert_eq!(both, left_error, "{label}: blas_matmul returned the right terms' error first");
        let s = &mut op_scratch.borrow();
        let hot_a = matmul_left_operand(module, short_a, &ctx, &keys, OneShot, s).unwrap();
        let hot_b = matmul_right_operand(module, short_b, &ctx, &keys, OneShot, s).unwrap();
        let prepared = module.blas_matmul_prepared(&mut res, &hot_a, &hot_b, &ctx, &keys, &keys, s);
        assert_eq!(
            error(&prepared),
            left_error,
            "{label}: the prepared product returned the right terms' error first"
        );
    }

    // Each operation, under every hint and mix, on exactly its query, then one byte short of it.
    let mut res = alloc(1, 1);
    let bytes = module.blas_matmul_tmp_bytes(&res, &ct, &column, &ctx, &tsk_layout, &atk_layout);
    assert_short_scratch::<BE, _>("blas_matmul", &mut res, bytes, |res, s| {
        module.blas_matmul(res, &ct, &column, &ctx, &keys, &keys, s)
    });
    for (hl, hr) in MIXES {
        let (l, r) = (&left[side(hl)], &right[side(hr)]);
        let op = &format!("blas_matmul_prepared {hl:?} x {hr:?}");
        // A product narrows `res`, and the query reads it: every call starts from a fresh one.
        let bytes = module.blas_matmul_prepared_tmp_bytes(&alloc(1, 1), l, r, &ctx, &tsk_layout, &atk_layout);
        with_scratch::<BE, _>(bytes, |s| {
            module.blas_matmul_prepared(&mut alloc(1, 1), l, r, &ctx, &keys, &keys, s)
        })
        .unwrap_or_else(|e| panic!("{op}: {e}"));
        assert_short_scratch::<BE, _>(op, &mut alloc(1, 1), bytes, |res, s| {
            module.blas_matmul_prepared(res, l, r, &ctx, &keys, &keys, s)
        });
    }
    for hint in HINTS {
        let op = &format!("blas_matmul_prepare_left {hint:?}");
        let bytes = module.blas_matmul_prepare_left_tmp_bytes(&ct, &ctx, &atk_layout, hint);
        with_scratch::<BE, _>(bytes, |s| matmul_left_operand(module, &ct, &ctx, &keys, hint, s))
            .unwrap_or_else(|e| panic!("{op}: {e}"));
        assert_short_scratch_panics::<BE, _>(op, bytes, |s| matmul_left_operand(module, &ct, &ctx, &keys, hint, s));
        let op = &format!("blas_matmul_prepare_right {hint:?}");
        let bytes = module.blas_matmul_prepare_right_tmp_bytes(&column, &ctx, &atk_layout, hint);
        with_scratch::<BE, _>(bytes, |s| matmul_right_operand(module, &column, &ctx, &keys, hint, s))
            .unwrap_or_else(|e| panic!("{op}: {e}"));
        assert_short_scratch_panics::<BE, _>(op, bytes, |s| matmul_right_operand(module, &column, &ctx, &keys, hint, s));
    }

    // Valid grids, a key store without rotation keys or without a tensor key. The preparations look up rotation keys
    // under either hint and no tensor key; the prepared product looks up the tensor key, and the rotation keys of a hot
    // side's terms.
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
    for hint in HINTS {
        let label = &format!("no rotation keys {hint:?}");
        let left = matmul_left_operand(module, &ct, &ctx, &no_atks, hint, &mut op_scratch.borrow());
        assert_rejected(label, "blas_matmul_prepare_left", MissingKey, left);
        let right = matmul_right_operand(module, &column, &ctx, &no_atks, hint, &mut op_scratch.borrow());
        assert_rejected(label, "blas_matmul_prepare_right", MissingKey, right);
        matmul_left_operand(module, &ct, &ctx, &no_tsk, hint, &mut op_scratch.borrow()).unwrap();
        matmul_right_operand(module, &column, &ctx, &no_tsk, hint, &mut op_scratch.borrow()).unwrap();
    }
    for (hl, hr) in MIXES {
        let (l, r) = (&left[side(hl)], &right[side(hr)]);
        let label = &format!("no tensor key {hl:?} x {hr:?}");
        assert_guard::<BE, _>(label, op, MissingTensorKey, &mut res, |res| {
            module.blas_matmul_prepared(res, l, r, &ctx, &no_tsk, &no_tsk, &mut op_scratch.borrow())
        });
        let mut call = |res: &mut BlasCiphertextOwned<BE>| {
            module.blas_matmul_prepared(res, l, r, &ctx, &no_atks, &no_atks, &mut op_scratch.borrow())
        };
        let label = &format!("no rotation keys {hl:?} x {hr:?}");
        if (hl, hr) == (Reuse, Reuse) {
            call(&mut res).unwrap_or_else(|e| panic!("{label}: {e}"));
        } else {
            assert_guard::<BE, _>(label, op, MissingKey, &mut res, call);
        }
    }
}

/// A long product: `dec(matmul(enc(A), enc(B)))` is `A B` for a `1 x 8` grid times an `8 x 1` grid of distinct `4 x 4`
/// blocks, whose 32 products one JKLS kernel sums, and `blas_matmul_prepared` of cold operands equals `blas_matmul` bit
/// for bit.
/// The blocks are distinct encryptions, so the terms' digits are independent, as `Module::max_base2k`'s model assumes:
/// copies of one block would add up coherently, outside it, past what an FFT suite's radix holds once the products
/// accumulate in the transform domain. It runs on its own ring, `N = 32`, at the suite's radix. The product by public
/// matrices is not repeated here: its giant steps add their products in the transform domain too, whose radix rule
/// `blas_matmul_pt_right` documents.
pub fn test_blas_matmul_long_sums<BE, F, E>(params: CKKSTestParams, _module: &Module<BE>, _host_module: &Module<HostBytesBackend>)
where
    BE: TestContextBackend<Ring = Standard> + BlasMatMulImpl + BlasEncodingImpl,
    Module<BE>: TestContextModule<BE>
        + CKKSEncodingOps<BE, F>
        + BlasEncodingOps<BE, F>
        + CKKSLinearTransformationOps<BE>
        + CnvPVecAlloc<BE>,
    F: TestScalar + DiagonalArithmetic,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
{
    let params = CKKSTestParams { n: 32, hw: 16, ..params };
    let module = &Module::<BE>::new(params.n as u64);
    let host_module = &Module::<HostBytesBackend>::new(params.n as u64);
    // One 4 x 4 block fills the 16 slots, whatever the suite's dimension.
    let (log_d, d) = (2, 4);
    let slots = params.n / 2;
    let lanes = slots / (d * d);
    let encoder = ReferenceEncoder::<E>::new(slots).unwrap();
    let (sk_raw, sk) = gen_sk_with_raw(&params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, module);
    let (ctx, atks) = context_and_keys::<BE, F>(&params, module, log_d, &sk_raw, &mut scratch.borrow());
    let tsk = gen_tsk(&params, module, &sk_raw, &mut scratch.borrow());
    let keys = TestKeys::<BE> { atks, tsk: Some(tsk) };
    let (tsk_layout, atk_layout) = (params.tsk_layout(), params.atk_layout());
    // `d` terms per inner block.
    let inner = 8;
    let (a, b) = (Grid::model(log_d, 1, inner, lanes, 7), Grid::model(log_d, inner, 1, lanes, 8));
    let [ct_a, ct_b] = [&a, &b].map(|grid| {
        encrypt_grid::<BE, F, E>(
            &params,
            module,
            host_module,
            &encoder,
            &sk,
            params.k,
            0,
            grid,
            &mut scratch.borrow(),
        )
    });
    // `inner` blocks add their noise: at most `log2(inner)` bits of precision go.
    let check = |label: &str, res: &BlasCiphertextOwned<BE>, want: &Grid, scratch: &mut ScratchArena<'_, BE>| {
        let (block, (re, im)) = (res.block(0, 0), &want.blocks(slots)[0]);
        let log_delta = block.log_delta() - inner.next_power_of_two().ilog2() as usize;
        assert_decrypt_precision_at_log_delta::<BE, F, E>(
            label,
            &params,
            module,
            &encoder,
            block,
            &sk,
            &to_f::<F>(re),
            &to_f::<F>(im),
            log_delta,
            scratch,
        );
    };

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
            &mut poisoned_scratch::<BE>(bytes).borrow(),
        )
        .unwrap();
    check("matmul", &res, &a.matmul(&b), &mut scratch.borrow());

    let bytes = module
        .blas_matmul_prepare_left_tmp_bytes(&ct_a, &ctx, &atk_layout, Reuse)
        .max(module.blas_matmul_prepare_right_tmp_bytes(&ct_b, &ctx, &atk_layout, Reuse));
    let mut prepare_scratch = poisoned_scratch::<BE>(bytes);
    let left = matmul_left_operand(module, &ct_a, &ctx, &keys, Reuse, &mut prepare_scratch.borrow()).unwrap();
    let right = matmul_right_operand(module, &ct_b, &ctx, &keys, Reuse, &mut prepare_scratch.borrow()).unwrap();
    let mut prepared = alloc_grid(&params, module, log_d, 1, 1, params.k);
    let bytes = module.blas_matmul_prepared_tmp_bytes(&prepared, &left, &right, &ctx, &tsk_layout, &atk_layout);
    module
        .blas_matmul_prepared(
            &mut prepared,
            &left,
            &right,
            &ctx,
            &keys,
            &keys,
            &mut poisoned_scratch::<BE>(bytes).borrow(),
        )
        .unwrap();
    assert_eq!(grid_snapshot::<BE, _>(&prepared), grid_snapshot::<BE, _>(&res));
}

/// On a grid holding one matrix repeated over the slots, each `rotate_rows_rep(k)`, a product by 1 encoded at
/// `log_delta = 0`, equals poulpy-ckks's rotation by `k d` exactly as a torus element, at the rotation's metadata: it
/// spends no scale. An exact tie may be written either way, since both outputs are normalized.
pub fn test_blas_replicated_shifts<BE, F, E>(
    params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
    log_d: usize,
) where
    BE: TestContextBackend<Ring = Standard> + BlasMatMulImpl + BlasEncodingImpl,
    Module<BE>: TestContextModule<BE>
        + CKKSEncodingOps<BE, F>
        + BlasEncodingOps<BE, F>
        + CKKSLinearTransformationOps<BE>
        + CKKSRotateOps<BE>
        + CnvPVecAlloc<BE>,
    F: TestScalar + DiagonalArithmetic,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
{
    let (slots, d) = (params.n / 2, 1usize << log_d);
    let encoder = ReferenceEncoder::<E>::new(slots).unwrap();
    let (sk_raw, sk) = gen_sk_with_raw(&params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, module);
    let (ctx, atks) = context_and_keys::<BE, F>(&params, module, log_d, &sk_raw, &mut scratch.borrow());
    let keys = TestKeys::<BE> { atks, tsk: None };
    let one_slot = slots.ilog2() as usize - 2 * log_d;
    let ct = encrypt_grid::<BE, F, E>(
        &params,
        module,
        host_module,
        &encoder,
        &sk,
        params.k,
        one_slot,
        &Grid::model(log_d, 1, 1, 1, 21),
        &mut scratch.borrow(),
    );
    assert!(replicated::<Standard>(&ct), "the fixture holds one matrix");
    let carried = |s: &Snapshot| -> Vec<i64> {
        let g = s.layout.glwe_layout;
        let (n, cols, bits) = (g.n.as_usize(), g.rank.as_usize() + 1, g.base2k.as_usize());
        let size = s.digits.len() / (n * cols);
        let mut out = s.digits.clone();
        for (col, c) in (0..cols).flat_map(|col| (0..n).map(move |c| (col, c))) {
            // The carry out of the top limb is dropped: it is the torus wrap.
            let mut carry = 0i128;
            for limb in (0..size).rev() {
                let i = (limb * cols + col) * n + c;
                let v = i128::from(out[i]) + carry;
                carry = (v + (1i128 << (bits - 1))) >> bits;
                out[i] = (v - (carry << bits)) as i64;
            }
        }
        out
    };
    for k in 1..d {
        let mut via_map = alloc_grid(&params, module, log_d, 1, 1, params.k);
        module
            .ckks_eval_linear_transformation_self_into(
                via_map.block_mut(0, 0),
                ct.block(0, 0),
                ctx.rotate_rows_rep(k),
                &keys,
                &mut scratch.borrow(),
            )
            .unwrap();
        // The eval leaves the label unset; every reference stamps the source's.
        via_map.block_mut(0, 0).set_log_sparsity(ct.block(0, 0).log_sparsity());
        let mut via_rotation = alloc_grid(&params, module, log_d, 1, 1, params.k);
        module
            .ckks_rotate_into(
                (k * d) as i64,
                via_rotation.block_mut(0, 0),
                ct.block(0, 0),
                &keys,
                &mut scratch.borrow(),
            )
            .unwrap();
        assert_eq!(
            layout_of(via_map.block(0, 0)),
            layout_of(via_rotation.block(0, 0)),
            "k={k}: metadata"
        );
        let (mut a, mut b) = (grid_snapshot::<BE, _>(&via_map), grid_snapshot::<BE, _>(&via_rotation));
        assert_eq!(carried(&a[0]), carried(&b[0]), "k={k}: torus value");
        (a[0].digits, b[0].digits) = (Vec::new(), Vec::new());
        assert_eq!(a, b, "k={k}: layout, flag and noise");
    }
}

/// An operand allocated from infos holds its storage and reports those infos and the batch `0`, which the prepared
/// product rejects on either side; one of another shape than its source, other dimensions or a narrower width, is
/// rejected before any write. The reference's operands hold the bytes the reference's queries report, and preparing
/// into one already prepared, from a grid of the same shape at another scale, keeps every buffer, gives a fresh
/// preparation's bytes and hot blocks and reports that grid's infos; a refill rejected for its shape or, cold, for its
/// budget leaves the operand as it was. Under either hint, on either side.
pub fn test_blas_matmul_operands<BE, F, E>(
    params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
    log_d: usize,
) where
    BE: TestContextBackend<Ring = Standard> + BlasImpl,
    Module<BE>: TestContextModule<BE>
        + CKKSEncodingOps<BE, F>
        + BlasEncodingOps<BE, F>
        + CKKSLinearTransformationOps<BE>
        + CnvPVecAlloc<BE>,
    F: TestScalar + DiagonalArithmetic,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
{
    let slots = params.n / 2;
    let encoder = ReferenceEncoder::<E>::new(slots).unwrap();
    let (sk_raw, sk) = gen_sk_with_raw(&params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, module);
    let (ctx, atks) = context_and_keys::<BE, F>(&params, module, log_d, &sk_raw, &mut scratch.borrow());
    let keys = TestKeys::<BE> { atks, tsk: None };
    let lanes = slots >> (2 * log_d);
    let mut encrypt = |params: &CKKSTestParams, k: usize, seed: u64| {
        let grid = Grid::model(log_d, 2, 2, lanes, seed);
        encrypt_grid::<BE, F, E>(params, module, host_module, &encoder, &sk, k, 0, &grid, &mut scratch.borrow())
    };
    let first = encrypt(&params, params.k, 9);
    // The shape of `first` at a lower scale, then at a narrower width.
    let second = encrypt(&at_scale(&params, other_scales(&params)[0]), params.k, 10);
    let narrow = encrypt(&params, params.k - params.base2k, 11);
    // The shape of `first`, one bit short of the `2p` its left terms spend.
    let p = params.prec().log_delta();
    let short = relabel(&first, |meta| meta.scale = Scale::Log(params.k - 2 * p + 1));
    let atk = params.atk_layout();
    for hint in HINTS {
        let (left, right) = (
            blas_matmul_left_operand_layout(&first, &ctx, hint),
            blas_matmul_right_operand_layout(&first, &ctx, hint),
        );
        let fresh_left = module.blas_matmul_left_alloc_from_infos(&left);
        assert_eq!(
            operand_infos(&fresh_left),
            (left.grid, Some(0), hint, left.source),
            "left {hint:?}"
        );
        let fresh_right = module.blas_matmul_right_alloc_from_infos(&right);
        assert_eq!(
            operand_infos(&fresh_right),
            (right.grid, Some(0), hint, right.source),
            "right {hint:?}"
        );
        let (left_bytes, right_bytes) = (
            module.blas_matmul_prepare_left_tmp_bytes(&first, &ctx, &atk, hint),
            module.blas_matmul_prepare_right_tmp_bytes(&first, &ctx, &atk, hint),
        );
        let mut op_scratch = poisoned_scratch::<BE>(left_bytes.max(right_bytes));
        let mut prepared_left = matmul_left_operand(module, &first, &ctx, &keys, hint, &mut op_scratch.borrow()).unwrap();
        let mut prepared_right = matmul_right_operand(module, &first, &ctx, &keys, hint, &mut op_scratch.borrow()).unwrap();
        let mut res = alloc_grid(&params, module, log_d, 2, 2, params.k);
        for (side, l, r) in [
            ("left", &fresh_left, &prepared_right),
            ("right", &prepared_left, &fresh_right),
        ] {
            assert_guard::<BE, _>(
                &format!("unprepared {side} {hint:?}"),
                "blas_matmul_prepared",
                InvalidOperand,
                &mut res,
                |res| module.blas_matmul_prepared(res, l, r, &ctx, &keys, &keys, &mut op_scratch.borrow()),
            );
        }

        let mut other = left;
        other.grid.rows += 1;
        let mut wrong = module.blas_matmul_left_alloc_from_infos(&other);
        assert_rejected(
            &format!("left {hint:?}"),
            "blas_matmul_prepare_left",
            InvalidOperand,
            module.blas_matmul_prepare_left(&mut wrong, &first, &ctx, &keys, &mut op_scratch.borrow()),
        );
        assert_eq!(
            operand_infos(&wrong),
            (other.grid, Some(0), hint, other.source),
            "left {hint:?}"
        );
        let infos = operand_infos(&prepared_left);
        assert_rejected(
            &format!("narrow left {hint:?}"),
            "blas_matmul_prepare_left",
            InvalidOperand,
            module.blas_matmul_prepare_left(&mut prepared_left, &narrow, &ctx, &keys, &mut op_scratch.borrow()),
        );
        assert_eq!(operand_infos(&prepared_left), infos, "narrow left {hint:?}");
        let infos = operand_infos(&prepared_right);
        assert_rejected(
            &format!("narrow right {hint:?}"),
            "blas_matmul_prepare_right",
            InvalidOperand,
            module.blas_matmul_prepare_right(&mut prepared_right, &narrow, &ctx, &keys, &mut op_scratch.borrow()),
        );
        assert_eq!(operand_infos(&prepared_right), infos, "narrow right {hint:?}");

        let mut left_rejected = vec![("narrow", &narrow, InvalidOperand, "blas_matmul_prepare_left")];
        if hint == Reuse {
            left_rejected.push(("budget", &short, Budget, "ckks_eval_linear_transformation_into"));
        }
        check_operand_refill(
            &format!("left {hint:?}"),
            || module.blas_matmul_left_alloc_from_infos(&left),
            module.blas_matmul_left_bytes_of_from_infos(&left),
            CKKSPreparedLeft::cnv,
            |op, a| {
                let s = &mut poisoned_scratch::<BE>(left_bytes);
                module.blas_matmul_prepare_left_reference(op, a, &ctx, &keys, &mut s.borrow())
            },
            |a| blas_matmul_left_operand_layout(a, &ctx, hint),
            [&first, &second],
            &left_rejected,
        );
        check_operand_refill(
            &format!("right {hint:?}"),
            || module.blas_matmul_right_alloc_from_infos(&right),
            module.blas_matmul_right_bytes_of_from_infos(&right),
            CKKSPreparedRight::cnv,
            |op, b| {
                let s = &mut poisoned_scratch::<BE>(right_bytes);
                module.blas_matmul_prepare_right_reference(op, b, &ctx, &keys, &mut s.borrow())
            },
            |b| blas_matmul_right_operand_layout(b, &ctx, hint),
            [&first, &second],
            &[("narrow", &narrow, InvalidOperand, "blas_matmul_prepare_right")],
        );
    }
}

/// A grid rejected in a refill, by label: the rejection and the op that raises it.
type Refused<'a, BE> = (&'a str, &'a BlasCiphertextOwned<BE>, Rejection, &'a str);

/// An operand from `alloc` holds `bytes_of` bytes. `prepare`, the reference, fills it from
/// `first`, then from `second`: it keeps every buffer, holds a fresh preparation's bytes and hot blocks (digits,
/// metadata, flag and noise) and reports `infos(second)` and `second`'s batch. Each of `rejected` then fails as it
/// states and leaves the operand as it was.
fn check_operand_refill<BE, P, V>(
    label: &str,
    alloc: impl Fn() -> BlasMatMulOperand<BE, P>,
    bytes_of: usize,
    cnv: fn(&P) -> &V,
    prepare: impl Fn(&mut BlasMatMulOperand<BE, P>, &BlasCiphertextOwned<BE>) -> CKKSResult<()>,
    infos: impl Fn(&BlasCiphertextOwned<BE>) -> BlasMatMulOperandLayout,
    [first, second]: [&BlasCiphertextOwned<BE>; 2],
    rejected: &[Refused<'_, BE>],
) where
    BE: Backend<OwnedBuf = AlignedBuf, ZnxWord = i64>,
    V: DataView<D = AlignedBuf>,
{
    let state = |op: &BlasMatMulOperand<BE, P>| {
        let bytes: Vec<Vec<u8>> = operand_buffers(op, cnv).iter().map(|b| b.to_vec()).collect();
        let blocks: Vec<Snapshot> = op.blocks.iter().map(snapshot::<BE, _>).collect();
        (bytes, blocks, operand_infos(op))
    };
    let mut op = alloc();
    assert_eq!(
        operand_buffers(&op, cnv).iter().map(|b| b.len()).sum::<usize>(),
        bytes_of,
        "{label}: bytes_of"
    );
    prepare(&mut op, first).unwrap();
    let addresses: Vec<_> = operand_buffers(&op, cnv).iter().map(|b| b.as_ptr()).collect();
    prepare(&mut op, second).unwrap();
    let mut fresh = alloc();
    prepare(&mut fresh, second).unwrap();
    assert!(
        operand_buffers(&op, cnv).iter().map(|b| b.as_ptr()).eq(addresses),
        "{label}: the refill reallocated"
    );
    let refilled = state(&op);
    assert!(
        refilled == state(&fresh),
        "{label}: the refill differs from a fresh preparation"
    );
    let want = infos(second);
    assert_eq!(
        refilled.2,
        (want.grid, second.mat_slots(), want.hint, want.source),
        "{label}: infos"
    );
    for &(name, source, rejection, op_name) in rejected {
        assert_rejected(&format!("{label} {name}"), op_name, rejection, prepare(&mut op, source));
        assert!(
            state(&op) == refilled,
            "{label} {name}: a rejected refill changed the operand"
        );
    }
}

/// The bytes of every buffer `op` holds, in order: its blocks' when hot, `cnv` of each term when cold.
fn operand_buffers<'a, BE, P, V>(op: &'a BlasMatMulOperand<BE, P>, cnv: impl Fn(&'a P) -> &'a V) -> Vec<&'a [u8]>
where
    BE: Backend<OwnedBuf = AlignedBuf>,
    V: DataView<D = AlignedBuf> + 'a,
{
    let blocks = op.blocks.iter().map(|block| &block.data().data()[..]);
    blocks.chain(op.terms.iter().map(|term| &cnv(term).data()[..])).collect()
}

/// `blas_matmul` of flag-clear factors whose blocks hold one limb past their width, which carries part of their value,
/// decrypts to the product of their values and equals the prepared product of two hot operands of them bit for bit: a
/// hot operand holds the live limbs of its source's blocks, normalized, from which the product makes the terms it makes
/// from the factors read in place.
pub fn test_blas_matmul_flag_clear<BE, F, E>(
    params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
    log_d: usize,
) where
    BE: TestContextBackend<Ring = Standard> + BlasImpl,
    Module<BE>: TestContextModule<BE>
        + CKKSEncodingOps<BE, F>
        + BlasEncodingOps<BE, F>
        + CKKSLinearTransformationOps<BE>
        + CnvPVecAlloc<BE>,
    F: TestScalar + DiagonalArithmetic,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
{
    let slots = params.n / 2;
    let encoder = ReferenceEncoder::<E>::new(slots).unwrap();
    let (sk_raw, sk) = gen_sk_with_raw(&params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, module);
    let (ctx, atks) = context_and_keys::<BE, F>(&params, module, log_d, &sk_raw, &mut scratch.borrow());
    let keys = TestKeys::<BE> {
        atks,
        tsk: Some(gen_tsk(&params, module, &sk_raw, &mut scratch.borrow())),
    };
    let lanes = slots >> (2 * log_d);
    let models = [9, 10].map(|seed| Grid::model(log_d, 2, 2, lanes, seed));
    let [a, b] = [&models[0], &models[1]].map(|grid| {
        let ct = encrypt_grid::<BE, F, E>(
            &params,
            module,
            host_module,
            &encoder,
            &sk,
            params.k,
            0,
            grid,
            &mut scratch.borrow(),
        );
        flag_clear(module, &ct, &mut scratch.borrow())
    });
    assert!(
        a.blocks()
            .iter()
            .chain(b.blocks())
            .all(|x| !x.is_canonical() && x.max_size() > x.size()),
        "a factor block is canonical or holds no limb past its width"
    );
    let alloc = || alloc_grid(&params, module, log_d, 2, 2, params.k);
    let mut in_place = alloc();
    let bytes = module.blas_matmul_tmp_bytes(&in_place, &a, &b, &ctx, &params.tsk_layout(), &params.atk_layout());
    let mut op_scratch = poisoned_scratch::<BE>(bytes);
    module
        .blas_matmul(&mut in_place, &a, &b, &ctx, &keys, &keys, &mut op_scratch.borrow())
        .unwrap();
    assert_grid_decrypt::<BE, F, E>(
        "matmul of flag-clear factors",
        &params,
        module,
        &encoder,
        &in_place,
        &sk,
        &models[0].matmul(&models[1]),
        &product_layout::<BE::Ring>(&a, &b, 2, 2, params.k, params.prec().log_delta()),
        &mut scratch.borrow(),
    );
    let left = matmul_left_operand(module, &a, &ctx, &keys, OneShot, &mut op_scratch.borrow()).unwrap();
    let right = matmul_right_operand(module, &b, &ctx, &keys, OneShot, &mut op_scratch.borrow()).unwrap();
    let mut prepared = alloc();
    module
        .blas_matmul_prepared(&mut prepared, &left, &right, &ctx, &keys, &keys, &mut op_scratch.borrow())
        .unwrap();
    assert!(
        grid_snapshot::<BE, _>(&in_place) == grid_snapshot::<BE, _>(&prepared),
        "blas_matmul and the prepared product of hot operands differ on flag-clear factors"
    );
}

/// `grid` with every block one limb wider and flag-clear, of the same value: every limb but the first, the new one
/// included, holds one unit of the limb above it.
fn flag_clear<BE>(
    module: &Module<BE>,
    grid: &BlasCiphertextOwned<BE>,
    scratch: &mut ScratchArena<'_, BE>,
) -> BlasCiphertextOwned<BE>
where
    BE: TestContextBackend,
    Module<BE>: CKKSModuleAlloc<BE> + CKKSCopyOps<BE>,
    for<'a> BE::BufMut<'a>: HostDataMut,
{
    let blocks = grid
        .blocks()
        .iter()
        .map(|block| {
            let (live, base2k) = (block.size(), block.base2k().as_usize());
            let mut layout = layout_of(block);
            layout.glwe_layout.k = ((live + 1) * base2k).into();
            let mut wide = module.ckks_ciphertext_alloc_from_infos(&layout);
            module.ckks_copy(&mut wide, block, scratch).unwrap();
            {
                let mut glwe = wide.to_mut::<BE>();
                let data = glwe.data_mut();
                for col in 0..data.cols() {
                    for limb in 1..=live {
                        data.at_mut(col, limb - 1).iter_mut().for_each(|digit| *digit -= 1);
                        data.at_mut(col, limb).iter_mut().for_each(|digit| *digit += 1 << base2k);
                    }
                }
            }
            GLWEToBackendMut::<BE>::set_canonical(&mut wide, false);
            wide
        })
        .collect();
    BlasCiphertext::from_blocks(blocks, grid.log_d(), grid.rows(), grid.cols()).unwrap()
}
