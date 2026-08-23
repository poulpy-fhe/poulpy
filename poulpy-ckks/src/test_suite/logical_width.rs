//! Logical-width regressions for CKKS ciphertext backend views.
//!
//! A CKKS ciphertext carries a logical width (`k`, hence `size()`) that can sit
//! below its physical allocation (`max_size()`). Every backend view of a
//! ciphertext must expose the logical width: a backend that reads or writes the
//! inactive tail would consume undefined limbs and move bytes nobody needs.
//! Plaintexts are excluded on purpose: they are full integer polynomials and
//! expose their whole `encoded_k()` allocation.

use std::collections::HashMap;

use poulpy_core::layouts::{
    Base2K, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, LWEInfos, LinearTransformationBabySteps, TorusPrecision,
};
use poulpy_hal::{
    api::{NegacyclicFFT, NegacyclicFFTNew, ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Backend, CyclotomicOrder, HostBytesBackend, HostDataMut, HostDataRef, Module, ScratchOwned, ZnxView, ZnxViewMut},
};

use crate::{
    CKKSInfos, CKKSMeta, CoeffsMeta, SetCKKSInfos, SlotsKind,
    api::{CKKSDFTMatrixOps, CKKSDFTOps, CKKSLinearTransformationOps, CKKSNegOps},
    layouts::{
        CKKSCiphertext, CKKSCiphertextOwned, CKKSModuleAlloc, CKKSPlaintextOwned, CKKSPlaintextVecHostCodec, DFTMatrix,
        DFTMatrixPrepared, DFTOutputFormat, DFTPlan, DFTType, Encode, ScratchArenaTakeCKKS, Standard, Unnormalized,
    },
    test_suite::{
        CKKSTestParams,
        helpers::{TestContextBackend, TestContextHostModule, TestContextModule, TestScalar, gen_atk, gen_sk_with_raw},
    },
};

/// Marker written into the inactive tail: any op that mistakes capacity for
/// logical width both reads it and overwrites it.
const POISON: i64 = 0x5EED_BEEF;

/// `log2` of the live complex slot count. Five factors, so the ping-pong chain
/// takes its odd-length branch (leading factor in place, then two pairs) and
/// writes the scratch ciphertext more than once.
const LOG_SLOTS: usize = 5;

/// Small transform parameters: `n = 2^(LOG_SLOTS + 1)` with a `k` sized for the
/// chained factor multiplies plus input scale and headroom.
fn small_params(params: &CKKSTestParams) -> CKKSTestParams {
    let base2k = params.base2k;
    let log_delta = params.prec().log_delta();
    CKKSTestParams {
        n: 1 << (LOG_SLOTS + 1),
        base2k,
        k: (log_delta * (LOG_SLOTS + 3)).next_multiple_of(base2k),
        prec_meta: CKKSMeta {
            log_sparsity: 0,
            log_delta,
            slots: SlotsKind::Complex,
        },
        prec_log_budget: 10,
        hw: params.hw.min(1 << LOG_SLOTS),
        dsize: params.dsize,
        rank: 1,
    }
}

/// Allocates a ciphertext of `physical` limbs whose first `active` limbs hold a
/// deterministic pattern and whose tail holds [`POISON`], then narrows `k` to
/// `active` limbs.
fn poisoned_ct<BE>(module: &Module<BE>, params: &CKKSTestParams, active: usize, physical: usize) -> CKKSCiphertextOwned<BE>
where
    BE: TestContextBackend,
    Module<BE>: CKKSModuleAlloc<BE>,
{
    let base2k = params.base2k;
    let mut layout = params.glwe_layout();
    layout.layout.k = (physical * base2k).into();
    let mut ct = module.ckks_ciphertext_alloc_from_glwe_infos(&layout);
    ct.set_meta(params.prec().meta);
    let cols = ct.rank().as_usize() + 1;
    let n = ct.n().as_usize();
    for col in 0..cols {
        for limb in 0..physical {
            let value = if limb < active { (col + limb) as i64 + 1 } else { POISON };
            ct.data_mut().at_mut(col, limb)[..n].fill(value);
        }
    }
    ct.set_k(TorusPrecision((active * base2k) as u32));
    ct
}

/// Every limb of every column, as a flat vector; `limbs` selects how many.
fn limbs_of<BE>(ct: &CKKSCiphertextOwned<BE>, limbs: usize) -> Vec<i64>
where
    BE: Backend<OwnedBuf = Vec<u8>, ZnxWord = i64>,
{
    let cols = ct.rank().as_usize() + 1;
    let n = ct.n().as_usize();
    let mut out = Vec::with_capacity(cols * limbs * n);
    for col in 0..cols {
        for limb in 0..limbs {
            out.extend_from_slice(&ct.data().at(col, limb)[..n]);
        }
    }
    out
}

/// **Views expose the logical width.** Owned, scratch-backed and unnormalized
/// write views of a ciphertext narrowed to three of four physical limbs must all
/// report three limbs, a real op through the narrowed view must leave the
/// poisoned fourth limb untouched, and a plaintext view must still expose its
/// whole encoded allocation.
pub fn test_backend_views_expose_logical_width<BE, F, E>(
    params: CKKSTestParams,
    _module: &Module<BE>,
    _host_module: &Module<HostBytesBackend>,
) where
    BE: TestContextBackend,
    Module<BE>: TestContextModule<BE>,
    Module<HostBytesBackend>: TestContextHostModule,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
    for<'a> <BE as Backend>::BufRef<'a>: HostDataRef,
    for<'a> <BE as Backend>::BufMut<'a>: HostDataMut,
{
    let params = small_params(&params);
    let module = Module::<BE>::new(params.n as u64);
    let base2k = params.base2k;
    let (active, physical) = (3usize, 4usize);

    let mut ct = poisoned_ct(&module, &params, active, physical);
    assert_eq!(ct.size(), active);
    assert_eq!(ct.max_size(), physical);

    // 2. Owned views, shared and mutable.
    assert_eq!(
        GLWEToBackendRef::<BE>::to_backend_ref(&ct).data().size(),
        active,
        "owned shared view exposes physical capacity"
    );
    assert_eq!(
        GLWEToBackendMut::<BE>::to_backend_mut(&mut ct).data().size(),
        active,
        "owned mutable view exposes physical capacity"
    );

    // 3. A real op through the narrowed view leaves the inactive tail alone.
    let poisoned_tail = |ct: &CKKSCiphertextOwned<BE>| -> Vec<i64> {
        let cols = ct.rank().as_usize() + 1;
        (0..cols)
            .flat_map(|col| ct.data().at(col, physical - 1)[..ct.n().as_usize()].to_vec())
            .collect()
    };
    let tail_before = poisoned_tail(&ct);
    module.ckks_neg_assign(&mut ct).unwrap();
    assert_eq!(
        tail_before,
        poisoned_tail(&ct),
        "op through a narrowed view wrote the inactive tail"
    );
    assert!(
        tail_before.iter().all(|&v| v == POISON),
        "test setup: the inactive tail is not poisoned"
    );

    // 2 (continued). Scratch-backed view.
    let mut layout = params.glwe_layout();
    layout.layout.k = (physical * base2k).into();
    let mut view_scratch = ScratchOwned::<BE>::alloc(module.ckks_ciphertext_alloc_from_glwe_infos(&layout).max_size() * 1024);
    {
        let arena = view_scratch.borrow();
        let (mut view, _) = arena.take_ckks_ciphertext_scratch(&layout, params.prec().meta);
        view.set_k(TorusPrecision((active * base2k) as u32));
        assert_eq!(
            GLWEToBackendRef::<BE>::to_backend_ref(&view).data().size(),
            active,
            "scratch-backed shared view exposes physical capacity"
        );
        assert_eq!(
            GLWEToBackendMut::<BE>::to_backend_mut(&mut view).data().size(),
            active,
            "scratch-backed mutable view exposes physical capacity"
        );
    }

    // 2 (continued). Unnormalized write view.
    let mut unnormalized = CKKSCiphertext::<_, _, Unnormalized>::new(poisoned_ct(&module, &params, active, physical));
    {
        let mut write_view = unnormalized.write_view();
        assert_eq!(
            GLWEToBackendRef::<BE>::to_backend_ref(&write_view).data().size(),
            active,
            "unnormalized write view exposes physical capacity"
        );
        assert_eq!(
            GLWEToBackendMut::<BE>::to_backend_mut(&mut write_view).data().size(),
            active,
            "unnormalized mutable write view exposes physical capacity"
        );
    }

    // 4. Plaintexts keep their full encoded allocation.
    let pt: CKKSPlaintextOwned<BE> = module.ckks_pt_vec_alloc(Base2K(base2k as u32), (physical * base2k).into());
    assert_eq!(
        GLWEToBackendRef::<BE>::to_backend_ref(&pt).data().size(),
        pt.max_size(),
        "plaintext view no longer exposes every encoded limb"
    );
}

/// Builds the prepared Encode/`Standard` transform for [`small_params`] plus the
/// automorphism keys it needs.
#[allow(clippy::type_complexity)]
fn encode_dft<BE, F>(
    params: &CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
    scratch: &mut ScratchOwned<BE>,
) -> (
    DFTMatrixPrepared<BE, Encode, Standard>,
    HashMap<i64, poulpy_core::layouts::prepared::GLWEAutomorphismKeyPrepared<BE::OwnedBuf, BE>>,
    poulpy_core::layouts::prepared::GLWESecretPrepared<BE::OwnedBuf, BE>,
)
where
    BE: TestContextBackend,
    Module<BE>: TestContextModule<BE> + CKKSDFTOps<BE> + CKKSDFTMatrixOps<BE, F>,
    Module<HostBytesBackend>: TestContextHostModule,
    F: TestScalar,
{
    let log_delta = params.prec().log_delta();
    let (sk_raw, sk) = gen_sk_with_raw(params, module, host_module, [7u8; 32]);
    let plan = DFTPlan::new(
        DFTType::Encode,
        vec![(1usize, 2usize); LOG_SLOTS],
        DFTOutputFormat::Standard,
        CoeffsMeta::from_delta_budget(log_delta, 10),
    )
    .unwrap();
    let unprepared: DFTMatrix<BE, Encode, Standard> = module
        .ckks_new_dft_matrix::<Encode, Standard>(Base2K(params.base2k as u32), &plan, &mut scratch.borrow())
        .unwrap();
    let dft = module.ckks_prepare_dft_matrix(&unprepared, &mut scratch.borrow());

    let order = module.cyclotomic_order();
    let mut atks = HashMap::new();
    for p in dft.galois_elements(order) {
        atks.entry(p)
            .or_insert_with(|| gen_atk(params, module, p, &sk_raw, &mut scratch.borrow()));
    }
    (dft, atks, sk)
}

/// **Inactive capacity is inert.** A compact ciphertext and an over-capacity one
/// with the same active prefix but a poisoned tail must produce identical
/// prepared baby steps and identical active DFT output limbs.
pub fn test_inactive_capacity_does_not_change_results<BE, F, E>(
    params: CKKSTestParams,
    _module: &Module<BE>,
    _host_module: &Module<HostBytesBackend>,
) where
    BE: TestContextBackend,
    Module<BE>: TestContextModule<BE> + CKKSDFTOps<BE> + CKKSDFTMatrixOps<BE, F> + CKKSLinearTransformationOps<BE>,
    Module<HostBytesBackend>: TestContextHostModule,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
    for<'a> <BE as Backend>::BufRef<'a>: HostDataRef,
    for<'a> <BE as Backend>::BufMut<'a>: HostDataMut,
    CKKSPlaintextOwned<HostBytesBackend>: CKKSPlaintextVecHostCodec<f64>,
{
    let params = small_params(&params);
    let module = Module::<BE>::new(params.n as u64);
    let host_module = Module::<HostBytesBackend>::new(params.n as u64);
    let mut scratch = crate::test_suite::helpers::alloc_scratch(&params, &module);
    let (dft, atks, _sk) = encode_dft::<BE, F>(&params, &module, &host_module, &mut scratch);

    let active = params.k / params.base2k;
    let compact = poisoned_ct(&module, &params, active, active);
    let padded = poisoned_ct(&module, &params, active, active + 2);
    assert_eq!(
        limbs_of::<BE>(&compact, active),
        limbs_of::<BE>(&padded, active),
        "test setup: prefixes differ"
    );

    // Prepared baby steps of the first factor must be byte-identical.
    let factor = dft.factors().first().expect("dft has no factors");
    let prepared = |ct: &CKKSCiphertextOwned<BE>, scratch: &mut ScratchOwned<BE>| {
        let mut babies = LinearTransformationBabySteps::<BE>::alloc(&module, factor.baby_steps(), ct);
        module
            .ckks_prepare_linear_transformation_baby_steps(&mut babies, ct, &atks, &mut scratch.borrow())
            .unwrap();
        babies
    };
    let babies_compact = prepared(&compact, &mut scratch);
    let babies_padded = prepared(&padded, &mut scratch);
    for rot in babies_compact.baby_steps() {
        assert_eq!(
            babies_compact.baby_step(rot).raw(),
            babies_padded.baby_step(rot).raw(),
            "prepared baby step {rot} depends on inactive capacity"
        );
    }

    // The whole transform must agree on every active limb.
    let mut compact = compact;
    let mut padded = padded;
    module
        .ckks_dft_evaluate_assign(&mut compact, &dft, &atks, &mut scratch.borrow())
        .unwrap();
    module
        .ckks_dft_evaluate_assign(&mut padded, &dft, &atks, &mut scratch.borrow())
        .unwrap();
    assert_eq!(compact.meta(), padded.meta(), "metadata depends on inactive capacity");
    assert_eq!(compact.k(), padded.k(), "torus width depends on inactive capacity");
    assert_eq!(
        limbs_of::<BE>(&compact, compact.size()),
        limbs_of::<BE>(&padded, padded.size()),
        "active DFT output depends on inactive capacity"
    );
}

/// **Ping-pong equals the per-factor assign chain.** The chained evaluator and a
/// factor-by-factor in-place loop must agree on metadata and on every active
/// output limb; the inactive backing capacity is unspecified and not compared.
/// Also pins the chain's scratch bound to `ckks_dft_evaluate_tmp_bytes`.
pub fn test_dft_ping_pong_matches_assign_chain<BE, F, E>(
    params: CKKSTestParams,
    _module: &Module<BE>,
    _host_module: &Module<HostBytesBackend>,
) where
    BE: TestContextBackend,
    Module<BE>: TestContextModule<BE> + CKKSDFTOps<BE> + CKKSDFTMatrixOps<BE, F> + CKKSLinearTransformationOps<BE>,
    Module<HostBytesBackend>: TestContextHostModule,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
    for<'a> <BE as Backend>::BufRef<'a>: HostDataRef,
    for<'a> <BE as Backend>::BufMut<'a>: HostDataMut,
    CKKSPlaintextOwned<HostBytesBackend>: CKKSPlaintextVecHostCodec<f64>,
{
    let params = small_params(&params);
    let module = Module::<BE>::new(params.n as u64);
    let host_module = Module::<HostBytesBackend>::new(params.n as u64);
    let mut scratch = crate::test_suite::helpers::alloc_scratch(&params, &module);
    let (dft, atks, _sk) = encode_dft::<BE, F>(&params, &module, &host_module, &mut scratch);

    let active = params.k / params.base2k;
    let mut ping_pong = poisoned_ct(&module, &params, active, active);
    let mut assign_chain = poisoned_ct(&module, &params, active, active);

    // The chain runs inside exactly the budget it advertises.
    let atk_layout = params.atk_layout();
    let chain_bytes = module.ckks_dft_evaluate_tmp_bytes(&ping_pong, &atk_layout);
    let mut chain_scratch = ScratchOwned::<BE>::alloc(chain_bytes);
    module
        .ckks_dft_evaluate_assign(&mut ping_pong, &dft, &atks, &mut chain_scratch.borrow())
        .unwrap();

    for factor in dft.factors() {
        module
            .ckks_eval_linear_transformation_self_assign(&mut assign_chain, factor, &atks, &mut scratch.borrow())
            .unwrap();
    }

    assert_eq!(ping_pong.meta(), assign_chain.meta(), "ping-pong metadata differs");
    assert_eq!(ping_pong.k(), assign_chain.k(), "ping-pong torus width differs");
    assert_eq!(
        limbs_of::<BE>(&ping_pong, ping_pong.size()),
        limbs_of::<BE>(&assign_chain, assign_chain.size()),
        "ping-pong active output differs from the per-factor assign chain"
    );
}
