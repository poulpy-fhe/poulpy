//! Keyless GLWE operation parity.
//!
//! These take no prepared key, so each test is fill, upload, run on both,
//! compare.

use poulpy_hal::{
    api::{
        CnvOps, CnvPVecAlloc, ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxAutomorphism, VecZnxDftAlloc, VecZnxIdftNormalize,
        VecZnxIdftNormalizeTmpBytes,
    },
    layouts::{
        Backend, CnvPVecLToBackendRef, CnvPVecRToBackendRef, HostDataMut, Module, ScratchOwned, VecZnxDftToBackendMut, ZnxView,
        ZnxViewMut,
    },
    source::Source,
    test_suite::{TestParams, vec_znx_backend_mut, vec_znx_backend_ref},
};

use crate::{
    GGSWRotate, GLWEAdd, GLWEBytesOf, GLWECopy, GLWEMaskFill, GLWEMulConst, GLWEMulPlain, GLWEMulPreparedPlainSum,
    GLWEMulXpMinusOne, GLWENegate, GLWENormalize, GLWEPreparedTensoring, GLWERotate, GLWEShift, GLWESub, GLWETensoring, GLWEZero,
    api::TransferInto,
    gadget::cnv_offset_to_limb_offset,
    glwe_prepare_left, glwe_prepare_left_tmp_bytes, glwe_prepare_right, glwe_prepare_right_tmp_bytes,
    layouts::{
        Base2K, Degree, GGSWAtViewMut, GLWEInfos, GLWELayout, GLWEToBackendMut, LWEInfos, ModuleCoreAlloc, Rank, TorusPrecision,
    },
    test_suite::parity::{ParityBackend, ParityShapes, poisoned_scratch, ref_glwe, unnormalized_twin},
};

/// Layouts swept by the keyless operation tests.
fn layouts(n: u32, base2k: usize, shapes: &ParityShapes) -> Vec<GLWELayout> {
    let mut out = Vec::new();
    for &rank in &shapes.ranks {
        let rank = rank as u32;
        for k in [1usize, base2k - 1, base2k, base2k + 1, 2 * base2k, 5 * base2k - 1] {
            out.push(GLWELayout {
                n: Degree(n),
                base2k: Base2K(base2k as u32),
                k: TorusPrecision(k as u32),
                rank: Rank(rank),
            });
        }
    }
    out
}

/// Runs `op` on both backends over uniform inputs and compares the results.
///
/// `op` receives `(module, res, a, b, scratch)`; tests that need only one
/// operand ignore `b`.
#[allow(clippy::too_many_arguments)]
fn compare<BR, BT, FR, FT>(
    params: &TestParams,
    shapes: &ParityShapes,
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
    label: &str,
    seed: u8,
    tmp_bytes: (usize, usize),
    op_ref: FR,
    op_test: FT,
) where
    BR: ParityBackend,
    BT: ParityBackend,
    BR::OwnedBuf: HostDataMut,
    Module<BR>: ModuleCoreAlloc<OwnedBuf = BR::OwnedBuf, ZnxWord = i64>,
    Module<BT>: ModuleCoreAlloc<OwnedBuf = BT::OwnedBuf, ZnxWord = i64>,
    ScratchOwned<BR>: ScratchOwnedAlloc<BR> + ScratchOwnedBorrow<BR>,
    ScratchOwned<BT>: ScratchOwnedAlloc<BT> + ScratchOwnedBorrow<BT>,
    FR: Fn(
        &Module<BR>,
        &mut crate::layouts::BackendGLWE<BR>,
        &crate::layouts::BackendGLWE<BR>,
        &crate::layouts::BackendGLWE<BR>,
        &mut crate::test_suite::PoisonedScratch<BR>,
    ),
    FT: Fn(
        &Module<BT>,
        &mut crate::layouts::BackendGLWE<BT>,
        &crate::layouts::BackendGLWE<BT>,
        &crate::layouts::BackendGLWE<BT>,
        &mut crate::test_suite::PoisonedScratch<BT>,
    ),
{
    assert!(params.n <= module_ref.n() && params.n <= module_test.n());

    let n = params.n as u32;
    let mut source = Source::new([seed; 32]);

    for a_infos in layouts(n, params.base2k, shapes) {
        for res_infos in layouts(n, params.base2k, shapes) {
            if res_infos.rank != a_infos.rank {
                continue;
            }

            let mut a_ref = ref_glwe(module_ref, &a_infos, &mut source);
            let b_ref = ref_glwe(module_ref, &a_infos, &mut source);
            let mut res_ref = ref_glwe(module_ref, &res_infos, &mut source);

            if label.starts_with("glwe_normalize") {
                for col in 0..a_ref.data.cols() {
                    for limb in 0..a_ref.data.size() {
                        for value in a_ref.data.at_mut(col, limb) {
                            *value *= 3;
                        }
                    }
                }
                for col in 0..res_ref.data.cols() {
                    for limb in 0..res_ref.data.size() {
                        for value in res_ref.data.at_mut(col, limb) {
                            *value *= 3;
                        }
                    }
                }
            }
            let mut a_test = module_test.glwe_alloc_from_infos(&a_infos);
            a_ref.transfer_into(&mut a_test);
            let mut b_test = module_test.glwe_alloc_from_infos(&a_infos);
            b_ref.transfer_into(&mut b_test);
            let mut res_test = module_test.glwe_alloc_from_infos(&res_infos);
            res_ref.transfer_into(&mut res_test);

            let mut scratch_ref = poisoned_scratch::<BR>(tmp_bytes.0);
            let mut scratch_test = poisoned_scratch::<BT>(tmp_bytes.1);

            op_ref(module_ref, &mut res_ref, &a_ref, &b_ref, &mut scratch_ref);
            op_test(module_test, &mut res_test, &a_test, &b_test, &mut scratch_test);

            if label == "glwe_copy" {
                let expected = if res_infos.k >= a_infos.k { a_ref.noise() } else { None };
                assert_eq!(res_ref.noise(), expected);
                assert_eq!(res_test.noise(), expected);
                // Copies keep the tag; every other operation must clear it.
                res_ref.noise = None;
                res_test.noise = None;
            }
            let mut have = module_ref.glwe_alloc_from_infos(&res_infos);
            res_test.transfer_into(&mut have);
            assert_glwe_eq!(
                res_ref,
                have,
                "{label}: a_k={:?} res_k={:?} rank={:?}",
                a_infos.k,
                res_infos.k,
                res_infos.rank
            );
        }
    }
}

/// `glwe_add_into` agrees with the selected comparison backend.
pub fn test_glwe_add_parity<BR, BT>(params: &TestParams, shapes: &ParityShapes, module_ref: &Module<BR>, module_test: &Module<BT>)
where
    BR: ParityBackend,
    BT: ParityBackend,
    BR::OwnedBuf: HostDataMut,
    Module<BR>: GLWEAdd<BR>,
    Module<BT>: GLWEAdd<BT>,
    ScratchOwned<BR>: ScratchOwnedAlloc<BR> + ScratchOwnedBorrow<BR>,
    ScratchOwned<BT>: ScratchOwnedAlloc<BT> + ScratchOwnedBorrow<BT>,
{
    compare(
        params,
        shapes,
        module_ref,
        module_test,
        "glwe_add_into",
        3,
        (0, 0),
        |m, res, a, b, _| m.glwe_add_into(res, a, b),
        |m, res, a, b, _| m.glwe_add_into(res, a, b),
    );
    compare(
        params,
        shapes,
        module_ref,
        module_test,
        "glwe_add_assign",
        13,
        (0, 0),
        |m, res, a, _, _| m.glwe_add_assign(res, a),
        |m, res, a, _, _| m.glwe_add_assign(res, a),
    );
}

/// `glwe_sub` agrees with the selected comparison backend.
pub fn test_glwe_sub_parity<BR, BT>(params: &TestParams, shapes: &ParityShapes, module_ref: &Module<BR>, module_test: &Module<BT>)
where
    BR: ParityBackend,
    BT: ParityBackend,
    BR::OwnedBuf: HostDataMut,
    Module<BR>: GLWESub<BR>,
    Module<BT>: GLWESub<BT>,
    ScratchOwned<BR>: ScratchOwnedAlloc<BR> + ScratchOwnedBorrow<BR>,
    ScratchOwned<BT>: ScratchOwnedAlloc<BT> + ScratchOwnedBorrow<BT>,
{
    compare(
        params,
        shapes,
        module_ref,
        module_test,
        "glwe_sub",
        5,
        (0, 0),
        |m, res, a, b, _| m.glwe_sub(res, a, b),
        |m, res, a, b, _| m.glwe_sub(res, a, b),
    );
    compare(
        params,
        shapes,
        module_ref,
        module_test,
        "glwe_sub_assign",
        13,
        (0, 0),
        |m, res, a, _, _| m.glwe_sub_assign(res, a),
        |m, res, a, _, _| m.glwe_sub_assign(res, a),
    );

    compare(
        params,
        shapes,
        module_ref,
        module_test,
        "glwe_sub_negate_assign",
        13,
        (0, 0),
        |m, res, a, _, _| m.glwe_sub_negate_assign(res, a),
        |m, res, a, _, _| m.glwe_sub_negate_assign(res, a),
    );
    // A rank-zero plaintext contributes to the body only, while a - ciphertext
    // negates every mask column and every destination limb beyond the plaintext.
    let mut source = Source::new([71; 32]);
    for &rank in &shapes.ranks {
        for plaintext_k in [params.base2k - 1, 3 * params.base2k + 1] {
            let res_infos = GLWELayout {
                n: (params.n as u32).into(),
                base2k: (params.base2k as u32).into(),
                k: (2 * params.base2k as u32 + 1).into(),
                rank: (rank as u32).into(),
            };
            let pt_infos = GLWELayout {
                rank: Rank(0),
                k: (plaintext_k as u32).into(),
                ..res_infos
            };
            let pt_ref = ref_glwe(module_ref, &pt_infos, &mut source);
            let mut res_ref = ref_glwe(module_ref, &res_infos, &mut source);
            let mut expected = module_ref.glwe_alloc_from_infos(&res_infos);
            for col in 0..expected.data.cols() {
                for limb in 0..expected.data.size() {
                    let old = res_ref.data.at(col, limb);
                    let pt = if col == 0 && limb < pt_ref.data.size() {
                        Some(pt_ref.data.at(0, limb))
                    } else {
                        None
                    };
                    for (i, value) in expected.data.at_mut(col, limb).iter_mut().enumerate() {
                        *value = pt.map_or(0, |coefficients| coefficients[i]) - old[i];
                    }
                }
            }
            let mut pt_test = module_test.glwe_alloc_from_infos(&pt_infos);
            pt_ref.transfer_into(&mut pt_test);
            let mut res_test = module_test.glwe_alloc_from_infos(&res_infos);
            res_ref.transfer_into(&mut res_test);
            module_ref.glwe_sub_negate_assign(&mut res_ref, &pt_ref);
            module_test.glwe_sub_negate_assign(&mut res_test, &pt_test);
            let mut have = module_ref.glwe_alloc_from_infos(&res_infos);
            res_test.transfer_into(&mut have);
            assert_eq!(res_ref, expected, "plaintext minus GLWE: reference mask sign and limb tails");
            assert_eq!(have, expected, "plaintext minus GLWE: backend mask sign and limb tails");
        }
    }
}

/// `glwe_negate` agrees with the selected comparison backend.
pub fn test_glwe_negate_parity<BR, BT>(
    params: &TestParams,
    shapes: &ParityShapes,
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    BR: ParityBackend,
    BT: ParityBackend,
    BR::OwnedBuf: HostDataMut,
    Module<BR>: GLWENegate<BR>,
    Module<BT>: GLWENegate<BT>,
    ScratchOwned<BR>: ScratchOwnedAlloc<BR> + ScratchOwnedBorrow<BR>,
    ScratchOwned<BT>: ScratchOwnedAlloc<BT> + ScratchOwnedBorrow<BT>,
{
    compare(
        params,
        shapes,
        module_ref,
        module_test,
        "glwe_negate",
        9,
        (0, 0),
        |m, res, a, _, _| m.glwe_negate(res, a),
        |m, res, a, _, _| m.glwe_negate(res, a),
    );
    compare(
        params,
        shapes,
        module_ref,
        module_test,
        "glwe_negate_assign",
        13,
        (0, 0),
        |m, res, _, _, _| m.glwe_negate_assign(res),
        |m, res, _, _, _| m.glwe_negate_assign(res),
    );
}

/// `glwe_normalize` agrees with the selected comparison backend.
///
/// The one keyless operation that carries limb-carry logic, so the one most
/// worth comparing byte-for-byte.
pub fn test_glwe_normalize_parity<BR, BT>(
    params: &TestParams,
    shapes: &ParityShapes,
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    BR: ParityBackend,
    BT: ParityBackend,
    BR::OwnedBuf: HostDataMut,
    Module<BR>: GLWENormalize<BR>,
    Module<BT>: GLWENormalize<BT>,
    ScratchOwned<BR>: ScratchOwnedAlloc<BR> + ScratchOwnedBorrow<BR>,
    ScratchOwned<BT>: ScratchOwnedAlloc<BT> + ScratchOwnedBorrow<BT>,
{
    let tmp = (module_ref.glwe_normalize_tmp_bytes(), module_test.glwe_normalize_tmp_bytes());
    compare(
        params,
        shapes,
        module_ref,
        module_test,
        "glwe_normalize",
        17,
        tmp,
        |m, res, a, _, s| m.glwe_normalize(res, a, &mut s.borrow()),
        |m, res, a, _, s| m.glwe_normalize(res, a, &mut s.borrow()),
    );
    compare(
        params,
        shapes,
        module_ref,
        module_test,
        "glwe_normalize_assign",
        13,
        tmp,
        |m, res, _, _, s| m.glwe_normalize_assign(res, &mut s.borrow()),
        |m, res, _, _, s| m.glwe_normalize_assign(res, &mut s.borrow()),
    );
}

/// `glwe_rotate` agrees with the selected comparison backend.
pub fn test_glwe_rotate_parity<BR, BT>(
    params: &TestParams,
    shapes: &ParityShapes,
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    BR: ParityBackend,
    BT: ParityBackend,
    BR::OwnedBuf: HostDataMut,
    Module<BR>: GLWERotate<BR>,
    Module<BT>: GLWERotate<BT>,
    ScratchOwned<BR>: ScratchOwnedAlloc<BR> + ScratchOwnedBorrow<BR>,
    ScratchOwned<BT>: ScratchOwnedAlloc<BT> + ScratchOwnedBorrow<BT>,
{
    for k in [-5i64, 1, 7] {
        compare(
            params,
            shapes,
            module_ref,
            module_test,
            "glwe_rotate",
            29,
            (0, 0),
            move |m, res, a, _, _| m.glwe_rotate(k, res, a),
            move |m, res, a, _, _| m.glwe_rotate(k, res, a),
        );
    }
    for k in [-5i64, 0, 1, params.n as i64, 2 * params.n as i64 + 7] {
        compare(
            params,
            shapes,
            module_ref,
            module_test,
            "glwe_rotate_assign",
            29,
            (module_ref.glwe_rotate_tmp_bytes(), module_test.glwe_rotate_tmp_bytes()),
            move |m, res, _, _, s| m.glwe_rotate_assign(k, res, &mut s.borrow()),
            move |m, res, _, _, s| m.glwe_rotate_assign(k, res, &mut s.borrow()),
        );
    }
}

/// Checks tensor apply and square across ranks against the selected comparison backend,
/// including specializations selected by rank and ring degree.
pub fn test_glwe_tensor_parity<BR, BT>(
    params: &TestParams,
    shapes: &ParityShapes,
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    BR: ParityBackend,
    BT: ParityBackend,
    BR::OwnedBuf: HostDataMut,
    Module<BR>: GLWETensoring<BR>
        + GLWEPreparedTensoring<BR>
        + ModuleCoreAlloc<OwnedBuf = BR::OwnedBuf, ZnxWord = i64>
        + CnvPVecAlloc<BR>
        + CnvOps<BR>
        + GLWENormalize<BR>
        + GLWEBytesOf<BR>,
    Module<BT>: GLWETensoring<BT>
        + GLWEPreparedTensoring<BT>
        + ModuleCoreAlloc<OwnedBuf = BT::OwnedBuf, ZnxWord = i64>
        + CnvPVecAlloc<BT>
        + CnvOps<BT>
        + GLWENormalize<BT>
        + GLWEBytesOf<BT>,
    ScratchOwned<BR>: ScratchOwnedAlloc<BR> + ScratchOwnedBorrow<BR>,
    ScratchOwned<BT>: ScratchOwnedAlloc<BT> + ScratchOwnedBorrow<BT>,
{
    assert!(params.n <= module_ref.n() && params.n <= module_test.n());
    let n = params.n as u32;
    let base2k = params.base2k;
    let mut source = Source::new([29u8; 32]);

    for a_infos in layouts(n, base2k, shapes).into_iter().flat_map(|infos| {
        [
            infos,
            GLWELayout {
                k: TorusPrecision(infos.k.0.saturating_sub(1).max(1)),
                ..infos
            },
        ]
    }) {
        test_glwe_tensor_parity_case(
            &a_infos,
            &[0, base2k - 1, base2k, a_infos.k.0 as usize],
            true,
            module_ref,
            module_test,
            &mut source,
        );
    }
}

/// Checks tensor multiplication and squaring for one caller-selected layout and
/// convolution offsets. Each operation uses exactly its queried, poisoned scratch
/// and compares canonical coefficients and metadata against the selected backend.
///
/// Backend registrations can use focused cases at expensive ring degrees while
/// retaining the full parameter sweep at smaller degrees, which alone also runs
/// unnormalized operands.
pub fn test_glwe_tensor_parity_for_layout<BR, BT>(
    layout: &GLWELayout,
    offsets: &[usize],
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    BR: ParityBackend,
    BT: ParityBackend,
    BR::OwnedBuf: HostDataMut,
    Module<BR>: GLWETensoring<BR>
        + GLWEPreparedTensoring<BR>
        + ModuleCoreAlloc<OwnedBuf = BR::OwnedBuf, ZnxWord = i64>
        + CnvPVecAlloc<BR>
        + CnvOps<BR>
        + GLWENormalize<BR>
        + GLWEBytesOf<BR>,
    Module<BT>: GLWETensoring<BT>
        + GLWEPreparedTensoring<BT>
        + ModuleCoreAlloc<OwnedBuf = BT::OwnedBuf, ZnxWord = i64>
        + CnvPVecAlloc<BT>
        + CnvOps<BT>
        + GLWENormalize<BT>
        + GLWEBytesOf<BT>,
    ScratchOwned<BR>: ScratchOwnedAlloc<BR> + ScratchOwnedBorrow<BR>,
    ScratchOwned<BT>: ScratchOwnedAlloc<BT> + ScratchOwnedBorrow<BT>,
{
    assert!(!offsets.is_empty(), "tensor parity requires at least one offset");
    test_glwe_tensor_parity_case(layout, offsets, false, module_ref, module_test, &mut Source::new([29u8; 32]));
}

fn test_glwe_tensor_parity_case<BR, BT>(
    a_infos: &GLWELayout,
    offsets: &[usize],
    with_twins: bool,
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
    source: &mut Source,
) where
    BR: ParityBackend,
    BT: ParityBackend,
    BR::OwnedBuf: HostDataMut,
    Module<BR>: GLWETensoring<BR>
        + GLWEPreparedTensoring<BR>
        + ModuleCoreAlloc<OwnedBuf = BR::OwnedBuf, ZnxWord = i64>
        + CnvPVecAlloc<BR>
        + CnvOps<BR>
        + GLWENormalize<BR>
        + GLWEBytesOf<BR>,
    Module<BT>: GLWETensoring<BT>
        + GLWEPreparedTensoring<BT>
        + ModuleCoreAlloc<OwnedBuf = BT::OwnedBuf, ZnxWord = i64>
        + CnvPVecAlloc<BT>
        + CnvOps<BT>
        + GLWENormalize<BT>
        + GLWEBytesOf<BT>,
    ScratchOwned<BR>: ScratchOwnedAlloc<BR> + ScratchOwnedBorrow<BR>,
    ScratchOwned<BT>: ScratchOwnedAlloc<BT> + ScratchOwnedBorrow<BT>,
{
    assert!(a_infos.n.as_usize() <= module_ref.n() && a_infos.n.as_usize() <= module_test.n());
    let res_infos = GLWELayout {
        n: a_infos.n,
        base2k: a_infos.base2k,
        k: TorusPrecision(2 * a_infos.k.0),
        rank: a_infos.rank,
    };

    let a_ref = ref_glwe(module_ref, a_infos, source);
    let b_ref = ref_glwe(module_ref, a_infos, source);
    let mut a_test = module_test.glwe_alloc_from_infos(a_infos);
    a_ref.transfer_into(&mut a_test);
    let mut b_test = module_test.glwe_alloc_from_infos(a_infos);
    b_ref.transfer_into(&mut b_test);

    let mut out_ref = module_ref.glwe_tensor_alloc_from_infos(&res_infos);
    let mut out_test = module_test.glwe_tensor_alloc_from_infos(&res_infos);
    // Operands are normalized before the convolution, so one offset covers the twins.
    let twins = with_twins.then(|| {
        let mut twin_a = module_test.glwe_alloc_from_infos(a_infos);
        let mut twin_b = module_test.glwe_alloc_from_infos(a_infos);
        unnormalized_twin::<BR, BT>(&a_ref, &mut twin_a);
        unnormalized_twin::<BR, BT>(&b_ref, &mut twin_b);
        (twin_a, twin_b)
    });
    let n = a_infos.n.as_usize();
    let cols = a_infos.rank.as_usize() + 1;
    let b_size = a_infos.k.as_usize().div_ceil(a_infos.base2k.as_usize());
    // The hints swap with the parity of `k`, which the layout sweep varies.
    let (a_hint, b_hint) = crate::test_suite::prepare_hints(a_infos.k.as_usize());
    let mut b_prep_ref = module_ref.cnv_pvec_right_alloc(n, cols, b_size, b_hint);
    let mut b_prep_test = module_test.cnv_pvec_right_alloc(n, cols, b_size, b_hint);
    let mut a_prep_ref = module_ref.cnv_pvec_left_alloc(n, cols, b_size, a_hint);
    let mut a_prep_test = module_test.cnv_pvec_left_alloc(n, cols, b_size, a_hint);
    // Each preparation on exactly its own budget.
    glwe_prepare_right(
        module_ref,
        &mut b_prep_ref,
        &b_ref,
        a_infos.k.as_usize(),
        &mut poisoned_scratch::<BR>(glwe_prepare_right_tmp_bytes(module_ref, b_size, a_infos)).borrow(),
    );
    glwe_prepare_right(
        module_test,
        &mut b_prep_test,
        &b_test,
        a_infos.k.as_usize(),
        &mut poisoned_scratch::<BT>(glwe_prepare_right_tmp_bytes(module_test, b_size, a_infos)).borrow(),
    );
    glwe_prepare_left(
        module_ref,
        &mut a_prep_ref,
        &a_ref,
        a_infos.k.as_usize(),
        &mut poisoned_scratch::<BR>(glwe_prepare_left_tmp_bytes(module_ref, b_size, a_infos)).borrow(),
    );
    // The test side prepares its left operand from the unnormalized twin, when there is one.
    glwe_prepare_left(
        module_test,
        &mut a_prep_test,
        twins.as_ref().map_or(&a_test, |(twin_a, _)| twin_a),
        a_infos.k.as_usize(),
        &mut poisoned_scratch::<BT>(glwe_prepare_left_tmp_bytes(module_test, b_size, a_infos)).borrow(),
    );
    for (i, &cnv_offset) in offsets.iter().enumerate() {
        let twins = twins.as_ref().filter(|_| i == 0);
        let mut scratch_ref = poisoned_scratch::<BR>(module_ref.glwe_tensor_apply_tmp_bytes(&out_ref, &a_ref, &b_ref));
        let mut scratch_test = poisoned_scratch::<BT>(module_test.glwe_tensor_apply_tmp_bytes(&out_test, &a_test, &b_test));
        module_ref.glwe_tensor_apply(cnv_offset, &mut out_ref, &a_ref, &b_ref, &mut scratch_ref.borrow());
        module_test.glwe_tensor_apply(cnv_offset, &mut out_test, &a_test, &b_test, &mut scratch_test.borrow());
        let mut have = module_ref.glwe_tensor_alloc_from_infos(&res_infos);
        out_test.transfer_into(&mut have);
        assert_eq!(
            out_ref, have,
            "glwe_tensor_apply: k={:?} rank={:?} offset={cnv_offset}",
            a_infos.k, a_infos.rank
        );
        if let Some((twin_a, twin_b)) = twins {
            let mut scratch_test = poisoned_scratch::<BT>(module_test.glwe_tensor_apply_tmp_bytes(&out_test, &a_test, &b_test));
            module_test.glwe_tensor_apply(cnv_offset, &mut out_test, twin_a, twin_b, &mut scratch_test.borrow());
            out_test.transfer_into(&mut have);
            assert_eq!(
                out_ref, have,
                "glwe_tensor_apply, unnormalized operands: k={:?} rank={:?} offset={cnv_offset}",
                a_infos.k, a_infos.rank
            );
        }

        // The prepared product follows the ordinary product's rounding exactly.
        let mut prepared_ref = module_ref.glwe_tensor_alloc_from_infos(&res_infos);
        let mut scratch_ref =
            poisoned_scratch::<BR>(module_ref.glwe_tensor_apply_prepared_right_tmp_bytes(&out_ref, &a_ref, a_ref.size(), b_size));
        let mut scratch_test = poisoned_scratch::<BT>(module_test.glwe_tensor_apply_prepared_right_tmp_bytes(
            &out_test,
            &a_test,
            a_test.size(),
            b_size,
        ));
        module_ref.glwe_tensor_apply_prepared_right(
            cnv_offset,
            &mut prepared_ref,
            &a_ref,
            &b_prep_ref,
            b_size,
            &mut scratch_ref.borrow(),
        );
        module_test.glwe_tensor_apply_prepared_right(
            cnv_offset,
            &mut out_test,
            &a_test,
            &b_prep_test,
            b_size,
            &mut scratch_test.borrow(),
        );
        assert_eq!(
            prepared_ref, out_ref,
            "glwe_tensor_apply_prepared_right versus glwe_tensor_apply: k={:?} rank={:?} offset={cnv_offset}",
            a_infos.k, a_infos.rank
        );
        out_test.transfer_into(&mut have);
        assert_eq!(
            prepared_ref, have,
            "glwe_tensor_apply_prepared_right: k={:?} rank={:?} offset={cnv_offset}",
            a_infos.k, a_infos.rank
        );

        // So does the product of two prepared operands, into fresh outputs.
        let mut both_ref = module_ref.glwe_tensor_alloc_from_infos(&res_infos);
        let mut both_test = module_test.glwe_tensor_alloc_from_infos(&res_infos);
        let mut scratch_ref =
            poisoned_scratch::<BR>(module_ref.glwe_tensor_apply_prepared_tmp_bytes(&out_ref, b_size, b_size, a_infos.base2k));
        let mut scratch_test =
            poisoned_scratch::<BT>(module_test.glwe_tensor_apply_prepared_tmp_bytes(&out_test, b_size, b_size, a_infos.base2k));
        module_ref.glwe_tensor_apply_prepared(
            cnv_offset,
            &mut both_ref,
            &a_prep_ref,
            b_size,
            &b_prep_ref,
            b_size,
            a_infos.base2k,
            &mut scratch_ref.borrow(),
        );
        module_test.glwe_tensor_apply_prepared(
            cnv_offset,
            &mut both_test,
            &a_prep_test,
            b_size,
            &b_prep_test,
            b_size,
            a_infos.base2k,
            &mut scratch_test.borrow(),
        );
        assert_eq!(
            both_ref, out_ref,
            "glwe_tensor_apply_prepared versus glwe_tensor_apply: k={:?} rank={:?} offset={cnv_offset}",
            a_infos.k, a_infos.rank
        );
        both_test.transfer_into(&mut have);
        assert_eq!(
            both_ref, have,
            "glwe_tensor_apply_prepared: k={:?} rank={:?} offset={cnv_offset}",
            a_infos.k, a_infos.rank
        );

        let mut scratch_ref = poisoned_scratch::<BR>(module_ref.glwe_tensor_square_apply_tmp_bytes(&out_ref, &a_ref));
        let mut scratch_test = poisoned_scratch::<BT>(module_test.glwe_tensor_square_apply_tmp_bytes(&out_test, &a_test));
        module_ref.glwe_tensor_square_apply(cnv_offset, &mut out_ref, &a_ref, &mut scratch_ref.borrow());
        module_test.glwe_tensor_square_apply(cnv_offset, &mut out_test, &a_test, &mut scratch_test.borrow());
        let mut have = module_ref.glwe_tensor_alloc_from_infos(&res_infos);
        out_test.transfer_into(&mut have);
        assert_eq!(
            out_ref, have,
            "glwe_tensor_square_apply: k={:?} rank={:?} offset={cnv_offset}",
            a_infos.k, a_infos.rank
        );
        if let Some((twin_a, _)) = twins {
            let mut scratch_test = poisoned_scratch::<BT>(module_test.glwe_tensor_square_apply_tmp_bytes(&out_test, &a_test));
            module_test.glwe_tensor_square_apply(cnv_offset, &mut out_test, twin_a, &mut scratch_test.borrow());
            out_test.transfer_into(&mut have);
            assert_eq!(
                out_ref, have,
                "glwe_tensor_square_apply, unnormalized operand: k={:?} rank={:?} offset={cnv_offset}",
                a_infos.k, a_infos.rank
            );
        }
    }
}

/// Zeroing and copying retain destination layout metadata, including partial limbs.
pub fn test_glwe_copy_zero_parity<BR, BT>(
    params: &TestParams,
    shapes: &ParityShapes,
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    BR: ParityBackend,
    BT: ParityBackend,
    BR::OwnedBuf: HostDataMut,
    Module<BR>: GLWECopy<BR> + GLWEZero<BR>,
    Module<BT>: GLWECopy<BT> + GLWEZero<BT>,
    ScratchOwned<BR>: ScratchOwnedAlloc<BR> + ScratchOwnedBorrow<BR>,
    ScratchOwned<BT>: ScratchOwnedAlloc<BT> + ScratchOwnedBorrow<BT>,
{
    compare(
        params,
        shapes,
        module_ref,
        module_test,
        "glwe_zero",
        41,
        (0, 0),
        |m, res, _, _, _| m.glwe_zero(res),
        |m, res, _, _, _| m.glwe_zero(res),
    );
    compare(
        params,
        shapes,
        module_ref,
        module_test,
        "glwe_copy",
        43,
        (0, 0),
        |m, res, a, _, _| m.glwe_copy(res, a, &mut poisoned_scratch::<BR>(m.glwe_copy_tmp_bytes(res, a)).borrow()),
        |m, res, a, _, _| m.glwe_copy(res, a, &mut poisoned_scratch::<BT>(m.glwe_copy_tmp_bytes(res, a)).borrow()),
    );
}

/// Shift and monomial-product variants, including whole-ring and precision boundaries.
pub fn test_glwe_shift_parity<BR, BT>(
    params: &TestParams,
    shapes: &ParityShapes,
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    BR: ParityBackend,
    BT: ParityBackend,
    BR::OwnedBuf: HostDataMut,
    Module<BR>: GLWEShift<BR> + GLWEMulXpMinusOne<BR>,
    Module<BT>: GLWEShift<BT> + GLWEMulXpMinusOne<BT>,
    ScratchOwned<BR>: ScratchOwnedAlloc<BR> + ScratchOwnedBorrow<BR>,
    ScratchOwned<BT>: ScratchOwnedAlloc<BT> + ScratchOwnedBorrow<BT>,
{
    for k in [0, 1, params.base2k - 1, params.base2k, 5 * params.base2k + 1] {
        compare(
            params,
            shapes,
            module_ref,
            module_test,
            "glwe_rsh",
            47,
            (0, 0),
            move |m, res, _, _, _| {
                let mut s = poisoned_scratch::<BR>(m.glwe_shift_tmp_bytes(res.size()));
                m.glwe_rsh(k, res, &mut s.borrow());
            },
            move |m, res, _, _, _| {
                let mut s = poisoned_scratch::<BT>(m.glwe_shift_tmp_bytes(res.size()));
                m.glwe_rsh(k, res, &mut s.borrow());
            },
        );
    }
    for k in [0, 1, params.base2k - 1, params.base2k, 5 * params.base2k + 1] {
        compare(
            params,
            shapes,
            module_ref,
            module_test,
            "glwe_lsh_assign",
            47,
            (0, 0),
            move |m, res, _, _, _| {
                let mut s = poisoned_scratch::<BR>(m.glwe_shift_tmp_bytes(res.size()));
                m.glwe_lsh_assign(res, k, &mut s.borrow());
            },
            move |m, res, _, _, _| {
                let mut s = poisoned_scratch::<BT>(m.glwe_shift_tmp_bytes(res.size()));
                m.glwe_lsh_assign(res, k, &mut s.borrow());
            },
        );
    }
    for k in [0, 1, params.base2k - 1, params.base2k, 5 * params.base2k + 1] {
        compare(
            params,
            shapes,
            module_ref,
            module_test,
            "glwe_lsh",
            47,
            (0, 0),
            move |m, res, a, _, _| {
                let mut s = poisoned_scratch::<BR>(m.glwe_shift_tmp_bytes(res.size()));
                m.glwe_lsh(res, a, k, &mut s.borrow());
            },
            move |m, res, a, _, _| {
                let mut s = poisoned_scratch::<BT>(m.glwe_shift_tmp_bytes(res.size()));
                m.glwe_lsh(res, a, k, &mut s.borrow());
            },
        );
    }
    for k in [0, 1, params.base2k - 1, params.base2k, 5 * params.base2k + 1] {
        compare(
            params,
            shapes,
            module_ref,
            module_test,
            "glwe_lsh_add",
            47,
            (0, 0),
            move |m, res, a, _, _| {
                let mut s = poisoned_scratch::<BR>(m.glwe_shift_tmp_bytes(res.size()));
                m.glwe_lsh_add(res, a, k, &mut s.borrow());
            },
            move |m, res, a, _, _| {
                let mut s = poisoned_scratch::<BT>(m.glwe_shift_tmp_bytes(res.size()));
                m.glwe_lsh_add(res, a, k, &mut s.borrow());
            },
        );
    }
    for k in [0, 1, params.base2k - 1, params.base2k, 5 * params.base2k + 1] {
        compare(
            params,
            shapes,
            module_ref,
            module_test,
            "glwe_lsh_sub",
            47,
            (0, 0),
            move |m, res, a, _, _| {
                let mut s = poisoned_scratch::<BR>(m.glwe_shift_tmp_bytes(res.size()));
                m.glwe_lsh_sub(res, a, k, &mut s.borrow());
            },
            move |m, res, a, _, _| {
                let mut s = poisoned_scratch::<BT>(m.glwe_shift_tmp_bytes(res.size()));
                m.glwe_lsh_sub(res, a, k, &mut s.borrow());
            },
        );
    }
    for k in [-5i64, 0, 1, params.n as i64, 2 * params.n as i64] {
        compare(
            params,
            shapes,
            module_ref,
            module_test,
            "glwe_mul_xp_minus_one",
            53,
            (0, 0),
            move |m, res, a, _, _| m.glwe_mul_xp_minus_one(k, res, a),
            move |m, res, a, _, _| m.glwe_mul_xp_minus_one(k, res, a),
        );
        compare(
            params,
            shapes,
            module_ref,
            module_test,
            "glwe_mul_xp_minus_one_assign",
            53,
            (0, 0),
            move |m, res, _, _, _| {
                let mut s = poisoned_scratch::<BR>(m.glwe_mul_xp_minus_one_assign_tmp_bytes(res.size()));
                m.glwe_mul_xp_minus_one_assign(k, res, &mut s.borrow());
            },
            move |m, res, _, _, _| {
                let mut s = poisoned_scratch::<BT>(m.glwe_mul_xp_minus_one_assign_tmp_bytes(res.size()));
                m.glwe_mul_xp_minus_one_assign(k, res, &mut s.borrow());
            },
        );
    }
    // Multiplication by X^p - 1 is raw-limb arithmetic, even when the two
    // layouts label those limbs with different radices. Destination metadata
    // and extension/truncation are checked against a direct polynomial oracle.
    let mut source = Source::new([73; 32]);
    let n = params.n;
    for &rank in &shapes.ranks {
        let a_infos = GLWELayout {
            n: (n as u32).into(),
            base2k: (params.base2k as u32).into(),
            k: (2 * params.base2k as u32 + 1).into(),
            rank: (rank as u32).into(),
        };
        let a_ref = ref_glwe(module_ref, &a_infos, &mut source);
        let mut a_test = module_test.glwe_alloc_from_infos(&a_infos);
        a_ref.transfer_into(&mut a_test);
        for res_limbs in [1, 4] {
            let radix = params.base2k - 1;
            let res_infos = GLWELayout {
                base2k: (radix as u32).into(),
                k: ((res_limbs * radix) as u32).into(),
                ..a_infos
            };
            for power in [-5_i64, 0, n as i64 + 1] {
                let mut expected = module_ref.glwe_alloc_from_infos(&res_infos);
                for col in 0..expected.data.cols() {
                    for limb in 0..expected.data.size() {
                        let out = expected.data.at_mut(col, limb);
                        out.fill(0);
                        if limb < a_ref.data.size() {
                            for (i, &value) in a_ref.data.at(col, limb).iter().enumerate() {
                                let j = (i as i64 + power).rem_euclid(2 * n as i64) as usize;
                                out[j % n] += if j < n { value } else { -value };
                                out[i] -= value;
                            }
                        }
                    }
                }
                let mut res_ref = ref_glwe(module_ref, &res_infos, &mut source);
                let mut res_test = module_test.glwe_alloc_from_infos(&res_infos);
                res_ref.transfer_into(&mut res_test);
                module_ref.glwe_mul_xp_minus_one(power, &mut res_ref, &a_ref);
                module_test.glwe_mul_xp_minus_one(power, &mut res_test, &a_test);
                assert_eq!(res_ref.base2k(), res_infos.base2k);
                assert_eq!(res_test.base2k(), res_infos.base2k);
                assert_eq!(res_ref.k(), res_infos.k);
                assert_eq!(res_test.k(), res_infos.k);
                let mut have = module_ref.glwe_alloc_from_infos(&res_infos);
                res_test.transfer_into(&mut have);
                assert_eq!(res_ref, expected, "raw X^p - 1: reference radix and limb tails");
                assert_eq!(have, expected, "raw X^p - 1: backend radix and limb tails");
            }
        }
    }
}

/// Plaintext and scalar products, and the summed product of prepared operands, compare canonical ciphertexts at
/// conversion boundaries.
pub fn test_glwe_multiplication_parity<BR, BT>(
    params: &TestParams,
    shapes: &ParityShapes,
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    BR: ParityBackend,
    BT: ParityBackend,
    BR::OwnedBuf: HostDataMut,
    Module<BR>: GLWEMulConst<BR>
        + GLWEMulPlain<BR>
        + GLWEMulPreparedPlainSum<BR>
        + GLWEAdd<BR>
        + CnvPVecAlloc<BR>
        + CnvOps<BR>
        + GLWENormalize<BR>
        + GLWEBytesOf<BR>
        + VecZnxAutomorphism<BR>
        + VecZnxDftAlloc<BR>
        + VecZnxIdftNormalize<BR>
        + VecZnxIdftNormalizeTmpBytes,
    Module<BT>: GLWEMulConst<BT>
        + GLWEMulPlain<BT>
        + GLWEMulPreparedPlainSum<BT>
        + CnvPVecAlloc<BT>
        + CnvOps<BT>
        + GLWENormalize<BT>
        + GLWEBytesOf<BT>
        + VecZnxDftAlloc<BT>
        + VecZnxIdftNormalize<BT>
        + VecZnxIdftNormalizeTmpBytes,
    ScratchOwned<BR>: ScratchOwnedAlloc<BR> + ScratchOwnedBorrow<BR>,
    ScratchOwned<BT>: ScratchOwnedAlloc<BT> + ScratchOwnedBorrow<BT>,
{
    let mut source = Source::new([59; 32]);
    for &rank in &shapes.ranks {
        for k in [params.base2k, 3 * params.base2k - 1] {
            let infos = GLWELayout {
                n: (params.n as u32).into(),
                base2k: (params.base2k as u32).into(),
                k: (k as u32).into(),
                rank: (rank as u32).into(),
            };
            let a_ref = ref_glwe(module_ref, &infos, &mut source);
            let mut a_test = module_test.glwe_alloc_from_infos(&infos);
            a_ref.transfer_into(&mut a_test);
            let mut plain_ref = module_ref.glwe_plaintext_alloc_from_infos(&infos);
            module_ref.fill_glwe_from_source(&mut plain_ref, &mut source);
            let mut plain_test = module_test.glwe_plaintext_alloc_from_infos(&infos);
            plain_ref.transfer_into(&mut plain_test);
            for offset in [0, params.base2k - 1, params.base2k] {
                for variant in 0..4 {
                    let mut out_ref = ref_glwe(module_ref, &infos, &mut source);
                    let mut out_test = module_test.glwe_alloc_from_infos(&infos);
                    if variant % 2 == 1 {
                        a_ref.transfer_into(&mut out_ref);
                    }
                    out_ref.transfer_into(&mut out_test);
                    let mut sr = poisoned_scratch::<BR>(if variant < 2 {
                        module_ref.glwe_mul_plain_tmp_bytes(&out_ref, &a_ref, &plain_ref)
                    } else {
                        module_ref.glwe_mul_const_tmp_bytes(&out_ref, &a_ref, &plain_ref)
                    });
                    let mut st = poisoned_scratch::<BT>(if variant < 2 {
                        module_test.glwe_mul_plain_tmp_bytes(&out_test, &a_test, &plain_test)
                    } else {
                        module_test.glwe_mul_const_tmp_bytes(&out_test, &a_test, &plain_test)
                    });
                    match variant {
                        0 => {
                            module_ref.glwe_mul_plain(offset, &mut out_ref, &a_ref, &plain_ref, &mut sr.borrow());
                            module_test.glwe_mul_plain(offset, &mut out_test, &a_test, &plain_test, &mut st.borrow());
                        }
                        1 => {
                            module_ref.glwe_mul_plain_assign(offset, &mut out_ref, &plain_ref, &mut sr.borrow());
                            module_test.glwe_mul_plain_assign(offset, &mut out_test, &plain_test, &mut st.borrow());
                        }
                        2 => {
                            module_ref.glwe_mul_const(offset, &mut out_ref, &a_ref, &plain_ref, params.n - 1, &mut sr.borrow());
                            module_test.glwe_mul_const(
                                offset,
                                &mut out_test,
                                &a_test,
                                &plain_test,
                                params.n - 1,
                                &mut st.borrow(),
                            );
                        }
                        _ => {
                            module_ref.glwe_mul_const_assign(offset, &mut out_ref, &plain_ref, params.n - 1, &mut sr.borrow());
                            module_test.glwe_mul_const_assign(offset, &mut out_test, &plain_test, params.n - 1, &mut st.borrow());
                        }
                    }
                    let mut have = module_ref.glwe_alloc_from_infos(&infos);
                    out_test.transfer_into(&mut have);
                    assert_glwe_eq!(
                        out_ref,
                        have,
                        "multiplication variant={variant} rank={rank} k={k} offset={offset}"
                    );

                    let mut twin = module_test.glwe_alloc_from_infos(&infos);
                    unnormalized_twin::<BR, BT>(&a_ref, &mut twin);
                    let mut st = poisoned_scratch::<BT>(if variant < 2 {
                        module_test.glwe_mul_plain_tmp_bytes(&out_test, &a_test, &plain_test)
                    } else {
                        module_test.glwe_mul_const_tmp_bytes(&out_test, &a_test, &plain_test)
                    });
                    let last = params.n - 1;
                    match variant {
                        0 => module_test.glwe_mul_plain(offset, &mut out_test, &twin, &plain_test, &mut st.borrow()),
                        1 => module_test.glwe_mul_plain_assign(offset, &mut twin, &plain_test, &mut st.borrow()),
                        2 => module_test.glwe_mul_const(offset, &mut out_test, &twin, &plain_test, last, &mut st.borrow()),
                        _ => module_test.glwe_mul_const_assign(offset, &mut twin, &plain_test, last, &mut st.borrow()),
                    }
                    if variant % 2 == 1 {
                        twin.transfer_into(&mut have);
                    } else {
                        out_test.transfer_into(&mut have);
                    }
                    assert_glwe_eq!(
                        out_ref,
                        have,
                        "multiplication variant={variant}, unnormalized operand: rank={rank} k={k} offset={offset}"
                    );
                }
            }
            test_glwe_mul_prepared_plain_sum_case(params, &infos, module_ref, module_test, &mut source);
        }
    }
}

/// The summed product of prepared ciphertexts by prepared plaintexts read through Galois elements, each call on exactly
/// its scratch: one term through `1` is `glwe_mul_plain` of the two sources and one through `5` is `glwe_mul_plain` of
/// the ciphertext by the plaintext's automorphism, bit for bit; three terms through `1`, `5` and `-1` agree across
/// backends, the two sides preparing under swapped hints, the test side from unnormalized twins, and equal on each
/// backend the DFT forms, the first term then the other two added, finished per column, the reference side on the
/// accumulator the sum spans and the test side on one limb more; an even element panics before `res` changes.
fn test_glwe_mul_prepared_plain_sum_case<BR, BT>(
    params: &TestParams,
    infos: &GLWELayout,
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
    source: &mut Source,
) where
    BR: ParityBackend,
    BT: ParityBackend,
    BR::OwnedBuf: HostDataMut,
    Module<BR>: GLWEMulPlain<BR>
        + GLWEMulPreparedPlainSum<BR>
        + GLWEAdd<BR>
        + CnvPVecAlloc<BR>
        + CnvOps<BR>
        + GLWENormalize<BR>
        + GLWEBytesOf<BR>
        + VecZnxAutomorphism<BR>
        + VecZnxDftAlloc<BR>
        + VecZnxIdftNormalize<BR>
        + VecZnxIdftNormalizeTmpBytes,
    Module<BT>: GLWEMulPreparedPlainSum<BT>
        + CnvPVecAlloc<BT>
        + CnvOps<BT>
        + GLWENormalize<BT>
        + GLWEBytesOf<BT>
        + VecZnxDftAlloc<BT>
        + VecZnxIdftNormalize<BT>
        + VecZnxIdftNormalizeTmpBytes,
    ScratchOwned<BR>: ScratchOwnedAlloc<BR> + ScratchOwnedBorrow<BR>,
    ScratchOwned<BT>: ScratchOwnedAlloc<BT> + ScratchOwnedBorrow<BT>,
{
    const GALOIS: [i64; 3] = [1, 5, -1];
    let (n, k, rank, base2k) = (infos.n.as_usize(), infos.k.as_usize(), infos.rank, infos.base2k);
    let (cols, size) = (rank.as_usize() + 1, k.div_ceil(base2k.as_usize()));
    let (hint, other) = crate::test_suite::prepare_hints(k);
    let (mut cts, mut plains) = (Vec::new(), Vec::new());
    let (mut a_ref, mut a_test, mut b_ref, mut b_test) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for _ in GALOIS {
        let ct = ref_glwe(module_ref, infos, source);
        let mut twin = module_test.glwe_alloc_from_infos(infos);
        unnormalized_twin::<BR, BT>(&ct, &mut twin);
        let mut plain = module_ref.glwe_plaintext_alloc_from_infos(infos);
        module_ref.fill_glwe_from_source(&mut plain, source);
        let mut plain_test = module_test.glwe_plaintext_alloc_from_infos(infos);
        plain.transfer_into(&mut plain_test);

        let mut left_ref = module_ref.cnv_pvec_left_alloc(n, cols, size, hint);
        let mut left_test = module_test.cnv_pvec_left_alloc(n, cols, size, other);
        let mut right_ref = module_ref.cnv_pvec_right_alloc(n, 1, size, other);
        let mut right_test = module_test.cnv_pvec_right_alloc(n, 1, size, hint);
        let bytes = glwe_prepare_left_tmp_bytes(module_ref, size, infos);
        glwe_prepare_left(module_ref, &mut left_ref, &ct, k, &mut poisoned_scratch::<BR>(bytes).borrow());
        let bytes = glwe_prepare_left_tmp_bytes(module_test, size, infos);
        glwe_prepare_left(
            module_test,
            &mut left_test,
            &twin,
            k,
            &mut poisoned_scratch::<BT>(bytes).borrow(),
        );
        let bytes = glwe_prepare_right_tmp_bytes(module_ref, size, &plain);
        glwe_prepare_right(
            module_ref,
            &mut right_ref,
            &plain,
            k,
            &mut poisoned_scratch::<BR>(bytes).borrow(),
        );
        let bytes = glwe_prepare_right_tmp_bytes(module_test, size, &plain_test);
        glwe_prepare_right(
            module_test,
            &mut right_test,
            &plain_test,
            k,
            &mut poisoned_scratch::<BT>(bytes).borrow(),
        );

        cts.push(ct);
        plains.push(plain);
        a_ref.push(left_ref);
        a_test.push(left_test);
        b_ref.push(right_ref);
        b_test.push(right_test);
    }
    let mut rotated = module_ref.glwe_plaintext_alloc_from_infos(infos);
    module_ref.vec_znx_automorphism(
        5,
        &mut vec_znx_backend_mut::<BR>(&mut rotated.data),
        0,
        &vec_znx_backend_ref::<BR>(&plains[0].data),
        0,
    );
    let (a_refs, b_refs): (Vec<&_>, Vec<&_>) = (a_ref.iter().collect(), b_ref.iter().collect());
    let (a_tests, b_tests): (Vec<&_>, Vec<&_>) = (a_test.iter().collect(), b_test.iter().collect());

    for offset in [0, params.base2k - 1, params.base2k] {
        for (g, plain) in [(1, &plains[0]), (5, &rotated)] {
            let mut want = module_ref.glwe_alloc_from_infos(infos);
            let mut have = module_ref.glwe_alloc_from_infos(infos);
            let bytes = module_ref.glwe_mul_plain_tmp_bytes(&want, &cts[0], plain);
            module_ref.glwe_mul_plain(offset, &mut want, &cts[0], plain, &mut poisoned_scratch::<BR>(bytes).borrow());
            let bytes = module_ref.glwe_mul_prepared_plain_sum_tmp_bytes(&have, size, size);
            module_ref.glwe_mul_prepared_plain_sum(
                offset,
                &mut have,
                &a_refs[..1],
                size,
                &b_refs[..1],
                size,
                &[g],
                base2k,
                &mut poisoned_scratch::<BR>(bytes).borrow(),
            );
            assert_glwe_eq!(
                want,
                have,
                "glwe_mul_prepared_plain_sum versus glwe_mul_plain: g={g} k={k} rank={rank:?} offset={offset}"
            );
        }

        let mut res_ref = ref_glwe(module_ref, infos, source);
        let mut res_test = module_test.glwe_alloc_from_infos(infos);
        res_ref.transfer_into(&mut res_test);
        let bytes = module_ref.glwe_mul_prepared_plain_sum_tmp_bytes(&res_ref, size, size);
        module_ref.glwe_mul_prepared_plain_sum(
            offset,
            &mut res_ref,
            &a_refs,
            size,
            &b_refs,
            size,
            &GALOIS,
            base2k,
            &mut poisoned_scratch::<BR>(bytes).borrow(),
        );
        let bytes = module_test.glwe_mul_prepared_plain_sum_tmp_bytes(&res_test, size, size);
        module_test.glwe_mul_prepared_plain_sum(
            offset,
            &mut res_test,
            &a_tests,
            size,
            &b_tests,
            size,
            &GALOIS,
            base2k,
            &mut poisoned_scratch::<BT>(bytes).borrow(),
        );
        let mut have = module_ref.glwe_alloc_from_infos(infos);
        res_test.transfer_into(&mut have);
        assert_glwe_eq!(
            res_ref,
            have,
            "glwe_mul_prepared_plain_sum, three terms: k={k} rank={rank:?} offset={offset}"
        );

        let (cnv_offset_hi, _) = cnv_offset_to_limb_offset(offset, base2k.as_usize());
        let mut dft_ref = ref_glwe(module_ref, infos, source);
        let mut dft_test = module_test.glwe_alloc_from_infos(infos);
        dft_ref.transfer_into(&mut dft_test);
        plain_sum_dft(
            module_ref,
            offset,
            &mut dft_ref,
            &a_refs,
            &b_refs,
            size,
            2 * size - cnv_offset_hi,
            base2k,
        );
        plain_sum_dft(
            module_test,
            offset,
            &mut dft_test,
            &a_tests,
            &b_tests,
            size,
            2 * size - cnv_offset_hi + 1,
            base2k,
        );
        assert_glwe_eq!(
            res_ref,
            dft_ref,
            "glwe_mul_prepared_plain_sum_dft versus glwe_mul_prepared_plain_sum: k={k} rank={rank:?} offset={offset}"
        );
        dft_test.transfer_into(&mut have);
        assert_glwe_eq!(
            res_ref,
            have,
            "glwe_mul_prepared_plain_sum_dft, three terms: k={k} rank={rank:?} offset={offset}"
        );
    }

    // The accumulate form, at offsets whose fraction of a limb is negative (below one limb), zero and positive: into a
    // zero `res` it is the sum itself, bit for bit; from any `res` it is the sum rounded losslessly, one limb wider than
    // its product, plus `res`, rounded once at `res.k()`; and both backends agree.
    for offset in [0, params.base2k - 1, params.base2k, params.base2k + 3, 2 * params.base2k + 1] {
        let mut sum = module_ref.glwe_alloc_from_infos(infos);
        let bytes = module_ref.glwe_mul_prepared_plain_sum_tmp_bytes(&sum, size, size);
        module_ref.glwe_mul_prepared_plain_sum(
            offset,
            &mut sum,
            &a_refs,
            size,
            &b_refs,
            size,
            &GALOIS,
            base2k,
            &mut poisoned_scratch::<BR>(bytes).borrow(),
        );
        let mut from_zero = module_ref.glwe_alloc_from_infos(infos);
        let bytes = module_ref.glwe_mul_prepared_plain_sum_add_tmp_bytes(&from_zero, size, size);
        module_ref.glwe_mul_prepared_plain_sum_add(
            offset,
            &mut from_zero,
            &a_refs,
            size,
            &b_refs,
            size,
            &GALOIS,
            base2k,
            &mut poisoned_scratch::<BR>(bytes).borrow(),
        );
        assert_glwe_eq!(
            sum,
            from_zero,
            "glwe_mul_prepared_plain_sum_add into zero versus the sum: k={k} rank={rank:?} offset={offset}"
        );

        let mut acc_ref = ref_glwe(module_ref, infos, source);
        let mut acc_test = module_test.glwe_alloc_from_infos(infos);
        acc_ref.transfer_into(&mut acc_test);
        let mut composed = module_ref.glwe_alloc_from_infos(infos);
        {
            let wide = GLWELayout {
                k: TorusPrecision(((2 * size + 1) * base2k.as_usize()) as u32),
                ..*infos
            };
            let mut lossless = module_ref.glwe_alloc_from_infos(&wide);
            let bytes = module_ref.glwe_mul_prepared_plain_sum_tmp_bytes(&lossless, size, size);
            module_ref.glwe_mul_prepared_plain_sum(
                offset,
                &mut lossless,
                &a_refs,
                size,
                &b_refs,
                size,
                &GALOIS,
                base2k,
                &mut poisoned_scratch::<BR>(bytes).borrow(),
            );
            module_ref.glwe_add_assign(&mut lossless, &acc_ref);
            let bytes = module_ref.glwe_normalize_tmp_bytes();
            module_ref.glwe_normalize(&mut composed, &lossless, &mut poisoned_scratch::<BR>(bytes).borrow());
        }
        let bytes = module_ref.glwe_mul_prepared_plain_sum_add_tmp_bytes(&acc_ref, size, size);
        module_ref.glwe_mul_prepared_plain_sum_add(
            offset,
            &mut acc_ref,
            &a_refs,
            size,
            &b_refs,
            size,
            &GALOIS,
            base2k,
            &mut poisoned_scratch::<BR>(bytes).borrow(),
        );
        let bytes = module_test.glwe_mul_prepared_plain_sum_add_tmp_bytes(&acc_test, size, size);
        module_test.glwe_mul_prepared_plain_sum_add(
            offset,
            &mut acc_test,
            &a_tests,
            size,
            &b_tests,
            size,
            &GALOIS,
            base2k,
            &mut poisoned_scratch::<BT>(bytes).borrow(),
        );
        assert_glwe_eq!(
            composed,
            acc_ref,
            "glwe_mul_prepared_plain_sum_add versus the lossless sum plus res: k={k} rank={rank:?} offset={offset}"
        );
        let mut have = module_ref.glwe_alloc_from_infos(infos);
        acc_test.transfer_into(&mut have);
        assert_glwe_eq!(
            acc_ref,
            have,
            "glwe_mul_prepared_plain_sum_add, three terms: k={k} rank={rank:?} offset={offset}"
        );
    }

    let mut res = ref_glwe(module_ref, infos, source);
    let mut before = module_ref.glwe_alloc_from_infos(infos);
    res.transfer_into(&mut before);
    let bytes = module_ref.glwe_mul_prepared_plain_sum_tmp_bytes(&res, size, size);
    let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        module_ref.glwe_mul_prepared_plain_sum(
            0,
            &mut res,
            &a_refs,
            size,
            &b_refs,
            size,
            &[1, 4, 5],
            base2k,
            &mut poisoned_scratch::<BR>(bytes).borrow(),
        )
    }))
    .is_err();
    assert!(panicked, "glwe_mul_prepared_plain_sum accepted an even Galois element");
    assert_eq!(
        res, before,
        "glwe_mul_prepared_plain_sum changed res before rejecting an even Galois element"
    );
}

/// `res` = the three-term sum by [`GLWEMulPreparedPlainSum::glwe_mul_prepared_plain_sum_dft`] of the first term,
/// [`GLWEMulPreparedPlainSum::glwe_mul_prepared_plain_sum_dft_add`] of the other two, into an accumulator of `acc_size`
/// limbs, then one `vec_znx_idft_normalize` per column, each call on exactly its scratch.
#[allow(clippy::too_many_arguments)]
fn plain_sum_dft<BE, R, AP, BP>(
    module: &Module<BE>,
    cnv_offset: usize,
    res: &mut R,
    a: &[&AP],
    b: &[&BP],
    size: usize,
    acc_size: usize,
    base2k: Base2K,
) where
    BE: Backend,
    Module<BE>: GLWEMulPreparedPlainSum<BE> + VecZnxDftAlloc<BE> + VecZnxIdftNormalize<BE> + VecZnxIdftNormalizeTmpBytes,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    R: GLWEToBackendMut<BE> + GLWEInfos,
    AP: CnvPVecLToBackendRef<BE>,
    BP: CnvPVecRToBackendRef<BE>,
{
    const GALOIS: [i64; 3] = [1, 5, -1];
    let (_, cnv_offset_lo) = cnv_offset_to_limb_offset(cnv_offset, base2k.as_usize());
    let cols = res.rank().as_usize() + 1;
    let mut acc = module.vec_znx_dft_alloc(res.n().as_usize(), cols, acc_size);
    let bytes = module.glwe_mul_prepared_plain_sum_dft_tmp_bytes(acc_size, size, size);
    module.glwe_mul_prepared_plain_sum_dft(
        cnv_offset,
        &mut acc.to_backend_mut(),
        &a[..1],
        size,
        &b[..1],
        size,
        &GALOIS[..1],
        base2k,
        &mut poisoned_scratch::<BE>(bytes).borrow(),
    );
    let bytes = module.glwe_mul_prepared_plain_sum_dft_add_tmp_bytes(acc_size, size, size);
    module.glwe_mul_prepared_plain_sum_dft_add(
        cnv_offset,
        &mut acc.to_backend_mut(),
        &a[1..],
        size,
        &b[1..],
        size,
        &GALOIS[1..],
        base2k,
        &mut poisoned_scratch::<BE>(bytes).borrow(),
    );
    let (res_base2k, res_k) = (res.base2k().as_usize(), res.k().as_usize());
    let mut scratch = poisoned_scratch::<BE>(module.vec_znx_idft_normalize_tmp_bytes(res.size(), acc.size()));
    for col in 0..cols {
        module.vec_znx_idft_normalize(
            &mut res.to_backend_mut().data,
            res_base2k,
            res_k,
            cnv_offset_lo,
            col,
            &mut acc.to_backend_mut(),
            col,
            base2k.as_usize(),
            None,
            &mut scratch.borrow(),
        );
    }
    res.set_canonical(true);
}

/// GGSW rotation visits every row, including the in-place mutation variant.
pub fn test_ggsw_rotate_parity<BR, BT>(
    params: &TestParams,
    shapes: &ParityShapes,
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    BR: ParityBackend,
    BT: ParityBackend,
    BR::OwnedBuf: HostDataMut,
    Module<BR>: GGSWRotate<BR>,
    Module<BT>: GGSWRotate<BT>,
    ScratchOwned<BR>: ScratchOwnedAlloc<BR> + ScratchOwnedBorrow<BR>,
    ScratchOwned<BT>: ScratchOwnedAlloc<BT> + ScratchOwnedBorrow<BT>,
{
    let mut source = Source::new([61; 32]);
    for &rank in &shapes.ranks {
        let infos = crate::layouts::GGSWLayout {
            n: (params.n as u32).into(),
            base2k: (params.base2k as u32).into(),
            dnum: 3u32.into(),
            k_digit: crate::layouts::k_digit_of(params.base2k, 2usize),
            k_aux: (2 * params.base2k as u32 + 1).into(),
            rank: (rank as u32).into(),
        };
        let mut a_ref = module_ref.ggsw_alloc_from_infos(&infos);
        for row in 0..infos.dnum.as_usize() {
            for col in 0..rank + 1 {
                module_ref.fill_glwe_from_source(&mut a_ref.at_view_mut(row, col), &mut source);
            }
        }
        let mut a_test = module_test.ggsw_alloc_from_infos(&infos);
        a_ref.transfer_into(&mut a_test);
        for k in [-5i64, 0, 1, params.n as i64, 2 * params.n as i64 + 1] {
            let mut out_ref = module_ref.ggsw_alloc_from_infos(&infos);
            let mut out_test = module_test.ggsw_alloc_from_infos(&infos);
            module_ref.ggsw_rotate(k, &mut out_ref, &a_ref);
            module_test.ggsw_rotate(k, &mut out_test, &a_test);
            let mut have = module_ref.ggsw_alloc_from_infos(&infos);
            out_test.transfer_into(&mut have);
            assert_eval_eq!(out_ref, have, "ggsw_rotate rank={rank} k={k}");
            a_ref.transfer_into(&mut out_ref);
            a_ref.transfer_into(&mut out_test);
            module_ref.ggsw_rotate_assign(
                k,
                &mut out_ref,
                &mut poisoned_scratch::<BR>(module_ref.ggsw_rotate_tmp_bytes()).borrow(),
            );
            module_test.ggsw_rotate_assign(
                k,
                &mut out_test,
                &mut poisoned_scratch::<BT>(module_test.ggsw_rotate_tmp_bytes()).borrow(),
            );
            out_test.transfer_into(&mut have);
            assert_eval_eq!(out_ref, have, "ggsw_rotate_assign rank={rank} k={k}");
        }
    }
}
