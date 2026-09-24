//! Keyless GLWE operation parity.
//!
//! These take no prepared key, so each test is fill, upload, run on both,
//! compare.

use poulpy_hal::{
    api::{ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{HostDataMut, Module, ScratchOwned, ZnxView, ZnxViewMut},
    source::Source,
    test_suite::TestParams,
};

use crate::{
    GGSWRotate, GLWEAdd, GLWECopy, GLWEMaskFill, GLWEMulConst, GLWEMulPlain, GLWEMulXpMinusOne, GLWENegate, GLWENormalize,
    GLWERotate, GLWEShift, GLWESub, GLWETensoring, GLWEZero,
    api::TransferInto,
    layouts::{Base2K, Degree, GGSWAtViewMut, GLWELayout, LWEInfos, ModuleCoreAlloc, Rank, TorusPrecision},
    test_suite::parity::{ParityBackend, ParityShapes, poisoned_scratch, ref_glwe},
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
    Module<BR>: ModuleCoreAlloc<OwnedBuf = BR::OwnedBuf, ZnxWord = i64> + GLWEMaskFill<BR>,
    Module<BT>: ModuleCoreAlloc<OwnedBuf = BT::OwnedBuf, ZnxWord = i64>,
    ScratchOwned<BR>: ScratchOwnedAlloc<BR> + ScratchOwnedBorrow<BR>,
    ScratchOwned<BT>: ScratchOwnedAlloc<BT> + ScratchOwnedBorrow<BT>,
    FR: Fn(
        &Module<BR>,
        &mut crate::layouts::BackendGLWE<BR>,
        &crate::layouts::BackendGLWE<BR>,
        &crate::layouts::BackendGLWE<BR>,
        &mut ScratchOwned<BR>,
    ),
    FT: Fn(
        &Module<BT>,
        &mut crate::layouts::BackendGLWE<BT>,
        &crate::layouts::BackendGLWE<BT>,
        &crate::layouts::BackendGLWE<BT>,
        &mut ScratchOwned<BT>,
    ),
{
    assert_eq!(module_ref.n(), module_test.n());

    let n = module_ref.n() as u32;
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

            let mut have = module_ref.glwe_alloc_from_infos(&res_infos);
            res_test.transfer_into(&mut have);
            assert_eq!(
                res_ref, have,
                "{label}: a_k={:?} res_k={:?} rank={:?}",
                a_infos.k, res_infos.k, res_infos.rank
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
    Module<BR>: GLWEAdd<BR> + GLWEMaskFill<BR>,
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
    Module<BR>: GLWESub<BR> + GLWEMaskFill<BR>,
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
                n: (module_ref.n() as u32).into(),
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
    Module<BR>: GLWENegate<BR> + GLWEMaskFill<BR>,
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
    Module<BR>: GLWENormalize<BR> + GLWEMaskFill<BR>,
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
    Module<BR>: GLWERotate<BR> + GLWEMaskFill<BR>,
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
    for k in [-5i64, 0, 1, module_ref.n() as i64, 2 * module_ref.n() as i64 + 7] {
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
    Module<BR>: GLWETensoring<BR> + ModuleCoreAlloc<OwnedBuf = BR::OwnedBuf, ZnxWord = i64> + GLWEMaskFill<BR>,
    Module<BT>: GLWETensoring<BT> + ModuleCoreAlloc<OwnedBuf = BT::OwnedBuf, ZnxWord = i64>,
    ScratchOwned<BR>: ScratchOwnedAlloc<BR> + ScratchOwnedBorrow<BR>,
    ScratchOwned<BT>: ScratchOwnedAlloc<BT> + ScratchOwnedBorrow<BT>,
{
    assert_eq!(module_ref.n(), module_test.n());
    let n = module_ref.n() as u32;
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
/// retaining the full parameter sweep at smaller degrees.
pub fn test_glwe_tensor_parity_for_layout<BR, BT>(
    layout: &GLWELayout,
    offsets: &[usize],
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    BR: ParityBackend,
    BT: ParityBackend,
    BR::OwnedBuf: HostDataMut,
    Module<BR>: GLWETensoring<BR> + ModuleCoreAlloc<OwnedBuf = BR::OwnedBuf, ZnxWord = i64> + GLWEMaskFill<BR>,
    Module<BT>: GLWETensoring<BT> + ModuleCoreAlloc<OwnedBuf = BT::OwnedBuf, ZnxWord = i64>,
    ScratchOwned<BR>: ScratchOwnedAlloc<BR> + ScratchOwnedBorrow<BR>,
    ScratchOwned<BT>: ScratchOwnedAlloc<BT> + ScratchOwnedBorrow<BT>,
{
    assert!(!offsets.is_empty(), "tensor parity requires at least one offset");
    test_glwe_tensor_parity_case(layout, offsets, module_ref, module_test, &mut Source::new([29u8; 32]));
}

fn test_glwe_tensor_parity_case<BR, BT>(
    a_infos: &GLWELayout,
    offsets: &[usize],
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
    source: &mut Source,
) where
    BR: ParityBackend,
    BT: ParityBackend,
    BR::OwnedBuf: HostDataMut,
    Module<BR>: GLWETensoring<BR> + ModuleCoreAlloc<OwnedBuf = BR::OwnedBuf, ZnxWord = i64> + GLWEMaskFill<BR>,
    Module<BT>: GLWETensoring<BT> + ModuleCoreAlloc<OwnedBuf = BT::OwnedBuf, ZnxWord = i64>,
    ScratchOwned<BR>: ScratchOwnedAlloc<BR> + ScratchOwnedBorrow<BR>,
    ScratchOwned<BT>: ScratchOwnedAlloc<BT> + ScratchOwnedBorrow<BT>,
{
    assert_eq!(module_ref.n(), module_test.n());
    assert_eq!(module_ref.n(), a_infos.n.as_usize());
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
    for &cnv_offset in offsets {
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
    }
}

/// Zeroing and copying retain destination metadata, including partial limbs.
pub fn test_glwe_copy_zero_parity<BR, BT>(
    params: &TestParams,
    shapes: &ParityShapes,
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    BR: ParityBackend,
    BT: ParityBackend,
    BR::OwnedBuf: HostDataMut,
    Module<BR>: GLWECopy<BR> + GLWEZero<BR> + GLWEMaskFill<BR>,
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
    Module<BR>: GLWEShift<BR> + GLWEMulXpMinusOne<BR> + GLWERotate<BR> + GLWEMaskFill<BR>,
    Module<BT>: GLWEShift<BT> + GLWEMulXpMinusOne<BT> + GLWERotate<BT>,
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
    for k in [-5i64, 0, 1, module_ref.n() as i64, 2 * module_ref.n() as i64] {
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
            (module_ref.glwe_rotate_tmp_bytes(), module_test.glwe_rotate_tmp_bytes()),
            move |m, res, _, _, s| m.glwe_mul_xp_minus_one_assign(k, res, &mut s.borrow()),
            move |m, res, _, _, s| m.glwe_mul_xp_minus_one_assign(k, res, &mut s.borrow()),
        );
    }
    // Multiplication by X^p - 1 is raw-limb arithmetic, even when the two
    // layouts label those limbs with different radices. Destination metadata
    // and extension/truncation are checked against a direct polynomial oracle.
    let mut source = Source::new([73; 32]);
    let n = module_ref.n();
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

/// Plaintext and scalar products compare canonical ciphertexts at conversion boundaries.
pub fn test_glwe_multiplication_parity<BR, BT>(
    params: &TestParams,
    shapes: &ParityShapes,
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    BR: ParityBackend,
    BT: ParityBackend,
    BR::OwnedBuf: HostDataMut,
    Module<BR>: GLWEMulConst<BR> + GLWEMulPlain<BR> + GLWEMaskFill<BR>,
    Module<BT>: GLWEMulConst<BT> + GLWEMulPlain<BT>,
    ScratchOwned<BR>: ScratchOwnedAlloc<BR> + ScratchOwnedBorrow<BR>,
    ScratchOwned<BT>: ScratchOwnedAlloc<BT> + ScratchOwnedBorrow<BT>,
{
    let mut source = Source::new([59; 32]);
    for &rank in &shapes.ranks {
        for k in [params.base2k, 3 * params.base2k - 1] {
            let infos = GLWELayout {
                n: (module_ref.n() as u32).into(),
                base2k: (params.base2k as u32).into(),
                k: (k as u32).into(),
                rank: (rank as u32).into(),
            };
            let a_ref = ref_glwe(module_ref, &infos, &mut source);
            let mut a_test = module_test.glwe_alloc_from_infos(&infos);
            a_ref.transfer_into(&mut a_test);
            let mut plain_ref = module_ref.glwe_plaintext_alloc(infos.base2k, infos.k);
            module_ref.fill_glwe_mask_from_source(params.base2k, &mut plain_ref, 0, 1, &mut source);
            let mut plain_test = module_test.glwe_plaintext_alloc(infos.base2k, infos.k);
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
                            module_ref.glwe_mul_const(
                                offset,
                                &mut out_ref,
                                &a_ref,
                                &plain_ref,
                                module_ref.n() - 1,
                                &mut sr.borrow(),
                            );
                            module_test.glwe_mul_const(
                                offset,
                                &mut out_test,
                                &a_test,
                                &plain_test,
                                module_test.n() - 1,
                                &mut st.borrow(),
                            );
                        }
                        _ => {
                            module_ref.glwe_mul_const_assign(
                                offset,
                                &mut out_ref,
                                &plain_ref,
                                module_ref.n() - 1,
                                &mut sr.borrow(),
                            );
                            module_test.glwe_mul_const_assign(
                                offset,
                                &mut out_test,
                                &plain_test,
                                module_test.n() - 1,
                                &mut st.borrow(),
                            );
                        }
                    }
                    let mut have = module_ref.glwe_alloc_from_infos(&infos);
                    out_test.transfer_into(&mut have);
                    assert_eq!(
                        out_ref, have,
                        "multiplication variant={variant} rank={rank} k={k} offset={offset}"
                    );
                }
            }
        }
    }
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
    Module<BR>: GGSWRotate<BR> + GLWEMaskFill<BR>,
    Module<BT>: GGSWRotate<BT>,
    ScratchOwned<BR>: ScratchOwnedAlloc<BR> + ScratchOwnedBorrow<BR>,
    ScratchOwned<BT>: ScratchOwnedAlloc<BT> + ScratchOwnedBorrow<BT>,
{
    let mut source = Source::new([61; 32]);
    for &rank in &shapes.ranks {
        let infos = crate::layouts::GGSWLayout {
            n: (module_ref.n() as u32).into(),
            base2k: (params.base2k as u32).into(),
            dnum: 3u32.into(),
            dsize: 2u32.into(),
            k_aux: (2 * params.base2k as u32 + 1).into(),
            rank: (rank as u32).into(),
        };
        let mut a_ref = module_ref.ggsw_alloc_from_infos(&infos);
        for row in 0..infos.dnum.as_usize() {
            for col in 0..rank + 1 {
                module_ref.fill_glwe_mask_from_source(params.base2k, &mut a_ref.at_view_mut(row, col), 0, rank + 1, &mut source);
            }
        }
        let mut a_test = module_test.ggsw_alloc_from_infos(&infos);
        a_ref.transfer_into(&mut a_test);
        for k in [-5i64, 0, 1, module_ref.n() as i64, 2 * module_ref.n() as i64 + 1] {
            let mut out_ref = module_ref.ggsw_alloc_from_infos(&infos);
            let mut out_test = module_test.ggsw_alloc_from_infos(&infos);
            module_ref.ggsw_rotate(k, &mut out_ref, &a_ref);
            module_test.ggsw_rotate(k, &mut out_test, &a_test);
            let mut have = module_ref.ggsw_alloc_from_infos(&infos);
            out_test.transfer_into(&mut have);
            assert_eq!(out_ref, have, "ggsw_rotate rank={rank} k={k}");
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
            assert_eq!(out_ref, have, "ggsw_rotate_assign rank={rank} k={k}");
        }
    }
}
