//! Keyless GLWE operation parity.
//!
//! These take no prepared key, so each test is fill, upload, run on both,
//! compare.

use poulpy_hal::{
    api::{ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{HostDataMut, Module, ScratchOwned},
    source::Source,
    test_suite::TestParams,
};

use crate::{
    GLWEAdd, GLWENegate, GLWENormalize, GLWERotate, GLWESub,
    api::TransferInto,
    layouts::{Base2K, Degree, GLWELayout, ModuleCoreAlloc, Rank, TorusPrecision},
    test_suite::parity::{ParityBackend, ref_glwe},
};

/// Layouts swept by the keyless operation tests.
fn layouts(n: u32, base2k: usize) -> Vec<GLWELayout> {
    let mut out = Vec::new();
    for rank in 1..3u32 {
        for limbs in [1usize, 2, 5] {
            out.push(GLWELayout {
                n: Degree(n),
                base2k: Base2K(base2k as u32),
                k: TorusPrecision((limbs * base2k) as u32),
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
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
    label: &str,
    seed: u8,
    tmp_bytes: usize,
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

    for a_infos in layouts(n, params.base2k) {
        for res_infos in layouts(n, params.base2k) {
            if res_infos.rank != a_infos.rank {
                continue;
            }

            let a_ref = ref_glwe(module_ref, &a_infos, &mut source);
            let b_ref = ref_glwe(module_ref, &a_infos, &mut source);
            let mut res_ref = ref_glwe(module_ref, &res_infos, &mut source);

            let mut a_test = module_test.glwe_alloc_from_infos(&a_infos);
            a_ref.transfer_into(&mut a_test);
            let mut b_test = module_test.glwe_alloc_from_infos(&a_infos);
            b_ref.transfer_into(&mut b_test);
            let mut res_test = module_test.glwe_alloc_from_infos(&res_infos);
            res_ref.transfer_into(&mut res_test);

            let mut scratch_ref: ScratchOwned<BR> = ScratchOwned::alloc(tmp_bytes.max(1));
            let mut scratch_test: ScratchOwned<BT> = ScratchOwned::alloc(tmp_bytes.max(1));

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

/// `glwe_add_into` agrees with the reference backend.
pub fn test_glwe_add_parity<BR, BT>(params: &TestParams, module_ref: &Module<BR>, module_test: &Module<BT>)
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
        module_ref,
        module_test,
        "glwe_add_into",
        3,
        0,
        |m, res, a, b, _| m.glwe_add_into(res, a, b),
        |m, res, a, b, _| m.glwe_add_into(res, a, b),
    );
}

/// `glwe_sub` agrees with the reference backend.
pub fn test_glwe_sub_parity<BR, BT>(params: &TestParams, module_ref: &Module<BR>, module_test: &Module<BT>)
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
        module_ref,
        module_test,
        "glwe_sub",
        5,
        0,
        |m, res, a, b, _| m.glwe_sub(res, a, b),
        |m, res, a, b, _| m.glwe_sub(res, a, b),
    );
}

/// `glwe_negate` agrees with the reference backend.
pub fn test_glwe_negate_parity<BR, BT>(params: &TestParams, module_ref: &Module<BR>, module_test: &Module<BT>)
where
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
        module_ref,
        module_test,
        "glwe_negate",
        9,
        0,
        |m, res, a, _, _| m.glwe_negate(res, a),
        |m, res, a, _, _| m.glwe_negate(res, a),
    );
}

/// `glwe_normalize` agrees with the reference backend.
///
/// The one keyless operation that carries limb-carry logic, so the one most
/// worth comparing byte-for-byte.
pub fn test_glwe_normalize_parity<BR, BT>(params: &TestParams, module_ref: &Module<BR>, module_test: &Module<BT>)
where
    BR: ParityBackend,
    BT: ParityBackend,
    BR::OwnedBuf: HostDataMut,
    Module<BR>: GLWENormalize<BR>,
    Module<BT>: GLWENormalize<BT>,
    ScratchOwned<BR>: ScratchOwnedAlloc<BR> + ScratchOwnedBorrow<BR>,
    ScratchOwned<BT>: ScratchOwnedAlloc<BT> + ScratchOwnedBorrow<BT>,
{
    let tmp = module_ref
        .glwe_normalize_tmp_bytes()
        .max(module_test.glwe_normalize_tmp_bytes());
    compare(
        params,
        module_ref,
        module_test,
        "glwe_normalize",
        17,
        tmp,
        |m, res, a, _, s| m.glwe_normalize(res, a, &mut s.borrow()),
        |m, res, a, _, s| m.glwe_normalize(res, a, &mut s.borrow()),
    );
}

/// `glwe_rotate` agrees with the reference backend.
pub fn test_glwe_rotate_parity<BR, BT>(params: &TestParams, module_ref: &Module<BR>, module_test: &Module<BT>)
where
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
            module_ref,
            module_test,
            "glwe_rotate",
            29,
            0,
            move |m, res, a, _, _| m.glwe_rotate(k, res, a),
            move |m, res, a, _, _| m.glwe_rotate(k, res, a),
        );
    }
}
