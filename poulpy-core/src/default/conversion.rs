//! Reference implementations of the [`ConversionDefault`] methods.
//!
//! Each free function carries the HAL bounds it actually needs in its own `where` clause.
//!
//! Re-exported publicly through `crate::oep::conversion_defaults`.

use crate::api::GLWEBytesOf;
use poulpy_hal::{
    api::{
        ModuleN, ScratchArenaTakeBasic, VecZnxBigAddSmallAssign, VecZnxBigBytesOf, VecZnxBigNormalize,
        VecZnxBigNormalizeTmpBytes, VecZnxCopyRangeBackend, VecZnxDftApply, VecZnxDftBytesOf, VecZnxDftZero,
        VecZnxExtractCoeffBackend, VecZnxIdftApply, VecZnxIdftApplyTmpBytes, VecZnxNormalize, VecZnxNormalizeTmpBytes,
        VecZnxRotateBackend, VecZnxZeroBackend,
    },
    layouts::{
        Backend, ScratchArena, VecZnxBackendRef, VecZnxBigToBackendRef, VecZnxDftBackendRef, VecZnxDftToBackendRef,
        VecZnxToBackendMut, VecZnxToBackendRef, ZnxInfos,
    },
};

use crate::{
    GLWERotate, ScratchArenaTakeCore,
    default::{
        keyswitching::glwe::{bound_for, bound_layout},
        keyswitching::{GGLWEProductDefault, gglwe_product_output_size},
        operations::GLWECopyDefault,
    },
    layouts::{
        GGLWEInfos, GGLWEToBackendRef, GGLWEUse, GGSWAtViewMut, GGSWInfos, GGSWToBackendMut, GLWEInfos, GLWELayout,
        GLWEToBackendMut, GLWEToBackendRef, GLWEViewMut, GLWEViewRef, LWEInfos, LWEMatrixInfos, LWEMatrixToBackendMut,
        LWEToBackendMut, LWEToBackendRef, Rank, TorusPrecision, glwe_backend_ref_from_mut,
        prepared::{GGLWEPreparedToBackendRef, GGLWEToGGSWKeyPreparedBackendRef, GGLWEToGGSWKeyPreparedToBackendRef},
    },
    oep::{ConversionDefault, GLWEKeyswitchDefault},
};

pub fn lwe_sample_extract_default<BE, M, R, A>(module: &M, res: &mut R, a: &A)
where
    BE: Backend,
    M: ModuleN + VecZnxCopyRangeBackend<BE> + VecZnxZeroBackend<BE>,
    R: LWEToBackendMut<BE> + LWEInfos,
    A: GLWEToBackendRef<BE> + GLWEInfos,
{
    let mut res = res.to_backend_mut();
    let a = a.to_backend_ref();

    assert!(res.n() <= a.n());
    assert_eq!(a.n(), module.n() as u32);
    assert!(res.base2k() == a.base2k());

    let min_size: usize = res.size().min(a.size());
    let n: usize = res.n().into();

    module.vec_znx_zero_backend(&mut res.body, 0);
    module.vec_znx_zero_backend(&mut res.mask, 0);
    (0..min_size).for_each(|i| {
        module.vec_znx_copy_range_backend(&mut res.body, 0, i, 0, &a.data, 0, i, 0, 1);
        module.vec_znx_copy_range_backend(&mut res.mask, 0, i, 0, &a.data, 1, i, 0, n);
    });
}

pub fn glwe_expand_lwe_tmp_bytes_default<BE, M, R, A>(module: &M, lwe_infos: &R, a_infos: &A) -> usize
where
    BE: Backend,
    M: ModuleN,
    R: LWEInfos,
    A: GLWEInfos,
{
    assert_eq!(
        a_infos.n().as_usize(),
        module.n(),
        "glwe_expand_lwe_tmp_bytes: GLWE.n() != module.n()"
    );
    assert_glwe_expand_lwe_lwe_layout(lwe_infos, a_infos, "glwe_expand_lwe_tmp_bytes");

    if a_infos.rank().as_usize() == 1 {
        0
    } else {
        BE::bytes_of_vec_znx(module.n(), 1, lwe_infos.size())
    }
}

fn assert_glwe_expand_lwe_lwe_layout<R, A>(lwe_infos: &R, a_infos: &A, context: &str)
where
    R: LWEInfos,
    A: GLWEInfos,
{
    let expected_lwe_n = a_infos.n().as_usize() * a_infos.rank().as_usize();
    assert_eq!(
        lwe_infos.n().as_usize(),
        expected_lwe_n,
        "{context}: LWE.n() must equal GLWE.n() * GLWE.rank()"
    );
    assert_eq!(
        lwe_infos.base2k(),
        a_infos.base2k(),
        "{context}: LWE.base2k() must equal GLWE.base2k()"
    );
    assert_eq!(
        lwe_infos.size(),
        a_infos.size(),
        "{context}: LWE.size() must equal GLWE.size()"
    );
}

pub fn glwe_expand_lwe_default<BE, M, R, A>(module: &M, res: &mut [R], a: &A, scratch: &mut ScratchArena<'_, BE>)
where
    BE: Backend,
    M: ModuleN + VecZnxExtractCoeffBackend<BE> + VecZnxRotateBackend<BE> + VecZnxCopyRangeBackend<BE>,
    R: LWEToBackendMut<BE> + LWEInfos,
    A: GLWEToBackendRef<BE> + GLWEInfos,
{
    let a = a.to_backend_ref();
    let n = module.n();
    let rank: usize = a.rank().into();

    assert_eq!(usize::from(a.n()), n, "glwe_expand_lwe: GLWE.n() != module.n()");
    assert!(res.len() <= n, "glwe_expand_lwe: res.len() > module.n()");
    for (idx, lwe) in res.iter().enumerate() {
        assert_glwe_expand_lwe_lwe_layout(lwe, &a, &format!("glwe_expand_lwe: res[{idx}]"));
    }

    if rank == 1 {
        for (i, lwe) in res.iter_mut().enumerate() {
            let mut lwe = lwe.to_backend_mut();
            module.vec_znx_extract_coeff_backend(&mut lwe.body, 0, &a.data, 0, i);
            module.vec_znx_rotate_backend(-(i as i64), &mut lwe.mask, 0, &a.data, 1);
        }
    } else {
        let lwe_size = res.first().map(|r| r.size()).unwrap_or(0);
        let (mut tmp, _) = scratch.borrow().take_vec_znx_scratch(n, 1, lwe_size);
        for (i, lwe) in res.iter_mut().enumerate() {
            let mut lwe = lwe.to_backend_mut();
            module.vec_znx_extract_coeff_backend(&mut lwe.body, 0, &a.data, 0, i);
            for j in 0..rank {
                {
                    let mut tmp_mut = tmp.to_backend_mut();
                    module.vec_znx_rotate_backend(-(i as i64), &mut tmp_mut, 0, &a.data, j + 1);
                }
                let tmp_ref = tmp.to_backend_ref();
                for l in 0..lwe_size {
                    module.vec_znx_copy_range_backend(&mut lwe.mask, 0, l, j * n, &tmp_ref, 0, l, 0, n);
                }
            }
        }
    }
}

pub fn glwe_expand_lwe_matrix_tmp_bytes_default<BE, M, R, A>(module: &M, _res_infos: &R, a_infos: &A) -> usize
where
    BE: Backend,
    M: ModuleN,
    R: LWEMatrixInfos,
    A: GLWEInfos,
{
    BE::bytes_of_vec_znx(module.n(), 1, a_infos.size())
}

pub fn glwe_expand_lwe_matrix_default<BE, M, R, A>(module: &M, res: &mut R, a: &A, scratch: &mut ScratchArena<'_, BE>)
where
    BE: Backend,
    M: ModuleN + VecZnxRotateBackend<BE> + VecZnxCopyRangeBackend<BE> + VecZnxZeroBackend<BE>,
    R: LWEMatrixToBackendMut<BE> + LWEMatrixInfos,
    A: GLWEToBackendRef<BE> + GLWEInfos,
{
    let a = a.to_backend_ref();
    let mut res = res.to_backend_mut();
    let n = module.n();
    let rank = a.rank().as_usize();
    let min_size = res.size().min(a.size());
    let rows = res.rows();

    assert_eq!(a.n().as_usize(), n, "glwe_expand_lwe_matrix: GLWE.n() != module.n()");
    assert_eq!(
        res.n().as_usize(),
        rank * n,
        "glwe_expand_lwe_matrix: invalid result LWE dimension"
    );
    assert!(rows <= n, "glwe_expand_lwe_matrix: rows > module.n()");
    assert_eq!(res.base2k(), a.base2k(), "glwe_expand_lwe_matrix: base2k mismatch");
    assert!(
        scratch.available() >= glwe_expand_lwe_matrix_tmp_bytes_default::<BE, _, _, _>(module, &res, &a),
        "scratch.available(): {} < GLWEExpandLWEMatrix::glwe_expand_lwe_matrix_tmp_bytes: {}",
        scratch.available(),
        glwe_expand_lwe_matrix_tmp_bytes_default::<BE, _, _, _>(module, &res, &a)
    );

    module.vec_znx_zero_backend(&mut res.body, 0);
    for col in 0..res.n().as_usize() {
        module.vec_znx_zero_backend(&mut res.mask, col);
    }

    let (mut tmp, _) = scratch.borrow().take_vec_znx_scratch(n, 1, min_size);
    for limb in 0..min_size {
        module.vec_znx_copy_range_backend(&mut res.body, 0, limb, 0, &a.data, 0, limb, 0, rows);
    }

    for row in 0..rows {
        for glwe_col in 0..rank {
            {
                let mut tmp = tmp.to_backend_mut();
                module.vec_znx_rotate_backend(-(row as i64), &mut tmp, 0, &a.data, glwe_col + 1);
            }
            let tmp_ref = tmp.to_backend_ref();
            for limb in 0..min_size {
                for coeff in 0..n {
                    module.vec_znx_copy_range_backend(
                        &mut res.mask,
                        glwe_col * n + coeff,
                        limb,
                        row,
                        &tmp_ref,
                        0,
                        limb,
                        coeff,
                        1,
                    );
                }
            }
        }
    }
}

pub fn glwe_from_lwe_tmp_bytes_default<BE, M, R, A, K>(module: &M, glwe_infos: &R, lwe_infos: &A, key_infos: &K) -> usize
where
    BE: Backend,
    M: GLWEBytesOf<BE> + ModuleN + GLWEKeyswitchDefault<BE> + VecZnxNormalizeTmpBytes,
    R: GLWEInfos,
    A: LWEInfos,
    K: GGLWEInfos,
{
    assert_eq!(module.n() as u32, glwe_infos.n());
    assert_eq!(module.n() as u32, key_infos.n());

    let lvl_0: usize = module.glwe_bytes_of(
        module.n().into(),
        key_infos.base2k(),
        lwe_infos.k().max(glwe_infos.k()),
        1u32.into(),
    );

    let lvl_1_ks: usize = module.glwe_keyswitch_tmp_bytes_default(glwe_infos, glwe_infos, key_infos);
    let lvl_1_a_conv: usize = if lwe_infos.base2k() == key_infos.base2k() {
        0
    } else {
        BE::bytes_of_vec_znx(module.n(), 1, lwe_infos.size()) + module.vec_znx_normalize_tmp_bytes()
    };

    let lvl_1: usize = lvl_1_ks.max(lvl_1_a_conv);

    lvl_0 + lvl_1
}

pub fn glwe_from_lwe_default<BE, M, R, A, K>(module: &M, res: &mut R, lwe: &A, ksk: &K, scratch: &mut ScratchArena<'_, BE>)
where
    BE: Backend,
    M: GLWEBytesOf<BE>
        + ConversionDefault<BE>
        + ModuleN
        + GLWEKeyswitchDefault<BE>
        + VecZnxCopyRangeBackend<BE>
        + VecZnxZeroBackend<BE>
        + VecZnxNormalize<BE>
        + VecZnxNormalizeTmpBytes,
    R: GLWEToBackendMut<BE> + GLWEInfos,
    A: LWEToBackendRef<BE> + LWEInfos,
    K: GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
{
    let res_infos = GLWELayout {
        n: res.n(),
        base2k: res.base2k(),
        k: res.k(),
        rank: res.rank(),
    };
    let lwe = lwe.to_backend_ref();

    assert_eq!(res_infos.n.as_u32(), module.n() as u32);
    assert_eq!(ksk.n(), module.n() as u32);
    assert!(lwe.n() <= module.n() as u32);
    assert!(
        scratch.available() >= module.glwe_from_lwe_tmp_bytes_default(&res_infos, &lwe, ksk),
        "scratch.available(): {} < GLWEFromLWE::glwe_from_lwe_tmp_bytes: {}",
        scratch.available(),
        module.glwe_from_lwe_tmp_bytes_default(&res_infos, &lwe, ksk)
    );

    let scratch = scratch.borrow();

    let (mut glwe, scratch_1) = scratch.take_glwe_scratch(&GLWELayout {
        n: ksk.n(),
        base2k: ksk.base2k(),
        k: lwe.k(),
        rank: 1u32.into(),
    });
    module.vec_znx_zero_backend(&mut glwe.data, 0);
    module.vec_znx_zero_backend(&mut glwe.data, 1);

    let n_lwe: usize = lwe.n().into();

    let mut scratch_1 = if lwe.base2k() == ksk.base2k() {
        for i in 0..lwe.size() {
            module.vec_znx_copy_range_backend(&mut glwe.data, 0, i, 0, &lwe.body, 0, i, 0, 1);
            module.vec_znx_copy_range_backend(&mut glwe.data, 1, i, 0, &lwe.mask, 0, i, 0, n_lwe);
        }
        scratch_1
    } else {
        let (mut a_conv, mut scratch_2) = scratch_1.take_vec_znx_scratch(module.n(), 1, lwe.size());
        module.vec_znx_zero_backend(&mut a_conv, 0);
        for j in 0..lwe.size() {
            module.vec_znx_copy_range_backend(&mut a_conv, 0, j, 0, &lwe.body, 0, j, 0, 1);
        }

        module.vec_znx_normalize(
            &mut glwe.data,
            ksk.base2k().into(),
            0,
            0,
            &a_conv.to_backend_ref(),
            lwe.base2k().into(),
            0,
            &mut scratch_2.borrow(),
        );

        module.vec_znx_zero_backend(&mut a_conv, 0);
        for j in 0..lwe.size() {
            module.vec_znx_copy_range_backend(&mut a_conv, 0, j, 0, &lwe.mask, 0, j, 0, n_lwe);
        }

        module.vec_znx_normalize(
            &mut glwe.data,
            ksk.base2k().into(),
            0,
            1,
            &a_conv.to_backend_ref(),
            lwe.base2k().into(),
            0,
            &mut scratch_2.borrow(),
        );

        scratch_2
    };

    let mut res_backend = res.to_backend_mut();
    let glwe_ref = glwe_backend_ref_from_mut::<BE>(&glwe);
    let glwe_view = &glwe_ref;
    let mut res_view = &mut res_backend;
    module.glwe_keyswitch_default(&mut res_view, &glwe_view, ksk, &mut scratch_1)
}

pub fn lwe_from_glwe_tmp_bytes_default<BE, M, R, A, K>(module: &M, lwe_infos: &R, glwe_infos: &A, key_infos: &K) -> usize
where
    BE: Backend,
    M: GLWEBytesOf<BE> + ModuleN + GLWEKeyswitchDefault<BE>,
    R: LWEInfos,
    A: GLWEInfos,
    K: GGLWEInfos,
{
    assert_eq!(module.n() as u32, glwe_infos.n());
    assert_eq!(module.n() as u32, key_infos.n());

    let res_infos: GLWELayout = GLWELayout {
        n: module.n().into(),
        base2k: lwe_infos.base2k(),
        k: lwe_infos.k(),
        rank: Rank(1),
    };

    let lvl_0: usize = module.glwe_bytes_of(module.n().into(), lwe_infos.base2k(), lwe_infos.k(), 1u32.into());
    let lvl_1: usize = module.glwe_keyswitch_tmp_bytes_default(&res_infos, glwe_infos, key_infos);
    let lvl_2: usize = module.glwe_bytes_of_from_infos(glwe_infos);

    lvl_0 + lvl_1 + lvl_2
}

pub fn lwe_from_glwe_default<BE, M, R, A, K>(
    module: &M,
    res: &mut R,
    a: &A,
    a_idx: usize,
    key: &K,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: Backend,
    M: GLWEBytesOf<BE>
        + ConversionDefault<BE>
        + ModuleN
        + GLWEKeyswitchDefault<BE>
        + GLWERotate<BE>
        + VecZnxCopyRangeBackend<BE>
        + VecZnxZeroBackend<BE>,
    R: LWEToBackendMut<BE> + LWEInfos,
    A: GLWEToBackendRef<BE> + GLWEInfos,
    K: GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
{
    let a_backend = a.to_backend_ref();

    assert_eq!(a.n(), module.n() as u32);
    assert_eq!(key.n(), module.n() as u32);
    assert!(res.n() <= module.n() as u32);
    assert!(
        scratch.available() >= module.lwe_from_glwe_tmp_bytes_default(res, a, key),
        "scratch.available(): {} < LWEFromGLWE::lwe_from_glwe_tmp_bytes: {}",
        scratch.available(),
        module.lwe_from_glwe_tmp_bytes_default(res, a, key)
    );

    let glwe_layout: GLWELayout = GLWELayout {
        n: module.n().into(),
        base2k: res.base2k(),
        k: res.k(),
        rank: Rank(1),
    };

    let scratch = scratch.borrow();
    let (mut tmp_glwe_rank_1, mut scratch_1) = scratch.take_glwe_scratch(&glwe_layout);

    let a_backend_view = &a_backend;
    module.glwe_keyswitch_default(&mut tmp_glwe_rank_1, &a_backend_view, key, &mut scratch_1);
    if a_idx != 0 {
        module.glwe_rotate_assign(-(a_idx as i64), &mut tmp_glwe_rank_1, &mut scratch_1);
    }

    let mut res_backend = res.to_backend_mut();
    let tmp_glwe_rank_1_ref = glwe_backend_ref_from_mut::<BE>(&tmp_glwe_rank_1);
    let min_size: usize = res_backend.size().min(tmp_glwe_rank_1_ref.size());
    let n: usize = res_backend.n().into();

    module.vec_znx_zero_backend(&mut res_backend.body, 0);
    module.vec_znx_zero_backend(&mut res_backend.mask, 0);
    for i in 0..min_size {
        module.vec_znx_copy_range_backend(&mut res_backend.body, 0, i, 0, &tmp_glwe_rank_1_ref.data, 0, i, 0, 1);
        module.vec_znx_copy_range_backend(&mut res_backend.mask, 0, i, 0, &tmp_glwe_rank_1_ref.data, 1, i, 0, n);
    }
}

pub fn ggsw_from_gglwe_tmp_bytes_default<BE, M, R, A>(module: &M, res_infos: &R, tsk_infos: &A) -> usize
where
    BE: Backend,
    M: ConversionDefault<BE>,
    R: GGSWInfos,
    A: GGLWEInfos,
{
    module.ggsw_expand_rows_tmp_bytes_default(res_infos, tsk_infos)
}

pub fn ggsw_from_gglwe_default<BE, M, R, A, T>(module: &M, res: &mut R, a: &A, tsk: &T, scratch: &mut ScratchArena<'_, BE>)
where
    BE: Backend,
    M: ConversionDefault<BE> + ModuleN + GLWECopyDefault<BE>,
    R: GGSWToBackendMut<BE> + GGSWInfos,
    A: GGLWEToBackendRef<BE> + GGLWEInfos,
    T: GGLWEToGGSWKeyPreparedToBackendRef<BE> + GGLWEInfos,
{
    let mut res_backend = res.to_backend_mut();
    let a_backend = a.to_backend_ref();

    assert_eq!(res_backend.rank(), a_backend.rank_out());
    assert_eq!(res_backend.dnum(), a_backend.dnum());
    assert_eq!(res_backend.n(), module.n() as u32);
    assert_eq!(a_backend.n(), module.n() as u32);
    assert_eq!(tsk.n(), module.n() as u32);
    assert_eq!(res_backend.base2k(), a_backend.base2k());
    assert!(
        scratch.available() >= module.ggsw_from_gglwe_tmp_bytes_default(&res_backend, tsk),
        "scratch.available(): {} < GGSWFromGGLWE::ggsw_from_gglwe_tmp_bytes: {}",
        scratch.available(),
        module.ggsw_from_gglwe_tmp_bytes_default(&res_backend, tsk)
    );

    for row in 0..res_backend.dnum().into() {
        let mut res_at = res_backend.at_view_mut(row, 0);
        let a_at = a_backend.at_view(row, 0);
        module.glwe_copy_default(&mut res_at, &a_at);
    }

    module.ggsw_expand_row_default(&mut res_backend, tsk, scratch)
}

pub fn ggsw_expand_rows_tmp_bytes_default<BE, M, R, A>(module: &M, res_infos: &R, tsk_infos: &A) -> usize
where
    BE: Backend,
    M: GLWEBytesOf<BE>
        + ModuleN
        + GGLWEProductDefault<BE>
        + VecZnxBigBytesOf
        + VecZnxBigNormalizeTmpBytes
        + VecZnxDftBytesOf
        + VecZnxIdftApplyTmpBytes
        + VecZnxNormalizeTmpBytes,
    R: GGSWInfos,
    A: GGLWEInfos,
{
    assert_eq!(module.n() as u32, res_infos.n());
    assert_eq!(module.n() as u32, tsk_infos.n());

    let tsk_base2k: usize = tsk_infos.base2k().into();

    let rank: usize = res_infos.rank().into();
    let cols: usize = rank + 1;

    let a_size: usize = res_infos.k().as_usize().div_ceil(tsk_base2k);
    let input_k: TorusPrecision = TorusPrecision((a_size * tsk_base2k) as u32);
    let output_size = gglwe_product_output_size::<BE, _, _, _>(res_infos, res_infos, &bound_layout(tsk_infos, input_k));

    let lvl_0: usize = module.bytes_of_vec_znx_dft(cols - 1, a_size) + BE::bytes_of_vec_znx(module.n(), 1, a_size);
    let lvl_1_res_dft: usize = module.bytes_of_vec_znx_dft(cols, output_size);
    let lvl_1_gglwe_prod: usize = match bound_for(tsk_infos, input_k) {
        GGLWEUse::Empty => 0,
        GGLWEUse::Active(active) => module.gglwe_product_dft_tmp_bytes_default(output_size, a_size, &active),
    };
    let lvl_1_big: usize = module.bytes_of_vec_znx_big(cols, output_size)
        + module
            .vec_znx_idft_apply_tmp_bytes()
            .max(module.vec_znx_big_normalize_tmp_bytes());
    let lvl_1: usize = lvl_1_res_dft + lvl_1_gglwe_prod.max(lvl_1_big);
    let lvl_2: usize = module.vec_znx_normalize_tmp_bytes();

    lvl_0 + lvl_1.max(lvl_2)
}

pub fn ggsw_expand_row_default<BE, M, R, T>(module: &M, res: &mut R, tsk: &T, scratch: &mut ScratchArena<'_, BE>)
where
    BE: Backend,
    M: GLWEBytesOf<BE>
        + ConversionDefault<BE>
        + ModuleN
        + GGLWEProductDefault<BE>
        + VecZnxBigAddSmallAssign<BE>
        + VecZnxBigBytesOf
        + VecZnxBigNormalize<BE>
        + VecZnxDftApply<BE>
        + VecZnxDftZero<BE>
        + VecZnxIdftApply<BE>
        + VecZnxNormalize<BE>,
    R: GGSWToBackendMut<BE> + GGSWInfos,
    T: GGLWEToGGSWKeyPreparedToBackendRef<BE> + GGLWEInfos,
{
    let mut res_backend = res.to_backend_mut();

    let res_base2k: usize = res_backend.base2k().into();
    let tsk_base2k: usize = tsk.base2k().into();
    let input_k: TorusPrecision = TorusPrecision((res_backend.k().as_usize().div_ceil(tsk_base2k) * tsk_base2k) as u32);
    let output_size = gglwe_product_output_size::<BE, _, _, _>(&res_backend, &res_backend, &bound_layout(tsk, input_k));

    assert!(
        scratch.available() >= module.ggsw_expand_rows_tmp_bytes_default(&res_backend, tsk),
        "scratch.available(): {} < GGSWExpandRows::ggsw_expand_rows_tmp_bytes: {}",
        scratch.available(),
        module.ggsw_expand_rows_tmp_bytes_default(&res_backend, tsk)
    );

    let rank: usize = res_backend.rank().into();
    let cols: usize = rank + 1;

    let res_conv_size: usize = res_backend.k().as_usize().div_ceil(tsk_base2k);
    {
        let (mut a_dft, scratch_1) = scratch.borrow().take_vec_znx_dft_scratch(module, cols - 1, res_conv_size);
        let (mut a_0, mut scratch_2) = scratch_1.take_vec_znx_scratch(module.n(), 1, res_conv_size);

        for row in 0..res_backend.dnum().as_usize() {
            {
                let glwe_mi_1: GLWEViewRef<'_, BE> = res_backend.at_view(row, 0);

                for i in 0..cols - 1 {
                    module.vec_znx_normalize(
                        &mut a_0,
                        tsk_base2k,
                        0,
                        0,
                        &glwe_mi_1.data,
                        res_base2k,
                        i + 1,
                        &mut scratch_2.borrow(),
                    );
                    let a_0_ref: VecZnxBackendRef<'_, BE> = a_0.to_backend_ref();
                    module.vec_znx_dft_apply(1, 0, &mut a_dft, i, &a_0_ref, 0);
                }
                module.vec_znx_normalize(
                    &mut a_0,
                    tsk_base2k,
                    0,
                    0,
                    &glwe_mi_1.data,
                    res_base2k,
                    0,
                    &mut scratch_2.borrow(),
                );
            }

            let a_0_ref: VecZnxBackendRef<'_, BE> = a_0.to_backend_ref();
            let a_dft_ref: VecZnxDftBackendRef<'_, BE> = a_dft.to_backend_ref();
            let mut scratch_row = scratch_2.borrow();
            ggsw_expand_rows_internal(
                module,
                row,
                &mut res_backend,
                &a_0_ref,
                &a_dft_ref,
                tsk,
                output_size,
                &mut scratch_row,
            );
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn ggsw_expand_rows_internal<'a, 'b, R, M, T, BE: Backend>(
    module: &M,
    row: usize,
    res: &mut R,
    a_0: &VecZnxBackendRef<'a, BE>,
    a_dft: &VecZnxDftBackendRef<'b, BE>,
    tsk: &T,
    output_size: usize,
    scratch: &mut ScratchArena<'_, BE>,
) where
    M: GLWEBytesOf<BE>
        + GGLWEProductDefault<BE>
        + ModuleN
        + VecZnxBigBytesOf
        + VecZnxBigAddSmallAssign<BE>
        + VecZnxBigNormalize<BE>
        + VecZnxDftZero<BE>
        + VecZnxIdftApply<BE>,
    R: GGSWAtViewMut<BE> + GGSWInfos,
    T: GGLWEToGGSWKeyPreparedToBackendRef<BE>,
{
    let tsk: GGLWEToGGSWKeyPreparedBackendRef<'_, BE> = tsk.to_backend_ref();
    let cols: usize = res.rank().as_usize() + 1;

    for col in 1..cols {
        let scratch_row = scratch.borrow();
        let (mut res_dft, mut scratch_1) = scratch_row.take_vec_znx_dft_scratch(module, cols, output_size);
        {
            let mut scratch_prod = scratch_1.borrow();
            let key = tsk.at(col - 1);
            let input_k: TorusPrecision = TorusPrecision((a_dft.size() * key.base2k().as_usize()) as u32);
            if let GGLWEUse::Active(active) = bound_for(key, input_k) {
                module.gglwe_product_dft_default(&mut res_dft, a_dft, key, &active, 1, &mut scratch_prod);
            }
        }

        let (mut res_big, mut scratch_2) = scratch_1.take_vec_znx_big_scratch(module, cols, res_dft.size());
        let res_dft_ref = res_dft.to_backend_ref();
        for j in 0..cols {
            scratch_2 = scratch_2.apply_mut(|scratch| module.vec_znx_idft_apply(&mut res_big, j, &res_dft_ref, j, scratch));
        }

        module.vec_znx_big_add_small_assign(&mut res_big, col, a_0, 0);
        let res_big_ref = res_big.to_backend_ref();

        let res_base2k: usize = res.base2k().as_usize();

        for j in 0..cols {
            let mut res_col: GLWEViewMut<'_, _> = res.at_view_mut(row, col);
            let scratch_norm = &mut scratch_2.borrow();
            module.vec_znx_big_normalize(
                &mut res_col.data,
                res_base2k,
                0,
                j,
                &res_big_ref,
                tsk.base2k().as_usize(),
                j,
                scratch_norm,
            );
        }
    }
}
