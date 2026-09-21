//! Portable algorithms expressed with HAL operations.
//! Inter-family core operations dispatch through the selected backend hooks.
use crate::api::GLWEKeyswitch;

use crate::api::GLWEBytesOf;
use poulpy_hal::{
    api::{
        ModuleN, ScratchArenaTakeBasic, VecZnxBigAddSmallAssign, VecZnxBigBytesOf, VecZnxBigNormalize,
        VecZnxBigNormalizeTmpBytes, VecZnxCopy, VecZnxDftApply, VecZnxDftBytesOf, VecZnxDftZero, VecZnxIdftApply,
        VecZnxIdftApplyTmpBytes, VecZnxNormalize, VecZnxNormalizeTmpBytes, VecZnxRotate, VecZnxZero,
    },
    layouts::{
        Backend, Module, ScratchArena, VecZnxBackendRef, VecZnxBigToBackendRef, VecZnxDftBackendRef, VecZnxDftToBackendRef,
        VecZnxToBackendMut, VecZnxToBackendRef, ZnxInfos, vec_znx_backend_mut_from_mut, vec_znx_backend_ref_from_ref,
    },
};

use crate::{
    ScratchArenaTakeCore,
    layouts::{
        GGLWEInfos, GGSWAtViewMut, GGSWInfos, GGSWToBackendMut, GLWEInfos, GLWELayout, GLWEToBackendMut, GLWEToBackendRef,
        GLWEViewMut, GLWEViewRef, LWEInfos, LWEMatrixInfos, LWEMatrixToBackendMut, LWEToBackendMut, LWEToBackendRef, Rank,
        glwe_backend_ref_from_mut,
        prepared::{GGLWEPreparedBackendRef, GGLWEPreparedToBackendRef, GGLWEToGGSWKeyPreparedBackendRef},
    },
    reference::keyswitching::{GGLWEProductReference, gglwe_product_output_size},
};

pub fn lwe_sample_extract_reference<BE, M, R, A>(module: &M, res: &mut R, a: &A)
where
    BE: Backend,
    M: ModuleN + VecZnxCopy<BE>,
    R: LWEToBackendMut<BE> + LWEInfos,
    A: GLWEToBackendRef<BE> + GLWEInfos,
{
    let mut res = res.to_backend_mut();
    let a = a.to_backend_ref();

    assert!(res.n() <= a.n());
    assert_eq!(a.n(), module.n() as u32);
    assert!(res.base2k() == a.base2k());

    let n: usize = res.n().into();
    module.vec_znx_copy(
        &mut res.body,
        0,
        &vec_znx_backend_ref_from_ref::<BE>(&a.data).window_coeffs(0, 1),
        0,
    );
    module.vec_znx_copy(
        &mut res.mask,
        0,
        &vec_znx_backend_ref_from_ref::<BE>(&a.data).window_coeffs(0, n),
        1,
    );
}

pub fn glwe_expand_lwe_tmp_bytes_reference<BE, M, R, A>(module: &M, lwe_infos: &R, a_infos: &A) -> usize
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

pub fn glwe_expand_lwe_reference<BE, M, R, A>(module: &M, res: &mut [R], a: &A, scratch: &mut ScratchArena<'_, BE>)
where
    BE: Backend,
    M: ModuleN + VecZnxCopy<BE> + VecZnxRotate<BE>,
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
            module.vec_znx_copy(
                &mut lwe.body,
                0,
                &vec_znx_backend_ref_from_ref::<BE>(&a.data).window_coeffs(i, 1),
                0,
            );
            module.vec_znx_rotate(-(i as i64), &mut lwe.mask, 0, &a.data, 1);
        }
    } else {
        let lwe_size = res.first().map(|r| r.size()).unwrap_or(0);
        let (mut tmp, _) = scratch.borrow().take_vec_znx_scratch(n, 1, lwe_size);
        for (i, lwe) in res.iter_mut().enumerate() {
            let mut lwe = lwe.to_backend_mut();
            module.vec_znx_copy(
                &mut lwe.body,
                0,
                &vec_znx_backend_ref_from_ref::<BE>(&a.data).window_coeffs(i, 1),
                0,
            );
            for j in 0..rank {
                {
                    let mut tmp_mut = tmp.to_backend_mut();
                    module.vec_znx_rotate(-(i as i64), &mut tmp_mut, 0, &a.data, j + 1);
                }
                let tmp_ref = tmp.to_backend_ref();
                module.vec_znx_copy(
                    &mut vec_znx_backend_mut_from_mut::<BE>(&mut lwe.mask).window_coeffs(j * n, n),
                    0,
                    &tmp_ref,
                    0,
                );
            }
        }
    }
}

pub fn glwe_expand_lwe_matrix_tmp_bytes_reference<BE, M, R, A>(module: &M, _res_infos: &R, a_infos: &A) -> usize
where
    BE: Backend,
    M: ModuleN,
    R: LWEMatrixInfos,
    A: GLWEInfos,
{
    BE::bytes_of_vec_znx(module.n(), 1, a_infos.size())
}

pub fn glwe_expand_lwe_matrix_reference<BE, M, R, A>(module: &M, res: &mut R, a: &A, scratch: &mut ScratchArena<'_, BE>)
where
    BE: Backend,
    M: ModuleN + VecZnxRotate<BE> + VecZnxCopy<BE> + VecZnxZero<BE>,
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
        scratch.available() >= glwe_expand_lwe_matrix_tmp_bytes_reference::<BE, _, _, _>(module, &res, &a),
        "scratch.available(): {} < GLWEExpandLWEMatrix::glwe_expand_lwe_matrix_tmp_bytes: {}",
        scratch.available(),
        glwe_expand_lwe_matrix_tmp_bytes_reference::<BE, _, _, _>(module, &res, &a)
    );

    for col in 0..res.n().as_usize() {
        module.vec_znx_zero(&mut res.mask, col);
    }

    let (mut tmp, _) = scratch.borrow().take_vec_znx_scratch(n, 1, min_size);
    module.vec_znx_copy(
        &mut res.body,
        0,
        &vec_znx_backend_ref_from_ref::<BE>(&a.data).window_coeffs(0, rows),
        0,
    );

    for row in 0..rows {
        for glwe_col in 0..rank {
            {
                let mut tmp = tmp.to_backend_mut();
                module.vec_znx_rotate(-(row as i64), &mut tmp, 0, &a.data, glwe_col + 1);
            }
            let tmp_ref = tmp.to_backend_ref();
            for coeff in 0..n {
                module.vec_znx_copy(
                    &mut vec_znx_backend_mut_from_mut::<BE>(&mut res.mask).window_coeffs(row, 1),
                    glwe_col * n + coeff,
                    &vec_znx_backend_ref_from_ref::<BE>(&tmp_ref).window_coeffs(coeff, 1),
                    0,
                );
            }
        }
    }
}

pub fn glwe_from_lwe_tmp_bytes_reference<BE, M, R, A, K>(module: &M, glwe_infos: &R, lwe_infos: &A, key_infos: &K) -> usize
where
    BE: Backend,
    M: GLWEBytesOf<BE> + ModuleN + GLWEKeyswitch<BE> + VecZnxNormalizeTmpBytes,
    R: GLWEInfos,
    A: LWEInfos,
    K: GGLWEInfos,
{
    assert_eq!(module.n() as u32, glwe_infos.n());
    assert_eq!(module.n() as u32, key_infos.n());

    // Match the actual rank-one, key-radix temporary passed to keyswitching.
    let lifted_infos = GLWELayout {
        n: module.n().into(),
        base2k: key_infos.base2k(),
        k: lwe_infos.k(),
        rank: Rank(1),
    };
    let lvl_0: usize = module.glwe_bytes_of_from_infos(&lifted_infos);
    let lvl_1_ks: usize = module.glwe_keyswitch_tmp_bytes(glwe_infos, &lifted_infos, key_infos);
    let lvl_1_a_conv: usize = if lwe_infos.base2k() == key_infos.base2k() {
        0
    } else {
        BE::bytes_of_vec_znx(module.n(), 1, lwe_infos.size()) + module.vec_znx_normalize_tmp_bytes()
    };

    let lvl_1: usize = lvl_1_ks.max(lvl_1_a_conv);

    lvl_0 + lvl_1
}

pub fn glwe_from_lwe_reference<BE, M, R, A>(
    module: &M,
    res: &mut R,
    lwe: &A,
    ksk: &GGLWEPreparedBackendRef<'_, BE>,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: Backend,
    M: GLWEBytesOf<BE>
        + ConversionReference<BE>
        + ModuleN
        + GLWEKeyswitch<BE>
        + VecZnxCopy<BE>
        + VecZnxZero<BE>
        + VecZnxNormalize<BE>
        + VecZnxNormalizeTmpBytes,
    R: GLWEToBackendMut<BE> + GLWEInfos,
    A: LWEToBackendRef<BE> + LWEInfos,
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
        scratch.available() >= module.glwe_from_lwe_tmp_bytes_reference(&res_infos, &lwe, ksk),
        "scratch.available(): {} < GLWEFromLWE::glwe_from_lwe_tmp_bytes: {}",
        scratch.available(),
        module.glwe_from_lwe_tmp_bytes_reference(&res_infos, &lwe, ksk)
    );

    let scratch = scratch.borrow();

    let (mut glwe, mut scratch_1) = scratch.take_glwe_scratch(&GLWELayout {
        n: ksk.n(),
        base2k: ksk.base2k(),
        k: lwe.k(),
        rank: 1u32.into(),
    });
    module.vec_znx_zero(&mut glwe.data, 0);
    module.vec_znx_zero(&mut glwe.data, 1);

    let n_lwe: usize = lwe.n().into();

    if lwe.base2k() == ksk.base2k() {
        module.vec_znx_copy(
            &mut vec_znx_backend_mut_from_mut::<BE>(&mut glwe.data).window_coeffs(0, 1),
            0,
            &lwe.body,
            0,
        );
        module.vec_znx_copy(
            &mut vec_znx_backend_mut_from_mut::<BE>(&mut glwe.data).window_coeffs(0, n_lwe),
            1,
            &lwe.mask,
            0,
        );
    } else {
        let (mut a_conv, mut scratch_2) = scratch_1.borrow().take_vec_znx_scratch(module.n(), 1, lwe.size());
        module.vec_znx_zero(&mut a_conv, 0);
        module.vec_znx_copy(&mut a_conv.to_backend_mut().window_coeffs(0, 1), 0, &lwe.body, 0);

        module.vec_znx_normalize(
            &mut glwe.data,
            ksk.base2k().into(),
            lwe.k().as_usize(),
            0,
            0,
            &a_conv.to_backend_ref(),
            lwe.base2k().into(),
            0,
            &mut scratch_2.borrow(),
        );

        module.vec_znx_zero(&mut a_conv, 0);
        module.vec_znx_copy(&mut a_conv.to_backend_mut().window_coeffs(0, n_lwe), 0, &lwe.mask, 0);

        module.vec_znx_normalize(
            &mut glwe.data,
            ksk.base2k().into(),
            lwe.k().as_usize(),
            0,
            1,
            &a_conv.to_backend_ref(),
            lwe.base2k().into(),
            0,
            &mut scratch_2.borrow(),
        );
    }

    let mut res_backend = res.to_backend_mut();
    let glwe_ref = glwe_backend_ref_from_mut::<BE>(&glwe);
    let glwe_view = &glwe_ref;
    let mut res_view = &mut res_backend;
    module.glwe_keyswitch(&mut res_view, &glwe_view, &ksk.to_backend_ref(), &mut scratch_1)
}

pub fn ggsw_expand_rows_tmp_bytes_reference<BE, M, R, A>(module: &M, res_infos: &R, tsk_infos: &A) -> usize
where
    BE: Backend,
    M: GLWEBytesOf<BE>
        + ModuleN
        + GGLWEProductReference<BE>
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
    let output_size = gglwe_product_output_size::<BE, _, _, _>(res_infos, res_infos, tsk_infos);

    let lvl_0: usize = module.bytes_of_vec_znx_dft(module.n(), cols - 1, a_size) + BE::bytes_of_vec_znx(module.n(), 1, a_size);
    let lvl_1_res_dft: usize = module.bytes_of_vec_znx_dft(module.n(), cols, output_size);
    let lvl_1_gglwe_prod: usize = module.gglwe_product_dft_tmp_bytes_reference(output_size, a_size, tsk_infos);
    let lvl_1_big: usize = module.bytes_of_vec_znx_big(module.n(), cols, output_size)
        + module
            .vec_znx_idft_apply_tmp_bytes()
            .max(module.vec_znx_big_normalize_tmp_bytes());
    let lvl_1: usize = lvl_1_res_dft + lvl_1_gglwe_prod.max(lvl_1_big);
    let lvl_2: usize = module.vec_znx_normalize_tmp_bytes();

    lvl_0 + lvl_1.max(lvl_2)
}

pub fn ggsw_expand_row_reference<BE, M, R>(
    module: &M,
    res: &mut R,
    tsk: &GGLWEToGGSWKeyPreparedBackendRef<'_, BE>,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: Backend,
    M: GLWEBytesOf<BE>
        + ConversionReference<BE>
        + ModuleN
        + GGLWEProductReference<BE>
        + VecZnxBigAddSmallAssign<BE>
        + VecZnxBigBytesOf
        + VecZnxBigNormalize<BE>
        + VecZnxDftApply<BE>
        + VecZnxDftZero<BE>
        + VecZnxIdftApply<BE>
        + VecZnxNormalize<BE>,
    R: GGSWToBackendMut<BE> + GGSWInfos,
{
    let mut res_backend = res.to_backend_mut();

    let output_size = gglwe_product_output_size::<BE, _, _, _>(&res_backend, &res_backend, tsk);
    let res_base2k: usize = res_backend.base2k().into();
    let tsk_base2k: usize = tsk.base2k().into();

    assert!(
        scratch.available() >= module.ggsw_expand_rows_tmp_bytes_reference(&res_backend, tsk),
        "scratch.available(): {} < GGSWExpandRows::ggsw_expand_rows_tmp_bytes: {}",
        scratch.available(),
        module.ggsw_expand_rows_tmp_bytes_reference(&res_backend, tsk)
    );

    let rank: usize = res_backend.rank().into();
    let cols: usize = rank + 1;

    let res_conv_size: usize = res_backend.k().as_usize().div_ceil(tsk_base2k);
    {
        let (mut a_dft, scratch_1) = scratch.borrow().take_vec_znx_dft_scratch(module.n(), cols - 1, res_conv_size);
        let (mut a_0, mut scratch_2) = scratch_1.take_vec_znx_scratch(module.n(), 1, res_conv_size);

        for row in 0..res_backend.dnum().as_usize() {
            {
                let glwe_mi_1: GLWEViewRef<'_, BE> = res_backend.at_view(row, 0);

                for i in 0..cols - 1 {
                    module.vec_znx_normalize(
                        &mut a_0,
                        tsk_base2k,
                        res_backend.k().as_usize(),
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
                    res_backend.k().as_usize(),
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
fn ggsw_expand_rows_internal<'a, 'b, R, M, BE: Backend>(
    module: &M,
    row: usize,
    res: &mut R,
    a_0: &VecZnxBackendRef<'a, BE>,
    a_dft: &VecZnxDftBackendRef<'b, BE>,
    tsk: &GGLWEToGGSWKeyPreparedBackendRef<'_, BE>,
    output_size: usize,
    scratch: &mut ScratchArena<'_, BE>,
) where
    M: GLWEBytesOf<BE>
        + GGLWEProductReference<BE>
        + ModuleN
        + VecZnxBigBytesOf
        + VecZnxBigAddSmallAssign<BE>
        + VecZnxBigNormalize<BE>
        + VecZnxDftZero<BE>
        + VecZnxIdftApply<BE>,
    R: GGSWAtViewMut<BE> + GGSWInfos,
{
    let cols: usize = res.rank().as_usize() + 1;
    let res_base2k = res.base2k().as_usize();
    let res_k = res.k().as_usize();

    for col in 1..cols {
        let scratch_row = scratch.borrow();
        let (mut res_dft, mut scratch_1) = scratch_row.take_vec_znx_dft_scratch(module.n(), cols, output_size);
        {
            let mut scratch_prod = scratch_1.borrow();
            module.gglwe_product_dft_reference(&mut res_dft, a_dft, tsk.at(col - 1), 1, &mut scratch_prod);
        }

        let (mut res_big, mut scratch_2) = scratch_1.take_vec_znx_big_scratch(module.n(), cols, res_dft.size());
        let res_dft_ref = res_dft.to_backend_ref();
        for j in 0..cols {
            scratch_2 = scratch_2.apply_mut(|scratch| module.vec_znx_idft_apply(&mut res_big, j, &res_dft_ref, j, scratch));
        }

        module.vec_znx_big_add_small_assign(&mut res_big, col, a_0, 0);
        let res_big_ref = res_big.to_backend_ref();

        let mut res_col: GLWEViewMut<'_, _> = res.at_view_mut(row, col);
        for j in 0..cols {
            let scratch_norm = &mut scratch_2.borrow();
            module.vec_znx_big_normalize(
                &mut res_col.data,
                res_base2k,
                res_k,
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

/// Portable HAL algorithm helpers. Available when their capabilities are present;
/// backend dispatch is selected independently through the corresponding `*Impl` trait.
pub trait ConversionReference<BE: Backend> {
    fn lwe_sample_extract_reference<R, A>(&self, res: &mut R, a: &A)
    where
        R: LWEToBackendMut<BE> + LWEInfos,
        A: GLWEToBackendRef<BE> + GLWEInfos;

    fn glwe_from_lwe_tmp_bytes_reference<R, A, K>(&self, glwe_infos: &R, lwe_infos: &A, key_infos: &K) -> usize
    where
        R: GLWEInfos,
        A: LWEInfos,
        K: GGLWEInfos;

    fn glwe_from_lwe_reference<R, A>(
        &self,
        res: &mut R,
        lwe: &A,
        ksk: &GGLWEPreparedBackendRef<'_, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        A: LWEToBackendRef<BE> + LWEInfos;

    fn glwe_expand_lwe_tmp_bytes_reference<R, A>(&self, lwe_infos: &R, a_infos: &A) -> usize
    where
        R: LWEInfos,
        A: GLWEInfos;

    fn glwe_expand_lwe_reference<R, A>(&self, res: &mut [R], a: &A, scratch: &mut ScratchArena<'_, BE>)
    where
        R: LWEToBackendMut<BE> + LWEInfos,
        A: GLWEToBackendRef<BE> + GLWEInfos;

    fn glwe_expand_lwe_matrix_tmp_bytes_reference<R, A>(&self, res_infos: &R, a_infos: &A) -> usize
    where
        R: LWEMatrixInfos,
        A: GLWEInfos;

    fn glwe_expand_lwe_matrix_reference<R, A>(&self, res: &mut R, a: &A, scratch: &mut ScratchArena<'_, BE>)
    where
        R: LWEMatrixToBackendMut<BE> + LWEMatrixInfos,
        A: GLWEToBackendRef<BE> + GLWEInfos;

    fn ggsw_expand_rows_tmp_bytes_reference<R, A>(&self, res_infos: &R, tsk_infos: &A) -> usize
    where
        R: GGSWInfos,
        A: GGLWEInfos;

    fn ggsw_expand_row_reference<R>(
        &self,
        res: &mut R,
        tsk: &GGLWEToGGSWKeyPreparedBackendRef<'_, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GGSWToBackendMut<BE> + GGSWInfos;
}

impl<BE: Backend> ConversionReference<BE> for ::poulpy_hal::layouts::Module<BE>
where
    Module<BE>: poulpy_hal::api::ModuleN
        + poulpy_hal::api::VecZnxCopy<BE>
        + poulpy_hal::api::VecZnxRotate<BE>
        + poulpy_hal::api::VecZnxZero<BE>
        + crate::api::GLWEBytesOf<BE>
        + crate::api::GLWEKeyswitch<BE>
        + poulpy_hal::api::VecZnxNormalizeTmpBytes
        + poulpy_hal::api::VecZnxNormalize<BE>
        + crate::reference::keyswitching::GGLWEProductReference<BE>
        + poulpy_hal::api::VecZnxBigBytesOf
        + poulpy_hal::api::VecZnxBigNormalizeTmpBytes
        + poulpy_hal::api::VecZnxDftBytesOf
        + poulpy_hal::api::VecZnxIdftApplyTmpBytes
        + poulpy_hal::api::VecZnxBigAddSmallAssign<BE>
        + poulpy_hal::api::VecZnxBigNormalize<BE>
        + poulpy_hal::api::VecZnxDftApply<BE>
        + poulpy_hal::api::VecZnxDftZero<BE>
        + poulpy_hal::api::VecZnxIdftApply<BE>,
{
    fn lwe_sample_extract_reference<R, A>(&self, res: &mut R, a: &A)
    where
        R: crate::layouts::LWEToBackendMut<BE> + crate::layouts::LWEInfos,
        A: crate::layouts::GLWEToBackendRef<BE> + crate::layouts::GLWEInfos,
    {
        crate::reference::conversion::lwe_sample_extract_reference::<BE, _, _, _>(self, res, a)
    }

    fn glwe_from_lwe_tmp_bytes_reference<R, A, K>(&self, glwe_infos: &R, lwe_infos: &A, key_infos: &K) -> usize
    where
        R: crate::layouts::GLWEInfos,
        A: crate::layouts::LWEInfos,
        K: crate::layouts::GGLWEInfos,
    {
        crate::reference::conversion::glwe_from_lwe_tmp_bytes_reference::<BE, _, _, _, _>(self, glwe_infos, lwe_infos, key_infos)
    }

    fn glwe_from_lwe_reference<R, A>(
        &self,
        res: &mut R,
        lwe: &A,
        ksk: &crate::layouts::prepared::GGLWEPreparedBackendRef<'_, BE>,
        scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, BE>,
    ) where
        R: crate::layouts::GLWEToBackendMut<BE> + crate::layouts::GLWEInfos,
        A: crate::layouts::LWEToBackendRef<BE> + crate::layouts::LWEInfos,
    {
        crate::reference::conversion::glwe_from_lwe_reference::<BE, _, _, _>(self, res, lwe, ksk, scratch)
    }

    fn glwe_expand_lwe_tmp_bytes_reference<R, A>(&self, lwe_infos: &R, a_infos: &A) -> usize
    where
        R: crate::layouts::LWEInfos,
        A: crate::layouts::GLWEInfos,
    {
        crate::reference::conversion::glwe_expand_lwe_tmp_bytes_reference::<BE, _, _, _>(self, lwe_infos, a_infos)
    }

    fn glwe_expand_lwe_reference<R, A>(&self, res: &mut [R], a: &A, scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, BE>)
    where
        R: crate::layouts::LWEToBackendMut<BE> + crate::layouts::LWEInfos,
        A: crate::layouts::GLWEToBackendRef<BE> + crate::layouts::GLWEInfos,
    {
        crate::reference::conversion::glwe_expand_lwe_reference::<BE, _, _, _>(self, res, a, scratch)
    }

    fn glwe_expand_lwe_matrix_tmp_bytes_reference<R, A>(&self, res_infos: &R, a_infos: &A) -> usize
    where
        R: crate::layouts::LWEMatrixInfos,
        A: crate::layouts::GLWEInfos,
    {
        crate::reference::conversion::glwe_expand_lwe_matrix_tmp_bytes_reference::<BE, _, _, _>(self, res_infos, a_infos)
    }

    fn glwe_expand_lwe_matrix_reference<R, A>(
        &self,
        res: &mut R,
        a: &A,
        scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, BE>,
    ) where
        R: crate::layouts::LWEMatrixToBackendMut<BE> + crate::layouts::LWEMatrixInfos,
        A: crate::layouts::GLWEToBackendRef<BE> + crate::layouts::GLWEInfos,
    {
        crate::reference::conversion::glwe_expand_lwe_matrix_reference::<BE, _, _, _>(self, res, a, scratch)
    }

    fn ggsw_expand_rows_tmp_bytes_reference<R, A>(&self, res_infos: &R, tsk_infos: &A) -> usize
    where
        R: crate::layouts::GGSWInfos,
        A: crate::layouts::GGLWEInfos,
    {
        crate::reference::conversion::ggsw_expand_rows_tmp_bytes_reference::<BE, _, _, _>(self, res_infos, tsk_infos)
    }

    fn ggsw_expand_row_reference<R>(
        &self,
        res: &mut R,
        tsk: &crate::layouts::prepared::GGLWEToGGSWKeyPreparedBackendRef<'_, BE>,
        scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, BE>,
    ) where
        R: crate::layouts::GGSWToBackendMut<BE> + crate::layouts::GGSWInfos,
    {
        crate::reference::conversion::ggsw_expand_row_reference::<BE, _, _>(self, res, tsk, scratch)
    }
}
