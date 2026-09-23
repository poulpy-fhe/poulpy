use crate::bdd_arithmetic::*;
use poulpy_core::{layouts::*, *};
use poulpy_hal::{api::*, layouts::*};
#[allow(clippy::too_many_arguments)]
/// Independently callable canonical implementation of [`Cswap::cswap_tmp_bytes`].
pub fn cswap_tmp_bytes_reference<BE: Backend<ZnxWord = i64>, R, A, S>(
    module: &Module<BE>,
    res_a_infos: &R,
    res_b_infos: &A,
    s_infos: &S,
) -> usize
where
    R: GLWEInfos,
    A: GLWEInfos,
    S: GGSWInfos,
    Module<BE>: GLWEBytesOf<BE>
        + Sized
        + ModuleN
        + GLWEExternalProductInternal<BE>
        + GLWESub<BE>
        + GLWECopy<BE>
        + GLWENormalize<BE>
        + VecZnxBigAddSmall<BE>
        + VecZnxBigBytesOf
        + VecZnxBigNormalize<BE>
        + VecZnxBigNormalizeTmpBytes
        + VecZnxBigSubSmallA<BE>
        + VecZnxDftAddAssign<BE>
        + VecZnxDftApply<BE>
        + VecZnxDftBytesOf
        + VecZnxDftZero<BE>
        + VecZnxIdftApply<BE>
        + VecZnxIdftApplyTmpBytes
        + VecZnxNormalizeTmpBytes
        + VmpApplyDftToDft<BE>
        + VmpApplyDftToDftTmpBytes,
{
    let tmp_c_infos = GLWELayout {
        n: s_infos.n(),
        base2k: s_infos.base2k(),
        k: res_a_infos.k().max(res_b_infos.k()),
        rank: s_infos.rank(),
    };
    let output_size = glwe_external_product_output_size::<BE, _, _, _>(res_a_infos, &tmp_c_infos, s_infos);
    let res_dft: usize = module.bytes_of_vec_znx_dft(module.n(), (s_infos.rank() + 1).into(), output_size);
    let mut tot = res_dft
        + (module.glwe_external_product_internal_tmp_bytes(res_a_infos, &tmp_c_infos, s_infos)
            + module.glwe_bytes_of_from_infos(&tmp_c_infos))
        .max(module.vec_znx_big_normalize_tmp_bytes());

    if res_a_infos.base2k() != s_infos.base2k() {
        tot += module.glwe_bytes_of_from_infos(&GLWELayout {
            n: res_a_infos.n(),
            base2k: s_infos.base2k(),
            k: res_a_infos.k(),
            rank: res_a_infos.rank(),
        });
        tot += module.glwe_bytes_of_from_infos(&GLWELayout {
            n: res_b_infos.n(),
            base2k: s_infos.base2k(),
            k: res_b_infos.k(),
            rank: res_b_infos.rank(),
        });
    } else {
        tot += module.glwe_bytes_of_from_infos(res_a_infos);
        tot += module.glwe_bytes_of_from_infos(res_b_infos);
    }

    // The full BIG output remains live during external product and IDFT.
    tot + module.bytes_of_vec_znx_big(module.n(), 1, output_size)
        + module.bytes_of_vec_znx_big(module.n(), (s_infos.rank() + 1).into(), output_size)
        + module.vec_znx_idft_apply_tmp_bytes()
        + module.glwe_normalize_tmp_bytes()
        + module
            .glwe_copy_tmp_bytes(res_a_infos, res_a_infos)
            .max(module.glwe_copy_tmp_bytes(res_b_infos, res_b_infos))
}
#[allow(clippy::too_many_arguments)]
/// Independently callable canonical implementation of [`Cswap::cswap`].
pub fn cswap_reference<BE, A, B>(
    module: &Module<BE>,
    res_a: &mut A,
    res_b: &mut B,
    s: &GGSWPreparedBackendRef<'_, BE>,
    scratch: &mut ScratchArena<'_, BE>,
) where
    A: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + GLWEInfos,
    B: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + GLWEInfos,
    BE: Backend<ZnxWord = i64>,
    Module<BE>: GLWEBytesOf<BE>
        + Sized
        + ModuleN
        + GLWEExternalProductInternal<BE>
        + GLWESub<BE>
        + GLWECopy<BE>
        + GLWENormalize<BE>
        + VecZnxBigAddSmall<BE>
        + VecZnxBigBytesOf
        + VecZnxBigNormalize<BE>
        + VecZnxBigNormalizeTmpBytes
        + VecZnxBigSubSmallA<BE>
        + VecZnxDftAddAssign<BE>
        + VecZnxDftApply<BE>
        + VecZnxDftBytesOf
        + VecZnxDftZero<BE>
        + VecZnxIdftApply<BE>
        + VecZnxIdftApplyTmpBytes
        + VecZnxNormalizeTmpBytes
        + VmpApplyDftToDft<BE>
        + VmpApplyDftToDftTmpBytes,
{
    assert_eq!(res_a.base2k(), res_b.base2k());
    assert_eq!(res_a.n(), module.n() as u32);
    assert_eq!(res_b.n(), module.n() as u32);
    assert_eq!(res_a.rank(), s.rank());
    assert_eq!(res_b.rank(), s.rank());

    let scratch = scratch.borrow();
    assert!(
        scratch.available() >= cswap_tmp_bytes_reference(module, res_a, res_b, s),
        "scratch.available(): {} < Cswap::cswap_tmp_bytes: {}",
        scratch.available(),
        cswap_tmp_bytes_reference(module, res_a, res_b, s)
    );

    let res_base2k: usize = res_a.base2k().as_usize();
    let res_a_k = res_a.k().as_usize();
    let res_b_k = res_b.k().as_usize();
    let s_base2k: usize = s.base2k().as_usize();
    let cols: usize = (s.rank() + 1).into();
    let tmp_c_infos = GLWELayout {
        n: s.n(),
        base2k: s.base2k(),
        k: res_a.k().max(res_b.k()),
        rank: s.rank(),
    };
    let output_size = glwe_external_product_output_size::<BE, _, _, _>(res_a, &tmp_c_infos, s);

    if res_base2k == s_base2k {
        let (mut a_prev, scratch_1) = scratch.take_glwe_scratch(res_a);
        let (mut b_prev, scratch_2) = scratch_1.take_glwe_scratch(res_b);
        let (mut res_dft, scratch_3) = scratch_2.take_vec_znx_dft_scratch(module.n(), cols, output_size);
        let (res_big_tmp, mut scratch_4) = scratch_3.take_vec_znx_big_scratch(module.n(), 1, output_size);
        module.glwe_copy(&mut a_prev, res_a, &mut scratch_4);
        module.glwe_copy(&mut b_prev, res_b, &mut scratch_4);

        let (res_big, mut scratch_norm): (VecZnxBigViewMut<'_, BE>, _);
        {
            let (mut tmp_c, scratch_5) = scratch_4.take_glwe_scratch(&tmp_c_infos);
            module.glwe_sub(&mut tmp_c, res_b, res_a);
            let (tmp_res_big, mut scratch_6) = scratch_5.take_vec_znx_big_scratch(module.n(), cols, output_size);
            let mut tmp_res_big = tmp_res_big;
            module.glwe_external_product_dft(&mut res_dft, &tmp_c, s, &mut scratch_6.borrow());
            let res_dft_ref = vec_znx_dft_backend_ref_from_mut::<BE>(&res_dft);
            for col in 0..cols {
                module.vec_znx_idft_apply(&mut tmp_res_big, col, &res_dft_ref, col, &mut scratch_6.borrow());
            }
            (res_big, scratch_norm) = (tmp_res_big, scratch_6);
        }

        let mut res_big_tmp = res_big_tmp;
        let res_big_ref = vec_znx_big_backend_ref_from_mut::<BE>(&res_big);
        let mut res_a_backend = res_a.to_backend_mut();

        for j in 0..cols {
            module.vec_znx_big_add_small(
                &mut res_big_tmp,
                0,
                &res_big_ref,
                j,
                &vec_znx_backend_ref_from_mut::<BE>(a_prev.data()),
                j,
            );
            let res_big_tmp_ref = vec_znx_big_backend_ref_from_mut::<BE>(&res_big_tmp);
            module.vec_znx_big_normalize(
                res_a_backend.data_mut(),
                res_base2k,
                res_a_k,
                0,
                j,
                &res_big_tmp_ref,
                s_base2k,
                0,
                &mut scratch_norm.borrow(),
            );
        }

        let mut res_b_backend = res_b.to_backend_mut();
        for j in 0..cols {
            module.vec_znx_big_sub_small_a(
                &mut res_big_tmp,
                0,
                &vec_znx_backend_ref_from_mut::<BE>(b_prev.data()),
                j,
                &res_big_ref,
                j,
            );
            let res_big_tmp_ref = vec_znx_big_backend_ref_from_mut::<BE>(&res_big_tmp);
            module.vec_znx_big_normalize(
                res_b_backend.data_mut(),
                res_base2k,
                res_b_k,
                0,
                j,
                &res_big_tmp_ref,
                s_base2k,
                0,
                &mut scratch_norm.borrow(),
            );
        }
    } else {
        let (mut tmp_a, scratch_1) = scratch.take_glwe_scratch(&GLWELayout {
            n: res_a.n(),
            base2k: s.base2k(),
            k: res_a.k(),
            rank: res_a.rank(),
        });
        let (mut tmp_b, mut scratch_2) = scratch_1.take_glwe_scratch(&GLWELayout {
            n: res_b.n(),
            base2k: s.base2k(),
            k: res_b.k(),
            rank: res_b.rank(),
        });

        module.glwe_normalize(&mut tmp_a, res_a, &mut scratch_2);
        module.glwe_normalize(&mut tmp_b, res_b, &mut scratch_2);

        let (mut res_dft, scratch_3) = scratch_2.take_vec_znx_dft_scratch(module.n(), cols, output_size);
        let (res_big_tmp, scratch_4) = scratch_3.take_vec_znx_big_scratch(module.n(), 1, output_size);

        let (res_big, mut scratch_norm): (VecZnxBigViewMut<'_, BE>, _);
        {
            let (mut tmp_c, scratch_5) = scratch_4.take_glwe_scratch(&tmp_c_infos);
            module.glwe_sub(&mut tmp_c, &tmp_b, &tmp_a);
            let (tmp_res_big, mut scratch_6) = scratch_5.take_vec_znx_big_scratch(module.n(), cols, output_size);
            let mut tmp_res_big = tmp_res_big;
            module.glwe_external_product_dft(&mut res_dft, &tmp_c, s, &mut scratch_6.borrow());
            let res_dft_ref = vec_znx_dft_backend_ref_from_mut::<BE>(&res_dft);
            for col in 0..cols {
                module.vec_znx_idft_apply(&mut tmp_res_big, col, &res_dft_ref, col, &mut scratch_6.borrow());
            }
            (res_big, scratch_norm) = (tmp_res_big, scratch_6);
        }

        let mut res_big_tmp = res_big_tmp;
        let res_big_ref = vec_znx_big_backend_ref_from_mut::<BE>(&res_big);
        let mut res_a_backend = res_a.to_backend_mut();

        for j in 0..cols {
            module.vec_znx_big_add_small(
                &mut res_big_tmp,
                0,
                &res_big_ref,
                j,
                &vec_znx_backend_ref_from_mut::<BE>(tmp_a.data()),
                j,
            );
            let res_big_tmp_ref = vec_znx_big_backend_ref_from_mut::<BE>(&res_big_tmp);
            module.vec_znx_big_normalize(
                res_a_backend.data_mut(),
                res_base2k,
                res_a_k,
                0,
                j,
                &res_big_tmp_ref,
                s_base2k,
                0,
                &mut scratch_norm.borrow(),
            );
        }

        let mut res_b_backend = res_b.to_backend_mut();
        for j in 0..cols {
            module.vec_znx_big_sub_small_a(
                &mut res_big_tmp,
                0,
                &vec_znx_backend_ref_from_mut::<BE>(tmp_b.data()),
                j,
                &res_big_ref,
                j,
            );
            let res_big_tmp_ref = vec_znx_big_backend_ref_from_mut::<BE>(&res_big_tmp);
            module.vec_znx_big_normalize(
                res_b_backend.data_mut(),
                res_base2k,
                res_b_k,
                0,
                j,
                &res_big_tmp_ref,
                s_base2k,
                0,
                &mut scratch_norm.borrow(),
            );
        }
    }
}
