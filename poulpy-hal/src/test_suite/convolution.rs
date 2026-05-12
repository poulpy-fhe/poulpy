use super::{download_vec_znx, upload_vec_znx, vec_znx_backend_mut, vec_znx_backend_ref};
use rand::Rng;

use crate::{
    api::{
        CnvPVecAlloc, Convolution, ModuleN, ScratchOwnedAlloc, VecZnxAddIntoBackend, VecZnxAlloc, VecZnxBigAlloc,
        VecZnxBigNormalize, VecZnxBigNormalizeTmpBytes, VecZnxCopyBackend, VecZnxDftAlloc, VecZnxDftApply, VecZnxIdftApplyTmpA,
        VecZnxNormalizeAssignBackend,
    },
    layouts::{
        Backend, CnvPVecL, CnvPVecLToBackendMut, CnvPVecLToBackendRef, CnvPVecR, CnvPVecRToBackendMut, CnvPVecRToBackendRef,
        FillUniform, ScratchArena, ScratchOwned, VecZnx, VecZnxBig, VecZnxBigToBackendMut, VecZnxBigToBackendRef, VecZnxDft,
        VecZnxDftToBackendMut, ZnxView, ZnxViewMut, ZnxZero,
    },
    source::Source,
};

type VecZnxBigOwned<BE> = VecZnxBig<<BE as Backend>::OwnedBuf, BE>;
type VecZnxDftOwned<BE> = VecZnxDft<<BE as Backend>::OwnedBuf, BE>;
type CnvPVecLOwned<BE> = CnvPVecL<<BE as Backend>::OwnedBuf, BE>;
type CnvPVecROwned<BE> = CnvPVecR<<BE as Backend>::OwnedBuf, BE>;

pub fn test_convolution_by_const<M, BE: crate::test_suite::TestBackend<OwnedBuf = Vec<u8>>>(module: &M, base2k: usize)
where
    M: ModuleN
        + Convolution<BE>
        + VecZnxAlloc<BE>
        + VecZnxBigNormalize<BE>
        + VecZnxBigNormalizeTmpBytes
        + VecZnxNormalizeAssignBackend<BE>
        + VecZnxBigAlloc<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE>,
{
    let mut source: Source = Source::new([0u8; 32]);

    let a_cols: usize = 2;
    let a_size: usize = 15;
    let b_size: usize = 15;
    let res_size: usize = a_size + b_size;

    let mut a: VecZnx<Vec<u8>> = module.vec_znx_alloc(a_cols, a_size);
    let mut b: VecZnx<Vec<u8>> = module.vec_znx_alloc(1, b_size);

    let mut res_want: VecZnx<Vec<u8>> = module.vec_znx_alloc(1, res_size);
    let mut res_big: VecZnxBigOwned<BE> = module.vec_znx_big_alloc(1, res_size);

    a.fill_uniform(17, &mut source);

    let mask = (1 << base2k) - 1;
    for j in 0..1 {
        let r = source.next_u64() & mask;
        b.at_mut(0, j)[0] = ((r << (64 - 17)) as i64) >> (64 - 17);
    }

    let a_backend = upload_vec_znx::<BE>(&a);
    let b_backend = upload_vec_znx::<BE>(&b);
    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
        module
            .cnv_by_const_apply_tmp_bytes(0, res_size, a_size, b_size)
            .max(module.vec_znx_big_normalize_tmp_bytes()),
    );

    for a_col in 0..a.cols() {
        for cnv_offset in 0..res_size {
            module.cnv_by_const_apply(
                cnv_offset,
                &mut res_big.to_backend_mut(),
                0,
                &vec_znx_backend_ref::<BE>(&a_backend),
                a_col,
                &vec_znx_backend_ref::<BE>(&b_backend),
                0,
                0,
                &mut scratch.arena(),
            );

            let res_host_template: VecZnx<Vec<u8>> = module.vec_znx_alloc(1, res_size);
            let mut res_have_backend = upload_vec_znx::<BE>(&res_host_template);
            module.vec_znx_big_normalize(
                &mut vec_znx_backend_mut::<BE>(&mut res_have_backend),
                base2k,
                0,
                0,
                &res_big.to_backend_ref(),
                base2k,
                0,
                &mut scratch.arena(),
            );
            let res_have = download_vec_znx::<BE>(&res_have_backend);

            bivariate_convolution_naive(
                module,
                base2k,
                (cnv_offset + 1) as i64,
                &mut res_want,
                0,
                &a,
                a_col,
                &b,
                0,
                &mut scratch.arena(),
            );

            assert_eq!(res_want, res_have);
        }
    }
}

pub fn test_convolution<M, BE: crate::test_suite::TestBackend<OwnedBuf = Vec<u8>>>(module: &M, base2k: usize)
where
    M: ModuleN
        + Convolution<BE>
        + CnvPVecAlloc<BE>
        + VecZnxAlloc<BE>
        + VecZnxDftAlloc<BE>
        + VecZnxDftApply<BE>
        + VecZnxIdftApplyTmpA<BE>
        + VecZnxBigNormalize<BE>
        + VecZnxBigNormalizeTmpBytes
        + VecZnxNormalizeAssignBackend<BE>
        + VecZnxBigAlloc<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE>,
{
    let mut source: Source = Source::new([0u8; 32]);

    let a_cols: usize = 2;
    let b_cols: usize = 2;
    let a_size: usize = 15;
    let b_size: usize = 15;
    let res_size: usize = a_size + b_size;

    let mut a: VecZnx<Vec<u8>> = module.vec_znx_alloc(a_cols, a_size);
    let mut b: VecZnx<Vec<u8>> = module.vec_znx_alloc(b_cols, b_size);

    let mut res_want: VecZnx<Vec<u8>> = module.vec_znx_alloc(1, res_size);
    let mut res_dft: VecZnxDftOwned<BE> = module.vec_znx_dft_alloc(1, res_size);
    let mut res_big: VecZnxBigOwned<BE> = module.vec_znx_big_alloc(1, res_size);

    a.fill_uniform(17, &mut source);
    b.fill_uniform(17, &mut source);

    let a_backend = upload_vec_znx::<BE>(&a);
    let b_backend = upload_vec_znx::<BE>(&b);

    let mut a_prep: CnvPVecLOwned<BE> = module.cnv_pvec_left_alloc(a_cols, a_size);
    let mut b_prep: CnvPVecROwned<BE> = module.cnv_pvec_right_alloc(b_cols, b_size);

    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
        module
            .cnv_apply_dft_tmp_bytes(0, res_size, a_size, b_size)
            .max(module.cnv_prepare_left_tmp_bytes(res_size, a_size))
            .max(module.cnv_prepare_right_tmp_bytes(res_size, b_size))
            .max(module.vec_znx_big_normalize_tmp_bytes()),
    );

    {
        let mut a_prep_backend = a_prep.to_backend_mut();
        module.cnv_prepare_left(
            &mut a_prep_backend,
            &vec_znx_backend_ref::<BE>(&a_backend),
            !0i64,
            &mut scratch.arena(),
        );
    }
    {
        let mut b_prep_backend = b_prep.to_backend_mut();
        module.cnv_prepare_right(
            &mut b_prep_backend,
            &vec_znx_backend_ref::<BE>(&b_backend),
            !0i64,
            &mut scratch.arena(),
        );
    }

    for a_col in 0..a.cols() {
        for b_col in 0..b.cols() {
            for cnv_offset in 0..res_size {
                module.cnv_apply_dft(
                    cnv_offset,
                    &mut res_dft.to_backend_mut(),
                    0,
                    &a_prep.to_backend_ref(),
                    a_col,
                    &b_prep.to_backend_ref(),
                    b_col,
                    &mut scratch.arena(),
                );

                module.vec_znx_idft_apply_tmpa(&mut res_big.to_backend_mut(), 0, &mut res_dft.to_backend_mut(), 0);

                let res_host_template: VecZnx<Vec<u8>> = module.vec_znx_alloc(1, res_size);
                let mut res_have_backend = upload_vec_znx::<BE>(&res_host_template);
                module.vec_znx_big_normalize(
                    &mut vec_znx_backend_mut::<BE>(&mut res_have_backend),
                    base2k,
                    0,
                    0,
                    &res_big.to_backend_ref(),
                    base2k,
                    0,
                    &mut scratch.arena(),
                );
                let res_have = download_vec_znx::<BE>(&res_have_backend);

                bivariate_convolution_naive(
                    module,
                    base2k,
                    (cnv_offset + 1) as i64,
                    &mut res_want,
                    0,
                    &a,
                    a_col,
                    &b,
                    b_col,
                    &mut scratch.arena(),
                );

                assert_eq!(res_want, res_have);
            }
        }
    }
}

pub fn test_convolution_pairwise<M, BE: crate::test_suite::TestBackend<OwnedBuf = Vec<u8>>>(module: &M, base2k: usize)
where
    M: ModuleN
        + Convolution<BE>
        + CnvPVecAlloc<BE>
        + VecZnxAlloc<BE>
        + VecZnxDftAlloc<BE>
        + VecZnxDftApply<BE>
        + VecZnxIdftApplyTmpA<BE>
        + VecZnxBigNormalize<BE>
        + VecZnxBigNormalizeTmpBytes
        + VecZnxNormalizeAssignBackend<BE>
        + VecZnxBigAlloc<BE>
        + VecZnxAddIntoBackend<BE>
        + VecZnxCopyBackend<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE>,
{
    let mut source: Source = Source::new([0u8; 32]);

    let cols: usize = 2;
    let a_size: usize = 15;
    let b_size: usize = 15;
    let res_size: usize = a_size + b_size;

    let mut a: VecZnx<Vec<u8>> = module.vec_znx_alloc(cols, a_size);
    let mut b: VecZnx<Vec<u8>> = module.vec_znx_alloc(cols, b_size);
    let mut tmp_a: VecZnx<Vec<u8>> = module.vec_znx_alloc(1, a_size);
    let mut tmp_b: VecZnx<Vec<u8>> = module.vec_znx_alloc(1, b_size);

    let mut res_want: VecZnx<Vec<u8>> = module.vec_znx_alloc(1, res_size);
    let mut res_dft: VecZnxDftOwned<BE> = module.vec_znx_dft_alloc(1, res_size);
    let mut res_big: VecZnxBigOwned<BE> = module.vec_znx_big_alloc(1, res_size);

    a.fill_uniform(17, &mut source);
    b.fill_uniform(17, &mut source);

    let a_backend = upload_vec_znx::<BE>(&a);
    let b_backend = upload_vec_znx::<BE>(&b);

    let mut a_prep: CnvPVecLOwned<BE> = module.cnv_pvec_left_alloc(cols, a_size);
    let mut b_prep: CnvPVecROwned<BE> = module.cnv_pvec_right_alloc(cols, b_size);

    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
        module
            .cnv_pairwise_apply_dft_tmp_bytes(0, res_size, a_size, b_size)
            .max(module.cnv_prepare_left_tmp_bytes(res_size, a_size))
            .max(module.cnv_prepare_right_tmp_bytes(res_size, b_size))
            .max(module.vec_znx_big_normalize_tmp_bytes()),
    );

    {
        let mut a_prep_backend = a_prep.to_backend_mut();
        module.cnv_prepare_left(
            &mut a_prep_backend,
            &vec_znx_backend_ref::<BE>(&a_backend),
            !0i64,
            &mut scratch.arena(),
        );
    }
    {
        let mut b_prep_backend = b_prep.to_backend_mut();
        module.cnv_prepare_right(
            &mut b_prep_backend,
            &vec_znx_backend_ref::<BE>(&b_backend),
            !0i64,
            &mut scratch.arena(),
        );
    }

    for col_i in 0..cols {
        for col_j in 0..cols {
            for cnv_offset in 0..res_size {
                module.cnv_pairwise_apply_dft(
                    cnv_offset,
                    &mut res_dft.to_backend_mut(),
                    0,
                    &a_prep.to_backend_ref(),
                    &b_prep.to_backend_ref(),
                    col_i,
                    col_j,
                    &mut scratch.arena(),
                );

                module.vec_znx_idft_apply_tmpa(&mut res_big.to_backend_mut(), 0, &mut res_dft.to_backend_mut(), 0);

                let res_host_template: VecZnx<Vec<u8>> = module.vec_znx_alloc(1, res_size);
                let mut res_have_backend = upload_vec_znx::<BE>(&res_host_template);
                module.vec_znx_big_normalize(
                    &mut vec_znx_backend_mut::<BE>(&mut res_have_backend),
                    base2k,
                    0,
                    0,
                    &res_big.to_backend_ref(),
                    base2k,
                    0,
                    &mut scratch.arena(),
                );
                let res_have = download_vec_znx::<BE>(&res_have_backend);

                let mut tmp_a_backend = upload_vec_znx::<BE>(&tmp_a);
                let mut tmp_b_backend = upload_vec_znx::<BE>(&tmp_b);
                if col_i != col_j {
                    module.vec_znx_add_into_backend(
                        &mut vec_znx_backend_mut::<BE>(&mut tmp_a_backend),
                        0,
                        &vec_znx_backend_ref::<BE>(&a_backend),
                        col_i,
                        &vec_znx_backend_ref::<BE>(&a_backend),
                        col_j,
                    );
                    module.vec_znx_add_into_backend(
                        &mut vec_znx_backend_mut::<BE>(&mut tmp_b_backend),
                        0,
                        &vec_znx_backend_ref::<BE>(&b_backend),
                        col_i,
                        &vec_znx_backend_ref::<BE>(&b_backend),
                        col_j,
                    );
                } else {
                    module.vec_znx_copy_backend(
                        &mut vec_znx_backend_mut::<BE>(&mut tmp_a_backend),
                        0,
                        &vec_znx_backend_ref::<BE>(&a_backend),
                        col_i,
                    );
                    module.vec_znx_copy_backend(
                        &mut vec_znx_backend_mut::<BE>(&mut tmp_b_backend),
                        0,
                        &vec_znx_backend_ref::<BE>(&b_backend),
                        col_j,
                    );
                }

                tmp_a = download_vec_znx::<BE>(&tmp_a_backend);
                tmp_b = download_vec_znx::<BE>(&tmp_b_backend);

                bivariate_convolution_naive(
                    module,
                    base2k,
                    (cnv_offset + 1) as i64,
                    &mut res_want,
                    0,
                    &tmp_a,
                    0,
                    &tmp_b,
                    0,
                    &mut scratch.arena(),
                );

                assert_eq!(res_want, res_have);
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn bivariate_convolution_naive<'s, M, BE: crate::test_suite::TestBackend>(
    module: &M,
    base2k: usize,
    k: i64,
    res: &mut VecZnx<Vec<u8>>,
    res_col: usize,
    a: &VecZnx<Vec<u8>>,
    a_col: usize,
    b: &VecZnx<Vec<u8>>,
    b_col: usize,
    scratch: &mut ScratchArena<'s, BE>,
) where
    M: VecZnxNormalizeAssignBackend<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE>,
{
    for j in 0..res.size() {
        res.zero_at(res_col, j);
    }

    for a_limb in 0..a.size() {
        for b_limb in 0..b.size() {
            let res_scale_abs = k.unsigned_abs() as usize;

            let mut res_limb: usize = a_limb + b_limb + 1;

            if k <= 0 {
                res_limb += res_scale_abs;

                if res_limb < res.size() {
                    negacyclic_convolution_naive_add(res.at_mut(res_col, res_limb), a.at(a_col, a_limb), b.at(b_col, b_limb));
                }
            } else if res_limb >= res_scale_abs {
                res_limb -= res_scale_abs;

                if res_limb < res.size() {
                    negacyclic_convolution_naive_add(res.at_mut(res_col, res_limb), a.at(a_col, a_limb), b.at(b_col, b_limb));
                }
            }
        }
    }

    let mut res_backend = upload_vec_znx::<BE>(res);
    module.vec_znx_normalize_assign_backend(base2k, &mut vec_znx_backend_mut::<BE>(&mut res_backend), res_col, scratch);
    *res = download_vec_znx::<BE>(&res_backend);
}

fn bivariate_tensoring_naive<'s, M, BE: crate::test_suite::TestBackend>(
    module: &M,
    base2k: usize,
    k: i64,
    res: &mut VecZnx<Vec<u8>>,
    a: &VecZnx<Vec<u8>>,
    b: &VecZnx<Vec<u8>>,
    scratch: &mut ScratchArena<'s, BE>,
) where
    M: VecZnxNormalizeAssignBackend<BE>,
{
    let cols = res.cols();

    assert!(res.cols() >= a.cols() + b.cols() - 1);

    res.zero();

    for a_col in 0..a.cols() {
        for a_limb in 0..a.size() {
            for b_col in 0..b.cols() {
                for b_limb in 0..b.size() {
                    let res_scale_abs = k.unsigned_abs() as usize;

                    let mut res_limb: usize = a_limb + b_limb + 1;

                    if k <= 0 {
                        res_limb += res_scale_abs;

                        if res_limb < res.size() {
                            negacyclic_convolution_naive_add(
                                res.at_mut(a_col + b_col, res_limb),
                                a.at(a_col, a_limb),
                                b.at(b_col, b_limb),
                            );
                        }
                    } else if res_limb >= res_scale_abs {
                        res_limb -= res_scale_abs;

                        if res_limb < res.size() {
                            negacyclic_convolution_naive_add(
                                res.at_mut(a_col + b_col, res_limb),
                                a.at(a_col, a_limb),
                                b.at(b_col, b_limb),
                            );
                        }
                    }
                }
            }
        }
    }

    let mut res_backend = upload_vec_znx::<BE>(res);
    for i in 0..cols {
        module.vec_znx_normalize_assign_backend(base2k, &mut vec_znx_backend_mut::<BE>(&mut res_backend), i, scratch);
    }
    *res = download_vec_znx::<BE>(&res_backend);
}

fn negacyclic_convolution_naive_add(res: &mut [i64], a: &[i64], b: &[i64]) {
    let n: usize = res.len();
    for i in 0..n {
        let ai: i64 = a[i];
        let lim: usize = n - i;
        for j in 0..lim {
            res[i + j] += ai * b[j];
        }
        for j in lim..n {
            res[i + j - n] -= ai * b[j];
        }
    }
}

fn negacyclic_convolution_naive(res: &mut [i64], a: &[i64], b: &[i64]) {
    let n: usize = res.len();
    res.fill(0);
    for i in 0..n {
        let ai: i64 = a[i];
        let lim: usize = n - i;
        for j in 0..lim {
            res[i + j] += ai * b[j];
        }
        for j in lim..n {
            res[i + j - n] -= ai * b[j];
        }
    }
}
