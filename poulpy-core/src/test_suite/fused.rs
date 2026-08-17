//! Direct parity tests for Core-owned backend fusion seams.

use crate::oep::{GGLWEProductDigitsStridedImpl, GLWETensorRank1DftImpl};
use poulpy_hal::{
    api::{
        CnvPVecAlloc, Convolution, ScratchOwnedAlloc, VecZnxAlloc, VecZnxBigAlloc, VecZnxBigNormalize,
        VecZnxBigNormalizeTmpBytes, VecZnxDftAlloc, VecZnxDftApply, VecZnxDftBytesOf, VecZnxDftCopy, VecZnxDftZero,
        VecZnxIdftApplyTmpA, VmpApplyDftToDft, VmpApplyDftToDftAccumulate, VmpApplyDftToDftTmpBytes, VmpPMatAlloc, VmpPrepare,
        VmpPrepareTmpBytes,
    },
    layouts::{
        CnvPVecLToBackendMut, CnvPVecLToBackendRef, CnvPVecRToBackendMut, CnvPVecRToBackendRef, FillUniform, HostDataMut,
        HostDataRef, MatZnxToBackendRef, Module, ScratchOwned, VecZnxBigToBackendMut, VecZnxBigToBackendRef, VecZnxDft,
        VecZnxDftToBackendMut, VecZnxDftToBackendRef, VecZnxToBackendRef, VmpPMatToBackendMut, VmpPMatToBackendRef,
    },
    source::Source,
    test_suite::{download_vec_znx, upload_vec_znx, vec_znx_backend_mut, vec_znx_backend_ref},
};

/// Verifies the Core rank-one tensor hook against generic convolutions.
pub fn test_glwe_tensor_rank1_dft<BE>(module: &Module<BE>, base2k: usize)
where
    BE: poulpy_hal::test_suite::TestBackend<OwnedBuf = Vec<u8>> + GLWETensorRank1DftImpl<BE>,
    Module<BE>: CnvPVecAlloc<BE>
        + Convolution<BE>
        + VecZnxAlloc<BE>
        + VecZnxDftAlloc<BE>
        + VecZnxIdftApplyTmpA<BE>
        + VecZnxBigNormalize<BE>
        + VecZnxBigNormalizeTmpBytes
        + VecZnxBigAlloc<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE>,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
{
    let mut source = Source::new([19u8; 32]);
    let size = 15;
    let res_size = 2 * size;
    let mut a = module.vec_znx_alloc(2, size);
    let mut b = module.vec_znx_alloc(2, size);
    a.fill_uniform(17, &mut source);
    b.fill_uniform(17, &mut source);
    let a_backend = upload_vec_znx::<BE>(&a);
    let b_backend = upload_vec_znx::<BE>(&b);
    let mut a_prep = module.cnv_pvec_left_alloc(2, size);
    let mut b_prep = module.cnv_pvec_right_alloc(2, size);
    let scratch_bytes = BE::glwe_tensor_rank1_dft_tmp_bytes(module, 0, res_size, size, size)
        .max(module.cnv_apply_dft_tmp_bytes(0, res_size, size, size))
        .max(module.cnv_prepare_left_tmp_bytes(size, size))
        .max(module.cnv_prepare_right_tmp_bytes(size, size))
        .max(module.vec_znx_big_normalize_tmp_bytes());
    let mut scratch = ScratchOwned::<BE>::alloc(scratch_bytes);
    module.cnv_prepare_left(
        &mut a_prep.to_backend_mut(),
        &vec_znx_backend_ref::<BE>(&a_backend),
        !0,
        &mut scratch.arena(),
    );
    module.cnv_prepare_right(
        &mut b_prep.to_backend_mut(),
        &vec_znx_backend_ref::<BE>(&b_backend),
        !0,
        &mut scratch.arena(),
    );

    for offset in 0..res_size {
        let mut have_dft = module.vec_znx_dft_alloc(3, res_size);
        let mut want_dft = module.vec_znx_dft_alloc(3, res_size);
        BE::glwe_tensor_rank1_dft(
            module,
            offset,
            &mut have_dft.to_backend_mut(),
            &a_prep.to_backend_ref(),
            &b_prep.to_backend_ref(),
            &mut scratch.arena(),
        );
        module.cnv_apply_dft(
            offset,
            &mut want_dft.to_backend_mut(),
            0,
            &a_prep.to_backend_ref(),
            0,
            &b_prep.to_backend_ref(),
            0,
            &mut scratch.arena(),
        );
        module.cnv_apply_dft(
            offset,
            &mut want_dft.to_backend_mut(),
            1,
            &a_prep.to_backend_ref(),
            0,
            &b_prep.to_backend_ref(),
            1,
            &mut scratch.arena(),
        );
        module.cnv_apply_dft_accumulate(
            offset,
            &mut want_dft.to_backend_mut(),
            1,
            &a_prep.to_backend_ref(),
            1,
            &b_prep.to_backend_ref(),
            0,
            &mut scratch.arena(),
        );
        module.cnv_apply_dft(
            offset,
            &mut want_dft.to_backend_mut(),
            2,
            &a_prep.to_backend_ref(),
            1,
            &b_prep.to_backend_ref(),
            1,
            &mut scratch.arena(),
        );

        let mut have_big = module.vec_znx_big_alloc(1, res_size);
        let mut want_big = module.vec_znx_big_alloc(1, res_size);
        let template = module.vec_znx_alloc(3, res_size);
        let mut have_backend = upload_vec_znx::<BE>(&template);
        let mut want_backend = upload_vec_znx::<BE>(&template);
        for col in 0..3 {
            module.vec_znx_idft_apply_tmpa(&mut have_big.to_backend_mut(), 0, &mut have_dft.to_backend_mut(), col);
            module.vec_znx_idft_apply_tmpa(&mut want_big.to_backend_mut(), 0, &mut want_dft.to_backend_mut(), col);
            module.vec_znx_big_normalize(
                &mut vec_znx_backend_mut::<BE>(&mut have_backend),
                base2k,
                0,
                col,
                &have_big.to_backend_ref(),
                base2k,
                0,
                &mut scratch.arena(),
            );
            module.vec_znx_big_normalize(
                &mut vec_znx_backend_mut::<BE>(&mut want_backend),
                base2k,
                0,
                col,
                &want_big.to_backend_ref(),
                base2k,
                0,
                &mut scratch.arena(),
            );
        }
        assert_eq!(download_vec_znx::<BE>(&have_backend), download_vec_znx::<BE>(&want_backend));
    }
}

/// Verifies the Core interleaved-digit hook, including overwrite and sparse
/// leading-row behavior, against the sequential HAL composition.
pub fn test_gglwe_product_digits_strided<BE>(module: &Module<BE>, base2k: usize)
where
    BE: poulpy_hal::test_suite::TestBackend + GGLWEProductDigitsStridedImpl<BE>,
    BE::OwnedBuf: HostDataMut,
    Module<BE>: VecZnxDftAlloc<BE>
        + VecZnxDftApply<BE>
        + VecZnxDftBytesOf
        + VecZnxDftCopy<BE>
        + VecZnxDftZero<BE>
        + VmpApplyDftToDft<BE>
        + VmpApplyDftToDftAccumulate<BE>
        + VmpApplyDftToDftTmpBytes
        + VmpPMatAlloc<BE>
        + VmpPrepare<BE>
        + VmpPrepareTmpBytes,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE>,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
{
    let mut source = Source::new([2u8; 32]);
    let cases: [(usize, usize, usize, usize); 6] = [
        (2, 1, 2, 4),
        (2, 2, 1, 5),
        (3, 1, 1, 7),
        (3, 2, 2, 2),
        (2, 1, 1, 1),
        (3, 2, 1, 8),
    ];

    for (dsize, cols_in, cols_out, a_size, sparse) in cases.into_iter().flat_map(|(dsize, cols_in, cols_out, a_size)| {
        [
            (dsize, cols_in, cols_out, a_size, false),
            (dsize, cols_in, cols_out, a_size, true),
        ]
    }) {
        let rows = a_size.div_ceil(dsize);
        let size_out = a_size;
        let product_tmp_bytes = module.bytes_of_vec_znx_dft(cols_in, rows)
            + module.bytes_of_vec_znx_dft(cols_out, size_out)
            + module.vmp_apply_dft_to_dft_tmp_bytes(size_out, rows, rows, cols_in, cols_out, size_out);
        let mut scratch =
            ScratchOwned::<BE>::alloc(product_tmp_bytes.max(module.vmp_prepare_tmp_bytes(rows, cols_in, cols_out, size_out)));

        let mut a = module.vec_znx_alloc(cols_in, a_size);
        a.fill_uniform(base2k, &mut source);
        let mut a_dft = module.vec_znx_dft_alloc(cols_in, a_size);
        for col in 0..cols_in {
            module.vec_znx_dft_apply(
                1,
                0,
                &mut a_dft.to_backend_mut(),
                col,
                &VecZnxToBackendRef::<BE>::to_backend_ref(&a),
                col,
            );
        }
        if sparse && a_size > 1 {
            let prefix_size = a_size - 1;
            let n = a_dft.n();
            let len = BE::bytes_of_vec_znx_dft(n, cols_in, prefix_size);
            let data = BE::region_mut(&mut a_dft.data, 0, len);
            let mut prefix = VecZnxDft::from_data(data, n, cols_in, prefix_size);
            for col in 0..cols_in {
                module.vec_znx_dft_zero(&mut prefix, col);
            }
        }

        let mut mat = module.mat_znx_alloc(rows, cols_in, cols_out, size_out);
        mat.fill_uniform(base2k, &mut source);
        let mut pmat = module.vmp_pmat_alloc(rows, cols_in, cols_out, size_out);
        module.vmp_prepare(
            &mut pmat.to_backend_mut(),
            &MatZnxToBackendRef::<BE>::to_backend_ref(&mat),
            &mut scratch.arena(),
        );

        let mut want = module.vec_znx_dft_alloc(cols_out, size_out);
        let sentinel = vec![1u8; BE::len_bytes(&want.data)];
        BE::copy_from_host(&mut want.data, &sentinel);
        crate::oep::gglwe_product_digits_strided_default(
            module,
            &mut want.to_backend_mut(),
            &a_dft.to_backend_ref(),
            dsize,
            &pmat.to_backend_ref(),
            &mut scratch.arena(),
        );

        let mut have = module.vec_znx_dft_alloc(cols_out, size_out);
        BE::copy_from_host(&mut have.data, &sentinel);
        BE::gglwe_product_digits_strided(
            module,
            &mut have.to_backend_mut(),
            &a_dft.to_backend_ref(),
            dsize,
            &pmat.to_backend_ref(),
            &mut scratch.arena(),
        );

        let want = BE::to_host_bytes(&want.data);
        let have = BE::to_host_bytes(&have.data);
        assert_ne!(want, sentinel, "reference VMP did not overwrite the destination");
        assert_eq!(
            have, want,
            "strided VMP mismatch for dsize={dsize}, a_size={a_size}, sparse={sparse}"
        );
    }
}
