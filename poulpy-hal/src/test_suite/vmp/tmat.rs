//! Cross-backend tests for the `tmat` tier of the VMP apply matrix.
use super::*;

vmp_to_dft_test!(
    test_vmp_apply_tmat_small_to_dft,
    vmp_apply_tmat_small_to_dft,
    vmp_apply_tmat_small_to_dft_tmp_bytes,
    tmat,
    small,
    accumulate: no,
    [VmpTMatAlloc, VmpPrepareTMat, VmpApplyTMatSmallToDft],
    [VmpPrepareTMatTmpBytes, VmpApplyTMatSmallToDftTmpBytes]
);

vmp_to_dft_test!(
    test_vmp_apply_tmat_dft_to_dft,
    vmp_apply_tmat_dft_to_dft,
    vmp_apply_tmat_dft_to_dft_tmp_bytes,
    tmat,
    dft,
    accumulate: no,
    [VmpTMatAlloc, VmpPrepareTMat, VmpApplyTMatDftToDft],
    [VmpPrepareTMatTmpBytes, VmpApplyTMatDftToDftTmpBytes]
);

vmp_to_dft_test!(
    test_vmp_apply_tmat_small_to_dft_accumulate,
    vmp_apply_tmat_small_to_dft_accumulate,
    vmp_apply_tmat_small_to_dft_accumulate_tmp_bytes,
    tmat,
    small,
    accumulate: yes,
    [VmpTMatAlloc, VmpPrepareTMat, VmpApplyTMatSmallToDftAccumulate],
    [VmpPrepareTMatTmpBytes, VmpApplyTMatSmallToDftAccumulateTmpBytes]
);

vmp_to_dft_test!(
    test_vmp_apply_tmat_dft_to_dft_accumulate,
    vmp_apply_tmat_dft_to_dft_accumulate,
    vmp_apply_tmat_dft_to_dft_accumulate_tmp_bytes,
    tmat,
    dft,
    accumulate: yes,
    [VmpTMatAlloc, VmpPrepareTMat, VmpApplyTMatDftToDftAccumulate],
    [VmpPrepareTMatTmpBytes, VmpApplyTMatDftToDftAccumulateTmpBytes]
);

vmp_to_big_test!(
    test_vmp_apply_tmat_small_to_big,
    vmp_apply_tmat_small_to_big,
    vmp_apply_tmat_small_to_big_tmp_bytes,
    tmat,
    small,
    [VmpTMatAlloc, VmpPrepareTMat, VmpApplyTMatSmallToBig],
    [VmpPrepareTMatTmpBytes, VmpApplyTMatSmallToBigTmpBytes]
);

vmp_to_big_test!(
    test_vmp_apply_tmat_dft_to_big,
    vmp_apply_tmat_dft_to_big,
    vmp_apply_tmat_dft_to_big_tmp_bytes,
    tmat,
    dft,
    [VmpTMatAlloc, VmpPrepareTMat, VmpApplyTMatDftToBig],
    [VmpPrepareTMatTmpBytes, VmpApplyTMatDftToBigTmpBytes]
);

vmp_to_small_test!(
    test_vmp_apply_tmat_small_to_small,
    vmp_apply_tmat_small_to_small,
    vmp_apply_tmat_small_to_small_tmp_bytes,
    tmat,
    small,
    [VmpTMatAlloc, VmpPrepareTMat, VmpApplyTMatSmallToSmall],
    [VmpPrepareTMatTmpBytes, VmpApplyTMatSmallToSmallTmpBytes]
);

vmp_to_small_test!(
    test_vmp_apply_tmat_dft_to_small,
    vmp_apply_tmat_dft_to_small,
    vmp_apply_tmat_dft_to_small_tmp_bytes,
    tmat,
    dft,
    [VmpTMatAlloc, VmpPrepareTMat, VmpApplyTMatDftToSmall],
    [VmpPrepareTMatTmpBytes, VmpApplyTMatDftToSmallTmpBytes]
);
