//! Cross-backend tests for the `pmat` tier of the VMP apply matrix.
use super::*;

vmp_to_dft_test!(
    test_vmp_apply_pmat_small_to_dft,
    vmp_apply_pmat_small_to_dft,
    vmp_apply_pmat_small_to_dft_tmp_bytes,
    pmat,
    small,
    accumulate: no,
    [VmpPMatAlloc, VmpPreparePMat, VmpApplyPMatSmallToDft],
    [VmpPreparePMatTmpBytes, VmpApplyPMatSmallToDftTmpBytes]
);

vmp_to_dft_test!(
    test_vmp_apply_pmat_dft_to_dft,
    vmp_apply_pmat_dft_to_dft,
    vmp_apply_pmat_dft_to_dft_tmp_bytes,
    pmat,
    dft,
    accumulate: no,
    [VmpPMatAlloc, VmpPreparePMat, VmpApplyPMatDftToDft],
    [VmpPreparePMatTmpBytes, VmpApplyPMatDftToDftTmpBytes]
);

vmp_to_dft_test!(
    test_vmp_apply_pmat_small_to_dft_accumulate,
    vmp_apply_pmat_small_to_dft_accumulate,
    vmp_apply_pmat_small_to_dft_accumulate_tmp_bytes,
    pmat,
    small,
    accumulate: yes,
    [VmpPMatAlloc, VmpPreparePMat, VmpApplyPMatSmallToDftAccumulate],
    [VmpPreparePMatTmpBytes, VmpApplyPMatSmallToDftAccumulateTmpBytes]
);

vmp_to_dft_test!(
    test_vmp_apply_pmat_dft_to_dft_accumulate,
    vmp_apply_pmat_dft_to_dft_accumulate,
    vmp_apply_pmat_dft_to_dft_accumulate_tmp_bytes,
    pmat,
    dft,
    accumulate: yes,
    [VmpPMatAlloc, VmpPreparePMat, VmpApplyPMatDftToDftAccumulate],
    [VmpPreparePMatTmpBytes, VmpApplyPMatDftToDftAccumulateTmpBytes]
);

vmp_to_big_test!(
    test_vmp_apply_pmat_small_to_big,
    vmp_apply_pmat_small_to_big,
    vmp_apply_pmat_small_to_big_tmp_bytes,
    pmat,
    small,
    [VmpPMatAlloc, VmpPreparePMat, VmpApplyPMatSmallToBig],
    [VmpPreparePMatTmpBytes, VmpApplyPMatSmallToBigTmpBytes]
);

vmp_to_big_test!(
    test_vmp_apply_pmat_dft_to_big,
    vmp_apply_pmat_dft_to_big,
    vmp_apply_pmat_dft_to_big_tmp_bytes,
    pmat,
    dft,
    [VmpPMatAlloc, VmpPreparePMat, VmpApplyPMatDftToBig],
    [VmpPreparePMatTmpBytes, VmpApplyPMatDftToBigTmpBytes]
);

vmp_to_small_test!(
    test_vmp_apply_pmat_small_to_small,
    vmp_apply_pmat_small_to_small,
    vmp_apply_pmat_small_to_small_tmp_bytes,
    pmat,
    small,
    [VmpPMatAlloc, VmpPreparePMat, VmpApplyPMatSmallToSmall],
    [VmpPreparePMatTmpBytes, VmpApplyPMatSmallToSmallTmpBytes]
);

vmp_to_small_test!(
    test_vmp_apply_pmat_dft_to_small,
    vmp_apply_pmat_dft_to_small,
    vmp_apply_pmat_dft_to_small_tmp_bytes,
    pmat,
    dft,
    [VmpPMatAlloc, VmpPreparePMat, VmpApplyPMatDftToSmall],
    [VmpPreparePMatTmpBytes, VmpApplyPMatDftToSmallTmpBytes]
);
