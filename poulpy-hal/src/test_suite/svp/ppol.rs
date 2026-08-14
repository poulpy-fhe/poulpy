//! Cross-backend tests for the `ppol` tier of the SVP apply matrix.
use super::*;

svp_to_dft_test!(
    test_svp_apply_ppol_small_to_dft,
    svp_apply_ppol_small_to_dft,
    ppol,
    small,
    [SvpPPolAlloc, SvpPreparePPol],
    SvpApplyPPolSmallToDft
);

svp_to_dft_test!(
    test_svp_apply_ppol_dft_to_dft,
    svp_apply_ppol_dft_to_dft,
    ppol,
    dft,
    [SvpPPolAlloc, SvpPreparePPol],
    SvpApplyPPolDftToDft
);

svp_assign_test!(
    test_svp_apply_ppol_dft_to_dft_assign,
    svp_apply_ppol_dft_to_dft_assign,
    ppol,
    [SvpPPolAlloc, SvpPreparePPol],
    SvpApplyPPolDftToDftAssign
);

svp_to_big_test!(
    test_svp_apply_ppol_small_to_big,
    svp_apply_ppol_small_to_big,
    ppol,
    small,
    [SvpPPolAlloc, SvpPreparePPol],
    SvpApplyPPolSmallToBig
);

svp_to_big_test!(
    test_svp_apply_ppol_dft_to_big,
    svp_apply_ppol_dft_to_big,
    ppol,
    dft,
    [SvpPPolAlloc, SvpPreparePPol],
    SvpApplyPPolDftToBig
);

svp_to_small_test!(
    test_svp_apply_ppol_small_to_small,
    svp_apply_ppol_small_to_small,
    ppol,
    small,
    [SvpPPolAlloc, SvpPreparePPol],
    SvpApplyPPolSmallToSmall
);

svp_to_small_test!(
    test_svp_apply_ppol_dft_to_small,
    svp_apply_ppol_dft_to_small,
    ppol,
    dft,
    [SvpPPolAlloc, SvpPreparePPol],
    SvpApplyPPolDftToSmall
);
