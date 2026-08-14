//! Cross-backend tests for the `tpol` tier of the SVP apply matrix.
use super::*;

svp_to_dft_test!(
    test_svp_apply_tpol_small_to_dft,
    svp_apply_tpol_small_to_dft,
    tpol,
    small,
    [SvpTPolAlloc, SvpPrepareTPol],
    SvpApplyTPolSmallToDft
);

svp_to_dft_test!(
    test_svp_apply_tpol_dft_to_dft,
    svp_apply_tpol_dft_to_dft,
    tpol,
    dft,
    [SvpTPolAlloc, SvpPrepareTPol],
    SvpApplyTPolDftToDft
);

svp_assign_test!(
    test_svp_apply_tpol_dft_to_dft_assign,
    svp_apply_tpol_dft_to_dft_assign,
    tpol,
    [SvpTPolAlloc, SvpPrepareTPol],
    SvpApplyTPolDftToDftAssign
);

svp_to_big_test!(
    test_svp_apply_tpol_small_to_big,
    svp_apply_tpol_small_to_big,
    tpol,
    small,
    [SvpTPolAlloc, SvpPrepareTPol],
    SvpApplyTPolSmallToBig
);

svp_to_big_test!(
    test_svp_apply_tpol_dft_to_big,
    svp_apply_tpol_dft_to_big,
    tpol,
    dft,
    [SvpTPolAlloc, SvpPrepareTPol],
    SvpApplyTPolDftToBig
);

svp_to_small_test!(
    test_svp_apply_tpol_small_to_small,
    svp_apply_tpol_small_to_small,
    tpol,
    small,
    [SvpTPolAlloc, SvpPrepareTPol],
    SvpApplyTPolSmallToSmall
);

svp_to_small_test!(
    test_svp_apply_tpol_dft_to_small,
    svp_apply_tpol_dft_to_small,
    tpol,
    dft,
    [SvpTPolAlloc, SvpPrepareTPol],
    SvpApplyTPolDftToSmall
);
