use poulpy_hal::{
    api::{
        ScratchArenaTakeBasic, VecZnxBigAddSmallAssign, VecZnxBigBytesOf, VecZnxBigColWeightedSum, VecZnxBigNormalize,
        VecZnxBigNormalizeTmpBytes, VecZnxCopyRangeBackend, VecZnxZeroBackend,
    },
    layouts::{Backend, Module, ScratchArena, VecZnxBigToBackendRef, VecZnxToBackendRef},
};

use crate::layouts::{
    GLWEInfos, GLWEToBackendMut, LWEInfos, LWEMatrixInfos, LWEMatrixToBackendRef, LWESecretToBackendRef, Rank, SetBase2k,
};

pub fn lwe_matrix_decrypt_tmp_bytes_default<BE: Backend, A>(module: &Module<BE>, infos: &A) -> usize
where
    Module<BE>: VecZnxBigBytesOf + VecZnxBigNormalizeTmpBytes,
    A: LWEMatrixInfos,
{
    module.bytes_of_vec_znx_big_n(infos.rows(), 1, infos.size())
        + module.bytes_of_vec_znx_n(infos.rows(), 1, infos.size())
        + module.vec_znx_big_normalize_tmp_bytes()
        + 3 * (BE::SCRATCH_ALIGN - 1)
}

pub fn lwe_matrix_decrypt_default<BE, R, P, S>(
    module: &Module<BE>,
    res: &R,
    pt: &mut P,
    sk: &S,
    scratch: &mut ScratchArena<'_, BE>,
) where
    Module<BE>: VecZnxZeroBackend<BE>
        + VecZnxBigColWeightedSum<BE>
        + VecZnxBigAddSmallAssign<BE>
        + VecZnxBigNormalize<BE>
        + VecZnxCopyRangeBackend<BE>
        + VecZnxBigBytesOf
        + VecZnxBigNormalizeTmpBytes,
    R: LWEMatrixToBackendRef<BE> + LWEMatrixInfos,
    P: GLWEToBackendMut<BE> + SetBase2k + GLWEInfos,
    S: LWESecretToBackendRef<BE> + LWEInfos,
    BE: Backend,
{
    let res = res.to_backend_ref();
    let sk = sk.to_backend_ref();

    assert_eq!(res.n(), sk.n(), "lwe_matrix_decrypt: secret dimension mismatch");
    assert!(
        scratch.available() >= lwe_matrix_decrypt_tmp_bytes_default::<BE, _>(module, &res),
        "scratch.available(): {} < LWEMatrixDecrypt::lwe_matrix_decrypt_tmp_bytes: {}",
        scratch.available(),
        lwe_matrix_decrypt_tmp_bytes_default::<BE, _>(module, &res)
    );

    let pt_base2k = pt.base2k().into();
    let res_base2k = res.base2k().into();
    let mut pt = pt.to_backend_mut();
    assert_eq!(pt.rank(), Rank(0), "lwe_matrix_decrypt: plaintext must have rank 0");
    assert!(
        res.rows() <= pt.n().as_usize(),
        "lwe_matrix_decrypt: plaintext ring degree is too small for row count"
    );
    assert_eq!(
        pt.size(),
        res.size(),
        "lwe_matrix_decrypt currently expects matching limb counts"
    );

    module.vec_znx_zero_backend(&mut pt.data, 0);

    let scratch = scratch.borrow();
    let (mut tmp, scratch_1) = scratch.take_vec_znx_big_scratch_n(res.rows(), 1, res.size());
    let (mut rows_pt, mut scratch_2) = scratch_1.take_vec_znx_scratch(res.rows(), 1, pt.size());

    module.vec_znx_big_col_weighted_sum(&mut tmp, 0, &res.mask, &sk.data, 0, res.n().as_usize(), res.rows());
    module.vec_znx_big_add_small_assign(&mut tmp, 0, &res.body, 0);
    module.vec_znx_big_normalize(
        &mut rows_pt,
        pt_base2k,
        0,
        0,
        &tmp.to_backend_ref(),
        res_base2k,
        0,
        &mut scratch_2,
    );

    let rows_pt_ref = rows_pt.to_backend_ref();
    for limb in 0..pt.size() {
        module.vec_znx_copy_range_backend(&mut pt.data, 0, limb, 0, &rows_pt_ref, 0, limb, 0, res.rows());
    }
}
