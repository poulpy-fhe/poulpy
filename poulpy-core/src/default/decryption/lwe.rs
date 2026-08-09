use poulpy_hal::{
    api::{
        ScratchArenaTakeBasic, VecZnxBigAddSmallAssign, VecZnxBigBytesOf, VecZnxBigInnerSumBackend, VecZnxBigNormalize,
        VecZnxBigNormalizeTmpBytes, VecZnxScalarProduct,
    },
    layouts::{Backend, ScratchArena, VecZnxBigToBackendRef},
};

use crate::layouts::{LWEInfos, LWEPlaintextToBackendMut, LWESecretToBackendRef, LWEToBackendRef, SetBase2k};

#[doc(hidden)]
pub fn lwe_decrypt_tmp_bytes_default<M, BE: Backend, A>(module: &M, infos: &A) -> usize
where
    M: VecZnxBigBytesOf + VecZnxBigNormalizeTmpBytes,
    A: LWEInfos,
{
    module.bytes_of_vec_znx_big_n(infos.n().as_usize(), 1, infos.size())
        + module.bytes_of_vec_znx_big_n(1, 1, infos.size())
        + module.vec_znx_big_normalize_tmp_bytes()
        + 2 * (BE::SCRATCH_ALIGN - 1)
}

#[doc(hidden)]
pub fn lwe_decrypt_default<M, BE, R, P, S>(module: &M, res: &R, pt: &mut P, sk: &S, scratch: &mut ScratchArena<'_, BE>)
where
    M: VecZnxScalarProduct<BE>
        + VecZnxBigInnerSumBackend<BE>
        + VecZnxBigAddSmallAssign<BE>
        + VecZnxBigNormalize<BE>
        + VecZnxBigBytesOf
        + VecZnxBigNormalizeTmpBytes,
    R: LWEToBackendRef<BE> + LWEInfos,
    P: LWEPlaintextToBackendMut<BE> + SetBase2k + LWEInfos,
    S: LWESecretToBackendRef<BE> + LWEInfos,
    BE: Backend,
{
    let res = res.to_backend_ref();
    let sk = sk.to_backend_ref();

    #[cfg(debug_assertions)]
    {
        assert_eq!(res.n(), sk.n());
    }
    assert!(
        scratch.available() >= lwe_decrypt_tmp_bytes_default::<M, BE, _>(module, &res),
        "scratch.available(): {} < LWEDecrypt::lwe_decrypt_tmp_bytes: {}",
        scratch.available(),
        lwe_decrypt_tmp_bytes_default::<M, BE, _>(module, &res)
    );

    let scratch = scratch.borrow();
    let (mut tmp_hadamard, scratch_1) = scratch.take_vec_znx_big_scratch_n(res.n().as_usize(), 1, res.size());
    module.vec_znx_scalar_product(&mut tmp_hadamard, 0, &res.mask, 0, &sk.data, 0);

    let (mut tmp_scalar, mut scratch_2) = scratch_1.take_vec_znx_big_scratch_n(1, 1, res.size());
    module.vec_znx_big_inner_sum_backend(&mut tmp_scalar, 0, 0, &tmp_hadamard.to_backend_ref(), 0);
    module.vec_znx_big_add_small_assign(&mut tmp_scalar, 0, &res.body, 0);

    let pt_base2k = pt.base2k().into();
    let res_base2k = res.base2k().into();
    let mut pt = pt.to_backend_mut();
    module.vec_znx_big_normalize(
        &mut pt.data,
        pt_base2k,
        0,
        0,
        &tmp_scalar.to_backend_ref(),
        res_base2k,
        0,
        &mut scratch_2,
    );
}
