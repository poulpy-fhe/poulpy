use poulpy_hal::{
    api::{
        ModuleN, ScratchArenaTakeBasic, SvpApplyDftToDftAssign, VecZnxBigAddAssign, VecZnxBigBytesOf, VecZnxBigFromSmallBackend,
        VecZnxBigNormalize, VecZnxBigNormalizeTmpBytes, VecZnxDftApply, VecZnxDftBytesOf, VecZnxIdftApplyTmpA,
    },
    layouts::{Backend, ScratchArena, VecZnxBigToBackendMut, VecZnxBigToBackendRef, VecZnxDftToBackendMut},
};

pub use crate::api::GLWEDecrypt;
use crate::{
    ScratchArenaTakeCore,
    layouts::{
        GLWEBackendMut, GLWEBackendRef, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, LWEInfos, SetLWEInfos,
        prepared::{GLWESecretPreparedBackendRef, GLWESecretPreparedToBackendRef},
    },
};

pub fn glwe_decrypt_tmp_bytes_default<M, BE: Backend, A>(module: &M, infos: &A) -> usize
where
    M: ModuleN + VecZnxDftBytesOf + VecZnxBigBytesOf + VecZnxBigNormalizeTmpBytes,
    A: GLWEInfos,
{
    let size: usize = infos.size();
    assert_eq!(module.n() as u32, infos.n());

    let lvl_0: usize = module.bytes_of_vec_znx_big(1, size);
    let lvl_1: usize = (module.bytes_of_vec_znx_dft(1, size) + module.bytes_of_vec_znx_big(1, size))
        .max(module.vec_znx_big_normalize_tmp_bytes());

    lvl_0 + lvl_1
}

pub fn glwe_decrypt_default<M, BE: Backend, R, P, S>(module: &M, res: &R, pt: &mut P, sk: &S, scratch: &mut ScratchArena<'_, BE>)
where
    M: ModuleN
        + VecZnxDftBytesOf
        + VecZnxBigBytesOf
        + VecZnxBigFromSmallBackend<BE>
        + VecZnxDftApply<BE>
        + SvpApplyDftToDftAssign<BE>
        + VecZnxIdftApplyTmpA<BE>
        + VecZnxBigAddAssign<BE>
        + VecZnxBigNormalize<BE>
        + VecZnxBigNormalizeTmpBytes,
    R: GLWEToBackendRef<BE> + GLWEInfos,
    P: GLWEToBackendMut<BE> + GLWEInfos + SetLWEInfos,
    S: GLWESecretPreparedToBackendRef<BE> + GLWEInfos,
    for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
{
    let res_backend = res.to_backend_ref();
    let mut pt_backend = pt.to_backend_mut();
    let sk_backend = sk.to_backend_ref();

    glwe_decrypt_backend_inner(module, &res_backend, &mut pt_backend, &sk_backend, scratch);
}

pub(crate) fn glwe_decrypt_backend_inner<'arena, 'scratch, M, BE: Backend>(
    module: &M,
    res: &GLWEBackendRef<'_, BE>,
    pt: &mut GLWEBackendMut<'_, BE>,
    sk: &GLWESecretPreparedBackendRef<'_, BE>,
    scratch: &'scratch mut ScratchArena<'arena, BE>,
) where
    M: ModuleN
        + VecZnxDftBytesOf
        + VecZnxBigBytesOf
        + VecZnxBigFromSmallBackend<BE>
        + VecZnxDftApply<BE>
        + SvpApplyDftToDftAssign<BE>
        + VecZnxIdftApplyTmpA<BE>
        + VecZnxBigAddAssign<BE>
        + VecZnxBigNormalize<BE>
        + VecZnxBigNormalizeTmpBytes,
    for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
{
    #[cfg(debug_assertions)]
    {
        assert_eq!(res.rank(), sk.rank());
        assert_eq!(res.n(), sk.n());
        assert_eq!(pt.n(), sk.n());
    }
    assert!(
        scratch.available() >= glwe_decrypt_tmp_bytes_default::<M, BE, _>(module, res),
        "scratch.available(): {} < GLWEDecrypt::glwe_decrypt_tmp_bytes: {}",
        scratch.available(),
        glwe_decrypt_tmp_bytes_default::<M, BE, _>(module, res)
    );

    let cols: usize = (res.rank() + 1).into();
    let (mut c0_big, mut scratch_1) = scratch.borrow().take_vec_znx_big_scratch(module, 1, res.size());
    module.vec_znx_big_from_small_backend(&mut c0_big, 0, &res.data, 0);

    for i in 1..cols {
        let (mut ci_dft, scratch_2) = scratch_1.borrow().take_vec_znx_dft_scratch(module, 1, res.size());
        module.vec_znx_dft_apply(1, 0, &mut ci_dft, 0, &res.data, i);
        {
            let mut ci_dft_backend = ci_dft.to_backend_mut();
            module.svp_apply_dft_to_dft_assign(&mut ci_dft_backend, 0, &sk.data, i - 1);
        }
        let (mut ci_big, _) = scratch_2.take_vec_znx_big_scratch(module, 1, res.size());
        {
            let mut ci_big_backend = ci_big.to_backend_mut();
            let mut ci_dft_backend = ci_dft.to_backend_mut();
            module.vec_znx_idft_apply_tmpa(&mut ci_big_backend, 0, &mut ci_dft_backend, 0);
        }
        let ci_big_ref = ci_big.to_backend_ref();
        module.vec_znx_big_add_assign(&mut c0_big, 0, &ci_big_ref, 0);
    }

    let c0_big_ref = c0_big.to_backend_ref();
    let pt_base2k = pt.base2k();
    let _ = scratch_1.apply_mut(|scratch| {
        module.vec_znx_big_normalize(
            &mut pt.data,
            pt_base2k.into(),
            0,
            0,
            &c0_big_ref,
            res.base2k().into(),
            0,
            scratch,
        )
    });
}
