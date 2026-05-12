use poulpy_hal::{
    api::{
        ModuleN, SvpApplyDftToDftAssign, SvpPPolBytesOf, SvpPPolCopyBackend, VecZnxBigAddAssign, VecZnxBigBytesOf,
        VecZnxBigFromSmallBackend, VecZnxBigNormalize, VecZnxBigNormalizeTmpBytes, VecZnxDftApply, VecZnxDftBytesOf,
        VecZnxIdftApplyTmpA,
    },
    layouts::{Backend, Data, ScratchArena},
};

use crate::{
    ScratchArenaTakeCore,
    decryption::{glwe_decrypt_backend_inner, glwe_decrypt_tmp_bytes_default},
    layouts::{
        GLWEInfos, GLWEPlaintext, GLWESecretPrepared, GLWESecretTensor, GLWESecretTensorPrepared, GLWETensor, GLWEToBackendMut,
        GLWEToBackendRef,
        prepared::{
            GLWESecretPreparedFactory, GLWESecretPreparedToBackendMut, GLWESecretPreparedToBackendRef,
            GLWESecretTensorPreparedToBackendRef, glwe_secret_prepared_backend_ref_from_mut,
        },
    },
};

pub fn glwe_tensor_decrypt_tmp_bytes_default<M, BE: Backend, A>(module: &M, infos: &A) -> usize
where
    M: ModuleN
        + VecZnxDftBytesOf
        + VecZnxBigBytesOf
        + VecZnxBigNormalizeTmpBytes
        + SvpPPolBytesOf
        + GLWESecretPreparedFactory<BE>,
    A: GLWEInfos,
{
    assert_eq!(module.n() as u32, infos.n());

    let rank: usize = infos.rank().into();
    let lvl_0: usize = module.glwe_secret_prepared_bytes_of((GLWESecretTensor::pairs(rank) + rank).into());
    let lvl_1: usize = glwe_decrypt_tmp_bytes_default::<M, BE, _>(module, infos);

    lvl_0 + lvl_1
}

pub fn glwe_tensor_decrypt_default<M, BE: Backend, R: Data, P: Data, S0: Data, S1: Data>(
    module: &M,
    res: &GLWETensor<R>,
    pt: &mut GLWEPlaintext<P>,
    sk: &GLWESecretPrepared<S0, BE>,
    sk_tensor: &GLWESecretTensorPrepared<S1, BE>,
    scratch: &mut ScratchArena<'_, BE>,
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
        + VecZnxBigNormalizeTmpBytes
        + SvpPPolBytesOf
        + SvpPPolCopyBackend<BE>
        + GLWESecretPreparedFactory<BE>,
    GLWETensor<R>: GLWEToBackendRef<BE> + GLWEInfos,
    GLWEPlaintext<P>: GLWEToBackendMut<BE> + GLWEInfos + crate::layouts::SetLWEInfos,
    GLWESecretPrepared<S0, BE>: GLWESecretPreparedToBackendRef<BE> + GLWEInfos,
    GLWESecretTensorPrepared<S1, BE>: GLWESecretTensorPreparedToBackendRef<BE> + GLWEInfos,
    for<'s> ScratchArena<'s, BE>: ScratchArenaTakeCore<'s, BE>,
{
    assert!(
        scratch.available() >= glwe_tensor_decrypt_tmp_bytes_default::<M, BE, _>(module, res),
        "scratch.available(): {} < GLWETensorDecrypt::glwe_tensor_decrypt_tmp_bytes: {}",
        scratch.available(),
        glwe_tensor_decrypt_tmp_bytes_default::<M, BE, _>(module, res)
    );

    let rank: usize = sk.rank().as_usize();

    let (mut sk_grouped, mut scratch_1) = scratch
        .borrow()
        .take_glwe_secret_prepared_scratch(module, (GLWESecretTensor::pairs(rank) + rank).into());

    {
        let binding = &mut sk_grouped;
        let mut grouped_backend = binding.to_backend_mut();
        let sk_backend = sk.to_backend_ref();
        let sk_tensor_backend = sk_tensor.to_backend_ref();

        for i in 0..rank {
            module.svp_ppol_copy_backend(&mut grouped_backend.data, i, &sk_backend.data, i);
        }

        for i in 0..(grouped_backend.rank().as_usize() - rank) {
            module.svp_ppol_copy_backend(&mut grouped_backend.data, i + rank, &sk_tensor_backend.data, i);
        }
    }

    let res_backend = res.to_backend_ref();
    let mut pt_backend = pt.to_backend_mut();
    let sk_grouped_ref = glwe_secret_prepared_backend_ref_from_mut(&sk_grouped);
    glwe_decrypt_backend_inner(module, &res_backend, &mut pt_backend, &sk_grouped_ref, &mut scratch_1);
}
