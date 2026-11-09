use poulpy_hal::{
    api::{
        ModuleN, SvpApplyDftToDftAssign, SvpPPolBytesOf, SvpPPolCopy, VecZnxBigAddAssign, VecZnxBigBytesOf, VecZnxBigFromSmall,
        VecZnxBigNormalize, VecZnxBigNormalizeTmpBytes, VecZnxDftApply, VecZnxDftBytesOf, VecZnxIdftApply,
        VecZnxIdftApplyTmpBytes,
    },
    layouts::{Backend, Data, Ring, ScratchArena},
};

use crate::layouts::operand_degree;
use crate::{
    ScratchArenaTakeGLWESecretPrepared,
    api::GLWESecretPreparedFactory,
    decryption::{glwe::glwe_decrypt_body_tmp_bytes, glwe_decrypt_backend_inner},
    layouts::{
        GLWEInfos, GLWEPlaintext, GLWESecretPrepared, GLWESecretTensorPrepared, GLWETensor, GLWEToBackendMut, GLWEToBackendRef,
        prepared::{GLWESecretPreparedToBackendMut, GLWESecretPreparedToBackendRef, GLWESecretTensorPreparedToBackendRef},
    },
};

pub fn glwe_tensor_decrypt_tmp_bytes_reference<M, BE: Backend, A, Rg: Ring>(module: &M, infos: &A) -> usize
where
    M: ModuleN
        + VecZnxDftBytesOf
        + VecZnxBigBytesOf
        + VecZnxBigNormalizeTmpBytes
        + VecZnxIdftApplyTmpBytes
        + SvpPPolBytesOf
        + GLWESecretPreparedFactory<BE, Rg>,
    A: GLWEInfos,
{
    operand_degree(module.n(), &[infos.n()]);

    let rank: usize = infos.rank().into();
    let lvl_0: usize = module.glwe_secret_prepared_bytes_of((crate::layouts::pairs(rank) + rank).into());
    let lvl_1: usize = glwe_decrypt_body_tmp_bytes::<M, _>(module, infos);

    lvl_0 + lvl_1
}

pub fn glwe_tensor_decrypt_reference<M, BE: Backend, R: Data, P: Data, S0: Data, S1: Data, Rg: Ring>(
    module: &M,
    res: &GLWETensor<R, BE::ZnxWord, Rg>,
    pt: &mut GLWEPlaintext<P, BE::ZnxWord, Rg>,
    sk: &GLWESecretPrepared<S0, BE, Rg>,
    sk_tensor: &GLWESecretTensorPrepared<S1, BE, Rg>,
    scratch: &mut ScratchArena<'_, BE>,
) where
    M: ModuleN
        + VecZnxDftBytesOf
        + VecZnxBigBytesOf
        + VecZnxBigFromSmall<BE, Rg>
        + VecZnxDftApply<BE, Rg>
        + SvpApplyDftToDftAssign<BE, Rg>
        + VecZnxIdftApply<BE, Rg>
        + VecZnxIdftApplyTmpBytes
        + VecZnxBigAddAssign<BE, Rg>
        + VecZnxBigNormalize<BE, Rg>
        + VecZnxBigNormalizeTmpBytes
        + SvpPPolBytesOf
        + SvpPPolCopy<BE, Rg>
        + GLWESecretPreparedFactory<BE, Rg>,
    GLWETensor<R, BE::ZnxWord, Rg>: GLWEToBackendRef<BE, Rg> + GLWEInfos,
    GLWEPlaintext<P, BE::ZnxWord, Rg>: GLWEToBackendMut<BE, Rg> + GLWEInfos + crate::layouts::SetBase2k,
    GLWESecretPrepared<S0, BE, Rg>: GLWESecretPreparedToBackendRef<BE, Rg> + GLWEInfos,
    GLWESecretTensorPrepared<S1, BE, Rg>: GLWESecretTensorPreparedToBackendRef<BE, Rg> + GLWEInfos,
{
    assert!(
        scratch.available() >= glwe_tensor_decrypt_tmp_bytes_reference::<M, BE, _, Rg>(module, res),
        "scratch.available(): {} < GLWETensorDecrypt::glwe_tensor_decrypt_tmp_bytes: {}",
        scratch.available(),
        glwe_tensor_decrypt_tmp_bytes_reference::<M, BE, _, Rg>(module, res)
    );
    let tmp_bytes: usize = glwe_tensor_decrypt_tmp_bytes_reference::<M, BE, _, Rg>(module, res);
    {
        let rank: usize = sk.rank().as_usize();

        let (mut sk_grouped, mut scratch_1) = scratch
            .borrow()
            .take_glwe_secret_prepared_scratch(sk.n(), (crate::layouts::pairs(rank) + rank).into());

        {
            let binding = &mut sk_grouped;
            let mut grouped_backend = binding.to_backend_mut();
            let sk_backend = sk.to_backend_ref();
            let sk_tensor_backend = sk_tensor.to_backend_ref();

            for i in 0..rank {
                module.svp_ppol_copy(&mut grouped_backend.data, i, &sk_backend.data, i);
            }

            for i in 0..(grouped_backend.rank().as_usize() - rank) {
                module.svp_ppol_copy(&mut grouped_backend.data, i + rank, &sk_tensor_backend.data, i);
            }
        }

        let res_backend = res.to_backend_ref();
        let mut pt_backend = pt.to_backend_mut();
        let sk_grouped_ref = sk_grouped.to_backend_ref();
        glwe_decrypt_backend_inner(module, &res_backend, &mut pt_backend, &sk_grouped_ref, &mut scratch_1);
    }
    scratch.wipe(tmp_bytes);
}
