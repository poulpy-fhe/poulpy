use poulpy_hal::{
    api::{
        ModuleN, ScratchArenaTakeBasic, SvpApplyDftToDftAssign, VecZnxBigAddAssign, VecZnxBigBytesOf, VecZnxBigFromSmall,
        VecZnxBigNormalize, VecZnxBigNormalizeTmpBytes, VecZnxDftApply, VecZnxDftBytesOf, VecZnxIdftApplyTmpA, VecZnxNormalize,
        VecZnxNormalizeTmpBytes,
    },
    layouts::{
        Backend, ScratchArena, VecZnxBackendRef, VecZnxBigToBackendMut, VecZnxBigToBackendRef, VecZnxDftToBackendMut,
        VecZnxToBackendMut, VecZnxToBackendRef,
    },
};

pub use crate::api::GLWEDecrypt;
use crate::layouts::operand_degree;
use crate::{
    ScratchArenaTakeCore,
    api::{GLWEBytesOf, GLWENormalize},
    layouts::{
        Base2K, GLWEBackendMut, GLWEBackendRef, GLWEInfos, GLWEMaskToBackendRef, GLWEToBackendMut, GLWEToBackendRef, LWEInfos,
        SetBase2k,
        prepared::{GLWESecretPreparedBackendRef, GLWESecretPreparedToBackendRef},
    },
};

pub fn glwe_decrypt_tmp_bytes_reference<M, BE: Backend, A>(module: &M, infos: &A) -> usize
where
    M: ModuleN + VecZnxDftBytesOf + VecZnxBigBytesOf + VecZnxBigNormalizeTmpBytes + GLWENormalize<BE> + GLWEBytesOf<BE>,
    A: GLWEInfos,
{
    BE::scratch_aligned(module.glwe_bytes_of_from_infos(infos))
        + glwe_decrypt_body_tmp_bytes::<M, _>(module, infos).max(module.glwe_normalize_tmp_bytes())
}

pub(crate) fn glwe_decrypt_body_tmp_bytes<M, A>(module: &M, infos: &A) -> usize
where
    M: ModuleN + VecZnxDftBytesOf + VecZnxBigBytesOf + VecZnxBigNormalizeTmpBytes,
    A: GLWEInfos,
{
    let size: usize = infos.size();
    let n: usize = operand_degree(module.n(), &[infos.n()]);

    let lvl_0: usize = module.bytes_of_vec_znx_big(n, 1, size);
    let lvl_1: usize = (module.bytes_of_vec_znx_dft(n, 1, size) + module.bytes_of_vec_znx_big(n, 1, size))
        .max(module.vec_znx_big_normalize_tmp_bytes());

    lvl_0 + lvl_1
}

pub fn glwe_decrypt_reference<M, BE: Backend, R, P, S>(
    module: &M,
    res: &R,
    pt: &mut P,
    sk: &S,
    scratch: &mut ScratchArena<'_, BE>,
) where
    M: ModuleN
        + VecZnxDftBytesOf
        + VecZnxBigBytesOf
        + VecZnxBigFromSmall<BE>
        + VecZnxDftApply<BE>
        + SvpApplyDftToDftAssign<BE>
        + VecZnxIdftApplyTmpA<BE>
        + VecZnxBigAddAssign<BE>
        + VecZnxBigNormalize<BE>
        + VecZnxBigNormalizeTmpBytes
        + GLWENormalize<BE>
        + GLWEBytesOf<BE>,
    R: GLWEToBackendRef<BE> + GLWEInfos,
    P: GLWEToBackendMut<BE> + GLWEInfos + SetBase2k,
    S: GLWESecretPreparedToBackendRef<BE> + GLWEInfos,
{
    operand_degree(module.n(), &[res.n(), pt.n(), sk.n()]);
    assert!(
        scratch.available() >= glwe_decrypt_tmp_bytes_reference::<M, BE, _>(module, res),
        "scratch.available(): {} < GLWEDecrypt::glwe_decrypt_tmp_bytes: {}",
        scratch.available(),
        glwe_decrypt_tmp_bytes_reference::<M, BE, _>(module, res)
    );
    let tmp_bytes: usize = glwe_decrypt_tmp_bytes_reference::<M, BE, _>(module, res);
    {
        let (mut res_tmp, mut scratch) = scratch.borrow().take_glwe_scratch(res);
        let res = if res.is_canonical() {
            res.to_backend_ref()
        } else {
            module.glwe_normalize(&mut res_tmp, res, &mut scratch.borrow());
            res_tmp.to_backend_ref()
        };
        let mut pt_backend = pt.to_backend_mut();
        let sk_backend = sk.to_backend_ref();

        glwe_decrypt_backend_inner(module, &res, &mut pt_backend, &sk_backend, &mut scratch);
    }
    scratch.wipe(tmp_bytes);
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
        + VecZnxBigFromSmall<BE>
        + VecZnxDftApply<BE>
        + SvpApplyDftToDftAssign<BE>
        + VecZnxIdftApplyTmpA<BE>
        + VecZnxBigAddAssign<BE>
        + VecZnxBigNormalize<BE>
        + VecZnxBigNormalizeTmpBytes,
{
    debug_assert_eq!(res.rank(), sk.rank());
    operand_degree(module.n(), &[res.n(), pt.n(), sk.n()]);
    assert!(
        scratch.available() >= glwe_decrypt_body_tmp_bytes::<M, _>(module, res),
        "scratch.available(): {} < GLWEDecrypt::glwe_decrypt_tmp_bytes: {}",
        scratch.available(),
        glwe_decrypt_body_tmp_bytes::<M, _>(module, res)
    );

    phase_backend_inner(
        module,
        &res.data,
        Some(0),
        1,
        res.rank().into(),
        res.base2k(),
        res.size(),
        pt,
        sk,
        scratch,
    );
}

/// Normalizes into `pt` the column `body` of `data`, if any, plus the products of
/// its `rank` columns from `mask` with the secret's.
#[allow(clippy::too_many_arguments)]
fn phase_backend_inner<'arena, 'scratch, M, BE: Backend>(
    module: &M,
    data: &VecZnxBackendRef<'_, BE>,
    body: Option<usize>,
    mask: usize,
    rank: usize,
    base2k: Base2K,
    size: usize,
    pt: &mut GLWEBackendMut<'_, BE>,
    sk: &GLWESecretPreparedBackendRef<'_, BE>,
    scratch: &'scratch mut ScratchArena<'arena, BE>,
) where
    M: ModuleN
        + VecZnxDftBytesOf
        + VecZnxBigBytesOf
        + VecZnxBigFromSmall<BE>
        + VecZnxDftApply<BE>
        + SvpApplyDftToDftAssign<BE>
        + VecZnxIdftApplyTmpA<BE>
        + VecZnxBigAddAssign<BE>
        + VecZnxBigNormalize<BE>
        + VecZnxBigNormalizeTmpBytes,
{
    let n: usize = data.n();
    let (mut c0_big, mut scratch_1) = scratch.borrow().take_vec_znx_big_scratch(n, 1, size);
    if let Some(body) = body {
        module.vec_znx_big_from_small(&mut c0_big, 0, data, body);
    }

    for i in 0..rank {
        let (mut ci_dft, scratch_2) = scratch_1.borrow().take_vec_znx_dft_scratch(n, 1, size);
        module.vec_znx_dft_apply(1, 0, &mut ci_dft, 0, data, mask + i);
        {
            let mut ci_dft_backend = ci_dft.to_backend_mut();
            module.svp_apply_dft_to_dft_assign(&mut ci_dft_backend, 0, &sk.data, i);
        }
        // Without a body, the first product starts the sum.
        if body.is_none() && i == 0 {
            let mut c0_big_backend = c0_big.to_backend_mut();
            let mut ci_dft_backend = ci_dft.to_backend_mut();
            module.vec_znx_idft_apply_tmpa(&mut c0_big_backend, 0, &mut ci_dft_backend, 0);
            continue;
        }
        let (mut ci_big, _) = scratch_2.take_vec_znx_big_scratch(n, 1, size);
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
    let pt_k = pt.k().as_usize();
    let _ = scratch_1.apply_mut(|scratch| {
        module.vec_znx_big_normalize(
            &mut pt.data,
            pt_base2k.into(),
            pt_k,
            0,
            0,
            &c0_big_ref,
            base2k.into(),
            0,
            scratch,
        )
    });
}

pub fn glwe_mask_decrypt_tmp_bytes_reference<M, BE: Backend, A>(module: &M, infos: &A) -> usize
where
    M: ModuleN + VecZnxDftBytesOf + VecZnxBigBytesOf + VecZnxBigNormalizeTmpBytes + VecZnxNormalizeTmpBytes,
    A: GLWEInfos,
{
    BE::scratch_aligned(BE::bytes_of_vec_znx(infos.n().into(), infos.rank().into(), infos.size()))
        + glwe_decrypt_body_tmp_bytes::<M, _>(module, infos).max(module.vec_znx_normalize_tmp_bytes())
}

/// Writes into `pt` the inner product of `mask` with `sk`: the decryption of a
/// ciphertext of that mask with a zero body. A flag-clear mask is normalized first.
pub fn glwe_mask_decrypt_reference<M, BE: Backend, A, P, S>(
    module: &M,
    mask: &A,
    pt: &mut P,
    sk: &S,
    scratch: &mut ScratchArena<'_, BE>,
) where
    M: ModuleN
        + VecZnxDftBytesOf
        + VecZnxBigBytesOf
        + VecZnxBigFromSmall<BE>
        + VecZnxDftApply<BE>
        + SvpApplyDftToDftAssign<BE>
        + VecZnxIdftApplyTmpA<BE>
        + VecZnxBigAddAssign<BE>
        + VecZnxBigNormalize<BE>
        + VecZnxBigNormalizeTmpBytes
        + VecZnxNormalize<BE>
        + VecZnxNormalizeTmpBytes,
    A: GLWEMaskToBackendRef<BE> + GLWEInfos,
    P: GLWEToBackendMut<BE> + GLWEInfos + SetBase2k,
    S: GLWESecretPreparedToBackendRef<BE> + GLWEInfos,
{
    let n: usize = operand_degree(module.n(), &[mask.n(), pt.n(), sk.n()]);
    assert!(mask.rank().as_usize() > 0, "GLWE mask rank must be positive");
    assert_eq!(mask.rank(), sk.rank(), "GLWE mask and secret key ranks must match");
    let tmp_bytes: usize = glwe_mask_decrypt_tmp_bytes_reference::<M, BE, _>(module, mask);
    assert!(
        scratch.available() >= tmp_bytes,
        "scratch.available(): {} < GLWEMaskDecrypt::glwe_mask_decrypt_tmp_bytes: {}",
        scratch.available(),
        tmp_bytes
    );
    {
        let mask = mask.to_mask_backend_ref();
        let (rank, base2k, size) = (mask.rank().as_usize(), mask.base2k(), mask.size());
        let (mut tmp, mut scratch) = scratch.borrow().take_vec_znx_scratch(n, rank, size);
        let (mut pt, sk) = (pt.to_backend_mut(), sk.to_backend_ref());
        if mask.is_canonical() {
            phase_backend_inner(
                module,
                &mask.data,
                None,
                mask.col(0),
                rank,
                base2k,
                size,
                &mut pt,
                &sk,
                &mut scratch,
            );
        } else {
            for j in 0..rank {
                module.vec_znx_normalize(
                    &mut tmp.to_backend_mut(),
                    base2k.into(),
                    mask.k().as_usize(),
                    0,
                    j,
                    &mask.data,
                    base2k.into(),
                    mask.col(j),
                    &mut scratch.borrow(),
                );
            }
            phase_backend_inner(
                module,
                &tmp.to_backend_ref(),
                None,
                0,
                rank,
                base2k,
                size,
                &mut pt,
                &sk,
                &mut scratch,
            );
        }
    }
    scratch.wipe(tmp_bytes);
}
