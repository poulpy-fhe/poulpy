use poulpy_hal::{
    api::{VecZnxNormalize, VecZnxNormalizeTmpBytes},
    layouts::{Backend, HostBackend, HostDataMut, HostDataRef, ScratchArena, ZnxView, ZnxViewMut},
};

use crate::{
    ScratchArenaTakeCore,
    layouts::{
        LWEInfos, LWEPlaintext, LWEPlaintextToBackendMut, LWEPlaintextToBackendRef, LWESecretToBackendRef, LWEToBackendRef,
        SetLWEInfos,
    },
};

pub fn lwe_decrypt_tmp_bytes_default<M, BE: Backend, A>(module: &M, infos: &A) -> usize
where
    M: VecZnxNormalizeTmpBytes,
    A: LWEInfos,
{
    let lvl_0: usize = LWEPlaintext::bytes_of(infos.size());
    let lvl_1: usize = module.vec_znx_normalize_tmp_bytes();

    lvl_0 + lvl_1
}

pub fn lwe_decrypt_default<M, BE, R, P, S>(module: &M, res: &R, pt: &mut P, sk: &S, scratch: &mut ScratchArena<'_, BE>)
where
    M: VecZnxNormalize<BE> + VecZnxNormalizeTmpBytes,
    R: LWEToBackendRef<BE> + LWEInfos,
    P: LWEPlaintextToBackendMut<BE> + SetLWEInfos + LWEInfos,
    S: LWESecretToBackendRef<BE> + LWEInfos,
    BE: Backend + HostBackend,
    for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
    for<'a> BE::BufMut<'a>: HostDataMut,
    for<'a> BE::BufRef<'a>: HostDataRef,
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

    let (mut tmp, mut scratch_1) = scratch.take_lwe_plaintext_scratch(&res);
    for i in 0..res.size() {
        tmp.data.at_mut(0, i)[0] = res.data.at(0, i)[0]
            + res.data.at(0, i)[1..]
                .iter()
                .zip(sk.data.at(0, 0))
                .map(|(x, y)| x * y)
                .sum::<i64>();
    }

    let pt_base2k = pt.base2k().into();
    let res_base2k = res.base2k().into();
    let mut pt = pt.to_backend_mut();
    let tmp_ref = tmp.to_backend_ref();
    module.vec_znx_normalize(&mut pt.data, pt_base2k, 0, 0, &tmp_ref.data, res_base2k, 0, &mut scratch_1);
}
