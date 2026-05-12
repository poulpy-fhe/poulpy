use poulpy_hal::{
    api::{
        ScratchArenaTakeBasic, VecZnxAddNormalSourceBackend, VecZnxCopyRangeBackend, VecZnxFillUniformSourceBackend,
        VecZnxNormalizeAssignBackend, VecZnxNormalizeTmpBytes, VecZnxSubInnerProductAssignBackend, VecZnxZeroBackend,
    },
    layouts::{Backend, Module, ScratchArena, VecZnx, VecZnxToBackendMut, VecZnxToBackendRef, vec_znx_backend_ref_from_mut},
    source::Source,
};

use crate::{
    EncryptionInfos, ScratchArenaTakeCore,
    layouts::{LWEInfos, LWEPlaintext, LWEPlaintextToBackendRef, LWESecretToBackendRef, LWEToBackendMut},
};

fn lwe_encrypt_sk_sub_mask<BE: Backend>(
    module: &Module<BE>,
    tmp: &mut VecZnx<BE::BufMut<'_>>,
    res_data: &VecZnx<BE::BufMut<'_>>,
    sk_data: &poulpy_hal::layouts::ScalarZnxBackendRef<'_, BE>,
    res_size: usize,
    res_n: usize,
) where
    Module<BE>: VecZnxSubInnerProductAssignBackend<BE>,
{
    let res_ref = vec_znx_backend_ref_from_mut::<BE>(res_data);
    for i in 0..res_size {
        module.vec_znx_sub_inner_product_assign_backend(tmp, 0, i, 0, &res_ref, 0, i, 1, sk_data, 0, 0, res_n);
    }
}

#[doc(hidden)]
pub trait LWEEncryptSkDefault<BE: Backend> {
    fn lwe_encrypt_sk_tmp_bytes_default<A>(&self, infos: &A) -> usize
    where
        A: LWEInfos;

    fn lwe_encrypt_sk_default<'s, R, P, S, E>(
        &self,
        res: &mut R,
        pt: &P,
        sk: &S,
        enc_infos: &E,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        R: LWEToBackendMut<BE>,
        P: LWEPlaintextToBackendRef<BE>,
        S: LWESecretToBackendRef<BE>,
        E: EncryptionInfos,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>;
}

impl<BE: Backend> LWEEncryptSkDefault<BE> for Module<BE>
where
    Self: Sized
        + VecZnxFillUniformSourceBackend<BE>
        + VecZnxAddNormalSourceBackend<BE>
        + VecZnxCopyRangeBackend<BE>
        + VecZnxNormalizeAssignBackend<BE>
        + VecZnxSubInnerProductAssignBackend<BE>
        + VecZnxNormalizeTmpBytes
        + VecZnxZeroBackend<BE>,
{
    fn lwe_encrypt_sk_tmp_bytes_default<A>(&self, infos: &A) -> usize
    where
        A: LWEInfos,
    {
        let size: usize = infos.size();

        let lvl_0: usize = LWEPlaintext::bytes_of(size);
        let lvl_1: usize = self.vec_znx_normalize_tmp_bytes();

        lvl_0 + lvl_1
    }

    #[allow(clippy::too_many_arguments)]
    fn lwe_encrypt_sk_default<'s, R, P, S, E>(
        &self,
        res: &mut R,
        pt: &P,
        sk: &S,
        enc_infos: &E,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        R: LWEToBackendMut<BE>,
        P: LWEPlaintextToBackendRef<BE>,
        S: LWESecretToBackendRef<BE>,
        E: EncryptionInfos,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
    {
        let res = &mut res.to_backend_mut();
        let pt = pt.to_backend_ref();
        let sk = sk.to_backend_ref();

        #[cfg(debug_assertions)]
        {
            assert_eq!(res.n(), sk.n())
        }

        assert!(
            scratch.available() >= self.lwe_encrypt_sk_tmp_bytes_default(res),
            "scratch.available(): {} < LWEEncryptSk::lwe_encrypt_sk_tmp_bytes: {}",
            scratch.available(),
            self.lwe_encrypt_sk_tmp_bytes_default(res)
        );

        let base2k: usize = res.base2k().into();
        let res_size = res.size();
        let res_n = usize::from(res.n());

        self.vec_znx_fill_uniform_source_backend(base2k, &mut res.data, 0, source_xa);

        let scratch = scratch.borrow();
        let (mut tmp_znx, scratch_1) = scratch.take_vec_znx_scratch(1, 1, res_size);
        self.vec_znx_zero_backend(&mut tmp_znx, 0);

        let min_size: usize = res_size.min(pt.size());

        for i in 0..min_size {
            self.vec_znx_copy_range_backend(&mut tmp_znx, 0, i, 0, &pt.data, 0, i, 0, 1);
        }

        lwe_encrypt_sk_sub_mask(self, &mut tmp_znx, &res.data, &sk.data, res_size, res_n);

        {
            let mut tmp_znx_mut = tmp_znx.to_backend_mut();
            self.vec_znx_add_normal_source_backend(base2k, &mut tmp_znx_mut, 0, enc_infos.noise_infos(), source_xe);
        }

        let _ = scratch_1.apply_mut(|scratch| self.vec_znx_normalize_assign_backend(base2k, &mut tmp_znx, 0, scratch));

        let tmp_znx_ref = tmp_znx.to_backend_ref();
        for i in 0..res_size {
            self.vec_znx_copy_range_backend(&mut res.data, 0, i, 0, &tmp_znx_ref, 0, i, 0, 1);
        }
    }
}
