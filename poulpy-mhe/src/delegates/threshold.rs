use poulpy_core::{
    EncryptionInfos, SmudgingInfos,
    layouts::{
        GLWEInfos, GLWEPublicKeyPreparedToBackendRef, GLWESecretPreparedToBackendRef, GLWESecretToBackendRef, GLWEToBackendMut,
        GLWEToBackendRef,
    },
};
use poulpy_hal::{
    layouts::{Backend, Module, ScratchArena},
    source::Source,
};

use crate::{
    api::{
        GLWEShamirMHEProtocol, GLWEThresholdKeyswitchMHEProtocol, GLWEThresholdPublicKeyswitchMHEProtocol, GLWEWideSecretPrepare,
    },
    layouts::{
        GLWEKeyswitchShareOwned, GLWEPublicKeyswitchShareOwned, GLWEShamirLayout, GLWEShamirPolynomialOwned,
        GLWEShamirShareOwned, GLWEWideSecretOwned, GLWEWideSecretPreparedOwned,
    },
    oep::{
        GLWEShamirMHEProtocolImpl, GLWEThresholdKeyswitchMHEProtocolImpl, GLWEThresholdPublicKeyswitchMHEProtocolImpl,
        GLWEWideSecretPrepareImpl,
    },
};

impl<BE: Backend + GLWEShamirMHEProtocolImpl> GLWEShamirMHEProtocol<BE> for Module<BE> {
    fn mhe_glwe_shamir_polynomial_gen_tmp_bytes(&self, layout: &GLWEShamirLayout) -> usize {
        BE::mhe_glwe_shamir_polynomial_gen_tmp_bytes(self, layout)
    }

    fn mhe_glwe_shamir_polynomial_gen<S>(
        &self,
        res: &mut GLWEShamirPolynomialOwned<BE>,
        sk: &S,
        source_xm: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        S: GLWESecretToBackendRef<BE> + GLWEInfos,
    {
        BE::mhe_glwe_shamir_polynomial_gen(self, res, sk, source_xm, scratch)
    }

    fn mhe_glwe_shamir_share_gen_tmp_bytes(&self, layout: &GLWEShamirLayout) -> usize {
        BE::mhe_glwe_shamir_share_gen_tmp_bytes(self, layout)
    }

    fn mhe_glwe_shamir_share_gen(
        &self,
        res: &mut GLWEShamirShareOwned<BE>,
        poly: &GLWEShamirPolynomialOwned<BE>,
        recipient: u32,
        scratch: &mut ScratchArena<'_, BE>,
    ) {
        BE::mhe_glwe_shamir_share_gen(self, res, poly, recipient, scratch)
    }

    fn mhe_glwe_shamir_share_aggregate(&self, res: &mut GLWEShamirShareOwned<BE>, a: &GLWEShamirShareOwned<BE>) {
        BE::mhe_glwe_shamir_share_aggregate(self, res, a)
    }

    fn mhe_glwe_shamir_share_finalize_tmp_bytes(&self) -> usize {
        BE::mhe_glwe_shamir_share_finalize_tmp_bytes(self)
    }

    fn mhe_glwe_shamir_share_finalize(
        &self,
        res: &mut GLWEWideSecretOwned<BE>,
        share: &GLWEShamirShareOwned<BE>,
        own: u32,
        actives: &[u32],
        scratch: &mut ScratchArena<'_, BE>,
    ) {
        BE::mhe_glwe_shamir_share_finalize(self, res, share, own, actives, scratch)
    }
}

impl<BE: Backend + GLWEWideSecretPrepareImpl> GLWEWideSecretPrepare<BE> for Module<BE> {
    fn mhe_glwe_wide_secret_prepare_tmp_bytes(&self, layout: &GLWEShamirLayout) -> usize {
        BE::mhe_glwe_wide_secret_prepare_tmp_bytes(self, layout)
    }

    fn mhe_glwe_wide_secret_prepare(
        &self,
        res: &mut GLWEWideSecretPreparedOwned<BE>,
        sk: &GLWEWideSecretOwned<BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) {
        BE::mhe_glwe_wide_secret_prepare(self, res, sk, scratch)
    }
}

impl<BE: Backend + GLWEThresholdKeyswitchMHEProtocolImpl> GLWEThresholdKeyswitchMHEProtocol<BE> for Module<BE> {
    fn mhe_glwe_threshold_keyswitch_share_gen_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GLWEInfos,
    {
        BE::mhe_glwe_threshold_keyswitch_share_gen_tmp_bytes(self, infos)
    }

    fn mhe_glwe_threshold_keyswitch_share_gen<C, S, E>(
        &self,
        res: &mut GLWEKeyswitchShareOwned<BE>,
        ct: &C,
        sk_in: &GLWEWideSecretPreparedOwned<BE>,
        sk_out: &S,
        flood: &E,
        failure_bits: usize,
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        C: GLWEToBackendRef<BE> + GLWEInfos,
        S: GLWESecretPreparedToBackendRef<BE> + GLWEInfos,
        E: SmudgingInfos,
    {
        BE::mhe_glwe_threshold_keyswitch_share_gen(self, res, ct, sk_in, sk_out, flood, failure_bits, source_xe, scratch)
    }

    fn mhe_glwe_threshold_keyswitch_share_aggregate(
        &self,
        res: &mut GLWEKeyswitchShareOwned<BE>,
        a: &GLWEKeyswitchShareOwned<BE>,
    ) {
        BE::mhe_glwe_threshold_keyswitch_share_aggregate(self, res, a)
    }

    fn mhe_glwe_threshold_keyswitch_share_finalize_tmp_bytes(&self) -> usize {
        BE::mhe_glwe_threshold_keyswitch_share_finalize_tmp_bytes(self)
    }

    fn mhe_glwe_threshold_keyswitch_share_finalize<R, C>(
        &self,
        res: &mut R,
        ct: &C,
        share: &GLWEKeyswitchShareOwned<BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        C: GLWEToBackendRef<BE> + GLWEInfos,
    {
        BE::mhe_glwe_threshold_keyswitch_share_finalize(self, res, ct, share, scratch)
    }
}

impl<BE: Backend + GLWEThresholdPublicKeyswitchMHEProtocolImpl> GLWEThresholdPublicKeyswitchMHEProtocol<BE> for Module<BE> {
    fn mhe_glwe_threshold_public_keyswitch_share_gen_tmp_bytes<A, B, P>(&self, ct_infos: &A, res_infos: &B, pk_infos: &P) -> usize
    where
        A: GLWEInfos,
        B: GLWEInfos,
        P: GLWEInfos,
    {
        BE::mhe_glwe_threshold_public_keyswitch_share_gen_tmp_bytes(self, ct_infos, res_infos, pk_infos)
    }

    fn mhe_glwe_threshold_public_keyswitch_share_gen<C, K, E1, E2>(
        &self,
        res: &mut GLWEPublicKeyswitchShareOwned<BE>,
        ct: &C,
        sk_in: &GLWEWideSecretPreparedOwned<BE>,
        pk_out: &K,
        flood: &E1,
        enc_infos: &E2,
        failure_bits: usize,
        source_xu: &mut Source,
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        C: GLWEToBackendRef<BE> + GLWEInfos,
        K: GLWEPublicKeyPreparedToBackendRef<BE> + GLWEInfos,
        E1: SmudgingInfos,
        E2: EncryptionInfos,
    {
        BE::mhe_glwe_threshold_public_keyswitch_share_gen(
            self,
            res,
            ct,
            sk_in,
            pk_out,
            flood,
            enc_infos,
            failure_bits,
            source_xu,
            source_xe,
            scratch,
        )
    }

    fn mhe_glwe_threshold_public_keyswitch_share_aggregate(
        &self,
        res: &mut GLWEPublicKeyswitchShareOwned<BE>,
        a: &GLWEPublicKeyswitchShareOwned<BE>,
    ) {
        BE::mhe_glwe_threshold_public_keyswitch_share_aggregate(self, res, a)
    }

    fn mhe_glwe_threshold_public_keyswitch_share_finalize_tmp_bytes(&self) -> usize {
        BE::mhe_glwe_threshold_public_keyswitch_share_finalize_tmp_bytes(self)
    }

    fn mhe_glwe_threshold_public_keyswitch_share_finalize<R, C>(
        &self,
        res: &mut R,
        ct: &C,
        share: &GLWEPublicKeyswitchShareOwned<BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        C: GLWEToBackendRef<BE> + GLWEInfos,
    {
        BE::mhe_glwe_threshold_public_keyswitch_share_finalize(self, res, ct, share, scratch)
    }
}
