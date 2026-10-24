use poulpy_hal::{
    api::VecZnxSwitchRing,
    layouts::{Backend, Data, Module, ScratchArena},
    source::Source,
};

use crate::{
    GetDistribution, GetDistributionMut,
    api::{GLWECIEmbed, GLWECIKeyEncryptSk, GLWECITrace, GLWESecretCIEmbed, GLWESwitchDegree},
    layouts::{
        GGLWEInfos, GGLWEToBackendMut, GLWECIEmbedKey, GLWECITraceKey, GLWEInfos, GLWESecretToBackendMut, GLWESecretToBackendRef,
        GLWEToBackendMut, GLWEToBackendRef,
    },
    oep::GLWECIConversionImpl,
};

impl<BE: Backend + GLWECIConversionImpl> GLWECIEmbed<BE> for Module<BE> {
    fn glwe_ci_embed<R, A>(&self, res: &mut R, a: &A)
    where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        A: GLWEToBackendRef<BE> + GLWEInfos,
    {
        BE::glwe_ci_embed(self, res, a)
    }
}

impl<BE: Backend + GLWECIConversionImpl> GLWECITrace<BE> for Module<BE> {
    fn glwe_ci_trace_tmp_bytes<R, A>(&self, res_infos: &R, a_infos: &A) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
    {
        BE::glwe_ci_trace_tmp_bytes(self, res_infos, a_infos)
    }

    fn glwe_ci_trace<R, A>(&self, res: &mut R, a: &A, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        A: GLWEToBackendRef<BE> + GLWEInfos,
    {
        BE::glwe_ci_trace(self, res, a, scratch)
    }
}

impl<BE: Backend + GLWECIConversionImpl> GLWESwitchDegree<BE> for Module<BE>
where
    Module<BE>: VecZnxSwitchRing<BE>,
{
    fn glwe_switch_degree<R, A>(&self, res: &mut R, a: &A)
    where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        A: GLWEToBackendRef<BE> + GLWEInfos,
    {
        BE::glwe_switch_degree(self, res, a)
    }
}

impl<BE: Backend + GLWECIConversionImpl> GLWESecretCIEmbed<BE> for Module<BE> {
    fn glwe_secret_ci_embed<R, S>(&self, res: &mut R, src: &S)
    where
        R: GLWESecretToBackendMut<BE> + GetDistributionMut,
        S: GLWESecretToBackendRef<BE>,
    {
        BE::glwe_secret_ci_embed(self, res, src)
    }
}

impl<BE: Backend + GLWECIConversionImpl> GLWECIKeyEncryptSk<BE> for Module<BE> {
    fn glwe_ci_key_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        BE::glwe_ci_key_encrypt_sk_tmp_bytes(self, infos)
    }

    fn glwe_ci_embed_key_encrypt_sk<D, S1, S2>(
        &self,
        res: &mut GLWECIEmbedKey<D, BE::ZnxWord>,
        sk_ci: &S1,
        sk: &S2,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        D: Data,
        GLWECIEmbedKey<D, BE::ZnxWord>: GGLWEToBackendMut<BE>,
        S1: GLWESecretToBackendRef<BE>,
        S2: GLWESecretToBackendRef<BE> + GetDistribution + GLWEInfos,
    {
        BE::glwe_ci_embed_key_encrypt_sk(self, res, sk_ci, sk, source_xe, source_xa, scratch)
    }

    fn glwe_ci_trace_key_encrypt_sk<D, S1, S2>(
        &self,
        res: &mut GLWECITraceKey<D, BE::ZnxWord>,
        sk_ci: &S1,
        sk: &S2,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        D: Data,
        GLWECITraceKey<D, BE::ZnxWord>: GGLWEToBackendMut<BE>,
        S1: GLWESecretToBackendRef<BE>,
        S2: GLWESecretToBackendRef<BE> + GetDistribution + GLWEInfos,
    {
        BE::glwe_ci_trace_key_encrypt_sk(self, res, sk_ci, sk, source_xe, source_xa, scratch)
    }
}
