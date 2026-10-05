//! Maps between conjugate-invariant GLWEs of degree `N` and standard GLWEs of
//! degree `2N`, expressed with HAL operations of the standard module, and the
//! derived encryption of the switching keys between their secrets.
use poulpy_hal::{
    api::{ModuleN, ScratchArenaTakeBasic, VecZnxCIEmbed, VecZnxCITrace, VecZnxNormalize, VecZnxNormalizeTmpBytes},
    layouts::{Backend, Data, Module, ScratchArena, VecZnxToBackendMut, VecZnxToBackendRef},
    source::Source,
};

use crate::{
    GLWECIKeyEncryptSk, GLWESwitchingKeyEncryptSk, GetDistribution,
    layouts::{
        GGLWEInfos, GGLWEToBackendMut, GLWECIEmbedKey, GLWECITraceKey, GLWEInfos, GLWESecretCIEmbed, GLWESecretToBackendRef,
        GLWEToBackendMut, GLWEToBackendRef, LWEInfos,
    },
};

/// Embeds each column of `a` into `res` at the radix of `a`; the digits are
/// exact but not renormalized.
pub fn glwe_ci_embed_reference<BE, M, R, A>(module: &M, res: &mut R, a: &A)
where
    BE: Backend,
    M: ModuleN + VecZnxCIEmbed<BE>,
    R: GLWEToBackendMut<BE> + GLWEInfos,
    A: GLWEToBackendRef<BE> + GLWEInfos,
{
    res.set_encryption_metadata(a.encryption_metadata());
    res.set_canonical(false);
    let mut res = res.to_backend_mut();
    let a = a.to_backend_ref();
    assert_eq!(res.n(), 2 * a.n());
    assert!(res.n().as_usize() <= module.n());
    assert_eq!(res.rank(), a.rank());
    assert_eq!(res.base2k(), a.base2k());
    for col in 0..res.rank().as_usize() + 1 {
        module.vec_znx_ci_embed(&mut res.data, col, &a.data, col);
    }
}

pub fn glwe_ci_trace_tmp_bytes_reference<BE, M, R, A>(module: &M, res_infos: &R, a_infos: &A) -> usize
where
    BE: Backend,
    M: ModuleN + VecZnxNormalizeTmpBytes,
    R: GLWEInfos,
    A: GLWEInfos,
{
    BE::bytes_of_vec_znx(res_infos.n().as_usize(), 1, a_infos.size()) + module.vec_znx_normalize_tmp_bytes()
}

/// Writes the trace of each column of `a` into a scratch polynomial at the radix of `a`,
/// then normalizes it into `res`.
pub fn glwe_ci_trace_reference<BE, M, R, A>(module: &M, res: &mut R, a: &A, scratch: &mut ScratchArena<'_, BE>)
where
    BE: Backend,
    M: ModuleN + VecZnxCITrace<BE> + VecZnxNormalize<BE> + VecZnxNormalizeTmpBytes,
    R: GLWEToBackendMut<BE> + GLWEInfos,
    A: GLWEToBackendRef<BE> + GLWEInfos,
{
    assert!(
        scratch.available() >= glwe_ci_trace_tmp_bytes_reference::<BE, _, _, _>(module, res, a),
        "scratch.available(): {} < GLWECITrace::glwe_ci_trace_tmp_bytes: {}",
        scratch.available(),
        glwe_ci_trace_tmp_bytes_reference::<BE, _, _, _>(module, res, a)
    );
    res.set_encryption_metadata(a.encryption_metadata());
    res.set_canonical(true);
    let mut res = res.to_backend_mut();
    let a = a.to_backend_ref();
    assert_eq!(a.n(), 2 * res.n());
    assert!(a.n().as_usize() <= module.n());
    assert_eq!(res.rank(), a.rank());
    let (res_base2k, res_k) = (res.base2k().as_usize(), res.k().as_usize());
    for col in 0..res.rank().as_usize() + 1 {
        let (mut tmp, mut scratch_1) = scratch.borrow().take_vec_znx_scratch(res.n().as_usize(), 1, a.size());
        module.vec_znx_ci_trace(&mut tmp.to_backend_mut(), 0, &a.data, col);
        module.vec_znx_normalize(
            &mut res.data,
            res_base2k,
            res_k,
            0,
            col,
            &tmp.to_backend_ref(),
            a.base2k().as_usize(),
            0,
            &mut scratch_1,
        );
    }
}

/// Derived: embeds the conjugate-invariant secret and encrypts a switching key.
impl<BE: Backend> GLWECIKeyEncryptSk<BE> for Module<BE>
where
    Module<BE>: GLWESecretCIEmbed<BE> + GLWESwitchingKeyEncryptSk<BE>,
{
    fn glwe_ci_key_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        self.glwe_switching_key_encrypt_sk_tmp_bytes(infos)
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
        let sk_embedded = self.glwe_secret_ci_embed(sk_ci);
        self.glwe_switching_key_encrypt_sk(res, &sk_embedded, sk, source_xe, source_xa, scratch);
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
        let sk_embedded = self.glwe_secret_ci_embed(sk_ci);
        self.glwe_switching_key_encrypt_sk(res, sk, &sk_embedded, source_xe, source_xa, scratch);
    }
}
