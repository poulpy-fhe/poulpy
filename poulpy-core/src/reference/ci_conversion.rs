//! Maps between conjugate-invariant GLWEs and secrets of degree `N` and standard
//! GLWEs and secrets of degree `2N`, expressed with HAL operations of the standard module.
use poulpy_hal::{
    api::{
        ModuleN, ScalarZnxCIEmbed, ScratchArenaTakeVecZnx, VecZnxCIEmbed, VecZnxCITrace, VecZnxNormalize, VecZnxNormalizeTmpBytes,
    },
    layouts::{Backend, Module, ScratchArena, VecZnxToBackendMut, VecZnxToBackendRef},
};

use crate::{
    GetDistributionMut,
    dist::Distribution,
    layouts::{GLWEInfos, GLWESecretToBackendMut, GLWESecretToBackendRef, GLWEToBackendMut, GLWEToBackendRef, LWEInfos},
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
    BE::vec_znx_bytes_of(res_infos.n().as_usize(), 1, a_infos.size()) + module.vec_znx_normalize_tmp_bytes()
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

/// Embeds each column of `src` into `res`, a standard secret of twice its degree, tagged
/// [`Distribution::ENCAPSULATED`]: the reference body of
/// [`GLWECIConversionImpl::glwe_secret_ci_embed`](crate::oep::GLWECIConversionImpl::glwe_secret_ci_embed), which
/// [`impl_glwe_ci_conversion_reference_full!`](crate::impl_glwe_ci_conversion_reference_full) forwards to.
pub fn glwe_secret_ci_embed_reference<BE, R, S>(module: &Module<BE>, res: &mut R, src: &S)
where
    BE: Backend,
    Module<BE>: ScalarZnxCIEmbed<BE>,
    R: GLWESecretToBackendMut<BE> + GetDistributionMut,
    S: GLWESecretToBackendRef<BE>,
{
    let src = src.to_backend_ref();
    {
        let mut res_ref = res.to_backend_mut();
        assert!(
            res_ref.n().as_usize() == 2 * src.n().as_usize(),
            "glwe_secret_ci_embed: embedded secret degree is not twice the source degree"
        );
        assert!(
            res_ref.n().as_usize() <= module.n(),
            "operand degree exceeds the module degree"
        );
        assert!(
            res_ref.rank() == src.rank(),
            "glwe_secret_ci_embed: embedded secret rank differs from the source rank"
        );
        for j in 0..src.rank().as_usize() {
            module.scalar_znx_ci_embed(res_ref.data_mut(), j, src.data(), j);
        }
    }
    *res.dist_mut() = Distribution::ENCAPSULATED("conjugate-invariant");
}
