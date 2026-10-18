//! Maps between conjugate-invariant and standard ciphertexts, composed from the
//! core conjugate-invariant maps and a key switch.
use crate::CKKSResult as Result;
use poulpy_core::{
    GLWEBytesOf, GLWECIEmbed, GLWECITrace, GLWEKeyswitch, GLWENormalize, GLWEShift, GLWESub, ScratchArenaTakeCore,
    layouts::{
        GGLWEInfos, GGLWEPreparedToBackendRef, GLWE, GLWECIEmbedKeyPrepared, GLWECITraceKeyPrepared, GLWEInfos, GLWELayout,
        GLWEToBackendMut, GLWEToBackendRef, LWEInfos, SetK,
    },
};
use poulpy_hal::layouts::{Backend, ConjugateInvariant, Data, Module, ScratchArena, Standard};

use crate::{
    CKKSCtBounds, CKKSInfos, GLWEPlaintextMeta, SetCKKSInfos, SlotsKind,
    api::CKKSCIRingMapOps,
    error::checked_log_budget_sub,
    layouts::{CKKSCiphertext, ScratchArenaTakeCKKS},
};

/// Derived: the keyless maps composed with a key switch of the standard module.
impl<BE: Backend<Ring = Standard>> CKKSCIRingMapOps<BE> for Module<BE>
where
    Module<BE>:
        GLWECIEmbed<BE> + GLWECITrace<BE> + GLWEKeyswitch<BE> + GLWEBytesOf<BE> + GLWENormalize<BE> + GLWEShift<BE> + GLWESub<BE>,
{
    fn ckks_ci_embed_tmp_bytes<R, K>(&self, res_infos: &R, key_infos: &K) -> usize
    where
        R: GLWEInfos,
        K: GGLWEInfos,
    {
        self.glwe_keyswitch_tmp_bytes(res_infos, res_infos, key_infos)
    }

    fn ckks_ci_embed<Dst, D, K>(
        &self,
        dst: &mut Dst,
        src: &CKKSCiphertext<D, BE::ZnxWord, ConjugateInvariant>,
        key: &GLWECIEmbedKeyPrepared<K, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        D: Data,
        GLWE<D, BE::ZnxWord>: GLWEToBackendRef<BE>,
        K: Data,
        GLWECIEmbedKeyPrepared<K, BE>: GGLWEPreparedToBackendRef<BE>,
    {
        crate::ckks_ensure!(key.n() == dst.n(), "the embed key has the standard degree");
        ckks_ci_embed_keyless(self, dst, src)?;
        self.glwe_keyswitch_assign(dst, &key.to_backend_ref(), scratch);
        Ok(())
    }

    fn ckks_ci_trace_tmp_bytes<R, A, K>(&self, res_infos: &R, a_infos: &A, key_infos: &K) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        K: GGLWEInfos,
    {
        self.glwe_bytes_of_from_infos(a_infos)
            + self
                .glwe_keyswitch_tmp_bytes(a_infos, a_infos, key_infos)
                .max(ckks_ci_trace_keyless_tmp_bytes(self, res_infos, a_infos))
    }

    fn ckks_ci_trace<D, Src, K>(
        &self,
        dst: &mut CKKSCiphertext<D, BE::ZnxWord, ConjugateInvariant>,
        src: &Src,
        key: &GLWECITraceKeyPrepared<K, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        D: Data,
        GLWE<D, BE::ZnxWord>: GLWEToBackendMut<BE>,
        Src: GLWEToBackendRef<BE> + CKKSCtBounds,
        K: Data,
        GLWECITraceKeyPrepared<K, BE>: GGLWEPreparedToBackendRef<BE>,
    {
        validate_ring_map(self, dst, src)?;
        crate::ckks_ensure!(key.n() == src.n(), "the trace key has the standard degree");
        let (mut switched, mut scratch_1) = scratch.borrow().take_ckks_ciphertext_scratch(src, src.meta());
        self.glwe_keyswitch(&mut switched, src, &key.to_backend_ref(), &mut scratch_1);
        ckks_ci_trace_keyless(self, dst, &switched, &mut scratch_1)
    }
}

pub(crate) fn ckks_ci_embed_keyless<BE, Dst, D>(
    module: &Module<BE>,
    dst: &mut Dst,
    src: &CKKSCiphertext<D, BE::ZnxWord, ConjugateInvariant>,
) -> Result<()>
where
    BE: Backend,
    Module<BE>: GLWECIEmbed<BE>,
    Dst: GLWEToBackendMut<BE> + GLWEInfos + SetCKKSInfos,
    D: Data,
    GLWE<D, BE::ZnxWord>: GLWEToBackendRef<BE>,
{
    validate_ring_map(module, src, dst)?;
    ensure_holds(dst, src.k().as_usize())?;
    crate::ckks_ensure!(dst.base2k() == src.base2k(), "embed keeps the radix of its input");
    dst.set_meta(GLWEPlaintextMeta {
        slots: SlotsKind::Real,
        ..src.meta()
    });
    dst.set_k(src.k());
    module.glwe_ci_embed(dst, &src.inner);
    Ok(())
}

/// Layout of the exact trace of `a` into a ciphertext like `res`: the width of `a`.
fn exact_trace_layout<R: GLWEInfos, A: GLWEInfos>(res_infos: &R, a_infos: &A) -> GLWELayout {
    GLWELayout {
        n: res_infos.n(),
        base2k: res_infos.base2k(),
        k: a_infos.k(),
        rank: res_infos.rank(),
    }
}

pub(crate) fn ckks_ci_trace_keyless_tmp_bytes<BE, R, A>(module: &Module<BE>, res_infos: &R, a_infos: &A) -> usize
where
    BE: Backend,
    Module<BE>: GLWECITrace<BE> + GLWEBytesOf<BE> + GLWENormalize<BE> + GLWEShift<BE>,
    R: GLWEInfos,
    A: GLWEInfos,
{
    let exact = exact_trace_layout(res_infos, a_infos);
    3 * module.glwe_bytes_of_from_infos(&exact)
        + module
            .glwe_ci_trace_tmp_bytes(&exact, a_infos)
            .max(module.glwe_normalize_tmp_bytes())
            .max(module.glwe_shift_tmp_bytes(exact.size()))
}

pub(crate) fn ckks_ci_trace_keyless<BE, D, Src>(
    module: &Module<BE>,
    dst: &mut CKKSCiphertext<D, BE::ZnxWord, ConjugateInvariant>,
    src: &Src,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: Backend,
    Module<BE>: GLWECITrace<BE> + GLWENormalize<BE> + GLWEShift<BE> + GLWESub<BE>,
    D: Data,
    GLWE<D, BE::ZnxWord>: GLWEToBackendMut<BE>,
    Src: GLWEToBackendRef<BE> + GLWEInfos + CKKSInfos,
{
    validate_ring_map(module, dst, src)?;
    // Tr = 2·Re(m): normalizing it one bit narrower halves it into canonical digits;
    // on a power-of-two modulus that bit is the cost of the halving.
    let log_budget = checked_log_budget_sub("ckks_ci_trace", src.log_budget(), 1)?;
    ensure_holds(dst, src.k().as_usize() - 1)?;
    dst.set_meta(GLWEPlaintextMeta {
        slots: SlotsKind::Real,
        ..src.meta()
    });
    dst.set_log_budget(log_budget);
    halve_trace_ties_to_even(module, &mut dst.inner, src, scratch);
    Ok(())
}

/// Writes into `res`, one bit narrower than `a`, the trace of `a` halved with its
/// ties rounded to even.
///
/// Half the coefficients of the exact trace `T` are odd, and rounding them half up
/// would move every one of both columns the same way: a bias that adds up in the
/// first slots. Rounding them to even moves them up and down alike. With `V` the
/// trace rounded to `res` and `U`, `W` the trace and `V` rounded one bit coarser,
/// `W − U` is one step of that coarser width exactly where `T ≡ 1 mod 4`, the ties
/// `V` rounded up to an odd value; half of it is one step of `res`, taken back.
fn halve_trace_ties_to_even<BE, R, A>(module: &Module<BE>, res: &mut R, a: &A, scratch: &mut ScratchArena<'_, BE>)
where
    BE: Backend,
    Module<BE>: GLWECITrace<BE> + GLWENormalize<BE> + GLWEShift<BE> + GLWESub<BE>,
    R: GLWEToBackendMut<BE> + GLWEInfos + SetK,
    A: GLWEToBackendRef<BE> + GLWEInfos,
{
    let k = a.k().as_usize();
    let layout = exact_trace_layout(res, a);
    let scratch = scratch.borrow();
    let (mut exact, scratch) = scratch.take_glwe_scratch(&layout);
    let (mut coarse, scratch) = scratch.take_glwe_scratch(&layout);
    let (mut ties, mut scratch) = scratch.take_glwe_scratch(&layout);

    module.glwe_ci_trace(&mut exact, a, &mut scratch);
    module.glwe_normalize(res, &exact, &mut scratch);
    coarse.set_k((k - 2).into());
    ties.set_k((k - 2).into());
    module.glwe_normalize(&mut coarse, &exact, &mut scratch);
    module.glwe_normalize(&mut ties, &*res, &mut scratch);
    module.glwe_sub_assign(&mut ties, &coarse);
    module.glwe_normalize_assign(&mut ties, &mut scratch);
    ties.set_k((k - 1).into());
    module.glwe_rsh(1, &mut ties, &mut scratch);
    module.glwe_sub_assign(res, &ties);
    module.glwe_normalize_assign(res, &mut scratch);
}

/// `standard` has twice the degree of `ci`, at most the module degree.
fn validate_ring_map<BE: Backend, C: GLWEInfos, S: GLWEInfos>(module: &Module<BE>, ci: &C, standard: &S) -> Result<()> {
    crate::ckks_ensure!(
        standard.n().as_usize() == 2 * ci.n().as_usize() && standard.n().as_usize() <= module.n(),
        "ring map requires a conjugate-invariant degree N and a standard degree 2N at most the module degree"
    );
    crate::ckks_ensure!(ci.rank() == standard.rank(), "ring map ranks do not match");
    Ok(())
}

/// `dst` has room for an output of `k` bits.
fn ensure_holds<D: LWEInfos>(dst: &D, k: usize) -> Result<()> {
    crate::ckks_ensure!(
        k <= dst.max_size() * dst.base2k().as_usize(),
        "ring map output storage cannot hold the output width"
    );
    Ok(())
}
