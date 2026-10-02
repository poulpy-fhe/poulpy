//! Maps between conjugate-invariant and standard ciphertexts, composed from the
//! core conjugate-invariant maps and a key switch.
use crate::CKKSResult as Result;
use poulpy_core::{
    GLWEBytesOf, GLWECIEmbed, GLWECITrace, GLWEKeyswitch,
    layouts::{
        GGLWEInfos, GGLWEPreparedToBackendRef, GLWE, GLWECIEmbedKeyPrepared, GLWECITraceKeyPrepared, GLWEInfos, GLWEToBackendMut,
        GLWEToBackendRef, LWEInfos,
    },
};
use poulpy_hal::layouts::{Backend, ConjugateInvariant, Data, Module, ScratchArena, Standard};

use crate::{
    CKKSCtBounds, CKKSInfos, CKKSMeta, SetCKKSInfos, SlotsKind,
    api::CKKSCIRingMapOps,
    error::checked_log_budget_sub,
    layouts::{CKKSCiphertext, ScratchArenaTakeCKKS},
};

/// Derived: the keyless maps composed with a key switch of the standard module.
impl<BE: Backend<Ring = Standard>> CKKSCIRingMapOps<BE> for Module<BE>
where
    Module<BE>: GLWECIEmbed<BE> + GLWECITrace<BE> + GLWEKeyswitch<BE> + GLWEBytesOf<BE>,
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
    dst.set_meta(CKKSMeta {
        slots: SlotsKind::Real,
        ..src.meta()
    });
    dst.set_k(src.k());
    module.glwe_ci_embed(dst, &src.inner);
    Ok(())
}

pub(crate) fn ckks_ci_trace_keyless_tmp_bytes<BE, R, A>(module: &Module<BE>, res_infos: &R, a_infos: &A) -> usize
where
    BE: Backend,
    Module<BE>: GLWECITrace<BE>,
    R: GLWEInfos,
    A: GLWEInfos,
{
    module.glwe_ci_trace_tmp_bytes(res_infos, a_infos)
}

pub(crate) fn ckks_ci_trace_keyless<BE, D, Src>(
    module: &Module<BE>,
    dst: &mut CKKSCiphertext<D, BE::ZnxWord, ConjugateInvariant>,
    src: &Src,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: Backend,
    Module<BE>: GLWECITrace<BE>,
    D: Data,
    GLWE<D, BE::ZnxWord>: GLWEToBackendMut<BE>,
    Src: GLWEToBackendRef<BE> + GLWEInfos + CKKSInfos,
{
    validate_ring_map(module, dst, src)?;
    // Tr = 2·Re(m): normalizing it one bit narrower halves it into canonical digits;
    // on a power-of-two modulus that bit is the cost of the halving.
    let log_budget = checked_log_budget_sub("ckks_ci_trace", src.log_budget(), 1)?;
    ensure_holds(dst, src.k().as_usize() - 1)?;
    dst.set_meta(CKKSMeta {
        slots: SlotsKind::Real,
        ..src.meta()
    });
    dst.set_log_budget(log_budget);
    module.glwe_ci_trace(&mut dst.inner, src, scratch);
    Ok(())
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
