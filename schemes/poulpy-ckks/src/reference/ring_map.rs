//! Maps between conjugate-invariant and standard ciphertexts over the core
//! conjugate-invariant maps and the GLWE key switch.
use crate::CKKSResult as Result;
use poulpy_core::{
    GLWEBytesOf, GLWECIEmbed, GLWECITrace, GLWEKeyswitch,
    layouts::{
        Degree, GGLWEInfos, GGLWEPreparedToBackendRef, GLWE, GLWECIEmbedKeyPrepared, GLWECITraceKeyPrepared, GLWEInfos,
        GLWELayout, GLWEToBackendMut, GLWEToBackendRef, LWEInfos, Rank, TorusPrecision,
    },
};
use poulpy_hal::{
    api::ModuleN,
    layouts::{Backend, ConjugateInvariant, Data, Module, ScratchArena},
};

use crate::{
    CKKSCompositionError, CKKSCtBounds, CKKSInfos, GLWEPlaintextMeta, SetCKKSInfos, SlotsKind,
    error::checked_log_budget_sub,
    layouts::{CKKSCiphertext, ScratchArenaTakeCKKSCiphertext},
};

/// Reference of [`CKKSCIRingMapImpl`](crate::oep::CKKSCIRingMapImpl), which
/// [`impl_ckks_ci_ring_map_reference_full!`](crate::impl_ckks_ci_ring_map_reference_full) forwards to: the core
/// conjugate-invariant embedding and trace composed with the GLWE key switch.
pub trait CKKSCIRingMapReference<BE: Backend> {
    /// The key switch at the allocated width of `res_infos`, the widest input the destination holds.
    fn ckks_ci_embed_tmp_bytes_reference<R, K>(&self, res_infos: &R, key_infos: &K) -> usize
    where
        Self: GLWEKeyswitch<BE>,
        R: GLWEInfos,
        K: GGLWEInfos,
    {
        let res = allocated(res_infos);
        self.glwe_keyswitch_tmp_bytes(&res, &res, key_infos)
    }

    /// Refuses before any write, then stamps `dst` (real slots, the width of `src`), embeds `src` with
    /// `glwe_ci_embed` and switches the result in place under `key`.
    fn ckks_ci_embed_reference<Dst, D, K>(
        &self,
        dst: &mut Dst,
        src: &CKKSCiphertext<D, BE::ZnxWord, ConjugateInvariant>,
        key: &GLWECIEmbedKeyPrepared<K, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: ModuleN + GLWECIEmbed<BE> + GLWEKeyswitch<BE>,
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        D: Data,
        GLWE<D, BE::ZnxWord>: GLWEToBackendRef<BE>,
        K: Data,
        GLWECIEmbedKeyPrepared<K, BE>: GGLWEPreparedToBackendRef<BE>,
    {
        const OP: &str = "ckks_ci_embed";
        check_embed(OP, self, dst, src)?;
        check_key(OP, "embed", key, dst.n(), dst.rank(), src.k())?;
        crate::ckks_ensure!(
            scratch.available() >= self.ckks_ci_embed_tmp_bytes_reference(dst, key),
            "{OP}: scratch below its query"
        );
        embed(self, dst, src);
        self.glwe_keyswitch_assign(dst, &key.to_backend_ref(), scratch);
        Ok(())
    }

    /// A copy shaped like `a_infos`, then the larger of the key switch at `a_infos` and `glwe_ci_trace` from
    /// `a_infos` to `res_infos`.
    fn ckks_ci_trace_tmp_bytes_reference<R, A, K>(&self, res_infos: &R, a_infos: &A, key_infos: &K) -> usize
    where
        Self: GLWEBytesOf<BE> + GLWEKeyswitch<BE> + GLWECITrace<BE>,
        R: GLWEInfos,
        A: GLWEInfos,
        K: GGLWEInfos,
    {
        BE::scratch_aligned(self.glwe_bytes_of_from_infos(a_infos))
            + self
                .glwe_keyswitch_tmp_bytes(a_infos, a_infos, key_infos)
                .max(self.glwe_ci_trace_tmp_bytes(res_infos, a_infos))
    }

    /// Refuses before any write, then switches `src` under `key` into a scratch copy, stamps `dst` (real slots, one
    /// bit less width and budget) and writes the `glwe_ci_trace` of the copy into it.
    fn ckks_ci_trace_reference<D, Src, K>(
        &self,
        dst: &mut CKKSCiphertext<D, BE::ZnxWord, ConjugateInvariant>,
        src: &Src,
        key: &GLWECITraceKeyPrepared<K, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: ModuleN + GLWEBytesOf<BE> + GLWEKeyswitch<BE> + GLWECITrace<BE>,
        D: Data,
        GLWE<D, BE::ZnxWord>: GLWEToBackendMut<BE>,
        Src: GLWEToBackendRef<BE> + CKKSCtBounds,
        K: Data,
        GLWECITraceKeyPrepared<K, BE>: GGLWEPreparedToBackendRef<BE>,
    {
        const OP: &str = "ckks_ci_trace";
        check_trace(OP, self, dst, src)?;
        check_key(OP, "trace", key, src.n(), src.rank(), src.k())?;
        let log_budget = trace_log_budget(OP, src)?;
        crate::ckks_ensure!(
            scratch.available() >= self.ckks_ci_trace_tmp_bytes_reference(dst, src, key),
            "{OP}: scratch below its query"
        );
        let (mut switched, mut scratch_1) = scratch.borrow().take_ckks_ciphertext_scratch(src, src.meta());
        self.glwe_keyswitch(&mut switched, src, &key.to_backend_ref(), &mut scratch_1);
        trace(self, dst, &switched, log_budget, &mut scratch_1);
        Ok(())
    }
}

impl<BE: Backend> CKKSCIRingMapReference<BE> for Module<BE> {}

/// The keyless embedding the fold shares, refusing under the caller's `op`.
pub(crate) fn ckks_ci_embed_keyless<BE, M, Dst, D>(
    op: &'static str,
    module: &M,
    dst: &mut Dst,
    src: &CKKSCiphertext<D, BE::ZnxWord, ConjugateInvariant>,
) -> Result<()>
where
    BE: Backend,
    M: ModuleN + GLWECIEmbed<BE> + ?Sized,
    Dst: GLWEToBackendMut<BE> + GLWEInfos + SetCKKSInfos,
    D: Data,
    GLWE<D, BE::ZnxWord>: GLWEToBackendRef<BE>,
{
    check_embed(op, module, dst, src)?;
    embed(module, dst, src);
    Ok(())
}

/// The keyless trace the unfold shares, refusing under the caller's `op`.
pub(crate) fn ckks_ci_trace_keyless<BE, M, D, Src>(
    op: &'static str,
    module: &M,
    dst: &mut CKKSCiphertext<D, BE::ZnxWord, ConjugateInvariant>,
    src: &Src,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: Backend,
    M: ModuleN + GLWECITrace<BE> + ?Sized,
    D: Data,
    GLWE<D, BE::ZnxWord>: GLWEToBackendMut<BE>,
    Src: GLWEToBackendRef<BE> + GLWEInfos + CKKSInfos,
{
    check_trace(op, module, dst, src)?;
    let log_budget = trace_log_budget(op, src)?;
    crate::ckks_ensure!(
        scratch.available() >= module.glwe_ci_trace_tmp_bytes(dst, src),
        "{op}: scratch below its query"
    );
    trace(module, dst, src, log_budget, scratch);
    Ok(())
}

fn embed<BE, M, Dst, D>(module: &M, dst: &mut Dst, src: &CKKSCiphertext<D, BE::ZnxWord, ConjugateInvariant>)
where
    BE: Backend,
    M: GLWECIEmbed<BE> + ?Sized,
    Dst: GLWEToBackendMut<BE> + GLWEInfos + SetCKKSInfos,
    D: Data,
    GLWE<D, BE::ZnxWord>: GLWEToBackendRef<BE>,
{
    dst.set_meta(GLWEPlaintextMeta {
        slots: SlotsKind::Real,
        ..src.meta()
    });
    dst.set_k(src.k());
    module.glwe_ci_embed(dst, &src.inner);
}

fn trace<BE, M, D, Src>(
    module: &M,
    dst: &mut CKKSCiphertext<D, BE::ZnxWord, ConjugateInvariant>,
    src: &Src,
    log_budget: usize,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: Backend,
    M: GLWECITrace<BE> + ?Sized,
    D: Data,
    GLWE<D, BE::ZnxWord>: GLWEToBackendMut<BE>,
    Src: GLWEToBackendRef<BE> + GLWEInfos + CKKSInfos,
{
    dst.set_meta(GLWEPlaintextMeta {
        slots: SlotsKind::Real,
        ..src.meta()
    });
    dst.set_log_budget(log_budget);
    module.glwe_ci_trace(&mut dst.inner, src, scratch);
}

fn invalid(op: &'static str, reason: impl Into<String>) -> crate::CKKSError {
    CKKSCompositionError::InvalidOperand {
        op,
        reason: reason.into(),
    }
    .into()
}

/// `standard` has twice the degree of `ci`, at most the module degree, and its rank.
fn check_ring_map<M: ModuleN + ?Sized, C: GLWEInfos, S: GLWEInfos>(
    op: &'static str,
    module: &M,
    ci: &C,
    standard: &S,
) -> Result<()> {
    let (ci_n, standard_n) = (ci.n().as_usize(), standard.n().as_usize());
    if standard_n != 2 * ci_n || standard_n > module.n() {
        return Err(invalid(
            op,
            format!(
                "the conjugate-invariant degree {ci_n} and standard degree {standard_n} are not N and 2N at most the module degree {}",
                module.n()
            ),
        ));
    }
    if ci.rank() != standard.rank() {
        return Err(invalid(op, "the conjugate-invariant and standard ranks differ"));
    }
    Ok(())
}

/// The ring map checks of the embedding, the destination holding the input width at its radix.
fn check_embed<M: ModuleN + ?Sized, Dst: GLWEInfos, Src: GLWEInfos>(
    op: &'static str,
    module: &M,
    dst: &Dst,
    src: &Src,
) -> Result<()> {
    check_ring_map(op, module, src, dst)?;
    check_holds(op, dst, src.k().as_usize())?;
    if dst.base2k() != src.base2k() {
        return Err(invalid(op, "the destination radix differs from the input's"));
    }
    Ok(())
}

/// The ring map checks of the trace, the destination holding the output, one bit narrower than the input.
fn check_trace<M: ModuleN + ?Sized, Dst: GLWEInfos, Src: GLWEInfos>(
    op: &'static str,
    module: &M,
    dst: &Dst,
    src: &Src,
) -> Result<()> {
    check_ring_map(op, module, dst, src)?;
    check_holds(op, dst, src.k().as_usize().saturating_sub(1))
}

/// Tr = 2·Re(m): normalizing it one bit narrower halves it into canonical digits; on a
/// power-of-two modulus that bit is the cost of the halving.
fn trace_log_budget<Src: CKKSInfos>(op: &'static str, src: &Src) -> Result<usize> {
    checked_log_budget_sub(op, src.log_budget(), 1)
}

fn check_holds<Dst: LWEInfos>(op: &'static str, dst: &Dst, k: usize) -> Result<()> {
    if k > dst.max_k().as_usize() {
        return Err(invalid(
            op,
            format!("the destination holds {} bits, the output needs {k}", dst.max_k().as_usize()),
        ));
    }
    Ok(())
}

/// `key` switches `rank`-ciphertexts of the standard degree `n` of an input `k` bits wide.
fn check_key<K: GGLWEInfos>(op: &'static str, name: &str, key: &K, n: Degree, rank: Rank, k: TorusPrecision) -> Result<()> {
    if key.n() != n {
        return Err(invalid(
            op,
            format!(
                "the {name} key degree {} is not the standard degree {}",
                key.n().as_usize(),
                n.as_usize()
            ),
        ));
    }
    if key.rank_in() != rank || key.rank_out() != rank {
        return Err(invalid(op, format!("the {name} key ranks differ from the ciphertext rank")));
    }
    if k > key.gglwe_layout().max_input_k() {
        return Err(invalid(op, format!("the input is wider than the {name} key's max_input_k")));
    }
    Ok(())
}

fn allocated<R: GLWEInfos>(infos: &R) -> GLWELayout {
    GLWELayout {
        n: infos.n(),
        base2k: infos.base2k(),
        k: infos.max_k(),
        rank: infos.rank(),
    }
}
