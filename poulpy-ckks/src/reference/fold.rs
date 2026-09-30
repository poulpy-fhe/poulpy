//! Folding of CKKS ciphertexts into the standard ciphertexts a bootstrap refreshes.
//!
//! A per-ring step ([`CKKSFoldRing`]) maps each input to a standard ciphertext of
//! degree `n` and back. Ring packing in between merges `g = N/n` of them, real pairs
//! as `x + i·y`, into `Σ_j X^j·ct_j(X^g)` of the bootstrap degree `N`, and switches
//! it to the bootstrap secret; unpacking switches back, and the component at `X^j`
//! of the result is `X^(-j)·ct` restricted to `X^g`, since the input secret lies in
//! that subring. Merging and splitting only move coefficients.

use crate::CKKSResult as Result;
use poulpy_core::{
    GLWEAdd, GLWEKeyswitch, GLWENormalize, GLWERotate, GLWEZero,
    layouts::{
        Base2K, GGLWEInfos, GLWE, GLWEInfos, GLWELayout, GLWEToBackendMut, GLWEToBackendRef, GetAutomorphismKey, LWEInfos,
        ModuleCoreAlloc, Rank, TorusPrecision, prepared::GGLWEPreparedToBackendRef,
    },
};
use poulpy_hal::{
    api::{ModuleN, VecZnxSwitchRing},
    layouts::{Backend, Module, Ring, ScratchArena, Standard, ZnxWord},
};

use crate::{
    CKKSCtBounds, CKKSInfos, CKKSLayout, CKKSMeta, SetCKKSInfos, SlotsKind,
    api::{CKKSAddOps, CKKSConjugateOps, CKKSImagOps, CKKSSubOps},
    layouts::{
        CKKSCiphertextOwned, CKKSFoldKeys, CKKSModuleAlloc,
        validation::{validate_gadget_backend_view, validate_storage_capacity},
    },
};

/// Batch positions: an input alone, or a real pair `(re, im)`.
pub type Unit = (usize, Option<usize>);

/// Per-ring step of the fold around ring packing, for inputs of backend `IN`
/// refreshed on backend `BE`.
pub trait CKKSFoldRing<BE: Backend, IN: Backend>: Ring {
    /// Groups `ins` into units; a pair shares one packed ciphertext as `re + i·im`.
    fn units(ins: &[CKKSCiphertextOwned<IN>]) -> Vec<Unit>;

    /// Degree of the standard ciphertexts that represent inputs of `input_module`.
    fn packed_degree(input_module: &Module<IN>) -> usize;

    /// Metadata of the standard ciphertext that represents an input labeled `meta`.
    fn packed_meta(meta: CKKSMeta) -> CKKSMeta;

    /// Scratch bound of [`Self::to_standard`] and [`Self::from_standard`] for
    /// outputs like `ct_out` split from standard ciphertexts like `part`.
    fn tmp_bytes<C1, C2, K>(module: &Module<BE>, input_module: &Module<IN>, ct_out: &C1, part: &C2, keys: &K) -> usize
    where
        C1: CKKSCtBounds,
        C2: CKKSCtBounds,
        K: CKKSFoldKeys<BE, IN>;

    /// Writes `src` into `dst`, a standard ciphertext of the packed degree.
    fn to_standard(
        module: &Module<BE>,
        input_module: &Module<IN>,
        dst: &mut CKKSCiphertextOwned<BE>,
        src: &CKKSCiphertextOwned<IN>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>;

    /// Writes the unit `outs`, labeled like `ins`, from the component at `X^j` of
    /// `unpacked`, the refreshed group under the input secret.
    #[allow(clippy::too_many_arguments)]
    fn from_standard<K>(
        module: &Module<BE>,
        input_module: &Module<IN>,
        outs: &mut [CKKSCiphertextOwned<IN>],
        ins: &[CKKSCiphertextOwned<IN>],
        unpacked: &CKKSCiphertextOwned<BE>,
        j: usize,
        keys: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        K: CKKSFoldKeys<BE, IN>;
}

/// Standard inputs: complex ones alone, consecutive real ones in pairs, split back
/// with the conjugation keys of the input secret as `z + conj(z)` and
/// `(z − conj(z))/i`. Halving drops one bit of the paired outputs.
impl<BE> CKKSFoldRing<BE, BE> for Standard
where
    BE: Backend<Ring = Standard>,
    Module<BE>: ModuleN
        + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = BE::ZnxWord>
        + VecZnxSwitchRing<BE>
        + GLWERotate<BE>
        + GLWENormalize<BE>
        + CKKSAddOps<BE>
        + CKKSSubOps<BE>
        + CKKSImagOps<BE>
        + CKKSConjugateOps<BE>
        + CKKSModuleAlloc<BE>,
    GLWE<BE::OwnedBuf, BE::ZnxWord>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE>,
    CKKSCiphertextOwned<BE>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
{
    fn units(ins: &[CKKSCiphertextOwned<BE>]) -> Vec<Unit> {
        let mut units = Vec::new();
        let mut i = 0;
        while i < ins.len() {
            let paired = i + 1 < ins.len() && ins[i].slots() == SlotsKind::Real && ins[i + 1].slots() == SlotsKind::Real;
            units.push((i, paired.then_some(i + 1)));
            i += 1 + usize::from(paired);
        }
        units
    }

    fn packed_degree(input_module: &Module<BE>) -> usize {
        input_module.n()
    }

    fn packed_meta(meta: CKKSMeta) -> CKKSMeta {
        meta
    }

    fn tmp_bytes<C1, C2, K>(_module: &Module<BE>, input_module: &Module<BE>, ct_out: &C1, part: &C2, keys: &K) -> usize
    where
        C1: CKKSCtBounds,
        C2: CKKSCtBounds,
        K: CKKSFoldKeys<BE, BE>,
    {
        let size = ct_out.size().max(part.size());
        let conjugate = keys
            .conjugation()
            .and_then(|h| h.lookup_automorphism_key(-1, part.k()).ok())
            .map_or(0, |key| input_module.ckks_conjugate_tmp_bytes(part, &key));
        input_module
            .glwe_normalize_tmp_bytes()
            .max(conjugate)
            .max(input_module.ckks_add_tmp_bytes(size))
            .max(input_module.ckks_sub_tmp_bytes(size))
            .max(input_module.ckks_div_i_tmp_bytes(size))
    }

    fn to_standard(
        module: &Module<BE>,
        _input_module: &Module<BE>,
        dst: &mut CKKSCiphertextOwned<BE>,
        src: &CKKSCiphertextOwned<BE>,
        _scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()> {
        switch_ring(module, dst, src);
        dst.set_meta(src.meta());
        Ok(())
    }

    fn from_standard<K>(
        module: &Module<BE>,
        input_module: &Module<BE>,
        outs: &mut [CKKSCiphertextOwned<BE>],
        ins: &[CKKSCiphertextOwned<BE>],
        unpacked: &CKKSCiphertextOwned<BE>,
        j: usize,
        keys: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        K: CKKSFoldKeys<BE, BE>,
    {
        let meta = ins[0].meta();
        let mut part =
            input_module.ckks_ciphertext_alloc_from_glwe_infos(&layout(input_module.n(), unpacked.base2k(), unpacked.k()));
        extract(module, &mut part, unpacked, j);
        part.set_meta(CKKSMeta {
            log_sparsity: meta.log_sparsity,
            ..unpacked.meta()
        });
        match outs {
            [out] => {
                out.set_k(part.k());
                input_module.glwe_normalize(out, &part, scratch);
                out.set_meta(meta);
            }
            [left, right] => {
                let conjugation = keys
                    .conjugation()
                    .ok_or_else(|| anyhow::anyhow!("real pairs need the conjugation keys of the input secret"))?;
                let mut conj = input_module.ckks_ciphertext_alloc_from_glwe_infos(&part);
                input_module.ckks_conjugate_into(&mut conj, &part, conjugation, scratch)?;
                input_module.ckks_add_into(left, &part, &conj, scratch)?;
                input_module.ckks_sub_into(right, &part, &conj, scratch)?;
                input_module.ckks_div_i_assign(right, scratch)?;
                // Both hold twice their part; relabeling at the input scale drops that bit.
                for out in [left, right] {
                    out.set_meta(CKKSMeta {
                        log_delta: meta.log_delta + 1,
                        slots: SlotsKind::Real,
                        ..meta
                    });
                    out.set_log_delta(meta.log_delta);
                }
            }
            _ => unreachable!("a unit has one or two ciphertexts"),
        }
        Ok(())
    }
}

/// Number of standard ciphertexts `ins` folds into on `module`.
pub fn ckks_fold_count_reference<BE, IN>(module: &Module<BE>, input_module: &Module<IN>, ins: &[CKKSCiphertextOwned<IN>]) -> usize
where
    BE: Backend,
    IN: Backend,
    IN::Ring: CKKSFoldRing<BE, IN>,
    Module<BE>: ModuleN,
{
    let g = module.n() / <IN::Ring as CKKSFoldRing<BE, IN>>::packed_degree(input_module);
    <IN::Ring as CKKSFoldRing<BE, IN>>::units(ins).len().div_ceil(g.max(1))
}

/// Layout of the standard ciphertexts that inputs like `ct_in` fold into on `module`.
pub fn ckks_fold_layout_reference<BE, IN, C, K>(module: &Module<BE>, ct_in: &C, keys: &K) -> CKKSLayout
where
    BE: Backend,
    IN: Backend,
    Module<BE>: ModuleN,
    C: CKKSCtBounds,
    K: CKKSFoldKeys<BE, IN>,
{
    let base2k = keys.ring_switch().map_or(ct_in.base2k(), |keys| keys.inbound.base2k());
    CKKSLayout {
        glwe_layout: layout(module.n(), base2k, ct_in.k()),
        meta: ct_in.meta(),
    }
}

/// Scratch bound of [`ckks_fold_reference`] and [`ckks_unfold_reference`] for
/// outputs like `ct_out` and inputs like `ct_in`.
pub fn ckks_fold_tmp_bytes_reference<BE, IN, C1, C2, K>(
    module: &Module<BE>,
    input_module: &Module<IN>,
    ct_out: &C1,
    ct_in: &C2,
    keys: &K,
) -> usize
where
    BE: Backend,
    IN: Backend,
    IN::Ring: CKKSFoldRing<BE, IN>,
    Module<BE>: ModuleN + GLWEKeyswitch<BE> + CKKSAddOps<BE> + CKKSImagOps<BE>,
    C1: CKKSCtBounds,
    C2: CKKSCtBounds,
    K: CKKSFoldKeys<BE, IN>,
{
    let folded = ckks_fold_layout_reference(module, ct_in, keys);
    let refreshed = CKKSLayout {
        glwe_layout: GLWELayout {
            k: ct_out.k(),
            ..folded.glwe_layout
        },
        meta: ct_out.meta(),
    };
    let packed = layout(module.n(), ct_in.base2k(), ct_in.k());
    let part = CKKSLayout {
        glwe_layout: GLWELayout {
            n: <IN::Ring as CKKSFoldRing<BE, IN>>::packed_degree(input_module).into(),
            ..refreshed.glwe_layout
        },
        meta: ct_out.meta(),
    };
    let size = packed.size().max(folded.size()).max(refreshed.size());
    let switches = keys.ring_switch().map_or(0, |keys| {
        module
            .glwe_keyswitch_tmp_bytes(&folded, &packed, &keys.inbound)
            .max(module.glwe_keyswitch_tmp_bytes(&refreshed, &refreshed, &keys.outbound))
    });
    switches
        .max(module.ckks_add_tmp_bytes(size))
        .max(module.ckks_mul_i_tmp_bytes(size))
        .max(<IN::Ring as CKKSFoldRing<BE, IN>>::tmp_bytes(
            module,
            input_module,
            ct_out,
            &part,
            keys,
        ))
}

/// Folds `ins` into `folded`, one standard ciphertext per bootstrap.
pub fn ckks_fold_reference<BE, IN, K>(
    module: &Module<BE>,
    input_module: &Module<IN>,
    folded: &mut [CKKSCiphertextOwned<BE>],
    ins: &[CKKSCiphertextOwned<IN>],
    keys: &K,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: Backend,
    IN: Backend,
    IN::Ring: CKKSFoldRing<BE, IN>,
    K: CKKSFoldKeys<BE, IN>,
    Module<BE>: ModuleN
        + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = BE::ZnxWord>
        + VecZnxSwitchRing<BE>
        + GLWERotate<BE>
        + GLWEAdd<BE>
        + GLWEZero<BE>
        + GLWEKeyswitch<BE>
        + CKKSAddOps<BE>
        + CKKSImagOps<BE>
        + CKKSModuleAlloc<BE>,
    GLWE<BE::OwnedBuf, BE::ZnxWord>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE>,
    CKKSCiphertextOwned<BE>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    CKKSCiphertextOwned<IN>: CKKSCtBounds,
{
    let g = validate_fold::<BE, IN, K>(module, input_module, ins, ins, keys)?;
    let input = &ins[0];
    if let Some(ring_switch) = keys.ring_switch() {
        validate_ring_switch_key("inbound ring-switch key", module, &ring_switch.inbound, input.k().as_usize())?;
    }
    let units = <IN::Ring as CKKSFoldRing<BE, IN>>::units(ins);
    crate::ckks_ensure!(
        folded.len() == units.len().div_ceil(g),
        "the batch folds into {} ciphertexts, got {}",
        units.len().div_ceil(g),
        folded.len()
    );
    let n = <IN::Ring as CKKSFoldRing<BE, IN>>::packed_degree(input_module);
    for (dst, group) in folded.iter_mut().zip(units.chunks(g)) {
        let to_standard = |i: usize, scratch: &mut ScratchArena<'_, BE>| {
            let mut part = module.ckks_ciphertext_alloc_from_glwe_infos(&layout(n, input.base2k(), input.k()));
            <IN::Ring as CKKSFoldRing<BE, IN>>::to_standard(module, input_module, &mut part, &ins[i], scratch).map(|_| part)
        };
        let mut parts = Vec::with_capacity(group.len());
        let mut imags = Vec::new();
        for &(re, im) in group {
            parts.push(to_standard(re, scratch)?);
            if let Some(im) = im {
                imags.push(to_standard(im, scratch)?);
            }
        }
        let mut meta = parts[0].meta();
        if g > 1 || !imags.is_empty() {
            meta.slots = SlotsKind::Complex;
        }
        if g > 1 {
            meta.log_sparsity = 0;
        }
        // The group shares the input secret, so it is merged before one inbound switch.
        let mut packed = module.ckks_ciphertext_alloc_from_glwe_infos(&layout(module.n(), input.base2k(), input.k()));
        merge(module, &mut packed, parts.iter().enumerate().map(|(j, ct)| (j as i64, ct)));
        packed.set_meta(meta);
        if !imags.is_empty() {
            let positions = group.iter().enumerate().filter(|(_, u)| u.1.is_some()).map(|(j, _)| j as i64);
            let mut imag = module.ckks_ciphertext_alloc_from_glwe_infos(&packed);
            merge(module, &mut imag, positions.zip(&imags));
            imag.set_meta(meta);
            module.ckks_mul_i_assign(&mut imag, scratch)?;
            module.ckks_add_assign(&mut packed, &imag, scratch)?;
        }
        dst.set_meta(meta);
        match keys.ring_switch() {
            Some(ring_switch) => module.glwe_keyswitch(dst, &packed, &ring_switch.inbound.to_backend_ref(), scratch),
            None => {
                dst.set_k(packed.k());
                switch_ring(module, dst, &packed);
            }
        }
    }
    Ok(())
}

/// Unfolds `refreshed`, the bootstrapped ciphertexts of [`ckks_fold_reference`],
/// into `outs`, labeled like `ins`.
#[allow(clippy::too_many_arguments)]
pub fn ckks_unfold_reference<BE, IN, K>(
    module: &Module<BE>,
    input_module: &Module<IN>,
    outs: &mut [CKKSCiphertextOwned<IN>],
    refreshed: &[CKKSCiphertextOwned<BE>],
    ins: &[CKKSCiphertextOwned<IN>],
    keys: &K,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: Backend,
    IN: Backend,
    IN::Ring: CKKSFoldRing<BE, IN>,
    K: CKKSFoldKeys<BE, IN>,
    Module<BE>: ModuleN + GLWEKeyswitch<BE> + CKKSModuleAlloc<BE>,
    CKKSCiphertextOwned<BE>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    CKKSCiphertextOwned<IN>: CKKSCtBounds,
{
    let g = validate_fold::<BE, IN, K>(module, input_module, outs, ins, keys)?;
    let units = <IN::Ring as CKKSFoldRing<BE, IN>>::units(ins);
    crate::ckks_ensure!(
        refreshed.len() == units.len().div_ceil(g),
        "the batch unfolds from {} ciphertexts, got {}",
        units.len().div_ceil(g),
        refreshed.len()
    );
    for (src, group) in refreshed.iter().zip(units.chunks(g)) {
        crate::ckks_ensure!(
            src.k().as_usize() > ins[0].log_delta() + 1,
            "the refreshed ciphertext has no message budget"
        );
        let switched = match keys.ring_switch() {
            Some(ring_switch) => {
                validate_ring_switch_key("outbound ring-switch key", module, &ring_switch.outbound, src.k().as_usize())?;
                let mut switched = module.ckks_ciphertext_alloc_from_glwe_infos(src);
                switched.set_meta(src.meta());
                module.glwe_keyswitch(&mut switched, src, &ring_switch.outbound.to_backend_ref(), scratch);
                Some(switched)
            }
            None => None,
        };
        let unpacked = switched.as_ref().unwrap_or(src);
        for (j, &(re, im)) in group.iter().enumerate() {
            let end = im.unwrap_or(re) + 1;
            <IN::Ring as CKKSFoldRing<BE, IN>>::from_standard(
                module,
                input_module,
                &mut outs[re..end],
                &ins[re..end],
                unpacked,
                j,
                keys,
                scratch,
            )?;
        }
    }
    Ok(())
}

/// Checks that `outs` and `ins` are nonempty rank-1 batches of the input module's
/// degree, each sharing one layout, with inputs sharing their scale and sparsity,
/// and returns the number `g` of packed ciphertexts merged per bootstrap.
fn validate_fold<BE, IN, K>(
    module: &Module<BE>,
    input_module: &Module<IN>,
    outs: &[CKKSCiphertextOwned<IN>],
    ins: &[CKKSCiphertextOwned<IN>],
    keys: &K,
) -> Result<usize>
where
    BE: Backend,
    IN: Backend,
    IN::Ring: CKKSFoldRing<BE, IN>,
    K: CKKSFoldKeys<BE, IN>,
    Module<BE>: ModuleN,
    CKKSCiphertextOwned<IN>: CKKSCtBounds,
{
    crate::ckks_ensure!(
        !ins.is_empty() && outs.len() == ins.len(),
        "a fold needs one output per input, got {} outputs for {} inputs",
        outs.len(),
        ins.len()
    );
    let (input, output) = (&ins[0], &outs[0]);
    for ct in ins.iter().chain(outs) {
        validate_storage_capacity("fold ciphertext", ct)?;
        crate::ckks_ensure!(
            ct.base2k().as_usize() <= <IN::ZnxWord as ZnxWord>::BITS - 2,
            "ciphertext radix exceeds the backend limit"
        );
    }
    crate::ckks_ensure!(
        input.n().as_usize() == input_module.n() && output.n().as_usize() == input_module.n(),
        "ciphertext degree does not match the input module"
    );
    crate::ckks_ensure!(
        input.rank().as_usize() == 1 && output.rank().as_usize() == 1,
        "folding supports rank-1 ciphertexts only"
    );
    crate::ckks_ensure!(input.log_delta() <= input.k().as_usize(), "input scale exceeds its width");
    crate::ckks_ensure!(
        ins.iter().all(|ct| ct.glwe_layout() == input.glwe_layout()
            && ct.log_delta() == input.log_delta()
            && ct.log_sparsity() == input.log_sparsity())
            && outs.iter().all(|ct| ct.glwe_layout() == output.glwe_layout()),
        "inputs and outputs must have matching layouts, scale and sparsity"
    );
    let n = <IN::Ring as CKKSFoldRing<BE, IN>>::packed_degree(input_module);
    crate::ckks_ensure!(
        module.n().is_multiple_of(n),
        "the packed degree {n} must divide the bootstrap degree {}",
        module.n()
    );
    let g = module.n() / n;
    crate::ckks_ensure!(
        g == 1 || keys.ring_switch().is_some(),
        "merging inputs of a smaller degree needs ring-switch keys"
    );
    if let Some(ring_switch) = keys.ring_switch() {
        let base2k = ring_switch.inbound.base2k();
        crate::ckks_ensure!(
            (1..=<BE::ZnxWord as ZnxWord>::BITS - 2).contains(&base2k.as_usize()),
            "invalid ring-switch key radix"
        );
        crate::ckks_ensure!(base2k == ring_switch.outbound.base2k(), "ring-switch key radices differ");
    }
    Ok(g)
}

fn validate_ring_switch_key<BE, S>(name: &str, module: &Module<BE>, key: &S, k: usize) -> Result<()>
where
    BE: Backend,
    Module<BE>: ModuleN,
    S: GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
{
    let base2k = key.base2k();
    validate_gadget_backend_view(
        name,
        key,
        &key.to_backend_ref(),
        module.n(),
        base2k,
        k.div_ceil(base2k.as_usize()),
    )?;
    crate::ckks_ensure!(
        key.gglwe_layout().gadget_k().as_usize() >= k,
        "{name} does not cover width {k}"
    );
    Ok(())
}

pub(crate) fn layout(n: usize, base2k: Base2K, k: TorusPrecision) -> GLWELayout {
    GLWELayout {
        n: n.into(),
        base2k,
        k,
        rank: Rank(1),
    }
}

/// Writes `Σ X^shift·src(X^g)` into `dst`, whose degree is `g` times that of each
/// `src`; all share the radix of `dst`.
pub(crate) fn merge<'s, BE, D, S>(module: &Module<BE>, dst: &mut D, srcs: impl IntoIterator<Item = (i64, &'s S)>)
where
    BE: Backend,
    Module<BE>: ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = BE::ZnxWord>
        + VecZnxSwitchRing<BE>
        + GLWERotate<BE>
        + GLWEAdd<BE>
        + GLWEZero<BE>,
    GLWE<BE::OwnedBuf, BE::ZnxWord>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE>,
    D: GLWEToBackendMut<BE> + GLWEInfos,
    S: GLWEToBackendRef<BE> + 's,
{
    let mut embedded = module.glwe_alloc_from_infos(dst);
    let mut shifted = module.glwe_alloc_from_infos(dst);
    module.glwe_zero(dst);
    for (shift, src) in srcs {
        switch_ring(module, &mut embedded, src);
        module.glwe_rotate(shift, &mut shifted, &embedded);
        module.glwe_add_assign(dst, &shifted);
    }
}

/// Writes the component of `src` at `X^j`, `X^(-j)·src` restricted to `X^g`, into
/// `dst` of `1/g` its degree. The kept coefficients are copies, so `dst` inherits
/// the canonical flag of `src`.
pub(crate) fn extract<BE, D, S>(module: &Module<BE>, dst: &mut D, src: &S, j: usize)
where
    BE: Backend,
    Module<BE>: ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = BE::ZnxWord> + VecZnxSwitchRing<BE> + GLWERotate<BE>,
    GLWE<BE::OwnedBuf, BE::ZnxWord>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE>,
    D: GLWEToBackendMut<BE>,
    S: GLWEToBackendRef<BE> + GLWEInfos,
{
    let mut shifted = module.glwe_alloc_from_infos(src);
    module.glwe_rotate(-(j as i64), &mut shifted, src);
    switch_ring(module, dst, &shifted);
    dst.set_canonical(src.to_backend_ref().is_canonical());
}

/// Copies `src` into `dst`, inserting or selecting coefficients when the degrees differ.
pub(crate) fn switch_ring<BE, D, S>(module: &Module<BE>, dst: &mut D, src: &S)
where
    BE: Backend,
    Module<BE>: VecZnxSwitchRing<BE>,
    D: GLWEToBackendMut<BE>,
    S: GLWEToBackendRef<BE>,
{
    let src = src.to_backend_ref();
    let canonical = src.is_canonical();
    {
        let mut view = dst.to_backend_mut();
        for col in 0..=src.rank().as_usize() {
            module.vec_znx_switch_ring(view.data_mut(), col, src.data(), col);
        }
    }
    dst.set_canonical(canonical);
}
