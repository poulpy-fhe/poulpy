//! Folding of CKKS ciphertexts into the standard ciphertexts a bootstrap refreshes.
//!
//! A per-ring step maps each input to a standard ciphertext of
//! degree `n` and back. Ring packing in between merges `g = N/n` of them, real pairs
//! as `x + i·y`, into `Σ_j X^j·ct_j(X^g)` of the bootstrap degree `N`, and switches
//! it to the bootstrap secret; unpacking switches back, and the component at `X^j`
//! of the result is `X^(-j)·ct` restricted to `X^g`, since the input secret lies in
//! that subring. Merging and splitting only move coefficients.

use std::collections::HashMap;

use crate::CKKSResult as Result;
use poulpy_core::{
    GLWEAdd, GLWEAutomorphism, GLWEKeyswitch, GLWENormalize, GLWERotate, GLWEShift, GLWESub, GLWEZero,
    layouts::{
        Base2K, Degree, GGLWEInfos, GLWE, GLWEInfos, GLWELayout, GLWEToBackendMut, GLWEToBackendRef, GetAutomorphismKey,
        GetGaloisElement, LWEInfos, ModuleCoreAlloc, Rank, TorusPrecision, prepared::GGLWEPreparedToBackendRef,
    },
};
use poulpy_hal::{
    api::{ModuleN, VecZnxSwitchRing},
    layouts::{Backend, Module, Ring, ScratchArena, Standard, ZnxWord, galois_element},
};

use crate::{
    CKKSCompositionError, CKKSCtBounds, CKKSInfos, CKKSLayout, CKKSMeta, SetCKKSInfos, SlotsKind,
    api::{CKKSAddOps, CKKSConjugateOps, CKKSImagOps, CKKSSubOps},
    layouts::{
        CKKSCiphertextOwned, CKKSFoldKeysLayout, CKKSModuleAlloc, CKKSRingCiphertext,
        validation::{validate_gadget_backend_view, validate_storage_capacity},
    },
};

/// Batch positions: an input alone, or a real pair `(re, im)`.
pub(crate) type Unit = (usize, Option<usize>);

/// Per-ring step of the fold around ring packing, for inputs of this ring
/// refreshed on the standard backend `BE`.
pub(crate) trait FoldRing<BE: Backend>: Ring {
    /// Groups `ins`, or outputs labeled like them, into units for a bootstrap on
    /// `module`; a pair shares one packed ciphertext as `re + i·im`.
    fn units(module: &Module<BE>, ins: &[CKKSRingCiphertext<BE, Self>]) -> Vec<Unit>;

    /// Whether real pairs split with the conjugation key of the input secret.
    const CONJUGATE_PAIRS: bool;

    /// Degree of the standard ciphertexts that represent inputs of degree `n`.
    fn packed_degree(n: usize) -> usize;

    /// Metadata of the standard ciphertext that represents an input labeled `meta`.
    fn packed_meta(meta: CKKSMeta) -> CKKSMeta;

    /// Scratch bound of [`Self::to_standard`] and [`Self::from_standard`] for
    /// outputs like `ct_out` split from standard ciphertexts like `part`.
    fn tmp_bytes<C1, C2>(module: &Module<BE>, ct_out: &C1, part: &C2, keys: &CKKSFoldKeysLayout) -> usize
    where
        C1: CKKSCtBounds,
        C2: CKKSCtBounds;

    /// Writes `src` into `dst`, a standard ciphertext of the packed degree.
    fn to_standard(
        module: &Module<BE>,
        dst: &mut CKKSCiphertextOwned<BE>,
        src: &CKKSRingCiphertext<BE, Self>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>;

    /// Writes the unit `outs` from `part`, its refreshed standard ciphertext of the
    /// packed degree under the input secret, keeping the labels of `outs`.
    fn from_standard<H>(
        module: &Module<BE>,
        outs: &mut [CKKSRingCiphertext<BE, Self>],
        part: &CKKSCiphertextOwned<BE>,
        automorphisms: Option<&H>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        H: GetAutomorphismKey<BE>;
}

/// Standard inputs: complex ones alone, and consecutive real ones of the bootstrap
/// degree in pairs, split back with the conjugation keys of the input secret as
/// `z + conj(z)` and `(z − conj(z))/i`. Halving drops one bit of the paired outputs.
impl<BE> FoldRing<BE> for Standard
where
    BE: Backend<Ring = Standard>,
    Module<BE>: ModuleN
        + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = BE::ZnxWord>
        + VecZnxSwitchRing<BE>
        + GLWERotate<BE>
        + GLWENormalize<BE>
        + GLWEAutomorphism<BE>
        + GLWEShift<BE>
        + CKKSAddOps<BE>
        + CKKSSubOps<BE>
        + CKKSImagOps<BE>
        + CKKSConjugateOps<BE>
        + CKKSModuleAlloc<BE>,
    GLWE<BE::OwnedBuf, BE::ZnxWord>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE>,
    CKKSCiphertextOwned<BE>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
{
    fn units(module: &Module<BE>, ins: &[CKKSCiphertextOwned<BE>]) -> Vec<Unit> {
        let pairs = ins.first().is_some_and(|ct| ct.n().as_usize() == module.n());
        let mut units = Vec::new();
        let mut i = 0;
        while i < ins.len() {
            let paired = pairs && i + 1 < ins.len() && ins[i].slots() == SlotsKind::Real && ins[i + 1].slots() == SlotsKind::Real;
            units.push((i, paired.then_some(i + 1)));
            i += 1 + usize::from(paired);
        }
        units
    }

    const CONJUGATE_PAIRS: bool = true;

    fn packed_degree(n: usize) -> usize {
        n
    }

    fn packed_meta(meta: CKKSMeta) -> CKKSMeta {
        meta
    }

    fn tmp_bytes<C1, C2>(module: &Module<BE>, ct_out: &C1, part: &C2, keys: &CKKSFoldKeysLayout) -> usize
    where
        C1: CKKSCtBounds,
        C2: CKKSCtBounds,
    {
        let size = ct_out.size().max(part.size());
        // Pairs and sparse parts are split at the bootstrap degree, sparse parts
        // `log_sparsity` bits wider.
        let split = keys.automorphism.filter(|key| key.n() == part.n()).map_or(0, |key| {
            let wide = layout(part.n().as_usize(), part.base2k(), part.k() + part.log_sparsity() as u32);
            module
                .ckks_conjugate_tmp_bytes(part, &key)
                .max(module.glwe_automorphism_tmp_bytes(&wide, &wide, &key))
                .max(module.glwe_shift_tmp_bytes(wide.size()))
        });
        module
            .glwe_normalize_tmp_bytes()
            .max(split)
            .max(module.ckks_add_tmp_bytes(size))
            .max(module.ckks_sub_tmp_bytes(size))
            .max(module.ckks_div_i_tmp_bytes(size))
    }

    fn to_standard(
        module: &Module<BE>,
        dst: &mut CKKSCiphertextOwned<BE>,
        src: &CKKSCiphertextOwned<BE>,
        _scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()> {
        switch_ring(module, dst, src);
        dst.set_meta(src.meta());
        Ok(())
    }

    fn from_standard<H>(
        module: &Module<BE>,
        outs: &mut [CKKSCiphertextOwned<BE>],
        part: &CKKSCiphertextOwned<BE>,
        automorphisms: Option<&H>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        H: GetAutomorphismKey<BE>,
    {
        let meta = outs[0].meta();
        match outs {
            [out] => {
                out.set_k(part.k());
                if part.is_canonical() {
                    switch_ring(module, out, part);
                } else {
                    module.glwe_normalize(out, part, scratch);
                }
                out.set_meta(meta);
            }
            [left, right] => {
                let automorphisms =
                    automorphisms.ok_or_else(|| anyhow::anyhow!("real pairs need the automorphism keys of the input secret"))?;
                let mut conj = module.ckks_ciphertext_alloc_from_glwe_infos(part);
                module.ckks_conjugate_into(&mut conj, part, automorphisms, scratch)?;
                module.ckks_add_into(left, part, &conj, scratch)?;
                module.ckks_sub_into(right, part, &conj, scratch)?;
                module.ckks_div_i_assign(right, scratch)?;
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

/// Reference planning queries of [`CKKSFoldLayoutOps`](crate::api::CKKSFoldLayoutOps).
pub trait CKKSFoldLayoutReference<BE: Backend> {
    fn ckks_fold_layout_reference<C: CKKSCtBounds>(&self, ct_in: &C, degree: Degree, keys: &CKKSFoldKeysLayout) -> GLWELayout;

    fn ckks_fold_tmp_bytes_reference<C1, C2>(&self, ct_out: &C1, ct_in: &C2, degree: Degree, keys: &CKKSFoldKeysLayout) -> usize
    where
        C1: CKKSCtBounds,
        C2: CKKSCtBounds;
}

impl<BE> CKKSFoldLayoutReference<BE> for Module<BE>
where
    BE: Backend,
    Module<BE>: ModuleN + GLWEKeyswitch<BE> + CKKSAddOps<BE> + CKKSImagOps<BE>,
    Standard: FoldRing<BE>,
{
    fn ckks_fold_layout_reference<C: CKKSCtBounds>(&self, ct_in: &C, degree: Degree, keys: &CKKSFoldKeysLayout) -> GLWELayout {
        let base2k = keys.ring_switch.map_or(ct_in.base2k(), |keys| keys.inbound.base2k());
        layout(degree.as_usize(), base2k, ct_in.k())
    }

    /// Covers every input ring: the packed parts are sized at the fold degree.
    fn ckks_fold_tmp_bytes_reference<C1, C2>(&self, ct_out: &C1, ct_in: &C2, degree: Degree, keys: &CKKSFoldKeysLayout) -> usize
    where
        C1: CKKSCtBounds,
        C2: CKKSCtBounds,
    {
        let folded = self.ckks_fold_layout_reference(ct_in, degree, keys);
        let refreshed = CKKSLayout {
            glwe_layout: GLWELayout { k: ct_out.k(), ..folded },
            meta: CKKSMeta {
                log_sparsity: ct_in.log_sparsity(),
                ..ct_out.meta()
            },
        };
        let packed = layout(degree.as_usize(), ct_in.base2k(), ct_in.k());
        let size = packed.size().max(folded.size()).max(refreshed.size());
        let switches = keys.ring_switch.map_or(0, |keys| {
            self.glwe_keyswitch_tmp_bytes(&folded, &packed, &keys.inbound)
                .max(self.glwe_keyswitch_tmp_bytes(&refreshed, &refreshed, &keys.outbound))
        });
        let split = CKKSLayout {
            glwe_layout: GLWELayout {
                base2k: ct_out.base2k(),
                ..refreshed.glwe_layout
            },
            meta: refreshed.meta,
        };
        switches
            .max(self.ckks_add_tmp_bytes(size))
            .max(self.ckks_mul_i_tmp_bytes(size))
            .max(Standard::tmp_bytes(self, ct_out, &split, keys))
    }
}

/// Reference fold of inputs of ring `R`, [`CKKSFoldOps`](crate::api::CKKSFoldOps).
pub trait CKKSFoldReference<BE: Backend, R: Ring> {
    fn ckks_fold_count_reference(&self, ins: &[CKKSRingCiphertext<BE, R>], degree: Degree) -> usize;

    fn ckks_unfold_galois_elements_reference<C: CKKSCtBounds>(&self, ct_in: &C) -> Vec<i64>;

    fn ckks_fold_reference<S>(
        &self,
        folded: &mut [CKKSCiphertextOwned<BE>],
        ins: &[CKKSRingCiphertext<BE, R>],
        inbound: Option<&S>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        S: GGLWEPreparedToBackendRef<BE> + GGLWEInfos;

    /// Unfolds `folded`, the ciphertexts of [`Self::ckks_fold_reference`] once
    /// bootstrapped, into `outs`, labeled like the inputs they were folded from;
    /// `folded` is switched back and normalized in place.
    fn ckks_unfold_reference<S, H>(
        &self,
        outs: &mut [CKKSRingCiphertext<BE, R>],
        folded: &mut [CKKSCiphertextOwned<BE>],
        outbound: Option<&S>,
        automorphisms: Option<&H>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        S: GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
        H: GetAutomorphismKey<BE>;
}

impl<BE, R> CKKSFoldReference<BE, R> for Module<BE>
where
    BE: Backend,
    R: FoldRing<BE>,
    Module<BE>: ModuleN
        + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = BE::ZnxWord>
        + VecZnxSwitchRing<BE>
        + GLWERotate<BE>
        + GLWEAdd<BE>
        + GLWESub<BE>
        + GLWEZero<BE>
        + GLWEShift<BE>
        + GLWENormalize<BE>
        + GLWEAutomorphism<BE>
        + GLWEKeyswitch<BE>
        + CKKSAddOps<BE>
        + CKKSImagOps<BE>
        + CKKSModuleAlloc<BE>,
    GLWE<BE::OwnedBuf, BE::ZnxWord>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE>,
    CKKSCiphertextOwned<BE>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
{
    fn ckks_fold_count_reference(&self, ins: &[CKKSRingCiphertext<BE, R>], degree: Degree) -> usize {
        let Some(input) = ins.first() else {
            return 0;
        };
        let g = degree.as_usize() / R::packed_degree(input.n().as_usize());
        R::units(self, ins)
            .len()
            .div_ceil((g << sparse_log::<BE, R>(self, input)).max(1))
    }

    /// `−1` when real inputs pair and split with the conjugation key, and one element
    /// per level of sparsity, at the degree of the packed parts. Pairs and sparse parts
    /// form at the module degree.
    fn ckks_unfold_galois_elements_reference<C: CKKSCtBounds>(&self, ct_in: &C) -> Vec<i64> {
        let n = R::packed_degree(ct_in.n().as_usize());
        let full = n == self.n();
        let pairs = R::CONJUGATE_PAIRS && full && ct_in.slots() == SlotsKind::Real;
        let log_g = if full { R::packed_meta(ct_in.meta()).log_sparsity } else { 0 };
        pairs
            .then_some(-1)
            .into_iter()
            .chain(sparse_split_galois_elements(n, log_g))
            .collect()
    }

    fn ckks_fold_reference<S>(
        &self,
        folded: &mut [CKKSCiphertextOwned<BE>],
        ins: &[CKKSRingCiphertext<BE, R>],
        inbound: Option<&S>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        S: GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
    {
        let module = self;
        let degree = folded_degree(folded)?;
        let g = validate_fold(module, ins, degree, inbound.is_some())?;
        let input = &ins[0];
        if let Some(inbound) = inbound {
            validate_ring_switch_key("inbound ring-switch key", inbound, degree, input.k().as_usize())?;
        }
        let units = R::units(module, ins);
        // A group fills every coefficient: `g` positions of the bootstrap degree, each
        // holding `2^log_g` sparse parts.
        let span = g << sparse_log::<BE, R>(module, input);
        crate::ckks_ensure!(
            folded.len() == units.len().div_ceil(span),
            "the batch folds into {} ciphertexts, got {}",
            units.len().div_ceil(span),
            folded.len()
        );
        let n = R::packed_degree(input.n().as_usize());
        for (dst, group) in folded.iter_mut().zip(units.chunks(span)) {
            let to_standard = |i: usize, scratch: &mut ScratchArena<'_, BE>| {
                let mut part = module.ckks_ciphertext_alloc_from_glwe_infos(&layout(n, input.base2k(), input.k()));
                R::to_standard(module, &mut part, &ins[i], scratch).map(|_| part)
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
            if span > 1 || !imags.is_empty() {
                meta.slots = SlotsKind::Complex;
            }
            if span > 1 {
                meta.log_sparsity = 0;
            }
            // The group shares the input secret, so it is merged before one inbound switch.
            let mut packed = module.ckks_ciphertext_alloc_from_glwe_infos(&layout(degree, input.base2k(), input.k()));
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
            match inbound {
                Some(inbound) => module.glwe_keyswitch(dst, &packed, &inbound.to_backend_ref(), scratch),
                None => {
                    dst.set_k(packed.k());
                    switch_ring(module, dst, &packed);
                }
            }
        }
        Ok(())
    }

    fn ckks_unfold_reference<S, H>(
        &self,
        outs: &mut [CKKSRingCiphertext<BE, R>],
        folded: &mut [CKKSCiphertextOwned<BE>],
        outbound: Option<&S>,
        automorphisms: Option<&H>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        S: GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
        H: GetAutomorphismKey<BE>,
    {
        let module = self;
        let degree = folded_degree(folded)?;
        let g = validate_fold(module, outs, degree, outbound.is_some())?;
        let units = R::units(module, outs);
        let log_g = sparse_log::<BE, R>(module, &outs[0]);
        let (n, log_delta, log_sparsity) = (
            R::packed_degree(outs[0].n().as_usize()),
            outs[0].log_delta(),
            outs[0].log_sparsity(),
        );
        let span = g << log_g;
        crate::ckks_ensure!(
            folded.len() == units.len().div_ceil(span),
            "the batch unfolds from {} ciphertexts, got {}",
            units.len().div_ceil(span),
            folded.len()
        );
        // Retain the validated views so a custom key source is not queried again
        // after any folded ciphertext or output has been changed.
        let mut group_keys = Vec::with_capacity(folded.len());
        for (src, group) in folded.iter().zip(units.chunks(span)) {
            crate::ckks_ensure!(
                src.base2k().as_usize() <= <BE::ZnxWord as ZnxWord>::BITS - 2,
                "folded ciphertext radix exceeds the backend limit"
            );
            crate::ckks_ensure!(src.log_delta() <= src.k().as_usize(), "the folded scale exceeds the width");
            crate::ckks_ensure!(
                src.k().as_usize() > log_delta + 1,
                "a folded ciphertext has no message budget"
            );
            crate::ckks_ensure!(
                src.k() <= outs[0].k(),
                "unfold output width {} does not cover refreshed width {}",
                outs[0].k(),
                src.k()
            );
            if let Some(outbound) = outbound {
                validate_ring_switch_key("outbound ring-switch key", outbound, degree, src.k().as_usize())?;
            }
            let pairs = R::CONJUGATE_PAIRS && group.iter().any(|(_, im)| im.is_some());
            let required = pairs
                .then_some((-1, src.k()))
                .into_iter()
                .chain(sparse_split_galois_elements(n, log_g).map(|p| (p, src.k() + log_g as u32)));
            let mut keys = Vec::new();
            for (p, k) in required {
                let missing = || CKKSCompositionError::MissingAutomorphismKey {
                    op: "ckks_unfold",
                    rotation: p,
                    k: k.into(),
                };
                let key = automorphisms
                    .ok_or_else(missing)?
                    .get_automorphism_key(p, k)
                    .map_err(|_| missing())?;
                validate_ring_switch_key("unfold automorphism key", &&key, n, k.as_usize())?;
                keys.push(key);
            }
            group_keys.push(keys);
        }
        for ((src, group), keys) in folded.iter_mut().zip(units.chunks(span)).zip(&group_keys) {
            let keys: HashMap<_, _> = keys.iter().map(|key| (key.p(), key)).collect();
            if let Some(outbound) = outbound {
                module.glwe_keyswitch_assign(src, &outbound.to_backend_ref(), scratch);
            }
            // Convert before extraction: normalization runs at the bootstrap degree,
            // while the extracted components may belong to a smaller ring.
            module.glwe_normalize_assign(src, scratch);
            let mut converted;
            let src = if src.base2k() == outs[0].base2k() {
                &*src
            } else {
                converted = module.ckks_ciphertext_alloc_from_glwe_infos(&layout(degree, outs[0].base2k(), src.k()));
                module.glwe_normalize(&mut converted, src, scratch);
                converted.set_meta(src.meta());
                &converted
            };
            let meta = CKKSMeta {
                log_sparsity,
                ..src.meta()
            };
            for t in 0..g.min(group.len()) {
                let mut part = module.ckks_ciphertext_alloc_from_glwe_infos(&layout(n, src.base2k(), src.k()));
                extract(module, &mut part, &*src, t);
                part.set_meta(meta);
                let parts = if log_g > 0 {
                    split_sparse(module, &part, log_g, Some(&keys), scratch)?
                } else {
                    vec![part]
                };
                for (u, part) in parts.iter().enumerate() {
                    let Some(&(re, im)) = group.get(t + g * u) else {
                        continue;
                    };
                    let end = im.unwrap_or(re) + 1;
                    R::from_standard(module, &mut outs[re..end], part, Some(&keys), scratch)?;
                }
            }
        }
        Ok(())
    }
}

/// Log of the number of sparse parts merged per position: the sparsity of the
/// packed parts, at the bootstrap degree.
fn sparse_log<BE, R>(module: &Module<BE>, input: &CKKSRingCiphertext<BE, R>) -> usize
where
    BE: Backend,
    R: FoldRing<BE>,
    Module<BE>: ModuleN,
{
    if R::packed_degree(input.n().as_usize()) == module.n() {
        R::packed_meta(input.meta()).log_sparsity
    } else {
        0
    }
}

/// Elements that split sparse parts of degree `n` merged `2^log_g` at a time, from
/// the order-2 automorphism down: level `l` fixes `Z[X^(2^l)]` and negates `X^(2^(l−1))`.
pub(crate) fn sparse_split_galois_elements(n: usize, log_g: usize) -> impl Iterator<Item = i64> {
    (1..=log_g).map(move |l| galois_element(1 << (n.ilog2() as usize - 1 - l), 2 * n as i64))
}

/// Splits `part`, which holds `Σ_u X^u·p_u` with every `p_u` in `Z[X^(2^log_g)]`, into
/// the `p_u` by the reverse of their merge: one right shift by `log_g` at the root,
/// then `c ± φ(c)` per level, with `X^(−2^(l−1))` on the odd side. Every leaf is the
/// normalized trace of `X^(−u)·part` onto `Z[X^(2^log_g)]`, which is exact on the
/// torus, unlike a halving at each level. The levels multiply the rounding of the
/// shift and of every automorphism by up to `2^log_g`, so the tree runs `log_g` bits
/// wider than `part` and each leaf is normalized back once.
fn split_sparse<BE, H>(
    module: &Module<BE>,
    part: &CKKSCiphertextOwned<BE>,
    log_g: usize,
    automorphisms: Option<&H>,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<Vec<CKKSCiphertextOwned<BE>>>
where
    BE: Backend,
    H: GetAutomorphismKey<BE>,
    Module<BE>: VecZnxSwitchRing<BE>
        + GLWERotate<BE>
        + GLWEAdd<BE>
        + GLWESub<BE>
        + GLWEShift<BE>
        + GLWENormalize<BE>
        + GLWEAutomorphism<BE>
        + CKKSModuleAlloc<BE>,
    CKKSCiphertextOwned<BE>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
{
    let automorphisms =
        automorphisms.ok_or_else(|| anyhow::anyhow!("sparse inputs need the automorphism keys of the input secret"))?;
    let n = part.n().as_usize();
    let wide = layout(n, part.base2k(), part.k() + log_g as u32);
    let mut root = module.ckks_ciphertext_alloc_from_glwe_infos(&wide);
    switch_ring(module, &mut root, part);
    module.glwe_rsh(log_g, &mut root, scratch);
    let mut nodes = vec![root];
    let mut image = module.ckks_ciphertext_alloc_from_glwe_infos(&wide);
    let mut difference = module.ckks_ciphertext_alloc_from_glwe_infos(&wide);
    for (level, p) in sparse_split_galois_elements(n, log_g).enumerate() {
        let key = automorphisms
            .get_automorphism_key(p, wide.k)
            .map_err(|_| CKKSCompositionError::MissingAutomorphismKey {
                op: "ckks_unfold",
                rotation: p,
                k: wide.k.into(),
            })?;
        let mut odds = Vec::with_capacity(nodes.len());
        for node in &mut nodes {
            module.glwe_automorphism(&mut image, node, &key, scratch);
            module.glwe_sub(&mut difference, node, &image);
            let mut odd = module.ckks_ciphertext_alloc_from_glwe_infos(&wide);
            module.glwe_rotate(-(1 << level), &mut odd, &difference);
            odds.push(odd);
            module.glwe_add_assign(node, &image);
        }
        nodes.extend(odds);
    }
    Ok(nodes
        .iter()
        .map(|node| {
            let mut leaf = module.ckks_ciphertext_alloc_from_glwe_infos(part);
            module.glwe_normalize(&mut leaf, node, scratch);
            leaf.set_meta(part.meta());
            leaf
        })
        .collect())
}

/// Checks that `cts`, the inputs of a fold or the outputs of an unfold labeled
/// like them, are a nonempty rank-1 batch sharing one layout, scale and sparsity,
/// and returns the number `g` of packed ciphertexts merged into a ciphertext of
/// `degree`, which needs a ring switch when it exceeds one.
fn validate_fold<BE, R>(module: &Module<BE>, cts: &[CKKSRingCiphertext<BE, R>], degree: usize, ring_switch: bool) -> Result<usize>
where
    BE: Backend,
    R: FoldRing<BE>,
    Module<BE>: ModuleN,
{
    crate::ckks_ensure!(!cts.is_empty(), "a fold needs at least one ciphertext");
    let head = &cts[0];
    for ct in cts {
        validate_storage_capacity("fold ciphertext", ct)?;
        crate::ckks_ensure!(
            ct.base2k().as_usize() <= <BE::ZnxWord as ZnxWord>::BITS - 2,
            "ciphertext radix exceeds the backend limit"
        );
    }
    crate::ckks_ensure!(head.rank().as_usize() == 1, "folding supports rank-1 ciphertexts only");
    crate::ckks_ensure!(head.log_delta() <= head.k().as_usize(), "the scale exceeds the width");
    crate::ckks_ensure!(
        cts.iter().all(|ct| ct.glwe_layout() == head.glwe_layout()
            && ct.log_delta() == head.log_delta()
            && ct.log_sparsity() == head.log_sparsity()),
        "a batch must share its layout, scale and sparsity"
    );
    crate::ckks_ensure!(
        degree == module.n(),
        "the fold degree {degree} must be the module degree {}",
        module.n()
    );
    let n = R::packed_degree(head.n().as_usize());
    crate::ckks_ensure!(
        R::packed_meta(head.meta()).log_sparsity < n.ilog2() as usize,
        "fold ciphertext sparsity exceeds its packed degree"
    );
    crate::ckks_ensure!(
        degree.is_multiple_of(n),
        "the packed degree {n} must divide the fold degree {degree}"
    );
    let g = degree / n;
    crate::ckks_ensure!(
        g == 1 || ring_switch,
        "merging inputs of a smaller degree needs ring-switch keys"
    );
    Ok(g)
}

/// The degree of `folded`, which its ciphertexts share.
fn folded_degree<C: GLWEInfos>(folded: &[C]) -> Result<usize> {
    for ct in folded {
        validate_storage_capacity("folded ciphertext", ct)?;
        crate::ckks_ensure!(ct.rank().as_usize() == 1, "folded ciphertexts must have rank 1");
    }
    let degree = folded.first().map_or(0, |ct| ct.n().as_usize());
    crate::ckks_ensure!(
        degree > 0 && folded.iter().all(|ct| ct.n().as_usize() == degree),
        "a fold needs folded ciphertexts of one degree"
    );
    Ok(degree)
}

fn validate_ring_switch_key<BE, S>(name: &str, key: &S, degree: usize, k: usize) -> Result<()>
where
    BE: Backend,
    S: GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
{
    let base2k = key.base2k();
    crate::ckks_ensure!(
        (1..=<BE::ZnxWord as ZnxWord>::BITS - 2).contains(&base2k.as_usize()),
        "invalid {name} radix"
    );
    validate_gadget_backend_view(
        name,
        key,
        &key.to_backend_ref(),
        degree,
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
