//! Fold strategies of the batched bootstrapping pipeline.
use crate::CKKSResult as Result;
use poulpy_core::{
    GLWEAdd, GLWEKeyswitch, GLWENormalize, GLWERotate, GLWEZero,
    layouts::{
        Base2K, GGLWEInfos, GLWE, GLWEInfos, GLWELayout, GLWEToBackendMut, GLWEToBackendRef, LWEInfos, ModuleCoreAlloc, Rank,
        TorusPrecision, prepared::GGLWEPreparedToBackendRef,
    },
};
use poulpy_hal::{
    api::{ModuleN, VecZnxSwitchRing},
    layouts::{Backend, Module, ScratchArena, Standard, ZnxWord},
};

use crate::{
    CKKSCtBounds, CKKSError, CKKSInfos, CKKSLayout, CKKSMeta, SetCKKSInfos, SlotsKind,
    api::{CKKSAddOps, CKKSCIRingMapOps, CKKSConjugateOps, CKKSImagOps, CKKSSubOps},
    layouts::{
        BootstrappingContext, BootstrappingKeys, BootstrappingKeysLayout, BootstrappingPipeline, CKKSCiphertextOwned,
        CKKSModuleAlloc, RingSwitchKeys,
        validation::{validate_gadget_backend_view, validate_storage_capacity},
    },
    oep::CIBridge,
};

/// Folds a batch of input ciphertexts into standard ciphertexts of the
/// bootstrapping module, refreshes them, and unfolds the results.
///
/// [`CKKSBootstrappingOps::ckks_bootstrap`](crate::api::CKKSBootstrappingOps::ckks_bootstrap)
/// refreshes `ins` into `outs` through a fold, which decides how the inputs
/// are merged into the standard ciphertexts each bootstrap refreshes.
pub trait CKKSBootstrapFold<BE: Backend> {
    /// Input and output ciphertexts.
    type Ciphertext: CKKSCtBounds;

    /// Layouts `(refreshed, folded)` of the standard ciphertexts bootstrapped for
    /// outputs like `ct_out` and inputs like `ct_in`.
    fn bootstrap_layouts<C1, C2>(&self, module: &Module<BE>, ct_out: &C1, ct_in: &C2) -> (CKKSLayout, CKKSLayout)
    where
        C1: CKKSCtBounds,
        C2: CKKSCtBounds;

    /// Scratch bound of [`Self::refresh`] for outputs like `ct_out` and inputs like
    /// `ct_in` under keys shaped like `keys_layout`, given the scratch bound
    /// `bootstrap_bytes` of one bootstrap.
    fn tmp_bytes<C1, C2>(
        &self,
        module: &Module<BE>,
        ct_out: &C1,
        ct_in: &C2,
        keys_layout: &BootstrappingKeysLayout,
        bootstrap_bytes: usize,
    ) -> usize
    where
        C1: CKKSCtBounds,
        C2: CKKSCtBounds;

    /// Folds `ins` into standard ciphertexts, refreshes each with `bootstrap`, and
    /// unfolds the results into `outs`, which has the length of `ins`. `keys` are
    /// the bootstrap keys, available to the fold's own steps.
    #[allow(clippy::too_many_arguments)]
    fn refresh<F, K, B>(
        &self,
        module: &Module<BE>,
        outs: &mut [Self::Ciphertext],
        ins: &[Self::Ciphertext],
        ctx: &BootstrappingContext<BE, F>,
        keys: &K,
        bootstrap: B,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        K: BootstrappingKeys<BE>,
        B: FnMut(&mut CKKSCiphertextOwned<BE>, &CKKSCiphertextOwned<BE>, &mut ScratchArena<'_, BE>) -> Result<()>;
}

/// Refreshes standard ciphertexts, pairing real inputs.
///
/// Consecutive real-slot inputs sharing their layout, scale and sparsity, into
/// outputs sharing their layout, are packed as `left + i·right` into one bootstrap
/// and split back with its conjugation key: `2·Re = z + conj(z)` and
/// `2·Im = (z − conj(z))/i`. Halving drops one bit, so paired outputs carry one bit
/// less than a single bootstrap: allocate them at `plan.bootstrap_k(output_k + 1, log_delta)`
/// to keep `output_k`. Other inputs are refreshed one per bootstrap.
#[derive(Clone, Copy, Debug, Default)]
pub struct StandardFold;

impl<BE: Backend> CKKSBootstrapFold<BE> for StandardFold
where
    Module<BE>: CKKSAddOps<BE> + CKKSSubOps<BE> + CKKSImagOps<BE> + CKKSConjugateOps<BE> + CKKSModuleAlloc<BE>,
    CKKSCiphertextOwned<BE>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
{
    type Ciphertext = CKKSCiphertextOwned<BE>;

    fn bootstrap_layouts<C1, C2>(&self, _module: &Module<BE>, ct_out: &C1, ct_in: &C2) -> (CKKSLayout, CKKSLayout)
    where
        C1: CKKSCtBounds,
        C2: CKKSCtBounds,
    {
        (
            CKKSLayout {
                glwe_layout: ct_out.glwe_layout(),
                meta: ct_out.meta(),
            },
            CKKSLayout {
                glwe_layout: ct_in.glwe_layout(),
                meta: ct_in.meta(),
            },
        )
    }

    fn tmp_bytes<C1, C2>(
        &self,
        module: &Module<BE>,
        ct_out: &C1,
        ct_in: &C2,
        keys_layout: &BootstrappingKeysLayout,
        bootstrap_bytes: usize,
    ) -> usize
    where
        C1: CKKSCtBounds,
        C2: CKKSCtBounds,
    {
        let size = ct_out.size().max(ct_in.size());
        bootstrap_bytes
            .max(module.ckks_mul_i_tmp_bytes(size))
            .max(module.ckks_add_tmp_bytes(size))
            .max(module.ckks_conjugate_tmp_bytes(ct_out, &keys_layout.automorphism_key))
            .max(module.ckks_sub_tmp_bytes(size))
            .max(module.ckks_div_i_tmp_bytes(size))
    }

    fn refresh<F, K, B>(
        &self,
        module: &Module<BE>,
        outs: &mut [Self::Ciphertext],
        ins: &[Self::Ciphertext],
        _ctx: &BootstrappingContext<BE, F>,
        keys: &K,
        mut bootstrap: B,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        K: BootstrappingKeys<BE>,
        B: FnMut(&mut CKKSCiphertextOwned<BE>, &CKKSCiphertextOwned<BE>, &mut ScratchArena<'_, BE>) -> Result<()>,
    {
        for (left, right) in units(outs, ins) {
            let left_in = &ins[left];
            let Some(right) = right else {
                bootstrap(&mut outs[left], left_in, scratch)?;
                continue;
            };
            let right_in = &ins[right];
            let mut packed = module.ckks_ciphertext_alloc_from_glwe_infos(left_in);
            module.ckks_mul_i_into(&mut packed, right_in, scratch)?;
            module.ckks_add_assign(&mut packed, left_in, scratch)?;
            packed.set_slots(SlotsKind::Complex);
            let [left, right] = &mut outs[left..=right] else {
                unreachable!()
            };
            bootstrap(left, &packed, scratch)?;
            let mut conj = module.ckks_ciphertext_alloc_from_glwe_infos(left);
            module.ckks_conjugate_into(&mut conj, left, keys.rotation_keys(), scratch)?;
            module.ckks_sub_into(right, left, &conj, scratch)?;
            module.ckks_div_i_assign(right, scratch)?;
            module.ckks_add_assign(left, &conj, scratch)?;
            // Both hold twice their part; relabeling at the input scale drops that bit.
            for out in [left, right] {
                let meta = out.meta();
                out.set_meta(CKKSMeta {
                    log_delta: meta.log_delta + 1,
                    slots: SlotsKind::Real,
                    ..meta
                });
                out.set_log_delta(left_in.log_delta());
            }
        }
        Ok(())
    }
}

/// Refreshes standard ciphertexts of degree `n` under their own secret through a
/// bootstrap of degree `N = g·n`.
///
/// Real inputs are paired as in [`StandardFold`]. Groups of `g` inputs or pairs are
/// merged as `Σ_j X^j·ct_j(X^g)` and switched once to the bootstrap secret; the
/// refreshed result is switched back and split into its components at each `X^j`,
/// which only moves coefficients. A group with pairs is also conjugated with the
/// conjugation key of the bootstrap keys and switched back: conjugation moves the
/// conjugate of the component at `X^j` to `X^(g−j)`, times `X^(-g)`, which yields the
/// real and imaginary parts of each pair. Inputs must share their layout, scale and
/// sparsity, and outputs their layout; paired outputs are one bit narrower.
///
/// With `g > 1` the context must use full-slot transforms (`log_slots = log2(N) − 1`),
/// and it must use an identity recipe.
pub struct MergeFold<'a, S> {
    keys: &'a RingSwitchKeys<S>,
}

impl<'a, S> MergeFold<'a, S> {
    /// Folds through the ring-switch `keys` between the input and bootstrap secrets.
    pub fn new(keys: &'a RingSwitchKeys<S>) -> Self {
        Self { keys }
    }
}

impl<BE, S> CKKSBootstrapFold<BE> for MergeFold<'_, S>
where
    BE: Backend<Ring = Standard>,
    Module<BE>: ModuleN
        + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = BE::ZnxWord>
        + VecZnxSwitchRing<BE>
        + GLWERotate<BE>
        + GLWEAdd<BE>
        + GLWEZero<BE>
        + GLWENormalize<BE>
        + GLWEKeyswitch<BE>
        + CKKSAddOps<BE>
        + CKKSSubOps<BE>
        + CKKSImagOps<BE>
        + CKKSConjugateOps<BE>
        + CKKSModuleAlloc<BE>,
    GLWE<BE::OwnedBuf, BE::ZnxWord>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE>,
    CKKSCiphertextOwned<BE>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    S: GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
{
    type Ciphertext = CKKSCiphertextOwned<BE>;

    fn bootstrap_layouts<C1, C2>(&self, module: &Module<BE>, ct_out: &C1, ct_in: &C2) -> (CKKSLayout, CKKSLayout)
    where
        C1: CKKSCtBounds,
        C2: CKKSCtBounds,
    {
        switched_layouts(module, self.keys, ct_out, ct_in)
    }

    fn tmp_bytes<C1, C2>(
        &self,
        module: &Module<BE>,
        ct_out: &C1,
        ct_in: &C2,
        keys_layout: &BootstrappingKeysLayout,
        bootstrap_bytes: usize,
    ) -> usize
    where
        C1: CKKSCtBounds,
        C2: CKKSCtBounds,
    {
        let (refreshed, switched) = self.bootstrap_layouts(module, ct_out, ct_in);
        let packed = GLWELayout {
            base2k: ct_in.base2k(),
            ..switched.glwe_layout
        };
        let size = packed.size().max(switched.size()).max(refreshed.size());
        module
            .glwe_keyswitch_tmp_bytes(&switched, &packed, &self.keys.inbound)
            .max(module.ckks_add_tmp_bytes(size))
            .max(module.ckks_sub_tmp_bytes(size))
            .max(module.ckks_mul_i_tmp_bytes(size))
            .max(module.ckks_div_i_tmp_bytes(size))
            .max(bootstrap_bytes)
            .max(module.ckks_conjugate_tmp_bytes(&refreshed, &keys_layout.automorphism_key))
            .max(module.glwe_keyswitch_tmp_bytes(&refreshed, &refreshed, &self.keys.outbound))
            .max(module.glwe_normalize_tmp_bytes())
    }

    fn refresh<F, K, B>(
        &self,
        module: &Module<BE>,
        outs: &mut [Self::Ciphertext],
        ins: &[Self::Ciphertext],
        ctx: &BootstrappingContext<BE, F>,
        keys: &K,
        mut bootstrap: B,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        K: BootstrappingKeys<BE>,
        B: FnMut(&mut CKKSCiphertextOwned<BE>, &CKKSCiphertextOwned<BE>, &mut ScratchArena<'_, BE>) -> Result<()>,
    {
        let (Some(first_in), Some(first_out)) = (ins.first(), outs.first()) else {
            return Ok(());
        };
        validate_batch::<BE::ZnxWord, _>(outs, ins)?;
        let n = first_in.n().as_usize();
        crate::ckks_ensure!(
            module.n().is_multiple_of(n) && first_out.n().as_usize() == n,
            "the input degree must divide the bootstrap degree"
        );
        let g = module.n() / n;
        validate_ring_switch(module, self.keys, ctx, first_in, first_out, g > 1)?;
        let (base2k, out_base2k) = (self.keys.inbound.base2k(), first_out.base2k());
        let units = units(outs, ins);
        for group in units.chunks(g) {
            let input = &ins[group[0].0];
            let pairs: Vec<usize> = (0..group.len()).filter(|&j| group[j].1.is_some()).collect();
            let mut meta = input.meta();
            if g > 1 || !pairs.is_empty() {
                meta.slots = SlotsKind::Complex;
            }
            if g > 1 {
                meta.log_sparsity = 0;
            }
            // The group shares the input secret, so it is merged before one inbound switch.
            let mut packed = module.ckks_ciphertext_alloc_from_glwe_infos(&layout(module.n(), input.base2k(), input.k()));
            merge(
                module,
                &mut packed,
                group.iter().enumerate().map(|(j, u)| (j as i64, &ins[u.0])),
            );
            packed.set_meta(meta);
            if !pairs.is_empty() {
                let mut imag = module.ckks_ciphertext_alloc_from_glwe_infos(&packed);
                merge(
                    module,
                    &mut imag,
                    pairs.iter().map(|&j| (j as i64, &ins[group[j].1.unwrap()])),
                );
                imag.set_meta(meta);
                module.ckks_mul_i_assign(&mut imag, scratch)?;
                module.ckks_add_assign(&mut packed, &imag, scratch)?;
            }
            let mut switched = module.ckks_ciphertext_alloc_from_glwe_infos(&layout(module.n(), base2k, input.k()));
            switched.set_meta(meta);
            module.glwe_keyswitch(&mut switched, &packed, &self.keys.inbound.to_backend_ref(), scratch);
            let mut refreshed = module.ckks_ciphertext_alloc_from_glwe_infos(&layout(module.n(), base2k, outs[group[0].0].k()));
            refreshed.set_meta(meta);
            bootstrap(&mut refreshed, &switched, scratch)?;
            let mut conj = module.ckks_ciphertext_alloc_from_glwe_infos(&refreshed);
            if !pairs.is_empty() {
                module.ckks_conjugate_into(&mut conj, &refreshed, keys.rotation_keys(), scratch)?;
                module.glwe_keyswitch_assign(&mut conj, &self.keys.outbound.to_backend_ref(), scratch);
            }
            module.glwe_keyswitch_assign(&mut refreshed, &self.keys.outbound.to_backend_ref(), scratch);

            let k = refreshed.k();
            let mut normalized = module.ckks_ciphertext_alloc_from_glwe_infos(&layout(module.n(), out_base2k, k));
            let singles = group.iter().enumerate().filter(|(_, u)| u.1.is_none()).map(|(j, u)| (j, u.0));
            split(module, &mut normalized, &refreshed, outs, ins, singles, false, scratch);
            if !pairs.is_empty() {
                // `conj` holds the conjugate of the component at `X^j` at `X^(g−j)`, times `X^(-g)`.
                let reflected: Vec<_> = pairs
                    .iter()
                    .map(|&j| {
                        let mut part = module.ckks_ciphertext_alloc_from_glwe_infos(&layout(n, base2k, k));
                        extract(module, &mut part, &conj, (g - j) % g);
                        part
                    })
                    .collect();
                let mut conjugates = module.ckks_ciphertext_alloc_from_glwe_infos(&refreshed);
                merge(
                    module,
                    &mut conjugates,
                    pairs
                        .iter()
                        .zip(&reflected)
                        .map(|(&j, part)| ((if j == 0 { 0 } else { j + g }) as i64, part)),
                );
                conjugates.set_meta(refreshed.meta());
                let mut real = module.ckks_ciphertext_alloc_from_glwe_infos(&refreshed);
                module.ckks_add_into(&mut real, &refreshed, &conjugates, scratch)?;
                let mut imag = module.ckks_ciphertext_alloc_from_glwe_infos(&refreshed);
                module.ckks_sub_into(&mut imag, &refreshed, &conjugates, scratch)?;
                module.ckks_div_i_assign(&mut imag, scratch)?;
                let lefts = pairs.iter().map(|&j| (j, group[j].0));
                split(module, &mut normalized, &real, outs, ins, lefts, true, scratch);
                let rights = pairs.iter().map(|&j| (j, group[j].1.unwrap()));
                split(module, &mut normalized, &imag, outs, ins, rights, true, scratch);
            }
        }
        Ok(())
    }
}

/// Refreshes conjugate-invariant ciphertexts of degree `N` through a standard
/// bootstrap of degree `g·2N`.
///
/// Inputs are taken in pairs and must share their layout, scale and sparsity.
/// Each is unfolded to degree `2N`; groups of `g` pairs are merged as
/// `Σ_j X^j·(left_j + i·right_j)(X^g)` and switched once to the standard secret.
/// The refreshed result is switched back to the unfolded CI secret, whose symmetry
/// makes the split keyless: the component at each `X^j` folds into the left output
/// and, after a division by `i`, into the right one. An odd tail is refreshed without
/// a right part. Outputs keep the input scale and sparsity, with real slots.
///
/// The context must use full-slot transforms (`log_slots = log2(g·N)`) and an
/// identity recipe. Outputs are allocated at `plan.bootstrap_k(output_k + 1, log_delta)`:
/// the extra bit absorbs the factor two of the fold.
pub struct CIFold<'a, CI: Backend, S> {
    module: &'a Module<CI>,
    keys: &'a RingSwitchKeys<S>,
}

impl<'a, CI: Backend, S> CIFold<'a, CI, S> {
    /// Folds through the CI module `module` and the ring-switch `keys`.
    pub fn new(module: &'a Module<CI>, keys: &'a RingSwitchKeys<S>) -> Self {
        Self { module, keys }
    }
}

impl<BE, S> CKKSBootstrapFold<BE> for CIFold<'_, BE::CI, S>
where
    BE: CIBridge,
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
    Module<BE::CI>: ModuleN + CKKSCIRingMapOps<BE::CI>,
    GLWE<BE::OwnedBuf, BE::ZnxWord>:
        GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + GLWEToBackendMut<BE::CI> + GLWEToBackendRef<BE::CI>,
    CKKSCiphertextOwned<BE>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    CKKSCiphertextOwned<BE::CI>: GLWEToBackendMut<BE::CI> + GLWEToBackendRef<BE::CI> + CKKSCtBounds + SetCKKSInfos,
    S: GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
{
    type Ciphertext = CKKSCiphertextOwned<BE::CI>;

    fn bootstrap_layouts<C1, C2>(&self, module: &Module<BE>, ct_out: &C1, ct_in: &C2) -> (CKKSLayout, CKKSLayout)
    where
        C1: CKKSCtBounds,
        C2: CKKSCtBounds,
    {
        switched_layouts(module, self.keys, ct_out, ct_in)
    }

    fn tmp_bytes<C1, C2>(
        &self,
        module: &Module<BE>,
        ct_out: &C1,
        ct_in: &C2,
        _keys_layout: &BootstrappingKeysLayout,
        bootstrap_bytes: usize,
    ) -> usize
    where
        C1: CKKSCtBounds,
        C2: CKKSCtBounds,
    {
        let (refreshed, switched) = self.bootstrap_layouts(module, ct_out, ct_in);
        let packed = GLWELayout {
            base2k: ct_in.base2k(),
            ..switched.glwe_layout
        };
        let part = GLWELayout {
            n: (2 * self.module.n()).into(),
            ..refreshed.glwe_layout
        };
        let size = packed.size().max(switched.size()).max(refreshed.size());
        module
            .glwe_keyswitch_tmp_bytes(&switched, &packed, &self.keys.inbound)
            .max(module.ckks_add_tmp_bytes(size))
            .max(module.ckks_mul_i_tmp_bytes(size))
            .max(module.ckks_div_i_tmp_bytes(size))
            .max(bootstrap_bytes)
            .max(module.glwe_keyswitch_tmp_bytes(&refreshed, &refreshed, &self.keys.outbound))
            .max(self.module.ckks_ci_fold_tmp_bytes(ct_out, &part))
    }

    fn refresh<F, K, B>(
        &self,
        module: &Module<BE>,
        outs: &mut [Self::Ciphertext],
        ins: &[Self::Ciphertext],
        ctx: &BootstrappingContext<BE, F>,
        _keys: &K,
        mut bootstrap: B,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        K: BootstrappingKeys<BE>,
        B: FnMut(&mut CKKSCiphertextOwned<BE>, &CKKSCiphertextOwned<BE>, &mut ScratchArena<'_, BE>) -> Result<()>,
    {
        let (Some(first_in), Some(first_out)) = (ins.first(), outs.first()) else {
            return Ok(());
        };
        let ci = self.module;
        validate_batch::<BE::ZnxWord, _>(outs, ins)?;
        crate::ckks_ensure!(
            first_in.n().as_usize() == ci.n() && first_out.n().as_usize() == ci.n(),
            "CI ciphertext degree does not match the CI module"
        );
        crate::ckks_ensure!(
            first_in.log_sparsity() <= ci.n().ilog2() as usize,
            "invalid CI input sparsity"
        );
        let n = 2 * ci.n();
        crate::ckks_ensure!(
            module.n().is_multiple_of(n),
            "twice the CI degree must divide the standard module degree"
        );
        let g = module.n() / n;
        validate_ring_switch(module, self.keys, ctx, first_in, first_out, true)?;

        let base2k = self.keys.inbound.base2k();
        let lefts: Vec<usize> = (0..ins.len()).step_by(2).collect();
        for group in lefts.chunks(g) {
            let input = &ins[group[0]];
            let unfold = |i: usize| {
                let mut unfolded = module.ckks_ciphertext_alloc_from_glwe_infos(&layout(n, input.base2k(), input.k()));
                ci.ckks_ci_unfold(&mut unfolded, &ins[i]).map(|_| unfolded)
            };
            let parts = group.iter().map(|&i| unfold(i)).collect::<Result<Vec<_>>>()?;
            let rights = group
                .iter()
                .filter(|&&i| i + 1 < ins.len())
                .map(|&i| unfold(i + 1))
                .collect::<Result<Vec<_>>>()?;
            let mut meta = input.meta();
            meta.slots = if g == 1 && rights.is_empty() {
                SlotsKind::Real
            } else {
                SlotsKind::Complex
            };
            if g > 1 {
                meta.log_sparsity = 0;
            }
            // Unfolded inputs share the unfolded secret, so a group is merged before one inbound switch.
            let mut packed = module.ckks_ciphertext_alloc_from_glwe_infos(&layout(module.n(), input.base2k(), input.k()));
            merge(module, &mut packed, parts.iter().enumerate().map(|(j, ct)| (j as i64, ct)));
            packed.set_meta(meta);
            if !rights.is_empty() {
                let mut imag = module.ckks_ciphertext_alloc_from_glwe_infos(&packed);
                merge(module, &mut imag, rights.iter().enumerate().map(|(j, ct)| (j as i64, ct)));
                imag.set_meta(meta);
                module.ckks_mul_i_assign(&mut imag, scratch)?;
                module.ckks_add_assign(&mut packed, &imag, scratch)?;
            }
            let mut switched = module.ckks_ciphertext_alloc_from_glwe_infos(&layout(module.n(), base2k, input.k()));
            switched.set_meta(meta);
            module.glwe_keyswitch(&mut switched, &packed, &self.keys.inbound.to_backend_ref(), scratch);
            let mut refreshed = module.ckks_ciphertext_alloc_from_glwe_infos(&layout(module.n(), base2k, outs[group[0]].k()));
            refreshed.set_meta(meta);
            bootstrap(&mut refreshed, &switched, scratch)?;
            module.glwe_keyswitch_assign(&mut refreshed, &self.keys.outbound.to_backend_ref(), scratch);

            // The fold doubles the real part; relabeling at the input scale drops that bit.
            let mut part = module.ckks_ciphertext_alloc_from_glwe_infos(&layout(n, base2k, refreshed.k()));
            for (right, count) in [(0, group.len()), (1, rights.len())] {
                if right == 1 && count > 0 {
                    module.ckks_div_i_assign(&mut refreshed, scratch)?;
                }
                for (j, &i) in group.iter().enumerate().take(count) {
                    extract(module, &mut part, &refreshed, j);
                    part.set_meta(CKKSMeta {
                        log_sparsity: input.log_sparsity(),
                        ..refreshed.meta()
                    });
                    let out = &mut outs[i + right];
                    ci.ckks_ci_fold(out, &part, &mut scratch.borrow().into_backend())?;
                    out.set_log_delta(input.log_delta());
                }
            }
        }
        Ok(())
    }
}

/// Normalizes `src` to the radix of `normalized` and writes its component at `X^j`
/// into `outs[i]` for each `(j, i)` of `picks`, labeled like `ins[i]`. Paired parts
/// hold twice their value: relabeling them at the input scale drops that bit.
#[allow(clippy::too_many_arguments)]
fn split<BE>(
    module: &Module<BE>,
    normalized: &mut CKKSCiphertextOwned<BE>,
    src: &CKKSCiphertextOwned<BE>,
    outs: &mut [CKKSCiphertextOwned<BE>],
    ins: &[CKKSCiphertextOwned<BE>],
    picks: impl Iterator<Item = (usize, usize)>,
    paired: bool,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: Backend,
    Module<BE>: ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = BE::ZnxWord>
        + VecZnxSwitchRing<BE>
        + GLWERotate<BE>
        + GLWENormalize<BE>,
    GLWE<BE::OwnedBuf, BE::ZnxWord>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE>,
    CKKSCiphertextOwned<BE>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
{
    module.glwe_normalize(normalized, src, scratch);
    for (j, i) in picks {
        let out = &mut outs[i];
        extract(module, out, normalized, j);
        out.set_k(src.k());
        let meta = ins[i].meta();
        if paired {
            out.set_meta(CKKSMeta {
                log_delta: meta.log_delta + 1,
                slots: SlotsKind::Real,
                ..meta
            });
            out.set_log_delta(meta.log_delta);
        } else {
            out.set_meta(meta);
        }
    }
}

/// Batch positions: an input alone, or a real pair `(left, right)`.
type Unit = (usize, Option<usize>);

/// Pairs consecutive real inputs sharing their layout, scale and sparsity, into
/// outputs sharing their layout.
fn units<C: CKKSCtBounds>(outs: &[C], ins: &[C]) -> Vec<Unit> {
    let mut units = Vec::new();
    let mut i = 0;
    while i < ins.len() {
        let paired = i + 1 < ins.len()
            && ins[i].slots() == SlotsKind::Real
            && ins[i + 1].meta() == ins[i].meta()
            && ins[i + 1].glwe_layout() == ins[i].glwe_layout()
            && outs[i + 1].glwe_layout() == outs[i].glwe_layout();
        units.push((i, paired.then_some(i + 1)));
        i += 1 + usize::from(paired);
    }
    units
}

fn layout(n: usize, base2k: Base2K, k: TorusPrecision) -> GLWELayout {
    GLWELayout {
        n: n.into(),
        base2k,
        k,
        rank: Rank(1),
    }
}

/// Layouts `(refreshed, switched)` at the degree of `module` and the radix of `keys`.
fn switched_layouts<BE, S, C1, C2>(
    module: &Module<BE>,
    keys: &RingSwitchKeys<S>,
    ct_out: &C1,
    ct_in: &C2,
) -> (CKKSLayout, CKKSLayout)
where
    BE: Backend,
    Module<BE>: ModuleN,
    S: GGLWEInfos,
    C1: CKKSCtBounds,
    C2: CKKSCtBounds,
{
    let base2k = keys.inbound.base2k();
    (
        CKKSLayout {
            glwe_layout: layout(module.n(), base2k, ct_out.k()),
            meta: ct_out.meta(),
        },
        CKKSLayout {
            glwe_layout: layout(module.n(), base2k, ct_in.k()),
            meta: ct_in.meta(),
        },
    )
}

/// Writes `Σ X^shift·src(X^g)` into `dst`, whose degree is `g` times that of each
/// `src`; all share the radix of `dst`.
fn merge<'s, BE, D, S>(module: &Module<BE>, dst: &mut D, srcs: impl IntoIterator<Item = (i64, &'s S)>)
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
fn extract<BE, D, S>(module: &Module<BE>, dst: &mut D, src: &S, j: usize)
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

fn switch_ring<BE, D, S>(module: &Module<BE>, dst: &mut D, src: &S)
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

/// Checks that the batch is rank-1 and that inputs share their layout, scale and
/// sparsity, and outputs their layout.
fn validate_batch<W: ZnxWord, C: CKKSCtBounds>(outs: &[C], ins: &[C]) -> Result<()> {
    let (input, output) = (&ins[0], &outs[0]);
    for ct in ins.iter().chain(outs) {
        validate_storage_capacity("bootstrap fold ciphertext", ct)?;
        crate::ckks_ensure!(
            ct.base2k().as_usize() <= W::BITS - 2,
            "ciphertext radix exceeds the backend limit"
        );
    }
    crate::ckks_ensure!(
        input.rank().as_usize() == 1 && output.rank().as_usize() == 1,
        "ring-switched bootstrapping supports rank-1 ciphertexts only"
    );
    crate::ckks_ensure!(input.log_delta() <= input.k().as_usize(), "input scale exceeds its width");
    crate::ckks_ensure!(
        ins.iter().all(|ct| ct.glwe_layout() == input.glwe_layout()
            && ct.log_delta() == input.log_delta()
            && ct.log_sparsity() == input.log_sparsity())
            && outs.iter().all(|ct| ct.glwe_layout() == output.glwe_layout()),
        "bootstrap inputs and outputs must have matching layouts, scale and sparsity"
    );
    Ok(())
}

/// Checks the context and the ring-switch keys for inputs like `input` refreshed
/// into outputs like `output` on `module`.
fn validate_ring_switch<BE, F, S, C>(
    module: &Module<BE>,
    keys: &RingSwitchKeys<S>,
    ctx: &BootstrappingContext<BE, F>,
    input: &C,
    output: &C,
    full_slot: bool,
) -> Result<()>
where
    BE: Backend,
    Module<BE>: ModuleN,
    S: GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
    C: CKKSCtBounds,
{
    let log_slots = module.n().ilog2() as usize - 1;
    crate::ckks_ensure!(
        !full_slot
            || ctx.coeffs_to_slots().plan().log_slots() == log_slots
                && ctx.slots_to_coeffs().plan().log_slots() == log_slots
                && ctx
                    .coeffs_to_slots_bypass()
                    .is_none_or(|dft| dft.plan().log_slots() == log_slots),
        "merged bootstrapping requires full-slot transforms"
    );
    crate::ckks_ensure!(
        ctx.functional_message_modulus().is_none(),
        "ring-switched bootstrapping requires an identity recipe"
    );
    crate::ckks_ensure!(
        ctx.pipeline() != BootstrappingPipeline::C2SFirst || input.log_delta() <= ctx.eval_mod().plan.f_mod_log_delta,
        "bootstrap input scale exceeds the C2S-first working scale"
    );
    let base2k = keys.inbound.base2k();
    crate::ckks_ensure!(
        (1..=<BE::ZnxWord as ZnxWord>::BITS - 2).contains(&base2k.as_usize()),
        "invalid ring-switch key radix"
    );
    crate::ckks_ensure!(base2k == keys.outbound.base2k(), "ring-switch key radices differ");
    validate_gadget_backend_view(
        "inbound ring-switch key",
        &keys.inbound,
        &keys.inbound.to_backend_ref(),
        module.n(),
        base2k,
        input.k().as_usize().div_ceil(base2k.as_usize()),
    )?;
    crate::ckks_ensure!(
        keys.inbound.gglwe_layout().gadget_k() >= input.k(),
        "inbound ring-switch key does not cover the input width"
    );
    // The return switch runs after the standard pipeline has consumed its budget.
    let return_k = output
        .k()
        .as_usize()
        .checked_sub(ctx.output_consumed_bits(input.log_delta()))
        .ok_or_else(|| CKKSError::from(anyhow::anyhow!("insufficient bootstrap output width")))?;
    crate::ckks_ensure!(return_k > input.log_delta() + 1, "bootstrap output has no message budget");
    validate_gadget_backend_view(
        "outbound ring-switch key",
        &keys.outbound,
        &keys.outbound.to_backend_ref(),
        module.n(),
        base2k,
        return_k.div_ceil(base2k.as_usize()),
    )?;
    crate::ckks_ensure!(
        keys.outbound.gglwe_layout().gadget_k().as_usize() >= return_k,
        "outbound ring-switch key does not cover the return width"
    );
    Ok(())
}
