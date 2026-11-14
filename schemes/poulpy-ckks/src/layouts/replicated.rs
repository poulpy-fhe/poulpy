//! Linear maps, conversions and bootstrapping of a replicated slot layout, the
//! small-batch scheme of [ePrint 2026/2513](https://eprint.iacr.org/2026/2513)
//! (see `docs/guides/replicated.md`).
use poulpy_core::GLWESecretSampling;
use poulpy_core::layouts::{GLWEAutomorphismKeyAlloc, GLWESecretAlloc, GLWESwitchingKeyAlloc, GLWETensorKeyAlloc};

use std::collections::HashMap;

use anyhow::{Context, Result, ensure};
use poulpy_core::{
    Distribution, GLWEAutomorphismKeyEncryptSk, GLWESwitchingKeyEncryptSk, GLWETensorKeyEncryptSk, GLWETrace, GetDistributionMut,
    layouts::{
        BackendGLWESecret, Base2K, Degree, Diagonals, GGLWELayout, GLWEAutomorphismKeyLayout, GLWEInfos, GLWELayout,
        GLWESecretLayout, GLWESwitchingKeyLayout, GLWETensorKeyLayout, GetDegree, LWEInfos, LinearTransformationBabySteps,
        LinearTransformationLayout, LinearTransformationPrepared, LinearTransformationStrategy, ModuleCoreAlloc, Rank,
        TorusPrecision, optimal_bsgs_giant_step,
    },
};
use poulpy_hal::{
    api::CnvPVecAlloc,
    layouts::{Backend, Module, Ring, ScratchArena, Standard},
    source::Source,
};

use crate::layouts::{DFTType, dft_matrices::ckks_dft_matrices_gen_blockwise};
use crate::{
    CKKSLayout, CKKSModuleInfos, CoeffsMeta, GLWEPlaintextMeta, ReplicatedLayout, Scale, SetCKKSInfos, SetGLWEPlaintextInfos,
    SlotLayout, SlotsKind,
    api::{CKKSEncodingHostOps, CKKSEncodingOps, CKKSEncodingScalar, CKKSLinearTransformationOps},
    encoding::linear_transformation::period_degree,
    layouts::{
        BootstrappingKeySet, BootstrappingKeysLayout, CKKSCiphertextAlloc, CKKSLinearTransformationAlloc, CKKSPlaintextAlloc,
        CKKSPlaintextOwned, ComplexDiagonals, EncapsulationKeysLayout, MaskedCoeffs,
        eval_mod::{
            EvalMod, EvalModPlan, EvalModType, ckks_eval_mod_replicated_compile, ckks_eval_mod_replicated_packed_compile,
        },
    },
    polynomial::SplitStrategy,
    shaping::conjugate_invariant,
};

/// Diagonals of the conversions between a CKKS ciphertext of period `k` and
/// a [`ReplicatedLayout`] of `k` values, each one linear transform.
///
/// With `n = kR` slots and `h = S/k`, `S` the ring's slots (`N/2` on the
/// standard ring, `N` on the conjugate-invariant ring):
///
/// - `to`: `k` diagonals writing value `z_(t/R mod k)` on the data slots `R | t`
///   and zero on the guards, quantized with shaping;
/// - `from` (when `R | k`): the gather diagonals, of weight `R`, reading only
///   data slots so that the normalized trace onto period `k` returns the
///   average of the `h/R` copies of each value, followed by `log2 h` trace
///   rotations by `k, 2k, ..., (h/2)k`.
///
/// [`Self::new_factored`] builds each map as the move between consecutive
/// slots and the data slots in `placement_factors` factors of about
/// `k^(1/placement_factors)` 0/1 mask diagonals, and `from` needs no `R | k`.
/// The last factor of `to` is quantized with shaping at `log_delta_diag`
/// bits, its others plainly at `inner_log_delta_diag` bits, enough for their
/// error to stay under the working precision. Every factor of `from` takes
/// `log_delta_diag` bits.
///
/// Both conversions compute `extra_bits` bits under the scale and consume
/// their diagonals' bits of budget. `log_delta` is the scale of the
/// replicated side.
pub struct ReplicatedConversion<BE: Backend, Rg: Ring = Standard> {
    pub(crate) layout: ReplicatedLayout,
    pub(crate) log_delta: usize,
    pub(crate) extra_bits: usize,
    pub(crate) to: ReplicatedTransform<BE, Rg>,
    pub(crate) from: Option<ReplicatedTransform<BE, Rg>>,
    pub(crate) trace: Vec<i64>,
}

/// Bits a diagonal holds above its scale: the largest weight, a sign and a carry.
const MAX_DIAGONAL_HEADROOM: usize = usize::BITS as usize + 2;

impl<BE: Backend, Rg: Ring> ReplicatedConversion<BE, Rg> {
    /// Scratch of [`Self::new`] and [`Self::new_factored`], for diagonals of at
    /// most `log_delta_diag` bits: for [`Self::new_factored`], the larger of its
    /// two diagonal scales.
    ///
    /// # Errors
    /// When `layout` needs more slots than the module has.
    pub fn new_tmp_bytes<F>(
        module: &Module<BE>,
        layout: &ReplicatedLayout,
        base2k: Base2K,
        log_delta_diag: usize,
    ) -> Result<usize>
    where
        F: CKKSEncodingScalar,
        Module<BE>: ReplicatedTransformModule<BE, F, Rg>,
    {
        ReplicatedTransform::<BE, Rg>::encode_tmp_bytes::<F>(module, layout, base2k, log_delta_diag)
    }

    /// Encodes and prepares the conversion maps on `module`, within the caller's `scratch`
    /// of [`Self::new_tmp_bytes`].
    pub fn new<F>(
        module: &Module<BE>,
        layout: ReplicatedLayout,
        base2k: Base2K,
        log_delta: usize,
        log_delta_diag: usize,
        extra_bits: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<Self>
    where
        F: CKKSEncodingScalar,
        Module<BE>: ReplicatedTransformModule<BE, F, Rg>,
    {
        let (k, r) = (1usize << layout.log_k(), 1usize << layout.log_r());
        let select = |s: usize, j: usize| {
            let read = s.is_multiple_of(r) && (s / r) % k == j;
            (if read { F::one() } else { F::zero() }, F::zero())
        };
        let to = ReplicatedTransform::from_period(module, layout, base2k, log_delta_diag, true, select, scratch)?;
        let identity = |c: usize, j: usize| (if c == j { F::one() } else { F::zero() }, F::zero());
        let from = if r <= k {
            Some(ReplicatedTransform::gather(
                module,
                layout,
                base2k,
                log_delta_diag,
                false,
                identity,
                scratch,
            )?)
        } else {
            None
        };
        let h = module.ckks_max_slots::<Rg>() / k;
        let trace = (0..h.trailing_zeros()).map(|l| (k << l) as i64).collect();
        Ok(Self {
            layout,
            log_delta,
            extra_bits,
            to,
            from,
            trace,
        })
    }

    /// [`Self::new`] with each map factored into the move between consecutive
    /// slots and the data slots, in `placement_factors` factors of about
    /// `k^(1/placement_factors)` mask diagonals, each costing about the square
    /// root of its diagonals in key switches, for the factors' bits.
    #[expect(clippy::too_many_arguments, reason = "the maps, their layout, scales and arena")]
    pub fn new_factored<F>(
        module: &Module<BE>,
        layout: ReplicatedLayout,
        base2k: Base2K,
        log_delta: usize,
        log_delta_diag: usize,
        inner_log_delta_diag: usize,
        extra_bits: usize,
        placement_factors: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<Self>
    where
        F: CKKSEncodingScalar,
        Module<BE>: ReplicatedTransformModule<BE, F, Rg>,
    {
        let (k, _) = check_period::<BE, Rg>(module, &layout)?;
        ensure!(
            (1..=layout.log_k()).contains(&placement_factors),
            "the placement takes between one and log2(k) = {} factors",
            layout.log_k()
        );
        let r = 1usize << layout.log_r();
        let factor = |log_delta_diag| {
            move |diagonals| ReplicatedFactor {
                diagonals,
                log_delta_diag,
                shaped: false,
                fitted_giant_step: true,
            }
        };
        let mut to: Vec<ReplicatedFactor<F>> = placement_spread::<F>(k, r, placement_factors)
            .into_iter()
            .map(factor(inner_log_delta_diag))
            .collect();
        let last = to.last_mut().context("the placement has factors")?;
        last.log_delta_diag = log_delta_diag;
        last.shaped = true;
        let weight = F::from_usize(r).context("the period is a small integer")?;
        let from = placement_compress::<F>(k, r, placement_factors, weight)
            .into_iter()
            .map(factor(log_delta_diag))
            .collect();
        let h = module.ckks_max_slots::<Rg>() / k;
        Ok(Self {
            layout,
            log_delta,
            extra_bits,
            to: ReplicatedTransform::from_factors(module, layout, base2k, to, scratch)?,
            from: Some(ReplicatedTransform::from_factors(module, layout, base2k, from, scratch)?),
            trace: (0..h.trailing_zeros()).map(|l| (k << l) as i64).collect(),
        })
    }

    pub fn layout(&self) -> ReplicatedLayout {
        self.layout
    }

    /// Allocates reusable baby steps for conversion into this layout from `src`.
    pub fn alloc_to_workspace<C: GLWEInfos>(&self, module: &Module<BE>, src: &C) -> ReplicatedTransformWorkspace<BE, Rg>
    where
        Module<BE>: CnvPVecAlloc<BE>,
    {
        self.to.alloc_workspace(
            module,
            &GLWELayout {
                k: (src.k().as_usize() + self.extra_bits).into(),
                ..src.glwe_layout()
            },
        )
    }

    /// Allocates reusable baby steps for conversion from this layout.
    pub fn alloc_from_workspace<C: GLWEInfos>(&self, module: &Module<BE>, src: &C) -> Result<ReplicatedTransformWorkspace<BE, Rg>>
    where
        Module<BE>: CnvPVecAlloc<BE>,
    {
        let from = self
            .from
            .as_ref()
            .context("the gather needs the period to divide the batch size")?;
        Ok(from.alloc_workspace(
            module,
            &GLWELayout {
                k: (src.k().as_usize() + self.extra_bits).into(),
                ..src.glwe_layout()
            },
        ))
    }

    /// Slot rotations the conversions use, for their automorphism keys.
    pub fn rotations(&self) -> Vec<i64> {
        let mut rotations = self.to.rotations();
        rotations.extend(self.from.iter().flat_map(ReplicatedTransform::rotations));
        rotations.extend(&self.trace);
        rotations.sort_unstable();
        rotations.dedup();
        rotations
    }
}

/// A linear map on the `n = kR` slots of a replicated layout's period, as a
/// chain of factors `y = Σ_o D_o ⊙ rot_o(x)`, each prepared as a BSGS transform
/// with giant step `⌈√d⌉`, `d` its diagonals, rounded up to a power of two.
/// The maps built from a matrix are one factor of `k` diagonals whatever the
/// map; the factored maps of the bootstrapping and the conversions chain a few
/// factors of few diagonals.
///
/// A shaped diagonal of giant step `rot` is quantized with the shaping
/// polynomial `g(X^(5^(-rot)))`, so that after the giant rotation its
/// quantization error is a multiple of `g`.
pub struct ReplicatedTransform<BE: Backend, Rg: Ring = Standard> {
    /// The factors in evaluation order, each with the bits of budget its diagonals consume.
    pub(crate) factors: Vec<(LinearTransformationPrepared<BE, Rg>, usize)>,
}

/// One factor of a [`ReplicatedTransform`] before encoding: its diagonals, the
/// bits of budget they consume, whether they are quantized with shaping, and
/// whether the BSGS giant step is fitted to its offsets. Otherwise the giant
/// step is the power of two whose square covers the diagonals, which the
/// dense maps of consecutive offsets share, and with it their keys.
pub(crate) struct ReplicatedFactor<F> {
    pub(crate) diagonals: ComplexDiagonals<F>,
    pub(crate) log_delta_diag: usize,
    pub(crate) shaped: bool,
    pub(crate) fitted_giant_step: bool,
}

/// Caller-owned prepared baby steps for one transform input layout.
/// Allocate once and reuse across evaluations; concurrent evaluations need
/// separate workspaces.
pub struct ReplicatedTransformWorkspace<BE: Backend, Rg: Ring = Standard> {
    pub(crate) input: GLWELayout,
    /// The baby steps of each factor, at the width its input has.
    pub(crate) babies: Vec<LinearTransformationBabySteps<BE, Rg>>,
}

impl<BE: Backend, Rg: Ring> ReplicatedTransform<BE, Rg> {
    /// Allocates baby steps for inputs with exactly the given working layout.
    pub fn alloc_workspace<C: GLWEInfos>(&self, module: &Module<BE>, input: &C) -> ReplicatedTransformWorkspace<BE, Rg>
    where
        Module<BE>: CnvPVecAlloc<BE>,
    {
        let babies = self
            .factor_inputs(&input.glwe_layout())
            .into_iter()
            .zip(&self.factors)
            .map(|(layout, (lt, _))| LinearTransformationBabySteps::alloc(module, lt.baby_steps(), &layout))
            .collect();
        ReplicatedTransformWorkspace {
            input: input.glwe_layout(),
            babies,
        }
    }

    /// The input layout of each factor for an input of layout `input`, each
    /// factor consuming its diagonals' bits.
    pub(crate) fn factor_inputs(&self, input: &GLWELayout) -> Vec<GLWELayout> {
        let mut k = input.k.as_usize();
        self.factors
            .iter()
            .map(|&(_, consumed)| {
                let layout = GLWELayout { k: k.into(), ..*input };
                k -= consumed;
                layout
            })
            .collect()
    }

    /// The bits of budget the chain consumes.
    pub fn consumed_bits(&self) -> usize {
        self.factors.iter().map(|&(_, consumed)| consumed).sum()
    }

    /// The number of factors the chain evaluates.
    pub fn factor_count(&self) -> usize {
        self.factors.len()
    }

    /// The map `y_s = Σ_(j<k) a(s, j)·x_j` from slots of period `k` to the `n`
    /// slots: `D_o[s] = a(s, (s + o) mod k)` for `0 <= o < k`. `shaped`
    /// quantizes the diagonals with the layout's shaping.
    pub fn from_period<F, A>(
        module: &Module<BE>,
        layout: ReplicatedLayout,
        base2k: Base2K,
        log_delta_diag: usize,
        shaped: bool,
        a: A,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<Self>
    where
        F: CKKSEncodingScalar,
        A: Fn(usize, usize) -> (F, F),
        Module<BE>: ReplicatedTransformModule<BE, F, Rg>,
    {
        Self::from_input_period(
            module,
            layout,
            1 << layout.log_k(),
            base2k,
            log_delta_diag,
            shaped,
            a,
            scratch,
        )
    }

    /// [`Self::from_period`] from slots of period `m`, a divisor of `n`.
    #[expect(clippy::too_many_arguments, reason = "the map, its layout, period, scale and arena")]
    pub fn from_input_period<F, A>(
        module: &Module<BE>,
        layout: ReplicatedLayout,
        m: usize,
        base2k: Base2K,
        log_delta_diag: usize,
        shaped: bool,
        a: A,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<Self>
    where
        F: CKKSEncodingScalar,
        A: Fn(usize, usize) -> (F, F),
        Module<BE>: ReplicatedTransformModule<BE, F, Rg>,
    {
        let (_, n) = check_period::<BE, Rg>(module, &layout)?;
        ensure!(n.is_multiple_of(m), "the input period {m} must divide the {n} slots");
        let weights = (0..m).map(|o| (o, (0..n).map(|s| a(s, (s + o) % m)).collect())).collect();
        Self::from_factors(
            module,
            layout,
            base2k,
            vec![ReplicatedFactor {
                diagonals: diagonals_from_weights(n, weights),
                log_delta_diag,
                shaped,
                fitted_giant_step: false,
            }],
            scratch,
        )
    }

    /// The gather `y` reading only the data slots, such that the normalized
    /// trace onto period `k`, which averages the `R` positions `c + k·i` of
    /// fiber `c`, returns `(Az)_c = Σ_j a(c, j)·z_j` for the data values `z`.
    /// Offset `o = R·u − (c mod R)` at `c + k·i` reads data value
    /// `m = ⌊c/R⌋ + (k/R)·i + u mod k`, weighted `R·a(c, m)`. Needs `R | k`.
    pub fn gather<F, A>(
        module: &Module<BE>,
        layout: ReplicatedLayout,
        base2k: Base2K,
        log_delta_diag: usize,
        shaped: bool,
        a: A,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<Self>
    where
        F: CKKSEncodingScalar,
        A: Fn(usize, usize) -> (F, F),
        Module<BE>: ReplicatedTransformModule<BE, F, Rg>,
    {
        let (k, n) = check_period::<BE, Rg>(module, &layout)?;
        let r = 1usize << layout.log_r();
        ensure!(r <= k, "a gather needs the period {r} to divide the batch size {k}");
        let weight = F::from_usize(r).unwrap();
        let mut weights: Vec<(usize, Vec<(F, F)>)> = Vec::new();
        for s in 0..n {
            let (c, i) = (s % k, s / k);
            for u in 0..k / r {
                let o = (r * u + n - c % r) % n;
                let m = (c / r + (k / r) * i + u) % k;
                let (re, im) = a(c, m);
                let index = match weights.iter().position(|(offset, _)| *offset == o) {
                    Some(index) => index,
                    None => {
                        weights.push((o, vec![(F::zero(), F::zero()); n]));
                        weights.len() - 1
                    }
                };
                weights[index].1[s] = (weight * re, weight * im);
            }
        }
        Self::from_factors(
            module,
            layout,
            base2k,
            vec![ReplicatedFactor {
                diagonals: diagonals_from_weights(n, weights),
                log_delta_diag,
                shaped,
                fitted_giant_step: false,
            }],
            scratch,
        )
    }

    /// Scratch of [`Self::from_period`], [`Self::from_input_period`] and [`Self::gather`] for `layout`
    /// with diagonals at scale `2^log_delta_diag`: the encoding of one diagonal and its preparation, at
    /// the widest diagonal a weight can need.
    ///
    /// # Errors
    /// When `layout` needs more slots than the module has.
    pub fn encode_tmp_bytes<F>(
        module: &Module<BE>,
        layout: &ReplicatedLayout,
        base2k: Base2K,
        log_delta_diag: usize,
    ) -> Result<usize>
    where
        F: CKKSEncodingScalar,
        Module<BE>: ReplicatedTransformModule<BE, F, Rg>,
    {
        let (_, n) = check_period::<BE, Rg>(module, layout)?;
        let degree = period_degree::<BE, _, Rg>(module, n);
        let diagonal = CKKSLayout {
            glwe_layout: GLWELayout {
                n: degree,
                base2k,
                k: TorusPrecision((log_delta_diag + MAX_DIAGONAL_HEADROOM) as u32),
                rank: Rank(0),
            },
            meta: GLWEPlaintextMeta {
                scale: Scale::Log(log_delta_diag),
                slots: SlotsKind::Complex,
                log_sparsity: 0,
                slot_layout: SlotLayout::Full,
            },
        };
        Ok(module
            .ckks_encoding_host_tmp_bytes(module.ckks_slots_of_degree::<Rg>(degree.as_usize()))
            .max(module.ckks_linear_transformation_rhs_prepare_tmp_bytes(&diagonal)))
    }

    /// Slot rotations of the transform, for their automorphism keys.
    pub fn rotations(&self) -> Vec<i64> {
        let lts = || self.factors.iter().map(|(lt, _)| lt);
        let babies = lts().flat_map(|lt| lt.giant_steps.iter().flat_map(|gs| gs.diagonals.iter().map(|d| d.baby)));
        let giants = lts().flat_map(|lt| lt.giant_steps.iter().map(|gs| gs.rot));
        let mut rotations: Vec<i64> = babies.chain(giants).filter(|&o| o != 0).collect();
        rotations.sort_unstable();
        rotations.dedup();
        rotations
    }

    /// Encodes and prepares `factors`, in evaluation order.
    pub(crate) fn from_factors<F>(
        module: &Module<BE>,
        layout: ReplicatedLayout,
        base2k: Base2K,
        factors: Vec<ReplicatedFactor<F>>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<Self>
    where
        F: CKKSEncodingScalar,
        Module<BE>: ReplicatedTransformModule<BE, F, Rg>,
    {
        let factors = factors
            .into_iter()
            .map(|factor| {
                let consumed = factor.log_delta_diag;
                Ok((Self::encode_factor(module, layout, base2k, factor, scratch)?, consumed))
            })
            .collect::<Result<_>>()?;
        Ok(Self { factors })
    }

    fn encode_factor<F>(
        module: &Module<BE>,
        layout: ReplicatedLayout,
        base2k: Base2K,
        factor: ReplicatedFactor<F>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<LinearTransformationPrepared<BE, Rg>>
    where
        F: CKKSEncodingScalar,
        Module<BE>: ReplicatedTransformModule<BE, F, Rg>,
    {
        let ReplicatedFactor {
            diagonals,
            log_delta_diag,
            shaped,
            fitted_giant_step,
        } = factor;
        let n = diagonals.slots();
        let values = |o: i64| {
            let (re, im) = (diagonals.re.get(o), diagonals.im.get(o));
            re.into_iter().chain(im).flatten().copied()
        };
        ensure!(
            !conjugate_invariant::<Rg>()
                || diagonals
                    .indexes()
                    .into_iter()
                    .all(|o| diagonals.im.get(o).is_none_or(|im| im.iter().all(|v| v.is_zero()))),
            "the real slots of the conjugate-invariant ring take real diagonals"
        );
        let largest = diagonals
            .indexes()
            .into_iter()
            .flat_map(values)
            .map(|v| v.abs())
            .fold(F::zero(), F::max);
        // Bits above the scale for the largest weight, plus a sign and a carry.
        let headroom = (usize::BITS - (largest.to_f64().unwrap_or(1.0).ceil() as usize).max(1).leading_zeros()) as usize + 2;
        debug_assert!(headroom <= MAX_DIAGONAL_HEADROOM);
        let count = diagonals.indexes().len();
        let strategy = if count <= 2 {
            LinearTransformationStrategy::Direct
        } else if fitted_giant_step {
            LinearTransformationStrategy::Bsgs {
                giant_step: optimal_bsgs_giant_step(diagonals.indexes(), n),
            }
        } else {
            let mut giant_step = 1;
            while giant_step * giant_step < count {
                giant_step <<= 1;
            }
            LinearTransformationStrategy::Bsgs { giant_step }
        };
        let meta = GLWEPlaintextMeta {
            scale: Scale::Log(log_delta_diag),
            slots: SlotsKind::Complex,
            log_sparsity: 0,
            slot_layout: SlotLayout::Full,
        };
        let infos = CKKSLayout {
            glwe_layout: GLWELayout {
                n: period_degree::<BE, _, Rg>(module, n),
                base2k,
                k: TorusPrecision((log_delta_diag + headroom) as u32),
                rank: Rank(0),
            },
            meta,
        };
        let lt_layout = LinearTransformationLayout {
            indexes: diagonals.indexes(),
            slots: n,
            strategy,
        };
        let mut lt = module.ckks_linear_transformation_alloc_from_infos::<_, Rg>(&lt_layout, &infos);
        let pt_slots = module.ckks_slots_of_degree::<Rg>(infos.glwe_layout.n.as_usize());
        let mut targets = lt.giant_steps.iter_mut().flat_map(|gs| {
            let rot = gs.rot;
            gs.diagonals.iter_mut().map(move |d| (rot, &mut d.plaintext))
        });
        let mut result = Ok(());
        // The builder visits the diagonals in the schedule's order.
        diagonals.build_transform(strategy, |re, im| {
            if let (Ok(()), Some((rot, pt))) = (&result, targets.next()) {
                let label = if shaped {
                    SlotLayout::Replicated(twisted(&layout, module.ckks_galois_element::<Rg>(-rot)))
                } else {
                    SlotLayout::Full
                };
                pt.set_meta(GLWEPlaintextMeta {
                    slot_layout: label,
                    ..meta
                });
                let reps = pt_slots / re.len();
                result = module
                    .ckks_encode_all_slots_host_into(pt, &re.repeat(reps), &im.repeat(reps), &mut scratch.borrow())
                    .map_err(anyhow::Error::from);
                // The diagonals multiply a ciphertext of the full layout.
                pt.set_slot_layout(SlotLayout::Full);
            }
        });
        result?;
        let first = lt.first_diagonal_plaintext().context("a transform has a diagonal")?;
        let mut prepared = LinearTransformationPrepared::<BE, Rg>::alloc_prepared_from_index(module, &lt.index(), first);
        module.ckks_linear_transformation_rhs_prepare(&mut prepared, &lt, scratch)?;
        Ok(prepared)
    }
}

/// `layout` with the shaping polynomial `U(Y^gal)`, the image of `g` under the
/// automorphism `X ↦ X^gal`: `Y = X^(n/(2R))` maps to `Y^(gal mod 4R)` in
/// `Z[Y]/(Y^(2R) + 1)`.
fn twisted(layout: &ReplicatedLayout, gal: i64) -> ReplicatedLayout {
    let two_r = 2usize << layout.log_r();
    let g = gal.rem_euclid(2 * two_r as i64) as usize;
    let mut coeffs = vec![0i16; two_r];
    for (j, &c) in layout.shaping().iter().enumerate() {
        let e = j * g % (2 * two_r);
        if e < two_r {
            coeffs[e] += c;
        } else {
            coeffs[e - two_r] -= c;
        }
    }
    ReplicatedLayout::new(layout.log_k() as u8, layout.log_r() as u8, &coeffs)
        .expect("an automorphism keeps the shaping polynomial invertible")
}

/// What building a [`ReplicatedTransform`] on the ring `Rg` needs of a module.
pub trait ReplicatedTransformModule<BE: Backend, F: CKKSEncodingScalar, Rg: Ring = Standard>:
    CKKSEncodingOps<BE, F, Rg>
    + CKKSCiphertextAlloc<BE>
    + CKKSPlaintextAlloc<BE>
    + CKKSModuleInfos
    + GetDegree
    + CKKSLinearTransformationOps<BE, Rg>
    + CnvPVecAlloc<BE>
{
}

impl<BE: Backend, F: CKKSEncodingScalar, Rg: Ring, M> ReplicatedTransformModule<BE, F, Rg> for M where
    M: CKKSEncodingOps<BE, F, Rg>
        + CKKSCiphertextAlloc<BE>
        + CKKSPlaintextAlloc<BE>
        + CKKSModuleInfos
        + GetDegree
        + CKKSLinearTransformationOps<BE, Rg>
        + CnvPVecAlloc<BE>
{
}

/// `(k, kR)` of a layout that fits the slots of the ring `Rg`.
fn check_period<BE: Backend, Rg: Ring>(module: &Module<BE>, layout: &ReplicatedLayout) -> Result<(usize, usize)>
where
    Module<BE>: CKKSModuleInfos,
{
    let (k, r) = (1usize << layout.log_k(), 1usize << layout.log_r());
    let slots = module.ckks_max_slots::<Rg>();
    ensure!(k * r <= slots, "{k} values of period {r} need more than the {slots} slots");
    Ok((k, k * r))
}

/// How the replicated bootstrapping evaluates `SlotsToCoeffs` and
/// `CoeffsToSlots`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReplicatedTransforms {
    /// Each map is one transform of `k` diagonals: `2·sqrt(k)` key switches
    /// and `k` diagonals over the `kR` slots of the period, one level each.
    Dense,
    /// Each map is the `k`-point DFT of the subring of degree `2k` in
    /// `s2c_factors` and `c2s_factors` factors of the DFT generator, and the
    /// move between the data slots and consecutive slots of `m` values in
    /// `placement_factors` factors of about `m^(1/placement_factors)` 0/1 mask
    /// diagonals. Every `SlotsToCoeffs` factor consumes
    /// `s2c_log_delta_diag` bits. `CoeffsToSlots` applies the DFT on the slots
    /// of period `k`, then the move onto the data slots, whose last factor
    /// alone is quantized with shaping and consumes `c2s_log_delta_diag` bits;
    /// the others consume `c2s_inner_log_delta_diag` bits, enough for their
    /// unshaped quantization error to stay under the output precision. The DFT
    /// factors leave the subring's coefficients in bit-reversed order, which
    /// the two maps undo together, and a packed plan must fit the ring.
    Factored {
        s2c_factors: usize,
        c2s_factors: usize,
        placement_factors: usize,
        c2s_inner_log_delta_diag: usize,
    },
}

/// Parameters of the bootstrapping of a [`ReplicatedLayout`] of `k` values,
/// `R | k`, in `SlotsToCoeffs`-first order.
#[derive(Clone, Copy, Debug)]
pub struct ReplicatedBootstrapPlan {
    pub layout: ReplicatedLayout,
    /// Scale of the bootstrapped message, `Δ_b`.
    pub log_delta: usize,
    /// Scale of the leveled ciphertexts: of the output, and of an input whose
    /// levels are spent.
    pub level_log_delta: usize,
    /// Scale of the `SlotsToCoeffs` diagonals, the bits it consumes.
    pub s2c_log_delta_diag: usize,
    /// Scale of the `CoeffsToSlots` diagonals.
    pub c2s_log_delta_diag: usize,
    /// Bits ModUp lifts the raised message by.
    pub mod_up_extra_bits: usize,
    /// Working bits of the linear maps, under the scale.
    pub extra_bits: usize,
    /// EvalMod, whose message ratio `q_b/Δ_b` is the coefficient message's.
    pub eval_mod: EvalModPlan,
    /// Evaluate EvalMod once on the real and imaginary parts side by side, when
    /// `2kR` slots fit in the ring.
    pub packed: bool,
    /// The evaluation of `SlotsToCoeffs` and `CoeffsToSlots`.
    pub transforms: ReplicatedTransforms,
}

impl ReplicatedBootstrapPlan {
    /// The parameters of ePrint 2026/2513 for `layout`: inputs at `2^30` with
    /// a ratio of `2^13`, outputs at `2^20`, the even cosine of degree 30 on
    /// `K = 16` with 3 double angles at the scale of
    /// [`Self::eval_mod_log_delta`] and constants at `2^26`, and the packed
    /// EvalMod.
    pub fn paper(layout: ReplicatedLayout) -> Self {
        Self {
            layout,
            log_delta: 30,
            level_log_delta: 20,
            s2c_log_delta_diag: 26,
            c2s_log_delta_diag: 34,
            mod_up_extra_bits: 6,
            extra_bits: 8,
            eval_mod: EvalModPlan {
                eval_mod_type: EvalModType::CosHKEven,
                log_msg_ratio: 13,
                f_mod_degree: 30,
                f_mod_interval: 16,
                f_mod_log_interval_reduction: 3,
                f_mod_inv_degree: None,
                scaling: None,
                split_strategy: SplitStrategy::MinDepth,
                coeffs_meta: CoeffsMeta::from_delta_budget(26, 4),
                f_mod_log_delta: Self::eval_mod_log_delta(&layout),
            },
            packed: true,
            transforms: ReplicatedTransforms::Dense,
        }
    }

    /// EvalMod scale of [`Self::paper`] for `layout`: `2^28`, or
    /// `ceil(guard_bits) + 15` bits for a unit whose guards grow by more than 13
    /// bits (`U4_15`, `U8_40`). Entering EvalMod rounds with shaping at this
    /// scale, which leaves noise of about `2^(guard_bits + log2(sqrt(N)))` units
    /// on the guard slots; the polynomial and the double angles square the
    /// guards, which must stay well below one.
    pub fn eval_mod_log_delta(layout: &ReplicatedLayout) -> usize {
        let guard_bits = crate::shaping::ShapingProfile::<Standard>::new(layout).guard_bits();
        28.max(guard_bits.ceil() as usize + 15)
    }

    /// The bits of budget `SlotsToCoeffs` consumes: its diagonal bits per factor.
    pub fn s2c_consumed_bits(&self) -> usize {
        match self.transforms {
            ReplicatedTransforms::Dense => self.s2c_log_delta_diag,
            ReplicatedTransforms::Factored {
                s2c_factors,
                placement_factors,
                ..
            } => (placement_factors + s2c_factors) * self.s2c_log_delta_diag,
        }
    }

    /// The bits of budget `CoeffsToSlots` consumes: the shaped last factor's
    /// diagonal bits and, factored, the inner factors'.
    pub fn c2s_consumed_bits(&self) -> usize {
        match self.transforms {
            ReplicatedTransforms::Dense => self.c2s_log_delta_diag,
            ReplicatedTransforms::Factored {
                c2s_factors,
                placement_factors,
                c2s_inner_log_delta_diag,
                ..
            } => (c2s_factors + placement_factors - 1) * c2s_inner_log_delta_diag + self.c2s_log_delta_diag,
        }
    }

    /// Width of the input at scale `Δ_b`: the message ratio and the bits
    /// `SlotsToCoeffs` consumes above it.
    pub fn k_in(&self) -> usize {
        self.log_delta + self.eval_mod.log_msg_ratio + self.s2c_consumed_bits()
    }

    /// Width after ModUp for an output of width `k_out`: `CoeffsToSlots`,
    /// EvalMod, and the rescale from `Δ_b`, lifted by the ModUp bits, to the
    /// leveled scale.
    pub fn k_boot(&self, k_out: usize) -> usize {
        k_out + self.c2s_consumed_bits() + self.eval_mod.consumed_bits() + self.log_delta + self.mod_up_extra_bits
            - self.level_log_delta
    }

    /// Width of the widest key switch: the raised modulus and the working bits
    /// of the top trace and `CoeffsToSlots`.
    pub fn key_k(&self, k_out: usize) -> usize {
        self.k_boot(k_out) + self.extra_bits
    }

    /// Key layouts for outputs of width `k_out`: the automorphism keys of
    /// `galois_elements` ([`ReplicatedBootstrapping::galois_elements`]) and the
    /// tensor key share one gadget, covering [`Self::key_k`] in digits of
    /// `k_digit` bits with an auxiliary modulus of `k_aux` bits; with
    /// `encapsulation`, the key to the ephemeral secret covers the ModUp input
    /// in one-limb digits and an auxiliary modulus of one limb and a bit. The
    /// key back covers the lifted ModUp input, with its rows offset to end at
    /// the raised modulus.
    #[allow(clippy::too_many_arguments)]
    pub fn key_layouts(
        &self,
        galois_elements: &[i64],
        n: Degree,
        base2k: Base2K,
        k_out: usize,
        k_digit: usize,
        k_aux: usize,
        encapsulation: bool,
    ) -> BootstrappingKeysLayout {
        let digit = TorusPrecision(k_digit as u32);
        let dnum = |k: usize, digit: TorusPrecision| GGLWELayout::dnum_for_input(TorusPrecision(k as u32), digit);
        let switching = |k: usize, digit: TorusPrecision, k_aux: usize| GLWESwitchingKeyLayout {
            n,
            base2k,
            dnum: dnum(k, digit),
            k_aux: TorusPrecision(k_aux as u32),
            rank_in: Rank(1),
            rank_out: Rank(1),
            k_digit: digit,
            k_offset: TorusPrecision(0),
        };
        let k_key = self.key_k(k_out);
        let one_limb = TorusPrecision(base2k.as_u32());
        let rotation_key = GLWEAutomorphismKeyLayout {
            n,
            base2k,
            dnum: dnum(k_key, digit),
            k_aux: TorusPrecision(k_aux as u32),
            rank: Rank(1),
            k_digit: digit,
        };
        BootstrappingKeysLayout {
            rotation_keys: galois_elements.iter().map(|&p| (p, rotation_key)).collect(),
            first_factor_key: None,
            conversion_keys: None,
            tensor_key: GLWETensorKeyLayout {
                n,
                base2k,
                dnum: dnum(k_key, digit),
                k_aux: TorusPrecision(k_aux as u32),
                rank: Rank(1),
                k_digit: digit,
            },
            encapsulation: encapsulation.then(|| {
                let k_small = self.log_delta + self.eval_mod.log_msg_ratio;
                let mut sparse_to_dense = switching(k_small + self.mod_up_extra_bits, digit, k_aux);
                let gadget = sparse_to_dense.dnum.0 as usize * k_digit;
                sparse_to_dense.k_offset = self.k_boot(k_out).saturating_sub(gadget).into();
                EncapsulationKeysLayout {
                    dense_to_sparse: switching(k_small, one_limb, base2k.as_usize() + 1),
                    sparse_to_dense: Some(sparse_to_dense),
                }
            }),
        }
    }

    /// [`Self::key_layouts`] with the fewest gadget digits within the modulus
    /// bounds: every key under the secret within `max_dense_modulus` bits and
    /// the key to the ephemeral secret within `max_sparse_modulus`, each with
    /// an auxiliary modulus of one digit,
    /// `log2(QP) = k_offset + (dnum + 1)·k_digit`.
    pub fn secure_key_layouts(
        &self,
        galois_elements: &[i64],
        n: Degree,
        base2k: Base2K,
        k_out: usize,
        max_dense_modulus: usize,
        max_sparse_modulus: usize,
    ) -> Result<BootstrappingKeysLayout> {
        let k_key = self.key_k(k_out);
        let k_q_b = self.log_delta + self.eval_mod.log_msg_ratio;
        let digit = |k: usize, max: usize| {
            (1..=k)
                .map(|dnum| k.div_ceil(dnum))
                .find(|&digit| k.div_ceil(digit) * digit + digit <= max)
                .with_context(|| format!("no gadget covers {k} bits within a modulus of {max} bits"))
        };
        let k_digit = digit(k_key, max_dense_modulus)?;
        let sparse_digit = digit(k_q_b, max_sparse_modulus)?;
        let raised = self.k_boot(k_out);
        let window = k_q_b + self.mod_up_extra_bits;
        let return_digit = (1..=window)
            .map(|dnum| window.div_ceil(dnum))
            .find(|&digit| (window.div_ceil(digit) * digit).max(raised) + digit <= max_dense_modulus)
            .context("no sparse-to-dense gadget fits the raised modulus and its auxiliary digit")?;
        let mut layouts = self.key_layouts(galois_elements, n, base2k, k_out, k_digit, k_digit, true);
        if let Some(encapsulation) = layouts.encapsulation.as_mut() {
            let sparse = TorusPrecision(sparse_digit as u32);
            encapsulation.dense_to_sparse.k_digit = sparse;
            encapsulation.dense_to_sparse.k_aux = sparse;
            encapsulation.dense_to_sparse.dnum = GGLWELayout::dnum_for_input(TorusPrecision(k_q_b as u32), sparse);
            let sparse_to_dense = encapsulation.sparse_to_dense.as_mut().unwrap();
            sparse_to_dense.k_digit = return_digit.into();
            sparse_to_dense.k_aux = return_digit.into();
            sparse_to_dense.dnum = GGLWELayout::dnum_for_input(window.into(), return_digit.into());
            sparse_to_dense.k_offset = raised.saturating_sub(sparse_to_dense.dnum.0 as usize * return_digit).into();
        }
        Ok(layouts)
    }
}

/// Compiled [`ReplicatedBootstrapPlan`]: the `SlotsToCoeffs` gather onto the
/// `k`-point embedding `V`, `V[c][j]` slot `c` of `Y'^j` with
/// `Y' = X^(N/(2k))`, which writes `z_j = m_j + i·m_(j+k)` as the coefficients
/// `m` of the subring of degree `2k`; the `CoeffsToSlots` map
/// `A = V^(-1)/(2K) = V^H/(2Kk)` onto the data slots (`[A; −iA]` on two batches
/// when packed), `K` the EvalMod interval; the trace rotations; and EvalMod
/// with masked constants.
pub struct ReplicatedBootstrapping<BE: Backend, F> {
    pub(crate) plan: ReplicatedBootstrapPlan,
    pub(crate) s2c: ReplicatedTransform<BE>,
    pub(crate) c2s: ReplicatedTransform<BE>,
    /// Skips of the traces onto periods `kR` and `k`.
    pub(crate) bottom_skip: usize,
    pub(crate) top_skip: usize,
    /// Sparsity of the subring of period `k`.
    pub(crate) top_sparsity: usize,
    /// The layout of the two batches of a packed EvalMod.
    pub(crate) packed: Option<ReplicatedLayout>,
    pub(crate) eval_mod: EvalMod<F, MaskedCoeffs<CKKSPlaintextOwned<BE>>>,
}

/// The layout of the two batches EvalMod runs on side by side, when the plan packs them and they fit.
fn packed_layout(plan: &ReplicatedBootstrapPlan, max_slots: usize) -> Result<Option<ReplicatedLayout>> {
    let layout = plan.layout;
    let (k, r) = (1usize << layout.log_k(), 1usize << layout.log_r());
    Ok(if plan.packed && 4 * k * r <= 2 * max_slots {
        Some(ReplicatedLayout::new(
            layout.log_k() as u8 + 1,
            layout.log_r() as u8,
            layout.shaping(),
        )?)
    } else {
        None
    })
}

impl<BE: Backend, F: CKKSEncodingScalar> ReplicatedBootstrapping<BE, F> {
    /// Scratch of [`Self::new`]: the host encodings of the embedding and of the EvalMod constants, and
    /// the transforms of both conversions.
    ///
    /// # Errors
    /// When the plan's layout needs more slots than the module has.
    pub fn new_tmp_bytes(module: &Module<BE>, plan: &ReplicatedBootstrapPlan, base2k: Base2K) -> Result<usize>
    where
        Module<BE>: ReplicatedTransformModule<BE, F>,
    {
        let c2s_layout = packed_layout(plan, module.ckks_max_slots::<Standard>())?.unwrap_or(plan.layout);
        let c2s_log_delta_diag = match plan.transforms {
            ReplicatedTransforms::Dense => plan.c2s_log_delta_diag,
            ReplicatedTransforms::Factored {
                c2s_inner_log_delta_diag,
                ..
            } => plan.c2s_log_delta_diag.max(c2s_inner_log_delta_diag),
        };
        Ok(module
            .ckks_encoding_host_tmp_bytes(module.ckks_max_slots::<Standard>())
            .max(ReplicatedTransform::<BE>::encode_tmp_bytes::<F>(
                module,
                &plan.layout,
                base2k,
                plan.s2c_log_delta_diag,
            )?)
            .max(ReplicatedTransform::<BE>::encode_tmp_bytes::<F>(
                module,
                &c2s_layout,
                base2k,
                c2s_log_delta_diag,
            )?))
    }

    /// Compiles `plan` on `module`, within the caller's `scratch` of [`Self::new_tmp_bytes`].
    pub fn new(
        module: &Module<BE>,
        plan: ReplicatedBootstrapPlan,
        base2k: Base2K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<Self>
    where
        Module<BE>: ReplicatedTransformModule<BE, F>,
    {
        let layout = plan.layout;
        check_period::<BE, Standard>(module, &layout)?;
        let (k, r) = (1usize << layout.log_k(), 1usize << layout.log_r());
        ensure!(r <= k, "bootstrapping needs the period {r} to divide the batch size {k}");
        let half = module.ckks_max_slots::<Standard>();
        let packed = packed_layout(&plan, half)?;
        let (s2c, c2s) = match plan.transforms {
            ReplicatedTransforms::Dense => {
                let v = embedding::<BE, F>(module, k, base2k, scratch)?;
                let s2c = ReplicatedTransform::gather(
                    module,
                    layout,
                    base2k,
                    plan.s2c_log_delta_diag,
                    false,
                    |c, j| v[c][j],
                    scratch,
                )?;
                let scale = F::from_usize(2 * k * plan.eval_mod.f_mod_interval)
                    .context("the CoeffsToSlots normalization is not representable")?
                    .recip();
                // `A[t][c] = conj(V[c][t])/(2Kk)`, written on the data slots.
                let a = |t: usize, c: usize| {
                    let (re, im) = v[c][t];
                    (re * scale, -im * scale)
                };
                let c2s = match packed {
                    // `−i·A` on the second batch.
                    Some(double) => ReplicatedTransform::from_input_period(
                        module,
                        double,
                        k,
                        base2k,
                        plan.c2s_log_delta_diag,
                        true,
                        |s, c| match (s % r, s / r) {
                            (0, t) if t < k => a(t, c),
                            (0, t) => {
                                let (re, im) = a(t - k, c);
                                (im, -re)
                            }
                            _ => (F::zero(), F::zero()),
                        },
                        scratch,
                    )?,
                    None => ReplicatedTransform::from_period(
                        module,
                        layout,
                        base2k,
                        plan.c2s_log_delta_diag,
                        true,
                        |s, c| {
                            if s.is_multiple_of(r) {
                                a(s / r, c)
                            } else {
                                (F::zero(), F::zero())
                            }
                        },
                        scratch,
                    )?,
                };
                (s2c, c2s)
            }
            ReplicatedTransforms::Factored { .. } => {
                ensure!(
                    packed.is_some() == plan.packed,
                    "a factored packed plan needs the two batches to fit the ring"
                );
                factored_transforms::<BE, F>(module, &plan, packed, base2k, scratch)?
            }
        };
        let eval_mod = match packed {
            Some(double) => {
                ckks_eval_mod_replicated_packed_compile::<BE, F, Standard>(base2k, plan.eval_mod, double, module, scratch)?
            }
            None => ckks_eval_mod_replicated_compile::<BE, F, Standard>(base2k, plan.eval_mod, layout, module, scratch)?,
        };
        Ok(Self {
            plan,
            s2c,
            c2s,
            bottom_skip: (k * r).trailing_zeros() as usize + 1,
            top_skip: k.trailing_zeros() as usize + 1,
            top_sparsity: (half / k).trailing_zeros() as usize,
            packed,
            eval_mod,
        })
    }
}

/// Reusable prepared baby steps of a replicated bootstrap, sized for its
/// output width. Concurrent evaluations use separate workspaces.
pub struct ReplicatedBootstrapWorkspace<BE: Backend> {
    pub(crate) s2c: ReplicatedTransformWorkspace<BE>,
    pub(crate) c2s: ReplicatedTransformWorkspace<BE>,
}

impl<BE: Backend, F> ReplicatedBootstrapping<BE, F> {
    /// Allocates reusable transform storage for bootstraps into `dst`.
    pub fn alloc_workspace<C: GLWEInfos>(&self, module: &Module<BE>, dst: &C) -> ReplicatedBootstrapWorkspace<BE>
    where
        Module<BE>: CnvPVecAlloc<BE>,
    {
        let at = |k: usize| GLWELayout {
            k: k.into(),
            ..dst.glwe_layout()
        };
        ReplicatedBootstrapWorkspace {
            s2c: self.s2c.alloc_workspace(module, &at(self.plan.k_in() + self.plan.extra_bits)),
            c2s: self
                .c2s
                .alloc_workspace(module, &at(self.plan.k_boot(dst.k().as_usize()) + self.plan.extra_bits)),
        }
    }

    pub fn plan(&self) -> &ReplicatedBootstrapPlan {
        &self.plan
    }

    /// Whether EvalMod runs once on the two parts packed side by side.
    pub fn is_packed(&self) -> bool {
        self.packed.is_some()
    }

    /// Galois elements of the automorphism keys the bootstrapping uses,
    /// conjugation (`−1`) among them.
    pub fn galois_elements(&self, module: &Module<BE>) -> Vec<i64>
    where
        Module<BE>: CKKSModuleInfos + GLWETrace<BE>,
    {
        let k = 1i64 << self.plan.layout.log_k();
        let r = 1i64 << self.plan.layout.log_r();
        let mut elements: Vec<i64> = self
            .s2c
            .rotations()
            .into_iter()
            .chain(self.c2s.rotations())
            .chain(self.packed.map(|_| k * r))
            .map(|o| module.ckks_galois_element::<Standard>(o))
            .chain(module.glwe_trace_galois_elements(module.n().into(), 0..module.log_n(), false))
            .chain([-1])
            .collect();
        elements.sort_unstable();
        elements.dedup();
        elements
    }

    /// Scratch of [`Self::generate_keys`] for `layout`: the encryption of each key.
    pub fn generate_keys_tmp_bytes(&self, module: &Module<BE>, layout: &BootstrappingKeysLayout) -> usize
    where
        Module<BE>: GLWEAutomorphismKeyEncryptSk<BE> + GLWETensorKeyEncryptSk<BE> + GLWESwitchingKeyEncryptSk<BE>,
    {
        let mut bytes = module.glwe_tensor_key_encrypt_sk_tmp_bytes(&layout.tensor_key);
        for key in layout.rotation_key_layouts() {
            bytes = bytes.max(module.glwe_automorphism_key_encrypt_sk_tmp_bytes(&key));
        }
        if let Some(encapsulation) = &layout.encapsulation {
            for key in [Some(encapsulation.dense_to_sparse), encapsulation.sparse_to_dense]
                .into_iter()
                .flatten()
            {
                bytes = bytes.max(module.glwe_switching_key_encrypt_sk_tmp_bytes(&key));
            }
        }
        bytes
    }

    /// Generates the keys of the bootstrapping under `sk`: the automorphism
    /// keys of [`Self::galois_elements`], the tensor key, and with
    /// `layout.encapsulation` the keys to and from an ephemeral secret of
    /// Hamming weight `hamming_weight` sampled from `source_xs`, within `scratch` of
    /// [`Self::generate_keys_tmp_bytes`].
    ///
    /// # Panics
    ///
    /// If `layout.rotation_keys` has no layout for one of [`Self::galois_elements`].
    #[allow(clippy::too_many_arguments)]
    pub fn generate_keys(
        &self,
        module: &Module<BE>,
        sk: &BackendGLWESecret<BE>,
        layout: &BootstrappingKeysLayout,
        hamming_weight: usize,
        source_xs: &mut Source,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> BootstrappingKeySet<BE::OwnedBuf, BE::ZnxWord>
    where
        Module<BE>: CKKSModuleInfos
            + GLWETrace<BE>
            + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = BE::ZnxWord>
            + GLWEAutomorphismKeyEncryptSk<BE>
            + GLWETensorKeyEncryptSk<BE>
            + GLWESwitchingKeyEncryptSk<BE>
            + GLWESecretSampling<BE>,
    {
        assert!(
            layout.first_factor_key.is_none(),
            "replicated bootstrapping requires a separate first factor"
        );
        assert!(
            layout.encapsulation.is_none_or(|keys| keys.sparse_to_dense.is_some()),
            "replicated bootstrapping requires a sparse-to-dense key for encapsulation"
        );
        let mut rotation_keys = HashMap::new();
        for gal in self.galois_elements(module) {
            let key_layout = layout
                .rotation_keys
                .get(&gal)
                .unwrap_or_else(|| panic!("the key layouts have no rotation key for Galois element {gal}"));
            let mut key = module.glwe_automorphism_key_alloc_from_infos(key_layout);
            module.glwe_automorphism_key_encrypt_sk(&mut key, gal, sk, source_xe, source_xa, scratch);
            rotation_keys.insert(gal, key);
        }
        let mut tensor_key = module.glwe_tensor_key_alloc_from_infos(&layout.tensor_key);
        module.glwe_tensor_key_encrypt_sk(&mut tensor_key, sk, source_xe, source_xa, scratch);
        let encapsulation_keys = layout.encapsulation.map(|encapsulation| {
            let mut sparse = module.glwe_secret_alloc_from_infos(&GLWESecretLayout {
                n: sk.n(),
                rank: sk.rank(),
            });
            module.glwe_secret_fill_ternary_hw(&mut sparse, hamming_weight, source_xs);
            *sparse.dist_mut() = Distribution::ENCAPSULATED("sparse-encapsulation");
            let mut dense_to_sparse = module.glwe_switching_key_alloc_from_infos(&encapsulation.dense_to_sparse);
            module.glwe_switching_key_encrypt_sk(&mut dense_to_sparse, sk, &sparse, source_xe, source_xa, scratch);
            let mut sparse_to_dense = module.glwe_switching_key_alloc_from_infos(&encapsulation.sparse_to_dense.unwrap());
            module.glwe_switching_key_encrypt_sk(&mut sparse_to_dense, &sparse, sk, source_xe, source_xa, scratch);
            (dense_to_sparse, Some(sparse_to_dense))
        });
        // The replicated EvalMod runs on the standard ring.
        BootstrappingKeySet {
            rotation_keys,
            tensor_key: Some(tensor_key),
            ci_tensor_key: None,
            encapsulation_keys,
            first_factor_keys: HashMap::new(),
            conversion_keys: None,
        }
    }
}

/// `V[c][j]`, slot `c` of the monomial `Y'^j` of the subring of degree `2k`,
/// read from the encoder.
fn embedding<BE: Backend, F>(
    module: &Module<BE>,
    k: usize,
    base2k: Base2K,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<Vec<Vec<(F, F)>>>
where
    F: CKKSEncodingScalar,
    Module<BE>: ReplicatedTransformModule<BE, F>,
{
    let log_delta = 40;
    let mut columns = Vec::with_capacity(k);
    for j in 0..k {
        let mut pt = module.ckks_plaintext_alloc_from_slots(k, base2k, TorusPrecision((log_delta + 2) as u32));
        pt.set_meta(GLWEPlaintextMeta {
            scale: Scale::Log(log_delta),
            slots: SlotsKind::Complex,
            log_sparsity: 0,
            slot_layout: SlotLayout::Full,
        });
        let mut coeffs = vec![F::zero(); 2 * k];
        coeffs[j] = F::one();
        module.ckks_encode_coeffs_host_into(&mut pt, &coeffs, &mut scratch.borrow())?;
        let (mut re, mut im) = (vec![F::zero(); k], vec![F::zero(); k]);
        module.ckks_decode_slots_host_into(&pt, &mut re, &mut im, &mut scratch.borrow())?;
        columns.push(re.into_iter().zip(im).collect::<Vec<_>>());
    }
    // `Y'^k` is `i` on every slot, so `m_j + i·m_(j+k)` reads `V[c][j]`.
    Ok((0..k).map(|c| (0..k).map(|j| columns[j][c]).collect()).collect())
}

/// The factored `SlotsToCoeffs` and `CoeffsToSlots` of
/// [`ReplicatedTransforms::Factored`]:
///
/// - `SlotsToCoeffs` moves the data slots onto the first `k` slots of each period,
///   weighted `R` for the normalized trace onto period `k` that completes the map
///   after ModUp, then applies `V·BR`, the generator's `Decode`, on them;
/// - `CoeffsToSlots` applies `BR·V^H/(2Kk)`, the generator's `Encode`, on the
///   slots of period `k`, then moves them onto the data slots of the `m` values
///   of `target` (`m = 2k` when packed, the second batch times `-i`); its last
///   factor is quantized with the shaping of `target`.
fn factored_transforms<BE, F>(
    module: &Module<BE>,
    plan: &ReplicatedBootstrapPlan,
    packed: Option<ReplicatedLayout>,
    base2k: Base2K,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<(ReplicatedTransform<BE>, ReplicatedTransform<BE>)>
where
    BE: Backend,
    F: CKKSEncodingScalar,
    Module<BE>: ReplicatedTransformModule<BE, F>,
{
    let ReplicatedTransforms::Factored {
        s2c_factors,
        c2s_factors,
        placement_factors,
        c2s_inner_log_delta_diag,
    } = plan.transforms
    else {
        anyhow::bail!("the plan's transforms are not factored");
    };
    let layout = plan.layout;
    let (log_k, log_r) = (layout.log_k(), layout.log_r());
    let (k, r) = (1usize << log_k, 1usize << log_r);
    ensure!(
        [s2c_factors, c2s_factors, placement_factors]
            .iter()
            .all(|f| (1..=log_k).contains(f)),
        "factored transforms take between one and log2(k) = {log_k} factors for each DFT and placement"
    );
    let unshaped = |log_delta_diag| {
        move |diagonals| ReplicatedFactor {
            diagonals,
            log_delta_diag,
            shaped: false,
            fitted_giant_step: true,
        }
    };
    let weight = F::from_usize(r).context("the period is a small integer")?;
    let s2c: Vec<ReplicatedFactor<F>> = placement_compress::<F>(k, r, placement_factors, weight)
        .into_iter()
        .chain(ckks_dft_matrices_gen_blockwise::<F>(
            DFTType::Decode,
            log_k,
            log_k + log_r,
            &split_depth(log_k, s2c_factors),
            1.0,
            false,
        ))
        .map(unshaped(plan.s2c_log_delta_diag))
        .collect();

    let (target, m) = match packed {
        Some(double) => (double, 2 * k),
        None => (layout, k),
    };
    let interval = plan.eval_mod.f_mod_interval as f64;
    let mut c2s: Vec<ReplicatedFactor<F>> = ckks_dft_matrices_gen_blockwise::<F>(
        DFTType::Encode,
        log_k,
        log_k,
        &split_depth(log_k, c2s_factors),
        1.0 / (2.0 * interval),
        false,
    )
    .into_iter()
    .chain(placement_spread::<F>(m, r, placement_factors))
    .map(unshaped(c2s_inner_log_delta_diag))
    .collect();
    let last = c2s.last_mut().context("CoeffsToSlots has factors")?;
    if m > k {
        // `-i` on the second batch.
        times_minus_i_from(&mut last.diagonals, r * k);
    }
    last.log_delta_diag = plan.c2s_log_delta_diag;
    last.shaped = true;
    Ok((
        ReplicatedTransform::from_factors(module, layout, base2k, s2c, scratch)?,
        ReplicatedTransform::from_factors(module, target, base2k, c2s, scratch)?,
    ))
}

/// Multiplies the outputs of slots `from..` of `diagonals` by `-i`:
/// `(re, im) -> (im, -re)`.
fn times_minus_i_from<F: CKKSEncodingScalar>(diagonals: &mut ComplexDiagonals<F>, from: usize) {
    let n = diagonals.slots();
    let full = |values: Option<&Vec<F>>| -> Vec<F> {
        values.map_or_else(|| vec![F::zero(); n], |v| (0..n).map(|s| v[s % v.len()]).collect())
    };
    for o in diagonals.indexes() {
        let (mut re, mut im) = (full(diagonals.re.get(o)), full(diagonals.im.get(o)));
        for s in from..n {
            (re[s], im[s]) = (im[s], -re[s]);
        }
        set_complex(diagonals, o, (re, im));
    }
}

/// The diagonals of `n` slots given by their full values, `(offset, values)`.
fn diagonals_from_weights<F: CKKSEncodingScalar>(n: usize, weights: Vec<(usize, Vec<(F, F)>)>) -> ComplexDiagonals<F> {
    let mut diagonals = ComplexDiagonals::new(Diagonals::new(n), Diagonals::new(n));
    for (o, w) in weights {
        let (re, im): (Vec<F>, Vec<F>) = w.into_iter().unzip();
        set_complex(&mut diagonals, o as i64, (re, im));
    }
    diagonals
}

/// Bits of `t` selected by `group`, kept in place.
fn group_bits(t: usize, group: &[usize]) -> usize {
    group.iter().map(|&b| t & (1 << b)).sum()
}

/// The bits of a `2^log_m`-value index in `factors` near-equal groups, from the highest.
fn bit_groups(log_m: usize, factors: usize) -> Vec<Vec<usize>> {
    let mut bits = (0..log_m).rev();
    split_depth(log_m, factors)
        .into_iter()
        .map(|size| bits.by_ref().take(size).collect())
        .collect()
}

/// Where value `t` sits once the bits of `moved` are strided: `t + (R - 1)·t_moved`.
fn placement_position(t: usize, moved: &[usize], r: usize) -> usize {
    t + (r - 1) * group_bits(t, moved)
}

/// A diagonal of `n` slots, `value` at the positions of `on` and zero elsewhere.
fn mask_diagonal<F: CKKSEncodingScalar>(n: usize, on: impl IntoIterator<Item = usize>, value: F) -> (Vec<F>, Vec<F>) {
    let mut re = vec![F::zero(); n];
    for s in on {
        re[s] = value;
    }
    (re, vec![F::zero()])
}

fn set_complex<F>(diagonals: &mut ComplexDiagonals<F>, index: i64, (re, im): (Vec<F>, Vec<F>)) {
    diagonals.re.set(index, re);
    diagonals.im.set(index, im);
}

/// The placement of `m` values from slots of period `k` onto the data slots of
/// stride `R`, `y_(R·t) = x_(t mod k)` for `t < m` and zero elsewhere, over the
/// `n = m·R` slots. The `log2(m)` bits of `t` are split into `factors`
/// near-equal groups, and one factor per group, from the highest, moves the
/// values whose group bits are `v` by `(R - 1)·v`. Taking the high bits first
/// keeps the positions `t + (R - 1)·t_moved` distinct, so every factor is a
/// set of 0/1 masks, `2^g` diagonals for a group of `g` bits.
pub(crate) fn placement_spread<F: CKKSEncodingScalar>(m: usize, r: usize, factors: usize) -> Vec<ComplexDiagonals<F>> {
    let n = m * r;
    let mut moved = Vec::new();
    let mut out = Vec::new();
    for group in bit_groups(m.trailing_zeros() as usize, factors) {
        let next: Vec<usize> = moved.iter().chain(&group).copied().collect();
        let mut factor = ComplexDiagonals::new(Diagonals::new(n), Diagonals::new(n));
        for v in (0..m)
            .map(|t| group_bits(t, &group))
            .collect::<std::collections::BTreeSet<_>>()
        {
            let on = (0..m)
                .filter(|&t| group_bits(t, &group) == v)
                .map(|t| placement_position(t, &next, r));
            set_complex(&mut factor, -(((r - 1) * v) as i64), mask_diagonal(n, on, F::one()));
        }
        out.push(factor);
        moved = next;
    }
    out
}

/// The inverse of [`placement_spread`] for `m = k`: `y_t = weight·x_(R·t)` for
/// `t < k` and zero elsewhere, over the `n = k·R` slots, reading only the data
/// slots `R | s`. The groups are undone from the lowest, the last factor
/// carrying `weight`.
pub(crate) fn placement_compress<F: CKKSEncodingScalar>(
    k: usize,
    r: usize,
    factors: usize,
    weight: F,
) -> Vec<ComplexDiagonals<F>> {
    let n = k * r;
    let groups = bit_groups(k.trailing_zeros() as usize, factors);
    let mut moved: Vec<usize> = groups.iter().flatten().copied().collect();
    let mut out = Vec::new();
    for (i, group) in groups.iter().rev().enumerate() {
        moved.retain(|b| !group.contains(b));
        let value = if i + 1 == groups.len() { weight } else { F::one() };
        let mut factor = ComplexDiagonals::new(Diagonals::new(n), Diagonals::new(n));
        for v in (0..k)
            .map(|t| group_bits(t, group))
            .collect::<std::collections::BTreeSet<_>>()
        {
            let on = (0..k)
                .filter(|&t| group_bits(t, group) == v)
                .map(|t| placement_position(t, &moved, r));
            set_complex(&mut factor, ((r - 1) * v) as i64, mask_diagonal(n, on, value));
        }
        out.push(factor);
    }
    out
}

/// `log_m` bits in `factors` near-equal groups, the larger first.
pub(crate) fn split_depth(log_m: usize, factors: usize) -> Vec<usize> {
    let factors = factors.clamp(1, log_m.max(1));
    (0..factors)
        .map(|i| log_m / factors + usize::from(i < log_m % factors))
        .collect()
}

#[cfg(test)]
mod tests {
    use poulpy_core::layouts::{Evaluate, LinearTransformationStrategy};

    use super::*;

    fn apply(factors: &[ComplexDiagonals<f64>], x: &[f64]) -> Vec<f64> {
        let mut re = x.to_vec();
        let mut im = vec![0.0; x.len()];
        for factor in factors {
            (re, im) = factor.evaluate((&re, &im), LinearTransformationStrategy::Direct);
        }
        assert!(im.iter().all(|&v| v == 0.0));
        re
    }

    fn pseudo_random(n: usize) -> Vec<f64> {
        (0..n).map(|i| ((i * 2654435761) % 1000) as f64 / 7.0 + 1.0).collect()
    }

    #[test]
    fn placement_spread_writes_the_data_slots() {
        for (k, m, r, factors) in [
            (8, 8, 4, 3),
            (16, 16, 8, 2),
            (64, 64, 16, 2),
            (64, 64, 2, 1),
            (32, 64, 16, 3),
            (16, 32, 32, 5),
        ] {
            let x = pseudo_random(k);
            let input: Vec<f64> = (0..m * r).map(|s| x[s % k]).collect();
            let log_group = (m.trailing_zeros() as usize).div_ceil(factors);
            let factors = placement_spread::<f64>(m, r, factors);
            assert!(factors.iter().all(|f| f.indexes().len() <= 1 << log_group));
            let want: Vec<f64> = (0..m * r).map(|s| if s % r == 0 { x[(s / r) % k] } else { 0.0 }).collect();
            assert_eq!(apply(&factors, &input), want, "k = {k}, m = {m}, R = {r}");
        }
    }

    #[test]
    fn placement_compress_reads_only_the_data_slots() {
        for (k, r, factors) in [(8, 4, 3), (16, 8, 2), (64, 16, 2), (32, 32, 1)] {
            let x = pseudo_random(k);
            // The guards hold garbage the compression must not read.
            let input: Vec<f64> = (0..k * r)
                .map(|s| if s % r == 0 { x[s / r] } else { 1e6 + s as f64 })
                .collect();
            let factors = placement_compress::<f64>(k, r, factors, r as f64);
            let want: Vec<f64> = (0..k * r).map(|s| if s < k { r as f64 * x[s] } else { 0.0 }).collect();
            assert_eq!(apply(&factors, &input), want, "k = {k}, R = {r}");
        }
    }

    #[test]
    fn paper_eval_mod_scale_covers_the_guards() {
        use crate::shaping::ShapingUnit;
        for (unit, log_delta) in [
            (ShapingUnit::U4_10, 28),
            (ShapingUnit::U4_15, 31),
            (ShapingUnit::U8_30, 28),
            (ShapingUnit::U8_40, 30),
            (ShapingUnit::U16_60, 28),
            (ShapingUnit::U32_45, 28),
        ] {
            let plan = ReplicatedBootstrapPlan::paper(unit.layout(4).unwrap());
            assert_eq!(plan.eval_mod.f_mod_log_delta, log_delta, "{unit:?}");
        }
    }

    #[test]
    fn split_depth_covers_the_bits() {
        assert_eq!(split_depth(10, 3), [4, 3, 3]);
        assert_eq!(split_depth(10, 1), [10]);
        assert_eq!(split_depth(2, 5), [1, 1]);
    }
}
