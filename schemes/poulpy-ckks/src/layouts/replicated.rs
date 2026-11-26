//! Linear maps, conversions and bootstrapping of a replicated slot layout, the
//! small-batch scheme of [ePrint 2026/2513](https://eprint.iacr.org/2026/2513)
//! (see `docs/guides/replicated.md`).
use anyhow::{Context, Result, ensure};
use poulpy_core::layouts::{
    Base2K, Degree, Diagonals, GLWEInfos, GLWELayout, GetDegree, LinearTransformation, LinearTransformationBabySteps,
    LinearTransformationLayout, LinearTransformationPrepared, LinearTransformationStrategy, Rank, TorusPrecision, galois_cmp,
    galois_is_identity, galois_of_rotation, galois_rotation, optimal_bsgs_giant_step,
};
use poulpy_hal::{
    api::CnvPVecAlloc,
    layouts::{Backend, Module, Ring, ScratchArena, Standard},
};

use crate::{
    CKKSLayout, CKKSModuleInfos, GLWEPlaintextMeta, ReplicatedLayout, Scale, SetCKKSInfos, SetGLWEPlaintextInfos, SlotLayout,
    SlotsKind,
    api::{CKKSEncodingHostOps, CKKSEncodingOps, CKKSEncodingScalar, CKKSLinearTransformationOps},
    encoding::linear_transformation::period_degree,
    layouts::{CKKSCiphertextAlloc, CKKSLinearTransformationAlloc, CKKSPlaintextAlloc, CKKSPlaintextOwned, ComplexDiagonals},
    oep::derived::replicated::working_layout,
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
    /// The degree of the ciphertexts the conversions take.
    pub(crate) n: Degree,
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

    /// Encodes and prepares the conversion maps on `module` for ciphertexts of degree `n`, within the caller's
    /// `scratch` of [`Self::new_tmp_bytes`].
    ///
    /// # Errors
    /// When `n` exceeds the degree of `module` or `layout` needs more slots than the ring of degree `n` has.
    #[expect(clippy::too_many_arguments, reason = "the maps, their degree, layout, scales and arena")]
    pub fn new<F>(
        module: &Module<BE>,
        n: Degree,
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
        let (k, _) = check_period_at::<BE, Rg>(module, n, &layout)?;
        let r = 1usize << layout.log_r();
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
        let h = module.ckks_slots_of_degree::<Rg>(n.as_usize()) / k;
        let trace = (0..h.trailing_zeros()).map(|l| (k << l) as i64).collect();
        Ok(Self {
            n,
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
    #[expect(clippy::too_many_arguments, reason = "the maps, their degree, layout, scales and arena")]
    pub fn new_factored<F>(
        module: &Module<BE>,
        n: Degree,
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
        let (k, _) = check_period_at::<BE, Rg>(module, n, &layout)?;
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
        let h = module.ckks_slots_of_degree::<Rg>(n.as_usize()) / k;
        Ok(Self {
            n,
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

    /// The degree of the ciphertexts the conversions take.
    pub fn n(&self) -> Degree {
        self.n
    }

    /// Allocates reusable baby steps for conversion into this layout from `a`.
    pub fn alloc_to_workspace<C: GLWEInfos>(&self, module: &Module<BE>, a: &C) -> ReplicatedTransformWorkspace<BE, Rg>
    where
        Module<BE>: CnvPVecAlloc<BE>,
    {
        self.to.alloc_workspace(module, &working_layout(a, self.extra_bits, 0))
    }

    /// Allocates reusable baby steps for conversion from this layout of `a`.
    pub fn alloc_from_workspace<C: GLWEInfos>(&self, module: &Module<BE>, a: &C) -> Result<ReplicatedTransformWorkspace<BE, Rg>>
    where
        Module<BE>: CnvPVecAlloc<BE>,
    {
        let from = self
            .from
            .as_ref()
            .context("the gather needs the period to divide the batch size")?;
        Ok(from.alloc_workspace(module, &working_layout(a, self.extra_bits, 0)))
    }

    /// Galois elements of the automorphisms the conversions use, for their keys, by increasing rotation.
    pub fn galois(&self) -> Vec<i64> {
        let mut galois = self.to.galois();
        galois.extend(self.from.iter().flat_map(ReplicatedTransform::galois));
        galois.extend(self.trace.iter().map(|&rotation| galois_of_rotation(rotation)));
        galois.sort_by(galois_cmp);
        galois.dedup();
        galois
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
            .map(|(layout, (lt, _))| LinearTransformationBabySteps::alloc(module, lt.baby_galois(), &layout))
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
        let factor = period_factor::<BE, F, A, Rg>(module, layout, m, log_delta_diag, shaped, a)?;
        Self::from_factors(module, layout, base2k, vec![factor], scratch)
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
        let factor = gather_factor::<BE, F, A, Rg>(module, layout, log_delta_diag, shaped, a)?;
        Self::from_factors(module, layout, base2k, vec![factor], scratch)
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

    /// Galois elements of the non-identity steps of the transform, for their automorphism keys, by increasing
    /// rotation.
    pub fn galois(&self) -> Vec<i64> {
        let lts = || self.factors.iter().map(|(lt, _)| lt);
        let babies = lts().flat_map(|lt| {
            lt.giant_steps
                .iter()
                .flat_map(|gs| gs.diagonals.iter().map(|d| d.baby_galois))
        });
        let giants = lts().flat_map(|lt| lt.giant_steps.iter().map(|gs| gs.galois));
        let mut galois: Vec<i64> = babies.chain(giants).filter(|&p| !galois_is_identity(p)).collect();
        galois.sort_by(galois_cmp);
        galois.dedup();
        galois
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
        let lt = Self::encode_factor_unprepared(module, layout, base2k, factor, scratch)?;
        let first = lt.first_diagonal_plaintext().context("a transform has a diagonal")?;
        let mut prepared = LinearTransformationPrepared::<BE, Rg>::alloc_prepared_from_index(module, &lt.index(), first);
        module.ckks_linear_transformation_rhs_prepare(&mut prepared, &lt, scratch)?;
        Ok(prepared)
    }

    /// Encodes `factor` for a ciphertext of `layout`'s period, its diagonals at the degree of the period, without
    /// preparing it.
    pub(crate) fn encode_factor_unprepared<F>(
        module: &Module<BE>,
        layout: ReplicatedLayout,
        base2k: Base2K,
        factor: ReplicatedFactor<F>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<LinearTransformation<CKKSPlaintextOwned<BE, Rg>>>
    where
        F: CKKSEncodingScalar,
        Module<BE>: ReplicatedEncodeModule<BE, F, Rg>,
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
                    .galois()
                    .into_iter()
                    .all(|o| diagonals.im.get(o).is_none_or(|im| im.iter().all(|v| v.is_zero()))),
            "the real slots of the conjugate-invariant ring take real diagonals"
        );
        let largest = diagonals
            .galois()
            .into_iter()
            .flat_map(values)
            .map(|v| v.abs())
            .fold(F::zero(), F::max);
        // Bits above the scale for the largest weight, plus a sign and a carry.
        let headroom = (usize::BITS - (largest.to_f64().unwrap_or(1.0).ceil() as usize).max(1).leading_zeros()) as usize + 2;
        debug_assert!(headroom <= MAX_DIAGONAL_HEADROOM);
        let count = diagonals.galois().len();
        let strategy = if count <= 2 {
            LinearTransformationStrategy::Direct
        } else if fitted_giant_step {
            LinearTransformationStrategy::Bsgs {
                giant_step: optimal_bsgs_giant_step(diagonals.galois(), n),
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
            galois: diagonals.galois(),
            slots: n,
            strategy,
        };
        let mut lt = module.ckks_linear_transformation_alloc_from_infos::<_, Rg>(&lt_layout, &infos);
        let pt_slots = module.ckks_slots_of_degree::<Rg>(infos.glwe_layout.n.as_usize());
        let mut targets = lt.giant_steps.iter_mut().flat_map(|gs| {
            let rot = galois_rotation(gs.galois, n);
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
        Ok(lt)
    }
}

/// The one factor of [`ReplicatedTransform::from_input_period`].
pub(crate) fn period_factor<BE, F, A, Rg: Ring>(
    module: &Module<BE>,
    layout: ReplicatedLayout,
    m: usize,
    log_delta_diag: usize,
    shaped: bool,
    a: A,
) -> Result<ReplicatedFactor<F>>
where
    BE: Backend,
    F: CKKSEncodingScalar,
    A: Fn(usize, usize) -> (F, F),
    Module<BE>: CKKSModuleInfos,
{
    let (_, n) = check_period::<BE, Rg>(module, &layout)?;
    ensure!(n.is_multiple_of(m), "the input period {m} must divide the {n} slots");
    let weights = (0..m).map(|o| (o, (0..n).map(|s| a(s, (s + o) % m)).collect())).collect();
    Ok(ReplicatedFactor {
        diagonals: diagonals_from_weights(n, weights),
        log_delta_diag,
        shaped,
        fitted_giant_step: false,
    })
}

/// The one factor of [`ReplicatedTransform::gather`].
pub(crate) fn gather_factor<BE, F, A, Rg: Ring>(
    module: &Module<BE>,
    layout: ReplicatedLayout,
    log_delta_diag: usize,
    shaped: bool,
    a: A,
) -> Result<ReplicatedFactor<F>>
where
    BE: Backend,
    F: CKKSEncodingScalar,
    A: Fn(usize, usize) -> (F, F),
    Module<BE>: CKKSModuleInfos,
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
    Ok(ReplicatedFactor {
        diagonals: diagonals_from_weights(n, weights),
        log_delta_diag,
        shaped,
        fitted_giant_step: false,
    })
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

/// What encoding the diagonals of a [`ReplicatedTransform`] on the ring `Rg` needs of a module.
pub trait ReplicatedEncodeModule<BE: Backend, F: CKKSEncodingScalar, Rg: Ring = Standard>:
    CKKSEncodingOps<BE, F, Rg> + CKKSPlaintextAlloc<BE> + CKKSLinearTransformationAlloc<BE> + CKKSModuleInfos + GetDegree
{
}

impl<BE: Backend, F: CKKSEncodingScalar, Rg: Ring, M> ReplicatedEncodeModule<BE, F, Rg> for M where
    M: CKKSEncodingOps<BE, F, Rg> + CKKSPlaintextAlloc<BE> + CKKSLinearTransformationAlloc<BE> + CKKSModuleInfos + GetDegree
{
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

/// `(k, kR)` of a layout that fits the slots of degree `n` of the ring `Rg`, at most the module's degree.
pub(crate) fn check_period_at<BE: Backend, Rg: Ring>(
    module: &Module<BE>,
    n: Degree,
    layout: &ReplicatedLayout,
) -> Result<(usize, usize)>
where
    Module<BE>: CKKSModuleInfos,
{
    ensure!(
        n.as_usize() <= module.n() && n.as_usize().is_power_of_two(),
        "the degree {} is not a power of two at most the module's {}",
        n.as_usize(),
        module.n()
    );
    let (k, r) = (1usize << layout.log_k(), 1usize << layout.log_r());
    let slots = module.ckks_slots_of_degree::<Rg>(n.as_usize());
    ensure!(
        k * r <= slots,
        "{k} values of period {r} need more than the {slots} slots of degree {}",
        n.as_usize()
    );
    Ok((k, k * r))
}

/// `(k, kR)` of a layout that fits the slots of the ring `Rg`.
pub(crate) fn check_period<BE: Backend, Rg: Ring>(module: &Module<BE>, layout: &ReplicatedLayout) -> Result<(usize, usize)>
where
    Module<BE>: CKKSModuleInfos,
{
    let (k, r) = (1usize << layout.log_k(), 1usize << layout.log_r());
    let slots = module.ckks_max_slots::<Rg>();
    ensure!(k * r <= slots, "{k} values of period {r} need more than the {slots} slots");
    Ok((k, k * r))
}

/// The diagonals of `n` slots given by their full values, `(offset, values)`.
fn diagonals_from_weights<F: CKKSEncodingScalar>(n: usize, weights: Vec<(usize, Vec<(F, F)>)>) -> ComplexDiagonals<F> {
    let mut diagonals = ComplexDiagonals::new(Diagonals::new(n), Diagonals::new(n));
    for (o, w) in weights {
        let (re, im): (Vec<F>, Vec<F>) = w.into_iter().unzip();
        set_complex(&mut diagonals, galois_of_rotation(o as i64), (re, im));
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

/// Sets both parts of the diagonal of the Galois element `galois`.
pub(crate) fn set_complex<F>(diagonals: &mut ComplexDiagonals<F>, galois: i64, (re, im): (Vec<F>, Vec<F>)) {
    diagonals.re.set(galois, re);
    diagonals.im.set(galois, im);
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
            set_complex(
                &mut factor,
                galois_of_rotation(-(((r - 1) * v) as i64)),
                mask_diagonal(n, on, F::one()),
            );
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
            set_complex(
                &mut factor,
                galois_of_rotation(((r - 1) * v) as i64),
                mask_diagonal(n, on, value),
            );
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
            assert!(factors.iter().all(|f| f.galois().len() <= 1 << log_group));
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
            let eval_mod = crate::layouts::ReplicatedRoute::paper_eval_mod(&unit.layout(4).unwrap());
            assert_eq!(eval_mod.f_mod_log_delta, log_delta, "{unit:?}");
        }
    }

    #[test]
    fn split_depth_covers_the_bits() {
        assert_eq!(split_depth(10, 3), [4, 3, 3]);
        assert_eq!(split_depth(10, 1), [10]);
        assert_eq!(split_depth(2, 5), [1, 1]);
    }
}
