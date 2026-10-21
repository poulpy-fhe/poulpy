//! Linear maps, conversions and bootstrapping of a replicated slot layout, the
//! small-batch scheme of [ePrint 2026/2513](https://eprint.iacr.org/2026/2513)
//! (see `docs/replicated.md`).

use std::collections::{BTreeMap, HashMap};

use anyhow::{Context, Result, ensure};
use poulpy_core::{
    Distribution, GLWEAutomorphismKeyEncryptSk, GLWESwitchingKeyEncryptSk, GLWETensorKeyEncryptSk, GLWETrace, GetDistributionMut,
    layouts::{
        BackendGLWESecret, Base2K, Degree, Diagonals, GGLWELayout, GLWEAutomorphismKeyLayout, GLWEInfos, GLWELayout,
        GLWESecretLayout, GLWESecretSampling, GLWESwitchingKeyLayout, GLWETensorKeyLayout, GetDegree, LWEInfos,
        LinearTransformationLayout, LinearTransformationPrepared, LinearTransformationStrategy, ModuleCoreAlloc, Rank,
        TorusPrecision,
    },
};
use poulpy_hal::{
    api::{CnvPVecAlloc, ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Backend, Module, ScratchArena, ScratchOwned},
    source::Source,
};

use crate::{
    CKKSLayout, CKKSModuleInfos, CoeffsMeta, GLWEPlaintextMeta, ReplicatedLayout, Scale, SetCKKSInfos, SetGLWEPlaintextInfos,
    SlotLayout, SlotsKind,
    api::{CKKSEncodingHostOps, CKKSEncodingOps, CKKSEncodingScalar, CKKSLinearTransformationOps},
    layouts::{
        BootstrappingKeySet, BootstrappingKeysLayout, CKKSModuleAlloc, CKKSPlaintextOwned, ComplexDiagonals,
        EncapsulationKeysLayout, MaskedCoeffs,
        eval_mod::{EvalMod, EvalModPlan, EvalModType, compile_eval_mod_replicated, compile_eval_mod_replicated_packed},
    },
    polynomial::SplitStrategy,
    reference::linear_transformation_diagonals::period_degree,
};

/// Diagonals of the conversions between a CKKS ciphertext of period `k` and
/// a [`ReplicatedLayout`] of `k` values, each one linear transform.
///
/// With `n = kR` slots and `h = N/(2k)`:
///
/// - `to`: `k` diagonals writing value `z_(t/R mod k)` on the data slots `R | t`
///   and zero on the guards, quantized with shaping;
/// - `from` (when `R | k`): the gather diagonals, of weight `R`, reading only
///   data slots so that the normalized trace onto period `k` returns the
///   average of the `h/R` copies of each value, followed by `log2 h` trace
///   rotations by `k, 2k, ..., (h/2)k`.
///
/// Both conversions compute `extra_bits` bits under the scale and consume the
/// diagonals' `log_delta_diag` bits of budget. `log_delta` is the scale of the
/// replicated side.
pub struct ReplicatedConversion<BE: Backend> {
    pub(crate) layout: ReplicatedLayout,
    pub(crate) log_delta: usize,
    pub(crate) extra_bits: usize,
    pub(crate) to: ReplicatedTransform<BE>,
    pub(crate) from: Option<ReplicatedTransform<BE>>,
    pub(crate) trace: Vec<i64>,
}

impl<BE: Backend> ReplicatedConversion<BE> {
    pub fn new<F>(
        module: &Module<BE>,
        layout: ReplicatedLayout,
        base2k: Base2K,
        log_delta: usize,
        log_delta_diag: usize,
        extra_bits: usize,
    ) -> Result<Self>
    where
        F: CKKSEncodingScalar,
        Module<BE>: ReplicatedTransformModule<BE, F>,
    {
        let (k, r) = (1usize << layout.log_k(), 1usize << layout.log_r());
        let select = |s: usize, j: usize| {
            let read = s.is_multiple_of(r) && (s / r) % k == j;
            (if read { F::one() } else { F::zero() }, F::zero())
        };
        let to = ReplicatedTransform::from_period(module, layout, base2k, log_delta_diag, true, select)?;
        let identity = |c: usize, j: usize| (if c == j { F::one() } else { F::zero() }, F::zero());
        let from = if r <= k {
            Some(ReplicatedTransform::gather(
                module,
                layout,
                base2k,
                log_delta_diag,
                false,
                identity,
            )?)
        } else {
            None
        };
        let h = module.ring_degree().as_usize() / (2 * k);
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

    pub fn layout(&self) -> ReplicatedLayout {
        self.layout
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

/// A linear map `y = Σ_o D_o ⊙ rot_o(x)` on the `n = kR` slots of a
/// replicated layout's period, `k` diagonals whatever the map, prepared as a
/// BSGS transform with giant step `⌈√k⌉` rounded up to a power of two.
///
/// A shaped diagonal of giant step `rot` is quantized with the shaping
/// polynomial `g(X^(5^(−rot)))`, so that after the giant rotation its
/// quantization error is a multiple of `g`.
pub struct ReplicatedTransform<BE: Backend> {
    pub(crate) lt: LinearTransformationPrepared<BE>,
}

impl<BE: Backend> ReplicatedTransform<BE> {
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
    ) -> Result<Self>
    where
        F: CKKSEncodingScalar,
        A: Fn(usize, usize) -> (F, F),
        Module<BE>: ReplicatedTransformModule<BE, F>,
    {
        Self::from_input_period(module, layout, 1 << layout.log_k(), base2k, log_delta_diag, shaped, a)
    }

    /// [`Self::from_period`] from slots of period `m`, a divisor of `n`.
    pub fn from_input_period<F, A>(
        module: &Module<BE>,
        layout: ReplicatedLayout,
        m: usize,
        base2k: Base2K,
        log_delta_diag: usize,
        shaped: bool,
        a: A,
    ) -> Result<Self>
    where
        F: CKKSEncodingScalar,
        A: Fn(usize, usize) -> (F, F),
        Module<BE>: ReplicatedTransformModule<BE, F>,
    {
        let (_, n) = check_period(module, &layout)?;
        ensure!(n.is_multiple_of(m), "the input period {m} must divide the {n} slots");
        let weights = (0..m).map(|o| (o, (0..n).map(|s| a(s, (s + o) % m)).collect())).collect();
        Self::encode(module, layout, base2k, log_delta_diag, shaped, weights)
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
    ) -> Result<Self>
    where
        F: CKKSEncodingScalar,
        A: Fn(usize, usize) -> (F, F),
        Module<BE>: ReplicatedTransformModule<BE, F>,
    {
        let (k, n) = check_period(module, &layout)?;
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
        Self::encode(module, layout, base2k, log_delta_diag, shaped, weights)
    }

    /// Slot rotations of the transform, for their automorphism keys.
    pub fn rotations(&self) -> Vec<i64> {
        let babies = self.lt.giant_steps.iter().flat_map(|gs| gs.diagonals.iter().map(|d| d.baby));
        let giants = self.lt.giant_steps.iter().map(|gs| gs.rot);
        let mut rotations: Vec<i64> = babies.chain(giants).filter(|&o| o != 0).collect();
        rotations.sort_unstable();
        rotations.dedup();
        rotations
    }

    fn encode<F>(
        module: &Module<BE>,
        layout: ReplicatedLayout,
        base2k: Base2K,
        log_delta_diag: usize,
        shaped: bool,
        weights: Vec<(usize, Vec<(F, F)>)>,
    ) -> Result<Self>
    where
        F: CKKSEncodingScalar,
        Module<BE>: ReplicatedTransformModule<BE, F>,
    {
        let n = weights.first().map_or(0, |(_, w)| w.len());
        let largest = weights
            .iter()
            .flat_map(|(_, w)| w.iter().map(|(re, im)| re.abs().max(im.abs())))
            .fold(F::zero(), F::max);
        // Bits above the scale for the largest weight, plus a sign and a carry.
        let headroom = (usize::BITS - (largest.to_f64().unwrap_or(1.0).ceil() as usize).max(1).leading_zeros()) as usize + 2;
        let mut diagonals = ComplexDiagonals {
            re: Diagonals::new(n),
            im: Diagonals::new(n),
        };
        for (o, w) in weights {
            let (re, im): (Vec<F>, Vec<F>) = w.into_iter().unzip();
            diagonals.re.set(o as i64, re);
            diagonals.im.set(o as i64, im);
        }
        let count = diagonals.indexes().len();
        let strategy = if count <= 2 {
            LinearTransformationStrategy::Direct
        } else {
            // The smallest power of two whose square covers the diagonals.
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
                n: period_degree::<BE, _>(module, n),
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
        let mut lt = module.ckks_linear_transformation_alloc_from_infos(&lt_layout, &infos);
        let pt_slots = module.ckks_slots_of_degree(infos.glwe_layout.n.as_usize());
        let mut scratch = ScratchOwned::<BE>::alloc(module.ckks_reim_tmp_bytes(pt_slots));
        let mut targets = lt.giant_steps.iter_mut().flat_map(|gs| {
            let rot = gs.rot;
            gs.diagonals.iter_mut().map(move |d| (rot, &mut d.plaintext))
        });
        let mut result = Ok(());
        // The builder visits the diagonals in the schedule's order.
        diagonals.build_transform(strategy, |re, im| {
            if let (Ok(()), Some((rot, pt))) = (&result, targets.next()) {
                let label = if shaped {
                    SlotLayout::Replicated(twisted(&layout, module.ckks_galois_element(-rot)))
                } else {
                    SlotLayout::Full
                };
                pt.set_meta(GLWEPlaintextMeta {
                    slot_layout: label,
                    ..meta
                });
                let reps = pt_slots / re.len();
                result = module
                    .ckks_encode_reim_slots_into(pt, &re.repeat(reps), &im.repeat(reps), &mut scratch.borrow())
                    .map_err(anyhow::Error::from);
                // The diagonals multiply a ciphertext of the full layout.
                pt.set_slot_layout(SlotLayout::Full);
            }
        });
        result?;
        let first = lt.first_diagonal_plaintext().context("a transform has a diagonal")?;
        let mut prepared = LinearTransformationPrepared::<BE>::alloc_prepared_from_index(module, &lt.index(), first);
        let mut scratch = ScratchOwned::<BE>::alloc(module.ckks_prepare_linear_transformation_rhs_tmp_bytes(first));
        module.ckks_prepare_linear_transformation_rhs(&mut prepared, &lt, &mut scratch.borrow());
        Ok(Self { lt: prepared })
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
    ReplicatedLayout::new(layout.log_k() as u8, layout.log_r() as u8, layout.w() as u8, &coeffs)
        .expect("an automorphism keeps the shaping polynomial invertible")
}

/// What building a [`ReplicatedTransform`] needs of a module.
pub trait ReplicatedTransformModule<BE: Backend, F: CKKSEncodingScalar>:
    CKKSEncodingOps<BE, F> + CKKSModuleAlloc<BE> + CKKSModuleInfos + GetDegree + CKKSLinearTransformationOps<BE> + CnvPVecAlloc<BE>
{
}

impl<BE: Backend, F: CKKSEncodingScalar, M> ReplicatedTransformModule<BE, F> for M where
    M: CKKSEncodingOps<BE, F>
        + CKKSModuleAlloc<BE>
        + CKKSModuleInfos
        + GetDegree
        + CKKSLinearTransformationOps<BE>
        + CnvPVecAlloc<BE>
{
}

/// `(k, kR)` of a layout that fits the ring's slots.
fn check_period<BE: Backend>(module: &Module<BE>, layout: &ReplicatedLayout) -> Result<(usize, usize)>
where
    Module<BE>: GetDegree,
{
    let (k, r) = (1usize << layout.log_k(), 1usize << layout.log_r());
    let half = module.ring_degree().as_usize() / 2;
    ensure!(k * r <= half, "{k} values of period {r} need more than the {half} slots");
    Ok((k, k * r))
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
}

impl ReplicatedBootstrapPlan {
    /// The parameters of ePrint 2026/2513 for `layout`: inputs at `2^30` with
    /// a ratio of `2^13`, outputs at `2^20`, the even cosine of degree 30 on
    /// `K = 16` with 3 double angles at `2^28` and constants at `2^26`, and
    /// the packed EvalMod.
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
                f_mod_log_delta: 28,
            },
            packed: true,
        }
    }

    /// Width of the input at scale `Δ_b`: the message ratio and the bits
    /// `SlotsToCoeffs` consumes above it.
    pub fn k_in(&self) -> usize {
        self.log_delta + self.eval_mod.log_msg_ratio + self.s2c_log_delta_diag
    }

    /// Width after ModUp for an output of width `k_out`: `CoeffsToSlots`,
    /// EvalMod, and the rescale from `Δ_b`, lifted by the ModUp bits, to the
    /// leveled scale.
    pub fn k_boot(&self, k_out: usize) -> usize {
        k_out + self.c2s_log_delta_diag + self.eval_mod.consumed_bits() + self.log_delta + self.mod_up_extra_bits
            - self.level_log_delta
    }

    /// Width of the widest key switch: the raised modulus and the working bits
    /// of the top trace and `CoeffsToSlots`.
    pub fn key_k(&self, k_out: usize) -> usize {
        self.k_boot(k_out) + self.extra_bits
    }

    /// Key layouts for outputs of width `k_out`: the automorphism and tensor
    /// keys cover [`Self::key_k`] in digits of `k_digit` bits, with an
    /// auxiliary modulus of `k_aux` bits; with `encapsulation`, the key to the
    /// ephemeral secret covers the ModUp input in one-limb digits and an
    /// auxiliary modulus of one limb and a bit, and the key back the raised
    /// modulus.
    pub fn key_layouts(
        &self,
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
        };
        let k_key = self.key_k(k_out);
        let one_limb = TorusPrecision(base2k.as_u32());
        BootstrappingKeysLayout {
            rotation_keys: BTreeMap::new(),
            automorphism_key: GLWEAutomorphismKeyLayout {
                n,
                base2k,
                dnum: dnum(k_key, digit),
                k_aux: TorusPrecision(k_aux as u32),
                rank: Rank(1),
                k_digit: digit,
            },
            tensor_key: GLWETensorKeyLayout {
                n,
                base2k,
                dnum: dnum(k_key, digit),
                k_aux: TorusPrecision(k_aux as u32),
                rank: Rank(1),
                k_digit: digit,
            },
            encapsulation: encapsulation.then(|| EncapsulationKeysLayout {
                dense_to_sparse: switching(self.log_delta + self.eval_mod.log_msg_ratio, one_limb, base2k.as_usize() + 1),
                sparse_to_dense: switching(self.k_boot(k_out), digit, k_aux),
            }),
        }
    }

    /// [`Self::key_layouts`] with the fewest gadget digits within the modulus
    /// bounds: every key under the secret within `max_dense_modulus` bits and
    /// the key to the ephemeral secret within `max_sparse_modulus`, each with
    /// an auxiliary modulus of one digit, `log2(QP) = (dnum + 1)·k_digit`.
    pub fn secure_key_layouts(
        &self,
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
        let mut layouts = self.key_layouts(n, base2k, k_out, k_digit, k_digit, true);
        if let Some(encapsulation) = layouts.encapsulation.as_mut() {
            let sparse = TorusPrecision(sparse_digit as u32);
            encapsulation.dense_to_sparse.k_digit = sparse;
            encapsulation.dense_to_sparse.k_aux = sparse;
            encapsulation.dense_to_sparse.dnum = GGLWELayout::dnum_for_input(TorusPrecision(k_q_b as u32), sparse);
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

impl<BE: Backend, F: CKKSEncodingScalar> ReplicatedBootstrapping<BE, F> {
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
        let (k, r) = (1usize << layout.log_k(), 1usize << layout.log_r());
        ensure!(r <= k, "bootstrapping needs the period {r} to divide the batch size {k}");
        let half = module.ckks_max_slots();
        let v = embedding::<BE, F>(module, k, base2k)?;
        let s2c = ReplicatedTransform::gather(module, layout, base2k, plan.s2c_log_delta_diag, false, |c, j| v[c][j])?;
        let scale = F::from_usize(2 * k * plan.eval_mod.f_mod_interval)
            .context("the CoeffsToSlots normalization is not representable")?
            .recip();
        // `A[t][c] = conj(V[c][t])/(2Kk)`, written on the data slots.
        let a = |t: usize, c: usize| {
            let (re, im) = v[c][t];
            (re * scale, -im * scale)
        };
        let packed = if plan.packed && 4 * k * r <= 2 * half {
            Some(ReplicatedLayout::new(
                layout.log_k() as u8 + 1,
                layout.log_r() as u8,
                layout.w() as u8,
                layout.shaping(),
            )?)
        } else {
            None
        };
        let c2s = match packed {
            // `−i·A` on the second batch.
            Some(double) => {
                ReplicatedTransform::from_input_period(module, double, k, base2k, plan.c2s_log_delta_diag, true, |s, c| {
                    match (s % r, s / r) {
                        (0, t) if t < k => a(t, c),
                        (0, t) => {
                            let (re, im) = a(t - k, c);
                            (im, -re)
                        }
                        _ => (F::zero(), F::zero()),
                    }
                })?
            }
            None => ReplicatedTransform::from_period(module, layout, base2k, plan.c2s_log_delta_diag, true, |s, c| {
                if s.is_multiple_of(r) {
                    a(s / r, c)
                } else {
                    (F::zero(), F::zero())
                }
            })?,
        };
        let eval_mod = match packed {
            Some(double) => compile_eval_mod_replicated_packed::<BE, F>(base2k, plan.eval_mod, double, module, scratch)?,
            None => compile_eval_mod_replicated::<BE, F>(base2k, plan.eval_mod, layout, module, scratch)?,
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

impl<BE: Backend, F> ReplicatedBootstrapping<BE, F> {
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
            .map(|o| module.ckks_galois_element(o))
            .chain(module.glwe_trace_galois_elements())
            .chain([-1])
            .collect();
        elements.sort_unstable();
        elements.dedup();
        elements
    }

    /// Generates the keys of the bootstrapping under `sk`: the automorphism
    /// keys of [`Self::galois_elements`], the tensor key, and with
    /// `layout.encapsulation` the keys to and from an ephemeral secret of
    /// Hamming weight `hamming_weight` sampled from `source_xs`.
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
        let mut rotation_keys = HashMap::new();
        for gal in self.galois_elements(module) {
            let mut key = module.glwe_automorphism_key_alloc_from_infos(&layout.rotation_key(gal));
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
            let mut sparse_to_dense = module.glwe_switching_key_alloc_from_infos(&encapsulation.sparse_to_dense);
            module.glwe_switching_key_encrypt_sk(&mut sparse_to_dense, &sparse, sk, source_xe, source_xa, scratch);
            (dense_to_sparse, sparse_to_dense)
        });
        BootstrappingKeySet {
            rotation_keys,
            tensor_key,
            encapsulation_keys,
        }
    }
}

/// `V[c][j]`, slot `c` of the monomial `Y'^j` of the subring of degree `2k`,
/// read from the encoder.
fn embedding<BE: Backend, F>(module: &Module<BE>, k: usize, base2k: Base2K) -> Result<Vec<Vec<(F, F)>>>
where
    F: CKKSEncodingScalar,
    Module<BE>: ReplicatedTransformModule<BE, F>,
{
    let log_delta = 40;
    let mut scratch = ScratchOwned::<BE>::alloc(module.ckks_reim_tmp_bytes(k));
    let mut columns = Vec::with_capacity(k);
    for j in 0..k {
        let mut pt = module.ckks_pt_vec_alloc(k, base2k, TorusPrecision((log_delta + 2) as u32));
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
        module.ckks_decode_reim_into(&pt, &mut re, &mut im, &mut scratch.borrow())?;
        columns.push(re.into_iter().zip(im).collect::<Vec<_>>());
    }
    // `Y'^k` is `i` on every slot, so `m_j + i·m_(j+k)` reads `V[c][j]`.
    Ok((0..k).map(|c| (0..k).map(|j| columns[j][c]).collect()).collect())
}
