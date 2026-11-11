//! How the keys of a bootstrap take their shape from the widths they switch.

use std::collections::BTreeMap;

use anyhow::{Context, Result, ensure};
use poulpy_core::layouts::{Base2K, Degree, GGLWELayout, GLWESwitchingKeyLayout, Rank, TorusPrecision};

use crate::{
    layouts::{BootstrappingKeysLayout, BootstrappingPlan, EncapsulationKeysLayout},
    presets::first_factor::{first_factor_diagonal_growth, first_factor_small_bits, first_factor_work},
};

/// Bits a key keeps between its modulus and the input plus one digit.
///
/// At that point the key noise is `sigma * sqrt(n * digits / 12)` units of the
/// input's last bit (see [`poulpy_core::layouts::key_k`]): about `2^9` at
/// `N = 2^16`, `2^7` with this margin. A bootstrapped ciphertext shows it only
/// past its own error, `log_delta - log2_precision` bits above its last bit:
/// 15 or more in the presets, which leaves 8. Measured on the S2C-first preset,
/// the precision starts to drop about 10 bits below this margin. A preset with a
/// narrower gap between scale and precision needs more.
const KEY_NOISE_MARGIN: usize = 2;

/// What the keys of a bootstrap are shaped under. The plan fixes the widths a bootstrap switches at, from the offsets of
/// its stages and steps, and this policy the layout of a key of a given width: the whole-limb digit of fewest rows, and at
/// that row count the narrowest.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BootstrappingKeyPolicy {
    /// Exponent of the ring degree.
    pub log_n: usize,
    /// Limb radix.
    pub base2k: usize,
    /// Rank of the keys.
    pub rank: usize,
    /// Largest modulus, in bits, of a key under the dense secret.
    pub dense_log_modulus: usize,
    /// Largest modulus, in bits, of a key under the sparse secret.
    pub sparse_log_modulus: usize,
    /// Widest digit, in limbs, of the high-modulus keys, `None` leaving them uncapped.
    pub max_dsize: Option<usize>,
    /// Digit, in limbs, of the dense-to-sparse key.
    pub dense_to_sparse_dsize: usize,
}

impl BootstrappingKeyPolicy {
    /// The layouts of the keys of `plan`, bootstrapping at `bootstrap_k` bits an input of `input_k` bits at scale
    /// `2^log_delta`. Each key covers the widest ciphertext it switches
    /// ([`key_widths`](BootstrappingPlan::key_widths)), and a key only used on narrower ciphertexts takes the digit of
    /// fewest rows those admit.
    ///
    /// # Errors
    ///
    /// When a key has no digit under [`Self::max_dsize`] that fits its modulus bound.
    pub fn keys_layout(
        &self,
        plan: &BootstrappingPlan,
        bootstrap_k: usize,
        input_k: usize,
        log_delta: usize,
    ) -> Result<BootstrappingKeysLayout> {
        let log_modulus = log_delta + plan.eval_mod().log_msg_ratio;
        let first_factor_on_raise = plan.coeffs_to_slots().first_factor_on_raise();
        let (dense, sparse) = (self.dense_log_modulus, self.sparse_log_modulus);
        let dense_to_sparse = self.key_layout(
            "dense-to-sparse",
            log_modulus,
            self.dense_to_sparse_dsize * self.base2k,
            sparse,
        )?;
        let sparse_to_dense = self.sparse_to_dense_layout(plan.raised_window(log_modulus), bootstrap_k, dense)?;

        let widths = plan.key_widths(self.log_n, bootstrap_k, input_k, log_delta);
        let shape = |name, k| self.key_layout(name, k, self.fewest_rows_k_digit(k, dense), dense);
        let rotation_keys = widths
            .rotation_keys
            .iter()
            .map(|(&p, &k)| Ok((p, shape("rotation", k)?.into())))
            .collect::<Result<BTreeMap<_, _>>>()?;
        let tensor_key = shape("tensor", widths.tensor_key)?;
        // The conversion keys are of degree 2N, under secrets of degree-N subrings
        // (`s(X²)`, the embedded CI secret), so the bounds of degree N apply.
        let wide = Self {
            log_n: self.log_n + 1,
            ..*self
        };
        let conversion_key = |k| wide.key_layout("conversion", k, wide.fewest_rows_k_digit(k, dense), dense);

        Ok(BootstrappingKeysLayout {
            rotation_keys,
            tensor_key: tensor_key.into(),
            encapsulation: Some(EncapsulationKeysLayout {
                dense_to_sparse: dense_to_sparse.into(),
                sparse_to_dense: (!first_factor_on_raise).then(|| sparse_to_dense.into()),
            }),
            first_factor_key: first_factor_on_raise
                .then(|| self.first_factor_key_layout(plan, log_modulus, bootstrap_k, dense))
                .transpose()?,
            conversion_keys: match widths.conversion_keys {
                Some((traced, embedded)) => Some((conversion_key(traced)?.into(), conversion_key(embedded)?.into())),
                None => None,
            },
        })
    }

    /// Layout of the sparse-to-dense key, whose switch is the raise: its rows sit
    /// under the raised ciphertext, `window` bits at the bottom of the raised
    /// width ([`BootstrappingPlan::raised_window`]), and end exactly at that width,
    /// so the key's modulus is the one a key covering the whole width would have.
    /// The digits are whole limbs, the fewest rows first; the bits their gadget
    /// adds over the window are a shift of the input inside it.
    fn sparse_to_dense_layout(&self, window: usize, bootstrap_k: usize, limit: usize) -> Result<GGLWELayout> {
        let shape = |k_digit: usize| {
            let gadget = window.div_ceil(k_digit) * k_digit;
            bootstrap_k.checked_sub(gadget).map(|k_offset| (gadget, k_offset))
        };
        let fits = |k_digit: usize| {
            shape(k_digit).is_some_and(|(gadget, k_offset)| self.key_fits(gadget, k_digit, limit.saturating_sub(k_offset)))
        };
        let k_digit = (1..=self.max_limbs(window))
            .map(|limbs| limbs * self.base2k)
            .filter(|&k_digit| fits(k_digit))
            .min_by_key(|&k_digit| (window.div_ceil(k_digit), k_digit))
            .unwrap_or(self.base2k);
        let (gadget, k_offset) = shape(k_digit).context("sparse-to-dense key: its digits overshoot the raised width")?;
        Ok(GGLWELayout {
            k_offset: TorusPrecision(k_offset as u32),
            ..self.key_layout("sparse-to-dense", gadget, k_digit, limit.saturating_sub(k_offset))?
        })
    }

    /// Layout of the keys of a first CoeffsToSlots factor applied by the raise:
    /// each switches a diagonal's term, the raised ciphertext with the guard of
    /// the working width ([`first_factor_work`]) below it and the growth of a
    /// product above it ([`first_factor_small_bits`]), so its rows sit under that
    /// many whole limbs and end at the working width. The digits are those of the
    /// key of fewest rows for that input, under the modulus bound less the offset.
    fn first_factor_key_layout(
        &self,
        plan: &BootstrappingPlan,
        log_modulus: usize,
        bootstrap_k: usize,
        limit: usize,
    ) -> Result<GLWESwitchingKeyLayout> {
        let (work, _) = first_factor_work(self.base2k, bootstrap_k);
        let raised = plan.raised_window(log_modulus);
        let growth = first_factor_diagonal_growth(self.base2k, &plan.coeffs_to_slots().coeffs_meta());
        let bits = first_factor_small_bits(self.base2k, self.log_n, bootstrap_k, raised, growth);
        let small = bits.div_ceil(self.base2k);
        ensure!(
            small < work,
            "first-factor key: a term of {small} limbs does not fit the {work} limbs the raise works at"
        );
        let k_offset = (work - small) * self.base2k;
        let below = limit.saturating_sub(k_offset);
        let input_k = small * self.base2k;
        let k_digit = self.fewest_rows_k_digit(input_k, below);
        Ok(GGLWELayout {
            k_offset: TorusPrecision(k_offset as u32),
            ..self.key_layout("first-factor", input_k, k_digit, below)?
        }
        .into())
    }

    /// Guard behind the digits of a key: a digit and `log_n` bits, or what the
    /// bound leaves when that is less.
    fn key_guard(&self, k_digit: usize, gadget: usize, limit: usize) -> usize {
        (k_digit + self.log_n).min(limit.saturating_sub(gadget))
    }

    /// Whether a key of `k_digit`-bit digits switching an input of `input_k` bits
    /// fits a modulus bound of `limit` bits: a guard of at least one digit (see
    /// [`poulpy_core::layouts::key_k`]) and [`KEY_NOISE_MARGIN`] past the input
    /// and one digit.
    fn key_fits(&self, input_k: usize, k_digit: usize, limit: usize) -> bool {
        let gadget = input_k.div_ceil(k_digit) * k_digit;
        let guard = self.key_guard(k_digit, gadget, limit);
        guard >= k_digit && gadget + guard >= input_k + k_digit + KEY_NOISE_MARGIN
    }

    /// Digit width, in bits, of the key of fewest rows that switches an input of
    /// `input_k` bits under a modulus bound of `limit` bits with digits of at most
    /// [`max_limbs`] limbs, and at that row count the narrowest, so the fewest
    /// operand limbs. Digits are whole limbs: a digit off a limb boundary costs the
    /// products the digit layout of their input.
    fn fewest_rows_k_digit(&self, input_k: usize, limit: usize) -> usize {
        let input_k = input_k.max(1);
        (1..=self.max_limbs(input_k))
            .map(|limbs| limbs * self.base2k)
            .filter(|&k_digit| self.key_fits(input_k, k_digit, limit))
            .min_by_key(|&k_digit| (input_k.div_ceil(k_digit), k_digit))
            .unwrap_or(self.base2k)
    }

    /// Widest digit, in limbs, of a key switching an input of `input_k` bits: one
    /// digit over the whole input, or the cap of [`Self::max_dsize`].
    fn max_limbs(&self, input_k: usize) -> usize {
        let limbs = input_k.div_ceil(self.base2k);
        self.max_dsize.map_or(limbs, |cap| cap.min(limbs))
    }

    /// Layout of a rank-preserving key of `k_digit`-bit digits covering an input
    /// of `input_k` bits under a modulus bound of `limit` bits: the digit count
    /// from [`GGLWELayout::dnum_for_input`], and behind it the guard of
    /// [`key_guard`], at least one digit.
    fn key_layout(&self, name: &str, input_k: usize, k_digit: usize, limit: usize) -> Result<GGLWELayout> {
        let base2k = Base2K(self.base2k as u32);
        let dnum = GGLWELayout::dnum_for_input(TorusPrecision(input_k as u32), TorusPrecision(k_digit as u32));
        let gadget = dnum.as_usize() * k_digit;
        let k_aux = self.key_guard(k_digit, gadget, limit);
        ensure!(
            self.key_fits(input_k, k_digit, limit),
            "{name} key: a digit of {k_digit} bits on a {input_k}-bit input needs a modulus of {} bits, above {limit}",
            (input_k + k_digit + KEY_NOISE_MARGIN).max(gadget + k_digit)
        );
        Ok(GGLWELayout {
            n: Degree(1 << self.log_n),
            base2k,
            dnum,
            k_aux: TorusPrecision(k_aux as u32),
            rank_in: Rank(self.rank as u32),
            rank_out: Rank(self.rank as u32),
            k_digit: TorusPrecision(k_digit as u32),
            k_offset: TorusPrecision(0),
            stride: 1,
        })
    }
}
