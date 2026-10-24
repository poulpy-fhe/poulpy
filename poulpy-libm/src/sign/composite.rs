//! Composite-minimax sign plans and host encoding.
//!
//! The construction follows Lee et al., ePrint 2020/834. Default coefficients
//! are ported from Lattigo.

#![allow(clippy::excessive_precision)]

use anyhow::{Result, anyhow, ensure};
use num_traits::{Float, FloatConst, FromPrimitive};
use poulpy_core::layouts::Base2K;
use poulpy_hal::layouts::{HostBytesBackend, Module};

use poulpy_ckks::{
    CKKSLayout,
    api::{Basis, Parity},
    layouts::{CKKSPlaintext, CKKSPlaintextVecHostCodec, CKKSScalar},
    polynomial::{BSGSPolynomial, EncodeBSGS, Polynomial, SplitStrategy},
};

use poulpy_ckks::approximation::{RemezOptions, sign_composite_coeffs};

/// Default Lattigo sign composite as Chebyshev coefficient rows.
#[rustfmt::skip]
pub const DEFAULT_SIGN_COMPOSITE_CHEBYSHEV: &[&[f64]] = &[
    &[0.0, 0.6371462957672043333, 0.0, -0.2138032460610765328, 0.0, 0.1300439303835664499, 0.0, -0.0948842756566191044, 0.0, 0.0760417811618939909, 0.0, -0.0647714820920817557, 0.0, 0.0577904411211959048, 0.0, -0.5275634328386103792],
    &[0.0, 0.6371463830322414578, 0.0, -0.2138032749880402509, 0.0, 0.1300439475440832118, 0.0, -0.0948842877009570762, 0.0, 0.0760417903036533484, 0.0, -0.0647714893343788749, 0.0, 0.0577904470018789283, 0.0, -0.5275633669027163690],
    &[0.0, 0.6371474873319408921, 0.0, -0.2138036410457105809, 0.0, 0.1300441647026617059, 0.0, -0.0948844401165889295, 0.0, 0.0760419059884502454, 0.0, -0.0647715809823254389, 0.0, 0.0577905214191996406, 0.0, -0.5275625325136631842],
    &[0.0, 0.6370469776996076431, 0.0, -0.2134526779726600620, 0.0, 0.1294300181775238920, 0.0, -0.0939692999460324791, 0.0, 0.0747629355709698798, 0.0, -0.0630298319949635571, 0.0, 0.0554299627688379896, 0.0, -0.0504549111784642023, 0.0, 0.5242368268605847996],
    &[0.0, 0.6371925153898374380, 0.0, -0.2127272333844484291, 0.0, 0.1280350175397897124, 0.0, -0.0918861831051024970, 0.0, 0.0719237384158242601, 0.0, -0.0593247422790627989, 0.0, 0.0506973946536399213, 0.0, -0.0444605229007162961, 0.0, 0.0397788020190944552, 0.0, -0.0361705584687241925, 0.0, 0.0333397971860406254, 0.0, -0.0310960060432036761, 0.0, 0.0293126335952747929, 0.0, -0.0279042579223662982, 0.0, 0.0268135229627401517, 0.0, -0.5128179323757194002],
    &[0.0, 0.6484328404896112084, 0.0, -0.2164688471885406655, 0.0, 0.1302737771018761402, 0.0, -0.0934786176742356885, 0.0, 0.0731553324133884104, 0.0, -0.0603252338481440981, 0.0, 0.0515366139595849853, 0.0, -0.0451803385226980999, 0.0, 0.0404062758116036740, 0.0, -0.0367241775307736352, 0.0, 0.0338327393147257876, 0.0, -0.0315379870551266008, 0.0, 0.0297110181467332488, 0.0, -0.0282647625290482803, 0.0, 0.0271406820054187399, 0.0, -0.5041440308249296747],
    &[0.0, 0.8988231150519633581, 0.0, -0.2996064625122592138, 0.0, 0.1797645789317822353, 0.0, -0.1284080039344265678, 0.0, 0.0998837306152582349, 0.0, -0.0817422066647773587, 0.0, 0.0691963884439569899, 0.0, -0.0600136111161848355, 0.0, 0.0530132660795356506, 0.0, -0.0475133961913746909, 0.0, 0.0430936248086665091, 0.0, -0.0394819050695222720, 0.0, 0.0364958013826412785, 0.0, -0.0340100990129699835, 0.0, 0.0319381346687564699, 0.0, -0.3095637759472512887],
    &[0.0, 1.2654405107323937767, 0.0, -0.4015427502443620045, 0.0, 0.2182109348265640036, 0.0, -0.1341692540177466882, 0.0, 0.0852282854825304735, 0.0, -0.0539043807248265057, 0.0, 0.0332611560159092728, 0.0, -0.0197419082926337129, 0.0, 0.0111368708758574529, 0.0, -0.0058990205011466309, 0.0, 0.0028925861201479251, 0.0, -0.0012889673944941461, 0.0, 0.0005081425552893727, 0.0, -0.0001696330470066833, 0.0, 0.0000440808328172753, 0.0, -0.0000071549240608255],
    &[0.0, 1.1962890625, 0.0, -0.2392578125, 0.0, 0.0478515625, 0.0, -0.0048828125],
];

/// Chebyshev `1.5·x − 0.5·x³` sign refinement.
pub const COEFFS_SIGN_X2_CHEBYSHEV: &[f64] = &[0.0, 1.125, 0.0, -0.125];

/// Degree-seven Chebyshev sign refinement.
pub const COEFFS_SIGN_X4_CHEBYSHEV: &[f64] = &[
    0.0,
    1.1962890625,
    0.0,
    -0.2392578125,
    0.0,
    0.0478515625,
    0.0,
    -0.0048828125,
];

/// Prepared composite sign polynomial.
pub struct SignComposite<F, P> {
    /// BSGS factors in evaluation order.
    pub polys_bsgs: Vec<BSGSPolynomial<P>>,
    /// Host factors for reference evaluation.
    pub polys_host: Vec<Polynomial<F>>,
    /// Coefficient scale.
    pub coeff_log_delta: usize,
}

impl<F> SignComposite<F, CKKSPlaintext<Vec<u8>, i64>>
where
    F: CKKSScalar + Float + FloatConst + FromPrimitive + std::fmt::Debug,
    CKKSPlaintext<Vec<u8>, i64>: CKKSPlaintextVecHostCodec<F>,
{
    /// Prepares odd Chebyshev factors on `[−1, 1]`.
    pub fn from_coeffs(
        base2k: Base2K,
        rows: &[&[f64]],
        coeff_meta: CKKSLayout,
        strategy: SplitStrategy,
        module: &Module<HostBytesBackend>,
    ) -> Result<Self> {
        ensure!(
            !rows.is_empty(),
            "sign composite: coefficient rows must be non-empty"
        );
        for (i, row) in rows.iter().enumerate() {
            ensure!(
                !row.is_empty(),
                "sign composite: coefficient row {i} is empty"
            );
            ensure!(
                (row.len() - 1) % 2 == 1,
                "sign composite: coefficient row {i} must have odd degree"
            );
        }
        let rows: Vec<Vec<F>> = rows
            .iter()
            .enumerate()
            .map(|(i, row)| {
                row.iter()
                    .map(|&c| {
                        F::from_f64(c).ok_or_else(|| {
                            anyhow!("sign composite: coefficient row {i}: {c} not representable")
                        })
                    })
                    .collect::<Result<_>>()
            })
            .collect::<Result<_>>()?;
        Self::from_rows(rows, base2k, coeff_meta, strategy, module)
    }

    /// Fits sign on `[−1, −tau] ∪ [tau, 1]` with odd minimax factors.
    /// The last degree is reused when needed.
    #[allow(clippy::too_many_arguments)]
    pub fn from_minimax(
        tau: F,
        target_bits: f64,
        degrees: &[usize],
        max_factors: usize,
        base2k: Base2K,
        coeff_meta: CKKSLayout,
        strategy: SplitStrategy,
        module: &Module<HostBytesBackend>,
    ) -> Result<Self> {
        let rows = sign_composite_coeffs::<F>(
            tau,
            target_bits,
            degrees,
            max_factors,
            RemezOptions::default(),
        )?;
        Self::from_rows(rows, base2k, coeff_meta, strategy, module)
    }

    /// Compiles [`DEFAULT_SIGN_COMPOSITE_CHEBYSHEV`].
    pub fn from_default(
        base2k: Base2K,
        coeff_meta: CKKSLayout,
        strategy: SplitStrategy,
        module: &Module<HostBytesBackend>,
    ) -> Result<Self> {
        Self::from_coeffs(
            base2k,
            DEFAULT_SIGN_COMPOSITE_CHEBYSHEV,
            coeff_meta,
            strategy,
            module,
        )
    }

    /// Encodes Chebyshev rows as half-scale BSGS factors.
    fn from_rows(
        rows: Vec<Vec<F>>,
        base2k: Base2K,
        coeff_meta: CKKSLayout,
        strategy: SplitStrategy,
        module: &Module<HostBytesBackend>,
    ) -> Result<Self> {
        let half =
            F::from_f64(0.5).ok_or_else(|| anyhow!("sign composite: 0.5 not representable"))?;
        let mut polys_bsgs = Vec::with_capacity(rows.len());
        let mut polys_host = Vec::with_capacity(rows.len());
        for (i, coeffs) in rows.into_iter().enumerate() {
            // Real cleaning restores the half-scale encoding.
            let poly = Polynomial::new_with_parity(Basis::Chebyshev, coeffs.clone(), Parity::Odd);
            let half_coeffs: Vec<F> = coeffs.into_iter().map(|c| c * half).collect();
            let poly_half = Polynomial::new_with_parity(Basis::Chebyshev, half_coeffs, Parity::Odd);
            polys_bsgs.push(
                poly_half
                    .encode_bsgs_with(module, base2k, coeff_meta.into(), strategy)
                    .map_err(|e| anyhow!("sign composite: coefficient row {i}: {e}"))?,
            );
            polys_host.push(poly);
        }
        Ok(Self {
            polys_bsgs,
            polys_host,
            coeff_log_delta: coeff_meta.meta.log_delta,
        })
    }
}

impl<F, P> SignComposite<F, P> {
    /// Number of factors composed.
    pub fn len(&self) -> usize {
        self.polys_bsgs.len()
    }

    /// Whether the composition has no factors.
    pub fn is_empty(&self) -> bool {
        self.polys_bsgs.is_empty()
    }

    /// Consumed modulus bits.
    pub fn consumed_bits(&self, input_log_delta: usize) -> usize {
        self.polys_bsgs
            .iter()
            .map(|p| p.consumed_bits(input_log_delta, self.coeff_log_delta))
            .sum()
    }

    /// Multiplicative depth.
    pub fn depth(&self) -> usize {
        self.polys_bsgs.iter().map(|p| p.eval_depth()).sum()
    }

    /// Maps prepared plaintexts.
    pub fn map_plaintexts<Q>(self, mut f: impl FnMut(&P) -> Q) -> SignComposite<F, Q> {
        SignComposite {
            polys_bsgs: self
                .polys_bsgs
                .iter()
                .map(|p| p.map_baby_steps_ref(&mut f))
                .collect(),
            polys_host: self.polys_host,
            coeff_log_delta: self.coeff_log_delta,
        }
    }
}

impl<F, P> SignComposite<F, P>
where
    F: Float + FloatConst + FromPrimitive + std::fmt::Debug,
{
    /// Evaluates the host composition.
    pub fn evaluate(&self, x: F) -> F {
        let mut y = x;
        for p in &self.polys_host {
            y = p.evaluate(y);
        }
        y
    }
}
