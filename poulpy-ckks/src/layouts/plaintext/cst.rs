use anyhow::Result;
use poulpy_core::layouts::{Base2K, LWEInfos, LWEPlaintext};
use poulpy_hal::layouts::ZnxView;
use rand_distr::num_traits::{Float, FromPrimitive, ToPrimitive};

use crate::{CKKSInfos, CKKSMeta};

#[derive(Debug, Clone, Copy, PartialEq)]
/// Constant CKKS plaintext in RNX form.
///
/// The real and imaginary parts are optional to support real-only constants
/// without allocating an unused complex component.
pub struct CKKSPlaintextCstRnx<F> {
    re: Option<F>,
    im: Option<F>,
}

impl<F> CKKSPlaintextCstRnx<F> {
    /// Creates a constant plaintext from optional real and imaginary parts.
    pub fn new(re: Option<F>, im: Option<F>) -> Self {
        Self { re, im }
    }

    /// Returns the real part if present.
    pub fn re(&self) -> Option<&F> {
        self.re.as_ref()
    }

    /// Returns the imaginary part if present.
    pub fn im(&self) -> Option<&F> {
        self.im.as_ref()
    }

    /// Splits the constant into owned real and imaginary parts.
    pub fn into_parts(self) -> (Option<F>, Option<F>) {
        (self.re, self.im)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
/// Constant CKKS plaintext in ZNX digit form.
pub struct CKKSPlaintextCstZnx {
    re: Option<Vec<i64>>,
    im: Option<Vec<i64>>,
    meta: CKKSMeta,
}

impl CKKSPlaintextCstZnx {
    /// Creates a quantized constant plaintext from encoded real and imaginary digits.
    pub fn new(re: Option<Vec<i64>>, im: Option<Vec<i64>>, meta: CKKSMeta) -> Self {
        Self { re, im, meta }
    }

    /// Returns the encoded real part if present.
    pub fn re(&self) -> Option<&[i64]> {
        self.re.as_deref()
    }

    /// Returns the encoded imaginary part if present.
    pub fn im(&self) -> Option<&[i64]> {
        self.im.as_deref()
    }

    /// Splits the constant into owned encoded parts.
    pub fn into_parts(self) -> (Option<Vec<i64>>, Option<Vec<i64>>) {
        (self.re, self.im)
    }
}

/// Conversion between scalar RNX constants and quantized ZNX constants.
pub trait CKKSConstPlaintextConversion {
    /// Maximum supported decimal precision for this conversion implementation.
    fn max_log_delta_prec() -> usize;

    /// Encodes a constant RNX plaintext into its default ZNX representation.
    ///
    /// This uses the natural plaintext precision for `prec`, namely
    /// `prec.min_k(base2k)`. This is the right encoding for operations such as
    /// `mul_const`, where the constant is consumed through the generic
    /// convolution path and does not need to be pre-aligned to a ciphertext's
    /// current remaining homomorphic capacity.
    fn to_znx(&self, base2k: Base2K, prec: CKKSMeta) -> Result<CKKSPlaintextCstZnx>;

    /// Encodes a constant RNX plaintext into a ZNX representation with an
    /// explicit effective torus precision `k`.
    ///
    /// This exists for direct in-ciphertext coefficient injection paths such as
    /// `add_const`. In that case the encoded digits are written straight into
    /// the ciphertext body, so they must already be aligned to the destination
    /// ciphertext precision, typically:
    ///
    /// `k = dst.log_budget() + prec.log_delta`
    ///
    /// Using `prec.min_k(base2k)` there would place the constant at the wrong
    /// bit position.
    fn to_znx_at_k(&self, base2k: Base2K, k: usize, log_delta: usize) -> Result<CKKSPlaintextCstZnx>;
}

fn max_log_delta_prec_for<F>() -> usize
where
    F: Float + ToPrimitive,
{
    ((-F::epsilon().log2()).round().to_usize().unwrap()) + 1
}

impl<F> CKKSConstPlaintextConversion for CKKSPlaintextCstRnx<F>
where
    F: Float + FromPrimitive + ToPrimitive,
{
    fn max_log_delta_prec() -> usize {
        max_log_delta_prec_for::<F>()
    }

    fn to_znx(&self, base2k: Base2K, prec: CKKSMeta) -> Result<CKKSPlaintextCstZnx> {
        self.to_znx_at_k(base2k, prec.min_k(base2k).as_usize(), prec.log_delta())
    }

    fn to_znx_at_k(&self, base2k: Base2K, k: usize, log_delta: usize) -> Result<CKKSPlaintextCstZnx> {
        let log_budget = k.saturating_sub(log_delta);

        anyhow::ensure!(log_delta <= Self::max_log_delta_prec());

        let scale = F::from_usize(log_delta).unwrap().exp2();
        let (re, im) = if log_delta + log_budget <= 63 {
            (
                self.re
                    .map(|re| encode_const_coeff_i64(base2k, k, (re * scale).round().to_i64().unwrap())),
                self.im
                    .map(|im| encode_const_coeff_i64(base2k, k, (im * scale).round().to_i64().unwrap())),
            )
        } else {
            (
                self.re
                    .map(|re| encode_const_coeff_i128(base2k, k, (re * scale).round().to_i128().unwrap())),
                self.im
                    .map(|im| encode_const_coeff_i128(base2k, k, (im * scale).round().to_i128().unwrap())),
            )
        };

        Ok(CKKSPlaintextCstZnx::new(re, im, CKKSMeta { log_delta, log_budget }))
    }
}

fn encode_const_coeff_i64(base2k: Base2K, k: usize, value: i64) -> Vec<i64> {
    let mut pt = LWEPlaintext::alloc(base2k, k.into());
    pt.encode_i64(value, k.into());
    (0..pt.size()).map(|limb| pt.data().at(0, limb)[0]).collect()
}

fn encode_const_coeff_i128(base2k: Base2K, k: usize, value: i128) -> Vec<i64> {
    let mut pt = LWEPlaintext::alloc(base2k, k.into());
    pt.encode_i128(value, k.into());
    (0..pt.size()).map(|limb| pt.data().at(0, limb)[0]).collect()
}

impl CKKSInfos for CKKSPlaintextCstZnx {
    fn meta(&self) -> CKKSMeta {
        self.meta
    }

    fn log_delta(&self) -> usize {
        self.meta.log_delta
    }

    fn log_budget(&self) -> usize {
        self.meta.log_budget
    }
}
