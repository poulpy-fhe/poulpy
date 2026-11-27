//! Lower the baby steps, hoisted giant products and output fold into value references.
use anyhow::Result;
use poulpy_core::layouts::{Basis, Parity, PolynomialInputTransform, SplitStrategy};
use poulpy_hal::layouts::Ring;

use super::{FactorStep, FactoredPolynomial};

/// The source polynomial's budget includes the powers supplied by the caller and the one-shot input transform.
#[derive(Clone, Copy, Debug)]
pub(crate) struct BsgsBudget {
    pub(crate) op: &'static str,
    pub(crate) degree: usize,
    pub(crate) strategy: SplitStrategy,
    pub(crate) parity: Parity,
    pub(crate) basis: Basis,
    pub(crate) transform: PolynomialInputTransform,
}

impl BsgsBudget {
    pub(super) fn required(self, delta: usize, coeff: usize) -> usize {
        let levels = match self.transform {
            PolynomialInputTransform::Identity => 0,
            PolynomialInputTransform::ChebyshevT2 | PolynomialInputTransform::Square => 1,
            PolynomialInputTransform::ChebyshevT2TimesInput | PolynomialInputTransform::SquareTimesInput => 2,
        };
        poulpy_core::layouts::bsgs_consumed_bits(self.degree, self.strategy, self.parity, self.basis, delta, coeff)
            + levels * delta
    }
}

/// Coefficient tables alternate real and imaginary blocks for a complex plan.
pub(crate) fn compile<Rg: Ring>(
    budget: BsgsBudget,
    sizes: &[usize],
    complex: bool,
    fold: bool,
) -> Result<(FactoredPolynomial<Rg>, Vec<usize>)> {
    let (steps, inputs) = FactorStep::compile_bsgs(budget.degree, budget.parity, sizes, complex, fold)?;
    let mut program = FactoredPolynomial::new(inputs.len(), sizes.to_vec(), steps)?;
    program.bsgs = Some(budget);
    Ok((program, inputs))
}

#[cfg(test)]
mod tests {
    use super::*;
    use poulpy_core::layouts::{BSGSPolynomial, Polynomial};
    use poulpy_hal::layouts::Standard;

    fn compare_budget(poly: &BSGSPolynomial<usize>, strategy: SplitStrategy) -> Result<()> {
        let budget = BsgsBudget {
            op: "test",
            degree: poly.degree(),
            strategy,
            parity: poly.parity(),
            basis: poly.basis(),
            transform: poly.input_transform(),
        };
        let sizes = poly.baby_steps();
        let can_fold = sizes.len() >= 2 && sizes.last() == Some(&1);
        for fold in [false, true].into_iter().take(1 + usize::from(can_fold)) {
            let (program, powers) = compile::<Standard>(budget, sizes, false, fold)?;
            assert_eq!(program.input_count(), powers.len());
            for (input, coefficient) in [(35, 20), (35, 35), (25, 50)] {
                assert_eq!(
                    program.required_budget_bits(input, coefficient),
                    poly.consumed_bits(input, coefficient)
                );
            }
        }
        Ok(())
    }

    #[test]
    fn compiled_budgets_match_bsgs_including_folded_inputs() -> Result<()> {
        for strategy in [SplitStrategy::MinDepth, SplitStrategy::MinMult] {
            for basis in [Basis::Monomial, Basis::Chebyshev] {
                for degree in 1..=64 {
                    let parity = if degree % 2 == 0 { Parity::Even } else { Parity::Odd };
                    for parity in [Parity::Full, parity] {
                        let coefficients = (0..=degree)
                            .map(|i| match (parity, i % 2) {
                                (Parity::Even, 1) | (Parity::Odd, 0) => 0.0,
                                _ => 1.0,
                            })
                            .collect::<Vec<f64>>();
                        let polynomial = Polynomial::new_with_parity(basis, coefficients, parity);
                        compare_budget(&polynomial.decompose_bsgs_with(strategy, |c| Ok(c.len()))?, strategy)?;
                        if parity != Parity::Full && degree >= 2 {
                            compare_budget(&polynomial.decompose_bsgs_folded_with(strategy, |c| Ok(c.len()))?, strategy)?;
                        }
                    }
                }
            }
        }
        Ok(())
    }
}
