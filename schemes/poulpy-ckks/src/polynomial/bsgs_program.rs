//! Lower the baby steps, hoisted giant products and output fold into value references.
use anyhow::{Result, ensure};
use poulpy_core::layouts::{Basis, Parity, PolynomialInputTransform, SplitStrategy};
use poulpy_hal::layouts::Ring;
use std::collections::{BTreeMap, BTreeSet};

use super::{Factor, FactorStep, FactorUpdate, FactoredPolynomial};

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

struct GiantRun {
    power: usize,
    pairs: Vec<(usize, usize)>,
}

/// Adjacent equal-degree blocks merge into the high block; a trailing odd block takes
/// the previous merged degree. A right operand is shared only within its run at that level.
fn giant_runs(degrees: &[usize]) -> Result<Vec<GiantRun>> {
    let mut active: Vec<_> = degrees.iter().copied().enumerate().collect();
    let mut runs = Vec::new();
    while active.len() > 1 {
        let previous_len = active.len();
        let mut next: Vec<(usize, usize)> = Vec::with_capacity(previous_len.div_ceil(2));
        let mut pairs = Vec::new();
        let mut blocks = active.iter().copied().peekable();
        while let Some((low, d)) = blocks.next() {
            if let Some((high, _)) = blocks.next_if(|&(_, second)| second == d) {
                let power = d
                    .checked_add(1)
                    .and_then(usize::checked_next_power_of_two)
                    .ok_or_else(|| anyhow::anyhow!("giant power overflows"))?;
                pairs.push((power, low, high));
                next.push((
                    high,
                    power
                        .checked_add(power - 1)
                        .ok_or_else(|| anyhow::anyhow!("giant degree overflows"))?,
                ));
            } else {
                let degree = if blocks.peek().is_none() {
                    next.last().map_or(d, |&(_, p)| p)
                } else {
                    d
                };
                next.push((low, degree));
            }
        }
        ensure!(
            next.len() < previous_len || next != active,
            "baby-step degrees do not form a giant-step tree"
        );
        for group in pairs.chunk_by(|a, b| a.0 == b.0) {
            runs.push(GiantRun {
                power: group[0].0,
                pairs: group.iter().map(|&(_, a, b)| (a, b)).collect(),
            });
        }
        active = next;
    }
    Ok(runs)
}

/// Coefficient tables alternate real and imaginary blocks for a complex plan.
pub(crate) fn compile<Rg: Ring>(
    budget: BsgsBudget,
    sizes: &[usize],
    complex: bool,
    fold: bool,
) -> Result<(FactoredPolynomial<Rg>, Vec<usize>)> {
    let parts = if complex { 2 } else { 1 };
    let degrees = sizes
        .iter()
        .step_by(parts)
        .map(|n| n.checked_sub(1).ok_or_else(|| anyhow::anyhow!("empty coefficient block")))
        .collect::<Result<Vec<_>>>()?;
    let count = degrees.len() - usize::from(fold);
    let runs = giant_runs(&degrees[..count])?;
    let (first, stride) = match budget.parity {
        Parity::Full => (1, 1),
        Parity::Even => (2, 2),
        Parity::Odd => (1, 2),
    };
    let terms = |degree| (first..=degree).step_by(stride);
    let mut degrees_needed = BTreeSet::from([1]);
    for &degree in &degrees[..count] {
        degrees_needed.extend(terms(degree));
    }
    degrees_needed.extend(runs.iter().map(|r| r.power));
    if fold {
        degrees_needed.insert(budget.degree);
    }
    let inputs: Vec<_> = degrees_needed.into_iter().collect();
    let input_ids: BTreeMap<_, _> = inputs.iter().copied().enumerate().map(|(i, d)| (d, i)).collect();
    let mut steps = Vec::new();
    let mut emit = |f, updates| {
        let id = inputs.len() + steps.len();
        steps.push(FactorStep::new(f, updates));
        id
    };
    let mut blocks = Vec::with_capacity(count);
    for (i, &degree) in degrees[..count].iter().enumerate() {
        let mut baby = |table, imag| {
            let weights: Vec<_> = terms(degree).map(|d| (input_ids[&d], d)).collect();
            let constant = (budget.parity != Parity::Odd).then_some(0);
            let mut updates = Vec::new();
            let factor = if weights.is_empty() {
                if let Some(c) = constant {
                    updates.push(FactorUpdate::AddConstant((table, c)));
                }
                Factor::Zero(input_ids[&1])
            } else {
                Factor::WeightedSum {
                    table,
                    terms: weights,
                    constant,
                }
            };
            if imag {
                updates.push(FactorUpdate::MulI);
            }
            emit(factor, updates)
        };
        let real = baby(i * parts, false);
        let block = if complex {
            let imag = baby(i * parts + 1, true);
            emit(Factor::Value(real), vec![FactorUpdate::Add(imag)])
        } else {
            real
        };
        blocks.push(block);
    }
    for run in runs {
        let prepared = emit(Factor::PrepareRight(input_ids[&run.power]), vec![]);
        for (low, high) in run.pairs {
            blocks[high] = emit(
                Factor::PreparedProduct(blocks[high], prepared),
                vec![FactorUpdate::Add(blocks[low])],
            );
        }
    }
    let mut updates = Vec::new();
    if fold {
        let power = input_ids[&budget.degree];
        updates.push(FactorUpdate::MulAddConstant(power, (count * parts, 0)));
        if complex {
            updates.push(FactorUpdate::MulAddImaginaryConstant(power, (count * parts + 1, 0)));
        }
    }
    let root = *blocks.last().ok_or_else(|| anyhow::anyhow!("missing final baby step"))?;
    emit(Factor::Value(root), updates);
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
