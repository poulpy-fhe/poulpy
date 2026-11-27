//! Coefficient-table and ciphertext-value references for a straight-line polynomial program.
use super::Parity;
use anyhow::{Result, ensure};
use std::collections::{BTreeMap, BTreeSet};

/// A value number refers to an input, then to the result of a preceding step.
/// A prepared right operand can only be read by a prepared product.
#[derive(Clone, Debug)]
pub enum Factor {
    /// Copy the value.
    Value(usize),
    /// Start a zero accumulator at the scale and budget of the given value.
    Zero(usize),
    Square(usize),
    Product(usize, usize),
    /// Prepare the value once for all its subsequent prepared products.
    PrepareRight(usize),
    /// Multiply the left ciphertext by the prepared right value.
    PreparedProduct(usize, usize),
    /// Sum terms in order. Each pair selects a value and a coefficient in `table`.
    /// The last term is the seed. An optional constant is added in BSGS baby-step order.
    WeightedSum {
        table: usize,
        terms: Vec<(usize, usize)>,
        constant: Option<usize>,
    },
}

/// An in-place update to the result of a factor. Constants are (table, coefficient) pairs.
#[derive(Clone, Copy, Debug)]
pub enum FactorUpdate {
    Add(usize),
    Sub(usize),
    AddConstant((usize, usize)),
    /// Add the product of a value and a coefficient in one operation.
    MulAddConstant(usize, (usize, usize)),
    /// Multiply the value by its real coefficient at accumulator width, multiply by i, and add.
    MulAddImaginaryConstant(usize, (usize, usize)),
    MulPow2(u32),
    /// Multiply by the imaginary unit in the standard ring.
    MulI,
}

/// A factor followed by its updates in evaluation order.
#[derive(Clone, Debug)]
pub struct FactorStep {
    pub factor: Factor,
    pub updates: Vec<FactorUpdate>,
}

impl FactorStep {
    pub fn new(factor: Factor, updates: impl Into<Vec<FactorUpdate>>) -> Self {
        Self {
            factor,
            updates: updates.into(),
        }
    }
}

pub(crate) struct GiantRun {
    pub(crate) power: usize,
    pub(crate) pairs: Vec<(usize, usize)>,
}

/// Adjacent equal-degree blocks merge into the high block; a trailing odd block takes
/// the previous merged degree. A right operand is shared only within its run at that level.
pub(crate) fn giant_runs(degrees: &[usize]) -> Result<Vec<GiantRun>> {
    ensure!(!degrees.is_empty(), "the polynomial has no baby step");
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

impl FactorStep {
    /// Lower the blocks of a BSGS polynomial to steps and the ordered powers they read.
    /// Lengths of real and imaginary coefficient tables alternate in a complex program.
    /// A trailing constant can fold into the output when its power is already available.
    #[doc(hidden)]
    pub fn compile_bsgs(
        degree: usize,
        parity: Parity,
        sizes: &[usize],
        complex: bool,
        fold: bool,
    ) -> Result<(Vec<Self>, Vec<usize>)> {
        let parts = if complex { 2 } else { 1 };
        ensure!(sizes.len().is_multiple_of(parts), "missing imaginary baby step");
        ensure!(
            !complex || sizes.chunks_exact(2).all(|pair| pair[0] == pair[1]),
            "real/imag baby-step lengths differ"
        );
        let degrees = sizes
            .iter()
            .step_by(parts)
            .map(|n| n.checked_sub(1).ok_or_else(|| anyhow::anyhow!("empty coefficient block")))
            .collect::<Result<Vec<_>>>()?;
        ensure!(
            !fold || (degrees.len() >= 2 && degrees.last() == Some(&0)),
            "expected a trailing constant block"
        );
        let count = degrees.len() - usize::from(fold);
        let runs = giant_runs(&degrees[..count])?;
        let terms = |degree| parity.baby_powers(degree);
        let mut degrees_needed = BTreeSet::from([1]);
        for &degree in &degrees[..count] {
            degrees_needed.extend(terms(degree));
        }
        degrees_needed.extend(runs.iter().map(|r| r.power));
        if fold {
            degrees_needed.insert(degree);
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
                let constant = (parity != Parity::Odd).then_some(0);
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
            let power = input_ids[&degree];
            updates.push(FactorUpdate::MulAddConstant(power, (count * parts, 0)));
            if complex {
                updates.push(FactorUpdate::MulAddImaginaryConstant(power, (count * parts + 1, 0)));
            }
        }
        let root = *blocks.last().ok_or_else(|| anyhow::anyhow!("missing final baby step"))?;
        emit(Factor::Value(root), updates);
        Ok((steps, inputs))
    }
}
