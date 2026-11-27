//! A polynomial as a checked sequence of CKKS operations on shared ciphertexts.
use std::{any::TypeId, marker::PhantomData};

use anyhow::{Result, ensure};
use poulpy_core::layouts::Base2K;
use poulpy_hal::layouts::{Backend, Module, Ring, ScratchArena, Standard};

use crate::{
    CKKSModuleInfos, CoeffsMeta, ReplicatedLayout,
    api::{CKKSEncodingOps, CKKSEncodingScalar},
    layouts::{
        CKKSPlaintextAlloc, CKKSPlaintextOwned, MaskedCoeffs, eval_mod::encode_scalar_constants, masked::encode_masked_constants,
    },
};

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

/// A ring-specific program. Values `0..inputs` are borrowed ciphertexts.
/// Step `i` defines value `inputs + i`; the final ciphertext is the result.
#[derive(Clone, Debug)]
pub struct FactoredPolynomial<Rg: Ring = Standard> {
    pub(crate) inputs: usize,
    pub(crate) table_sizes: Vec<usize>,
    pub(crate) steps: Vec<FactorStep>,
    pub(crate) reuse: Vec<Option<usize>>,
    pub(crate) release: Vec<Vec<usize>>,
    pub(crate) temporaries: usize,
    ring: PhantomData<Rg>,
}

impl<Rg: Ring> FactoredPolynomial<Rg> {
    /// Validates references, table bounds and operand kinds and computes the last use of each value.
    pub fn new(inputs: usize, table_sizes: Vec<usize>, steps: Vec<FactorStep>) -> Result<Self> {
        ensure!(inputs > 0 && !steps.is_empty(), "a program needs inputs and an output step");
        ensure!(table_sizes.iter().all(|&n| n > 0), "empty coefficient table");
        ensure!(inputs.checked_add(steps.len()).is_some(), "program value count overflows");
        let mut last_use: Vec<usize> = (0..steps.len()).collect();
        let is_prepared = |v: usize| v >= inputs && matches!(steps[v - inputs].factor, Factor::PrepareRight(_));
        let mut candidates = Vec::with_capacity(steps.len());
        let coefficient = |table: usize, idx: usize| {
            ensure!(
                table_sizes.get(table).is_some_and(|&n| idx < n),
                "missing coefficient ({table}, {idx})"
            );
            Ok(())
        };
        for (i, step) in steps.iter().enumerate() {
            let mut uses = Vec::new();
            let mut read = |v: usize, prepared: bool| {
                ensure!(v < inputs || v - inputs < i, "step {i} reads unavailable value {v}");
                ensure!(is_prepared(v) == prepared, "wrong operand kind for value {v} in step {i}");
                if v >= inputs {
                    last_use[v - inputs] = i;
                }
                uses.push(v);
                Ok(())
            };
            let candidate = match &step.factor {
                Factor::Value(v) | Factor::Square(v) => {
                    read(*v, false)?;
                    Some(*v)
                }
                Factor::Zero(v) => {
                    read(*v, false)?;
                    None
                }
                Factor::Product(a, b) | Factor::PreparedProduct(a, b) => {
                    read(*a, false)?;
                    read(*b, matches!(step.factor, Factor::PreparedProduct(..)))?;
                    Some(*a)
                }
                Factor::PrepareRight(v) => {
                    read(*v, false)?;
                    ensure!(
                        step.updates.is_empty() && i + 1 < steps.len(),
                        "a prepared operand cannot be updated or returned"
                    );
                    None
                }
                Factor::WeightedSum { table, terms, constant } => {
                    ensure!(!terms.is_empty(), "empty weighted sum");
                    for &(v, c) in terms {
                        read(v, false)?;
                        coefficient(*table, c)?;
                    }
                    if let Some(c) = constant {
                        coefficient(*table, *c)?;
                    }
                    None
                }
            };
            for update in &step.updates {
                match *update {
                    FactorUpdate::Add(v) | FactorUpdate::Sub(v) => read(v, false)?,
                    FactorUpdate::MulAddConstant(v, (t, c)) => {
                        read(v, false)?;
                        coefficient(t, c)?;
                    }
                    FactorUpdate::AddConstant((t, c)) => coefficient(t, c)?,
                    FactorUpdate::MulPow2(bits) => ensure!(i32::try_from(bits).is_ok(), "exponent exceeds the signed range"),
                    FactorUpdate::MulI => ensure!(
                        TypeId::of::<Rg>() == TypeId::of::<Standard>(),
                        "the imaginary unit requires the standard ring"
                    ),
                }
            }
            // An operand read by an update must still contain its value after the initial factor.
            candidates.push(candidate.filter(|&v| v >= inputs && uses.iter().filter(|&&u| u == v).count() == 1));
        }
        let mut release = vec![Vec::new(); steps.len()];
        for (i, &last) in last_use.iter().enumerate().take(steps.len() - 1) {
            release[last].push(inputs + i);
        }
        let mut temporaries = 0;
        let mut live = 0;
        for (i, reuse) in candidates.iter_mut().enumerate() {
            *reuse = reuse.filter(|&v| last_use[v - inputs] == i);
            if !matches!(steps[i].factor, Factor::PrepareRight(_)) && reuse.is_none() {
                live += 1;
            }
            temporaries = temporaries.max(live);
            live -= release[i].iter().filter(|&&v| Some(v) != *reuse && !is_prepared(v)).count();
        }
        Ok(Self {
            inputs,
            table_sizes,
            steps,
            reuse: candidates,
            release,
            temporaries,
            ring: PhantomData,
        })
    }

    /// Longest dependency cost with unit scales, counting encoded coefficient products.
    pub fn eval_depth(&self) -> usize {
        self.required_budget_bits(1, 1)
    }

    /// Budget above the output scale for equally scaled inputs and coefficient tables.
    pub fn required_budget_bits(&self, input_log_delta: usize, coefficient_log_delta: usize) -> usize {
        self.budget_cost(vec![0; self.inputs], input_log_delta, |_| coefficient_log_delta)
    }

    pub(crate) fn budget_cost(&self, mut used: Vec<usize>, delta: usize, scale: impl Fn(usize) -> usize) -> usize {
        for step in &self.steps {
            let mut cost = match &step.factor {
                Factor::Value(v) | Factor::Zero(v) | Factor::PrepareRight(v) => used[*v],
                Factor::Square(v) => used[*v].saturating_add(delta),
                Factor::Product(a, b) | Factor::PreparedProduct(a, b) => used[*a].max(used[*b]).saturating_add(delta),
                Factor::WeightedSum { table, terms, .. } => terms
                    .iter()
                    .map(|&(v, _)| used[v])
                    .max()
                    .unwrap_or(0)
                    .saturating_add(scale(*table)),
            };
            for u in &step.updates {
                match *u {
                    FactorUpdate::Add(v) | FactorUpdate::Sub(v) => cost = cost.max(used[v]),
                    FactorUpdate::MulAddConstant(v, (table, _)) => cost = cost.max(used[v].saturating_add(scale(table))),
                    _ => {}
                }
            }
            used.push(cost);
        }
        used.into_iter().max().unwrap_or(0)
    }

    /// Bind already encoded plaintext tables. The program and plaintexts are borrowed for repeated evaluations.
    pub fn bind<'a, P>(&'a self, coefficients: Vec<&'a P>) -> Result<EncodedFactoredPolynomial<'a, P, Rg>> {
        ensure!(
            coefficients.len() == self.table_sizes.len(),
            "wrong number of coefficient tables"
        );
        Ok(EncodedFactoredPolynomial {
            plan: self,
            tables: Tables::Borrowed(coefficients),
        })
    }

    /// Encode each coefficient table once on the destination backend.
    pub fn encode<'a, BE: Backend, F: CKKSEncodingScalar>(
        &'a self,
        module: &Module<BE>,
        base2k: Base2K,
        meta: CoeffsMeta,
        values: &[&[F]],
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<EncodedFactoredPolynomial<'a, CKKSPlaintextOwned<BE, Rg>, Rg>>
    where
        Module<BE>: CKKSPlaintextAlloc<BE> + CKKSEncodingOps<BE, F, Rg>,
    {
        self.encode_with(base2k, meta, values, |v| {
            encode_scalar_constants(module, base2k, meta, v, scratch)
        })
    }

    /// Encode tables with the masked constants of a replicated slot layout.
    pub fn encode_replicated<'a, BE: Backend, F: CKKSEncodingScalar>(
        &'a self,
        module: &Module<BE>,
        base2k: Base2K,
        meta: CoeffsMeta,
        layout: ReplicatedLayout,
        values: &[&[F]],
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<EncodedFactoredPolynomial<'a, MaskedCoeffs<CKKSPlaintextOwned<BE, Rg>>, Rg>>
    where
        Module<BE>: CKKSPlaintextAlloc<BE> + CKKSEncodingOps<BE, F, Rg> + CKKSModuleInfos,
    {
        self.encode_with(base2k, meta, values, |v| {
            encode_masked_constants(module, base2k, meta, layout, v, scratch)
        })
    }

    fn encode_with<P, F: CKKSEncodingScalar>(
        &self,
        base2k: Base2K,
        meta: CoeffsMeta,
        values: &[&[F]],
        encode: impl FnMut(&[F]) -> Result<P>,
    ) -> Result<EncodedFactoredPolynomial<'_, P, Rg>> {
        ensure!(
            base2k.as_usize() > 0 && meta.k.as_usize() > 0,
            "zero coefficient radix or width"
        );
        ensure!(
            self.table_sizes.iter().copied().eq(values.iter().map(|v| v.len())),
            "coefficient table lengths differ"
        );
        ensure!(
            values.iter().all(|v| v.iter().all(|c| c.is_finite())),
            "nonfinite coefficient"
        );
        Ok(EncodedFactoredPolynomial {
            plan: self,
            tables: Tables::Owned(values.iter().copied().map(encode).collect::<Result<_>>()?),
        })
    }
}

enum Tables<'a, P> {
    Borrowed(Vec<&'a P>),
    Owned(Vec<P>),
}

/// A validated program with its borrowed or owned backend coefficient tables.
pub struct EncodedFactoredPolynomial<'a, P, Rg: Ring = Standard> {
    pub(crate) plan: &'a FactoredPolynomial<Rg>,
    tables: Tables<'a, P>,
}

impl<P, Rg: Ring> EncodedFactoredPolynomial<'_, P, Rg> {
    pub fn plan(&self) -> &FactoredPolynomial<Rg> {
        self.plan
    }

    /// The backend plaintext table at index `i`.
    pub fn coefficients(&self, i: usize) -> Option<&P> {
        match &self.tables {
            Tables::Borrowed(v) => v.get(i).copied(),
            Tables::Owned(v) => v.get(i),
        }
    }

    pub(crate) fn table(&self, i: usize) -> &P {
        match &self.tables {
            Tables::Borrowed(v) => v[i],
            Tables::Owned(v) => &v[i],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use poulpy_hal::layouts::ConjugateInvariant;

    #[test]
    fn shared_right_operand_lives_until_both_products() -> Result<()> {
        use Factor::*;
        let program = FactoredPolynomial::<Standard>::new(
            2,
            vec![2],
            vec![
                FactorStep::new(PrepareRight(1), []),
                FactorStep::new(Square(0), []),
                FactorStep::new(PreparedProduct(3, 2), []),
                FactorStep::new(
                    PreparedProduct(0, 2),
                    [FactorUpdate::Sub(4), FactorUpdate::MulAddConstant(1, (0, 1))],
                ),
            ],
        )?;
        assert_eq!(program.reuse[2], Some(3));
        assert_eq!(program.temporaries, 2);
        assert!(program.release[3].contains(&2));
        assert_eq!(program.required_budget_bits(35, 18), 70);
        Ok(())
    }

    #[test]
    fn rejects_invalid_references_and_imaginary_ci_programs() {
        use Factor::*;
        for steps in [
            vec![FactorStep::new(Square(1), [])],
            vec![FactorStep::new(Value(0), [FactorUpdate::AddConstant((1, 0))])],
            vec![FactorStep::new(Value(0), [FactorUpdate::MulAddConstant(0, (0, 1))])],
            vec![FactorStep::new(PreparedProduct(0, 0), [])],
            vec![FactorStep::new(PrepareRight(0), []), FactorStep::new(Value(1), [])],
            vec![FactorStep::new(PrepareRight(0), [])],
        ] {
            assert!(FactoredPolynomial::<Standard>::new(1, vec![1], steps).is_err());
        }
        let steps = vec![FactorStep::new(Value(0), [FactorUpdate::MulI])];
        assert!(FactoredPolynomial::<ConjugateInvariant>::new(1, vec![], steps.clone()).is_err());
        assert!(FactoredPolynomial::<Standard>::new(1, vec![], steps).is_ok());
    }
}
