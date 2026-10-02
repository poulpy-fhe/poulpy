//! BSGS schedule parity with a small exact-integer policy.
//!
//! Core owns the schedule; schemes own arithmetic, scaling and key selection.
//! This policy stores exact integers in backend buffers, transfers them through
//! the backend API, and checks scheduling independently of CKKS arithmetic.

use crate::{
    BSGSOps, GLWEPolynomialEvaluation,
    layouts::{
        BSGSMeta, BabyStep, BackendGLWE, Base2K, Basis, Degree, GLWEBackendMut, GLWEBackendRef, GLWEInfos, GLWELayout,
        GLWETensorKeyPreparedBackendRef, GLWEToBackendMut, GLWEToBackendRef, GetTensorKey, LWEInfos, ModuleCoreAlloc, Parity,
        PowerBasis, Rank, SetBSGSMeta, TorusPrecision,
    },
    test_suite::parity::{ParityBackend, ParityShapes},
};
use anyhow::{Result, bail};
use poulpy_hal::{
    api::{ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Backend, Module, ScratchArena, ScratchOwned},
    test_suite::TestParams,
};
use std::cell::Cell;

struct Value<BE: Backend> {
    data: BackendGLWE<BE>,
    budget: usize,
    delta: usize,
}
impl<BE: Backend> GLWEToBackendRef<BE> for Value<BE> {
    fn to_backend_ref(&self) -> GLWEBackendRef<'_, BE> {
        <BackendGLWE<BE> as GLWEToBackendRef<BE>>::to_backend_ref(&self.data)
    }
}
impl<BE: Backend> GLWEToBackendMut<BE> for Value<BE> {
    fn to_backend_mut(&mut self) -> GLWEBackendMut<'_, BE> {
        <BackendGLWE<BE> as GLWEToBackendMut<BE>>::to_backend_mut(&mut self.data)
    }
    fn set_canonical(&mut self, canonical: bool) {
        <BackendGLWE<BE> as GLWEToBackendMut<BE>>::set_canonical(&mut self.data, canonical)
    }
}
impl<BE: Backend> LWEInfos for Value<BE> {
    fn n(&self) -> Degree {
        self.data.n()
    }
    fn k(&self) -> TorusPrecision {
        self.data.k()
    }
    fn base2k(&self) -> Base2K {
        self.data.base2k()
    }
    fn max_size(&self) -> usize {
        self.data.max_size()
    }
}
impl<BE: Backend> GLWEInfos for Value<BE> {
    fn rank(&self) -> Rank {
        self.data.rank()
    }
}
impl<BE: Backend> BSGSMeta for Value<BE> {
    fn bsgs_log_budget(&self) -> usize {
        self.budget
    }
    fn bsgs_log_delta(&self) -> usize {
        self.delta
    }
}
impl<BE: Backend> SetBSGSMeta for Value<BE> {
    fn set_bsgs_log_budget(&mut self, budget: usize) {
        self.budget = budget;
    }
    fn set_bsgs_log_delta(&mut self, delta: usize) {
        self.delta = delta;
    }
}
impl<BE: Backend> Value<BE> {
    fn read(&self, index: usize) -> i64 {
        let mut bytes = vec![0; BE::len_bytes(self.data.data.data())];
        BE::copy_to_host(self.data.data.data(), &mut bytes);
        i64::from_ne_bytes(bytes[index * 8..index * 8 + 8].try_into().unwrap())
    }
    fn write(&mut self, values: &[i64]) {
        BE::copy_from_host(self.data.data.data_mut(), bytemuck::cast_slice(values));
    }
}
fn value<BE: ParityBackend>(module: &Module<BE>, values: &[i64], budget: usize) -> Value<BE>
where
    Module<BE>: ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = i64>,
{
    let data = module.glwe_alloc_from_infos(&GLWELayout {
        n: Degree(values.len().next_power_of_two() as u32),
        base2k: Base2K(12),
        k: TorusPrecision(12),
        rank: Rank(0),
    });
    let mut value = Value { data, budget, delta: 3 };
    value.write(values);
    value
}
struct Step<BE: Backend> {
    value: Value<BE>,
    degree: usize,
}
impl<BE: Backend> BabyStep<BE> for Step<BE> {
    type Value = Value<BE>;
    fn degree(&self) -> usize {
        self.degree
    }
    fn get(&self) -> &Self::Value {
        &self.value
    }
    fn get_mut(&mut self) -> &mut Self::Value {
        &mut self.value
    }
}
struct NoKey;
impl<BE: Backend> GetTensorKey<BE> for NoKey {
    fn get_tensor_key(&self, _: TorusPrecision) -> crate::Result<GLWETensorKeyPreparedBackendRef<'_, BE>> {
        panic!("exact scalar policy must not request a cryptographic key")
    }
}
struct ExactOps {
    fused: bool,
    prepare_calls: Cell<usize>,
    fail: bool,
}
impl<BE: Backend> BSGSOps<BE, Value<BE>, Value<BE>, Value<BE>> for ExactOps {
    type Prepared = (i64, usize);
    fn init_accumulator(
        &self,
        _: &Module<BE>,
        res: &mut Value<BE>,
        seed: &Value<BE>,
        _: &mut ScratchArena<'_, BE>,
    ) -> Result<()> {
        res.write(&[0]);
        res.budget = seed.budget;
        res.delta = seed.delta;
        Ok(())
    }
    fn add_pt_const_assign(
        &self,
        _: &Module<BE>,
        res: &mut Value<BE>,
        res_coeff: usize,
        coeffs: &Value<BE>,
        idx: usize,
        _: &mut ScratchArena<'_, BE>,
    ) -> Result<()> {
        assert_eq!(res_coeff, 0);
        res.write(&[res.read(0) + coeffs.read(idx)]);
        Ok(())
    }
    fn eval_baby_linear_combination(
        &self,
        _: &Module<BE>,
        res: &mut Value<BE>,
        terms: &[(&Value<BE>, usize)],
        coeffs: &Value<BE>,
        _: &mut ScratchArena<'_, BE>,
    ) -> Result<bool> {
        if self.fail {
            bail!("policy sentinel");
        }
        if !self.fused {
            return Ok(false);
        }
        res.write(&[terms.iter().map(|(a, i)| a.read(0) * coeffs.read(*i)).sum()]);
        res.budget = terms.last().unwrap().0.budget;
        Ok(true)
    }
    fn mul_pt_const(
        &self,
        _: &Module<BE>,
        res: &mut Value<BE>,
        a: &Value<BE>,
        coeffs: &Value<BE>,
        idx: usize,
        _: &mut ScratchArena<'_, BE>,
    ) -> Result<()> {
        res.write(&[a.read(0) * coeffs.read(idx)]);
        res.budget = a.budget;
        Ok(())
    }
    fn mul_add_pt_const(
        &self,
        _: &Module<BE>,
        res: &mut Value<BE>,
        a: &Value<BE>,
        coeffs: &Value<BE>,
        idx: usize,
        _: &mut ScratchArena<'_, BE>,
    ) -> Result<()> {
        res.write(&[res.read(0) + a.read(0) * coeffs.read(idx)]);
        Ok(())
    }
    fn prepare_right(&self, _: &Module<BE>, a: &Value<BE>, _: &mut ScratchArena<'_, BE>) -> Result<Self::Prepared> {
        self.prepare_calls.set(self.prepare_calls.get() + 1);
        Ok((a.read(0), a.budget))
    }
    fn mul_prepared_assign<H: GetTensorKey<BE>>(
        &self,
        _: &Module<BE>,
        dst: &mut Value<BE>,
        prepared: &Self::Prepared,
        _: &H,
        _: &mut ScratchArena<'_, BE>,
    ) -> Result<()> {
        dst.write(&[dst.read(0) * prepared.0]);
        dst.budget = dst.budget.min(prepared.1) - 1;
        Ok(())
    }
    fn add_assign(&self, _: &Module<BE>, dst: &mut Value<BE>, a: &Value<BE>, _: &mut ScratchArena<'_, BE>) -> Result<()> {
        dst.write(&[dst.read(0) + a.read(0)]);
        dst.budget = dst.budget.min(a.budget);
        Ok(())
    }
    fn copy(&self, _: &Module<BE>, res: &mut Value<BE>, src: &Value<BE>, _: &mut ScratchArena<'_, BE>) -> Result<()> {
        res.write(&[src.read(0)]);
        res.budget = src.budget;
        res.delta = src.delta;
        Ok(())
    }
}
fn exercise<BE: ParityBackend>(module: &Module<BE>) -> Vec<(i64, usize, usize)>
where
    Module<BE>: ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = i64> + GLWEPolynomialEvaluation<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let mut scratch = ScratchOwned::<BE>::alloc(0); // The exact policy needs no arena storage.
    let coeffs = value(module, &[3, -2, 4, 1], 30);
    let mut basis = PowerBasis::new(Basis::Monomial, value(module, &[2], 29));
    for power in 2..=8 {
        basis.set_power(power, value(module, &[2_i64.pow(power as u32)], 30 - power));
    }
    let mut out = Vec::new();
    for fused in [false, true] {
        let ops = ExactOps {
            fused,
            prepare_calls: Cell::new(0),
            fail: false,
        };
        for (parity, expected, budget) in [(Parity::Full, 23, 27), (Parity::Even, 19, 28), (Parity::Odd, 4, 27)] {
            let mut res = value(module, &[-99], 1);
            module
                .glwe_eval_baby_step(&ops, &mut res, parity, &coeffs, &basis, &mut scratch.borrow())
                .unwrap();
            assert_eq!((res.read(0), res.budget), (expected, budget));
            out.push((res.read(0), res.budget, res.delta));
        }
        let mut constant = value(module, &[-99], 1);
        let constant_coeffs = value(module, &[3], 30);
        module
            .glwe_eval_baby_step(
                &ops,
                &mut constant,
                Parity::Full,
                &constant_coeffs,
                &basis,
                &mut scratch.borrow(),
            )
            .unwrap();
        assert_eq!((constant.read(0), constant.budget), (3, 29));
        ops.mul_pt_const(
            module,
            &mut constant,
            basis.get_stored(1).unwrap(),
            &coeffs,
            1,
            &mut scratch.borrow(),
        )
        .unwrap();
        assert_eq!((constant.read(0), constant.budget), (-4, 29));
        // Four degree-one blocks evaluate at x^2=4. The first level must
        // prepare x^2 once for both pairs, followed by one preparation of x^4.
        let mut steps: Vec<_> = (1..=4)
            .map(|i| Step {
                value: value(module, &[i], 29),
                degree: 1,
            })
            .collect();
        let mut res = value(module, &[-99], 1);
        module
            .glwe_eval_giant_steps(&ops, &mut res, &mut steps, &basis, &NoKey, &mut scratch.borrow())
            .unwrap();
        assert_eq!(res.read(0), 1 + 2 * 4 + 3 * 16 + 4 * 64);
        assert_eq!(ops.prepare_calls.get(), 2);
        out.push((res.read(0), res.budget, res.delta));
        let mut single = vec![Step {
            value: value(module, &[11], 21),
            degree: 0,
        }];
        module
            .glwe_eval_giant_steps(&ops, &mut res, &mut single, &basis, &NoKey, &mut scratch.borrow())
            .unwrap();
        assert_eq!((res.read(0), res.budget), (11, 21));
        assert_eq!(ops.prepare_calls.get(), 2, "single baby step must only copy");
        let mut empty: Vec<Step<BE>> = Vec::new();
        assert!(
            module
                .glwe_eval_giant_steps(&ops, &mut res, &mut empty, &basis, &NoKey, &mut scratch.borrow())
                .is_err()
        );
    }
    let ops = ExactOps {
        fused: false,
        prepare_calls: Cell::new(0),
        fail: true,
    };
    let mut res = value(module, &[-99], 1);
    let error = module
        .glwe_eval_baby_step(&ops, &mut res, Parity::Full, &coeffs, &basis, &mut scratch.borrow())
        .unwrap_err();
    assert!(error.to_string().contains("policy sentinel"));
    assert_eq!(res.read(0), -99);
    basis.take_power(3);
    assert!(
        module
            .glwe_eval_baby_step(&ops, &mut res, Parity::Full, &coeffs, &basis, &mut scratch.borrow())
            .is_err()
    );
    out
}
/// Both BSGS phases agree with the selected comparison backend and exact
/// integer expectations, including fused/fallback policies, parity, hoisting,
/// metadata, zero scratch, missing powers and policy-error propagation.
pub fn test_polynomial_evaluation_parity<BR: ParityBackend, BT: ParityBackend>(
    _: &TestParams,
    _: &ParityShapes,
    r: &Module<BR>,
    t: &Module<BT>,
) where
    Module<BR>: ModuleCoreAlloc<OwnedBuf = BR::OwnedBuf, ZnxWord = i64> + GLWEPolynomialEvaluation<BR>,
    Module<BT>: ModuleCoreAlloc<OwnedBuf = BT::OwnedBuf, ZnxWord = i64> + GLWEPolynomialEvaluation<BT>,
    ScratchOwned<BR>: ScratchOwnedAlloc<BR> + ScratchOwnedBorrow<BR>,
    ScratchOwned<BT>: ScratchOwnedAlloc<BT> + ScratchOwnedBorrow<BT>,
{
    assert_eq!(exercise(r), exercise(t));
}
