//! A shared square in both factors of (0.75 x² + 1)(x² − 0.25).
use super::*;
use crate::{
    api::CKKSFactoredPolynomialEvaluationOps,
    polynomial::{Factor, FactorStep, FactorUpdate, FactoredPolynomial},
};

pub fn test_factored_polynomial<BE, F, Rg: Ring>(
    params: CKKSTestParams,
    module: &Module<BE>,
    host: &Module<HostBytesBackend>,
) -> Result<()>
where
    BE: TestContextBackend,
    for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    F: TestScalar,
    Module<BE>: TestContextSharedModule<BE, Rg> + CKKSEncodingOps<BE, F, Rg> + CKKSFactoredPolynomialEvaluationOps<BE, Rg>,
{
    use Factor::{PrepareRight, PreparedProduct, Square, Value, WeightedSum};
    use FactorUpdate::AddConstant;
    let plan = FactoredPolynomial::<Rg>::new(
        1,
        vec![2, 1],
        vec![
            FactorStep::new(Square(0), []),
            FactorStep::new(
                WeightedSum {
                    table: 0,
                    terms: vec![(1, 0)],
                    constant: Some(1),
                },
                [],
            ),
            FactorStep::new(Value(1), [AddConstant((1, 0))]),
            FactorStep::new(PrepareRight(3), []),
            FactorStep::new(PreparedProduct(2, 4), []),
        ],
    )?;
    let quarter = F::from_f64(0.25).unwrap();
    let coefficients: &[&[F]] = &[&[quarter + quarter + quarter, F::one()], &[-quarter]];
    for slots in [SlotLayout::Full, SlotLayout::Replicated(layout())] {
        let mut setup = polynomial_setup::<BE, F, Rg>(&params, module, host, slots)?;
        let before = snapshot::<BE, _, Rg>(&setup.x);
        let base = params.base2k.into();
        let meta = setup.coeff_meta;
        let inputs = [&setup.x];
        macro_rules! evaluate {
            ($poly:expr) => {{
                let encoded = $poly;
                let rebound = plan.bind(vec![encoded.coefficients(0).unwrap(), encoded.coefficients(1).unwrap()])?;
                let bytes = module.ckks_eval_poly_factored_tmp_bytes(&setup.out, &inputs, &rebound, &params.tsk_layout());
                with_scratch::<BE, _>(bytes, |s| {
                    module.ckks_eval_poly_factored(&mut setup.out, &inputs, &rebound, &setup.tsk, s)
                })?;
                let result = snapshot::<BE, _, Rg>(&setup.out);
                let missing: &[&CKKSCiphertextOwned<BE, Rg>] = &[];
                with_scratch::<BE, _>(0, |s| {
                    module.ckks_eval_poly_factored(&mut setup.out, missing, &rebound, &setup.tsk, s)
                })
                .unwrap_err();
                assert_eq!(result, snapshot::<BE, _, Rg>(&setup.out));
            }};
        }
        match slots {
            SlotLayout::Full => evaluate!(plan.encode(module, base, meta, coefficients, &mut setup.encoding.borrow())?),
            SlotLayout::Replicated(layout) => {
                evaluate!(plan.encode_replicated(module, base, meta, layout, coefficients, &mut setup.encoding.borrow())?)
            }
        }
        assert_eq!(before, snapshot::<BE, _, Rg>(&setup.x));
        assert_eq!(setup.out.log_delta(), setup.x.log_delta());
        assert_eq!(setup.out.slot_layout(), slots);
        assert_eq!(
            setup.x.log_budget() - setup.out.log_budget(),
            plan.required_budget_bits(setup.x.log_delta(), meta.log_delta())
        );
        let got = decrypt_batch::<BE, F, Rg>(&params, module, &setup.sk, &setup.out, setup.xs.len());
        let expected: Vec<_> = setup
            .xs
            .iter()
            .map(|&x| {
                let square = x * x;
                ((quarter + quarter + quarter) * square + F::one()) * (square - quarter)
            })
            .collect();
        // The full-layout product loses precision to encryption and three scaled products.
        let bits = precision((&got.0, &got.1), (&expected, &vec![F::zero(); expected.len()]));
        assert!(
            bits >= (LOG_DELTA / 2) as f64,
            "factored polynomial on {slots:?}: {bits} bits"
        );
    }
    Ok(())
}
