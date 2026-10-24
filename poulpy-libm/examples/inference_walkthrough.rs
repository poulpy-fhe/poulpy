//! Turn encrypted model scores into probabilities.
//!
//! Run the example with the following command.
//!
//! ```text
//! cargo run --release --features ref --example inference_walkthrough
//! ```

mod common;

use anyhow::Result;
use poulpy_ckks::{
    CKKSInfos, api::CKKSAddOps, polynomial::SplitStrategy, test_suite::helpers::upload_pt,
};
use poulpy_hal::api::ScratchOwnedBorrow;
use poulpy_libm::{
    arithmetic::CKKSArithmeticOps,
    hyperbolic::{CKKSHyperbolicOps, HyperbolicOptions, TanhPlan},
};

use common::{BASE2K, LOG_DELTA, SLOTS, Setup, coeff_layout, dense_interval, verify};

fn run() -> Result<()> {
    // A binary classifier commonly converts its score with sigmoid(x).
    // The identity sigmoid(x) = (tanh(x / 2) + 1) / 2 gives a short CKKS circuit using one nonlinear plan and exact power-of-two scaling.
    let setup = Setup::new()?;
    let host_tanh = TanhPlan::from_precision(
        -1.0,
        1.0,
        BASE2K.into(),
        coeff_layout(),
        HyperbolicOptions {
            target_bits: 20.0,
            max_degree: 15,
            strategy: SplitStrategy::MinDepth,
        },
        &setup.host_module,
    )?;
    let half_score_log_delta = LOG_DELTA + 1;
    let tanh_consumed = host_tanh.consumed_bits(half_score_log_delta);

    // Each division by two consumes one modulus bit.
    // Add-one is a linear Poulpy operation and does not consume modulus.
    let circuit_consumed = tanh_consumed + 2;
    let tanh = host_tanh.map_plaintexts(|pt| upload_pt(&setup.module, pt));
    let mut context = setup.prepare(circuit_consumed, |module, sizing, params| {
        module
            .ckks_tanh_tmp_bytes(sizing, &params.tsk_layout(), &tanh)
            .max(module.ckks_scalbn_tmp_bytes())
    });

    // One ciphertext carries an entire batch of independent model scores.
    // The declared tanh interval supports scores from -2 to 2 after the first division by two.
    let scores = dense_interval(-2.0, 2.0);
    let encrypted_scores = context.encrypt(&scores);
    let input_budget = encrypted_scores.log_budget();

    // First compute x / 2 without decrypting.
    // Poulpy records the division by moving one bit from the budget to log_delta, so the tanh cost must use the adjusted scale.
    let mut half_scores = context.output();
    context.module.ckks_scalbn(
        &mut half_scores,
        &encrypted_scores,
        -1,
        &mut context.scratch.borrow(),
    )?;
    assert_eq!(input_budget - half_scores.log_budget(), 1);

    // Evaluate tanh on every packed score with the same prepared plan.
    let mut probabilities = context.output();
    context.module.ckks_tanh(
        &mut probabilities,
        &half_scores,
        &tanh,
        &context.tsk,
        &mut context.scratch.borrow(),
    )?;
    assert_eq!(
        half_scores.log_budget() - probabilities.log_budget(),
        tanh_consumed
    );

    // Finish the identity with Poulpy's exact add-one path and libm-style power-of-two scaling.
    let tanh_budget = probabilities.log_budget();
    context
        .module
        .ckks_add_one_assign(&mut probabilities, &mut context.scratch.borrow())?;
    assert_eq!(tanh_budget, probabilities.log_budget());
    let mut output = context.output();
    context.module.ckks_scalbn(
        &mut output,
        &probabilities,
        -1,
        &mut context.scratch.borrow(),
    )?;
    assert_eq!(probabilities.log_budget() - output.log_budget(), 1);
    assert_eq!(input_budget - output.log_budget(), circuit_consumed);

    // Decryption happens only after the complete inference circuit.
    // Poulpy's precision helper checks every packed result against ordinary sigmoid.
    let output_log_delta = output.log_delta();
    let got = context.decrypt(&output);
    let want: Vec<f64> = scores
        .iter()
        .map(|score| 1.0 / (1.0 + (-score).exp()))
        .collect();
    verify("inference/sigmoid", &got, &want, output_log_delta);
    for index in [0, SLOTS / 2, SLOTS - 1] {
        println!("score={:.3}, probability={:.6}", scores[index], got[index]);
    }

    Ok(())
}

fn main() -> Result<()> {
    run()
}

#[cfg(test)]
mod tests {
    #[test]
    fn walkthrough_runs() {
        super::run().unwrap();
    }
}
