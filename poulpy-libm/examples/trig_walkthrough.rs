//! Evaluate `sin` and `cos` on the same encrypted values.
//!
//! Run the example with the following command.
//!
//! ```text
//! cargo run --release --features ref --example trig_walkthrough
//! ```

mod common;

use anyhow::Result;
use poulpy_ckks::{CKKSInfos, polynomial::SplitStrategy, test_suite::helpers::upload_pt};
use poulpy_hal::api::ScratchOwnedBorrow;
use poulpy_libm::trig::{CKKSTrigOps, CosPlan, SinPlan, TrigOptions};

use common::{BASE2K, LOG_DELTA, Setup, coeff_layout, dense_interval, verify};

fn run() -> Result<()> {
    // 1. Create one reusable plan for each function before encryption.
    // Both plans accept values from -1 to 1 and request the same accuracy.
    let setup = Setup::new()?;
    let options = TrigOptions {
        target_bits: 20.0,
        max_degree: 15,
        strategy: SplitStrategy::MinDepth,
    };
    let host_sin = SinPlan::from_precision(
        -1.0,
        1.0,
        BASE2K.into(),
        coeff_layout(),
        options,
        &setup.host_module,
    )?;
    let host_cos = CosPlan::from_precision(
        -1.0,
        1.0,
        BASE2K.into(),
        coeff_layout(),
        options,
        &setup.host_module,
    )?;
    let sin_consumed = host_sin.consumed_bits(LOG_DELTA);
    let cos_consumed = host_cos.consumed_bits(LOG_DELTA);

    // 2. Prepare both plans for encrypted evaluation.
    // sincos reuses one temporary memory area, so the larger reported size is sufficient.
    let sin_plan = host_sin.map_plaintexts(|pt| upload_pt(&setup.module, pt));
    let cos_plan = host_cos.map_plaintexts(|pt| upload_pt(&setup.module, pt));
    let max_consumed = sin_consumed.max(cos_consumed);
    let mut context = setup.prepare(max_consumed, |module, sizing, params| {
        module
            .ckks_sin_tmp_bytes(sizing, &params.tsk_layout(), &sin_plan)
            .max(module.ckks_cos_tmp_bytes(sizing, &params.tsk_layout(), &cos_plan))
    });

    // 3. Pack and encrypt the input once because both functions use the same values.
    // Every encrypted value must stay inside the interval selected above.
    let input = dense_interval(-1.0, 1.0);
    let encrypted = context.encrypt(&input);
    let input_budget = encrypted.log_budget();
    let mut sin_output = context.output();
    let mut cos_output = context.output();

    // 4. Evaluate both functions without decrypting the input.
    // Each result is written to its own ciphertext.
    context.module.ckks_sincos(
        &mut sin_output,
        &mut cos_output,
        &encrypted,
        &sin_plan,
        &cos_plan,
        &context.tsk,
        &mut context.scratch.borrow(),
    )?;
    assert_eq!(input_budget - sin_output.log_budget(), sin_consumed);
    assert_eq!(input_budget - cos_output.log_budget(), cos_consumed);

    // 5. Decrypt both results and verify them with Poulpy's precision helper.
    let sin_log_delta = sin_output.log_delta();
    let cos_log_delta = cos_output.log_delta();
    let got_sin = context.decrypt(&sin_output);
    let got_cos = context.decrypt(&cos_output);
    let want_sin: Vec<f64> = input.iter().map(|x| x.sin()).collect();
    let want_cos: Vec<f64> = input.iter().map(|x| x.cos()).collect();
    verify("sin", &got_sin, &want_sin, sin_log_delta);
    verify("cos", &got_cos, &want_cos, cos_log_delta);
    println!(
        "x={:.3}, sin(x)={:.6}, cos(x)={:.6}",
        input[0], got_sin[0], got_cos[0],
    );

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
