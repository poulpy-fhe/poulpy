//! Evaluate `exp` on encrypted values.
//!
//! Run the example with the following command.
//!
//! ```text
//! cargo run --release --features ref --example exp_walkthrough
//! ```

mod common;

use anyhow::Result;
use poulpy_ckks::{CKKSInfos, polynomial::SplitStrategy, test_suite::helpers::upload_pt};
use poulpy_hal::api::ScratchOwnedBorrow;
use poulpy_libm::exp::{CKKSExpOps, ExpOptions, ExpPlan};

use common::{BASE2K, LOG_DELTA, Setup, coeff_layout, dense_interval, verify};

fn run() -> Result<()> {
    // 1. Create the reusable approximation before encrypting any values.
    // The first two arguments declare the supported input interval.
    // target_bits requests about 20 bits of approximation accuracy.
    let setup = Setup::new()?;
    let host_plan = ExpPlan::from_precision(
        -1.0,
        1.0,
        BASE2K.into(),
        coeff_layout(),
        ExpOptions {
            target_bits: 20.0,
            max_degree: 15,
            reduction_steps: None,
            strategy: SplitStrategy::MinDepth,
        },
        &setup.host_module,
    )?;
    let consumed = host_plan.consumed_bits(LOG_DELTA);

    // 2. Prepare the plan for encrypted evaluation.
    // The setup helper also creates the multiplication key and allocates the temporary memory reported by the operation.
    let plan = host_plan.map_plaintexts(|pt| upload_pt(&setup.module, pt));
    let mut context = setup.prepare(consumed, |module, sizing, params| {
        module.ckks_exp_tmp_bytes(sizing, &params.tsk_layout(), &plan)
    });

    // 3. Pack many values into one ciphertext and encrypt them together.
    // This example fills the ciphertext with evenly spaced values from -1 to 1.
    let input = dense_interval(-1.0, 1.0);
    let encrypted = context.encrypt(&input);
    let input_budget = encrypted.log_budget();

    // 4. Evaluate exp without decrypting the input.
    // The result is written to a separate ciphertext.
    let mut output = context.output();
    context.module.ckks_exp(
        &mut output,
        &encrypted,
        &plan,
        &context.tsk,
        &mut context.scratch.borrow(),
    )?;
    assert_eq!(input_budget - output.log_budget(), consumed);

    // 5. Decrypt the result after the encrypted computation is complete.
    // Verification uses Poulpy's precision helper instead of an arbitrary error threshold.
    let output_log_delta = output.log_delta();
    let got = context.decrypt(&output);
    let want: Vec<f64> = input.iter().map(|x| x.exp()).collect();
    verify("exp", &got, &want, output_log_delta);
    println!("exp({:.3}) = {:.6}", input[0], got[0]);

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
