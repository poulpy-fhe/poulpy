//! Compute log power from encrypted I/Q samples.
//!
//! Run the example with the following command.
//!
//! ```text
//! cargo run --release --features ref --example signal_walkthrough
//! ```

mod common;

use anyhow::Result;
use poulpy_ckks::{CKKSInfos, polynomial::SplitStrategy, test_suite::helpers::upload_pt};
use poulpy_hal::api::ScratchOwnedBorrow;
use poulpy_libm::{
    arithmetic::CKKSArithmeticOps,
    log::{CKKSLogOps, LogOptions, LogPlan},
    roots::{CKKSRootOps, HypotPlan, RootOptions},
};

use common::{BASE2K, LOG_DELTA, Setup, coeff_layout, dense_interval, verify};

fn run() -> Result<()> {
    // A receiver can represent each complex sample with encrypted in-phase and quadrature components.
    // Log power is ln(I^2 + Q^2), which can be composed as 2 * ln(hypot(I, Q)).
    let setup = Setup::new()?;
    let options = RootOptions {
        target_bits: 20.0,
        max_degree: 31,
        strategy: SplitStrategy::MinDepth,
    };
    let host_hypot = HypotPlan::from_precision(
        -1.0,
        -0.5,
        0.5,
        1.0,
        BASE2K.into(),
        coeff_layout(),
        options,
        &setup.host_module,
    )?;

    // The component intervals imply a magnitude between sqrt(0.5) and sqrt(2).
    // A slightly wider logarithm interval leaves clear room for approximation error at the boundary.
    let host_log = LogPlan::from_precision(
        0.7,
        1.5,
        BASE2K.into(),
        coeff_layout(),
        LogOptions {
            target_bits: 20.0,
            max_degree: 31,
            strategy: SplitStrategy::MinDepth,
        },
        &setup.host_module,
    )?;
    let hypot_consumed = host_hypot.consumed_bits(LOG_DELTA);
    let log_consumed = host_log.consumed_bits(LOG_DELTA);
    let circuit_consumed = hypot_consumed + log_consumed;

    // Transfer both plans once and reserve enough scratch for the larger stage.
    let hypot = host_hypot.map_plaintexts(|pt| upload_pt(&setup.module, pt));
    let log = host_log.map_plaintexts(|pt| upload_pt(&setup.module, pt));
    let mut context = setup.prepare(circuit_consumed, |module, sizing, params| {
        module
            .ckks_hypot_tmp_bytes(sizing, &params.tsk_layout(), &hypot)
            .max(module.ckks_log_tmp_bytes(sizing, &params.tsk_layout(), &log))
            .max(module.ckks_scalbn_tmp_bytes())
    });

    // The two vectors are encrypted separately because they are independent input channels.
    // Every slot still processes one I/Q pair in parallel.
    let i_samples = dense_interval(-1.0, -0.5);
    let q_samples = dense_interval(1.0, 0.5);
    let encrypted_i = context.encrypt(&i_samples);
    let encrypted_q = context.encrypt(&q_samples);
    let input_budget = encrypted_i.log_budget();

    // hypot combines both encrypted channels and produces their magnitude.
    let mut magnitude = context.output();
    context.module.ckks_hypot(
        &mut magnitude,
        &encrypted_i,
        &encrypted_q,
        &hypot,
        &context.tsk,
        &mut context.scratch.borrow(),
    )?;
    assert_eq!(input_budget - magnitude.log_budget(), hypot_consumed);

    // The logarithm consumes the output of hypot directly because both plans use compatible CKKS metadata.
    let mut log_magnitude = context.output();
    context.module.ckks_log(
        &mut log_magnitude,
        &magnitude,
        &log,
        &context.tsk,
        &mut context.scratch.borrow(),
    )?;
    assert_eq!(input_budget - log_magnitude.log_budget(), circuit_consumed);

    // Multiplication by two is exact power-of-two scaling and does not consume more modulus.
    let mut log_power = context.output();
    context.module.ckks_scalbn(
        &mut log_power,
        &log_magnitude,
        1,
        &mut context.scratch.borrow(),
    )?;
    assert_eq!(log_magnitude.log_budget(), log_power.log_budget());

    // Verify the complete pipeline against ln(I^2 + Q^2), not its intermediate stages.
    let output_log_delta = log_power.log_delta();
    let got = context.decrypt(&log_power);
    let want: Vec<f64> = i_samples
        .iter()
        .zip(&q_samples)
        .map(|(i, q)| (i * i + q * q).ln())
        .collect();
    verify("signal/log_power", &got, &want, output_log_delta);
    println!(
        "I={:.3}, Q={:.3}, log power={:.6}",
        i_samples[0], q_samples[0], got[0]
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
