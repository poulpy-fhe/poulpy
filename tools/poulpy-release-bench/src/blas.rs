//! Release measurements of `poulpy-blas` operations on one `d x d` block per operand on the conjugate-invariant ring at
//! `N = 2^15`, on the point builders of poulpy-bench, with the evaluated backend as its own client on the CPU and
//! `NTT4x30Portable` as the client of a GPU backend.
//!
//! A builder makes its whole point outside the timed region: the module, the secret, the context or the public matrix,
//! the rotation keys its operation looks up (and the tensor key of a ciphertext by ciphertext product), and the
//! encrypted inputs. It checks the radix against the lazy sums of the operation, calls it once and asserts every entry
//! of the result within the error bound of poulpy-bench, then the same call is timed. A ciphertext holds
//! `2^15 / d^2` matrices, each of distinct entries, as many as the standard ring at `N = 2^16`. The operations consume
//! at most 120 bits, a modulus `N = 2^15` holds with room to spare.

use std::time::Instant;

use anyhow::{Result, bail};
use poulpy_bench::blas::{
    intra::point_blas_reduction,
    matmul::{BlasClientModule, BlasEvalModule, BlasPoint, point_blas_matmul, point_blas_matmul_pt_right, point_blas_transpose},
    params::BlasBenchParams,
};
use poulpy_blas::{
    layouts::BlasPtAlgorithm,
    oep::{BlasMatIntraImpl, BlasMatMulImpl},
};
use poulpy_core::layouts::{GGLWEInfos, LWEInfos};
use poulpy_hal::{
    api::{ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Backend, ConjugateInvariant, HostStaged, MaxBase2k, Module, ScratchOwned},
};
use serde_json::{Value, json};

use crate::runner::{CaseArgs, direct};

const LOG_N: usize = 15;
const LOG_DS: [usize; 3] = [5, 6, 7];
const OPS: [Op; 4] = [Op::Matmul, Op::MatmulPtRight, Op::Transpose, Op::SumRows];

#[derive(Clone, Copy)]
enum Op {
    /// Ciphertext by ciphertext, distinct matrices on both sides.
    Matmul,
    /// Ciphertext by a public matrix on the right, under `Jkls`.
    MatmulPtRight,
    Transpose,
    /// `blas_sum_rows` of distinct matrices.
    SumRows,
}

impl Op {
    fn name(self) -> &'static str {
        match self {
            Op::Matmul => "matmul",
            Op::MatmulPtRight => "matmul_pt_right",
            Op::Transpose => "transpose",
            Op::SumRows => "sum_rows",
        }
    }

    fn function(self) -> &'static str {
        match self {
            Op::Matmul => "blas_matmul",
            Op::MatmulPtRight => "blas_matmul_pt_right (Jkls)",
            Op::Transpose => "blas_transpose",
            Op::SumRows => "blas_sum_rows",
        }
    }
}

/// The cases of the release set, `<op>_d<d>`, every operation at every block dimension.
pub fn cases() -> Vec<String> {
    OPS.iter()
        .flat_map(|op| LOG_DS.iter().map(move |log_d| format!("{}_d{}", op.name(), 1 << log_d)))
        .collect()
}

fn case(name: &str) -> Result<(Op, BlasBenchParams)> {
    for op in OPS {
        for log_d in LOG_DS {
            if name == format!("{}_d{}", op.name(), 1 << log_d) {
                // The parameters of the poulpy-bench sweep, at the release ring and degree.
                return Ok((
                    op,
                    BlasBenchParams {
                        n: 1 << LOG_N,
                        log_d,
                        base2k: 52,
                        k: 208,
                        log_delta: 40,
                        p: 40,
                        dsize: 1,
                    },
                ));
            }
        }
    }
    bail!("no blas case named {name}")
}

/// Builds the case `args.case` on `BE` with `BC` as the client, times its calls, and reports its parameters, modulus
/// and precision.
pub fn run<BE, BC>(args: &CaseArgs) -> Result<Value>
where
    BE: Backend<Ring = ConjugateInvariant> + HostStaged + 'static + BlasMatIntraImpl + BlasMatMulImpl + MaxBase2k,
    BC: Backend<Ring = ConjugateInvariant> + HostStaged + 'static,
    Module<BE>: BlasEvalModule<BE>,
    Module<BC>: BlasClientModule<BC>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    ScratchOwned<BC>: ScratchOwnedAlloc<BC> + ScratchOwnedBorrow<BC>,
{
    let (op, bp) = case(&args.case)?;
    let matrices = bp.batch_on::<ConjugateInvariant>();
    let start = Instant::now();
    let BlasPoint { mut call, check } = match op {
        Op::Matmul => point_blas_matmul::<BE, BC>(&bp, true),
        Op::MatmulPtRight => point_blas_matmul_pt_right::<BE, BC>(&bp, BlasPtAlgorithm::Jkls),
        Op::Transpose => point_blas_transpose::<BE, BC>(&bp),
        Op::SumRows => point_blas_reduction::<BE, BC>(&bp, "blas_sum_rows", true, matrices),
    };
    let setup_ns = start.elapsed().as_nanos() as u64;
    let timings = direct(&mut call, &|| {}, args.warmup, args.iters)?.expect("direct timing is always available");

    let key = &check.key_layout;
    Ok(json!({
        "blas": {
            "case": args.case,
            "op": op.name(),
            "function": op.function(),
            "ring": "ConjugateInvariant",
            "log_n": LOG_N,
            "n": bp.n,
            "d": bp.d(),
            "matrices": matrices,
            "base2k": bp.base2k,
            "log_delta": bp.log_delta,
            "p": bp.p,
            "rotation_keys": check.keys,
            "tensor_key": matches!(op, Op::Matmul),
            "max_base2k": check.max_base2k,
        },
        "modulus": {
            "log_q": bp.k,
            "consumed_bits": check.consumed_bits,
            "output_log_budget": check.output_log_budget,
            "key_log_qp": key.k().as_usize(),
            "key_dsize": bp.dsize,
            "key_dnum": key.dnum().as_usize(),
            "key_k_aux": key.k_aux().as_usize(),
        },
        "setup_ns": setup_ns,
        "warmup_ns": timings.warmup_ns,
        "samples_ns": timings.samples_ns,
        "precision": {
            "max_abs_err": check.max_err,
            "abs_bits": -check.max_err.log2(),
            "bound_bits": -check.bound.log2(),
            "meets_target": check.max_err <= check.bound,
        },
    }))
}
