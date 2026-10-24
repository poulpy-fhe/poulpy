//! PaCo bootstrapping drivers: the sequential and parallel branch schedules over the
//! selected CKKS and core operations, their preflight and scratch sizing, and the slot folds.

mod bootstrap;
mod parallel;
mod preflight;
mod slot;

pub(crate) use parallel::{
    ckks_paco_bootstrap_direct_into_derived, ckks_paco_bootstrap_into_derived, ckks_paco_bootstrap_parallel_direct_into_derived,
    ckks_paco_bootstrap_parallel_into_derived,
};
pub(crate) use preflight::{ckks_paco_bootstrap_direct_tmp_bytes_derived, ckks_paco_bootstrap_tmp_bytes_derived};
pub(crate) use slot::{ckks_slot_product_assign_derived, ckks_slot_trace_assign_derived};
