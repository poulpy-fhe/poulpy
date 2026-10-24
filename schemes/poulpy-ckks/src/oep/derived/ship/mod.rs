//! SHIP half-bootstrapping drivers over the selected CKKS and core operations and
//! the masking and mux hooks, with their preflight and scratch sizing.

use std::collections::HashMap;

use poulpy_hal::{layouts::Module, oep::HalVecZnxDftImpl};

use crate::layouts::HMuxRotKeyPrepared;

mod bootstrap;
mod preflight;

pub(crate) use bootstrap::{ckks_ship_bootstrap_complex_into_derived, ckks_ship_bootstrap_into_derived};
pub(crate) use preflight::ckks_ship_bootstrap_tmp_bytes_derived;

/// Automorphism plans of the mux output twists, keyed by Galois element.
pub(crate) type ShipMuxPlans<BE> = HashMap<i64, <BE as HalVecZnxDftImpl>::AutomorphismPlan>;

/// Builds the automorphism plans of every distinct mux Galois element once;
/// the same rotation amounts recur across all support slots.
pub(crate) fn ship_mux_plans<'a, BE>(
    module: &Module<BE>,
    groups: impl Iterator<Item = &'a [HMuxRotKeyPrepared<BE::OwnedBuf, BE>]>,
) -> ShipMuxPlans<BE>
where
    BE: HalVecZnxDftImpl + 'a,
{
    let mut plans = ShipMuxPlans::<BE>::new();
    for group in groups {
        for key in group {
            if key.gal_el != 1 {
                plans
                    .entry(key.gal_el)
                    .or_insert_with(|| BE::vec_znx_dft_automorphism_plan(module, module.n(), key.gal_el));
            }
        }
    }
    plans
}
