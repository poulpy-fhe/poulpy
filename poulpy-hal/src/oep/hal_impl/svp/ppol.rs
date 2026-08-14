//! Open extension points whose prepared operand is the packed cold-prep [`SvpPPol`](crate::layouts::SvpPPol).

#![allow(clippy::too_many_arguments)]

use crate::layouts::{Backend, Module, ScalarZnxBackendRef};

hal_svp_tier_impl!(
    /// Backend extension points for the `ppol` tier of the SVP family.
    HalSvpPPolImpl,
    PPol,
    ppol
);
