//! Open extension points whose prepared operand is the transformed hot-prep [`SvpTPol`](crate::layouts::SvpTPol).

#![allow(clippy::too_many_arguments)]

use crate::layouts::{Backend, Module, ScalarZnxBackendRef};

hal_svp_tier_impl!(
    /// Backend extension points for the `tpol` tier of the SVP family.
    HalSvpTPolImpl,
    TPol,
    tpol
);
