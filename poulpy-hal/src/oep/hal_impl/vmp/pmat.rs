//! Open extension points whose prepared operand is the packed cold-prep [`VmpPMat`](crate::layouts::VmpPMat).

#![allow(clippy::too_many_arguments)]

use crate::layouts::{Backend, Module, ScratchArena};

hal_vmp_tier_impl!(
    /// Backend extension points for the `pmat` tier of the VMP family.
    HalVmpPMatImpl,
    PMat,
    pmat
);
