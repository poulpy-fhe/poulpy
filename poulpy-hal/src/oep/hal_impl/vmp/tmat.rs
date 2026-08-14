//! Open extension points whose prepared operand is the transformed hot-prep [`VmpTMat`](crate::layouts::VmpTMat).

#![allow(clippy::too_many_arguments)]

use crate::layouts::{Backend, Module, ScratchArena};

hal_vmp_tier_impl!(
    /// Backend extension points for the `tmat` tier of the VMP family.
    HalVmpTMatImpl,
    TMat,
    tmat
);
