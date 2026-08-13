use std::collections::HashMap;

use poulpy_core::layouts::{GLWEAutomorphismKeyCompressed, GLWETensorKeyCompressed};
use poulpy_hal::layouts::{Data, ZnxWord};

use crate::blind_rotation::{BlindRotationAlgo, BlindRotationKeyCompressed};

#[allow(dead_code)]
pub struct CircuitBootstrappingKey<D: Data, BRA: BlindRotationAlgo, W: ZnxWord> {
    pub(crate) brk: BlindRotationKeyCompressed<D, BRA, W>,
    pub(crate) tsk: GLWETensorKeyCompressed<D, W>,
    pub(crate) atk: HashMap<i64, GLWEAutomorphismKeyCompressed<D, W>>,
}
