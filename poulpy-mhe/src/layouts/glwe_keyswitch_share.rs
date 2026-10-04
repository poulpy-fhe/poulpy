use std::{
    fmt,
    io::{self, Read, Write},
};

use poulpy_core::layouts::{
    Base2K, Degree, GLWE, GLWEBackendMut, GLWEBackendRef, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, LWEInfos, Rank,
    TorusPrecision,
};
use poulpy_hal::layouts::{Backend, Data, HostDataMut, HostDataRef, ReaderFrom, WriterTo, ZnxWord};

use crate::layouts::glwe_share::glwe_share;

glwe_share!(
    /// One party's share of a collective key switch to a secret key: a rank-0
    /// GLWE, the body the finalization adds to the ciphertext.
    GLWEPrivateKeyswitchShare,
    GLWEPrivateKeyswitchShareOwned
);

glwe_share!(
    /// One party's share of a collective key switch to a public key: a GLWE
    /// under the output public key, at its rank.
    GLWEPublicKeyswitchShare,
    GLWEPublicKeyswitchShareOwned
);
