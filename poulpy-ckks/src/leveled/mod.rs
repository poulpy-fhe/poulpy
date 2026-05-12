//! Re-export shim that preserves the historical `crate::leveled::api::*` import
//! paths from before the crate was reorganised to mirror `poulpy-core`.
//!
//! The canonical locations are the crate-root modules:
//!
//! - [`crate::api`]
//! - [`crate::oep`]
//! - [`crate::delegates`]
//! - [`crate::default`]
//! - [`crate::layouts`]
//!
//! New code should import from those paths directly.  `crate::leveled::api`
//! will remain as an alias as long as internal code still uses it.

pub mod api {
    pub use crate::api::*;
}

#[allow(unused_imports)]
pub(crate) mod delegates {
    pub(crate) use crate::delegates::*;
}

#[allow(unused_imports)]
pub(crate) mod default {
    pub(crate) use crate::default::*;

    pub(crate) mod add {
        pub(crate) use crate::default::add::*;
    }
    pub(crate) mod conjugate {
        pub(crate) use crate::default::conjugate::*;
    }
    pub(crate) mod encryption {
        pub(crate) use crate::default::encryption::*;
    }
    pub(crate) mod imag {
        pub(crate) use crate::default::imag::*;
    }
    pub(crate) mod mul {
        pub(crate) use crate::default::mul::*;
    }
    pub(crate) mod neg {
        pub(crate) use crate::default::neg::*;
    }
    pub(crate) mod pow2 {
        pub(crate) use crate::default::pow2::*;
    }
    pub(crate) mod plaintext {
        pub(crate) use crate::default::plaintext::*;
    }
    pub(crate) mod rescale {
        pub(crate) use crate::default::rescale::*;
    }
    pub(crate) mod rotate {
        pub(crate) use crate::default::rotate::*;
    }
    pub(crate) mod sub {
        pub(crate) use crate::default::sub::*;
    }
}
