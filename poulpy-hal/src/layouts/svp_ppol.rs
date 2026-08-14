use std::{
    fmt,
    hash::{DefaultHasher, Hasher},
    marker::PhantomData,
};

use crate::layouts::{
    Backend, Data, DataView, DataViewMut, DftWord, DigestU64, HostDataRef, ScalarZnxShape, VecZnxInfos, ZnxInfos, ZnxView,
};

svp_pol_family!(
    /// Packed (cold-prep) scalar polynomial for scalar-vector products.
    ///
    /// An `SvpPPol` holds the prepared form of `cols` scalar polynomials, in the
    /// representation named by the [`DftWord`] type `W`.
    ///
    /// The internal arrangement is entirely backend-defined: the FFT64 backends
    /// store the reim DFT, the NTT backends a q120c encoding whose element view
    /// differs from the DFT-domain one they multiply against. Nothing outside a
    /// backend's own kernels may assume a coefficient layout.
    /// [`Backend::bytes_of_svp_ppol`] is authoritative for the buffer size, and
    /// cross-backend reinterpretation is gated on
    /// [`SvpPPolLayoutCompatible`](crate::layouts::SvpPPolLayoutCompatible).
    ///
    /// `SvpPPol` is the expensive-to-build prepared form, optimized for amortized
    /// repeated apply; [`SvpTPol`](crate::layouts::SvpTPol) is the cheaper hot-prep
    /// form for short reuse.
    ///
    /// Create via [`SvpPreparePPol`](crate::api::SvpPreparePPol) from a
    /// coefficient-domain [`ScalarZnx`](crate::layouts::ScalarZnx), then consume
    /// through the `svp_apply_ppol_*` family.
    SvpPPol,
    svp_ppol
);
