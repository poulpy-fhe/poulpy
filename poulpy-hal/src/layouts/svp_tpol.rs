use std::{
    fmt,
    hash::{DefaultHasher, Hasher},
    marker::PhantomData,
};

use crate::layouts::{
    Backend, Data, DataView, DataViewMut, DftWord, DigestU64, HostDataRef, ScalarZnxShape, VecZnxInfos, ZnxInfos, ZnxView,
};

svp_pol_family!(
    /// Transformed (hot-prep) scalar polynomial for scalar-vector products.
    ///
    /// An `SvpTPol` holds the prepared form of `cols` scalar polynomials, in the
    /// representation named by the [`DftWord`] type `W`.
    ///
    /// The internal arrangement is entirely backend-defined: nothing outside that
    /// backend's own kernels may assume a coefficient layout.
    /// [`Backend::bytes_of_svp_tpol`] is authoritative for the buffer size, and
    /// cross-backend reinterpretation is gated on
    /// [`SvpTPolLayoutCompatible`](crate::layouts::SvpTPolLayoutCompatible).
    ///
    /// `SvpTPol` is the cheap-to-build prepared form, meant for short reuse or
    /// one-shot use; [`SvpPPol`](crate::layouts::SvpPPol) is the packed form,
    /// more expensive to build but optimized for amortized repeated apply. The
    /// two are distinct types even where a backend gives them the same physical
    /// storage shape.
    ///
    /// Create via [`SvpPrepareTPol`](crate::api::SvpPrepareTPol) from a
    /// coefficient-domain [`ScalarZnx`](crate::layouts::ScalarZnx), then consume
    /// through the `svp_apply_tpol_*` family.
    SvpTPol,
    svp_tpol
);
