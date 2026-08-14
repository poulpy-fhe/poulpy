use std::{
    hash::{DefaultHasher, Hasher},
    marker::PhantomData,
};

use crate::layouts::{Backend, Data, DataView, DataViewMut, DftWord, DigestU64, HostDataMut, HostDataRef, MatZnxInfos, ZnxInfos};

vmp_mat_family!(
    /// Transformed (hot-prep) polynomial matrix for vector-matrix products.
    ///
    /// A `VmpTMat` holds the prepared form of a `rows * cols_in` by `cols_out`
    /// matrix of `size`-limb polynomials, in the representation named by the
    /// [`DftWord`] type `W`.
    ///
    /// The internal arrangement is entirely backend-defined: a backend is free to
    /// interleave, transpose or repack, and nothing outside that backend's own
    /// kernels may assume an entry layout. [`Backend::bytes_of_vmp_tmat`] is
    /// authoritative for the buffer size, and cross-backend reinterpretation is
    /// gated on [`VmpTMatLayoutCompatible`](crate::layouts::VmpTMatLayoutCompatible).
    ///
    /// `VmpTMat` is the cheap-to-build prepared form, meant for short reuse;
    /// [`VmpPMat`](crate::layouts::VmpPMat) is the packed form, more expensive to
    /// build but optimized for amortized repeated apply. The two are distinct types
    /// even where a backend gives them the same physical storage shape.
    ///
    /// Create via [`VmpPrepareTMat`](crate::api::VmpPrepareTMat) from a
    /// coefficient-domain [`MatZnx`](crate::layouts::MatZnx), then consume through
    /// the `vmp_apply_tmat_*` family.
    VmpTMat,
    vmp_tmat
);
