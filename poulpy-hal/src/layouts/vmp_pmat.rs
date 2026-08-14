use std::{
    hash::{DefaultHasher, Hasher},
    marker::PhantomData,
};

use crate::layouts::{Backend, Data, DataView, DataViewMut, DftWord, DigestU64, HostDataMut, HostDataRef, MatZnxInfos, ZnxInfos};

vmp_mat_family!(
    /// Prepared (DFT-domain) polynomial matrix for vector-matrix products.
    ///
    /// A `VmpPMat` holds the prepared form of a `rows * cols_in` by `cols_out`
    /// matrix of `size`-limb polynomials, in the representation named by the
    /// [`DftWord`] type `W`.
    ///
    /// The internal arrangement is entirely backend-defined and is **not** a
    /// sequence of [`VecZnxDft`](crate::layouts::VecZnxDft) entries: the FFT64
    /// backends store reim4 block-interleaved, the reference NTT4x30 backend q120c
    /// block-interleaved, and the accelerated NTT backends prime-major planar. A
    /// backend may also pack more compactly than `size_of::<W>()` per coefficient,
    /// as the AVX-512 IFMA backend does. Nothing outside a backend's own kernels may assume
    /// an entry layout: [`Backend::bytes_of_vmp_pmat`] is authoritative for the
    /// buffer size, and cross-backend reinterpretation is gated on
    /// [`VmpPMatLayoutCompatible`](crate::layouts::VmpPMatLayoutCompatible), which
    /// is deliberately not declared between the reference and accelerated NTT4x30
    /// backends.
    ///
    /// Used as the matrix operand in
    /// [`VmpApplyPMatDftToDft`](crate::api::VmpApplyPMatDftToDft). Create via
    /// [`VmpPreparePMat`](crate::api::VmpPreparePMat) from a coefficient-domain
    /// [`MatZnx`](crate::layouts::MatZnx).
    VmpPMat,
    vmp_pmat
);
