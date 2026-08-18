//! Operand construction for the core benchmarks.
//!
//! Benchmarks measure time, and this arithmetic is data-independent, so the
//! operands need not be genuine ciphertexts: an encryption of zero is uniform
//! limbs too. Filling directly removes secrets and encryption from the runners,
//! which is what keeps them open to a device backend.
//!
//! Operands are built on a host staging module and transferred in, rather than
//! sampled through the tested backend. A device backend has no reason to own a
//! uniform-sampling kernel, and requiring one would exclude exactly the
//! backends these runners exist to serve. This is setup, outside
//! `bencher.iter`, so the transfer costs nothing measurable.
//!
//! Uniform noise rather than zeros is deliberate: a float FFT over zeroed
//! buffers can hit denormals and mistime the kernel.

use poulpy_core::layouts::{
    GGLWE, GGLWEInfos, GGSW, GGSWInfos, GLWE, GLWEAutomorphismKey, GLWEInfos, GLWEPlaintext, GLWETensor, GLWETensorKey,
    ModuleCoreAlloc,
};
use poulpy_hal::{
    layouts::{FillUniform, HostBytesBackend, Module},
    source::Source,
};

/// Host staging module for a ring degree. `HostBytesBackend::new` is a dangling
/// handle, so this is free.
pub fn staging(n: usize) -> Module<HostBytesBackend> {
    Module::<HostBytesBackend>::new(n as u64)
}

macro_rules! host_builder {
    ($( $name:ident => $ty:ty, $alloc:ident, $infos:path ),+ $(,)?) => {
        $(
            /// Allocates on the staging module and fills with uniform noise.
            pub fn $name<A: $infos>(host: &Module<HostBytesBackend>, infos: &A, source: &mut Source) -> $ty {
                let mut out = host.$alloc(infos);
                out.fill_uniform(infos.base2k().into(), source);
                out
            }
        )+
    };
}

host_builder! {
    host_glwe => GLWE<Vec<u8>, i64>, glwe_alloc_from_infos, GLWEInfos,
    host_glwe_plaintext => GLWEPlaintext<Vec<u8>, i64>, glwe_plaintext_alloc_from_infos, GLWEInfos,
    host_glwe_tensor => GLWETensor<Vec<u8>, i64>, glwe_tensor_alloc_from_infos, GLWEInfos,
    host_gglwe => GGLWE<Vec<u8>, i64>, gglwe_alloc_from_infos, GGLWEInfos,
    host_ggsw => GGSW<Vec<u8>, i64>, ggsw_alloc_from_infos, GGSWInfos,
    host_glwe_automorphism_key => GLWEAutomorphismKey<Vec<u8>, i64>, glwe_automorphism_key_alloc_from_infos, GGLWEInfos,
    host_glwe_tensor_key => GLWETensorKey<Vec<u8>, i64>, glwe_tensor_key_alloc_from_infos, GGLWEInfos,
}
