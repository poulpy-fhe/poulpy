pub use crate::api::ExecuteBDDCircuit1WTo1W;
use poulpy_core::layouts::{GLWEToBackendMut, GetAutomorphismKey};
use poulpy_hal::layouts::{Backend, ScratchArena};

use crate::bdd_arithmetic::{FheUint, FheUintPrepared, UnsignedInteger, circuits};

#[macro_export]
macro_rules! define_bdd_1w_to_1w_trait {
    ($(#[$meta:meta])* $vis:vis $trait_name:ident, $method_name:ident) => {
        paste::paste! {
            $(#[$meta])*
            $vis trait $trait_name<T: UnsignedInteger, BE: Backend> {

                /// Single-threaded version
                fn $method_name<M, H>(
                    &mut self,
                    module: &M,
                    a: &FheUintPrepared<BE::OwnedBuf, T, BE>,
                    key: &H,
                    scratch: &mut ScratchArena<'_, BE>,
                ) where
                    M: ExecuteBDDCircuit1WTo1W<BE>,
                    H: GetAutomorphismKey<BE>,
                    BE: Backend<ZnxWord = i64>,
                    Self: GLWEToBackendMut<BE>;

                /// Multithreaded version – same vis, method_name + "_multi_thread"
                fn [<$method_name _multi_thread>]<M, H>(
                    &mut self,
                    threads: usize,
                    module: &M,
                    a: &FheUintPrepared<BE::OwnedBuf, T, BE>,
                    key: &H,
                    scratch: &mut ScratchArena<'_, BE>,
                ) where
                    M: ExecuteBDDCircuit1WTo1W<BE>,
                    H: GetAutomorphismKey<BE>,
                    BE: Backend<ZnxWord = i64>,
                    Self: GLWEToBackendMut<BE>;
            }
        }
    };
}

#[macro_export]
macro_rules! impl_bdd_1w_to_1w_trait {
    ($trait_name:ident, $method_name:ident, $ty:ty, $circuit_ty:ty, $output_circuits:path) => {
        paste::paste! {
            impl<BE: Backend<ZnxWord = i64>> $trait_name<$ty, BE> for FheUint<BE::OwnedBuf, $ty, BE::ZnxWord> {

                fn $method_name<M, H>(
                    &mut self,
                    module: &M,
                    a: &FheUintPrepared<BE::OwnedBuf, $ty, BE>,
                    key: &H,
                    scratch: &mut ScratchArena<'_, BE>,
                ) where
                    M: ExecuteBDDCircuit1WTo1W<BE>,
                    H: GetAutomorphismKey<BE>,
                    BE: Backend<ZnxWord = i64>,
                {
                    module.execute_bdd_circuit_1w_to_1w(self, &$output_circuits, a, key, scratch)
                }

                fn [<$method_name _multi_thread>]<M, H>(
                    &mut self,
                    threads: usize,
                    module: &M,
                    a: &FheUintPrepared<BE::OwnedBuf, $ty, BE>,
                    key: &H,
                    scratch: &mut ScratchArena<'_, BE>,
                ) where
                    M: ExecuteBDDCircuit1WTo1W<BE>,
                    H: GetAutomorphismKey<BE>,
                    BE: Backend<ZnxWord = i64>,
                {
                    module.execute_bdd_circuit_1w_to_1w_multi_thread(threads, self, &$output_circuits, a, key, scratch)
                }
            }
        }
    };
}
define_bdd_1w_to_1w_trait!(
    /// Homomorphic identity function (`out = a`).
    ///
    /// Re-bootstraps all bits of `a` through the BDD circuit and repacks the
    /// result into a fresh [`FheUint`].  Useful for noise refreshing without
    /// computing any arithmetic.
    pub Identity, identity);

impl_bdd_1w_to_1w_trait!(
    Identity,
    identity,
    u32,
    circuits::u32::identity_codgen::AnyBitCircuit,
    circuits::u32::identity_codgen::OUTPUT_CIRCUITS
);
