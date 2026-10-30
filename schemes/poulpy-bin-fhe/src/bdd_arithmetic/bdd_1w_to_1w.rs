pub use crate::api::BDDCircuitExecute1WTo1W;
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
                    M: BDDCircuitExecute1WTo1W<BE>,
                    H: GetAutomorphismKey<BE>,
                    BE: Backend<ZnxWord = i64>,
                    Self: GLWEToBackendMut<BE>;

                /// Multithreaded version – same vis, method_name + "_multi_thread"
                fn [<$method_name _multi_thread>]<M, H>(
                    &mut self,
                    module: &M,
                    threads: usize,
                    a: &FheUintPrepared<BE::OwnedBuf, T, BE>,
                    key: &H,
                    scratch: &mut ScratchArena<'_, BE>,
                ) where
                    M: BDDCircuitExecute1WTo1W<BE>,
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
                    M: BDDCircuitExecute1WTo1W<BE>,
                    H: GetAutomorphismKey<BE>,
                    BE: Backend<ZnxWord = i64>,
                {
                    module.bdd_circuit_execute_1w_to_1w(self, a, &$output_circuits, key, scratch)
                }

                fn [<$method_name _multi_thread>]<M, H>(
                    &mut self,
                    module: &M,
                    threads: usize,
                    a: &FheUintPrepared<BE::OwnedBuf, $ty, BE>,
                    key: &H,
                    scratch: &mut ScratchArena<'_, BE>,
                ) where
                    M: BDDCircuitExecute1WTo1W<BE>,
                    H: GetAutomorphismKey<BE>,
                    BE: Backend<ZnxWord = i64>,
                {
                    module.bdd_circuit_execute_1w_to_1w_multi_thread(threads, self, a, &$output_circuits, key, scratch)
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
    pub FheUintIdentity, identity);

impl_bdd_1w_to_1w_trait!(
    FheUintIdentity,
    identity,
    u32,
    circuits::u32::identity_codegen::AnyBitCircuit,
    circuits::u32::identity_codegen::OUTPUT_CIRCUITS
);
