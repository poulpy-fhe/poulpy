use poulpy_core::{
    GLWEBytesOf, GLWECopy, GLWEPacking, ScratchArenaTakeCore,
    layouts::{GLWEToBackendMut, GetAutomorphismKey},
};
use poulpy_hal::layouts::CoeffNormalized;
use poulpy_hal::{
    api::ModuleLogN,
    layouts::{Backend, Module, ScratchArena},
};

use crate::bdd_arithmetic::{ExecuteBDDCircuit, FheUint, FheUintPrepared, GetBitCircuitInfo, UnsignedInteger, circuits};
impl<BE: Backend<ZnxWord = i64>> ExecuteBDDCircuit1WTo1W<BE> for Module<BE> where
    Self: Sized + ExecuteBDDCircuit<BE> + GLWEPacking<BE> + GLWECopy<BE>
{
}

/// Backend-level executor for single-input BDD circuits (`Z → Z`).
///
/// Evaluates a BDD circuit that reads one encrypted integer and produces
/// one encrypted integer.  After evaluating the per-bit BDD levels, the
/// output bits are repacked into a single [`FheUint`] polynomial via
/// [`GLWEPacking`].
pub trait ExecuteBDDCircuit1WTo1W<BE: Backend>
where
    Self: GLWEBytesOf<BE>,
    Self: Sized + ModuleLogN + ExecuteBDDCircuit<BE> + GLWEPacking<BE> + GLWECopy<BE>,
{
    fn execute_bdd_circuit_1w_to_1w<C, H, T>(
        &self,
        out: &mut FheUint<BE::OwnedBuf, T, BE::ZnxWord>,
        circuit: &C,
        a: &FheUintPrepared<BE::OwnedBuf, T, BE>,
        key: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        T: UnsignedInteger,
        C: GetBitCircuitInfo,
        H: GetAutomorphismKey<BE>,
        BE: Backend<ZnxWord = i64>,
    {
        self.execute_bdd_circuit_1w_to_1w_multi_thread(1, out, circuit, a, key, scratch);
    }

    #[allow(clippy::too_many_arguments)]
    /// Operations Z x Z -> Z
    fn execute_bdd_circuit_1w_to_1w_multi_thread<C, H, T>(
        &self,
        threads: usize,
        out: &mut FheUint<BE::OwnedBuf, T, BE::ZnxWord>,
        circuit: &C,
        a: &FheUintPrepared<BE::OwnedBuf, T, BE>,
        key: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        T: UnsignedInteger,
        C: GetBitCircuitInfo,
        H: GetAutomorphismKey<BE>,
        BE: Backend<ZnxWord = i64>,
    {
        let (mut out_bits, mut scratch_1) = scratch.borrow().take_glwe_slice_scratch(T::BITS as usize, out);

        // Evaluates out[i] = circuit[i](a, b)
        self.execute_bdd_circuit_multi_thread(threads, &mut out_bits, a, circuit, &mut scratch_1);

        // Repacks the bits
        out.pack(self, out_bits, key, &mut scratch_1);
    }
}

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
                    Self: GLWEToBackendMut<BE, State = CoeffNormalized>;

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
                    Self: GLWEToBackendMut<BE, State = CoeffNormalized>;
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
