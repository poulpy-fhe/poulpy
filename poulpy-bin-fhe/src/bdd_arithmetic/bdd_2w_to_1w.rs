use poulpy_hal::layouts::Normalized;
use std::marker::PhantomData;

use poulpy_core::{
    GLWECopy, GLWEPacking, ScratchArenaTakeCore,
    layouts::{GGSWInfos, GLWEInfos, GLWEToBackendMut, GetAutomorphismKey, prepared::GGSWPrepared},
};
use poulpy_hal::{
    api::ModuleLogN,
    layouts::{Backend, Module, ScratchArena},
};

use crate::bdd_arithmetic::{
    BitSize, ExecuteBDDCircuit, FheUint, FheUintPrepared, GetBitCircuitInfo, GetGGSWBit, UnsignedInteger, circuits,
};
use poulpy_core::GLWEBytesOf;

impl<BE: Backend<ZnxWord = i64>> ExecuteBDDCircuit2WTo1W<BE> for Module<BE> where
    Self: Sized + ExecuteBDDCircuit<BE> + GLWEPacking<BE> + GLWECopy<BE>
{
}

/// Backend-level executor for two-input BDD circuits (`Z × Z → Z`).
///
/// Evaluates a BDD circuit that reads two encrypted integers and produces
/// one encrypted integer.  The two input [`FheUintPrepared`] values are
/// concatenated into a virtual bit-array, with `a` occupying bits
/// `[0, T::BITS)` and `b` occupying `[T::BITS, 2*T::BITS)`.  After BDD
/// evaluation the output bits are repacked into a single [`FheUint`].
pub trait ExecuteBDDCircuit2WTo1W<BE: Backend>
where
    Self: Sized + GLWEBytesOf<BE> + ModuleLogN + ExecuteBDDCircuit<BE> + GLWEPacking<BE> + GLWECopy<BE>,
{
    fn execute_bdd_circuit_2w_to_1w<C, H, T>(
        &self,
        out: &mut FheUint<BE::OwnedBuf, T, BE::ZnxWord>,
        circuit: &C,
        a: &FheUintPrepared<BE::OwnedBuf, T, BE>,
        b: &FheUintPrepared<BE::OwnedBuf, T, BE>,
        key: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        T: UnsignedInteger,
        C: GetBitCircuitInfo,
        H: GetAutomorphismKey<BE>,
        BE: Backend<ZnxWord = i64>,
    {
        self.execute_bdd_circuit_2w_to_1w_multi_thread(1, out, circuit, a, b, key, scratch);
    }

    /// Minimum scratch size in bytes for [`execute_bdd_circuit_2w_to_1w`][Self::execute_bdd_circuit_2w_to_1w]
    /// (one BDD worker).
    fn execute_bdd_circuit_2w_to_1w_tmp_bytes<C, T, R, G, H>(&self, circuit: &C, res_infos: &R, ggsw_infos: &G, key: &H) -> usize
    where
        C: GetBitCircuitInfo,
        T: UnsignedInteger,
        R: GLWEInfos,
        G: GGSWInfos,
        H: GetAutomorphismKey<BE>,
    {
        let glwe_slot_bytes = T::BITS as usize * self.glwe_bytes_of_from_infos(res_infos);
        let bdd_bytes = self.execute_bdd_circuit_tmp_bytes(res_infos, circuit.max_state_size(), ggsw_infos);
        let atk = key.get_automorphism_key(-1, res_infos.k()).unwrap_or_else(|e| panic!("{e}"));
        let pack_bytes = self.glwe_pack_tmp_bytes(res_infos, &atk);
        glwe_slot_bytes + bdd_bytes.max(pack_bytes)
    }

    /// Minimum scratch size in bytes for [`execute_bdd_circuit_2w_to_1w_multi_thread`][Self::execute_bdd_circuit_2w_to_1w_multi_thread].
    fn execute_bdd_circuit_2w_to_1w_multi_thread_tmp_bytes<C, T, R, G, H>(
        &self,
        threads: usize,
        circuit: &C,
        res_infos: &R,
        ggsw_infos: &G,
        key: &H,
    ) -> usize
    where
        C: GetBitCircuitInfo,
        T: UnsignedInteger,
        R: GLWEInfos,
        G: GGSWInfos,
        H: GetAutomorphismKey<BE>,
    {
        let glwe_slot_bytes = T::BITS as usize * self.glwe_bytes_of_from_infos(res_infos);
        let bdd_per_thread = poulpy_hal::execution::worker_scratch_bytes::<BE>(self.execute_bdd_circuit_tmp_bytes(
            res_infos,
            circuit.max_state_size(),
            ggsw_infos,
        ));
        let atk = key.get_automorphism_key(-1, res_infos.k()).unwrap_or_else(|e| panic!("{e}"));
        let pack_bytes = self.glwe_pack_tmp_bytes(res_infos, &atk);
        glwe_slot_bytes + (threads * bdd_per_thread).max(pack_bytes)
    }

    #[allow(clippy::too_many_arguments)]
    /// Operations Z x Z -> Z
    fn execute_bdd_circuit_2w_to_1w_multi_thread<C, H, T>(
        &self,
        threads: usize,
        out: &mut FheUint<BE::OwnedBuf, T, BE::ZnxWord>,
        circuit: &C,
        a: &FheUintPrepared<BE::OwnedBuf, T, BE>,
        b: &FheUintPrepared<BE::OwnedBuf, T, BE>,
        key: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        T: UnsignedInteger,
        C: GetBitCircuitInfo,
        H: GetAutomorphismKey<BE>,
        BE: Backend<ZnxWord = i64>,
    {
        // Collects inputs into a single array
        let inputs: Vec<&dyn GetGGSWBit<BE>> = [a as &dyn GetGGSWBit<BE>, b as &dyn GetGGSWBit<BE>].to_vec();
        let helper: FheUintHelper<'_, T, BE> = FheUintHelper {
            data: inputs,
            _phantom: PhantomData,
        };

        let (mut out_bits, mut scratch_1) = scratch.borrow().take_glwe_slice_scratch(T::BITS as usize, out);

        // Evaluates out[i] = circuit[i](a, b)
        self.execute_bdd_circuit_multi_thread(threads, &mut out_bits, &helper, circuit, &mut scratch_1);

        // Repacks the bits
        out.pack(self, out_bits, key, &mut scratch_1);
    }
}

struct FheUintHelper<'a, T: UnsignedInteger, BE: Backend<ZnxWord = i64>> {
    data: Vec<&'a dyn GetGGSWBit<BE>>,
    _phantom: PhantomData<T>,
}

impl<'a, T: UnsignedInteger, BE: Backend<ZnxWord = i64>> GetGGSWBit<BE> for FheUintHelper<'a, T, BE> {
    fn get_bit(&self, bit: usize) -> &GGSWPrepared<BE::OwnedBuf, BE> {
        let lo: usize = bit % T::BITS as usize;
        let hi: usize = bit / T::BITS as usize;
        self.data[hi].get_bit(lo)
    }
}

impl<'a, T: UnsignedInteger, BE: Backend<ZnxWord = i64>> BitSize for FheUintHelper<'a, T, BE> {
    fn bit_size(&self) -> usize {
        T::BITS as usize * self.data.len()
    }
}

#[macro_export]
macro_rules! define_bdd_2w_to_1w_trait {
    ($(#[$meta:meta])* $vis:vis $trait_name:ident, $method_name:ident) => {
        paste::paste! {
            $(#[$meta])*
            $vis trait $trait_name<T: UnsignedInteger, BE: Backend> {

                /// Single-threaded version
                fn $method_name<M, H>(
                    &mut self,
                    module: &M,
                    a: &FheUintPrepared<BE::OwnedBuf, T, BE>,
                    b: &FheUintPrepared<BE::OwnedBuf, T, BE>,
                    key: &H,
                    scratch: &mut ScratchArena<'_, BE>,
                ) where
                    M: GLWEBytesOf<BE> + ExecuteBDDCircuit2WTo1W<BE>,
                    H: GetAutomorphismKey<BE>,
                    BE: Backend<ZnxWord = i64>,
                    Self: GLWEToBackendMut<BE, State = Normalized>;

                /// Multithreaded version – same vis, method_name + "_multi_thread"
                fn [<$method_name _multi_thread>]<M, H>(
                    &mut self,
                    threads: usize,
                    module: &M,
                    a: &FheUintPrepared<BE::OwnedBuf, T, BE>,
                    b: &FheUintPrepared<BE::OwnedBuf, T, BE>,
                    key: &H,
                    scratch: &mut ScratchArena<'_, BE>,
                ) where
                    M: GLWEBytesOf<BE> + ExecuteBDDCircuit2WTo1W<BE>,
                    H: GetAutomorphismKey<BE>,
                    BE: Backend<ZnxWord = i64>,
                    Self: GLWEToBackendMut<BE, State = Normalized>;

                fn [<$method_name _tmp_bytes>]<M, R, G, H>(
                    &self,
                    module: &M,
                    res_infos: &R,
                    ggsw_infos: &G,
                    key: &H,
                ) -> usize
                where
                    M: GLWEBytesOf<BE> + ExecuteBDDCircuit2WTo1W<BE>,
                    R: GLWEInfos,
                    G: GGSWInfos,
                    H: GetAutomorphismKey<BE>,
                    BE: Backend<ZnxWord = i64>;

                fn [<$method_name _multi_thread_tmp_bytes>]<M, R, G, H>(
                    &self,
                    module: &M,
                    threads: usize,
                    res_infos: &R,
                    ggsw_infos: &G,
                    key: &H,
                ) -> usize
                where
                    M: GLWEBytesOf<BE> + ExecuteBDDCircuit2WTo1W<BE>,
                    R: GLWEInfos,
                    G: GGSWInfos,
                    H: GetAutomorphismKey<BE>,
                    BE: Backend<ZnxWord = i64>;
            }
        }
    };
}

#[macro_export]
macro_rules! impl_bdd_2w_to_1w_trait {
    ($trait_name:ident, $method_name:ident, $ty:ty, $circuit_ty:ty, $output_circuits:path) => {
        paste::paste! {
            impl<BE: Backend<ZnxWord = i64>> $trait_name<$ty, BE> for FheUint<BE::OwnedBuf, $ty, BE::ZnxWord> {

                fn $method_name<M, H>(
                    &mut self,
                    module: &M,
                    a: &FheUintPrepared<BE::OwnedBuf, $ty, BE>,
                    b: &FheUintPrepared<BE::OwnedBuf, $ty, BE>,
                    key: &H,
                    scratch: &mut ScratchArena<'_, BE>,
                ) where
                    M: GLWEBytesOf<BE> + ExecuteBDDCircuit2WTo1W<BE>,
                    H: GetAutomorphismKey<BE>,
                    BE: Backend<ZnxWord = i64>,
                {
                    module.execute_bdd_circuit_2w_to_1w(self, &$output_circuits, a, b, key, scratch)
                }

                fn [<$method_name _multi_thread>]<M, H>(
                    &mut self,
                    threads: usize,
                    module: &M,
                    a: &FheUintPrepared<BE::OwnedBuf, $ty, BE>,
                    b: &FheUintPrepared<BE::OwnedBuf, $ty, BE>,
                    key: &H,
                    scratch: &mut ScratchArena<'_, BE>,
                ) where
                    M: GLWEBytesOf<BE> + ExecuteBDDCircuit2WTo1W<BE>,
                    H: GetAutomorphismKey<BE>,
                    BE: Backend<ZnxWord = i64>,
                {
                    module.execute_bdd_circuit_2w_to_1w_multi_thread(threads, self, &$output_circuits, a, b, key, scratch)
                }

                fn [<$method_name _tmp_bytes>]<M, R, G, H>(
                    &self,
                    module: &M,
                    res_infos: &R,
                    ggsw_infos: &G,
                    key: &H,
                ) -> usize
                where
                    M: GLWEBytesOf<BE> + ExecuteBDDCircuit2WTo1W<BE>,
                    R: GLWEInfos,
                    G: GGSWInfos,
                    H: GetAutomorphismKey<BE>,
                    BE: Backend<ZnxWord = i64>,
                {
                    module.execute_bdd_circuit_2w_to_1w_tmp_bytes::<_, $ty, _, _, _>(
                        &$output_circuits,
                        res_infos,
                        ggsw_infos,
                        key,
                    )
                }

                fn [<$method_name _multi_thread_tmp_bytes>]<M, R, G, H>(
                    &self,
                    module: &M,
                    threads: usize,
                    res_infos: &R,
                    ggsw_infos: &G,
                    key: &H,
                ) -> usize
                where
                    M: GLWEBytesOf<BE> + ExecuteBDDCircuit2WTo1W<BE>,
                    R: GLWEInfos,
                    G: GGSWInfos,
                    H: GetAutomorphismKey<BE>,
                    BE: Backend<ZnxWord = i64>,
                {
                    module.execute_bdd_circuit_2w_to_1w_multi_thread_tmp_bytes::<_, $ty, _, _, _>(
                        threads,
                        &$output_circuits,
                        res_infos,
                        ggsw_infos,
                        key,
                    )
                }
            }
        }
    };
}

define_bdd_2w_to_1w_trait!(
    /// Homomorphic addition (`out = a + b`).
    pub Add, add);
define_bdd_2w_to_1w_trait!(
    /// Homomorphic subtraction (`out = a - b`).
    pub Sub, sub);
define_bdd_2w_to_1w_trait!(
    /// Homomorphic logical left shift (`out = a << b`).
    pub Sll, sll);
define_bdd_2w_to_1w_trait!(
    /// Homomorphic arithmetic right shift (`out = a >> b`, sign-extending).
    pub Sra, sra);
define_bdd_2w_to_1w_trait!(
    /// Homomorphic logical right shift (`out = a >> b`, zero-extending).
    pub Srl, srl);
define_bdd_2w_to_1w_trait!(
    /// Homomorphic signed less-than comparison (`out = (a as signed) < (b as signed)`).
    ///
    /// The result is `1` (all-bits-set in the packed-GLWE encoding) when
    /// the signed interpretation of `a` is strictly less than that of `b`,
    /// and `0` otherwise.
    pub Slt, slt);
define_bdd_2w_to_1w_trait!(
    /// Homomorphic unsigned less-than comparison (`out = a < b`).
    ///
    /// The result is `1` when the unsigned value of `a` is strictly less than
    /// that of `b`, and `0` otherwise.
    pub Sltu, sltu);
define_bdd_2w_to_1w_trait!(
    /// Homomorphic bitwise OR (`out = a | b`).
    pub Or, or);
define_bdd_2w_to_1w_trait!(
    /// Homomorphic bitwise AND (`out = a & b`).
    pub And, and);
define_bdd_2w_to_1w_trait!(
    /// Homomorphic bitwise XOR (`out = a ^ b`).
    pub Xor, xor);

impl_bdd_2w_to_1w_trait!(
    Add,
    add,
    u32,
    circuits::u32::add_codegen::AnyBitCircuit,
    circuits::u32::add_codegen::OUTPUT_CIRCUITS
);

impl_bdd_2w_to_1w_trait!(
    Sub,
    sub,
    u32,
    circuits::u32::sub_codegen::AnyBitCircuit,
    circuits::u32::sub_codegen::OUTPUT_CIRCUITS
);

impl_bdd_2w_to_1w_trait!(
    Sll,
    sll,
    u32,
    circuits::u32::sll_codegen::AnyBitCircuit,
    circuits::u32::sll_codegen::OUTPUT_CIRCUITS
);

impl_bdd_2w_to_1w_trait!(
    Sra,
    sra,
    u32,
    circuits::u32::sra_codegen::AnyBitCircuit,
    circuits::u32::sra_codegen::OUTPUT_CIRCUITS
);

impl_bdd_2w_to_1w_trait!(
    Srl,
    srl,
    u32,
    circuits::u32::srl_codegen::AnyBitCircuit,
    circuits::u32::srl_codegen::OUTPUT_CIRCUITS
);

impl_bdd_2w_to_1w_trait!(
    Slt,
    slt,
    u32,
    circuits::u32::slt_codegen::AnyBitCircuit,
    circuits::u32::slt_codegen::OUTPUT_CIRCUITS
);

impl_bdd_2w_to_1w_trait!(
    Sltu,
    sltu,
    u32,
    circuits::u32::sltu_codegen::AnyBitCircuit,
    circuits::u32::sltu_codegen::OUTPUT_CIRCUITS
);

impl_bdd_2w_to_1w_trait!(
    And,
    and,
    u32,
    circuits::u32::and_codegen::AnyBitCircuit,
    circuits::u32::and_codegen::OUTPUT_CIRCUITS
);

impl_bdd_2w_to_1w_trait!(
    Or,
    or,
    u32,
    circuits::u32::or_codegen::AnyBitCircuit,
    circuits::u32::or_codegen::OUTPUT_CIRCUITS
);

impl_bdd_2w_to_1w_trait!(
    Xor,
    xor,
    u32,
    circuits::u32::xor_codegen::AnyBitCircuit,
    circuits::u32::xor_codegen::OUTPUT_CIRCUITS
);
