pub use crate::api::ExecuteBDDCircuit2WTo1W;
use std::marker::PhantomData;

use poulpy_core::layouts::{GGSWInfos, GLWEInfos, GLWEToBackendMut, GetAutomorphismKey, prepared::GGSWPrepared};
use poulpy_hal::layouts::{Backend, ScratchArena};

use crate::bdd_arithmetic::{BitSize, FheUint, FheUintPrepared, GetGGSWBit, UnsignedInteger, circuits};
use poulpy_core::GLWEBytesOf;

pub(crate) struct FheUintHelper<'a, T: UnsignedInteger, BE: Backend<ZnxWord = i64>> {
    pub(crate) data: Vec<&'a dyn GetGGSWBit<BE>>,
    pub(crate) _phantom: PhantomData<T>,
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
                    Self: GLWEToBackendMut<BE>;

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
                    Self: GLWEToBackendMut<BE>;

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
