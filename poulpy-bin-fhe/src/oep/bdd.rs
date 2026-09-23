//! Binary decision diagram operation oep.
mod executebdd_circuit;
pub use executebdd_circuit::*;
mod cswap;
pub use cswap::*;
mod cmux;
pub use cmux::*;
mod ggsw_blind_rotation;
pub use ggsw_blind_rotation::*;
mod glwe_blind_rotation;
pub use glwe_blind_rotation::*;
mod glwe_blind_selection;
pub use glwe_blind_selection::*;
mod glwe_blind_retrieval;
pub use glwe_blind_retrieval::*;
mod executebdd_circuit1w_to1w;
pub use executebdd_circuit1w_to1w::*;
mod executebdd_circuit2w_to1w;
pub use executebdd_circuit2w_to1w::*;
mod fhe_uint_prepared_encrypt_sk;
pub use fhe_uint_prepared_encrypt_sk::*;
mod fhe_uint_prepare;
pub use fhe_uint_prepare::*;
mod bdd_key_encrypt_sk;
pub use bdd_key_encrypt_sk::*;
mod bdd_key_prepared_factory;
pub use bdd_key_prepared_factory::*;

/// Selects canonical BDD implementations and derived defaults.
#[macro_export]
macro_rules! impl_bin_fhe_bdd_reference {
    ($be:ty, $algo:ty) => {
        const _: () = {
            use poulpy_core::{
                layouts::{prepared::*, *},
                *,
            };
            use poulpy_hal::{api::*, layouts::*, source::Source};
            #[allow(unused_imports)]
            use std::{collections::HashMap, marker::PhantomData};
            use $crate::{
                bdd_arithmetic::*,
                blind_rotation::{BlindRotationAlgo, BlindRotationKeyInfos},
                circuit_bootstrapping::*,
            };
            unsafe impl $crate::oep::ExecuteBDDCircuitImpl for $be {
                #[allow(clippy::too_many_arguments)]
                fn execute_bdd_circuit_tmp_bytes<R, G>(
                    module: &Module<$be>,
                    res_infos: &R,
                    state_size: usize,
                    ggsw_infos: &G,
                ) -> usize
                where
                    R: GLWEInfos,
                    G: GGSWInfos,
                {
                    $crate::reference::bdd::execute_bdd_circuit_tmp_bytes_reference::<$be, _, _>(
                        module, res_infos, state_size, ggsw_infos,
                    )
                }
                #[allow(clippy::too_many_arguments)]
                fn execute_bdd_circuit_multi_thread<C, G, O>(
                    module: &Module<$be>,
                    threads: usize,
                    out: &mut [O],
                    inputs: &G,
                    circuit: &C,
                    scratch: &mut ScratchArena<'_, $be>,
                ) where
                    G: GetGGSWBit<$be> + BitSize,
                    C: GetBitCircuitInfo,
                    O: GLWEToBackendMut<$be> + GLWEInfos + Send,
                {
                    $crate::reference::bdd::execute_bdd_circuit_multi_thread_reference::<$be, _, _, _>(
                        module, threads, out, inputs, circuit, scratch,
                    )
                }
            }
        };
        const _: () = {
            use poulpy_core::{
                layouts::{prepared::*, *},
                *,
            };
            use poulpy_hal::{api::*, layouts::*, source::Source};
            #[allow(unused_imports)]
            use std::{collections::HashMap, marker::PhantomData};
            use $crate::{
                bdd_arithmetic::*,
                blind_rotation::{BlindRotationAlgo, BlindRotationKeyInfos},
                circuit_bootstrapping::*,
            };
            unsafe impl $crate::oep::CswapImpl for $be {
                #[allow(clippy::too_many_arguments)]
                fn cswap_tmp_bytes<R, A, S>(module: &Module<$be>, res_a_infos: &R, res_b_infos: &A, s_infos: &S) -> usize
                where
                    R: GLWEInfos,
                    A: GLWEInfos,
                    S: GGSWInfos,
                {
                    $crate::reference::bdd::cswap_tmp_bytes_reference::<$be, _, _, _>(module, res_a_infos, res_b_infos, s_infos)
                }
                #[allow(clippy::too_many_arguments)]
                fn cswap<'k, A, B>(
                    module: &Module<$be>,
                    res_a: &mut A,
                    res_b: &mut B,
                    s: &GGSWPreparedBackendRef<'k, $be>,
                    scratch: &mut ScratchArena<'_, $be>,
                ) where
                    A: GLWEToBackendMut<$be> + GLWEToBackendRef<$be> + GLWEInfos,
                    B: GLWEToBackendMut<$be> + GLWEToBackendRef<$be> + GLWEInfos,
                    $be: 'k,
                {
                    $crate::reference::bdd::cswap_reference::<$be, _, _>(module, res_a, res_b, s, scratch)
                }
            }
        };
        const _: () = {
            use poulpy_core::{
                layouts::{prepared::*, *},
                *,
            };
            use poulpy_hal::{api::*, layouts::*, source::Source};
            #[allow(unused_imports)]
            use std::{collections::HashMap, marker::PhantomData};
            use $crate::{
                bdd_arithmetic::*,
                blind_rotation::{BlindRotationAlgo, BlindRotationKeyInfos},
                circuit_bootstrapping::*,
            };
            unsafe impl $crate::oep::CmuxImpl for $be {
                #[allow(clippy::too_many_arguments)]
                fn cmux_tmp_bytes<R, A, B>(module: &Module<$be>, res_infos: &R, a_infos: &A, selector_infos: &B) -> usize
                where
                    R: GLWEInfos,
                    A: GLWEInfos,
                    B: GGSWInfos,
                {
                    $crate::reference::bdd::cmux_tmp_bytes_reference::<$be, _, _, _>(module, res_infos, a_infos, selector_infos)
                }
                #[allow(clippy::too_many_arguments)]
                fn cmux<'k, R, T, F>(
                    module: &Module<$be>,
                    res: &mut R,
                    t: &T,
                    f: &F,
                    s: &GGSWPreparedBackendRef<'k, $be>,
                    scratch: &mut ScratchArena<'_, $be>,
                ) where
                    R: GLWEToBackendMut<$be> + GLWEInfos,
                    T: GLWEToBackendRef<$be>,
                    F: GLWEToBackendRef<$be>,
                    $be: 'k,
                {
                    $crate::reference::bdd::cmux_reference::<$be, _, T, _>(module, res, t, f, s, scratch)
                }
                #[allow(clippy::too_many_arguments)]
                fn cmux_assign_neg<'k, R, A>(
                    module: &Module<$be>,
                    res: &mut R,
                    a: &A,
                    s: &GGSWPreparedBackendRef<'k, $be>,
                    scratch: &mut ScratchArena<'_, $be>,
                ) where
                    R: GLWEToBackendMut<$be> + GLWEInfos,
                    A: GLWEToBackendRef<$be>,
                    $be: 'k,
                {
                    $crate::reference::bdd::cmux_assign_neg_reference::<$be, _, _>(module, res, a, s, scratch)
                }
                #[allow(clippy::too_many_arguments)]
                fn cmux_assign<'k, R, A>(
                    module: &Module<$be>,
                    res: &mut R,
                    a: &A,
                    s: &GGSWPreparedBackendRef<'k, $be>,
                    scratch: &mut ScratchArena<'_, $be>,
                ) where
                    R: GLWEToBackendMut<$be> + GLWEInfos,
                    A: GLWEToBackendRef<$be>,
                    $be: 'k,
                {
                    $crate::reference::bdd::cmux_assign_reference::<$be, _, _>(module, res, a, s, scratch)
                }
            }
        };
        const _: () = {
            use poulpy_core::{
                layouts::{prepared::*, *},
                *,
            };
            use poulpy_hal::{api::*, layouts::*, source::Source};
            #[allow(unused_imports)]
            use std::{collections::HashMap, marker::PhantomData};
            use $crate::{
                bdd_arithmetic::*,
                blind_rotation::{BlindRotationAlgo, BlindRotationKeyInfos},
                circuit_bootstrapping::*,
            };
            unsafe impl<T: UnsignedInteger> $crate::oep::GGSWBlindRotationImpl<T> for $be {
                #[allow(clippy::too_many_arguments)]
                fn scalar_to_ggsw_blind_rotation_tmp_bytes<R, K>(module: &Module<$be>, res_infos: &R, k_infos: &K) -> usize
                where
                    R: GLWEInfos,
                    K: GGSWInfos,
                {
                    $crate::reference::bdd::scalar_to_ggsw_blind_rotation_tmp_bytes_reference::<T, $be, _, _>(
                        module, res_infos, k_infos,
                    )
                }
                #[allow(clippy::too_many_arguments)]
                fn scalar_to_ggsw_blind_rotation<R, A, K>(
                    module: &Module<$be>,
                    res: &mut R,
                    test_vector: &A,
                    fhe_uint: &K,
                    sign: bool,
                    bit_rsh: usize,
                    bit_mask: usize,
                    bit_lsh: usize,
                    scratch: &mut ScratchArena<'_, $be>,
                ) where
                    R: GGSWToBackendMut<$be> + GGSWAtViewMut<$be> + GGSWInfos,
                    A: ScalarZnxToBackendRef<$be>,
                    K: GetGGSWBit<$be>,
                    $be: Backend<ZnxWord = i64>,
                {
                    $crate::reference::bdd::scalar_to_ggsw_blind_rotation_reference::<T, $be, _, _, _>(
                        module,
                        res,
                        test_vector,
                        fhe_uint,
                        sign,
                        bit_rsh,
                        bit_mask,
                        bit_lsh,
                        scratch,
                    )
                }
            }
        };
        const _: () = {
            use poulpy_core::{
                layouts::{prepared::*, *},
                *,
            };
            use poulpy_hal::{api::*, layouts::*, source::Source};
            #[allow(unused_imports)]
            use std::{collections::HashMap, marker::PhantomData};
            use $crate::{
                bdd_arithmetic::*,
                blind_rotation::{BlindRotationAlgo, BlindRotationKeyInfos},
                circuit_bootstrapping::*,
            };
            unsafe impl $crate::oep::GLWEBlindRotationImpl for $be {
                #[allow(clippy::too_many_arguments)]
                fn glwe_blind_rotation_tmp_bytes<R, A, K>(module: &Module<$be>, res_infos: &R, a_infos: &A, k_infos: &K) -> usize
                where
                    R: GLWEInfos,
                    A: GLWEInfos,
                    K: GGSWInfos,
                {
                    $crate::reference::bdd::glwe_blind_rotation_tmp_bytes_reference::<$be, _, _, _>(
                        module, res_infos, a_infos, k_infos,
                    )
                }
                fn glwe_blind_rotation_assign_tmp_bytes<R, K>(module: &Module<$be>, res_infos: &R, k_infos: &K) -> usize
                where
                    R: GLWEInfos,
                    K: GGSWInfos,
                {
                    $crate::reference::bdd::glwe_blind_rotation_assign_tmp_bytes_reference::<$be, _, _>(
                        module, res_infos, k_infos,
                    )
                }
                #[allow(clippy::too_many_arguments)]
                fn glwe_blind_rotation_assign<R, K>(
                    module: &Module<$be>,
                    res: &mut R,
                    value: &K,
                    sign: bool,
                    bit_rsh: usize,
                    bit_mask: usize,
                    bit_lsh: usize,
                    scratch: &mut ScratchArena<'_, $be>,
                ) where
                    R: GLWEToBackendMut<$be> + GLWEInfos,
                    K: GetGGSWBit<$be>,
                    $be: Backend<ZnxWord = i64>,
                {
                    $crate::reference::bdd::glwe_blind_rotation_assign_reference::<$be, _, _>(
                        module, res, value, sign, bit_rsh, bit_mask, bit_lsh, scratch,
                    )
                }
                #[allow(clippy::too_many_arguments)]
                fn glwe_blind_rotation<R, A, K>(
                    module: &Module<$be>,
                    res: &mut R,
                    a: &A,
                    fhe_uint: &K,
                    sign: bool,
                    bit_rsh: usize,
                    bit_mask: usize,
                    bit_lsh: usize,
                    scratch: &mut ScratchArena<'_, $be>,
                ) where
                    R: GLWEToBackendMut<$be> + GLWEInfos,
                    A: GLWEToBackendRef<$be>,
                    K: GetGGSWBit<$be>,
                    $be: Backend<ZnxWord = i64>,
                {
                    $crate::reference::bdd::glwe_blind_rotation_reference::<$be, _, _, _>(
                        module, res, a, fhe_uint, sign, bit_rsh, bit_mask, bit_lsh, scratch,
                    )
                }
            }
        };
        const _: () = {
            use poulpy_core::{
                layouts::{prepared::*, *},
                *,
            };
            use poulpy_hal::{api::*, layouts::*, source::Source};
            #[allow(unused_imports)]
            use std::{collections::HashMap, marker::PhantomData};
            use $crate::{
                bdd_arithmetic::*,
                blind_rotation::{BlindRotationAlgo, BlindRotationKeyInfos},
                circuit_bootstrapping::*,
            };
            unsafe impl<T: UnsignedInteger> $crate::oep::GLWEBlindSelectionImpl<T> for $be {
                #[allow(clippy::too_many_arguments)]
                fn glwe_blind_selection_tmp_bytes<R, A, K>(
                    module: &Module<$be>,
                    res_infos: &R,
                    input_infos: &[A],
                    k_infos: &K,
                ) -> usize
                where
                    R: GLWEInfos,
                    A: GLWEInfos,
                    K: GGSWInfos,
                {
                    $crate::reference::bdd::glwe_blind_selection_tmp_bytes_reference::<T, $be, _, _, _>(
                        module,
                        res_infos,
                        input_infos,
                        k_infos,
                    )
                }
                #[allow(clippy::too_many_arguments)]
                fn glwe_blind_selection<R, A, K>(
                    module: &Module<$be>,
                    res: &mut R,
                    a: HashMap<usize, &mut A>,
                    fhe_uint: &K,
                    bit_rsh: usize,
                    bit_mask: usize,
                    scratch: &mut ScratchArena<'_, $be>,
                ) where
                    R: GLWEToBackendMut<$be> + GLWEInfos,
                    A: GLWEToBackendMut<$be> + GLWEToBackendRef<$be> + GLWEInfos,
                    K: GetGGSWBit<$be>,
                {
                    $crate::reference::bdd::glwe_blind_selection_reference::<T, $be, _, _, _>(
                        module, res, a, fhe_uint, bit_rsh, bit_mask, scratch,
                    )
                }
            }
        };
        const _: () = {
            use poulpy_core::{
                layouts::{prepared::*, *},
                *,
            };
            use poulpy_hal::{api::*, layouts::*, source::Source};
            #[allow(unused_imports)]
            use std::{collections::HashMap, marker::PhantomData};
            use $crate::{
                bdd_arithmetic::*,
                blind_rotation::{BlindRotationAlgo, BlindRotationKeyInfos},
                circuit_bootstrapping::*,
            };
            unsafe impl $crate::oep::GLWEBlindRetrievalImpl for $be {}
        };
        const _: () = {
            use poulpy_core::{
                layouts::{prepared::*, *},
                *,
            };
            use poulpy_hal::{api::*, layouts::*, source::Source};
            #[allow(unused_imports)]
            use std::{collections::HashMap, marker::PhantomData};
            use $crate::{
                bdd_arithmetic::*,
                blind_rotation::{BlindRotationAlgo, BlindRotationKeyInfos},
                circuit_bootstrapping::*,
            };
            unsafe impl $crate::oep::ExecuteBDDCircuit1WTo1WImpl for $be {
                fn execute_bdd_circuit_1w_to_1w_multi_thread_tmp_bytes<C, T, R, G, H>(
                    module: &Module<$be>,
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
                    H: GetAutomorphismKey<$be>,
                {
                    $crate::reference::bdd::execute_bdd_circuit_1w_to_1w_multi_thread_tmp_bytes_reference::<$be, C, T, R, G, H>(
                        module, threads, circuit, res_infos, ggsw_infos, key,
                    )
                }
                #[allow(clippy::too_many_arguments)]
                fn execute_bdd_circuit_1w_to_1w_multi_thread<C, H, T>(
                    module: &Module<$be>,
                    threads: usize,
                    out: &mut FheUint<Self::OwnedBuf, T, Self::ZnxWord>,
                    circuit: &C,
                    a: &FheUintPrepared<Self::OwnedBuf, T, $be>,
                    key: &H,
                    scratch: &mut ScratchArena<'_, $be>,
                ) where
                    T: UnsignedInteger,
                    C: GetBitCircuitInfo,
                    H: GetAutomorphismKey<$be>,
                    $be: Backend<ZnxWord = i64>,
                {
                    $crate::reference::bdd::execute_bdd_circuit_1w_to_1w_multi_thread_reference::<$be, _, _, T>(
                        module, threads, out, circuit, a, key, scratch,
                    )
                }
            }
        };
        const _: () = {
            use poulpy_core::{
                layouts::{prepared::*, *},
                *,
            };
            use poulpy_hal::{api::*, layouts::*, source::Source};
            #[allow(unused_imports)]
            use std::{collections::HashMap, marker::PhantomData};
            use $crate::{
                bdd_arithmetic::*,
                blind_rotation::{BlindRotationAlgo, BlindRotationKeyInfos},
                circuit_bootstrapping::*,
            };
            unsafe impl $crate::oep::ExecuteBDDCircuit2WTo1WImpl for $be {
                #[allow(clippy::too_many_arguments)]
                fn execute_bdd_circuit_2w_to_1w_multi_thread_tmp_bytes<C, T, R, G, H>(
                    module: &Module<$be>,
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
                    H: GetAutomorphismKey<$be>,
                {
                    $crate::reference::bdd::execute_bdd_circuit_2w_to_1w_multi_thread_tmp_bytes_reference::<$be, _, T, _, _, _>(
                        module, threads, circuit, res_infos, ggsw_infos, key,
                    )
                }
                #[allow(clippy::too_many_arguments)]
                fn execute_bdd_circuit_2w_to_1w_multi_thread<C, H, T>(
                    module: &Module<$be>,
                    threads: usize,
                    out: &mut FheUint<Self::OwnedBuf, T, Self::ZnxWord>,
                    circuit: &C,
                    a: &FheUintPrepared<Self::OwnedBuf, T, $be>,
                    b: &FheUintPrepared<Self::OwnedBuf, T, $be>,
                    key: &H,
                    scratch: &mut ScratchArena<'_, $be>,
                ) where
                    T: UnsignedInteger,
                    C: GetBitCircuitInfo,
                    H: GetAutomorphismKey<$be>,
                    $be: Backend<ZnxWord = i64>,
                {
                    $crate::reference::bdd::execute_bdd_circuit_2w_to_1w_multi_thread_reference::<$be, _, _, T>(
                        module, threads, out, circuit, a, b, key, scratch,
                    )
                }
            }
        };
        const _: () = {
            use poulpy_core::{
                layouts::{prepared::*, *},
                *,
            };
            use poulpy_hal::{api::*, layouts::*, source::Source};
            #[allow(unused_imports)]
            use std::{collections::HashMap, marker::PhantomData};
            use $crate::{
                bdd_arithmetic::*,
                blind_rotation::{BlindRotationAlgo, BlindRotationKeyInfos},
                circuit_bootstrapping::*,
            };
            unsafe impl<T: UnsignedInteger + ToBits> $crate::oep::FheUintPreparedEncryptSkImpl<T> for $be {
                fn fhe_uint_prepared_encrypt_sk_tmp_bytes<A: GGSWInfos>(module: &Module<$be>, infos: &A) -> usize {
                    $crate::reference::bdd::fhe_uint_prepared_encrypt_sk_tmp_bytes_reference(module, infos)
                }
                #[allow(clippy::too_many_arguments)]
                fn fhe_uint_prepared_encrypt_sk<S, E>(
                    module: &Module<$be>,
                    res: &mut FheUintPrepared<Self::OwnedBuf, T, $be>,
                    value: T,
                    sk: &S,
                    enc_infos: &E,
                    source_xe: &mut Source,
                    source_xa: &mut Source,
                    scratch: &mut ScratchArena<'_, $be>,
                ) where
                    S: GLWESecretPreparedToBackendRef<$be> + GLWEInfos,
                    E: EncryptionInfos,
                {
                    $crate::reference::bdd::fhe_uint_prepared_encrypt_sk_reference::<T, $be, _, _>(
                        module, res, value, sk, enc_infos, source_xe, source_xa, scratch,
                    )
                }
            }
        };
        const _: () = {
            use poulpy_core::{
                layouts::{prepared::*, *},
                *,
            };
            use poulpy_hal::{api::*, layouts::*, source::Source};
            #[allow(unused_imports)]
            use std::{collections::HashMap, marker::PhantomData};
            use $crate::{
                bdd_arithmetic::*,
                blind_rotation::{BlindRotationAlgo, BlindRotationKeyInfos},
                circuit_bootstrapping::*,
            };
            unsafe impl $crate::oep::FheUintPrepareImpl<$algo> for $be {
                #[allow(clippy::too_many_arguments)]
                fn fhe_uint_prepare_tmp_bytes<R, A, B>(
                    module: &Module<$be>,
                    block_size: usize,
                    extension_factor: usize,
                    res_infos: &R,
                    bits_infos: &A,
                    bdd_infos: &B,
                ) -> usize
                where
                    R: GGSWInfos,
                    A: GLWEInfos,
                    B: BDDKeyInfos,
                {
                    $crate::reference::bdd::fhe_uint_prepare_tmp_bytes_reference::<$algo, $be, _, _, _>(
                        module,
                        block_size,
                        extension_factor,
                        res_infos,
                        bits_infos,
                        bdd_infos,
                    )
                }
                #[allow(clippy::too_many_arguments)]
                fn fhe_uint_prepare_custom_multi_thread<K, T: UnsignedInteger>(
                    module: &Module<$be>,
                    threads: usize,
                    res: &mut FheUintPrepared<Self::OwnedBuf, T, $be>,
                    bits: &FheUint<Self::OwnedBuf, T, Self::ZnxWord>,
                    bit_start: usize,
                    bit_count: usize,
                    key: &K,
                    scratch: &mut ScratchArena<'_, $be>,
                ) where
                    K: BDDKeyHelper<Self::OwnedBuf, $algo, $be> + BDDKeyInfos,
                {
                    $crate::reference::bdd::fhe_uint_prepare_custom_multi_thread_reference::<$algo, $be, _, T>(
                        module, threads, res, bits, bit_start, bit_count, key, scratch,
                    )
                }
            }
        };
        const _: () = {
            use poulpy_core::{
                layouts::{prepared::*, *},
                *,
            };
            use poulpy_hal::{api::*, layouts::*, source::Source};
            #[allow(unused_imports)]
            use std::{collections::HashMap, marker::PhantomData};
            use $crate::{
                bdd_arithmetic::*,
                blind_rotation::{BlindRotationAlgo, BlindRotationKeyInfos},
                circuit_bootstrapping::*,
            };
            unsafe impl $crate::oep::BDDKeyEncryptSkImpl<$algo> for $be {
                #[allow(clippy::too_many_arguments)]
                fn bdd_key_encrypt_sk_tmp_bytes<A>(module: &Module<$be>, infos: &A) -> usize
                where
                    A: BDDKeyInfos,
                {
                    $crate::reference::bdd::bdd_key_encrypt_sk_tmp_bytes_reference::<$algo, $be, _>(module, infos)
                }
                #[allow(clippy::too_many_arguments)]
                fn bdd_key_encrypt_sk<S0, S1>(
                    module: &Module<$be>,
                    res: &mut BDDKey<Self::OwnedBuf, $algo, Self::ZnxWord>,
                    sk_lwe: &S0,
                    sk_glwe: &S1,
                    enc_infos: &BDDEncryptionInfos,
                    source_xe: &mut Source,
                    source_xa: &mut Source,
                    scratch: &mut ScratchArena<'_, $be>,
                ) where
                    S0: LWESecretToBackendRef<$be> + GetDistribution + LWEInfos,
                    S1: GLWESecretToBackendRef<$be> + GetDistribution + GLWEInfos,
                {
                    $crate::reference::bdd::bdd_key_encrypt_sk_reference::<$algo, $be, _, _>(
                        module, res, sk_lwe, sk_glwe, enc_infos, source_xe, source_xa, scratch,
                    )
                }
            }
        };
        const _: () = {
            use poulpy_core::{
                layouts::{prepared::*, *},
                *,
            };
            use poulpy_hal::{api::*, layouts::*, source::Source};
            #[allow(unused_imports)]
            use std::{collections::HashMap, marker::PhantomData};
            use $crate::{
                bdd_arithmetic::*,
                blind_rotation::{BlindRotationAlgo, BlindRotationKeyInfos},
                circuit_bootstrapping::*,
            };
            unsafe impl $crate::oep::BDDKeyPreparedFactoryImpl<$algo> for $be {
                #[allow(clippy::too_many_arguments)]
                fn alloc_bdd_key_from_infos<A>(module: &Module<$be>, infos: &A) -> BDDKeyPrepared<Self::OwnedBuf, $algo, $be>
                where
                    A: BDDKeyInfos,
                {
                    $crate::reference::bdd::alloc_bdd_key_from_infos_reference::<$algo, $be, _>(module, infos)
                }
                #[allow(clippy::too_many_arguments)]
                fn prepare_bdd_key_tmp_bytes<A>(module: &Module<$be>, infos: &A) -> usize
                where
                    A: BDDKeyInfos,
                {
                    $crate::reference::bdd::prepare_bdd_key_tmp_bytes_reference::<$algo, $be, _>(module, infos)
                }
                #[allow(clippy::too_many_arguments)]
                fn prepare_bdd_key(
                    module: &Module<$be>,
                    res: &mut BDDKeyPrepared<Self::OwnedBuf, $algo, $be>,
                    other: &BDDKey<Self::OwnedBuf, $algo, Self::ZnxWord>,
                    scratch: &mut ScratchArena<'_, $be>,
                ) {
                    $crate::reference::bdd::prepare_bdd_key_reference::<$algo, $be>(module, res, other, scratch)
                }
            }
        };
    };
}
