//! Binary decision diagram operation oep.
mod bdd_circuit_execute;
pub use bdd_circuit_execute::*;
mod glwe_cswap;
pub use glwe_cswap::*;
mod glwe_cmux;
pub use glwe_cmux::*;
mod ggsw_blind_rotation;
pub use ggsw_blind_rotation::*;
mod glwe_blind_rotation;
pub use glwe_blind_rotation::*;
mod glwe_blind_selection;
pub use glwe_blind_selection::*;
mod glwe_blind_retrieval;
pub use glwe_blind_retrieval::*;
mod bdd_circuit_execute_1w_to_1w;
pub use bdd_circuit_execute_1w_to_1w::*;
mod bdd_circuit_execute_2w_to_1w;
pub use bdd_circuit_execute_2w_to_1w::*;
mod fhe_uint_prepared_encrypt_sk;
pub use fhe_uint_prepared_encrypt_sk::*;
mod fhe_uint_prepare;
pub use fhe_uint_prepare::*;
mod bdd_key_encrypt_sk;
pub use bdd_key_encrypt_sk::*;
mod bdd_key_prepared_factory;
pub use bdd_key_prepared_factory::*;

/// Selects the reference implementation of [`BDDCircuitExecuteImpl`].
#[macro_export]
macro_rules! impl_bdd_circuit_execute_reference_full {
    ($be:ty) => {
        const _: () = {
            use ::poulpy_core::{
                layouts::{prepared::*, *},
                *,
            };
            use ::poulpy_hal::{api::*, layouts::*, source::Source};
            #[allow(unused_imports)]
            use ::std::{collections::HashMap, marker::PhantomData};
            use $crate::{
                bdd_arithmetic::*,
                blind_rotation::{BlindRotationAlgo, BlindRotationKeyInfos},
                circuit_bootstrapping::*,
            };
            unsafe impl $crate::oep::BDDCircuitExecuteImpl for $be {
                #[allow(clippy::too_many_arguments)]
                fn bdd_circuit_execute_tmp_bytes<R, G>(
                    module: &Module<$be>,
                    state_size: usize,
                    res_infos: &R,
                    ggsw_infos: &G,
                ) -> usize
                where
                    R: GLWEInfos,
                    G: GGSWInfos,
                {
                    <::poulpy_hal::layouts::Module<$be> as $crate::reference::bdd::BDDCircuitExecuteReference<$be>>::bdd_circuit_execute_tmp_bytes_reference(
                        module, state_size, res_infos, ggsw_infos,
                    )
                }
                #[allow(clippy::too_many_arguments)]
                fn bdd_circuit_execute_multi_thread<C, G, O>(
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
                    <::poulpy_hal::layouts::Module<$be> as $crate::reference::bdd::BDDCircuitExecuteReference<$be>>::bdd_circuit_execute_multi_thread_reference(
                        module, threads, out, inputs, circuit, scratch,
                    )
                }
            }
        };
    };
}

/// Selects the reference implementation of [`GLWECswapImpl`].
#[macro_export]
macro_rules! impl_glwe_cswap_reference_full {
    ($be:ty) => {
        const _: () = {
            use ::poulpy_core::{
                layouts::{prepared::*, *},
                *,
            };
            use ::poulpy_hal::{api::*, layouts::*, source::Source};
            #[allow(unused_imports)]
            use ::std::{collections::HashMap, marker::PhantomData};
            use $crate::{
                bdd_arithmetic::*,
                blind_rotation::{BlindRotationAlgo, BlindRotationKeyInfos},
                circuit_bootstrapping::*,
            };
            unsafe impl $crate::oep::GLWECswapImpl for $be {
                #[allow(clippy::too_many_arguments)]
                fn glwe_cswap_assign_tmp_bytes<R, A, S>(
                    module: &Module<$be>,
                    res_a_infos: &R,
                    res_b_infos: &A,
                    s_infos: &S,
                ) -> usize
                where
                    R: GLWEInfos,
                    A: GLWEInfos,
                    S: GGSWInfos,
                {
                    <::poulpy_hal::layouts::Module<$be> as $crate::reference::bdd::GLWECswapReference<$be>>::glwe_cswap_assign_tmp_bytes_reference(
                        module,
                        res_a_infos,
                        res_b_infos,
                        s_infos,
                    )
                }
                #[allow(clippy::too_many_arguments)]
                fn glwe_cswap_assign<A, B>(
                    module: &Module<$be>,
                    res_a: &mut A,
                    res_b: &mut B,
                    s: &GGSWPreparedBackendRef<'_, $be>,
                    scratch: &mut ScratchArena<'_, $be>,
                ) where
                    A: GLWEToBackendMut<$be> + GLWEToBackendRef<$be> + GLWEInfos,
                    B: GLWEToBackendMut<$be> + GLWEToBackendRef<$be> + GLWEInfos,
                {
                    <::poulpy_hal::layouts::Module<$be> as $crate::reference::bdd::GLWECswapReference<$be>>::glwe_cswap_assign_reference(module, res_a, res_b, s, scratch)
                }
            }
        };
    };
}

/// Selects the reference implementation of [`GLWECmuxImpl`].
#[macro_export]
macro_rules! impl_glwe_cmux_reference_full {
    ($be:ty) => {
        const _: () = {
            use ::poulpy_core::{
                layouts::{prepared::*, *},
                *,
            };
            use ::poulpy_hal::{api::*, layouts::*, source::Source};
            #[allow(unused_imports)]
            use ::std::{collections::HashMap, marker::PhantomData};
            use $crate::{
                bdd_arithmetic::*,
                blind_rotation::{BlindRotationAlgo, BlindRotationKeyInfos},
                circuit_bootstrapping::*,
            };
            unsafe impl $crate::oep::GLWECmuxImpl for $be {
                #[allow(clippy::too_many_arguments)]
                fn glwe_cmux_tmp_bytes<R, A, B>(module: &Module<$be>, res_infos: &R, a_infos: &A, selector_infos: &B) -> usize
                where
                    R: GLWEInfos,
                    A: GLWEInfos,
                    B: GGSWInfos,
                {
                    <::poulpy_hal::layouts::Module<$be> as $crate::reference::bdd::GLWECmuxReference<$be>>::glwe_cmux_tmp_bytes_reference(
                        module,
                        res_infos,
                        a_infos,
                        selector_infos,
                    )
                }
                #[allow(clippy::too_many_arguments)]
                fn glwe_cmux<R, T, F>(
                    module: &Module<$be>,
                    res: &mut R,
                    t: &T,
                    f: &F,
                    s: &GGSWPreparedBackendRef<'_, $be>,
                    scratch: &mut ScratchArena<'_, $be>,
                ) where
                    R: GLWEToBackendMut<$be> + GLWEInfos,
                    T: GLWEToBackendRef<$be>,
                    F: GLWEToBackendRef<$be>,
                {
                    <::poulpy_hal::layouts::Module<$be> as $crate::reference::bdd::GLWECmuxReference<$be>>::glwe_cmux_reference(module, res, t, f, s, scratch)
                }
                #[allow(clippy::too_many_arguments)]
                fn glwe_cmux_neg_assign<R, A>(
                    module: &Module<$be>,
                    res: &mut R,
                    a: &A,
                    s: &GGSWPreparedBackendRef<'_, $be>,
                    scratch: &mut ScratchArena<'_, $be>,
                ) where
                    R: GLWEToBackendMut<$be> + GLWEInfos,
                    A: GLWEToBackendRef<$be>,
                {
                    <::poulpy_hal::layouts::Module<$be> as $crate::reference::bdd::GLWECmuxReference<$be>>::glwe_cmux_neg_assign_reference(module, res, a, s, scratch)
                }
                #[allow(clippy::too_many_arguments)]
                fn glwe_cmux_assign<R, A>(
                    module: &Module<$be>,
                    res: &mut R,
                    a: &A,
                    s: &GGSWPreparedBackendRef<'_, $be>,
                    scratch: &mut ScratchArena<'_, $be>,
                ) where
                    R: GLWEToBackendMut<$be> + GLWEInfos,
                    A: GLWEToBackendRef<$be>,
                {
                    <::poulpy_hal::layouts::Module<$be> as $crate::reference::bdd::GLWECmuxReference<$be>>::glwe_cmux_assign_reference(module, res, a, s, scratch)
                }
            }
        };
    };
}

/// Selects the reference implementation of [`GGSWBlindRotationImpl`].
#[macro_export]
macro_rules! impl_ggsw_blind_rotation_reference_full {
    ($be:ty) => {
        const _: () = {
            use ::poulpy_core::{
                layouts::{prepared::*, *},
                *,
            };
            use ::poulpy_hal::{api::*, layouts::*, source::Source};
            #[allow(unused_imports)]
            use ::std::{collections::HashMap, marker::PhantomData};
            use $crate::{
                bdd_arithmetic::*,
                blind_rotation::{BlindRotationAlgo, BlindRotationKeyInfos},
                circuit_bootstrapping::*,
            };
            unsafe impl<T: UnsignedInteger> $crate::oep::GGSWBlindRotationImpl<T> for $be {
                #[allow(clippy::too_many_arguments)]
                fn ggsw_blind_rotation_from_scalar_tmp_bytes<R, K>(module: &Module<$be>, res_infos: &R, k_infos: &K) -> usize
                where
                    R: GLWEInfos,
                    K: GGSWInfos,
                {
                    <::poulpy_hal::layouts::Module<$be> as $crate::reference::bdd::GGSWBlindRotationReference<T, $be>>::ggsw_blind_rotation_from_scalar_tmp_bytes_reference(
                        module, res_infos, k_infos,
                    )
                }
                #[allow(clippy::too_many_arguments)]
                fn ggsw_blind_rotation_from_scalar<R, A, K>(
                    module: &Module<$be>,
                    sign: bool,
                    bit_rsh: usize,
                    bit_mask: usize,
                    bit_lsh: usize,
                    res: &mut R,
                    test_vector: &A,
                    fhe_uint: &K,
                    scratch: &mut ScratchArena<'_, $be>,
                ) where
                    R: GGSWToBackendMut<$be> + GGSWAtViewMut<$be> + GGSWInfos,
                    A: ScalarZnxToBackendRef<$be>,
                    K: GetGGSWBit<$be>,
                    $be: Backend<ZnxWord = i64>,
                {
                    <::poulpy_hal::layouts::Module<$be> as $crate::reference::bdd::GGSWBlindRotationReference<T, $be>>::ggsw_blind_rotation_from_scalar_reference(
                        module,
                        sign,
                        bit_rsh,
                        bit_mask,
                        bit_lsh,
                        res,
                        test_vector,
                        fhe_uint,
                        scratch,
                    )
                }
            }
        };
    };
}

/// Selects the reference implementation of [`GLWEBlindRotationImpl`].
#[macro_export]
macro_rules! impl_glwe_blind_rotation_reference_full {
    ($be:ty) => {
        const _: () = {
            use ::poulpy_core::{
                layouts::{prepared::*, *},
                *,
            };
            use ::poulpy_hal::{api::*, layouts::*, source::Source};
            #[allow(unused_imports)]
            use ::std::{collections::HashMap, marker::PhantomData};
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
                    <::poulpy_hal::layouts::Module<$be> as $crate::reference::bdd::GLWEBlindRotationReference<$be>>::glwe_blind_rotation_tmp_bytes_reference(
                        module, res_infos, a_infos, k_infos,
                    )
                }
                #[allow(clippy::too_many_arguments)]
                fn glwe_blind_rotation_assign<R, K>(
                    module: &Module<$be>,
                    sign: bool,
                    bit_rsh: usize,
                    bit_mask: usize,
                    bit_lsh: usize,
                    res: &mut R,
                    value: &K,
                    scratch: &mut ScratchArena<'_, $be>,
                ) where
                    R: GLWEToBackendMut<$be> + GLWEInfos,
                    K: GetGGSWBit<$be>,
                    $be: Backend<ZnxWord = i64>,
                {
                    <::poulpy_hal::layouts::Module<$be> as $crate::reference::bdd::GLWEBlindRotationReference<$be>>::glwe_blind_rotation_assign_reference(
                        module, sign, bit_rsh, bit_mask, bit_lsh, res, value, scratch,
                    )
                }
                #[allow(clippy::too_many_arguments)]
                fn glwe_blind_rotation<R, A, K>(
                    module: &Module<$be>,
                    sign: bool,
                    bit_rsh: usize,
                    bit_mask: usize,
                    bit_lsh: usize,
                    res: &mut R,
                    a: &A,
                    fhe_uint: &K,
                    scratch: &mut ScratchArena<'_, $be>,
                ) where
                    R: GLWEToBackendMut<$be> + GLWEInfos,
                    A: GLWEToBackendRef<$be>,
                    K: GetGGSWBit<$be>,
                    $be: Backend<ZnxWord = i64>,
                {
                    <::poulpy_hal::layouts::Module<$be> as $crate::reference::bdd::GLWEBlindRotationReference<$be>>::glwe_blind_rotation_reference(
                        module, sign, bit_rsh, bit_mask, bit_lsh, res, a, fhe_uint, scratch,
                    )
                }
            }
        };
    };
}

/// Selects the reference implementation of [`GLWEBlindSelectionImpl`].
#[macro_export]
macro_rules! impl_glwe_blind_selection_reference_full {
    ($be:ty) => {
        const _: () = {
            use ::poulpy_core::{
                layouts::{prepared::*, *},
                *,
            };
            use ::poulpy_hal::{api::*, layouts::*, source::Source};
            #[allow(unused_imports)]
            use ::std::{collections::HashMap, marker::PhantomData};
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
                    <::poulpy_hal::layouts::Module<$be> as $crate::reference::bdd::GLWEBlindSelectionReference<T, $be>>::glwe_blind_selection_tmp_bytes_reference(
                        module,
                        res_infos,
                        input_infos,
                        k_infos,
                    )
                }
                #[allow(clippy::too_many_arguments)]
                fn glwe_blind_selection<R, A, K>(
                    module: &Module<$be>,
                    bit_rsh: usize,
                    bit_mask: usize,
                    res: &mut R,
                    a: HashMap<usize, &mut A>,
                    fhe_uint: &K,
                    scratch: &mut ScratchArena<'_, $be>,
                ) where
                    R: GLWEToBackendMut<$be> + GLWEInfos,
                    A: GLWEToBackendMut<$be> + GLWEToBackendRef<$be> + GLWEInfos,
                    K: GetGGSWBit<$be>,
                {
                    <::poulpy_hal::layouts::Module<$be> as $crate::reference::bdd::GLWEBlindSelectionReference<T, $be>>::glwe_blind_selection_reference(
                        module, bit_rsh, bit_mask, res, a, fhe_uint, scratch,
                    )
                }
            }
        };
    };
}

/// Selects the same-layer derived defaults for [`GLWEBlindRetrievalImpl`].
#[macro_export]
macro_rules! impl_glwe_blind_retrieval_derived_full {
    ($be:ty) => {
        unsafe impl $crate::oep::GLWEBlindRetrievalImpl for $be {}
    };
}

/// Selects the reference implementation of [`BDDCircuitExecute1WTo1WImpl`].
#[macro_export]
macro_rules! impl_bdd_circuit_execute_1w_to_1w_reference_full {
    ($be:ty) => {
        const _: () = {
            use ::poulpy_core::{
                layouts::{prepared::*, *},
                *,
            };
            use ::poulpy_hal::{api::*, layouts::*, source::Source};
            #[allow(unused_imports)]
            use ::std::{collections::HashMap, marker::PhantomData};
            use $crate::{
                bdd_arithmetic::*,
                blind_rotation::{BlindRotationAlgo, BlindRotationKeyInfos},
                circuit_bootstrapping::*,
            };
            unsafe impl $crate::oep::BDDCircuitExecute1WTo1WImpl for $be {
                fn bdd_circuit_execute_1w_to_1w_multi_thread_tmp_bytes<C, T, R, G, H>(
                    module: &Module<$be>,
                    threads: usize,
                    res_infos: &R,
                    ggsw_infos: &G,
                    circuit: &C,
                    key: &H,
                ) -> usize
                where
                    C: GetBitCircuitInfo,
                    T: UnsignedInteger,
                    R: GLWEInfos,
                    G: GGSWInfos,
                    H: GetAutomorphismKey<$be>,
                {
                    <::poulpy_hal::layouts::Module<$be> as $crate::reference::bdd::BDDCircuitExecute1WTo1WReference<$be>>::bdd_circuit_execute_1w_to_1w_multi_thread_tmp_bytes_reference::<C, T, R, G, H>(
                        module, threads, res_infos, ggsw_infos, circuit, key,
                    )
                }
                #[allow(clippy::too_many_arguments)]
                fn bdd_circuit_execute_1w_to_1w_multi_thread<C, H, T>(
                    module: &Module<$be>,
                    threads: usize,
                    out: &mut FheUint<Self::OwnedBuf, T, Self::ZnxWord>,
                    a: &FheUintPrepared<Self::OwnedBuf, T, $be>,
                    circuit: &C,
                    key: &H,
                    scratch: &mut ScratchArena<'_, $be>,
                ) where
                    T: UnsignedInteger,
                    C: GetBitCircuitInfo,
                    H: GetAutomorphismKey<$be>,
                    $be: Backend<ZnxWord = i64>,
                {
                    <::poulpy_hal::layouts::Module<$be> as $crate::reference::bdd::BDDCircuitExecute1WTo1WReference<$be>>::bdd_circuit_execute_1w_to_1w_multi_thread_reference(
                        module, threads, out, a, circuit, key, scratch,
                    )
                }
            }
        };
    };
}

/// Selects the reference implementation of [`BDDCircuitExecute2WTo1WImpl`].
#[macro_export]
macro_rules! impl_bdd_circuit_execute_2w_to_1w_reference_full {
    ($be:ty) => {
        const _: () = {
            use ::poulpy_core::{
                layouts::{prepared::*, *},
                *,
            };
            use ::poulpy_hal::{api::*, layouts::*, source::Source};
            #[allow(unused_imports)]
            use ::std::{collections::HashMap, marker::PhantomData};
            use $crate::{
                bdd_arithmetic::*,
                blind_rotation::{BlindRotationAlgo, BlindRotationKeyInfos},
                circuit_bootstrapping::*,
            };
            unsafe impl $crate::oep::BDDCircuitExecute2WTo1WImpl for $be {
                #[allow(clippy::too_many_arguments)]
                fn bdd_circuit_execute_2w_to_1w_multi_thread_tmp_bytes<C, T, R, G, H>(
                    module: &Module<$be>,
                    threads: usize,
                    res_infos: &R,
                    ggsw_infos: &G,
                    circuit: &C,
                    key: &H,
                ) -> usize
                where
                    C: GetBitCircuitInfo,
                    T: UnsignedInteger,
                    R: GLWEInfos,
                    G: GGSWInfos,
                    H: GetAutomorphismKey<$be>,
                {
                    <::poulpy_hal::layouts::Module<$be> as $crate::reference::bdd::BDDCircuitExecute2WTo1WReference<$be>>::bdd_circuit_execute_2w_to_1w_multi_thread_tmp_bytes_reference::<_, T, _, _, _>(
                        module, threads, res_infos, ggsw_infos, circuit, key,
                    )
                }
                #[allow(clippy::too_many_arguments)]
                fn bdd_circuit_execute_2w_to_1w_multi_thread<C, H, T>(
                    module: &Module<$be>,
                    threads: usize,
                    out: &mut FheUint<Self::OwnedBuf, T, Self::ZnxWord>,
                    a: &FheUintPrepared<Self::OwnedBuf, T, $be>,
                    b: &FheUintPrepared<Self::OwnedBuf, T, $be>,
                    circuit: &C,
                    key: &H,
                    scratch: &mut ScratchArena<'_, $be>,
                ) where
                    T: UnsignedInteger,
                    C: GetBitCircuitInfo,
                    H: GetAutomorphismKey<$be>,
                    $be: Backend<ZnxWord = i64>,
                {
                    <::poulpy_hal::layouts::Module<$be> as $crate::reference::bdd::BDDCircuitExecute2WTo1WReference<$be>>::bdd_circuit_execute_2w_to_1w_multi_thread_reference(
                        module, threads, out, a, b, circuit, key, scratch,
                    )
                }
            }
        };
    };
}

/// Selects the reference implementation of [`FheUintPreparedEncryptSkImpl`].
#[macro_export]
macro_rules! impl_fhe_uint_prepared_encrypt_sk_reference_full {
    ($be:ty) => {
        const _: () = {
            use ::poulpy_core::{
                layouts::{prepared::*, *},
                *,
            };
            use ::poulpy_hal::{api::*, layouts::*, source::Source};
            #[allow(unused_imports)]
            use ::std::{collections::HashMap, marker::PhantomData};
            use $crate::{
                bdd_arithmetic::*,
                blind_rotation::{BlindRotationAlgo, BlindRotationKeyInfos},
                circuit_bootstrapping::*,
            };
            unsafe impl<T: UnsignedInteger + ToBits> $crate::oep::FheUintPreparedEncryptSkImpl<T> for $be {
                fn fhe_uint_prepared_encrypt_sk_tmp_bytes<A: GGSWInfos>(module: &Module<$be>, infos: &A) -> usize {
                    <::poulpy_hal::layouts::Module<$be> as $crate::reference::bdd::FheUintPreparedEncryptSkReference<T, $be>>::fhe_uint_prepared_encrypt_sk_tmp_bytes_reference(module, infos)
                }
                #[allow(clippy::too_many_arguments)]
                fn fhe_uint_prepared_encrypt_sk<S>(
                    module: &Module<$be>,
                    res: &mut FheUintPrepared<Self::OwnedBuf, T, $be>,
                    value: T,
                    sk: &S,
                    source_xe: &mut Source,
                    source_xa: &mut Source,
                    scratch: &mut ScratchArena<'_, $be>,
                ) where
                    S: GLWESecretPreparedToBackendRef<$be> + GLWEInfos,
                {
                    <::poulpy_hal::layouts::Module<$be> as $crate::reference::bdd::FheUintPreparedEncryptSkReference<T, $be>>::fhe_uint_prepared_encrypt_sk_reference(
                        module, res, value, sk, source_xe, source_xa, scratch,
                    )
                }
            }
        };
    };
}

/// Selects the reference implementation of [`FheUintPrepareImpl`].
#[macro_export]
macro_rules! impl_fhe_uint_prepare_reference_full {
    ($be:ty, $algo:ty) => {
        const _: () = {
            use ::poulpy_core::{
                layouts::{prepared::*, *},
                *,
            };
            use ::poulpy_hal::{api::*, layouts::*, source::Source};
            #[allow(unused_imports)]
            use ::std::{collections::HashMap, marker::PhantomData};
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
                    <::poulpy_hal::layouts::Module<$be> as $crate::reference::bdd::FheUintPrepareReference<$algo, $be>>::fhe_uint_prepare_tmp_bytes_reference(
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
                    <::poulpy_hal::layouts::Module<$be> as $crate::reference::bdd::FheUintPrepareReference<$algo, $be>>::fhe_uint_prepare_custom_multi_thread_reference(
                        module, threads, res, bits, bit_start, bit_count, key, scratch,
                    )
                }
            }
        };
    };
}

/// Selects the reference implementation of [`BDDKeyEncryptSkImpl`].
#[macro_export]
macro_rules! impl_bdd_key_encrypt_sk_reference_full {
    ($be:ty, $algo:ty) => {
        const _: () = {
            use ::poulpy_core::{
                layouts::{prepared::*, *},
                *,
            };
            use ::poulpy_hal::{api::*, layouts::*, source::Source};
            #[allow(unused_imports)]
            use ::std::{collections::HashMap, marker::PhantomData};
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
                    <::poulpy_hal::layouts::Module<$be> as $crate::reference::bdd::BDDKeyEncryptSkReference<$algo, $be>>::bdd_key_encrypt_sk_tmp_bytes_reference(module, infos)
                }
                #[allow(clippy::too_many_arguments)]
                fn bdd_key_encrypt_sk<S0, S1>(
                    module: &Module<$be>,
                    res: &mut BDDKey<Self::OwnedBuf, $algo, Self::ZnxWord>,
                    sk_lwe: &S0,
                    sk_glwe: &S1,
                    source_xe: &mut Source,
                    source_xa: &mut Source,
                    scratch: &mut ScratchArena<'_, $be>,
                ) where
                    S0: LWESecretToBackendRef<$be> + GetDistribution + LWEInfos,
                    S1: GLWESecretToBackendRef<$be> + GetDistribution + GLWEInfos,
                {
                    <::poulpy_hal::layouts::Module<$be> as $crate::reference::bdd::BDDKeyEncryptSkReference<$algo, $be>>::bdd_key_encrypt_sk_reference(
                        module, res, sk_lwe, sk_glwe, source_xe, source_xa, scratch,
                    )
                }
            }
        };
    };
}

/// Selects the reference implementation of [`BDDKeyPreparedFactoryImpl`].
#[macro_export]
macro_rules! impl_bdd_key_prepared_factory_reference_full {
    ($be:ty, $algo:ty) => {
        const _: () = {
            use ::poulpy_core::{
                layouts::{prepared::*, *},
                *,
            };
            use ::poulpy_hal::{api::*, layouts::*, source::Source};
            #[allow(unused_imports)]
            use ::std::{collections::HashMap, marker::PhantomData};
            use $crate::{
                bdd_arithmetic::*,
                blind_rotation::{BlindRotationAlgo, BlindRotationKeyInfos},
                circuit_bootstrapping::*,
            };
            unsafe impl $crate::oep::BDDKeyPreparedFactoryImpl<$algo> for $be {
                fn bdd_key_prepare_tmp_bytes<A>(module: &Module<$be>, infos: &A) -> usize
                where
                    A: BDDKeyInfos,
                {
                    <::poulpy_hal::layouts::Module<$be> as $crate::reference::bdd::BDDKeyPreparedFactoryReference<$algo, $be>>::bdd_key_prepare_tmp_bytes_reference(module, infos)
                }
                fn bdd_key_prepare(
                    module: &Module<$be>,
                    res: &mut BDDKeyPrepared<Self::OwnedBuf, $algo, $be>,
                    other: &BDDKey<Self::OwnedBuf, $algo, Self::ZnxWord>,
                    scratch: &mut ScratchArena<'_, $be>,
                ) {
                    <::poulpy_hal::layouts::Module<$be> as $crate::reference::bdd::BDDKeyPreparedFactoryReference<$algo, $be>>::bdd_key_prepare_reference(module, res, other, scratch)
                }
            }
        };
    };
}

/// Selects all canonical BDD implementations and same-layer derived defaults.
///
/// When overriding one operation family, select its peers using their individual
/// `impl_<family>_reference_full!` or `impl_<family>_derived_full!` macros instead.
#[macro_export]
macro_rules! impl_bdd_reference_full {
    ($be:ty, $algo:ty) => {
        $crate::impl_bdd_circuit_execute_reference_full!($be);
        $crate::impl_glwe_cswap_reference_full!($be);
        $crate::impl_glwe_cmux_reference_full!($be);
        $crate::impl_ggsw_blind_rotation_reference_full!($be);
        $crate::impl_glwe_blind_rotation_reference_full!($be);
        $crate::impl_glwe_blind_selection_reference_full!($be);
        $crate::impl_glwe_blind_retrieval_derived_full!($be);
        $crate::impl_bdd_circuit_execute_1w_to_1w_reference_full!($be);
        $crate::impl_bdd_circuit_execute_2w_to_1w_reference_full!($be);
        $crate::impl_fhe_uint_prepared_encrypt_sk_reference_full!($be);
        $crate::impl_fhe_uint_prepare_reference_full!($be, $algo);
        $crate::impl_bdd_key_encrypt_sk_reference_full!($be, $algo);
        $crate::impl_bdd_key_prepared_factory_reference_full!($be, $algo);
    };
}
