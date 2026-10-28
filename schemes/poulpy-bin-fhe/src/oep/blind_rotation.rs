//! Backend hooks for blind rotation and its key lifecycle.
#![allow(clippy::too_many_arguments)]
mod lookup_table_factory;
pub use lookup_table_factory::LookupTableFactoryImpl;

use crate::blind_rotation::{
    BlindRotationAlgo, BlindRotationKey, BlindRotationKeyCompressed, BlindRotationKeyInfos, BlindRotationKeyPrepared,
    LookUpTableRotationDirection, LookupTable,
};
use poulpy_core::{GetDistribution, layouts::*};
use poulpy_hal::{layouts::*, source::Source};

/// Backend implementation contract for [`BlindRotationExecute`](crate::api::BlindRotationExecute), on the standard ring
/// only.
///
/// # Safety
/// Implementations reproduce the bodies their methods document bit for bit, report for each
/// `_tmp_bytes` query a bound that covers the body, and refuse as the reference does. The memory,
/// layout, synchrony and aliasing rules of `poulpy-hal/docs/backend_safety_contract.md` apply.
pub unsafe trait BlindRotationExecuteImpl<BRA: BlindRotationAlgo>: Backend<Ring = Standard> {
    /// Scratch bound of [`Self::blind_rotation_execute`]; may report less than the reference, never less than the body
    /// takes. A block key reserves the products of a whole block, each with its own worker scratch, exactly when
    /// `TaskExecutor::IS_PARALLEL`.
    fn blind_rotation_execute_tmp_bytes<G, B>(
        module: &Module<Self>,
        block_size: usize,
        extension_factor: usize,
        glwe_infos: &G,
        brk_infos: &B,
    ) -> usize
    where
        G: GLWEInfos,
        B: BlindRotationKeyInfos;
    /// Reproduces
    /// [`blind_rotation_execute_reference`](crate::reference::blind_rotation::blind_rotation_execute_reference) bit for
    /// bit, within [`Self::blind_rotation_execute_tmp_bytes`]; free in kernels and scratch use, bit-identical for every
    /// executor and pool width.
    fn blind_rotation_execute<R, L>(
        module: &Module<Self>,
        res: &mut R,
        lwe: &L,
        lut: &LookupTable<Self::OwnedBuf, Self::ZnxWord>,
        brk: &BlindRotationKeyPrepared<Self::OwnedBuf, BRA, Self>,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: GLWEToBackendMut<Self> + GLWEInfos,
        L: LWEToBackendRef<Self> + LWEInfos;
}

/// Selects the reference implementation of `BlindRotationExecuteImpl<CGGI>` for the backend.
#[macro_export]
macro_rules! impl_blind_rotation_execute_reference_full {
    ($backend:ty) => {
        const _: () = {
            use poulpy_core::layouts::{GLWEInfos, GLWEToBackendMut, LWEInfos, LWEToBackendRef};
            use poulpy_hal::layouts::{Module, ScratchArena};
            use $crate::blind_rotation::{BlindRotationKeyInfos, BlindRotationKeyPrepared, CGGI, LookupTable};

            unsafe impl $crate::oep::BlindRotationExecuteImpl<$crate::blind_rotation::CGGI> for $backend {
                fn blind_rotation_execute_tmp_bytes<G, B>(
                    module: &Module<Self>,
                    block_size: usize,
                    extension_factor: usize,
                    glwe_infos: &G,
                    brk_infos: &B,
                ) -> usize
                where
                    G: GLWEInfos,
                    B: BlindRotationKeyInfos,
                {
                    $crate::reference::blind_rotation::blind_rotation_execute_tmp_bytes_reference::<Self, _, _>(
                        module,
                        block_size,
                        extension_factor,
                        glwe_infos,
                        brk_infos,
                    )
                }
                fn blind_rotation_execute<R, L>(
                    module: &Module<Self>,
                    res: &mut R,
                    lwe: &L,
                    lut: &LookupTable<Self::OwnedBuf, Self::ZnxWord>,
                    brk: &BlindRotationKeyPrepared<Self::OwnedBuf, CGGI, Self>,
                    scratch: &mut ScratchArena<'_, Self>,
                ) where
                    R: GLWEToBackendMut<Self> + GLWEInfos,
                    L: LWEToBackendRef<Self> + LWEInfos,
                {
                    $crate::reference::blind_rotation::blind_rotation_execute_reference::<Self, _, _>(
                        module, res, lwe, lut, brk, scratch,
                    )
                }
            }
        };
    };
}

/// Backend implementation contract for [`BlindRotationKeyEncryptSk`](crate::api::BlindRotationKeyEncryptSk).
///
/// # Safety
/// Implementations reproduce the bodies their methods document bit for bit, report for each
/// `_tmp_bytes` query a bound that covers the body, and refuse as the reference does.
/// Every method taking scratch wipes the scratch its query reports before returning.
/// The memory, layout, synchrony and aliasing rules of `poulpy-hal/docs/backend_safety_contract.md` apply.
pub unsafe trait BlindRotationKeyEncryptSkImpl<BRA: BlindRotationAlgo>: Backend {
    /// Scratch bound of [`Self::blind_rotation_key_encrypt_sk`]; may report less than the reference, never less than the
    /// body takes.
    fn blind_rotation_key_encrypt_sk_tmp_bytes<A>(module: &Module<Self>, infos: &A) -> usize
    where
        A: GGSWInfos;
    /// Reproduces
    /// [`blind_rotation_key_encrypt_sk_reference`](crate::reference::blind_rotation::blind_rotation_key_encrypt_sk_reference)
    /// bit for bit, within [`Self::blind_rotation_key_encrypt_sk_tmp_bytes`]; free in kernels and scratch use before the
    /// wipe.
    fn blind_rotation_key_encrypt_sk<S0, S1>(
        module: &Module<Self>,
        res: &mut BlindRotationKey<Self::OwnedBuf, BRA, Self::ZnxWord>,
        sk_lwe: &S0,
        sk_glwe: &S1,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        S0: LWESecretToBackendRef<Self> + LWEInfos + GetDistribution,
        S1: GLWESecretPreparedToBackendRef<Self> + GLWEInfos;
}

/// Selects the reference implementation of `BlindRotationKeyEncryptSkImpl<CGGI>` for the backend.
#[macro_export]
macro_rules! impl_blind_rotation_key_encrypt_sk_reference_full {
    ($backend:ty) => {
        const _: () = {
            use poulpy_core::{
                GetDistribution,
                layouts::{GGSWInfos, GLWEInfos, GLWESecretPreparedToBackendRef, LWEInfos, LWESecretToBackendRef},
            };
            use poulpy_hal::{
                layouts::{Module, ScratchArena},
                source::Source,
            };
            use $crate::blind_rotation::{BlindRotationKey, CGGI};

            unsafe impl $crate::oep::BlindRotationKeyEncryptSkImpl<$crate::blind_rotation::CGGI> for $backend {
                fn blind_rotation_key_encrypt_sk_tmp_bytes<A>(module: &Module<Self>, infos: &A) -> usize
                where
                    A: GGSWInfos,
                {
                    $crate::reference::blind_rotation::blind_rotation_key_encrypt_sk_tmp_bytes_reference::<Self, _>(module, infos)
                }
                fn blind_rotation_key_encrypt_sk<S0, S1>(
                    module: &Module<Self>,
                    res: &mut BlindRotationKey<Self::OwnedBuf, CGGI, Self::ZnxWord>,
                    sk_lwe: &S0,
                    sk_glwe: &S1,
                    source_xe: &mut Source,
                    source_xa: &mut Source,
                    scratch: &mut ScratchArena<'_, Self>,
                ) where
                    S0: LWESecretToBackendRef<Self> + LWEInfos + GetDistribution,
                    S1: GLWESecretPreparedToBackendRef<Self> + GLWEInfos,
                {
                    $crate::reference::blind_rotation::blind_rotation_key_encrypt_sk_reference::<Self, _, _>(
                        module, res, sk_lwe, sk_glwe, source_xe, source_xa, scratch,
                    )
                }
            }
        };
    };
}

/// Backend implementation contract for
/// [`BlindRotationKeyCompressedEncryptSk`](crate::api::BlindRotationKeyCompressedEncryptSk).
///
/// # Safety
/// Implementations reproduce the bodies their methods document bit for bit, report for each
/// `_tmp_bytes` query a bound that covers the body, and refuse as the reference does.
/// Every method taking scratch wipes the scratch its query reports before returning.
/// The memory, layout, synchrony and aliasing rules of `poulpy-hal/docs/backend_safety_contract.md` apply.
pub unsafe trait BlindRotationKeyCompressedEncryptSkImpl<BRA: BlindRotationAlgo>: Backend {
    /// Scratch bound of [`Self::blind_rotation_key_compressed_encrypt_sk`]; may report less than the reference, never
    /// less than the body takes.
    fn blind_rotation_key_compressed_encrypt_sk_tmp_bytes<A>(module: &Module<Self>, infos: &A) -> usize
    where
        A: GGSWInfos;
    /// Reproduces
    /// [`blind_rotation_key_compressed_encrypt_sk_reference`](crate::reference::blind_rotation::blind_rotation_key_compressed_encrypt_sk_reference)
    /// bit for bit, within [`Self::blind_rotation_key_compressed_encrypt_sk_tmp_bytes`]; free in kernels and scratch use
    /// before the wipe.
    fn blind_rotation_key_compressed_encrypt_sk<S0, S1>(
        module: &Module<Self>,
        res: &mut BlindRotationKeyCompressed<Self::OwnedBuf, BRA, Self::ZnxWord>,
        sk_lwe: &S0,
        sk_glwe: &S1,
        seed_xa: [u8; 32],
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        S0: LWESecretToBackendRef<Self> + LWEInfos + GetDistribution,
        S1: GLWESecretPreparedToBackendRef<Self> + GLWEInfos;
}

/// Selects the reference implementation of `BlindRotationKeyCompressedEncryptSkImpl<CGGI>` for the backend.
#[macro_export]
macro_rules! impl_blind_rotation_key_compressed_encrypt_sk_reference_full {
    ($backend:ty) => {
        const _: () = {
            use poulpy_core::{
                GetDistribution,
                layouts::{GGSWInfos, GLWEInfos, GLWESecretPreparedToBackendRef, LWEInfos, LWESecretToBackendRef},
            };
            use poulpy_hal::{
                layouts::{Module, ScratchArena},
                source::Source,
            };
            use $crate::blind_rotation::{BlindRotationKeyCompressed, CGGI};

            unsafe impl $crate::oep::BlindRotationKeyCompressedEncryptSkImpl<$crate::blind_rotation::CGGI> for $backend {
                fn blind_rotation_key_compressed_encrypt_sk_tmp_bytes<A>(module: &Module<Self>, infos: &A) -> usize
                where
                    A: GGSWInfos,
                {
                    $crate::reference::blind_rotation::blind_rotation_key_compressed_encrypt_sk_tmp_bytes_reference::<Self, _>(
                        module, infos,
                    )
                }
                fn blind_rotation_key_compressed_encrypt_sk<S0, S1>(
                    module: &Module<Self>,
                    res: &mut BlindRotationKeyCompressed<Self::OwnedBuf, CGGI, Self::ZnxWord>,
                    sk_lwe: &S0,
                    sk_glwe: &S1,
                    seed_xa: [u8; 32],
                    source_xe: &mut Source,
                    scratch: &mut ScratchArena<'_, Self>,
                ) where
                    S0: LWESecretToBackendRef<Self> + LWEInfos + GetDistribution,
                    S1: GLWESecretPreparedToBackendRef<Self> + GLWEInfos,
                {
                    $crate::reference::blind_rotation::blind_rotation_key_compressed_encrypt_sk_reference::<Self, _, _>(
                        module, res, sk_lwe, sk_glwe, seed_xa, source_xe, scratch,
                    )
                }
            }
        };
    };
}

/// Backend implementation contract for [`BlindRotationKeyPreparedFactory`](crate::api::BlindRotationKeyPreparedFactory).
///
/// # Safety
/// Implementations reproduce the bodies their methods document bit for bit, report for each
/// `_tmp_bytes` query a bound that covers the body, and refuse as the reference does. The memory,
/// layout, synchrony and aliasing rules of `poulpy-hal/docs/backend_safety_contract.md` apply.
pub unsafe trait BlindRotationKeyPreparedImpl<BRA: BlindRotationAlgo>: Backend {
    /// Scratch bound of [`Self::blind_rotation_key_prepare`]; may report less than the reference, never less than the
    /// body takes.
    fn blind_rotation_key_prepare_tmp_bytes<A>(module: &Module<Self>, infos: &A) -> usize
    where
        A: BlindRotationKeyInfos;
    /// Reproduces
    /// [`blind_rotation_key_prepare_reference`](crate::reference::blind_rotation::blind_rotation_key_prepare_reference)
    /// bit for bit, within [`Self::blind_rotation_key_prepare_tmp_bytes`]; free in kernels and scratch use.
    fn blind_rotation_key_prepare(
        module: &Module<Self>,
        res: &mut BlindRotationKeyPrepared<Self::OwnedBuf, BRA, Self>,
        other: &BlindRotationKey<Self::OwnedBuf, BRA, Self::ZnxWord>,
        scratch: &mut ScratchArena<'_, Self>,
    );
}

/// Selects the reference implementation of `BlindRotationKeyPreparedImpl<CGGI>` for the backend.
#[macro_export]
macro_rules! impl_blind_rotation_key_prepared_reference_full {
    ($backend:ty) => {
        const _: () = {
            use poulpy_hal::layouts::{Module, ScratchArena};
            use $crate::blind_rotation::{BlindRotationKey, BlindRotationKeyInfos, BlindRotationKeyPrepared, CGGI};

            unsafe impl $crate::oep::BlindRotationKeyPreparedImpl<$crate::blind_rotation::CGGI> for $backend {
                fn blind_rotation_key_prepare_tmp_bytes<A>(module: &Module<Self>, infos: &A) -> usize
                where
                    A: BlindRotationKeyInfos,
                {
                    $crate::reference::blind_rotation::blind_rotation_key_prepare_tmp_bytes_reference::<Self, _>(module, infos)
                }
                fn blind_rotation_key_prepare(
                    module: &Module<Self>,
                    res: &mut BlindRotationKeyPrepared<Self::OwnedBuf, CGGI, Self>,
                    other: &BlindRotationKey<Self::OwnedBuf, CGGI, Self::ZnxWord>,
                    scratch: &mut ScratchArena<'_, Self>,
                ) {
                    $crate::reference::blind_rotation::blind_rotation_key_prepare_reference::<Self>(module, res, other, scratch)
                }
            }
        };
    };
}

/// Backend implementation contract for [`BlindRotationModSwitch`](crate::api::BlindRotationModSwitch), the host staging
/// boundary of the exponents.
///
/// # Safety
/// Implementations reproduce the bodies their methods document bit for bit and refuse as the reference does.
/// The memory, layout, synchrony and aliasing rules of `poulpy-hal/docs/backend_safety_contract.md` apply.
pub unsafe trait BlindRotationModSwitchImpl: Backend<ZnxWord = i64> {
    /// Reproduces
    /// [`blind_rotation_mod_switch_reference`](crate::reference::blind_rotation::blind_rotation_mod_switch_reference) bit
    /// for bit; free in kernels. A host step, it
    /// reads the coefficients of `lwe` and writes the body then the mask into `res`.
    fn blind_rotation_mod_switch<L: LWEToBackendRef<Self> + LWEInfos>(
        module: &Module<Self>,
        modulus: usize,
        direction: LookUpTableRotationDirection,
        res: &mut [i64],
        lwe: &L,
    );
}
/// Backend implementation contract for [`BlindRotationKeyDecompress`](crate::api::BlindRotationKeyDecompress).
///
/// # Safety
/// Implementations reproduce the bodies their methods document bit for bit, report for each
/// `_tmp_bytes` query a bound that covers the body, and refuse as the reference does. The memory,
/// layout, synchrony and aliasing rules of `poulpy-hal/docs/backend_safety_contract.md` apply.
pub unsafe trait BlindRotationKeyDecompressImpl<BRA: BlindRotationAlgo>: Backend {
    /// Scratch bound of [`Self::blind_rotation_key_decompress`]; may report less than the reference, never less than the
    /// body takes.
    fn blind_rotation_key_decompress_tmp_bytes<A: BlindRotationKeyInfos>(module: &Module<Self>, infos: &A) -> usize;
    /// Reproduces
    /// [`blind_rotation_key_decompress_reference`](crate::reference::blind_rotation::blind_rotation_key_decompress_reference)
    /// bit for bit, within [`Self::blind_rotation_key_decompress_tmp_bytes`]; free in kernels and scratch use.
    fn blind_rotation_key_decompress(
        module: &Module<Self>,
        res: &mut BlindRotationKey<Self::OwnedBuf, BRA, Self::ZnxWord>,
        src: &BlindRotationKeyCompressed<Self::OwnedBuf, BRA, Self::ZnxWord>,
        scratch: &mut ScratchArena<'_, Self>,
    );
}
/// Selects the canonical host staging boundary.
#[macro_export]
macro_rules! impl_blind_rotation_mod_switch_reference_full {
    ($backend:ty) => {
        unsafe impl $crate::oep::BlindRotationModSwitchImpl for $backend {
            fn blind_rotation_mod_switch<L: poulpy_core::layouts::LWEToBackendRef<Self> + poulpy_core::layouts::LWEInfos>(
                module: &poulpy_hal::layouts::Module<Self>,
                modulus: usize,
                direction: $crate::blind_rotation::LookUpTableRotationDirection,
                res: &mut [i64],
                lwe: &L,
            ) {
                $crate::reference::blind_rotation::blind_rotation_mod_switch_reference::<Self, _>(
                    module, modulus, direction, res, lwe,
                )
            }
        }
    };
}
/// Selects coefficient-domain decompression through core operations.
#[macro_export]
macro_rules! impl_blind_rotation_key_decompress_reference_full {
    ($backend:ty) => {
        unsafe impl $crate::oep::BlindRotationKeyDecompressImpl<$crate::blind_rotation::CGGI> for $backend {
            fn blind_rotation_key_decompress_tmp_bytes<A: $crate::blind_rotation::BlindRotationKeyInfos>(
                module: &poulpy_hal::layouts::Module<Self>,
                infos: &A,
            ) -> usize {
                $crate::reference::blind_rotation::blind_rotation_key_decompress_tmp_bytes_reference::<Self, _>(module, infos)
            }
            fn blind_rotation_key_decompress(
                module: &poulpy_hal::layouts::Module<Self>,
                res: &mut $crate::blind_rotation::BlindRotationKey<Self::OwnedBuf, $crate::blind_rotation::CGGI, Self::ZnxWord>,
                src: &$crate::blind_rotation::BlindRotationKeyCompressed<
                    Self::OwnedBuf,
                    $crate::blind_rotation::CGGI,
                    Self::ZnxWord,
                >,
                scratch: &mut poulpy_hal::layouts::ScratchArena<'_, Self>,
            ) {
                $crate::reference::blind_rotation::blind_rotation_key_decompress_reference::<Self>(module, res, src, scratch)
            }
        }
    };
}
/// Opts a backend into all canonical CGGI operations.
#[macro_export]
macro_rules! impl_blind_rotation_reference_full {
    ($backend:ty) => {
        $crate::impl_blind_rotation_execute_reference_full!($backend);
        $crate::impl_blind_rotation_key_encrypt_sk_reference_full!($backend);
        $crate::impl_blind_rotation_key_compressed_encrypt_sk_reference_full!($backend);
        $crate::impl_blind_rotation_key_prepared_reference_full!($backend);
        $crate::impl_blind_rotation_mod_switch_reference_full!($backend);
        $crate::impl_blind_rotation_key_decompress_reference_full!($backend);
        $crate::impl_lookup_table_factory_reference_full!($backend);
    };
}
