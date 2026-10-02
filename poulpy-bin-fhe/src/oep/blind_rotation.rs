//! Backend hooks for blind rotation and its key lifecycle.
#![allow(clippy::too_many_arguments)]
mod blind_rotation_key_compressed_factory;
mod lookup_table_factory;
pub use blind_rotation_key_compressed_factory::BlindRotationKeyCompressedFactoryImpl;
pub use lookup_table_factory::LookupTableFactoryImpl;

use crate::blind_rotation::{
    BlindRotationAlgo, BlindRotationKey, BlindRotationKeyCompressed, BlindRotationKeyInfos, BlindRotationKeyPrepared,
    LookUpTableRotationDirection, LookupTable,
};
use poulpy_core::{EncryptionInfos, GetDistribution, layouts::*};
use poulpy_hal::{layouts::*, source::Source};

/// Backend implementation of the blind-rotation operation family.
///
/// # Safety
/// Preserve the reference circuit, metadata, and advertised scratch bounds.
pub unsafe trait BlindRotationExecuteImpl<BRA: BlindRotationAlgo>: Backend {
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

/// Explicitly selects the canonical implementation for this operation family.
#[macro_export]
macro_rules! impl_bin_fhe_blind_rotation_execute_reference {
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
                    $crate::reference::blind_rotation::blind_rotation_execute_tmp_bytes_ref::<Self, _, _>(
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
                    $crate::reference::blind_rotation::blind_rotation_execute_ref::<Self, _, _>(
                        module, res, lwe, lut, brk, scratch,
                    )
                }
            }
        };
    };
}

/// Backend implementation of the blind-rotation operation family.
///
/// # Safety
/// Preserve the reference circuit, metadata, and advertised scratch bounds.
pub unsafe trait BlindRotationKeyEncryptSkImpl<BRA: BlindRotationAlgo>: Backend {
    fn blind_rotation_key_encrypt_sk_tmp_bytes<A: GGSWInfos>(module: &Module<Self>, infos: &A) -> usize;
    fn blind_rotation_key_encrypt_sk<S0, S1, E>(
        module: &Module<Self>,
        res: &mut BlindRotationKey<Self::OwnedBuf, BRA, Self::ZnxWord>,
        sk_glwe: &S0,
        sk_lwe: &S1,
        enc_infos: &E,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        S0: GLWESecretPreparedToBackendRef<Self> + GLWEInfos,
        E: EncryptionInfos,
        S1: LWESecretToBackendRef<Self> + LWEInfos + GetDistribution;
}

/// Explicitly selects the canonical implementation for this operation family.
#[macro_export]
macro_rules! impl_bin_fhe_blind_rotation_key_encrypt_reference {
    ($backend:ty) => {
        const _: () = {
            use poulpy_core::{
                EncryptionInfos, GetDistribution,
                layouts::{GGSWInfos, GLWEInfos, GLWESecretPreparedToBackendRef, LWEInfos, LWESecretToBackendRef},
            };
            use poulpy_hal::{
                layouts::{Module, ScratchArena},
                source::Source,
            };
            use $crate::blind_rotation::{BlindRotationKey, CGGI};

            unsafe impl $crate::oep::BlindRotationKeyEncryptSkImpl<$crate::blind_rotation::CGGI> for $backend {
                fn blind_rotation_key_encrypt_sk_tmp_bytes<A: GGSWInfos>(module: &Module<Self>, infos: &A) -> usize {
                    $crate::reference::blind_rotation::blind_rotation_key_encrypt_sk_tmp_bytes_ref::<Self, _>(module, infos)
                }
                fn blind_rotation_key_encrypt_sk<S0, S1, E>(
                    module: &Module<Self>,
                    res: &mut BlindRotationKey<Self::OwnedBuf, CGGI, Self::ZnxWord>,
                    sk_glwe: &S0,
                    sk_lwe: &S1,
                    enc_infos: &E,
                    source_xe: &mut Source,
                    source_xa: &mut Source,
                    scratch: &mut ScratchArena<'_, Self>,
                ) where
                    S0: GLWESecretPreparedToBackendRef<Self> + GLWEInfos,
                    E: EncryptionInfos,
                    S1: LWESecretToBackendRef<Self> + LWEInfos + GetDistribution,
                {
                    $crate::reference::blind_rotation::blind_rotation_key_encrypt_sk_ref::<Self, _, _, _>(
                        module, res, sk_glwe, sk_lwe, enc_infos, source_xe, source_xa, scratch,
                    )
                }
            }
        };
    };
}

/// Backend implementation of the blind-rotation operation family.
///
/// # Safety
/// Preserve the reference circuit, metadata, and advertised scratch bounds.
pub unsafe trait BlindRotationKeyCompressedEncryptSkImpl<BRA: BlindRotationAlgo>: Backend {
    fn blind_rotation_key_compressed_encrypt_sk_tmp_bytes<A>(module: &Module<Self>, infos: &A) -> usize
    where
        A: GGSWInfos;
    fn blind_rotation_key_compressed_encrypt_sk<S0, S1, E>(
        module: &Module<Self>,
        res: &mut BlindRotationKeyCompressed<Self::OwnedBuf, BRA, Self::ZnxWord>,
        sk_glwe: &S0,
        sk_lwe: &S1,
        seed_xa: [u8; 32],
        enc_infos: &E,
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        S0: GLWESecretPreparedToBackendRef<Self> + GLWEInfos,
        E: EncryptionInfos,
        S1: LWESecretToBackendRef<Self> + LWEInfos + GetDistribution;
}

/// Explicitly selects the canonical implementation for this operation family.
#[macro_export]
macro_rules! impl_bin_fhe_blind_rotation_key_compressed_reference {
    ($backend:ty) => {
        const _: () = {
            use poulpy_core::{
                EncryptionInfos, GetDistribution,
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
                    $crate::reference::blind_rotation::blind_rotation_key_compressed_encrypt_sk_tmp_bytes_ref::<Self, _>(
                        module, infos,
                    )
                }
                fn blind_rotation_key_compressed_encrypt_sk<S0, S1, E>(
                    module: &Module<Self>,
                    res: &mut BlindRotationKeyCompressed<Self::OwnedBuf, CGGI, Self::ZnxWord>,
                    sk_glwe: &S0,
                    sk_lwe: &S1,
                    seed_xa: [u8; 32],
                    enc_infos: &E,
                    source_xe: &mut Source,
                    scratch: &mut ScratchArena<'_, Self>,
                ) where
                    S0: GLWESecretPreparedToBackendRef<Self> + GLWEInfos,
                    E: EncryptionInfos,
                    S1: LWESecretToBackendRef<Self> + LWEInfos + GetDistribution,
                {
                    $crate::reference::blind_rotation::blind_rotation_key_compressed_encrypt_sk_ref::<Self, _, _, _>(
                        module, res, sk_glwe, sk_lwe, seed_xa, enc_infos, source_xe, scratch,
                    )
                }
            }
        };
    };
}

/// Backend implementation of the blind-rotation operation family.
///
/// # Safety
/// Preserve the reference circuit, metadata, and advertised scratch bounds.
pub unsafe trait BlindRotationKeyPreparedImpl<BRA: BlindRotationAlgo>: Backend {
    fn blind_rotation_key_prepared_alloc<A>(
        module: &Module<Self>,
        infos: &A,
    ) -> BlindRotationKeyPrepared<Self::OwnedBuf, BRA, Self>
    where
        A: BlindRotationKeyInfos;
    fn blind_rotation_key_prepare_tmp_bytes<A>(module: &Module<Self>, infos: &A) -> usize
    where
        A: BlindRotationKeyInfos;
    fn prepare_blind_rotation_key(
        module: &Module<Self>,
        res: &mut BlindRotationKeyPrepared<Self::OwnedBuf, BRA, Self>,
        other: &BlindRotationKey<Self::OwnedBuf, BRA, Self::ZnxWord>,
        scratch: &mut ScratchArena<'_, Self>,
    );
}

/// Explicitly selects the canonical implementation for this operation family.
#[macro_export]
macro_rules! impl_bin_fhe_blind_rotation_key_prepared_reference {
    ($backend:ty) => {
        const _: () = {
            use poulpy_hal::layouts::{Module, ScratchArena};
            use $crate::blind_rotation::{BlindRotationKey, BlindRotationKeyInfos, BlindRotationKeyPrepared, CGGI};

            unsafe impl $crate::oep::BlindRotationKeyPreparedImpl<$crate::blind_rotation::CGGI> for $backend {
                fn blind_rotation_key_prepared_alloc<A>(
                    module: &Module<Self>,
                    infos: &A,
                ) -> BlindRotationKeyPrepared<Self::OwnedBuf, CGGI, Self>
                where
                    A: BlindRotationKeyInfos,
                {
                    $crate::reference::blind_rotation::blind_rotation_key_prepared_alloc_ref::<Self, _>(module, infos)
                }
                fn blind_rotation_key_prepare_tmp_bytes<A>(module: &Module<Self>, infos: &A) -> usize
                where
                    A: BlindRotationKeyInfos,
                {
                    $crate::reference::blind_rotation::blind_rotation_key_prepare_tmp_bytes_ref::<Self, _>(module, infos)
                }
                fn prepare_blind_rotation_key(
                    module: &Module<Self>,
                    res: &mut BlindRotationKeyPrepared<Self::OwnedBuf, CGGI, Self>,
                    other: &BlindRotationKey<Self::OwnedBuf, CGGI, Self::ZnxWord>,
                    scratch: &mut ScratchArena<'_, Self>,
                ) {
                    $crate::reference::blind_rotation::prepare_blind_rotation_key_ref::<Self>(module, res, other, scratch)
                }
            }
        };
    };
}

/// Backend staging of blind-rotation indices into host-visible exponent values.
///
/// # Safety
/// Match the canonical signed-limb rounding and rotation-direction convention.
pub unsafe trait BlindRotationModSwitchImpl: Backend<ZnxWord = i64> {
    fn blind_rotation_mod_switch<L: LWEToBackendRef<Self> + LWEInfos>(
        module: &Module<Self>,
        modulus: usize,
        res: &mut [i64],
        lwe: &L,
        direction: LookUpTableRotationDirection,
    );
}
/// Backend coefficient-domain key decompression.
///
/// # Safety
/// Reproduce the source key metadata and mask sampling circuit within the budget.
pub unsafe trait BlindRotationKeyDecompressImpl<BRA: BlindRotationAlgo>: Backend {
    fn blind_rotation_key_decompress_tmp_bytes<A: BlindRotationKeyInfos>(module: &Module<Self>, infos: &A) -> usize;
    fn blind_rotation_key_decompress(
        module: &Module<Self>,
        res: &mut BlindRotationKey<Self::OwnedBuf, BRA, Self::ZnxWord>,
        src: &BlindRotationKeyCompressed<Self::OwnedBuf, BRA, Self::ZnxWord>,
        scratch: &mut ScratchArena<'_, Self>,
    );
}
/// Selects the canonical host staging boundary.
#[macro_export]
macro_rules! impl_bin_fhe_blind_rotation_mod_switch_reference {
    ($backend:ty) => {
        unsafe impl $crate::oep::BlindRotationModSwitchImpl for $backend {
            fn blind_rotation_mod_switch<L: poulpy_core::layouts::LWEToBackendRef<Self> + poulpy_core::layouts::LWEInfos>(
                _module: &poulpy_hal::layouts::Module<Self>,
                modulus: usize,
                res: &mut [i64],
                lwe: &L,
                direction: $crate::blind_rotation::LookUpTableRotationDirection,
            ) {
                $crate::reference::blind_rotation::mod_switch_2n_ref::<Self, _>(modulus, res, lwe, direction)
            }
        }
    };
}
/// Selects coefficient-domain decompression through core operations.
#[macro_export]
macro_rules! impl_bin_fhe_blind_rotation_key_decompress_reference {
    ($backend:ty) => {
        unsafe impl $crate::oep::BlindRotationKeyDecompressImpl<$crate::blind_rotation::CGGI> for $backend {
            fn blind_rotation_key_decompress_tmp_bytes<A: $crate::blind_rotation::BlindRotationKeyInfos>(
                module: &poulpy_hal::layouts::Module<Self>,
                infos: &A,
            ) -> usize {
                $crate::reference::blind_rotation::blind_rotation_key_decompress_tmp_bytes_ref::<Self, _>(module, infos)
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
                $crate::reference::blind_rotation::blind_rotation_key_decompress_ref::<Self>(module, res, src, scratch)
            }
        }
    };
}
/// Opts a backend into all canonical CGGI operations.
#[macro_export]
macro_rules! impl_bin_fhe_blind_rotation_reference {
    ($backend:ty) => {
        $crate::impl_bin_fhe_blind_rotation_execute_reference!($backend);
        $crate::impl_bin_fhe_blind_rotation_key_encrypt_reference!($backend);
        $crate::impl_bin_fhe_blind_rotation_key_compressed_reference!($backend);
        $crate::impl_bin_fhe_blind_rotation_key_prepared_reference!($backend);
        $crate::impl_bin_fhe_blind_rotation_mod_switch_reference!($backend);
        $crate::impl_bin_fhe_blind_rotation_key_decompress_reference!($backend);
        $crate::impl_bin_fhe_blind_rotation_key_compressed_factory_reference!($backend);
        $crate::impl_bin_fhe_lookup_table_reference!($backend);
    };
    ($backend:ty, scheduling = parallel) => {
        $crate::impl_bin_fhe_blind_rotation_execute_parallel!($backend);
        $crate::impl_bin_fhe_blind_rotation_key_encrypt_reference!($backend);
        $crate::impl_bin_fhe_blind_rotation_key_compressed_reference!($backend);
        $crate::impl_bin_fhe_blind_rotation_key_prepared_reference!($backend);
        $crate::impl_bin_fhe_blind_rotation_mod_switch_reference!($backend);
        $crate::impl_bin_fhe_blind_rotation_key_decompress_reference!($backend);
        $crate::impl_bin_fhe_blind_rotation_key_compressed_factory_reference!($backend);
        $crate::impl_bin_fhe_lookup_table_reference!($backend);
    };
}

/// Explicitly selects the explicit parallel implementation for this operation family.
#[macro_export]
macro_rules! impl_bin_fhe_blind_rotation_execute_parallel {
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
                    $crate::reference::blind_rotation::blind_rotation_execute_tmp_bytes_parallel::<Self, _, _>(
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
                    $crate::reference::blind_rotation::blind_rotation_execute_parallel::<Self, _, _>(
                        module, res, lwe, lut, brk, scratch,
                    )
                }
            }
        };
    };
}
