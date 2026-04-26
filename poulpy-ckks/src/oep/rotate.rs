#[macro_export]
macro_rules! impl_ckks_rotate_default_methods {
    ($backend:ty) => {
        fn ckks_rotate_tmp_bytes<C: poulpy_core::layouts::GLWEInfos, K: poulpy_core::layouts::GGLWEInfos>(
            module: &poulpy_hal::layouts::Module<$backend>,
            ct_infos: &C,
            key_infos: &K,
        ) -> usize
        where
            poulpy_hal::layouts::Module<$backend>: poulpy_core::GLWEAutomorphism<$backend>,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::default::rotate::CKKSRotateDefault<$backend>>::ckks_rotate_tmp_bytes_default(module, ct_infos, key_infos)
        }

        fn ckks_rotate_into<H, K>(
            module: &poulpy_hal::layouts::Module<$backend>,
            dst: &mut $crate::layouts::CKKSCiphertext<impl poulpy_hal::layouts::DataMut>,
            src: &$crate::layouts::CKKSCiphertext<impl poulpy_hal::layouts::DataRef>,
            k: i64,
            keys: &H,
            scratch: &mut poulpy_hal::layouts::Scratch<$backend>,
        ) -> anyhow::Result<()>
        where
            poulpy_hal::layouts::Module<$backend>: poulpy_core::GLWEAutomorphism<$backend> + poulpy_core::GLWEShift<$backend>,
            K: poulpy_core::layouts::GGLWEPreparedToRef<$backend> + poulpy_core::layouts::GetGaloisElement + poulpy_core::layouts::GGLWEInfos,
            H: poulpy_core::layouts::GLWEAutomorphismKeyHelper<K, $backend>,
            poulpy_hal::layouts::Scratch<$backend>: poulpy_core::ScratchTakeCore<$backend>,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::default::rotate::CKKSRotateDefault<$backend>>::ckks_rotate_into_default(module, dst, src, k, keys, scratch)
        }

        fn ckks_rotate_assign<H, K>(
            module: &poulpy_hal::layouts::Module<$backend>,
            dst: &mut $crate::layouts::CKKSCiphertext<impl poulpy_hal::layouts::DataMut>,
            k: i64,
            keys: &H,
            scratch: &mut poulpy_hal::layouts::Scratch<$backend>,
        ) -> anyhow::Result<()>
        where
            poulpy_hal::layouts::Module<$backend>: poulpy_core::GLWEAutomorphism<$backend>,
            K: poulpy_core::layouts::GGLWEPreparedToRef<$backend> + poulpy_core::layouts::GetGaloisElement + poulpy_core::layouts::GGLWEInfos,
            H: poulpy_core::layouts::GLWEAutomorphismKeyHelper<K, $backend>,
            poulpy_hal::layouts::Scratch<$backend>: poulpy_core::ScratchTakeCore<$backend>,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::default::rotate::CKKSRotateDefault<$backend>>::ckks_rotate_assign_default(module, dst, k, keys, scratch)
        }
    };
}

pub use crate::impl_ckks_rotate_default_methods;
