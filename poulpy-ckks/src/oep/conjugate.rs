#[macro_export]
macro_rules! impl_ckks_conjugate_default_methods {
    ($backend:ty) => {
        fn ckks_conjugate_tmp_bytes<C: poulpy_core::layouts::GLWEInfos, K: poulpy_core::layouts::GGLWEInfos>(
            module: &poulpy_hal::layouts::Module<$backend>,
            ct_infos: &C,
            key_infos: &K,
        ) -> usize
        where
            poulpy_hal::layouts::Module<$backend>: poulpy_core::GLWEAutomorphism<$backend>,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::default::conjugate::CKKSConjugateDefault<$backend>>::ckks_conjugate_tmp_bytes_default(module, ct_infos, key_infos)
        }

        fn ckks_conjugate_into(
            module: &poulpy_hal::layouts::Module<$backend>,
            dst: &mut $crate::layouts::CKKSCiphertext<impl poulpy_hal::layouts::DataMut>,
            src: &$crate::layouts::CKKSCiphertext<impl poulpy_hal::layouts::DataRef>,
            key: &poulpy_core::layouts::GLWEAutomorphismKeyPrepared<impl poulpy_hal::layouts::DataRef, $backend>,
            scratch: &mut poulpy_hal::layouts::Scratch<$backend>,
        ) -> anyhow::Result<()>
        where
            poulpy_hal::layouts::Module<$backend>: poulpy_core::GLWEAutomorphism<$backend> + poulpy_core::GLWEShift<$backend>,
            poulpy_hal::layouts::Scratch<$backend>: poulpy_core::ScratchTakeCore<$backend>,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::default::conjugate::CKKSConjugateDefault<$backend>>::ckks_conjugate_into_default(module, dst, src, key, scratch)
        }

        fn ckks_conjugate_assign(
            module: &poulpy_hal::layouts::Module<$backend>,
            dst: &mut $crate::layouts::CKKSCiphertext<impl poulpy_hal::layouts::DataMut>,
            key: &poulpy_core::layouts::GLWEAutomorphismKeyPrepared<impl poulpy_hal::layouts::DataRef, $backend>,
            scratch: &mut poulpy_hal::layouts::Scratch<$backend>,
        ) -> anyhow::Result<()>
        where
            poulpy_hal::layouts::Module<$backend>: poulpy_core::GLWEAutomorphism<$backend>,
            poulpy_hal::layouts::Scratch<$backend>: poulpy_core::ScratchTakeCore<$backend>,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::default::conjugate::CKKSConjugateDefault<$backend>>::ckks_conjugate_assign_default(module, dst, key, scratch)
        }
    };
}

pub use crate::impl_ckks_conjugate_default_methods;
