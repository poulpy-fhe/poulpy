#[macro_export]
macro_rules! impl_ckks_neg_default_methods {
    ($backend:ty) => {
        fn ckks_neg_tmp_bytes(module: &poulpy_hal::layouts::Module<$backend>) -> usize
        where
            poulpy_hal::layouts::Module<$backend>: poulpy_core::GLWEShift<$backend>,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::default::neg::CKKSNegDefault<$backend>>::ckks_neg_tmp_bytes_default(module)
        }

        fn ckks_neg_into(
            module: &poulpy_hal::layouts::Module<$backend>,
            dst: &mut $crate::layouts::CKKSCiphertext<impl poulpy_hal::layouts::DataMut>,
            src: &$crate::layouts::CKKSCiphertext<impl poulpy_hal::layouts::DataRef>,
            scratch: &mut poulpy_hal::layouts::Scratch<$backend>,
        ) -> anyhow::Result<()>
        where
            poulpy_hal::layouts::Module<$backend>: poulpy_core::GLWENegate + poulpy_core::GLWEShift<$backend>,
            poulpy_hal::layouts::Scratch<$backend>: poulpy_hal::api::ScratchAvailable + poulpy_core::ScratchTakeCore<$backend>,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::default::neg::CKKSNegDefault<$backend>>::ckks_neg_into_default(module, dst, src, scratch)
        }

        fn ckks_neg_assign(
            module: &poulpy_hal::layouts::Module<$backend>,
            dst: &mut $crate::layouts::CKKSCiphertext<impl poulpy_hal::layouts::DataMut>,
        ) -> anyhow::Result<()>
        where
            poulpy_hal::layouts::Module<$backend>: poulpy_core::GLWENegate,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::default::neg::CKKSNegDefault<$backend>>::ckks_neg_assign_default(module, dst)
        }
    };
}

pub use crate::impl_ckks_neg_default_methods;
