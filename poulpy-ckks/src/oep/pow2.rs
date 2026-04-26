#[macro_export]
macro_rules! impl_ckks_pow2_default_methods {
    ($backend:ty) => {
        fn ckks_mul_pow2_tmp_bytes(module: &poulpy_hal::layouts::Module<$backend>) -> usize
        where
            poulpy_hal::layouts::Module<$backend>: poulpy_core::GLWEShift<$backend>,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::default::pow2::CKKSPow2Default<$backend>>::ckks_mul_pow2_tmp_bytes_default(module)
        }

        fn ckks_mul_pow2_into(
            module: &poulpy_hal::layouts::Module<$backend>,
            dst: &mut $crate::layouts::CKKSCiphertext<impl poulpy_hal::layouts::DataMut>,
            src: &$crate::layouts::CKKSCiphertext<impl poulpy_hal::layouts::DataRef>,
            bits: usize,
            scratch: &mut poulpy_hal::layouts::Scratch<$backend>,
        ) -> anyhow::Result<()>
        where
            poulpy_hal::layouts::Module<$backend>: poulpy_core::GLWEShift<$backend>,
            poulpy_hal::layouts::Scratch<$backend>: poulpy_core::ScratchTakeCore<$backend>,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::default::pow2::CKKSPow2Default<$backend>>::ckks_mul_pow2_into_default(module, dst, src, bits, scratch)
        }

        fn ckks_mul_pow2_assign(
            module: &poulpy_hal::layouts::Module<$backend>,
            dst: &mut $crate::layouts::CKKSCiphertext<impl poulpy_hal::layouts::DataMut>,
            bits: usize,
            scratch: &mut poulpy_hal::layouts::Scratch<$backend>,
        ) -> anyhow::Result<()>
        where
            poulpy_hal::layouts::Module<$backend>: poulpy_core::GLWEShift<$backend>,
            poulpy_hal::layouts::Scratch<$backend>: poulpy_core::ScratchTakeCore<$backend>,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::default::pow2::CKKSPow2Default<$backend>>::ckks_mul_pow2_assign_default(module, dst, bits, scratch)
        }

        fn ckks_div_pow2_tmp_bytes(module: &poulpy_hal::layouts::Module<$backend>) -> usize
        where
            poulpy_hal::layouts::Module<$backend>: poulpy_core::GLWEShift<$backend>,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::default::pow2::CKKSPow2Default<$backend>>::ckks_div_pow2_tmp_bytes_default(module)
        }

        fn ckks_div_pow2_into(
            module: &poulpy_hal::layouts::Module<$backend>,
            dst: &mut $crate::layouts::CKKSCiphertext<impl poulpy_hal::layouts::DataMut>,
            src: &$crate::layouts::CKKSCiphertext<impl poulpy_hal::layouts::DataRef>,
            bits: usize,
            scratch: &mut poulpy_hal::layouts::Scratch<$backend>,
        ) -> anyhow::Result<()>
        where
            poulpy_hal::layouts::Module<$backend>: poulpy_core::GLWEShift<$backend> + poulpy_core::GLWECopy,
            poulpy_hal::layouts::Scratch<$backend>: poulpy_core::ScratchTakeCore<$backend>,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::default::pow2::CKKSPow2Default<$backend>>::ckks_div_pow2_into_default(module, dst, src, bits, scratch)
        }

        fn ckks_div_pow2_assign(
            module: &poulpy_hal::layouts::Module<$backend>,
            dst: &mut $crate::layouts::CKKSCiphertext<impl poulpy_hal::layouts::DataMut>,
            bits: usize,
        ) -> anyhow::Result<()> {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::default::pow2::CKKSPow2Default<$backend>>::ckks_div_pow2_assign_default(module, dst, bits)
        }
    };
}

pub use crate::impl_ckks_pow2_default_methods;
