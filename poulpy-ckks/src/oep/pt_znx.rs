#[macro_export]
macro_rules! impl_ckks_pt_znx_default_methods {
    ($backend:ty) => {
        fn ckks_extract_pt_znx_tmp_bytes(module: &poulpy_hal::layouts::Module<$backend>) -> usize
        where
            poulpy_hal::layouts::Module<$backend>: poulpy_hal::api::VecZnxLshTmpBytes + poulpy_hal::api::VecZnxRshTmpBytes,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::default::pt_znx::CKKSPlaintextZnxDefault<$backend>>::ckks_extract_pt_znx_tmp_bytes_default(module)
        }

        fn ckks_extract_pt_znx<S: $crate::CKKSInfos>(
            module: &poulpy_hal::layouts::Module<$backend>,
            dst: &mut $crate::layouts::plaintext::CKKSPlaintextVecZnx<impl poulpy_hal::layouts::DataMut>,
            src: &poulpy_core::layouts::GLWEPlaintext<impl poulpy_hal::layouts::DataRef>,
            src_meta: &S,
            scratch: &mut poulpy_hal::layouts::Scratch<$backend>,
        ) -> anyhow::Result<()>
        where
            poulpy_hal::layouts::Scratch<$backend>: poulpy_core::ScratchTakeCore<$backend>,
            poulpy_hal::layouts::Module<$backend>: poulpy_hal::api::VecZnxLsh<$backend> + poulpy_hal::api::VecZnxRsh<$backend>,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::default::pt_znx::CKKSPlaintextZnxDefault<$backend>>::ckks_extract_pt_znx_default(module, dst, src, src_meta, scratch)
        }
    };
}

pub use crate::impl_ckks_pt_znx_default_methods;
