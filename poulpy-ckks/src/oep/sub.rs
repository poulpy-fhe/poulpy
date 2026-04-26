#[macro_export]
macro_rules! impl_ckks_sub_default_methods {
    ($backend:ty) => {
        fn ckks_sub_tmp_bytes(module: &poulpy_hal::layouts::Module<$backend>) -> usize
        where
            poulpy_hal::layouts::Module<$backend>: poulpy_core::GLWEShift<$backend> + poulpy_hal::api::VecZnxRshTmpBytes,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::default::sub::CKKSSubDefault<$backend>>::ckks_sub_tmp_bytes_default(module)
        }

        fn ckks_sub_pt_vec_znx_tmp_bytes(module: &poulpy_hal::layouts::Module<$backend>) -> usize
        where
            poulpy_hal::layouts::Module<$backend>: poulpy_core::GLWEShift<$backend> + poulpy_hal::api::VecZnxRshTmpBytes,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::default::sub::CKKSSubDefault<$backend>>::ckks_sub_pt_vec_znx_tmp_bytes_default(module)
        }

        fn ckks_sub_into(
            module: &poulpy_hal::layouts::Module<$backend>,
            dst: &mut $crate::layouts::CKKSCiphertext<impl poulpy_hal::layouts::DataMut>,
            a: &$crate::layouts::CKKSCiphertext<impl poulpy_hal::layouts::DataRef>,
            b: &$crate::layouts::CKKSCiphertext<impl poulpy_hal::layouts::DataRef>,
            scratch: &mut poulpy_hal::layouts::Scratch<$backend>,
        ) -> anyhow::Result<()>
        where
            poulpy_hal::layouts::Module<$backend>: poulpy_core::GLWESub + poulpy_core::GLWEShift<$backend>,
            poulpy_hal::layouts::Scratch<$backend>: poulpy_hal::api::ScratchAvailable + poulpy_core::ScratchTakeCore<$backend>,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::default::sub::CKKSSubDefault<$backend>>::ckks_sub_into_default(module, dst, a, b, scratch)
        }

        fn ckks_sub_assign(
            module: &poulpy_hal::layouts::Module<$backend>,
            dst: &mut $crate::layouts::CKKSCiphertext<impl poulpy_hal::layouts::DataMut>,
            a: &$crate::layouts::CKKSCiphertext<impl poulpy_hal::layouts::DataRef>,
            scratch: &mut poulpy_hal::layouts::Scratch<$backend>,
        ) -> anyhow::Result<()>
        where
            poulpy_hal::layouts::Module<$backend>: poulpy_core::GLWESub + poulpy_core::GLWEShift<$backend>,
            poulpy_hal::layouts::Scratch<$backend>: poulpy_hal::api::ScratchAvailable + poulpy_core::ScratchTakeCore<$backend>,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::default::sub::CKKSSubDefault<$backend>>::ckks_sub_assign_default(module, dst, a, scratch)
        }

        fn ckks_sub_pt_vec_znx_into(
            module: &poulpy_hal::layouts::Module<$backend>,
            dst: &mut $crate::layouts::CKKSCiphertext<impl poulpy_hal::layouts::DataMut>,
            a: &$crate::layouts::CKKSCiphertext<impl poulpy_hal::layouts::DataRef>,
            pt_znx: &$crate::layouts::plaintext::CKKSPlaintextVecZnx<impl poulpy_hal::layouts::DataRef>,
            scratch: &mut poulpy_hal::layouts::Scratch<$backend>,
        ) -> anyhow::Result<()>
        where
            poulpy_hal::layouts::Module<$backend>: poulpy_hal::api::VecZnxRshSub<$backend> + poulpy_core::GLWEShift<$backend>,
            poulpy_hal::layouts::Scratch<$backend>: poulpy_hal::api::ScratchAvailable + poulpy_core::ScratchTakeCore<$backend>,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::default::sub::CKKSSubDefault<$backend>>::ckks_sub_pt_vec_znx_into_default(module, dst, a, pt_znx, scratch)
        }

        fn ckks_sub_pt_vec_znx_assign(
            module: &poulpy_hal::layouts::Module<$backend>,
            dst: &mut $crate::layouts::CKKSCiphertext<impl poulpy_hal::layouts::DataMut>,
            pt_znx: &$crate::layouts::plaintext::CKKSPlaintextVecZnx<impl poulpy_hal::layouts::DataRef>,
            scratch: &mut poulpy_hal::layouts::Scratch<$backend>,
        ) -> anyhow::Result<()>
        where
            poulpy_hal::layouts::Module<$backend>: poulpy_hal::api::VecZnxRshSub<$backend>,
            poulpy_hal::layouts::Scratch<$backend>: poulpy_hal::api::ScratchAvailable + poulpy_core::ScratchTakeCore<$backend>,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::default::sub::CKKSSubDefault<$backend>>::ckks_sub_pt_vec_znx_assign_default(module, dst, pt_znx, scratch)
        }

        fn ckks_sub_pt_vec_rnx_tmp_bytes<R: poulpy_core::layouts::GLWEInfos, A: poulpy_core::layouts::GLWEInfos>(
            module: &poulpy_hal::layouts::Module<$backend>,
            res: &R,
            a: &A,
            b: &$crate::CKKSMeta,
        ) -> usize
        where
            poulpy_hal::layouts::Module<$backend>: poulpy_hal::api::ModuleN + poulpy_core::GLWEShift<$backend> + poulpy_hal::api::VecZnxRshTmpBytes,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::default::sub::CKKSSubDefault<$backend>>::ckks_sub_pt_vec_rnx_tmp_bytes_default(module, res, a, b)
        }

        fn ckks_sub_pt_vec_rnx_into<F>(
            module: &poulpy_hal::layouts::Module<$backend>,
            dst: &mut $crate::layouts::CKKSCiphertext<impl poulpy_hal::layouts::DataMut>,
            a: &$crate::layouts::CKKSCiphertext<impl poulpy_hal::layouts::DataRef>,
            pt_rnx: &$crate::layouts::plaintext::CKKSPlaintextVecRnx<F>,
            prec: $crate::CKKSMeta,
            scratch: &mut poulpy_hal::layouts::Scratch<$backend>,
        ) -> anyhow::Result<()>
        where
            poulpy_hal::layouts::Module<$backend>: poulpy_hal::api::ModuleN + poulpy_hal::api::VecZnxRshSub<$backend> + poulpy_core::GLWEShift<$backend>,
            poulpy_hal::layouts::Scratch<$backend>: poulpy_hal::api::ScratchAvailable + poulpy_core::ScratchTakeCore<$backend>,
            $crate::layouts::plaintext::CKKSPlaintextVecRnx<F>: $crate::layouts::plaintext::CKKSPlaintextConversion,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::default::sub::CKKSSubDefault<$backend>>::ckks_sub_pt_vec_rnx_into_default(module, dst, a, pt_rnx, prec, scratch)
        }

        fn ckks_sub_pt_vec_rnx_assign<F>(
            module: &poulpy_hal::layouts::Module<$backend>,
            dst: &mut $crate::layouts::CKKSCiphertext<impl poulpy_hal::layouts::DataMut>,
            pt_rnx: &$crate::layouts::plaintext::CKKSPlaintextVecRnx<F>,
            prec: $crate::CKKSMeta,
            scratch: &mut poulpy_hal::layouts::Scratch<$backend>,
        ) -> anyhow::Result<()>
        where
            poulpy_hal::layouts::Module<$backend>: poulpy_hal::api::ModuleN + poulpy_hal::api::VecZnxRshSub<$backend>,
            poulpy_hal::layouts::Scratch<$backend>: poulpy_hal::api::ScratchAvailable + poulpy_core::ScratchTakeCore<$backend>,
            $crate::layouts::plaintext::CKKSPlaintextVecRnx<F>: $crate::layouts::plaintext::CKKSPlaintextConversion,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::default::sub::CKKSSubDefault<$backend>>::ckks_sub_pt_vec_rnx_assign_default(module, dst, pt_rnx, prec, scratch)
        }

        fn ckks_sub_pt_const_tmp_bytes(module: &poulpy_hal::layouts::Module<$backend>) -> usize
        where
            poulpy_hal::layouts::Module<$backend>: poulpy_core::GLWEShift<$backend>,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::default::sub::CKKSSubDefault<$backend>>::ckks_sub_pt_const_tmp_bytes_default(module)
        }

        fn ckks_sub_pt_const_znx_into(
            module: &poulpy_hal::layouts::Module<$backend>,
            dst: &mut $crate::layouts::CKKSCiphertext<impl poulpy_hal::layouts::DataMut>,
            a: &$crate::layouts::CKKSCiphertext<impl poulpy_hal::layouts::DataRef>,
            cst_znx: &$crate::layouts::plaintext::CKKSPlaintextCstZnx,
            scratch: &mut poulpy_hal::layouts::Scratch<$backend>,
        ) -> anyhow::Result<()>
        where
            poulpy_hal::layouts::Module<$backend>: poulpy_core::GLWEShift<$backend>,
            poulpy_hal::layouts::Scratch<$backend>: poulpy_hal::api::ScratchAvailable + poulpy_core::ScratchTakeCore<$backend>,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::default::sub::CKKSSubDefault<$backend>>::ckks_sub_pt_const_znx_into_default(module, dst, a, cst_znx, scratch)
        }

        fn ckks_sub_pt_const_znx_assign(
            module: &poulpy_hal::layouts::Module<$backend>,
            dst: &mut $crate::layouts::CKKSCiphertext<impl poulpy_hal::layouts::DataMut>,
            cst_znx: &$crate::layouts::plaintext::CKKSPlaintextCstZnx,
            scratch: &mut poulpy_hal::layouts::Scratch<$backend>,
        ) -> anyhow::Result<()>
        where
            poulpy_hal::layouts::Scratch<$backend>: poulpy_hal::api::ScratchAvailable + poulpy_core::ScratchTakeCore<$backend>,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::default::sub::CKKSSubDefault<$backend>>::ckks_sub_pt_const_znx_assign_default(module, dst, cst_znx, scratch)
        }

        fn ckks_sub_pt_const_rnx_into<F>(
            module: &poulpy_hal::layouts::Module<$backend>,
            dst: &mut $crate::layouts::CKKSCiphertext<impl poulpy_hal::layouts::DataMut>,
            a: &$crate::layouts::CKKSCiphertext<impl poulpy_hal::layouts::DataRef>,
            cst_rnx: &$crate::layouts::plaintext::CKKSPlaintextCstRnx<F>,
            prec: $crate::CKKSMeta,
            scratch: &mut poulpy_hal::layouts::Scratch<$backend>,
        ) -> anyhow::Result<()>
        where
            poulpy_hal::layouts::Module<$backend>: poulpy_core::GLWEShift<$backend>,
            poulpy_hal::layouts::Scratch<$backend>: poulpy_hal::api::ScratchAvailable + poulpy_core::ScratchTakeCore<$backend>,
            $crate::layouts::plaintext::CKKSPlaintextCstRnx<F>: $crate::layouts::plaintext::CKKSConstPlaintextConversion,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::default::sub::CKKSSubDefault<$backend>>::ckks_sub_pt_const_rnx_into_default(module, dst, a, cst_rnx, prec, scratch)
        }

        fn ckks_sub_pt_const_rnx_assign<F>(
            module: &poulpy_hal::layouts::Module<$backend>,
            dst: &mut $crate::layouts::CKKSCiphertext<impl poulpy_hal::layouts::DataMut>,
            cst_rnx: &$crate::layouts::plaintext::CKKSPlaintextCstRnx<F>,
            prec: $crate::CKKSMeta,
            scratch: &mut poulpy_hal::layouts::Scratch<$backend>,
        ) -> anyhow::Result<()>
        where
            poulpy_hal::layouts::Scratch<$backend>: poulpy_hal::api::ScratchAvailable + poulpy_core::ScratchTakeCore<$backend>,
            $crate::layouts::plaintext::CKKSPlaintextCstRnx<F>: $crate::layouts::plaintext::CKKSConstPlaintextConversion,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::default::sub::CKKSSubDefault<$backend>>::ckks_sub_pt_const_rnx_assign_default(module, dst, cst_rnx, prec, scratch)
        }
    };
}

pub use crate::impl_ckks_sub_default_methods;
