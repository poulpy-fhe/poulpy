#[macro_export]
macro_rules! impl_ckks_mul_default_methods {
    ($backend:ty) => {
        fn ckks_mul_tmp_bytes<R: poulpy_core::layouts::GLWEInfos, T: poulpy_core::layouts::GGLWEInfos>(
            module: &poulpy_hal::layouts::Module<$backend>,
            res: &R,
            tsk: &T,
        ) -> usize
        where
            poulpy_hal::layouts::Module<$backend>: poulpy_core::GLWETensoring<$backend>,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::default::mul::CKKSMulDefault<$backend>>::ckks_mul_tmp_bytes_default(module, res, tsk)
        }

        fn ckks_square_tmp_bytes<R: poulpy_core::layouts::GLWEInfos, T: poulpy_core::layouts::GGLWEInfos>(
            module: &poulpy_hal::layouts::Module<$backend>,
            res: &R,
            tsk: &T,
        ) -> usize
        where
            poulpy_hal::layouts::Module<$backend>: poulpy_core::GLWETensoring<$backend>,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::default::mul::CKKSMulDefault<$backend>>::ckks_square_tmp_bytes_default(module, res, tsk)
        }

        fn ckks_mul_pt_vec_znx_tmp_bytes<R: poulpy_core::layouts::GLWEInfos, A: poulpy_core::layouts::GLWEInfos>(
            module: &poulpy_hal::layouts::Module<$backend>,
            res: &R,
            a: &A,
            b: &$crate::CKKSMeta,
        ) -> usize
        where
            poulpy_hal::layouts::Module<$backend>: poulpy_core::GLWEMulPlain<$backend>,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::default::mul::CKKSMulDefault<$backend>>::ckks_mul_pt_vec_znx_tmp_bytes_default(module, res, a, b)
        }

        fn ckks_mul_pt_vec_rnx_tmp_bytes<R: poulpy_core::layouts::GLWEInfos, A: poulpy_core::layouts::GLWEInfos>(
            module: &poulpy_hal::layouts::Module<$backend>,
            res: &R,
            a: &A,
            b: &$crate::CKKSMeta,
        ) -> usize
        where
            poulpy_hal::layouts::Module<$backend>: poulpy_hal::api::ModuleN + poulpy_core::GLWEMulPlain<$backend>,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::default::mul::CKKSMulDefault<$backend>>::ckks_mul_pt_vec_rnx_tmp_bytes_default(module, res, a, b)
        }

        fn ckks_mul_pt_const_tmp_bytes<R: poulpy_core::layouts::GLWEInfos, A: poulpy_core::layouts::GLWEInfos>(
            module: &poulpy_hal::layouts::Module<$backend>,
            res: &R,
            a: &A,
            b: &$crate::CKKSMeta,
        ) -> usize
        where
            poulpy_hal::layouts::Module<$backend>: poulpy_core::GLWEMulConst<$backend> + poulpy_core::GLWERotate<$backend>,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::default::mul::CKKSMulDefault<$backend>>::ckks_mul_pt_const_tmp_bytes_default(module, res, a, b)
        }

        fn ckks_mul_into(
            module: &poulpy_hal::layouts::Module<$backend>,
            dst: &mut $crate::layouts::CKKSCiphertext<impl poulpy_hal::layouts::DataMut>,
            a: &$crate::layouts::CKKSCiphertext<impl poulpy_hal::layouts::DataRef>,
            b: &$crate::layouts::CKKSCiphertext<impl poulpy_hal::layouts::DataRef>,
            tsk: &poulpy_core::layouts::GLWETensorKeyPrepared<impl poulpy_hal::layouts::DataRef, $backend>,
            scratch: &mut poulpy_hal::layouts::Scratch<$backend>,
        ) -> anyhow::Result<()>
        where
            poulpy_hal::layouts::Module<$backend>: poulpy_core::GLWETensoring<$backend>,
            poulpy_hal::layouts::Scratch<$backend>: poulpy_hal::api::ScratchAvailable + poulpy_core::ScratchTakeCore<$backend>,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::default::mul::CKKSMulDefault<$backend>>::ckks_mul_into_default(module, dst, a, b, tsk, scratch)
        }

        fn ckks_mul_assign(
            module: &poulpy_hal::layouts::Module<$backend>,
            dst: &mut $crate::layouts::CKKSCiphertext<impl poulpy_hal::layouts::DataMut>,
            a: &$crate::layouts::CKKSCiphertext<impl poulpy_hal::layouts::DataRef>,
            tsk: &poulpy_core::layouts::GLWETensorKeyPrepared<impl poulpy_hal::layouts::DataRef, $backend>,
            scratch: &mut poulpy_hal::layouts::Scratch<$backend>,
        ) -> anyhow::Result<()>
        where
            poulpy_hal::layouts::Module<$backend>: poulpy_core::GLWETensoring<$backend>,
            poulpy_hal::layouts::Scratch<$backend>: poulpy_hal::api::ScratchAvailable + poulpy_core::ScratchTakeCore<$backend>,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::default::mul::CKKSMulDefault<$backend>>::ckks_mul_assign_default(module, dst, a, tsk, scratch)
        }

        fn ckks_square_into(
            module: &poulpy_hal::layouts::Module<$backend>,
            dst: &mut $crate::layouts::CKKSCiphertext<impl poulpy_hal::layouts::DataMut>,
            a: &$crate::layouts::CKKSCiphertext<impl poulpy_hal::layouts::DataRef>,
            tsk: &poulpy_core::layouts::GLWETensorKeyPrepared<impl poulpy_hal::layouts::DataRef, $backend>,
            scratch: &mut poulpy_hal::layouts::Scratch<$backend>,
        ) -> anyhow::Result<()>
        where
            poulpy_hal::layouts::Module<$backend>: poulpy_core::GLWETensoring<$backend>,
            poulpy_hal::layouts::Scratch<$backend>: poulpy_hal::api::ScratchAvailable + poulpy_core::ScratchTakeCore<$backend>,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::default::mul::CKKSMulDefault<$backend>>::ckks_square_into_default(module, dst, a, tsk, scratch)
        }

        fn ckks_square_assign(
            module: &poulpy_hal::layouts::Module<$backend>,
            dst: &mut $crate::layouts::CKKSCiphertext<impl poulpy_hal::layouts::DataMut>,
            tsk: &poulpy_core::layouts::GLWETensorKeyPrepared<impl poulpy_hal::layouts::DataRef, $backend>,
            scratch: &mut poulpy_hal::layouts::Scratch<$backend>,
        ) -> anyhow::Result<()>
        where
            poulpy_hal::layouts::Module<$backend>: poulpy_core::GLWETensoring<$backend>,
            poulpy_hal::layouts::Scratch<$backend>: poulpy_hal::api::ScratchAvailable + poulpy_core::ScratchTakeCore<$backend>,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::default::mul::CKKSMulDefault<$backend>>::ckks_square_assign_default(module, dst, tsk, scratch)
        }

        fn ckks_mul_pt_vec_znx_into(
            module: &poulpy_hal::layouts::Module<$backend>,
            dst: &mut $crate::layouts::CKKSCiphertext<impl poulpy_hal::layouts::DataMut>,
            a: &$crate::layouts::CKKSCiphertext<impl poulpy_hal::layouts::DataRef>,
            pt_znx: &$crate::layouts::plaintext::CKKSPlaintextVecZnx<impl poulpy_hal::layouts::DataRef>,
            scratch: &mut poulpy_hal::layouts::Scratch<$backend>,
        ) -> anyhow::Result<()>
        where
            poulpy_hal::layouts::Module<$backend>: poulpy_core::GLWEMulPlain<$backend>,
            poulpy_hal::layouts::Scratch<$backend>: poulpy_hal::api::ScratchAvailable + poulpy_core::ScratchTakeCore<$backend>,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::default::mul::CKKSMulDefault<$backend>>::ckks_mul_pt_vec_znx_into_default(module, dst, a, pt_znx, scratch)
        }

        fn ckks_mul_pt_vec_znx_assign(
            module: &poulpy_hal::layouts::Module<$backend>,
            dst: &mut $crate::layouts::CKKSCiphertext<impl poulpy_hal::layouts::DataMut>,
            pt_znx: &$crate::layouts::plaintext::CKKSPlaintextVecZnx<impl poulpy_hal::layouts::DataRef>,
            scratch: &mut poulpy_hal::layouts::Scratch<$backend>,
        ) -> anyhow::Result<()>
        where
            poulpy_hal::layouts::Module<$backend>: poulpy_core::GLWEMulPlain<$backend>,
            poulpy_hal::layouts::Scratch<$backend>: poulpy_hal::api::ScratchAvailable + poulpy_core::ScratchTakeCore<$backend>,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::default::mul::CKKSMulDefault<$backend>>::ckks_mul_pt_vec_znx_assign_default(module, dst, pt_znx, scratch)
        }

        fn ckks_mul_pt_vec_rnx_into<F>(
            module: &poulpy_hal::layouts::Module<$backend>,
            dst: &mut $crate::layouts::CKKSCiphertext<impl poulpy_hal::layouts::DataMut>,
            a: &$crate::layouts::CKKSCiphertext<impl poulpy_hal::layouts::DataRef>,
            pt_rnx: &$crate::layouts::plaintext::CKKSPlaintextVecRnx<F>,
            prec: $crate::CKKSMeta,
            scratch: &mut poulpy_hal::layouts::Scratch<$backend>,
        ) -> anyhow::Result<()>
        where
            poulpy_hal::layouts::Module<$backend>: poulpy_hal::api::ModuleN + poulpy_core::GLWEMulPlain<$backend>,
            $crate::layouts::plaintext::CKKSPlaintextVecRnx<F>: $crate::layouts::plaintext::CKKSPlaintextConversion,
            poulpy_hal::layouts::Scratch<$backend>: poulpy_hal::api::ScratchAvailable + poulpy_core::ScratchTakeCore<$backend>,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::default::mul::CKKSMulDefault<$backend>>::ckks_mul_pt_vec_rnx_into_default(module, dst, a, pt_rnx, prec, scratch)
        }

        fn ckks_mul_pt_vec_rnx_assign<F>(
            module: &poulpy_hal::layouts::Module<$backend>,
            dst: &mut $crate::layouts::CKKSCiphertext<impl poulpy_hal::layouts::DataMut>,
            pt_rnx: &$crate::layouts::plaintext::CKKSPlaintextVecRnx<F>,
            prec: $crate::CKKSMeta,
            scratch: &mut poulpy_hal::layouts::Scratch<$backend>,
        ) -> anyhow::Result<()>
        where
            poulpy_hal::layouts::Module<$backend>: poulpy_hal::api::ModuleN + poulpy_core::GLWEMulPlain<$backend>,
            $crate::layouts::plaintext::CKKSPlaintextVecRnx<F>: $crate::layouts::plaintext::CKKSPlaintextConversion,
            poulpy_hal::layouts::Scratch<$backend>: poulpy_hal::api::ScratchAvailable + poulpy_core::ScratchTakeCore<$backend>,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::default::mul::CKKSMulDefault<$backend>>::ckks_mul_pt_vec_rnx_assign_default(module, dst, pt_rnx, prec, scratch)
        }

        fn ckks_mul_pt_const_znx_into(
            module: &poulpy_hal::layouts::Module<$backend>,
            dst: &mut $crate::layouts::CKKSCiphertext<impl poulpy_hal::layouts::DataMut>,
            a: &$crate::layouts::CKKSCiphertext<impl poulpy_hal::layouts::DataRef>,
            cst_znx: &$crate::layouts::plaintext::CKKSPlaintextCstZnx,
            scratch: &mut poulpy_hal::layouts::Scratch<$backend>,
        ) -> anyhow::Result<()>
        where
            poulpy_hal::layouts::Module<$backend>: poulpy_core::GLWEAdd + poulpy_core::GLWEMulConst<$backend> + poulpy_core::GLWERotate<$backend>,
            poulpy_hal::layouts::Scratch<$backend>: poulpy_hal::api::ScratchAvailable + poulpy_core::ScratchTakeCore<$backend>,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::default::mul::CKKSMulDefault<$backend>>::ckks_mul_pt_const_znx_into_default(module, dst, a, cst_znx, scratch)
        }

        fn ckks_mul_pt_const_znx_assign(
            module: &poulpy_hal::layouts::Module<$backend>,
            dst: &mut $crate::layouts::CKKSCiphertext<impl poulpy_hal::layouts::DataMut>,
            cst_znx: &$crate::layouts::plaintext::CKKSPlaintextCstZnx,
            scratch: &mut poulpy_hal::layouts::Scratch<$backend>,
        ) -> anyhow::Result<()>
        where
            poulpy_hal::layouts::Module<$backend>: poulpy_core::GLWEAdd + poulpy_core::GLWEMulConst<$backend> + poulpy_core::GLWERotate<$backend>,
            poulpy_hal::layouts::Scratch<$backend>: poulpy_hal::api::ScratchAvailable + poulpy_core::ScratchTakeCore<$backend>,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::default::mul::CKKSMulDefault<$backend>>::ckks_mul_pt_const_znx_assign_default(module, dst, cst_znx, scratch)
        }

        fn ckks_mul_pt_const_rnx_into<F>(
            module: &poulpy_hal::layouts::Module<$backend>,
            dst: &mut $crate::layouts::CKKSCiphertext<impl poulpy_hal::layouts::DataMut>,
            a: &$crate::layouts::CKKSCiphertext<impl poulpy_hal::layouts::DataRef>,
            cst_rnx: &$crate::layouts::plaintext::CKKSPlaintextCstRnx<F>,
            prec: $crate::CKKSMeta,
            scratch: &mut poulpy_hal::layouts::Scratch<$backend>,
        ) -> anyhow::Result<()>
        where
            poulpy_hal::layouts::Module<$backend>: poulpy_core::GLWEAdd + poulpy_core::GLWEMulConst<$backend> + poulpy_core::GLWERotate<$backend>,
            poulpy_hal::layouts::Scratch<$backend>: poulpy_hal::api::ScratchAvailable + poulpy_core::ScratchTakeCore<$backend>,
            $crate::layouts::plaintext::CKKSPlaintextCstRnx<F>: $crate::layouts::plaintext::CKKSConstPlaintextConversion,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::default::mul::CKKSMulDefault<$backend>>::ckks_mul_pt_const_rnx_into_default(module, dst, a, cst_rnx, prec, scratch)
        }

        fn ckks_mul_pt_const_rnx_assign<F>(
            module: &poulpy_hal::layouts::Module<$backend>,
            dst: &mut $crate::layouts::CKKSCiphertext<impl poulpy_hal::layouts::DataMut>,
            cst_rnx: &$crate::layouts::plaintext::CKKSPlaintextCstRnx<F>,
            prec: $crate::CKKSMeta,
            scratch: &mut poulpy_hal::layouts::Scratch<$backend>,
        ) -> anyhow::Result<()>
        where
            poulpy_hal::layouts::Module<$backend>: poulpy_core::GLWEAdd + poulpy_core::GLWEMulConst<$backend> + poulpy_core::GLWERotate<$backend>,
            poulpy_hal::layouts::Scratch<$backend>: poulpy_hal::api::ScratchAvailable + poulpy_core::ScratchTakeCore<$backend>,
            $crate::layouts::plaintext::CKKSPlaintextCstRnx<F>: $crate::layouts::plaintext::CKKSConstPlaintextConversion,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::default::mul::CKKSMulDefault<$backend>>::ckks_mul_pt_const_rnx_assign_default(module, dst, cst_rnx, prec, scratch)
        }
    };
}

pub use crate::impl_ckks_mul_default_methods;
