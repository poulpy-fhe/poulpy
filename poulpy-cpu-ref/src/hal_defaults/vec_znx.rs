//! Backend extension points for coefficient-domain [`VecZnx`](poulpy_hal::layouts::VecZnx) operations.

use std::mem::size_of;

use crate::hal_defaults::scratch::HalScratchDefaults;
use crate::reference::vec_znx::{
    vec_znx_add_assign, vec_znx_add_into, vec_znx_add_normal_ref, vec_znx_add_scalar_assign, vec_znx_add_scalar_into,
    vec_znx_automorphism, vec_znx_automorphism_assign, vec_znx_automorphism_assign_tmp_bytes, vec_znx_copy,
    vec_znx_fill_normal_ref, vec_znx_fill_uniform_ref, vec_znx_lsh, vec_znx_lsh_assign, vec_znx_lsh_sub, vec_znx_lsh_tmp_bytes,
    vec_znx_merge_rings, vec_znx_merge_rings_tmp_bytes, vec_znx_mul_xp_minus_one, vec_znx_mul_xp_minus_one_assign,
    vec_znx_mul_xp_minus_one_assign_tmp_bytes, vec_znx_negate, vec_znx_negate_assign, vec_znx_normalize,
    vec_znx_normalize_assign, vec_znx_normalize_tmp_bytes, vec_znx_rotate, vec_znx_rotate_assign,
    vec_znx_rotate_assign_tmp_bytes, vec_znx_rsh, vec_znx_rsh_assign, vec_znx_rsh_sub, vec_znx_rsh_tmp_bytes, vec_znx_split_ring,
    vec_znx_split_ring_tmp_bytes, vec_znx_sub, vec_znx_sub_assign, vec_znx_sub_negate_assign, vec_znx_sub_scalar,
    vec_znx_sub_scalar_assign, vec_znx_switch_ring, vec_znx_zero,
};
use crate::reference::znx::{
    ZnxAdd, ZnxAddAssign, ZnxAutomorphism, ZnxCopy, ZnxExtractDigitAddMul, ZnxMulPowerOfTwoAssign, ZnxNegate, ZnxNegateAssign,
    ZnxNormalizeDigit, ZnxNormalizeFinalStep, ZnxNormalizeFinalStepAssign, ZnxNormalizeFinalStepSub, ZnxNormalizeFirstStep,
    ZnxNormalizeFirstStepAssign, ZnxNormalizeFirstStepCarryOnly, ZnxNormalizeMiddleStep, ZnxNormalizeMiddleStepAssign,
    ZnxNormalizeMiddleStepCarryOnly, ZnxNormalizeMiddleStepSub, ZnxRotate, ZnxSub, ZnxSubAssign, ZnxSubNegateAssign,
    ZnxSwitchRing, ZnxZero,
};
use poulpy_hal::{
    layouts::{Backend, Module, NoiseInfos, ScalarZnxToRef, Scratch, VecZnxToMut, VecZnxToRef},
    source::Source,
};
#[doc(hidden)]
pub trait HalVecZnxDefaults<BE: Backend>: Backend {
    fn vec_znx_zero_default<R>(_module: &Module<BE>, res: &mut R, res_col: usize)
    where
        BE: ZnxZero,
        R: VecZnxToMut,
    {
        vec_znx_zero::<R, BE>(res, res_col);
    }

    fn vec_znx_normalize_tmp_bytes_default(module: &Module<BE>) -> usize {
        vec_znx_normalize_tmp_bytes(module.n())
    }

    #[allow(clippy::too_many_arguments)]
    fn vec_znx_normalize_default<R, A>(
        module: &Module<BE>,
        res: &mut R,
        res_base2k: usize,
        res_offset: i64,
        res_col: usize,
        a: &A,
        a_base2k: usize,
        a_col: usize,
        scratch: &mut Scratch<BE>,
    ) where
        BE: ZnxZero
            + ZnxCopy
            + ZnxAddAssign
            + ZnxMulPowerOfTwoAssign
            + ZnxNormalizeFirstStepCarryOnly
            + ZnxNormalizeMiddleStepCarryOnly
            + ZnxNormalizeMiddleStep
            + ZnxNormalizeFinalStep
            + ZnxNormalizeFirstStep
            + ZnxExtractDigitAddMul
            + ZnxNormalizeMiddleStepAssign
            + ZnxNormalizeFinalStepAssign
            + ZnxNormalizeDigit,
        R: VecZnxToMut,
        A: VecZnxToRef,
    {
        let byte_count = vec_znx_normalize_tmp_bytes(module.n());
        assert!(
            byte_count.is_multiple_of(size_of::<i64>()),
            "Scratch buffer size {} must be divisible by {}",
            byte_count,
            size_of::<i64>()
        );
        let (carry, _) = <BE as HalScratchDefaults<BE>>::take_slice_default::<i64>(scratch, byte_count / size_of::<i64>());
        vec_znx_normalize::<R, A, BE>(res, res_base2k, res_offset, res_col, a, a_base2k, a_col, carry);
    }

    fn vec_znx_normalize_assign_default<R>(
        module: &Module<BE>,
        base2k: usize,
        res: &mut R,
        res_col: usize,
        scratch: &mut Scratch<BE>,
    ) where
        BE: ZnxNormalizeFirstStepAssign + ZnxNormalizeMiddleStepAssign + ZnxNormalizeFinalStepAssign,
        R: VecZnxToMut,
    {
        let byte_count = vec_znx_normalize_tmp_bytes(module.n());
        assert!(
            byte_count.is_multiple_of(size_of::<i64>()),
            "Scratch buffer size {} must be divisible by {}",
            byte_count,
            size_of::<i64>()
        );
        let (carry, _) = <BE as HalScratchDefaults<BE>>::take_slice_default::<i64>(scratch, byte_count / size_of::<i64>());
        vec_znx_normalize_assign::<R, BE>(base2k, res, res_col, carry);
    }

    fn vec_znx_add_into_default<R, A, C>(
        _module: &Module<BE>,
        res: &mut R,
        res_col: usize,
        a: &A,
        a_col: usize,
        b: &C,
        b_col: usize,
    ) where
        BE: ZnxAdd + ZnxCopy + ZnxZero,
        R: VecZnxToMut,
        A: VecZnxToRef,
        C: VecZnxToRef,
    {
        vec_znx_add_into::<R, A, C, BE>(res, res_col, a, a_col, b, b_col);
    }

    fn vec_znx_add_assign_default<R, A>(_module: &Module<BE>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        BE: ZnxAddAssign,
        R: VecZnxToMut,
        A: VecZnxToRef,
    {
        vec_znx_add_assign::<R, A, BE>(res, res_col, a, a_col);
    }

    #[allow(clippy::too_many_arguments)]
    fn vec_znx_add_scalar_into_default<R, A, B>(
        _module: &Module<BE>,
        res: &mut R,
        res_col: usize,
        a: &A,
        a_col: usize,
        b: &B,
        b_col: usize,
        b_limb: usize,
    ) where
        BE: ZnxAdd + ZnxCopy + ZnxZero,
        R: VecZnxToMut,
        A: ScalarZnxToRef,
        B: VecZnxToRef,
    {
        vec_znx_add_scalar_into::<R, A, B, BE>(res, res_col, a, a_col, b, b_col, b_limb);
    }

    fn vec_znx_add_scalar_assign_default<R, A>(
        _module: &Module<BE>,
        res: &mut R,
        res_col: usize,
        res_limb: usize,
        a: &A,
        a_col: usize,
    ) where
        BE: ZnxAddAssign,
        R: VecZnxToMut,
        A: ScalarZnxToRef,
    {
        vec_znx_add_scalar_assign::<R, A, BE>(res, res_col, res_limb, a, a_col);
    }

    fn vec_znx_sub_default<R, A, C>(_module: &Module<BE>, res: &mut R, res_col: usize, a: &A, a_col: usize, b: &C, b_col: usize)
    where
        BE: ZnxSub + ZnxNegate + ZnxZero + ZnxCopy,
        R: VecZnxToMut,
        A: VecZnxToRef,
        C: VecZnxToRef,
    {
        vec_znx_sub::<R, A, C, BE>(res, res_col, a, a_col, b, b_col);
    }

    fn vec_znx_sub_assign_default<R, A>(_module: &Module<BE>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        BE: ZnxSubAssign,
        R: VecZnxToMut,
        A: VecZnxToRef,
    {
        vec_znx_sub_assign::<R, A, BE>(res, res_col, a, a_col);
    }

    fn vec_znx_sub_negate_assign_default<R, A>(_module: &Module<BE>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        BE: ZnxSubNegateAssign + ZnxNegateAssign,
        R: VecZnxToMut,
        A: VecZnxToRef,
    {
        vec_znx_sub_negate_assign::<R, A, BE>(res, res_col, a, a_col);
    }

    #[allow(clippy::too_many_arguments)]
    fn vec_znx_sub_scalar_default<R, A, B>(
        _module: &Module<BE>,
        res: &mut R,
        res_col: usize,
        a: &A,
        a_col: usize,
        b: &B,
        b_col: usize,
        b_limb: usize,
    ) where
        BE: ZnxSub + ZnxZero,
        R: VecZnxToMut,
        A: ScalarZnxToRef,
        B: VecZnxToRef,
    {
        vec_znx_sub_scalar::<R, A, B, BE>(res, res_col, a, a_col, b, b_col, b_limb);
    }

    fn vec_znx_sub_scalar_assign_default<R, A>(
        _module: &Module<BE>,
        res: &mut R,
        res_col: usize,
        res_limb: usize,
        a: &A,
        a_col: usize,
    ) where
        BE: ZnxSubAssign,
        R: VecZnxToMut,
        A: ScalarZnxToRef,
    {
        vec_znx_sub_scalar_assign::<R, A, BE>(res, res_col, res_limb, a, a_col);
    }

    fn vec_znx_negate_default<R, A>(_module: &Module<BE>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        BE: ZnxNegate + ZnxZero,
        R: VecZnxToMut,
        A: VecZnxToRef,
    {
        vec_znx_negate::<R, A, BE>(res, res_col, a, a_col);
    }

    fn vec_znx_negate_assign_default<R>(_module: &Module<BE>, res: &mut R, res_col: usize)
    where
        BE: ZnxNegateAssign,
        R: VecZnxToMut,
    {
        vec_znx_negate_assign::<R, BE>(res, res_col);
    }

    fn vec_znx_rsh_tmp_bytes_default(module: &Module<BE>) -> usize {
        vec_znx_rsh_tmp_bytes(module.n())
    }

    #[allow(clippy::too_many_arguments)]
    fn vec_znx_rsh_default<R, A>(
        module: &Module<BE>,
        base2k: usize,
        k: usize,
        res: &mut R,
        res_col: usize,
        a: &A,
        a_col: usize,
        scratch: &mut Scratch<BE>,
    ) where
        BE: ZnxZero
            + ZnxCopy
            + ZnxNormalizeFirstStepCarryOnly
            + ZnxNormalizeMiddleStepCarryOnly
            + ZnxNormalizeFirstStep
            + ZnxNormalizeMiddleStep
            + ZnxNormalizeMiddleStepAssign
            + ZnxNormalizeFirstStepAssign
            + ZnxNormalizeFinalStepAssign,
        R: VecZnxToMut,
        A: VecZnxToRef,
    {
        let (carry, _) = <BE as HalScratchDefaults<BE>>::take_slice_default::<i64>(
            scratch,
            vec_znx_rsh_tmp_bytes(module.n()) / size_of::<i64>(),
        );
        vec_znx_rsh::<R, A, BE, true>(base2k, k, res, res_col, a, a_col, carry);
    }

    #[allow(clippy::too_many_arguments)]
    fn vec_znx_rsh_add_into_default<R, A>(
        module: &Module<BE>,
        base2k: usize,
        k: usize,
        res: &mut R,
        res_col: usize,
        a: &A,
        a_col: usize,
        scratch: &mut Scratch<BE>,
    ) where
        BE: ZnxZero
            + ZnxCopy
            + ZnxNormalizeFirstStepCarryOnly
            + ZnxNormalizeMiddleStepCarryOnly
            + ZnxNormalizeFirstStep
            + ZnxNormalizeMiddleStep
            + ZnxNormalizeMiddleStepAssign
            + ZnxNormalizeFirstStepAssign
            + ZnxNormalizeFinalStepAssign,
        R: VecZnxToMut,
        A: VecZnxToRef,
    {
        let (carry, _) = <BE as HalScratchDefaults<BE>>::take_slice_default::<i64>(
            scratch,
            vec_znx_rsh_tmp_bytes(module.n()) / size_of::<i64>(),
        );
        vec_znx_rsh::<R, A, BE, false>(base2k, k, res, res_col, a, a_col, carry);
    }

    fn vec_znx_lsh_tmp_bytes_default(module: &Module<BE>) -> usize {
        vec_znx_lsh_tmp_bytes(module.n())
    }

    #[allow(clippy::too_many_arguments)]
    fn vec_znx_lsh_default<R, A>(
        module: &Module<BE>,
        base2k: usize,
        k: usize,
        res: &mut R,
        res_col: usize,
        a: &A,
        a_col: usize,
        scratch: &mut Scratch<BE>,
    ) where
        BE: ZnxZero
            + ZnxNormalizeFirstStep
            + ZnxNormalizeMiddleStep
            + ZnxCopy
            + ZnxNormalizeFinalStep
            + ZnxNormalizeFirstStepCarryOnly
            + ZnxNormalizeMiddleStepCarryOnly,
        R: VecZnxToMut,
        A: VecZnxToRef,
    {
        let (carry, _) = <BE as HalScratchDefaults<BE>>::take_slice_default::<i64>(
            scratch,
            vec_znx_lsh_tmp_bytes(module.n()) / size_of::<i64>(),
        );
        vec_znx_lsh::<R, A, BE, true>(base2k, k, res, res_col, a, a_col, carry);
    }

    #[allow(clippy::too_many_arguments)]
    fn vec_znx_lsh_add_into_default<R, A>(
        module: &Module<BE>,
        base2k: usize,
        k: usize,
        res: &mut R,
        res_col: usize,
        a: &A,
        a_col: usize,
        scratch: &mut Scratch<BE>,
    ) where
        BE: ZnxZero
            + ZnxNormalizeFirstStep
            + ZnxNormalizeMiddleStep
            + ZnxCopy
            + ZnxNormalizeFinalStep
            + ZnxNormalizeFirstStepCarryOnly
            + ZnxNormalizeMiddleStepCarryOnly,
        R: VecZnxToMut,
        A: VecZnxToRef,
    {
        let (carry, _) = <BE as HalScratchDefaults<BE>>::take_slice_default::<i64>(
            scratch,
            vec_znx_lsh_tmp_bytes(module.n()) / size_of::<i64>(),
        );
        vec_znx_lsh::<R, A, BE, false>(base2k, k, res, res_col, a, a_col, carry);
    }

    #[allow(clippy::too_many_arguments)]
    fn vec_znx_lsh_sub_default<R, A>(
        module: &Module<BE>,
        base2k: usize,
        k: usize,
        res: &mut R,
        res_col: usize,
        a: &A,
        a_col: usize,
        scratch: &mut Scratch<BE>,
    ) where
        BE: ZnxZero
            + ZnxNormalizeFirstStepCarryOnly
            + ZnxNormalizeMiddleStepSub
            + ZnxNormalizeFinalStepSub
            + ZnxNormalizeMiddleStepCarryOnly,
        R: VecZnxToMut,
        A: VecZnxToRef,
    {
        let (carry, _) = <BE as HalScratchDefaults<BE>>::take_slice_default::<i64>(
            scratch,
            vec_znx_lsh_tmp_bytes(module.n()) / size_of::<i64>(),
        );
        vec_znx_lsh_sub::<R, A, BE>(base2k, k, res, res_col, a, a_col, carry);
    }

    #[allow(clippy::too_many_arguments)]
    fn vec_znx_rsh_sub_default<R, A>(
        module: &Module<BE>,
        base2k: usize,
        k: usize,
        res: &mut R,
        res_col: usize,
        a: &A,
        a_col: usize,
        scratch: &mut Scratch<BE>,
    ) where
        BE: ZnxZero
            + ZnxCopy
            + ZnxNormalizeFirstStepCarryOnly
            + ZnxNormalizeMiddleStepCarryOnly
            + ZnxNormalizeMiddleStepSub
            + ZnxNormalizeMiddleStepAssign
            + ZnxNormalizeFirstStepAssign
            + ZnxNormalizeFinalStepAssign,
        R: VecZnxToMut,
        A: VecZnxToRef,
    {
        let (carry, _) = <BE as HalScratchDefaults<BE>>::take_slice_default::<i64>(
            scratch,
            vec_znx_rsh_tmp_bytes(module.n()) / size_of::<i64>(),
        );
        vec_znx_rsh_sub::<R, A, BE>(base2k, k, res, res_col, a, a_col, carry);
    }

    fn vec_znx_rsh_assign_default<R>(
        module: &Module<BE>,
        base2k: usize,
        k: usize,
        res: &mut R,
        res_col: usize,
        scratch: &mut Scratch<BE>,
    ) where
        BE: ZnxZero
            + ZnxCopy
            + ZnxNormalizeFirstStepCarryOnly
            + ZnxNormalizeMiddleStepCarryOnly
            + ZnxNormalizeMiddleStep
            + ZnxNormalizeMiddleStepAssign
            + ZnxNormalizeFirstStepAssign
            + ZnxNormalizeFinalStepAssign,
        R: VecZnxToMut,
    {
        let (carry, _) = <BE as HalScratchDefaults<BE>>::take_slice_default::<i64>(
            scratch,
            vec_znx_rsh_tmp_bytes(module.n()) / size_of::<i64>(),
        );
        vec_znx_rsh_assign::<R, BE>(base2k, k, res, res_col, carry);
    }

    fn vec_znx_lsh_assign_default<R>(
        module: &Module<BE>,
        base2k: usize,
        k: usize,
        res: &mut R,
        res_col: usize,
        scratch: &mut Scratch<BE>,
    ) where
        BE: ZnxZero + ZnxCopy + ZnxNormalizeFirstStepAssign + ZnxNormalizeMiddleStepAssign + ZnxNormalizeFinalStepAssign,
        R: VecZnxToMut,
    {
        let (carry, _) = <BE as HalScratchDefaults<BE>>::take_slice_default::<i64>(
            scratch,
            vec_znx_lsh_tmp_bytes(module.n()) / size_of::<i64>(),
        );
        vec_znx_lsh_assign::<R, BE>(base2k, k, res, res_col, carry);
    }

    fn vec_znx_rotate_default<R, A>(_module: &Module<BE>, p: i64, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        BE: ZnxRotate + ZnxZero,
        R: VecZnxToMut,
        A: VecZnxToRef,
    {
        vec_znx_rotate::<R, A, BE>(p, res, res_col, a, a_col);
    }

    fn vec_znx_rotate_assign_tmp_bytes_default(module: &Module<BE>) -> usize {
        vec_znx_rotate_assign_tmp_bytes(module.n())
    }

    fn vec_znx_rotate_assign_default<R>(module: &Module<BE>, p: i64, res: &mut R, res_col: usize, scratch: &mut Scratch<BE>)
    where
        BE: ZnxRotate + ZnxCopy,
        R: VecZnxToMut,
    {
        let (tmp, _) = <BE as HalScratchDefaults<BE>>::take_slice_default::<i64>(
            scratch,
            vec_znx_rotate_assign_tmp_bytes(module.n()) / size_of::<i64>(),
        );
        vec_znx_rotate_assign::<R, BE>(p, res, res_col, tmp);
    }

    fn vec_znx_automorphism_default<R, A>(_module: &Module<BE>, p: i64, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        BE: ZnxAutomorphism + ZnxZero,
        R: VecZnxToMut,
        A: VecZnxToRef,
    {
        vec_znx_automorphism::<R, A, BE>(p, res, res_col, a, a_col);
    }

    fn vec_znx_automorphism_assign_tmp_bytes_default(module: &Module<BE>) -> usize {
        vec_znx_automorphism_assign_tmp_bytes(module.n())
    }

    fn vec_znx_automorphism_assign_default<R>(module: &Module<BE>, p: i64, res: &mut R, res_col: usize, scratch: &mut Scratch<BE>)
    where
        BE: ZnxAutomorphism + ZnxCopy,
        R: VecZnxToMut,
    {
        let (tmp, _) = <BE as HalScratchDefaults<BE>>::take_slice_default::<i64>(
            scratch,
            vec_znx_automorphism_assign_tmp_bytes(module.n()) / size_of::<i64>(),
        );
        vec_znx_automorphism_assign::<R, BE>(p, res, res_col, tmp);
    }

    fn vec_znx_mul_xp_minus_one_default<R, A>(_module: &Module<BE>, p: i64, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        BE: ZnxRotate + ZnxZero + ZnxSubAssign,
        R: VecZnxToMut,
        A: VecZnxToRef,
    {
        vec_znx_mul_xp_minus_one::<R, A, BE>(p, res, res_col, a, a_col);
    }

    fn vec_znx_mul_xp_minus_one_assign_tmp_bytes_default(module: &Module<BE>) -> usize {
        vec_znx_mul_xp_minus_one_assign_tmp_bytes(module.n())
    }

    fn vec_znx_mul_xp_minus_one_assign_default<R>(
        module: &Module<BE>,
        p: i64,
        res: &mut R,
        res_col: usize,
        scratch: &mut Scratch<BE>,
    ) where
        BE: ZnxRotate + ZnxNegate + ZnxSubNegateAssign,
        R: VecZnxToMut,
    {
        let (tmp, _) = <BE as HalScratchDefaults<BE>>::take_slice_default::<i64>(
            scratch,
            vec_znx_mul_xp_minus_one_assign_tmp_bytes(module.n()) / size_of::<i64>(),
        );
        vec_znx_mul_xp_minus_one_assign::<R, BE>(p, res, res_col, tmp);
    }

    fn vec_znx_split_ring_tmp_bytes_default(module: &Module<BE>) -> usize {
        vec_znx_split_ring_tmp_bytes(module.n())
    }

    fn vec_znx_split_ring_default<R, A>(
        module: &Module<BE>,
        res: &mut [R],
        res_col: usize,
        a: &A,
        a_col: usize,
        scratch: &mut Scratch<BE>,
    ) where
        BE: ZnxSwitchRing + ZnxRotate + ZnxZero,
        R: VecZnxToMut,
        A: VecZnxToRef,
    {
        let (tmp, _) = <BE as HalScratchDefaults<BE>>::take_slice_default::<i64>(
            scratch,
            vec_znx_split_ring_tmp_bytes(module.n()) / size_of::<i64>(),
        );
        vec_znx_split_ring::<R, A, BE>(res, res_col, a, a_col, tmp);
    }

    fn vec_znx_merge_rings_tmp_bytes_default(module: &Module<BE>) -> usize {
        vec_znx_merge_rings_tmp_bytes(module.n())
    }

    fn vec_znx_merge_rings_default<R, A>(
        module: &Module<BE>,
        res: &mut R,
        res_col: usize,
        a: &[A],
        a_col: usize,
        scratch: &mut Scratch<BE>,
    ) where
        BE: ZnxCopy + ZnxSwitchRing + ZnxRotate + ZnxZero,
        R: VecZnxToMut,
        A: VecZnxToRef,
    {
        let (tmp, _) = <BE as HalScratchDefaults<BE>>::take_slice_default::<i64>(
            scratch,
            vec_znx_merge_rings_tmp_bytes(module.n()) / size_of::<i64>(),
        );
        vec_znx_merge_rings::<R, A, BE>(res, res_col, a, a_col, tmp);
    }

    fn vec_znx_switch_ring_default<R, A>(_module: &Module<BE>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        BE: ZnxCopy + ZnxSwitchRing + ZnxZero,
        R: VecZnxToMut,
        A: VecZnxToRef,
    {
        vec_znx_switch_ring::<R, A, BE>(res, res_col, a, a_col);
    }

    fn vec_znx_copy_default<R, A>(_module: &Module<BE>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        BE: ZnxCopy + ZnxZero,
        R: VecZnxToMut,
        A: VecZnxToRef,
    {
        vec_znx_copy::<R, A, BE>(res, res_col, a, a_col);
    }

    fn vec_znx_fill_uniform_default<R>(_module: &Module<BE>, base2k: usize, res: &mut R, res_col: usize, source: &mut Source)
    where
        R: VecZnxToMut,
    {
        vec_znx_fill_uniform_ref(base2k, res, res_col, source);
    }

    fn vec_znx_fill_normal_default<R>(
        _module: &Module<BE>,
        res_base2k: usize,
        res: &mut R,
        res_col: usize,
        noise_infos: NoiseInfos,
        source: &mut Source,
    ) where
        R: VecZnxToMut,
    {
        vec_znx_fill_normal_ref(res_base2k, res, res_col, noise_infos, source);
    }

    fn vec_znx_add_normal_default<R>(
        _module: &Module<BE>,
        res_base2k: usize,
        res: &mut R,
        res_col: usize,
        noise_infos: NoiseInfos,
        source: &mut Source,
    ) where
        R: VecZnxToMut,
    {
        vec_znx_add_normal_ref(res_base2k, res, res_col, noise_infos, source);
    }
}

impl<BE: Backend> HalVecZnxDefaults<BE> for BE {}
