#![allow(clippy::too_many_arguments)]

use crate::layouts::{Backend, Module, ScalarZnxBackendRef, ScratchArena, VecZnxBackendMut, VecZnxBackendRef};

/// Module construction extension point.
///
/// # Safety
/// Implementations must return a module handle that is valid for the backend
/// and ring degree, and uphold the backend safety contract.
pub unsafe trait HalModuleImpl<BE: Backend>: Backend {
    #[allow(clippy::new_ret_no_self)]
    fn new(n: u64) -> Module<BE>;
}

/// Coefficient-domain `VecZnx` extension point.
///
/// # Safety
/// Implementations must uphold the backend safety contract for layout access,
/// aliasing, scratch usage, and arithmetic correctness.
pub unsafe trait HalVecZnxImpl<BE: Backend>: Backend {
    fn vec_znx_zero(module: &Module<BE>, res: &mut VecZnxBackendMut<'_, BE>, res_col: usize);

    fn vec_znx_normalize_tmp_bytes(module: &Module<BE>) -> usize;

    #[allow(clippy::too_many_arguments)]
    fn vec_znx_normalize(
        module: &Module<BE>,
        res: &mut VecZnxBackendMut<'_, BE>,
        res_base2k: usize,
        res_k: usize,
        res_offset: i64,
        res_col: usize,
        a: &VecZnxBackendRef<'_, BE>,
        a_base2k: usize,
        a_col: usize,
        scratch: &mut ScratchArena<'_, BE>,
    );

    fn vec_znx_normalize_assign(
        module: &Module<BE>,
        base2k: usize,
        k: usize,
        a_offset: i64,
        a: &mut VecZnxBackendMut<'_, BE>,
        a_col: usize,
        scratch: &mut ScratchArena<'_, BE>,
    );

    fn vec_znx_add(
        module: &Module<BE>,
        res: &mut VecZnxBackendMut<'_, BE>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, BE>,
        a_col: usize,
        b: &VecZnxBackendRef<'_, BE>,
        b_col: usize,
    );

    fn vec_znx_add_assign(
        module: &Module<BE>,
        res: &mut VecZnxBackendMut<'_, BE>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, BE>,
        a_col: usize,
    );

    /// `res[res_col][res_limb] += a[a_col]`, the scalar seen as a one-limb
    /// `VecZnx`. Default body: `vec_znx_add_scalar_assign_derived`, an
    /// `add_assign` on the one-limb window of `res`.
    fn vec_znx_add_scalar_assign(
        module: &Module<BE>,
        res: &mut VecZnxBackendMut<'_, BE>,
        res_col: usize,
        res_limb: usize,
        a: &ScalarZnxBackendRef<'_, BE>,
        a_col: usize,
    ) {
        crate::oep::vec_znx_add_scalar_assign_derived::<Self, BE>(module, res, res_col, res_limb, a, a_col)
    }

    fn vec_znx_sub(
        module: &Module<BE>,
        res: &mut VecZnxBackendMut<'_, BE>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, BE>,
        a_col: usize,
        b: &VecZnxBackendRef<'_, BE>,
        b_col: usize,
    );

    fn vec_znx_sub_assign(
        module: &Module<BE>,
        res: &mut VecZnxBackendMut<'_, BE>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, BE>,
        a_col: usize,
    );

    fn vec_znx_sub_negate_assign(
        module: &Module<BE>,
        res: &mut VecZnxBackendMut<'_, BE>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, BE>,
        a_col: usize,
    );

    fn vec_znx_negate(
        module: &Module<BE>,
        res: &mut VecZnxBackendMut<'_, BE>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, BE>,
        a_col: usize,
    );

    fn vec_znx_negate_assign(module: &Module<BE>, a: &mut VecZnxBackendMut<'_, BE>, a_col: usize);

    /// Returns scratch bytes for the whole right-shift family, [`Self::vec_znx_rsh`],
    /// [`Self::vec_znx_rsh_add`], [`Self::vec_znx_rsh_sub`] and
    /// [`Self::vec_znx_rsh_assign`], on a destination of `res_size` limbs.
    /// Default body: `vec_znx_rsh_tmp_bytes_derived`.
    ///
    /// Family rule: the value must cover the `res_size`-limb temporary the `_add`
    /// / `_sub` / `_assign` default bodies carve, not only the carry
    /// `vec_znx_rsh` itself needs. A backend that overrides some but not all
    /// bodies of the family must therefore either leave this on the default or
    /// return a value large enough for the ones it did not override.
    fn vec_znx_rsh_tmp_bytes(module: &Module<BE>, res_size: usize) -> usize {
        crate::oep::vec_znx_rsh_tmp_bytes_derived::<Self, BE>(module, res_size)
    }

    /// `res[res_col] = a[a_col] / 2^k` at the same radix. Default body:
    /// `vec_znx_rsh_derived`, a `vec_znx_normalize` with `offset = -k`; it uses
    /// only the normalization carry out of [`Self::vec_znx_rsh_tmp_bytes`].
    fn vec_znx_rsh(
        module: &Module<BE>,
        base2k: usize,
        k: usize,
        res: &mut VecZnxBackendMut<'_, BE>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, BE>,
        a_col: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) {
        crate::oep::vec_znx_rsh_derived::<Self, BE>(module, base2k, k, res, res_col, a, a_col, scratch)
    }

    /// `res[res_col] += a[a_col] / 2^k`. Default body: `vec_znx_rsh_add_derived`,
    /// a `vec_znx_rsh` into a `res.size()`-limb temporary carved out of
    /// [`Self::vec_znx_rsh_tmp_bytes`], then `vec_znx_add_assign`.
    fn vec_znx_rsh_add(
        module: &Module<BE>,
        base2k: usize,
        k: usize,
        res: &mut VecZnxBackendMut<'_, BE>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, BE>,
        a_col: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) {
        crate::oep::vec_znx_rsh_add_derived::<Self, BE>(module, base2k, k, res, res_col, a, a_col, scratch)
    }

    /// Returns scratch bytes for the whole left-shift family, [`Self::vec_znx_lsh`],
    /// [`Self::vec_znx_lsh_add`], [`Self::vec_znx_lsh_sub`] and
    /// [`Self::vec_znx_lsh_assign`], on a destination of `res_size` limbs.
    /// Default body: `vec_znx_lsh_tmp_bytes_derived`.
    ///
    /// Family rule: the value must cover the `res_size`-limb temporary the `_add`
    /// / `_sub` / `_assign` default bodies carve, not only the carry
    /// `vec_znx_lsh` itself needs. A backend that overrides some but not all
    /// bodies of the family must therefore either leave this on the default or
    /// return a value large enough for the ones it did not override.
    /// `poulpy-cpu-ref` does exactly that: it overrides `vec_znx_lsh_assign` and
    /// leaves this `_tmp_bytes` on the default.
    fn vec_znx_lsh_tmp_bytes(module: &Module<BE>, res_size: usize) -> usize {
        crate::oep::vec_znx_lsh_tmp_bytes_derived::<Self, BE>(module, res_size)
    }

    /// `res[res_col] = a[a_col] * 2^k` at the same radix. Default body:
    /// `vec_znx_lsh_derived`, a `vec_znx_normalize` with `offset = +k`; it uses
    /// only the normalization carry out of [`Self::vec_znx_lsh_tmp_bytes`].
    fn vec_znx_lsh(
        module: &Module<BE>,
        base2k: usize,
        k: usize,
        res: &mut VecZnxBackendMut<'_, BE>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, BE>,
        a_col: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) {
        crate::oep::vec_znx_lsh_derived::<Self, BE>(module, base2k, k, res, res_col, a, a_col, scratch)
    }

    /// `res[res_col] += a[a_col] * 2^k`. Default body: `vec_znx_lsh_add_derived`,
    /// a `vec_znx_lsh` into a `res.size()`-limb temporary carved out of
    /// [`Self::vec_znx_lsh_tmp_bytes`], then `vec_znx_add_assign`.
    fn vec_znx_lsh_add(
        module: &Module<BE>,
        base2k: usize,
        k: usize,
        res: &mut VecZnxBackendMut<'_, BE>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, BE>,
        a_col: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) {
        crate::oep::vec_znx_lsh_add_derived::<Self, BE>(module, base2k, k, res, res_col, a, a_col, scratch)
    }

    /// `res[res_col] -= a[a_col] * 2^k`. Default body: `vec_znx_lsh_sub_derived`,
    /// a `vec_znx_lsh` into a `res.size()`-limb temporary carved out of
    /// [`Self::vec_znx_lsh_tmp_bytes`], then `vec_znx_sub_assign`.
    fn vec_znx_lsh_sub(
        module: &Module<BE>,
        base2k: usize,
        k: usize,
        res: &mut VecZnxBackendMut<'_, BE>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, BE>,
        a_col: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) {
        crate::oep::vec_znx_lsh_sub_derived::<Self, BE>(module, base2k, k, res, res_col, a, a_col, scratch)
    }

    /// `res[res_col] -= a[a_col] / 2^k`. Default body: `vec_znx_rsh_sub_derived`,
    /// a `vec_znx_rsh` into a `res.size()`-limb temporary carved out of
    /// [`Self::vec_znx_rsh_tmp_bytes`], then `vec_znx_sub_assign`.
    fn vec_znx_rsh_sub(
        module: &Module<BE>,
        base2k: usize,
        k: usize,
        res: &mut VecZnxBackendMut<'_, BE>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, BE>,
        a_col: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) {
        crate::oep::vec_znx_rsh_sub_derived::<Self, BE>(module, base2k, k, res, res_col, a, a_col, scratch)
    }

    /// `a[a_col] /= 2^k`, in place on the one selected column. Default body:
    /// `vec_znx_rsh_assign_derived`, a `vec_znx_rsh` into an `a.size()`-limb
    /// temporary carved out of [`Self::vec_znx_rsh_tmp_bytes`], then a copy back.
    fn vec_znx_rsh_assign(
        module: &Module<BE>,
        base2k: usize,
        k: usize,
        a: &mut VecZnxBackendMut<'_, BE>,
        a_col: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) {
        crate::oep::vec_znx_rsh_assign_derived::<Self, BE>(module, base2k, k, a, a_col, scratch)
    }

    /// `a[a_col] *= 2^k`, in place on the one selected column. Default body:
    /// `vec_znx_lsh_assign_derived`, a `vec_znx_lsh` into an `a.size()`-limb
    /// temporary carved out of [`Self::vec_znx_lsh_tmp_bytes`], then a copy back.
    fn vec_znx_lsh_assign(
        module: &Module<BE>,
        base2k: usize,
        k: usize,
        a: &mut VecZnxBackendMut<'_, BE>,
        a_col: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) {
        crate::oep::vec_znx_lsh_assign_derived::<Self, BE>(module, base2k, k, a, a_col, scratch)
    }

    fn vec_znx_rotate(
        module: &Module<BE>,
        k: i64,
        res: &mut VecZnxBackendMut<'_, BE>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, BE>,
        a_col: usize,
    );

    fn vec_znx_rotate_assign_tmp_bytes(module: &Module<BE>) -> usize;

    fn vec_znx_rotate_assign(
        module: &Module<BE>,
        k: i64,
        a: &mut VecZnxBackendMut<'_, BE>,
        a_col: usize,
        scratch: &mut ScratchArena<'_, BE>,
    );

    fn vec_znx_automorphism(
        module: &Module<BE>,
        k: i64,
        res: &mut VecZnxBackendMut<'_, BE>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, BE>,
        a_col: usize,
    );

    fn vec_znx_automorphism_assign_tmp_bytes(module: &Module<BE>) -> usize;

    fn vec_znx_automorphism_assign(
        module: &Module<BE>,
        k: i64,
        res: &mut VecZnxBackendMut<'_, BE>,
        res_col: usize,
        scratch: &mut ScratchArena<'_, BE>,
    );

    /// `res[res_col] = (X^k - 1) * a[a_col]`. Default body:
    /// `vec_znx_mul_xp_minus_one_derived`, a `vec_znx_rotate` then a
    /// `vec_znx_sub_assign` of the unrotated value. Takes no scratch.
    fn vec_znx_mul_xp_minus_one(
        module: &Module<BE>,
        k: i64,
        res: &mut VecZnxBackendMut<'_, BE>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, BE>,
        a_col: usize,
    ) {
        crate::oep::vec_znx_mul_xp_minus_one_derived::<Self, BE>(module, k, res, res_col, a, a_col)
    }

    /// Returns scratch bytes required for
    /// [`Self::vec_znx_mul_xp_minus_one_assign`] on a `size`-limb operand.
    /// Default body: `vec_znx_mul_xp_minus_one_assign_tmp_bytes_derived`, one
    /// `size`-limb `VecZnx` for the rotated copy.
    fn vec_znx_mul_xp_minus_one_assign_tmp_bytes(module: &Module<BE>, size: usize) -> usize {
        crate::oep::vec_znx_mul_xp_minus_one_assign_tmp_bytes_derived::<Self, BE>(module, size)
    }

    /// `res[res_col] = (X^k - 1) * res[res_col]`. Default body:
    /// `vec_znx_mul_xp_minus_one_assign_derived`, which carves the full-width
    /// rotated copy out of [`Self::vec_znx_mul_xp_minus_one_assign_tmp_bytes`]
    /// and copies it back.
    fn vec_znx_mul_xp_minus_one_assign(
        module: &Module<BE>,
        k: i64,
        res: &mut VecZnxBackendMut<'_, BE>,
        res_col: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) {
        crate::oep::vec_znx_mul_xp_minus_one_assign_derived::<Self, BE>(module, k, res, res_col, scratch)
    }

    fn vec_znx_switch_ring(
        module: &Module<BE>,
        res: &mut VecZnxBackendMut<'_, BE>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, BE>,
        a_col: usize,
    );

    fn vec_znx_copy(
        module: &Module<BE>,
        res: &mut VecZnxBackendMut<'_, BE>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, BE>,
        a_col: usize,
    );

    fn vec_znx_fill_uniform(
        module: &Module<BE>,
        base2k: usize,
        k: usize,
        res: &mut VecZnxBackendMut<'_, BE>,
        res_col: usize,
        seed: [u8; 32],
    );
}

/// Big-coefficient `VecZnxBig` extension point.
///
/// # Safety
/// Implementations must uphold the backend safety contract for backend-native
/// accumulator layouts and arithmetic correctness.
pub unsafe trait HalVecZnxBigImpl<BE: Backend>: Backend {
    fn vec_znx_big_from_small(
        res: &mut crate::layouts::VecZnxBigBackendMut<'_, BE>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, BE>,
        a_col: usize,
    );

    fn vec_znx_big_add(
        module: &Module<BE>,
        res: &mut crate::layouts::VecZnxBigBackendMut<'_, BE>,
        res_col: usize,
        a: &crate::layouts::VecZnxBigBackendRef<'_, BE>,
        a_col: usize,
        b: &crate::layouts::VecZnxBigBackendRef<'_, BE>,
        b_col: usize,
    );

    fn vec_znx_big_add_assign(
        module: &Module<BE>,
        res: &mut crate::layouts::VecZnxBigBackendMut<'_, BE>,
        res_col: usize,
        a: &crate::layouts::VecZnxBigBackendRef<'_, BE>,
        a_col: usize,
    );

    /// Default body: [`crate::oep::vec_znx_big_add_small_derived`]. Over the
    /// whole of `res`, limbs only `b` reaches keep `b`, limbs only `a` reaches
    /// keep `a`, limbs beyond both operands are zero.
    fn vec_znx_big_add_small(
        module: &Module<BE>,
        res: &mut crate::layouts::VecZnxBigBackendMut<'_, BE>,
        res_col: usize,
        a: &crate::layouts::VecZnxBigBackendRef<'_, BE>,
        a_col: usize,
        b: &VecZnxBackendRef<'_, BE>,
        b_col: usize,
    ) {
        crate::oep::vec_znx_big_add_small_derived::<Self, BE>(module, res, res_col, a, a_col, b, b_col)
    }

    /// Required, not derived: the composition would need a `VecZnxBig`
    /// temporary and this form takes no scratch (spec section 4.2).
    fn vec_znx_big_add_small_assign(
        module: &Module<BE>,
        res: &mut crate::layouts::VecZnxBigBackendMut<'_, BE>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, BE>,
        a_col: usize,
    );

    fn vec_znx_big_sub(
        module: &Module<BE>,
        res: &mut crate::layouts::VecZnxBigBackendMut<'_, BE>,
        res_col: usize,
        a: &crate::layouts::VecZnxBigBackendRef<'_, BE>,
        a_col: usize,
        b: &crate::layouts::VecZnxBigBackendRef<'_, BE>,
        b_col: usize,
    );

    fn vec_znx_big_sub_assign(
        module: &Module<BE>,
        res: &mut crate::layouts::VecZnxBigBackendMut<'_, BE>,
        res_col: usize,
        a: &crate::layouts::VecZnxBigBackendRef<'_, BE>,
        a_col: usize,
    );

    fn vec_znx_big_sub_negate_assign(
        module: &Module<BE>,
        res: &mut crate::layouts::VecZnxBigBackendMut<'_, BE>,
        res_col: usize,
        a: &crate::layouts::VecZnxBigBackendRef<'_, BE>,
        a_col: usize,
    );

    /// Default body: [`crate::oep::vec_znx_big_sub_small_a_derived`]. Over the
    /// whole of `res`, limbs only `a` reaches keep `a`, limbs beyond
    /// `a.size()` that `b` reaches keep `-b`, limbs beyond both operands are
    /// zero.
    fn vec_znx_big_sub_small_a(
        module: &Module<BE>,
        res: &mut crate::layouts::VecZnxBigBackendMut<'_, BE>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, BE>,
        a_col: usize,
        b: &crate::layouts::VecZnxBigBackendRef<'_, BE>,
        b_col: usize,
    ) {
        crate::oep::vec_znx_big_sub_small_a_derived::<Self, BE>(module, res, res_col, a, a_col, b, b_col)
    }

    /// Required, not derived: the composition would need a `VecZnxBig`
    /// temporary and this form takes no scratch (spec section 4.2).
    fn vec_znx_big_sub_small_assign(
        module: &Module<BE>,
        res: &mut crate::layouts::VecZnxBigBackendMut<'_, BE>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, BE>,
        a_col: usize,
    );

    /// Default body: [`crate::oep::vec_znx_big_sub_small_b_derived`]. Over the
    /// whole of `res`, limbs only `a` reaches keep `a`, limbs beyond
    /// `a.size()` that `b` reaches keep `-b`, limbs beyond both operands are
    /// zero.
    fn vec_znx_big_sub_small_b(
        module: &Module<BE>,
        res: &mut crate::layouts::VecZnxBigBackendMut<'_, BE>,
        res_col: usize,
        a: &crate::layouts::VecZnxBigBackendRef<'_, BE>,
        a_col: usize,
        b: &VecZnxBackendRef<'_, BE>,
        b_col: usize,
    ) {
        crate::oep::vec_znx_big_sub_small_b_derived::<Self, BE>(module, res, res_col, a, a_col, b, b_col)
    }

    /// Required, not derived: the composition would need a `VecZnxBig`
    /// temporary and this form takes no scratch (spec section 4.2).
    fn vec_znx_big_sub_small_negate_assign(
        module: &Module<BE>,
        res: &mut crate::layouts::VecZnxBigBackendMut<'_, BE>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, BE>,
        a_col: usize,
    );

    fn vec_znx_big_inner_sum(
        module: &Module<BE>,
        res: &mut crate::layouts::VecZnxBigBackendMut<'_, BE>,
        res_col: usize,
        res_coeff: usize,
        a: &crate::layouts::VecZnxBigBackendRef<'_, BE>,
        a_col: usize,
    );

    fn vec_znx_big_col_weighted_sum(
        module: &Module<BE>,
        res: &mut crate::layouts::VecZnxBigBackendMut<'_, BE>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, BE>,
        weights: &ScalarZnxBackendRef<'_, BE>,
        weights_col: usize,
        cols: usize,
        coeffs: usize,
    );

    fn vec_znx_scalar_product(
        module: &Module<BE>,
        res: &mut crate::layouts::VecZnxBigBackendMut<'_, BE>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, BE>,
        a_col: usize,
        b: &ScalarZnxBackendRef<'_, BE>,
        b_col: usize,
    );

    fn vec_znx_big_negate(
        module: &Module<BE>,
        res: &mut crate::layouts::VecZnxBigBackendMut<'_, BE>,
        res_col: usize,
        a: &crate::layouts::VecZnxBigBackendRef<'_, BE>,
        a_col: usize,
    );

    fn vec_znx_big_negate_assign(module: &Module<BE>, a: &mut crate::layouts::VecZnxBigBackendMut<'_, BE>, a_col: usize);

    fn vec_znx_big_normalize_tmp_bytes(module: &Module<BE>) -> usize;

    #[allow(clippy::too_many_arguments)]
    fn vec_znx_big_normalize(
        module: &Module<BE>,
        res: &mut VecZnxBackendMut<'_, BE>,
        res_base2k: usize,
        res_k: usize,
        res_offset: i64,
        res_col: usize,
        a: &crate::layouts::VecZnxBigBackendRef<'_, BE>,
        a_base2k: usize,
        a_col: usize,
        scratch: &mut ScratchArena<'_, BE>,
    );

    fn vec_znx_big_automorphism(
        module: &Module<BE>,
        k: i64,
        res: &mut crate::layouts::VecZnxBigBackendMut<'_, BE>,
        res_col: usize,
        a: &crate::layouts::VecZnxBigBackendRef<'_, BE>,
        a_col: usize,
    );

    fn vec_znx_big_automorphism_assign_tmp_bytes(module: &Module<BE>) -> usize;

    fn vec_znx_big_automorphism_assign(
        module: &Module<BE>,
        k: i64,
        a: &mut crate::layouts::VecZnxBigBackendMut<'_, BE>,
        a_col: usize,
        scratch: &mut ScratchArena<'_, BE>,
    );
}

/// Prepared / DFT-domain `VecZnxDft` extension point.
///
/// # Safety
/// Implementations must uphold the backend safety contract for prepared-domain
/// layouts, transforms, and arithmetic correctness.
pub unsafe trait HalVecZnxDftImpl<BE: Backend>: Backend + HalVecZnxBigImpl<BE> {
    fn vec_znx_dft_apply(
        module: &Module<BE>,
        step: usize,
        offset: usize,
        res: &mut crate::layouts::VecZnxDftBackendMut<'_, BE>,
        res_col: usize,
        a: &crate::layouts::VecZnxBackendRef<'_, BE>,
        a_col: usize,
    );

    fn vec_znx_idft_apply_tmp_bytes(module: &Module<BE>) -> usize;

    fn vec_znx_idft_apply(
        module: &Module<BE>,
        res: &mut crate::layouts::VecZnxBigBackendMut<'_, BE>,
        res_col: usize,
        a: &crate::layouts::VecZnxDftBackendRef<'_, BE>,
        a_col: usize,
        scratch: &mut ScratchArena<'_, BE>,
    );

    /// Required, not derived: the composition is `vec_znx_idft_apply(res, a)`,
    /// which needs `vec_znx_idft_apply_tmp_bytes` scratch this signature does
    /// not carry (spec section 4.3, PR4 deviation).
    fn vec_znx_idft_apply_tmpa(
        module: &Module<BE>,
        res: &mut crate::layouts::VecZnxBigBackendMut<'_, BE>,
        res_col: usize,
        a: &mut crate::layouts::VecZnxDftBackendMut<'_, BE>,
        a_col: usize,
    );

    /// Returns scratch bytes required for
    /// [`Self::vec_znx_idft_normalize_consume`]. Default body:
    /// `vec_znx_idft_normalize_consume_tmp_bytes_derived`, the `a_size`-limb
    /// `VecZnxBig` the inverse transform lands in plus the big normalization's
    /// own carry.
    fn vec_znx_idft_normalize_consume_tmp_bytes(module: &Module<BE>, res_size: usize, a_size: usize) -> usize {
        crate::oep::vec_znx_idft_normalize_consume_tmp_bytes_derived::<Self, BE>(module, res_size, a_size)
    }

    /// `res[res_col] = normalize(idft(a[a_col]) + addend)`, clobbering `a[a_col]`.
    #[allow(clippy::too_many_arguments)]
    fn vec_znx_idft_normalize_consume(
        module: &Module<BE>,
        res: &mut crate::layouts::VecZnxBackendMut<'_, BE>,
        res_base2k: usize,
        res_k: usize,
        res_col: usize,
        a: &mut crate::layouts::VecZnxDftBackendMut<'_, BE>,
        a_col: usize,
        a_base2k: usize,
        addend: Option<(&crate::layouts::VecZnxBackendRef<'_, BE>, usize)>,
        scratch: &mut ScratchArena<'_, BE>,
    ) {
        crate::oep::vec_znx_idft_normalize_consume_derived::<Self, BE>(
            module, res, res_base2k, res_k, res_col, a, a_col, a_base2k, addend, scratch,
        )
    }

    fn vec_znx_dft_add(
        module: &Module<BE>,
        res: &mut crate::layouts::VecZnxDftBackendMut<'_, BE>,
        res_col: usize,
        a: &crate::layouts::VecZnxDftBackendRef<'_, BE>,
        a_col: usize,
        b: &crate::layouts::VecZnxDftBackendRef<'_, BE>,
        b_col: usize,
    );

    fn vec_znx_dft_add_assign(
        module: &Module<BE>,
        res: &mut crate::layouts::VecZnxDftBackendMut<'_, BE>,
        res_col: usize,
        a: &crate::layouts::VecZnxDftBackendRef<'_, BE>,
        a_col: usize,
    );

    fn vec_znx_dft_sub(
        module: &Module<BE>,
        res: &mut crate::layouts::VecZnxDftBackendMut<'_, BE>,
        res_col: usize,
        a: &crate::layouts::VecZnxDftBackendRef<'_, BE>,
        a_col: usize,
        b: &crate::layouts::VecZnxDftBackendRef<'_, BE>,
        b_col: usize,
    );

    fn vec_znx_dft_sub_assign(
        module: &Module<BE>,
        res: &mut crate::layouts::VecZnxDftBackendMut<'_, BE>,
        res_col: usize,
        a: &crate::layouts::VecZnxDftBackendRef<'_, BE>,
        a_col: usize,
    );

    fn vec_znx_dft_sub_negate_assign(
        module: &Module<BE>,
        res: &mut crate::layouts::VecZnxDftBackendMut<'_, BE>,
        res_col: usize,
        a: &crate::layouts::VecZnxDftBackendRef<'_, BE>,
        a_col: usize,
    );

    fn vec_znx_dft_copy(
        module: &Module<BE>,
        step: usize,
        offset: usize,
        res: &mut crate::layouts::VecZnxDftBackendMut<'_, BE>,
        res_col: usize,
        a: &crate::layouts::VecZnxDftBackendRef<'_, BE>,
        a_col: usize,
    );

    fn vec_znx_dft_zero(module: &Module<BE>, res: &mut crate::layouts::VecZnxDftBackendMut<'_, BE>, res_col: usize);

    /// Backend-specific automorphism plan (e.g. a `Fft64AutomorphismPlan`
    /// for FFT64 backends, a pure-permutation plan for NTT backends).
    type AutomorphismPlan: Send + Sync;

    fn vec_znx_dft_automorphism_plan(module: &Module<BE>, p: i64) -> Self::AutomorphismPlan;

    fn vec_znx_dft_automorphism_with_plan(
        module: &Module<BE>,
        plan: &Self::AutomorphismPlan,
        res: &mut crate::layouts::VecZnxDftBackendMut<'_, BE>,
        res_col: usize,
        a: &crate::layouts::VecZnxDftBackendRef<'_, BE>,
        a_col: usize,
    );

    /// Returns scratch bytes required for
    /// [`Self::vec_znx_dft_automorphism_add_with_plan`]. Default body:
    /// `vec_znx_dft_automorphism_add_with_plan_tmp_bytes_derived`, one
    /// `min(res_size, a_size)`-limb `VecZnxDft` for the rotated operand.
    fn vec_znx_dft_automorphism_add_with_plan_tmp_bytes(module: &Module<BE>, res_size: usize, a_size: usize) -> usize {
        crate::oep::vec_znx_dft_automorphism_add_with_plan_tmp_bytes_derived::<Self, BE>(module, res_size, a_size)
    }

    /// `res[res_col] += automorphism(a[a_col])` over `min(res.size(), a.size())` limbs;
    /// res limbs beyond that are left untouched.
    ///
    /// `scratch` must hold at least
    /// [`Self::vec_znx_dft_automorphism_add_with_plan_tmp_bytes`] on the same
    /// sizes: the default body carves one `min(res.size(), a.size())`-limb,
    /// one-column `VecZnxDft` for the rotated operand.
    #[allow(clippy::too_many_arguments)]
    fn vec_znx_dft_automorphism_add_with_plan(
        module: &Module<BE>,
        plan: &Self::AutomorphismPlan,
        res: &mut crate::layouts::VecZnxDftBackendMut<'_, BE>,
        res_col: usize,
        a: &crate::layouts::VecZnxDftBackendRef<'_, BE>,
        a_col: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) {
        crate::oep::vec_znx_dft_automorphism_add_with_plan_derived::<Self, BE>(module, plan, res, res_col, a, a_col, scratch)
    }
}

/// Scalar-vector product family extension point.
///
/// # Safety
/// Implementations must uphold the backend safety contract for prepared
/// polynomial layouts and arithmetic correctness.
pub unsafe trait HalSvpImpl<BE: Backend>: Backend + HalVecZnxDftImpl<BE> {
    fn svp_prepare(
        module: &Module<BE>,
        res: &mut crate::layouts::SvpPPolBackendMut<'_, BE>,
        res_col: usize,
        a: &ScalarZnxBackendRef<'_, BE>,
        a_col: usize,
    );

    fn svp_ppol_copy(
        module: &Module<BE>,
        res: &mut crate::layouts::SvpPPolBackendMut<'_, BE>,
        res_col: usize,
        a: &crate::layouts::SvpPPolBackendRef<'_, BE>,
        a_col: usize,
    );

    /// Returns scratch bytes required for [`Self::svp_apply_dft`]. Default body:
    /// `svp_apply_dft_tmp_bytes_derived`, one `b_size`-limb `VecZnxDft` for the
    /// transformed right operand.
    fn svp_apply_dft_tmp_bytes(module: &Module<BE>, b_size: usize) -> usize {
        crate::oep::svp_apply_dft_tmp_bytes_derived::<Self, BE>(module, b_size)
    }

    /// `res[res_col] = a[a_col] * dft(b[b_col])`. Limbs of `res` beyond
    /// `min(res.size(), b.size())` are zeroed.
    ///
    /// `scratch` must hold at least [`Self::svp_apply_dft_tmp_bytes`] on
    /// `b.size()`: the default body carves one `b.size()`-limb, one-column
    /// `VecZnxDft` for the transformed right operand.
    #[allow(clippy::too_many_arguments)]
    fn svp_apply_dft(
        module: &Module<BE>,
        res: &mut crate::layouts::VecZnxDftBackendMut<'_, BE>,
        res_col: usize,
        a: &crate::layouts::SvpPPolBackendRef<'_, BE>,
        a_col: usize,
        b: &crate::layouts::VecZnxBackendRef<'_, BE>,
        b_col: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) {
        crate::oep::svp_apply_dft_derived::<Self, BE>(module, res, res_col, a, a_col, b, b_col, scratch)
    }

    fn svp_apply_dft_to_dft(
        module: &Module<BE>,
        res: &mut crate::layouts::VecZnxDftBackendMut<'_, BE>,
        res_col: usize,
        a: &crate::layouts::SvpPPolBackendRef<'_, BE>,
        a_col: usize,
        b: &crate::layouts::VecZnxDftBackendRef<'_, BE>,
        b_col: usize,
    );

    /// Required, not derived: a default body would need a `VecZnxDft` temporary
    /// the scratch-free signature does not carry (spec section 4.2).
    fn svp_apply_dft_to_dft_assign(
        module: &Module<BE>,
        res: &mut crate::layouts::VecZnxDftBackendMut<'_, BE>,
        res_col: usize,
        a: &crate::layouts::SvpPPolBackendRef<'_, BE>,
        a_col: usize,
    );
}

/// Vector-matrix product family extension point.
///
/// # Safety
/// Implementations must uphold the backend safety contract for prepared matrix
/// layouts, scratch usage, and arithmetic correctness.
pub unsafe trait HalVmpImpl<BE: Backend>: Backend + HalVecZnxDftImpl<BE> {
    fn vmp_prepare_tmp_bytes(module: &Module<BE>, rows: usize, cols_in: usize, cols_out: usize, size: usize) -> usize;

    fn vmp_prepare(
        module: &Module<BE>,
        res: &mut crate::layouts::VmpPMatBackendMut<'_, BE>,
        a: &crate::layouts::MatZnxBackendRef<'_, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    );

    /// Returns scratch bytes required for [`Self::vmp_apply_dft`]. Default body:
    /// `vmp_apply_dft_tmp_bytes_derived`, one `min(a_size, b_rows)`-limb
    /// `VecZnxDft` for the transformed input plus whatever
    /// [`Self::vmp_apply_dft_to_dft`] needs.
    #[allow(clippy::too_many_arguments)]
    fn vmp_apply_dft_tmp_bytes(
        module: &Module<BE>,
        res_size: usize,
        a_size: usize,
        b_rows: usize,
        b_cols_in: usize,
        b_cols_out: usize,
        b_size: usize,
    ) -> usize {
        crate::oep::vmp_apply_dft_tmp_bytes_derived::<Self, BE>(module, res_size, a_size, b_rows, b_cols_in, b_cols_out, b_size)
    }

    /// `res = dft(a) * pmat`. Default body: `vmp_apply_dft_derived`, which carves
    /// the transformed input out of [`Self::vmp_apply_dft_tmp_bytes`], aligns
    /// `a`'s columns with `pmat.cols_in()`, then applies in the DFT domain.
    fn vmp_apply_dft<R>(
        module: &Module<BE>,
        res: &mut R,
        a: &crate::layouts::VecZnxBackendRef<'_, BE>,
        b: &crate::layouts::VmpPMatBackendRef<'_, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: crate::layouts::VecZnxDftToBackendMut<BE>,
    {
        crate::oep::vmp_apply_dft_derived::<Self, BE, R>(module, res, a, b, scratch)
    }

    #[allow(clippy::too_many_arguments)]
    fn vmp_apply_dft_to_dft_tmp_bytes(
        module: &Module<BE>,
        res_size: usize,
        a_size: usize,
        b_rows: usize,
        b_cols_in: usize,
        b_cols_out: usize,
        b_size: usize,
    ) -> usize;

    fn vmp_apply_dft_to_dft(
        module: &Module<BE>,
        res: &mut crate::layouts::VecZnxDftBackendMut<'_, BE>,
        a: &crate::layouts::VecZnxDftBackendRef<'_, BE>,
        b: &crate::layouts::VmpPMatBackendRef<'_, BE>,
        limb_offset: usize,
        scratch: &mut ScratchArena<'_, BE>,
    );

    /// Returns scratch bytes required for [`Self::vmp_apply_dft_to_dft_add`].
    /// Default body: `vmp_apply_dft_to_dft_add_tmp_bytes_derived`, a full-width
    /// `res_size`-limb staging accumulator plus the product's own scratch.
    #[allow(clippy::too_many_arguments)]
    fn vmp_apply_dft_to_dft_add_tmp_bytes(
        module: &Module<BE>,
        res_size: usize,
        a_size: usize,
        b_rows: usize,
        b_cols_in: usize,
        b_cols_out: usize,
        b_size: usize,
    ) -> usize {
        crate::oep::vmp_apply_dft_to_dft_add_tmp_bytes_derived::<Self, BE>(
            module, res_size, a_size, b_rows, b_cols_in, b_cols_out, b_size,
        )
    }

    /// `res += a * pmat` in the DFT domain. Default body:
    /// `vmp_apply_dft_to_dft_add_derived`, which carves the zeroed staging
    /// accumulator out of [`Self::vmp_apply_dft_to_dft_add_tmp_bytes`] and folds
    /// it into `res` column by column.
    fn vmp_apply_dft_to_dft_add(
        module: &Module<BE>,
        res: &mut crate::layouts::VecZnxDftBackendMut<'_, BE>,
        a: &crate::layouts::VecZnxDftBackendRef<'_, BE>,
        b: &crate::layouts::VmpPMatBackendRef<'_, BE>,
        limb_offset: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) {
        crate::oep::vmp_apply_dft_to_dft_add_derived::<Self, BE>(module, res, a, b, limb_offset, scratch)
    }

    fn vmp_extract_selected_rows(
        module: &Module<BE>,
        res: &mut crate::layouts::VmpPMatBackendMut<'_, BE>,
        a: &crate::layouts::VmpPMatBackendRef<'_, BE>,
        first_row: usize,
        row_step: usize,
    );

    fn vmp_zero(module: &Module<BE>, res: &mut crate::layouts::VmpPMatBackendMut<'_, BE>);
}

/// Convolution family extension point.
///
/// # Safety
/// Implementations must uphold the backend safety contract for prepared matrix
/// layouts, scratch usage, and arithmetic correctness.
pub unsafe trait HalConvolutionImpl<BE: Backend>: Backend + HalVecZnxDftImpl<BE> + HalVecZnxBigImpl<BE> {
    fn cnv_prepare_left_tmp_bytes(module: &Module<BE>, res_size: usize, a_size: usize) -> usize;

    fn cnv_prepare_left(
        module: &Module<BE>,
        res: &mut crate::layouts::CnvPVecLBackendMut<'_, BE>,
        a: &crate::layouts::VecZnxBackendRef<'_, BE>,
        mask: i64,
        scratch: &mut ScratchArena<'_, BE>,
    );

    fn cnv_prepare_right_tmp_bytes(module: &Module<BE>, res_size: usize, a_size: usize) -> usize;

    fn cnv_prepare_right(
        module: &Module<BE>,
        res: &mut crate::layouts::CnvPVecRBackendMut<'_, BE>,
        a: &crate::layouts::VecZnxBackendRef<'_, BE>,
        mask: i64,
        scratch: &mut ScratchArena<'_, BE>,
    );

    fn cnv_apply_dft_tmp_bytes(module: &Module<BE>, cnv_offset: usize, res_size: usize, a_size: usize, b_size: usize) -> usize;

    fn cnv_by_const_apply_tmp_bytes(
        module: &Module<BE>,
        cnv_offset: usize,
        res_size: usize,
        a_size: usize,
        b_size: usize,
    ) -> usize;

    /// Required, not derived: this is an exact big-domain product of `a` with
    /// one coefficient column of `b`. The spec's DFT decomposition would route
    /// it through an approximate transform on FFT64 (spec section 4.3, PR4
    /// deviation).
    #[allow(clippy::too_many_arguments)]
    fn cnv_by_const_apply(
        module: &Module<BE>,
        cnv_offset: usize,
        res: &mut crate::layouts::VecZnxBigBackendMut<'_, BE>,
        res_col: usize,
        a: &crate::layouts::VecZnxBackendRef<'_, BE>,
        a_col: usize,
        b: &crate::layouts::VecZnxBackendRef<'_, BE>,
        b_col: usize,
        b_coeff: usize,
        scratch: &mut ScratchArena<'_, BE>,
    );

    /// Returns scratch bytes required for [`HalConvolutionImpl::cnv_by_const_apply_add`].
    fn cnv_by_const_apply_add_tmp_bytes(
        module: &Module<BE>,
        cnv_offset: usize,
        res_size: usize,
        a_size: usize,
        b_size: usize,
    ) -> usize {
        crate::oep::cnv_by_const_apply_add_tmp_bytes_derived::<Self, BE>(module, cnv_offset, res_size, a_size, b_size)
    }

    /// `res[res_col] +=` the [`Self::cnv_by_const_apply`] result; limbs the
    /// convolution would zero-fill are left untouched.
    #[allow(clippy::too_many_arguments)]
    fn cnv_by_const_apply_add(
        module: &Module<BE>,
        cnv_offset: usize,
        res: &mut crate::layouts::VecZnxBigBackendMut<'_, BE>,
        res_col: usize,
        a: &crate::layouts::VecZnxBackendRef<'_, BE>,
        a_col: usize,
        b: &crate::layouts::VecZnxBackendRef<'_, BE>,
        b_col: usize,
        b_coeff: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) {
        crate::oep::cnv_by_const_apply_add_derived::<Self, BE>(
            module, cnv_offset, res, res_col, a, a_col, b, b_col, b_coeff, scratch,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn cnv_apply_dft(
        module: &Module<BE>,
        cnv_offset: usize,
        res: &mut crate::layouts::VecZnxDftBackendMut<'_, BE>,
        res_col: usize,
        a: &crate::layouts::CnvPVecLBackendRef<'_, BE>,
        a_col: usize,
        b: &crate::layouts::CnvPVecRBackendRef<'_, BE>,
        b_col: usize,
        scratch: &mut ScratchArena<'_, BE>,
    );

    /// Returns scratch bytes required for [`HalConvolutionImpl::cnv_apply_dft_add`].
    fn cnv_apply_dft_add_tmp_bytes(
        module: &Module<BE>,
        cnv_offset: usize,
        res_size: usize,
        a_size: usize,
        b_size: usize,
    ) -> usize {
        crate::oep::cnv_apply_dft_add_tmp_bytes_derived::<Self, BE>(module, cnv_offset, res_size, a_size, b_size)
    }

    /// `res[res_col] += a (x) b`. Default body: `cnv_apply_dft_add_derived`,
    /// which carves a `res.size()`-limb staging `VecZnxDft` out of
    /// [`Self::cnv_apply_dft_add_tmp_bytes`] and adds it into `res`.
    #[allow(clippy::too_many_arguments)]
    fn cnv_apply_dft_add(
        module: &Module<BE>,
        cnv_offset: usize,
        res: &mut crate::layouts::VecZnxDftBackendMut<'_, BE>,
        res_col: usize,
        a: &crate::layouts::CnvPVecLBackendRef<'_, BE>,
        a_col: usize,
        b: &crate::layouts::CnvPVecRBackendRef<'_, BE>,
        b_col: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) {
        crate::oep::cnv_apply_dft_add_derived::<Self, BE>(module, cnv_offset, res, res_col, a, a_col, b, b_col, scratch)
    }

    /// Returns scratch bytes required for [`HalConvolutionImpl::cnv_apply_dft_sum`].
    ///
    /// The default sizes the per-term fallback (one `cnv_apply_dft` /
    /// `cnv_apply_dft_add` scratch). Backends with a fused kernel should
    /// override both methods together.
    fn cnv_apply_dft_sum_tmp_bytes(
        module: &Module<BE>,
        cnv_offset: usize,
        res_size: usize,
        a_size: usize,
        b_size: usize,
    ) -> usize {
        crate::oep::cnv_apply_dft_sum_tmp_bytes_derived::<Self, BE>(module, cnv_offset, res_size, a_size, b_size)
    }

    /// Computes `res[res_col] = Σ_t a_t ⊛ b_t` (overwriting).
    ///
    /// The default implementation overwrites with the first term
    /// (`cnv_apply_dft`, which also zeroes the limbs past the convolution
    /// bound) and folds the remaining terms with `cnv_apply_dft_add`.
    /// Backends should override it with a fused kernel that keeps the lazy
    /// accumulators live across terms.
    fn cnv_apply_dft_sum<'a>(
        module: &Module<BE>,
        cnv_offset: usize,
        res: &mut crate::layouts::VecZnxDftBackendMut<'_, BE>,
        res_col: usize,
        terms: &[crate::layouts::CnvDftAccTerm<'a, BE>],
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        BE: 'a,
    {
        crate::oep::cnv_apply_dft_sum_derived::<Self, BE>(module, cnv_offset, res, res_col, terms, scratch)
    }

    /// Returns scratch bytes required for [`Self::cnv_pairwise_apply_dft`].
    /// Default body: `cnv_pairwise_apply_dft_tmp_bytes_derived`, the larger of
    /// the one overwriting and the three accumulating products it chains.
    fn cnv_pairwise_apply_dft_tmp_bytes(
        module: &Module<BE>,
        cnv_offset: usize,
        res_size: usize,
        a_size: usize,
        b_size: usize,
    ) -> usize {
        crate::oep::cnv_pairwise_apply_dft_tmp_bytes_derived::<Self, BE>(module, cnv_offset, res_size, a_size, b_size)
    }

    /// `res[res_col] = (a[i] + a[j]) (x) (b[i] + b[j])`, expanded in the DFT
    /// domain where the prepared operands are linear. Default body:
    /// `cnv_pairwise_apply_dft_derived`; scratch is
    /// [`Self::cnv_pairwise_apply_dft_tmp_bytes`].
    #[allow(clippy::too_many_arguments)]
    fn cnv_pairwise_apply_dft(
        module: &Module<BE>,
        cnv_offset: usize,
        res: &mut crate::layouts::VecZnxDftBackendMut<'_, BE>,
        res_col: usize,
        a: &crate::layouts::CnvPVecLBackendRef<'_, BE>,
        b: &crate::layouts::CnvPVecRBackendRef<'_, BE>,
        i: usize,
        j: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) {
        crate::oep::cnv_pairwise_apply_dft_derived::<Self, BE>(module, cnv_offset, res, res_col, a, b, i, j, scratch)
    }

    /// Returns scratch bytes required for [`Self::cnv_prepare_self`]. Default
    /// body: `cnv_prepare_self_tmp_bytes_derived`, the larger of the two
    /// prepares.
    fn cnv_prepare_self_tmp_bytes(module: &Module<BE>, res_size: usize, a_size: usize) -> usize {
        crate::oep::cnv_prepare_self_tmp_bytes_derived::<Self, BE>(module, res_size, a_size)
    }

    /// Prepares `a` as both a left and a right convolution factor. Default body:
    /// `cnv_prepare_self_derived`, one [`Self::cnv_prepare_left`] then one
    /// [`Self::cnv_prepare_right`].
    fn cnv_prepare_self(
        module: &Module<BE>,
        left: &mut crate::layouts::CnvPVecLBackendMut<'_, BE>,
        right: &mut crate::layouts::CnvPVecRBackendMut<'_, BE>,
        a: &crate::layouts::VecZnxBackendRef<'_, BE>,
        mask: i64,
        scratch: &mut ScratchArena<'_, BE>,
    ) {
        crate::oep::cnv_prepare_self_derived::<Self, BE>(module, left, right, a, mask, scratch)
    }
}
