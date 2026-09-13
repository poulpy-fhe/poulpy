use crate::{
    layouts::{Backend, ScalarZnxBackendMut, ScalarZnxBackendRef, ScratchArena, VecZnxBackendMut, VecZnxBackendRef},
    source::Source,
};

pub trait VecZnxNormalizeTmpBytes {
    /// Returns the minimum number of bytes necessary for normalization.
    fn vec_znx_normalize_tmp_bytes(&self) -> usize;
}

pub trait VecZnxZero<B: Backend> {
    fn vec_znx_zero(&self, res: &mut VecZnxBackendMut<'_, B>, res_col: usize);
}

/// Converts a column to centered digits, rounding once at the destination precision.
///
/// For i64 backends, each input limb coefficient must lie in `[-2^62, 2^62]`,
/// and both radix widths must lie in `1..=62`. These bounds leave room for
/// shifted digits and propagated carries. They are caller preconditions;
/// normalization does not scan the input to validate them.
pub trait VecZnxNormalize<B: Backend> {
    #[allow(clippy::too_many_arguments)]
    /// Normalizes the selected column of `a` at `res_k` bits into `res`.
    fn vec_znx_normalize(
        &self,
        res: &mut VecZnxBackendMut<'_, B>,
        res_base2k: usize,
        res_k: usize,
        res_offset: i64,
        res_col: usize,
        a: &VecZnxBackendRef<'_, B>,
        a_base2k: usize,
        a_col: usize,
        scratch: &mut ScratchArena<'_, B>,
    );
}

/// In-place normalization with the input and radix bounds of [`VecZnxNormalize`].
pub trait VecZnxNormalizeAssign<B: Backend> {
    fn vec_znx_normalize_assign(
        &self,
        base2k: usize,
        k: usize,
        a: &mut VecZnxBackendMut<'_, B>,
        a_col: usize,
        scratch: &mut ScratchArena<'_, B>,
    );
}

pub trait VecZnxAdd<B: Backend> {
    /// Adds the selected backend-native column of `a` to the selected backend-native column of `b`.
    fn vec_znx_add(
        &self,
        res: &mut VecZnxBackendMut<'_, B>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, B>,
        a_col: usize,
        b: &VecZnxBackendRef<'_, B>,
        b_col: usize,
    );
}

pub trait VecZnxAddAssign<B: Backend> {
    fn vec_znx_add_assign(&self, res: &mut VecZnxBackendMut<'_, B>, res_col: usize, a: &VecZnxBackendRef<'_, B>, a_col: usize);
}

pub trait VecZnxAddScalarAssign<B: Backend> {
    fn vec_znx_add_scalar_assign(
        &self,
        res: &mut VecZnxBackendMut<'_, B>,
        res_col: usize,
        res_limb: usize,
        a: &ScalarZnxBackendRef<'_, B>,
        a_col: usize,
    );
}

pub trait VecZnxSub<B: Backend> {
    fn vec_znx_sub(
        &self,
        res: &mut VecZnxBackendMut<'_, B>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, B>,
        a_col: usize,
        b: &VecZnxBackendRef<'_, B>,
        b_col: usize,
    );
}

pub trait VecZnxSubAssign<B: Backend> {
    fn vec_znx_sub_assign(&self, res: &mut VecZnxBackendMut<'_, B>, res_col: usize, a: &VecZnxBackendRef<'_, B>, a_col: usize);
}

pub trait VecZnxSubNegateAssign<B: Backend> {
    fn vec_znx_sub_negate_assign(
        &self,
        res: &mut VecZnxBackendMut<'_, B>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, B>,
        a_col: usize,
    );
}

pub trait VecZnxNegate<B: Backend> {
    fn vec_znx_negate(&self, res: &mut VecZnxBackendMut<'_, B>, res_col: usize, a: &VecZnxBackendRef<'_, B>, a_col: usize);
}

pub trait VecZnxNegateAssign<B: Backend> {
    fn vec_znx_negate_assign(&self, a: &mut VecZnxBackendMut<'_, B>, a_col: usize);
}

/// Returns scratch bytes required for the left-shift family:
/// [`VecZnxLsh::vec_znx_lsh`], [`VecZnxLshAdd::vec_znx_lsh_add`],
/// [`VecZnxLshSub::vec_znx_lsh_sub`] and
/// [`VecZnxLshAssign::vec_znx_lsh_assign`], on a destination of `res_size`
/// limbs.
pub trait VecZnxLshTmpBytes {
    fn vec_znx_lsh_tmp_bytes(&self, res_size: usize) -> usize;
}

pub trait VecZnxLsh<B: Backend> {
    /// `res[res_col] = a[a_col] * 2^k`, both operands at the same radix: the
    /// operation is exactly `vec_znx_normalize` with `res_base2k = a_base2k =
    /// base2k`, `res_k = res.size() * base2k` and `offset = +k`.
    ///
    /// A destination shorter than the source is not a truncation: the limbs of
    /// the shifted value that `res` cannot hold are rounded into its last limb
    /// jointly, as the normalization does.
    ///
    /// Scratch is the family's
    /// [`vec_znx_lsh_tmp_bytes`](VecZnxLshTmpBytes::vec_znx_lsh_tmp_bytes) on
    /// `res.size()`.
    #[allow(clippy::too_many_arguments)]
    fn vec_znx_lsh(
        &self,
        base2k: usize,
        k: usize,
        res: &mut VecZnxBackendMut<'_, B>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, B>,
        a_col: usize,
        scratch: &mut ScratchArena<'_, B>,
    );
}

pub trait VecZnxLshAdd<B: Backend> {
    /// `res[res_col] += a[a_col] * 2^k`, both operands at the same radix: the
    /// addend is exactly [`vec_znx_lsh`](VecZnxLsh::vec_znx_lsh) on a
    /// `res.size()`-limb destination — the shift followed by a normalize at
    /// `base2k`.
    ///
    /// An addend shorter than the source is not a truncation: the limbs of the
    /// shifted value that `res.size()` cannot hold are rounded into the last
    /// one jointly, as the normalization does.
    ///
    /// Scratch is the family's
    /// [`vec_znx_lsh_tmp_bytes`](VecZnxLshTmpBytes::vec_znx_lsh_tmp_bytes) on
    /// `res.size()`.
    ///
    /// Normalization contract: the shifted operand `a` is normalized on the fly
    /// (its own inter-limb carries are propagated), so the addend is in the
    /// canonical `base2k` digit range. The addition into `res` is **not**
    /// re-normalized: adding a normalized digit onto an already-normalized `res`
    /// limb can leave that limb one bit beyond the `base2k` range. Callers that
    /// require a normalized `res` afterwards must normalize it themselves; this
    /// op alone does not restore the digit contract on `res`.
    #[allow(clippy::too_many_arguments)]
    fn vec_znx_lsh_add(
        &self,
        base2k: usize,
        k: usize,
        res: &mut VecZnxBackendMut<'_, B>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, B>,
        a_col: usize,
        scratch: &mut ScratchArena<'_, B>,
    );
}

/// Returns scratch bytes required for the right-shift family:
/// [`VecZnxRsh::vec_znx_rsh`], [`VecZnxRshAdd::vec_znx_rsh_add`],
/// [`VecZnxRshSub::vec_znx_rsh_sub`] and
/// [`VecZnxRshAssign::vec_znx_rsh_assign`], on a destination of `res_size`
/// limbs.
pub trait VecZnxRshTmpBytes {
    fn vec_znx_rsh_tmp_bytes(&self, res_size: usize) -> usize;
}

pub trait VecZnxRsh<B: Backend> {
    /// `res[res_col] = a[a_col] / 2^k`, both operands at the same radix: the
    /// operation is exactly `vec_znx_normalize` with `res_base2k = a_base2k =
    /// base2k`, `res_k = res.size() * base2k` and `offset = -k`.
    ///
    /// A destination shorter than the source is not a truncation: the limbs of
    /// the shifted value that `res` cannot hold are rounded into its last limb
    /// jointly, as the normalization does. A `k` beyond `res.size() * base2k`
    /// has shifted every bit past the destination's last limb, so `res` is
    /// zero.
    ///
    /// Scratch is the family's
    /// [`vec_znx_rsh_tmp_bytes`](VecZnxRshTmpBytes::vec_znx_rsh_tmp_bytes) on
    /// `res.size()`.
    #[allow(clippy::too_many_arguments)]
    fn vec_znx_rsh(
        &self,
        base2k: usize,
        k: usize,
        res: &mut VecZnxBackendMut<'_, B>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, B>,
        a_col: usize,
        scratch: &mut ScratchArena<'_, B>,
    );
}

pub trait VecZnxRshAdd<B: Backend> {
    /// `res[res_col] += a[a_col] / 2^k`, both operands at the same radix: the
    /// addend is exactly [`vec_znx_rsh`](VecZnxRsh::vec_znx_rsh) on a
    /// `res.size()`-limb destination — the shift followed by a normalize at
    /// `base2k`.
    ///
    /// An addend shorter than the source is not a truncation: the limbs of the
    /// shifted value that `res.size()` cannot hold are rounded into the last
    /// one jointly, as the normalization does. A `k` beyond
    /// `res.size() * base2k` has shifted every bit past the last limb, so
    /// nothing is added.
    ///
    /// Scratch is the family's
    /// [`vec_znx_rsh_tmp_bytes`](VecZnxRshTmpBytes::vec_znx_rsh_tmp_bytes) on
    /// `res.size()`.
    ///
    /// Normalization contract: the shifted operand `a` is normalized on the fly
    /// (its own inter-limb carries are propagated), so the addend is in the
    /// canonical `base2k` digit range. The addition into `res` is **not**
    /// re-normalized: adding a normalized digit onto an already-normalized `res`
    /// limb can leave that limb one bit beyond the `base2k` range. Callers that
    /// require a normalized `res` afterwards must normalize it themselves; this
    /// op alone does not restore the digit contract on `res`.
    #[allow(clippy::too_many_arguments)]
    fn vec_znx_rsh_add(
        &self,
        base2k: usize,
        k: usize,
        res: &mut VecZnxBackendMut<'_, B>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, B>,
        a_col: usize,
        scratch: &mut ScratchArena<'_, B>,
    );
}

pub trait VecZnxLshSub<B: Backend> {
    /// `res[res_col] -= a[a_col] * 2^k`, both operands at the same radix: the
    /// subtrahend is exactly [`vec_znx_lsh`](VecZnxLsh::vec_znx_lsh) on a
    /// `res.size()`-limb destination — the shift followed by a normalize at
    /// `base2k`.
    ///
    /// A subtrahend shorter than the source is not a truncation: the limbs of
    /// the shifted value that `res.size()` cannot hold are rounded into the
    /// last one jointly, as the normalization does.
    ///
    /// Scratch is the family's
    /// [`vec_znx_lsh_tmp_bytes`](VecZnxLshTmpBytes::vec_znx_lsh_tmp_bytes) on
    /// `res.size()`.
    ///
    /// Normalization contract: as for
    /// [`vec_znx_lsh_add`](VecZnxLshAdd::vec_znx_lsh_add), the subtraction into
    /// `res` is **not** re-normalized; callers that need a normalized `res`
    /// afterwards must normalize it themselves.
    #[allow(clippy::too_many_arguments)]
    fn vec_znx_lsh_sub(
        &self,
        base2k: usize,
        k: usize,
        res: &mut VecZnxBackendMut<'_, B>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, B>,
        a_col: usize,
        scratch: &mut ScratchArena<'_, B>,
    );
}

pub trait VecZnxRshSub<B: Backend> {
    /// `res[res_col] -= a[a_col] / 2^k`, both operands at the same radix: the
    /// subtrahend is exactly [`vec_znx_rsh`](VecZnxRsh::vec_znx_rsh) on a
    /// `res.size()`-limb destination — the shift followed by a normalize at
    /// `base2k`.
    ///
    /// A subtrahend shorter than the source is not a truncation: the limbs of
    /// the shifted value that `res.size()` cannot hold are rounded into the
    /// last one jointly, as the normalization does. A `k` beyond
    /// `res.size() * base2k` has shifted every bit past the last limb, so
    /// nothing is subtracted.
    ///
    /// Scratch is the family's
    /// [`vec_znx_rsh_tmp_bytes`](VecZnxRshTmpBytes::vec_znx_rsh_tmp_bytes) on
    /// `res.size()`.
    ///
    /// Normalization contract: as for
    /// [`vec_znx_rsh_add`](VecZnxRshAdd::vec_znx_rsh_add), the subtraction into
    /// `res` is **not** re-normalized; callers that need a normalized `res`
    /// afterwards must normalize it themselves.
    #[allow(clippy::too_many_arguments)]
    fn vec_znx_rsh_sub(
        &self,
        base2k: usize,
        k: usize,
        res: &mut VecZnxBackendMut<'_, B>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, B>,
        a_col: usize,
        scratch: &mut ScratchArena<'_, B>,
    );
}

pub trait VecZnxLshAssign<B: Backend> {
    /// `a[a_col] *= 2^k`, in place on the one selected column: the operation
    /// is exactly [`vec_znx_lsh`](VecZnxLsh::vec_znx_lsh) with `a` as both
    /// source and destination — the shift followed by a normalize at `base2k`
    /// over `a.size()` limbs. The other columns of `a` are untouched.
    ///
    /// Scratch is the family's
    /// [`vec_znx_lsh_tmp_bytes`](VecZnxLshTmpBytes::vec_znx_lsh_tmp_bytes) on
    /// `a.size()`.
    fn vec_znx_lsh_assign(
        &self,
        base2k: usize,
        k: usize,
        a: &mut VecZnxBackendMut<'_, B>,
        a_col: usize,
        scratch: &mut ScratchArena<'_, B>,
    );
}

pub trait VecZnxRshAssign<B: Backend> {
    /// `a[a_col] /= 2^k`, in place on the one selected column: the operation
    /// is exactly [`vec_znx_rsh`](VecZnxRsh::vec_znx_rsh) with `a` as both
    /// source and destination — the shift followed by a normalize at `base2k`
    /// over `a.size()` limbs. A `k` beyond `a.size() * base2k` has shifted
    /// every bit past the last limb, so the column is zeroed. The other
    /// columns of `a` are untouched.
    ///
    /// Scratch is the family's
    /// [`vec_znx_rsh_tmp_bytes`](VecZnxRshTmpBytes::vec_znx_rsh_tmp_bytes) on
    /// `a.size()`.
    fn vec_znx_rsh_assign(
        &self,
        base2k: usize,
        k: usize,
        a: &mut VecZnxBackendMut<'_, B>,
        a_col: usize,
        scratch: &mut ScratchArena<'_, B>,
    );
}

pub trait VecZnxRotate<B: Backend> {
    /// Multiplies the selected column of `a` by X^k and stores the result in `res_col` of `res`.
    fn vec_znx_rotate(
        &self,
        p: i64,
        res: &mut VecZnxBackendMut<'_, B>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, B>,
        a_col: usize,
    );
}

pub trait VecZnxRotateAssignTmpBytes {
    fn vec_znx_rotate_assign_tmp_bytes(&self) -> usize;
}

pub trait VecZnxRotateAssign<B: Backend> {
    /// Multiplies the selected column of `a` by X^k.
    fn vec_znx_rotate_assign(&self, p: i64, a: &mut VecZnxBackendMut<'_, B>, a_col: usize, scratch: &mut ScratchArena<'_, B>);
}

pub trait VecZnxAutomorphism<B: Backend> {
    /// Applies the automorphism X^i -> X^ik on the selected column of `a` and stores the result in `res_col` column of `res`.
    fn vec_znx_automorphism(
        &self,
        k: i64,
        res: &mut VecZnxBackendMut<'_, B>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, B>,
        a_col: usize,
    );
}

pub trait VecZnxAutomorphismAssignTmpBytes {
    fn vec_znx_automorphism_assign_tmp_bytes(&self) -> usize;
}

pub trait VecZnxAutomorphismAssign<B: Backend> {
    /// Applies the automorphism X^i -> X^ik on the selected column of `a`.
    fn vec_znx_automorphism_assign(
        &self,
        k: i64,
        res: &mut VecZnxBackendMut<'_, B>,
        res_col: usize,
        scratch: &mut ScratchArena<'_, B>,
    );
}

pub trait ScalarZnxAutomorphism<B: Backend> {
    /// Applies the automorphism X^i -> X^ik on the selected column of `a` and stores the result in `res_col` column of `res`.
    fn scalar_znx_automorphism(
        &self,
        k: i64,
        res: &mut ScalarZnxBackendMut<'_, B>,
        res_col: usize,
        a: &ScalarZnxBackendRef<'_, B>,
        a_col: usize,
    );
}

/// Multiplies the selected column by `(X^p - 1)` in `Z[X]/(X^N + 1)`.
pub trait VecZnxMulXpMinusOne<B: Backend> {
    fn vec_znx_mul_xp_minus_one(
        &self,
        p: i64,
        res: &mut VecZnxBackendMut<'_, B>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, B>,
        a_col: usize,
    );
}

/// Returns scratch bytes required for
/// [`VecZnxMulXpMinusOneAssign::vec_znx_mul_xp_minus_one_assign`] on a
/// destination of `size` limbs.
pub trait VecZnxMulXpMinusOneAssignTmpBytes {
    fn vec_znx_mul_xp_minus_one_assign_tmp_bytes(&self, size: usize) -> usize;
}

pub trait VecZnxMulXpMinusOneAssign<B: Backend> {
    fn vec_znx_mul_xp_minus_one_assign(
        &self,
        p: i64,
        res: &mut VecZnxBackendMut<'_, B>,
        res_col: usize,
        scratch: &mut ScratchArena<'_, B>,
    );
}

/// Switches ring degree between `a` and `res` by truncation or zero-padding.
pub trait VecZnxSwitchRing<B: Backend> {
    fn vec_znx_switch_ring(&self, res: &mut VecZnxBackendMut<'_, B>, res_col: usize, a: &VecZnxBackendRef<'_, B>, a_col: usize);
}

pub trait VecZnxCopy<B: Backend> {
    fn vec_znx_copy(&self, res: &mut VecZnxBackendMut<'_, B>, res_col: usize, a: &VecZnxBackendRef<'_, B>, a_col: usize);
}

pub trait VecZnxFillUniformSource<B: Backend> {
    /// Fills a column with a uniform `k`-bit torus value in base `2^base2k`.
    ///
    /// Unused low bits in the last live limb and limbs above `k` are zeroed.
    fn vec_znx_fill_uniform_source(
        &self,
        base2k: usize,
        k: usize,
        res: &mut VecZnxBackendMut<'_, B>,
        res_col: usize,
        source: &mut Source,
    );
}
