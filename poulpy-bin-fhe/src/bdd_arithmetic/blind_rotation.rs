use poulpy_core::{
    GLWECopy, GLWERotate, GLWEZero, ScratchArenaTakeCore,
    layouts::{
        GGSWAtViewMut, GGSWAtViewRef, GGSWInfos, GGSWToBackendMut, GGSWToBackendRef, GLWEInfos, GLWEToBackendMut,
        GLWEToBackendRef,
    },
};
use poulpy_hal::layouts::CoeffNormalized;
use poulpy_hal::{
    api::{VecZnxAddScalarAssignBackend, VecZnxNormalizeAssignBackend},
    layouts::{Backend, Module, ScalarZnxToBackendRef, ScratchArena, VecZnxToBackendMut},
};

use crate::bdd_arithmetic::{Cmux, GetGGSWBit, UnsignedInteger};
use poulpy_core::GLWEBytesOf;
use poulpy_core::layouts::prepared::GGSWPreparedToBackendRef;
use poulpy_hal::layouts::BorrowedCarryView;

impl<T: UnsignedInteger, BE: Backend<ZnxWord = i64>> GGSWBlindRotation<T, BE> for Module<BE> where
    Self: GLWEBytesOf<BE>
        + GLWEBlindRotation<BE>
        + GLWEZero<BE>
        + VecZnxAddScalarAssignBackend<BE>
        + VecZnxNormalizeAssignBackend<BE>
{
}

/// Extends [`GLWEBlindRotation`] to lift blind rotation to GGSW matrices and
/// to construct a GGSW from a scalar test-vector via blind rotation.
///
/// A GGSW matrix is a `(dnum × (rank+1))` array of GLWE ciphertexts.  The two
/// methods in this trait apply [`GLWEBlindRotation`] row-by-row:
///
/// - `ggsw_blind_rotation`: rotates each GLWE row of an existing GGSW by the
///   encrypted exponent derived from `fhe_uint`.
/// - `scalar_to_ggsw_blind_rotation`: constructs a fresh GGSW by first placing
///   the scalar test-vector into each row of a temporary GLWE and then rotating.
pub trait GGSWBlindRotation<T: UnsignedInteger, BE: Backend>
where
    Self: GLWEBytesOf<BE>
        + GLWEBlindRotation<BE>
        + GLWEZero<BE>
        + VecZnxAddScalarAssignBackend<BE>
        + VecZnxNormalizeAssignBackend<BE>,
{
    /// Returns the minimum scratch-space size in bytes required by
    /// [`ggsw_blind_rotation`][Self::ggsw_blind_rotation].
    fn ggsw_to_ggsw_blind_rotation_tmp_bytes<R, K>(&self, res_infos: &R, k_infos: &K) -> usize
    where
        R: GLWEInfos,
        K: GGSWInfos,
    {
        self.glwe_blind_rotation_tmp_bytes(res_infos, k_infos)
    }

    #[allow(clippy::too_many_arguments)]
    /// res <- res * X^{((k>>bit_rsh) % 2^bit_mask) << bit_lsh}.
    fn ggsw_blind_rotation_assign<R, K>(
        &self,
        res: &mut R,
        fhe_uint: &K,
        sign: bool,
        bit_rsh: usize,
        bit_mask: usize,
        bit_lsh: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GGSWToBackendMut<BE> + GGSWAtViewMut<BE> + GGSWInfos,
        K: GetGGSWBit<BE>,
        BE: Backend<ZnxWord = i64> + 'static,
    {
        for col in 0..(res.rank() + 1).into() {
            for row in 0..res.dnum().into() {
                self.glwe_blind_rotation_assign(
                    &mut res.at_view_mut(row, col),
                    fhe_uint,
                    sign,
                    bit_rsh,
                    bit_mask,
                    bit_lsh,
                    scratch,
                );
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    /// res <- a * X^{((k>>bit_rsh) % 2^bit_mask) << bit_lsh}.
    fn ggsw_blind_rotation<R, A, K>(
        &self,
        res: &mut R,
        a: &A,
        fhe_uint: &K,
        sign: bool,
        bit_rsh: usize,
        bit_mask: usize,
        bit_lsh: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GGSWToBackendMut<BE> + GGSWAtViewMut<BE> + GGSWInfos,
        A: GGSWToBackendRef<BE> + GGSWAtViewRef<BE> + GGSWInfos,
        K: GetGGSWBit<BE>,
        BE: Backend<ZnxWord = i64> + 'static,
    {
        assert!(res.dnum() <= a.dnum());
        assert_eq!(res.dsize(), a.dsize());

        for col in 0..(res.rank() + 1).into() {
            for row in 0..res.dnum().into() {
                self.glwe_blind_rotation(
                    &mut res.at_view_mut(row, col),
                    &a.at_view(row, col),
                    fhe_uint,
                    sign,
                    bit_rsh,
                    bit_mask,
                    bit_lsh,
                    scratch,
                );
            }
        }
    }

    fn scalar_to_ggsw_blind_rotation_tmp_bytes<R, K>(&self, res_infos: &R, k_infos: &K) -> usize
    where
        Self: GLWEBytesOf<BE>,
        R: GLWEInfos,
        K: GGSWInfos,
    {
        self.glwe_blind_rotation_tmp_bytes(res_infos, k_infos) + self.glwe_bytes_of_from_infos(res_infos)
    }

    #[allow(clippy::too_many_arguments)]
    fn scalar_to_ggsw_blind_rotation<R, A, K>(
        &self,
        res: &mut R,
        test_vector: &A,
        fhe_uint: &K,
        sign: bool,
        bit_rsh: usize,
        bit_mask: usize,
        bit_lsh: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GGSWToBackendMut<BE> + GGSWAtViewMut<BE> + GGSWInfos,
        A: ScalarZnxToBackendRef<BE>,
        K: GetGGSWBit<BE>,
        BE: Backend<ZnxWord = i64> + 'static,
    {
        let base2k: usize = res.base2k().into();
        let dsize: usize = res.dsize().into();
        let (mut tmp_glwe, mut scratch_1) = scratch.borrow().take_glwe_scratch(res);
        let test_vector = test_vector.to_backend_ref();

        for col in 0..(res.rank() + 1).into() {
            for row in 0..res.dnum().into() {
                self.glwe_zero(&mut tmp_glwe);
                {
                    let mut tmp_glwe_inner = tmp_glwe.data_mut();
                    let mut tmp_glwe_data = VecZnxToBackendMut::<BE>::to_backend_mut(&mut tmp_glwe_inner).borrowed_carry_view();
                    self.vec_znx_add_scalar_assign_backend(&mut tmp_glwe_data, col, (dsize - 1) + row * dsize, &test_vector, 0);
                    self.vec_znx_normalize_assign_backend(base2k, &mut tmp_glwe_data, col, &mut scratch_1.borrow());
                }

                self.glwe_blind_rotation(
                    &mut res.at_view_mut(row, col),
                    &tmp_glwe,
                    fhe_uint,
                    sign,
                    bit_rsh,
                    bit_mask,
                    bit_lsh,
                    &mut scratch_1.borrow(),
                );
            }
        }
    }
}

impl<BE: Backend<ZnxWord = i64>> GLWEBlindRotation<BE> for Module<BE> where
    Self: GLWEBytesOf<BE> + GLWECopy<BE> + GLWERotate<BE> + Cmux<BE>
{
}

/// Homomorphic rotation of a GLWE ciphertext by an encrypted exponent.
///
/// Given a GLWE ciphertext `a` and a set of GGSW ciphertexts encoding the bits
/// of an integer `k`, computes:
///
/// ```text
/// res = a * X^{sign * ((k >> bit_rsh) % 2^bit_mask) << bit_lsh}
/// ```
///
/// where `sign` controls whether the rotation is positive or negative.
/// The operation is performed using `bit_mask` successive CMux gates, one per
/// bit of the shift amount.
pub trait GLWEBlindRotation<BE: Backend>
where
    Self: GLWEBytesOf<BE> + GLWECopy<BE> + GLWERotate<BE> + Cmux<BE>,
{
    /// Returns the minimum scratch-space size in bytes required by
    /// [`glwe_blind_rotation`][Self::glwe_blind_rotation].
    fn glwe_blind_rotation_tmp_bytes<R, K>(&self, res_infos: &R, k_infos: &K) -> usize
    where
        Self: GLWEBytesOf<BE>,
        R: GLWEInfos,
        K: GGSWInfos,
    {
        self.cmux_tmp_bytes(res_infos, res_infos, k_infos) + self.glwe_bytes_of_from_infos(res_infos)
    }

    #[allow(clippy::too_many_arguments)]
    fn glwe_blind_rotation_assign<R, K>(
        &self,
        res: &mut R,
        value: &K,
        sign: bool,
        bit_rsh: usize,
        bit_mask: usize,
        bit_lsh: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE, State = CoeffNormalized> + GLWEInfos,
        K: GetGGSWBit<BE>,
        BE: Backend<ZnxWord = i64> + 'static,
    {
        let (mut tmp_res, mut scratch_1) = scratch.borrow().take_glwe_scratch(res);
        let mut res_is_cur = true;

        for i in 0..bit_mask {
            if res_is_cur {
                match sign {
                    true => self.glwe_rotate(1 << (i + bit_lsh), &mut tmp_res, res),
                    false => self.glwe_rotate(-1 << (i + bit_lsh), &mut tmp_res, res),
                }

                let bit = value.get_bit(i + bit_rsh);
                self.cmux_assign(&mut tmp_res, res, &bit.to_backend_ref(), &mut scratch_1.borrow());
            } else {
                match sign {
                    true => self.glwe_rotate(1 << (i + bit_lsh), res, &tmp_res),
                    false => self.glwe_rotate(-1 << (i + bit_lsh), res, &tmp_res),
                }

                let bit = value.get_bit(i + bit_rsh);
                self.cmux_assign(res, &tmp_res, &bit.to_backend_ref(), &mut scratch_1.borrow());
            }

            res_is_cur = !res_is_cur;
        }

        if !res_is_cur {
            self.glwe_copy(res, &tmp_res);
        }
    }

    #[allow(clippy::too_many_arguments)]
    /// res <- a * X^{sign * ((k>>bit_rsh) % 2^bit_mask) << bit_lsh}.
    fn glwe_blind_rotation<R, A, K>(
        &self,
        res: &mut R,
        a: &A,
        fhe_uint: &K,
        sign: bool,
        bit_rsh: usize,
        bit_mask: usize,
        bit_lsh: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE, State = CoeffNormalized> + GLWEInfos,
        A: GLWEToBackendRef<BE, State = CoeffNormalized>,
        K: GetGGSWBit<BE>,
        BE: Backend<ZnxWord = i64> + 'static,
    {
        self.glwe_copy(res, a);
        self.glwe_blind_rotation_assign(res, fhe_uint, sign, bit_rsh, bit_mask, bit_lsh, scratch);
    }
}
