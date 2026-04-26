use poulpy_core::{
    GLWECopy, GLWERotate, ScratchTakeCore,
    layouts::{GGSW, GGSWInfos, GGSWToMut, GGSWToRef, GLWE, GLWEInfos, GLWEToMut, GLWEToRef, LWEInfos},
};
use poulpy_hal::{
    api::{VecZnxAddScalarAssign, VecZnxNormalizeAssign},
    layouts::{Backend, Module, ScalarZnx, ScalarZnxToRef, Scratch, ZnxZero},
};

use crate::bin_fhe::bdd_arithmetic::{Cmux, GetGGSWBit, UnsignedInteger};

impl<T: UnsignedInteger, BE: Backend> GGSWBlindRotation<T, BE> for Module<BE>
where
    Self: GLWEBlindRotation<BE> + VecZnxAddScalarAssign + VecZnxNormalizeAssign<BE>,
    Scratch<BE>: ScratchTakeCore<BE>,
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
    Self: GLWEBlindRotation<BE> + VecZnxAddScalarAssign + VecZnxNormalizeAssign<BE>,
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
        scratch: &mut Scratch<BE>,
    ) where
        R: GGSWToMut,
        K: GetGGSWBit<BE>,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        let res: &mut GGSW<&mut [u8]> = &mut res.to_mut();

        for col in 0..(res.rank() + 1).into() {
            for row in 0..res.dnum().into() {
                self.glwe_blind_rotation_assign(&mut res.at_mut(row, col), fhe_uint, sign, bit_rsh, bit_mask, bit_lsh, scratch);
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
        scratch: &mut Scratch<BE>,
    ) where
        R: GGSWToMut,
        A: GGSWToRef,
        K: GetGGSWBit<BE>,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        let res: &mut GGSW<&mut [u8]> = &mut res.to_mut();
        let a: &GGSW<&[u8]> = &a.to_ref();

        assert!(res.dnum() <= a.dnum());
        assert_eq!(res.dsize(), a.dsize());

        for col in 0..(res.rank() + 1).into() {
            for row in 0..res.dnum().into() {
                self.glwe_blind_rotation(
                    &mut res.at_mut(row, col),
                    &a.at(row, col),
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
        R: GLWEInfos,
        K: GGSWInfos,
    {
        self.glwe_blind_rotation_tmp_bytes(res_infos, k_infos) + GLWE::<Vec<u8>>::bytes_of_from_infos(res_infos)
    }

    #[allow(clippy::too_many_arguments)]
    fn scalar_to_ggsw_blind_rotation<R, S, K>(
        &self,
        res: &mut R,
        test_vector: &S,
        fhe_uint: &K,
        sign: bool,
        bit_rsh: usize,
        bit_mask: usize,
        bit_lsh: usize,
        scratch: &mut Scratch<BE>,
    ) where
        R: GGSWToMut,
        S: ScalarZnxToRef,
        K: GetGGSWBit<BE>,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        let res: &mut GGSW<&mut [u8]> = &mut res.to_mut();
        let test_vector: &ScalarZnx<&[u8]> = &test_vector.to_ref();

        let base2k: usize = res.base2k().into();
        let dsize: usize = res.dsize().into();

        let (mut tmp_glwe, scratch_1) = scratch.take_glwe(res);

        for col in 0..(res.rank() + 1).into() {
            for row in 0..res.dnum().into() {
                tmp_glwe.data_mut().zero();
                self.vec_znx_add_scalar_assign(tmp_glwe.data_mut(), col, (dsize - 1) + row * dsize, test_vector, 0);
                self.vec_znx_normalize_assign(base2k, tmp_glwe.data_mut(), col, scratch_1);

                self.glwe_blind_rotation(
                    &mut res.at_mut(row, col),
                    &tmp_glwe,
                    fhe_uint,
                    sign,
                    bit_rsh,
                    bit_mask,
                    bit_lsh,
                    scratch_1,
                );
            }
        }
    }
}

impl<BE: Backend> GLWEBlindRotation<BE> for Module<BE>
where
    Self: GLWECopy + GLWERotate<BE> + Cmux<BE>,
    Scratch<BE>: ScratchTakeCore<BE>,
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
    Self: GLWECopy + GLWERotate<BE> + Cmux<BE>,
{
    /// Returns the minimum scratch-space size in bytes required by
    /// [`glwe_blind_rotation`][Self::glwe_blind_rotation].
    fn glwe_blind_rotation_tmp_bytes<R, K>(&self, res_infos: &R, k_infos: &K) -> usize
    where
        R: GLWEInfos,
        K: GGSWInfos,
    {
        self.cmux_tmp_bytes(res_infos, res_infos, k_infos) + GLWE::<Vec<u8>>::bytes_of_from_infos(res_infos)
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
        scratch: &mut Scratch<BE>,
    ) where
        R: GLWEToMut,
        K: GetGGSWBit<BE>,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        let mut res: GLWE<&mut [u8]> = res.to_mut();

        let (mut tmp_res, scratch_1) = scratch.take_glwe(&res);

        // a_is_res = true  => (a, b) = (&mut res, &mut tmp_res)
        // a_is_res = false => (a, b) = (&mut tmp_res, &mut res)
        let mut a_is_res: bool = true;

        for i in 0..bit_mask {
            let (a, b) = if a_is_res {
                (&mut res, &mut tmp_res)
            } else {
                (&mut tmp_res, &mut res)
            };

            // a <- a ; b <- a * X^{-2^{i + bit_lsh}}
            match sign {
                true => self.glwe_rotate(1 << (i + bit_lsh), b, a),
                false => self.glwe_rotate(-1 << (i + bit_lsh), b, a),
            }

            // b <- (b - a) * GGSW(b[i]) + a
            self.cmux_assign(b, a, &value.get_bit(i + bit_rsh), scratch_1);

            // ping-pong roles for next iter
            a_is_res = !a_is_res;
        }

        // Ensure the final value ends up in `res`
        if !a_is_res {
            self.glwe_copy(&mut res, &tmp_res);
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
        scratch: &mut Scratch<BE>,
    ) where
        R: GLWEToMut,
        A: GLWEToRef,
        K: GetGGSWBit<BE>,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        self.glwe_copy(res, a);
        self.glwe_blind_rotation_assign(res, fhe_uint, sign, bit_rsh, bit_mask, bit_lsh, scratch);
    }
}
