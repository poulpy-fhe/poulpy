use poulpy_core::{
    GLWECopy, GLWERotate, ScratchArenaTakeCore,
    layouts::{
        GGSWAtViewMut, GGSWAtViewRef, GGSWInfos, GGSWToBackendMut, GGSWToBackendRef, GLWE, GLWEInfos, GLWEToBackendMut,
        GLWEToBackendRef, ModuleCoreAlloc,
    },
};
use poulpy_hal::{
    api::{VecZnxAddScalarAssignBackend, VecZnxNormalizeAssignBackend},
    layouts::{Backend, HostDataMut, Module, ScalarZnx, ScalarZnxToBackendRef, ScratchArena, VecZnxToBackendMut, ZnxZero},
};

use crate::bdd_arithmetic::{Cmux, GetGGSWBit, UnsignedInteger};

impl<T: UnsignedInteger, BE: Backend<OwnedBuf = Vec<u8>>> GGSWBlindRotation<T, BE> for Module<BE>
where
    Self: GLWEBlindRotation<BE> + VecZnxAddScalarAssignBackend<BE> + VecZnxNormalizeAssignBackend<BE>,
    for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
    for<'a> BE::BufMut<'a>: HostDataMut,
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
pub trait GGSWBlindRotation<T: UnsignedInteger, BE: Backend<OwnedBuf = Vec<u8>>>
where
    Self: GLWEBlindRotation<BE>
        + VecZnxAddScalarAssignBackend<BE>
        + VecZnxNormalizeAssignBackend<BE>
        + ModuleCoreAlloc<OwnedBuf = Vec<u8>>,
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
        BE: Backend<OwnedBuf = Vec<u8>>,
        BE: 'static,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
        for<'a> BE::BufMut<'a>: HostDataMut,
        for<'a> BE: Backend<BufMut<'a> = &'a mut [u8], BufRef<'a> = &'a [u8]>,
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
        BE: Backend<OwnedBuf = Vec<u8>>,
        BE: 'static,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
        for<'a> BE::BufMut<'a>: HostDataMut,
        for<'a> BE: Backend<BufMut<'a> = &'a mut [u8], BufRef<'a> = &'a [u8]>,
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
        R: GLWEInfos,
        K: GGSWInfos,
    {
        self.glwe_blind_rotation_tmp_bytes(res_infos, k_infos) + GLWE::<Vec<u8>>::bytes_of_from_infos(res_infos)
    }

    #[allow(clippy::too_many_arguments)]
    fn scalar_to_ggsw_blind_rotation<R, K>(
        &self,
        res: &mut R,
        test_vector: &ScalarZnx<&[u8]>,
        fhe_uint: &K,
        sign: bool,
        bit_rsh: usize,
        bit_mask: usize,
        bit_lsh: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GGSWToBackendMut<BE> + GGSWAtViewMut<BE> + GGSWInfos,
        K: GetGGSWBit<BE>,
        BE: Backend<OwnedBuf = Vec<u8>>,
        BE: 'static,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
        for<'a> BE::BufMut<'a>: HostDataMut,
        for<'a> BE: Backend<BufMut<'a> = &'a mut [u8], BufRef<'a> = &'a [u8]>,
    {
        let base2k: usize = res.base2k().into();
        let dsize: usize = res.dsize().into();

        // TODO(device): this helper still stages a host-owned GLWE row before
        // calling backend-generic blind rotation.
        let mut tmp_glwe: GLWE<BE::OwnedBuf> = self.glwe_alloc_from_infos(res);
        let mut scratch_1 = scratch.borrow();
        let test_vector_backend: ScalarZnx<BE::OwnedBuf> =
            ScalarZnx::from_data(BE::from_host_bytes(test_vector.data), test_vector.n(), test_vector.cols());

        for col in 0..(res.rank() + 1).into() {
            for row in 0..res.dnum().into() {
                tmp_glwe.data_mut().zero();
                {
                    let mut tmp_glwe_data =
                        <poulpy_hal::layouts::VecZnx<Vec<u8>> as VecZnxToBackendMut<BE>>::to_backend_mut(tmp_glwe.data_mut());
                    self.vec_znx_add_scalar_assign_backend(
                        &mut tmp_glwe_data,
                        col,
                        (dsize - 1) + row * dsize,
                        &<ScalarZnx<BE::OwnedBuf> as ScalarZnxToBackendRef<BE>>::to_backend_ref(&test_vector_backend),
                        0,
                    );
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

impl<BE: Backend<OwnedBuf = Vec<u8>>> GLWEBlindRotation<BE> for Module<BE>
where
    Self: GLWECopy<BE> + GLWERotate<BE> + Cmux<BE>,
    for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
    for<'a> BE::BufMut<'a>: HostDataMut,
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
pub trait GLWEBlindRotation<BE: Backend<OwnedBuf = Vec<u8>>>
where
    Self: GLWECopy<BE> + GLWERotate<BE> + Cmux<BE> + ModuleCoreAlloc<OwnedBuf = Vec<u8>>,
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
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        K: GetGGSWBit<BE>,
        BE: Backend<OwnedBuf = Vec<u8>>,
        BE: 'static,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
        for<'a> BE::BufMut<'a>: HostDataMut,
        for<'a> BE: Backend<BufMut<'a> = &'a mut [u8], BufRef<'a> = &'a [u8]>,
    {
        // TODO(device): this ping-pong helper still relies on a host-owned
        // temporary ciphertexts for both ping-pong branches.
        let mut res_cur: GLWE<BE::OwnedBuf> = self.glwe_alloc_from_infos(res);
        let mut tmp_res: GLWE<BE::OwnedBuf> = self.glwe_alloc_from_infos(res);
        let mut scratch_1 = scratch.borrow();
        self.glwe_copy(&mut res_cur, res);

        // a_is_cur = true  => source is `res_cur`, dest is `tmp_res`
        // a_is_cur = false => source is `tmp_res`, dest is `res_cur`
        let mut a_is_cur: bool = true;

        for i in 0..bit_mask {
            if a_is_cur {
                match sign {
                    true => self.glwe_rotate(1 << (i + bit_lsh), &mut tmp_res, &res_cur),
                    false => self.glwe_rotate(-1 << (i + bit_lsh), &mut tmp_res, &res_cur),
                }

                let bit = value.get_bit(i + bit_rsh);
                self.cmux_assign(&mut tmp_res, &res_cur, bit, &mut scratch_1.borrow());
            } else {
                match sign {
                    true => self.glwe_rotate(1 << (i + bit_lsh), &mut res_cur, &tmp_res),
                    false => self.glwe_rotate(-1 << (i + bit_lsh), &mut res_cur, &tmp_res),
                }

                let bit = value.get_bit(i + bit_rsh);
                self.cmux_assign(&mut res_cur, &tmp_res, bit, &mut scratch_1.borrow());
            }

            // ping-pong roles for next iter
            a_is_cur = !a_is_cur;
        }

        let final_res: &GLWE<BE::OwnedBuf> = if a_is_cur { &res_cur } else { &tmp_res };
        self.glwe_copy(res, final_res);
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
        R: GLWEToBackendMut<BE> + GLWEInfos,
        A: GLWEToBackendRef<BE>,
        K: GetGGSWBit<BE>,
        BE: Backend<OwnedBuf = Vec<u8>>,
        BE: 'static,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
        for<'a> BE::BufMut<'a>: HostDataMut,
        for<'a> BE: Backend<BufMut<'a> = &'a mut [u8], BufRef<'a> = &'a [u8]>,
    {
        self.glwe_copy(res, a);
        self.glwe_blind_rotation_assign(res, fhe_uint, sign, bit_rsh, bit_mask, bit_lsh, scratch);
    }
}
