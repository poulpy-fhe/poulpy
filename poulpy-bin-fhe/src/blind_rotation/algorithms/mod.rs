mod cggi;

pub use cggi::*;

use std::mem::size_of;

use itertools::izip;
use poulpy_core::layouts::{GLWEInfos, GLWEToBackendMut, LWEInfos, LWEToBackendRef, ModuleCoreAlloc};
use poulpy_hal::{
    api::ModuleN,
    layouts::{Backend, Host, ScratchArena},
};

use crate::blind_rotation::{
    BlindRotationKey, BlindRotationKeyInfos, BlindRotationKeyPrepared, LookUpTableRotationDirection, LookupTable,
};

/// Marker trait for blind-rotation algorithm variants.
///
/// Implementors act as phantom types that bind a specific algorithm identity
/// to key and execution types.  This prevents accidental cross-algorithm key
/// usage at the type level.  Currently the only implementation is [`CGGI`].
pub trait BlindRotationAlgo: Sync {
    /// Allocates a zero-filled [`BlindRotationKey`] from a dimension descriptor.
    fn alloc_key<M, A>(module: &M, infos: &A) -> BlindRotationKey<M::OwnedBuf, Self, M::ZnxWord>
    where
        M: ModuleCoreAlloc + ModuleN,
        A: BlindRotationKeyInfos,
        Self: Sized;
}

/// Trait for executing the blind rotation algorithm.
///
/// Implemented for `Module<BE>` when the backend satisfies the required
/// polynomial-arithmetic trait bounds.  Callers should prefer the convenience
/// method on [`BlindRotationKeyPrepared::execute`] rather than calling these
/// methods directly.
pub trait BlindRotationExecute<BRA: BlindRotationAlgo, BE: Backend> {
    /// Returns the minimum scratch-space size in bytes required by
    /// [`blind_rotation_execute`][Self::blind_rotation_execute].
    ///
    /// `block_size` is the number of LWE coefficients processed per GGSW
    /// product (1 for standard CGGI, > 1 for block-binary).
    /// `extension_factor` is the number of LUT polynomials (must be a power
    /// of two; 1 for the classical single-polynomial LUT).
    fn blind_rotation_execute_tmp_bytes<G, B>(
        &self,
        block_size: usize,
        extension_factor: usize,
        glwe_infos: &G,
        brk_infos: &B,
    ) -> usize
    where
        G: GLWEInfos,
        B: BlindRotationKeyInfos;

    /// Evaluates the lookup table `lut` at the index encrypted in `lwe`,
    /// writing the result GLWE ciphertext into `res`.
    ///
    /// After a successful call, decrypting `res` and reading coefficient 0
    /// yields `lut(dec(lwe))` (up to rounding noise from the decomposition).
    ///
    /// # Panics
    ///
    /// Panics in debug mode if dimension mismatches are detected between `res`,
    /// `lwe`, `lut`, and `brk`.
    fn blind_rotation_execute<R, L>(
        &self,
        res: &mut R,
        lwe: &L,
        lut: &LookupTable<BE::OwnedBuf, BE::ZnxWord>,
        brk: &BlindRotationKeyPrepared<BE::OwnedBuf, BRA, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        L: LWEToBackendRef<BE> + LWEInfos;
}

impl<BRA: BlindRotationAlgo, BE: Backend> BlindRotationKeyPrepared<BE::OwnedBuf, BRA, BE> {
    /// Performs blind rotation using `self` as the bootstrapping key.
    ///
    /// Convenience wrapper around [`BlindRotationExecute::blind_rotation_execute`].
    pub fn execute<R, L, M>(
        &self,
        module: &M,
        res: &mut R,
        lwe: &L,
        lut: &LookupTable<BE::OwnedBuf, BE::ZnxWord>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        M: BlindRotationExecute<BRA, BE>,
        R: GLWEToBackendMut<BE> + GLWEInfos,
        L: LWEToBackendRef<BE> + LWEInfos,
    {
        module.blind_rotation_execute(res, lwe, lut, self, scratch);
    }
}

impl<BE: Backend, BRA: BlindRotationAlgo> BlindRotationKeyPrepared<BE::OwnedBuf, BRA, BE> {
    /// Returns the minimum scratch-space size in bytes required by
    /// [`BlindRotationKeyPrepared::execute`].
    ///
    /// See [`BlindRotationExecute::blind_rotation_execute_tmp_bytes`].
    pub fn execute_tmp_bytes<A, B, M>(
        module: &M,
        block_size: usize,
        extension_factor: usize,
        glwe_infos: &A,
        brk_infos: &B,
    ) -> usize
    where
        A: GLWEInfos,
        B: BlindRotationKeyInfos,
        M: BlindRotationExecute<BRA, BE>,
    {
        module.blind_rotation_execute_tmp_bytes(block_size, extension_factor, glwe_infos, brk_infos)
    }
}

/// Modulus-switches the LWE ciphertext coefficients into the range `[0, 2n)`,
/// writing them into `res`.
///
/// This function converts the multi-limb LWE representation (with base `2^k`)
/// to a single-integer representation modulo `2n` (or its signed extension
/// `[-n, n)` for the rotation index).  The conversion applies rounding to
/// reduce the probability of switching errors.
///
/// `rot_dir` controls the sign convention: `Left` negates all coefficients
/// (so that the rotation is `X^{-a_i}` in the accumulator), while `Right`
/// keeps them as-is.
///
/// # Arguments
///
/// - `n`: The extended domain size `2 × extension_factor × n_glwe`.
/// - `res`: Output slice of length `lwe.n() + 1` (b, a_0, …, a_{n-1}).
/// - `lwe`: The LWE ciphertext to switch.
/// - `rot_dir`: Rotation sign convention.
pub fn mod_switch_2n<BE, A>(n: usize, res: &mut [i64], lwe: &A, rot_dir: LookUpTableRotationDirection)
where
    A: LWEToBackendRef<BE> + LWEInfos,
    BE: Backend<Location = Host, ZnxWord = i64>,
{
    let lwe = lwe.to_backend_ref();
    let base2k: usize = lwe.base2k().into();

    let body_bytes = BE::bytes_of_vec_znx(lwe.body().n(), lwe.body().cols(), lwe.body().size());
    let mask_bytes = BE::bytes_of_vec_znx(lwe.mask().n(), lwe.mask().cols(), lwe.mask().size());
    let mut body_host = vec![0u8; body_bytes];
    let mut mask_host = vec![0u8; mask_bytes];
    BE::copy_view_to_host(&BE::region_ref(&lwe.body().data, 0, body_bytes), &mut body_host);
    BE::copy_view_to_host(&BE::region_ref(&lwe.mask().data, 0, mask_bytes), &mut mask_host);
    let word = |bytes: &[u8], index: usize| {
        let start = index * size_of::<i64>();
        i64::from_ne_bytes(bytes[start..start + size_of::<i64>()].try_into().unwrap())
    };

    let log2n: usize = usize::BITS as usize - (n - 1).leading_zeros() as usize + 1;

    res[0] = word(&body_host, 0);
    for (index, value) in res[1..].iter_mut().enumerate() {
        *value = word(&mask_host, index);
    }

    match rot_dir {
        LookUpTableRotationDirection::Left => {
            res.iter_mut().for_each(|x| *x = -*x);
        }
        LookUpTableRotationDirection::Right => {}
    }

    if base2k > log2n {
        let diff: usize = base2k - (log2n - 1); // additional -1 because we map to [-N/2, N/2) instead of [0, N)
        res.iter_mut().for_each(|x| {
            *x = div_round_by_pow2(x, diff);
        })
    } else {
        let rem: usize = base2k - (log2n % base2k);
        let size: usize = log2n.div_ceil(base2k);
        (1..size).for_each(|i| {
            let body_i = word(&body_host, i);
            let coeffs = std::iter::once(body_i).chain((0..lwe.mask().n()).map(|j| word(&mask_host, i * lwe.mask().n() + j)));
            if i == size - 1 && rem != base2k {
                let k_rem: usize = base2k - rem;
                izip!(coeffs, res.iter_mut()).for_each(|(x, y)| {
                    *y = (*y << k_rem) + (x >> rem);
                });
            } else {
                izip!(coeffs, res.iter_mut()).for_each(|(x, y)| {
                    *y = (*y << base2k) + x;
                });
            }
        })
    }
}

#[inline(always)]
fn div_round_by_pow2(x: &i64, k: usize) -> i64 {
    (x + (1 << (k - 1))) >> k
}
