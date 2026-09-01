use std::collections::HashMap;

use poulpy_core::{
    GLWECopy, GLWEZero, ScratchArenaTakeCore,
    layouts::{GGSWInfos, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef},
};
use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use crate::bdd_arithmetic::{Cmux, GetGGSWBit, UnsignedInteger};
use poulpy_core::GLWEBytesOf;
use poulpy_core::layouts::prepared::GGSWPreparedToBackendRef;

impl<T: UnsignedInteger, BE: Backend<ZnxWord = i64> + 'static> GLWEBlindSelection<T, BE> for Module<BE> where
    Self: GLWECopy<BE> + Cmux<BE> + GLWEZero<BE>
{
}

/// Oblivious selection of one GLWE ciphertext from an encrypted-indexed map.
///
/// Given a `HashMap` of GLWE ciphertexts keyed by integer index and a set of
/// GGSW ciphertexts encoding a selection index `k`, selects:
///
/// ```text
/// res = a[(k >> bit_rsh) % 2^bit_mask]
/// ```
///
/// The selection is performed via a binary-tree reduction of CMux gates over the
/// `bit_mask` most-significant bits of the selected index sub-field, traversing
/// from MSB to LSB.  Indices absent from the map are treated as encryptions of
/// zero.
pub trait GLWEBlindSelection<T: UnsignedInteger, BE: Backend + 'static>
where
    Self: GLWECopy<BE> + Cmux<BE> + GLWEZero<BE>,
{
    /// Returns the minimum scratch-space size in bytes required by
    /// [`glwe_blind_selection`][Self::glwe_blind_selection].
    fn glwe_blind_selection_tmp_bytes<R, K>(&self, res_infos: &R, k_infos: &K) -> usize
    where
        Self: GLWEBytesOf<BE>,
        R: GLWEInfos,
        K: GGSWInfos,
    {
        self.cmux_tmp_bytes(res_infos, res_infos, k_infos) + self.glwe_bytes_of_from_infos(res_infos)
    }

    #[allow(clippy::too_many_arguments)]
    fn glwe_blind_selection<R, A, K>(
        &self,
        res: &mut R,
        mut a: HashMap<usize, &mut A>,
        fhe_uint: &K,
        bit_rsh: usize,
        bit_mask: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        A: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + GLWEInfos,
        K: GetGGSWBit<BE> + 'static,
    {
        assert!(bit_rsh + bit_mask <= T::BITS as usize);
        let (mut zero, mut scratch) = scratch.borrow().take_glwe_scratch(res);

        for i in 0..bit_mask {
            let t: usize = 1 << (bit_mask - i - 1);

            let bit = fhe_uint.get_bit(bit_rsh + bit_mask - i - 1); // MSB -> LSB traversal

            for j in 0..t {
                let hi: Option<&mut A> = a.remove(&j);
                let lo: Option<&mut A> = a.remove(&(j + t));

                match (lo, hi) {
                    (Some(lo), Some(hi)) => {
                        self.cmux_assign(lo, hi, &bit.to_backend_ref(), &mut scratch.borrow());
                        a.insert(j, lo);
                    }

                    (Some(lo), None) => {
                        self.glwe_zero(&mut zero);
                        self.cmux_assign(lo, &zero, &bit.to_backend_ref(), &mut scratch.borrow());
                        a.insert(j, lo);
                    }

                    (None, Some(hi)) => {
                        self.glwe_zero(&mut zero);
                        self.cmux_assign(&mut zero, hi, &bit.to_backend_ref(), &mut scratch.borrow());
                        self.glwe_copy(hi, &zero);
                        a.insert(j, hi);
                    }

                    (None, None) => {
                        // No low or high branch — nothing to insert
                        // leave empty; future iterations will combine actual ciphertexts
                    }
                }
            }
        }

        let out: Option<&mut A> = a.remove(&0);

        if let Some(out) = out {
            self.glwe_copy(res, out);
        } else {
            self.glwe_zero(res);
        }
    }
}
