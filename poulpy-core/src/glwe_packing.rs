use std::collections::HashMap;

use poulpy_hal::{
    api::{ModuleLogN, ScratchAvailable},
    layouts::{Backend, GaloisElement, Module, Scratch},
};

pub use crate::api::GLWEPacking;
use crate::{
    GLWEAdd, GLWEAutomorphism, GLWECopy, GLWENormalize, GLWERotate, GLWEShift, GLWESub, GLWETrace, ScratchTakeCore,
    layouts::{GGLWEInfos, GGLWEPreparedToRef, GLWE, GLWEAutomorphismKeyHelper, GLWEInfos, GLWEToMut, GetGaloisElement},
};

#[allow(clippy::too_many_arguments)]
fn pack_internal<M, A, B, K, BE: Backend>(
    module: &M,
    a: &mut Option<&mut A>,
    b: &mut Option<&mut B>,
    i: usize,
    auto_key: &K,
    scratch: &mut Scratch<BE>,
) where
    M: GLWEAutomorphism<BE> + GLWERotate<BE> + GLWESub + GLWEShift<BE> + GLWEAdd + GLWENormalize<BE> + ?Sized,
    A: GLWEToMut + GLWEInfos,
    B: GLWEToMut + GLWEInfos,
    K: GGLWEPreparedToRef<BE> + GetGaloisElement + GGLWEInfos,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    // Goal is to evaluate: a = a + b*X^t + phi(a - b*X^t))
    // We also use the identity: AUTO(a * X^t, g) = -X^t * AUTO(a, g)
    // where t = 2^(log_n - i - 1) and g = 5^{2^(i - 1)}
    // Different cases for wether a and/or b are zero.
    //
    // Implicite RSH without modulus switch, introduces extra I(X) * Q/2 on decryption.
    // Necessary so that the scaling of the plaintext remains constant.
    // It however is ok to do so here because coefficients are eventually
    // either mapped to garbage or twice their value which vanishes I(X)
    // since 2*(I(X) * Q/2) = I(X) * Q = 0 mod Q.
    if let Some(a) = a.as_deref_mut() {
        let t: i64 = 1 << (a.n().log2() - i - 1);

        if let Some(b) = b.as_deref_mut() {
            let (mut tmp_b, scratch_1) = scratch.take_glwe(a);

            // a = a * X^-t
            module.glwe_rotate_assign(-t, a, scratch_1);

            // tmp_b = a * X^-t - b
            module.glwe_sub(&mut tmp_b, a, b);
            module.glwe_rsh(1, &mut tmp_b, scratch_1);

            // a = a * X^-t + b
            module.glwe_add_assign(a, b);
            module.glwe_rsh(1, a, scratch_1);

            module.glwe_normalize_assign(&mut tmp_b, scratch_1);

            // tmp_b = phi(a * X^-t - b)
            module.glwe_automorphism_assign(&mut tmp_b, auto_key, scratch_1);

            // a = a * X^-t + b - phi(a * X^-t - b)
            module.glwe_sub_assign(a, &tmp_b);
            module.glwe_normalize_assign(a, scratch_1);

            // a = a + b * X^t - phi(a * X^-t - b) * X^t
            //   = a + b * X^t - phi(a * X^-t - b) * - phi(X^t)
            //   = a + b * X^t + phi(a - b * X^t)
            module.glwe_rotate_assign(t, a, scratch_1);
        } else {
            module.glwe_rsh(1, a, scratch);
            // a = a + phi(a)
            module.glwe_automorphism_add_assign(a, auto_key, scratch);
        }
    } else if let Some(b) = b.as_deref_mut() {
        let t: i64 = 1 << (b.n().log2() - i - 1);

        let (mut tmp_b, scratch_1) = scratch.take_glwe(b);
        module.glwe_rotate(t, &mut tmp_b, b);
        module.glwe_rsh(1, &mut tmp_b, scratch_1);

        // a = (b* X^t - phi(b* X^t))
        module.glwe_automorphism_sub_negate(b, &tmp_b, auto_key, scratch_1);
    }
}

#[doc(hidden)]
pub trait GLWEPackingDefault<BE: Backend>
where
    Self: GLWEAutomorphism<BE>
        + GaloisElement
        + ModuleLogN
        + GLWERotate<BE>
        + GLWESub
        + GLWEShift<BE>
        + GLWEAdd
        + GLWENormalize<BE>
        + GLWECopy
        + GLWETrace<BE>,
{
    fn glwe_pack_galois_elements_default(&self) -> Vec<i64> {
        self.glwe_trace_galois_elements()
    }

    fn glwe_pack_tmp_bytes_default<R, K>(&self, res: &R, key: &K) -> usize
    where
        R: GLWEInfos,
        K: GGLWEInfos,
    {
        assert_eq!(self.n() as u32, res.n());
        assert_eq!(self.n() as u32, key.n());

        let lvl_0: usize = GLWE::<Vec<u8>>::bytes_of_from_infos(res);
        let lvl_1: usize = self
            .glwe_rotate_tmp_bytes()
            .max(self.glwe_shift_tmp_bytes())
            .max(self.glwe_normalize_tmp_bytes())
            .max(self.glwe_automorphism_tmp_bytes(res, res, key));

        (lvl_0 + lvl_1).max(self.glwe_trace_tmp_bytes(res, res, key))
    }

    fn glwe_pack_default<R, A, K, H>(
        &self,
        res: &mut R,
        mut a: HashMap<usize, &mut A>,
        log_gap_out: usize,
        keys: &H,
        scratch: &mut Scratch<BE>,
    ) where
        R: GLWEToMut + GLWEInfos,
        A: GLWEToMut + GLWEInfos,
        K: GGLWEPreparedToRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        assert!(*a.keys().max().unwrap() < self.n());
        let key_infos = keys.automorphism_key_infos();
        assert!(
            scratch.available() >= self.glwe_pack_tmp_bytes_default(res, &key_infos),
            "scratch.available(): {} < GLWEPacking::glwe_pack_tmp_bytes: {}",
            scratch.available(),
            self.glwe_pack_tmp_bytes_default(res, &key_infos)
        );

        let log_n: usize = self.log_n();

        for i in 0..(log_n - log_gap_out) {
            let t: usize = (1 << log_n).min(1 << (log_n - 1 - i));

            let key: &K = if i == 0 {
                keys.get_automorphism_key(-1).unwrap()
            } else {
                keys.get_automorphism_key(self.galois_element(1 << (i - 1))).unwrap()
            };

            for j in 0..t {
                let mut lo: Option<&mut A> = a.remove(&j);
                let mut hi: Option<&mut A> = a.remove(&(j + t));

                pack_internal(self, &mut lo, &mut hi, i, key, scratch);

                if let Some(lo) = lo {
                    a.insert(j, lo);
                } else if let Some(hi) = hi {
                    a.insert(j, hi);
                }
            }
        }

        self.glwe_trace(res, log_n - log_gap_out, *a.get(&0).unwrap(), keys, scratch);
    }
}

impl<BE: Backend> GLWEPackingDefault<BE> for Module<BE>
where
    Self: GLWEAutomorphism<BE>
        + GaloisElement
        + ModuleLogN
        + GLWERotate<BE>
        + GLWESub
        + GLWEShift<BE>
        + GLWEAdd
        + GLWENormalize<BE>
        + GLWECopy
        + GLWETrace<BE>,
    Scratch<BE>: ScratchTakeCore<BE>,
{
}
