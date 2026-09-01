use crate::api::GLWEBytesOf;
use std::collections::HashMap;

use poulpy_hal::{
    api::ModuleLogN,
    layouts::{Backend, GaloisElement, ScratchArena},
};

use crate::{
    GLWEAdd, GLWEAutomorphism, GLWECopy, GLWENormalize, GLWERotate, GLWEShift, GLWESub, GLWETrace,
    layouts::{GGLWEInfos, GLWEInfos, GLWEToBackendMut, GetAutomorphismKey, LWEInfos, ModuleCoreAlloc},
};

#[allow(clippy::too_many_arguments)]
fn pack_internal<M, A, B, H, BE: Backend>(
    module: &M,
    a: &mut Option<&mut A>,
    b: &mut Option<&mut B>,
    i: usize,
    p: i64,
    keys: &H,
    scratch: &mut ScratchArena<'_, BE>,
) where
    M: GLWEBytesOf<BE>
        + GLWEAutomorphism<BE>
        + GLWERotate<BE>
        + GLWESub<BE>
        + GLWEShift<BE>
        + GLWEAdd<BE>
        + GLWENormalize<BE>
        + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = BE::ZnxWord>
        + ?Sized,
    A: GLWEToBackendMut<BE> + GLWEInfos,
    B: GLWEToBackendMut<BE> + GLWEInfos,
    H: GetAutomorphismKey<BE>,
{
    // Goal is to evaluate: a = a + b*X^t + phi(a - b*X^t))
    // We also use the identity: AUTO(a * X^t, g) = -X^t * AUTO(a, g)
    // where t = 2^(log_n - i - 1) and g = 5^{2^(i - 1)}
    if let Some(a) = a.as_deref_mut() {
        let t: i64 = 1 << (a.n().log2() - i - 1);

        if let Some(b) = b.as_deref_mut() {
            let a_layout = a.glwe_layout();
            let mut tmp_b = module.glwe_alloc_from_infos(&a_layout);
            module.glwe_rotate_assign(-t, a, scratch);
            module.glwe_sub(&mut tmp_b, a, b);
            module.glwe_rsh(1, &mut tmp_b, scratch);
            module.glwe_add_assign(a, b);
            module.glwe_rsh(1, a, scratch);
            module.glwe_normalize_assign(&mut tmp_b, scratch);
            let key = keys
                .get_automorphism_key(p, tmp_b.k())
                .unwrap_or_else(|e| panic!("pack rotation {p}: {e}"));
            module.glwe_automorphism_assign(&mut tmp_b, &key, scratch);
            module.glwe_sub_assign(a, &tmp_b);
            module.glwe_normalize_assign(a, scratch);
            module.glwe_rotate_assign(t, a, scratch);
        } else {
            module.glwe_rsh(1, a, scratch);
            let key = keys
                .get_automorphism_key(p, a.k())
                .unwrap_or_else(|e| panic!("pack rotation {p}: {e}"));
            module.glwe_automorphism_add_assign(a, &key, scratch)
        }
    } else if let Some(b) = b.as_deref_mut() {
        let t: i64 = 1 << (b.n().log2() - i - 1);

        let b_layout = b.glwe_layout();
        let mut tmp_b = module.glwe_alloc_from_infos(&b_layout);
        module.glwe_rotate(t, &mut tmp_b, b);
        module.glwe_rsh(1, &mut tmp_b, scratch);
        let key = keys
            .get_automorphism_key(p, tmp_b.k())
            .unwrap_or_else(|e| panic!("pack rotation {p}: {e}"));
        module.glwe_automorphism_sub_negate(b, &tmp_b, &key, scratch)
    }
}

#[doc(hidden)]
pub trait GLWEPackingDefault<BE: Backend> {
    fn glwe_pack_galois_elements_default(&self) -> Vec<i64>;

    fn glwe_pack_tmp_bytes_default<R, K>(&self, res: &R, key: &K) -> usize
    where
        R: GLWEInfos,
        K: GGLWEInfos;

    fn glwe_pack_default<R, A, H>(
        &self,
        res: &mut R,
        a: HashMap<usize, &mut A>,
        log_gap_out: usize,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        A: GLWEToBackendMut<BE> + GLWEInfos,
        H: GetAutomorphismKey<BE>;
}

/// Reference implementations of the [`GLWEPackingDefault`] methods.
pub mod glwe_packing_defaults_impl {
    use super::*;

    pub fn glwe_pack_galois_elements_default<BE, M>(module: &M) -> Vec<i64>
    where
        BE: Backend,
        M: GLWETrace<BE>,
    {
        module.glwe_trace_galois_elements()
    }

    pub fn glwe_pack_tmp_bytes_default<BE, M, R, K>(module: &M, res: &R, key: &K) -> usize
    where
        BE: Backend,
        M: GLWEBytesOf<BE>
            + GLWEAutomorphism<BE>
            + ModuleLogN
            + GLWERotate<BE>
            + GLWEShift<BE>
            + GLWENormalize<BE>
            + GLWETrace<BE>,
        R: GLWEInfos,
        K: GGLWEInfos,
    {
        assert_eq!(module.n() as u32, res.n());
        assert_eq!(module.n() as u32, key.n());

        let lvl_0: usize = module.glwe_bytes_of_from_infos(res);
        let lvl_1: usize = module
            .glwe_rotate_tmp_bytes()
            .max(module.glwe_shift_tmp_bytes())
            .max(module.glwe_normalize_tmp_bytes())
            .max(module.glwe_automorphism_tmp_bytes(res, res, key));

        (lvl_0 + lvl_1).max(module.glwe_trace_tmp_bytes(res, res, key))
    }

    pub fn glwe_pack_default<BE, M, R, A, H>(
        module: &M,
        res: &mut R,
        mut a: HashMap<usize, &mut A>,
        log_gap_out: usize,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        BE: Backend,
        M: GLWEBytesOf<BE>
            + GLWEAutomorphism<BE>
            + GaloisElement
            + ModuleLogN
            + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = BE::ZnxWord>
            + GLWERotate<BE>
            + GLWESub<BE>
            + GLWEShift<BE>
            + GLWEAdd<BE>
            + GLWENormalize<BE>
            + GLWECopy<BE>
            + GLWETrace<BE>,
        R: GLWEToBackendMut<BE> + GLWEInfos,
        A: GLWEToBackendMut<BE> + GLWEInfos,
        H: GetAutomorphismKey<BE>,
    {
        assert!(*a.keys().max().unwrap() < module.n());
        // Keys may differ per rotation; the bound is read off the first one, as
        // on any path sized from a single key layout.
        let key_infos = keys
            .get_automorphism_key(-1, res.k())
            .unwrap_or_else(|e| panic!("packing rotation -1: {e}"));
        assert!(
            scratch.available() >= glwe_pack_tmp_bytes_default::<BE, _, _, _>(module, res, &key_infos),
            "scratch.available(): {} < GLWEPacking::glwe_pack_tmp_bytes: {}",
            scratch.available(),
            glwe_pack_tmp_bytes_default::<BE, _, _, _>(module, res, &key_infos)
        );

        let mut scratch_local = scratch.borrow();
        let log_n: usize = module.log_n();
        for i in 0..(log_n - log_gap_out) {
            let t: usize = (1 << log_n).min(1 << (log_n - 1 - i));

            let p: i64 = if i == 0 { -1 } else { module.galois_element(1 << (i - 1)) };

            for j in 0..t {
                let mut lo: Option<&mut A> = a.remove(&j);
                let mut hi: Option<&mut A> = a.remove(&(j + t));

                scratch_local = scratch_local.apply_mut(|scratch| {
                    pack_internal(module, &mut lo, &mut hi, i, p, keys, scratch);
                });

                if let Some(lo) = lo {
                    a.insert(j, lo);
                } else if let Some(hi) = hi {
                    a.insert(j, hi);
                }
            }
        }

        scratch_local.apply_mut(|scratch| {
            module.glwe_trace(res, log_n - log_gap_out, *a.get_mut(&0).unwrap(), keys, scratch);
        });
    }
}
