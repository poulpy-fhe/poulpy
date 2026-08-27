use crate::api::GLWEBytesOf;
use std::collections::HashMap;

use poulpy_hal::{
    api::ModuleLogN,
    layouts::{Backend, GaloisElement, ScratchArena},
};

use crate::{
    GLWEAdd, GLWEAutomorphism, GLWECopy, GLWENormalize, GLWERotate, GLWEShift, GLWESub, GLWETrace,
    layouts::{
        GGLWEInfos, GLWEAutomorphismKeyHelper, GLWEAutomorphismKeyLayoutHelper, GLWEInfos, GLWEToBackendMut, GetGaloisElement,
        ModuleCoreAlloc, WithEffectiveDsize, prepared::GGLWEPreparedToBackendRef,
    },
};

#[allow(clippy::too_many_arguments)]
fn pack_internal<M, A, B, K, BE: Backend>(
    module: &M,
    a: &mut Option<&mut A>,
    b: &mut Option<&mut B>,
    i: usize,
    auto_key: &K,
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
    K: GGLWEPreparedToBackendRef<BE> + GetGaloisElement + GGLWEInfos,
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
            module.glwe_automorphism_assign(&mut tmp_b, auto_key, scratch);
            module.glwe_sub_assign(a, &tmp_b);
            module.glwe_normalize_assign(a, scratch);
            module.glwe_rotate_assign(t, a, scratch);
        } else {
            module.glwe_rsh(1, a, scratch);
            module.glwe_automorphism_add_assign(a, auto_key, scratch)
        }
    } else if let Some(b) = b.as_deref_mut() {
        let t: i64 = 1 << (b.n().log2() - i - 1);

        let b_layout = b.glwe_layout();
        let mut tmp_b = module.glwe_alloc_from_infos(&b_layout);
        module.glwe_rotate(t, &mut tmp_b, b);
        module.glwe_rsh(1, &mut tmp_b, scratch);
        module.glwe_automorphism_sub_negate(b, &tmp_b, auto_key, scratch)
    }
}

#[doc(hidden)]
pub trait GLWEPackingDefault<BE: Backend> {
    fn glwe_pack_galois_elements_default(&self) -> Vec<i64>;

    fn glwe_pack_tmp_bytes_default<R, L, H>(&self, res: &R, keys: &H) -> usize
    where
        R: GLWEInfos,
        L: GGLWEInfos,
        H: GLWEAutomorphismKeyLayoutHelper<L>;

    fn glwe_pack_default<R, A, K, H>(
        &self,
        res: &mut R,
        a: HashMap<usize, &mut A>,
        log_gap_out: usize,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        A: GLWEToBackendMut<BE> + GLWEInfos,
        K: GGLWEPreparedToBackendRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K> + GLWEAutomorphismKeyLayoutHelper<K>;
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

    pub fn glwe_pack_tmp_bytes_default<BE, M, R, L, H>(module: &M, res: &R, keys: &H) -> usize
    where
        BE: Backend,
        M: GLWEBytesOf<BE>
            + GLWEAutomorphism<BE>
            + GaloisElement
            + ModuleLogN
            + GLWERotate<BE>
            + GLWEShift<BE>
            + GLWENormalize<BE>
            + GLWETrace<BE>,
        R: GLWEInfos,
        L: GGLWEInfos,
        H: GLWEAutomorphismKeyLayoutHelper<L>,
    {
        assert_eq!(module.n() as u32, res.n());

        // Each rotation resolves its own key and effective dsize.
        let mut worst: usize = 0;
        for p in pack_rotations(module, 0) {
            let (layout, effective_dsize) = keys
                .get_automorphism_key_layout_for(p, res.k())
                .unwrap_or_else(|e| panic!("{e}"));
            assert_eq!(module.n() as u32, layout.n());
            worst = worst.max(module.glwe_automorphism_tmp_bytes(res, res, &layout.with_dsize(effective_dsize)));
        }

        let lvl_0: usize = module.glwe_bytes_of_from_infos(res);
        let lvl_1: usize = module
            .glwe_rotate_tmp_bytes()
            .max(module.glwe_shift_tmp_bytes())
            .max(module.glwe_normalize_tmp_bytes())
            .max(worst);

        (lvl_0 + lvl_1).max(module.glwe_trace_tmp_bytes(res, res, keys))
    }

    /// Rotations the packing loop visits, in order.
    pub fn pack_rotations<M>(module: &M, log_gap_out: usize) -> impl Iterator<Item = i64> + use<'_, M>
    where
        M: ModuleLogN + GaloisElement + ?Sized,
    {
        (0..module.log_n() - log_gap_out).map(|i| if i == 0 { -1 } else { module.galois_element(1 << (i - 1)) })
    }

    pub fn glwe_pack_default<BE, M, R, A, K, H>(
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
        K: GGLWEPreparedToBackendRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K> + GLWEAutomorphismKeyLayoutHelper<K>,
    {
        assert!(*a.keys().max().unwrap() < module.n());
        assert!(
            scratch.available() >= glwe_pack_tmp_bytes_default::<BE, _, _, _, _>(module, res, keys),
            "scratch.available(): {} < GLWEPacking::glwe_pack_tmp_bytes: {}",
            scratch.available(),
            glwe_pack_tmp_bytes_default::<BE, _, _, _, _>(module, res, keys)
        );

        let mut scratch_local = scratch.borrow();
        let log_n: usize = module.log_n();
        for i in 0..(log_n - log_gap_out) {
            let t: usize = (1 << log_n).min(1 << (log_n - 1 - i));

            let p: i64 = if i == 0 { -1 } else { module.galois_element(1 << (i - 1)) };
            let (key, effective_dsize) = keys.get_automorphism_key_for(p, res.k()).unwrap_or_else(|e| panic!("{e}"));
            let key = &key.with_dsize(effective_dsize);

            for j in 0..t {
                let mut lo: Option<&mut A> = a.remove(&j);
                let mut hi: Option<&mut A> = a.remove(&(j + t));

                scratch_local = scratch_local.apply_mut(|scratch| {
                    pack_internal(module, &mut lo, &mut hi, i, key, scratch);
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
