//! GLWE trace operation (sum of Galois automorphisms).
//!
//! The trace maps a GLWE ciphertext encrypting a polynomial `m(X)` to one
//! encrypting the sum of its Galois conjugates:
//!
//! `Trace(ct) = sum_{i in S} phi_i(ct)`
//!
//! where `phi_i` are the Galois automorphisms `X -> X^{g^i}`.
//! This is the dual operation of slot packing: it projects a ciphertext
//! onto a smaller subspace of plaintext slots, effectively replicating
//! a single slot value across multiple positions.
//!
//! The `skip` parameter controls how many initial automorphism levels
//! are skipped, allowing partial traces that project onto larger subspaces.
//!
//! Requires automorphism keys indexed by the Galois elements returned
//! from [`GLWETrace::glwe_trace_galois_elements`].

use crate::api::GLWEBytesOf;
use poulpy_hal::{
    api::ModuleLogN,
    layouts::{Backend, CyclotomicOrder, GaloisElement, ScratchArena, galois_element},
};

use crate::{
    GLWEAutomorphism, GLWECopy, GLWENormalize, GLWEShift, ScratchArenaTakeCore,
    layouts::{
        Base2K, GGLWEInfos, GLWEAutomorphismKeyHelper, GLWEAutomorphismKeyLayoutHelper, GLWEInfos, GLWELayout, GLWEToBackendMut,
        GLWEToBackendRef, GetGaloisElement, LWEInfos, WithEffectiveDsize, prepared::GGLWEPreparedToBackendRef,
    },
};

#[inline(always)]
pub fn trace_galois_elements(log_n: usize, cyclotomic_order: i64) -> Vec<i64> {
    (0..log_n)
        .map(|i| {
            if i == 0 {
                -1
            } else {
                galois_element(1 << (i - 1), cyclotomic_order)
            }
        })
        .collect()
}

fn trace_assign_internal<M, K, H, R, BE: Backend>(
    module: &M,
    res: &mut R,
    skip: usize,
    keys: &H,
    scratch: &mut ScratchArena<'_, BE>,
) where
    M: GLWEBytesOf<BE>
        + ModuleLogN
        + GaloisElement
        + GLWEAutomorphism<BE>
        + GLWEShift<BE>
        + GLWECopy<BE>
        + CyclotomicOrder
        + GLWENormalize<BE>
        + GLWETraceDefault<BE>
        + ?Sized,
    K: GGLWEPreparedToBackendRef<BE> + GetGaloisElement + GGLWEInfos,
    H: GLWEAutomorphismKeyHelper<K> + GLWEAutomorphismKeyLayoutHelper<K>,
    R: GLWEToBackendMut<BE> + GLWEInfos,
{
    let log_n: usize = module.log_n();
    assert_eq!(res.n(), module.n() as u32);
    assert!(skip <= log_n);
    assert!(
        scratch.available() >= module.glwe_trace_assign_tmp_bytes_default(res, keys),
        "scratch.available(): {} < GLWETrace::glwe_trace_assign_tmp_bytes: {}",
        scratch.available(),
        module.glwe_trace_assign_tmp_bytes_default(res, keys)
    );

    // Keys may differ per rotation; only the radix is common, so it is read off
    // the first one rather than from a layout shared by all.
    let Some(first) = glwe_trace_defaults_impl::trace_rotations(module, skip).next() else {
        return;
    };
    let (first_key, _) = keys
        .get_automorphism_key_for(first, res.k())
        .unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(first_key.n(), module.n() as u32);
    assert_eq!(first_key.rank_in(), res.rank());
    assert_eq!(first_key.rank_out(), res.rank());
    let key_base2k: Base2K = first_key.base2k();

    if res.base2k() != key_base2k {
        let res_conv_layout = GLWELayout {
            n: module.n().into(),
            base2k: key_base2k,
            k: res.k(),
            rank: res.rank(),
        };
        let scratch_local = scratch.borrow();
        let (mut res_conv, scratch_1) = scratch_local.take_glwe_scratch(&res_conv_layout);
        let mut scratch_1 = scratch_1;

        scratch_1 = scratch_1.apply_mut(|scratch| {
            module.glwe_normalize(&mut res_conv, res, scratch);
        });

        scratch_1 = scratch_1.apply_mut(|scratch| {
            trace_assign_internal::<M, K, H, _, BE>(module, &mut res_conv, skip, keys, scratch);
        });

        scratch_1.apply_mut(|scratch| {
            module.glwe_normalize(res, &res_conv, scratch);
        });
        return;
    }

    for p in glwe_trace_defaults_impl::trace_rotations(module, skip) {
        let (key, effective_dsize) = keys.get_automorphism_key_for(p, res.k()).unwrap_or_else(|e| panic!("{e}"));
        module.glwe_rsh(1, res, scratch);
        module.glwe_automorphism_add_assign(res, &key.with_dsize(effective_dsize), scratch);
    }
}

#[doc(hidden)]
pub trait GLWETraceDefault<BE: Backend> {
    fn glwe_trace_assign_tmp_bytes_default<A, L, H>(&self, a_infos: &A, keys: &H) -> usize
    where
        A: GLWEInfos,
        L: GGLWEInfos,
        H: GLWEAutomorphismKeyLayoutHelper<L>;

    fn glwe_trace_galois_elements_default(&self) -> Vec<i64>;

    fn glwe_trace_tmp_bytes_default<R, A, L, H>(&self, res_infos: &R, a_infos: &A, keys: &H) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        L: GGLWEInfos,
        H: GLWEAutomorphismKeyLayoutHelper<L>;

    fn glwe_trace_default<R, A, K, H>(&self, res: &mut R, skip: usize, a: &A, keys: &H, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        A: GLWEToBackendRef<BE> + GLWEInfos,
        K: GGLWEPreparedToBackendRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K> + GLWEAutomorphismKeyLayoutHelper<K>;

    fn glwe_trace_assign_default<R, K, H>(&self, res: &mut R, skip: usize, keys: &H, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        K: GGLWEPreparedToBackendRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K> + GLWEAutomorphismKeyLayoutHelper<K>;
}

/// Reference implementations of the [`GLWETraceDefault`] methods.
pub mod glwe_trace_defaults_impl {
    use super::*;

    /// Rotations the trace loop visits, in order.
    pub fn trace_rotations<M>(module: &M, skip: usize) -> impl Iterator<Item = i64> + use<'_, M>
    where
        M: ModuleLogN + GaloisElement + ?Sized,
    {
        (skip..module.log_n()).map(|i| if i == 0 { -1 } else { module.galois_element(1 << (i - 1)) })
    }

    /// Scratch bound as the maximum over the rotations, each resolved to its own
    /// key and effective `dsize`. A bare layout plans as one shape for all.
    pub fn glwe_trace_assign_tmp_bytes_default<BE, M, A, L, H>(module: &M, a_infos: &A, keys: &H) -> usize
    where
        BE: Backend,
        M: GLWEBytesOf<BE>
            + GLWETraceDefault<BE>
            + ModuleLogN
            + GaloisElement
            + GLWEAutomorphism<BE>
            + GLWEShift<BE>
            + GLWECopy<BE>
            + CyclotomicOrder
            + GLWENormalize<BE>,
        A: GLWEInfos,
        L: GGLWEInfos,
        H: GLWEAutomorphismKeyLayoutHelper<L>,
    {
        assert_eq!(module.n() as u32, a_infos.n());

        let mut worst: usize = 0;
        let mut key_base2k: Option<Base2K> = None;
        for p in trace_rotations(module, 0) {
            let (layout, effective_dsize) = keys
                .get_automorphism_key_layout_for(p, a_infos.k())
                .unwrap_or_else(|e| panic!("{e}"));
            assert_eq!(module.n() as u32, layout.n());
            key_base2k = Some(layout.base2k());
            worst = worst.max(module.glwe_automorphism_tmp_bytes(a_infos, a_infos, &layout.with_dsize(effective_dsize)));
        }
        let Some(key_base2k) = key_base2k else {
            return 0;
        };

        if a_infos.base2k() != key_base2k {
            let a_conv_infos: GLWELayout = GLWELayout {
                n: a_infos.n(),
                base2k: key_base2k,
                k: a_infos.k(),
                rank: a_infos.rank(),
            };
            let lvl_0: usize = module.glwe_bytes_of_from_infos(&a_conv_infos);
            let lvl_1: usize = module
                .glwe_normalize_tmp_bytes()
                .max(module.glwe_trace_assign_tmp_bytes_default(&a_conv_infos, keys));
            return lvl_0 + lvl_1;
        }

        module.glwe_shift_tmp_bytes().max(worst)
    }

    pub fn glwe_trace_galois_elements_default<BE, M>(module: &M) -> Vec<i64>
    where
        BE: Backend,
        M: ModuleLogN + CyclotomicOrder,
    {
        trace_galois_elements(module.log_n(), module.cyclotomic_order())
    }

    pub fn glwe_trace_tmp_bytes_default<BE, M, R, A, L, H>(module: &M, res_infos: &R, a_infos: &A, keys: &H) -> usize
    where
        BE: Backend,
        M: GLWEBytesOf<BE>
            + GLWETraceDefault<BE>
            + ModuleLogN
            + GaloisElement
            + GLWEAutomorphism<BE>
            + GLWEShift<BE>
            + GLWECopy<BE>
            + CyclotomicOrder
            + GLWENormalize<BE>,
        R: GLWEInfos,
        A: GLWEInfos,
        L: GGLWEInfos,
        H: GLWEAutomorphismKeyLayoutHelper<L>,
    {
        assert_eq!(module.n() as u32, res_infos.n());
        assert_eq!(module.n() as u32, a_infos.n());

        // Only the radix is shared by the rotations' keys.
        let key_base2k: Base2K = trace_rotations(module, 0)
            .next()
            .map(|p| {
                keys.get_automorphism_key_layout_for(p, a_infos.k().max(res_infos.k()))
                    .unwrap_or_else(|e| panic!("{e}"))
                    .0
                    .base2k()
            })
            .unwrap_or_else(|| a_infos.base2k());

        let tmp_infos: GLWELayout = GLWELayout {
            n: res_infos.n(),
            base2k: key_base2k,
            k: a_infos.k().max(res_infos.k()),
            rank: res_infos.rank(),
        };
        let lvl_0: usize = module.glwe_bytes_of_from_infos(&tmp_infos);
        let lvl_1: usize = if a_infos.base2k() == key_base2k {
            0
        } else {
            module.glwe_normalize_tmp_bytes()
        };
        let lvl_2: usize = module.glwe_trace_assign_tmp_bytes_default(&tmp_infos, keys);
        let lvl_3: usize = if res_infos.base2k() == key_base2k {
            0
        } else {
            module.glwe_bytes_of_from_infos(res_infos) + module.glwe_normalize_tmp_bytes()
        };

        lvl_0 + lvl_1.max(lvl_2).max(lvl_3)
    }

    pub fn glwe_trace_default<BE, M, R, A, K, H>(
        module: &M,
        res: &mut R,
        skip: usize,
        a: &A,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        BE: Backend,
        M: GLWEBytesOf<BE>
            + GLWETraceDefault<BE>
            + ModuleLogN
            + GaloisElement
            + GLWEAutomorphism<BE>
            + GLWEShift<BE>
            + GLWECopy<BE>
            + CyclotomicOrder
            + GLWENormalize<BE>,
        R: GLWEToBackendMut<BE> + GLWEInfos,
        A: GLWEToBackendRef<BE> + GLWEInfos,
        K: GGLWEPreparedToBackendRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K> + GLWEAutomorphismKeyLayoutHelper<K>,
    {
        assert!(
            scratch.available() >= module.glwe_trace_tmp_bytes_default(res, a, keys),
            "scratch.available(): {} < GLWETrace::glwe_trace_tmp_bytes: {}",
            scratch.available(),
            module.glwe_trace_tmp_bytes_default(res, a, keys)
        );

        // The radix is the only thing common to the rotations' keys. A trace that
        // visits none needs no key, and then no conversion either.
        let key_base2k: Base2K = glwe_trace_defaults_impl::trace_rotations(module, skip)
            .next()
            .map(|p| {
                keys.get_automorphism_key_layout_for(p, a.k().max(res.k()))
                    .unwrap_or_else(|e| panic!("{e}"))
                    .0
                    .base2k()
            })
            .unwrap_or_else(|| a.base2k());

        let scratch_local = scratch.borrow();
        let (mut tmp, scratch_1) = scratch_local.take_glwe_scratch(&GLWELayout {
            n: res.n(),
            base2k: key_base2k,
            k: a.k().max(res.k()),
            rank: res.rank(),
        });
        let mut scratch_1 = scratch_1;

        if a.base2k() == key_base2k {
            module.glwe_copy(&mut tmp, a);
        } else {
            scratch_1 = scratch_1.apply_mut(|scratch| {
                module.glwe_normalize(&mut tmp, a, scratch);
            });
        }

        {
            scratch_1 = scratch_1.apply_mut(|scratch| {
                trace_assign_internal::<M, K, H, _, BE>(module, &mut tmp, skip, keys, scratch);
            });
        }

        if res.base2k() == key_base2k {
            module.glwe_copy(res, &tmp);
        } else {
            let (mut res_out, scratch_2) = scratch_1.take_glwe_scratch(&res.glwe_layout());
            {
                scratch_2.apply_mut(|scratch| {
                    module.glwe_normalize(&mut res_out, &tmp, scratch);
                });
            }
            module.glwe_copy(res, &res_out);
        }
    }

    pub fn glwe_trace_assign_default<BE, M, R, K, H>(
        module: &M,
        res: &mut R,
        skip: usize,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        BE: Backend,
        M: GLWEBytesOf<BE>
            + GLWETraceDefault<BE>
            + ModuleLogN
            + GaloisElement
            + GLWEAutomorphism<BE>
            + GLWEShift<BE>
            + GLWECopy<BE>
            + CyclotomicOrder
            + GLWENormalize<BE>,
        R: GLWEToBackendMut<BE> + GLWEInfos,
        K: GGLWEPreparedToBackendRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K> + GLWEAutomorphismKeyLayoutHelper<K>,
    {
        trace_assign_internal::<M, K, H, _, BE>(module, res, skip, keys, scratch);
    }
}
