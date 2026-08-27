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
    default::{keyswitching::GLWEKeyswitchInternal, operations::GLWENormalizeDefault},
    error::Result,
    layouts::{
        Base2K, GGLWEInfos, GGLWELayout, GLWEAutomorphismKeyHelper, GLWEAutomorphismKeyLayoutHelper, GLWEAutomorphismKeyMap,
        GLWEInfos, GLWELayout, GLWEToBackendMut, GLWEToBackendRef, GetGaloisElement, LWEInfos,
        prepared::GGLWEPreparedToBackendRef,
    },
    oep::{GLWEAutomorphismDefault, GLWEKeyswitchDefault},
};
use poulpy_hal::api::{
    VecZnxBigAddSmallAssign, VecZnxBigAutomorphismAssign, VecZnxBigBytesOf, VecZnxBigNormalize, VecZnxDftBytesOf, VecZnxIdftApply,
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
    H: GLWEAutomorphismKeyMap<K, BE>,
    R: GLWEToBackendMut<BE> + GLWEInfos,
{
    let ksk_infos: &GGLWELayout = &keys.automorphism_key_infos();
    let log_n: usize = module.log_n();

    assert_eq!(res.n(), module.n() as u32);
    assert_eq!(ksk_infos.n(), module.n() as u32);
    assert!(skip <= log_n);
    assert_eq!(ksk_infos.rank_in(), res.rank());
    assert_eq!(ksk_infos.rank_out(), res.rank());
    assert!(
        scratch.available() >= module.glwe_trace_assign_tmp_bytes_default(res, ksk_infos),
        "scratch.available(): {} < GLWETrace::glwe_trace_assign_tmp_bytes: {}",
        scratch.available(),
        module.glwe_trace_assign_tmp_bytes_default(res, ksk_infos)
    );

    if res.base2k() != ksk_infos.base2k() {
        let res_conv_layout = GLWELayout {
            n: module.n().into(),
            base2k: ksk_infos.base2k(),
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

    for i in skip..log_n {
        let p: i64 = if i == 0 { -1 } else { module.galois_element(1 << (i - 1)) };
        module.glwe_rsh(1, res, scratch);
        if let Some(key) = keys.get_automorphism_key(p) {
            module.glwe_automorphism_add_assign(res, key, scratch);
        } else {
            panic!("keys[{p}] is empty")
        }
    }
}

#[doc(hidden)]
pub trait GLWETraceDefault<BE: Backend> {
    fn glwe_trace_assign_tmp_bytes_default<A, K>(&self, a_infos: &A, key_infos: &K) -> usize
    where
        A: GLWEInfos,
        K: GGLWEInfos;

    fn glwe_trace_galois_elements_default(&self) -> Vec<i64>;

    fn glwe_trace_tmp_bytes_default<R, A, K>(&self, res_infos: &R, a_infos: &A, key_infos: &K) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        K: GGLWEInfos;

    fn glwe_trace_default<R, A, K, H>(&self, res: &mut R, skip: usize, a: &A, keys: &H, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        A: GLWEToBackendRef<BE> + GLWEInfos,
        K: GGLWEPreparedToBackendRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyMap<K, BE>;

    fn glwe_trace_assign_default<R, K, H>(&self, res: &mut R, skip: usize, keys: &H, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        K: GGLWEPreparedToBackendRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyMap<K, BE>;
}

/// Reference implementations of the [`GLWETraceDefault`] methods.
pub mod glwe_trace_defaults_impl {
    use super::*;

    pub fn glwe_trace_assign_tmp_bytes_default<BE, M, A, K>(module: &M, a_infos: &A, key_infos: &K) -> usize
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
        K: GGLWEInfos,
    {
        assert_eq!(module.n() as u32, a_infos.n());
        assert_eq!(module.n() as u32, key_infos.n());

        if a_infos.base2k() != key_infos.base2k() {
            let a_conv_infos: GLWELayout = GLWELayout {
                n: a_infos.n(),
                base2k: key_infos.base2k(),
                k: a_infos.k(),
                rank: a_infos.rank(),
            };
            let lvl_0: usize = module.glwe_bytes_of_from_infos(&a_conv_infos);
            let lvl_1: usize = module
                .glwe_normalize_tmp_bytes()
                .max(module.glwe_trace_assign_tmp_bytes_default(&a_conv_infos, key_infos));
            return lvl_0 + lvl_1;
        }

        module
            .glwe_shift_tmp_bytes()
            .max(module.glwe_automorphism_tmp_bytes(a_infos, a_infos, key_infos))
    }

    /// Rotations `glwe_trace_assign` visits, in loop order.
    pub fn trace_rotations<M>(module: &M, skip: usize) -> impl Iterator<Item = i64> + use<'_, M>
    where
        M: ModuleLogN + GaloisElement,
    {
        (skip..module.log_n()).map(|i| if i == 0 { -1 } else { module.galois_element(1 << (i - 1)) })
    }

    /// Scratch bound of [`glwe_trace_selected_assign`], as the maximum over the
    /// rotations actually visited.
    ///
    /// Replaces the assumption that every automorphism key shares one physical
    /// layout: each rotation resolves its own key and effective `dsize`.
    pub fn glwe_trace_selected_assign_tmp_bytes<BE, M, A, L, H>(module: &M, a_infos: &A, skip: usize, keys: &H) -> Result<usize>
    where
        BE: Backend,
        M: GLWEBytesOf<BE>
            + GLWEAutomorphismDefault<BE>
            + ModuleLogN
            + GaloisElement
            + GLWEShift<BE>
            + CyclotomicOrder
            + GLWENormalize<BE>,
        A: GLWEInfos,
        L: GGLWEInfos,
        H: GLWEAutomorphismKeyLayoutHelper<L>,
    {
        assert_eq!(module.n() as u32, a_infos.n());

        let mut worst: usize = 0;
        let mut key_base2k: Option<Base2K> = None;
        for p in trace_rotations(module, skip) {
            let (layout, effective_dsize) = keys.get_automorphism_key_layout_for(p, a_infos.k())?;
            assert_eq!(module.n() as u32, layout.n());
            key_base2k = Some(layout.base2k());
            worst = worst.max(module.glwe_automorphism_selected_tmp_bytes_default(a_infos, a_infos, layout, effective_dsize));
        }
        let Some(key_base2k) = key_base2k else {
            return Ok(0);
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
                .max(glwe_trace_selected_assign_tmp_bytes::<BE, _, _, _, _>(
                    module,
                    &a_conv_infos,
                    skip,
                    keys,
                )?);
            return Ok(lvl_0 + lvl_1);
        }

        Ok(module.glwe_shift_tmp_bytes().max(worst))
    }

    /// `res <- trace(res)`, resolving each rotation's key and effective `dsize`
    /// from `keys` at the exact precision of `res`.
    pub fn glwe_trace_selected_assign<BE, M, R, K, H>(
        module: &M,
        res: &mut R,
        skip: usize,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        BE: Backend,
        M: GLWEBytesOf<BE>
            + GLWEAutomorphismDefault<BE>
            + GLWEKeyswitchDefault<BE>
            + GLWEKeyswitchInternal<BE>
            + GLWENormalizeDefault<BE>
            + ModuleLogN
            + GaloisElement
            + GLWEShift<BE>
            + GLWECopy<BE>
            + CyclotomicOrder
            + GLWENormalize<BE>
            + VecZnxBigAutomorphismAssign<BE>
            + VecZnxBigAddSmallAssign<BE>
            + VecZnxBigBytesOf
            + VecZnxBigNormalize<BE>
            + VecZnxDftBytesOf
            + VecZnxIdftApply<BE>,
        R: GLWEToBackendMut<BE> + GLWEInfos,
        K: GGLWEPreparedToBackendRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K>,
    {
        assert_eq!(res.n(), module.n() as u32);
        assert!(skip <= module.log_n());

        for p in trace_rotations(module, skip) {
            let (key, effective_dsize) = keys.get_automorphism_key_for(p, res.k())?;
            assert_eq!(key.rank_in(), res.rank());
            assert_eq!(key.rank_out(), res.rank());
            assert_eq!(res.base2k(), key.base2k(), "base2k conversion is the caller's job");
            module.glwe_rsh(1, res, scratch);
            crate::default::automorphism::glwe::glwe_automorphism_add_assign_selected_default(
                module,
                res,
                key,
                effective_dsize,
                &mut scratch.borrow(),
            );
        }
        Ok(())
    }

    pub fn glwe_trace_galois_elements_default<BE, M>(module: &M) -> Vec<i64>
    where
        BE: Backend,
        M: ModuleLogN + CyclotomicOrder,
    {
        trace_galois_elements(module.log_n(), module.cyclotomic_order())
    }

    pub fn glwe_trace_tmp_bytes_default<BE, M, R, A, K>(module: &M, res_infos: &R, a_infos: &A, key_infos: &K) -> usize
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
        K: GGLWEInfos,
    {
        assert_eq!(module.n() as u32, res_infos.n());
        assert_eq!(module.n() as u32, a_infos.n());
        assert_eq!(module.n() as u32, key_infos.n());

        let tmp_infos: GLWELayout = GLWELayout {
            n: res_infos.n(),
            base2k: key_infos.base2k(),
            k: a_infos.k().max(res_infos.k()),
            rank: res_infos.rank(),
        };
        let lvl_0: usize = module.glwe_bytes_of_from_infos(&tmp_infos);
        let lvl_1: usize = if a_infos.base2k() == key_infos.base2k() {
            0
        } else {
            module.glwe_normalize_tmp_bytes()
        };
        let lvl_2: usize = module.glwe_trace_assign_tmp_bytes_default(&tmp_infos, key_infos);
        let lvl_3: usize = if res_infos.base2k() == key_infos.base2k() {
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
        H: GLWEAutomorphismKeyMap<K, BE>,
    {
        let atk_layout: &GGLWELayout = &keys.automorphism_key_infos();
        assert!(
            scratch.available() >= module.glwe_trace_tmp_bytes_default(res, a, atk_layout),
            "scratch.available(): {} < GLWETrace::glwe_trace_tmp_bytes: {}",
            scratch.available(),
            module.glwe_trace_tmp_bytes_default(res, a, atk_layout)
        );

        let scratch_local = scratch.borrow();
        let (mut tmp, scratch_1) = scratch_local.take_glwe_scratch(&GLWELayout {
            n: res.n(),
            base2k: atk_layout.base2k(),
            k: a.k().max(res.k()),
            rank: res.rank(),
        });
        let mut scratch_1 = scratch_1;

        if a.base2k() == atk_layout.base2k() {
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

        if res.base2k() == atk_layout.base2k() {
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
        H: GLWEAutomorphismKeyMap<K, BE>,
    {
        trace_assign_internal::<M, K, H, _, BE>(module, res, skip, keys, scratch);
    }
}
