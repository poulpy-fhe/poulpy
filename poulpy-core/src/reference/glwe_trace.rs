//! Normalized GLWE trace (projection by Galois averaging).
//!
//! Each visited level right-shifts the ciphertext by one bit, then adds its
//! Galois conjugate. Ignoring finite-precision rounding and key-switch noise,
//! the message action is the average of the selected conjugates, not their sum:
//! `NormalizedTrace(ct) = 2^(-(log_n - skip)) * sum_{i in S} phi_i(ct)`.
//!
//! `skip` omits the initial automorphism levels and must lie in `0..=log_n`.
//! With `skip == log_n`, the out-of-place operation copies the input at the
//! destination's layout and the assign operation leaves the input unchanged;
//! neither path needs an automorphism key. Each nonempty level rounds according
//! to the core right-shift contract before its automorphism and addition.
//!
//! Automorphism keys are indexed by the Galois elements returned from
//! [`GLWETrace::glwe_trace_galois_elements`](crate::api::GLWETrace::glwe_trace_galois_elements).

use crate::api::GLWEBytesOf;
use poulpy_hal::{
    api::ModuleLogN,
    layouts::{Backend, CyclotomicOrder, GaloisElement, ScratchArena, galois_element},
};

use crate::{
    GLWEAutomorphism, GLWECopy, GLWENormalize, GLWEShift, ScratchArenaTakeCore,
    layouts::{GGLWEInfos, GLWEInfos, GLWELayout, GLWEToBackendMut, GLWEToBackendRef, GetAutomorphismKey, LWEInfos},
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

fn trace_assign_internal<M, H, R, BE: Backend>(module: &M, res: &mut R, skip: usize, keys: &H, scratch: &mut ScratchArena<'_, BE>)
where
    M: GLWEBytesOf<BE>
        + ModuleLogN
        + GaloisElement
        + GLWEAutomorphism<BE>
        + GLWEShift<BE>
        + GLWECopy<BE>
        + CyclotomicOrder
        + GLWENormalize<BE>
        + GLWETraceReference<BE>
        + ?Sized,
    H: GetAutomorphismKey<BE>,
    R: GLWEToBackendMut<BE> + GLWEInfos,
{
    let log_n: usize = module.log_n();

    assert_eq!(res.n(), module.n() as u32);
    assert!(skip <= log_n);
    // Keys may differ per rotation, so the radix and the scratch bound are read
    // off the first one the source resolves, as on any path sized from a single
    // key layout.
    let Some(first) = trace_rotations(module, skip).next() else {
        return;
    };
    let ksk_infos = keys
        .get_automorphism_key(first, res.k())
        .unwrap_or_else(|e| panic!("trace rotation {first}: {e}"));
    assert_eq!(ksk_infos.n(), module.n() as u32);
    assert_eq!(ksk_infos.rank_in(), res.rank());
    assert_eq!(ksk_infos.rank_out(), res.rank());
    assert!(
        scratch.available() >= module.glwe_trace_assign_tmp_bytes_reference(res, &ksk_infos),
        "scratch.available(): {} < GLWETrace::glwe_trace_assign_tmp_bytes: {}",
        scratch.available(),
        module.glwe_trace_assign_tmp_bytes_reference(res, &ksk_infos)
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
            trace_assign_internal::<M, H, _, BE>(module, &mut res_conv, skip, keys, scratch);
        });

        scratch_1.apply_mut(|scratch| {
            module.glwe_normalize(res, &res_conv, scratch);
        });
        return;
    }

    for p in trace_rotations(module, skip) {
        module.glwe_rsh(1, res, scratch);
        let key = keys
            .get_automorphism_key(p, res.k())
            .unwrap_or_else(|e| panic!("trace rotation {p}: {e}"));
        module.glwe_automorphism_add_assign(res, &key, scratch);
    }
}

/// Rotations the trace loop visits, in order.
fn trace_rotations<M>(module: &M, skip: usize) -> impl Iterator<Item = i64> + use<'_, M>
where
    M: ModuleLogN + GaloisElement + ?Sized,
{
    (skip..module.log_n()).map(|i| if i == 0 { -1 } else { module.galois_element(1 << (i - 1)) })
}

/// Backend override contract; opt into its portable body with the matching forwarding macro.
pub trait GLWETraceReference<BE: Backend> {
    fn glwe_trace_assign_tmp_bytes_reference<A, K>(&self, a_infos: &A, key_infos: &K) -> usize
    where
        A: GLWEInfos,
        K: GGLWEInfos;

    fn glwe_trace_galois_elements_reference(&self) -> Vec<i64>;

    fn glwe_trace_tmp_bytes_reference<R, A, K>(&self, res_infos: &R, a_infos: &A, key_infos: &K) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        K: GGLWEInfos;

    fn glwe_trace_reference<R, A, H>(&self, res: &mut R, skip: usize, a: &A, keys: &H, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        A: GLWEToBackendRef<BE> + GLWEInfos,
        H: GetAutomorphismKey<BE>;

    fn glwe_trace_assign_reference<R, H>(&self, res: &mut R, skip: usize, keys: &H, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        H: GetAutomorphismKey<BE>;
}

/// Reference implementations of the [`GLWETraceReference`] methods.
pub mod glwe_trace_reference_impl {
    use super::*;

    pub fn glwe_trace_assign_tmp_bytes_reference<BE, M, A, K>(module: &M, a_infos: &A, key_infos: &K) -> usize
    where
        BE: Backend,
        M: GLWEBytesOf<BE>
            + GLWETraceReference<BE>
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
                .max(module.glwe_trace_assign_tmp_bytes_reference(&a_conv_infos, key_infos));
            return lvl_0 + lvl_1;
        }

        module
            .glwe_shift_tmp_bytes(a_infos.size())
            .max(module.glwe_automorphism_tmp_bytes(a_infos, a_infos, key_infos))
    }

    pub fn glwe_trace_galois_elements_reference<BE, M>(module: &M) -> Vec<i64>
    where
        BE: Backend,
        M: ModuleLogN + CyclotomicOrder,
    {
        trace_galois_elements(module.log_n(), module.cyclotomic_order())
    }

    pub fn glwe_trace_tmp_bytes_reference<BE, M, R, A, K>(module: &M, res_infos: &R, a_infos: &A, key_infos: &K) -> usize
    where
        BE: Backend,
        M: GLWEBytesOf<BE>
            + GLWETraceReference<BE>
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
        let lvl_1 = module.glwe_copy_tmp_bytes(&tmp_infos, a_infos);
        let lvl_2: usize = module.glwe_trace_assign_tmp_bytes_reference(&tmp_infos, key_infos);
        let lvl_3 = module.glwe_copy_tmp_bytes(res_infos, &tmp_infos);

        // An empty trace copies directly without consulting a key; a backend's
        // direct-copy scratch requirement may differ from the two-stage path.
        (lvl_0 + lvl_1.max(lvl_2).max(lvl_3)).max(module.glwe_copy_tmp_bytes(res_infos, a_infos))
    }

    pub fn glwe_trace_reference<BE, M, R, A, H>(
        module: &M,
        res: &mut R,
        skip: usize,
        a: &A,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        BE: Backend,
        M: GLWEBytesOf<BE>
            + GLWETraceReference<BE>
            + ModuleLogN
            + GaloisElement
            + GLWEAutomorphism<BE>
            + GLWEShift<BE>
            + GLWECopy<BE>
            + CyclotomicOrder
            + GLWENormalize<BE>,
        R: GLWEToBackendMut<BE> + GLWEInfos,
        A: GLWEToBackendRef<BE> + GLWEInfos,
        H: GetAutomorphismKey<BE>,
    {
        assert_eq!(res.n(), module.n() as u32);
        assert_eq!(a.n(), module.n() as u32);
        assert!(skip <= module.log_n(), "trace skip exceeds log_n");
        let Some(first) = trace_rotations(module, skip).next() else {
            module.glwe_copy(res, a, scratch);
            return;
        };
        let atk_layout = keys
            .get_automorphism_key(first, a.k().max(res.k()))
            .unwrap_or_else(|e| panic!("trace rotation {first}: {e}"));
        assert!(
            scratch.available() >= module.glwe_trace_tmp_bytes_reference(res, a, &atk_layout),
            "scratch.available(): {} < GLWETrace::glwe_trace_tmp_bytes: {}",
            scratch.available(),
            module.glwe_trace_tmp_bytes_reference(res, a, &atk_layout)
        );

        let scratch_local = scratch.borrow();
        let (mut tmp, scratch_1) = scratch_local.take_glwe_scratch(&GLWELayout {
            n: res.n(),
            base2k: atk_layout.base2k(),
            k: a.k().max(res.k()),
            rank: res.rank(),
        });
        let mut scratch_1 = scratch_1;

        module.glwe_copy(&mut tmp, a, &mut scratch_1);

        {
            scratch_1 = scratch_1.apply_mut(|scratch| {
                trace_assign_internal::<M, H, _, BE>(module, &mut tmp, skip, keys, scratch);
            });
        }

        module.glwe_copy(res, &tmp, &mut scratch_1);
    }

    pub fn glwe_trace_assign_reference<BE, M, R, H>(
        module: &M,
        res: &mut R,
        skip: usize,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        BE: Backend,
        M: GLWEBytesOf<BE>
            + GLWETraceReference<BE>
            + ModuleLogN
            + GaloisElement
            + GLWEAutomorphism<BE>
            + GLWEShift<BE>
            + GLWECopy<BE>
            + CyclotomicOrder
            + GLWENormalize<BE>,
        R: GLWEToBackendMut<BE> + GLWEInfos,
        H: GetAutomorphismKey<BE>,
    {
        trace_assign_internal::<M, H, _, BE>(module, res, skip, keys, scratch);
    }
}
