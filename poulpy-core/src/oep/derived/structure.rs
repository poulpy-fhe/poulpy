//! Derived trace and packing algorithms built from public core operations.
//!
//! Core calls go through the backend dispatch, so an override of an automorphism,
//! copy, normalization, rotation, or trace operation is reused by these defaults.

mod trace {
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
        GLWEAutomorphism, GLWECopy, GLWENormalize, GLWEShift, GLWETrace, ScratchArenaTakeCore,
        layouts::{GGLWEInfos, GLWEInfos, GLWELayout, GLWEToBackendMut, GLWEToBackendRef, GetAutomorphismKey, LWEInfos},
    };

    #[inline(always)]
    pub(crate) fn trace_galois_elements(log_n: usize, cyclotomic_order: i64) -> Vec<i64> {
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

    fn trace_assign_internal<M, H, R, BE: Backend>(
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
            + GLWETrace<BE>,
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
            scratch.available() >= glwe_trace_assign_tmp_bytes_derived::<BE, _, _, _>(module, res, &ksk_infos),
            "scratch.available(): {} < GLWETrace::glwe_trace_assign_tmp_bytes: {}",
            scratch.available(),
            glwe_trace_assign_tmp_bytes_derived::<BE, _, _, _>(module, res, &ksk_infos)
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
                module.glwe_trace_assign(&mut res_conv, skip, keys, scratch);
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

    pub(crate) fn glwe_trace_assign_tmp_bytes_derived<BE, M, A, K>(module: &M, a_infos: &A, key_infos: &K) -> usize
    where
        BE: Backend,
        M: GLWEBytesOf<BE>
            + GLWETrace<BE>
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
            let lvl_0: usize = BE::scratch_aligned(module.glwe_bytes_of_from_infos(&a_conv_infos));
            let lvl_1: usize = module
                .glwe_normalize_tmp_bytes()
                .max(module.glwe_trace_assign_tmp_bytes(&a_conv_infos, key_infos));
            return lvl_0 + lvl_1;
        }

        module
            .glwe_shift_tmp_bytes(a_infos.size())
            .max(module.glwe_automorphism_tmp_bytes(a_infos, a_infos, key_infos))
    }

    pub(crate) fn glwe_trace_galois_elements_derived<M>(module: &M) -> Vec<i64>
    where
        M: ModuleLogN + CyclotomicOrder,
    {
        trace_galois_elements(module.log_n(), module.cyclotomic_order())
    }

    pub(crate) fn glwe_trace_tmp_bytes_derived<BE, M, R, A, K>(module: &M, res_infos: &R, a_infos: &A, key_infos: &K) -> usize
    where
        BE: Backend,
        M: GLWEBytesOf<BE>
            + GLWETrace<BE>
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
        let lvl_0: usize = BE::scratch_aligned(module.glwe_bytes_of_from_infos(&tmp_infos));
        let lvl_1 = module.glwe_copy_tmp_bytes(&tmp_infos, a_infos);
        let lvl_2: usize = module.glwe_trace_assign_tmp_bytes(&tmp_infos, key_infos);
        let lvl_3 = module.glwe_copy_tmp_bytes(res_infos, &tmp_infos);

        // An empty trace copies directly without consulting a key; a backend's
        // direct-copy scratch requirement may differ from the two-stage path.
        (lvl_0 + lvl_1.max(lvl_2).max(lvl_3)).max(module.glwe_copy_tmp_bytes(res_infos, a_infos))
    }

    pub(crate) fn glwe_trace_derived<BE, M, R, A, H>(
        module: &M,
        res: &mut R,
        skip: usize,
        a: &A,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        BE: Backend,
        M: GLWEBytesOf<BE>
            + GLWETrace<BE>
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
            scratch.available() >= glwe_trace_tmp_bytes_derived::<BE, _, _, _, _>(module, res, a, &atk_layout),
            "scratch.available(): {} < GLWETrace::glwe_trace_tmp_bytes: {}",
            scratch.available(),
            glwe_trace_tmp_bytes_derived::<BE, _, _, _, _>(module, res, a, &atk_layout)
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
                module.glwe_trace_assign(&mut tmp, skip, keys, scratch);
            });
        }

        module.glwe_copy(res, &tmp, &mut scratch_1);
    }

    pub(crate) fn glwe_trace_assign_derived<BE, M, R, H>(
        module: &M,
        res: &mut R,
        skip: usize,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        BE: Backend,
        M: GLWEBytesOf<BE>
            + GLWETrace<BE>
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

pub(crate) use trace::{
    glwe_trace_assign_derived, glwe_trace_assign_tmp_bytes_derived, glwe_trace_derived, glwe_trace_galois_elements_derived,
    glwe_trace_tmp_bytes_derived, trace_galois_elements,
};

mod packing {
    //! GLWE coefficient packing through a binary merge tree and normalized trace.
    //!
    //! Input indices identify positions in the degree-`N` output ring. The map must
    //! be nonempty, `log_gap_out <= log2(N)`, and every index must be below `N` and
    //! divisible by `2^log_gap_out`. These conditions are checked before input or
    //! destination mutation. The input ciphertexts are consumed by valid calls.

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

    pub(crate) fn glwe_pack_galois_elements_derived<BE, M>(module: &M) -> Vec<i64>
    where
        BE: Backend,
        M: GLWETrace<BE>,
    {
        module.glwe_trace_galois_elements()
    }

    pub(crate) fn glwe_pack_tmp_bytes_derived<BE, M, R, A, K>(module: &M, res: &R, a: &A, key: &K) -> usize
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
        A: GLWEInfos,
        K: GGLWEInfos,
    {
        assert_eq!(module.n() as u32, res.n());
        assert_eq!(module.n() as u32, a.n());
        assert_eq!(module.n() as u32, key.n());

        // Every merge-tree operation runs on the inputs, not on the destination:
        // the accumulator stays at the input layout until the closing trace.
        let lvl_0: usize = BE::scratch_aligned(module.glwe_bytes_of_from_infos(a));
        let lvl_1: usize = module
            .glwe_rotate_tmp_bytes()
            .max(module.glwe_shift_tmp_bytes(a.size()))
            .max(module.glwe_normalize_tmp_bytes())
            .max(module.glwe_automorphism_tmp_bytes(a, a, key));

        (lvl_0 + lvl_1).max(module.glwe_trace_tmp_bytes(res, a, key))
    }

    pub(crate) fn glwe_pack_derived<BE, M, R, A, H>(
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
        assert!(log_gap_out <= module.log_n(), "packing log_gap_out exceeds log_n");
        let gap = 1usize << log_gap_out;
        assert!(!a.is_empty(), "packing requires at least one input");
        assert!(
            a.keys().all(|&index| index < module.n() && index % gap == 0),
            "packing indices must be below N and divisible by 2^log_gap_out"
        );
        // The merge tree and the closing trace both run at the input layout, which
        // every input shares. Keys may differ per rotation; the bound is read off
        // the first one, as on any path sized from a single key layout.
        let a_layout = a
            .values()
            .next()
            .map(|input| input.glwe_layout())
            .expect("packing requires at least one input");
        let key_infos = keys
            .get_automorphism_key(-1, a_layout.k().max(res.k()))
            .unwrap_or_else(|e| panic!("packing rotation -1: {e}"));
        assert!(
            scratch.available() >= glwe_pack_tmp_bytes_derived::<BE, _, _, _, _>(module, res, &a_layout, &key_infos),
            "scratch.available(): {} < GLWEPacking::glwe_pack_tmp_bytes: {}",
            scratch.available(),
            glwe_pack_tmp_bytes_derived::<BE, _, _, _, _>(module, res, &a_layout, &key_infos)
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

pub(crate) use packing::{glwe_pack_derived, glwe_pack_galois_elements_derived, glwe_pack_tmp_bytes_derived};
