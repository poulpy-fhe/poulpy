//! Derived trace and packing algorithms built from public core operations.
//!
//! Core calls go through the backend dispatch, so an override of an automorphism,
//! copy, normalization, rotation, or trace operation is reused by these defaults.

mod trace {
    //! GLWE trace over a window of levels, summed or normalized.
    //!
    //! Level `0` applies the conjugation, on the standard ring only, and level `i >= 1` the rotation by `2^(i-1)`
    //! slots, up to the level of the rotation by half the slots: a full trace ends at `log2(n)` on the standard ring and
    //! at `log2(n) + 1` on the conjugate-invariant ring. Each visited level adds its Galois conjugate to the ciphertext,
    //! `x <- x + phi(x)`, the automorphism reading `x` through a normalized copy and leaving the sum canonical.
    //! Normalized, `x` is first right-shifted by one bit with the core right-shift's rounding, so the message action is
    //! the average of the visited conjugates; summed, it is their sum. `reverse` visits the levels from the last down
    //! to the first, each by its inverse Galois element: the replicate.
    //!
    //! An empty window copies the input at the destination's layout out of place and leaves it unchanged in place;
    //! neither path needs an automorphism key. Automorphism keys are indexed by the Galois elements
    //! [`GLWETrace::glwe_trace_galois_elements`](crate::api::GLWETrace::glwe_trace_galois_elements) returns.
    use std::ops::Range;

    use crate::layouts::{operand_degree, operand_galois_element};

    use crate::api::GLWEBytesOf;
    use poulpy_hal::{
        api::ModuleLogN,
        layouts::{Backend, CyclotomicOrder, GaloisElement, Ring, ScratchArena, Standard},
    };

    use crate::{
        GLWEAutomorphism, GLWECopy, GLWENormalize, GLWEShift, GLWETrace, ScratchArenaTakeGLWE,
        layouts::{Degree, GGLWEInfos, GLWEInfos, GLWELayout, GLWEToBackendMut, GLWEToBackendRef, GetAutomorphismKey, LWEInfos},
    };

    /// The full trace of a degree-`n` operand: the conjugation first, on the standard ring only since the
    /// conjugate-invariant ring is fixed by it, then one level per doubling rotation up to half the slots.
    fn ring_levels<BE: Backend>(n: usize) -> Range<usize> {
        let first = usize::from(BE::Ring::CYCLOTOMIC_ORDER_FACTOR != Standard::CYCLOTOMIC_ORDER_FACTOR);
        first..(BE::Ring::CYCLOTOMIC_ORDER_FACTOR as usize * n).ilog2() as usize - 1
    }

    /// # Panics
    ///
    /// Before any write, unless `levels` lies within the trace of a degree-`n` operand.
    fn assert_levels<BE: Backend>(n: usize, levels: &Range<usize>) {
        let trace = ring_levels::<BE>(n);
        assert!(
            levels.start <= levels.end && levels.end <= trace.end,
            "trace: levels outside the ring's trace"
        );
        assert!(
            levels.is_empty() || levels.start >= trace.start,
            "trace: the conjugate-invariant ring has no conjugation level"
        );
    }

    /// The Galois elements a trace of a degree-`n` operand visits over `levels`, in order.
    fn trace_rotations<BE: Backend, M: GaloisElement + ?Sized>(
        module: &M,
        n: usize,
        levels: Range<usize>,
        reverse: bool,
    ) -> impl Iterator<Item = i64> + use<'_, BE, M> {
        let slots = BE::Ring::CYCLOTOMIC_ORDER_FACTOR * n as i64 / 4;
        (0..levels.len()).map(move |t| {
            let level = if reverse { levels.end - 1 - t } else { levels.start + t };
            if level == 0 {
                return -1;
            }
            let rotation = 1i64 << (level - 1);
            operand_galois_element(module, n, if reverse { slots - rotation } else { rotation })
        })
    }

    fn trace_assign_internal<M, H, R, BE: Backend>(
        module: &M,
        res: &mut R,
        levels: Range<usize>,
        reverse: bool,
        normalize: bool,
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
        operand_degree(module.n(), &[res.n()]);
        let n = res.n().as_usize();
        assert_levels::<BE>(n, &levels);
        // Keys may differ per rotation, so the radix and the scratch bound are read
        // off the first one the source resolves, as on any path sized from a single
        // key layout.
        let Some(first) = trace_rotations::<BE, _>(module, n, levels.clone(), reverse).next() else {
            return;
        };
        let ksk_infos = keys
            .get_automorphism_key(first, res.k())
            .unwrap_or_else(|e| panic!("trace rotation {first}: {e}"));
        operand_degree(module.n(), &[ksk_infos.n()]);
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
                n: res.n(),
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
                module.glwe_trace_assign(&mut res_conv, levels, reverse, normalize, keys, scratch);
            });

            scratch_1.apply_mut(|scratch| {
                module.glwe_normalize(res, &res_conv, scratch);
            });
            return;
        }

        for p in trace_rotations::<BE, _>(module, n, levels, reverse) {
            if normalize {
                module.glwe_rsh_assign(1, res, scratch);
            }
            let key = keys
                .get_automorphism_key(p, res.k())
                .unwrap_or_else(|e| panic!("trace rotation {p}: {e}"));
            module.glwe_automorphism_add_assign(res, &key, scratch);
        }
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
        operand_degree(module.n(), &[a_infos.n(), key_infos.n()]);

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

    pub(crate) fn glwe_trace_galois_elements_derived<BE, M>(
        module: &M,
        n: Degree,
        levels: Range<usize>,
        reverse: bool,
    ) -> Vec<i64>
    where
        BE: Backend,
        M: GaloisElement,
    {
        let n = operand_degree(module.n(), &[n]);
        assert_levels::<BE>(n, &levels);
        trace_rotations::<BE, _>(module, n, levels, reverse).collect()
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
        operand_degree(module.n(), &[res_infos.n(), a_infos.n(), key_infos.n()]);

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

    #[expect(
        clippy::too_many_arguments,
        reason = "the trace's operands, levels, direction and normalization, keys and scratch"
    )]
    pub(crate) fn glwe_trace_derived<BE, M, R, A, H>(
        module: &M,
        res: &mut R,
        levels: Range<usize>,
        reverse: bool,
        normalize: bool,
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
        operand_degree(module.n(), &[res.n(), a.n()]);
        let n = res.n().as_usize();
        assert_levels::<BE>(n, &levels);
        let Some(first) = trace_rotations::<BE, _>(module, n, levels.clone(), reverse).next() else {
            module.glwe_copy(res, a, scratch);
            res.set_noise(None);
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
                module.glwe_trace_assign(&mut tmp, levels, reverse, normalize, keys, scratch);
            });
        }

        module.glwe_copy(res, &tmp, &mut scratch_1);
    }

    pub(crate) fn glwe_trace_assign_derived<BE, M, R, H>(
        module: &M,
        res: &mut R,
        levels: Range<usize>,
        reverse: bool,
        normalize: bool,
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
        trace_assign_internal::<M, H, _, BE>(module, res, levels, reverse, normalize, keys, scratch);
        res.set_noise(None);
    }
}

pub(crate) use trace::{
    glwe_trace_assign_derived, glwe_trace_assign_tmp_bytes_derived, glwe_trace_derived, glwe_trace_galois_elements_derived,
    glwe_trace_tmp_bytes_derived,
};

mod packing {
    //! GLWE coefficient packing through a binary merge tree and normalized trace.
    //!
    //! Input indices identify positions in the degree-`N` output ring. The map must
    //! be nonempty, `log_gap_out <= log2(N)`, and every index must be below `N` and
    //! divisible by `2^log_gap_out`. All inputs must share the same degree, radix,
    //! precision and rank. These conditions are checked before input or destination
    //! mutation. The input ciphertexts are consumed by valid calls.
    use crate::layouts::{operand_degree, operand_galois_element};

    use crate::api::GLWEBytesOf;
    use std::collections::HashMap;

    use poulpy_hal::{
        api::ModuleLogN,
        layouts::{Backend, GaloisElement, ScratchArena},
    };

    use crate::{
        GLWEAdd, GLWEAutomorphism, GLWENormalize, GLWERotate, GLWEShift, GLWESub, GLWETrace, ScratchArenaTakeGLWE,
        layouts::{GGLWEInfos, GLWEInfos, GLWEToBackendMut, GetAutomorphismKey, LWEInfos},
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
                let (mut tmp_b, mut scratch_1) = scratch.borrow().take_glwe_scratch(&a_layout);
                module.glwe_rotate_assign(-t, a, &mut scratch_1);
                module.glwe_sub(&mut tmp_b, a, b);
                module.glwe_rsh_assign(1, &mut tmp_b, &mut scratch_1);
                module.glwe_add_assign(a, b);
                module.glwe_rsh_assign(1, a, &mut scratch_1);
                module.glwe_normalize_assign(&mut tmp_b, &mut scratch_1);
                let key = keys
                    .get_automorphism_key(p, tmp_b.k())
                    .unwrap_or_else(|e| panic!("pack rotation {p}: {e}"));
                module.glwe_automorphism_assign(&mut tmp_b, &key, &mut scratch_1);
                module.glwe_sub_assign(a, &tmp_b);
                module.glwe_normalize_assign(a, &mut scratch_1);
                module.glwe_rotate_assign(t, a, &mut scratch_1);
            } else {
                module.glwe_rsh_assign(1, a, scratch);
                let key = keys
                    .get_automorphism_key(p, a.k())
                    .unwrap_or_else(|e| panic!("pack rotation {p}: {e}"));
                module.glwe_automorphism_add_assign(a, &key, scratch)
            }
        } else if let Some(b) = b.as_deref_mut() {
            let t: i64 = 1 << (b.n().log2() - i - 1);

            let b_layout = b.glwe_layout();
            let (mut tmp_b, mut scratch_1) = scratch.borrow().take_glwe_scratch(&b_layout);
            module.glwe_rotate(t, &mut tmp_b, b);
            module.glwe_rsh_assign(1, &mut tmp_b, &mut scratch_1);
            let key = keys
                .get_automorphism_key(p, tmp_b.k())
                .unwrap_or_else(|e| panic!("pack rotation {p}: {e}"));
            module.glwe_automorphism_sub_negate(b, &tmp_b, &key, &mut scratch_1)
        }
    }

    pub(crate) fn glwe_pack_galois_elements_derived<BE, M>(module: &M) -> Vec<i64>
    where
        BE: Backend,
        M: GLWETrace<BE> + ModuleLogN,
    {
        module.glwe_trace_galois_elements(module.n().into(), 0..module.log_n(), false)
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
        operand_degree(module.n(), &[res.n(), a.n(), key.n()]);

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
            + GLWERotate<BE>
            + GLWESub<BE>
            + GLWEShift<BE>
            + GLWEAdd<BE>
            + GLWENormalize<BE>
            + GLWETrace<BE>,
        R: GLWEToBackendMut<BE> + GLWEInfos,
        A: GLWEToBackendMut<BE> + GLWEInfos,
        H: GetAutomorphismKey<BE>,
    {
        assert!(log_gap_out <= res.log_n(), "packing log_gap_out exceeds log_n");
        let gap = 1usize << log_gap_out;
        assert!(!a.is_empty(), "packing requires at least one input");
        assert!(
            a.keys().all(|&index| index < res.n().as_usize() && index % gap == 0),
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
        for (&index, input) in &a {
            assert_eq!(input.glwe_layout(), a_layout, "packing input {index} has a different layout");
        }
        let key_infos = keys
            .get_automorphism_key(-1, a_layout.k().max(res.k()))
            .unwrap_or_else(|e| panic!("packing rotation -1: {e}"));
        assert!(
            scratch.available() >= glwe_pack_tmp_bytes_derived::<BE, _, _, _, _>(module, res, &a_layout, &key_infos),
            "scratch.available(): {} < GLWEPack::glwe_pack_tmp_bytes: {}",
            scratch.available(),
            glwe_pack_tmp_bytes_derived::<BE, _, _, _, _>(module, res, &a_layout, &key_infos)
        );

        let mut scratch_local = scratch.borrow();
        let log_n: usize = res.log_n();
        for i in 0..(log_n - log_gap_out) {
            let t: usize = (1 << log_n).min(1 << (log_n - 1 - i));

            let p: i64 = if i == 0 {
                -1
            } else {
                operand_galois_element(module, 1 << log_n, 1 << (i - 1))
            };

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
            module.glwe_trace(
                res,
                log_n - log_gap_out..log_n,
                false,
                true,
                *a.get_mut(&0).unwrap(),
                keys,
                scratch,
            );
        });
    }
}

pub(crate) use packing::{glwe_pack_derived, glwe_pack_galois_elements_derived, glwe_pack_tmp_bytes_derived};
