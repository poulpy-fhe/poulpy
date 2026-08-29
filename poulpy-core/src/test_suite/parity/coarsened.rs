//! Coarsened-key parity: an operation reading a stored key through a coarser
//! `dsize` agrees, byte for byte, with the same operation over a key natively
//! stored at that `dsize`.
//!
//! The two keys are never copied into one another. [`fill_by_digit`] draws each
//! digit from a source seeded by its digit index, so the rows the coarsening
//! reaches are identical by construction and every row it skips is poisoned:
//! reading one row too many, or the wrong one, moves the output.
//!
//! This is what the coarsening claims, stated as equality. The noise of the
//! key on the right is already pinned by the noise suite, so nothing here
//! needs a secret, an encryption or a bound.

use std::collections::HashMap;

use poulpy_hal::{
    api::{ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Backend, CyclotomicOrder, FillUniform, HostDataMut, HostDataRef, Module, ScratchOwned, ZnxView, ZnxViewMut},
    source::Source,
    test_suite::TestParams,
};

use crate::{
    GLWEAutomorphism, GLWETensoring, GLWETrace,
    layouts::{
        Base2K, Degree, Dnum, Dsize, GGLWEInfos, GGLWELayout, GLWE, GLWEAutomorphismKeyLayout,
        GLWEAutomorphismKeyPreparedFactory, GLWELayout, GLWETensorKeyLayout, GLWETensorKeyPreparedFactory, ModuleCoreAlloc, Rank,
        TorusPrecision,
        prepared::{GLWEAutomorphismKeyPrepared, GLWETensorKeyPrepared},
    },
    test_suite::keys::{AtDsize, fill_by_digit},
};

/// Backends this suite can run on: host-resident coefficients, so rows can be
/// filled and outputs compared without a transfer.
pub trait CoarsenBackend: Backend<ZnxWord = i64, OwnedBuf: HostDataMut> {}

impl<BE: Backend<ZnxWord = i64, OwnedBuf: HostDataMut>> CoarsenBackend for BE {}

/// The coarsened twin of `parent`, and the stride the parent must be filled at
/// to match it.
fn twin_layout(parent: &GGLWELayout, dsize: Dsize) -> (GGLWELayout, usize) {
    let twin: GGLWELayout = parent.gglwe_layout_at_dsize(dsize).unwrap();
    (twin, dsize.as_usize() / parent.dsize().as_usize())
}

/// Byte equality of two GLWE ciphertexts.
fn same<D: HostDataRef, E: HostDataRef>(have: &GLWE<D, i64>, want: &GLWE<E, i64>, what: &str) {
    assert_eq!(have.data.raw(), want.data.raw(), "{what}");
}

/// `glwe_automorphism` over a coarsened key equals it over the key it stands
/// in for, for the plain, `add_assign` and `assign` forms.
pub fn test_glwe_automorphism_coarsened<BE: CoarsenBackend>(params: &TestParams, module: &Module<BE>)
where
    Module<BE>: GLWEAutomorphism<BE> + GLWEAutomorphismKeyPreparedFactory<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let base2k: usize = params.base2k;
    let n: usize = module.n();
    let p: i64 = -5;
    let mut source: Source = Source::new([17u8; 32]);

    // (stored dsize, stored dnum, coarsening factor, rank).
    for (dsize, dnum, s, rank) in [(1usize, 8usize, 2usize, 1usize), (1, 12, 4, 2), (2, 8, 2, 1), (4, 8, 2, 1)] {
        let stored = GLWEAutomorphismKeyLayout {
            n: Degree(n as u32),
            base2k: Base2K(base2k as u32),
            dnum: Dnum(dnum as u32),
            dsize: Dsize(dsize as u32),
            k_aux: TorusPrecision((dsize * base2k + module.log_n()) as u32),
            rank: Rank(rank as u32),
        };
        let effective = Dsize((dsize * s) as u32);
        let (twin, stride) = twin_layout(&stored.gglwe_layout(), effective);
        let twin = GLWEAutomorphismKeyLayout {
            dnum: twin.dnum,
            dsize: twin.dsize,
            k_aux: twin.k_aux,
            ..stored
        };
        assert_eq!(stride, s);

        let mut key = module.glwe_automorphism_key_alloc_from_infos(&stored);
        let mut key_twin = module.glwe_automorphism_key_alloc_from_infos(&twin);
        let seed: [u8; 32] = [7u8; 32];
        fill_by_digit(&mut key.key, stride, &mut Source::new(seed));
        fill_by_digit(&mut key_twin.key, 1, &mut Source::new(seed));
        key.p = p;
        key_twin.p = p;

        let mut prep = ScratchOwned::<BE>::alloc(
            module
                .glwe_automorphism_key_prepare_tmp_bytes(&stored)
                .max(module.glwe_automorphism_key_prepare_tmp_bytes(&twin)),
        );
        let mut key_prep: GLWEAutomorphismKeyPrepared<BE::OwnedBuf, BE> =
            module.glwe_automorphism_key_prepared_alloc_from_infos(&stored);
        let mut twin_prep: GLWEAutomorphismKeyPrepared<BE::OwnedBuf, BE> =
            module.glwe_automorphism_key_prepared_alloc_from_infos(&twin);
        module.glwe_automorphism_key_prepare(&mut key_prep, &key, &mut prep.borrow());
        module.glwe_automorphism_key_prepare(&mut twin_prep, &key_twin, &mut prep.borrow());

        // `valid_dsizes` is exactly the set `with_dsize` accepts.
        for stride in 1..stored.dnum.as_usize() + 3 {
            let d = Dsize((stride * dsize) as u32);
            let listed = stored.valid_dsizes().into_iter().find(|(x, _)| *x == d);
            match (listed, key_prep.with_dsize(d)) {
                (Some((_, dnum)), Ok(view)) => assert_eq!(view.dnum(), dnum, "dnum disagrees at dsize={d}"),
                (None, Err(_)) => {}
                (l, v) => panic!("valid_dsizes says {l:?} at dsize={d}, with_dsize says {}", v.is_ok()),
            }
        }
        if dsize > 1 {
            assert!(
                key_prep.with_dsize(Dsize(dsize as u32 + 1)).is_err(),
                "accepted a non-multiple"
            );
        }

        let view = key_prep.with_dsize(effective).unwrap();
        assert_eq!(view.stride(), s);
        assert_eq!(view.gglwe_layout(), twin.gglwe_layout());

        let ct_infos = GLWELayout {
            n: Degree(n as u32),
            base2k: Base2K(base2k as u32),
            k: TorusPrecision((twin.dnum.as_usize() * twin.dsize.as_usize() * base2k) as u32),
            rank: Rank(rank as u32),
        };
        let mut ct_in = module.glwe_alloc_from_infos(&ct_infos);
        ct_in.fill_uniform(base2k, &mut source);

        let mut have = module.glwe_alloc_from_infos(&ct_infos);
        let mut want = module.glwe_alloc_from_infos(&ct_infos);
        let mut scratch = ScratchOwned::<BE>::alloc(
            module
                .glwe_automorphism_tmp_bytes(&ct_infos, &ct_infos, &view)
                .max(module.glwe_automorphism_tmp_bytes(&ct_infos, &ct_infos, &twin)),
        );

        module.glwe_automorphism(&mut have, &ct_in, p, &AtDsize(&key_prep, effective), &mut scratch.borrow());
        module.glwe_automorphism(&mut want, &ct_in, p, &twin_prep, &mut scratch.borrow());
        same(
            &have,
            &want,
            &format!("automorphism dsize={dsize} dnum={dnum} s={s} rank={rank}"),
        );

        have.data.raw_mut().copy_from_slice(ct_in.data.raw());
        want.data.raw_mut().copy_from_slice(ct_in.data.raw());
        module.glwe_automorphism_add_assign(&mut have, p, &AtDsize(&key_prep, effective), &mut scratch.borrow());
        module.glwe_automorphism_add_assign(&mut want, p, &twin_prep, &mut scratch.borrow());
        same(
            &have,
            &want,
            &format!("add_assign dsize={dsize} dnum={dnum} s={s} rank={rank}"),
        );

        have.data.raw_mut().copy_from_slice(ct_in.data.raw());
        want.data.raw_mut().copy_from_slice(ct_in.data.raw());
        module.glwe_automorphism_assign(&mut have, p, &AtDsize(&key_prep, effective), &mut scratch.borrow());
        module.glwe_automorphism_assign(&mut want, p, &twin_prep, &mut scratch.borrow());
        same(&have, &want, &format!("assign dsize={dsize} dnum={dnum} s={s} rank={rank}"));
    }
}

/// `glwe_trace_assign` over a key set read at one coarse `dsize` equals it over
/// the key set natively stored there.
///
/// The rotations carry different keys, and the stored shapes differ from one
/// another, so no single `GGLWELayout` describes the set: scratch is the
/// maximum over the keys the loop actually visits.
pub fn test_glwe_trace_coarsened<BE: CoarsenBackend>(params: &TestParams, module: &Module<BE>)
where
    Module<BE>: GLWETrace<BE> + GLWEAutomorphismKeyPreparedFactory<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let base2k: usize = params.base2k;
    let n: usize = module.n();
    let rank: usize = 1;
    let effective = Dsize(2);
    let mut source: Source = Source::new([23u8; 32]);

    // Deliberately unequal shapes: the set has no common layout.
    let shapes: [(usize, usize); 3] = [(1, 8), (1, 12), (2, 8)];
    let mut keys: HashMap<i64, GLWEAutomorphismKeyPrepared<BE::OwnedBuf, BE>> = HashMap::new();
    let mut twins: HashMap<i64, GLWEAutomorphismKeyPrepared<BE::OwnedBuf, BE>> = HashMap::new();
    let mut trace_bytes: usize = 0;

    let gal_els: Vec<i64> = crate::default::glwe_trace::trace_galois_elements(module.log_n(), module.cyclotomic_order());
    for (salt, gal_el) in gal_els.iter().enumerate() {
        let (dsize, dnum) = shapes[salt % shapes.len()];
        let stored = GLWEAutomorphismKeyLayout {
            n: Degree(n as u32),
            base2k: Base2K(base2k as u32),
            dnum: Dnum(dnum as u32),
            dsize: Dsize(dsize as u32),
            k_aux: TorusPrecision((dsize * base2k + module.log_n()) as u32),
            rank: Rank(rank as u32),
        };
        let (coarse, stride) = twin_layout(&stored.gglwe_layout(), effective);
        let twin = GLWEAutomorphismKeyLayout {
            dnum: coarse.dnum,
            dsize: coarse.dsize,
            k_aux: coarse.k_aux,
            ..stored
        };

        let mut key = module.glwe_automorphism_key_alloc_from_infos(&stored);
        let mut key_twin = module.glwe_automorphism_key_alloc_from_infos(&twin);
        // A distinct stream per rotation: the keys of a trace must differ.
        let mut seed: [u8; 32] = [11u8; 32];
        seed[0..8].copy_from_slice(&(salt as u64).to_le_bytes());
        fill_by_digit(&mut key.key, stride, &mut Source::new(seed));
        fill_by_digit(&mut key_twin.key, 1, &mut Source::new(seed));
        key.p = *gal_el;
        key_twin.p = *gal_el;

        let mut prep = ScratchOwned::<BE>::alloc(
            module
                .glwe_automorphism_key_prepare_tmp_bytes(&stored)
                .max(module.glwe_automorphism_key_prepare_tmp_bytes(&twin)),
        );
        let mut key_prep = module.glwe_automorphism_key_prepared_alloc_from_infos(&stored);
        let mut twin_prep = module.glwe_automorphism_key_prepared_alloc_from_infos(&twin);
        module.glwe_automorphism_key_prepare(&mut key_prep, &key, &mut prep.borrow());
        module.glwe_automorphism_key_prepare(&mut twin_prep, &key_twin, &mut prep.borrow());
        trace_bytes = trace_bytes.max(module.glwe_trace_tmp_bytes(&twin, &twin, &key_prep.with_dsize(effective).unwrap()));

        keys.insert(*gal_el, key_prep);
        twins.insert(*gal_el, twin_prep);
    }

    let ct_infos = GLWELayout {
        n: Degree(n as u32),
        base2k: Base2K(base2k as u32),
        k: TorusPrecision((4 * base2k) as u32),
        rank: Rank(rank as u32),
    };
    let mut have = module.glwe_alloc_from_infos(&ct_infos);
    have.fill_uniform(base2k, &mut source);
    let mut want = module.glwe_alloc_from_infos(&ct_infos);
    want.data.raw_mut().copy_from_slice(have.data.raw());

    let mut scratch = ScratchOwned::<BE>::alloc(trace_bytes);
    module.glwe_trace_assign(&mut have, 0, &AtDsize(&keys, effective), &mut scratch.borrow());
    module.glwe_trace_assign(&mut want, 0, &twins, &mut scratch.borrow());
    same(&have, &want, "trace");
}

/// `glwe_tensor_relinearize` over a coarsened tensor key equals it over the
/// tensor key natively stored at that `dsize`.
pub fn test_glwe_tensor_relinearize_coarsened<BE: CoarsenBackend>(params: &TestParams, module: &Module<BE>)
where
    Module<BE>: GLWETensoring<BE> + GLWETensorKeyPreparedFactory<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let base2k: usize = params.base2k;
    let n: usize = module.n();
    let mut source: Source = Source::new([29u8; 32]);

    for (dsize, dnum, s, rank) in [(1usize, 12usize, 2usize, 1usize), (1, 12, 3, 2), (2, 8, 2, 1)] {
        let stored = GLWETensorKeyLayout {
            n: Degree(n as u32),
            base2k: Base2K(base2k as u32),
            dnum: Dnum(dnum as u32),
            dsize: Dsize(dsize as u32),
            k_aux: TorusPrecision((dsize * base2k + module.log_n()) as u32),
            rank: Rank(rank as u32),
        };
        let effective = Dsize((dsize * s) as u32);
        let (coarse, stride) = twin_layout(&stored.gglwe_layout(), effective);
        let twin = GLWETensorKeyLayout {
            dnum: coarse.dnum,
            dsize: coarse.dsize,
            k_aux: coarse.k_aux,
            ..stored
        };

        let mut tsk = module.glwe_tensor_key_alloc_from_infos(&stored);
        let mut tsk_twin = module.glwe_tensor_key_alloc_from_infos(&twin);
        let seed: [u8; 32] = [13u8; 32];
        fill_by_digit(&mut tsk.0, stride, &mut Source::new(seed));
        fill_by_digit(&mut tsk_twin.0, 1, &mut Source::new(seed));

        let mut prep = ScratchOwned::<BE>::alloc(
            module
                .prepare_tensor_key_tmp_bytes(&stored)
                .max(module.prepare_tensor_key_tmp_bytes(&twin)),
        );
        let mut tsk_prep: GLWETensorKeyPrepared<BE::OwnedBuf, BE> = module.alloc_tensor_key_prepared_from_infos(&stored);
        let mut twin_prep: GLWETensorKeyPrepared<BE::OwnedBuf, BE> = module.alloc_tensor_key_prepared_from_infos(&twin);
        module.prepare_tensor_key(&mut tsk_prep, &tsk, &mut prep.borrow());
        module.prepare_tensor_key(&mut twin_prep, &tsk_twin, &mut prep.borrow());

        let ct_infos = GLWELayout {
            n: Degree(n as u32),
            base2k: Base2K(base2k as u32),
            k: TorusPrecision((twin.dnum.as_usize() * twin.dsize.as_usize() * base2k) as u32),
            rank: Rank(rank as u32),
        };
        let mut a = module.glwe_tensor_alloc_from_infos(&ct_infos);
        a.data.fill_uniform(base2k, &mut source);

        let mut have = module.glwe_alloc_from_infos(&ct_infos);
        let mut want = module.glwe_alloc_from_infos(&ct_infos);
        let view = tsk_prep.with_dsize(effective).unwrap();
        let mut scratch = ScratchOwned::<BE>::alloc(
            module
                .glwe_tensor_relinearize_tmp_bytes(&ct_infos, &a, &view)
                .max(module.glwe_tensor_relinearize_tmp_bytes(&ct_infos, &a, &twin)),
        );

        module.glwe_tensor_relinearize(&mut have, &a, &AtDsize(&tsk_prep, effective), &mut scratch.borrow());
        module.glwe_tensor_relinearize(&mut want, &a, &twin_prep, &mut scratch.borrow());
        same(
            &have,
            &want,
            &format!("relinearize dsize={dsize} dnum={dnum} s={s} rank={rank}"),
        );
    }
}
