//! Canonical cross-backend parity for every linear-transformation phase.

use std::collections::HashMap;

use poulpy_hal::{
    api::{CnvPVecAlloc, CnvPVecBytesOf, Convolution, ModuleN, ScratchOwnedBorrow},
    layouts::{Backend, Data, DataViewMut, GaloisElement, HostDataMut, Module, ScratchArena, VecZnxDftBackendMut, ZnxViewMut},
    source::Source,
    test_suite::TestParams,
};

use crate::{
    GLWELinearTransformations, GLWEMaskFill,
    api::TransferInto,
    layouts::{
        Base2K, Degree, Dnum, Dsize, GLWE, GLWEAutomorphismKeyLayout, GLWEBackendRef, GLWEInfos, GLWELayout, GLWEToBackendRef,
        IntPolyInfos, LWEInfos, LinearTransformation, LinearTransformationDiagonal, LinearTransformationGiantStep,
        LinearTransformationLayout, LinearTransformationStrategy, ModuleCoreAlloc, Rank, TorusPrecision,
        prepared::{GLWEAutomorphismKeyPreparedFactory, LinearTransformationBabySteps, PreparedDiagonal},
    },
    reference::linear_transformation::{DiagonalProd, glwe_accumulate_streamed_baby_steps_dft},
    test_suite::{
        keys::fill_by_digit,
        parity::{ParityBackend, ParityShapes, poisoned_scratch, ref_glwe},
    },
};

/// Scheme-independent test diagonal: its storage represents an integer polynomial.
struct StreamedDiagonal<D: Data>(GLWE<D, i64>);

impl<D: Data> LWEInfos for StreamedDiagonal<D> {
    fn n(&self) -> Degree {
        self.0.n()
    }
    fn base2k(&self) -> Base2K {
        self.0.base2k()
    }
    fn k(&self) -> TorusPrecision {
        self.0.k()
    }
    fn max_size(&self) -> usize {
        self.0.max_size()
    }
}
impl<D: Data> GLWEInfos for StreamedDiagonal<D> {
    fn rank(&self) -> Rank {
        self.0.rank()
    }
}
impl<D: Data> IntPolyInfos for StreamedDiagonal<D> {
    fn encoded_k(&self) -> TorusPrecision {
        self.max_k()
    }
}
impl<D: Data, BE: Backend> GLWEToBackendRef<BE> for StreamedDiagonal<D>
where
    GLWE<D, i64>: GLWEToBackendRef<BE>,
{
    fn to_backend_ref(&self) -> GLWEBackendRef<'_, BE> {
        self.0.to_backend_ref()
    }
}
impl<BE: Backend, D: Data> DiagonalProd<BE> for StreamedDiagonal<D>
where
    GLWE<D, i64>: GLWEToBackendRef<BE>,
{
    fn accumulate_giant_prod<M>(
        module: &M,
        cnv_offset_hi: usize,
        prod_dft: &mut VecZnxDftBackendMut<'_, BE>,
        lhs: &LinearTransformationBabySteps<BE>,
        gs: &LinearTransformationGiantStep<Self>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        M: CnvPVecBytesOf + Convolution<BE> + ModuleN,
    {
        glwe_accumulate_streamed_baby_steps_dft(module, cnv_offset_hi, prod_dft, lhs, gs, scratch);
    }
}

fn poison<B: Backend>(buffer: &mut B::OwnedBuf) {
    let bytes = B::len_bytes(buffer);
    B::copy_from_host(buffer, &vec![0xA5; bytes]);
}

/// Queries are exercised by their consumers using independently poisoned, exact budgets.
/// Prepared caches are compared only through canonical output and semantic metadata.
pub fn test_linear_transformation_parity<BR, BT>(params: &TestParams, shapes: &ParityShapes, r: &Module<BR>, t: &Module<BT>)
where
    BR: ParityBackend,
    BT: ParityBackend,
    BR::OwnedBuf: HostDataMut,
    Module<BR>: GLWELinearTransformations<BR> + GLWEAutomorphismKeyPreparedFactory<BR> + CnvPVecAlloc<BR> + GLWEMaskFill<BR>,
    Module<BT>: GLWELinearTransformations<BT> + GLWEAutomorphismKeyPreparedFactory<BT> + CnvPVecAlloc<BT>,
{
    assert_eq!(r.n(), t.n());
    let base = params.base2k.min(12);
    let k = 2 * base + 1;
    let mut source = Source::new([187; 32]);
    let schedule = LinearTransformationLayout {
        indexes: vec![0, 1, 4, 5],
        slots: r.n() / 2,
        strategy: LinearTransformationStrategy::Bsgs { giant_step: 4 },
    };
    let plan = schedule.index();
    // The cache may hold rotations unused by this matrix.
    let baby_steps = [0, 1, 3];
    for &rank in &shapes.ranks {
        for dsize in shapes.dsizes(k, base) {
            let ct = GLWELayout {
                n: r.n().into(),
                base2k: base.into(),
                k: k.into(),
                rank: rank.into(),
            };
            for pt_k in [base + 1, 2 * base + 1, 4 * base + 1] {
                let pt = GLWELayout {
                    rank: Rank(0),
                    k: pt_k.into(),
                    ..ct
                };
                let key = GLWEAutomorphismKeyLayout {
                    n: ct.n,
                    base2k: ct.base2k,
                    rank: ct.rank,
                    dnum: Dnum(k.div_ceil(base * dsize) as u32),
                    dsize: Dsize(dsize as u32),
                    k_aux: TorusPrecision((base * dsize + r.log_n()) as u32),
                };
                let input_r = ref_glwe(r, &ct, &mut source);
                let mut input_t = t.glwe_alloc_from_infos(&ct);
                input_r.transfer_into(&mut input_t);
                let mut keys_r = HashMap::new();
                let mut keys_t = HashMap::new();
                for rotation in [1, 3, 4] {
                    let p = r.galois_element(rotation);
                    let mut key_r = r.glwe_automorphism_key_alloc_from_infos(&key);
                    fill_by_digit(r, &mut key_r, 1, &mut source);
                    key_r.p = p;
                    let mut key_t = t.glwe_automorphism_key_alloc_from_infos(&key);
                    key_r.transfer_into(&mut key_t);
                    let mut prepared_r = r.glwe_automorphism_key_prepared_alloc_from_infos(&key);
                    let mut prepared_t = t.glwe_automorphism_key_prepared_alloc_from_infos(&key);
                    r.glwe_automorphism_key_prepare(
                        &mut prepared_r,
                        &key_r,
                        &mut poisoned_scratch::<BR>(r.glwe_automorphism_key_prepare_tmp_bytes(&key)).borrow(),
                    );
                    t.glwe_automorphism_key_prepare(
                        &mut prepared_t,
                        &key_t,
                        &mut poisoned_scratch::<BT>(t.glwe_automorphism_key_prepare_tmp_bytes(&key)).borrow(),
                    );
                    keys_r.insert(p, prepared_r);
                    keys_t.insert(p, prepared_t);
                }
                let mut lhs_r = LinearTransformationBabySteps::alloc(r, &baby_steps, &ct);
                let mut lhs_t = LinearTransformationBabySteps::alloc(t, &baby_steps, &ct);
                for value in lhs_r.values.values_mut() {
                    poison::<BR>(value.data_mut());
                }
                for value in lhs_t.values.values_mut() {
                    poison::<BT>(value.data_mut());
                }
                r.glwe_prepare_linear_transformation_baby_steps(
                    &mut lhs_r,
                    &input_r,
                    &keys_r,
                    &mut poisoned_scratch::<BR>(r.glwe_prepare_linear_transformation_baby_steps_tmp_bytes(&ct, &key)).borrow(),
                );
                t.glwe_prepare_linear_transformation_baby_steps(
                    &mut lhs_t,
                    &input_t,
                    &keys_t,
                    &mut poisoned_scratch::<BT>(t.glwe_prepare_linear_transformation_baby_steps_tmp_bytes(&ct, &key)).borrow(),
                );
                assert_eq!(lhs_r.baby_steps().collect::<Vec<_>>(), lhs_t.baby_steps().collect::<Vec<_>>());
                assert_eq!((lhs_r.size(), lhs_r.cols()), (lhs_t.size(), lhs_t.cols()));

                if dsize == 1 && pt_k == base + 1 {
                    // A single identity diagonal has an exact mathematical
                    // oracle; it must copy every input coefficient and require
                    // no giant rotation or key switching. The one-limb
                    // diagonal alignment is removed by a base-bit offset.
                    let one_layout = GLWELayout {
                        rank: Rank(0),
                        k: base.into(),
                        ..ct
                    };
                    let identity_schedule = LinearTransformationLayout {
                        indexes: vec![0],
                        slots: r.n() / 2,
                        strategy: LinearTransformationStrategy::Bsgs { giant_step: 4 },
                    };
                    let mut one_r = r.glwe_alloc_from_infos(&one_layout);
                    one_r.data.at_mut(0, 0)[0] = 1;
                    let mut one_t = t.glwe_alloc_from_infos(&one_layout);
                    one_r.transfer_into(&mut one_t);
                    let identity_r = LinearTransformation {
                        baby_steps: vec![0],
                        giant_steps: vec![LinearTransformationGiantStep {
                            rot: 0,
                            diagonals: vec![LinearTransformationDiagonal {
                                baby: 0,
                                plaintext: StreamedDiagonal(one_r),
                            }],
                        }],
                    };
                    let identity_t = LinearTransformation {
                        baby_steps: vec![0],
                        giant_steps: vec![LinearTransformationGiantStep {
                            rot: 0,
                            diagonals: vec![LinearTransformationDiagonal {
                                baby: 0,
                                plaintext: StreamedDiagonal(one_t),
                            }],
                        }],
                    };
                    let mut resident_r: LinearTransformation<PreparedDiagonal<BR::OwnedBuf, BR>> =
                        LinearTransformation::alloc_prepared(r, &identity_schedule, &one_layout);
                    let mut resident_t: LinearTransformation<PreparedDiagonal<BT::OwnedBuf, BT>> =
                        LinearTransformation::alloc_prepared(t, &identity_schedule, &one_layout);
                    r.glwe_prepare_linear_transformation_rhs(
                        &mut resident_r,
                        &identity_r,
                        &mut poisoned_scratch::<BR>(r.glwe_prepare_linear_transformation_rhs_tmp_bytes(&one_layout)).borrow(),
                    );
                    t.glwe_prepare_linear_transformation_rhs(
                        &mut resident_t,
                        &identity_t,
                        &mut poisoned_scratch::<BT>(t.glwe_prepare_linear_transformation_rhs_tmp_bytes(&one_layout)).borrow(),
                    );
                    let mut out_r = ref_glwe(r, &ct, &mut source);
                    let mut out_t = t.glwe_alloc_from_infos(&ct);
                    out_r.transfer_into(&mut out_t);
                    r.glwe_eval_linear_transformation_into(
                        base,
                        &mut out_r,
                        &lhs_r,
                        &resident_r,
                        &keys_r,
                        &mut poisoned_scratch::<BR>(r.glwe_eval_linear_transformation_tmp_bytes(&ct, &ct, &one_layout, &key))
                            .borrow(),
                    );
                    t.glwe_eval_linear_transformation_into(
                        base,
                        &mut out_t,
                        &lhs_t,
                        &resident_t,
                        &keys_t,
                        &mut poisoned_scratch::<BT>(t.glwe_eval_linear_transformation_tmp_bytes(&ct, &ct, &one_layout, &key))
                            .borrow(),
                    );
                    let mut have = r.glwe_alloc_from_infos(&ct);
                    out_t.transfer_into(&mut have);
                    assert_eq!(out_r, input_r, "reference identity linear transformation rank={rank}");
                    assert_eq!(have, input_r, "backend identity linear transformation rank={rank}");
                    r.glwe_eval_linear_transformation_into(
                        base,
                        &mut out_r,
                        &lhs_r,
                        &identity_r,
                        &keys_r,
                        &mut poisoned_scratch::<BR>(r.glwe_eval_linear_transformation_unprepared_rhs_tmp_bytes(
                            &ct,
                            &ct,
                            &one_layout,
                            &key,
                        ))
                        .borrow(),
                    );
                    t.glwe_eval_linear_transformation_into(
                        base,
                        &mut out_t,
                        &lhs_t,
                        &identity_t,
                        &keys_t,
                        &mut poisoned_scratch::<BT>(t.glwe_eval_linear_transformation_unprepared_rhs_tmp_bytes(
                            &ct,
                            &ct,
                            &one_layout,
                            &key,
                        ))
                        .borrow(),
                    );
                    out_t.transfer_into(&mut have);
                    assert_eq!(out_r, input_r, "reference streamed identity rank={rank}");
                    assert_eq!(have, input_r, "backend streamed identity rank={rank}");
                }

                let mut rhs_r = LinearTransformation {
                    baby_steps: plan.baby_steps.clone(),
                    giant_steps: Vec::new(),
                };
                let mut rhs_t = LinearTransformation {
                    baby_steps: plan.baby_steps.clone(),
                    giant_steps: Vec::new(),
                };
                for (&rot, babies) in plan.giant_steps.iter().zip(&plan.index) {
                    let mut diagonals_r = Vec::new();
                    let mut diagonals_t = Vec::new();
                    for &baby in babies {
                        let diagonal_r = ref_glwe(r, &pt, &mut source);
                        let mut diagonal_t = t.glwe_alloc_from_infos(&pt);
                        diagonal_r.transfer_into(&mut diagonal_t);
                        diagonals_r.push(LinearTransformationDiagonal {
                            baby,
                            plaintext: StreamedDiagonal(diagonal_r),
                        });
                        diagonals_t.push(LinearTransformationDiagonal {
                            baby,
                            plaintext: StreamedDiagonal(diagonal_t),
                        });
                    }
                    rhs_r.giant_steps.push(LinearTransformationGiantStep {
                        rot,
                        diagonals: diagonals_r,
                    });
                    rhs_t.giant_steps.push(LinearTransformationGiantStep {
                        rot,
                        diagonals: diagonals_t,
                    });
                }
                let mut resident_r: LinearTransformation<PreparedDiagonal<BR::OwnedBuf, BR>> =
                    LinearTransformation::alloc_prepared(r, &schedule, &pt);
                let mut resident_t: LinearTransformation<PreparedDiagonal<BT::OwnedBuf, BT>> =
                    LinearTransformation::alloc_prepared(t, &schedule, &pt);
                for giant in &mut resident_r.giant_steps {
                    for diagonal in &mut giant.diagonals {
                        poison::<BR>(diagonal.plaintext.cnv_mut().data_mut());
                    }
                }
                for giant in &mut resident_t.giant_steps {
                    for diagonal in &mut giant.diagonals {
                        poison::<BT>(diagonal.plaintext.cnv_mut().data_mut());
                    }
                }
                resident_r.set_log_scale(19);
                resident_t.set_log_scale(19);
                r.glwe_prepare_linear_transformation_rhs(
                    &mut resident_r,
                    &rhs_r,
                    &mut poisoned_scratch::<BR>(r.glwe_prepare_linear_transformation_rhs_tmp_bytes(&pt)).borrow(),
                );
                t.glwe_prepare_linear_transformation_rhs(
                    &mut resident_t,
                    &rhs_t,
                    &mut poisoned_scratch::<BT>(t.glwe_prepare_linear_transformation_rhs_tmp_bytes(&pt)).borrow(),
                );
                assert_eq!(resident_r.index(), resident_t.index());
                assert_eq!((resident_r.log_scale(), resident_t.log_scale()), (19, 19));
                for (gr, gt) in resident_r.giant_steps.iter().zip(&resident_t.giant_steps) {
                    for (dr, dt) in gr.diagonals.iter().zip(&gt.diagonals) {
                        assert_eq!(dr.plaintext.glwe_layout(), dt.plaintext.glwe_layout());
                    }
                }
                // Different destination radix forces the per-giant normalized
                // fallback instead of the lazy DFT accumulation path.
                for out_base in [base, base - 1] {
                    let out = GLWELayout {
                        base2k: out_base.into(),
                        ..ct
                    };
                    for offset in [0, base - 1, 2 * base, 2 * base + 1] {
                        let mut out_r = ref_glwe(r, &out, &mut source);
                        let mut out_t = t.glwe_alloc_from_infos(&out);
                        out_r.transfer_into(&mut out_t);
                        r.glwe_eval_linear_transformation_into(
                            offset,
                            &mut out_r,
                            &lhs_r,
                            &resident_r,
                            &keys_r,
                            &mut poisoned_scratch::<BR>(r.glwe_eval_linear_transformation_tmp_bytes(&out, &ct, &pt, &key))
                                .borrow(),
                        );
                        t.glwe_eval_linear_transformation_into(
                            offset,
                            &mut out_t,
                            &lhs_t,
                            &resident_t,
                            &keys_t,
                            &mut poisoned_scratch::<BT>(t.glwe_eval_linear_transformation_tmp_bytes(&out, &ct, &pt, &key))
                                .borrow(),
                        );
                        let mut have = r.glwe_alloc_from_infos(&out);
                        out_t.transfer_into(&mut have);
                        assert_eq!(
                            out_r, have,
                            "resident linear transform rank={rank} dsize={dsize} offset={offset} out_base={out_base}"
                        );
                        let mut streamed_r = ref_glwe(r, &out, &mut source);
                        let mut streamed_t = t.glwe_alloc_from_infos(&out);
                        streamed_r.transfer_into(&mut streamed_t);
                        r.glwe_eval_linear_transformation_into(
                            offset,
                            &mut streamed_r,
                            &lhs_r,
                            &rhs_r,
                            &keys_r,
                            &mut poisoned_scratch::<BR>(
                                r.glwe_eval_linear_transformation_unprepared_rhs_tmp_bytes(&out, &ct, &pt, &key),
                            )
                            .borrow(),
                        );
                        t.glwe_eval_linear_transformation_into(
                            offset,
                            &mut streamed_t,
                            &lhs_t,
                            &rhs_t,
                            &keys_t,
                            &mut poisoned_scratch::<BT>(
                                t.glwe_eval_linear_transformation_unprepared_rhs_tmp_bytes(&out, &ct, &pt, &key),
                            )
                            .borrow(),
                        );
                        streamed_t.transfer_into(&mut have);
                        assert_eq!(
                            streamed_r, have,
                            "streamed linear transform rank={rank} dsize={dsize} offset={offset} out_base={out_base}"
                        );
                        assert_eq!(
                            out_r, streamed_r,
                            "resident/streamed linear transform rank={rank} dsize={dsize} offset={offset} out_base={out_base}"
                        );
                    }
                }
            }
        }
    }
}
