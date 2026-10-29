//! Canonical cross-backend parity for every linear-transformation phase.

use std::collections::HashMap;

use poulpy_hal::{
    api::{CnvPVecAlloc, CnvPVecBytesOf, Convolution, ModuleN},
    layouts::{Backend, Data, DataViewMut, GaloisElement, HostDataMut, Module, ScratchArena, VecZnxDftBackendMut, ZnxViewMut},
    source::Source,
    test_suite::TestParams,
};

use crate::{
    DiagonalProd, GLWELinearTransformations,
    api::{GLWEAutomorphismKeyPreparedFactory, GLWETensorKeyPreparedFactory, TransferInto},
    layouts::{
        Base2K, Degree, Dnum, GGLWEInfos, GLWE, GLWEAutomorphismKeyLayout, GLWEBackendRef, GLWEInfos, GLWELayout,
        GLWETensorKeyLayout, GLWEToBackendRef, IntPolyInfos, LWEInfos, LinearTransformation, LinearTransformationDiagonal,
        LinearTransformationGiantStep, LinearTransformationLayout, LinearTransformationStrategy, ModuleCoreAlloc, Rank,
        TorusPrecision, TransformKeys,
        prepared::{LinearTransformationBabySteps, PreparedDiagonal},
    },
    reference::linear_transformation::glwe_accumulate_streamed_baby_steps_dft,
    test_suite::{
        keys::fill_by_digit,
        noise::assert_refuses_uncovered_input,
        parity::{ParityBackend, ParityShapes, poisoned_scratch, ref_glwe, unnormalized_twin},
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
impl<BE: Backend<ZnxWord = i64>> DiagonalProd<BE> for StreamedDiagonal<BE::OwnedBuf>
where
    GLWE<BE::OwnedBuf, i64>: GLWEToBackendRef<BE>,
{
    fn mul_right_operand(&self) -> Option<crate::layouts::prepared::CiphertextDiagonal<'_, BE>> {
        Some(crate::layouts::MulRightOperand::Ciphertext(&self.0))
    }

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
/// Then diagonals of the source's rank, multiplied as ciphertexts, on the per-rotation schedule.
pub fn test_linear_transformation_parity<BR, BT>(params: &TestParams, shapes: &ParityShapes, r: &Module<BR>, t: &Module<BT>)
where
    BR: ParityBackend,
    BT: ParityBackend,
    BR::OwnedBuf: HostDataMut,
    Module<BR>: GLWELinearTransformations<BR>
        + GLWEAutomorphismKeyPreparedFactory<BR>
        + GLWETensorKeyPreparedFactory<BR>
        + CnvPVecAlloc<BR>,
    Module<BT>: GLWELinearTransformations<BT>
        + GLWEAutomorphismKeyPreparedFactory<BT>
        + GLWETensorKeyPreparedFactory<BT>
        + CnvPVecAlloc<BT>,
{
    assert!(params.n <= r.n() && params.n <= t.n());
    let base = params.base2k.min(12);
    let k = 2 * base + 1;
    let mut source = Source::new([187; 32]);
    let schedule = LinearTransformationLayout {
        indexes: vec![0, 1, 4, 5],
        slots: params.n / 2,
        strategy: LinearTransformationStrategy::Bsgs { giant_step: 4 },
    };
    let plan = schedule.index();
    // The cache may hold rotations unused by this matrix.
    let baby_steps = [0, 1, 3];
    for &rank in &shapes.ranks {
        for k_digit in shapes.k_digits(k, base, rank) {
            let ct = GLWELayout {
                n: params.n.into(),
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
                    dnum: Dnum(k.div_ceil(k_digit) as u32),
                    k_digit: TorusPrecision(k_digit as u32),
                    k_aux: TorusPrecision((k_digit + (params.n.ilog2() as usize)) as u32),
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
                    &mut poisoned_scratch::<BR>(r.glwe_prepare_linear_transformation_baby_steps_tmp_bytes(&ct, &plan, &key))
                        .borrow(),
                );
                t.glwe_prepare_linear_transformation_baby_steps(
                    &mut lhs_t,
                    &input_t,
                    &keys_t,
                    &mut poisoned_scratch::<BT>(t.glwe_prepare_linear_transformation_baby_steps_tmp_bytes(&ct, &plan, &key))
                        .borrow(),
                );
                assert_eq!(lhs_r.baby_steps().collect::<Vec<_>>(), lhs_t.baby_steps().collect::<Vec<_>>());
                assert_eq!((lhs_r.size(), lhs_r.cols()), (lhs_t.size(), lhs_t.cols()));

                let mut twin_t = t.glwe_alloc_from_infos(&ct);
                unnormalized_twin::<BR, BT>(&input_r, &mut twin_t);
                let mut lhs_twin = LinearTransformationBabySteps::alloc(t, &baby_steps, &ct);
                t.glwe_prepare_linear_transformation_baby_steps(
                    &mut lhs_twin,
                    &twin_t,
                    &keys_t,
                    &mut poisoned_scratch::<BT>(t.glwe_prepare_linear_transformation_baby_steps_tmp_bytes(&ct, &plan, &key))
                        .borrow(),
                );
                for (have, want) in lhs_twin.values.values().zip(lhs_t.values.values()) {
                    assert!(
                        BT::to_host_bytes(poulpy_hal::layouts::DataView::data(have))
                            == BT::to_host_bytes(poulpy_hal::layouts::DataView::data(want)),
                        "baby steps, unnormalized input rank={rank}"
                    );
                }

                // A second source for the summed evaluations below.
                let input2_r = ref_glwe(r, &ct, &mut source);
                let mut input2_t = t.glwe_alloc_from_infos(&ct);
                input2_r.transfer_into(&mut input2_t);
                let mut lhs2_r = LinearTransformationBabySteps::alloc(r, &baby_steps, &ct);
                let mut lhs2_t = LinearTransformationBabySteps::alloc(t, &baby_steps, &ct);
                r.glwe_prepare_linear_transformation_baby_steps(
                    &mut lhs2_r,
                    &input2_r,
                    &keys_r,
                    &mut poisoned_scratch::<BR>(r.glwe_prepare_linear_transformation_baby_steps_tmp_bytes(&ct, &plan, &key))
                        .borrow(),
                );
                t.glwe_prepare_linear_transformation_baby_steps(
                    &mut lhs2_t,
                    &input2_t,
                    &keys_t,
                    &mut poisoned_scratch::<BT>(t.glwe_prepare_linear_transformation_baby_steps_tmp_bytes(&ct, &plan, &key))
                        .borrow(),
                );

                if k_digit == base && pt_k == base + 1 {
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
                        slots: params.n / 2,
                        strategy: LinearTransformationStrategy::Bsgs { giant_step: 4 },
                    };
                    let mut one_r = r.glwe_alloc_from_infos(&one_layout);
                    one_r.data.at_mut(0, 0)[0] = 1;
                    let mut one_t = t.glwe_alloc_from_infos(&one_layout);
                    one_r.transfer_into(&mut one_t);
                    let identity_r = LinearTransformation {
                        baby_steps: vec![0],
                        rotations: Default::default(),
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
                        rotations: Default::default(),
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
                        &mut poisoned_scratch::<BR>(r.glwe_eval_linear_transformation_tmp_bytes(
                            &ct,
                            &ct,
                            &one_layout,
                            &identity_schedule.index(),
                            &key,
                            &key,
                        ))
                        .borrow(),
                    );
                    t.glwe_eval_linear_transformation_into(
                        base,
                        &mut out_t,
                        &lhs_t,
                        &resident_t,
                        &keys_t,
                        &mut poisoned_scratch::<BT>(t.glwe_eval_linear_transformation_tmp_bytes(
                            &ct,
                            &ct,
                            &one_layout,
                            &identity_schedule.index(),
                            &key,
                            &key,
                        ))
                        .borrow(),
                    );
                    let mut have = r.glwe_alloc_from_infos(&ct);
                    out_t.transfer_into(&mut have);
                    // The input's coefficients without its noise estimate: an evaluation is not a copy.
                    let mut want = r.glwe_alloc_from_infos(&ct);
                    input_r.transfer_into(&mut want);
                    want.noise = None;
                    assert_eq!(out_r, want, "reference identity linear transformation rank={rank}");
                    assert_eq!(have, want, "backend identity linear transformation rank={rank}");
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
                            &identity_schedule.index(),
                            &key,
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
                            &identity_schedule.index(),
                            &key,
                            &key,
                        ))
                        .borrow(),
                    );
                    out_t.transfer_into(&mut have);
                    assert_eq!(out_r, want, "reference streamed identity rank={rank}");
                    assert_eq!(have, want, "backend streamed identity rank={rank}");
                }

                // Random diagonals on `schedule`, the same on both backends, and
                // their resident preparations.
                let transform = |source: &mut Source| {
                    let mut rhs_r = LinearTransformation {
                        baby_steps: plan.baby_steps.clone(),
                        rotations: plan.rotations,
                        giant_steps: Vec::new(),
                    };
                    let mut rhs_t = LinearTransformation {
                        baby_steps: plan.baby_steps.clone(),
                        rotations: plan.rotations,
                        giant_steps: Vec::new(),
                    };
                    for (&rot, babies) in plan.giant_steps.iter().zip(&plan.index) {
                        let mut diagonals_r = Vec::new();
                        let mut diagonals_t = Vec::new();
                        for &baby in babies {
                            let diagonal_r = ref_glwe(r, &pt, source);
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
                    (rhs_r, rhs_t, resident_r, resident_t)
                };
                let (rhs_r, rhs_t, resident_r, resident_t) = transform(&mut source);
                // The summed evaluations' second transform: with one transform
                // for both terms, a sum that applied the first term's transform
                // to every term would give the same bits.
                let (_, _, resident2_r, resident2_t) = transform(&mut source);
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
                            &mut poisoned_scratch::<BR>(
                                r.glwe_eval_linear_transformation_tmp_bytes(&out, &ct, &pt, &plan, &key, &key),
                            )
                            .borrow(),
                        );
                        t.glwe_eval_linear_transformation_into(
                            offset,
                            &mut out_t,
                            &lhs_t,
                            &resident_t,
                            &keys_t,
                            &mut poisoned_scratch::<BT>(
                                t.glwe_eval_linear_transformation_tmp_bytes(&out, &ct, &pt, &plan, &key, &key),
                            )
                            .borrow(),
                        );
                        let mut have = r.glwe_alloc_from_infos(&out);
                        out_t.transfer_into(&mut have);
                        assert_glwe_eq!(
                            out_r,
                            have,
                            "resident linear transform rank={rank} k_digit={k_digit} offset={offset} out_base={out_base}"
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
                                r.glwe_eval_linear_transformation_unprepared_rhs_tmp_bytes(&out, &ct, &pt, &plan, &key, &key),
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
                                t.glwe_eval_linear_transformation_unprepared_rhs_tmp_bytes(&out, &ct, &pt, &plan, &key, &key),
                            )
                            .borrow(),
                        );
                        streamed_t.transfer_into(&mut have);
                        assert_glwe_eq!(
                            streamed_r,
                            have,
                            "streamed linear transform rank={rank} k_digit={k_digit} offset={offset} out_base={out_base}"
                        );
                        assert_glwe_eq!(
                            out_r,
                            streamed_r,
                            "resident/streamed linear transform rank={rank} k_digit={k_digit} offset={offset} out_base={out_base}"
                        );
                        // The second term goes through the sum's scratch bucket.
                        let mut sum_r = ref_glwe(r, &out, &mut source);
                        let mut sum_t = t.glwe_alloc_from_infos(&out);
                        sum_r.transfer_into(&mut sum_t);
                        r.glwe_eval_linear_transformation_sum_into(
                            offset,
                            &mut sum_r,
                            &[&lhs_r, &lhs2_r],
                            &[&resident_r, &resident2_r],
                            &keys_r,
                            &mut poisoned_scratch::<BR>(
                                r.glwe_eval_linear_transformation_sum_tmp_bytes(&out, &ct, &pt, &plan, &key, &key),
                            )
                            .borrow(),
                        );
                        t.glwe_eval_linear_transformation_sum_into(
                            offset,
                            &mut sum_t,
                            &[&lhs_t, &lhs2_t],
                            &[&resident_t, &resident2_t],
                            &keys_t,
                            &mut poisoned_scratch::<BT>(
                                t.glwe_eval_linear_transformation_sum_tmp_bytes(&out, &ct, &pt, &plan, &key, &key),
                            )
                            .borrow(),
                        );
                        sum_t.transfer_into(&mut have);
                        assert_glwe_eq!(
                            sum_r,
                            have,
                            "summed linear transform rank={rank} k_digit={k_digit} offset={offset} out_base={out_base}"
                        );
                        // One term is the single transform, bit for bit, on the tested backend too.
                        let mut one_t = t.glwe_alloc_from_infos(&out);
                        t.glwe_eval_linear_transformation_sum_into(
                            offset,
                            &mut one_t,
                            &[&lhs_t],
                            &[&resident_t],
                            &keys_t,
                            &mut poisoned_scratch::<BT>(
                                t.glwe_eval_linear_transformation_sum_tmp_bytes(&out, &ct, &pt, &plan, &key, &key),
                            )
                            .borrow(),
                        );
                        one_t.transfer_into(&mut have);
                        assert_glwe_eq!(
                            out_r,
                            have,
                            "one-term linear transform sum rank={rank} k_digit={k_digit} offset={offset} out_base={out_base}"
                        );
                    }
                }
                let wide = GLWELayout {
                    k: TorusPrecision(key.gglwe_layout().max_input_k().0 + 1),
                    ..ct
                };
                let mut wide_t = t.glwe_alloc_from_infos(&wide);
                assert_refuses_uncovered_input("giant steps took an output one bit past their keys", || {
                    t.glwe_eval_linear_transformation_into(
                        0,
                        &mut wide_t,
                        &lhs_t,
                        &resident_t,
                        &keys_t,
                        &mut poisoned_scratch::<BT>(
                            t.glwe_eval_linear_transformation_tmp_bytes(&wide, &ct, &pt, &plan, &key, &key),
                        )
                        .borrow(),
                    )
                });
            }
        }
    }
    ciphertext_diagonals_parity(
        params,
        shapes,
        r,
        t,
        LinearTransformationStrategy::Bsgs { giant_step: 4 },
        &mut source,
    );
}

/// The transformation of diagonals `-16..16` with the keys of the power-of-two rotations, the
/// baby steps by doubling, whose second pass switches two ciphertexts with one key, and the giant
/// steps as a tree whose first level switches four: the same bits on both backends, on the lazy route and on the
/// normalized one of a destination of another radix. Then the diagonals `{0, 1, 60, 61}`, whose giant digits are
/// `{0, 15}` with 128 slots or more: their tree takes a buffer per switch, more than two per giant step. Then
/// diagonals of the source's rank, multiplied as ciphertexts, on a power-of-two schedule.
/// Every call runs on the scratch its query returns.
pub fn test_linear_transformation_powers_of_two_parity<BR, BT>(
    params: &TestParams,
    shapes: &ParityShapes,
    r: &Module<BR>,
    t: &Module<BT>,
) where
    BR: ParityBackend,
    BT: ParityBackend,
    BR::OwnedBuf: HostDataMut,
    Module<BR>: GLWELinearTransformations<BR>
        + GLWEAutomorphismKeyPreparedFactory<BR>
        + GLWETensorKeyPreparedFactory<BR>
        + CnvPVecAlloc<BR>,
    Module<BT>: GLWELinearTransformations<BT>
        + GLWEAutomorphismKeyPreparedFactory<BT>
        + GLWETensorKeyPreparedFactory<BT>
        + CnvPVecAlloc<BT>,
{
    assert!(params.n <= r.n() && params.n <= t.n());
    let base = params.base2k.min(12);
    let k = 2 * base + 1;
    let mut source = Source::new([188; 32]);
    let slots = params.n / 2;
    for (indexes, pass_sizes) in [((-16..16).collect(), vec![1, 2]), (vec![0, 1, 60, 61], vec![1])] {
        let schedule = LinearTransformationLayout {
            indexes,
            slots,
            strategy: LinearTransformationStrategy::BsgsPowersOfTwo { giant_step: 4 },
        };
        let plan = schedule.index();
        // Every rank and digit width of the shapes with keys at the input's radix, whose passes are
        // hoisted, then keys one bit narrower, which every pass switches rotation by rotation, on
        // diagonals of two limbs, and diagonals of one limb.
        let mut cases: Vec<(usize, usize, usize, usize)> = Vec::new();
        for &rank in &shapes.ranks {
            cases.extend(
                shapes
                    .k_digits(k, base, rank)
                    .into_iter()
                    .map(|k_digit| (rank, base, k_digit, base + 1)),
            );
        }
        cases.push((1, base - 1, base - 1, base + 1));
        cases.push((1, base, base, base));
        for (rank, key_base, k_digit, pt_k) in cases {
            let ct = GLWELayout {
                n: params.n.into(),
                base2k: base.into(),
                k: k.into(),
                rank: rank.into(),
            };
            let pt = GLWELayout {
                rank: Rank(0),
                k: pt_k.into(),
                ..ct
            };
            let key = GLWEAutomorphismKeyLayout {
                n: ct.n,
                base2k: key_base.into(),
                rank: ct.rank,
                dnum: Dnum(k.div_ceil(k_digit) as u32),
                k_digit: TorusPrecision(k_digit as u32),
                k_aux: TorusPrecision((k_digit + (params.n.ilog2() as usize)) as u32),
            };
            let input_r = ref_glwe(r, &ct, &mut source);
            let mut input_t = t.glwe_alloc_from_infos(&ct);
            input_r.transfer_into(&mut input_t);
            let rotations = plan.rotations;
            let passes = rotations.baby_passes(plan.baby_steps.iter().copied());
            assert_eq!(passes.iter().map(|(_, pairs)| pairs.len()).collect::<Vec<_>>(), pass_sizes);
            let mut keyed: Vec<i64> = rotations.baby_key_rotations(plan.baby_steps.iter().copied());
            keyed.extend(rotations.giant_key_rotations(plan.giant_steps.iter().copied()));
            let mut keys_r = HashMap::new();
            let mut keys_t = HashMap::new();
            for rotation in keyed {
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
            let mut lhs_r = LinearTransformationBabySteps::alloc_with(r, &plan.baby_steps, rotations, &ct);
            let mut lhs_t = LinearTransformationBabySteps::alloc_with(t, &plan.baby_steps, rotations, &ct);
            r.glwe_prepare_linear_transformation_baby_steps(
                &mut lhs_r,
                &input_r,
                &keys_r,
                &mut poisoned_scratch::<BR>(r.glwe_prepare_linear_transformation_baby_steps_tmp_bytes(&ct, &plan, &key)).borrow(),
            );
            t.glwe_prepare_linear_transformation_baby_steps(
                &mut lhs_t,
                &input_t,
                &keys_t,
                &mut poisoned_scratch::<BT>(t.glwe_prepare_linear_transformation_baby_steps_tmp_bytes(&ct, &plan, &key)).borrow(),
            );

            let mut resident_r: LinearTransformation<PreparedDiagonal<BR::OwnedBuf, BR>> =
                LinearTransformation::alloc_prepared(r, &schedule, &pt);
            let mut resident_t: LinearTransformation<PreparedDiagonal<BT::OwnedBuf, BT>> =
                LinearTransformation::alloc_prepared(t, &schedule, &pt);
            let mut rhs_r = LinearTransformation {
                baby_steps: plan.baby_steps.clone(),
                rotations,
                giant_steps: Vec::new(),
            };
            let mut rhs_t = LinearTransformation {
                baby_steps: plan.baby_steps.clone(),
                rotations,
                giant_steps: Vec::new(),
            };
            for (&rot, babies) in plan.giant_steps.iter().zip(&plan.index) {
                let (mut diagonals_r, mut diagonals_t) = (Vec::new(), Vec::new());
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
            for out_base in [base, base - 1] {
                let out = GLWELayout {
                    base2k: out_base.into(),
                    ..ct
                };
                let mut out_r = ref_glwe(r, &out, &mut source);
                let mut out_t = t.glwe_alloc_from_infos(&out);
                out_r.transfer_into(&mut out_t);
                r.glwe_eval_linear_transformation_into(
                    base,
                    &mut out_r,
                    &lhs_r,
                    &resident_r,
                    &keys_r,
                    &mut poisoned_scratch::<BR>(r.glwe_eval_linear_transformation_tmp_bytes(&out, &ct, &pt, &plan, &key, &key))
                        .borrow(),
                );
                t.glwe_eval_linear_transformation_into(
                    base,
                    &mut out_t,
                    &lhs_t,
                    &resident_t,
                    &keys_t,
                    &mut poisoned_scratch::<BT>(t.glwe_eval_linear_transformation_tmp_bytes(&out, &ct, &pt, &plan, &key, &key))
                        .borrow(),
                );
                let mut have = r.glwe_alloc_from_infos(&out);
                out_t.transfer_into(&mut have);
                assert_glwe_eq!(
                    out_r,
                    have,
                    "powers-of-two linear transform rank={rank} key_base={key_base} k_digit={k_digit} out_base={out_base}"
                );
            }
        }
    }
    ciphertext_diagonals_parity(
        params,
        shapes,
        r,
        t,
        LinearTransformationStrategy::BsgsPowersOfTwo { giant_step: 4 },
        &mut source,
    );
}

/// Diagonals of the rank of the source on `strategy`'s schedule of `{0, 1, 4, 5}`, multiplied as ciphertexts: the
/// resident transformation, the streamed one, a two-term and a one-term sum, the same bits on both backends and in
/// either form. At rank 1, keys at the source's radix take the one-pass products from an offset of a limb, and with
/// digits of two limbs a backend's fused giant steps where it has them; keys one bit narrower take the tensor products
/// and the normalized giant route. Every call runs on the scratch its query returns.
fn ciphertext_diagonals_parity<BR, BT>(
    params: &TestParams,
    shapes: &ParityShapes,
    r: &Module<BR>,
    t: &Module<BT>,
    strategy: LinearTransformationStrategy,
    source: &mut Source,
) where
    BR: ParityBackend,
    BT: ParityBackend,
    BR::OwnedBuf: HostDataMut,
    Module<BR>: GLWELinearTransformations<BR>
        + GLWEAutomorphismKeyPreparedFactory<BR>
        + GLWETensorKeyPreparedFactory<BR>
        + CnvPVecAlloc<BR>,
    Module<BT>: GLWELinearTransformations<BT>
        + GLWEAutomorphismKeyPreparedFactory<BT>
        + GLWETensorKeyPreparedFactory<BT>
        + CnvPVecAlloc<BT>,
{
    let base = params.base2k.min(12);
    let k = 2 * base + 1;
    let schedule = LinearTransformationLayout {
        indexes: vec![0, 1, 4, 5],
        slots: params.n / 2,
        strategy,
    };
    let plan = schedule.index();
    let rotations = plan.rotations;
    // The giant rotations themselves are keyed per rotation, a tree's levels otherwise.
    let mut keyed: Vec<i64> = rotations.baby_key_rotations(plan.baby_steps.iter().copied());
    keyed.extend(rotations.giant_key_rotations(plan.giant_steps.iter().copied()));
    keyed.extend(plan.giant_steps.iter().filter(|&&rot| rot != 0));
    keyed.sort_unstable();
    keyed.dedup();
    for &rank in &shapes.ranks {
        let ct = GLWELayout {
            n: params.n.into(),
            base2k: base.into(),
            k: k.into(),
            rank: rank.into(),
        };
        // Digits off a limb boundary are the keys' concern, swept by the plaintext cases.
        let mut cases: Vec<(usize, usize)> = shapes.dsizes(k, base).into_iter().map(|dsize| (base, dsize * base)).collect();
        cases.push((base - 1, base - 1));
        // A higher rank takes the tensor products whatever the keys and the offset.
        let offsets: &[usize] = if rank == 1 {
            &[base - 1, 2 * base, 2 * base + 1]
        } else {
            cases.truncate(1);
            &[2 * base + 1]
        };
        for (key_base, k_digit) in cases {
            let key = GLWEAutomorphismKeyLayout {
                n: ct.n,
                base2k: key_base.into(),
                rank: ct.rank,
                dnum: Dnum(k.div_ceil(k_digit) as u32),
                k_digit: TorusPrecision(k_digit as u32),
                k_aux: TorusPrecision((k_digit + (params.n.ilog2() as usize)) as u32),
            };
            let tsk = GLWETensorKeyLayout {
                n: key.n,
                base2k: key.base2k,
                rank: key.rank,
                dnum: key.dnum,
                k_digit: key.k_digit,
                k_aux: key.k_aux,
            };
            let mut atks_r = HashMap::new();
            let mut atks_t = HashMap::new();
            for &rotation in &keyed {
                let p = r.galois_element(rotation);
                let mut key_r = r.glwe_automorphism_key_alloc_from_infos(&key);
                fill_by_digit(r, &mut key_r, 1, source);
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
                atks_r.insert(p, prepared_r);
                atks_t.insert(p, prepared_t);
            }
            let mut tsk_r = r.glwe_tensor_key_alloc_from_infos(&tsk);
            fill_by_digit(r, &mut tsk_r, 1, source);
            let mut tsk_t = t.glwe_tensor_key_alloc_from_infos(&tsk);
            tsk_r.transfer_into(&mut tsk_t);
            let mut tsk_prepared_r = r.alloc_tensor_key_prepared_from_infos(&tsk);
            let mut tsk_prepared_t = t.alloc_tensor_key_prepared_from_infos(&tsk);
            r.glwe_tensor_key_prepare(
                &mut tsk_prepared_r,
                &tsk_r,
                &mut poisoned_scratch::<BR>(r.glwe_tensor_key_prepare_tmp_bytes(&tsk)).borrow(),
            );
            t.glwe_tensor_key_prepare(
                &mut tsk_prepared_t,
                &tsk_t,
                &mut poisoned_scratch::<BT>(t.glwe_tensor_key_prepare_tmp_bytes(&tsk)).borrow(),
            );
            let keys_r = TransformKeys {
                automorphism: &atks_r,
                tensor: &tsk_prepared_r,
            };
            let keys_t = TransformKeys {
                automorphism: &atks_t,
                tensor: &tsk_prepared_t,
            };

            // The baby steps of a random source, the same on both backends.
            let baby_steps = |source: &mut Source| {
                let input_r = ref_glwe(r, &ct, source);
                let mut input_t = t.glwe_alloc_from_infos(&ct);
                input_r.transfer_into(&mut input_t);
                let mut lhs_r = LinearTransformationBabySteps::alloc_with(r, &plan.baby_steps, rotations, &ct);
                let mut lhs_t = LinearTransformationBabySteps::alloc_with(t, &plan.baby_steps, rotations, &ct);
                r.glwe_prepare_linear_transformation_baby_steps(
                    &mut lhs_r,
                    &input_r,
                    &atks_r,
                    &mut poisoned_scratch::<BR>(r.glwe_prepare_linear_transformation_baby_steps_tmp_bytes(&ct, &plan, &key))
                        .borrow(),
                );
                t.glwe_prepare_linear_transformation_baby_steps(
                    &mut lhs_t,
                    &input_t,
                    &atks_t,
                    &mut poisoned_scratch::<BT>(t.glwe_prepare_linear_transformation_baby_steps_tmp_bytes(&ct, &plan, &key))
                        .borrow(),
                );
                (lhs_r, lhs_t)
            };
            // Random ciphertext diagonals, the same on both backends, and their resident preparations.
            let transform = |source: &mut Source| {
                let mut rhs_r = LinearTransformation {
                    baby_steps: plan.baby_steps.clone(),
                    rotations,
                    giant_steps: Vec::new(),
                };
                let mut rhs_t = LinearTransformation {
                    baby_steps: plan.baby_steps.clone(),
                    rotations,
                    giant_steps: Vec::new(),
                };
                for (&rot, babies) in plan.giant_steps.iter().zip(&plan.index) {
                    let (mut diagonals_r, mut diagonals_t) = (Vec::new(), Vec::new());
                    for &baby in babies {
                        let diagonal_r = ref_glwe(r, &ct, source);
                        let mut diagonal_t = t.glwe_alloc_from_infos(&ct);
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
                    LinearTransformation::alloc_prepared(r, &schedule, &ct);
                let mut resident_t: LinearTransformation<PreparedDiagonal<BT::OwnedBuf, BT>> =
                    LinearTransformation::alloc_prepared(t, &schedule, &ct);
                r.glwe_prepare_linear_transformation_rhs(
                    &mut resident_r,
                    &rhs_r,
                    &mut poisoned_scratch::<BR>(r.glwe_prepare_linear_transformation_rhs_tmp_bytes(&ct)).borrow(),
                );
                t.glwe_prepare_linear_transformation_rhs(
                    &mut resident_t,
                    &rhs_t,
                    &mut poisoned_scratch::<BT>(t.glwe_prepare_linear_transformation_rhs_tmp_bytes(&ct)).borrow(),
                );
                (rhs_r, rhs_t, resident_r, resident_t)
            };
            let (lhs_r, lhs_t) = baby_steps(source);
            let (lhs2_r, lhs2_t) = baby_steps(source);
            let (rhs_r, rhs_t, resident_r, resident_t) = transform(source);
            let (_, _, resident2_r, resident2_t) = transform(source);
            let mut have = r.glwe_alloc_from_infos(&ct);
            for &offset in offsets {
                let label = format!("rank={rank} key_base={key_base} k_digit={k_digit} offset={offset} {strategy:?}");
                let mut out_r = ref_glwe(r, &ct, source);
                let mut out_t = t.glwe_alloc_from_infos(&ct);
                out_r.transfer_into(&mut out_t);
                r.glwe_eval_linear_transformation_into(
                    offset,
                    &mut out_r,
                    &lhs_r,
                    &resident_r,
                    &keys_r,
                    &mut poisoned_scratch::<BR>(r.glwe_eval_linear_transformation_tmp_bytes(&ct, &ct, &ct, &plan, &tsk, &key))
                        .borrow(),
                );
                t.glwe_eval_linear_transformation_into(
                    offset,
                    &mut out_t,
                    &lhs_t,
                    &resident_t,
                    &keys_t,
                    &mut poisoned_scratch::<BT>(t.glwe_eval_linear_transformation_tmp_bytes(&ct, &ct, &ct, &plan, &tsk, &key))
                        .borrow(),
                );
                out_t.transfer_into(&mut have);
                assert_glwe_eq!(out_r, have, "resident ciphertext-diagonal transform {label}");
                let mut streamed_r = ref_glwe(r, &ct, source);
                let mut streamed_t = t.glwe_alloc_from_infos(&ct);
                streamed_r.transfer_into(&mut streamed_t);
                r.glwe_eval_linear_transformation_into(
                    offset,
                    &mut streamed_r,
                    &lhs_r,
                    &rhs_r,
                    &keys_r,
                    &mut poisoned_scratch::<BR>(
                        r.glwe_eval_linear_transformation_unprepared_rhs_tmp_bytes(&ct, &ct, &ct, &plan, &tsk, &key),
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
                        t.glwe_eval_linear_transformation_unprepared_rhs_tmp_bytes(&ct, &ct, &ct, &plan, &tsk, &key),
                    )
                    .borrow(),
                );
                streamed_t.transfer_into(&mut have);
                assert_glwe_eq!(streamed_r, have, "streamed ciphertext-diagonal transform {label}");
                assert_glwe_eq!(out_r, streamed_r, "resident/streamed ciphertext-diagonal transform {label}");
                let mut sum_r = ref_glwe(r, &ct, source);
                let mut sum_t = t.glwe_alloc_from_infos(&ct);
                sum_r.transfer_into(&mut sum_t);
                r.glwe_eval_linear_transformation_sum_into(
                    offset,
                    &mut sum_r,
                    &[&lhs_r, &lhs2_r],
                    &[&resident_r, &resident2_r],
                    &keys_r,
                    &mut poisoned_scratch::<BR>(
                        r.glwe_eval_linear_transformation_sum_tmp_bytes(&ct, &ct, &ct, &plan, &tsk, &key),
                    )
                    .borrow(),
                );
                t.glwe_eval_linear_transformation_sum_into(
                    offset,
                    &mut sum_t,
                    &[&lhs_t, &lhs2_t],
                    &[&resident_t, &resident2_t],
                    &keys_t,
                    &mut poisoned_scratch::<BT>(
                        t.glwe_eval_linear_transformation_sum_tmp_bytes(&ct, &ct, &ct, &plan, &tsk, &key),
                    )
                    .borrow(),
                );
                sum_t.transfer_into(&mut have);
                assert_glwe_eq!(sum_r, have, "summed ciphertext-diagonal transform {label}");
                t.glwe_eval_linear_transformation_sum_into(
                    offset,
                    &mut sum_t,
                    &[&lhs_t],
                    &[&resident_t],
                    &keys_t,
                    &mut poisoned_scratch::<BT>(
                        t.glwe_eval_linear_transformation_sum_tmp_bytes(&ct, &ct, &ct, &plan, &tsk, &key),
                    )
                    .borrow(),
                );
                sum_t.transfer_into(&mut have);
                assert_glwe_eq!(out_r, have, "one-term ciphertext-diagonal transform sum {label}");
            }
        }
    }
}
