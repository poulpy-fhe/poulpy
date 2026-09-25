//! Deterministic LWE/GLWE conversion and extraction parity.
use super::{ParityBackend, ParityShapes, poisoned_scratch, ref_glwe};
use crate::{
    GLWEExpandLWE, GLWEExpandLWEMatrix, GLWEFromLWE, GLWEMaskFill, LWEFillMask, LWEFromGLWE, LWEKeyswitch, LWESampleExtract,
    api::TransferInto,
    layouts::{
        Base2K, Degree, Dnum, GLWELayout, GLWEToLWEKeyLayout, LWEInfos, LWELayout, LWEMatrixInfos, LWEMatrixLayout,
        LWESwitchingKeyLayout, LWEToGLWEKeyLayout, ModuleCoreAlloc, Rank, TorusPrecision,
        prepared::{
            GGLWEPreparedToBackendRef, GLWEToLWEKeyPreparedFactory, LWESwitchingKeyPreparedFactory, LWEToGLWEKeyPreparedFactory,
        },
    },
    test_suite::keys::fill_by_digit,
};
use poulpy_hal::api::VecZnxFillUniformSource;
use poulpy_hal::{
    api::{ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{HostDataMut, Module, ScratchOwned},
    source::Source,
    test_suite::TestParams,
};

/// Covers all extraction/conversion methods with matching logical coefficients,
/// distinct preparation and exact scratch budgets for each backend.
pub fn test_lwe_conversion_parity<BR, BT>(params: &TestParams, shapes: &ParityShapes, r: &Module<BR>, t: &Module<BT>)
where
    BR: ParityBackend,
    BT: ParityBackend,
    BR::OwnedBuf: HostDataMut,
    Module<BR>: GLWEExpandLWE<BR>
        + GLWEExpandLWEMatrix<BR>
        + LWESampleExtract<BR>
        + GLWEFromLWE<BR>
        + LWEFromGLWE<BR>
        + LWEKeyswitch<BR>
        + GLWEToLWEKeyPreparedFactory<BR>
        + LWEToGLWEKeyPreparedFactory<BR>
        + LWESwitchingKeyPreparedFactory<BR>
        + GLWEMaskFill<BR>
        + VecZnxFillUniformSource<BR>
        + LWEFillMask<BR>,
    Module<BT>: GLWEExpandLWE<BT>
        + GLWEExpandLWEMatrix<BT>
        + LWESampleExtract<BT>
        + GLWEFromLWE<BT>
        + LWEFromGLWE<BT>
        + LWEKeyswitch<BT>
        + GLWEToLWEKeyPreparedFactory<BT>
        + LWEToGLWEKeyPreparedFactory<BT>
        + LWESwitchingKeyPreparedFactory<BT>,
    ScratchOwned<BR>: ScratchOwnedAlloc<BR> + ScratchOwnedBorrow<BR>,
    ScratchOwned<BT>: ScratchOwnedAlloc<BT> + ScratchOwnedBorrow<BT>,
{
    let b = params.base2k;
    let mut source = Source::new([113; 32]);
    for &rank in &shapes.ranks {
        for k in [b - 1, 2 * b, 2 * b + 1] {
            let g = GLWELayout {
                n: Degree(r.n() as u32),
                base2k: Base2K(b as u32),
                k: TorusPrecision(k as u32),
                rank: Rank(rank as u32),
            };
            let a_r = ref_glwe(r, &g, &mut source);
            let mut a_t = t.glwe_alloc_from_infos(&g);
            a_r.transfer_into(&mut a_t);
            let l = LWELayout {
                n: Degree((r.n() * rank) as u32),
                base2k: g.base2k,
                k: g.k,
            };
            for count in [1, r.n()] {
                let mut out_r: Vec<_> = (0..count)
                    .map(|_| {
                        let mut v = r.lwe_alloc_from_infos(&l);
                        r.fill_lwe_mask_from_source(b, &mut v, &mut source);
                        v
                    })
                    .collect();
                let mut out_t: Vec<_> = out_r
                    .iter()
                    .map(|v| {
                        let mut x = t.lwe_alloc_from_infos(&l);
                        v.transfer_into(&mut x);
                        x
                    })
                    .collect();
                r.glwe_expand_lwe(
                    &mut out_r,
                    &a_r,
                    &mut poisoned_scratch::<BR>(r.glwe_expand_lwe_tmp_bytes(&l, &g)).borrow(),
                );
                t.glwe_expand_lwe(
                    &mut out_t,
                    &a_t,
                    &mut poisoned_scratch::<BT>(t.glwe_expand_lwe_tmp_bytes(&l, &g)).borrow(),
                );
                for (i, (want, out)) in out_r.iter().zip(out_t.iter()).enumerate() {
                    let mut have = r.lwe_alloc_from_infos(&l);
                    out.transfer_into(&mut have);
                    assert_eq!(*want, have, "glwe_expand_lwe rank={rank} k={k} count={count} row={i}");
                }
                let m = LWEMatrixLayout {
                    rows: count,
                    n: l.n,
                    base2k: l.base2k,
                    k: l.k,
                };
                let mut matrix_r = r.lwe_matrix_alloc_from_infos(&m);
                let mut matrix_t = t.lwe_matrix_alloc_from_infos(&m);
                let bytes = BR::len_bytes(matrix_r.body.data());
                BR::copy_from_host(matrix_r.body.data_mut(), &vec![0x55; bytes]);
                let bytes = BR::len_bytes(matrix_r.mask.data());
                BR::copy_from_host(matrix_r.mask.data_mut(), &vec![0x55; bytes]);
                let bytes = BT::len_bytes(matrix_t.body.data());
                BT::copy_from_host(matrix_t.body.data_mut(), &vec![0x77; bytes]);
                let bytes = BT::len_bytes(matrix_t.mask.data());
                BT::copy_from_host(matrix_t.mask.data_mut(), &vec![0x77; bytes]);
                r.glwe_expand_lwe_matrix(
                    &mut matrix_r,
                    &a_r,
                    &mut poisoned_scratch::<BR>(r.glwe_expand_lwe_matrix_tmp_bytes(&m, &g)).borrow(),
                );
                t.glwe_expand_lwe_matrix(
                    &mut matrix_t,
                    &a_t,
                    &mut poisoned_scratch::<BT>(t.glwe_expand_lwe_matrix_tmp_bytes(&m, &g)).borrow(),
                );
                assert_eq!(matrix_r.lwe_layout(), matrix_t.lwe_layout());
                assert_eq!(matrix_r.rows(), matrix_t.rows());
                for (left, right) in [(&matrix_r.body, &matrix_t.body), (&matrix_r.mask, &matrix_t.mask)] {
                    let live = left.n() * left.cols() * left.size() * std::mem::size_of::<i64>();
                    let want = BR::to_host_bytes(left.data());
                    let have = BT::to_host_bytes(right.data());
                    assert_eq!(
                        &want[..live],
                        &have[..live],
                        "matrix logical output rank={rank} count={count}"
                    );
                    assert!(
                        want[live..].iter().all(|&byte| byte == 0x55),
                        "reference matrix padding changed"
                    );
                    assert!(
                        have[live..].iter().all(|&byte| byte == 0x77),
                        "backend matrix padding changed"
                    );
                }
            }
            // Sample extraction permits a truncated mask and copies the first GLWE mask column.
            for dimension in [1, r.n()] {
                let l = LWELayout {
                    n: Degree(dimension as u32),
                    ..l
                };
                let mut out_r = r.lwe_alloc_from_infos(&l);
                r.fill_lwe_mask_from_source(b, &mut out_r, &mut source);
                let mut out_t = t.lwe_alloc_from_infos(&l);
                out_r.transfer_into(&mut out_t);
                r.lwe_sample_extract(&mut out_r, &a_r);
                t.lwe_sample_extract(&mut out_t, &a_t);
                let mut have = r.lwe_alloc_from_infos(&l);
                out_t.transfer_into(&mut have);
                assert_eq!(out_r, have, "lwe_sample_extract rank={rank} k={k} n={dimension}");
            }
            let l = LWELayout {
                n: Degree((r.n() / 2) as u32),
                base2k: Base2K((b - 1) as u32),
                k: g.k,
            };
            let kt = GLWEToLWEKeyLayout {
                n: g.n,
                base2k: g.base2k,
                dnum: Dnum(k.div_ceil(b) as u32),
                k_aux: TorusPrecision((b + 1) as u32),
                rank_in: g.rank,
            };
            let mut key_r = r.glwe_to_lwe_key_alloc_from_infos(&kt);
            fill_by_digit(r, &mut key_r, 1, &mut source);
            let mut key_t = t.glwe_to_lwe_key_alloc_from_infos(&kt);
            key_r.transfer_into(&mut key_t);
            let mut kp_r = r.glwe_to_lwe_key_prepared_alloc_from_infos(&kt);
            let mut kp_t = t.glwe_to_lwe_key_prepared_alloc_from_infos(&kt);
            r.glwe_to_lwe_key_prepare(
                &mut kp_r,
                &key_r,
                &mut poisoned_scratch::<BR>(r.glwe_to_lwe_key_prepare_tmp_bytes(&kt)).borrow(),
            );
            t.glwe_to_lwe_key_prepare(
                &mut kp_t,
                &key_t,
                &mut poisoned_scratch::<BT>(t.glwe_to_lwe_key_prepare_tmp_bytes(&kt)).borrow(),
            );
            for index in [0, r.n() - 1] {
                let mut out_r = r.lwe_alloc_from_infos(&l);
                r.fill_lwe_mask_from_source(b - 1, &mut out_r, &mut source);
                let mut out_t = t.lwe_alloc_from_infos(&l);
                out_r.transfer_into(&mut out_t);
                r.lwe_from_glwe(
                    &mut out_r,
                    &a_r,
                    index,
                    &kp_r.to_backend_ref(),
                    &mut poisoned_scratch::<BR>(r.lwe_from_glwe_tmp_bytes(&l, &g, &kt)).borrow(),
                );
                t.lwe_from_glwe(
                    &mut out_t,
                    &a_t,
                    index,
                    &kp_t.to_backend_ref(),
                    &mut poisoned_scratch::<BT>(t.lwe_from_glwe_tmp_bytes(&l, &g, &kt)).borrow(),
                );
                let mut have = r.lwe_alloc_from_infos(&l);
                out_t.transfer_into(&mut have);
                assert_eq!(out_r, have, "lwe_from_glwe rank={rank} k={k} index={index}");
            }
            let kt = LWEToGLWEKeyLayout {
                n: g.n,
                base2k: g.base2k,
                dnum: Dnum(k.div_ceil(b) as u32),
                k_aux: TorusPrecision((b + 1) as u32),
                rank_out: g.rank,
            };
            let mut key_r = r.lwe_to_glwe_key_alloc_from_infos(&kt);
            fill_by_digit(r, &mut key_r, 1, &mut source);
            let mut key_t = t.lwe_to_glwe_key_alloc_from_infos(&kt);
            key_r.0.transfer_into(&mut key_t.0);
            let mut kp_r = r.lwe_to_glwe_key_prepared_alloc_from_infos(&kt);
            let mut kp_t = t.lwe_to_glwe_key_prepared_alloc_from_infos(&kt);
            r.lwe_to_glwe_key_prepare(
                &mut kp_r,
                &key_r,
                &mut poisoned_scratch::<BR>(r.lwe_to_glwe_key_prepare_tmp_bytes(&kt)).borrow(),
            );
            t.lwe_to_glwe_key_prepare(
                &mut kp_t,
                &key_t,
                &mut poisoned_scratch::<BT>(t.lwe_to_glwe_key_prepare_tmp_bytes(&kt)).borrow(),
            );
            let mut a_r = r.lwe_alloc_from_infos(&l);
            r.fill_lwe_mask_from_source(b - 1, &mut a_r, &mut source);
            let mut a_t = t.lwe_alloc_from_infos(&l);
            a_r.transfer_into(&mut a_t);
            let mut out_r = ref_glwe(r, &g, &mut source);
            let mut out_t = t.glwe_alloc_from_infos(&g);
            out_r.transfer_into(&mut out_t);
            r.glwe_from_lwe(
                &mut out_r,
                &a_r,
                &kp_r.to_backend_ref(),
                &mut poisoned_scratch::<BR>(r.glwe_from_lwe_tmp_bytes(&g, &l, &kt)).borrow(),
            );
            t.glwe_from_lwe(
                &mut out_t,
                &a_t,
                &kp_t.to_backend_ref(),
                &mut poisoned_scratch::<BT>(t.glwe_from_lwe_tmp_bytes(&g, &l, &kt)).borrow(),
            );
            let mut have = r.glwe_alloc_from_infos(&g);
            out_t.transfer_into(&mut have);
            assert_eq!(out_r, have, "glwe_from_lwe rank={rank} k={k}");
            let kt = LWESwitchingKeyLayout {
                n: g.n,
                base2k: g.base2k,
                dnum: Dnum(k.div_ceil(b) as u32),
                k_aux: TorusPrecision((b + 1) as u32),
            };
            let mut key_r = r.lwe_switching_key_alloc_from_infos(&kt);
            fill_by_digit(r, &mut key_r, 1, &mut source);
            let mut key_t = t.lwe_switching_key_alloc_from_infos(&kt);
            key_r.0.transfer_into(&mut key_t.0);
            let mut kp_r = r.lwe_switching_key_prepared_alloc_from_infos(&kt);
            let mut kp_t = t.lwe_switching_key_prepared_alloc_from_infos(&kt);
            r.lwe_switching_key_prepare(
                &mut kp_r,
                &key_r,
                &mut poisoned_scratch::<BR>(r.lwe_switching_key_prepare_tmp_bytes(&kt)).borrow(),
            );
            t.lwe_switching_key_prepare(
                &mut kp_t,
                &key_t,
                &mut poisoned_scratch::<BT>(t.lwe_switching_key_prepare_tmp_bytes(&kt)).borrow(),
            );
            let res = LWELayout {
                n: Degree((r.n() / 4) as u32),
                base2k: Base2K((b - 2) as u32),
                ..l
            };
            let mut out_r = r.lwe_alloc_from_infos(&res);
            r.fill_lwe_mask_from_source(b - 2, &mut out_r, &mut source);
            let mut out_t = t.lwe_alloc_from_infos(&res);
            out_r.transfer_into(&mut out_t);
            r.lwe_keyswitch(
                &mut out_r,
                &a_r,
                &kp_r.to_backend_ref(),
                &mut poisoned_scratch::<BR>(r.lwe_keyswitch_tmp_bytes(&res, &l, &kt)).borrow(),
            );
            t.lwe_keyswitch(
                &mut out_t,
                &a_t,
                &kp_t.to_backend_ref(),
                &mut poisoned_scratch::<BT>(t.lwe_keyswitch_tmp_bytes(&res, &l, &kt)).borrow(),
            );
            let mut have = r.lwe_alloc_from_infos(&res);
            out_t.transfer_into(&mut have);
            assert_eq!(out_r, have, "lwe_keyswitch rank={rank} k={k}");
        }
    }
}
