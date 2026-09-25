//! Composite gadget operations, with each backend preparing its own keys.

use super::{ParityBackend, ParityShapes, poisoned_scratch, ref_gglwe};
use crate::layouts::GLWEToBackendMut;
use crate::layouts::LWEInfos;
use crate::{
    GGLWEExternalProduct, GGSWAutomorphism, GGSWExpandRows, GGSWExternalProduct, GGSWFromGGLWE, GGSWKeyswitch,
    GLWEAutomorphismKeyAutomorphism,
    api::TransferInto,
    layouts::{
        Base2K, Degree, Dnum, Dsize, GGLWEAtViewMut, GGLWELayout, GGLWEToGGSWKeyLayout, GGSWAtViewMut, GGSWLayout,
        GLWEAutomorphismKeyLayout, ModuleCoreAlloc, Rank, TorusPrecision,
        prepared::{
            GGLWEPreparedFactory, GGLWEPreparedToBackendRef, GGLWEToGGSWKeyPreparedFactory, GGLWEToGGSWKeyPreparedToBackendRef,
            GGSWPreparedFactory, GGSWPreparedToBackendRef, GLWEAutomorphismKeyPreparedFactory,
            GLWEAutomorphismKeyPreparedToBackendRef,
        },
    },
    test_suite::keys::fill_by_digit,
};
use poulpy_hal::api::VecZnxFillUniformSourceAll;
use poulpy_hal::{
    api::{ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{DataView, HostDataMut, Module, ScratchOwned},
    source::Source,
    test_suite::TestParams,
};

/// Compare GGLWE and GGSW external products, including assign forms.
pub fn test_gadget_external_product_parity<BR, BT>(params: &TestParams, shapes: &ParityShapes, r: &Module<BR>, t: &Module<BT>)
where
    BR: ParityBackend,
    BT: ParityBackend,
    BR::OwnedBuf: HostDataMut,
    Module<BR>: GGLWEExternalProduct<BR> + GGSWExternalProduct<BR> + GGSWPreparedFactory<BR> + VecZnxFillUniformSourceAll<BR>,
    Module<BT>: GGLWEExternalProduct<BT> + GGSWExternalProduct<BT> + GGSWPreparedFactory<BT>,
    ScratchOwned<BR>: ScratchOwnedAlloc<BR> + ScratchOwnedBorrow<BR>,
    ScratchOwned<BT>: ScratchOwnedAlloc<BT> + ScratchOwnedBorrow<BT>,
{
    let mut source = Source::new([107; 32]);
    let b = params.base2k;
    for &rank in &shapes.ranks {
        for dsize in shapes.dsizes(2 * b + 1, b) {
            let g = GGSWLayout {
                n: Degree(r.n() as u32),
                base2k: Base2K(b as u32),
                dnum: Dnum(3),
                dsize: Dsize(1),
                k_aux: TorusPrecision(b as u32 + 1),
                rank: Rank(rank as u32),
            };
            let k = GGSWLayout {
                dsize: Dsize(dsize as u32),
                k_aux: TorusPrecision((b * dsize + 1) as u32),
                dnum: Dnum((4 * b + 1).div_ceil(b * dsize) as u32),
                ..g
            };
            let h = GGLWELayout {
                n: g.n,
                base2k: g.base2k,
                dnum: g.dnum,
                dsize: g.dsize,
                k_aux: g.k_aux,
                rank_in: g.rank,
                rank_out: g.rank,
                stride: 1,
            };
            let mut key_r = r.ggsw_alloc_from_infos(&k);
            for row in 0..k.dnum.as_usize() {
                for col in 0..rank + 1 {
                    r.vec_znx_fill_uniform_source_all(
                        b,
                        key_r.k().as_usize(),
                        &mut GLWEToBackendMut::<BR>::to_backend_mut(&mut key_r.at_view_mut(row, col)).data_mut(),
                        &mut source,
                    );
                }
            }
            let mut key_t = t.ggsw_alloc_from_infos(&k);
            key_r.transfer_into(&mut key_t);
            let mut prep_r = r.ggsw_prepared_alloc_from_infos(&k);
            let mut prep_t = t.ggsw_prepared_alloc_from_infos(&k);
            r.ggsw_prepare(
                &mut prep_r,
                &key_r,
                &mut poisoned_scratch::<BR>(r.ggsw_prepare_tmp_bytes(&k)).borrow(),
            );
            t.ggsw_prepare(
                &mut prep_t,
                &key_t,
                &mut poisoned_scratch::<BT>(t.ggsw_prepare_tmp_bytes(&k)).borrow(),
            );
            macro_rules! check {
                ($alloc:ident, $infos:ident, $apply:ident, $assign:ident, $query:ident) => {{
                    let mut a_r = r.$alloc(&$infos);
                    let mut out_r = r.$alloc(&$infos);
                    let (rows, cols_in) = (a_r.data.rows(), a_r.data.cols_in());
                    for row in 0..rows {
                        for col in 0..cols_in {
                            r.vec_znx_fill_uniform_source_all(
                                b,
                                a_r.k().as_usize(),
                                &mut GLWEToBackendMut::<BR>::to_backend_mut(&mut a_r.at_view_mut(row, col)).data_mut(),
                                &mut source,
                            );
                            r.vec_znx_fill_uniform_source_all(
                                b,
                                out_r.k().as_usize(),
                                &mut GLWEToBackendMut::<BR>::to_backend_mut(&mut out_r.at_view_mut(row, col)).data_mut(),
                                &mut source,
                            );
                        }
                    }
                    let mut a_t = t.$alloc(&$infos);
                    a_r.transfer_into(&mut a_t);
                    let mut out_t = t.$alloc(&$infos);
                    out_r.transfer_into(&mut out_t);
                    r.$apply(
                        &mut out_r,
                        &a_r,
                        &prep_r.to_backend_ref(),
                        &mut poisoned_scratch::<BR>(r.$query(&$infos, &$infos, &k)).borrow(),
                    );
                    t.$apply(
                        &mut out_t,
                        &a_t,
                        &prep_t.to_backend_ref(),
                        &mut poisoned_scratch::<BT>(t.$query(&$infos, &$infos, &k)).borrow(),
                    );
                    let mut have = r.$alloc(&$infos);
                    out_t.transfer_into(&mut have);
                    assert_eq!(out_r, have, "{} rank={rank} dsize={dsize}", stringify!($apply));
                    // Also compare assign with out-of-place semantics, in addition to pairwise backend parity.
                    r.$assign(
                        &mut a_r,
                        &prep_r.to_backend_ref(),
                        &mut poisoned_scratch::<BR>(r.$query(&$infos, &$infos, &k)).borrow(),
                    );
                    t.$assign(
                        &mut a_t,
                        &prep_t.to_backend_ref(),
                        &mut poisoned_scratch::<BT>(t.$query(&$infos, &$infos, &k)).borrow(),
                    );
                    a_t.transfer_into(&mut have);
                    assert_eq!(a_r, have, "{} rank={rank} dsize={dsize}", stringify!($assign));
                    assert_eq!(a_r, out_r, "{} differs from into form", stringify!($assign));
                }};
            }
            check!(
                gglwe_alloc_from_infos,
                h,
                gglwe_external_product,
                gglwe_external_product_assign,
                gglwe_external_product_tmp_bytes
            );
            check!(
                ggsw_alloc_from_infos,
                g,
                ggsw_external_product,
                ggsw_external_product_assign,
                ggsw_external_product_tmp_bytes
            );
            // Prepared zeroing is checked through its mathematical effect.
            r.ggsw_zero(&mut prep_r);
            t.ggsw_zero(&mut prep_t);
            let mut a_r = r.ggsw_alloc_from_infos(&g);
            for row in 0..g.dnum.as_usize() {
                for col in 0..rank + 1 {
                    r.vec_znx_fill_uniform_source_all(
                        b,
                        a_r.k().as_usize(),
                        &mut GLWEToBackendMut::<BR>::to_backend_mut(&mut a_r.at_view_mut(row, col)).data_mut(),
                        &mut source,
                    );
                }
            }
            let mut a_t = t.ggsw_alloc_from_infos(&g);
            a_r.transfer_into(&mut a_t);
            r.ggsw_external_product_assign(
                &mut a_r,
                &prep_r.to_backend_ref(),
                &mut poisoned_scratch::<BR>(r.ggsw_external_product_tmp_bytes(&g, &g, &k)).borrow(),
            );
            t.ggsw_external_product_assign(
                &mut a_t,
                &prep_t.to_backend_ref(),
                &mut poisoned_scratch::<BT>(t.ggsw_external_product_tmp_bytes(&g, &g, &k)).borrow(),
            );
            assert!(
                BR::to_host_bytes(a_r.data.data()).iter().all(|&x| x == 0),
                "reference prepared zero product"
            );
            assert!(
                BT::to_host_bytes(a_t.data.data()).iter().all(|&x| x == 0),
                "backend prepared zero product"
            );
        }
    }
}

/// Compare gadget automorphism, GGSW keyswitching, row conversion and expansion.
/// The input/key polynomials are arbitrary canonical integer data: no matching
/// random streams or encryption assumptions enter the oracle.
pub fn test_gadget_conversion_parity<BR, BT>(params: &TestParams, shapes: &ParityShapes, r: &Module<BR>, t: &Module<BT>)
where
    BR: ParityBackend,
    BT: ParityBackend,
    BR::OwnedBuf: HostDataMut,
    Module<BR>: GGSWAutomorphism<BR>
        + GGSWKeyswitch<BR>
        + GLWEAutomorphismKeyAutomorphism<BR>
        + GGSWFromGGLWE<BR>
        + GGSWExpandRows<BR>
        + GGLWEPreparedFactory<BR>
        + GLWEAutomorphismKeyPreparedFactory<BR>
        + GGLWEToGGSWKeyPreparedFactory<BR>
        + VecZnxFillUniformSourceAll<BR>,
    Module<BT>: GGSWAutomorphism<BT>
        + GGSWKeyswitch<BT>
        + GLWEAutomorphismKeyAutomorphism<BT>
        + GGSWFromGGLWE<BT>
        + GGSWExpandRows<BT>
        + GGLWEPreparedFactory<BT>
        + GLWEAutomorphismKeyPreparedFactory<BT>
        + GGLWEToGGSWKeyPreparedFactory<BT>,
    ScratchOwned<BR>: ScratchOwnedAlloc<BR> + ScratchOwnedBorrow<BR>,
    ScratchOwned<BT>: ScratchOwnedAlloc<BT> + ScratchOwnedBorrow<BT>,
{
    let mut source = Source::new([109; 32]);
    let b = params.base2k;
    for &rank in &shapes.ranks {
        for dsize in shapes.dsizes(2 * b + 1, b) {
            let g = GGSWLayout {
                n: Degree(r.n() as u32),
                base2k: Base2K(b as u32),
                dnum: Dnum(3),
                dsize: Dsize(1),
                k_aux: TorusPrecision(b as u32 + 1),
                rank: Rank(rank as u32),
            };
            let k = GGLWELayout {
                n: g.n,
                base2k: g.base2k,
                dnum: Dnum((4 * b + 1).div_ceil(b * dsize) as u32),
                dsize: Dsize(dsize as u32),
                k_aux: TorusPrecision((b * dsize + 1) as u32),
                rank_in: g.rank,
                rank_out: g.rank,
                stride: 1,
            };
            let ak = GLWEAutomorphismKeyLayout {
                n: k.n,
                base2k: k.base2k,
                dnum: k.dnum,
                dsize: k.dsize,
                k_aux: k.k_aux,
                rank: g.rank,
            };
            let tk = GGLWEToGGSWKeyLayout {
                n: k.n,
                base2k: k.base2k,
                dnum: k.dnum,
                dsize: k.dsize,
                k_aux: k.k_aux,
                rank: g.rank,
            };
            let key_r = ref_gglwe(r, &k, &mut source);
            let mut key_t = t.gglwe_alloc_from_infos(&k);
            key_r.transfer_into(&mut key_t);
            let mut keyp_r = r.gglwe_prepared_alloc_from_infos(&k);
            let mut keyp_t = t.gglwe_prepared_alloc_from_infos(&k);
            r.gglwe_prepare(
                &mut keyp_r,
                &key_r,
                &mut poisoned_scratch::<BR>(r.gglwe_prepare_tmp_bytes(&k)).borrow(),
            );
            t.gglwe_prepare(
                &mut keyp_t,
                &key_t,
                &mut poisoned_scratch::<BT>(t.gglwe_prepare_tmp_bytes(&k)).borrow(),
            );
            let mut auto_r = r.glwe_automorphism_key_alloc_from_infos(&ak);
            fill_by_digit(r, &mut auto_r, 1, &mut source);
            auto_r.p = -5;
            let mut auto_t = t.glwe_automorphism_key_alloc_from_infos(&ak);
            auto_r.transfer_into(&mut auto_t);
            let mut autop_r = r.glwe_automorphism_key_prepared_alloc_from_infos(&ak);
            let mut autop_t = t.glwe_automorphism_key_prepared_alloc_from_infos(&ak);
            r.glwe_automorphism_key_prepare(
                &mut autop_r,
                &auto_r,
                &mut poisoned_scratch::<BR>(r.glwe_automorphism_key_prepare_tmp_bytes(&ak)).borrow(),
            );
            t.glwe_automorphism_key_prepare(
                &mut autop_t,
                &auto_t,
                &mut poisoned_scratch::<BT>(t.glwe_automorphism_key_prepare_tmp_bytes(&ak)).borrow(),
            );
            let mut tensor_r = r.gglwe_to_ggsw_key_alloc_from_infos(&tk);
            for key in tensor_r.keys.iter_mut() {
                fill_by_digit(r, key, 1, &mut source);
            }
            let mut tensor_t = t.gglwe_to_ggsw_key_alloc_from_infos(&tk);
            tensor_r.transfer_into(&mut tensor_t);
            let mut tensorp_r = r.gglwe_to_ggsw_key_prepared_alloc_from_infos(&tk);
            let mut tensorp_t = t.gglwe_to_ggsw_key_prepared_alloc_from_infos(&tk);
            r.gglwe_to_ggsw_key_prepare(
                &mut tensorp_r,
                &tensor_r,
                &mut poisoned_scratch::<BR>(r.gglwe_to_ggsw_key_prepare_tmp_bytes(&tk)).borrow(),
            );
            t.gglwe_to_ggsw_key_prepare(
                &mut tensorp_t,
                &tensor_t,
                &mut poisoned_scratch::<BT>(t.gglwe_to_ggsw_key_prepare_tmp_bytes(&tk)).borrow(),
            );
            let auto_view_r = GLWEAutomorphismKeyPreparedToBackendRef::to_backend_ref(&autop_r);
            let auto_view_t = GLWEAutomorphismKeyPreparedToBackendRef::to_backend_ref(&autop_t);
            let key_view_r = GGLWEPreparedToBackendRef::to_backend_ref(&keyp_r);
            let key_view_t = GGLWEPreparedToBackendRef::to_backend_ref(&keyp_t);
            let mut a_r = r.ggsw_alloc_from_infos(&g);
            for row in 0..g.dnum.as_usize() {
                for col in 0..rank + 1 {
                    r.vec_znx_fill_uniform_source_all(
                        b,
                        a_r.k().as_usize(),
                        &mut GLWEToBackendMut::<BR>::to_backend_mut(&mut a_r.at_view_mut(row, col)).data_mut(),
                        &mut source,
                    );
                }
            }
            let mut a_t = t.ggsw_alloc_from_infos(&g);
            a_r.transfer_into(&mut a_t);
            macro_rules! check {
                ($apply:ident, $assign:ident, $query:ident, $kr:ident, $kt:ident) => {{
                    let mut out_r = r.ggsw_alloc_from_infos(&g);
                    for row in 0..g.dnum.as_usize() {
                        for col in 0..rank + 1 {
                            r.vec_znx_fill_uniform_source_all(
                                b,
                                out_r.k().as_usize(),
                                &mut GLWEToBackendMut::<BR>::to_backend_mut(&mut out_r.at_view_mut(row, col)).data_mut(),
                                &mut source,
                            );
                        }
                    }
                    let mut out_t = t.ggsw_alloc_from_infos(&g);
                    out_r.transfer_into(&mut out_t);
                    r.$apply(
                        &mut out_r,
                        &a_r,
                        &$kr,
                        &tensorp_r.to_backend_ref(),
                        &mut poisoned_scratch::<BR>(r.$query(&g, &g, &k, &tk)).borrow(),
                    );
                    t.$apply(
                        &mut out_t,
                        &a_t,
                        &$kt,
                        &tensorp_t.to_backend_ref(),
                        &mut poisoned_scratch::<BT>(t.$query(&g, &g, &k, &tk)).borrow(),
                    );
                    let mut have = r.ggsw_alloc_from_infos(&g);
                    out_t.transfer_into(&mut have);
                    assert_eq!(out_r, have, "{} rank={rank} dsize={dsize}", stringify!($apply));
                    a_r.transfer_into(&mut out_r);
                    a_r.transfer_into(&mut out_t);
                    r.$assign(
                        &mut out_r,
                        &$kr,
                        &tensorp_r.to_backend_ref(),
                        &mut poisoned_scratch::<BR>(r.$query(&g, &g, &k, &tk)).borrow(),
                    );
                    t.$assign(
                        &mut out_t,
                        &$kt,
                        &tensorp_t.to_backend_ref(),
                        &mut poisoned_scratch::<BT>(t.$query(&g, &g, &k, &tk)).borrow(),
                    );
                    out_t.transfer_into(&mut have);
                    assert_eq!(out_r, have, "{} rank={rank} dsize={dsize}", stringify!($assign));
                }};
            }
            check!(
                ggsw_keyswitch,
                ggsw_keyswitch_assign,
                ggsw_keyswitch_tmp_bytes,
                key_view_r,
                key_view_t
            );
            check!(
                ggsw_automorphism,
                ggsw_automorphism_assign,
                ggsw_automorphism_tmp_bytes,
                auto_view_r,
                auto_view_t
            );
            // A destination with fewer gadget rows must only write its own rows.
            let short = GGSWLayout { dnum: Dnum(2), ..g };
            let mut short_r = r.ggsw_alloc_from_infos(&short);
            for row in 0..short.dnum.as_usize() {
                for col in 0..rank + 1 {
                    r.vec_znx_fill_uniform_source_all(
                        b,
                        short_r.k().as_usize(),
                        &mut GLWEToBackendMut::<BR>::to_backend_mut(&mut short_r.at_view_mut(row, col)).data_mut(),
                        &mut source,
                    );
                }
            }
            let mut short_t = t.ggsw_alloc_from_infos(&short);
            short_r.transfer_into(&mut short_t);
            r.ggsw_keyswitch(
                &mut short_r,
                &a_r,
                &key_view_r,
                &tensorp_r.to_backend_ref(),
                &mut poisoned_scratch::<BR>(r.ggsw_keyswitch_tmp_bytes(&short, &g, &k, &tk)).borrow(),
            );
            t.ggsw_keyswitch(
                &mut short_t,
                &a_t,
                &key_view_t,
                &tensorp_t.to_backend_ref(),
                &mut poisoned_scratch::<BT>(t.ggsw_keyswitch_tmp_bytes(&short, &g, &k, &tk)).borrow(),
            );
            let mut short_have = r.ggsw_alloc_from_infos(&short);
            short_t.transfer_into(&mut short_have);
            assert_eq!(
                short_r, short_have,
                "ggsw keyswitch shorter result: rank={rank} dsize={dsize}"
            );
            let mut out_r = r.glwe_automorphism_key_alloc_from_infos(&ak);
            let mut out_t = t.glwe_automorphism_key_alloc_from_infos(&ak);
            fill_by_digit(r, &mut out_r, 1, &mut source);
            out_r.transfer_into(&mut out_t);
            r.glwe_automorphism_key_automorphism(
                &mut out_r,
                &auto_r,
                &auto_view_r,
                &mut poisoned_scratch::<BR>(r.glwe_automorphism_key_automorphism_tmp_bytes(&ak, &ak, &ak)).borrow(),
            );
            t.glwe_automorphism_key_automorphism(
                &mut out_t,
                &auto_t,
                &auto_view_t,
                &mut poisoned_scratch::<BT>(t.glwe_automorphism_key_automorphism_tmp_bytes(&ak, &ak, &ak)).borrow(),
            );
            let mut have = r.glwe_automorphism_key_alloc_from_infos(&ak);
            out_t.transfer_into(&mut have);
            assert_eq!(out_r, have, "automorphism key into: rank={rank} dsize={dsize}");
            auto_r.transfer_into(&mut out_r);
            auto_r.transfer_into(&mut out_t);
            r.glwe_automorphism_key_automorphism_assign(
                &mut out_r,
                &auto_view_r,
                &mut poisoned_scratch::<BR>(r.glwe_automorphism_key_automorphism_tmp_bytes(&ak, &ak, &ak)).borrow(),
            );
            t.glwe_automorphism_key_automorphism_assign(
                &mut out_t,
                &auto_view_t,
                &mut poisoned_scratch::<BT>(t.glwe_automorphism_key_automorphism_tmp_bytes(&ak, &ak, &ak)).borrow(),
            );
            out_t.transfer_into(&mut have);
            assert_eq!(out_r, have, "automorphism key assign: rank={rank} dsize={dsize}");
            let h = GGLWELayout {
                dnum: g.dnum,
                dsize: g.dsize,
                k_aux: g.k_aux,
                ..k
            };
            let a = ref_gglwe(r, &h, &mut source);
            let mut a_t = t.gglwe_alloc_from_infos(&h);
            a.transfer_into(&mut a_t);
            let mut out_r = r.ggsw_alloc_from_infos(&g);
            for row in 0..g.dnum.as_usize() {
                for col in 0..rank + 1 {
                    r.vec_znx_fill_uniform_source_all(
                        b,
                        out_r.k().as_usize(),
                        &mut GLWEToBackendMut::<BR>::to_backend_mut(&mut out_r.at_view_mut(row, col)).data_mut(),
                        &mut source,
                    );
                }
            }
            let mut out_t = t.ggsw_alloc_from_infos(&g);
            out_r.transfer_into(&mut out_t);
            r.ggsw_from_gglwe(
                &mut out_r,
                &a,
                &tensorp_r.to_backend_ref(),
                &mut poisoned_scratch::<BR>(r.ggsw_from_gglwe_tmp_bytes(&g, &h, &tk)).borrow(),
            );
            t.ggsw_from_gglwe(
                &mut out_t,
                &a_t,
                &tensorp_t.to_backend_ref(),
                &mut poisoned_scratch::<BT>(t.ggsw_from_gglwe_tmp_bytes(&g, &h, &tk)).borrow(),
            );
            let mut have = r.ggsw_alloc_from_infos(&g);
            out_t.transfer_into(&mut have);
            assert_eq!(out_r, have, "ggsw_from_gglwe: rank={rank} dsize={dsize}");
            // Row expansion consumes an existing GGSW body row and replaces mask rows.
            a_r.transfer_into(&mut out_r);
            a_r.transfer_into(&mut out_t);
            r.ggsw_expand_row(
                &mut out_r,
                &tensorp_r.to_backend_ref(),
                &mut poisoned_scratch::<BR>(r.ggsw_expand_rows_tmp_bytes(&g, &tk)).borrow(),
            );
            t.ggsw_expand_row(
                &mut out_t,
                &tensorp_t.to_backend_ref(),
                &mut poisoned_scratch::<BT>(t.ggsw_expand_rows_tmp_bytes(&g, &tk)).borrow(),
            );
            out_t.transfer_into(&mut have);
            assert_eq!(out_r, have, "ggsw_expand_row: rank={rank} dsize={dsize}");
        }
    }
}
