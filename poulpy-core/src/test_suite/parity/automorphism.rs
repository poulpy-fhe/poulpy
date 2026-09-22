//! Automorphism parity: the key-switch digit loop composed with a Galois map.

use poulpy_hal::{
    api::{ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{FillUniform, HostDataMut, Module, ScratchOwned},
    source::Source,
    test_suite::TestParams,
};

use crate::layouts::prepared::GLWEAutomorphismKeyPreparedToBackendRef;
use crate::{
    GLWEAutomorphism,
    api::TransferInto,
    layouts::{
        BackendGLWEAutomorphismKey, Base2K, Degree, Dnum, Dsize, GLWEAutomorphismKeyLayout, GLWELayout, ModuleCoreAlloc, Rank,
        TorusPrecision, prepared::GLWEAutomorphismKeyPreparedFactory,
    },
    test_suite::parity::{ParityBackend, ParityShapes, poisoned_scratch, ref_glwe},
};

/// Allocates an automorphism key on the reference module, filled with noise.
fn ref_key<BR>(
    module_ref: &Module<BR>,
    infos: &GLWEAutomorphismKeyLayout,
    p: i64,
    source: &mut Source,
) -> BackendGLWEAutomorphismKey<BR>
where
    BR: ParityBackend,
    BR::OwnedBuf: HostDataMut,
{
    let mut key = module_ref.glwe_automorphism_key_alloc_from_infos(infos);
    key.key.fill_uniform(infos.base2k.into(), source);
    key.p = p;
    key
}

/// `glwe_automorphism` agrees with the selected comparison backend byte-for-byte.
pub fn test_glwe_automorphism_parity<BR, BT>(
    params: &TestParams,
    shapes: &ParityShapes,
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    BR: ParityBackend,
    BT: ParityBackend,
    BR::OwnedBuf: HostDataMut,
    Module<BR>: GLWEAutomorphism<BR> + GLWEAutomorphismKeyPreparedFactory<BR>,
    Module<BT>: GLWEAutomorphism<BT> + GLWEAutomorphismKeyPreparedFactory<BT>,
    ScratchOwned<BR>: ScratchOwnedAlloc<BR> + ScratchOwnedBorrow<BR>,
    ScratchOwned<BT>: ScratchOwnedAlloc<BT> + ScratchOwnedBorrow<BT>,
{
    assert_eq!(module_ref.n(), module_test.n());

    let n = module_ref.n() as u32;
    let base2k = params.base2k;
    let k = 4 * base2k + 1;
    let mut source = Source::new([31u8; 32]);

    for &rank in &shapes.ranks {
        for dsize in shapes.dsizes(k, base2k) {
            for p in [-1i64, 5] {
                let a_infos = GLWELayout {
                    n: Degree(n),
                    base2k: Base2K(base2k as u32),
                    k: TorusPrecision(k as u32),
                    rank: Rank(rank as u32),
                };
                let res_infos = GLWELayout {
                    n: Degree(n),
                    base2k: Base2K(base2k as u32),
                    k: TorusPrecision((k + base2k * dsize) as u32),
                    rank: Rank(rank as u32),
                };
                let key_infos = GLWEAutomorphismKeyLayout {
                    n: Degree(n),
                    base2k: Base2K(base2k as u32),
                    dnum: Dnum(k.div_ceil(base2k * dsize) as u32),
                    dsize: Dsize(dsize as u32),
                    k_aux: TorusPrecision((dsize * base2k) as u32),
                    rank: Rank(rank as u32),
                };

                let a_ref = ref_glwe(module_ref, &a_infos, &mut source);
                let key_ref_coeffs = ref_key(module_ref, &key_infos, p, &mut source);

                let mut a_test = module_test.glwe_alloc_from_infos(&a_infos);
                a_ref.transfer_into(&mut a_test);
                let mut key_test_coeffs = module_test.glwe_automorphism_key_alloc_from_infos(&key_infos);
                key_ref_coeffs.transfer_into(&mut key_test_coeffs);

                let mut res_ref = module_ref.glwe_alloc_from_infos(&res_infos);
                let mut res_test = module_test.glwe_alloc_from_infos(&res_infos);

                let mut scratch_ref = poisoned_scratch::<BR>(module_ref.glwe_automorphism_key_prepare_tmp_bytes(&key_infos));
                let mut scratch_test = poisoned_scratch::<BT>(module_test.glwe_automorphism_key_prepare_tmp_bytes(&key_infos));

                let mut key_ref = module_ref.glwe_automorphism_key_prepared_alloc_from_infos(&key_infos);
                module_ref.glwe_automorphism_key_prepare(&mut key_ref, &key_ref_coeffs, &mut scratch_ref.borrow());
                let mut key_test = module_test.glwe_automorphism_key_prepared_alloc_from_infos(&key_infos);
                module_test.glwe_automorphism_key_prepare(&mut key_test, &key_test_coeffs, &mut scratch_test.borrow());

                // Every variant starts with the same independently uploaded accumulator.
                let initial = ref_glwe(module_ref, &res_infos, &mut source);
                macro_rules! check {
                    ($method:ident, $assign:tt) => {{
                        initial.transfer_into(&mut res_ref);
                        initial.transfer_into(&mut res_test);
                        let input_infos = if $assign { &res_infos } else { &a_infos };
                        let mut scratch_ref = poisoned_scratch::<BR>(module_ref.glwe_automorphism_tmp_bytes(&res_infos, input_infos, &key_infos));
                        let mut scratch_test = poisoned_scratch::<BT>(module_test.glwe_automorphism_tmp_bytes(&res_infos, input_infos, &key_infos));
                        check!(@call module_ref, res_ref, a_ref, key_ref, scratch_ref, $method, $assign);
                        check!(@call module_test, res_test, a_test, key_test, scratch_test, $method, $assign);
                        let mut have = module_ref.glwe_alloc_from_infos(&res_infos);
                        res_test.transfer_into(&mut have);
                        assert_eq!(res_ref, have, "{}: rank={rank} dsize={dsize} p={p}", stringify!($method));
                    }};
                    (@call $module:ident, $res:ident, $a:ident, $key:ident, $scratch:ident, $method:ident, false) => {
                        $module.$method(&mut $res, &$a, &$key.to_backend_ref(), &mut $scratch.borrow());
                    };
                    (@call $module:ident, $res:ident, $a:ident, $key:ident, $scratch:ident, $method:ident, true) => {
                        $module.$method(&mut $res, &$key.to_backend_ref(), &mut $scratch.borrow());
                    };
                }
                check!(glwe_automorphism, false);
                check!(glwe_automorphism_add, false);
                check!(glwe_automorphism_sub, false);
                check!(glwe_automorphism_sub_negate, false);
                check!(glwe_automorphism_assign, true);
                check!(glwe_automorphism_add_assign, true);
                check!(glwe_automorphism_sub_assign, true);
                check!(glwe_automorphism_sub_negate_assign, true);
            }
        }
    }
}
