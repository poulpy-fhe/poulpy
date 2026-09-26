//! Prepared allocation/size-helper contracts. Opaque prepared bytes are never
//! compared between backends; parity of their logical consumers lives beside
//! this suite. Each preparation receives only its own advertised scratch.
use crate::{
    layouts::*,
    test_suite::parity::{ParityBackend, ParityShapes, poisoned_scratch},
};
use poulpy_hal::{
    api::{ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{DataView, Module, ScratchOwned},
    test_suite::TestParams,
};

pub trait PreparationBounds<BE: ParityBackend>:
    ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = i64>
    + GLWEPreparedFactory<BE>
    + GLWEPublicKeyPreparedFactory<BE>
    + GLWESecretPreparedFactory<BE>
    + GLWESecretTensorPreparedFactory<BE>
    + GGLWEPreparedFactory<BE>
    + GGSWPreparedFactory<BE>
    + GLWEAutomorphismKeyPreparedFactory<BE>
    + GLWESwitchingKeyPreparedFactory<BE>
    + GLWETensorKeyPreparedFactory<BE>
    + LWESwitchingKeyPreparedFactory<BE>
    + GLWEToLWEKeyPreparedFactory<BE>
    + LWEToGLWEKeyPreparedFactory<BE>
    + GGLWEToGGSWKeyPreparedFactory<BE>
{
}
impl<BE: ParityBackend> PreparationBounds<BE> for Module<BE> where
    Self: ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = i64>
        + GLWEPreparedFactory<BE>
        + GLWEPublicKeyPreparedFactory<BE>
        + GLWESecretPreparedFactory<BE>
        + GLWESecretTensorPreparedFactory<BE>
        + GGLWEPreparedFactory<BE>
        + GGSWPreparedFactory<BE>
        + GLWEAutomorphismKeyPreparedFactory<BE>
        + GLWESwitchingKeyPreparedFactory<BE>
        + GLWETensorKeyPreparedFactory<BE>
        + LWESwitchingKeyPreparedFactory<BE>
        + GLWEToLWEKeyPreparedFactory<BE>
        + LWEToGLWEKeyPreparedFactory<BE>
        + GGLWEToGGSWKeyPreparedFactory<BE>
{
}

fn exercise<BE: ParityBackend>(params: &TestParams, shapes: &ParityShapes, m: &Module<BE>)
where
    Module<BE>: PreparationBounds<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    macro_rules! check {
        ($alloc:ident,$from:ident,$size:ident,$size_from:ident,$infos:expr,[$($arg:expr),*],$p:ident,$bytes:expr) => {{
            let expected=m.$size($($arg),*);
            let $p=m.$alloc($($arg),*);
            assert_eq!($bytes,expected,"allocation size: {}",stringify!($alloc));
            let $p=m.$from(&$infos);
            assert_eq!($bytes,expected,"allocation from infos: {}",stringify!($from));
            assert_eq!(m.$size_from(&$infos),expected,"size from infos: {}",stringify!($size_from));
            $p
        }};
    }
    for &rank in &shapes.ranks {
        for dsize in [1, 2] {
            let rank = Rank(rank as u32);
            let b = Base2K(params.base2k as u32);
            let k = TorusPrecision(2 * b.as_u32() + 1);
            let dn = Dnum(2);
            let ds = Dsize(dsize);
            let aux = TorusPrecision(b.as_u32() * dsize + 1);
            let plain = GLWELayout {
                n: Degree(m.n() as u32),
                base2k: b,
                k,
                rank,
            };
            let gadget = GGLWELayout {
                n: plain.n,
                base2k: b,
                dnum: dn,
                dsize: ds,
                k_aux: aux,
                rank_in: rank,
                rank_out: rank,
                stride: 1,
            };
            let tensor_infos = GGLWELayout {
                rank_in: Rank((rank.as_u32() + 1) * rank.as_u32() / 2),
                ..gadget
            };
            let ggsw = GGSWLayout {
                n: plain.n,
                base2k: b,
                dnum: dn,
                dsize: ds,
                k_aux: aux,
                rank,
            };
            let lwe = GGLWELayout {
                dsize: Dsize(1),
                rank_in: Rank(1),
                rank_out: Rank(1),
                ..gadget
            };
            let to_lwe = GGLWELayout { rank_in: rank, ..lwe };
            let from_lwe = GGLWELayout { rank_out: rank, ..lwe };
            let mut prepared = check!(
                glwe_prepared_alloc,
                glwe_prepared_alloc_from_infos,
                glwe_prepared_bytes_of,
                glwe_prepared_bytes_of_from_infos,
                plain,
                [b, k, rank],
                p,
                BE::len_bytes(p.data.data())
            );
            let source = m.glwe_alloc_from_infos(&plain);
            m.glwe_prepare(
                &mut prepared,
                &source,
                &mut poisoned_scratch::<BE>(m.glwe_prepare_tmp_bytes(&plain)).borrow(),
            );
            let mut prepared = check!(
                glwe_public_key_prepared_alloc,
                glwe_public_key_prepared_alloc_from_infos,
                glwe_public_key_prepared_bytes_of,
                glwe_public_key_prepared_bytes_of_from_infos,
                plain,
                [b, k, rank],
                p,
                BE::len_bytes(p.key.data.data())
            );
            let source = m.glwe_public_key_alloc_from_infos(&plain);
            m.glwe_public_key_prepare(
                &mut prepared,
                &source,
                &mut poisoned_scratch::<BE>(m.glwe_public_key_prepare_tmp_bytes(&plain)).borrow(),
            );
            let mut prepared = check!(
                glwe_secret_prepared_alloc,
                glwe_secret_prepared_alloc_from_infos,
                glwe_secret_prepared_bytes_of,
                glwe_secret_prepared_bytes_of_from_infos,
                plain,
                [rank],
                p,
                BE::len_bytes(p.data.data())
            );
            let source = m.glwe_secret_alloc_from_infos(&plain);
            m.glwe_secret_prepare(&mut prepared, &source);
            let mut prepared = check!(
                glwe_secret_tensor_prepared_alloc,
                glwe_secret_tensor_prepared_alloc_from_infos,
                glwe_secret_tensor_prepared_bytes_of,
                glwe_secret_tensor_prepared_bytes_of_from_infos,
                plain,
                [rank],
                p,
                BE::len_bytes(p.data.data())
            );
            let source = m.glwe_secret_tensor_alloc_from_infos(&plain);
            m.glwe_secret_tensor_prepared_prepare(&mut prepared, &source);
            let mut prepared = check!(
                gglwe_prepared_alloc,
                gglwe_prepared_alloc_from_infos,
                gglwe_prepared_bytes_of,
                gglwe_prepared_bytes_of_from_infos,
                gadget,
                [b, dn, ds, aux, rank, rank],
                p,
                BE::len_bytes(p.data.data())
            );
            let source = m.gglwe_alloc_from_infos(&gadget);
            m.gglwe_prepare(
                &mut prepared,
                &source,
                &mut poisoned_scratch::<BE>(m.gglwe_prepare_tmp_bytes(&source)).borrow(),
            );
            let mut prepared = check!(
                ggsw_prepared_alloc,
                ggsw_prepared_alloc_from_infos,
                ggsw_prepared_bytes_of,
                ggsw_prepared_bytes_of_from_infos,
                ggsw,
                [b, dn, ds, aux, rank],
                p,
                BE::len_bytes(p.data.data())
            );
            let source = m.ggsw_alloc_from_infos(&ggsw);
            m.ggsw_prepare(
                &mut prepared,
                &source,
                &mut poisoned_scratch::<BE>(m.ggsw_prepare_tmp_bytes(&source)).borrow(),
            );
            let mut prepared = check!(
                glwe_automorphism_key_prepared_alloc,
                glwe_automorphism_key_prepared_alloc_from_infos,
                glwe_automorphism_key_prepared_bytes_of,
                glwe_automorphism_key_prepared_bytes_of_from_infos,
                gadget,
                [b, dn, ds, aux, rank],
                p,
                BE::len_bytes(p.key.data.data())
            );
            let source = m.glwe_automorphism_key_alloc_from_infos(&gadget);
            m.glwe_automorphism_key_prepare(
                &mut prepared,
                &source,
                &mut poisoned_scratch::<BE>(m.glwe_automorphism_key_prepare_tmp_bytes(&source)).borrow(),
            );
            let mut prepared = check!(
                glwe_switching_key_prepared_alloc,
                glwe_switching_key_prepared_alloc_from_infos,
                bytes_of_glwe_key_prepared,
                glwe_switching_key_prepared_bytes_of_from_infos,
                gadget,
                [b, dn, ds, aux, rank, rank],
                p,
                BE::len_bytes(p.key.data.data())
            );
            let source = m.glwe_switching_key_alloc_from_infos(&gadget);
            m.glwe_switching_key_prepare(
                &mut prepared,
                &source,
                &mut poisoned_scratch::<BE>(m.glwe_switching_key_prepare_tmp_bytes(&source)).borrow(),
            );
            let mut prepared = check!(
                alloc_tensor_key_prepared,
                alloc_tensor_key_prepared_from_infos,
                bytes_of_tensor_key_prepared,
                bytes_of_tensor_key_prepared_from_infos,
                tensor_infos,
                [b, dn, ds, aux, rank],
                p,
                BE::len_bytes(p.0.data.data())
            );
            let source = m.glwe_tensor_key_alloc_from_infos(&tensor_infos);
            m.prepare_tensor_key(
                &mut prepared,
                &source,
                &mut poisoned_scratch::<BE>(m.prepare_tensor_key_tmp_bytes(&source)).borrow(),
            );
            let mut prepared = check!(
                lwe_switching_key_prepared_alloc,
                lwe_switching_key_prepared_alloc_from_infos,
                lwe_switching_key_prepared_bytes_of,
                lwe_switching_key_prepared_bytes_of_from_infos,
                lwe,
                [b, dn, aux],
                p,
                BE::len_bytes(p.0.key.data.data())
            );
            let source = m.lwe_switching_key_alloc_from_infos(&lwe);
            m.lwe_switching_key_prepare(
                &mut prepared,
                &source,
                &mut poisoned_scratch::<BE>(m.lwe_switching_key_prepare_tmp_bytes(&source)).borrow(),
            );
            let mut prepared = check!(
                glwe_to_lwe_key_prepared_alloc,
                glwe_to_lwe_key_prepared_alloc_from_infos,
                glwe_to_lwe_key_prepared_bytes_of,
                glwe_to_lwe_key_prepared_bytes_of_from_infos,
                to_lwe,
                [b, dn, aux, rank],
                p,
                BE::len_bytes(p.0.key.data.data())
            );
            let source = m.glwe_to_lwe_key_alloc_from_infos(&to_lwe);
            m.glwe_to_lwe_key_prepare(
                &mut prepared,
                &source,
                &mut poisoned_scratch::<BE>(m.glwe_to_lwe_key_prepare_tmp_bytes(&source)).borrow(),
            );
            let mut prepared = check!(
                lwe_to_glwe_key_prepared_alloc,
                lwe_to_glwe_key_prepared_alloc_from_infos,
                lwe_to_glwe_key_prepared_bytes_of,
                lwe_to_glwe_key_prepared_bytes_of_from_infos,
                from_lwe,
                [b, dn, aux, rank],
                p,
                BE::len_bytes(p.0.key.data.data())
            );
            let source = m.lwe_to_glwe_key_alloc_from_infos(&from_lwe);
            m.lwe_to_glwe_key_prepare(
                &mut prepared,
                &source,
                &mut poisoned_scratch::<BE>(m.lwe_to_glwe_key_prepare_tmp_bytes(&source)).borrow(),
            );
            let mut prepared = check!(
                gglwe_to_ggsw_key_prepared_alloc,
                gglwe_to_ggsw_key_prepared_alloc_from_infos,
                bytes_of_gglwe_to_ggsw,
                bytes_of_gglwe_to_ggsw_from_infos,
                gadget,
                [b, dn, ds, aux, rank],
                p,
                p.keys.iter().map(|key| BE::len_bytes(key.data.data())).sum::<usize>()
            );
            let source = m.gglwe_to_ggsw_key_alloc_from_infos(&gadget);
            m.gglwe_to_ggsw_key_prepare(
                &mut prepared,
                &source,
                &mut poisoned_scratch::<BE>(m.gglwe_to_ggsw_key_prepare_tmp_bytes(&source)).borrow(),
            );
        }
    }
}
/// Exercises each prepared factory's direct/from-info allocation and size
/// queries, then prepares with an independently budgeted poisoned arena.
pub fn test_preparation_contract<BR: ParityBackend, BT: ParityBackend>(
    params: &TestParams,
    shapes: &ParityShapes,
    r: &Module<BR>,
    t: &Module<BT>,
) where
    Module<BR>: PreparationBounds<BR>,
    Module<BT>: PreparationBounds<BT>,
    ScratchOwned<BR>: ScratchOwnedAlloc<BR> + ScratchOwnedBorrow<BR>,
    ScratchOwned<BT>: ScratchOwnedAlloc<BT> + ScratchOwnedBorrow<BT>,
{
    exercise(params, shapes, r);
    exercise(params, shapes, t);
}
