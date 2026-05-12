//! Reference implementations of the [`GGLWEAutomorphismDefault`] methods.
//!
//! Each free function carries the HAL bounds it actually needs in its own `where` clause.
//!
//! Re-exported publicly through `crate::oep::gglwe_automorphism_defaults`.

use poulpy_hal::{
    api::{ModuleN, VecZnxAutomorphismAssign, VecZnxAutomorphismAssignTmpBytes, VecZnxAutomorphismBackend},
    layouts::{Backend, CyclotomicOrder, GaloisElement, ScratchArena},
};

use crate::{
    ScratchArenaTakeCore,
    layouts::{
        GGLWEInfos, GGLWEToBackendMut, GGLWEToBackendRef, GLWE, GLWEViewRef, GetGaloisElement, SetGaloisElement,
        glwe_backend_ref_from_mut, prepared::GGLWEPreparedToBackendRef,
    },
    oep::{GGLWEAutomorphismDefault, GLWEKeyswitchDefault},
};

pub fn glwe_automorphism_key_automorphism_tmp_bytes_default<BE, M, R, A, K>(
    module: &M,
    res_infos: &R,
    a_infos: &A,
    key_infos: &K,
) -> usize
where
    BE: Backend,
    M: ModuleN + GLWEKeyswitchDefault<BE> + VecZnxAutomorphismAssignTmpBytes,
    R: GGLWEInfos,
    A: GGLWEInfos,
    K: GGLWEInfos,
    for<'s> ScratchArena<'s, BE>: ScratchArenaTakeCore<'s, BE>,
{
    assert_eq!(module.n() as u32, res_infos.n());
    assert_eq!(module.n() as u32, a_infos.n());
    assert_eq!(module.n() as u32, key_infos.n());

    let lvl_0: usize = module.glwe_keyswitch_tmp_bytes(res_infos, a_infos, key_infos);
    let lvl_1: usize = module.vec_znx_automorphism_assign_tmp_bytes();

    if res_infos.glwe_layout() == a_infos.glwe_layout() {
        lvl_0.max(lvl_1)
    } else {
        GLWE::<Vec<u8>>::bytes_of_from_infos(a_infos) + lvl_0.max(lvl_1)
    }
}

pub fn glwe_automorphism_key_automorphism_default<'s, BE, M, R, A, K>(
    module: &M,
    res: &mut R,
    a: &A,
    key: &K,
    key_size: usize,
    scratch: &mut ScratchArena<'s, BE>,
) where
    BE: Backend + 's,
    M: GGLWEAutomorphismDefault<BE>
        + GaloisElement
        + GLWEKeyswitchDefault<BE>
        + VecZnxAutomorphismBackend<BE>
        + VecZnxAutomorphismAssign<BE>
        + VecZnxAutomorphismAssignTmpBytes
        + CyclotomicOrder,
    R: GGLWEToBackendMut<BE> + SetGaloisElement + GGLWEInfos,
    A: GGLWEToBackendRef<BE> + GetGaloisElement + GGLWEInfos,
    K: GGLWEPreparedToBackendRef<BE> + GetGaloisElement + GGLWEInfos,
    for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
{
    assert!(
        res.dnum().as_u32() <= a.dnum().as_u32(),
        "res dnum: {} > a dnum: {}",
        res.dnum(),
        a.dnum()
    );

    assert_eq!(res.dsize(), a.dsize(), "res dnum: {} != a dnum: {}", res.dsize(), a.dsize());
    assert_eq!(res.base2k(), a.base2k());
    assert!(
        scratch.available() >= module.glwe_automorphism_key_automorphism_tmp_bytes(res, a, key),
        "scratch.available(): {} < GLWEAutomorphismKeyAutomorphism::glwe_automorphism_key_automorphism_tmp_bytes: {}",
        scratch.available(),
        module.glwe_automorphism_key_automorphism_tmp_bytes(res, a, key)
    );

    let cols_out: usize = (key.rank_out() + 1).into();
    let cols_in: usize = key.rank_in().into();
    let p: i64 = a.p();
    let p_inv: i64 = module.galois_element_inv(p);
    let same_layout: bool = res.glwe_layout() == a.glwe_layout();

    {
        let res = &mut res.to_backend_mut();
        let a_backend = <A as GGLWEToBackendRef<BE>>::to_backend_ref(a);

        for row in 0..res.dnum().as_usize() {
            for col in 0..cols_in {
                let mut res_tmp = res.at_view_mut(row, col);
                let a_ct_backend: GLWEViewRef<'_, BE> = a_backend.at_view(row, col);

                if same_layout {
                    for i in 0..cols_out {
                        module.vec_znx_automorphism_backend(p, &mut res_tmp.data, i, &a_ct_backend.data, i);
                    }

                    let mut scratch_iter = scratch.borrow();
                    module.glwe_keyswitch_assign(&mut res_tmp, key, key_size, &mut scratch_iter);

                    for i in 0..cols_out {
                        module.vec_znx_automorphism_assign(p_inv, &mut res_tmp.data, i, &mut scratch_iter);
                    }
                } else {
                    let (mut tmp_glwe, mut scratch_iter) = scratch.borrow().take_glwe_scratch(&a_ct_backend);

                    for i in 0..cols_out {
                        module.vec_znx_automorphism_backend(p, &mut tmp_glwe.data, i, &a_ct_backend.data, i);
                    }

                    let tmp_glwe_ref = glwe_backend_ref_from_mut::<BE>(&tmp_glwe);
                    let tmp_glwe_view = &tmp_glwe_ref;
                    module.glwe_keyswitch(&mut res_tmp, &tmp_glwe_view, key, key_size, &mut scratch_iter);

                    for i in 0..cols_out {
                        module.vec_znx_automorphism_assign(p_inv, &mut res_tmp.data, i, &mut scratch_iter);
                    }
                }
            }
        }
    }

    res.set_p((p * key.p()) % module.cyclotomic_order());
}

pub fn glwe_automorphism_key_automorphism_assign_default<'s, BE, M, R, K>(
    module: &M,
    res: &mut R,
    key: &K,
    key_size: usize,
    scratch: &mut ScratchArena<'s, BE>,
) where
    BE: Backend + 's,
    M: GaloisElement + GLWEKeyswitchDefault<BE> + VecZnxAutomorphismAssign<BE> + CyclotomicOrder,
    R: GGLWEToBackendMut<BE> + SetGaloisElement + GetGaloisElement + GGLWEInfos,
    K: GGLWEPreparedToBackendRef<BE> + GetGaloisElement + GGLWEInfos,
    for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
{
    assert_eq!(res.rank(), key.rank(), "key rank: {} != key rank: {}", res.rank(), key.rank());

    let cols_out: usize = (key.rank_out() + 1).into();
    let cols_in: usize = key.rank_in().into();
    let p: i64 = res.p();
    let p_inv: i64 = module.galois_element_inv(p);

    {
        let res = &mut res.to_backend_mut();
        for row in 0..res.dnum().as_usize() {
            for col in 0..cols_in {
                let mut res_tmp = res.at_view_mut(row, col);

                let mut scratch_iter = scratch.borrow();
                for i in 0..cols_out {
                    module.vec_znx_automorphism_assign(p, &mut res_tmp.data, i, &mut scratch_iter);
                }

                module.glwe_keyswitch_assign(&mut res_tmp, key, key_size, &mut scratch_iter);

                for i in 0..cols_out {
                    module.vec_znx_automorphism_assign(p_inv, &mut res_tmp.data, i, &mut scratch_iter);
                }
            }
        }
    }

    res.set_p((res.p() * key.p()) % module.cyclotomic_order());
}
