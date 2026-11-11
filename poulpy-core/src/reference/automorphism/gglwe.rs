//! Portable algorithms expressed with HAL operations.
//! Inter-family core operations dispatch through the selected backend hooks.
use crate::api::GLWEKeyswitch;

use crate::api::GLWEBytesOf;
use poulpy_hal::{
    api::{ModuleN, VecZnxAutomorphism, VecZnxAutomorphismAssign, VecZnxAutomorphismAssignTmpBytes},
    layouts::{Backend, GaloisElement, Ring, ScratchArena},
};

use crate::layouts::operand_degree;
use crate::{
    ScratchArenaTakeGLWE,
    layouts::{
        GGLWEInfos, GGLWEToBackendMut, GGLWEToBackendRef, GLWEInfos, GLWEViewRef, GetGaloisElement, SetGaloisElement,
        glwe_backend_ref_from_mut,
        prepared::{GGLWEPreparedToBackendRef, GLWEAutomorphismKeyPreparedBackendRef},
    },
    reference::automorphism::GLWEAutomorphismKeyAutomorphismReference,
};

pub fn glwe_automorphism_key_automorphism_tmp_bytes_reference<BE, M, R, A, K, Rg: Ring>(
    module: &M,
    res_infos: &R,
    a_infos: &A,
    key_infos: &K,
) -> usize
where
    BE: Backend,
    M: GLWEBytesOf<BE> + ModuleN + GLWEKeyswitch<BE, Rg> + VecZnxAutomorphismAssignTmpBytes,
    R: GGLWEInfos,
    A: GGLWEInfos,
    K: GGLWEInfos,
{
    operand_degree(module.n(), &[res_infos.n(), a_infos.n(), key_infos.n()]);

    let lvl_0: usize = module.glwe_keyswitch_tmp_bytes(res_infos, a_infos, key_infos);
    let lvl_1: usize = module.vec_znx_automorphism_assign_tmp_bytes();

    if res_infos.glwe_layout() == a_infos.glwe_layout() {
        lvl_0.max(lvl_1)
    } else {
        module.glwe_bytes_of_from_infos(a_infos) + lvl_0.max(lvl_1)
    }
}

pub fn glwe_automorphism_key_automorphism_reference<BE, M, R, A, Rg: Ring>(
    module: &M,
    res: &mut R,
    a: &A,
    key: &GLWEAutomorphismKeyPreparedBackendRef<'_, BE, Rg>,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: Backend,
    M: GLWEBytesOf<BE>
        + GLWEAutomorphismKeyAutomorphismReference<BE, Rg>
        + GaloisElement
        + GLWEKeyswitch<BE, Rg>
        + VecZnxAutomorphism<BE, Rg>
        + VecZnxAutomorphismAssign<BE, Rg>
        + VecZnxAutomorphismAssignTmpBytes,
    R: GGLWEToBackendMut<BE, Rg> + SetGaloisElement + GGLWEInfos,
    A: GGLWEToBackendRef<BE, Rg> + GetGaloisElement + GGLWEInfos,
{
    assert!(
        res.dnum().as_u32() <= a.dnum().as_u32(),
        "res dnum: {} > a dnum: {}",
        res.dnum(),
        a.dnum()
    );

    assert_eq!(
        res.k_digit(),
        a.k_digit(),
        "res k_digit: {} != a k_digit: {}",
        res.k_digit(),
        a.k_digit()
    );
    assert_eq!(
        res.k_offset(),
        a.k_offset(),
        "res k_offset: {} != a k_offset: {}",
        res.k_offset(),
        a.k_offset()
    );
    assert_eq!(res.base2k(), a.base2k());
    assert!(
        scratch.available() >= module.glwe_automorphism_key_automorphism_tmp_bytes_reference(res, a, key),
        "scratch.available(): {} < GLWEAutomorphismKeyAutomorphism::glwe_automorphism_key_automorphism_tmp_bytes: {}",
        scratch.available(),
        module.glwe_automorphism_key_automorphism_tmp_bytes_reference(res, a, key)
    );

    let cols_out: usize = (key.rank_out() + 1).into();
    let cols_in: usize = key.rank_in().into();
    let p: i64 = a.p();
    let p_inv: i64 = module.galois_element_inv::<Rg>(p);
    let same_layout: bool = res.glwe_layout() == a.glwe_layout();

    {
        let res = &mut res.to_backend_mut();
        let a_backend = <A as GGLWEToBackendRef<BE, Rg>>::to_backend_ref(a);

        for row in 0..res.dnum().as_usize() {
            for col in 0..cols_in {
                let mut res_tmp = res.at_view_mut(row, col);
                let a_ct_backend: GLWEViewRef<'_, BE, Rg> = a_backend.at_view(row, col);

                if same_layout {
                    for i in 0..cols_out {
                        module.vec_znx_automorphism(p, &mut res_tmp.data, i, &a_ct_backend.data, i);
                    }

                    let mut scratch_iter = scratch.borrow();
                    module.glwe_keyswitch_assign(&mut res_tmp, &key.to_backend_ref(), &mut scratch_iter);

                    for i in 0..cols_out {
                        module.vec_znx_automorphism_assign(p_inv, &mut res_tmp.data, i, &mut scratch_iter);
                    }
                } else {
                    let (mut tmp_glwe, mut scratch_iter) = scratch.borrow().take_glwe_scratch(&a_ct_backend);

                    for i in 0..cols_out {
                        module.vec_znx_automorphism(p, &mut tmp_glwe.data, i, &a_ct_backend.data, i);
                    }
                    tmp_glwe.set_canonical(a_ct_backend.is_canonical());

                    let tmp_glwe_ref = glwe_backend_ref_from_mut::<BE, Rg>(&tmp_glwe);
                    let tmp_glwe_view = &tmp_glwe_ref;
                    module.glwe_keyswitch(&mut res_tmp, &tmp_glwe_view, &key.to_backend_ref(), &mut scratch_iter);

                    for i in 0..cols_out {
                        module.vec_znx_automorphism_assign(p_inv, &mut res_tmp.data, i, &mut scratch_iter);
                    }
                }
            }
        }
    }

    res.set_p((p * key.p()) % (Rg::CYCLOTOMIC_ORDER_FACTOR * res.n().as_usize() as i64));
}

pub fn glwe_automorphism_key_automorphism_assign_reference<BE, M, R, Rg: Ring>(
    module: &M,
    res: &mut R,
    key: &GLWEAutomorphismKeyPreparedBackendRef<'_, BE, Rg>,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: Backend,
    M: GaloisElement + GLWEKeyswitch<BE, Rg> + VecZnxAutomorphismAssign<BE, Rg>,
    R: GGLWEToBackendMut<BE, Rg> + SetGaloisElement + GetGaloisElement + GGLWEInfos,
{
    assert_eq!(res.rank(), key.rank(), "key rank: {} != key rank: {}", res.rank(), key.rank());

    let cols_out: usize = (key.rank_out() + 1).into();
    let cols_in: usize = key.rank_in().into();
    let p: i64 = res.p();
    let p_inv: i64 = module.galois_element_inv::<Rg>(p);

    {
        let res = &mut res.to_backend_mut();
        for row in 0..res.dnum().as_usize() {
            for col in 0..cols_in {
                let mut res_tmp = res.at_view_mut(row, col);

                let mut scratch_iter = scratch.borrow();
                for i in 0..cols_out {
                    module.vec_znx_automorphism_assign(p, &mut res_tmp.data, i, &mut scratch_iter);
                }

                module.glwe_keyswitch_assign(&mut res_tmp, &key.to_backend_ref(), &mut scratch_iter);

                for i in 0..cols_out {
                    module.vec_znx_automorphism_assign(p_inv, &mut res_tmp.data, i, &mut scratch_iter);
                }
            }
        }
    }

    res.set_p((res.p() * key.p()) % (Rg::CYCLOTOMIC_ORDER_FACTOR * res.n().as_usize() as i64));
}
