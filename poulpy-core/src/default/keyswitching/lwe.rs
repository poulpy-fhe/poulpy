//! Reference implementations of the [`LWEKeyswitchDefault`] methods.
//!
//! Re-exported publicly through `crate::oep::lwe_keyswitch_defaults`.

#![allow(private_bounds)]

use poulpy_hal::{
    api::{ModuleN, VecZnxCopyRangeBackend, VecZnxZeroBackend},
    layouts::{Backend, ScratchArena},
};

use crate::{
    ScratchArenaTakeCore,
    layouts::{
        GGLWEInfos, GLWE, GLWELayout, LWEInfos, LWEToBackendMut, LWEToBackendRef, Rank, TorusPrecision,
        glwe_backend_ref_from_mut, prepared::GGLWEPreparedToBackendRef,
    },
    oep::{GLWEKeyswitchDefault, LWEKeyswitchDefault},
};

pub fn lwe_keyswitch_tmp_bytes_default<BE, M, R, A, K>(module: &M, res_infos: &R, a_infos: &A, key_infos: &K) -> usize
where
    BE: Backend,
    M: ModuleN + GLWEKeyswitchDefault<BE>,
    R: LWEInfos,
    A: LWEInfos,
    K: GGLWEInfos,
    for<'s> ScratchArena<'s, BE>: ScratchArenaTakeCore<'s, BE>,
{
    assert_eq!(module.n() as u32, key_infos.n());

    let max_k: TorusPrecision = a_infos.max_k().max(res_infos.max_k());

    let glwe_a_infos: GLWELayout = GLWELayout {
        n: module.n().into(),
        base2k: a_infos.base2k(),
        k: max_k,
        rank: Rank(1),
    };

    let glwe_res_infos: GLWELayout = GLWELayout {
        n: module.n().into(),
        base2k: res_infos.base2k(),
        k: max_k,
        rank: Rank(1),
    };

    let lvl_0: usize = GLWE::<Vec<u8>>::bytes_of_from_infos(&glwe_a_infos);
    let lvl_1: usize = GLWE::<Vec<u8>>::bytes_of_from_infos(&glwe_res_infos);
    let lvl_2: usize = module.glwe_keyswitch_tmp_bytes(&glwe_res_infos, &glwe_a_infos, key_infos);

    lvl_0 + lvl_1 + lvl_2
}

pub fn lwe_keyswitch_default<'s, BE, M, R, A, K>(
    module: &M,
    res: &mut R,
    a: &A,
    ksk: &K,
    key_size: usize,
    scratch: &mut ScratchArena<'s, BE>,
) where
    BE: Backend,
    M: LWEKeyswitchDefault<BE> + ModuleN + GLWEKeyswitchDefault<BE> + VecZnxCopyRangeBackend<BE> + VecZnxZeroBackend<BE>,
    R: LWEToBackendMut<BE> + LWEInfos,
    A: LWEToBackendRef<BE> + LWEInfos,
    K: GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
    for<'x> ScratchArena<'x, BE>: ScratchArenaTakeCore<'x, BE>,
{
    assert!(res.n().as_usize() <= module.n());
    assert!(a.n().as_usize() <= module.n());
    assert_eq!(ksk.n(), module.n() as u32);
    assert!(
        scratch.available() >= module.lwe_keyswitch_tmp_bytes(res, a, ksk),
        "scratch.available(): {} < LWEKeySwitch::lwe_keyswitch_tmp_bytes: {}",
        scratch.available(),
        module.lwe_keyswitch_tmp_bytes(res, a, ksk)
    );

    let scratch = scratch.borrow();
    let a_backend = a.to_backend_ref();
    let (mut glwe_in, scratch_1) = scratch.take_glwe_scratch(&GLWELayout {
        n: ksk.n(),
        base2k: a.base2k(),
        k: a.max_k(),
        rank: Rank(1),
    });
    module.vec_znx_zero_backend(&mut glwe_in.data, 0);
    module.vec_znx_zero_backend(&mut glwe_in.data, 1);

    let n_lwe: usize = a.n().into();

    for i in 0..a.size() {
        module.vec_znx_copy_range_backend(&mut glwe_in.data, 0, i, 0, &a_backend.data, 0, i, 0, 1);
        module.vec_znx_copy_range_backend(&mut glwe_in.data, 1, i, 0, &a_backend.data, 0, i, 1, n_lwe);
    }

    let (mut glwe_out, mut scratch_2) = scratch_1.take_glwe_scratch(&GLWELayout {
        n: ksk.n(),
        base2k: res.base2k(),
        k: res.max_k(),
        rank: Rank(1),
    });

    let glwe_in_ref = glwe_backend_ref_from_mut::<BE>(&glwe_in);
    let glwe_in_view = &glwe_in_ref;
    module.glwe_keyswitch(&mut glwe_out, &glwe_in_view, ksk, key_size, &mut scratch_2);

    let mut res_backend = res.to_backend_mut();
    let glwe_out_ref = glwe_backend_ref_from_mut::<BE>(&glwe_out);
    let min_size: usize = res_backend.size().min(glwe_out_ref.size());
    let n: usize = res_backend.n().into();

    module.vec_znx_zero_backend(&mut res_backend.data, 0);
    for i in 0..min_size {
        module.vec_znx_copy_range_backend(&mut res_backend.data, 0, i, 0, &glwe_out_ref.data, 0, i, 0, 1);
        module.vec_znx_copy_range_backend(&mut res_backend.data, 0, i, 1, &glwe_out_ref.data, 1, i, 0, n);
    }
}
