//! Reference implementations of the [`LWEKeyswitchDefault`] methods.
//!
//! Re-exported publicly through `crate::oep::lwe_keyswitch_defaults`.

use crate::api::GLWEBytesOf;
use poulpy_hal::{
    api::{ModuleN, VecZnxCopy, VecZnxZero},
    layouts::{Backend, ScratchArena, vec_znx_backend_mut_from_mut, vec_znx_backend_ref_from_ref},
};

use crate::{
    ScratchArenaTakeCore,
    layouts::{
        GGLWEInfos, GLWELayout, LWEInfos, LWEToBackendMut, LWEToBackendRef, Rank, TorusPrecision, glwe_backend_ref_from_mut,
        prepared::{GGLWEPreparedBackendRef, GGLWEPreparedToBackendRef},
    },
    oep::{GLWEKeyswitchDefault, LWEKeyswitchDefault},
};

pub fn lwe_keyswitch_tmp_bytes_default<BE, M, R, A, K>(module: &M, res_infos: &R, a_infos: &A, key_infos: &K) -> usize
where
    BE: Backend,
    M: GLWEBytesOf<BE> + ModuleN + GLWEKeyswitchDefault<BE>,
    R: LWEInfos,
    A: LWEInfos,
    K: GGLWEInfos,
{
    assert_eq!(module.n() as u32, key_infos.n());

    let k: TorusPrecision = a_infos.k().max(res_infos.k());

    let glwe_a_infos: GLWELayout = GLWELayout {
        n: module.n().into(),
        base2k: a_infos.base2k(),
        k,
        rank: Rank(1),
    };

    let glwe_res_infos: GLWELayout = GLWELayout {
        n: module.n().into(),
        base2k: res_infos.base2k(),
        k,
        rank: Rank(1),
    };

    let lvl_0: usize = module.glwe_bytes_of_from_infos(&glwe_a_infos);
    let lvl_1: usize = module.glwe_bytes_of_from_infos(&glwe_res_infos);
    let lvl_2: usize = module.glwe_keyswitch_tmp_bytes_default(&glwe_res_infos, &glwe_a_infos, key_infos);

    lvl_0 + lvl_1 + lvl_2
}

pub fn lwe_keyswitch_default<BE, M, R, A>(
    module: &M,
    res: &mut R,
    a: &A,
    ksk: &GGLWEPreparedBackendRef<'_, BE>,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: Backend,
    M: GLWEBytesOf<BE> + LWEKeyswitchDefault<BE> + ModuleN + GLWEKeyswitchDefault<BE> + VecZnxCopy<BE> + VecZnxZero<BE>,
    R: LWEToBackendMut<BE> + LWEInfos,
    A: LWEToBackendRef<BE> + LWEInfos,
{
    assert!(res.n().as_usize() <= module.n());
    assert!(a.n().as_usize() <= module.n());
    assert_eq!(ksk.n(), module.n() as u32);
    assert!(
        scratch.available() >= module.lwe_keyswitch_tmp_bytes_default(res, a, ksk),
        "scratch.available(): {} < LWEKeyswitch::lwe_keyswitch_tmp_bytes: {}",
        scratch.available(),
        module.lwe_keyswitch_tmp_bytes_default(res, a, ksk)
    );

    let scratch = scratch.borrow();
    let a_backend = a.to_backend_ref();
    let (mut glwe_in, scratch_1) = scratch.take_glwe_scratch(&GLWELayout {
        n: ksk.n(),
        base2k: a.base2k(),
        k: a.k(),
        rank: Rank(1),
    });
    module.vec_znx_zero(&mut glwe_in.data, 0);
    module.vec_znx_zero(&mut glwe_in.data, 1);

    let n_lwe: usize = a.n().into();

    module.vec_znx_copy(
        &mut vec_znx_backend_mut_from_mut::<BE>(&mut glwe_in.data).window_coeffs(0, 1),
        0,
        &a_backend.body,
        0,
    );
    module.vec_znx_copy(
        &mut vec_znx_backend_mut_from_mut::<BE>(&mut glwe_in.data).window_coeffs(0, n_lwe),
        1,
        &a_backend.mask,
        0,
    );

    let (mut glwe_out, mut scratch_2) = scratch_1.take_glwe_scratch(&GLWELayout {
        n: ksk.n(),
        base2k: res.base2k(),
        k: res.k(),
        rank: Rank(1),
    });

    let glwe_in_ref = glwe_backend_ref_from_mut::<BE>(&glwe_in);
    let glwe_in_view = &glwe_in_ref;
    module.glwe_keyswitch_default(&mut glwe_out, &glwe_in_view, &ksk.to_backend_ref(), &mut scratch_2);

    let mut res_backend = res.to_backend_mut();
    let glwe_out_ref = glwe_backend_ref_from_mut::<BE>(&glwe_out);
    let n: usize = res_backend.n().into();

    module.vec_znx_copy(
        &mut res_backend.body,
        0,
        &vec_znx_backend_ref_from_ref::<BE>(&glwe_out_ref.data).window_coeffs(0, 1),
        0,
    );
    module.vec_znx_copy(
        &mut res_backend.mask,
        0,
        &vec_znx_backend_ref_from_ref::<BE>(&glwe_out_ref.data).window_coeffs(0, n),
        1,
    );
}
