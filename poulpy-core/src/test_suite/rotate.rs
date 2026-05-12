use poulpy_hal::{
    api::{ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Module, ScratchOwned, ZnxView, ZnxViewMut},
    test_suite::TestParams,
};

use crate::{
    GLWERotate, ScratchArenaTakeCore,
    layouts::{GLWE, GLWEInfos, GLWELayout},
};

fn negacyclic_rotate(src: &[i64], k: i64) -> Vec<i64> {
    let n = src.len() as i64;
    let period = 2 * n;
    let k = k.rem_euclid(period);
    let mut dst = vec![0i64; src.len()];

    for (i, &value) in src.iter().enumerate() {
        let t = (i as i64 + k).rem_euclid(period);
        if t < n {
            dst[t as usize] = value;
        } else {
            dst[(t - n) as usize] = -value;
        }
    }

    dst
}

pub fn test_glwe_rotate<BE: crate::test_suite::TestBackend>(params: &TestParams, module: &Module<BE>)
where
    BE::OwnedBuf: poulpy_hal::layouts::HostDataMut,
    for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: GLWERotate<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    for<'a> poulpy_hal::layouts::ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
{
    let layout = GLWELayout {
        n: module.n().into(),
        base2k: params.base2k.into(),
        k: params.base2k.into(),
        rank: 2u32.into(),
    };

    let n = module.n();
    let cols = layout.rank().as_usize() + 1;
    let shifts = [-((n as i64) + 3), -5, -1, 0, 1, 7, n as i64 - 1, n as i64 + 2];

    for &shift in &shifts {
        let mut src = GLWE::<Vec<u8>>::alloc_from_infos(&layout);
        let mut out = GLWE::<Vec<u8>>::alloc_from_infos(&layout);
        let mut inplace = GLWE::<Vec<u8>>::alloc_from_infos(&layout);

        for col in 0..cols {
            let poly = src.data.at_mut(col, 0);
            for (j, coeff) in poly.iter_mut().enumerate() {
                *coeff = ((col as i64 + 1) * 1000) + j as i64 - 17;
            }
        }
        inplace.data.raw_mut().copy_from_slice(src.data.raw());

        module.glwe_rotate(shift, &mut out, &src);

        let mut scratch = ScratchOwned::<BE>::alloc(cols * module.glwe_rotate_tmp_bytes());
        module.glwe_rotate_assign(shift, &mut inplace, &mut scratch.borrow());

        for col in 0..cols {
            let expected = negacyclic_rotate(src.data.at(col, 0), shift);
            assert_eq!(
                out.data.at(col, 0),
                expected.as_slice(),
                "out-of-place mismatch for shift {shift}, col {col}"
            );
            assert_eq!(
                inplace.data.at(col, 0),
                expected.as_slice(),
                "inplace mismatch for shift {shift}, col {col}"
            );
        }
    }
}
