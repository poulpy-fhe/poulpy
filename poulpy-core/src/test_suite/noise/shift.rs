//! GLWE shifts on a destination whose `k` sits below its allocation: a shift
//! past the operand width yields zero instead of wrapping in the `i64` cast,
//! and every path fits in exactly `glwe_shift_tmp_bytes`.

use poulpy_hal::{
    api::{ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{HostDataMut, Module, ScratchOwned, VecZnx, ZnxView, ZnxViewMut},
    test_suite::TestParams,
};

use crate::{
    GLWEShift,
    layouts::{GLWE, GLWELayout, ModuleCoreAlloc},
    test_suite::noise::glwe_tensor::assert_canonical,
};

/// Precision of the shifted ciphertexts: two limbs, three bits short of the
/// allocation, so `glwe_rsh` and `glwe_lsh_assign` take their partial-width paths.
fn partial_k(params: &TestParams) -> usize {
    2 * params.base2k - 3
}

fn partial_layout(params: &TestParams, n: usize) -> GLWELayout {
    GLWELayout {
        n: (n as u32).into(),
        base2k: params.base2k.into(),
        k: partial_k(params).into(),
        rank: 1u32.into(),
    }
}

/// Small, non-zero digits in every limb of every column.
fn fill(data: &mut VecZnx<impl HostDataMut, i64>) {
    for col in 0..data.cols() {
        for limb in 0..data.size() {
            for (j, coeff) in data.at_mut(col, limb).iter_mut().enumerate() {
                *coeff = ((col as i64 + 1) * 1000 + limb as i64 * 37 + j as i64) % 1021 - 510;
            }
        }
    }
}

pub fn test_glwe_shift_saturates<BE: crate::test_suite::noise::TestBackend>(params: &TestParams, module: &Module<BE>)
where
    BE::OwnedBuf: poulpy_hal::layouts::HostDataMut,
    for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: GLWEShift<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let layout = partial_layout(params, module.n());
    let mut src: GLWE<BE::OwnedBuf, BE::ZnxWord> = module.glwe_alloc_from_infos(&layout);
    fill(&mut src.data);
    let mut scratch = ScratchOwned::<BE>::alloc(module.glwe_shift_tmp_bytes(src.data.size()));

    for &k in &[usize::MAX, 1usize << 63, (1usize << 63) + 1] {
        let mut res: GLWE<BE::OwnedBuf, BE::ZnxWord> = module.glwe_alloc_from_infos(&layout);
        res.data.raw_mut().copy_from_slice(src.data.raw());
        module.glwe_rsh(k, &mut res, &mut scratch.borrow());
        assert!(res.data.raw().iter().all(|&d| d == 0), "glwe_rsh by {k} is not zero");

        let mut res: GLWE<BE::OwnedBuf, BE::ZnxWord> = module.glwe_alloc_from_infos(&layout);
        module.glwe_lsh(&mut res, &src, k, &mut scratch.borrow());
        assert!(res.data.raw().iter().all(|&d| d == 0), "glwe_lsh by {k} is not zero");

        let mut res: GLWE<BE::OwnedBuf, BE::ZnxWord> = module.glwe_alloc_from_infos(&layout);
        res.data.raw_mut().copy_from_slice(src.data.raw());
        module.glwe_lsh_assign(&mut res, k, &mut scratch.borrow());
        assert!(res.data.raw().iter().all(|&d| d == 0), "glwe_lsh_assign by {k} is not zero");
    }
}

pub fn test_glwe_shift_exact_scratch<BE: crate::test_suite::noise::TestBackend>(params: &TestParams, module: &Module<BE>)
where
    BE::OwnedBuf: poulpy_hal::layouts::HostDataMut,
    for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: GLWEShift<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let layout = partial_layout(params, module.n());
    let base2k = params.base2k;
    let k_res = partial_k(params);
    let mut src: GLWE<BE::OwnedBuf, BE::ZnxWord> = module.glwe_alloc_from_infos(&layout);
    fill(&mut src.data);
    // Exactly the advertised scratch: every path must fit, and the three
    // paths that normalize at `k` must leave the destination canonical there.
    let mut scratch = ScratchOwned::<BE>::alloc(module.glwe_shift_tmp_bytes(src.data.size()));

    for &k in &[0usize, 1, 3, base2k - 1, base2k, base2k + 2] {
        let mut res: GLWE<BE::OwnedBuf, BE::ZnxWord> = module.glwe_alloc_from_infos(&layout);
        res.data.raw_mut().copy_from_slice(src.data.raw());
        module.glwe_rsh(k, &mut res, &mut scratch.borrow());
        assert_canonical(&res.data, base2k, k_res);

        let mut res: GLWE<BE::OwnedBuf, BE::ZnxWord> = module.glwe_alloc_from_infos(&layout);
        module.glwe_lsh(&mut res, &src, k, &mut scratch.borrow());
        assert_canonical(&res.data, base2k, k_res);

        let mut res: GLWE<BE::OwnedBuf, BE::ZnxWord> = module.glwe_alloc_from_infos(&layout);
        res.data.raw_mut().copy_from_slice(src.data.raw());
        module.glwe_lsh_assign(&mut res, k, &mut scratch.borrow());
        assert_canonical(&res.data, base2k, k_res);

        let mut res: GLWE<BE::OwnedBuf, BE::ZnxWord> = module.glwe_alloc_from_infos(&layout);
        res.data.raw_mut().copy_from_slice(src.data.raw());
        module.glwe_lsh_add(&mut res, &src, k, &mut scratch.borrow());

        let mut res: GLWE<BE::OwnedBuf, BE::ZnxWord> = module.glwe_alloc_from_infos(&layout);
        res.data.raw_mut().copy_from_slice(src.data.raw());
        module.glwe_lsh_sub(&mut res, &src, k, &mut scratch.borrow());
    }
}
