use poulpy_cpu_portable::kernels::vec_znx::vec_znx_fill_uniform_portable;
use poulpy_hal::{
    AlignedBuf,
    layouts::{HostBytesBackend, Module, Ring, ScratchArena, Standard, VecZnx, VecZnxBackendMut, VecZnxBackendRef},
    oep::{HalVecZnxCIImpl, HalVecZnxImpl, HalVecZnxMonomialImpl},
    source::Source,
};

use super::operand::{automorphism_map, column_mut, column_ref, empty_like, permute_assign};
use crate::{
    NTT4x30Gpu,
    runtime::{GpuMut, GpuRuntime},
    znx::{Map, Term, ci_embed, ci_trace, gadget_decompose, lincomb, lincomb_decimating, normalize, permute},
};

unsafe impl<RT: GpuRuntime, R: Ring> HalVecZnxImpl for NTT4x30Gpu<RT, R> {
    fn vec_znx_zero(_module: &Module<Self>, res: &mut VecZnxBackendMut<'_, Self>, res_col: usize) {
        let Some(res) = column_mut(res, res_col) else { return };
        lincomb::<RT>(&res, res.size, Term::zero(&res), Term::zero(&res));
    }

    fn vec_znx_normalize_tmp_bytes(_module: &Module<Self>) -> usize {
        0
    }

    fn vec_znx_normalize(
        _module: &Module<Self>,
        res: &mut VecZnxBackendMut<'_, Self>,
        res_base2k: usize,
        res_k: usize,
        res_offset: i64,
        res_col: usize,
        a: &VecZnxBackendRef<'_, Self>,
        a_base2k: usize,
        a_col: usize,
        _scratch: &mut ScratchArena<'_, Self>,
    ) {
        let Some(res) = column_mut(res, res_col) else { return };
        let a = column_ref(a, a_col).unwrap_or_else(|| empty_like(&res));
        normalize::<RT>(&res, res_base2k, res_k, res_offset, &a, a_base2k);
    }

    fn vec_znx_gadget_decompose_tmp_bytes(_module: &Module<Self>) -> usize {
        0
    }

    fn vec_znx_gadget_decompose(
        _module: &Module<Self>,
        res: &mut VecZnxBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, Self>,
        a_col: usize,
        base2k: usize,
        k_digit: usize,
        _scratch: &mut ScratchArena<'_, Self>,
    ) {
        let Some(res) = column_mut(res, res_col) else { return };
        let a = column_ref(a, a_col).unwrap_or_else(|| empty_like(&res));
        gadget_decompose::<RT>(&res, &a, base2k, k_digit);
    }

    fn vec_znx_normalize_assign(
        _module: &Module<Self>,
        base2k: usize,
        k: usize,
        a_offset: i64,
        a: &mut VecZnxBackendMut<'_, Self>,
        a_col: usize,
        _scratch: &mut ScratchArena<'_, Self>,
    ) {
        assert!(a_offset <= 0, "vec_znx_normalize_assign: a_offset must be <= 0");
        let Some(a) = column_mut(a, a_col) else { return };
        normalize::<RT>(&a, base2k, k, a_offset, &a, base2k);
    }

    fn vec_znx_add(
        _module: &Module<Self>,
        res: &mut VecZnxBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, Self>,
        a_col: usize,
        b: &VecZnxBackendRef<'_, Self>,
        b_col: usize,
    ) {
        let Some(res) = column_mut(res, res_col) else { return };
        let a = column_ref(a, a_col).unwrap_or_else(|| empty_like(&res));
        let b = column_ref(b, b_col).unwrap_or_else(|| empty_like(&res));
        lincomb::<RT>(&res, res.size, Term::plus(&a), Term::plus(&b));
    }

    fn vec_znx_add_assign(
        _module: &Module<Self>,
        res: &mut VecZnxBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, Self>,
        a_col: usize,
    ) {
        let Some(res) = column_mut(res, res_col) else { return };
        let Some(a) = column_ref(a, a_col) else { return };
        lincomb::<RT>(&res, res.size.min(a.size), Term::plus(&res), Term::plus(&a));
    }

    fn vec_znx_sub(
        _module: &Module<Self>,
        res: &mut VecZnxBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, Self>,
        a_col: usize,
        b: &VecZnxBackendRef<'_, Self>,
        b_col: usize,
    ) {
        let Some(res) = column_mut(res, res_col) else { return };
        let a = column_ref(a, a_col).unwrap_or_else(|| empty_like(&res));
        let b = column_ref(b, b_col).unwrap_or_else(|| empty_like(&res));
        lincomb::<RT>(&res, res.size, Term::plus(&a), Term::minus(&b));
    }

    fn vec_znx_sub_assign(
        _module: &Module<Self>,
        res: &mut VecZnxBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, Self>,
        a_col: usize,
    ) {
        let Some(res) = column_mut(res, res_col) else { return };
        let Some(a) = column_ref(a, a_col) else { return };
        lincomb::<RT>(&res, res.size.min(a.size), Term::plus(&res), Term::minus(&a));
    }

    fn vec_znx_sub_negate_assign(
        _module: &Module<Self>,
        res: &mut VecZnxBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, Self>,
        a_col: usize,
    ) {
        let Some(res) = column_mut(res, res_col) else { return };
        let a = column_ref(a, a_col).unwrap_or_else(|| empty_like(&res));
        lincomb::<RT>(&res, res.size, Term::plus(&a), Term::minus(&res));
    }

    fn vec_znx_negate(
        _module: &Module<Self>,
        res: &mut VecZnxBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, Self>,
        a_col: usize,
    ) {
        let Some(res) = column_mut(res, res_col) else { return };
        let a = column_ref(a, a_col).unwrap_or_else(|| empty_like(&res));
        assert_eq!(a.n, res.n);
        lincomb::<RT>(&res, res.size, Term::minus(&a), Term::zero(&a));
    }

    fn vec_znx_negate_assign(_module: &Module<Self>, a: &mut VecZnxBackendMut<'_, Self>, a_col: usize) {
        let Some(a) = column_mut(a, a_col) else { return };
        lincomb::<RT>(&a, a.size, Term::minus(&a), Term::zero(&a));
    }

    fn vec_znx_automorphism(
        _module: &Module<Self>,
        k: i64,
        res: &mut VecZnxBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, Self>,
        a_col: usize,
    ) {
        poulpy_hal::layouts::assert_dense(res, "vec_znx_automorphism");
        poulpy_hal::layouts::assert_dense(a, "vec_znx_automorphism");
        assert_eq!(a.n(), res.n());
        let Some(res) = column_mut(res, res_col) else { return };
        let a = column_ref(a, a_col).unwrap_or_else(|| empty_like(&res));
        permute::<RT>(&res, &a, automorphism_map::<R>(k));
    }

    fn vec_znx_automorphism_assign_tmp_bytes(_module: &Module<Self>) -> usize {
        0
    }

    fn vec_znx_automorphism_assign(
        module: &Module<Self>,
        k: i64,
        res: &mut VecZnxBackendMut<'_, Self>,
        res_col: usize,
        _scratch: &mut ScratchArena<'_, Self>,
    ) {
        poulpy_hal::layouts::assert_dense(res, "vec_znx_automorphism_assign");
        let Some(res) = column_mut(res, res_col) else { return };
        permute_assign(module, &res, automorphism_map::<R>(k));
    }

    fn vec_znx_switch_ring(
        _module: &Module<Self>,
        res: &mut VecZnxBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, Self>,
        a_col: usize,
    ) {
        poulpy_hal::layouts::assert_dense(res, "vec_znx_switch_ring");
        poulpy_hal::layouts::assert_dense(a, "vec_znx_switch_ring");
        let Some(res) = column_mut(res, res_col) else { return };
        let a = column_ref(a, a_col).unwrap_or_else(|| empty_like(&res));
        lincomb_decimating::<RT>(&res, res.size, Term::plus(&a), Term::zero(&res));
    }

    fn vec_znx_copy(
        _module: &Module<Self>,
        res: &mut VecZnxBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, Self>,
        a_col: usize,
    ) {
        assert_eq!(res.n(), a.n());
        let Some(res) = column_mut(res, res_col) else { return };
        let a = column_ref(a, a_col).unwrap_or_else(|| empty_like(&res));
        lincomb::<RT>(&res, res.size, Term::plus(&a), Term::zero(&a));
    }

    fn vec_znx_fill_uniform(
        _module: &Module<Self>,
        base2k: usize,
        k: usize,
        res: &mut VecZnxBackendMut<'_, Self>,
        res_col: usize,
        seed: [u8; 32],
    ) {
        // The stream is part of the contract, so the draws come from the
        // portable sampler and reach the device limb by limb.
        assert!(res_col < res.cols(), "vec_znx_fill_uniform: column {res_col} out of range");
        let (n, size) = (res.n(), res.size());
        let mut host = VecZnx::<AlignedBuf, i64>::from_data(AlignedBuf::from(vec![0u8; n * size * size_of::<i64>()]), n, 1, size);
        let mut source = Source::new(seed);
        {
            let mut host_mut: VecZnxBackendMut<'_, HostBytesBackend> = VecZnx::from_data(&mut host.data_mut()[..], n, 1, size);
            vec_znx_fill_uniform_portable::<HostBytesBackend>(base2k, k, &mut host_mut, 0, &mut source);
        }
        upload_column::<RT>(res, res_col, host.data());
    }
}

/// Writes the `size * n` coefficients of `host`, limb after limb, into column `col` of `res`.
fn upload_column<RT: GpuRuntime>(res: &mut VecZnx<GpuMut<'_, RT>, i64>, col: usize, host: &[u8]) {
    let shape = res.shape();
    let limb_bytes = shape.n() * size_of::<i64>();
    for limb in 0..shape.size() {
        let offset = shape.scalar_offset(col, limb) * size_of::<i64>();
        res.data_mut()
            .region_mut(offset, limb_bytes)
            .write_prefix(&host[limb * limb_bytes..(limb + 1) * limb_bytes]);
    }
}

unsafe impl<RT: GpuRuntime> HalVecZnxMonomialImpl for NTT4x30Gpu<RT, Standard> {
    fn vec_znx_rotate(
        _module: &Module<Self>,
        k: i64,
        res: &mut VecZnxBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, Self>,
        a_col: usize,
    ) {
        poulpy_hal::layouts::assert_dense(res, "vec_znx_rotate");
        poulpy_hal::layouts::assert_dense(a, "vec_znx_rotate");
        assert_eq!(res.n(), a.n());
        let Some(res) = column_mut(res, res_col) else { return };
        let a = column_ref(a, a_col).unwrap_or_else(|| empty_like(&res));
        permute::<RT>(&res, &a, Map::Rotation { p: k });
    }

    fn vec_znx_rotate_assign_tmp_bytes(_module: &Module<Self>) -> usize {
        0
    }

    fn vec_znx_rotate_assign(
        module: &Module<Self>,
        k: i64,
        a: &mut VecZnxBackendMut<'_, Self>,
        a_col: usize,
        _scratch: &mut ScratchArena<'_, Self>,
    ) {
        poulpy_hal::layouts::assert_dense(a, "vec_znx_rotate_assign");
        let Some(a) = column_mut(a, a_col) else { return };
        permute_assign(module, &a, Map::Rotation { p: k });
    }
}

unsafe impl<RT: GpuRuntime> HalVecZnxCIImpl for NTT4x30Gpu<RT, Standard> {
    fn vec_znx_ci_embed(
        _module: &Module<Self>,
        res: &mut VecZnxBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, Self>,
        a_col: usize,
    ) {
        poulpy_hal::layouts::assert_dense(res, "vec_znx_ci_embed");
        poulpy_hal::layouts::assert_dense(a, "vec_znx_ci_embed");
        let Some(res) = column_mut(res, res_col) else { return };
        let Some(a) = column_ref(a, a_col) else {
            lincomb::<RT>(&res, res.size, Term::zero(&res), Term::zero(&res));
            return;
        };
        ci_embed::<RT>(&res, &a);
    }

    fn vec_znx_ci_trace(
        _module: &Module<Self>,
        res: &mut VecZnxBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, Self>,
        a_col: usize,
    ) {
        poulpy_hal::layouts::assert_dense(res, "vec_znx_ci_trace");
        poulpy_hal::layouts::assert_dense(a, "vec_znx_ci_trace");
        let Some(res) = column_mut(res, res_col) else { return };
        let Some(a) = column_ref(a, a_col) else {
            lincomb::<RT>(&res, res.size, Term::zero(&res), Term::zero(&res));
            return;
        };
        ci_trace::<RT>(&res, &a);
    }
}
