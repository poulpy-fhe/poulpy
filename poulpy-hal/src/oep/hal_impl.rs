#![allow(clippy::too_many_arguments)]

use crate::{
    layouts::{
        Backend, Module, NoiseInfos, ScalarZnxToRef, Scratch, ScratchOwned, VecZnx, VecZnxToMut, VecZnxToRef, ZnxView, ZnxViewMut,
    },
    source::Source,
};

/// Backend-owned `poulpy-hal` extension point.
///
/// `HalImpl` carries both:
/// - the portable bucket (`scratch`, `vec_znx`)
/// - the transformed-domain family surface (`module`, `vec_znx_big`,
///   `vec_znx_dft`, `svp_ppol`, `vmp_pmat`, `convolution`)
///
/// Default implementations live in backend crates (for example `poulpy-cpu-ref`),
/// and backend `HalImpl` impls can delegate to those defaults or override selectively.
///
/// # Safety
/// Implementors must uphold all backend invariants for scratch allocation and
/// vector operations. In particular, methods must not violate aliasing or
/// alignment requirements, must honor input/output size expectations, and must
/// only write within the provided buffers.
pub unsafe trait HalImpl<BE: Backend>: Backend {
    // Scratch
    fn scratch_owned_alloc(size: usize) -> ScratchOwned<BE>;
    fn scratch_owned_borrow(scratch: &mut ScratchOwned<BE>) -> &mut Scratch<BE>;
    fn scratch_from_bytes(data: &mut [u8]) -> &mut Scratch<BE>;
    fn scratch_available(scratch: &Scratch<BE>) -> usize;
    fn take_slice<T>(scratch: &mut Scratch<BE>, len: usize) -> (&mut [T], &mut Scratch<BE>);

    // VecZnx
    fn vec_znx_zero<R>(module: &Module<BE>, res: &mut R, res_col: usize)
    where
        R: VecZnxToMut;

    fn vec_znx_normalize_tmp_bytes(module: &Module<BE>) -> usize;

    #[allow(clippy::too_many_arguments)]
    fn vec_znx_normalize<R, A>(
        module: &Module<BE>,
        res: &mut R,
        res_base2k: usize,
        res_offset: i64,
        res_col: usize,
        a: &A,
        a_base2k: usize,
        a_col: usize,
        scratch: &mut Scratch<BE>,
    ) where
        R: VecZnxToMut,
        A: VecZnxToRef;

    fn vec_znx_normalize_assign<A>(module: &Module<BE>, base2k: usize, a: &mut A, a_col: usize, scratch: &mut Scratch<BE>)
    where
        A: VecZnxToMut;

    fn vec_znx_add_into<R, A, C>(module: &Module<BE>, res: &mut R, res_col: usize, a: &A, a_col: usize, b: &C, b_col: usize)
    where
        R: VecZnxToMut,
        A: VecZnxToRef,
        C: VecZnxToRef;

    fn vec_znx_add_assign<R, A>(module: &Module<BE>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxToMut,
        A: VecZnxToRef;

    #[allow(clippy::too_many_arguments)]
    fn vec_znx_add_scalar_into<R, A, B>(
        module: &Module<BE>,
        res: &mut R,
        res_col: usize,
        a: &A,
        a_col: usize,
        b: &B,
        b_col: usize,
        b_limb: usize,
    ) where
        R: VecZnxToMut,
        A: ScalarZnxToRef,
        B: VecZnxToRef;

    fn vec_znx_add_scalar_assign<R, A>(module: &Module<BE>, res: &mut R, res_col: usize, res_limb: usize, a: &A, a_col: usize)
    where
        R: VecZnxToMut,
        A: ScalarZnxToRef;

    fn vec_znx_sub<R, A, C>(module: &Module<BE>, res: &mut R, res_col: usize, a: &A, a_col: usize, b: &C, b_col: usize)
    where
        R: VecZnxToMut,
        A: VecZnxToRef,
        C: VecZnxToRef;

    fn vec_znx_sub_assign<R, A>(module: &Module<BE>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxToMut,
        A: VecZnxToRef;

    fn vec_znx_sub_negate_assign<R, A>(module: &Module<BE>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxToMut,
        A: VecZnxToRef;

    #[allow(clippy::too_many_arguments)]
    fn vec_znx_sub_scalar<R, A, B>(
        module: &Module<BE>,
        res: &mut R,
        res_col: usize,
        a: &A,
        a_col: usize,
        b: &B,
        b_col: usize,
        b_limb: usize,
    ) where
        R: VecZnxToMut,
        A: ScalarZnxToRef,
        B: VecZnxToRef;

    fn vec_znx_sub_scalar_assign<R, A>(module: &Module<BE>, res: &mut R, res_col: usize, res_limb: usize, a: &A, a_col: usize)
    where
        R: VecZnxToMut,
        A: ScalarZnxToRef;

    fn vec_znx_negate<R, A>(module: &Module<BE>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxToMut,
        A: VecZnxToRef;

    fn vec_znx_negate_assign<A>(module: &Module<BE>, a: &mut A, a_col: usize)
    where
        A: VecZnxToMut;

    fn vec_znx_rsh_tmp_bytes(module: &Module<BE>) -> usize;

    fn vec_znx_rsh<R, A>(
        module: &Module<BE>,
        base2k: usize,
        k: usize,
        res: &mut R,
        res_col: usize,
        a: &A,
        a_col: usize,
        scratch: &mut Scratch<BE>,
    ) where
        R: VecZnxToMut,
        A: VecZnxToRef;

    fn vec_znx_rsh_add_into<R, A>(
        module: &Module<BE>,
        base2k: usize,
        k: usize,
        res: &mut R,
        res_col: usize,
        a: &A,
        a_col: usize,
        scratch: &mut Scratch<BE>,
    ) where
        R: VecZnxToMut,
        A: VecZnxToRef;

    fn vec_znx_lsh_tmp_bytes(module: &Module<BE>) -> usize;

    fn vec_znx_lsh<R, A>(
        module: &Module<BE>,
        base2k: usize,
        k: usize,
        res: &mut R,
        res_col: usize,
        a: &A,
        a_col: usize,
        scratch: &mut Scratch<BE>,
    ) where
        R: VecZnxToMut,
        A: VecZnxToRef;

    fn vec_znx_lsh_add_into<R, A>(
        module: &Module<BE>,
        base2k: usize,
        k: usize,
        res: &mut R,
        res_col: usize,
        a: &A,
        a_col: usize,
        scratch: &mut Scratch<BE>,
    ) where
        R: VecZnxToMut,
        A: VecZnxToRef;

    fn vec_znx_lsh_sub<R, A>(
        module: &Module<BE>,
        base2k: usize,
        k: usize,
        res: &mut R,
        res_col: usize,
        a: &A,
        a_col: usize,
        scratch: &mut Scratch<BE>,
    ) where
        R: VecZnxToMut,
        A: VecZnxToRef;

    fn vec_znx_rsh_sub<R, A>(
        module: &Module<BE>,
        base2k: usize,
        k: usize,
        res: &mut R,
        res_col: usize,
        a: &A,
        a_col: usize,
        scratch: &mut Scratch<BE>,
    ) where
        R: VecZnxToMut,
        A: VecZnxToRef;

    fn vec_znx_rsh_assign<R>(module: &Module<BE>, base2k: usize, k: usize, a: &mut R, a_col: usize, scratch: &mut Scratch<BE>)
    where
        R: VecZnxToMut;

    fn vec_znx_lsh_assign<R>(module: &Module<BE>, base2k: usize, k: usize, a: &mut R, a_col: usize, scratch: &mut Scratch<BE>)
    where
        R: VecZnxToMut;

    fn vec_znx_rotate<R, A>(module: &Module<BE>, k: i64, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxToMut,
        A: VecZnxToRef;

    fn vec_znx_rotate_assign_tmp_bytes(module: &Module<BE>) -> usize;

    fn vec_znx_rotate_assign<A>(module: &Module<BE>, k: i64, a: &mut A, a_col: usize, scratch: &mut Scratch<BE>)
    where
        A: VecZnxToMut;

    fn vec_znx_automorphism<R, A>(module: &Module<BE>, k: i64, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxToMut,
        A: VecZnxToRef;

    fn vec_znx_automorphism_assign_tmp_bytes(module: &Module<BE>) -> usize;

    fn vec_znx_automorphism_assign<R>(module: &Module<BE>, k: i64, res: &mut R, res_col: usize, scratch: &mut Scratch<BE>)
    where
        R: VecZnxToMut;

    fn vec_znx_mul_xp_minus_one<R, A>(module: &Module<BE>, k: i64, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxToMut,
        A: VecZnxToRef;

    fn vec_znx_mul_xp_minus_one_assign_tmp_bytes(module: &Module<BE>) -> usize;

    fn vec_znx_mul_xp_minus_one_assign<R>(module: &Module<BE>, k: i64, res: &mut R, res_col: usize, scratch: &mut Scratch<BE>)
    where
        R: VecZnxToMut;

    fn vec_znx_split_ring_tmp_bytes(module: &Module<BE>) -> usize;

    fn vec_znx_split_ring<R, A>(
        module: &Module<BE>,
        res: &mut [R],
        res_col: usize,
        a: &A,
        a_col: usize,
        scratch: &mut Scratch<BE>,
    ) where
        R: VecZnxToMut,
        A: VecZnxToRef;

    fn vec_znx_merge_rings_tmp_bytes(module: &Module<BE>) -> usize;

    fn vec_znx_merge_rings<R, A>(
        module: &Module<BE>,
        res: &mut R,
        res_col: usize,
        a: &[A],
        a_col: usize,
        scratch: &mut Scratch<BE>,
    ) where
        R: VecZnxToMut,
        A: VecZnxToRef;

    fn vec_znx_switch_ring<R, A>(module: &Module<BE>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxToMut,
        A: VecZnxToRef;

    fn vec_znx_copy<R, A>(module: &Module<BE>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxToMut,
        A: VecZnxToRef;

    fn vec_znx_fill_uniform<R>(module: &Module<BE>, base2k: usize, res: &mut R, res_col: usize, source: &mut Source)
    where
        R: VecZnxToMut;

    fn vec_znx_fill_normal<R>(
        module: &Module<BE>,
        res_base2k: usize,
        res: &mut R,
        res_col: usize,
        noise_infos: NoiseInfos,
        source: &mut Source,
    ) where
        R: VecZnxToMut;

    fn vec_znx_add_normal<R>(
        module: &Module<BE>,
        res_base2k: usize,
        res: &mut R,
        res_col: usize,
        noise_infos: NoiseInfos,
        source: &mut Source,
    ) where
        R: VecZnxToMut;

    // Module
    #[allow(clippy::new_ret_no_self)]
    fn new(n: u64) -> Module<BE>;

    // VecZnxBig
    fn vec_znx_big_from_small<R, A>(res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: crate::layouts::VecZnxBigToMut<BE>,
        A: VecZnxToRef;

    fn vec_znx_big_add_normal<R>(
        module: &Module<BE>,
        res_base2k: usize,
        res: &mut R,
        res_col: usize,
        noise_infos: NoiseInfos,
        source: &mut Source,
    ) where
        R: crate::layouts::VecZnxBigToMut<BE>;

    fn vec_znx_big_add_into<R, A, C>(module: &Module<BE>, res: &mut R, res_col: usize, a: &A, a_col: usize, b: &C, b_col: usize)
    where
        R: crate::layouts::VecZnxBigToMut<BE>,
        A: crate::layouts::VecZnxBigToRef<BE>,
        C: crate::layouts::VecZnxBigToRef<BE>;

    fn vec_znx_big_add_assign<R, A>(module: &Module<BE>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: crate::layouts::VecZnxBigToMut<BE>,
        A: crate::layouts::VecZnxBigToRef<BE>;

    fn vec_znx_big_add_small_into<R, A, C>(
        module: &Module<BE>,
        res: &mut R,
        res_col: usize,
        a: &A,
        a_col: usize,
        b: &C,
        b_col: usize,
    ) where
        R: crate::layouts::VecZnxBigToMut<BE>,
        A: crate::layouts::VecZnxBigToRef<BE>,
        C: VecZnxToRef;

    fn vec_znx_big_add_small_assign<R, A>(module: &Module<BE>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: crate::layouts::VecZnxBigToMut<BE>,
        A: VecZnxToRef;

    fn vec_znx_big_sub<R, A, C>(module: &Module<BE>, res: &mut R, res_col: usize, a: &A, a_col: usize, b: &C, b_col: usize)
    where
        R: crate::layouts::VecZnxBigToMut<BE>,
        A: crate::layouts::VecZnxBigToRef<BE>,
        C: crate::layouts::VecZnxBigToRef<BE>;

    fn vec_znx_big_sub_assign<R, A>(module: &Module<BE>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: crate::layouts::VecZnxBigToMut<BE>,
        A: crate::layouts::VecZnxBigToRef<BE>;

    fn vec_znx_big_sub_negate_assign<R, A>(module: &Module<BE>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: crate::layouts::VecZnxBigToMut<BE>,
        A: crate::layouts::VecZnxBigToRef<BE>;

    fn vec_znx_big_sub_small_a<R, A, C>(
        module: &Module<BE>,
        res: &mut R,
        res_col: usize,
        a: &A,
        a_col: usize,
        b: &C,
        b_col: usize,
    ) where
        R: crate::layouts::VecZnxBigToMut<BE>,
        A: VecZnxToRef,
        C: crate::layouts::VecZnxBigToRef<BE>;

    fn vec_znx_big_sub_small_assign<R, A>(module: &Module<BE>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: crate::layouts::VecZnxBigToMut<BE>,
        A: VecZnxToRef;

    fn vec_znx_big_sub_small_b<R, A, C>(
        module: &Module<BE>,
        res: &mut R,
        res_col: usize,
        a: &A,
        a_col: usize,
        b: &C,
        b_col: usize,
    ) where
        R: crate::layouts::VecZnxBigToMut<BE>,
        A: crate::layouts::VecZnxBigToRef<BE>,
        C: VecZnxToRef;

    fn vec_znx_big_sub_small_negate_assign<R, A>(module: &Module<BE>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: crate::layouts::VecZnxBigToMut<BE>,
        A: VecZnxToRef;

    fn vec_znx_big_negate<R, A>(module: &Module<BE>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: crate::layouts::VecZnxBigToMut<BE>,
        A: crate::layouts::VecZnxBigToRef<BE>;

    fn vec_znx_big_negate_assign<A>(module: &Module<BE>, a: &mut A, a_col: usize)
    where
        A: crate::layouts::VecZnxBigToMut<BE>;

    fn vec_znx_big_normalize_tmp_bytes(module: &Module<BE>) -> usize;

    #[allow(clippy::too_many_arguments)]
    fn vec_znx_big_normalize<R, A>(
        module: &Module<BE>,
        res: &mut R,
        res_base2k: usize,
        res_offset: i64,
        res_col: usize,
        a: &A,
        a_base2k: usize,
        a_col: usize,
        scratch: &mut Scratch<BE>,
    ) where
        R: VecZnxToMut,
        A: crate::layouts::VecZnxBigToRef<BE>;

    #[allow(clippy::too_many_arguments)]
    #[doc(hidden)]
    fn vec_znx_big_normalize_assign_fallback<R, A, const SUB: bool>(
        module: &Module<BE>,
        res: &mut R,
        res_base2k: usize,
        res_offset: i64,
        res_col: usize,
        a: &A,
        a_base2k: usize,
        a_col: usize,
        scratch: &mut Scratch<BE>,
    ) where
        R: VecZnxToMut,
        A: crate::layouts::VecZnxBigToRef<BE>,
    {
        let (n, size) = {
            let res_ref = res.to_mut();
            (res_ref.n, res_ref.size)
        };

        let mut tmp = VecZnx::alloc(n, 1, size);
        Self::vec_znx_big_normalize(module, &mut tmp, res_base2k, res_offset, 0, a, a_base2k, a_col, scratch);

        let mut res_ref = res.to_mut();
        for j in 0..size {
            for (ri, ti) in res_ref.at_mut(res_col, j).iter_mut().zip(tmp.at(0, j).iter()) {
                *ri = if SUB { ri.wrapping_sub(*ti) } else { ri.wrapping_add(*ti) };
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn vec_znx_big_normalize_add_assign<R, A>(
        module: &Module<BE>,
        res: &mut R,
        res_base2k: usize,
        res_offset: i64,
        res_col: usize,
        a: &A,
        a_base2k: usize,
        a_col: usize,
        scratch: &mut Scratch<BE>,
    ) where
        R: VecZnxToMut,
        A: crate::layouts::VecZnxBigToRef<BE>,
    {
        Self::vec_znx_big_normalize_assign_fallback::<R, A, false>(
            module, res, res_base2k, res_offset, res_col, a, a_base2k, a_col, scratch,
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn vec_znx_big_normalize_sub_assign<R, A>(
        module: &Module<BE>,
        res: &mut R,
        res_base2k: usize,
        res_offset: i64,
        res_col: usize,
        a: &A,
        a_base2k: usize,
        a_col: usize,
        scratch: &mut Scratch<BE>,
    ) where
        R: VecZnxToMut,
        A: crate::layouts::VecZnxBigToRef<BE>,
    {
        Self::vec_znx_big_normalize_assign_fallback::<R, A, true>(
            module, res, res_base2k, res_offset, res_col, a, a_base2k, a_col, scratch,
        );
    }

    fn vec_znx_big_automorphism<R, A>(module: &Module<BE>, k: i64, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: crate::layouts::VecZnxBigToMut<BE>,
        A: crate::layouts::VecZnxBigToRef<BE>;

    fn vec_znx_big_automorphism_assign_tmp_bytes(module: &Module<BE>) -> usize;

    fn vec_znx_big_automorphism_assign<A>(module: &Module<BE>, k: i64, a: &mut A, a_col: usize, scratch: &mut Scratch<BE>)
    where
        A: crate::layouts::VecZnxBigToMut<BE>;

    // VecZnxDft
    fn vec_znx_dft_apply<R, A>(module: &Module<BE>, step: usize, offset: usize, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: crate::layouts::VecZnxDftToMut<BE>,
        A: VecZnxToRef;

    fn vec_znx_idft_apply_tmp_bytes(module: &Module<BE>) -> usize;

    fn vec_znx_idft_apply<R, A>(module: &Module<BE>, res: &mut R, res_col: usize, a: &A, a_col: usize, scratch: &mut Scratch<BE>)
    where
        R: crate::layouts::VecZnxBigToMut<BE>,
        A: crate::layouts::VecZnxDftToRef<BE>;

    fn vec_znx_idft_apply_tmpa<R, A>(module: &Module<BE>, res: &mut R, res_col: usize, a: &mut A, a_col: usize)
    where
        R: crate::layouts::VecZnxBigToMut<BE>,
        A: crate::layouts::VecZnxDftToMut<BE>;

    fn vec_znx_idft_apply_consume<D: crate::layouts::Data>(
        module: &Module<BE>,
        a: crate::layouts::VecZnxDft<D, BE>,
    ) -> crate::layouts::VecZnxBig<D, BE>
    where
        crate::layouts::VecZnxDft<D, BE>: crate::layouts::VecZnxDftToMut<BE>;

    fn vec_znx_dft_add_into<R, A, D>(module: &Module<BE>, res: &mut R, res_col: usize, a: &A, a_col: usize, b: &D, b_col: usize)
    where
        R: crate::layouts::VecZnxDftToMut<BE>,
        A: crate::layouts::VecZnxDftToRef<BE>,
        D: crate::layouts::VecZnxDftToRef<BE>;

    fn vec_znx_dft_add_scaled_assign<R, A>(module: &Module<BE>, res: &mut R, res_col: usize, a: &A, a_col: usize, a_scale: i64)
    where
        R: crate::layouts::VecZnxDftToMut<BE>,
        A: crate::layouts::VecZnxDftToRef<BE>;

    fn vec_znx_dft_add_assign<R, A>(module: &Module<BE>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: crate::layouts::VecZnxDftToMut<BE>,
        A: crate::layouts::VecZnxDftToRef<BE>;

    fn vec_znx_dft_sub<R, A, D>(module: &Module<BE>, res: &mut R, res_col: usize, a: &A, a_col: usize, b: &D, b_col: usize)
    where
        R: crate::layouts::VecZnxDftToMut<BE>,
        A: crate::layouts::VecZnxDftToRef<BE>,
        D: crate::layouts::VecZnxDftToRef<BE>;

    fn vec_znx_dft_sub_assign<R, A>(module: &Module<BE>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: crate::layouts::VecZnxDftToMut<BE>,
        A: crate::layouts::VecZnxDftToRef<BE>;

    fn vec_znx_dft_sub_negate_assign<R, A>(module: &Module<BE>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: crate::layouts::VecZnxDftToMut<BE>,
        A: crate::layouts::VecZnxDftToRef<BE>;

    fn vec_znx_dft_copy<R, A>(module: &Module<BE>, step: usize, offset: usize, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: crate::layouts::VecZnxDftToMut<BE>,
        A: crate::layouts::VecZnxDftToRef<BE>;

    fn vec_znx_dft_zero<R>(module: &Module<BE>, res: &mut R, res_col: usize)
    where
        R: crate::layouts::VecZnxDftToMut<BE>;

    // SVP
    fn svp_prepare<R, A>(module: &Module<BE>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: crate::layouts::SvpPPolToMut<BE>,
        A: ScalarZnxToRef;

    fn svp_apply_dft<R, A, C>(module: &Module<BE>, res: &mut R, res_col: usize, a: &A, a_col: usize, b: &C, b_col: usize)
    where
        R: crate::layouts::VecZnxDftToMut<BE>,
        A: crate::layouts::SvpPPolToRef<BE>,
        C: VecZnxToRef;

    fn svp_apply_dft_to_dft<R, A, C>(module: &Module<BE>, res: &mut R, res_col: usize, a: &A, a_col: usize, b: &C, b_col: usize)
    where
        R: crate::layouts::VecZnxDftToMut<BE>,
        A: crate::layouts::SvpPPolToRef<BE>,
        C: crate::layouts::VecZnxDftToRef<BE>;

    fn svp_apply_dft_to_dft_assign<R, A>(module: &Module<BE>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: crate::layouts::VecZnxDftToMut<BE>,
        A: crate::layouts::SvpPPolToRef<BE>;

    // VMP
    fn vmp_prepare_tmp_bytes(module: &Module<BE>, rows: usize, cols_in: usize, cols_out: usize, size: usize) -> usize;

    fn vmp_prepare<R, A>(module: &Module<BE>, res: &mut R, a: &A, scratch: &mut Scratch<BE>)
    where
        R: crate::layouts::VmpPMatToMut<BE>,
        A: crate::layouts::MatZnxToRef;

    #[allow(clippy::too_many_arguments)]
    fn vmp_apply_dft_tmp_bytes(
        module: &Module<BE>,
        res_size: usize,
        a_size: usize,
        b_rows: usize,
        b_cols_in: usize,
        b_cols_out: usize,
        b_size: usize,
    ) -> usize;

    fn vmp_apply_dft<R, A, C>(module: &Module<BE>, res: &mut R, a: &A, b: &C, scratch: &mut Scratch<BE>)
    where
        R: crate::layouts::VecZnxDftToMut<BE>,
        A: VecZnxToRef,
        C: crate::layouts::VmpPMatToRef<BE>;

    #[allow(clippy::too_many_arguments)]
    fn vmp_apply_dft_to_dft_tmp_bytes(
        module: &Module<BE>,
        res_size: usize,
        a_size: usize,
        b_rows: usize,
        b_cols_in: usize,
        b_cols_out: usize,
        b_size: usize,
    ) -> usize;

    fn vmp_apply_dft_to_dft<R, A, C>(
        module: &Module<BE>,
        res: &mut R,
        a: &A,
        b: &C,
        limb_offset: usize,
        scratch: &mut Scratch<BE>,
    ) where
        R: crate::layouts::VecZnxDftToMut<BE>,
        A: crate::layouts::VecZnxDftToRef<BE>,
        C: crate::layouts::VmpPMatToRef<BE>;

    fn vmp_zero<R>(module: &Module<BE>, res: &mut R)
    where
        R: crate::layouts::VmpPMatToMut<BE>;

    // Convolution
    fn cnv_prepare_left_tmp_bytes(module: &Module<BE>, res_size: usize, a_size: usize) -> usize;

    fn cnv_prepare_left<R, A>(module: &Module<BE>, res: &mut R, a: &A, mask: i64, scratch: &mut Scratch<BE>)
    where
        R: crate::layouts::CnvPVecLToMut<BE>,
        A: VecZnxToRef;

    fn cnv_prepare_right_tmp_bytes(module: &Module<BE>, res_size: usize, a_size: usize) -> usize;

    fn cnv_prepare_right<R, A>(module: &Module<BE>, res: &mut R, a: &A, mask: i64, scratch: &mut Scratch<BE>)
    where
        R: crate::layouts::CnvPVecRToMut<BE>,
        A: VecZnxToRef + crate::layouts::ZnxInfos;

    fn cnv_apply_dft_tmp_bytes(module: &Module<BE>, cnv_offset: usize, res_size: usize, a_size: usize, b_size: usize) -> usize;

    fn cnv_by_const_apply_tmp_bytes(
        module: &Module<BE>,
        cnv_offset: usize,
        res_size: usize,
        a_size: usize,
        b_size: usize,
    ) -> usize;

    #[allow(clippy::too_many_arguments)]
    fn cnv_by_const_apply<R, A>(
        module: &Module<BE>,
        cnv_offset: usize,
        res: &mut R,
        res_col: usize,
        a: &A,
        a_col: usize,
        b: &[i64],
        scratch: &mut Scratch<BE>,
    ) where
        R: crate::layouts::VecZnxBigToMut<BE>,
        A: VecZnxToRef;

    #[allow(clippy::too_many_arguments)]
    fn cnv_apply_dft<R, A, B>(
        module: &Module<BE>,
        cnv_offset: usize,
        res: &mut R,
        res_col: usize,
        a: &A,
        a_col: usize,
        b: &B,
        b_col: usize,
        scratch: &mut Scratch<BE>,
    ) where
        R: crate::layouts::VecZnxDftToMut<BE>,
        A: crate::layouts::CnvPVecLToRef<BE>,
        B: crate::layouts::CnvPVecRToRef<BE>;

    fn cnv_pairwise_apply_dft_tmp_bytes(
        module: &Module<BE>,
        cnv_offset: usize,
        res_size: usize,
        a_size: usize,
        b_size: usize,
    ) -> usize;

    #[allow(clippy::too_many_arguments)]
    fn cnv_pairwise_apply_dft<R, A, B>(
        module: &Module<BE>,
        cnv_offset: usize,
        res: &mut R,
        res_col: usize,
        a: &A,
        b: &B,
        i: usize,
        j: usize,
        scratch: &mut Scratch<BE>,
    ) where
        R: crate::layouts::VecZnxDftToMut<BE>,
        A: crate::layouts::CnvPVecLToRef<BE>,
        B: crate::layouts::CnvPVecRToRef<BE>;

    fn cnv_prepare_self_tmp_bytes(module: &Module<BE>, res_size: usize, a_size: usize) -> usize;

    fn cnv_prepare_self<L, R, A>(module: &Module<BE>, left: &mut L, right: &mut R, a: &A, mask: i64, scratch: &mut Scratch<BE>)
    where
        L: crate::layouts::CnvPVecLToMut<BE>,
        R: crate::layouts::CnvPVecRToMut<BE>,
        A: VecZnxToRef + crate::layouts::ZnxInfos;
}
