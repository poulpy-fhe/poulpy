//! GLWE external-product internals + reference implementations of the
//! [`GLWEExternalProductDefault`] methods.
//!
//! Re-exported publicly through `crate::oep::glwe_external_product_defaults`.

use crate::api::GLWEBytesOf;
use poulpy_hal::layouts::VecZnxDftBackendMut;
use poulpy_hal::{
    api::{
        ModuleN, ScratchArenaTakeBasic, VecZnxBigBytesOf, VecZnxBigNormalize, VecZnxBigNormalizeTmpBytes, VecZnxDftApply,
        VecZnxDftBytesOf, VecZnxIdftApply, VecZnxIdftApplyTmpBytes, VecZnxNormalize, VecZnxNormalizeTmpBytes, VmpApplyDftToDft,
        VmpApplyDftToDftAccumulate, VmpApplyDftToDftTmpBytes,
    },
    layouts::{Backend, Module, ScratchArena, VecZnxBigToBackendRef, VecZnxDftToBackendRef},
};

use crate::{
    ScratchArenaTakeCore,
    api::GLWEExternalProductInternal,
    default::operations::GLWENormalizeDefault,
    layouts::{
        GGSWInfos, GGSWPreparedBackendRef, GLWEBackendRef, GLWEInfos, GLWELayout, GLWEToBackendMut, GLWEToBackendRef,
        GadgetProductOutputSizeParams, LWEInfos, gadget_product_output_size, prepared::GGSWPreparedToBackendRef,
    },
    oep::{GLWEExternalProductDefault, gglwe_product_digit_output_size},
};

/// Practical limb window used for an immediately normalized GGSW external
/// product.
///
/// This is public so fused higher-level operations which use
/// [`GLWEExternalProductInternal`] can size their DFT/BIG intermediates with
/// exactly the same rule as the default implementation. Although any lower
/// limb can affect rounding through a sufficiently long carry chain, the
/// window includes the worst-case norm growth of the signed DFT products and
/// their VMP accumulation.
pub fn glwe_external_product_output_size<BE, R, A, G>(res_infos: &R, a_infos: &A, ggsw_infos: &G) -> usize
where
    BE: Backend,
    R: GLWEInfos,
    A: GLWEInfos,
    G: GGSWInfos,
{
    let product_terms = ggsw_infos
        .n()
        .as_usize()
        .saturating_mul(ggsw_infos.dnum().as_usize())
        .saturating_mul(ggsw_infos.dsize().as_usize())
        .saturating_mul((ggsw_infos.rank() + 1).as_usize());
    gadget_product_output_size(GadgetProductOutputSizeParams {
        key_size: ggsw_infos.size(),
        key_base2k: ggsw_infos.base2k(),
        input_k: a_infos.k(),
        output_k: res_infos.k(),
        dsize: ggsw_infos.dsize(),
        k_aux: ggsw_infos.k_aux(),
        dft_is_exact: BE::DFT_IS_EXACT,
        product_terms,
        extra_live_limbs: 0,
    })
}

fn glwe_external_product_dft_fill<BE, M>(
    module: &M,
    res_dft: &mut VecZnxDftBackendMut<'_, BE>,
    a: GLWEBackendRef<'_, BE>,
    ggsw: &GGSWPreparedBackendRef<'_, BE>,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: Backend,
    M: GLWEBytesOf<BE>
        + ModuleN
        + VecZnxDftBytesOf
        + VmpApplyDftToDftTmpBytes
        + VecZnxNormalizeTmpBytes
        + VecZnxDftApply<BE>
        + VmpApplyDftToDft<BE>
        + VmpApplyDftToDftAccumulate<BE>
        + VecZnxIdftApply<BE>
        + VecZnxIdftApplyTmpBytes,
{
    let cols: usize = (ggsw.rank() + 1).into();
    let dsize: usize = ggsw.dsize().into();
    let a_size: usize = a.size();
    {
        if dsize == 1 {
            let (mut a_dft, mut scratch_1) = scratch.borrow().take_vec_znx_dft_scratch(module, cols, a_size);
            for j in 0..cols {
                module.vec_znx_dft_apply(1, 0, &mut a_dft, j, &a.data, j);
            }
            let a_dft_ref = a_dft.to_backend_ref();
            module.vmp_apply_dft_to_dft(res_dft, &a_dft_ref, &ggsw.data, 0, &mut scratch_1.borrow());
        } else {
            // Same shape as `gglwe_product_dft_default`, and the same two
            // constraints hold; see the comment there for why. In short:
            // `di == 0` is the overwriting pass and must run at the **full**
            // width so no limb of `res_dft` keeps stale scratch, and it must be
            // `di == 0` because `vmp_apply_dft_to_dft` covers its destination
            // fully only at `limb_offset == 0`. The accumulating passes may keep
            // the narrow view. `- 2` rather than `- 1` because an elementary
            // limb product spans two limbs; do not tighten it.
            //
            // The one difference from the keyswitch: there the operand arrives
            // already in DFT and each digit is sliced out with
            // `vec_znx_dft_copy`, here it arrives in coefficients and the stride
            // is folded into `vec_znx_dft_apply`, so no full-width DFT of `a` is
            // ever materialized.
            for di in 0..dsize {
                let (mut a_dft, mut scratch_1) = scratch.borrow().take_vec_znx_dft_scratch(module, cols, (a_size + di) / dsize);

                for j in 0..cols {
                    module.vec_znx_dft_apply(dsize, dsize - 1 - di, &mut a_dft, j, &a.data, j);
                }

                if di == 0 {
                    module.vmp_apply_dft_to_dft(res_dft, &a_dft.to_backend_ref(), &ggsw.data, 0, &mut scratch_1.borrow());
                } else {
                    let res_compute_size = gglwe_product_digit_output_size(res_dft.size(), ggsw.size(), dsize, di);
                    let mut res_view = res_dft.with_size_mut(res_compute_size);
                    module.vmp_apply_dft_to_dft_accumulate(
                        &mut res_view,
                        &a_dft.to_backend_ref(),
                        &ggsw.data,
                        di,
                        &mut scratch_1.borrow(),
                    );
                }
            }
        }
    }
}

impl<BE: Backend> GLWEExternalProductInternal<BE> for Module<BE>
where
    Self: ModuleN
        + VecZnxDftBytesOf
        + VmpApplyDftToDftTmpBytes
        + VecZnxNormalizeTmpBytes
        + VecZnxDftApply<BE>
        + VmpApplyDftToDft<BE>
        + VmpApplyDftToDftAccumulate<BE>
        + VecZnxBigBytesOf
        + VecZnxIdftApply<BE>
        + VecZnxIdftApplyTmpBytes
        + VecZnxBigNormalize<BE>
        + VecZnxNormalize<BE>,
{
    fn glwe_external_product_internal_tmp_bytes<R, A, B>(&self, res_infos: &R, a_infos: &A, b_infos: &B) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        B: GGSWInfos,
    {
        let align: usize = BE::SCRATCH_ALIGN;
        let in_size: usize = a_infos.k().div_ceil(b_infos.base2k()).div_ceil(b_infos.dsize().into()) as usize;
        let output_size = glwe_external_product_output_size::<BE, _, _, _>(res_infos, a_infos, b_infos);
        let cols: usize = (b_infos.rank() + 1).into();
        let lvl_0: usize = self.bytes_of_vec_znx_dft(cols, in_size);
        let lvl_1: usize = if b_infos.dsize() > 1 {
            self.bytes_of_vec_znx_dft(cols, output_size)
        } else {
            0
        };
        let lvl_2: usize = self.vmp_apply_dft_to_dft_tmp_bytes(output_size, in_size, in_size, cols, cols, b_infos.size());
        let lvl_3: usize =
            self.bytes_of_vec_znx_big(cols, output_size).next_multiple_of(align) + self.vec_znx_idft_apply_tmp_bytes();
        (lvl_0.next_multiple_of(align) + lvl_1.next_multiple_of(align) + lvl_2).max(lvl_3)
    }

    fn glwe_external_product_dft<'r, A, G>(
        &self,
        res_dft: &mut VecZnxDftBackendMut<'r, BE>,
        a: &A,
        ggsw: &G,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        A: GLWEToBackendRef<BE>,
        G: GGSWPreparedToBackendRef<BE>,
    {
        let ggsw: GGSWPreparedBackendRef<'_, BE> = ggsw.to_backend_ref();
        let a = a.to_backend_ref();
        glwe_external_product_dft_fill(self, res_dft, a, &ggsw, scratch);
    }
}

// === Free-function defaults for GLWEExternalProductDefault ===

pub fn glwe_external_product_dft_fill_tmp_bytes_default<BE, M, A, G>(module: &M, a_infos: &A, ggsw_infos: &G) -> usize
where
    BE: Backend,
    M: VecZnxDftBytesOf + VmpApplyDftToDftTmpBytes,
    A: GLWEInfos,
    G: GGSWInfos,
{
    let align: usize = BE::SCRATCH_ALIGN;
    let in_size: usize = a_infos.k().div_ceil(ggsw_infos.base2k()).div_ceil(ggsw_infos.dsize().into()) as usize;
    let ggsw_size: usize = ggsw_infos.size();
    let cols: usize = (ggsw_infos.rank() + 1).into();
    let lvl_0: usize = module.bytes_of_vec_znx_dft(cols, in_size);
    let lvl_1: usize = if ggsw_infos.dsize() > 1 {
        module.bytes_of_vec_znx_dft(cols, ggsw_size)
    } else {
        0
    };
    let lvl_2: usize = module.vmp_apply_dft_to_dft_tmp_bytes(ggsw_size, in_size, in_size, cols, cols, ggsw_size);
    lvl_0.next_multiple_of(align) + lvl_1.next_multiple_of(align) + lvl_2
}

pub fn glwe_external_product_tmp_bytes_default<BE, M, R, A, G>(module: &M, res_infos: &R, a_infos: &A, ggsw_infos: &G) -> usize
where
    BE: Backend,
    M: GLWEBytesOf<BE>
        + GLWEExternalProductDefault<BE>
        + GLWEExternalProductInternal<BE>
        + GLWENormalizeDefault<BE>
        + ModuleN
        + VecZnxDftBytesOf
        + VecZnxBigBytesOf
        + VmpApplyDftToDftTmpBytes
        + VecZnxIdftApplyTmpBytes
        + VecZnxBigNormalizeTmpBytes,
    R: GLWEInfos,
    A: GLWEInfos,
    G: GGSWInfos,
{
    let align: usize = BE::SCRATCH_ALIGN;
    let cols: usize = res_infos.rank().as_usize() + 1;
    let output_size = glwe_external_product_output_size::<BE, _, _, _>(res_infos, a_infos, ggsw_infos);
    let lvl_0: usize = module.bytes_of_vec_znx_dft(cols, output_size);
    let lvl_1: usize = module.bytes_of_vec_znx_big(cols, output_size).next_multiple_of(align)
        + module
            .vec_znx_idft_apply_tmp_bytes()
            .max(module.vec_znx_big_normalize_tmp_bytes());
    let lvl_2: usize = if a_infos.base2k() != ggsw_infos.base2k() {
        let a_conv_infos = GLWELayout {
            n: a_infos.n(),
            base2k: ggsw_infos.base2k(),
            k: a_infos.k(),
            rank: a_infos.rank(),
        };
        let lvl_2_0: usize = module.glwe_bytes_of_from_infos(&a_conv_infos);
        let lvl_2_1: usize = module
            .glwe_normalize_tmp_bytes_default()
            .max(module.glwe_external_product_dft_fill_tmp_bytes_default(&a_conv_infos, ggsw_infos));
        lvl_2_0 + lvl_2_1
    } else {
        module.glwe_external_product_internal_tmp_bytes(res_infos, a_infos, ggsw_infos)
    };
    lvl_0.next_multiple_of(align) + lvl_1.max(lvl_2)
}

pub fn glwe_external_product_default<BE, M, R, A, G>(module: &M, res: &mut R, a: &A, ggsw: &G, scratch: &mut ScratchArena<'_, BE>)
where
    BE: Backend,
    M: GLWEBytesOf<BE>
        + GLWEExternalProductDefault<BE>
        + GLWEExternalProductInternal<BE>
        + GLWENormalizeDefault<BE>
        + ModuleN
        + VecZnxBigBytesOf
        + VecZnxBigNormalize<BE>
        + VecZnxDftBytesOf
        + VecZnxIdftApply<BE>,
    R: GLWEToBackendMut<BE> + GLWEInfos,
    A: GLWEToBackendRef<BE> + GLWEInfos,
    G: GGSWPreparedToBackendRef<BE> + GGSWInfos,
{
    assert_eq!(ggsw.rank(), a.rank());
    assert_eq!(ggsw.rank(), res.rank());
    assert_eq!(ggsw.n(), res.n());
    assert_eq!(a.n(), res.n());
    assert!(
        scratch.available() >= module.glwe_external_product_tmp_bytes_default(res, a, ggsw),
        "scratch.available(): {} < GLWEExternalProduct::glwe_external_product_tmp_bytes: {}",
        scratch.available(),
        module.glwe_external_product_tmp_bytes_default(res, a, ggsw)
    );

    let output_size = glwe_external_product_output_size::<BE, _, _, _>(res, a, ggsw);

    let a_base2k: usize = a.base2k().into();
    let ggsw_base2k: usize = ggsw.base2k().into();
    let res_base2k: usize = res.base2k().into();
    let cols: usize = (res.rank() + 1).into();
    let (mut res_dft, scratch_1) = scratch
        .borrow()
        .take_vec_znx_dft_scratch(module, (res.rank() + 1).into(), output_size);

    let mut scratch = scratch_1;
    if a_base2k != ggsw_base2k {
        scratch.scope(|scratch_phase| {
            let (mut a_conv, mut scratch_2) = scratch_phase.take_glwe_scratch(&GLWELayout {
                n: a.n(),
                base2k: ggsw.base2k(),
                k: a.k(),
                rank: a.rank(),
            });
            module.glwe_normalize_default(&mut a_conv, a, &mut scratch_2.borrow());
            module.glwe_external_product_dft(&mut res_dft, &a_conv, ggsw, &mut scratch_2);
        });
    } else {
        module.glwe_external_product_dft(&mut res_dft, a, ggsw, &mut scratch.borrow());
    }

    let (mut res_big, mut scratch) = scratch.borrow().take_vec_znx_big_scratch(module, cols, res_dft.size());
    let res_dft_ref = res_dft.to_backend_ref();
    for col in 0..cols {
        module.vec_znx_idft_apply(&mut res_big, col, &res_dft_ref, col, &mut scratch.borrow());
    }
    let res_big_ref = res_big.to_backend_ref();
    let mut res_ref = res.to_backend_mut();
    for j in 0..cols {
        module.vec_znx_big_normalize(
            &mut res_ref.data,
            res_base2k,
            0,
            j,
            &res_big_ref,
            ggsw_base2k,
            j,
            &mut scratch.borrow(),
        );
    }
}

pub fn glwe_external_product_assign_default<BE, M, R, G>(module: &M, res: &mut R, ggsw: &G, scratch: &mut ScratchArena<'_, BE>)
where
    BE: Backend,
    M: GLWEBytesOf<BE>
        + GLWEExternalProductDefault<BE>
        + GLWEExternalProductInternal<BE>
        + GLWENormalizeDefault<BE>
        + ModuleN
        + VecZnxBigBytesOf
        + VecZnxBigNormalize<BE>
        + VecZnxDftBytesOf
        + VecZnxIdftApply<BE>,
    R: GLWEToBackendMut<BE> + GLWEInfos,
    G: GGSWPreparedToBackendRef<BE> + GGSWInfos,
{
    assert_eq!(ggsw.rank(), res.rank());
    assert_eq!(ggsw.n(), res.n());
    assert!(
        scratch.available() >= module.glwe_external_product_tmp_bytes_default(res, res, ggsw),
        "scratch.available(): {} < GLWEExternalProduct::glwe_external_product_tmp_bytes: {}",
        scratch.available(),
        module.glwe_external_product_tmp_bytes_default(res, res, ggsw)
    );

    let output_size = glwe_external_product_output_size::<BE, _, _, _>(res, res, ggsw);
    let res_base2k: usize = res.base2k().as_usize();
    let ggsw_base2k: usize = ggsw.base2k().as_usize();
    let cols: usize = (res.rank() + 1).into();
    let (mut res_dft, scratch_1) = scratch
        .borrow()
        .take_vec_znx_dft_scratch(module, (res.rank() + 1).into(), output_size);

    let mut scratch = scratch_1;
    if res_base2k != ggsw_base2k {
        scratch.scope(|scratch_phase| {
            let (mut res_conv, mut scratch_2) = scratch_phase.take_glwe_scratch(&GLWELayout {
                n: res.n(),
                base2k: ggsw.base2k(),
                k: res.k(),
                rank: res.rank(),
            });
            module.glwe_normalize_default(&mut res_conv, res, &mut scratch_2.borrow());
            module.glwe_external_product_dft(&mut res_dft, &res_conv, ggsw, &mut scratch_2);
        });
    } else {
        module.glwe_external_product_dft(&mut res_dft, res, ggsw, &mut scratch.borrow());
    }

    let (mut res_big, mut scratch) = scratch.borrow().take_vec_znx_big_scratch(module, cols, res_dft.size());
    let res_dft_ref = res_dft.to_backend_ref();
    for col in 0..cols {
        module.vec_znx_idft_apply(&mut res_big, col, &res_dft_ref, col, &mut scratch.borrow());
    }
    let res_big_ref = res_big.to_backend_ref();
    let mut res_ref = res.to_backend_mut();
    for j in 0..cols {
        module.vec_znx_big_normalize(
            &mut res_ref.data,
            res_base2k,
            0,
            j,
            &res_big_ref,
            ggsw_base2k,
            j,
            &mut scratch.borrow(),
        );
    }
}
