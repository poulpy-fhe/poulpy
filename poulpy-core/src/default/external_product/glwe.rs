//! GLWE external-product internals + reference implementations of the
//! [`GLWEExternalProductDefault`] methods.
//!
//! Re-exported publicly through `crate::oep::glwe_external_product_defaults`.

#![allow(private_bounds)]

use poulpy_hal::{
    api::{
        ModuleN, ScratchArenaTakeBasic, VecZnxBigBytesOf, VecZnxBigNormalize, VecZnxBigNormalizeTmpBytes, VecZnxDftAddAssign,
        VecZnxDftApply, VecZnxDftBytesOf, VecZnxDftZero, VecZnxIdftApply, VecZnxIdftApplyTmpBytes, VecZnxNormalize,
        VecZnxNormalizeTmpBytes, VmpApplyDftToDft, VmpApplyDftToDftTmpBytes,
    },
    layouts::{Backend, Module, ScratchArena, VecZnxBigToBackendRef, VecZnxDft, VecZnxDftToBackendRef},
};

use crate::{
    ScratchArenaTakeCore,
    api::GLWEExternalProductInternal,
    default::operations::GLWENormalizeDefault,
    layouts::{
        GGSWInfos, GGSWPreparedBackendRef, GLWE, GLWEBackendRef, GLWEInfos, GLWELayout, GLWEToBackendMut, GLWEToBackendRef,
        LWEInfos, prepared::GGSWPreparedToBackendRef,
    },
    oep::GLWEExternalProductDefault,
};

fn glwe_external_product_dft_fill<'s, 'r, BE, M>(
    module: &M,
    res_dft: &mut VecZnxDft<BE::BufMut<'r>, BE>,
    a: GLWEBackendRef<'_, BE>,
    ggsw: &GGSWPreparedBackendRef<'_, BE>,
    scratch: &mut ScratchArena<'s, BE>,
) where
    BE: Backend + 's,
    M: ModuleN
        + VecZnxDftBytesOf
        + VmpApplyDftToDftTmpBytes
        + VecZnxNormalizeTmpBytes
        + VecZnxDftApply<BE>
        + VmpApplyDftToDft<BE>
        + VecZnxDftAddAssign<BE>
        + VecZnxIdftApply<BE>
        + VecZnxIdftApplyTmpBytes
        + VecZnxDftZero<BE>,
    for<'x> ScratchArena<'x, BE>: ScratchArenaTakeCore<'x, BE>,
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
            for di in 0..dsize {
                let (mut a_dft, mut scratch_1) = scratch
                    .borrow()
                    .take_vec_znx_dft_scratch(module, cols, (a.size() + di) / dsize);
                res_dft.set_size(res_dft.max_size() - ((dsize - di) as isize - 2).max(0) as usize);

                for j in 0..cols {
                    module.vec_znx_dft_apply(dsize, dsize - 1 - di, &mut a_dft, j, &a.data, j);
                }

                if di == 0 {
                    module.vmp_apply_dft_to_dft(res_dft, &a_dft.to_backend_ref(), &ggsw.data, 0, &mut scratch_1.borrow());
                } else {
                    let (mut res_dft_tmp, mut scratch_2) =
                        scratch_1
                            .borrow()
                            .take_vec_znx_dft_scratch(module, res_dft.cols(), res_dft.size());
                    module.vmp_apply_dft_to_dft(
                        &mut res_dft_tmp,
                        &a_dft.to_backend_ref(),
                        &ggsw.data,
                        di,
                        &mut scratch_2.borrow(),
                    );
                    for col in 0..cols {
                        module.vec_znx_dft_add_assign(res_dft, col, &res_dft_tmp.to_backend_ref(), col);
                    }
                }
            }
        }
    }
    res_dft.set_size(res_dft.max_size());
}

impl<BE: Backend> GLWEExternalProductInternal<BE> for Module<BE>
where
    Self: ModuleN
        + VecZnxDftBytesOf
        + VmpApplyDftToDftTmpBytes
        + VecZnxNormalizeTmpBytes
        + VecZnxDftApply<BE>
        + VmpApplyDftToDft<BE>
        + VecZnxDftAddAssign<BE>
        + VecZnxBigBytesOf
        + VecZnxIdftApply<BE>
        + VecZnxIdftApplyTmpBytes
        + VecZnxBigNormalize<BE>
        + VecZnxNormalize<BE>
        + VecZnxDftZero<BE>,
{
    fn glwe_external_product_internal_tmp_bytes<R, A, B>(&self, _res_infos: &R, a_infos: &A, b_infos: &B) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        B: GGSWInfos,
    {
        let align: usize = BE::SCRATCH_ALIGN;
        let in_size: usize = a_infos.max_k().div_ceil(b_infos.base2k()).div_ceil(b_infos.dsize().into()) as usize;
        let ggsw_size: usize = b_infos.size();
        let cols: usize = (b_infos.rank() + 1).into();
        let lvl_0: usize = self.bytes_of_vec_znx_dft(cols, in_size);
        let lvl_1: usize = if b_infos.dsize() > 1 {
            self.bytes_of_vec_znx_dft(cols, ggsw_size)
        } else {
            0
        };
        let lvl_2: usize = self.vmp_apply_dft_to_dft_tmp_bytes(ggsw_size, in_size, in_size, cols, cols, ggsw_size);
        let lvl_3: usize =
            self.bytes_of_vec_znx_big(cols, ggsw_size).next_multiple_of(align) + self.vec_znx_idft_apply_tmp_bytes();
        (lvl_0.next_multiple_of(align) + lvl_1.next_multiple_of(align) + lvl_2).max(lvl_3)
    }

    fn glwe_external_product_dft<'s, 'r, A, G>(
        &self,
        res_dft: &mut VecZnxDft<BE::BufMut<'r>, BE>,
        a: &A,
        ggsw: &G,
        _key_size: usize,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        A: GLWEToBackendRef<BE>,
        G: GGSWPreparedToBackendRef<BE>,
        for<'b> ScratchArena<'b, BE>: ScratchArenaTakeCore<'b, BE>,
        BE: 's,
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
    let in_size: usize = a_infos
        .max_k()
        .div_ceil(ggsw_infos.base2k())
        .div_ceil(ggsw_infos.dsize().into()) as usize;
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
    M: GLWEExternalProductDefault<BE>
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
    for<'s> ScratchArena<'s, BE>: ScratchArenaTakeCore<'s, BE>,
{
    let align: usize = BE::SCRATCH_ALIGN;
    let cols: usize = res_infos.rank().as_usize() + 1;
    let lvl_0: usize = module.bytes_of_vec_znx_dft(cols, ggsw_infos.size());
    let lvl_1: usize = module.bytes_of_vec_znx_big(cols, ggsw_infos.size()).next_multiple_of(align)
        + module
            .vec_znx_idft_apply_tmp_bytes()
            .max(module.vec_znx_big_normalize_tmp_bytes());
    let lvl_2: usize = if a_infos.base2k() != ggsw_infos.base2k() {
        let a_conv_infos = GLWELayout {
            n: a_infos.n(),
            base2k: ggsw_infos.base2k(),
            k: a_infos.max_k(),
            rank: a_infos.rank(),
        };
        let lvl_2_0: usize = GLWE::<Vec<u8>>::bytes_of_from_infos(&a_conv_infos);
        let lvl_2_1: usize = module
            .glwe_normalize_tmp_bytes_default()
            .max(module.glwe_external_product_dft_fill_tmp_bytes_default(&a_conv_infos, ggsw_infos));
        lvl_2_0 + lvl_2_1
    } else {
        module.glwe_external_product_internal_tmp_bytes(res_infos, a_infos, ggsw_infos)
    };
    lvl_0.next_multiple_of(align) + lvl_1.max(lvl_2)
}

pub fn glwe_external_product_default<'s, BE, M, R, A, G>(
    module: &M,
    res: &mut R,
    a: &A,
    ggsw: &G,
    key_size: usize,
    scratch: &mut ScratchArena<'s, BE>,
) where
    BE: Backend + 's,
    M: GLWEExternalProductDefault<BE>
        + GLWEExternalProductInternal<BE>
        + GLWENormalizeDefault<BE>
        + ModuleN
        + VecZnxBigBytesOf
        + VecZnxBigNormalize<BE>
        + VecZnxDftBytesOf
        + VecZnxDftZero<BE>
        + VecZnxIdftApply<BE>,
    R: GLWEToBackendMut<BE> + GLWEInfos,
    A: GLWEToBackendRef<BE> + GLWEInfos,
    G: GGSWPreparedToBackendRef<BE> + GGSWInfos,
    for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
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

    let key_size = ggsw.size().min(key_size);

    let a_base2k: usize = a.base2k().into();
    let ggsw_base2k: usize = ggsw.base2k().into();
    let res_base2k: usize = res.base2k().into();
    let cols: usize = (res.rank() + 1).into();
    let (mut res_dft, scratch_1) = scratch
        .borrow()
        .take_vec_znx_dft_scratch(module, (res.rank() + 1).into(), key_size);
    for col in 0..res_dft.cols() {
        module.vec_znx_dft_zero(&mut res_dft, col);
    }

    let mut scratch = scratch_1;
    if a_base2k != ggsw_base2k {
        scratch.scope(|scratch_phase| {
            let (mut a_conv, mut scratch_2) = scratch_phase.take_glwe_scratch(&GLWELayout {
                n: a.n(),
                base2k: ggsw.base2k(),
                k: a.max_k(),
                rank: a.rank(),
            });
            module.glwe_normalize_default(&mut a_conv, a, &mut scratch_2.borrow());
            module.glwe_external_product_dft(&mut res_dft, &a_conv, ggsw, key_size, &mut scratch_2);
        });
    } else {
        module.glwe_external_product_dft(&mut res_dft, a, ggsw, key_size, &mut scratch.borrow());
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

pub fn glwe_external_product_assign_default<'s, BE, M, R, G>(
    module: &M,
    res: &mut R,
    ggsw: &G,
    key_size: usize,
    scratch: &mut ScratchArena<'s, BE>,
) where
    BE: Backend + 's,
    M: GLWEExternalProductDefault<BE>
        + GLWEExternalProductInternal<BE>
        + GLWENormalizeDefault<BE>
        + ModuleN
        + VecZnxBigBytesOf
        + VecZnxBigNormalize<BE>
        + VecZnxDftBytesOf
        + VecZnxDftZero<BE>
        + VecZnxIdftApply<BE>,
    R: GLWEToBackendMut<BE> + GLWEInfos,
    G: GGSWPreparedToBackendRef<BE> + GGSWInfos,
    for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
{
    assert_eq!(ggsw.rank(), res.rank());
    assert_eq!(ggsw.n(), res.n());
    assert!(
        scratch.available() >= module.glwe_external_product_tmp_bytes_default(res, res, ggsw),
        "scratch.available(): {} < GLWEExternalProduct::glwe_external_product_tmp_bytes: {}",
        scratch.available(),
        module.glwe_external_product_tmp_bytes_default(res, res, ggsw)
    );

    let res_base2k: usize = res.base2k().as_usize();
    let ggsw_base2k: usize = ggsw.base2k().as_usize();
    let cols: usize = (res.rank() + 1).into();
    let (mut res_dft, scratch_1) = scratch
        .borrow()
        .take_vec_znx_dft_scratch(module, (res.rank() + 1).into(), key_size);
    for col in 0..res_dft.cols() {
        module.vec_znx_dft_zero(&mut res_dft, col);
    }

    let mut scratch = scratch_1;
    if res_base2k != ggsw_base2k {
        scratch.scope(|scratch_phase| {
            let (mut res_conv, mut scratch_2) = scratch_phase.take_glwe_scratch(&GLWELayout {
                n: res.n(),
                base2k: ggsw.base2k(),
                k: res.max_k(),
                rank: res.rank(),
            });
            module.glwe_normalize_default(&mut res_conv, res, &mut scratch_2.borrow());
            module.glwe_external_product_dft(&mut res_dft, &res_conv, ggsw, key_size, &mut scratch_2);
        });
    } else {
        module.glwe_external_product_dft(&mut res_dft, res, ggsw, key_size, &mut scratch.borrow());
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
