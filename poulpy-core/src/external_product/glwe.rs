use poulpy_hal::{
    api::{
        ModuleN, ScratchAvailable, ScratchTakeBasic, VecZnxBigAddSmallAssign, VecZnxBigNormalize, VecZnxBigNormalizeTmpBytes,
        VecZnxDftAddAssign, VecZnxDftApply, VecZnxDftBytesOf, VecZnxIdftApplyConsume, VecZnxNormalize, VecZnxNormalizeTmpBytes,
        VmpApplyDftToDft, VmpApplyDftToDftTmpBytes,
    },
    layouts::{Backend, DataMut, DataViewMut, Module, Scratch, VecZnxBig, VecZnxDft, ZnxInfos, ZnxZero},
};

pub use crate::api::GLWEExternalProduct;
use crate::api::GLWEExternalProductInternal;
use crate::{
    GLWENormalize, ScratchTakeCore,
    layouts::{
        GGSWInfos, GLWE, GLWEInfos, GLWELayout, GLWEToMut, GLWEToRef, LWEInfos,
        prepared::{GGSWPrepared, GGSWPreparedToRef},
    },
};

pub(crate) trait GLWEExternalProductDefault<BE: Backend>:
    Sized
    + GLWEExternalProductInternal<BE>
    + VecZnxDftBytesOf
    + VecZnxBigNormalize<BE>
    + VecZnxBigNormalizeTmpBytes
    + VecZnxBigAddSmallAssign<BE>
    + GLWENormalize<BE>
where
    Scratch<BE>: ScratchTakeCore<BE>,
{
    fn glwe_external_product_tmp_bytes_default<R, A, B>(&self, res: &R, a: &A, ggsw: &B) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        B: GGSWInfos,
    {
        let cols: usize = res.rank().as_usize() + 1;
        let lvl_0: usize = self.bytes_of_vec_znx_dft(cols, ggsw.size());
        let lvl_1: usize = self.vec_znx_big_normalize_tmp_bytes();
        let lvl_2: usize = if a.base2k() != ggsw.base2k() {
            let a_conv_infos: GLWELayout = GLWELayout {
                n: a.n(),
                base2k: ggsw.base2k(),
                k: a.max_k(),
                rank: a.rank(),
            };
            let lvl_2_0: usize = GLWE::<Vec<u8>>::bytes_of_from_infos(&a_conv_infos);
            let lvl_2_1: usize = self
                .glwe_normalize_tmp_bytes()
                .max(self.glwe_external_product_internal_tmp_bytes(res, &a_conv_infos, ggsw));
            lvl_2_0 + lvl_2_1
        } else {
            self.glwe_external_product_internal_tmp_bytes(res, a, ggsw)
        };

        lvl_0 + lvl_1.max(lvl_2)
    }

    fn glwe_external_product_assign_default<R, D>(&self, res: &mut R, ggsw: &D, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut + GLWEInfos,
        D: GGSWPreparedToRef<BE> + GGSWInfos,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        assert_eq!(ggsw.rank(), res.rank());
        assert_eq!(ggsw.n(), res.n());
        assert!(
            scratch.available() >= self.glwe_external_product_tmp_bytes_default(res, res, ggsw),
            "scratch.available(): {} < GLWEExternalProduct::glwe_external_product_tmp_bytes: {}",
            scratch.available(),
            self.glwe_external_product_tmp_bytes_default(res, res, ggsw)
        );

        let res_base2k: usize = res.base2k().as_usize();
        let ggsw_base2k: usize = ggsw.base2k().as_usize();

        let (mut res_dft, scratch_1) = scratch.take_vec_znx_dft(self, (res.rank() + 1).into(), ggsw.size()); // Todo optimise
        res_dft.zero(); // TODO: REMOVE ONCE ABOVE TAKES CORRECT SIZE

        let res_big: VecZnxBig<&mut [u8], BE> = if res_base2k != ggsw_base2k {
            let (mut res_conv, scratch_2) = scratch_1.take_glwe(&GLWELayout {
                n: res.n(),
                base2k: ggsw.base2k(),
                k: res.max_k(),
                rank: res.rank(),
            });
            self.glwe_normalize(&mut res_conv, res, scratch_2);
            self.glwe_external_product_internal(res_dft, &res_conv, ggsw, scratch_2)
        } else {
            self.glwe_external_product_internal(res_dft, res, ggsw, scratch_1)
        };

        let res: &mut GLWE<&mut [u8]> = &mut res.to_mut();
        for j in 0..(res.rank() + 1).into() {
            self.vec_znx_big_normalize(res.data_mut(), res_base2k, 0, j, &res_big, ggsw_base2k, j, scratch_1);
        }
    }

    fn glwe_external_product_default<R, A, G>(&self, res: &mut R, a: &A, ggsw: &G, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut + GLWEInfos,
        A: GLWEToRef + GLWEInfos,
        G: GGSWPreparedToRef<BE> + GGSWInfos,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        assert_eq!(ggsw.rank(), a.rank());
        assert_eq!(ggsw.rank(), res.rank());
        assert_eq!(ggsw.n(), res.n());
        assert_eq!(a.n(), res.n());
        assert!(
            scratch.available() >= self.glwe_external_product_tmp_bytes_default(res, a, ggsw),
            "scratch.available(): {} < GLWEExternalProduct::glwe_external_product_tmp_bytes: {}",
            scratch.available(),
            self.glwe_external_product_tmp_bytes_default(res, a, ggsw)
        );

        let a_base2k: usize = a.base2k().into();
        let ggsw_base2k: usize = ggsw.base2k().into();
        let res_base2k: usize = res.base2k().into();

        let (mut res_dft, scratch_1) = scratch.take_vec_znx_dft(self, (res.rank() + 1).into(), ggsw.size()); // Todo optimise
        res_dft.zero(); // TODO: REMOVE ONCE ABOVE TAKES CORRECT SIZE

        let res_big: VecZnxBig<&mut [u8], BE> = if a_base2k != ggsw_base2k {
            let (mut a_conv, scratch_2) = scratch_1.take_glwe(&GLWELayout {
                n: a.n(),
                base2k: ggsw.base2k(),
                k: a.max_k(),
                rank: a.rank(),
            });
            self.glwe_normalize(&mut a_conv, a, scratch_2);
            self.glwe_external_product_internal(res_dft, &a_conv, ggsw, scratch_2)
        } else {
            self.glwe_external_product_internal(res_dft, a, ggsw, scratch_1)
        };

        let res: &mut GLWE<&mut [u8]> = &mut res.to_mut();
        for j in 0..(res.rank() + 1).into() {
            self.vec_znx_big_normalize(res.data_mut(), res_base2k, 0, j, &res_big, ggsw_base2k, j, scratch_1);
        }
    }
}

impl<BE: Backend> GLWEExternalProductDefault<BE> for Module<BE>
where
    Self: GLWEExternalProductInternal<BE>
        + VecZnxDftBytesOf
        + VecZnxBigNormalize<BE>
        + VecZnxBigNormalizeTmpBytes
        + VecZnxBigAddSmallAssign<BE>
        + GLWENormalize<BE>,
    Scratch<BE>: ScratchTakeCore<BE>,
{
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
        + VecZnxIdftApplyConsume<BE>
        + VecZnxBigNormalize<BE>
        + VecZnxNormalize<BE>
        + VecZnxDftBytesOf
        + VmpApplyDftToDftTmpBytes
        + VecZnxNormalizeTmpBytes,
{
    fn glwe_external_product_internal_tmp_bytes<R, A, B>(&self, res_infos: &R, a_infos: &A, b_infos: &B) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        B: GGSWInfos,
    {
        let in_size: usize = a_infos.max_k().div_ceil(b_infos.base2k()).div_ceil(b_infos.dsize().into()) as usize;
        let out_size: usize = res_infos.size();
        let ggsw_size: usize = b_infos.size();
        let cols: usize = (b_infos.rank() + 1).into();
        let lvl_0: usize = self.bytes_of_vec_znx_dft(cols, in_size);
        let lvl_1: usize = if b_infos.dsize() > 1 {
            self.bytes_of_vec_znx_dft(cols, ggsw_size)
        } else {
            0
        };
        let lvl_2: usize = self.vmp_apply_dft_to_dft_tmp_bytes(
            out_size, in_size, in_size, // rows
            cols,    // cols in
            cols,    // cols out
            ggsw_size,
        );
        lvl_0 + lvl_1 + lvl_2
    }

    fn glwe_external_product_internal<DR, A, G>(
        &self,
        mut res_dft: VecZnxDft<DR, BE>,
        a: &A,
        ggsw: &G,
        scratch: &mut Scratch<BE>,
    ) -> VecZnxBig<DR, BE>
    where
        DR: DataMut,
        A: GLWEToRef,
        G: GGSWPreparedToRef<BE>,
        Scratch<BE>: ScratchTakeCore<BE> + ScratchAvailable,
    {
        let a: &GLWE<&[u8]> = &a.to_ref();
        let ggsw: &GGSWPrepared<&[u8], BE> = &ggsw.to_ref();

        assert_eq!(a.base2k(), ggsw.base2k());
        assert!(
            scratch.available() >= self.glwe_external_product_internal_tmp_bytes(ggsw, a, ggsw),
            "scratch.available(): {} < GLWEExternalProductInternal::glwe_external_product_internal_tmp_bytes: {}",
            scratch.available(),
            self.glwe_external_product_internal_tmp_bytes(ggsw, a, ggsw)
        );

        let cols: usize = (ggsw.rank() + 1).into();
        let dsize: usize = ggsw.dsize().into();
        let a_size: usize = a.size();

        let (mut a_dft, scratch_1) = scratch.take_vec_znx_dft(self, cols, a_size.div_ceil(dsize));
        a_dft.data_mut().fill(0);

        if dsize == 1 {
            a_dft.set_size(a_size);
            res_dft.set_size(ggsw.size());
            for j in 0..cols {
                self.vec_znx_dft_apply(1, 0, &mut a_dft, j, &a.data, j);
            }
            self.vmp_apply_dft_to_dft(&mut res_dft, &a_dft, &ggsw.data, 0, scratch_1);
        } else {
            // Tmp buffer: vmp result for di > 0 before folding into res_dft.
            // Writing to a fresh sequential buffer avoids scattered-write cache thrashing.
            let (mut res_dft_tmp, scratch_2) = scratch_1.take_vec_znx_dft(self, res_dft.cols(), ggsw.size());

            for di in 0..dsize {
                // (lhs.size() + di) / dsize = (a - (digit - di - 1)).div_ceil(dsize)
                a_dft.set_size((a.size() + di) / dsize);

                // Small optimization for dsize > 2
                // VMP produce some error e, and since we aggregate vmp * 2^{di * B}, then
                // we also aggregate ei * 2^{di * B}, with the largest error being ei * 2^{(dsize-1) * B}.
                // As such we can ignore the last dsize-2 limbs safely of the sum of vmp products.
                // It is possible to further ignore the last dsize-1 limbs, but this introduce
                // ~0.5 to 1 bit of additional noise, and thus not chosen here to ensure that the same
                // noise is kept with respect to the ideal functionality.
                res_dft.set_size(ggsw.size() - ((dsize - di) as isize - 2).max(0) as usize);

                for j in 0..cols {
                    self.vec_znx_dft_apply(dsize, dsize - 1 - di, &mut a_dft, j, &a.data, j);
                }

                if di == 0 {
                    self.vmp_apply_dft_to_dft(&mut res_dft, &a_dft, &ggsw.data, 0, scratch_2);
                } else {
                    // Overwrite tmp with shifted product, then fold into res_dft.
                    res_dft_tmp.set_size(res_dft.size());
                    self.vmp_apply_dft_to_dft(&mut res_dft_tmp, &a_dft, &ggsw.data, di, scratch_2);
                    for col in 0..cols {
                        self.vec_znx_dft_add_assign(&mut res_dft, col, &res_dft_tmp, col);
                    }
                }
            }
        }

        self.vec_znx_idft_apply_consume(res_dft)
    }
}
