use poulpy_hal::{
    api::{
        ModuleN, ScratchAvailable, ScratchTakeBasic, VecZnxBigAddSmallAssign, VecZnxBigNormalize, VecZnxBigNormalizeTmpBytes,
        VecZnxDftAddAssign, VecZnxDftApply, VecZnxDftBytesOf, VecZnxDftCopy, VecZnxIdftApplyConsume, VecZnxNormalize,
        VecZnxNormalizeTmpBytes, VmpApplyDftToDft, VmpApplyDftToDftTmpBytes,
    },
    layouts::{Backend, DataMut, Module, Scratch, VecZnxBig, VecZnxDft, VecZnxDftToRef, VmpPMat, ZnxInfos, ZnxZero},
};

pub use crate::api::GLWEKeyswitch;
use crate::{
    GLWENormalize, ScratchTakeCore,
    layouts::{GGLWEInfos, GGLWEPrepared, GGLWEPreparedToRef, GLWE, GLWEInfos, GLWELayout, GLWEToMut, GLWEToRef, LWEInfos},
};

pub(crate) trait GLWEKeyswitchDefault<BE: Backend>:
    Sized + GLWEKeySwitchInternal<BE> + VecZnxBigNormalizeTmpBytes + VecZnxBigNormalize<BE> + GLWENormalize<BE>
where
    Scratch<BE>: ScratchTakeCore<BE>,
{
    fn glwe_keyswitch_tmp_bytes_default<R, A, B>(&self, res_infos: &R, a_infos: &A, key_infos: &B) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        B: GGLWEInfos,
    {
        assert_eq!(self.n() as u32, res_infos.n());
        assert_eq!(self.n() as u32, a_infos.n());
        assert_eq!(self.n() as u32, key_infos.n());

        let cols: usize = res_infos.rank().as_usize() + 1;
        let lvl_0: usize = self.bytes_of_vec_znx_dft(cols, key_infos.size());
        let lvl_1: usize = self.vec_znx_big_normalize_tmp_bytes();
        let lvl_2: usize = if a_infos.base2k() != key_infos.base2k() {
            let a_conv_infos: GLWELayout = GLWELayout {
                n: a_infos.n(),
                base2k: key_infos.base2k(),
                k: a_infos.max_k(),
                rank: a_infos.rank(),
            };
            let lvl_2_0: usize = GLWE::<Vec<u8>>::bytes_of_from_infos(&a_conv_infos);
            let lvl_2_1: usize =
                self.glwe_normalize_tmp_bytes()
                    .max(self.glwe_keyswitch_internal_tmp_bytes(res_infos, &a_conv_infos, key_infos));
            lvl_2_0 + lvl_2_1
        } else {
            self.glwe_keyswitch_internal_tmp_bytes(res_infos, a_infos, key_infos)
        };

        lvl_0 + lvl_1.max(lvl_2)
    }

    fn glwe_keyswitch_default<R, A, K>(&self, res: &mut R, a: &A, key: &K, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut + GLWEInfos,
        A: GLWEToRef + GLWEInfos,
        K: GGLWEPreparedToRef<BE> + GGLWEInfos,
    {
        assert_eq!(
            a.rank(),
            key.rank_in(),
            "a.rank(): {} != b.rank_in(): {}",
            a.rank(),
            key.rank_in()
        );
        assert_eq!(
            res.rank(),
            key.rank_out(),
            "res.rank(): {} != b.rank_out(): {}",
            res.rank(),
            key.rank_out()
        );

        assert_eq!(res.n(), self.n() as u32);
        assert_eq!(a.n(), self.n() as u32);
        assert_eq!(key.n(), self.n() as u32);

        assert!(
            scratch.available() >= self.glwe_keyswitch_tmp_bytes_default(res, a, key),
            "scratch.available(): {} < GLWEKeyswitch::glwe_keyswitch_tmp_bytes: {}",
            scratch.available(),
            self.glwe_keyswitch_tmp_bytes_default(res, a, key)
        );

        let a_base2k: usize = a.base2k().into();
        let key_base2k: usize = key.base2k().into();
        let res_base2k: usize = res.base2k().into();

        let (mut res_dft, scratch_1) = scratch.take_vec_znx_dft(self, (res.rank() + 1).into(), key.size()); // Todo optimise
        res_dft.zero(); // TODO: remove once the above has the correct size

        let res_big: VecZnxBig<&mut [u8], BE> = if a_base2k != key_base2k {
            let (mut a_conv, scratch_2) = scratch_1.take_glwe(&GLWELayout {
                n: a.n(),
                base2k: key.base2k(),
                k: a.max_k(),
                rank: a.rank(),
            });
            self.glwe_normalize(&mut a_conv, a, scratch_2);
            self.glwe_keyswitch_internal(res_dft, &a_conv, key, scratch_2)
        } else {
            self.glwe_keyswitch_internal(res_dft, a, key, scratch_1)
        };

        let res: &mut GLWE<&mut [u8]> = &mut res.to_mut();
        for i in 0..(res.rank() + 1).into() {
            self.vec_znx_big_normalize(res.data_mut(), res_base2k, 0, i, &res_big, key_base2k, i, scratch_1);
        }
    }

    fn glwe_keyswitch_assign_default<R, K>(&self, res: &mut R, key: &K, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut + GLWEInfos,
        K: GGLWEPreparedToRef<BE> + GGLWEInfos,
    {
        assert_eq!(
            res.rank(),
            key.rank_in(),
            "res.rank(): {} != a.rank_in(): {}",
            res.rank(),
            key.rank_in()
        );
        assert_eq!(
            res.rank(),
            key.rank_out(),
            "res.rank(): {} != b.rank_out(): {}",
            res.rank(),
            key.rank_out()
        );

        assert_eq!(res.n(), self.n() as u32);
        assert_eq!(key.n(), self.n() as u32);

        assert!(
            scratch.available() >= self.glwe_keyswitch_tmp_bytes_default(res, res, key),
            "scratch.available(): {} < GLWEKeyswitch::glwe_keyswitch_tmp_bytes: {}",
            scratch.available(),
            self.glwe_keyswitch_tmp_bytes_default(res, res, key)
        );

        let res_base2k: usize = res.base2k().as_usize();
        let key_base2k: usize = key.base2k().as_usize();

        let (mut res_dft, scratch_1) = scratch.take_vec_znx_dft(self, (res.rank() + 1).into(), key.size()); // TODO: optimise
        res_dft.zero(); // TODO: remove once the above has the correct size

        let res_big: VecZnxBig<&mut [u8], BE> = if res_base2k != key_base2k {
            let (mut res_conv, scratch_2) = scratch_1.take_glwe(&GLWELayout {
                n: res.n(),
                base2k: key.base2k(),
                k: res.max_k(),
                rank: res.rank(),
            });
            self.glwe_normalize(&mut res_conv, res, scratch_2);

            self.glwe_keyswitch_internal(res_dft, &res_conv, key, scratch_2)
        } else {
            self.glwe_keyswitch_internal(res_dft, res, key, scratch_1)
        };

        let res: &mut GLWE<&mut [u8]> = &mut res.to_mut();
        for i in 0..(res.rank() + 1).into() {
            self.vec_znx_big_normalize(res.data_mut(), res_base2k, 0, i, &res_big, key_base2k, i, scratch_1);
        }
    }
}

impl<BE: Backend> GLWEKeyswitchDefault<BE> for Module<BE>
where
    Self: Sized + GLWEKeySwitchInternal<BE> + VecZnxBigNormalizeTmpBytes + VecZnxBigNormalize<BE> + GLWENormalize<BE>,
    Scratch<BE>: ScratchTakeCore<BE>,
{
}

impl<BE: Backend> GLWEKeySwitchInternal<BE> for Module<BE> where
    Self: GGLWEProduct<BE>
        + VecZnxDftApply<BE>
        + VecZnxNormalize<BE>
        + VecZnxIdftApplyConsume<BE>
        + VecZnxBigAddSmallAssign<BE>
        + VecZnxNormalizeTmpBytes
{
}

pub(crate) trait GLWEKeySwitchInternal<BE: Backend>
where
    Self: GGLWEProduct<BE>
        + VecZnxDftApply<BE>
        + VecZnxNormalize<BE>
        + VecZnxIdftApplyConsume<BE>
        + VecZnxBigAddSmallAssign<BE>
        + VecZnxNormalizeTmpBytes,
{
    fn glwe_keyswitch_internal_tmp_bytes<R, A, K>(&self, res_infos: &R, a_infos: &A, key_infos: &K) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        K: GGLWEInfos,
    {
        let cols: usize = (a_infos.rank() + 1).into();
        let a_size: usize = a_infos.size();
        let lvl_0: usize = self.bytes_of_vec_znx_dft(cols - 1, a_size);
        let lvl_1: usize = self.gglwe_product_dft_tmp_bytes(res_infos.size(), a_size, key_infos);
        lvl_0 + lvl_1
    }

    fn glwe_keyswitch_internal<DR, A, K>(
        &self,
        mut res: VecZnxDft<DR, BE>,
        a: &A,
        key: &K,
        scratch: &mut Scratch<BE>,
    ) -> VecZnxBig<DR, BE>
    where
        DR: DataMut,
        A: GLWEToRef,
        K: GGLWEPreparedToRef<BE>,
        Scratch<BE>: ScratchTakeCore<BE> + ScratchAvailable,
    {
        let a: &GLWE<&[u8]> = &a.to_ref();
        let key: &GGLWEPrepared<&[u8], BE> = &key.to_ref();
        assert_eq!(a.base2k(), key.base2k());
        assert!(
            scratch.available() >= self.glwe_keyswitch_internal_tmp_bytes(key, a, key),
            "scratch.available(): {} < GLWEKeySwitchInternal::glwe_keyswitch_internal_tmp_bytes: {}",
            scratch.available(),
            self.glwe_keyswitch_internal_tmp_bytes(key, a, key)
        );
        let cols: usize = (a.rank() + 1).into();
        let a_size: usize = a.size();
        let (mut a_dft, scratch_1) = scratch.take_vec_znx_dft(self, cols - 1, a_size);
        for col_i in 0..cols - 1 {
            self.vec_znx_dft_apply(1, 0, &mut a_dft, col_i, a.data(), col_i + 1);
        }
        self.gglwe_product_dft(&mut res, &a_dft, key, scratch_1);
        let mut res_big: VecZnxBig<DR, BE> = self.vec_znx_idft_apply_consume(res);
        self.vec_znx_big_add_small_assign(&mut res_big, 0, a.data(), 0);
        res_big
    }
}

impl<BE: Backend> GGLWEProduct<BE> for Module<BE> where
    Self: Sized
        + ModuleN
        + VecZnxDftBytesOf
        + VmpApplyDftToDftTmpBytes
        + VmpApplyDftToDft<BE>
        + VecZnxDftAddAssign<BE>
        + VecZnxDftCopy<BE>
{
}

pub(crate) trait GGLWEProduct<BE: Backend>
where
    Self: Sized
        + ModuleN
        + VecZnxDftBytesOf
        + VmpApplyDftToDftTmpBytes
        + VmpApplyDftToDft<BE>
        + VecZnxDftAddAssign<BE>
        + VecZnxDftCopy<BE>,
{
    fn gglwe_product_dft_tmp_bytes<K>(&self, res_size: usize, a_size: usize, key_infos: &K) -> usize
    where
        K: GGLWEInfos,
    {
        let dsize: usize = key_infos.dsize().as_usize();

        if dsize == 1 {
            let lvl_0: usize = self.vmp_apply_dft_to_dft_tmp_bytes(
                res_size,
                a_size,
                key_infos.dnum().into(),
                (key_infos.rank_in()).into(),
                (key_infos.rank_out() + 1).into(),
                key_infos.size(),
            );
            lvl_0
        } else {
            let dnum: usize = key_infos.dnum().into();
            let a_size: usize = a_size.div_ceil(dsize).min(dnum);
            let cols_out: usize = (key_infos.rank_out() + 1).into();
            let lvl_0: usize = self.bytes_of_vec_znx_dft(key_infos.rank_in().into(), a_size);
            let lvl_1: usize = self.bytes_of_vec_znx_dft(cols_out, key_infos.size());
            let lvl_2: usize = self.vmp_apply_dft_to_dft_tmp_bytes(
                res_size,
                a_size,
                dnum,
                (key_infos.rank_in()).into(),
                (key_infos.rank_out() + 1).into(),
                key_infos.size(),
            );

            lvl_0 + lvl_1 + lvl_2
        }
    }

    fn gglwe_product_dft<DR, A, K>(&self, res: &mut VecZnxDft<DR, BE>, a: &A, key: &K, scratch: &mut Scratch<BE>)
    where
        DR: DataMut,
        A: VecZnxDftToRef<BE>,
        K: GGLWEPreparedToRef<BE> + GGLWEInfos,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        let a: &VecZnxDft<&[u8], BE> = &a.to_ref();
        let key: &GGLWEPrepared<&[u8], BE> = &key.to_ref();

        let cols: usize = a.cols();
        let a_size: usize = a.size();
        let pmat: &VmpPMat<&[u8], BE> = &key.data;
        assert!(
            scratch.available() >= self.gglwe_product_dft_tmp_bytes(res.size(), a_size, key),
            "scratch.available(): {} < GGLWEProduct::gglwe_product_dft_tmp_bytes: {}",
            scratch.available(),
            self.gglwe_product_dft_tmp_bytes(res.size(), a_size, key)
        );

        // If dsize == 1, then the digit decomposition is equal to Base2K and we can simply
        // can the vmp API.
        if key.dsize() == 1 {
            self.vmp_apply_dft_to_dft(res, a, pmat, 0, scratch);
        // If dsize != 1, then the digit decomposition is k * Base2K with k > 1.
        // As such we need to perform a bivariate polynomial convolution in (X, Y) / (X^{N}+1) with Y = 2^-K
        // (instead of yn univariate one in X).
        //
        // Since the basis in Y is small (in practice degree 6-7 max), we perform it naiveley.
        // To do so, we group the different limbs of ai_dft by their respective degree in Y
        // which are multiples of the current digit.
        // For example if dsize = 3, with ai_dft = [a0, a1, a2, a3, a4, a5, a6],
        // we group them as [[a0, a3, a5], [a1, a4, a6], [a2, a5, 0]]
        // and evaluate sum(a_di * pmat * 2^{di*Base2k})
        } else {
            let dsize: usize = key.dsize().into();
            let dnum: usize = key.dnum().into();
            let cols_out: usize = res.cols();

            // We bound ai_dft size by the number of rows of the matrix
            let (mut ai_dft, scratch_1) = scratch.take_vec_znx_dft(self, cols, a_size.div_ceil(dsize).min(dnum));
            ai_dft.zero();

            // Tmp buffer: vmp result for di > 0 before folding into res.
            // Writing to a fresh sequential buffer avoids scattered-write cache thrashing.
            let (mut res_dft_tmp, scratch_2) = scratch_1.take_vec_znx_dft(self, cols_out, pmat.size());
            res_dft_tmp.zero();

            for di in 0..dsize {
                // Sets ai_dft size according to the current digit (if dsize does not divides a_size),
                // bounded by the number of rows (digits) in the prepared matrix.
                ai_dft.set_size(((a_size + di) / dsize).min(dnum));

                // Small optimization for dsize > 2
                // VMP produce some error e, and since we aggregate vmp * 2^{di * Base2k}, then
                // we also aggregate ei * 2^{di * Base2k}, with the largest error being ei * 2^{(dsize-1) * Base2k}.
                // As such we can ignore the last dsize-2 limbs safely of the sum of vmp products.
                // It is possible to further ignore the last dsize-1 limbs, but this introduce
                // ~0.5 to 1 bit of additional noise, and thus not chosen here to ensure that the same
                // noise is kept with respect to the ideal functionality.
                res.set_size(pmat.size() - ((dsize - di) as isize - 2).max(0) as usize);

                for j in 0..cols {
                    self.vec_znx_dft_copy(dsize, dsize - di - 1, &mut ai_dft, j, a, j);
                }

                if di == 0 {
                    // res = pmat * ai_dft
                    self.vmp_apply_dft_to_dft(res, &ai_dft, pmat, 0, scratch_2);
                } else {
                    // Overwrite tmp with shifted product, then fold into res.
                    // This avoids scattered read-add-write on the res DFT buffer.
                    res_dft_tmp.set_size(res.size());
                    self.vmp_apply_dft_to_dft(&mut res_dft_tmp, &ai_dft, pmat, di, scratch_2);
                    for col in 0..cols_out {
                        self.vec_znx_dft_add_assign(res, col, &res_dft_tmp, col);
                    }
                }
            }

            res.set_size(res.max_size());
        }
    }
}
