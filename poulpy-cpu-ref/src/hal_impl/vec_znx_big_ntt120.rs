macro_rules! hal_impl_vec_znx_big_ntt120 {
    () => {
        fn vec_znx_big_from_small<R, A>(res: &mut R, res_col: usize, a: &A, a_col: usize)
        where
            R: VecZnxBigToMut<Self>,
            A: VecZnxToRef,
        {
            <Self as NTT120VecZnxBigDefaults<Self>>::vec_znx_big_from_small_default(res, res_col, a, a_col)
        }

        fn vec_znx_big_add_normal<R>(
            module: &Module<Self>,
            res_base2k: usize,
            res: &mut R,
            res_col: usize,
            noise_infos: NoiseInfos,
            source: &mut Source,
        ) where
            R: VecZnxBigToMut<Self>,
        {
            <Self as NTT120VecZnxBigDefaults<Self>>::vec_znx_big_add_normal_default(
                module,
                res_base2k,
                res,
                res_col,
                noise_infos,
                source,
            )
        }

        fn vec_znx_big_add_into<R, A, C>(
            module: &Module<Self>,
            res: &mut R,
            res_col: usize,
            a: &A,
            a_col: usize,
            b: &C,
            b_col: usize,
        ) where
            R: VecZnxBigToMut<Self>,
            A: VecZnxBigToRef<Self>,
            C: VecZnxBigToRef<Self>,
        {
            <Self as NTT120VecZnxBigDefaults<Self>>::vec_znx_big_add_into_default(module, res, res_col, a, a_col, b, b_col)
        }

        fn vec_znx_big_add_assign<R, A>(module: &Module<Self>, res: &mut R, res_col: usize, a: &A, a_col: usize)
        where
            R: VecZnxBigToMut<Self>,
            A: VecZnxBigToRef<Self>,
        {
            <Self as NTT120VecZnxBigDefaults<Self>>::vec_znx_big_add_assign_default(module, res, res_col, a, a_col)
        }

        fn vec_znx_big_add_small_into<R, A, C>(
            module: &Module<Self>,
            res: &mut R,
            res_col: usize,
            a: &A,
            a_col: usize,
            b: &C,
            b_col: usize,
        ) where
            R: VecZnxBigToMut<Self>,
            A: VecZnxBigToRef<Self>,
            C: VecZnxToRef,
        {
            <Self as NTT120VecZnxBigDefaults<Self>>::vec_znx_big_add_small_into_default(module, res, res_col, a, a_col, b, b_col)
        }

        fn vec_znx_big_add_small_assign<R, A>(module: &Module<Self>, res: &mut R, res_col: usize, a: &A, a_col: usize)
        where
            R: VecZnxBigToMut<Self>,
            A: VecZnxToRef,
        {
            <Self as NTT120VecZnxBigDefaults<Self>>::vec_znx_big_add_small_assign_default(module, res, res_col, a, a_col)
        }

        fn vec_znx_big_sub<R, A, C>(module: &Module<Self>, res: &mut R, res_col: usize, a: &A, a_col: usize, b: &C, b_col: usize)
        where
            R: VecZnxBigToMut<Self>,
            A: VecZnxBigToRef<Self>,
            C: VecZnxBigToRef<Self>,
        {
            <Self as NTT120VecZnxBigDefaults<Self>>::vec_znx_big_sub_default(module, res, res_col, a, a_col, b, b_col)
        }

        fn vec_znx_big_sub_assign<R, A>(module: &Module<Self>, res: &mut R, res_col: usize, a: &A, a_col: usize)
        where
            R: VecZnxBigToMut<Self>,
            A: VecZnxBigToRef<Self>,
        {
            <Self as NTT120VecZnxBigDefaults<Self>>::vec_znx_big_sub_assign_default(module, res, res_col, a, a_col)
        }

        fn vec_znx_big_sub_negate_assign<R, A>(module: &Module<Self>, res: &mut R, res_col: usize, a: &A, a_col: usize)
        where
            R: VecZnxBigToMut<Self>,
            A: VecZnxBigToRef<Self>,
        {
            <Self as NTT120VecZnxBigDefaults<Self>>::vec_znx_big_sub_negate_assign_default(module, res, res_col, a, a_col)
        }

        fn vec_znx_big_sub_small_a<R, A, C>(
            module: &Module<Self>,
            res: &mut R,
            res_col: usize,
            a: &A,
            a_col: usize,
            b: &C,
            b_col: usize,
        ) where
            R: VecZnxBigToMut<Self>,
            A: VecZnxToRef,
            C: VecZnxBigToRef<Self>,
        {
            <Self as NTT120VecZnxBigDefaults<Self>>::vec_znx_big_sub_small_a_default(module, res, res_col, a, a_col, b, b_col)
        }

        fn vec_znx_big_sub_small_assign<R, A>(module: &Module<Self>, res: &mut R, res_col: usize, a: &A, a_col: usize)
        where
            R: VecZnxBigToMut<Self>,
            A: VecZnxToRef,
        {
            <Self as NTT120VecZnxBigDefaults<Self>>::vec_znx_big_sub_small_assign_default(module, res, res_col, a, a_col)
        }

        fn vec_znx_big_sub_small_b<R, A, C>(
            module: &Module<Self>,
            res: &mut R,
            res_col: usize,
            a: &A,
            a_col: usize,
            b: &C,
            b_col: usize,
        ) where
            R: VecZnxBigToMut<Self>,
            A: VecZnxBigToRef<Self>,
            C: VecZnxToRef,
        {
            <Self as NTT120VecZnxBigDefaults<Self>>::vec_znx_big_sub_small_b_default(module, res, res_col, a, a_col, b, b_col)
        }

        fn vec_znx_big_sub_small_negate_assign<R, A>(module: &Module<Self>, res: &mut R, res_col: usize, a: &A, a_col: usize)
        where
            R: VecZnxBigToMut<Self>,
            A: VecZnxToRef,
        {
            <Self as NTT120VecZnxBigDefaults<Self>>::vec_znx_big_sub_small_negate_assign_default(module, res, res_col, a, a_col)
        }

        fn vec_znx_big_negate<R, A>(module: &Module<Self>, res: &mut R, res_col: usize, a: &A, a_col: usize)
        where
            R: VecZnxBigToMut<Self>,
            A: VecZnxBigToRef<Self>,
        {
            <Self as NTT120VecZnxBigDefaults<Self>>::vec_znx_big_negate_default(module, res, res_col, a, a_col)
        }

        fn vec_znx_big_negate_assign<A>(module: &Module<Self>, a: &mut A, a_col: usize)
        where
            A: VecZnxBigToMut<Self>,
        {
            <Self as NTT120VecZnxBigDefaults<Self>>::vec_znx_big_negate_assign_default(module, a, a_col)
        }

        fn vec_znx_big_normalize_tmp_bytes(module: &Module<Self>) -> usize {
            <Self as NTT120VecZnxBigDefaults<Self>>::vec_znx_big_normalize_tmp_bytes_default(module)
        }

        fn vec_znx_big_normalize<R, A>(
            module: &Module<Self>,
            res: &mut R,
            res_base2k: usize,
            res_offset: i64,
            res_col: usize,
            a: &A,
            a_base2k: usize,
            a_col: usize,
            scratch: &mut Scratch<Self>,
        ) where
            R: VecZnxToMut,
            A: VecZnxBigToRef<Self>,
        {
            <Self as NTT120VecZnxBigDefaults<Self>>::vec_znx_big_normalize_default(
                module, res, res_base2k, res_offset, res_col, a, a_base2k, a_col, scratch,
            )
        }

        fn vec_znx_big_normalize_add_assign<R, A>(
            module: &Module<Self>,
            res: &mut R,
            res_base2k: usize,
            res_offset: i64,
            res_col: usize,
            a: &A,
            a_base2k: usize,
            a_col: usize,
            scratch: &mut Scratch<Self>,
        ) where
            R: VecZnxToMut,
            A: VecZnxBigToRef<Self>,
        {
            <Self as NTT120VecZnxBigDefaults<Self>>::vec_znx_big_normalize_add_assign_default(
                module, res, res_base2k, res_offset, res_col, a, a_base2k, a_col, scratch,
            )
        }

        fn vec_znx_big_normalize_sub_assign<R, A>(
            module: &Module<Self>,
            res: &mut R,
            res_base2k: usize,
            res_offset: i64,
            res_col: usize,
            a: &A,
            a_base2k: usize,
            a_col: usize,
            scratch: &mut Scratch<Self>,
        ) where
            R: VecZnxToMut,
            A: VecZnxBigToRef<Self>,
        {
            <Self as NTT120VecZnxBigDefaults<Self>>::vec_znx_big_normalize_sub_assign_default(
                module, res, res_base2k, res_offset, res_col, a, a_base2k, a_col, scratch,
            )
        }

        fn vec_znx_big_automorphism<R, A>(module: &Module<Self>, k: i64, res: &mut R, res_col: usize, a: &A, a_col: usize)
        where
            R: VecZnxBigToMut<Self>,
            A: VecZnxBigToRef<Self>,
        {
            <Self as NTT120VecZnxBigDefaults<Self>>::vec_znx_big_automorphism_default(module, k, res, res_col, a, a_col)
        }

        fn vec_znx_big_automorphism_assign_tmp_bytes(module: &Module<Self>) -> usize {
            <Self as NTT120VecZnxBigDefaults<Self>>::vec_znx_big_automorphism_assign_tmp_bytes_default(module)
        }

        fn vec_znx_big_automorphism_assign<A>(module: &Module<Self>, k: i64, a: &mut A, a_col: usize, scratch: &mut Scratch<Self>)
        where
            A: VecZnxBigToMut<Self>,
        {
            <Self as NTT120VecZnxBigDefaults<Self>>::vec_znx_big_automorphism_assign_default(module, k, a, a_col, scratch)
        }
    };
}
