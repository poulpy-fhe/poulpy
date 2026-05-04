macro_rules! hal_impl_vec_znx_big_ntt126_ifma {
    () => {
        fn vec_znx_big_from_small<R, A>(res: &mut R, res_col: usize, a: &A, a_col: usize)
        where
            R: VecZnxBigToMut<Self>,
            A: VecZnxToRef,
        {
            poulpy_cpu_ref::reference::ntt120::vec_znx_big::ntt120_vec_znx_big_from_small(res, res_col, a, a_col)
        }

        fn vec_znx_big_add_normal<R>(
            _module: &Module<Self>,
            res_base2k: usize,
            res: &mut R,
            res_col: usize,
            noise_infos: NoiseInfos,
            source: &mut Source,
        ) where
            R: VecZnxBigToMut<Self>,
        {
            poulpy_cpu_ref::reference::ntt120::vec_znx_big::ntt120_vec_znx_big_add_normal_ref(
                res_base2k,
                res,
                res_col,
                noise_infos,
                source,
            )
        }

        fn vec_znx_big_add_into<R, A, C>(
            _module: &Module<Self>,
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
            poulpy_cpu_ref::reference::ntt120::vec_znx_big::ntt120_vec_znx_big_add_into(res, res_col, a, a_col, b, b_col)
        }

        fn vec_znx_big_add_assign<R, A>(_module: &Module<Self>, res: &mut R, res_col: usize, a: &A, a_col: usize)
        where
            R: VecZnxBigToMut<Self>,
            A: VecZnxBigToRef<Self>,
        {
            poulpy_cpu_ref::reference::ntt120::vec_znx_big::ntt120_vec_znx_big_add_assign(res, res_col, a, a_col)
        }

        fn vec_znx_big_add_small_into<R, A, C>(
            _module: &Module<Self>,
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
            poulpy_cpu_ref::reference::ntt120::vec_znx_big::ntt120_vec_znx_big_add_small_into(res, res_col, a, a_col, b, b_col)
        }

        fn vec_znx_big_add_small_assign<R, A>(_module: &Module<Self>, res: &mut R, res_col: usize, a: &A, a_col: usize)
        where
            R: VecZnxBigToMut<Self>,
            A: VecZnxToRef,
        {
            poulpy_cpu_ref::reference::ntt120::vec_znx_big::ntt120_vec_znx_big_add_small_assign(res, res_col, a, a_col)
        }

        fn vec_znx_big_sub<R, A, C>(_module: &Module<Self>, res: &mut R, res_col: usize, a: &A, a_col: usize, b: &C, b_col: usize)
        where
            R: VecZnxBigToMut<Self>,
            A: VecZnxBigToRef<Self>,
            C: VecZnxBigToRef<Self>,
        {
            poulpy_cpu_ref::reference::ntt120::vec_znx_big::ntt120_vec_znx_big_sub(res, res_col, a, a_col, b, b_col)
        }

        fn vec_znx_big_sub_assign<R, A>(_module: &Module<Self>, res: &mut R, res_col: usize, a: &A, a_col: usize)
        where
            R: VecZnxBigToMut<Self>,
            A: VecZnxBigToRef<Self>,
        {
            poulpy_cpu_ref::reference::ntt120::vec_znx_big::ntt120_vec_znx_big_sub_assign(res, res_col, a, a_col)
        }

        fn vec_znx_big_sub_negate_assign<R, A>(_module: &Module<Self>, res: &mut R, res_col: usize, a: &A, a_col: usize)
        where
            R: VecZnxBigToMut<Self>,
            A: VecZnxBigToRef<Self>,
        {
            poulpy_cpu_ref::reference::ntt120::vec_znx_big::ntt120_vec_znx_big_sub_negate_assign(res, res_col, a, a_col)
        }

        fn vec_znx_big_sub_small_a<R, A, C>(
            _module: &Module<Self>,
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
            poulpy_cpu_ref::reference::ntt120::vec_znx_big::ntt120_vec_znx_big_sub_small_a(res, res_col, a, a_col, b, b_col)
        }

        fn vec_znx_big_sub_small_assign<R, A>(_module: &Module<Self>, res: &mut R, res_col: usize, a: &A, a_col: usize)
        where
            R: VecZnxBigToMut<Self>,
            A: VecZnxToRef,
        {
            poulpy_cpu_ref::reference::ntt120::vec_znx_big::ntt120_vec_znx_big_sub_small_assign(res, res_col, a, a_col)
        }

        fn vec_znx_big_sub_small_b<R, A, C>(
            _module: &Module<Self>,
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
            poulpy_cpu_ref::reference::ntt120::vec_znx_big::ntt120_vec_znx_big_sub_small_b(res, res_col, a, a_col, b, b_col)
        }

        fn vec_znx_big_sub_small_negate_assign<R, A>(_module: &Module<Self>, res: &mut R, res_col: usize, a: &A, a_col: usize)
        where
            R: VecZnxBigToMut<Self>,
            A: VecZnxToRef,
        {
            poulpy_cpu_ref::reference::ntt120::vec_znx_big::ntt120_vec_znx_big_sub_small_negate_assign(res, res_col, a, a_col)
        }

        fn vec_znx_big_negate<R, A>(_module: &Module<Self>, res: &mut R, res_col: usize, a: &A, a_col: usize)
        where
            R: VecZnxBigToMut<Self>,
            A: VecZnxBigToRef<Self>,
        {
            poulpy_cpu_ref::reference::ntt120::vec_znx_big::ntt120_vec_znx_big_negate(res, res_col, a, a_col)
        }

        fn vec_znx_big_negate_assign<A>(_module: &Module<Self>, a: &mut A, a_col: usize)
        where
            A: VecZnxBigToMut<Self>,
        {
            poulpy_cpu_ref::reference::ntt120::vec_znx_big::ntt120_vec_znx_big_negate_assign(a, a_col)
        }

        fn vec_znx_big_normalize_tmp_bytes(module: &Module<Self>) -> usize {
            poulpy_cpu_ref::reference::ntt120::vec_znx_big::ntt120_vec_znx_big_normalize_tmp_bytes(module.n())
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
            use poulpy_hal::api::TakeSlice;
            let (carry, _) = scratch.take_slice(
                poulpy_cpu_ref::reference::ntt120::vec_znx_big::ntt120_vec_znx_big_normalize_tmp_bytes(module.n())
                    / std::mem::size_of::<i128>(),
            );
            poulpy_cpu_ref::reference::ntt120::vec_znx_big::ntt120_vec_znx_big_normalize(
                res, res_base2k, res_offset, res_col, a, a_base2k, a_col, carry,
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
            use poulpy_hal::api::TakeSlice;
            let (carry, _) = scratch.take_slice(
                poulpy_cpu_ref::reference::ntt120::vec_znx_big::ntt120_vec_znx_big_normalize_tmp_bytes(module.n())
                    / std::mem::size_of::<i128>(),
            );
            poulpy_cpu_ref::reference::ntt120::vec_znx_big::ntt120_vec_znx_big_normalize_add_assign(
                res, res_base2k, res_offset, res_col, a, a_base2k, a_col, carry,
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
            use poulpy_hal::api::TakeSlice;
            let (carry, _) = scratch.take_slice(
                poulpy_cpu_ref::reference::ntt120::vec_znx_big::ntt120_vec_znx_big_normalize_tmp_bytes(module.n())
                    / std::mem::size_of::<i128>(),
            );
            poulpy_cpu_ref::reference::ntt120::vec_znx_big::ntt120_vec_znx_big_normalize_sub_assign(
                res, res_base2k, res_offset, res_col, a, a_base2k, a_col, carry,
            )
        }

        fn vec_znx_big_automorphism<R, A>(_module: &Module<Self>, k: i64, res: &mut R, res_col: usize, a: &A, a_col: usize)
        where
            R: VecZnxBigToMut<Self>,
            A: VecZnxBigToRef<Self>,
        {
            poulpy_cpu_ref::reference::ntt120::vec_znx_big::ntt120_vec_znx_big_automorphism(k, res, res_col, a, a_col)
        }

        fn vec_znx_big_automorphism_assign_tmp_bytes(module: &Module<Self>) -> usize {
            poulpy_cpu_ref::reference::ntt120::vec_znx_big::ntt120_vec_znx_big_automorphism_assign_tmp_bytes(module.n())
        }

        fn vec_znx_big_automorphism_assign<A>(module: &Module<Self>, k: i64, a: &mut A, a_col: usize, scratch: &mut Scratch<Self>)
        where
            A: VecZnxBigToMut<Self>,
        {
            use poulpy_hal::api::TakeSlice;
            let (tmp, _) = scratch.take_slice(
                poulpy_cpu_ref::reference::ntt120::vec_znx_big::ntt120_vec_znx_big_automorphism_assign_tmp_bytes(module.n())
                    / std::mem::size_of::<i128>(),
            );
            poulpy_cpu_ref::reference::ntt120::vec_znx_big::ntt120_vec_znx_big_automorphism_assign(k, a, a_col, tmp)
        }
    };
}
