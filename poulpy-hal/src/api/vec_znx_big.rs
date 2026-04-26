use crate::{
    layouts::{
        Backend, NoiseInfos, Scratch, VecZnx, VecZnxBigOwned, VecZnxBigToMut, VecZnxBigToRef, VecZnxToMut, VecZnxToRef, ZnxView,
        ZnxViewMut,
    },
    source::Source,
};

/// Converts a coefficient-domain [`VecZnx`](crate::layouts::VecZnx) column
/// into a [`VecZnxBig`](crate::layouts::VecZnxBig) column.
pub trait VecZnxBigFromSmall<B: Backend> {
    fn vec_znx_big_from_small<R, A>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxBigToMut<B>,
        A: VecZnxToRef;
}

/// Allocates as [crate::layouts::VecZnxBig].
pub trait VecZnxBigAlloc<B: Backend> {
    fn vec_znx_big_alloc(&self, cols: usize, size: usize) -> VecZnxBigOwned<B>;
}

/// Returns the size in bytes to allocate a [crate::layouts::VecZnxBig].
pub trait VecZnxBigBytesOf {
    fn bytes_of_vec_znx_big(&self, cols: usize, size: usize) -> usize;
}

/// Consume a vector of bytes into a [crate::layouts::VecZnxBig].
/// User must ensure that bytes is memory aligned and that its length is equal to [VecZnxBigBytesOf::bytes_of_vec_znx_big].
pub trait VecZnxBigFromBytes<B: Backend> {
    fn vec_znx_big_from_bytes(&self, cols: usize, size: usize, bytes: Vec<u8>) -> VecZnxBigOwned<B>;
}

#[allow(clippy::too_many_arguments)]
/// Add a discrete normal distribution on res.
///
/// # Arguments
/// * `base2k`: base two logarithm of the bivariate representation
/// * `res`: receiver.
/// * `res_col`: column of the receiver on which the operation is performed/stored.
/// * `k`:
/// * `source`: random coin source.
/// * `sigma`: standard deviation of the discrete normal distribution.
/// * `bound`: rejection sampling bound.
pub trait VecZnxBigAddNormal<B: Backend> {
    fn vec_znx_big_add_normal<R: VecZnxBigToMut<B>>(
        &self,
        base2k: usize,
        res: &mut R,
        res_col: usize,
        noise_infos: NoiseInfos,
        source: &mut Source,
    );
}

pub trait VecZnxBigAddInto<B: Backend> {
    /// Adds `a` to `b` and stores the result on `c`.
    fn vec_znx_big_add_into<R, A, C>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize, b: &C, b_col: usize)
    where
        R: VecZnxBigToMut<B>,
        A: VecZnxBigToRef<B>,
        C: VecZnxBigToRef<B>;
}

pub trait VecZnxBigAddAssign<B: Backend> {
    /// Adds `a` to `b` and stores the result on `b`.
    fn vec_znx_big_add_assign<R, A>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxBigToMut<B>,
        A: VecZnxBigToRef<B>;
}

pub trait VecZnxBigAddSmallInto<B: Backend> {
    /// Adds `a` to `b` and stores the result on `c`.
    fn vec_znx_big_add_small_into<R, A, C>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize, b: &C, b_col: usize)
    where
        R: VecZnxBigToMut<B>,
        A: VecZnxBigToRef<B>,
        C: VecZnxToRef;
}

pub trait VecZnxBigAddSmallAssign<B: Backend> {
    /// Adds `a` to `b` and stores the result on `b`.
    fn vec_znx_big_add_small_assign<R, A>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxBigToMut<B>,
        A: VecZnxToRef;
}

pub trait VecZnxBigSub<B: Backend> {
    /// Subtracts `a` to `b` and stores the result on `c`.
    fn vec_znx_big_sub<R, A, C>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize, b: &C, b_col: usize)
    where
        R: VecZnxBigToMut<B>,
        A: VecZnxBigToRef<B>,
        C: VecZnxBigToRef<B>;
}

pub trait VecZnxBigSubAssign<B: Backend> {
    /// Subtracts `a` from `b` and stores the result on `b`.
    fn vec_znx_big_sub_assign<R, A>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxBigToMut<B>,
        A: VecZnxBigToRef<B>;
}

pub trait VecZnxBigSubNegateAssign<B: Backend> {
    /// Subtracts `b` from `a` and stores the result on `b`.
    fn vec_znx_big_sub_negate_assign<R, A>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxBigToMut<B>,
        A: VecZnxBigToRef<B>;
}

pub trait VecZnxBigSubSmallA<B: Backend> {
    /// Subtracts `b` from `a` and stores the result on `c`.
    fn vec_znx_big_sub_small_a<R, A, C>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize, b: &C, b_col: usize)
    where
        R: VecZnxBigToMut<B>,
        A: VecZnxToRef,
        C: VecZnxBigToRef<B>;
}

pub trait VecZnxBigSubSmallAssign<B: Backend> {
    /// Subtracts `a` from `res` and stores the result on `res`.
    fn vec_znx_big_sub_small_assign<R, A>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxBigToMut<B>,
        A: VecZnxToRef;
}

pub trait VecZnxBigSubSmallB<B: Backend> {
    /// Subtracts `b` from `a` and stores the result on `c`.
    fn vec_znx_big_sub_small_b<R, A, C>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize, b: &C, b_col: usize)
    where
        R: VecZnxBigToMut<B>,
        A: VecZnxBigToRef<B>,
        C: VecZnxToRef;
}

pub trait VecZnxBigSubSmallNegateAssign<B: Backend> {
    /// Subtracts `res` from `a` and stores the result on `res`.
    fn vec_znx_big_sub_small_negate_assign<R, A>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxBigToMut<B>,
        A: VecZnxToRef;
}

/// Negates the selected column of `a` and stores the result in `res`.
pub trait VecZnxBigNegate<B: Backend> {
    fn vec_znx_big_negate<R, A>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxBigToMut<B>,
        A: VecZnxBigToRef<B>;
}

/// Negates the selected column of `res` in-place.
pub trait VecZnxBigNegateAssign<B: Backend> {
    fn vec_znx_big_negate_assign<R>(&self, res: &mut R, res_col: usize)
    where
        R: VecZnxBigToMut<B>;
}

/// Returns scratch bytes required for [`VecZnxBigNormalize`].
pub trait VecZnxBigNormalizeTmpBytes {
    fn vec_znx_big_normalize_tmp_bytes(&self) -> usize;
}

#[allow(clippy::too_many_arguments)]
/// Normalizes a [`VecZnxBig`](crate::layouts::VecZnxBig) into a coefficient-domain
/// [`VecZnx`](crate::layouts::VecZnx) with the target base and offset.
pub trait VecZnxBigNormalize<B: Backend> {
    #[allow(clippy::too_many_arguments)]
    fn vec_znx_big_normalize_into<R, A>(
        &self,
        res: &mut R,
        res_base2k: usize,
        res_offset: i64,
        res_col: usize,
        a: &A,
        a_base2k: usize,
        a_col: usize,
        scratch: &mut Scratch<B>,
    ) where
        R: VecZnxToMut,
        A: VecZnxBigToRef<B>,
    {
        self.vec_znx_big_normalize(res, res_base2k, res_offset, res_col, a, a_base2k, a_col, scratch);
    }

    fn vec_znx_big_normalize<R, A>(
        &self,
        res: &mut R,
        res_base2k: usize,
        res_offset: i64,
        res_col: usize,
        a: &A,
        a_base2k: usize,
        a_col: usize,
        scratch: &mut Scratch<B>,
    ) where
        R: VecZnxToMut,
        A: VecZnxBigToRef<B>;

    #[allow(clippy::too_many_arguments)]
    fn vec_znx_big_normalize_add_assign<R, A>(
        &self,
        res: &mut R,
        res_base2k: usize,
        res_offset: i64,
        res_col: usize,
        a: &A,
        a_base2k: usize,
        a_col: usize,
        scratch: &mut Scratch<B>,
    ) where
        R: VecZnxToMut,
        A: VecZnxBigToRef<B>,
    {
        // TODO: Override in backends to normalize directly into `res` without a temporary.
        let (n, size) = {
            let res_ref = res.to_mut();
            (res_ref.n, res_ref.size)
        };

        let mut tmp = VecZnx::alloc(n, 1, size);
        self.vec_znx_big_normalize(&mut tmp, res_base2k, res_offset, 0, a, a_base2k, a_col, scratch);

        let mut res_ref = res.to_mut();
        for j in 0..size {
            for (ri, ti) in res_ref.at_mut(res_col, j).iter_mut().zip(tmp.at(0, j).iter()) {
                *ri = ri.wrapping_add(*ti);
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn vec_znx_big_normalize_sub_assign<R, A>(
        &self,
        res: &mut R,
        res_base2k: usize,
        res_offset: i64,
        res_col: usize,
        a: &A,
        a_base2k: usize,
        a_col: usize,
        scratch: &mut Scratch<B>,
    ) where
        R: VecZnxToMut,
        A: VecZnxBigToRef<B>,
    {
        // TODO: Override in backends to normalize directly into `res` without a temporary.
        let (n, size) = {
            let res_ref = res.to_mut();
            (res_ref.n, res_ref.size)
        };

        let mut tmp = VecZnx::alloc(n, 1, size);
        self.vec_znx_big_normalize(&mut tmp, res_base2k, res_offset, 0, a, a_base2k, a_col, scratch);

        let mut res_ref = res.to_mut();
        for j in 0..size {
            for (ri, ti) in res_ref.at_mut(res_col, j).iter_mut().zip(tmp.at(0, j).iter()) {
                *ri = ri.wrapping_sub(*ti);
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn vec_znx_big_normalize_negate<R, A>(
        &self,
        res: &mut R,
        res_base2k: usize,
        res_offset: i64,
        res_col: usize,
        a: &A,
        a_base2k: usize,
        a_col: usize,
        scratch: &mut Scratch<B>,
    ) where
        R: VecZnxToMut,
        A: VecZnxBigToRef<B>,
    {
        self.vec_znx_big_normalize(res, res_base2k, res_offset, res_col, a, a_base2k, a_col, scratch);
        let mut res_ref = res.to_mut();
        for j in 0..res_ref.size {
            for ri in res_ref.at_mut(res_col, j) {
                *ri = ri.wrapping_neg();
            }
        }
    }
}

/// Returns scratch bytes required for in-place automorphism on [`VecZnxBig`](crate::layouts::VecZnxBig).
pub trait VecZnxBigAutomorphismAssignTmpBytes {
    fn vec_znx_big_automorphism_assign_tmp_bytes(&self) -> usize;
}

pub trait VecZnxBigAutomorphism<B: Backend> {
    /// Applies the automorphism X^i -> X^ik on `a` and stores the result on `b`.
    fn vec_znx_big_automorphism<R, A>(&self, p: i64, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxBigToMut<B>,
        A: VecZnxBigToRef<B>;
}

pub trait VecZnxBigAutomorphismAssign<B: Backend> {
    /// Applies the automorphism X^i -> X^ik on `a` and stores the result on `a`.
    fn vec_znx_big_automorphism_assign<R>(&self, p: i64, res: &mut R, res_col: usize, scratch: &mut Scratch<B>)
    where
        R: VecZnxBigToMut<B>;
}
