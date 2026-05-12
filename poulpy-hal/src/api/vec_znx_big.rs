use crate::{
    layouts::{
        Backend, NoiseInfos, ScratchArena, VecZnxBackendMut, VecZnxBackendRef, VecZnxBigBackendMut, VecZnxBigBackendRef,
        VecZnxBigOwned,
    },
    source::Source,
};

/// Converts a coefficient-domain [`VecZnx`](crate::layouts::VecZnx) column
/// into a [`VecZnxBig`](crate::layouts::VecZnxBig) column.
pub trait VecZnxBigFromSmallBackend<B: Backend> {
    fn vec_znx_big_from_small_backend(
        &self,
        res: &mut VecZnxBigBackendMut<'_, B>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, B>,
        a_col: usize,
    );
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
    fn vec_znx_big_add_normal(
        &self,
        base2k: usize,
        res: &mut VecZnxBigBackendMut<'_, B>,
        res_col: usize,
        noise_infos: NoiseInfos,
        source: &mut Source,
    );
}

#[allow(clippy::too_many_arguments)]
pub trait VecZnxBigAddNormalBackend<B: Backend> {
    fn vec_znx_big_add_normal_backend(
        &self,
        base2k: usize,
        res: &mut VecZnxBigBackendMut<'_, B>,
        res_col: usize,
        noise_infos: NoiseInfos,
        seed: [u8; 32],
    );
}

pub trait VecZnxBigAddInto<B: Backend> {
    /// Adds `a` to `b` and stores the result on `c`.
    fn vec_znx_big_add_into(
        &self,
        res: &mut VecZnxBigBackendMut<'_, B>,
        res_col: usize,
        a: &VecZnxBigBackendRef<'_, B>,
        a_col: usize,
        b: &VecZnxBigBackendRef<'_, B>,
        b_col: usize,
    );
}

pub trait VecZnxBigAddAssign<B: Backend> {
    /// Adds `a` to `b` and stores the result on `b`.
    fn vec_znx_big_add_assign(
        &self,
        res: &mut VecZnxBigBackendMut<'_, B>,
        res_col: usize,
        a: &VecZnxBigBackendRef<'_, B>,
        a_col: usize,
    );
}

pub trait VecZnxBigAddSmallIntoBackend<B: Backend> {
    /// Adds `a` to `b` and stores the result on `c`.
    fn vec_znx_big_add_small_into_backend(
        &self,
        res: &mut VecZnxBigBackendMut<'_, B>,
        res_col: usize,
        a: &VecZnxBigBackendRef<'_, B>,
        a_col: usize,
        b: &VecZnxBackendRef<'_, B>,
        b_col: usize,
    );
}

pub trait VecZnxBigAddSmallAssign<B: Backend> {
    /// Adds `a` to `b` and stores the result on `b`.
    fn vec_znx_big_add_small_assign<'r, 'a>(
        &self,
        res: &mut VecZnxBigBackendMut<'r, B>,
        res_col: usize,
        a: &VecZnxBackendRef<'a, B>,
        a_col: usize,
    );
}

pub trait VecZnxBigSub<B: Backend> {
    /// Subtracts `a` to `b` and stores the result on `c`.
    fn vec_znx_big_sub(
        &self,
        res: &mut VecZnxBigBackendMut<'_, B>,
        res_col: usize,
        a: &VecZnxBigBackendRef<'_, B>,
        a_col: usize,
        b: &VecZnxBigBackendRef<'_, B>,
        b_col: usize,
    );
}

pub trait VecZnxBigSubAssign<B: Backend> {
    /// Subtracts `a` from `b` and stores the result on `b`.
    fn vec_znx_big_sub_assign(
        &self,
        res: &mut VecZnxBigBackendMut<'_, B>,
        res_col: usize,
        a: &VecZnxBigBackendRef<'_, B>,
        a_col: usize,
    );
}

pub trait VecZnxBigSubNegateAssign<B: Backend> {
    /// Subtracts `b` from `a` and stores the result on `b`.
    fn vec_znx_big_sub_negate_assign(
        &self,
        res: &mut VecZnxBigBackendMut<'_, B>,
        res_col: usize,
        a: &VecZnxBigBackendRef<'_, B>,
        a_col: usize,
    );
}

pub trait VecZnxBigSubSmallABackend<B: Backend> {
    /// Subtracts `b` from `a` and stores the result on `c`.
    fn vec_znx_big_sub_small_a_backend(
        &self,
        res: &mut VecZnxBigBackendMut<'_, B>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, B>,
        a_col: usize,
        b: &VecZnxBigBackendRef<'_, B>,
        b_col: usize,
    );
}

pub trait VecZnxBigSubSmallAssign<B: Backend> {
    /// Subtracts `a` from `res` and stores the result on `res`.
    fn vec_znx_big_sub_small_assign<'r, 'a>(
        &self,
        res: &mut VecZnxBigBackendMut<'r, B>,
        res_col: usize,
        a: &VecZnxBackendRef<'a, B>,
        a_col: usize,
    );
}

pub trait VecZnxBigSubSmallBBackend<B: Backend> {
    /// Subtracts `b` from `a` and stores the result on `c`.
    fn vec_znx_big_sub_small_b_backend(
        &self,
        res: &mut VecZnxBigBackendMut<'_, B>,
        res_col: usize,
        a: &VecZnxBigBackendRef<'_, B>,
        a_col: usize,
        b: &VecZnxBackendRef<'_, B>,
        b_col: usize,
    );
}

pub trait VecZnxBigSubSmallNegateAssign<B: Backend> {
    /// Subtracts `res` from `a` and stores the result on `res`.
    fn vec_znx_big_sub_small_negate_assign<'r, 'a>(
        &self,
        res: &mut VecZnxBigBackendMut<'r, B>,
        res_col: usize,
        a: &VecZnxBackendRef<'a, B>,
        a_col: usize,
    );
}

/// Negates the selected column of `a` and stores the result in `res`.
pub trait VecZnxBigNegate<B: Backend> {
    fn vec_znx_big_negate(
        &self,
        res: &mut VecZnxBigBackendMut<'_, B>,
        res_col: usize,
        a: &VecZnxBigBackendRef<'_, B>,
        a_col: usize,
    );
}

/// Negates the selected column of `res` in-place.
pub trait VecZnxBigNegateAssign<B: Backend> {
    fn vec_znx_big_negate_assign(&self, res: &mut VecZnxBigBackendMut<'_, B>, res_col: usize);
}

/// Returns scratch bytes required for [`VecZnxBigNormalize`].
pub trait VecZnxBigNormalizeTmpBytes {
    fn vec_znx_big_normalize_tmp_bytes(&self) -> usize;
}

#[allow(clippy::too_many_arguments)]
/// Normalizes a [`VecZnxBig`](crate::layouts::VecZnxBig) into a coefficient-domain
/// [`VecZnx`](crate::layouts::VecZnx) with the target base and offset.
pub trait VecZnxBigNormalize<B: Backend> {
    fn vec_znx_big_normalize<'s, 'r, 'a>(
        &self,
        res: &mut VecZnxBackendMut<'r, B>,
        res_base2k: usize,
        res_offset: i64,
        res_col: usize,
        a: &VecZnxBigBackendRef<'a, B>,
        a_base2k: usize,
        a_col: usize,
        scratch: &mut ScratchArena<'s, B>,
    );
}

/// Returns scratch bytes required for in-place automorphism on [`VecZnxBig`](crate::layouts::VecZnxBig).
pub trait VecZnxBigAutomorphismAssignTmpBytes {
    fn vec_znx_big_automorphism_assign_tmp_bytes(&self) -> usize;
}

pub trait VecZnxBigAutomorphism<B: Backend> {
    /// Applies the automorphism X^i -> X^ik on `a` and stores the result on `b`.
    fn vec_znx_big_automorphism(
        &self,
        p: i64,
        res: &mut VecZnxBigBackendMut<'_, B>,
        res_col: usize,
        a: &VecZnxBigBackendRef<'_, B>,
        a_col: usize,
    );
}

pub trait VecZnxBigAutomorphismAssign<B: Backend> {
    /// Applies the automorphism X^i -> X^ik on `a` and stores the result on `a`.
    fn vec_znx_big_automorphism_assign<'s>(
        &self,
        p: i64,
        res: &mut VecZnxBigBackendMut<'_, B>,
        res_col: usize,
        scratch: &mut ScratchArena<'s, B>,
    );
}
