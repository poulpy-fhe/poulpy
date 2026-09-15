use super::{HostBufMut, take_host_typed};

use std::mem::size_of;

use crate::reference::vec_znx::{
    vec_znx_add, vec_znx_automorphism, vec_znx_automorphism_assign, vec_znx_automorphism_assign_tmp_bytes, vec_znx_copy,
    vec_znx_fill_uniform_ref, vec_znx_negate, vec_znx_negate_assign, vec_znx_rotate, vec_znx_rotate_assign,
    vec_znx_rotate_assign_tmp_bytes, vec_znx_sub, vec_znx_sub_assign, vec_znx_sub_negate_assign, vec_znx_switch_ring,
    vec_znx_zero,
};
use crate::reference::znx::{
    ZnxAdd, ZnxAddAssign, ZnxAutomorphism, ZnxCopy, ZnxNegate, ZnxNegateAssign, ZnxRotate, ZnxSub, ZnxSubAssign,
    ZnxSubNegateAssign, ZnxSwitchRing, ZnxZero,
};
use poulpy_hal::{
    layouts::{Backend, HostDataMut, Module, ScratchArena, VecZnxBackendMut, VecZnxBackendRef, ZnxView, ZnxViewMut},
    source::Source,
};

pub trait HalVecZnxDefault: Backend<ZnxWord = i64>
where
    Self::OwnedBuf: poulpy_hal::layouts::HostDataMut,
{
    fn vec_znx_zero_default(_module: &Module<Self>, res: &mut VecZnxBackendMut<'_, Self>, res_col: usize)
    where
        Self: ZnxZero,
        for<'x> Self::BufMut<'x>: HostDataMut,
    {
        vec_znx_zero::<Self>(res, res_col);
    }

    fn vec_znx_add_default<'a>(
        _module: &Module<Self>,
        res: &mut VecZnxBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxBackendRef<'a, Self>,
        a_col: usize,
        b: &VecZnxBackendRef<'a, Self>,
        b_col: usize,
    ) where
        Self: ZnxAdd + ZnxCopy + ZnxZero,
        for<'x> Self::BufMut<'x>: HostDataMut,
        for<'x> Self::BufRef<'x>: PartialEq + Eq + Sized + Default + AsRef<[u8]> + Sync,
    {
        vec_znx_add::<Self>(res, res_col, a, a_col, b, b_col);
    }

    fn vec_znx_add_assign_default(
        _module: &Module<Self>,
        res: &mut VecZnxBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, Self>,
        a_col: usize,
    ) where
        Self: ZnxAddAssign,
        for<'x> Self::BufMut<'x>: HostDataMut,
        for<'x> Self::BufRef<'x>: poulpy_hal::layouts::HostDataRef,
    {
        {
            assert_eq!(a.n(), res.n());
        }

        let sum_size: usize = a.size().min(res.size());

        for j in 0..sum_size {
            Self::znx_add_assign(res.at_mut(res_col, j), a.at(a_col, j));
        }
    }

    fn vec_znx_sub_default<'a>(
        _module: &Module<Self>,
        res: &mut VecZnxBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxBackendRef<'a, Self>,
        a_col: usize,
        b: &VecZnxBackendRef<'a, Self>,
        b_col: usize,
    ) where
        Self: ZnxSub + ZnxNegate + ZnxZero + ZnxCopy,
        for<'x> Self::BufMut<'x>: HostDataMut,
        for<'x> Self::BufRef<'x>: poulpy_hal::layouts::HostDataRef,
    {
        vec_znx_sub::<Self>(res, res_col, a, a_col, b, b_col);
    }

    fn vec_znx_sub_assign_default(
        _module: &Module<Self>,
        res: &mut VecZnxBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, Self>,
        a_col: usize,
    ) where
        Self: ZnxSubAssign,
        for<'x> Self::BufMut<'x>: HostDataMut,
        for<'x> Self::BufRef<'x>: poulpy_hal::layouts::HostDataRef,
    {
        vec_znx_sub_assign::<Self>(res, res_col, a, a_col);
    }

    fn vec_znx_sub_negate_assign_default(
        _module: &Module<Self>,
        res: &mut VecZnxBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, Self>,
        a_col: usize,
    ) where
        Self: ZnxSubNegateAssign + ZnxNegateAssign,
        for<'x> Self::BufMut<'x>: HostDataMut,
        for<'x> Self::BufRef<'x>: poulpy_hal::layouts::HostDataRef,
    {
        vec_znx_sub_negate_assign::<Self>(res, res_col, a, a_col);
    }

    fn vec_znx_negate_default(
        _module: &Module<Self>,
        res: &mut VecZnxBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, Self>,
        a_col: usize,
    ) where
        Self: ZnxNegate + ZnxZero,
        for<'x> Self::BufMut<'x>: HostDataMut,
        for<'x> Self::BufRef<'x>: poulpy_hal::layouts::HostDataRef,
    {
        vec_znx_negate::<Self>(res, res_col, a, a_col);
    }

    fn vec_znx_negate_assign_default(_module: &Module<Self>, res: &mut VecZnxBackendMut<'_, Self>, res_col: usize)
    where
        Self: ZnxNegateAssign,
        for<'x> Self::BufMut<'x>: HostDataMut,
    {
        vec_znx_negate_assign::<Self>(res, res_col);
    }

    fn vec_znx_rotate_default(
        _module: &Module<Self>,
        p: i64,
        res: &mut VecZnxBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, Self>,
        a_col: usize,
    ) where
        Self: ZnxRotate + ZnxZero,
        for<'x> Self::BufMut<'x>: HostDataMut,
        for<'x> Self::BufRef<'x>: poulpy_hal::layouts::HostDataRef,
    {
        vec_znx_rotate::<Self>(p, res, res_col, a, a_col);
    }

    fn vec_znx_rotate_assign_tmp_bytes_default(module: &Module<Self>) -> usize {
        vec_znx_rotate_assign_tmp_bytes(module.n())
    }

    fn vec_znx_rotate_assign_default(
        module: &Module<Self>,
        p: i64,
        res: &mut VecZnxBackendMut<'_, Self>,
        res_col: usize,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        Self: ZnxRotate + ZnxCopy,
        for<'x> Self::BufMut<'x>: HostDataMut,
        for<'x> Self::BufMut<'x>: HostBufMut<'x>,
    {
        let (tmp, _) = take_host_typed::<Self, i64>(
            scratch.borrow(),
            vec_znx_rotate_assign_tmp_bytes(module.n()) / size_of::<i64>(),
        );
        vec_znx_rotate_assign::<Self>(p, res, res_col, tmp);
    }

    fn vec_znx_automorphism_default(
        _module: &Module<Self>,
        p: i64,
        res: &mut VecZnxBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, Self>,
        a_col: usize,
    ) where
        Self: ZnxAutomorphism + ZnxZero,
        for<'x> Self::BufMut<'x>: HostDataMut,
        for<'x> Self::BufRef<'x>: poulpy_hal::layouts::HostDataRef,
    {
        vec_znx_automorphism::<Self>(p, res, res_col, a, a_col);
    }

    fn vec_znx_automorphism_assign_tmp_bytes_default(module: &Module<Self>) -> usize {
        vec_znx_automorphism_assign_tmp_bytes(module.n())
    }

    fn vec_znx_automorphism_assign_default(
        module: &Module<Self>,
        p: i64,
        res: &mut VecZnxBackendMut<'_, Self>,
        res_col: usize,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        Self: ZnxAutomorphism + ZnxCopy,
        for<'x> Self::BufMut<'x>: HostDataMut,
        for<'x> Self::BufMut<'x>: HostBufMut<'x>,
    {
        let (tmp, _) = take_host_typed::<Self, i64>(
            scratch.borrow(),
            vec_znx_automorphism_assign_tmp_bytes(module.n()) / size_of::<i64>(),
        );
        vec_znx_automorphism_assign::<Self>(p, res, res_col, tmp);
    }

    fn vec_znx_switch_ring_default(
        _module: &Module<Self>,
        res: &mut VecZnxBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, Self>,
        a_col: usize,
    ) where
        Self: ZnxCopy + ZnxSwitchRing + ZnxZero,
        for<'x> Self::BufMut<'x>: HostDataMut,
        for<'x> Self::BufRef<'x>: poulpy_hal::layouts::HostDataRef,
    {
        vec_znx_switch_ring::<Self>(res, res_col, a, a_col);
    }

    fn vec_znx_copy_default(
        _module: &Module<Self>,
        res: &mut VecZnxBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, Self>,
        a_col: usize,
    ) where
        Self: ZnxCopy + ZnxZero,
        for<'x> Self::BufMut<'x>: HostDataMut,
        for<'x> Self::BufRef<'x>: poulpy_hal::layouts::HostDataRef,
    {
        vec_znx_copy::<Self>(res, res_col, a, a_col);
    }

    fn vec_znx_fill_uniform_default(
        _module: &Module<Self>,
        base2k: usize,
        k: usize,
        res: &mut VecZnxBackendMut<'_, Self>,
        res_col: usize,
        seed: [u8; 32],
    ) where
        for<'x> Self::BufMut<'x>: HostDataMut,
    {
        let mut source = Source::new(seed);
        vec_znx_fill_uniform_ref::<Self>(base2k, k, res, res_col, &mut source);
    }
}

impl<BE: Backend<ZnxWord = i64>> HalVecZnxDefault for BE where BE::OwnedBuf: poulpy_hal::layouts::HostDataMut {}
