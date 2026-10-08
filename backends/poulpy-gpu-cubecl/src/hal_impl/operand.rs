//! Layouts as kernel operands.

use poulpy_hal::layouts::{Backend, DataView, Module, Ring, VecZnx, VecZnxBig, VecZnxShape};

use crate::{
    NTT4x30Gpu,
    module::handle,
    runtime::{Binding, GpuMut, GpuRef, GpuRuntime},
    znx::{Column, Map, Term, lincomb, permute},
};

fn column(binding: Option<Binding>, shape: VecZnxShape, col: usize, words: u32) -> Option<Column> {
    assert!(col < shape.cols(), "column {col} out of range ({} columns)", shape.cols());
    let binding = binding?;
    let base = binding.base::<u64>() as usize + shape.scalar_offset(col, 0) * words as usize;
    let stride = shape.limb_step() * shape.cols() * shape.n_full() * words as usize;
    Some(Column {
        base: u32::try_from(base).expect("device view starts beyond the 32-bit word range"),
        stride: u32::try_from(stride).expect("limb stride beyond the 32-bit word range"),
        words,
        n: shape.n(),
        size: shape.size(),
        binding,
    })
}

/// Column `col` of a shared view, or `None` when the view holds no storage.
pub(crate) fn column_ref<RT: GpuRuntime>(a: &VecZnx<GpuRef<'_, RT>, i64>, col: usize) -> Option<Column> {
    column(a.data().binding(), a.shape(), col, 1)
}

/// Column `col` of an exclusive view, or `None` when the view holds no storage.
pub(crate) fn column_mut<RT: GpuRuntime>(a: &VecZnx<GpuMut<'_, RT>, i64>, col: usize) -> Option<Column> {
    column(a.data().binding(), a.shape(), col, 1)
}

/// Column `col` of a shared big view, or `None` when the view holds no storage.
pub(crate) fn big_column_ref<RT: GpuRuntime, B: Backend<BigWord = i128>>(
    a: &VecZnxBig<GpuRef<'_, RT>, i128, B>,
    col: usize,
) -> Option<Column> {
    column(a.data().binding(), a.shape(), col, 2)
}

/// Column `col` of an exclusive big view, or `None` when the view holds no storage.
pub(crate) fn big_column_mut<RT: GpuRuntime, B: Backend<BigWord = i128>>(
    a: &VecZnxBig<GpuMut<'_, RT>, i128, B>,
    col: usize,
) -> Option<Column> {
    column(a.data().binding(), a.shape(), col, 2)
}

/// An operand with no limbs, bound to `like`'s storage so the launch has a buffer.
pub(crate) fn empty_like(like: &Column) -> Column {
    Column {
        binding: like.binding.clone(),
        base: like.base,
        stride: like.stride,
        words: like.words,
        n: like.n,
        size: 0,
    }
}

pub(crate) fn automorphism_map<R: Ring>(p: i64) -> Map {
    if R::CYCLOTOMIC_ORDER_FACTOR == 4 {
        Map::AutomorphismCI { p }
    } else {
        Map::Automorphism { p }
    }
}

/// `res = map(res)` through the module's device temporary.
pub(crate) fn permute_assign<RT: GpuRuntime, R: Ring>(module: &Module<NTT4x30Gpu<RT, R>>, res: &Column, map: Map) {
    let words = res.n * res.size * res.words as usize;
    handle(module).with_temp(words * size_of::<u64>(), |temp| {
        let tmp = Column {
            binding: temp.view().binding().expect("temporary holds storage"),
            base: 0,
            stride: res.n as u32 * res.words,
            words: res.words,
            n: res.n,
            size: res.size,
        };
        permute::<RT>(&tmp, res, map);
        lincomb::<RT>(res, res.size, Term::plus(&tmp), Term::zero(&tmp));
    });
}
