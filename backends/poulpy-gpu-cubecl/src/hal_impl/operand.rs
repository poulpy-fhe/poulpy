//! Layouts as kernel operands.

use poulpy_cpu_portable::kernels::ntt4x30::primes::Primes30;
use poulpy_hal::layouts::{Backend, CrtWord, DataView, Module, Ring, VecZnx, VecZnxBig, VecZnxDft, VecZnxShape};

use crate::{
    NTT4x30Gpu,
    module::handle,
    ntt4x30::DftColumn,
    runtime::{Binding, GpuMut, GpuRef, GpuRuntime},
    znx::{Column, Map, Term, lincomb, permute},
};

type Word = CrtWord<Primes30, u32>;

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

/// A DFT-domain vector is dense: limb `j` of column `col` is block `j * cols + col`.
fn dft_column(binding: Option<Binding>, n: usize, cols: usize, size: usize, col: usize) -> Option<DftColumn> {
    assert!(col < cols, "column {col} out of range ({cols} columns)");
    let binding = binding?;
    let block = 4 * n;
    Some(DftColumn {
        base: u32::try_from(binding.base::<u32>() as usize + col * block)
            .expect("device view starts beyond the 32-bit word range"),
        stride: u32::try_from(cols * block).expect("limb stride beyond the 32-bit word range"),
        n,
        size,
        binding,
    })
}

/// Column `col` of a shared DFT view, or `None` when the view holds no storage.
pub(crate) fn dft_column_ref<RT: GpuRuntime, B: Backend<DftWord = Word>>(
    a: &VecZnxDft<GpuRef<'_, RT>, Word, B>,
    col: usize,
) -> Option<DftColumn> {
    dft_column(a.data().binding(), a.n(), a.cols(), a.size(), col)
}

/// Column `col` of an exclusive DFT view, or `None` when the view holds no storage.
pub(crate) fn dft_column_mut<RT: GpuRuntime, B: Backend<DftWord = Word>>(
    a: &VecZnxDft<GpuMut<'_, RT>, Word, B>,
    col: usize,
) -> Option<DftColumn> {
    dft_column(a.data().binding(), a.n(), a.cols(), a.size(), col)
}

/// A DFT operand with no limbs, bound to `like`'s storage so the launch has a buffer.
pub(crate) fn empty_dft_like(like: &DftColumn) -> DftColumn {
    like.limbs(0, 1, 0)
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
