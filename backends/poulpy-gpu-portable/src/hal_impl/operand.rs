//! Layouts as kernel operands.

use poulpy_cpu_portable::kernels::ntt4x30::primes::Primes30;
use poulpy_hal::layouts::{
    Backend, CnvPVecL, CnvPVecR, CrtWord, Data, DataView, MatZnx, Module, Ring, ScalarZnx, SvpPPol, VecZnx, VecZnxBig, VecZnxDft,
    VecZnxShape, VmpPMat,
};

use crate::{
    NTT4x30Gpu,
    module::handle,
    ntt4x30::{DftColumn, Pmat},
    runtime::{Binding, GpuBuf, GpuMut, GpuRef, GpuRuntime},
    znx::{Column, Map, Term, lincomb, permute},
};

type Word = CrtWord<Primes30, u32>;

/// Panics unless the view of `binding` holds its first `elems` elements of `T`,
/// each at an index in the kernels' 32-bit range: the kernels launch
/// unchecked, so a layout wider than its storage must stop here.
fn assert_holds<T>(binding: &Binding, elems: usize) {
    let end = (binding.base::<T>() as usize)
        .checked_add(elems)
        .expect("operand size overflows usize");
    assert!(
        end.checked_mul(size_of::<T>())
            .is_some_and(|bytes| bytes <= binding.offset + binding.len),
        "device storage shorter than its layout"
    );
    assert!(u32::try_from(end).is_ok(), "device column ends beyond the 32-bit word range");
}

/// The product of layout dimensions, in elements.
fn elems(dims: &[usize]) -> usize {
    dims.iter()
        .try_fold(1usize, |product, &dim| product.checked_mul(dim))
        .expect("operand size overflows usize")
}

fn column(binding: Option<Binding>, shape: VecZnxShape, col: usize, words: u32) -> Option<Column> {
    assert!(col < shape.cols(), "operand column out of range");
    let binding = binding?;
    let overflow = "operand size overflows usize";
    let base = elems(&[shape.scalar_offset(col, 0), words as usize])
        .checked_add(binding.base::<u64>() as usize)
        .expect(overflow);
    let stride = elems(&[shape.limb_step(), shape.cols(), shape.n_full(), words as usize]);
    // The kernels index in 32 bits: the last word of the column must be in range.
    let end = elems(&[stride, shape.size().saturating_sub(1)])
        .checked_add(elems(&[shape.n(), words as usize]))
        .and_then(|span| span.checked_add(base))
        .expect(overflow);
    assert!(u32::try_from(end).is_ok(), "device column ends beyond the 32-bit word range");
    if shape.size() > 0 {
        assert_holds::<u64>(&binding, end - binding.base::<u64>() as usize);
    }
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
    assert!(col < cols, "operand column out of range");
    let binding = binding?;
    let block = 4 * n;
    assert_holds::<u32>(&binding, elems(&[cols, size, block]));
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

/// Storage a layout can be bound from.
pub(crate) trait Bind: Data {
    /// The launch binding of the storage, or `None` when it is empty.
    fn bind(&self) -> Option<Binding>;
}

impl<RT: GpuRuntime> Bind for GpuRef<'_, RT> {
    fn bind(&self) -> Option<Binding> {
        self.binding()
    }
}

impl<RT: GpuRuntime> Bind for GpuMut<'_, RT> {
    fn bind(&self) -> Option<Binding> {
        self.binding()
    }
}

impl<RT: GpuRuntime> Bind for GpuBuf<RT> {
    fn bind(&self) -> Option<Binding> {
        self.view().binding()
    }
}

/// Column `col` of a prepared left convolution operand, which has the layout of a DFT-domain vector.
pub(crate) fn cnv_left_column<D: Bind, B: Backend<DftWord = Word>>(a: &CnvPVecL<D, Word, B>, col: usize) -> Option<DftColumn> {
    dft_column(a.data().bind(), a.n(), a.cols(), a.size(), col)
}

/// Column `col` of a prepared right convolution operand, which has the layout of a DFT-domain vector.
pub(crate) fn cnv_right_column<D: Bind, B: Backend<DftWord = Word>>(a: &CnvPVecR<D, Word, B>, col: usize) -> Option<DftColumn> {
    dft_column(a.data().bind(), a.n(), a.cols(), a.size(), col)
}

/// Column `col` of a prepared scalar polynomial: one block per column.
pub(crate) fn svp_column<D: Bind, B: Backend<DftWord = Word>>(a: &SvpPPol<D, Word, B>, col: usize) -> Option<DftColumn> {
    dft_column(a.data().bind(), a.n(), a.cols(), 1, col)
}

/// A prepared matrix, or `None` when it holds no storage.
pub(crate) fn pmat<D: Bind, B: Backend<DftWord = Word>>(a: &VmpPMat<D, Word, B>) -> Option<Pmat> {
    let binding = a.data().bind()?;
    assert_holds::<u32>(&binding, elems(&[a.rows(), a.cols_in(), a.cols_out(), a.size(), 4, a.n()]));
    Some(Pmat {
        base: binding.base::<u32>(),
        n: a.n(),
        rows: a.rows(),
        cols_in: a.cols_in(),
        cols_out: a.cols_out(),
        size: a.size(),
        binding,
    })
}

/// Every polynomial of a coefficient-domain matrix as the limbs of one
/// column, in the block order of the prepared matrix.
pub(crate) fn mat_blocks<D: Bind>(a: &MatZnx<D, i64>) -> Option<Column> {
    let binding = a.data().bind()?;
    assert_holds::<u64>(&binding, elems(&[a.rows(), a.cols_in(), a.cols_out(), a.size(), a.n()]));
    Some(Column {
        base: binding.base::<u64>(),
        stride: a.n() as u32,
        words: 1,
        n: a.n(),
        size: a.rows() * a.cols_in() * a.cols_out() * a.size(),
        binding,
    })
}

/// Column `col` of a scalar polynomial vector, as a one-limb column.
pub(crate) fn scalar_column<D: Bind>(a: &ScalarZnx<D, i64>, col: usize) -> Option<Column> {
    assert!(col < a.cols(), "operand column out of range");
    let binding = a.data().bind()?;
    assert_holds::<u64>(&binding, elems(&[a.cols(), a.n()]));
    Some(Column {
        base: u32::try_from(binding.base::<u64>() as usize + col * a.n())
            .expect("device view starts beyond the 32-bit word range"),
        stride: a.n() as u32,
        words: 1,
        n: a.n(),
        size: 1,
        binding,
    })
}

/// The first `limbs` limbs of every column of a dense vector as the limbs of
/// one column: block `limb * cols + col` of the vector is limb
/// `limb * cols + col` of the result.
pub(crate) fn all_blocks(column0: &Column, cols: usize, limbs: usize) -> Column {
    // Column 0 was checked up to its own last limb: the other columns reach further.
    let start = (column0.base - column0.binding.base::<u64>()) as usize;
    assert_holds::<u64>(
        &column0.binding,
        start
            .checked_add(elems(&[limbs, cols, column0.n, column0.words as usize]))
            .expect("operand size overflows usize"),
    );
    Column {
        binding: column0.binding.clone(),
        base: column0.base,
        stride: column0.n as u32 * column0.words,
        words: column0.words,
        n: column0.n,
        size: limbs * cols,
    }
}

/// [`all_blocks`] of a DFT-domain vector given by its column 0.
pub(crate) fn all_dft_blocks(column0: &DftColumn, limbs: usize) -> DftColumn {
    let block = 4 * column0.n as u32;
    DftColumn {
        binding: column0.binding.clone(),
        base: column0.base,
        stride: block,
        n: column0.n,
        size: limbs * (column0.stride / block) as usize,
    }
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
