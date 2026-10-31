//! Repacking between the lines of a grid's matrices and vectors.

use poulpy_ckks::{
    CoeffsMeta, SetGLWEPlaintextInfos,
    api::{CKKSEncodingOps, CKKSLinearTransformationOps},
    test_suite::{
        CKKSTestParams,
        helpers::{TestContextBackend, TestContextModule, TestScalar, alloc_scratch, gen_sk_with_raw},
        parity::helpers::with_scratch,
        reference_encoder::ReferenceEncoder,
    },
};
use poulpy_core::layouts::{Base2K, DiagonalArithmetic};
use poulpy_hal::{
    api::{CnvPVecAlloc, ReimNew, ScratchOwnedBorrow},
    layouts::{CyclotomicOrder, HostBytesBackend, HostDataMut, HostDataRef, Module, ScratchArena, Standard},
};

use super::{
    helpers::{
        Grid, Misshapen,
        Rejection::{Budget, InvalidOperand, MissingKey},
        TestKeys, alloc_grid, assert_grid_decrypt, assert_guard, assert_guard_on, assert_short_scratch_panics, encrypt_grid,
        foreign_layouts, grid_snapshot, poisoned_scratch, rejected_grids, relabel, rotation_keys,
    },
    matvec::{
        alloc_vector, assert_vector_decrypt, assert_vector_guard, encrypt_vector, matvec_entry, matvec_operands_of, vector_entry,
        vector_snapshot,
    },
};
use crate::{
    api::{BlasEncodingOps, BlasMatVecOps},
    layouts::{
        BlasCiphertextOwned, BlasGridInfos, BlasGridLine, BlasGridMut, BlasLayout, BlasMatVecOperandLayout, BlasVectorShape,
        blas_grid_to_vector_output_layout, blas_vector_to_grid_output_layout, matvec_pt_galois_elements,
    },
    oep::{BlasEncodingImpl, BlasMatVecImpl},
};

/// Whether the cases of these tests fit `slots` slots at blocks of dimension `d = 2^log_d`: their vectors span 16 slots
/// and their periods `4 d^2`.
pub(crate) fn repack_fits(slots: usize, log_d: usize) -> bool {
    slots >= 16 && 4 << (2 * log_d) <= slots
}

/// Entry `(s, r, c)` of the grids of these tests, dyadic in `[-1, 1]`, real.
fn grid_entry(s: usize, r: usize, c: usize) -> f64 {
    ((s * 11 + r * 5 + c * 3) % 17) as f64 / 8.0 - 1.0
}

/// The identity matrix, which extracts or places a line as it is.
fn identity_entry(row: usize, col: usize) -> f64 {
    f64::from(u8::from(row == col))
}

/// `W v` for the matrix `W` of `entry`, of `rows` entries.
pub(crate) fn matrix_times(entry: fn(usize, usize) -> f64, rows: usize, v: &[f64]) -> Vec<f64> {
    (0..rows)
        .map(|row| v.iter().enumerate().map(|(col, x)| entry(row, col) * x).sum())
        .collect()
}

/// A grid-to-vector case: the line read, the operand's layout and its matrix, the matrices the label of the grid read
/// proves, which records the two of [`GRID`], and its blocks' `log_d`.
pub(crate) struct GridToVectorCase {
    pub line: BlasGridLine,
    pub layout: BlasMatVecOperandLayout,
    pub entry: fn(usize, usize) -> f64,
    pub proven: usize,
    pub log_d: usize,
}

impl GridToVectorCase {
    /// The label of the grid the case reads: `log2(slots / (proven d^2))`.
    pub(crate) fn label(&self, slots: usize) -> usize {
        ((slots >> (2 * self.log_d)) / self.proven).ilog2() as usize
    }
}

/// The `2 x 3` grids of `d x d` blocks recording two matrices the grid-to-vector cases read.
pub(crate) const GRID: (usize, usize, usize) = (2, 3, 2);

/// The grid-to-vector cases on `slots` slots at blocks of dimension `d = 2^log_d`, which [`repack_fits`], at `params`'
/// radix and precision, reading a grid of [`GRID`] whose label proves both matrices: a row of the second matrix past
/// the first block row through a public matrix into one dense ciphertext, and through the identity, which extracts it;
/// a column of the first matrix past the first block column, read at a period longer than the grid's, into two
/// ciphertexts repeating every 8 slots. Then a row of the second matrix of a grid whose label proves one, past the
/// period it proves, in its repetition.
pub(crate) fn grid_to_vector_cases(params: &CKKSTestParams, slots: usize, log_d: usize) -> Vec<GridToVectorCase> {
    let (base2k, meta): (Base2K, CoeffsMeta) = (params.base2k.into(), params.prec().into());
    let (d, (rows, cols, lanes)) = (1 << log_d, GRID);
    let row = BlasGridLine::Row { mat_slot: 1, row: d + 1 };
    let col = BlasGridLine::Col {
        mat_slot: 0,
        col: 2 * d + 1,
    };
    let shape = |line: BlasGridLine| {
        let (len, offset, stride) = match line {
            BlasGridLine::Row { mat_slot, row } => (cols * d, mat_slot * d * d + (row % d) * d, 1),
            BlasGridLine::Col { mat_slot, col } => (rows * d, mat_slot * d * d + col % d, d),
        };
        BlasVectorShape::new(len, d, offset, stride)
    };
    let case = |line, p_in, output, p_out, entry, proven| GridToVectorCase {
        line,
        layout: BlasMatVecOperandLayout::new(slots, shape(line), p_in, output, p_out, base2k, meta),
        entry,
        proven,
        log_d,
    };
    let repeated = BlasGridLine::Row { mat_slot: 1, row: 1 };
    let period = lanes * d * d;
    vec![
        case(row, period, BlasVectorShape::new(4, 4, 0, 1), slots, matvec_entry, lanes),
        case(
            row,
            period,
            BlasVectorShape::new(cols * d, cols * d, 2, 1),
            slots,
            identity_entry,
            lanes,
        ),
        case(col, 2 * period, BlasVectorShape::new(3, 2, 1, 3), 8, matvec_entry, lanes),
        case(repeated, period, BlasVectorShape::new(4, 4, 0, 1), slots, matvec_entry, 1),
    ]
}

/// The entries of `line` of `grid`, whose `proven` matrices repeat over the batch.
pub(crate) fn line_values(grid: &Grid, line: BlasGridLine, proven: usize) -> Vec<f64> {
    match line {
        BlasGridLine::Row { mat_slot, row } => (0..grid.cols * grid.d())
            .map(|c| grid.at(mat_slot % proven, row, c).0)
            .collect(),
        BlasGridLine::Col { mat_slot, col } => (0..grid.rows * grid.d())
            .map(|r| grid.at(mat_slot % proven, r, col).0)
            .collect(),
    }
}

/// The real grid of [`GRID`]'s dimensions of `2^log_d x 2^log_d` blocks holding [`grid_entry`] in `proven` matrices.
pub(crate) fn grid_model(log_d: usize, proven: usize) -> Grid {
    let (rows, cols, _) = GRID;
    Grid::from_fn(log_d, rows, cols, proven, |s, r, c| (grid_entry(s, r, c), 0.0))
}

/// `dec(grid_to_vector(W, enc(a), line))` is `W x` for `x` the line of `a`, at every repetition of the output's period
/// and zero elsewhere, for rows and columns of a `2 x 3` grid of `d x d` blocks recording two matrices, past the first
/// block and matrix, through a public matrix and the identity, at an input period equal to and longer than the grid's,
/// and in a matrix the grid's label proves only as a repetition of the first, with the operand resident and streamed,
/// which give the same bits, on the operand's key set alone, at the metadata [`blas_grid_to_vector_output_layout`]
/// gives; `res` is one limb wider than the result. One byte short of its query, the call panics before it writes
/// `res`. Below `4 d^2` slots, or 16, it is skipped with a message.
pub fn test_blas_grid_to_vector<BE, F, E>(
    params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
    log_d: usize,
) where
    BE: TestContextBackend<Ring = Standard> + BlasMatVecImpl + BlasEncodingImpl,
    Module<BE>: TestContextModule<BE>
        + CKKSEncodingOps<BE, F>
        + BlasEncodingOps<BE, F>
        + CKKSLinearTransformationOps<BE>
        + CnvPVecAlloc<BE>,
    F: TestScalar + DiagonalArithmetic,
    E: ReimNew<F>,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
{
    let slots = params.n / 2;
    if !repack_fits(slots, log_d) {
        eprintln!("grid_to_vector skips log_d {log_d}: its cases span 4 d^2 and 16 slots, the ring holds {slots}");
        return;
    }
    let encoder = ReferenceEncoder::<E>::new(slots).unwrap();
    let (sk_raw, sk) = gen_sk_with_raw(&params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, module);
    for case in grid_to_vector_cases(&params, slots, log_d) {
        let (line, layout, entry, proven) = (case.line, case.layout, case.entry, case.proven);
        let label = format!("grid_to_vector {line:?} {layout:?}, {proven} matrices proven");
        let model = grid_model(log_d, proven);
        let mut a = encrypt_grid::<BE, F, E>(
            &params,
            module,
            host_module,
            &encoder,
            &sk,
            params.k,
            case.label(slots),
            &model,
            &mut scratch.borrow(),
        );
        BlasGridMut::<BE>::set_mat_slots(&mut a, GRID.2);
        assert_eq!(line.shape(&a), layout.input, "{label}: line shape");
        let (streamed, resident) = matvec_operands_of::<BE, F>(module, &layout, entry);
        let keys = TestKeys::<BE> {
            atks: rotation_keys(
                &params,
                module,
                &sk_raw,
                matvec_pt_galois_elements(module.cyclotomic_order(), &layout),
                &mut scratch.borrow(),
            ),
            tsk: None,
        };
        let want = matrix_times(entry, layout.output.len, &line_values(&model, line, proven));
        let res_k = params.k + params.base2k;
        let template = alloc_vector(&params, module, layout.output, res_k);
        let want_layout = blas_grid_to_vector_output_layout(&template, &layout, &a);
        let resident_bytes = module.blas_grid_to_vector_tmp_bytes(&template, &resident, &a, &params.atk_layout());
        let streamed_bytes = module.blas_grid_to_vector_tmp_bytes(&template, &streamed, &a, &params.atk_layout());
        let mut have = alloc_vector(&params, module, layout.output, res_k);
        let mut op_scratch = poisoned_scratch::<BE>(resident_bytes);
        module
            .blas_grid_to_vector(&mut have, &resident, &a, line, &keys, &mut op_scratch.borrow())
            .unwrap();
        assert_vector_decrypt::<BE, F, E>(
            &label,
            &params,
            module,
            &encoder,
            &have,
            &sk,
            &want,
            layout.output_period,
            &want_layout,
            &mut scratch.borrow(),
        );
        let mut have_streamed = alloc_vector(&params, module, layout.output, res_k);
        let mut op_scratch = poisoned_scratch::<BE>(streamed_bytes);
        module
            .blas_grid_to_vector(&mut have_streamed, &streamed, &a, line, &keys, &mut op_scratch.borrow())
            .unwrap();
        assert_eq!(
            vector_snapshot::<BE>(&have_streamed),
            vector_snapshot::<BE>(&have),
            "{label}: streamed against resident"
        );
        let mut res = alloc_vector(&params, module, layout.output, res_k);
        let before = vector_snapshot::<BE>(&res);
        assert_short_scratch_panics::<BE, _>("blas_grid_to_vector", resident_bytes, |s| {
            module.blas_grid_to_vector(&mut res, &resident, &a, line, &keys, s)
        });
        assert!(vector_snapshot::<BE>(&res) == before, "{label}: wrote res before panicking");
    }
}

/// A vector-to-grid case: the line written, the dimensions of `res`, the operand's layout and its matrix, and the
/// blocks' `log_d`.
pub(crate) struct VectorToGridCase {
    pub line: BlasGridLine,
    pub rows: usize,
    pub cols: usize,
    pub layout: BlasMatVecOperandLayout,
    pub entry: fn(usize, usize) -> f64,
    pub log_d: usize,
}

/// The vector-to-grid cases on `slots` slots at blocks of dimension `d = 2^log_d`, which [`repack_fits`], at `params`'
/// radix and precision: a row of the second matrix into a `1 x 3` grid holding two matrices from one dense ciphertext;
/// a column of the third matrix into a `2 x 1` grid holding four matrices from a vector repeating every 8 slots; the
/// last column of the second matrix placed through the identity into a `2 x 1` grid holding two; a scalar repeating
/// over every slot spread into the last row of a `1 x 2` grid holding one matrix, through maps that rotate nothing.
pub(crate) fn vector_to_grid_cases(params: &CKKSTestParams, slots: usize, log_d: usize) -> Vec<VectorToGridCase> {
    let (base2k, meta): (Base2K, CoeffsMeta) = (params.base2k.into(), params.prec().into());
    let d = 1 << log_d;
    let case = |line, (rows, cols), input, p_in, output, p_out, entry| VectorToGridCase {
        line,
        rows,
        cols,
        layout: BlasMatVecOperandLayout::new(slots, input, p_in, output, p_out, base2k, meta),
        entry,
        log_d,
    };
    vec![
        case(
            BlasGridLine::Row { mat_slot: 1, row: 1 },
            (1, 3),
            BlasVectorShape::new(5, 5, 0, 1),
            slots,
            BlasVectorShape::new(3 * d, d, d * d + d, 1),
            2 * d * d,
            matvec_entry,
        ),
        case(
            BlasGridLine::Col { mat_slot: 2, col: 1 },
            (2, 1),
            BlasVectorShape::new(3, 3, 1, 2),
            8,
            BlasVectorShape::new(2 * d, d, 2 * d * d + 1, d),
            4 * d * d,
            matvec_entry,
        ),
        case(
            BlasGridLine::Col { mat_slot: 1, col: d - 1 },
            (2, 1),
            BlasVectorShape::new(2 * d, 2 * d, 0, 1),
            slots,
            BlasVectorShape::new(2 * d, d, d * d + d - 1, d),
            2 * d * d,
            identity_entry,
        ),
        case(
            BlasGridLine::Row { mat_slot: 0, row: d - 1 },
            (1, 2),
            BlasVectorShape::new(1, 1, 0, 1),
            1,
            BlasVectorShape::new(2 * d, d, (d - 1) * d, 1),
            d * d,
            matvec_entry,
        ),
    ]
}

/// The grid `res` of a vector-to-grid case holds: `y` as `line` of its matrix, every other entry of the batch its
/// operand records zero.
pub(crate) fn placed(case: &VectorToGridCase, y: &[f64]) -> Grid {
    let batch = case.layout.output_period >> (2 * case.log_d);
    Grid::from_fn(case.log_d, case.rows, case.cols, batch, |s, r, c| match case.line {
        BlasGridLine::Row { mat_slot, row } if s == mat_slot && r == row => (y[c], 0.0),
        BlasGridLine::Col { mat_slot, col } if s == mat_slot && c == col => (y[r], 0.0),
        _ => (0.0, 0.0),
    })
}

/// `dec(vector_to_grid(line, W, enc(v)))` holds `W v` as the line of its matrix and zero elsewhere, the batch recorded
/// `output_period / d^2`, for a row of the second matrix into a `1 x 3` grid and a column of the third into a `2 x 1`
/// grid, from dense and repeating vectors, for a column placed through the identity, and for a scalar spread into a
/// row through maps that rotate nothing, resident and streamed giving the same bits, on the operand's key set alone, at
/// the layout [`blas_vector_to_grid_output_layout`] gives; `res` is one limb wider than the result. One byte short of
/// its query, the call panics before it writes `res`. Below `4 d^2` slots, or 16, it is skipped with a message.
pub fn test_blas_vector_to_grid<BE, F, E>(
    params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
    log_d: usize,
) where
    BE: TestContextBackend<Ring = Standard> + BlasMatVecImpl + BlasEncodingImpl,
    Module<BE>: TestContextModule<BE>
        + CKKSEncodingOps<BE, F>
        + BlasEncodingOps<BE, F>
        + CKKSLinearTransformationOps<BE>
        + CnvPVecAlloc<BE>,
    F: TestScalar + DiagonalArithmetic,
    E: ReimNew<F>,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
{
    let slots = params.n / 2;
    if !repack_fits(slots, log_d) {
        eprintln!("vector_to_grid skips log_d {log_d}: its cases span 4 d^2 and 16 slots, the ring holds {slots}");
        return;
    }
    let encoder = ReferenceEncoder::<E>::new(slots).unwrap();
    let (sk_raw, sk) = gen_sk_with_raw(&params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, module);
    for case in vector_to_grid_cases(&params, slots, log_d) {
        let (line, layout) = (case.line, case.layout);
        let label = format!("vector_to_grid {line:?} {layout:?}");
        let (streamed, resident) = matvec_operands_of::<BE, F>(module, &layout, case.entry);
        let keys = TestKeys::<BE> {
            atks: rotation_keys(
                &params,
                module,
                &sk_raw,
                matvec_pt_galois_elements(module.cyclotomic_order(), &layout),
                &mut scratch.borrow(),
            ),
            tsk: None,
        };
        let v: Vec<f64> = (0..layout.input.len).map(vector_entry).collect();
        let a = encrypt_vector::<BE, F, E>(
            &params,
            module,
            host_module,
            &encoder,
            &sk,
            params.k,
            layout.input,
            layout.input_period,
            &v,
            &mut scratch.borrow(),
        );
        let res_k = params.k + params.base2k;
        let alloc = |k| alloc_grid(&params, module, log_d, case.rows, case.cols, k);
        let template = alloc(res_k);
        assert_eq!(line.shape(&template), layout.output, "{label}: line shape");
        let want_layout: BlasLayout = blas_vector_to_grid_output_layout(&template, &layout, &a);
        let want = placed(&case, &matrix_times(case.entry, layout.output.len, &v));
        let resident_bytes = module.blas_vector_to_grid_tmp_bytes(&template, &resident, &a, &params.atk_layout());
        let streamed_bytes = module.blas_vector_to_grid_tmp_bytes(&template, &streamed, &a, &params.atk_layout());
        let mut have: BlasCiphertextOwned<BE> = alloc(res_k);
        let mut op_scratch = poisoned_scratch::<BE>(resident_bytes);
        module
            .blas_vector_to_grid(&mut have, line, &resident, &a, &keys, &mut op_scratch.borrow())
            .unwrap();
        assert_grid_decrypt::<BE, F, E>(
            &label,
            &params,
            module,
            &encoder,
            &have,
            &sk,
            &want,
            &want_layout,
            &mut scratch.borrow(),
        );
        let mut have_streamed = alloc(res_k);
        let mut op_scratch = poisoned_scratch::<BE>(streamed_bytes);
        module
            .blas_vector_to_grid(&mut have_streamed, line, &streamed, &a, &keys, &mut op_scratch.borrow())
            .unwrap();
        assert!(
            grid_snapshot::<BE, _>(&have_streamed) == grid_snapshot::<BE, _>(&have)
                && have_streamed.mat_slots() == have.mat_slots(),
            "{label}: streamed against resident"
        );
        let mut res = alloc(res_k);
        let before = (res.mat_slots(), grid_snapshot::<BE, _>(&res));
        assert_short_scratch_panics::<BE, _>("blas_vector_to_grid", resident_bytes, |s| {
            module.blas_vector_to_grid(&mut res, line, &resident, &a, &keys, s)
        });
        assert!(
            (res.mat_slots(), grid_snapshot::<BE, _>(&res)) == before,
            "{label}: wrote res before panicking"
        );
    }
}

/// `blas_grid_to_vector` rejects, before any write: a row or a column outside the grid, a matrix outside its batch, a
/// line of another shape than the operand's input, every grid of [`rejected_grids`] and [`foreign_layouts`], a grid
/// whose blocks do not tile its shape, a grid repeating with a longer period than the operand reads, a store lacking a
/// key of the operand's set, `a` and `res` both of another rank than the keys, a `res` of another chunk and of another
/// length, and an input whose budget is below the operand's scale, which poulpy-ckks's summed transformation rejects.
/// `blas_vector_to_grid` rejects: a `res` of two block rows for a row or two block columns for a column, a row or a
/// column past its block, a matrix outside the batch the operand records, a line of another shape than the operand's
/// output, an input of another shape than the operand's input, an operand whose output period holds no matrix, every
/// `res` of [`rejected_grids`] and [`foreign_layouts`], a `res` whose blocks do not tile its shape, an input whose
/// entries pass the period its label proves, a store lacking a key, `a` and `res` both of another rank than the keys,
/// and an input whose budget is below the operand's scale. Below `4 d^2` slots, or 16, it is skipped with a message.
pub fn test_blas_repack_guards<BE, F, E>(
    params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
    log_d: usize,
) where
    BE: TestContextBackend<Ring = Standard> + BlasMatVecImpl + BlasEncodingImpl,
    Module<BE>: TestContextModule<BE>
        + CKKSEncodingOps<BE, F>
        + BlasEncodingOps<BE, F>
        + CKKSLinearTransformationOps<BE>
        + CnvPVecAlloc<BE>,
    F: TestScalar + DiagonalArithmetic,
    E: ReimNew<F>,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
{
    let slots = params.n / 2;
    if !repack_fits(slots, log_d) {
        eprintln!("repack_guards skips log_d {log_d}: its cases span 4 d^2 and 16 slots, the ring holds {slots}");
        return;
    }
    let d = 1 << log_d;
    let encoder = ReferenceEncoder::<E>::new(slots).unwrap();
    let (sk_raw, sk) = gen_sk_with_raw(&params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, module);
    let key_set = |layout: &BlasMatVecOperandLayout, skip: usize, scratch: &mut ScratchArena<'_, BE>| TestKeys::<BE> {
        atks: rotation_keys(
            &params,
            module,
            &sk_raw,
            matvec_pt_galois_elements(module.cyclotomic_order(), layout)
                .into_iter()
                .skip(skip),
            scratch,
        ),
        tsk: None,
    };

    // Grid to vector, on its first case.
    let op = "blas_grid_to_vector";
    let case = grid_to_vector_cases(&params, slots, log_d).swap_remove(0);
    let (line, layout, entry) = (case.line, case.layout, case.entry);
    let (_, w) = matvec_operands_of::<BE, F>(module, &layout, entry);
    let keys = key_set(&layout, 0, &mut scratch.borrow());
    let encrypt = |model: &Grid, k: usize, label: usize, scratch: &mut ScratchArena<'_, BE>| {
        encrypt_grid::<BE, F, E>(&params, module, host_module, &encoder, &sk, k, label, model, scratch)
    };
    let model = grid_model(log_d, case.proven);
    let a = encrypt(&model, params.k, case.label(slots), &mut scratch.borrow());
    let mut res = alloc_vector(&params, module, layout.output, params.k);
    let bytes = module.blas_grid_to_vector_tmp_bytes(&res, &w, &a, &params.atk_layout());
    let mut call = |what: &str, rejection, line, a: &BlasCiphertextOwned<BE>, keys: &TestKeys<BE>| {
        assert_vector_guard::<BE>(what, op, rejection, &mut res, |res| {
            with_scratch::<BE, _>(bytes, |s| module.blas_grid_to_vector(res, &w, a, line, keys, s))
        });
    };
    let (rows, cols, lanes) = GRID;
    call(
        "a row outside the grid",
        InvalidOperand,
        BlasGridLine::Row {
            mat_slot: 1,
            row: rows * d,
        },
        &a,
        &keys,
    );
    call(
        "a matrix outside the batch",
        InvalidOperand,
        BlasGridLine::Row {
            mat_slot: lanes,
            row: d + 1,
        },
        &a,
        &keys,
    );
    call(
        "a column outside the grid",
        InvalidOperand,
        BlasGridLine::Col {
            mat_slot: 0,
            col: cols * d,
        },
        &a,
        &keys,
    );
    call(
        "a column's matrix outside the batch",
        InvalidOperand,
        BlasGridLine::Col { mat_slot: lanes, col: 0 },
        &a,
        &keys,
    );
    call(
        "a line of another shape than the operand's input",
        InvalidOperand,
        BlasGridLine::Row { mat_slot: 1, row: d },
        &a,
        &keys,
    );
    for (what, bad) in rejected_grids(module, &a).into_iter().chain(foreign_layouts(module, &a)) {
        call(&format!("grid: {what}"), InvalidOperand, line, &bad, &keys);
    }
    let (grid_rows, grid_cols, _) = GRID;
    let dense = Grid::from_fn(log_d, grid_rows, grid_cols, slots / (d * d), |s, r, c| {
        (grid_entry(s, r, c), 0.0)
    });
    let dense = encrypt(&dense, params.k, 0, &mut scratch.borrow());
    call(
        "a grid repeating past the operand's period",
        InvalidOperand,
        line,
        &dense,
        &keys,
    );
    let missing = key_set(&layout, 1, &mut scratch.borrow());
    call("a missing rotation key", MissingKey, line, &a, &missing);
    let misshapen = Misshapen {
        grid: a.clone(),
        rows: rows + 1,
        cols,
    };
    assert_vector_guard::<BE>(
        "a grid whose blocks do not tile its shape",
        op,
        InvalidOperand,
        &mut res,
        |res| with_scratch::<BE, _>(bytes, |s| module.blas_grid_to_vector(res, &w, &misshapen, line, &keys, s)),
    );
    // `a` and `res` both of another rank than the keys, which the first key of the operand's maps rejects.
    let rank_params = CKKSTestParams {
        rank: params.rank + 1,
        ..params
    };
    let [.., (_, rank_a)] = foreign_layouts(module, &a);
    let rank_a = relabel(&rank_a, |meta| meta.log_sparsity = case.label(slots));
    let mut rank_res = alloc_vector(&rank_params, module, layout.output, params.k);
    assert_vector_guard::<BE>("key rank", op, InvalidOperand, &mut rank_res, |res| {
        with_scratch::<BE, _>(bytes, |s| module.blas_grid_to_vector(res, &w, &rank_a, line, &keys, s))
    });
    for (what, shape) in [
        ("res of another chunk", BlasVectorShape::new(4, 2, 0, 1)),
        ("res of another length", BlasVectorShape::new(5, 5, 0, 1)),
    ] {
        let mut other_res = alloc_vector(&params, module, shape, params.k);
        assert_vector_guard::<BE>(what, op, InvalidOperand, &mut other_res, |res| {
            with_scratch::<BE, _>(bytes, |s| module.blas_grid_to_vector(res, &w, &a, line, &keys, s))
        });
    }
    // Budget: blocks at the operand's width, whose budget is below its scale.
    let low = encrypt(
        &model,
        layout.coeffs_meta.k.as_usize(),
        case.label(slots),
        &mut scratch.borrow(),
    );
    assert_vector_guard::<BE>(
        "an input budget below the operand's scale",
        "ckks_linear_transformation_eval_sum_into",
        Budget,
        &mut res,
        |res| with_scratch::<BE, _>(bytes, |s| module.blas_grid_to_vector(res, &w, &low, line, &keys, s)),
    );

    // Vector to grid, on its first case.
    let op = "blas_vector_to_grid";
    let case = vector_to_grid_cases(&params, slots, log_d).swap_remove(0);
    let (line, layout) = (case.line, case.layout);
    let (_, w) = matvec_operands_of::<BE, F>(module, &layout, case.entry);
    let keys = key_set(&layout, 0, &mut scratch.borrow());
    let v: Vec<f64> = (0..layout.input.len).map(vector_entry).collect();
    let a = encrypt_vector::<BE, F, E>(
        &params,
        module,
        host_module,
        &encoder,
        &sk,
        params.k,
        layout.input,
        layout.input_period,
        &v,
        &mut scratch.borrow(),
    );
    let alloc = |rows, cols| alloc_grid(&params, module, log_d, rows, cols, params.k);
    let bytes = module.blas_vector_to_grid_tmp_bytes(&alloc(case.rows, case.cols), &w, &a, &params.atk_layout());
    let call = |what: &str, rejection, line, res: &mut BlasCiphertextOwned<BE>, a: &_, keys: &TestKeys<BE>| {
        assert_guard::<BE, _>(what, op, rejection, res, |res| {
            with_scratch::<BE, _>(bytes, |s| module.blas_vector_to_grid(res, line, &w, a, keys, s))
        });
    };
    call(
        "a row into two block rows",
        InvalidOperand,
        line,
        &mut alloc(2, case.cols),
        &a,
        &keys,
    );
    call(
        "a row past its block row",
        InvalidOperand,
        BlasGridLine::Row { mat_slot: 1, row: d },
        &mut alloc(case.rows, case.cols),
        &a,
        &keys,
    );
    call(
        "a column into two block columns",
        InvalidOperand,
        BlasGridLine::Col { mat_slot: 1, col: 1 },
        &mut alloc(2, 2),
        &a,
        &keys,
    );
    call(
        "a column past its block column",
        InvalidOperand,
        BlasGridLine::Col { mat_slot: 1, col: d },
        &mut alloc(2, 1),
        &a,
        &keys,
    );
    let batch = layout.output_period >> (2 * log_d);
    call(
        "a row's matrix outside the batch the operand records",
        InvalidOperand,
        BlasGridLine::Row { mat_slot: batch, row: 1 },
        &mut alloc(case.rows, case.cols),
        &a,
        &keys,
    );
    call(
        "a column's matrix outside the batch the operand records",
        InvalidOperand,
        BlasGridLine::Col { mat_slot: batch, col: 1 },
        &mut alloc(2, 1),
        &a,
        &keys,
    );
    call(
        "a line of another shape than the operand's output",
        InvalidOperand,
        BlasGridLine::Row { mat_slot: 1, row: 0 },
        &mut alloc(case.rows, case.cols),
        &a,
        &keys,
    );
    let other_a = alloc_vector(&params, module, BlasVectorShape::new(4, 4, 0, 1), params.k);
    call(
        "an input of another shape than the operand's input",
        InvalidOperand,
        line,
        &mut alloc(case.rows, case.cols),
        &other_a,
        &keys,
    );
    let bad_res = rejected_grids(module, &alloc(case.rows, case.cols))
        .into_iter()
        .chain(foreign_layouts(module, &alloc(case.rows, case.cols)));
    for (what, mut bad) in bad_res {
        call(&format!("res: {what}"), InvalidOperand, line, &mut bad, &a, &keys);
    }
    let mut misshapen = Misshapen {
        grid: alloc(case.rows, case.cols),
        rows: case.rows + 1,
        cols: case.cols,
    };
    assert_guard_on::<BE, _, _>(
        "a res whose blocks do not tile its shape",
        op,
        InvalidOperand,
        &mut misshapen,
        |m| &m.grid,
        |res| with_scratch::<BE, _>(bytes, |s| module.blas_vector_to_grid(res, line, &w, &a, &keys, s)),
    );
    let mut sparse = a.clone();
    sparse.ct_mut(0).set_log_sparsity((slots / 4).ilog2() as usize);
    call(
        "an input past the period its label proves",
        InvalidOperand,
        line,
        &mut alloc(case.rows, case.cols),
        &sparse,
        &keys,
    );
    let missing = key_set(&layout, 1, &mut scratch.borrow());
    call(
        "a missing rotation key",
        MissingKey,
        line,
        &mut alloc(case.rows, case.cols),
        &a,
        &missing,
    );
    // `a` and `res` both of another rank than the keys, which the first key of the operand's maps rejects.
    let rank_params = CKKSTestParams {
        rank: params.rank + 1,
        ..params
    };
    let [.., (_, mut rank_res)] = foreign_layouts(module, &alloc(case.rows, case.cols));
    let rank_a = alloc_vector(&rank_params, module, layout.input, params.k);
    call("key rank", InvalidOperand, line, &mut rank_res, &rank_a, &keys);
    // Budget: an input at the operand's width, whose budget is below its scale.
    let low = encrypt_vector::<BE, F, E>(
        &params,
        module,
        host_module,
        &encoder,
        &sk,
        layout.coeffs_meta.k.as_usize(),
        layout.input,
        layout.input_period,
        &v,
        &mut scratch.borrow(),
    );
    assert_guard::<BE, _>(
        "an input budget below the operand's scale",
        "ckks_linear_transformation_eval_sum_into",
        Budget,
        &mut alloc(case.rows, case.cols),
        |res| with_scratch::<BE, _>(bytes, |s| module.blas_vector_to_grid(res, line, &w, &low, &keys, s)),
    );
    // An operand writing row 0 of the first matrix at a period of `d` slots, below one block.
    let short = BlasMatVecOperandLayout {
        output: BlasVectorShape::new(3 * d, d, 0, 1),
        output_period: d,
        ..layout
    };
    let (_, w_short) = matvec_operands_of::<BE, F>(module, &short, case.entry);
    let short_keys = key_set(&short, 0, &mut scratch.borrow());
    let mut res = alloc(case.rows, case.cols);
    let short_bytes = module.blas_vector_to_grid_tmp_bytes(&res, &w_short, &a, &params.atk_layout());
    assert_guard::<BE, _>("an operand writing no matrix", op, InvalidOperand, &mut res, |res| {
        with_scratch::<BE, _>(short_bytes, |s| {
            module.blas_vector_to_grid(res, BlasGridLine::Row { mat_slot: 0, row: 0 }, &w_short, &a, &short_keys, s)
        })
    });
}
