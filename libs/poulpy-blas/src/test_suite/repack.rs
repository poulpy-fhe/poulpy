//! Repacking between the lines of a grid's matrices and vectors.

use poulpy_ckks::{
    CKKSInfos, CoeffsMeta, SetGLWEPlaintextInfos,
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
    api::{CnvPVecAlloc, NegacyclicFFT, NegacyclicFFTNew, ScratchOwnedBorrow},
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
        alloc_vector, assert_vector_decrypt, assert_vector_guard, encrypt_vector, matvec_entry, matvec_model, matvec_operands_of,
        vector_entry, vector_snapshot,
    },
};
use crate::{
    api::{BlasEncodingOps, BlasMatVecOps},
    layouts::{
        BlasCiphertextOwned, BlasGridInfos, BlasGridLine, BlasGridMut, BlasLayout, BlasMatVecOperandLayout, BlasVectorShape,
        blas_grid_to_vector_key_uses, blas_grid_to_vector_output_layout, blas_vector_to_grid_key_uses,
        blas_vector_to_grid_output_layout, matvec_pt_galois_elements,
    },
    oep::{BlasEncodingImpl, BlasMatVecImpl},
};

/// `log2` of the block dimension of these tests.
const LOG_D: usize = 1;
const D: usize = 1 << LOG_D;

/// Entry `(s, r, c)` of the grids of these tests, dyadic in `[-1, 1]`, real.
fn grid_entry(s: usize, r: usize, c: usize) -> f64 {
    ((s * 11 + r * 5 + c * 3) % 17) as f64 / 8.0 - 1.0
}

/// The identity matrix, which extracts or places a line as it is.
fn identity_entry(row: usize, col: usize) -> f64 {
    f64::from(u8::from(row == col))
}

/// A grid-to-vector case: the line read, the operand's layout and its matrix, and the matrices the label of the grid
/// read proves, which records the two of [`GRID`].
pub(crate) struct GridToVectorCase {
    pub line: BlasGridLine,
    pub layout: BlasMatVecOperandLayout,
    pub entry: fn(usize, usize) -> f64,
    pub proven: usize,
}

impl GridToVectorCase {
    /// The label of the grid the case reads: `log2(slots / (proven d^2))`.
    pub(crate) fn label(&self, slots: usize) -> usize {
        (slots / (self.proven * D * D)).ilog2() as usize
    }
}

/// The `2 x 3` grids of `2 x 2` blocks recording two matrices the grid-to-vector cases read.
pub(crate) const GRID: (usize, usize, usize) = (2, 3, 2);

/// The grid-to-vector cases on `slots` slots, at least 16, at `params`' radix and precision, reading a grid of [`GRID`]
/// whose label proves both matrices: a row of the second matrix past the first block row through a public matrix into
/// one dense ciphertext, and through the identity, which extracts it; a column of the first matrix past the first block
/// column, read at a period longer than the grid's, into two ciphertexts repeating every 8 slots. Then a row of the
/// second matrix of a grid whose label proves one, past the period it proves, in its repetition.
pub(crate) fn grid_to_vector_cases(params: &CKKSTestParams, slots: usize) -> Vec<GridToVectorCase> {
    let (base2k, meta): (Base2K, CoeffsMeta) = (params.base2k.into(), params.prec().into());
    let (rows, cols, _) = GRID;
    let row = BlasGridLine::Row { mat_slot: 1, row: 3 };
    let col = BlasGridLine::Col { mat_slot: 0, col: 5 };
    let shape = |line: BlasGridLine| {
        let (len, offset, stride) = match line {
            BlasGridLine::Row { mat_slot, row } => (cols * D, mat_slot * D * D + (row % D) * D, 1),
            BlasGridLine::Col { mat_slot, col } => (rows * D, mat_slot * D * D + col % D, D),
        };
        BlasVectorShape::new(len, D, offset, stride)
    };
    let case = |line, p_in, output, p_out, entry, proven| GridToVectorCase {
        line,
        layout: BlasMatVecOperandLayout::new(slots, shape(line), p_in, output, p_out, base2k, meta),
        entry,
        proven,
    };
    let lanes = GRID.2;
    let repeated = BlasGridLine::Row { mat_slot: 1, row: 1 };
    vec![
        case(row, 8, BlasVectorShape::new(4, 4, 0, 1), slots, matvec_entry, lanes),
        case(row, 8, BlasVectorShape::new(6, 6, 2, 1), slots, identity_entry, lanes),
        case(col, 16, BlasVectorShape::new(3, 2, 1, 3), 8, matvec_entry, lanes),
        case(repeated, 8, BlasVectorShape::new(4, 4, 0, 1), slots, matvec_entry, 1),
    ]
}

/// The entries of `line` of `grid`, whose `proven` matrices repeat over the batch.
fn line_values(grid: &Grid, line: BlasGridLine, proven: usize) -> Vec<f64> {
    match line {
        BlasGridLine::Row { mat_slot, row } => (0..grid.cols * grid.d())
            .map(|c| grid.at(mat_slot % proven, row, c).0)
            .collect(),
        BlasGridLine::Col { mat_slot, col } => (0..grid.rows * grid.d())
            .map(|r| grid.at(mat_slot % proven, r, col).0)
            .collect(),
    }
}

/// The real grid of [`GRID`]'s dimensions holding [`grid_entry`] in `proven` matrices.
fn grid_model(proven: usize) -> Grid {
    let (rows, cols, _) = GRID;
    Grid::from_fn(LOG_D, rows, cols, proven, |s, r, c| (grid_entry(s, r, c), 0.0))
}

/// `dec(grid_to_vector(W, enc(a), line))` is `W x` for `x` the line of `a`, at every repetition of the output's period
/// and zero elsewhere, for rows and columns of a `2 x 3` grid of `2 x 2` blocks recording two matrices, past the first
/// block and matrix, through a public matrix and the identity, at an input period equal to and longer than the grid's,
/// and in a matrix the grid's label proves only as a repetition of the first, with the operand resident and streamed,
/// which give the same bits, on the operand's key set alone, at the metadata [`blas_grid_to_vector_output_layout`]
/// gives; `res` is one limb wider than the result. On a store answering only at the widths
/// [`blas_grid_to_vector_key_uses`] lists, it runs into `res` wider and narrower than the result. One byte short of its
/// query, the call panics before it writes `res`.
pub fn test_blas_grid_to_vector<BE, F, E>(params: CKKSTestParams, module: &Module<BE>, host_module: &Module<HostBytesBackend>)
where
    BE: TestContextBackend<Ring = Standard> + BlasMatVecImpl + BlasEncodingImpl,
    Module<BE>: TestContextModule<BE>
        + CKKSEncodingOps<BE, F>
        + BlasEncodingOps<BE, F>
        + CKKSLinearTransformationOps<BE>
        + CnvPVecAlloc<BE>,
    F: TestScalar + DiagonalArithmetic,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
{
    let slots = params.n / 2;
    if slots < 16 {
        eprintln!("grid_to_vector skipped: its vectors span 16 slots, the ring holds {slots}");
        return;
    }
    let encoder = ReferenceEncoder::<E>::new(slots).unwrap();
    let (sk_raw, sk) = gen_sk_with_raw(&params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, module);
    let p_w = params.prec().log_delta();
    for case in grid_to_vector_cases(&params, slots) {
        let (line, layout, entry, proven) = (case.line, case.layout, case.entry, case.proven);
        let label = format!("grid_to_vector {line:?} {layout:?}, {proven} matrices proven");
        let model = grid_model(proven);
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
        let x = line_values(&model, line, proven);
        let want: Vec<f64> = (0..layout.output.len)
            .map(|row| x.iter().enumerate().map(|(col, v)| entry(row, col) * v).sum())
            .collect();
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
        // Keys at the widths the key-use query lists and no other, into `res` wider and narrower than the result.
        for k in [res_k, params.k - p_w - params.base2k] {
            let mut res = alloc_vector(&params, module, layout.output, k);
            let uses = blas_grid_to_vector_key_uses(module, &res, &layout, &a);
            let strict = keys.at_widths(uses.rotations.iter().map(|&(p, k)| (p, k.as_usize())), []);
            let bytes = module.blas_grid_to_vector_tmp_bytes(&res, &resident, &a, &params.atk_layout());
            with_scratch::<BE, _>(bytes, |s| {
                module.blas_grid_to_vector(&mut res, &resident, &a, line, &strict, s)
            })
            .unwrap_or_else(|e| panic!("{label}: res.k={k} on the listed keys: {e}"));
        }
        let mut res = alloc_vector(&params, module, layout.output, res_k);
        let before = vector_snapshot::<BE>(&res);
        assert_short_scratch_panics::<BE, _>("blas_grid_to_vector", resident_bytes, |s| {
            module.blas_grid_to_vector(&mut res, &resident, &a, line, &keys, s)
        });
        assert!(vector_snapshot::<BE>(&res) == before, "{label}: wrote res before panicking");
    }
}

/// A vector-to-grid case: the line written, the dimensions of `res` and the operand's layout.
pub(crate) struct VectorToGridCase {
    pub line: BlasGridLine,
    pub rows: usize,
    pub cols: usize,
    pub layout: BlasMatVecOperandLayout,
}

/// The vector-to-grid cases on `slots` slots, at least 16, at `params`' radix and precision: a row of the second matrix
/// into a `1 x 3` grid holding two matrices from one dense ciphertext; a column of the third matrix into a `2 x 1` grid
/// holding four matrices from a vector repeating every 8 slots.
pub(crate) fn vector_to_grid_cases(params: &CKKSTestParams, slots: usize) -> Vec<VectorToGridCase> {
    let (base2k, meta): (Base2K, CoeffsMeta) = (params.base2k.into(), params.prec().into());
    let case = |line, rows, cols, input, p_in, output, p_out| VectorToGridCase {
        line,
        rows,
        cols,
        layout: BlasMatVecOperandLayout::new(slots, input, p_in, output, p_out, base2k, meta),
    };
    vec![
        case(
            BlasGridLine::Row { mat_slot: 1, row: 1 },
            1,
            3,
            BlasVectorShape::new(5, 5, 0, 1),
            slots,
            BlasVectorShape::new(3 * D, D, D * D + D, 1),
            2 * D * D,
        ),
        case(
            BlasGridLine::Col { mat_slot: 2, col: 1 },
            2,
            1,
            BlasVectorShape::new(3, 3, 1, 2),
            8,
            BlasVectorShape::new(2 * D, D, 2 * D * D + 1, D),
            4 * D * D,
        ),
    ]
}

/// The grid `res` of a vector-to-grid case holds: `y` as `line` of its matrix, every other entry of its `batch`
/// matrices zero.
fn placed(case: &VectorToGridCase, batch: usize, y: &[f64]) -> Grid {
    Grid::from_fn(LOG_D, case.rows, case.cols, batch, |s, r, c| match case.line {
        BlasGridLine::Row { mat_slot, row } if s == mat_slot && r == row => (y[c], 0.0),
        BlasGridLine::Col { mat_slot, col } if s == mat_slot && c == col => (y[r], 0.0),
        _ => (0.0, 0.0),
    })
}

/// `dec(vector_to_grid(line, W, enc(v)))` holds `W v` as the line of its matrix and zero elsewhere, the batch recorded
/// `output_period / d^2`, for a row of the second matrix into a `1 x 3` grid and a column of the third into a `2 x 1`
/// grid, from dense and repeating vectors, resident and streamed giving the same bits, on the operand's key set alone,
/// at the layout [`blas_vector_to_grid_output_layout`] gives; `res` is one limb wider than the result. On a store
/// answering only at the widths [`blas_vector_to_grid_key_uses`] lists, it runs into `res` wider and narrower than the
/// result. One byte short of its query, the call panics before it writes `res`.
pub fn test_blas_vector_to_grid<BE, F, E>(params: CKKSTestParams, module: &Module<BE>, host_module: &Module<HostBytesBackend>)
where
    BE: TestContextBackend<Ring = Standard> + BlasMatVecImpl + BlasEncodingImpl,
    Module<BE>: TestContextModule<BE>
        + CKKSEncodingOps<BE, F>
        + BlasEncodingOps<BE, F>
        + CKKSLinearTransformationOps<BE>
        + CnvPVecAlloc<BE>,
    F: TestScalar + DiagonalArithmetic,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
{
    let slots = params.n / 2;
    if slots < 16 {
        eprintln!("vector_to_grid skipped: its vectors span 16 slots, the ring holds {slots}");
        return;
    }
    let encoder = ReferenceEncoder::<E>::new(slots).unwrap();
    let (sk_raw, sk) = gen_sk_with_raw(&params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, module);
    let p_w = params.prec().log_delta();
    for case in vector_to_grid_cases(&params, slots) {
        let (line, layout) = (case.line, case.layout);
        let label = format!("vector_to_grid {line:?} {layout:?}");
        let (streamed, resident) = matvec_operands_of::<BE, F>(module, &layout, matvec_entry);
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
        let alloc = |k| alloc_grid(&params, module, LOG_D, case.rows, case.cols, k);
        let template = alloc(res_k);
        assert_eq!(line.shape(&template), layout.output, "{label}: line shape");
        let want_layout: BlasLayout = blas_vector_to_grid_output_layout(&template, &layout, &a);
        let want = placed(&case, layout.output_period / (D * D), &matvec_model(layout.output.len, &v));
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
        for k in [res_k, params.k - p_w - params.base2k] {
            let mut res = alloc(k);
            let uses = blas_vector_to_grid_key_uses(module, &res, &layout, &a);
            let strict = keys.at_widths(uses.rotations.iter().map(|&(p, k)| (p, k.as_usize())), []);
            let bytes = module.blas_vector_to_grid_tmp_bytes(&res, &resident, &a, &params.atk_layout());
            with_scratch::<BE, _>(bytes, |s| {
                module.blas_vector_to_grid(&mut res, line, &resident, &a, &strict, s)
            })
            .unwrap_or_else(|e| panic!("{label}: res.k={k} on the listed keys: {e}"));
        }
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
/// key of the operand's set, `a` and `res` both of another rank than the keys, a `res` of another shape, and an input
/// whose budget is below the operand's scale, which poulpy-ckks's summed transformation rejects. `blas_vector_to_grid`
/// rejects: a `res` of two block rows for a row or two block columns for a column, a row or a column past its block,
/// a matrix outside the slots' dense batch, a line of another shape than the operand's output, an operand whose output
/// period holds no matrix, every `res` of [`rejected_grids`] and [`foreign_layouts`], a `res` whose blocks do not tile
/// its shape, an input whose entries pass the period its label proves, a store lacking a key, `a` and `res` both of
/// another rank than the keys, and an input whose budget is below the operand's scale.
pub fn test_blas_repack_guards<BE, F, E>(params: CKKSTestParams, module: &Module<BE>, host_module: &Module<HostBytesBackend>)
where
    BE: TestContextBackend<Ring = Standard> + BlasMatVecImpl + BlasEncodingImpl,
    Module<BE>: TestContextModule<BE>
        + CKKSEncodingOps<BE, F>
        + BlasEncodingOps<BE, F>
        + CKKSLinearTransformationOps<BE>
        + CnvPVecAlloc<BE>,
    F: TestScalar + DiagonalArithmetic,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
{
    let slots = params.n / 2;
    if slots < 16 {
        eprintln!("repack_guards skipped: its vectors span 16 slots, the ring holds {slots}");
        return;
    }
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
    let case = grid_to_vector_cases(&params, slots).swap_remove(0);
    let (line, layout, entry) = (case.line, case.layout, case.entry);
    let (_, w) = matvec_operands_of::<BE, F>(module, &layout, entry);
    let keys = key_set(&layout, 0, &mut scratch.borrow());
    let encrypt = |model: &Grid, k: usize, label: usize, scratch: &mut ScratchArena<'_, BE>| {
        encrypt_grid::<BE, F, E>(&params, module, host_module, &encoder, &sk, k, label, model, scratch)
    };
    let model = grid_model(case.proven);
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
            row: rows * D,
        },
        &a,
        &keys,
    );
    call(
        "a matrix outside the batch",
        InvalidOperand,
        BlasGridLine::Row { mat_slot: lanes, row: 3 },
        &a,
        &keys,
    );
    call(
        "a column outside the grid",
        InvalidOperand,
        BlasGridLine::Col {
            mat_slot: 0,
            col: cols * D,
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
        BlasGridLine::Row { mat_slot: 1, row: 2 },
        &a,
        &keys,
    );
    for (what, bad) in rejected_grids(module, &a).into_iter().chain(foreign_layouts(module, &a)) {
        call(&format!("grid: {what}"), InvalidOperand, line, &bad, &keys);
    }
    let (grid_rows, grid_cols, _) = GRID;
    let dense = Grid::from_fn(LOG_D, grid_rows, grid_cols, slots / (D * D), |s, r, c| {
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
    let mut other_res = alloc_vector(&params, module, BlasVectorShape::new(4, 2, 0, 1), params.k);
    assert_vector_guard::<BE>("res of another shape", op, InvalidOperand, &mut other_res, |res| {
        with_scratch::<BE, _>(bytes, |s| module.blas_grid_to_vector(res, &w, &a, line, &keys, s))
    });
    // Budget: blocks at the operand's width, whose budget is below its scale.
    let low = encrypt(
        &model,
        layout.coeffs_meta.k.as_usize(),
        case.label(slots),
        &mut scratch.borrow(),
    );
    assert_vector_guard::<BE>(
        "an input budget below the operand's scale",
        "ckks_eval_linear_transformation_sum_into",
        Budget,
        &mut res,
        |res| with_scratch::<BE, _>(bytes, |s| module.blas_grid_to_vector(res, &w, &low, line, &keys, s)),
    );

    // Vector to grid, on its first case.
    let op = "blas_vector_to_grid";
    let case = vector_to_grid_cases(&params, slots).swap_remove(0);
    let (line, layout) = (case.line, case.layout);
    let (_, w) = matvec_operands_of::<BE, F>(module, &layout, matvec_entry);
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
    let alloc = |rows, cols| alloc_grid(&params, module, LOG_D, rows, cols, params.k);
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
        BlasGridLine::Row { mat_slot: 1, row: D },
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
        BlasGridLine::Col { mat_slot: 1, col: D },
        &mut alloc(2, 1),
        &a,
        &keys,
    );
    let dense_batch = slots / (D * D);
    call(
        "a row's matrix outside the dense batch",
        InvalidOperand,
        BlasGridLine::Row {
            mat_slot: dense_batch,
            row: 1,
        },
        &mut alloc(case.rows, case.cols),
        &a,
        &keys,
    );
    call(
        "a column's matrix outside the dense batch",
        InvalidOperand,
        BlasGridLine::Col {
            mat_slot: dense_batch,
            col: 1,
        },
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
        "ckks_eval_linear_transformation_sum_into",
        Budget,
        &mut alloc(case.rows, case.cols),
        |res| with_scratch::<BE, _>(bytes, |s| module.blas_vector_to_grid(res, line, &w, &low, &keys, s)),
    );
    // An operand writing row 0 of the first matrix at a period of 2 slots, below one block.
    let short = BlasMatVecOperandLayout {
        output: BlasVectorShape::new(3 * D, D, 0, 1),
        output_period: D,
        ..layout
    };
    let (_, w_short) = matvec_operands_of::<BE, F>(module, &short, matvec_entry);
    let short_keys = key_set(&short, 0, &mut scratch.borrow());
    let mut res = alloc(case.rows, case.cols);
    let short_bytes = module.blas_vector_to_grid_tmp_bytes(&res, &w_short, &a, &params.atk_layout());
    assert_guard::<BE, _>("an operand writing no matrix", op, InvalidOperand, &mut res, |res| {
        with_scratch::<BE, _>(short_bytes, |s| {
            module.blas_vector_to_grid(res, BlasGridLine::Row { mat_slot: 0, row: 0 }, &w_short, &a, &short_keys, s)
        })
    });
}
