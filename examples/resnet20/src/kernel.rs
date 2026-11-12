use anyhow::{Result, ensure};

use crate::layout::CoeffLayout;

/// The polynomials of a `kernel x kernel` convolution of stride `stride` from `input` to `output`, on the one-dimensional
/// sample index of their row stride, as [`conv2d`](crate::reference::conv2d), as `n` coefficients each, for an input in `Z[X^g]`, `g = 2^log_gap`: one per residue of the input index
/// `g * k` modulo `g * base`, `base` the smaller of `cosets` and the input step over `g` times the stride.
///
/// Polynomial `r` multiplies the input coefficients with `k = r` modulo `base`, in place. Tap `(dk, dl)` from input
/// channel `q` to output channel `o` reads coefficient `I0 + step * stride * (i * row_stride + j)` for output `(i, j)`,
/// with `I0 = bases_in[q] + step * (dk * row_stride + dl)`: it sits in polynomial `(I0 / g) mod base`, at exponent
/// `bases_out[o] - I0`, negated past the ring's end. The convolution computes every stride-1 position of an output
/// channel's grid; a stride keeps one in `stride^2`.
///
/// `weights` is `[out][in][kernel][kernel]`, row-major, as PyTorch stores a convolution. Fails before building anything
/// when the layouts do not chain, when the input leaves the subring, or when a term of a nonzero weight lands on an
/// output sample it does not compute.
pub(crate) fn kernel_polynomials(
    input: &CoeffLayout,
    output: &CoeffLayout,
    weights: &[f64],
    kernel: usize,
    stride: usize,
    cosets: usize,
    log_gap: usize,
) -> Result<Vec<Vec<f64>>> {
    ensure!(
        input.n() == output.n() && input.row_stride() == output.row_stride(),
        "the layouts must share the ring and the row stride"
    );
    ensure!(
        stride > 0
            && output.step() == input.step() * stride
            && output.height() == (input.height() - 1) / stride + 1
            && output.width() == (input.width() - 1) / stride + 1,
        "the output grid must be the input grid strided by `stride`"
    );
    ensure!(kernel % 2 == 1, "the kernel size must be odd");
    ensure!(
        weights.len() == output.channels() * input.channels() * kernel * kernel,
        "the weights must hold out * in * kernel * kernel values"
    );
    ensure!(
        cosets.is_power_of_two() && cosets <= input.n(),
        "the coset count must be a power of two"
    );
    ensure!(input.log_gap() >= log_gap, "the input must lie in the subring of the gap");
    let base = (input.step() * stride).min(cosets << log_gap);
    check_terms(input, output, weights, kernel, stride, base)?;

    let n = input.n() as i64;
    let mut polys = vec![vec![0.0; input.n()]; base >> log_gap];
    for (t, &w) in weights.iter().enumerate().filter(|&(_, &w)| w != 0.0) {
        let tap = Tap::new(t, input, kernel);
        let r = tap.i0.rem_euclid(base as i64) >> log_gap;
        let e = (output.bases()[tap.o] as i64 - tap.i0).rem_euclid(2 * n);
        let (e, sign) = if e < n { (e, 1.0) } else { (e - n, -1.0) };
        polys[r as usize][e as usize] += sign * w;
    }
    Ok(polys)
}

/// Weight `t` of `[out][in][kernel][kernel]`: output channel `o` reads input channel `q` at offset `(dk, dl)`, whose
/// first read is input coefficient `i0`.
struct Tap {
    o: usize,
    q: usize,
    dk: i64,
    dl: i64,
    i0: i64,
}

impl Tap {
    fn new(t: usize, input: &CoeffLayout, kernel: usize) -> Self {
        let half = (kernel / 2) as i64;
        let (o, q) = (
            t / (input.channels() * kernel * kernel),
            t / (kernel * kernel) % input.channels(),
        );
        let (dk, dl) = ((t / kernel % kernel) as i64 - half, (t % kernel) as i64 - half);
        let i0 = input.bases()[q] as i64 + input.step() as i64 * (dk * input.row_stride() as i64 + dl);
        Self { o, q, dk, dl, i0 }
    }
}

/// Fails when a term of a nonzero weight lands on an output sample other than the one it computes.
fn check_terms(
    input: &CoeffLayout,
    output: &CoeffLayout,
    weights: &[f64],
    kernel: usize,
    stride: usize,
    base: usize,
) -> Result<()> {
    let n = input.n() as i64;
    let mut useful = vec![false; input.n()];
    for o in 0..output.channels() {
        for i in 0..output.height() {
            for j in 0..output.width() {
                useful[output.index(o, i, j)] = true;
            }
        }
    }

    // Input samples `(q, i, j, index)` by the polynomial that reads them.
    let mut parts = vec![Vec::new(); base];
    for q in 0..input.channels() {
        for i in 0..input.height() {
            for j in 0..input.width() {
                let idx = input.index(q, i, j);
                parts[idx % base].push((q, i as i64, j as i64, idx as i64));
            }
        }
    }

    let (stride, out_height, out_width) = (stride as i64, output.height() as i64, output.width() as i64);
    let row_stride = input.row_stride() as i64;
    for (t, _) in weights.iter().enumerate().filter(|&(_, &w)| w != 0.0) {
        let tap = Tap::new(t, input, kernel);
        let shift = output.bases()[tap.o] as i64 - tap.i0;
        for &(q, i, j, idx) in &parts[tap.i0.rem_euclid(base as i64) as usize] {
            let read = row_stride * (i - tap.dk) + j - tap.dl;
            let written = read.div_euclid(stride);
            let intended = q == tap.q
                && read.rem_euclid(stride) == 0
                && (0..out_height).contains(&written.div_euclid(row_stride))
                && (0..out_width).contains(&written.rem_euclid(row_stride));
            ensure!(
                intended || !useful[(idx + shift).rem_euclid(n) as usize],
                "a term of the product lands on an output sample it does not compute"
            );
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use poulpy_hal::source::Source;

    use super::kernel_polynomials;
    use crate::{layout::CoeffLayout, network::Layouts, reference::conv2d};

    const N: usize = 1 << 15;
    /// The blocks of the last SlotsToCoeffs factor.
    const MERGED_COSETS: usize = 64;

    /// Integers in `[-20, 20)`, so every sum below is exact in `f64`.
    fn integers(len: usize, source: &mut Source) -> Vec<f64> {
        (0..len).map(|_| source.next_f64(-20.0, 20.0).floor()).collect()
    }

    /// The convolution in the clear: each input coefficient, at its own index, times the polynomial of its residue.
    fn product(input: &[f64], polys: &[Vec<f64>], log_gap: usize) -> Vec<f64> {
        let base = polys.len();
        let mut res = vec![0.0; N];
        for (idx, &x) in input.iter().enumerate().filter(|&(_, &x)| x != 0.0) {
            for (e, &w) in polys[(idx >> log_gap) % base].iter().enumerate().filter(|&(_, &w)| w != 0.0) {
                let d = idx + e;
                let (d, sign) = if d < N { (d, 1.0) } else { (d - N, -1.0) };
                res[d] += sign * w * x;
            }
        }
        res
    }

    /// Runs one convolution of random integers whose weights `keep(o, q, tap)` selects, and checks it exactly.
    fn check(
        input: &CoeffLayout,
        output: &CoeffLayout,
        kernel: usize,
        stride: usize,
        keep: impl Fn(usize, usize, usize) -> bool,
    ) {
        let (c, m) = (input.channels(), output.channels());
        let mut source = Source::new([7u8; 32]);
        let tensor = integers(c * input.height() * input.width(), &mut source);
        let weights: Vec<f64> = integers(m * c * kernel * kernel, &mut source)
            .into_iter()
            .enumerate()
            .map(|(t, w)| {
                let keep = keep(t / (c * kernel * kernel), t / (kernel * kernel) % c, t % (kernel * kernel));
                if keep { w } else { 0.0 }
            })
            .collect();

        let log_gap = input.log_gap();
        let polys = kernel_polynomials(input, output, &weights, kernel, stride, MERGED_COSETS, log_gap).unwrap();
        let res = product(&input.pack(&tensor).unwrap(), &polys, log_gap);
        let want = conv2d(
            &tensor,
            &weights,
            c,
            m,
            input.height(),
            input.width(),
            input.row_stride(),
            kernel,
            stride,
        );
        assert_eq!(output.unpack(&res).unwrap(), want);
    }

    fn all(_: usize, _: usize, _: usize) -> bool {
        true
    }

    /// Option A: output channel `o` copies input channel `o - in_channels / 2` at stride 2.
    fn option_a(in_channels: usize) -> impl Fn(usize, usize, usize) -> bool {
        move |o, q, _| o == q + in_channels / 2
    }

    #[test]
    fn stage1_convolution() {
        let l = Layouts::new(N).unwrap();
        check(&l.stage1, &l.stage1, 3, 1, all);
    }

    #[test]
    fn stride_2_into_stage2_with_its_shortcut() {
        let l = Layouts::new(N).unwrap();
        check(&l.stage1, &l.stage2, 3, 2, all);
        check(&l.stage1, &l.stage2, 1, 2, option_a(16));
    }

    #[test]
    fn stage2_convolution() {
        let l = Layouts::new(N).unwrap();
        check(&l.stage2, &l.stage2, 3, 1, all);
    }

    #[test]
    fn stride_2_into_stage3_with_its_shortcut() {
        let l = Layouts::new(N).unwrap();
        check(&l.stage2, &l.stage3, 3, 2, all);
        check(&l.stage2, &l.stage3, 1, 2, option_a(32));
    }

    #[test]
    fn stage3_convolution() {
        let l = Layouts::new(N).unwrap();
        check(&l.stage3, &l.stage3, 3, 1, all);
    }

    #[test]
    fn stride_1_shortcuts_are_identities() {
        let l = Layouts::new(N).unwrap();
        for layout in [&l.stage1, &l.stage2, &l.stage3] {
            check(layout, layout, 1, 1, |o, q, _| o == q);
        }
    }

    #[test]
    fn inputs_off_the_subring_are_refused() {
        let l = Layouts::new(N).unwrap();
        let weights = vec![1.0; 16 * 16 * 9];
        assert!(kernel_polynomials(&l.stage1, &l.stage1, &weights, 3, 1, MERGED_COSETS, 1).is_err());
    }

    #[test]
    fn colliding_layouts_are_refused() {
        // Two grids in one coset: one polynomial reads both.
        let grids = CoeffLayout::new(N, 8, 8, 32, 64, vec![0, 64 * 16]).unwrap();
        let weights = vec![1.0; 2 * 2 * 9];
        assert!(kernel_polynomials(&grids, &grids, &weights, 3, 1, MERGED_COSETS, 0).is_err());
    }
}
