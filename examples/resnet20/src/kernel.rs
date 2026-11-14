//! A convolution on tensors packed in the coefficients is a product by a sparse polynomial: the tap from input channel
//! `q` to output channel `o` is the monomial that moves each sample it reads onto the sample it writes. The reads of a
//! tap are a fixed number of coefficients apart, so they share one class of the coefficient indexes modulo that number,
//! and the last SlotsToCoeffs factor multiplies each class by its own polynomial: a convolution is one polynomial per
//! class, the sum of the monomials of the taps that read it.

use anyhow::{Result, ensure};

use crate::{layout::CoeffLayout, model::Window};

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
    fn new(t: usize, input: &CoeffLayout, window: Window) -> Self {
        let Window { kernel, padding, .. } = window;
        let (o, q) = (
            t / (input.channels() * kernel * kernel),
            t / (kernel * kernel) % input.channels(),
        );
        let (dk, dl) = (
            (t / kernel % kernel) as i64 - padding as i64,
            (t % kernel) as i64 - padding as i64,
        );
        let i0 = input.bases()[q] as i64 + input.step() as i64 * (dk * input.row_stride() as i64 + dl);
        Self { o, q, dk, dl, i0 }
    }
}

/// The polynomials of the convolution of window `window` from `input` to `output`, on the one-dimensional sample
/// index of their row stride as [`conv2d`](crate::model::conv2d), of `n` coefficients each.
///
/// The reads of a tap are `step * stride` coefficients apart and the last SlotsToCoeffs factor tells the indexes
/// apart modulo `cosets`: with `period` the greatest common divisor of the two, polynomial `r` multiplies the input
/// coefficients of index `r` modulo `period`, in place. The tap `(dk, dl)` from channel `q` to channel `o`
/// first reads coefficient `i0 = bases_in[q] + step * (dk * row_stride + dl)`: it sits in the polynomial of `i0`, at
/// exponent `bases_out[o] - i0`, negated past the ring's end.
///
/// `weights` is `[out][in][kernel][kernel]`, row-major, as PyTorch stores a convolution. Fails when the layouts do not
/// chain, or when a term of a nonzero weight lands on an output sample it does not compute.
pub(crate) fn kernel_polynomials(
    input: &CoeffLayout,
    output: &CoeffLayout,
    weights: &[f64],
    window: Window,
    cosets: usize,
) -> Result<Vec<Vec<f64>>> {
    let Window { kernel, stride, padding } = window;
    ensure!(
        input.n() == output.n() && input.row_stride() == output.row_stride(),
        "the layouts must share the ring and the row stride"
    );
    let strided = |side: usize| (side + 2 * padding).checked_sub(kernel).map(|room| room / stride + 1);
    ensure!(
        kernel > 0
            && stride > 0
            && output.step() == input.step() * stride
            && Some(output.height()) == strided(input.height())
            && Some(output.width()) == strided(input.width()),
        "the output grid must be the input grid under the window"
    );
    ensure!(
        weights.len() == output.channels() * input.channels() * kernel * kernel,
        "the weights must hold out * in * kernel * kernel values"
    );
    ensure!(
        cosets.is_power_of_two() && cosets <= input.n(),
        "the coset count must be a power of two"
    );
    let period = 1usize << (input.step() * stride).trailing_zeros().min(cosets.trailing_zeros());

    // Input samples `(q, i, j, index)` by the polynomial that reads them.
    let mut parts = vec![Vec::new(); period];
    for (q, i, j, idx) in input.samples() {
        parts[idx % period].push((q, i as i64, j as i64, idx as i64));
    }
    let occupied = output.occupied();
    let (n, row_stride) = (input.n() as i64, input.row_stride() as i64);
    let (stride, out_height, out_width) = (stride as i64, output.height() as i64, output.width() as i64);

    let mut polys = vec![vec![0.0; input.n()]; period];
    for (t, &w) in weights.iter().enumerate().filter(|&(_, &w)| w != 0.0) {
        let tap = Tap::new(t, input, window);
        let shift = output.bases()[tap.o] as i64 - tap.i0;
        let r = tap.i0.rem_euclid(period as i64) as usize;
        // The polynomial also multiplies the samples of its class that the tap does not read.
        for &(q, i, j, idx) in &parts[r] {
            let read = row_stride * (i - tap.dk) + j - tap.dl;
            let written = read.div_euclid(stride);
            let intended = q == tap.q
                && read.rem_euclid(stride) == 0
                && (0..out_height).contains(&written.div_euclid(row_stride))
                && (0..out_width).contains(&written.rem_euclid(row_stride));
            ensure!(
                intended || !occupied[(idx + shift).rem_euclid(n) as usize],
                "a term of the product lands on an output sample it does not compute"
            );
        }
        let e = shift.rem_euclid(2 * n);
        let (e, sign) = if e < n { (e, 1.0) } else { (e - n, -1.0) };
        polys[r][e as usize] += sign * w;
    }
    Ok(polys)
}

#[cfg(test)]
mod tests {
    use poulpy_hal::source::Source;

    use super::kernel_polynomials;
    use crate::{
        layout::CoeffLayout,
        model::{Window, conv2d, shortcut_weights},
        params::{Layouts, MERGED_COSETS},
    };

    const N: usize = 1 << 15;

    /// Integers in `[-20, 20)`, so every sum below is exact in `f64`.
    fn integers(len: usize, source: &mut Source) -> Vec<f64> {
        (0..len).map(|_| source.next_f64(-20.0, 20.0).floor()).collect()
    }

    /// The convolution in the clear: each input coefficient, at its own index, times the polynomial of its class.
    fn product(input: &[f64], polys: &[Vec<f64>]) -> Vec<f64> {
        let mut res = vec![0.0; N];
        for (idx, &x) in input.iter().enumerate().filter(|&(_, &x)| x != 0.0) {
            let poly = &polys[idx % polys.len()];
            for (e, &w) in poly.iter().enumerate().filter(|&(_, &w)| w != 0.0) {
                let d = idx + e;
                let (d, sign) = if d < N { (d, 1.0) } else { (d - N, -1.0) };
                res[d] += sign * w * x;
            }
        }
        res
    }

    /// Runs the convolution of `weights`, or of random integers, on a tensor of random integers, and checks it exactly.
    fn check(input: &CoeffLayout, output: &CoeffLayout, window: Window, weights: Option<Vec<f64>>) {
        let mut source = Source::new([7u8; 32]);
        let tensor = integers(input.len(), &mut source);
        let dense = output.channels() * input.channels() * window.kernel * window.kernel;
        let weights = weights.unwrap_or_else(|| integers(dense, &mut source));
        let polys = kernel_polynomials(input, output, &weights, window, MERGED_COSETS).unwrap();
        let res = product(&input.pack(&tensor).unwrap(), &polys);
        let want = conv2d(&tensor, &weights, input.height(), input.width(), input.row_stride(), window);
        assert_eq!(output.unpack(&res).unwrap(), want);
    }

    /// A `3 x 3` window, which keeps the grid at stride 1.
    fn conv(stride: usize) -> Window {
        Window {
            kernel: 3,
            stride,
            padding: 1,
        }
    }

    /// The `1 x 1` window of a shortcut, with its weights from `channels` to `out` channels.
    fn shortcut(stride: usize, channels: usize, out: usize) -> (Window, Option<Vec<f64>>) {
        let window = Window {
            kernel: 1,
            stride,
            padding: 0,
        };
        (window, Some(shortcut_weights(channels, out, 3.0)))
    }

    #[test]
    fn the_stem() {
        let l = Layouts::new(N).unwrap();
        check(&l.image, &l.stages[0], conv(1), None);
    }

    #[test]
    fn convolutions_of_a_stage() {
        let l = Layouts::new(N).unwrap();
        for stage in &l.stages {
            check(stage, stage, conv(1), None);
        }
    }

    #[test]
    fn stride_2_into_the_next_stage() {
        let l = Layouts::new(N).unwrap();
        for (input, output) in l.stages.iter().zip(&l.stages[1..]) {
            check(input, output, conv(2), None);
        }
    }

    #[test]
    fn shortcuts() {
        let l = Layouts::new(N).unwrap();
        for stage in &l.stages {
            let (window, weights) = shortcut(1, stage.channels(), stage.channels());
            check(stage, stage, window, weights);
        }
        for (input, output) in l.stages.iter().zip(&l.stages[1..]) {
            let (window, weights) = shortcut(2, input.channels(), output.channels());
            check(input, output, window, weights);
        }
    }

    #[test]
    fn the_head() {
        let l = Layouts::new(N).unwrap();
        let side = l.stages[2].height();
        let pool = Window {
            kernel: side,
            stride: side,
            padding: 0,
        };
        check(&l.stages[2], &l.logits, pool, None);
    }

    #[test]
    fn colliding_layouts_are_refused() {
        // Two grids in one coset: one polynomial reads both.
        let grids = CoeffLayout::new(N, 8, 8, 32, 64, vec![0, 64 * 16]).unwrap();
        let weights = vec![1.0; 2 * 2 * 9];
        assert!(kernel_polynomials(&grids, &grids, &weights, conv(1), MERGED_COSETS).is_err());
    }
}
