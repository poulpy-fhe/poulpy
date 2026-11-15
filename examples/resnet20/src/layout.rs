use anyhow::{Result, ensure};

/// Where a `channels x height x width` tensor sits in the coefficients of a degree-`n` ring.
///
/// Sample `(q, i, j)` sits at coefficient `bases[q] + step * (i * row_stride + j)`. Every other coefficient is zero
/// when the tensor is packed, which gives a convolution its zero padding wherever the grid leaves a gap.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct CoeffLayout {
    n: usize,
    height: usize,
    width: usize,
    row_stride: usize,
    step: usize,
    bases: Vec<usize>,
}

/// Real slot `s` of `slots` holds the coefficient of this index.
fn bit_reverse(s: usize, slots: usize) -> usize {
    s.reverse_bits() >> (usize::BITS - slots.trailing_zeros())
}

impl CoeffLayout {
    /// Fails unless `n` is a power of two, `row_stride >= width`, and every sample has its own coefficient below `n`.
    pub(crate) fn new(n: usize, height: usize, width: usize, row_stride: usize, step: usize, bases: Vec<usize>) -> Result<Self> {
        ensure!(n.is_power_of_two(), "the ring degree must be a power of two");
        ensure!(
            height > 0 && width > 0 && step > 0 && !bases.is_empty(),
            "the height, width, step and channel count must be positive"
        );
        ensure!(row_stride >= width, "the row stride must cover a row");
        let layout = Self {
            n,
            height,
            width,
            row_stride,
            step,
            bases,
        };
        let mut taken = vec![false; n];
        for (.., idx) in layout.samples() {
            ensure!(
                idx < n && !taken[idx],
                "two samples share a coefficient or a sample leaves the ring"
            );
            taken[idx] = true;
        }
        Ok(layout)
    }

    pub(crate) fn n(&self) -> usize {
        self.n
    }

    pub(crate) fn channels(&self) -> usize {
        self.bases.len()
    }

    pub(crate) fn height(&self) -> usize {
        self.height
    }

    pub(crate) fn width(&self) -> usize {
        self.width
    }

    /// The number of samples.
    pub(crate) fn len(&self) -> usize {
        self.channels() * self.height * self.width
    }

    /// Samples between the starts of two rows.
    pub(crate) fn row_stride(&self) -> usize {
        self.row_stride
    }

    /// Coefficients between two neighbouring samples of a row.
    pub(crate) fn step(&self) -> usize {
        self.step
    }

    /// Coefficient of each channel's first sample.
    pub(crate) fn bases(&self) -> &[usize] {
        &self.bases
    }

    /// The samples, as `(channel, row, column, coefficient)`, in the order of a `[channel][row][column]` tensor.
    pub(crate) fn samples(&self) -> impl Iterator<Item = (usize, usize, usize, usize)> + '_ {
        let rows = move |q| (0..self.height).map(move |i| (q, i));
        let row = move |(q, i)| (0..self.width).map(move |j| (q, i, j, self.bases[q] + self.step * (i * self.row_stride + j)));
        (0..self.channels()).flat_map(rows).flat_map(row)
    }

    /// Whether each of the `n` coefficients holds a sample.
    pub(crate) fn occupied(&self) -> Vec<bool> {
        let mut occupied = vec![false; self.n];
        for (.., idx) in self.samples() {
            occupied[idx] = true;
        }
        occupied
    }

    /// The `n` coefficients holding `tensor`, given as `[channel][row][column]`, zero elsewhere.
    pub(crate) fn pack(&self, tensor: &[f64]) -> Result<Vec<f64>> {
        ensure!(
            tensor.len() == self.len(),
            "the tensor must hold channels * height * width values"
        );
        let mut coeffs = vec![0.0; self.n];
        for ((.., idx), &value) in self.samples().zip(tensor) {
            coeffs[idx] = value;
        }
        Ok(coeffs)
    }

    /// The tensor, as `[channel][row][column]`, read from `n` coefficients.
    pub(crate) fn unpack(&self, coeffs: &[f64]) -> Result<Vec<f64>> {
        ensure!(coeffs.len() == self.n, "unpacking needs the n coefficients");
        Ok(self.samples().map(|(.., idx)| coeffs[idx]).collect())
    }

    /// The `n / 2` real slots that SlotsToCoeffs sends to the coefficients holding `tensor`. Real slots carry the lower
    /// half of the coefficients: fails on a layout with a sample in the upper half.
    pub(crate) fn pack_slots(&self, tensor: &[f64]) -> Result<Vec<f64>> {
        let (slots, coeffs) = (self.n / 2, self.pack(tensor)?);
        ensure!(
            self.samples().all(|(.., idx)| idx < slots),
            "real slots carry the lower half of the coefficients"
        );
        Ok((0..slots).map(|s| coeffs[bit_reverse(s, slots)]).collect())
    }
}
