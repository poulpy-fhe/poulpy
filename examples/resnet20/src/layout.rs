use anyhow::{Result, ensure};

/// Where a `channels x height x width` tensor sits in the coefficients of a degree-`n` ring.
///
/// Sample `(q, i, j)` sits at coefficient `bases[q] + step * (i * row_stride + j)`. Every other coefficient is zero
/// when the tensor is packed, which gives a convolution its zero padding wherever the grid leaves a gap.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CoeffLayout {
    n: usize,
    height: usize,
    width: usize,
    row_stride: usize,
    step: usize,
    bases: Vec<usize>,
}

impl CoeffLayout {
    /// Fails unless `n` is a power of two, `row_stride >= width`, and every sample has its own coefficient below `n`.
    pub fn new(n: usize, height: usize, width: usize, row_stride: usize, step: usize, bases: Vec<usize>) -> Result<Self> {
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
        for q in 0..layout.channels() {
            for i in 0..height {
                for j in 0..width {
                    let idx = layout.index(q, i, j);
                    ensure!(
                        idx < n && !taken[idx],
                        "two samples share a coefficient or a sample leaves the ring"
                    );
                    taken[idx] = true;
                }
            }
        }
        Ok(layout)
    }

    /// `log2` of the largest `g` whose multiples hold every sample.
    pub fn log_gap(&self) -> usize {
        self.bases.iter().fold(self.step, |gcd, &b| gcd | b).trailing_zeros() as usize
    }

    pub fn n(&self) -> usize {
        self.n
    }

    pub fn channels(&self) -> usize {
        self.bases.len()
    }

    pub fn height(&self) -> usize {
        self.height
    }

    pub fn width(&self) -> usize {
        self.width
    }

    /// Samples between the starts of two rows.
    pub fn row_stride(&self) -> usize {
        self.row_stride
    }

    /// Coefficients between two neighbouring samples of a row.
    pub fn step(&self) -> usize {
        self.step
    }

    /// Coefficient of each channel's first sample.
    pub fn bases(&self) -> &[usize] {
        &self.bases
    }

    /// Coefficient index of channel `q`, row `i`, column `j`.
    pub fn index(&self, q: usize, i: usize, j: usize) -> usize {
        self.bases[q] + self.step * (i * self.row_stride + j)
    }

    /// The `n` coefficients holding `tensor`, given as `[channel][row][column]`, zero elsewhere.
    pub fn pack(&self, tensor: &[f64]) -> Result<Vec<f64>> {
        ensure!(
            tensor.len() == self.channels() * self.height * self.width,
            "the tensor must hold channels * height * width values"
        );
        let mut coeffs = vec![0.0; self.n];
        for (q, channel) in tensor.chunks_exact(self.height * self.width).enumerate() {
            for (i, row) in channel.chunks_exact(self.width).enumerate() {
                for (j, &value) in row.iter().enumerate() {
                    coeffs[self.index(q, i, j)] = value;
                }
            }
        }
        Ok(coeffs)
    }

    /// The tensor, as `[channel][row][column]`, read from `n` coefficients.
    pub fn unpack(&self, coeffs: &[f64]) -> Result<Vec<f64>> {
        ensure!(coeffs.len() == self.n, "unpacking needs the n coefficients");
        let mut tensor = Vec::with_capacity(self.channels() * self.height * self.width);
        for q in 0..self.channels() {
            for i in 0..self.height {
                tensor.extend((0..self.width).map(|j| coeffs[self.index(q, i, j)]));
            }
        }
        Ok(tensor)
    }
}
