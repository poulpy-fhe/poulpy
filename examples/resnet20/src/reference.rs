/// `kernel x kernel` convolution of stride `stride` and zero padding `kernel / 2`, on `[channel][row][column]` tensors
/// and `[out][in][kernel][kernel]` weights.
#[allow(clippy::too_many_arguments)]
pub(crate) fn conv2d(
    input: &[f64],
    weights: &[f64],
    in_channels: usize,
    out_channels: usize,
    height: usize,
    width: usize,
    kernel: usize,
    stride: usize,
) -> Vec<f64> {
    let half = (kernel / 2) as i64;
    let (out_height, out_width) = ((height - 1) / stride + 1, (width - 1) / stride + 1);
    let mut res = vec![0.0; out_channels * out_height * out_width];
    for o in 0..out_channels {
        for c in 0..in_channels {
            let taps = &weights[(o * in_channels + c) * kernel * kernel..][..kernel * kernel];
            let channel = &input[c * height * width..][..height * width];
            for i in 0..out_height {
                for j in 0..out_width {
                    let mut acc = 0.0;
                    for (t, &w) in taps.iter().enumerate() {
                        let si = (stride * i) as i64 + (t / kernel) as i64 - half;
                        let sj = (stride * j) as i64 + (t % kernel) as i64 - half;
                        if (0..height as i64).contains(&si) && (0..width as i64).contains(&sj) {
                            acc += w * channel[si as usize * width + sj as usize];
                        }
                    }
                    res[(o * out_height + i) * out_width + j] += acc;
                }
            }
        }
    }
    res
}
