use dashu_float::{FBig, round::mode::HalfEven};

use crate::layouts::{Backend, HostDataRef, VecZnx, VecZnxBig};

/// Summary statistics (max absolute value and standard deviation) of a
/// polynomial vector's decoded floating-point coefficients.
pub struct Stats {
    max: f64,
    std: f64,
}

impl Stats {
    /// Returns the maximum absolute coefficient value.
    pub fn max(&self) -> f64 {
        self.max
    }

    /// Returns the standard deviation of the coefficients.
    pub fn std(&self) -> f64 {
        self.std
    }
}

impl<D: HostDataRef> VecZnx<D> {
    /// Computes [`Stats`] (max absolute value and standard deviation) for
    /// column `col` by decoding all limbs into arbitrary-precision floats.
    pub fn stats(&self, base2k: usize, col: usize) -> Stats {
        let mut data: Vec<FBig<HalfEven>> = (0..self.n()).map(|_| FBig::ZERO).collect();
        self.decode_vec_float(base2k, col, &mut data);

        // std = sqrt(sum((xi - avg)^2) / n)
        let mut avg: FBig<HalfEven> = FBig::ZERO;
        let mut max: FBig<HalfEven> = FBig::ZERO;

        data.iter().for_each(|x| {
            avg = avg.clone() + x.clone();
            let abs_x = if x < &FBig::<HalfEven>::ZERO { -x.clone() } else { x.clone() };
            if abs_x > max {
                max = abs_x;
            }
        });
        avg /= FBig::from(data.len() as i64);
        data.iter_mut().for_each(|x| {
            *x = x.clone() - avg.clone();
        });
        let mut variance: FBig<HalfEven> = FBig::ZERO;
        data.iter().for_each(|x| {
            variance = variance.clone() + x.clone() * x.clone();
        });
        variance /= FBig::from(data.len() as i64);

        // Final output is f64; to_f64() always succeeds (returns nearest f64).
        // f64::try_from(FBig) fails with LossOfPrecision for nearly all values.
        Stats {
            std: variance.to_f64().value().sqrt(),
            max: max.to_f64().value(),
        }
    }
}

impl<D: HostDataRef, B: Backend + Backend<ScalarBig = i64>> VecZnxBig<D, B> {
    pub fn stats(&self, base2k: usize, col: usize) -> Stats {
        let shape = self.shape();
        let znx: VecZnx<&[u8]> =
            VecZnx::from_data_with_max_size(self.data.as_ref(), shape.n(), shape.cols(), shape.size(), shape.max_size());
        znx.stats(base2k, col)
    }
}
