use poulpy_ckks::api::CKKSEncodingScalar;
use poulpy_hal::api::{NegacyclicFFT, NegacyclicFFTNew};

struct Butterfly<F> {
    start: usize,
    half: usize,
    cos: F,
    sin: F,
    rotated: bool,
}

/// Independent schedule of the canonical CKKS butterfly dependency graph.
pub struct EncodingFFTTable<F> {
    m: usize,
    log_order: u32,
    butterflies: Vec<Butterfly<F>>,
}

fn reversed_fraction<F: CKKSEncodingScalar>(mut index: usize) -> F {
    let mut value = F::zero();
    let mut weight = F::from_f64(0.5).unwrap();
    while index != 0 {
        if index & 1 != 0 {
            value = value + weight;
        }
        weight = weight * F::from_f64(0.5).unwrap();
        index >>= 1;
    }
    value
}

impl<F: CKKSEncodingScalar> EncodingFFTTable<F> {
    fn push(&mut self, start: usize, half: usize, turn: F, rotated: bool) {
        let (cos, sin) = crate::ckks_roots::root_of_unity(turn, self.log_order);
        self.butterflies.push(Butterfly {
            start,
            half,
            cos,
            sin,
            rotated,
        });
    }

    fn schedule(&mut self, start: usize, m: usize, phase: F) {
        let two = F::one() + F::one();
        let four = two + two;
        if m > 2048 {
            self.push(start, m / 2, phase / two, false);
            self.schedule(start, m / 2, phase / two);
            self.schedule(start + m / 2, m / 2, phase / two + F::from_f64(0.5).unwrap());
        } else if m <= 16 {
            let mut half = m / 2;
            while half != 0 {
                for block in 0..m / (2 * half) {
                    let turn = phase / F::from_usize(m / half).unwrap() + reversed_fraction::<F>(block & !1) / two;
                    self.push(start + block * 2 * half, half, turn, block & 1 != 0);
                }
                half /= 2;
            }
        } else {
            let mut width = m;
            let mut phase = phase;
            if m.trailing_zeros() & 1 != 0 {
                phase = phase / two;
                width /= 2;
                self.push(start, width, phase, false);
            }
            while width > 16 {
                phase = phase / four;
                for block in 0..m / width {
                    let turn = phase + reversed_fraction::<F>(block) / four;
                    let offset = start + block * width;
                    self.push(offset, width / 2, two * turn, false);
                    self.push(offset, width / 4, turn, false);
                    self.push(offset + width / 2, width / 4, turn, true);
                }
                width /= 4;
            }
            for block in 0..m / 16 {
                self.schedule(start + 16 * block, 16, phase + reversed_fraction::<F>(block));
            }
        }
    }
}

impl<F: CKKSEncodingScalar> NegacyclicFFTNew<F> for EncodingFFTTable<F> {
    fn new(m: usize) -> Self {
        assert!(m.is_power_of_two());
        let mut table = Self {
            m,
            log_order: (4 * m).trailing_zeros().max(2),
            butterflies: Vec::new(),
        };
        table.schedule(0, m, F::from_f64(0.25).unwrap());
        table
    }
}

impl<F: CKKSEncodingScalar> NegacyclicFFT<F> for EncodingFFTTable<F> {
    fn m(&self) -> usize {
        self.m
    }

    fn fft(&self, data: &mut [F]) {
        assert_eq!(data.len(), 2 * self.m);
        let (re, im) = data.split_at_mut(self.m);
        for b in &self.butterflies {
            for a in b.start..b.start + b.half {
                let c = a + b.half;
                let (ar, ai, br, bi) = (re[a], im[a], re[c], im[c]);
                let (real, imag) = if b.rotated {
                    (br * b.sin + bi * b.cos, br * b.cos - bi * b.sin)
                } else {
                    (br * b.cos - bi * b.sin, br * b.sin + bi * b.cos)
                };
                re[a] = if b.rotated { ar - real } else { ar + real };
                re[c] = if b.rotated { ar + real } else { ar - real };
                im[a] = ai + imag;
                im[c] = ai - imag;
            }
        }
    }

    fn ifft(&self, data: &mut [F]) {
        assert_eq!(data.len(), 2 * self.m);
        let (re, im) = data.split_at_mut(self.m);
        for b in self.butterflies.iter().rev() {
            for a in b.start..b.start + b.half {
                let c = a + b.half;
                let (ar, ai, br, bi) = (re[a], im[a], re[c], im[c]);
                let (real, imag) = (ar - br, ai - bi);
                re[a] = ar + br;
                im[a] = ai + bi;
                re[c] = if b.rotated {
                    real * -b.sin + imag * b.cos
                } else {
                    real * b.cos - imag * -b.sin
                };
                im[c] = if b.rotated {
                    -real * b.cos + imag * -b.sin
                } else {
                    real * -b.sin + imag * b.cos
                };
            }
        }
    }
}
