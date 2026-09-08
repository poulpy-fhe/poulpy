use std::fmt::Display;

#[derive(Debug, Clone)]
pub struct HalSweepParms {
    pub n: usize,
    pub cols: usize,
    pub size: usize,
}

impl Display for HalSweepParms {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}x({}x{})", self.n, self.cols, self.size)
    }
}

pub struct VmpSweepParms {
    pub n: usize,
    pub rows: usize,
    pub cols_in: usize,
    pub cols_out: usize,
    pub size: usize,
}

impl Display for VmpSweepParms {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}x({}x{})x({}x{})",
            self.n, self.rows, self.cols_in, self.cols_out, self.size
        )
    }
}

pub struct CnvSweepParms {
    pub n: usize,
    pub size: usize,
}

impl Display for CnvSweepParms {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}x{}", self.n, self.size)
    }
}

/// Sweep parameters for the negacyclic reim FFT/IFFT (`m` is the transform
/// half-length passed to `NegacyclicFFTNew::new`; the transformed data has
/// length `2 * m`).
#[derive(Debug, Clone)]
pub struct ReimSweepParams {
    pub m: usize,
}

impl Display for ReimSweepParams {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "m={}", self.m)
    }
}

pub fn default_bench_params_hal() -> Vec<HalSweepParms> {
    vec![
        HalSweepParms {
            n: 1 << 10,
            cols: 2,
            size: 2,
        },
        HalSweepParms {
            n: 1 << 11,
            cols: 2,
            size: 4,
        },
        HalSweepParms {
            n: 1 << 12,
            cols: 2,
            size: 8,
        },
        HalSweepParms {
            n: 1 << 13,
            cols: 2,
            size: 16,
        },
        HalSweepParms {
            n: 1 << 14,
            cols: 2,
            size: 32,
        },
        HalSweepParms {
            n: 1 << 15,
            cols: 2,
            size: 64,
        },
    ]
}

pub fn default_bench_params_vmp() -> Vec<VmpSweepParms> {
    vec![
        VmpSweepParms {
            n: 1 << 10,
            rows: 2,
            cols_in: 1,
            cols_out: 2,
            size: 3,
        },
        VmpSweepParms {
            n: 1 << 11,
            rows: 4,
            cols_in: 1,
            cols_out: 2,
            size: 5,
        },
        VmpSweepParms {
            n: 1 << 12,
            rows: 7,
            cols_in: 1,
            cols_out: 2,
            size: 8,
        },
        VmpSweepParms {
            n: 1 << 13,
            rows: 15,
            cols_in: 1,
            cols_out: 2,
            size: 16,
        },
        VmpSweepParms {
            n: 1 << 14,
            rows: 31,
            cols_in: 1,
            cols_out: 2,
            size: 32,
        },
        VmpSweepParms {
            n: 1 << 15,
            rows: 63,
            cols_in: 1,
            cols_out: 2,
            size: 64,
        },
    ]
}

pub fn default_bench_params_cnv() -> Vec<CnvSweepParms> {
    vec![
        CnvSweepParms { n: 1 << 10, size: 2 },
        CnvSweepParms { n: 1 << 11, size: 4 },
        CnvSweepParms { n: 1 << 12, size: 8 },
        CnvSweepParms { n: 1 << 13, size: 16 },
        CnvSweepParms { n: 1 << 14, size: 32 },
        CnvSweepParms { n: 1 << 15, size: 64 },
    ]
}

/// Base conversion and displacement parameters for normalization benchmarks.
#[derive(Debug, Clone)]
pub struct NormalizeSweepParams {
    pub shape: HalSweepParms,
    pub a_base2k: usize,
    pub res_base2k: usize,
    pub res_offset: i64,
}

impl Display for NormalizeSweepParams {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}/k={}->{}/offset={}",
            self.shape, self.a_base2k, self.res_base2k, self.res_offset
        )
    }
}

pub fn default_bench_params_normalize() -> Vec<NormalizeSweepParams> {
    let mut result = Vec::new();
    for n in [1 << 12, 1 << 16] {
        for size in [4, 8, 16] {
            for a_base2k in [17, 19, 21, 50, 51] {
                for res_base2k in [17, 19, 21, 50, 51] {
                    for res_offset in [-(a_base2k as i64), 0, 1, a_base2k as i64] {
                        result.push(NormalizeSweepParams {
                            shape: HalSweepParms { n, cols: 1, size },
                            a_base2k,
                            res_base2k,
                            res_offset,
                        });
                    }
                }
            }
        }
    }
    result
}
