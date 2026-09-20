use super::*;
use poulpy_hal::layouts::PrimeSet;

// Four adjacent output sums share each key column: z[j] = sum_d sum_r x[d,r] K[r,j+d].
struct Contraction<'a> {
    n: usize,
    rows: usize,
    first_row: usize,
    row_count: usize,
    starts: [usize; 4],
    ends: [usize; 4],
    key_size: usize,
    output_size: usize,
    input: &'a [u64],
    key: &'a [u64],
    output: SendPtr<u64>,
}

pub(super) fn apply<E: TaskExecutor>(
    res: &mut VecZnxDftBackendMut<'_, crate::NTT3x42Ifma>,
    a: &VecZnxDftBackendRef<'_, crate::NTT3x42Ifma>,
    pmat: &VmpPMatBackendRef<'_, crate::NTT3x42Ifma>,
    zero_prefix: Option<usize>,
    tmp: &mut [u64],
) {
    let n = a.n();
    let input: &[u64] = cast_slice(a.data());
    let ends = std::array::from_fn(|d| ((a.size() + d) / 4).min(pmat.rows()));
    let starts: [usize; 4] = std::array::from_fn(|d| {
        zero_prefix.map_or_else(
            || {
                (0..ends[d])
                    .take_while(|&r| {
                        let start = (3 - d + 4 * r) * 2 * n;
                        input[start..start + 2 * n].iter().all(|&v| v == 0)
                    })
                    .count()
            },
            |prefix| ((prefix + d) / 4).min(ends[d]),
        )
    });
    let output_size = res.size();
    let output: &mut [u64] = &mut cast_slice_mut::<_, u64>(res.data_mut())[..4 * n * output_size];
    let row_count = *ends.iter().max().unwrap();
    let first_row = *starts.iter().min().unwrap();
    if output_size == 0 {
        return;
    }
    if first_row == row_count || pmat.size() == 0 {
        output.fill(0);
        return;
    }
    let contraction = Contraction {
        n,
        rows: pmat.rows(),
        first_row,
        row_count,
        starts,
        ends,
        key_size: pmat.size(),
        output_size,
        input,
        key: cast_slice(pmat.data()),
        output: SendPtr(output.as_mut_ptr()),
    };
    let tmp = &mut tmp[16..];
    let per_worker = 2 * row_count * 24;
    let groups = n / 8;
    if E::is_parallel() {
        let tasks = groups.min(8);
        E::for_each_chunked(tasks, tmp, per_worker, |tmp, task| unsafe {
            contraction.run::<true>(groups * task / tasks, groups * (task + 1) / tasks, tmp);
        });
    } else {
        unsafe { contraction.run::<false>(0, groups, &mut tmp[..per_worker]) };
    }
}

impl Contraction<'_> {
    #[target_feature(enable = "avx512ifma,avx512vl")]
    unsafe fn run<const STREAM: bool>(&self, begin: usize, end: usize, _tmp: &mut [u64]) {
        unsafe {
            let zero = _mm512_setzero_si512();
            let mut input = [0u64; 4 * 8 * 24];
            for bq in begin..end {
                for r in self.first_row..self.row_count {
                    for d in 0..4 {
                        let v = if r >= self.starts[d] && r < self.ends[d] {
                            extract_blk_quad_prime_major_row(self.n, bq, 3 - d + 4 * r, self.input)
                        } else {
                            [zero; 3]
                        };
                        let dst = input.as_mut_ptr().add((r * 4 + d) * 24);
                        for (p, v) in v.into_iter().enumerate() {
                            _mm512_storeu_si512(dst.add(p * 8).cast(), v);
                        }
                    }
                }
                for component in 0..2 {
                    contract::<STREAM>(
                        self.key
                            .as_ptr()
                            .add((bq * 2 * self.key_size + component) * self.rows * 16 + self.first_row * 16),
                        self.key_size,
                        self.rows * 256,
                        input.as_ptr().add(self.first_row * 4 * 24),
                        self.row_count - self.first_row,
                        self.output.get().add(component * 2 * self.n + 16 * bq),
                        self.n * 32,
                        self.output_size,
                    );
                }
            }
            if STREAM {
                _mm_sfence();
            }
        }
    }
}

const REDUCTION: [u64; 19] = {
    let mut c = [0u64; 19];
    c[0] = (1 << 42) - 1;
    c[1] = (1 << 20) - 1;
    c[2] = (1 << 22) - 1;
    c[3] = (1 << 52) - 1;
    let mut p = 0;
    while p < 3 {
        let q = Primes42::Q[p];
        let pow52 = (1u64 << 52) % q;
        c[4 + 5 * p] = (1u64 << 42) - q;
        c[5 + 5 * p] = pow52;
        c[6 + 5 * p] = ((pow52 as u128 * (1u128 << 52)) / (q as u128)) as u64;
        c[7 + 5 * p] = q.wrapping_neg();
        c[8 + 5 * p] = q;
        p += 1;
    }
    c
};
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx512ifma,avx512vl")]
unsafe fn contract<const STREAM: bool>(
    key: *const u64,
    key_size: usize,
    key_stride: usize,
    input: *const u64,
    rows: usize,
    output: *mut u64,
    output_stride: usize,
    size: usize,
) {
    unsafe {
        // Each rolling sum owns six registers: low/high accumulators for three primes.
        macro_rules! run { ($store0:literal,$store1:literal) => { core::arch::asm!(
            "vpxorq zmm0, zmm0, zmm0",
            "vpxorq zmm1, zmm1, zmm1",
            "vpxorq zmm2, zmm2, zmm2",
            "vpxorq zmm3, zmm3, zmm3",
            "vpxorq zmm4, zmm4, zmm4",
            "vpxorq zmm5, zmm5, zmm5",
            "vpxorq zmm6, zmm6, zmm6",
            "vpxorq zmm7, zmm7, zmm7",
            "vpxorq zmm8, zmm8, zmm8",
            "vpxorq zmm9, zmm9, zmm9",
            "vpxorq zmm10, zmm10, zmm10",
            "vpxorq zmm11, zmm11, zmm11",
            "vpxorq zmm12, zmm12, zmm12",
            "vpxorq zmm13, zmm13, zmm13",
            "vpxorq zmm14, zmm14, zmm14",
            "vpxorq zmm15, zmm15, zmm15",
            "vpxorq zmm16, zmm16, zmm16",
            "vpxorq zmm17, zmm17, zmm17",
            "vpxorq zmm18, zmm18, zmm18",
            "vpxorq zmm19, zmm19, zmm19",
            "vpxorq zmm20, zmm20, zmm20",
            "vpxorq zmm21, zmm21, zmm21",
            "vpxorq zmm22, zmm22, zmm22",
            "vpxorq zmm23, zmm23, zmm23",
            "vpbroadcastq zmm30, [{constants}]",
            "vpbroadcastq zmm31, [{constants}+8]",
            "xor {k:e}, {k:e}",
            "2:",
            "cmp {key}, {key_end}",
            "jae 22f",
            "mov {yptr}, {key}",
            "mov {xptr}, {input}",
            "mov {remaining}, {rows}",
            "20:",
            "vmovdqu64 zmm24, [{yptr}]",
            "vmovdqu64 zmm25, [{yptr}+64]",
            "vpandq zmm26, zmm24, zmm30",
            "vmovdqu64 zmm27, [{xptr}+0]",
            "vpmadd52luq zmm0, zmm27, zmm26",
            "vpmadd52huq zmm1, zmm27, zmm26",
            "vmovdqu64 zmm27, [{xptr}+192]",
            "vpmadd52luq zmm18, zmm27, zmm26",
            "vpmadd52huq zmm19, zmm27, zmm26",
            "vmovdqu64 zmm27, [{xptr}+384]",
            "vpmadd52luq zmm12, zmm27, zmm26",
            "vpmadd52huq zmm13, zmm27, zmm26",
            "vmovdqu64 zmm27, [{xptr}+576]",
            "vpmadd52luq zmm6, zmm27, zmm26",
            "vpmadd52huq zmm7, zmm27, zmm26",
            "vpsrlq zmm26, zmm24, 42",
            "vpandq zmm24, zmm25, zmm31",
            "vpsllq zmm24, zmm24, 22",
            "vporq zmm26, zmm26, zmm24",
            "vmovdqu64 zmm27, [{xptr}+64]",
            "vpmadd52luq zmm2, zmm27, zmm26",
            "vpmadd52huq zmm3, zmm27, zmm26",
            "vmovdqu64 zmm27, [{xptr}+256]",
            "vpmadd52luq zmm20, zmm27, zmm26",
            "vpmadd52huq zmm21, zmm27, zmm26",
            "vmovdqu64 zmm27, [{xptr}+448]",
            "vpmadd52luq zmm14, zmm27, zmm26",
            "vpmadd52huq zmm15, zmm27, zmm26",
            "vmovdqu64 zmm27, [{xptr}+640]",
            "vpmadd52luq zmm8, zmm27, zmm26",
            "vpmadd52huq zmm9, zmm27, zmm26",
            "vpsrlq zmm26, zmm25, 20",
            "vmovdqu64 zmm27, [{xptr}+128]",
            "vpmadd52luq zmm4, zmm27, zmm26",
            "vpmadd52huq zmm5, zmm27, zmm26",
            "vmovdqu64 zmm27, [{xptr}+320]",
            "vpmadd52luq zmm22, zmm27, zmm26",
            "vpmadd52huq zmm23, zmm27, zmm26",
            "vmovdqu64 zmm27, [{xptr}+512]",
            "vpmadd52luq zmm16, zmm27, zmm26",
            "vpmadd52huq zmm17, zmm27, zmm26",
            "vmovdqu64 zmm27, [{xptr}+704]",
            "vpmadd52luq zmm10, zmm27, zmm26",
            "vpmadd52huq zmm11, zmm27, zmm26",
            "add {yptr}, 128",
            "add {xptr}, 768",
            "dec {remaining}",
            "jnz 20b",
            "22:",
            "cmp {k}, 3",
            "jb 23f",
            "vpbroadcastq zmm24, [{constants}+32]",
            "vpsrlq zmm25, zmm6, 42",
            "vpandq zmm6, zmm6, zmm30",
            "vpmadd52luq zmm6, zmm25, zmm24",
            "vpsrlq zmm25, zmm6, 42",
            "vpandq zmm6, zmm6, zmm30",
            "vpmadd52luq zmm6, zmm25, zmm24",
            "vpbroadcastq zmm24, [{constants}+48]",
            "vpxorq zmm25, zmm25, zmm25",
            "vpmadd52huq zmm25, zmm7, zmm24",
            "vpbroadcastq zmm24, [{constants}+40]",
            "vpxorq zmm26, zmm26, zmm26",
            "vpmadd52luq zmm26, zmm7, zmm24",
            "vpbroadcastq zmm24, [{constants}+56]",
            "vpmadd52luq zmm26, zmm25, zmm24",
            "vpbroadcastq zmm24, [{constants}+24]",
            "vpandq zmm26, zmm26, zmm24",
            "vpaddq zmm6, zmm6, zmm26",
            "vpbroadcastq zmm24, [{constants}+64]",
            "vpsllq zmm25, zmm24, 1",
            "vpsubq zmm26, zmm6, zmm25",
            "vpminuq zmm6, zmm6, zmm26",
            "vpsubq zmm26, zmm6, zmm24",
            "vpminuq zmm6, zmm6, zmm26",
            "vpbroadcastq zmm24, [{constants}+72]",
            "vpsrlq zmm25, zmm8, 42",
            "vpandq zmm8, zmm8, zmm30",
            "vpmadd52luq zmm8, zmm25, zmm24",
            "vpsrlq zmm25, zmm8, 42",
            "vpandq zmm8, zmm8, zmm30",
            "vpmadd52luq zmm8, zmm25, zmm24",
            "vpbroadcastq zmm24, [{constants}+88]",
            "vpxorq zmm25, zmm25, zmm25",
            "vpmadd52huq zmm25, zmm9, zmm24",
            "vpbroadcastq zmm24, [{constants}+80]",
            "vpxorq zmm26, zmm26, zmm26",
            "vpmadd52luq zmm26, zmm9, zmm24",
            "vpbroadcastq zmm24, [{constants}+96]",
            "vpmadd52luq zmm26, zmm25, zmm24",
            "vpbroadcastq zmm24, [{constants}+24]",
            "vpandq zmm26, zmm26, zmm24",
            "vpaddq zmm8, zmm8, zmm26",
            "vpbroadcastq zmm24, [{constants}+104]",
            "vpsllq zmm25, zmm24, 1",
            "vpsubq zmm26, zmm8, zmm25",
            "vpminuq zmm8, zmm8, zmm26",
            "vpsubq zmm26, zmm8, zmm24",
            "vpminuq zmm8, zmm8, zmm26",
            "vpbroadcastq zmm24, [{constants}+112]",
            "vpsrlq zmm25, zmm10, 42",
            "vpandq zmm10, zmm10, zmm30",
            "vpmadd52luq zmm10, zmm25, zmm24",
            "vpsrlq zmm25, zmm10, 42",
            "vpandq zmm10, zmm10, zmm30",
            "vpmadd52luq zmm10, zmm25, zmm24",
            "vpbroadcastq zmm24, [{constants}+128]",
            "vpxorq zmm25, zmm25, zmm25",
            "vpmadd52huq zmm25, zmm11, zmm24",
            "vpbroadcastq zmm24, [{constants}+120]",
            "vpxorq zmm26, zmm26, zmm26",
            "vpmadd52luq zmm26, zmm11, zmm24",
            "vpbroadcastq zmm24, [{constants}+136]",
            "vpmadd52luq zmm26, zmm25, zmm24",
            "vpbroadcastq zmm24, [{constants}+24]",
            "vpandq zmm26, zmm26, zmm24",
            "vpaddq zmm10, zmm10, zmm26",
            "vpbroadcastq zmm24, [{constants}+144]",
            "vpsllq zmm25, zmm24, 1",
            "vpsubq zmm26, zmm10, zmm25",
            "vpminuq zmm10, zmm10, zmm26",
            "vpsubq zmm26, zmm10, zmm24",
            "vpminuq zmm10, zmm10, zmm26",
            "vpbroadcastq zmm24, [{constants}+16]",
            "vpandq zmm24, zmm8, zmm24",
            "vpsllq zmm24, zmm24, 42",
            "vporq zmm24, zmm24, zmm6",
            "vpsrlq zmm25, zmm8, 22",
            "vpsllq zmm26, zmm10, 20",
            "vporq zmm25, zmm25, zmm26",
            $store0,
            $store1,
            "add {output}, {output_stride}",
            "23:",
            "vpxorq zmm6, zmm6, zmm6",
            "vpxorq zmm7, zmm7, zmm7",
            "vpxorq zmm8, zmm8, zmm8",
            "vpxorq zmm9, zmm9, zmm9",
            "vpxorq zmm10, zmm10, zmm10",
            "vpxorq zmm11, zmm11, zmm11",
            "add {key}, {key_stride}",
            "inc {k}",
            "cmp {k}, {limit}",
            "jae 90f",
            "cmp {key}, {key_end}",
            "jae 32f",
            "mov {yptr}, {key}",
            "mov {xptr}, {input}",
            "mov {remaining}, {rows}",
            "30:",
            "vmovdqu64 zmm24, [{yptr}]",
            "vmovdqu64 zmm25, [{yptr}+64]",
            "vpandq zmm26, zmm24, zmm30",
            "vmovdqu64 zmm27, [{xptr}+0]",
            "vpmadd52luq zmm6, zmm27, zmm26",
            "vpmadd52huq zmm7, zmm27, zmm26",
            "vmovdqu64 zmm27, [{xptr}+192]",
            "vpmadd52luq zmm0, zmm27, zmm26",
            "vpmadd52huq zmm1, zmm27, zmm26",
            "vmovdqu64 zmm27, [{xptr}+384]",
            "vpmadd52luq zmm18, zmm27, zmm26",
            "vpmadd52huq zmm19, zmm27, zmm26",
            "vmovdqu64 zmm27, [{xptr}+576]",
            "vpmadd52luq zmm12, zmm27, zmm26",
            "vpmadd52huq zmm13, zmm27, zmm26",
            "vpsrlq zmm26, zmm24, 42",
            "vpandq zmm24, zmm25, zmm31",
            "vpsllq zmm24, zmm24, 22",
            "vporq zmm26, zmm26, zmm24",
            "vmovdqu64 zmm27, [{xptr}+64]",
            "vpmadd52luq zmm8, zmm27, zmm26",
            "vpmadd52huq zmm9, zmm27, zmm26",
            "vmovdqu64 zmm27, [{xptr}+256]",
            "vpmadd52luq zmm2, zmm27, zmm26",
            "vpmadd52huq zmm3, zmm27, zmm26",
            "vmovdqu64 zmm27, [{xptr}+448]",
            "vpmadd52luq zmm20, zmm27, zmm26",
            "vpmadd52huq zmm21, zmm27, zmm26",
            "vmovdqu64 zmm27, [{xptr}+640]",
            "vpmadd52luq zmm14, zmm27, zmm26",
            "vpmadd52huq zmm15, zmm27, zmm26",
            "vpsrlq zmm26, zmm25, 20",
            "vmovdqu64 zmm27, [{xptr}+128]",
            "vpmadd52luq zmm10, zmm27, zmm26",
            "vpmadd52huq zmm11, zmm27, zmm26",
            "vmovdqu64 zmm27, [{xptr}+320]",
            "vpmadd52luq zmm4, zmm27, zmm26",
            "vpmadd52huq zmm5, zmm27, zmm26",
            "vmovdqu64 zmm27, [{xptr}+512]",
            "vpmadd52luq zmm22, zmm27, zmm26",
            "vpmadd52huq zmm23, zmm27, zmm26",
            "vmovdqu64 zmm27, [{xptr}+704]",
            "vpmadd52luq zmm16, zmm27, zmm26",
            "vpmadd52huq zmm17, zmm27, zmm26",
            "add {yptr}, 128",
            "add {xptr}, 768",
            "dec {remaining}",
            "jnz 30b",
            "32:",
            "cmp {k}, 3",
            "jb 33f",
            "vpbroadcastq zmm24, [{constants}+32]",
            "vpsrlq zmm25, zmm12, 42",
            "vpandq zmm12, zmm12, zmm30",
            "vpmadd52luq zmm12, zmm25, zmm24",
            "vpsrlq zmm25, zmm12, 42",
            "vpandq zmm12, zmm12, zmm30",
            "vpmadd52luq zmm12, zmm25, zmm24",
            "vpbroadcastq zmm24, [{constants}+48]",
            "vpxorq zmm25, zmm25, zmm25",
            "vpmadd52huq zmm25, zmm13, zmm24",
            "vpbroadcastq zmm24, [{constants}+40]",
            "vpxorq zmm26, zmm26, zmm26",
            "vpmadd52luq zmm26, zmm13, zmm24",
            "vpbroadcastq zmm24, [{constants}+56]",
            "vpmadd52luq zmm26, zmm25, zmm24",
            "vpbroadcastq zmm24, [{constants}+24]",
            "vpandq zmm26, zmm26, zmm24",
            "vpaddq zmm12, zmm12, zmm26",
            "vpbroadcastq zmm24, [{constants}+64]",
            "vpsllq zmm25, zmm24, 1",
            "vpsubq zmm26, zmm12, zmm25",
            "vpminuq zmm12, zmm12, zmm26",
            "vpsubq zmm26, zmm12, zmm24",
            "vpminuq zmm12, zmm12, zmm26",
            "vpbroadcastq zmm24, [{constants}+72]",
            "vpsrlq zmm25, zmm14, 42",
            "vpandq zmm14, zmm14, zmm30",
            "vpmadd52luq zmm14, zmm25, zmm24",
            "vpsrlq zmm25, zmm14, 42",
            "vpandq zmm14, zmm14, zmm30",
            "vpmadd52luq zmm14, zmm25, zmm24",
            "vpbroadcastq zmm24, [{constants}+88]",
            "vpxorq zmm25, zmm25, zmm25",
            "vpmadd52huq zmm25, zmm15, zmm24",
            "vpbroadcastq zmm24, [{constants}+80]",
            "vpxorq zmm26, zmm26, zmm26",
            "vpmadd52luq zmm26, zmm15, zmm24",
            "vpbroadcastq zmm24, [{constants}+96]",
            "vpmadd52luq zmm26, zmm25, zmm24",
            "vpbroadcastq zmm24, [{constants}+24]",
            "vpandq zmm26, zmm26, zmm24",
            "vpaddq zmm14, zmm14, zmm26",
            "vpbroadcastq zmm24, [{constants}+104]",
            "vpsllq zmm25, zmm24, 1",
            "vpsubq zmm26, zmm14, zmm25",
            "vpminuq zmm14, zmm14, zmm26",
            "vpsubq zmm26, zmm14, zmm24",
            "vpminuq zmm14, zmm14, zmm26",
            "vpbroadcastq zmm24, [{constants}+112]",
            "vpsrlq zmm25, zmm16, 42",
            "vpandq zmm16, zmm16, zmm30",
            "vpmadd52luq zmm16, zmm25, zmm24",
            "vpsrlq zmm25, zmm16, 42",
            "vpandq zmm16, zmm16, zmm30",
            "vpmadd52luq zmm16, zmm25, zmm24",
            "vpbroadcastq zmm24, [{constants}+128]",
            "vpxorq zmm25, zmm25, zmm25",
            "vpmadd52huq zmm25, zmm17, zmm24",
            "vpbroadcastq zmm24, [{constants}+120]",
            "vpxorq zmm26, zmm26, zmm26",
            "vpmadd52luq zmm26, zmm17, zmm24",
            "vpbroadcastq zmm24, [{constants}+136]",
            "vpmadd52luq zmm26, zmm25, zmm24",
            "vpbroadcastq zmm24, [{constants}+24]",
            "vpandq zmm26, zmm26, zmm24",
            "vpaddq zmm16, zmm16, zmm26",
            "vpbroadcastq zmm24, [{constants}+144]",
            "vpsllq zmm25, zmm24, 1",
            "vpsubq zmm26, zmm16, zmm25",
            "vpminuq zmm16, zmm16, zmm26",
            "vpsubq zmm26, zmm16, zmm24",
            "vpminuq zmm16, zmm16, zmm26",
            "vpbroadcastq zmm24, [{constants}+16]",
            "vpandq zmm24, zmm14, zmm24",
            "vpsllq zmm24, zmm24, 42",
            "vporq zmm24, zmm24, zmm12",
            "vpsrlq zmm25, zmm14, 22",
            "vpsllq zmm26, zmm16, 20",
            "vporq zmm25, zmm25, zmm26",
            $store0,
            $store1,
            "add {output}, {output_stride}",
            "33:",
            "vpxorq zmm12, zmm12, zmm12",
            "vpxorq zmm13, zmm13, zmm13",
            "vpxorq zmm14, zmm14, zmm14",
            "vpxorq zmm15, zmm15, zmm15",
            "vpxorq zmm16, zmm16, zmm16",
            "vpxorq zmm17, zmm17, zmm17",
            "add {key}, {key_stride}",
            "inc {k}",
            "cmp {k}, {limit}",
            "jae 90f",
            "cmp {key}, {key_end}",
            "jae 42f",
            "mov {yptr}, {key}",
            "mov {xptr}, {input}",
            "mov {remaining}, {rows}",
            "40:",
            "vmovdqu64 zmm24, [{yptr}]",
            "vmovdqu64 zmm25, [{yptr}+64]",
            "vpandq zmm26, zmm24, zmm30",
            "vmovdqu64 zmm27, [{xptr}+0]",
            "vpmadd52luq zmm12, zmm27, zmm26",
            "vpmadd52huq zmm13, zmm27, zmm26",
            "vmovdqu64 zmm27, [{xptr}+192]",
            "vpmadd52luq zmm6, zmm27, zmm26",
            "vpmadd52huq zmm7, zmm27, zmm26",
            "vmovdqu64 zmm27, [{xptr}+384]",
            "vpmadd52luq zmm0, zmm27, zmm26",
            "vpmadd52huq zmm1, zmm27, zmm26",
            "vmovdqu64 zmm27, [{xptr}+576]",
            "vpmadd52luq zmm18, zmm27, zmm26",
            "vpmadd52huq zmm19, zmm27, zmm26",
            "vpsrlq zmm26, zmm24, 42",
            "vpandq zmm24, zmm25, zmm31",
            "vpsllq zmm24, zmm24, 22",
            "vporq zmm26, zmm26, zmm24",
            "vmovdqu64 zmm27, [{xptr}+64]",
            "vpmadd52luq zmm14, zmm27, zmm26",
            "vpmadd52huq zmm15, zmm27, zmm26",
            "vmovdqu64 zmm27, [{xptr}+256]",
            "vpmadd52luq zmm8, zmm27, zmm26",
            "vpmadd52huq zmm9, zmm27, zmm26",
            "vmovdqu64 zmm27, [{xptr}+448]",
            "vpmadd52luq zmm2, zmm27, zmm26",
            "vpmadd52huq zmm3, zmm27, zmm26",
            "vmovdqu64 zmm27, [{xptr}+640]",
            "vpmadd52luq zmm20, zmm27, zmm26",
            "vpmadd52huq zmm21, zmm27, zmm26",
            "vpsrlq zmm26, zmm25, 20",
            "vmovdqu64 zmm27, [{xptr}+128]",
            "vpmadd52luq zmm16, zmm27, zmm26",
            "vpmadd52huq zmm17, zmm27, zmm26",
            "vmovdqu64 zmm27, [{xptr}+320]",
            "vpmadd52luq zmm10, zmm27, zmm26",
            "vpmadd52huq zmm11, zmm27, zmm26",
            "vmovdqu64 zmm27, [{xptr}+512]",
            "vpmadd52luq zmm4, zmm27, zmm26",
            "vpmadd52huq zmm5, zmm27, zmm26",
            "vmovdqu64 zmm27, [{xptr}+704]",
            "vpmadd52luq zmm22, zmm27, zmm26",
            "vpmadd52huq zmm23, zmm27, zmm26",
            "add {yptr}, 128",
            "add {xptr}, 768",
            "dec {remaining}",
            "jnz 40b",
            "42:",
            "cmp {k}, 3",
            "jb 43f",
            "vpbroadcastq zmm24, [{constants}+32]",
            "vpsrlq zmm25, zmm18, 42",
            "vpandq zmm18, zmm18, zmm30",
            "vpmadd52luq zmm18, zmm25, zmm24",
            "vpsrlq zmm25, zmm18, 42",
            "vpandq zmm18, zmm18, zmm30",
            "vpmadd52luq zmm18, zmm25, zmm24",
            "vpbroadcastq zmm24, [{constants}+48]",
            "vpxorq zmm25, zmm25, zmm25",
            "vpmadd52huq zmm25, zmm19, zmm24",
            "vpbroadcastq zmm24, [{constants}+40]",
            "vpxorq zmm26, zmm26, zmm26",
            "vpmadd52luq zmm26, zmm19, zmm24",
            "vpbroadcastq zmm24, [{constants}+56]",
            "vpmadd52luq zmm26, zmm25, zmm24",
            "vpbroadcastq zmm24, [{constants}+24]",
            "vpandq zmm26, zmm26, zmm24",
            "vpaddq zmm18, zmm18, zmm26",
            "vpbroadcastq zmm24, [{constants}+64]",
            "vpsllq zmm25, zmm24, 1",
            "vpsubq zmm26, zmm18, zmm25",
            "vpminuq zmm18, zmm18, zmm26",
            "vpsubq zmm26, zmm18, zmm24",
            "vpminuq zmm18, zmm18, zmm26",
            "vpbroadcastq zmm24, [{constants}+72]",
            "vpsrlq zmm25, zmm20, 42",
            "vpandq zmm20, zmm20, zmm30",
            "vpmadd52luq zmm20, zmm25, zmm24",
            "vpsrlq zmm25, zmm20, 42",
            "vpandq zmm20, zmm20, zmm30",
            "vpmadd52luq zmm20, zmm25, zmm24",
            "vpbroadcastq zmm24, [{constants}+88]",
            "vpxorq zmm25, zmm25, zmm25",
            "vpmadd52huq zmm25, zmm21, zmm24",
            "vpbroadcastq zmm24, [{constants}+80]",
            "vpxorq zmm26, zmm26, zmm26",
            "vpmadd52luq zmm26, zmm21, zmm24",
            "vpbroadcastq zmm24, [{constants}+96]",
            "vpmadd52luq zmm26, zmm25, zmm24",
            "vpbroadcastq zmm24, [{constants}+24]",
            "vpandq zmm26, zmm26, zmm24",
            "vpaddq zmm20, zmm20, zmm26",
            "vpbroadcastq zmm24, [{constants}+104]",
            "vpsllq zmm25, zmm24, 1",
            "vpsubq zmm26, zmm20, zmm25",
            "vpminuq zmm20, zmm20, zmm26",
            "vpsubq zmm26, zmm20, zmm24",
            "vpminuq zmm20, zmm20, zmm26",
            "vpbroadcastq zmm24, [{constants}+112]",
            "vpsrlq zmm25, zmm22, 42",
            "vpandq zmm22, zmm22, zmm30",
            "vpmadd52luq zmm22, zmm25, zmm24",
            "vpsrlq zmm25, zmm22, 42",
            "vpandq zmm22, zmm22, zmm30",
            "vpmadd52luq zmm22, zmm25, zmm24",
            "vpbroadcastq zmm24, [{constants}+128]",
            "vpxorq zmm25, zmm25, zmm25",
            "vpmadd52huq zmm25, zmm23, zmm24",
            "vpbroadcastq zmm24, [{constants}+120]",
            "vpxorq zmm26, zmm26, zmm26",
            "vpmadd52luq zmm26, zmm23, zmm24",
            "vpbroadcastq zmm24, [{constants}+136]",
            "vpmadd52luq zmm26, zmm25, zmm24",
            "vpbroadcastq zmm24, [{constants}+24]",
            "vpandq zmm26, zmm26, zmm24",
            "vpaddq zmm22, zmm22, zmm26",
            "vpbroadcastq zmm24, [{constants}+144]",
            "vpsllq zmm25, zmm24, 1",
            "vpsubq zmm26, zmm22, zmm25",
            "vpminuq zmm22, zmm22, zmm26",
            "vpsubq zmm26, zmm22, zmm24",
            "vpminuq zmm22, zmm22, zmm26",
            "vpbroadcastq zmm24, [{constants}+16]",
            "vpandq zmm24, zmm20, zmm24",
            "vpsllq zmm24, zmm24, 42",
            "vporq zmm24, zmm24, zmm18",
            "vpsrlq zmm25, zmm20, 22",
            "vpsllq zmm26, zmm22, 20",
            "vporq zmm25, zmm25, zmm26",
            $store0,
            $store1,
            "add {output}, {output_stride}",
            "43:",
            "vpxorq zmm18, zmm18, zmm18",
            "vpxorq zmm19, zmm19, zmm19",
            "vpxorq zmm20, zmm20, zmm20",
            "vpxorq zmm21, zmm21, zmm21",
            "vpxorq zmm22, zmm22, zmm22",
            "vpxorq zmm23, zmm23, zmm23",
            "add {key}, {key_stride}",
            "inc {k}",
            "cmp {k}, {limit}",
            "jae 90f",
            "cmp {key}, {key_end}",
            "jae 52f",
            "mov {yptr}, {key}",
            "mov {xptr}, {input}",
            "mov {remaining}, {rows}",
            "50:",
            "vmovdqu64 zmm24, [{yptr}]",
            "vmovdqu64 zmm25, [{yptr}+64]",
            "vpandq zmm26, zmm24, zmm30",
            "vmovdqu64 zmm27, [{xptr}+0]",
            "vpmadd52luq zmm18, zmm27, zmm26",
            "vpmadd52huq zmm19, zmm27, zmm26",
            "vmovdqu64 zmm27, [{xptr}+192]",
            "vpmadd52luq zmm12, zmm27, zmm26",
            "vpmadd52huq zmm13, zmm27, zmm26",
            "vmovdqu64 zmm27, [{xptr}+384]",
            "vpmadd52luq zmm6, zmm27, zmm26",
            "vpmadd52huq zmm7, zmm27, zmm26",
            "vmovdqu64 zmm27, [{xptr}+576]",
            "vpmadd52luq zmm0, zmm27, zmm26",
            "vpmadd52huq zmm1, zmm27, zmm26",
            "vpsrlq zmm26, zmm24, 42",
            "vpandq zmm24, zmm25, zmm31",
            "vpsllq zmm24, zmm24, 22",
            "vporq zmm26, zmm26, zmm24",
            "vmovdqu64 zmm27, [{xptr}+64]",
            "vpmadd52luq zmm20, zmm27, zmm26",
            "vpmadd52huq zmm21, zmm27, zmm26",
            "vmovdqu64 zmm27, [{xptr}+256]",
            "vpmadd52luq zmm14, zmm27, zmm26",
            "vpmadd52huq zmm15, zmm27, zmm26",
            "vmovdqu64 zmm27, [{xptr}+448]",
            "vpmadd52luq zmm8, zmm27, zmm26",
            "vpmadd52huq zmm9, zmm27, zmm26",
            "vmovdqu64 zmm27, [{xptr}+640]",
            "vpmadd52luq zmm2, zmm27, zmm26",
            "vpmadd52huq zmm3, zmm27, zmm26",
            "vpsrlq zmm26, zmm25, 20",
            "vmovdqu64 zmm27, [{xptr}+128]",
            "vpmadd52luq zmm22, zmm27, zmm26",
            "vpmadd52huq zmm23, zmm27, zmm26",
            "vmovdqu64 zmm27, [{xptr}+320]",
            "vpmadd52luq zmm16, zmm27, zmm26",
            "vpmadd52huq zmm17, zmm27, zmm26",
            "vmovdqu64 zmm27, [{xptr}+512]",
            "vpmadd52luq zmm10, zmm27, zmm26",
            "vpmadd52huq zmm11, zmm27, zmm26",
            "vmovdqu64 zmm27, [{xptr}+704]",
            "vpmadd52luq zmm4, zmm27, zmm26",
            "vpmadd52huq zmm5, zmm27, zmm26",
            "add {yptr}, 128",
            "add {xptr}, 768",
            "dec {remaining}",
            "jnz 50b",
            "52:",
            "cmp {k}, 3",
            "jb 53f",
            "vpbroadcastq zmm24, [{constants}+32]",
            "vpsrlq zmm25, zmm0, 42",
            "vpandq zmm0, zmm0, zmm30",
            "vpmadd52luq zmm0, zmm25, zmm24",
            "vpsrlq zmm25, zmm0, 42",
            "vpandq zmm0, zmm0, zmm30",
            "vpmadd52luq zmm0, zmm25, zmm24",
            "vpbroadcastq zmm24, [{constants}+48]",
            "vpxorq zmm25, zmm25, zmm25",
            "vpmadd52huq zmm25, zmm1, zmm24",
            "vpbroadcastq zmm24, [{constants}+40]",
            "vpxorq zmm26, zmm26, zmm26",
            "vpmadd52luq zmm26, zmm1, zmm24",
            "vpbroadcastq zmm24, [{constants}+56]",
            "vpmadd52luq zmm26, zmm25, zmm24",
            "vpbroadcastq zmm24, [{constants}+24]",
            "vpandq zmm26, zmm26, zmm24",
            "vpaddq zmm0, zmm0, zmm26",
            "vpbroadcastq zmm24, [{constants}+64]",
            "vpsllq zmm25, zmm24, 1",
            "vpsubq zmm26, zmm0, zmm25",
            "vpminuq zmm0, zmm0, zmm26",
            "vpsubq zmm26, zmm0, zmm24",
            "vpminuq zmm0, zmm0, zmm26",
            "vpbroadcastq zmm24, [{constants}+72]",
            "vpsrlq zmm25, zmm2, 42",
            "vpandq zmm2, zmm2, zmm30",
            "vpmadd52luq zmm2, zmm25, zmm24",
            "vpsrlq zmm25, zmm2, 42",
            "vpandq zmm2, zmm2, zmm30",
            "vpmadd52luq zmm2, zmm25, zmm24",
            "vpbroadcastq zmm24, [{constants}+88]",
            "vpxorq zmm25, zmm25, zmm25",
            "vpmadd52huq zmm25, zmm3, zmm24",
            "vpbroadcastq zmm24, [{constants}+80]",
            "vpxorq zmm26, zmm26, zmm26",
            "vpmadd52luq zmm26, zmm3, zmm24",
            "vpbroadcastq zmm24, [{constants}+96]",
            "vpmadd52luq zmm26, zmm25, zmm24",
            "vpbroadcastq zmm24, [{constants}+24]",
            "vpandq zmm26, zmm26, zmm24",
            "vpaddq zmm2, zmm2, zmm26",
            "vpbroadcastq zmm24, [{constants}+104]",
            "vpsllq zmm25, zmm24, 1",
            "vpsubq zmm26, zmm2, zmm25",
            "vpminuq zmm2, zmm2, zmm26",
            "vpsubq zmm26, zmm2, zmm24",
            "vpminuq zmm2, zmm2, zmm26",
            "vpbroadcastq zmm24, [{constants}+112]",
            "vpsrlq zmm25, zmm4, 42",
            "vpandq zmm4, zmm4, zmm30",
            "vpmadd52luq zmm4, zmm25, zmm24",
            "vpsrlq zmm25, zmm4, 42",
            "vpandq zmm4, zmm4, zmm30",
            "vpmadd52luq zmm4, zmm25, zmm24",
            "vpbroadcastq zmm24, [{constants}+128]",
            "vpxorq zmm25, zmm25, zmm25",
            "vpmadd52huq zmm25, zmm5, zmm24",
            "vpbroadcastq zmm24, [{constants}+120]",
            "vpxorq zmm26, zmm26, zmm26",
            "vpmadd52luq zmm26, zmm5, zmm24",
            "vpbroadcastq zmm24, [{constants}+136]",
            "vpmadd52luq zmm26, zmm25, zmm24",
            "vpbroadcastq zmm24, [{constants}+24]",
            "vpandq zmm26, zmm26, zmm24",
            "vpaddq zmm4, zmm4, zmm26",
            "vpbroadcastq zmm24, [{constants}+144]",
            "vpsllq zmm25, zmm24, 1",
            "vpsubq zmm26, zmm4, zmm25",
            "vpminuq zmm4, zmm4, zmm26",
            "vpsubq zmm26, zmm4, zmm24",
            "vpminuq zmm4, zmm4, zmm26",
            "vpbroadcastq zmm24, [{constants}+16]",
            "vpandq zmm24, zmm2, zmm24",
            "vpsllq zmm24, zmm24, 42",
            "vporq zmm24, zmm24, zmm0",
            "vpsrlq zmm25, zmm2, 22",
            "vpsllq zmm26, zmm4, 20",
            "vporq zmm25, zmm25, zmm26",
            $store0,
            $store1,
            "add {output}, {output_stride}",
            "53:",
            "vpxorq zmm0, zmm0, zmm0",
            "vpxorq zmm1, zmm1, zmm1",
            "vpxorq zmm2, zmm2, zmm2",
            "vpxorq zmm3, zmm3, zmm3",
            "vpxorq zmm4, zmm4, zmm4",
            "vpxorq zmm5, zmm5, zmm5",
            "add {key}, {key_stride}",
            "inc {k}",
            "cmp {k}, {limit}",
            "jae 90f",
            "jmp 2b",
            "90:",
            key=inout(reg) key=>_, key_end=in(reg) key.wrapping_byte_add(key_size*key_stride),
            key_stride=in(reg) key_stride, input=in(reg) input, rows=in(reg) rows,
            output=inout(reg) output=>_, output_stride=in(reg) output_stride, limit=in(reg) size+3,
            constants=in(reg) REDUCTION.as_ptr(),
            k=out(reg) _, yptr=out(reg) _, xptr=out(reg) _, remaining=out(reg) _,
            out("zmm0") _,
            out("zmm1") _,
            out("zmm2") _,
            out("zmm3") _,
            out("zmm4") _,
            out("zmm5") _,
            out("zmm6") _,
            out("zmm7") _,
            out("zmm8") _,
            out("zmm9") _,
            out("zmm10") _,
            out("zmm11") _,
            out("zmm12") _,
            out("zmm13") _,
            out("zmm14") _,
            out("zmm15") _,
            out("zmm16") _,
            out("zmm17") _,
            out("zmm18") _,
            out("zmm19") _,
            out("zmm20") _,
            out("zmm21") _,
            out("zmm22") _,
            out("zmm23") _,
            out("zmm24") _,
            out("zmm25") _,
            out("zmm26") _,
            out("zmm27") _,
            out("zmm28") _,
            out("zmm29") _,
            out("zmm30") _,
            out("zmm31") _,
            options(nostack)
        ); }; }
        if STREAM {
            run!("vmovntdq [{output}], zmm24", "vmovntdq [{output}+64], zmm25");
        } else {
            run!("vmovdqu64 [{output}], zmm24", "vmovdqu64 [{output}+64], zmm25");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use poulpy_hal::{api::*, execution::SerialTaskExecutor, layouts::*};

    #[test]
    fn contraction_matches_digit_products() {
        let n = 64;
        let module = Module::<crate::NTT3x42Ifma>::new(n as u64);
        #[cfg(feature = "enable-rayon")]
        let pool = rayon::ThreadPoolBuilder::new().num_threads(4).build().unwrap();
        let mut seed = 0x768325dea912_u64;
        for edge in [false, true] {
            for a_size in [16, 17, 18, 19, 20, 23, 24, 27, 28, 31, 32] {
                for key_size in [0, 3, 33] {
                    let mut input = module.vec_znx_dft_alloc(n, 1, a_size);
                    let mut key = module.vmp_pmat_alloc(n, 8, 1, 2, key_size, PrepareHint::Reuse);
                    for data in [input.data.as_mut_slice(), key.data_mut().as_mut()] {
                        for group in cast_slice_mut::<_, u64>(data).chunks_exact_mut(16) {
                            for lane in 0..8 {
                                let mut y = [0; 3];
                                for (value, q) in y.iter_mut().zip(Primes42::Q) {
                                    seed ^= seed << 13;
                                    seed ^= seed >> 7;
                                    seed ^= seed << 17;
                                    *value = if edge { q - 1 } else { seed % q };
                                }
                                group[lane] = y[0] | (y[1] & ((1 << 22) - 1)) << 42;
                                group[8 + lane] = (y[1] >> 22) | y[2] << 20;
                            }
                        }
                    }
                    for prefix in [0, 1, a_size - 1, a_size] {
                        input.data[..16 * n * prefix].fill(0);
                        for output_size in [0, 2, 26, 36] {
                            let mut expected = module.vec_znx_dft_alloc(n, 2, output_size);
                            let mut actual = module.vec_znx_dft_alloc(n, 2, output_size);
                            let bytes = vmp_apply_digits_strided_tmp_bytes_ifma(1, a_size, 4, 8, 1, 4);
                            let mut tmp = vec![0; bytes / size_of::<u64>()];
                            vmp_apply_dft_to_dft_digits_strided_ifma_impl::<SerialTaskExecutor, false>(
                                &mut expected.to_backend_mut(),
                                &input.to_backend_ref(),
                                4,
                                3,
                                &key.to_backend_ref(),
                                None,
                                &mut tmp,
                            );
                            actual.data.fill(0xa5);
                            apply::<SerialTaskExecutor>(
                                &mut actual.to_backend_mut(),
                                &input.to_backend_ref(),
                                &key.to_backend_ref(),
                                None,
                                &mut tmp,
                            );
                            assert_eq!(
                                actual.data, expected.data,
                                "a={a_size}, key={key_size}, output={output_size}, prefix={prefix}, edge={edge}"
                            );
                            #[cfg(feature = "enable-rayon")]
                            pool.install(|| {
                                actual.data.fill(0xa5);
                                apply::<poulpy_cpu_rayon::RayonTaskExecutor>(
                                    &mut actual.to_backend_mut(),
                                    &input.to_backend_ref(),
                                    &key.to_backend_ref(),
                                    Some(prefix),
                                    &mut tmp,
                                );
                                assert_eq!(
                                    actual.data, expected.data,
                                    "parallel a={a_size}, key={key_size}, output={output_size}, prefix={prefix}, edge={edge}"
                                );
                            });
                        }
                    }
                }
            }
        }
    }
}
