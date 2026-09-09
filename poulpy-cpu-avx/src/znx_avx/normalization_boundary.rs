use std::arch::x86_64::*;

/// Masks and shifts shared by the narrow and split-word rounding kernels.
pub(crate) struct NormalizationBoundaryAvx {
    mask: __m256i,
    source_mask: __m256i,
    sign: __m256i,
    digit_mask: __m256i,
    digit_sign: __m256i,
    half: __m256i,
    source_bias: __m256i,
    carry_bias: __m256i,
    source_shift: __m128i,
    carry_shift: __m128i,
    lsh: __m128i,
    padding: __m128i,
    width: __m128i,
    round_shift: __m128i,
}

impl NormalizationBoundaryAvx {
    #[inline(always)]
    pub(crate) unsafe fn new(base2k: usize, lsh: usize, padding: usize) -> Self {
        assert!((1..=63).contains(&base2k) && lsh < base2k && padding < base2k);
        unsafe {
            let source_bits = base2k - lsh;
            let width = base2k - padding;
            Self {
                mask: _mm256_set1_epi64x(((1u64 << base2k) - 1) as i64),
                source_mask: _mm256_set1_epi64x(((1u64 << source_bits) - 1) as i64),
                sign: _mm256_set1_epi64x(i64::MIN),
                digit_mask: _mm256_set1_epi64x(((1u64 << width) - 1) as i64),
                digit_sign: _mm256_set1_epi64x((1u64 << (width - 1)) as i64),
                half: _mm256_set1_epi64x(if padding == 0 { 0 } else { 1i64 << (padding - 1) }),
                source_bias: _mm256_set1_epi64x((1u64 << (63 - source_bits)) as i64),
                carry_bias: _mm256_set1_epi64x((1u64 << (63 - base2k)) as i64),
                source_shift: _mm_cvtsi64_si128(source_bits as i64),
                carry_shift: _mm_cvtsi64_si128(base2k as i64),
                lsh: _mm_cvtsi64_si128(lsh as i64),
                padding: _mm_cvtsi64_si128(padding as i64),
                width: _mm_cvtsi64_si128(width as i64),
                round_shift: _mm_cvtsi64_si128((base2k - 1) as i64),
            }
        }
    }

    #[inline(always)]
    pub(crate) unsafe fn floor_low<const CARRY_IN: bool>(&self, a: __m256i, carry: __m256i) -> (__m256i, __m256i) {
        unsafe {
            let low_a = _mm256_sll_epi64(_mm256_and_si256(a, self.source_mask), self.lsh);
            let sum = if CARRY_IN {
                _mm256_add_epi64(low_a, _mm256_and_si256(carry, self.mask))
            } else {
                low_a
            };
            (_mm256_and_si256(sum, self.mask), _mm256_srl_epi64(sum, self.carry_shift))
        }
    }

    #[inline(always)]
    unsafe fn floor<const CARRY_IN: bool>(&self, a: __m256i, carry: __m256i) -> (__m256i, __m256i) {
        unsafe {
            let (low, overflow) = self.floor_low::<CARRY_IN>(a, carry);
            // Biasing the sign bit turns a logical shift into signed floor division.
            let source = _mm256_sub_epi64(
                _mm256_srl_epi64(_mm256_xor_si256(a, self.sign), self.source_shift),
                self.source_bias,
            );
            let high = if CARRY_IN {
                let incoming = _mm256_sub_epi64(
                    _mm256_srl_epi64(_mm256_xor_si256(carry, self.sign), self.carry_shift),
                    self.carry_bias,
                );
                _mm256_add_epi64(source, incoming)
            } else {
                source
            };
            (low, _mm256_add_epi64(high, overflow))
        }
    }

    #[inline(always)]
    pub(crate) unsafe fn round_bit(&self, low: __m256i) -> __m256i {
        unsafe { _mm256_srl_epi64(low, self.round_shift) }
    }

    #[inline(always)]
    pub(crate) unsafe fn round<const PAD: bool>(&self, low: __m256i) -> (__m256i, __m256i) {
        unsafe {
            let rounded = _mm256_srl_epi64(_mm256_add_epi64(low, self.half), self.padding);
            let digit = _mm256_sub_epi64(
                _mm256_xor_si256(_mm256_and_si256(rounded, self.digit_mask), self.digit_sign),
                self.digit_sign,
            );
            let carry = _mm256_srl_epi64(_mm256_sub_epi64(rounded, digit), self.width);
            (if PAD { _mm256_sll_epi64(digit, self.padding) } else { digit }, carry)
        }
    }
}

/// # Safety
/// Requires AVX2. Slice lengths and radix parameters are checked before access.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn znx_normalize_floor_avx<const CARRY_IN: bool, const ROUND: bool>(
    base2k: usize,
    lsh: usize,
    a: &[i64],
    carry: &mut [i64],
) {
    assert!(a.len() >= carry.len());
    unsafe {
        let boundary = NormalizationBoundaryAvx::new(base2k, lsh, 0);
        let end = carry.len() / 4 * 4;
        for i in (0..end).step_by(4) {
            let source = _mm256_loadu_si256(a.as_ptr().add(i).cast());
            let incoming = if CARRY_IN {
                _mm256_loadu_si256(carry.as_ptr().add(i).cast())
            } else {
                _mm256_setzero_si256()
            };
            let (low, mut high) = boundary.floor::<CARRY_IN>(source, incoming);
            if ROUND {
                high = _mm256_add_epi64(high, boundary.round_bit(low));
            }
            _mm256_storeu_si256(carry.as_mut_ptr().add(i).cast(), high);
        }
        poulpy_cpu_ref::reference::normalization::znx_normalize_floor_ref::<CARRY_IN, ROUND>(
            base2k,
            lsh,
            &a[end..],
            &mut carry[end..],
        );
    }
}

/// # Safety
/// Requires AVX2. Slice lengths and radix parameters are checked before access.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn znx_normalize_round_avx<const CARRY_IN: bool, const PAD: bool>(
    base2k: usize,
    lsh: usize,
    padding: usize,
    res: &mut [i64],
    a: &[i64],
    carry: &mut [i64],
) {
    assert!(a.len() >= res.len() && carry.len() >= res.len());
    unsafe {
        let boundary = NormalizationBoundaryAvx::new(base2k, lsh, padding);
        let end = res.len() / 4 * 4;
        for i in (0..end).step_by(4) {
            let source = _mm256_loadu_si256(a.as_ptr().add(i).cast());
            let incoming = if CARRY_IN {
                _mm256_loadu_si256(carry.as_ptr().add(i).cast())
            } else {
                _mm256_setzero_si256()
            };
            let (low, high) = boundary.floor::<CARRY_IN>(source, incoming);
            let (digit, extra) = boundary.round::<PAD>(low);
            _mm256_storeu_si256(res.as_mut_ptr().add(i).cast(), digit);
            _mm256_storeu_si256(carry.as_mut_ptr().add(i).cast(), _mm256_add_epi64(high, extra));
        }
        poulpy_cpu_ref::reference::normalization::znx_normalize_round_ref::<CARRY_IN, PAD>(
            base2k,
            lsh,
            padding,
            &mut res[end..],
            &a[end..],
            &mut carry[end..],
        );
    }
}

/// # Safety
/// Requires AVX2. Slice lengths and radix parameters are checked before access.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn znx_normalize_round_assign_avx<const CARRY_IN: bool>(
    base2k: usize,
    lsh: usize,
    padding: usize,
    res: &mut [i64],
    carry: &mut [i64],
) {
    assert!(carry.len() >= res.len());
    unsafe {
        let boundary = NormalizationBoundaryAvx::new(base2k, lsh, padding);
        let end = res.len() / 4 * 4;
        for i in (0..end).step_by(4) {
            let source = _mm256_loadu_si256(res.as_ptr().add(i).cast());
            let incoming = if CARRY_IN {
                _mm256_loadu_si256(carry.as_ptr().add(i).cast())
            } else {
                _mm256_setzero_si256()
            };
            let (low, high) = boundary.floor::<CARRY_IN>(source, incoming);
            let (digit, extra) = boundary.round::<true>(low);
            _mm256_storeu_si256(res.as_mut_ptr().add(i).cast(), digit);
            _mm256_storeu_si256(carry.as_mut_ptr().add(i).cast(), _mm256_add_epi64(high, extra));
        }
        poulpy_cpu_ref::reference::normalization::znx_normalize_round_assign_ref::<CARRY_IN>(
            base2k,
            lsh,
            padding,
            &mut res[end..],
            &mut carry[end..],
        );
    }
}
