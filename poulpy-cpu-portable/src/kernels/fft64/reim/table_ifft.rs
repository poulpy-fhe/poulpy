// ----------------------------------------------------------------------
// DISCLAIMER
//
// This module contains code that has been directly ported from the
// spqlios-arithmetic library
// (https://github.com/tfhe/spqlios-arithmetic), which is licensed
// under the Apache License, Version 2.0.
//
// The porting process from C to Rust was done with minimal changes
// in order to preserve the semantics and performance characteristics
// of the original implementation.
//
// Both Poulpy and spqlios-arithmetic are distributed under the terms
// of the Apache License, Version 2.0. See the LICENSE file for details.
//
// ----------------------------------------------------------------------

use std::fmt::Debug;

use rand_distr::num_traits::{Float, FloatConst};

use crate::kernels::fft64::reim::{ReimFFTExecute, frac_rev_bits, ifft_portable::ifft_portable, platform_root};
use bytemuck::Zeroable;
use poulpy_hal::{AlignedVec, alloc_aligned};

pub struct ReimIFFTPortable;

impl ReimFFTExecute<ReimIFFTTable<f64>, f64> for ReimIFFTPortable {
    fn reim_dft_execute(table: &ReimIFFTTable<f64>, data: &mut [f64]) {
        ifft_portable(table.m, &table.omg, data);
    }
}

pub struct ReimIFFTTable<R: Float + FloatConst + Debug + Zeroable> {
    m: usize,
    omg: AlignedVec<R>,
}

impl<R: Float + FloatConst + Debug + Zeroable> ReimIFFTTable<R> {
    pub fn new(m: usize) -> Self {
        Self::new_with_roots(m, platform_root)
    }

    /// Builds the table from `root(t) = (cos 2 pi t, sin 2 pi t)`, called on
    /// turns `t` that are exact multiples of `1 / (4m)`. The inverse stores
    /// the conjugate roots.
    pub fn new_with_roots(m: usize, root: impl Fn(R) -> (R, R) + Copy) -> Self {
        Self::new_with_phase(m, R::from(1. / 4.).unwrap(), root)
    }

    pub(crate) fn new_cyclic(m: usize) -> Self {
        Self::new_with_phase(m, R::zero(), platform_root)
    }

    fn new_with_phase(m: usize, phase: R, root: impl Fn(R) -> (R, R) + Copy) -> Self {
        assert!(m & (m - 1) == 0, "m must be a power of two but is {m}");
        let mut omg: AlignedVec<R> = alloc_aligned::<R>(2 * m);

        if m <= 16 {
            match m {
                1 => {}
                2 => {
                    fill_ifft2_omegas::<R>(phase, &mut omg, 0, root);
                }
                4 => {
                    fill_ifft4_omegas(phase, &mut omg, 0, root);
                }
                8 => {
                    fill_ifft8_omegas(phase, &mut omg, 0, root);
                }
                16 => {
                    fill_ifft16_omegas(phase, &mut omg, 0, root);
                }
                _ => {}
            }
        } else if m <= 2048 {
            fill_ifft_bfs_16_omegas(m, phase, &mut omg, 0, root);
        } else {
            fill_ifft_rec_16_omegas(m, phase, &mut omg, 0, root);
        }

        Self { m, omg }
    }

    pub fn execute(&self, data: &mut [R]) {
        ifft_portable(self.m, &self.omg, data);
    }

    pub fn m(&self) -> usize {
        self.m
    }

    pub fn omg(&self) -> &[R] {
        &self.omg
    }
}

#[inline(always)]
fn fill_ifft2_omegas<R: Float + FloatConst>(j: R, omg: &mut [R], pos: usize, root: impl Fn(R) -> (R, R) + Copy) -> usize {
    let omg_pos: &mut [R] = &mut omg[pos..];
    assert!(omg_pos.len() >= 2);
    let angle: R = j / R::from(2).unwrap();
    let (c, s) = root(angle);
    (omg_pos[0], omg_pos[1]) = (c, -s);
    pos + 2
}

#[inline(always)]
fn fill_ifft4_omegas<R: Float + FloatConst>(j: R, omg: &mut [R], pos: usize, root: impl Fn(R) -> (R, R) + Copy) -> usize {
    let omg_pos: &mut [R] = &mut omg[pos..];
    assert!(omg_pos.len() >= 4);
    let angle_1: R = j / R::from(2).unwrap();
    let angle_2: R = j / R::from(4).unwrap();
    let (c, s) = root(angle_2);
    (omg_pos[0], omg_pos[1]) = (c, -s);
    let (c, s) = root(angle_1);
    (omg_pos[2], omg_pos[3]) = (c, -s);
    pos + 4
}

#[inline(always)]
fn fill_ifft8_omegas<R: Float + FloatConst>(j: R, omg: &mut [R], pos: usize, root: impl Fn(R) -> (R, R) + Copy) -> usize {
    let omg_pos: &mut [R] = &mut omg[pos..];
    assert!(omg_pos.len() >= 8);
    let _8th: R = R::from(1. / 8.).unwrap();
    let angle_1: R = j / R::from(2).unwrap();
    let angle_2: R = j / R::from(4).unwrap();
    let angle_4: R = j / R::from(8).unwrap();
    let (c, s) = root(angle_4);
    (omg_pos[0], omg_pos[2]) = (c, -s);
    let (c, s) = root(angle_4 + _8th);
    (omg_pos[1], omg_pos[3]) = (c, -s);
    let (c, s) = root(angle_2);
    (omg_pos[4], omg_pos[5]) = (c, -s);
    let (c, s) = root(angle_1);
    (omg_pos[6], omg_pos[7]) = (c, -s);
    pos + 8
}

#[inline(always)]
fn fill_ifft16_omegas<R: Float + FloatConst>(j: R, omg: &mut [R], pos: usize, root: impl Fn(R) -> (R, R) + Copy) -> usize {
    let omg_pos: &mut [R] = &mut omg[pos..];
    assert!(omg_pos.len() >= 16);
    let _8th: R = R::from(1. / 8.).unwrap();
    let _16th: R = R::from(1. / 16.).unwrap();
    let angle_1: R = j / R::from(2).unwrap();
    let angle_2: R = j / R::from(4).unwrap();
    let angle_4: R = j / R::from(8).unwrap();
    let angle_8: R = j / R::from(16).unwrap();
    let (c, s) = root(angle_8);
    (omg_pos[0], omg_pos[4]) = (c, -s);
    let (c, s) = root(angle_8 + _8th);
    (omg_pos[1], omg_pos[5]) = (c, -s);
    let (c, s) = root(angle_8 + _16th);
    (omg_pos[2], omg_pos[6]) = (c, -s);
    let (c, s) = root(angle_8 + _8th + _16th);
    (omg_pos[3], omg_pos[7]) = (c, -s);
    let (c, s) = root(angle_4);
    (omg_pos[8], omg_pos[9]) = (c, -s);
    let (c, s) = root(angle_4 + _8th);
    (omg_pos[10], omg_pos[11]) = (c, -s);
    let (c, s) = root(angle_2);
    (omg_pos[12], omg_pos[13]) = (c, -s);
    let (c, s) = root(angle_1);
    (omg_pos[14], omg_pos[15]) = (c, -s);
    pos + 16
}

#[inline(always)]
fn fill_ifft_bfs_16_omegas<R: Float + FloatConst + Debug>(
    m: usize,
    j: R,
    omg: &mut [R],
    mut pos: usize,
    root: impl Fn(R) -> (R, R) + Copy,
) -> usize {
    let log_m: usize = (usize::BITS - (m - 1).leading_zeros()) as usize;
    let mut jj: R = j * R::from(16).unwrap() / R::from(m).unwrap();

    for i in (0..m).step_by(16) {
        let j = jj + frac_rev_bits(i >> 4);
        fill_ifft16_omegas(j, omg, pos, root);
        pos += 16
    }

    let mut h: usize = 16;
    let m_half: usize = m >> 1;

    while h < m_half {
        let mm: usize = h << 2;
        for i in (0..m).step_by(mm) {
            let rs_0 = jj + frac_rev_bits::<R>(i / mm) / R::from(4).unwrap();
            let rs_1 = R::from(2).unwrap() * rs_0;
            let (c, s) = root(rs_0);
            (omg[pos], omg[pos + 1]) = (c, -s);
            let (c, s) = root(rs_1);
            (omg[pos + 2], omg[pos + 3]) = (c, -s);
            pos += 4;
        }
        h = mm;
        jj = jj * R::from(4).unwrap();
    }

    if !log_m.is_multiple_of(2) {
        let (c, s) = root(jj);
        (omg[pos], omg[pos + 1]) = (c, -s);
        pos += 2;
        jj = jj * R::from(2).unwrap();
    }

    assert_eq!(jj, j);

    pos
}

#[inline(always)]
fn fill_ifft_rec_16_omegas<R: Float + FloatConst + Debug>(
    m: usize,
    j: R,
    omg: &mut [R],
    mut pos: usize,
    root: impl Fn(R) -> (R, R) + Copy,
) -> usize {
    if m <= 2048 {
        return fill_ifft_bfs_16_omegas(m, j, omg, pos, root);
    }
    let h: usize = m >> 1;
    let s: R = j / R::from(2).unwrap();
    pos = fill_ifft_rec_16_omegas(h, s, omg, pos, root);
    pos = fill_ifft_rec_16_omegas(h, s + R::from(0.5).unwrap(), omg, pos, root);
    let (c, s) = root(s);
    (omg[pos], omg[pos + 1]) = (c, -s);
    pos += 2;
    pos
}
