//! Shared Astro constants and mantissa access for portable scalar math and roots.

use astro_float_num::{Consts, WORD_BIT_SIZE, Word};
use std::cell::RefCell;

std::thread_local! {
    static CONSTANTS: RefCell<Consts> = RefCell::new(Consts::new().expect("failed to initialize portable math constants"));
}

pub(crate) fn with_constants<T>(f: impl FnOnce(&mut Consts) -> T) -> T {
    CONSTANTS.with(|constants| f(&mut constants.borrow_mut()))
}

#[inline]
pub(crate) fn mantissa_bit(words: &[Word], bit: usize) -> bool {
    let word = bit / WORD_BIT_SIZE;
    word < words.len() && (words[word] & ((1 as Word) << (bit % WORD_BIT_SIZE))) != 0
}

/// Shift a little-endian mantissa right and round it to nearest-even.
pub(crate) fn rounded_shift(words: &[Word], shift: usize) -> u128 {
    let mut value = 0u128;
    for bit in 0..128 {
        if mantissa_bit(words, shift + bit) {
            value |= 1u128 << bit;
        }
    }
    if shift != 0 {
        let halfway = mantissa_bit(words, shift - 1);
        let sticky = (0..shift - 1).any(|bit| mantissa_bit(words, bit));
        if halfway && (sticky || value & 1 != 0) {
            value += 1;
        }
    }
    value
}
