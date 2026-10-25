use poulpy_cpu_portable::{FFT64Portable, hal_defaults::take_host_typed};
use poulpy_hal::{api::ScratchOwnedAlloc, layouts::ScratchOwned};

#[test]
fn typed_scratch_reuses_existing_bytes() {
    let mut scratch = ScratchOwned::<FFT64Portable>::alloc(64);
    let (bytes, _) = take_host_typed::<FFT64Portable, u8>(scratch.arena(), 64);
    bytes.fill(0xff);

    let (words, _) = take_host_typed::<FFT64Portable, i128>(scratch.arena(), 4);
    assert_eq!(words, &[-1; 4]);
    words[0] = i128::MIN;

    let (bytes, _) = take_host_typed::<FFT64Portable, u8>(scratch.arena(), 64);
    assert_eq!(&bytes[..16], &i128::MIN.to_ne_bytes());
    bytes.fill(0xff);

    let (words, _) = take_host_typed::<FFT64Portable, u64>(scratch.arena(), 8);
    assert_eq!(words, &[u64::MAX; 8]);

    let (floats, _) = take_host_typed::<FFT64Portable, f64>(scratch.arena(), 8);
    assert!(floats.iter().all(|value| value.to_bits() == u64::MAX));
}

#[test]
fn typed_scratch_preserves_empty_and_zero_sized_slices() {
    let mut scratch = ScratchOwned::<FFT64Portable>::alloc(64);
    let arena = scratch.arena();
    let available = arena.available();
    let (empty, arena) = take_host_typed::<FFT64Portable, i128>(arena, 0);
    assert!(empty.is_empty());
    assert_eq!(arena.available(), available);

    let (units, arena) = take_host_typed::<FFT64Portable, ()>(arena, 3);
    assert_eq!(units, &[(); 3]);
    assert_eq!(arena.available(), available);

    let (bytes, _) = take_host_typed::<FFT64Portable, u8>(arena, available);
    bytes.fill(0xab);
    assert_eq!(units.len(), 3);
}
