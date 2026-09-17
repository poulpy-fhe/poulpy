//! Host transfers against the padding of owned buffers: an allocation of `len`
//! bytes may be longer than `len`, and every transfer must copy exactly the
//! caller's bytes and zero the rest.

use crate::{
    layouts::Module,
    test_suite::{TestBackend, TestParams},
};

/// `from_host_bytes`, `copy_from_host`, `copy_to_host` and the view copies
/// accept a byte length that is not a multiple of the allocation padding, round
/// trip the bytes, and leave the padding zero.
pub fn test_transfer_padded_lengths<BE: TestBackend>(_params: &TestParams, _module: &Module<BE>) {
    for len in [1usize, 8, 63, 100, 129] {
        let src: Vec<u8> = (0..len).map(|i| (i * 7 + 3) as u8).collect();

        // Owned buffer from host bytes.
        let owned = BE::from_host_bytes(&src);
        let padded = BE::len_bytes(&owned);
        assert!(padded >= len, "from_host_bytes: len {len}");
        let mut all = vec![0xffu8; padded];
        BE::copy_to_host(&owned, &mut all);
        assert_eq!(&all[..len], &src[..], "from_host_bytes round trip, len {len}");
        assert!(
            all[len..].iter().all(|&b| b == 0),
            "from_host_bytes padding not zero, len {len}"
        );
        let mut back = vec![0u8; len];
        BE::copy_to_host(&owned, &mut back);
        assert_eq!(back, src, "copy_to_host at the unpadded length, len {len}");

        // Owned buffer filled through copy_from_host.
        let mut owned2 = BE::alloc_bytes(len);
        BE::copy_from_host(&mut owned2, &src);
        let mut all2 = vec![0xffu8; BE::len_bytes(&owned2)];
        BE::copy_to_host(&owned2, &mut all2);
        assert_eq!(&all2[..len], &src[..], "copy_from_host, len {len}");
        assert!(
            all2[len..].iter().all(|&b| b == 0),
            "copy_from_host padding not zero, len {len}"
        );

        // View copies over the whole padded buffer.
        let mut owned3 = BE::alloc_bytes(len);
        BE::copy_host_to_view(&mut BE::view_mut(&mut owned3), &src);
        let mut back3 = vec![0u8; len];
        BE::copy_view_to_host(&BE::view(&owned3), &mut back3);
        assert_eq!(back3, src, "view copies, len {len}");
        let mut all3 = vec![0xffu8; BE::len_bytes(&owned3)];
        BE::copy_view_to_host(&BE::view(&owned3), &mut all3);
        assert!(
            all3[len..].iter().all(|&b| b == 0),
            "copy_host_to_view padding not zero, len {len}"
        );
    }
}
