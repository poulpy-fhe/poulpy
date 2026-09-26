# poulpy-mhe

Backend-agnostic multiparty homomorphic encryption built on `poulpy-core` and
`poulpy-hal`. The crate depends on no concrete backend.

## Layouts

Parties exchange public aggregatable transcripts (PATs), one type per shape:

- `GLWEPatCompressed`: seeded GLWE body (collective public key, share-to-encryption).
- `GGLWEPatCompressed`: seeded GGLWE bodies (switching and automorphism keys).
- `GGLWEPat`: unseeded full GGLWE (public-key based relinearization key).
- `GLWESwitchingKeyPatCompressed`, `GLWEAutomorphismKeyPatCompressed`: a
  `GGLWEPatCompressed` with core's key metadata (degrees, Galois element).
- Unseeded GLWE transcripts (collective key switching) are core `GLWE`s.

A PAT carries a canonical flag; `write_to` requires it set. Allocate PATs
through `MHEModuleAlloc` on a `Module`.

## Operations

Every operation follows `API -> delegate -> OEP`; `reference` is the default
implementation, built from `poulpy-core` and `poulpy-hal` operations. See the
[operation contracts](docs/mhe-contracts.md).

- `PatAggregate`, `PatNormalize`, `PatFinalize`: sum shares, restore canonical
  digits, expand a PAT into the ciphertext it transcribes.
- `GLWEPublicKeyShare`: the collective public key, from per-party shares to a
  `GLWEPublicKey` of the ideal secret.
- `GLWESwitchingKeyShare`, `GLWEAutomorphismKeyShare`: collective switching and
  automorphism keys, finalized into core `GLWESwitchingKey` and
  `GLWEAutomorphismKey` of the ideal secrets.
- `GLWEKeyswitchShare`, `GLWEPublicKeyswitchShare`: collective key switching
  of a ciphertext to the ideal output secret or to a public key.
