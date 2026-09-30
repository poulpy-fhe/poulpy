# poulpy-mhe

Backend-agnostic multiparty homomorphic encryption built on `poulpy-core` and
`poulpy-hal`. The crate depends on no concrete backend.

## Layouts

Parties exchange public aggregatable transcripts (PATs), one type per shape:

- `GLWEPatCompressed`: seeded GLWE body (collective public key).
- `GGLWEPatCompressed`: seeded GGLWE bodies (switching and automorphism keys).
- `GGLWEPat`: unseeded full GGLWE.
- Unseeded GLWE transcripts are core `GLWE`s.

Each protocol has its own share type, a wrapper of these PATs:

- `GLWEPublicKeyShare`: a core `GLWEPublicKeyCompressed`, one seeded body per
  public key entry, with the distribution of the secret it was generated with.
- `GLWESwitchingKeyShare`, `GLWEAutomorphismKeyShare`: a `GGLWEPatCompressed`
  with core's key metadata (degrees, Galois element).

Finalization produces canonical output without changing the PAT or share.
Allocate both through `MHEModuleAlloc` on a `Module`.

## Operations

Every operation follows `API -> delegate -> OEP`; `reference` is the default
implementation, built from `poulpy-core` and `poulpy-hal` operations. See the
[operation contracts](docs/mhe-contracts.md).

- `GLWEPatCompressedOps`, `GGLWEPatCompressedOps`, `GGLWEPatOps`: one trait
  per PAT type to sum PATs and expand one into the canonical ciphertext it
  transcribes.
- `GLWEPublicKeyMHEProtocol`, `GLWESwitchingKeyMHEProtocol`,
  `GLWEAutomorphismKeyMHEProtocol`: one trait per protocol, `mhe_*_share_gen` a
  party's share, `mhe_*_share_aggregate` two shares, `mhe_*_share_finalize` the
  core `GLWEPublicKey`, `GLWESwitchingKey` or `GLWEAutomorphismKey` of the ideal
  secrets.

## Randomness

A public mask seed is common to all parties contributing to one protocol
result. Use a fresh seed for every new result, separated by protocol, session
and key identity, including the Galois element. A key set therefore needs
separate seeds derived from the common reference string. Reusing masks for
switching keys with different input secrets and one output secret exposes the
gadget-scaled difference of those input secrets plus small error.

Keep `source_xe` (errors)
private. Seed its streams
independently for each party and purpose, and consume fresh samples without
replaying a stream. Never initialize a private stream from a public mask seed.
