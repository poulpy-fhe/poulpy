# poulpy-mhe

Backend-agnostic multiparty homomorphic encryption built on `poulpy-core` and
`poulpy-hal`. The crate depends on no concrete backend.

## Layouts

Parties exchange public aggregatable transcripts (PATs), one type per shape:

- `GLWEPatCompressed`: seeded GLWE body (collective public key).
- `GGLWEPatCompressed`: seeded GGLWE bodies (switching and automorphism keys).
- `GGLWEPat`: unseeded full GGLWE (public-key based tensor key).
- Unseeded GLWE transcripts (collective key switching) are core `GLWE`s.

Each protocol has its own share type, a wrapper of these PATs:

- `GLWEPublicKeyShare`: a core `GLWEPublicKeyCompressed`, one seeded body per
  public key entry, with the distribution of the secret it was generated with.
- `GLWESwitchingKeyShare`, `GLWEAutomorphismKeyShare`: a `GGLWEPatCompressed`
  with core's key metadata (degrees, Galois element).
- `GLWETensorKeyShare`: a `GGLWEPat` laid out as core's `GLWETensorKey`.
- `GGSWShare`: seeded `GGLWEPatCompressed`s for column 0 and the two halves of
  the circular product of every other column.
- `GLWEKeyswitchShare`, `GLWEPublicKeyswitchShare`: a core `GLWE`, the share of
  a collective key switch to a secret key (rank 0) or to a public key.

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
  `GLWEAutomorphismKeyMHEProtocol`, `GLWETensorKeyMHEProtocol`,
  `GGSWMHEProtocol`: one trait per protocol, `mhe_*_share_gen` a party's
  share, `mhe_*_share_aggregate` two shares, `mhe_*_share_finalize` the core
  `GLWEPublicKey`, `GLWESwitchingKey`, `GLWEAutomorphismKey`, `GLWETensorKey`
  or `GGSW` of the ideal secrets. The GGSW finalizes with an ephemeral key, in
  one round.
- `GLWEKeyswitchMHEProtocol`, `GLWEPublicKeyswitchMHEProtocol`: collective key
  switching of a ciphertext to the ideal output secret or to a public key.

## Smudging

Key switching takes
caller-selected `SmudgingNoise` parameters: exact discrete Gaussian
(`gaussian(k, log_sigma, cutoff)`) or contiguous uniform (`uniform(k, bits)`).
Both preserve the full integer grid across multiple limbs. Follow the [smudging parameter contract](docs/mhe-contracts.md#smudging-parameters)
for the statistical margin, sampling lattice, Gaussian tail budget and
correctness headroom. The exact CPU Gaussian sampler uses integer rejection
sampling and has variable runtime. Ordinary encryption still uses `NoiseInfos`;
existing flood call sites must migrate to `SmudgingNoise`. Small test parameters do not establish production
security.

## Randomness

A public mask seed is common to all parties contributing to one protocol
result. Use a fresh seed for every new result, separated by protocol, session
and key identity, including the Galois element. A key set therefore needs
separate seeds derived from the common reference string. Reusing masks for
switching keys with different input secrets and one output secret exposes the
gadget-scaled difference of those input secrets plus small error.

Keep `source_xe` (errors) and `source_xu` (public-key ephemeral secrets)
private. Seed their streams
independently for each party and purpose, and consume fresh samples without
replaying a stream. Never initialize a private stream from a public mask seed.

The GGSW protocol derives its sub-seeds internally and intentionally uses the
same mask for the two halves of each circular column. An already finalized
ephemeral key can serve multiple GGSWs as described in the
[collective GGSW contract](docs/mhe-contracts.md#collective-ggsw); each GGSW still
needs its own seed.
