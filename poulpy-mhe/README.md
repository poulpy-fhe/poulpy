# poulpy-mhe

Backend-agnostic multiparty homomorphic encryption built on `poulpy-core` and
`poulpy-hal`. The crate depends on no concrete backend.

## Layouts

Parties exchange public aggregatable transcripts (PATs), one type per shape:

- `GLWEPatCompressed`: seeded GLWE body (collective public key, shares to
  encryption).
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
  the circular product of every other column, over one `r x r` mask matrix
  per gadget row.
- `GLWEPrivateKeyswitchShare`, `GLWEPublicKeyswitchShare`: a core `GLWE`, the share of
  a collective key switch to a secret key (rank 0) or to a public key.
- `GLWEEncToShareShare`: a rank-0 core `GLWE`, the public share of an
  encryption-to-shares conversion; `GLWEShareToEncShare`: a
  `GLWEPatCompressed`, the share of a shares-to-encryption conversion.

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
- `GLWEPrivateKeyswitchMHEProtocol`, `GLWEPublicKeyswitchMHEProtocol`: collective key
  switching of a ciphertext to the ideal output secret or to a public key; shares
  are generated from the ciphertext's mask alone, a core `GLWEMask`.
- `GLWEEncToShareMHEProtocol`, `GLWEShareToEncMHEProtocol`: conversions between
  a ciphertext and additive shares of its plaintext on the torus.

## Smudging

Key switching and encryption-to-shares take a caller-selected `SmudgingNoise`
flood, a discrete Gaussian or a uniform distribution on consecutive integers,
sampled on the share's own precision grid. Size it with the
[smudging contract](docs/mhe-contracts.md#smudging); small test parameters do
not establish security.

## Security

The protocols are secure against passive adversaries. Seed rules, private
streams and the threat model common to every protocol are in the
[operation contracts](docs/mhe-contracts.md); each protocol trait documents
its own.
