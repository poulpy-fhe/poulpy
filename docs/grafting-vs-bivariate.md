# Grafting vs. the Bivariate Representation

Poulpy represents Torus polynomials in the **bivariate** (base-$2^K$) form rather than the residue number system (RNS) used by most CKKS libraries.
A natural question is how this compares to **Grafting** ([IACR ePrint 2024/1014](https://eprint.iacr.org/2024/1014)), a recent technique that targets a very similar pain point, the loss of *bit granularity* in scale and modulus management, but from inside the RNS world.
This document compares the two.

## The shared problem: scale and modulus coupling in RNS-CKKS

In RNS-CKKS the ciphertext modulus $Q = q_0 \cdots q_\ell$ is a product of NTT-friendly primes, and rescaling divides $Q$ by one of those primes.
This ties two logically distinct quantities together.

- The **scale factor** $\Delta$, which governs numerical precision.
- The **ciphertext modulus**, which provides the cryptographic structure.

Because rescaling can only remove a whole prime, the primes must be chosen close to the scale factor, typically 30 to 60 bits.
This has several consequences.

- Precision can only be managed at **prime-chain granularity**, not bit granularity.
- The machine word of 64 bits is under-utilized, since primes are smaller than a word.
- Parameter sets become **circuit-specific**, because the prime chain encodes a level schedule, and key-switching keys are tied to that schedule.
- Suitable small NTT primes are scarce for ring dimensions $2^{14}$ to $2^{17}$, which constrains parameter design.
- Plaintexts must themselves be carried in RNS across the entire moduli chain, so a ciphertext over 30 primes implies at least a 30x expansion of every plaintext.

Both Grafting and the bivariate representation set out to recover bit-granular scale and precision management.
They differ fundamentally in *how*.

## Two strategies

**Grafting keeps RNS and decouples scale from modulus with added machinery.**
It introduces two tools.

- **Rational rescale.** Rescaling by a rational factor $Q/Q'$ where $Q'$ need not divide $Q$, realized as an integer multiplication followed by a `ModDown`. The scale can then drop by an arbitrary bit-length rather than by a whole prime.
- **Universal sprouts.** A small reusable power-of-two factor $r$ grafted onto the modulus alongside word-size NTT primes. A *universal* sprout $r_{\text{top}}$ is chosen so that its divisors can represent almost any bit-length, which enables rescaling to almost any modulus without generating new key-switching keys. A technique called modulus resurrection reuses top-modulus factors to preserve key-switchability.

The payoff is that the modulus is packed mostly into machine word-size primes, so there are fewer RNS factors and therefore fewer NTT and iNTT operations.
Scale factors become freely adjustable even after the keys are fixed, and a single application-independent parameter set can serve many circuits.
Reported gains are roughly 1.3x to 2.1x speed-ups for key-switching and multiplication, up to 62% smaller keys, and up to 41% less modulus consumption.
These gains are relative to prior RNS-CKKS, and the technique is deployable inside the existing RNS and NTT ecosystem.

**The bivariate representation removes RNS, so the coupling never arises.**
Instead of representing a large coefficient in RNS, each coefficient is decomposed in base $2^{K}$.
This is the bivariate ring $\mathbb{Z}[X, Y]$ with $Y = 2^{K}$, in which cyclotomic arithmetic in the $X$ dimension is decoupled from large-integer arithmetic in the $Y$ or limb dimension.
There is no scale-to-prime link to break in the first place.
Rescaling and scale management are bit shifts, the digit decomposition needed for key-switching is implicit in the representation, and there are no NTT primes to pick at all.
Plaintexts stay native integer polynomials of a single limb, with no RNS expansion.

## Side-by-side

| Aspect | Grafting (RNS-CKKS) | Bivariate (Poulpy) |
|---|---|---|
| Base representation | RNS over NTT primes | Base-$2^K$ digits in $\mathbb{Z}[X,Y]$, $Y=2^{K}$ |
| Scale and modulus coupling | Decoupled via added tooling | Absent by construction |
| Rescale granularity | Arbitrary, but approximate | Exact bit shift |
| Key-switching DFTs | Grows with RNS factor count | Linear in limbs, decomposition implicit |
| NTT primes | Word-size primes still needed | None required |
| Machine-word use | Word-size primes, awkward sprouts | Sub-word limbs, flat vectorizable layout |
| Plaintext layouts | Full moduli chain | One limb |
| Parameterization | Universal via added sprout | Circuit-independent, native |
| Scheme unification | None, RNS-specific | Common plaintext space for all FHE schemes |
| Implementation | Closed-source | Open-source |

## Takeaway

Grafting and the bivariate representation converge on the same goal, bit-granular CKKS in which the scale is decoupled from the modulus and parameter sets are circuit-independent.
They reach it from opposite directions.
Grafting is an evolutionary additive fix.
It preserves the RNS and NTT ecosystem and its word-size-prime throughput, and it recovers bit granularity by adding rational rescale and universal sprouts.
The bivariate representation is a by design structural choice.
By decomposing coefficients in base $2^K$ it never incurs the scale and modulus coupling, it gets bit-granular rescaling and implicit digit decomposition for free, and it avoids NTT primes entirely.
The cost is leaving the RNS ecosystem and adopting a new arithmetic stack.

The most consequential structural difference is in key-switching, the most frequent and expensive FHE operation.
Grafting reduces its cost by lowering the number of RNS factors.
The bivariate representation changes its asymptotics, making the number of DFTs linear rather than quadratic in the number of limbs, because the digit decomposition is implicit in the representation.
The two approaches are not opposed in spirit, since both recognize that coupling precision to the prime chain is the core inefficiency.
Poulpy chooses to dissolve that coupling at the representation level rather than manage it within RNS.
