
# Failure estimates for base-2^K arithmetic

Consider one output polynomial in $\mathbb Z[X]/(X^N+1)$:

$$
C=\sum_{r=1}^{d} A_r B_r.
$$

Let $P=2^{K-1}$ be the maximum digit magnitude. Model independent input coefficients $U,V$ as centered uniform variables on $[-P,P]$. Each individual coefficient product is bounded by $P^2=2^{2K-2}$ in magnitude, and each output coefficient contains $dN$ independent signed products. We neglect integer endpoint corrections and use Gaussian tail approximations; variance alone does not establish their accuracy at extremely small probabilities.

## NTT backend: integer reconstruction

Let $Q$ be the **total CRT modulus**. Centered reconstruction succeeds when $|C_j|<Q/2$.

For one input coefficient, the uniform interval has width $2P$. Thus

$$
\mathbb E[U]=0,
\qquad
\mathbb E[U^2]=\operatorname{Var}(U)
\simeq\frac{(2P)^2}{12}=\frac{P^2}{3}.
$$

**The product $UV$ is not uniform**, even though $|UV|\le P^2$. Its variance must be computed from the input moments, rather than by applying the uniform-variance formula to its magnitude bound. Independence gives $\mathbb E[UV]=0$ and

$$
\underbrace{\operatorname{Var}(UV)}_{\text{variance of one product}}
=\mathbb E[U^2V^2]-(\mathbb E[UV])^2
=\mathbb E[U^2]\mathbb E[V^2]
\simeq\frac{P^4}{9}.
$$

Taking its square root gives the **standard deviation**:

$$
\sigma(UV)=\sqrt{\operatorname{Var}(UV)}
\simeq\frac{P^2}{3}=\frac{2^{2K-2}}{3}.
$$

The expression $P^2/\sqrt{12}$ would describe the standard deviation of a uniform variable on an interval of width $P^2$; it does not describe $UV$.

Finally, summing $dN$ independent products adds their variances:

$$
\boxed{
\sigma_c\simeq\frac{P^2\sqrt{dN}}{3}
=\frac{2^{2K-2}\sqrt{dN}}{3}
=\frac{2^{2K}\sqrt{dN}}{12}.
}
$$

The estimated two-sided failure probability **per coefficient** is

$$
\boxed{
\widehat p_{\mathrm{NTT}}
=\operatorname{erfc}\!\left(\frac{Q/2}{\sqrt{2}\,\sigma_c}\right).
}
$$

The $Q/2$ threshold is essential: $Q$ denotes the modulus, not the maximum centered magnitude.

## FFT64 backend: numerical error

Let $\sigma_F(N,K)$ be the measured standard deviation, per coefficient, of the numerical error in **one polynomial multiplication**, including forward FFTs, pointwise multiplication, and inverse FFT. It differs from $\sigma_c$, which measures the size of the exact integer result.

Assuming independent, approximately centered Gaussian multiplication errors,

$$
\sigma_e^2=d\,\sigma_F(N,K)^2.
$$

Rounding recovers the integer coefficient when the accumulated numerical error has magnitude below $1/2$. Thus

$$
\boxed{
\widehat p_{\mathrm{FFT}}
=\operatorname{erfc}\!\left(\frac{1/2}{\sqrt{2}\,\sigma_e}\right)
=\operatorname{erfc}\!\left(\frac{1}{\sqrt{8d}\,\sigma_F(N,K)}\right).
}
$$

This agrees with §6.1 and Lemma 6.1 of [the original paper](https://eprint.iacr.org/2023/771), with $d=2\ell$. Under its additional assumption of independent Gaussian coordinates, the probability that any coefficient fails is $1-[1-\widehat p_{\mathrm{FFT}}]^N$.

For a kernel that accumulates products in the Fourier domain before one inverse FFT, calibrate $\sigma_e$ for that kernel directly; $\sqrt d\,\sigma_F$ need not describe its errors. An integer result below $2^{53}$ alone does not ensure correct rounding.

## Any coefficient, and logarithms

For either backend, substitute its per-coefficient estimate $\widehat p$ into the union bound:

$$
\boxed{
\widehat P_{\mathrm{any,UB}}=\min(1,N\widehat p),
\qquad
\log_2\widehat P_{\mathrm{any,UB}}
=\min\!\left(0,\log_2N+\log_2\widehat p\right).
}
$$

The union bound requires no independence between output coefficients; its numerical estimate still inherits the Gaussian approximation. The $N$ inside $\sigma_c$ counts products in one coefficient, while the $N$ outside counts output coefficients. For $m$ output polynomials in a vector–matrix product, replace the outside factor $N$ by $mN$; $d$ is the inner dimension. For FFT, dependence on $N$ is already included in $\sigma_F(N,K)$.

## Example: N = 2^16, d = 32, K = 52, Q = 2^120

Here $P=2^{51}$, so for NTT,

$$
\sigma_c\simeq\frac{2^{102}\sqrt{2^{21}}}{3}
\simeq 2^{110.91504}.
$$

Using $Q=2^{120}$, the centered reconstruction threshold is $Q/2=2^{119}$, and

$$
\frac{Q/2}{\sqrt{2}\,\sigma_c}\simeq192.
$$

Consequently, the Gaussian estimates are

$$
\boxed{
\widehat p_{\mathrm{NTT}}\simeq\operatorname{erfc}(192)
\simeq 2^{-53191.921},
\qquad
\widehat P_{\mathrm{any,UB}}
\simeq2^{16}\operatorname{erfc}(192)
\simeq 2^{-53175.921}.
}
$$

These are Gaussian far-tail extrapolations, not certified bounds. The example uses $Q=2^{120}$; for an implementation, substitute the actual product of CRT primes.

For FFT64, the paper's experiments support limb widths around $K=19$, not $K=52$ (§6.1, Table 3). A direct $K=52$ FFT64 multiplication has no justified small failure estimate from these data; it requires limb splitting or higher precision and a corresponding numerical-error analysis.

## Selecting a radix for a failure target

`Module::<BE>::max_base2k_for_failure(N, d, failure_bits)` uses the NTT model above. A positive `failure_bits` value $\lambda$ requests an estimated probability at most $2^{-\lambda}$ that any coefficient of one output polynomial fails. The result is a `const`-evaluable `Option<usize>`: `Some(K)` for CRT backends, `Some(0)` if no positive radix fits, and `None` for backends without a CRT modulus.

To avoid evaluating or inverting `erfc` in a constant expression, the query uses the conservative envelope

$$
\operatorname{erfc}(x)\le e^{-x^2},\qquad x\ge0.
$$

Together with the union bound, it suffices that

$$
\left(\frac{Q/2}{\sqrt{2}\,\sigma_c}\right)^2
\ge (\lambda+\log_2N)\ln2.
$$

Substituting $\sigma_c=2^{2K}\sqrt{dN}/12$ gives

$$
K\le\frac12\left[
\log_2Q+\log_2 6
-\frac12(\log_2d+\log_2N)
-\frac12\log_2\!\left(2\ln2\,(\lambda+\log_2N)\right)
\right].
$$

The query rounds down, caps the result at the supported radix 62, and leaves a small numerical margin at integer thresholds. It computes the largest radix satisfying this envelope, which can be more conservative than evaluating `erfc` directly. The envelope is a bound on the Gaussian estimate; it does not establish Gaussian accuracy for the actual far tail.

For the actual NTT4x30 modulus, $\log_2Q\simeq119.8861552574811$, $d=32$, and $\lambda=128$, the query gives $K=54$ at both $N=2^{15}$ and $N=2^{16}$. At $N=2^{16}$, increasing the target to $\lambda=256$ gives $K=53$. For $m$ output polynomials, add $\lceil\log_2m\rceil$ to $\lambda$ to distribute a total failure budget across them.

This model assumes independent centered input coefficients and neglects discrete endpoint corrections. Computationally pseudorandom ciphertext components motivate that assumption for suitable FHE operations; IND-CPA alone does not imply joint independence of reused or squared operands. A polynomial whose coefficients are all $-2^{52}$, squared at $N=2^{15}$, produces a coefficient $2^{119}>Q/2$ and wraps. That structured example lies outside this uniform-input model.

The existing `Module::<BE>::max_base2k(N)` remains a degree-only starting recommendation and does not specify a failure target. The probability query returns `None` for FFT: selecting a radix from a numerical-error target requires the calibrated $\sigma_e$ of the actual accumulated kernel discussed above.
