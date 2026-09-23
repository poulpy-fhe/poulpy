# Failure estimates for base-2^K arithmetic

`Module::<BE>::max_base2k(N, d, failure_bits)` delegates to `MaxBase2k` to select a radix for one output polynomial $C=\sum_{r=1}^{d}A_rB_r$ in $\mathbb Z[X]/(X^N+1)$. Here $d$ counts accumulated polynomial products, and `failure_bits` $=\lambda$ requests an estimated probability at most $2^{-\lambda}$ that any output coefficient fails.

Each term may be an independent product $A_rB_r$ or a square $A_r^2$; the helper conservatively covers both without a workload parameter. Distinct terms must use independent inputs, with independent centered uniform coefficients of magnitude at most $2^{K-1}$. The models neglect discrete endpoint corrections and approximate output tails as Gaussian. These estimates are not guarantees for arbitrary inputs or other correlations, including operand reuse across terms.

## NTT: centered reconstruction

Let $Q$ be the actual product of CRT primes. Squaring repeats off-diagonal coefficient pairs, giving at most twice the variance of an independent product. For even $N$, the two diagonal squares in an even-indexed coefficient have opposite signs, so their means cancel. A common standard-deviation budget for $d$ independent terms is

$$
\sigma_c\simeq\frac{2^{2K}\sqrt{2dN}}{12},
\qquad
\widehat p_{\mathrm{NTT}}=\operatorname{erfc}\!\left(\frac{Q/2}{\sqrt2\,\sigma_c}\right).
$$

The threshold is **$Q/2$**, the centered reconstruction limit. The product of two uniform inputs is not itself uniform; $\sigma_c$ uses the product's variance.

## FFT64: rounding error

The model covers forward transforms (shared when squaring), $d$ sequentially accumulated complex products, and one inverse transform. With $u=2^{-53}$ and $L=\log_2N-1$,

$$
R(N,d)=\frac{25}{3}L+\max\!\left(\frac23+\frac{d+1}{6}-\frac{1}{3d},\;\frac{d+1/2}{3}\right),
$$

$$
\sigma_e\simeq u\,\sigma_c\sqrt{R(N,d)},
\qquad
\widehat p_{\mathrm{FFT}}=\operatorname{erfc}\!\left(\frac{1/2}{\sqrt2\,\sigma_e}\right).
$$

Here $\sigma_e$ measures numerical error, and **$1/2$** is the rounding threshold. The model assumes centered relative roundoff of variance $u^2/3$, independent of inputs and other roundoff, approximately isotropic complex values, and twiddle errors of mean square at most $2u^2$ whose propagated contributions are treated as uncorrelated, including across terms. Squaring shares the forward error: $(X+e)^2-X^2\simeq2Xe$. Relative to the doubled coefficient-variance budget, the forward and inverse contributions are $4(5/3)L$ and $(5/3)L$, giving $25L/3$. The maximum in $R$ covers separate and fused multiply-add accumulation. These assumptions do not certify Gaussian far tails.

## Selecting the radix

For either model, the union bound gives $\widehat P_{\mathrm{any}}\le\min(1,N\widehat p)$. The helper uses the [Mills-ratio upper bound](https://dlmf.nist.gov/7.8.E2), which is tighter than $e^{-x^2}$:

$$
\operatorname{erfc}(x)\le\frac{2e^{-x^2}}{\sqrt\pi\left(x+\sqrt{x^2+4/\pi}\right)}
=\exp\!\left[-x^2-\operatorname{asinh}(\sqrt\pi\,x/2)\right],\qquad x\ge0.
$$

With $x=T/(\sqrt2\,\sigma)$ and $(T,\sigma)=(Q/2,\sigma_c)$ or $(1/2,\sigma_e)$, it searches for the largest integer $K$ satisfying $x^2+\operatorname{asinh}(\sqrt\pi\,x/2)\ge(\lambda+\log_2N)\ln2$. This evaluates the logarithm of the bound rather than the tiny probability itself.

The search caps the radix at `BE::ZnxWord::BITS - 2`: **62 for `i64`, 30 for a future `i32` backend**, and slightly reduces $x$ for a numerical margin at the threshold. The runtime query needs no module allocation; it returns `Some(K)`, `Some(0)` if no positive radix fits, or `None` without an applicable model. `N` must be a power of two at least 2 and the backend's minimum degree; `d` and `failure_bits` must be positive. For $m$ output polynomials, add $\lceil\log_2m\rceil$ to `failure_bits` to allocate a total failure budget.

## Why a probabilistic bound helps

A hard bound assumes every product has maximum magnitude and all signs reinforce each other. Independent centered inputs usually cancel: their standard deviation grows like $\sqrt{dN}$ instead of $dN$.

For **NTT4x30**, take $\log_2Q\simeq119.8861552574811$, $N=2^{16}$, $d=32$, and $K=53$. The hard bound is $dN\,2^{2K-2}=2^{125}>Q/2$, so it rejects this radix and requires $K\le49$. The Gaussian polynomial envelope is approximately **$2^{-1409.135}<2^{-128}$**, so the probabilistic selector admits **53 bits instead of 49**. For the same degree, count, and target, FFT64 selects **18 bits**, with a modeled envelope approximately $2^{-772.439}$. Both choices budget for squaring.

## Leave room for coefficient-domain additions

The word-size cap is not a budget for repeated additions. A sum or difference of $a$ normalized digits can reach magnitude $a\,2^{K-1}$; that intermediate must fit the coefficient word and the normalization contract. With `i64`, eight digits at $K=62$ can overflow, whereas $K=60$ bounds their sum by $2^{62}$, within the current normalization input range. Choose a smaller radix or normalize earlier when an addition chain needs more headroom.
