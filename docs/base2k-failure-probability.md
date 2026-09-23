# Failure estimates for base-2^K arithmetic

`Module::<BE>::max_base2k(N, d, failure_bits)` delegates to `MaxBase2k` to select a radix for one output polynomial $C=\sum_{r=1}^{d}A_rB_r$ in $\mathbb Z[X]/(X^N+1)$. Here $d$ counts accumulated polynomial products, and `failure_bits` $=\lambda$ requests an estimated probability at most $2^{-\lambda}$ that any output coefficient fails.

The models assume independent, centered uniform input coefficients of magnitude at most $2^{K-1}$, neglect discrete endpoint corrections, and approximate output tails as Gaussian. These estimates are not guarantees for arbitrary or correlated inputs, including reused or squared operands.

## NTT: centered reconstruction

Let $Q$ be the actual product of CRT primes. Each output coefficient sums $dN$ products; their variances add, giving

$$
\sigma_c\simeq\frac{2^{2K}\sqrt{dN}}{12},
\qquad
\widehat p_{\mathrm{NTT}}=\operatorname{erfc}\!\left(\frac{Q/2}{\sqrt2\,\sigma_c}\right).
$$

The threshold is **$Q/2$**, the centered reconstruction limit. The product of two uniform inputs is not itself uniform; $\sigma_c$ uses the product's variance.

## FFT64: rounding error

The model covers two forward transforms per product, $d$ sequentially accumulated complex products, and one inverse transform. With $u=2^{-53}$ and $L=\log_2N-1$,

$$
R(N,d)=5L+\max\!\left(\frac23+\frac{d+1}{6}-\frac{1}{3d},\;\frac{d+1/2}{3}\right),
$$

$$
\sigma_e\simeq u\,\sigma_c\sqrt{R(N,d)},
\qquad
\widehat p_{\mathrm{FFT}}=\operatorname{erfc}\!\left(\frac{1/2}{\sqrt2\,\sigma_e}\right).
$$

Here $\sigma_e$ measures numerical error, and **$1/2$** is the rounding threshold. The model assumes centered, uncorrelated relative roundoff of variance $u^2/3$, approximately isotropic complex values, and uncorrelated twiddle errors of mean square at most $2u^2$. The maximum in $R$ covers separate and fused multiply-add accumulation. These assumptions do not certify Gaussian far tails.

## Selecting the radix

For either model, the union bound gives $\widehat P_{\mathrm{any}}\le\min(1,N\widehat p)$. The helper uses $\operatorname{erfc}(x)\le e^{-x^2}$ and selects the largest integer $K$ satisfying

$$
N\exp\!\left(-\frac{T^2}{2\sigma^2}\right)\le2^{-\lambda},
\qquad
(T,\sigma)=(Q/2,\sigma_c)\ \text{or}\ (1/2,\sigma_e).
$$

It rounds down with a small numerical margin and caps the result at `BE::ZnxWord::BITS - 2`: **62 for `i64`, 30 for a future `i32` backend**. The runtime query needs no module allocation; it returns `Some(K)`, `Some(0)` if no positive radix fits, or `None` without an applicable model. `N` must be a power of two at least the backend's minimum degree; `d` and `failure_bits` must be positive. For $m$ output polynomials, add $\lceil\log_2m\rceil$ to `failure_bits` to allocate a total failure budget.

## Why a probabilistic bound helps

A hard bound assumes every product has maximum magnitude and all signs reinforce each other. Independent centered inputs usually cancel: their standard deviation grows like $\sqrt{dN}$ instead of $dN$.

For **NTT4x30**, take $\log_2Q\simeq119.8861552574811$, $N=2^{16}$, $d=32$, and $K=54$. The hard bound is $dN\,2^{2K-2}=2^{127}>Q/2$, so it rejects this radix and requires $K\le49$. The Gaussian polynomial envelope is approximately **$2^{-161.417}<2^{-128}$**, so the probabilistic selector admits **54 bits instead of 49**. For the same degree, count, and target, FFT64 selects **19 bits**, with a modeled envelope approximately $2^{-138.903}$.

## Leave room for coefficient-domain additions

The word-size cap is not a budget for repeated additions. A sum or difference of $a$ normalized digits can reach magnitude $a\,2^{K-1}$; that intermediate must fit the coefficient word and the normalization contract. With `i64`, eight digits at $K=62$ can overflow, whereas $K=60$ bounds their sum by $2^{62}$, within the current normalization input range. Choose a smaller radix or normalize earlier when an addition chain needs more headroom.
