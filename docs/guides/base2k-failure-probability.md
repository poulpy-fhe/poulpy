# Failure estimates for base-2^K arithmetic

`Module::<BE>::max_base2k(N, d, failure_bits, squaring)` delegates to `MaxBase2k` to select a radix for one output polynomial $C=\sum_{r=1}^{d}A_rB_r$ in $\mathbb Z[X]/(X^N+1)$, or in the conjugate-invariant ring of degree $N$ ([below](#conjugate-invariant-ring)). Here $d$ counts accumulated polynomial products, and `failure_bits` $=\lambda$ requests an estimated probability at most $2^{-\lambda}$ that any output coefficient fails.

Use `squaring = false` for independent products, and `true` if any term is a square $A_r^2$. Count each square once: a single square uses `d = 1`. The square model conservatively budgets every term as a square. Distinct terms must use independent inputs, with independent centered uniform coefficients of magnitude at most $2^{K-1}$. The models neglect discrete endpoint corrections and approximate output tails as Gaussian. These estimates are not guarantees for arbitrary inputs or other correlations, including operand reuse across terms.

## NTT: centered reconstruction

Let $Q$ be the actual product of CRT primes. Squaring repeats off-diagonal coefficient pairs, giving at most twice the variance of an independent product. For even $N$, the two diagonal squares in an even-indexed coefficient have opposite signs, so their means cancel. Let $s=2$ when `squaring` is true and $s=1$ otherwise. The standard-deviation budget for $d$ independent terms is

$$
\sigma_c\simeq\frac{2^{2K}\sqrt{sdN}}{12},
\qquad
\widehat p_{\mathrm{NTT}}=\operatorname{erfc}\!\left(\frac{Q/2}{\sqrt2\,\sigma_c}\right).
$$

The threshold is **$Q/2$**, the centered reconstruction limit. The product of two uniform inputs is not itself uniform; $\sigma_c$ uses the product's variance. For NTT, setting `squaring = true` is equivalent to doubling $d$ in the independent-product model; the helper does this internally.

## FFT64: rounding error

The model covers forward transforms (shared when squaring), $d$ sequentially accumulated complex products, and one inverse transform. With $u=2^{-53}$, $L=\log_2N-1$, and $\alpha=25/3$ for squares or $5$ for independent products,

$$
R(N,d)=\alpha L+\max\!\left(\frac23+\frac{d+1}{6}-\frac{1}{3d},\;\frac{d+1/2}{3}\right),
$$

$$
\sigma_e\simeq u\,\sigma_c\sqrt{R(N,d)},
\qquad
\widehat p_{\mathrm{FFT}}=\operatorname{erfc}\!\left(\frac{1/2}{\sqrt2\,\sigma_e}\right).
$$

Here $\sigma_e$ measures numerical error, and **$1/2$** is the rounding threshold. The model assumes centered relative roundoff of variance $u^2/3$, independent of inputs and other roundoff, approximately isotropic complex values, and twiddle errors of mean square at most $2u^2$ whose propagated contributions are treated as uncorrelated, including across terms. Squaring shares the forward error: $(X+e)^2-X^2\simeq2Xe$. Relative to the doubled coefficient-variance budget, the forward and inverse contributions are $4(5/3)L$ and $(5/3)L$, giving $25L/3$. Thus doubling $d$ alone does not model FFT squaring. The maximum in $R$ uses the actual accumulation count and covers separate and fused multiply-add accumulation. These assumptions do not certify Gaussian far tails.

## Conjugate-invariant ring

An element is $a_0+\sum_{k=1}^{N-1}a_k(X^k+X^{-k})$ in $\mathbb Z[X]/(X^{2N}+1)$. With $e_k$ the basis elements, output coefficient $c$ of one product has variance $\sigma^4\sum_{k,m}(e_ke_m)_c^2$: **$4N-3$** at $c=0$, where every $a_kb_k$ with $k\ge1$ lands twice, and $2N-2$ elsewhere. A square's coefficient $c$ has mean $\sigma^2\sum_k(e_ke_k)_c$, which is **$2N-1$** at $c=0$ and zero elsewhere. Both models bound every coefficient by the constant one.

**NTT.** The variance budget uses $4N-3$ in place of $N$. A square's mean, budgeted for all $d$ terms, $\mu=d(2N-1)2^{2K}/12$, comes off the threshold:

$$
\widehat p_{\mathrm{NTT}}=\operatorname{erfc}\!\left(\frac{Q/2-\mu}{\sqrt2\,\sigma_c}\right),\qquad \mu<Q/2.
$$

The mean grows like $dN$ and $\sigma_c$ like $\sqrt{dN}$, so for squares the mean, not the tail, sets the radix.

**FFT64.** The transforms are a DCT-III and a DCT-II through a cyclic FFT of $N/2$ points. Their pre- and post-processing take two rotations and two additions per value, about $14/3$ in the accounting above, budgeted as three more stages: $\rho=\frac53(L+3)$ per transform. The slots are real, with variance $V=(2N-1)\sigma^2$. Independent slot errors of variance $E$ reach the constant coefficient with variance $E/N$, twice what the other coefficients get, in the proportion of the output's own variances. For products, with real multiplications and sequential accumulation (a fused accumulation rounds less on real slots),

$$
E=u^2V^2d\left(3\rho+\frac13+\frac{d+1}{6}-\frac1{3d}\right).
$$

A square's slots $y^2$ have $\mathbb E[y^4]\le3V^2$, the shared forward error contributes $\mathbb E[(2ye)^2]\le4\rho u^2\,\mathbb E[y^4]$ per term, and a partial sum of $i$ squares has second moment $(i^2+2i)V^2$:

$$
E=u^2V^2\left(\rho\,d(d+14)+d-1+\frac{d(d+1)(2d+7)}{18}\right).
$$

The squares' slots are nonnegative, so they also sum coherently into the constant coefficient, through additions only. That sum's roundoff stays there instead of spreading over the output: its variance is at most $2u^2/3$ times the squared mean $(dV)^2$. Then

$$
\sigma_e^2=\frac EN+\frac23u^2d^2V^2\quad\text{(the last term for squares only)},
$$

with threshold $1/2$ as above.

For $N=2^{16}$, $d=32$ and $\lambda=128$, NTT4x30 selects **53** bits and FFT64 **18**. For a single square, NTT4x30 selects **52**: at $K=53$, $\mu=(2^{17}-1)2^{106}/12$ exceeds $Q/2$ by itself. FFT64 selects **17**.

## Selecting the radix

For either model, the union bound gives $\widehat P_{\mathrm{any}}\le\min(1,N\widehat p)$. The helper uses the [Mills-ratio upper bound](https://dlmf.nist.gov/7.8.E2), which is tighter than $e^{-x^2}$:

$$
\operatorname{erfc}(x)\le\frac{2e^{-x^2}}{\sqrt\pi\left(x+\sqrt{x^2+4/\pi}\right)}
=\exp\!\left[-x^2-\operatorname{asinh}(\sqrt\pi\,x/2)\right],\qquad x\ge0.
$$

With $x=(T-\mu)/(\sqrt2\,\sigma)$, $(T,\sigma)=(Q/2,\sigma_c)$ or $(1/2,\sigma_e)$, and $\mu$ zero except for conjugate-invariant NTT squares, it searches for the largest integer $K$ satisfying $x^2+\operatorname{asinh}(\sqrt\pi\,x/2)\ge(\lambda+\log_2N)\ln2$. This evaluates the logarithm of the bound rather than the tiny probability itself.

The search caps the radix at `BE::ZnxWord::BITS - 2`: **62 for `i64`, 30 for a future `i32` backend**, and slightly reduces $x$ for a numerical margin at the threshold. The runtime query needs no module allocation; it returns $K$, or 0 if no positive radix fits; a backend without an applicable model does not implement `MaxBase2k`. `N` must be a power of two at least 2 and the backend's minimum degree; `d` and `failure_bits` must be positive. For $m$ output polynomials, add $\lceil\log_2m\rceil$ to `failure_bits` to allocate a total failure budget.

## Why a probabilistic bound helps

A hard bound assumes every product has maximum magnitude and all signs reinforce each other. Independent centered inputs usually cancel: their standard deviation grows like $\sqrt{dN}$ instead of $dN$.

For **NTT4x30**, take $\log_2Q\simeq119.8861552574811$, $N=2^{16}$, $d=32$, and $K=54$ with `squaring = false`. The hard bound is $dN\,2^{2K-2}=2^{127}>Q/2$, so it rejects this radix and requires $K\le49$. The Gaussian polynomial envelope is approximately **$2^{-165.718}<2^{-128}$**, so the probabilistic selector admits **54 bits instead of 49**. For the same degree, count, and target, FFT64 selects **19 bits**, with a modeled envelope approximately $2^{-143.107}$.

## Guard limbs of gadget products

A gadget product (key-switch or external product) is normalized from a window of limbs: the live precision plus $g$ guard limbs. The limbs below the window are dropped. They act as an error on the last live limb, and the same model sizes $g$.

One accumulated coefficient has standard deviation $\sigma_c$ with $dN$ replaced by the number of accumulated coefficient products. The first dropped limb weighs $2^{-(g+1)K}$ units of the last live limb, and all dropped limbs together have deviation at most $\sqrt{4/3}$ times that of the first:

$$
\sigma_g\le\sqrt{4/3}\;\sigma_c\,2^{-(g+1)K},
\qquad
\widehat p_g=\operatorname{erfc}\!\left(\frac{1/2}{\sqrt2\,\sigma_g}\right).
$$

The threshold is **$1/2$**, the rounding threshold of the last live limb. `gadget_product_guard_limbs` returns the smallest $g$ whose union bound over the product meets $\lambda=128$, with the Mills-ratio bound below. For $N=2^{16}$, 28 gadget limbs and $K=52$ this gives $g=2$, where making the last live limb exact in the worst case takes 3.

## Leave room for coefficient-domain additions

The word-size cap is not a budget for repeated additions. A sum or difference of $a$ normalized digits can reach magnitude $a\,2^{K-1}$ in the worst case; that intermediate must fit the coefficient word and the normalization contract, whose input range is $[-2^{62},2^{62}]$ for `i64`. With independent centered uniform digits the sum is again close to Gaussian, with

$$
\sigma_a\simeq\frac{2^K\sqrt a}{\sqrt{12}},
\qquad
\widehat p_a=\operatorname{erfc}\!\left(\frac{2^{62}}{\sqrt2\,\sigma_a}\right),
$$

and the same Mills-ratio and union bounds give the largest $a$ for a target $\lambda$ over $m$ coefficients. A lazy sum may hold the larger of that count and the worst case $2^{63-K}$, which the Gaussian tail underestimates for a few terms near the word size. `poulpy_core::layouts::max_lazy_terms` returns it with $\lambda=128$ over the coefficients it is given. For $m=2^{16}\cdot2\cdot20$ coefficients, $K=52$ holds 62526 digits where the worst case allows 2048, $K=56$ holds 244 against 128, and from $K=57$ up the worst case binds: 8 digits at $K=60$, 2 at $K=62$. As everywhere in this document, correlated inputs, such as many copies of one value, are outside the model.
