.text
.globl  ntt126_ifma_b_to_znx128_asm
.hidden ntt126_ifma_b_to_znx128_asm
.p2align 4, 0x90
.type   ntt126_ifma_b_to_znx128_asm,@function
ntt126_ifma_b_to_znx128_asm:
# SysV args:
#   %rdi = coefficient count, rounded down to a multiple of 4
#   %rsi = destination i128*
#   %rdx = source Q126b u64* with 4 u64 lanes per coefficient
    testq   %rdi, %rdi
    je      .Ldone_empty

    pushq   %rbx
    pushq   %rbp
    pushq   %r12
    pushq   %r13
    pushq   %r14
    pushq   %r15

    movq    %rdx, %rbp
    movq    %rsi, %r12
    shrq    $2, %rdi
    movq    %rdi, %r15

    vmovdqa64 ntt126_ifma_qvec(%rip), %ymm24
    vpbroadcastq ntt126_ifma_q1(%rip), %ymm25
    vpbroadcastq ntt126_ifma_q2(%rip), %ymm26
    vpbroadcastq ntt126_ifma_q2x2(%rip), %ymm27
    vpbroadcastq ntt126_ifma_inv01(%rip), %ymm28
    vpbroadcastq ntt126_ifma_inv01_quot(%rip), %ymm29
    vpbroadcastq ntt126_ifma_inv012(%rip), %ymm30
    vpbroadcastq ntt126_ifma_inv012_quot(%rip), %ymm31
    vpbroadcastq ntt126_ifma_q0_mod_q2_quot(%rip), %ymm21
    vpbroadcastq ntt126_ifma_mask52(%rip), %ymm22
    vpbroadcastq ntt126_ifma_q0_mod_q2(%rip), %ymm23

    movabsq $0x3ffffe80001, %r11
    movabsq $0xe8000c4fffba0001, %r10
    movabsq $0xc8dfe0ec007bffff, %r14
    movabsq $0xc00007bfff83b004, %rdi

.macro RECOMP_ONE offset
    # Inputs: %r8 = v0, %rdx = v1, %rsi = v2.
    mulx    %r11, %rax, %rbx

    movq    %rsi, %rdx
    mulx    %r10, %rcx, %r13

    movq    %rsi, %r9
    leaq    (%rsi,%rsi), %rdx
    shlq    $20, %r9
    subq    %rdx, %r9

    xorl    %edx, %edx
    adcx    %r8, %rax
    adcx    %rdx, %rbx
    adox    %r9, %r13
    adcx    %rax, %rcx
    adox    %rbx, %r13
    adcx    %rdx, %r13

    movq    %r14, %rax
    addq    %rcx, %rax
    movq    %rdi, %rbx
    adcq    %r13, %rbx

    cmpq    %rcx, ntt126_ifma_half_big_lo(%rip)
    movq    ntt126_ifma_half_big_hi(%rip), %r9
    sbbq    %r13, %r9
    cmovae  %rcx, %rax
    cmovae  %r13, %rbx

    movq    %rax, \offset(%r12)
    movq    %rbx, \offset+8(%r12)
.endm

.p2align 4, 0x90
.Lloop4:
    # Load four AoS coefficients and reduce [0, 2q) to [0, q).
    vmovdqu 0(%rbp), %ymm0
    vmovdqu 32(%rbp), %ymm1
    vmovdqu 64(%rbp), %ymm2
    vmovdqu 96(%rbp), %ymm3

    vpsubq  %ymm24, %ymm0, %ymm4
    vpminuq %ymm4, %ymm0, %ymm0
    vpsubq  %ymm24, %ymm1, %ymm4
    vpminuq %ymm4, %ymm1, %ymm1
    vpsubq  %ymm24, %ymm2, %ymm4
    vpminuq %ymm4, %ymm2, %ymm2
    vpsubq  %ymm24, %ymm3, %ymm4
    vpminuq %ymm4, %ymm3, %ymm3

    # Transpose AoS residues into per-prime vectors.
    vpunpcklqdq %ymm1, %ymm0, %ymm4
    vpunpckhqdq %ymm1, %ymm0, %ymm5
    vpunpcklqdq %ymm3, %ymm2, %ymm6
    vpunpckhqdq %ymm3, %ymm2, %ymm7
    vperm2i128 $0x20, %ymm6, %ymm4, %ymm8
    vperm2i128 $0x20, %ymm7, %ymm5, %ymm9
    vperm2i128 $0x31, %ymm6, %ymm4, %ymm10

    # v1 = (r1 - v0) * inv(Q0 mod Q1) mod Q1.
    vpsubq  %ymm25, %ymm8, %ymm12
    vpminuq %ymm12, %ymm8, %ymm12
    vpaddq  %ymm25, %ymm9, %ymm13
    vpsubq  %ymm12, %ymm13, %ymm13
    vpsubq  %ymm25, %ymm13, %ymm12
    vpminuq %ymm12, %ymm13, %ymm12

    vpxorq  %xmm13, %xmm13, %xmm13
    vpmadd52huq %ymm29, %ymm12, %ymm13
    vpxorq  %xmm11, %xmm11, %xmm11
    vpmadd52luq %ymm28, %ymm12, %ymm11
    vpxorq  %xmm14, %xmm14, %xmm14
    vpmadd52luq %ymm25, %ymm13, %ymm14
    vpsubq  %ymm14, %ymm11, %ymm11
    vpandq  %ymm22, %ymm11, %ymm11
    vpsubq  %ymm25, %ymm11, %ymm12
    vpminuq %ymm12, %ymm11, %ymm11

    # v2 = (r2 - (v0 + v1 * Q0) mod Q2) * inv(Q0Q1 mod Q2) mod Q2.
    vpsubq  %ymm26, %ymm8, %ymm12
    vpminuq %ymm12, %ymm8, %ymm12

    vpxorq  %xmm13, %xmm13, %xmm13
    vpmadd52huq %ymm21, %ymm11, %ymm13
    vpxorq  %xmm14, %xmm14, %xmm14
    vpmadd52luq %ymm23, %ymm11, %ymm14
    vpxorq  %xmm15, %xmm15, %xmm15
    vpmadd52luq %ymm26, %ymm13, %ymm15
    vpsubq  %ymm15, %ymm14, %ymm14
    vpandq  %ymm22, %ymm14, %ymm14

    vpaddq  %ymm14, %ymm12, %ymm12
    vpsubq  %ymm27, %ymm12, %ymm13
    vpminuq %ymm13, %ymm12, %ymm12
    vpsubq  %ymm26, %ymm12, %ymm13
    vpminuq %ymm13, %ymm12, %ymm12

    vpaddq  %ymm26, %ymm10, %ymm13
    vpsubq  %ymm12, %ymm13, %ymm13
    vpsubq  %ymm26, %ymm13, %ymm12
    vpminuq %ymm12, %ymm13, %ymm12

    vpxorq  %xmm13, %xmm13, %xmm13
    vpmadd52huq %ymm31, %ymm12, %ymm13
    vpxorq  %xmm15, %xmm15, %xmm15
    vpmadd52luq %ymm30, %ymm12, %ymm15
    vpxorq  %xmm14, %xmm14, %xmm14
    vpmadd52luq %ymm26, %ymm13, %ymm14
    vpsubq  %ymm14, %ymm15, %ymm15
    vpandq  %ymm22, %ymm15, %ymm15
    vpsubq  %ymm26, %ymm15, %ymm14
    vpminuq %ymm14, %ymm15, %ymm15

    # Recompose Garner digits into signed i128 outputs.
    vmovq   %xmm8, %r8
    vmovq   %xmm11, %rdx
    vmovq   %xmm15, %rsi
    RECOMP_ONE 0

    vpextrq $1, %xmm8, %r8
    vpextrq $1, %xmm11, %rdx
    vpextrq $1, %xmm15, %rsi
    RECOMP_ONE 16

    vextracti128 $1, %ymm8, %xmm0
    vextracti128 $1, %ymm11, %xmm1
    vextracti128 $1, %ymm15, %xmm2
    vmovq   %xmm0, %r8
    vmovq   %xmm1, %rdx
    vmovq   %xmm2, %rsi
    RECOMP_ONE 32

    vpextrq $1, %xmm0, %r8
    vpextrq $1, %xmm1, %rdx
    vpextrq $1, %xmm2, %rsi
    RECOMP_ONE 48

    addq    $128, %rbp
    addq    $64, %r12
    decq    %r15
    jne     .Lloop4

    vzeroupper
    popq    %r15
    popq    %r14
    popq    %r13
    popq    %r12
    popq    %rbp
    popq    %rbx
.Ldone_empty:
    retq

.size ntt126_ifma_b_to_znx128_asm, .-ntt126_ifma_b_to_znx128_asm

.section .rodata.cst32,"aM",@progbits,32
.p2align 5
ntt126_ifma_qvec:
    .quad 0x3ffffe80001
    .quad 0x3ffffd20001
    .quad 0x3ffffca0001
    .quad 0x0

.section .rodata.cst8,"aM",@progbits,8
.p2align 3
ntt126_ifma_q1:
    .quad 0x3ffffd20001
ntt126_ifma_q2:
    .quad 0x3ffffca0001
ntt126_ifma_q2x2:
    .quad 0x7ffff940002
ntt126_ifma_inv01:
    .quad 0x5d17131748
ntt126_ifma_inv01_quot:
    .quad 0x1745c5d1745d1
ntt126_ifma_inv012:
    .quad 0xbbbacb6ef7
ntt126_ifma_inv012_quot:
    .quad 0x2eeeb55554444
ntt126_ifma_q0_mod_q2:
    .quad 0x1e0000
ntt126_ifma_q0_mod_q2_quot:
    .quad 0x78000654
ntt126_ifma_mask52:
    .quad 0xfffffffffffff
ntt126_ifma_half_big_lo:
    .quad 0x9b900f89ffc20000
ntt126_ifma_half_big_hi:
    .quad 0x1ffffc20003e27fd
