# ----------------------------------------------------------------------
# This kernel is a direct port of the IFFT16 routine from spqlios-arithmetic
# (https://github.com/tfhe/spqlios-arithmetic)
# ----------------------------------------------------------------------
#

# Body only: the object-format-specific symbol decoration (.globl, visibility,
# alignment, label, .size) is emitted by the `global_asm!` block in `mod.rs`.

vmovupd     (%rdi),%ymm0       # ra0
vmovupd     0x20(%rdi),%ymm1   # ra4
vmovupd     0x40(%rdi),%ymm2   # ra8
vmovupd     0x60(%rdi),%ymm3   # ra12
vmovupd     (%rsi),%ymm4       # ia0
vmovupd     0x20(%rsi),%ymm5   # ia4
vmovupd     0x40(%rsi),%ymm6   # ia8
vmovupd     0x60(%rsi),%ymm7   # ia12

1:
vmovupd     0x00(%rdx),%ymm12
vmovupd     0x20(%rdx),%ymm13

vperm2f128 $0x31,%ymm2,%ymm0,%ymm8   # ymm8 contains re to mul (tw)
vperm2f128 $0x31,%ymm3,%ymm1,%ymm9   # ymm9 contains re to mul (itw)
vperm2f128 $0x31,%ymm6,%ymm4,%ymm10   # ymm10 contains im to mul (tw)
vperm2f128 $0x31,%ymm7,%ymm5,%ymm11   # ymm11 contains im to mul (itw)
vperm2f128 $0x20,%ymm2,%ymm0,%ymm0   # ymm0 contains re to add (tw)
vperm2f128 $0x20,%ymm3,%ymm1,%ymm1   # ymm1 contains re to add (itw)
vperm2f128 $0x20,%ymm6,%ymm4,%ymm2   # ymm2 contains im to add (tw)
vperm2f128 $0x20,%ymm7,%ymm5,%ymm3   # ymm3 contains im to add (itw)

vunpckhpd %ymm1,%ymm0,%ymm4   # (0,1) -> (0,4)
vunpckhpd %ymm3,%ymm2,%ymm6   # (2,3) -> (2,6)
vunpckhpd %ymm9,%ymm8,%ymm5   # (8,9) -> (1,5)
vunpckhpd %ymm11,%ymm10,%ymm7 # (10,11) -> (3,7)
vunpcklpd %ymm1,%ymm0,%ymm0
vunpcklpd %ymm3,%ymm2,%ymm2
vunpcklpd %ymm9,%ymm8,%ymm1
vunpcklpd %ymm11,%ymm10,%ymm3

# invctwiddle Re:(ymm0,ymm4) and Im:(ymm2,ymm6) with omega=(ymm12,ymm13)
# invcitwiddle Re:(ymm1,ymm5) and Im:(ymm3,ymm7) with omega=(ymm12,ymm13)
vsubpd      %ymm4,%ymm0,%ymm8  # retw
vsubpd      %ymm5,%ymm1,%ymm9  # reitw
vsubpd      %ymm6,%ymm2,%ymm10 # imtw
vsubpd      %ymm7,%ymm3,%ymm11 # imitw
vaddpd      %ymm4,%ymm0,%ymm0
vaddpd      %ymm5,%ymm1,%ymm1
vaddpd      %ymm6,%ymm2,%ymm2
vaddpd      %ymm7,%ymm3,%ymm3
# multiply 8,9,10,11 by 12,13, result to: 4,5,6,7
# twiddles use reom=ymm12, imom=ymm13
# invtwiddles use reom=ymm13, imom=-ymm12
vmulpd      %ymm10,%ymm13,%ymm4 # imtw.omai  (tw)
vmulpd      %ymm11,%ymm12,%ymm5 # imitw.omar (itw)
vmulpd      %ymm8,%ymm13,%ymm6  # retw.omai  (tw)
vmulpd      %ymm9,%ymm12,%ymm7  # reitw.omar (itw)
vfmsub231pd %ymm8,%ymm12,%ymm4  # rprod0 (tw)
vfmadd231pd %ymm9,%ymm13,%ymm5  # rprod4 (itw)
vfmadd231pd %ymm10,%ymm12,%ymm6 # iprod0 (tw)
vfmsub231pd %ymm11,%ymm13,%ymm7 # iprod4 (itw)

vunpckhpd %ymm7,%ymm3,%ymm11   # (0,4) -> (0,1)
vunpckhpd %ymm5,%ymm1,%ymm9    # (2,6) -> (2,3)
vunpcklpd %ymm7,%ymm3,%ymm10
vunpcklpd %ymm5,%ymm1,%ymm8
vunpckhpd %ymm6,%ymm2,%ymm3    # (1,5) -> (8,9)
vunpckhpd %ymm4,%ymm0,%ymm1    # (3,7) -> (10,11)
vunpcklpd %ymm6,%ymm2,%ymm2
vunpcklpd %ymm4,%ymm0,%ymm0

2:
vmovupd     0x40(%rdx),%ymm12
vshufpd     $15, %ymm12, %ymm12, %ymm13  # ymm13: omaiii'i'
vshufpd     $0,  %ymm12, %ymm12, %ymm12  # ymm12: omarrr'r'

# invctwiddle Re:(ymm0,ymm8) and Im:(ymm2,ymm10) with omega=(ymm12,ymm13)
# invcitwiddle Re:(ymm1,ymm9) and Im:(ymm3,ymm11) with omega=(ymm12,ymm13)
vsubpd      %ymm8,%ymm0,%ymm4  # retw
vsubpd      %ymm9,%ymm1,%ymm5  # reitw
vsubpd      %ymm10,%ymm2,%ymm6 # imtw
vsubpd      %ymm11,%ymm3,%ymm7 # imitw
vaddpd      %ymm8,%ymm0,%ymm0
vaddpd      %ymm9,%ymm1,%ymm1
vaddpd      %ymm10,%ymm2,%ymm2
vaddpd      %ymm11,%ymm3,%ymm3
# multiply 4,5,6,7 by 12,13, result to 8,9,10,11
# twiddles use reom=ymm12, imom=ymm13
# invtwiddles use reom=ymm13, imom=-ymm12
vmulpd      %ymm6,%ymm13,%ymm8  # imtw.omai (tw)
vmulpd      %ymm7,%ymm12,%ymm9  # imitw.omar (itw)
vmulpd      %ymm4,%ymm13,%ymm10 # retw.omai (tw)
vmulpd      %ymm5,%ymm12,%ymm11 # reitw.omar (itw)
vfmsub231pd %ymm4,%ymm12,%ymm8 # rprod0 (tw)
vfmadd231pd %ymm5,%ymm13,%ymm9 # rprod4 (itw)
vfmadd231pd %ymm6,%ymm12,%ymm10 # iprod0 (tw)
vfmsub231pd %ymm7,%ymm13,%ymm11 # iprod4 (itw)

vperm2f128 $0x31,%ymm10,%ymm2,%ymm6
vperm2f128 $0x31,%ymm11,%ymm3,%ymm7
vperm2f128 $0x20,%ymm10,%ymm2,%ymm4
vperm2f128 $0x20,%ymm11,%ymm3,%ymm5
vperm2f128 $0x31,%ymm8,%ymm0,%ymm2
vperm2f128 $0x31,%ymm9,%ymm1,%ymm3
vperm2f128 $0x20,%ymm8,%ymm0,%ymm0
vperm2f128 $0x20,%ymm9,%ymm1,%ymm1

3:
vmovupd     0x60(%rdx),%xmm12
vinsertf128 $1, %xmm12, %ymm12, %ymm12   # omriri
vshufpd     $15, %ymm12, %ymm12, %ymm13  # ymm13: omai
vshufpd     $0,  %ymm12, %ymm12, %ymm12  # ymm12: omar

# invctwiddle Re:(ymm0,ymm1) and Im:(ymm4,ymm5) with omega=(ymm12,ymm13)
# invcitwiddle Re:(ymm2,ymm3) and Im:(ymm6,ymm7) with omega=(ymm12,ymm13)
vsubpd      %ymm1,%ymm0,%ymm8   # retw
vsubpd      %ymm3,%ymm2,%ymm9   # reitw
vsubpd      %ymm5,%ymm4,%ymm10  # imtw
vsubpd      %ymm7,%ymm6,%ymm11  # imitw
vaddpd      %ymm1,%ymm0,%ymm0
vaddpd      %ymm3,%ymm2,%ymm2
vaddpd      %ymm5,%ymm4,%ymm4
vaddpd      %ymm7,%ymm6,%ymm6
# multiply 8,9,10,11 by 12,13, result to 1,3,5,7
# twiddles use reom=ymm12, imom=ymm13
# invtwiddles use reom=ymm13, imom=-ymm12
vmulpd      %ymm10,%ymm13,%ymm1 # imtw.omai (tw)
vmulpd      %ymm11,%ymm12,%ymm3 # imitw.omar (itw)
vmulpd      %ymm8,%ymm13,%ymm5  # retw.omai (tw)
vmulpd      %ymm9,%ymm12,%ymm7  # reitw.omar (itw)
vfmsub231pd %ymm8,%ymm12,%ymm1 # rprod0 (tw)
vfmadd231pd %ymm9,%ymm13,%ymm3 # rprod4 (itw)
vfmadd231pd %ymm10,%ymm12,%ymm5 # iprod0 (tw)
vfmsub231pd %ymm11,%ymm13,%ymm7 # iprod4 (itw)

4:
vmovupd     0x70(%rdx),%xmm12
vinsertf128 $1, %xmm12, %ymm12, %ymm12   # omriri
vshufpd     $15, %ymm12, %ymm12, %ymm13  # ymm13: omai
vshufpd     $0,  %ymm12, %ymm12, %ymm12  # ymm12: omar

# invctwiddle Re:(ymm0,ymm2) and Im:(ymm4,ymm6) with omega=(ymm12,ymm13)
# invctwiddle Re:(ymm1,ymm3) and Im:(ymm5,ymm7) with omega=(ymm12,ymm13)
vsubpd      %ymm2,%ymm0,%ymm8  # retw1
vsubpd      %ymm3,%ymm1,%ymm9  # retw2
vsubpd      %ymm6,%ymm4,%ymm10 # imtw1
vsubpd      %ymm7,%ymm5,%ymm11 # imtw2
vaddpd      %ymm2,%ymm0,%ymm0
vaddpd      %ymm3,%ymm1,%ymm1
vaddpd      %ymm6,%ymm4,%ymm4
vaddpd      %ymm7,%ymm5,%ymm5
# multiply 8,9,10,11 by 12,13, result to 2,3,6,7
# twiddles use reom=ymm12, imom=ymm13
vmulpd      %ymm10,%ymm13,%ymm2 # imtw1.omai
vmulpd      %ymm11,%ymm13,%ymm3 # imtw2.omai
vmulpd      %ymm8,%ymm13,%ymm6  # retw1.omai
vmulpd      %ymm9,%ymm13,%ymm7  # retw2.omai
vfmsub231pd %ymm8,%ymm12,%ymm2 # rprod0
vfmsub231pd %ymm9,%ymm12,%ymm3 # rprod4
vfmadd231pd %ymm10,%ymm12,%ymm6 # iprod0
vfmadd231pd %ymm11,%ymm12,%ymm7 # iprod4

5:
vmovupd     %ymm0,(%rdi)       # ra0
vmovupd     %ymm1,0x20(%rdi)   # ra4
vmovupd     %ymm2,0x40(%rdi)   # ra8
vmovupd     %ymm3,0x60(%rdi)   # ra12
vmovupd     %ymm4,(%rsi)       # ia0
vmovupd     %ymm5,0x20(%rsi)   # ia4
vmovupd     %ymm6,0x40(%rsi)   # ia8
vmovupd     %ymm7,0x60(%rsi)   # ia12
vzeroupper
ret
