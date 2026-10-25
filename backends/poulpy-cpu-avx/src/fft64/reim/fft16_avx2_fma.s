# ----------------------------------------------------------------------
# This kernel is a direct port of the FFT16 routine from spqlios-arithmetic
# (https://github.com/tfhe/spqlios-arithmetic)
# ----------------------------------------------------------------------
#

# Body only: the object-format-specific symbol decoration (.globl, visibility,
# alignment, label, .size) is emitted by the `global_asm!` block in `mod.rs`.

# SysV args: %rdi = re*, %rsi = im*, %rdx = omg*
# stage 0: load inputs
vmovupd     (%rdi),%ymm0       # ra0
vmovupd     0x20(%rdi),%ymm1   # ra4
vmovupd     0x40(%rdi),%ymm2   # ra8
vmovupd     0x60(%rdi),%ymm3   # ra12
vmovupd     (%rsi),%ymm4       # ia0
vmovupd     0x20(%rsi),%ymm5   # ia4
vmovupd     0x40(%rsi),%ymm6   # ia8
vmovupd     0x60(%rsi),%ymm7   # ia12

# stage 1
vmovupd     (%rdx),%xmm12
vinsertf128 $1, %xmm12, %ymm12, %ymm12   # omriri
vshufpd     $15, %ymm12, %ymm12, %ymm13  # omai
vshufpd     $0,  %ymm12, %ymm12, %ymm12  # omar
vmulpd      %ymm6,%ymm13,%ymm8
vmulpd      %ymm7,%ymm13,%ymm9
vmulpd      %ymm2,%ymm13,%ymm10
vmulpd      %ymm3,%ymm13,%ymm11
vfmsub231pd %ymm2,%ymm12,%ymm8
vfmsub231pd %ymm3,%ymm12,%ymm9
vfmadd231pd %ymm6,%ymm12,%ymm10
vfmadd231pd %ymm7,%ymm12,%ymm11
vsubpd      %ymm8,%ymm0,%ymm2
vsubpd      %ymm9,%ymm1,%ymm3
vsubpd      %ymm10,%ymm4,%ymm6
vsubpd      %ymm11,%ymm5,%ymm7
vaddpd      %ymm8,%ymm0,%ymm0
vaddpd      %ymm9,%ymm1,%ymm1
vaddpd      %ymm10,%ymm4,%ymm4
vaddpd      %ymm11,%ymm5,%ymm5

# stage 2
vmovupd     16(%rdx),%xmm12
vinsertf128 $1, %xmm12, %ymm12, %ymm12   # omriri
vshufpd     $15, %ymm12, %ymm12, %ymm13  # omai
vshufpd     $0,  %ymm12, %ymm12, %ymm12  # omar
vmulpd      %ymm5,%ymm13,%ymm8
vmulpd      %ymm7,%ymm12,%ymm9
vmulpd      %ymm1,%ymm13,%ymm10
vmulpd      %ymm3,%ymm12,%ymm11
vfmsub231pd %ymm1,%ymm12,%ymm8
vfmadd231pd %ymm3,%ymm13,%ymm9
vfmadd231pd %ymm5,%ymm12,%ymm10
vfmsub231pd %ymm7,%ymm13,%ymm11
vsubpd      %ymm8,%ymm0,%ymm1
vaddpd      %ymm9,%ymm2,%ymm3
vsubpd      %ymm10,%ymm4,%ymm5
vaddpd      %ymm11,%ymm6,%ymm7
vaddpd      %ymm8,%ymm0,%ymm0
vsubpd      %ymm9,%ymm2,%ymm2
vaddpd      %ymm10,%ymm4,%ymm4
vsubpd      %ymm11,%ymm6,%ymm6

# stage 3
vmovupd     0x20(%rdx),%ymm12
vshufpd     $15, %ymm12, %ymm12, %ymm13  # omai
vshufpd     $0,  %ymm12, %ymm12, %ymm12  # omar

vperm2f128  $0x31,%ymm2,%ymm0,%ymm8
vperm2f128  $0x31,%ymm3,%ymm1,%ymm9
vperm2f128  $0x31,%ymm6,%ymm4,%ymm10
vperm2f128  $0x31,%ymm7,%ymm5,%ymm11
vperm2f128  $0x20,%ymm2,%ymm0,%ymm0
vperm2f128  $0x20,%ymm3,%ymm1,%ymm1
vperm2f128  $0x20,%ymm6,%ymm4,%ymm2
vperm2f128  $0x20,%ymm7,%ymm5,%ymm3

vmulpd      %ymm10,%ymm13,%ymm4
vmulpd      %ymm11,%ymm12,%ymm5
vmulpd      %ymm8,%ymm13,%ymm6
vmulpd      %ymm9,%ymm12,%ymm7
vfmsub231pd %ymm8,%ymm12,%ymm4
vfmadd231pd %ymm9,%ymm13,%ymm5
vfmadd231pd %ymm10,%ymm12,%ymm6
vfmsub231pd %ymm11,%ymm13,%ymm7
vsubpd      %ymm4,%ymm0,%ymm8
vaddpd      %ymm5,%ymm1,%ymm9
vsubpd      %ymm6,%ymm2,%ymm10
vaddpd      %ymm7,%ymm3,%ymm11
vaddpd      %ymm4,%ymm0,%ymm0
vsubpd      %ymm5,%ymm1,%ymm1
vaddpd      %ymm6,%ymm2,%ymm2
vsubpd      %ymm7,%ymm3,%ymm3

# stage 4
vmovupd     0x40(%rdx),%ymm12
vmovupd     0x60(%rdx),%ymm13

vunpckhpd   %ymm1,%ymm0,%ymm4
vunpckhpd   %ymm3,%ymm2,%ymm6
vunpckhpd   %ymm9,%ymm8,%ymm5
vunpckhpd   %ymm11,%ymm10,%ymm7
vunpcklpd   %ymm1,%ymm0,%ymm0
vunpcklpd   %ymm3,%ymm2,%ymm2
vunpcklpd   %ymm9,%ymm8,%ymm1
vunpcklpd   %ymm11,%ymm10,%ymm3

vmulpd      %ymm6,%ymm13,%ymm8
vmulpd      %ymm7,%ymm12,%ymm9
vmulpd      %ymm4,%ymm13,%ymm10
vmulpd      %ymm5,%ymm12,%ymm11
vfmsub231pd %ymm4,%ymm12,%ymm8
vfmadd231pd %ymm5,%ymm13,%ymm9
vfmadd231pd %ymm6,%ymm12,%ymm10
vfmsub231pd %ymm7,%ymm13,%ymm11
vsubpd      %ymm8,%ymm0,%ymm4
vaddpd      %ymm9,%ymm1,%ymm5
vsubpd      %ymm10,%ymm2,%ymm6
vaddpd      %ymm11,%ymm3,%ymm7
vaddpd      %ymm8,%ymm0,%ymm0
vsubpd      %ymm9,%ymm1,%ymm1
vaddpd      %ymm10,%ymm2,%ymm2
vsubpd      %ymm11,%ymm3,%ymm3

vunpckhpd   %ymm7,%ymm3,%ymm11
vunpckhpd   %ymm5,%ymm1,%ymm9
vunpcklpd   %ymm7,%ymm3,%ymm10
vunpcklpd   %ymm5,%ymm1,%ymm8
vunpckhpd   %ymm6,%ymm2,%ymm3
vunpckhpd   %ymm4,%ymm0,%ymm1
vunpcklpd   %ymm6,%ymm2,%ymm2
vunpcklpd   %ymm4,%ymm0,%ymm0

vperm2f128  $0x31,%ymm10,%ymm2,%ymm6
vperm2f128  $0x31,%ymm11,%ymm3,%ymm7
vperm2f128  $0x20,%ymm10,%ymm2,%ymm4
vperm2f128  $0x20,%ymm11,%ymm3,%ymm5
vperm2f128  $0x31,%ymm8,%ymm0,%ymm2
vperm2f128  $0x31,%ymm9,%ymm1,%ymm3
vperm2f128  $0x20,%ymm8,%ymm0,%ymm0
vperm2f128  $0x20,%ymm9,%ymm1,%ymm1

# stores
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
