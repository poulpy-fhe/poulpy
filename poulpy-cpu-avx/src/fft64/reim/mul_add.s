# ymm14 is scratch; the FFT16 kernels use ymm0 through ymm13.
.macro poulpy_madd a, b, c
.if {fused}
    vfmadd231pd \a,\b,\c
.else
    vmulpd \a,\b,%ymm14
    vaddpd %ymm14,\c,\c
.endif
.endm

.macro poulpy_msub a, b, c
.if {fused}
    vfmsub231pd \a,\b,\c
.else
    vmulpd \a,\b,%ymm14
    vsubpd \c,%ymm14,\c
.endif
.endm
