# Checkpoint 1 — Precision-matrix likelihood matches Kalman (PASSED)

**Validated.** `jointstar.computeLogLik` builds the stacked prior precision
`H_α = H'S⁻¹H` sparsely (block-tridiagonal, pentadiagonal where a second lag
enters) and matches a textbook Kalman filter to **better than 1e-8** across a
3×2×2 grid of `(ρ, σ_η, σ_ε)` on the AR(1)+noise toy, T=200 — including
nonzero initial mean/variance and missing-data masking (both a contiguous
early-sample gap and scattered gaps). 10,000 posterior state draws from
`drawStates` match the RTS smoother's mean and variance within 5-sigma Monte
Carlo bands at every t. Sparsity is preserved end-to-end (Cholesky factor
density 0.6% at full size).

**Timing (single core, JointSTAR-sized synthetic system: m=16, T=260, p=11,
AR(2) block, LDL′ covariance, ~35% missing):** **10.4 ms per likelihood
evaluation** vs the 50–100 ms target; 0.1 ms per state draw reusing the
factor. Headroom is ~5–10×, so 2000 particles × ~20 stages × 4 MH steps ≈
160k evaluations ≈ 30 core-minutes — comfortably inside the 30-minute budget
even before parallelisation.

**Surprise:** none material; MATLAB's sparse `chol` with the `'vector'`
fill-reducing permutation handles the pentadiagonal structure without any
hand-tuning.

**Next:** Checkpoint 2 needs the full model specification and the baseline
model's posterior mode — currently blocked; see `QUESTIONS.md`.
