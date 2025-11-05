# Alternative Plan: Solver-First, Minimal-Maintenance ΔFFR Pipeline

**Date**: November 5, 2025  
**Purpose**: Provide a second, pragmatic pathway leveraging existing open-source solvers with simpler geometry assumptions to minimize custom code and maintenance.

---

## Executive Summary

- Use SimVascular's svOneDSolver as the primary engine (validated 1D solver).  
- Keep reference lumen simple: outer-wall–constrained exponential taper (no ensemble).  
- Offer two compute modes:
  - Track A: ΔFFR-fast via analytical Lyras formula (seconds/artery).
  - Track B: ΔFFR-1D via svOneDSolver (1–3 min/artery) for robustness/cross-check.  
- Boundary conditions from Murray’s law and literature hyperemia.  
- Minimal 3D CFD (3–5 cases) only if calibration data are scarce.  

Outcome: Lower engineering burden, easier maintenance, credible results with open-source tooling.

---

## What’s Different vs future1.md

- **Solver-first**: Primary reliance on svOneDSolver instead of a custom 1D implementation.
- **Simpler reference lumen**: Single hybrid method (outer-wall constrained exponential taper) instead of adaptive ensemble.
- **Two-track outputs**: Fast analytic and robust 1D-solver ΔFFR reported side-by-side.
- **Even leaner 3D**: 3–5-case verification instead of 10–15.

---

## Tooling (100% Open-Source)

- Geometry: VMTK (centerlines, cross-sectional areas)
- 1D Solver: SimVascular svOneDSolver
- Visualization/I-O: PyVista, NumPy/SciPy, Pandas, Matplotlib
- Optional 3D: OpenFOAM (tiny subset)

---

## Pipeline Overview

```
STL_lumen + STL_outer
      ↓  (VMTK)
Centerline s + A_lumen(s) + A_outer(s)
      ↓  (Reference Lumen)
Outer-wall–constrained exponential taper → A_ref(s)
      ↓  (Lesion metrics)
A_min, L_lesion, % stenosis
      ↓
Track A: Lyras ΔP → ΔFFR_fast
Track B: svOneDSolver network → proximal/distal pressures → ΔFFR_1D
      ↓
ΔFFR_fast, ΔFFR_1D, QC & CI
```

---

## Geometry & Reference Lumen (Simplified)

1. Extract centerline and A_lumen(s) with VMTK; sample outer wall along same path (nearest-point distance).
2. Identify healthy segments (low gradient, lumen ≈ smoothed lumen).
3. Compute healthy wall thickness: median(r_outer − r_lumen) in healthy zones; clamp 0.4–1.2 mm.
4. Fit exponential taper to healthy diameters: D_ref(s) = D0·exp(−λ·s).  
5. Enforce outer-wall constraint: r_ref(s) = min{ r_outer(s) − T_wall, D_ref(s)/2 }, then ensure r_ref ≥ 1.05·r_lumen in stenosis to avoid non-physiology.
6. Define A_ref(s) = π·r_ref(s)^2.  

Uncertainty (lightweight): Monte-Carlo on T_wall (±0.2 mm), λ (±20%), and Q (±15%).

---

## Boundary Conditions

- Aortic inlet pressure: P_a = patient MAP if available, else 93 mmHg.
- Hyperemic flow: Q = Q_rest × 3.5; if myocardial mass available → 1.0 mL/min/g; else vessel defaults: LAD 1.2, RCA 1.0, LCx 0.8 mL/s baseline before hyperemia.
- Flow split for branches (if present): Murray exponent α ≈ 2.7–3.0 using proximal areas.
- Distal microvascular model (svOneDSolver):
  - Minimal: pure resistance R_out = P_a / Q_hyperemia.
  - Preferred: 3-element Windkessel (R1, C, R2) from literature scaling; reduce R_total by ~75% for hyperemia.

---

## Track A: ΔFFR-fast (Analytical)

- Model: Lyras reduced-order formula
- Inputs: A_min (mm²), L_lesion (mm), Q_hyperemic (m³/s), μ≈0.004 Pa·s, ρ≈1050 kg/m³.
- Output: ΔP_fast (mmHg), ΔFFR_fast = ΔP_fast / P_a.
- Calibration: affine correction against invasive FFR subset (a·ΔP + b) when available.
- Runtime: seconds per artery.

Use as a triage and for cases where svOneDSolver setup fails/QC flags.

---

## Track B: ΔFFR-1D (svOneDSolver)

- Build a 1D network from the VMTK centerline of the segmented segment (single vessel or small tree).
- Assign segment properties: length, proximal/distal areas, blood density/viscosity.
- Inlet: prescribed P_a.
- Outlet(s): R or RCR elements tuned for hyperemia as above.
- Solve steady (or quasi-steady) hyperemic flow; extract proximal/distal pressures around the lesion; compute ΔP_1D and ΔFFR_1D.
- Runtime: 1–3 minutes per vessel on a laptop.

Why this helps: Handles entrance/exit losses and gradual/diffuse disease more realistically than a single A_min.

---

## QC & Uncertainty

- Geometry QC: min length ≥ 20 mm, no self-intersections, outer–lumen gap within [0.3, 1.5] mm median.
- Reference QC: ensure A_ref ≥ A_lumen and taper monotonicity except near ostia.
- Solver QC: check mass balance, non-negative pressures, physiologic velocities (0.1–2 m/s under hyperemia).
- UQ: 100-sample MC varying T_wall, λ, Q, and outlet R; report median and 5–95% CI for ΔFFR_fast and ΔFFR_1D.

---

## Validation & Calibration (Lean)

- Invasive FFR subset: N = 20–50 (preferred).  
  - Calibrate (a, b) for ΔP_fast.  
  - Optionally tune RCR scaling factors to match observed ΔP.
- Minimal 3D CFD: N = 3–5 difficult cases (severe stenosis, tortuosity).  
  - Confirm ΔP_1D bias < 3 mmHg; otherwise apply small correction.
- Reproducibility: N = 20 re-processed by a second operator; target ΔFFR ICC > 0.9.

---

## Outputs

Per lesion (CSV):
- id, ΔP_fast, ΔFFR_fast, ΔFFR_fast_CI_low/high, ΔP_1D, ΔFFR_1D, ΔFFR_1D_CI_low/high, A_min, L_lesion, QC_flags

Study-level: ROC/AUC vs invasive FFR<0.80, calibration plot, Bland–Altman, uncertainty distributions.

---

## Timeline (6–8 Weeks)

- Week 1: Environment + svOneDSolver install; VMTK scripts; process 5 pilot cases (Track A).
- Week 2: Build 1D network exporter; run 5 pilot cases in svOneDSolver (Track B); QC rules.
- Week 3: Process 20–30 calibration cases; fit (a, b) for ΔP_fast; tune RCR scaling.
- Week 4: Validation on 50–100 cases; generate ROC/AUC; compare Track A vs Track B.
- Week 5: Minimal 3D CFD (3–5 cases) and finalize corrections if needed.
- Week 6: Batch all remaining arteries; produce final CSV and QC report.
- Weeks 7–8 (optional): Manuscript drafting and figures.

---

## Resources

- Personnel: ~240–300 hours total (slightly less than future1 due to simpler reference and solver reuse).
- Compute: Laptop for Track A; standard workstation for Track B batch; no HPC required.
- Cost: €0 software.

---

## Go/No-Go Checks

- After Week 2: ≥80% solver runs successful; ΔFFR_fast and ΔFFR_1D within 0.05 on median.
- After Week 4: AUC ≥ 0.88 vs invasive FFR; CI width ≤ 0.06 for most cases.

---

## Minimal Implementation Notes

- 1D network generation: export centerline points and local area to svOneDSolver format (.1d) with one inlet, one (or few) outlets. If branches missing, treat as single terminal vessel; set R_out = P_a/Q.
- Reference lumen: store T_wall, λ used per case for auditability.
- Parallelization: multiprocessing for Track A; jobfile list for Track B.

---

## When to Prefer This Plan

- You want lower maintenance and to stand on a widely used open-source solver.
- You’re comfortable with a slightly simpler reference lumen assumption.
- You value dual outputs (fast analytic + solver) for internal cross-validation.

---

## Decision

Proceed with this solver-first plan if installation of svOneDSolver is feasible on your systems and you prefer minimized custom modeling. Otherwise, default to future1.md (custom Python core) as the primary path.
