# Definitive Strategic Plan (flow2): CT-FFR Options, Data Needs, and Open-Source Strategy

**Date**: November 5, 2025  
**Context**: DISCHARGE trial, hundreds of arteries with lumen + outer-wall STL, prognostic outcomes available  
**Goal**: Deliver ΔFFR per lesion using open-source methods with defensible validation and minimal cost

---

## 1) Executive Overview

You have four viable pathways. The default path is a staged rollout that starts fast and scales safely:
- **Default**: Week 1–2 Minimalist POC → Week 3–10 Physics-Hybrid → Week 11–14 Prognostic analysis

---

## 2) Options at a Glance

- **Option A — Physics-Hybrid (future1)**
  - Core: Custom Python Lyras model + outer-wall–informed reference (ensemble) + Monte Carlo UQ
  - Pros: Methodological novelty, full control, fast (<90 s/artery)
  - Cons: Moderate code maintenance, needs N≥20–50 FFR for calibration
  - Typical: 8–12 weeks, 320 hours, AUC target >0.90

- **Option B — Solver-First (future2)**
  - Core: svOneDSolver (SimVascular) + simple constrained taper reference; dual outputs (analytic + 1D)
  - Pros: Lower maintenance, validated solver, internal cross-check
  - Cons: 1–3 min/artery, svOneDSolver install/formatting
  - Typical: 6–8 weeks, 240–300 hours, AUC target >0.88

- **Option C — ML-Driven (future3)**
  - Core: XGBoost/LightGBM using 100+ geometric/plaque features; labels from invasive FFR or 3D CFD
  - Pros: May outperform 1D for complex anatomies; instant inference at scale
  - Cons: Data-hungry (N≥100 labeled), less interpretable, upfront effort
  - Typical: 10–14 weeks, 350–400 hours (+HPC if CFD), AUC >0.92 possible

- **Option D — Minimalist POC (future4)**
  - Core: Lumen-only + simple taper + uncalibrated Lyras
  - Pros: Fastest (2–3 weeks), lowest effort (60–100 hours)
  - Cons: No novelty, limited validation, coarse BCs
  - Use: Feasibility check before committing

---

## 3) Data Requirements and Readiness

- **Segmentations**
  - Lumen STL (required), Outer wall STL (highly recommended, strategic advantage)
  - Coverage ≥20 mm per segment, reasonable spacing, artifact review
- **Clinical labels**
  - Invasive FFR (for calibration/validation)
    - Minimum: 10–20 (Options A/B)
    - Optimal: 50+ (Option A best), 30+ (Option B)
    - ML path (Option C): 100–200
  - Prognostic outcomes (MACE, revasc., etc.) for Cox models (100–500+)
- **Patient metadata**
  - Mean aortic pressure (MAP) or BP, vessel type (LAD/LCx/RCA), optional myocardial mass

Readiness checklist (this week):
- Count STL pairs (lumen + outer), map to IDs/vessels
- Count invasive FFR cases matched to STL
- Confirm outcome dataset with follow-up and censoring rules
- Spot-check 10 cases for segmentation quality

---

## 4) Open-Source Stack (Optimal Use)

- **Core (all options)**
  - VMTK: centerlines, cross-sectional areas
  - Python: NumPy, SciPy, Pandas, Matplotlib/Seaborn, PyVista
  - Boundary conditions: Murray’s law scaling, hyperemia factor ×3.5–4, P_a ≈ 93 mmHg
  
  Setup
  ```bash
  conda create -n ctffr python=3.11 vmtk pyvista scipy numpy pandas tqdm -c vmtk
  conda activate ctffr
  pip install matplotlib seaborn scikit-learn
  ```

- **Option-specific**
  - Option B: SimVascular svOneDSolver (validated 1D)
  - Option C: XGBoost/LightGBM + SHAP; optional OpenFOAM for CFD labels
  - Optional 3D subset (A/B): OpenFOAM 3–15 cases for verification

---

## 5) Optimal Use of Segmented Data (Your Edge)

- **Outer-wall–informed reference lumen (Options A/B)**
  - Healthy wall thickness T_wall from non-diseased segments
  - Reference radius r_ref(s) = r_outer(s) − T_wall (clamped ≥1.05×r_lumen)
  - Ensemble (A) or simple constrained taper (B)
  - Benefit: Captures Glagov remodeling; more realistic counterfactual lumen

- **Plaque burden metrics (Option C)**
  - Plaque area = A_outer − A_lumen; volume by ∫ plaque_area ds
  - Remodeling index, max thickness, burden indices → powerful ML features

- **QC enabled by outer wall**
  - Physiologic wall thickness [0.3–1.5] mm median
  - Flag outliers (segmentation repair/curation before batch)

- **Partial tree support**
  - ΔFFR across a focal lesion does not require the complete tree; use Murray for outlet splits if branches present

---

## 6) Optimal Use of Prognostic Data

- **Primary**: Cox proportional hazards
  - Time-to-MACE ~ ΔFFR + % stenosis + age + risk factors
  - Test incremental value of ΔFFR over anatomy (LR test, ΔC-index)
- **Thresholding**: Determine optimal ΔFFR cutoff (Youden index; expect ~0.15–0.20)
- **Decision-impact simulation**: Compare current care vs CT-FFR-guided pathways (avoided angiograms, missed ischemia, NNT)
- **Subgroups**: Diabetes, proximal vs distal, multi-vessel, age strata

---

## 7) Recommended Rollout (Definitive)

- **Phase 1 (Weeks 1–2): Option D POC**
  - Process 10–20 mixed cases; lumen-only + simple taper + uncalibrated Lyras
  - Deliverable: Feasibility report, QC plots, ΔFFR distribution
  - Go/No-Go: ≥80% success, plausible ΔP (0–50 mmHg)

- **Phase 2 (Weeks 3–10): Option A Physics-Hybrid**
  - Implement outer-wall–informed reference (ensemble), calibrated Lyras, UQ
  - Calibrate K1/K2 and affine correction on N≥20–50 invasive FFR
  - Optional: Verify 5–10 cases with OpenFOAM (if concerns)

- **Phase 3 (Weeks 11–14): Prognostic Analysis**
  - Cox models, thresholds, decision impact, manuscript figures

Fallbacks:
- If solver maintenance preferred → Option B
- If N≥100 labeled or HPC + ML expertise → Option C

---

## 8) Decision Matrix (Condensed)

- **Fastest start**: D
- **Highest novelty + control**: A
- **Lowest maintenance**: B
- **Potential best accuracy (data-rich)**: C
- **Least data dependence**: D → A/B

---

## 9) Immediate Actions (This Week)

- **Data**
  - Inventory STL pairs and link to patient IDs/vessels
  - Count invasive FFR matched to STL (N=?)
  - Confirm outcomes dataset and definitions (MACE)
- **Pilot**
  - Select 3–5 representative STL pairs (LAD/LCx/RCA; mild/moderate/severe)
  - Install environment and run VMTK extraction
- **Team**
  - Assign PI, programmer, and clinical collaborator

---

## 10) Deliverables

- Phase 1: Feasibility memo + small CSV (ΔP, ΔFFR, A_min, L_lesion, QC)
- Phase 2: Trial-scale CSV + UQ + QC flags; calibration report (ROC/AUC, Bland–Altman)
- Phase 3: Prognostic analysis notebook; decision-impact results; manuscript draft

---

## 11) Key Technical Defaults (for consistency)

- Blood viscosity μ = 0.004 Pa·s; density ρ = 1050 kg/m³
- Hyperemia = ×3.5–4 of rest
- Aortic mean pressure P_a = 93 mmHg (or patient MAP)
- Murray exponent α ≈ 2.7–3.0 for branch flow allocation

---

## 12) Final Recommendation

Start with **Option D POC (2 weeks)** to derisk geometry/processing, then execute **Option A** for full-scale analysis and publication. Use **Option B** if you prefer solver maintainability, or **Option C** if you possess ≥100 labeled cases/HPC + ML expertise. Leverage your **outer-wall segmentations** and **prognostic outcomes** to produce both a technical and clinical-impact paper using a fully open-source stack.
