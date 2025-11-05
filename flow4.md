# Definitive Strategic Plan (flow4): CT-FFR Implementation for DISCHARGE Trial

**Date**: November 5, 2025  
**Objective**: Provide a consolidated decision framework for deriving ΔFFR from coronary artery segmentations using open-source tools, leveraging unique data assets (lumen + outer wall STL, prognostic outcomes).

---

## 1. Executive Summary: Strategic Positioning

Your institution has a unique opportunity to advance non-invasive cardiac functional assessment with the DISCHARGE trial data. This document synthesizes four strategic pathways to achieve this, focusing on open-source solutions and optimal use of your data.

- **Unique Assets**: Dual segmentations (lumen + outer wall) enable innovative reference lumen modeling, while prognostic outcomes allow clinical impact studies.
- **Consensus**: A 1D reduced-order model is the most resource-efficient and scalable approach for most scenarios.
- **Recommended Path**: Start with a low-risk **Minimalist Proof-of-Concept (2-3 weeks)** to assess feasibility, then scale to a **Physics-Hybrid model (8-10 weeks)** for high-impact results.

---

## 2. Strategic Options: Comparative Overview

Below are the four distinct pathways derived from `future1-4.md`, each tailored to different resource levels and goals.

| Option | Core Engine | Key Feature | Timeline | Resources (Person-Hours) | Best For |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **A: Physics-Hybrid** | Custom Python (Lyras) | Adaptive Ensemble Reference | 8-12 weeks | ~320 | **High-impact publication & control** |
| **B: Solver-First** | svOneDSolver (SimVascular) | Dual Analytic/Solver Output | 6-8 weeks | ~300 | **Lower maintenance & robustness** |
| **C: ML-Driven** | XGBoost / LightGBM | Implicit Physics from Data | 10-14+ weeks| ~400 + HPC | **Data-rich scenarios (N≥100 FFR)** |
| **D: Minimalist POC** | Basic Python (Lyras) | Lumen-Only, Simple Taper | 2-3 weeks | ~80 | **Rapid feasibility testing** |

---

## 3. Data Requirements: Matching Assets to Strategy

Success hinges on aligning your data availability with the chosen pathway.

| Data Asset | Option A (Physics) | Option B (Solver) | Option C (ML) | Option D (POC) |
| :--- | :--- | :--- | :--- | :--- |
| **Lumen STL** | ✅ Required | ✅ Required | ✅ Required | ✅ Required |
| **Outer Wall STL** | ⭐ **Crucial Advantage** | ⭐ **Crucial Advantage** | ⭐ **Crucial Advantage** | Optional |
| **Invasive FFR** | **N=20-50** (calibration) | **N=20-30** (calibration) | **N≥100** (training) | **N=0-5** (sanity check) |
| **Prognostic Outcomes** | ✅ Required for impact | ✅ Required for impact | ✅ Required for impact | Optional |
| **Patient Metadata (MAP, Vessel Type)** | ✅ Recommended | ✅ Recommended | Optional (as feature) | Not required |

**Immediate Action**: Quantify your invasive FFR data. The number of cases with matched STL pairs and FFR values is the primary determinant for selecting between options.

---

## 4. Optimal Leverage of Unique Data Assets

### A. Segmented Data (Lumen + Outer Wall)

Your dual segmentations are a rare asset, enabling methodological breakthroughs.

1. **Physiologically Grounded Reference Lumen (Options A & B)**:
   - **Approach**: Use the outer wall to model the healthy artery by calculating median wall thickness in non-diseased segments (`T_wall = r_outer - r_lumen`), then define reference as `r_ref(s) = r_outer(s) - T_wall`.
   - **Impact**: Captures Glagov compensatory remodeling, offering a more accurate counterfactual than lumen-only methods—a publishable innovation.

2. **Feature Engineering for Machine Learning (Option C)**:
   - **Approach**: Extract unique geometric features from outer wall data, such as total plaque volume (`∫(A_outer - A_lumen)ds`), plaque burden, and remodeling index (`A_outer_lesion / A_outer_reference`).
   - **Impact**: Enhances ML model performance by providing richer lesion context.

3. **Quality Control Enhancement (All Options)**:
   - **Approach**: Use wall thickness as a segmentation quality metric, flagging non-physiologic values (<0.3mm or >2.0mm) for review.
   - **Impact**: Ensures reliable processing and reduces errors in downstream analysis.

### B. Prognostic Data (DISCHARGE Trial Outcomes)

This elevates your work from technical validation to clinical relevance.

1. **Prognostic Value of ΔFFR**:
   - **Method**: Employ Cox Proportional Hazards modeling to assess if ΔFFR predicts Major Adverse Cardiac Events (MACE) independently of anatomic stenosis (`Time-to-MACE ~ ΔFFR + % Stenosis + Covariates`).
   - **Impact**: Demonstrates clinical utility, strengthening publication potential.

2. **Optimal Clinical Cutoff**:
   - **Method**: Use ROC analysis with Youden's index to determine the ΔFFR threshold best separating high-risk patients (likely ~0.15-0.20).
   - **Impact**: Provides actionable clinical guidance.

3. **Decision-Impact Simulation**:
   - **Method**: Compare current diagnostic pathways (e.g., invasive angiography for % stenosis > 50%) with a CT-FFR-guided approach (e.g., refer only if ΔFFR > 0.18), quantifying avoided procedures and missed cases.
   - **Impact**: Highlights healthcare system benefits, enhancing manuscript appeal.

---

## 5. Optimizing Open-Source Resources

Your project can be executed at **zero software cost** using robust, community-supported tools.

- **Core Toolkit (All Options)**:
  - **VMTK**: Geometry extraction (centerlines, areas) with `vmtkCenterlines` and `vmtkCenterlineSections`.
  - **Python Stack**: `NumPy` for computations, `SciPy` for optimization, `Pandas` for data handling, `PyVista` for 3D visualization, `Matplotlib` for plots.
  - **Setup**: 
    ```bash
    conda create -n ctffr python=3.11 vmtk -c vmtk
    conda activate ctffr
    pip install numpy scipy pandas pyvista matplotlib
    ```

- **Option-Specific Tools**:
  - **SimVascular (svOneDSolver)**: For Option B, a validated 1D solver. Install from SimVascular.org or build from source.
  - **XGBoost/LightGBM**: For Option C, high-performance ML libraries (`pip install xgboost lightgbm shap`).
  - **OpenFOAM**: For small 3D CFD subsets in Options A/B (3-15 cases) or ground truth generation in Option C (100-200 cases). Install via `apt-get` on Ubuntu or from openfoam.org.

**Optimization Strategy**: Start with the core toolkit for initial testing (Option D). Add specific tools only after confirming the chosen pathway to avoid unnecessary setup overhead.

---

## 6. Recommended Strategic Path: Phased Implementation

This approach minimizes risk while maximizing learning and impact.

### **Phase 1: Minimalist Proof-of-Concept (Weeks 1-3)**
- **Action**: Implement **Option D** on 10-20 diverse cases to test feasibility.
- **Goal**: Confirm segmentation quality and processing pipeline viability with minimal investment.
- **Go/No-Go Decision**: Proceed if ≥80% of cases process successfully and ΔFFR values are plausible (0-0.5 range).

### **Phase 2: Full Pipeline Implementation (Weeks 4-12)**
- **Action**: Scale to **Option A (Physics-Hybrid)** as the default for full analysis.
- **Goal**: Develop a robust, validated pipeline with outer-wall informed reference lumen, calibration against invasive FFR, and uncertainty quantification.
- **Fallbacks**: Switch to **Option B (Solver-First)** if maintenance is a concern, or **Option C (ML-Driven)** if FFR data is abundant (N≥100) and ML expertise is available.

### **Phase 3: Clinical & Prognostic Analysis (Weeks 13-16)**
- **Action**: Apply the validated model to the full DISCHARGE cohort.
- **Goal**: Link ΔFFR to clinical outcomes via Cox modeling, determine optimal thresholds, and simulate clinical impact for manuscript preparation.

---

## 7. Decision Framework: Choosing Your Path

Answer these key questions post-POC to finalize your strategy:

1. **How many matched FFR/STL cases are available?**
   - **N < 20**: Calibration is limited. Proceed with Option A or B, acknowledging uncertainty, or generate CFD data.
   - **N = 20-100**: Ideal for **Option A (Physics-Hybrid)** or **Option B (Solver-First)**.
   - **N > 100**: Consider **Option C (ML-Driven)** for potentially superior accuracy.

2. **What are your team’s core competencies?**
   - **Strong in Python & physics**: **Option A** aligns best.
   - **Preference for established tools**: **Option B** reduces custom coding needs.
   - **Data science & ML expertise**: **Option C** offers an exciting opportunity.

3. **What is the primary objective?**
   - **High-impact paper with novelty**: **Option A** for outer-wall reference innovation.
   - **Robust, maintainable tool**: **Option B** for community-supported solver.
   - **Boundary-pushing accuracy**: **Option C** with highest potential AUC.
   - **Quick feasibility check**: **Option D** as a standalone or precursor.

**Default Recommendation**: Post-POC, pursue **Option A (Physics-Hybrid)** for its balance of innovation, interpretability, and impact, leveraging your unique outer wall data for a publishable methodological advance.

---

## 8. Immediate Action Items

### This Week (5-10 hours)
1. ✅ Quantify data assets: Count STL pairs (lumen + outer), invasive FFR cases, and prognostic outcome records.
2. ✅ Select 3-5 representative STL pairs for initial testing (diverse vessels and stenosis severity).
3. ✅ Assign team roles: Primary Investigator, programmer, clinical collaborator for FFR/outcome data.
4. ✅ Confirm computing resources: Access to laptop/workstation for initial POC.

### Week 1 (30-40 hours)
1. ✅ Install core environment: VMTK and Python stack for Option D POC.
2. ✅ Process 3-5 test cases: Extract geometry and compute basic ΔFFR.
3. ✅ Review initial QC: Check for processing errors or implausible results.
4. ✅ Plan POC expansion: Prepare for 10-20 case batch in Week 2.

---

## 9. Expected Outcomes

- **Technical Validation**: Achieve AUC >0.85-0.92 vs invasive FFR (depending on option and data).
- **Clinical Impact**: Demonstrate ΔFFR as an independent predictor of MACE, with actionable thresholds.
- **Publication**: Target high-impact journals (*European Heart Journal*, *JACC: Cardiovascular Imaging*) with a dual focus on methodological innovation (outer-wall reference) and clinical relevance (prognostic value).
- **Open-Source Contribution**: Release a citable GitHub repository (`discharge-ctffr`) with code, documentation, and sample data (if possible).

---

## 10. Conclusion

Your unique combination of lumen + outer wall segmentations and DISCHARGE trial outcomes positions you to make a significant contribution to CT-FFR research. The recommended phased approach—starting with a low-risk POC and scaling to a Physics-Hybrid pipeline—balances innovation with feasibility. The key to success is aligning your data assets and team resources with the chosen strategy.

**Next Step**: Approve the Week 1-3 POC (Option D, ~80 hours) to confirm feasibility before committing to a full pipeline.

**Decision Required**: Do you approve proceeding with the Week 1 pilot? [ ] YES [ ] NO [ ] NEED MORE INFO

**If YES, provide**:
1. Count of invasive FFR cases (N=?)
2. Paths to 3-5 representative STL pairs for testing
3. Assigned personnel (PI, programmer)
4. Target start date
