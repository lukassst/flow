# Definitive Strategic Plan (flow3): CT-FFR Pathways for the DISCHARGE Trial

**Date**: November 5, 2025  
**Objective**: Synthesize all prior analyses (`future1-4`) into a clear decision framework for deriving ΔFFR from your unique dataset (lumen + outer wall STL, prognostic outcomes) using open-source tools.

---

## 1. Executive Summary: Your Strategic Position

Your institution is uniquely positioned to make a high-impact contribution to non-invasive cardiac functional assessment. You have four viable, open-source pathways to achieve this.

- **Your Core Advantage**: The combination of **lumen + outer wall segmentations** and **prognostic clinical outcomes** from the DISCHARGE trial. This allows for both methodological innovation and clinical validation beyond what most studies can achieve.
- **The Consensus**: All analyses confirm that a **1D reduced-order model** is the most resource-efficient and scalable approach for your primary goal.
- **The Recommended Path**: A phased approach is wisest. Start with a **Minimalist Proof-of-Concept (2-3 weeks)** to de-risk the project, then scale up to a **Physics-Hybrid model (8-10 weeks)** for a high-impact publication.

---

## 2. The Four Strategic Options: A Comparative Overview

Here are the four distinct strategies, distilled from `future1-4.md`.

| Option | Core Engine | Key Feature | Timeline | Resources (Person-Hours) | Best For... |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **A: Physics-Hybrid** | Custom Python (Lyras) | Adaptive Ensemble Reference | 8-12 weeks | ~320 | **High-impact publication & control** |
| **B: Solver-First** | svOneDSolver (SimVascular) | Dual Analytic/Solver Output | 6-8 weeks | ~300 | **Lower maintenance & robustness** |
| **C: ML-Driven** | XGBoost / LightGBM | Implicit Physics from Data | 10-14+ weeks| ~400 + HPC | **Data-rich scenarios (N≥100 FFR)** |
| **D: Minimalist POC** | Basic Python (Lyras) | Lumen-Only, Simple Taper | 2-3 weeks | ~80 | **Rapid feasibility testing** |

---

## 3. Data Requirements: What You Need for Each Pathway

Your success depends on matching your available data to the chosen strategy.

| Data Asset | Option A (Physics) | Option B (Solver) | Option C (ML) | Option D (POC) |
| :--- | :--- | :--- | :--- | :--- |
| **Lumen STL** | ✅ Required | ✅ Required | ✅ Required | ✅ Required |
| **Outer Wall STL** | ⭐ **Crucial Advantage** | ⭐ **Crucial Advantage** | ⭐ **Crucial Advantage** | Optional |
| **Invasive FFR** | **N=20-50** (for calibration) | **N=20-30** (for calibration) | **N≥100** (for training) | **N=0-5** (for sanity check) |
| **Prognostic Outcomes** | ✅ Required for clinical paper| ✅ Required for clinical paper| ✅ Required for clinical paper| Optional |
| **Patient MAP** | ✅ Recommended | ✅ Recommended | Optional (can be a feature) | Not required |

**Immediate Action**: Your first step is to **quantify your invasive FFR data**. The number of cases with both STL pairs and invasive FFR is the single most important factor in choosing between these options.

---

## 4. Optimal Leverage of Your Unique Data Assets

### A. Leveraging Your Segmented Data (Lumen + Outer Wall)

This is your **methodological innovation**. Most literature lacks outer wall data.

1.  **Physiologically-Aware Reference Lumen (Options A & B)**:
    - **Concept**: Model the healthy, non-diseased artery by accounting for Glagov's compensatory remodeling, which is only possible with the outer wall boundary.
    - **Implementation**: Calculate the median healthy wall thickness (`T_wall = r_outer - r_lumen`) in non-diseased segments and use it to define a reference lumen under the plaque: `r_ref(s) = r_outer(s) - T_wall`.
    - **Impact**: This is a more accurate counterfactual than simple tapering and is a key publishable finding.

2.  **Rich Feature Engineering for Machine Learning (Option C)**:
    - **Concept**: The outer wall unlocks powerful geometric features that correlate with hemodynamic significance.
    - **Features**: Total Plaque Volume (`∫(A_outer - A_lumen)ds`), Plaque Burden (`Plaque Volume / Outer Wall Volume`), Remodeling Index (`A_outer_lesion / A_outer_reference`), and max plaque thickness.
    - **Impact**: These features provide the ML model with a much richer understanding of the lesion's 3D structure.

3.  **Enhanced Quality Control (All Options)**:
    - **Concept**: Use the wall thickness as a sanity check for segmentation quality.
    - **Implementation**: Flag any segments where the calculated wall thickness is non-physiologic (e.g., <0.3mm or >2.0mm) for manual review.

### B. Leveraging Your Prognostic Data (DISCHARGE Trial Outcomes)

This is your **clinical innovation**. It elevates the project from a technical validation to a clinical impact study.

1.  **Establish Prognostic Value of ΔFFR**:
    - **Primary Goal**: Use a **Cox Proportional Hazards model** to prove that your calculated ΔFFR is an independent predictor of Major Adverse Cardiac Events (MACE), even after accounting for traditional anatomical metrics like % stenosis.
    - **Hypothesis**: `Time-to-MACE ~ ΔFFR + % Stenosis + Age + ...` where the coefficient for ΔFFR is statistically significant (p < 0.05).

2.  **Determine an Optimal Clinical Cutoff**:
    - **Method**: Use ROC analysis (Youden's index) to find the ΔFFR threshold that best separates patients with and without adverse outcomes. This defines a clinically actionable cutoff (e.g., a ΔFFR > 0.18 indicates high risk).

3.  **Simulate Clinical Impact**:
    - **Method**: Compare the current diagnostic pathway (e.g., refer to invasive angiography if % stenosis > 50%) with a proposed CT-FFR-guided pathway (e.g., refer only if ΔFFR > 0.18). Quantify the number of potentially avoided invasive procedures and the number of missed significant lesions.

---

## 5. Optimizing Your Open-Source Toolkit

Your project can be completed with **zero software cost**.

- **Core Stack (All Options)**:
  - **VMTK**: The workhorse for geometry. Use `vmtkCenterlines` to get the vessel path and radius profile.
  - **Python Scientific Stack**: `NumPy` for calculations, `SciPy` for fitting/interpolation, `Pandas` for data management, `PyVista` for 3D visualization, and `Matplotlib` for 2D plots.
  - **Setup**: `conda create -n ctffr python=3.11 vmtk -c vmtk` is the only complex installation step.

- **Option-Specific Additions**:
  - **SimVascular (`svOneDSolver`)**: For Option B. A robust, community-supported 1D solver.
  - **XGBoost/LightGBM**: For Option C. High-performance gradient boosting libraries.
  - **OpenFOAM**: For optional 3D CFD verification on a small subset of cases (3-15) to build confidence in the 1D model's accuracy.

---

## 6. The Recommended Strategic Path: A Phased Approach

This sequence maximizes learning while minimizing initial risk and resource commitment.

### **Phase 1: Minimalist Proof-of-Concept (Weeks 1-3)**
- **Action**: Implement **Option D** on 10-20 diverse cases.
- **Goal**: De-risk the project. Confirm that your segmentations are suitable for automated processing and that the basic physics produce plausible results.
- **Go/No-Go Decision**: At the end of this phase, if processing is successful and results are logical, proceed. If not, pause to address data quality issues.

### **Phase 2: Full Pipeline Implementation (Weeks 4-12)**
- **Action**: Based on your data and resources, scale up to one of the main options. **Option A (Physics-Hybrid) is the recommended default**.
- **Goal**: Build the robust, validated pipeline. This involves implementing the outer-wall reference method, calibrating the model against your invasive FFR data, and performing uncertainty quantification.

### **Phase 3: Clinical & Prognostic Analysis (Weeks 13-16)**
- **Action**: Apply the validated model to your full DISCHARGE cohort.
- **Goal**: Connect the calculated ΔFFR to the clinical outcomes. Run the Cox models, determine the optimal cutoff, and simulate the clinical impact. This phase generates the core findings for your main publication.

---

## 7. Final Decision Framework

To choose your path after the POC, answer these questions:

1.  **How many matched FFR/STL cases do we have?**
    - If **N < 20**: Your ability to calibrate is limited. Stick with Option A/B but acknowledge this limitation, or generate CFD data.
    - If **N is 20-100**: You are perfectly positioned for **Option A or B**.
    - If **N > 100**: You have the rare opportunity to pursue **Option C (ML-Driven)**, which could yield the highest accuracy.

2.  **What is our team's primary skillset?**
    - If **strong in Python & physics**: **Option A** is a natural fit.
    - If **prefer to rely on external tools**: **Option B** reduces the custom coding burden.
    - If **strong in data science & ML**: **Option C** is the most exciting path.

3.  **What is our primary goal?**
    - If **a high-impact paper with methodological novelty**: **Option A**'s outer-wall reference is a strong story.
    - If **a robust, maintainable internal tool**: **Option B** is the most practical.
    - If **pushing the boundaries of accuracy**: **Option C** has the highest ceiling.

**Default Recommendation**: The combination of your unique data and the goals of a research institution strongly favors **Option A (Physics-Hybrid)** as the primary target after a successful POC, as it offers the best balance of innovation, interpretability, and impact.
