# Definitive Strategic Plan: CT-FFR for DISCHARGE Trial
## Comprehensive Decision Framework & Implementation Roadmap

**Date**: November 5, 2025  
**Institution**: Research institution with DISCHARGE trial data  
**Unique Assets**: Lumen + outer wall segmentations, prognostic outcomes, large European cohort

---

# Executive Summary

You have **four viable pathways** to derive functional pressure drop estimates (ΔFFR) from your coronary artery segmentations. This document provides a decision framework to select the optimal approach based on your specific constraints and goals.

## Your Unique Strategic Advantages

1. **Dual Segmentations**: Lumen + outer wall (rare in literature) → enables Glagov remodeling modeling
2. **Prognostic Data**: DISCHARGE trial outcomes → can correlate ΔFFR with real-world clinical endpoints
3. **Scale**: Hundreds of cases → sufficient for validation studies and ML approaches
4. **Open-Source Context**: No budget for commercial software → leverages free tools optimally

---

# Four Strategic Options

## Option 1: Physics-Hybrid (future1.md) ⭐ RECOMMENDED FOR MOST SCENARIOS

**Best For**: Balance of sophistication, control, and publication impact

### Core Approach
- Custom Python implementation of Lyras 2021 reduced-order model
- Adaptive ensemble reference lumen (3 methods: taper, outer-wall constraint, spline)
- Monte Carlo uncertainty quantification
- Calibration on invasive FFR subset

### Timeline & Resources
- **8 weeks** to results, **12 weeks** to manuscript
- **320 person-hours** (phased over 10 weeks)
- **Standard workstation** (no HPC required)
- **Processing speed**: 30-90 sec/artery

### Data Requirements
| Data Type | Minimum | Optimal | Purpose |
|-----------|---------|---------|---------|
| **STL pairs (lumen + outer)** | 20 | 500+ | Full analysis |
| **Invasive FFR** | 10 | 50 | Calibration |
| **Prognostic outcomes** | 100 | 500+ | Clinical correlation |
| **Patient metadata** | MAP, vessel type | + demographics | BC refinement |

### Key Innovation
**Outer-wall informed reference reconstruction** → accounts for compensatory remodeling (Glagov), providing more physiologically accurate counterfactual healthy lumen than literature methods.

### Open-Source Stack
```
VMTK (geometry) → Python/NumPy (Lyras model) → SciPy (optimization) 
→ Pandas (batch processing) → Matplotlib (visualization)
```

### Expected Outcomes
- **Primary Publication**: *European Heart Journal* or *JACC: Cardiovascular Imaging*
- **Methodological Innovation**: Novel outer-wall constrained reference method
- **Clinical Impact**: Correlation between ΔFFR and prognostic endpoints
- **AUC Target**: >0.90 vs invasive FFR

### When to Choose This
✅ You want high-impact publication with methodological novelty  
✅ You have 50+ cases with invasive FFR for calibration  
✅ You can dedicate 8-10 weeks with 1 programmer  
✅ You value interpretability and full control over the physics

---

## Option 2: Solver-First (future2.md) ⭐ RECOMMENDED FOR LOWER MAINTENANCE

**Best For**: Leveraging established solvers, minimizing custom code

### Core Approach
- SimVascular svOneDSolver as primary engine (validated 1D solver)
- Simpler outer-wall constrained exponential taper for reference
- Dual outputs: ΔFFR_fast (analytic) and ΔFFR_1D (solver)
- Minimal 3D CFD verification (3-5 cases)

### Timeline & Resources
- **6-8 weeks** to results
- **240-300 person-hours**
- **Standard workstation**
- **Processing speed**: 1-3 min/artery (solver mode)

### Data Requirements
| Data Type | Minimum | Optimal | Purpose |
|-----------|---------|---------|---------|
| **STL pairs (lumen + outer)** | 20 | 500+ | Full analysis |
| **Invasive FFR** | 10 | 30 | Calibration |
| **Prognostic outcomes** | 100 | 500+ | Clinical correlation |

### Key Advantage
**Lower maintenance burden** → relies on widely-used, well-maintained svOneDSolver instead of custom physics implementation.

### Open-Source Stack
```
VMTK (geometry) → SimVascular svOneDSolver (1D solver) 
→ Python/NumPy (pre/post-processing) → Pandas (batch)
```

### Expected Outcomes
- **Publication**: High-quality methods paper
- **Cross-Validation**: Dual ΔFFR outputs (fast vs. solver) provide internal consistency check
- **AUC Target**: >0.88 vs invasive FFR

### When to Choose This
✅ You want to minimize custom modeling code  
✅ SimVascular installation is feasible on your systems  
✅ You value having a solver-based cross-check  
✅ You prefer relying on community-maintained tools

---

## Option 3: ML-Driven (future3.md) ⭐ RECOMMENDED IF DATA-RICH

**Best For**: Large datasets with extensive invasive FFR or willingness to generate CFD ground truth

### Core Approach
- XGBoost/LightGBM learns geometry → ΔFFR relationship from data
- 100+ geometric features extracted (stenosis metrics, plaque burden, shape profiles)
- Bypasses explicit physics solvers
- Requires high-quality ground truth (invasive FFR or 3D CFD)

### Timeline & Resources
- **10-14 weeks** (data generation is critical path)
- **350-400 person-hours** + HPC if using CFD for ground truth
- **Requires ML expertise**
- **Inference speed**: <1 sec/artery once trained

### Data Requirements (CRITICAL)
| Data Type | Minimum | Optimal | Purpose |
|-----------|---------|---------|---------|
| **STL pairs (lumen + outer)** | 200 | 500+ | Full dataset |
| **Invasive FFR (GOLD)** | **100** | **200+** | **Training labels** |
| **OR 3D CFD simulations (SILVER)** | **100-200** | **300+** | **Synthetic labels** |
| **Prognostic outcomes** | 200 | 500+ | Secondary analysis |

### Key Advantage
**Learns complex physics** → may outperform 1D models for tortuous vessels, tandem lesions, or complex geometries where simple assumptions fail.

### Open-Source Stack
```
VMTK (geometry) → Python/NumPy (feature engineering) 
→ XGBoost/LightGBM (ML model) → SHAP (interpretability)
→ Optional: OpenFOAM (CFD ground truth generation)
```

### Expected Outcomes
- **Publication**: Novel ML approach to CT-FFR, potentially *Nature Digital Medicine* level if accuracy is high
- **Performance**: Potentially higher AUC (>0.92) than physics models
- **Interpretability**: SHAP values reveal which geometric features drive hemodynamic significance

### When to Choose This
✅ You have N≥100 cases with invasive FFR OR can run 100-200 3D CFD simulations  
✅ You have in-house ML expertise (XGBoost, feature engineering)  
✅ You suspect 1D assumptions may be insufficient for complex anatomies  
✅ You can afford heavy upfront investment for faster batch inference later

---

## Option 4: Minimalist POC (future4.md) ⭐ RECOMMENDED FOR RAPID FEASIBILITY TEST

**Best For**: Quick proof-of-concept before committing to full pipeline

### Core Approach
- Stripped-down Lyras formula with simplest possible geometry processing
- Basic exponential taper reference (no outer wall constraint, no ensemble)
- Process only 10-20 cases to test feasibility
- No advanced validation

### Timeline & Resources
- **2-3 weeks** to initial results
- **60-100 person-hours**
- **Standard laptop**
- **Processing speed**: 10-30 sec/artery

### Data Requirements
| Data Type | Minimum | Optimal | Purpose |
|-----------|---------|---------|---------|
| **STL lumen only** | 10 | 20 | POC test |
| **Invasive FFR (optional)** | 0 | 5 | Sanity check |

### Key Advantage
**Ultra-low risk** → minimal investment to confirm that basic ΔFFR estimation is feasible with your segmentation quality before scaling up.

### Open-Source Stack
```
VMTK (geometry) → Python/NumPy (basic Lyras) → Matplotlib (QC plots)
```

### Expected Outcomes
- **Internal Report**: Feasibility assessment for stakeholders
- **Go/No-Go Decision**: Proceed to Option 1 or 2 if successful

### When to Choose This
✅ You are **uncertain about data quality** or segmentation reliability  
✅ You need **quick results to secure buy-in** for larger investment  
✅ You want to **test the concept** before committing 8-12 weeks  
✅ You have **extremely limited resources** for initial exploration

---

# Data Assets: Optimal Leverage Strategy

## Your Segmented Data (Lumen + Outer Wall)

### What You Have (Unique Asset)
- **Lumen STL**: Inner boundary of blood flow channel
- **Outer Wall STL**: Adventitial boundary (includes plaque)
- **Hundreds of cases** from DISCHARGE trial

### How to Leverage Optimally

#### 1. **Reference Lumen Reconstruction** (Options 1 & 2)
The outer wall enables **physiologically grounded counterfactual modeling**:

```
Healthy Lumen = Outer Wall - Healthy Wall Thickness
```

**Why This Matters**:
- Most CT-FFR methods extrapolate from diseased lumen only
- Your outer wall data captures **Glagov compensatory remodeling**
- Provides more accurate "what the artery should be" estimate
- **This is a publishable methodological innovation**

**Implementation** (from future1.md):
```python
# Measure healthy wall thickness in non-diseased segments
r_lumen_healthy = sqrt(A_lumen[healthy_mask] / π)
r_outer_healthy = sqrt(A_outer[healthy_mask] / π)
T_wall_healthy = median(r_outer_healthy - r_lumen_healthy)

# Apply to entire vessel
r_ref = r_outer - T_wall_healthy
A_ref = π * r_ref²
```

#### 2. **Plaque Burden Quantification** (Option 3)
For ML approach, outer wall enables unique features:

```python
# Features not possible with lumen-only data
plaque_area = A_outer - A_lumen
total_plaque_volume = integrate(plaque_area, ds)
max_plaque_thickness = max(sqrt(plaque_area / π))
remodeling_index = A_outer[stenosis] / A_outer[reference]
```

#### 3. **Quality Control**
Outer wall enables geometric consistency checks:

```python
# QC: Wall thickness should be physiologic
wall_thickness = sqrt(A_outer/π) - sqrt(A_lumen/π)
flag_poor_segmentation = (wall_thickness < 0.3) | (wall_thickness > 2.0)
```

### Data Quality Requirements

| Aspect | Requirement | Impact if Not Met |
|--------|-------------|-------------------|
| **Segmentation coverage** | ≥20mm continuous vessel | Short segments → high uncertainty |
| **Lumen-outer gap** | 0.3-1.5 mm median | Inconsistent → poor reference |
| **Spatial resolution** | <0.5 mm voxel size | Coarse → inaccurate A_min |
| **Artifact-free** | No stents, severe calcification | Artifacts → failed processing |

---

## Your Prognostic Data (DISCHARGE Trial)

### What You Have (Strategic Asset)
- Clinical outcomes (MACE, revascularization, etc.)
- Follow-up duration
- Baseline patient characteristics
- European multi-center cohort

### How to Leverage Optimally

#### 1. **Primary Analysis**: ΔFFR as Prognostic Marker
This is your **unique contribution** beyond technical validation:

**Research Question**: Does CT-derived ΔFFR predict adverse outcomes independent of anatomic stenosis severity?

**Statistical Approach**:
```
Cox Proportional Hazards Model:
  Time-to-MACE ~ ΔFFR + % stenosis + age + risk factors
  
Primary Hypothesis: 
  ΔFFR adds prognostic value beyond anatomic measures
  (likelihood ratio test, p<0.05)
```

**Expected Finding**:
- High ΔFFR (>0.20) → increased MACE risk
- Independent of % stenosis
- **This elevates your paper from technical to clinical impact**

#### 2. **Secondary Analyses**: Subgroup Effects

Explore how ΔFFR performs across patient subgroups:

| Subgroup | Analysis | Clinical Relevance |
|----------|----------|-------------------|
| **Diabetes vs. non-DM** | ΔFFR threshold differs? | Microvascular disease impact |
| **Proximal vs. distal lesions** | ΔFFR accuracy varies? | Flow reserve differences |
| **Single vs. multi-vessel CAD** | ΔFFR additive value? | Cumulative ischemic burden |
| **Age <60 vs. ≥60** | Model performance? | Vessel compliance changes |

#### 3. **Decision-Impact Simulation**

**Question**: How many invasive angiograms could be avoided using CT-FFR?

**Method**:
```
Current Practice: Invasive angio if % stenosis >50%
CT-FFR Strategy: Invasive angio if ΔFFR >0.15

Calculate:
- Avoided procedures (true negatives)
- Missed ischemia (false negatives)
- Net clinical benefit
```

**Impact**: Demonstrates healthcare system value, strengthens manuscript.

#### 4. **Optimal Threshold Determination**

Use prognostic data to define clinical cutoffs:

```python
# Youden's index for optimal threshold
from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(MACE_occurred, deltaFFR)
youden_index = tpr - fpr
optimal_threshold = thresholds[np.argmax(youden_index)]

# Typically: ΔFFR > 0.15-0.20 indicates hemodynamic significance
```

---

# Open-Source Resource Optimization

## Core Toolkit (All Options)

### 1. VMTK (Vascular Modeling Toolkit)
**Purpose**: Geometry extraction (centerlines, cross-sectional areas)

**Installation**:
```bash
conda create -n ctffr python=3.11 vmtk -c vmtk
conda activate ctffr
```

**Key Functions**:
- `vmtkCenterlines`: Extract vessel centerline from STL
- `vmtkCenterlineSections`: Compute cross-sectional areas along centerline
- `vmtkSurfaceDistance`: Outer wall distance calculation

**Performance**: ~5-15 sec per STL

---

### 2. Python Scientific Stack
**Purpose**: Numerical computation, optimization, batch processing

**Installation**:
```bash
pip install numpy scipy pandas matplotlib seaborn scikit-learn
```

**Usage**:
- **NumPy**: Array operations, Lyras formula implementation
- **SciPy**: Optimization (curve fitting for taper), interpolation
- **Pandas**: CSV I/O, batch result tabulation
- **Matplotlib/Seaborn**: Visualization

---

### 3. PyVista
**Purpose**: STL I/O, visualization, mesh operations

**Installation**:
```bash
pip install pyvista
```

**Usage**:
```python
import pyvista as pv

# Read STL
mesh = pv.read('lumen.stl')

# Visualize with cross-sections
mesh.plot(scalars='area', cmap='viridis')
```

---

## Option-Specific Tools

### Option 1 & 4: No Additional Tools
Pure Python implementation, maximum portability.

### Option 2: SimVascular svOneDSolver
**Purpose**: Validated 1D hemodynamic solver

**Installation**:
```bash
# Download from SimVascular.org
# Or build from source (GitHub: SimVascular/svOneDSolver)
```

**Input Format**: `.1d` network file (vessel segments + boundary conditions)

**Output**: Pressure/flow at each node along the network

**Performance**: 30-120 sec per network

---

### Option 3: XGBoost / LightGBM
**Purpose**: Gradient boosting for ML approach

**Installation**:
```bash
pip install xgboost lightgbm shap scikit-learn
```

**Usage**:
```python
import xgboost as xgb

# Train model
model = xgb.XGBRegressor(n_estimators=500, max_depth=6)
model.fit(X_train, y_train)

# Predict
deltaFFR_pred = model.predict(X_test)
```

---

### Optional: OpenFOAM (3D CFD Verification)
**Purpose**: High-fidelity 3D simulations for validation subset (3-15 cases)

**Installation**:
```bash
# Ubuntu/Debian
sudo apt-get install openfoam
# Or from openfoam.org
```

**Usage**: Convert STL → mesh → solve Navier-Stokes → extract pressure drop

**Performance**: 1-4 hours per case on 16-core workstation

**When to Use**: 
- Option 1: 10-15 cases for calibration verification
- Option 2: 3-5 cases for minimal validation
- Option 3: 100-200 cases for ground truth generation (requires HPC cluster)

---

# Decision Framework

## Step 1: Assess Your Constraints

| Constraint | Option 1 | Option 2 | Option 3 | Option 4 |
|------------|----------|----------|----------|----------|
| **Timeline to results** | 8 weeks | 6-8 weeks | 10-14 weeks | 2-3 weeks |
| **Person-hours** | 320 | 240-300 | 350-400 | 60-100 |
| **Invasive FFR cases** | 50 optimal | 30 optimal | **100 required** | 0-5 |
| **ML expertise** | Not needed | Not needed | **Required** | Not needed |
| **HPC access** | No | No | Yes (for CFD) | No |
| **Code maintenance** | Moderate | Low | Low | Minimal |

## Step 2: Match to Your Goals

### Goal A: High-Impact Publication with Methodological Innovation
→ **Option 1** (Physics-Hybrid)
- Outer-wall reference method is novel
- Comprehensive validation
- Target: *European Heart Journal*, *JACC: CI*

### Goal B: Rapid, Reliable Results with Minimal Maintenance
→ **Option 2** (Solver-First)
- Leverages established svOneDSolver
- Dual outputs for robustness
- Target: High-quality methods journal

### Goal C: Maximum Accuracy with Large Labeled Dataset
→ **Option 3** (ML-Driven)
- Best performance potential (AUC >0.92)
- Requires N≥100 FFR cases
- Target: *Nature Digital Medicine*, *JACC*

### Goal D: Quick Feasibility Test Before Full Commitment
→ **Option 4** (Minimalist POC)
- 2-3 week turnaround
- Minimal risk
- Stepping stone to Option 1 or 2

---

## Step 3: Recommended Sequential Approach

### Phase 1: Proof-of-Concept (Weeks 1-3)
**Start with Option 4** regardless of ultimate goal:
1. Process 10-20 cases with basic pipeline
2. Assess segmentation quality and feasibility
3. Identify any fundamental data issues
4. **Go/No-Go Decision**: Proceed if ≥80% success rate

### Phase 2: Full Implementation (Weeks 4-12)
**Choose based on constraints**:
- **If FFR N≥50 + want novelty** → Option 1
- **If FFR N=20-40 + want simplicity** → Option 2
- **If FFR N≥100 + ML expertise** → Option 3

### Phase 3: Prognostic Analysis (Weeks 13-16)
**All options converge here**:
1. Correlate ΔFFR with DISCHARGE outcomes
2. Cox regression for prognostic value
3. Decision-impact simulation
4. Manuscript drafting

---

# Implementation Checklist

## Data Preparation

### Immediate (This Week)
- [ ] **Inventory STL files**: Count lumen/outer pairs, identify missing data
- [ ] **Assess segmentation quality**: Manually inspect 10 representative cases
- [ ] **Catalog invasive FFR data**: Count N, extract values, match to STL files
- [ ] **Prepare prognostic data**: Clean outcomes database, define MACE endpoint
- [ ] **Identify test cases**: Select 3-5 diverse cases (LAD/LCx/RCA, mild/moderate/severe stenosis)

### Week 1
- [ ] **Environment setup**: Install VMTK, Python packages
- [ ] **Test geometry extraction**: Process 3-5 test cases with VMTK
- [ ] **Validate STL quality**: Check for artifacts, resolution, continuity
- [ ] **Match metadata**: Link STL files to patient IDs, vessel types, outcomes

---

## Resource Allocation

### Personnel Requirements

| Role | Option 1 | Option 2 | Option 3 | Option 4 |
|------|----------|----------|----------|----------|
| **Primary Investigator** (MD/PhD) | 100 hrs | 80 hrs | 120 hrs | 30 hrs |
| **Research Programmer** | 160 hrs | 120 hrs | 180 hrs | 40 hrs |
| **Clinical Collaborator** (FFR data) | 40 hrs | 40 hrs | 80 hrs | 10 hrs |
| **ML Specialist** (if Option 3) | - | - | 100 hrs | - |

### Computing Requirements

| Resource | Option 1-2 | Option 3 | Option 4 |
|----------|-----------|----------|----------|
| **Development** | Laptop (8GB RAM) | Laptop | Laptop |
| **Production** | Workstation (16-core) | Workstation + HPC | Laptop |
| **Storage** | 5-10 GB | 20-50 GB (CFD) | 1 GB |

---

# Expected Scientific Outputs

## Primary Manuscript (All Options)

**Title**: *"CT-Derived Fractional Flow Reserve using Outer Wall Segmentation for Prognostic Stratification in Coronary Artery Disease: The DISCHARGE Analysis"*

**Structure**:
1. **Background**: Limitations of anatomic stenosis severity, need for functional assessment
2. **Methods**: 
   - Outer-wall informed reference lumen reconstruction (KEY INNOVATION)
   - Reduced-order hemodynamic modeling (Option 1/2) OR ML approach (Option 3)
   - Validation against invasive FFR
3. **Results**:
   - Technical performance (AUC, RMSE vs FFR)
   - Prognostic analysis (Cox regression, MACE prediction)
   - Decision-impact simulation
4. **Discussion**: Clinical implications, healthcare system impact

**Target Journals**:
- *European Heart Journal* (IF ~35)
- *JACC: Cardiovascular Imaging* (IF ~25)
- *Circulation: Cardiovascular Imaging* (IF ~9)

---

## Secondary Outputs

### Technical Methods Paper
**Focus**: Detailed description of outer-wall constrained reference reconstruction algorithm

**Target**: *IEEE Transactions on Biomedical Engineering* or *Medical Image Analysis*

### Conference Presentations
- **ESC 2026**: Late-breaking abstract
- **ACC 2026**: Featured science
- **SCCT 2026**: Technical innovation

### Open-Source Contribution
**GitHub Repository**: `discharge-ctffr`
- Complete Python package
- Sample data (anonymized)
- Documentation and tutorials
- Citable via Zenodo DOI

---

# Risk Mitigation

## Technical Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| **Poor segmentation quality** | Medium | High | Automated QC flags, manual review protocol |
| **Insufficient FFR data (N<20)** | Low-Medium | Medium | Use 3D CFD for calibration, or proceed with Option 4 POC |
| **Model underperformance (AUC<0.85)** | Low | High | Ensemble reference + UQ provides robustness; pivot to ML if needed |
| **Prognostic data incomplete** | Low | Medium | Power analysis to determine minimum N for meaningful correlation |
| **Computational bottleneck** | Very Low | Low | 60 sec/artery = trivial for batch processing |

## Strategic Risks

| Risk | Mitigation |
|------|------------|
| **Scope creep** | Stick to phased timeline; resist adding features mid-project |
| **Publication delay** | Start manuscript outline at Week 8, not Week 12 |
| **Reviewer skepticism** | Comprehensive validation (FFR + CFD + prognostic) addresses concerns |
| **Reproducibility concerns** | Full code release, detailed methods, open data (if possible) |

---

# Final Recommendation

## Default Path: **Option 1 (Physics-Hybrid)** with **Option 4 (Minimalist POC)** as Week 1-2 Pilot

### Rationale
1. **Option 4 POC (Weeks 1-2)**: Low-risk feasibility test
2. **Decision Point (Week 2)**: Assess segmentation quality, processing success rate
3. **Option 1 Full Implementation (Weeks 3-10)**: If POC successful
4. **Prognostic Analysis (Weeks 11-14)**: Leverage DISCHARGE outcomes
5. **Manuscript (Weeks 15-18)**: High-impact submission

### Why Option 1 as Primary?
✅ **Methodological novelty**: Outer-wall reference is publishable innovation  
✅ **Full control**: Custom Python code, interpretable physics  
✅ **Optimal resource use**: 320 hours over 10 weeks is feasible  
✅ **High impact**: Targets top-tier cardiology journals  
✅ **Leverages your unique asset**: Outer wall segmentation fully exploited

### Fallback Options
- **If segmentation quality issues** → Simplify to Option 2
- **If FFR N≥100 and ML expertise available** → Consider Option 3 for superior accuracy
- **If timeline pressure** → Option 4 alone for rapid internal report

---

# Immediate Action Items

## This Week (5 hours)
1. ✅ Review this document with team
2. ✅ Inventory data: Count STL pairs, FFR cases, prognostic outcomes
3. ✅ Select 3-5 test cases for Week 1 pilot
4. ✅ Assign personnel: PI, programmer, clinical collaborator
5. ✅ Confirm computing resources: workstation access, storage

## Week 1 (40 hours)
1. ✅ Install environment (VMTK, Python)
2. ✅ Process 3-5 test cases with basic geometry extraction
3. ✅ Implement simple Lyras formula (Option 4 POC)
4. ✅ Generate QC visualizations
5. ✅ **Go/No-Go Decision**: Proceed to Option 1 if successful

---

# Conclusion

You have a **unique opportunity** to generate high-impact science from your DISCHARGE data:

1. **Lumen + outer wall segmentations** → novel reference reconstruction method
2. **Prognostic outcomes** → demonstrate clinical value beyond technical validation
3. **Large European cohort** → sufficient power for robust conclusions
4. **100% open-source** → reproducible, globally disseminable

**The limiting factor is not the method—it's starting execution.**

**Recommended first step**: Approve Week 1 pilot (Option 4 POC, 40 hours) to test feasibility before committing to full Option 1 pipeline.

---

## Decision Required

**Approve Week 1 pilot?** [ ] YES  [ ] NO  [ ] NEED MORE INFO

**If YES, provide**:
1. Name of assigned programmer
2. Count of available invasive FFR cases (N=?)
3. Paths to 3-5 representative STL pairs for testing
4. Expected start date

**Questions?** Refer to future1-4.md for detailed technical specifications.
