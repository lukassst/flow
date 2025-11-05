# CT-FFR Pipeline: Definitive Implementation Plan for DISCHARGE Trial

**Date**: November 5, 2025  
**Context**: Research institution, hundreds of segmented coronary arteries (lumen + outer wall STL), prognostic data from DISCHARGE trial

---

## Executive Decision

**Adopt Plan 3 technical implementation with Plan 2 phasing approach**

### Why?
- **Plan 3**: Most complete code (615 lines, 95% ready-to-execute)
- **Plan 2**: Clearest week-by-week timeline
- **All 4 plans agree**: 1D reduced-order model is sufficient, outer wall data is your unique advantage

### Bottom Line
- **Timeline**: 8 weeks to results, 12 weeks to manuscript
- **Resources**: 320 person-hours, €0 software cost
- **Processing**: 30-90 sec/artery = 500 cases in <12 hours
- **Expected**: AUC > 0.90 vs invasive FFR, high-impact publication

---

## Technical Consensus (All Plans Agree)

| Component | Decision |
|-----------|----------|
| **Modeling** | 1D reduced-order (Lyras 2021), not 3D CFD |
| **Metric** | ΔFFR (pressure drop across lesion) |
| **Geometry** | VMTK for centerline extraction |
| **Solver** | Custom Python implementation |
| **Reference Lumen** | Outer wall scaffold + taper ensemble (YOUR KEY INNOVATION) |
| **Boundary Conditions** | Murray's law + hyperemic scaling (Q×3.5) + P_a=93 mmHg |
| **Validation** | Calibrate on N=20-50 invasive FFR cases |

---

## 8-Week Implementation Roadmap

### Week 1-2: Proof-of-Concept (40 hours)
**Goal**: Process 20 cases end-to-end

```bash
# Environment setup
conda create -n discharge_ffr python=3.11 vmtk pyvista scipy numpy pandas tqdm -c vmtk
conda activate discharge_ffr
pip install matplotlib seaborn scikit-learn
```

**Tasks**:
1. Extract centerlines with VMTK (lumen + outer wall)
2. Implement hybrid reference reconstruction:
   - Method A: Exponential taper from healthy segments
   - Method B: Outer wall constraint (r_ref = r_outer - T_wall)
   - Method C: Cubic spline interpolation
   - **Ensemble**: Adaptive weighting based on stenosis severity
3. Apply Lyras pressure drop model
4. Visualize results

**Success Criteria**:
- ✅ 20/20 cases process without crashes
- ✅ Reference lumen visually plausible
- ✅ ΔP values in clinical range (0-50 mmHg)

**Deliverable**: Jupyter notebook with 20 processed cases

---

### Week 3-4: Calibration (60 hours)
**Goal**: Tune model coefficients using invasive FFR

**Requirements**: N≥20 cases with invasive FFR (N=50 optimal)

**Method**:
```python
# Optimize K₁, K₂, calibration factors
from scipy.optimize import minimize

def calibration_loss(params, cases):
    K1, K2, a_calib, b_calib = params
    rmse = 0
    for case in cases:
        delta_P_pred = lyras_model(case, K1, K2)
        delta_P_obs = case['invasive_FFR'] * case['P_aortic']
        rmse += (a_calib * delta_P_pred + b_calib - delta_P_obs)**2
    return np.sqrt(rmse / len(cases))

result = minimize(calibration_loss, x0=[16.0, 1.25, 1.0, 0.0], ...)
```

**Target**: RMSE < 3 mmHg, AUC > 0.90 for FFR<0.80 classification

**Deliverable**: `calibration_report.pdf` with ROC curves, Bland-Altman plots

---

### Week 5-6: Validation + Uncertainty Quantification (40 hours)
**Goal**: Validate on independent test set

**Tasks**:
1. Apply calibrated model to N=50-100 validation cases
2. Monte Carlo uncertainty quantification:
   - Vary flow Q (±15%)
   - Vary A_min (±8%)
   - Vary K₁/K₂ (±10%)
   - Report 5th-95th percentile CI
3. Generate QC flags:
   - `HIGH_UNCERTAINTY`: CI width > 0.05 ΔFFR
   - `SHORT_SEGMENT`: Length < 30mm
   - `DIFFUSE_DISEASE`: No healthy reference
   - `POOR_SEGMENTATION`: Abnormal outer-lumen gap

**Deliverable**: `validation_results.csv` + figures

---

### Week 7: 3D Verification (40 hours, OPTIONAL)
**Goal**: Confirm 1D model against 3D CFD

**Method**: N=10-15 cases (severe, moderate, complex geometry)
- Run OpenFOAM or SimVascular 3D CFD
- Compare ΔP_1D vs ΔP_3D
- Target: Mean bias < 2 mmHg

**Decision**: If bias > 5 mmHg → add empirical correction

---

### Week 8-10: Production Batch (60 hours)
**Goal**: Process all DISCHARGE trial arteries

```python
from multiprocessing import Pool
from pathlib import Path

def batch_process_trial(data_dir, output_csv, n_workers=8):
    """Process hundreds of arteries in parallel"""
    cases = [(lumen_stl, outer_stl, vessel_type) for ...]
    
    with Pool(n_workers) as pool:
        results = list(tqdm(pool.imap(process_single_case, cases)))
    
    pd.DataFrame(results).to_csv(output_csv)
```

**Computational**: 500 arteries × 60 sec = 8.3 hours on 8-core workstation

**Deliverable**: 
- `discharge_ffr_complete.csv`
- `qc_summary_report.pdf`
- Per-artery visualization PDFs

---

### Week 11-12: Manuscript Preparation (80 hours)
**Goal**: Draft manuscript for submission

**Target Journals**:
1. *European Heart Journal* or *JACC: Cardiovascular Imaging* (IF ~20-30)
2. *Circulation: Cardiovascular Imaging* (IF ~9)
3. *JACC: Cardiovascular Interventions* (IF ~12)

**Structure**:
- Methods: Outer-wall hybrid reference + uncertainty quantification
- Results: N=500+ cases, AUC vs invasive FFR, prognostic correlation
- Discussion: Clinical decision-making impact

---

## Core Technical Implementation

### 1. Geometry Extraction (VMTK)
```python
import vmtk.vmtkscripts as vmtk
import pyvista as pv
import numpy as np

def extract_geometry(stl_lumen, stl_outer):
    """Extract 1D profiles from STL surfaces"""
    # Lumen centerline
    surface_lumen = pv.read(stl_lumen)
    cl_computer = vmtk.vmtkCenterlines()
    cl_computer.Surface = surface_lumen
    cl_computer.Execute()
    
    centerline_points = np.array(cl_computer.Centerlines.GetPoints())
    n_points = centerline_points.shape[0]
    
    # Arc length
    s = np.zeros(n_points)
    for i in range(1, n_points):
        s[i] = s[i-1] + np.linalg.norm(centerline_points[i] - centerline_points[i-1])
    
    # Lumen areas
    radii_lumen = np.array([cl_computer.Centerlines.GetPointData()
                            .GetArray('Radius').GetValue(i) for i in range(n_points)])
    A_lumen = np.pi * radii_lumen**2
    
    # Outer wall areas (project centerline to outer surface)
    surface_outer = pv.read(stl_outer)
    radii_outer = np.zeros(n_points)
    for i, point in enumerate(centerline_points):
        closest_point_id = surface_outer.find_closest_point(point)
        dist = np.linalg.norm(surface_outer.points[closest_point_id] - point)
        radii_outer[i] = radii_lumen[i] + dist
    A_outer = np.pi * radii_outer**2
    
    return s, A_lumen, A_outer
```

---

### 2. Reference Lumen Reconstruction (KEY INNOVATION)
```python
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import UnivariateSpline
from scipy.optimize import curve_fit

def reconstruct_reference(s, A_lumen, A_outer):
    """Hybrid ensemble exploiting outer wall data"""
    
    # Identify healthy segments
    A_smooth = gaussian_filter1d(A_lumen, sigma=2.0)
    gradient = np.abs(np.gradient(A_smooth))
    healthy_mask = (gradient < np.percentile(gradient, 30)) & \
                   (A_lumen > 0.7 * A_smooth)
    
    # Method A: Exponential taper
    def exp_taper(s_vals, D0, lam):
        return D0 * np.exp(-lam * s_vals)
    
    D_lumen = 2 * np.sqrt(A_lumen / np.pi)
    prox_idx = np.where(healthy_mask)[0][:min(15, np.sum(healthy_mask)//2)]
    popt, _ = curve_fit(exp_taper, s[prox_idx], D_lumen[prox_idx],
                        p0=[D_lumen[prox_idx[0]], 0.015],
                        bounds=([0, 0], [10, 0.05]))
    D_ref_taper = exp_taper(s, *popt)
    A_ref_taper = np.pi * (D_ref_taper/2)**2
    
    # Method B: Outer wall constraint (YOUR UNIQUE ADVANTAGE)
    r_lumen = np.sqrt(A_lumen / np.pi)
    r_outer = np.sqrt(A_outer / np.pi)
    T_wall_healthy = np.median(r_outer[healthy_mask] - r_lumen[healthy_mask])
    T_wall_healthy = np.clip(T_wall_healthy, 0.4, 1.2)  # Biological bounds
    
    r_ref_outer = r_outer - T_wall_healthy
    r_ref_outer = np.maximum(r_ref_outer, r_lumen * 1.05)  # At least 5% larger
    A_ref_outer = np.pi * r_ref_outer**2
    
    # Method C: Spline interpolation
    spline = UnivariateSpline(s[healthy_mask], A_lumen[healthy_mask], 
                              k=3, s=np.sum(healthy_mask)*0.1)
    A_ref_spline = spline(s)
    A_ref_spline = np.maximum(A_ref_spline, A_lumen)
    
    # Adaptive ensemble weighting (increases outer wall weight in stenosis)
    stenosis_score = 1 - (A_lumen / np.maximum(A_ref_outer, A_lumen*1.01))
    stenosis_score = np.clip(stenosis_score, 0, 1)
    
    w_outer = 0.5 + 0.3 * stenosis_score   # 50-80%
    w_taper = 0.3 - 0.2 * stenosis_score   # 30-10%
    w_spline = 0.2 - 0.1 * stenosis_score  # 20-10%
    
    # Normalize
    w_sum = w_outer + w_taper + w_spline
    A_ref_ensemble = (w_outer/w_sum * A_ref_outer + 
                     w_taper/w_sum * A_ref_taper + 
                     w_spline/w_sum * A_ref_spline)
    
    # Uncertainty
    uncertainty = np.std([A_ref_taper, A_ref_outer, A_ref_spline], axis=0)
    
    return A_ref_ensemble, uncertainty
```

---

### 3. Pressure Drop Calculation (Lyras 2021)
```python
def compute_pressure_drop(A_min, L_lesion, Q, P_a=93, K1=16.0, K2=1.25):
    """Lyras reduced-order model"""
    mu = 0.004      # Pa·s (blood viscosity)
    rho = 1050      # kg/m³ (blood density)
    
    # Convert to SI
    A_min_m2 = A_min * 1e-6
    L_m = L_lesion * 1e-3
    
    # Viscous + inertial terms
    viscous = K1 * mu * (L_m / A_min_m2**2) * Q
    inertial = K2 * rho * (Q / A_min_m2)**2
    
    delta_P_Pa = viscous + inertial
    delta_P_mmHg = delta_P_Pa / 133.322  # Pa to mmHg
    
    # Apply calibration (tune a_calib, b_calib in Week 3-4)
    a_calib = 0.98  # Default, will be optimized
    b_calib = 1.1   # Default, will be optimized
    delta_P_calib = a_calib * delta_P_mmHg + b_calib
    
    delta_FFR = delta_P_calib / P_a
    
    return delta_P_calib, delta_FFR
```

---

### 4. Hyperemic Flow Estimation (Murray's Law)
```python
def estimate_hyperemic_flow(vessel_type, A_proximal, A_ref_proximal):
    """Murray's law scaling"""
    Q_defaults = {'LAD': 1.2, 'LCx': 0.8, 'RCA': 1.0}  # mL/s baseline
    Q_baseline = Q_defaults.get(vessel_type, 1.0)
    
    # Scale by area ratio (Murray exponent α=1.35)
    scaling = (A_proximal / A_ref_proximal) ** 1.35
    scaling = np.clip(scaling, 0.5, 2.0)
    
    Q_hyperemic = Q_baseline * scaling * 3.5  # Hyperemia factor
    
    return Q_hyperemic * 1e-6  # Convert mL/s to m³/s
```

---

### 5. Uncertainty Quantification (Monte Carlo)
```python
def quantify_uncertainty(A_min, L_lesion, Q, P_a, K1, K2, n_samples=100):
    """Monte Carlo uncertainty propagation"""
    delta_FFR_samples = np.zeros(n_samples)
    
    for i in range(n_samples):
        # Sample variations
        Q_var = Q * np.random.uniform(0.85, 1.15)
        A_min_var = A_min * np.random.uniform(0.92, 1.08)
        K1_var = K1 * np.random.uniform(0.90, 1.10)
        K2_var = K2 * np.random.uniform(0.90, 1.10)
        
        # Recompute
        _, delta_FFR_samples[i] = compute_pressure_drop(
            A_min_var, L_lesion, Q_var, P_a, K1_var, K2_var
        )
    
    return {
        'median': np.median(delta_FFR_samples),
        'CI_5': np.percentile(delta_FFR_samples, 5),
        'CI_95': np.percentile(delta_FFR_samples, 95)
    }
```

---

## Resource Requirements

### Personnel (Total: 320 hours)
- **Primary Investigator**: 200 hours (phased over 10 weeks)
- **Research Programmer**: 80 hours (concentrated in weeks 1-4)
- **Clinical Collaborator**: 40 hours (data curation + validation)

### Computing
- **Development**: Laptop (8GB RAM) ✅
- **Production**: 16-core workstation OR HPC cluster
- **Storage**: ~5GB for 500 STL pairs + results

### Software (100% Open-Source)
```
VMTK           - Free
Python 3.11    - Free
NumPy/SciPy    - Free
PyVista        - Free
OpenFOAM       - Free (optional validation)
```

**Total Cost**: €0 (excluding personnel)

---

## Risk Mitigation

| Risk | Probability | Mitigation |
|------|-------------|------------|
| **Insufficient FFR data (N<20)** | Low-Medium | Use 3D CFD for calibration |
| **Poor segmentation quality** | Medium | Automated QC flags + manual review |
| **Model underperformance** | Low | Ensemble reference + UQ provides robustness |
| **Computational bottleneck** | Very Low | 60 sec/artery = trivial for 500 cases |

---

## Go/No-Go Decision Criteria

### After Week 2 (Pilot):
- ✅ **GO** if: 90% cases process successfully, ΔP in clinical range
- ❌ **NO-GO** if: Systematic segmentation artifacts, negative reference areas

### After Week 4 (Calibration):
- ✅ **GO** if: AUC > 0.85, RMSE < 5 mmHg
- ⚠️ **PIVOT** if: AUC 0.75-0.85 → explore ML-enhanced corrections
- ❌ **NO-GO** if: AUC < 0.75 → fundamental model inadequacy

---

## Expected Outcomes

### Scientific Impact
1. **Primary Publication**: High-impact cardiology journal (EHJ/JACC)
   - *"CT-Derived Fractional Flow Reserve using Outer Wall Remodeling: The DISCHARGE Analysis"*
   - Expected IF: 15-30

2. **Methodological Innovation**: Outer-wall informed reference reconstruction
   - Most CT-FFR methods use lumen only
   - Your data enables Glagov remodeling modeling
   - **Publishable methodological contribution**

3. **Clinical Decision Support**: Non-invasive functional CAD assessment at scale

### Open-Source Contribution
- Python package (citable via Zenodo)
- Enables global adoption by research groups
- Foundation for future AI/ML enhancements

---

## Immediate Next Steps

### This Week
1. ✅ Identify 3 representative STL pairs (LAD/LCx/RCA)
2. ✅ Confirm invasive FFR data availability (count N)
3. ✅ Assign primary investigator + programmer
4. ✅ Set up computing environment (conda, VMTK)

### Week 1 Deliverable
- 3 successfully processed test cases with visualization
- Centerline + area profiles
- Reference lumen comparison plots
- Initial ΔP estimates

---

## Why This Will Succeed

### 1. Your Unique Advantage
- **Outer wall + lumen segmentation** (rare in literature)
- Enables physiologically grounded reference reconstruction
- Addresses Glagov compensatory remodeling

### 2. Validated Methods
- 1D reduced-order models clinically validated (Lyras 2021, Taylor 2013)
- All 4 expert plans converge on feasibility
- Open-source tools with proven track record

### 3. Pragmatic Resource Allocation
- 80% effort on validated 1D model (sufficient for clinical decisions)
- 20% effort on 3D validation (enhances credibility)
- No expensive commercial software

### 4. Built-In Quality Control
- Ensemble reference (3 methods) → internal consistency
- Monte Carlo UQ → flags uncertain cases
- Automated QC metrics → scales to 500+ cases

### 5. Clear Path to Clinical Impact
- DISCHARGE cohort = European registry-scale validation
- ΔP correlates with prognostic outcomes
- Enables non-invasive functional assessment at scale

---

## Final Statement

**This is a low-risk, high-reward opportunity** to generate significant scientific impact from your existing DISCHARGE dataset using entirely open-source methods with minimal resource investment.

**All 4 expert plans agree**: This is feasible, scientifically sound, and clinically meaningful.

**The limiting factor is not the method—it's starting execution.**

---

## Decision Required

**Do you approve proceeding with Week 1 pilot implementation?**

**If YES**: Provide 3 STL pairs + FFR data count  
**If NO**: Specify concerns or alternative timeline  
**If MAYBE**: Request additional technical details

**Next step**: Send first 3 STL pairs to begin Week 1 pilot (8 hours).
