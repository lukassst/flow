# Alternative Plan: Minimalist Proof-of-Concept ΔFFR Pipeline

**Date**: November 5, 2025  
**Purpose**: Provide a fourth, ultra-lean pathway to rapidly deploy a functional ΔFFR estimation pipeline with minimal resource investment, prioritizing speed to initial results for a proof-of-concept.

---

## Executive Summary

- **Core Idea**: Implement a stripped-down version of the Lyras reduced-order model with the simplest possible geometry processing and reference lumen estimation.
- **Rapid Deployment**: Focus on getting initial results within 2-3 weeks for a small subset of cases to demonstrate feasibility.
- **Minimalist Approach**: Use only essential open-source tools, avoid complex ensemble methods or solver integrations, and limit validation to a tiny subset.
- **Scalability Trade-off**: Designed as a stepping stone—can be expanded later with more sophisticated methods if the concept proves viable.
- **Outcome**: Quick feedback on whether ΔFFR estimation is feasible with your data, enabling informed decisions on further investment.

---

## What’s Different vs future1.md, future2.md & future3.md

- **Ultra-Lean Scope**: Targets only a small initial batch (10-20 cases) to prove the concept before committing to full trial analysis.
- **Simplest Reference Lumen**: Uses a basic exponential taper without outer wall constraints or ensemble weighting.
- **No Advanced Solvers or ML**: Relies solely on a basic Python implementation of the Lyras formula, avoiding external solver dependencies (like svOneDSolver) or data-heavy ML approaches.
- **Minimal Validation**: Limits calibration/validation to a tiny subset (5-10 cases with invasive FFR if available, or none if not).
- **Shortest Timeline**: Aims for initial results in 2-3 weeks with under 100 person-hours.

---

## Tooling (100% Open-Source, Minimal Set)

- Geometry: VMTK (basic centerline and area extraction only)
- Computation: Python, NumPy (for Lyras formula implementation)
- Visualization: Matplotlib (basic plots for QC)

---

## Pipeline Overview

```
STL_lumen (outer wall optional)
      ↓  (VMTK)
Centerline s + A_lumen(s)
      ↓  (Basic Reference Lumen)
Simple Exponential Taper → A_ref(s)
      ↓  (Lesion Metrics)
A_min, L_lesion
      ↓  (Lyras Formula)
ΔP → ΔFFR
      ↓  (Minimal QC)
Basic QC Flags & Visuals for 10-20 cases
```

---

## Phase 1: Minimal Geometry Processing

1. **Input**: Use only lumen STL (outer wall optional, ignored for simplicity).
2. **Centerline Extraction**: Use VMTK to extract centerline and cross-sectional areas (A_lumen(s)).
3. **Basic Reference Lumen**: Identify proximal healthy segment (first 10% of length), assume constant reference area (A_ref = mean of proximal segment), or apply a simple exponential taper with fixed decay rate (λ=0.015).
4. **Lesion Metrics**: Compute A_min (minimal lumen area) and L_lesion (length where A_lumen < 0.7 * A_ref).

```python
import vmtk.vmtkscripts as vmtk
import pyvista as pv
import numpy as np

def extract_basic_geometry(stl_lumen):
    """Minimal VMTK centerline and area extraction"""
    surface_lumen = pv.read(stl_lumen)
    cl_computer = vmtk.vmtkCenterlines()
    cl_computer.Surface = surface_lumen
    cl_computer.Execute()
    
    centerline_points = np.array(cl_computer.Centerlines.GetPoints())
    n_points = centerline_points.shape[0]
    
    s = np.zeros(n_points)
    for i in range(1, n_points):
        s[i] = s[i-1] + np.linalg.norm(centerline_points[i] - centerline_points[i-1])
    
    radii_lumen = np.array([cl_computer.Centerlines.GetPointData()
                            .GetArray('Radius').GetValue(i) for i in range(n_points)])
    A_lumen = np.pi * radii_lumen**2
    
    # Simple reference: mean of first 10% points as proxy
    ref_idx = int(0.1 * n_points)
    A_ref = np.mean(A_lumen[:ref_idx])
    
    return s, A_lumen, A_ref
```

---

## Phase 2: Basic Pressure Drop Calculation

- Implement the Lyras reduced-order formula with default parameters (no calibration).
- Use fixed boundary conditions: P_a = 93 mmHg, Q_hyperemic based on vessel type defaults (LAD: 1.2 * 3.5, LCx: 0.8 * 3.5, RCA: 1.0 * 3.5 mL/s).
- Output ΔP and ΔFFR = ΔP / P_a.

```python
def basic_pressure_drop(A_min, L_lesion, vessel_type='LAD'):
    """Ultra-simple Lyras model implementation"""
    mu = 0.004  # Pa·s
    rho = 1050  # kg/m³
    K1 = 16.0   # Default, no calibration
    K2 = 1.25   # Default, no calibration
    
    Q_defaults = {'LAD': 1.2, 'LCx': 0.8, 'RCA': 1.0}  # mL/s baseline
    Q_baseline = Q_defaults.get(vessel_type, 1.0)
    Q_hyperemic = Q_baseline * 3.5 * 1e-6  # m³/s
    
    A_min_m2 = A_min * 1e-6
    L_m = L_lesion * 1e-3 if L_lesion > 0 else 0.001
    
    viscous = K1 * mu * (L_m / A_min_m2**2) * Q_hyperemic
    inertial = K2 * rho * (Q_hyperemic / A_min_m2)**2
    
    delta_P = (viscous + inertial) / 133.322  # Pa to mmHg
    P_a = 93  # Fixed aortic pressure
    delta_FFR = delta_P / P_a
    
    return delta_P, delta_FFR
```

---

## Phase 3: Minimal QC and Visualization

- **Basic QC**: Check only for processing errors (e.g., failed centerline extraction) and extreme values (ΔP > 50 mmHg or < 0).
- **Simple Plots**: Generate basic area vs. length plots to visually inspect A_lumen and A_ref for sanity.
- No uncertainty quantification or advanced flagging.

---

## Phase 4: Tiny Validation (if possible)

- If invasive FFR data is available for even 5-10 cases, compare ΔFFR against it for a rough sanity check (no formal calibration).
- If no FFR data, rely on literature benchmarks (e.g., ΔP > 15 mmHg often correlates with FFR < 0.80).
- No 3D CFD verification in this minimalist plan.

---

## Timeline (2-3 Weeks)

- **Week 1**: Install minimal tools (VMTK, Python, NumPy), extract geometry for 5 test cases, implement basic Lyras formula.
- **Week 2**: Process 10-20 cases, generate basic QC plots, compare to any available FFR data.
- **Week 3 (optional)**: Draft a short internal report or abstract summarizing feasibility findings and next steps.

---

## Resources

- **Personnel**: ~60-100 person-hours total (1-2 weeks for a single developer).
- **Compute**: Standard laptop (no HPC or workstation needed).
- **Cost**: €0 software.

---

## Go/No-Go Checks

- **After Week 2**: Proceed if ≥80% of 10-20 cases process without errors and ΔFFR values are in a plausible range (0-0.5). Halt if geometry extraction fails systematically or results are nonsensical.

---

## Outputs

- Per case (simple CSV): id, ΔP (mmHg), ΔFFR, A_min (mm²), L_lesion (mm), QC_pass (yes/no).
- Study-level: Basic summary of ΔFFR distribution for the 10-20 cases.

---

## When to Prefer This Plan

- You want the **fastest possible feedback** on whether ΔFFR estimation is feasible with your data.
- You have **extremely limited resources** (time, personnel, or compute) for initial exploration.
- You are **unsure about committing to a full pipeline** and want a low-risk test run.
- You plan to use this as a **stepping stone** to a more sophisticated approach (like future1.md or future2.md) if the concept proves viable.

---

## Decision

Proceed with this minimalist plan if speed to initial results and minimal resource investment are your top priorities. This can serve as a low-risk test before scaling up to a more comprehensive pipeline like future1.md or future2.md.
