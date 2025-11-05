# Alternative Plan: Machine Learning–Driven ΔFFR Estimation

**Date**: November 5, 2025  
**Purpose**: Provide a third, data-driven pathway using machine learning to directly predict ΔFFR from geometric features, bypassing explicit physics solvers.

---

## Executive Summary

- **Core Idea**: Train a machine learning model (e.g., Gradient Boosting) to learn the complex relationship between arterial geometry and pressure drop.
- **Bypasses Physics**: Avoids explicit 1D solvers (Lyras, svOneDSolver) and manual boundary condition assumptions.
- **Heavy Upfront Cost**: Requires a high-quality "ground truth" dataset (N≥100) with known outcomes, either from **invasive FFR** (gold standard) or **high-fidelity 3D CFD simulations** (silver standard).
- **Instant Inference**: Once trained, the model can predict ΔFFR for hundreds of cases in minutes.
- **Outcome**: A potentially more accurate model for complex anatomies where 1D assumptions may fail, but with higher upfront investment and less physical interpretability.

---

## What’s Different vs future1.md & future2.md

- **No Physics Solver**: The model *learns* the flow physics implicitly from data rather than solving explicit equations.
- **No Explicit Reference Lumen**: The complex reference reconstruction is replaced by feeding geometric ratios and plaque metrics directly to the model as features.
- **Data-Hungry**: Success is highly dependent on the size and quality of the training dataset.
- **Interpretability**: Relies on post-hoc methods (e.g., SHAP) to explain predictions, unlike physics models where terms (viscous, inertial) are explicit.

---

## Tooling

- Geometry/Feature Extraction: VMTK, PyVista, NumPy/SciPy
- Machine Learning: Scikit-learn, XGBoost/LightGBM
- Ground Truth Generation: OpenFOAM/SimVascular (if using CFD), or clinical data records (for invasive FFR)
- Visualization: Matplotlib, Seaborn

---

## Pipeline Overview

```
STL_lumen + STL_outer
      ↓  (VMTK/Python)
Feature Vector Extraction (100+ geometric features)
      ↓  (Ground Truth Generation - N≥100 cases)
Train XGBoost Model: Features → ΔFFR_invasive or ΔFFR_CFD
      ↓  (Validation)
Hold-out test set evaluation (MAE, RMSE, AUC)
      ↓  (Deployment)
Predict ΔFFR for all remaining cases
```

---

## Phase 1: Geometric Feature Engineering

This is the most critical phase. The goal is to create a rich, numerical representation of each artery.

**Feature Categories**:
1.  **Global Vessel Metrics**: Total length, average radius, average curvature, tortuosity, taper rate.
2.  **Stenosis Metrics**: Minimal Lumen Area (MLA), lesion length, % area stenosis, % diameter stenosis.
3.  **Plaque Metrics (using outer wall)**: Max/mean plaque thickness, total plaque volume (`V_outer - V_lumen`), plaque burden index.
4.  **Shape & Profile Metrics**: Skewness/kurtosis of the area profile, number of local minima, distance from ostium to MLA.
5.  **Hemodynamic Proxies**: Ratios like `(A_proximal / A_min)^2`, curvature at MLA, inlet/outlet diameter ratio.

```python
# Example feature extraction
def extract_features(s, A_lumen, A_outer):
    features = {}
    # Global
    features['length'] = s[-1]
    features['tortuosity'] = ...
    # Stenosis
    i_min = np.argmin(A_lumen)
    A_min = A_lumen[i_min]
    A_ref = np.mean(A_lumen[:5]) # Simplified reference
    features['MLA'] = A_min
    features['percent_stenosis'] = 100 * (1 - A_min / A_ref)
    # Plaque
    plaque_area = A_outer - A_lumen
    features['max_plaque_thickness'] = np.sqrt(np.max(plaque_area)/np.pi)
    features['total_plaque_volume'] = np.trapz(plaque_area, s)
    return features
```

---

## Phase 2: Ground Truth Generation (The Bottleneck)

Choose one path. This requires significant upfront effort.

- **Path A (Gold Standard)**: Curate all cases with available **invasive FFR** data. This provides the most clinically relevant target.
  - **Required**: N ≥ 100 for a robust model. N=50 is a bare minimum for a proof-of-concept.
- **Path B (Silver Standard)**: If FFR data is sparse, generate a synthetic dataset using **high-fidelity 3D CFD**.
  - **Process**: For N=100-200 cases, run full 3D simulations (e.g., OpenFOAM) with standardized boundary conditions to get a simulated ΔFFR.
  - **Cost**: Computationally very expensive (many core-hours per case).

---

## Phase 3: Model Training

- **Model**: **XGBoost (Gradient Boosted Trees)** is the recommended starting point. It is highly effective on tabular data, robust to feature scaling, and has good performance.
- **Training Process**:
  1.  Split the ground truth dataset (e.g., 80% train, 20% test).
  2.  Train the XGBoost regressor to predict `ΔFFR` from the feature vector.
  3.  Use cross-validation to tune hyperparameters (e.g., learning rate, tree depth).

```python
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# X is the feature matrix, y is the ground truth ΔFFR
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=500)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
print(f"Validation MAE: {mae:.2f} mmHg")
```

---

## Phase 4: Validation & Deployment

- **Validation**: On the held-out test set, evaluate:
  - **Accuracy**: Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) in mmHg.
  - **Classification Performance**: Area Under the Curve (AUC) for identifying hemodynamically significant stenosis (e.g., invasive FFR < 0.80).
- **Interpretability**: Use SHAP (SHapley Additive exPlanations) to understand which geometric features are driving the model's predictions.
- **Deployment**: Once the model is validated, use the `model.predict()` method on the feature vectors of the remaining hundreds of arteries. This step is extremely fast.

---

## Timeline (10–14 Weeks)

- **Weeks 1–4**: Feature engineering pipeline development and refinement.
- **Weeks 1–8 (Parallel)**: **Ground truth generation**. This is the critical path. If using CFD, this could take 8+ weeks on an HPC cluster.
- **Week 9**: Model training, hyperparameter tuning, and cross-validation.
- **Week 10**: Validation on test set, SHAP analysis for interpretability.
- **Week 11**: Batch processing of the full trial dataset.
- **Weeks 12–14**: Manuscript drafting.

---

## Resources

- **Personnel**: Requires a team member with strong ML experience (Scikit-learn, XGBoost, feature engineering). Total effort ~350-400 hours.
- **Compute**: If using CFD for ground truth, an HPC cluster is mandatory. Model training itself can be done on a standard workstation.
- **Cost**: €0 software, but potentially significant computational cost if using cloud HPC for CFD.

---

## When to Prefer This Plan

- You have a **large dataset (N≥100) with high-quality outcome labels** (invasive FFR).
- You suspect that **1D physics assumptions are insufficient** for a significant portion of your cases (e.g., very tortuous vessels, tandem lesions, complex bifurcations).
- You have **in-house machine learning expertise**.
- **Inference speed** for the final batch processing is the highest priority, and you can afford a heavy upfront investment in creating the training data.
