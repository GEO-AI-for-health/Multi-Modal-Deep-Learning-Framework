# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pandas as pd
import cv2
import shap
import warnings
import pickle
import traceback
import matplotlib.colors as mcolors
import matplotlib.cm as _cm
from matplotlib.patches import Patch
from matplotlib.colors import LinearSegmentedColormap
from torch.utils.data import Subset
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, classification_report,
    precision_recall_fscore_support
)
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy.ndimage import uniform_filter1d
from scipy.spatial import cKDTree
from pyproj import Transformer as ProjTransformer

# Optional dependencies (OSM basemap)
try:
    import contextily as ctx
    import geopandas as gpd
    from shapely.geometry import Point
    HAS_CONTEXTILY = True
except ImportError:
    HAS_CONTEXTILY = False

# Optional dependencies (GAM)
try:
    from pygam import LinearGAM, s as gam_s
    HAS_PYGAM = True
except ImportError:
    HAS_PYGAM = False

# Suppress warnings
warnings.filterwarnings("ignore")

# ================= 1. Basic Configuration =================
DATA_DIR = "./processed_data_stress_new"
MODEL_PATH = "best_38_clip_mobilenet_gru_binary_l2.pth"
OUTPUT_DIR = "final_analysis_v11_full_features"  # Modified output directory
BATCH_SIZE = 32
SAT_DIR = r"./satellite_images"                  # New: Satellite image directory
SHAP_SAMPLES = 800       # GradientExplainer is slower, 50-100 is moderate
PATCH_SHAP_SAMPLES = 500 # Number of sequences randomly sampled for Patch SHAP (using T=0 timestep)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEQ_LEN = 30  # Sequence length

# ================= Timestep Selection Configuration =================
SPECIFIC_TIME_STEP = 0  # 0 uses the first timestep, None uses all timesteps flattened

# Try to import from train.py
try:
    from trian_delesomevaribles import (
        FullStackPredictor, NPZDataset, custom_collate,
        FEATURE_CONFIG, ADAPTER_TYPE, TEMPORAL_MODEL,
        ABLATION_CONFIG, ADAPTER_OUTPUT_DIM, ADAPTER_PATCH_DIM,
        ADAPTER_CLS_DIM, USE_CLS_IN_ADAPTER, CROSS_MODAL_FUSION,
        TABULAR_OUTPUT_DIM, TABULAR_PROCESSING, CENTER_AWARE,
        CENTER_WEIGHT_INIT, NUM_CLASSES
    )

    print("✅ Successfully imported configurations from train.py.")
except ImportError:
    print("❌ Error: train.py not found.")
    sys.exit(1)

# Feature columns
SVI_COLS = [
    'svi_road', 'svi_sidewalk', 'svi_building', 'svi_sky',
    'svi_greenery', 'svi_person', 'svi_rider', 'svi_vehicle'
]

ENV_COLS = ['noise', 'pm2.5', 'temperature', 'humidity']

POI_COLS = [
    'poi_total', 'poi_transportation', 'poi_residence', 'poi_industry',
    'poi_company', 'poi_shopping', 'poi_restaurant', 'poi_entertainment',
    'poi_recreation_tourism'
]

CONTEXT_COLS = ['work_study', 'housework', 'personal_affair', 'leisure', 'travel']
PERSONAL_COLS = ['age', 'family_income', 'gender', 'education_level']
ALL_TABULAR_COLS = SVI_COLS + ENV_COLS + POI_COLS + CONTEXT_COLS + PERSONAL_COLS

print(f"📊 Feature dimension check: SVI={len(SVI_COLS)}, ENV={len(ENV_COLS)}, POI={len(POI_COLS)}, "
      f"CONTEXT={len(CONTEXT_COLS)}, PERSONAL={len(PERSONAL_COLS)}, Total={len(ALL_TABULAR_COLS)}")


# ================= New: Load Scalers =================
def load_scalers(data_dir):
    """
    Load scalers saved during training

    Returns:
        tuple: (tabular_scaler, feature_cols, norm_method)
    """
    scaler_path = os.path.join(data_dir, "scalers.pkl")

    if not os.path.exists(scaler_path):
        print(f"⚠️ Warning: Scaler file not found at {scaler_path}")
        print("   Unable to perform feature value inversion, PDP X-axis will show normalized values")
        return None, None, None

    try:
        with open(scaler_path, 'rb') as f:
            scaler_data = pickle.load(f)

        tabular_scaler = scaler_data.get('tabular_scaler')
        feature_cols = scaler_data.get('feature_cols')
        norm_method = scaler_data.get('norm_method', 'unknown')

        print(f"✅ Successfully loaded scalers:")
        print(f"   - Normalization method: {norm_method}")
        print(f"   - Number of features: {len(feature_cols)}")

        return tabular_scaler, feature_cols, norm_method

    except Exception as e:
        print(f"❌ Failed to load scalers: {e}")
        return None, None, None


def inverse_transform_feature(feature_values, feature_idx, scaler, feature_cols):
    """
    Inverse transform a single feature

    Args:
        feature_values: Normalized feature values (1D array)
        feature_idx: Feature index in feature_cols
        scaler: Scaler from training
        feature_cols: List of feature column names

    Returns:
        Inverse-transformed feature values
    """
    if scaler is None:
        return feature_values

    try:
        # Create an all-zero array, fill only the target feature position
        n_features = len(feature_cols)
        dummy_data = np.zeros((len(feature_values), n_features))
        dummy_data[:, feature_idx] = feature_values

        # Inverse transform
        inversed_data = scaler.inverse_transform(dummy_data)

        # Extract target feature
        return inversed_data[:, feature_idx]

    except Exception as e:
        print(f"   ⚠️ Inverse transform failed: {e}")
        return feature_values


# ================= 2. Core Utility Functions =================

def set_mixed_mode(model):
    """
    Force gradient computation path (model.train), but freeze randomness (BN/Dropout eval).
    """
    model.train()
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.Dropout, nn.Dropout2d)):
            module.eval()


# ================= New: Validation Set Evaluation Function =================

def evaluate_on_val(model, val_dataset, save_dir):
    """
    Calculate classification metrics using val.npz data, and plot confusion matrix and ROC-AUC curve.

    Args:
        model: Loaded model
        val_dataset: Validation set NPZDataset
        save_dir: Result save directory
    """
    print("\n" + "=" * 60)
    print("📊 Starting validation set evaluation...")
    print("=" * 60)

    model.eval()
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=custom_collate)

    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="   Inference in progress"):
            sat_feats = batch['sat_feats'].to(DEVICE)
            gps_x = batch['gps_x'].to(DEVICE)
            gps_y = batch['gps_y'].to(DEVICE)
            tabular = batch['tabular'].to(DEVICE).float()
            mask = batch['mask'].to(DEVICE)
            labels = batch['label'].cpu().numpy()  # shape: (B,)

            batch_input = {
                'sat_feats': sat_feats,
                'gps_x': gps_x,
                'gps_y': gps_y,
                'tabular': tabular,
                'mask': mask
            }
            logits = model(batch_input)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()  # Positive class probability
            preds = (probs >= 0.5).astype(int)

            all_labels.extend(labels.tolist())
            all_preds.extend(preds.tolist())
            all_probs.extend(probs.tolist())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    # ---------- Calculate Metrics ----------
    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='binary', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='binary', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='binary', zero_division=0)
    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)

    print(f"\n   ✅ Validation set sample count: {len(all_labels)}")
    print(f"   ┌──────────────────────────────────┐")
    print(f"   │ Accuracy  : {acc:.4f}               │")
    print(f"   │ Precision : {precision:.4f}               │")
    print(f"   │ Recall    : {recall:.4f}               │")
    print(f"   │ F1 Score  : {f1:.4f}               │")
    print(f"   │ ROC-AUC   : {roc_auc:.4f}               │")
    print(f"   └──────────────────────────────────┘")

    # Detailed classification report (keep 4 decimal places, manually formatted)
    report_dict = classification_report(
        all_labels, all_preds,
        target_names=['Non-Stress', 'Stress'],
        output_dict=True
    )
    print(f"\n   📋 Classification Report:\n")
    header = f"   {'':>16s}  {'precision':>10s}  {'recall':>10s}  {'f1-score':>10s}  {'support':>8s}"
    print(header)
    print(f"   {'-' * 62}")
    for cls_name in ['Non-Stress', 'Stress']:
        d = report_dict[cls_name]
        print(f"   {cls_name:>16s}  {d['precision']:>10.4f}  {d['recall']:>10.4f}  {d['f1-score']:>10.4f}  {int(d['support']):>8d}")
    print(f"   {'-' * 62}")
    for avg_key in ['accuracy', 'macro avg', 'weighted avg']:
        d = report_dict[avg_key]
        if avg_key == 'accuracy':
            print(f"   {'accuracy':>16s}  {'':>10s}  {'':>10s}  {d:>10.4f}  {len(all_labels):>8d}")
        else:
            print(f"   {avg_key:>16s}  {d['precision']:>10.4f}  {d['recall']:>10.4f}  {d['f1-score']:>10.4f}  {int(d['support']):>8d}")

    # Save metrics to CSV
    metrics_df = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC'],
        'Value': [acc, precision, recall, f1, roc_auc]
    })
    metrics_df.to_csv(os.path.join(save_dir, "val_metrics.csv"), index=False)
    print(f"   💾 Metrics saved to val_metrics.csv")

    # ---------- Confusion Matrix (Percentage + Count dual-line annotation) ----------
    cm = confusion_matrix(all_labels, all_preds)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)   # Row normalization → percentage
    # Annotation text: two lines per cell (percentage + count)
    annot_labels = np.array([
        [f"{cm_norm[i,j]:.2%}\n({cm[i,j]})" for j in range(cm.shape[1])]
        for i in range(cm.shape[0])
    ])
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm_norm, annot=annot_labels, fmt='', cmap='Blues',
        vmin=0, vmax=1,
        xticklabels=['Non-Stress', 'Stress'],
        yticklabels=['Non-Stress', 'Stress'],
        linewidths=0.5, linecolor='gray',
        annot_kws={"size": 13}
    )
    plt.xlabel('Predicted Label', fontsize=13)
    plt.ylabel('True Label', fontsize=13)
    plt.title(
        f"Confusion Matrix (Val Set)\n"
        f"Acc={acc:.4f} | P={precision:.4f} | R={recall:.4f} | F1={f1:.4f}",
        fontsize=12, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "val_confusion_matrix.pdf"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   💾 Confusion matrix saved to val_confusion_matrix.pdf")

    # ---------- ROC-AUC Curve + Save CSV ----------
    # Save FPR/TPR/Threshold data for multi-model comparison plotting
    roc_df = pd.DataFrame({
        'fpr':       fpr,
        'tpr':       tpr,
        'threshold': thresholds if len(thresholds) == len(fpr) else np.append(thresholds, np.nan),
        'auc':       roc_auc,
        'model':     os.path.basename(MODEL_PATH),
    })
    roc_csv_path = os.path.join(save_dir, "val_roc_data.csv")
    roc_df.to_csv(roc_csv_path, index=False)
    print(f"   💾 ROC data saved to val_roc_data.csv ({len(fpr)} threshold points)")

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color='steelblue', lw=2,
             label=f'ROC Curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=1.5, linestyle='--', label='Random Classifier')
    plt.fill_between(fpr, tpr, alpha=0.08, color='steelblue')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.02])
    plt.xlabel('False Positive Rate', fontsize=13)
    plt.ylabel('True Positive Rate', fontsize=13)
    plt.title(f'ROC Curve (Val Set)\nAUC = {roc_auc:.4f}', fontsize=13, fontweight='bold')
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "val_roc_auc.pdf"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   💾 ROC-AUC curve saved to val_roc_auc.pdf")

    print("✅ Validation set evaluation complete!\n")

    return {
        'accuracy': acc, 'precision': precision,
        'recall': recall, 'f1': f1, 'roc_auc': roc_auc
    }


# ================= 3. SHAP Logic (Fixed Version + Inverse Transform) =================

def run_shap_analysis(model, dataset, feature_names, save_dir, specific_timestep=None, scaler_info=None):
    """
    🔥 Fixed SHAP analysis function - Remove batch calculation + Add feature value inversion

    Args:
        specific_timestep: None means use all timesteps (flattened), 0-29 means specific timestep
        scaler_info: tuple of (tabular_scaler, feature_cols, norm_method)
    """
    # Unpack scaler info
    tabular_scaler, scaler_feature_cols, norm_method = scaler_info if scaler_info else (None, None, None)

    if specific_timestep is not None:
        if not (0 <= specific_timestep < SEQ_LEN):
            raise ValueError(f"Timestep must be between 0-{SEQ_LEN - 1}, current value: {specific_timestep}")
        print(f"\n🧠 Starting SHAP analysis - 【Analyzing only timestep T={specific_timestep}】")
        print(f"   Sampling {SHAP_SAMPLES} samples, analyzing all {len(feature_names)} features...")
        print(f"   📊 Target variable: Stress category logit value (preserving non-linear relationships)")
    else:
        print(f"\n🧠 Starting SHAP analysis - 【All timesteps flattened for mixed analysis】")
        print(f"   Sampling {SHAP_SAMPLES} samples, analyzing all {len(feature_names)} features...")
        print(f"   📊 Target variable: Stress category logit value (preserving non-linear relationships)")

    if tabular_scaler is not None:
        print(f"   🔄 Feature value inversion: Enabled (normalization method: {norm_method})")
    else:
        print(f"   ⚠️ Feature value inversion: Disabled (scaler not found)")

    try:
        # 1. Ensure model is in mixed mode
        set_mixed_mode(model)

        # 2. Sample background data
        loader = DataLoader(dataset, batch_size=SHAP_SAMPLES, shuffle=True, collate_fn=custom_collate)
        batch = next(iter(loader))

        tabular = batch['tabular'].to(DEVICE).float().requires_grad_(True)
        sat_feats = batch['sat_feats'].to(DEVICE)
        gps_x = batch['gps_x'].to(DEVICE)
        gps_y = batch['gps_y'].to(DEVICE)
        mask = batch['mask'].to(DEVICE)

        print(f"   📐 Tabular shape: {tabular.shape}, Expected features: {len(feature_names)}")

        # Wrapper class
        class TabularModelWrapper(nn.Module):
            def __init__(self, original_model, fixed_sat, fixed_gps_x, fixed_gps_y, fixed_mask):
                super().__init__()
                self.model = original_model
                self.fixed_sat = fixed_sat
                self.fixed_gps_x = fixed_gps_x
                self.fixed_gps_y = fixed_gps_y
                self.fixed_mask = fixed_mask

            def forward(self, tabular_input):
                self.model.train()  # Ensure train mode
                curr_bs = tabular_input.shape[0]
                orig_bs = self.fixed_sat.shape[0]

                sat, gx, gy, mk = self.fixed_sat, self.fixed_gps_x, self.fixed_gps_y, self.fixed_mask

                if curr_bs != orig_bs:
                    if curr_bs % orig_bs == 0:
                        rpt = curr_bs // orig_bs
                        sat = sat.repeat(rpt, 1, 1, 1)
                        gx = gx.repeat(rpt, 1)
                        gy = gy.repeat(rpt, 1)
                        mk = mk.repeat(rpt, 1)
                    else:
                        limit = min(curr_bs, orig_bs)
                        sat = sat[:limit]
                        gx = gx[:limit]
                        gy = gy[:limit]
                        mk = mk[:limit]
                        if tabular_input.shape[0] > limit:
                            tabular_input = tabular_input[:limit]

                batch_input = {'sat_feats': sat, 'gps_x': gx, 'gps_y': gy, 'tabular': tabular_input, 'mask': mk}
                logits = self.model(batch_input)

                # Return original logits, preserving non-linear relationships
                return logits[:, 1].unsqueeze(1)

        wrapper_model = TabularModelWrapper(model, sat_feats, gps_x, gps_y, mask)

        # 3. Calculate SHAP (🔥 Critical fix: one-time calculation, no batching)
        print("   👉 Computing Gradient SHAP (temporarily disabling cuDNN)...")
        print(f"      ⏳ Sample count: {tabular.shape[0]}, Feature count: {len(feature_names)}")

        original_cudnn_state = torch.backends.cudnn.enabled
        torch.backends.cudnn.enabled = False

        try:
            explainer = shap.GradientExplainer(wrapper_model, tabular)

            # 🔥 Fix: one-time calculation, no batching
            print(f"      🔄 Starting SHAP value calculation...")
            shap_values = explainer.shap_values(tabular)
            print("      ✅ SHAP value calculation complete!")

        finally:
            torch.backends.cudnn.enabled = original_cudnn_state

        # Process return values
        if isinstance(shap_values, list):
            vals = shap_values[0]
        else:
            vals = shap_values

        # Force conversion to NumPy array
        if hasattr(vals, 'detach'):
            vals = vals.detach().cpu().numpy()

        # Remove extra dimensions
        if vals.ndim == 4 and vals.shape[-1] == 1:
            vals = vals.squeeze(-1)
            print(f"   🔧 Fixed SHAP value dimensions: {vals.shape}")

        print(f"   SHAP Values Shape: {vals.shape}, Type: {type(vals)}")

        # 4. Data organization
        n_features = len(feature_names)

        if specific_timestep is not None:
            # Only analyze specified timestep
            print(f"   📌 Extracting data from timestep T={specific_timestep}...")

            vals_at_t = vals[:, specific_timestep, :]
            tabular_at_t = tabular[:, specific_timestep, :].detach().cpu().numpy()
            mask_at_t = mask[:, specific_timestep].cpu().numpy()

            valid_indices = (mask_at_t == 1)
            flat_vals_valid = vals_at_t[valid_indices]
            flat_data_valid = tabular_at_t[valid_indices]

            suffix = f"_T{specific_timestep}"
            title_suffix = f" (Time Step T={specific_timestep})"

        else:
            # Flatten all timesteps
            print(f"   📌 Flattening all timesteps for mixed analysis...")

            valid_indices = mask.cpu().numpy().flatten() == 1
            flat_vals = vals.reshape(-1, n_features)
            flat_data = tabular.detach().cpu().numpy().reshape(-1, n_features)

            flat_vals_valid = flat_vals[valid_indices]
            flat_data_valid = flat_data[valid_indices]

            suffix = "_all_timesteps"
            title_suffix = " (All Time Steps Pooled)"

        print(f"   ✅ Valid sample count: {flat_vals_valid.shape[0]}, Feature count: {flat_vals_valid.shape[1]}")

        # 5. Basic SHAP plots
        print("   📊 Generating SHAP summary plots...")
        plt.figure(figsize=(10, max(8, len(feature_names) * 0.3)))
        shap.summary_plot(flat_vals_valid, flat_data_valid, feature_names=feature_names,
                          show=False, max_display=len(feature_names))
        plt.title(f"SHAP Summary Plot{title_suffix}")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"shap_summary_beeswarm{suffix}.pdf"), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"      ✅ Saved: shap_summary_beeswarm{suffix}.png")

        plt.figure(figsize=(10, max(8, len(feature_names) * 0.3)))
        shap.summary_plot(flat_vals_valid, flat_data_valid, feature_names=feature_names,
                          plot_type="bar", show=False, max_display=len(feature_names))
        plt.title(f"Global Feature Importance{title_suffix}")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"shap_importance_bar{suffix}.pdf"), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"      ✅ Saved: shap_importance_bar{suffix}.png")

        # 6. SHAP Dependence Plots (🔥 New: with feature value inversion)
        print(f"   📉 Generating SHAP Dependence Plots (all {len(feature_names)} features)...")

        if tabular_scaler is not None:
            print(f"      🔄 Enabling feature value inversion to original scale")

        print(f"      📐 Debug info:")
        print(f"         - flat_vals_valid shape: {flat_vals_valid.shape}")
        print(f"         - flat_data_valid shape: {flat_data_valid.shape}")

        # Ensure data is 2D
        if flat_vals_valid.ndim > 2:
            print(f"      ⚠️ Detected abnormal SHAP value dimensions, correcting...")
            flat_vals_valid = flat_vals_valid.reshape(-1, n_features)
            print(f"      ✅ Corrected shape: {flat_vals_valid.shape}")

        if flat_data_valid.ndim > 2:
            print(f"      ⚠️ Detected abnormal feature data dimensions, correcting...")
            flat_data_valid = flat_data_valid.reshape(-1, n_features)
            print(f"      ✅ Corrected shape: {flat_data_valid.shape}")

        dep_dir = os.path.join(save_dir, f"shap_dependence_plots{suffix}")
        os.makedirs(dep_dir, exist_ok=True)

        # Calculate global importance ranking
        global_importance = np.abs(flat_vals_valid).mean(axis=0)
        sorted_indices = np.argsort(global_importance)[::-1]

        # Generate dependence plot for all features
        # 🔥 Modification: Don't use shap.dependence_plot, manually plot to support:
        #   1. Remove interaction color (points uniformly light blue)
        #   2. Add GAM trend red dashed line
        #   3. Global Times New Roman font
        if not HAS_PYGAM:
            print("   ⚠️ pygam not installed, skipping GAM analysis")
            return

        failed_features = []
        for rank, idx in enumerate(tqdm(sorted_indices, desc="      Generating Dependence Plots", unit="feature"), 1):
            idx = int(idx)
            feat_name = feature_names[idx]
            importance = global_importance[idx]

            fig, ax = plt.subplots(figsize=(8, 6))

            try:
                # Inverse transform current feature
                feature_data_for_plot = flat_data_valid.copy()
                x_label = feat_name

                if tabular_scaler is not None and scaler_feature_cols is not None:
                    original_values = inverse_transform_feature(
                        flat_data_valid[:, idx],
                        idx,
                        tabular_scaler,
                        scaler_feature_cols
                    )
                    feature_data_for_plot[:, idx] = original_values
                    x_label = f"{feat_name} (Original Scale)"

                x_vals  = feature_data_for_plot[:, idx]   # Feature values (N,)
                y_vals  = flat_vals_valid[:, idx]          # SHAP values (N,)

                # ── Scatter plot (light blue, no interaction color) ───────────────────────────
                ax.scatter(x_vals, y_vals,
                           color='steelblue', alpha=0.45, s=18,
                           linewidths=0, zorder=2)

                # ── GAM trend red dashed line ─────────────────────────────────
                try:
                    valid_mask = np.isfinite(x_vals) & np.isfinite(y_vals)
                    if valid_mask.sum() >= 10:
                        # n_splines: number of spline bases, larger means more curved trend line (default 25 too many, changed to 10)
                        # lam: regularization strength, larger means smoother; use gridsearch to automatically select optimal from candidates
                        gam = LinearGAM(gam_s(0, n_splines=10))
                        lam_candidates = np.logspace(-3, 3, 30)  # 0.001 ~ 1000
                        gam.gridsearch(
                            x_vals[valid_mask].reshape(-1, 1),
                            y_vals[valid_mask],
                            lam=lam_candidates,
                            progress=False
                        )
                        x_grid = np.linspace(x_vals[valid_mask].min(),
                                             x_vals[valid_mask].max(), 300)
                        y_pred = gam.predict(x_grid.reshape(-1, 1))
                        ax.plot(x_grid, y_pred,
                                color='red', linestyle='--', linewidth=2.0,
                                label='GAM trend', zorder=3)
                        ax.legend(fontsize=11, prop={'family': 'Times New Roman'})
                except Exception:
                    pass  # When GAM fails, only keep scatter plot

                # ── Title / Axis labels ─────────────────────────────────────
                title_str = f"Rank {rank}: {feat_name}{title_suffix}\n(Importance: {importance:.4f}"
                if tabular_scaler is not None:
                    title_str += ", X-axis: Original Scale)"
                else:
                    title_str += ")"
                ax.set_title(title_str, fontsize=12,
                             fontfamily='Times New Roman')
                ax.set_xlabel(x_label, fontsize=12,
                              fontfamily='Times New Roman')
                ax.set_ylabel(f"SHAP value for {feat_name}", fontsize=12,
                              fontfamily='Times New Roman')
                ax.tick_params(labelsize=10)
                for lbl in ax.get_xticklabels() + ax.get_yticklabels():
                    lbl.set_fontfamily('Times New Roman')

                ax.axhline(0, color='gray', linewidth=0.8, linestyle='-', alpha=0.5)
                ax.grid(True, alpha=0.25, linestyle='--')

                safe_name = feat_name.replace("/", "_").replace(" ", "_").replace("_percent", "")
                plt.tight_layout()
                plt.savefig(os.path.join(dep_dir, f"{rank:02d}_dependence_{safe_name}.pdf"),
                            dpi=300, bbox_inches='tight')
                plt.close()

            except Exception as e:
                failed_features.append((feat_name, str(e)))
                plt.close()

        if failed_features:
            print(f"      ⚠️ {len(failed_features)} features failed to generate:")
            for fname, err in failed_features[:5]:
                print(f"         - {fname}: {err}")
            if len(failed_features) > 5:
                print(f"         ... and {len(failed_features) - 5} more")
        else:
            print(f"      ✅ All {len(feature_names)} features generated successfully!")

        # 7. Save importance ranking table
        importance_df = pd.DataFrame({
            'Rank': range(1, len(feature_names) + 1),
            'Feature': [feature_names[int(i)] for i in sorted_indices],
            'Importance': global_importance[sorted_indices].flatten()
        })
        csv_name = f"feature_importance_ranking{suffix}.csv"
        importance_df.to_csv(os.path.join(save_dir, csv_name), index=False)
        print(f"   💾 Feature importance ranking saved to {csv_name}")

        print(f"✅ SHAP analysis{title_suffix} completed successfully!")

    except Exception as e:
        print(f"⚠️ SHAP analysis failed: {e}")
        traceback.print_exc()


# ================= 4. Regular Plotting Functions =================

def plot_modal_temporal(importance_dict, save_dir, seq_len=30):
    print(f"\n📊 Generating modal-temporal importance plot (sequence length={seq_len})...")

    modalities = list(importance_dict.keys())

    print("   Original importance value statistics:")
    for m in modalities:
        vals = importance_dict[m]
        print(f"     {m:12s}: mean={np.mean(vals):.6f}, std={np.std(vals):.6f}, "
              f"min={np.min(vals):.6f}, max={np.max(vals):.6f}")

    matrix = np.array([importance_dict[m][:seq_len] for m in modalities])
    normalized_matrix = np.zeros_like(matrix)

    for i, m in enumerate(modalities):
        row = matrix[i]
        row_min, row_max = row.min(), row.max()
        if row_max - row_min > 1e-8:
            normalized_matrix[i] = (row - row_min) / (row_max - row_min)
        else:
            normalized_matrix[i] = np.zeros_like(row)

    plt.figure(figsize=(14, 6))
    ax = sns.heatmap(normalized_matrix, cmap="YlOrRd",
                     xticklabels=range(1, seq_len + 1),
                     yticklabels=modalities,
                     annot=False, fmt='.2f',
                     cbar_kws={'label': 'Normalized Importance (0-1)'})

    plt.title(f"Modal-Temporal Importance (Sequence Length: {seq_len})",
              fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Time Step', fontsize=12)
    plt.ylabel('Modality', fontsize=12)

    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha='center')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "modal_temporal_importance.pdf"), dpi=300, bbox_inches='tight')
    plt.close()

    df = pd.DataFrame(normalized_matrix, index=modalities,
                      columns=[f"T{i + 1}" for i in range(seq_len)])
    df.to_csv(os.path.join(save_dir, "modal_temporal_importance.csv"))
    print(f"   💾 Modal-temporal importance saved to modal_temporal_importance.csv")

    plt.figure(figsize=(12, 6))
    for i, m in enumerate(modalities):
        plt.plot(range(1, seq_len + 1), normalized_matrix[i],
                 marker='o', label=m, linewidth=2, markersize=4)

    plt.xlabel('Time Step', fontsize=12)
    plt.ylabel('Normalized Importance', fontsize=12)
    plt.title('Temporal Importance Trends by Modality', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11, loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "modal_temporal_trends.pdf"), dpi=300, bbox_inches='tight')
    plt.close()

    print("✅ Modal-temporal analysis complete")


def generate_clip_gradcam(sat_feats, sat_grads, output_path, time_step=None):
    cam_weights = (sat_feats * sat_grads).sum(dim=-1)
    patch_weights = cam_weights[:, :, 1:]
    b, t, _ = patch_weights.shape
    cams = patch_weights.view(b, t, 7, 7)
    cams = F.relu(cams)

    if time_step is None:
        time_step = t // 2

    sample_cam = cams[0, time_step].detach().cpu().numpy()
    sample_cam = (sample_cam - sample_cam.min()) / (sample_cam.max() - sample_cam.min() + 1e-8)
    heatmap = cv2.resize(sample_cam, (224, 224))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    cv2.imwrite(output_path, heatmap)



# ================= New: Batch GradCAM Generation Function =================



# ================= New: Batch Patch-SHAP Heatmap Generation Function =================

def generate_patch_shap_maps(model, dataset, save_dir, sat_dir,
                              seq_len=30, n_samples=PATCH_SHAP_SAMPLES):
    """
    Randomly sample n_samples sequences, for each sequence's T=0 timestep satellite patch features
    use GradientExplainer (Expected Gradients) to calculate true SHAP attribution heatmaps.

    Principle:
    - Background data: sat_feats[:, 0, :, :] from randomly sampled batches, i.e., (N, 50, 768)
    - Data to explain: sat_feats[:, 0, :, :] from the same batch, i.e., self-explanation on background
    - GradientExplainer takes Expected Gradients over multiple background baselines, satisfying SHAP axioms
    - For each sample's 49 patch tokens (excluding CLS), take mean over 768 dims, get 7×7 signed importance
    - Positive (red) = promote high stress prediction, Negative (blue) = suppress high stress prediction

    Args:
        model:     Loaded model
        dataset:   Complete dataset (ConcatDataset or NPZDataset)
        save_dir:  Result save directory
        sat_dir:   Satellite image directory (for overlay base map)
        seq_len:   Sequence length
        n_samples: Number of randomly sampled sequences (corresponding to PATCH_SHAP_SAMPLES config)
    """

    patch_shap_dir   = os.path.join(save_dir, 'sat_patch_shap')
    patch_shap_dir_0 = os.path.join(patch_shap_dir, 'pred_0_non_stress')
    patch_shap_dir_1 = os.path.join(patch_shap_dir, 'pred_1_stress')
    os.makedirs(patch_shap_dir_0, exist_ok=True)
    os.makedirs(patch_shap_dir_1, exist_ok=True)

    print("\n" + "=" * 60)
    print("🛰️  Starting batch Patch SHAP heatmap generation (GradientExplainer)")
    print(f"   Randomly sampling sequence count: {n_samples}  (adjustable via PATCH_SHAP_SAMPLES)")
    print(f"   Analysis timestep:     T=0")
    print(f"   Save directory: {patch_shap_dir}")
    print(f"   Original image directory: {sat_dir}")
    print("=" * 60)

    set_mixed_mode(model)

    # ── 1. Collect all img_names and randomly determine sample indices ───────────────────
    if isinstance(dataset, ConcatDataset):
        sub_datasets = dataset.datasets
    else:
        sub_datasets = [dataset]

    all_img_names = []
    for ds in sub_datasets:
        if hasattr(ds, 'img_names'):
            for row in ds.img_names:
                all_img_names.append(list(row))
        else:
            print("   warning: sub-dataset has no img_names attribute, skipping")

    total_samples = len(dataset)
    actual_n = min(n_samples, total_samples)
    sampled_indices = sorted(
        np.random.choice(total_samples, size=actual_n, replace=False).tolist()
    )
    print(f"   Total dataset size: {total_samples}  →  Actual sample count: {actual_n} sequences")

    # ── 2. Use Subset + DataLoader to load all sampled data at once ─────────────────
    sampled_dataset = Subset(dataset, sampled_indices)
    # batch_size = actual_n, load all sampled samples at once, use as background and data to explain
    bg_loader = DataLoader(sampled_dataset, batch_size=actual_n,
                           shuffle=False, collate_fn=custom_collate)
    bg_batch = next(iter(bg_loader))

    sat_all   = bg_batch['sat_feats'].to(DEVICE)   # (N, SEQ_LEN, 50, 768)
    gps_x_all = bg_batch['gps_x'].to(DEVICE)
    gps_y_all = bg_batch['gps_y'].to(DEVICE)
    tab_all   = bg_batch['tabular'].to(DEVICE).float()
    mask_all  = bg_batch['mask'].to(DEVICE)

    # Extract T=0 patch features as background and input to explain: (N, 50, 768)
    sat_t0_all = sat_all[:, 0, :, :]

    print(f"   📐 sat_t0_all shape: {sat_t0_all.shape}  (N, n_patches+1, patch_dim)")

    # ── 3. Define Patch Wrapper ────────────────────────────────────────
    class PatchT0Wrapper(nn.Module):
        """
        Only perturb T=0 timestep patch features for SHAP,
        other timesteps and other modalities fixed to corresponding sample's original values.

        Input: sat_t0 (B, 50, 768)  — SHAP perturbed T=0 patch features
        Output: Stress category logit (B, 1), with sign
        """
        def __init__(self, base_model, full_sat, full_gps_x,
                     full_gps_y, full_tabular, full_mask):
            super().__init__()
            self.model      = base_model
            self.full_sat   = full_sat      # (N, SEQ_LEN, 50, 768)
            self.full_gps_x = full_gps_x   # (N, SEQ_LEN)
            self.full_gps_y = full_gps_y   # (N, SEQ_LEN)
            self.full_tab   = full_tabular  # (N, SEQ_LEN, tabular_dim)
            self.full_mask  = full_mask     # (N, SEQ_LEN)

        def _align(self, tensor, curr_bs):
            """repeat / truncate to align tensor with current batch size"""
            orig = tensor.shape[0]
            if curr_bs == orig:
                return tensor
            if curr_bs > orig and curr_bs % orig == 0:
                return tensor.repeat(curr_bs // orig,
                                     *([1] * (tensor.dim() - 1)))
            return tensor[:curr_bs]

        def forward(self, sat_t0_input):
            """
            sat_t0_input: (B, 50, 768) — Current batch perturbed T=0 patch features
            """
            self.model.train()  # Keep train mode to activate gradient path
            B = sat_t0_input.shape[0]

            # Take aligned complete satellite features, replace T=0 with perturbed values
            full_sat_aligned = self._align(self.full_sat, B).clone()
            full_sat_aligned[:, 0, :, :] = sat_t0_input  # Replace T=0

            batch_input = {
                'sat_feats': full_sat_aligned,
                'gps_x':     self._align(self.full_gps_x, B),
                'gps_y':     self._align(self.full_gps_y, B),
                'tabular':   self._align(self.full_tab,   B),
                'mask':      self._align(self.full_mask,  B),
            }
            logits = self.model(batch_input)
            # Return stress category logit (with sign, preserving non-linear relationships)
            return logits[:, 1].unsqueeze(1)

    wrapper = PatchT0Wrapper(
        model, sat_all, gps_x_all, gps_y_all, tab_all, mask_all
    )

    # ── 4. GradientExplainer: Use all sampled T=0 features as background ────────
    print("   👉 Building GradientExplainer (background = all sampled T=0 patch features)...")
    print("      Temporarily disabling cuDNN...")

    orig_cudnn = torch.backends.cudnn.enabled
    torch.backends.cudnn.enabled = False

    try:
        # Background data: sat_t0_all (N, 50, 768)
        sat_t0_bg = sat_t0_all.detach().requires_grad_(False)
        explainer = shap.GradientExplainer(wrapper, sat_t0_bg)

        print(f"      🔄 Computing SHAP values for {actual_n} samples (one-time, no batching)...")
        sat_t0_explain = sat_t0_all.detach().requires_grad_(True)
        shap_values = explainer.shap_values(sat_t0_explain)
        print("      ✅ Patch SHAP value calculation complete!")

    finally:
        torch.backends.cudnn.enabled = orig_cudnn

    # ── 5. Process SHAP return values ─────────────────────────────────────────
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    if hasattr(shap_values, 'detach'):
        shap_values = shap_values.detach().cpu().numpy()
    # Remove output dimension: (N, 50, 768, 1) → (N, 50, 768)
    if shap_values.ndim == 4 and shap_values.shape[-1] == 1:
        shap_values = shap_values.squeeze(-1)

    print(f"   ✅ Patch SHAP shape: {shap_values.shape}")
    # shap_values: (N, 50, 768)

    # ── 6. Generate heatmap for each sample and overlay on original image ─────────────────────────────────
    TARGET_SIZE = 1024
    rdbu_r = _cm.get_cmap('RdBu_r')

    # Get predicted classes for sampled samples (using complete model, eval mode)
    model.eval()
    with torch.no_grad():
        logits_all = []
        for i in range(0, actual_n, BATCH_SIZE):
            end = min(i + BATCH_SIZE, actual_n)
            bi = {
                'sat_feats': sat_all[i:end],
                'gps_x':     gps_x_all[i:end],
                'gps_y':     gps_y_all[i:end],
                'tabular':   tab_all[i:end],
                'mask':      mask_all[i:end],
            }
            logits_all.append(model(bi).cpu())
        logits_all = torch.cat(logits_all, dim=0)
        pred_classes = torch.argmax(logits_all, dim=1).numpy()
    set_mixed_mode(model)  # Restore mixed mode for subsequent use

    success_count = 0
    skip_no_img   = 0

    for local_idx, global_idx in enumerate(tqdm(sampled_indices,
                                                 desc='   Rendering Patch SHAP heatmaps')):
        # ── Get image name ──────────────────────────────────────────────
        if global_idx >= len(all_img_names):
            skip_no_img += 1
            continue

        raw_name = str(all_img_names[global_idx][0])  # T=0 corresponds to index=0
        if not raw_name or raw_name in ('', 'nan', 'None'):
            skip_no_img += 1
            continue

        base_name = os.path.splitext(raw_name)[0]
        img_path  = os.path.join(sat_dir, base_name + '.jpg')

        # ── Calculate 7×7 patch importance ────────────────────────────────────
        # shap_values[local_idx]: (50, 768)
        # Remove CLS token (index=0) → (49, 768)
        patch_shap = shap_values[local_idx, 1:, :]   # (49, 768)

        # Take mean over 768 dims, get signed scalar importance (49,)
        patch_importance = patch_shap.mean(axis=-1)

        grid_size = int(len(patch_importance) ** 0.5)  # 7
        shap_map  = patch_importance.reshape(grid_size, grid_size)

        # ── Symmetric normalization to [-1, 1] ─────────────────────────────────────
        abs_max  = max(abs(shap_map.max()), abs(shap_map.min())) + 1e-8
        shap_map = shap_map / abs_max

        # ── Upsampling + Gaussian smoothing ─────────────────────────────────────────
        heatmap = cv2.resize(shap_map.astype(np.float32),
                             (TARGET_SIZE, TARGET_SIZE),
                             interpolation=cv2.INTER_CUBIC)
        k = (TARGET_SIZE // grid_size // 2) * 2 + 1   # ~73, odd number
        heatmap = cv2.GaussianBlur(heatmap, (k, k), sigmaX=0)

        # ── Two-color rendering: Blue(negative contribution)→White(zero)→Red(positive contribution) ───────────────────
        heatmap_01    = np.clip((heatmap + 1) / 2, 0, 1)
        cam_rgba      = rdbu_r(heatmap_01)
        cam_rgb       = (cam_rgba[:, :, :3] * 255).astype(np.uint8)
        heatmap_color = cv2.cvtColor(cam_rgb, cv2.COLOR_RGB2BGR)

        # ── Overlay on original image ──────────────────────────────────────────────────
        if os.path.exists(img_path):
            orig_img = cv2.imread(img_path)
            if orig_img is not None:
                if orig_img.shape[:2] != (TARGET_SIZE, TARGET_SIZE):
                    orig_img = cv2.resize(orig_img, (TARGET_SIZE, TARGET_SIZE),
                                          interpolation=cv2.INTER_AREA)
            else:
                orig_img = np.zeros((TARGET_SIZE, TARGET_SIZE, 3), dtype=np.uint8)
        else:
            orig_img = np.zeros((TARGET_SIZE, TARGET_SIZE, 3), dtype=np.uint8)

        combined = cv2.addWeighted(orig_img, 0.5, heatmap_color, 0.5, 0)

        # ── Save to corresponding subdirectory ──────────────────────────────────────────
        pred_class = int(pred_classes[local_idx])
        sub_dir    = patch_shap_dir_1 if pred_class == 1 else patch_shap_dir_0
        out_path   = os.path.join(sub_dir, base_name + '.png')
        cv2.imwrite(out_path, combined)
        success_count += 1

    print(f"\n   Successfully generated: {success_count} images")
    if skip_no_img:
        print(f"   Skipped (img_name empty or index out of bounds): {skip_no_img} samples")
    print("✅ Batch Patch SHAP heatmap generation complete!\n")


# ================= New: Modal-Temporal SHAP Analysis =================

def run_modal_temporal_shap(model, dataset, save_dir,
                             seq_len=30, shap_samples=200):
    """
    Use GradientExplainer to calculate SHAP importance of three modality types at each timestep:

      1. GPS      : Input (N, SEQ_LEN, 2) [lon, lat], output merged SHAP absolute mean per timestep
      2. Satellite: Input (N, SEQ_LEN, 50, 768), take mean over patch×dim then get SHAP per timestep
      3. Tabular  : Input (N, SEQ_LEN, n_feat), output SHAP per timestep × per feature dimension

    Visualization:
      - Heatmap: rows=modalities/features, columns=timesteps (normalized for display)
      - Line plot: Importance trends of each modality over timesteps
      - Tabular-specific heatmap: rows=feature names, columns=timesteps

    Args:
        model:        Loaded model
        dataset:      Complete dataset
        save_dir:     Result save directory
        seq_len:      Sequence length
        shap_samples: Background sample count
    """
    print("\n" + "=" * 60)
    print("📊 Starting modal-temporal analysis")
    print(f"   GPS / Satellite: Gradient×input IG method, full dataset")
    print(f"   Tabular: GradientExplainer SHAP, {shap_samples} sampled samples")
    print("=" * 60)

    set_mixed_mode(model)

    # ── 1. GPS & Satellite: Gradient×input, iterate over full dataset ──────────────────
    # Completely consistent with original IG implementation: accumulate |grad * input| for all batches,
    # unaffected by zero padding (valid timesteps themselves have non-zero input, mask ensures correct aggregation)
    loader_full = DataLoader(dataset, batch_size=BATCH_SIZE,
                             shuffle=False, collate_fn=custom_collate)

    gps_imp = np.zeros(seq_len)
    sat_imp = np.zeros(seq_len)
    gps_cnt = 0
    sat_cnt = 0

    print("\n   🔄 [1/2] Computing GPS & Satellite gradient importance (full dataset)...")
    for batch in tqdm(loader_full, desc="      IG iteration", leave=False):
        sat_feats = batch['sat_feats'].to(DEVICE).requires_grad_(True)
        gps_x     = batch['gps_x'].to(DEVICE).float().requires_grad_(True)
        gps_y     = batch['gps_y'].to(DEVICE).float().requires_grad_(True)
        tabular   = batch['tabular'].to(DEVICE).float()
        mask      = batch['mask'].to(DEVICE)

        sat_feats.retain_grad()
        gps_x.retain_grad()
        gps_y.retain_grad()

        logits = model({'sat_feats': sat_feats, 'gps_x': gps_x,
                        'gps_y': gps_y, 'tabular': tabular, 'mask': mask})
        logits[:, 1].sum().backward()

        with torch.no_grad():
            if sat_feats.grad is not None:
                # (B, T, 50, 768) -> (B, T) -> (T,)
                imp = torch.abs(sat_feats * sat_feats.grad).mean(dim=(2, 3))
                sat_imp[:imp.shape[1]] += imp.mean(dim=0).cpu().numpy()[:seq_len]
                sat_cnt += 1
            if gps_x.grad is not None and gps_y.grad is not None:
                # (B, T) -> (T,)
                imp = (torch.abs(gps_x * gps_x.grad) +
                       torch.abs(gps_y * gps_y.grad))
                gps_imp += imp.mean(dim=0).cpu().numpy()[:seq_len]
                gps_cnt += 1

    if sat_cnt > 0: sat_imp /= sat_cnt
    if gps_cnt > 0: gps_imp /= gps_cnt
    gps_t = gps_imp   # (T,)
    sat_t = sat_imp   # (T,)
    print("      ✅ GPS & Satellite IG calculation complete")

    # valid_counts: use mask to count valid samples per timestep (for visualization confidence only)
    all_masks = []
    for batch in DataLoader(dataset, batch_size=BATCH_SIZE,
                            shuffle=False, collate_fn=custom_collate):
        all_masks.append(batch['mask'].numpy())
    all_masks    = np.concatenate(all_masks, axis=0)
    valid_counts = all_masks[:, :seq_len].sum(axis=0).astype(int)

    # ── 2. Tabular: SHAP full sequence, background/explanation independently sampled ─────────────────
    # Background sample count limited to shap_samples // 4, avoid GradientExplainer initialization OOM
    tab_bg_size = max(32, shap_samples // 4)
    tab_ex_size = max(32, shap_samples // 4)
    print(f"\n   🔄 [2/2] Computing Tabular SHAP "
          f"(background={tab_bg_size}, explain={tab_ex_size})...")

    # Clean GPU cache left from IG phase
    torch.cuda.empty_cache()

    pool_loader = iter(DataLoader(dataset,
                                  batch_size=tab_bg_size + tab_ex_size,
                                  shuffle=True, collate_fn=custom_collate))
    pool_batch  = next(pool_loader)

    def _to(b, key, float_=False):
        t = b[key].to(DEVICE)
        return t.float() if float_ else t

    # Background batch: only send tabular features and fixed modalities to GPU, satellite features largest volume, truncate separately
    bg_tab  = _to(pool_batch, 'tabular', True)[:tab_bg_size]
    ex_tab  = _to(pool_batch, 'tabular', True)[tab_bg_size:tab_bg_size + tab_ex_size]
    bg_sat  = _to(pool_batch, 'sat_feats')[:tab_bg_size]
    bg_gx   = _to(pool_batch, 'gps_x',   True)[:tab_bg_size]
    bg_gy   = _to(pool_batch, 'gps_y',   True)[:tab_bg_size]
    bg_msk  = _to(pool_batch, 'mask')[:tab_bg_size]
    ex_mask_np = pool_batch['mask'].numpy()[tab_bg_size:tab_bg_size + tab_ex_size]

    # Clean again, ensure only necessary tensors are kept
    torch.cuda.empty_cache()

    n_feat = bg_tab.shape[2]

    class TabWrapper(nn.Module):
        def __init__(self, m, sat, gx, gy, msk):
            super().__init__()
            self.m = m; self.sat = sat
            self.gx = gx; self.gy = gy; self.msk = msk

        def _a(self, t, B):
            return t.repeat(B // t.shape[0], *([1] * (t.dim()-1))) \
                   if B > t.shape[0] and B % t.shape[0] == 0 else t[:B]

        def forward(self, tab):
            self.m.train()
            B = tab.shape[0]
            return self.m({'sat_feats': self._a(self.sat, B),
                           'gps_x':    self._a(self.gx,  B),
                           'gps_y':    self._a(self.gy,  B),
                           'tabular':  tab,
                           'mask':     self._a(self.msk, B)})[:, 1].unsqueeze(1)

    orig_cudnn = torch.backends.cudnn.enabled
    torch.backends.cudnn.enabled = False
    try:
        tab_w    = TabWrapper(model, bg_sat, bg_gx, bg_gy, bg_msk)
        tab_exp  = shap.GradientExplainer(tab_w, bg_tab)
        tab_shap = tab_exp.shap_values(ex_tab)
        if isinstance(tab_shap, list): tab_shap = tab_shap[0]
        if hasattr(tab_shap, 'detach'): tab_shap = tab_shap.detach().cpu().numpy()
        if tab_shap.ndim == 4 and tab_shap.shape[-1] == 1:
            tab_shap = tab_shap.squeeze(-1)
        print(f"      ✅ Tabular SHAP shape: {tab_shap.shape}")
    finally:
        torch.backends.cudnn.enabled = orig_cudnn
        torch.cuda.empty_cache()

    # ── 3. Aggregate Tabular (using mask) ───────────────────────────────────────
    tab_per_t  = np.abs(tab_shap).mean(axis=2)   # (N, T)
    tab_feat_t = np.zeros((seq_len, n_feat))
    tab_t      = np.zeros(seq_len)

    for t in range(seq_len):
        valid = ex_mask_np[:, t] == 1
        if valid.sum() > 0:
            tab_t[t]      = tab_per_t[valid, t].mean()
            tab_feat_t[t] = np.abs(tab_shap[valid, t, :]).mean(axis=0)

    # ── 4. Save raw data ───────────────────────────────────────────────
    modal_df = pd.DataFrame({
        'TimeStep':    [f"T{t+1}" for t in range(seq_len)],
        'GPS':          gps_t,
        'Satellite':    sat_t,
        'Tabular':      tab_t,
        'ValidSamples': valid_counts,
    })
    modal_df.to_csv(os.path.join(save_dir, "modal_temporal_shap.csv"), index=False)
    print("\n   💾 Modal-temporal importance saved: modal_temporal_shap.csv")

    feat_df = pd.DataFrame(
        tab_feat_t,
        index=[f"T{t+1}" for t in range(seq_len)],
        columns=ALL_TABULAR_COLS
    )
    feat_df.to_csv(os.path.join(save_dir, "tabular_feature_temporal_shap.csv"))
    print("   💾 Tabular feature temporal SHAP saved: tabular_feature_temporal_shap.csv")

    # ── 5. Visualization ─────────────────────────────────────────────────────
    _plot_modal_temporal_shap(gps_t, sat_t, tab_t, tab_feat_t,
                              valid_counts, ALL_TABULAR_COLS, seq_len, save_dir)

    print("✅ Modal-temporal analysis complete!\n")



def _plot_modal_temporal_shap(gps_t, sat_t, tab_t, tab_feat_t,
                               valid_counts, feature_names, seq_len, save_dir):
    """
    Merged heatmap: row order is
      GPS | RSIs | SVIs | Mobile sensors | POI compositions | Activity contexts
    Tabular features grouped by aggregation (mean), PERSONAL_COLS not included in temporal analysis.
    Unreliable timestep columns covered with gray mask.
    """

    # T0 = most recent timestep (leftmost column), T-1, T-2 ... progressively right
    time_labels = ["T0"] + [f"T-{t}" for t in range(1, seq_len)]

    # ── Reliability ────────────────────────────────────────────────────────
    max_count    = valid_counts.max() if valid_counts.max() > 0 else 1
    reliable_mask = valid_counts >= max(3, max_count * 0.15)

    def row_normalize(arr_1d):
        mn, mx = arr_1d.min(), arr_1d.max()
        return (arr_1d - mn) / (mx - mn + 1e-8)

    # ── Tabular feature group indices (relative to ALL_TABULAR_COLS, excluding PERSONAL_COLS)──
    # ALL_TABULAR_COLS = SVI(8) + ENV(4) + POI(9) + CONTEXT(5) + PERSONAL(4)
    # tab_feat_t shape: (T, 26)  (includes PERSONAL, total 26 dims)
    # Column indices in tab_feat_t for each group
    svi_idx     = list(range(0,  8))   # 8 cols
    env_idx     = list(range(8,  12))  # 4 cols
    poi_idx     = list(range(12, 21))  # 9 cols
    ctx_idx     = list(range(21, 26))  # 5 cols
    # PERSONAL_COLS (idx 26-29) not included in temporal analysis, directly skip

    # Take absolute mean over all timesteps for each group → (T,)
    svi_t   = tab_feat_t[:, svi_idx].mean(axis=1)
    env_t   = tab_feat_t[:, env_idx].mean(axis=1)
    poi_t   = tab_feat_t[:, poi_idx].mean(axis=1)
    ctx_t   = tab_feat_t[:, ctx_idx].mean(axis=1)

    # ── Build heatmap matrix (6 rows × T columns) ───────────────────────────────────
    row_labels = [
        'GPS',
        'RSIs',
        'SVIs',
        'Mobile sensors',
        'POI compositions',
        'Activity contexts',
    ]
    matrix = np.array([
        row_normalize(gps_t),
        row_normalize(sat_t),
        row_normalize(svi_t),
        row_normalize(env_t),
        row_normalize(poi_t),
        row_normalize(ctx_t),
    ])   # (6, T)

    n_rows = len(row_labels)

    # ── Plotting ──────────────────────────────────────────────────────────
    fig_w = max(16, seq_len * 0.55)
    fig_h = 4.5
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    sns.heatmap(matrix, cmap='YlOrRd',
                xticklabels=time_labels,
                yticklabels=row_labels,
                vmin=0, vmax=1,
                annot=False,
                cbar_kws={'label': 'Normalized Importance (0–1)',
                          'shrink': 0.8},
                ax=ax)

    # Gray mask for unreliable timesteps
    for t in range(seq_len):
        if not reliable_mask[t]:
            ax.add_patch(plt.Rectangle((t, 0), 1, n_rows,
                                        color='gray', alpha=0.45, zorder=3))

    ax.set_title(
        'Temporal Importance of Modalities & Feature Groups',
        fontsize=13, fontweight='bold', fontfamily='Times New Roman', pad=10
    )
    ax.set_xlabel('Time Step', fontsize=11, fontfamily='Times New Roman')
    ax.set_ylabel('Modality / Feature Group', fontsize=11,
                  fontfamily='Times New Roman')
    ax.set_xticklabels(ax.get_xticklabels(),
                       rotation=0, ha='center', fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(),
                       rotation=0, fontsize=10, fontfamily='Times New Roman')

    # Unreliable interval legend
    if not reliable_mask.all():
        legend_patch = Patch(facecolor='gray', alpha=0.45,
                             label='Low sample count (unreliable)')
        ax.legend(handles=[legend_patch], fontsize=8,
                  prop={'family': 'Times New Roman'},
                  loc='upper right', bbox_to_anchor=(1.18, 1.02))

    plt.tight_layout()
    out_path = os.path.join(save_dir, "modal_temporal_importance_heatmap.pdf")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print("   💾 Merged heatmap saved: modal_temporal_importance_heatmap.pdf")


# ================= New: GPS SHAP Spatial Map =================

def run_gps_shap_analysis(model, dataset, save_dir, shap_samples=200):
    """
    Use GradientExplainer to calculate SHAP values for GPS (x, y) coordinates,
    and generate spatial risk maps with positive and negative contributions.

    Positive values (red) = location promotes high stress prediction
    Negative values (blue) = location suppresses high stress prediction

    Args:
        model: Loaded model
        dataset: Complete dataset (ConcatDataset or NPZDataset)
        save_dir: Result save directory
        shap_samples: Background sample count
    """
    print("\n" + "=" * 60)
    print("🗺️  Starting GPS SHAP analysis (positive-negative contribution spatial map)")
    print(f"   Sampling {shap_samples} background samples...")
    print(f"   📊 Target variable: Stress category logit value (with sign)")
    print(f"   🎨 Red=promotes high stress, Blue=suppresses high stress")
    print("=" * 60)

    set_mixed_mode(model)

    # ── 1. Sample background data ──────────────────────────────────────────────
    bg_loader = DataLoader(
        dataset, batch_size=shap_samples,
        shuffle=True, collate_fn=custom_collate
    )
    bg_batch = next(iter(bg_loader))

    bg_gps_x   = bg_batch['gps_x'].to(DEVICE).float()    # (N, SEQ_LEN)
    bg_gps_y   = bg_batch['gps_y'].to(DEVICE).float()    # (N, SEQ_LEN)
    bg_sat     = bg_batch['sat_feats'].to(DEVICE)
    bg_tabular = bg_batch['tabular'].to(DEVICE).float()
    bg_mask    = bg_batch['mask'].to(DEVICE)

    # Concatenate x, y → (N, SEQ_LEN, 2), as SHAP perturbation input
    bg_gps_concat = torch.stack([bg_gps_x, bg_gps_y], dim=-1)  # (N, SEQ_LEN, 2)
    print(f"   📐 GPS concat shape: {bg_gps_concat.shape}")

    # ── 2. Define GPS Wrapper ─────────────────────────────────────────
    class GPSModelWrapper(nn.Module):
        """
        Only perturb GPS (x, y) for SHAP, other modalities fixed to background data.
        Input: gps_concat (B, SEQ_LEN, 2)
        Output: Stress category logit (B, 1), with sign preserving non-linear relationships
        """
        def __init__(self, base_model, fixed_sat, fixed_tabular,
                     fixed_mask):
            super().__init__()
            self.model      = base_model
            self.fixed_sat  = fixed_sat
            self.fixed_tab  = fixed_tabular
            self.fixed_mask = fixed_mask

        def _align(self, tensor, curr_bs):
            """Repeat/truncate fixed tensor to align with current batch size"""
            orig = tensor.shape[0]
            if curr_bs == orig:
                return tensor
            if curr_bs > orig and curr_bs % orig == 0:
                return tensor.repeat(curr_bs // orig,
                                     *([1] * (tensor.dim() - 1)))
            return tensor[:curr_bs]

        def forward(self, gps_concat):
            """
            gps_concat: (B, SEQ_LEN, 2) — SHAP perturbed GPS input
            """
            self.model.train()  # Keep train mode to activate gradient path
            B  = gps_concat.shape[0]
            gx = gps_concat[:, :, 0]   # (B, SEQ_LEN) longitude
            gy = gps_concat[:, :, 1]   # (B, SEQ_LEN) latitude

            batch_input = {
                'sat_feats': self._align(self.fixed_sat,  B),
                'tabular':   self._align(self.fixed_tab,  B),
                'mask':      self._align(self.fixed_mask, B),
                'gps_x':     gx,
                'gps_y':     gy,
            }
            logits = self.model(batch_input)
            # Return stress category logit (with sign, preserving non-linearity)
            return logits[:, 1].unsqueeze(1)

    wrapper = GPSModelWrapper(model, bg_sat, bg_tabular, bg_mask)

    # ── 3. Calculate GPS GradientSHAP ────────────────────────────────────
    print("   👉 Computing GPS GradientSHAP (temporarily disabling cuDNN)...")
    print(f"      ⏳ Sample count: {bg_gps_concat.shape[0]}, "
          f"Sequence length: {bg_gps_concat.shape[1]}, Coordinate dimensions: 2 (lon, lat)")

    orig_cudnn = torch.backends.cudnn.enabled
    torch.backends.cudnn.enabled = False

    try:
        explainer   = shap.GradientExplainer(wrapper, bg_gps_concat)
        print("      🔄 Starting SHAP value calculation (one-time calculation, no batching)...")
        shap_values = explainer.shap_values(bg_gps_concat)
        print("      ✅ GPS SHAP value calculation complete!")
    finally:
        torch.backends.cudnn.enabled = orig_cudnn

    # ── 4. Process SHAP return values ─────────────────────────────────────────
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    if hasattr(shap_values, 'detach'):
        shap_values = shap_values.detach().cpu().numpy()
    # Remove output dimension (N, SEQ_LEN, 2, 1) → (N, SEQ_LEN, 2)
    if shap_values.ndim == 4 and shap_values.shape[-1] == 1:
        shap_values = shap_values.squeeze(-1)

    print(f"   ✅ GPS SHAP shape: {shap_values.shape}")
    # shap_values: (N, SEQ_LEN, 2)  → dim 2: [shap_lon, shap_lat]

    # ── 5. Build background sample SHAP lookup table (KD-Tree), then iterate over full dataset ────
    # SHAP calculation only used background samples, but map needs to display all sample GPS points.
    # For each GPS point in all samples, find the nearest coordinate in background at same timestep,
    # use its SHAP value as that point's color.
    bg_gps_x_np = bg_gps_x.cpu().numpy()   # (N_bg, T)
    bg_gps_y_np = bg_gps_y.cpu().numpy()   # (N_bg, T)
    bg_mask_np  = bg_mask.cpu().numpy()    # (N_bg, T)
    N_bg, T = bg_gps_x_np.shape

    # Flatten all valid points in background, build (x, y, t) three-dimensional KD-Tree
    bg_records = []
    for i in range(N_bg):
        for t in range(T):
            if bg_mask_np[i, t] == 1 and abs(bg_gps_x_np[i, t]) > 1e-6:
                bg_records.append({
                    'x': bg_gps_x_np[i, t],
                    'y': bg_gps_y_np[i, t],
                    't': t,
                    'shap_lon':   float(shap_values[i, t, 0]),
                    'shap_lat':   float(shap_values[i, t, 1]),
                    'shap_total': float(shap_values[i, t, 0]) + float(shap_values[i, t, 1]),
                })
    bg_df   = pd.DataFrame(bg_records)
    # KD-Tree uses normalized coordinates + timestep (coordinate unit meters, t needs scaling to similar magnitude)
    xy_scale = max(bg_df['x'].std(), bg_df['y'].std()) + 1e-8
    t_scale  = xy_scale   # Make timestep weight comparable to spatial
    tree = cKDTree(np.column_stack([
        bg_df['x'].values / xy_scale,
        bg_df['y'].values / xy_scale,
        bg_df['t'].values * (xy_scale / t_scale),
    ]))
    print(f"   📐 Background valid point count: {len(bg_df):,}, KD-Tree construction complete")

    # Iterate over full dataset, collect all GPS points
    print("   🔄 Iterating over full dataset to collect GPS points...")
    full_loader = DataLoader(dataset, batch_size=BATCH_SIZE,
                             shuffle=False, collate_fn=custom_collate)
    records = []
    for batch in tqdm(full_loader, desc="      Collecting GPS points", leave=False):
        gx_np  = batch['gps_x'].numpy()    # (B, T)
        gy_np  = batch['gps_y'].numpy()    # (B, T)
        msk_np = batch['mask'].numpy()     # (B, T)
        B = gx_np.shape[0]
        for i in range(B):
            for t in range(T):
                if msk_np[i, t] == 1 and abs(gx_np[i, t]) > 1e-6:
                    # Find nearest background point
                    q = [gx_np[i, t] / xy_scale,
                         gy_np[i, t] / xy_scale,
                         t * (xy_scale / t_scale)]
                    _, idx = tree.query(q, k=1)
                    records.append({
                        'lon':        gx_np[i, t],
                        'lat':        gy_np[i, t],
                        'shap_lon':   bg_df.iloc[idx]['shap_lon'],
                        'shap_lat':   bg_df.iloc[idx]['shap_lat'],
                        'shap_total': bg_df.iloc[idx]['shap_total'],
                    })
    print(f"   ✅ Full GPS point collection complete, total {len(records):,} records")

    if not records:
        print("   ⚠️ No valid GPS records, skipping GPS SHAP map")
        return

    df = pd.DataFrame(records)

    # Aggregate by location (take mean)
    df['lon_r'] = df['lon'].round(5)
    df['lat_r'] = df['lat'].round(5)
    agg = df.groupby(['lon_r', 'lat_r']).agg(
        shap_total=('shap_total', 'mean'),
        shap_lon=('shap_lon',   'mean'),
        shap_lat=('shap_lat',   'mean'),
        count=('shap_total',    'count')
    ).reset_index()
    agg.rename(columns={'lon_r': 'lon', 'lat_r': 'lat'}, inplace=True)

    print(f"   Unique location count: {len(agg):,}, Total record count: {len(df):,}")
    print(f"   SHAP range: [{agg['shap_total'].min():.4f}, {agg['shap_total'].max():.4f}]")
    print(f"   Positive contribution locations (stress promotion): {(agg['shap_total'] > 0).sum():,}")
    print(f"   Negative contribution locations (stress suppression): {(agg['shap_total'] < 0).sum():,}")

    # Save raw data
    csv_path = os.path.join(save_dir, "gps_shap_data.csv")
    agg.to_csv(csv_path, index=False)
    print(f"   💾 GPS SHAP data saved: gps_shap_data.csv (projected coordinates EPSG:8857)")

    # ── Inverse projection: EPSG:8857 → WGS84 (EPSG:4326), save second CSV ──────────
    try:
        inv_proj = ProjTransformer.from_crs("EPSG:8857", "EPSG:4326", always_xy=True)
        wgs_lon, wgs_lat = inv_proj.transform(agg['lon'].values, agg['lat'].values)
        agg_wgs = agg.copy()
        agg_wgs['lon_wgs84'] = wgs_lon
        agg_wgs['lat_wgs84'] = wgs_lat
        wgs_csv = os.path.join(save_dir, "gps_shap_wgs84.csv")
        agg_wgs[['lon_wgs84', 'lat_wgs84', 'shap_lon', 'shap_lat',
                  'shap_total', 'count']].to_csv(wgs_csv, index=False)
        print(f"   💾 GPS SHAP WGS84 data saved: gps_shap_wgs84.csv")
        print(f"      Longitude range: {wgs_lon.min():.4f} ~ {wgs_lon.max():.4f}")
        print(f"      Latitude range: {wgs_lat.min():.4f} ~ {wgs_lat.max():.4f}")
    except ImportError:
        print("   ⚠️ pyproj not installed, skipping inverse projection CSV")
        agg_wgs = None

    # ── 6. Plotting ──────────────────────────────────────────────────────
    _plot_gps_shap_map(agg, agg_wgs, save_dir)

    print("✅ GPS SHAP analysis complete!\n")


def _plot_gps_shap_map(agg_df, agg_wgs, save_dir):
    """
    Plot GPS SHAP spatial map.
    - gps_shap_map.png        : Projected coordinates, no basemap (offline available)
    - gps_shap_map_osm.png    : WGS84 coordinates + OSM basemap (requires contextily + network)
    Colormap RdYlBu_r: Deep blue(strong negative) → Yellow(zero) → Deep red(strong positive)
    """

    sat_cmap = plt.cm.RdYlBu_r

    def symmetric_norm(series):
        vmax = np.abs(series).quantile(0.98) + 1e-8
        return mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    def _decorate_ax(ax, title, xlabel, ylabel):
        ax.set_title(title, fontsize=13, fontweight='bold',
                     fontfamily='Times New Roman')
        ax.set_xlabel(xlabel, fontsize=12, fontfamily='Times New Roman')
        ax.set_ylabel(ylabel, fontsize=12, fontfamily='Times New Roman')
        ax.tick_params(labelsize=10)
        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_fontfamily('Times New Roman')

    dot_size = max(2, min(20, 3000 // len(agg_df)))
    norm     = symmetric_norm(agg_df['shap_total'])

    # ── Figure 1: Projected coordinates, no basemap ──────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 7))
    sc = ax.scatter(agg_df['lon'], agg_df['lat'],
                    c=agg_df['shap_total'], cmap=sat_cmap, norm=norm,
                    s=dot_size, alpha=0.85, linewidths=0)
    cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('SHAP Value', fontsize=11, fontfamily='Times New Roman')
    cbar.ax.tick_params(labelsize=9)
    ax.grid(True, alpha=0.20, linestyle='--')
    _decorate_ax(ax,
        "GPS Location SHAP Values (Combined)\nRed = Stress-Promoting  |  Blue = Stress-Suppressing",
        'X (EPSG:8857, m)', 'Y (EPSG:8857, m)')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "gps_shap_map.pdf"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   💾 GPS SHAP map saved: gps_shap_map.png")

    # ── Figure 2: WGS84 + OSM basemap ─────────────────────────────────────────
    if agg_wgs is None:
        print("   ⚠️ No inverse projection data, skipping OSM basemap version")
        return
    if not HAS_CONTEXTILY:
        print("   ⚠️ OSM basemap requires contextily + geopandas: pip install contextily geopandas")
        return
    try:
        norm_wgs = symmetric_norm(agg_wgs['shap_total'])
        dot_size_wgs = max(2, min(20, 3000 // len(agg_wgs)))

        gdf = gpd.GeoDataFrame(
            agg_wgs,
            geometry=[Point(x, y) for x, y in
                      zip(agg_wgs['lon_wgs84'], agg_wgs['lat_wgs84'])],
            crs='EPSG:4326'
        ).to_crs(epsg=3857)   # Web Mercator for contextily

        fig, ax = plt.subplots(figsize=(10, 8))
        sc = ax.scatter(gdf.geometry.x, gdf.geometry.y,
                        c=agg_wgs['shap_total'], cmap=sat_cmap, norm=norm_wgs,
                        s=dot_size_wgs, alpha=0.80, linewidths=0, zorder=3)
        CARTO_LIGHT = ("https://basemaps.cartocdn.com/light_nolabels"
                        "/{z}/{x}/{y}.png")
        ctx.add_basemap(ax, source=CARTO_LIGHT, zoom='auto', alpha=0.9)
        cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('SHAP Value', fontsize=11, fontfamily='Times New Roman')
        cbar.ax.tick_params(labelsize=9)
        _decorate_ax(ax,
            "GPS Location SHAP Values (Combined) — OSM Basemap\nRed = Stress-Promoting  |  Blue = Stress-Suppressing",
            'Longitude', 'Latitude')
        ax.set_axis_off()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "gps_shap_map_osm.pdf"),
                    dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   💾 GPS SHAP OSM map saved: gps_shap_map_osm.png")
    except Exception as e:
        print(f"   ⚠️ OSM basemap generation failed: {e}")

    # ── Contribution distribution histogram ───────────────────────────────────────────────
    pos_pct = (agg_df['shap_total'] > 0).mean() * 100
    neg_pct = 100 - pos_pct

    fig, ax = plt.subplots(figsize=(8, 5))
    n, bins, patches = ax.hist(
        agg_df['shap_total'], bins=60,
        edgecolor='white', linewidth=0.3, alpha=0.85
    )
    # Color positive values red, negative values blue
    for patch, left in zip(patches, bins[:-1]):
        patch.set_facecolor('#c0392b' if left >= 0 else '#2980b9')

    ax.axvline(0, color='black', linewidth=1.8, linestyle='--',
               label='Zero (neutral)')
    ax.set_title(
        f"Distribution of GPS SHAP Values (Combined)\n"
        f"Stress-promoting: {pos_pct:.1f}%  |  "
        f"Stress-suppressing: {neg_pct:.1f}%",
        fontsize=12, fontweight='bold', fontfamily='Times New Roman'
    )
    ax.set_xlabel('SHAP Value (Longitude + Latitude)',
                  fontsize=11, fontfamily='Times New Roman')
    ax.set_ylabel('Count', fontsize=11, fontfamily='Times New Roman')
    ax.legend(fontsize=10, prop={'family': 'Times New Roman'})
    ax.tick_params(labelsize=9)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontfamily('Times New Roman')
    ax.grid(True, alpha=0.28, linestyle='--')
    plt.tight_layout()
    dist_path = os.path.join(save_dir, "gps_shap_distribution.png")
    plt.savefig(dist_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   💾 GPS SHAP distribution plot saved: gps_shap_distribution.png")


# ================= 5. Main Program =================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 80)
    print(f"🚀 Starting full analysis (v11 fixed version: Remove SHAP batch calculation + Feature value inversion)")
    print(f"   Output directory: {OUTPUT_DIR}")
    print(f"   Sequence length: {SEQ_LEN}")

    if SPECIFIC_TIME_STEP is not None:
        print(f"   🎯 Timestep mode: Analyzing only T={SPECIFIC_TIME_STEP}")
    else:
        print(f"   🎯 Timestep mode: All timesteps flattened and mixed")

    print("=" * 80)

    # 0. 🔥 New: Load scalers
    print("\n🔧 Loading scalers...")
    scaler_info = load_scalers(DATA_DIR)

    # 1. Load data
    print("\n📂 Reading data...")
    datasets = []
    val_dataset = None  # New: Save validation set separately

    for split in ['train', 'val']:
        path = os.path.join(DATA_DIR, f"{split}.npz")
        if os.path.exists(path):
            ds = NPZDataset(path, feature_config=FEATURE_CONFIG)
            datasets.append(ds)
            print(f"   ✅ {split}.npz: {len(ds)} samples")
            if split == 'val':
                val_dataset = ds  # New: Record validation set

    full_dataset = ConcatDataset(datasets)
    loader = DataLoader(full_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=custom_collate)
    print(f"   Total sample count: {len(full_dataset)}")

    # 2. Load model
    print(f"\n🛠 Loading model: {MODEL_PATH}")
    model = FullStackPredictor(
        tabular_dim=30, num_classes=NUM_CLASSES,
        adapter_type=ADAPTER_TYPE, temporal_model=TEMPORAL_MODEL,
        ablation_config=ABLATION_CONFIG, feature_config=FEATURE_CONFIG,
        adapter_output_dim=ADAPTER_OUTPUT_DIM, adapter_patch_dim=ADAPTER_PATCH_DIM,
        adapter_cls_dim=ADAPTER_CLS_DIM, use_cls_in_adapter=USE_CLS_IN_ADAPTER,
        cross_modal_fusion=CROSS_MODAL_FUSION,
        tabular_output_dim=TABULAR_OUTPUT_DIM, tabular_processing=TABULAR_PROCESSING,
        center_aware=CENTER_AWARE, center_weight_init=CENTER_WEIGHT_INIT
    ).to(DEVICE)

    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    print("   ✅ Model loaded successfully")

    # New: 2.5 Validation set evaluation (complete quick evaluation before SHAP)
    if val_dataset is not None:
        evaluate_on_val(model, val_dataset, OUTPUT_DIR)
    else:
        print("⚠️ val.npz not found, skipping validation set evaluation")

    # 3. Run SHAP (🔥 Pass scaler_info)
    run_shap_analysis(model, full_dataset, ALL_TABULAR_COLS, OUTPUT_DIR,
                      specific_timestep=SPECIFIC_TIME_STEP, scaler_info=scaler_info)

    # 4. Modal-temporal SHAP analysis
    #    For GPS (combined x+y), Satellite (patch mean), Tabular (each feature separately)
    #    Use GradientExplainer to calculate SHAP importance at each timestep
    run_modal_temporal_shap(model, full_dataset, OUTPUT_DIR,
                            seq_len=SEQ_LEN, shap_samples=SHAP_SAMPLES)

    # 5. GPS SHAP spatial map (replaces original gradient-based GPS risk map)
    #    Use GradientExplainer to calculate GPS (x, y) SHAP contribution to stress prediction,
    #    Positive (red) = promotes high stress, Negative (blue) = suppresses high stress
    run_gps_shap_analysis(model, full_dataset, OUTPUT_DIR, shap_samples=SHAP_SAMPLES)

    # Batch Patch SHAP heatmaps (randomly sample PATCH_SHAP_SAMPLES sequences, T=0 timestep, overlay original images)
    # Use GradientExplainer (Expected Gradients / true SHAP), replacing original IG implementation
    generate_patch_shap_maps(model, full_dataset, OUTPUT_DIR, SAT_DIR,
                             seq_len=SEQ_LEN, n_samples=PATCH_SHAP_SAMPLES)

    print("\n" + "=" * 80)
    print(f"✅ All analyses complete! Results saved in: {OUTPUT_DIR}")
    if SPECIFIC_TIME_STEP is not None:
        print(f"   📊 SHAP analysis mode: Only timestep T={SPECIFIC_TIME_STEP}")
    else:
        print(f"   📊 SHAP analysis mode: All timesteps flattened")
    print("=" * 80)


if __name__ == "__main__":
    # Global font settings: Times New Roman (for SHAP plots), Chinese fallback font retained
    plt.rcParams['font.family']       = 'serif'
    plt.rcParams['font.serif']        = ['Times New Roman', 'DejaVu Serif', 'serif']
    plt.rcParams['font.sans-serif']   = ['SimHei', 'Arial', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['axes.titlesize']    = 12
    plt.rcParams['axes.labelsize']    = 12
    plt.rcParams['xtick.labelsize']   = 10
    plt.rcParams['ytick.labelsize']   = 10
    plt.rcParams['legend.fontsize']   = 11
    main()