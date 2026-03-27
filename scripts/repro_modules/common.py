from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import shutil
import warnings
import zipfile
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Iterable
from xml.etree import ElementTree as ET

import fitz
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from matplotlib import transforms
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from PIL import Image  # noqa: F401
from scipy.stats import gaussian_kde
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, ParameterGrid
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor

warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn.impute._iterative")


# --- constants.py ---

ROOT = Path(__file__).resolve().parents[2]

PRIMARY_DATA_XLSX = ROOT / "11.xlsx"

OUTPUT_DIR = ROOT / "outputs"
DEFAULT_SPLIT_SEED = 10

RAW_COLUMN_MAP = {
    "Coordination metal": "metal",
    "Modification method": "modification",
    "Surface areas(m2/g)": "sa",
    "Reporting pore diameter(nm)\n": "pd",
    "pore volume(cm3/g)": "pv",
    "Initial concentration(mg/L)": "ci",
    "Adsorbent dosage(mg/L)\n": "ad",
    "time(min)": "time",
    "pH": "ph",
    "T(K)": "temp",
    "Adsorption \ncapacities (mg/g)": "q",
    "DOI": "doi",
}

DESCRIPTOR_PRESETS = {
    "neutral_atomic": {
        "Ag": {"ionic_charge": 1.0, "atomic_radius": 160.0, "polarizability": 8.15, "electronegativity": 1.93},
        "Co": {"ionic_charge": 2.0, "atomic_radius": 135.0, "polarizability": 8.15, "electronegativity": 1.88},
        "Cr": {"ionic_charge": 3.0, "atomic_radius": 140.0, "polarizability": 12.30, "electronegativity": 1.66},
        "Cu": {"ionic_charge": 2.0, "atomic_radius": 135.0, "polarizability": 6.89, "electronegativity": 1.90},
        "Fe": {"ionic_charge": 3.0, "atomic_radius": 140.0, "polarizability": 9.19, "electronegativity": 1.83},
        "In": {"ionic_charge": 3.0, "atomic_radius": 155.0, "polarizability": 7.60, "electronegativity": 1.78},
        "Nd": {"ionic_charge": 3.0, "atomic_radius": 182.0, "polarizability": 27.00, "electronegativity": 1.14},
        "Ni": {"ionic_charge": 2.0, "atomic_radius": 135.0, "polarizability": 6.8, "electronegativity": 1.91},
        "Sm": {"ionic_charge": 3.0, "atomic_radius": 181.0, "polarizability": 24.70, "electronegativity": 1.17},
        "Ti": {"ionic_charge": 4.0, "atomic_radius": 147.0, "polarizability": 10.60, "electronegativity": 1.54},
        "Zn": {"ionic_charge": 2.0, "atomic_radius": 135.0, "polarizability": 5.73, "electronegativity": 1.65},
        "Zr": {"ionic_charge": 4.0, "atomic_radius": 155.0, "polarizability": 16.60, "electronegativity": 1.33},
    },
    "calibrated_mixed": {
        "Ag": {"ionic_charge": 1.0, "atomic_radius": 165.0, "polarizability": 6.8, "electronegativity": 1.93},
        "Co": {"ionic_charge": 2.0, "atomic_radius": 125.0, "polarizability": 1.7, "electronegativity": 1.88},
        "Cr": {"ionic_charge": 3.0, "atomic_radius": 128.0, "polarizability": 1.1, "electronegativity": 1.66},
        "Cu": {"ionic_charge": 2.0, "atomic_radius": 128.0, "polarizability": 6.1, "electronegativity": 1.90},
        "Fe": {"ionic_charge": 3.0, "atomic_radius": 126.0, "polarizability": 8.4, "electronegativity": 1.83},
        "In": {"ionic_charge": 3.0, "atomic_radius": 150.0, "polarizability": 7.5, "electronegativity": 1.78},
        "Nd": {"ionic_charge": 3.0, "atomic_radius": 181.0, "polarizability": 24.0, "electronegativity": 1.14},
        "Ni": {"ionic_charge": 2.0, "atomic_radius": 125.0, "polarizability": 6.7, "electronegativity": 1.91},
        "Sm": {"ionic_charge": 3.0, "atomic_radius": 180.0, "polarizability": 23.0, "electronegativity": 1.17},
        "Ti": {"ionic_charge": 4.0, "atomic_radius": 140.0, "polarizability": 10.0, "electronegativity": 1.54},
        "Zn": {"ionic_charge": 2.0, "atomic_radius": 134.0, "polarizability": 5.8, "electronegativity": 1.65},
        "Zr": {"ionic_charge": 4.0, "atomic_radius": 160.0, "polarizability": 11.1, "electronegativity": 1.33},
    },
    "ionic_radius": {
        "Ag": {"ionic_charge": 1.0, "atomic_radius": 129.0, "polarizability": 8.15, "electronegativity": 1.93},
        "Co": {"ionic_charge": 2.0, "atomic_radius": 74.5, "polarizability": 8.15, "electronegativity": 1.88},
        "Cr": {"ionic_charge": 3.0, "atomic_radius": 61.5, "polarizability": 12.30, "electronegativity": 1.66},
        "Cu": {"ionic_charge": 2.0, "atomic_radius": 73.0, "polarizability": 6.89, "electronegativity": 1.90},
        "Fe": {"ionic_charge": 3.0, "atomic_radius": 64.5, "polarizability": 9.19, "electronegativity": 1.83},
        "In": {"ionic_charge": 3.0, "atomic_radius": 80.0, "polarizability": 7.60, "electronegativity": 1.78},
        "Nd": {"ionic_charge": 3.0, "atomic_radius": 98.3, "polarizability": 27.00, "electronegativity": 1.14},
        "Ni": {"ionic_charge": 2.0, "atomic_radius": 69.0, "polarizability": 6.8, "electronegativity": 1.91},
        "Sm": {"ionic_charge": 3.0, "atomic_radius": 95.8, "polarizability": 24.70, "electronegativity": 1.17},
        "Ti": {"ionic_charge": 4.0, "atomic_radius": 74.5, "polarizability": 10.60, "electronegativity": 1.54},
        "Zn": {"ionic_charge": 2.0, "atomic_radius": 74.0, "polarizability": 5.73, "electronegativity": 1.65},
        "Zr": {"ionic_charge": 4.0, "atomic_radius": 72.0, "polarizability": 16.60, "electronegativity": 1.33},
    },
}

ATOMIC_WEIGHTS = {
    "Ag": 107.8682,
    "Al": 26.9815385,
    "As": 74.921595,
    "Au": 196.96657,
    "B": 10.81,
    "Bi": 208.9804,
    "Br": 79.904,
    "C": 12.011,
    "Ca": 40.078,
    "Cd": 112.414,
    "Ce": 140.116,
    "Cl": 35.45,
    "Co": 58.933194,
    "Cr": 51.9961,
    "Cs": 132.90545196,
    "Cu": 63.546,
    "Dy": 162.5,
    "Er": 167.259,
    "Eu": 151.964,
    "F": 18.998403163,
    "Fe": 55.845,
    "Ga": 69.723,
    "Gd": 157.25,
    "Ge": 72.63,
    "H": 1.008,
    "Hf": 178.49,
    "Hg": 200.592,
    "Ho": 164.93033,
    "I": 126.90447,
    "In": 114.818,
    "Ir": 192.217,
    "K": 39.0983,
    "La": 138.90547,
    "Li": 6.94,
    "Lu": 174.9668,
    "Mg": 24.305,
    "Mn": 54.938044,
    "Mo": 95.95,
    "N": 14.007,
    "Na": 22.98976928,
    "Nb": 92.90637,
    "Nd": 144.242,
    "Ni": 58.6934,
    "Np": 237.0,
    "O": 15.999,
    "P": 30.973761998,
    "Pb": 207.2,
    "Pd": 106.42,
    "Pr": 140.90766,
    "Pt": 195.084,
    "Pu": 244.0,
    "Rb": 85.4678,
    "Rh": 102.9055,
    "Ru": 101.07,
    "S": 32.06,
    "Sb": 121.76,
    "Sc": 44.955908,
    "Se": 78.971,
    "Si": 28.085,
    "Sm": 150.36,
    "Tb": 158.92535,
    "Th": 232.0377,
    "Ti": 47.867,
    "Tm": 168.93422,
    "U": 238.02891,
    "V": 50.9415,
    "W": 183.84,
    "Y": 88.90584,
    "Yb": 173.045,
    "Zn": 65.38,
    "Zr": 91.224,
}

MOD_ENCODING_PRESETS = {
    "default": {
        "Unmodified": 0,
        "Functionalized": 1,
        "Carbonized": 2,
        "Magnetic": 3,
        "Composite": 4,
    },
    "calibrated": {
        "Carbonized": 2,
        "Composite": 3,
        "Functionalized": 1,
        "Magnetic": 4,
        "Unmodified": 0,
    },
}

BASE_FEATURES = [
    "ionic_charge",
    "atomic_radius",
    "polarizability",
    "electronegativity",
    "sa",
    "mpd",
    "pd",
    "pv",
    "ci",
    "ad",
    "time",
    "ph",
    "temp",
]

TRAINING_FEATURES = [*BASE_FEATURES, "mod_code"]

STRUCTURAL_IMPUTE_TARGETS = ["pv", "pd"]

STRUCTURAL_IMPUTE_CONTEXT = [
    "sa",
    "mpd",
    "ci",
    "ad",
    "time",
    "ph",
    "temp",
    "mod_code",
    "ionic_charge",
    "atomic_radius",
    "polarizability",
    "electronegativity",
]

MOD_LABELS = {
    "Unmodified": "UN",
    "Functionalized": "FU",
    "Magnetic": "MA",
    "Carbonized": "CA",
    "Composite": "CO",
}

METAL_ORDER = ["Fe", "Cr", "Zr", "Zn", "Co", "Cu", "Ag"]

MOD_ORDER = ["Unmodified", "Functionalized", "Magnetic", "Carbonized", "Composite"]
SCREENING_MOD_STRATEGY = "training_distribution_marginalized"

TARGET_CORE_METALS = ["Zn", "Cu", "Zr", "Cr", "Nd", "Sm", "Ni"]

MODEL_ORDER = ["RF", "GBDT", "XGB", "LR", "KNN", "SVR"]
GROUP_AWARE_TUNING_GROUP_THRESHOLD = 20

PAPER_FEATURE_LABELS = {
    "ionic_charge": "IC",
    "atomic_radius": "AR",
    "polarizability": "Pol",
    "electronegativity": "Ele",
    "sa": "SA",
    "mpd": "MPD",
    "pd": "PD",
    "pv": "PV",
    "ci": "CI",
    "ad": "AD",
    "time": "Time",
    "ph": "pH",
    "temp": "Tem",
    "mod_code": "ModM",
}

FIG2_GREEN = "#42b97a"
S1_PANEL_COLORS = {
    "sa": "#e39aac",
    "mpd": "#d2a6d8",
    "pd": "#c7bff2",
    "pv": "#aebee8",
    "ci": "#e39aac",
    "ad": "#d2a6d8",
    "time": "#c7bff2",
    "ph": "#aebee8",
    "temp": "#aebee8",
}
S2_METAL_COLORS = {
    "Fe": "#d98d9c",
    "Cr": "#c6a04c",
    "Zr": "#8fae4d",
    "Zn": "#51b39f",
    "Co": "#58aeb8",
    "Cu": "#9ea7dd",
    "Ag": "#d79ad2",
}
S2_MOD_COLORS = {
    "UN": "#d9788b",
    "FU": "#b49a47",
    "CA": "#63a84b",
    "MA": "#4da29d",
    "CO": "#5b98c8",
}
S2_TEMP_COLORS = {
    293: "#d98d9c",
    298: "#d0954d",
    303: "#b2aa52",
    313: "#58aeb8",
    323: "#58aeb8",
}

@dataclass
class SplitConfig:
    name: str
    mode: str
    descriptor_preset: str
    mod_encoding: str
    group_recipe: str = "paper_43"
    train_group_count: int | None = None
    random_state: int | None = None

@dataclass
class SplitBundle:
    train_idx: object
    test_idx: object
    train_group_count: int
    test_group_count: int


# --- style.py ---

def set_paper_rcparams() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.8,
        }
    )

def style_small_axis(ax: plt.Axes) -> None:
    ax.tick_params(direction="in", length=3, width=0.8)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)

def add_caption(fig: plt.Figure, text: str, y: float = 0.02) -> None:
    fig.text(0.5, y, text, ha="center", va="bottom", fontsize=8)


# --- core.py ---

def positive_reference_floor(values: pd.Series, quantile: float = 0.01, default: float = 1e-6) -> float:
    numeric = pd.to_numeric(values, errors="coerce")
    numeric = numeric[numeric > 0]
    if numeric.empty:
        return float(default)
    return max(float(numeric.quantile(quantile)), float(default))


def mode_or_nan(values: pd.Series) -> float:
    modes = pd.to_numeric(values, errors="coerce").dropna().mode()
    if modes.empty:
        return float("nan")
    return float(modes.iloc[0])


def fill_with_stratified_median(
    out: pd.DataFrame,
    fit_df: pd.DataFrame,
    column: str,
    strata: list[list[str]],
) -> pd.Series:
    filled = pd.to_numeric(out[column], errors="coerce").copy()
    fit_numeric = pd.to_numeric(fit_df[column], errors="coerce")
    fit_frame = fit_df.copy()
    fit_frame[column] = fit_numeric

    for keys in strata:
        missing_mask = filled.isna()
        if not missing_mask.any():
            break
        stats = fit_frame.groupby(keys, dropna=False)[column].median()
        mapped = (
            out.loc[missing_mask, keys]
            .apply(lambda row: stats.get(tuple(row.values) if len(keys) > 1 else row.iloc[0], np.nan), axis=1)
            .astype(float)
        )
        filled.loc[missing_mask] = mapped.to_numpy()

    if filled.isna().any():
        global_median = float(fit_numeric.median()) if fit_numeric.notna().any() else float("nan")
        global_mode = mode_or_nan(fit_numeric)
        fallback = global_median if np.isfinite(global_median) else global_mode
        if not np.isfinite(fallback):
            fallback = 1.0
        filled = filled.fillna(float(fallback))
    return filled


def fill_experimental_conditions(out: pd.DataFrame, fit_df: pd.DataFrame) -> pd.DataFrame:
    """Fill CI/AD with stratified medians and positive lower bounds."""
    result = out.copy()
    strata = [["metal", "modification"], ["metal"], ["modification"]]
    for column in ["ci", "ad"]:
        result[column] = fill_with_stratified_median(result, fit_df, column, strata)
        floor = positive_reference_floor(fit_df[column])
        result[column] = pd.to_numeric(result[column], errors="coerce").clip(lower=floor)
    return result


def fit_rf_structural_imputer(
    fit_df: pd.DataFrame,
    target: str,
    predictors: list[str],
) -> tuple[RandomForestRegressor | None, dict[str, float], float]:
    train = fit_df.copy()
    train[target] = pd.to_numeric(train[target], errors="coerce")
    predictor_frame = train[predictors].apply(pd.to_numeric, errors="coerce")
    predictor_fill = predictor_frame.median(numeric_only=True).to_dict()
    target_floor = positive_reference_floor(train[target])
    valid_mask = train[target].notna()
    if int(valid_mask.sum()) < 5:
        return None, predictor_fill, target_floor
    X_train = predictor_frame.loc[valid_mask].fillna(predictor_fill)
    y_train = train.loc[valid_mask, target].to_numpy(dtype=float)
    model = RandomForestRegressor(
        n_estimators=300,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model, predictor_fill, target_floor


def warn_if_imputed_distribution_anomalous(
    target: str,
    imputed_values: pd.Series,
    reference_values: pd.Series,
    floor: float,
) -> None:
    imputed = pd.to_numeric(imputed_values, errors="coerce").dropna()
    reference = pd.to_numeric(reference_values, errors="coerce").dropna()
    if imputed.empty or reference.empty:
        return
    ref_low, ref_high = reference.quantile([0.01, 0.99]).tolist()
    imp_median = float(imputed.median())
    floor_share = float((imputed <= floor * 1.000001).mean())
    if imp_median < ref_low or imp_median > ref_high or floor_share > 0.25:
        warnings.warn(
            (
                f"Imputed {target} distribution may be abnormal: "
                f"median={imp_median:.4g}, reference 1%-99%=[{ref_low:.4g}, {ref_high:.4g}], "
                f"floor_share={floor_share:.1%}"
            ),
            RuntimeWarning,
            stacklevel=2,
        )


def impute_structural_feature(
    out: pd.DataFrame,
    fit_df: pd.DataFrame,
    target: str,
    predictors: list[str],
) -> pd.DataFrame:
    result = out.copy()
    result[target] = pd.to_numeric(result[target], errors="coerce")
    model, predictor_fill, floor = fit_rf_structural_imputer(fit_df, target, predictors)
    missing_mask = result[target].isna()
    if not missing_mask.any():
        result[target] = result[target].clip(lower=floor)
        return result
    if model is None:
        fallback = float(pd.to_numeric(fit_df[target], errors="coerce").median())
        if not np.isfinite(fallback):
            fallback = floor
        result.loc[missing_mask, target] = fallback
        result[target] = pd.to_numeric(result[target], errors="coerce").clip(lower=floor)
        warn_if_imputed_distribution_anomalous(target, result.loc[missing_mask, target], fit_df[target], floor)
        return result
    X_missing = result.loc[missing_mask, predictors].apply(pd.to_numeric, errors="coerce").fillna(predictor_fill)
    predicted = model.predict(X_missing)
    result.loc[missing_mask, target] = np.maximum(predicted, floor)
    result[target] = pd.to_numeric(result[target], errors="coerce").clip(lower=floor)
    warn_if_imputed_distribution_anomalous(target, result.loc[missing_mask, target], fit_df[target], floor)
    return result

def sanitize_physical_features(df: pd.DataFrame, fit_df: pd.DataFrame) -> pd.DataFrame:
    """Clip physically invalid values after imputation using train-side limits only."""
    out = df.copy()
    positive_sa = pd.to_numeric(fit_df["sa"], errors="coerce")
    positive_sa = positive_sa[positive_sa > 0]
    sa_floor = float(positive_sa.quantile(0.01)) if not positive_sa.empty else 1e-6
    out["sa"] = pd.to_numeric(out["sa"], errors="coerce").clip(lower=max(sa_floor, 1e-6))
    for col in ["pd", "pv", "ci", "ad"]:
        floor = positive_reference_floor(fit_df[col])
        out[col] = pd.to_numeric(out[col], errors="coerce").clip(lower=floor)
    return out

def sanitize_candidate_core_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply only minimal physical cleanup for CoRE candidates.

    The candidate library should keep its native structural spread instead of
    being pulled up to train-side lower quantiles, otherwise Fig. 5 collapses
    many candidates onto the same lower-bound SA value.
    """
    out = df.copy()
    out["sa"] = pd.to_numeric(out["sa"], errors="coerce").clip(lower=1e-6)
    out["pd"] = pd.to_numeric(out["pd"], errors="coerce").clip(lower=1e-6)
    out["pv"] = pd.to_numeric(out["pv"], errors="coerce").clip(lower=1e-6)
    out["ci"] = pd.to_numeric(out["ci"], errors="coerce").clip(lower=1e-6)
    out["ad"] = pd.to_numeric(out["ad"], errors="coerce").clip(lower=1e-6)
    out["vf"] = pd.to_numeric(out["vf"], errors="coerce").clip(lower=0.0, upper=1.0)
    return out

def derive_mpd(pv: pd.Series, sa: pd.Series) -> pd.Series:
    """Back-fill theoretical MPD from accessible PV and SA as described in the paper."""
    safe_sa = pd.to_numeric(sa, errors="coerce").replace(0, np.nan)
    mpd = 4000.0 * pd.to_numeric(pv, errors="coerce") / safe_sa
    return mpd.replace([np.inf, -np.inf], np.nan)

def resolve_data_xlsx() -> Path:
    if PRIMARY_DATA_XLSX.exists():
        return PRIMARY_DATA_XLSX
    raise FileNotFoundError(f"Training workbook not found: {PRIMARY_DATA_XLSX.name}")

def encode_modification_codes(modifications: pd.Series, mod_encoding: str) -> tuple[pd.Series, dict[str, int], list[str]]:
    preset_mapping = MOD_ENCODING_PRESETS[mod_encoding].copy()
    normalized = (
        modifications.fillna("Unknown")
        .astype(str)
        .str.strip()
        .replace("", "Unknown")
    )
    codes = normalized.map(preset_mapping)
    unknown_values = sorted(normalized.loc[codes.isna()].unique().tolist())
    if unknown_values:
        next_code = max(preset_mapping.values(), default=-1) + 1
        for value in unknown_values:
            preset_mapping[value] = next_code
            next_code += 1
        warnings.warn(
            f"Unknown modification categories encountered and dynamically encoded: {', '.join(unknown_values)}",
            RuntimeWarning,
            stacklevel=2,
        )
        codes = normalized.map(preset_mapping)
    return codes.astype(int), preset_mapping, unknown_values

def get_mod_plot_label(name: str) -> str:
    if name in MOD_LABELS:
        return MOD_LABELS[name]
    alnum = re.sub(r"[^A-Za-z0-9]+", "", str(name))
    if not alnum:
        return "OT"
    return alnum[:4].upper()

def prepare_model_table(df: pd.DataFrame, fit_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Apply train-only preprocessing with stratified CI/AD fill and RF structural imputation for PD/PV."""
    fit_df = df if fit_df is None else fit_df
    out = df.copy()
    fit_numeric = fit_df.copy()
    out["ph"] = pd.to_numeric(out["ph"], errors="coerce").fillna(7.0)
    out["temp"] = pd.to_numeric(out["temp"], errors="coerce").fillna(298.0)
    out["time"] = pd.to_numeric(out["time"], errors="coerce")
    out["sa"] = pd.to_numeric(out["sa"], errors="coerce")
    fit_numeric["time"] = pd.to_numeric(fit_numeric["time"], errors="coerce")
    fit_numeric["sa"] = pd.to_numeric(fit_numeric["sa"], errors="coerce")
    if out["time"].isna().any():
        time_fill = float(pd.to_numeric(fit_numeric["time"], errors="coerce").median())
        if not np.isfinite(time_fill):
            time_fill = 0.0
        out["time"] = out["time"].fillna(time_fill)
    if out["sa"].isna().any():
        sa_fill = float(pd.to_numeric(fit_numeric["sa"], errors="coerce").median())
        if not np.isfinite(sa_fill):
            sa_fill = positive_reference_floor(fit_numeric["sa"])
        out["sa"] = out["sa"].fillna(sa_fill)
    out = fill_experimental_conditions(out, fit_numeric)
    out = sanitize_physical_features(out, fit_df)
    out["mpd"] = derive_mpd(out["pv"], out["sa"])
    pv_predictors = [feature for feature in [*STRUCTURAL_IMPUTE_CONTEXT, "pd"] if feature != "pv"]
    out = impute_structural_feature(out, fit_numeric, "pv", pv_predictors)
    out["mpd"] = derive_mpd(out["pv"], out["sa"])
    pd_predictors = [feature for feature in [*STRUCTURAL_IMPUTE_CONTEXT, "pv"] if feature != "pd"]
    out = impute_structural_feature(out, fit_numeric, "pd", pd_predictors)
    out = sanitize_physical_features(out, fit_df)
    out["mpd"] = derive_mpd(out["pv"], out["sa"])
    median_mpd = float(out["mpd"].dropna().median()) if out["mpd"].notna().any() else 0.0
    out["mpd"] = out["mpd"].fillna(median_mpd).clip(lower=0.0)
    for feature_name in TRAINING_FEATURES:
        if out[feature_name].isna().any():
            fit_values = pd.to_numeric(fit_df[feature_name], errors="coerce")
            if fit_values.notna().any():
                fill_value = float(fit_values.median())
            else:
                fill_value = float(pd.to_numeric(out[feature_name], errors="coerce").dropna().median()) if out[feature_name].notna().any() else 0.0
            out[feature_name] = pd.to_numeric(out[feature_name], errors="coerce").fillna(fill_value)
    return out

def make_group_ids(raw_df: pd.DataFrame, recipe: str) -> pd.Series:
    if recipe == "metal_only":
        signature = raw_df["Coordination metal"].astype(str).str.strip()
    elif recipe == "metal_family":
        signature = pd.DataFrame(
            {
                "doi": raw_df["DOI"].ffill().astype(str).str.strip(),
                "metal": raw_df["Coordination metal"].astype(str).str.strip(),
                "modification": raw_df["Modification method"].astype(str).str.strip(),
            }
        ).astype(str).agg("|".join, axis=1)
    elif recipe == "paper_43":
        cols = ["Coordination metal", "Modification method", "Surface areas(m2/g)"]
        signature = raw_df[cols].astype(str).agg("|".join, axis=1)
    elif recipe == "structural_44":
        cols = [
            "Coordination metal",
            "Modification method",
            "Surface areas(m2/g)",
            "Reporting pore diameter(nm)\n",
            "pore volume(cm3/g)",
        ]
        signature = raw_df[cols].astype(str).agg("|".join, axis=1)
    else:
        raise ValueError(f"Unknown group recipe: {recipe}")
    return pd.Series(pd.factorize(signature)[0], index=raw_df.index, name="group_id")


# --- loaders.py ---

def load_raw_original_sheet() -> pd.DataFrame:
    data_xlsx = resolve_data_xlsx()
    raw = pd.read_excel(data_xlsx, sheet_name="Original dataset", header=None)
    df = raw.iloc[2:].copy()
    df.columns = raw.iloc[1].tolist()
    return df.reset_index(drop=True)

def load_training_table(descriptor_preset: str, mod_encoding: str, group_recipe: str) -> pd.DataFrame:
    raw_df = load_raw_original_sheet()
    group_id = make_group_ids(raw_df, group_recipe)

    df = raw_df.rename(columns=RAW_COLUMN_MAP)[list(RAW_COLUMN_MAP.values())].copy()
    for col in ["sa", "pd", "pv", "ci", "ad", "time", "ph", "temp", "q"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["metal"] = df["metal"].astype(str).str.strip()
    df["modification"] = df["modification"].astype(str).str.strip()
    df["doi"] = df["doi"].ffill()
    df["ph"] = df["ph"].fillna(7.0)
    df["temp"] = df["temp"].fillna(298.0)
    df["group_id"] = group_id

    props = df["metal"].map(DESCRIPTOR_PRESETS[descriptor_preset]).apply(pd.Series)
    if props.isna().all(axis=1).any():
        unknown_metals = sorted(df.loc[props.isna().all(axis=1), "metal"].dropna().astype(str).unique().tolist())
        raise ValueError(
            f"Descriptor preset '{descriptor_preset}' does not cover metals: {', '.join(unknown_metals)}"
        )
    df = pd.concat([df, props], axis=1)
    df["mod_code"], _, _ = encode_modification_codes(df["modification"], mod_encoding)
    df["mpd"] = derive_mpd(df["pv"], df["sa"])
    return df

def load_core_mof_table() -> pd.DataFrame | None:
    csv_path = ROOT / "2019-11-01-ASR-internal_14142.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        required = {"filename", "LCD", "AV_cm3_g", "ASA_m2_g", "AV_VF", "All_Metals"}
        if not required.difference(df.columns):
            out = df.rename(
                columns={
                    "filename": "cif_file",
                    "All_Metals": "metal",
                    "LCD": "pd_raw",
                    "AV_cm3_g": "pv_raw",
                    "ASA_m2_g": "sa_raw",
                    "AV_VF": "vf_raw",
                }
            ).copy()
            out["pd"] = pd.to_numeric(out["pd_raw"], errors="coerce")
            out["pv"] = pd.to_numeric(out["pv_raw"], errors="coerce")
            out["sa"] = pd.to_numeric(out["sa_raw"], errors="coerce")
            out["vf"] = pd.to_numeric(out["vf_raw"], errors="coerce")
            out["mpd"] = derive_mpd(out["pv"], out["sa"])
            out["source_file"] = csv_path.name
            out = out.dropna(subset=["pv", "pd", "sa", "vf"])
            if not out.empty:
                return out

    candidates = [
        ROOT / "mof_data_extended_completed.xlsx",
        ROOT / "mof_data_extended_mpd_lcd_filled.xlsx",
    ]
    for path in candidates:
        if not path.exists():
            continue
        df = pd.read_excel(path)
        if {"CIF File", "Metal Types", "Max Pore Diameter (MPD)", "Max Cavity Diameter (LCD)"}.difference(df.columns):
            continue

        pv_col = "PV" if "PV" in df.columns else "Pore Volume" if "Pore Volume" in df.columns else None
        sa_col = "SA" if "SA" in df.columns else "Surface Area (SA)" if "Surface Area (SA)" in df.columns else None
        if pv_col is None or sa_col is None:
            continue

        out = df.rename(
            columns={
                "CIF File": "cif_file",
                "Formula": "formula",
                "Volume": "volume",
                "Metal Types": "metal",
                pv_col: "pv_raw",
                "Max Pore Diameter (MPD)": "mpd_raw",
                "Max Cavity Diameter (LCD)": "lcd_raw",
                sa_col: "sa_raw",
            }
        ).copy()
        out["pv"] = pd.to_numeric(out["pv_raw"], errors="coerce")
        out["mpd"] = pd.to_numeric(out["mpd_raw"], errors="coerce")
        out["pd"] = pd.to_numeric(out["lcd_raw"], errors="coerce")
        out["sa"] = pd.to_numeric(out["sa_raw"], errors="coerce")
        out["volume"] = pd.to_numeric(out["volume"], errors="coerce")
        out["source_file"] = path.name
        out = out.dropna(subset=["pv", "pd", "sa"])
        if not out.empty:
            return out
    return None

def export_target_core_metal_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    csv_path = ROOT / "2019-11-01-ASR-internal_14142.csv"
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path)
    all_metals = df["All_Metals"].astype(str).str.strip()
    single_mask = all_metals.isin(TARGET_CORE_METALS)
    subset = df.loc[single_mask].copy()
    subset.insert(1, "target_metal_basis", subset["All_Metals"].astype(str).str.strip())

    count_rows: list[dict[str, int | str]] = []
    for metal in TARGET_CORE_METALS:
        exact_mask = all_metals == metal
        contains_mask = all_metals.str.contains(rf"(?:^|[^A-Za-z]){metal}(?:[^A-Za-z]|$)", regex=True, na=False)
        count_rows.append(
            {
                "metal": metal,
                "exact_single_metal_rows": int(exact_mask.sum()),
                "contains_rows": int(contains_mask.sum()),
            }
        )

    subset.to_csv(ROOT / "outputs" / "core_target_metals_single_metal.csv", index=False)
    count_df = pd.DataFrame(count_rows)
    count_df.to_csv(ROOT / "outputs" / "core_target_metal_counts.csv", index=False)
    return subset, count_df

def infer_doi_from_cif_file(cif_file: str) -> str | None:
    """
    Recover only high-confidence DOI strings from filename patterns.

    Many CoRE filenames are plain CSD-like refcodes, so this helper only fills
    obvious ACS/RSC-style article identifiers and leaves the rest blank.
    """
    name = str(cif_file).strip()
    if not name:
        return None
    lower = name.lower()

    match = re.match(r"^(jacs\.[0-9a-z]+)", lower)
    if match:
        return f"10.1021/{match.group(1)}"

    match = re.match(r"^(ja\d{5,}[a-z]?)", lower)
    if match:
        return f"10.1021/{match.group(1)}"

    match = re.match(r"^(jp\d{5,}[a-z]?)", lower)
    if match:
        return f"10.1021/{match.group(1)}"

    match = re.match(r"^(cg\d{5,}[a-z]?)", lower)
    if match:
        return f"10.1021/{match.group(1)}"

    match = re.match(r"^(c\d[a-z]{2}\d{5}[a-z])", lower)
    if match:
        return f"10.1039/{match.group(1)}"

    return None

def enrich_candidate_doi_public(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    raw_doi = (
        out.get("doi_public", pd.Series(index=out.index, dtype=object))
        .astype(str)
        .str.strip()
        .replace({"": np.nan, "nan": np.nan, "NaN": np.nan, "None": np.nan})
    )
    inferred = out["cif_file"].map(infer_doi_from_cif_file)
    out["doi_public"] = raw_doi.fillna(inferred)
    return out

def build_target_core_feature_table(
    raw_subset: pd.DataFrame,
    descriptor_preset: str,
    fit_df: pd.DataFrame,
    mod_encoding: str,
) -> pd.DataFrame:
    # Build the candidate CoRE-MOF feature table used by screening/Fig. 5.
    out = raw_subset.copy()
    out = out.rename(columns={"filename": "cif_file", "All_Metals": "metal", "DOI_public": "doi_public"}).copy()
    out["metal"] = out["metal"].astype(str).str.strip()
    out["pd"] = pd.to_numeric(out["LCD"], errors="coerce")
    out["pv"] = pd.to_numeric(out["AV_cm3_g"], errors="coerce")
    out["sa"] = pd.to_numeric(out["ASA_m2_g"], errors="coerce")
    out["vf"] = pd.to_numeric(out["AV_VF"], errors="coerce")
    # Remove CoRE entries with zero accessible surface area instead of
    # compressing them to a tiny positive placeholder. These rows distort
    # Fig. 5(c) and are not useful for SA-based structure-performance analysis.
    out = out.loc[out["sa"] > 0].copy()
    out["ci"] = 300.0
    out["ad"] = 500.0
    out["time"] = 720.0
    out["ph"] = 7.0
    out["temp"] = 298.0
    out = sanitize_candidate_core_features(out)
    out["mpd"] = derive_mpd(out["pv"], out["sa"]).fillna(float(fit_df["mpd"].median()))
    out["modification"] = "Marginalized"
    out["mod_code"] = np.nan
    out["mod_strategy"] = SCREENING_MOD_STRATEGY
    props = out["metal"].map(DESCRIPTOR_PRESETS[descriptor_preset]).apply(pd.Series)
    out = pd.concat([out, props], axis=1)
    out = enrich_candidate_doi_public(out)
    out["source_file"] = "2019-11-01-ASR-internal_14142.csv"
    ordered_columns = [
        "cif_file",
        "metal",
        "modification",
        "mod_code",
        "mod_strategy",
        "sa",
        "mpd",
        "pd",
        "pv",
        "vf",
        "ci",
        "ad",
        "time",
        "ph",
        "temp",
        "ionic_charge",
        "atomic_radius",
        "polarizability",
        "electronegativity",
        "doi_public",
        "source_file",
    ]
    required_columns = [
        col
        for col in ordered_columns
        if col not in {"modification", "mod_code", "mod_strategy", "doi_public"}
    ]
    out = out[ordered_columns].dropna(subset=required_columns).reset_index(drop=True)
    return out


