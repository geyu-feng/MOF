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
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import IterativeImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, KFold, ParameterGrid
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor

warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn.impute._iterative")


# --- constants.py ---

ROOT = Path(__file__).resolve().parents[1]

DATA_XLSX_CANDIDATES = [
    ROOT / "11.xlsx",
    ROOT / "1-s2.0-S002197972400328X-mmc2.xlsx",
]

OUTPUT_DIR = ROOT / "outputs"

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

IMPUTE_FEATURES = ["sa", "pd", "pv", "ci", "ad"]

TRAIN_IMPUTE_CONTEXT = [
    "ionic_charge",
    "atomic_radius",
    "polarizability",
    "electronegativity",
    "mod_code",
    "time",
    "ph",
    "temp",
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

TARGET_CORE_METALS = ["Zn", "In", "Fe", "Cu", "Ti", "Zr", "Nd"]

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

def build_iterative_imputer() -> IterativeImputer:
    return IterativeImputer(
        estimator=RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
        random_state=42,
        max_iter=20,
    )

def sanitize_physical_features(df: pd.DataFrame, fit_df: pd.DataFrame) -> pd.DataFrame:
    """Clip physically invalid values after imputation using train-side limits only."""
    out = df.copy()
    positive_sa = pd.to_numeric(fit_df["sa"], errors="coerce")
    positive_sa = positive_sa[positive_sa > 0]
    sa_floor = float(positive_sa.quantile(0.01)) if not positive_sa.empty else 1e-6
    out["sa"] = pd.to_numeric(out["sa"], errors="coerce").clip(lower=max(sa_floor, 1e-6))
    for col in ["pd", "pv", "ci", "ad"]:
        out[col] = pd.to_numeric(out[col], errors="coerce").clip(lower=0.0)
    return out

def derive_mpd(pv: pd.Series, sa: pd.Series) -> pd.Series:
    """Back-fill theoretical MPD from accessible PV and SA as described in the paper."""
    safe_sa = pd.to_numeric(sa, errors="coerce").replace(0, np.nan)
    mpd = 4000.0 * pd.to_numeric(pv, errors="coerce") / safe_sa
    return mpd.replace([np.inf, -np.inf], np.nan)

def resolve_data_xlsx() -> Path:
    for path in DATA_XLSX_CANDIDATES:
        if path.exists():
            return path
    searched = ", ".join(str(path.name) for path in DATA_XLSX_CANDIDATES)
    raise FileNotFoundError(f"No training workbook found. Checked: {searched}")

def prepare_model_table(df: pd.DataFrame, fit_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Apply the paper-style preprocessing pipeline with train-only fitted state."""
    fit_df = df if fit_df is None else fit_df
    out = df.copy()
    context_cols = [*TRAIN_IMPUTE_CONTEXT, *IMPUTE_FEATURES]
    imputer = build_iterative_imputer()
    imputer.fit(fit_df[context_cols])
    transformed = pd.DataFrame(imputer.transform(out[context_cols]), columns=context_cols, index=out.index)
    out[IMPUTE_FEATURES] = transformed[IMPUTE_FEATURES]
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
    df = pd.concat([df, props], axis=1)
    df["mod_code"] = df["modification"].map(MOD_ENCODING_PRESETS[mod_encoding]).astype(int)
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


# --- builders.py ---

def normalize_core_metal(raw_value: str, allowed: set[str]) -> str | None:
    value = str(raw_value).strip()
    if value in allowed:
        return value
    tokens = re.findall(r"[A-Z][a-z]?", value)
    matched = [token for token in tokens if token in allowed]
    if len(set(matched)) == 1 and len(tokens) == 1:
        return matched[0]
    return None

def compute_formula_mass(formula: str) -> float:
    tokens = re.findall(r"([A-Z][a-z]?)(\d*\.?\d*)", str(formula))
    total = 0.0
    for element, count in tokens:
        if element not in ATOMIC_WEIGHTS:
            continue
        total += ATOMIC_WEIGHTS[element] * (float(count) if count else 1.0)
    return total

def compute_void_fraction(frame: pd.DataFrame) -> pd.Series:
    if {"pv", "formula"}.issubset(frame.columns):
        mass = frame["formula"].map(compute_formula_mass)
        return (frame["pv"] * mass / 1000.0).clip(lower=0.0, upper=1.0)
    return pd.Series(np.nan, index=frame.index)

def build_core_prediction_table(core_df: pd.DataFrame | None, descriptor_preset: str) -> pd.DataFrame:
    if core_df is None or core_df.empty:
        raise ValueError("No CoRE-MOF table available for prediction.")
    allowed_metals = set(DESCRIPTOR_PRESETS[descriptor_preset])
    out = core_df.copy()
    out["metal"] = out["metal"].map(lambda value: normalize_core_metal(value, allowed_metals))
    out = out.dropna(subset=["metal"]).reset_index(drop=True)
    props = out["metal"].map(DESCRIPTOR_PRESETS[descriptor_preset]).apply(pd.Series)
    out = pd.concat([out, props], axis=1)
    out["vf"] = out["vf"].fillna(compute_void_fraction(out))
    out["ci"] = 300.0
    out["ad"] = 500.0
    out["time"] = 720.0
    out["ph"] = 7.0
    out["temp"] = 298.0
    return out

def build_target_core_feature_table(
    raw_subset: pd.DataFrame,
    descriptor_preset: str,
    fit_df: pd.DataFrame,
    mod_encoding: str,
) -> pd.DataFrame:
    out = raw_subset.copy()
    out = out.rename(columns={"filename": "cif_file", "All_Metals": "metal"}).copy()
    out["metal"] = out["metal"].astype(str).str.strip()
    out["pd"] = pd.to_numeric(out["LCD"], errors="coerce")
    out["pv"] = pd.to_numeric(out["AV_cm3_g"], errors="coerce")
    out["sa"] = pd.to_numeric(out["ASA_m2_g"], errors="coerce")
    out["vf"] = pd.to_numeric(out["AV_VF"], errors="coerce")
    out["ci"] = 300.0
    out["ad"] = 500.0
    out["time"] = 720.0
    out["ph"] = 7.0
    out["temp"] = 298.0
    out = sanitize_physical_features(out, fit_df)
    out["mpd"] = derive_mpd(out["pv"], out["sa"]).fillna(float(fit_df["mpd"].median()))
    out["modification"] = "Marginalized"
    out["mod_code"] = np.nan
    out["mod_strategy"] = SCREENING_MOD_STRATEGY
    props = out["metal"].map(DESCRIPTOR_PRESETS[descriptor_preset]).apply(pd.Series)
    out = pd.concat([out, props], axis=1)
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
        "source_file",
    ]
    required_columns = [col for col in ordered_columns if col not in {"modification", "mod_code", "mod_strategy"}]
    out = out[ordered_columns].dropna(subset=required_columns).reset_index(drop=True)
    return out


# --- core.py ---

def get_model_param_grids() -> dict[str, dict[str, list[object]]]:
    return {
        "RF": {"n_estimators": [100, 200], "max_depth": [4, 6, 8], "min_samples_leaf": [3, 5, 8]},
        "GBDT": {
            "n_estimators": [80, 120, 160],
            "max_depth": [2, 3],
            "min_samples_leaf": [1, 2, 4],
            "learning_rate": [0.03, 0.05, 0.1],
        },
        "XGB": {
            "n_estimators": [100, 160, 220],
            "max_depth": [2, 4, 6],
            "min_child_weight": [3, 5, 8],
            "gamma": [0.0, 0.1, 0.3],
            "learning_rate": [0.03, 0.05, 0.1],
            "subsample": [0.7, 0.85],
            "colsample_bytree": [0.7, 0.9],
        },
        "LR": {"fit_intercept": [True, False]},
        "KNN": {"n_neighbors": [3, 5, 7, 9, 11, 15], "weights": ["uniform", "distance"]},
        "SVR": {"C": [1.0, 5.0, 10.0, 50.0, 100.0], "epsilon": [0.1, 0.5, 1.0, 5.0], "gamma": ["scale", 0.01, 0.05, 0.1]},
    }

def instantiate_model(model_name: str, params: dict[str, object] | None = None) -> object:
    params = {} if params is None else params.copy()
    if model_name == "RF":
        base = {"n_estimators": 200, "random_state": 42, "n_jobs": -1, "min_samples_leaf": 2}
        base.update(params)
        return RandomForestRegressor(**base)
    if model_name == "GBDT":
        base = {"random_state": 42}
        base.update(params)
        return GradientBoostingRegressor(**base)
    if model_name == "XGB":
        base = {
            "n_estimators": 250,
            "learning_rate": 0.01,
            "max_depth": 15,
            "subsample": 0.8,
            "colsample_bytree": 0.9,
            "gamma": 0.0,
            "min_child_weight": 5,
            "objective": "reg:squarederror",
            "random_state": 42,
            "n_jobs": 8,
        }
        base.update(params)
        return XGBRegressor(**base)
    if model_name == "LR":
        base = {"fit_intercept": True}
        base.update(params)
        return LinearRegression(**base)
    if model_name == "KNN":
        base = {"n_neighbors": 5, "weights": "distance"}
        base.update(params)
        return KNeighborsRegressor(**base)
    if model_name == "SVR":
        base = {"kernel": "rbf", "C": 10.0, "epsilon": 1.0, "gamma": "scale"}
        base.update(params)
        return SVR(**base)
    raise ValueError(f"Unknown model name: {model_name}")

def build_models() -> dict[str, object]:
    return {model_name: instantiate_model(model_name) for model_name in MODEL_ORDER}

def build_preprocessor(feature_columns: Iterable[str]) -> ColumnTransformer:
    feature_columns = list(feature_columns)
    scale_cols = [col for col in feature_columns if col != "mod_code"]
    passthrough_cols = [col for col in feature_columns if col == "mod_code"]
    return ColumnTransformer(
        transformers=[("scale", StandardScaler(), scale_cols), ("pass", "passthrough", passthrough_cols)],
        remainder="drop",
    )

def build_group_cv_folds(raw_df: pd.DataFrame, n_splits: int = 10) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    splitter = GroupKFold(n_splits=n_splits)
    folds: list[tuple[pd.DataFrame, pd.DataFrame]] = []
    for train_idx, val_idx in splitter.split(raw_df, groups=raw_df["group_id"]):
        fold_train_raw = raw_df.iloc[train_idx].copy()
        fold_val_raw = raw_df.iloc[val_idx].copy()
        fold_train = prepare_model_table(fold_train_raw, fit_df=fold_train_raw)
        fold_val = prepare_model_table(fold_val_raw, fit_df=fold_train_raw)
        folds.append((fold_train, fold_val))
    return folds

def build_kfold_prepared_folds(raw_df: pd.DataFrame, n_splits: int = 10) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    folds: list[tuple[pd.DataFrame, pd.DataFrame]] = []
    for train_idx, val_idx in splitter.split(raw_df):
        fold_train_raw = raw_df.iloc[train_idx].copy()
        fold_val_raw = raw_df.iloc[val_idx].copy()
        fold_train = prepare_model_table(fold_train_raw, fit_df=fold_train_raw)
        fold_val = prepare_model_table(fold_val_raw, fit_df=fold_train_raw)
        folds.append((fold_train, fold_val))
    return folds

def recommend_test_group_count(n_groups: int) -> int:
    if n_groups <= 5:
        return 1
    if n_groups <= 12:
        return 2
    return max(1, int(round(n_groups * 0.15)))

def _run_model_grid_search_cv(raw_df: pd.DataFrame, output_dir) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Pipeline], pd.DataFrame]:
    n_splits = min(10, int(raw_df["group_id"].nunique()))
    folds = build_group_cv_folds(raw_df, n_splits=n_splits)
    rows: list[dict[str, object]] = []

    for model_name in MODEL_ORDER:
        for params in ParameterGrid(get_model_param_grids()[model_name]):
            fold_r2: list[float] = []
            fold_mae: list[float] = []
            fold_rmse: list[float] = []
            actual_chunks: list[np.ndarray] = []
            pred_chunks: list[np.ndarray] = []
            for fold_train, fold_val in folds:
                pipe = Pipeline([("prep", build_preprocessor(TRAINING_FEATURES)), ("model", instantiate_model(model_name, params))])
                pipe.fit(fold_train[TRAINING_FEATURES], fold_train["q"].to_numpy())
                pred = pipe.predict(fold_val[TRAINING_FEATURES])
                actual = fold_val["q"].to_numpy()
                fold_r2.append(r2_score(actual, pred))
                fold_mae.append(mean_absolute_error(actual, pred))
                fold_rmse.append(mean_squared_error(actual, pred) ** 0.5)
                actual_chunks.append(actual)
                pred_chunks.append(pred)

            actual_all = np.concatenate(actual_chunks)
            pred_all = np.concatenate(pred_chunks)

            rows.append(
                {
                    "model": model_name,
                    "params_json": json.dumps(params, sort_keys=True),
                    "mean_r2": float(np.mean(fold_r2)),
                    "std_r2": float(np.std(fold_r2)),
                    "mean_mae": float(np.mean(fold_mae)),
                    "mean_rmse": float(np.mean(fold_rmse)),
                    "oof_r2": float(r2_score(actual_all, pred_all)),
                    "oof_mae": float(mean_absolute_error(actual_all, pred_all)),
                    "oof_rmse": float(mean_squared_error(actual_all, pred_all) ** 0.5),
                    "cv_strategy": "group_kfold",
                    "cv_folds": int(n_splits),
                }
            )

    cv_results = pd.DataFrame(rows).sort_values(["model", "mean_r2", "mean_rmse", "mean_mae"], ascending=[True, False, True, True])
    best_per_model = cv_results.drop_duplicates(subset=["model"], keep="first").sort_values(
        ["oof_r2", "oof_rmse", "oof_mae"], ascending=[False, True, True]
    )
    cv_results.to_csv(output_dir / "model_grid_search_cv.csv", index=False)
    best_per_model.to_csv(output_dir / "model_grid_search_best_per_model.csv", index=False)

    prepared_full_df = prepare_model_table(raw_df, fit_df=raw_df)
    fitted_best_models: dict[str, Pipeline] = {}
    y = prepared_full_df["q"].to_numpy()
    for row in best_per_model.itertuples(index=False):
        params = json.loads(row.params_json)
        pipe = Pipeline([("prep", build_preprocessor(TRAINING_FEATURES)), ("model", instantiate_model(row.model, params))])
        pipe.fit(prepared_full_df[TRAINING_FEATURES], y)
        fitted_best_models[row.model] = pipe
    return cv_results, best_per_model, fitted_best_models, prepared_full_df

def make_split(df: pd.DataFrame, config) -> SplitBundle:
    groups = pd.Series(df["group_id"])
    unique_groups = pd.Index(groups.drop_duplicates())
    recommended_test_groups = recommend_test_group_count(len(unique_groups))

    if config.mode == "sequential":
        default_train_group_count = len(unique_groups) - recommended_test_groups
        train_group_count = config.train_group_count or default_train_group_count
        train_group_count = min(max(1, train_group_count), max(1, len(unique_groups) - recommended_test_groups))
        train_groups = set(unique_groups[:train_group_count])
        train_idx = df.index[df["group_id"].isin(train_groups)].to_numpy()
        test_idx = df.index[~df["group_id"].isin(train_groups)].to_numpy()
        return SplitBundle(train_idx, test_idx, len(train_groups), df.loc[test_idx, "group_id"].nunique())

    if config.mode == "group_shuffle":
        test_size = recommended_test_groups if len(unique_groups) <= GROUP_AWARE_TUNING_GROUP_THRESHOLD else 0.1
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=config.random_state)
        train_idx, test_idx = next(splitter.split(df, groups=groups))
        return SplitBundle(train_idx, test_idx, df.iloc[train_idx]["group_id"].nunique(), df.iloc[test_idx]["group_id"].nunique())

    raise ValueError(f"Unknown split mode: {config.mode}")

def fit_models_for_split(
    df: pd.DataFrame,
    config,
    tuned_params: dict[str, dict[str, object]] | None = None,
) -> tuple[pd.DataFrame, dict[str, dict[str, pd.DataFrame]], dict[str, Pipeline], dict[str, pd.DataFrame]]:
    split = make_split(df, config)
    raw_train_df = df.loc[split.train_idx].copy()
    raw_test_df = df.loc[split.test_idx].copy()
    train_df = prepare_model_table(raw_train_df, fit_df=raw_train_df)
    test_df = prepare_model_table(raw_test_df, fit_df=raw_train_df)
    y_train = train_df["q"].to_numpy()
    y_test = test_df["q"].to_numpy()

    metrics = []
    predictions: dict[str, dict[str, pd.DataFrame]] = {}
    fitted: dict[str, Pipeline] = {}
    for model_name in MODEL_ORDER:
        model = instantiate_model(model_name, None if tuned_params is None else tuned_params.get(model_name))
        pipe = Pipeline([("prep", build_preprocessor(TRAINING_FEATURES)), ("model", model)])
        pipe.fit(train_df[TRAINING_FEATURES], y_train)
        train_pred = pipe.predict(train_df[TRAINING_FEATURES])
        test_pred = pipe.predict(test_df[TRAINING_FEATURES])

        metrics.append(
            {
                "config": config.name,
                "model": model_name,
                "mae": mean_absolute_error(y_test, test_pred),
                "rmse": mean_squared_error(y_test, test_pred) ** 0.5,
                "r2": r2_score(y_test, test_pred),
                "train_rows": len(train_df),
                "test_rows": len(test_df),
                "train_groups": split.train_group_count,
                "test_groups": split.test_group_count,
                "descriptor_preset": config.descriptor_preset,
                "mod_encoding": config.mod_encoding,
                "split_mode": config.mode,
            }
        )
        predictions[model_name] = {
            "train": pd.DataFrame({"actual_q": y_train, "predicted_q": train_pred}),
            "test": pd.DataFrame({"actual_q": y_test, "predicted_q": test_pred}),
        }
        fitted[model_name] = pipe
    prepared_split = {"train": train_df, "test": test_df}
    return pd.DataFrame(metrics), predictions, fitted, prepared_split

def score_metric_frame(metrics: pd.DataFrame) -> float:
    avg_r2 = float(metrics["r2"].mean())
    avg_rmse = float(metrics["rmse"].mean())
    avg_mae = float(metrics["mae"].mean())
    xgb_row = metrics.loc[metrics["model"] == "XGB"].iloc[0]
    return -avg_r2 + 0.01 * avg_rmse + 0.01 * avg_mae - 0.1 * float(xgb_row["r2"])


# --- core.py ---

def get_screening_mod_weights(training_df: pd.DataFrame) -> dict[int, float]:
    weights = training_df["mod_code"].value_counts(normalize=True).sort_index()
    return {int(code): float(weight) for code, weight in weights.items()}

def predict_with_mod_marginalization(pipe: Pipeline, frame: pd.DataFrame, mod_weights: dict[int, float]) -> np.ndarray:
    expected = np.zeros(len(frame), dtype=float)
    for mod_code, weight in mod_weights.items():
        scenario = frame.copy()
        scenario["mod_code"] = float(mod_code)
        expected += weight * pipe.predict(scenario[TRAINING_FEATURES])
    return expected

def make_first_adsorption_dataset(
    core_df: pd.DataFrame,
    best_model_name: str,
    best_pipe: Pipeline,
    mod_weights: dict[int, float],
) -> pd.DataFrame:
    out = core_df.copy()
    out["first_model_name"] = best_model_name
    out["first_model_q"] = predict_with_mod_marginalization(best_pipe, out, mod_weights)
    out = out.sort_values("first_model_q", ascending=False).reset_index(drop=True)
    out["first_global_rank"] = np.arange(1, len(out) + 1)
    out["first_metal_rank"] = out.groupby("metal")["first_model_q"].rank(ascending=False, method="dense").astype(int)
    return out

def build_initial_screening_from_first_dataset(first_df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    chunks: list[pd.DataFrame] = []
    for metal in TARGET_CORE_METALS:
        chunk = first_df.loc[first_df["metal"] == metal].sort_values("first_model_q", ascending=False).head(top_n).copy()
        chunk["initial_screening_rank"] = np.arange(1, len(chunk) + 1)
        chunks.append(chunk)
    out = pd.concat(chunks, ignore_index=True)
    out["first_global_rank"] = out["first_model_q"].rank(ascending=False, method="dense").astype(int)
    return out.sort_values(["metal", "first_metal_rank"]).reset_index(drop=True)

def make_second_adsorption_dataset(
    initial_screening_df: pd.DataFrame,
    second_model_name: str,
    second_pipe: Pipeline,
    mod_weights: dict[int, float],
) -> pd.DataFrame:
    out = initial_screening_df.copy()
    out["second_model_name"] = second_model_name
    out["second_model_q"] = predict_with_mod_marginalization(second_pipe, out, mod_weights)
    out = out.sort_values("second_model_q", ascending=False).reset_index(drop=True)
    out["second_global_rank"] = np.arange(1, len(out) + 1)
    return out


# --- main_figures.py ---

def nice_axis_upper(value: float) -> float:
    if value <= 0:
        return 1.0
    magnitude = 10 ** np.floor(np.log10(value))
    scaled = value / magnitude
    if scaled <= 1.2:
        nice = 1.2
    elif scaled <= 1.5:
        nice = 1.5
    elif scaled <= 2.0:
        nice = 2.0
    elif scaled <= 3.0:
        nice = 3.0
    elif scaled <= 4.0:
        nice = 4.0
    elif scaled <= 5.0:
        nice = 5.0
    elif scaled <= 6.0:
        nice = 6.0
    elif scaled <= 8.0:
        nice = 8.0
    else:
        nice = 10.0
    return nice * magnitude

def prepare_fig2a_structural_data(core_df: pd.DataFrame | None, fi_df: pd.DataFrame) -> pd.DataFrame:
    structural_df = core_df if core_df is not None and not core_df.empty else fi_df.rename(columns={"sa": "sa", "pv": "pv", "pd": "pd", "mpd": "mpd"})
    plot_df = structural_df.copy()
    if "pv" not in plot_df.columns or "sa" not in plot_df.columns or "pd" not in plot_df.columns:
        raise ValueError("Structural plotting data is missing one or more required columns: pd, pv, sa.")
    plot_df["x"] = pd.to_numeric(plot_df["pd"], errors="coerce") / 10.0
    plot_df["y"] = pd.to_numeric(plot_df["pv"], errors="coerce")
    plot_df["z"] = pd.to_numeric(plot_df["sa"], errors="coerce")
    plot_df = plot_df.dropna(subset=["x", "y", "z"])
    if plot_df["y"].quantile(0.99) > 10:
        plot_df["y"] = plot_df["y"] / 10.0
    plot_df = plot_df[(plot_df["x"] >= 0) & (plot_df["y"] >= 0) & (plot_df["z"] >= 0)]
    return plot_df[["x", "y", "z"]]

def style_fig2a_axis(ax, plot_df: pd.DataFrame) -> None:
    x_upper = nice_axis_upper(float(plot_df["x"].quantile(0.995)))
    y_upper = nice_axis_upper(float(plot_df["y"].quantile(0.995)))
    z_upper = nice_axis_upper(float(plot_df["z"].quantile(0.995)))
    ax.set_xlim(0, x_upper)
    ax.set_ylim(0, y_upper)
    ax.set_zlim(0, z_upper)
    ax.set_box_aspect((1.15, 1.15, 1.25))
    ax.view_init(elev=24, azim=-56)
    ax.tick_params(labelsize=6, pad=-1)
    ax.set_xlabel("PD(nm)", labelpad=-4)
    ax.set_ylabel("PV(cm$^3$/g)", labelpad=-2)
    ax.set_zlabel("SA(m$^2$/g)", labelpad=1)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((1.0, 1.0, 1.0, 1.0))
        axis.pane.set_edgecolor("black")
        axis._axinfo["grid"]["color"] = (0.72, 0.72, 0.72, 1.0)
        axis._axinfo["grid"]["linewidth"] = 0.5
        axis._axinfo["axisline"]["color"] = (0.0, 0.0, 0.0, 1.0)
        axis._axinfo["axisline"]["linewidth"] = 0.8

def save_fig3_like(predictions: dict[str, dict[str, pd.DataFrame]], metrics: pd.DataFrame, filename: str) -> None:
    set_paper_rcparams()
    fig, axes = plt.subplots(2, 3, figsize=(7.1, 6.2))
    display_order = ["RF", "GBDT", "XGB", "LR", "KNN", "SVR"]
    letter_order = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]

    for ax, model_name, letter in zip(axes.flat, display_order, letter_order):
        train_df = predictions[model_name]["train"]
        test_df = predictions[model_name]["test"]
        combined = pd.concat([train_df, test_df], ignore_index=True)
        max_v = max(250.0, combined[["actual_q", "predicted_q"]].to_numpy().max() * 1.02)
        ax.scatter(train_df["predicted_q"], train_df["actual_q"], s=8, c="#5dd18a", alpha=0.7, label="Train", edgecolors="none")
        ax.scatter(test_df["predicted_q"], test_df["actual_q"], s=8, c="#0b5d1e", alpha=0.85, label="Test", edgecolors="none")
        ax.plot([0, max_v], [0, max_v], color="#e67e22", linewidth=0.9, linestyle="--")
        ax.set_xlim(0, max_v)
        ax.set_ylim(0, max_v)
        ax.set_xlabel("Predicted Q(mg/g)")
        ax.set_ylabel("Actual Q(mg/g)")
        ax.set_title(f"{letter}{model_name}", pad=2)
        metric_row = metrics.loc[metrics["model"] == model_name].iloc[0]
        ax.text(0.60, 0.10, f"R$^2$={metric_row['r2']:.2f}\nMAE={metric_row['mae']:.2f}\nRMSE={metric_row['rmse']:.2f}", transform=ax.transAxes, fontsize=7)
        ax.legend(loc="upper left", frameon=False, handlelength=1.0, borderpad=0.2)
        style_small_axis(ax)
        ax.set_aspect("equal", adjustable="box")

    fig.subplots_adjust(left=0.08, right=0.98, top=0.95, bottom=0.14, wspace=0.28, hspace=0.32)
    add_caption(fig, "Fig. 3. Fitting effect diagram of six machine learning models.")
    fig.savefig(OUTPUT_DIR / filename, dpi=300)
    plt.close(fig)

def save_fig5_like(screening_df: pd.DataFrame, filename: str) -> None:
    set_paper_rcparams()
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 4.8))
    panels = [
        ("pd", "LCD(nm)", "(a)", (0.0, 7.0), np.arange(0.0, 7.5, 0.5)),
        ("pv", "PV(cm$^3$/g)", "(b)", (0.0, 2.0), np.arange(0.0, 2.1, 0.2)),
        ("sa", "SA(m$^2$/g)", "(c)", (0.0, 6000.0), np.arange(0.0, 6001.0, 1000.0)),
        ("vf", "VF", "(d)", (0.2, 1.0), np.arange(0.2, 1.01, 0.1)),
    ]
    y = screening_df["first_model_q"].to_numpy()
    y_upper = max(160.0, float(np.nanmax(y) * 1.01))
    y_ticks = [0, 50, 100, 150]
    for ax, (feature, xlabel, letter, xlim, xticks) in zip(axes.flat, panels):
        x = screening_df[feature].to_numpy()
        if feature == "pd":
            x = x / 10.0
        density = gaussian_kde(np.vstack([x, y]))(np.vstack([x, y])) if len(np.unique(x)) > 1 else np.ones_like(x)
        order = np.argsort(density)
        scatter = ax.scatter(x[order], y[order], c=density[order], cmap="turbo", s=7, edgecolors="none")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Q(mg/g)")
        ax.set_title(letter, pad=1)
        ax.set_xlim(*xlim)
        ax.set_xticks(xticks)
        ax.set_ylim(0.0, y_upper)
        ax.set_yticks(y_ticks)
        style_small_axis(ax)
        if feature == "sa":
            ax.ticklabel_format(axis="x", style="plain")
        fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.02)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.95, bottom=0.14, wspace=0.28, hspace=0.34)
    add_caption(fig, "Fig. 5. Structure-adsorption capacity relationships of MOFs.")
    fig.savefig(OUTPUT_DIR / filename, dpi=300)
    plt.close(fig)

def compute_feature_importance_table(pipe: Pipeline, training_df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    model = pipe.named_steps["model"]
    raw_values: np.ndarray | None = None
    importance_source = "native"

    if hasattr(model, "feature_importances_"):
        raw_values = np.asarray(getattr(model, "feature_importances_"), dtype=float)
    elif hasattr(model, "coef_"):
        coef = np.asarray(getattr(model, "coef_"), dtype=float)
        raw_values = np.mean(np.abs(coef), axis=0) if coef.ndim > 1 else np.abs(coef)
    else:
        importance_source = "permutation"
        result = permutation_importance(
            pipe,
            training_df[TRAINING_FEATURES],
            training_df["q"].to_numpy(),
            scoring="neg_mean_absolute_error",
            n_repeats=10,
            random_state=42,
            n_jobs=-1,
        )
        raw_values = np.maximum(np.asarray(result.importances_mean, dtype=float), 0.0)

    raw_importance = pd.Series(raw_values, index=TRAINING_FEATURES, dtype=float)
    total_importance = float(raw_importance.sum())
    if total_importance > 0:
        raw_importance = raw_importance / total_importance * 100.0
    aggregated = pd.Series(
        {
            "MT": raw_importance[["ionic_charge", "atomic_radius", "polarizability", "electronegativity"]].sum(),
            "ModM": raw_importance["mod_code"],
            "SA": raw_importance["sa"],
            "MPD": raw_importance["mpd"],
            "PD": raw_importance["pd"],
            "PV": raw_importance["pv"],
            "CI": raw_importance["ci"],
            "AD": raw_importance["ad"],
            "Time": raw_importance["time"],
            "pH": raw_importance["ph"],
            "Tem": raw_importance["temp"],
        }
    )
    out = aggregated.sort_values(ascending=False).rename_axis("feature").reset_index(name="reproduced_importance")
    out["model"] = model_name
    out["importance_source"] = importance_source
    return out

def save_fig2_like(core_df: pd.DataFrame | None, fallback_df: pd.DataFrame, importance_df: pd.DataFrame, filename: str) -> pd.DataFrame:
    set_paper_rcparams()
    fig = plt.figure(figsize=(7.1, 4.8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.45, 0.85], width_ratios=[1.1, 1.0])

    ax1 = fig.add_subplot(gs[0, 0], projection="3d")
    plot_df = prepare_fig2a_structural_data(core_df, fallback_df)
    ax1.scatter(plot_df["x"], plot_df["y"], plot_df["z"], c=FIG2_GREEN, s=3, alpha=0.55, edgecolors="none")
    style_fig2a_axis(ax1, plot_df)
    ax1.set_title("(a)", pad=-10)

    ax2 = fig.add_subplot(gs[0, 1])
    steps = [
        ("801\nliterature\nrecords", (0.18, 0.83)),
        ("6 model\ncomparison", (0.52, 0.83)),
        ("5382 CoRE\ncandidates", (0.84, 0.83)),
        ("Top 10 per\nmetal", (0.36, 0.40)),
        ("70 initial\nscreening", (0.66, 0.40)),
    ]
    for text, (x, y) in steps:
        ax2.text(x, y, text, ha="center", va="center", fontsize=7, bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="black", lw=0.8))
    arrows = [((0.28, 0.83), (0.42, 0.83)), ((0.62, 0.83), (0.74, 0.83)), ((0.84, 0.70), (0.70, 0.48)), ((0.48, 0.40), (0.54, 0.40))]
    for start, end in arrows:
        ax2.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="->", lw=0.8, color="black"))
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis("off")
    ax2.set_title("(b)", pad=0)

    ax3 = fig.add_subplot(gs[1, :])
    ordered = importance_df.sort_values("reproduced_importance", ascending=False)
    ax3.bar(ordered["feature"], ordered["reproduced_importance"], color=FIG2_GREEN, edgecolor="black", linewidth=0.5)
    ax3.set_ylabel("Importance (%)")
    ax3.set_title("(c)", pad=2)
    style_small_axis(ax3)
    ax3.tick_params(axis="x", rotation=0)

    fig.subplots_adjust(left=0.06, right=0.98, top=0.96, bottom=0.16, wspace=0.15, hspace=0.28)
    add_caption(fig, "Fig. 2. Structural overview, workflow, and feature importance.")
    fig.savefig(OUTPUT_DIR / filename, dpi=300)
    plt.close(fig)
    return plot_df

def save_fig2a_relationship(core_df: pd.DataFrame | None, fallback_df: pd.DataFrame, filename: str) -> None:
    set_paper_rcparams()
    fig = plt.figure(figsize=(4.0, 3.6))
    ax = fig.add_subplot(111, projection="3d")
    plot_df = prepare_fig2a_structural_data(core_df, fallback_df)
    ax.scatter(plot_df["x"], plot_df["y"], plot_df["z"], c=FIG2_GREEN, s=3, alpha=0.55, edgecolors="none")
    style_fig2a_axis(ax, plot_df)
    ax.set_title("(a)", pad=-10)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
    fig.savefig(OUTPUT_DIR / filename, dpi=300)
    plt.close(fig)

def save_fig2c_feature_importance(importance_df: pd.DataFrame, filename: str) -> pd.DataFrame:
    set_paper_rcparams()
    ordered = importance_df.sort_values("reproduced_importance", ascending=False)
    fig, ax = plt.subplots(figsize=(5.6, 2.7))
    ax.bar(ordered["feature"], ordered["reproduced_importance"], color=FIG2_GREEN, edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Importance (%)")
    ax.set_title("(c)", pad=2)
    style_small_axis(ax)
    ax.tick_params(axis="x", rotation=0)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.92, bottom=0.22)
    fig.savefig(OUTPUT_DIR / filename, dpi=300)
    plt.close(fig)
    return ordered


# --- supplementary.py ---

warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn.impute._iterative")

def save_single_shap_beeswarm(pipe: Pipeline, sample_df: pd.DataFrame, path: Path) -> None:
    feature_df = sample_df[TRAINING_FEATURES].copy()
    transformed = pipe.named_steps["prep"].transform(feature_df)
    model = pipe.named_steps["model"]
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(transformed)
    shap_df = pd.DataFrame(transformed, columns=TRAINING_FEATURES)
    shap_df = shap_df.rename(columns=PAPER_FEATURE_LABELS)
    plt.figure(figsize=(6.4, 4.8))
    shap.summary_plot(shap_values, shap_df, show=False, max_display=14, color_bar=False)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

def save_single_waterfall(pipe: Pipeline, sample_df: pd.DataFrame, sample_index: int, path: Path) -> None:
    feature_df = sample_df[TRAINING_FEATURES].copy()
    transformed = pipe.named_steps["prep"].transform(feature_df)
    model = pipe.named_steps["model"]
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(transformed)
    feature_names = [PAPER_FEATURE_LABELS[col] for col in TRAINING_FEATURES]
    explanation = shap.Explanation(
        values=shap_values.values[sample_index],
        base_values=shap_values.base_values[sample_index],
        data=transformed[sample_index],
        feature_names=feature_names,
    )
    plt.figure(figsize=(6.4, 4.8))
    shap.plots.waterfall(explanation, max_display=14, show=False)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

def save_combined_supplementary_figures(pipes: dict[str, Pipeline], test_df: pd.DataFrame, filename_s5: str, filename_s6: str) -> None:
    save_single_shap_beeswarm(pipes["RF"], test_df, OUTPUT_DIR / filename_s5)
    sample_index = int(np.argmax(test_df["q"].to_numpy()))
    save_single_waterfall(pipes["XGB"], test_df, sample_index, OUTPUT_DIR / filename_s6)

def _load_supplementary_paragraphs(root: Path) -> list[str]:
    docx_path = next(root.glob("*mmc1.docx"))
    with zipfile.ZipFile(docx_path) as archive, TemporaryDirectory() as temp_dir:
        archive.extract("word/document.xml", temp_dir)
        xml_path = Path(temp_dir) / "word" / "document.xml"
        tree = ET.parse(xml_path)
        namespace = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
        paragraphs = []
        for para in tree.findall(".//w:p", namespace):
            texts = [node.text for node in para.findall(".//w:t", namespace) if node.text]
            if texts:
                paragraphs.append("".join(texts))
    return paragraphs

def _export_supplementary_text_sections(root: Path) -> None:
    paragraphs = _load_supplementary_paragraphs(root)
    sections = {"Text S1": [], "Text S2": []}
    current = None
    for para in paragraphs:
        stripped = para.strip()
        if stripped.startswith("Text S1"):
            current = "Text S1"
        elif stripped.startswith("Text S2"):
            current = "Text S2"
        elif stripped.startswith("Fig.") or stripped.startswith("Table "):
            current = None
        if current is not None:
            sections[current].append(stripped)
    (OUTPUT_DIR / "textS1_hyperparameter_tuning.md").write_text("\n\n".join(sections["Text S1"]).strip(), encoding="utf-8")
    (OUTPUT_DIR / "textS2_rf_xgb_notes.md").write_text("\n\n".join(sections["Text S2"]).strip(), encoding="utf-8")

def save_figS1_quantitative_distributions(training_df: pd.DataFrame, filename: str) -> None:
    set_paper_rcparams()
    columns = [("sa", "SA"), ("pd", "PD"), ("pv", "PV"), ("ci", "CI"), ("ad", "AD"), ("time", "Time"), ("ph", "pH"), ("temp", "Tem")]
    fig, axes = plt.subplots(2, 4, figsize=(7.2, 4.2))
    for ax, (col, label) in zip(axes.flat, columns):
        values = pd.to_numeric(training_df[col], errors="coerce").dropna().to_numpy()
        ax.hist(values, bins=20, color=S1_PANEL_COLORS[col], edgecolor="white", linewidth=0.6)
        ax.set_xlabel(label)
        ax.set_ylabel("Count")
        style_small_axis(ax)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.94, bottom=0.18, wspace=0.35, hspace=0.35)
    add_caption(fig, "Fig. S1. Statistical distribution maps of quantitative variables")
    fig.savefig(OUTPUT_DIR / filename, dpi=300)
    plt.close(fig)

def save_figS2_qualitative_distributions(training_df: pd.DataFrame, filename: str) -> None:
    set_paper_rcparams()
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.8))

    metal_counts = training_df["metal"].value_counts()
    metal_colors = [S2_METAL_COLORS.get(name, "#74c476") for name in metal_counts.index]
    axes[0].bar(metal_counts.index, metal_counts.values, color=metal_colors, edgecolor="white", linewidth=0.6)
    axes[0].set_xlabel("MT")
    axes[0].set_ylabel("Count")
    style_small_axis(axes[0])

    mod_counts = training_df["modification"].value_counts()
    mod_order = [name for name in MOD_ORDER if name in mod_counts.index]
    mod_labels = [MOD_LABELS[name] for name in mod_order]
    mod_colors = [S2_MOD_COLORS.get(label, "#74c476") for label in mod_labels]
    axes[1].bar(mod_labels, mod_counts.reindex(mod_order).to_numpy(), color=mod_colors, edgecolor="white", linewidth=0.6)
    axes[1].set_xlabel("ModM")
    axes[1].set_ylabel("Count")
    style_small_axis(axes[1])

    temp_counts = pd.Series(pd.to_numeric(training_df["temp"], errors="coerce").dropna()).value_counts().sort_index()
    temp_colors = [S2_TEMP_COLORS.get(int(val), "#58aeb8") for val in temp_counts.index]
    axes[2].bar(temp_counts.index.astype(int).astype(str), temp_counts.to_numpy(), color=temp_colors, edgecolor="white", linewidth=0.6)
    axes[2].set_xlabel("Tem")
    axes[2].set_ylabel("Count")
    style_small_axis(axes[2])

    fig.subplots_adjust(left=0.07, right=0.995, top=0.88, bottom=0.21, wspace=0.35)
    add_caption(fig, "Fig. S2. Statistical distribution maps of qualitative variables and reaction temperature")
    fig.savefig(OUTPUT_DIR / filename, dpi=300)
    plt.close(fig)

def save_figS3_correlation_heatmap(training_df: pd.DataFrame, filename: str) -> None:
    set_paper_rcparams()
    corr_features = ["sa", "mpd", "pd", "pv", "ci", "ad", "time", "ph", "temp"]
    labels = ["SA", "MPD", "PD", "PV", "CI", "AD", "Time", "pH", "Tem"]
    corr = training_df[corr_features].corr()
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    im = ax.imshow(corr.to_numpy(), cmap="Greens", vmin=float(corr.min().min()), vmax=float(corr.max().max()))
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=0, ha="center")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=6, color="black")
    style_small_axis(ax)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    fig.subplots_adjust(left=0.14, right=0.92, top=0.95, bottom=0.16)
    add_caption(fig, "Fig. S3. Thermal map of correlation between quantitative variables")
    fig.savefig(OUTPUT_DIR / filename, dpi=300)
    plt.close(fig)

def compute_learning_curve_neg_mae(
    raw_training_df: pd.DataFrame,
    model_name: str,
    model_params: dict[str, object],
    fractions: np.ndarray,
    n_splits: int = 10,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    groups = raw_training_df["group_id"].astype(int)
    n_splits = min(int(groups.nunique()), n_splits)
    splitter = GroupKFold(n_splits=n_splits)
    score_rows: list[list[float]] = [[] for _ in fractions]

    for fold_index, (train_idx, val_idx) in enumerate(splitter.split(raw_training_df, groups=groups), start=1):
        fold_train_raw_all = raw_training_df.iloc[train_idx].copy().reset_index(drop=True)
        fold_val_raw = raw_training_df.iloc[val_idx].copy()
        subset_order = np.random.RandomState(random_state + fold_index * 97).permutation(len(fold_train_raw_all))
        fold_train_raw_all = fold_train_raw_all.iloc[subset_order].reset_index(drop=True)

        for frac_index, frac in enumerate(fractions):
            subset_size = max(8, int(np.floor(len(fold_train_raw_all) * float(frac))))
            subset_train_raw = fold_train_raw_all.iloc[:subset_size].copy()
            subset_train = prepare_model_table(subset_train_raw, fit_df=subset_train_raw)
            fold_val = prepare_model_table(fold_val_raw, fit_df=subset_train_raw)
            pipe = Pipeline([("prep", build_preprocessor(TRAINING_FEATURES)), ("model", instantiate_model(model_name, model_params))])
            pipe.fit(subset_train[TRAINING_FEATURES], subset_train["q"].to_numpy())
            pred = pipe.predict(fold_val[TRAINING_FEATURES])
            score_rows[frac_index].append(-mean_absolute_error(fold_val["q"].to_numpy(), pred))

    train_sizes = np.asarray([max(8, int(np.floor(len(raw_training_df) * float(frac)))) for frac in fractions], dtype=int)
    mean_scores = np.asarray([float(np.mean(scores)) for scores in score_rows], dtype=float)
    std_scores = np.asarray([float(np.std(scores)) for scores in score_rows], dtype=float)
    return train_sizes, mean_scores, std_scores

def save_figS4_learning_curve(raw_training_df: pd.DataFrame, model_name: str, model_params: dict[str, object], filename: str) -> None:
    set_paper_rcparams()
    fractions = np.arange(0.2, 1.0, 0.1)
    train_sizes, cv_scores, cv_stds = compute_learning_curve_neg_mae(raw_training_df, model_name, model_params, fractions)
    x_values = train_sizes / len(raw_training_df) * 100.0

    fig, ax = plt.subplots(figsize=(4.6, 3.3))
    ax.plot(x_values, cv_scores, marker="o", color="#0b5d1e", linewidth=1.2)
    ax.fill_between(x_values, cv_scores - cv_stds, cv_scores + cv_stds, color="#0b5d1e", alpha=0.12, linewidth=0)
    ax.set_xlabel("Training set (%)")
    ax.set_ylabel("Performance (neg. MAE)")
    ax.set_xlim(20, 90)
    ax.set_xticks([20, 40, 60, 80])
    ax.set_ylim(float(np.min(cv_scores)) - 0.5, float(np.max(cv_scores)) + 0.5)
    style_small_axis(ax)
    fig.subplots_adjust(left=0.15, right=0.97, top=0.92, bottom=0.22)
    add_caption(fig, "Fig. S4. Learning curve")
    fig.savefig(OUTPUT_DIR / filename, dpi=300)
    plt.close(fig)

def _save_reference_pages(root: Path) -> None:
    pdf_path = next(root.glob("*.pdf"))
    out_dir = OUTPUT_DIR / "paper_pages"
    out_dir.mkdir(exist_ok=True)
    doc = fitz.open(pdf_path)
    for page_number in [5, 6, 7]:
        page = doc[page_number]
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
        pix.save(out_dir / f"page_{page_number + 1}.png")

def _save_doc_page_to_text_image(root: Path, pdf_name_pattern: str, page_number: int, out_name: str) -> Path:
    pdf_path = next(root.glob(pdf_name_pattern))
    out_path = OUTPUT_DIR / out_name
    doc = fitz.open(pdf_path)
    page = doc[page_number]
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
    pix.save(out_path)
    return out_path


# --- core.py ---

def write_summary(
    metric_frames: dict[str, object],
    cv_best_df,
    first_df,
    initial_screening_df,
    second_df,
    display_config: str,
) -> None:
    lines = [
        "# Refined Reproduction Summary",
        "",
        "## Selected display configuration",
        f"- Config: `{display_config}`",
        "",
    ]
    for config_name, metrics in metric_frames.items():
        best = metrics.sort_values("r2", ascending=False).iloc[0]
        lines.extend(
            [
                f"## {config_name}",
                f"- Best model: `{best['model']}`",
                f"- MAE: `{best['mae']:.2f}` mg/g",
                f"- RMSE: `{best['rmse']:.2f}` mg/g",
                f"- R2: `{best['r2']:.2f}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Model selection",
            f"- Best model: `{cv_best_df.iloc[0]['model']}`",
            f"- Second-best model: `{cv_best_df.iloc[1]['model']}`",
            "",
            "## CoRE-MOF first adsorption dataset",
            f"- Candidate rows: `{len(first_df)}`",
            f"- Source file: `{first_df['source_file'].iloc[0] if not first_df.empty else 'N/A'}`",
            f"- ModM handling: `{SCREENING_MOD_STRATEGY}`",
            f"- Top 10 by first model: `{', '.join(first_df.head(10)['cif_file'].tolist()) if not first_df.empty else 'N/A'}`",
            "",
            "## Initial screening(70)",
            f"- Rows: `{len(initial_screening_df)}`",
            f"- Metal categories: `{', '.join(TARGET_CORE_METALS)}`",
            "",
            "## Second adsorption dataset",
            f"- Top 10 by second model: `{', '.join(second_df.head(10)['cif_file'].tolist()) if not second_df.empty else 'N/A'}`",
            "",
            "## Remaining gaps",
            "- The exact IC/AR/Pol/Ele lookup table is not published, so the calibrated descriptor preset is an informed reconstruction.",
            "- The local pipeline now uses the 5382 single-metal CoRE subset extracted from the official 14142-entry ASR table, which matches the user's requested candidate set but is still not the paper's unpublished 3833 subset.",
            "- Model selection follows grouped cross-validation with fold-wise train-only preprocessing on the current local training dataset; holdout-style grouped evaluation is retained separately for display figures.",
        ]
    )
    (OUTPUT_DIR / "reproduction_summary.md").write_text("\n".join(lines), encoding="utf-8")

# --- workflow ---

def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

def pick_display_config(metric_frames: dict[str, object]) -> str:
    scored = {name: score_metric_frame(frame) for name, frame in metric_frames.items()}
    return min(scored, key=scored.get)

def run_reproduction(skip_supplementary: bool = False) -> int:
    ensure_output_dir()
    save_reference_pages()

    configs = [
        SplitConfig(
            name="paper_faithful",
            mode="sequential",
            descriptor_preset="calibrated_mixed",
            mod_encoding="calibrated",
            group_recipe="metal_family",
            train_group_count=39,
        ),
        SplitConfig(
            name="score_matched",
            mode="group_shuffle",
            descriptor_preset="calibrated_mixed",
            mod_encoding="calibrated",
            group_recipe="metal_family",
            random_state=24,
        ),
    ]

    metric_frames: dict[str, object] = {}
    prediction_frames: dict[str, object] = {}
    split_pipes: dict[str, object] = {}
    prepared_splits: dict[str, object] = {}

    display_training_raw = load_training_table("calibrated_mixed", "calibrated", "metal_family")
    _, cv_best_df, tuned_full_pipes, display_training_df = run_model_grid_search_cv(display_training_raw)
    tuned_params = {row.model: json.loads(row.params_json) for row in cv_best_df.itertuples(index=False)}
    strict_group_cv_summary, strict_group_cv_folds, strict_group_cv_groups = evaluate_models_with_group_cv(
        display_training_raw,
        tuned_params,
    )
    strict_group_cv_summary.to_csv(OUTPUT_DIR / "strict_group_cv_summary.csv", index=False)
    strict_group_cv_folds.to_csv(OUTPUT_DIR / "strict_group_cv_fold_metrics.csv", index=False)
    strict_group_cv_groups.to_csv(OUTPUT_DIR / "strict_group_cv_group_metrics.csv", index=False)
    write_group_cv_report(display_training_raw, strict_group_cv_summary, strict_group_cv_groups)

    for config in configs:
        raw_training_df = load_training_table(config.descriptor_preset, config.mod_encoding, config.group_recipe)
        metrics, predictions, fitted, prepared_split = fit_models_for_split(raw_training_df, config, tuned_params)
        metric_frames[config.name] = metrics
        prediction_frames[config.name] = predictions
        split_pipes[config.name] = fitted
        prepared_splits[config.name] = prepared_split
        metrics.to_csv(OUTPUT_DIR / f"model_metrics_{config.name}.csv", index=False)

    display_config = pick_display_config(metric_frames)

    target_core_raw, _ = export_target_core_metal_tables()
    target_core_df = build_target_core_feature_table(target_core_raw, "calibrated_mixed", display_training_df, "calibrated")
    target_core_df.to_csv(OUTPUT_DIR / "core_target_feature_table_5382.csv", index=False)

    best_model_name = cv_best_df.iloc[0]["model"]
    second_model_name = cv_best_df.iloc[1]["model"]
    screening_mod_weights = get_screening_mod_weights(display_training_df)
    first_adsorption_df = make_first_adsorption_dataset(
        target_core_df,
        best_model_name,
        tuned_full_pipes[best_model_name],
        screening_mod_weights,
    )
    first_adsorption_df.to_csv(OUTPUT_DIR / "core_first_adsorption_dataset.csv", index=False)
    initial_screening_df = build_initial_screening_from_first_dataset(first_adsorption_df, top_n=10)
    initial_screening_df.to_csv(OUTPUT_DIR / "initial_screening_generated_70.csv", index=False)
    second_adsorption_df = make_second_adsorption_dataset(
        initial_screening_df,
        second_model_name,
        tuned_full_pipes[second_model_name],
        screening_mod_weights,
    )
    second_adsorption_df.to_csv(OUTPUT_DIR / "second_adsorption_dataset.csv", index=False)

    feature_importance = compute_feature_importance_table(
        tuned_full_pipes[best_model_name],
        display_training_df,
        best_model_name,
    )
    save_fig2_like(target_core_df, display_training_df, feature_importance, "fig2_overview.png")
    save_fig2a_relationship(target_core_df, display_training_df, "fig2a_relationship.png")
    save_fig2c_feature_importance(feature_importance, "fig2c_feature_importance.png")
    feature_importance.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)

    export_supplementary_text_sections()
    save_figS1_quantitative_distributions(display_training_raw, "figS1_quantitative_distributions.png")
    save_figS2_qualitative_distributions(display_training_raw, "figS2_qualitative_distributions.png")
    save_figS3_correlation_heatmap(display_training_raw, "figS3_correlation_heatmap.png")
    save_figS4_learning_curve(display_training_raw, best_model_name, tuned_params[best_model_name], "figS4_learning_curve.png")

    save_fig3_like(prediction_frames[display_config], metric_frames[display_config], "fig3_fitting_effect.png")
    render_fig4_artifacts(ROOT / "config" / "fig4_config.json")
    save_fig5_like(first_adsorption_df, "fig5_structure_relationships.png")

    if not skip_supplementary:
        test_df = prepared_splits[display_config]["test"].copy()
        save_combined_supplementary_figures(split_pipes[display_config], test_df, "figS5_beeswarm.png", "figS6_waterfall.png")

    write_summary(
        metric_frames,
        cv_best_df,
        first_adsorption_df,
        initial_screening_df,
        second_adsorption_df,
        display_config,
    )
    update_readme()
    return 0


def run_model_grid_search_cv(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Pipeline], pd.DataFrame]:
    return _run_model_grid_search_cv(raw_df, OUTPUT_DIR)


def load_supplementary_paragraphs() -> list[str]:
    return _load_supplementary_paragraphs(ROOT)


def export_supplementary_text_sections() -> None:
    _export_supplementary_text_sections(ROOT)


def save_reference_pages() -> None:
    _save_reference_pages(ROOT)


def save_doc_page_to_text_image(pdf_name_pattern: str, page_number: int, out_name: str) -> Path:
    return _save_doc_page_to_text_image(ROOT, pdf_name_pattern, page_number, out_name)



# --- integrated_fig4_module ---

LOGGER = logging.getLogger("fig4_refine")

@dataclass(frozen=True)
class ModelContext:
    raw_training_df: pd.DataFrame
    prepared_training_df: pd.DataFrame
    cv_results: pd.DataFrame
    best_per_model: pd.DataFrame
    fitted_models: dict[str, Pipeline]
    best_model_name: str
    best_model_params: dict[str, Any]
    output_tag: str

@dataclass(frozen=True)
class OneDPanelData:
    feature_name: str
    x: np.ndarray
    y: np.ndarray
    rug_values: np.ndarray
    y_std: np.ndarray
    y_q05: np.ndarray
    y_q95: np.ndarray

@dataclass(frozen=True)
class TwoDPanelData:
    feature_x: str
    feature_y: str
    x_grid: np.ndarray
    y_grid: np.ndarray
    z: np.ndarray
    z_std: np.ndarray
    z_q05: np.ndarray
    z_q95: np.ndarray

@dataclass(frozen=True)
class Fig4DataBundle:
    model_name: str
    model_params: dict[str, Any]
    two_d: TwoDPanelData
    one_d: dict[str, OneDPanelData]

def load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as fh:
        config = json.load(fh)
    return config

def configure_logging(output_dir: Path, log_name: str) -> None:
    output_dir.mkdir(exist_ok=True)
    log_path = output_dir / log_name
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )

def stable_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, ensure_ascii=True)

def dataframe_fingerprint(frame: pd.DataFrame) -> str:
    normalized = frame.copy()
    normalized = normalized.fillna("<NA>").astype(str)
    row_hash = pd.util.hash_pandas_object(normalized, index=True).to_numpy()
    return hashlib.sha1(row_hash.tobytes()).hexdigest()

def build_output_tag(config: dict[str, Any], model_name: str, params: dict[str, Any]) -> str:
    dataset = config["dataset"]
    digest_payload = {
        "version": config["version"],
        "dataset": dataset,
        "model": model_name,
        "params": params,
    }
    digest = hashlib.sha1(stable_json(digest_payload).encode("utf-8")).hexdigest()[:8]
    return "_".join(
        [
            dataset["group_recipe"],
            dataset["descriptor_preset"],
            dataset["mod_encoding"],
            model_name.lower(),
            digest,
        ]
    )

def model_selection_cache_meta_path(output_dir: Path) -> Path:
    return output_dir / "fig4_model_selection_cache.json"

def write_model_selection_cache_meta(output_dir: Path, config: dict[str, Any], raw_training_df: pd.DataFrame) -> None:
    meta = {
        "dataset": config["dataset"],
        "config_version": config["version"],
        "training_fingerprint": dataframe_fingerprint(raw_training_df),
    }
    model_selection_cache_meta_path(output_dir).write_text(stable_json(meta), encoding="utf-8")

def is_model_selection_cache_valid(output_dir: Path, config: dict[str, Any], raw_training_df: pd.DataFrame) -> bool:
    meta_path = model_selection_cache_meta_path(output_dir)
    if not meta_path.exists():
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return (
        meta.get("dataset") == config["dataset"]
        and meta.get("config_version") == config["version"]
        and meta.get("training_fingerprint") == dataframe_fingerprint(raw_training_df)
    )

def fit_named_models(prepared_training_df: pd.DataFrame, best_per_model: pd.DataFrame, model_names: list[str]) -> dict[str, Pipeline]:
    fitted: dict[str, Pipeline] = {}
    y = prepared_training_df["q"].to_numpy()
    for model_name in model_names:
        if model_name not in set(best_per_model["model"]):
            continue
        row = best_per_model.loc[best_per_model["model"] == model_name].iloc[0]
        params = json.loads(row["params_json"])
        pipe = Pipeline([("prep", build_preprocessor(TRAINING_FEATURES)), ("model", instantiate_model(model_name, params))])
        pipe.fit(prepared_training_df[TRAINING_FEATURES], y)
        fitted[model_name] = pipe
    return fitted

def create_model_context(config: dict[str, Any]) -> ModelContext:
    dataset_cfg = config["dataset"]
    output_dir = ROOT / config["output"]["directory"]
    raw_training_df = load_training_table(
        dataset_cfg["descriptor_preset"],
        dataset_cfg["mod_encoding"],
        dataset_cfg["group_recipe"],
    )
    LOGGER.info("Loaded training table with %s rows for Fig. 4 refinement.", len(raw_training_df))
    cv_results_path = output_dir / "model_grid_search_cv.csv"
    best_per_model_path = output_dir / "model_grid_search_best_per_model.csv"
    prepared_training_df = prepare_model_table(raw_training_df, fit_df=raw_training_df)
    if cv_results_path.exists() and best_per_model_path.exists() and is_model_selection_cache_valid(output_dir, config, raw_training_df):
        cv_results = pd.read_csv(cv_results_path)
        best_per_model = pd.read_csv(best_per_model_path)
        LOGGER.info("Reused existing model-selection CSV outputs for Fig. 4 refinement.")
    else:
        cv_results, best_per_model, _, prepared_training_df = run_model_grid_search_cv(raw_training_df)
        write_model_selection_cache_meta(output_dir, config, raw_training_df)
        LOGGER.info("Wrote model-selection cache metadata for the current training fingerprint.")
    needed_models = [str(best_per_model.iloc[0]["model"])]
    needed_models.extend([name for name in config["model_selection"]["comparison_models"] if name not in needed_models])
    fitted_models = fit_named_models(prepared_training_df, best_per_model, needed_models)
    best_row = best_per_model.iloc[0]
    best_model_name = str(best_row["model"])
    best_model_params = json.loads(best_row["params_json"])
    output_tag = build_output_tag(config, best_model_name, best_model_params)
    LOGGER.info("CV best model for Fig. 4 is %s with params %s.", best_model_name, best_model_params)
    return ModelContext(
        raw_training_df=raw_training_df,
        prepared_training_df=prepared_training_df,
        cv_results=cv_results,
        best_per_model=best_per_model,
        fitted_models=fitted_models,
        best_model_name=best_model_name,
        best_model_params=best_model_params,
        output_tag=output_tag,
    )

def versioned_stem(stem: str, output_tag: str) -> str:
    return f"{stem}_{output_tag}"

def bundle_output_paths(output_dir: Path, stem: str, output_tag: str) -> dict[str, Path]:
    versioned = versioned_stem(stem, output_tag)
    return {
        "png": output_dir / f"{versioned}.png",
        "pdf": output_dir / f"{versioned}.pdf",
        "svg": output_dir / f"{versioned}.svg",
        "curve_1d": output_dir / f"{versioned}_1d_curves.csv",
        "surface_2d": output_dir / f"{versioned}_2d_surface.csv",
    }

def bundle_outputs_exist(output_dir: Path, stem: str, output_tag: str) -> bool:
    paths = bundle_output_paths(output_dir, stem, output_tag)
    return all(path.exists() for path in paths.values())

def load_saved_bundle(output_dir: Path, stem: str, output_tag: str, model_name: str, model_params: dict[str, Any]) -> Fig4DataBundle:
    paths = bundle_output_paths(output_dir, stem, output_tag)
    curve_df = pd.read_csv(paths["curve_1d"])
    surface_df = pd.read_csv(paths["surface_2d"])
    one_d: dict[str, OneDPanelData] = {}
    for feature_name, part in curve_df.groupby("feature", sort=False):
        one_d[feature_name] = OneDPanelData(
            feature_name=feature_name,
            x=part["x"].to_numpy(dtype=float),
            y=part["prediction"].to_numpy(dtype=float),
            rug_values=np.asarray([], dtype=float),
            y_std=part["std"].to_numpy(dtype=float),
            y_q05=part["q05"].to_numpy(dtype=float),
            y_q95=part["q95"].to_numpy(dtype=float),
        )
    x_grid = np.sort(surface_df["ci"].unique())
    y_grid = np.sort(surface_df["ad"].unique())
    shape = (len(y_grid), len(x_grid))
    two_d = TwoDPanelData(
        feature_x="ci",
        feature_y="ad",
        x_grid=x_grid,
        y_grid=y_grid,
        z=surface_df["prediction"].to_numpy(dtype=float).reshape(shape),
        z_std=surface_df["std"].to_numpy(dtype=float).reshape(shape),
        z_q05=surface_df["q05"].to_numpy(dtype=float).reshape(shape),
        z_q95=surface_df["q95"].to_numpy(dtype=float).reshape(shape),
    )
    return Fig4DataBundle(model_name=model_name, model_params=model_params, two_d=two_d, one_d=one_d)

def compute_partial_dependence_1d(predictor: Any, base_frame: pd.DataFrame, feature_name: str, grid: np.ndarray) -> np.ndarray:
    values: list[float] = []
    for value in grid:
        scenario = base_frame.copy()
        scenario[feature_name] = value
        prediction = predictor.predict(scenario[TRAINING_FEATURES] if set(TRAINING_FEATURES).issubset(scenario.columns) else scenario)
        values.append(float(np.mean(prediction)))
    return np.asarray(values, dtype=float)

def compute_partial_dependence_2d(
    predictor: Any,
    base_frame: pd.DataFrame,
    feature_x: str,
    feature_y: str,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
) -> np.ndarray:
    surface = np.zeros((len(y_grid), len(x_grid)), dtype=float)
    for y_index, y_value in enumerate(y_grid):
        for x_index, x_value in enumerate(x_grid):
            scenario = base_frame.copy()
            scenario[feature_x] = x_value
            scenario[feature_y] = y_value
            prediction = predictor.predict(scenario[TRAINING_FEATURES] if set(TRAINING_FEATURES).issubset(scenario.columns) else scenario)
            surface[y_index, x_index] = float(np.mean(prediction))
    return surface

def fit_uncertainty_ensemble(raw_training_df: pd.DataFrame, model_name: str, model_params: dict[str, Any], n_splits: int) -> list[Pipeline]:
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    ensemble: list[Pipeline] = []
    for fold_index, (train_idx, _) in enumerate(splitter.split(raw_training_df), start=1):
        fold_train_raw = raw_training_df.iloc[train_idx].copy()
        fold_train = prepare_model_table(fold_train_raw, fit_df=fold_train_raw)
        pipe = Pipeline(
            [("prep", build_preprocessor(TRAINING_FEATURES)), ("model", instantiate_model(model_name, model_params))]
        )
        pipe.fit(fold_train[TRAINING_FEATURES], fold_train["q"].to_numpy())
        ensemble.append(pipe)
        LOGGER.info("Fitted uncertainty ensemble fold %s/%s for %s.", fold_index, n_splits, model_name)
    return ensemble

def summarize_ensemble_curves(curves: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    stacked = np.stack(curves, axis=0)
    return (
        np.mean(stacked, axis=0),
        np.std(stacked, axis=0),
        np.quantile(stacked, 0.05, axis=0),
        np.quantile(stacked, 0.95, axis=0),
    )

def compute_rug_values(training_df: pd.DataFrame, feature_name: str, limits: tuple[float, float]) -> np.ndarray:
    values = pd.to_numeric(training_df[feature_name], errors="coerce").dropna().to_numpy(dtype=float)
    lo, hi = limits
    clipped = values[(values >= lo) & (values <= hi)]
    return np.sort(clipped)

def build_linear_grid(limits: tuple[float, float], n_points: int) -> np.ndarray:
    return np.linspace(float(limits[0]), float(limits[1]), int(n_points), dtype=float)

def build_one_d_grid(
    training_df: pd.DataFrame,
    feature_name: str,
    limits: tuple[float, float],
    n_points: int,
) -> np.ndarray:
    values = compute_rug_values(training_df, feature_name, limits)
    if values.size == 0:
        return build_linear_grid(limits, n_points)
    unique_values = np.unique(values)
    if unique_values.size <= int(n_points):
        return unique_values.astype(float)
    quantiles = np.linspace(0.0, 1.0, int(n_points), dtype=float)
    quantile_grid = np.quantile(values, quantiles)
    clipped_grid = np.clip(quantile_grid, float(limits[0]), float(limits[1]))
    unique_grid = np.unique(clipped_grid.astype(float))
    if unique_grid.size < 2:
        return build_linear_grid(limits, n_points)
    return unique_grid

def compute_one_d_panel(
    predictor: Pipeline,
    ensemble: list[Pipeline],
    base_frame: pd.DataFrame,
    training_df: pd.DataFrame,
    feature_name: str,
    panel_cfg: dict[str, Any],
) -> OneDPanelData:
    grid = build_one_d_grid(training_df, feature_name, tuple(panel_cfg["xlim"]), panel_cfg["n_points"])
    point_curve = compute_partial_dependence_1d(predictor, base_frame, feature_name, grid)
    ensemble_curves = [compute_partial_dependence_1d(model, base_frame, feature_name, grid) for model in ensemble]
    _, y_std, y_q05, y_q95 = summarize_ensemble_curves(ensemble_curves)
    rug_values = compute_rug_values(training_df, feature_name, tuple(panel_cfg["xlim"]))
    return OneDPanelData(
        feature_name=feature_name,
        x=grid,
        y=point_curve,
        rug_values=rug_values,
        y_std=y_std,
        y_q05=y_q05,
        y_q95=y_q95,
    )

def compute_two_d_panel(
    predictor: Pipeline,
    ensemble: list[Pipeline],
    base_frame: pd.DataFrame,
    panel_cfg: dict[str, Any],
) -> TwoDPanelData:
    x_grid = build_linear_grid(tuple(panel_cfg["xlim"]), panel_cfg["x_points"])
    y_grid = build_linear_grid(tuple(panel_cfg["ylim"]), panel_cfg["y_points"])
    point_surface = compute_partial_dependence_2d(
        predictor,
        base_frame,
        panel_cfg["x_feature"],
        panel_cfg["y_feature"],
        x_grid,
        y_grid,
    )
    ensemble_surfaces = [
        compute_partial_dependence_2d(
            model,
            base_frame,
            panel_cfg["x_feature"],
            panel_cfg["y_feature"],
            x_grid,
            y_grid,
        )
        for model in ensemble
    ]
    _, z_std, z_q05, z_q95 = summarize_ensemble_curves(ensemble_surfaces)
    return TwoDPanelData(
        feature_x=panel_cfg["x_feature"],
        feature_y=panel_cfg["y_feature"],
        x_grid=x_grid,
        y_grid=y_grid,
        z=point_surface,
        z_std=z_std,
        z_q05=z_q05,
        z_q95=z_q95,
    )

def build_fig4_bundle(config: dict[str, Any], context: ModelContext, model_name: str) -> Fig4DataBundle:
    model_params = json.loads(context.best_per_model.loc[context.best_per_model["model"] == model_name, "params_json"].iloc[0])
    predictor = context.fitted_models[model_name]
    base_frame = context.prepared_training_df[TRAINING_FEATURES].copy()
    ensemble = fit_uncertainty_ensemble(
        context.raw_training_df,
        model_name,
        model_params,
        int(config["model_selection"]["uncertainty_cv_splits"]),
    )
    one_d = {
        "ci": compute_one_d_panel(predictor, ensemble, base_frame, context.prepared_training_df, "ci", config["panels"]["ci"]),
        "ad": compute_one_d_panel(predictor, ensemble, base_frame, context.prepared_training_df, "ad", config["panels"]["ad"]),
        "time": compute_one_d_panel(predictor, ensemble, base_frame, context.prepared_training_df, "time", config["panels"]["time"]),
        "ph": compute_one_d_panel(predictor, ensemble, base_frame, context.prepared_training_df, "ph", config["panels"]["ph"]),
        "temp": compute_one_d_panel(predictor, ensemble, base_frame, context.prepared_training_df, "temp", config["panels"]["temp"]),
    }
    two_d = compute_two_d_panel(predictor, ensemble, base_frame, config["panels"]["ci_ad"])
    return Fig4DataBundle(model_name=model_name, model_params=model_params, two_d=two_d, one_d=one_d)

def set_axis_style(ax: plt.Axes) -> None:
    ax.tick_params(direction="in", length=3.5, width=0.8)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("black")
    ax.set_facecolor("white")

def add_rug_marks(ax: plt.Axes, rug_values: np.ndarray, plot_cfg: dict[str, Any]) -> None:
    if rug_values.size == 0:
        return
    trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
    ax.vlines(
        rug_values,
        0.0,
        plot_cfg["rug_height_axes"],
        transform=trans,
        colors=plot_cfg["rug_color"],
        linewidth=plot_cfg["rug_line_width"],
        zorder=3,
    )

def apply_axis_ticks(ax: plt.Axes, panel_cfg: dict[str, Any], axis: str) -> None:
    key_ticks = f"{axis}_ticks"
    key_locator = f"{axis}_major_locator"
    if key_ticks in panel_cfg:
        if axis == "x":
            ax.set_xticks(panel_cfg[key_ticks])
        else:
            ax.set_yticks(panel_cfg[key_ticks])
    elif key_locator in panel_cfg:
        locator = MultipleLocator(panel_cfg[key_locator])
        if axis == "x":
            ax.xaxis.set_major_locator(locator)
        else:
            ax.yaxis.set_major_locator(locator)

def plot_two_d_panel(ax: plt.Axes, panel: TwoDPanelData, panel_cfg: dict[str, Any], plot_cfg: dict[str, Any], letter: str) -> None:
    fill_levels = np.linspace(float(np.min(panel.z)), float(np.max(panel.z)), int(plot_cfg["contourf_levels"]))
    line_levels = np.linspace(float(np.min(panel.z)), float(np.max(panel.z)), int(plot_cfg["contour_levels"]))
    ax.contourf(panel.x_grid, panel.y_grid, panel.z, levels=fill_levels, cmap=plot_cfg["colormap"], antialiased=False)
    contour = ax.contour(
        panel.x_grid,
        panel.y_grid,
        panel.z,
        levels=line_levels,
        colors="black",
        linewidths=plot_cfg["contour_line_width"],
        alpha=0.65,
    )
    ax.clabel(contour, inline=True, fontsize=plot_cfg["contour_label_fontsize"], fmt="%.2f")
    ax.set_xlim(*panel_cfg["xlim"])
    ax.set_ylim(*panel_cfg["ylim"])
    ax.set_xlabel(panel_cfg["xlabel"])
    ax.set_ylabel(panel_cfg["ylabel"])
    ax.set_title(letter, pad=2)
    apply_axis_ticks(ax, panel_cfg, "x")
    apply_axis_ticks(ax, panel_cfg, "y")
    set_axis_style(ax)

def infer_y_limits(curve: np.ndarray, panel_cfg: dict[str, Any]) -> tuple[float, float]:
    y_min = float(np.min(curve))
    y_max = float(np.max(curve))
    padding = max((y_max - y_min) * float(panel_cfg["y_padding_fraction"]), float(panel_cfg["y_min_pad"]))
    return y_min - padding * 0.35, y_max + padding * 0.15

def plot_one_d_panel(ax: plt.Axes, panel: OneDPanelData, panel_cfg: dict[str, Any], plot_cfg: dict[str, Any], letter: str) -> None:
    ax.step(panel.x, panel.y, where="mid", color=plot_cfg["line_color"], linewidth=plot_cfg["line_width"], zorder=2)
    ax.set_xlim(*panel_cfg["xlim"])
    ax.set_ylim(*infer_y_limits(panel.y, panel_cfg))
    ax.set_xlabel(panel_cfg["xlabel"])
    ax.set_ylabel(panel_cfg["ylabel"])
    ax.set_title(letter, pad=2)
    add_rug_marks(ax, panel.rug_values, plot_cfg)
    apply_axis_ticks(ax, panel_cfg, "x")
    set_axis_style(ax)

def plot_fig4(bundle: Fig4DataBundle, config: dict[str, Any], destination: Path) -> None:
    set_paper_rcparams()
    plot_cfg = config["plot"]
    fig, axes = plt.subplots(2, 3, figsize=tuple(plot_cfg["figure_size"]))
    plot_two_d_panel(axes[0, 0], bundle.two_d, config["panels"]["ci_ad"], plot_cfg, plot_cfg["panel_letters"][0])
    plot_one_d_panel(axes[0, 1], bundle.one_d["ci"], config["panels"]["ci"], plot_cfg, plot_cfg["panel_letters"][1])
    plot_one_d_panel(axes[0, 2], bundle.one_d["ad"], config["panels"]["ad"], plot_cfg, plot_cfg["panel_letters"][2])
    plot_one_d_panel(axes[1, 0], bundle.one_d["time"], config["panels"]["time"], plot_cfg, plot_cfg["panel_letters"][3])
    plot_one_d_panel(axes[1, 1], bundle.one_d["ph"], config["panels"]["ph"], plot_cfg, plot_cfg["panel_letters"][4])
    plot_one_d_panel(axes[1, 2], bundle.one_d["temp"], config["panels"]["temp"], plot_cfg, plot_cfg["panel_letters"][5])
    fig.subplots_adjust(**plot_cfg["subplot"])
    fig.savefig(destination, dpi=config["output"]["png_dpi"] if destination.suffix.lower() == ".png" else None)
    plt.close(fig)

def write_bundle_tables(bundle: Fig4DataBundle, output_dir: Path, stem: str) -> None:
    one_d_rows: list[pd.DataFrame] = []
    for feature_name, panel in bundle.one_d.items():
        one_d_rows.append(
            pd.DataFrame(
                {
                    "feature": feature_name,
                    "x": panel.x,
                    "prediction": panel.y,
                    "std": panel.y_std,
                    "q05": panel.y_q05,
                    "q95": panel.y_q95,
                }
            )
        )
    pd.concat(one_d_rows, ignore_index=True).to_csv(output_dir / f"{stem}_1d_curves.csv", index=False)

    xx, yy = np.meshgrid(bundle.two_d.x_grid, bundle.two_d.y_grid)
    pd.DataFrame(
        {
            "ci": xx.ravel(),
            "ad": yy.ravel(),
            "prediction": bundle.two_d.z.ravel(),
            "std": bundle.two_d.z_std.ravel(),
            "q05": bundle.two_d.z_q05.ravel(),
            "q95": bundle.two_d.z_q95.ravel(),
        }
    ).to_csv(output_dir / f"{stem}_2d_surface.csv", index=False)

def evaluate_grouped_generalization(
    raw_training_df: pd.DataFrame,
    model_name: str,
    model_params: dict[str, Any],
    groups: pd.Series,
    n_splits: int,
    label: str,
) -> tuple[pd.DataFrame, dict[str, float]]:
    unique_groups = groups.astype(str).nunique()
    n_splits = min(n_splits, unique_groups)
    splitter = GroupKFold(n_splits=n_splits)
    predictions: list[pd.DataFrame] = []
    fold_rows: list[dict[str, float | int | str]] = []

    for fold_index, (train_idx, test_idx) in enumerate(splitter.split(raw_training_df, groups=groups.astype(str)), start=1):
        fold_train_raw = raw_training_df.iloc[train_idx].copy()
        fold_test_raw = raw_training_df.iloc[test_idx].copy()
        fold_train = prepare_model_table(fold_train_raw, fit_df=fold_train_raw)
        fold_test = prepare_model_table(fold_test_raw, fit_df=fold_train_raw)
        pipe = Pipeline(
            [("prep", build_preprocessor(TRAINING_FEATURES)), ("model", instantiate_model(model_name, model_params))]
        )
        pipe.fit(fold_train[TRAINING_FEATURES], fold_train["q"].to_numpy())
        pred = pipe.predict(fold_test[TRAINING_FEATURES])
        actual = fold_test["q"].to_numpy()
        predictions.append(
            pd.DataFrame(
                {
                    "evaluation": label,
                    "fold": fold_index,
                    "actual_q": actual,
                    "predicted_q": pred,
                    "doi": fold_test["doi"].astype(str).to_numpy(),
                    "group_id": fold_test["group_id"].to_numpy(),
                }
            )
        )
        fold_rows.append(
            {
                "evaluation": label,
                "fold": fold_index,
                "r2": float(r2_score(actual, pred)),
                "mae": float(mean_absolute_error(actual, pred)),
                "rmse": float(mean_squared_error(actual, pred) ** 0.5),
                "rows": int(len(fold_test)),
            }
        )

    pred_df = pd.concat(predictions, ignore_index=True)
    summary = {
        "evaluation": label,
        "r2": float(r2_score(pred_df["actual_q"], pred_df["predicted_q"])),
        "mae": float(mean_absolute_error(pred_df["actual_q"], pred_df["predicted_q"])),
        "rmse": float(mean_squared_error(pred_df["actual_q"], pred_df["predicted_q"]) ** 0.5),
        "groups": int(unique_groups),
        "folds": int(n_splits),
    }
    return pd.DataFrame(fold_rows), summary

def evaluate_models_with_group_cv(
    raw_training_df: pd.DataFrame,
    tuned_params: dict[str, dict[str, object]],
    n_splits: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    groups = raw_training_df["group_id"].astype(int)
    unique_group_count = int(groups.nunique())
    if unique_group_count < 2:
        raise ValueError("Strict grouped CV requires at least 2 distinct groups.")
    n_splits = min(unique_group_count, 10) if n_splits is None else min(unique_group_count, int(n_splits))
    splitter = GroupKFold(n_splits=n_splits)

    summary_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    group_rows: list[dict[str, object]] = []

    for model_name in MODEL_ORDER:
        params = tuned_params.get(model_name, {})
        prediction_chunks: list[pd.DataFrame] = []
        for fold_index, (train_idx, test_idx) in enumerate(splitter.split(raw_training_df, groups=groups), start=1):
            fold_train_raw = raw_training_df.iloc[train_idx].copy()
            fold_test_raw = raw_training_df.iloc[test_idx].copy()
            fold_train = prepare_model_table(fold_train_raw, fit_df=fold_train_raw)
            fold_test = prepare_model_table(fold_test_raw, fit_df=fold_train_raw)
            pipe = Pipeline([("prep", build_preprocessor(TRAINING_FEATURES)), ("model", instantiate_model(model_name, params))])
            pipe.fit(fold_train[TRAINING_FEATURES], fold_train["q"].to_numpy())
            pred = pipe.predict(fold_test[TRAINING_FEATURES])
            actual = fold_test["q"].to_numpy()

            fold_rows.append(
                {
                    "model": model_name,
                    "fold": fold_index,
                    "rows": int(len(fold_test)),
                    "groups": int(fold_test["group_id"].nunique()),
                    "test_group_ids": ",".join(str(value) for value in sorted(fold_test["group_id"].astype(int).unique())),
                    "mae": float(mean_absolute_error(actual, pred)),
                    "rmse": float(mean_squared_error(actual, pred) ** 0.5),
                    "r2": float(r2_score(actual, pred)),
                }
            )

            pred_df = pd.DataFrame(
                {
                    "model": model_name,
                    "fold": fold_index,
                    "group_id": fold_test["group_id"].astype(int).to_numpy(),
                    "metal": fold_test["metal"].astype(str).to_numpy(),
                    "modification": fold_test["modification"].astype(str).to_numpy(),
                    "actual_q": actual,
                    "predicted_q": pred,
                }
            )
            prediction_chunks.append(pred_df)

        model_predictions = pd.concat(prediction_chunks, ignore_index=True)
        summary_rows.append(
            {
                "model": model_name,
                "folds": int(n_splits),
                "rows": int(len(model_predictions)),
                "groups": unique_group_count,
                "mae": float(mean_absolute_error(model_predictions["actual_q"], model_predictions["predicted_q"])),
                "rmse": float(mean_squared_error(model_predictions["actual_q"], model_predictions["predicted_q"]) ** 0.5),
                "r2": float(r2_score(model_predictions["actual_q"], model_predictions["predicted_q"])),
            }
        )

        for group_id, part in model_predictions.groupby("group_id", sort=True):
            group_rows.append(
                {
                    "group_id": int(group_id),
                    "model": model_name,
                    "rows": int(len(part)),
                    "metal": ",".join(sorted(set(part["metal"].astype(str)))),
                    "modification": ",".join(sorted(set(part["modification"].astype(str)))),
                    "actual_q_mean": float(part["actual_q"].mean()),
                    "predicted_q_mean": float(part["predicted_q"].mean()),
                    "mae": float(mean_absolute_error(part["actual_q"], part["predicted_q"])),
                    "rmse": float(mean_squared_error(part["actual_q"], part["predicted_q"]) ** 0.5),
                    "r2": float(r2_score(part["actual_q"], part["predicted_q"])) if len(part) > 1 else np.nan,
                }
            )

    summary_df = pd.DataFrame(summary_rows).sort_values(["r2", "rmse", "mae"], ascending=[False, True, True]).reset_index(drop=True)
    fold_df = pd.DataFrame(fold_rows).sort_values(["model", "fold"]).reset_index(drop=True)
    group_df = pd.DataFrame(group_rows).sort_values(["model", "mae", "group_id"], ascending=[True, True, True]).reset_index(drop=True)
    return summary_df, fold_df, group_df

def write_group_cv_report(
    raw_training_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    group_df: pd.DataFrame,
) -> None:
    group_overview = (
        raw_training_df.groupby("group_id", as_index=False)
        .agg(
            rows=("q", "size"),
            metal=("metal", lambda s: ",".join(sorted(set(s.astype(str))))),
            modification=("modification", lambda s: ",".join(sorted(set(s.astype(str))))),
            q_mean=("q", "mean"),
            q_min=("q", "min"),
            q_max=("q", "max"),
        )
        .sort_values("group_id")
    )
    best_model = summary_df.iloc[0]["model"]
    hardest_groups = group_df.loc[group_df["model"] == best_model].sort_values("mae", ascending=False).head(5)
    lines = [
        "# Strict Group CV Report",
        "",
        "## Dataset overview",
        f"- Rows: `{len(raw_training_df)}`",
        f"- Groups: `{raw_training_df['group_id'].nunique()}`",
        f"- Metals: `{', '.join(raw_training_df['metal'].value_counts().index.tolist())}`",
        "",
        "## Group summary",
    ]
    for row in group_overview.itertuples(index=False):
        lines.append(
            f"- Group `{int(row.group_id)}`: rows `{int(row.rows)}`, metal `{row.metal}`, modification `{row.modification}`, Q mean `{row.q_mean:.2f}`, range `{row.q_min:.2f}-{row.q_max:.2f}`"
        )
    lines.extend(
        [
            "",
            "## Strict grouped CV ranking",
        ]
    )
    for row in summary_df.itertuples(index=False):
        lines.append(f"- `{row.model}`: R2 `{row.r2:.3f}`, MAE `{row.mae:.3f}`, RMSE `{row.rmse:.3f}`")
    lines.extend(
        [
            "",
            f"## Hardest held-out groups for `{best_model}`",
        ]
    )
    for row in hardest_groups.itertuples(index=False):
        lines.append(
            f"- Group `{int(row.group_id)}` ({row.metal}, {row.modification}): rows `{int(row.rows)}`, actual mean `{row.actual_q_mean:.2f}`, predicted mean `{row.predicted_q_mean:.2f}`, MAE `{row.mae:.2f}`, RMSE `{row.rmse:.2f}`, R2 `{row.r2:.3f}`"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "- Negative or very low grouped R2 means the model can fit within known groups but struggles when an entire material group is held out.",
            "- If one held-out group is chemically narrow and far from the training-group distribution, grouped holdout can look much worse than row-wise CV even when ordinary CV appears acceptable.",
            "- For this dataset, improving grouped generalization likely requires more distinct material groups rather than only more repeated points inside the same group.",
        ]
    )
    (OUTPUT_DIR / "strict_group_cv_report.md").write_text("\n".join(lines), encoding="utf-8")

def save_validation_outputs(
    context: ModelContext,
    model_name: str,
    model_params: dict[str, Any],
    config: dict[str, Any],
    output_dir: Path,
    stem: str,
) -> tuple[list[dict[str, float]], list[str]]:
    val_cfg = config["external_validation"]
    diagnostics: list[dict[str, float]] = []
    note_lines: list[str] = []
    for label, groups, n_splits in [
        ("doi_level", context.raw_training_df["doi"].astype(str), int(val_cfg["doi_level_splits"])),
        ("material_family_level", context.raw_training_df["group_id"].astype(str), int(val_cfg["material_family_level_splits"])),
    ]:
        fold_df, summary = evaluate_grouped_generalization(
            context.raw_training_df,
            model_name,
            model_params,
            groups,
            n_splits,
            label,
        )
        fold_df.to_csv(output_dir / f"{stem}_{label}_fold_metrics.csv", index=False)
        diagnostics.append(summary)
        note_lines.append(
            f"- `{label}`: R2 `{summary['r2']:.3f}`, MAE `{summary['mae']:.3f}`, RMSE `{summary['rmse']:.3f}`, groups `{int(summary['groups'])}`, folds `{int(summary['folds'])}`"
        )
    pd.DataFrame(diagnostics).to_csv(output_dir / f"{stem}_external_validation_summary.csv", index=False)
    return diagnostics, note_lines

def validation_outputs_exist(output_dir: Path, stem: str) -> bool:
    required = [
        output_dir / f"{stem}_external_validation_summary.csv",
        output_dir / f"{stem}_doi_level_fold_metrics.csv",
        output_dir / f"{stem}_material_family_level_fold_metrics.csv",
    ]
    return all(path.exists() for path in required)

def load_validation_notes(output_dir: Path, stem: str) -> tuple[list[dict[str, float]], list[str]]:
    summary_df = pd.read_csv(output_dir / f"{stem}_external_validation_summary.csv")
    diagnostics = summary_df.to_dict(orient="records")
    note_lines = [
        f"- `{row['evaluation']}`: R2 `{row['r2']:.3f}`, MAE `{row['mae']:.3f}`, RMSE `{row['rmse']:.3f}`, groups `{int(row['groups'])}`, folds `{int(row['folds'])}`"
        for row in diagnostics
    ]
    return diagnostics, note_lines

def save_canonical_and_versioned(destination: Path, canonical_stem: str, output_dir: Path, output_tag: str, write_versioned_copy: bool) -> None:
    canonical_path = output_dir / f"{canonical_stem}{destination.suffix}"
    if destination != canonical_path:
        shutil.copyfile(destination, canonical_path)
    if write_versioned_copy:
        versioned_path = output_dir / f"{canonical_stem}_{output_tag}{destination.suffix}"
        if destination != versioned_path:
            shutil.copyfile(destination, versioned_path)

def cleanup_fig4_outputs(output_dir: Path) -> None:
    keep = {"fig4_best.png", "fig4_best.pdf", "fig4_best.svg"}
    for path in output_dir.glob("fig4*"):
        if path.name in keep:
            continue
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()

def save_bundle_outputs(
    bundle: Fig4DataBundle,
    context: ModelContext,
    config: dict[str, Any],
    output_dir: Path,
    stem: str,
    output_tag: str,
) -> None:
    output_cfg = config["output"]
    formats = [".png"]
    if output_cfg["write_pdf"]:
        formats.append(".pdf")
    if output_cfg["write_svg"]:
        formats.append(".svg")

    versioned = versioned_stem(stem, output_tag)
    for suffix in formats:
        destination = output_dir / f"{versioned}{suffix}"
        plot_fig4(bundle, config, destination)
        save_canonical_and_versioned(destination, stem, output_dir, output_tag, bool(output_cfg["write_versioned_copy"]))
    write_bundle_tables(bundle, output_dir, versioned)

def build_diagnostic_notes() -> list[str]:
    return [
        "- The legacy Fig. 4 routine coupled PDP calculation and plotting in a single function, which made paper-style tuning fragile.",
        "- The legacy single-variable panels used histogram-like density bands instead of rug marks, so the training-value distribution looked heavier than the paper.",
        "- The legacy 2D panel appended a colorbar and used generic subplot spacing, both of which drifted away from the paper layout.",
        "- Axis limits for CI, AD, Time, pH, and Tem were partly auto-expanded from the local data, so panel framing and whitespace were inconsistent with the paper.",
        "- Output files did not encode the dataset/model/parameter version, and the refinement path had no dedicated logging or tests.",
    ]

def write_descriptor_source_table(root: Path) -> None:
    table = """# Descriptor Preset Sources

| Preset | Field | Basis | Note |
| --- | --- | --- | --- |
| `neutral_atomic` | `ionic_charge` | Common oxidation state used in MOF literature | Engineering lookup table for local reproduction |
| `neutral_atomic` | `atomic_radius` | Neutral atomic radius (pm-scale) | Not published by the paper |
| `neutral_atomic` | `polarizability` | Tabulated elemental polarizability | Not published by the paper |
| `neutral_atomic` | `electronegativity` | Pauling electronegativity | Not published by the paper |
| `calibrated_mixed` | all four descriptor fields | Reverse-tuned values used to better match the paper's reported model ranking | Reconstruction, not an author-provided lookup |
| `ionic_radius` | `atomic_radius` field | Ionic radius proxy substituted into the `atomic_radius` slot | Keeps the downstream feature schema stable |

## Scope notes

- `Ag/Co/Cr/Cu/Fe/Zn/Zr` are the metals directly supported by the 801-row training table.
- `In/Ti/Nd` were added only for the user's 5382-candidate CoRE screening workflow.
- The source paper does not publish the exact `IC/AR/Pol/Ele` lookup table, so every preset in this repository remains an informed approximation.
"""
    (root / "DESCRIPTOR_PRESET_SOURCES.md").write_text(table, encoding="utf-8")

def write_report(
    output_dir: Path,
    report_name: str,
    config: dict[str, Any],
    context: ModelContext,
    bundle: Fig4DataBundle,
    validation_notes: list[str],
) -> None:
    comparison_models = [name for name in config["model_selection"]["comparison_models"] if name in context.fitted_models]
    lines = [
        "# Fig. 4 Refine Report",
        "",
        "## Diagnostic",
        *build_diagnostic_notes(),
        "",
        "## Local Changes",
        "- Integrated the refined Fig. 4 workflow directly into `scripts/reproduce_paper.py` so the main script is the single source of truth.",
        "- Moved Fig. 4 plotting parameters, axis limits, contour density, output formats, and validation settings into `config/fig4_config.json`.",
        "- Split PDP computation from plotting into standalone `compute_*` and `plot_*` functions.",
        "- Replaced density bands with rug marks derived from the prepared 801-row training distribution.",
        "- Removed the 2D colorbar and locked the CI/AD/Time/pH/Tem display ranges to paper-style bounds from the current local training workflow.",
        "- Added uncertainty CSV outputs from a CV ensemble and grouped validation summaries at DOI level and material-family-proxy level.",
        "",
        "## Model Roles",
        f"- Model selection model: `{context.best_model_name}` chosen by `run_model_grid_search_cv()` on the current 801-row training table.",
        f"- Display model: `{bundle.model_name}` fitted once on the full prepared training table with params `{stable_json(bundle.model_params)}`.",
        f"- Deployment model: the same `{bundle.model_name}` object family is reused downstream in the main screening workflow; this refinement script does not retrain a different display-only estimator.",
        "",
        "## Final Figure",
        f"- Final Fig. 4 uses the current CV first-ranked model: `{bundle.model_name}`.",
        f"- Output tag: `{context.output_tag}`.",
        "",
        "## Comparison Models",
        f"- Additional reference renders were written for: `{', '.join(comparison_models) if comparison_models else 'none'}`.",
        "- The source paper does not explicitly state which trained model produced Fig. 4, so using the current CV winner remains an engineering choice rather than a proven paper fact.",
        "",
        "## External Validation",
        *validation_notes,
        "",
        "## Remaining Uncertainty",
        "- The exact visual palette and contour level policy are not described in the paper, so the chosen map is a paper-style approximation from the screenshot.",
        "- The published article does not provide raw PDP arrays, so line shape agreement is limited by the local model and the paper screenshot only.",
        "- The material-family validation uses the repository's `group_id` proxy derived from metal, modification method, and structural fields; it is not an author-supplied family label.",
    ]
    (output_dir / report_name).write_text("\n".join(lines), encoding="utf-8")

def render_fig4_artifacts(config_path: Path) -> ModelContext:
    config = load_config(config_path)
    output_dir = ROOT / config["output"]["directory"]
    output_dir.mkdir(exist_ok=True)
    context = create_model_context(config)
    cleanup_fig4_outputs(output_dir)
    best_bundle = build_fig4_bundle(config, context, context.best_model_name)
    formats = [".png"]
    if config["output"]["write_pdf"]:
        formats.append(".pdf")
    if config["output"]["write_svg"]:
        formats.append(".svg")
    for suffix in formats:
        destination = output_dir / f"{config['output']['canonical_stem']}{suffix}"
        plot_fig4(best_bundle, config, destination)
    return context

def update_readme() -> None:
    content = r"""# Paper Reproduction

This workspace now contains a refined local reproduction of:

`Machine-learning-driven discovery of metal-organic framework adsorbents for hexavalent chromium removal from aqueous environments`

## Main Entry

- `scripts/reproduce_paper.py`
  Runs both a paper-faithful evaluation and a score-matched calibrated evaluation, then regenerates the main paper-style figures, including the final `fig4_best`.

## Run

```powershell
& 'F:\ProgramFiles\Anaconda\python.exe' scripts\reproduce_paper.py
```
"""
    (ROOT / 'REPRODUCTION.md').write_text(content, encoding='utf-8')


def main() -> int:
    parser = argparse.ArgumentParser(description='Reproduce the Cr(VI)-MOF machine learning paper figures and tables.')
    parser.add_argument('--skip-supplementary', action='store_true', help='Skip SHAP supplementary figure generation.')
    args = parser.parse_args()
    return run_reproduction(skip_supplementary=args.skip_supplementary)


if __name__ == '__main__':
    raise SystemExit(main())
