from __future__ import annotations

from scripts.repro_modules.common import *
from scripts.repro_modules.modeling import *

# This module owns all non-Fig.4 figure rendering and figure-adjacent
# export helpers used by the main reproduction workflow.

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

def annotate_bar_values(ax: plt.Axes, bars, values: Iterable[float], fontsize: float = 9.0) -> None:
    values = list(values)
    if not values:
        return
    for bar, value in zip(bars, values):
        ax.annotate(
            f"{int(round(float(value)))}",
            xy=(bar.get_x() + bar.get_width() / 2.0, bar.get_height()),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=fontsize,
            fontweight="bold",
            color="black",
            clip_on=False,
        )

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
    # Main-text Fig. 3: six-model fitting/parity panels.
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
    # Main-text Fig. 5: structure-performance relationships on first adsorption dataset.
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

def save_fig2_like(
    core_df: pd.DataFrame | None,
    fallback_df: pd.DataFrame,
    importance_df: pd.DataFrame,
    filename: str,
    workflow_counts: dict[str, int] | None = None,
) -> pd.DataFrame:
    # Main-text Fig. 2 combined panel: (a) structure relation, (b) workflow, (c) feature importance.
    set_paper_rcparams()
    fig = plt.figure(figsize=(7.1, 4.8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.45, 0.85], width_ratios=[1.1, 1.0])

    ax1 = fig.add_subplot(gs[0, 0], projection="3d")
    plot_df = prepare_fig2a_structural_data(core_df, fallback_df)
    ax1.scatter(plot_df["x"], plot_df["y"], plot_df["z"], c=FIG2_GREEN, s=3, alpha=0.55, edgecolors="none")
    style_fig2a_axis(ax1, plot_df)
    ax1.set_title("(a)", pad=-10)

    ax2 = fig.add_subplot(gs[0, 1])
    workflow_counts = {} if workflow_counts is None else workflow_counts
    training_rows = int(workflow_counts.get("training_rows", len(fallback_df)))
    model_count = int(workflow_counts.get("model_count", len(MODEL_ORDER)))
    candidate_rows = int(workflow_counts.get("candidate_rows", 0 if core_df is None else len(core_df)))
    initial_rows = int(workflow_counts.get("initial_rows", 0))
    steps = [
        (f"{training_rows}\nliterature\nrecords", (0.18, 0.83)),
        (f"{model_count} model\ncomparison", (0.52, 0.83)),
        (f"{candidate_rows} CoRE\ncandidates", (0.84, 0.83)),
        ("Top 10 per\nmetal", (0.36, 0.40)),
        (f"{initial_rows} initial\nscreening", (0.66, 0.40)),
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
    bars = ax3.bar(ordered["feature"], ordered["reproduced_importance"], color=FIG2_GREEN, edgecolor="black", linewidth=0.5)
    ax3.set_ylabel("Importance (%)")
    ax3.set_title("(c)", pad=2)
    ax3.set_ylim(0, float(ordered["reproduced_importance"].max()) * 1.28)
    ax3.margins(y=0.08)
    style_small_axis(ax3)
    ax3.tick_params(axis="x", rotation=0)
    annotate_bar_values(ax3, bars, ordered["reproduced_importance"])

    fig.subplots_adjust(left=0.06, right=0.98, top=0.96, bottom=0.16, wspace=0.15, hspace=0.28)
    add_caption(fig, "Fig. 2. Structural overview, workflow, and feature importance.")
    fig.savefig(OUTPUT_DIR / filename, dpi=300)
    plt.close(fig)
    return plot_df

def save_fig2a_relationship(core_df: pd.DataFrame | None, fallback_df: pd.DataFrame, filename: str) -> None:
    # Standalone export of Fig. 2(a): 3D structural relationship panel.
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
    # Standalone export of Fig. 2(c): feature importance bar chart.
    set_paper_rcparams()
    ordered = importance_df.sort_values("reproduced_importance", ascending=False)
    fig, ax = plt.subplots(figsize=(5.6, 2.7))
    bars = ax.bar(ordered["feature"], ordered["reproduced_importance"], color=FIG2_GREEN, edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Importance (%)")
    ax.set_title("(c)", pad=2)
    ax.set_ylim(0, float(ordered["reproduced_importance"].max()) * 1.28)
    ax.margins(y=0.08)
    style_small_axis(ax)
    ax.tick_params(axis="x", rotation=0)
    annotate_bar_values(ax, bars, ordered["reproduced_importance"])
    fig.subplots_adjust(left=0.08, right=0.99, top=0.92, bottom=0.22)
    fig.savefig(OUTPUT_DIR / filename, dpi=300)
    plt.close(fig)
    return ordered


# --- supplementary.py ---

warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn.impute._iterative")

def save_single_shap_beeswarm(pipe: Pipeline, sample_df: pd.DataFrame, path: Path) -> None:
    # Supplementary Fig. S5: SHAP beeswarm summary.
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
    # Supplementary Fig. S6: SHAP waterfall for one selected sample.
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
    # Convenience wrapper used by the main workflow to emit Fig. S5 and Fig. S6 together.
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
    # Export supplementary Text S1 and Text S2 as markdown files for local inspection.
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
    # Supplementary Fig. S1: quantitative variable distributions.
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
    # Supplementary Fig. S2: qualitative variable/category distributions.
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
    mod_order.extend([name for name in mod_counts.index if name not in mod_order])
    mod_labels = [get_mod_plot_label(name) for name in mod_order]
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
    # Supplementary Fig. S3: correlation heatmap of quantitative variables.
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
    # Supplementary Fig. S4: learning curve of the current display/best model.
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
    # Helper export: save paper PDF pages locally for visual reference during figure alignment.
    pdf_path = next(root.glob("*.pdf"))
    out_dir = OUTPUT_DIR / "paper_pages"
    out_dir.mkdir(exist_ok=True)
    doc = fitz.open(pdf_path)
    for page_number in [5, 6, 7]:
        page = doc[page_number]
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
        pix.save(out_dir / f"page_{page_number + 1}.png")

def _save_doc_page_to_text_image(root: Path, pdf_name_pattern: str, page_number: int, out_name: str) -> Path:
    # Helper export: render one PDF page to an image file for debugging/reference.
    pdf_path = next(root.glob(pdf_name_pattern))
    out_path = OUTPUT_DIR / out_name
    doc = fitz.open(pdf_path)
    page = doc[page_number]
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
    pix.save(out_path)
    return out_path


