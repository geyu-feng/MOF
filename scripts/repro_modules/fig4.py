from __future__ import annotations

from scripts.repro_modules.common import *
from scripts.repro_modules.modeling import *
from scripts.repro_modules.modeling import _run_model_grid_search_cv
from scripts.repro_modules.plots import *

# Dedicated Fig. 4 workflow module.
# This file does not only draw the figure; it also prepares the model context,
# computes PDP data, writes validation/report artifacts, and exports fig4_best.*.

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

def build_model_selection_signature(config: dict[str, Any]) -> str:
    payload = {
        "dataset": config["dataset"],
        "model_selection": config["model_selection"],
        "model_order": MODEL_ORDER,
        "param_grids": get_model_param_grids(),
        "common_source_sha1": hashlib.sha1((ROOT / "scripts" / "repro_modules" / "common.py").read_bytes()).hexdigest(),
        "modeling_source_sha1": hashlib.sha1((ROOT / "scripts" / "repro_modules" / "modeling.py").read_bytes()).hexdigest(),
        "fig4_source_sha1": hashlib.sha1((ROOT / "scripts" / "repro_modules" / "fig4.py").read_bytes()).hexdigest(),
    }
    return hashlib.sha1(stable_json(payload).encode("utf-8")).hexdigest()

def model_selection_cache_meta_path(output_dir: Path) -> Path:
    return output_dir / "fig4_model_selection_cache.json"

def write_model_selection_cache_meta(output_dir: Path, config: dict[str, Any], raw_training_df: pd.DataFrame) -> None:
    meta = {
        "dataset": config["dataset"],
        "config_version": config["version"],
        "training_fingerprint": dataframe_fingerprint(raw_training_df),
        "selection_signature": build_model_selection_signature(config),
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
        and meta.get("selection_signature") == build_model_selection_signature(config)
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
    # Build the training/model state used specifically by Fig. 4.
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
        cv_results, best_per_model, _, prepared_training_df = _run_model_grid_search_cv(raw_training_df, output_dir)
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

def build_model_context_from_workflow(
    config: dict[str, Any],
    raw_training_df: pd.DataFrame,
    cv_results: pd.DataFrame,
    best_per_model: pd.DataFrame,
    fitted_models: dict[str, Pipeline],
    prepared_training_df: pd.DataFrame,
) -> ModelContext:
    best_row = best_per_model.iloc[0]
    best_model_name = str(best_row["model"])
    best_model_params = json.loads(best_row["params_json"])
    output_tag = build_output_tag(config, best_model_name, best_model_params)
    LOGGER.info("Reused main workflow best model %s for Fig. 4.", best_model_name)
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
    group_count = int(raw_training_df["group_id"].nunique())
    if group_count < 2:
        fold_train = prepare_model_table(raw_training_df, fit_df=raw_training_df)
        pipe = Pipeline(
            [("prep", build_preprocessor(TRAINING_FEATURES)), ("model", instantiate_model(model_name, model_params))]
        )
        pipe.fit(fold_train[TRAINING_FEATURES], fold_train["q"].to_numpy())
        LOGGER.warning("Fig. 4 uncertainty ensemble fell back to a single full-data fit because only one group was available.")
        return [pipe]
    n_splits = min(int(n_splits), group_count)
    splitter = GroupKFold(n_splits=n_splits)
    ensemble: list[Pipeline] = []
    for fold_index, (train_idx, _) in enumerate(splitter.split(raw_training_df, groups=raw_training_df["group_id"]), start=1):
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

def build_quantile_grid(values: np.ndarray, limits: tuple[float, float], n_points: int) -> np.ndarray:
    lo = float(limits[0])
    hi = float(limits[1])
    if values.size == 0:
        return build_linear_grid(limits, n_points)
    clipped = values[(values >= lo) & (values <= hi)]
    if clipped.size == 0:
        return build_linear_grid(limits, n_points)
    unique_values = np.unique(clipped.astype(float))
    if unique_values.size <= int(n_points) - 2:
        grid = np.concatenate(([lo], unique_values, [hi]))
    else:
        quantiles = np.linspace(0.0, 1.0, max(int(n_points) - 2, 2), dtype=float)
        quantile_grid = np.quantile(clipped, quantiles)
        grid = np.concatenate(([lo], quantile_grid.astype(float), [hi]))
    grid = np.clip(grid, lo, hi)
    grid = np.unique(grid.astype(float))
    if grid.size < 2:
        return build_linear_grid(limits, n_points)
    return grid

def build_one_d_grid(
    training_df: pd.DataFrame,
    feature_name: str,
    limits: tuple[float, float],
    n_points: int,
) -> np.ndarray:
    values = compute_rug_values(training_df, feature_name, limits)
    return build_quantile_grid(values, limits, n_points)

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
    training_df: pd.DataFrame,
    panel_cfg: dict[str, Any],
) -> TwoDPanelData:
    x_feature = panel_cfg["x_feature"]
    y_feature = panel_cfg["y_feature"]
    x_values = compute_rug_values(training_df, x_feature, tuple(panel_cfg["xlim"]))
    y_values = compute_rug_values(training_df, y_feature, tuple(panel_cfg["ylim"]))
    x_grid = build_quantile_grid(x_values, tuple(panel_cfg["xlim"]), panel_cfg["x_points"])
    y_grid = build_quantile_grid(y_values, tuple(panel_cfg["ylim"]), panel_cfg["y_points"])
    point_surface = compute_partial_dependence_2d(
        predictor,
        base_frame,
        x_feature,
        y_feature,
        x_grid,
        y_grid,
    )
    ensemble_surfaces = [
        compute_partial_dependence_2d(
            model,
            base_frame,
            x_feature,
            y_feature,
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
    # Assemble all six panels of Fig. 4:
    # (a) 2D PDP for CI x AD
    # (b)-(f) 1D PDPs for CI, AD, Time, pH, Tem
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
    two_d = compute_two_d_panel(predictor, ensemble, base_frame, context.prepared_training_df, config["panels"]["ci_ad"])
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

def build_contour_levels(surface: np.ndarray, n_levels: int) -> np.ndarray:
    flattened = np.asarray(surface, dtype=float).ravel()
    quantiles = np.linspace(0.0, 1.0, int(n_levels), dtype=float)
    levels = np.quantile(flattened, quantiles)
    levels = np.unique(levels.astype(float))
    if levels.size < 2:
        levels = np.linspace(float(np.min(flattened)), float(np.max(flattened)) + 1e-9, max(int(n_levels), 2))
    return levels

def plot_two_d_panel(ax: plt.Axes, panel: TwoDPanelData, panel_cfg: dict[str, Any], plot_cfg: dict[str, Any], letter: str) -> None:
    xx, yy = np.meshgrid(panel.x_grid, panel.y_grid)
    ax.pcolormesh(xx, yy, panel.z, cmap=plot_cfg["colormap"], shading="nearest")
    ax.set_xlim(float(np.min(panel.x_grid)), float(np.max(panel.x_grid)))
    ax.set_ylim(float(np.min(panel.y_grid)), float(np.max(panel.y_grid)))
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
    ax.plot(panel.x, panel.y, linestyle="none", marker="o", markersize=2.4, color=plot_cfg["line_color"], zorder=2.2)
    ax.set_xlim(float(np.min(panel.x)), float(np.max(panel.x)))
    ax.set_ylim(*infer_y_limits(panel.y, panel_cfg))
    ax.set_xlabel(panel_cfg["xlabel"])
    ax.set_ylabel(panel_cfg["ylabel"])
    ax.set_title(letter, pad=2)
    add_rug_marks(ax, panel.rug_values, plot_cfg)
    apply_axis_ticks(ax, panel_cfg, "x")
    set_axis_style(ax)

def plot_fig4(bundle: Fig4DataBundle, config: dict[str, Any], destination: Path) -> None:
    # Final renderer for Fig. 4 output files, typically fig4_best.png/pdf/svg.
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
    # Save the underlying Fig. 4 1D/2D PDP arrays for audit/debugging.
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
    # Diagnostic validation for the Fig. 4 model context; not a paper figure itself.
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
    # Strict grouped CV diagnostics used by reports and debugging around Fig. 4/modeling.
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
    # Write strict grouped-CV markdown summary for local review.
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
    # Export DOI-level and material-family-level validation CSV files for Fig. 4 diagnostics.
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
    # Keep only the canonical fig4_best.* outputs in the output directory.
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
    # Write Fig. 4专项说明文件: fig4_refine_report.md
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

def render_fig4_artifacts(
    config_path: Path,
    *,
    raw_training_df: pd.DataFrame | None = None,
    cv_results: pd.DataFrame | None = None,
    best_per_model: pd.DataFrame | None = None,
    fitted_models: dict[str, Pipeline] | None = None,
    prepared_training_df: pd.DataFrame | None = None,
) -> ModelContext:
    # Public Fig. 4 entrypoint used by the main workflow.
    # It creates/loads model context, rebuilds fig4_best.*, and returns the context.
    config = load_config(config_path)
    output_dir = ROOT / config["output"]["directory"]
    output_dir.mkdir(exist_ok=True)
    if (
        raw_training_df is not None
        and cv_results is not None
        and best_per_model is not None
        and fitted_models is not None
        and prepared_training_df is not None
    ):
        context = build_model_context_from_workflow(
            config,
            raw_training_df,
            cv_results,
            best_per_model,
            fitted_models,
            prepared_training_df,
        )
    else:
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

