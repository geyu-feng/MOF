from __future__ import annotations

import math
from itertools import combinations

from scripts.repro_modules.common import *

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

def choose_group_cv_splits(group_count: int) -> int:
    """Use fewer grouped folds on small material-family datasets to stabilize CV."""
    return max(2, min(5, group_count))


def build_group_cv_folds(raw_df: pd.DataFrame, n_splits: int = 5) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    validate_group_count(raw_df, "GroupKFold model selection", minimum=2)
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


def resolve_test_group_count(n_groups: int, config) -> int:
    if config.mode == "sequential":
        if config.train_group_count is not None:
            return max(1, min(n_groups - 1, n_groups - int(config.train_group_count)))
        return min(n_groups - 1, recommend_test_group_count(n_groups))
    if config.mode == "group_shuffle":
        if n_groups <= GROUP_AWARE_TUNING_GROUP_THRESHOLD:
            return min(n_groups - 1, recommend_test_group_count(n_groups))
        return max(1, min(n_groups - 1, int(math.ceil(n_groups * 0.1))))
    raise ValueError(f"Unknown split mode: {config.mode}")


def _iter_candidate_group_combinations(group_ids: list[int], test_group_count: int, seed: int) -> list[tuple[int, ...]]:
    total = math.comb(len(group_ids), test_group_count)
    if total <= 200000:
        return [tuple(combo) for combo in combinations(group_ids, test_group_count)]
    rng = np.random.default_rng(seed)
    target_samples = min(10000, total)
    seen: set[tuple[int, ...]] = set()
    combos: list[tuple[int, ...]] = []
    while len(combos) < target_samples:
        combo = tuple(sorted(int(x) for x in rng.choice(group_ids, size=test_group_count, replace=False)))
        if combo in seen:
            continue
        seen.add(combo)
        combos.append(combo)
    return combos


def select_balanced_test_groups(raw_df: pd.DataFrame, test_group_count: int, random_state: int | None = None) -> set[int]:
    """Pick a representative grouped holdout instead of relying on group-id order."""
    group_ids = sorted(int(x) for x in raw_df["group_id"].drop_duplicates().tolist())
    if test_group_count <= 0 or test_group_count >= len(group_ids):
        raise ValueError(
            f"Invalid test_group_count={test_group_count} for {len(group_ids)} groups."
        )

    rng = np.random.default_rng(42 if random_state is None else int(random_state))
    candidate_combos = _iter_candidate_group_combinations(group_ids, test_group_count, 42 if random_state is None else int(random_state))

    group_rows = raw_df.groupby("group_id").size().astype(float)
    metal_counts = raw_df.groupby(["group_id", "metal"]).size().unstack(fill_value=0).astype(float)
    total_rows = float(len(raw_df))
    overall_q_mean = float(raw_df["q"].mean())
    overall_q_std = float(raw_df["q"].std(ddof=0))
    overall_q_std = overall_q_std if overall_q_std > 0 else 1.0
    overall_metal_props = metal_counts.sum(axis=0) / total_rows
    target_row_ratio = test_group_count / len(group_ids)
    mean_group_rows = float(group_rows.mean()) if len(group_rows) else 1.0

    best_combo: tuple[int, ...] | None = None
    best_score: float | None = None

    for combo in candidate_combos:
        combo_ids = list(combo)
        test_mask = raw_df["group_id"].isin(combo_ids)
        test_rows = float(test_mask.sum())
        if test_rows <= 0:
            continue
        row_ratio = test_rows / total_rows
        row_loss = abs(row_ratio - target_row_ratio)

        test_q = raw_df.loc[test_mask, "q"]
        test_q_mean = float(test_q.mean())
        test_q_std = float(test_q.std(ddof=0)) if len(test_q) > 1 else 0.0
        q_mean_loss = abs(test_q_mean - overall_q_mean) / overall_q_std
        q_std_loss = abs(test_q_std - overall_q_std) / overall_q_std

        test_metal_props = metal_counts.loc[combo_ids].sum(axis=0) / test_rows
        metal_loss = float((test_metal_props - overall_metal_props).abs().sum())

        combo_rows = group_rows.loc[combo_ids]
        singleton_penalty = float((combo_rows <= 1).mean())
        size_balance_loss = abs(float(combo_rows.mean()) - mean_group_rows) / max(mean_group_rows, 1.0)

        score = (
            4.0 * row_loss
            + 2.0 * metal_loss
            + 1.5 * q_mean_loss
            + 1.0 * q_std_loss
            + 1.0 * singleton_penalty
            + 0.5 * size_balance_loss
        )
        score += float(rng.uniform(0.0, 1e-9))

        if best_score is None or score < best_score:
            best_score = score
            best_combo = combo

    if best_combo is None:
        raise RuntimeError("Unable to select a balanced grouped test split.")
    return set(int(x) for x in best_combo)

def _run_model_grid_search_cv(raw_df: pd.DataFrame, output_dir) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Pipeline], pd.DataFrame]:
    group_count = validate_group_count(raw_df, "Grouped model selection", minimum=2)
    n_splits = choose_group_cv_splits(group_count)
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
    if len(unique_groups) < 2:
        raise ValueError(
            f"Split config '{config.name}' requires at least 2 distinct groups, got {len(unique_groups)}."
        )
    test_group_count = resolve_test_group_count(len(unique_groups), config)
    test_groups = select_balanced_test_groups(df, test_group_count, random_state=config.random_state)
    train_idx = df.index[~df["group_id"].isin(test_groups)].to_numpy()
    test_idx = df.index[df["group_id"].isin(test_groups)].to_numpy()
    return SplitBundle(train_idx, test_idx, df.iloc[train_idx]["group_id"].nunique(), df.iloc[test_idx]["group_id"].nunique())

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


