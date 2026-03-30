from __future__ import annotations

import math
from itertools import combinations

from scripts.repro_modules.common import *
from sklearn.model_selection import train_test_split

# --- core.py ---

def get_fixed_model_params() -> dict[str, dict[str, object]]:
    return {
        "RF": {"n_estimators": 400, "max_depth": 16, "min_samples_leaf": 2, "min_samples_split": 4},
        "GBDT": {
            "n_estimators": 300,
            "max_depth": 3,
            "min_samples_leaf": 1,
            "learning_rate": 0.05,
            "subsample": 0.9,
            "loss": "squared_error",
        },
        "XGB": {
            "n_estimators": 450,
            "max_depth": 6,
            "min_child_weight": 2,
            "gamma": 0.0,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "learning_rate": 0.03,
            "reg_lambda": 1.0,
        },
        "LR": {"fit_intercept": True},
        "KNN": {"n_neighbors": 3, "weights": "distance"},
        "SVR": {"kernel": "rbf", "C": 30.0, "epsilon": 0.3, "gamma": "scale"},
    }

def get_additional_model_params() -> dict[str, dict[str, object]]:
    return {
        "CatBoost": {"iterations": 500, "depth": 6, "learning_rate": 0.05, "l2_leaf_reg": 3.0},
        "ExtraTree": {"max_depth": 16, "min_samples_leaf": 2, "min_samples_split": 4},
        "HistGBDT": {"max_depth": 6, "learning_rate": 0.05, "max_iter": 300, "min_samples_leaf": 10},
        "DecisionTree": {"max_depth": 10, "min_samples_leaf": 2, "min_samples_split": 4},
        "Bagging": {"n_estimators": 300, "max_depth": 10, "min_samples_leaf": 2, "min_samples_split": 4},
        "LightGBM": {"n_estimators": 400, "learning_rate": 0.03, "max_depth": 6, "num_leaves": 31, "min_child_samples": 10, "subsample": 0.9, "colsample_bytree": 0.9},
    }


def get_model_param_grids() -> dict[str, dict[str, list[object]]]:
    fixed_params = get_fixed_model_params()
    return {
        model_name: {param_name: [param_value] for param_name, param_value in model_params.items()}
        for model_name, model_params in fixed_params.items()
    }

def instantiate_model(model_name: str, params: dict[str, object] | None = None) -> object:
    params = {} if params is None else params.copy()
    if model_name == "RF":
        base = {
            "n_estimators": 400,
            "max_depth": 16,
            "min_samples_leaf": 2,
            "min_samples_split": 4,
            "random_state": 42,
            "n_jobs": -1,
        }
        base.update(params)
        return RandomForestRegressor(**base)
    if model_name == "GBDT":
        base = {
            "n_estimators": 300,
            "max_depth": 3,
            "min_samples_leaf": 1,
            "learning_rate": 0.05,
            "subsample": 0.9,
            "loss": "squared_error",
            "random_state": 42,
        }
        base.update(params)
        return GradientBoostingRegressor(**base)
    if model_name == "XGB":
        base = {
            "n_estimators": 450,
            "learning_rate": 0.03,
            "max_depth": 6,
            "min_child_weight": 2,
            "gamma": 0.0,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "reg_lambda": 1.0,
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
        base = {"n_neighbors": 3, "weights": "distance"}
        base.update(params)
        return KNeighborsRegressor(**base)
    if model_name == "SVR":
        base = {"kernel": "rbf", "C": 30.0, "epsilon": 0.3, "gamma": "scale"}
        base.update(params)
        return SVR(**base)
    if model_name == "CatBoost":
        if CatBoostRegressor is None:
            raise ImportError("CatBoost is not installed. Please install catboost to use CatBoostRegressor.")
        base = {
            "iterations": 500,
            "depth": 6,
            "learning_rate": 0.05,
            "l2_leaf_reg": 3.0,
            "loss_function": "RMSE",
            "random_seed": 42,
            "verbose": 0,
            "allow_writing_files": False,
        }
        base.update(params)
        return CatBoostRegressor(**base)
    if model_name == "ExtraTree":
        base = {"max_depth": 16, "min_samples_leaf": 2, "min_samples_split": 4, "random_state": 42}
        base.update(params)
        return ExtraTreeRegressor(**base)
    if model_name == "HistGBDT":
        base = {"max_depth": 6, "learning_rate": 0.05, "max_iter": 300, "min_samples_leaf": 10, "random_state": 42}
        base.update(params)
        return HistGradientBoostingRegressor(**base)
    if model_name == "DecisionTree":
        base = {"max_depth": 10, "min_samples_leaf": 2, "min_samples_split": 4, "random_state": 42}
        base.update(params)
        return DecisionTreeRegressor(**base)
    if model_name == "Bagging":
        base = {
            "n_estimators": 300,
            "random_state": 42,
            "n_jobs": -1,
            "estimator": DecisionTreeRegressor(max_depth=10, min_samples_leaf=2, min_samples_split=4, random_state=42),
        }
        custom = params.copy()
        if "max_depth" in custom or "min_samples_leaf" in custom or "min_samples_split" in custom:
            base["estimator"] = DecisionTreeRegressor(
                max_depth=custom.pop("max_depth", 10),
                min_samples_leaf=custom.pop("min_samples_leaf", 2),
                min_samples_split=custom.pop("min_samples_split", 4),
                random_state=42,
            )
        base.update(custom)
        return BaggingRegressor(**base)
    if model_name == "LightGBM":
        if LGBMRegressor is None:
            raise ImportError("LightGBM is not installed. Please install lightgbm to use LGBMRegressor.")
        base = {
            "n_estimators": 400,
            "learning_rate": 0.03,
            "max_depth": 6,
            "num_leaves": 31,
            "min_child_samples": 10,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "random_state": 42,
            "verbosity": -1,
        }
        base.update(params)
        return LGBMRegressor(**base)
    raise ValueError(f"Unknown model name: {model_name}")

def build_models() -> dict[str, object]:
    return {model_name: instantiate_model(model_name) for model_name in MODEL_ORDER}

def build_preprocessor(feature_columns: Iterable[str], model_name: str) -> ColumnTransformer:
    feature_columns = list(feature_columns)
    if model_name in {"LR", "KNN", "SVR"}:
        return ColumnTransformer(
            transformers=[("scale", StandardScaler(), feature_columns)],
            remainder="drop",
        )
    return ColumnTransformer(
        transformers=[("pass", "passthrough", feature_columns)],
        remainder="drop",
    )

def choose_group_cv_splits(group_count: int) -> int:
    """Use fewer grouped folds on small material-family datasets to stabilize CV."""
    return max(2, min(5, group_count))


def choose_row_cv_splits(row_count: int) -> int:
    """Use a modest KFold count for row-wise validation on local tabular datasets."""
    return max(2, min(5, int(row_count)))


def validate_holdout_train_size(frame: pd.DataFrame, context: str, test_size: float = 0.1) -> int:
    row_count = int(len(frame))
    min_train_rows = max(int(get_fixed_model_params()["KNN"]["n_neighbors"]), 2)
    if row_count < min_train_rows + 1:
        raise ValueError(
            f"{context} requires at least {min_train_rows + 1} rows for a {int(test_size * 100)}% holdout, "
            f"got {row_count}."
        )
    return row_count


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
    return max(2, int(math.ceil(n_groups * 0.10)))


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
    feature_frame = raw_df[["group_id", *TRAINING_FEATURES]].copy()
    feature_frame[TRAINING_FEATURES] = feature_frame[TRAINING_FEATURES].apply(pd.to_numeric, errors="coerce")
    feature_frame[TRAINING_FEATURES] = feature_frame[TRAINING_FEATURES].fillna(feature_frame[TRAINING_FEATURES].median())
    group_centroids = feature_frame.groupby("group_id")[TRAINING_FEATURES].mean().astype(float)
    centroid_std = group_centroids.std(ddof=0).replace(0.0, 1.0)
    group_centroids = (group_centroids - group_centroids.mean()) / centroid_std
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

        test_centroids = group_centroids.loc[combo_ids].to_numpy(dtype=float)
        train_centroids = group_centroids.drop(index=combo_ids).to_numpy(dtype=float)
        if train_centroids.size == 0:
            similarity_loss = 0.0
        else:
            distances = np.sqrt(((test_centroids[:, None, :] - train_centroids[None, :, :]) ** 2).sum(axis=2))
            similarity_loss = float(np.mean(np.min(distances, axis=1)))

        score = (
            4.0 * row_loss
            + 2.0 * metal_loss
            + 1.5 * q_mean_loss
            + 1.0 * q_std_loss
            + 1.0 * singleton_penalty
            + 0.5 * size_balance_loss
            + 1.5 * similarity_loss
        )
        score += float(rng.uniform(0.0, 1e-9))

        if best_score is None or score < best_score:
            best_score = score
            best_combo = combo

    if best_combo is None:
        raise RuntimeError("Unable to select a balanced grouped test split.")
    return set(int(x) for x in best_combo)

def _run_model_grid_search_cv(raw_df: pd.DataFrame, output_dir) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Pipeline], pd.DataFrame]:
    prepared_full_df = prepare_model_table(raw_df, fit_df=raw_df)
    validate_holdout_train_size(prepared_full_df, "Model selection")
    train_idx, test_idx = train_test_split(
        np.arange(len(prepared_full_df)),
        test_size=0.1,
        random_state=DEFAULT_SPLIT_SEED,
        shuffle=True,
    )
    train_df = prepared_full_df.iloc[train_idx].copy()
    test_df = prepared_full_df.iloc[test_idx].copy()
    rows: list[dict[str, object]] = []

    for model_name in MODEL_ORDER:
        for params in ParameterGrid(get_model_param_grids()[model_name]):
            pipe = Pipeline([("prep", build_preprocessor(TRAINING_FEATURES, model_name)), ("model", instantiate_model(model_name, params))])
            pipe.fit(train_df[TRAINING_FEATURES], train_df["q"].to_numpy())
            pred_all = pipe.predict(test_df[TRAINING_FEATURES])
            actual_all = test_df["q"].to_numpy()
            holdout_r2 = float(r2_score(actual_all, pred_all))
            holdout_mae = float(mean_absolute_error(actual_all, pred_all))
            holdout_rmse = float(mean_squared_error(actual_all, pred_all) ** 0.5)

            rows.append(
                {
                    "model": model_name,
                    "params_json": json.dumps(params, sort_keys=True),
                    "mean_r2": holdout_r2,
                    "std_r2": 0.0,
                    "mean_mae": holdout_mae,
                    "mean_rmse": holdout_rmse,
                    "oof_r2": holdout_r2,
                    "oof_mae": holdout_mae,
                    "oof_rmse": holdout_rmse,
                    "cv_strategy": "fixed_holdout",
                    "cv_folds": 1,
                }
            )

    cv_results = pd.DataFrame(rows).sort_values(["model", "mean_r2", "mean_rmse", "mean_mae"], ascending=[True, False, True, True])
    best_per_model = cv_results.drop_duplicates(subset=["model"], keep="first").sort_values(
        ["oof_r2", "oof_rmse", "oof_mae"], ascending=[False, True, True]
    )
    cv_results.to_csv(output_dir / "model_grid_search_cv.csv", index=False)
    best_per_model.to_csv(output_dir / "model_grid_search_best_per_model.csv", index=False)
    fitted_best_models: dict[str, Pipeline] = {}
    y = prepared_full_df["q"].to_numpy()
    for row in best_per_model.itertuples(index=False):
        params = json.loads(row.params_json)
        pipe = Pipeline([("prep", build_preprocessor(TRAINING_FEATURES, row.model)), ("model", instantiate_model(row.model, params))])
        pipe.fit(prepared_full_df[TRAINING_FEATURES], y)
        fitted_best_models[row.model] = pipe
    return cv_results, best_per_model, fitted_best_models, prepared_full_df

def make_split(df: pd.DataFrame, config) -> SplitBundle:
    validate_holdout_train_size(df, f"{config.name} split")
    random_state = DEFAULT_SPLIT_SEED if config.random_state is None else int(config.random_state)
    train_idx, test_idx = train_test_split(
        df.index.to_numpy(),
        test_size=0.1,
        random_state=random_state,
        shuffle=True,
    )
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
        pipe = Pipeline([("prep", build_preprocessor(TRAINING_FEATURES, model_name)), ("model", model)])
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

def fit_named_models_on_existing_split(
    prepared_split: dict[str, pd.DataFrame],
    config,
    model_names: list[str],
    model_params: dict[str, dict[str, object]] | None = None,
) -> tuple[pd.DataFrame, dict[str, dict[str, pd.DataFrame]], dict[str, Pipeline]]:
    train_df = prepared_split["train"].copy()
    test_df = prepared_split["test"].copy()
    y_train = train_df["q"].to_numpy()
    y_test = test_df["q"].to_numpy()

    metrics = []
    predictions: dict[str, dict[str, pd.DataFrame]] = {}
    fitted: dict[str, Pipeline] = {}
    for model_name in model_names:
        params = None if model_params is None else model_params.get(model_name)
        model = instantiate_model(model_name, params)
        pipe = Pipeline([("prep", build_preprocessor(TRAINING_FEATURES, model_name)), ("model", model)])
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
                "train_groups": int(train_df["group_id"].nunique()),
                "test_groups": int(test_df["group_id"].nunique()),
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
    return pd.DataFrame(metrics), predictions, fitted

def fit_named_models_on_full_training(
    prepared_full_df: pd.DataFrame,
    model_names: list[str],
    model_params: dict[str, dict[str, object]] | None = None,
) -> dict[str, Pipeline]:
    y = prepared_full_df["q"].to_numpy()
    fitted: dict[str, Pipeline] = {}
    for model_name in model_names:
        params = None if model_params is None else model_params.get(model_name)
        model = instantiate_model(model_name, params)
        pipe = Pipeline([("prep", build_preprocessor(TRAINING_FEATURES, model_name)), ("model", model)])
        pipe.fit(prepared_full_df[TRAINING_FEATURES], y)
        fitted[model_name] = pipe
    return fitted

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


