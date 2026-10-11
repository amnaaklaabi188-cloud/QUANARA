"""Model factory: every algorithm is wrapped in a single sklearn-compatible estimator that
contains its own preprocessing, so the saved artefact is the complete pipeline.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.svm import SVR

from . import config
from .features import categorical_features, numeric_features

try:
    from catboost import CatBoostRegressor
    HAS_CATBOOST = True
except Exception:  # pragma: no cover
    HAS_CATBOOST = False
try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except Exception:  # pragma: no cover
    HAS_XGBOOST = False
try:  # LightGBM is optional and not part of this environment; it is offered only if importable
    import lightgbm  # noqa: F401
    HAS_LIGHTGBM = True
except Exception:
    HAS_LIGHTGBM = False


# name -> (label, type, default, min, max, step, help)
HYPERPARAMS: dict[str, dict[str, tuple]] = {
    "seasonal_naive": {},
    "ridge": {
        "alpha": ("Regularisation α", "float", 10.0, 0.01, 1000.0, None, "L2 penalty strength (log scale)."),
    },
    "random_forest": {
        "n_estimators": ("Trees", "int", 400, 50, 1000, 50, "Number of trees."),
        "max_depth": ("Max depth (0 = unlimited)", "int", 12, 0, 40, 1, "Maximum tree depth."),
        "min_samples_leaf": ("Min samples per leaf", "int", 3, 1, 50, 1, "Larger values smooth predictions."),
        "max_features": ("Feature fraction per split", "float", 0.5, 0.1, 1.0, 0.05, "Share of features considered at each split."),
    },
    "extra_trees": {
        "n_estimators": ("Trees", "int", 400, 50, 1000, 50, "Number of trees."),
        "max_depth": ("Max depth (0 = unlimited)", "int", 14, 0, 40, 1, "Maximum tree depth."),
        "min_samples_leaf": ("Min samples per leaf", "int", 3, 1, 50, 1, "Larger values smooth predictions."),
        "max_features": ("Feature fraction per split", "float", 0.6, 0.1, 1.0, 0.05, "Share of features considered at each split."),
    },
    "catboost": {
        "iterations": ("Iterations", "int", 800, 100, 3000, 100, "Boosting rounds."),
        "learning_rate": ("Learning rate", "float", 0.05, 0.005, 0.3, 0.005, "Shrinkage per round."),
        "depth": ("Depth", "int", 6, 2, 10, 1, "Symmetric tree depth."),
        "l2_leaf_reg": ("L2 leaf regularisation", "float", 3.0, 0.1, 30.0, 0.1, "Leaf-value penalty."),
    },
    "xgboost": {
        "n_estimators": ("Trees", "int", 600, 50, 3000, 50, "Boosting rounds."),
        "learning_rate": ("Learning rate", "float", 0.05, 0.005, 0.3, 0.005, "Shrinkage per round."),
        "max_depth": ("Max depth", "int", 5, 2, 12, 1, "Tree depth."),
        "subsample": ("Row subsample", "float", 0.8, 0.3, 1.0, 0.05, "Row sampling per tree."),
        "colsample_bytree": ("Column subsample", "float", 0.8, 0.3, 1.0, 0.05, "Feature sampling per tree."),
        "min_child_weight": ("Min child weight", "float", 3.0, 0.0, 50.0, 0.5, "Minimum hessian per leaf."),
    },
    "svr": {
        "C": ("C", "float", 3.0, 0.01, 100.0, None, "Inverse regularisation (on the standardised target)."),
        "epsilon": ("ε", "float", 0.05, 0.0, 1.0, 0.01, "Insensitive-tube width (standardised target units)."),
        "gamma": ("γ (RBF)", "choice", "scale", ["scale", "auto"], None, None, "RBF kernel width."),
    },
}

ALGORITHMS = {
    "seasonal_naive": "Seasonal Naive Baseline (same weekday last year)",
    "ridge": "Ridge Regression",
    "random_forest": "Random Forest",
    "extra_trees": "Extra Trees",
    "catboost": "CatBoost",
    "xgboost": "XGBoost",
    "svr": "Support Vector Regression (RBF)",
}


# one-line description · whether it can drive the What-if Simulator · monotone support
ALGORITHM_INFO = {
    "seasonal_naive": ("Benchmark: Guests on the same weekday last year. Uses no aviation input, so it cannot respond to a scenario.", False, False),
    "ridge": ("Linear model with L2 penalty on standardised inputs. Fast and transparent; extrapolates linearly.", True, False),
    "random_forest": ("Averaged decision trees on bootstrap samples. Robust; cannot extrapolate beyond the training range.", True, True),
    "extra_trees": ("Like Random Forest with randomised split points — smoother, often less variance.", True, True),
    "catboost": ("Gradient-boosted symmetric trees with native categorical handling.", True, True),
    "xgboost": ("Gradient-boosted trees (histogram method).", True, True),
    "svr": ("Support Vector Regression with an RBF kernel on standardised inputs and target. Slow on the market panel.", True, False),
}
NEGATIVE_POLICIES = {"floor_zero": "Floor at zero", "none": "Keep raw values"}  # display labels; raw negatives are always counted


def available_algorithms() -> dict[str, str]:
    a = dict(ALGORITHMS)
    if not HAS_CATBOOST:
        a.pop("catboost")
    if not HAS_XGBOOST:
        a.pop("xgboost")
    return a


def default_params(algorithm: str) -> dict:
    return {k: v[2] for k, v in HYPERPARAMS[algorithm].items()}


class SeasonalNaive(BaseEstimator, RegressorMixin):
    """y_hat(t) = Guests(t-364); falls back to the 7-day mean a year earlier, then the
    training median of the market (or overall). Uses no other features."""

    def __init__(self, by_market: bool = False):
        self.by_market = by_market

    def fit(self, X: pd.DataFrame, y):
        y = pd.Series(np.asarray(y, dtype=float), index=X.index)
        self.global_median_ = float(np.nanmedian(y))
        self.market_median_ = y.groupby(X["market"]).median().to_dict() if self.by_market and "market" in X else {}
        return self

    def predict(self, X: pd.DataFrame):
        p = X["guests_lag364"].astype(float)
        if "guests_lag364_7d" in X:
            p = p.fillna(X["guests_lag364_7d"].astype(float))
        if self.by_market and "market" in X:
            p = p.fillna(X["market"].map(self.market_median_).astype(float))
        return p.fillna(self.global_median_).to_numpy()


def _to_str(X):
    return X.astype(str)


def _cat_cast(X: pd.DataFrame, cats: list[str]) -> pd.DataFrame:
    X = X.copy()
    for c in cats:
        X[c] = X[c].astype(str)
    return X


class CatBoostPipeline(BaseEstimator, RegressorMixin):
    """CatBoost with native categorical handling and column selection."""

    def __init__(self, features=None, categorical=None, params=None, monotone=None, random_state: int = config.RANDOM_STATE):
        self.features = features
        self.categorical = categorical
        self.params = params
        self.monotone = monotone
        self.random_state = random_state

    def fit(self, X, y):
        p = dict(self.params or {})
        if self.monotone:
            p["monotone_constraints"] = {f: 1 for f in self.monotone}
        self.model_ = CatBoostRegressor(loss_function="RMSE", random_seed=self.random_state, verbose=False,
                                        allow_writing_files=False, thread_count=-1, cat_features=list(self.categorical or []), **p)
        self.model_.fit(_cat_cast(X[self.features], self.categorical or []), np.asarray(y, dtype=float))
        return self

    def predict(self, X):
        return self.model_.predict(_cat_cast(X[self.features], self.categorical or []))

    def native_importance(self) -> dict[str, float]:
        return dict(zip(self.features, self.model_.get_feature_importance()))


class PrepModel(BaseEstimator, RegressorMixin):
    """Preprocessor + tree model whose monotone-constraint vector is aligned to the
    transformed columns at fit time (imputation indicators and one-hot columns stay free)."""

    def __init__(self, prep=None, model=None, monotone=None):
        self.prep = prep
        self.model = model
        self.monotone = monotone

    def fit(self, X, y):
        self.prep_ = clone(self.prep)
        Xt = self.prep_.fit_transform(X)
        self.names_ = list(self.prep_.get_feature_names_out())
        self.model_ = clone(self.model)
        if self.monotone:
            mono = {f"num__{f}" for f in self.monotone}
            cst = [1 if n in mono else 0 for n in self.names_]
            key = "monotonic_cst" if isinstance(self.model_, (RandomForestRegressor, ExtraTreesRegressor)) else "monotone_constraints"
            self.model_.set_params(**{key: cst if key == "monotonic_cst" else tuple(cst)})
        self.model_.fit(Xt, np.asarray(y, dtype=float))
        return self

    def predict(self, X):
        return self.model_.predict(self.prep_.transform(X))


def _preprocessor(num: list[str], cat: list[str], scale: bool) -> ColumnTransformer:
    num_steps = [("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True))]
    if scale:
        num_steps.append(("scale", StandardScaler()))
    transformers = [("num", Pipeline(num_steps), num)]
    if cat:
        transformers.append(("cat", Pipeline([("str", FunctionTransformer(_to_str, feature_names_out="one-to-one")),
                                              ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat))
    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=True)


MONOTONE_ALGORITHMS = ("random_forest", "extra_trees", "catboost", "xgboost")


class NonNegative(BaseEstimator, RegressorMixin):
    """Floors predictions at zero (a documented policy: guest counts cannot be negative)."""

    def __init__(self, estimator=None):
        self.estimator = estimator

    def fit(self, X, y):
        self.estimator_ = clone(self.estimator).fit(X, y)
        return self

    def predict(self, X):
        return np.maximum(self.estimator_.predict(X), 0.0)

    def predict_raw(self, X):
        return self.estimator_.predict(X)


def monotone_features(algorithm: str, features: list[str], monotone: bool) -> list[str]:
    from .features import MONOTONE_INCREASING
    return [f for f in numeric_features(features) if f in MONOTONE_INCREASING] if (monotone and algorithm in MONOTONE_ALGORITHMS) else []


def make_model(algorithm: str, features: list[str], params: dict | None = None, log_target: bool = False,
               by_market: bool = False, random_state: int = config.RANDOM_STATE, monotone: bool = False,
               negative_policy: str = "none") -> BaseEstimator:
    est = _make_model(algorithm, features, params, log_target, by_market, random_state, monotone)
    return NonNegative(est) if negative_policy == "floor_zero" and algorithm != "seasonal_naive" else est


def _make_model(algorithm: str, features: list[str], params: dict | None, log_target: bool, by_market: bool,
                random_state: int, monotone: bool) -> BaseEstimator:
    params = {**default_params(algorithm), **(params or {})}
    num, cat = numeric_features(features), categorical_features(features)
    mono = monotone_features(algorithm, features, monotone) or None

    if algorithm == "seasonal_naive":
        return SeasonalNaive(by_market=by_market)
    if algorithm == "catboost":
        if not HAS_CATBOOST:
            raise RuntimeError("CatBoost is not installed")
        est = CatBoostPipeline(features=list(features), categorical=cat, params=params, monotone=mono, random_state=random_state)
    elif algorithm == "ridge":
        est = Pipeline([("prep", _preprocessor(num, cat, scale=True)), ("model", Ridge(alpha=float(params["alpha"])))])
    elif algorithm == "random_forest":
        md = int(params["max_depth"]) or None
        est = PrepModel(_preprocessor(num, cat, scale=False),
                        RandomForestRegressor(n_estimators=int(params["n_estimators"]), max_depth=md,
                                              min_samples_leaf=int(params["min_samples_leaf"]),
                                              max_features=float(params["max_features"]),
                                              n_jobs=-1, random_state=random_state), mono)
    elif algorithm == "extra_trees":
        md = int(params["max_depth"]) or None
        est = PrepModel(_preprocessor(num, cat, scale=False),
                        ExtraTreesRegressor(n_estimators=int(params["n_estimators"]), max_depth=md,
                                            min_samples_leaf=int(params["min_samples_leaf"]),
                                            max_features=float(params["max_features"]),
                                            n_jobs=-1, random_state=random_state), mono)
    elif algorithm == "xgboost":
        if not HAS_XGBOOST:
            raise RuntimeError("XGBoost is not installed")
        est = PrepModel(_preprocessor(num, cat, scale=False),
                        XGBRegressor(n_estimators=int(params["n_estimators"]), learning_rate=float(params["learning_rate"]),
                                     max_depth=int(params["max_depth"]), subsample=float(params["subsample"]),
                                     colsample_bytree=float(params["colsample_bytree"]),
                                     min_child_weight=float(params["min_child_weight"]),
                                     tree_method="hist", n_jobs=-1, random_state=random_state), mono)
    elif algorithm == "svr":
        est = Pipeline([("prep", _preprocessor(num, cat, scale=True)),
                        ("model", SVR(kernel="rbf", C=float(params["C"]), epsilon=float(params["epsilon"]), gamma=params["gamma"]))])
        # SVR needs a standardised target; optional log is applied first.
        y_tf = Pipeline([("log", FunctionTransformer(np.log1p, np.expm1, check_inverse=False)), ("scale", StandardScaler())]) if log_target \
            else StandardScaler()
        return TransformedTargetRegressor(regressor=est, transformer=y_tf, check_inverse=False)
    else:
        raise ValueError(f"Unknown algorithm {algorithm}")

    if log_target:
        return TransformedTargetRegressor(regressor=est, func=np.log1p, inverse_func=np.expm1, check_inverse=False)
    return est


def native_importance(model: BaseEstimator, features: list[str]) -> dict[str, float] | None:
    """Importance mapped back to the ORIGINAL feature names (one-hot / indicator columns summed)."""
    m = model.estimator_ if isinstance(model, NonNegative) else model
    m = m.regressor_ if isinstance(m, TransformedTargetRegressor) else m
    if isinstance(m, SeasonalNaive):
        return None
    if isinstance(m, CatBoostPipeline):
        return m.native_importance()
    if isinstance(m, PrepModel):
        final, names = m.model_, m.names_
    elif isinstance(m, Pipeline):
        final, names = m.named_steps["model"], m.named_steps["prep"].get_feature_names_out()
    else:
        return None
    if hasattr(final, "feature_importances_"):
        vals = np.asarray(final.feature_importances_, dtype=float)
    elif hasattr(final, "coef_"):
        vals = np.ravel(final.coef_)  # signed: coefficient per standard deviation of the input
    else:
        return None
    out = {f: 0.0 for f in features}
    for n, v in zip(names, vals):
        raw = n.split("__", 1)[1]
        if raw.startswith("missingindicator_"):
            raw = raw[len("missingindicator_"):]
        base = next((f for f in sorted(features, key=len, reverse=True) if raw == f or raw.startswith(f + "_")), None)
        if base is not None:
            out[base] += float(v)
    return out


def importance_kind(algorithm: str) -> str:
    return {"ridge": "Coefficient on the standardised input (guests per +1 SD; summed over one-hot / indicator columns)",
            "random_forest": "Mean decrease in impurity (split importance)", "extra_trees": "Mean decrease in impurity (split importance)",
            "xgboost": "XGBoost gain importance", "catboost": "CatBoost prediction-values change",
            "svr": "Not available for an RBF kernel", "seasonal_naive": "Not applicable"}.get(algorithm, "")


def fresh(model: BaseEstimator) -> BaseEstimator:
    return clone(model)
