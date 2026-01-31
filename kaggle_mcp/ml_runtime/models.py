"""Model factory for ML models."""

from enum import Enum
from typing import Any, Optional
from dataclasses import dataclass


class ModelType(str, Enum):
    """Supported model types."""
    LOGISTIC_REGRESSION = "logistic_regression"
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    LINEAR_REGRESSION = "linear_regression"
    RIDGE = "ridge"
    LASSO = "lasso"
    SVM = "svm"
    KNN = "knn"


class TaskType(str, Enum):
    """ML task types."""
    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"
    REGRESSION = "regression"


@dataclass
class ModelConfig:
    """Configuration for a model."""
    model_type: ModelType
    task_type: TaskType
    hyperparameters: dict[str, Any]
    random_state: int = 42


DEFAULT_HYPERPARAMETERS = {
    ModelType.LOGISTIC_REGRESSION: {
        "C": 1.0,
        "max_iter": 1000,
        "solver": "lbfgs",
    },
    ModelType.RANDOM_FOREST: {
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "n_jobs": -1,
    },
    ModelType.GRADIENT_BOOSTING: {
        "n_estimators": 100,
        "max_depth": 5,
        "learning_rate": 0.1,
        "subsample": 0.8,
    },
    ModelType.XGBOOST: {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "n_jobs": -1,
    },
    ModelType.LIGHTGBM: {
        "n_estimators": 100,
        "max_depth": -1,
        "num_leaves": 31,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "n_jobs": -1,
        "verbose": -1,
    },
    ModelType.LINEAR_REGRESSION: {},
    ModelType.RIDGE: {
        "alpha": 1.0,
    },
    ModelType.LASSO: {
        "alpha": 1.0,
        "max_iter": 1000,
    },
    ModelType.SVM: {
        "C": 1.0,
        "kernel": "rbf",
    },
    ModelType.KNN: {
        "n_neighbors": 5,
        "weights": "uniform",
        "n_jobs": -1,
    },
}


class ModelFactory:
    """Factory for creating ML models."""

    @staticmethod
    def create_model(
        model_type: ModelType,
        task_type: TaskType,
        hyperparameters: Optional[dict[str, Any]] = None,
        random_state: int = 42,
    ):
        """Create a model instance."""
        # Merge default and custom hyperparameters
        params = DEFAULT_HYPERPARAMETERS.get(model_type, {}).copy()
        if hyperparameters:
            params.update(hyperparameters)

        # Add random state where applicable
        if "random_state" not in params and model_type not in [ModelType.KNN]:
            params["random_state"] = random_state

        if model_type == ModelType.LOGISTIC_REGRESSION:
            from sklearn.linear_model import LogisticRegression
            return LogisticRegression(**params)

        elif model_type == ModelType.RANDOM_FOREST:
            if task_type == TaskType.REGRESSION:
                from sklearn.ensemble import RandomForestRegressor
                return RandomForestRegressor(**params)
            else:
                from sklearn.ensemble import RandomForestClassifier
                return RandomForestClassifier(**params)

        elif model_type == ModelType.GRADIENT_BOOSTING:
            if task_type == TaskType.REGRESSION:
                from sklearn.ensemble import GradientBoostingRegressor
                return GradientBoostingRegressor(**params)
            else:
                from sklearn.ensemble import GradientBoostingClassifier
                return GradientBoostingClassifier(**params)

        elif model_type == ModelType.XGBOOST:
            import xgboost as xgb
            if task_type == TaskType.REGRESSION:
                return xgb.XGBRegressor(**params)
            else:
                return xgb.XGBClassifier(**params)

        elif model_type == ModelType.LIGHTGBM:
            import lightgbm as lgb
            if task_type == TaskType.REGRESSION:
                return lgb.LGBMRegressor(**params)
            else:
                return lgb.LGBMClassifier(**params)

        elif model_type == ModelType.LINEAR_REGRESSION:
            from sklearn.linear_model import LinearRegression
            return LinearRegression(**params)

        elif model_type == ModelType.RIDGE:
            from sklearn.linear_model import Ridge
            return Ridge(**params)

        elif model_type == ModelType.LASSO:
            from sklearn.linear_model import Lasso
            return Lasso(**params)

        elif model_type == ModelType.SVM:
            if task_type == TaskType.REGRESSION:
                from sklearn.svm import SVR
                return SVR(**params)
            else:
                from sklearn.svm import SVC
                params["probability"] = True
                return SVC(**params)

        elif model_type == ModelType.KNN:
            if task_type == TaskType.REGRESSION:
                from sklearn.neighbors import KNeighborsRegressor
                return KNeighborsRegressor(**params)
            else:
                from sklearn.neighbors import KNeighborsClassifier
                return KNeighborsClassifier(**params)

        else:
            raise ValueError(f"Unknown model type: {model_type}")

    @staticmethod
    def get_default_hyperparameters(model_type: ModelType) -> dict[str, Any]:
        """Get default hyperparameters for a model type."""
        return DEFAULT_HYPERPARAMETERS.get(model_type, {}).copy()

    @staticmethod
    def get_tuning_space(model_type: ModelType) -> dict[str, Any]:
        """Get hyperparameter tuning space for Optuna."""
        spaces = {
            ModelType.XGBOOST: {
                "n_estimators": ("int", 50, 500),
                "max_depth": ("int", 3, 10),
                "learning_rate": ("float_log", 0.01, 0.3),
                "subsample": ("float", 0.6, 1.0),
                "colsample_bytree": ("float", 0.6, 1.0),
                "reg_alpha": ("float_log", 1e-8, 10.0),
                "reg_lambda": ("float_log", 1e-8, 10.0),
            },
            ModelType.LIGHTGBM: {
                "n_estimators": ("int", 50, 500),
                "num_leaves": ("int", 20, 150),
                "learning_rate": ("float_log", 0.01, 0.3),
                "subsample": ("float", 0.6, 1.0),
                "colsample_bytree": ("float", 0.6, 1.0),
                "reg_alpha": ("float_log", 1e-8, 10.0),
                "reg_lambda": ("float_log", 1e-8, 10.0),
            },
            ModelType.RANDOM_FOREST: {
                "n_estimators": ("int", 50, 300),
                "max_depth": ("int", 3, 20),
                "min_samples_split": ("int", 2, 20),
                "min_samples_leaf": ("int", 1, 10),
            },
            ModelType.GRADIENT_BOOSTING: {
                "n_estimators": ("int", 50, 300),
                "max_depth": ("int", 3, 10),
                "learning_rate": ("float_log", 0.01, 0.3),
                "subsample": ("float", 0.6, 1.0),
            },
        }
        return spaces.get(model_type, {})
