"""ML Trainer for model training and evaluation."""

import asyncio
import pickle
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Callable
import numpy as np
import pandas as pd

from .models import ModelFactory, ModelType, TaskType
from .features import FeatureEngineer, detect_task_type


@dataclass
class TrainingResult:
    """Result of a training run."""
    run_id: str
    model_type: str
    cv_score: float
    cv_std: float
    cv_scores: list[float]
    training_time_seconds: float
    hyperparameters: dict[str, Any]
    feature_importance: Optional[dict[str, float]] = None
    artifact_path: Optional[str] = None
    predictions: Optional[np.ndarray] = None
    oof_predictions: Optional[np.ndarray] = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "run_id": self.run_id,
            "model_type": self.model_type,
            "cv_score": self.cv_score,
            "cv_std": self.cv_std,
            "cv_scores": self.cv_scores,
            "training_time_seconds": self.training_time_seconds,
            "hyperparameters": self.hyperparameters,
            "feature_importance": self.feature_importance,
            "artifact_path": self.artifact_path,
        }


@dataclass
class TrainingConfig:
    """Configuration for training."""
    n_folds: int = 5
    shuffle: bool = True
    random_state: int = 42
    early_stopping_rounds: Optional[int] = 50
    verbose: bool = False
    save_models: bool = True
    artifact_dir: Path = field(default_factory=lambda: Path("./kaggle_data/models"))


class MLTrainer:
    """Trainer for ML models with cross-validation."""

    def __init__(
        self,
        config: Optional[TrainingConfig] = None,
        timeout_callback: Optional[Callable[[], bool]] = None,
    ):
        self.config = config or TrainingConfig()
        self.timeout_callback = timeout_callback
        self._feature_engineer: Optional[FeatureEngineer] = None

    async def train(
        self,
        train_df: pd.DataFrame,
        target_column: str,
        model_type: ModelType,
        hyperparameters: Optional[dict[str, Any]] = None,
        feature_columns: Optional[list[str]] = None,
        metric: str = "auto",
    ) -> TrainingResult:
        """Train a model with cross-validation."""
        run_id = str(uuid.uuid4())[:8]
        start_time = datetime.now()

        # Prepare data
        X, y, feature_names = self._prepare_data(train_df, target_column, feature_columns)

        # Detect task type
        task_type_str = detect_task_type(y)
        task_type = TaskType(task_type_str)

        # Determine metric
        if metric == "auto":
            if task_type == TaskType.REGRESSION:
                metric = "neg_root_mean_squared_error"
            elif task_type == TaskType.BINARY_CLASSIFICATION:
                metric = "roc_auc"
            else:
                metric = "accuracy"

        # Run cross-validation in executor to not block
        loop = asyncio.get_event_loop()
        cv_result = await loop.run_in_executor(
            None,
            self._run_cv,
            X,
            y,
            model_type,
            task_type,
            hyperparameters,
            metric,
        )

        training_time = (datetime.now() - start_time).total_seconds()

        # Get feature importance
        feature_importance = self._get_feature_importance(
            cv_result["models"][0] if cv_result["models"] else None,
            feature_names,
        )

        # Save model if configured
        artifact_path = None
        if self.config.save_models and cv_result["models"]:
            artifact_path = self._save_model(
                run_id,
                cv_result["models"],
                model_type,
                hyperparameters or {},
            )

        return TrainingResult(
            run_id=run_id,
            model_type=model_type.value,
            cv_score=cv_result["mean_score"],
            cv_std=cv_result["std_score"],
            cv_scores=cv_result["scores"],
            training_time_seconds=training_time,
            hyperparameters=hyperparameters or ModelFactory.get_default_hyperparameters(model_type),
            feature_importance=feature_importance,
            artifact_path=artifact_path,
            oof_predictions=cv_result.get("oof_predictions"),
        )

    def _prepare_data(
        self,
        df: pd.DataFrame,
        target_column: str,
        feature_columns: Optional[list[str]] = None,
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Prepare data for training."""
        if feature_columns:
            X = df[feature_columns].copy()
        else:
            X = df.drop(columns=[target_column]).copy()

        y = df[target_column].values

        # Apply feature engineering
        self._feature_engineer = FeatureEngineer()
        X = self._feature_engineer.fit_transform(X)

        feature_names = X.columns.tolist()

        return X.values, y, feature_names

    def _run_cv(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_type: ModelType,
        task_type: TaskType,
        hyperparameters: Optional[dict[str, Any]],
        metric: str,
    ) -> dict:
        """Run cross-validation."""
        from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score

        # Choose CV strategy
        if task_type in [TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION]:
            cv = StratifiedKFold(
                n_splits=self.config.n_folds,
                shuffle=self.config.shuffle,
                random_state=self.config.random_state,
            )
        else:
            cv = KFold(
                n_splits=self.config.n_folds,
                shuffle=self.config.shuffle,
                random_state=self.config.random_state,
            )

        # Create model
        model = ModelFactory.create_model(
            model_type,
            task_type,
            hyperparameters,
            random_state=self.config.random_state,
        )

        # Run cross-validation
        scores = cross_val_score(
            model,
            X,
            y,
            cv=cv,
            scoring=metric,
            n_jobs=-1,
        )

        # Handle negative scores (sklearn convention for some metrics)
        if metric.startswith("neg_"):
            scores = -scores

        # Train models for each fold to get OOF predictions
        models = []
        oof_predictions = np.zeros(len(y))

        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
            if self.timeout_callback and self.timeout_callback():
                break

            fold_model = ModelFactory.create_model(
                model_type,
                task_type,
                hyperparameters,
                random_state=self.config.random_state,
            )

            fold_model.fit(X[train_idx], y[train_idx])
            models.append(fold_model)

            # Get OOF predictions
            if task_type in [TaskType.BINARY_CLASSIFICATION]:
                if hasattr(fold_model, "predict_proba"):
                    oof_predictions[val_idx] = fold_model.predict_proba(X[val_idx])[:, 1]
                else:
                    oof_predictions[val_idx] = fold_model.predict(X[val_idx])
            else:
                oof_predictions[val_idx] = fold_model.predict(X[val_idx])

        return {
            "scores": scores.tolist(),
            "mean_score": float(np.mean(scores)),
            "std_score": float(np.std(scores)),
            "models": models,
            "oof_predictions": oof_predictions,
        }

    def _get_feature_importance(
        self,
        model: Any,
        feature_names: list[str],
    ) -> Optional[dict[str, float]]:
        """Extract feature importance from model."""
        if model is None:
            return None

        importance = None

        if hasattr(model, "feature_importances_"):
            importance = model.feature_importances_
        elif hasattr(model, "coef_"):
            importance = np.abs(model.coef_).flatten()

        if importance is not None and len(importance) == len(feature_names):
            # Normalize
            total = np.sum(importance)
            if total > 0:
                importance = importance / total
            return dict(zip(feature_names, importance.tolist()))

        return None

    def _save_model(
        self,
        run_id: str,
        models: list,
        model_type: ModelType,
        hyperparameters: dict,
    ) -> str:
        """Save trained models."""
        self.config.artifact_dir.mkdir(parents=True, exist_ok=True)

        artifact_path = self.config.artifact_dir / f"{run_id}_{model_type.value}.pkl"

        artifact = {
            "run_id": run_id,
            "model_type": model_type.value,
            "hyperparameters": hyperparameters,
            "models": models,
            "feature_engineer": self._feature_engineer,
            "created_at": datetime.now().isoformat(),
        }

        with open(artifact_path, "wb") as f:
            pickle.dump(artifact, f)

        return str(artifact_path)

    async def predict(
        self,
        test_df: pd.DataFrame,
        artifact_path: str,
        feature_columns: Optional[list[str]] = None,
    ) -> np.ndarray:
        """Generate predictions using saved models."""
        with open(artifact_path, "rb") as f:
            artifact = pickle.load(f)

        models = artifact["models"]
        feature_engineer = artifact.get("feature_engineer")

        # Prepare test data
        if feature_columns:
            X = test_df[feature_columns].copy()
        else:
            X = test_df.copy()

        # Apply feature engineering
        if feature_engineer:
            X = feature_engineer.transform(X)

        X = X.values

        # Average predictions from all fold models
        predictions = np.zeros(len(X))
        for model in models:
            if hasattr(model, "predict_proba"):
                predictions += model.predict_proba(X)[:, 1]
            else:
                predictions += model.predict(X)

        predictions /= len(models)

        return predictions

    async def tune_hyperparameters(
        self,
        train_df: pd.DataFrame,
        target_column: str,
        model_type: ModelType,
        n_trials: int = 50,
        metric: str = "auto",
        timeout: Optional[int] = None,
    ) -> dict[str, Any]:
        """Tune hyperparameters using Optuna."""
        import optuna
        from optuna.samplers import TPESampler

        # Prepare data
        X, y, feature_names = self._prepare_data(train_df, target_column)

        # Detect task type
        task_type_str = detect_task_type(y)
        task_type = TaskType(task_type_str)

        # Determine metric and direction
        if metric == "auto":
            if task_type == TaskType.REGRESSION:
                metric = "neg_root_mean_squared_error"
            elif task_type == TaskType.BINARY_CLASSIFICATION:
                metric = "roc_auc"
            else:
                metric = "accuracy"

        direction = "maximize"
        if metric.startswith("neg_"):
            direction = "maximize"  # sklearn negates, we negate again

        tuning_space = ModelFactory.get_tuning_space(model_type)

        def objective(trial: optuna.Trial) -> float:
            params = {}
            for param_name, param_config in tuning_space.items():
                param_type = param_config[0]
                if param_type == "int":
                    params[param_name] = trial.suggest_int(param_name, param_config[1], param_config[2])
                elif param_type == "float":
                    params[param_name] = trial.suggest_float(param_name, param_config[1], param_config[2])
                elif param_type == "float_log":
                    params[param_name] = trial.suggest_float(param_name, param_config[1], param_config[2], log=True)
                elif param_type == "categorical":
                    params[param_name] = trial.suggest_categorical(param_name, param_config[1])

            result = self._run_cv(X, y, model_type, task_type, params, metric)
            return result["mean_score"]

        # Run optimization
        sampler = TPESampler(seed=self.config.random_state)
        study = optuna.create_study(direction=direction, sampler=sampler)

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=True),
        )

        return {
            "best_params": study.best_params,
            "best_score": study.best_value,
            "n_trials": len(study.trials),
        }
