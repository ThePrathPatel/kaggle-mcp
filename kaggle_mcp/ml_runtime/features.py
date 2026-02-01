"""Feature engineering utilities."""

from typing import Optional, Any
from dataclasses import dataclass, field
import numpy as np
import pandas as pd


@dataclass
class FeatureConfig:
    """Configuration for feature engineering."""
    handle_missing: bool = True
    encode_categoricals: bool = True
    scale_numerics: bool = True
    create_interactions: bool = False
    polynomial_degree: int = 1


@dataclass
class FeatureInfo:
    """Information about engineered features."""
    original_columns: list[str]
    numeric_columns: list[str]
    categorical_columns: list[str]
    engineered_columns: list[str]
    total_features: int
    missing_stats: dict[str, float] = field(default_factory=dict)


class FeatureEngineer:
    """Automated feature engineering for tabular data."""

    def __init__(self, config: Optional[FeatureConfig] = None):
        self.config = config or FeatureConfig()
        self._encoders: dict[str, Any] = {}
        self._scaler = None
        self._fitted = False
        self._feature_info: Optional[FeatureInfo] = None

    def analyze_dataframe(self, df: pd.DataFrame) -> dict:
        """Analyze a dataframe and return statistics."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

        analysis = {
            "shape": df.shape,
            "columns": df.columns.tolist(),
            "numeric_columns": numeric_cols,
            "categorical_columns": categorical_cols,
            "missing_values": df.isnull().sum().to_dict(),
            "missing_percentage": (df.isnull().sum() / len(df) * 100).to_dict(),
            "dtypes": df.dtypes.astype(str).to_dict(),
        }

        # Numeric statistics
        if numeric_cols:
            analysis["numeric_stats"] = df[numeric_cols].describe().to_dict()

        # Categorical statistics
        if categorical_cols:
            analysis["categorical_stats"] = {
                col: {
                    "unique_count": df[col].nunique(),
                    "top_values": df[col].value_counts().head(5).to_dict(),
                }
                for col in categorical_cols
            }

        return analysis

    def fit(self, df: pd.DataFrame, target_column: Optional[str] = None) -> "FeatureEngineer":
        """Fit the feature engineer on training data."""
        # Exclude target column from features
        feature_df = df.drop(columns=[target_column]) if target_column and target_column in df.columns else df

        numeric_cols = feature_df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = feature_df.select_dtypes(include=["object", "category"]).columns.tolist()

        # Fit encoders for categorical columns
        if self.config.encode_categoricals and categorical_cols:
            from sklearn.preprocessing import LabelEncoder
            for col in categorical_cols:
                le = LabelEncoder()
                # Handle missing values by converting to string
                values = feature_df[col].fillna("__MISSING__").astype(str)
                le.fit(values)
                self._encoders[col] = le

        # Fit scaler for numeric columns
        if self.config.scale_numerics and numeric_cols:
            from sklearn.preprocessing import StandardScaler
            self._scaler = StandardScaler()
            # Handle missing values with mean for fitting
            numeric_data = feature_df[numeric_cols].fillna(feature_df[numeric_cols].mean())
            self._scaler.fit(numeric_data)

        self._feature_info = FeatureInfo(
            original_columns=feature_df.columns.tolist(),
            numeric_columns=numeric_cols,
            categorical_columns=categorical_cols,
            engineered_columns=[],
            total_features=len(numeric_cols) + len(categorical_cols),
            missing_stats=(feature_df.isnull().sum() / len(feature_df)).to_dict(),
        )

        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame, target_column: Optional[str] = None) -> pd.DataFrame:
        """Transform data using fitted feature engineer."""
        if not self._fitted:
            raise RuntimeError("FeatureEngineer must be fitted before transform")

        # Separate target if present
        target = None
        if target_column and target_column in df.columns:
            target = df[target_column]
            df = df.drop(columns=[target_column])

        result = df.copy()

        # Handle missing values for numeric columns
        if self.config.handle_missing and self._feature_info:
            for col in self._feature_info.numeric_columns:
                if col in result.columns:
                    result[col] = result[col].fillna(result[col].median())

        # Encode categorical columns
        if self.config.encode_categoricals:
            for col, encoder in self._encoders.items():
                if col in result.columns:
                    values = result[col].fillna("__MISSING__").astype(str)
                    # Handle unseen categories
                    mask = ~values.isin(encoder.classes_)
                    if mask.any():
                        values = values.copy()
                        values[mask] = encoder.classes_[0]  # Use first class as fallback
                    result[col] = encoder.transform(values)

        # Scale numeric columns
        if self.config.scale_numerics and self._scaler and self._feature_info:
            numeric_cols = [c for c in self._feature_info.numeric_columns if c in result.columns]
            if numeric_cols:
                result[numeric_cols] = self._scaler.transform(result[numeric_cols])

        # Add target back if present
        if target is not None:
            result[target_column] = target

        return result

    def fit_transform(
        self,
        df: pd.DataFrame,
        target_column: Optional[str] = None,
    ) -> pd.DataFrame:
        """Fit and transform in one step."""
        self.fit(df, target_column)
        return self.transform(df, target_column)

    def get_feature_names(self) -> list[str]:
        """Get list of feature names after transformation."""
        if not self._fitted or not self._feature_info:
            return []
        return self._feature_info.numeric_columns + self._feature_info.categorical_columns

    def get_feature_info(self) -> Optional[FeatureInfo]:
        """Get feature information."""
        return self._feature_info


def detect_target_type(y) -> str:
    """Detect if target is classification or regression."""
    if isinstance(y, np.ndarray):
        unique_values = len(np.unique(y))
    else:
        unique_values = y.nunique()

    # If very few unique values, likely classification
    if unique_values <= 20:
        return "classification"

    # If values are integers and few unique, likely classification
    if y.dtype in [np.int64, np.int32] and unique_values <= 50:
        return "classification"

    # Otherwise assume regression
    return "regression"


def detect_task_type(y) -> str:
    """Detect the ML task type."""
    if isinstance(y, np.ndarray):
        unique_values = len(np.unique(y))
    else:
        unique_values = y.nunique()

    if unique_values == 2:
        return "binary_classification"
    elif unique_values <= 20 or (y.dtype in [np.int64, np.int32] and unique_values <= 50):
        return "multiclass_classification"
    else:
        return "regression"
