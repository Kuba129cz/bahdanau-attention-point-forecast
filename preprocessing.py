# src/preprocessing.py
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
from pathlib import Path
from typing import Union
import joblib

class Preprocessor:
    """
    Handles feature and target scaling for a single validation fold to prevent data leakage.
    """
    def __init__(self, lookback_cols: list[str], horizon_cols: list[str], target_col: str):
        """
        Initializes the Preprocessor with columns to be scaled and separate scalers.

        Args:
            lookback_cols: List of column names used in the lookback window.
            horizon_cols: List of column names used in the horizon window.
            target_col: Name of the target variable column.
        """
        self.lookback_cols = lookback_cols
        self.horizon_cols = horizon_cols
        self.target_col = target_col
        
        self.lookback_cols_to_scale = [col for col in self.lookback_cols if not (col.startswith("sin_") or col.startswith("cos_"))]
        self.horizon_cols_to_scale = [col for col in self.horizon_cols if not (col.startswith("sin_") or col.startswith("cos_"))]
        
        self.lookback_scaler = StandardScaler()
        self.horizon_scaler = StandardScaler()
        self.target_scaler = StandardScaler()

    @property
    def target_std(self):
        """
        Returns the scaling factor (standard deviation) for the target.
        """
        if not hasattr(self.target_scaler, "scale_"):
            raise ValueError("Scaler has not been fitted yet. Run process_fold first.")
        return self.target_scaler.scale_[0]

    @property
    def scaled_zero(self):
        """
        Returns the value of 0.0 in the target variable space mapped to the scaled space.
        """
        if not hasattr(self.target_scaler, "mean_"):
            raise ValueError("Scaler has not been fitted yet. Run process_fold first.")
        # Z-score: (x - mean) / std
        return (0.0 - self.target_scaler.mean_[0]) / self.target_scaler.scale_[0]
    
    def process_fold(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Fits scalers on training data and transforms both training and validation sets.

        Args:
            train_df: Training DataFrame containing all necessary features.
            val_df: Validation DataFrame.

        Returns:
            A tuple of (train_scaled, val_scaled) DataFrames.
        """
        train_scaled = train_df.copy()
        val_scaled = val_df.copy()

        if self.lookback_cols_to_scale:
            train_scaled[self.lookback_cols_to_scale] = self.lookback_scaler.fit_transform(train_df[self.lookback_cols_to_scale])
            val_scaled[self.lookback_cols_to_scale] = self.lookback_scaler.transform(val_df[self.lookback_cols_to_scale])

        if self.horizon_cols_to_scale:
            train_scaled[self.horizon_cols_to_scale] = self.horizon_scaler.fit_transform(train_df[self.horizon_cols_to_scale])
            val_scaled[self.horizon_cols_to_scale] = self.horizon_scaler.transform(val_df[self.horizon_cols_to_scale])

        train_scaled[[self.target_col]] = self.target_scaler.fit_transform(train_df[[self.target_col]])
        val_scaled[[self.target_col]] = self.target_scaler.transform(val_df[[self.target_col]])

        return train_scaled, val_scaled

    def save_scalers(self, save_dir: Union[str, Path] = "checkpoints/scalers"):
        """
        Saves fitted scalers and metadata to disk using joblib.
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True) 
        
        joblib.dump(self.lookback_scaler, save_dir / "lookback_scaler.pkl")
        joblib.dump(self.horizon_scaler, save_dir / "horizon_scaler.pkl")
        joblib.dump(self.target_scaler, save_dir / "target_scaler.pkl")
        
        joblib.dump(self.lookback_cols_to_scale, save_dir / "scaled_lookback_cols.pkl")
        joblib.dump(self.horizon_cols_to_scale, save_dir / "scaled_horizon_cols.pkl")
    
    def load_scalers(self, save_dir: Path):
        """
        Loads fitted scalers and metadata from disk to restore state.
        """
        save_dir = Path(save_dir)

        lookback_scaler_path = save_dir / "lookback_scaler.pkl"
        horizon_scaler_path = save_dir / "horizon_scaler.pkl"
        target_scaler_path = save_dir / "target_scaler.pkl"

        lookback_cols_to_scale_path = save_dir / "scaled_lookback_cols.pkl"
        horizon_cols_to_scale_path = save_dir / "scaled_horizon_cols.pkl"

        if not (lookback_scaler_path.exists() and horizon_scaler_path.exists() and target_scaler_path.exists()):
            raise FileNotFoundError(f"No scaler files found in the folder: {save_dir}!")
        
        if not (lookback_cols_to_scale_path.exists() and horizon_cols_to_scale_path.exists()):
            raise FileNotFoundError(f"No lookback or horizon cols files found in the folder: {save_dir}!")

        self.lookback_scaler = joblib.load(lookback_scaler_path)
        self.horizon_scaler = joblib.load(horizon_scaler_path)
        self.target_scaler = joblib.load(target_scaler_path)

        self.lookback_cols_to_scale = joblib.load(lookback_cols_to_scale_path)
        self.horizon_cols_to_scale = joblib.load(horizon_cols_to_scale_path)
    
    def transform(self, lookback_df: pd.DataFrame, horizon_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Transforms input DataFrames using pre-fitted scalers for inference.

        Args:
            lookback_df: Raw lookback DataFrame.
            horizon_df: Raw horizon DataFrame.

        Returns:
            A tuple of (lookback_df_scaled, horizon_df_scaled).
        """
        lookback_df_scaled = lookback_df.copy()
        horizon_df_scaled = horizon_df.copy()

        if self.lookback_cols_to_scale:
            lookback_df_scaled[self.lookback_cols_to_scale] = self.lookback_scaler.transform(lookback_df[self.lookback_cols_to_scale])
        
        if self.horizon_cols_to_scale:
            horizon_df_scaled[self.horizon_cols_to_scale] = self.horizon_scaler.transform(horizon_df[self.horizon_cols_to_scale])
            
        return lookback_df_scaled, horizon_df_scaled
    
    def inverse_transform_target(self, y_scaled: np.array):
        """
        Inverse transforms scaled target predictions back to original physical units.

        Args:
            y_scaled: Predictions as numpy array with shape (batch, seq, target).

        Returns:
            Inverse transformed data in the original input shape.
        """
        orig_shape = y_scaled.shape
        
        y_flat = y_scaled.reshape(-1, 1)
        y_df = pd.DataFrame(y_flat, columns=[self.target_col])
        y_inv_flat = self.target_scaler.inverse_transform(y_df)
        y_inverse = y_inv_flat.reshape(orig_shape)

        return y_inverse
    
    def transform_target(self, y_raw: Union[pd.DataFrame, pd.Series, np.ndarray]) -> np.ndarray:
        """
        Transforms raw target values (e.g., actual energy in kWh) into scaled target space.

        Args:
            y_raw: Raw target values as DataFrame, Series, or NumPy array.

        Returns:
            Scaled target values matching the original input shape.
        """
        if not hasattr(self.target_scaler, "mean_"):
            raise ValueError("Target scaler has not been fitted or loaded yet!")
        
        if isinstance(y_raw, (pd.DataFrame, pd.Series)):
            values = y_raw.values
        else:
            values = y_raw
        
        orig_shape = values.shape
        y_flat = values.reshape(-1, 1)
        y_df = pd.DataFrame(y_flat, columns=[self.target_col])
        y_scaled_flat = self.target_scaler.transform(y_df)

        return y_scaled_flat.reshape(orig_shape)

