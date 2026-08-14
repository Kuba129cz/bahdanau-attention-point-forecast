import json
import pathlib
from functools import lru_cache
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

import preprocessing
from model import Model 


@lru_cache(maxsize=128)
def _load_config_cached(config_file_str: str) -> SimpleNamespace:
    config_file = pathlib.Path(config_file_str)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found at: {config_file}")

    with open(config_file, mode="r", encoding="utf-8") as f:
        config_dict = json.load(f)

    return SimpleNamespace(**config_dict)


@lru_cache(maxsize=128)
def _load_preprocessor_cached(scalers_dir_str: str, config_file_str: str) -> preprocessing.Preprocessor:
    args = _load_config_cached(config_file_str)
    scalers_dir = pathlib.Path(scalers_dir_str)

    preprocessor = preprocessing.Preprocessor(
        lookback_cols=args.lookback_cols, 
        horizon_cols=args.horizon_cols, 
        target_col=args.target_col
    )
    preprocessor.load_scalers(save_dir=scalers_dir)
    return preprocessor


class EnsemblePredictor:
    def __init__(
        self, 
        config_path: str = "config.json", 
        models_dir: str = "models", 
        scalers_dir: str = "scalers"
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config_path = pathlib.Path(config_path)
        self.models_dir = pathlib.Path(models_dir)
        self.scalers_dir = pathlib.Path(scalers_dir)
        
        self.args = _load_config_cached(str(self.config_path))
        self.preprocessor = _load_preprocessor_cached(
            scalers_dir_str=str(self.scalers_dir), 
            config_file_str=str(self.config_path)
        )
        self.models = self._load_models(extension="*.pth")

    def _load_models(self, extension: str = "*.pth") -> list[torch.nn.Module]:
        if not self.models_dir.exists():
            raise FileNotFoundError(f"Directory not found: {self.models_dir}")

        model_paths = sorted(self.models_dir.glob(extension))

        if not model_paths:
            raise FileNotFoundError(f"No model files matching '{extension}' found in: {self.models_dir}")

        models = []
        for path in model_paths:
            try:
                model = Model(args=self.args)
                state_dict = torch.load(f=path, map_location=self.device)
                model.load_state_dict(state_dict)
                model.to(device=self.device)
                model.eval()
                models.append(model)
            except Exception as e:
                raise RuntimeError(f"Failed to load PyTorch model {path.name}: {str(e)}")

        return models

    def predict(
        self, 
        df_history: pd.DataFrame, 
        df_future: pd.DataFrame, 
        df_time: pd.Series, 
        known_targets_df: pd.DataFrame = None
    ) -> list[dict]:
        x_past_df, x_future_df = self.preprocessor.transform(df_history, df_future)
        
        if known_targets_df is not None and not known_targets_df.empty:
            known_targets_clean = known_targets_df[[self.args.target_col]]
            known_targets_np = self.preprocessor.transform_target(known_targets_clean)
            known_targets = torch.from_numpy(known_targets_np).float().unsqueeze(0).to(self.device)
        else:
            known_targets = None
            
        x_past = torch.from_numpy(x_past_df.to_numpy()).float().unsqueeze(0).to(self.device)
        x_future = torch.from_numpy(x_future_df.to_numpy()).float().unsqueeze(0).to(self.device)

        predictions = []
        night_mask = df_future['irradiance_0'].values <= 0.0
        
        with torch.no_grad():
            for model in self.models:
                preds = model(x_past=x_past, x_future=x_future, known_targets=known_targets)
                
                preds = preds.squeeze(dim=0).cpu().numpy()
                preds = self.preprocessor.inverse_transform_target(preds).flatten()

                preds = np.clip(preds, 0, None)
                preds[night_mask] = 0.0
                
                predictions.append(preds)

        hourly_power = np.mean(predictions, axis=0)

        if df_time.dt.tz is None:
            df_time_utc = df_time.dt.tz_localize("UTC")
        else:
            df_time_utc = df_time.dt.tz_convert("UTC")

        result = [
            {
                "time": t.isoformat(),
                "power": round(float(avg_p), 2),
                "model_predictions": [round(float(m_preds[i]), 2) for m_preds in predictions]
            }
            for i, (t, avg_p) in enumerate(zip(df_time_utc, hourly_power))
        ]

        return result