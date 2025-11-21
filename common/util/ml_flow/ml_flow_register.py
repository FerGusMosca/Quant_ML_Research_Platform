# ml_flow_register.py
# Utility to register XGBoost/RandomForest models in MLflow
# All comments MUST be in English

import os
import mlflow
from pathlib import Path

from common.util.std_in_out.ml_settings_loader import MLSettingsLoader


class MLflowModelRegistrar:
    """
    Utility responsible for:
      - Parsing model file names
      - Creating MLflow experiments
      - Starting MLflow runs
      - Registering all artifacts (pkl, booster.json, scaler, label_encoder)
    """

    def __init__(self):
        # Load MLflow URI from commands_mgr.ini
        loader = MLSettingsLoader()
        config_settings = loader.load_settings("./configs/commands_mgr.ini")

        mlflow_uri = config_settings.get("MLFLOW_TRACKING_URI")
        if not mlflow_uri:
            raise RuntimeError("MLFLOW_TRACKING_URI missing in commands_mgr.ini")

        # Apply tracking URI
        mlflow.set_tracking_uri(mlflow_uri)

    # ---------------------------------------------------------
    # Parse a model filename like:
    #   xgb_model_XLK_2010_2018_M7.pkl
    # ---------------------------------------------------------
    def parse_model_filename(self, model_output: str) -> dict:
        """
        Example: xgb_model_XLK_2010_2018_M7.pkl

        Returns:
            {
              "symbol": "XLK",
              "start": "2010",
              "end": "2018",
              "model_version": "M7",
              "base_name": "xgb_model_XLK_2010_2018_M7"
            }
        """
        base = Path(model_output).stem  # xgb_model_XLK_2010_2018_M7
        parts = base.split("_")

        if len(parts) < 6:
            raise ValueError(f"Invalid model filename: {model_output}")

        # Pattern:
        # 0: xgb
        # 1: model
        # 2: SYMBOL
        # 3: START
        # 4: END
        # 5: M7
        symbol = parts[2]
        start = parts[3]
        end = parts[4]
        model_version = parts[5]

        return {
            "symbol": symbol,
            "start": start,
            "end": end,
            "model_version": model_version,
            "base_name": base
        }

    # ---------------------------------------------------------
    # Create or reuse an MLflow experiment
    # ---------------------------------------------------------
    def start_experiment(self, algo: str, symbol: str, start: str, end: str):
        """
        Create or reuse an MLflow experiment.
        If the experiment exists in 'deleted' stage, restore it.
        All comments MUST be in English.
        """
        from mlflow.tracking import MlflowClient

        experiment_name = f"{algo}_{symbol}_{start}_{end}"
        client = MlflowClient()

        # Try to locate experiment by name
        exp = client.get_experiment_by_name(experiment_name)

        # If exists but deleted → restore
        if exp is not None and exp.lifecycle_stage == "deleted":
            client.restore_experiment(exp.experiment_id)

        # Now safe to set experiment (creates if missing)
        mlflow.set_experiment(experiment_name)
        return experiment_name


    # ---------------------------------------------------------
    # Start a run inside the experiment
    # ---------------------------------------------------------
    def start_run(self, run_name: str):
        """
        Starts an MLflow run.
        """
        return mlflow.start_run(run_name=run_name)

    # ---------------------------------------------------------
    # Register all model artifacts in MLflow
    # ---------------------------------------------------------
    def log_artifacts(
        self,
        model_output: str,
        booster_path: str,
        scaler,
        label_encoder
    ):
        """
        Logs all artifacts in the SAME directory as model_output.
        All comments MUST be in English.
        """
        output_dir = Path(model_output).parent
        base_name = Path(model_output).stem

        # Log raw model (.pkl)
        mlflow.log_artifact(model_output)

        # Log booster.json
        if os.path.exists(booster_path):
            mlflow.log_artifact(booster_path)

        # Save + log scaler into models/
        if scaler is not None:
            scaler_path = output_dir / f"{base_name}_scaler.pkl"
            import joblib
            joblib.dump(scaler, scaler_path)
            mlflow.log_artifact(str(scaler_path))

        # Save + log label encoder into models/
        if label_encoder is not None:
            label_path = output_dir / f"{base_name}_label_encoder.pkl"
            import joblib
            joblib.dump(label_encoder, label_path)
            mlflow.log_artifact(str(label_path))


    # ---------------------------------------------------------
    # Full registration process (called from training)
    # ---------------------------------------------------------
    def register_model(
        self,
        algo: str,
        model_output: str,
        booster_path: str,
        scaler,
        label_encoder,
        register_model: bool,
        model=None
    ):
        """
        Top-level call:
            - Parse filename
            - Create experiment
            - Start run
            - Log all artifacts

        If register_model=False → do nothing.
        """
        if not register_model:
            return

        parsed = self.parse_model_filename(model_output)
        experiment_name = self.start_experiment(algo,parsed["symbol"], parsed["start"], parsed["end"])
        run_name = parsed["model_version"]

        with self.start_run(run_name):
            # Do NOT auto-register the model
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model"
            )

            # Log artifacts
            self.log_artifacts(
                model_output=model_output,
                booster_path=booster_path,
                scaler=scaler,
                label_encoder=label_encoder
            )
