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

    If MLFLOW_TRACKING_URI is missing/None, or the tracking server is
    unreachable, MLflow is silently disabled instead of raising.
    """

    def __init__(self):
        # Load MLflow URI from commands_mgr.ini
        self.mlflow_enabled = False
        mlflow_uri = None

        try:
            loader = MLSettingsLoader()
            config_settings = loader.load_settings("./configs/commands_mgr.ini")
            mlflow_uri = config_settings.get("MLFLOW_TRACKING_URI")
        except Exception as e:
            print(f"[MLflow] Could not read settings ({e}). MLflow disabled.")
            return

        if not mlflow_uri or str(mlflow_uri).strip().lower() in ("none", "null", ""):
            print("[MLflow] MLFLOW_TRACKING_URI not set. MLflow disabled.")
            return

        # Apply tracking URI
        mlflow.set_tracking_uri(mlflow_uri)
        self.mlflow_enabled = True

    # ---------------------------------------------------------
    # Parse a model filename like:
    #   xgb_model_XLK_2010_2018_M7.pkl
    # ---------------------------------------------------------
    def parse_model_filename(self, model_output: str) -> dict:
        """
        Example: xgb_model_XLK_2010_2018_M7.pkl
        """
        base = Path(model_output).stem
        parts = base.split("_")

        if len(parts) < 6:
            raise ValueError(f"Invalid model filename: {model_output}")

        return {
            "symbol": parts[2],
            "start": parts[3],
            "end": parts[4],
            "model_version": parts[5],
            "base_name": base
        }

    # ---------------------------------------------------------
    # Create or reuse an MLflow experiment
    # ---------------------------------------------------------
    def start_experiment(self, algo: str, symbol: str, start: str, end: str):
        """
        Create or reuse an MLflow experiment.
        If the experiment exists in 'deleted' stage, restore it.
        """
        if not self.mlflow_enabled:
            return None

        from mlflow.tracking import MlflowClient

        experiment_name = f"{algo}_{symbol}_{start}_{end}"
        client = MlflowClient()

        exp = client.get_experiment_by_name(experiment_name)

        if exp is not None and exp.lifecycle_stage == "deleted":
            client.restore_experiment(exp.experiment_id)

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
        """
        if not self.mlflow_enabled:
            return

        output_dir = Path(model_output).parent
        base_name = Path(model_output).stem

        # Log raw model (.pkl)
        mlflow.log_artifact(model_output)

        # Log booster.json
        if os.path.exists(booster_path):
            mlflow.log_artifact(booster_path)

        # Save + log scaler into models/
        if scaler is not None:
            import joblib
            scaler_path = output_dir / f"{base_name}_scaler.pkl"
            joblib.dump(scaler, scaler_path)
            mlflow.log_artifact(str(scaler_path))

        # Save + log label encoder into models/
        if label_encoder is not None:
            import joblib
            label_path = output_dir / f"{base_name}_label_encoder.pkl"
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
        Top-level call. Never breaks training: any MLflow failure
        (no URI, server down, timeout) is logged and swallowed.
        """
        if not register_model or not self.mlflow_enabled:
            return

        try:
            parsed = self.parse_model_filename(model_output)
            self.start_experiment(
                algo,
                parsed["symbol"],
                parsed["start"],
                parsed["end"]
            )
            run_name = parsed["model_version"]

            with self.start_run(run_name):
                mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path="model"
                )

                self.log_artifacts(
                    model_output=model_output,
                    booster_path=booster_path,
                    scaler=scaler,
                    label_encoder=label_encoder
                )

        except Exception as e:
            # MLflow must never break the training pipeline
            print(f"[MLflow] Registration skipped due to error: {e}")
            try:
                if mlflow.active_run() is not None:
                    mlflow.end_run()
            except Exception:
                pass