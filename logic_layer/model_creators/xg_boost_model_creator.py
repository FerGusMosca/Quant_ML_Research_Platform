from pathlib import Path
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import TimeSeriesSplit
import pandas as pd
import numpy as np
import sklearn
from packaging import version
from common.util.pandas_dataframes.dataframe_filler import DataframeFiller
from common.util.graphs.graph_builder import GraphBuilder
from common.util.logging.light_logger import LightLogger
from logic_layer.model_creators.base_model_creator import BaseModelCreator

_OUTPUT_DATE_FORMAT = '%m/%d/%Y %H:%M:%S'

class XGBoostModelCreator(BaseModelCreator):

    def __build_features_for_inference__(self,
                                         raw_df: pd.DataFrame,
                                         feature_cols: list,
                                         symbol: str,
                                         make_stationary: bool = True,
                                         state: dict | None = None) -> tuple[pd.DataFrame, dict | None]:
        """
        Prepare features for inference so they exactly match the training feature set.

        Steps:
          - Ensure 'date' is datetime and sorted.
          - Optionally make series stationary using __make_stationary_with_memory__.
          - Harmonize OHLC names (open/high/low/close vs open_SYMBOL/...).
          - Keep only ['date'] + feature_cols, in training order.
          - Coerce to numeric, handle Inf/NaN, cast to float32.

        Returns:
          (features_df_ready, updated_state)
        """

        df = raw_df.copy()

        # 0) date handling
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

        # 1) Stationarity (your method, with state)
        updated_state = state
        if make_stationary:
            # Use your differencing method; it mutates and returns (df, state)
            df, updated_state = self.__make_stationary_with_memory__(df, state=state)

        # 2) Harmonize OHLC names (if training expected suffixed or unsuffixed)
        base_ohlc = ["open", "high", "low", "close"]
        # base -> base_SYMBOL
        for base in base_ohlc:
            expected = f"{base}_{symbol}"
            if expected in feature_cols and expected not in df.columns and base in df.columns:
                df.loc[:, expected] = df[base]
        # base_SYMBOL -> base
        for base in base_ohlc:
            expected_no = base
            expected_suf = f"{base}_{symbol}"
            if expected_no in feature_cols and expected_no not in df.columns and expected_suf in df.columns:
                df.loc[:, expected_no] = df[expected_suf]

        # 3) Ensure all training features exist (create missing as NaN)
        for col in feature_cols:
            if col not in df.columns:
                df.loc[:, col] = np.nan

        # 4) Keep only date + features, in training order
        out = pd.concat([df[["date"]], df[feature_cols]], axis=1)

        # 5) Coerce to numeric (only features), then sanitize
        out[feature_cols] = out[feature_cols].apply(pd.to_numeric, errors="coerce")
        out[feature_cols] = out[feature_cols].replace([np.inf, -np.inf], np.nan)

        # Drop rows that still have NaN in any feature
        mask_ok = np.isfinite(out[feature_cols].to_numpy()).all(axis=1)
        out = out.loc[mask_ok].reset_index(drop=True)

        # 6) Final dtype for XGBoost
        out[feature_cols] = out[feature_cols].astype(np.float32)

        return out, updated_state

    def __eval_class_distributions__(self, model, label_encoder, probs):
        long_index = None
        if label_encoder is not None:
            class_labels = model.classes_
            long_label = label_encoder.transform(["LONG"])[0]
            if long_label in class_labels:
                long_index = list(class_labels).index(long_label)
                LightLogger.do_log(f"[XGB TEST] Class index mapping → LONG={long_index}")
            else:
                LightLogger.do_log("[XGB TEST] WARNING: Class 'LONG' not found in model.classes_")
        else:
            long_index = 0  # fallback default

        if long_index is not None and probs.shape[1] > long_index:
            prob_long = probs[:, long_index]
        else:
            prob_long = np.zeros(probs.shape[0])
            LightLogger.do_log("[XGB TEST] WARNING: Missing LONG prob. Defaulting to 0% LONG.")


    def train_xgboost_daily(
            self,
            training_series_df,
            model_output,
            symbol,
            classif_key,
            series_csv,
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            subsample=1.0,
            colsample_bytree=1.0,
            make_stationary=False,
            class_weight=None,  # kept for API compatibility (not used here)
    ):
        """
        Train an XGBoost classifier for daily scalping and persist a version-stable bundle.
        Post-training probability calibration is applied (Platt/sigmoid).
        The loader will be updated later to consume the calibrated model.
        """
        try:
            # 1) Fill gaps and (optionally) make series stationary
            training_series_df = DataframeFiller.fill_missing_values(training_series_df)
            if make_stationary:
                training_series_df = self.__make_stationary__(training_series_df)

            # 2) Any DF-level preformatting you already do
            self.__preformat_training_set__(training_series_df)

            # 3) Build train/test matrices with the EXACT feature list (order matters)
            X_train, X_test, y_train, y_test, label_encoder, scaler = self.__get_training_sets__(
                training_series_df,
                symbol_col="trading_symbol",
                date_col="date",
                classif_key=classif_key,
                variables_csv=series_csv,
                test_size=0.2,
                return_encoder_and_scaler=True,
            )

            # Real feature list used by the matrix builder (order preserved)
            expected_features = series_csv.split(',')

            print("Class distribution in y_train:", np.bincount(y_train))

            # 4) CV folds (only for logging metrics/importance)
            tscv = TimeSeriesSplit(n_splits=3)
            val_accuracies, val_f1_scores = [], []

            for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train)):
                print(f"\nTraining fold {fold + 1}/{tscv.n_splits}...")

                X_tr, X_val = X_train[train_idx], X_train[val_idx]
                y_tr, y_val = y_train[train_idx], y_train[val_idx]

                model = XGBClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    learning_rate=learning_rate,
                    subsample=subsample,
                    colsample_bytree=colsample_bytree,
                    objective="multi:softprob",
                    num_class=len(np.unique(y_train)),
                    eval_metric="mlogloss",
                    use_label_encoder=False,
                    random_state=42,
                    verbosity=1,
                )
                model.fit(X_tr, y_tr)

                # Log importances against the REAL feature names (not DF column order)
                importances = model.feature_importances_
                for name, imp in zip(expected_features, importances):
                    LightLogger.do_log(f"[XGB TRAIN] Feature importance → {name}: {imp:.5f}")

                y_pred = np.argmax(model.predict_proba(X_val), axis=1)
                acc = accuracy_score(y_val, y_pred)
                f1 = f1_score(y_val, y_pred, average='weighted')

                print(f"Fold {fold + 1} - Accuracy: {acc:.4f} - F1 Score: {f1:.4f}")
                print("Confusion Matrix:\n", confusion_matrix(y_val, y_pred))

                val_accuracies.append(acc)
                val_f1_scores.append(f1)

            print(f"\nAverage accuracy: {np.mean(val_accuracies):.4f}")
            print(f"Average F1 Score: {np.mean(val_f1_scores):.4f}")

            # 5) Train the final model on the full training split
            final_model = XGBClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                objective="multi:softprob",
                num_class=len(np.unique(y_train)),
                eval_metric="mlogloss",
                use_label_encoder=False,
                random_state=42,
            )
            final_model.fit(X_train, y_train)

            print("Unique encoded labels:", np.unique(y_train))

            # 6) Probability calibration (post-training, keeps ranking but fixes scale)
            # Use Platt (sigmoid). Calibrate on the held-out test split.
            calib_method = "sigmoid"

            try:
                # Newer sklearn: CalibratedClassifierCV(estimator=..., cv="prefit")
                if version.parse(sklearn.__version__) >= version.parse("1.0"):
                    calibrated_clf = CalibratedClassifierCV(estimator=final_model, method=calib_method, cv="prefit")
                else:
                    # Older sklearn used base_estimator
                    calibrated_clf = CalibratedClassifierCV(base_estimator=final_model, method=calib_method,
                                                            cv="prefit")
            except TypeError:
                # Fallback if version parsing fails but signature is new
                calibrated_clf = CalibratedClassifierCV(estimator=final_model, method=calib_method, cv="prefit")

            calibrated_clf.fit(X_test, y_test)
            print(f"[XGB TRAIN] Probabilities calibrated via {calib_method} on hold-out set (n={X_test.shape[0]}).")

            # 7) Sanity checks before saving (prevent future mismatches)
            booster_n = final_model.get_booster().num_features()
            if booster_n != len(expected_features) or booster_n != X_train.shape[1]:
                raise RuntimeError(
                    f"[SAVE MISMATCH] booster={booster_n}, expected_features={len(expected_features)}, "
                    f"X_train_cols={X_train.shape[1]}"
                )

            # 8) Save artifacts using the EXACT feature list used for training (and the calibrator)
            self.__save_xgb_model_bundle__(
                final_model,
                expected_features,  # ✅ exact features (order preserved)
                label_encoder,
                scaler,
                model_output,
                calibrated_model=calibrated_clf,  # <-- NEW: persist calibrator
                calibration_method="sigmoid",  # <-- optional meta
            )
            print(f"[XGBOOST TRAIN] Saved at base: {Path(model_output).with_suffix('')}")
            print(f"XGBoost model bundle saved to {model_output}")

            return label_encoder

        except Exception as e:
            raise Exception(f"Fatal error during XGBoost training: {e}")

    def test_XGBoost_scalping(
            self,
            symbol_df,
            symbol,
            features_df,  # DF with OHLC and raw inputs for feature builder
            model_filename,
            label_encoder,
            bias="LONG",
            draw_statistics=False,
            make_stationary=True,
            classif_threshold=0.6,
            debug=False,
    ):
        """
        Predict scalping signals using a trained XGBoost bundle.
        Prefers calibrated probabilities when available; otherwise falls back to raw Booster.

        Pipeline:
          1) Load bundle (booster + feature_cols + encoder + optional calibrated_model).
          2) Build features EXACTLY as in training (order, names, dtypes).
          3) Predict probabilities (calibrated if possible).
          4) Convert probs -> labels robustly (binary vs. multiclass).
          5) Return result_df + features_ready.
        """

        # --- 1) Load bundle ---
        _loaded = self.__load_xgb_model_bundle__(model_filename)
        # Support both new (6-tuple) and old (5-tuple) loaders
        if len(_loaded) == 6:
            booster, label_encoder_saved, scaler, feature_cols, calibrated_model, _xgb_params = _loaded
        else:
            booster, label_encoder_saved, scaler, feature_cols, _xgb_params = _loaded
            calibrated_model = None

        if debug:
            print("[DEBUG] booster.num_features() =", booster.num_features())
            print("[DEBUG] len(feature_cols)      =", len(feature_cols))
            print("[DEBUG] calibrated_model?      =", calibrated_model is not None)

        # Prefer the label encoder from the bundle
        label_encoder = label_encoder_saved or label_encoder

        # --- 2) Build features exactly as training expects ---
        features_ready, _ = self.__build_features_for_inference__(
            raw_df=features_df,
            feature_cols=feature_cols,
            symbol=symbol,
            make_stationary=make_stationary,
            state=None,
        )

        # Join on 'date' to carry price for output (keeps strict training order for X)
        merged_df = pd.merge(
            features_ready[["date"] + feature_cols],
            symbol_df[["date", "close"]],
            on="date",
            how="inner",
        )

        # --- 3) Final matrix for inference (strict column order) ---
        X = merged_df[feature_cols].to_numpy(dtype=np.float32)

        # --- 4) Predict class probabilities ---
        if calibrated_model is not None:
            # Use calibrated sklearn wrapper directly on raw features
            y_probs = calibrated_model.predict_proba(X)  # shape: (n, K)
            if debug:
                print("[DEBUG] Using CALIBRATED probabilities.")
        else:
            # Fallback to raw Booster probabilities
            dmat = xgb.DMatrix(X, missing=np.nan, feature_names=feature_cols)
            y_probs = booster.predict(dmat)  # may be (n,) for binary or (n, K) for multiclass
            if y_probs.ndim == 1:
                # Normalize to 2D for binary logistic: column 1 will be the positive class
                y_probs = np.column_stack([1.0 - y_probs, y_probs])
            if debug:
                print("[DEBUG] Using RAW Booster probabilities.")

        # --- 5) Robust class-index mapping via the encoder ---
        classes = np.array(getattr(label_encoder, "classes_", []))
        idx_map = {c: int(np.where(classes == c)[0][0]) for c in classes}  # e.g. {'LONG':1,'SHORT':0,'FLAT':2}
        long_idx = idx_map.get("LONG")
        short_idx = idx_map.get("SHORT")
        flat_idx = idx_map.get("FLAT")
        n_classes = y_probs.shape[1]

        # --- 6) Decision rule ---
        if n_classes >= 3 and flat_idx is not None:
            # Multiclass: pick argmax; optionally route low-confidence to FLAT
            conf = y_probs.max(axis=1)
            y_pred_ids = y_probs.argmax(axis=1)
            if classif_threshold > 0.0:
                y_pred_ids = np.where(conf >= classif_threshold, y_pred_ids, flat_idx)
        else:
            # Binary: threshold on LONG probability (never assume fixed column order)
            if long_idx is None or short_idx is None:
                raise RuntimeError("LabelEncoder must contain LONG and SHORT for binary classification.")
            prob_long = y_probs[:, long_idx]
            y_pred_ids = np.where(prob_long >= classif_threshold, long_idx, short_idx)

        # Map back to string labels
        y_class = label_encoder.inverse_transform(y_pred_ids)

        # --- 7) Build result dataframe ---
        # Align dates to merged_df (the one used for X)
        dates = pd.to_datetime(merged_df["date"], unit="s")
        formatted_dates = dates.dt.strftime(_OUTPUT_DATE_FORMAT)  # ensure this constant exists globally

        # Probability columns: proba_0..proba_{n_classes-1}
        proba_cols = {f"proba_{k}": y_probs[:, k] for k in range(n_classes)}

        result_df = (
            pd.DataFrame({
                "date": merged_df["date"].values,
                "formatted_date": formatted_dates.values,
                "Prediction": y_class,
                "close": merged_df["close"].values,
                "bias": bias,
            })
            .assign(**proba_cols)
        )

        # --- 8) Optional plots ---
        if draw_statistics and n_classes > 1 and long_idx is not None:
            GraphBuilder.plot_long_probability_distributions(y_probs[:, long_idx],
                                                             threshold=classif_threshold)
            importances = GraphBuilder.booster_importances_aligned(booster, feature_cols)
            GraphBuilder.plot_feature_importances_from_values(
                importances, feature_cols, output_path="out/feature_importance_xgb.png"
            )

        if debug:
            print(f"[DEBUG] feature_cols (bundle): {len(feature_cols)}")
            print(f"[DEBUG] X shape: {X.shape}")
            uniq, cnt = np.unique(y_class, return_counts=True)
            print("[DEBUG] preds distrib:", dict(zip(uniq, cnt)))

        return result_df, features_ready








