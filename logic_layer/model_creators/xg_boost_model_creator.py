from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import TimeSeriesSplit
import pandas as pd
import numpy as np
import joblib

from common.util.pandas_dataframes.dataframe_filler import DataframeFiller
from common.util.graphs.graph_builder import GraphBuilder
from common.util.logging.light_logger import LightLogger
from logic_layer.model_creators.base_model_creator import BaseModelCreator

_OUTPUT_DATE_FORMAT = '%m/%d/%Y %H:%M:%S'

class XGBoostModelCreator(BaseModelCreator):

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

    def train_xgboost_daily(self, training_series_df, model_output, symbol, classif_key,
                            series_csv, n_estimators=100, max_depth=3, learning_rate=0.1,
                            subsample=1.0, colsample_bytree=1.0,
                            make_stationary=False, class_weight=None):
        try:
            training_series_df = DataframeFiller.fill_missing_values(training_series_df)

            if make_stationary:
                training_series_df = self.__make_stationary__(training_series_df)

            self.__preformat_training_set__(training_series_df)

            X_train, X_test, y_train, y_test, label_encoder, scaler = self.__get_training_sets__(
                training_series_df,
                symbol_col="trading_symbol",
                date_col="date",
                classif_key=classif_key,
                variables_csv=series_csv,
                test_size=0.2,
                return_encoder_and_scaler=True
            )

            print("Class distribution in y_train:", np.bincount(y_train))

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
                    verbosity=1
                )
                model.fit(X_tr, y_tr)

                importances = model.feature_importances_
                for name, imp in zip(training_series_df.columns, importances):
                    if name not in ['symbol', 'date', classif_key]:
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
                random_state=42
            )
            final_model.fit(X_train, y_train)

            print("Labels in y_train (raw):", y_train)
            print("Unique encoded labels:", np.unique(y_train))

            model_bundle = {
                "model": final_model,
                "label_encoder": label_encoder,
                "scaler": scaler
            }
            joblib.dump(model_bundle, model_output)
            print(f"XGBoost model bundle saved to {model_output}")

            return label_encoder

        except Exception as e:
            raise Exception(f"Fatal error during XGBoost training: {e}")
