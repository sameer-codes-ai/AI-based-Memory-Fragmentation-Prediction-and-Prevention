"""
PREFRAG: Generalization, Ablation, and Explainability Suites
"""

from typing import List, Dict, Any, Tuple
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from ..traces.trace_schema import MemoryEvent
from ..traces.workload_generator import WorkloadGenerator, WorkloadFamily
from ..features.feature_extractor import FeatureExtractor, TemporalFeatureVector
from ..models.temporal_nn import TemporalForecasterLSTM
from ..models.baselines import RandomForestBaseline, GradientBoostingBaseline


class GeneralizationSuite:
    """
    Evaluates cross-workload model generalization (Train on Workload A, Test on B, C, D, E).
    """

    def __init__(self, extractor: FeatureExtractor):
        self.extractor = extractor
        self.generator = WorkloadGenerator(seed=42)

    def run_cross_workload_matrix(self, num_events: int = 2500) -> Dict[str, Dict[str, float]]:
        """
        Trains model on WEB_SERVER and evaluates on all workload families.
        """
        # Generate datasets for all families
        families = [
            WorkloadFamily.WEB_SERVER,
            WorkloadFamily.DATABASE,
            WorkloadFamily.SCIENTIFIC,
            WorkloadFamily.MICROSERVICE,
            WorkloadFamily.ADVERSARIAL_CHURN
        ]

        from .runner import ExperimentRunner
        runner = ExperimentRunner(window_size=self.extractor.window_size)
        datasets = {}

        for fam in families:
            trace = self.generator.generate_trace(fam, num_events=num_events)
            X, Y_reg, Y_cls, _ = runner.prepare_dataset_for_workload(trace)
            datasets[fam.value] = (X, Y_reg)

        # Train primary model on WEB_SERVER (80% split)
        X_web, Y_web = datasets[WorkloadFamily.WEB_SERVER.value]
        split_idx = int(0.8 * len(X_web))
        X_train, Y_train = X_web[:split_idx], Y_web[:split_idx]
        X_val, Y_val = X_web[split_idx:], Y_web[split_idx:]

        model = TemporalForecasterLSTM(input_dim=X_train.shape[1], hidden_dim=48, output_dim=4)
        model.model_name = "Generalization LSTM"
        model.fit(X_train, Y_train)

        results = {}
        for fam_name, (X_test, Y_test) in datasets.items():
            if fam_name == WorkloadFamily.WEB_SERVER.value:
                X_eval, Y_eval = X_val, Y_val
            else:
                X_eval, Y_eval = X_test, Y_test

            preds = model.predict(X_eval)
            mae_t10 = float(mean_absolute_error(Y_eval[:, 2], preds[:, 2]))
            rmse_t10 = float(np.sqrt(mean_squared_error(Y_eval[:, 2], preds[:, 2])))
            r2_t10 = float(r2_score(Y_eval[:, 2], preds[:, 2]))

            results[fam_name] = {
                "mae_t10": round(mae_t10, 4),
                "rmse_t10": round(rmse_t10, 4),
                "r2_t10": round(r2_t10, 4),
                "in_domain": (fam_name == WorkloadFamily.WEB_SERVER.value)
            }

        return results


class FeatureAblationSuite:
    """
    Evaluates impact of systematically dropping each of the 6 feature groups.
    """

    def __init__(self):
        self.feature_groups = TemporalFeatureVector.feature_groups()

    def run_ablation(
        self,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        X_test: np.ndarray,
        Y_test: np.ndarray
    ) -> Dict[str, Dict[str, float]]:
        results = {}

        # 1. Full Model (Baseline)
        model_full = TemporalForecasterLSTM(input_dim=X_train.shape[1], hidden_dim=48, output_dim=4)
        model_full.model_name = "Full Ablation LSTM"
        model_full.fit(X_train, Y_train)
        pred_full = model_full.predict(X_test)
        base_mae = float(mean_absolute_error(Y_test[:, 2], pred_full[:, 2]))
        base_r2 = float(r2_score(Y_test[:, 2], pred_full[:, 2]))

        results["Full Model (All Features)"] = {
            "mae_t10": round(base_mae, 4),
            "r2_t10": round(base_r2, 4),
            "delta_mae_pct": 0.0
        }

        # 2. Iterate and zero-out each feature group
        for group_name, group_indices in self.feature_groups.items():
            X_tr_abl = X_train.copy()
            X_te_abl = X_test.copy()
            X_tr_abl[:, group_indices] = 0.0
            X_te_abl[:, group_indices] = 0.0

            m = TemporalForecasterLSTM(input_dim=X_train.shape[1], hidden_dim=48, output_dim=4)
            m.model_name = f"Ablate [{group_name[:8]}]"
            m.fit(X_tr_abl, Y_train)
            preds = m.predict(X_te_abl)

            mae = float(mean_absolute_error(Y_test[:, 2], preds[:, 2]))
            r2 = float(r2_score(Y_test[:, 2], preds[:, 2]))
            delta_pct = ((mae - base_mae) / (base_mae + 1e-9)) * 100.0

            results[f"Ablate: {group_name.capitalize()} Features"] = {
                "mae_t10": round(mae, 4),
                "r2_t10": round(r2, 4),
                "delta_mae_pct": round(delta_pct, 2)
            }

        return results


class ExplainabilitySuite:
    """
    Analyzes feature attribution and ranking using permutation importance and Gini impurity metrics.
    """

    @staticmethod
    def get_feature_importances(X: np.ndarray, Y: np.ndarray) -> List[Tuple[str, float]]:
        names = TemporalFeatureVector.feature_names()
        rf = RandomForestBaseline(n_estimators=100, max_depth=10)
        rf.fit(X, Y)
        imps = rf.get_feature_importances()
        ranked = sorted(zip(names, imps), key=lambda x: x[1], reverse=True)
        return [(name, round(float(val), 4)) for name, val in ranked]
