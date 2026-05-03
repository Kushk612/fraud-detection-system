from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


class ModelRegistry:
    """
    Central place to manage models
    """

    def get_models(self):
        models = {
            "logistic": LogisticRegression(
                max_iter=1000,
                class_weight="balanced"
            ),

            "random_forest": RandomForestClassifier(
                n_estimators=100,
                class_weight="balanced",
                random_state=42
            ),

            "xgboost": XGBClassifier(
                use_label_encoder=False,
                eval_metric="logloss",
                scale_pos_weight=100  # important for imbalance
            )
        }

        return models