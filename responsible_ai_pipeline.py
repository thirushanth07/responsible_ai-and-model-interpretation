import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import shap
from fairlearn.metrics import (
    MetricFrame,
    false_positive_rate,
    selection_rate,
    true_positive_rate,
)
from lime.lime_tabular import LimeTabularExplainer
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42


def load_dataset():
    data = load_breast_cancer(as_frame=True)
    X = data.data.copy()
    y = data.target.copy()

    # The dataset has no protected attribute, so this uses a feature-based split
    # only as a proxy group for demonstration of fairness metric workflows.
    sensitive_group = (X["mean radius"] > X["mean radius"].median()).astype(int)
    sensitive_group.name = "sensitive_group"

    return X, y, sensitive_group


def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    return model


def get_feature_importance(model, feature_names):
    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    return importance_df


def save_feature_importance_plot(importance_df, output_dir, top_n=10):
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=importance_df.head(top_n),
        x="importance",
        y="feature",
        hue="feature",
        legend=False,
        palette="Blues_r",
    )
    plt.title(f"Top {top_n} Feature Importances")
    plt.tight_layout()
    plt.savefig(output_dir / "feature_importance.png", dpi=200)
    plt.close()


def save_shap_summary(model, X_test, output_dir):
    sample = X_test.sample(min(120, len(X_test)), random_state=RANDOM_STATE)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample)

    if isinstance(shap_values, list):
        shap_for_plot = shap_values[1]
    elif getattr(shap_values, "ndim", 0) == 3:
        shap_for_plot = shap_values[:, :, 1]
    else:
        shap_for_plot = shap_values

    shap.summary_plot(shap_for_plot, sample, show=False)
    plt.tight_layout()
    plt.savefig(output_dir / "shap_summary.png", dpi=200)
    plt.close()


def save_lime_explanation(model, X_train, X_test, output_dir):
    explainer = LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=X_train.columns.tolist(),
        class_names=["malignant", "benign"],
        mode="classification",
        random_state=RANDOM_STATE,
    )

    def predict_with_feature_names(values):
        frame = pd.DataFrame(values, columns=X_train.columns)
        return model.predict_proba(frame)

    explanation = explainer.explain_instance(
        X_test.iloc[0].values,
        predict_with_feature_names,
        num_features=10,
    )

    fig = explanation.as_pyplot_figure()
    fig.tight_layout()
    fig.savefig(output_dir / "lime_explanation.png", dpi=200)
    plt.close(fig)


def evaluate_fairness(y_true, y_pred, sensitive_group):
    metrics = {
        "selection_rate": selection_rate,
        "true_positive_rate": true_positive_rate,
        "false_positive_rate": false_positive_rate,
    }

    frame = MetricFrame(
        metrics=metrics,
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive_group,
    )

    fairness_by_group = frame.by_group.reset_index().rename(
        columns={"sensitive_group": "group"}
    )
    selection_rate_range = (
        fairness_by_group["selection_rate"].max()
        - fairness_by_group["selection_rate"].min()
    )

    return fairness_by_group, selection_rate_range


def save_fairness_plot(fairness_df, output_dir):
    plt.figure(figsize=(6, 4))
    sns.barplot(
        data=fairness_df,
        x="group",
        y="selection_rate",
        hue="group",
        legend=False,
        palette="viridis",
    )
    plt.title("Selection Rate by Sensitive Group")
    plt.xlabel("Sensitive Group")
    plt.ylabel("Selection Rate")
    plt.tight_layout()
    plt.savefig(output_dir / "fairness_selection_rate.png", dpi=200)
    plt.close()


def mitigation_recommendations(selection_rate_gap):
    if selection_rate_gap <= 0.05:
        return [
            "Bias appears low; continue monitoring fairness over time.",
            "Track fairness metrics in production to detect data drift.",
        ]

    return [
        "Review feature engineering choices for proxy variables.",
        "Try reweighting, threshold optimization, or fairness-constrained training.",
        "Conduct subgroup error analysis and retrain with representative samples.",
    ]


def main(output_dir="outputs"):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    X, y, sensitive_group = load_dataset()
    X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
        X,
        y,
        sensitive_group,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    model = train_model(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    importance_df = get_feature_importance(model, X.columns)
    save_feature_importance_plot(importance_df, output_path)
    save_shap_summary(model, X_test, output_path)
    save_lime_explanation(model, X_train, X_test, output_path)

    fairness_df, selection_rate_gap = evaluate_fairness(y_test, y_pred, s_test)
    save_fairness_plot(fairness_df, output_path)

    summary = {
        "accuracy": accuracy,
        "classification_report": report,
        "fairness_by_group": fairness_df.to_dict(orient="records"),
        "selection_rate_gap": selection_rate_gap,
        "recommendations": mitigation_recommendations(selection_rate_gap),
        "top_features": importance_df.head(10).to_dict(orient="records"),
    }

    with open(output_path / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Model accuracy: {accuracy:.4f}")
    print(f"Selection rate gap: {selection_rate_gap:.4f}")
    print(f"Saved outputs to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
