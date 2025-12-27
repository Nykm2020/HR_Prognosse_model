from pathlib import Path
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

def build_pipeline() -> Pipeline:
    numeric_features = [
        "satisfaction_level",
        "last_evaluation",
        "number_project",
        "average_montly_hours",
        "time_spend_company",
        "Work_accident",
        "promotion_last_5years",
    ]
    categorical_features = ["Department"]
    ordinal_features = ["salary"]  # low < medium < high

    preprocess = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_features),
            ("dept", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("sal", OrdinalEncoder(categories=[["low", "medium", "high"]]), ordinal_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    clf = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        solver="lbfgs",
    )

    return Pipeline(steps=[("preprocess", preprocess), ("clf", clf)])

def main() -> None:
    base_dir = Path(__file__).resolve().parent
    csv_path = base_dir / "data" / "HR_comma_sep.csv"
    model_path = base_dir / "model" / "logreg_pipeline.joblib"

    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV nicht gefunden: {csv_path}\n"
            "Lege die Datei HR_comma_sep.csv in den Ordner ./data/"
        )

    df = pd.read_csv(csv_path).drop_duplicates()
    if "left" not in df.columns:
        raise ValueError("Spalte 'left' fehlt in der CSV. Bitte Original-Spaltennamen verwenden.")

    y = df["left"].astype(int)
    X = df.drop(columns=["left"])

    pipe = build_pipeline()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )

    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]
    print("\n📋 Klassifikationsbericht (Test):")
    print(classification_report(y_test, y_pred))
    print(f"📈 ROC-AUC (Test): {roc_auc_score(y_test, y_proba):.4f}")

    try:
        cv_scores = cross_val_score(pipe, X, y, cv=5, scoring="f1")
        print(f"🧪 F1 (CV 5-fold, mean): {cv_scores.mean():.4f}")
    except Exception as e:
        print(f"⚠️ Cross-Validation übersprungen: {e}")

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, model_path)
    print(f"\n✅ Modell gespeichert: {model_path}")

if __name__ == "__main__":
    main()
