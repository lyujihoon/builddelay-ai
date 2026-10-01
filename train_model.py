"""가상 건설현장 데이터로 Random Forest 분류 모델을 학습한다."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split


FEATURE_NAMES = [
    "전체 공사 일정 경과율", "실제 공정률", "인력 충족률", "자재 납품 지연일",
    "악천후 작업 중단일", "설계변경 횟수", "품질검사 지적 건수", "협력업체 지연 발생 여부", "공정 편차",
]
TARGET_NAME = "지연 위험도"
CLASS_NAMES = ["Low", "Moderate", "High"]


def main() -> None:
    """CSV를 80/20으로 나누어 학습하고 모델과 실제 측정치를 저장한다."""
    root = Path(__file__).resolve().parent
    dataset = pd.read_csv(root / "data" / "construction_delay_dataset.csv", encoding="utf-8-sig")
    X, y = dataset[FEATURE_NAMES], dataset[TARGET_NAME]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, predictions, labels=CLASS_NAMES, average="macro", zero_division=0
    )
    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "classification_report": classification_report(y_test, predictions, labels=CLASS_NAMES, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=CLASS_NAMES).tolist(),
    }
    artifact = {
        "model": model,
        "feature_names": FEATURE_NAMES,
        "class_names": list(model.classes_),
        "metrics": metrics,
        "feature_importances": dict(zip(FEATURE_NAMES, model.feature_importances_)),
    }
    output_path = root / "model" / "delay_classifier.joblib"
    output_path.parent.mkdir(exist_ok=True)
    joblib.dump(artifact, output_path)

    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print(f"Macro Precision: {metrics['macro_precision']:.3f}")
    print(f"Macro Recall: {metrics['macro_recall']:.3f}")
    print(f"Macro F1-score: {metrics['macro_f1']:.3f}")
    print("Classification Report:")
    print(metrics["classification_report"])
    print("Confusion Matrix (Low, Moderate, High):")
    print(pd.DataFrame(metrics["confusion_matrix"], index=CLASS_NAMES, columns=CLASS_NAMES).to_string())
    print(f"모델 저장 위치: {output_path}")


if __name__ == "__main__":
    main()
