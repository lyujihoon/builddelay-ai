"""교육용 건축공사 일정 지연 위험도 가상 데이터를 생성한다."""

from pathlib import Path

import numpy as np
import pandas as pd


RANDOM_STATE = 42
N_SAMPLES = 600
FEATURE_NAMES = [
    "전체 공사 일정 경과율",
    "실제 공정률",
    "인력 충족률",
    "자재 납품 지연일",
    "악천후 작업 중단일",
    "설계변경 횟수",
    "품질검사 지적 건수",
    "협력업체 지연 발생 여부",
    "공정 편차",
]
TARGET_NAME = "지연 위험도"


def main() -> None:
    """고정된 난수 시드로 600개의 합성 현장 기록을 저장한다."""
    rng = np.random.default_rng(RANDOM_STATE)
    schedule = rng.integers(15, 91, N_SAMPLES)
    progress_gap = rng.normal(-3, 10, N_SAMPLES)
    actual = np.clip(schedule + progress_gap, 0, 100).round(1)
    manpower = rng.integers(60, 101, N_SAMPLES)
    material_delay = np.clip(rng.poisson(5, N_SAMPLES), 0, 20)
    weather_stop = np.clip(rng.poisson(3, N_SAMPLES), 0, 15)
    design_changes = np.clip(rng.poisson(2, N_SAMPLES), 0, 10)
    quality_issues = np.clip(rng.poisson(2, N_SAMPLES), 0, 10)
    subcontractor_delay = rng.binomial(1, 0.3, N_SAMPLES)
    deviation = (actual - schedule).round(1)

    # 위험요인이 클수록 점수가 커지도록 하되, 작은 노이즈로 현실적 변동을 둔다.
    risk_score = (
        np.maximum(0, -deviation) * 1.7
        + np.maximum(0, 90 - manpower) * 0.7
        + material_delay * 1.0
        + weather_stop * 0.8
        + design_changes * 1.8
        + quality_issues * 2.0
        + subcontractor_delay * 6.0
        + rng.normal(0, 3, N_SAMPLES)
    )
    low_cutoff, high_cutoff = np.quantile(risk_score, [1 / 3, 2 / 3])
    labels = np.where(risk_score <= low_cutoff, "Low", np.where(risk_score <= high_cutoff, "Moderate", "High"))

    dataset = pd.DataFrame(
        {
            "전체 공사 일정 경과율": schedule,
            "실제 공정률": actual,
            "인력 충족률": manpower,
            "자재 납품 지연일": material_delay,
            "악천후 작업 중단일": weather_stop,
            "설계변경 횟수": design_changes,
            "품질검사 지적 건수": quality_issues,
            "협력업체 지연 발생 여부": subcontractor_delay,
            "공정 편차": deviation,
            TARGET_NAME: labels,
        }
    )

    output_path = Path(__file__).resolve().parent / "data" / "construction_delay_dataset.csv"
    output_path.parent.mkdir(exist_ok=True)
    dataset.to_csv(output_path, index=False, encoding="utf-8-sig")

    counts = dataset[TARGET_NAME].value_counts().reindex(["Low", "Moderate", "High"])
    print(f"데이터 행 수: {len(dataset)}")
    print(f"컬럼 목록: {', '.join(dataset.columns)}")
    print(f"결측치 여부: {'없음' if not dataset.isna().any().any() else '있음'}")
    print("클래스별 개수:")
    print(counts.to_string())
    print("클래스별 비율:")
    print((counts / len(dataset) * 100).round(1).astype(str).add('%').to_string())
    print(f"CSV 저장 위치: {output_path}")


if __name__ == "__main__":
    main()
