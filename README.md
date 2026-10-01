# BuildDelay AI

건축공사 현장 입력값을 바탕으로 일정 지연 위험도를 `Low(정상)`, `Moderate(주의)`, `High(지연 위험)`으로 분류하는 가천대학교 건축공학 AI 과제용 Streamlit 웹앱입니다.

## 과제 목적과 주요 기능

교육용 가상 건설현장 데이터를 생성하고 Random Forest 분류 모델을 실제로 학습한 뒤, 사용자가 8개 현장 값을 입력해 위험도·모델 Confidence·클래스별 예측 확률·규칙 기반 위험요인을 확인합니다. 정상/주의/고위험 샘플 3종과 Plotly 확률 막대그래프도 제공합니다.

## 기술과 데이터

- Python, Streamlit, pandas, NumPy, scikit-learn, joblib, Plotly
- 모델: `RandomForestClassifier(n_estimators=200, random_state=42)`
- 데이터: 재현 가능한 난수 시드(42)로 생성한 600개 **가상** 현장 기록
- 분리: 학습 80%, 테스트 20%, `random_state=42`, `stratify=y`

입력 변수는 전체 공사 일정 경과율, 실제 공정률, 인력 충족률, 자재 납품 지연일, 악천후 작업 중단일, 설계변경 횟수, 품질검사 지적 건수, 협력업체 지연 여부이며, `공정 편차 = 실제 공정률 - 전체 공사 일정 경과율`을 파생변수로 함께 사용합니다.

라벨은 실제 관측값이 아니라 공정 지연, 인력 부족, 자재·악천후·설계·품질·협력업체 요인이 커질수록 높아지는 잠재 위험점수와 작은 노이즈를 바탕으로 만들었습니다.

## 실제 측정 모델 성능

생성 데이터의 독립 테스트셋(120개)에서 측정한 결과입니다.

| 지표 | 값 |
| --- | ---: |
| Accuracy | 0.792 |
| Macro Precision | 0.810 |
| Macro Recall | 0.792 |
| Macro F1-score | 0.796 |

## 설치 및 실행

```powershell
pip install -r requirements.txt
python generate_data.py
python train_model.py
streamlit run app.py
```

Streamlit 배포 링크: 배포 후 추가 예정

## 프로젝트 구조

```text
builddelay-ai/
├── PRD.md
├── generate_data.py                 # 가상 데이터 생성
├── train_model.py                   # 모델 학습 및 평가
├── app.py                           # Streamlit 웹앱
├── data/construction_delay_dataset.csv
└── model/delay_classifier.joblib
```

## 샘플 테스트

앱의 샘플 선택에서 정상, 주의, 고위험 현장을 고른 후 분석을 실행할 수 있습니다. 결과는 학습된 모델의 실제 출력이며 샘플 결과를 임의로 조작하지 않습니다.

## 한계와 교육용 고지

학습 데이터와 라벨은 실제 건설현장 관측값이 아닌 가상 데이터·설계 규칙 기반입니다. 따라서 모델 성능은 이 가상 규칙을 재현하는 성능에 가깝고, Confidence는 실제 지연 발생 확률이 아닙니다.

> 본 결과는 건축공학 AI 수업을 위해 생성한 가상 데이터 기반의 교육용 분석 결과이며, 실제 건설현장의 공정관리 또는 의사결정을 대체하지 않습니다.
