"""BuildDelay AI Streamlit 앱: 교육용 가상 데이터 기반 위험도 분류기."""

from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st


FEATURE_NAMES = [
    "전체 공사 일정 경과율", "실제 공정률", "인력 충족률", "자재 납품 지연일",
    "악천후 작업 중단일", "설계변경 횟수", "품질검사 지적 건수", "협력업체 지연 발생 여부", "공정 편차",
]
CLASS_LABELS = {"Low": "정상", "Moderate": "주의", "High": "지연 위험"}
COLORS = {"Low": "#2E8B57", "Moderate": "#F39C12", "High": "#D64541"}
SAMPLES = {
    "직접 입력": (40, 43, 98, 0, 1, 0, 0, 0),
    "정상 샘플": (40, 43, 98, 0, 1, 0, 0, 0),
    "주의 샘플": (60, 52, 82, 4, 3, 2, 1, 0),
    "고위험 샘플": (75, 48, 65, 15, 8, 5, 4, 1),
}


@st.cache_resource
def load_artifact():
    """상대경로의 학습 완료 모델을 한 번만 불러온다."""
    model_path = Path(__file__).resolve().parent / "model" / "delay_classifier.joblib"
    return joblib.load(model_path)


def risk_factors(values: dict[str, float]) -> list[str]:
    """현재 입력에서 확인되는 규칙 기반 위험요인만 간단히 설명한다."""
    factors = []
    if values["공정 편차"] <= -5:
        factors.append(f"계획 대비 실제 공정률이 {abs(values['공정 편차']):.1f}%p 낮습니다.")
    if values["인력 충족률"] < 85:
        factors.append(f"인력 충족률이 {values['인력 충족률']:.0f}%입니다.")
    if values["자재 납품 지연일"] >= 3:
        factors.append(f"자재 납품이 {values['자재 납품 지연일']:.0f}일 지연되었습니다.")
    if values["악천후 작업 중단일"] >= 3:
        factors.append(f"악천후로 {values['악천후 작업 중단일']:.0f}일 작업이 중단되었습니다.")
    if values["설계변경 횟수"] >= 2:
        factors.append(f"설계변경이 {values['설계변경 횟수']:.0f}회 발생했습니다.")
    if values["품질검사 지적 건수"] >= 2:
        factors.append(f"품질검사 지적이 {values['품질검사 지적 건수']:.0f}건입니다.")
    if values["협력업체 지연 발생 여부"]:
        factors.append("협력업체 지연이 발생했습니다.")
    return factors or ["현재 입력에서 뚜렷한 주요 위험요인이 확인되지 않았습니다."]


st.set_page_config(page_title="BuildDelay AI", page_icon="🏗️", layout="wide")
st.title("BuildDelay AI")
st.subheader("건축공사 일정 지연 위험도 분류기")
st.caption("교육용 가상 데이터와 Random Forest 모델을 사용합니다.")

try:
    artifact = load_artifact()
except FileNotFoundError:
    st.error("학습 모델이 없습니다. 프로젝트 루트에서 `python train_model.py`를 먼저 실행하세요.")
    st.stop()
except Exception as error:
    st.error(f"모델을 불러오지 못했습니다: {error}")
    st.stop()

sample_name = st.selectbox("샘플 현장", list(SAMPLES))
defaults = SAMPLES[sample_name]
left, right = st.columns(2)

with left:
    st.markdown("### 현장 데이터 입력")
    schedule = st.number_input("전체 공사 일정 경과율 (%)", 0.0, 100.0, float(defaults[0]))
    actual = st.number_input("실제 공정률 (%)", 0.0, 100.0, float(defaults[1]))
    manpower = st.number_input("인력 충족률 (%)", 0.0, 100.0, float(defaults[2]))
    material = st.number_input("자재 납품 지연일", 0, 365, int(defaults[3]))
    weather = st.number_input("악천후 작업 중단일", 0, 365, int(defaults[4]))
    changes = st.number_input("설계변경 횟수", 0, 100, int(defaults[5]))
    quality = st.number_input("품질검사 지적 건수", 0, 100, int(defaults[6]))
    subcontractor = st.selectbox("협력업체 지연 발생 여부", ["없음", "있음"], index=int(defaults[7]))
    analyze = st.button("분석 실행", type="primary")

deviation = actual - schedule
values = dict(zip(FEATURE_NAMES, [schedule, actual, manpower, material, weather, changes, quality, int(subcontractor == "있음"), deviation]))

with right:
    st.markdown("### 분석 결과")
    st.metric("공정 편차", f"{deviation:.1f}%p")
    if analyze:
        input_frame = pd.DataFrame([values], columns=artifact["feature_names"])
        probabilities = artifact["model"].predict_proba(input_frame)[0]
        probability_map = dict(zip(artifact["class_names"], probabilities))
        predicted = max(probability_map, key=probability_map.get)
        st.markdown(f"## :{ {'Low': 'green', 'Moderate': 'orange', 'High': 'red'}[predicted] }[예측 등급: {predicted} / {CLASS_LABELS[predicted]}]")
        st.metric("모델 Confidence", f"{probability_map[predicted] * 100:.1f}%")
        st.caption("Confidence는 모델이 각 클래스에 배정한 상대적 예측 확률이며 실제 지연 발생 확률이 아닙니다.")

        probability_data = pd.DataFrame({
            "등급": ["Low", "Moderate", "High"],
            "설명": [CLASS_LABELS[name] for name in ["Low", "Moderate", "High"]],
            "예측 확률": [probability_map[name] * 100 for name in ["Low", "Moderate", "High"]],
        })
        st.dataframe(probability_data, hide_index=True, width="stretch")
        chart = px.bar(probability_data, x="등급", y="예측 확률", color="등급", text_auto=".1f", color_discrete_map=COLORS, range_y=[0, 100])
        chart.update_traces(texttemplate="%{y:.1f}%")
        chart.update_layout(showlegend=False, yaxis_title="예측 확률 (%)")
        st.plotly_chart(chart, width="stretch")

        st.markdown("#### 주요 위험요인 (규칙 기반)")
        for factor in risk_factors(values):
            st.write(f"- {factor}")
    else:
        st.info("입력값을 확인한 뒤 분석 실행 버튼을 누르세요.")

st.divider()
st.warning("본 결과는 건축공학 AI 수업을 위해 생성한 가상 데이터 기반의 교육용 분석 결과이며, 실제 건설현장의 공정관리 또는 의사결정을 대체하지 않습니다.")
