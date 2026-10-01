"""BuildDelay AI Streamlit 앱: 기업용 건설 공정관리 AI 위험도 분석 모듈 (Enterprise Demo)."""

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
STATUS_COLORS = {"Low": "green", "Moderate": "orange", "High": "red"}
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


def on_sample_change():
    """샘플 시나리오 선택 시 입력 위젯 값을 해당 프리셋으로 동기화한다."""
    selected = st.session_state.get("selected_sample", "직접 입력")
    vals = SAMPLES[selected]
    st.session_state["schedule"] = float(vals[0])
    st.session_state["actual"] = float(vals[1])
    st.session_state["manpower"] = float(vals[2])
    st.session_state["material"] = int(vals[3])
    st.session_state["weather"] = int(vals[4])
    st.session_state["changes"] = int(vals[5])
    st.session_state["quality"] = int(vals[6])
    st.session_state["subcontractor"] = "있음" if vals[7] else "없음"


# 세션 상태 초기화
if "selected_sample" not in st.session_state:
    st.session_state["selected_sample"] = "직접 입력"
if "schedule" not in st.session_state:
    on_sample_change()

st.set_page_config(
    page_title="BuildDelay AI | Construction Risk Intelligence",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 사이드바: Enterprise Demo Workspace 정보
with st.sidebar:
    st.markdown("### 🏗️ BuildDelay AI")
    st.caption("Construction Risk Intelligence Module v1.0")

    with st.container(border=True):
        st.markdown("**Enterprise Demo Workspace**")
        st.caption("🏢 회사/조직: **가천건설**")
        st.caption("📍 관리현장: **성남 복합시설 신축공사**")
        st.caption("👤 사용자: **데모 사용자**")
        st.caption("💼 역할: **공정관리자**")
        st.caption("🟢 엔진: **Random Forest (Active)**")

    with st.container(border=True):
        st.markdown("💡 **Enterprise Demo 안내**")
        st.caption(
            "본 시스템은 실제 회원가입/인증 기능 없이 기업용 건설 공정관리 SaaS 환경을 체험할 수 있도록 구성된 포트폴리오 데모입니다."
        )

# 상단 헤더 및 기업용 컨텍스트 바
with st.container(border=True):
    col_brand, col_context = st.columns([7, 5])
    with col_brand:
        st.title("BuildDelay AI")
        st.markdown("##### Construction Risk Intelligence")
        st.caption("교육용 가상 데이터 기반 건설공사 일정 지연 위험도 예측 및 의사결정 지원 대시보드")

        badge_columns = st.columns(4)
        badge_columns[0].badge("Enterprise Demo", color="blue")
        badge_columns[1].badge("Random Forest", color="gray")
        badge_columns[2].badge("Synthetic Dataset 600", color="gray")
        badge_columns[3].badge("3 Risk Classes", color="green")

    with col_context:
        with st.container(border=True):
            st.markdown("💼 **Enterprise Context**")
            selected_project = st.selectbox(
                "관리 대상 현장 (Demo)",
                [
                    "성남 복합시설 신축공사 (진행중)",
                    "판교 제2테크노밸리 오피스 (계획)",
                    "위례 복합문화타운 (설계)",
                ],
                index=0,
                help="기업용 공정관리 시스템의 다중 현장 선택 UI 시뮬레이션입니다.",
            )
            c1, c2 = st.columns(2)
            c1.caption("🏢 **조직:** 가천건설")
            c2.caption("👤 **사용자:** 데모 사용자 (공정관리자)")

# 모델 로드
try:
    artifact = load_artifact()
except FileNotFoundError:
    st.error("학습 모델이 없습니다. 프로젝트 루트에서 `python train_model.py`를 먼저 실행하세요.")
    st.stop()
except Exception as error:
    st.error(f"모델을 불러오지 못했습니다: {error}")
    st.stop()

# 탭 기반 Enterprise 모듈 네비게이션
tab_analysis, tab_project, tab_model = st.tabs([
    "📊 AI 공정 리스크 분석",
    "🏢 프로젝트 종합 개요",
    "ℹ️ 데이터 및 모델 정보",
])

# 1. AI 공정 리스크 분석 모듈 (메인 대시보드)
with tab_analysis:
    st.markdown("#### ⚡ 빠른 시나리오 선택")
    sample_name = st.radio(
        "샘플 시나리오",
        list(SAMPLES),
        horizontal=True,
        label_visibility="collapsed",
        key="selected_sample",
        on_change=on_sample_change,
        help="검증된 샘플 시나리오를 선택하여 즉시 분석을 테스트할 수 있습니다.",
    )

    left, right = st.columns([5, 7])

    with left:
        st.markdown("### 📋 현장 데이터 입력")
        st.caption("현재 현장 조건을 입력하면 공정 편차와 일정 지연 위험도를 함께 분석합니다.")

        with st.container(border=True):
            st.markdown("##### 1. 공정 진행 현황")
            schedule = st.number_input(
                "전체 공사 일정 경과율 (%)", 0.0, 100.0,
                help="전체 계획 일정 중 현재까지 경과한 비율입니다.",
                key="schedule",
            )
            actual = st.number_input(
                "실제 공정률 (%)", 0.0, 100.0,
                help="실제로 완료된 공사의 비율입니다.",
                key="actual",
            )
            deviation = actual - schedule
            dev_status = (
                "계획 대비 초과달성" if deviation > 0
                else "계획 공정 일치" if deviation == 0
                else f"계획 대비 지연 ({abs(deviation):.1f}%p)"
            )
            st.caption(f"💡 산출 공정 편차: **{deviation:+.1f}%p** ({dev_status})")

        with st.container(border=True):
            st.markdown("##### 2. 자원 및 조달")
            manpower = st.number_input(
                "인력 충족률 (%)", 0.0, 100.0,
                help="계획 인력 대비 실제 확보 인력의 비율입니다.",
                key="manpower",
            )
            material = st.number_input(
                "자재 납품 지연일 (일)", 0, 365,
                help="주요 자재가 계획보다 늦어진 일수입니다.",
                key="material",
            )

        with st.container(border=True):
            st.markdown("##### 3. 현장 리스크 요인")
            weather = st.number_input(
                "악천후 작업 중단일 (일)", 0, 365,
                help="악천후로 작업을 중단한 일수입니다.",
                key="weather",
            )
            changes = st.number_input(
                "설계변경 횟수 (회)", 0, 100,
                help="공사 중 발생한 설계변경 횟수입니다.",
                key="changes",
            )
            quality = st.number_input(
                "품질검사 지적 건수 (건)", 0, 100,
                help="품질검사에서 발생한 지적 건수입니다.",
                key="quality",
            )
            subcontractor = st.selectbox(
                "협력업체 지연 발생 여부", ["없음", "있음"],
                help="협력업체 사유의 지연이 발생했는지 선택합니다.",
                key="subcontractor",
            )

        analyze = st.button("🚀 위험도 분석 실행", type="primary", use_container_width=True)

    deviation = actual - schedule
    values = dict(zip(
        FEATURE_NAMES,
        [schedule, actual, manpower, material, weather, changes, quality, int(subcontractor == "있음"), deviation]
    ))

    with right:
        st.markdown("### 📈 AI 분석 결과")
        if analyze:
            input_frame = pd.DataFrame([values], columns=artifact["feature_names"])
            probabilities = artifact["model"].predict_proba(input_frame)[0]
            probability_map = dict(zip(artifact["class_names"], probabilities))
            predicted = max(probability_map, key=probability_map.get)

            # 1. 핵심 진단 카드
            with st.container(border=True):
                st.caption("AI 위험도 예측 등급 (Primary Risk Diagnosis)")
                st.markdown(f"## :{STATUS_COLORS[predicted]}[{predicted.upper()}] — {CLASS_LABELS[predicted]}")

                m1, m2 = st.columns(2)
                m1.metric("Model Confidence", f"{probability_map[predicted] * 100:.1f}%")
                m2.metric("공정 편차", f"{deviation:+.1f}%p")
                st.caption("※ **Confidence 안내**: 모델의 `predict_proba()` 기반 상대적 예측 확률(신뢰도)이며 실제 공사 지연 발생 확률이나 모델의 정확도가 아닙니다.")

            # 2. 클래스별 예측 확률
            st.markdown("#### 📊 클래스별 예측 확률")
            probability_data = pd.DataFrame({
                "등급": ["Low", "Moderate", "High"],
                "설명": [CLASS_LABELS[name] for name in ["Low", "Moderate", "High"]],
                "예측 확률 (%)": [probability_map[name] * 100 for name in ["Low", "Moderate", "High"]],
            })
            st.dataframe(probability_data, hide_index=True, width="stretch")

            chart = px.bar(
                probability_data,
                x="등급",
                y="예측 확률 (%)",
                color="등급",
                text_auto=".1f",
                color_discrete_map=COLORS,
                range_y=[0, 100],
            )
            chart.update_traces(texttemplate="%{y:.1f}%", textposition="outside")
            chart.update_layout(
                showlegend=False,
                yaxis_title="예측 확률 (%)",
                xaxis_title="",
                margin=dict(l=20, r=20, t=20, b=20),
                height=260,
            )
            st.plotly_chart(chart, width="stretch")

            # 3. KPI 요약 카드
            st.markdown("#### 📌 현장 핵심 KPI 요약")
            kpi_cols = st.columns(3)
            kpi_cols[0].metric("인력 충족률", f"{manpower:.0f}%")
            kpi_cols[1].metric("자재 지연", f"{material}일")
            kpi_cols[2].metric("악천후 중단", f"{weather}일")

            kpi_cols2 = st.columns(2)
            kpi_cols2[0].metric("설계변경 / 품질지적", f"{changes}회 / {quality}건")
            kpi_cols2[1].metric("협력업체 지연", subcontractor)

            # 4. 주요 위험요인 분석
            with st.container(border=True):
                st.markdown("#### ⚠️ 주요 위험요인 분석 (규칙 기반)")
                for factor in risk_factors(values):
                    if "뚜렷한 주요 위험요인이 확인되지 않았습니다" in factor:
                        st.markdown(f"✅ {factor}")
                    else:
                        st.markdown(f"• {factor}")
                st.caption("※ 위 분석은 현장 입력 수치에 기반한 규칙 기반 진단 결과입니다.")

            # 5. 모델 설명 expander
            with st.expander("🔍 이 모델은 어떻게 판단하나요?"):
                st.markdown(
                    f"- **모델 아키텍처**: Random Forest Classifier (`n_estimators=200`)\n"
                    f"- **학습 데이터**: 600개 교육용 가상 데이터 (Train 80% / Test 20% Stratified Split)\n"
                    f"- **8개 현장 변수 및 공정 편차**를 결합하여 Low / Moderate / High 세 등급으로 분류\n"
                    f"- **실제 테스트 검증 성능**: Accuracy {artifact['metrics']['accuracy'] * 100:.1f}%, Macro F1 {artifact['metrics']['macro_f1'] * 100:.1f}%\n"
                    f"- **Confidence 해석**: `predict_proba()` 기반의 상대적 분류 신뢰도이며 실제 현장의 지연 발생 확률이 아닙니다."
                )
        else:
            with st.container(border=True):
                st.info("👈 좌측 현장 데이터를 확인하거나 상단 [빠른 시나리오]를 선택한 뒤 **[🚀 위험도 분석 실행]** 버튼을 누르세요.")

# 2. 프로젝트 개요 탭
with tab_project:
    st.markdown("### 🏢 프로젝트 종합 개요 (Enterprise Profile)")
    st.caption("선택된 관리 프로젝트의 기본 사업 개요 및 공정관리 프로파일입니다.")

    with st.container(border=True):
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            st.markdown("#### 📌 사업 개요")
            st.write("**프로젝트명**: 성남 복합시설 신축공사 (본관동)")
            st.write("**발주처**: 가천개발사업단")
            st.write("**시공사**: 가천건설 (주) 건축사업본부")
            st.write("**현장 위치**: 경기도 성남시 수정구 성남대로 일원")
            st.write("**공사 기간**: 2025.03.01 ~ 2027.02.28 (24개월)")
        with col_p2:
            st.markdown("#### 📐 공사 규모 및 구조")
            st.write("**건축 용도**: 업무 및 복합문화시설")
            st.write("**대지면적 / 연면적**: 12,450 ㎡ / 48,500 ㎡")
            st.write("**건물 규모**: 지하 3층 / 지상 15층")
            st.write("**주요 구조**: 철근콘크리트(RC) 및 철골조(SC)")
            st.write("**현재 중점 공종**: 지상 골조 공사 및 외벽 커튼월")

    with st.container(border=True):
        st.markdown("#### 🛡️ AI 공정관리 모듈 연계 상태")
        st.write("• **실시간 리스크 예측 엔진**: `RandomForestClassifier` 가동 중")
        st.write("• **공정 데이터 파이프라인**: 8대 현장 핵심 지표 및 공정 편차 자동 연계")
        st.write("• **리스크 대응 프로토콜**: High 등급 감지 시 현장 공정관리자 조기경보 발령 체계")

# 3. 데이터 및 모델 정보 탭
with tab_model:
    st.markdown("### ℹ️ AI 모델 아키텍처 및 데이터 거버넌스")
    st.caption("머신러닝 파이프라인 구조, 검증 지표 및 가상 데이터 거버넌스 정책 안내입니다.")

    with st.container(border=True):
        st.markdown("#### 1. 머신러닝 파이프라인 아키텍처")
        st.write("• **모델 알고리즘**: Random Forest Classifier (`sklearn.ensemble.RandomForestClassifier`)")
        st.write("• **핵심 파라미터**: `n_estimators=200, random_state=42`")
        st.write("• **데이터 분할**: Train 80% (480건) / Test 20% (120건) Stratified Split 적용")
        st.write(
            f"• **검증 세트 성능**: Accuracy **{artifact['metrics']['accuracy'] * 100:.1f}%** | "
            f"Macro Precision **{artifact['metrics']['macro_precision'] * 100:.1f}%** | "
            f"Macro Recall **{artifact['metrics']['macro_recall'] * 100:.1f}%** | "
            f"Macro F1 **{artifact['metrics']['macro_f1'] * 100:.1f}%**"
        )

    with st.container(border=True):
        st.markdown("#### 2. 입력 Feature 및 파생변수 정의 (8종 + 1 파생변수)")
        st.write("1. **전체 공사 일정 경과율 (%)**: 계획 전체 일정 대비 현재 시점까지의 경과 비율")
        st.write("2. **실제 공정률 (%)**: 현장에서 실제 완료된 작업의 진행 비율")
        st.write("3. **공정 편차 (%p)**: `실제 공정률 - 전체 공사 일정 경과율` (핵심 파생변수)")
        st.write("4. **인력 충족률 (%)**: 계획 인력 대비 현장 투입 인력 비율")
        st.write("5. **자재 납품 지연일 (일)**: 레미콘, 철근 등 주요 자재 입고 지연 일수")
        st.write("6. **악천후 작업 중단일 (일)**: 강우, 강풍 등으로 인한 작업 중단 일수")
        st.write("7. **설계변경 횟수 (회)**: 시공 중 승인된 설계변경 누적 건수")
        st.write("8. **품질검사 지적 건수 (건)**: 감리단 및 품질점검 지적 사항 건수")
        st.write("9. **협력업체 지연 발생 여부**: 협력사 공정 지연 발생 여부 (0: 없음, 1: 있음)")

    with st.container(border=True):
        st.markdown("#### 3. 설명 및 데이터 거버넌스 원칙")
        st.write("• **Confidence 정의**: Random Forest 모델의 `predict_proba()` 기반 상대적 분류 신뢰도이며 실제 현장의 지연 발생 확률이 아닙니다.")
        st.write("• **합성 데이터의 한계**: 본 시스템은 600개의 교육용 가상 데이터셋으로 학습되었으므로 실제 건설 현장의 공식 의사결정을 대체하지 않습니다.")

# 하단 고지
st.divider()
st.warning("본 결과는 건축공학 AI 수업을 위해 생성한 가상 데이터 기반의 교육용 분석 결과이며, 실제 건설현장의 공정관리 또는 의사결정을 대체하지 않습니다.")
