import pandas as pd
import plotly.express as px
import streamlit as st

APP_TITLE = "뇌졸중 예측 실습실"
PAGE_TITLE = "탐색"
PAGE_ICON = "🔍"
DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"

st.set_page_config(page_title=f"{PAGE_TITLE} - {APP_TITLE}", page_icon=PAGE_ICON, layout="wide")


# ---------- 데이터 불러오기 (첫 화면과 같은 주소, UTF-8) ----------
@st.cache_data(show_spinner="데이터를 불러오는 중입니다...")
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_URL, encoding="utf-8")


def stroke_rate_by(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """0/1 열(고혈압, 심장병)의 값마다 뇌졸중 비율(%)을 계산합니다."""
    labels = {0: "없음", 1: "있음"}
    out = (
        df.groupby(column)["stroke"]
        .agg(사람_수="size", 뇌졸중_사람_수="sum", 뇌졸중_비율="mean")
        .reset_index()
    )
    out["뇌졸중_비율"] = out["뇌졸중_비율"] * 100
    out["구분"] = out[column].map(labels)
    return out


st.title(f"{PAGE_ICON} {PAGE_TITLE}")
st.caption(f"{APP_TITLE} · 데이터를 여러 방향에서 살펴봅니다.")

try:
    df = load_data()
except Exception as e:  # 네트워크 문제 등
    st.error(f"데이터를 불러오지 못했습니다. 인터넷 연결을 확인해 주세요. ({e})")
    st.stop()

df = df.copy()
df["뇌졸중 여부"] = df["stroke"].map({0: "겪지 않음", 1: "겪음"})
group_order = {"뇌졸중 여부": ["겪지 않음", "겪음"]}

# ---------- 1. 히스토그램 두 개 ----------
st.subheader("1. 나이와 평균 혈당의 분포")
h1, h2 = st.columns(2)
with h1:
    fig = px.histogram(df, x="age", nbins=30, title="나이 분포", labels={"age": "나이", "count": "사람 수"})
    fig.update_layout(yaxis_title="사람 수", bargap=0.05)
    st.plotly_chart(fig)
with h2:
    fig = px.histogram(
        df,
        x="avg_glucose_level",
        nbins=40,
        title="평균 혈당 분포",
        labels={"avg_glucose_level": "평균 혈당"},
    )
    fig.update_layout(yaxis_title="사람 수", bargap=0.05)
    st.plotly_chart(fig)

# ---------- 2. 뇌졸중 여부별 상자그림 + 평균 표 ----------
st.subheader("2. 뇌졸중을 겪은 사람과 겪지 않은 사람 비교")
b1, b2 = st.columns(2)
with b1:
    fig = px.box(
        df,
        x="뇌졸중 여부",
        y="age",
        color="뇌졸중 여부",
        category_orders=group_order,
        title="나이 비교",
        labels={"age": "나이"},
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig)
with b2:
    fig = px.box(
        df,
        x="뇌졸중 여부",
        y="avg_glucose_level",
        color="뇌졸중 여부",
        category_orders=group_order,
        title="평균 혈당 비교",
        labels={"avg_glucose_level": "평균 혈당"},
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig)

st.write("두 그룹의 평균값")
mean_table = (
    df.groupby("뇌졸중 여부")[["age", "avg_glucose_level"]]
    .mean()
    .reindex(group_order["뇌졸중 여부"])
    .round(2)
    .rename(columns={"age": "나이 평균", "avg_glucose_level": "평균 혈당 평균"})
    .reset_index()
)
st.dataframe(mean_table, hide_index=True)

# ---------- 3. 고혈압·심장병별 뇌졸중 비율 ----------
st.subheader("3. 고혈압·심장병 여부에 따른 뇌졸중 비율")
r1, r2 = st.columns(2)
for col, column_name, title in [
    (r1, "hypertension", "고혈압 여부별 뇌졸중 비율"),
    (r2, "heart_disease", "심장병 여부별 뇌졸중 비율"),
]:
    rate = stroke_rate_by(df, column_name)
    with col:
        fig = px.bar(
            rate,
            x="구분",
            y="뇌졸중_비율",
            text=rate["뇌졸중_비율"].map(lambda v: f"{v:.1f}%"),
            title=title,
            labels={"구분": "있는지 여부", "뇌졸중_비율": "뇌졸중 비율 (%)"},
            category_orders={"구분": ["없음", "있음"]},
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(yaxis_title="뇌졸중 비율 (%)")
        st.plotly_chart(fig)

# ---------- 4. bmi가 빈 사람들 ----------
st.subheader("4. 체질량지수(bmi)가 비어 있는 사람들")
missing = df[df["bmi"].isna()]


def summarize(name: str, part: pd.DataFrame) -> dict:
    n = len(part)
    n_stroke = int((part["stroke"] == 1).sum())
    rate = n_stroke / n * 100 if n else 0.0
    return {
        "구분": name,
        "사람 수": n,
        "뇌졸중 사람 수": n_stroke,
        "뇌졸중 비율": f"{rate:.1f}%",
    }


bmi_table = pd.DataFrame(
    [
        summarize("bmi가 빈 사람들", missing),
        summarize("전체", df),
    ]
)
st.dataframe(bmi_table, hide_index=True)

# ---------- 5. 흡연 상태별 사람 수 ----------
st.subheader("5. 흡연 상태별 사람 수")
smoking = (
    df["smoking_status"]
    .value_counts(dropna=False)
    .rename_axis("흡연 상태")
    .reset_index(name="사람 수")
)
st.dataframe(smoking, hide_index=True)
