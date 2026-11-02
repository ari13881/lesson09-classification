import json
from pathlib import Path

import pandas as pd
import streamlit as st

APP_TITLE = "뇌졸중 예측 실습실"
APP_ICON = "🧠"
DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"
NOTE_DIR = Path(__file__).parent / "내_기록"
NOTE_FILE = NOTE_DIR / "기록.json"

st.set_page_config(page_title=APP_TITLE, page_icon=APP_ICON, layout="wide")

st.markdown(
    """
    <style>
    [data-testid="stMetricValue"] { font-size: 2.8rem; font-weight: 700; }
    [data-testid="stMetricLabel"] p { font-size: 1.05rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------- 데이터 불러오기 ----------
@st.cache_data(show_spinner="데이터를 불러오는 중입니다...")
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_URL, encoding="utf-8")


# ---------- 내 기록(json) 읽기·쓰기 ----------
def load_notes() -> dict:
    try:
        with open(NOTE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        meanings = data.get("우리말 뜻", {})
        source = data.get("데이터 출처", "")
        return {
            "우리말 뜻": meanings if isinstance(meanings, dict) else {},
            "데이터 출처": source if isinstance(source, str) else "",
        }
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {"우리말 뜻": {}, "데이터 출처": ""}


def save_notes(notes: dict) -> None:
    NOTE_DIR.mkdir(parents=True, exist_ok=True)
    with open(NOTE_FILE, "w", encoding="utf-8") as f:
        json.dump(notes, f, ensure_ascii=False, indent=2)


def describe_values(series: pd.Series) -> str:
    """열의 값 종류를 짧은 글로 설명합니다."""
    values = series.dropna()
    n_unique = values.nunique()
    if series.dtype == "object" or n_unique <= 10:
        shown = sorted(values.unique().tolist(), key=str)
        return ", ".join(str(v) for v in shown)
    return f"숫자 (최소 {values.min():g} ~ 최대 {values.max():g}, 서로 다른 값 {n_unique}개)"


# ---------- 화면 맨 위 ----------
st.title(f"{APP_ICON} {APP_TITLE}")
st.caption("첫 화면: 이 데이터가 무엇인지 살펴봅니다.")

try:
    df = load_data()
except Exception as e:  # 네트워크 문제 등
    st.error(f"데이터를 불러오지 못했습니다. 인터넷 연결을 확인해 주세요. ({e})")
    st.stop()

notes = load_notes()

# ---------- 1. 큰 숫자 카드 네 개 ----------
total = len(df)
n_cols = df.shape[1]
n_stroke = int((df["stroke"] == 1).sum())
ratio = n_stroke / total * 100 if total else 0.0

c1, c2, c3, c4 = st.columns(4)
with c1:
    with st.container(border=True):
        st.metric("전체 사람 수", f"{total:,}명")
with c2:
    with st.container(border=True):
        st.metric("열 개수", f"{n_cols}개")
with c3:
    with st.container(border=True):
        st.metric("stroke가 1인 사람 수", f"{n_stroke:,}명")
with c4:
    with st.container(border=True):
        st.metric("stroke가 1인 비율", f"{ratio:.1f}%")

# ---------- 2. 열 설명 표 ----------
st.subheader("열 설명 표")
st.write("**우리말 뜻** 칸을 눌러 교재를 보고 직접 적은 뒤, 아래 저장 버튼을 눌러 주세요.")

info = pd.DataFrame(
    {
        "열 이름": df.columns,
        "우리말 뜻": [notes["우리말 뜻"].get(c, "") for c in df.columns],
        "값의 종류": [describe_values(df[c]) for c in df.columns],
        "빈 값 개수": [int(df[c].isna().sum()) for c in df.columns],
    }
)

with st.form("meaning_form"):
    edited = st.data_editor(
        info,
        hide_index=True,
        use_container_width=True,
        disabled=["열 이름", "값의 종류", "빈 값 개수"],
        column_config={
            "우리말 뜻": st.column_config.TextColumn("우리말 뜻", help="교재를 보고 직접 적어 보세요."),
        },
        key="meaning_editor",
    )
    saved_meaning = st.form_submit_button("우리말 뜻 저장하기")

if saved_meaning:
    latest = load_notes()
    latest["우리말 뜻"] = {
        row["열 이름"]: (row["우리말 뜻"] or "") for _, row in edited.iterrows()
    }
    try:
        save_notes(latest)
        st.success("우리말 뜻을 저장했습니다.")
    except OSError as e:
        st.error(f"저장하지 못했습니다: {e}")

# ---------- 3. 처음 다섯 줄 ----------
st.subheader("데이터 처음 다섯 줄")
st.dataframe(df.head(5), use_container_width=True, hide_index=True)

# ---------- 4. 데이터 출처 ----------
st.subheader("데이터 출처")
with st.form("source_form"):
    source_text = st.text_area(
        "교재에 있는 출처 글을 적어 주세요.",
        value=notes["데이터 출처"],
        height=140,
    )
    saved_source = st.form_submit_button("출처 저장하기")

if saved_source:
    latest = load_notes()
    latest["데이터 출처"] = source_text
    try:
        save_notes(latest)
        st.success("출처를 저장했습니다.")
    except OSError as e:
        st.error(f"저장하지 못했습니다: {e}")
