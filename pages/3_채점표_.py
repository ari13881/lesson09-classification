import numpy as np
import pandas as pd
import streamlit as st
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

APP_TITLE = "뇌졸중 예측 실습실"
PAGE_TITLE = "채점표"
PAGE_ICON = "📝"
DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"

# '분류 모델' 페이지에서 처음에 골라져 있는 입력 속성(체질량지수를 뺀 넷)과 같다
FEATURES = ["age", "avg_glucose_level", "hypertension", "heart_disease"]

LOGIT_NAME = "로지스틱 회귀(확률로 답하는 모델)"
TREE_NAME = "의사결정트리(질문으로 답하는 모델)"
BASE_NAME = "기준 모델(입력을 하나도 보지 않고 훈련용에서 많은 쪽으로만 답하는 모델)"

NA_TEXT = "계산할 수 없음"
MAX_ROWS = 20

st.set_page_config(page_title=f"{PAGE_TITLE} - {APP_TITLE}", page_icon=PAGE_ICON, layout="wide")


# ---------- 데이터 불러오기 (다른 화면과 같은 주소, UTF-8) ----------
@st.cache_data(show_spinner="데이터를 불러오는 중입니다...")
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_URL, encoding="utf-8")


# ---------- 도우미 함수 ----------
def confusion_counts(y_true, y_pred) -> dict:
    """뇌졸중(1)을 양성으로 두고 TP·FP·FN·TN을 센다."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return {
        "TP": int(((y_true == 1) & (y_pred == 1)).sum()),
        "FP": int(((y_true == 0) & (y_pred == 1)).sum()),
        "FN": int(((y_true == 1) & (y_pred == 0)).sum()),
        "TN": int(((y_true == 0) & (y_pred == 0)).sum()),
    }


def confusion_table(c: dict) -> pd.DataFrame:
    """행은 실제, 열은 예측인 2×2 표(합계 포함)."""
    tp, fp, fn, tn = c["TP"], c["FP"], c["FN"], c["TN"]
    rows = {
        "실제: 뇌졸중": [f"TP · {tp:,}명", f"FN · {fn:,}명", f"{tp + fn:,}명"],
        "실제: 아님": [f"FP · {fp:,}명", f"TN · {tn:,}명", f"{fp + tn:,}명"],
        "합계": [f"{tp + fp:,}명", f"{fn + tn:,}명", f"{tp + fp + fn + tn:,}명"],
    }
    return pd.DataFrame.from_dict(rows, orient="index", columns=["예측: 뇌졸중", "예측: 아님", "합계"])


def metrics_from(c: dict) -> dict:
    """네 칸에서 정확도·재현율·정밀도·F1을 구한다. 나눌 수 없으면 None."""
    tp, fp, fn, tn = c["TP"], c["FP"], c["FN"], c["TN"]
    total = tp + fp + fn + tn
    accuracy = (tp + tn) / total if total > 0 else None
    recall = tp / (tp + fn) if (tp + fn) > 0 else None  # 분모: 실제 뇌졸중인 사람 수
    precision = tp / (tp + fp) if (tp + fp) > 0 else None  # 분모: 뇌졸중이라 예측한 사람 수
    if precision is None or recall is None:
        f1 = None
    elif precision + recall == 0:
        f1 = 0.0  # 맞힌 뇌졸중이 하나도 없는 경우
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return {"정확도": accuracy, "재현율": recall, "정밀도": precision, "F1": f1}


def pct(value) -> str:
    return NA_TEXT if value is None else f"{value * 100:.1f}%"


def person_list(people: pd.DataFrame) -> pd.DataFrame:
    """번호·나이·평균 혈당·고혈압 여부·심장병 여부를 나이 많은 순으로 정리한다."""
    ordered = people.sort_values(["age", "id"], ascending=[False, True])
    return pd.DataFrame(
        {
            "번호": ordered["id"].to_numpy(),
            "나이": ordered["age"].to_numpy(),
            "평균 혈당": ordered["avg_glucose_level"].to_numpy(),
            "고혈압": np.where(ordered["hypertension"].to_numpy() == 1, "있음", "없음"),
            "심장병": np.where(ordered["heart_disease"].to_numpy() == 1, "있음", "없음"),
        }
    )


def show_list(title: str, people: pd.DataFrame) -> None:
    total = len(people)
    shown = person_list(people).head(MAX_ROWS)
    suffix = f"전체 {total:,}명 중 앞 {MAX_ROWS}명" if total > MAX_ROWS else f"전체 {total:,}명"
    st.markdown(f"#### {title} — {suffix}")
    if total == 0:
        st.caption("해당하는 사람이 없어요.")
    else:
        st.dataframe(shown, hide_index=True)


# ---------- 화면 맨 위 ----------
st.title(f"{PAGE_ICON} {PAGE_TITLE}")
st.caption(f"{APP_TITLE} · 두 모델이 테스트 데이터에서 어떻게 맞히고 틀렸는지 채점합니다. 뇌졸중(stroke=1)을 양성으로 둡니다.")

try:
    raw = load_data()
except Exception as e:  # 네트워크 문제 등
    st.error(f"데이터를 불러오지 못했습니다. 인터넷 연결을 확인해 주세요. ({e})")
    st.stop()

# ---------- 데이터 나누기와 모델 학습 ('분류 모델' 페이지와 같은 방법) ----------
df = raw.sort_values("id").reset_index(drop=True)
is_test = (df.index % 10) < 3  # 번호 순으로 열 명씩 묶고, 각 묶음의 앞 세 명이 테스트용
train_df, test_df = df[~is_test], df[is_test]
y_train, y_test = train_df["stroke"], test_df["stroke"]
X_train, X_test = train_df[FEATURES].copy(), test_df[FEATURES].copy()

if X_train.isna().any().any() or X_test.isna().any().any():
    st.error("입력 속성에 빈 값이 있어 학습할 수 없어요.")
    st.stop()

scaler = StandardScaler().fit(X_train)  # 크기 맞추기: 훈련용으로만 계산
Xtr_scaled = scaler.transform(X_train)
Xte_scaled = scaler.transform(X_test)

logit = LogisticRegression().fit(Xtr_scaled, y_train)
tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, random_state=42).fit(X_train, y_train)
base = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)

predictions = {
    LOGIT_NAME: logit.predict(Xte_scaled),
    TREE_NAME: tree.predict(X_test),
    BASE_NAME: base.predict(X_test),
}
counts = {name: confusion_counts(y_test, pred) for name, pred in predictions.items()}

st.caption(
    f"입력 속성은 나이, 평균 혈당, 고혈압, 심장병이에요. 훈련용 {len(train_df):,}명으로 학습하고 "
    f"테스트용 {len(test_df):,}명으로 채점해요."
)

# ---------- 1. 혼동행렬 ----------
st.subheader("1. 혼동행렬 (행은 실제, 열은 예측)")
left, right = st.columns(2)
for col, name in [(left, LOGIT_NAME), (right, TREE_NAME)]:
    with col:
        st.markdown(f"**{name}**")
        st.table(confusion_table(counts[name]))
st.caption(
    "TP: 뇌졸중이라 예측했고 실제로도 뇌졸중 · FP: 뇌졸중이라 예측했지만 실제는 아님 · "
    "FN: 아니라고 예측했지만 실제는 뇌졸중 · TN: 아니라고 예측했고 실제로도 아님"
)

# ---------- 2. 지표 표 ----------
st.subheader("2. 정확도·재현율·정밀도·F1")
n_test = len(test_df)
n_pos = int((y_test == 1).sum())
st.write(f"테스트용 {n_test:,}명 중 뇌졸중인 사람은 {n_pos:,}명이에요.")

metric_rows = []
for name in [LOGIT_NAME, TREE_NAME, BASE_NAME]:
    m = metrics_from(counts[name])
    metric_rows.append(
        {
            "모델": name,
            "정확도": pct(m["정확도"]),
            "재현율": pct(m["재현율"]),
            "정밀도": pct(m["정밀도"]),
            "F1": pct(m["F1"]),
        }
    )
st.table(pd.DataFrame(metric_rows).set_index("모델"))
st.caption(
    "재현율 = TP ÷ (TP + FN), 실제 뇌졸중인 사람 중 맞힌 비율 · "
    "정밀도 = TP ÷ (TP + FP), 뇌졸중이라 예측한 사람 중 실제 뇌졸중인 비율 · "
    "F1 = 2 × 정밀도 × 재현율 ÷ (정밀도 + 재현율)"
)

# ---------- 3. 놓친 환자와 헛짚은 사람 ----------
st.subheader("3. 놓친 환자와 헛짚은 사람")
chosen = st.radio("목록을 볼 모델", [LOGIT_NAME, TREE_NAME], horizontal=True)
pred = predictions[chosen]
actual = y_test.to_numpy()
missed = test_df[(actual == 1) & (pred == 0)]  # 실제 뇌졸중인데 아니라고 예측
wrong = test_df[(actual == 0) & (pred == 1)]  # 아닌데 뇌졸중이라고 예측

lcol, rcol = st.columns(2)
with lcol:
    show_list("놓친 환자 (실제 뇌졸중인데 아니라고 예측한 사람)", missed)
with rcol:
    show_list("헛짚은 사람 (아닌데 뇌졸중이라고 예측한 사람)", wrong)

fn_cell, fp_cell = counts[chosen]["FN"], counts[chosen]["FP"]
if len(missed) == fn_cell and len(wrong) == fp_cell:
    st.write(
        f"두 목록의 사람 수({len(missed):,}명, {len(wrong):,}명)는 "
        f"{chosen}의 혼동행렬 FN 칸({fn_cell:,}명), FP 칸({fp_cell:,}명)과 같아요."
    )
else:
    st.warning(
        f"두 목록의 사람 수({len(missed):,}명, {len(wrong):,}명)가 혼동행렬 FN 칸({fn_cell:,}명), "
        f"FP 칸({fp_cell:,}명)과 달라요. 계산을 다시 확인해야 해요."
    )
