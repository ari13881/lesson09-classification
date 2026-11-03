import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

APP_TITLE = "뇌졸중 예측 실습실"
PAGE_TITLE = "분류 모델"
PAGE_ICON = "🤖"
DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"

ALL_FEATURES = ["age", "avg_glucose_level", "bmi", "hypertension", "heart_disease"]
DEFAULT_FEATURES = ["age", "avg_glucose_level", "hypertension", "heart_disease"]  # bmi는 처음에 뺀다
KO = {
    "age": "나이",
    "avg_glucose_level": "평균 혈당",
    "bmi": "체질량지수",
    "hypertension": "고혈압",
    "heart_disease": "심장병",
}
BINARY = ["hypertension", "heart_disease"]  # 0/1 값을 가진 속성
BINARY_QUESTION = {"hypertension": "고혈압이 없나요?", "heart_disease": "심장병이 없나요?"}

LOGIT_NAME = "로지스틱 회귀(확률로 답하는 모델)"
TREE_NAME = "의사결정트리(질문으로 답하는 모델)"
BASE_NAME = "기준 모델(입력을 하나도 보지 않고 훈련용에서 많은 쪽으로만 답하는 모델)"

COLOR_NO = "#1f77b4"
COLOR_YES = "#d62728"
REGION_NO = "rgba(31, 119, 180, 0.15)"
REGION_YES = "rgba(214, 39, 40, 0.18)"
LINE_COLOR = "#2ca02c"
NODE_ASK = "#eeeeee"
NODE_NO = "#cfe3f5"
NODE_YES = "#f6cfcf"
FONT = "Malgun Gothic, Apple SD Gothic Neo, Noto Sans CJK KR, sans-serif"

st.set_page_config(page_title=f"{PAGE_TITLE} - {APP_TITLE}", page_icon=PAGE_ICON, layout="wide")

st.markdown(
    """
    <style>
    .acc-title { font-weight: 600; min-height: 3.8em; }
    .acc-big { font-size: 2.8rem; font-weight: 700; line-height: 1.1; }
    .acc-cap { opacity: .7; font-size: .9rem; margin-bottom: .6rem; }
    .acc-row { display: flex; gap: 1.5rem; }
    .acc-small { font-size: .85rem; opacity: .85; }
    .acc-small b { font-size: 1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------- 데이터 불러오기 (다른 화면과 같은 주소, UTF-8) ----------
@st.cache_data(show_spinner="데이터를 불러오는 중입니다...")
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_URL, encoding="utf-8")


# ---------- 도우미 함수 ----------
def fmt_num(value: float) -> str:
    """12.50 -> 12.5, 3.00 -> 3 처럼 끝의 0을 지워 글로 만듭니다."""
    return f"{float(value):.2f}".rstrip("0").rstrip(".")


def accuracy_card(name: str, train_acc: float, test_acc: float) -> None:
    html = (
        f'<div class="acc-title">{name}</div>'
        f'<div class="acc-big">{test_acc * 100:.1f}%</div>'
        f'<div class="acc-cap">테스트 데이터 정확도</div>'
        f'<div class="acc-row">'
        f'<div class="acc-small">훈련 데이터 정확도<br><b>{train_acc * 100:.1f}%</b></div>'
        f'<div class="acc-small">테스트 데이터 정확도<br><b>{test_acc * 100:.1f}%</b></div>'
        f"</div>"
    )
    st.markdown(html, unsafe_allow_html=True)


def axis_range(series: pd.Series, column: str) -> tuple:
    if column in BINARY:
        return -0.3, 1.3
    lo, hi = float(series.min()), float(series.max())
    pad = (hi - lo) * 0.05 or 0.5
    return lo - pad, hi + pad


def jitter(values: pd.Series, column: str, rng: np.random.Generator) -> np.ndarray:
    """0/1 속성은 점이 겹치지 않게 그림에서만 살짝 흩어 찍습니다."""
    v = values.to_numpy(dtype=float)
    if column in BINARY:
        v = v + rng.uniform(-0.08, 0.08, size=len(v))
    return v


def build_tree_dot(model: DecisionTreeClassifier, X: pd.DataFrame, y: pd.Series, features: list):
    """의사결정트리를 DOT 문자열로 만듭니다. 훈련용 사람 수는 직접 세어 넣습니다."""
    t = model.tree_
    path = model.decision_path(X)
    n_node = np.asarray(path.sum(axis=0)).ravel().astype(int)
    n_pos = np.asarray(path.T.dot(np.asarray(y, dtype=int))).ravel().astype(int)

    lines = [
        "digraph Tree {",
        'graph [bgcolor="transparent", nodesep=0.3, ranksep=0.5];',
        f'node [shape=box, style="rounded,filled", fontname="{FONT}", fontcolor="black", margin="0.18,0.1"];',
        f'edge [fontname="{FONT}", color="#888888", fontcolor="#888888"];',
    ]
    leaf_answers = []
    for i in range(t.node_count):
        n, s = int(n_node[i]), int(n_pos[i])
        count_text = f"훈련용 {n:,}명 중 뇌졸중 {s:,}명 ({s / n * 100:.1f}%)"
        left, right = int(t.children_left[i]), int(t.children_right[i])
        if left == -1:  # 더 묻지 않고 답을 내는 마디
            answer = 1 if s > n - s else 0
            leaf_answers.append(answer)
            head = "답: 뇌졸중" if answer == 1 else "답: 아님"
            fill = NODE_YES if answer == 1 else NODE_NO
        else:
            column = features[int(t.feature[i])]
            if column in BINARY:
                head = BINARY_QUESTION[column]
            else:
                head = f"{KO[column]} ≤ {fmt_num(t.threshold[i])} ?"
            fill = NODE_ASK
        lines.append(f'n{i} [label="{head}\\n{count_text}", fillcolor="{fill}"];')
        if left != -1:
            lines.append(f'n{i} -> n{left} [label="예"];')
            lines.append(f'n{i} -> n{right} [label="아니요"];')
    lines.append("}")
    return "\n".join(lines), leaf_answers


# ---------- 화면 맨 위 ----------
st.title(f"{PAGE_ICON} {PAGE_TITLE}")
st.caption(f"{APP_TITLE} · stroke가 1이면 뇌졸중(양성), 0이면 아님으로 두고 모델 둘을 비교합니다.")

try:
    raw = load_data()
except Exception as e:  # 네트워크 문제 등
    st.error(f"데이터를 불러오지 못했습니다. 인터넷 연결을 확인해 주세요. ({e})")
    st.stop()

# ---------- 1. 입력 속성 고르기 ----------
st.subheader("1. 입력으로 사용할 속성 고르기")
selected = st.multiselect(
    "입력 속성",
    ALL_FEATURES,
    default=DEFAULT_FEATURES,
    format_func=KO.get,
)
features = [c for c in ALL_FEATURES if c in selected]  # 항상 같은 순서로 맞춘다
if len(features) < 2:
    st.info(f"입력 속성을 두 개 이상 골라 주세요. 지금은 {len(features)}개가 골라져 있어요.")
    st.stop()

# ---------- 2. 훈련용·테스트용 나누기 ----------
st.subheader("2. 데이터 나누기")
df = raw.sort_values("id").reset_index(drop=True)
is_test = (df.index % 10) < 3  # 번호 순으로 열 명씩 묶고, 각 묶음의 앞 세 명이 테스트용
train_df, test_df = df[~is_test], df[is_test]
y_train, y_test = train_df["stroke"], test_df["stroke"]
X_train, X_test = train_df[features].copy(), test_df[features].copy()

st.write(f"훈련용 {len(train_df):,}명, 테스트용 {len(test_df):,}명으로 나누었어요.")
st.caption("번호(id) 순으로 정렬해 열 명씩 묶고, 각 묶음의 앞 세 명을 테스트용으로 고정했어요. 나머지 일곱 명으로만 학습과 크기 맞추기를 해요.")

if "bmi" in features:
    bmi_median = X_train["bmi"].median()
    miss_train = int(X_train["bmi"].isna().sum())
    miss_test = int(X_test["bmi"].isna().sum())
    X_train["bmi"] = X_train["bmi"].fillna(bmi_median)
    X_test["bmi"] = X_test["bmi"].fillna(bmi_median)
    st.write(
        f"체질량지수가 빈 사람(훈련용 {miss_train:,}명, 테스트용 {miss_test:,}명)은 "
        f"훈련용의 중앙값 {fmt_num(bmi_median)}로 채웠어요."
    )

if X_train.isna().any().any() or X_test.isna().any().any():
    st.error("고른 속성에 빈 값이 남아 있어 학습할 수 없어요.")
    st.stop()

# ---------- 모델 만들기 ----------
scaler = StandardScaler().fit(X_train)  # 크기 맞추기: 훈련용으로만 계산
Xtr_scaled = scaler.transform(X_train)
Xte_scaled = scaler.transform(X_test)

logit = LogisticRegression().fit(Xtr_scaled, y_train)
tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, random_state=42).fit(X_train, y_train)
base = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)

acc = {
    LOGIT_NAME: (
        accuracy_score(y_train, logit.predict(Xtr_scaled)),
        accuracy_score(y_test, logit.predict(Xte_scaled)),
    ),
    TREE_NAME: (
        accuracy_score(y_train, tree.predict(X_train)),
        accuracy_score(y_test, tree.predict(X_test)),
    ),
    BASE_NAME: (
        accuracy_score(y_train, base.predict(X_train)),
        accuracy_score(y_test, base.predict(X_test)),
    ),
}

# ---------- 3. 정확도 카드 ----------
st.subheader("3. 정확도 비교")
cards = st.columns(3)
for col, (name, (train_acc, test_acc)) in zip(cards, acc.items()):
    with col:
        with st.container(border=True):
            accuracy_card(name, train_acc, test_acc)
st.caption(
    "로지스틱 회귀는 크기를 맞춘 값으로 학습해요. 의사결정트리는 값의 크기와 상관없이 같은 질문을 던지므로 "
    "원래 값 그대로 학습하고, 그래서 질문의 기준값도 원래 단위로 읽을 수 있어요."
)

# ---------- 4. 두 속성으로 보는 그림 ----------
st.subheader("4. 두 속성으로 보는 테스트 데이터와 경계")
ax1, ax2 = st.columns(2)
with ax1:
    x_col = st.selectbox("가로축", features, format_func=KO.get)
with ax2:
    y_col = st.selectbox("세로축", [c for c in features if c != x_col], format_func=KO.get)

x_lo, x_hi = axis_range(X_test[x_col], x_col)
y_lo, y_hi = axis_range(X_test[y_col], y_col)
gx = np.linspace(x_lo, x_hi, 150)
gy = np.linspace(y_lo, y_hi, 150)
XX, YY = np.meshgrid(gx, gy)

fixed = X_test.median()  # 두 축이 아닌 속성은 테스트 데이터의 중앙값에 세운다
grid = pd.DataFrame(
    {
        c: (XX.ravel() if c == x_col else (YY.ravel() if c == y_col else np.full(XX.size, fixed[c])))
        for c in features
    }
)
p_grid = logit.predict_proba(scaler.transform(grid))[:, 1].reshape(XX.shape)  # 뇌졸중(양성)일 확률
tree_grid = tree.predict(grid).reshape(XX.shape)
crosses = bool(p_grid.min() < 0.5 <= p_grid.max())

fig = go.Figure()
fig.add_trace(
    go.Heatmap(
        x=gx,
        y=gy,
        z=tree_grid,
        zmin=0,
        zmax=1,
        colorscale=[[0, REGION_NO], [1, REGION_YES]],
        showscale=False,
        hoverinfo="skip",
    )
)
if crosses:
    fig.add_trace(
        go.Contour(
            x=gx,
            y=gy,
            z=p_grid,
            contours=dict(start=0.5, end=0.5, size=1, coloring="none", showlabels=False),
            line=dict(color=LINE_COLOR, width=2.5, dash="dash"),
            showscale=False,
            hoverinfo="skip",
        )
    )

rng = np.random.default_rng(0)
for label, name, color, opacity in [
    (0, "실제: 아님", COLOR_NO, 0.45),
    (1, "실제: 뇌졸중", COLOR_YES, 0.9),
]:
    mask = (y_test == label).to_numpy()
    if not mask.any():
        continue
    xs, ys = X_test.loc[mask, x_col], X_test.loc[mask, y_col]
    fig.add_trace(
        go.Scatter(
            x=jitter(xs, x_col, rng),
            y=jitter(ys, y_col, rng),
            mode="markers",
            name=name,
            marker=dict(size=6, color=color, opacity=opacity),
            customdata=np.column_stack([xs.to_numpy(), ys.to_numpy()]),
            hovertemplate=(
                f"{KO[x_col]}: %{{customdata[0]}}<br>{KO[y_col]}: %{{customdata[1]}}<extra>{name}</extra>"
            ),
        )
    )

# 범례 전용 표시 (칠한 칸과 선은 범례에 직접 나오지 않으므로 빈 점으로 대신 단다)
for text, region_color, edge_color in [
    ("의사결정트리가 '아님'이라고 답하는 칸", REGION_NO, COLOR_NO),
    ("의사결정트리가 '뇌졸중'이라고 답하는 칸", REGION_YES, COLOR_YES),
]:
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            name=text,
            marker=dict(symbol="square", size=13, color=region_color, line=dict(width=1, color=edge_color)),
        )
    )
if crosses:
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="lines",
            name="로지스틱 회귀의 0.5 경계선",
            line=dict(color=LINE_COLOR, width=2.5, dash="dash"),
        )
    )

fig.update_layout(
    height=600,
    xaxis_title=KO[x_col],
    yaxis_title=KO[y_col],
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    margin=dict(t=90),
)
fig.update_xaxes(range=[x_lo, x_hi])
fig.update_yaxes(range=[y_lo, y_hi])
if x_col in BINARY:
    fig.update_xaxes(tickvals=[0, 1], ticktext=["없음", "있음"])
if y_col in BINARY:
    fig.update_yaxes(tickvals=[0, 1], ticktext=["없음", "있음"])
st.plotly_chart(fig)

st.caption(
    f"점은 테스트 데이터 {len(test_df):,}명이고, 색은 실제 뇌졸중 여부예요. "
    f"옅게 칠한 바탕은 {TREE_NAME}가 답을 나눈 칸이에요."
    + (" 고혈압·심장병처럼 0과 1만 있는 속성은 점이 겹치지 않게 그림에서만 살짝 흩어 찍었어요." if (x_col in BINARY or y_col in BINARY) else "")
)

others = [c for c in features if c not in (x_col, y_col)]
if others:
    parts = ", ".join(f"{KO[c]} {fmt_num(fixed[c])}" for c in others)
    extra = " (체질량지수는 빈 값을 채운 뒤의 값이에요.)" if "bmi" in others else ""
    st.write(f"가로축·세로축이 아닌 속성은 테스트 데이터의 중앙값에 세워 두고 계산했어요: {parts}.{extra}")
else:
    st.write("고른 속성이 두 개뿐이라 따로 세워 둔 속성은 없어요.")

if crosses:
    st.write(f"{LOGIT_NAME}가 뇌졸중일 확률을 0.5로 가르는 선을 초록 점선으로 그렸어요.")
elif p_grid.max() < 0.5:
    st.write(
        f"{LOGIT_NAME}의 0.5 경계선은 이 그림 밖에 있어서 그리지 못했어요. "
        f"그림 안의 뇌졸중 확률은 {p_grid.min():.1%}에서 {p_grid.max():.1%}까지로, 어디서나 0.5보다 낮아요."
    )
else:
    st.write(
        f"{LOGIT_NAME}의 0.5 경계선은 이 그림 밖에 있어서 그리지 못했어요. "
        f"그림 안의 뇌졸중 확률은 {p_grid.min():.1%}에서 {p_grid.max():.1%}까지로, 어디서나 0.5 이상이에요."
    )

# ---------- 5. 의사결정트리 가지 그림 ----------
st.subheader("5. 의사결정트리가 던진 질문")
st.caption("질문은 최대 세 번까지, 마지막 마디에 사람이 다섯 명 미만이 되는 나눔은 하지 않도록 했어요. 난수는 42로 고정했어요.")
dot, leaf_answers = build_tree_dot(tree, X_train, y_train, features)
st.graphviz_chart(dot)

n_leaves = len(leaf_answers)
n_no = sum(1 for a in leaf_answers if a == 0)
asked = sorted({int(f) for f in tree.tree_.feature if f >= 0})
asked_text = ", ".join(KO[features[i]] for i in asked) if asked else "없음"
st.markdown(
    f"- 답을 내는 마디는 모두 {n_leaves}칸이고, 그중 {n_no}칸이 '아님'이라고 답해요.\n"
    f"- 고른 속성 가운데 이 나무가 실제로 물은 것: {asked_text}"
)
