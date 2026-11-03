import itertools

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

APP_TITLE = "뇌졸중 예측 실습실"
PAGE_TITLE = "성능 높이기"
PAGE_ICON = "🚀"
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

LOGIT_NAME = "로지스틱 회귀(확률로 답하는 모델)"
TREE_NAME = "의사결정트리(질문으로 답하는 모델)"

NA_TEXT = "계산할 수 없음"
COLOR_NO = "#1f77b4"
COLOR_YES = "#d62728"
COLOR_BEFORE = "#2ca02c"  # 가중치를 맞추기 전 0.5 경계: 초록
COLOR_AFTER = "#9467bd"  # 가중치를 맞춘 뒤 0.5 경계: 보라

CELL_LABELS = [
    ("TP", "TP (찾아낸 환자)"),
    ("FP", "FP (헛짚은 사람)"),
    ("FN", "FN (놓친 환자)"),
    ("TN", "TN (제대로 아니라고 한 사람)"),
]

st.set_page_config(page_title=f"{PAGE_TITLE} - {APP_TITLE}", page_icon=PAGE_ICON, layout="wide")


# ---------- 데이터 불러오기 (다른 화면과 같은 주소, UTF-8) ----------
@st.cache_data(show_spinner="데이터를 불러오는 중입니다...")
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_URL, encoding="utf-8")


# ---------- 도우미 함수 ----------
def fmt_num(value: float) -> str:
    """12.50 -> 12.5, 3.00 -> 3 처럼 끝의 0을 지워 글로 만듭니다."""
    return f"{float(value):.2f}".rstrip("0").rstrip(".")


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


def before_after_table(before: dict, after: dict) -> pd.DataFrame:
    """네 칸과 네 지표를 맞추기 전·후로 나란히 놓은 표."""
    m0, m1 = metrics_from(before), metrics_from(after)
    rows = {}
    for key, label in CELL_LABELS:
        rows[label] = [f"{before[key]:,}명", f"{after[key]:,}명"]
    for key in ["정확도", "재현율", "정밀도", "F1"]:
        rows[key] = [pct(m0[key]), pct(m1[key])]
    return pd.DataFrame.from_dict(rows, orient="index", columns=["맞추기 전", "맞춘 뒤"])


def change_line(before: dict, after: dict) -> str:
    return (
        f"찾아낸 환자 {before['TP']:,}명 → {after['TP']:,}명, "
        f"놓친 환자 {before['FN']:,}명 → {after['FN']:,}명, "
        f"헛짚은 사람 {before['FP']:,}명 → {after['FP']:,}명으로 바뀌었어요."
    )


def linear_form(model: LogisticRegression, scaler: StandardScaler, features: list):
    """크기를 맞춘 모델의 식을 원래 단위의 식(절편 + 계수×값)으로 바꾼다. 0.5 경계는 식의 값이 0인 곳."""
    w = model.coef_[0]
    coefs = dict(zip(features, w / scaler.scale_))
    intercept = float(model.intercept_[0] - np.sum(w * scaler.mean_ / scaler.scale_))
    return coefs, intercept


def box_probability_range(coefs, intercept, fixed, axes, ranges, features):
    """세 축이 만드는 상자 안에서 뇌졸중 확률의 가장 낮은 값과 높은 값(상자 꼭짓점에서 정해진다)."""
    c0 = intercept + sum(coefs[f] * fixed[f] for f in features if f not in axes)
    probs = []
    for corner in itertools.product(*ranges):
        f = c0 + sum(coefs[a] * v for a, v in zip(axes, corner))
        probs.append(1.0 / (1.0 + np.exp(-f)))
    return min(probs), max(probs)


def plane_surface(coefs, intercept, fixed, axes, ranges, features, n=40):
    """0.5 평면의 좌표(2차원 배열 셋). 계수가 가장 큰 축을 나머지 둘로 풀어 그린다."""
    c0 = intercept + sum(coefs[f] * fixed[f] for f in features if f not in axes)
    a = [coefs[ax] for ax in axes]
    k = int(np.argmax(np.abs(a)))
    if abs(a[k]) < 1e-12:
        return None
    i, j = [m for m in range(3) if m != k]
    U, V = np.meshgrid(np.linspace(*ranges[i], n), np.linspace(*ranges[j], n))
    S = -(c0 + a[i] * U + a[j] * V) / a[k]
    lo, hi = ranges[k]
    S = np.where((S >= lo) & (S <= hi), S, np.nan)  # 상자 밖은 자른다
    if not np.isfinite(S).any():
        return None
    coords = [None, None, None]
    coords[i], coords[j], coords[k] = U, V, S
    return coords


def boundary_text(label: str, shape: str, low: float, high: float, drawn: bool, line_desc: str) -> str:
    if drawn:
        return f"{label} 0.5 {shape}은 {line_desc}으로 그렸어요."
    if high < 0.5:
        return (
            f"{label} 0.5 {shape}은 이 그림 밖에 있어서 그리지 못했어요. "
            f"그림 안의 뇌졸중 확률은 {low:.1%}에서 {high:.1%}까지로, 어디서나 0.5보다 낮아요."
        )
    return (
        f"{label} 0.5 {shape}은 이 그림 밖에 있어서 그리지 못했어요. "
        f"그림 안의 뇌졸중 확률은 {low:.1%}에서 {high:.1%}까지로, 어디서나 0.5 이상이에요."
    )


def scene_axis(column: str, value_range: tuple) -> dict:
    axis = {"title": KO[column], "range": [value_range[0], value_range[1]]}
    if column in BINARY:
        axis.update(tickvals=[0, 1], ticktext=["없음", "있음"])
    return axis


# ---------- 화면 맨 위 ----------
st.title(f"{PAGE_ICON} {PAGE_TITLE}")
st.caption(
    f"{APP_TITLE} · 뇌졸중인 사람이 아주 드물어서, 양쪽의 가중치를 맞추기 전과 맞춘 뒤의 결과를 나란히 비교합니다. "
    "뇌졸중(stroke=1)을 양성으로 둡니다."
)

# ---------- 1. 입력 속성 고르기 ('분류 모델' 페이지와 같은 자리) ----------
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

try:
    raw = load_data()
except Exception as e:  # 네트워크 문제 등
    st.error(f"데이터를 불러오지 못했습니다. 인터넷 연결을 확인해 주세요. ({e})")
    st.stop()

# ---------- 2. 데이터 나누기 ('채점표' 페이지와 같은 방법) ----------
st.subheader("2. 데이터 나누기와 가중치")
df = raw.sort_values("id").reset_index(drop=True)
is_test = (df.index % 10) < 3  # 번호 순으로 열 명씩 묶고, 각 묶음의 앞 세 명이 테스트용
train_df, test_df = df[~is_test], df[is_test]
y_train, y_test = train_df["stroke"], test_df["stroke"]
X_train, X_test = train_df[features].copy(), test_df[features].copy()

st.write(f"훈련용 {len(train_df):,}명, 테스트용 {len(test_df):,}명으로 나누었어요.")

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
if y_train.nunique() < 2:
    st.error("훈련용에 뇌졸중인 사람과 아닌 사람이 모두 있어야 학습할 수 있어요.")
    st.stop()

n_train = len(y_train)
n_pos_train = int((y_train == 1).sum())
n_neg_train = n_train - n_pos_train
w_pos = n_train / (2 * n_pos_train)
w_neg = n_train / (2 * n_neg_train)
st.write(
    f"훈련용 {n_train:,}명 중 뇌졸중인 사람은 {n_pos_train:,}명({n_pos_train / n_train * 100:.1f}%)이에요. "
    f"가중치를 맞추면 뇌졸중인 사람 한 명의 무게는 {w_pos:.2f}, 아닌 사람 한 명의 무게는 {w_neg:.2f}가 되어 "
    "양쪽 무게의 합이 같아져요."
)

# ---------- 모델 만들기: 가중치를 맞추기 전과 맞춘 뒤 ----------
scaler = StandardScaler().fit(X_train)  # 크기 맞추기: 훈련용으로만 계산
Xtr_scaled = scaler.transform(X_train)
Xte_scaled = scaler.transform(X_test)

logit_before = LogisticRegression().fit(Xtr_scaled, y_train)
logit_after = LogisticRegression(class_weight="balanced").fit(Xtr_scaled, y_train)
tree_before = DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, random_state=42).fit(X_train, y_train)
tree_after = DecisionTreeClassifier(
    max_depth=3, min_samples_leaf=5, random_state=42, class_weight="balanced"
).fit(X_train, y_train)

logit_counts = (
    confusion_counts(y_test, logit_before.predict(Xte_scaled)),
    confusion_counts(y_test, logit_after.predict(Xte_scaled)),
)
tree_counts = (
    confusion_counts(y_test, tree_before.predict(X_test)),
    confusion_counts(y_test, tree_after.predict(X_test)),
)

# ---------- 3. 전·후 비교 표 ----------
st.subheader("3. 가중치를 맞추기 전과 맞춘 뒤 (테스트 데이터)")
st.caption(
    f"테스트용 {len(test_df):,}명 중 뇌졸중인 사람은 {int((y_test == 1).sum()):,}명이에요. "
    "표의 네 칸은 사람 수이고, 정밀도와 F1은 뇌졸중이라 예측한 사람이 0명이면 계산할 수 없어요."
)
lcol, rcol = st.columns(2)
for col, name, (before, after) in [
    (lcol, LOGIT_NAME, logit_counts),
    (rcol, TREE_NAME, tree_counts),
]:
    with col:
        st.markdown(f"**{name}**")
        st.table(before_after_table(before, after))
        st.write(change_line(before, after))

# ---------- 4. 두 속성으로 보는 0.5 경계선 ----------
st.subheader(f"4. 두 속성으로 보는 {LOGIT_NAME}의 0.5 경계선")
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

fixed = X_test.median()  # 고르지 않은 축의 속성은 테스트 데이터의 중앙값에 세운다
grid = pd.DataFrame(
    {
        c: (XX.ravel() if c == x_col else (YY.ravel() if c == y_col else np.full(XX.size, fixed[c])))
        for c in features
    }
)
grid_scaled = scaler.transform(grid)
p_before = logit_before.predict_proba(grid_scaled)[:, 1].reshape(XX.shape)  # 뇌졸중(양성)일 확률
p_after = logit_after.predict_proba(grid_scaled)[:, 1].reshape(XX.shape)
cross_before = bool(p_before.min() < 0.5 <= p_before.max())
cross_after = bool(p_after.min() < 0.5 <= p_after.max())

fig2d = go.Figure()
for p_grid, crosses, color, dash in [
    (p_before, cross_before, COLOR_BEFORE, "dash"),
    (p_after, cross_after, COLOR_AFTER, "solid"),
]:
    if crosses:
        fig2d.add_trace(
            go.Contour(
                x=gx,
                y=gy,
                z=p_grid,
                contours=dict(start=0.5, end=0.5, size=1, coloring="none", showlabels=False),
                line=dict(color=color, width=3, dash=dash),
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
    fig2d.add_trace(
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

# 범례 전용 표시 (경계선은 범례에 직접 나오지 않으므로 빈 선으로 대신 단다)
for crosses, name, color, dash in [
    (cross_before, "가중치를 맞추기 전 0.5 경계선", COLOR_BEFORE, "dash"),
    (cross_after, "가중치를 맞춘 뒤 0.5 경계선", COLOR_AFTER, "solid"),
]:
    if crosses:
        fig2d.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="lines",
                name=name,
                line=dict(color=color, width=3, dash=dash),
            )
        )

fig2d.update_layout(
    height=600,
    xaxis_title=KO[x_col],
    yaxis_title=KO[y_col],
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    margin=dict(t=90),
)
fig2d.update_xaxes(range=[x_lo, x_hi])
fig2d.update_yaxes(range=[y_lo, y_hi])
if x_col in BINARY:
    fig2d.update_xaxes(tickvals=[0, 1], ticktext=["없음", "있음"])
if y_col in BINARY:
    fig2d.update_yaxes(tickvals=[0, 1], ticktext=["없음", "있음"])
st.plotly_chart(fig2d)

st.caption(
    f"점은 테스트 데이터 {len(test_df):,}명이고, 색은 실제 뇌졸중 여부예요."
    + (" 고혈압·심장병처럼 0과 1만 있는 속성은 점이 겹치지 않게 그림에서만 살짝 흩어 찍었어요." if (x_col in BINARY or y_col in BINARY) else "")
)

others_2d = [c for c in features if c not in (x_col, y_col)]
if others_2d:
    parts = ", ".join(f"{KO[c]} {fmt_num(fixed[c])}" for c in others_2d)
    extra = " (체질량지수는 빈 값을 채운 뒤의 값이에요.)" if "bmi" in others_2d else ""
    st.write(f"가로축·세로축이 아닌 속성은 테스트 데이터의 중앙값에 세워 두고 계산했어요: {parts}.{extra}")
else:
    st.write("고른 속성이 두 개뿐이라 따로 세워 둔 속성은 없어요.")

st.write(
    boundary_text("가중치를 맞추기 전", "경계선", float(p_before.min()), float(p_before.max()), cross_before, "초록 점선")
)
st.write(
    boundary_text("가중치를 맞춘 뒤", "경계선", float(p_after.min()), float(p_after.max()), cross_after, "보라 실선")
)

# ---------- 5. 세 속성으로 보는 0.5 평면 ----------
st.subheader(f"5. 세 속성으로 보는 {LOGIT_NAME}의 0.5 평면")
if len(features) < 3:
    st.caption("입력 속성을 세 개 이상 고르면 마우스로 돌려 볼 수 있는 3차원 그림이 나타나요.")
else:
    a1, a2, a3 = st.columns(3)
    with a1:
        x3 = st.selectbox("가로축 (3차원)", features, format_func=KO.get)
    with a2:
        y3 = st.selectbox("세로축 (3차원)", [c for c in features if c != x3], format_func=KO.get)
    with a3:
        z3 = st.selectbox("깊이축 (3차원)", [c for c in features if c not in (x3, y3)], format_func=KO.get)

    axes = [x3, y3, z3]
    ranges = [axis_range(X_test[c], c) for c in axes]
    coefs_before, icpt_before = linear_form(logit_before, scaler, features)
    coefs_after, icpt_after = linear_form(logit_after, scaler, features)

    fig3d = go.Figure()
    rng3 = np.random.default_rng(0)
    for label, name, color, opacity in [
        (0, "실제: 아님", COLOR_NO, 0.4),
        (1, "실제: 뇌졸중", COLOR_YES, 0.9),
    ]:
        mask = (y_test == label).to_numpy()
        if not mask.any():
            continue
        xs, ys, zs = (X_test.loc[mask, c] for c in axes)
        fig3d.add_trace(
            go.Scatter3d(
                x=jitter(xs, x3, rng3),
                y=jitter(ys, y3, rng3),
                z=jitter(zs, z3, rng3),
                mode="markers",
                name=name,
                marker=dict(size=3, color=color, opacity=opacity),
                customdata=np.column_stack([xs.to_numpy(), ys.to_numpy(), zs.to_numpy()]),
                hovertemplate=(
                    f"{KO[x3]}: %{{customdata[0]}}<br>{KO[y3]}: %{{customdata[1]}}"
                    f"<br>{KO[z3]}: %{{customdata[2]}}<extra>{name}</extra>"
                ),
            )
        )

    plane_texts = []
    for label, coefs, icpt, color, color_name in [
        ("가중치를 맞추기 전", coefs_before, icpt_before, COLOR_BEFORE, "초록색"),
        ("가중치를 맞춘 뒤", coefs_after, icpt_after, COLOR_AFTER, "보라색"),
    ]:
        low, high = box_probability_range(coefs, icpt, fixed, axes, ranges, features)
        coords = plane_surface(coefs, icpt, fixed, axes, ranges, features) if low < 0.5 <= high else None
        if coords is not None:
            fig3d.add_trace(
                go.Surface(
                    x=coords[0],
                    y=coords[1],
                    z=coords[2],
                    colorscale=[[0, color], [1, color]],
                    showscale=False,
                    opacity=0.45,
                    hoverinfo="skip",
                )
            )
            # 범례 전용 표시
            fig3d.add_trace(
                go.Scatter3d(
                    x=[None],
                    y=[None],
                    z=[None],
                    mode="lines",
                    name=f"{label} 0.5 평면",
                    line=dict(color=color, width=8),
                )
            )
        plane_texts.append(boundary_text(label, "평면", low, high, coords is not None, color_name))

    fig3d.update_layout(
        height=720,
        scene=dict(
            xaxis=scene_axis(x3, ranges[0]),
            yaxis=scene_axis(y3, ranges[1]),
            zaxis=scene_axis(z3, ranges[2]),
            aspectmode="cube",
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0),
        margin=dict(t=40, l=0, r=0, b=0),
    )
    st.plotly_chart(fig3d)

    st.caption(
        f"점은 테스트 데이터 {len(test_df):,}명이고, 색은 실제 뇌졸중 여부예요. 마우스로 끌면 돌려 볼 수 있어요."
        + (" 고혈압·심장병처럼 0과 1만 있는 속성은 점이 겹치지 않게 그림에서만 살짝 흩어 찍었어요." if any(c in BINARY for c in axes) else "")
    )
    others_3d = [c for c in features if c not in axes]
    if others_3d:
        parts = ", ".join(f"{KO[c]} {fmt_num(fixed[c])}" for c in others_3d)
        extra = " (체질량지수는 빈 값을 채운 뒤의 값이에요.)" if "bmi" in others_3d else ""
        st.write(f"세 축이 아닌 속성은 테스트 데이터의 중앙값에 세워 두고 계산했어요: {parts}.{extra}")
    else:
        st.write("고른 속성이 세 개뿐이라 따로 세워 둔 속성은 없어요.")
    for text in plane_texts:
        st.write(text)
