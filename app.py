from pathlib import Path
import html
import random
import re

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from train_model import (
    DATA_DIR,
    MODEL_PATH,
    PRICE_CANDIDATES,
    clean_price,
    enrich_artwork_features,
    ensure_artist_public_info,
    ensure_data_available,
    filter_image_records,
    find_first_existing_column,
    merge_artist_public_info,
    normalize_artist_key,
    read_source_info,
    train_and_save_model,
    write_source_info,
)

st.set_page_config(page_title="你能打败 AI 艺术品估值师吗？", layout="wide")

DISPLAY_COLUMNS = ["description", "medium", "size", "dimensions", "category", "art_form", "material_used", "preservation_status", "cultural_heritage_level", "historical_score", "artist_reputation", "market_demand"]
FIELD_LABELS = {
    "description": "作品描述",
    "medium": "媒介",
    "size": "尺寸",
    "dimensions": "尺寸",
    "category": "类别",
    "art_form": "艺术形式",
    "material_used": "材料",
    "preservation_status": "保存状态",
    "cultural_heritage_level": "文化遗产级别",
    "historical_score": "历史评分",
    "artist_reputation": "艺术家声誉评分",
    "market_demand": "市场需求评分",
}


def normalize_column_name(name):
    return str(name).strip().lower().replace(" ", "_")


def format_money(value):
    if pd.isna(value):
        return "未知"
    return f"${float(value):,.0f}"


def format_year(value):
    try:
        if pd.isna(value):
            return "未知"
        return f"{float(value):.0f}"
    except (TypeError, ValueError):
        return "未知"


def clue_value(artwork, column, fallback="未知"):
    value = artwork.get(column)
    if pd.isna(value) or not str(value).strip():
        return fallback
    if column == "purchase_price":
        return format_money(clean_price(value))
    if column == "holding_period_years":
        return f"{float(value):.0f} 年"
    if column in {"artwork_year", "artist_birth_year", "artist_death_year"}:
        return format_year(value)
    return value


@st.cache_data
def load_artist_info_lookup():
    info = ensure_artist_public_info().copy()
    info["_artist_key"] = info["artist_name"].apply(normalize_artist_key)
    return info.set_index("_artist_key").to_dict(orient="index")


def artist_background_value(artwork, column):
    value = artwork.get(column)
    if pd.notna(value) and str(value).strip():
        return value
    artist_key = normalize_artist_key(artwork.get("artist_name") or artwork.get("artist"))
    return load_artist_info_lookup().get(artist_key, {}).get(column)


def artist_era_value(artwork):
    birth = clue_value(artwork, "artist_birth_year")
    death = clue_value(artwork, "artist_death_year", "在世或未知")
    if birth == "未知" and death == "在世或未知":
        return "未知"
    return f"{birth} - {death}"


def safe_float(value):
    try:
        if pd.isna(value):
            return np.nan
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def save_uploaded_csv(uploaded_file):
    DATA_DIR.mkdir(exist_ok=True)
    safe_name = re.sub(r"[^A-Za-z0-9_.-]", "_", uploaded_file.name)
    target = DATA_DIR / f"uploaded_{safe_name}"
    if MODEL_PATH.exists():
        MODEL_PATH.unlink()
    target.write_bytes(uploaded_file.getbuffer())
    write_source_info("uploaded_csv", f"正在使用你上传的 CSV 文件：{uploaded_file.name}。")
    st.cache_data.clear()
    st.cache_resource.clear()


@st.cache_data
def load_game_data():
    ensure_data_available()
    frames = []
    for csv_file in sorted(DATA_DIR.glob("*.csv")):
        if csv_file.name in {"artist_public_info.csv", "valid_image_urls.csv"}:
            continue
        frame = pd.read_csv(csv_file)
        frame.columns = [normalize_column_name(col) for col in frame.columns]
        frame["source_file"] = csv_file.name
        frames.append(frame)
    if not frames:
        return None, "data/ 文件夹中没有找到可用 CSV。"
    df = merge_artist_public_info(enrich_artwork_features(pd.concat(frames, ignore_index=True, sort=False)))
    price_col = find_first_existing_column(df.columns, PRICE_CANDIDATES)
    if price_col is None:
        return None, "CSV 中没有找到可用的成交价/价格列。"
    df["true_price"] = df[price_col].apply(clean_price)
    df = df.dropna(subset=["true_price"])
    df = df[df["true_price"] > 0].reset_index(drop=True)
    df = filter_image_records(df)
    if df.empty:
        return None, "没有找到可用于游戏展示的有图价格记录。"
    return df, None


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return train_and_save_model()
    return joblib.load(MODEL_PATH)


def choose_new_round(df):
    st.session_state.current_index = random.randint(0, len(df) - 1)
    st.session_state.round_finished = False
    st.session_state.last_result = None


def init_state(df):
    defaults = {"user_score": 0, "ai_score": 0, "rounds": 0, "current_index": None, "round_finished": False, "last_result": None}
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)
    if st.session_state.current_index is None and df is not None:
        choose_new_round(df)


def prepare_model_input(artwork, feature_columns):
    model_input = pd.DataFrame([artwork.reindex(feature_columns)])
    for col in ["purchase_price"]:
        if col in model_input.columns:
            model_input[col] = model_input[col].apply(clean_price)
    return model_input


def build_prediction_explanation(artwork, ai_prediction, df, model_artifact):
    purchase_price = clean_price(artwork.get("purchase_price"))
    holding_years = safe_float(artwork.get("holding_period_years"))
    artwork_year = safe_float(artwork.get("artwork_year"))
    artist_name = clue_value(artwork, "artist_name")
    artist_movement = artist_background_value(artwork, "artist_movement")
    median_price = float(df["true_price"].median())
    median_purchase = df["purchase_price"].apply(clean_price).dropna().median() if "purchase_price" in df.columns else np.nan
    median_holding = pd.to_numeric(df["holding_period_years"], errors="coerce").dropna().median() if "holding_period_years" in df.columns else np.nan
    reasons = []
    if pd.notna(purchase_price):
        ratio = ai_prediction / purchase_price if purchase_price > 0 else np.nan
        if pd.notna(median_purchase) and purchase_price > median_purchase:
            reasons.append(f"历史购入成本是 {format_money(purchase_price)}，高于样本中位数 {format_money(median_purchase)}，这是偏高价信号。")
        else:
            reasons.append(f"历史购入成本是 {format_money(purchase_price)}，模型会把它作为估价起点之一。")
        if pd.notna(ratio):
            reasons.append(f"AI 预测价约为购入成本的 {ratio:.1f} 倍，倍数来自相似记录中的综合模式。")
    if pd.notna(holding_years):
        if pd.notna(median_holding) and holding_years > median_holding:
            reasons.append(f"持有时间约 {holding_years:.0f} 年，高于样本中位数 {median_holding:.0f} 年，长期持有常伴随更大价格变化。")
        else:
            reasons.append(f"持有时间约 {holding_years:.0f} 年，模型会结合购入成本判断升值空间。")
    if pd.notna(artwork_year):
        reasons.append(f"作品年份约为 {artwork_year:.0f} 年，模型会把它和艺术家年代、作品描述一起比较。")
    if artist_name != "未知":
        movement_text = f"，相关背景为{artist_movement}" if pd.notna(artist_movement) and str(artist_movement).strip() else ""
        reasons.append(f"艺术家是 {artist_name}{movement_text}；模型会从该艺术家及相近记录的历史价格中学习模式。")
    direction = f"AI 预测高于样本成交价中位数 {format_money(median_price)}，说明线索整体偏向高价区间。" if ai_prediction > median_price else f"AI 预测低于样本成交价中位数 {format_money(median_price)}，说明线索整体更接近中低价区间。"
    return {"direction": direction, "reasons": reasons[:4], "model_name": model_artifact.get("model_name", "机器学习回归模型")}


st.markdown("""
<style>
.stApp {background: linear-gradient(135deg, #f5efe7 0%, #eef5f2 46%, #f5f2ff 100%);} 
.block-container {max-width: 1180px; padding-top: 2rem; padding-bottom: 3rem;}
.game-title,.round-box,.guess-panel {border:1px solid rgba(17,24,39,.14); border-radius:8px; background:rgba(255,255,255,.93); box-shadow:0 14px 34px rgba(17,24,39,.10); padding:1.15rem;}
.game-title {margin-bottom:1rem;} .game-title h1{margin:0;color:#111827;font-size:2.4rem;} .game-title p{color:#4b5563;}
.score-card{padding:.8rem 1rem;border-radius:8px;background:#111827;color:white;min-height:80px}.score-card .label{color:#d1d5db}.score-card .value{font-size:1.7rem;font-weight:800}
.source-strip{display:flex;gap:.55rem;flex-wrap:wrap;padding:.55rem .75rem;border:1px solid rgba(17,24,39,.10);border-radius:8px;background:rgba(255,255,255,.66);margin-bottom:1rem;font-size:.88rem}.source-strip small{color:#6b7280}
.art-title{font-size:1.5rem;font-weight:800;color:#111827}.art-artist{font-weight:700;color:#4b5563}.art-kicker{font-weight:800;color:#b45309;font-size:.82rem}
.field-row{padding:.55rem 0;border-bottom:1px solid #e5e7eb}.field-label{color:#6b7280;font-size:.86rem;font-weight:700}.field-value{color:#111827;line-height:1.45}
.clue-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:.65rem;margin:.9rem 0}.clue-item{background:#fff;border:1px solid #fed7aa;border-radius:8px;padding:.7rem}.clue-label{color:#9a3412;font-size:.78rem;font-weight:800}.clue-value{font-weight:800;color:#111827}.clue-help{color:#6b7280;font-size:.78rem;margin-top:.2rem}
.artist-info,.explain-box{margin-top:1rem;padding:1rem;border-radius:8px}.artist-info{background:#eff6ff;border:1px solid #bfdbfe}.explain-box{background:#eef2ff;border:1px solid #c7d2fe}.winner{border:1px solid #16a34a;border-radius:8px;padding:1rem;background:#f0fdf4;color:#14532d;font-weight:800}.ai-winner{border-color:#2563eb;background:#eff6ff;color:#1e3a8a}.tie-winner{border-color:#a16207;background:#fefce8;color:#713f12}
.stButton>button{border-radius:8px;border:1px solid #111827;background:#111827;color:#fff;font-weight:700}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="game-title">
  <div class="art-kicker">AI 艺术品估值小游戏</div>
  <h1>你能打败 AI 艺术品估值师吗？</h1>
  <p>每一轮你会看到一件艺术品的信息，猜它的成交价。AI 也会给出预测，谁离真实价格更近，谁就得分。本项目仅用于教学演示，不是专业艺术品估值工具。</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.subheader("数据设置")
    uploaded_file = st.file_uploader("上传艺术品拍卖 CSV", type=["csv"], help="请使用公开或自己整理的 CSV，并包含成交价/价格列。")
    if uploaded_file is not None:
        save_uploaded_csv(uploaded_file)
        st.success("CSV 已上传，模型会自动重新训练。")
        st.rerun()

model_artifact = load_model()
with st.sidebar.expander("模型信息", expanded=False):
    st.write(f"最佳模型：`{model_artifact.get('model_name', 'model')}`")
    st.write(f"训练集：{model_artifact.get('train_rows', '未知')} 条")
    st.write(f"测试集：{model_artifact.get('test_rows', '未知')} 条")
    st.write(f"MAE：{format_money(model_artifact.get('mae'))}")
    st.write(f"R²：{model_artifact.get('r2', 0):.3f}")

df, data_error = load_game_data()
if data_error:
    st.warning(data_error)
    st.stop()
init_state(df)
source_info = read_source_info()
source_label = {"kaggle": "Kaggle 公开数据", "uploaded_csv": "用户上传 CSV", "local_csv": "本地 CSV", "demo": "教学演示数据"}.get(source_info.get("source_type"), "未知数据")
st.markdown(f'<div class="source-strip"><span>当前数据来源</span><strong>{html.escape(source_label)}</strong><small>{html.escape(source_info.get("message", ""))}</small></div>', unsafe_allow_html=True)

score_a, score_b, score_c = st.columns(3)
score_a.markdown(f'<div class="score-card"><div class="label">你的得分</div><div class="value">{st.session_state.user_score}</div></div>', unsafe_allow_html=True)
score_b.markdown(f'<div class="score-card"><div class="label">AI 得分</div><div class="value">{st.session_state.ai_score}</div></div>', unsafe_allow_html=True)
score_c.markdown(f'<div class="score-card"><div class="label">已完成回合</div><div class="value">{st.session_state.rounds}</div></div>', unsafe_allow_html=True)

artwork = df.iloc[st.session_state.current_index]
feature_columns = model_artifact["feature_columns"]
price_col = model_artifact["price_column"]
art_col, play_col = st.columns([1.25, 1])

with art_col:
    artist_name = clue_value(artwork, "artist_name")
    title = clue_value(artwork, "title")
    st.markdown(f'<div class="round-box"><div class="art-kicker">本轮艺术品</div><div class="art-title">{html.escape(str(title))}</div><div class="art-artist">{html.escape(str(artist_name))}</div>', unsafe_allow_html=True)
    image_url = artwork.get("image_url")
    if pd.notna(image_url) and str(image_url).startswith("http"):
        st.image(str(image_url), width="stretch")
    for col in DISPLAY_COLUMNS:
        if col in df.columns and col != price_col:
            value = artwork.get(col)
            if pd.notna(value) and str(value).strip():
                st.markdown(f'<div class="field-row"><div class="field-label">{FIELD_LABELS.get(col, col)}</div><div class="field-value">{html.escape(str(value))}</div></div>', unsafe_allow_html=True)
    artist_summary = artist_background_value(artwork, "artist_summary")
    artist_country = artist_background_value(artwork, "artist_country")
    artist_movement = artist_background_value(artwork, "artist_movement")
    artist_source = artist_background_value(artwork, "artist_source")
    if any(pd.notna(v) and str(v).strip() for v in [artist_summary, artist_country, artist_movement]):
        source_link = f'<p><a href="{html.escape(str(artist_source))}" target="_blank">查看公开来源</a></p>' if pd.notna(artist_source) and str(artist_source).startswith("http") else ""
        st.markdown(f'<div class="artist-info"><strong>艺术家背景</strong><p><b>国家/地区：</b>{html.escape(str(artist_country)) if pd.notna(artist_country) else "未知"}</p><p><b>艺术流派：</b>{html.escape(str(artist_movement)) if pd.notna(artist_movement) else "未知"}</p><p><b>简介：</b>{html.escape(str(artist_summary)) if pd.notna(artist_summary) else "暂无"}</p>{source_link}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with play_col:
    st.markdown('<div class="guess-panel">', unsafe_allow_html=True)
    st.subheader("你的估价")
    st.caption("真实成交价先保密。输入你的猜测，然后看看能不能比 AI 更接近。")
    st.markdown(f'''
    <div class="clue-grid">
      <div class="clue-item"><div class="clue-label">购入成本</div><div class="clue-value">{html.escape(str(clue_value(artwork, "purchase_price")))}</div><div class="clue-help">上一笔可见购入记录</div></div>
      <div class="clue-item"><div class="clue-label">持有时间</div><div class="clue-value">{html.escape(str(clue_value(artwork, "holding_period_years")))}</div><div class="clue-help">从购入到再出售的大致年数</div></div>
      <div class="clue-item"><div class="clue-label">作品年份</div><div class="clue-value">{html.escape(str(clue_value(artwork, "artwork_year")))}</div><div class="clue-help">从作品链接中解析</div></div>
      <div class="clue-item"><div class="clue-label">艺术家年代</div><div class="clue-value">{html.escape(str(artist_era_value(artwork)))}</div><div class="clue-help">来自艺术家字段</div></div>
    </div>
    ''', unsafe_allow_html=True)
    guess = st.number_input("你认为这件作品成交价是多少？", min_value=0.0, value=1_000_000.0, step=100_000.0, format="%.0f")
    submit = st.button("提交估价", disabled=st.session_state.round_finished)
    st.caption("价格单位按美元显示，因为当前 Kaggle 数据以美元记录。")
    st.markdown('</div>', unsafe_allow_html=True)

if submit:
    model_input = prepare_model_input(artwork, feature_columns)
    ai_prediction = max(0, float(model_artifact["model"].predict(model_input)[0]))
    true_price = float(artwork["true_price"])
    user_error = abs(guess - true_price)
    ai_error = abs(ai_prediction - true_price)
    if user_error < ai_error:
        winner, winner_class = "你赢了这一轮！", "winner"
        st.session_state.user_score += 1
    elif ai_error < user_error:
        winner, winner_class = "AI 这一轮更接近。", "winner ai-winner"
        st.session_state.ai_score += 1
    else:
        winner, winner_class = "这一轮平局。", "winner tie-winner"
    st.session_state.rounds += 1
    st.session_state.round_finished = True
    st.session_state.last_result = {"guess": guess, "ai_prediction": ai_prediction, "true_price": true_price, "user_error": user_error, "ai_error": ai_error, "winner": winner, "winner_class": winner_class, "explanation": build_prediction_explanation(artwork, ai_prediction, df, model_artifact)}

if st.session_state.last_result:
    result = st.session_state.last_result
    with play_col:
        st.subheader("揭晓结果")
        c1, c2 = st.columns(2)
        c1.metric("你的估价", format_money(result["guess"]))
        c2.metric("AI 预测", format_money(result["ai_prediction"]))
        st.metric("真实成交价", format_money(result["true_price"]))
        c3, c4 = st.columns(2)
        c3.metric("你的误差", format_money(result["user_error"]))
        c4.metric("AI 误差", format_money(result["ai_error"]))
        st.markdown(f'<div class="{result.get("winner_class", "winner")}">{result["winner"]}</div>', unsafe_allow_html=True)
        explanation = result.get("explanation", {})
        reason_items = "".join(f"<li>{html.escape(str(reason))}</li>" for reason in explanation.get("reasons", []))
        st.markdown(f'<div class="explain-box"><h4>AI 预测逻辑</h4><p>{html.escape(str(explanation.get("direction", "")))}</p><ul>{reason_items}</ul><p><b>模型：</b>{html.escape(str(explanation.get("model_name", "机器学习回归模型")))}</p><p>说明：这不是人工写死的公式，而是模型从训练集中学习到的综合模式。</p></div>', unsafe_allow_html=True)

if st.button("下一件艺术品"):
    choose_new_round(df)
    st.rerun()
