import streamlit as st
import pandas as pd
import numpy as np
import os
import time
import joblib
import re
import warnings
warnings.filterwarnings("ignore")

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, f1_score
)
from sklearn.preprocessing import LabelEncoder
import plotly.graph_objects as go
import plotly.express as px

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TruthLens — Fake News Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS — LIGHT PURPLE THEME
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;800&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ── RESET & BASE ─────────────────────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [data-testid="stAppViewContainer"] {
    background: #f3eeff !important;
    font-family: 'DM Sans', sans-serif;
    color: #1a0a2e;
}

[data-testid="stAppViewContainer"] { padding: 0 !important; }
[data-testid="stHeader"] { display: none !important; }
[data-testid="stSidebar"] { display: none !important; }
section[data-testid="stMain"] { padding: 0 !important; }
.block-container { padding: 0 !important; max-width: 100% !important; }
footer { display: none !important; }
#MainMenu { display: none !important; }

/* ── SCROLLBAR ────────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #e8dcff; }
::-webkit-scrollbar-thumb { background: #9b6ee8; border-radius: 3px; }

/* ── SPLASH PAGE ──────────────────────────────────────────────────── */
.splash-wrap {
    min-height: 100vh;
    background: linear-gradient(135deg, #e8d5ff 0%, #d4b8ff 30%, #c9a6ff 60%, #b494f5 100%);
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    padding: 40px 20px; text-align: center;
    position: relative; overflow: hidden;
}
.splash-wrap::before {
    content: '';
    position: absolute; inset: 0;
    background:
        radial-gradient(circle at 20% 20%, rgba(255,255,255,0.35) 0%, transparent 50%),
        radial-gradient(circle at 80% 80%, rgba(120,60,220,0.20) 0%, transparent 50%);
    pointer-events: none;
}
.splash-eyebrow {
    font-family: 'DM Sans', sans-serif; font-weight: 500;
    font-size: 0.85rem; letter-spacing: 4px; text-transform: uppercase;
    color: #5c2d9e; background: rgba(255,255,255,0.50);
    padding: 6px 20px; border-radius: 30px; margin-bottom: 28px;
    border: 1px solid rgba(130,60,220,0.30);
}
.splash-title {
    font-family: 'Playfair Display', serif; font-weight: 800;
    font-size: clamp(3.5rem, 8vw, 7rem); line-height: 1.0;
    color: #2b0a5e;
    text-shadow: 0 4px 30px rgba(90,30,180,0.25);
    margin-bottom: 12px; position: relative; z-index: 1;
}
.splash-sub {
    font-family: 'DM Sans', sans-serif; font-size: clamp(1rem, 2vw, 1.35rem);
    color: #3d1572; max-width: 620px; line-height: 1.7;
    margin: 0 auto 40px; position: relative; z-index: 1;
}
.splash-cards {
    display: flex; gap: 18px; flex-wrap: wrap;
    justify-content: center; max-width: 780px; margin-bottom: 48px;
    position: relative; z-index: 1;
}
.splash-card {
    background: rgba(255,255,255,0.60); backdrop-filter: blur(10px);
    border: 1px solid rgba(160,100,255,0.35); border-radius: 16px;
    padding: 18px 22px; text-align: left; width: 220px;
}
.splash-card-icon { font-size: 1.8rem; margin-bottom: 8px; }
.splash-card-title { font-weight: 600; font-size: 0.95rem; color: #2b0a5e; margin-bottom: 4px; }
.splash-card-desc { font-size: 0.82rem; color: #5c2d9e; line-height: 1.5; }
.splash-btn-area { position: relative; z-index: 1; }
.decorative-orbs {
    position: absolute; inset: 0; pointer-events: none; overflow: hidden;
}
.orb {
    position: absolute; border-radius: 50%;
    background: radial-gradient(circle, rgba(200,160,255,0.4), transparent 70%);
    animation: float-orb 8s ease-in-out infinite;
}
.orb-1 { width: 300px; height: 300px; top: -80px; left: -80px; animation-delay: 0s; }
.orb-2 { width: 200px; height: 200px; bottom: 50px; right: -50px; animation-delay: 3s; }
.orb-3 { width: 150px; height: 150px; top: 40%; left: 60%; animation-delay: 5s; }
@keyframes float-orb {
    0%, 100% { transform: translate(0,0) scale(1); }
    50% { transform: translate(20px,-30px) scale(1.05); }
}

/* ── MAIN APP WRAPPER ─────────────────────────────────────────────── */
.app-wrap { min-height: 100vh; background: #f3eeff; }

/* ── TOP NAV ──────────────────────────────────────────────────────── */
.topnav {
    background: rgba(255,255,255,0.90); backdrop-filter: blur(14px);
    border-bottom: 1px solid #dac8ff;
    padding: 0 40px; display: flex; align-items: center;
    justify-content: space-between; height: 64px;
    position: sticky; top: 0; z-index: 100;
}
.nav-logo {
    font-family: 'Playfair Display', serif; font-weight: 800;
    font-size: 1.45rem; color: #2b0a5e; display: flex; align-items: center; gap: 8px;
}
.nav-logo span { color: #8b3cf7; }
.nav-pills { display: flex; gap: 6px; }
.nav-pill {
    font-size: 0.82rem; font-weight: 500; color: #5c2d9e;
    background: transparent; border: 1px solid #c4a4f5;
    padding: 5px 14px; border-radius: 20px; cursor: default;
}
.nav-pill.active { background: #8b3cf7; color: white; border-color: #8b3cf7; }

/* ── HERO BAR ─────────────────────────────────────────────────────── */
.hero-bar {
    background: linear-gradient(90deg, #7b2ff7 0%, #9b59f5 50%, #b07af8 100%);
    padding: 36px 40px; color: white;
    display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 20px;
}
.hero-title { font-family: 'Playfair Display', serif; font-size: clamp(1.6rem, 3vw, 2.4rem); font-weight: 800; line-height: 1.2; }
.hero-sub { font-size: 0.95rem; opacity: 0.88; margin-top: 6px; max-width: 500px; line-height: 1.5; }
.hero-stats { display: flex; gap: 28px; flex-wrap: wrap; }
.hstat { text-align: center; }
.hstat-num { font-family: 'Playfair Display', serif; font-size: 1.9rem; font-weight: 800; line-height: 1; }
.hstat-lbl { font-size: 0.72rem; opacity: 0.80; letter-spacing: 1px; text-transform: uppercase; margin-top: 2px; }

/* ── SECTION LAYOUT ───────────────────────────────────────────────── */
.content-area { padding: 32px 40px; }
.section-title {
    font-family: 'Playfair Display', serif; font-size: 1.55rem; font-weight: 700;
    color: #2b0a5e; margin-bottom: 20px; display: flex; align-items: center; gap: 10px;
}
.section-title::after {
    content: ''; flex: 1; height: 2px;
    background: linear-gradient(90deg, #c4a4f5, transparent);
}

/* ── CARDS ────────────────────────────────────────────────────────── */
.card {
    background: white; border-radius: 20px;
    border: 1px solid #e0ceff; padding: 24px;
    box-shadow: 0 4px 24px rgba(120,60,220,0.08);
}
.card-header {
    font-weight: 600; font-size: 1rem; color: #2b0a5e;
    margin-bottom: 16px; padding-bottom: 12px;
    border-bottom: 1px solid #f0e8ff;
    display: flex; align-items: center; gap: 8px;
}

/* ── STAT BOXES ───────────────────────────────────────────────────── */
.stat-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px,1fr)); gap: 16px; margin-bottom: 24px; }
.stat-box {
    background: white; border-radius: 16px; padding: 18px 20px;
    border: 1px solid #e0ceff;
    box-shadow: 0 2px 12px rgba(120,60,220,0.07);
}
.stat-box-val {
    font-family: 'Playfair Display', serif; font-size: 2rem; font-weight: 800;
    color: #7b2ff7; line-height: 1;
}
.stat-box-lbl { font-size: 0.78rem; color: #5c2d9e; margin-top: 4px; font-weight: 500; letter-spacing: 0.5px; }

/* ── PREDICT AREA ─────────────────────────────────────────────────── */
.predict-wrap { background: white; border-radius: 24px; border: 1.5px solid #c4a4f5; padding: 32px; box-shadow: 0 8px 40px rgba(120,60,220,0.12); }
.predict-label { font-weight: 600; color: #2b0a5e; font-size: 0.95rem; margin-bottom: 8px; }

/* ── RESULT BADGES ────────────────────────────────────────────────── */
.result-card {
    border-radius: 20px; padding: 28px 32px; margin-top: 24px;
    display: flex; align-items: center; gap: 24px;
    border: 2px solid; flex-wrap: wrap;
}
.result-real { background: #f0fff4; border-color: #48bb78; }
.result-fake { background: #fff5f5; border-color: #fc8181; }
.result-icon { font-size: 3rem; flex-shrink: 0; }
.result-label { font-family: 'Playfair Display', serif; font-size: 1.8rem; font-weight: 800; }
.result-real .result-label { color: #276749; }
.result-fake .result-label { color: #9b2335; }
.result-conf { font-size: 0.9rem; color: #555; margin-top: 6px; }
.result-detail { font-size: 0.88rem; color: #444; margin-top: 4px; }
.badge-grid { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 16px; }
.badge {
    font-size: 0.78rem; font-weight: 600; padding: 5px 14px; border-radius: 20px;
    border: 1.5px solid;
}
.badge-green { background: #d4f8e8; border-color: #48bb78; color: #276749; }
.badge-red { background: #ffe4e4; border-color: #fc8181; color: #9b2335; }
.badge-purple { background: #ede9ff; border-color: #9b6ee8; color: #4c1d95; }
.badge-orange { background: #fff3e0; border-color: #f6ad55; color: #7b341e; }

/* ── CONFIDENCE BAR ───────────────────────────────────────────────── */
.conf-track { background: #f0e8ff; border-radius: 8px; height: 10px; width: 100%; margin: 8px 0; overflow: hidden; }
.conf-fill { height: 100%; border-radius: 8px; transition: width 0.8s ease; }
.conf-row { display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px; }
.conf-lbl { font-size: 0.82rem; color: #3d1572; font-weight: 500; }
.conf-pct { font-size: 0.82rem; color: #7b2ff7; font-weight: 700; }

/* ── TEXTAREA ─────────────────────────────────────────────────────── */
.stTextArea textarea {
    background: #faf7ff !important;
    border: 1.5px solid #c4a4f5 !important;
    border-radius: 14px !important;
    color: #1a0a2e !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.95rem !important;
    padding: 14px !important;
}
.stTextArea textarea:focus {
    border-color: #8b3cf7 !important;
    box-shadow: 0 0 0 3px rgba(139,60,247,0.15) !important;
}
.stTextArea textarea::placeholder { color: #9b7ec8 !important; }
.stTextArea label { color: #2b0a5e !important; font-weight: 600 !important; font-size: 0.95rem !important; }

/* ── SELECT / INPUT ───────────────────────────────────────────────── */
.stSelectbox label { color: #2b0a5e !important; font-weight: 600 !important; font-size: 0.9rem !important; }
.stSelectbox > div > div {
    background: #faf7ff !important; border: 1.5px solid #c4a4f5 !important;
    border-radius: 12px !important; color: #1a0a2e !important;
}

/* ── BUTTONS ──────────────────────────────────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, #7b2ff7, #9b59f5) !important;
    color: white !important; border: none !important;
    border-radius: 12px !important; font-weight: 600 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.95rem !important; padding: 10px 28px !important;
    transition: all 0.2s ease !important; letter-spacing: 0.3px;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #6920e0, #8b3cf7) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(120,60,220,0.35) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* ── TAB STYLING ──────────────────────────────────────────────────── */
[data-testid="stTabs"] button {
    color: #5c2d9e !important; font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important; font-size: 0.9rem !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #7b2ff7 !important; font-weight: 700 !important;
    border-bottom-color: #7b2ff7 !important;
}
[data-testid="stTabsContent"] { padding-top: 0 !important; }

/* ── METRIC ───────────────────────────────────────────────────────── */
[data-testid="metric-container"] {
    background: white !important; border: 1px solid #e0ceff !important;
    border-radius: 16px !important; padding: 14px 18px !important;
}
[data-testid="metric-container"] label { color: #5c2d9e !important; font-size: 0.78rem !important; }
[data-testid="metric-container"] [data-testid="stMetricValue"] { color: #7b2ff7 !important; font-family: 'Playfair Display', serif !important; }

/* ── PROGRESS ─────────────────────────────────────────────────────── */
.stProgress > div > div { background: #8b3cf7 !important; border-radius: 6px !important; }
.stProgress > div { background: #e8dcff !important; border-radius: 6px !important; }

/* ── ALERTS ───────────────────────────────────────────────────────── */
.stAlert { border-radius: 14px !important; }
[data-testid="stInfo"] { background: #eef2ff !important; border-color: #9b6ee8 !important; color: #2b0a5e !important; }
[data-testid="stSuccess"] { background: #f0fff4 !important; color: #276749 !important; }
[data-testid="stWarning"] { background: #fffbeb !important; color: #7b341e !important; }
[data-testid="stError"] { background: #fff5f5 !important; color: #9b2335 !important; }

/* ── DATAFRAME ────────────────────────────────────────────────────── */
.stDataFrame { border-radius: 14px !important; overflow: hidden !important; }
[data-testid="stDataFrame"] th { background: #7b2ff7 !important; color: white !important; font-weight: 600 !important; }

/* ── DIVIDERS ─────────────────────────────────────────────────────── */
hr { border: none !important; border-top: 1px solid #e0ceff !important; margin: 24px 0 !important; }

/* ── SPINNER ──────────────────────────────────────────────────────── */
[data-testid="stSpinner"] > div > div { border-top-color: #8b3cf7 !important; }

/* ── TRAINING LOG ─────────────────────────────────────────────────── */
.train-log {
    background: #1a0a2e; border-radius: 14px; padding: 20px;
    font-family: 'Courier New', monospace; font-size: 0.82rem; color: #c4a4f5;
    max-height: 260px; overflow-y: auto; line-height: 1.7;
    border: 1px solid #4a1d9e;
}
.train-log .ok { color: #68d391; }
.train-log .info { color: #76e4f7; }
.train-log .warn { color: #f6ad55; }

/* ── ANALYSIS CHIP ────────────────────────────────────────────────── */
.chip-row { display: flex; flex-wrap: wrap; gap: 8px; margin: 12px 0; }
.chip {
    background: #ede9ff; border: 1px solid #c4a4f5;
    border-radius: 20px; padding: 4px 14px;
    font-size: 0.80rem; color: #4c1d95; font-weight: 500;
}

/* ── HISTORY TABLE ────────────────────────────────────────────────── */
.hist-row {
    display: flex; align-items: center; gap: 12px;
    padding: 10px 14px; border-radius: 12px;
    background: #faf7ff; border: 1px solid #e8dcff;
    margin-bottom: 8px;
}
.hist-lbl { flex: 1; font-size: 0.85rem; color: #2b0a5e; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.hist-tag { font-size: 0.75rem; font-weight: 700; padding: 3px 12px; border-radius: 12px; }
.hist-real-tag { background: #d4f8e8; color: #276749; }
.hist-fake-tag { background: #ffe4e4; color: #9b2335; }
.hist-conf-small { font-size: 0.75rem; color: #7b2ff7; font-weight: 600; min-width: 40px; text-align: right; }

/* ── LOADING SCREEN ───────────────────────────────────────────────── */
.loading-wrap {
    min-height: 100vh;
    background: linear-gradient(135deg, #e8d5ff, #d4b8ff, #c9a6ff);
    display: flex; flex-direction: column; align-items: center; justify-content: center;
}
.loading-spinner {
    width: 60px; height: 60px; border-radius: 50%;
    border: 6px solid rgba(255,255,255,0.3);
    border-top-color: #5c2d9e;
    animation: spin 0.9s linear infinite; margin-bottom: 24px;
}
@keyframes spin { to { transform: rotate(360deg); } }
.loading-text { font-family: 'DM Sans', sans-serif; font-size: 1.1rem; color: #3d1572; font-weight: 500; }

/* ── ABOUT SECTION ────────────────────────────────────────────────── */
.about-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px,1fr)); gap: 16px; }
.about-item {
    background: white; border-radius: 16px; padding: 20px;
    border: 1px solid #e0ceff;
}
.about-icon { font-size: 2rem; margin-bottom: 10px; }
.about-title { font-weight: 700; font-size: 0.95rem; color: #2b0a5e; margin-bottom: 6px; }
.about-text { font-size: 0.83rem; color: #5c2d9e; line-height: 1.55; }

/* ── RESPONSIVE ───────────────────────────────────────────────────── */
@media (max-width: 700px) {
    .content-area { padding: 20px 16px; }
    .hero-bar { padding: 24px 16px; }
    .topnav { padding: 0 16px; }
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
DATA_DIR   = "."   # change if data is elsewhere
MODEL_PATH = "truthlens_model.pkl"
TSV_COLS   = ['id','label','statement','subjects','speaker','job',
              'state','party','barely_true','false_c','half_true',
              'mostly_true','pants_fire','context']

LABEL_MAP = {
    "pants-fire": "fake", "false":       "fake",
    "barely-true":"fake", "half-true":   "fake",
    "mostly-true":"real", "true":        "real",
}

LABEL_COLORS = {
    "pants-fire": "#e53e3e", "false": "#fc8181",
    "barely-true":"#f6ad55", "half-true": "#ecc94b",
    "mostly-true":"#68d391", "true": "#38a169",
}

# ─────────────────────────────────────────────────────────────────────────────
# HELPER: TEXT FEATURES
# ─────────────────────────────────────────────────────────────────────────────
def preprocess(text: str) -> str:
    text = str(text).lower().strip()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text

def combine_features(row) -> str:
    parts = [
        str(row.get('statement', '')),
        str(row.get('speaker', '')),
        str(row.get('subjects', '')),
        str(row.get('party', '')),
        str(row.get('context', '')),
    ]
    return preprocess(" ".join(parts))

def text_analysis(text: str) -> dict:
    words = text.split()
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    caps_ratio   = sum(1 for w in words if w.isupper() and len(w) > 1) / max(len(words), 1)
    excl_count   = text.count('!')
    q_count      = text.count('?')
    num_count    = sum(1 for w in words if any(c.isdigit() for c in w))
    avg_word_len = np.mean([len(w) for w in words]) if words else 0

    hedging_words = ['maybe','perhaps','might','could','possibly','allegedly',
                     'reportedly','claimed','suggested','appears','seems']
    hedge_count  = sum(1 for w in words if w.lower() in hedging_words)

    clickbait    = ['shocking','unbelievable','you won\'t believe','exclusive','breaking',
                    'bombshell','secret','truth','exposed','revealed','conspiracy']
    cb_count     = sum(1 for p in clickbait if p in text.lower())

    return {
        "word_count":    len(words),
        "sentence_count":len(sentences),
        "caps_ratio":    round(caps_ratio, 3),
        "exclamations":  excl_count,
        "questions":     q_count,
        "numbers":       num_count,
        "avg_word_len":  round(avg_word_len, 2),
        "hedge_words":   hedge_count,
        "clickbait":     cb_count,
    }


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_split(path):
    df = pd.read_csv(path, sep='\t', header=None, names=TSV_COLS)
    df['binary'] = df['label'].map(LABEL_MAP)
    df.dropna(subset=['binary'], inplace=True)
    df['features'] = df.apply(combine_features, axis=1)
    return df

def load_data():
    train_paths = ["train.tsv", "data/train.tsv", "/mnt/user-data/uploads/train.tsv"]
    test_paths  = ["test.tsv",  "data/test.tsv",  "/mnt/user-data/uploads/test.tsv"]
    valid_paths = ["valid.tsv", "data/valid.tsv", "/mnt/user-data/uploads/valid.tsv"]

    train = test = valid = None
    for p in train_paths:
        if os.path.exists(p): train = load_split(p); break
    for p in test_paths:
        if os.path.exists(p): test  = load_split(p); break
    for p in valid_paths:
        if os.path.exists(p): valid = load_split(p); break

    # fallback: look inside zip
    if train is None:
        zip_path = "/mnt/user-data/uploads/archive__14_.zip"
        if os.path.exists(zip_path):
            import zipfile, io
            with zipfile.ZipFile(zip_path) as zf:
                for fname, attr in [("train.tsv","train"),("test.tsv","test"),("valid.tsv","valid")]:
                    if fname in zf.namelist():
                        with zf.open(fname) as f:
                            df = pd.read_csv(f, sep='\t', header=None, names=TSV_COLS)
                            df['binary'] = df['label'].map(LABEL_MAP)
                            df.dropna(subset=['binary'], inplace=True)
                            df['features'] = df.apply(combine_features, axis=1)
                            if attr == "train":   train = df
                            elif attr == "test":  test  = df
                            elif attr == "valid": valid = df
    return train, test, valid


# ─────────────────────────────────────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────────────────────────────────────
def build_pipeline():
    # Use two Logistic Regression classifiers so that
    # predict_proba() is always available (LinearSVC does NOT
    # support predict_proba, which caused the hard-voting
    # fallback to silently flip real↔fake labels).
    pipe_lr1 = Pipeline([
        ('tfidf', TfidfVectorizer(
            ngram_range=(1, 2), max_features=60000,
            sublinear_tf=True, min_df=2
        )),
        ('clf', LogisticRegression(
            C=2.5, max_iter=1000,
            class_weight='balanced', solver='lbfgs'
        ))
    ])
 
    pipe_lr2 = Pipeline([
        ('tfidf', TfidfVectorizer(
            ngram_range=(1, 2), max_features=60000,
            sublinear_tf=True, min_df=2
        )),
        ('clf', LogisticRegression(
            C=1.0, max_iter=1000,
            class_weight='balanced', solver='saga'
        ))
    ])
 
    voting = VotingClassifier(
        estimators=[('lr1', pipe_lr1), ('lr2', pipe_lr2)],
        voting='soft'          # ← KEY FIX: soft voting uses
                               #   predict_proba, never flips labels
    )
    return voting

def train_model(train_df, valid_df, log_container):
    def log(msg, cls="info"):
        log_container.markdown(
            f'<div class="train-log"><span class="{cls}">{msg}</span></div>',
            unsafe_allow_html=True
        )

    log("▶  Preparing training corpus …", "info")
    X_train, y_train = train_df['features'], train_df['binary']
    X_valid, y_valid = valid_df['features'], valid_df['binary']
    log(f"   Train: {len(X_train):,} samples | Valid: {len(X_valid):,} samples", "info")

    log("▶  Building ensemble (LR + SVC) …", "info")
    model = build_pipeline()

    log("▶  Fitting model — this may take ~30 s …", "warn")
    t0 = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - t0

    log(f"✔  Training complete in {elapsed:.1f}s", "ok")

    y_pred = model.predict(X_valid)
    acc = accuracy_score(y_valid, y_pred)
    f1  = f1_score(y_valid, y_pred, pos_label='real')
    log(f"✔  Validation Accuracy: {acc:.4f} | F1 (real): {f1:.4f}", "ok")

    joblib.dump({"model": model, "accuracy": acc, "f1": f1,
                 "train_size": len(X_train)}, MODEL_PATH)
    log(f"✔  Model saved → {MODEL_PATH}", "ok")

    return model, acc, f1

@st.cache_resource(show_spinner=False)
def load_model():
    if os.path.exists(MODEL_PATH):
        d = joblib.load(MODEL_PATH)
        return d["model"], d.get("accuracy", 0), d.get("f1", 0)
    return None, 0, 0


# ─────────────────────────────────────────────────────────────────────────────
# PREDICTION
# ─────────────────────────────────────────────────────────────────────────────
def predict(model, statement: str, speaker="", context="", party="", subjects=""):
    row = {
        "statement": statement,
        "speaker": speaker,
        "subjects": subjects,
        "party": party,
        "context": context
    }

    feat = combine_features(row)

    # STEP 1: Get probabilities
    try:
        proba = model.predict_proba([feat])[0]
        classes = list(model.classes_)
    except:
        # fallback if predict_proba not available
        label = model.predict([feat])[0]
        if label == 'real':
            prob_dict = {'real': 0.8, 'fake': 0.2}
        else:
            prob_dict = {'real': 0.2, 'fake': 0.8}

        conf = prob_dict[label]
        return label, conf, prob_dict

    # STEP 2: Fix class order
    if classes[0] == 'real':
        prob_dict = {
            'real': proba[0],
            'fake': proba[1]
        }
    else:
        prob_dict = {
            'fake': proba[0],
            'real': proba[1]
        }

    # STEP 3: Decide label AFTER prob_dict is created
    if prob_dict['real'] > prob_dict['fake']:
        label = 'real'
    else:
        label = 'fake'

    conf = prob_dict[label]

    return label, conf, prob_dict

# ─────────────────────────────────────────────────────────────────────────────
# PLOTLY CHART HELPERS
# ─────────────────────────────────────────────────────────────────────────────
CHART_TEMPLATE = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Sans, sans-serif", color="#2b0a5e"),
    margin=dict(l=20, r=20, t=30, b=20),
)

def label_dist_chart(df):
    vc = df['label'].value_counts().reset_index()
    vc.columns = ['label', 'count']
    colors = [LABEL_COLORS.get(l, "#9b6ee8") for l in vc['label']]
    fig = go.Figure(go.Bar(
        x=vc['label'], y=vc['count'],
        marker_color=colors,
        text=vc['count'], textposition='outside',
        textfont=dict(color="#2b0a5e", size=11)
    ))
    fig.update_layout(**CHART_TEMPLATE,
        title=dict(text="Label Distribution", x=0.5, font=dict(size=14, color="#2b0a5e")),
        xaxis=dict(showgrid=False, tickfont=dict(size=11)),
        yaxis=dict(showgrid=True, gridcolor="#e8dcff"),
        height=280
    )
    return fig

def party_dist_chart(df):
    top_parties = df['party'].value_counts().head(6).reset_index()
    top_parties.columns = ['party', 'count']
    PURPLES = ["#7b2ff7","#9b59f5","#b07af8","#c4a4f5","#dac8ff","#ede9ff"]
    fig = go.Figure(go.Pie(
        labels=top_parties['party'], values=top_parties['count'],
        hole=0.5, marker=dict(colors=PURPLES),
        textfont=dict(size=11, color="#2b0a5e")
    ))
    fig.update_layout(**CHART_TEMPLATE,
        title=dict(text="Party Distribution", x=0.5, font=dict(size=14, color="#2b0a5e")),
        height=280, showlegend=True,
        legend=dict(font=dict(size=10, color="#2b0a5e"))
    )
    return fig

def conf_matrix_chart(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=["real","fake"])
    fig = go.Figure(go.Heatmap(
        z=cm, x=["Pred: Real","Pred: Fake"], y=["True: Real","True: Fake"],
        text=cm, texttemplate="%{text}",
        colorscale=[[0,"#ede9ff"],[0.5,"#9b6ee8"],[1,"#4c1d95"]],
        showscale=False
    ))
    fig.update_layout(**CHART_TEMPLATE,
        title=dict(text="Confusion Matrix", x=0.5, font=dict(size=14, color="#2b0a5e")),
        xaxis=dict(tickfont=dict(color="#2b0a5e")),
        yaxis=dict(tickfont=dict(color="#2b0a5e")),
        height=280
    )
    return fig

def binary_dist_chart(df):
    vc = df['binary'].value_counts().reset_index()
    vc.columns = ['label', 'count']
    fig = go.Figure(go.Pie(
        labels=vc['label'], values=vc['count'],
        hole=0.6, marker=dict(colors=["#48bb78","#fc8181"]),
        textfont=dict(size=13, color="white")
    ))
    fig.update_layout(**CHART_TEMPLATE, height=260,
        title=dict(text="Real vs Fake Split", x=0.5, font=dict(size=14, color="#2b0a5e")),
        legend=dict(font=dict(color="#2b0a5e", size=11))
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────────────────────────────────────
if "page" not in st.session_state:
    st.session_state.page = "splash"
if "model" not in st.session_state:
    st.session_state.model = None
if "model_acc" not in st.session_state:
    st.session_state.model_acc = 0
if "model_f1" not in st.session_state:
    st.session_state.model_f1 = 0
if "history" not in st.session_state:
    st.session_state.history = []
if "train_df" not in st.session_state:
    st.session_state.train_df = None
if "test_df" not in st.session_state:
    st.session_state.test_df = None
if "valid_df" not in st.session_state:
    st.session_state.valid_df = None


# ─────────────────────────────────────────────────────────────────────────────
# ══════════════ SPLASH PAGE ══════════════
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.page == "splash":
    st.markdown("""
    <div class="splash-wrap">
      <div class="decorative-orbs">
        <div class="orb orb-1"></div>
        <div class="orb orb-2"></div>
        <div class="orb orb-3"></div>
      </div>
      <div class="splash-eyebrow">🔍 AI-Powered Analysis</div>
      <div class="splash-title">TruthLens</div>
      <p class="splash-sub">
        Advanced machine learning for political fact-checking &amp; misinformation detection.
        Trained on the LIAR benchmark dataset — 10,000+ real political statements.
      </p>
      <div class="splash-cards">
        <div class="splash-card">
          <div class="splash-card-icon">🧠</div>
          <div class="splash-card-title">Ensemble ML</div>
          <div class="splash-card-desc">Logistic Regression + SVM voting ensemble for high accuracy</div>
        </div>
        <div class="splash-card">
          <div class="splash-card-icon">💾</div>
          <div class="splash-card-title">Persistent Model</div>
          <div class="splash-card-desc">Train once, saves as .pkl — instant load on every restart</div>
        </div>
        <div class="splash-card">
          <div class="splash-card-icon">📊</div>
          <div class="splash-card-title">Deep Analytics</div>
          <div class="splash-card-desc">Confidence scores, confusion matrix, label breakdowns</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("🚀  Enter TruthLens", use_container_width=True):
            st.session_state.page = "app"
            st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# ══════════════ MAIN APP ══════════════
# ─────────────────────────────────────────────────────────────────────────────
elif st.session_state.page == "app":

    # ── Auto-load model if exists ──────────────────────────────────────────
    if st.session_state.model is None:
        m, acc, f1 = load_model()
        if m is not None:
            st.session_state.model     = m
            st.session_state.model_acc = acc
            st.session_state.model_f1  = f1

    # ── Top Nav ────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="topnav">
      <div class="nav-logo">🔍 Truth<span>Lens</span></div>
      <div class="nav-pills">
        <span class="nav-pill">LIAR Dataset</span>
        <span class="nav-pill active">Ensemble ML</span>
        <span class="nav-pill">TF-IDF Features</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Hero Bar ───────────────────────────────────────────────────────────
    model_status = "✅ Loaded" if st.session_state.model else "⚠️ Not Trained"


    st.markdown(f"""
    <div class="hero-bar">
      <div>
        <div class="hero-title">Fake News Detection Engine</div>
        <div class="hero-sub">Political misinformation analysis powered by TF-IDF + Voting Ensemble trained on LIAR benchmark data</div>
      </div>
      <div class="hero-stats">
        <div class="hstat"><div class="hstat-num">10K+</div><div class="hstat-lbl">Statements</div></div>
        <div class="hstat"><div class="hstat-num">{model_status}</div><div class="hstat-lbl">Model</div></div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Load data lazily ───────────────────────────────────────────────────
    if st.session_state.train_df is None:
        with st.spinner("Loading dataset …"):
            train, test, valid = load_data()
            st.session_state.train_df = train
            st.session_state.test_df  = test
            st.session_state.valid_df = valid

    train_df = st.session_state.train_df
    test_df  = st.session_state.test_df
    valid_df = st.session_state.valid_df

    # ── Main Tabs ──────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔮 Detect",
        "🏋️ Train Model",
        "📊 Analytics",
        "📈 Model Stats",
        "ℹ️ About"
    ])

    # ══════════════ TAB 1 — DETECT ══════════════
    with tab1:
        st.markdown('<div class="content-area">', unsafe_allow_html=True)

        if st.session_state.model is None:
            st.warning("⚠️ No model loaded. Go to **Train Model** tab first.")
        else:
            st.markdown('<div class="section-title">🔮 Analyze a Statement</div>', unsafe_allow_html=True)

            with st.container():
              

                statement = st.text_area(
                    "📝 Statement to Analyze",
                    placeholder="Paste any political statement, claim, or news headline here …",
                    height=110, key="stmt_input"
                )

                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    speaker = st.text_input("👤 Speaker (optional)", placeholder="e.g. Barack Obama")
                with col_b:
                    party = st.selectbox("🏛️ Party (optional)",
                        ["", "democrat", "republican", "none", "independent", "libertarian"])
                with col_c:
                    context = st.text_input("📍 Context (optional)", placeholder="e.g. Twitter, a debate")

                subjects = st.text_input("🏷️ Subjects / Topics (optional)",
                                          placeholder="e.g. economy, healthcare, immigration")

                col_btn1, col_btn2, _ = st.columns([1, 1, 3])
                with col_btn1:
                    run_btn = st.button("🔍 Analyze Statement", use_container_width=True)
                with col_btn2:
                    clear_btn = st.button("🗑️ Clear", use_container_width=True)

                st.markdown('</div>', unsafe_allow_html=True)

            if clear_btn:
                st.rerun()

            if run_btn:
                if not statement.strip():
                    st.error("Please enter a statement to analyze.")
                else:
                    with st.spinner("Analyzing …"):
                        label, conf, prob_dict = predict(
                            st.session_state.model,
                            statement, speaker, context, party, subjects
                        )

                    # Save to history
                    st.session_state.history.insert(0, {
                        "statement": statement[:100],
                        "label": label,
                        "confidence": conf,
                        "speaker": speaker or "—",
                        "party": party or "—"
                    })

                    # ── Result Card ───────────────────────────────────────
                    is_real  = label == "real"
                    cls_card = "result-real" if is_real else "result-fake"
                    icon     = "✅" if is_real else "❌"
                    verdict  = "LIKELY REAL" if is_real else "LIKELY FAKE"
                    conf_pct = f"{conf:.1%}"
                    desc_txt = ("High credibility — the claim aligns with patterns in verified statements."
                                if is_real else
                                "Low credibility — the claim matches misinformation patterns.")

                    st.markdown(f"""
                    <div class="result-card {cls_card}">
                      <div class="result-icon">{icon}</div>
                      <div style="flex:1">
                        <div class="result-label">{verdict}</div>
                        <div class="result-conf">Confidence: <strong>{conf_pct}</strong></div>
                        <div class="result-detail">{desc_txt}</div>
                      </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # ── Confidence Breakdown ──────────────────────────────
                    st.markdown("<br>", unsafe_allow_html=True)
                    c1, c2 = st.columns(2)

                    with c1:
                        st.markdown('<div class="card">', unsafe_allow_html=True)
                        st.markdown('<div class="card-header">📊 Class Probabilities</div>', unsafe_allow_html=True)
                        for cls_name, prob in sorted(prob_dict.items(), key=lambda x: -x[1]):
                            bar_color = "#48bb78" if cls_name == "real" else "#fc8181"
                            st.markdown(f"""
                            <div class="conf-row">
                              <span class="conf-lbl">{'✅ Real' if cls_name=='real' else '❌ Fake'}</span>
                              <span class="conf-pct">{prob:.1%}</span>
                            </div>
                            <div class="conf-track">
                              <div class="conf-fill" style="width:{prob*100:.1f}%;background:{bar_color}"></div>
                            </div>
                            """, unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)

                    with c2:
                        ana = text_analysis(statement)
                        st.markdown('<div class="card">', unsafe_allow_html=True)
                        st.markdown('<div class="card-header">🔬 Text Signals</div>', unsafe_allow_html=True)
                        st.markdown(f"""
                        <div class="chip-row">
                          <span class="chip">📝 {ana['word_count']} words</span>
                          <span class="chip">💬 {ana['sentence_count']} sentences</span>
                          <span class="chip">🔢 {ana['numbers']} numbers</span>
                          <span class="chip">❗ {ana['exclamations']} exclamations</span>
                          <span class="chip">🤔 {ana['hedge_words']} hedge words</span>
                          <span class="chip">🎯 {ana['clickbait']} clickbait signals</span>
                          <span class="chip">🔤 avg {ana['avg_word_len']} char/word</span>
                        </div>
                        """, unsafe_allow_html=True)
                        risk_items = []
                        if ana['caps_ratio'] > 0.10:
                            risk_items.append(('<span class="badge badge-red">⚠️ Excessive caps</span>'))
                        if ana['clickbait'] > 1:
                            risk_items.append('<span class="badge badge-red">⚠️ Clickbait language</span>')
                        if ana['exclamations'] > 2:
                            risk_items.append('<span class="badge badge-orange">⚠️ Many exclamations</span>')
                        if ana['hedge_words'] > 2:
                            risk_items.append('<span class="badge badge-purple">ℹ️ Hedging language</span>')
                        if not risk_items:
                            risk_items = ['<span class="badge badge-green">✅ No major red flags</span>']
                        st.markdown(f'<div class="badge-grid">{"".join(risk_items)}</div>', unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)

        # ── History ────────────────────────────────────────────────────────
        if st.session_state.history:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="section-title">🕑 Recent Predictions</div>', unsafe_allow_html=True)
            for h in st.session_state.history[:8]:
                tag_cls  = "hist-real-tag" if h['label'] == 'real' else "hist-fake-tag"
                tag_text = "REAL" if h['label'] == 'real' else "FAKE"
                conf_str = f"{h['confidence']:.0%}"
                st.markdown(f"""
                <div class="hist-row">
                  <div class="hist-lbl" title="{h['statement']}">{h['statement']}</div>
                  <span class="hist-tag {tag_cls}">{tag_text}</span>
                  <span class="hist-conf-small">{conf_str}</span>
                </div>
                """, unsafe_allow_html=True)
            if st.button("Clear History"):
                st.session_state.history = []
                st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)

    # ══════════════ TAB 2 — TRAIN ══════════════
    with tab2:
        st.markdown('<div class="content-area">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">🏋️ Model Training</div>', unsafe_allow_html=True)

        # Status
        if os.path.exists(MODEL_PATH):
            mtime = os.path.getmtime(MODEL_PATH)
            mtime_str = time.strftime("%b %d %Y, %H:%M", time.localtime(mtime))
            st.success(f"✅ Saved model found (`{MODEL_PATH}`) — last trained {mtime_str}. No retraining needed unless you want fresh weights.")
        else:
            st.info("ℹ️ No saved model found. Click **Train** to build and save the model.")

        cA, cB = st.columns(2)
        with cA:
            
            st.markdown('<div class="card-header">⚙️ Training Configuration</div>', unsafe_allow_html=True)
            if train_df is not None:
                st.markdown(f"""
                <div class="chip-row">
                  <span class="chip">🗂 Train: {len(train_df):,}</span>
                  <span class="chip">🧪 Valid: {len(valid_df):,}</span>
                  <span class="chip">📝 Test: {len(test_df):,}</span>
                </div>
                """, unsafe_allow_html=True)
            st.markdown("""
            <div style="font-size:0.85rem; color:#3d1572; margin-top:12px; line-height:1.7">
              <strong>Algorithm:</strong> Voting Ensemble (LR + LinearSVC)<br>
              <strong>Features:</strong> TF-IDF (1–2 grams, 60K vocab)<br>
              <strong>Balancing:</strong> class_weight='balanced'<br>
              <strong>Inputs:</strong> statement, speaker, party, context, subjects
            </div>
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with cB:
    
            st.markdown('<div class="card-header">📋 Training Log</div>', unsafe_allow_html=True)
            log_ph = st.empty()
            log_ph.markdown('<div class="train-log">Waiting to start …</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        col_t1, col_t2, _ = st.columns([1,1,3])
        with col_t1:
            retrain = st.button("🚂 Train / Retrain Model", use_container_width=True)
        with col_t2:
            if os.path.exists(MODEL_PATH):
                if st.button("🗑️ Delete Saved Model", use_container_width=True):
                    os.remove(MODEL_PATH)
                    st.session_state.model     = None
                    st.session_state.model_acc = 0
                    st.session_state.model_f1  = 0
                    load_model.clear()
                    st.success("Model deleted. Retrain to rebuild.")
                    st.rerun()

        if retrain:
            if train_df is None or valid_df is None:
                st.error("Dataset not loaded. Check file paths.")
            else:
                with st.spinner("Training in progress …"):
                    model, acc, f1 = train_model(train_df, valid_df, log_ph)
                st.session_state.model     = model
                st.session_state.model_acc = acc
                st.session_state.model_f1  = f1
                load_model.clear()
                st.success(f"✅ Training complete! Accuracy: {acc:.2%} | F1: {f1:.3f}")
                st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)

    # ══════════════ TAB 3 — ANALYTICS ══════════════
    with tab3:
        st.markdown('<div class="content-area">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">📊 Dataset Analytics</div>', unsafe_allow_html=True)

        if train_df is not None:
            # Stat boxes
            real_n = (train_df['binary'] == 'real').sum()
            fake_n = (train_df['binary'] == 'fake').sum()
            st.markdown(f"""
            <div class="stat-grid">
              <div class="stat-box"><div class="stat-box-val">{len(train_df):,}</div><div class="stat-box-lbl">Training Statements</div></div>
              <div class="stat-box"><div class="stat-box-val">{real_n:,}</div><div class="stat-box-lbl">Real Labels</div></div>
              <div class="stat-box"><div class="stat-box-val">{fake_n:,}</div><div class="stat-box-lbl">Fake Labels</div></div>
              <div class="stat-box"><div class="stat-box-val">{train_df['speaker'].nunique():,}</div><div class="stat-box-lbl">Unique Speakers</div></div>
              <div class="stat-box"><div class="stat-box-val">{train_df['party'].nunique()}</div><div class="stat-box-lbl">Parties</div></div>
            </div>
            """, unsafe_allow_html=True)

            r1c1, r1c2 = st.columns(2)
            with r1c1:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.plotly_chart(label_dist_chart(train_df), use_container_width=True, config={"displayModeBar": False})
                st.markdown('</div>', unsafe_allow_html=True)
            with r1c2:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.plotly_chart(binary_dist_chart(train_df), use_container_width=True, config={"displayModeBar": False})
                st.markdown('</div>', unsafe_allow_html=True)

            r2c1, r2c2 = st.columns(2)
            with r2c1:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.plotly_chart(party_dist_chart(train_df), use_container_width=True, config={"displayModeBar": False})
                st.markdown('</div>', unsafe_allow_html=True)
            with r2c2:
                # Statement length distribution
                train_df['stmt_len'] = train_df['statement'].apply(lambda x: len(str(x).split()))
                fig = px.histogram(train_df, x='stmt_len', nbins=40,
                                   color_discrete_sequence=["#8b3cf7"],
                                   labels={"stmt_len": "Word Count"})
                fig.update_layout(**CHART_TEMPLATE, height=280,
                    title=dict(text="Statement Length Distribution", x=0.5, font=dict(size=14, color="#2b0a5e")),
                    yaxis=dict(showgrid=True, gridcolor="#e8dcff"),
                    xaxis=dict(showgrid=False)
                )
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
                st.markdown('</div>', unsafe_allow_html=True)

            # Top speakers
            st.markdown('<div class="section-title">🏆 Top Speakers by Statement Count</div>', unsafe_allow_html=True)
            top_speakers = train_df['speaker'].value_counts().head(10).reset_index()
            top_speakers.columns = ['Speaker', 'Count']
            fig_spk = go.Figure(go.Bar(
                y=top_speakers['Speaker'], x=top_speakers['Count'],
                orientation='h',
                marker=dict(color="#8b3cf7"),
                text=top_speakers['Count'], textposition='outside',
                textfont=dict(color="#2b0a5e")
            ))
            fig_spk.update_layout(**CHART_TEMPLATE, height=340,
                title=dict(text="", x=0.5),
                yaxis=dict(autorange="reversed", tickfont=dict(size=11)),
                xaxis=dict(showgrid=True, gridcolor="#e8dcff")
            )
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.plotly_chart(fig_spk, use_container_width=True, config={"displayModeBar": False})
            st.markdown('</div>', unsafe_allow_html=True)

        else:
            st.warning("Dataset not loaded.")

        st.markdown('</div>', unsafe_allow_html=True)

    # ══════════════ TAB 4 — MODEL STATS ══════════════
    with tab4:
        st.markdown('<div class="content-area">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">📈 Model Evaluation</div>', unsafe_allow_html=True)

        if st.session_state.model is None:
            st.warning("Train the model first.")
        else:
            model = st.session_state.model

            # Stats
            st.markdown(f"""
            <div class="stat-grid">
              <div class="stat-box"><div class="stat-box-val">Ensemble</div><div class="stat-box-lbl">Algorithm</div></div>
              <div class="stat-box"><div class="stat-box-val">LR + SVC</div><div class="stat-box-lbl">Sub-models</div></div>
            </div>
            """, unsafe_allow_html=True)

            if valid_df is not None:
                with st.spinner("Evaluating on validation set …"):
                    X_val = valid_df['features']
                    y_val = valid_df['binary']
                    y_pred = model.predict(X_val)

                mc1, mc2 = st.columns(2)
                with mc1:
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.plotly_chart(conf_matrix_chart(y_val, y_pred), use_container_width=True, config={"displayModeBar": False})
                    st.markdown('</div>', unsafe_allow_html=True)
                with mc2:
                    report = classification_report(y_val, y_pred, output_dict=True)
                    rows = []
                    for cls in ['real', 'fake']:
                        r = report.get(cls, {})
                        rows.append({
                            "Class": cls.upper(),
                            "Precision": f"{r.get('precision',0):.3f}",
                            "Recall":    f"{r.get('recall',0):.3f}",
                            "F1-Score":  f"{r.get('f1-score',0):.3f}",
                            "Support":   int(r.get('support',0))
                        })
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.markdown('<div class="card-header">📋 Classification Report</div>', unsafe_allow_html=True)
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                    st.markdown('</div>', unsafe_allow_html=True)



            # Feature importance (LR coefficients)
            try:
                lr_pipe = model.estimators_[0][1]
                vocab   = lr_pipe.named_steps['tfidf'].vocabulary_
                coefs   = lr_pipe.named_steps['clf'].coef_[0]
                idx_to_word = {v: k for k, v in vocab.items()}
                top_real = np.argsort(coefs)[-15:][::-1]
                top_fake = np.argsort(coefs)[:15]

                st.markdown('<div class="section-title">🔑 Most Influential Words</div>', unsafe_allow_html=True)
                wc1, wc2 = st.columns(2)
                with wc1:
                    real_words = [idx_to_word.get(i, '') for i in top_real]
                    real_vals  = [coefs[i] for i in top_real]
                    fig_r = go.Figure(go.Bar(y=real_words, x=real_vals, orientation='h',
                                             marker_color="#48bb78",
                                             text=[f"+{v:.2f}" for v in real_vals],
                                             textposition='outside', textfont=dict(color="#276749")))
                    fig_r.update_layout(**CHART_TEMPLATE, height=320,
                        title=dict(text="✅ Top Real Signals", x=0.5, font=dict(size=13, color="#276749")),
                        yaxis=dict(autorange="reversed", tickfont=dict(size=10)),
                        xaxis=dict(showgrid=True, gridcolor="#e8dcff"))
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.plotly_chart(fig_r, use_container_width=True, config={"displayModeBar": False})
                    st.markdown('</div>', unsafe_allow_html=True)
                with wc2:
                    fake_words = [idx_to_word.get(i, '') for i in top_fake]
                    fake_vals  = [abs(coefs[i]) for i in top_fake]
                    fig_f = go.Figure(go.Bar(y=fake_words, x=fake_vals, orientation='h',
                                             marker_color="#fc8181",
                                             text=[f"-{v:.2f}" for v in fake_vals],
                                             textposition='outside', textfont=dict(color="#9b2335")))
                    fig_f.update_layout(**CHART_TEMPLATE, height=320,
                        title=dict(text="❌ Top Fake Signals", x=0.5, font=dict(size=13, color="#9b2335")),
                        yaxis=dict(autorange="reversed", tickfont=dict(size=10)),
                        xaxis=dict(showgrid=True, gridcolor="#e8dcff"))
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.plotly_chart(fig_f, use_container_width=True, config={"displayModeBar": False})
                    st.markdown('</div>', unsafe_allow_html=True)
            except Exception:
                st.info("Feature importance visualization unavailable for this model variant.")

        st.markdown('</div>', unsafe_allow_html=True)

    # ══════════════ TAB 5 — ABOUT ══════════════
    with tab5:
        st.markdown('<div class="content-area">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">ℹ️ About TruthLens</div>', unsafe_allow_html=True)

        st.markdown("""
        <div class="about-grid">
          <div class="about-item">
            <div class="about-icon">📚</div>
            <div class="about-title">LIAR Dataset</div>
            <div class="about-text">
              Trained on the LIAR benchmark by William Yang Wang (ACL 2017).
              10,240 labelled political statements from PolitiFact covering
              pants-fire, false, barely-true, half-true, mostly-true, and true.
            </div>
          </div>
          <div class="about-item">
            <div class="about-icon">🧠</div>
            <div class="about-title">Model Architecture</div>
            <div class="about-text">
              Voting ensemble combining Logistic Regression and LinearSVC.
              Features extracted via TF-IDF (unigrams + bigrams, 60K vocabulary)
              from statement, speaker, party, context, and subject fields.
            </div>
          </div>
          <div class="about-item">
            <div class="about-icon">💾</div>
            <div class="about-title">Persistent Storage</div>
            <div class="about-text">
              Model serialised with joblib to <code>truthlens_model.pkl</code>.
              On every subsequent launch, the saved weights are loaded instantly
              — no retraining required unless you explicitly request it.
            </div>
          </div>
          <div class="about-item">
            <div class="about-icon">🔬</div>
            <div class="about-title">Text Analysis</div>
            <div class="about-text">
              Beyond the ML prediction, every statement is analysed for clickbait
              signals, hedging language, excessive capitalisation, exclamation
              frequency, and numerical density — surfaced as visual chips.
            </div>
          </div>
          <div class="about-item">
            <div class="about-icon">⚖️</div>
            <div class="about-title">Binary Classification</div>
            <div class="about-text">
              The six PolitiFact labels are collapsed to binary:
              <em>real</em> (mostly-true + true) and
              <em>fake</em> (pants-fire + false + barely-true + half-true)
              for a practical two-class detector.
            </div>
          </div>
          <div class="about-item">
            <div class="about-icon">⚠️</div>
            <div class="about-title">Limitations</div>
            <div class="about-text">
              This tool is for educational and research purposes.
              It is trained on US political statements and may not generalise to
              other domains. Always cross-check with primary sources.
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div style="text-align:center; padding: 20px; color: #7b2ff7; font-size:0.85rem;">
          TruthLens · Built with Streamlit &amp; scikit-learn · LIAR Dataset (ACL 2017)
        </div>
        """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)
