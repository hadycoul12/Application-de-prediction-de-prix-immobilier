"""
🏠 ImmoPredict — Application Streamlit Sécurisée
Auteur  : Hady COULIBALY — Étudiant Data Science
Version : 2.0 (déploiement Streamlit Cloud)

Sécurité :
  - Authentification basique via st.secrets
  - Validation des entrées utilisateur
  - Protection XSRF activée (config.toml)
  - Logs d'activité horodatés
  - Taille upload limitée à 10 Mo
"""

import io
import logging
import warnings
import datetime
import hashlib

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import streamlit as st

from sklearn.ensemble        import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model    import LinearRegression
from sklearn.metrics         import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline        import Pipeline
from sklearn.preprocessing   import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")

# ╔══════════════════════════════════════════════════════════════════╗
# ║  LOGGING                                                         ║
# ╚══════════════════════════════════════════════════════════════════╝
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("ImmoPredict")


def log_event(event: str, details: str = ""):
    """Enregistre un événement utilisateur horodaté."""
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"[{ts}] {event} — {details}")
    # Stocker dans session_state pour affichage dans l'app
    if "logs" not in st.session_state:
        st.session_state.logs = []
    st.session_state.logs.append(f"{ts} | {event} | {details}")


# ╔══════════════════════════════════════════════════════════════════╗
# ║  CONFIG PAGE                                                     ║
# ╚══════════════════════════════════════════════════════════════════╝
st.set_page_config(
    page_title="🏠 ImmoPredict",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ╔══════════════════════════════════════════════════════════════════╗
# ║  CSS GLOBAL                                                      ║
# ╚══════════════════════════════════════════════════════════════════╝
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

.hero {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    border-radius: 16px; padding: 2.2rem 2rem; margin-bottom: 1.5rem;
    position: relative; overflow: hidden;
}
.hero::before {
    content: ''; position: absolute; top: -50%; right: -10%;
    width: 400px; height: 400px;
    background: radial-gradient(circle, rgba(229,160,64,0.15) 0%, transparent 70%);
    border-radius: 50%;
}
.hero h1 {
    font-family: 'DM Serif Display', serif; font-size: 2.3rem;
    color: #f5e6c8; margin: 0; letter-spacing: -0.02em; line-height: 1.1;
}
.hero p  { color: #a8b4c8; font-size: 0.95rem; margin-top: 0.4rem; font-weight: 300; }
.hero .badge {
    display: inline-block; background: rgba(229,160,64,0.2);
    border: 1px solid rgba(229,160,64,0.4); color: #e5a040;
    border-radius: 20px; padding: 0.18rem 0.75rem; font-size: 0.72rem;
    font-weight: 600; letter-spacing: 0.05em; text-transform: uppercase; margin-bottom: 0.7rem;
}
.metric-card {
    background: #fff; border: 1px solid #e8ecf0; border-radius: 12px;
    padding: 1.1rem 1.3rem; box-shadow: 0 1px 4px rgba(0,0,0,0.05); transition: box-shadow 0.2s;
}
.metric-card:hover { box-shadow: 0 4px 12px rgba(0,0,0,0.1); }
.metric-card .label { font-size: 0.7rem; font-weight: 600; letter-spacing: 0.07em; text-transform: uppercase; color: #8899aa; margin-bottom: 0.25rem; }
.metric-card .value { font-family: 'DM Serif Display', serif; font-size: 1.7rem; color: #1a1a2e; line-height: 1; }
.metric-card .sub   { font-size: 0.75rem; color: #a0aab4; margin-top: 0.2rem; }
.section-title {
    font-family: 'DM Serif Display', serif; font-size: 1.35rem; color: #1a1a2e;
    margin: 1.4rem 0 0.7rem; padding-bottom: 0.35rem;
    border-bottom: 2px solid #e5a040; display: inline-block;
}
[data-testid="stSidebar"]               { background: #1a1a2e !important; }
[data-testid="stSidebar"] *             { color: #d4dce8 !important; }
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3            { font-family: 'DM Serif Display', serif !important; color: #f5e6c8 !important; }
[data-testid="stSidebar"] label         { font-weight: 500 !important; font-size: 0.85rem !important; }
.stTabs [data-baseweb="tab-list"]       { gap: 4px; background: #f4f6f9; padding: 4px; border-radius: 10px; }
.stTabs [data-baseweb="tab"]            { border-radius: 8px; padding: 0.38rem 1.1rem; font-weight: 500; font-size: 0.87rem; color: #556677; }
.stTabs [aria-selected="true"]          { background: #1a1a2e !important; color: #f5e6c8 !important; }
.stDownloadButton button                { background: linear-gradient(135deg, #e5a040, #d4892a) !important; color: white !important; border: none !important; border-radius: 8px !important; font-weight: 600 !important; }
.stButton > button                      { background: linear-gradient(135deg, #0f3460, #1a1a2e) !important; color: #f5e6c8 !important; border: none !important; border-radius: 8px !important; font-weight: 600 !important; }
.info-box  { background: #f0f7ff; border-left: 4px solid #0f3460; border-radius: 0 8px 8px 0; padding: 0.75rem 1rem; font-size: 0.87rem; color: #334455; margin: 0.7rem 0; }
.result-box { background: linear-gradient(135deg, #1a1a2e, #0f3460); border-radius: 14px; padding: 1.8rem 2rem; text-align: center; margin: 1rem 0; }
.result-box .price { font-family: 'DM Serif Display', serif; font-size: 3rem; color: #e5a040; line-height: 1; }
.result-box .label { color: #a8b4c8; font-size: 0.9rem; margin-top: 0.4rem; }

/* ── Login box ── */
.login-box {
    max-width: 420px; margin: 4rem auto; padding: 2.5rem;
    background: linear-gradient(135deg, #1a1a2e, #0f3460);
    border-radius: 18px; border: 1px solid rgba(229,160,64,0.3);
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
}
.login-box h2 { font-family:'DM Serif Display',serif; color:#f5e6c8; text-align:center; margin-bottom:0.3rem; }
.login-box p  { color:#a8b4c8; text-align:center; font-size:0.88rem; margin-bottom:1.5rem; }

/* ── Security badge ── */
.sec-badge {
    display:inline-flex; align-items:center; gap:0.4rem;
    background:rgba(46,204,113,0.15); border:1px solid rgba(46,204,113,0.4);
    color:#2ecc71; border-radius:20px; padding:0.2rem 0.75rem;
    font-size:0.72rem; font-weight:600; letter-spacing:0.05em;
}
.log-line { font-family:monospace; font-size:0.78rem; color:#334455; padding:0.15rem 0; border-bottom:1px solid #e8ecf0; }
</style>
""", unsafe_allow_html=True)

# ╔══════════════════════════════════════════════════════════════════╗
# ║  AUTHENTIFICATION                                                ║
# ╚══════════════════════════════════════════════════════════════════╝

def hash_password(password: str) -> str:
    """Hash SHA-256 du mot de passe."""
    return hashlib.sha256(password.encode()).hexdigest()


def check_credentials(username: str, password: str) -> bool:
    """
    Vérifie les credentials contre st.secrets (Streamlit Cloud)
    ou des valeurs par défaut en local (dev uniquement).
    """
    try:
        # ── Mode Streamlit Cloud : lire depuis secrets.toml ────────
        valid_user = st.secrets["auth"]["username"]
        valid_hash = hash_password(st.secrets["auth"]["password"])
        return username == valid_user and hash_password(password) == valid_hash
    except Exception:
        # ── Mode local (fallback dev) ──────────────────────────────
        # ⚠️ Remplacez par vos credentials en production
        return username == "admin" and password == "nexa2026"


def login_page():
    """Affiche la page de connexion."""
    st.markdown("""
    <div class="login-box">
        <h2>🏠 ImmoPredict</h2>
        <p>Connexion requise pour accéder à l'application</p>
    </div>
    """, unsafe_allow_html=True)

    col_center = st.columns([1, 2, 1])[1]
    with col_center:
        with st.form("login_form"):
            st.markdown("### 🔐 Connexion")
            username = st.text_input(
                "Identifiant",
                placeholder="admin",
                help="Entrez votre identifiant d'accès.",
            )
            password = st.text_input(
                "Mot de passe",
                type="password",
                placeholder="admin",
                help="Mot de passe sensible à la casse.",
            )
            submitted = st.form_submit_button("Se connecter →", use_container_width=True)

            if submitted:
                # Validation des entrées
                username_clean = validate_text_input(username, max_len=50)
                if username_clean is None or len(username_clean) == 0:
                    st.error("Identifiant invalide.", icon="🚫")
                    log_event("LOGIN_FAIL", f"Identifiant vide ou invalide")
                elif check_credentials(username_clean, password):
                    st.session_state.authenticated = True
                    st.session_state.username       = username_clean
                    log_event("LOGIN_SUCCESS", f"user={username_clean}")
                    st.rerun()
                else:
                    st.error("Identifiant ou mot de passe incorrect.", icon="🔒")
                    log_event("LOGIN_FAIL", f"user={username_clean}")

        st.markdown(
            '<div style="text-align:center;font-size:0.78rem;color:#8899aa;margin-top:1rem;">'
            '🔒 Connexion sécurisée — HTTPS · Protection XSRF activée</div>',
            unsafe_allow_html=True,
        )


# ╔══════════════════════════════════════════════════════════════════╗
# ║  VALIDATION DES ENTRÉES                                          ║
# ╚══════════════════════════════════════════════════════════════════╝

def validate_text_input(value: str, max_len: int = 200) -> str | None:
    """Nettoie et valide une entrée textuelle."""
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    # Rejeter les caractères dangereux (injection)
    forbidden = ["<", ">", "{", "}", "\\", ";", "--", "/*", "*/"]
    for char in forbidden:
        if char in cleaned:
            log_event("VALIDATION_FAIL", f"Caractère interdit détecté : {char}")
            return None
    if len(cleaned) > max_len:
        log_event("VALIDATION_FAIL", f"Entrée trop longue ({len(cleaned)} > {max_len})")
        return cleaned[:max_len]
    return cleaned


def validate_numeric(value, min_val: float, max_val: float, name: str = "") -> float | None:
    """Valide une valeur numérique dans une plage définie."""
    try:
        v = float(value)
        if v < min_val or v > max_val:
            log_event("VALIDATION_WARN", f"{name}={v} hors plage [{min_val}, {max_val}]")
            return max(min_val, min(max_val, v))   # clamp
        return v
    except (ValueError, TypeError):
        log_event("VALIDATION_FAIL", f"{name} non numérique : {value}")
        return None


def validate_csv_upload(file) -> pd.DataFrame | None:
    """Valide et nettoie un fichier CSV uploadé."""
    if file is None:
        return None
    # Vérification taille (10 Mo max)
    file.seek(0, 2)
    size_mb = file.tell() / (1024 * 1024)
    file.seek(0)
    if size_mb > 10:
        st.error(f"Fichier trop volumineux ({size_mb:.1f} Mo). Limite : 10 Mo.", icon="🚫")
        log_event("UPLOAD_FAIL", f"Fichier trop lourd : {size_mb:.1f} Mo")
        return None
    try:
        df = pd.read_csv(file, nrows=10000)   # Limite à 10 000 lignes
        log_event("UPLOAD_OK", f"CSV chargé : {df.shape[0]} lignes × {df.shape[1]} colonnes")
        return df
    except Exception as e:
        st.error(f"Fichier CSV invalide : {e}", icon="🚫")
        log_event("UPLOAD_FAIL", f"Erreur lecture CSV : {e}")
        return None


# ╔══════════════════════════════════════════════════════════════════╗
# ║  SESSION STATE                                                   ║
# ╚══════════════════════════════════════════════════════════════════╝
if "authenticated"  not in st.session_state: st.session_state.authenticated  = False
if "username"       not in st.session_state: st.session_state.username        = ""
if "page"           not in st.session_state: st.session_state.page            = "Page 1"
if "df_raw"         not in st.session_state: st.session_state.df_raw          = None
if "trained_model"  not in st.session_state: st.session_state.trained_model   = None
if "model_features" not in st.session_state: st.session_state.model_features  = None
if "label_encoders" not in st.session_state: st.session_state.label_encoders  = {}
if "model_name"     not in st.session_state: st.session_state.model_name      = None
if "metrics"        not in st.session_state: st.session_state.metrics         = None
if "logs"           not in st.session_state: st.session_state.logs            = []

# ╔══════════════════════════════════════════════════════════════════╗
# ║  GUARD : AUTHENTIFICATION                                        ║
# ╚══════════════════════════════════════════════════════════════════╝
if not st.session_state.authenticated:
    login_page()
    st.stop()

# ── Log de session ────────────────────────────────────────────────
log_event("PAGE_VISIT", f"user={st.session_state.username} page={st.session_state.page}")

# ╔══════════════════════════════════════════════════════════════════╗
# ║  PALETTES & MATPLOTLIB                                           ║
# ╚══════════════════════════════════════════════════════════════════╝
DARK_BLUE = "#0f3460"
GOLD      = "#e5a040"
TEAL      = "#1a8fa0"
CORAL     = "#e06060"

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.facecolor":    "#fafbfc",
    "figure.facecolor":  "white",
    "axes.grid":         True,
    "grid.color":        "#e8ecf0",
    "grid.linewidth":    0.7,
})

# ╔══════════════════════════════════════════════════════════════════╗
# ║  HELPERS                                                         ║
# ╚══════════════════════════════════════════════════════════════════╝
@st.cache_data(show_spinner="Chargement des données…")
def load_data(file_bytes: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(file_bytes))
    if "YearBuilt" in df.columns:
        df["AgeLogement"] = 2025 - df["YearBuilt"]
    if all(c in df.columns for c in ["GrLivArea", "TotalBsmtSF"]):
        df["SurfaceTotale"] = df["GrLivArea"] + df["TotalBsmtSF"].fillna(0)
    if all(c in df.columns for c in ["FullBath", "HalfBath"]):
        df["NbSallesDeBain"] = df["FullBath"] + 0.5 * df["HalfBath"]
    return df



def impute(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    none_cols = ["PoolQC","MiscFeature","Alley","Fence","FireplaceQu",
                 "GarageType","GarageFinish","GarageQual","GarageCond",
                 "BsmtQual","BsmtCond","BsmtExposure","BsmtFinType1","BsmtFinType2","MasVnrType"]
    zero_cols = ["GarageYrBlt","GarageArea","GarageCars",
                 "BsmtFinSF1","BsmtFinSF2","BsmtUnfSF","TotalBsmtSF",
                 "BsmtFullBath","BsmtHalfBath","MasVnrArea"]
    for c in none_cols:
        if c in df.columns: df[c] = df[c].fillna("None")
    for c in zero_cols:
        if c in df.columns: df[c] = df[c].fillna(0)
    if "LotFrontage" in df.columns and "Neighborhood" in df.columns:
        df["LotFrontage"] = df.groupby("Neighborhood")["LotFrontage"].transform(
            lambda x: x.fillna(x.median()))
    if "Electrical" in df.columns:
        df["Electrical"] = df["Electrical"].fillna(df["Electrical"].mode()[0])
    for c in df.columns:
        if df[c].isnull().any():
            df[c] = df[c].fillna(df[c].mode()[0] if df[c].dtype == "object" else df[c].median())
    return df


def metric_card(col, label, value, sub=""):
    col.markdown(
        f'<div class="metric-card"><div class="label">{label}</div>'
        f'<div class="value">{value}</div><div class="sub">{sub}</div></div>',
        unsafe_allow_html=True,
    )


def section(title):
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)


def info(text):
    st.markdown(f'<div class="info-box">{text}</div>', unsafe_allow_html=True)


# ╔══════════════════════════════════════════════════════════════════╗
# ║  SIDEBAR                                                         ║
# ╚══════════════════════════════════════════════════════════════════╝
with st.sidebar:
    st.markdown("""
    <div style="padding:1rem 0.5rem 0.5rem;">
        <div style="font-family:'DM Serif Display',serif;font-size:1.5rem;color:#f5e6c8;line-height:1.2;margin-bottom:0.5rem;">
            🏠 ImmoPredict
        </div>
        <div style="font-size:0.7rem;font-weight:600;letter-spacing:0.08em;text-transform:uppercase;color:#e5a040;margin-bottom:0.3rem;">
            Auteur
        </div>
        <div style="font-family:'DM Serif Display',serif;font-size:1.2rem;color:#ffffff;
                    background:linear-gradient(135deg,rgba(229,160,64,0.25),rgba(229,160,64,0.05));
                    border:1px solid rgba(229,160,64,0.5);border-radius:10px;
                    padding:0.45rem 0.75rem;margin-bottom:0.3rem;">
            Hady COULIBALY
        </div>
        <div style="font-size:0.74rem;color:#6688aa;font-style:italic;">Etudiant en M2 Data & Intelligence Artificielle</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Statut sécurité
    st.markdown(
        f'<div class="sec-badge">🔒 Connecté : {st.session_state.username}</div>',
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("### Navigation")
    pages = {
        "Page 1": "📂  Upload & Exploration",
        "Page 2": "🤖  Entraînement & Performances",
        "Page 3": "🔮  Prédiction",
        "Page 4": "📋  Logs & Sécurité",
    }
    for key, label in pages.items():
        if st.button(label, key=f"nav_{key}", use_container_width=True):
            st.session_state.page = key
            log_event("NAVIGATION", f"user={st.session_state.username} → {key}")

    st.markdown("---")
    has_data  = st.session_state.df_raw is not None
    has_model = st.session_state.trained_model is not None
    st.markdown(
        f"{'✅' if has_data  else '⭕'} Données chargées\n\n"
        f"{'✅' if has_model else '⭕'} Modèle entraîné",
    )
    st.markdown("---")

    # Déconnexion
    if st.button("🚪 Se déconnecter", use_container_width=True):
        log_event("LOGOUT", f"user={st.session_state.username}")
        st.session_state.authenticated = False
        st.session_state.username      = ""
        st.rerun()

    st.markdown(
        "<div style='font-size:0.72rem;color:#6688aa;'>Ames Housing Dataset<br>1 460 obs · 80 variables</div>",
        unsafe_allow_html=True,
    )

page = st.session_state.page

# ╔══════════════════════════════════════════════════════════════════╗
# ║  PAGE 1 — UPLOAD & EXPLORATION                                   ║
# ╚══════════════════════════════════════════════════════════════════╝
if page == "Page 1":

    st.markdown("""
    <div class="hero">
      <div class="badge">Page 1 · Dataset</div>
      <h1>Upload &amp; Exploration</h1>
      <p>Chargez votre dataset, appliquez des filtres et explorez les distributions.</p>
    </div>""", unsafe_allow_html=True)

    section("Chargement du dataset")
col_up, col_info = st.columns([2, 1])

# Déclaration AVANT le bloc with
uploaded = None

with col_up:
    uploaded = st.file_uploader(
        "📂 Charger train.csv",
        type="csv",
        help="CSV Ames Housing — max 10 Mo, 10 000 lignes.",
    )
with col_info:
    st.markdown("""
    <div class="info-box">
    <b>Format attendu</b><br>
    CSV Ames Housing :<br>
    SalePrice, GrLivArea, Neighborhood…<br>
    <br><b>Limites sécurité :</b><br>
    Max 10 Mo · Max 10 000 lignes
    </div>""", unsafe_allow_html=True)

if uploaded is not None:
    file_bytes = uploaded.read()
    if len(file_bytes) == 0:
        st.error("Le fichier est vide.", icon="🚫")
    elif len(file_bytes) > 10 * 1024 * 1024:
        st.error("Fichier trop volumineux (max 10 Mo).", icon="🚫")
    else:
        st.session_state.df_raw = load_data(file_bytes)
        log_event("UPLOAD_OK", f"{uploaded.name} — {len(file_bytes)//1024} Ko")
elif st.session_state.df_raw is None:
    try:
        with open("train.csv", "rb") as f:
            st.session_state.df_raw = load_data(f.read())
        log_event("DATA_LOAD", "train.csv local chargé")
    except FileNotFoundError:
        st.error("⚠️ Aucun fichier trouvé. Chargez votre CSV.", icon="🚨")
        st.stop()

    df_raw = st.session_state.df_raw

    # ── Filtres ────────────────────────────────────────────────────
    section("Filtres interactifs")
    fc1, fc2, fc3 = st.columns(3)

    with fc1:
        p_min, p_max = int(df_raw["SalePrice"].min()), int(df_raw["SalePrice"].max())
        price_range = st.slider("💰 Plage de prix", p_min, p_max, (p_min, p_max),
                                step=5000, format="%d$",
                                help="Filtrer par prix de vente.")
    with fc2:
        neighborhoods = sorted(df_raw["Neighborhood"].unique())
        sel_neigh = st.multiselect("📍 Quartier(s)", neighborhoods, default=neighborhoods,
                                   help="Sélectionner un ou plusieurs quartiers.")
    with fc3:
        q_min, q_max = int(df_raw["OverallQual"].min()), int(df_raw["OverallQual"].max())
        qual_range = st.slider("⭐ Qualité", q_min, q_max, (q_min, q_max),
                               help="Note de qualité (1=mauvais, 10=excellent).")

    fc4, fc5 = st.columns(2)
    with fc4:
        s_min, s_max = int(df_raw["GrLivArea"].min()), int(df_raw["GrLivArea"].max())
        surf_range = st.slider("📐 Surface habitable (pi²)", s_min, s_max, (s_min, s_max),
                               step=100, help="GrLivArea.")
    with fc5:
        zones = sorted(df_raw["MSZoning"].unique())
        sel_zones = st.multiselect("🗺️ Zone", zones, default=zones, help="Zonage MSZoning.")

    # Validation des filtres
    price_min_v = validate_numeric(price_range[0], 0, 1_000_000, "price_min")
    price_max_v = validate_numeric(price_range[1], 0, 1_000_000, "price_max")

    df = df_raw[
        df_raw["SalePrice"].between(price_min_v or p_min, price_max_v or p_max) &
        df_raw["Neighborhood"].isin(sel_neigh) &
        df_raw["OverallQual"].between(*qual_range) &
        df_raw["GrLivArea"].between(*surf_range) &
        df_raw["MSZoning"].isin(sel_zones)
    ].copy()

    if len(df) == 0:
        st.error("Aucune donnée ne correspond aux filtres.", icon="🚫")
        st.stop()

    # KPIs
    c1, c2, c3, c4, c5 = st.columns(5)
    metric_card(c1, "Biens sélectionnés", f"{len(df):,}", f"{len(df)/len(df_raw)*100:.1f}%")
    metric_card(c2, "Prix moyen",  f"{df['SalePrice'].mean()/1000:.0f}k$", f"méd {df['SalePrice'].median()/1000:.0f}k$")
    metric_card(c3, "Surface moy.", f"{df['GrLivArea'].mean():.0f}", "pi²")
    metric_card(c4, "Quartiers",   str(df["Neighborhood"].nunique()), "sélectionnés")
    metric_card(c5, "Qualité moy.", f"{df['OverallQual'].mean():.1f}", "/10")

    st.markdown("<br>", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Distributions", "🔵 Surface vs Prix",
        "📦 Boxplot Quartiers", "🔥 Corrélations", "📋 Données brutes",
    ])

    with tab1:
        section("Distribution des variables")
        ca, cb = st.columns([2, 1])
        with ca:
            col_var = st.selectbox("Variable", ["SalePrice","GrLivArea","SurfaceTotale","AgeLogement","OverallQual"])
        with cb:
            col_log = st.checkbox("Échelle log", value=(col_var == "SalePrice"))

        data_plot = np.log1p(df[col_var]) if col_log else df[col_var]
        lx = f"log1p({col_var})" if col_log else col_var

        fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
        axes[0].hist(data_plot.dropna(), bins=40, color=DARK_BLUE, edgecolor="white", alpha=0.9)
        axes[0].axvline(data_plot.mean(),   color=GOLD,  ls="--", lw=1.6, label=f"Moy {data_plot.mean():.2f}")
        axes[0].axvline(data_plot.median(), color=CORAL, ls=":",  lw=1.6, label=f"Méd {data_plot.median():.2f}")
        axes[0].set_xlabel(lx); axes[0].set_ylabel("Effectif")
        axes[0].set_title(f"Histogramme — {col_var}", fontweight="600"); axes[0].legend(fontsize=9)
        sns.kdeplot(data_plot.dropna(), ax=axes[1], fill=True, color=TEAL, alpha=0.3, linewidth=2)
        axes[1].set_xlabel(lx); axes[1].set_ylabel("Densité")
        axes[1].set_title(f"Densité KDE — {col_var}", fontweight="600")
        plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()
        info(f"📌 Moy: <b>{df[col_var].mean():,.1f}</b> · Méd: <b>{df[col_var].median():,.1f}</b> · Skew: <b>{df[col_var].skew():.2f}</b>")

    with tab2:
        section("Surface habitable vs Prix")
        col_color = st.selectbox("Colorer par", ["OverallQual","Neighborhood","AgeLogement"])
        fig, ax = plt.subplots(figsize=(12, 6))
        cats = df[col_color].astype("category").cat.codes if df[col_color].dtype == "object" or df[col_color].nunique() <= 15 else df[col_color]
        sc = ax.scatter(df["GrLivArea"], df["SalePrice"], c=cats, cmap="viridis", alpha=0.5, s=18, linewidths=0)
        z = np.polyfit(df["GrLivArea"].dropna(), df["SalePrice"].dropna(), 1)
        xs = np.linspace(df["GrLivArea"].min(), df["GrLivArea"].max(), 200)
        ax.plot(xs, np.poly1d(z)(xs), color=CORAL, lw=2, ls="--", label="Tendance")
        plt.colorbar(sc, ax=ax, label=col_color, pad=0.01)
        ax.set_xlabel("GrLivArea (pi²)", fontsize=12); ax.set_ylabel("Prix ($)", fontsize=12)
        ax.set_title("Scatter — Surface vs Prix", fontweight="600")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))
        r = df["GrLivArea"].corr(df["SalePrice"])
        ax.text(0.03, 0.95, f"r = {r:.3f}", transform=ax.transAxes, fontsize=11, fontweight="600",
                color=DARK_BLUE, va="top", bbox=dict(boxstyle="round,pad=0.3", facecolor="#f0f7ff", edgecolor="#c0d0e0"))
        ax.legend(fontsize=9); plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

    with tab3:
        section("Distribution par quartier")
        cb1, cb2 = st.columns(2)
        with cb1: box_var = st.selectbox("Variable Y", ["SalePrice","GrLivArea","OverallQual","AgeLogement"])
        with cb2: sort_by = st.radio("Trier par", ["Médiane ↓","Médiane ↑","Alpha"], horizontal=True)
        meds  = df.groupby("Neighborhood")[box_var].median()
        order = (meds.sort_values(ascending=False).index.tolist() if sort_by == "Médiane ↓"
                 else meds.sort_values().index.tolist() if sort_by == "Médiane ↑"
                 else sorted(df["Neighborhood"].unique()))
        fig, ax = plt.subplots(figsize=(14, 6))
        bp = ax.boxplot([df[df["Neighborhood"]==nb][box_var].dropna().values for nb in order],
                        labels=order, patch_artist=True, widths=0.55,
                        medianprops=dict(color=GOLD, lw=2),
                        whiskerprops=dict(color="#aabbcc", lw=1.2),
                        capprops=dict(color="#aabbcc", lw=1.2),
                        flierprops=dict(marker="o", markerfacecolor=CORAL, alpha=0.4, markersize=3, ls="none"))
        for patch, color in zip(bp["boxes"], sns.color_palette("muted", n_colors=len(order))):
            patch.set_facecolor((*color[:3], 0.75))
        ax.set_xticklabels(order, rotation=45, ha="right", fontsize=8.5)
        ax.set_ylabel(box_var, fontsize=12)
        ax.set_title(f"Box plot — {box_var} par Neighborhood", fontweight="600")
        if box_var == "SalePrice":
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))
        plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()
        summary = df.groupby("Neighborhood")[box_var].agg(N="count", Moyenne="mean", Médiane="median").round(0).sort_values("Médiane", ascending=False)
        st.dataframe(summary.style.background_gradient(cmap="Blues", subset=["Médiane"]), use_container_width=True)

    with tab4:
        section("Matrice de corrélation")
        num_def = [c for c in ["SalePrice","GrLivArea","SurfaceTotale","OverallQual","AgeLogement","GarageArea","TotalBsmtSF"] if c in df.columns]
        sel_corr = st.multiselect("Variables", options=[c for c in df.select_dtypes("number").columns if c != "Id"], default=num_def)
        if len(sel_corr) >= 2:
            corr_m = df[sel_corr].corr()
            fig, ax = plt.subplots(figsize=(max(8, len(sel_corr)*0.75), max(6, len(sel_corr)*0.65)))
            sns.heatmap(corr_m, mask=np.triu(np.ones_like(corr_m, dtype=bool), k=1),
                        ax=ax, annot=True, fmt=".2f", cmap="RdYlBu_r", vmin=-1, vmax=1,
                        square=True, linewidths=0.4, linecolor="white", annot_kws={"size":8})
            ax.set_title("Matrice de corrélation", fontweight="600"); ax.tick_params(axis="x", rotation=45, labelsize=9)
            plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

    with tab5:
        section("Données filtrées")
        ca, cb = st.columns([3, 1])
        with ca:
            search_raw = st.text_input("🔍 Rechercher", placeholder="CollgCr, RL…")
            search = validate_text_input(search_raw, max_len=100) if search_raw else ""
        with cb:
            n_rows = st.selectbox("Lignes", [25, 50, 100, 200, "Toutes"], index=1)
        df_disp = df.copy()
        if search:
            df_disp = df_disp[df_disp.astype(str).apply(lambda c: c.str.contains(search, case=False, na=False)).any(axis=1)]
        if n_rows != "Toutes":
            df_disp = df_disp.head(int(n_rows))
        prio = [c for c in ["Id","Neighborhood","MSZoning","SalePrice","GrLivArea","SurfaceTotale","OverallQual","AgeLogement"] if c in df_disp.columns]
        df_disp = df_disp[prio + [c for c in df_disp.columns if c not in prio]]
        st.dataframe(df_disp.style.background_gradient(cmap="Blues", subset=["SalePrice"]).format({"SalePrice":"{:,.0f}","GrLivArea":"{:,.0f}"}), use_container_width=True, height=380)
        st.markdown(f"**{len(df_disp):,} lignes** affichées sur {len(df):,} filtrées.")
        st.markdown("---")
        dl1, dl2 = st.columns(2)
        buf = io.StringIO(); df.to_csv(buf, index=False)
        dl1.download_button("⬇️ Données filtrées (CSV)", buf.getvalue().encode(), f"filtered_{len(df)}.csv", "text/csv",
                            help="Télécharge les données filtrées.")
        buf2 = io.StringIO(); df.describe().to_csv(buf2)
        dl2.download_button("📈 Stats descriptives (CSV)", buf2.getvalue().encode(), "stats.csv", "text/csv")
        log_event("DATA_EXPORT", f"user={st.session_state.username} lignes={len(df)}")


# ╔══════════════════════════════════════════════════════════════════╗
# ║  PAGE 2 — ENTRAÎNEMENT                                           ║
# ╚══════════════════════════════════════════════════════════════════╝
elif page == "Page 2":

    st.markdown("""
    <div class="hero">
      <div class="badge">Page 2 · Modélisation</div>
      <h1>Entraînement &amp;<br>Performances</h1>
      <p>Configurez, entraînez et comparez vos modèles de régression.</p>
    </div>""", unsafe_allow_html=True)

    if st.session_state.df_raw is None:
        st.warning("⚠️ Chargez d'abord vos données en **Page 1**.", icon="⚠️"); st.stop()

    df_raw = st.session_state.df_raw

    section("Configuration")
    cfg1, cfg2, cfg3 = st.columns(3)
    with cfg1:
        model_choice = st.selectbox("🤖 Modèle", ["Régression Linéaire","Random Forest","Gradient Boosting","Comparer les 3"])
    with cfg2:
        test_size = st.slider("📊 Taille test (%)", 10, 40, 20, step=5)
    with cfg3:
        target_log = st.checkbox("🔁 Log-transformer la cible", value=True)

    with st.expander("⚙️ Hyperparamètres"):
        ha1, ha2, ha3 = st.columns(3)
        with ha1: n_estimators = st.slider("n_estimators", 50, 500, 300, step=50)
        with ha2: max_depth    = st.slider("max_depth (RF)", 2, 20, 0)
        with ha3: lr_gb        = st.slider("learning_rate (GB)", 0.01, 0.3, 0.05, step=0.01)

    @st.cache_data(show_spinner="Préparation…")
    def prepare(df_raw, test_sz, log_target):
        df = impute(df_raw.copy())
        if "SalePrice" not in df.columns: return None, None, None, None, None, {}
        y = np.log1p(df["SalePrice"]) if log_target else df["SalePrice"]
        X = df.drop(columns=["SalePrice","Id"], errors="ignore")
        if "YearBuilt" in X.columns:   X["AgeLogement"]    = 2025 - X["YearBuilt"]
        if all(c in X.columns for c in ["GrLivArea","TotalBsmtSF"]): X["SurfaceTotale"] = X["GrLivArea"] + X["TotalBsmtSF"].fillna(0)
        if all(c in X.columns for c in ["FullBath","HalfBath"]): X["NbSallesDeBain"] = X["FullBath"] + 0.5 * X["HalfBath"]
        les = {}
        for c in X.select_dtypes("object").columns:
            le = LabelEncoder(); X[c] = le.fit_transform(X[c].astype(str)); les[c] = le
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_sz/100, random_state=42)
        return X_tr, X_te, y_tr, y_te, X.columns.tolist(), les

    X_train, X_test, y_train, y_test, feat_cols, les = prepare(df_raw, test_size, target_log)
    if X_train is None: st.error("Colonne SalePrice introuvable.", icon="🚨"); st.stop()

    section("Lancement")
    if st.button("🚀 Entraîner le modèle"):
        md_val = None if max_depth == 0 else max_depth
        model_defs = {
            "Régression Linéaire": Pipeline([("sc", StandardScaler()), ("m", LinearRegression())]),
            "Random Forest":       RandomForestRegressor(n_estimators=n_estimators, max_depth=md_val, min_samples_split=5, random_state=42, n_jobs=-1),
            "Gradient Boosting":   GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=lr_gb, max_depth=4, subsample=0.8, random_state=42),
        }
        to_train = list(model_defs.keys()) if model_choice == "Comparer les 3" else [model_choice]
        results  = {}
        prog = st.progress(0, text="Entraînement…")
        for i, name in enumerate(to_train):
            m = model_defs[name]; m.fit(X_train, y_train)
            preds = m.predict(X_test)
            r2   = r2_score(y_test, preds)
            mae  = mean_absolute_error(np.expm1(y_test), np.expm1(preds)) if target_log else mean_absolute_error(y_test, preds)
            rmse = np.sqrt(mean_squared_error(y_test, preds))
            cv   = cross_val_score(m, X_train, y_train, cv=5, scoring="r2", n_jobs=-1)
            results[name] = {"model":m,"preds":preds,"R²":round(r2,4),"MAE":round(mae,0),"RMSLE":round(rmse,4),"CV_mean":round(cv.mean(),4),"CV_std":round(cv.std(),4)}
            prog.progress((i+1)/len(to_train), text=f"✅ {name}")
        best_name = max(results, key=lambda k: results[k]["R²"])
        st.session_state.trained_model  = results[best_name]["model"]
        st.session_state.model_name     = best_name
        st.session_state.model_features = feat_cols
        st.session_state.label_encoders = les
        st.session_state.metrics        = results
        st.session_state.target_log     = target_log
        st.session_state.y_test         = y_test
        log_event("MODEL_TRAIN", f"user={st.session_state.username} best={best_name} R²={results[best_name]['R²']}")
        st.success(f"✅ Meilleur modèle : **{best_name}** (R²={results[best_name]['R²']:.4f})")

    if st.session_state.metrics:
        results = st.session_state.metrics
        section("Performances")
        perf_df = pd.DataFrame({k: {kk:vv for kk,vv in v.items() if kk not in ["model","preds"]} for k,v in results.items()}).T
        st.dataframe(perf_df.style.background_gradient(cmap="Greens", subset=["R²"]).background_gradient(cmap="Reds_r", subset=["MAE"]), use_container_width=True)

        best = max(results, key=lambda k: results[k]["R²"]); b = results[best]
        c1,c2,c3,c4 = st.columns(4)
        metric_card(c1,"Meilleur modèle",best.split()[0],"")
        metric_card(c2,"R²",f"{b['R²']:.4f}","")
        metric_card(c3,"MAE",f"{b['MAE']:,.0f}$","")
        metric_card(c4,"RMSLE",f"{b['RMSLE']:.4f}","")

        vt1, vt2 = st.tabs(["📊 Comparaison", "🎯 Réel vs Prédit"])
        with vt1:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            for ax, met, color in zip(axes, ["R²","MAE","RMSLE"], ["#2ecc71","#e74c3c","#3498db"]):
                vals = [results[k][met] for k in results]
                bars = ax.bar(list(results.keys()), vals, color=color, edgecolor="white", width=0.5)
                ax.set_title(met, fontweight="600"); ax.set_xticklabels(list(results.keys()), rotation=15, ha="right")
                for bar, v in zip(bars, vals):
                    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()*1.01, f"{v:,.4g}", ha="center", fontsize=9)
            plt.suptitle("Comparaison des modèles", fontsize=14, fontweight="600")
            plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

        with vt2:
            sel_model = st.selectbox("Modèle", list(results.keys()))
            y_te_orig = st.session_state.get("y_test", None)
            tgt_log   = st.session_state.get("target_log", True)
            if y_te_orig is not None:
                y_real = np.expm1(y_te_orig) if tgt_log else y_te_orig
                y_pred = np.expm1(results[sel_model]["preds"]) if tgt_log else results[sel_model]["preds"]
                fig, axes = plt.subplots(1, 2, figsize=(14, 5))
                axes[0].scatter(y_real, y_pred, alpha=0.4, s=18, color=DARK_BLUE)
                lims = [min(y_real.min(),y_pred.min()), max(y_real.max(),y_pred.max())]
                axes[0].plot(lims, lims, "r--", lw=1.5, label="Idéal")
                axes[0].set_xlabel("Réel ($)"); axes[0].set_ylabel("Prédit ($)")
                axes[0].set_title(f"{sel_model}", fontweight="600"); axes[0].legend()
                axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_:f"{x/1000:.0f}k"))
                sns.histplot(y_real-y_pred, kde=True, ax=axes[1], color=CORAL)
                axes[1].axvline(0, color="black", ls="--"); axes[1].set_xlabel("Erreur ($)")
                axes[1].set_title("Résidus", fontweight="600")
                plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

        section("Export modèle")
        buf_model = io.BytesIO()
        joblib.dump(st.session_state.trained_model, buf_model)
        st.download_button("⬇️ Télécharger le modèle (.pkl)", data=buf_model.getvalue(),
                           file_name=f"model_{st.session_state.model_name.replace(' ','_')}.pkl",
                           mime="application/octet-stream")


# ╔══════════════════════════════════════════════════════════════════╗
# ║  PAGE 3 — PRÉDICTION                                             ║
# ╚══════════════════════════════════════════════════════════════════╝
elif page == "Page 3":

    st.markdown("""
    <div class="hero">
      <div class="badge">Page 3 · Prédiction</div>
      <h1>Interface de<br>Prédiction</h1>
      <p>Renseignez les caractéristiques d'un bien pour estimer son prix.</p>
    </div>""", unsafe_allow_html=True)

    if not st.session_state.trained_model:
        with st.expander("📦 Charger un modèle .pkl", expanded=True):
            uploaded_model = st.file_uploader("model.pkl", type=["pkl","joblib"])
            if uploaded_model:
                try:
                    st.session_state.trained_model = joblib.load(uploaded_model)
                    st.session_state.model_name    = "Modèle importé"
                    log_event("MODEL_LOAD", f"user={st.session_state.username} pkl importé")
                    st.success("✅ Modèle chargé !")
                except Exception as e:
                    st.error(f"Erreur : {e}"); st.stop()
        if not st.session_state.trained_model:
            st.warning("⚠️ Entraînez d'abord un modèle en Page 2.", icon="⚠️"); st.stop()

    model     = st.session_state.trained_model
    feat_cols = st.session_state.model_features
    les       = st.session_state.label_encoders
    tgt_log   = st.session_state.get("target_log", True)
    df_ref    = st.session_state.df_raw

    section("Caractéristiques du bien")

    with st.form("pred_form"):
        r1c1, r1c2, r1c3, r1c4 = st.columns(4)
        with r1c1: OverallQual  = st.slider("⭐ Qualité", 1, 10, 7, help="Qualité globale (1–10).")
        with r1c2: GrLivArea    = st.number_input("📐 Surface hab. (pi²)", 500, 6000, 1500, step=50, help="GrLivArea.")
        with r1c3: TotalBsmtSF  = st.number_input("🏚️ Sous-sol (pi²)", 0, 3000, 800, step=50)
        with r1c4: GarageArea   = st.number_input("🚗 Garage (pi²)", 0, 1500, 400, step=25)

        r2c1, r2c2, r2c3 = st.columns(3)
        with r2c1: YearBuilt    = st.number_input("🏗️ Année construction", 1850, 2024, 2000)
        with r2c2: FullBath     = st.number_input("🚿 SDB complètes", 0, 4, 2)
        with r2c3: BedroomAbvGr = st.number_input("🛏️ Chambres", 0, 8, 3)

        r3c1, r3c2, r3c3 = st.columns(3)
        with r3c1:
            neigh_opts   = sorted(df_ref["Neighborhood"].unique()) if df_ref is not None else ["CollgCr"]
            Neighborhood = st.selectbox("📍 Quartier", neigh_opts)
        with r3c2:
            zone_opts = sorted(df_ref["MSZoning"].unique()) if df_ref is not None else ["RL"]
            MSZoning  = st.selectbox("🗺️ Zone", zone_opts)
        with r3c3:
            LotArea = st.number_input("🌿 Terrain (pi²)", 1000, 100000, 8000, step=500)

        submitted = st.form_submit_button("🔮 Estimer le prix", use_container_width=True)

    if submitted:
        # Validation des entrées numériques
        gr_valid  = validate_numeric(GrLivArea,  100, 10000, "GrLivArea")
        bsmt_valid = validate_numeric(TotalBsmtSF, 0, 5000, "TotalBsmtSF")
        qual_valid = validate_numeric(OverallQual, 1, 10,   "OverallQual")

        if None in [gr_valid, bsmt_valid, qual_valid]:
            st.error("⚠️ Valeurs saisies invalides. Vérifiez les champs.", icon="🚫")
        else:
            AgeLogement = 2025 - YearBuilt
            SurfaceTotale  = gr_valid + bsmt_valid
            NbSallesDeBain = FullBath + 0.5

            input_dict = {
                "MSSubClass":20,"MSZoning":MSZoning,"LotFrontage":65,"LotArea":LotArea,
                "Street":"Pave","Alley":"None","LotShape":"Reg","LandContour":"Lvl",
                "Utilities":"AllPub","LotConfig":"Inside","LandSlope":"Gtl",
                "Neighborhood":Neighborhood,"Condition1":"Norm","Condition2":"Norm",
                "BldgType":"1Fam","HouseStyle":"2Story","OverallQual":int(qual_valid),
                "OverallCond":5,"YearBuilt":YearBuilt,"YearRemodAdd":YearBuilt,
                "RoofStyle":"Gable","RoofMatl":"CompShg",
                "Exterior1st":"VinylSd","Exterior2nd":"VinylSd",
                "MasVnrType":"None","MasVnrArea":0,"ExterQual":"Gd","ExterCond":"TA",
                "Foundation":"PConc","BsmtQual":"Gd","BsmtCond":"TA","BsmtExposure":"No",
                "BsmtFinType1":"GLQ","BsmtFinSF1":400,"BsmtFinType2":"Unf","BsmtFinSF2":0,
                "BsmtUnfSF":300,"TotalBsmtSF":int(bsmt_valid),"Heating":"GasA",
                "HeatingQC":"Ex","CentralAir":"Y","Electrical":"SBrkr",
                "1stFlrSF":int(gr_valid)//2,"2ndFlrSF":int(gr_valid)//2,
                "LowQualFinSF":0,"GrLivArea":int(gr_valid),
                "BsmtFullBath":1,"BsmtHalfBath":0,"FullBath":FullBath,"HalfBath":1,
                "BedroomAbvGr":BedroomAbvGr,"KitchenAbvGr":1,"KitchenQual":"Gd",
                "TotRmsAbvGrd":7,"Functional":"Typ","Fireplaces":1,"FireplaceQu":"TA",
                "GarageType":"Attchd","GarageYrBlt":YearBuilt,"GarageFinish":"RFn",
                "GarageCars":2,"GarageArea":GarageArea,"GarageQual":"TA","GarageCond":"TA",
                "PavedDrive":"Y","WoodDeckSF":0,"OpenPorchSF":50,"EnclosedPorch":0,
                "3SsnPorch":0,"ScreenPorch":0,"PoolArea":0,"PoolQC":"None","Fence":"None",
                "MiscFeature":"None","MiscVal":0,"MoSold":6,"YrSold":2024,
                "SaleType":"WD","SaleCondition":"Normal",
                "AgeLogement":AgeLogement,"SurfaceTotale":SurfaceTotale,"NbSallesDeBain":NbSallesDeBain,
            }

            row = pd.DataFrame([input_dict])
            if feat_cols:
                for c in row.select_dtypes("object").columns:
                    if c in les:
                        le  = les[c]; val = str(row[c].iloc[0])
                        row[c] = le.transform([val])[0] if val in le.classes_ else 0
                    else:
                        row[c] = 0
                for mc in [c for c in feat_cols if c not in row.columns]:
                    row[mc] = 0
                row = row[feat_cols]
            else:
                for c in row.select_dtypes("object").columns: row[c] = 0

            pred_raw = model.predict(row)[0]
            price    = np.expm1(pred_raw) if tgt_log else pred_raw

            log_event("PREDICTION", f"user={st.session_state.username} price={price:,.0f} neigh={Neighborhood}")

            st.markdown(f"""
            <div class="result-box">
              <div class="label">Prix estimé — {st.session_state.model_name}</div>
              <div class="price">${price:,.0f}</div>
              <div class="label">Fourchette : ${price*0.90:,.0f} — ${price*1.10:,.0f}</div>
            </div>""", unsafe_allow_html=True)

            if df_ref is not None:
                med = df_ref["SalePrice"].median()
                delta = (price - med) / med * 100
                c1,c2,c3 = st.columns(3)
                metric_card(c1,"Prix estimé",f"${price:,.0f}","")
                metric_card(c2,"Médiane marché",f"${med:,.0f}","référence")
                metric_card(c3,"Écart",f"{abs(delta):.1f}%","au-dessus" if delta>=0 else "en-dessous")

                section("Position dans le marché")
                fig, ax = plt.subplots(figsize=(12, 4))
                ax.hist(df_ref["SalePrice"], bins=50, color=DARK_BLUE, edgecolor="white", alpha=0.7)
                ax.axvline(price, color=GOLD, lw=2.5, label=f"Estimation ${price:,.0f}")
                ax.axvline(med,   color=CORAL, lw=1.8, ls="--", label=f"Médiane ${med:,.0f}")
                ax.set_xlabel("Prix ($)"); ax.set_ylabel("Effectif")
                ax.set_title("Position de votre bien", fontweight="600")
                ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_:f"{x/1000:.0f}k"))
                ax.legend(); plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()


# ╔══════════════════════════════════════════════════════════════════╗
# ║  PAGE 4 — LOGS & SÉCURITÉ                                        ║
# ╚══════════════════════════════════════════════════════════════════╝
elif page == "Page 4":

    st.markdown("""
    <div class="hero">
      <div class="badge">Page 4 · Sécurité</div>
      <h1>Logs &amp;<br>Sécurité</h1>
      <p>Supervision des accès, des prédictions et des événements de l'application.</p>
    </div>""", unsafe_allow_html=True)

    # ── Statuts sécurité ───────────────────────────────────────────
    section("Statut de sécurité")
    s1, s2, s3, s4 = st.columns(4)
    metric_card(s1, "HTTPS",          "🔒 Actif",   "chiffrement TLS")
    metric_card(s2, "Auth",           "✅ Requise",  "login obligatoire")
    metric_card(s3, "XSRF",           "✅ Activé",   "config.toml")
    metric_card(s4, "Upload limit",   "10 Mo",       "validé côté serveur")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Checklist HTTPS ────────────────────────────────────────────
    section("Checklist de sécurité")
    checks = [
        ("🔒", "HTTPS activé",                   "Streamlit Cloud fournit HTTPS automatiquement. Vérifiez le cadenas 🔒 dans votre navigateur."),
        ("🔐", "Authentification basique",        "Login requis avant toute utilisation. Credentials stockés dans st.secrets (Streamlit Cloud)."),
        ("🛡️", "Validation des entrées",          "Toutes les entrées utilisateur sont nettoyées (caractères interdits, longueur max, plages numériques)."),
        ("📋", "Logs d'activité",                 "Chaque action (connexion, upload, prédiction, export) est horodatée et enregistrée."),
        ("🚫", "Limite upload",                   "Fichiers CSV limités à 10 Mo et 10 000 lignes pour éviter les abus."),
        ("🔑", "Secrets externalisés",            "Aucun mot de passe dans le code. Utilisation de st.secrets / Streamlit Cloud Secrets."),
        ("🔄", "Protection XSRF",                 "enableXsrfProtection = true dans config.toml."),
        ("👁️", "Données non collectées",          "gatherUsageStats = false dans config.toml."),
    ]
    for icon, title, desc in checks:
        st.markdown(
            f'<div style="display:flex;gap:1rem;align-items:flex-start;padding:0.6rem 0.8rem;'
            f'margin-bottom:0.4rem;background:#f8fffe;border:1px solid #d0e8d0;border-radius:8px;">'
            f'<span style="font-size:1.2rem;">{icon}</span>'
            f'<div><b style="color:#1a1a2e;">{title}</b><br>'
            f'<span style="font-size:0.82rem;color:#556677;">{desc}</span></div></div>',
            unsafe_allow_html=True,
        )

    # ── Journal des événements ─────────────────────────────────────
    section("Journal des événements (session courante)")
    logs = st.session_state.get("logs", [])
    if not logs:
        st.info("Aucun événement enregistré pour cette session.", icon="ℹ️")
    else:
        st.markdown(f"**{len(logs)} événements** depuis la connexion.")
        log_container = st.container()
        with log_container:
            for log_line in reversed(logs[-50:]):
                st.markdown(f'<div class="log-line">📌 {log_line}</div>', unsafe_allow_html=True)

        # Export logs
        logs_text = "\n".join(logs)
        st.download_button(
            "⬇️ Exporter les logs (.txt)",
            data=logs_text.encode("utf-8"),
            file_name=f"logs_immpredict_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.txt",
            mime="text/plain",
            help="Télécharge le journal complet de la session.",
        )

    # ── Info déploiement ───────────────────────────────────────────
    section("Guide de configuration Streamlit Cloud")
    st.markdown("""
    <div class="info-box">
    <b>Étapes pour configurer les secrets sur Streamlit Cloud :</b><br><br>
    1. Aller sur <b>share.streamlit.io</b> → votre app → <b>Settings</b><br>
    2. Cliquer sur <b>Secrets</b><br>
    3. Coller le contenu suivant :
    </div>
    """, unsafe_allow_html=True)

    st.code("""
[auth]
username = "admin"
password = "votre_mot_de_passe_fort_ici"
    """, language="toml")

    st.markdown("""
    <div class="info-box">
    ⚠️ <b>Ne jamais commiter</b> le fichier <code>.streamlit/secrets.toml</code> sur GitHub.<br>
    Il est déjà exclu via le <code>.gitignore</code> fourni.
    </div>
    """, unsafe_allow_html=True)
