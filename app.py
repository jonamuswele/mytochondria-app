import math
import random
from datetime import datetime, timedelta, date
from typing import List, Dict, Any, Optional, Tuple
import streamlit as st
import pandas as pd
import requests
import json, os
import sqlite3
import altair as alt

USERS_FILE = "users.json"

DB_FILE = "users.db"

AER_TEXTURE_DEFAULT = {
    "I": "sand",
    "IIa": "loam",
    "IIb": "sand",
    "III": "clay"
}

CROP_SPACING_DEFAULT = {
    "maize": (75, 25),
    "beans": (45, 20),
    "rice": (30, 20),
    "maize+beans": (75, 25)  # fallback, or you could ask both crops
}
# --- Initialize theme early ---
if "theme" not in st.session_state:
    st.session_state.theme = "light"

logo_col, space_col, mode_col = st.columns([1, 5, 1])
with logo_col:
    st.image("https://mytochondria.com/assets/logo.png", width=60)

with mode_col:
    # Toggle reflects current theme
    new_toggle = st.toggle(
        "🌙 Dark Mode",
        value=(st.session_state.theme == "dark"),
        key="theme_toggle"
    )
    # Update state immediately when changed
    st.session_state.theme = "dark" if new_toggle else "light"

if st.session_state.theme == "dark":
    st.markdown("""
        <style>
        /* === GLOBAL LAYOUT === */
        body, .stApp, .block-container {
            background-color: #121212 ;
            color: #e0e0e0 ;
        }
        h1, h2, h3, h4 {
            color: #2563eb ;
        }

        /* === NAVIGATION + BUTTONS === */
        div[data-testid="stHorizontalBlock"] div.stButton > button,
        .stButton > button,
        div.stDownloadButton > button {
            background-color: #2563eb ;
            color: white ;
            border: 1px solid #2563eb ;
            border-radius: 6px;
            font-weight: 600;
            transition: all 0.2s ease;
        }
        div[data-testid="stHorizontalBlock"] div.stButton > button:hover,
        .stButton > button:hover,
        div.stDownloadButton > button:hover {
            background-color: #1e40af ;
        }
        div[data-testid="stHorizontalBlock"] div.stButton > button:active,
        div[data-testid="stHorizontalBlock"] div.stButton > button.nav-active,
        .stButton > button:active,
        div.stDownloadButton > button:active {
            background-color: #121212 ;
            color: #2563eb ;
            border: 1px solid #2563eb ;
        }

        /* === INPUTS === */
        div[data-baseweb="select"] > div,
        input, textarea, .stDateInput input {
            background-color: #1c1c28 ;
            color: #e0e0e0 ;
            border: 1px solid #2563eb ;
            border-radius: 6px;
        }
        .stRadio div[role="radiogroup"] label,
        .stCheckbox label {
            color: #e0e0e0 ;
        }
        .stNumberInput, .stNumberInput > div, .stNumberInput input {
            background-color: transparent ;
        }
        .stNumberInput button {
            border: 1px solid #2563eb !important;
            border-radius: 4px !important;
            background-color: inherit !important;
            color: inherit !important;
        }
        .stSlider [role="slider"] {
            background-color: #2563eb !important;
            border: 2px solid #2563eb !important;
        }
        .stSlider > div > div {
            background: linear-gradient(to right, #2563eb, #1e40af) !important;
        }

        /* === CARDS === */
        .card, .metric {
            background: linear-gradient(135deg, #121212, #1e1e2f);
            border: 1px solid #2563eb55;
            color: #e0e0e0 !important;
            border-radius: 10px;
            padding: 8px;
        }

        /* === TABLES (Dark Mode) === */
        div[data-testid="stDataFrame"] {
            border-radius: 10px !important;
            border: 1px solid #2563eb55 !important;
            overflow: hidden !important;
        }
        div[data-testid="stDataFrame"] thead tr th {
            background-color: #2563eb !important;
            color: white !important;
            font-weight: 600 !important;
            border: none !important;
        }
        div[data-testid="stDataFrame"] tbody tr td {
            background-color: #1c1c28 !important;
            color: #e0e0e0 !important;
            border: none !important;
        }
        div[data-testid="stDataFrame"] tbody tr:hover td {
            background-color: rgba(37,99,235,0.15) !important;
        }

        /* === CHART CONTAINERS === */
        .stPlotlyChart, .stVegaLiteChart, .stAltairChart,
        .stPydeckChart, .stDeckGlChart, .stEchartsChart {
            background-color: #1c1c28 !important;
            border-radius: 10px !important;
            border: 1px solid #2563eb55 !important;
            padding: 8px !important;
            overflow: hidden !important;
        }
        .vega-bindings, .vega-bindings span, .vega-bindings label {
            color: #e0e0e0 !important;
        }
        .vega-tooltip {
            background-color: #1c1c28 !important;
            color: #e0e0e0 !important;
            border: 1px solid #2563eb !important;
        }

        /* === EXPANDERS === */
        .streamlit-expanderHeader {
            font-weight: 600 !important;
            border: 1px solid #2563eb !important;
            border-radius: 6px !important;
        }
        </style>
    """, unsafe_allow_html=True)

else:
    st.markdown("""
        <style>
        /* === GLOBAL LAYOUT === */
        body, .stApp, .block-container {
            background-color: #ffffff !important;
            color: #111 !important;
        }
        h1, h2, h3, h4 {
            color: #4caf50 !important;
        }

        /* === NAVIGATION + BUTTONS === */
        div[data-testid="stHorizontalBlock"] div.stButton > button,
        .stButton > button,
        div.stDownloadButton > button {
            background-color: #4caf50 !important;
            color: white !important;
            border: 1px solid #4caf50 !important;
            border-radius: 6px;
            font-weight: 600;
            transition: all 0.2s ease;
        }
        div[data-testid="stHorizontalBlock"] div.stButton > button:hover,
        .stButton > button:hover,
        div.stDownloadButton > button:hover {
            background-color: #388e3c !important;
        }
        div[data-testid="stHorizontalBlock"] div.stButton > button:active,
        div[data-testid="stHorizontalBlock"] div.stButton > button.nav-active,
        .stButton > button:active,
        div.stDownloadButton > button:active {
            background-color: #ffffff !important;
            color: #4caf50 !important;
            border: 1px solid #4caf50 !important;
        }

        /* === INPUTS === */
        div[data-baseweb="select"] > div,
        input, textarea, .stDateInput input {
            background-color: #f9fcf9 !important;
            color: #111 !important;
            border: 1px solid #4caf50 !important;
            border-radius: 6px;
        }
        .stRadio div[role="radiogroup"] label,
        .stCheckbox label {
            color: #111 !important;
        }
        .stNumberInput, .stNumberInput > div, .stNumberInput input {
            background-color: transparent !important;
        }
        .stNumberInput button {
            border: 1px solid #4caf50 !important;
            border-radius: 4px !important;
            background-color: inherit !important;
            color: inherit !important;
        }
        .stSlider [role="slider"] {
            background-color: #4caf50 !important;
            border: 2px solid #4caf50 !important;
        }
        .stSlider > div > div {
            background: linear-gradient(to right, #4caf50, #a8d5a2) !important;
        }

        /* === CARDS === */
        .card, .metric {
            background: linear-gradient(135deg, #ffffff, #f4f9f4);
            border: 1px solid #4caf5044;
            color: #111 !important;
            border-radius: 10px;
            padding: 8px;
        }

        /* === TABLES (Light Mode) === */
        div[data-testid="stDataFrame"] {
            border-radius: 10px !important;
            border: 1px solid #4caf5044 !important;
            overflow: hidden !important;
        }
        div[data-testid="stDataFrame"] thead tr th {
            background-color: #4caf50 !important;
            color: white !important;
            font-weight: 600 !important;
            border: none !important;
        }
        div[data-testid="stDataFrame"] tbody tr td {
            background-color: #ffffff !important;
            color: #111 !important;
            border: none !important;
        }
        div[data-testid="stDataFrame"] tbody tr:hover td {
            background-color: rgba(76,175,80,0.15) !important;
        }

        /* === CHART CONTAINERS === */
        .stPlotlyChart, .stVegaLiteChart, .stAltairChart,
        .stPydeckChart, .stDeckGlChart, .stEchartsChart {
            background-color: #ffffff !important;
            border-radius: 10px !important;
            border: 1px solid #4caf5044 !important;
            padding: 8px !important;
            overflow: hidden !important;
        }
        .vega-bindings, .vega-bindings span, .vega-bindings label {
            color: #111 !important;
        }
        .vega-tooltip {
            background-color: #ffffff !important;
            color: #111 !important;
            border: 1px solid #4caf50 !important;
        }

        /* === EXPANDERS === */
        .streamlit-expanderHeader {
            font-weight: 600 !important;
            border: 1px solid #4caf50 !important;
            border-radius: 6px !important;
        }
        </style>
    """, unsafe_allow_html=True)

if "active_tab" not in st.session_state:
    st.session_state.active_tab = 0  # default Home

tab_names = ["Home", "Sensor Mode", "Crop Planner", "Tips & Tricks","AI Imagery Analysis", "Manage Account"]

def _accent(theme: str) -> str:
    return "#2563eb" if theme == "dark" else "#4caf50"

def _text_bg_grid(theme: str) -> tuple[str, str, str]:
    text = "#e0e0e0" if theme == "dark" else "#111"
    bg   = "#1c1c28" if theme == "dark" else "#f9fcf9"
    grid = (_accent(theme) + "33")
    return text, bg, grid

def style_chart(chart, theme: str):
    accent = "#2563eb" if theme == "dark" else "#4caf50"
    text   = "#e0e0e0" if theme == "dark" else "#111"
    bg     = "#1c1c28" if theme == "dark" else "#ffffff"   # light mode = pure white
    grid   = accent + "33"

    return (
        chart
        # Force chart area background
        .configure_view(
            strokeOpacity=0,
            fill=bg
        )
        # Force full chart background (outside of plot area)
        .configure(background=bg)
        # Axis styling
        .configure_axis(
            labelColor=text,
            titleColor=text,
            grid=True,
            gridColor=grid
        )
        # Legend styling
        .configure_legend(
            labelColor=text,
            titleColor=text
        )
    )

def _db():
    # Streamlit runs multi-threaded; set check_same_thread=False
    return sqlite3.connect(DB_FILE, check_same_thread=False)


def delete_farm(farm_id: str):
    with _db() as conn:
        c = conn.cursor()
        c.execute("DELETE FROM farms WHERE farm_id=?", (farm_id,))
        conn.commit()


def init_db():
    with _db() as conn:
        c = conn.cursor()
        c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password TEXT,
            email    TEXT
        )
        """)
        c.execute("""
        CREATE TABLE IF NOT EXISTS farms (
            farm_id       TEXT PRIMARY KEY,
            username      TEXT NOT NULL,
            system_id     TEXT,
            crop          TEXT,
            location      TEXT,
            lat           REAL,
            lon           REAL,
            soil_texture  TEXT,
            row_cm        INTEGER,
            plant_cm      INTEGER,
            spacing       TEXT,
            planting_date TEXT,
            compliance    TEXT,
            yield_factor  REAL,
            om_pct        REAL,
            agent_json    TEXT,
            FOREIGN KEY(username) REFERENCES users(username)
        )
        """)
        conn.commit()


# Optional: one-time migration from users.json → users.db (safe to keep; no-op if file missing)
def migrate_json_to_db():
    try:
        if not os.path.exists("users.json"):
            return
        with open("users.json", "r") as f:
            users = json.load(f)
        with _db() as conn:
            c = conn.cursor()
            for u in users:
                c.execute("INSERT OR IGNORE INTO users (username, password, email) VALUES (?,?,?)",
                          (u.get("username"), u.get("password"), u.get("email")))
                for f_ in u.get("farms", []):
                    c.execute("""
                        INSERT OR IGNORE INTO farms
                        (farm_id, username, system_id, crop, location, lat, lon, soil_texture,
                         row_cm, plant_cm, spacing, planting_date, compliance, yield_factor, om_pct, agent_json)
                        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """, (
                        f_.get("farm_id"), u.get("username"), f_.get("system_id"),
                        f_.get("crop"), f_.get("location"), f_.get("lat"), f_.get("lon"),
                        f_.get("soil_texture"), f_.get("row_cm"), f_.get("plant_cm"),
                        f_.get("spacing"), f_.get("planting_date"), f_.get("compliance"),
                        f_.get("yield_factor", 1.0), f_.get("om_pct", 2.0),
                        json.dumps(f_.get("agent", {}))
                    ))
            conn.commit()
        # prevent re-import on next run
        os.rename("users.json", "users.json.migrated.bak")
    except Exception:
        # If anything fails here, just continue — app will still run on DB only.
        pass


init_db()
migrate_json_to_db()


def load_users():
    if not os.path.exists(USERS_FILE):
        return []
    with open(USERS_FILE, "r") as f:
        return json.load(f)


def save_users(users):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=2)


def user_exists(username: str) -> bool:
    with _db() as conn:
        c = conn.cursor()
        c.execute("SELECT 1 FROM users WHERE username=?", (username,))
        return c.fetchone() is not None


def create_user(username: str, password: str, email: str):
    with _db() as conn:
        c = conn.cursor()
        c.execute("INSERT INTO users (username, password, email) VALUES (?,?,?)", (username, password, email))
        conn.commit()


def _load_farms_for(username: str) -> list[dict]:
    with _db() as conn:
        c = conn.cursor()
        c.execute(
            "SELECT farm_id, username, system_id, crop, location, lat, lon, soil_texture, row_cm, plant_cm, spacing, planting_date, compliance, yield_factor, om_pct, agent_json FROM farms WHERE username=?",
            (username,))
        rows = c.fetchall()
    farms = []
    for r in rows:
        farms.append({
            "farm_id": r[0], "username": r[1], "system_id": r[2], "crop": r[3], "location": r[4],
            "lat": r[5], "lon": r[6], "soil_texture": r[7], "row_cm": r[8], "plant_cm": r[9],
            "spacing": r[10], "planting_date": r[11], "compliance": r[12],
            "yield_factor": r[13], "om_pct": r[14],
            "agent": json.loads(r[15] or "{}"),
        })
    return farms


# Keep the same name/signature your app already uses:
def find_user(username, password=None):
    with _db() as conn:
        c = conn.cursor()
        if password is None:
            c.execute("SELECT username, password, email FROM users WHERE username=?", (username,))
        else:
            c.execute("SELECT username, password, email FROM users WHERE username=? AND password=?",
                      (username, password))
        row = c.fetchone()
    if not row:
        return None
    return {"username": row[0], "password": row[1], "email": row[2], "farms": _load_farms_for(row[0])}


def save_farm(username: str, farm: dict):
    with _db() as conn:
        c = conn.cursor()
        c.execute("""
            INSERT OR REPLACE INTO farms
            (farm_id, username, system_id, crop, location, lat, lon, soil_texture,
             row_cm, plant_cm, spacing, planting_date, compliance, yield_factor, om_pct, agent_json)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            farm["farm_id"], username, farm.get("system_id"),
            farm.get("crop"), farm.get("location"), farm.get("lat"), farm.get("lon"),
            farm.get("soil_texture"), farm.get("row_cm"), farm.get("plant_cm"),
            farm.get("spacing"), farm.get("planting_date"), farm.get("compliance"),
            farm.get("yield_factor", 1.0), farm.get("om_pct", 2.0),
            json.dumps(farm.get("agent", {}))
        ))
        conn.commit()


# ------------------------------
# Authentication
# ------------------------------
if "user" not in st.session_state:
    st.session_state.user = None

if st.session_state.user is None:
    st.title("👩‍🌾 Mytochondria Farmer Login")

    tab_login, tab_register = st.tabs(["Login", "Register"])

    with tab_login:
        uname = st.text_input("Username", key="login_user")
        pword = st.text_input("Password", type="password", key="login_pass")
        if st.button("Login"):
            user = find_user(uname, pword)
            if user:
                st.session_state.user = user
                st.success("Welcome back!")
                st.rerun()
            else:
                st.error("Invalid username or password.")

    with tab_register:
        new_user = st.text_input("Choose username", key="reg_user")
        new_pass = st.text_input("Choose password", type="password", key="reg_pass")
        new_email = st.text_input("Email", key="reg_email")
        if st.button("Register"):
            if user_exists(new_user):
                st.error("Username already exists.")
            else:
                create_user(new_user, new_pass, new_email)
                st.success("Account created! Please login.")
    st.stop()

user = st.session_state.user
user_farms = user.get("farms", [])

# ==== NPK unit mapping (demo-calibrated caps) ====
SUPPLY_CAP_KG_HA = {  # "100%" corresponds to this much plant-available nutrient
    "n": 120.0,  # kg N/ha
    "p": 60.0,  # kg P2O5/ha (demo cap)
    "k": 100.0,  # kg K2O/ha (demo cap)
}


def pct_to_kg(nutrient: str, pct: float) -> float:
    return max(0.0, (pct / 100.0) * SUPPLY_CAP_KG_HA[nutrient])


def kg_to_pct(nutrient: str, kg: float) -> float:
    cap = SUPPLY_CAP_KG_HA[nutrient]
    if cap <= 0: return 0.0
    return clamp((kg / cap) * 100.0, 0.0, 100.0)

def _card(title: str, value: str, sub: str = "", color: str = "gray", emoji: str = ""):
    html = f"""
    <div class="card {color}">
      <div class="title">{emoji and f'<span class="emoji">{emoji}</span>'}{title}</div>
      <div class="big">{value}</div>
      {f'<div class="sub">{sub}</div>' if sub else ''}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def _color_by_status(level: str) -> str:
    # level is "High" | "Medium" | "Low"
    return "green" if level == "High" else ("amber" if level == "Medium" else "red")


def _moisture_color(m: float) -> str:
    # <=30 red, 30-50 amber, >50 green
    return "red" if m < 30 else ("amber" if m < 50 else "green")


def _expected_days_to_harvest(crop: str) -> int:
    # simple demo durations (tune later)
    c = (crop or "").lower()
    return 110 if c == "maize" else (85 if c == "beans" else (120 if c == "rice" else 100))


# ==== Soil/management multipliers ====

FARMS = [
    dict(
        id="FARM-01", name="Mukuni South", lat=-17.858, lon=25.863, aer="I", province="Southern",
        size_ha=12, crop="maize", soil_texture="sand", om_pct=1.2, yield_factor=1.00,
        row_cm=75, plant_cm=25, agent=dict(compliance=0.50, delay_min_h=12, delay_max_h=48)
    ),
    dict(
        id="FARM-02", name="Kafubu Estate", lat=-12.968, lon=28.635, aer="IIa", province="Copperbelt",
        size_ha=8, crop="beans", soil_texture="loam", om_pct=2.5, yield_factor=1.05,
        row_cm=45, plant_cm=20, agent=dict(compliance=0.80, delay_min_h=0, delay_max_h=12)
    ),
    dict(
        id="FARM-03", name="Barotse Sands", lat=-15.254, lon=23.125, aer="IIb", province="Western",
        size_ha=15, crop="maize", soil_texture="sand", om_pct=1.0, yield_factor=0.95,
        row_cm=75, plant_cm=30, agent=dict(compliance=0.60, delay_min_h=6, delay_max_h=24)
    ),
    dict(
        id="FARM-04", name="Kasama Paddies", lat=-10.212, lon=31.180, aer="III", province="Northern",
        size_ha=10, crop="rice", soil_texture="clay", om_pct=3.0, yield_factor=1.10,
        row_cm=30, plant_cm=20, agent=dict(compliance=0.90, delay_min_h=0, delay_max_h=6)
    ),
    dict(
        id="FARM-05", name="Chipata Strips", lat=-13.636, lon=32.645, aer="IIa", province="Eastern",
        size_ha=9, crop="maize+beans", soil_texture="loam", om_pct=2.2, yield_factor=1.00,
        row_cm=75, plant_cm=25, agent=dict(compliance=0.70, delay_min_h=6, delay_max_h=24)
    ),
]

TEXTURE_UPTAKE_MULT = {  # effect on plant uptake demand
    "sand": 1.15, "loam": 1.00, "clay": 0.90
}
TEXTURE_LEACH_MULT = {  # effect on N leaching when it rains
    "sand": 1.30, "loam": 1.00, "clay": 0.70
}

# Very simple OM mineralization model (kg N/ha/day) — demo-level
# Rule of thumb-like placeholder: ~0.08 kg N/ha/day per 1% OM
N_MINERALIZE_KG_PER_DAY_PER_OM = 0.08

# Extra N loss on a rainy (leaching) day — base, scaled by texture
N_LEACH_KG_PER_HEAVY_RAIN = 1.2  # demo-safe base, adjust later

# Bounds for management multipliers
YIELD_FACTOR_MIN, YIELD_FACTOR_MAX = 0.8, 1.2
DENSITY_FACTOR_MIN, DENSITY_FACTOR_MAX = 0.8, 1.2

# ===== Baseline plant densities (demo defaults; tune to your agronomy) =====
BASE_PLANT_DENSITY_HA = {
    "maize": 53333,  # ~75 cm x 25 cm  → ~53,333 plants/ha
    "beans": 200000,  # example baseline
    "rice": 250000,  # ~20 cm x 20 cm  → ~250,000 plants/ha
}

# Clamp bounds for how far density can scale uptake (same ones you already use)
DENSITY_FACTOR_MIN, DENSITY_FACTOR_MAX = 0.8, 1.2

# --- Agro-ecological regions (Zambia) & typical soils (simplified) ---
AER = {
    "I": {"rain_mm": "<800", "soil_note": "Sandy-loam to sandy; low nutrient retention, drier south/west"},
    "IIa": {"rain_mm": "800–1000", "soil_note": "Loam to clay-loam; moderately leached; most productive"},
    "IIb": {"rain_mm": "800–1000", "soil_note": "Kalahari sands; strongly acidic, low water & nutrient holding"},
    "III": {"rain_mm": "1000–1500", "soil_note": "Loam to clay; humid north; higher leaching, weathered clays"},
}

# Province/town → (lat, lon, AER)
ZAMBIA_SITES = {
    "Lusaka (Lusaka Prov)": (-15.416, 28.283, "IIa"),
    "Ndola (Copperbelt)": (-12.968, 28.635, "IIa"),
    "Kitwe (Copperbelt)": (-12.818, 28.214, "IIa"),
    "Solwezi (North-Western)": (-12.173, 26.389, "IIa"),
    "Mongu (Western)": (-15.254, 23.125, "IIb"),
    "Livingstone (Southern)": (-17.858, 25.863, "I"),
    "Choma (Southern)": (-16.806, 26.953, "I"),
    "Chipata (Eastern)": (-13.636, 32.645, "IIa"),
    "Kasama (Northern)": (-10.212, 31.180, "III"),
    "Mansa (Luapula)": (-11.199, 28.894, "III"),
}

# --- Crop coefficients Kc by stage (very simplified FAO-56 style) ---
CROP_KC = {
    "maize": [
        ("initial", 0.35, 20),
        ("dev", 0.75, 25),
        ("mid", 1.15, 40),
        ("late", 0.80, 30),
    ],
    "beans": [
        ("initial", 0.40, 15),
        ("dev", 0.75, 20),
        ("mid", 1.05, 25),
        ("late", 0.80, 20),
    ],
    "rice": [
        ("initial", 1.05, 20),
        ("dev", 1.10, 25),
        ("mid", 1.20, 40),
        ("late", 0.90, 30),
    ],
}

# Default root-zone depth for kg/ha conversion (top 0–20 cm), bulk density slider will override
DEFAULT_DEPTH_M = 0.20

# Percent ↔ kg/ha caps you already use in Sensor tab (reuse if defined)
SUPPLY_CAP_KG_HA = {"n": 120.0, "p": 60.0, "k": 100.0}

# Soil texture multipliers you already use (reuse to stay consistent)
TEXTURE_UPTAKE_MULT = {"sand": 1.15, "loam": 1.00, "clay": 0.90}

# Open-Meteo endpoints & common hourly vars
OPEN_METEO = "https://api.open-meteo.com/v1/forecast"
HOURLY_VARS = "precipitation,et0_fao_evapotranspiration,temperature_2m_max"


def _split_mm(total_mm: float) -> str:
    """Make irrigation amounts farmer-friendly (split into 2–3 waterings)."""
    if total_mm <= 10:
        return f"{int(round(total_mm))} mm once"
    if total_mm <= 30:
        each = int(round(total_mm / 2))
        return f"{int(round(total_mm))} mm total (split: {each} mm × 2)"
    each = int(round(total_mm / 3))
    return f"{int(round(total_mm))} mm total (split: {each} mm × 3)"


def generate_simple_actions(p: Dict[str, Any], plan: Dict[str, Any], weekly: "pd.DataFrame",
                            risks: Dict[str, Any], df_daily: "pd.DataFrame") -> list[str]:
    """Turn the plan + risks into short, clear action sentences."""
    actions: list[str] = []

    # — Timing —
    actions.append(f"Plant on **{p['planting_date'].strftime('%b %d, %Y')}** if soil is moist (not waterlogged).")

    # — Irrigation (next 2 weeks) —
    wk = weekly.head(2).copy()
    names = ["This week", "Next week"]
    for i in range(len(wk)):
        irr = float(wk.iloc[i]["Irrigation_mm"])
        if irr > 0.5:
            actions.append(f"{names[i]}: **Irrigate** " + _split_mm(irr) + ". Water early morning or late evening.")
        else:
            actions.append(f"{names[i]}: **No irrigation** needed if rainfall arrives as forecast.")

    # — Erosion & heat/cold risks —
    if risks.get("erosion_days", 0) >= 1:
        actions.append("Heavy rain expected: **keep residue mulch** and use **contour ridges** to reduce erosion.")
    if risks.get("heat_stress_days", 0) >= 1 and p["crop"] == "maize":
        actions.append("Hot days ahead (≥35°C): **avoid water stress** from 1 week before to 2 weeks after tasseling.")
    if risks.get("cool_germination_days", 0) >= 1:
        actions.append(
            "Cool spell during emergence: **delay planting** or use **shallow planting** to improve germination.")

    # — pH / salinity —
    if p["ph"] < 5.5:
        actions.append("Soil is acidic: **apply agricultural lime** to move pH towards 6.0–6.5 before planting.")
    if p["ec"] >= 2.0:
        actions.append(
            "Salinity risk: **avoid KCl**, prefer sulfate forms; plan a **leaching irrigation** after heavy rain.")

    # — Nutrient plan —
    n_need = float(plan["N_rec_kg_ha"])
    p_need = float(plan["P2O5_rec_kg_ha"])
    k_need = float(plan["K2O_rec_kg_ha"])
    if n_need >= 5:
        split_n = max(0, int(round(n_need * 0.4)))
        topdress = max(0, int(round(n_need - split_n)))
        actions.append(
            f"Nitrogen: **{int(round(n_need))} kg/ha** total. Apply **{split_n} kg/ha at planting**, then **{topdress} kg/ha** at 4–6 weeks.")
        if p.get("om_pct", 0) < 2.0:
            actions.append(
                "Boost soil **organic matter** (compost/manure) to supply slow-release N and improve water holding.")
    else:
        actions.append("Nitrogen: **no extra N** required now (OM credit and soil N are adequate).")

    if p_need >= 10:
        actions.append(
            f"Phosphorus: apply **{int(p_need)} kg/ha P₂O₅** **at planting** (band near seed; don’t mix with urea).")
    else:
        actions.append("Phosphorus: **no extra P** needed for this season.")

    if k_need >= 10:
        actions.append(f"Potassium: apply **{int(k_need)} kg/ha K₂O** (use sulfate on saline soils).")
    else:
        actions.append("Potassium: **no extra K** needed now.")

    # — Spacing / density (already computed) —
    actions.append(f"Keep spacing to hit **{int(p['plants_ha']):,} plants/ha** for good canopy and yield potential.")

    # — Intercrop note —
    if p.get("crop2"):
        actions.append(
            f"Intercrop with **{p['crop2']}**: use **alternate rows** or a **1:1 strip**; keep fertilizer mainly with the main crop’s row.")

    # — Simple housekeeping —
    actions.append(
        "After each rain or irrigation, **check for crusting/ponding** and break crust lightly to help emergence.")
    actions.append("Keep **weeds below 10 cm**; early weeding saves water and nutrients for your crop.")

    return actions


@st.cache_data(ttl=60 * 30)
def fetch_weather(lat: float, lon: float, days_forward: int = 10, days_past: int = 7, tz: str = "Africa/Lusaka") -> \
Dict[str, Any]:
    """Fetch hourly/daily weather (past & next) from Open-Meteo (no key)."""
    # daily summaries are easier for water balances
    params = {
        "latitude": lat, "longitude": lon, "timezone": tz,
        "hourly": "precipitation,et0_fao_evapotranspiration,temperature_2m",
        "daily": "precipitation_sum,et0_fao_evapotranspiration,temperature_2m_max,temperature_2m_min",
        "forecast_days": max(1, min(16, days_forward)),
        "past_days": max(0, min(92, days_past)),
    }
    r = requests.get(OPEN_METEO, params=params, timeout=20)
    r.raise_for_status()
    return r.json()


def kc_for_day_since_planting(crop: str, dsp: int) -> float:
    """Rough Kc by growth stage timeline."""
    plan = CROP_KC.get(crop.lower(), CROP_KC["maize"])
    day_cursor = 0
    for _, kc, length in plan:
        if dsp <= day_cursor + length:
            return kc
        day_cursor += length
    return plan[-1][1]


def compute_density_factor(crop: str, row_cm: float, plant_cm: float, base_density_ha: Optional[float] = None) -> Tuple[
    float, float]:
    BASE_PLANT_DENSITY_HA = {"maize": 53333, "beans": 200000, "rice": 250000}
    base = base_density_ha or BASE_PLANT_DENSITY_HA.get((crop or "").lower(), 100000.0)
    row_m = max(0.0001, row_cm / 100.0);
    plant_m = max(0.0001, plant_cm / 100.0)
    plants_per_ha = 10000.0 / (row_m * plant_m)
    raw = plants_per_ha / base
    return (clamp(raw, 0.8, 1.2), plants_per_ha)


def mgkg_to_kgha(mg_per_kg: float, bulk_density_g_cm3: float, depth_m: float = DEFAULT_DEPTH_M) -> float:
    """Convert lab mg/kg to kg/ha for the sampling depth & bulk density."""
    # kg/ha = mg/kg × (soil mass per ha in kg); soil mass/ha = BD (t/m3)*1000 × depth (m) × 10,000 m2
    soil_mass_kg = (bulk_density_g_cm3 * 1000) * depth_m * 10_000  # e.g., 1.3*1000*0.2*10k = 2,600,000 kg/ha
    return max(0.0, mg_per_kg * soil_mass_kg / 1e6)


def kgha_to_mgkg(kg_ha: float, bulk_density_g_cm3: float, depth_m: float = DEFAULT_DEPTH_M) -> float:
    soil_mass_kg = (bulk_density_g_cm3 * 1000) * depth_m * 10_000
    return max(0.0, kg_ha * 1e6 / soil_mass_kg)


def irrigation_recommendations(crop: str, planting_date: date, daily: Dict[str, List], yield_factor: float,
                               density_factor: float) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Compute ETc, rain, deficit & simple weekly irrigation plan + risk flags."""
    dts = [pd.to_datetime(t).date() for t in daily["time"]]
    dsp = [(d - planting_date).days for d in dts]
    kc = [kc_for_day_since_planting(crop, x) for x in dsp]
    et0 = daily["et0_fao_evapotranspiration"]
    rain = daily["precipitation_sum"]
    tmax = daily["temperature_2m_max"]

    etc = [max(0.0, (et0[i] or 0.0) * kc[i] * yield_factor * density_factor) for i in range(len(dts))]
    deficit = [max(0.0, etc[i] - (rain[i] or 0.0)) for i in range(len(dts))]

    df = pd.DataFrame({
        "date": dts, "Kc": kc, "ET0_mm": et0, "Rain_mm": rain, "ETc_mm": etc, "Deficit_mm": deficit, "Tmax_C": tmax
    })
    df["week"] = df["date"].apply(lambda d: f"{d.isocalendar().year}-W{d.isocalendar().week:02d}")
    weekly = df.groupby("week", as_index=False).agg({
        "ETc_mm": "sum", "Rain_mm": "sum", "Deficit_mm": "sum"
    })
    weekly["Irrigation_mm"] = weekly["Deficit_mm"].apply(lambda x: max(0.0, round(x, 1)))

    risks = {
        "erosion_days": int((df["Rain_mm"] >= 30).sum()),  # very wet days
        "heat_stress_days": int((df["Tmax_C"] >= 35).sum()),
        "cool_germination_days": int(((df["Tmax_C"] < 18) & (pd.Series(dsp) <= 10)).sum()),
    }
    return df, {"weekly_plan": weekly, "risks": risks}


def nutrient_plan_from_lab(crop: str, yield_factor: float, om_pct: float,
                           n_kgha: float, p_mgkg: float, k_mgkg: float,
                           bd: float, depth_m: float = DEFAULT_DEPTH_M) -> Dict[str, Any]:
    """Very simple, region-agnostic plan: N need scales with yield; P,K if lab is low."""
    # Convert lab P,K mg/kg → kg/ha pool (rule-of-thumb)
    p_pool = mgkg_to_kgha(p_mgkg, bd, depth_m)
    k_pool = mgkg_to_kgha(k_mgkg, bd, depth_m)

    # Target ranges (demo; calibrate later per local recs)
    # N: 60–120 kg/ha, scale with yield factor; OM mineralization offsets ~ 0.08*OM%*days (growing season 110 d assumed)
    n_target = 90.0 * yield_factor  # mid-range
    n_om_credit = max(0.0, om_pct * 0.08 * 110)  # ~OM% * 8.8 kg/ha
    n_rec = max(0.0, n_target - n_kgha - n_om_credit)

    # P & K: if pools are small, recommend 30–60 kg/ha P2O5/K2O
    p_rec = 0.0 if p_pool >= 60 else (60 if p_pool < 30 else 30)
    k_rec = 0.0 if k_pool >= 100 else (60 if k_pool < 50 else 30)

    # Organic matter option: if pH < 5.5 or Kalahari sands (IIb), push OM/manure & liming
    return {
        "N_rec_kg_ha": round(n_rec, 1),
        "P2O5_rec_kg_ha": p_rec,
        "K2O_rec_kg_ha": k_rec,
        "notes": {
            "n_credit_om": round(n_om_credit, 1),
            "p_pool_kgha": round(p_pool, 1),
            "k_pool_kgha": round(k_pool, 1),
        }
    }


def _rng_for(*parts) -> random.Random:
    # deterministic RNG based on parts + GLOBAL_SEED
    s = "|".join(map(str, parts))
    return random.Random(hash(s) ^ GLOBAL_SEED)


def weather_hourly_df(lat: float, lon: float, days_past=7, days_forward=2, tz="Africa/Lusaka") -> pd.DataFrame:
    j = fetch_weather(lat, lon, days_forward=days_forward, days_past=days_past, tz=tz)
    # Open-Meteo hourly arrays align on index
    h = j["hourly"]
    df = pd.DataFrame({
        "ts": pd.to_datetime(h["time"]),
        "precip_mm": h.get("precipitation", [0] * len(h["time"])),
        "et0_mm": h.get("et0_fao_evapotranspiration", [0] * len(h["time"])),
        "t2m": h.get("temperature_2m", [None] * len(h["time"])),
    })
    df["ts"] = df["ts"].dt.tz_localize(None).dt.floor("H")
    return df


def ensure_farm_initialized(farm: Dict[str, Any]):
    fid = farm["farm_id"]
    if fid in st.session_state.farm_hist and st.session_state.farm_hist[fid]:
        return  # already initialized

    # Planting ~ one month ago
    planting_date = datetime.fromisoformat(farm.get("planting_date", str(datetime.now().date()))).date()
    st.session_state.farm_last_ts[fid] = datetime.combine(planting_date, datetime.min.time())

    # Start pools & environment
    moisture = 65.0
    ph = 6.5
    ec = 1.2
    n_kg = pct_to_kg("n", 70.0)
    p_kg = pct_to_kg("p", 65.0)
    k_kg = pct_to_kg("k", 60.0)
    temp = 24.0

    st.session_state.farm_hist[fid] = [{
        "timestamp": st.session_state.farm_last_ts[fid],
        "moisture": moisture, "temperature": temp,
        "ph": ph, "ec": ec,
        "n": kg_to_pct("n", n_kg), "p": kg_to_pct("p", p_kg), "k": kg_to_pct("k", k_kg),
        "planting_date": planting_date, "crop": farm["crop"]
    }]
    st.session_state.farm_pending_actions[fid] = []
    st.session_state.farm_alerts[fid] = []


def farm_density_factor(farm: Dict[str, Any]) -> float:
    crop = "maize" if farm["crop"] == "maize+beans" else farm["crop"]
    return compute_density_factor(crop, farm["row_cm"], farm["plant_cm"])[0]


def update_farm_until_now(farm: Dict[str, Any]):
    """Append missing hours for this farm up to the current hour, using real weather & agent effects."""
    ensure_farm_initialized(farm)
    fid = farm["farm_id"]
    hist = st.session_state.farm_hist[fid]
    last_ts = st.session_state.farm_last_ts[fid] or hist[-1]["timestamp"]
    now_hr = datetime.now().replace(minute=0, second=0, microsecond=0)

    if last_ts >= now_hr:
        return  # nothing to do

    # Build weather frame around the gap
    wx = weather_hourly_df(farm["lat"], farm["lon"], days_past=10, days_forward=2)
    wx = wx.set_index("ts")

    # Unpack latest state (convert NPK back to kg for the engine)
    s = hist[-1].copy()
    n_kg = pct_to_kg("n", s["n"]);
    p_kg = pct_to_kg("p", s["p"]);
    k_kg = pct_to_kg("k", s["k"])
    moisture = s["moisture"];
    ph = s["ph"];
    ec = s["ec"];
    day_mean = s.get("temperature", 24.0)

    # Management multipliers
    density_factor = farm_density_factor(farm)
    texture_mult = TEXTURE_UPTAKE_MULT[farm["soil_texture"]]
    weekly_mult = texture_mult * farm["yield_factor"] * density_factor

    # per-hour iterate
    cur = (last_ts + timedelta(hours=1))
    while cur <= now_hr:
        hour = cur.hour
        dsp = (cur.date() - s["planting_date"]).days
        # Daily 06:00 alerts & scheduling (once per day)
        if hour == 6:
            # generate daily alerts based on state
            todays_alerts = []
            # water alert
            if moisture < 30:
                todays_alerts.append(dict(type="water", title="Irrigate Today", mm=25))
            # N top-dress window for maize
            crop0 = ("maize" if farm["crop"].startswith("maize") else farm["crop"])
            if crop0 == "maize" and 28 <= dsp <= 56 and n_kg < 0.45 * SUPPLY_CAP_KG_HA["n"]:
                todays_alerts.append(dict(type="nitrogen", title="Top-dress N (~50 kg/ha)"))
            # lime if acidic
            if ph < 5.7:
                todays_alerts.append(dict(type="lime", title="Apply Lime"))
            # leach if EC high
            if ec >= 2.0:
                todays_alerts.append(dict(type="leach", title="Leach Salts"))

            # agent decision per alert (deterministic, but human-like)
            for a in todays_alerts:
                rng = _rng_for(fid, cur.date().isoformat(), a["type"])
                will_do = (rng.random() < farm["agent"]["compliance"])
                if will_do:
                    delay_h = rng.randint(farm["agent"]["delay_min_h"], farm["agent"]["delay_max_h"])
                    st.session_state.farm_pending_actions[fid].append(
                        dict(ts=cur + timedelta(hours=delay_h), type=a["type"], meta=a)
                    )
                st.session_state.farm_alerts[fid].append(
                    dict(day=cur.date().isoformat(), **a, status=("scheduled" if will_do else "missed"))
                )

        # Apply scheduled actions that “arrive” this hour
        pending = st.session_state.farm_pending_actions[fid]
        due_now = [x for x in pending if x["ts"] == cur]
        if due_now:
            for act in due_now:
                if act["type"] == "water":
                    mm = act["meta"].get("mm", 25)
                    moisture = clamp(moisture + 0.9 * (mm / 1.0), 0, 95)  # bump moisture; simple mm→% proxy
                    ec = clamp(ec - 0.05, 0.1, 3.0)
                elif act["type"] == "nitrogen":
                    n_kg = min(SUPPLY_CAP_KG_HA["n"], n_kg + 0.18 * SUPPLY_CAP_KG_HA["n"])
                    ec = clamp(ec + 0.35, 0.1, 3.0)
                elif act["type"] == "lime":
                    ph = clamp(ph + 0.15, 3.5, 9.0)
                elif act["type"] == "leach":
                    moisture = clamp(moisture + 10, 0, 95)
                    ec = clamp(ec - 0.30, 0.1, 3.0)

                if fid not in st.session_state.farm_actions_log:
                    st.session_state.farm_actions_log[fid] = []
                st.session_state.farm_actions_log[fid].append({
                    "ts": act["ts"], "type": act["type"],
                    "title": act["meta"].get("title", act["type"]).title(),
                    "status": "done_auto",
                    "effect": act["meta"]
                })
            # remove applied
            st.session_state.farm_pending_actions[fid] = [x for x in pending if x["ts"] != cur]

        # Weather-driven water balance this hour
        w = wx.loc[cur] if cur in wx.index else None
        et0 = float(w["et0_mm"]) if w is not None and pd.notna(w["et0_mm"]) else 0.0
        precip = float(w["precip_mm"]) if w is not None and pd.notna(w["precip_mm"]) else 0.0
        t2m = float(w["t2m"]) if w is not None and pd.notna(w["t2m"]) else day_mean

        # ETc scaling by Kc(stage) & management
        kc = kc_for_day_since_planting(("maize" if farm["crop"].startswith("maize") else farm["crop"]), dsp)
        etc_mm = et0 * kc * weekly_mult
        # convert to a % moisture draw (simple proxy; keep stable for demo)
        evap_pct = clamp(etc_mm * 0.10, 0.05, 0.35)  # 0.05–0.35 %/h
        moisture = clamp(moisture - evap_pct + (0.35 * precip), 0, 100)

        # EC concentrates on dry-down
        ec = clamp(ec + evap_pct * 0.02 - precip * 0.01, 0.1, 3.0)
        # pH slow drift
        ph = clamp(ph + (_rng_for(fid, cur.hour).random() - 0.5) * 0.01, 3.5, 9.0)

        # Daily nutrient balance at 06:05 (after alerts queued)
        if hour == 6:
            # OM mineralization
            n_kg += max(0.0, farm["om_pct"] * N_MINERALIZE_KG_PER_DAY_PER_OM)
            # Rain leaching (N) if it’s a wet day (sum > ~5 mm)
            day_slice = wx.loc[(wx.index.date == cur.date())] if not wx.empty else pd.DataFrame()
            wet_day = (day_slice["precip_mm"].sum() if not day_slice.empty else 0.0) > 5.0
            if wet_day:
                n_kg -= N_LEACH_KG_PER_HEAVY_RAIN * TEXTURE_LEACH_MULT[farm["soil_texture"]]
            # Plant uptake (weekly % → kg/day) with multipliers
            n_w, p_w, k_w = crop_uptake_weekly(("maize" if farm["crop"].startswith("maize") else farm["crop"]), dsp)
            n_uptake = (n_w / 7.0) / 100.0 * SUPPLY_CAP_KG_HA["n"] * weekly_mult
            p_uptake = (p_w / 7.0) / 100.0 * SUPPLY_CAP_KG_HA["p"] * weekly_mult
            k_uptake = (k_w / 7.0) / 100.0 * SUPPLY_CAP_KG_HA["k"] * weekly_mult
            n_kg = max(0.0, n_kg - n_uptake)
            p_kg = max(0.0, p_kg - p_uptake)
            k_kg = max(0.0, k_kg - k_uptake)

        # update day_mean towards t2m slowly
        day_mean = clamp(day_mean + (t2m - day_mean) * 0.1, 15, 36)

        # Append the hour
        point = {
            "timestamp": cur, "moisture": moisture, "temperature": t2m,
            "ph": ph, "ec": ec, "n": kg_to_pct("n", n_kg), "p": kg_to_pct("p", p_kg), "k": kg_to_pct("k", k_kg),
            "planting_date": s["planting_date"], "crop": farm["crop"]
        }
        hist.append(point)
        st.session_state.farm_last_ts[fid] = cur
        cur += timedelta(hours=1)


# ------------------------------
# Utility + Global Config
# ------------------------------

st.set_page_config(
    page_title="Mytochondria Soil Advisor ",
    page_icon="🌱",
    layout="wide",
)
st.markdown("""
<style>
/* ===================================================
   EMERALD GREEN THEME – Light & Dark Unified
   Fixes red slider track, active borders, highlights
   =================================================== */
:root {
  --accent: #059669;
  --accent-dark: #047857;
  --accent-light: #10b981;
  --bg-light: #ecfdf5;
  --bg-dark: #0b1b12;
  --text-light: #1e293b;
  --text-dark: #e2e8f0;
  --border-soft: rgba(5,150,105,0.2);
}

/* ===== GLOBAL LAYOUT ===== */
body, .stApp, .block-container {
  background: linear-gradient(to bottom, var(--bg-light), #ffffff) !important;
  color: var(--text-light) !important;
  font-family: 'Inter', sans-serif;
}
html[data-theme="dark"], body[data-theme="dark"], .stApp[data-theme="dark"] {
  background: linear-gradient(to bottom, var(--bg-dark), #000) !important;
  color: var(--text-dark) !important;
}
h1, h2, h3, h4 {
  color: var(--accent-dark) !important;
  font-weight: 700 !important;
}

/* ===== BUTTONS ===== */
.stButton > button, div[data-testid="stDownloadButton"] > button {
  background-color: var(--accent) !important;
  color: white !important;
  border: none !important;
  border-radius: 10px !important;
  padding: 0.6rem 1.2rem !important;
  font-weight: 600 !important;
  transition: all 0.25s ease;
  box-shadow: 0 4px 8px rgba(5,150,105,0.25);
}
.stButton > button:hover, div[data-testid="stDownloadButton"] > button:hover {
  background-color: var(--accent-dark) !important;
  transform: translateY(-1px);
}
.stButton > button:active { transform: scale(0.97); }

/* ===== NAVIGATION ===== */
div[data-testid="stHorizontalBlock"] div.stButton > button {
  background-color: white !important;
  color: var(--accent-dark) !important;
  border: 1px solid var(--accent) !important;
  border-radius: 10px !important;
  font-weight: 600 !important;
  transition: all 0.3s ease;
}
div[data-testid="stHorizontalBlock"] div.stButton > button:hover {
  background-color: var(--bg-light) !important;
}
div[data-testid="stHorizontalBlock"] div.stButton > button.nav-active {
  background-color: var(--accent) !important;
  color: white !important;
}

/* ===== INPUTS ===== */
input, textarea, select, .stDateInput input {
  background-color: #ffffff !important;
  color: var(--text-light) !important;
  border: 1px solid var(--border-soft) !important;
  border-radius: 8px !important;
  transition: border-color 0.3s ease;
}
input:focus, textarea:focus, select:focus {
  border-color: var(--accent) !important;
  box-shadow: 0 0 0 2px rgba(5,150,105,0.25) !important;
}

/* ===== LABELS ===== */
.stNumberInput label,
.stRadio label,
.stSelectbox label,
.stSlider label,
.stDateInput label {
  color: var(--accent-dark) !important;
  font-weight: 600 !important;
}

/* ===== NUMBER INPUTS ===== */
.stNumberInput, .stNumberInput > div, .stNumberInput input {
  background-color: #ffffff !important;
  color: var(--text-light) !important;
  border: 1px solid rgba(5,150,105,0.3) !important;
  border-radius: 8px !important;
}
.stNumberInput button {
  background-color: var(--bg-light) !important;
  color: var(--accent-dark) !important;
  border: 1px solid rgba(5,150,105,0.4) !important;
  border-radius: 6px !important;
}
.stNumberInput button:hover {
  background-color: #d1fae5 !important;
  border-color: var(--accent) !important;
}

/* ===== SLIDERS ===== */
[data-testid="stSlider"] [role="slider"] {
  background-color: var(--accent) !important;
  border: 2px solid var(--accent) !important;
  box-shadow: 0 0 6px rgba(5,150,105,0.6) !important;
}
[data-testid="stSlider"] > div > div {
  background: linear-gradient(to right, var(--accent), var(--accent-light)) !important;
}
[data-testid="stSlider"] [data-baseweb="slider"] div[role="presentation"] > div {
  background: linear-gradient(to right, var(--accent), var(--accent-light)) !important;
}
[data-testid="stSlider"] .stSliderValue, 
[data-testid="stSlider"] .css-1qrvfrg, 
[data-testid="stSlider"] .css-14xtw13 {
  color: var(--accent-dark) !important;
  font-weight: 600 !important;
}

/* ===== TABLES ===== */
div[data-testid="stDataFrame"] {
  border-radius: 14px !important;
  border: 1px solid var(--border-soft) !important;
  box-shadow: 0 4px 14px rgba(5,150,105,0.08) !important;
  overflow: hidden !important;
}
div[data-testid="stDataFrame"] thead tr th {
  background-color: var(--accent) !important;
  color: white !important;
  font-weight: 600 !important;
}
div[data-testid="stDataFrame"] tbody tr:hover td {
  background-color: rgba(5,150,105,0.07) !important;
}

/* ===== CARDS ===== */
.card, .metric {
  background: #ffffff !important;
  border: 1px solid var(--border-soft) !important;
  border-radius: 16px !important;
  box-shadow: 0 8px 20px rgba(2,6,23,0.05) !important;
  padding: 14px 16px !important;
  color: var(--text-light) !important;
}
.card .title { font-weight: 600; font-size: 0.9rem; }
.card .big { font-size: 1.4rem; font-weight: 700; color: var(--accent-dark); }

/* ===== ALERT BOXES ===== */
.stInfo, .stSuccess, .stWarning, .stError {
  border-radius: 12px !important;
  border-left: 5px solid var(--accent) !important;
  box-shadow: 0 2px 10px rgba(5,150,105,0.1);
}

/* ===== CHARTS ===== */
.stAltairChart, .stPlotlyChart {
  background-color: #ffffff !important;
  border-radius: 16px !important;
  border: 1px solid var(--border-soft) !important;
  box-shadow: 0 8px 20px rgba(2,6,23,0.06) !important;
  padding: 10px !important;
}
.vega-bindings, .vega-bindings label {
  color: var(--text-light) !important;
}
.vega-tooltip {
  background-color: white !important;
  border: 1px solid var(--accent-light) !important;
  border-radius: 6px !important;
}

/* ===== EXPANDERS ===== */
.streamlit-expanderHeader {
  font-weight: 600 !important;
  border: 1px solid var(--accent) !important;
  border-radius: 10px !important;
  background-color: #ffffff !important;
  color: var(--accent-dark) !important;
}

/* ===== SCROLLBAR ===== */
::-webkit-scrollbar { width: 8px; }
::-webkit-scrollbar-thumb {
  background-color: rgba(5,150,105,0.4);
  border-radius: 4px;
}
::-webkit-scrollbar-thumb:hover {
  background-color: rgba(5,150,105,0.7);
}

/* ===== DARK MODE OVERRIDES ===== */
html[data-theme="dark"] body, html[data-theme="dark"] .stApp {
  background: linear-gradient(to bottom, var(--bg-dark), #000) !important;
  color: var(--text-dark) !important;
}
html[data-theme="dark"] [data-testid="stSlider"] [role="slider"] {
  background-color: var(--accent-light) !important;
  border-color: var(--accent-light) !important;
}
html[data-theme="dark"] [data-testid="stSlider"] > div > div {
  background: linear-gradient(to right, var(--accent-light), var(--accent-dark)) !important;
}
html[data-theme="dark"] .stDataFrame thead tr th {
  background-color: var(--accent-dark) !important;
}
html[data-theme="dark"] .card, html[data-theme="dark"] .stAltairChart {
  background-color: #0f1f17 !important;
  border-color: rgba(16,185,129,0.3) !important;
  color: var(--text-dark) !important;
}
html[data-theme="dark"] .streamlit-expanderHeader {
  background-color: #0b1b12 !important;
  color: var(--accent-light) !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
/* === FIX PATCH: BUTTONS, SLIDERS, CHECKMARKS === */

/* Fix black buttons (Save confirmations etc.) */
button[kind="secondary"], button[kind="primary"], .stCheckbox button, .stDownloadButton > button {
  background-color: var(--accent) !important;
  color: #fff !important;
  border: none !important;
  border-radius: 10px !important;
  font-weight: 600 !important;
  box-shadow: 0 4px 8px rgba(5,150,105,0.25) !important;
}
button[kind="secondary"]:hover, button[kind="primary"]:hover {
  background-color: var(--accent-dark) !important;
}

/* Fix checkboxes (ticks) */
.stCheckbox [data-testid="stMarkdownContainer"] p {
  color: var(--text-light) !important;
}
.stCheckbox input[type="checkbox"]:checked {
  accent-color: var(--accent) !important;
}
input[type="checkbox"]:checked::before {
  background-color: var(--accent) !important;
  border-color: var(--accent-dark) !important;
}

/* Fix slider track + thumb */
[data-testid="stSlider"] [role="slider"] {
  background-color: white !important;
  border: 2px solid var(--accent) !important;
  box-shadow: 0 0 8px rgba(5,150,105,0.4) !important;
}
[data-testid="stSlider"] > div > div {
  background: linear-gradient(to right, var(--accent-light), var(--accent)) !important;
}
[data-testid="stSlider"] [data-baseweb="slider"] div[role="presentation"] > div {
  background: linear-gradient(to right, var(--accent-light), var(--accent)) !important;
}
[data-testid="stSlider"] .stSliderValue {
  color: var(--accent-dark) !important;
  font-weight: 700 !important;
}

/* Make slider labels visible and soft */
[data-testid="stSlider"] label {
  color: var(--accent-dark) !important;
  font-weight: 600 !important;
}

/* Light theme checkbox focus highlight */
input[type="checkbox"]:focus {
  outline: 2px solid rgba(5,150,105,0.4) !important;
  border-radius: 4px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
/* ===================================================
   FINAL PATCH — Fix black 'Save' buttons in forms
   Ensures all Streamlit buttons follow emerald theme
   =================================================== */

/* Default buttons (primary, secondary, form submit, etc.) */
button[kind="primary"], button[kind="secondary"],
.stButton > button, div[data-testid="stButton"] > button,
div[data-testid="stFormSubmitButton"] > button,
div[data-testid="stDownloadButton"] > button {
    background-color: var(--accent) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    box-shadow: 0 4px 10px rgba(5,150,105,0.25) !important;
    transition: all 0.25s ease-in-out;
}

/* Hover + active */
button[kind="primary"]:hover, button[kind="secondary"]:hover,
div[data-testid="stButton"] > button:hover,
div[data-testid="stFormSubmitButton"] > button:hover {
    background-color: var(--accent-dark) !important;
    transform: translateY(-1px);
}

button[kind="primary"]:active, button[kind="secondary"]:active,
div[data-testid="stButton"] > button:active,
div[data-testid="stFormSubmitButton"] > button:active {
    background-color: var(--accent-light) !important;
    color: white !important;
    transform: scale(0.98);
}

/* Fix for forms inside expanders */
div[data-testid="stFormSubmitButton"] > button {
    width: 100% !important;
    margin-top: 6px !important;
}

/* Ensure disabled buttons still readable */
button[disabled] {
    background-color: rgba(5,150,105,0.3) !important;
    color: rgba(255,255,255,0.7) !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
/* ===================================================
   FIX – Strong yellow background for warnings
   Works with all Streamlit versions
   =================================================== */

/* Match both legacy and new alert containers */
div[data-testid="stAlert"][role="alert"],
.stAlert, .stAlert-warning {
  background-color: #fbbf24 !important;    /* deeper amber */
  border-left: 6px solid #ca8a04 !important;
  color: #000 !important;
}

/* Make every child element inside readable */
div[data-testid="stAlert"][role="alert"] *,
.stAlert *, .stAlert-warning * {
  color: #000 !important;
}

/* Optional: subtle shadow for contrast */
div[data-testid="stAlert"][role="alert"] {
  box-shadow: 0 2px 8px rgba(0,0,0,0.08);
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
/* ===================================================
   PATCH – Make checkbox and radio labels always visible
   =================================================== */

/* Light + dark mode safe label color */
.stCheckbox label, .stRadio label {
    color: var(--text-light) !important;
    font-weight: 500 !important;
}

/* For dark mode overrides */
html[data-theme="dark"] .stCheckbox label,
html[data-theme="dark"] .stRadio label {
    color: var(--text-dark) !important;
}

/* Ensure the descriptive text (like 'kg/ha available N') is visible */
.stCheckbox div[data-testid="stMarkdownContainer"] p,
.stRadio div[data-testid="stMarkdownContainer"] p {
    color: var(--text-light) !important;
}

/* Explicitly handle focus and hover */
.stCheckbox label:hover, .stRadio label:hover {
    color: var(--accent-dark) !important;
}
</style>
""", unsafe_allow_html=True)

# --- Top Navigation Bar ---
st.markdown(
    """
    <style>
    div[data-testid="stHorizontalBlock"] div.stButton > button {
        display: inline-block;
        font-weight: 600;
        border-radius: 6px;
        padding: 8px 0;
        margin: 0 4px;
        cursor: pointer;
        border: 1px solid #2d572c;
        background-color: white;
        color: #2d572c;
        transition: all 0.3s ease;
    }
    div[data-testid="stHorizontalBlock"] div.stButton > button:hover {
        background-color: #eaf4ea;
    }
    div[data-testid="stHorizontalBlock"] div.stButton > button.nav-active {
        background-color: #2d572c !important;
        color: white !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)



nav_cols = st.columns(len(tab_names))
for i, name in enumerate(tab_names):
    if nav_cols[i].button(name, key=f"nav_{i}", use_container_width=True):
        st.session_state.active_tab = i
        st.rerun()

# Apply active class to highlight selected
active_idx = st.session_state.active_tab
st.markdown(
    f"""
    <script>
    const buttons = window.parent.document.querySelectorAll('div[data-testid="stHorizontalBlock"] div.stButton > button');
    buttons.forEach((btn, idx) => {{
        if (idx === {active_idx}) {{
            btn.classList.add("nav-active");
        }} else {{
            btn.classList.remove("nav-active");
        }}
    }});
    </script>
    """,
    unsafe_allow_html=True,
)
# ------------------------------
# State initialization
# ------------------------------

if "planting_date" not in st.session_state:
    st.session_state.planting_date = datetime.now().date()

if "sensor_history" not in st.session_state:
    st.session_state.sensor_history: List[Dict[str, Any]] = []

if "last_sensor" not in st.session_state:
    st.session_state.last_sensor: Optional[Dict[str, Any]] = None

if "checklist" not in st.session_state:
    # stores {task_id: {"label": str, "done": bool, "applied_effect": bool}}
    st.session_state.checklist: Dict[str, Dict[str, Any]] = {}

if "manual_latest" not in st.session_state:
    st.session_state.manual_latest: Optional[Dict[str, Any]] = None

if "alert_history" not in st.session_state:
    st.session_state.alert_history = []

# ---- Multi-farm live state ----
if "farm_hist" not in st.session_state:
    # farm_hist[farm_id] = list of hourly dict rows (same schema as your sensor_history points)
    st.session_state.farm_hist: Dict[str, List[Dict[str, Any]]] = {}

if "farm_last_ts" not in st.session_state:
    # farm_last_ts[farm_id] = last appended hour (datetime, hour-rounded)
    st.session_state.farm_last_ts: Dict[str, Optional[datetime]] = {}

if "farm_pending_actions" not in st.session_state:
    # farm_pending_actions[farm_id] = list of scheduled actions with ts and effect
    st.session_state.farm_pending_actions: Dict[str, List[Dict[str, Any]]] = {}

if "farm_alerts" not in st.session_state:
    # farm_alerts[farm_id] = list of daily alerts issued at ~06:00 (for log)
    st.session_state.farm_alerts: Dict[str, List[Dict[str, Any]]] = {}

if "farm_actions_log" not in st.session_state:
    # farm_actions_log[farm_id] = list of dicts: {ts, type, title, status, effect}
    st.session_state.farm_actions_log: Dict[str, List[Dict[str, Any]]] = {}
if "farm_actions_confirm" not in st.session_state:
    # farmer ticks: set of (farm_id, ts_iso, type)
    st.session_state.farm_actions_confirm: set[tuple] = set()
# Seed the world: use a fixed seed so behavior is repeatable for the demo
GLOBAL_SEED = 424242


# ------------------------------
# Data Generators (ported from dataGenerators.ts)  :contentReference[oaicite:2]{index=2}
# ------------------------------

def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def pct_to_cat(x: float) -> str:
    return "High" if x >= 60 else ("Medium" if x >= 30 else "Low")


# crop-stage weekly depletion (simple agronomy patterns)
def crop_uptake_weekly(crop: str, days_since_planting: int) -> Tuple[float, float, float]:
    w = max(0, days_since_planting // 7)
    c = (crop or "").strip().lower()
    if c == "maize":
        n = 5.0 if 4 <= w <= 8 else 2.5
        p = 2.0 if w <= 4 else 1.0
        k = 2.5 if 6 <= w <= 10 else 1.5
    elif c == "beans":
        n, p, k = 1.0, 1.5, 2.0
    elif c == "rice":
        n = 3.0 if 3 <= w <= 8 else 1.5
        p = 1.0
        k = 2.5 if 3 <= w <= 8 else 1.5
    else:
        n, p, k = 2.0, 1.2, 1.8
    return n, p, k


def record_alert_history(mode: str, alerts: List[Dict[str, Any]]):
    ts = datetime.now().isoformat(timespec="seconds")
    for a in alerts:
        st.session_state.alert_history.append({
            "time": ts, "mode": mode,
            "title": a["title"], "type": a["type"],
            "priority": a["priority"], "action": a.get("action", "")
        })
    # trim
    if len(st.session_state.alert_history) > 500:
        st.session_state.alert_history = st.session_state.alert_history[-500:]


def simulate_future_series(last_point: Dict[str, Any], planting_date: date, crop: str, days: int) -> list:
    series = []
    cur = last_point.copy()
    cur_ts = datetime.now()
    for d in range(1, days + 1):
        cur_ts = cur_ts + timedelta(days=1)
        # simple rain/leach vs dry-down
        is_rain = random.random() < 0.18
        if is_rain:
            cur["moisture"] = clamp(cur["moisture"] + 10 + random.random() * 6, 0, 95)
            cur["ec"] = clamp(cur["ec"] - 0.2, 0.1, 3.0)
        else:
            cur["moisture"] = clamp(cur["moisture"] - (1.5 + random.random() * 1.5), 0, 95)

        # temp seasonal wiggle + noise
        cur["temperature"] = clamp(24 + math.sin(d * math.pi / 14) * 3 + (random.random() - 0.5) * 3, 15, 35)
        # slow pH drift
        cur["ph"] = clamp(cur["ph"] + (random.random() - 0.5) * 0.05, 3.5, 9.0)

        # crop-stage depletion
        days_since_planting = (cur_ts.date() - planting_date).days
        n_w, p_w, k_w = crop_uptake_weekly(crop, days_since_planting)
        cur["n"] = clamp(cur["n"] - n_w / 7, 0, 100)
        cur["p"] = clamp(cur["p"] - p_w / 7, 0, 100)
        cur["k"] = clamp(cur["k"] - k_w / 7, 0, 100)

        series.append({
            "timestamp": cur_ts,
            "moisture": cur["moisture"],
            "temperature": cur["temperature"],
            "ph": cur["ph"],
            "ec": cur["ec"],
            "n": cur["n"],
            "p": cur["p"],
            "k": cur["k"],
        })
    return series


def gen_sensor_point(ts: datetime) -> Dict[str, Any]:
    """
    Generate one realistic hourly sensor point.
    Mirrors the TS behavior:
      - temperature follows daily cycle
      - gradual moisture/EC/pH drift
      - N, P, K gradual depletion with noise
    """
    hour = ts.hour
    base_temp = 24 + math.sin((hour - 6) * math.pi / 12) * 5

    last = st.session_state.last_sensor
    if last:
        data = {
            "timestamp": ts,
            "moisture": clamp(last["moisture"] + (random.random() - 0.5) * 2, 10, 95),
            "temperature": clamp(base_temp + (random.random() - 0.5) * 2, 15, 35),
            "ph": clamp(last["ph"] + (random.random() - 0.5) * 0.1, 3.5, 9.0),
            "ec": clamp(last["ec"] + (random.random() - 0.5) * 0.1, 0.1, 3.0),
            # Gradual depletion signals
            "n": clamp(last["n"] - 0.5 + (random.random() - 0.5) * 1, 5, 95),
            "p": clamp(last["p"] - 0.2 + (random.random() - 0.5) * 0.5, 5, 95),
            "k": clamp(last["k"] - 0.3 + (random.random() - 0.5) * 0.8, 5, 95),
        }
        st.session_state.last_sensor = data
        return data

    # first point
    data = {
        "timestamp": ts,
        "moisture": 65 + random.random() * 20,
        "temperature": base_temp,
        "ph": 6.5 + (random.random() - 0.5) * 1.5,
        "ec": 1.2 + (random.random() - 0.5) * 0.8,
        "n": 60 + random.random() * 30,
        "p": 50 + random.random() * 40,
        "k": 55 + random.random() * 35,
    }
    st.session_state.last_sensor = data
    return data


def gen_demo_series(days: int = 30) -> List[Dict[str, Any]]:
    """
    30-day realistic series:
      - 15% rainy days -> moisture + leach (EC down)
      - fertilizer every 14 days -> N,P,K bumps + EC up
      - natural depletion & slow pH drift
    Ported from generateDemoData in TS.  :contentReference[oaicite:3]{index=3}
    """
    data = []
    start = datetime.now().date() - timedelta(days=days)
    currentMoisture = 70.0
    currentTemp = 24.0
    currentPh = 6.8
    currentEc = 1.3
    currentN = 80.0
    currentP = 75.0
    currentK = 70.0

    for i in range(days):
        d = start + timedelta(days=i)
        isRainy = random.random() < 0.15
        isFert = (i % 14 == 0 and i > 0)

        if isRainy:
            currentMoisture = clamp(currentMoisture + 15 + random.random() * 10, 0, 95)
            currentEc = clamp(currentEc - 0.3, 0.1, 3.0)
        else:
            currentMoisture = clamp(currentMoisture - 2 - random.random() * 2, 0, 95)

        if isFert:
            currentN = clamp(currentN + 20, 0, 95)
            currentP = clamp(currentP + 15, 0, 95)
            currentK = clamp(currentK + 18, 0, 95)
            currentEc = clamp(currentEc + 0.5, 0.1, 3.0)

        # depletion & noise
        currentN = clamp(currentN - (1.2 + random.random() * 0.8), 10, 100)
        currentP = clamp(currentP - (0.5 + random.random() * 0.3), 10, 100)
        currentK = clamp(currentK - (0.8 + random.random() * 0.5), 10, 100)

        currentTemp = 24 + math.sin(i * math.pi / 30) * 3 + (random.random() - 0.5) * 4
        currentPh += (random.random() - 0.5) * 0.05
        currentPh = clamp(currentPh, 5.5, 8.0)

        data.append({
            "timestamp": datetime(d.year, d.month, d.day),
            "moisture": currentMoisture,
            "temperature": currentTemp,
            "ph": currentPh,
            "ec": currentEc,
            "n": currentN,
            "p": currentP,
            "k": currentK,
        })
    return data


# ------------------------------
# Depletion forecast utilities
# ------------------------------

def forecast_depletion_days(current_pct: float, weekly_drop_pct: float, floor_pct: float = 30) -> Optional[int]:
    if current_pct <= floor_pct:
        return 0
    if weekly_drop_pct <= 0:
        return None
    weeks = (current_pct - floor_pct) / weekly_drop_pct
    return max(0, int(round(weeks * 7)))


# ------------------------------
# Rule Engine (ported + expanded)  :contentReference[oaicite:5]{index=5}
# ------------------------------

def generate_insights(input_row: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Port of generateInsights from TS plus expansions:
      - water, pH, fertilizer (N/P/K), disease
      - forecast outputs with dates
    """
    out: List[Dict[str, Any]] = []
    planting = input_row["planting_date"]
    days_from_planting = (datetime.now().date() - planting).days
    crop = input_row.get("crop", "").strip().lower()

    # --- Water management
    if input_row["moisture"] < 30:
        out.append({
            "type": "water", "priority": "high",
            "title": "Low Soil Moisture Alert",
            "description": f"Soil moisture is critically low at {input_row['moisture']:.1f}%. Immediate irrigation recommended.",
            "action": "Apply ~25–30 mm of water via irrigation", "icon": "💧"
        })
    elif input_row["moisture"] < 50:
        out.append({
            "type": "water", "priority": "medium",
            "title": "Monitor Soil Moisture",
            "description": f"Soil moisture at {input_row['moisture']:.1f}% is below optimal. Consider irrigation soon.",
            "action": "Plan irrigation within 24–48 hours", "icon": "💧"
        })
    elif input_row["moisture"] > 85:
        out.append({
            "type": "water", "priority": "medium",
            "title": "Excess Moisture Warning",
            "description": f"High soil moisture ({input_row['moisture']:.1f}%) may lead to root rot or fungal diseases.",
            "action": "Improve drainage and reduce irrigation", "icon": "🌧️"
        })

    # --- pH management
    if input_row["ph"] < 6.0:
        out.append({
            "type": "ph", "priority": "high",
            "title": "Acidic Soil Detected",
            "description": f"Soil pH is {input_row['ph']:.1f}, which is too acidic for optimal nutrient uptake.",
            "action": "Apply agricultural lime at 2–3 t/ha", "icon": "⚗️"
        })
    elif input_row["ph"] > 8.0:
        out.append({
            "type": "ph", "priority": "high",
            "title": "Alkaline Soil Warning",
            "description": f"Soil pH is {input_row['ph']:.1f}, limiting nutrient availability.",
            "action": "Apply elemental sulfur or organic matter to lower pH", "icon": "⚗️"
        })

    # --- Nutrient management
    if input_row["n"] < 30:
        out.append({
            "type": "fertilizer", "priority": "high",
            "title": "Nitrogen Deficiency Critical",
            "description": f"Nitrogen levels are critically low ({input_row['n']:.0f}%). Plants may show yellowing leaves.",
            "action": "Apply nitrogen fertilizer now (e.g., ~50 kg/ha urea)", "icon": "🍃"
        })
    if input_row["p"] < 25:
        out.append({
            "type": "fertilizer", "priority": "medium",
            "title": "Phosphorus Below Optimal",
            "description": f"Phosphorus at {input_row['p']:.0f}% may limit root development and flowering.",
            "action": "Apply P fertilizer (e.g., ~30 kg/ha DAP)", "icon": "🫐"
        })
    if input_row["k"] < 30:
        out.append({
            "type": "fertilizer", "priority": "medium",
            "title": "Potassium Needs Attention",
            "description": f"Potassium at {input_row['k']:.0f}% is low, affecting disease resistance.",
            "action": "Apply K fertilizer (e.g., ~25 kg/ha KCl)", "icon": "🛡️"
        })

    # --- Crop-specific forecast (add dates)
    n_week, p_week, k_week = crop_uptake_weekly(crop, days_from_planting)

    n_days = forecast_depletion_days(input_row["n"], n_week)
    p_days = forecast_depletion_days(input_row["p"], p_week)
    k_days = forecast_depletion_days(input_row["k"], k_week)

    def add_forecast(name: str, days_left: Optional[int], pre_msg: str):
        if days_left is None:
            return
        target_date = datetime.now().date() + timedelta(days=days_left)
        when_txt = "now" if days_left == 0 else target_date.strftime("%b %d, %Y")
        out.append({
            "type": "forecast", "priority": "medium",
            "title": f"{name} Depletion Forecast",
            "description": f"{pre_msg} Estimated to reach LOW around {when_txt}.",
            "action": f"Plan application ~{max(days_left - 7, 0)} days before that date",
            "icon": "📅"
        })

    add_forecast("Nitrogen", n_days, "Based on crop uptake patterns, nitrogen will fall to LOW.")
    add_forecast("Phosphorus", p_days, "Phosphorus reserve trending down.")
    add_forecast("Potassium", k_days, "Potassium reserve trending down.")

    # --- Disease risk
    if input_row["moisture"] > 70 and input_row["temperature"] > 25:
        out.append({
            "type": "disease", "priority": "medium",
            "title": "Fungal Disease Risk",
            "description": "High moisture and temperature create favorable conditions for fungal diseases.",
            "action": "Scout fields, improve drainage; consider preventive fungicide per label",
            "icon": "🦠"
        })

    # Sort by priority
    priority_score = {"high": 3, "medium": 2, "low": 1}
    out.sort(key=lambda r: priority_score[r["priority"]], reverse=True)
    return out


# ------------------------------
# Checklist apply-effects
# ------------------------------
def compute_health_score(farm, latest, fid):
    """
    Compute a farm health score using real-time soil indicators and alerts.
    Inputs:
        farm: dict (farm metadata)
        latest: dict (latest reading from st.session_state.farm_hist[fid])
        fid: farm_id (used for alerts)
    Output:
        float in [0, 1]
    """

    # --- Normalize key indicators ---
    # Moisture (ideal 40–70%)
    m = latest.get("moisture", 0)
    if m < 30:
        norm_m = 0.3
    elif m > 85:
        norm_m = 0.5
    else:
        norm_m = (m - 30) / (70 - 30)  # 0 at 30%, 1 at 70%
        norm_m = max(0, min(norm_m, 1))

    # Nutrients (average of N,P,K availability, %)
    n = latest.get("n", 0)
    p = latest.get("p", 0)
    k = latest.get("k", 0)
    avg_npk = (n + p + k) / 3
    norm_npk = max(0, min(avg_npk / 100, 1))

    # pH (ideal 6–7.5)
    ph = latest.get("ph", 6.5)
    norm_ph = 1 - abs(ph - 6.75) / 2  # deviation penalty
    norm_ph = max(0, min(norm_ph, 1))

    # EC (salinity; ideal 0.2–1.5 dS/m)
    ec = latest.get("ec", 1.0)
    if ec < 0.2:
        norm_ec = 0.8  # too low → less fertile but not terrible
    elif ec > 3.0:
        norm_ec = 0.2
    else:
        norm_ec = 1 - (abs(ec - 1.0) / 2.0)
        norm_ec = max(0, min(norm_ec, 1))

    # --- Depletion rate penalty ---
    # Compare last 24h NPK trend (if enough data)
    hist = st.session_state.farm_hist.get(fid, [])
    if len(hist) > 24:
        last_day = hist[-24:]
        delta_n = last_day[-1]["n"] - last_day[0]["n"]
        delta_p = last_day[-1]["p"] - last_day[0]["p"]
        delta_k = last_day[-1]["k"] - last_day[0]["k"]
        avg_drop = -(delta_n + delta_p + delta_k) / 3.0
        if avg_drop > 5:
            depletion_penalty = 0.2
        elif avg_drop > 2:
            depletion_penalty = 0.1
        else:
            depletion_penalty = 0.0
    else:
        depletion_penalty = 0.0

    # --- Alerts penalty ---
    alerts = len(st.session_state.farm_alerts.get(fid, []))
    alert_penalty = min(alerts * 0.02, 0.2)  # 10 alerts max penalty 0.2

    # --- Weighted combination ---
    w_m, w_npk, w_ph, w_ec = 0.35, 0.35, 0.15, 0.15
    base = (w_m * norm_m) + (w_npk * norm_npk) + (w_ph * norm_ph) + (w_ec * norm_ec)
    base = base * (1 - depletion_penalty) * (1 - alert_penalty)

    # Scale into 0.4–1.0
    health_score = 0.4 + 0.6 * base
    return round(max(0, min(health_score, 1.0)), 2)

def apply_action_effects(state: Dict[str, Any], task_label: str):
    """
    When a user ticks a checklist item, adjust the model state for next insight calc.
    Simple MVP effects:
      - "Apply agricultural lime" -> pH +0.3 over time (here: immediate for demo)
      - "Apply nitrogen fertilizer" -> N +15%, EC +0.3
      - "Improve drainage" -> lower moisture by small amount if very wet
    """
    label = task_label.lower()

    # apply lime
    if "lime" in label:
        state["ph"] = clamp(state["ph"] + 0.3, 3.5, 9.0)

    # apply nitrogen
    if "nitrogen" in label or "urea" in label or "top-dress" in label:
        state["n"] = clamp(state["n"] + 15, 0, 100)
        state["ec"] = clamp(state["ec"] + 0.3, 0.1, 3.0)

    # drainage
    if "drainage" in label or "reduce irrigation" in label:
        if state["moisture"] > 80:
            state["moisture"] = clamp(state["moisture"] - 5, 0, 100)


# ------------------------------
# UI Components
# ------------------------------

AER_TEXTURE_DEFAULT = {
    "I": "sand",
    "IIa": "loam",
    "IIb": "sand",
    "III": "clay"
}

CROP_SPACING_DEFAULT = {
    "maize": (75, 25),
    "beans": (45, 20),
    "rice": (30, 20),
    "maize+beans": (75, 25)
}

active = tab_names[st.session_state.active_tab]

if active == "Home":

    st.markdown("""
            <div style='padding:20px; border-radius:16px; background:linear-gradient(to right,#ecfdf5,#ffffff); 
            box-shadow:0 4px 12px rgba(0,0,0,0.05); border:1px solid rgba(5,150,105,0.2);'>
                <h2 style='margin-bottom:6px;'>Welcome to your Farmer Portal</h2>
                <p>Use <b>imagery</b> or <b>Mytochondria smart sensors</b> to get actionable soil & crop recommendations.<br>
                You can switch or upgrade anytime.</p>
            </div>
        """, unsafe_allow_html=True)
    st.write("")
    # --- Existing alerts at the top ---
    st.markdown("### 🚨 Active Alerts")
    if not user_farms:
        st.info("No farms yet. Add a farm to start receiving alerts.")
    else:
        for f in user_farms:
            fid = f["farm_id"]
            alerts = st.session_state.farm_alerts.get(fid, [])
            if alerts:
                for a in alerts[-3:]:
                    st.warning(f"**{f['crop']} ({f['location']})** → {a['title']} ({a['status']})")
            else:
                st.success(f"No new alerts for **{f['crop']} ({f['location']})**")

    st.divider()

    # --- Onboarding Banner ---


    c1, c2 = st.columns([2, 1])
    with c1:
        colA, colB = st.columns(2)
        with colA:
            if st.button("📡 I have sensors — show live data"):
                st.session_state.active_tab = 1
                st.rerun()
        with colB:
            st.button("🖼️ I'll upload images", key="btnImagery")
            st.session_state.active_tab = 4  
            st.rerun()
        st.caption("💡 Tip: For best imagery results use natural light and include a coin or ruler for scale.")

    with c2:
        st.markdown("""
            <div style='padding:16px; border-radius:12px; border:1px solid rgba(5,150,105,0.2); background:white; box-shadow:0 2px 8px rgba(0,0,0,0.05); text-align:center'>
                <b>Sensors recommended</b><br>
                <span style='font-size:24px; color:#059669; font-weight:700'>3 / ha</span><br>
                <small>Affordable kits from Mytochondria</small><br><br>
            </div>
        """, unsafe_allow_html=True)

    st.divider()

    # --- Main content area (Left: controls / Right: farms & data) ---
    left, right = st.columns([1, 2])

    with left:
        st.markdown("### ⚙️ Mode & Quick Actions")
        mode = st.radio("Choose mode", ["Sensor", "Imagery"], horizontal=True)
        st.button("➕ Add Farm", on_click=lambda: setattr(st.session_state, "active_tab", 4))
        st.button("📂 Upload Images", key="upload_imgs")
        st.download_button("⬇️ Download Report", "Farm report placeholder", file_name="farm_report.txt")

        st.markdown("---")
        st.markdown("### 📷 Imagery Capture Tips")
        st.info("• Use daylight, avoid shadows\n\n• Take 3 soil images per hectare\n\n• Include clear leaf close-ups")

    with right:
        st.markdown("### 🧭 My Farms")
        if not user_farms:
            st.info("No farms yet. Add one in Manage Account to start.")
        else:
            for f in user_farms:
                st.markdown(f"**{f['farm_id']}** — {f['crop']} in {f['location']} ({f['soil_texture']})")
                st.caption(f"Planted: {f['planting_date']} • Spacing: {f['spacing']}")
                fid = f["farm_id"]
                if fid in st.session_state.farm_hist and st.session_state.farm_hist[fid]:
                    latest = st.session_state.farm_hist[fid][-1]
                    health_score = compute_health_score(f, latest, fid)
                else:
                    health_score = random.uniform(0.6, 0.95)

                if health_score > 0.85:
                    label = "🌿 Excellent"
                elif health_score > 0.7:
                    label = "👍 Good"
                else:
                    label = "⚠️ Needs Attention"

                st.progress(health_score, text=f"Farm Health: {label}")
                st.caption("Calculated from soil moisture, NPK balance, pH, EC, and nutrient trends.")
                st.divider()

        # Default “dynamic” area
        st.markdown("### 📊 Get Started")
        st.info("Select a farm to see live sensor data or upload imagery for analysis.")

    st.markdown("<br>", unsafe_allow_html=True)
    st.caption("© Mytochondria — Built for African farmers 🌍")
elif active == "Sensor Mode":
    st.subheader("Sensor Mode Live ")

    # 1) Backfill all farms to current hour so data is always up-to-date
    for _farm in user_farms:
        update_farm_until_now(_farm)

    if not user_farms:
        st.info("You don’t have any farms yet. Add one in **Manage Account** or use **Crop pplanner**.")
        st.caption("You can still generate plans in the Non-Sensor tab without adding a farm.")
    else:
        # ↓↓↓ everything from the "2) Choose ONE farm..." line to the end of the Sensor tab goes under this 'else:'
        # 2) Choose ONE farm (no multi-select)

        # 2) Choose ONE farm (no multi-select)
        top_l, top_r = st.columns([3, 1])
        with top_l:
            ids = [f["farm_id"] for f in user_farms]
            sel_id = st.selectbox("Choose a farm", ids, index=0, key="one_farm_select")
        with top_r:
            if st.button("Sync to current hour"):
                for _farm in user_farms:
                    update_farm_until_now(_farm)
                st.rerun()

        # 3) Load selected farm + latest data
        farm = next(f for f in user_farms if f["farm_id"] == sel_id)
        fid = farm["farm_id"]
        df = pd.DataFrame(st.session_state.farm_hist[fid]).sort_values("timestamp")
        latest = df.iloc[-1].to_dict()
        planting_date = df.iloc[0]["planting_date"]
        days_since = (datetime.now().date() - planting_date).days
        eta_days = max(0, _expected_days_to_harvest(farm["crop"]) - days_since)

        # 4) Layout: left (cards + insights + logs + charts) / right (weather)
        left, right = st.columns([3, 1])

        # LEFT — summary cards (squares) + insights + full history + charts
        with left:
            st.markdown("#### Farm overview")

            # Grid of cards (“squares”)
            st.markdown('<div class="card-grid">', unsafe_allow_html=True)

            # Spacing & dates
            _card("📅 Planting & Spacing",
                  f"{planting_date.strftime('%b %d')} • {farm['row_cm']}×{farm['plant_cm']} cm",
                  sub=f"Expected harvest: ~{eta_days} days", color="blue")

            # Sensor health
            last72 = df.tail(72)
            uptime = 100.0 * len(last72) / 72.0 if len(df) >= 72 else 100.0
            _card("🩺 Sensor Health", f"{uptime:.0f}%", sub="Expand for details",
                  color=("green" if uptime >= 95 else ("amber" if uptime >= 80 else "red")))

            # Moisture card (traffic-light)
            mcol = _moisture_color(latest["moisture"])
            _card("💧 Soil Moisture", f"{latest['moisture']:.0f}%",
                  sub=("Adequate" if mcol == "green" else ("Watch" if mcol == "amber" else "Low")), color=mcol)

            # NPK status cards (H/M/L)
            ncat, pcat, kcat = pct_to_cat(latest["n"]), pct_to_cat(latest["p"]), pct_to_cat(latest["k"])
            _card("🟢 N", ncat, sub="Nitrogen", color=_color_by_status(ncat))
            _card("🔵 P", pcat, sub="Phosphorus", color=_color_by_status(pcat))
            _card("🟠 K", kcat, sub="Potassium", color=_color_by_status(kcat))

            # Number of sensors per field (demo: 3)
            _card("📡 Sensors", "3", sub="per field", color="gray")

            st.markdown('</div>', unsafe_allow_html=True)

            # Expandable sensor health details
            with st.expander("Sensor health details"):
                st.write(f"Uptime last 72h: **{uptime:.0f}%**")
                st.write(f"Last reading time: **{df['timestamp'].iloc[-1]}**")
                st.write("Out-of-range count (demo): 0")

            # TODAY’S INSIGHTS (plain language)
            st.markdown("#### Today’s insights")
            latest["planting_date"] = planting_date
            latest["crop"] = farm["crop"]
            insights = generate_insights(latest)

            # Weather-aware tweak: if water needed and rain tomorrow >=10mm → suggest waiting
            try:
                wx_day = fetch_weather(farm["lat"], farm["lon"], days_forward=3, days_past=0)
                rain_next = float(wx_day["daily"]["precipitation_sum"][1]) if len(
                    wx_day["daily"]["precipitation_sum"]) > 1 else 0.0
            except Exception:
                rain_next = 0.0

            for it in insights:
                msg = f"**[{it['priority'].upper()}] {it['title']}** : {it['action']}"
                # Smarter context
                if it["type"] == "water" and rain_next >= 10:
                    msg += f" _(Rain ~{int(rain_next)} mm expected tomorrow: you may delay irrigation.)_"
                if it["type"] == "fertilizer" and latest["moisture"] < 40:
                    msg += " ⚠️ Soil moisture is low, apply nutrients only after irrigation for better uptake."
                if it["type"] == "ph" and farm["soil_texture"] == "sand":
                    msg += " Sandy soils lose lime faster, repeat liming every 2–3 years."
                st.info(msg)

                # Simple organic tips (farmer-friendly)
            with st.expander("🌍 Soil & Plant Health Tips"):
                # pool of rotating, localizable tips
                tips = [
                    "🌱 **Nitrogen**: composted manure or legume residues; intercrop maize with beans for N boost.",
                    "🌿 **Phosphorus**: bone meal, manure, or rock phosphate on acidic soils.",
                    "🍌 **Potassium**: banana peels, wood ash (lightly, avoid in alkaline soils).",
                    "🪱 **Soil health**: add compost/vermicompost to build microbes & water holding.",
                    "🌾 **Irrigation timing**: water early morning/evening to reduce evaporation.",
                    "🍂 **Residues**: leave crop residues on the field to protect against erosion.",
                    "🌳 **Agroforestry**: plant trees on field edges for shade, litter, and moisture retention.",
                    "🐓 **Manure use**: poultry manure adds quick N, cattle manure adds long-term OM."
                ]
                for tip in random.sample(tips, 3):
                    st.success("• " + tip)
                    # PAST: daily alerts & applied actions with farmer “ticks”
            st.markdown("#### Past alerts & actions")

            # Daily alerts (all)
            alert_rows = st.session_state.farm_alerts.get(fid, [])
            alert_df = pd.DataFrame(alert_rows) if alert_rows else pd.DataFrame(
                columns=["day", "type", "title", "status"])
            if not alert_df.empty:
                st.write("**Daily alerts (all)**")
                st.dataframe(alert_df.sort_values("day", ascending=False),
                             use_container_width=True, hide_index=True)

            # Applied actions (all) + confirmation ticks
            action_rows = st.session_state.farm_actions_log.get(fid, [])
            act_df = pd.DataFrame([{
                "time": a["ts"], "type": a["type"], "title": a["title"], "status": a["status"]
            } for a in action_rows]) if action_rows else pd.DataFrame(
                columns=["time", "type", "title", "status"])
            if not act_df.empty:
                st.write("**Applied actions (all)**")
                act_df = act_df.sort_values("time", ascending=False)
                # Quick confirmation/checklist for the latest 20
                with st.form(f"confirm_actions_{fid}"):
                    show = act_df.head(20).copy()
                    confirms = []
                    for i, row in show.iterrows():
                        k = (fid, str(row["time"]), row["type"])
                        checked = k in st.session_state.farm_actions_confirm
                        confirms.append(st.checkbox(
                            f"{row['time']} : {row['title']} ({row['type']})",
                            value=checked, key=f"cfm_{fid}_{i}"
                        ))
                    if st.form_submit_button("Save confirmations"):
                        for i, row in show.iterrows():
                            k = (fid, str(row["time"]), row["type"])
                            if confirms[i]:
                                st.session_state.farm_actions_confirm.add(k)
                            else:
                                st.session_state.farm_actions_confirm.discard(k)
                st.dataframe(act_df, use_container_width=True, hide_index=True)

            # CHARTS — full history at the bottom
            st.markdown("#### Charts")
            tail = df.copy()  # keep 'timestamp' column
            accent = _accent(st.session_state.theme)

            # 1) Moisture & Temperature (two series)
            moist_temp = (
                alt.Chart(tail)
                .transform_fold(["moisture", "temperature"], as_=["Metric", "Value"])
                .mark_line(point=True)
                .encode(
                    x=alt.X("timestamp:T", title=""),
                    y=alt.Y("Value:Q", title=""),
                    color=alt.Color(
                        "Metric:N",
                        scale=alt.Scale(range=[accent, "#8899aa"]),  # accent + neutral
                        legend=alt.Legend(title=None)
                    ),
                    tooltip=[
                        alt.Tooltip("timestamp:T", title="Time"),
                        alt.Tooltip("Metric:N"),
                        alt.Tooltip("Value:Q", format=".2f")
                    ],
                )
            )
            st.altair_chart(style_chart(moist_temp, st.session_state.theme), use_container_width=True)

            # 2) pH & EC
            ph_ec = (
                alt.Chart(tail)
                .transform_fold(["ph", "ec"], as_=["Metric", "Value"])
                .mark_line(point=True)
                .encode(
                    x=alt.X("timestamp:T", title=""),
                    y=alt.Y("Value:Q", title=""),
                    color=alt.Color(
                        "Metric:N",
                        scale=alt.Scale(range=[accent, "#c0a5ff"]),
                        legend=alt.Legend(title=None)
                    ),
                    tooltip=[
                        alt.Tooltip("timestamp:T", title="Time"),
                        alt.Tooltip("Metric:N"),
                        alt.Tooltip("Value:Q", format=".2f")
                    ],
                )
            )
            st.altair_chart(style_chart(ph_ec, st.session_state.theme), use_container_width=True)

            # 3) N, P, K
            npk = (
                alt.Chart(tail)
                .transform_fold(["n", "p", "k"], as_=["Nutrient", "Percent"])
                .mark_line(point=True)
                .encode(
                    x=alt.X("timestamp:T", title=""),
                    y=alt.Y("Percent:Q", title=""),
                    color=alt.Color(
                        "Nutrient:N",
                        scale=alt.Scale(range=[accent, "#6aa07b", "#c59b30"]),
                        legend=alt.Legend(title=None)
                    ),
                    tooltip=[
                        alt.Tooltip("timestamp:T", title="Time"),
                        alt.Tooltip("Nutrient:N"),
                        alt.Tooltip("Percent:Q", format=".0f")
                    ],
                )
            )
            st.altair_chart(style_chart(npk, st.session_state.theme), use_container_width=True)
        # RIGHT — past & forecast weather
        with right:
            st.markdown("#### Weather (past & forecast)")
            try:
                wx = fetch_weather(farm["lat"], farm["lon"], days_forward=5, days_past=5)
                days = [pd.to_datetime(d).date().strftime("%b %d") for d in wx["daily"]["time"]]
                rain = wx["daily"]["precipitation_sum"]
                et0 = wx["daily"]["et0_fao_evapotranspiration"]
                tmax = wx["daily"]["temperature_2m_max"]
                wdf = pd.DataFrame({"Day": days, "Rain (mm)": rain, "ET₀ (mm)": et0, "Tmax (°C)": tmax})
                # split around “today”
                today_idx = [i for i, d in enumerate(wx["daily"]["time"]) if
                             pd.to_datetime(d).date() == date.today()]
                cut = today_idx[0] if today_idx else len(days) // 2
                st.write("**Past days**")
                st.dataframe(wdf.iloc[:cut], use_container_width=True, hide_index=True)
                st.write("**Upcoming days**")
                st.dataframe(wdf.iloc[cut:], use_container_width=True, hide_index=True)
            except Exception:
                st.info("Weather unavailable right now.")

elif active == "Crop Planner":
    left, right = st.columns([2, 1])

    with right:
        st.subheader("Location & Planting")
        site = st.selectbox("Choose your location", list(ZAMBIA_SITES.keys()) + ["Custom lat/lon"])
        if site == "Custom lat/lon":
            lat = st.number_input("Latitude", -90.0, 90.0, -15.416, 0.001)
            lon = st.number_input("Longitude", -180.0, 180.0, 28.283, 0.001)
            aer = st.selectbox("Agro-ecological Region", list(AER.keys()), index=1)
        else:
            lat, lon, aer = ZAMBIA_SITES[site]

        st.caption(
            f"AER **{aer}** : typical annual rainfall {AER[aer]['rain_mm']} mm; soils: {AER[aer]['soil_note']}.")

        planting_date = st.date_input("Planned planting date", value=date.today())
        crop = st.selectbox("Main crop", ["maize", "beans", "rice"])
        # intercropping (optional)
        with st.expander("Add a second crop (optional)"):
            crop2_on = st.checkbox("Enable intercropping")
            crop2 = st.selectbox("Second crop", ["beans", "maize", "rice"], index=0, disabled=not crop2_on)

        st.subheader("🌾 Plant Spacing and Density")

        col1, col2 = st.columns(2)
        with col1:
            row_cm = st.number_input("Row spacing (cm)", min_value=10.0, max_value=150.0,
                                     value=75.0, step=5.0, key="lab_row_spacing_cm")
        with col2:
            plant_cm = st.number_input("In-row spacing (cm)", min_value=5.0, max_value=100.0,
                                       value=25.0, step=5.0, key="lab_plant_spacing_cm")

        dens_factor, plants_ha = compute_density_factor(crop, row_cm, plant_cm)

        if crop2_on:
            col3, col4 = st.columns(2)
            with col3:
                row2 = st.number_input("Row spacing (2nd crop) (cm)", 10.0, 150.0, 45.0, 5.0,
                                       key="lab_row_spacing2_cm")
            with col4:
                plant2 = st.number_input("In-row spacing (2nd crop) (cm)", 5.0, 100.0, 20.0, 5.0,
                                         key="lab_plant_spacing2_cm")
            dens2, plants2 = compute_density_factor(crop2, row2, plant2)
            dens_factor = clamp(dens_factor + 0.5 * dens2, 0.8, 1.4)
            plants_ha = plants_ha + 0.5 * plants2

        st.markdown(f"**Estimated stand:** {plants_ha:,.0f} plants/ha, density factor used {dens_factor:.2f}")

        # --- Hidden constants (not user inputs anymore) ---
        organic_matter = 3.00  # %
        target_yield_factor = 1.00  # default multiplier
        bulk_density = 1.30  # g/cm³
        sampling_depth = 0.20  # m
        st.subheader("Management & Soil")
        soil_texture = st.selectbox("Soil texture (typical)", ["loam", "sand", "clay"],
                                    index=0 if aer != "IIb" else 1,
                                    key="lab_soil_texture")
        om_pct = 3.00
        yield_factor = 1.00
        bd = 1.30
        depth_m = 0.20

        st.subheader("Lab results (enter real values)")
        ph = st.number_input("pH (water)", 3.5, 9.0, 6.0, 0.1)
        ec = st.number_input("EC (dS/m): Electric conductivity of the soil", 0.0, 5.0, 0.8, 0.1)
        # N: allow entering nitrate-N kg/ha or estimate from mg/kg
        n_mode = st.radio("Nitrogen input mode", ["kg/ha available N", "mg/kg nitrate-N"], horizontal=True)
        if n_mode == "kg/ha available N":
            n_kgha = st.number_input("Available N (kg/ha)", 0.0, 300.0, 20.0, 1.0)
        else:
            n_mgkg = st.number_input("Nitrate-N (mg/kg)", 0.0, 100.0, 10.0, 0.5)
            n_kgha = mgkg_to_kgha(n_mgkg, bd, depth_m)
        p_mgkg = st.number_input("Soil test P (mg/kg)", 0.0, 200.0, 12.0, 0.5)
        k_mgkg = st.number_input("Soil test K (mg/kg)", 0.0, 400.0, 80.0, 1.0)

        if st.button("Generate plan with real weather"):
            wx = fetch_weather(lat, lon, days_forward=10, days_past=7)
            st.session_state.ns_last = dict(
                lat=lat, lon=lon, aer=aer, planting_date=planting_date, crop=crop,
                dens_factor=dens_factor, plants_ha=plants_ha, soil_texture=soil_texture,
                om_pct=om_pct, yield_factor=yield_factor, bd=bd, depth_m=depth_m,
                ph=ph, ec=ec, n_kgha=n_kgha, p_mgkg=p_mgkg, k_mgkg=k_mgkg,
                weather=wx, crop2=crop2 if crop2_on else None
            )

    with left:
        st.subheader("Insights & Actions")
        st.subheader("                  ")
        if "ns_last" not in st.session_state:
            st.info("Fill the panel and click **Generate plan**.")
        else:
            p = st.session_state.ns_last
            daily = p["weather"]["daily"]
            df, water = irrigation_recommendations(
                p["crop"], p["planting_date"], daily, p["yield_factor"], p["dens_factor"]
            )
            weekly = water["weekly_plan"]
            risks = water["risks"]

            plan = nutrient_plan_from_lab(
                p["crop"], p["yield_factor"], p["om_pct"],
                p["n_kgha"], p["p_mgkg"], p["k_mgkg"], p["bd"], p["depth_m"]
            )

            # 5) actions in simple english
            actions = generate_simple_actions(p, plan, weekly, risks, df)
            st.markdown("### What to do (simple steps)")
            for a in actions:
                st.write("• " + a)

            # (optional) one-click export
            from io import StringIO

            buf = StringIO()
            buf.write("Mytochondria – Non-Sensor Plan\n\n")
            for a in actions:
                buf.write("• " + a + "\n")
            st.download_button("Download actions as text", buf.getvalue(), file_name="field_actions.txt",
                               key="lab_actions_dl")

            # 1) Weather-driven irrigation & risks
            st.markdown("### Water plan (next 10 days)")
            st.dataframe(weekly, use_container_width=True, hide_index=True)
            st.write(f"- **Erosion risk days** (≥30 mm/day): **{risks['erosion_days']}**")
            st.write(f"- **Heat-stress days** (Tmax ≥35°C): **{risks['heat_stress_days']}**")
            st.write(
                f"- **Cool germination risk** (early stage, low Tmax): **{risks['cool_germination_days']}**")

            rain_line = (
                alt.Chart(df)
                .mark_line(point=True, color=_accent(st.session_state.theme))
                .encode(
                    x=alt.X("date:T", title=""),
                    y=alt.Y("Rain_mm:Q", title="Rain (mm)"),
                    tooltip=[alt.Tooltip("date:T", title="Day"),
                             alt.Tooltip("Rain_mm:Q", title="Rain (mm)", format=".1f")]
                )
            )
            st.altair_chart(style_chart(rain_line, st.session_state.theme), use_container_width=True)
            # 2) Nutrient plan from lab

            st.markdown("### Nutrient plan")
            st.write(f"**Nitrogen (N):** apply ~**{plan['N_rec_kg_ha']} kg/ha** "
                     f"(credit from OM: {plan['notes']['n_credit_om']} kg/ha). "
                     "👉 Alternative: use **farmyard manure, compost, or incorporate legume residues** to slow depletion of N.")

            st.write(f"**Phosphorus (P₂O₅):** **{plan['P2O5_rec_kg_ha']} kg/ha** "
                     "👉 Alternative: use **bone meal or rock phosphate** on acidic soils.")

            st.write(f"**Potassium (K₂O):** **{plan['K2O_rec_kg_ha']} kg/ha** "
                     "👉 Alternative: apply **wood ash (small amounts), composted banana peels, or sulfate K if saline soils**.")

            st.caption(f"Estimated P pool: {plan['notes']['p_pool_kgha']} kg/ha, "
                       f"K pool: {plan['notes']['k_pool_kgha']} kg/ha (top {p['depth_m']} m).")

            # --- Extra irrigation plan (based on last year's weather) ---
            st.markdown("### Irrigation plan (based on last year’s forecast)")
            try:
                past_wx = fetch_weather(p["lat"], p["lon"], days_forward=0, days_past=365)
                df_past, irr_plan = irrigation_recommendations(
                    p["crop"], p["planting_date"], past_wx["daily"], p["yield_factor"], p["dens_factor"]
                )
                st.write(
                    "Using last year’s rainfall and ET₀ (evapotranspiration), here is a weekly irrigation plan:")

                st.dataframe(irr_plan["weekly_plan"], use_container_width=True, hide_index=True)

                st.markdown("#### 👩‍🌾 Coach’s Note")
                st.info("""
                    This plan shows **how much water your crop will miss if you do nothing**.  
                    👉 Apply the suggested irrigation in 2–3 splits per week.  
                    👉 Focus water at root zone (20–30 cm).  
                    👉 Watch for erosion if >30 mm rain/day is forecast.  
                    """)

                with st.expander("ℹ️ Understanding the irrigation table"):
                    st.write("""
                        - **Week**: Calendar week.
                        - **ETc_mm**: Crop demand.
                        - **Rain_mm**: Rain received.
                        - **Deficit_mm**: Shortfall. If positive, irrigation is needed.
                        - **Irrigation_mm**: Suggested water to close the gap.
                        """)
                    st.caption(
                        "Tip: Apply irrigation in 2–3 smaller splits per week rather than one large application.")

                #  Optional explanation text
                st.markdown("#### Why deficits matter")
                st.write("""
                    • **ETc (crop water demand)** is how much water the crop actually needs each week.  
                    • **Rainfall** supplies part of that demand.  
                    • **Deficit** is the shortfall: if positive, the crop will stress unless you irrigate.  
                    • The plan tells you how much to irrigate to keep plants healthy.  
                    """)
            except Exception:
                st.info("Irrigation plan unavailable (weather data fetch failed).")

            # 3) Condition-specific tips (pH/EC/texture/AER/weather)
            tips = []
            if p["ph"] < 5.5:
                tips += [
                    "Soil is acidic: consider liming to reach ~pH 6.0–6.5 before planting (apply 2–4 months ahead)."]
            if p["ec"] >= 2.0:
                tips += [
                    "High salinity risk: avoid chloride-heavy K sources; schedule leaching irrigation after heavy rains."]
            if p["aer"] == "IIb" or p["soil_texture"] == "sand":
                tips += [
                    "Kalahari sands/sandy soils: add **organic matter** (residues/compost/manure) to improve water & nutrient holding."]

            if risks["erosion_days"] >= 1:
                tips += [
                    "Forecast has ≥30 mm/day rain: keep residue cover, contour ploughing or tied ridges to reduce runoff."]
            if risks["heat_stress_days"] >= 1 and p["crop"] == "maize":
                tips += [
                    "Heat near flowering can cut kernel set: ensure no water stress 1 week before to 2 weeks after tasseling."]

            if tips:
                st.markdown("### Tips")
                for t in tips:
                    st.write("• " + t)

            # 4) Yield outlook (very simple score)
            total_etc = df["ETc_mm"].sum()
            total_rain = df["Rain_mm"].sum()
            water_ratio = (total_rain + weekly["Irrigation_mm"].sum()) / max(1.0, total_etc)
            n_ok = (plan["N_rec_kg_ha"] < 20)  # if additional N need is small, we assume N adequate
            if water_ratio >= 0.9 and n_ok and risks["heat_stress_days"] == 0:
                outlook = "Good"
            elif water_ratio >= 0.75:
                outlook = "Watch"
            else:
                outlook = "At risk"
            st.markdown(f"### Yield outlook: **{outlook}**")
            st.caption("Heuristic: compares ETc vs rain+irrigation, checks N sufficiency & heat-stress days.")

elif active == "Tips & Tricks":
    st.subheader("💡 Tips & Tricks for Farmers")

    do_list = [
        "Rotate crops each season to improve soil health.",
        "Use compost or manure to build organic matter.",
        "Mulch around plants to reduce evaporation.",
        "Irrigate early morning or late evening to save water."
    ]
    dont_list = [
        "Don’t over-irrigate during the day (high losses).",
        "Avoid mixing lime with ammonium fertilizers.",
        "Never burn crop residues — compost them instead.",
        "Don’t apply fertilizer right before heavy rain."
    ]

    st.markdown("### ✅ Do’s")
    for d in do_list:
        st.success(d)

    st.markdown("### ❌ Don’ts")
    for d in dont_list:
        st.error(d)

elif active == "AI Imagery Analysis" :
    st.header("📷 AI Imagery Analysis – Soil & Plant Health")

    st.markdown("""
    ### 🧭 How to Take Good Images
    To get the most accurate results:
    - **Use natural light**, avoid strong shadows or artificial lighting.  
    - **Hold your camera parallel** to the soil or plant leaf surface.  
    - **Do not edit or filter** the photo before uploading.  
    - Ensure the image is **clear and in focus** (no blur).  
    - Take **multiple angles** if possible for better AI accuracy.
    """)

    st.markdown("---")

    col1, col2 = st.columns(2)

    # === Soil Imagery ===
    with col1:
        st.subheader("🌍 Soil Imagery")
        soil_file = st.file_uploader("Upload Soil Image", type=["jpg", "jpeg", "png"], key="soil_image")
        if soil_file is not None:
            st.image(soil_file, caption="Uploaded Soil Image", use_column_width=True)
            if st.button("Analyze Soil Image", key="analyze_soil"):
                with st.spinner("Analyzing soil image with AI..."):
                    # 🚀 Placeholder for AI call — replace with actual model API later
                    st.success("✅ AI detected rich organic matter and good soil texture balance.")
                    st.markdown("""
                    **AI Summary:**
                    - Soil appears **loamy-sand** with good structure.  
                    - Color indicates **moderate organic content (2–3%)**.  
                    - Suggest **adding compost or manure** to boost microbial activity.  
                    - Check **pH (expected ~6.5)** before planting.
                    """)

    # === Plant Imagery ===
    with col2:
        st.subheader("🌿 Plant Health Imagery")
        plant_file = st.file_uploader("Upload Plant Image", type=["jpg", "jpeg", "png"], key="plant_image")
        if plant_file is not None:
            st.image(plant_file, caption="Uploaded Plant Image", use_column_width=True)
            if st.button("Analyze Plant Image", key="analyze_plant"):
                with st.spinner("Analyzing plant image with AI..."):
                    # 🚀 Placeholder for AI call — replace with actual model API later
                    st.success("✅ AI analyzed the plant successfully.")
                    st.markdown("""
                    **AI Summary:**
                    - Leaves show **mild nitrogen deficiency** (light green discoloration).  
                    - Recommend **topdressing with N-rich fertilizer (urea or organic)**.  
                    - No visible fungal lesions or pest damage.  
                    - Overall **plant vigor: good**.
                    """)

elif active == "Manage Account":
    st.subheader("👤 Account Dashboard")

    left, right = st.columns([2, 1])

    with right:
        st.markdown("### Account Info")
        st.write(f"**Username:** {user['username']}")
        st.write(f"**Email:** {user.get('email', '-')}")
        if st.button("Logout"):
            st.session_state.user = None
            st.rerun()
    # ------------------------------------------------
    # Add New Farm Section
    # ------------------------------------------------
    with left:
        st.markdown("### 🌱 Your Farms")
        for farm in user.get("farms", []):
            st.write(f"**{farm['farm_id']}** : {farm['crop']} at {farm['location']}")
            st.caption(
                f"Planted: {farm['planting_date']} • "
                f"Spacing: {farm['spacing']} • "
                f"Row: {farm.get('row_cm', '?')} cm • Plant: {farm.get('plant_cm', '?')} cm • "
                f"Texture: {farm.get('soil_texture', '?')} • "
                f"Compliance: {farm['compliance']}"
            )

            del_key = f"delete_{farm['farm_id']}"
            if st.button(f"🗑️ Delete {farm['farm_id']}", key=del_key):
                st.session_state.confirm_delete = farm["farm_id"]

        # Confirmation prompt
        if "confirm_delete" in st.session_state and st.session_state.confirm_delete:
            fid = st.session_state.confirm_delete
            st.error(f"⚠️ Are you sure you want to delete farm {fid}? This action cannot be undone.")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("✅ Yes, delete permanently"):
                    delete_farm(fid)
                    st.session_state.user = find_user(user["username"])
                    st.success(f"Farm {fid} deleted successfully.")
                    st.session_state.confirm_delete = None
                    st.rerun()
            with col2:
                if st.button("❌ Cancel"):
                    st.session_state.confirm_delete = None
                    st.info("Deletion canceled.")

        # Toggle to show the Add Farm form
        st.markdown("### ➕ Add New Farm")
        if st.button("Add Farm"):
            st.session_state.show_add_farm = not st.session_state.get("show_add_farm", False)

        if st.session_state.get("show_add_farm", False):
            crop_choice = st.selectbox("Crop", ["maize", "beans", "rice", "maize+beans"], key="add_crop")
            location_choice = st.selectbox("Location", list(ZAMBIA_SITES.keys()), key="add_location")
            lat, lon, aer = ZAMBIA_SITES.get(location_choice, (None, None, None))

            rec_row, rec_plant = CROP_SPACING_DEFAULT.get(crop_choice, (60, 20))
            rec_texture = AER_TEXTURE_DEFAULT.get(aer, "loam")

            st.info(f"📌 Recommended for {crop_choice} in {location_choice}: "
                    f"{rec_row}×{rec_plant} cm • Soil: {rec_texture} ")
            st.warning("⚠️ Adjust manually if your soil/spacing differs.")

            with st.form("add_farm", clear_on_submit=True):
                farm_id = st.text_input("Farm ID", key="add_farm_id")
                system_id = st.text_input("System ID", key="add_system_id")
                row_cm = st.number_input("Row spacing (cm)", 10, 150,
                                         value=rec_row, step=5, key="add_row_cm")
                plant_cm = st.number_input("Plant spacing (cm)", 5, 100,
                                           value=rec_plant, step=5, key="add_plant_cm")
                soil_texture = st.selectbox("Soil texture", ["sand", "loam", "clay"],
                                            index=["sand", "loam", "clay"].index(rec_texture),
                                            key="add_soil_texture")
                planting_date = st.date_input("Planting date", key="add_planting_date")
                compliance = st.selectbox("Compliance behavior", ["immediate", "delayed"], key="add_compliance")

                if st.form_submit_button("Save Farm"):
                    spacing = f"{row_cm}x{plant_cm} cm"
                    new_farm = {
                        "farm_id": farm_id, "system_id": system_id, "crop": crop_choice,
                        "location": location_choice, "lat": lat, "lon": lon,
                        "soil_texture": soil_texture,
                        "row_cm": row_cm, "plant_cm": plant_cm, "spacing": spacing,
                        "planting_date": str(planting_date), "compliance": compliance,
                        "yield_factor": 1.0, "om_pct": 2.0,
                        "agent": {
                            "compliance": (0.8 if compliance == "immediate" else 0.5),
                            "delay_min_h": 6,
                            "delay_max_h": 24
                        }
                    }
                    save_farm(user["username"], new_farm)
                    st.session_state.user = find_user(user["username"])
                    st.success("✅ Farm added successfully!")
                    st.session_state.show_add_farm = False
                    st.rerun()
