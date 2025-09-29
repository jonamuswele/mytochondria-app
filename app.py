
import math
import random
from datetime import datetime, timedelta, date
from typing import List, Dict, Any, Optional, Tuple
import streamlit as st
import pandas as pd

# ==== NPK unit mapping (demo-calibrated caps) ====
SUPPLY_CAP_KG_HA = {  # "100%" corresponds to this much plant-available nutrient
    "n": 120.0,   # kg N/ha
    "p": 60.0,    # kg P2O5/ha (demo cap)
    "k": 100.0,   # kg K2O/ha (demo cap)
}

def pct_to_kg(nutrient: str, pct: float) -> float:
    return max(0.0, (pct / 100.0) * SUPPLY_CAP_KG_HA[nutrient])

def kg_to_pct(nutrient: str, kg: float) -> float:
    cap = SUPPLY_CAP_KG_HA[nutrient]
    if cap <= 0: return 0.0
    return clamp((kg / cap) * 100.0, 0.0, 100.0)

# ==== Soil/management multipliers ====
TEXTURE_UPTAKE_MULT = {  # effect on plant uptake demand
    "sand": 1.15, "loam": 1.00, "clay": 0.90
}
TEXTURE_LEACH_MULT = {   # effect on N leaching when it rains
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

# ------------------------------
# Utility + Global Config
# ------------------------------

st.set_page_config(
    page_title="Mytochondria Soil Advisor ",
    page_icon="🌱",
    layout="wide",
)

def pct_to_cat(x: float) -> str:
    """Map numeric percent (0-100) to High/Medium/Low for NPK display."""
    if x >= 60:
        return "High"
    if x >= 30:
        return "Medium"
    return "Low"

def clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))

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

def simulate_from_planting(planting_date: date, crop: str) -> Tuple[List[Dict[str,Any]], List[Dict[str,Any]]]:
    """
    Build an hourly time series from planting_date up to now.
    Daily at ~06:00 we evaluate 'alerts' and AUTO-APPLY them to the data:
      - Water: moisture jump, EC small drop (dilution)
      - Top-dress N (crop window + low N): N up, EC up
      - Lime (if pH too low): pH up slowly
      - Leach salts (if EC high): moisture up, EC down
    Also simulates rainfall (random) and diurnal temperature.
    """
    series: List[Dict[str,Any]] = []
    alerts_log: List[Dict[str,Any]] = []

    # initial state
    cur_time = datetime.combine(planting_date, datetime.min.time())
    end_time = datetime.now().replace(minute=0, second=0, microsecond=0)

    # starting values (reasonable)
    moisture = 65.0
    ph = 6.5
    ec = 1.2

    n_kg = pct_to_kg("n", 70.0)
    p_kg = pct_to_kg("p", 65.0)
    k_kg = pct_to_kg("k", 60.0)

    n, p, k = 70.0, 65.0, 60.0
    day_mean = 24.0

    def add_point(ts):
        series.append({
            "timestamp": ts, "moisture": moisture, "temperature": temp,
            "ph": ph, "ec": ec, "n": kg_to_pct("n", n_kg), "p": kg_to_pct("p", p_kg), "k": kg_to_pct("k", k_kg)
        })

    prev_day = cur_time.date()
    daily_rain = False

    while cur_time <= end_time:
        hour = cur_time.hour
        day_index = (cur_time.date() - planting_date).days

        # start of a new day: set daily context
        if cur_time.date() != prev_day:
            prev_day = cur_time.date()
            day_index = (cur_time.date() - planting_date).days

            # Daily baseline updates
            day_mean = clamp(day_mean + (random.random() - 0.5) * 1.0, 18, 30)

            # Decide rainfall for the day (20% chance)
            daily_rain = (random.random() < 0.20)
            if daily_rain:
                # rainfall event increases moisture; leaches salts a bit
                moisture = clamp(moisture + 15 + random.random() * 10, 0, 95)
                ec = clamp(ec - 0.25, 0.1, 3.0)

            # -------------------------------
            # 06:00 DAILY ALERTS (AUTO-APPLY)
            # -------------------------------
            todays_alerts = []
            cap_n = SUPPLY_CAP_KG_HA["n"]

            # 1) Irrigate if too dry
            if moisture < 30:
                moisture = clamp(moisture + 20, 0, 95)  # irrigation raises moisture
                ec = clamp(ec - 0.05, 0.1, 3.0)  # slight dilution of salts
                todays_alerts.append({
                    "type": "water", "priority": "high",
                    "title": "Irrigate Today",
                    "action": "Applied ~25–30 mm irrigation at 06:00"
                })

            # 2) Top-dress N if maize window & N low (threshold = 45% of cap)
            if crop.strip().lower() == "maize" and 28 <= day_index <= 56 and n_kg < 0.45 * cap_n:
                # add ~18% of cap as kg, bump EC slightly
                n_kg = min(cap_n, n_kg + 0.18 * cap_n)
                ec = clamp(ec + 0.35, 0.1, 3.0)
                todays_alerts.append({
                    "type": "fertilizer", "priority": "high",
                    "title": "Top-dress Nitrogen",
                    "action": "Applied ~50 kg/ha urea at 06:00"
                })

            # 3) Lime if pH too low
            if ph < 5.7:
                ph = clamp(ph + 0.15, 3.5, 9.0)  # immediate portion of liming response
                todays_alerts.append({
                    "type": "ph", "priority": "medium",
                    "title": "Apply Lime",
                    "action": "Applied lime at 06:00 (pH buffering)"
                })

            # 4) Leach salts if EC high
            if ec >= 2.0:
                moisture = clamp(moisture + 10, 0, 95)
                ec = clamp(ec - 0.30, 0.1, 3.0)
                todays_alerts.append({
                    "type": "salinity", "priority": "medium",
                    "title": "Leach Salts",
                    "action": "Leaching irrigation at 06:00"
                })

            # stamp and log today's auto-applied actions
            for a in todays_alerts:
                a["day"] = cur_time.date().isoformat()
                alerts_log.append(a)

            # ------------------------------------------
            # ---- DAILY NUTRIENT BALANCE (kg/ha) ------
            # ------------------------------------------

            # 1) N mineralization from OM (supply)
            n_kg += max(0.0, om_pct * N_MINERALIZE_KG_PER_DAY_PER_OM)

            # 2) Extra rain leaching (loss) on rainy days (N only, demo)
            if daily_rain:
                n_kg -= N_LEACH_KG_PER_HEAVY_RAIN * TEXTURE_LEACH_MULT[soil_texture]

            # 3) Plant uptake (loss) — use weekly % rates, scaled to daily and by management
            n_w, p_w, k_w = crop_uptake_weekly(crop, day_index)  # percent-points/week
            # Convert to kg/day using caps, then scale by texture/yield/density
            n_uptake_kg_day = (n_w / 7.0) / 100.0 * SUPPLY_CAP_KG_HA["n"]
            p_uptake_kg_day = (p_w / 7.0) / 100.0 * SUPPLY_CAP_KG_HA["p"]
            k_uptake_kg_day = (k_w / 7.0) / 100.0 * SUPPLY_CAP_KG_HA["k"]

            mgmt_mult = TEXTURE_UPTAKE_MULT[soil_texture] * yield_factor * density_factor
            n_kg -= n_uptake_kg_day * mgmt_mult
            p_kg -= p_uptake_kg_day * mgmt_mult
            k_kg -= k_uptake_kg_day * mgmt_mult

            # 4) Keep pools non-negative
            n_kg = max(0.0, n_kg)
            p_kg = max(0.0, p_kg)
            k_kg = max(0.0, k_kg)

        # diurnal temperature + evapotranspiration hourly
        temp = clamp(day_mean + 6*math.sin((hour-14) * math.pi/12) + (random.random()-0.5)*1.2, 15, 36)

        # hourly evap (scaled by heat)
        evap = max(0.05, (temp-15)/25) * 0.22  # ~0.1–0.35 %/h
        moisture = clamp(moisture - evap, 0, 100)

        # EC concentration rises slightly as water leaves
        ec = clamp(ec + evap*0.02, 0.1, 3.0)

        # slow pH drift
        ph = clamp(ph + (random.random()-0.5)*0.01, 3.5, 9.0)

        # write point
        add_point(cur_time)

        # next hour
        cur_time += timedelta(hours=1)

    return series, alerts_log
def record_alert_history(mode: str, alerts: List[Dict[str, Any]]):
    ts = datetime.now().isoformat(timespec="seconds")
    for a in alerts:
        st.session_state.alert_history.append({
            "time": ts, "mode": mode,
            "title": a["title"], "type": a["type"],
            "priority": a["priority"], "action": a.get("action","")
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
        cur["n"] = clamp(cur["n"] - n_w/7, 0, 100)
        cur["p"] = clamp(cur["p"] - p_w/7, 0, 100)
        cur["k"] = clamp(cur["k"] - k_w/7, 0, 100)

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

# keep alert history in session

def crop_uptake_weekly(crop: str, days_since_planting: int) -> Tuple[float, float, float]:
    """
    Return estimated weekly depletion for N,P,K based on crop & stage.
    Mirrors the ideas from insightsEngine.ts (maize window) and our spec. :contentReference[oaicite:4]{index=4}
    """
    w = days_since_planting // 7
    crop = (crop or "").strip().lower()

    if crop == "maize":
        # N heavy weeks 4-8, moderate otherwise
        n = 5.0 if 4 <= w <= 8 else 2.5
        p = 2.0 if w <= 4 else 1.0
        k = 2.5 if 6 <= w <= 10 else 1.5
    elif crop == "beans":
        # N lower (fixation), P/K moderate
        n = 1.0
        p = 1.5
        k = 2.0
    elif crop == "rice":
        # consistent water crop, N/K relevant at tillering/panicle initiation (approx w3-8)
        n = 3.0 if 3 <= w <= 8 else 1.5
        p = 1.0
        k = 2.5 if 3 <= w <= 8 else 1.5
    else:
        # unknown crop -> conservative
        n = 2.0
        p = 1.2
        k = 1.8

    return n, p, k

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

st.title("🌱 Mytochondria AgriAdvisor ")

tabs = st.tabs(["Sensor Mode", "Non-Sensor (Lab) Mode", "30-Day Demo"])

# ---------------------------------
# 1) SENSOR MODE
# ---------------------------------
with tabs[0]:
    left, right = st.columns([2,1])

    # Controls
    with right:
        st.subheader("Setup")
        st.session_state.planting_date = st.date_input(
            "Planting Date",
            value=st.session_state.planting_date,
            help="Hourly data will be simulated from this date up to now"
        )
        crop_opt = st.selectbox("Crop", ["maize","beans","rice","Other (not listed)"])
        custom_crop = st.text_input("If not listed, type crop name")
        chosen_crop = custom_crop.strip() if crop_opt.startswith("Other") and custom_crop.strip() else crop_opt

        # Soil & management (affect uptake/leaching)
        soil_texture = st.selectbox("Soil texture", ["loam", "sand", "clay"], index=0)
        om_pct = st.slider("Organic matter (%)", 0.0, 6.0, 2.0, 0.1, help="Affects N mineralization")
        yield_factor = st.slider("Yield target factor", YIELD_FACTOR_MIN, YIELD_FACTOR_MAX, 1.0, 0.05)
        density_factor = st.slider("Planting density factor", DENSITY_FACTOR_MIN, DENSITY_FACTOR_MAX, 1.0, 0.05)

        if st.button("Build Series from Planting → Now"):
            series, daily_alerts = simulate_from_planting(st.session_state.planting_date, chosen_crop)
            st.session_state.sensor_history = series
            st.session_state.last_sensor = series[-1] if series else None
            # store alerts in history (auto-applied)
            record_alert_history("sensor", [
                {"type":a["type"], "priority":a["priority"], "title":a["title"], "action":a["action"]}
                for a in daily_alerts
            ])

        show_n_points = st.slider("Show last N points", 24, 24*120, 24*72, step=24, help="Chart window (hours)")

    # Insights above charts + depletion dates + applied alerts list
    with left:
        st.subheader("Insights (live)")
        if st.session_state.sensor_history:

            df = pd.DataFrame(st.session_state.sensor_history).sort_values("timestamp")
            latest = df.iloc[-1].to_dict()
            latest["planting_date"] = st.session_state.planting_date
            latest["crop"] = chosen_crop

            # call your existing insights function (kept from your app)
            insights = generate_insights(latest)  # uses current state
            # show insights first
            for i, ins in enumerate(insights):
                with st.expander(f"{ins['icon']} [{ins['priority'].upper()}] {ins['title']}", expanded=(i==0)):
                    st.write(ins["description"])
                    if ins.get("action"):
                        st.caption(f"Action already applied automatically when triggered: **{ins['action']}**")

            # depletion dates (N/P/K)
            days_since = (datetime.now().date() - st.session_state.planting_date).days
            n_w, p_w, k_w = crop_uptake_weekly(chosen_crop, days_since)

            # scale weekly % by management multipliers (same as simulator)
            weekly_mult = TEXTURE_UPTAKE_MULT[soil_texture] * yield_factor * density_factor
            n_w_eff = n_w * weekly_mult
            p_w_eff = p_w * weekly_mult
            k_w_eff = k_w * weekly_mult

            def eta(cur, weekly):
                d = forecast_depletion_days(cur, weekly)
                return "now" if d == 0 else (datetime.now().date() + timedelta(days=d)).strftime("%b %d, %Y")
            st.markdown("**Depletion Forecasts:**")
            st.write(f"• Nitrogen reaches LOW around: **{eta(latest['n'], n_w_eff)}**")
            st.write(f"• Phosphorus reaches LOW around: **{eta(latest['p'], p_w_eff)}**")
            st.write(f"• Potassium reaches LOW around: **{eta(latest['k'], k_w_eff)}**")
            st.caption("Tip: plan applications ~7 days before the forecast date.")

            st.caption("NPK categories: N = {}, P = {}, K = {}".format(
                pct_to_cat(latest["n"]), pct_to_cat(latest["p"]), pct_to_cat(latest["k"])
            ))

            # Charts
            st.subheader("Charts")
            dfv = df.tail(show_n_points).set_index("timestamp")
            st.line_chart(dfv[["moisture","temperature"]], use_container_width=True)
            st.line_chart(dfv[["ph","ec"]], use_container_width=True)
            st.line_chart(dfv[["n","p","k"]], use_container_width=True)

            # Applied daily alerts (read-only checklist)
            with st.expander("Daily Actions Applied (auto)"):
                # filter today's generation
                day_strs = set(pd.to_datetime(df["timestamp"]).dt.date.astype(str))
                # show last ~30 days of actions
                recent = [a for a in st.session_state.alert_history[-300:] if a["mode"]=="sensor"]
                if recent:
                    recent = recent[::-1]
                    for a in recent[:100]:
                        st.write(f"✅ {a['title']} — _{a['action']}_  ({a['time']})")
                else:
                    st.write("No actions yet.")
        else:
            st.info("Click **Build Series from Planting → Now** to generate hourly data.")

# ---------------------------------
# 2) NON-SENSOR (LAB) MODE
# ---------------------------------
with tabs[1]:
    st.subheader("Enter Exact Lab/Field Test Readings")

    with st.form("manual_form"):
        colA, colB, colC = st.columns(3)
        with colA:
            moisture = st.slider("Soil Moisture (%)", 0, 100, 55)
            temperature = st.slider("Soil Temperature (°C)", 0, 50, 24)
            ph = st.number_input("Soil pH", min_value=3.5, max_value=9.0, value=6.5, step=0.1, format="%.1f")
        with colB:
            ec = st.slider("EC (dS/m), proxy salts", 0.1, 3.0, 1.2, step=0.1)
            n_pct = st.slider("Nitrogen reserve (%)", 0, 100, 60)
            p_pct = st.slider("Phosphorus reserve (%)", 0, 100, 50)
        with colC:
            k_pct = st.slider("Potassium reserve (%)", 0, 100, 55)
            crop = st.selectbox("Crop", ["maize", "beans", "rice", "Other (not listed)"])
            custom_crop = st.text_input("If not listed, type crop name here")
            planting_date = st.date_input("Planting Date", value=st.session_state.planting_date)

        submitted = st.form_submit_button("Analyze")
        if submitted:
            chosen_crop = custom_crop.strip() if crop.startswith("Other") and custom_crop.strip() else crop
            st.session_state.manual_latest = {
                "moisture": float(moisture),
                "temperature": float(temperature),
                "ph": float(ph),
                "ec": float(ec),
                "n": float(n_pct),
                "p": float(p_pct),
                "k": float(k_pct),
                "crop": chosen_crop,
                "planting_date": planting_date,
            }

    if st.session_state.manual_latest:
        cur = st.session_state.manual_latest.copy()
        st.markdown("### Current Status")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Moisture (%)", f"{cur['moisture']:.1f}")
        c2.metric("pH", f"{cur['ph']:.1f}")
        c3.metric("EC (dS/m)", f"{cur['ec']:.1f}")
        c4.metric("Temp (°C)", f"{cur['temperature']:.1f}")

        c1, c2, c3 = st.columns(3)
        c1.metric("N", f"{pct_to_cat(cur['n'])} ({cur['n']:.0f}%)")
        c2.metric("P", f"{pct_to_cat(cur['p'])} ({cur['p']:.0f}%)")
        c3.metric("K", f"{pct_to_cat(cur['k'])} ({cur['k']:.0f}%)")

        insights = generate_insights(cur)

        st.markdown("### Insights & Actions")
        for i, ins in enumerate(insights):
            with st.expander(f"{ins['icon']} [{ins['priority'].upper()}] {ins['title']}", expanded=(i == 0)):
                st.write(ins["description"])
                if ins.get("action"):
                    t_id = f"manual-{i}"
                    prev = st.session_state.checklist.get(t_id, {"label": ins["action"], "done": False, "applied_effect": False})
                    done = st.checkbox(f"Mark done: {ins['action']}", value=prev["done"], key=t_id)
                    st.session_state.checklist[t_id] = {"label": ins["action"], "done": done, "applied_effect": prev.get("applied_effect", False)}
                    if done and not st.session_state.checklist[t_id]["applied_effect"]:
                        apply_action_effects(cur, ins["action"])
                        # persist the change so next analysis uses the new state
                        st.session_state.manual_latest = cur.copy()
                        st.session_state.checklist[t_id]["applied_effect"] = True

        # Custom crop submission
        if cur["crop"].lower() not in ["maize", "beans", "rice"]:
            st.info("Your crop is not in the list. We saved your crop name and data for follow-up advisory.")

# ---------------------------------
# 3) 30-DAY DEMO
# ---------------------------------
with tabs[2]:
    st.subheader("Synthetic 30-Day Dataset")
    days = st.slider("Days", 10, 60, 30, step=5)
    if st.button(f"Generate {days}-Day Demo Data"):
        demo = gen_demo_series(days=days)
        df = pd.DataFrame(demo).sort_values("timestamp")
        st.line_chart(df.set_index("timestamp")[["moisture", "temperature"]], use_container_width=True)
        st.line_chart(df.set_index("timestamp")[["ph", "ec"]], use_container_width=True)
        st.line_chart(df.set_index("timestamp")[["n", "p", "k"]], use_container_width=True)

        latest = df.iloc[-1].to_dict()
        latest["planting_date"] = st.session_state.planting_date
        latest["crop"] = "maize"  # pick one for demo forecasts
        insights = generate_insights(latest)

        st.markdown("### Generated Insights (from last day)")
        for ins in insights:
            st.write(f"- **{ins['title']}** , {ins['description']}" + (f" _Action:_ {ins['action']}" if ins.get("action") else ""))


