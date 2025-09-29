
import math
import random
from datetime import datetime, timedelta, date
from typing import List, Dict, Any, Optional, Tuple
import streamlit as st
import pandas as pd
import requests
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

# ===== Baseline plant densities (demo defaults; tune to your agronomy) =====
BASE_PLANT_DENSITY_HA = {
    "maize": 53333,     # ~75 cm x 25 cm  → ~53,333 plants/ha
    "beans": 200000,    # example baseline
    "rice": 250000,     # ~20 cm x 20 cm  → ~250,000 plants/ha
}

# Clamp bounds for how far density can scale uptake (same ones you already use)
DENSITY_FACTOR_MIN, DENSITY_FACTOR_MAX = 0.8, 1.2

# --- Agro-ecological regions (Zambia) & typical soils (simplified) ---
AER = {
    "I":   {"rain_mm": "<800",  "soil_note": "Sandy-loam to sandy; low nutrient retention, drier south/west"},
    "IIa": {"rain_mm": "800–1000", "soil_note": "Loam to clay-loam; moderately leached; most productive"},
    "IIb": {"rain_mm": "800–1000", "soil_note": "Kalahari sands; strongly acidic, low water & nutrient holding"},
    "III": {"rain_mm": "1000–1500", "soil_note": "Loam to clay; humid north; higher leaching, weathered clays"},
}

# Province/town → (lat, lon, AER)
ZAMBIA_SITES = {
    "Lusaka (Lusaka Prov)":       (-15.416, 28.283, "IIa"),
    "Ndola (Copperbelt)":         (-12.968, 28.635, "IIa"),
    "Kitwe (Copperbelt)":         (-12.818, 28.214, "IIa"),
    "Solwezi (North-Western)":    (-12.173, 26.389, "IIa"),
    "Mongu (Western)":            (-15.254, 23.125, "IIb"),
    "Livingstone (Southern)":     (-17.858, 25.863, "I"),
    "Choma (Southern)":           (-16.806, 26.953, "I"),
    "Chipata (Eastern)":          (-13.636, 32.645, "IIa"),
    "Kasama (Northern)":          (-10.212, 31.180, "III"),
    "Mansa (Luapula)":            (-11.199, 28.894, "III"),
}

# --- Crop coefficients Kc by stage (very simplified FAO-56 style) ---
CROP_KC = {
    "maize": [
        ("initial",    0.35, 20),
        ("dev",        0.75, 25),
        ("mid",        1.15, 40),
        ("late",       0.80, 30),
    ],
    "beans": [
        ("initial",    0.40, 15),
        ("dev",        0.75, 20),
        ("mid",        1.05, 25),
        ("late",       0.80, 20),
    ],
    "rice": [
        ("initial",    1.05, 20),
        ("dev",        1.10, 25),
        ("mid",        1.20, 40),
        ("late",       0.90, 30),
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
        each = int(round(total_mm/2))
        return f"{int(round(total_mm))} mm total (split: {each} mm × 2)"
    each = int(round(total_mm/3))
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
        actions.append("Cool spell during emergence: **delay planting** or use **shallow planting** to improve germination.")

    # — pH / salinity —
    if p["ph"] < 5.5:
        actions.append("Soil is acidic: **apply agricultural lime** to move pH towards 6.0–6.5 before planting.")
    if p["ec"] >= 2.0:
        actions.append("Salinity risk: **avoid KCl**, prefer sulfate forms; plan a **leaching irrigation** after heavy rain.")

    # — Nutrient plan —
    n_need = float(plan["N_rec_kg_ha"])
    p_need = float(plan["P2O5_rec_kg_ha"])
    k_need = float(plan["K2O_rec_kg_ha"])
    if n_need >= 5:
        split_n = max(0, int(round(n_need*0.4)))
        topdress = max(0, int(round(n_need - split_n)))
        actions.append(f"Nitrogen: **{int(round(n_need))} kg/ha** total. Apply **{split_n} kg/ha at planting**, then **{topdress} kg/ha** at 4–6 weeks.")
        if p.get("om_pct", 0) < 2.0:
            actions.append("Boost soil **organic matter** (compost/manure) to supply slow-release N and improve water holding.")
    else:
        actions.append("Nitrogen: **no extra N** required now (OM credit and soil N are adequate).")

    if p_need >= 10:
        actions.append(f"Phosphorus: apply **{int(p_need)} kg/ha P₂O₅** **at planting** (band near seed; don’t mix with urea).")
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
        actions.append(f"Intercrop with **{p['crop2']}**: use **alternate rows** or a **1:1 strip**; keep fertilizer mainly with the main crop’s row.")

    # — Simple housekeeping —
    actions.append("After each rain or irrigation, **check for crusting/ponding** and break crust lightly to help emergence.")
    actions.append("Keep **weeds below 10 cm**; early weeding saves water and nutrients for your crop.")

    return actions
@st.cache_data(ttl=60*30)
def fetch_weather(lat: float, lon: float, days_forward: int = 10, days_past: int = 7, tz: str = "Africa/Lusaka") -> Dict[str, Any]:
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

def compute_density_factor(crop: str, row_cm: float, plant_cm: float, base_density_ha: Optional[float] = None) -> Tuple[float, float]:
    BASE_PLANT_DENSITY_HA = {"maize": 53333, "beans": 200000, "rice": 250000}
    base = base_density_ha or BASE_PLANT_DENSITY_HA.get((crop or "").lower(), 100000.0)
    row_m = max(0.0001, row_cm/100.0); plant_m = max(0.0001, plant_cm/100.0)
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

def irrigation_recommendations(crop: str, planting_date: date, daily: Dict[str, List], yield_factor: float, density_factor: float) -> Tuple[pd.DataFrame, Dict[str, Any]]:
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
        "erosion_days": int((df["Rain_mm"]>=30).sum()),   # very wet days
        "heat_stress_days": int((df["Tmax_C"]>=35).sum()),
        "cool_germination_days": int(((df["Tmax_C"]<18) & (pd.Series(dsp)<=10)).sum()),
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
    n_target = 90.0 * yield_factor   # mid-range
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

def compute_density_factor(crop: str, row_spacing_cm: float, plant_spacing_cm: float) -> tuple[float, float]:
    """
    Return (density_factor, plants_per_ha) from spacings.
    factor is clamped to [DENSITY_FACTOR_MIN, DENSITY_FACTOR_MAX] for stability.
    """
    row_m = max(0.0001, row_spacing_cm / 100.0)
    plant_m = max(0.0001, plant_spacing_cm / 100.0)
    plants_per_ha = 10000.0 / (row_m * plant_m)  # 10,000 m² per ha

    base = BASE_PLANT_DENSITY_HA.get((crop or "").lower(), 100000.0)
    raw_factor = plants_per_ha / base
    factor = clamp(raw_factor, DENSITY_FACTOR_MIN, DENSITY_FACTOR_MAX)
    return factor, plants_per_ha

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

tabs = st.tabs(["Sensor Mode", "Planting recommendations", "30-Day Demo"])

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
        soil_texture = st.selectbox("Soil texture", ["loam", "sand", "clay"], index=0, key="sensor_soil_texture")
        om_pct = st.slider("Organic matter (%)", 0.0, 6.0, 2.0, 0.1, help="Affects N mineralization",
                           key="sensor_om_pct")
        yield_factor = st.slider("Yield target factor", YIELD_FACTOR_MIN, YIELD_FACTOR_MAX, 1.0, 0.05,
                                 key="sensor_yield_factor")
        # Spacing → density
        row_spacing_cm = st.slider("Row spacing (cm)", 20.0, 100.0, 75.0, 5.0, key="sensor_row_spacing_cm")
        plant_spacing_cm = st.slider("Plant spacing in row (cm)", 10.0, 60.0, 25.0, 5.0, key="sensor_plant_spacing_cm")

        density_factor, plants_per_ha = compute_density_factor(chosen_crop, row_spacing_cm, plant_spacing_cm)
        st.caption(f"Estimated density: {plants_per_ha:,.0f} plants/ha  •  factor used: {density_factor:.2f}")

        if st.button("Build Series from Planting → Now"):
            series, daily_alerts = simulate_from_planting(st.session_state.planting_date, chosen_crop)
            st.session_state.sensor_history = series
            st.session_state.last_sensor = series[-1] if series else None
            # store alerts in history (auto-applied)
            record_alert_history("sensor", [
                {"type":a["type"], "priority":a["priority"], "title":a["title"], "action":a["action"]}
                for a in daily_alerts
            ])

        show_n_points = st.slider("Show last N points", 24, 24*120, 24*72, step=24, help="Chart window (hours)", key="sensor_show_n")
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
                        st.write(f"✅ {a['title']} : _{a['action']}_  ({a['time']})")
                else:
                    st.write("No actions yet.")
        else:
            st.info("Click **Build Series from Planting → Now** to generate hourly data.")

# ---------------------------------
# 2) NON-SENSOR (LAB) MODE
# ---------------------------------
with tabs[1]:
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

        st.caption(f"AER **{aer}** : typical annual rainfall {AER[aer]['rain_mm']} mm; soils: {AER[aer]['soil_note']}.")

        planting_date = st.date_input("Planned planting date", value=date.today())
        crop = st.selectbox("Main crop", ["maize", "beans", "rice"])
        # intercropping (optional)
        with st.expander("Add a second crop (optional)"):
            crop2_on = st.checkbox("Enable intercropping")
            crop2 = st.selectbox("Second crop", ["beans","maize","rice"], index=0, disabled=not crop2_on)

        st.subheader("Space between your plants")
        row_cm = st.slider("Row spacing (cm)", 20.0, 60.0, 75.0, 5.0, key="lab_row_spacing_cm")

        plant_cm = st.slider("In-row spacing (cm)", 10.0, 50.0, 25.0, 5.0, key="lab_plant_spacing_cm")
        dens_factor, plants_ha = compute_density_factor(crop, row_cm, plant_cm)
        if crop2_on:
            row2 = st.slider("Row spacing (2nd crop) (cm)", 20.0, 60.0, 45.0, 5.0, key="lab_row_spacing2_cm")
            plant2 = st.slider("In-row spacing (2nd crop) (cm)", 10.0, 50.0, 20.0, 5.0, key="lab_plant_spacing2_cm")

            dens2, plants2 = compute_density_factor(crop2, row2, plant2)
            # combined density capped (simple)
            dens_factor = clamp(dens_factor + 0.5*dens2, 0.8, 1.4)
            plants_ha = plants_ha + 0.5*plants2
        st.caption(f"Estimated stand: **{plants_ha:,.0f} plants/ha**, density factor used **{dens_factor:.2f}**")

        st.subheader("Management & Soil")
        soil_texture = st.selectbox("Soil texture (typical)", ["loam", "sand", "clay"], index=0 if aer != "IIb" else 1,
                                    key="lab_soil_texture")
        om_pct = st.slider("Organic matter (%)", 0.0, 6.0, 2.0, 0.1, key="lab_om_pct")
        yield_factor = st.slider("Target yield factor", 0.8, 1.2, 1.0, 0.05, key="lab_yield_factor")
        bd = st.slider("Bulk density (g/cm³)", 1.1, 1.6, 1.3, 0.05, key="lab_bd")
        depth_m = st.slider("Sampling depth (m)", 0.10, 0.30, 0.20, 0.01, key="lab_depth_m")

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

            # 1) Weather-driven irrigation & risks
            st.markdown("### Water plan (next 10 days)")
            st.dataframe(weekly, use_container_width=True, hide_index=True)
            st.write(f"- **Erosion risk days** (≥30 mm/day): **{risks['erosion_days']}**")
            st.write(f"- **Heat-stress days** (Tmax ≥35°C): **{risks['heat_stress_days']}**")
            st.write(f"- **Cool germination risk** (early stage, low Tmax): **{risks['cool_germination_days']}**")
            st.line_chart(df.set_index("date")[["Rain_mm","ETc_mm","Deficit_mm"]], use_container_width=True)

            # 2) Nutrient plan from lab
            plan = nutrient_plan_from_lab(
                p["crop"], p["yield_factor"], p["om_pct"],
                p["n_kgha"], p["p_mgkg"], p["k_mgkg"], p["bd"], p["depth_m"]
            )
            st.markdown("### Nutrient plan")
            st.write(f"**Nitrogen (N):** apply ~**{plan['N_rec_kg_ha']} kg/ha** (credit from OM: {plan['notes']['n_credit_om']} kg/ha).")
            st.write(f"**Phosphorus (P₂O₅):** **{plan['P2O5_rec_kg_ha']} kg/ha**  •  **Potassium (K₂O):** **{plan['K2O_rec_kg_ha']} kg/ha**")
            st.caption(f"Estimated P pool: {plan['notes']['p_pool_kgha']} kg/ha, K pool: {plan['notes']['k_pool_kgha']} kg/ha (top {p['depth_m']} m).")

            # 3) Condition-specific tips (pH/EC/texture/AER/weather)
            tips = []
            if p["ph"] < 5.5:
                tips += ["Soil is acidic: consider liming to reach ~pH 6.0–6.5 before planting (apply 2–4 months ahead)."]
            if p["ec"] >= 2.0:
                tips += ["High salinity risk: avoid chloride-heavy K sources; schedule leaching irrigation after heavy rains."]
            if p["aer"] == "IIb" or p["soil_texture"] == "sand":
                tips += ["Kalahari sands/sandy soils: add **organic matter** (residues/compost/manure) to improve water & nutrient holding."]

            if risks["erosion_days"] >= 1:
                tips += ["Forecast has ≥30 mm/day rain: keep residue cover, contour ploughing or tied ridges to reduce runoff."]
            if risks["heat_stress_days"] >= 1 and p["crop"] == "maize":
                tips += ["Heat near flowering can cut kernel set: ensure no water stress 1 week before to 2 weeks after tasseling."]

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



