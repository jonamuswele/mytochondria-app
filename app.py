
import math
import random
from datetime import datetime, timedelta, date
from typing import List, Dict, Any, Optional, Tuple

import streamlit as st
import pandas as pd

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

# ------------------------------
# Data Generators (ported from dataGenerators.ts)  :contentReference[oaicite:2]{index=2}
# ------------------------------

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
    """
    Estimate days until a nutrient falls to 'floor_pct' (LOW threshold).
    Very simple linear model (MVP) — enough for demo and date outputs.
    """
    if current_pct <= floor_pct:
        return 0
    if weekly_drop_pct <= 0:
        return None
    weeks_needed = (current_pct - floor_pct) / weekly_drop_pct
    days = int(round(weeks_needed * 7))
    return max(days, 0)

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
    c1, c2 = st.columns([2,1])
    with c2:
        st.subheader("Planting")
        st.session_state.planting_date = st.date_input(
            "Planting Date",
            value=st.session_state.planting_date,
            help="Used for stage-based forecasts"
        )

        st.markdown("**Stream Controls**")
        advance_hours = st.number_input("Advance hours", 1, 24, 1)
        if st.button("Generate Next Reading(s)"):
            for i in range(int(advance_hours)):
                ts = datetime.now() + timedelta(hours=i)
                point = gen_sensor_point(ts)
                st.session_state.sensor_history.append(point)

        if st.button("Reset Stream"):
            st.session_state.sensor_history = []
            st.session_state.last_sensor = None

        show_n_points = st.slider("Show last N points", min_value=10, max_value=300, value=100, step=10)

    with c1:
        st.subheader("Live Sensor Charts (hourly simulated)")
        if st.session_state.sensor_history:
            df = pd.DataFrame(st.session_state.sensor_history[-show_n_points:])
            df = df.sort_values("timestamp")
            st.line_chart(df.set_index("timestamp")[["moisture", "temperature"]], use_container_width=True)
            st.line_chart(df.set_index("timestamp")[["ph", "ec"]], use_container_width=True)
            st.line_chart(df.set_index("timestamp")[["n", "p", "k"]], use_container_width=True)

            latest = df.iloc[-1].to_dict()
            latest["planting_date"] = st.session_state.planting_date
            insights = generate_insights(latest)

            st.subheader("Insights")
            for i, ins in enumerate(insights):
                with st.expander(f"{ins['icon']} [{ins['priority'].upper()}] {ins['title']}", expanded=(i == 0)):
                    st.write(ins["description"])
                    if ins.get("action"):
                        task_id = f"sensor-{i}"
                        already = st.session_state.checklist.get(task_id, {"label": ins["action"], "done": False, "applied_effect": False})
                        done = st.checkbox(f"Mark done: {ins['action']}", value=already["done"], key=task_id)
                        st.session_state.checklist[task_id] = {"label": ins["action"], "done": done, "applied_effect": already.get("applied_effect", False)}
                        # apply effect if newly done
                        if done and not st.session_state.checklist[task_id]["applied_effect"]:
                            apply_action_effects(latest, ins["action"])
                            st.session_state.checklist[task_id]["applied_effect"] = True
            st.caption("NPK categories now: N = {}, P = {}, K = {}".format(
                pct_to_cat(latest["n"]), pct_to_cat(latest["p"]), pct_to_cat(latest["k"])
            ))
        else:
            st.info("Click **Generate Next Reading(s)** to start the sensor stream.")

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
    if st.button("Generate 30-Day Demo Data"):
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


