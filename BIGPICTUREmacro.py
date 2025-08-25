#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from fredapi import Fred
import datetime as dt
import numpy as np

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(layout="wide", page_title="Macro Dashboard")
fred = Fred(api_key="b81975b746a1ad458cb81ca5843153e6")  # <= ta clé FRED

# -----------------------------
# DATES
# -----------------------------
START_2020 = dt.datetime(2020, 1, 1)
START_2024 = dt.datetime(2024, 1, 1)
TODAY = dt.datetime.today()

# -----------------------------
# HELPERS
# -----------------------------
def last_val(series: pd.Series) -> float:
    return float(series.dropna().iloc[-1])

def pct(n: float) -> str:
    return f"{n:.2f}%"

# -----------------------------
# DATA DOWNLOAD
# -----------------------------
# PIB réel (trimestriel) - 2020+
gdp = fred.get_series("GDPC1", START_2020, TODAY).dropna()
gdp.index = pd.to_datetime(gdp.index)
gdp_ma = gdp.rolling(8).mean()

# VIX (2020+), robust au schéma de colonnes
vix_df = yf.download("^VIX", start=START_2020, end=TODAY, progress=False)
vix = vix_df["Adj Close"] if "Adj Close" in vix_df.columns else vix_df["Close"]
vix = vix.dropna()

# BOFA ICE AAA / BBB (2020+), codes FRED corrects
aaa = fred.get_series("BAMLC0A1CAAA", START_2020, TODAY).dropna()
bbb = fred.get_series("BAMLC0A4CBBB", START_2020, TODAY).dropna()

# Treasury 2Y & 10Y (2024 seulement)
y2 = fred.get_series("GS2", START_2024, TODAY).dropna()
y10 = fred.get_series("GS10", START_2024, TODAY).dropna()

# -----------------------------
# CALCULS — PIB (règles Excel)
# -----------------------------
g_last = gdp.iloc[-1]
g_prev = gdp.iloc[-2]
g_prev_year = gdp.iloc[-5]  # 4 trimestres en arrière

qoq = (g_last / g_prev - 1) * 100                    # trimestriel (%)
qoq_annual = ((g_last / g_prev) ** 4 - 1) * 100      # QoQ annualisé (%)
yoy = (g_last / g_prev_year - 1) * 100               # YoY (%)

# Phase (logique Excel)
if (qoq < 0) and (yoy < 0):
    gdp_phase = "Récession"
elif (qoq > 0) and (qoq < 0.5) and (yoy < 0):
    gdp_phase = "Creux"
elif (qoq > 0) and (yoy > 0):
    gdp_phase = "Reprise"
elif (qoq > 1.5) and (yoy > 1):
    gdp_phase = "Expansion"
elif (qoq < 0.15) and (yoy > 1):
    gdp_phase = "Pic"
elif (qoq > 0) and (qoq < 1.5) and (yoy < 1):
    gdp_phase = "Contraction"
else:
    gdp_phase = "Ralentissement"

# -----------------------------
# CALCULS — VIX (règle Excel)
# -----------------------------
vix_last = last_val(vix)
if vix_last < 15:
    vix_env = "Risk-On"
elif vix_last <= 20:
    vix_env = "Neutre"
else:
    vix_env = "Risk-Off"
sp_move_potential = vix_last / 16 / 100.0  # en %

# -----------------------------
# CALCULS — Yield Curve (règles Excel)
# -----------------------------
yc = pd.concat([y2.rename("y2"), y10.rename("y10")], axis=1).dropna()
yc["spread"] = yc["y10"] - yc["y2"]
spread_curr = yc["spread"].iloc[-1]
spread_prev = yc["spread"].iloc[-2]

if (spread_curr < 0) and (spread_prev >= 0):
    yield_phase = "Inversion"
elif (spread_curr < 0) and (spread_curr < spread_prev):
    yield_phase = "Aplatissement"
elif (spread_curr < 0) and (spread_curr > spread_prev):
    yield_phase = "Repentification"
elif abs(spread_curr) < 0.05:
    yield_phase = "Courbe plate"
elif spread_curr > spread_prev:
    yield_phase = "Repentification"
else:
    yield_phase = "Aplatissement"

y2_last = float(yc["y2"].iloc[-1])
y10_last = float(yc["y10"].iloc[-1])

# -----------------------------
# CALCULS — BOFA AAA / BBB (règles Excel)
# -----------------------------
def rolling_mean_3m(s: pd.Series) -> pd.Series:
    return s.shift(1).rolling(3).mean()  # moyenne des 3 mois précédents

aaa_df = pd.DataFrame({"taux": aaa})
aaa_df["moyenne"] = rolling_mean_3m(aaa_df["taux"])
bbb_df = pd.DataFrame({"taux": bbb})
bbb_df["moyenne"] = rolling_mean_3m(bbb_df["taux"])

def risk_cycle(row):
    if pd.isna(row["moyenne"]):
        return "Neutre"
    if row["taux"] > row["moyenne"]:
        return "Risk off"
    if row["taux"] < row["moyenne"]:
        return "Risk on"
    return "Neutre"

aaa_df["cycle"] = aaa_df.apply(risk_cycle, axis=1)
bbb_df["cycle"] = bbb_df.apply(risk_cycle, axis=1)

aaa_last, aaa_mean_last, aaa_cycle_last = float(aaa_df["taux"].iloc[-1]), float(aaa_df["moyenne"].iloc[-1]), aaa_df["cycle"].iloc[-1]
bbb_last, bbb_mean_last, bbb_cycle_last = float(bbb_df["taux"].iloc[-1]), float(bbb_df["moyenne"].iloc[-1]), bbb_df["cycle"].iloc[-1]

# -----------------------------
#  CYCLE + SECTEURS (MAPPING EXCEL)  <<< AJOUTÉ ICI AVANT LAYOUT
# -----------------------------
def determine_cycle_from_excel_rules(gdp_phase: str, yield_phase: str) -> str:
    mapping = {
        ("Expansion", "Repentification"): "Expansion / Boom",
        ("Expansion", "Inversion"): "Fin de cycle / Tension",
        ("Ralentissement", "Inversion"): "Entrée en récession",
        ("Récession", "Inversion"): "Récession confirmée",
        ("Récession", "Repentification"): "Reprise probable",
        ("Reprise", "Repentification"): "Reprise confirmée",
        ("Reprise", "Aplatissement"): "Croissance fragile",
        ("Contraction", "Inversion"): "Ralentissement confirmé",
        ("Pic", "Inversion"): "Fin de cycle probable",
        ("Creux", "Repentification"): "Creux / Tension sur rebond",
    }
    return mapping.get((gdp_phase, yield_phase), "Cycle incertain")

cycle_macro = determine_cycle_from_excel_rules(gdp_phase, yield_phase)

SECTORS_UP = {
    "Expansion / Boom": ["Manufacturing", "Retail Trade", "Educational Services"],
    "Fin de cycle / Tension": ["Health Care", "Real Estate", "Utilities"],
    "Entrée en récession": ["Utilities", "Educational Services", "Health Care"],
    "Récession confirmée": ["Utilities", "Health Care"],
    "Reprise probable": ["Retail Trade", "Manufacturing", "Transportation & Warehousing"],
    "Reprise confirmée": ["Retail Trade", "Manufacturing", "Wholesale Trade"],
    "Croissance fragile": ["Real Estate", "Health Care", "Utilities"],
    "Cycle incertain": ["Utilities", "Health Care", "Consumer Staples", "Telecommunications"],
    "Ralentissement confirmé": ["Utilities", "Health Care", "Real Estate"],
    "Fin de cycle probable": ["Staples", "Telecoms", "Real Estate"],
    "Creux / Tension sur rebond": ["Industrials", "Construction", "Materials"],
}
SECTORS_DOWN = {
    "Expansion / Boom": ["Utilities", "Staples", "Health Care"],
    "Fin de cycle / Tension": ["Cyclicals", "Industrials", "Banks"],
    "Entrée en récession": ["Discretionary", "Materials", "Banks"],
    "Récession confirmée": ["Discretionnaire", "Énergie", "Industries"],
    "Reprise probable": ["Temporary Defensives: Utilities, Staples"],
    "Reprise confirmée": ["Health Care", "Staples"],
    "Croissance fragile": ["Cyclicals", "Materials"],
    "Cycle incertain": ["Cyclicals", "Industrials", "Discretionary"],
    "Ralentissement confirmé": ["Banks", "Discretionary", "Industrials"],
    "Fin de cycle probable": ["Industrials", "Cyclicals"],
    "Creux / Tension sur rebond": ["Defensives", "Utilities", "Staples"],
}
sectors_up_final = SECTORS_UP.get(cycle_macro, [])
sectors_down_final = SECTORS_DOWN.get(cycle_macro, [])

# -----------------------------
# LAYOUT
# -----------------------------
st.title("🌍 Macro Dashboard — Cycle Économique (style Excel)")

# ===== Ligne 1 : PIB & VIX =====
c1, c2 = st.columns(2)

with c1:
    st.subheader("PHASE PIB RÉEL (2020+)")
    fig, ax = plt.subplots(figsize=(7,3.2))
    ax.plot(gdp.index, gdp, label="PIB Réel", linewidth=2.2, color="#FFA500")
    ax.plot(gdp_ma.index, gdp_ma, "--", label="Moy. mobile 8 trimestres", linewidth=2, color="#DDDDDD")
    ax.set_facecolor("#000000")
    ax.legend(facecolor="#D0D0D0")
    st.pyplot(fig, clear_figure=True)

    st.markdown(
        f"""
        <div style="background:#f4f4f4;padding:10px;border:2px solid #ff8c00">
            <b>QoQ annualisé :</b> {pct(qoq_annual)}<br>
            <b>YoY :</b> {pct(yoy)}<br>
            <b>Cycle trimestriel PIB réel :</b> {gdp_phase}
        </div>
        """,
        unsafe_allow_html=True
    )

with c2:
    st.subheader("VIX (2020+)")
    fig, ax = plt.subplots(figsize=(7,3.2))
    ax.plot(vix.index, vix, linewidth=1.6, color="#FFA500")
    ax.set_facecolor("#000000")
    st.pyplot(fig, clear_figure=True)

    st.markdown(
        f"""
        <div style="background:#f4f4f4;padding:10px;border:2px solid #ff8c00">
            <b>Cotation :</b> {vix_last:.2f}<br>
            <b>Environnement Risk :</b> {vix_env}<br>
            <b>Mouv. potentiel S&amp;P :</b> {sp_move_potential:.2%}
        </div>
        """,
        unsafe_allow_html=True
    )

# ===== Ligne 2 : Yield Curve & BOFA =====
c3, c4 = st.columns(2)

with c3:
    st.subheader("YIELD CURVE (2024)")
    fig, ax = plt.subplots(figsize=(7,3.2))
    ax.plot(yc.index, yc["y2"], label="2Y", linewidth=2.0, color="#FFA500")
    ax.plot(yc.index, yc["y10"], label="10Y", linewidth=2.0, color="#FFFFFF")
    ax.set_facecolor("#000000")
    ax.legend(facecolor="#D0D0D0")
    st.pyplot(fig, clear_figure=True)

    st.markdown(
        f"""
        <div style="background:#f4f4f4;padding:10px;border:2px solid #ff8c00">
            <b>2Y % :</b> {y2_last:.2f}<br>
            <b>10Y % :</b> {y10_last:.2f}<br>
            <b>Cycle courbe des taux annualisé :</b> {yield_phase}
        </div>
        """,
        unsafe_allow_html=True
    )

with c4:
    st.subheader("AGGREGATE BONDS — ICE BofA AAA & BBB (2020+)")
    fig, ax = plt.subplots(figsize=(7,3.2))
    ax.plot(aaa_df.index, aaa_df["taux"], label="AAA", linewidth=2.0, color="#FFA500")
    ax.plot(bbb_df.index, bbb_df["taux"], label="BBB", linewidth=2.0, color="#FFFFFF")
    ax.set_facecolor("#000000")
    ax.legend(facecolor="#D0D0D0")
    st.pyplot(fig, clear_figure=True)

    st.markdown(
        f"""
        <div style="background:#f4f4f4;padding:10px;border:2px solid #ff8c00">
            <b>Taux AAA :</b> {aaa_last:.2f} &nbsp; | &nbsp; <b>Moyenne (3m) :</b> {aaa_mean_last:.2f} &nbsp; | &nbsp; <b>Cycle AAA :</b> {aaa_cycle_last}<br>
            <b>Taux BBB :</b> {bbb_last:.2f} &nbsp; | &nbsp; <b>Moyenne (3m) :</b> {bbb_mean_last:.2f} &nbsp; | &nbsp; <b>Cycle BBB :</b> {bbb_cycle_last}
        </div>
        """,
        unsafe_allow_html=True
    )

# ===== Ligne 3 : Synthèse Cycle & Secteurs =====
st.markdown("---")
s1, s2, s3 = st.columns(3)
with s1:
    st.markdown("### 🌀 Cycle Économique Possible")
    st.markdown(
        f"<div style='background:#ffffff;padding:18px;border:2px solid #ff8c00;text-align:center;font-weight:700'>{cycle_macro}</div>",
        unsafe_allow_html=True
    )
with s2:
    st.markdown("### 📈 Secteur UP")
    st.markdown(
        f"<div style='background:#e7f6e7;padding:18px;border:2px solid #ff8c00;text-align:center'>{', '.join(sectors_up_final)}</div>",
        unsafe_allow_html=True
    )
with s3:
    st.markdown("### 📉 Secteur DOWN")
    st.markdown(
        f"<div style='background:#fde3e3;padding:18px;border:2px solid #ff8c00;text-align:center'>{', '.join(sectors_down_final)}</div>",
        unsafe_allow_html=True
    )

