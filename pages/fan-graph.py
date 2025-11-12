import math
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Fan Selection – ΔT Method", page_icon="🌀", layout="wide")

# --------------------------
# Constants & helpers
# --------------------------
AIR_CP = 1005.0          # J/(kg·K)
AIR_DENSITY = 1.2        # kg/m³
K_COEFF = AIR_CP * AIR_DENSITY / 3600.0  # ≈ 0.335 W per (m³/h·K)

def required_airflow(power_w: float, delta_t_k: float) -> float:
    """Required airflow in m³/h to dissipate `power_w` at ΔT."""
    if delta_t_k <= 0:
        return float("inf")
    return power_w / (K_COEFF * delta_t_k)

def power_from_airflow(airflow_m3h: float, delta_t_k: float) -> float:
    """Power dissipated in W for given airflow and ΔT."""
    return K_COEFF * airflow_m3h * delta_t_k

@st.cache_data
def default_fans():
    # Name, airflow (m3/h), noise (dBA), power draw (W)
    return pd.DataFrame([
        {"name": "Fan 24",  "airflow_m3h": 24,  "noise_dBA": 25, "power_W": 2.5},
        {"name": "Fan 55",  "airflow_m3h": 55,  "noise_dBA": 28, "power_W": 3.0},
        {"name": "Fan 100", "airflow_m3h": 100, "noise_dBA": 30, "power_W": 4.0},
        {"name": "Fan 250", "airflow_m3h": 250, "noise_dBA": 34, "power_W": 6.0},
        {"name": "Fan 400", "airflow_m3h": 400, "noise_dBA": 38, "power_W": 8.0},
        {"name": "Fan 500", "airflow_m3h": 500, "noise_dBA": 41, "power_W": 10.0},
        {"name": "Fan 630", "airflow_m3h": 630, "noise_dBA": 44, "power_W": 12.0},
        {"name": "Fan 800", "airflow_m3h": 800, "noise_dBA": 47, "power_W": 16.0},
        {"name": "Fan 1000","airflow_m3h": 1000,"noise_dBA": 50, "power_W": 20.0},
    ])

# --------------------------
# Sidebar inputs
# --------------------------
st.sidebar.header("Thermal inputs")
colP, colMode = st.sidebar.columns([1,1], gap="small")
power_w = colP.number_input("Power dissipation P (W)", min_value=1.0, max_value=5000.0, value=600.0, step=10.0)

mode = colMode.radio("ΔT input mode", ["Manual ΔT", "From temperatures"], index=0)

if mode == "Manual ΔT":
    delta_t_limit = st.sidebar.number_input("Allowed ΔT (K)", min_value=1.0, max_value=40.0, value=10.0, step=1.0)
else:
    amb = st.sidebar.number_input("Ambient (°C)", value=25.0, step=0.5)
    internal_setpoint = st.sidebar.number_input("Max cabinet inside (°C)", value=35.0, step=0.5)
    delta_t_limit = max(1.0, internal_setpoint - amb)  # avoid 0 or negative
    st.sidebar.caption(f"ΔT = {delta_t_limit:.1f} K")

st.sidebar.divider()
margin_pct = st.sidebar.number_input("Airflow safety margin (%)", value=20, min_value=0, max_value=200, step=5,
                                     help="Adds margin to required airflow for filters/ducts/obstructions.")

st.sidebar.divider()
st.sidebar.subheader("Fan catalog")
uploaded = st.sidebar.file_uploader("Upload CSV (name, airflow_m3h, noise_dBA, power_W)", type=["csv"])

if "fans_df" not in st.session_state:
    st.session_state["fans_df"] = default_fans()

if uploaded:
    try:
        user_df = pd.read_csv(uploaded)
        needed_cols = {"name", "airflow_m3h"}
        if not needed_cols.issubset(user_df.columns):
            st.sidebar.error("CSV must include at least: 'name', 'airflow_m3h'")
        else:
            st.session_state["fans_df"] = user_df
    except Exception as e:
        st.sidebar.error(f"Failed to parse CSV: {e}")

fans_df = st.session_state["fans_df"].copy()
fans_df = fans_df.sort_values("airflow_m3h").reset_index(drop=True)

# --------------------------
# Headline & metrics
# --------------------------
st.title("🌀 Fan Selection (ΔT method)")
st.caption("Formula: **P = ρ · cₚ · (m³/h / 3600) · ΔT ≈ 0.335 · airflow · ΔT** (linear scales).")

raw_req_airflow = required_airflow(power_w, delta_t_limit)
req_airflow = raw_req_airflow * (1.0 + margin_pct/100.0)
reco_idx = (fans_df["airflow_m3h"] >= req_airflow).idxmin() if (fans_df["airflow_m3h"] >= req_airflow).any() else None
recommended = fans_df.loc[reco_idx] if reco_idx is not None else None

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Power", f"{power_w:,.0f} W")
m2.metric("ΔT limit", f"{delta_t_limit:.1f} K")
m3.metric("Required airflow (no margin)", f"{raw_req_airflow:,.0f} m³/h")
m4.metric("Required airflow (+margin)", f"{req_airflow:,.0f} m³/h")
if recommended is not None:
    m5.metric("Recommended fan", f"{recommended['name']} ({recommended['airflow_m3h']:,.0f} m³/h)")
else:
    m5.metric("Recommended fan", "No fan large enough")

# --------------------------
# Selection chart (linear)
# --------------------------
st.subheader("Selection chart")
iso_dts = [5, 10, 15, 20, 25, 30]  # Keep same isolines as your image
max_airflow_axis = max(1.2*req_airflow, fans_df["airflow_m3h"].max()*1.15, 300.0)
max_power_axis = max(power_w*1.35, power_from_airflow(max_airflow_axis, max(iso_dts)), 800.0)

fig = go.Figure()

# ΔT ISO-LINES (green)
x_vals = list(range(0, int(max_airflow_axis)+1, max(1, int(max_airflow_axis//60))))  # ~60 points
for dt in iso_dts:
    y_vals = [power_from_airflow(x, dt) for x in x_vals]
    fig.add_trace(go.Scatter(
        x=x_vals, y=y_vals, mode="lines", name=f"{dt} K",
        line=dict(width=2),
        hovertemplate="Airflow: %{x:.0f} m³/h<br>P: %{y:.0f} W<br>ΔT: "+str(dt)+" K<extra></extra>"
    ))

# Required point + red dashed guides (like the catalog)
fig.add_trace(go.Scatter(
    x=[req_airflow], y=[power_w], mode="markers",
    marker=dict(size=10, symbol="x"),
    name="Required point",
    hovertemplate="Required airflow: %{x:.0f} m³/h<br>Power: %{y:.0f} W<extra></extra>",
))
fig.add_hline(y=power_w, line=dict(dash="dash"), line_color="red",
              annotation_text=f"P = {power_w:.0f} W", annotation_position="left")
fig.add_vline(x=req_airflow, line=dict(dash="dash"), line_color="red",
              annotation_text=f"{req_airflow:.0f} m³/h", annotation_position="top")

# Axis & grid styling (keeps linear scale)
fig.update_layout(
    xaxis_title="Air Flow (m³/h)",
    yaxis_title="Power dissipation (W)",
    legend_title="Temperature difference (ΔT) inside vs outside",
    height=540,
    margin=dict(l=10, r=10, t=40, b=10),
)
fig.update_xaxes(range=[0, max_airflow_axis], showgrid=True, gridwidth=1, gridcolor="rgba(0,0,0,0.1)")
fig.update_yaxes(range=[0, max_power_axis], showgrid=True, gridwidth=1, gridcolor="rgba(0,0,0,0.1)")

st.plotly_chart(fig, use_container_width=True)

# --------------------------
# Fan catalog (editable) + highlight
# --------------------------
st.subheader("Fan options")
left, right = st.columns([1, 1], gap="large")

with left:
    st.markdown("**Catalog** (edit inline)")
    fans_df = st.data_editor(
        fans_df,
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        column_config={
            "name": st.column_config.TextColumn("Name"),
            "airflow_m3h": st.column_config.NumberColumn("Airflow (m³/h)", min_value=0),
            "noise_dBA": st.column_config.NumberColumn("Noise (dBA)", min_value=0, step=0.5),
            "power_W": st.column_config.NumberColumn("Power draw (W)", min_value=0, step=0.1),
        },
        key="editor_fans"
    )
    st.session_state["fans_df"] = fans_df

    # quick badges
    meets = (fans_df["airflow_m3h"] >= req_airflow).sum()
    st.caption(f"✅ Meets requirement: **{meets}**  |  ❌ Below requirement: **{len(fans_df)-meets}**")

    # Optional CSV download
    csv = fans_df.to_csv(index=False).encode("utf-8")
    st.download_button("💾 Download catalog CSV", data=csv, file_name="fan_catalog.csv", mime="text/csv")

with right:
    fans_df = fans_df.copy()
    fans_df["meets_requirement"] = fans_df["airflow_m3h"] >= req_airflow
    fans_df["label"] = fans_df.apply(lambda r: f"{r['name']} — {r['airflow_m3h']:.0f} m³/h", axis=1)

    # Choose a fan to highlight
    sel = st.selectbox(
        "Highlight a fan",
        options=["(none)"] + fans_df["name"].tolist(),
        index=0
    )

    bar_fig = px.bar(
        fans_df.sort_values("airflow_m3h"),
        x="airflow_m3h",
        y="label",
        orientation="h",
        title="Airflow of available fans",
        hover_data=["noise_dBA", "power_W", "airflow_m3h", "name"],
    )
    bar_fig.add_vline(x=req_airflow, line=dict(dash="dash", color="red"))

    # highlight selected fan with a marker
    if sel != "(none)":
        row = fans_df.loc[fans_df["name"] == sel].iloc[0]
        bar_fig.add_trace(go.Scatter(
            x=[row["airflow_m3h"]],
            y=[f"{row['name']} — {row['airflow_m3h']:.0f} m³/h"],
            mode="markers",
            marker=dict(size=12, symbol="diamond"),
            name=f"Selected: {row['name']}",
            hovertemplate="Selected: %{y}<extra></extra>"
        ))

    bar_fig.update_layout(xaxis_title="Air Flow (m³/h)", yaxis_title="")
    st.plotly_chart(bar_fig, use_container_width=True)

# --------------------------
# Notes
# --------------------------
with st.expander("Method & assumptions"):
    st.write(
        """
- Standard air assumed at ~20 °C: density 1.2 kg/m³, heat capacity 1005 J/(kg·K).
- Heat balance (linear): \\(P = \\rho c_p \\dot V \\Delta T\\), with \\(\\dot V\\) in m³/s.
- Converting to m³/h ⇒ **P ≈ 0.335 · airflow(m³/h) · ΔT(K)**.
- Add margin for filters, grilles, duct losses, altitude, hot spots, etc.
        """
    )
