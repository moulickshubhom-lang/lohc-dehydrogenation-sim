"""
app.py — LOHC Dehydrogenation Simulator
Reactive Streamlit app. Run with: streamlit run app.py
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from simulation import (
    pfr_results, temperature_sweep, pressure_sweep, tea,
    kp_equilibrium, rate_constants, MW, DH_STEP, DS_STEP, R_GAS,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LOHC Dehydrogenation Simulator",
    page_icon="⚗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Colour palette (consistent with notebook) ─────────────────────────────────
CLR = {
    'H18': '#E8593C', 'H12': '#F2A623',
    'H6':  '#1D9E75', 'DBT': '#534AB7', 'H2': '#185FA5',
}
CAPEX_COLORS = ['#534AB7', '#185FA5', '#E8593C', '#1D9E75', '#F2A623', '#888780']
OPEX_COLORS  = ['#E8593C', '#534AB7', '#1D9E75', '#F2A623']

# ── Cached compute functions ──────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def cached_pfr(T_C, p_H2):
    return pfr_results(T_C, p_H2)

@st.cache_data(show_spinner=False)
def cached_T_sweep(p_H2):
    return temperature_sweep(p_H2)

@st.cache_data(show_spinner=False)
def cached_P_sweep(T_C):
    return pressure_sweep(T_C)

@st.cache_data(show_spinner=False)
def cached_tea(T_C, p_H2, plant_mw, elec_price, pt_price, discount_rate):
    return tea(T_C, p_H2, plant_mw, elec_price, pt_price, discount_rate)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚗️ Parameters")

    st.subheader("Reactor conditions")
    T_C  = st.slider("Temperature [°C]",        260, 330, 300, 5)
    p_H2 = st.slider("H₂ partial pressure [bar]", 0.05, 5.0, 1.0, 0.05,
                     format="%.2f")

    st.subheader("Plant economics")
    plant_mw      = st.slider("Plant capacity [MW H₂]",    0.1, 10.0, 1.0, 0.1)
    elec_price    = st.slider("Electricity price [€/kWh]", 0.04, 0.20, 0.08, 0.01,
                              format="%.2f")
    pt_price      = st.slider("Pt price [€/kg]",    15_000, 50_000, 30_000, 1_000,
                              format="%d")
    discount_rate = st.slider("Discount rate [%]", 4, 12, 8, 1) / 100.0

    st.divider()
    st.caption(
        "DBT/H₁₈-DBT system · Isothermal PFR · "
        "Pt/Al₂O₃ · Langmuir-Hinshelwood kinetics\n\n"
        "[GitHub](https://github.com/moulickshubhom-lang/lohc-dehydrogenation-sim)"
    )

# ── Header ────────────────────────────────────────────────────────────────────
st.title("LOHC Dehydrogenation — Process Simulator")
st.caption(
    "Steady-state PFR simulation + techno-economic analysis for "
    "H₁₈-DBT → DBT + 9 H₂ over Pt/Al₂O₃"
)

# ── Run cached computations ───────────────────────────────────────────────────
with st.spinner("Computing..."):
    sim   = cached_pfr(T_C, p_H2)
    econ  = cached_tea(T_C, p_H2, plant_mw, elec_price, pt_price, discount_rate)

# ── Top KPI row ───────────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
k1.metric("WHSV at 95% conversion",
          f"{sim['WHSV95']:.3f} h⁻¹" if sim['WHSV95'] else "—")
k2.metric("Max H₂ yield",
          f"{sim['max_yield_pct']:.1f} %")
k3.metric("LCOH (dehydrogenation)",
          f"€ {econ['LCOH']:.2f} / kg H₂")
k4.metric("Thermal duty",
          f"{econ['thermal_kW']:.0f} kW_th")

st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["⚗️ Reactor", "📈 Parametric study", "💶 Economics"])

# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 · REACTOR
# ════════════════════════════════════════════════════════════════════════════════
with tab1:
    col1, col2 = st.columns(2)

    # ── Species profiles ────────────────────────────────────────────────────
    with col1:
        WS   = sim['WHSV']
        mask = sim['mask']
        F0   = sim['F0']

        fig = go.Figure()
        for name, F_i, clr, dash in [
            ('H₁₈-DBT', sim['F_H18'], CLR['H18'], 'solid'),
            ('H₁₂-DBT', sim['F_H12'], CLR['H12'], 'solid'),
            ('H₆-DBT',  sim['F_H6'],  CLR['H6'],  'solid'),
            ('DBT',     sim['F_DBT'], CLR['DBT'], 'solid'),
            ('H₂ / 9',  sim['F_H2']/9, CLR['H2'],  'dash'),
        ]:
            fig.add_trace(go.Scatter(
                x=WS[mask], y=(F_i / F0)[mask],
                name=name,
                line=dict(color=clr, dash=dash, width=2.2),
                hovertemplate='WHSV: %{x:.3f} h⁻¹<br>F/F₀: %{y:.3f}<extra>' + name + '</extra>',
            ))

        if sim['WHSV95']:
            fig.add_vline(
                x=sim['WHSV95'], line_dash='dot', line_color='gray', line_width=1.2,
                annotation_text=f"WHSV₉₅ = {sim['WHSV95']:.3f} h⁻¹",
                annotation_position='top right', annotation_font_size=11,
            )

        fig.update_xaxes(type='log', autorange='reversed',
                         title='WHSV [h⁻¹]  →  increasing contact time')
        fig.update_yaxes(title='Normalised molar flow  F / F₀')
        fig.update_layout(
            title=f'Species profiles — {T_C} °C, P_H₂ = {p_H2:.2f} bar',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, x=0),
            height=420, margin=dict(t=80),
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Conversion & H2 yield ───────────────────────────────────────────────
    with col2:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=WS[mask], y=(sim['X'] * 100)[mask],
            name='X(H₁₈-DBT)', line=dict(color=CLR['H18'], width=2.2),
            hovertemplate='WHSV: %{x:.3f} h⁻¹<br>X: %{y:.1f}%<extra>Conversion</extra>',
        ))
        fig2.add_trace(go.Scatter(
            x=WS[mask], y=(sim['DoD'] * 100)[mask],
            name='H₂ yield / DoD', line=dict(color=CLR['H2'], width=2.2, dash='dash'),
            hovertemplate='WHSV: %{x:.3f} h⁻¹<br>Yield: %{y:.1f}%<extra>H₂ yield</extra>',
        ))

        if sim['WHSV95']:
            fig2.add_vline(
                x=sim['WHSV95'], line_dash='dot', line_color='gray', line_width=1.2,
            )

        fig2.update_xaxes(type='log', autorange='reversed',
                          title='WHSV [h⁻¹]  →  increasing contact time')
        fig2.update_yaxes(title='[%]', range=[0, 105])
        fig2.update_layout(
            title=f'Conversion & H₂ yield — {T_C} °C, P_H₂ = {p_H2:.2f} bar',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, x=0),
            height=420, margin=dict(t=80),
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ── H6-DBT bottleneck callout ───────────────────────────────────────────
    h6_peak_idx = int(np.argmax(sim['F_H6']))
    h6_peak_ws  = float(WS[h6_peak_idx])
    h6_peak_val = float(sim['F_H6'][h6_peak_idx] / F0 * 100)
    st.info(
        f"**H₆-DBT accumulation peak:** {h6_peak_val:.1f}% of feed at "
        f"WHSV ≈ {h6_peak_ws:.2f} h⁻¹ — this is the rate-limiting bottleneck "
        f"(step 3 has the highest activation energy, Eₐ = 110 kJ/mol)."
    )

# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 · PARAMETRIC STUDY
# ════════════════════════════════════════════════════════════════════════════════
with tab2:
    with st.spinner("Running parametric sweeps..."):
        t_sweep = cached_T_sweep(p_H2)
        p_sweep = cached_P_sweep(T_C)

    col1, col2 = st.columns(2)

    # ── Temperature sweep ───────────────────────────────────────────────────
    with col1:
        t_colors = [
            '#313695','#4575b4','#74add1','#abd9e9',
            '#fee090','#fdae61','#f46d43','#d73027',
        ]
        fig_t = go.Figure()
        for (T_key, res), clr in zip(sorted(t_sweep.items()), t_colors):
            mask_t = res['W'] > 5
            fig_t.add_trace(go.Scatter(
                x=res['WHSV'][mask_t], y=(res['DoD'] * 100)[mask_t],
                name=f'{T_key} °C',
                line=dict(color=clr, width=2),
                hovertemplate=f'{T_key}°C — WHSV: %{{x:.3f}} h⁻¹, yield: %{{y:.1f}}%<extra></extra>',
            ))

        fig_t.update_xaxes(type='log', autorange='reversed',
                           title='WHSV [h⁻¹]  →  increasing contact time')
        fig_t.update_yaxes(title='H₂ yield [%]', range=[0, 105])
        fig_t.update_layout(
            title=f'Temperature effect on H₂ yield  (P_H₂ = {p_H2:.2f} bar)',
            legend=dict(title='Temperature', orientation='v'),
            height=440, margin=dict(t=60),
        )
        st.plotly_chart(fig_t, use_container_width=True)

    # ── H2 back-pressure sweep ──────────────────────────────────────────────
    with col2:
        import plotly.express as px
        p_colors = px.colors.sequential.Blues[1:]
        P_cases = [0.05, 0.25, 0.5, 1.0, 2.0, 5.0]
        fig_p = go.Figure()
        for P, clr in zip(P_cases, p_colors):
            res   = p_sweep[P]
            mask_p = res['W'] > 5
            fig_p.add_trace(go.Scatter(
                x=res['WHSV'][mask_p], y=(res['DoD'] * 100)[mask_p],
                name=f'{P} bar',
                line=dict(color=clr, width=2),
                hovertemplate=f'P={P} bar — WHSV: %{{x:.3f}} h⁻¹, yield: %{{y:.1f}}%<extra></extra>',
            ))

        fig_p.update_xaxes(type='log', autorange='reversed',
                           title='WHSV [h⁻¹]  →  increasing contact time')
        fig_p.update_yaxes(title='H₂ yield [%]', range=[0, 105])
        fig_p.update_layout(
            title=f'H₂ back-pressure inhibition — {T_C} °C',
            legend=dict(title='P_H₂', orientation='v'),
            height=440, margin=dict(t=60),
        )
        st.plotly_chart(fig_p, use_container_width=True)

    st.caption(
        "Both sweeps are cached — move the sidebar sliders and only the "
        "affected sweep recomputes."
    )

# ════════════════════════════════════════════════════════════════════════════════
# TAB 3 · ECONOMICS
# ════════════════════════════════════════════════════════════════════════════════
with tab3:
    # ── KPIs ────────────────────────────────────────────────────────────────
    e1, e2, e3, e4 = st.columns(4)
    e1.metric("LCOH",         f"€ {econ['LCOH']:.2f} / kg H₂")
    e2.metric("CAPEX",        f"€ {econ['CAPEX']/1e6:.2f} M")
    e3.metric("OPEX",         f"€ {econ['OPEX']/1e3:.0f} k / yr")
    e4.metric("Annual H₂",    f"{econ['annual_H2_t']:.0f} t / yr")

    st.divider()
    col1, col2 = st.columns(2)

    # ── CAPEX bar ────────────────────────────────────────────────────────────
    with col1:
        names  = list(econ['capex_items'].keys())
        values = [v / 1e3 for v in econ['capex_items'].values()]
        fig_c = go.Figure(go.Bar(
            x=names, y=values,
            marker_color=CAPEX_COLORS,
            text=[f'€{v:.0f}k' for v in values],
            textposition='outside',
        ))
        fig_c.update_layout(
            title=f'CAPEX breakdown  (Total: €{econ["CAPEX"]/1e6:.2f} M)',
            yaxis_title='Cost [k€]',
            height=380, margin=dict(t=60, b=100),
            showlegend=False,
        )
        fig_c.update_xaxes(tickangle=-30)
        st.plotly_chart(fig_c, use_container_width=True)

    # ── OPEX pie ─────────────────────────────────────────────────────────────
    with col2:
        names_o  = list(econ['opex_items'].keys())
        values_o = list(econ['opex_items'].values())
        fig_o = go.Figure(go.Pie(
            labels=names_o, values=values_o,
            marker_colors=OPEX_COLORS,
            hole=0.38,
            textinfo='label+percent',
            hovertemplate='%{label}: €%{value:,.0f}/yr<extra></extra>',
        ))
        fig_o.update_layout(
            title=f'OPEX breakdown  (€{econ["OPEX"]/1e3:.0f} k / yr)',
            height=380, margin=dict(t=60),
        )
        st.plotly_chart(fig_o, use_container_width=True)

    # ── LCOH contribution bar ────────────────────────────────────────────────
    col3, col4 = st.columns(2)
    with col3:
        lnames  = list(econ['lcoh_items'].keys())
        lvalues = list(econ['lcoh_items'].values())
        fig_l = go.Figure(go.Bar(
            y=lnames, x=lvalues,
            orientation='h',
            marker_color=OPEX_COLORS + ['#185FA5'],
            text=[f'€{v:.3f}' for v in lvalues],
            textposition='outside',
        ))
        fig_l.update_layout(
            title=f'LCOH contribution per cost driver  (€{econ["LCOH"]:.2f}/kg H₂)',
            xaxis_title='€ / kg H₂',
            height=360, margin=dict(t=60, r=80),
            showlegend=False,
        )
        st.plotly_chart(fig_l, use_container_width=True)

    # ── Tornado chart ────────────────────────────────────────────────────────
    with col4:
        base   = econ['LCOH']
        sens   = econ['sensitivity']
        params = list(sens.keys())
        lows   = [v[0] for v in sens.values()]
        highs  = [v[1] for v in sens.values()]

        # Sort by swing
        swings = [h - l for h, l in zip(highs, lows)]
        order  = sorted(range(len(swings)), key=lambda i: swings[i])
        params = [params[i] for i in order]
        lows   = [lows[i]   for i in order]
        highs  = [highs[i]  for i in order]

        fig_s = go.Figure()
        fig_s.add_trace(go.Bar(
            y=params, x=[h - base for h in highs],
            base=base, orientation='h',
            name='High case', marker_color='#E8593C',
            hovertemplate='%{y}<br>High: €%{x:.2f}/kg H₂<extra></extra>',
        ))
        fig_s.add_trace(go.Bar(
            y=params, x=[l - base for l in lows],
            base=base, orientation='h',
            name='Low case', marker_color='#185FA5',
            hovertemplate='%{y}<br>Low: €%{x:.2f}/kg H₂<extra></extra>',
        ))
        fig_s.add_vline(x=base, line_width=1.5, line_color='black',
                        annotation_text=f'Base: €{base:.2f}',
                        annotation_position='top', annotation_font_size=11)
        fig_s.update_layout(
            title='Sensitivity — tornado chart',
            xaxis_title='LCOH [€/kg H₂]',
            barmode='overlay',
            height=360, margin=dict(t=60),
            legend=dict(orientation='h', yanchor='bottom', y=1.02, x=0),
        )
        st.plotly_chart(fig_s, use_container_width=True)

    st.info(
        "**Note:** This models the dehydrogenation step only. "
        "The full LOHC cycle (hydrogenation + logistics) adds approximately "
        "€0.5–1.0 / kg H₂ on top of the figure shown."
    )
