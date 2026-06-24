"""
simulation.py — LOHC dehydrogenation physics
All science in one place. No Streamlit or plotting imports.
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.optimize import brentq

# ── Physical constants ────────────────────────────────────────────────────────
R_GAS = 8.314  # J/(mol·K)

MW = {
    'H18DBT': 290.4,
    'H12DBT': 284.4,
    'H6DBT':  278.4,
    'DBT':    272.4,
    'H2':     2.016,
}

DH_PER_H2 = 65.4e3   # J/mol H2 (endothermic)
DS_PER_H2 = 130.0    # J/(mol·K) per mol H2
DH_STEP   = 3 * DH_PER_H2
DS_STEP   = 3 * DS_PER_H2

RHO_LIQ = 895.0
C_LIQ_0 = (RHO_LIQ * 1e3 / MW['H18DBT']) / 1e3  # mol/L ≈ 3.08

# PFR basis: F0 = 1 mol/s H18-DBT
F0_REF = 1.0
F0_G_H = F0_REF * MW['H18DBT'] * 3600  # g/h ≈ 1.045e6

# ── Kinetic parameters (Pt/Al2O3, calibrated to literature) ──────────────────
K0       = np.array([2.0e5, 1.5e5, 8.0e4])  # L/(kg_cat·s)
EA       = np.array([95e3, 102e3, 110e3])    # J/mol
K_H2_ADS = 0.20                              # bar⁻¹

# ── Helper functions ──────────────────────────────────────────────────────────

def whsv(W_kg):
    """Catalyst mass [kg] → WHSV [h⁻¹] at F0 = 1 mol/s basis."""
    return F0_G_H / (np.asarray(W_kg) * 1000 + 1e-9)


def rate_constants(T):
    """Arrhenius rate constants at T [K]. Returns array [k1, k2, k3]."""
    return K0 * np.exp(-EA / (R_GAS * T))


def kp_equilibrium(T):
    """Equilibrium constant Kp for one dehydrogenation step at T [K]."""
    dG = DH_STEP - T * DS_STEP
    return np.exp(-dG / (R_GAS * T))


def reaction_rates(F, T, p_H2=1.0):
    F_liq       = np.maximum(F[:4], 0.0)
    F_liq_total = F_liq.sum()
    x_liq       = F_liq / (F_liq_total + 1e-30)
    C           = x_liq[:3] * C_LIQ_0
    k           = rate_constants(T)
    return k * C / (1.0 + K_H2_ADS * p_H2)


def pfr_odes(W, F, T, p_H2=1.0):
    r = reaction_rates(F, T, p_H2)
    return [
        -r[0],
         r[0] - r[1],
         r[1] - r[2],
         r[2],
         3.0 * (r[0] + r[1] + r[2]),
    ]


def solve_pfr(T_K, F0_H18=1.0, W_max=3000.0, p_H2=1.0, n_pts=500):
    """Solve isothermal PFR ODEs. T_K in Kelvin."""
    return solve_ivp(
        pfr_odes,
        (0.0, W_max),
        [F0_H18, 0.0, 0.0, 0.0, 0.0],
        args=(T_K, p_H2),
        t_eval=np.linspace(0, W_max, n_pts),
        method='RK45',
        rtol=1e-7,
        atol=1e-10,
    )


# ── High-level result extractors (called by Streamlit cached functions) ───────

def pfr_results(T_C, p_H2, W_max=3000.0):
    """
    Run PFR and return a dict of arrays ready for plotting.
    T_C in Celsius.
    """
    sol = solve_pfr(T_C + 273.15, p_H2=p_H2, W_max=W_max)
    W  = sol.t
    F_H18, F_H12, F_H6, F_DBT, F_H2 = sol.y
    F0 = F_H18[0]

    X   = 1 - F_H18 / F0
    DoD = F_H2 / (9 * F0)
    WS  = whsv(W + 1e-9)

    # Find W and WHSV at 95 % conversion
    mask = W > 5
    W95 = whsv95 = None
    try:
        ip   = interp1d(W[mask], X[mask])
        W95  = brentq(lambda w: ip(w) - 0.95, W[mask][0], W[mask][-1])
        whsv95 = float(whsv(W95))
    except Exception:
        pass

    return {
        'W': W, 'WHSV': WS, 'mask': mask,
        'F_H18': F_H18, 'F_H12': F_H12,
        'F_H6':  F_H6,  'F_DBT': F_DBT, 'F_H2': F_H2,
        'F0': F0, 'X': X, 'DoD': DoD,
        'W95': W95, 'WHSV95': whsv95,
        'max_yield_pct': float(DoD[mask][-1]) * 100,
        'max_X_pct':     float(X[mask][-1])   * 100,
    }


def temperature_sweep(p_H2, W_max=3000.0):
    """Run PFR for 8 temperatures (260–330 °C). Returns dict keyed by T_C."""
    out = {}
    for T_C in range(260, 340, 10):
        sol = solve_pfr(T_C + 273.15, p_H2=p_H2, W_max=W_max)
        W   = sol.t
        DoD = sol.y[4] / (9 * sol.y[0, 0])
        X   = 1 - sol.y[0] / sol.y[0, 0]
        WS  = whsv(W + 1e-9)
        out[T_C] = {'WHSV': WS, 'DoD': DoD, 'X': X, 'W': W}
    return out


def pressure_sweep(T_C, W_max=3000.0):
    """Run PFR for 6 H2 pressures. Returns dict keyed by p_H2."""
    out = {}
    for P in [0.05, 0.25, 0.5, 1.0, 2.0, 5.0]:
        sol = solve_pfr(T_C + 273.15, p_H2=P, W_max=W_max)
        W   = sol.t
        DoD = sol.y[4] / (9 * sol.y[0, 0])
        WS  = whsv(W + 1e-9)
        out[P] = {'WHSV': WS, 'DoD': DoD, 'W': W}
    return out


# ── Techno-economic analysis ──────────────────────────────────────────────────

def tea(T_C, p_H2, plant_mw=1.0, elec_price=0.08,
        pt_price=30_000, discount_rate=0.08, plant_life=20):
    """
    Full TEA for a continuous dehydrogenation plant.

    Parameters
    ----------
    T_C           : operating temperature [°C]
    p_H2          : H2 partial pressure [bar]
    plant_mw      : target H2 output [MW LHV]
    elec_price    : electricity cost [€/kWh]
    pt_price      : Pt spot price [€/kg]
    discount_rate : WACC [-]
    plant_life    : plant lifetime [years]

    Returns
    -------
    dict with CAPEX, OPEX, LCOH, and all breakdowns
    """
    LHV_H2       = 120e6          # J/kg
    MW_H2_KG     = MW['H2'] / 1e3 # kg/mol
    ANNUAL_HOURS = 8_000
    TARGET_X     = 0.95
    PT_LOAD      = 0.005
    AL2O3_PRICE  = 5              # €/kg

    # ── Plant sizing ─────────────────────────────────────────────────────────
    P_W        = plant_mw * 1e6
    F_H2_req   = P_W / LHV_H2 / MW_H2_KG        # mol/s
    F_H18_req  = (F_H2_req / 9) / TARGET_X       # mol/s feed

    # Catalyst mass from PFR result
    sol  = solve_pfr(T_C + 273.15, p_H2=p_H2, W_max=3000.0)
    X    = 1 - sol.y[0] / sol.y[0, 0]
    W_arr = sol.t
    try:
        ip   = interp1d(W_arr[W_arr > 5], X[W_arr > 5])
        W95  = brentq(lambda w: ip(w) - TARGET_X,
                      W_arr[W_arr > 5][0], W_arr[W_arr > 5][-1])
    except Exception:
        W95 = W_arr[-1]
    W_cat = W95 * F_H18_req      # kg catalyst at plant scale

    # ── CAPEX ─────────────────────────────────────────────────────────────────
    cat_cost     = W_cat * (PT_LOAD * pt_price + AL2O3_PRICE)
    reactor_cost = W_cat * 1_200 + 25_000
    thermal_kW   = (F_H2_req * DH_PER_H2) / 1_000
    heater_cost  = thermal_kW * 250
    h2_sep_cost  = 200_000
    ancillary    = 0.30 * (reactor_cost + heater_cost)
    equip_total  = cat_cost + reactor_cost + heater_cost + h2_sep_cost + ancillary
    CAPEX        = 1.8 * equip_total

    # ── OPEX ──────────────────────────────────────────────────────────────────
    elec_annual  = thermal_kW * ANNUAL_HOURS * elec_price
    cat_repl     = cat_cost / 3
    dbt_kgh      = F_H18_req * MW['H18DBT'] / 1e3 * 3600
    dbt_makeup   = dbt_kgh * ANNUAL_HOURS * 0.001 * 3.5
    labor_maint  = 0.05 * CAPEX
    OPEX         = elec_annual + cat_repl + dbt_makeup + labor_maint

    # ── LCOH ──────────────────────────────────────────────────────────────────
    CRF          = (discount_rate * (1 + discount_rate) ** plant_life /
                    ((1 + discount_rate) ** plant_life - 1))
    annual_H2_kg = F_H2_req * MW_H2_KG * 3600 * ANNUAL_HOURS
    LCOH         = (CAPEX * CRF + OPEX) / annual_H2_kg

    # ── Sensitivity helper: recompute LCOH varying one param ─────────────────
    def lcoh_vary(elec_p=elec_price, pt_p=pt_price,
                  cap_scale=1.0, disc=discount_rate, life=plant_life):
        cat_c  = W_cat * (PT_LOAD * pt_p + AL2O3_PRICE)
        eq_c   = cat_c + reactor_cost + heater_cost + h2_sep_cost + \
                 0.30 * (reactor_cost + heater_cost)
        cap    = 1.8 * eq_c * cap_scale
        opex_v = (thermal_kW * ANNUAL_HOURS * elec_p
                  + cat_c / 3 + dbt_makeup + 0.05 * cap)
        crf_v  = disc * (1 + disc) ** life / ((1 + disc) ** life - 1)
        return (cap * crf_v + opex_v) / annual_H2_kg

    sensitivity = {
        'Electricity (0.06-0.12 €/kWh)': (
            lcoh_vary(elec_p=0.06), lcoh_vary(elec_p=0.12)),
        'Pt price (21k-39k €/kg)': (
            lcoh_vary(pt_p=21_000), lcoh_vary(pt_p=39_000)),
        'CAPEX estimate (±25%)': (
            lcoh_vary(cap_scale=0.75), lcoh_vary(cap_scale=1.25)),
        'Discount rate (5%-12%)': (
            lcoh_vary(disc=0.05), lcoh_vary(disc=0.12)),
        'Plant lifetime (15-25 yr)': (
            lcoh_vary(life=25), lcoh_vary(life=15)),
    }

    return {
        'CAPEX': CAPEX, 'OPEX': OPEX, 'LCOH': LCOH,
        'CRF': CRF, 'annual_H2_t': annual_H2_kg / 1e3,
        'thermal_kW': thermal_kW, 'W_cat': W_cat,
        'capex_items': {
            'Catalyst':           cat_cost,
            'Reactor vessel':     reactor_cost,
            'Electric heaters':   heater_cost,
            'H2 separator':       h2_sep_cost,
            'Ancillary':          ancillary,
            'Installation (1.8x)': CAPEX - equip_total,
        },
        'opex_items': {
            'Electricity':         elec_annual,
            'Catalyst replacement': cat_repl,
            'DBT makeup':          dbt_makeup,
            'Labor & maintenance': labor_maint,
        },
        'lcoh_items': {
            'Electricity':    elec_annual * CRF / annual_H2_kg,
            'Catalyst repl.': cat_repl    * CRF / annual_H2_kg,
            'DBT makeup':     dbt_makeup  * CRF / annual_H2_kg,
            'Labor & maint.': labor_maint * CRF / annual_H2_kg,
            'CAPEX (ann.)':   CAPEX * CRF  / annual_H2_kg,
        },
        'sensitivity': sensitivity,
    }
