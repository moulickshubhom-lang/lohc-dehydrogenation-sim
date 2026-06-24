# LOHC Dehydrogenation: Process Simulation & TEA

**Steady-state PFR model + techno-economic analysis for the catalytic dehydrogenation of dibenzyltoluene (DBT)**

---

## Overview

Liquid Organic Hydrogen Carriers (LOHCs) enable safe, large-scale hydrogen storage at ambient conditions.
This project builds a **computational process model** for the catalytic release (dehydrogenation) step
of the DBT/H₁₈-DBT system, paired with a **techno-economic analysis (TEA)** estimating the Levelised
Cost of Hydrogen (LCOH) for a 1 MW_H₂ plant.

Built as a portfolio project by a chemical engineer specialising in LOHC systems (HI ERN, Erlangen).

---

## What's inside

| File | Description |
|------|-------------|
| `LOHC_Dehydrogenation_Simulation.ipynb` | Main Jupyter notebook — all simulation + TEA |
| `figures/` | Auto-generated plots (created on first run) |
| `requirements.txt` | Python dependencies |

### Notebook structure

1. **System parameters & thermodynamics** — Kp vs T, ΔG spontaneity threshold
2. **Kinetic model** — Langmuir-Hinshelwood rate expressions, Arrhenius analysis
3. **Isothermal PFR simulation** — SciPy ODE solver, species profiles
4. **Parametric studies** — temperature (260–330 °C), H₂ back-pressure (0.05–5 bar)
5. **Techno-economic analysis** — CAPEX, OPEX breakdown, LCOH calculation
6. **Sensitivity analysis** — tornado chart for 6 key parameters

---

## Key results

| | |
|---|---|
| H₂ gravimetric capacity (H₁₈-DBT) | **6.2 wt%** |
| Onset of thermodynamic spontaneity | **~230 °C** |
| Recommended operating range | **290–310 °C**, P_H₂ < 1 bar |
| Rate-limiting step | H₆-DBT → DBT (highest Eₐ = 110 kJ/mol) |
| LCOH (dehydrogenation only, 1 MW plant) | see notebook output |
| Dominant cost driver | Electricity for endothermic heat supply |

---

## Reaction network

```
H₁₈-DBT  ──k₁──►  H₁₂-DBT + 3H₂   ΔH = +196 kJ/mol
H₁₂-DBT  ──k₂──►  H₆-DBT  + 3H₂   ΔH = +196 kJ/mol
H₆-DBT   ──k₃──►  DBT      + 3H₂   ΔH = +196 kJ/mol
────────────────────────────────────────────────────────
Overall:  H₁₈-DBT → DBT + 9H₂       ΔH = +588 kJ/mol
```

Catalyst: **0.5 wt% Pt/Al₂O₃** (fixed bed)  
Kinetics: Langmuir-Hinshelwood with competitive H₂ adsorption inhibition

---

## Setup

```bash
git clone https://github.com/<your-username>/lohc-dehydrogenation-sim
cd lohc-dehydrogenation-sim
pip install -r requirements.txt
jupyter notebook LOHC_Dehydrogenation_Simulation.ipynb
```

---

## References

- Jorschick, H. et al. (2019). *Hydrogen-rich organic liquid carrier compounds as an interconnector
  between renewable and fossil energy systems.* Applied Energy.
- Müller, K. et al. (2021). *Evaluation of current developments and future challenges in LOHC
  technology.* Energy & Environmental Science.
- Niermann, M. et al. (2021). *Liquid organic hydrogen carriers — Techno-economic analysis of a
  future energy system.* Energy & Environmental Science.
- Hydrogenious LOHC Technologies GmbH — product data for Marlotherm SH (DBT).

---

## License

MIT — free to use and adapt with attribution.
