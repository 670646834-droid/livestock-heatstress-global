# Global Livestock Heat Stress Loss Model

Endogenized global livestock heat stress loss model with piecewise dose-response functions, humidity-dependent cooling efficiency, and cost-effectiveness constrained adaptation.

## Key Results (2024, THI>=72, 5-year payback)
- Global gross losses: $279.15 billion (2005 constant USD)
- Avoidable losses: $26.41 billion (9.5%), sensitivity 8.1-13.5% across 3-10 year horizons
- Residual losses: $252.74 billion (90.5%)
- Methane intensity increase: 0.8-3.3% across species, 0.1881 Gt CO2eq/year additional
- 2000-2024 loss growth: +47.6%

## Data Sources
- GLW4 (Wisser & Cinardi 2024): global livestock distribution 0.083 deg
- NASA POWER (Gelaro 2017): daily T2M + RH2M
- ERA5 (Hersbach 2020): global stratified sampling sensitivity
- FAOSTAT (FAO 2025): production, prices, deflators

## Repository Structure
- scripts/: analysis and modeling scripts
- data/: key output datasets (CSV)

## Citation
Submitted to Nature Food, 2026.
