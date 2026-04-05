# DeveillanceFit
Ultrasonic Phased Array Simulation — Deveillance Applied Physics Trial

Two N×N transducer arrays at 40 kHz beamsteered to a common focal point. Models beam interaction via linear superposition with nonlinear extensions (second-harmonic generation, Gor'kov radiation force potential).

## Quick Start

Using uv (recommended):

uv add k-wave-python

## Output Files (in `output/`)

| File | Description |
|------|-------------|
| `01_linear_pressure_fields.png` | 6-panel: individual + combined fields, linear & dB |
| `02_nonlinear_effects.png` | 6-panel: fundamental, 2nd harmonic, harmonic ratio, total RMS, Gor'kov potential, radiation force field |
| `03_cross_sections.png` | 4-panel: horizontal & vertical cuts, harmonic comparison, Gor'kov potential trap sites |
| `04_kwave_comparison.png` | Side-by-side analytical vs k-Wave on matched domain (if k-Wave installed) |
| `assumptions_log.txt` | Physical assumptions with justifications |
| `interaction_writeup.txt` | Superposition model + nonlinear effects discussion |
| `next_steps.txt` | What I'd improve with 2 more hours |

## Dependencies

| Package | Version | Required | Purpose |
|---------|---------|----------|---------|
| `numpy` | >= 1.24 | Yes | Array computation, Green's function |
| `matplotlib` | >= 3.7 | Yes | Plotting |
| `scipy` | >= 1.10 | Yes | Signal processing (trap-site detection) |
| `k-wave-python` | >= 0.4.0 | No | FDTD time-domain solver |

## Physics

### Phase Calculation (Beamsteering)
Each element gets a focusing phase `phi_n = k * |r_focal - r_n|` so all contributions arrive in-phase at the focal point. No optimizer needed — this is the geometric delay-and-sum.

### Linear Superposition
```
p_total(r) = p_array1(r) + p_array2(r)
```
Coherent complex addition preserving phase. At the focus, both beams constructively interfere (~6 dB gain).

### Nonlinear: Second-Harmonic Generation
Fubini-Blackstock quasilinear perturbation:
```
|p_2f|(r) ≈ (β·k·|p_f|²) / (2·ρ₀·c₀²) · σ_eff
```
where β = 1.2 for air, and σ_eff is capped at the shock formation distance.

### Nonlinear: Gor'kov Radiation Force
```
U = V_p · [f₀·⟨p²⟩/(4ρ₀c₀²) − 3f₁ρ₀·⟨v²⟩/8]
F = −∇U
```
Local minima of U are stable acoustic levitation sites.

### k-Wave 
k-Wave solves the full wave equation with `medium.BonA = 0.4` for cumulative nonlinearity. Both methods use the exact same physical domain `[-5, 30] × [-20, 20] cm` for direct comparison.

## Physical Parameters

| Parameter | Value | Justification |
|-----------|-------|---------------|
| Frequency | 40 kHz (λ ≈ 8.575 mm) | Problem statement |
| Medium | Air, c = 343 m/s, ρ = 1.225 kg/m³ | Room temperature |
| Elements | 8 per array, λ/2 pitch | Grating-lobe-free |
| Arrays | y = ±10 cm, same x | Problem geometry |
| Focus | (15, 0) cm | Chosen focal point |
| B/(2A) | 0.2 (β = 1.2) | Air nonlinearity |
| Particle | EPS, ρ = 29 kg/m³, a = 1 mm | Levitation standard |

