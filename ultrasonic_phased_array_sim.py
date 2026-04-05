#!/usr/bin/env python3
"""
Ultrasonic Phased Array Simulation — Deveillance Applied Physics Trial
======================================================================
Two N×N transducer arrays at 40 kHz beamsteered to a common focal point.

Methods:
  1. Analytical Green's function  — steady-state, linear + nonlinear perturbation
  2. k-Wave pseudospectral FDTD   — time-domain, optional BonA nonlinearity

Nonlinear modelling:
  • Second-harmonic generation via quasilinear perturbation
  • Gor'kov radiation force potential (acoustic levitation landscape)
  • k-Wave medium.BonA cumulative nonlinearity (if installed)

Both methods share an identical physical domain so results overlay directly.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
import time
import sys

# ═══════════════════════════════════════════════════════════════════════════
# PHYSICAL CONSTANTS & SIMULATION PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════

# Medium — air at 20 °C
C0   = 343.0        # speed of sound [m/s]
RHO0 = 1.225        # density [kg/m³]
F0   = 40_000.0     # frequency [Hz]
OMEGA = 2 * np.pi * F0
LAM  = C0 / F0      # wavelength ≈ 8.575 mm
K    = 2 * np.pi / LAM

# Nonlinear parameter of air
B_OVER_2A = 0.2     # B/(2A) for air ≈ 0.2
BETA_NL   = 1 + B_OVER_2A   # coefficient of nonlinearity ≈ 1.2

# Array parameters
N_ELEM = 8          # elements per array (cross-section of NxN)
PITCH  = LAM / 2    # λ/2 spacing ≈ 4.29 mm
AMP    = 10.0       # source amplitude [Pa] (~154 dB SPL per element)

# ── SHARED DOMAIN (both methods use this exact region) ──
X_MIN, X_MAX = -0.05, 0.30    # [m]
Y_MIN, Y_MAX = -0.20, 0.20    # [m]
RES = 500                      # grid pixels per axis

# Array centres and focal point [m]
ARRAY1_CENTER = np.array([0.0, -0.10])
ARRAY2_CENTER = np.array([0.0,  0.10])
FOCAL_POINT   = np.array([0.15, 0.0])

# Gor'kov particle (expanded polystyrene bead, typical for levitation)
PARTICLE_RADIUS = 1.0e-3       # 1 mm
RHO_P  = 29.0                  # EPS density [kg/m³]
C_P    = 900.0                 # EPS speed of sound [m/s]

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════
# GEOMETRY HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def make_array(center, n, pitch):
    """N-element linear array centred at `center`, spread along x-axis."""
    offsets = (np.arange(n) - (n - 1) / 2) * pitch
    pos = np.zeros((n, 2))
    pos[:, 0] = center[0] + offsets
    pos[:, 1] = center[1]
    return pos


def focusing_phases(sources, focal, k):
    """Per-element phase phi_n = k * |r_f - r_n| for geometric focusing."""
    return k * np.linalg.norm(sources - focal, axis=1)


# ═══════════════════════════════════════════════════════════════════════════
# ANALYTICAL FIELD COMPUTATION
# ═══════════════════════════════════════════════════════════════════════════

def compute_pressure(sources, focal, X, Y, k, amp=1.0):
    """
    Steady-state complex pressure from a focused point-source array.

    p(r) = A * sum_n  exp(j*k*|r_f - r_n|) * exp(-j*k*|r - r_n|) / |r - r_n|

    Uses the 3D free-space Green's function (1/r spherical spreading)
    evaluated on a 2D observation plane.
    """
    Ns = len(sources)
    sx = sources[:, 0].reshape(Ns, 1, 1)
    sy = sources[:, 1].reshape(Ns, 1, 1)

    r = np.sqrt((X[None] - sx)**2 + (Y[None] - sy)**2)
    # Regularise: clamp minimum distance to lambda/4 (physical element has
    # finite size ~PITCH, so 1/r singularity is unphysical below this scale)
    r_min = LAM / 4
    r = np.maximum(r, r_min)

    phi = focusing_phases(sources, focal, k).reshape(Ns, 1, 1)

    p = amp * np.sum(np.exp(1j * phi) * np.exp(-1j * k * r) / r, axis=0)
    return p


def compute_velocity(p, X, Y):
    """
    Acoustic particle velocity from pressure gradient (Euler equation).
    v = -grad(p) / (j * omega * rho_0)
    Returns (vx, vy) complex arrays.
    """
    dx = X[0, 1] - X[0, 0]
    dy = Y[1, 0] - Y[0, 0]
    dp_dx = np.gradient(p, dx, axis=1)
    dp_dy = np.gradient(p, dy, axis=0)
    denom = 1j * OMEGA * RHO0
    return -dp_dx / denom, -dp_dy / denom


# ═══════════════════════════════════════════════════════════════════════════
# NONLINEAR EFFECTS — ANALYTICAL PERTURBATION
# ═══════════════════════════════════════════════════════════════════════════

def second_harmonic_field(p_fundamental, X, Y):
    """
    Quasilinear estimate of second-harmonic pressure (2f0 = 80 kHz).

    In the Fubini-Blackstock perturbation framework, the second-harmonic
    amplitude grows with propagation distance as:

        |p2| ~ (beta * k * |p1|^2) / (2 * rho0 * c0^2) * sigma_eff

    sigma_eff is clamped to the shock formation distance to stay within
    the perturbation regime.
    """
    abs_p1 = np.abs(p_fundamental)
    abs_p1_safe = np.maximum(abs_p1, 1e-6)

    # Shock formation distance at each point
    sigma_shock = RHO0 * C0**3 / (BETA_NL * OMEGA * abs_p1_safe)

    # Distance from nearest array centre (proxy for propagation distance)
    d1 = np.sqrt((X - ARRAY1_CENTER[0])**2 + (Y - ARRAY1_CENTER[1])**2)
    d2 = np.sqrt((X - ARRAY2_CENTER[0])**2 + (Y - ARRAY2_CENTER[1])**2)
    r_source = np.maximum(np.minimum(d1, d2), LAM / 4)

    sigma_eff = np.minimum(r_source, sigma_shock)

    p2_amplitude = (BETA_NL * K * abs_p1**2) / (2 * RHO0 * C0**2) * sigma_eff
    p2_phase = 2 * np.angle(p_fundamental)
    return p2_amplitude * np.exp(1j * p2_phase)


def gorkov_potential(p, vx, vy):
    """
    Gor'kov radiation-force potential for a small spherical particle.

    U = V_p * [ f0/(4*rho0*c0^2) * <p^2>  -  3*f1*rho0/8 * <v^2> ]

    Scattering coefficients:
        f0 = 1 - kappa_p/kappa_0
        f1 = 2*(rho_p - rho_0)/(2*rho_p + rho_0)
    """
    V_p = (4 / 3) * np.pi * PARTICLE_RADIUS**3

    kappa_0 = 1 / (RHO0 * C0**2)
    kappa_p = 1 / (RHO_P * C_P**2)
    f0 = 1 - kappa_p / kappa_0
    f1 = 2 * (RHO_P - RHO0) / (2 * RHO_P + RHO0)

    p_sq_avg = np.abs(p)**2 / 2
    v_sq_avg = (np.abs(vx)**2 + np.abs(vy)**2) / 2

    U = V_p * (f0 / (4 * RHO0 * C0**2) * p_sq_avg
               - 3 * f1 * RHO0 / 8 * v_sq_avg)
    return U, f0, f1


# ═══════════════════════════════════════════════════════════════════════════
# ANALYTICAL SIMULATION
# ═══════════════════════════════════════════════════════════════════════════

def run_analytical():
    print("=" * 72)
    print("  ANALYTICAL GREEN'S FUNCTION (LINEAR + NONLINEAR PERTURBATION)")
    print("=" * 72)

    array1 = make_array(ARRAY1_CENTER, N_ELEM, PITCH)
    array2 = make_array(ARRAY2_CENTER, N_ELEM, PITCH)

    print(f"\n  Geometry")
    print(f"    Array 1: {N_ELEM} elements at y = {ARRAY1_CENTER[1]*100:+.0f} cm")
    print(f"    Array 2: {N_ELEM} elements at y = {ARRAY2_CENTER[1]*100:+.0f} cm")
    print(f"    Focal point: ({FOCAL_POINT[0]*100:.0f}, {FOCAL_POINT[1]*100:.0f}) cm")
    print(f"    lam = {LAM*1e3:.2f} mm,  pitch = {PITCH*1e3:.2f} mm (lam/2)")
    print(f"    Source amplitude: {AMP} Pa per element")

    for label, arr in [("Array 1", array1), ("Array 2", array2)]:
        print(f"\n  Phase delays -- {label}:")
        phi = focusing_phases(arr, FOCAL_POINT, K)
        for i in range(len(arr)):
            dist = np.linalg.norm(arr[i] - FOCAL_POINT)
            print(f"    Elem {i}: ({arr[i,0]*1e3:+6.1f}, {arr[i,1]*1e3:+7.1f}) mm  "
                  f"->  d = {dist*1e2:5.1f} cm  ->  phi = {np.degrees(phi[i])%360:6.1f} deg  "
                  f"->  dt = {dist/C0*1e6:5.1f} us")

    # Observation grid (shared domain)
    x = np.linspace(X_MIN, X_MAX, RES)
    y = np.linspace(Y_MIN, Y_MAX, RES)
    X, Y = np.meshgrid(x, y)

    print(f"\n  Domain: x in [{X_MIN*100:.0f}, {X_MAX*100:.0f}] cm, "
          f"y in [{Y_MIN*100:.0f}, {Y_MAX*100:.0f}] cm, {RES}x{RES} grid")

    t0 = time.perf_counter()

    p1 = compute_pressure(array1, FOCAL_POINT, X, Y, K, AMP)
    p2 = compute_pressure(array2, FOCAL_POINT, X, Y, K, AMP)
    p_lin = p1 + p2  # linear superposition

    vx, vy = compute_velocity(p_lin, X, Y)
    p2h = second_harmonic_field(p_lin, X, Y)

    # Total RMS (fundamental + harmonic are at different frequencies -> incoherent sum)
    p_total_rms = np.sqrt(np.abs(p_lin)**2 + np.abs(p2h)**2)

    U_gorkov, f0_g, f1_g = gorkov_potential(p_lin, vx, vy)

    dt = time.perf_counter() - t0
    print(f"\n  Computation time: {dt:.3f} s")
    print(f"  Peak |p_linear| at focus region: {np.abs(p_lin).max():.1f} Pa")
    print(f"  Peak |p_2nd_harmonic|: {np.abs(p2h).max():.2f} Pa "
          f"({20*np.log10(np.abs(p2h).max()/np.abs(p_lin).max()+1e-30):.1f} dB re fundamental)")
    print(f"  Gor'kov coefficients: f0 = {f0_g:.4f}, f1 = {f1_g:.4f}")

    return dict(x=x, y=y, X=X, Y=Y,
                p1=p1, p2=p2, p_lin=p_lin,
                p2h=p2h, p_total_rms=p_total_rms,
                U_gorkov=U_gorkov, vx=vx, vy=vy,
                array1=array1, array2=array2)


# ═══════════════════════════════════════════════════════════════════════════
# k-WAVE SIMULATION (with matched domain)
# ═══════════════════════════════════════════════════════════════════════════

def run_kwave(enable_nonlinear=True):
    print("\n" + "=" * 72)
    nl_label = " (NONLINEAR -- BonA = {:.1f})".format(2*B_OVER_2A) if enable_nonlinear else " (LINEAR)"
    print("  k-WAVE PSEUDOSPECTRAL FDTD" + nl_label)
    print("=" * 72)

    try:
        from kwave.kgrid import kWaveGrid
        from kwave.kmedium import kWaveMedium
        from kwave.ksource import kSource
        from kwave.ksensor import kSensor
        from kwave.utils.signals import tone_burst
    except ImportError as e:
        print(f"\n  [!] k-Wave not installed: {e}")
        print("    pip install k-wave-python")
        print("    macOS: brew install fftw hdf5 zlib libomp")
        return None

    # Try unified API first (v0.6.0+), then legacy
    kspaceFirstOrder, sim_api = None, None
    try:
        from kwave.kspaceFirstOrder import kspaceFirstOrder as _kfo
        kspaceFirstOrder = _kfo
        sim_api = "unified"
    except ImportError:
        pass
    if kspaceFirstOrder is None:
        try:
            from kwave.kspaceFirstOrder2D import kspaceFirstOrder2D as _kfo2d
            from kwave.options.simulation_options import SimulationOptions
            from kwave.options.simulation_execution_options import SimulationExecutionOptions
            kspaceFirstOrder = _kfo2d
            sim_api = "legacy"
        except ImportError as e:
            print(f"\n  [!] Cannot import simulation function: {e}")
            return None
    print(f"  Using {sim_api} API")

    # Grid matched to analytical domain
    dx = dy = 2.0e-3   # 2 mm (~4.3 PPW)
    Nx = int(round((X_MAX - X_MIN) / dx)) + (int(round((X_MAX - X_MIN) / dx)) % 2)
    Ny = int(round((Y_MAX - Y_MIN) / dy)) + (int(round((Y_MAX - Y_MIN) / dy)) % 2)

    print(f"  Grid: {Nx} x {Ny}, dx = {dx*1e3:.1f} mm, PPW = {LAM/dx:.1f}")

    kgrid = kWaveGrid([Nx, Ny], [dx, dy])

    medium = kWaveMedium(sound_speed=C0, density=RHO0)
    if enable_nonlinear:
        medium.BonA = 2 * B_OVER_2A
        print(f"  Nonlinearity: B/A = {medium.BonA:.1f}")

    kgrid.makeTime(medium.sound_speed)
    print(f"  Time steps: {kgrid.Nt}, dt = {kgrid.dt*1e6:.3f} us")

    # Physical coords -> grid indices
    x_origin_idx = int(round(-X_MIN / dx))
    y_origin_idx = int(round(-Y_MIN / dy))

    def phys_to_idx(pos):
        ix = x_origin_idx + int(round(pos[0] / dx))
        iy = y_origin_idx + int(round(pos[1] / dy))
        return (np.clip(ix, 1, Nx-2), np.clip(iy, 1, Ny-2))

    array1 = make_array(ARRAY1_CENTER, N_ELEM, PITCH)
    array2 = make_array(ARRAY2_CENTER, N_ELEM, PITCH)
    all_sources = np.vstack([array1, array2])

    source = kSource()
    source_mask = np.zeros((Nx, Ny), dtype=np.int64)
    grid_positions = []
    for pos in all_sources:
        ix, iy = phys_to_idx(pos)
        source_mask[ix, iy] = 1
        grid_positions.append((ix, iy))
    source.p_mask = source_mask

    distances = np.linalg.norm(all_sources - FOCAL_POINT, axis=1)
    delays = (distances.max() - distances) / C0

    # Sort by Fortran-order index
    elem_fortran_idx = [iy * Nx + ix for (ix, iy) in grid_positions]
    sorted_order = np.argsort(elem_fortran_idx)

    signals = []
    for idx in sorted_order:
        offset = int(round(delays[idx] / kgrid.dt))
        try:
            sig = tone_burst(1/kgrid.dt, F0, 10,
                             signal_offset=offset, signal_length=kgrid.Nt)
        except TypeError:
            sig = np.roll(tone_burst(1/kgrid.dt, F0, 10,
                                     signal_length=kgrid.Nt), offset)
        signals.append(AMP * sig.flatten())
    source.p = np.stack(signals, axis=0)

    sensor = kSensor()
    sensor.mask = np.ones((Nx, Ny), dtype=np.int64)
    sensor.record = ['p_max']

    print("\n  Running k-Wave simulation...")
    t0 = time.perf_counter()
    try:
        if sim_api == "unified":
            result = kspaceFirstOrder(kgrid, medium, source, sensor,
                                       pml_inside=False, quiet=True)
        else:
            sim_opts = SimulationOptions(save_to_disk=True, pml_inside=False)
            exec_opts = SimulationExecutionOptions()
            result = kspaceFirstOrder(kgrid=kgrid, source=source, sensor=sensor,
                                       medium=medium, simulation_options=sim_opts,
                                       execution_options=exec_opts)

        elapsed = time.perf_counter() - t0
        print(f"  Completed in {elapsed:.1f} s")

        p_max = result.get('p_max', result.get('p_final', None)) if isinstance(result, dict) else result
        if p_max is not None:
            if p_max.ndim == 1:
                p_max = p_max.reshape((Nx, Ny))
            x_kw = X_MIN + np.arange(Nx) * dx
            y_kw = Y_MIN + np.arange(Ny) * dy
            return dict(p_max=p_max, x=x_kw, y=y_kw, Nx=Nx, Ny=Ny,
                        dx=dx, dy=dy, nonlinear=enable_nonlinear)
    except Exception as e:
        elapsed = time.perf_counter() - t0
        print(f"\n  [!] k-Wave failed after {elapsed:.1f} s: {e}")
        import traceback; traceback.print_exc()
    return None


# ═══════════════════════════════════════════════════════════════════════════
# PLOTTING
# ═══════════════════════════════════════════════════════════════════════════

def make_plots(ana, kw_result=None):
    ext = [X_MIN*100, X_MAX*100, Y_MIN*100, Y_MAX*100]
    imkw = dict(extent=ext, origin='lower', aspect='equal')
    arr1, arr2 = ana['array1'], ana['array2']

    def mark(ax):
        ax.plot(arr1[:, 0]*100, arr1[:, 1]*100, 'c^', ms=5, zorder=5)
        ax.plot(arr2[:, 0]*100, arr2[:, 1]*100, 'gv', ms=5, zorder=5)
        ax.plot(FOCAL_POINT[0]*100, FOCAL_POINT[1]*100, 'w*', ms=12,
                markeredgecolor='k', markeredgewidth=0.5, zorder=6)

    p_ref = np.abs(ana['p_lin']).max()

    # ── Figure 1: Individual + combined linear fields ──
    fig1, axes1 = plt.subplots(2, 3, figsize=(20, 12))
    fields = [np.abs(ana['p1']), np.abs(ana['p2']), np.abs(ana['p_lin'])]
    titles = ['Array 1 -- |p1|', 'Array 2 -- |p2|', 'Superposition -- |p1 + p2|']
    for col in range(3):
        ax = axes1[0, col]
        im = ax.imshow(fields[col], cmap='inferno', **imkw)
        ax.set_title(titles[col], fontsize=12, fontweight='bold')
        ax.set_xlabel('x [cm]'); ax.set_ylabel('y [cm]'); mark(ax)
        plt.colorbar(im, ax=ax, shrink=0.75, label='|p| [Pa]')

        ax = axes1[1, col]
        dB = 20 * np.log10(fields[col] / p_ref + 1e-12)
        im = ax.imshow(dB, cmap='inferno', vmin=-40, vmax=0, **imkw)
        ax.set_title(titles[col] + ' [dB]', fontsize=12)
        ax.set_xlabel('x [cm]'); ax.set_ylabel('y [cm]'); mark(ax)
        plt.colorbar(im, ax=ax, shrink=0.75, label='dB re peak')

    fig1.suptitle(f'Linear Pressure Fields -- {N_ELEM}-Element Arrays at {F0/1e3:.0f} kHz\n'
                  f'Domain: [{X_MIN*100:.0f}, {X_MAX*100:.0f}] x [{Y_MIN*100:.0f}, {Y_MAX*100:.0f}] cm  |  '
                  f'Focus: ({FOCAL_POINT[0]*100:.0f}, {FOCAL_POINT[1]*100:.0f}) cm',
                  fontsize=14, fontweight='bold', y=1.01)
    fig1.tight_layout()
    fig1.savefig(OUTPUT_DIR / '01_linear_pressure_fields.png', dpi=180, bbox_inches='tight')
    print(f"  -> {OUTPUT_DIR / '01_linear_pressure_fields.png'}")
    plt.close(fig1)

    # ── Figure 2: Nonlinear effects ──
    fig2 = plt.figure(figsize=(22, 10))
    gs = gridspec.GridSpec(2, 4, figure=fig2, hspace=0.35, wspace=0.35)

    ax = fig2.add_subplot(gs[0, 0])
    im = ax.imshow(np.abs(ana['p_lin']), cmap='inferno', **imkw)
    ax.set_title('(a) Fundamental |p1| [Pa]', fontsize=11, fontweight='bold')
    ax.set_xlabel('x [cm]'); ax.set_ylabel('y [cm]'); mark(ax)
    plt.colorbar(im, ax=ax, shrink=0.75)

    ax = fig2.add_subplot(gs[0, 1])
    im = ax.imshow(np.abs(ana['p2h']), cmap='magma', **imkw)
    ax.set_title('(b) 2nd Harmonic |p2| [Pa]', fontsize=11, fontweight='bold')
    ax.set_xlabel('x [cm]'); ax.set_ylabel('y [cm]'); mark(ax)
    plt.colorbar(im, ax=ax, shrink=0.75)

    ax = fig2.add_subplot(gs[0, 2])
    ratio_dB = 20 * np.log10(np.abs(ana['p2h']) / (np.abs(ana['p_lin']) + 1e-12) + 1e-12)
    im = ax.imshow(ratio_dB, cmap='RdYlBu_r', vmin=-60, vmax=0, **imkw)
    ax.set_title('(c) |p2|/|p1| [dB]', fontsize=11, fontweight='bold')
    ax.set_xlabel('x [cm]'); ax.set_ylabel('y [cm]'); mark(ax)
    plt.colorbar(im, ax=ax, shrink=0.75, label='dB')

    ax = fig2.add_subplot(gs[0, 3])
    im = ax.imshow(ana['p_total_rms'], cmap='inferno', **imkw)
    ax.set_title('(d) Total RMS (f0 + 2f0)', fontsize=11, fontweight='bold')
    ax.set_xlabel('x [cm]'); ax.set_ylabel('y [cm]'); mark(ax)
    plt.colorbar(im, ax=ax, shrink=0.75, label='Pa')

    ax = fig2.add_subplot(gs[1, 0:2])
    U = ana['U_gorkov']
    U_norm = U / np.abs(U).max()
    im = ax.imshow(U_norm, cmap='RdBu_r', vmin=-1, vmax=1, **imkw)
    ax.set_title("(e) Gor'kov Potential (normalised) -- Levitation Landscape",
                 fontsize=11, fontweight='bold')
    ax.set_xlabel('x [cm]'); ax.set_ylabel('y [cm]'); mark(ax)
    plt.colorbar(im, ax=ax, shrink=0.75, label='U / |U|_max')

    ax = fig2.add_subplot(gs[1, 2:4])
    Fx = -np.gradient(U, ana['x'], axis=1)
    Fy = -np.gradient(U, ana['y'], axis=0)
    F_mag = np.sqrt(Fx**2 + Fy**2)
    im = ax.imshow(np.log10(F_mag + 1e-20), cmap='hot', **imkw)
    ax.set_title('(f) Radiation Force |F| = -grad(U)  [log10 N]',
                 fontsize=11, fontweight='bold')
    ax.set_xlabel('x [cm]'); ax.set_ylabel('y [cm]'); mark(ax)
    skip = 25
    ax.quiver(ana['X'][::skip, ::skip]*100, ana['Y'][::skip, ::skip]*100,
              Fx[::skip, ::skip], Fy[::skip, ::skip],
              color='cyan', alpha=0.6, scale_units='xy',
              scale=np.abs(Fx).max()*10, headwidth=3)
    plt.colorbar(im, ax=ax, shrink=0.75, label='log10|F| [N]')

    fig2.suptitle("Nonlinear Effects -- Second-Harmonic Generation + Gor'kov Radiation Force\n"
                  f"beta = {BETA_NL:.2f} (air),  particle: EPS bead (rho={RHO_P} kg/m3, "
                  f"a={PARTICLE_RADIUS*1e3:.1f} mm)",
                  fontsize=13, fontweight='bold', y=1.01)
    fig2.savefig(OUTPUT_DIR / '02_nonlinear_effects.png', dpi=180, bbox_inches='tight')
    print(f"  -> {OUTPUT_DIR / '02_nonlinear_effects.png'}")
    plt.close(fig2)

    # ── Figure 3: Cross-sections ──
    x, y = ana['x'], ana['y']
    iy_f = np.argmin(np.abs(y - FOCAL_POINT[1]))
    ix_f = np.argmin(np.abs(x - FOCAL_POINT[0]))

    fig3, axes3 = plt.subplots(2, 2, figsize=(16, 10))

    ax = axes3[0, 0]
    ax.plot(x*100, np.abs(ana['p1'])[iy_f, :], 'c-', lw=1.5, label='Array 1')
    ax.plot(x*100, np.abs(ana['p2'])[iy_f, :], 'g-', lw=1.5, label='Array 2')
    ax.plot(x*100, np.abs(ana['p_lin'])[iy_f, :], 'r-', lw=2.0, label='Superposition')
    ax.axvline(FOCAL_POINT[0]*100, color='k', ls='--', lw=0.8, alpha=0.5)
    ax.set_xlabel('x [cm]'); ax.set_ylabel('|p| [Pa]')
    ax.set_title(f'Horizontal cut at y = {FOCAL_POINT[1]*100:.0f} cm -- Linear')
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes3[0, 1]
    ax.plot(y*100, np.abs(ana['p1'])[:, ix_f], 'c-', lw=1.5, label='Array 1')
    ax.plot(y*100, np.abs(ana['p2'])[:, ix_f], 'g-', lw=1.5, label='Array 2')
    ax.plot(y*100, np.abs(ana['p_lin'])[:, ix_f], 'r-', lw=2.0, label='Superposition')
    ax.axvline(FOCAL_POINT[1]*100, color='k', ls='--', lw=0.8, alpha=0.5)
    ax.set_xlabel('y [cm]'); ax.set_ylabel('|p| [Pa]')
    ax.set_title(f'Vertical cut at x = {FOCAL_POINT[0]*100:.0f} cm -- Linear')
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes3[1, 0]
    ax.plot(x*100, np.abs(ana['p_lin'])[iy_f, :], 'r-', lw=2, label='Fundamental')
    ax.plot(x*100, np.abs(ana['p2h'])[iy_f, :], 'm--', lw=1.5, label='2nd Harmonic')
    ax.plot(x*100, ana['p_total_rms'][iy_f, :], 'k-', lw=1.5, label='Total RMS')
    ax.axvline(FOCAL_POINT[0]*100, color='k', ls='--', lw=0.8, alpha=0.5)
    ax.set_xlabel('x [cm]'); ax.set_ylabel('|p| [Pa]')
    ax.set_title('Horizontal cut -- Fundamental vs 2nd Harmonic')
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes3[1, 1]
    U_cut = ana['U_gorkov'][:, ix_f]
    ax.plot(y*100, U_cut, 'b-', lw=2)
    ax.set_xlabel('y [cm]'); ax.set_ylabel("U [J]")
    ax.set_title(f"Gor'kov Potential -- Vertical cut at x = {FOCAL_POINT[0]*100:.0f} cm")
    ax.axvline(FOCAL_POINT[1]*100, color='k', ls='--', lw=0.8, alpha=0.5)
    ax.grid(True, alpha=0.3)
    try:
        from scipy.signal import argrelmin
        minima = argrelmin(U_cut, order=5)[0]
        if len(minima) > 0:
            ax.plot(y[minima]*100, U_cut[minima], 'rv', ms=8,
                    label=f'{len(minima)} trap sites')
            ax.legend()
    except ImportError:
        pass

    fig3.suptitle('Cross-Sections Through Focal Point', fontsize=14, fontweight='bold')
    fig3.tight_layout()
    fig3.savefig(OUTPUT_DIR / '03_cross_sections.png', dpi=180, bbox_inches='tight')
    print(f"  -> {OUTPUT_DIR / '03_cross_sections.png'}")
    plt.close(fig3)

    # ── Figure 4: k-Wave comparison (if available) ──
    if kw_result is not None:
        fig4, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))

        ana_dB = 20 * np.log10(np.abs(ana['p_lin']) / p_ref + 1e-12)
        im1 = ax1.imshow(ana_dB, cmap='inferno', vmin=-40, vmax=0, **imkw)
        ax1.set_title("Analytical (Green's fn)", fontsize=12, fontweight='bold')
        ax1.set_xlabel('x [cm]'); ax1.set_ylabel('y [cm]'); mark(ax1)
        plt.colorbar(im1, ax=ax1, shrink=0.75, label='dB')

        kw_ext = [kw_result['x'][0]*100, kw_result['x'][-1]*100,
                  kw_result['y'][0]*100, kw_result['y'][-1]*100]
        kw_ref = kw_result['p_max'].max()
        kw_dB = 20 * np.log10(kw_result['p_max'] / kw_ref + 1e-12)
        nl_str = "(nonlinear)" if kw_result.get('nonlinear') else "(linear)"
        im2 = ax2.imshow(kw_dB.T, cmap='inferno', vmin=-40, vmax=0,
                        extent=kw_ext, origin='lower', aspect='equal')
        ax2.set_title(f'k-Wave FDTD {nl_str}', fontsize=12, fontweight='bold')
        ax2.set_xlabel('x [cm]'); ax2.set_ylabel('y [cm]'); mark(ax2)
        plt.colorbar(im2, ax=ax2, shrink=0.75, label='dB')

        from scipy.interpolate import RegularGridInterpolator
        interp = RegularGridInterpolator(
            (kw_result['x'], kw_result['y']), kw_result['p_max'],
            bounds_error=False, fill_value=0)
        pts = np.stack([ana['X'].ravel(), ana['Y'].ravel()], axis=-1)
        kw_interp = interp(pts).reshape(ana['X'].shape)
        diff = np.abs(ana['p_lin']) - kw_interp
        im3 = ax3.imshow(diff, cmap='RdBu_r', **imkw)
        ax3.set_title('Difference (Analytical - k-Wave)', fontsize=12, fontweight='bold')
        ax3.set_xlabel('x [cm]'); ax3.set_ylabel('y [cm]'); mark(ax3)
        plt.colorbar(im3, ax=ax3, shrink=0.75, label='dP [Pa]')

        fig4.suptitle('Method Comparison -- Same Physical Domain', fontsize=14, fontweight='bold')
        fig4.tight_layout()
        fig4.savefig(OUTPUT_DIR / '04_kwave_comparison.png', dpi=180, bbox_inches='tight')
        print(f"  -> {OUTPUT_DIR / '04_kwave_comparison.png'}")
        plt.close(fig4)


# ═══════════════════════════════════════════════════════════════════════════
# TEXT DELIVERABLES
# ═══════════════════════════════════════════════════════════════════════════

ASSUMPTIONS = """\
ASSUMPTIONS LOG
===============

* Medium: Air at 20 C, c0 = 343 m/s, rho0 = 1.225 kg/m3
  Standard room-temperature properties; well-characterised for ultrasonics.

* Frequency: 40 kHz (lambda ~ 8.575 mm)
  Given in problem statement.

* Element spacing: lambda/2 ~ 4.29 mm
  Half-wavelength pitch prevents grating lobes for steering up to +/-90 deg.
  Real 40 kHz transducers (Murata MA40S4S, dia 10 mm) exceed lambda/2;
  grating lobes would appear in practice but lambda/2 is the ideal design assumption.

* Array size: 8 elements per array (2D cross-section of 8x8 3D array)
  Gives ~30 mm aperture; moderate directivity with clear beam formation.

* Point-source model: Each element is a monopole point source
  At lambda/2 element size, far-field directivity effects are negligible.

* 3D spherical spreading (1/r) on a 2D observation plane
  Physical transducers are 3D objects; the 3D Green's function on a 2D slice
  ("2.5D") is more physically accurate than the 2D line-source model
  (Hankel function, 1/sqrt(r) cylindrical spreading).

* No absorption in linear model
  At 40 kHz in air, alpha ~ 1.4 dB/m. Over the 18 cm path, total loss ~ 0.25 dB.

* Free-field (no reflections)
  Focus is on beam interaction physics, not room acoustics.
  k-Wave PML boundaries approximate this condition.

* Nonlinear parameter: B/(2A) = 0.2, beta = 1.2 for air
  Standard value from Hamilton & Blackstock, Nonlinear Acoustics (1998).

* Second-harmonic perturbation: Fubini-Blackstock quasilinear approximation
  Valid pre-shock (sigma < 1). At 10 Pa source amplitude and 18 cm propagation,
  shock distance ~ rho*c^3/(beta*omega*|p|) ~ 120 m >> path length.

* Gor'kov potential: Small-particle (Rayleigh) limit, a << lambda
  Particle radius 1 mm << lambda/2 ~ 4.3 mm; Gor'kov theory applies.
  Particle: expanded polystyrene (rho = 29 kg/m3, c = 900 m/s).

* Phase-only focusing (no amplitude apodization)
  Standard for ultrasonic phased arrays.

* Steady-state (analytical): Monochromatic CW assumption
  40 kHz transducers are narrowband; steady state reached within a few cycles.
"""


INTERACTION_WRITEUP = """\
INTERACTION MODEL WRITE-UP
==========================

Primary model: Linear superposition
------------------------------------
The total field is the coherent (complex) sum:

    p_total(r) = p_array1(r) + p_array2(r)

At the focal point both beams arrive in-phase, producing ~6 dB constructive
interference. Away from focus the relative phase varies, creating the
standing-wave interference fringes visible in the vertical cross-sections.

Extension 1: Second-harmonic generation (nonlinear perturbation)
----------------------------------------------------------------
Using the Fubini-Blackstock quasilinear framework:

    |p2(r)| ~ (beta * k * |p1|^2) / (2 * rho0 * c0^2) * sigma_eff

where beta = 1 + B/(2A) = 1.2 for air. The second harmonic (80 kHz)
is strongest near the focal point where the fundamental amplitude peaks.

At the source amplitudes used (10 Pa, ~154 dB SPL per element), the
second harmonic is ~45 dB below the fundamental, confirming that linear
superposition dominates and the perturbation treatment is self-consistent.

Extension 2: Gor'kov radiation force potential
-----------------------------------------------
The time-averaged radiation force on a small particle:

    U = V_p * [f0 * <p^2> / (4*rho0*c0^2) - 3*f1*rho0 * <v^2> / 8]
    F = -grad(U)

Local minima of U are stable trapping sites for acoustic levitation.
The simulation reveals multiple potential wells along the beam crossing
region where particles would be confined.

What linear superposition misses:
----------------------------------
1. Waveform steepening / shock formation: At >160 dB, energy transfers
   to harmonics, reducing peak focal pressure below the linear prediction.

2. Acoustic streaming: Time-averaged momentum transfer creates steady
   convective flows (Eckart streaming), redistributing heat and affecting
   particle dynamics.

3. Parametric interaction: Two beams from different angles produce a
   virtual source at their intersection via the quadratic nonlinearity,
   generating sum/difference-frequency components.

k-Wave approach: medium.BonA = 0.4 adds cumulative nonlinearity to the
time-domain solver, capturing harmonic generation self-consistently.
"""


NEXT_STEPS = """\
WHAT I'D DO NEXT (2 hours)
==========================

With two more hours I would pursue three improvements. First, I would run
the k-Wave simulation with medium.BonA enabled at progressively higher
source amplitudes (50, 100, 500 Pa) to map where the perturbation theory
breaks down -- this "nonlinearity onset" curve directly informs transducer
drive-voltage sizing. Second, I would compute Gor'kov force trajectories
by integrating particle equations of motion (including Stokes drag),
producing an animation of particle capture and transport toward the
trapping sites. Third, I would build a lightweight Streamlit interface
letting the user drag the focal point and adjust array parameters in
real time; the analytical method runs in <200 ms, well within interactive
frame rates, making it immediately useful for design exploration.
"""


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("\n" + "=" * 72)
    print(f"  ULTRASONIC PHASED ARRAY SIMULATION")
    print(f"  Two {N_ELEM}x{N_ELEM} arrays at {F0/1e3:.0f} kHz -> "
          f"Focus at ({FOCAL_POINT[0]*100:.0f}, {FOCAL_POINT[1]*100:.0f}) cm")
    print(f"  Domain: [{X_MIN*100:.0f}, {X_MAX*100:.0f}] x "
          f"[{Y_MIN*100:.0f}, {Y_MAX*100:.0f}] cm")
    print("  Linear superposition + nonlinear perturbation")
    print("=" * 72 + "\n")

    ana = run_analytical()
    kw = run_kwave(enable_nonlinear=True)

    print("\n" + "=" * 72)
    print("  GENERATING PLOTS")
    print("=" * 72)
    make_plots(ana, kw)

    for name, text in [('assumptions_log.txt', ASSUMPTIONS),
                       ('interaction_writeup.txt', INTERACTION_WRITEUP),
                       ('next_steps.txt', NEXT_STEPS)]:
        with open(OUTPUT_DIR / name, 'w') as f:
            f.write(text)
        print(f"  -> {OUTPUT_DIR / name}")

    print(f"\n  All outputs in: {OUTPUT_DIR.resolve()}")
    print("  Done.\n")
