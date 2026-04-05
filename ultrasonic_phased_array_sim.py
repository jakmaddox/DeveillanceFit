#!/usr/bin/env python3
"""
Ultrasonic Phased Array Simulation — Deveillance Applied Physics Trial
======================================================================
Simulates the 2D pressure field from two N×N transducer arrays at 40 kHz,
beamsteered toward a common focal point, using:
  1. Analytical Green's function (Huygens–Fresnel principle)
  2. k-Wave pseudospectral FDTD (time-domain full-wave solver)

Both approaches compute the superposition of the two beams at and around
the focal point. The analytical method gives the steady-state monochromatic
field; k-Wave time-steps to that same steady state while capturing transient
effects.

Author: [Your name]
Date:   2025
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
import time
import sys
import os

# ═══════════════════════════════════════════════════════════════════════════
# PHYSICAL CONSTANTS & SIMULATION PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════

# Medium: air at 20 °C
C0        = 343.0          # speed of sound [m/s]
RHO0      = 1.225          # density [kg/m³]
F0        = 40_000.0       # operating frequency [Hz]
WAVELENGTH = C0 / F0       # ≈ 8.575 mm
K          = 2 * np.pi / WAVELENGTH  # wavenumber [rad/m]

# Array parameters
N_ELEM     = 8             # N×N array → in 2D cross-section this is N elements
PITCH      = WAVELENGTH / 2  # λ/2 element spacing ≈ 4.29 mm (prevents grating lobes)
AMPLITUDE  = 1.0           # normalised source amplitude [Pa]

# Array positioning (along y-axis, separated in x)
ARRAY1_CENTER = np.array([0.0, -0.10])    # 10 cm below focal plane
ARRAY2_CENTER = np.array([0.0,  0.10])    # 10 cm above focal plane
FOCAL_POINT   = np.array([0.15, 0.0])     # 15 cm to the right, on centerline

# Observation domain
X_RANGE = (-0.05, 0.30)    # [m]
Y_RANGE = (-0.20, 0.20)    # [m]
RES     = 400              # grid resolution (pixels per axis)

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════
# HELPER: BUILD ARRAY ELEMENT POSITIONS
# ═══════════════════════════════════════════════════════════════════════════

def make_array(center: np.ndarray, n: int, pitch: float) -> np.ndarray:
    """
    Create an N-element linear array centered at `center`, spread along x-axis.
    Returns shape (n, 2) array of [x, y] positions.
    """
    offsets = (np.arange(n) - (n - 1) / 2) * pitch
    pos = np.zeros((n, 2))
    pos[:, 0] = center[0] + offsets   # spread elements along x
    pos[:, 1] = center[1]             # all at same y
    return pos


# ═══════════════════════════════════════════════════════════════════════════
# METHOD 1: ANALYTICAL GREEN'S FUNCTION (HUYGENS–FRESNEL)
# ═══════════════════════════════════════════════════════════════════════════

def compute_focusing_phases(sources: np.ndarray, focal: np.ndarray, k: float) -> np.ndarray:
    """
    Compute per-element focusing phases so all contributions arrive in-phase
    at the focal point.

    Phase for element n:  φ_n = k · |r_focal − r_n|

    When we apply exp(+j φ_n) to each source and propagate with
    exp(−jk|r − r_n|), the total phase at r=focal becomes zero for every
    element → constructive interference.
    """
    d = np.linalg.norm(sources - focal, axis=1)
    return k * d


def analytical_pressure_field(sources: np.ndarray, focal: np.ndarray,
                               X: np.ndarray, Y: np.ndarray,
                               k: float, amplitude: float = 1.0) -> np.ndarray:
    """
    Steady-state complex pressure from a focused point-source array.

    Uses the 3D free-space Green's function (spherical spreading, 1/r)
    evaluated on a 2D observation plane:

        p(r) = A · Σ_n  exp(j·k·|r_f − r_n|) · exp(−j·k·|r − r_n|) / |r − r_n|

    Parameters
    ----------
    sources : (N, 2) element positions
    focal   : (2,)   focal point
    X, Y    : 2D meshgrid arrays
    k       : wavenumber
    amplitude : source amplitude

    Returns
    -------
    p : complex pressure field, same shape as X
    """
    Ns = len(sources)
    sx = sources[:, 0].reshape(Ns, 1, 1)
    sy = sources[:, 1].reshape(Ns, 1, 1)

    # Distance from each source to every grid point
    r = np.sqrt((X[np.newaxis] - sx)**2 + (Y[np.newaxis] - sy)**2)
    r = np.maximum(r, 1e-10)  # regularise to avoid division by zero

    # Focusing phase: compensate path-length differences to focal point
    phi = compute_focusing_phases(sources, focal, k).reshape(Ns, 1, 1)

    # Coherent summation (Green's function with focusing weights)
    p = amplitude * np.sum(np.exp(1j * phi) * np.exp(-1j * k * r) / r, axis=0)
    return p


def run_analytical_simulation():
    """Run the full analytical simulation and produce plots."""
    print("=" * 70)
    print("METHOD 1: ANALYTICAL GREEN'S FUNCTION")
    print("=" * 70)

    # Build arrays
    array1 = make_array(ARRAY1_CENTER, N_ELEM, PITCH)
    array2 = make_array(ARRAY2_CENTER, N_ELEM, PITCH)

    print(f"  Array 1: {N_ELEM} elements at y = {ARRAY1_CENTER[1]*100:.1f} cm")
    print(f"  Array 2: {N_ELEM} elements at y = {ARRAY2_CENTER[1]*100:.1f} cm")
    print(f"  Focal point: ({FOCAL_POINT[0]*100:.1f}, {FOCAL_POINT[1]*100:.1f}) cm")
    print(f"  Frequency: {F0/1e3:.0f} kHz, λ = {WAVELENGTH*1e3:.2f} mm")
    print(f"  Element pitch: {PITCH*1e3:.2f} mm (λ/2)")

    # Print phase delays
    print("\n  Phase delays (Array 1):")
    phases1 = compute_focusing_phases(array1, FOCAL_POINT, K)
    for i, (pos, phi) in enumerate(zip(array1, phases1)):
        print(f"    Element {i}: pos=({pos[0]*1e3:+6.1f}, {pos[1]*1e3:+6.1f}) mm, "
              f"phase = {np.degrees(phi) % 360:6.1f}°")

    print("\n  Phase delays (Array 2):")
    phases2 = compute_focusing_phases(array2, FOCAL_POINT, K)
    for i, (pos, phi) in enumerate(zip(array2, phases2)):
        print(f"    Element {i}: pos=({pos[0]*1e3:+6.1f}, {pos[1]*1e3:+6.1f}) mm, "
              f"phase = {np.degrees(phi) % 360:6.1f}°")

    # Build observation grid
    x = np.linspace(*X_RANGE, RES)
    y = np.linspace(*Y_RANGE, RES)
    X, Y = np.meshgrid(x, y)

    # Compute individual and combined fields
    t0 = time.perf_counter()
    p1 = analytical_pressure_field(array1, FOCAL_POINT, X, Y, K, AMPLITUDE)
    p2 = analytical_pressure_field(array2, FOCAL_POINT, X, Y, K, AMPLITUDE)
    p_total = p1 + p2  # LINEAR SUPERPOSITION
    dt = time.perf_counter() - t0
    print(f"\n  Computation time: {dt:.3f} s")

    # Convert to SPL (dB re max)
    abs_p1 = np.abs(p1)
    abs_p2 = np.abs(p2)
    abs_total = np.abs(p_total)

    # Reference: peak of combined field
    p_ref = abs_total.max()
    spl1 = 20 * np.log10(abs_p1 / p_ref + 1e-12)
    spl2 = 20 * np.log10(abs_p2 / p_ref + 1e-12)
    spl_total = 20 * np.log10(abs_total / p_ref + 1e-12)

    # ── PLOTTING ──
    ext = [X_RANGE[0]*100, X_RANGE[1]*100, Y_RANGE[0]*100, Y_RANGE[1]*100]
    imkw = dict(extent=ext, origin='lower', aspect='equal', cmap='inferno')

    fig = plt.figure(figsize=(20, 14))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.30, wspace=0.30)

    # -- Row 1: Individual arrays + combined (pressure magnitude) --
    titles_row1 = ['Array 1 — |p|', 'Array 2 — |p|', 'Superposition — |p|']
    data_row1   = [abs_p1, abs_p2, abs_total]
    for col, (title, data) in enumerate(zip(titles_row1, data_row1)):
        ax = fig.add_subplot(gs[0, col])
        im = ax.imshow(data, **imkw)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_xlabel('x [cm]')
        ax.set_ylabel('y [cm]')
        # Mark arrays and focal point
        ax.plot(array1[:, 0]*100, array1[:, 1]*100, 'c^', ms=6, label='Array 1')
        ax.plot(array2[:, 0]*100, array2[:, 1]*100, 'gv', ms=6, label='Array 2')
        ax.plot(FOCAL_POINT[0]*100, FOCAL_POINT[1]*100, 'w*', ms=14,
                markeredgecolor='k', markeredgewidth=0.5, label='Focus')
        plt.colorbar(im, ax=ax, shrink=0.8, label='Pressure [arb.]')
        if col == 0:
            ax.legend(loc='upper left', fontsize=8)

    # -- Row 2: SPL (dB) versions --
    titles_row2 = ['Array 1 — SPL (dB re peak)', 'Array 2 — SPL (dB re peak)',
                   'Superposition — SPL (dB re peak)']
    data_row2   = [spl1, spl2, spl_total]
    for col, (title, data) in enumerate(zip(titles_row2, data_row2)):
        ax = fig.add_subplot(gs[1, col])
        im = ax.imshow(data, vmin=-40, vmax=0, **imkw)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_xlabel('x [cm]')
        ax.set_ylabel('y [cm]')
        ax.plot(array1[:, 0]*100, array1[:, 1]*100, 'c^', ms=6)
        ax.plot(array2[:, 0]*100, array2[:, 1]*100, 'gv', ms=6)
        ax.plot(FOCAL_POINT[0]*100, FOCAL_POINT[1]*100, 'w*', ms=14,
                markeredgecolor='k', markeredgewidth=0.5)
        plt.colorbar(im, ax=ax, shrink=0.8, label='dB')

    fig.suptitle('Analytical Green\'s Function — Two 8-Element Arrays at 40 kHz\n'
                 'Beamsteered to Common Focus via Linear Superposition',
                 fontsize=15, fontweight='bold', y=0.98)

    fig.savefig(OUTPUT_DIR / 'analytical_pressure_field.png', dpi=200,
                bbox_inches='tight')
    print(f"  Saved: {OUTPUT_DIR / 'analytical_pressure_field.png'}")
    plt.close(fig)

    # ── CROSS-SECTION PLOTS (through focal point) ──
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Horizontal cut (y = 0)
    iy_focal = np.argmin(np.abs(y - FOCAL_POINT[1]))
    ax1.plot(x*100, abs_p1[iy_focal, :], 'c-', lw=1.5, label='Array 1')
    ax1.plot(x*100, abs_p2[iy_focal, :], 'g-', lw=1.5, label='Array 2')
    ax1.plot(x*100, abs_total[iy_focal, :], 'r-', lw=2, label='Superposition')
    ax1.axvline(FOCAL_POINT[0]*100, color='k', ls='--', lw=1, alpha=0.5)
    ax1.set_xlabel('x [cm]')
    ax1.set_ylabel('|p| [arb.]')
    ax1.set_title(f'Horizontal cut at y = {FOCAL_POINT[1]*100:.1f} cm')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Vertical cut (x = focal_x)
    ix_focal = np.argmin(np.abs(x - FOCAL_POINT[0]))
    ax2.plot(y*100, abs_p1[:, ix_focal], 'c-', lw=1.5, label='Array 1')
    ax2.plot(y*100, abs_p2[:, ix_focal], 'g-', lw=1.5, label='Array 2')
    ax2.plot(y*100, abs_total[:, ix_focal], 'r-', lw=2, label='Superposition')
    ax2.axvline(FOCAL_POINT[1]*100, color='k', ls='--', lw=1, alpha=0.5)
    ax2.set_xlabel('y [cm]')
    ax2.set_ylabel('|p| [arb.]')
    ax2.set_title(f'Vertical cut at x = {FOCAL_POINT[0]*100:.1f} cm')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig2.suptitle('Cross-Sections Through Focal Point', fontsize=14, fontweight='bold')
    fig2.savefig(OUTPUT_DIR / 'analytical_cross_sections.png', dpi=200,
                 bbox_inches='tight')
    print(f"  Saved: {OUTPUT_DIR / 'analytical_cross_sections.png'}")
    plt.close(fig2)

    return p1, p2, p_total, X, Y


# ═══════════════════════════════════════════════════════════════════════════
# METHOD 2: k-WAVE (PSEUDOSPECTRAL FDTD TIME-DOMAIN)
# ═══════════════════════════════════════════════════════════════════════════

def run_kwave_simulation():
    """
    Run a k-Wave 2D simulation with two phased arrays.

    NOTE: k-Wave Python (pip install k-wave-python) requires:
      - Python ≥ 3.10
      - On macOS: brew install fftw hdf5 zlib libomp
      - The C++ backend auto-downloads binaries on first run
      - Alternatively, the pure-Python backend works without binaries

    The correct import paths (as of k-wave-python ≥ 0.4.0) are:
      from kwave.kgrid   import kWaveGrid
      from kwave.kmedium import kWaveMedium
      from kwave.ksource import kSource
      from kwave.ksensor import kSensor
    """
    print("\n" + "=" * 70)
    print("METHOD 2: k-WAVE PSEUDOSPECTRAL FDTD")
    print("=" * 70)

    try:
        from kwave.kgrid import kWaveGrid
        from kwave.kmedium import kWaveMedium
        from kwave.ksource import kSource
        from kwave.ksensor import kSensor
        from kwave.utils.signals import tone_burst
    except ImportError as e:
        print(f"\n  ⚠ k-Wave import failed: {e}")
        print("  Install with: pip install k-wave-python")
        print("  On macOS also: brew install fftw hdf5 zlib libomp")
        print("  Skipping k-Wave simulation.\n")
        return None

    # Try the unified API first (v0.6.0+), fall back to legacy
    try:
        from kwave.kspaceFirstOrder import kspaceFirstOrder
        use_unified = True
        print("  Using unified kspaceFirstOrder API (v0.6.0+)")
    except ImportError:
        try:
            from kwave.kspaceFirstOrder2D import kspaceFirstOrder2D
            from kwave.options.simulation_options import SimulationOptions
            from kwave.options.simulation_execution_options import SimulationExecutionOptions
            use_unified = False
            print("  Using legacy kspaceFirstOrder2D API")
        except ImportError as e:
            print(f"\n  ⚠ Cannot import simulation function: {e}")
            return None

    # ── Grid setup ──
    # At 40 kHz in air, λ ≈ 8.575 mm. Need ≥ 2 PPW → dx ≤ 4.3 mm.
    # Using dx = 1 mm for ~8.6 PPW (good accuracy).
    dx = 1.0e-3   # 2 mm
    dy = 1.0e-3

    # Domain: cover both arrays and focal region
    # x: from −5 cm to +25 cm → 150 grid points
    # y: from −22 cm to +22 cm → 220 grid points
    Nx = 150
    Ny = 220

    print(f"  Grid: {Nx} × {Ny}, dx = {dx*1e3:.1f} mm")
    print(f"  Domain: {Nx*dx*100:.0f} × {Ny*dy*100:.0f} cm")
    print(f"  Points per wavelength: {WAVELENGTH/dx:.1f}")

    # Create grid
    kgrid = kWaveGrid([Nx, Ny], [dx, dy])

    # Medium
    medium = kWaveMedium(sound_speed=C0, density=RHO0)
    medium.BonA = 0.4

    # Time array — let k-Wave auto-calculate or set manually
    kgrid.makeTime(medium.sound_speed)
    print(f"  Time steps: {kgrid.Nt}, dt = {kgrid.dt*1e6:.3f} µs")
    print(f"  Total time: {kgrid.Nt * kgrid.dt * 1e4:.2f} ms")

    # ── Source setup ──
    # Grid origin is at center of domain. Convert physical coords to grid indices.
    # kWaveGrid coordinate: grid center = (0, 0), indexed as (ix, iy)
    # Physical position → grid index:
    #   ix = Nx//2 + round(x_phys / dx)
    #   iy = Ny//2 + round(y_phys / dy)

    def phys_to_grid(pos, Nx, Ny, dx, dy):
        ix = Nx // 2 + int(round(pos[0] / dx))
        iy = Ny // 2 + int(round(pos[1] / dy))
        return np.clip(ix, 0, Nx-1), np.clip(iy, 0, Ny-1)

    array1 = make_array(ARRAY1_CENTER, N_ELEM, PITCH)
    array2 = make_array(ARRAY2_CENTER, N_ELEM, PITCH)
    all_sources = np.vstack([array1, array2])
    n_total = len(all_sources)

    # Build source mask
    source = kSource()
    source_mask = np.zeros((Nx, Ny), dtype=np.int64)
    grid_positions = []
    for pos in all_sources:
        ix, iy = phys_to_grid(pos, Nx, Ny, dx, dy)
        source_mask[ix, iy] = 1
        grid_positions.append((ix, iy))

    source.p_mask = source_mask

    # Compute focusing delays for ALL elements
    distances = np.linalg.norm(all_sources - FOCAL_POINT, axis=1)
    delays_sec = distances / C0
    delays_sec = delays_sec.max() - delays_sec  # invert: farthest fires first

    # Build per-element signals with phase offsets
    num_cycles = 5
    signals = []

    # Source points are ordered by column-major (Fortran) linear indexing of the mask.
    # We need to map our element list to that ordering.
    mask_indices = np.argwhere(source_mask.T.ravel() > 0)  # Fortran-order indices
    # Actually, k-Wave uses the order of nonzero elements in the mask
    # scanned in column-major order (y changes fastest for Fortran, but
    # numpy default is row-major). We need to match the ordering k-Wave expects.
    # For safety, let's build source.p indexed by the mask's nonzero order.

    # Get mask indices in column-major order (as k-Wave expects)
    fortran_flat = source_mask.ravel(order='F')
    source_linear_idx = np.where(fortran_flat > 0)[0]

    # Map each source element to its position in the fortran-order list
    element_to_fortran = {}
    for i, pos in enumerate(all_sources):
        ix, iy = grid_positions[i]
        fortran_idx = iy * Nx + ix  # column-major: col * nrows + row
        element_to_fortran[i] = fortran_idx

    # Sort elements by their Fortran-order index
    sorted_elements = sorted(range(n_total),
                             key=lambda i: element_to_fortran[i])

    for elem_idx in sorted_elements:
        offset_samples = int(round(delays_sec[elem_idx] / kgrid.dt))
        try:
            sig = tone_burst(1 / kgrid.dt, F0, num_cycles,
                             signal_offset=offset_samples,
                             signal_length=kgrid.Nt)
        except TypeError:
            # Older API might not have signal_offset; apply manually
            sig = tone_burst(1 / kgrid.dt, F0, num_cycles,
                             signal_length=kgrid.Nt)
            sig = np.roll(sig, offset_samples)

        signals.append(AMPLITUDE * sig.flatten())

    source.p = np.stack(signals, axis=0)  # (n_sources, Nt)

    # ── Sensor: record the max pressure everywhere ──
    sensor = kSensor()
    sensor.mask = np.ones((Nx, Ny), dtype=np.int64)
    sensor.record = ['p_max']

    # ── Run simulation ──
    print("\n  Running k-Wave simulation (this may take a few minutes)...")
    t0 = time.perf_counter()

    try:
        if use_unified:
            result = kspaceFirstOrder(kgrid, medium, source, sensor,
                                       pml_inside=False, quiet=True)
        else:
            sim_options = SimulationOptions(
                save_to_disk=True,
                pml_inside=False,
            )
            exec_options = SimulationExecutionOptions()
            result = kspaceFirstOrder2D(
                kgrid=kgrid, source=source, sensor=sensor,
                medium=medium,
                simulation_options=sim_options,
                execution_options=exec_options,
            )

        dt = time.perf_counter() - t0
        print(f"  k-Wave simulation completed in {dt:.1f} s")

        # Extract max pressure field
        if isinstance(result, dict):
            p_max = result.get('p_max', result.get('p_final', None))
        else:
            p_max = result

        if p_max is not None:
            if p_max.ndim == 1:
                p_max = p_max.reshape((Nx, Ny))

            # Plot k-Wave result
            fig, ax = plt.subplots(figsize=(10, 7))
            ext = [0, Nx*dx*100, 0, Ny*dy*100]

            # Shift to physical coordinates
            x_phys = (np.arange(Nx) - Nx//2) * dx * 100
            y_phys = (np.arange(Ny) - Ny//2) * dy * 100
            ext = [x_phys[0], x_phys[-1], y_phys[0], y_phys[-1]]

            im = ax.imshow(p_max.T, origin='lower', extent=ext,
                          cmap='inferno', aspect='equal')
            ax.set_xlabel('x [cm]')
            ax.set_ylabel('y [cm]')
            ax.set_title('k-Wave FDTD — Max Pressure (Combined Arrays)',
                        fontsize=14, fontweight='bold')
            ax.plot(array1[:, 0]*100, array1[:, 1]*100, 'c^', ms=6, label='Array 1')
            ax.plot(array2[:, 0]*100, array2[:, 1]*100, 'gv', ms=6, label='Array 2')
            ax.plot(FOCAL_POINT[0]*100, FOCAL_POINT[1]*100, 'w*', ms=14,
                    markeredgecolor='k', markeredgewidth=0.5, label='Focus')
            ax.legend(loc='upper left')
            plt.colorbar(im, ax=ax, label='Max pressure [Pa]')

            fig.savefig(OUTPUT_DIR / 'kwave_pressure_field.png', dpi=200,
                        bbox_inches='tight')
            print(f"  Saved: {OUTPUT_DIR / 'kwave_pressure_field.png'}")
            plt.close(fig)
            return p_max
        else:
            print("  ⚠ Could not extract pressure field from k-Wave result.")
            return None

    except Exception as e:
        dt = time.perf_counter() - t0
        print(f"\n  ⚠ k-Wave simulation failed after {dt:.1f} s: {e}")
        print(f"  Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        print("\n  The analytical results above are still valid.")
        return None


# ═══════════════════════════════════════════════════════════════════════════
# ASSUMPTIONS LOG
# ═══════════════════════════════════════════════════════════════════════════

ASSUMPTIONS = """
╔══════════════════════════════════════════════════════════════════════║
║                          ASSUMPTIONS LOG                             ║
╠══════════════════════════════════════════════════════════════════════║
║                                                                      ║
║  • Medium: Air at 20 °C, c = 343 m/s, ρ = 1.225 kg/m³                ║
║    Justification: Standard room-temperature air properties.          ║
║                                                                      ║
║  • Frequency: 40 kHz (λ ≈ 8.575 mm)                                  ║
║    Justification: Given in problem statement.                        ║
║                                                                      ║
║  • Element spacing: λ/2 ≈ 4.29 mm                                    ║
║    Justification: Half-wavelength pitch is the standard choice to    ║
║    prevent grating lobes for steering up to ±90°. Real 40 kHz        ║
║    transducers (e.g., Murata MA40S4S, ∅10 mm) are larger, which      ║
║    would produce grating lobes; λ/2 is the ideal-case assumption.    ║
║                                                                      ║
║  • Array size: 8 elements (2D cross-section of 8×8 3D array)         ║
║    Justification: Moderate array that produces a well-defined beam   ║
║    with ~30 mm aperture. 2D simulation uses the cross-sectional row. ║
║                                                                      ║
║  • Point-source model: Each element is a monopole point source       ║
║    Justification: Element diameter (λ/2 ≈ 4.3 mm) is small relative  ║
║    to wavelength; far-field directivity effects are negligible.      ║
║                                                                      ║
║  • Propagation model: 3D spherical spreading (1/r) on 2D slice       ║
║    Justification: Physical transducers are finite objects; 3D Green's║
║    function evaluated on a 2D plane ("2.5D") is more accurate than   ║
║    the 2D line-source model (1/√r, Hankel function).                 ║
║                                                                      ║
║  • No absorption: Attenuation in air ignored                         ║
║    Justification: At 40 kHz, α ≈ 1.4 dB/m in air. Over 15 cm path,   ║
║    loss is ~0.2 dB — negligible for this demonstration.              ║
║                                                                      ║
║  • No reflections / free-field: Infinite homogeneous medium          ║
║    Justification: Focus is on beam interaction, not room acoustics.  ║
║    PML in k-Wave approximates this condition.                        ║
║                                                                      ║
║  • Steady-state (analytical): Monochromatic CW assumption            ║
║    Justification: Transducers are narrowband; steady state is reached║
║    within a few cycles. k-Wave provides transient validation.        ║
║                                                                      ║
║  • Linear superposition: p_total = p_array1 + p_array2               ║
║    Justification: At typical ultrasonic levitation SPLs (~150 dB),   ║
║    nonlinear effects begin to matter but linear superposition is a   ║
║    reasonable first-order model. See interaction write-up below.     ║
║                                                                      ║
║  • Focusing: Geometric phase delays (no amplitude apodization)       ║
║    Justification: Phase-only steering is standard for ultrasonic     ║
║    phased arrays. Apodization would reduce sidelobes but also peak   ║
║    pressure.                                                         ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════║
"""


# ═════════════════════════════════════════════════════════════════════║
# INTERACTION MODEL WRITE-UP║
# ═════════════════════════════════════════════════════════════════════║

INTERACTION_WRITEUP = """
╔══════════════════════════════════════════════════════════════════════║
║                   INTERACTION MODEL WRITE-UP                         ║
╠══════════════════════════════════════════════════════════════════════║
║                                                                      ║
║  Model: Linear Superposition                                         ║
║  ─────────────────────────────                                       ║
║  The total field is computed as the coherent (complex) sum of the    ║
║  two individual array fields:                                        ║
║                                                                      ║
║      p_total(r) = p_array1(r) + p_array2(r)                          ║
║                                                                      ║
║  This preserves phase information: at the focal point, both beams    ║
║  arrive in-phase → constructive interference → ~2× the individual    ║
║  amplitude (6 dB gain). Away from focus, the relative phase varies   ║
║  and the beams partially cancel.                                     ║
║                                                                      ║
║  Physical effect missed: Nonlinear acoustic interaction              ║
║  ────────────────────────────────────────────────────────            ║
║  At high SPLs (>140 dB, common in ultrasonic levitation), the        ║
║  linear wave equation breaks down and the Westervelt equation        ║
║  applies. The key nonlinear effects are:                             ║
║                                                                      ║
║  1. Harmonic generation: The fundamental (40 kHz) distorts into a    ║
║     sawtooth, transferring energy to 80 kHz, 120 kHz, etc. At the    ║
║     focal point where amplitude is highest, this steepening is       ║
║     strongest, effectively broadening the frequency content.         ║
║                                                                      ║
║  2. Acoustic radiation force (Gor'kov potential): The time-averaged  ║
║     nonlinear pressure creates a steady-state force field. In the    ║
║     superposition of two beams, standing-wave nodes between them     ║
║     produce trapping sites (used in acoustic levitation). Linear     ║
║     superposition captures the interference pattern but not the      ║
║     actual radiation force magnitude.                                ║
║                                                                      ║
║  3. Parametric interaction (difference frequency): When two beams    ║
║     at slightly different frequencies intersect, nonlinearity        ║
║     generates a difference-frequency component (demodulation).       ║
║     Even at the same frequency, the interaction of beams from        ║
║     different angles produces a virtual source at the intersection   ║
║     with modified directivity. This is the basis of parametric       ║
║     loudspeakers.                                                    ║
║                                                                      ║
║  Impact on field: Nonlinear effects would (a) reduce peak pressure   ║
║  at focus below the linear prediction due to energy transfer to      ║
║  harmonics, (b) sharpen the focal spot slightly, and (c) create a    ║
║  time-averaged force landscape not captured by the linear model.     ║
║                                                                      ║
║  To model this, one would use k-Wave with medium.BonA = 0.4 (air),   ║
║  which adds the B/A nonlinearity term to the governing equations,    ║
║  or solve the Westervelt/KZK equation directly.                      ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════║
"""

NEXT_STEPS = """
╔══════════════════════════════════════════════════════════════════════║
║                        WHAT I'D DO NEXT                              ║
╠══════════════════════════════════════════════════════════════════════║
║                                                                      ║
║  With another 2 hours,  I'd add an interactive                       ║
║  visualization (e.g., Plotly or a small Streamlit app) that lets the ║
║  user drag the focal point and adjust array parameters in real time, ║
║  watching the beam pattern update — the analytical method is fast    ║
║  enough (<100 ms) to support this, and it would make the tool far    ║
║  more useful for design exploration.                                 ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════║
"""


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("\n" + "─" * 70)
    print("  ULTRASONIC PHASED ARRAY SIMULATION")
    print("  Two 8×8 arrays at 40 kHz → Common focal point")
    print("  Superposition model")
    print("─" * 70 + "\n")

    # Run analytical simulation (always works)
    p1, p2, p_total, X, Y = run_analytical_simulation()

    # Attempt k-Wave simulation
    kwave_result = run_kwave_simulation()

    # Print deliverables
    print(ASSUMPTIONS)
    print(INTERACTION_WRITEUP)
    print(NEXT_STEPS)

    # Save text deliverables
    with open(OUTPUT_DIR / 'assumptions_log.txt', 'w') as f:
        f.write(ASSUMPTIONS)
    with open(OUTPUT_DIR / 'interaction_writeup.txt', 'w') as f:
        f.write(INTERACTION_WRITEUP)
    with open(OUTPUT_DIR / 'next_steps.txt', 'w') as f:
        f.write(NEXT_STEPS)

    print(f"\nAll outputs saved to: {OUTPUT_DIR.resolve()}")
    print("Done.")
