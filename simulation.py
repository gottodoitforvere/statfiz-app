"""
Simulation of particles in a 2D box with temperature-dependent walls.

This module defines a Simulation class that models the motion of many
identical spherical particles in a unit square.  The particles move
ballistically until they collide with one another or the box walls.  When
they hit the top or bottom walls they reflect elastically.  When they
strike the left or right walls, their tangential velocity component
reflects specularly while the normal component is reset so that its
kinetic energy matches the wall temperature.  Thermalization uses a
deterministic mapping between temperature and speed, so every reflected
particle leaves the wall with the same speed magnitude implied by the
wall temperature.

Unlike the original version of this project, there are no "spring"
particles linked by a harmonic potential.  All particles are free
entities.  Functions and state pertaining to the spring particles and
their potential energy have been removed.  To update wall temperatures
at run time, call ``Simulation.set_params`` with the appropriate keywords.
"""

from __future__ import annotations

import itertools
import json
import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, Optional, Tuple, Union

import numpy as np
from numpy import ndarray

from paths import resource_file

################################################################################
# Thermal wall utility functions
################################################################################

# Boltzmann constant (scaled to simulation units).  Together with the
# particle mass this sets the characteristic thermal speed.  The value
# here is tuned so that wall temperatures of a few hundred kelvin
# produce noticeable motion for particles whose masses are taken from
# ``config.json``.
kB: float = 3.0e-6

# Base integration step in simulation time units.  Larger values speed up
# motion on screen but may slightly increase numerical error.
TIME_STEP: float = 6.0e-4

# Duration (seconds) of the rolling window used for cumulative heat plots.
HEAT_CUMULATIVE_WINDOW: float = 5.0

# Create a default random number generator.  If NumPy is unavailable
# ``rng`` will remain ``None`` and fallbacks will be used instead.
try:
    rng: np.random.Generator = np.random.default_rng()
except Exception:
    rng = None

_APP_CONFIG_CACHE: Optional[dict] = None


def _load_app_config_dict() -> Optional[dict]:
    """Return the parsed application config or ``None`` if unavailable."""
    global _APP_CONFIG_CACHE
    if _APP_CONFIG_CACHE is not None:
        return _APP_CONFIG_CACHE
    try:
        cfg_path = resource_file('config.json')
    except Exception:
        cfg_path = Path(__file__).resolve().parent / 'config.json'
    if not cfg_path.exists():
        alt = Path.cwd() / 'config.json'
        if alt.exists():
            cfg_path = alt
        else:
            return None
    try:
        with cfg_path.open('r', encoding='utf-8') as fh:
            data = json.load(fh)
    except Exception:
        return None
    if isinstance(data, dict):
        _APP_CONFIG_CACHE = data
        return data
    return None


def _thermal_speed(T: float, masses: np.ndarray) -> np.ndarray:
    """Return deterministic speeds matching the kinetic energy ``k_B T``."""
    T_value = max(float(T or 0.0), 0.0)
    speeds_sq = np.maximum(2.0 * kB * T_value / masses, 0.0)
    return np.sqrt(speeds_sq)


def _isotropic_vectors_from_speeds(speeds: np.ndarray) -> np.ndarray:
    """Create 2D velocity vectors with random directions and fixed speeds."""
    count = speeds.shape[0]
    if count == 0:
        return np.zeros((2, 0))
    if rng is not None:
        angles = rng.uniform(0.0, 2.0 * math.pi, size=count)
    else:
        angles = np.random.uniform(0.0, 2.0 * math.pi, size=count)
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)
    return np.vstack((speeds * cos_a, speeds * sin_a))

def reflect_from_wall(
    vx: np.ndarray,
    vy: np.ndarray,
    side: str,
    T_wall: float,
    masses: np.ndarray,
    mask: np.ndarray,
) -> None:
    """Apply thermal reflection rules when particles collide with a wall.

    Only the elements selected by ``mask`` are modified.  For the top and
    bottom walls a specular reflection is performed.  For the left and
    right walls, particles keep the incidence/reflection angle while the
    overall kinetic energy is re-initialised to match the wall
    temperature.  Instead of drawing new speeds from a distribution, the
    post-collision speed magnitude is set deterministically from ``k_B T``.

    Parameters
    ----------
    vx, vy: ndarray
        Arrays of x‑ and y‑velocity components for all particles.
    side: str
        One of ``'left'``, ``'right'``, ``'top'`` or ``'bottom'``.
    T_wall: float or None
        Temperature of the wall for diffusive reflection; ignored for
        top and bottom walls.
    masses: ndarray
        Masses of the particles.
    mask: ndarray of bool
        Boolean mask selecting which particles are touching the wall and
        moving toward it.  Only these particles are updated.
    """
    if not np.any(mask):
        return

    if side == 'top' or side == 'bottom':
        # Purely specular: flip the sign of the y‑component
        vy[mask] = -vy[mask]
        return

    # For left and right walls the local outward normal is +x on the left
    # wall and -x on the right wall.
    if side in ('left', 'right'):
        vx_in = vx[mask]
        vy_in = vy[mask]
        if vx_in.size == 0:
            return

        out_sign = 1.0 if side == 'left' else -1.0
        v_spec_normal = np.abs(vx_in)

        if T_wall is None or T_wall <= 0.0:
            vx[mask] = out_sign * v_spec_normal
            vy[mask] = vy_in
            return

        masses_sel = masses[mask]
        # Direction follows the specular reflection law (mirror relative to the normal)
        spec_vec = np.vstack((out_sign * v_spec_normal, vy_in))
        norms = np.linalg.norm(spec_vec, axis=0)
        zero_mask = norms < 1e-12
        if np.any(zero_mask):
            spec_vec[0, zero_mask] = out_sign
            spec_vec[1, zero_mask] = 0.0
            norms[zero_mask] = 1.0
        unit_dir = spec_vec / norms

        speed_samples = _thermal_speed(float(T_wall), masses_sel)

        vx[mask] = unit_dir[0] * speed_samples
        vy[mask] = unit_dir[1] * speed_samples
        return


@dataclass
class ThermalWallConfig:
    """Dataclass storing thermal wall parameters.

    Attributes
    ----------
    T_left, T_right: float
        Temperatures (K) of the left and right walls.  A higher temperature
        results in particles leaving the wall with higher average speed.
    """
    T_left: float = 500.0
    T_right: float = 500.0


def _load_thermal_wall_config() -> ThermalWallConfig:
    """Read default wall temperatures from ``config.json`` if available."""
    data = _load_app_config_dict()
    defaults = ThermalWallConfig()
    if not isinstance(data, dict):
        return defaults

    section = data.get('wall_temperatures')
    if not isinstance(section, dict):
        return defaults

    def _extract(entry, fallback):
        if isinstance(entry, dict):
            value = entry.get('initial', entry.get('value', fallback))
        else:
            value = entry
        try:
            return float(value)
        except (TypeError, ValueError):
            return fallback

    left_entry = section.get('left', section.get('T_left'))
    right_entry = section.get('right', section.get('T_right'))
    if right_entry is None:
        right_entry = left_entry
    return ThermalWallConfig(
        T_left=_extract(left_entry, defaults.T_left),
        T_right=_extract(right_entry, defaults.T_right),
    )


thermal_cfg = _load_thermal_wall_config()


@dataclass
class HeatFluxConfig:
    """Configuration for midplane heat flux calculations."""

    axis: str = 'x'
    position: float = 0.5
    area: Optional[float] = None
    area_provided: bool = False
    average_window: float = 0.0
    csv_path: Optional[str] = None
    csv_interval: float = 0.0


def _load_heat_flux_config() -> HeatFluxConfig:
    """Load optional heat flux settings from ``config.json`` if available."""
    defaults = HeatFluxConfig()
    data = _load_app_config_dict()
    if not isinstance(data, dict):
        return defaults

    section = data.get('midplane') or data.get('heat_flux') or {}
    if not isinstance(section, dict):
        section = {}

    axis = str(section.get('axis', defaults.axis)).lower()
    if axis not in ('x', 'y'):
        axis = defaults.axis

    position = section.get('position', defaults.position)
    area_provided = False
    area = None
    if 'area' in section:
        area = section.get('area')
        area_provided = True
    elif 'thickness' in section:
        area = section.get('thickness')
        area_provided = True
    average_window = section.get('average_window', section.get('window', defaults.average_window))
    csv_path = section.get('csv_path', defaults.csv_path)
    csv_interval = section.get('csv_interval', defaults.csv_interval)

    try:
        position = float(position)
    except (TypeError, ValueError):
        position = defaults.position
    if area is not None:
        try:
            area = max(float(area), 1e-12)
        except (TypeError, ValueError):
            area = None
    try:
        average_window = max(float(average_window), 0.0)
    except (TypeError, ValueError):
        average_window = defaults.average_window
    try:
        csv_interval = max(float(csv_interval), 0.0)
    except (TypeError, ValueError):
        csv_interval = defaults.csv_interval

    if csv_path is not None:
        csv_path = str(csv_path).strip()
        if not csv_path:
            csv_path = None

    return HeatFluxConfig(
        axis=axis,
        position=position,
        area=area,
        area_provided=area_provided,
        average_window=average_window,
        csv_path=csv_path,
        csv_interval=csv_interval,
    )


heat_flux_cfg = _load_heat_flux_config()

################################################################################
# Simulation class
################################################################################

class Simulation:
    """Evolve a gas of particles in a box with thermal walls.

    The simulation uses a simple elastic collision model between
    particles.  Particle positions and velocities are stored in
    continuous arrays for efficiency.  At each time step particles may
    collide with one another or the walls.  Left and right wall
    collisions reset the normal velocity according to the specified wall
    temperatures while the tangential component reflects specularly.
    Top and bottom wall collisions reflect particles specularly.  No
    external potentials or springs are present in this variant.
    """

    def __init__(
        self,
        gamma: float,
        k: float,
        l_0: float,
        R: float,
        particles_cnt: int,
        T: float,
        m: ndarray,
    ):
        """Create a simulation with the given parameters.

        Parameters
        ----------
        gamma, k, l_0: float
            Legacy parameters from the original model (spring potential),
            unused in this version but accepted for API compatibility.
        R: float
            Radius of each particle in box units.  The simulation assumes
            the box extends from 0 to 1 in both x and y directions, so
            ``R`` must satisfy ``0 < R < 0.5``.
        particles_cnt: int
            Number of gas particles to simulate.
        T: float
            Initial gas temperature (K).  Velocities are initialised to
            the deterministic speed implied by this temperature.
        m: ndarray
            Masses of the particles (length ``particles_cnt``).  If a
            scalar is provided, it will be broadcast to the required
            shape.  Masses should be provided in SI units consistent
            with ``kB``.
        """
        # Store constants and parameters
        self._k_boltz: float = kB
        self._gamma: float = gamma
        self._k: float = k
        self._l_0: float = l_0
        self._R: float = R
        self._box_width: float = 1.0  # horizontal extent of the domain in box units

        # Number of gas particles
        self._n_particles: int = int(particles_cnt)
        self._n_spring: int = 0  # no spring particles

        # Ensure masses have correct shape
        masses = np.asarray(m, dtype=float)
        if masses.ndim == 0:
            masses = np.full((self._n_particles,), float(masses))
        elif masses.ndim == 1 and masses.shape[0] != self._n_particles:
            raise ValueError("Length of m must equal particles_cnt")
        self._m = masses

        # Initialize positions uniformly inside the rectangular domain, avoiding walls
        x_positions = np.random.uniform(low=self._R, high=self._box_width - self._R, size=self._n_particles)
        y_positions = np.random.uniform(low=self._R, high=1.0 - self._R, size=self._n_particles)
        self._r = np.vstack((x_positions, y_positions))

        # Initialize velocities so that every particle has the deterministic
        # speed implied by k_B T while pointing in a random direction.
        base_speeds = _thermal_speed(T, self._m)
        self._v = _isotropic_vectors_from_speeds(base_speeds)

        # Save initial target temperature and energy
        self._potential_energy = []
        self._kinetic_energy = []
        self._E_full: float = self.calc_full_energy()
        self._T_tar: float = self.T

        # Thermal wall parameters
        self.T_left: float = thermal_cfg.T_left
        self.T_right: float = thermal_cfg.T_right

        # Prepare collision pairs for particle collisions
        self._init_ids_pairs()

        # Frame counter for energy fixes (unused, kept for API compatibility)
        self._frame_no: int = 1
        # Base integration time step and current scaling factor (slow-mo support)
        self._base_dt: float = TIME_STEP
        self._time_scale: float = 1.0
        self._dt: float = self._base_dt * self._time_scale

        # Track particle indices that touched each wall during the last step
        self._last_wall_hits: Dict[str, np.ndarray] = {
            'left': np.empty(0, dtype=int),
            'right': np.empty(0, dtype=int),
            'top': np.empty(0, dtype=int),
            'bottom': np.empty(0, dtype=int),
        }
        self._midplane_axis: str = heat_flux_cfg.axis
        self._midplane_axis_index: int = 0 if self._midplane_axis == 'x' else 1
        axis_range_max = self._box_width if self._midplane_axis == 'x' else 1.0
        self._midplane_position: float = float(np.clip(heat_flux_cfg.position, 0.0 + R, axis_range_max - R))
        default_area = self._box_width if self._midplane_axis == 'y' else 1.0
        area_value = heat_flux_cfg.area if heat_flux_cfg.area_provided and heat_flux_cfg.area is not None else default_area
        self._midplane_area: float = float(max(area_value, 1e-12))
        self._midplane_area_locked: bool = heat_flux_cfg.area_provided
        self._flux_average_window: float = float(max(heat_flux_cfg.average_window, 0.0))
        self._last_midplane_flux_raw: float = 0.0
        self._last_midplane_flux: float = 0.0
        self._last_midplane_heat_transfer: float = 0.0
        self._last_midplane_crossings: Dict[str, np.ndarray] = {
            'positive': np.empty(0, dtype=int),
            'negative': np.empty(0, dtype=int),
        }
        self._last_midplane_counts: Dict[str, int] = {
            'positive': 0,
            'negative': 0,
        }
        self._last_midplane_energy: Dict[str, float] = {
            'positive': 0.0,
            'negative': 0.0,
        }
        self._flux_history: list[Tuple[float, float, float]] = []
        self._max_flux_history: int = 5000
        self._flux_window: Deque[Tuple[float, float, float]] = deque()
        self._flux_window_energy: float = 0.0
        self._flux_window_duration: float = 0.0
        self._heat_cumulative_window: Deque[Tuple[float, float]] = deque()
        self._heat_cumulative_span: float = float(max(HEAT_CUMULATIVE_WINDOW, 0.0))
        self._cumulative_heat: float = 0.0
        self._flux_csv_path: Optional[str] = heat_flux_cfg.csv_path
        self._flux_csv_interval: float = float(max(heat_flux_cfg.csv_interval, 0.0))
        self._last_flux_csv_time: float = -math.inf
        if self._flux_csv_path:
            csv_file = Path(self._flux_csv_path).expanduser()
            try:
                self._flux_csv_header_written = csv_file.exists() and csv_file.stat().st_size > 0
            except OSError:
                self._flux_csv_header_written = False
        else:
            self._flux_csv_header_written = False
        self._elapsed_time: float = 0.0

    # -------------------------------------------------------------------------
    # Properties to expose slices of the state
    @property
    def r(self) -> ndarray:
        """Return positions of gas particles as a 2×N array."""
        return self._r

    @property
    def r_spring(self) -> ndarray:
        """Return positions of spring particles (empty array)."""
        return np.zeros((2, 0), dtype=float)

    @property
    def v(self) -> ndarray:
        """Return velocities of gas particles as a 2×N array."""
        return self._v

    @property
    def v_spring(self) -> ndarray:
        """Return velocities of spring particles (empty array)."""
        return np.zeros((2, 0), dtype=float)

    @property
    def m(self) -> ndarray:
        """Return masses of gas particles."""
        return self._m

    @property
    def m_spring(self) -> ndarray:
        """Return masses of spring particles (empty array)."""
        return np.zeros((0,), dtype=float)

    @property
    def R(self) -> float:
        """Radius of gas particles."""
        return self._R

    @property
    def R_spring(self) -> float:
        """Radius of spring particles (zero since none exist)."""
        return 0.0

    # -------------------------------------------------------------------------
    # Time scaling helpers ----------------------------------------------------
    def set_time_scale(self, scale: float) -> None:
        """Adjust integration step multiplier used for slow-motion or speed-up."""
        try:
            scale = float(scale)
        except (TypeError, ValueError):
            scale = 1.0
        if not math.isfinite(scale):
            scale = 1.0
        # Clamp to a sensible range to keep the integrator stable
        scale = float(np.clip(scale, 0.05, 5.0))
        self._time_scale = scale
        self._dt = self._base_dt * self._time_scale

    def get_time_scale(self) -> float:
        """Return the current integration step multiplier."""
        return self._time_scale

    def get_last_wall_hits(self) -> Dict[str, np.ndarray]:
        """Return indices of particles that touched each wall during the last step."""
        return {side: hits.copy() for side, hits in self._last_wall_hits.items()}

    def get_last_midplane_flux(self) -> float:
        """Return the most recent (optionally averaged) heat flux through the midplane."""
        return self._last_midplane_flux

    def get_last_midplane_flux_raw(self) -> float:
        """Return the most recent instantaneous heat flux sample."""
        return self._last_midplane_flux_raw

    def get_last_midplane_counts(self) -> Dict[str, int]:
        """Return counts of crossings through the midplane for the last step (positive/negative)."""
        return self._last_midplane_counts.copy()

    def get_last_midplane_energy(self) -> Dict[str, float]:
        """Return signed heat transfer contributions (per unit area) for the last step."""
        return self._last_midplane_energy.copy()

    def get_midplane_crossings(self) -> Dict[str, np.ndarray]:
        """Return indices of particles that crossed the midplane in the last step."""
        return {side: indices.copy() for side, indices in self._last_midplane_crossings.items()}

    def get_midplane_flux_history(self, raw: bool = False) -> list[Tuple[float, float]]:
        """Return recorded heat flux history as (time, flux) pairs."""
        if raw:
            return [(t, raw_flux) for t, raw_flux, _ in self._flux_history]
        return [(t, avg_flux) for t, _, avg_flux in self._flux_history]

    def get_cumulative_midplane_heat(self) -> float:
        """Return heat transferred during roughly the last five seconds (per unit area)."""
        return self._cumulative_heat

    def get_heat_cumulative_span(self) -> float:
        """Return the configured duration (seconds) used for the cumulative heat window."""
        return self._heat_cumulative_span

    def get_last_midplane_heat_transfer(self) -> float:
        """Return heat transferred during the most recent step (per unit area)."""
        return self._last_midplane_heat_transfer

    def get_midplane_axis(self) -> str:
        """Return the axis normal to the tracked midplane ('x' or 'y')."""
        return self._midplane_axis

    def get_midplane_position(self) -> float:
        """Return the midplane coordinate along its axis."""
        return self._midplane_position

    def get_midplane_area(self) -> float:
        """Return the effective cross-sectional area used for flux calculations."""
        return self._midplane_area

    def get_elapsed_time(self) -> float:
        """Return the total elapsed simulation time."""
        return self._elapsed_time

    def get_box_width(self) -> float:
        """Return the horizontal extent of the simulation domain."""
        return self._box_width

    # -------------------------------------------------------------------------
    # Thermodynamic properties
    @property
    def T(self) -> float:
        """Compute instantaneous temperature from kinetic energies (K)."""
        # The factor 2 accounts for 2 degrees of freedom per particle
        return np.mean((np.linalg.norm(self._v, axis=0) ** 2) * self._m) / (2 * self._k_boltz)

    @T.setter
    def T(self, val: float) -> None:
        if val <= 0:
            raise ValueError("Temperature must be positive")
        delta = val / self._T_tar
        # Scale velocities to achieve new temperature
        self._v *= np.sqrt(delta)
        self._E_full = self.calc_full_energy()
        self._T_tar = val

    # -------------------------------------------------------------------------
    def _init_ids_pairs(self) -> None:
        """Compute index pairs for potential collisions between gas particles."""
        particles_ids = np.arange(self._n_particles)
        self._particles_ids_pairs = np.asarray(list(itertools.combinations(particles_ids, 2)), dtype=int)

    # -------------------------------------------------------------------------
    @staticmethod
    def get_deltad2_pairs(r: np.ndarray, ids_pairs: np.ndarray) -> np.ndarray:
        """Compute squared distances between all pairs of points given by indices."""
        dx = np.diff(np.stack([r[0][ids_pairs[:, 0]], r[0][ids_pairs[:, 1]]]).T).squeeze()
        dy = np.diff(np.stack([r[1][ids_pairs[:, 0]], r[1][ids_pairs[:, 1]]]).T).squeeze()
        return dx ** 2 + dy ** 2

    @staticmethod
    def compute_new_v(
        v1: np.ndarray, v2: np.ndarray, r1: np.ndarray, r2: np.ndarray, m1: np.ndarray, m2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute post‑collision velocities for an elastic collision of two particles."""
        m_s = m1 + m2
        dr = r1 - r2
        dr_norm_sq = np.linalg.norm(dr, axis=0) ** 2
        dot1 = np.sum((2 * m2 / m_s) * (v1 - v2) * dr, axis=0)
        dot2 = np.sum((2 * m1 / m_s) * (v2 - v1) * dr, axis=0)
        v1new = v1 - (dot1 * dr) / dr_norm_sq
        v2new = v2 - (dot2 * dr) / dr_norm_sq
        return v1new, v2new

    # -------------------------------------------------------------------------
    def motion(self, dt: float) -> float:
        """Advance the system by one time step of length ``dt``.

        Returns
        -------
        float
            Always returns ``0.0`` in this version.  In the original code
            this value was the work done by the spring force.
        """
        # ------------------------------------------------------------------
        # Handle particle–particle collisions
        if self._particles_ids_pairs.size:
            # Determine which pairs are colliding based on overlap
            d2 = self.get_deltad2_pairs(self._r, self._particles_ids_pairs)
            # A collision occurs when the squared distance is less than the sum of radii squared
            colliding_mask = d2 < (2 * self._R) ** 2
            ic_particles = self._particles_ids_pairs[colliding_mask]
        else:
            ic_particles = np.zeros((0, 2), dtype=int)

        if ic_particles.size:
            # Resolve collisions by updating velocities
            v1 = self._v[:, ic_particles[:, 0]]
            v2 = self._v[:, ic_particles[:, 1]]
            r1 = self._r[:, ic_particles[:, 0]]
            r2 = self._r[:, ic_particles[:, 1]]
            m1 = self._m[ic_particles[:, 0]]
            m2 = self._m[ic_particles[:, 1]]
            v1new, v2new = self.compute_new_v(v1, v2, r1, r2, m1, m2)
            self._v[:, ic_particles[:, 0]] = v1new
            self._v[:, ic_particles[:, 1]] = v2new

            # ------------------------------------------------------------------
            # Positional correction: separate overlapping particles
            # After updating velocities, particles may still overlap because they
            # penetrated each other within one time step.  To prevent "sticking"
            # and repeated collisions, move them apart along the line of centers.
            idx_i = ic_particles[:, 0]
            idx_j = ic_particles[:, 1]
            # Vector from j to i for each colliding pair
            dr = self._r[:, idx_i] - self._r[:, idx_j]  # shape (2, K)
            # Euclidean distance between centers
            dist = np.linalg.norm(dr, axis=0)
            # Compute how much they overlap: (2R - dist).  Negative values mean no overlap
            overlap = (2.0 * self._R) - dist
            # Mask of truly overlapping pairs (distance < 2R)
            overlap_mask = overlap > 0.0
            if np.any(overlap_mask):
                # Normalize the direction vector for overlapping pairs
                # To avoid division by zero, clip very small distances
                safe_dist = np.copy(dist[overlap_mask])
                safe_dist[safe_dist < 1e-12] = 1e-12
                n = dr[:, overlap_mask] / safe_dist  # shape (2, M)
                # Each particle moves half the overlap distance in opposite directions
                shift = 0.5 * overlap[overlap_mask]
                # Broadcast shift to both components
                self._r[:, idx_i[overlap_mask]] += n * shift
                self._r[:, idx_j[overlap_mask]] -= n * shift

        # ------------------------------------------------------------------
        # Handle wall collisions: determine which particles hit which wall
        x = self._r[0]
        y = self._r[1]
        vx = self._v[0]
        vy = self._v[1]
        masses = self._m

        min_x = self._R
        max_x = self._box_width - self._R
        min_y = self._R
        max_y = 1.0 - self._R

        # Compute masks for each wall.  Particles must be moving toward the wall.
        mask_left = (x <= min_x) & (vx < 0.0)
        mask_right = (x >= max_x) & (vx > 0.0)
        mask_bottom = (y <= min_y) & (vy < 0.0)
        mask_top = (y >= max_y) & (vy > 0.0)

        # Remember which particles touched the walls this step
        self._last_wall_hits['left'] = np.where(mask_left)[0]
        self._last_wall_hits['right'] = np.where(mask_right)[0]
        self._last_wall_hits['bottom'] = np.where(mask_bottom)[0]
        self._last_wall_hits['top'] = np.where(mask_top)[0]

        # Reposition any particles that penetrated the wall back onto the boundary
        if np.any(mask_left):
            x[mask_left] = min_x
        if np.any(mask_right):
            x[mask_right] = max_x
        if np.any(mask_bottom):
            y[mask_bottom] = min_y
        if np.any(mask_top):
            y[mask_top] = max_y

        # Apply reflection or thermalization
        reflect_from_wall(vx, vy, 'left', self.T_left, masses, mask_left)
        reflect_from_wall(vx, vy, 'right', self.T_right, masses, mask_right)
        reflect_from_wall(vx, vy, 'bottom', None, masses, mask_bottom)
        reflect_from_wall(vx, vy, 'top', None, masses, mask_top)

        # ------------------------------------------------------------------
        # Integrate positions
        prev_axis_coord = self._r[self._midplane_axis_index].copy()
        self._r += self._v * dt
        self._update_midplane_flux(prev_axis_coord, dt)

        # ------------------------------------------------------------------
        return 0.0

    # -------------------------------------------------------------------------
    def __iter__(self) -> 'Simulation':
        return self

    def __next__(self) -> Tuple[ndarray, ndarray, ndarray, ndarray, float]:
        """Advance the simulation and return state arrays.

        The tuple contains positions and velocities of gas particles and
        (empty) spring particles, plus the value returned by ``motion``.
        """
        f = self.motion(dt=self._dt)
        self._frame_no = (self._frame_no + 1) % 5

        self._potential_energy.append(self.calc_potential_energy())
        self._kinetic_energy.append(self.calc_kinetic_energy())

        if self._frame_no == 0:
            self._fix_energy()

        return self.r, self.r_spring, self.v, self.v_spring, f

    # -------------------------------------------------------------------------
    def _update_midplane_flux(self, prev_axis: np.ndarray, dt: float) -> None:
        """Update heat flux across the midplane based on the last step."""
        plane = self._midplane_position if self._midplane_axis == 'y' else self._midplane_position
        if self._midplane_axis == 'x':
            plane = self._midplane_position
        axis_idx = self._midplane_axis_index
        new_axis = self._r[axis_idx]
        prev_axis_arr = np.asarray(prev_axis, dtype=float)
        if prev_axis_arr.shape != new_axis.shape:
            prev_axis_arr = np.reshape(prev_axis_arr, new_axis.shape)

        positive_mask = (prev_axis_arr < plane) & (new_axis >= plane)
        negative_mask = (prev_axis_arr > plane) & (new_axis <= plane)
        positive_idx = np.nonzero(positive_mask)[0]
        negative_idx = np.nonzero(negative_mask)[0]

        self._last_midplane_crossings['positive'] = positive_idx
        self._last_midplane_crossings['negative'] = negative_idx

        pos_count = int(positive_idx.size)
        neg_count = int(negative_idx.size)
        self._last_midplane_counts['positive'] = pos_count
        self._last_midplane_counts['negative'] = neg_count

        velocities_sq = np.sum(self._v * self._v, axis=0)
        energies = 0.5 * self._m * velocities_sq
        pos_energy = float(np.sum(energies[positive_idx])) if pos_count else 0.0
        neg_energy = float(np.sum(energies[negative_idx])) if neg_count else 0.0
        net_energy = pos_energy - neg_energy

        if dt > 0.0:
            self._elapsed_time += dt
        timestamp = self._elapsed_time

        heat_transfer = net_energy / self._midplane_area
        self._last_midplane_heat_transfer = heat_transfer
        self._update_cumulative_heat_window(timestamp, heat_transfer)
        self._last_midplane_energy['positive'] = pos_energy / self._midplane_area
        self._last_midplane_energy['negative'] = -neg_energy / self._midplane_area

        flux_raw = heat_transfer / dt if dt > 0.0 else 0.0
        self._last_midplane_flux_raw = flux_raw

        if self._flux_average_window > 0.0 and dt > 0.0:
            avg_flux = self._update_flux_window(net_energy, dt)
        else:
            avg_flux = flux_raw
            if self._flux_average_window <= 0.0:
                self._flux_window.clear()
                self._flux_window_energy = 0.0
                self._flux_window_duration = 0.0
        self._last_midplane_flux = avg_flux

        self._flux_history.append((timestamp, flux_raw, avg_flux))
        if len(self._flux_history) > self._max_flux_history:
            self._flux_history.pop(0)

        self._maybe_write_flux_csv(timestamp)

    def _update_flux_window(self, net_energy: float, dt: float) -> float:
        """Update the rolling window used to time-average the flux."""
        window = self._flux_average_window
        if window <= 0.0 or dt <= 0.0:
            return self._last_midplane_flux_raw

        timestamp = self._elapsed_time
        self._flux_window.append((timestamp, net_energy, dt))
        self._flux_window_energy += net_energy
        self._flux_window_duration += dt

        # Trim to the configured window length, allowing partial removal.
        while self._flux_window and self._flux_window_duration - self._flux_window[0][2] > window:
            _, energy_old, dt_old = self._flux_window.popleft()
            self._flux_window_energy -= energy_old
            self._flux_window_duration -= dt_old

        if self._flux_window and self._flux_window_duration > window:
            excess = self._flux_window_duration - window
            ts_old, energy_old, dt_old = self._flux_window[0]
            if dt_old > 0.0:
                fraction = min(1.0, excess / dt_old)
                energy_remove = energy_old * fraction
                dt_remove = dt_old * fraction
                remaining_energy = max(0.0, energy_old - energy_remove)
                remaining_dt = max(0.0, dt_old - dt_remove)
                self._flux_window[0] = (ts_old, remaining_energy, remaining_dt)
                self._flux_window_energy -= energy_remove
                self._flux_window_duration -= dt_remove

        duration = self._flux_window_duration if self._flux_window_duration > 0.0 else dt
        if duration <= 0.0:
            return self._last_midplane_flux_raw
        return (self._flux_window_energy / self._midplane_area) / duration

    def _update_cumulative_heat_window(self, timestamp: float, heat_transfer: float) -> None:
        """Keep only the last few seconds of heat contributions in the cumulative total."""
        if not math.isfinite(heat_transfer):
            return
        window = self._heat_cumulative_span
        if window <= 0.0:
            self._cumulative_heat += heat_transfer
            return

        if not math.isfinite(timestamp):
            timestamp = self._elapsed_time

        self._heat_cumulative_window.append((timestamp, heat_transfer))
        self._cumulative_heat += heat_transfer

        cutoff = timestamp - window
        while self._heat_cumulative_window and self._heat_cumulative_window[0][0] < cutoff:
            _, old_value = self._heat_cumulative_window.popleft()
            self._cumulative_heat -= old_value

        # Avoid accumulating tiny residuals over long runs
        if abs(self._cumulative_heat) < 1e-18:
            self._cumulative_heat = 0.0

    def _maybe_write_flux_csv(self, timestamp: float) -> None:
        """Optionally append the latest flux sample to a CSV log."""
        path = self._flux_csv_path
        if not path:
            return
        interval = self._flux_csv_interval
        if interval > 0.0 and timestamp - self._last_flux_csv_time < interval:
            return

        file_path = Path(path).expanduser()
        try:
            if file_path.parent and not file_path.parent.exists():
                file_path.parent.mkdir(parents=True, exist_ok=True)
            need_header = not file_path.exists() or not self._flux_csv_header_written
            with file_path.open('a', encoding='utf-8') as fh:
                if need_header:
                    fh.write('time,heat_flux,cumulative_heat\n')
                    self._flux_csv_header_written = True
                fh.write(f"{timestamp:.9f},{self._last_midplane_flux:.9g},{self._cumulative_heat:.9g}\n")
        except OSError:
            # Ignore logging errors to keep the simulation running.
            return

        self._last_flux_csv_time = timestamp

    # -------------------------------------------------------------------------
    def reset_midplane_statistics(self) -> None:
        """Clear accumulated flux statistics and heat counters."""
        self._cumulative_heat = 0.0
        self._last_midplane_heat_transfer = 0.0
        self._last_midplane_flux = 0.0
        self._last_midplane_flux_raw = 0.0
        self._flux_history.clear()
        self._flux_window.clear()
        self._flux_window_energy = 0.0
        self._flux_window_duration = 0.0
        self._heat_cumulative_window.clear()
        self._last_midplane_counts = {'positive': 0, 'negative': 0}
        self._last_midplane_crossings = {
            'positive': np.empty(0, dtype=int),
            'negative': np.empty(0, dtype=int),
        }
        self._last_midplane_energy = {'positive': 0.0, 'negative': 0.0}
        # Keep elapsed time so timestamps remain monotonic; callers can
        # decide whether to interpret time relative to the reset moment.

    # -------------------------------------------------------------------------
    def add_particles(self, r: ndarray, v: ndarray, m: ndarray) -> None:
        """Add new gas particles to the simulation.

        Parameters
        ----------
        r: ndarray
            Positions of the new particles, shape (2, N_new).
        v: ndarray
            Velocities of the new particles, shape (2, N_new).
        m: ndarray
            Masses of the new particles, shape (N_new,).
        """
        if r.shape != v.shape or r.shape[0] != 2 or r.shape[1] != m.shape[0]:
            raise ValueError("Shapes of r, v and m are inconsistent")
        self._r = np.hstack([self._r, r])
        self._v = np.hstack([self._v, v])
        self._m = np.hstack([self._m, m])
        self._n_particles += r.shape[1]
        # Recompute collision pairs
        self._init_ids_pairs()
        self._E_full = self.calc_full_energy()
        self._T_tar = self.T

    # -------------------------------------------------------------------------
    def _set_particles_cnt(self, particles_cnt: int) -> None:
        """Reset the number of gas particles to the given count.

        If the new count is smaller than the current count, particles are
        removed from the end.  If it is larger, new particles are added
        with random positions and velocities sampled from the current
        speed distribution and median mass.
        """
        if particles_cnt < 0:
            raise ValueError("particles_cnt must be >= 0")
        if particles_cnt < self._n_particles:
            idx = slice(particles_cnt)
            self._r = self._r[:, idx]
            self._v = self._v[:, idx]
            self._m = self._m[idx]
        if particles_cnt > self._n_particles:
            new_cnt = particles_cnt - self._n_particles
            # Positions uniformly distributed away from walls
            x_new = np.random.uniform(low=self._R, high=self._box_width - self._R, size=new_cnt)
            y_new = np.random.uniform(low=self._R, high=1.0 - self._R, size=new_cnt)
            new_r = np.vstack((x_new, y_new))
            # Assign deterministic speeds based on the current target temperature
            new_m = np.full((new_cnt,), np.median(self._m) if self._m.size > 0 else 1.0)
            new_speed = _thermal_speed(self._T_tar, new_m)
            new_v = _isotropic_vectors_from_speeds(new_speed)
            self.add_particles(new_r, new_v, new_m)
        if particles_cnt != self._n_particles:
            self._n_particles = particles_cnt
            self._init_ids_pairs()
        self._E_full = self.calc_full_energy()
        self._T_tar = self.T

    # -------------------------------------------------------------------------
    def set_params(
        self,
        gamma: float = None,
        k: float = None,
        l_0: float = None,
        R: float = None,
        T: float = None,
        m: float = None,
        particles_cnt: int = None,
        T_left: float = None,
        T_right: float = None,
    ) -> None:
        """Update simulation parameters on the fly.

        Parameters correspond to those accepted by the constructor.  Any
        parameter passed as ``None`` will be left unchanged.
        """
        if gamma is not None:
            self._gamma = float(gamma)
        if k is not None:
            self._k = float(k)
        if l_0 is not None:
            self._l_0 = float(l_0)
        if R is not None:
            self._R = float(R)
            min_width = max(2.0 * self._R + 1e-6, 1e-3)
            if self._box_width < min_width:
                self.set_box_width(min_width)
        if T is not None:
            self.T = float(T)
        if m is not None:
            if m <= 0:
                raise ValueError("m must be > 0")
            self._m[:] = float(m)
        if particles_cnt is not None:
            self._set_particles_cnt(int(particles_cnt))
        if T_left is not None:
            self.T_left = float(T_left)
        if T_right is not None:
            self.T_right = float(T_right)
        # Recompute full energy after parameter changes
        self._E_full = self.calc_full_energy()
        self._T_tar = self.T

    def set_box_width(self, width: float) -> None:
        """Adjust the horizontal size of the simulation domain."""
        width = float(max(width, 2.0 * self._R + 1e-6))
        if width <= 0:
            width = 1.0
        if abs(width - self._box_width) < 1e-9:
            return
        old_width = self._box_width
        scale = width / old_width if old_width > 0 else 1.0
        self._box_width = width
        self._r[0] *= scale
        self._r[0] = np.clip(self._r[0], self._R, self._box_width - self._R)
        if self._midplane_axis == 'x':
            self._midplane_position = float(np.clip(self._midplane_position * scale, self._R, self._box_width - self._R))
        else:
            self._midplane_position = float(np.clip(self._midplane_position, self._R, 1.0 - self._R))
        if self._midplane_axis == 'y' and not self._midplane_area_locked:
            self._midplane_area = float(max(self._box_width, 1e-12))
        self._init_ids_pairs()

    # -------------------------------------------------------------------------
    def expected_potential_energy(self) -> float:
        """Return zero since no external potential exists."""
        return 0.0

    def expected_kinetic_energy(self) -> float:
        """Return the expected kinetic energy per particle (k_B T)."""
        return float(self._k_boltz * self._T_tar)

    def calc_kinetic_energy(self) -> float:
        """Calculate mean kinetic energy of gas particles."""
        return np.mean((np.linalg.norm(self._v, axis=0) ** 2) * self._m) / 2.0

    def calc_full_kinetic_energy(self) -> float:
        """Calculate total kinetic energy of gas particles."""
        return np.sum((np.linalg.norm(self._v, axis=0) ** 2) * self._m) / 2.0

    def _fix_energy(self) -> None:
        """Gently counteract numerical drift without blocking heat exchange.

        The total energy should change only due to wall interactions or
        deliberate parameter updates.  We therefore track a slowly varying
        target energy and only rescale velocities when the instantaneous
        energy deviates slightly from that target (typical of integration
        error).  Substantial changes driven by the walls are preserved.
        """
        current_E = self.calc_full_energy()
        if current_E <= 0.0:
            return

        if self._E_full <= 0.0:
            self._E_full = current_E
            return

        relax = 0.1
        self._E_full = (1.0 - relax) * self._E_full + relax * current_E
        scale = self._E_full / current_E
        if scale <= 0.0:
            return
        scale = math.sqrt(scale)
        if abs(scale - 1.0) < 0.05:
            self._v *= scale

    def calc_full_energy(self) -> float:
        """Return the total energy (purely kinetic)."""
        return self.calc_full_kinetic_energy()

    def calc_potential_energy(self) -> float:
        """Return zero since there is no potential energy."""
        return 0.0

    def mean_potential_energy(self, frames_c: Union[int, None] = None) -> float:
        """Always return zero in absence of potential energy."""
        return 0.0

    def mean_kinetic_energy(self, frames_c: Union[int, None] = None) -> float:
        """Return the mean of the stored kinetic energy history."""
        if frames_c is None:
            if not self._kinetic_energy:
                return 0.0
            return float(np.mean(self._kinetic_energy))
        else:
            if not self._kinetic_energy:
                return 0.0
            return float(np.mean(self._kinetic_energy[-frames_c:]))
