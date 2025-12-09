"""
Modified Demo module without spring particles.

This version of ``demo.py`` interfaces with the updated ``Simulation``
class that lacks spring particles.  It removes all logic associated with
the spring particle pair and renders only the gas particles.  The
parameters for the spring (``R``, ``m_spring``) are ignored.  The
potential energy arrays remain for compatibility but will contain zeroes.
"""

import math
import pygame
import numpy as np
import config
from typing import Dict, Optional
from simulation import Simulation, kB

# Unified simulation speed range shared with the UI.
SPEED_MIN = 0.2
SPEED_MAX = 6.0

# Flux graph visuals
FLUX_GRAPH_DEFAULT_HALF_RANGE = 60.0
FLUX_GRAPH_DEFAULT_GRID_LINES = 5
FLUX_GRID_COLOR = (222, 227, 242)
FLUX_GRAPH_FOLLOW_WINDOW = 240
FLUX_GRAPH_PADDING_FRACTION = 0.15
FLUX_GRAPH_EXTRA_PADDING = 0.25
# Time range (seconds) shown on the flux chart
FLUX_GRAPH_TIME_WINDOW = 10.0
# Graph zooming and axis shaping
FLUX_GRAPH_PARTICLE_REF = 200.0
FLUX_GRAPH_MIN_ZOOM = 0.35
FLUX_GRAPH_NEGATIVE_LIMIT_FRACTION = 0.25
# Visual scale for displaying flux/heat numbers (multiplies raw values).
# Reduced to keep onscreen numbers in a readable range.
FLUX_GRAPH_DISPLAY_SCALE = 1.0e2
FLUX_HIGHLIGHT_COLOR = (180, 88, 88, 120)
FLUX_HIGHLIGHT_WIDTH = 2

# Midplane divider visuals
MIDPLANE_WALL_THICKNESS_RATIO = 0.0024
MIDPLANE_WALL_COLOR = (0, 200, 0, 215)


class Demo:
    def __init__(
        self,
        app,
        position,
        demo_size,
        bg_color,
        border_color,
        bg_screen_color,
        params,
        wall_temp_bounds: tuple[float, float] | None = None,
    ):
        """
        Initialize a new demonstration instance.

        Parameters
        ----------
        app : App
            Reference to the parent application containing the pygame screen.
        position : tuple
            (x, y) coordinates of the top‑left corner of the simulation box.
        demo_size : tuple
            (width, height) of the simulation area in pixels.
        bg_color : tuple
            RGB background colour of the simulation area.
        border_color : tuple
            RGB colour of the border around the simulation area.
        bg_screen_color : tuple
            Colour used for the masked outer border region.
        params : dict
            Dictionary of initial simulation parameters (gamma, k, R, T, r, etc.).
        """
        self.app = app
        self.screen = app.screen
        self.bg_color = bg_color
        self.bg_screen_color = bg_screen_color
        self.bd_color = border_color
        self.position = position
        # Pygame rect describing the simulation area
        self.main = pygame.Rect(*position, *demo_size)
        # Store individual dimensions so the demo can be rectangular
        self.width, self.height = demo_size
        if wall_temp_bounds and wall_temp_bounds[1] > wall_temp_bounds[0]:
            self.wall_temp_bounds = (float(wall_temp_bounds[0]), float(wall_temp_bounds[1]))
        else:
            self.wall_temp_bounds = (100.0, 2000.0)
        self._wall_colors = {'left': (220, 70, 40), 'right': (60, 130, 255)}
        speed_default = params.get('speed', params.get('slowmo', 1.0))
        # Copy of the initial parameter values used by sliders
        # Copy of the initial parameter values used by sliders.  Remove keys
        # corresponding to unused simulation parameters (gamma, k, mass of spring,
        # spring radius, etc.) to avoid storing unneeded data.  If those keys
        # are not present, ``pop`` simply returns ``None``.  This keeps
        # ``self.params`` compact and eliminates references to unused legacy
        # spring parameters.
        self.params = dict(params)
        for unused_key in ('gamma', 'k', 'm_spring', 'R', 'R_spring', 'radius_scale'):
            self.params.pop(unused_key, None)
        self.params.pop('slowmo', None)
        self.modified_par = None
        self.flux_display_scale: float = FLUX_GRAPH_DISPLAY_SCALE
        loader = config.ConfigLoader()
        self._configure_flux_graph(loader)

        # Scale factors for physical collisions and rendering.  Start 30% larger
        # to make particles more visible by default.
        initial_scale = float(params.get('size_scale', 1.3))
        if initial_scale < 0.1:
            initial_scale = 0.1
        self.physical_radius_scale = initial_scale
        self.draw_radius_factor = initial_scale

        # Keep track of indices of tagged (marked) particles.  These will
        # be drawn in a distinct colour and can be analysed separately.
        self.tagged_indices: list[int] = []
        # Tagged particles must stay clearly visible on projectors with
        # high brightness, so we render them in a dark brown tone instead
        # of yellow.
        self.tagged_color: tuple[int, int, int] = (48, 32, 20)
        # Rendering helpers for highlight/dim features
        self.dim_untracked: bool = False
        self.dim_color: tuple[int, int, int] = (212, 212, 218)

        # Trajectory tracking for a tagged particle
        self.trail_enabled: bool = False
        self.trail_points: list[tuple[int, int]] = []
        self.max_trail_points: int = 1500
        self.tracked_particle_id: Optional[int] = None

        # Counters for wall contacts by tagged particles
        self.wall_hits = {'left': 0, 'right': 0}
        # Flux tracking across the midplane (x = 0.5)
        self.midplane_axis: str = 'x'
        self.midplane_position: float = 0.5
        self.midplane_flux_samples: list[tuple[float, float, float, float]] = []
        self.max_flux_samples: int = 2000
        self.last_flux_limits: Optional[tuple[float, float]] = None
        self.reset_wall_hit_counters()

        # Colour scaling for particle temperatures
        self.color_gamma: float = 0.75
        self._color_scale_min: float = 0.0
        self._color_scale_max: float = 1.0
        self._set_color_scale_bounds(self.wall_temp_bounds)

        # Unified speed control: slider value maps to step count and integrator scale
        self.time_scale: float = 1.0
        self.speed_factor: float = self._normalize_speed_value(speed_default)
        self._speed_steps: int = max(1, int(math.floor(self.speed_factor)))
        self.params['speed'] = self.speed_factor

        # Masses for gas particles only.  ``self.params['r']`` specifies the
        # number of gas particles; there are no spring masses in this
        # simplified model.  Masses are drawn from configuration.
        m = np.ones((self.params['r'],), dtype=float) * loader["R_mass"]

        # Legacy parameters (gamma, k, l_0) are passed through for API compatibility
        l_0 = loader['l_0']
        # Cache the base particle radius from the configuration.  ``R_size``
        # defines the nominal physical radius (in box units).  By
        # storing this value we can compute new radii when the user
        # adjusts the particle size via the UI.
        self.base_radius = loader["R_size"]
        # Initialize the simulation.  Multiply the base radius by our
        # physical scale factor to enlarge the physical collision size.
        # Legacy parameters may be absent in ``params`` because the UI
        # does not expose sliders for them.  Fetch them with default
        # fallbacks.  ``gamma`` and ``k`` are unused in the current
        # simulation but accepted for API compatibility.
        gamma_val = params.get('gamma', 1.0)
        k_val = params.get('k', 1.0)
        # Create the simulation with legacy parameters, particle radius and counts.
        self.simulation = Simulation(
            gamma=gamma_val,
            k=k_val,
            l_0=l_0,
            R=self.base_radius * self.physical_radius_scale,
            particles_cnt=self.params['r'],
            T=self.params['T'],
            m=m,
        )
        self._update_simulation_box_aspect()
        self.midplane_axis = self.simulation.get_midplane_axis()
        self.midplane_position = self.simulation.get_midplane_position()
        # If thermal wall parameters are provided in params, set them on the simulation
        # (These keys may not exist in older configs.)
        t_left = params.get('T_left')
        t_right = params.get('T_right')
        update_kwargs: Dict[str, float] = {}
        if t_left is not None:
            update_kwargs['T_left'] = t_left
        if t_right is not None:
            update_kwargs['T_right'] = t_right
        if update_kwargs:
            self.simulation.set_params(**update_kwargs)

        # Apply the initial speed factor after the simulation is created.
        self.update_speed_factor(self.speed_factor, force=True)

    def _update_simulation_box_aspect(self) -> None:
        """Resize the simulation domain horizontally to match the viewport ratio."""
        if self.height <= 0:
            return
        wall_thickness = max(6, int(round(self.width * 0.015)))
        usable_width = max(1, self.width - 2 * wall_thickness)
        self.wall_thickness_px = wall_thickness
        self.sim_left_px = self.position[0] + wall_thickness
        self.sim_width_px = usable_width
        aspect = usable_width / self.height if self.height > 0 else 1.0
        if aspect <= 0:
            aspect = 1.0
        self.simulation.set_box_width(aspect)

    def _temperature_to_color(self, temperature: float) -> tuple[int, int, int]:
        """
        Map a wall temperature to the same blue-to-red gradient used for particles.
        """
        try:
            temp_value = float(temperature)
        except (TypeError, ValueError):
            temp_value = 0.0
        normalized = self._normalize_temperatures(np.array([temp_value], dtype=float))
        norm = float(normalized[0]) if normalized.size else 0.0
        norm = max(0.0, min(1.0, norm))
        red = int(round(255 * norm))
        blue = int(round(255 * (1.0 - norm)))
        return red, 0, blue

    def get_wall_color(self, side: str) -> tuple[int, int, int]:
        """Return the most recently rendered colour for the given wall."""
        return self._wall_colors.get(side, (255, 255, 255))

    def update_radius_scale(self, scale: float) -> None:
        """
        Update both the physical and visual radius scales of the particles.

        This method is intended to be called from the demo screen when
        the user changes the particle size slider.  It sets the
        ``physical_radius_scale`` and ``draw_radius_factor`` to the same
        value, updates the simulation's physical radius accordingly,
        and stores the new scale.  Changing the physical radius
        influences collision detection, while changing the draw factor
        adjusts the rendered size on screen.  Both are applied at once
        so that the user perceives the change consistently.

        Parameters
        ----------
        scale : float
            New scaling factor (1.0 means default size; higher values
            enlarge particles; lower values shrink them).
        """
        # Avoid zero or negative scales to keep a valid radius.
        if scale < 0.1:
            scale = 0.1
        # Store the new scale for both physical interactions and drawing
        self.physical_radius_scale = scale
        self.draw_radius_factor = scale
        # Compute the new physical radius using the cached base radius
        new_R = self.base_radius * self.physical_radius_scale
        # Update the simulation with the new radius
        self.simulation.set_params(R=new_R)

    def set_time_scale(self, scale: float) -> None:
        """Expose slow-motion control for the simulation loop."""
        if scale < 0.05:
            scale = 0.05
        self.time_scale = float(scale)
        self.simulation.set_time_scale(self.time_scale)

    def _normalize_speed_value(self, value: float) -> float:
        """Clamp raw speed input to the supported slider range."""
        try:
            value = float(value)
        except (TypeError, ValueError):
            value = 1.0
        if not math.isfinite(value):
            value = 1.0
        return max(SPEED_MIN, min(SPEED_MAX, value))

    def update_speed_factor(self, raw_value: float, force: bool = False) -> float:
        """
        Update simulation stepping based on a single unified speed slider.

        The slider value controls both the number of integration steps per frame
        and the integrator time multiplier.  Values above 1.0 execute multiple
        substeps while keeping each substep stable; values below 1.0 slow the
        simulation down.
        """
        value = self._normalize_speed_value(raw_value)
        steps = max(1, int(math.floor(value)))
        step_scale = value / steps
        speed_changed = force or steps != getattr(self, '_speed_steps', 1)
        scale_changed = force or abs(step_scale - self.time_scale) > 1e-4
        self.speed_factor = value
        self._speed_steps = steps
        self.params['speed'] = value
        if speed_changed or scale_changed:
            self.set_time_scale(step_scale)
        return self.speed_factor

    def resize_viewport(self, position: tuple[int, int], demo_size: tuple[int, int]) -> None:
        """Adjust the rendering viewport to a new rectangle."""
        self.position = position
        self.main = pygame.Rect(*position, *demo_size)
        self.width, self.height = demo_size
        self.screen = self.app.screen
        self._update_simulation_box_aspect()


    def set_dim_untracked(self, dim: bool) -> None:
        """Enable or disable dimming of untagged particles."""
        self.dim_untracked = bool(dim)

    def set_trail_enabled(self, enabled: bool) -> None:
        """Toggle trajectory rendering for the tracked tagged particle."""
        new_state = bool(enabled)
        if new_state and not self.trail_enabled:
            # Reset trail when enabling to avoid stale points.
            self.trail_points.clear()
        if not new_state:
            self.trail_points.clear()
        self.trail_enabled = new_state
        self._ensure_tracked_particle()

    def has_tagged_particles(self) -> bool:
        """Return True when at least one tagged particle exists."""
        return bool(self.tagged_indices)

    def get_wall_hit_counts(self) -> tuple[int, int]:
        """Return accumulated counts of tagged particles hitting left/right walls."""
        return self.wall_hits['left'], self.wall_hits['right']

    def reset_wall_hit_counters(self) -> None:
        """Clear stored hit counts for both walls."""
        self.wall_hits['left'] = 0
        self.wall_hits['right'] = 0
    def _sync_tagged_indices(self, particle_count: int) -> None:
        """Ensure tagged indices remain valid after particle count changes."""
        if not self.tagged_indices:
            return
        max_index = max(0, int(particle_count))
        filtered = [idx for idx in self.tagged_indices if 0 <= int(idx) < max_index]
        if len(filtered) != len(self.tagged_indices):
            self.tagged_indices = filtered
            self.reset_wall_hit_counters()
            if self.trail_points:
                self.trail_points.clear()
        if not self.tagged_indices:
            self.tracked_particle_id = None
            return
        if self.tracked_particle_id not in self.tagged_indices:
            self.tracked_particle_id = self.tagged_indices[0]
            if self.trail_points:
                self.trail_points.clear()

    def get_half_concentrations(self) -> tuple[float, float]:
        """Return particle concentration in the left and right halves of the box."""
        positions = self.simulation.r[0]
        width = max(self.simulation.get_box_width(), 1e-9)
        half = width / 2.0
        left_count = float(np.count_nonzero(positions < half))
        right_count = float(positions.size - left_count)
        half_area = max(half, 1e-9)
        left_density = left_count / half_area
        right_density = right_count / half_area
        return left_density, right_density

    def reset_measurements(self) -> None:
        """Reset visual counters, flux history and the tracked trail."""
        self.reset_wall_hit_counters()
        self.midplane_flux_samples.clear()
        self.last_flux_limits = None
        if self.trail_points:
            self.trail_points.clear()
        self.simulation.reset_midplane_statistics()
        self._ensure_tracked_particle()

    def clear_tagged_particles(self) -> int:
        """Remove all tagged particles that were added to the simulation."""
        removed = len(self.tagged_indices)
        if removed <= 0:
            if self.trail_points:
                self.trail_points.clear()
            self._ensure_tracked_particle()
            return 0
        target_count = max(0, int(self.simulation._n_particles) - removed)
        self.simulation.set_params(particles_cnt=target_count)
        self.params['r'] = target_count
        self.tagged_indices.clear()
        self.tracked_particle_id = None
        if self.trail_points:
            self.trail_points.clear()
        self.reset_wall_hit_counters()
        self._ensure_tracked_particle()
        return removed

    def _ensure_tracked_particle(self) -> None:
        """Ensure the tracked particle id refers to an existing tagged particle."""
        if not self.tagged_indices:
            self.tracked_particle_id = None
            if self.trail_points:
                self.trail_points.clear()
            return
        if self.tracked_particle_id not in self.tagged_indices:
            # Prefer the earliest tagged particle for consistency.
            self.tracked_particle_id = self.tagged_indices[0]
            self.trail_points.clear()

    def _update_wall_hit_counters(self) -> None:
        """Increment counters when tagged particles touch left/right walls."""
        if not self.tagged_indices:
            return
        tagged_set = set(self.tagged_indices)
        hits = self.simulation.get_last_wall_hits()
        for idx in hits.get('left', []):
            if int(idx) in tagged_set:
                self.wall_hits['left'] += 1
        for idx in hits.get('right', []):
            if int(idx) in tagged_set:
                self.wall_hits['right'] += 1

    def _normalize_temperatures(self, temperatures: np.ndarray) -> np.ndarray:
        """Return colour-normalised intensities for the supplied temperatures."""
        temps = np.asarray(temperatures, dtype=float)
        if temps.size == 0:
            return np.zeros_like(temps, dtype=float)
        low = float(self._color_scale_min)
        high = float(self._color_scale_max)
        span = max(high - low, 1e-9)
        temps = np.nan_to_num(temps, nan=low, neginf=low, posinf=high)
        normalized = np.clip((temps - low) / span, 0.0, 1.0)
        gamma = float(self.color_gamma)
        if gamma not in (1.0, 0.0):
            normalized = np.power(normalized, gamma)
        return normalized

    def _set_color_scale_bounds(self, bounds: Optional[tuple[float, float]]) -> None:
        """Configure the fixed temperature range used for colour gradients."""
        default_low, default_high = 100.0, 2000.0
        low, high = default_low, default_high
        if bounds and len(bounds) == 2:
            try:
                low = float(bounds[0])
                high = float(bounds[1])
            except (TypeError, ValueError):
                low, high = default_low, default_high
        if not math.isfinite(low):
            low = default_low
        if not math.isfinite(high):
            high = default_high
        if high <= low:
            high = low + 1.0
        self.wall_temp_bounds = (low, high)
        self._color_scale_min = max(0.0, low)
        self._color_scale_max = max(self._color_scale_min + 1e-6, high)

    def set_wall_color_range(self, bounds: Optional[tuple[float, float]]) -> None:
        """
        Public helper to update the temperature bounds driving the colour gradient.
        """
        self._set_color_scale_bounds(bounds or self.wall_temp_bounds)

    def _record_trail_point(self, pixel_positions: np.ndarray) -> None:
        """Append the current screen-space position of the tracked particle."""
        if not self.trail_enabled:
            return
        self._ensure_tracked_particle()
        if self.tracked_particle_id is None:
            return
        idx = int(self.tracked_particle_id)
        if idx >= pixel_positions.shape[1]:
            return
        point = (int(pixel_positions[0, idx]), int(pixel_positions[1, idx]))
        if self.trail_points and self.trail_points[-1] == point:
            return
        self.trail_points.append(point)

    def _store_flux_sample(self, timestamp: float, flux_raw: float, flux_avg: float, cumulative: float) -> None:
        """Cache the latest heat-flux values for future visualisation."""
        if not math.isfinite(timestamp):
            return
        if not math.isfinite(flux_raw) or not math.isfinite(flux_avg) or not math.isfinite(cumulative):
            return
        if self.midplane_flux_samples and abs(self.midplane_flux_samples[-1][0] - timestamp) < 1e-9:
            self.midplane_flux_samples[-1] = (timestamp, flux_raw, flux_avg, cumulative)
        else:
            self.midplane_flux_samples.append((timestamp, flux_raw, flux_avg, cumulative))
        if len(self.midplane_flux_samples) > self.max_flux_samples:
            self.midplane_flux_samples.pop(0)

        window_span = getattr(self, 'flux_time_window', FLUX_GRAPH_TIME_WINDOW)
        try:
            window_span = float(window_span)
        except (TypeError, ValueError):
            window_span = FLUX_GRAPH_TIME_WINDOW
        if not math.isfinite(window_span) or window_span <= 0.0:
            window_span = FLUX_GRAPH_TIME_WINDOW

        sim_window = None
        if self.simulation is not None:
            try:
                sim_window = float(self.simulation.get_heat_cumulative_span())
            except (TypeError, ValueError):
                sim_window = None
        if sim_window is not None and math.isfinite(sim_window) and sim_window > 0.0:
            window_span = max(window_span, sim_window)

        cutoff = timestamp - window_span
        while self.midplane_flux_samples and self.midplane_flux_samples[0][0] < cutoff:
            self.midplane_flux_samples.pop(0)

    def _draw_midplane_wall(self) -> None:
        """Render a semi-transparent dashed divider at the domain centre."""
        color = MIDPLANE_WALL_COLOR
        ratio = MIDPLANE_WALL_THICKNESS_RATIO
        rect = pygame.Rect(self.position[0], self.position[1], self.width, self.height)
        box_width_units = max(self.simulation.get_box_width(), 1e-6)
        plane_coord = self.simulation.get_midplane_position()
        if self.midplane_axis == 'y':
            wall_height = max(1, int(round(rect.height * ratio)))
            dash_length = max(10, int(round(self.sim_width_px * 0.06)))
            gap_length = max(6, int(round(dash_length * 0.6)))
            wall_surface = pygame.Surface((self.sim_width_px, wall_height), pygame.SRCALPHA)
            center_y = wall_height // 2
            x = 0
            while x < self.sim_width_px:
                end_x = min(self.sim_width_px, x + dash_length)
                pygame.draw.line(wall_surface, color, (x, center_y), (end_x, center_y), wall_height)
                x = end_x + gap_length
            norm = float(np.clip(plane_coord, self.simulation.R, 1.0 - self.simulation.R))
            wall_y = rect.top + rect.height - int(round(rect.height * norm)) - wall_height // 2
            wall_y = max(rect.top, min(rect.bottom - wall_height, wall_y))
            self.screen.blit(wall_surface, (self.sim_left_px, wall_y))
        else:
            wall_width = max(1, int(round(rect.width * ratio)))
            dash_length = max(10, int(round(rect.height * 0.06)))
            gap_length = max(6, int(round(dash_length * 0.6)))
            wall_surface = pygame.Surface((wall_width, rect.height), pygame.SRCALPHA)
            center_x = wall_width // 2
            y = 0
            while y < rect.height:
                end_y = min(rect.height, y + dash_length)
                pygame.draw.line(wall_surface, color, (center_x, y), (center_x, end_y), wall_width)
                y = end_y + gap_length
            norm = float(np.clip(plane_coord / box_width_units, 0.0, 1.0))
            wall_x = self.sim_left_px + int(round(self.sim_width_px * norm)) - wall_width // 2
            wall_x = max(self.sim_left_px, min(self.sim_left_px + self.sim_width_px - wall_width, wall_x))
            self.screen.blit(wall_surface, (wall_x, rect.top))

    def set_params(self, params, par):
        # Dispatch updated simulation parameters based on the changed
        # parameter name.  Legacy parameters such as ``gamma`` and ``k``
        # are no longer processed, because they have no effect in the
        # current model.  Updates to particle size are handled in the
        # DemoScreen via ``update_radius_scale``.
        if par == 'T':
            self.simulation.set_params(T=params['T'])
        elif par == 'r':
            self.simulation.set_params(particles_cnt=params['r'])
            self._sync_tagged_indices(self.simulation._n_particles)
        elif par == 'speed':
            self.update_speed_factor(params.get('speed', self.speed_factor), force=True)
        # ignore any other parameters (gamma, k, R, etc.)

    def draw_check(self, params):
        # Draw background box
        pygame.draw.rect(self.screen, self.bg_color, self.main)

        # Detect parameter changes and reset change flag per frame
        params['is_changed'] = False
        self.modified_par = None
        for i, par1, par2 in zip(range(len(self.params)), params['params'].values(), self.params.values()):
            if abs(par1 - par2) > 1e-4:
                self.modified_par = list(self.params.keys())[i]
                params['is_changed'] = True
                break

        # Advance simulation and record energies
        loader = config.ConfigLoader()
        speed_raw = params['params'].get('speed', self.speed_factor)
        self.update_speed_factor(speed_raw)
        params['params']['speed'] = self.speed_factor
        steps = self._speed_steps

        new_args = None
        for i in range(steps):
            new_args = next(self.simulation)
            self._update_wall_hit_counters()
            if i < len(params['kinetic']):
                params['kinetic'][i] = self.simulation.calc_kinetic_energy()
                params['potential'][i] = self.simulation.calc_potential_energy()
                params['mean_kinetic'][i] = self.simulation.mean_kinetic_energy(loader['sim_avg_frames_c'])
                params['mean_potential'][i] = self.simulation.mean_potential_energy(loader['sim_avg_frames_c'])
        for i in range(steps, len(params['kinetic'])):
            params['kinetic'][i] = -1
            params['potential'][i] = -1
            params['mean_kinetic'][i] = -1
            params['mean_potential'][i] = -1

        if new_args is None:
            new_args = (
                self.simulation.r,
                self.simulation.r_spring,
                self.simulation.v,
                self.simulation.v_spring,
                0.0,
            )

        # Unpack positions; r_spring is empty
        r = np.array(new_args[0], copy=True)

        # Compute instantaneous kinetic temperatures for colour scaling
        v = new_args[2] if isinstance(new_args, (list, tuple)) and len(new_args) >= 3 else self.simulation.v
        speeds_sq = np.sum(v * v, axis=0)
        masses = self.simulation.m
        if masses.shape != speeds_sq.shape:
            masses = np.broadcast_to(masses, speeds_sq.shape)
        temperatures = 0.5 * masses * speeds_sq / max(kB, 1e-12)
        normalized = self._normalize_temperatures(temperatures)

        # Compute the physical radius in box units. Simulation radius already
        # includes the physical scaling; adjust by the ratio between draw and
        # physical factors so the rendered size follows the actual collision size.
        draw_scale = self.draw_radius_factor
        phys_scale = self.physical_radius_scale if self.physical_radius_scale > 0 else 1.0
        radius_box_units = self.simulation.R * (draw_scale / phys_scale)

        # Draw coloured wall strips aligned with the outer rectangle (not the inner square)
        outer_rect = pygame.Rect(self.position[0], self.position[1], self.width, self.height)
        wall_thickness = self.wall_thickness_px
        left_wall_rect = pygame.Rect(
            outer_rect.left,
            outer_rect.top,
            wall_thickness,
            outer_rect.height,
        )
        right_wall_rect = pygame.Rect(
            outer_rect.right - wall_thickness,
            outer_rect.top,
            wall_thickness,
            outer_rect.height,
        )
        param_values = params.get('params') if isinstance(params, dict) else None
        left_temp_raw = self.simulation.T_left
        right_temp_raw = self.simulation.T_right
        if isinstance(param_values, dict):
            left_temp_raw = param_values.get('T_left', left_temp_raw)
            right_temp_raw = param_values.get('T_right', right_temp_raw)
        try:
            left_temp = float(left_temp_raw)
        except (TypeError, ValueError):
            left_temp = float(self.simulation.T_left)
        try:
            right_temp = float(right_temp_raw)
        except (TypeError, ValueError):
            right_temp = float(self.simulation.T_right)
        left_color = self._temperature_to_color(left_temp)
        right_color = self._temperature_to_color(right_temp)
        self._wall_colors['left'] = left_color
        self._wall_colors['right'] = right_color
        pygame.draw.rect(self.screen, left_color, left_wall_rect)
        pygame.draw.rect(self.screen, right_color, right_wall_rect)

        # Transform positions from simulation coordinates (box_width × 1) to screen coordinates
        box_width_units = max(self.simulation.get_box_width(), 1e-6)
        x_scale = self.sim_width_px / box_width_units
        y_scale = self.height
        r[0] = self.sim_left_px + r[0] * x_scale
        r[1] = self.position[1] + self.height - r[1] * y_scale
        r = np.round(r).astype(int)
        unit_scale = min(x_scale, y_scale)
        r_radius = max(1, int(round(unit_scale * radius_box_units)))

        # Store the current position of the tracked particle for the trail feature
        self._record_trail_point(r)
        flux_timestamp = self.simulation.get_elapsed_time()
        flux_raw = self.simulation.get_last_midplane_flux_raw()
        flux_avg = self.simulation.get_last_midplane_flux()
        cumulative_heat = self.simulation.get_cumulative_midplane_heat()
        self._store_flux_sample(flux_timestamp, flux_raw, flux_avg, cumulative_heat)

        # Precompute sets for tagged particles and highlight handling
        tagged_set = set(self.tagged_indices)

        # Draw gas particles with dimming/highlighting options
        for idx in range(r.shape[1]):
            point = (int(r[0, idx]), int(r[1, idx]))
            if idx in tagged_set:
                color = self.tagged_color
            elif self.dim_untracked:
                color = self.dim_color
            else:
                c = float(normalized[idx]) if normalized.size else 0.0
                red = int(255 * c)
                green = 0
                blue = int(255 * (1.0 - c))
                color = (red, green, blue)
            pygame.draw.circle(self.screen, color, point, r_radius)

        # Draw trajectory trail over the particles so it remains visible
        if self.trail_enabled and len(self.trail_points) >= 2:
            pygame.draw.lines(self.screen, self.tagged_color, False, self.trail_points, 2)

        if self.trail_enabled:
            self._ensure_tracked_particle()
        if self.tracked_particle_id is not None and self.tracked_particle_id < r.shape[1]:
            focus_point = (int(r[0, self.tracked_particle_id]), int(r[1, self.tracked_particle_id]))
            pygame.draw.circle(self.screen, (255, 255, 255), focus_point, r_radius + 3, 2)

        self._draw_midplane_wall()

        # Draw border
        inner_border = 3
        mask_border = 50
        pygame.draw.rect(
            self.screen,
            self.bg_screen_color,
            (
                self.position[0] - mask_border,
                self.position[1] - mask_border,
                self.width + mask_border * 2,
                self.height + mask_border * 2,
            ),
            mask_border,
        )
        pygame.draw.rect(
            self.screen,
            self.bd_color,
            (
                self.position[0] - inner_border,
                self.position[1] - inner_border,
                self.width + inner_border * 2,
                self.height + inner_border * 2,
            ),
            inner_border,
        )

    def _refresh_iter(self, params):
        if self.modified_par is not None:
            self.set_params(params['params'], self.modified_par)
            self.params[self.modified_par] = params['params'][self.modified_par]

    def _configure_flux_graph(self, loader: config.ConfigLoader) -> None:
        """Initialise fixed axis limits and grid styling for the flux chart."""
        config_dict = getattr(loader, '_loader', {})
        graph_cfg = {}
        if isinstance(config_dict, dict):
            graph_cfg = config_dict.get('flux_graph', {}) or {}

        scale_value = graph_cfg.get('display_scale', FLUX_GRAPH_DISPLAY_SCALE)
        try:
            display_scale = float(scale_value)
        except (TypeError, ValueError):
            display_scale = FLUX_GRAPH_DISPLAY_SCALE
        if not math.isfinite(display_scale) or display_scale <= 0.0:
            display_scale = 1.0
        self.flux_display_scale = display_scale

        axis_limits = graph_cfg.get('axis_limits')
        min_limit = max_limit = None
        if isinstance(axis_limits, (list, tuple)) and len(axis_limits) == 2:
            try:
                min_limit = float(axis_limits[0])
                max_limit = float(axis_limits[1])
            except (TypeError, ValueError):
                min_limit = max_limit = None
        axis_limits_specified = 'axis_limits' in graph_cfg or 'axis_limit' in graph_cfg

        needs_default = (
            min_limit is None
            or max_limit is None
            or not math.isfinite(min_limit)
            or not math.isfinite(max_limit)
            or max_limit <= min_limit
        )
        if needs_default:
            axis_limit_value = graph_cfg.get('axis_limit', FLUX_GRAPH_DEFAULT_HALF_RANGE)
            try:
                half_range = abs(float(axis_limit_value))
            except (TypeError, ValueError):
                half_range = FLUX_GRAPH_DEFAULT_HALF_RANGE
            if not math.isfinite(half_range) or half_range <= 0.0:
                half_range = FLUX_GRAPH_DEFAULT_HALF_RANGE
            min_limit = -half_range
            max_limit = half_range

        self.flux_axis_min: float = float(min_limit)
        self.flux_axis_max: float = float(max_limit)
        if self.flux_axis_max <= self.flux_axis_min:
            half_range = max(abs(self.flux_axis_min), abs(self.flux_axis_max), FLUX_GRAPH_DEFAULT_HALF_RANGE)
            self.flux_axis_min = -half_range
            self.flux_axis_max = half_range
        self.flux_axis_span: float = self.flux_axis_max - self.flux_axis_min
        if self.flux_axis_span <= 0.0:
            self.flux_axis_min = -FLUX_GRAPH_DEFAULT_HALF_RANGE
            self.flux_axis_max = FLUX_GRAPH_DEFAULT_HALF_RANGE
            self.flux_axis_span = self.flux_axis_max - self.flux_axis_min
        self.flux_axis_default_half_range: float = max(
            abs(self.flux_axis_min),
            abs(self.flux_axis_max),
            FLUX_GRAPH_DEFAULT_HALF_RANGE,
        )
        if axis_limits_specified:
            self.flux_axis_fixed_limits = (self.flux_axis_min, self.flux_axis_max)
        else:
            self.flux_axis_fixed_limits = None

        grid_lines_value = graph_cfg.get('grid_lines', FLUX_GRAPH_DEFAULT_GRID_LINES)
        try:
            grid_lines = int(grid_lines_value)
        except (TypeError, ValueError):
            grid_lines = FLUX_GRAPH_DEFAULT_GRID_LINES
        self.flux_grid_lines: int = max(2, grid_lines)

        grid_color_value = graph_cfg.get('grid_color')
        color = FLUX_GRID_COLOR
        if isinstance(grid_color_value, (list, tuple)) and len(grid_color_value) == 3:
            try:
                rgb = tuple(int(max(0, min(255, component))) for component in grid_color_value)
                if len(rgb) == 3:
                    color = rgb
            except (TypeError, ValueError):
                color = FLUX_GRID_COLOR
        self.flux_grid_color: tuple[int, int, int] = color

        follow_window_value = graph_cfg.get('follow_window_samples', FLUX_GRAPH_FOLLOW_WINDOW)
        try:
            follow_window = int(follow_window_value)
        except (TypeError, ValueError):
            follow_window = FLUX_GRAPH_FOLLOW_WINDOW
        self.flux_follow_window_samples: int = max(8, follow_window)

        padding_value = graph_cfg.get('follow_padding_fraction', FLUX_GRAPH_PADDING_FRACTION)
        try:
            padding_fraction = float(padding_value)
        except (TypeError, ValueError):
            padding_fraction = FLUX_GRAPH_PADDING_FRACTION
        if not math.isfinite(padding_fraction) or padding_fraction < 0.0:
            padding_fraction = FLUX_GRAPH_PADDING_FRACTION
        self.flux_follow_padding: float = padding_fraction

        extra_padding_value = graph_cfg.get('extra_padding_fraction', FLUX_GRAPH_EXTRA_PADDING)
        try:
            extra_padding_fraction = float(extra_padding_value)
        except (TypeError, ValueError):
            extra_padding_fraction = FLUX_GRAPH_EXTRA_PADDING
        if not math.isfinite(extra_padding_fraction) or extra_padding_fraction < 0.0:
            extra_padding_fraction = FLUX_GRAPH_EXTRA_PADDING
        self.flux_axis_extra_padding: float = extra_padding_fraction

        tick_step_value = None
        tick_step_in_config = 'tick_step' in graph_cfg
        if tick_step_in_config:
            tick_step_value = graph_cfg.get('tick_step')
            try:
                tick_step_value = float(tick_step_value)
            except (TypeError, ValueError):
                tick_step_value = None
            if tick_step_value is not None and (not math.isfinite(tick_step_value) or tick_step_value <= 0.0):
                tick_step_value = None
        # Only store a fixed tick-step if the user asked for it; otherwise pick it dynamically per frame.
        self.flux_tick_step: float | None = tick_step_value
        window_value = graph_cfg.get('time_window_seconds', FLUX_GRAPH_TIME_WINDOW)
        try:
            window_seconds = float(window_value)
        except (TypeError, ValueError):
            window_seconds = FLUX_GRAPH_TIME_WINDOW
        if not math.isfinite(window_seconds) or window_seconds <= 0.0:
            window_seconds = FLUX_GRAPH_TIME_WINDOW
        self.flux_time_window: float = window_seconds

    def _flux_zoom_factor(self) -> float:
        """Return a multiplier (<1 zooms in) based on current particle count."""
        count = None
        if getattr(self, 'params', None):
            count = self.params.get('r')
        if count is None and getattr(self, 'simulation', None):
            count = getattr(self.simulation, '_n_particles', None)
        try:
            count = int(count)
        except (TypeError, ValueError):
            return 1.0
        if count <= 0:
            return 1.0
        ref = FLUX_GRAPH_PARTICLE_REF
        min_zoom = FLUX_GRAPH_MIN_ZOOM
        zoom = math.sqrt(count / ref) if ref > 0 else 1.0
        return min(1.0, max(min_zoom, zoom))

    def _compute_dynamic_flux_limits(self, values: list[float]) -> tuple[float, float]:
        """Return adaptive axis limits that track the recent flux signal."""
        zoom = self._flux_zoom_factor()
        base_half_range = getattr(self, 'flux_axis_default_half_range', FLUX_GRAPH_DEFAULT_HALF_RANGE) * zoom
        try:
            follow_window = int(getattr(self, 'flux_follow_window_samples', FLUX_GRAPH_FOLLOW_WINDOW))
        except (TypeError, ValueError):
            follow_window = FLUX_GRAPH_FOLLOW_WINDOW
        follow_window = max(8, follow_window)
        subset = values[-follow_window:] if len(values) > follow_window else list(values)
        if not subset:
            return -base_half_range, base_half_range

        local_min = min(subset)
        local_max = max(subset)
        if not (math.isfinite(local_min) and math.isfinite(local_max)):
            return -base_half_range, base_half_range
        if local_max < local_min:
            local_min, local_max = local_max, local_min

        grid_lines = max(2, getattr(self, 'flux_grid_lines', FLUX_GRAPH_DEFAULT_GRID_LINES))
        configured_step = getattr(self, 'flux_tick_step', None)
        if configured_step is not None and math.isfinite(configured_step) and configured_step > 0.0:
            step = configured_step
        else:
            span_from_data = local_max - local_min
            step = span_from_data / grid_lines if grid_lines else span_from_data
            if not math.isfinite(step) or step <= 0.0:
                step = base_half_range / max(1, grid_lines)

        padding_fraction = getattr(self, 'flux_follow_padding', FLUX_GRAPH_PADDING_FRACTION)
        if not math.isfinite(padding_fraction) or padding_fraction < 0.0:
            padding_fraction = FLUX_GRAPH_PADDING_FRACTION

        span = max(local_max - local_min, 0.0)
        padding = span * padding_fraction
        if step > 0.0:
            padding = max(padding, step * 0.5)
        elif padding <= 0.0:
            padding = base_half_range * 0.1

        desired_min = local_min - padding
        desired_max = local_max + padding
        if not math.isfinite(desired_min) or not math.isfinite(desired_max):
            return -base_half_range, base_half_range
        if desired_max <= desired_min:
            epsilon = step if step > 0.0 else base_half_range
            desired_min -= epsilon
            desired_max += epsilon

        def align_down(value: float) -> float:
            if step <= 0.0:
                return value
            return math.floor(value / step) * step

        def align_up(value: float) -> float:
            if step <= 0.0:
                return value
            return math.ceil(value / step) * step

        axis_min = align_down(desired_min)
        axis_max = align_up(desired_max)
        try:
            negative_fraction = float(FLUX_GRAPH_NEGATIVE_LIMIT_FRACTION)
        except (TypeError, ValueError):
            negative_fraction = FLUX_GRAPH_NEGATIVE_LIMIT_FRACTION
        if not math.isfinite(negative_fraction) or negative_fraction < 0.0:
            negative_fraction = FLUX_GRAPH_NEGATIVE_LIMIT_FRACTION
        if axis_max > 0.0:
            allowed_negative = -axis_max * negative_fraction
            if local_min >= allowed_negative:
                axis_min = max(axis_min, allowed_negative)

        extra_fraction = getattr(self, 'flux_axis_extra_padding', FLUX_GRAPH_EXTRA_PADDING)
        try:
            extra_fraction = float(extra_fraction)
        except (TypeError, ValueError):
            extra_fraction = FLUX_GRAPH_EXTRA_PADDING
        if not math.isfinite(extra_fraction) or extra_fraction < 0.0:
            extra_fraction = FLUX_GRAPH_EXTRA_PADDING
        axis_span = axis_max - axis_min
        extra_margin = 0.0
        if math.isfinite(axis_span) and axis_span > 0.0:
            extra_margin = axis_span * extra_fraction
            if step > 0.0:
                extra_margin = max(extra_margin, step * 0.5)
        if extra_margin > 0.0 and math.isfinite(extra_margin):
            axis_min -= extra_margin
            axis_max += extra_margin

        # Ensure the zero level is always visible on the vertical axis.
        if axis_max < 0.0:
            axis_max = 0.0
        if axis_min > 0.0:
            axis_min = 0.0
        if axis_max <= axis_min:
            epsilon = step if step > 0.0 else base_half_range
            axis_min -= epsilon
            axis_max += epsilon

        if not math.isfinite(axis_min) or not math.isfinite(axis_max) or axis_max <= axis_min:
            half_range = base_half_range
            axis_min = -half_range
            axis_max = half_range
        return axis_min, axis_max

    def draw_midplane_flux_graph(
        self,
        target_surface: pygame.Surface,
        rect: pygame.Rect,
        samples: Optional[list[tuple[float, ...]]] = None,
        line_color: tuple[int, int, int] = (90, 180, 255),
        baseline_color: tuple[int, int, int] = (180, 180, 180),
        background: Optional[tuple[int, int, int, int]] = (0, 0, 0, 0),
        series: str = 'avg',
        draw_highlight_lines: bool = True,
    ) -> Optional[tuple[float, float]]:
        """
        Render a step-style graph of the recorded midplane heat quantities.

        The function draws onto ``target_surface`` but does not blit the
        result to the on-screen display.  This lets callers integrate the
        graph into their own layout later.

        Returns
        -------
        Optional[Tuple[float, float]]
            The ``(min, max)`` flux values plotted on the vertical axis.
        """
        self.last_flux_limits = None
        data = self.midplane_flux_samples if samples is None else samples
        rect = pygame.Rect(rect)
        if len(data) < 2 or rect.width <= 1 or rect.height <= 1:
            return None

        original_series = series
        series_key = series.lower().strip()
        idx_map = {
            'raw': 1,
            'raw_flux': 1,
            'avg': 2,
            'average': 2,
            'flux': 2,
            'cumulative': 3,
            'heat': 3,
            'total': 3,
            'heat_total': 3,
            'cumulative_heat': 3,
        }
        idx = idx_map.get(series_key, idx_map.get(original_series, 2))
        scale = getattr(self, 'flux_display_scale', 1.0)
        try:
            scale = float(scale)
        except (TypeError, ValueError):
            scale = 1.0
        if not math.isfinite(scale) or scale <= 0.0:
            scale = 1.0

        times: list[float] = []
        values: list[float] = []
        for sample in data:
            if not sample:
                continue
            t = float(sample[0])
            value_idx = min(idx, len(sample) - 1)
            if value_idx <= 0 or value_idx >= len(sample):
                continue
            val = float(sample[value_idx]) * scale
            times.append(t)
            values.append(val)

        if len(times) < 2:
            return None

        window_span = getattr(self, 'flux_time_window', FLUX_GRAPH_TIME_WINDOW)
        try:
            window_span = float(window_span)
        except (TypeError, ValueError):
            window_span = FLUX_GRAPH_TIME_WINDOW
        if not math.isfinite(window_span) or window_span <= 0.0:
            window_span = FLUX_GRAPH_TIME_WINDOW

        latest_time = times[-1]
        window_start = latest_time - window_span
        filtered_times: list[float] = []
        filtered_values: list[float] = []
        prev_time = None
        prev_value = None
        for t, v in zip(times, values):
            if t < window_start:
                prev_time = t
                prev_value = v
                continue
            if not filtered_times and prev_time is not None and prev_time < window_start and t > window_start:
                filtered_times.append(window_start)
                filtered_values.append(prev_value)
            filtered_times.append(t)
            filtered_values.append(v)

        if len(filtered_times) < 2:
            return None

        if filtered_times and filtered_times[0] > window_start:
            window_start = filtered_times[0]

        times = [t - window_start for t in filtered_times]
        values = filtered_values

        t_min = 0.0
        t_max = times[-1]
        if not math.isfinite(t_max) or t_max <= t_min:
            return None

        actual_min = min(values)
        actual_max = max(values)
        if not (math.isfinite(actual_min) and math.isfinite(actual_max)):
            return None

        time_span = t_max - t_min
        if time_span <= 0.0 or not math.isfinite(time_span):
            return None

        fixed_limits = getattr(self, 'flux_axis_fixed_limits', None)
        axis_min = axis_max = None
        if isinstance(fixed_limits, (list, tuple)) and len(fixed_limits) == 2:
            try:
                axis_min = float(fixed_limits[0])
                axis_max = float(fixed_limits[1])
            except (TypeError, ValueError):
                axis_min = axis_max = None
        if (
            axis_min is None
            or axis_max is None
            or not (math.isfinite(axis_min) and math.isfinite(axis_max))
            or axis_max <= axis_min
        ):
            axis_min, axis_max = self._compute_dynamic_flux_limits(values)

        if axis_max < 0.0:
            axis_max = 0.0
        if axis_min > 0.0:
            axis_min = 0.0

        # Keep the recent extrema visible even if fixed limits were set.
        span_for_padding = max(1.0, abs(axis_max - axis_min))
        pad = span_for_padding * 0.02
        axis_min = min(axis_min, actual_min - pad)
        axis_max = max(axis_max, actual_max + pad)

        if axis_max <= axis_min:
            spread = max(1.0, abs(actual_min), abs(actual_max))
            axis_min = -spread
            axis_max = spread

        axis_span = axis_max - axis_min
        if not math.isfinite(axis_span) or axis_span <= 0.0:
            half_range = getattr(self, 'flux_axis_default_half_range', FLUX_GRAPH_DEFAULT_HALF_RANGE)
            axis_min = -half_range
            axis_max = half_range
            axis_span = axis_max - axis_min

        self.last_flux_limits = (axis_min, axis_max)

        graph_surface = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
        if background is not None:
            graph_surface.fill(background)

        grid_lines = max(2, getattr(self, 'flux_grid_lines', FLUX_GRAPH_DEFAULT_GRID_LINES))
        grid_color = getattr(self, 'flux_grid_color', FLUX_GRID_COLOR)
        for idx in range(grid_lines + 1):
            frac = idx / grid_lines
            value = axis_min + axis_span * frac
            y_rel = (value - axis_min) / axis_span
            y = rect.height - y_rel * rect.height
            pygame.draw.line(
                graph_surface,
                grid_color,
                (0, int(round(y))),
                (rect.width, int(round(y))),
                1,
            )

        def value_to_y(value: float) -> int:
            y_rel = (value - axis_min) / axis_span
            return int(round(rect.height - y_rel * rect.height))

        if draw_highlight_lines:
            highlight_color = FLUX_HIGHLIGHT_COLOR
            window_min = min(values)
            window_max = max(values)
            min_y = value_to_y(window_min)
            max_y = value_to_y(window_max)
            pygame.draw.line(
                graph_surface,
                highlight_color,
                (0, min_y),
                (rect.width, min_y),
                FLUX_HIGHLIGHT_WIDTH,
            )
            if window_max != window_min:
                pygame.draw.line(
                    graph_surface,
                    highlight_color,
                    (0, max_y),
                    (rect.width, max_y),
                    FLUX_HIGHLIGHT_WIDTH,
                )

        if axis_min <= 0.0 <= axis_max:
            zero_y = value_to_y(0.0)
            zero_color = (0, 0, 0)
            pygame.draw.line(
                graph_surface,
                zero_color,
                (0, zero_y),
                (rect.width, zero_y),
                3,
            )

        def clamp_value(value: float) -> float:
            if not math.isfinite(value):
                return max(axis_min, min(axis_max, 0.0))
            return max(axis_min, min(axis_max, value))

        def to_point(time_value: float, flux_value: float) -> tuple[int, int]:
            x_rel = (time_value - t_min) / time_span
            y_rel = (flux_value - axis_min) / axis_span
            x = x_rel * rect.width
            y = rect.height - y_rel * rect.height
            return int(round(max(0.0, min(rect.width, x)))), int(round(max(0.0, min(rect.height, y))))

        points: list[tuple[int, int]] = []
        prev_time = times[0]
        prev_val = clamp_value(values[0])
        points.append(to_point(prev_time, prev_val))
        for time_value, flux_value in zip(times[1:], values[1:]):
            clamped = clamp_value(flux_value)
            points.append(to_point(time_value, prev_val))
            points.append(to_point(time_value, clamped))
            prev_time = time_value
            prev_val = clamped

        if len(points) >= 2:
            pygame.draw.lines(graph_surface, line_color, False, points, 2)

        target_surface.blit(graph_surface, rect.topleft)
        return self.last_flux_limits

    # -----------------------------------------------------------------
    def add_tagged_particles(self, count: int) -> int:
        """
        Add a specified number of tagged (coloured) particles at the
        centre of the box.

        The new particles are placed at the centre of the simulation
        domain (``x = 0.5``, ``y = 0.5``) with random jitter to avoid
        immediate overlap.  Their velocities are sampled from the
        current gas temperature distribution.  The indices of these
        particles are recorded in ``self.tagged_indices`` so that they
        can be drawn in a different colour.

        Parameters
        ----------
        count : int
            Number of particles to add.

        Returns
        -------
        int
            Actual number of tagged particles appended to the simulation.
        """
        count = max(0, int(count))
        if count <= 0:
            return 0
        # Determine the starting index for new particles
        n_old = int(self.simulation._n_particles)
        self.reset_wall_hit_counters()
        # Build positions at the box centre with small random jitter
        jitter = 0.001  # small displacement to avoid stacking
        # Uniformly distribute jitter within a tiny square around centre
        box_width = max(self.simulation.get_box_width(), 1e-9)
        center_x = box_width / 2.0
        r_new = np.tile(np.array([[center_x], [0.5]]), (1, count))
        if count > 0:
            r_new = r_new + (np.random.uniform(low=-jitter, high=jitter, size=(2, count)))
        # Ensure the new particles respect the walls (stay within [R, 1-R])
        # Clip in case jitter pushes them outside
        R_phys = self.simulation.R
        box_width = max(self.simulation.get_box_width(), 1e-9)
        r_new[0] = np.clip(r_new[0], R_phys, box_width - R_phys)
        r_new[1] = np.clip(r_new[1], R_phys, 1.0 - R_phys)
        # Sample velocities consistent with current temperature
        # Compute standard deviation for Maxwell distribution using scaled k_boltz
        # Use median mass in case of varying masses
        masses = self.simulation.m
        if masses.size > 0:
            m_typ = np.median(masses)
        else:
            m_typ = 1.0
        sigma = np.sqrt(self.simulation._k_boltz * self.simulation.T / m_typ)
        v_new = np.random.normal(loc=0.0, scale=sigma, size=(2, count))
        # Masses for new particles
        m_new = np.full((count,), m_typ)
        # Add to simulation
        self.simulation.add_particles(r_new, v_new, m_new)
        n_new = int(self.simulation._n_particles)
        actual_added = max(0, n_new - n_old)
        if actual_added <= 0:
            return 0
        # Record tagged indices
        new_indices = list(range(n_old, n_old + actual_added))
        self.tagged_indices.extend(new_indices)
        self._ensure_tracked_particle()
        if self.trail_enabled:
            self.trail_points.clear()
        # Update parameter dictionary to reflect increased number of particles
        self.params['r'] = n_new
        return actual_added
