from __future__ import annotations

import math
from collections import OrderedDict
from typing import Any, List, Tuple

import pygame

import language
from button import Button
from slider import ParamSlider
from demo import Demo, SPEED_MIN, SPEED_MAX, MIDPLANE_WALL_THICKNESS_RATIO
import config
from onboarding import OnboardingGuide
from ui_base import ResponsiveScreen, get_font, build_vertical_gradient, calc_scale

MIN_TOTAL_PARTICLES = 5
MAX_TOTAL_PARTICLES = 500


class DemoScreen(ResponsiveScreen):
    def __init__(self, app):
        super().__init__(app)
        self.lang = language.Language()
        self.primary_color = (72, 104, 255)
        self.bg_color = (234, 236, 246)
        self.panel_color = (248, 249, 253)
        self.border_color = (214, 220, 235)
        self.shadow_color = (18, 24, 60, 60)

        self.counter_base_font_size = 30
        self.panel_title_base_font_size = 24
        self.placeholder_base_font_size = 20
        self.conc_label_base_font_size = 15
        self.conc_value_base_font_size = 22
        self.graph_value_base_font_size = 24
        self.graph_counts_base_font_size = 18
        self.counter_font = get_font(self.counter_base_font_size, bold=True)
        self.panel_title_font = get_font(self.panel_title_base_font_size, bold=True)
        self.placeholder_font = get_font(self.placeholder_base_font_size)
        self.conc_label_font = get_font(self.conc_label_base_font_size, bold=True)
        self.conc_value_font = get_font(self.conc_value_base_font_size, bold=True)
        self.graph_value_font = get_font(self.graph_value_base_font_size)
        self.graph_counts_font = get_font(self.graph_counts_base_font_size, bold=True)
        self.graph_max_samples = 600
        self.layout_scale = 1.0
        self.badge_section_height = 0
        self._badge_layout_cache: dict[str, Any] | None = None
        self._button_metrics: dict[str, int] | None = None
        self._slider_card_metrics: dict[str, int] | None = None

        self.slider_panel_rect: pygame.Rect | None = None
        self.right_panel_rect: pygame.Rect | None = None
        self.sim_rect: pygame.Rect | None = None

        self.background: pygame.Surface | None = None

        self.buttons: List[Button] = []
        self.dim_button: Button | None = None
        self.trail_button: Button | None = None
        self.add_button: Button | None = None
        self.apply_button: Button | None = None
        self.reset_button: Button | None = None
        self.menu_button: Button | None = None

        self.graph_value_template: str = ''
        self.graph_counts_template: str = ''

        self._localize_strings()
        self._build_button_layout()

        self.sliders: List[ParamSlider] = []
        self.slider_param_keys: List[str] = []
        self.slider_speed = None
        self.slider_T_left = None
        self.slider_T_right = None
        self.slider_size_scale = None
        self.slider_tag_count = None
        self.slider_particle_count = None
        self.slider_lookup: dict[str, ParamSlider] = {}
        self._last_speed_value: float | None = None

        self.demo: Demo | None = None
        self.demo_config = {}
        self.dim_active = False
        self.trail_active = False
        self.slider_grabbed = False
        self.tagged_count = 10
        self._left_wall_bounds: tuple[float, float] = (500.0, 2000.0)
        self._right_wall_bounds: tuple[float, float] = (100.0, 500.0)
        self._global_wall_bounds: tuple[float, float] = (100.0, 2000.0)

        (
            self.slider_definitions,
            original_bounds,
            self.initial_params,
        ) = self._build_slider_definitions()

        frames_count = original_bounds[-1][1] if original_bounds else 10
        self.demo_config = {
            'params': self.initial_params.copy(),
            'kinetic': [0] * frames_count,
            'mean_kinetic': [0] * frames_count,
            'potential': [0] * frames_count,
            'mean_potential': [0] * frames_count,
            'is_changed': False,
        }
        self.tagged_count = int(self.initial_params.get('tagged_count', 10))
        self._last_wall_temperatures = {'left': None, 'right': None}
        self._last_particle_count = int(self.initial_params.get('r', 0))
        self._last_flux_graph_rect: pygame.Rect | None = None

        self.graph_border_toggle_label: str = ''
        self._flux_border_enabled = False
        self._flux_border_toggle_rect: pygame.Rect | None = None

        self.onboarding = OnboardingGuide(app)
        self._onboarding_started = False
        self._pending_onboarding = True

        self.background = build_vertical_gradient(self.app.window_size, (230, 236, 255), (246, 248, 254))
        self._relayout(self.app.window_size)

    def on_language_change(self) -> None:
        self.lang = language.Language()
        self._localize_strings()
        self._build_button_layout()
        current_values = self.demo_config.get('params', {}).copy()
        self.slider_definitions, _, self.initial_params = self._build_slider_definitions()
        for key, value in current_values.items():
            if key in self.initial_params:
                self.initial_params[key] = value
        self.demo_config['params'].update(self.initial_params)
        self.tagged_count = int(self.demo_config['params'].get('tagged_count', self.tagged_count))
        self._last_particle_count = int(self.demo_config['params'].get('r', self._last_particle_count))
        self._relayout(self.app.window_size)
        if self.dim_button is not None:
            dim_label = self.dim_labels[1] if self.dim_active else self.dim_labels[0]
            self.dim_button._prep_msg(dim_label)
        if self.trail_button is not None:
            trail_label = self.trail_labels[1] if self.trail_active else self.trail_labels[0]
            self.trail_button._prep_msg(trail_label)
        if self.reset_button is not None:
            self.reset_button._prep_msg(self.reset_label)
        self._refresh_onboarding_locale()

    def _build_button_layout(self) -> None:
        apply_label = self.lang['btn_apply']
        menu_label = self.lang['btn_menu']
        # Arrange buttons so primary actions stay together and occupy minimal space.
        self.button_layout = [
            {'label': self.label_add_particles, 'handler': self.add_marked_particles, 'primary': True, 'attr': 'add_button'},
            {'label': apply_label, 'handler': self.apply, 'primary': True, 'attr': 'apply_button'},
            {'label': self.dim_labels[0], 'handler': self.toggle_dim_particles, 'id': 'dim'},
            {'label': self.trail_labels[0], 'handler': self.toggle_trail, 'id': 'trail'},
            {'label': self.reset_label, 'handler': self.reset_measurements, 'attr': 'reset_button'},
            {'label': menu_label, 'handler': self.to_menu, 'attr': 'menu_button'},
        ]

    def _localize_strings(self) -> None:
        def _fallback(key: str, default: str) -> str:
            try:
                return self.lang[key]
            except KeyError:
                return default

        if self.lang.lang == 'rus':
            default_add = 'Добавить частицы'
            dim_pair = ('Затусклить фон', 'Вернуть цвета')
            trail_pair = ('Показать след', 'Скрыть след')
            panel = 'Параметры'
            graphs = 'Поток через середину области'
            placeholder = 'Недостаточно данных для графика'
            counter_left = 'Левая'
            counter_right = 'Правая'
            counts_default = 'Q: {cumulative_display} усл. ед.'
            flux_default = 'Поток: {flux_display} усл. ед./с'
            scale_default = 'Диапазон по оси: [{min}; {max}] усл. ед./с'
            reset_default = 'Сброс'
        else:
            default_add = 'Add particles'
            dim_pair = ('Dim background', 'Restore colors')
            trail_pair = ('Show trail', 'Hide trail')
            panel = 'Parameters'
            graphs = 'Midplane heat flux'
            placeholder = 'Not enough data yet'
            counter_left = 'Left'
            counter_right = 'Right'
            counts_default = 'Q: {cumulative_display} units'
            flux_default = 'Flux: {flux_display} units/s'
            reset_default = 'Reset'

        add_label = _fallback('btn_add_particles', default_add)
        if self.lang.lang == 'rus':
            add_label = 'Добавить частицы'
        elif self.lang.lang == 'eng':
            add_label = 'Add particles'
        self.label_add_particles = add_label
        self.reset_label = _fallback('btn_reset_counters', reset_default)
        dim_enable = _fallback('btn_dim_enable', dim_pair[0])
        dim_disable = _fallback('btn_dim_disable', dim_pair[1])
        trail_enable = _fallback('btn_trail_enable', trail_pair[0])
        trail_disable = _fallback('btn_trail_disable', trail_pair[1])
        self.dim_labels = (dim_enable, dim_disable)
        self.trail_labels = (trail_enable, trail_disable)
        self.panel_title_text = _fallback('panel_title_parameters', panel)
        self.graph_title_text = _fallback('panel_title_charts', graphs)
        self.placeholder_text = _fallback('charts_placeholder', placeholder)
        value_template = ''
        counts_template = _fallback('flux_counts_label', counts_default)
        # Remove ΔQ segment if present in translation to avoid clutter
        counts_template = counts_template.replace('(ΔQ {step_display})', '').replace('(ΔQ {step})', '')
        counts_template = counts_template.replace('(ΔQ {step_display})'.replace(' ', ''), '')
        counts_template = counts_template.strip()
        if counts_template.endswith('()'):
            counts_template = counts_template[:-2].rstrip()
        if '{cumulative' not in counts_template:
            counts_template = counts_default
        flux_template = ''
        self.graph_value_template = value_template
        self.graph_counts_template = counts_template
        self.graph_flux_template = flux_template
        axis_units_default = 'Усл. ед.' if self.lang.lang == 'rus' else 'Units'
        self.axis_units_label = _fallback('flux_axis_units', axis_units_default)
        border_toggle_default = 'Границы графика' if self.lang.lang == 'rus' else 'Graph borders'
        self.graph_border_toggle_label = _fallback('graph_border_toggle', border_toggle_default)
        left_conc_default = 'Слева: {value:.2f}' if self.lang.lang == 'rus' else 'Left: {value:.2f}'
        right_conc_default = 'Справа: {value:.2f}' if self.lang.lang == 'rus' else 'Right: {value:.2f}'
        self.conc_left_template = _fallback('label_concentration_left', left_conc_default)
        self.conc_right_template = _fallback('label_concentration_right', right_conc_default)
        if self.lang.lang == 'rus':
            self.conc_side_labels = {'left': 'Слева', 'right': 'Справа'}
            self.conc_unit_label = 'ч/ед²'
        else:
            self.conc_side_labels = {'left': 'Left', 'right': 'Right'}
            # ASCII-friendly unit to avoid missing glyphs in some fonts
            self.conc_unit_label = 'part/unit^2'
        default_left_template = f'{counter_left}: {{count}}'
        default_right_template = f'{counter_right}: {{count}}'
        left_value = _fallback('counter_left', default_left_template)
        right_value = _fallback('counter_right', default_right_template)
        self.counter_left_template = left_value if '{count}' in left_value else f'{left_value}: {{count}}'
        self.counter_right_template = right_value if '{count}' in right_value else f'{right_value}: {{count}}'

    # ------------------------------------------------------------------ Layout
    def _update_fonts(self, scale: float) -> None:
        s = max(0.55, scale)
        self.counter_font = get_font(max(16, int(self.counter_base_font_size * s)), bold=True)
        self.panel_title_font = get_font(max(16, int(self.panel_title_base_font_size * s)), bold=True)
        self.placeholder_font = get_font(max(12, int(self.placeholder_base_font_size * s)))
        self.conc_label_font = get_font(max(10, int(self.conc_label_base_font_size * s)), bold=True)
        self.conc_value_font = get_font(max(14, int(self.conc_value_base_font_size * s)), bold=True)
        self.graph_value_font = get_font(max(13, int(self.graph_value_base_font_size * s)))
        self.graph_counts_font = get_font(max(12, int(self.graph_counts_base_font_size * s)), bold=True)
        self._badge_layout_cache = None

    def _compute_button_metrics(self, panel_width: int) -> dict[str, int]:
        scale = getattr(self, 'layout_scale', 1.0)
        inner_padding = max(int(10 * scale), panel_width // 42)
        gap = max(3, min(12, int(inner_padding // 2)))
        height = max(int(30 * scale), min(int(44 * scale), int(panel_width * 0.06)))
        font_size = max(12, int(height * 0.33))
        return {'inner_padding': inner_padding, 'gap': gap, 'height': height, 'font_size': font_size}

    def _compute_slider_card_metrics(self) -> dict[str, int]:
        scale = getattr(self, 'layout_scale', 1.0)
        padding = max(4, int(6 * scale))
        label_size = max(11, int(14 * scale))
        value_size = max(13, int(18 * scale))
        track_height = max(4, int(5 * scale))
        label_height = get_font(label_size, bold=True).get_height()
        value_height = get_font(value_size, bold=True).get_height()
        knob_radius = max(8, int(track_height * 1.8))
        label_gap = max(2, padding // 2)
        value_gap = max(3, padding // 2)
        min_height = int(
            2 * padding
            + label_height
            + label_gap
            + value_height
            + value_gap
            + track_height * 0.4
            + knob_radius
            + 4
        )
        min_height = max(min_height, max(44, int(54 * scale)))
        max_height = max(min_height, max(78, int(88 * scale)))
        return {
            'padding': padding,
            'label_size': label_size,
            'value_size': value_size,
            'track_height': track_height,
            'min_height': min_height,
            'max_height': max_height,
        }

    def _measure_slider_panel_height(self, panel_width: int) -> int:
        scale = getattr(self, 'layout_scale', 1.0)
        slider_metrics = self._slider_card_metrics or self._compute_slider_card_metrics()
        button_metrics = self._button_metrics or self._compute_button_metrics(panel_width)
        inner_padding = max(int(14 * scale), panel_width // 34)
        gap = max(int(8 * scale), inner_padding // 3)
        badge_gap_top = max(int(8 * scale), inner_padding // 4)
        badge_gap_bottom = max(int(8 * scale), inner_padding // 4)
        title_height = self.panel_title_font.get_height()
        dummy_rect = pygame.Rect(0, 0, panel_width, 10)
        badge_height = self._estimate_badge_section_height(dummy_rect)

        total_sliders = len(self.slider_definitions)
        grid_height = 0
        if total_sliders > 0:
            max_columns = min(3, total_sliders)
            min_card_width = max(150, int(170 * scale))
            chosen_columns = 1
            for columns in range(max_columns, 0, -1):
                width_available = panel_width - 2 * inner_padding - gap * (columns - 1)
                if width_available <= 0:
                    continue
                card_width = width_available / columns
                if card_width >= min_card_width:
                    chosen_columns = columns
                    break
            rows = math.ceil(total_sliders / chosen_columns)
            grid_height = rows * slider_metrics['min_height'] + max(0, (rows - 1) * gap)
        slider_block = (
            inner_padding
            + title_height
            + badge_gap_top
            + badge_height
            + badge_gap_bottom
            + grid_height
        )
        if total_sliders:
            slider_block += gap

        primary_row = [spec for spec in self.button_layout if spec.get('primary')]
        secondary_row = [spec for spec in self.button_layout if not spec.get('primary')]
        if not primary_row and self.button_layout:
            primary_row = [self.button_layout[0]]
            secondary_row = self.button_layout[1:]
        rows_count = len([row for row in (primary_row, secondary_row) if row])
        buttons_height = 0
        if rows_count:
            buttons_height = (
                rows_count * button_metrics['height']
                + max(0, (rows_count - 1) * button_metrics['gap'])
            )

        total_height = slider_block + buttons_height + inner_padding
        return int(math.ceil(total_height))

    def _relayout(self, size: tuple[int, int]) -> None:
        width, height = size
        self.layout_scale = calc_scale(size, min_scale=0.6, max_scale=1.2)
        scale = self.layout_scale
        self._update_fonts(scale)
        self.background = build_vertical_gradient((width, height), (230, 236, 255), (246, 248, 254))
        self.badge_section_height = 0

        margin = max(int(8 * scale), width // 150)
        counter_padding = self.counter_font.get_height() + int(4 * scale)
        min_counter_padding = int(16 * scale)
        margin = max(margin, counter_padding, min_counter_padding)
        gap_between = max(int(8 * scale), 8)
        available_width = max(320, width - 2 * margin - gap_between)
        # Bottom split: exact halves for controls and chart
        left_panel_width = max(int(available_width * 0.5), int(320 * scale))
        right_panel_width = max(int(available_width - left_panel_width), int(280 * scale))
        if left_panel_width + right_panel_width > available_width:
            right_panel_width = available_width - left_panel_width
        self._button_metrics = self._compute_button_metrics(left_panel_width)
        self._button_metrics = self._compute_button_metrics(left_panel_width)
        self._slider_card_metrics = self._compute_slider_card_metrics()
        required_bottom = self._measure_slider_panel_height(left_panel_width)

        screen_budget = max(0, height - 2 * margin - gap_between)
        sim_height = max(int(screen_budget * 0.58), int(height * 0.5))
        bottom_height = max(required_bottom, screen_budget - sim_height)
        if sim_height + bottom_height > screen_budget:
            overflow = sim_height + bottom_height - screen_budget
            sim_height = max(int(screen_budget * 0.5), sim_height - overflow // 2)
            bottom_height = max(required_bottom, screen_budget - sim_height)

        self.sim_rect = pygame.Rect(margin, margin, width - 2 * margin, sim_height)

        bottom_top = self.sim_rect.bottom + gap_between
        self.slider_panel_rect = pygame.Rect(margin, bottom_top, left_panel_width, bottom_height)
        self.right_panel_rect = pygame.Rect(
            self.slider_panel_rect.right + gap_between,
            bottom_top,
            right_panel_width,
            bottom_height,
        )
        self.badge_section_height = self._estimate_badge_section_height(self.slider_panel_rect)

        self._build_buttons()
        self._build_sliders()
        self._ensure_demo()

    def _build_buttons(self) -> None:
        assert self.slider_panel_rect is not None
        scale = getattr(self, 'layout_scale', 1.0)
        panel = self.slider_panel_rect
        metrics = self._button_metrics or self._compute_button_metrics(panel.width)
        inner_padding = metrics['inner_padding']
        gap = metrics['gap']
        button_height = metrics['height']

        primary_row = [spec for spec in self.button_layout if spec.get('primary')]
        secondary_row = [spec for spec in self.button_layout if not spec.get('primary')]
        if not primary_row and self.button_layout:
            primary_row = [self.button_layout[0]]
            secondary_row = self.button_layout[1:]
        self.buttons = []
        self.dim_button = None
        self.trail_button = None
        self.add_button = None
        self.apply_button = None
        self.menu_button = None
        rows = [row for row in (primary_row, secondary_row) if row]
        if not rows:
            self.button_area_top = panel.bottom - inner_padding
            self.button_area_bottom = self.button_area_top
            return

        row_width_available = panel.width - 2 * inner_padding
        total_rows_height = len(rows) * button_height + (len(rows) - 1) * gap
        start_y = panel.bottom - inner_padding - total_rows_height
        self.button_area_top = start_y

        font_size = metrics['font_size']
        button_font = get_font(font_size, bold=True)
        current_y = start_y
        for row in rows:
            count = len(row)
            if count == 0:
                continue
            base_padding = max(int(24 * scale), int(button_height * 0.6))
            min_padding = max(int(12 * scale), int(button_height * 0.3))
            base_widths: List[float] = []
            min_widths: List[float] = []
            for spec in row:
                label_width = button_font.size(spec['label'])[0]
                base_width = label_width + base_padding
                min_width = label_width + min_padding
                base_width = max(base_width, int(130 * scale))
                min_width = max(min_width, int(100 * scale))
                base_widths.append(base_width)
                min_widths.append(min_width)

            available_width = max(0, row_width_available - gap * (count - 1))
            widths = list(base_widths)
            total_width = sum(widths)
            if total_width > available_width and total_width > 0:
                overflow = total_width - available_width
                reducible = sum(max(0, base - minw) for base, minw in zip(base_widths, min_widths))
                if reducible > 0:
                    reduction_ratio = min(1.0, overflow / reducible)
                    widths = [
                        base - int((base - minw) * reduction_ratio)
                        for base, minw in zip(base_widths, min_widths)
                    ]
                else:
                    shrink_ratio = max(0.65, available_width / total_width)
                    widths = [
                        max(minw, int(base * shrink_ratio))
                        for base, minw in zip(base_widths, min_widths)
                    ]
                total_width = sum(widths)
                if total_width > available_width and available_width > 0:
                    fit_ratio = max(0.45, available_width / total_width)
                    min_cap = max(int(60 * scale), 48)
                    widths = [max(int(w * fit_ratio), min_cap) for w in widths]
                    total_width = sum(widths)
                    if total_width > available_width and total_width > 0:
                        fit_ratio = max(0.35, available_width / total_width)
                        widths = [max(int(w * fit_ratio), min_cap) for w in widths]
                        total_width = sum(widths)

            extra_space = max(0.0, available_width - total_width)
            start_x = panel.left + inner_padding + extra_space / 2
            current_x = start_x

            for spec, button_width in zip(row, widths):
                is_primary = bool(spec.get('primary'))
                button = Button(
                    self.app,
                    spec['label'],
                    (int(current_x), int(current_y)),
                    (int(button_width), button_height),
                    spec['handler'],
                    font=["SF Pro Display", "Segoe UI", "Arial"],
                    fontSize=font_size,
                    bold=True,
                    button_color=self.primary_color if is_primary else (249, 250, 253),
                    text_color=(255, 255, 255) if is_primary else (35, 38, 46),
                    border_radius=18,
                    border_color=None if is_primary else self.border_color,
                    shadow_offset=7 if is_primary else 4,
                    shadow_color=(64, 99, 255, 120) if is_primary else (0, 0, 0, 38),
                )
                self.buttons.append(button)
                if spec.get('id') == 'dim':
                    self.dim_button = button
                elif spec.get('id') == 'trail':
                    self.trail_button = button
                attr_name = spec.get('attr')
                if attr_name == 'add_button':
                    self.add_button = button
                elif attr_name == 'apply_button':
                    self.apply_button = button
                elif attr_name == 'reset_button':
                    self.reset_button = button
                elif attr_name == 'menu_button':
                    self.menu_button = button
                current_x += button_width + gap

            current_y += button_height + gap

        self.button_area_bottom = current_y - gap

    def _build_sliders(self) -> None:
        assert self.slider_panel_rect is not None
        scale = getattr(self, 'layout_scale', 1.0)
        panel = self.slider_panel_rect
        slider_metrics = self._slider_card_metrics or self._compute_slider_card_metrics()
        inner_padding = max(int(8 * scale), panel.width // 42)
        gap = max(int(5 * scale), inner_padding // 3)
        title_height = self.panel_title_font.get_height()
        badge_gap_top = max(int(4 * scale), inner_padding // 6)
        badge_gap_bottom = max(int(4 * scale), inner_padding // 6)
        content_top = panel.top + inner_padding + title_height + badge_gap_top + self.badge_section_height + badge_gap_bottom
        buttons_top = getattr(self, 'button_area_top', panel.bottom - inner_padding)
        content_bottom = buttons_top - gap
        if content_bottom <= content_top:
            content_top = panel.top + inner_padding + title_height
            content_bottom = panel.bottom - inner_padding
        available_height = max(40.0, content_bottom - content_top)

        total_sliders = len(self.slider_definitions)
        if total_sliders == 0:
            self.sliders = []
            self.slider_param_keys = []
            return

        max_columns = min(3, total_sliders)
        min_card_width = max(120, int(150 * scale))
        min_card_height = slider_metrics['min_height']
        best_layout: tuple[int, int, float, float] | None = None
        fallback_layout: tuple[int, int, float, float] | None = None
        for columns in range(max_columns, 0, -1):
            width_available = panel.width - 2 * inner_padding - gap * (columns - 1)
            if width_available <= 0:
                continue
            card_width = width_available / columns
            if card_width < min_card_width:
                continue
            rows = math.ceil(total_sliders / columns)
            height_available = available_height - gap * (rows - 1)
            if height_available <= 0:
                continue
            card_height_raw = height_available / rows
            candidate = (columns, rows, card_width, card_height_raw)
            if card_height_raw >= min_card_height:
                best_layout = candidate
                break
            fallback_layout = candidate

        if best_layout is None:
            if fallback_layout is not None:
                columns, rows, card_width, card_height_raw = fallback_layout
            else:
                columns = 1
                rows = max(1, total_sliders)
                width_available = panel.width - 2 * inner_padding
                card_width = width_available if width_available > 0 else float(panel.width)
                height_available = available_height - gap * (rows - 1)
                if height_available <= 0:
                    height_available = available_height
                card_height_raw = height_available / rows if rows else float(available_height)
        else:
            columns, rows, card_width, card_height_raw = best_layout

        max_card_height = slider_metrics['max_height']
        card_height = max(min_card_height, min(card_height_raw, max_card_height))

        grid_height = rows * card_height + max(0, (rows - 1) * gap)
        extra_space = (content_bottom - content_top) - grid_height
        if extra_space > 1.0:
            max_shift = max(10.0, inner_padding * 0.4)
            content_top += min(extra_space * 0.3, max_shift)

        self.sliders = []
        self.slider_param_keys = []
        self.slider_speed = None
        self.slider_T_left = None
        self.slider_T_right = None
        self.slider_accommodation = None
        self.slider_size_scale = None
        self.slider_tag_count = None
        self.slider_particle_count = None
        self.slider_lookup = {}
        self._last_speed_value = None
        unit_map = {
            'r': {'rus': 'шт.', 'eng': 'pcs'},
            'speed': {'rus': '×', 'eng': '×'},
            'T_left': {'rus': 'K', 'eng': 'K'},
            'T_right': {'rus': 'K', 'eng': 'K'},
            'size_scale': {'rus': '×', 'eng': '×'},
            'tagged_count': {'rus': 'шт.', 'eng': 'pcs'},
        }
        lang_code = getattr(self.lang, 'lang', 'eng')
        for index, definition in enumerate(self.slider_definitions):
            row = index // columns
            col = index % columns
            x = panel.left + inner_padding + col * (card_width + gap)
            y = content_top + row * (card_height + gap)
            rect = (int(x), int(y), int(card_width), int(card_height))
            key = definition['key']
            current_raw = self.demo_config['params'].get(key, self.initial_params[key])
            current_value = self._round_value(current_raw, definition['decimals'])
            ratio = 0.0
            minimum, maximum = definition['bounds']
            if maximum > minimum:
                ratio = (current_value - minimum) / (maximum - minimum)
            suffix_entry = unit_map.get(key)
            if isinstance(suffix_entry, dict):
                suffix_value = suffix_entry.get(lang_code, next(iter(suffix_entry.values())))
            else:
                suffix_value = suffix_entry or ''
            slider = ParamSlider(
                self.app,
                definition['label'],
                rect,
                definition['bounds'],
                definition['step'],
                key,
                definition['decimals'],
                ratio,
                padding=slider_metrics['padding'],
                label_size=slider_metrics['label_size'],
                value_size=slider_metrics['value_size'],
                track_height=slider_metrics['track_height'],
                value_suffix=suffix_value,
            )
            slider.set_value(current_value)
            self.sliders.append(slider)
            self.slider_param_keys.append(key)
            self.slider_lookup[key] = slider
            self.demo_config['params'][key] = current_value
            self.initial_params[key] = current_value
            if key == 'T_left':
                self.slider_T_left = slider
            elif key == 'T_right':
                self.slider_T_right = slider
            elif key == 'accommodation':
                self.slider_accommodation = slider
            elif key == 'r':
                self.slider_particle_count = slider
            elif key == 'speed':
                self.slider_speed = slider
                self._last_speed_value = slider.getValue()
            elif key == 'size_scale':
                self.slider_size_scale = slider
            elif key == 'tagged_count':
                self.slider_tag_count = slider

    def _compute_badge_layout(self, panel_rect: pygame.Rect) -> dict[str, Any]:
        scale = getattr(self, 'layout_scale', 1.0)
        padding = max(int(8 * scale), panel_rect.width // 60)
        gap = max(int(6 * scale), panel_rect.width // 100)
        available_width = max(1, panel_rect.width - padding * 2)
        columns = 2 if available_width >= int(280 * scale) + gap else 1
        columns = max(1, columns)
        width_budget = max(1, available_width - gap * (columns - 1))
        badge_width = width_budget / columns if columns > 0 else width_budget
        badge_width = min(int(190 * scale), badge_width)
        badge_width = max(int(120 * scale), badge_width)
        if badge_width * columns > width_budget:
            badge_width = max(1, width_budget / columns if columns > 0 else width_budget)
        min_badge_height = self.conc_value_font.get_height() + self.conc_label_font.get_height() + int(12 * scale)
        badge_height = max(int(42 * scale), min_badge_height)
        badge_height = min(badge_height, int(64 * scale))
        rows = math.ceil(2 / columns)
        total_height = rows * badge_height + max(0, (rows - 1)) * gap
        return {
            'padding': padding,
            'gap': gap,
            'columns': columns,
            'rows': rows,
            'width': badge_width,
            'height': badge_height,
            'total_height': total_height,
        }

    def _estimate_badge_section_height(self, panel_rect: pygame.Rect) -> int:
        scale = getattr(self, 'layout_scale', 1.0)
        layout = self._compute_badge_layout(panel_rect)
        self._badge_layout_cache = layout
        extra_gap = max(int(4 * scale), 4)
        minimum = max(int(18 * scale), int(layout['total_height'] + extra_gap))
        return minimum

    def _ensure_demo(self) -> None:
        assert self.sim_rect is not None
        params_initial = OrderedDict()
        for key, slider in zip(self.slider_param_keys, self.sliders):
            params_initial[key] = slider.getValue()
        if 'T' in self.demo_config['params']:
            params_initial['T'] = self.demo_config['params']['T']
        elif 'T' in self.initial_params:
            params_initial['T'] = self.initial_params['T']
        self.demo_config['params'] = OrderedDict(params_initial)
        wall_bounds = getattr(self, '_global_wall_bounds', None)
        if self.demo is None:
            self.demo = Demo(
                self.app,
                (self.sim_rect.left, self.sim_rect.top),
                (self.sim_rect.width, self.sim_rect.height),
                (255, 255, 255),
                self.border_color,
                self.bg_color,
                params_initial.copy(),
                wall_temp_bounds=wall_bounds,
            )
            self.demo.set_dim_untracked(self.dim_active)
            self.demo.set_trail_enabled(self.trail_active)
        else:
            self.demo.resize_viewport((self.sim_rect.left, self.sim_rect.top), (self.sim_rect.width, self.sim_rect.height))
            self.demo.screen = self.app.screen
            if wall_bounds is not None:
                self.demo.set_wall_color_range(wall_bounds)
        if self.demo is not None:
            self._last_wall_temperatures['left'] = float(self.demo.simulation.T_left)
            self._last_wall_temperatures['right'] = float(self.demo.simulation.T_right)
            self._last_particle_count = int(self.demo.params.get('r', self._last_particle_count))

    # ------------------------------------------------------------------ Slider data
    def _round_value(self, value: float, decimals: int) -> float:
        return int(round(value, 0)) if decimals == 0 else round(value, decimals)

    def _build_slider_definitions(self):
        param_names, _, _, param_bounds, param_initial, param_step, par4sim, dec_numbers = self._load_params()
        original_bounds = list(param_bounds)

        entries = list(zip(param_names, param_bounds, param_initial, param_step, par4sim, dec_numbers))
        ignore_params = {'gamma', 'k', 'm_spring', 'R', 'R_spring', 'T'}
        base_initial_values: OrderedDict[str, float] = OrderedDict()
        base_decimals: dict[str, int] = {}
        slider_defs: list[dict] = []
        for name, bounds, initial_value, step, name_par, dec_number in entries:
            if name_par == 'speed':
                bounds = (SPEED_MIN, SPEED_MAX)
                min_val_override, max_val_override = bounds
                step = 0.01
                dec_number = 2
                initial_value = max(min_val_override, min(max_val_override, initial_value))
            if name_par == 'T':
                step = 5
                dec_number = 0
            if name_par == 'r':
                step = 5
                dec_number = 0
                min_bound, max_bound = bounds
                min_bound = max(MIN_TOTAL_PARTICLES, min_bound)
                max_bound = min(MAX_TOTAL_PARTICLES, max_bound)
                if max_bound < min_bound:
                    max_bound = min_bound
                bounds = (min_bound, max_bound)
            min_val, max_val = bounds
            if max_val <= min_val:
                clamped_value = min_val
                ratio = 0.0
            else:
                clamped_value = max(min_val, min(max_val, initial_value))
                ratio = (clamped_value - min_val) / (max_val - min_val)
            base_initial_values[name_par] = clamped_value
            base_decimals[name_par] = dec_number
            if name_par in ignore_params:
                continue
            slider_defs.append(
                {
                    'label': name,
                    'bounds': bounds,
                    'initial_pos': ratio,
                    'step': step,
                    'key': name_par,
                    'decimals': dec_number,
                }
            )

        # Additional sliders
        default_left_bounds = (500.0, 2000.0)
        default_right_bounds = (100.0, 500.0)
        default_left_initial = 500.0
        default_right_initial = 500.0
        try:
            wall_section = config.ConfigLoader()['wall_temperatures']
        except (KeyError, TypeError):
            wall_section = {}
        if not isinstance(wall_section, dict):
            wall_section = {}

        def _wall_values(entry, default_value, default_bounds):
            bounds = default_bounds
            value = default_value
            if isinstance(entry, dict):
                maybe_bounds = entry.get('bounds', bounds)
                if (
                    isinstance(maybe_bounds, (list, tuple))
                    and len(maybe_bounds) == 2
                    and all(isinstance(v, (int, float)) for v in maybe_bounds)
                ):
                    bounds = (float(maybe_bounds[0]), float(maybe_bounds[1]))
                maybe_value = entry.get('initial', entry.get('value', value))
                if isinstance(maybe_value, (int, float)):
                    value = float(maybe_value)
            elif isinstance(entry, (int, float)):
                value = float(entry)
            return value, bounds

        left_entry = wall_section.get('left') if wall_section else None
        if left_entry is None and isinstance(wall_section, dict):
            left_entry = wall_section.get('T_left')
        right_entry = wall_section.get('right') if wall_section else None
        if right_entry is None and isinstance(wall_section, dict):
            right_entry = wall_section.get('T_right')
        if right_entry is None:
            right_entry = left_entry

        initial_T_left, left_bounds = _wall_values(left_entry, default_left_initial, default_left_bounds)
        initial_T_right, right_bounds = _wall_values(right_entry, default_right_initial, default_right_bounds)
        self._left_wall_bounds = left_bounds
        self._right_wall_bounds = right_bounds
        self._global_wall_bounds = (
            float(min(left_bounds[0], right_bounds[0])),
            float(max(left_bounds[1], right_bounds[1])),
        )
        size_bounds = (0.5, 1.7)
        size_initial = 1.3
        tag_bounds = (1, 100)
        tag_initial = 5

        slider_defs.extend(
            [
                {
                    'label': self.lang['slider_T_left'],
                    'bounds': left_bounds,
                    'initial_pos': (initial_T_left - left_bounds[0]) / (left_bounds[1] - left_bounds[0]),
                    'step': max(5.0, (left_bounds[1] - left_bounds[0]) / 100.0),
                    'key': 'T_left',
                    'decimals': 0,
                },
                {
                    'label': self.lang['slider_T_right'],
                    'bounds': right_bounds,
                    'initial_pos': (initial_T_right - right_bounds[0]) / (right_bounds[1] - right_bounds[0]),
                    'step': max(5.0, (right_bounds[1] - right_bounds[0]) / 100.0),
                    'key': 'T_right',
                    'decimals': 0,
                },
                {
                    'label': self.lang['slider_size_scale'],
                    'bounds': size_bounds,
                    'initial_pos': (size_initial - size_bounds[0]) / (size_bounds[1] - size_bounds[0]),
                    'step': (size_bounds[1] - size_bounds[0]) / 100.0,
                    'key': 'size_scale',
                    'decimals': 2,
                },
                {
                    'label': self.lang['slider_tagged_count'],
                    'bounds': tag_bounds,
                    'initial_pos': (tag_initial - tag_bounds[0]) / (tag_bounds[1] - tag_bounds[0]),
                    'step': 1,
                    'key': 'tagged_count',
                    'decimals': 0,
                },
            ]
        )

        # Keep the wall temperature sliders at the front so left/right appear side by side.
        prioritized_keys = ['T_left', 'T_right']
        prioritized_sliders = []
        for key in prioritized_keys:
            for definition in slider_defs:
                if definition['key'] == key:
                    prioritized_sliders.append(definition)
                    break
        remaining_sliders = [definition for definition in slider_defs if definition['key'] not in set(prioritized_keys)]
        slider_defs = prioritized_sliders + remaining_sliders

        initial_params = OrderedDict()
        for definition in slider_defs:
            min_val, max_val = definition['bounds']
            if max_val <= min_val:
                value = min_val
            else:
                value = min_val + definition['initial_pos'] * (max_val - min_val)
            initial_params[definition['key']] = self._round_value(value, definition['decimals'])

        if 'T' in base_initial_values:
            initial_params['T'] = self._round_value(base_initial_values['T'], base_decimals.get('T', 0))

        return slider_defs, original_bounds, initial_params

    def _apply_wall_temperature_sliders(self) -> None:
        """Apply wall temperature slider changes immediately to the simulation."""
        if not self.demo or self.slider_T_left is None or self.slider_T_right is None:
            return
        new_left = float(self.slider_T_left.getValue())
        new_right = float(self.slider_T_right.getValue())
        last_left = self._last_wall_temperatures.get('left')
        last_right = self._last_wall_temperatures.get('right')
        if last_left == new_left and last_right == new_right:
            return
        self.demo.simulation.set_params(T_left=new_left, T_right=new_right)
        self.demo.params['T_left'] = new_left
        self.demo.params['T_right'] = new_right
        self.demo_config['params']['T_left'] = new_left
        self.demo_config['params']['T_right'] = new_right
        self._last_wall_temperatures['left'] = new_left
        self._last_wall_temperatures['right'] = new_right

    def _apply_particle_count_slider(self) -> None:
        """Ensure particle-count slider changes take effect immediately."""
        if not self.demo or self.slider_particle_count is None:
            return
        try:
            raw_count = int(self.slider_particle_count.getValue())
        except (TypeError, ValueError):
            raw_count = self._last_particle_count or MIN_TOTAL_PARTICLES
        new_count = max(MIN_TOTAL_PARTICLES, min(MAX_TOTAL_PARTICLES, raw_count))
        if self._last_particle_count == new_count:
            return
        self.demo_config['params']['r'] = new_count
        self.demo.params['r'] = new_count
        self.demo.set_params(self.demo_config['params'], 'r')
        self._last_particle_count = new_count

    def _apply_speed_slider(self) -> None:
        """Apply speed changes immediately to keep playback responsive."""
        if not self.demo or self.slider_speed is None:
            return
        try:
            new_speed = float(self.slider_speed.getValue())
        except (TypeError, ValueError):
            return
        if self._last_speed_value is not None and abs(self._last_speed_value - new_speed) < 1e-6:
            return
        applied = self.demo.update_speed_factor(new_speed, force=True)
        self.demo_config['params']['speed'] = applied
        self.demo.params['speed'] = applied
        self._last_speed_value = applied

    def _get_tagged_slider_value(self) -> int:
        """Return the current value of the tagged-particle slider."""
        if self.slider_tag_count is not None:
            raw = self.slider_tag_count.getValue()
        else:
            raw = self.demo_config.get('params', {}).get('tagged_count', self.tagged_count)
        try:
            value = int(raw)
        except (TypeError, ValueError):
            value = int(self.tagged_count)
        value = max(0, value)
        self.tagged_count = value
        if 'params' in self.demo_config:
            self.demo_config['params']['tagged_count'] = value
        return value

    # ------------------------------------------------------------------ Buttons actions
    def apply(self):
        self.demo_config['params'].pop('accommodation', None)

        size_val = float(self.slider_size_scale.getValue())
        self.demo.update_radius_scale(size_val)
        self.demo_config['params']['size_scale'] = size_val

        self.tagged_count = self._get_tagged_slider_value()
        self.demo_config['params']['tagged_count'] = self.tagged_count

        speed_val = float(self.slider_speed.getValue()) if self.slider_speed is not None else 1.0
        applied_speed = self.demo.update_speed_factor(speed_val, force=True)
        self.demo_config['params']['speed'] = applied_speed

        self.demo._refresh_iter(self.demo_config)
        self.demo_config['is_changed'] = False

    def add_marked_particles(self):
        if not self.demo:
            return
        count = self._get_tagged_slider_value()
        if count <= 0:
            return
        current_total = int(self.demo.params.get('r', self.demo.simulation._n_particles))
        remaining_capacity = MAX_TOTAL_PARTICLES - current_total
        if remaining_capacity <= 0:
            return
        count = min(count, remaining_capacity)
        if count <= 0:
            return
        added = self.demo.add_tagged_particles(count)
        actual_count = int(getattr(self.demo.simulation, '_n_particles', self.demo.params.get('r', 0)))
        self.demo_config['params']['r'] = actual_count
        self._last_particle_count = actual_count
        if self.slider_particle_count is not None:
            self.slider_particle_count.set_value(actual_count)
        self.demo_config['is_changed'] = bool(added)

    def reset_measurements(self):
        if not self.demo:
            return
        removed = 0
        if hasattr(self.demo, 'clear_tagged_particles'):
            removed = self.demo.clear_tagged_particles()
        self.demo.reset_measurements()
        if removed:
            actual_count = int(self.demo.params.get('r', self.demo.simulation._n_particles))
            self.demo_config['params']['r'] = actual_count
            self._last_particle_count = actual_count
            if self.slider_particle_count is not None:
                self.slider_particle_count.set_value(actual_count)
        self.demo_config['is_changed'] = False

    def toggle_dim_particles(self):
        self.dim_active = not self.dim_active
        self.demo.set_dim_untracked(self.dim_active)
        if self.dim_button is not None:
            new_label = self.dim_labels[1] if self.dim_active else self.dim_labels[0]
            self.dim_button._prep_msg(new_label)

    def toggle_trail(self):
        self.trail_active = not self.trail_active
        self.demo.set_trail_enabled(self.trail_active)
        if self.trail_button is not None:
            new_label = self.trail_labels[1] if self.trail_active else self.trail_labels[0]
            self.trail_button._prep_msg(new_label)

    def modes(self):
        return

    def to_menu(self):
        self.app.active_screen = self.app.menu_screen

    def _load_params(self):
        loader = config.ConfigLoader()
        lang = language.Language()
        param_names = [lang[name] for name in loader['param_names']]
        sliders_gap = loader['sliders_gap']
        param_poses = [(self.app.monitor.width * 0.82 + 40, h) for h in range(50, 150 + len(param_names) * sliders_gap + 1, sliders_gap)]
        param_bounds = []
        param_initial = []
        for param_name in loader['param_names']:
            param_bounds.append(tuple(loader['param_bounds'][param_name]))
            param_initial.append(loader['param_initial'][param_name])
        # Use coarser, practical steps: temperature 5 units, particle count 5, rest auto-sized.
        param_step = [round((b[1] - b[0]) / 100, 3) for b in param_bounds]
        if len(param_step) > 1:
            param_step[1] = 5
        if len(param_step) > 2:
            param_step[2] = 5
        par4sim = loader['par4sim']
        dec_numbers = [1, 0, 0, 0, 1, 0, 0]
        return param_names, sliders_gap, param_poses, param_bounds, param_initial, param_step, par4sim, dec_numbers

    # ------------------------------------------------------------------ Drawing
    def _update_screen(self):
        assert self.sim_rect is not None
        self.screen.blit(self.background, (0, 0))

        # Apply live slider changes before rendering so simulation and UI stay in sync
        self._apply_wall_temperature_sliders()
        self._apply_particle_count_slider()
        self._apply_speed_slider()

        self._draw_simulation_shadow()
        self.demo.draw_check(self.demo_config)
        self._draw_simulation_border()
        self._draw_counters()

        self._draw_slider_panel_background()
        for slider in self.sliders:
            slider.draw_check(self.demo_config['params'])

        for button in self.buttons:
            button.draw_button()

        if self.right_panel_rect:
            self._draw_flux_panel()
        self._ensure_onboarding()
        if self.onboarding.active:
            self.onboarding.draw()

    def _draw_simulation_shadow(self) -> None:
        rect = self.sim_rect
        shadow_rect = rect.copy()
        shadow_rect.x += 14
        shadow_rect.y += 18
        shadow_surface = pygame.Surface(rect.size, pygame.SRCALPHA)
        pygame.draw.rect(shadow_surface, self.shadow_color, shadow_surface.get_rect(), border_radius=24)
        self.screen.blit(shadow_surface, shadow_rect.topleft)

    def _draw_simulation_border(self) -> None:
        rect = self.sim_rect
        pygame.draw.rect(self.screen, self.border_color, rect, width=2, border_radius=20)

    def _draw_slider_panel_background(self) -> None:
        assert self.slider_panel_rect is not None
        rect = self.slider_panel_rect
        scale = getattr(self, 'layout_scale', 1.0)
        shadow_rect = rect.copy()
        shadow_rect.x += 12
        shadow_rect.y += 16
        shadow_surface = pygame.Surface(rect.size, pygame.SRCALPHA)
        pygame.draw.rect(shadow_surface, self.shadow_color, shadow_surface.get_rect(), border_radius=22)
        self.screen.blit(shadow_surface, shadow_rect.topleft)
        pygame.draw.rect(self.screen, self.panel_color, rect, border_radius=20)
        pygame.draw.rect(self.screen, self.border_color, rect, width=2, border_radius=20)

        title_surface = self.panel_title_font.render(self.panel_title_text, True, (38, 44, 60))
        title_rect = title_surface.get_rect()
        title_rect.topleft = (rect.left + max(24, rect.width // 20), rect.top + max(18, rect.height // 18))
        self.screen.blit(title_surface, title_rect)
        conc_left = conc_right = 0.0
        wall_colors = {'left': (220, 70, 40), 'right': (60, 130, 255)}
        if self.demo:
            conc_left, conc_right = self.demo.get_half_concentrations()
            wall_colors['left'] = self.demo.get_wall_color('left')
            wall_colors['right'] = self.demo.get_wall_color('right')
        layout = self._badge_layout_cache or self._compute_badge_layout(rect)
        self._badge_layout_cache = layout
        vertical_center = title_rect.top + max(0, (title_rect.height - layout['total_height']) // 2)
        badges_top = max(rect.top + layout['padding'], vertical_center)
        self._draw_concentration_badges(
            rect,
            badges_top,
            layout['total_height'],
            (conc_left, conc_right),
            wall_colors,
            layout=layout,
            align_right=True,
        )

    def _draw_flux_panel(self) -> None:
        assert self.right_panel_rect is not None
        rect = self.right_panel_rect
        self._last_flux_graph_rect = None
        self._flux_border_toggle_rect = None
        panel_scale = getattr(self, 'layout_scale', 1.0)
        shadow_rect = rect.copy()
        shadow_rect.x += 12
        shadow_rect.y += 16
        shadow_surface = pygame.Surface(rect.size, pygame.SRCALPHA)
        pygame.draw.rect(shadow_surface, self.shadow_color, shadow_surface.get_rect(), border_radius=22)
        self.screen.blit(shadow_surface, shadow_rect.topleft)
        pygame.draw.rect(self.screen, self.panel_color, rect, border_radius=20)
        pygame.draw.rect(self.screen, self.border_color, rect, width=2, border_radius=20)

        title_surface = self.panel_title_font.render(self.graph_title_text, True, (38, 44, 60))
        title_rect = title_surface.get_rect()
        padding = max(int(18 * panel_scale), rect.width // 28)
        title_rect.topleft = (rect.left + padding, rect.top + padding)
        self.screen.blit(title_surface, title_rect)

        toggle_rect = None
        self._flux_border_toggle_rect = None
        # Always draw the toggle; fall back to a hardcoded label so it can't vanish due to locale issues.
        toggle_label = self.graph_border_toggle_label
        if not toggle_label:
            toggle_label = 'Границы графика' if getattr(self.lang, 'lang', 'eng') == 'rus' else 'Graph borders'
        toggle_label = toggle_label.strip() or ('Границы графика' if getattr(self.lang, 'lang', 'eng') == 'rus' else 'Graph borders')

        enabled = bool(self._flux_border_enabled)
        status_on = 'Вкл.' if getattr(self.lang, 'lang', 'eng') == 'rus' else 'On'
        status_off = 'Выкл.' if getattr(self.lang, 'lang', 'eng') == 'rus' else 'Off'
        status = status_on if enabled else status_off
        button_label = f'{toggle_label}: {status}'
        label_color = (255, 255, 255) if enabled else (70, 76, 94)
        label_surface = self.graph_counts_font.render(button_label, True, label_color)
        button_padding_x = max(18, int(panel_scale * 16))
        button_padding_y = max(9, int(panel_scale * 8))
        button_height = max(label_surface.get_height() + button_padding_y, int(panel_scale * 34))
        indicator_radius = max(6, int(button_height * 0.18))
        indicator_gap = max(6, int(panel_scale * 6))
        indicator_space = indicator_radius * 2 + indicator_gap
        button_width = label_surface.get_width() + button_padding_x * 2 + indicator_space

        max_toggle_width = max(100, rect.width - 2 * padding)
        if button_width > max_toggle_width:
            button_width = max_toggle_width

        toggle_top = rect.top + padding + max(0, (title_rect.height - button_height) // 2)
        toggle_left = rect.right - padding - button_width

        min_gap = max(12, int(panel_scale * 10))
        if toggle_left <= title_rect.right + min_gap:
            toggle_top = title_rect.bottom + max(6, int(panel_scale * 6))
            toggle_left = rect.right - padding - button_width
            toggle_left = max(toggle_left, rect.left + padding)

        toggle_rect = pygame.Rect(toggle_left, toggle_top, button_width, button_height)
        fill_color = (72, 104, 255) if enabled else (244, 245, 249)
        border_color = (46, 76, 210) if enabled else (130, 138, 156)
        border_radius = max(12, button_height // 2)
        shadow_surface = pygame.Surface((toggle_rect.width + 8, toggle_rect.height + 8), pygame.SRCALPHA)
        pygame.draw.rect(shadow_surface, (0, 0, 0, 50), shadow_surface.get_rect(), border_radius=border_radius + 3)
        self.screen.blit(shadow_surface, (toggle_rect.x + 2, toggle_rect.y + 2))
        pygame.draw.rect(self.screen, fill_color, toggle_rect, border_radius=border_radius)
        pygame.draw.rect(self.screen, border_color, toggle_rect, width=2, border_radius=border_radius)
        indicator_center = (
            toggle_rect.left + button_padding_x + indicator_radius,
            toggle_rect.centery,
        )
        indicator_color = (235, 240, 255) if enabled else (245, 247, 252)
        indicator_ring = (255, 80, 80) if enabled else (170, 176, 192)
        pygame.draw.circle(self.screen, indicator_ring, indicator_center, indicator_radius + 3)
        pygame.draw.circle(self.screen, indicator_color, indicator_center, indicator_radius)

        label_rect = label_surface.get_rect()
        label_rect.centery = toggle_rect.centery
        label_rect.left = indicator_center[0] + indicator_radius + indicator_gap
        self.screen.blit(label_surface, label_rect)

        self._flux_border_toggle_rect = toggle_rect

        cumulative_heat = 0.0
        step_heat = 0.0
        flux_value = 0.0
        flux_raw_value = 0.0
        value_text = None
        counts_text = None
        if self.demo and getattr(self.demo, 'simulation', None):
            sim = self.demo.simulation
            cumulative_heat = float(sim.get_cumulative_midplane_heat())
            step_heat = float(sim.get_last_midplane_heat_transfer())
            flux_value = float(sim.get_last_midplane_flux())
            flux_raw_value = float(sim.get_last_midplane_flux_raw())
            scale = getattr(self.demo, 'flux_display_scale', 1.0)
            try:
                scale = float(scale)
            except (TypeError, ValueError):
                scale = 1.0
            if not math.isfinite(scale) or scale <= 0.0:
                scale = 1.0
            cumulative_heat *= scale
            step_heat *= scale
            flux_value *= scale
            flux_raw_value *= scale
            formatter = self._format_flux_number
            format_payload = {
                'step': step_heat,
                'cumulative': cumulative_heat,
                'flux': flux_value,
                'flux_raw': flux_raw_value,
                'step_display': formatter(step_heat),
                'cumulative_display': formatter(cumulative_heat),
                'flux_display': formatter(flux_value),
                'flux_raw_display': formatter(flux_raw_value),
            }
            energy_template = self.graph_flux_template
            if energy_template:
                try:
                    value_text = energy_template.format(**format_payload)
                except (KeyError, ValueError):
                    value_text = energy_template
            counts_template = self.graph_counts_template
            if counts_template:
                try:
                    counts_text = counts_template.format(**format_payload)
                except (KeyError, ValueError):
                    counts_text = counts_template

        text_y_anchor = title_rect.bottom
        if toggle_rect:
            text_y_anchor = max(text_y_anchor, toggle_rect.bottom)
        text_y = text_y_anchor + max(10, padding // 4)
        line_gap = max(12, padding // 3)
        if value_text:
            value_surface = self.graph_value_font.render(value_text, True, (42, 48, 66))
            value_rect = value_surface.get_rect()
            value_rect.topleft = (rect.left + padding, text_y)
            self.screen.blit(value_surface, value_rect)
            text_y = value_rect.bottom + line_gap
        if counts_text:
            counts_surface = self.graph_counts_font.render(counts_text, True, (70, 76, 94))
            counts_rect = counts_surface.get_rect()
            counts_rect.topleft = (rect.left + padding, text_y)
            self.screen.blit(counts_surface, counts_rect)
            text_y = counts_rect.bottom + line_gap
        else:
            text_y += line_gap

        graph_bottom = rect.bottom - padding
        if graph_bottom <= text_y + 20:
            return

        graph_rect = pygame.Rect(rect.left + padding, text_y, rect.width - 2 * padding, graph_bottom - text_y)
        pygame.draw.rect(self.screen, (254, 255, 255), graph_rect, border_radius=18)
        pygame.draw.rect(self.screen, self.border_color, graph_rect, width=1, border_radius=18)
        self._last_flux_graph_rect = graph_rect

        inner_margin = max(10, int(graph_rect.width * 0.03))
        inner_margin = min(inner_margin, graph_rect.width // 2 - 1 if graph_rect.width > 2 else inner_margin)
        inner_margin = min(inner_margin, graph_rect.height // 2 - 1 if graph_rect.height > 2 else inner_margin)
        plot_rect = graph_rect.inflate(-2 * inner_margin, -2 * inner_margin)
        if plot_rect.width <= 2 or plot_rect.height <= 2:
            plot_rect = graph_rect.inflate(-8, -8)

        samples: list[tuple[float, ...]] = []
        if self.demo:
            samples = list(self.demo.midplane_flux_samples)
        if len(samples) > self.graph_max_samples:
            samples = samples[-self.graph_max_samples:]

        scale_limits: tuple[float, float] | None = None

        if self.demo and len(samples) >= 2 and plot_rect.width > 1 and plot_rect.height > 1:
            scale_limits = self.demo.draw_midplane_flux_graph(
                self.screen,
                plot_rect,
                samples=samples,
                line_color=self.primary_color,
                baseline_color=(186, 190, 208),
                background=(0, 0, 0, 0),
                series='cumulative',
                draw_highlight_lines=self._flux_border_enabled,
            )
        else:
            placeholder_surface = self.placeholder_font.render(self.placeholder_text, True, (120, 128, 146))
            placeholder_rect = placeholder_surface.get_rect()
            placeholder_rect.center = plot_rect.center if plot_rect.width > 1 and plot_rect.height > 1 else graph_rect.center
            self.screen.blit(placeholder_surface, placeholder_rect)

        if scale_limits is not None:
            self._draw_flux_axis_labels(plot_rect, graph_rect, scale_limits)


    def _draw_flux_axis_labels(
        self,
        plot_rect: pygame.Rect,
        graph_rect: pygame.Rect,
        scale_limits: tuple[float, float],
    ) -> None:
        if not self.demo or scale_limits is None:
            return
        min_val, max_val = scale_limits
        axis_span = max_val - min_val
        if not math.isfinite(axis_span) or axis_span <= 0.0 or plot_rect.height <= 0.0:
            return

        grid_lines = max(2, getattr(self.demo, 'flux_grid_lines', 4))
        axis_label_offset = max(12, min(32, plot_rect.width // 16))
        axis_label_right = max(plot_rect.left + axis_label_offset, graph_rect.left + 12)
        axis_label_right = min(axis_label_right, plot_rect.left + max(plot_rect.width - 4, 0))
        label_font = self.graph_counts_font
        label_color = (42, 48, 66)
        axis_label_top_limit = graph_rect.top + 2
        axis_units = getattr(self, 'axis_units_label', None)
        if axis_units:
            unit_surface = label_font.render(axis_units, True, label_color)
            unit_rect = unit_surface.get_rect()
            unit_rect.right = axis_label_right
            unit_rect.top = graph_rect.top + 4
            self.screen.blit(unit_surface, unit_rect.topleft)
            axis_label_top_limit = max(axis_label_top_limit, unit_rect.bottom + 6)

        for idx in range(grid_lines + 1):
            fraction = idx / grid_lines
            value = min_val + axis_span * fraction
            if not math.isfinite(value):
                continue
            y_position = plot_rect.bottom - fraction * plot_rect.height

            label_text = self._format_flux_number(value)
            if not label_text:
                continue
            label_surface = label_font.render(label_text, True, label_color)
            label_rect = label_surface.get_rect()
            label_rect.right = axis_label_right
            label_rect.centery = int(round(y_position))
            if label_rect.top < axis_label_top_limit:
                label_rect.top = axis_label_top_limit
            if label_rect.bottom > graph_rect.bottom - 2:
                label_rect.bottom = graph_rect.bottom - 2

            bg_rect = label_rect.inflate(10, 4)
            bg_rect.left = max(bg_rect.left, graph_rect.left + 2)
            bg_surface = pygame.Surface(bg_rect.size, pygame.SRCALPHA)
            bg_surface.fill((255, 255, 255, 220))
            self.screen.blit(bg_surface, bg_rect.topleft)
            self.screen.blit(label_surface, label_rect.topleft)

    def _draw_counters(self):
        """Render counters showing how many tagged particles hit each wall."""
        if not self.demo or not self.demo.has_tagged_particles():
            return
        left_hits, right_hits = self.demo.get_wall_hit_counts()
        left_text = self.counter_left_template.format(count=left_hits)
        right_text = self.counter_right_template.format(count=right_hits)
        left_surface = self.counter_font.render(left_text, True, (220, 70, 40))
        right_surface = self.counter_font.render(right_text, True, (60, 130, 255))
        padding = 20
        top_y = max(12, self.sim_rect.top - left_surface.get_height() - 12)
        left_x = self.sim_rect.left + padding
        right_x = self.sim_rect.left + self.sim_rect.width - right_surface.get_width() - padding
        self.screen.blit(left_surface, (left_x, top_y))
        self.screen.blit(right_surface, (right_x, top_y))

    def _draw_concentration_badges(
        self,
        panel_rect: pygame.Rect,
        top: int,
        height: int,
        values: tuple[float, float],
        wall_colors: dict[str, tuple[int, int, int]],
        *,
        layout: dict[str, Any] | None = None,
        align_right: bool = False,
    ) -> None:
        labels = getattr(self, 'conc_side_labels', {'left': 'Left', 'right': 'Right'})
        unit_label = getattr(self, 'conc_unit_label', 'part/unit²')
        layout = layout or self._compute_badge_layout(panel_rect)
        badge_gap = layout['gap']
        padding = layout['padding']
        badge_width = layout['width']
        badge_height = layout['height']
        rows = layout['rows']
        columns = layout['columns']
        row_block_height = rows * badge_height + max(0, (rows - 1)) * badge_gap
        area_height = max(height, row_block_height)
        total_badge_width = columns * badge_width + max(0, columns - 1) * badge_gap
        row_start_x = panel_rect.left + padding
        if align_right:
            row_start_x = max(panel_rect.left + padding, panel_rect.right - padding - total_badge_width)
        current_y = top + max(0, (area_height - row_block_height) // 2)
        current_x = row_start_x
        max_x = panel_rect.right - padding

        for idx, side in enumerate(('left', 'right')):
            if current_x + badge_width > max_x and idx != 0:
                current_x = row_start_x
                current_y += badge_height + badge_gap
            card_rect = pygame.Rect(int(current_x), int(current_y), int(badge_width), int(badge_height))
            accent = wall_colors.get(side, (72, 104, 255))
            background = self._blend_with_white(accent, 0.78)
            card_surface = pygame.Surface(card_rect.size, pygame.SRCALPHA)
            pygame.draw.rect(card_surface, (*background, 245), card_surface.get_rect(), border_radius=16)
            pygame.draw.rect(card_surface, accent, card_surface.get_rect(), width=2, border_radius=16)
            self.screen.blit(card_surface, card_rect.topleft)

            label_text = labels.get(side, side.title())
            label_surface = self.conc_label_font.render(label_text, True, accent)
            label_rect = label_surface.get_rect()
            label_rect.topleft = (card_rect.left + 14, card_rect.top + 10)
            self.screen.blit(label_surface, label_rect)

            value = values[idx]
            value_text = f'{value:.2f}'
            value_font = self.conc_value_font
            unit_font = self.conc_label_font
            gap = 6
            inner_margin = 16
            max_line_width = card_rect.width - inner_margin * 2

            def render_pair(v_font):
                v_surface = v_font.render(value_text, True, (24, 27, 39))
                u_surface = unit_font.render(unit_label, True, (64, 70, 94))
                return v_surface, u_surface

            value_surface, unit_surface = render_pair(value_font)
            combined_width = value_surface.get_width() + gap + unit_surface.get_width()
            while combined_width > max_line_width and value_font.get_height() > 14:
                new_size = max(14, value_font.get_height() - 2)
                value_font = get_font(new_size, bold=True)
                value_surface, unit_surface = render_pair(value_font)
                combined_width = value_surface.get_width() + gap + unit_surface.get_width()

            content_bottom = card_rect.bottom - 12
            value_rect = value_surface.get_rect()
            unit_rect = unit_surface.get_rect()
            total_width = value_surface.get_width() + gap + unit_surface.get_width()
            value_rect.right = card_rect.right - inner_margin - unit_surface.get_width() - gap
            unit_rect.left = value_rect.right + gap
            value_rect.bottom = content_bottom
            unit_rect.bottom = content_bottom
            self.screen.blit(value_surface, value_rect)
            self.screen.blit(unit_surface, unit_rect)

            current_x += badge_width + badge_gap

    def _format_flux_number(self, value: float) -> str:
        if not math.isfinite(value):
            return '0'
        abs_val = abs(value)
        if abs_val >= 1000:
            text = f"{value:,.0f}"
        elif abs_val >= 100:
            text = f"{value:,.1f}"
        elif abs_val >= 10:
            text = f"{value:,.2f}"
        elif abs_val >= 1:
            text = f"{value:,.3f}"
        else:
            text = f"{value:.4f}"
        text = text.replace(',', ' ')
        if '.' in text:
            text = text.rstrip('0').rstrip('.')
        return text

    @staticmethod
    def _blend_with_white(color: tuple[int, int, int], factor: float) -> tuple[int, int, int]:
        factor = max(0.0, min(1.0, factor))
        return tuple(int(component + (255 - component) * factor) for component in color)

    # ------------------------------------------------------------------ Events
    def _check_events(self):
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
                quit()
            elif event.type == pygame.VIDEORESIZE:
                self.app.handle_resize(event.size)
            elif self.onboarding.handle_event(event):
                continue
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_position = pygame.mouse.get_pos()
                if self._handle_flux_border_toggle(mouse_position):
                    continue
                self._check_buttons(mouse_position)
        mouse_pos = pygame.mouse.get_pos()
        mouse = pygame.mouse.get_pressed()
        self._check_sliders(mouse_pos, mouse)

    def _check_buttons(self, mouse_position):
        for button in self.buttons:
            if button.rect.collidepoint(mouse_position):
                button.command()

    def _handle_flux_border_toggle(self, mouse_position: tuple[int, int]) -> bool:
        rect = self._flux_border_toggle_rect
        if rect is None:
            return False
        if rect.collidepoint(mouse_position):
            self._flux_border_enabled = not self._flux_border_enabled
            return True
        return False

    def _check_sliders(self, mouse_position, mouse_pressed):
        for slider in self.sliders:
            slider.slider.hovered = False
            if slider.slider.button_rect.collidepoint(mouse_position):
                if mouse_pressed[0] and not self.slider_grabbed:
                    slider.slider.grabbed = True
                    self.slider_grabbed = True
            if not mouse_pressed[0]:
                slider.slider.grabbed = False
                self.slider_grabbed = False
            if slider.slider.button_rect.collidepoint(mouse_position):
                slider.slider.hovered = True
            if slider.slider.grabbed:
                slider.slider.move_slider(mouse_position)
                slider.slider.hovered = True

    def correct_limits(self):
        return

    # ---------------------------------------------------------------- Onboarding
    def queue_onboarding_from_menu(self) -> None:
        """Schedule onboarding to start when this screen becomes active."""
        self._pending_onboarding = True

    def _ensure_onboarding(self) -> None:
        if getattr(self.app, 'onboarding_demo_done', False):
            return
        if self.onboarding.active or self._onboarding_started:
            return
        if not self._pending_onboarding:
            return
        self._pending_onboarding = False
        self._onboarding_started = True
        steps = self._build_onboarding_steps()
        self.onboarding.start(steps, on_complete=self._finish_onboarding, hint=self._nav_hint_text())

    def _refresh_onboarding_locale(self) -> None:
        if not self.onboarding.active:
            return
        steps = self._build_onboarding_steps()
        self.onboarding.update_steps(steps, hint=self._nav_hint_text(), keep_index=True)

    def _finish_onboarding(self, aborted: bool) -> None:
        self.app.mark_onboarding_done('demo_done')
        if aborted:
            return

    def _nav_hint_text(self) -> str:
        return (
            "ЛКМ — дальше, ПКМ/Backspace — назад, Esc — пропустить"
            if self.lang.lang == "rus"
            else "LMB — next, RMB/Backspace — back, Esc — skip"
        )

    def _build_onboarding_steps(self) -> list[dict]:
        lang_ru = self.lang.lang == "rus"
        steps: list[dict] = []
        steps.append(
            {
                'title': "Демонстрация" if lang_ru else "Demonstration",
                'body': (
                    "Это область движения частиц между левой и правой стенкой. Стартуем без градиента: обе стенки "
                    "одинаковой температуры, но вы можете сразу изменить её слайдерами. Частицы сталкиваются и переносят тепло, тем самым возникает градиент температур"
                    if lang_ru
                    else "Particles move between the left and right walls. We start with no gradient—both walls share the "
                    "same temperature—but you can change it with the sliders. Collisions move heat across the domain."
                ),
                'rect': lambda: self.sim_rect,
                'placement': 'bottom',
                'padding': 26,
            }
        )
        steps.append(
            {
                'title': "Температура стенок" if lang_ru else "Wall temperatures",
                'body': (
                    "Эти слайдеры изменяют температуру стенов: можете увеличивать температуру левой и уменьшать температуру правой. "
                    "Левая стенка горячая, правая холодная: величина градиента задается этими слайдерами."
                    if lang_ru
                    else "These sliders change the walls temperature: you can heat the left one and cool down the right one. "
                    "The left wall is hot, the right wall is cold — the value of the gradient can be changed here"
                ),
                'rects': self._rects_for_sliders(['T_left', 'T_right']) + self._wall_rects(['left', 'right']),
                'placement': 'top',
            }
        )
        steps.append(
            {
                'title': "Количество частиц" if lang_ru else "Amount of particles",
                'body': (
                    "Слайдер количества частиц работает мгновенно: двигайте вправо и влево - число точек на экране сразу меняется."
                    if lang_ru
                    else "The particle-count slider is live: move it left or right and the particles update immediately."
                ),
                'rects': self._rects_for_sliders(['r']) + [self._sim_rect()],
                'placement': 'right',
            }
        )
        steps.append(
            {
                'title': "Скорость симуляции" if lang_ru else "Simulation speed",
                'body': (
                    "Скорость симуляции тоже меняется сразу: можете ускорять для проверки эффектов или замедлять при надобности"
                    if lang_ru
                    else "Simulation speed also respond instantly: you can speed it up to test your hypothesis or slow down if needed."
                ),
                'rects': self._rects_for_sliders(['speed']),
                'placement': 'top',
            }
        )
        steps.append(
            {
                'title': "Число частиц примеси" if lang_ru else "Impurity particles count",
                'body': (
                    "Число частиц примеси отвечает за то, сколько новых черных частиц вы хотите добавить в центр симуляции. Эти частицы не отличаются по физическим свойствам от других частиц в симуляции, только цветом."
                    if lang_ru
                    else "Impurity particle count is responsible for the amount of black particles that you are able to add to the center of the simulation. They have the same physical attributes as other in the simulation."
                ),
                'rects': self._rects_for_sliders(['tagged_count']),
                'placement': 'top',
            }
        )
        steps.append(
            {
                'title': "Размер частиц" if lang_ru else "Particle size",
                'body': (
                    "Размер частиц меняется после нажатия «Применить». Подберите масштаб, затем нажмите кнопку, чтобы обновить физический и визуальный размеры частиц"
                    if lang_ru
                    else "Particle size updates only after you press Apply. Choose a scale, then hit the button to refresh collisions physics and rendering."
                ),
                'rects': self._rects_for_sliders(['size_scale']) + [self._button_rect(self.apply_button)],
                'placement': 'top',
            }
        )
        steps.append(
            {
                'title': "Поток через середину" if lang_ru else "Flux through the middle",
                'body': (
                    "График справа рисует тепловой поток через стенку посередине "
                    "Зелёная стенка посередине — отсюда берём поток: частицы, перелетающие через неё, дают вклад в кривую, равный энергии частицы"
                    if lang_ru
                    else "The right graph shows heat flux "
                    "The green divider in the middle is the measuring wall: particles crossing it feed the plotted flux."
                ),
                'rects': [self._flux_rect(), self._midplane_rect()],
                'placement': 'left',
            }
        )
        steps.append(
            {
                'title': self.label_add_particles if lang_ru else self.label_add_particles,
                'body': (
                    "Добавляет отмеченные частицы (число задаёте в соответствующем слайдере). "
                    "Они появятся сразу и затем программа считает столкновения с горячей и холодной стенкой."
                    if lang_ru
                    else "Adds tagged particles (count set by the tagged slider). "
                    "They spawn immediately and start hitting the hot and cold walls."
                ),
                'rect': lambda: self._button_rect(self.add_button),
                'placement': 'top',
            }
        )
        steps.append(
            {
                'title': self.dim_labels[0] + " / " + self.trail_labels[0],
                'body': (
                    "«Обесцветить фон» оставляет только отмеченные частицы яркими, «Показать след» рисует траекторию выбранной случайной частицы. "
                    "Полезно, чтобы увидеть, как чдвижутся частицы в разных областях."
                    if lang_ru
                    else "“Dim background” keeps only tagged particles bright; “Show trail” draws a path for the tracked one. "
                    "Great for seeing how particles move in the different areas."
                ),
                'rects': [self._button_rect(self.dim_button), self._button_rect(self.trail_button)],
                'placement': 'top',
            }
        )
        steps.append(
            {
                'title': self.reset_label,
                'body': (
                    "Сбрасывает счётчики столкновений и график потока — удобно начать замер заново."
                    if lang_ru
                    else "Resets hit counters and the flux graph so you can start measurements fresh."
                ),
                'rect': lambda: self._button_rect(self.reset_button),
                'placement': 'top',
            }
        )
        steps.append(
            {
                'title': self.lang['btn_menu'],
                'body': (
                    "Вернуться в меню, если хочется снова пройти теорию или обучение."
                    if lang_ru
                    else "Return to the menu whenever you want theory or the text tutorial."
                ),
                'rect': lambda: self._button_rect(self.menu_button),
                'placement': 'top',
            }
        )
        return steps

    def _rects_for_sliders(self, keys: list[str]) -> list[pygame.Rect]:
        rects = []
        for key in keys:
            rect = self._slider_rect(key)
            if rect is not None:
                rects.append(rect)
        return rects

    def _wall_rects(self, sides: list[str]) -> list[pygame.Rect]:
        rects = []
        for side in sides:
            rect = self._wall_rect(side)
            if rect is not None:
                rects.append(rect)
        return rects

    def _slider_rect(self, key: str):
        slider = self.slider_lookup.get(key)
        return getattr(slider, 'card_rect', None) if slider else None

    def _button_rect(self, button: Button | None):
        return getattr(button, 'rect', None) if button else None

    def _sim_rect(self):
        return self.sim_rect

    def _wall_rect(self, side: str):
        if not self.demo or not self.sim_rect:
            return None
        thickness = max(4, int(getattr(self.demo, 'wall_thickness_px', 8)))
        rect = self.sim_rect
        if side == 'left':
            return pygame.Rect(rect.left, rect.top, thickness, rect.height)
        if side == 'right':
            return pygame.Rect(rect.right - thickness, rect.top, thickness, rect.height)
        return None

    def _midplane_rect(self):
        if not self.demo or not self.sim_rect:
            return None
        rect = self.sim_rect
        axis = getattr(self.demo, 'midplane_axis', 'x')
        plane_coord = float(getattr(self.demo, 'midplane_position', 0.5))
        ratio = MIDPLANE_WALL_THICKNESS_RATIO
        if axis == 'y':
            wall_height = max(2, int(round(rect.height * ratio)))
            clamped = max(self.demo.simulation.R, min(1.0 - self.demo.simulation.R, plane_coord))
            wall_y = rect.top + rect.height - int(round(rect.height * clamped))
            return pygame.Rect(rect.left, wall_y - wall_height // 2, rect.width, wall_height)
        wall_width = max(2, int(round(rect.width * ratio)))
        box_width_units = max(self.demo.simulation.get_box_width(), 1e-6)
        norm = max(0.0, min(1.0, plane_coord / box_width_units))
        wall_x = rect.left + int(round(rect.width * norm))
        return pygame.Rect(wall_x - wall_width // 2, rect.top, wall_width, rect.height)

    def _flux_rect(self):
        return self._last_flux_graph_rect
