from __future__ import annotations

from pathlib import Path
import io
from typing import List, Tuple

import pygame

try:
    import fitz  # type: ignore
except ImportError:  # pragma: no cover
    fitz = None  # type: ignore

import language
from button import Button
from slider import ParamSlider
from ui_base import ResponsiveScreen, get_font
from paths import resource_file


class TheoryScreen(ResponsiveScreen):
    PDF_RENDER_SCALE = 2.0
    """Base DPR used when rasterizing the PDF pages."""
    DEFAULT_PAGE_ZOOM = 1.4
    """Default zoom level applied to the fitted page."""
    MIN_PAGE_ZOOM = 0.8
    """Minimum zoom available through the slider."""
    MAX_PAGE_ZOOM = 2.4
    """Maximum zoom available through the slider."""
    ZOOM_STEP = 0.05
    """Step size applied when dragging the zoom slider."""
    SCROLL_STEP = 48
    """Pixels scrolled per mouse wheel notch."""

    def __init__(self, app):
        super().__init__(app)
        self.lang = language.Language()

        self.background_color = (240, 244, 255)
        self.card_color = (255, 255, 255)
        self.card_border = (215, 220, 235)
        self.shadow_color = (18, 24, 60, 55)
        self.text_color = (34, 38, 52)
        self.muted_text = (110, 118, 134)
        self.accent_color = (72, 104, 255)

        self.pdf_paths = {
            "rus": "_internal/theory/theory_ru.pdf",
            "eng": "_internal/theory/theory_en.pdf",
        }

        self.lang_code = self.lang.lang
        self.original_pages: List[pygame.Surface] = []
        self.scaled_pages: List[Tuple[pygame.Surface, pygame.Rect]] = []
        self.status_lines: List[str] = []
        self.status_surfaces: List[Tuple[pygame.Surface, pygame.Rect]] = []

        self.content_rect: pygame.Rect | None = None
        self.page_index = 0

        self.page_zoom = self.DEFAULT_PAGE_ZOOM
        self.page_scroll = 0
        self.page_scroll_limit = 0
        self.page_view_rect: pygame.Rect | None = None
        self.zoom_slider: ParamSlider | None = None
        self._slider_dragging = False
        self.slider_panel_height = 0
        self.slider_panel_gap = 0
        self.slider_panel_margin = 0
        self.slider_panel_width = 0

        self.title_font: pygame.font.Font | None = None
        self.body_font: pygame.font.Font | None = None
        self.label_font: pygame.font.Font | None = None

        self.prev_button: Button | None = None
        self.menu_button: Button | None = None
        self.next_button: Button | None = None

        self.page_label_surface: pygame.Surface | None = None
        self.page_label_rect: pygame.Rect | None = None

        self.last_size = self.app.window_size
        self._load_pdf_pages()
        self._relayout(self.app.window_size)

    # ------------------------------------------------------------------ Layout
    def _relayout(self, size: tuple[int, int]) -> None:
        self.last_size = size
        width, height = size
        width = max(width, 900)
        height = max(height, 620)

        if self.lang.lang != self.lang_code:
            self.lang_code = self.lang.lang
            self._load_pdf_pages()

        margin_x = max(56, int(width * 0.08))
        margin_top = max(48, int(height * 0.08))
        margin_bottom = max(150, int(height * 0.22))
        content_width = max(480, width - 2 * margin_x)
        content_height = max(260, height - margin_top - margin_bottom)
        slider_gap = max(28, int(width * 0.02))
        slider_height = max(90, min(150, int(content_height * 0.15)))
        slider_width = min(320, max(200, int(width * 0.2)))
        self.slider_panel_height = slider_height
        self.slider_panel_gap = slider_gap
        self.slider_panel_margin = max(12, slider_gap // 2)
        self.slider_panel_width = slider_width
        view_margin_bottom = max(80, int(content_height * 0.18))

        self.content_rect = pygame.Rect(
            margin_x - 30,
            margin_top - 24,
            content_width + 60,
            content_height + 48,
        )

        view_margin_x = max(32, int(content_width * 0.06))
        view_margin_top = max(24, int(content_height * 0.08))
        view_width = max(360, content_width - view_margin_x * 2)
        view_height = max(260, content_height - view_margin_top - view_margin_bottom)
        self.page_view_rect = pygame.Rect(0, 0, view_width, view_height)
        self.page_view_rect.centerx = self.content_rect.centerx
        self.page_view_rect.top = self.content_rect.top + view_margin_top

        title_size = max(30, int(content_width * 0.052))
        body_size = max(20, int(content_width * 0.032))
        label_size = max(18, int(content_width * 0.028))

        self.title_font = get_font(title_size, bold=True)
        self.body_font = get_font(body_size)
        self.label_font = get_font(label_size, bold=False)

        view_width = self.page_view_rect.width if self.page_view_rect else content_width
        view_height = self.page_view_rect.height if self.page_view_rect else content_height
        self._scale_pages(view_width, view_height)
        self._build_status_surfaces(
            margin_x=margin_x,
            margin_top=margin_top,
            available_height=view_height,
            content_width=content_width,
        )
        self._build_buttons(width, height)
        self._build_zoom_slider(width)
        self._update_page_label()

    def _scale_pages(self, content_width: int, content_height: int) -> None:
        if not self.original_pages:
            self.scaled_pages = []
            self.page_scroll = 0
            self.page_scroll_limit = 0
            return

        target_width = content_width
        target_height = content_height
        padding = 32
        self.scaled_pages = []

        for surface in self.original_pages:
            sw, sh = surface.get_size()
            rw = max(1, target_width - padding)
            rh = max(1, target_height - padding)
            scale_fit = min(rw / sw, rh / sh)
            scale = scale_fit * self.page_zoom
            new_size = (max(1, int(sw * scale)), max(1, int(sh * scale)))
            scaled = pygame.transform.smoothscale(surface, new_size)
            rect = scaled.get_rect()
            if self.page_view_rect:
                rect.centerx = self.page_view_rect.centerx
                rect.top = self.page_view_rect.top
            elif self.content_rect:
                rect.centerx = self.content_rect.centerx
                rect.top = self.content_rect.top + 24
            else:
                rect.center = (new_size[0] // 2, new_size[1] // 2)
            self.scaled_pages.append((scaled, rect))

        self.page_index = min(self.page_index, len(self.scaled_pages) - 1) if self.scaled_pages else 0
        self.page_scroll = 0
        self._update_scroll_limits()

    def _update_scroll_limits(self) -> None:
        if not self.scaled_pages or self.page_view_rect is None:
            self.page_scroll = 0
            self.page_scroll_limit = 0
            return
        _, rect = self.scaled_pages[self.page_index]
        view_height = self.page_view_rect.height
        self.page_scroll_limit = max(0, rect.height - view_height)
        self.page_scroll = min(self.page_scroll, self.page_scroll_limit)

    def _build_status_surfaces(
        self,
        *,
        margin_x: int,
        margin_top: int,
        available_height: int,
        content_width: int,
    ) -> None:
        self.status_surfaces = []
        if not self.status_lines or self.body_font is None:
            return

        line_spacing = max(6, int(self.body_font.get_linesize() * 0.25))
        line_height = self.body_font.get_linesize()
        total_height = len(self.status_lines) * line_height + (len(self.status_lines) - 1) * line_spacing
        start_y = margin_top + max(0, (available_height - total_height) // 2)
        center_x = margin_x + content_width // 2

        current_y = start_y
        for line in self.status_lines:
            surface = self.body_font.render(line, True, self.text_color)
            rect = surface.get_rect()
            rect.centerx = center_x
            rect.y = current_y
            self.status_surfaces.append((surface, rect))
            current_y += line_height + line_spacing

    def _build_buttons(self, width: int, height: int) -> None:
        button_height = max(60, int(height * 0.09))
        nav_width = max(110, int(width * 0.1))
        menu_width = max(280, int(width * 0.3))
        bottom_margin = max(40, int(height * 0.05))
        y = height - bottom_margin - button_height

        button_font = ["SF Pro Display", "Segoe UI", "Arial"]
        button_kwargs = {
            "font": button_font,
            "fontSize": max(24, int(button_height * 0.45)),
            "bold": True,
            "border_radius": 18,
        }

        if self.content_rect is None:
            base_x = 42
            self.content_rect = pygame.Rect(base_x, base_x, width - base_x * 2, height - base_x * 3)

        self.prev_button = Button(
            self.app,
            "<",
            (self.content_rect.left, y),
            (nav_width, button_height),
            self.last_page,
            button_color=(245, 247, 252),
            text_color=self.accent_color,
            border_color=(210, 214, 230),
            shadow_offset=6,
            shadow_color=(0, 0, 0, 40),
            **button_kwargs,
        )

        self.next_button = Button(
            self.app,
            ">",
            (self.content_rect.right - nav_width, y),
            (nav_width, button_height),
            self.next_page,
            button_color=(245, 247, 252),
            text_color=self.accent_color,
            border_color=(210, 214, 230),
            shadow_offset=6,
            shadow_color=(0, 0, 0, 40),
            **button_kwargs,
        )

        left_limit = self.content_rect.left + nav_width + 20
        right_limit = self.content_rect.right - nav_width - 20 - menu_width
        center_x = (width - menu_width) // 2
        menu_x = max(left_limit, center_x)
        if right_limit >= left_limit:
            menu_x = min(menu_x, right_limit)
        self.menu_button = Button(
            self.app,
            self.lang["btn_menu"],
            (menu_x, y),
            (menu_width, button_height),
            self.to_menu,
            button_color=self.accent_color,
            text_color=(255, 255, 255),
            shadow_offset=9,
            shadow_color=(64, 99, 255, 110),
            **button_kwargs,
        )

    def _build_zoom_slider(self, width: int) -> None:
        if self.content_rect is None:
            self.zoom_slider = None
            return
        slider_height = self.slider_panel_height or max(90, min(170, int(self.content_rect.height * 0.15)))
        slider_gap = self.slider_panel_gap or 24
        slider_width = self.slider_panel_width or 240
        margin_x = max(56, int(width * 0.08))
        target_x = self.content_rect.right + slider_gap
        max_x = width - margin_x - slider_width
        slider_x = min(max_x, target_x)
        slider_y = self.content_rect.centery - slider_height // 2
        slider_rect = (int(slider_x), int(slider_y), int(slider_width), int(slider_height))
        slider_rect = (int(slider_x), int(slider_y), int(slider_width), int(slider_height))
        slider_label = self.lang["theory_zoom_slider"]
        slider_range = max(self.MAX_PAGE_ZOOM - self.MIN_PAGE_ZOOM, 1e-3)
        initial_ratio = (self.page_zoom - self.MIN_PAGE_ZOOM) / slider_range
        initial_ratio = max(0.0, min(1.0, initial_ratio))
        slider = ParamSlider(
            self.app,
            slider_label,
            slider_rect,
            (self.MIN_PAGE_ZOOM, self.MAX_PAGE_ZOOM),
            self.ZOOM_STEP,
            "page_zoom",
            2,
            initial_ratio,
            padding=18,
            label_size=max(16, slider_height // 7),
            value_size=max(20, slider_height // 6),
            track_height=max(8, slider_height // 12),
            value_suffix="×",
        )
        slider.set_value(self.page_zoom)
        self.zoom_slider = slider
        self._slider_dragging = False

    def _apply_zoom_from_slider(self) -> None:
        if self.zoom_slider is None:
            return
        new_zoom = self.zoom_slider.getValue()
        if abs(new_zoom - self.page_zoom) < 1e-4:
            return
        self.page_zoom = new_zoom
        view_width = self.page_view_rect.width if self.page_view_rect else (self.content_rect.width if self.content_rect else 0)
        view_height = self.page_view_rect.height if self.page_view_rect else (self.content_rect.height if self.content_rect else 0)
        if view_width <= 0 or view_height <= 0:
            return
        self._scale_pages(view_width, view_height)
        self._update_page_label()

    def _handle_zoom_slider_mouse_down(self, mouse_position: tuple[int, int]) -> bool:
        if self.zoom_slider is None:
            return False
        slider_impl = self.zoom_slider.slider
        if slider_impl.button_rect.collidepoint(mouse_position) or slider_impl.track.collidepoint(mouse_position):
            slider_impl.grabbed = True
            slider_impl.hovered = True
            self._slider_dragging = True
            slider_impl.move_slider(mouse_position)
            self._apply_zoom_from_slider()
            return True
        return False

    def _stop_zoom_slider_drag(self) -> None:
        if self.zoom_slider is None:
            return
        self.zoom_slider.slider.grabbed = False
        self._slider_dragging = False

    def _handle_zoom_slider_motion(self, mouse_position: tuple[int, int]) -> None:
        if self.zoom_slider is None or not self._slider_dragging:
            return
        slider_impl = self.zoom_slider.slider
        slider_impl.move_slider(mouse_position)
        slider_impl.hovered = True
        self._apply_zoom_from_slider()

    def _update_page_label(self) -> None:
        if self.label_font is None or not self.scaled_pages:
            self.page_label_surface = None
            self.page_label_rect = None
            return

        total = len(self.scaled_pages)
        current = self.page_index + 1
        templates = {
            "rus": "Страница {current}/{total}",
            "eng": "Page {current}/{total}",
        }
        text = templates.get(self.lang.lang, templates["eng"]).format(current=current, total=total)
        self.page_label_surface = self.label_font.render(text, True, self.muted_text)
        rect = self.page_label_surface.get_rect()
        rect.centerx = self.content_rect.centerx if self.content_rect else self.app.screen.get_rect().centerx
        rect.bottom = (self.menu_button.rect.top - 16) if self.menu_button else self.app.screen.get_rect().height - 48
        self.page_label_rect = rect

    # ------------------------------------------------------------------ Loading helpers
    def _load_pdf_pages(self) -> None:
        self.original_pages = []
        self.scaled_pages = []
        self.status_lines = []
        self.page_index = 0

        relative = self.pdf_paths.get(self.lang_code, self.pdf_paths["eng"])
        path = resource_file(relative)

        if fitz is None:
            self.status_lines = self._status_message("missing_lib", relative)
            return

        if not path.exists():
            self.status_lines = self._status_message("missing_file", relative)
            return

        try:
            doc = fitz.open(path)
        except Exception as exc:  # pragma: no cover
            self.status_lines = self._status_message("load_error", str(exc))
            return

        try:
            scale = fitz.Matrix(self.PDF_RENDER_SCALE, self.PDF_RENDER_SCALE)
            for page in doc:
                pix = page.get_pixmap(matrix=scale)
                image_bytes = pix.tobytes("png")
                buffer = io.BytesIO(image_bytes)
                try:
                    surface = pygame.image.load(buffer)
                finally:
                    buffer.close()
                if surface.get_flags() & pygame.SRCALPHA:
                    surface = surface.convert_alpha()
                else:
                    surface = surface.convert()
                self.original_pages.append(surface)
        finally:
            doc.close()

        if not self.original_pages:
            self.status_lines = self._status_message("empty_pdf", str(path))

    def _status_message(self, key: str, detail: str) -> List[str]:
        templates = {
            "missing_lib": {
                "rus": [
                    "Не удалось отобразить PDF.",
                    "Установите пакет pymupdf: pip install pymupdf.",
                ],
                "eng": [
                    "Cannot render PDF pages.",
                    "Install the pymupdf package: pip install pymupdf.",
                ],
            },
            "missing_file": {
                "rus": [
                    "Файл теории не найден:",
                    "{detail}",
                    "Скомпилируйте theory_ru.tex и theory_en.tex и поместите PDF в _internal/theory/.",
                ],
                "eng": [
                    "Theory PDF is missing:",
                    "{detail}",
                    "Compile theory_en.tex and theory_ru.tex, then place PDFs into _internal/theory/.",
                ],
            },
            "load_error": {
                "rus": [
                    "Ошибка при чтении PDF:",
                    "{detail}",
                ],
                "eng": [
                    "Error reading PDF:",
                    "{detail}",
                ],
            },
            "empty_pdf": {
                "rus": [
                    "PDF не содержит страниц: {detail}",
                ],
                "eng": [
                    "PDF has no pages: {detail}",
                ],
            },
        }

        messages = templates.get(key, templates["load_error"])
        selected = messages.get(self.lang_code, messages["eng"])
        return [line.format(detail=detail) for line in selected]

    # ---------------------------------------------------------------- Buttons
    def to_menu(self):
        self.app.active_screen = self.app.menu_screen

    def last_page(self):
        if self.page_index > 0:
            self.page_index -= 1
            self.page_scroll = 0
            self._update_scroll_limits()
            self._update_page_label()

    def next_page(self):
        if self.page_index < len(self.scaled_pages) - 1:
            self.page_index += 1
            self.page_scroll = 0
            self._update_scroll_limits()
            self._update_page_label()

    # ---------------------------------------------------------------- Drawing
    def _update_screen(self):
        self.screen.fill(self.background_color)

        if self.content_rect:
            shadow_rect = self.content_rect.move(14, 18)
            shadow_surface = pygame.Surface(self.content_rect.size, pygame.SRCALPHA)
            pygame.draw.rect(shadow_surface, self.shadow_color, shadow_surface.get_rect(), border_radius=26)
            self.screen.blit(shadow_surface, shadow_rect.topleft)
            pygame.draw.rect(self.screen, self.card_color, self.content_rect, border_radius=24)
            pygame.draw.rect(self.screen, self.card_border, self.content_rect, width=2, border_radius=24)

        if self.scaled_pages:
            prev_clip = None
            if self.page_view_rect:
                prev_clip = self.screen.get_clip()
                self.screen.set_clip(self.page_view_rect)
            surface, rect = self.scaled_pages[self.page_index]
            rect_with_scroll = rect.copy()
            rect_with_scroll.y -= self.page_scroll
            self.screen.blit(surface, rect_with_scroll)
            if prev_clip is not None:
                self.screen.set_clip(prev_clip)
        elif self.status_surfaces:
            for surface, rect in self.status_surfaces:
                self.screen.blit(surface, rect)

        if self.zoom_slider:
            self.zoom_slider.draw_check({})

        for button in (self.prev_button, self.menu_button, self.next_button):
            if button:
                button.draw_button()

        if self.page_label_surface and self.page_label_rect:
            self.screen.blit(self.page_label_surface, self.page_label_rect)

    # ---------------------------------------------------------------- Events
    def _check_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quit()
            elif event.type == pygame.VIDEORESIZE:
                self.app.handle_resize(event.size)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1 and self._handle_zoom_slider_mouse_down(event.pos):
                    continue
                mouse_position = pygame.mouse.get_pos()
                self._check_buttons(mouse_position)
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    self._stop_zoom_slider_drag()
            elif event.type == pygame.MOUSEMOTION:
                self._handle_zoom_slider_motion(event.pos)
            elif event.type == pygame.MOUSEWHEEL:
                self._handle_scroll(event.y)

    def _check_buttons(self, mouse_position):
        for button in (self.prev_button, self.menu_button, self.next_button):
            if button and button.rect.collidepoint(mouse_position):
                button.command()

    def _handle_scroll(self, delta: int) -> None:
        if delta == 0 or self.page_scroll_limit <= 0:
            return
        scroll_amount = -delta * self.SCROLL_STEP
        self.page_scroll = min(max(self.page_scroll + scroll_amount, 0), self.page_scroll_limit)
