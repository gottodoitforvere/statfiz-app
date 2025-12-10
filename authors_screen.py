from __future__ import annotations

from typing import List, Tuple

import pygame

import language
from button import Button
from ui_base import ResponsiveScreen, get_font, build_vertical_gradient
from paths import resource_file


class AuthorsScreen(ResponsiveScreen):
    MENU_BUTTON_HEIGHT = 64
    MENU_BUTTON_GAP = 20

    def __init__(self, app):
        super().__init__(app)
        self.lang = language.Language()
        self.assets_dir = "_internal/images"

        self.primary_color = (72, 104, 255)
        self.text_color = (33, 36, 44)
        self.muted_text = (96, 105, 124)
        self.accent_text = (72, 104, 255)

        self.background: pygame.Surface | None = None
        self.header_surfaces: List[Tuple[pygame.Surface, pygame.Rect]] = []
        self.author_cards: List[dict] = []
        self.menu_button: Button | None = None

        self.logo_raw = pygame.image.load(resource_file(f"{self.assets_dir}/msu_logo.jpg")).convert()
        self.author_images = {
            'author_pavshinsky': pygame.image.load(resource_file(f"{self.assets_dir}/Daniil_Pavshinsky.jpg")).convert(),
            'author_kozlova': pygame.image.load(resource_file(f"{self.assets_dir}/Kate_Kozlova.png")).convert(),
        }

        self.authors = self._collect_authors()
        self._relayout(self.app.window_size)

    def on_language_change(self) -> None:
        self.lang = language.Language()
        self.authors = self._collect_authors()
        self._relayout(self.app.window_size)

    def _collect_authors(self) -> List[dict]:
        raw_keys = [key for key in self.lang._loader.keys() if key.startswith('author')]
        keys: List[str] = []
        for key in sorted(raw_keys):
            value = self.lang._loader.get(key)
            if isinstance(value, str):
                keys.append(key)
        if not keys:
            keys = ['author_pavshinsky', 'author_kozlova']
        contacts_map = self.lang._loader.get('author_contacts', {}) if isinstance(self.lang._loader, dict) else {}
        authors = []
        for key in keys:
            raw_name = self.lang[key] if key in self.lang._loader else key.replace('author_', '').title()
            name = raw_name if isinstance(raw_name, str) else str(raw_name)
            image = self.author_images.get(key)
            raw_contacts = contacts_map.get(key, []) if isinstance(contacts_map, dict) else []
            contacts = []
            if isinstance(raw_contacts, list):
                contacts = [str(item).strip() for item in raw_contacts if str(item).strip()]
            authors.append({'name': name, 'image_key': key, 'image': image, 'contacts': contacts})
        return authors

    # ------------------------------------------------------------------ Layout
    def _relayout(self, size: tuple[int, int]) -> None:
        width, height = size
        width = max(width, 900)
        height = max(height, 650)
        self.background = build_vertical_gradient(size, (235, 240, 255), (249, 250, 254))

        self._build_header(width, height)
        self._build_cards(width, height)
        self._build_menu_button(width, height)

    def _build_header(self, width: int, height: int) -> None:
        horizontal_padding = max(48, width // 18)
        vertical_padding = max(12, min(horizontal_padding, width // 32))
        top = max(8, vertical_padding // 2)
        lines = [
            (self.lang['university_name'], max(26, width // 34), True, self.primary_color),
            (self.lang['faculty_name'], max(22, width // 38), False, self.text_color),
            (self.lang['lecturer'], max(20, width // 42), False, self.muted_text),
            (self.lang['supervisor'], max(20, width // 42), False, self.muted_text),
        ]
        current_y = top
        self.header_surfaces = []

        for text, size, bold, color in lines:
            font = get_font(size, bold=bold)
            surface = font.render(text, True, color)
            rect = surface.get_rect()
            rect.x = horizontal_padding
            rect.y = current_y
            self.header_surfaces.append((surface, rect))
            current_y = rect.bottom + max(14, size // 4)

        logo_size = max(120, min(180, width // 8))
        self.logo_surface = pygame.transform.smoothscale(self.logo_raw, (logo_size, logo_size))
        self.logo_position = (width - horizontal_padding - logo_size, max(8, vertical_padding // 2))
        header_gap = max(10, min(logo_size // 12 + 4, 20))
        self.cards_top = current_y + header_gap

    def _build_cards(self, width: int, height: int) -> None:
        horizontal_padding = max(48, width // 18)
        gap = max(18, width // 60)
        cards_area_width = width - horizontal_padding * 2

        num_authors = len(self.authors)
        if num_authors == 0:
            self.author_cards = []
            self.cards_bottom = self.cards_top
            return

        max_columns = min(3, num_authors)
        preferred_card_width = 520
        possible_columns = cards_area_width // preferred_card_width
        columns = max(1, min(max_columns, possible_columns or 1))
        columns = min(columns, num_authors)
        rows = (num_authors + columns - 1) // columns
        card_width = (cards_area_width - gap * (columns - 1)) // columns

        button_height = self.MENU_BUTTON_HEIGHT
        button_gap = max(self.MENU_BUTTON_GAP, horizontal_padding // 2)
        available_total_height = height - self.cards_top - horizontal_padding - button_gap - button_height
        available_total_height = max(available_total_height, 0)

        def build_layouts(scale: float) -> List[dict]:
            return [self._prepare_card_layout(author, card_width, scale=scale) for author in self.authors]

        scale = 1.0
        card_layouts = build_layouts(scale)
        if available_total_height > 0:
            max_layout_height = max(layout['height'] for layout in card_layouts)
            default_total_height = max_layout_height * rows + gap * (rows - 1)
            if default_total_height > available_total_height:
                scale = max(0.75, available_total_height / default_total_height)
                card_layouts = build_layouts(scale)

        self.author_cards = []
        current_y = self.cards_top
        for row_start in range(0, num_authors, columns):
            row_layouts = card_layouts[row_start : row_start + columns]
            row_height = max(layout['height'] for layout in row_layouts)
            for column, layout in enumerate(row_layouts):
                card_x = horizontal_padding + column * (card_width + gap)
                rect = pygame.Rect(card_x, current_y, card_width, layout['height'])
                card = {'rect': rect, 'photo': None, 'texts': []}
                center_x = rect.centerx
                if layout['photo']:
                    photo_surface = layout['photo']['surface']
                    photo_rect = photo_surface.get_rect()
                    photo_rect.centerx = center_x
                    photo_rect.y = rect.y + layout['photo']['offset']
                    card['photo'] = (photo_surface, photo_rect)
                for text_info in layout['texts']:
                    text_surface = text_info['surface']
                    text_rect = text_surface.get_rect()
                    text_rect.centerx = center_x
                    text_rect.y = rect.y + text_info['offset']
                    card['texts'].append((text_surface, text_rect))
                self.author_cards.append(card)
            current_y += row_height + gap

        self.cards_bottom = max(card['rect'].bottom for card in self.author_cards)

    def _prepare_card_layout(self, author: dict, card_width: int, *, scale: float = 1.0) -> dict:
        scale = max(0.75, min(1.0, scale))
        inner_padding = max(18, int(card_width // 15 * scale))
        photo_area = max(0, card_width - inner_padding * 2)
        photo_size = max(120, min(photo_area, int(card_width * 0.55 * scale)))
        photo_surface = self._build_author_photo(author, photo_size)

        offset = inner_padding
        photo_offset = offset
        offset += photo_size + max(6, inner_padding // 3)

        texts: List[dict] = []
        name_font = get_font(max(20, int(card_width * 0.07 * scale)), bold=True)
        name_surface = name_font.render(author['name'], True, self.text_color)
        name_offset = offset
        texts.append({'surface': name_surface, 'offset': name_offset})
        offset = name_offset + name_surface.get_height() + max(4, int(6 * scale))

        tagline = 'Автор' if self.lang.lang == 'rus' else 'Author'
        tagline_font = get_font(max(16, int(card_width * 0.045 * scale)))
        tagline_surface = tagline_font.render(tagline, True, self.muted_text)
        tagline_offset = offset
        texts.append({'surface': tagline_surface, 'offset': tagline_offset})
        offset = tagline_offset + tagline_surface.get_height() + max(3, int(5 * scale))

        contact_lines = author.get('contacts', [])
        contact_label = self.lang._loader.get('contacts_label', 'Контакты' if self.lang.lang == 'rus' else 'Contacts')
        if contact_lines:
            spacing_before_contact = max(6, inner_padding // 5)
            offset += spacing_before_contact
            label_font = get_font(max(16, int(card_width * 0.042 * scale)), bold=True)
            label_surface = label_font.render(str(contact_label), True, self.accent_text)
            label_offset = offset
            texts.append({'surface': label_surface, 'offset': label_offset})
            offset = label_offset + label_surface.get_height() + max(4, int(6 * scale))

            contact_font = get_font(max(14, int(card_width * 0.038 * scale)))
            line_spacing = max(2, int(3 * scale))
            for line in contact_lines:
                line_surface = contact_font.render(line, True, self.muted_text)
                line_offset = offset
                texts.append({'surface': line_surface, 'offset': line_offset})
                offset = line_offset + line_surface.get_height() + line_spacing

        offset += inner_padding
        min_card_height = max(360, int(card_width * 0.9 * scale))
        card_height = max(min_card_height, offset)

        return {
            'height': card_height,
            'photo': {'surface': photo_surface, 'offset': photo_offset},
            'texts': texts,
        }

    def _build_author_photo(self, author: dict, size: int) -> pygame.Surface:
        if author['image'] is not None:
            return pygame.transform.smoothscale(author['image'], (size, size))

        surface = pygame.Surface((size, size), pygame.SRCALPHA)
        pygame.draw.rect(surface, (240, 243, 255), (0, 0, size, size), border_radius=size // 5)
        pygame.draw.rect(surface, (215, 222, 245), (0, 0, size, size), width=2, border_radius=size // 5)
        initials = ''.join(part[0] for part in author['name'].split()[:2]).upper()
        font = get_font(max(24, size // 3), bold=True)
        text_surface = font.render(initials, True, (72, 104, 255))
        text_rect = text_surface.get_rect(center=(size // 2, size // 2))
        surface.blit(text_surface, text_rect)
        return surface

    def _build_menu_button(self, width: int, height: int) -> None:
        horizontal_padding = max(48, width // 18)
        button_width = min(320, max(240, width // 5))
        button_height = self.MENU_BUTTON_HEIGHT
        button_x = (width - button_width) // 2
        button_gap = max(self.MENU_BUTTON_GAP, horizontal_padding // 2)
        cards_bottom = getattr(self, "cards_bottom", self.cards_top)
        desired_y = cards_bottom + button_gap
        max_y = height - horizontal_padding - button_height
        button_y = min(desired_y, max_y)
        self.menu_button = Button(
            self.app,
            self.lang['btn_menu'],
            (button_x, button_y),
            (button_width, button_height),
            self.to_menu,
            font=["SF Pro Display", "Segoe UI", "Arial"],
            fontSize=max(22, int(button_height * 0.42)),
            bold=True,
            button_color=self.primary_color,
            text_color=(255, 255, 255),
            border_radius=18,
            shadow_offset=8,
            shadow_color=(64, 99, 255, 120),
        )

    # ---------------------------------------------------------------- Events
    def to_menu(self):
        self.app.active_screen = self.app.menu_screen

    def _check_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quit()
            elif event.type == pygame.VIDEORESIZE:
                self.app.handle_resize(event.size)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_position = pygame.mouse.get_pos()
                if self.menu_button and self.menu_button.rect.collidepoint(mouse_position):
                    self.menu_button.command()

    # ---------------------------------------------------------------- Drawing
    def _update_screen(self):
        self.screen.blit(self.background, (0, 0))
        for surface, rect in self.header_surfaces:
            self.screen.blit(surface, rect)
        self.screen.blit(self.logo_surface, self.logo_position)

        for card in self.author_cards:
            self._draw_card(card)

        if self.menu_button:
            self.menu_button.draw_button()

    def _draw_card(self, card: dict) -> None:
        rect: pygame.Rect = card['rect']
        shadow_rect = rect.copy()
        shadow_rect.x += 10
        shadow_rect.y += 14
        shadow_surface = pygame.Surface(rect.size, pygame.SRCALPHA)
        pygame.draw.rect(shadow_surface, (18, 25, 64, 55), shadow_surface.get_rect(), border_radius=22)
        self.screen.blit(shadow_surface, shadow_rect.topleft)
        pygame.draw.rect(self.screen, (255, 255, 255), rect, border_radius=20)
        pygame.draw.rect(self.screen, (216, 221, 236), rect, width=2, border_radius=20)

        if card['photo']:
            surface, photo_rect = card['photo']
            self.screen.blit(surface, photo_rect)
        for surface, text_rect in card['texts']:
            self.screen.blit(surface, text_rect)
