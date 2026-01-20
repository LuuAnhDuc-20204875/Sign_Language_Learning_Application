from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Tuple, Optional, List

import cv2
import cvzone
import numpy as np


class SnakeGame:
    """Grid-based snake controlled by the user's index finger direction.

    - Board is drawn higher by reserving a bottom "hand space" so user can keep hand in frame.
    - Food occupies a kxk block (default 3x3).
    - Snake can move in 8 directions (including diagonals).
    """

    def __init__(self, food_image: str, game_width: int, game_height: int):
        self.game_w = int(game_width)
        self.game_h = int(game_height)

        # ========= BOARD LAYOUT =========
        # Left/right margin
        self.margin_x = 70
        # Top margin (keep some space for header text inside frame)
        self.margin_top = 70
        # Extra bottom space reserved for hand tracking comfort
        self.hand_space_bottom = 150  # <-- tăng lên 200~260 nếu tay hay mất ở đáy defaul 170!
        # Bottom margin = normal margin + reserved hand space
        self.margin_bottom = 70 + self.hand_space_bottom

        self.cell = 26  # pixel per grid cell

        # Grid size computed from usable area (height excludes bottom hand space)
        usable_w = max(1, self.game_w - 2 * self.margin_x)
        usable_h = max(1, self.game_h - self.margin_top - self.margin_bottom)

        self.grid_w = max(8, usable_w // self.cell)
        self.grid_h = max(8, usable_h // self.cell)

        # ========= FOOD SETTINGS =========
        self.food_cells = 3
        self._food_img_scale = 0.92

        self.imgFood = self._load_food(food_image)
        self.imgFood = self._fit_food_to_block(self.imgFood, self.cell, self.food_cells, self._food_img_scale)
        self.hFood, self.wFood = self.imgFood.shape[:2]

        # movement
        self.move_interval = 0.12
        self._last_move_t = time.time()
        self._eat_flash_until = 0.0

        self.score = 0
        self.gameOver = False
        self.reset()

    # -------------------------
    # Assets
    # -------------------------
    def _load_food(self, path: str) -> np.ndarray:
        p = Path(path)
        if p.exists():
            img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
            if img is not None and img.size > 0:
                if img.ndim == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
                elif img.shape[2] == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
                return img

        # fallback RGBA icon
        img = np.zeros((72, 72, 4), dtype=np.uint8)
        cv2.circle(img, (36, 40), 26, (40, 70, 220, 255), -1)
        cv2.circle(img, (28, 34), 10, (90, 120, 255, 200), -1)
        cv2.ellipse(
            img,
            center=(46, 20),
            axes=(10, 6),
            angle=-20,
            startAngle=0,
            endAngle=360,
            color=(40, 200, 60, 255),
            thickness=-1,
        )
        # cv2.ellipse(img, (46, 20), (10, 6), -20,hook := 0, startAngle := 0, endAngle := 360, color := (40, 200, 60, 255), thickness := -1)

        cv2.line(img, (36, 22), (36, 10), (40, 80, 120, 255), 4)
        return img

    def _fit_food_to_block(self, img: np.ndarray, cell: int, cells: int, fill: float) -> np.ndarray:
        target = int(cell * cells)
        canvas = np.zeros((target, target, 4), dtype=np.uint8)

        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
        elif img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

        fill = max(0.50, min(0.99, float(fill)))
        new_w = max(1, int(target * fill))
        new_h = max(1, int(target * fill))
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        x1 = (target - new_w) // 2
        y1 = (target - new_h) // 2
        canvas[y1:y1 + new_h, x1:x1 + new_w] = resized
        return canvas

    # -------------------------
    # Game state
    # -------------------------
    def reset(self):
        self.score = 0
        self.gameOver = False
        self._eat_flash_until = 0.0
        self._last_move_t = time.time()

        self._dir = (1, 0)
        self._pending_dir = (1, 0)

        cx, cy = self.grid_w // 2, self.grid_h // 2
        self.snake: List[Tuple[int, int]] = [(cx, cy), (cx - 1, cy), (cx - 2, cy)]
        self._grow = 0
        self._paused = False
        self._last_hand_seen_t = time.time()

        self.food_tl = (0, 0)
        self._random_food_location()

    def _food_cells_list(self, tl: Tuple[int, int]) -> List[Tuple[int, int]]:
        x0, y0 = tl
        k = self.food_cells
        return [(x0 + i, y0 + j) for i in range(k) for j in range(k)]

    def _random_food_location(self):
        k = self.food_cells
        if self.grid_w < k or self.grid_h < k:
            self.food_tl = (max(0, self.grid_w // 2), max(0, self.grid_h // 2))
            return

        snake_set = set(self.snake)
        candidates = []
        for x in range(0, self.grid_w - k + 1):
            for y in range(0, self.grid_h - k + 1):
                block = self._food_cells_list((x, y))
                if all(c not in snake_set for c in block):
                    candidates.append((x, y))

        if not candidates:
            self.food_tl = (max(0, (self.grid_w - k) // 2), max(0, (self.grid_h - k) // 2))
            return

        self.food_tl = random.choice(candidates)

    # -------------------------
    # Board origin (IMPORTANT: moved up)
    # -------------------------
    def _board_origin(self) -> Tuple[int, int]:
        # center inside usable area (vertical excludes bottom hand space)
        usable_w = self.game_w - 2 * self.margin_x
        usable_h = self.game_h - self.margin_top - self.margin_bottom

        bw = self.grid_w * self.cell
        bh = self.grid_h * self.cell

        ox = self.margin_x + max(0, (usable_w - bw) // 2)
        oy = self.margin_top + max(0, (usable_h - bh) // 2)
        return int(ox), int(oy)

    # -------------------------
    # Input: finger -> direction
    # -------------------------
    def _head_pixel_center(self) -> Tuple[int, int]:
        gx, gy = self.snake[0]
        ox, oy = self._board_origin()
        return (int(ox + gx * self.cell + self.cell * 0.5), int(oy + gy * self.cell + self.cell * 0.5))

    def _update_direction_from_finger(self, finger_px: Tuple[int, int]):
        hx, hy = self._head_pixel_center()
        fx, fy = int(finger_px[0]), int(finger_px[1])
        dxp, dyp = fx - hx, fy - hy

        dead = self.cell * 0.55
        if abs(dxp) < dead and abs(dyp) < dead:
            return

        sx = 0 if abs(dxp) < dead else (1 if dxp > 0 else -1)
        sy = 0 if abs(dyp) < dead else (1 if dyp > 0 else -1)

        if sx != 0 and sy != 0:
            cand = (sx, sy)
        else:
            if abs(dxp) >= abs(dyp):
                cand = (1, 0) if dxp > 0 else (-1, 0)
            else:
                cand = (0, 1) if dyp > 0 else (0, -1)

        if cand[0] == -self._dir[0] and cand[1] == -self._dir[1]:
            return

        self._pending_dir = cand

    # -------------------------
    # Drawing helpers
    # -------------------------
    def _alpha_rect(self, img, pt1, pt2, color, alpha=0.35):
        overlay = img.copy()
        cv2.rectangle(overlay, pt1, pt2, color, -1)
        return cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

    def _draw_grid(self, img, ox: int, oy: int, bw: int, bh: int):
        for i in range(self.grid_w + 1):
            x = ox + i * self.cell
            cv2.line(img, (x, oy), (x, oy + bh), (40, 40, 40), 1)
        for j in range(self.grid_h + 1):
            y = oy + j * self.cell
            cv2.line(img, (ox, y), (ox + bw, y), (40, 40, 40), 1)

    def _cell_center(self, ox: int, oy: int, cell_xy: Tuple[int, int]) -> Tuple[int, int]:
        x, y = cell_xy
        return (int(ox + x * self.cell + self.cell * 0.5), int(oy + y * self.cell + self.cell * 0.5))

    def _cell_topleft(self, ox: int, oy: int, cell_xy: Tuple[int, int]) -> Tuple[int, int]:
        x, y = cell_xy
        return (int(ox + x * self.cell), int(oy + y * self.cell))

    def _draw_snake(self, img, ox: int, oy: int):
        now = time.time()
        body_color = (60, 200, 80)
        body_dark = (30, 120, 40)
        head_color = (40, 220, 90)

        r_body = int(self.cell * 0.44)
        r_head = int(self.cell * 0.50)

        for idx in range(len(self.snake) - 1, -1, -1):
            c = self._cell_center(ox, oy, self.snake[idx])
            r = r_head if idx == 0 else r_body

            cv2.circle(img, (c[0] + 2, c[1] + 3), r, (0, 0, 0), -1, lineType=cv2.LINE_AA)

            if idx == 0:
                cv2.circle(img, c, r, head_color, -1, lineType=cv2.LINE_AA)
                cv2.circle(img, c, int(r * 0.92), body_color, 2, lineType=cv2.LINE_AA)
            else:
                t = idx / max(1, len(self.snake) - 1)
                col = (
                    int(body_color[0] * (1 - 0.15 * t)),
                    int(body_color[1] * (1 - 0.20 * t)),
                    int(body_color[2] * (1 - 0.15 * t)),
                )
                cv2.circle(img, c, r, col, -1, lineType=cv2.LINE_AA)
                cv2.circle(img, c, int(r * 0.92), body_dark, 2, lineType=cv2.LINE_AA)
                cv2.circle(img, (c[0] - int(r * 0.25), c[1] - int(r * 0.20)),
                           int(r * 0.18), (120, 255, 170), -1, lineType=cv2.LINE_AA)

        hx, hy = self._cell_center(ox, oy, self.snake[0])
        dx, dy = self._dir

        eye_off = int(r_head * 0.35)
        eye_sep = int(r_head * 0.30)

        if dx != 0:
            ex = hx + int(dx * eye_off)
            e1 = (ex, hy - eye_sep)
            e2 = (ex, hy + eye_sep)
        else:
            ey = hy + int(dy * eye_off)
            e1 = (hx - eye_sep, ey)
            e2 = (hx + eye_sep, ey)

        for e in (e1, e2):
            cv2.circle(img, e, int(r_head * 0.18), (255, 255, 255), -1, lineType=cv2.LINE_AA)
            cv2.circle(img, e, int(r_head * 0.09), (20, 20, 20), -1, lineType=cv2.LINE_AA)

        if now < self._eat_flash_until:
            tip = (hx + int(dx * r_head * 0.85), hy + int(dy * r_head * 0.85))
            cv2.circle(img, tip, int(r_head * 0.13), (0, 0, 255), -1, lineType=cv2.LINE_AA)

    def _draw_food(self, img, ox: int, oy: int):
        x0, y0 = self.food_tl
        px, py = self._cell_topleft(ox, oy, (x0, y0))
        try:
            return cvzone.overlayPNG(img, self.imgFood, (px, py))
        except Exception:
            k = self.food_cells
            cx, cy = self._cell_center(ox, oy, (x0 + k // 2, y0 + k // 2))
            cv2.circle(img, (cx, cy), int(self.cell * 0.55), (0, 0, 255), -1, lineType=cv2.LINE_AA)
            return img

    # -------------------------
    # Main update
    # -------------------------
    def update(self, imgMain, currentHead: Optional[Tuple[int, int]]):
        now = time.time()

        if self.gameOver:
            overlay = imgMain.copy()
            cv2.rectangle(overlay, (140, 180), (self.game_w - 140, self.game_h - 140), (0, 0, 0), -1)
            imgMain = cv2.addWeighted(overlay, 0.62, imgMain, 0.38, 0)

            cvzone.putTextRect(imgMain, "SNAKE - GAME OVER", [190, 250],
                               scale=2.6, thickness=3, offset=18,
                               colorT=(255, 255, 255), colorR=(30, 30, 220))
            cvzone.putTextRect(imgMain, f"Score: {self.score}", [190, 330],
                               scale=2.3, thickness=2, offset=14)
            cvzone.putTextRect(imgMain, "Press R to Restart", [190, 410],
                               scale=1.8, thickness=2, offset=10)
            return imgMain

        if currentHead is not None:
            self._last_hand_seen_t = now
            self._paused = False
            self._update_direction_from_finger(currentHead)
        else:
            if now - self._last_hand_seen_t > 0.6:
                self._paused = True

        if (not self._paused) and (now - self._last_move_t >= self.move_interval):
            steps = int((now - self._last_move_t) / self.move_interval)
            steps = min(steps, 3)

            for _ in range(steps):
                self._dir = self._pending_dir
                hx, hy = self.snake[0]
                nx, ny = hx + self._dir[0], hy + self._dir[1]
                new_head = (nx, ny)

                if nx < 0 or ny < 0 or nx >= self.grid_w or ny >= self.grid_h or new_head in self.snake:
                    self.gameOver = True
                    break

                self.snake.insert(0, new_head)

                food_cells = set(self._food_cells_list(self.food_tl))
                if new_head in food_cells:
                    self.score += 1
                    self._grow += 2
                    self._eat_flash_until = now + 0.35
                    self._random_food_location()

                if self._grow > 0:
                    self._grow -= 1
                else:
                    self.snake.pop()

            self._last_move_t = now

        ox, oy = self._board_origin()
        bw = self.grid_w * self.cell
        bh = self.grid_h * self.cell

        imgMain = self._alpha_rect(imgMain, (ox - 14, oy - 14), (ox + bw + 14, oy + bh + 14), (0, 0, 0), alpha=0.35)
        cv2.rectangle(imgMain, (ox - 14, oy - 14), (ox + bw + 14, oy + bh + 14), (80, 80, 80), 2)
        self._draw_grid(imgMain, ox, oy, bw, bh)

        imgMain = self._draw_food(imgMain, ox, oy)
        self._draw_snake(imgMain, ox, oy)

        cvzone.putTextRect(imgMain, f"Score: {self.score}", [40, 60],
                           scale=2, thickness=2, offset=8,
                           colorR=(50, 50, 50), colorT=(255, 255, 255))
        if self._paused:
            cvzone.putTextRect(imgMain, "Hand lost - PAUSED", [40, 115],
                               scale=1.5, thickness=2, offset=8,
                               colorR=(60, 60, 60), colorT=(255, 255, 255))

        return imgMain
