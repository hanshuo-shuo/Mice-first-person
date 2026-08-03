"""Playable binocular Cellworld app with optional human-demonstration capture.

The macOS app bundle in this repository launches this module.  It can also be
run directly from the Mice-BotEvade environment:

    conda run -n Mice-BotEvade python -B mouse_play_app.py

Controls
--------
Arrow up/down     forward/backward velocity
Arrow left/right  body yaw rate
A / D             head/gaze yaw rate
Space             stop translation
R                 toggle demonstration recording
N                 reset / start a new episode
P                 pause
M or Escape       return to the menu
Q                 quit
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CACHE_ROOT = PROJECT_ROOT / "cellworld_cache"
DEFAULT_DATA_ROOT = PROJECT_ROOT / "datasets" / "human_demos"
DEFAULT_WORLD = "21_05"
RESOURCE_BASE_URL = (
    "https://raw.githubusercontent.com/germanespinosa/cellworld_data/master"
)

# cellworld reads this variable once, when its util module is imported.
os.environ.setdefault("CELLWORLD_CACHE", str(DEFAULT_CACHE_ROOT))
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".cache" / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_ROOT / ".cache"))
os.chdir(PROJECT_ROOT)

import pygame


Color = Tuple[int, int, int]
WHITE: Color = (239, 244, 246)
MUTED: Color = (151, 164, 171)
BACKGROUND: Color = (13, 18, 22)
PANEL: Color = (24, 31, 36)
PANEL_LIGHT: Color = (35, 44, 50)
CYAN: Color = (45, 205, 224)
GREEN: Color = (58, 222, 113)
RED: Color = (244, 72, 72)
AMBER: Color = (246, 190, 62)


def _resource_specs(world_name: str) -> List[Tuple[str, bool]]:
    """Return cache-relative resources and whether each one is optional."""

    return [
        ("world_configuration/hexagonal", False),
        ("world_implementation/hexagonal.canonical", False),
        (f"cell_group/hexagonal.{world_name}.occlusions", False),
        (f"paths/hexagonal.{world_name}.astar.robot", False),
        (f"cell_group/hexagonal.{world_name}.spawn_locations", False),
        (f"cell_group/hexagonal.{world_name}.lppo", True),
        (f"graph/hexagonal.{world_name}.cell_visibility", False),
    ]


def _atomic_json_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def ensure_cellworld_cache(
    world_name: str = DEFAULT_WORLD,
    cache_root: Path = DEFAULT_CACHE_ROOT,
    progress: Optional[Callable[[str], None]] = None,
) -> None:
    """Populate the resources needed by CellWorldLoader for offline use."""

    cache_root = Path(cache_root)
    for relative_path, optional in _resource_specs(world_name):
        destination = cache_root / relative_path
        if destination.exists():
            continue
        if progress is not None:
            progress(f"正在准备本地地图：{Path(relative_path).name}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        url = f"{RESOURCE_BASE_URL}/{relative_path}"
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "MouseFirstPersonLab/1.0"},
        )
        try:
            with urllib.request.urlopen(request, timeout=25) as response:
                payload = response.read().decode("utf-8")
            parsed = json.loads(payload)
        except Exception as error:
            if optional:
                parsed = []
            else:
                raise RuntimeError(
                    "缺少 Cellworld 本地地图资源，且自动下载失败：\n"
                    f"{relative_path}\n\n{error}",
                ) from error
        _atomic_json_write(destination, parsed)


def _json_scalar(value: Any) -> Any:
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, str) or value is None:
        return value
    return str(value)


class DemonstrationRecorder:
    """Buffer and atomically save VLA-ready human demonstration chunks."""

    ARRAY_KEYS = (
        "image_left",
        "image_right",
        "proprio",
        "previous_action",
        "action",
        "reward",
        "terminated",
        "truncated",
        "sim_time",
        "privileged_state",
    )

    def __init__(
        self,
        data_root: Path,
        session_metadata: Dict[str, Any],
        max_steps_per_file: int = 600,
    ) -> None:
        if max_steps_per_file <= 0:
            raise ValueError("max_steps_per_file must be positive")
        self.data_root = Path(data_root)
        self.session_metadata = dict(session_metadata)
        self.max_steps_per_file = int(max_steps_per_file)
        self.active = False
        self.session_dir: Optional[Path] = None
        self.latest_saved_path: Optional[Path] = None
        self._episode_index = 0
        self._episode_records: List[Dict[str, Any]] = []
        self._buffer: Dict[str, List[Any]] = {}
        self._reset_buffer()

    @property
    def buffered_steps(self) -> int:
        return len(self._buffer["action"])

    def _reset_buffer(self) -> None:
        self._buffer = {key: [] for key in self.ARRAY_KEYS}

    def _ensure_session(self) -> Path:
        if self.session_dir is not None:
            return self.session_dir
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        suffix = f"{os.getpid() % 10000:04d}"
        self.session_dir = self.data_root / f"session_{timestamp}_{suffix}"
        self.session_dir.mkdir(parents=True, exist_ok=False)
        metadata = {
            "format_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "storage": "compressed_npz_without_pickle",
            "transition_convention": "observation_t, action_t, reward_t, done_t",
            **self.session_metadata,
            "episodes": [],
        }
        _atomic_json_write(self.session_dir / "session.json", metadata)
        return self.session_dir

    def start(self) -> Path:
        path = self._ensure_session()
        self.active = True
        return path

    def stop(
        self,
        reason: str = "recording_stopped",
        final_info: Optional[Dict[str, Any]] = None,
    ) -> Optional[Path]:
        saved = self.flush(reason=reason, final_info=final_info)
        self.active = False
        return saved

    def record_transition(
        self,
        observation: Dict[str, np.ndarray],
        action: np.ndarray,
        reward: float,
        terminated: bool,
        truncated: bool,
        sim_time: float,
        privileged_state: np.ndarray,
        final_info: Optional[Dict[str, Any]] = None,
    ) -> Optional[Path]:
        if not self.active:
            return None
        self._ensure_session()
        self._buffer["image_left"].append(observation["image_left"].copy())
        self._buffer["image_right"].append(observation["image_right"].copy())
        self._buffer["proprio"].append(
            np.asarray(observation["proprio"], dtype=np.float32).copy(),
        )
        self._buffer["previous_action"].append(
            np.asarray(observation["previous_action"], dtype=np.float32).copy(),
        )
        self._buffer["action"].append(np.asarray(action, dtype=np.float32).copy())
        self._buffer["reward"].append(np.float32(reward))
        self._buffer["terminated"].append(bool(terminated))
        self._buffer["truncated"].append(bool(truncated))
        self._buffer["sim_time"].append(np.float32(sim_time))
        self._buffer["privileged_state"].append(
            np.asarray(privileged_state, dtype=np.float32).copy(),
        )

        if terminated or truncated:
            return self.flush(
                reason="terminated" if terminated else "truncated",
                final_info=final_info,
            )
        if self.buffered_steps >= self.max_steps_per_file:
            return self.flush(reason="chunk_limit", final_info=final_info)
        return None

    def flush(
        self,
        reason: str,
        final_info: Optional[Dict[str, Any]] = None,
    ) -> Optional[Path]:
        if self.buffered_steps == 0:
            return None
        session_dir = self._ensure_session()
        stem = f"episode_{self._episode_index:05d}"
        destination = session_dir / f"{stem}.npz"
        temporary = session_dir / f".{stem}.tmp.npz"

        arrays = {
            "image_left": np.stack(self._buffer["image_left"]).astype(np.uint8),
            "image_right": np.stack(self._buffer["image_right"]).astype(np.uint8),
            "proprio": np.stack(self._buffer["proprio"]).astype(np.float32),
            "previous_action": np.stack(self._buffer["previous_action"]).astype(
                np.float32,
            ),
            "action": np.stack(self._buffer["action"]).astype(np.float32),
            "reward": np.asarray(self._buffer["reward"], dtype=np.float32),
            "terminated": np.asarray(self._buffer["terminated"], dtype=np.bool_),
            "truncated": np.asarray(self._buffer["truncated"], dtype=np.bool_),
            "sim_time": np.asarray(self._buffer["sim_time"], dtype=np.float32),
            "privileged_state": np.stack(self._buffer["privileged_state"]).astype(
                np.float32,
            ),
        }
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, destination)

        info = final_info or {}
        episode_metadata = {
            "index": self._episode_index,
            "file": destination.name,
            "steps": self.buffered_steps,
            "return": float(arrays["reward"].sum()),
            "ended_reason": reason,
            "is_success": _json_scalar(info.get("is_success")),
            "captures": _json_scalar(info.get("captures")),
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }
        _atomic_json_write(session_dir / f"{stem}.json", episode_metadata)
        self._episode_records.append(episode_metadata)
        session_metadata_path = session_dir / "session.json"
        session_metadata = json.loads(session_metadata_path.read_text(encoding="utf-8"))
        session_metadata["episodes"] = self._episode_records
        _atomic_json_write(session_metadata_path, session_metadata)

        self.latest_saved_path = destination
        self._episode_index += 1
        self._reset_buffer()
        return destination


def make_environment(world_name: str = DEFAULT_WORLD):
    """Construct the interactive first-person environment after cache setup."""

    from botevade_gym import BotEvadeEnv, FirstPersonBotEvadeEnv

    return FirstPersonBotEvadeEnv(
        world_name=world_name,
        use_lppos=False,
        use_predator=True,
        max_step=1800,
        time_step=0.10,
        render=False,
        real_time=False,
        action_type=BotEvadeEnv.ActionType.CONTINUOUS,
        frame_stack_k=1,
        predator_prey_forward_speed_ratio=0.15,
        vision_width=192,
        vision_height=128,
        vision_fov=120.0,
        observation_mode="mouse",
        action_mode="egocentric_velocity_head",
        max_body_turn_rate=180.0,
        max_head_turn_rate=240.0,
        head_yaw_limit=60.0,
        render_mode="rgb_array",
    )


PRIVILEGED_STATE_NAMES = (
    "prey_x",
    "prey_y",
    "prey_vx",
    "prey_vy",
    "body_heading_degrees",
    "head_yaw_degrees",
    "predator_x",
    "predator_y",
    "predator_visible",
)


def snapshot_privileged_state(env) -> np.ndarray:
    model = env.unwrapped.model
    prey = model.prey
    prey_velocity = getattr(prey.state, "velocity", (0.0, 0.0))
    predator_x = math.nan
    predator_y = math.nan
    if getattr(model, "use_predator", False) and hasattr(model, "predator"):
        predator_x, predator_y = model.predator.state.location
    prey_data = getattr(model, "prey_data", None)
    predator_visible = float(bool(getattr(prey_data, "predator_visible", False)))
    return np.asarray(
        (
            prey.state.location[0],
            prey.state.location[1],
            prey_velocity[0],
            prey_velocity[1],
            env.body_heading_degrees,
            env.head_yaw_degrees,
            predator_x,
            predator_y,
            predator_visible,
        ),
        dtype=np.float32,
    )


def _load_font(size: int) -> pygame.font.Font:
    candidates = (
        Path("/System/Library/Fonts/STHeiti Medium.ttc"),
        Path("/System/Library/Fonts/Hiragino Sans GB.ttc"),
    )
    for candidate in candidates:
        if candidate.exists():
            try:
                return pygame.font.Font(str(candidate), size)
            except pygame.error:
                pass
    return pygame.font.Font(None, size)


class Button:
    def __init__(self, action: str, label: str, accent: Color = CYAN) -> None:
        self.action = action
        self.label = label
        self.accent = accent
        self.rect = pygame.Rect(0, 0, 10, 10)

    def draw(
        self,
        surface: pygame.Surface,
        font: pygame.font.Font,
        mouse_position: Tuple[int, int],
        active: bool = False,
    ) -> None:
        hovered = self.rect.collidepoint(mouse_position)
        fill = self.accent if active else (PANEL_LIGHT if hovered else PANEL)
        foreground = BACKGROUND if active else WHITE
        pygame.draw.rect(surface, fill, self.rect, border_radius=10)
        border = self.accent if not active else fill
        pygame.draw.rect(surface, border, self.rect, width=2, border_radius=10)
        text = font.render(self.label, True, foreground)
        surface.blit(text, text.get_rect(center=self.rect.center))


class MousePlayApp:
    WINDOW_SIZE = (1280, 760)
    CONTROL_HZ = 10.0

    def __init__(
        self,
        data_root: Path = DEFAULT_DATA_ROOT,
        world_name: str = DEFAULT_WORLD,
    ) -> None:
        pygame.init()
        pygame.display.set_caption("小鼠第一人称实验室 · Mouse First-Person Lab")
        icon_path = PROJECT_ROOT / "cellworld_game-main" / "cellworld_game" / "files" / "prey.png"
        if icon_path.exists():
            try:
                pygame.display.set_icon(pygame.image.load(str(icon_path)))
            except pygame.error:
                pass
        self.screen = pygame.display.set_mode(self.WINDOW_SIZE, pygame.RESIZABLE)
        self._activate_macos_window()
        self.clock = pygame.time.Clock()
        self.font_small = _load_font(17)
        self.font = _load_font(21)
        self.font_medium = _load_font(30)
        self.font_large = _load_font(54)

        self.data_root = Path(data_root)
        self.world_name = world_name
        self.running = True
        self.state = "loading"
        self.error_message = ""
        self.env = None
        self.observation: Optional[Dict[str, np.ndarray]] = None
        self.info: Dict[str, Any] = {}
        self.recorder: Optional[DemonstrationRecorder] = None
        self.paused = False
        self.episode_done = False
        self.episode_steps = 0
        self.episode_return = 0.0
        self.control_accumulator = 0.0
        self.left_surface: Optional[pygame.Surface] = None
        self.right_surface: Optional[pygame.Surface] = None
        self.flash_text = ""
        self.flash_color = WHITE
        self.flash_until = 0

        self.menu_buttons = {
            "play": Button("play", "开始试玩", CYAN),
            "collect": Button("collect", "开始并采集数据", RED),
            "quit": Button("quit", "退出", MUTED),
        }
        self.game_buttons = {
            "record": Button("record", "R  开始录制", RED),
            "reset": Button("reset", "N  新一局", CYAN),
            "pause": Button("pause", "P  暂停", AMBER),
            "menu": Button("menu", "M  主菜单", MUTED),
        }

    @staticmethod
    def _activate_macos_window() -> None:
        """Bring SDL's Python child process to the front when launched by .app."""

        if sys.platform != "darwin" or os.environ.get("SDL_VIDEODRIVER") == "dummy":
            return
        try:
            from AppKit import (
                NSApplication,
                NSApplicationActivationPolicyRegular,
            )

            application = NSApplication.sharedApplication()
            application.setActivationPolicy_(NSApplicationActivationPolicyRegular)
            application.activateIgnoringOtherApps_(True)
        except Exception:
            # The game remains usable if PyObjC is absent; it may simply open
            # behind the currently focused window when started from a shell.
            pass

    def _draw_loading(self, message: str) -> None:
        self.screen.fill(BACKGROUND)
        width, height = self.screen.get_size()
        title = self.font_large.render("Mouse First-Person Lab", True, WHITE)
        self.screen.blit(title, title.get_rect(center=(width // 2, height // 2 - 55)))
        detail = self.font.render(message, True, CYAN)
        self.screen.blit(detail, detail.get_rect(center=(width // 2, height // 2 + 25)))
        pygame.display.flip()
        pygame.event.pump()

    def _prepare(self) -> None:
        self._draw_loading("正在检查本地地图资源…")
        ensure_cellworld_cache(
            world_name=self.world_name,
            progress=self._draw_loading,
        )
        self.state = "menu"

    def _session_metadata(self) -> Dict[str, Any]:
        assert self.env is not None
        return {
            "environment": "FirstPersonBotEvadeEnv",
            "world_name": self.world_name,
            "control_hz": self.CONTROL_HZ,
            "action_mode": self.env.action_mode,
            "action_names": list(self.env.action_names),
            "action_range": [-1.0, 1.0],
            "proprio_names": list(self.env.proprio_names),
            "privileged_state_names": list(PRIVILEGED_STATE_NAMES),
            "image_layout": "HWC uint8 RGB, separate left/right eyes",
            "eye_shape": list(self.observation["image_left"].shape),
        }

    def _start_game(self, collect: bool) -> None:
        self._draw_loading("正在进入 Cellworld…")
        self._close_game(reason="restart")
        self.env = make_environment(self.world_name)
        observation, info = self.env.reset()
        if not self.env.observation_space.contains(observation):
            raise RuntimeError("Environment returned an invalid first-person observation")
        self.observation = observation
        self.info = info
        self.recorder = DemonstrationRecorder(
            data_root=self.data_root,
            session_metadata=self._session_metadata(),
            max_steps_per_file=int(self.CONTROL_HZ * 60),
        )
        if collect:
            session = self.recorder.start()
            self._flash(f"开始录制：{session.name}", RED, 3500)
        self.episode_done = False
        self.paused = False
        self.episode_steps = 0
        self.episode_return = 0.0
        self.control_accumulator = 0.0
        self._update_eye_surfaces()
        self.state = "play"

    def _close_game(self, reason: str = "app_exit") -> None:
        if self.recorder is not None:
            self.recorder.stop(reason=reason, final_info=self.info)
        if self.env is not None:
            self.env.close()
        self.env = None
        self.observation = None
        self.recorder = None

    def _return_to_menu(self) -> None:
        self._close_game(reason="returned_to_menu")
        self.state = "menu"

    def _reset_episode(self) -> None:
        if self.env is None:
            return
        if self.recorder is not None and self.recorder.active:
            self.recorder.flush(reason="manual_reset", final_info=self.info)
        self.observation, self.info = self.env.reset()
        self.episode_done = False
        self.paused = False
        self.episode_steps = 0
        self.episode_return = 0.0
        self.control_accumulator = 0.0
        self._update_eye_surfaces()
        self._flash("新一局开始", CYAN)

    def _toggle_recording(self) -> None:
        if self.recorder is None:
            return
        if self.recorder.active:
            saved = self.recorder.stop(
                reason="recording_stopped",
                final_info=self.info,
            )
            label = saved.name if saved is not None else "没有新帧"
            self._flash(f"录制已停止 · {label}", WHITE, 3000)
        else:
            session = self.recorder.start()
            self._flash(f"录制中 · {session.name}", RED, 3000)

    def _toggle_pause(self) -> None:
        if self.episode_done:
            return
        self.paused = not self.paused
        self.control_accumulator = 0.0
        self._flash("已暂停" if self.paused else "继续", AMBER)

    def _flash(self, text: str, color: Color = WHITE, duration_ms: int = 1800) -> None:
        self.flash_text = text
        self.flash_color = color
        self.flash_until = pygame.time.get_ticks() + duration_ms

    def _current_action(self) -> np.ndarray:
        keys = pygame.key.get_pressed()
        forward = float(keys[pygame.K_UP]) - float(keys[pygame.K_DOWN])
        body_turn = float(keys[pygame.K_LEFT]) - float(keys[pygame.K_RIGHT])
        head_turn = float(keys[pygame.K_a]) - float(keys[pygame.K_d])
        if keys[pygame.K_SPACE]:
            forward = 0.0
        return np.asarray((forward, body_turn, head_turn), dtype=np.float32)

    def _step_game(self) -> None:
        if self.env is None or self.observation is None or self.episode_done:
            return
        action = self._current_action()
        observation_t = self.observation
        state_t = snapshot_privileged_state(self.env)
        sim_time_t = float(self.env.unwrapped.model.time)
        next_observation, reward, terminated, truncated, info = self.env.step(action)
        self.observation = next_observation
        self.info = info
        self.episode_steps += 1
        self.episode_return += float(reward)

        if self.recorder is not None:
            saved = self.recorder.record_transition(
                observation=observation_t,
                action=action,
                reward=float(reward),
                terminated=bool(terminated),
                truncated=bool(truncated),
                sim_time=sim_time_t,
                privileged_state=state_t,
                final_info=info,
            )
            if saved is not None and not (terminated or truncated):
                self._flash(f"已自动保存 · {saved.name}", GREEN, 2400)

        self._update_eye_surfaces()
        if terminated or truncated:
            self.episode_done = True
            success = bool(info.get("is_success", False))
            if success:
                self._flash("到达目标！按 N 再来一局", GREEN, 8000)
            else:
                self._flash("本局结束 · 按 N 再来一局", AMBER, 8000)

    def _update_eye_surfaces(self) -> None:
        if self.observation is None:
            return
        left = np.transpose(self.observation["image_left"], (1, 0, 2))
        right = np.transpose(self.observation["image_right"], (1, 0, 2))
        self.left_surface = pygame.surfarray.make_surface(left)
        self.right_surface = pygame.surfarray.make_surface(right)

    def _handle_key(self, key: int) -> None:
        if self.state == "menu":
            if key in (pygame.K_RETURN, pygame.K_SPACE):
                self._start_game(collect=False)
            elif key == pygame.K_r:
                self._start_game(collect=True)
            elif key in (pygame.K_ESCAPE, pygame.K_q):
                self.running = False
            return

        if self.state != "play":
            if key in (pygame.K_ESCAPE, pygame.K_q):
                self.running = False
            return
        if key == pygame.K_r:
            self._toggle_recording()
        elif key == pygame.K_n:
            self._reset_episode()
        elif key == pygame.K_p:
            self._toggle_pause()
        elif key in (pygame.K_m, pygame.K_ESCAPE):
            self._return_to_menu()
        elif key == pygame.K_q:
            self.running = False

    def _handle_click(self, position: Tuple[int, int]) -> None:
        if self.state == "menu":
            for button in self.menu_buttons.values():
                if button.rect.collidepoint(position):
                    if button.action == "play":
                        self._start_game(collect=False)
                    elif button.action == "collect":
                        self._start_game(collect=True)
                    elif button.action == "quit":
                        self.running = False
                    return
        elif self.state == "play":
            for button in self.game_buttons.values():
                if not button.rect.collidepoint(position):
                    continue
                if button.action == "record":
                    self._toggle_recording()
                elif button.action == "reset":
                    self._reset_episode()
                elif button.action == "pause":
                    self._toggle_pause()
                elif button.action == "menu":
                    self._return_to_menu()
                return

    @staticmethod
    def _fit_rect(container: pygame.Rect, source_size: Tuple[int, int]) -> pygame.Rect:
        scale = min(
            container.width / source_size[0],
            container.height / source_size[1],
        )
        size = (
            max(1, int(round(source_size[0] * scale))),
            max(1, int(round(source_size[1] * scale))),
        )
        result = pygame.Rect(0, 0, *size)
        result.center = container.center
        return result

    def _draw_menu(self) -> None:
        self.screen.fill(BACKGROUND)
        width, height = self.screen.get_size()
        pygame.draw.circle(self.screen, (22, 55, 61), (width // 2, 135), 175)
        title = self.font_large.render("小鼠第一人称实验室", True, WHITE)
        self.screen.blit(title, title.get_rect(center=(width // 2, 105)))
        subtitle = self.font.render(
            "双眼视角 · 键盘控制 · VLA 人类示范采集",
            True,
            CYAN,
        )
        self.screen.blit(subtitle, subtitle.get_rect(center=(width // 2, 158)))

        button_width = min(420, width - 80)
        button_height = 58
        top = max(250, height // 2 - 95)
        for index, name in enumerate(("play", "collect", "quit")):
            button = self.menu_buttons[name]
            button.rect = pygame.Rect(
                width // 2 - button_width // 2,
                top + index * 76,
                button_width,
                button_height,
            )
            button.draw(self.screen, self.font, pygame.mouse.get_pos())

        controls = "方向键：移动/转向    A / D：探头    R：随时开关录制"
        text = self.font_small.render(controls, True, MUTED)
        self.screen.blit(text, text.get_rect(center=(width // 2, height - 48)))

    def _draw_eye(
        self,
        eye_surface: pygame.Surface,
        container: pygame.Rect,
        label: str,
    ) -> None:
        pygame.draw.rect(self.screen, (7, 10, 12), container, border_radius=12)
        target = self._fit_rect(container.inflate(-12, -12), eye_surface.get_size())
        scaled = pygame.transform.smoothscale(eye_surface, target.size)
        self.screen.blit(scaled, target)
        pygame.draw.rect(self.screen, PANEL_LIGHT, container, width=2, border_radius=12)
        label_surface = self.font_small.render(label, True, WHITE)
        label_background = label_surface.get_rect(topleft=(container.x + 14, container.y + 12))
        label_background.inflate_ip(14, 8)
        pygame.draw.rect(self.screen, (12, 17, 20), label_background, border_radius=7)
        self.screen.blit(label_surface, (container.x + 21, container.y + 16))

    def _draw_action_meter(
        self,
        rect: pygame.Rect,
        label: str,
        value: float,
        color: Color,
    ) -> None:
        text = self.font_small.render(label, True, MUTED)
        self.screen.blit(text, (rect.x, rect.y - 22))
        pygame.draw.rect(self.screen, PANEL_LIGHT, rect, border_radius=5)
        centre = rect.centerx
        extent = int((rect.width / 2 - 2) * min(abs(value), 1.0))
        if value >= 0:
            value_rect = pygame.Rect(centre, rect.y + 2, extent, rect.height - 4)
        else:
            value_rect = pygame.Rect(centre - extent, rect.y + 2, extent, rect.height - 4)
        if extent:
            pygame.draw.rect(self.screen, color, value_rect, border_radius=4)
        pygame.draw.line(self.screen, WHITE, (centre, rect.y), (centre, rect.bottom), 1)

    def _draw_game(self) -> None:
        self.screen.fill(BACKGROUND)
        width, height = self.screen.get_size()
        recording = bool(self.recorder is not None and self.recorder.active)

        # Top status bar.
        pygame.draw.rect(self.screen, PANEL, (0, 0, width, 78))
        title = self.font_medium.render("Mouse First-Person Lab", True, WHITE)
        self.screen.blit(title, (24, 18))
        status_text = "●  正在录制" if recording else "○  未录制"
        status_color = RED if recording else MUTED
        status = self.font.render(status_text, True, status_color)
        self.screen.blit(status, (width - status.get_width() - 24, 25))

        # Two separate eye images; they are never geometrically squashed.
        view_top = 92
        view_bottom = max(view_top + 160, height - 202)
        view_height = view_bottom - view_top
        gap = 14
        margin = 22
        eye_width = max(100, (width - 2 * margin - gap) // 2)
        left_rect = pygame.Rect(margin, view_top, eye_width, view_height)
        right_rect = pygame.Rect(margin + eye_width + gap, view_top, eye_width, view_height)
        if self.left_surface is not None and self.right_surface is not None:
            self._draw_eye(self.left_surface, left_rect, "左眼  LEFT EYE")
            self._draw_eye(self.right_surface, right_rect, "右眼  RIGHT EYE")

        # Bottom HUD and live action meters.
        hud_top = view_bottom + 12
        pygame.draw.rect(self.screen, PANEL, (0, hud_top, width, height - hud_top))
        action = self._current_action() if not self.paused else np.zeros(3, dtype=np.float32)
        meter_width = min(180, max(110, (width - 640) // 3))
        meter_y = hud_top + 55
        self._draw_action_meter(
            pygame.Rect(24, meter_y, meter_width, 14),
            "前进 / 后退  ↑ ↓",
            float(action[0]),
            GREEN,
        )
        self._draw_action_meter(
            pygame.Rect(44 + meter_width, meter_y, meter_width, 14),
            "身体转向  ← →",
            float(action[1]),
            CYAN,
        )
        self._draw_action_meter(
            pygame.Rect(64 + 2 * meter_width, meter_y, meter_width, 14),
            "头部视线  A / D",
            float(action[2]),
            AMBER,
        )

        goal_distance = math.nan
        if self.env is not None:
            prey_data = getattr(self.env.unwrapped.model, "prey_data", None)
            goal_distance = float(getattr(prey_data, "prey_goal_distance", math.nan))
        info_text = (
            f"步数 {self.episode_steps}    "
            f"目标距离 {goal_distance:.3f}    "
            f"回报 {self.episode_return:.2f}"
        )
        info_surface = self.font_small.render(info_text, True, MUTED)
        self.screen.blit(info_surface, (24, hud_top + 102))

        # Clickable controls on the right side of the HUD.
        button_width = 132
        button_height = 40
        total_width = button_width * 4 + 10 * 3
        start_x = max(24, width - total_width - 24)
        for index, name in enumerate(("record", "reset", "pause", "menu")):
            button = self.game_buttons[name]
            if name == "record":
                button.label = "R  停止录制" if recording else "R  开始录制"
            elif name == "pause":
                button.label = "P  继续" if self.paused else "P  暂停"
            button.rect = pygame.Rect(
                start_x + index * (button_width + 10),
                hud_top + 25,
                button_width,
                button_height,
            )
            button.draw(
                self.screen,
                self.font_small,
                pygame.mouse.get_pos(),
                active=(name == "record" and recording),
            )

        if self.recorder is not None and self.recorder.session_dir is not None:
            try:
                display_path = self.recorder.session_dir.relative_to(PROJECT_ROOT)
            except ValueError:
                display_path = self.recorder.session_dir
            path_surface = self.font_small.render(f"数据：{display_path}", True, MUTED)
            self.screen.blit(path_surface, (start_x, hud_top + 82))

        if self.paused or self.episode_done:
            overlay = pygame.Surface((width, view_height), pygame.SRCALPHA)
            overlay.fill((4, 8, 10, 160))
            self.screen.blit(overlay, (0, view_top))
            message = "已暂停 · 按 P 继续" if self.paused else "本局结束 · 按 N 重新开始"
            message_surface = self.font_medium.render(message, True, WHITE)
            self.screen.blit(
                message_surface,
                message_surface.get_rect(center=(width // 2, view_top + view_height // 2)),
            )

        if self.flash_text and pygame.time.get_ticks() < self.flash_until:
            message = self.font.render(self.flash_text, True, self.flash_color)
            box = message.get_rect(center=(width // 2, 55))
            background_rect = box.inflate(28, 14)
            pygame.draw.rect(self.screen, (8, 12, 14), background_rect, border_radius=10)
            self.screen.blit(message, box)

    def _draw_error(self) -> None:
        self.screen.fill((32, 12, 14))
        width, height = self.screen.get_size()
        title = self.font_large.render("启动失败", True, RED)
        self.screen.blit(title, title.get_rect(center=(width // 2, 110)))
        lines = self.error_message.splitlines() or ["Unknown error"]
        y = 190
        for line in lines[:12]:
            rendered = self.font_small.render(line[:120], True, WHITE)
            self.screen.blit(rendered, rendered.get_rect(center=(width // 2, y)))
            y += 28
        hint = self.font.render("按 Escape 退出", True, MUTED)
        self.screen.blit(hint, hint.get_rect(center=(width // 2, height - 55)))

    def run(self) -> None:
        try:
            self._prepare()
            previous_time = time.perf_counter()
            while self.running:
                current_time = time.perf_counter()
                delta_time = min(current_time - previous_time, 0.25)
                previous_time = current_time

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.running = False
                    elif event.type == pygame.KEYDOWN:
                        self._handle_key(event.key)
                    elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                        self._handle_click(event.pos)

                if self.state == "play" and not self.paused and not self.episode_done:
                    self.control_accumulator += delta_time
                    control_period = 1.0 / self.CONTROL_HZ
                    steps_this_frame = 0
                    while self.control_accumulator >= control_period and steps_this_frame < 3:
                        self._step_game()
                        self.control_accumulator -= control_period
                        steps_this_frame += 1
                    if steps_this_frame == 3:
                        self.control_accumulator = 0.0

                if self.state == "menu":
                    self._draw_menu()
                elif self.state == "play":
                    self._draw_game()
                else:
                    self._draw_error()
                pygame.display.flip()
                self.clock.tick(60)
        except Exception:
            self.error_message = traceback.format_exc()
            self.state = "error"
            log_path = PROJECT_ROOT / "logs" / "mouse_play_app_error.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text(self.error_message, encoding="utf-8")
            while self.running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT or (
                        event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
                    ):
                        self.running = False
                self._draw_error()
                pygame.display.flip()
                self.clock.tick(30)
        finally:
            self._close_game(reason="app_exit")
            pygame.quit()


def run_smoke_test(data_root: Path, world_name: str) -> Path:
    """Headless integration check for environment stepping and dataset output."""

    ensure_cellworld_cache(world_name=world_name)
    env = make_environment(world_name)
    try:
        observation, info = env.reset(seed=23)
        recorder = DemonstrationRecorder(
            data_root=data_root,
            session_metadata={
                "environment": "FirstPersonBotEvadeEnv",
                "world_name": world_name,
                "control_hz": MousePlayApp.CONTROL_HZ,
                "action_names": list(env.action_names),
                "proprio_names": list(env.proprio_names),
                "privileged_state_names": list(PRIVILEGED_STATE_NAMES),
                "eye_shape": list(observation["image_left"].shape),
                "smoke_test": True,
            },
            max_steps_per_file=20,
        )
        recorder.start()
        actions = (
            (0.6, 0.0, 0.0),
            (0.6, 0.5, 0.0),
            (0.0, 0.0, 1.0),
            (0.0, 0.0, 0.0),
        )
        for action_values in actions:
            action = np.asarray(action_values, dtype=np.float32)
            state = snapshot_privileged_state(env)
            sim_time = float(env.unwrapped.model.time)
            next_observation, reward, terminated, truncated, info = env.step(action)
            recorder.record_transition(
                observation=observation,
                action=action,
                reward=reward,
                terminated=terminated,
                truncated=truncated,
                sim_time=sim_time,
                privileged_state=state,
                final_info=info,
            )
            observation = next_observation
            if terminated or truncated:
                break
        saved = recorder.stop(reason="smoke_test", final_info=info)
        if saved is None or not saved.exists():
            raise RuntimeError("Smoke test did not produce a dataset file")
        with np.load(saved, allow_pickle=False) as episode:
            if episode["action"].shape[1:] != (3,):
                raise RuntimeError("Recorded action schema is invalid")
            if episode["image_left"].dtype != np.uint8:
                raise RuntimeError("Recorded image dtype is invalid")
        return saved
    finally:
        env.close()


def render_ui_smoke_screenshot(
    output_path: Path,
    data_root: Path,
    world_name: str,
) -> Path:
    """Render the actual gameplay HUD with a real environment for visual QA."""

    app = MousePlayApp(data_root=data_root, world_name=world_name)
    try:
        app._prepare()
        app._start_game(collect=True)
        assert app.env is not None and app.observation is not None
        for action_values in ((0.5, 0.0, 0.0), (0.5, 0.4, -0.3)):
            observation_t = app.observation
            state_t = snapshot_privileged_state(app.env)
            sim_time_t = float(app.env.unwrapped.model.time)
            action = np.asarray(action_values, dtype=np.float32)
            app.observation, reward, terminated, truncated, app.info = app.env.step(
                action,
            )
            app.episode_steps += 1
            app.episode_return += float(reward)
            assert app.recorder is not None
            app.recorder.record_transition(
                observation=observation_t,
                action=action,
                reward=reward,
                terminated=terminated,
                truncated=truncated,
                sim_time=sim_time_t,
                privileged_state=state_t,
                final_info=app.info,
            )
        app._update_eye_surfaces()
        app._draw_game()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pygame.image.save(app.screen, str(output_path))
        return output_path
    finally:
        app._close_game(reason="ui_smoke_screenshot")
        pygame.quit()


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world", default=DEFAULT_WORLD)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--prepare-cache", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--ui-screenshot", type=Path)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.prepare_cache:
        ensure_cellworld_cache(world_name=args.world)
        print(f"Cellworld cache ready: {DEFAULT_CACHE_ROOT}")
        return 0
    if args.smoke_test:
        saved = run_smoke_test(data_root=args.data_root, world_name=args.world)
        print(f"Mouse play app smoke test saved: {saved}")
        return 0
    if args.ui_screenshot:
        screenshot = render_ui_smoke_screenshot(
            output_path=args.ui_screenshot,
            data_root=args.data_root,
            world_name=args.world,
        )
        print(f"Mouse play app UI screenshot saved: {screenshot}")
        return 0
    app = MousePlayApp(data_root=args.data_root, world_name=args.world)
    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
