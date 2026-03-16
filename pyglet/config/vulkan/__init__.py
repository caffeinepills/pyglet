from __future__ import annotations

from dataclasses import asdict
from typing import TYPE_CHECKING

from pyglet.config import SurfaceConfig, UserConfig, VulkanConfig

if TYPE_CHECKING:
    from pyglet.window import Window


class VulkanSurfaceConfig(SurfaceConfig):
    config: VulkanConfig

    def __init__(self, window: Window, config: VulkanConfig, handle: object | None = None) -> None:
        super().__init__(window, config, handle)
        for name, value in asdict(config).items():
            setattr(self, name, value)


def get_surface_config(user_config: UserConfig, surface: Window) -> SurfaceConfig | None:
    if not isinstance(user_config, VulkanConfig):
        return None
    return VulkanSurfaceConfig(surface, user_config, handle=None)
