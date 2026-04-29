from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

from pyglet.config import SurfaceConfig, UserConfig
from pyglet.enums import GraphicsAPI

if TYPE_CHECKING:
    from pyglet.window import Window



# Platform Specific Configurations

@dataclass
class VulkanUserConfig(UserConfig):
    """An OpenGL Graphics configuration."""
    #: Specify the presence of a back-buffer for every color buffer.
    double_buffer: bool | None = None
    #: Specify the presence of separate left and right buffer sets.
    stereo: bool | None = None
    #: Total bits per sample per color buffer.
    buffer_size: int | None = None
    #: The number of auxiliary color buffers.
    aux_buffers: int | None = None
    #: The number of multisample buffers.
    sample_buffers: int | None = None
    #: The number of samples per pixel, or 0 if there are no multisample buffers.
    samples: int | None = None
    #: Bits per sample per buffer devoted to the red component.
    red_size: int | None = None
    #: Bits per sample per buffer devoted to the green component.
    green_size: int | None = None
    #: Bits per sample per buffer devoted to the blue component.
    blue_size: int | None = None
    #: Bits per sample per buffer devoted to the alpha component.
    alpha_size: int | None = None
    #: Bits per sample in the depth buffer.
    depth_size: int | None = None
    #: Bits per sample in the stencil buffer.
    stencil_size: int | None = None
    #: Bits per pixel devoted to the red component in the accumulation buffer. Deprecated.
    accum_red_size: int | None = None
    #: Bits per pixel devoted to the green component in the accumulation buffer. Deprecated.
    accum_green_size: int | None = None
    #: Bits per pixel devoted to the blue component in the accumulation buffer. Deprecated.
    accum_blue_size: int | None = None
    #: Bits per pixel devoted to the alpha component in the accumulation buffer. Deprecated.
    accum_alpha_size: int | None = None
    #: The OpenGL major version.
    major_version: int | None = None
    #: The OpenGL minor version.
    minor_version: int | None = None
    #: Whether to use forward compatibility mode.
    forward_compatible: bool | None = None
    #: Debug mode.
    debug: bool | None = None
    #: If the framebuffer should be transparent.
    transparent_framebuffer: bool | None = None
    #: Which rendering API is being used (GL, ES, etc.).
    api: GraphicsAPI = GraphicsAPI.VULKAN

    @property
    def is_finalized(self) -> bool:
        return False

class VulkanSurfaceConfig(SurfaceConfig):
    config: VulkanUserConfig

    def __init__(self, window: Window, config: VulkanUserConfig, handle: object | None = None) -> None:
        super().__init__(window, config, handle)
        for name, value in asdict(config).items():
            setattr(self, name, value)


def get_surface_config(user_config: UserConfig, surface: Window) -> SurfaceConfig | None:
    if not isinstance(user_config, VulkanUserConfig):
        return None
    return VulkanSurfaceConfig(surface, user_config, handle=None)
