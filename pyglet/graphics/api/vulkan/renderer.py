from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pyglet.graphics.api.base import BackendRenderer
from pyglet.graphics.api.vulkan import DeviceFunc, c_array_list
from pyglet.libs.shared.vulkan_lib.vulkan_core import VkExtent2D, VkOffset2D, VkRect2D, VkViewport

if TYPE_CHECKING:
    from pyglet.graphics.api.vulkan.instance import VulkanSurfaceContext


class VulkanRenderer(BackendRenderer):
    """Vulkan renderer helpers for draw-context state transitions."""

    surface_ctx: VulkanSurfaceContext

    def __init__(self, surface_ctx: VulkanSurfaceContext) -> None:
        super().__init__(surface_ctx)

    def _active_command_buffer(self):
        return self.surface_ctx.frame_context.backend_ctx.command_buffer

    def set_viewport(self, x: int, y: int, width: int, height: int) -> None:
        command_buffer = self._active_command_buffer()
        if command_buffer is None:
            return

        viewport = VkViewport(float(x), float(y), float(width), float(height), 0.0, 1.0)
        viewports = c_array_list([viewport], VkViewport)
        DeviceFunc.vkCmdSetViewport(command_buffer, 0, 1, viewports)

    def set_scissor(self, scissor: Any | None) -> None:
        command_buffer = self._active_command_buffer()
        if command_buffer is None:
            return

        if scissor is None:
            extent = self.surface_ctx.swapchain.extent
            rect = VkRect2D(offset=VkOffset2D(x=0, y=0), extent=extent)
        else:
            x, y, width, height = scissor.area
            rect = VkRect2D(
                offset=VkOffset2D(x=int(x), y=int(y)),
                extent=VkExtent2D(max(0, int(width)), max(0, int(height))),
            )

        rects = c_array_list([rect], VkRect2D)
        DeviceFunc.vkCmdSetScissor(command_buffer, 0, 1, rects)

    def set_clear_color(self, r: float, g: float, b: float, a: float) -> None:
        self.surface_ctx.clear_color = (r, g, b, a)
