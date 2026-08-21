from __future__ import annotations

from typing import TYPE_CHECKING

from pyglet.enums import GraphicsAPI
from pyglet.graphics.api.base import SurfaceFeatures, SurfaceInfo
from pyglet.libs.shared.vulkan_lib.vulkan_core import (
    VK_API_VERSION_MAJOR,
    VK_API_VERSION_MINOR,
    VK_API_VERSION_PATCH,
    VK_SAMPLE_COUNT_1_BIT,
    VK_SAMPLE_COUNT_2_BIT,
    VK_SAMPLE_COUNT_4_BIT,
    VK_SAMPLE_COUNT_8_BIT,
    VK_SAMPLE_COUNT_16_BIT,
    VK_SAMPLE_COUNT_32_BIT,
    VK_SAMPLE_COUNT_64_BIT,
)

if TYPE_CHECKING:
    from pyglet.graphics.api.vulkan.instance import VulkanSurfaceContext


class VulkanInfo(SurfaceInfo):
    """Information interface for a single Vulkan context/surface."""

    def __init__(self) -> None:
        super().__init__()
        self._physical_features = None

    @staticmethod
    def _max_sample_count(sample_flags: int) -> int:
        if sample_flags & VK_SAMPLE_COUNT_64_BIT:
            return 64
        if sample_flags & VK_SAMPLE_COUNT_32_BIT:
            return 32
        if sample_flags & VK_SAMPLE_COUNT_16_BIT:
            return 16
        if sample_flags & VK_SAMPLE_COUNT_8_BIT:
            return 8
        if sample_flags & VK_SAMPLE_COUNT_4_BIT:
            return 4
        if sample_flags & VK_SAMPLE_COUNT_2_BIT:
            return 2
        if sample_flags & VK_SAMPLE_COUNT_1_BIT:
            return 1
        return 0

    def query(self, context: VulkanSurfaceContext) -> None:
        physical = context.devices.physical_device
        props = physical.properties
        limits = props.limits

        self.api = GraphicsAPI.VULKAN.value
        self.vendor = f"0x{props.vendorID:04X}"
        self.renderer = physical.name

        api_version = int(props.apiVersion)
        major = VK_API_VERSION_MAJOR(api_version)
        minor = VK_API_VERSION_MINOR(api_version)
        patch = VK_API_VERSION_PATCH(api_version)
        self.major_version = major
        self.minor_version = minor
        self.version = f"{major}.{minor}.{patch}"

        self.shading_language_version = "SPIR-V"
        self.extensions = physical.get_extensions()

        self.MAX_ARRAY_TEXTURE_LAYERS = int(limits.maxImageArrayLayers)
        self.MAX_TEXTURE_SIZE = int(limits.maxImageDimension2D)
        self.MAX_COLOR_ATTACHMENTS = int(limits.maxColorAttachments)
        self.MAX_SAMPLES = self._max_sample_count(int(limits.framebufferColorSampleCounts))
        self.MAX_COLOR_TEXTURE_SAMPLES = self._max_sample_count(int(limits.sampledImageColorSampleCounts))
        self.MAX_TEXTURE_IMAGE_UNITS = int(limits.maxPerStageDescriptorSampledImages)
        self.MAX_COMBINED_TEXTURE_IMAGE_UNITS = int(limits.maxDescriptorSetSampledImages)
        self.MAX_UNIFORM_BUFFER_BINDINGS = int(limits.maxPerStageDescriptorUniformBuffers)
        self.MAX_UNIFORM_BLOCK_SIZE = int(limits.maxUniformBufferRange)
        self.MAX_VERTEX_ATTRIBS = int(limits.maxVertexInputAttributes)
        self.MAX_UNIFORM_BUFFER_OFFSET_ALIGNMENT = int(limits.minUniformBufferOffsetAlignment)

        self._physical_features = physical.features
        self.update_features()
        self.was_queried = True

    def update_features(self) -> None:
        """Populate Vulkan feature support for the selected physical device."""
        physical_features = self._physical_features
        self.features = SurfaceFeatures(
            # These operations are provided by the Vulkan core API and the
            # queues required when creating this backend's logical device.
            compute_shaders=True,
            shader_storage_buffers=True,
            uniform_buffers=True,
            sync_objects=True,
            geometry_shaders=bool(physical_features and physical_features.geometryShader),
            tessellation_shaders=bool(physical_features and physical_features.tessellationShader),
            base_vertex=True,
            persistent_buffers=True,
            separate_shader_objects=True,
            pixel_buffer_objects=True,
            texture_storage=True,
        )
