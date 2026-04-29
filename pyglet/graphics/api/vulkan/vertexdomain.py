from __future__ import annotations

import ctypes
from typing import TYPE_CHECKING, Any, Sequence

import pyglet
from pyglet.graphics.api.base import SurfaceContext
from pyglet.graphics.api.vulkan import DeviceFunc
from pyglet.graphics.api.vulkan.buffer import AttributeBufferObject, IndexBufferObject
from pyglet.graphics.api.vulkan.shader import get_vulkan_format
from pyglet.graphics.vertexdomain import (
    InstanceStream,
    IndexStream,
    IndexedVertexDomain as BaseIndexedVertexDomain,
    IndexedVertexList as BaseIndexedVertexList,
    VertexArrayBinding,
    VertexArrayProtocol,
    VertexDomain as BaseVertexDomain,
    VertexList as BaseVertexList,
    VertexStream,
    _RunningIndexSupport,
)
from pyglet.libs.shared.vulkan_lib.vulkan_core import (
    VkBuffer,
    VkVertexInputAttributeDescription,
    VkVertexInputBindingDescription,
)

if TYPE_CHECKING:
    from pyglet.customtypes import DataTypes
    from pyglet.enums import GeometryMode
    from pyglet.graphics.shader import Attribute
    from pyglet.graphics import Group
    from pyglet.graphics.api.vulkan.instance import VulkanSurfaceContext


class _NullVAO:
    def bind(self) -> None:
        pass

    def unbind(self) -> None:
        pass


class _VulkanAttribute:
    """Adapter for shared `Attribute` -> Vulkan buffer attribute metadata."""

    __slots__ = (
        "name",
        "location",
        "count",
        "data_type",
        "normalized",
        "instance",
        "c_type",
        "element_size",
        "stride",
        "vk_format",
    )

    def __init__(self, attribute: Attribute) -> None:
        self.name = attribute.fmt.name
        self.location = attribute.location
        self.count = attribute.fmt.components
        self.data_type = attribute.fmt.data_type
        self.normalized = attribute.fmt.normalized
        self.instance = attribute.fmt.is_instanced
        self.c_type = attribute.c_type
        self.element_size = attribute.element_size
        self.stride = self.count * self.element_size

        vk_format = get_vulkan_format(self.count, self.data_type, self.normalized)
        if vk_format is None:
            msg = (
                f"Invalid Vulkan attribute format for '{self.name}': "
                f"components={self.count}, type='{self.data_type}', normalized={self.normalized}."
            )
            raise ValueError(msg)
        self.vk_format = vk_format


class VulkanAttributeBufferObject(AttributeBufferObject):
    @property
    def stride(self) -> int:
        return self.attribute.stride

    @property
    def element_count(self) -> int:
        return self.attribute.count

    def get_region(self, start: int, count: int):
        return self.get_data_region_no_copy(start, count)

    def set_region(self, start: int, count: int, data: Sequence[float | int]) -> None:  # noqa: ARG002
        self.set_data_region(start, data)

    def invalidate_region(self, start: int, count: int) -> None:  # noqa: ARG002
        pass

    def commit(self) -> None:
        pass


class VulkanIndexBufferObject(IndexBufferObject):
    stride = ctypes.sizeof(ctypes.c_uint16)

    def get_region(self, start: int, count: int):
        return self.get_data_region(start, count)

    def set_region(self, start: int, count: int, data: Sequence[int | float]) -> None:  # noqa: ARG002
        self.set_data_region(start, data)

    def copy_region(self, dst: int, src: int, count: int) -> None:
        self.set_region(dst, count, self.get_region(src, count))

    def bind_to_index_buffer(self) -> None:
        pass

    def commit(self) -> None:
        pass


class VulkanVertexStream(VertexStream):
    _ctx: VulkanSurfaceContext
    attrib_name_buffers: dict[str, VulkanAttributeBufferObject]

    def __init__(
        self,
        ctx: VulkanSurfaceContext,
        devices,
        initial_size: int,
        attrs: Sequence[Attribute],
        *,
        divisor: int = 0,
    ) -> None:
        self._devices = devices
        self._binding_desc: list[VkVertexInputBindingDescription] = []
        self._attrib_desc: list[VkVertexInputAttributeDescription] = []
        super().__init__(ctx, initial_size, attrs, divisor=divisor)
        self._refresh_descriptors()

    def get_graphics_attribute(self, attribute: Attribute, _view):
        return _VulkanAttribute(attribute)

    def get_buffer(self, _size: int, attribute: _VulkanAttribute) -> VulkanAttributeBufferObject:
        index = len(self.buffers)
        buffer = VulkanAttributeBufferObject(
            attribute.data_type,
            self._devices,
            index,
            attribute,
            self.allocator.capacity,
        )
        buffer.create(self._devices)
        return buffer

    def bind_into(self, _vao) -> None:
        pass

    def bind(self, command_buffer, binding_start: int = 0) -> None:
        if not self.buffers:
            return
        buffers = [buffer.buffer.vk_buffer for buffer in self.buffers]
        offsets = [0] * len(buffers)
        buffer_array = (VkBuffer * len(buffers))(*buffers)
        offset_array = (ctypes.c_uint64 * len(offsets))(*offsets)
        DeviceFunc.vkCmdBindVertexBuffers(command_buffer, binding_start, len(buffers), buffer_array, offset_array)

    def _refresh_descriptors(self) -> None:
        self._binding_desc = [buffer.bind_descriptor for buffer in self.buffers]
        self._attrib_desc = [
            VkVertexInputAttributeDescription(
                location=buffer.attribute.location,
                binding=buffer.index,
                format=buffer.attribute.vk_format,
                offset=0,
            )
            for buffer in self.buffers
        ]

    def get_binding_descriptions(self) -> list[VkVertexInputBindingDescription]:
        return self._binding_desc

    def get_attribute_descriptions(self) -> list[VkVertexInputAttributeDescription]:
        return self._attrib_desc

    def delete(self) -> None:
        for buffer in self.buffers:
            buffer.delete()


class VulkanIndexStream(IndexStream):
    index_element_size: int

    def __init__(self, ctx: VulkanSurfaceContext, devices, data_type: DataTypes, initial_elems: int) -> None:
        if data_type != "H":
            msg = f"Vulkan currently supports only uint16 index buffers. Received '{data_type}'."
            raise ValueError(msg)
        self._devices = devices
        self.index_element_size = ctypes.sizeof(ctypes.c_uint16)
        super().__init__(ctx, data_type, initial_elems)

    def _create_buffer(self) -> VulkanIndexBufferObject:
        buffer = VulkanIndexBufferObject(self.allocator.capacity * self.index_element_size)
        buffer.create(self._devices)
        return buffer

    def bind_into(self, _vao) -> None:
        pass

    def bind(self, command_buffer) -> None:
        self.buffer.bind(command_buffer)

    def delete(self) -> None:
        self.buffer.delete()


class VulkanVertexArrayBinding(VertexArrayBinding):
    streams: list[VulkanVertexStream | VulkanIndexStream]

    def _create_vao(self) -> VertexArrayProtocol:
        return _NullVAO()

    def _link(self) -> None:
        pass

    def bind(self) -> None:
        pass

    def unbind(self) -> None:
        pass


class VulkanVertexList(BaseVertexList):
    domain: VertexDomain

    def __init__(self, domain: VertexDomain, group: Group, start: int, count: int) -> None:  # noqa: D107
        super().__init__(domain, group, start, count)

    def set_attribute_data(self, name: str, data: Any) -> None:
        stream = self.domain.attrib_name_buffers[name]
        buffer = stream.attrib_name_buffers[name]
        buffer.set_region(self.start, self.count, data)


class VulkanIndexedVertexList(BaseIndexedVertexList):
    domain: IndexedVertexDomain

    def __init__(
        self,
        domain: IndexedVertexDomain,
        group: Group,
        start: int,
        count: int,
        index_start: int,
        index_count: int,
    ) -> None:  # noqa: D107
        super().__init__(domain, group, start, count, index_start, index_count)

    def set_attribute_data(self, name: str, data: Any) -> None:
        stream = self.domain.attrib_name_buffers[name]
        buffer = stream.attrib_name_buffers[name]
        buffer.set_region(self.start, self.count, data)


# Preserve the conventional module-level names used elsewhere.
VertexList = VulkanVertexList
IndexedVertexList = VulkanIndexedVertexList


class VertexDomain(BaseVertexDomain):
    _vertex_class = VulkanVertexList
    vertex_buffers: VulkanVertexStream

    def __init__(self, context: SurfaceContext | None, initial_count: int, attribute_meta: dict[str, Attribute]) -> None:
        ctx = pyglet.graphics.api.core.resolve_context(context)
        self.devices = ctx.devices
        super().__init__(ctx, initial_count, attribute_meta)

    def _has_multi_draw_extension(self, _ctx: SurfaceContext) -> bool:
        return False

    def _create_vertex_class(self) -> type:
        return type(self._vertex_class.__name__, (self._vertex_class,), self.vertex_buffers._property_dict)

    def _create_vao(self) -> VulkanVertexArrayBinding:
        return VulkanVertexArrayBinding(self._context, self._streams)

    def _create_streams(self, size: int) -> list[VertexStream | IndexStream | InstanceStream]:
        self.vertex_buffers = VulkanVertexStream(self._context, self.devices, size, self.per_vertex)
        return [self.vertex_buffers]

    def get_binding_descriptions(self) -> list[VkVertexInputBindingDescription]:
        return self.vertex_buffers.get_binding_descriptions()

    def get_attribute_descriptions(self) -> list[VkVertexInputAttributeDescription]:
        return self.vertex_buffers.get_attribute_descriptions()

    def draw(self, command_buffer) -> None:
        self.vertex_buffers.bind(command_buffer)

        starts, sizes = self.vertex_buffers.allocator.get_allocated_regions()
        for start, size in zip(starts, sizes):
            DeviceFunc.vkCmdDraw(command_buffer, size, 1, start, 0)

    def draw_subset(self, mode: GeometryMode, vertex_list: VertexList) -> None:  # noqa: ARG002
        # Vulkan draws through recorded batch command buffers. For debug-style direct
        # `vertex_list.draw()` calls, fall back to drawing the owning batch.
        for batch in tuple(vertex_list.group._assigned_batches):  # noqa: SLF001
            batch.draw()
            return

        msg = "Vulkan subset drawing requires a batch assignment."
        raise NotImplementedError(msg)

    def delete(self) -> None:
        self.vertex_buffers.delete()


class IndexedVertexDomain(BaseIndexedVertexDomain):
    _vertex_class = VulkanIndexedVertexList
    vertex_buffers: VulkanVertexStream
    index_stream: VulkanIndexStream

    def __init__(
        self,
        context: SurfaceContext | None,
        initial_count: int,
        attribute_meta: dict[str, Attribute],
        index_type: DataTypes = "H",
    ) -> None:
        self.index_type = index_type
        self._supports_base_vertex = False

        ctx = pyglet.graphics.api.core.resolve_context(context)
        self.devices = ctx.devices
        BaseVertexDomain.__init__(self, ctx, initial_count, attribute_meta)

    def _has_multi_draw_extension(self, _ctx: SurfaceContext) -> bool:
        return False

    def _create_vertex_class(self) -> type:
        return type(
            self._vertex_class.__name__,
            (_RunningIndexSupport, self._vertex_class),
            self.vertex_buffers._property_dict,
        )

    def _create_vao(self) -> VulkanVertexArrayBinding:
        return VulkanVertexArrayBinding(self._context, self._streams)

    def _create_streams(self, size: int) -> list[VertexStream | IndexStream | InstanceStream]:
        self.vertex_buffers = VulkanVertexStream(self._context, self.devices, size, self.per_vertex)
        self.index_stream = VulkanIndexStream(self._context, self.devices, self.index_type, size)
        return [self.vertex_buffers, self.index_stream]

    def get_binding_descriptions(self) -> list[VkVertexInputBindingDescription]:
        return self.vertex_buffers.get_binding_descriptions()

    def get_attribute_descriptions(self) -> list[VkVertexInputAttributeDescription]:
        return self.vertex_buffers.get_attribute_descriptions()

    def draw(self, command_buffer) -> None:
        self.vertex_buffers.bind(command_buffer)
        self.index_stream.bind(command_buffer)

        starts, sizes = self.index_stream.allocator.get_allocated_regions()
        for start, size in zip(starts, sizes):
            DeviceFunc.vkCmdDrawIndexed(command_buffer, size, 1, start, 0, 0)

    def draw_subset(self, mode: GeometryMode, vertex_list: IndexedVertexList) -> None:  # noqa: ARG002
        # Vulkan draws through recorded batch command buffers. For debug-style direct
        # `vertex_list.draw()` calls, fall back to drawing the owning batch.
        for batch in tuple(vertex_list.group._assigned_batches):  # noqa: SLF001
            batch.draw()
            return

        msg = "Vulkan indexed subset drawing requires a batch assignment."
        raise NotImplementedError(msg)

    def delete(self) -> None:
        self.index_stream.delete()
        self.vertex_buffers.delete()


class InstancedVertexDomain:
    def __init__(self, *args, **kwargs):  # noqa: ANN002, D107
        raise NotImplementedError("Instanced Vulkan vertex domains are not implemented.")


class InstancedIndexedVertexDomain:
    def __init__(self, *args, **kwargs):  # noqa: ANN002, D107
        raise NotImplementedError("Instanced indexed Vulkan vertex domains are not implemented.")
