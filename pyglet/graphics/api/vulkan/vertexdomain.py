from __future__ import annotations

import ctypes
from typing import TYPE_CHECKING, Any, Sequence

import pyglet
from pyglet.graphics.api.vulkan import DeviceFunc
from pyglet.graphics.api.vulkan.buffer import AttributeBufferObject, IndexBufferObject
from pyglet.graphics.api.vulkan.shader import get_vulkan_format
from pyglet.graphics.instance import InstanceBucket, InstanceDomain
from pyglet.graphics.vertexdomain import (
    InstanceIndexedVertexList as BaseInstanceIndexedVertexList,
    InstanceStream,
    InstanceVertexList as BaseInstanceVertexList,
    InstancedIndexedVertexDomain as BaseInstancedIndexedVertexDomain,
    InstancedVertexDomain as BaseInstancedVertexDomain,
    IndexStream,
    IndexedVertexDomain as BaseIndexedVertexDomain,
    IndexedVertexList as BaseIndexedVertexList,
    VertexArrayBinding,
    VertexArrayProtocol,
    VertexDomain as BaseVertexDomain,
    VertexGroupBucket,
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
    from pyglet.graphics.api.base import SurfaceContext
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

    def set_region(self, start: int, count: int, data: Sequence[float | int]) -> None:
        expected = count * self.attribute.count
        if len(data) != expected:
            msg = f"Invalid data size. Expected {expected}, got {len(data)}."
            raise ValueError(msg)
        self.set_data_region(start, data)

    def invalidate_region(self, start: int, count: int) -> None:  # noqa: ARG002
        pass

    def commit(self) -> None:
        pass


class VulkanIndexBufferObject(IndexBufferObject):
    stride = ctypes.sizeof(ctypes.c_uint16)

    def get_region(self, start: int, count: int):
        return self.get_data_region(start, count)

    def set_region(self, start: int, count: int, data: Sequence[int | float]) -> None:
        if len(data) != count:
            msg = f"Invalid index data size. Expected {count}, got {len(data)}."
            raise ValueError(msg)
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
        binding_offset: int = 0,
    ) -> None:
        self._devices = devices
        self._binding_offset = binding_offset
        self._binding_desc: list[VkVertexInputBindingDescription] = []
        self._attrib_desc: list[VkVertexInputAttributeDescription] = []
        super().__init__(ctx, initial_size, attrs, divisor=divisor)
        self._refresh_descriptors()

    def get_graphics_attribute(self, attribute: Attribute, _view):
        return _VulkanAttribute(attribute)

    def get_buffer(self, _size: int, attribute: _VulkanAttribute) -> VulkanAttributeBufferObject:
        index = self._binding_offset + len(self.buffers)
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

    def bind(self, command_buffer, binding_start: int | None = None) -> None:
        if not self.buffers:
            return

        if binding_start is None:
            binding_start = self.buffers[0].index

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


class VulkanInstanceStream(VulkanVertexStream, InstanceStream):
    pass


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
    domain: VulkanVertexDomain

    def __init__(self, domain: VulkanVertexDomain, group: Group, start: int, count: int) -> None:  # noqa: D107
        super().__init__(domain, group, start, count)


class VulkanInstanceVertexList(BaseInstanceVertexList):
    domain: VulkanInstancedVertexDomain


class VulkanIndexedVertexList(BaseIndexedVertexList):
    domain: VulkanIndexedVertexDomain

    def __init__(
        self,
        domain: VulkanIndexedVertexDomain,
        group: Group,
        start: int,
        count: int,
        index_start: int,
        index_count: int,
    ) -> None:  # noqa: D107
        super().__init__(domain, group, start, count, index_start, index_count)


class VulkanInstanceIndexedVertexList(BaseInstanceIndexedVertexList):
    domain: VulkanInstancedIndexedVertexDomain

    def delete(self) -> None:
        BaseVertexList.delete(self)
        self.domain.index_stream.dealloc(self.index_start, self.index_count)


class VulkanVertexDomain(BaseVertexDomain):
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

    def _draw_subset_command(self, command_buffer, _mode: GeometryMode, vertex_list: VulkanVertexList) -> None:
        self.vertex_buffers.bind(command_buffer)
        DeviceFunc.vkCmdDraw(command_buffer, vertex_list.count, 1, vertex_list.start, 0)

    def draw(self, command_buffer) -> None:
        self.vertex_buffers.bind(command_buffer)

        starts, sizes = self.vertex_buffers.allocator.get_allocated_regions()
        for start, size in zip(starts, sizes):
            DeviceFunc.vkCmdDraw(command_buffer, size, 1, start, 0)

    def draw_buckets(self, command_buffer, buckets: list[VertexGroupBucket]) -> None:
        self.vertex_buffers.bind(command_buffer)

        for bucket in buckets:
            for start, size in bucket.merged_ranges:
                DeviceFunc.vkCmdDraw(command_buffer, size, 1, start, 0)

    def draw_subset(self, mode: GeometryMode, vertex_list: VulkanVertexList) -> None:
        for batch in tuple(vertex_list.group._assigned_batches):  # noqa: SLF001
            batch.draw_subset([vertex_list])
            return

        msg = "Vulkan subset drawing requires a batch assignment."
        raise NotImplementedError(msg)

    def delete(self) -> None:
        self.vertex_buffers.delete()


class VulkanIndexedVertexDomain(BaseIndexedVertexDomain):
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

    def _draw_subset_command(self, command_buffer, _mode: GeometryMode, vertex_list: VulkanIndexedVertexList) -> None:
        self.vertex_buffers.bind(command_buffer)
        self.index_stream.bind(command_buffer)
        DeviceFunc.vkCmdDrawIndexed(command_buffer, vertex_list.index_count, 1, vertex_list.index_start, 0, 0)

    def draw(self, command_buffer) -> None:
        self.vertex_buffers.bind(command_buffer)
        self.index_stream.bind(command_buffer)

        starts, sizes = self.index_stream.allocator.get_allocated_regions()
        for start, size in zip(starts, sizes):
            DeviceFunc.vkCmdDrawIndexed(command_buffer, size, 1, start, 0, 0)

    def draw_buckets(self, command_buffer, buckets: list[VertexGroupBucket]) -> None:
        self.vertex_buffers.bind(command_buffer)
        self.index_stream.bind(command_buffer)

        for bucket in buckets:
            for start, size in bucket.merged_ranges:
                DeviceFunc.vkCmdDrawIndexed(command_buffer, size, 1, start, 0, 0)

    def draw_subset(self, mode: GeometryMode, vertex_list: VulkanIndexedVertexList) -> None:
        for batch in tuple(vertex_list.group._assigned_batches):  # noqa: SLF001
            batch.draw_subset([vertex_list])
            return

        msg = "Vulkan indexed subset drawing requires a batch assignment."
        raise NotImplementedError(msg)

    def delete(self) -> None:
        self.index_stream.delete()
        self.vertex_buffers.delete()


class VulkanInstanceDomainArrays(InstanceDomain):
    def __init__(self, domain: Any, initial_instances: int) -> None:
        super().__init__(domain, initial_instances)
        self._ctx = domain._context
        self._devices = domain.devices
        self._instance_binding_start = len(domain.vertex_buffers.buffers)

    def _create_bucket_arrays(self) -> InstanceBucket:
        istream = VulkanInstanceStream(
            self._ctx,
            self._devices,
            self._initial,
            self._domain.per_instance,
            divisor=1,
            binding_offset=self._instance_binding_start,
        )
        vao = VulkanVertexArrayBinding(self._ctx, [self._domain.vertex_buffers, istream])
        return InstanceBucket(istream, vao)

    def _create_bucket_elements(self) -> InstanceBucket:
        raise NotImplementedError("Use VulkanInstanceDomainElements for indexed draws")

    def draw(self, command_buffer) -> None:
        for bucket in self._buckets.values():
            self.draw_bucket(command_buffer, bucket)

    def draw_bucket(self, command_buffer, bucket: InstanceBucket) -> None:
        if bucket.instance_count <= 0:
            return

        first_vertex, vertex_count = self._geom[bucket]
        self._domain.vertex_buffers.bind(command_buffer)
        bucket.stream.bind(command_buffer)
        DeviceFunc.vkCmdDraw(command_buffer, vertex_count, bucket.instance_count, first_vertex, 0)

    def draw_subset(self, command_buffer, vertex_list: VulkanInstanceVertexList) -> None:
        bucket = vertex_list.instance_bucket
        if bucket.instance_count <= 0:
            return

        self._domain.vertex_buffers.bind(command_buffer)
        bucket.stream.bind(command_buffer)
        DeviceFunc.vkCmdDraw(command_buffer, vertex_list.count, bucket.instance_count, vertex_list.start, 0)

    def delete(self) -> None:
        for bucket in self._buckets.values():
            bucket.stream.delete()
        self._buckets.clear()
        self._geom.clear()


class VulkanInstanceDomainElements(InstanceDomain):
    def __init__(self, domain: Any, initial_instances: int, index_stream: VulkanIndexStream) -> None:
        super().__init__(domain, initial_instances)
        self._ctx = domain._context
        self._devices = domain.devices
        self._index_stream = index_stream
        self._instance_binding_start = len(domain.vertex_buffers.buffers)

    def _create_bucket_elements(self) -> InstanceBucket:
        istream = VulkanInstanceStream(
            self._ctx,
            self._devices,
            self._initial,
            self._domain.per_instance,
            divisor=1,
            binding_offset=self._instance_binding_start,
        )
        vao = VulkanVertexArrayBinding(self._ctx, [self._domain.vertex_buffers, istream, self._index_stream])
        return InstanceBucket(istream, vao)

    def _create_bucket_arrays(self) -> InstanceBucket:
        raise NotImplementedError("Use VulkanInstanceDomainArrays for non-indexed draws")

    def draw_bucket(self, command_buffer, bucket: InstanceBucket) -> None:
        if bucket.instance_count <= 0:
            return

        first_index, index_count, _, base_vertex = self._geom[bucket]
        self._domain.vertex_buffers.bind(command_buffer)
        bucket.stream.bind(command_buffer)
        self._index_stream.bind(command_buffer)
        DeviceFunc.vkCmdDrawIndexed(command_buffer, index_count, bucket.instance_count, first_index, base_vertex, 0)

    def draw(self, command_buffer) -> None:
        for bucket in self._buckets.values():
            self.draw_bucket(command_buffer, bucket)

    def draw_subset(self, command_buffer, vertex_list: VulkanInstanceIndexedVertexList) -> None:
        bucket = vertex_list.instance_bucket
        if bucket.instance_count <= 0:
            return

        self._domain.vertex_buffers.bind(command_buffer)
        bucket.stream.bind(command_buffer)
        self._index_stream.bind(command_buffer)
        DeviceFunc.vkCmdDrawIndexed(
            command_buffer,
            vertex_list.index_count,
            bucket.instance_count,
            vertex_list.index_start,
            vertex_list.base_vertex,
            0,
        )

    def delete(self) -> None:
        for bucket in self._buckets.values():
            bucket.stream.delete()
        self._buckets.clear()
        self._geom.clear()


class VulkanInstancedVertexDomain(BaseInstancedVertexDomain, VulkanVertexDomain):
    _vertex_class = VulkanInstanceVertexList

    def create_instance_domain(self, size: int) -> VulkanInstanceDomainArrays:
        return VulkanInstanceDomainArrays(self, size)

    def _get_instance_stream(self) -> VulkanInstanceStream | None:
        for bucket in self.instance_domain._buckets.values():  # noqa: SLF001
            return bucket.stream
        return None

    def get_binding_descriptions(self) -> list[VkVertexInputBindingDescription]:
        bindings = list(self.vertex_buffers.get_binding_descriptions())
        inst_stream = self._get_instance_stream()
        if inst_stream is not None:
            bindings.extend(inst_stream.get_binding_descriptions())
        return bindings

    def get_attribute_descriptions(self) -> list[VkVertexInputAttributeDescription]:
        attributes = list(self.vertex_buffers.get_attribute_descriptions())
        inst_stream = self._get_instance_stream()
        if inst_stream is not None:
            attributes.extend(inst_stream.get_attribute_descriptions())
        return attributes

    def _draw_subset_command(self, command_buffer, _mode: GeometryMode, vertex_list: VulkanInstanceVertexList) -> None:
        self.instance_domain.draw_subset(command_buffer, vertex_list)

    def draw_buckets(self, command_buffer, buckets: list[VertexGroupBucket]) -> None:
        for bucket in buckets:
            for vl_range in bucket.ranges:
                self.instance_domain.draw_bucket(command_buffer, self._instance_map[vl_range])

    def draw(self, command_buffer) -> None:
        self.instance_domain.draw(command_buffer)

    def draw_subset(self, mode: GeometryMode, vertex_list: VulkanInstanceVertexList) -> None:
        for batch in tuple(vertex_list.group._assigned_batches):  # noqa: SLF001
            batch.draw_subset([vertex_list])
            return

        msg = "Vulkan instanced subset drawing requires a batch assignment."
        raise NotImplementedError(msg)

    def delete(self) -> None:
        self.instance_domain.delete()
        self.vertex_buffers.delete()


class VulkanInstancedIndexedVertexDomain(BaseInstancedIndexedVertexDomain, VulkanIndexedVertexDomain):
    _initial_index_count = 16
    _vertex_class = VulkanInstanceIndexedVertexList

    def __init__(
        self,
        context: SurfaceContext | None,
        initial_count: int,
        attribute_meta: dict[str, Attribute],
        index_type: DataTypes = "H",
    ) -> None:
        super().__init__(context, initial_count, attribute_meta, index_type=index_type)

    def create_instance_domain(self, size: int) -> VulkanInstanceDomainElements:
        return VulkanInstanceDomainElements(self, size, index_stream=self.index_stream)

    def _get_instance_stream(self) -> VulkanInstanceStream | None:
        for bucket in self.instance_domain._buckets.values():  # noqa: SLF001
            return bucket.stream
        return None

    def get_binding_descriptions(self) -> list[VkVertexInputBindingDescription]:
        bindings = list(self.vertex_buffers.get_binding_descriptions())
        inst_stream = self._get_instance_stream()
        if inst_stream is not None:
            bindings.extend(inst_stream.get_binding_descriptions())
        return bindings

    def get_attribute_descriptions(self) -> list[VkVertexInputAttributeDescription]:
        attributes = list(self.vertex_buffers.get_attribute_descriptions())
        inst_stream = self._get_instance_stream()
        if inst_stream is not None:
            attributes.extend(inst_stream.get_attribute_descriptions())
        return attributes

    def _draw_subset_command(
        self,
        command_buffer,
        _mode: GeometryMode,
        vertex_list: VulkanInstanceIndexedVertexList,
    ) -> None:
        self.instance_domain.draw_subset(command_buffer, vertex_list)

    def draw_buckets(self, command_buffer, buckets: list[VertexGroupBucket]) -> None:
        for bucket in buckets:
            for vl_range in bucket.ranges:
                self.instance_domain.draw_bucket(command_buffer, self._instance_map[vl_range])

    def draw(self, command_buffer) -> None:
        self.instance_domain.draw(command_buffer)

    def draw_subset(self, mode: GeometryMode, vertex_list: VulkanInstanceIndexedVertexList) -> None:
        for batch in tuple(vertex_list.group._assigned_batches):  # noqa: SLF001
            batch.draw_subset([vertex_list])
            return

        msg = "Vulkan instanced indexed subset drawing requires a batch assignment."
        raise NotImplementedError(msg)

    def delete(self) -> None:
        self.instance_domain.delete()
        self.index_stream.delete()
        self.vertex_buffers.delete()
