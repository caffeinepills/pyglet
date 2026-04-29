from __future__ import annotations

import ctypes
from ctypes import byref, c_byte, c_void_p
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Sequence

import pyglet
from pyglet.graphics.api.vulkan import DeviceFunc, c_array_list
from pyglet.graphics.buffer import (
    AbstractBuffer,
    MappedBufferObject as BaseMappedBufferObject,
    UniformBufferObject,
)
from pyglet.libs.shared.vulkan_lib.vulkan_core import (
    VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
    VK_BUFFER_USAGE_TRANSFER_DST_BIT,
    VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
    VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
    VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
    VK_INDEX_TYPE_UINT16,
    VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
    VK_SHARING_MODE_EXCLUSIVE,
    VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
    VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
    VkBuffer,
    VkBufferCopy,
    VkBufferCreateInfo,
    VkBufferUsageFlags,
    VkCommandBuffer,
    VkDescriptorSetLayoutBinding,
    VkDevice,
    VkDeviceMemory,
    VkFence,
    VkMemoryAllocateInfo,
    VkMemoryPropertyFlagBits,
    VkMemoryRequirements,
    VkSharingMode,
    VkVertexInputBindingDescription,
    VK_VERTEX_INPUT_RATE_INSTANCE,
    VK_VERTEX_INPUT_RATE_VERTEX,
)

if TYPE_CHECKING:
    from pyglet.graphics.api.vulkan.instance import VulkanSurfaceContext, VulkanInstance, VulkanGlobal
    from ctypes import Structure
    from pyglet.customtypes import CType, CTypesPointer, DataTypes
    from pyglet.graphics.api.vulkan.devices import VulkanDevices
    from pyglet.graphics.api.vulkan.shader import VulkanAttribute

"""
Memory Property Flags:

- VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT:
  Optimized for GPU access; not directly accessible by the CPU.
  Use for resources mainly used by the GPU, like textures and vertex buffers.

- VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT:
  CPU can read/write directly. Ideal for resources updated by the CPU,
  such as dynamic uniform buffers and staging buffers.

- VK_MEMORY_PROPERTY_HOST_COHERENT_BIT:
  Ensures automatic synchronization between CPU and GPU. Useful for small, frequently
  updated buffers, removing the need for manual cache management.

- VK_MEMORY_PROPERTY_HOST_CACHED_BIT:
  Cached by the CPU for faster read access. Good for data read back to the CPU,
  like screenshots or debugging info.

- VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT:
  Memory allocated on-demand by the GPU. Best for transient resources
  like temporary images or framebuffers used briefly.
"""


_DATA_TYPE_TO_CTYPE: dict[str, type[Any]] = {
    "?": ctypes.c_bool,
    "b": ctypes.c_byte,
    "B": ctypes.c_ubyte,
    "h": ctypes.c_short,
    "H": ctypes.c_ushort,
    "i": ctypes.c_int,
    "I": ctypes.c_uint,
    "q": ctypes.c_longlong,
    "Q": ctypes.c_ulonglong,
    "f": ctypes.c_float,
    "d": ctypes.c_double,
}


def _ctype_from_data_type(data_type: DataTypes) -> type[CType]:
    c_type = _DATA_TYPE_TO_CTYPE.get(data_type)
    if c_type is None:
        msg = f"Unsupported data type '{data_type}'."
        raise ValueError(msg)
    return c_type


def _devices_ready(devices: Any) -> bool:
    logical = getattr(devices, "logical_device", None)
    return bool(
        logical
        and getattr(logical, "vk_device", None) is not None
        and hasattr(logical, "vkCreateBuffer")
    )


class VulkanBufferResource:
    __slots__ = ("vk_buffer", "vk_device", "vk_device_memory")

    def __init__(self, vk_device: VkDevice, vk_buffer: VkBuffer, vk_device_memory: VkDeviceMemory) -> None:
        self.vk_device = vk_device
        self.vk_buffer = vk_buffer
        self.vk_device_memory = vk_device_memory

    def bind(self) -> None:
        DeviceFunc.vkBindBufferMemory(self.vk_device, self.vk_buffer, self.vk_device_memory, 0)

    def delete(self) -> None:
        if self.vk_device and self.vk_buffer:
            DeviceFunc.vkDestroyBuffer(self.vk_device, self.vk_buffer, None)
        if self.vk_device and self.vk_device_memory:
            DeviceFunc.vkFreeMemory(self.vk_device, self.vk_device_memory, None)
        self.vk_buffer = None
        self.vk_device_memory = None
        self.vk_device = None

    def __repr__(self) -> str:
        vk_hex = hex(self.vk_buffer.value) if self.vk_buffer and self.vk_buffer.value else "0x0"
        return f"VulkanBufferResource(vk={vk_hex})"


class VulkanBufferObject(AbstractBuffer):
    devices: VulkanDevices | None
    buffer: VulkanBufferResource | None

    def __init__(
        self,
        size: int,
        usage: VkBufferUsageFlags,
        memory_properties: VkMemoryPropertyFlagBits,
        sharing_mode: VkSharingMode = VK_SHARING_MODE_EXCLUSIVE,
    ) -> None:
        super().__init__(size)
        assert size > 0, "Size cannot be 0."
        self.usage = usage
        self.memory_properties = memory_properties
        self.sharing_mode = sharing_mode
        self.devices = None
        self.buffer = None
        self._persistent_map_ptr: int | None = None

    @property
    def id(self) -> int:
        if self.buffer and self.buffer.vk_buffer and self.buffer.vk_buffer.value:
            return int(self.buffer.vk_buffer.value)
        return 0

    def create(self, devices: VulkanDevices) -> None:
        if self.buffer is not None:
            return
        self.devices = devices
        info = self.get_info(self.size, self.usage, self.sharing_mode)
        vk_buffer, vk_device_memory = self.create_buffer(devices, info, self.memory_properties)
        self.buffer = VulkanBufferResource(devices.logical_device.vk_device, vk_buffer, vk_device_memory)
        self.buffer.bind()

    def _resolve_devices(self) -> VulkanDevices | None:
        if self.devices is not None:
            return self.devices
        return None

    def _ensure_created(self) -> None:
        return

    def bind(self) -> None:
        if self.buffer:
            self.buffer.bind()

    def unbind(self) -> None:
        # Vulkan has no global buffer unbind state.
        return

    @staticmethod
    def get_info(size: int, usage: VkBufferUsageFlags, sharing_mode: VkSharingMode) -> VkBufferCreateInfo:
        return VkBufferCreateInfo(
            sType=VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            size=size,
            usage=usage,
            sharingMode=sharing_mode,
        )

    @staticmethod
    def create_buffer(
        devices: VulkanDevices,
        info: VkBufferCreateInfo,
        memory_properties: VkMemoryPropertyFlagBits,
    ) -> tuple[VkBuffer, VkDeviceMemory]:
        vk_logical = devices.logical_device.vk_device
        vk_buffer = VkBuffer()
        devices.logical_device.vkCreateBuffer(vk_logical, byref(info), None, byref(vk_buffer))

        mem_requirements = VkMemoryRequirements()
        devices.logical_device.vkGetBufferMemoryRequirements(vk_logical, vk_buffer, byref(mem_requirements))

        memory_info = VkMemoryAllocateInfo(
            sType=VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            allocationSize=mem_requirements.size,
            memoryTypeIndex=devices.find_memory_type(mem_requirements.memoryTypeBits, memory_properties),
        )

        memory = VkDeviceMemory()
        devices.logical_device.vkAllocateMemory(vk_logical, byref(memory_info), None, byref(memory))
        return vk_buffer, memory

    def _require_allocation(self) -> tuple[VulkanDevices, VulkanBufferResource]:
        self._ensure_created()
        assert self.devices is not None, "Device not set. Call `create` first."
        assert self.buffer is not None, "Buffer is not created. Call `create` first."
        return self.devices, self.buffer

    def _map_raw(self, offset: int, length: int) -> c_void_p:
        devices, buffer = self._require_allocation()
        if self.memory_properties & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT:
            if self._persistent_map_ptr is None:
                mapped_ptr = c_void_p()
                devices.logical_device.vkMapMemory(
                    devices.logical_device.vk_device,
                    buffer.vk_device_memory,
                    0,
                    self.size,
                    0,
                    byref(mapped_ptr),
                )
                self._persistent_map_ptr = int(mapped_ptr.value or 0)
            assert self._persistent_map_ptr is not None and self._persistent_map_ptr != 0
            return c_void_p(self._persistent_map_ptr + offset)

        mapped_ptr = c_void_p()
        devices.logical_device.vkMapMemory(
            devices.logical_device.vk_device,
            buffer.vk_device_memory,
            offset,
            length,
            0,
            byref(mapped_ptr),
        )
        return mapped_ptr

    def _unmap_raw(self) -> None:
        if self.memory_properties & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT:
            return
        devices, buffer = self._require_allocation()
        devices.logical_device.vkUnmapMemory(devices.logical_device.vk_device, buffer.vk_device_memory)

    def get_bytes(self) -> bytes:
        return self.get_bytes_region(0, self.size)

    def get_bytes_region(self, offset: int, length: int) -> bytes:
        assert offset >= 0 and length >= 0, "Offset and length must be non-negative."
        assert offset + length <= self.size, (
            f"Byte range [{offset}, {offset + length}) exceeds buffer size {self.size}."
        )
        ptr = self._map_raw(offset, length)
        try:
            return ctypes.string_at(ptr, length)
        finally:
            self._unmap_raw()

    def set_bytes(self, data: bytes | bytearray | memoryview) -> None:
        raw = bytes(data)
        assert len(raw) == self.size, f"Expected {self.size} bytes for full upload, got {len(raw)}."
        self.set_bytes_region(0, raw)

    def set_bytes_region(self, offset: int, data: bytes | bytearray | memoryview) -> None:
        raw = bytes(data)
        assert offset >= 0 and offset + len(raw) <= self.size, (
            f"Byte range [{offset}, {offset + len(raw)}) exceeds buffer size {self.size}."
        )
        ptr = self._map_raw(offset, len(raw))
        try:
            ctypes.memmove(ptr, raw, len(raw))
        finally:
            self._unmap_raw()

    def copy_buffer(self, dst_buffer: VkBuffer, size: int) -> VkFence:
        devices, buffer = self._require_allocation()
        pool = pyglet.graphics.api.core.command_pool
        command_buffer, fence = pool.get_single_use_fence()
        with command_buffer as vk_cmd_buffer:
            copy_region = VkBufferCopy(srcOffset=0, dstOffset=0, size=size)
            region_array = c_array_list([copy_region], VkBufferCopy)
            devices.logical_device.vkCmdCopyBuffer(vk_cmd_buffer, buffer.vk_buffer, dst_buffer, 1, region_array)
        pool.free([command_buffer])
        return fence

    def _retire_old_resource(self, old_resource: VulkanBufferResource) -> None:
        core = getattr(pyglet.graphics.api, "core", None)
        if core is None:
            old_resource.delete()
            return

        removal = getattr(core, "resource_removal", None)
        try:
            current_window = core.current_window
        except Exception:
            current_window = None
        frame_sync = getattr(current_window, "frame_sync", None) if current_window else None

        if removal and frame_sync and frame_sync.fences:
            fence = frame_sync.fences[frame_sync.current_frame]
            removal.queue_fence(old_resource, fence, False)
            return

        old_resource.delete()

    def resize(self, size: int) -> None:
        assert size > 0, "Size must be greater than 0."
        if size == self.size:
            return

        if self.buffer is None or self.devices is None:
            self.size = size
            return

        copy_size = min(size, self.size)
        old_resource = self.buffer
        old_data = self.get_bytes_region(0, copy_size) if copy_size > 0 else b""

        if self._persistent_map_ptr:
            self.devices.logical_device.vkUnmapMemory(
                self.devices.logical_device.vk_device,
                old_resource.vk_device_memory,
            )
            self._persistent_map_ptr = None

        self.size = size
        self.buffer = None
        self.create(self.devices)

        if old_data:
            self.set_bytes_region(0, old_data)

        self._retire_old_resource(old_resource)

    def delete(self) -> None:
        if self.buffer:
            if self._persistent_map_ptr and self.devices is not None:
                self.devices.logical_device.vkUnmapMemory(
                    self.devices.logical_device.vk_device,
                    self.buffer.vk_device_memory,
                )
            self._persistent_map_ptr = None
            self.buffer.delete()
            self.buffer = None

    def __del__(self) -> None:
        try:
            self.delete()
        except Exception:
            # Interpreter shutdown / partially-torn runtime.
            pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self.id}, size={self.size})"


class MappedBufferObject(VulkanBufferObject, BaseMappedBufferObject):
    data_type: DataTypes
    c_type: type[CType]
    c_type_size: int

    def __init__(
        self,
        data_type: DataTypes,
        size: int,
        usage: VkBufferUsageFlags,
        memory_properties: VkMemoryPropertyFlagBits,
        sharing_mode: VkSharingMode = VK_SHARING_MODE_EXCLUSIVE,
    ) -> None:
        self.data_type = data_type
        self.c_type = _ctype_from_data_type(data_type)
        self.c_type_size = ctypes.sizeof(self.c_type)
        super().__init__(size, usage, memory_properties, sharing_mode)

    def map(self, bits: int = 0) -> CTypesPointer:  # noqa: ARG002
        return self._map_raw(0, self.size)

    def map_range(
        self,
        start: int,
        size: int,
        ptr_type: type[CTypesPointer] | None = None,
        bits: int = 0,  # noqa: ARG002
    ) -> CTypesPointer:
        ptr = self._map_raw(start, size)
        if ptr_type is None:
            return ptr
        return ctypes.cast(ptr, ptr_type).contents

    def unmap(self) -> None:
        self._unmap_raw()

    @lru_cache(maxsize=None)
    def get_region(self, start: int, count: int):
        return self.get_data_region(start, count)

    def get_data(self) -> ctypes.Array[CType]:
        element_count = self.size // self.c_type_size
        return self.get_data_region(0, element_count)

    def get_data_region(self, start: int, length: int) -> ctypes.Array[CType]:
        byte_offset = start * self.c_type_size
        byte_size = length * self.c_type_size
        raw = self.get_bytes_region(byte_offset, byte_size)
        data_type = self.c_type * length
        return data_type.from_buffer_copy(raw)

    def set_data(self, data: Sequence[int | float] | ctypes.Array[Any], start: int = 0) -> None:
        self.set_data_region(start, data)

    def set_data_region(self, start: int, data: Sequence[int | float] | ctypes.Array[Any]) -> None:
        elements = len(data)
        if elements == 0:
            return
        byte_offset = start * self.c_type_size
        data_type = self.c_type * elements
        c_array = data_type(*data)
        byte_size = ctypes.sizeof(c_array)
        self.set_bytes_region(byte_offset, ctypes.string_at(ctypes.addressof(c_array), byte_size))
        cache_clear = getattr(self.get_region, "cache_clear", None)
        if cache_clear:
            cache_clear()

    def clear(self) -> None:
        self.set_bytes(b"\x00" * self.size)

    def commit(self) -> None:
        # Host visible/coherent buffers are written immediately.
        return

    def invalidate_region(self, start: int, count: int) -> None:  # noqa: ARG002
        return


class UniformBuffer(MappedBufferObject):
    def __init__(
        self,
        data_type: DataTypes,
        size: int,
        memory_properties: VkMemoryPropertyFlagBits = (
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
        ),
        sharing_mode: VkSharingMode = VK_SHARING_MODE_EXCLUSIVE,
    ) -> None:
        super().__init__(data_type, size, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, memory_properties, sharing_mode)


class BackedBufferObject(MappedBufferObject):
    """Compatibility shim for older Vulkan buffer code paths."""


class PersistentBufferObject(MappedBufferObject):
    """Compatibility shim for a persistently mapped Vulkan buffer."""

    def __init__(self, size: int, usage: int) -> None:
        super().__init__(
            "b",
            size,
            usage,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        )


class AttributeBufferObject(MappedBufferObject):
    def __init__(
        self,
        data_type: DataTypes,
        devices: VulkanDevices,
        index: int,
        attribute: VulkanAttribute,
        buffer_size: int,
    ) -> None:
        self.devices = devices
        self.attribute = attribute
        self.index = index
        self.input_rate = VK_VERTEX_INPUT_RATE_INSTANCE if attribute.instance else VK_VERTEX_INPUT_RATE_VERTEX

        byte_size = buffer_size * self.attribute.stride
        super().__init__(
            data_type,
            byte_size,
            usage=VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
            memory_properties=VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        )

        self.bind_descriptor = VkVertexInputBindingDescription(
            binding=self.index,
            stride=self.attribute.stride,
            inputRate=self.input_rate,
        )

    def bind(self, command_buffer: VkCommandBuffer | None = None, offset: int = 0) -> None:
        if command_buffer is None:
            super().bind()
            return
        assert self.buffer is not None, "Buffer is not created. Call `create` first."
        buffer_array = c_array_list([self.buffer.vk_buffer], VkBuffer)
        offsets = c_array_list([offset], ctypes.c_uint64)
        DeviceFunc.vkCmdBindVertexBuffers(command_buffer, self.index, 1, buffer_array, offsets)

    def get_data_region(self, start: int, length: int) -> ctypes.Array[CType]:
        byte_offset = start * self.attribute.stride
        byte_size = length * self.attribute.stride
        raw = self.get_bytes_region(byte_offset, byte_size)
        element_ct = self.attribute.count * length
        data_type = self.attribute.c_type * element_ct
        return data_type.from_buffer_copy(raw)

    def get_data_region_no_copy(self, start: int, length: int) -> ctypes.Array[CType]:
        byte_offset = start * self.attribute.stride
        byte_size = length * self.attribute.stride
        assert byte_offset + byte_size <= self.size, (
            f"Mapping range [{byte_offset}, {byte_offset + byte_size}) exceeds buffer size {self.size}."
        )

        if self._persistent_map_ptr is None:
            self._map_raw(0, self.size)
        if self._persistent_map_ptr:
            element_ct = self.attribute.count * length
            data_type = self.attribute.c_type * element_ct
            return data_type.from_address(self._persistent_map_ptr + byte_offset)

        # Fallback for non host-visible paths.
        return self.get_data_region(start, length)

    def set_data_region(self, start: int, data: Sequence[int | float] | ctypes.Array[Any]) -> None:
        elements = len(data)
        if elements == 0:
            return
        byte_offset = start * self.attribute.stride
        data_type = self.attribute.c_type * elements
        c_array = data_type(*data)
        byte_size = ctypes.sizeof(c_array)
        assert byte_offset + byte_size <= self.size, (
            f"Mapping range [{byte_offset}, {byte_offset + byte_size}) exceeds buffer size {self.size}."
        )
        self.set_bytes_region(byte_offset, ctypes.string_at(ctypes.addressof(c_array), byte_size))

    def set_region(self, start: int, count: int, data: Sequence[int | float]) -> None:
        expected = count * self.attribute.count
        if len(data) != expected:
            msg = f"Invalid data size. Expected {expected}, got {len(data)}."
            raise ValueError(msg)
        self.set_data_region(start, data)

    def get_region(self, start: int, count: int):
        return self.get_data_region_no_copy(start, count)

    def invalidate_region(self, start: int, count: int) -> None:  # noqa: ARG002
        return

    def commit(self) -> None:
        return


class IndexBufferObject(MappedBufferObject):
    def __init__(self, size: int):
        super().__init__(
            "H",
            size,
            usage=VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
            memory_properties=VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        )

    def bind(self, command_buffer: VkCommandBuffer | None = None, offset: int = 0) -> None:
        if command_buffer is None:
            super().bind()
            return
        assert self.buffer is not None, "Buffer is not created. Call `create` first."
        DeviceFunc.vkCmdBindIndexBuffer(command_buffer, self.buffer.vk_buffer, offset, VK_INDEX_TYPE_UINT16)

    def get_region(self, start: int, count: int) -> list[int]:
        data = self.get_data_region(start, count)
        return list(data)

    def set_region(self, start: int, count: int, data: Sequence[int | float]) -> None:
        if len(data) != count:
            msg = f"Invalid index data size. Expected {count}, got {len(data)}."
            raise ValueError(msg)
        self.set_data_region(start, data)

    def copy_region(self, dst: int, src: int, count: int) -> None:
        self.set_region(dst, count, self.get_region(src, count))

    def bind_to_index_buffer(self) -> None:
        # Vulkan index binding is recorded per-command-buffer.
        return

    def commit(self) -> None:
        return


class StagingBufferObject(MappedBufferObject):
    """A host-visible transfer source buffer."""

    def __init__(self, size: int):
        super().__init__(
            "b",
            size,
            usage=VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            memory_properties=VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        )

    def set_data_as_type(self, data: bytes, _ctype_type) -> None:
        raw = bytes(data)
        assert len(raw) == self.size, f"Data size {len(raw)} exceeds buffer size {self.size}."
        self.set_bytes(raw)


class StagingBufferObject2(MappedBufferObject):
    """A host-visible transfer destination buffer."""

    def __init__(self, size: int):
        super().__init__(
            "b",
            size,
            usage=VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            memory_properties=VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        )


class VulkanUniformBufferObject(UniformBufferObject):
    buffer: UniformBuffer
    view: Structure
    _view_ptr: CTypesPointer[Structure]
    binding: int
    layout_binding: VkDescriptorSetLayoutBinding | None
    __slots__ = ("_context", "_pending_upload", "layout_binding")

    def __init__(
        self,
        context: Any,
        view_class: type[Structure],
        buffer_size: int,
        binding: int,
        layout_binding: VkDescriptorSetLayoutBinding | None = None,
    ) -> None:
        self._context = context
        self._pending_upload = False
        super().__init__(context, view_class, buffer_size, binding)
        self.layout_binding = layout_binding

    def _resolve_devices(self):
        for candidate in (
            getattr(self._context, "devices", None),
            getattr(getattr(pyglet.graphics.api, "core", None), "devices", None),
        ):
            if candidate is not None and _devices_ready(candidate):
                return candidate

        core = getattr(pyglet.graphics.api, "core", None)
        if core is not None:
            try:
                current_ctx = core.current_context
            except Exception:
                current_ctx = None
            devices = getattr(current_ctx, "devices", None)
            if _devices_ready(devices):
                return devices
        return None

    def _ensure_buffer_created(self) -> None:
        if self.buffer is None:
            print("No Bfer")
            return
        if self.buffer.buffer is not None and self.buffer.devices is not None:
            self._flush_pending_upload()
            return
        devices = self._resolve_devices()
        if devices is not None and self.buffer.buffer is None:
            self.buffer.create(devices)
            self._flush_pending_upload()

    def _flush_pending_upload(self) -> None:
        if not self._pending_upload:
            return
        if self.buffer is None or self.buffer.buffer is None:
            return
        self.buffer.set_data_ptr(0, ctypes.sizeof(self.view), self._view_ptr)
        self._pending_upload = False

    def _create_buffer(self, context: VulkanGlobal, buffer_size: int) -> UniformBuffer:
        buffer = UniformBuffer("b", buffer_size)
        buffer.create(context.devices)
        return buffer

    def delete(self) -> None:
        if self.buffer:
            self.buffer.delete()
        self.buffer = None
        self.view = None

    @property
    def id(self) -> int:
        self._ensure_buffer_created()
        if self.buffer and self.buffer.buffer and self.buffer.buffer.vk_buffer:
            return int(self.buffer.buffer.vk_buffer.value)
        return 0

    def bind(self) -> None:
        # Vulkan UBO usage is controlled through descriptor set writes/binds.
        return

    def unbind(self) -> None:
        return

    def read(self) -> bytes:
        self._ensure_buffer_created()
        return self.buffer.get_bytes()

    def __exit__(self, _exc_type, _exc_val, _exc_tb) -> None:  # noqa: ANN001
        self._pending_upload = True
        self._ensure_buffer_created()
        self._flush_pending_upload()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self.id}, binding={self.binding})"
