from __future__ import annotations

import ctypes
import sys
from ctypes import byref, sizeof, memmove, c_byte, c_void_p, cast, POINTER, Array, addressof
from functools import lru_cache
from typing import Sequence, TYPE_CHECKING

import pyglet
from pyglet.graphics.api.vulkan import DeviceFunc, c_array_list
from pyglet.graphics.buffer import AbstractBuffer, MappedBufferObject
from pyglet.libs.shared.vulkan_lib.vulkan_core import VkBufferCreateInfo, VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, \
    VK_SHARING_MODE_EXCLUSIVE, \
    VkMemoryAllocateInfo, VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, \
    VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, \
    VkVertexInputBindingDescription, VK_VERTEX_INPUT_RATE_INSTANCE, VK_VERTEX_INPUT_RATE_VERTEX, VkBuffer, \
    VkMemoryRequirements, VkDeviceMemory, VkMemoryPropertyFlagBits, VkSharingMode, VkBufferUsageFlags, \
    VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_BUFFER_USAGE_INDEX_BUFFER_BIT, VK_INDEX_TYPE_UINT16, \
    VK_BUFFER_USAGE_TRANSFER_DST_BIT, VkBufferCopy, VkDevice, VkCommandBuffer, VkFence

if TYPE_CHECKING:
    from pyglet.customtypes import DataTypes, CType, CTypesPointer
    from pyglet.graphics.api.vulkan.shader import VulkanAttribute
    from ctypes import Array
    from pyglet.graphics.api.vulkan.devices import VulkanDevices

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

class VulkanBufferResource:
    __slots__ = ("vk_buffer", "vk_device", "vk_device_memory")

    def __init__(self, vk_device: VkDevice, vk_buffer: VkBuffer, vk_device_memory: VkDeviceMemory) -> None:
        self.vk_device = vk_device
        self.vk_buffer = vk_buffer
        self.vk_device_memory = vk_device_memory

    def bind(self) -> None:
        DeviceFunc.vkBindBufferMemory(self.vk_device, self.vk_buffer, self.vk_device_memory, 0)

    def delete(self) -> None:
        """Free the buffer and associated memory."""
        if self.vk_device:
            DeviceFunc.vkDestroyBuffer(self.vk_device, self.vk_buffer, None)
            DeviceFunc.vkFreeMemory(self.vk_device, self.vk_device_memory, None)

        self.vk_buffer = None
        self.vk_device_memory = None
        self.vk_device = None

    def __repr__(self) -> str:
        return f"Buffer(py={hex(id(self.vk_buffer))}, vk={hex(self.vk_buffer.value)})"


class VulkanBufferObject(AbstractBuffer):
    devices: VulkanDevices | None

    size: int
    usage: int
    memory_properties: int

    def __init__(self, data_type: DataTypes, size: int,
                 usage: VkBufferUsageFlags,
                 memory_properties: VkMemoryPropertyFlagBits,
                 sharing_mode: VkSharingMode = VK_SHARING_MODE_EXCLUSIVE) -> None:
        """Initialize the BufferObject.

        Args:
            size: The size of the buffer in bytes.
            usage: The VK_BUFFER_USAGE_X bits.
            memory_properties: The VkMemoryType to use for this buffer.
        """
        super().__init__(data_type, size)
        assert size > 0, "Size cannot be 0."
        self.usage = usage
        self.devices = None
        self.memory_properties = memory_properties
        self.sharing_mode = sharing_mode

        # Filled by create later
        self.buffer = None

    def create(self, devices: VulkanDevices) -> None:
        self.devices = devices
        info = self.get_info(self.size, self.usage, self.sharing_mode)

        vk_buffer, vk_device_memory = self.create_buffer(devices, info, self.memory_properties)
        self.buffer = VulkanBufferResource(self.devices.logical_device.vk_device, vk_buffer, vk_device_memory)
        self.buffer.bind()

    @staticmethod
    def get_info(size: int, usage: VkBufferUsageFlags, sharing_mode: VkSharingMode) -> VkBufferCreateInfo:
        return VkBufferCreateInfo(
            sType=VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            size=size,
            usage=usage,
            sharingMode=sharing_mode,
        )

    @classmethod
    def create_buffers(cls, devices: VulkanDevices,
                       info: VkBufferCreateInfo,
                       memory_properties: VkMemoryPropertyFlagBits,
                       ) -> list[VulkanBufferResource]:
        pass

    @staticmethod
    def create_buffer(devices: VulkanDevices, info: VkBufferCreateInfo,
                      memory_properties: VkMemoryPropertyFlagBits) -> tuple[VkBuffer, VkDeviceMemory]:
        """Create a Vulkan buffer and allocate memory for it."""
        vk_logical = devices.logical_device.vk_device

        vk_buffer = VkBuffer()
        devices.logical_device.vkCreateBuffer(vk_logical, byref(info), None, byref(vk_buffer))

        # Get memory requirements of buffer.
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

    def copy_buffer(self, dst_buffer: VkBuffer, size: int) -> VkFence:
        """Copy this buffer into a destination buffer.

        Not as useful because both buffers need to have been made with the right transfer flags for both
        source and destination, which may not always be used. Adding unnecessary flags can reduce performance.

        Furthermore, in those cases you would need a staging buffer to transfer using a command buffer. At that point
        it may just be faster to map the data locally then map it to the new one.
        """
        pool = pyglet.graphics.api.core.command_pool
        command_buffer, fence = pool.get_single_use_fence()
        with command_buffer as vk_cmd_buffer:
            copy_region = VkBufferCopy(srcOffset=0, dstOffset=0, size=size)
            region_array = c_array_list([copy_region], VkBufferCopy)
            self.devices.logical_device.vkCmdCopyBuffer(vk_cmd_buffer, self.buffer.vk_buffer, dst_buffer, 1, region_array)
        pool.free([command_buffer])
        return fence

    def delete(self) -> None:
        """Free the buffer and associated memory."""
        if self.devices and self.buffer and self.devices.logical_device and self.devices.logical_device.vk_device:
            self.buffer.delete()
            self.buffer = None

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(buffer={self.buffer}, size={self.size}, data_type={self.data_type})"

class MappedBufferObject(VulkanBufferObject, MappedBufferObject):
    """Object containing a Vulkan Buffer Object."""



    @lru_cache(maxsize=None)  # noqa: B019
    def get_bytes(self) -> bytes:
        size = self.size * self.c_type_size
        data_type = c_byte * size
        ptr = cast(self.map(), POINTER(data_type))
        new_data = data_type.from_buffer(ptr.contents)
        self.unmap()
        return new_data

    def clear(self):
        zero_data = [0] * (self.size // self.c_type_size)
        self.set_data_region(0, zero_data)

    def get_data(self) -> ctypes.Array[CType]:
        element_count = self.size // self.c_type_size
        return self.get_data_region(0, element_count)

    def get_data_region(self, start: int, length: int) -> ctypes.Array[CType]:
        # Convert elements to bytes.
        byte_size = length * self.c_type_size
        byte_offset = start * self.c_type_size
        data = (self.c_type * length)()

        ptr = self.map_range(byte_offset, byte_size)
        mapped_data = ctypes.cast(ptr, ctypes.POINTER(self.c_type * length)).contents
        data[:] = mapped_data[:]
        self.unmap()
        return data

    def get_bytes_region(self, offset: int, length: int) -> Array:
        ptr = self.map_range(offset, length)
        data = (c_byte * length).from_buffer_copy(ptr)
        self.unmap()
        return data

    def set_bytes_region(self, offset: int, size: int, data: Sequence):
        ptr = self.map_range(offset, size)
        memmove(ptr, data, size)
        self.unmap()

    def set_bytes(self, data: bytes) -> None:
        size = len(data)
        assert size <= self.size, "Data size exceeds buffer size."
        self.set_bytes_region(0, size, data)

    def set_data_ptr(self, ptr: CTypesPointer) -> None:
        """Copy data from ptr into the data_ptr at offset."""
        logical = self.devices.logical_device
        vk_logical = logical.vk_device

        mapped_ptr = c_void_p()
        logical.vkMapMemory(vk_logical, self.buffer.vk_device_memory, 0, self.size, 0, byref(mapped_ptr))
        memmove(mapped_ptr, ptr, self.size)
        logical.vkUnmapMemory(vk_logical, self.buffer.vk_device_memory)

    def set_data(self, data: Array[CType]) -> None:
        """Copy data to the buffer.

        Args:
            data: Data to copy, in bytes.
        """
        byte_size = self.c_type_size * len(data)
        self.set_bytes_region(0, byte_size, data)

    def map_range(self, offset: int, length: int) -> CTypesPointer:
        """Map the buffer memory and return a pointer to it."""
        assert self.devices is not None, "Device not set. Call `create` first."
        mapped_data = c_void_p()
        self.devices.logical_device.vkMapMemory(
            self.devices.logical_device.vk_device, self.buffer.vk_device_memory, offset, length, 0, byref(mapped_data),
        )
        return mapped_data

    def map(self) -> CTypesPointer:
        """Map the buffer memory and return a pointer to it."""
        return self.map_range(0, self.size)

    def unmap(self) -> None:
        """Unmap the buffer memory."""
        self.devices.logical_device.vkUnmapMemory(self.devices.logical_device.vk_device, self.buffer.vk_device_memory)

    def set_data_region(self, start: int, data: Sequence[float | int]) -> None:
        """Copy data to the buffer.

        Args:
            data: Data to copy, in bytes.
        """
        assert self.devices is not None, "Device not set. Call `create` first."
        assert start + len(data) <= self.size, "Data exceeds buffer size."

        size = len(data)
        offset = start * self.c_type_size
        length = size * self.c_type_size

        mapped_data = cast(self.map_range(offset, length), POINTER(self.c_type * size)).contents
        mapped_data[:] = data
        self.unmap()

    def __del__(self) -> None:
        """Clean up the buffer when it goes out of scope."""
        if self.buffer is not None:
            self.delete()

    def resize(self, new_size: int) -> None:
        """Resize the buffer to a new size, reallocating if necessary."""
        if new_size == self.size:
            return

        # Backup old info
        current_size = self.size
        current_buffer = self.buffer

        data = (self.c_type * current_size)()
        ptr = self.map()
        #print(cast(mapped_ptr, POINTER(c_byte * self.size)).contents[:])
        memmove(data, ptr, current_size)
        self.unmap()

        # Update size and create buffer with new size
        self.size = new_size
        self.create(self.devices)

        # If old buffer, copy it into the new one.
        # Restore old data if it was saved
        if data:
            self.set_bytes_region(0, current_size, data)

        # Add the old buffer to be removed at the end of being rendered.
        framesync = pyglet.graphics.api.core.current_window.frame_sync
        fence = framesync.fences[framesync.current_frame]
        pyglet.graphics.api.core.resource_removal.queue_fence(current_buffer, fence, False)

class UniformBuffer(MappedBufferObject):
    def __init__(self, data_type: DataTypes, size: int,
                 memory_properties: VkMemoryPropertyFlagBits = \
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 sharing_mode: VkSharingMode = VK_SHARING_MODE_EXCLUSIVE) -> None:
        super().__init__(data_type, size, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, memory_properties, sharing_mode)

class BackedBufferObject(VulkanBufferObject):
    """A Vulkan buffer backed by system memory, with changes synced on commit."""

    data: Array
    _dirty_min: int
    _dirty_max: int
    _dirty: bool
    stride: int
    ctype: type

    def __init__(self, data_type: CType, size: int, c_type: type, stride: int, usage: int, memory_properties: int) -> None:
        super().__init__(data_type, size, usage, memory_properties)
        self.c_type = c_type
        self._ctypes_size = sizeof(c_type)
        number = size // self._ctypes_size
        self.data = (c_type * number)()
        self.data_ptr = addressof(self.data)

        self._dirty_min = sys.maxsize
        self._dirty_max = 0
        self._dirty = False

        self.stride = stride

    def commit(self) -> None:
        """Commits all saved changes to the underlying buffer before drawing.

        Allows submitting multiple changes at once, rather than having to call glBufferSubData for every change.
        """
        if not self._dirty:
            return

        #glBindBuffer(GL_ARRAY_BUFFER, self.id)
        size = self._dirty_max - self._dirty_min
        if size > 0:
            #if size == self.size:
                #glBufferData(GL_ARRAY_BUFFER, self.size, self.data, self.usage)
            #else:
                #glBufferSubData(GL_ARRAY_BUFFER, self._dirty_min, size, self.data_ptr + self._dirty_min)

            self._dirty_min = sys.maxsize
            self._dirty_max = 0
            self._dirty = False

    def set_data(self, data: Sequence[int]) -> None:
        """Sets data in the local ctypes array."""
        assert len(data) * sizeof(self.ctype) <= self.size, "Data size exceeds buffer size."
        memmove(self.data, (self.ctype * len(data))(*data), len(data) * sizeof(self.ctype))
        self._dirty = True
        self._dirty_min = 0
        self._dirty_max = self.size

    def set_region(self, start: int, count: int, data: Sequence[float]) -> None:
        array_start = self.count * start
        array_end = self.count * count + array_start

        self.data[array_start:array_end] = data

        # replicated from self.invalidate_region
        byte_start = self.stride * start
        byte_end = byte_start + self.stride * count
        # As of Python 3.11, this is faster than min/max:
        if byte_start < self._dirty_min:
            self._dirty_min = byte_start
        if byte_end > self._dirty_max:
            self._dirty_max = byte_end
        self._dirty = True


    # def set_region(self, data: Sequence[int], start: int, length: int) -> None:
    #     """Sets a specific region of the local ctypes array."""
    #     byte_start = start * self.stride
    #     byte_end = byte_start + length * self.stride
    #
    #     for i in range(len(data)):
    #         self.data[byte_start + i] = data[i]
    #
    #     # Mark this region as dirty for synchronization
    #     self._dirty_min = min(self._dirty_min, byte_start)
    #     self._dirty_max = max(self._dirty_max, byte_end)
    #     self._dirty = True

    def commit(self) -> None:
        """Synchronizes local changes to the GPU buffer."""
        if not self._dirty:
            return

        vk_logical = self.devices.logical_device.vk_device

        # Map, copy the data range if necessary, then unmap
        mapped_memory = vkMapMemory(vk_logical, self.vk_device_memory, 0, self.size, 0)
        if self._dirty_max - self._dirty_min == self.size:
            ctypes.memmove(mapped_memory, self.data, self.size)
        else:
            ctypes.memmove(
                ctypes.addressof(mapped_memory) + self._dirty_min,
                ctypes.addressof(self.data) + self._dirty_min,
                self._dirty_max - self._dirty_min,
            )
        vkUnmapMemory(vk_logical, self.vk_device_memory)

        # Reset dirty markers
        self._dirty_min = self.size
        self._dirty_max = 0
        self._dirty = False

    def resize(self, new_size: int) -> None:
        """Resizes the local ctypes array and Vulkan buffer."""
        # Store old data and resize local data
        new_data = (self.ctype * (new_size // ctypes.sizeof(self.ctype)))()
        ctypes.memmove(new_data, self.data, min(self.size, new_size))
        self.data = new_data
        self.data_ptr = ctypes.addressof(self.data)

        # Resize Vulkan buffer
        super().resize(new_size)

        # Mark everything as dirty after resize
        self._dirty_min = 0
        self._dirty_max = new_size
        self._dirty = True

class PersistentBufferObject(VulkanBufferObject):
    """A Vulkan buffer that remains persistently mapped for direct access."""

    mapped_memory: POINTER(c_byte)

    def __init__(self, size: int, usage: int) -> None:
        super().__init__(size, usage, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
        self.mapped_memory = None

    def create(self, devices: VulkanDevices) -> None:
        super().create(devices)

        # Map the memory and keep it mapped persistently
        vk_logical = devices.logical_device.vk_device
        self.mapped_memory = vkMapMemory(vk_logical, self.vk_device_memory, 0, self.size, 0)

    def set_data(self, data: bytes) -> None:
        """Directly writes data into the mapped memory."""
        assert len(data) <= self.size, "Data size exceeds buffer size."
        ctypes.memmove(self.mapped_memory, data, len(data))

    def __del__(self) -> None:
        """Clean up and unmap the memory when done."""
        if self.mapped_memory is not None:
            vkUnmapMemory(self.devices.logical_device.vk_device, self.vk_device_memory)
            self.mapped_memory = None
        super().__del__()



class AttributeBufferObject(MappedBufferObject):
    """Represents an attribute buffer, supporting both per-vertex and per-instance data."""

    def __init__(self, data_type: DataTypes, devices: VulkanDevices, index: int, attribute: VulkanAttribute, buffer_size: int) -> None:
        """Args:
        devices: Vulkan devices.
        index: Attribute index in relation to the list.
        attribute: The Attribute defining this attribute.
        vertex_count: Vertex element_count to use for starting buffer size.
        """
        self.devices = devices
        self.attribute = attribute
        self.index = index
        self.input_rate = VK_VERTEX_INPUT_RATE_INSTANCE if attribute.instance else VK_VERTEX_INPUT_RATE_VERTEX

        # Determine starting buffer size based on whether the attribute is per-vertex or per-instance
        #element_count = 1 if attribute.instance else attribute.element_count
        #buffer_size = attribute.stride
        buffer_size *= self.attribute.stride

        # Initialize the base BufferObject with the calculated size and usage flags
        super().__init__(data_type,
            size=buffer_size,
            usage=VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
            memory_properties=VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        )

        self.bind_descriptor = VkVertexInputBindingDescription(
            binding=self.index, stride=self.attribute.stride, inputRate=self.input_rate,
        )

    def clear(self):
        zero_data = [0] * (self.size // self.attribute.stride)
        self.set_data_region(0, zero_data)  # Fill the entire buffer

    def bind(self, command_buffer: VkCommandBuffer, offset: int = 0) -> None:
        """Bind the buffer to the command buffer.

        It's better to bind multiple at a time, but this works for testing.
        """
        array_buf = c_array_list([self.buffer.vk_buffer], VkBuffer)
        offsets = c_array_list([0], ctypes.c_uint64)
        DeviceFunc.vkCmdBindVertexBuffers(command_buffer, self.index, 1, array_buf, offsets)

    def get_data(self) -> ctypes.Array[CType]:
        element_count = self.size // self.attribute.stride
        return self.get_data_region(0, element_count)

    def get_data_region(self, start: int, length: int) -> ctypes.Array[CType]:
        # Convert elements to bytes.
        byte_size = length * self.attribute.stride
        byte_offset = start * self.attribute.stride
        data_type = (self.attribute.c_type * length)
        mapped_data = cast(self.map_range(byte_offset, byte_size), POINTER(data_type)).contents
        data = data_type.from_buffer_copy(mapped_data)
        self.unmap()
        return data

    def get_data_region_no_copy(self, start: int, length: int) -> ctypes.Array[CType]:
        """Does not copy the buffer, be sure not to write into this buffer using this function."""
        # Convert elements to bytes.
        #print("START", start, length)
        byte_size = length * self.attribute.stride
        byte_offset = start * self.attribute.stride
        element_ct = self.attribute.count * length
        data_type = (self.attribute.c_type * element_ct)
        mapped_data = cast(self.map_range(byte_offset, byte_size), POINTER(data_type)).contents
        data = data_type.from_buffer(mapped_data)
        self.unmap()
        return data

    def set_data_region(self, start: int, data: Sequence) -> None:
        """Copy data to the buffer, considering stride and components."""
        # Calculate the byte offset and size
        elements = len(data)
        byte_offset = start * self.attribute.stride
        byte_size = elements * self.c_type_size

        # Validate the memory range
        assert byte_offset + byte_size <= self.size, (
            f"Mapping range {byte_offset + byte_size} exceeds memory size {self.size=}. {start=}, {data=}, {elements=}, {byte_size=}, {byte_offset=}"
        )

        # Map, cast, copy, and unmap memory
        data_type = (self.attribute.c_type * elements)
        mapped_data = cast(self.map_range(byte_offset, byte_size), POINTER(data_type)).contents
        mapped_data[:] = data
        self.unmap()

    def set_data(self, data: bytes, start: int = 0) -> None:
        """Copy data to the buffer.

        Args:
            data: Data to copy, in bytes.
        """
        size = sizeof(data)
        assert self.devices is not None, "Device not set. Call `create` first."
        assert size <= self.size, "Data size exceeds buffer size."

        logical = self.devices.logical_device
        vk_device = self.devices.logical_device.vk_device

        # fix later. start pos could potentially be different if different sized arrays were done. change start of caller.
        start_pos = start * self.attribute.stride

        mapped_ptr = c_void_p()
        logical.vkMapMemory(vk_device, self.buffer.vk_device_memory, start_pos, size, 0, byref(mapped_ptr))
        memmove(mapped_ptr, data, size)
        logical.vkUnmapMemory(vk_device, self.buffer.vk_device_memory)
    #
    # def bind(self, command_buffer, binding_index):
    #     """Bind this attribute buffer to a specific binding index in the command buffer."""
    #     vkCmdBindVertexBuffers(command_buffer, binding_index, 1, [self.vk_buffer], [0])

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(vk_buffer={self.buffer}, attribute={self.attribute}, size={self.size})"

#
# class InterleavedAttributeBufferObject(BufferObject):
#     """Manages a single interleaved Vulkan buffer for both per-vertex and per-instance attributes."""
#
#     def __init__(self, device, attributes, vertex_count, instance_count):
#         """
#         Args:
#             device: Vulkan logical device.
#             attributes: List of Attribute objects defining each attribute.
#             vertex_count: Number of vertices.
#             instance_count: Number of instances.
#         """
#         self.device = device
#         self.attributes = attributes
#         self.vertex_count = vertex_count
#         self.instance_count = instance_count
#
#         # Separate vertex and instance attributes
#         self.vertex_attributes = [attr for attr in attributes if not attr.instance]
#         self.instance_attributes = [attr for attr in attributes if attr.instance]
#
#         # Calculate strides for vertex and instance data
#         self.vertex_stride = sum(attr.element_size * attr.element_count for attr in self.vertex_attributes)
#         self.instance_stride = sum(attr.element_size * attr.element_count for attr in self.instance_attributes)
#
#         # Calculate total buffer size for interleaving both types of data
#         self.vertex_data_size = self.vertex_stride * vertex_count
#         self.instance_data_size = self.instance_stride * instance_count
#         self.buffer_size = self.vertex_data_size + self.instance_data_size
#
#         # Initialize the base BufferObject
#         super().__init__(
#             size=self.buffer_size,
#             usage=VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
#             memory_properties=VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
#         )
#         self.create(device)
#
#         # Cache offsets for each attribute
#         self.attribute_offsets = {}
#         vertex_offset = 0
#         instance_offset = self.vertex_data_size  # Instance data starts after all vertex data
#
#         for attr in attributes:
#             if attr.instance:
#                 self.attribute_offsets[attr.name] = instance_offset
#                 instance_offset += attr.element_size * attr.element_count
#             else:
#                 self.attribute_offsets[attr.name] = vertex_offset
#                 vertex_offset += attr.element_size * attr.element_count
#
#     def set_data(self, attribute_name, data, start=0):
#         """Set data for a specific attribute within the interleaved buffer."""
#         offset = self.attribute_offsets.get(attribute_name)
#         if offset is None:
#             raise ValueError(f"Attribute '{attribute_name}' not found in interleaved buffer.")
#
#         # Determine stride based on whether attribute is per-vertex or per-instance
#         stride = self.vertex_stride if attribute_name in [attr.name for attr in
#                                                           self.vertex_attributes] else self.instance_stride
#
#         # Map memory and copy data with the appropriate offset for each attribute
#         mapped_memory = self.map()
#         for i, value in enumerate(data):
#             byte_offset = start * stride + offset + i * stride
#             ctypes.memmove(mapped_memory[byte_offset:], value, len(value))
#         self.unmap()
#
#     def get_binding_descriptions(self):
#         """Generate binding descriptions for the pipeline setup."""
#         binding_descriptions = []
#
#         # Binding 0: Per-vertex attributes
#         if self.vertex_attributes:
#             binding_descriptions.append(VkVertexInputBindingDescription(
#                 binding=0,
#                 stride=self.vertex_stride,
#                 inputRate=VK_VERTEX_INPUT_RATE_VERTEX
#             ))
#
#         # Binding 1: Per-instance attributes
#         if self.instance_attributes:
#             binding_descriptions.append(VkVertexInputBindingDescription(
#                 binding=1,
#                 stride=self.instance_stride,
#                 inputRate=VK_VERTEX_INPUT_RATE_INSTANCE
#             ))
#
#         return binding_descriptions
#
#     def get_attribute_descriptions(self):
#         """Generate attribute descriptions for the pipeline setup."""
#         attribute_descriptions = []
#
#         for attr in self.vertex_attributes:
#             attribute_descriptions.append(VkVertexInputAttributeDescription(
#                 location=attr.location,
#                 binding=0,  # Per-vertex binding
#                 format=attr.vk_format,  # Assume `attr.vk_format` provides the Vulkan format
#                 offset=self.attribute_offsets[attr.name]
#             ))
#
#         for attr in self.instance_attributes:
#             attribute_descriptions.append(VkVertexInputAttributeDescription(
#                 location=attr.location,
#                 binding=1,  # Per-instance binding
#                 format=attr.vk_format,  # Assume `attr.vk_format` provides the Vulkan format
#                 offset=self.attribute_offsets[attr.name]
#             ))
#
#         return attribute_descriptions
#
#     def bind(self, command_buffer, binding_start=0):
#         """Bind the interleaved buffer to the command buffer."""
#         vkCmdBindVertexBuffers(command_buffer, binding_start, 1, [self.vk_buffer], [0])
#
#     def delete(self):
#         """Delete the buffer."""
#         self.delete()

class IndexBufferObject(MappedBufferObject):
    def __init__(self, size: int):
        super().__init__("H", size,
                         usage=VK_BUFFER_USAGE_INDEX_BUFFER_BIT,  # Used for transferring somewhere.
                         memory_properties=VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |  # Allows CPU to map and access memory.
                                           VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)  # Automatic sync CPU<->GPU

    def bind(self, command_buffer: VkCommandBuffer, offset: int = 0) -> None:
        DeviceFunc.vkCmdBindIndexBuffer(command_buffer, self.buffer.vk_buffer, offset, VK_INDEX_TYPE_UINT16)

    def get_data(self) -> ctypes.Array[CType]:
        element_count = self.size // self.c_type_size
        return self.get_data_region(0, element_count)

    def get_data_region(self, start: int, length: int) -> Sequence[int | float]:
        """Read a region of data from the buffer."""
        byte_offset = start * self.c_type_size
        byte_length = length * self.c_type_size

        ptr = self.map_range(byte_offset, byte_length)
        assert ptr is not None, "Memory mapping failed."

        data = (self.c_type * length)()
        ctypes.memmove(data, ptr, byte_length)
        self.unmap()

        return list(data)

    def set_data(self, data: Array[CType]) -> None:
        """Copy data to the buffer.

        Args:
            data: Data to copy, in bytes.
        """
        byte_size = self.c_type_size * len(data)
        self.set_data_region(0, byte_size, data)

    def set_data_region(self, start: int, data: Sequence[int | float]) -> None:
        assert self.devices is not None, "Device not set. Call `create` first."
        assert start + len(data) <= self.size // self.c_type_size, "Data exceeds buffer size."

        byte_offset = start * self.c_type_size
        byte_length = len(data) * self.c_type_size

        ptr = self.map_range(byte_offset, byte_length)
        ctypes.memmove(ptr, (self.c_type * len(data))(*data), byte_length)
        self.unmap()

class StagingBufferObject(MappedBufferObject):
    """A buffer to transfer data from the CPU to device-local memory."""
    def __init__(self, size: int):
        super().__init__("b", size,
                         usage=VK_BUFFER_USAGE_TRANSFER_SRC_BIT,  # One way. CPU -> GPU.
                         memory_properties=VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |  # Allows CPU to map and access memory.
                                           VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)  # Automatic sync CPU<->GPU
    def set_data_as_type(self, data: bytes, ctype_type) -> None:
        """Copy data to the buffer.

        Args:
            data: Data to copy, in bytes.
        """
        assert self.devices is not None, "Device not set. Call `create` first."
        assert len(data) == self.size, f"Data size: {len(data)} exceeds buffer size: {self.size}."
        logical = self.devices.logical_device
        vk_logical = logical.vk_device

        mapped_ptr = c_void_p()
        logical.vkMapMemory(vk_logical, self.buffer.vk_device_memory, 0, self.size, 0, byref(mapped_ptr))
        memmove(mapped_ptr, data, self.size)
        logical.vkUnmapMemory(vk_logical, self.buffer.vk_device_memory)

class StagingBufferObject2(MappedBufferObject):
    """A buffer to transfer data from the CPU to device-local memory."""
    def __init__(self, size: int):
        super().__init__("b", size,
                         usage=VK_BUFFER_USAGE_TRANSFER_DST_BIT,  # One way. CPU -> GPU.
                         memory_properties=VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |  # Allows CPU to map and access memory.
                                           VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)  # Automatic sync CPU<->GPU
