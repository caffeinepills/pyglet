from __future__ import annotations

from typing import TYPE_CHECKING, Sequence
from ctypes import byref

from pyglet.graphics.api.vulkan import DeviceFunc, c_array_list
from pyglet.graphics.api.vulkan.sync import create_fence
from pyglet.libs.shared.vulkan_lib.vulkan_core import VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT, \
    VkCommandBufferBeginInfo, \
    VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, VkCommandBuffer, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, \
    VkCommandPoolCreateInfo, \
    VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, VkCommandBufferAllocateInfo, \
    VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, VK_COMMAND_BUFFER_LEVEL_PRIMARY, VkCommandPool, \
    VkCommandBufferLevel, VkCommandBufferUsageFlagBits, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, VkSubmitInfo, \
    VK_STRUCTURE_TYPE_SUBMIT_INFO, VK_NULL_HANDLE, VkQueue, VkFence, VkFenceCreateInfo, \
    VK_STRUCTURE_TYPE_FENCE_CREATE_INFO

if TYPE_CHECKING:
    from pyglet.graphics.api.vulkan.devices import VulkanLogicalDevice, QueueFamily


class CommandBuffer:
    __slots__ = ("command_buffer", "begin_info")
    # VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT: The command buffer will be rerecorded right after executing it once.
    # VK_COMMAND_BUFFER_USAGE_RENDER_PASS_CONTINUE_BIT: This is a secondary command buffer that will be entirely within a single render pass.
    # VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT:  The command buffer can be resubmitted while it is also already pending execution.

    def __init__(self, command_buffer: VkCommandBuffer,
                 usage_flags: VkCommandBufferUsageFlagBits =VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT):
        """Initializes the CommandBuffer wrapper with a Vulkan command buffer handle."""
        self.command_buffer = command_buffer
        self.begin_info = VkCommandBufferBeginInfo(
            sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            flags=usage_flags,
        )

    def begin(self):
        DeviceFunc.vkBeginCommandBuffer(self.command_buffer, byref(self.begin_info))

    def end(self):
        DeviceFunc.vkEndCommandBuffer(self.command_buffer)

    def reset(self):
        DeviceFunc.vkResetCommandBuffer(self.command_buffer, 0)

    def __enter__(self):
        """Begins recording the command buffer using cached begin_info and enters the context."""
        DeviceFunc.vkBeginCommandBuffer(self.command_buffer, byref(self.begin_info))
        return self.command_buffer

    def __exit__(self, exc_type, exc_value, traceback):
        """Ends recording the command buffer and exits the context."""
        DeviceFunc.vkEndCommandBuffer(self.command_buffer)


class SingleTimeCommandBuffer(CommandBuffer):
    __slots__ = ("command_buffer", "begin_info", "vk_queue", "vk_fence")

    def __init__(self, command_buffer: VkCommandBuffer, queue: VkQueue, fence: VkFence = VK_NULL_HANDLE) -> None:
        self.vk_queue = queue
        self.vk_fence = fence
        super().__init__(command_buffer, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT)

    def __exit__(self, exc_type, exc_value, traceback):
        """Ends recording the command buffer and exits the context."""
        DeviceFunc.vkEndCommandBuffer(self.command_buffer)

        cmd_buff_array = c_array_list([self.command_buffer], VkCommandBuffer)
        submit_info = VkSubmitInfo(
            sType=VK_STRUCTURE_TYPE_SUBMIT_INFO,
            commandBufferCount=1,
            pCommandBuffers=cmd_buff_array,
        )

        #print("Submitting command buffer for clear and transition...")
        submit_info = [submit_info]
        submit_info_array = c_array_list(submit_info, VkSubmitInfo)

        DeviceFunc.vkQueueSubmit(self.vk_queue, 1, submit_info_array, self.vk_fence)
        if not self.vk_fence:
            DeviceFunc.vkQueueWaitIdle(self.vk_queue)
        #print("Command buffer execution completed.")


class SingleUseCommandBufferGroup:
    def __init__(self, device: VulkanLogicalDevice, command_pool: VkCommandPool,
                 vk_cmdbuffers: list[VkCommandBuffer], vk_queue: VkQueue,
                 fence: bool = False) -> None:
        """Create a group of single use buffers.

        Args:
            device:
                The logical Vulkan device.
            command_pool:
                The command pool that generated these VkCommandBuffers.
            vk_cmdbuffers:
                A sequence of VkCommandBuffer objects.
            vk_queue:
                The queue to use for command buffers.
            fence:
                If true, will create a fence to be used for the submission instead of waiting.
        """
        self.device = device
        self.command_pool = command_pool
        self.vk_queue = vk_queue
        self._vk_cmdbuffer_array = c_array_list(vk_cmdbuffers, VkCommandBuffer)
        self._buffers = [SingleTimeCommandBuffer(vk_cmdbuffer, vk_queue) for vk_cmdbuffer in vk_cmdbuffers]
        self.vk_fence: VkFence | VK_NULL_HANDLE = VK_NULL_HANDLE

        if fence:
            fence_info = VkFenceCreateInfo(
                sType=VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
                flags=0,
            )
            self.vk_fence = create_fence(self.device, fence_info)
        self._active = True

    @property
    def buffers(self) -> list[SingleTimeCommandBuffer]:
        return self._buffers

    def submit(self) -> VkFence | VK_NULL_HANDLE:
        submit_info = VkSubmitInfo(
            sType=VK_STRUCTURE_TYPE_SUBMIT_INFO,
            commandBufferCount=len(self._buffers),
            pCommandBuffers=self._vk_cmdbuffer_array,
        )

        submit_info = [submit_info]
        submit_info_array = c_array_list(submit_info, VkSubmitInfo)

        DeviceFunc.vkQueueSubmit(self.vk_queue, 1, submit_info_array, self.vk_fence)
        if not self.vk_fence:
            DeviceFunc.vkQueueWaitIdle(self.vk_queue)

        return self.vk_fence

    def __del__(self) -> None:
        self.free()

    def free(self) -> None:
        """Free all command buffers in the group."""
        if self._active:
            DeviceFunc.vkFreeCommandBuffers(
                self.device.vk_device, self.command_pool, len(self._buffers), self._vk_cmdbuffer_array,
            )

        self._active = False
        self.vk_fence = None
        self.device = None
        self.command_pool = None
        self.vk_queue = None
        self._vk_cmdbuffer_array = None
        self._buffers.clear()


class CommandPool:
    def __init__(self, logical_device: VulkanLogicalDevice) -> None:
        self.device = logical_device
        self.command_pool = None
        self.vkCreateCommandPool = None
        self.queue_family = None
        self.command_buffers = []

    def create(self, flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT):  # Allow buffers to be reset
        """Creates a command pool for allocating command buffers."""
        if self.command_pool:
            return

        self.queue_family = self.device.graphics_queue

        pool_info = VkCommandPoolCreateInfo(
            sType=VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            queueFamilyIndex=self.queue_family.indice,
            flags=flags,
        )

        self.command_pool = VkCommandPool()
        DeviceFunc.vkCreateCommandPool(self.device.vk_device, byref(pool_info), None, byref(self.command_pool))

        self.vkCreateCommandPool = self.device.vkCreateCommandPool

    def _ensure_created(self) -> None:
        if self.command_pool is None:
            self.create()

    def allocate_command_buffers(self, buffer_count: int,
                                 level: VkCommandBufferLevel = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                                 flags: VkCommandBufferUsageFlagBits = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT) -> list[CommandBuffer]:
        """Allocates command buffers from the command pool."""
        self._ensure_created()

        alloc_info = VkCommandBufferAllocateInfo(
            sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            commandPool=self.command_pool,
            level=level,
            commandBufferCount=buffer_count,
        )
        vk_command_buffers = (VkCommandBuffer * buffer_count)()
        DeviceFunc.vkAllocateCommandBuffers(self.device.vk_device, byref(alloc_info), vk_command_buffers)
        self.command_buffers = [CommandBuffer(cmd_buffer, flags) for cmd_buffer in vk_command_buffers]
        return self.command_buffers

    def get_single_use_group(self, buffer_count: int) -> SingleUseCommandBufferGroup:
        self._ensure_created()

        alloc_info = VkCommandBufferAllocateInfo(
            sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            commandPool=self.command_pool,
            level=VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            commandBufferCount=buffer_count,
        )

        vk_command_buffers = (VkCommandBuffer * buffer_count)()
        DeviceFunc.vkAllocateCommandBuffers(self.device.vk_device, byref(alloc_info), vk_command_buffers)

        return SingleUseCommandBufferGroup(self.device, self.command_pool,
                                               vk_command_buffers,
                                               self.queue_family.vk_queue,
                                               )

    def get_single_use_fence(self) -> tuple[SingleTimeCommandBuffer, VkFence]:
        self._ensure_created()

        alloc_info = VkCommandBufferAllocateInfo(
            sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            commandPool=self.command_pool,
            level=VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            commandBufferCount=1,
        )

        vk_command_buffers = (VkCommandBuffer * 1)()
        DeviceFunc.vkAllocateCommandBuffers(self.device.vk_device, byref(alloc_info), vk_command_buffers)

        fence_info = VkFenceCreateInfo(
            sType=VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
            flags=0,
        )
        vk_fence = create_fence(self.device, fence_info)
        return SingleTimeCommandBuffer(vk_command_buffers[0], self.queue_family.vk_queue, vk_fence), vk_fence

    def get_single_use(self, buffer_count: int) -> list[SingleTimeCommandBuffer]:
        self._ensure_created()

        alloc_info = VkCommandBufferAllocateInfo(
            sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            commandPool=self.command_pool,
            level=VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            commandBufferCount=buffer_count,
        )

        vk_command_buffers = (VkCommandBuffer * buffer_count)()
        DeviceFunc.vkAllocateCommandBuffers(self.device.vk_device, byref(alloc_info), vk_command_buffers)

        return [SingleTimeCommandBuffer(cmd_buffer, self.queue_family.vk_queue) for cmd_buffer in vk_command_buffers]

    def free(self, command_buffers: list[CommandBuffer | SingleTimeCommandBuffer]):
        if not self.command_pool:
            return
        command_buff_ct = len(command_buffers)
        vkcmd_buffers = [cb.command_buffer for cb in command_buffers]
        command_buff_array = c_array_list(vkcmd_buffers, VkCommandBuffer)
        DeviceFunc.vkFreeCommandBuffers(self.device.vk_device, self.command_pool, command_buff_ct, command_buff_array)

    def __del__(self):
        self.delete()

    def delete(self) -> None:
        """Cleans up the command pool and buffers.

        Buffers get freed with the pool.
        """
        vk_device = getattr(self.device, "vk_device", None) if self.device is not None else None
        if self.command_pool and vk_device:
            self.device.vkDestroyCommandPool(vk_device, self.command_pool, None)
            self.command_pool = None
        elif self.command_pool:
            self.command_pool = None
        self.device = None
        self.command_buffers.clear()
