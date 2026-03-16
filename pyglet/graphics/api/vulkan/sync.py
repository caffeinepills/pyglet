from __future__ import annotations


from _ctypes import byref
from ctypes import c_uint32
from typing import TYPE_CHECKING
from pyglet.graphics.api.vulkan import c_array_list, DeviceFunc
from pyglet.libs.shared.vulkan_lib.exceptions import VulkanNotReadyException

from pyglet.libs.shared.vulkan_lib.vulkan_core import VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, \
    VkSemaphoreCreateInfo, VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, VkFenceCreateInfo, \
    VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, VK_FENCE_CREATE_SIGNALED_BIT, VkSemaphore, VkFence, VK_TRUE, VkCommandBuffer, \
    VkSubmitInfo, VK_STRUCTURE_TYPE_SUBMIT_INFO, VkSwapchainKHR, VkPresentInfoKHR, VK_STRUCTURE_TYPE_PRESENT_INFO_KHR, \
    UINT64_MAX, VK_SUCCESS

if TYPE_CHECKING:
    from pyglet.graphics.api.vulkan.swapchain import VulkanSwapchain
    from pyglet.graphics.api.vulkan.descriptor import DescriptorManager
    from pyglet.graphics.api.vulkan.devices import VulkanLogicalDevice
    from pyglet.graphics.api.vulkan.commands import CommandBuffer, CommandPool


def create_semaphore(device: VulkanLogicalDevice, info: VkSemaphoreCreateInfo):
    vk_semaphore = VkSemaphore()
    device.vkCreateSemaphore(device.vk_device, byref(info), None, byref(vk_semaphore))
    return vk_semaphore


def create_fence(device: VulkanLogicalDevice, info: VkFenceCreateInfo):
    vk_fence = VkFence()
    device.vkCreateFence(device.vk_device, byref(info), None, byref(vk_fence))
    return vk_fence


class DeferredResourceRemoval:
    """Some resources may need to be deferred for removal until after the frame is completed."""

    def __init__(self, device: VulkanLogicalDevice) -> None:
        self.device = device
        self.pending_fences = []

    def queue_fence(self, resource, fence: VkFence, destroy: bool):
        """Queue a resource with a fence to be destroyed after the frame is completed.

        If destroy is True, then it's expected the fence needs to be destroyed after the resource is.
        """
        self.pending_fences.append((resource, fence, destroy))

    def delete(self):
        """Clear all pending on API cleanup."""
        while self.pending_fences:
            resource, fence, destroy = self.pending_fences.pop()
            if destroy:
                self.device.vkDestroyFence(self.device.vk_device, fence, None)
            resource.delete()

    def process(self):
        for resource, fence, destroy in list(self.pending_fences):
            try:
                status = self.device.vkGetFenceStatus(self.device.vk_device, fence)
            except VulkanNotReadyException:
                continue

            if status == VK_SUCCESS:
                print("Destroying resource", resource)
                resource.delete()  # Destroy the resource
                if destroy:
                    self.device.vkDestroyFence(self.device.vk_device, fence, None)
                self.pending_fences.remove((resource, fence, destroy))


        # for resource in self.pending_resources:
        #     resource.delete()
        # self.pending_resources.clear()


class FrameSync:
    command_buffers: list[dict[int, CommandBuffer]]

    def __init__(self, device: VulkanLogicalDevice,
                 swapchain: VulkanSwapchain,
                 descriptor_mgr: DescriptorManager,
                 command_pool: CommandPool,
                 frames_in_flight: int,
                 stats: bool = True,
                 ):
        self.current_frame = 0
        self.frames_in_flight = frames_in_flight
        self.device = device
        self.swapchain = swapchain
        self.command_pool = command_pool
        self.command_buffers = [{} for _ in range(frames_in_flight)]
        self.image_index = [0 for _ in range(frames_in_flight)]
        self.counter = 0
        self._wait_stages = c_array_list([VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT], c_uint32)

        self.vkAcquireNextImageKHR = DeviceFunc.vkAcquireNextImageKHR
        self.vkQueuePresentKHR = DeviceFunc.vkQueuePresentKHR

        semaphore_info = VkSemaphoreCreateInfo(
            sType=VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
            flags=0,
        )

        self.command_pool.create()

        # Create a descriptor pool for resources.
        descriptor_mgr.create_pool(frames_in_flight, max_sets=20)

        fence_info = VkFenceCreateInfo(
            sType=VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
            flags=VK_FENCE_CREATE_SIGNALED_BIT,
        )

        # Wait semaphores
        self.image_available_semaphores = [create_semaphore(device, semaphore_info) for _ in range(frames_in_flight)]

        # Signal semaphores
        self.render_finished_semaphores = [create_semaphore(device, semaphore_info) for _ in range(frames_in_flight)]

        self.fences = [create_fence(device, fence_info) for _ in range(frames_in_flight)]

        # query_pool_info = VkQueryPoolCreateInfo(
        #     sType=VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
        #     queryType=VK_QUERY_TYPE_TIMESTAMP,
        #     queryCount=2,
        # )
        #
        # self.query_pool = VkQueryPool()
        # DeviceFunc.vkCreateQueryPool(device.vk_device, byref(query_pool_info), None, byref(self.query_pool))

        self.vkWaitForFences = DeviceFunc.vkWaitForFences
        self.vkResetFences = DeviceFunc.vkResetFences
        self.vkQueueSubmit = DeviceFunc.vkQueueSubmit

        # Allocate initial command buffers.
        self.get_command_buffer()

    def get_ubo(self):
        pass

    def allocate_descriptor_set(self):
        pass

    def get_command_buffer(self) -> int:
        """Creates a command buffer that will exist in each frame in flight."""
        allocated_buffers = self.command_pool.allocate_command_buffers(self.frames_in_flight)
        cmd_buf_id = self.counter

        for i in range(self.frames_in_flight):
            self.command_buffers[i][cmd_buf_id] = allocated_buffers[i]

        self.counter +=1
        return cmd_buf_id

    def get_current_command_buffer(self, idx: int) -> CommandBuffer:
        return self.command_buffers[self.current_frame][idx]

    def get_frame_resources(self):
        return

    def before_draw(self):
        vk_device = self.device.vk_device
        fence_array = c_array_list([self.fences[self.current_frame]], VkFence)

        # DeviceFunc.vkCmdWriteTimestamp(
        #     self.get_current_command_buffer(0).command_buffer,
        #     VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        #     self.query_pool, 0,
        # )

        self.vkWaitForFences(vk_device, 1, fence_array, VK_TRUE, UINT64_MAX)
        self.vkResetFences(vk_device, 1, fence_array)

        img_index = c_uint32()
        self.vkAcquireNextImageKHR(
            vk_device,
            self.swapchain.swapchain,
            UINT64_MAX,
            self.image_available_semaphores[self.current_frame],
            0,
            byref(img_index),
        )

        self.image_index[self.current_frame] = img_index.value

    def current_image_index(self) -> int:
        return self.image_index[self.current_frame]

    def flip(self):
        signal_semaphores = c_array_list([self.render_finished_semaphores[self.current_frame]], VkSemaphore)
        wait_semaphores = c_array_list([self.image_available_semaphores[self.current_frame]], VkSemaphore)

        cmd_buffers = c_array_list([cb.command_buffer for cb in self.command_buffers[self.current_frame].values()], VkCommandBuffer)
        command_count = len(self.command_buffers[self.current_frame])

        submit_create = VkSubmitInfo(
            sType=VK_STRUCTURE_TYPE_SUBMIT_INFO,
            waitSemaphoreCount=1,
            pWaitSemaphores=wait_semaphores,
            pWaitDstStageMask=self._wait_stages,
            commandBufferCount=command_count,
            pCommandBuffers=cmd_buffers,
            signalSemaphoreCount=1,
            pSignalSemaphores=signal_semaphores)

        submit_array = c_array_list([submit_create], VkSubmitInfo)

        self.vkQueueSubmit(self.device.graphics_queue.vk_queue,
                      1, submit_array, self.fences[self.current_frame])

        swapchains = c_array_list([self.swapchain.swapchain], VkSwapchainKHR)
        image_indices = c_array_list([self.image_index[self.current_frame]], c_uint32)

        present_create = VkPresentInfoKHR(
            sType=VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
            waitSemaphoreCount=1,
            pWaitSemaphores=signal_semaphores,
            swapchainCount=1,
            pSwapchains=swapchains,
            pImageIndices=image_indices)

        self.vkQueuePresentKHR(self.device.present_queue.vk_queue, byref(present_create))

        # DeviceFunc.vkCmdWriteTimestamp(
        #     self.get_current_command_buffer(0).command_buffer,
        #     VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
        #     self.query_pool, 1,
        # )
        #
        # results = (ctypes.c_uint64 * 2)()
        # DeviceFunc.vkGetQueryPoolResults(
        #     self.device.vk_device,
        #     self.query_pool,
        #     firstQuery=0,
        #     queryCount=2,
        #     dataSize=ctypes.sizeof(results),
        #     pData=ctypes.byref(results),
        #     stride=ctypes.sizeof(ctypes.c_uint64),
        #     flags=VK_QUERY_RESULT_64_BIT,
        # )
        # # Calculate elapsed time
        # start_time = results[0]
        # end_time = results[1]
        # elapsed_ns = end_time - start_time
        # elapsed_ms = elapsed_ns / 1_000_000  # Convert to milliseconds
        # fps = 1000.0 / elapsed_ms
        # print(f"Frame Time: {elapsed_ms:.2f} ms, FPS: {fps:.2f}")

        # Advance the frame element_count.
        self.current_frame = (self.current_frame + 1) % self.frames_in_flight

    def __del__(self) -> None:
        self.delete()

    def delete(self) -> None:
        for semaphore in self.image_available_semaphores:
            self.device.vkDestroySemaphore(self.device.vk_device, semaphore, None)
        self.image_available_semaphores.clear()

        for semaphore in self.render_finished_semaphores:
            self.device.vkDestroySemaphore(self.device.vk_device, semaphore, None)
        self.render_finished_semaphores.clear()

        for fence in self.fences:
            self.device.vkDestroyFence(self.device.vk_device, fence, None)
        self.fences.clear()

        if self.command_pool:
            self.command_pool.delete()
            self.command_pool = None
            self.command_buffers.clear()
