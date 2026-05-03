from __future__ import annotations


from _ctypes import byref
import ctypes
from ctypes import POINTER, c_uint32, c_uint64
from typing import TYPE_CHECKING
import pyglet
from pyglet.graphics.api.vulkan import c_array_list, DeviceFunc
from pyglet.libs.shared.vulkan_lib.exceptions import VulkanNotReadyException

from pyglet.libs.shared.vulkan_lib.vulkan_core import VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, \
    VkSemaphoreCreateInfo, VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, VkFenceCreateInfo, \
    VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, VK_FENCE_CREATE_SIGNALED_BIT, VkSemaphore, VkFence, VK_TRUE, VkCommandBuffer, \
    VkSubmitInfo, VK_STRUCTURE_TYPE_SUBMIT_INFO, VkSwapchainKHR, VkPresentInfoKHR, VK_STRUCTURE_TYPE_PRESENT_INFO_KHR, \
    UINT64_MAX, VK_SUCCESS, VkSemaphoreTypeCreateInfo, VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO, \
    VK_SEMAPHORE_TYPE_TIMELINE, VkTimelineSemaphoreSubmitInfo, VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO, \
    VkSemaphoreWaitInfo, VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO, VK_NULL_HANDLE

if TYPE_CHECKING:
    from pyglet.graphics.api.vulkan.swapchain import VulkanSwapchain
    from pyglet.graphics.api.vulkan.descriptor import DescriptorManager
    from pyglet.graphics.api.vulkan.devices import VulkanLogicalDevice
    from pyglet.graphics.api.vulkan.commands import CommandBuffer, CommandPool

_debug_api = pyglet.options.debug_api

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

    @staticmethod
    def _is_callable_loaded(func) -> bool:
        return callable(func) and getattr(func, "__name__", "") != "MissingFunction"

    def _resolve_wait_semaphores(self):
        candidates = (
            ("vkWaitSemaphores", getattr(self.device, "vkWaitSemaphores", None)),
            ("vkWaitSemaphoresKHR", getattr(self.device, "vkWaitSemaphoresKHR", None)),
            ("vkWaitSemaphores", getattr(DeviceFunc, "vkWaitSemaphores", None)),
            ("vkWaitSemaphoresKHR", getattr(DeviceFunc, "vkWaitSemaphoresKHR", None)),
        )
        for name, wait_fn in candidates:
            if self._is_callable_loaded(wait_fn):
                return wait_fn, name
        return None, None

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
        self.image_fences: list[VkFence | None] = []
        self.image_sync_values: list[int] = []
        self.counter = 0
        self._wait_stages = c_array_list([VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT], c_uint32)
        self.vkWaitSemaphores, self._wait_semaphore_fn_name = self._resolve_wait_semaphores()
        self.timeline_enabled = device.timeline_semaphore_enabled and self.vkWaitSemaphores is not None
        self.timeline_semaphore: VkSemaphore | None = None
        self.timeline_value = 0
        self.frame_sync_values = [0 for _ in range(frames_in_flight)]

        self.vkAcquireNextImageKHR = getattr(device, "vkAcquireNextImageKHR", DeviceFunc.vkAcquireNextImageKHR)
        self.vkQueuePresentKHR = getattr(device, "vkQueuePresentKHR", DeviceFunc.vkQueuePresentKHR)

        if _debug_api:
            if self.timeline_enabled:
                print(
                    "(Vulkan) Frame sync using timeline semaphore path "
                    f"(wait_fn={self._wait_semaphore_fn_name})."
                )
            elif device.timeline_semaphore_enabled:
                print("(Vulkan) Timeline semaphore feature detected, but no wait function is available; using fences.")
            else:
                print("(Vulkan) Frame sync using fence path.")

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

        # Signal semaphores are tracked by swapchain image index (not frame index),
        # so reuse only occurs after that exact image is acquired again.
        swapchain_image_count = len(self.swapchain.swapchain_images)
        self.render_finished_semaphores = [create_semaphore(device, semaphore_info) for _ in range(swapchain_image_count)]
        self.image_fences = [None] * swapchain_image_count
        self.image_sync_values = [0] * swapchain_image_count

        if self.timeline_enabled:
            timeline_type_info = VkSemaphoreTypeCreateInfo(
                sType=VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
                pNext=None,
                semaphoreType=VK_SEMAPHORE_TYPE_TIMELINE,
                initialValue=0,
            )
            timeline_semaphore_info = VkSemaphoreCreateInfo(
                sType=VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
                pNext=ctypes.cast(ctypes.pointer(timeline_type_info), POINTER(None)),
                flags=0,
            )
            self.timeline_semaphore = create_semaphore(device, timeline_semaphore_info)
            self.fences = []
        else:
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

    def _wait_timeline(self, value: int) -> None:
        if not self.timeline_enabled or value <= 0 or self.timeline_semaphore is None:
            return

        wait_semaphores = c_array_list([self.timeline_semaphore], VkSemaphore)
        wait_values = c_array_list([value], c_uint64)
        wait_info = VkSemaphoreWaitInfo(
            sType=VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
            pNext=None,
            flags=0,
            semaphoreCount=1,
            pSemaphores=wait_semaphores,
            pValues=wait_values,
        )
        self.vkWaitSemaphores(self.device.vk_device, byref(wait_info), UINT64_MAX)

    def before_draw(self):
        vk_device = self.device.vk_device
        if self.timeline_enabled:
            frame_value = self.frame_sync_values[self.current_frame]
            if frame_value:
                self._wait_timeline(frame_value)
        else:
            fence_array = c_array_list([self.fences[self.current_frame]], VkFence)
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
        image_idx = img_index.value

        if self.timeline_enabled:
            image_value = self.image_sync_values[image_idx]
            if image_value and image_value != self.frame_sync_values[self.current_frame]:
                self._wait_timeline(image_value)
        else:
            # If this image is already in flight from another frame, wait for that frame to complete.
            image_fence = self.image_fences[image_idx]
            if image_fence is not None and image_fence != self.fences[self.current_frame]:
                image_fence_array = c_array_list([image_fence], VkFence)
                self.vkWaitForFences(vk_device, 1, image_fence_array, VK_TRUE, UINT64_MAX)

            self.image_fences[image_idx] = self.fences[self.current_frame]

    def current_image_index(self) -> int:
        return self.image_index[self.current_frame]

    def flip(self):
        image_idx = self.image_index[self.current_frame]
        wait_semaphores = c_array_list([self.image_available_semaphores[self.current_frame]], VkSemaphore)

        cmd_buffers = c_array_list([cb.command_buffer for cb in self.command_buffers[self.current_frame].values()], VkCommandBuffer)
        command_count = len(self.command_buffers[self.current_frame])

        submit_pnext = None
        queue_fence = VK_NULL_HANDLE
        signal_list = [self.render_finished_semaphores[image_idx]]
        timeline_signal_value = None

        timeline_submit_info = None
        if self.timeline_enabled and self.timeline_semaphore is not None:
            timeline_signal_value = self.timeline_value + 1

            signal_list.append(self.timeline_semaphore)

            # Values map one-to-one with VkSubmitInfo semaphores.
            wait_values = c_array_list([0], c_uint64)
            signal_values = c_array_list([0, timeline_signal_value], c_uint64)
            timeline_submit_info = VkTimelineSemaphoreSubmitInfo(
                sType=VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO,
                pNext=None,
                waitSemaphoreValueCount=1,
                pWaitSemaphoreValues=wait_values,
                signalSemaphoreValueCount=len(signal_list),
                pSignalSemaphoreValues=signal_values,
            )
            submit_pnext = ctypes.cast(ctypes.pointer(timeline_submit_info), POINTER(None))
        else:
            queue_fence = self.fences[self.current_frame]

        signal_semaphores = c_array_list(signal_list, VkSemaphore)
        submit_create = VkSubmitInfo(
            sType=VK_STRUCTURE_TYPE_SUBMIT_INFO,
            pNext=submit_pnext,
            waitSemaphoreCount=1,
            pWaitSemaphores=wait_semaphores,
            pWaitDstStageMask=self._wait_stages,
            commandBufferCount=command_count,
            pCommandBuffers=cmd_buffers,
            signalSemaphoreCount=len(signal_list),
            pSignalSemaphores=signal_semaphores)

        submit_array = c_array_list([submit_create], VkSubmitInfo)

        self.vkQueueSubmit(self.device.graphics_queue.vk_queue,
                  1, submit_array, queue_fence)

        if timeline_signal_value is not None:
            self.timeline_value = timeline_signal_value
            self.frame_sync_values[self.current_frame] = timeline_signal_value
            self.image_sync_values[image_idx] = timeline_signal_value

        swapchains = c_array_list([self.swapchain.swapchain], VkSwapchainKHR)
        image_indices = c_array_list([image_idx], c_uint32)

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

        if self.timeline_semaphore:
            self.device.vkDestroySemaphore(self.device.vk_device, self.timeline_semaphore, None)
            self.timeline_semaphore = None

        for fence in self.fences:
            self.device.vkDestroyFence(self.device.vk_device, fence, None)
        self.fences.clear()

        if self.command_pool:
            self.command_pool.delete()
            self.command_pool = None
            self.command_buffers.clear()
