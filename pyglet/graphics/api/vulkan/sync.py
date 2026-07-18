from __future__ import annotations


from _ctypes import byref
import ctypes
from ctypes import POINTER, c_uint32, c_uint64
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable
import pyglet
from pyglet.graphics.api.vulkan import c_array_list, DeviceFunc
from pyglet.libs.shared.vulkan_lib.exceptions import VulkanNotReadyException, VulkanOutOfDateKHRException

from pyglet.libs.shared.vulkan_lib.vulkan_core import VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, \
    VkSemaphoreCreateInfo, VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, VkFenceCreateInfo, \
    VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, VK_FENCE_CREATE_SIGNALED_BIT, VkSemaphore, VkFence, VK_TRUE, VkCommandBuffer, \
    VkSubmitInfo, VK_STRUCTURE_TYPE_SUBMIT_INFO, VkSwapchainKHR, VkPresentInfoKHR, VK_STRUCTURE_TYPE_PRESENT_INFO_KHR, \
    UINT64_MAX, VK_SUCCESS, VkSemaphoreTypeCreateInfo, VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO, \
    VK_SEMAPHORE_TYPE_TIMELINE, VkTimelineSemaphoreSubmitInfo, VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO, \
    VkSemaphoreWaitInfo, VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO, VK_NULL_HANDLE, \
    VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, \
    VK_IMAGE_LAYOUT_UNDEFINED, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, \
    VK_QUEUE_FAMILY_IGNORED, VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, VkImageMemoryBarrier, VkImageSubresourceRange

if TYPE_CHECKING:
    from pyglet.graphics.api.vulkan.swapchain import VulkanOffscreenSwapchain, VulkanSwapchain
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
        self.pending_timeline = []
        self.vkGetSemaphoreCounterValue, self._semaphore_counter_fn_name = self._resolve_semaphore_counter_fn()

    def _resolve_semaphore_counter_fn(self):
        candidates = (
            ("vkGetSemaphoreCounterValue", getattr(self.device, "vkGetSemaphoreCounterValue", None)),
            ("vkGetSemaphoreCounterValueKHR", getattr(self.device, "vkGetSemaphoreCounterValueKHR", None)),
            ("vkGetSemaphoreCounterValue", getattr(DeviceFunc, "vkGetSemaphoreCounterValue", None)),
            ("vkGetSemaphoreCounterValueKHR", getattr(DeviceFunc, "vkGetSemaphoreCounterValueKHR", None)),
        )
        for name, fn in candidates:
            if _is_callable_loaded(fn):
                return fn, name
        return None, None

    def queue_fence(self, resource, fence: VkFence, destroy: bool):
        """Queue a resource with one fence to be destroyed after completion."""
        self.queue_fences(resource, (fence,), destroy)

    def queue_fences(self, resource, fences: tuple[VkFence, ...] | list[VkFence], destroy: bool):
        """Queue a resource to be destroyed after all fences are signaled."""
        valid_fences = tuple(fence for fence in fences if fence)
        if not valid_fences:
            resource.delete()
            return

        self.pending_fences.append((resource, valid_fences, destroy))

    def queue_timeline(self, resource, semaphore: VkSemaphore | None, value: int, destroy: bool, wait_fallback=None):
        """Queue a resource to be destroyed when timeline semaphore reaches value."""
        if semaphore is None or value <= 0:
            resource.delete()
            return

        self.pending_timeline.append((resource, semaphore, int(value), destroy, wait_fallback))

    def queue_frame_sync(self, resource, frame_sync, destroy: bool):
        """Queue a resource against the active frame sync strategy."""
        if getattr(frame_sync, "timeline_enabled", False):
            self.queue_timeline(
                resource,
                getattr(frame_sync, "timeline_semaphore", None),
                int(getattr(frame_sync, "timeline_value", 0) or 0),
                destroy,
                getattr(frame_sync, "wait_for_last_submission", None),
            )
            return

        self.queue_fences(resource, tuple(getattr(frame_sync, "fences", ()) or ()), destroy)

    def delete(self):
        """Clear all pending on API cleanup."""
        self.drain(wait=False)

    @staticmethod
    def _resource_owned_by(resource, owner) -> bool:
        if owner is None:
            return True
        return getattr(resource, "owner_context", None) is owner

    def _wait_for_fences(self, fences: tuple[VkFence, ...]) -> None:
        if not fences:
            return
        fence_array = c_array_list(list(fences), VkFence)
        self.device.vkWaitForFences(self.device.vk_device, len(fences), fence_array, VK_TRUE, UINT64_MAX)

    def drain(self, wait: bool = False, owner=None) -> None:
        """Destroy pending resources, optionally limiting to one owner context."""
        for resource, fences, destroy in list(self.pending_fences):
            if not self._resource_owned_by(resource, owner):
                continue
            if wait:
                self._wait_for_fences(fences)
            if destroy:
                for fence in fences:
                    self.device.vkDestroyFence(self.device.vk_device, fence, None)
            resource.delete()
            self.pending_fences.remove((resource, fences, destroy))

        for resource, semaphore, target_value, destroy, wait_fallback in list(self.pending_timeline):
            if not self._resource_owned_by(resource, owner):
                continue
            if wait and callable(wait_fallback):
                wait_fallback()
            if destroy and semaphore:
                self.device.vkDestroySemaphore(self.device.vk_device, semaphore, None)
            resource.delete()
            self.pending_timeline.remove((resource, semaphore, target_value, destroy, wait_fallback))

    def drain_context(self, context, wait: bool = False) -> None:
        """Destroy pending resources owned by one surface context."""
        self.drain(wait=wait, owner=context)

    def process(self):
        for resource, fences, destroy in list(self.pending_fences):
            all_signaled = True
            for fence in fences:
                try:
                    status = self.device.vkGetFenceStatus(self.device.vk_device, fence)
                except VulkanNotReadyException:
                    all_signaled = False
                    break

                if status != VK_SUCCESS:
                    all_signaled = False
                    break

            if all_signaled:
                print("Destroying resource", resource)
                resource.delete()
                if destroy:
                    for fence in fences:
                        self.device.vkDestroyFence(self.device.vk_device, fence, None)
                self.pending_fences.remove((resource, fences, destroy))

        for resource, semaphore, target_value, destroy, wait_fallback in list(self.pending_timeline):
            reached = False
            if self.vkGetSemaphoreCounterValue is not None:
                current_value = c_uint64(0)
                try:
                    result = self.vkGetSemaphoreCounterValue(self.device.vk_device, semaphore, byref(current_value))
                    reached = result == VK_SUCCESS and int(current_value.value) >= int(target_value)
                except VulkanNotReadyException:
                    reached = False
                except Exception:
                    reached = False
            elif callable(wait_fallback):
                # Fallback if semaphore counter query is unavailable on this loader/runtime.
                wait_fallback()
                reached = True

            if reached:
                print("Destroying resource", resource)
                resource.delete()
                if destroy and semaphore:
                    self.device.vkDestroySemaphore(self.device.vk_device, semaphore, None)
                self.pending_timeline.remove((resource, semaphore, target_value, destroy, wait_fallback))


        # for resource in self.pending_resources:
        #     resource.delete()
        # self.pending_resources.clear()


def _is_callable_loaded(func: Any) -> bool:
    return callable(func) and getattr(func, "__name__", "") != "MissingFunction"


def _resolve_wait_semaphores(device: VulkanLogicalDevice) -> tuple[Callable[..., Any] | None, str | None]:
    candidates = (
        ("vkWaitSemaphores", getattr(device, "vkWaitSemaphores", None)),
        ("vkWaitSemaphoresKHR", getattr(device, "vkWaitSemaphoresKHR", None)),
        ("vkWaitSemaphores", getattr(DeviceFunc, "vkWaitSemaphores", None)),
        ("vkWaitSemaphoresKHR", getattr(DeviceFunc, "vkWaitSemaphoresKHR", None)),
    )
    for name, wait_fn in candidates:
        if _is_callable_loaded(wait_fn):
            return wait_fn, name
    return None, None


def _resolve_semaphore_counter(device: VulkanLogicalDevice) -> Callable[..., Any] | None:
    candidates = (
        getattr(device, "vkGetSemaphoreCounterValue", None),
        getattr(device, "vkGetSemaphoreCounterValueKHR", None),
        getattr(DeviceFunc, "vkGetSemaphoreCounterValue", None),
        getattr(DeviceFunc, "vkGetSemaphoreCounterValueKHR", None),
    )
    for counter_fn in candidates:
        if _is_callable_loaded(counter_fn):
            return counter_fn
    return None


@dataclass(frozen=True, slots=True)
class FrameCompletionToken:
    """Borrowed Vulkan completion primitive for one submitted frame slot.

    The token is intentionally not an owning wrapper. Fences and timeline
    semaphores remain owned by ``_FrameSyncBase`` and are reused every frame.
    ``FrameResourceManager`` can poll or wait on this token through
    ``VulkanSurfaceContext`` to decide when CPU-side ranges/resources are safe
    to reuse.
    """

    frame_index: int
    fence: VkFence | None = None
    timeline_value: int = 0


class _FrameSyncBase:
    """Owns Vulkan frame submission and per-frame GPU synchronization.

    This class is Vulkan-specific. It owns the swapchain acquire/present
    synchronization and reusable command buffers for each frame in flight. It
    does not allocate descriptors or track backend-neutral resource lifetimes
    such as ring-buffer ranges; descriptor ownership stays with
    ``DescriptorManager`` and resource reuse stays with ``FrameResourceManager``.

    The bridge between the two layers is ``get_current_frame_completion``:
    before a frame is submitted, the surface context asks this object for a
    borrowed completion token. The shared frame resource manager stores that
    token and later calls back into Vulkan to poll/wait before reusing CPU-side
    resources for the same frame slot.
    """

    command_buffers: list[dict[int, CommandBuffer]]
    submit_command_buffer_ids: list[list[int]]

    def __init__(self, device: VulkanLogicalDevice,
                 swapchain: VulkanSwapchain | VulkanOffscreenSwapchain,
                 command_pool: CommandPool,
                 frames_in_flight: int,
                 stats: bool = True):
        self.current_frame = 0
        self.frames_in_flight = frames_in_flight
        self.stats = stats
        self.device = device
        self.swapchain = swapchain
        self.offscreen = not device.supports_presentation()
        self.command_pool = command_pool
        self.command_buffers = [{} for _ in range(frames_in_flight)]
        self.submit_command_buffer_ids = [[] for _ in range(frames_in_flight)]
        self.image_index = [0 for _ in range(frames_in_flight)]
        self.image_fences: list[VkFence | None] = []
        self.image_sync_values: list[int] = []
        self.counter = 0
        self._wait_stages = c_array_list([VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT], c_uint32)
        self.timeline_enabled = bool(getattr(self, "timeline_enabled", False))
        self.timeline_semaphore: VkSemaphore | None = None
        self._has_acquired_image = False
        self.last_acquired_image_index: int | None = None
        self.last_presented_image_index: int | None = None

        self.vkAcquireNextImageKHR = (
            getattr(device, "vkAcquireNextImageKHR", DeviceFunc.vkAcquireNextImageKHR)
            if not self.offscreen else None
        )
        self.vkQueuePresentKHR = (
            getattr(device, "vkQueuePresentKHR", DeviceFunc.vkQueuePresentKHR)
            if not self.offscreen else None
        )

        semaphore_info = VkSemaphoreCreateInfo(
            sType=VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
            flags=0,
        )

        self.command_pool.create()

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

        self.fences: list[VkFence] = []
        self._initialize_sync_primitives(fence_info)

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

    def set_frame_index(self, frame_index: int) -> int:
        self.current_frame = int(frame_index) % self.frames_in_flight
        return self.current_frame

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

    def queue_command_buffer_for_submit(self, idx: int) -> None:
        submit_ids = self.submit_command_buffer_ids[self.current_frame]
        try:
            submit_ids.remove(idx)
        except ValueError:
            pass
        submit_ids.append(idx)

    def get_frame_resources(self):
        return

    def get_current_frame_completion(self) -> FrameCompletionToken | None:
        """Return a borrowed completion token for the current frame submission."""
        if self.timeline_enabled:
            return FrameCompletionToken(self.current_frame, timeline_value=self.timeline_value + 1)
        if self.fences:
            return FrameCompletionToken(self.current_frame, fence=self.fences[self.current_frame])
        return None

    def is_frame_complete(self, token: FrameCompletionToken | None) -> bool:
        """Return whether a borrowed frame completion token has signaled."""
        if token is None:
            return True

        if token.timeline_value:
            counter_fn = _resolve_semaphore_counter(self.device)
            if counter_fn is None or self.timeline_semaphore is None:
                return False
            current_value = c_uint64(0)
            try:
                result = counter_fn(self.device.vk_device, self.timeline_semaphore, byref(current_value))
            except VulkanNotReadyException:
                return False
            return result == VK_SUCCESS and int(current_value.value) >= int(token.timeline_value)

        if token.fence:
            try:
                status = self.device.vkGetFenceStatus(self.device.vk_device, token.fence)
            except VulkanNotReadyException:
                return False
            return status == VK_SUCCESS

        return True

    def wait_for_frame_completion(self, token: FrameCompletionToken | None) -> None:
        """Block until a borrowed frame completion token has signaled."""
        if token is None:
            return

        if token.timeline_value:
            self._wait_timeline(token.timeline_value)
            return

        if token.fence:
            fence_array = c_array_list([token.fence], VkFence)
            self.vkWaitForFences(self.device.vk_device, 1, fence_array, VK_TRUE, UINT64_MAX)

    def release_frame_completion(self, _token: FrameCompletionToken | None) -> None:
        """Release a borrowed completion token.

        The underlying Vulkan handles are reusable frame-sync handles owned by
        this object, so there is no per-token destruction.
        """
        return

    def _initialize_sync_primitives(self, fence_info: VkFenceCreateInfo) -> None:
        raise NotImplementedError

    def _wait_for_current_frame(self, vk_device) -> None:
        raise NotImplementedError

    def _on_prepare_offscreen(self, vk_device) -> None:
        raise NotImplementedError

    def _on_image_acquired(self, vk_device, image_idx: int) -> None:
        raise NotImplementedError

    def _build_submit_sync(self, signal_list: list[VkSemaphore]) -> tuple[Any, VkFence, int | None]:
        raise NotImplementedError

    def _on_submit_complete(self, image_idx: int, timeline_signal_value: int | None) -> None:
        raise NotImplementedError

    def _destroy_sync_primitives(self) -> None:
        raise NotImplementedError

    def wait_for_last_submission(self) -> None:
        """Wait for the most recent submission to complete, if applicable."""
        return

    def _wait_timeline(self, _value: int) -> None:
        return

    def frame_begin(self, frame_index: int | None = None) -> bool:
        if frame_index is not None:
            self.set_frame_index(frame_index)
        self._has_acquired_image = False
        self.submit_command_buffer_ids[self.current_frame].clear()
        vk_device = self.device.vk_device
        self._wait_for_current_frame(vk_device)

        if self.offscreen:
            self._on_prepare_offscreen(vk_device)
            image_idx = self.current_frame % max(1, len(self.swapchain.swapchain_images))
            self.image_index[self.current_frame] = image_idx
            self.last_acquired_image_index = image_idx
            self._has_acquired_image = True
            return True

        img_index = c_uint32()
        try:
            self.vkAcquireNextImageKHR(
                vk_device,
                self.swapchain.swapchain,
                UINT64_MAX,
                self.image_available_semaphores[self.current_frame],
                0,
                byref(img_index),
            )
        except VulkanOutOfDateKHRException:
            return False

        self.image_index[self.current_frame] = img_index.value
        image_idx = img_index.value
        self.last_acquired_image_index = image_idx

        self._on_image_acquired(vk_device, image_idx)
        self._has_acquired_image = True
        return True

    def current_image_index(self) -> int:
        return self.image_index[self.current_frame]

    def present(self):
        if not self._has_acquired_image:
            return False

        image_idx = self.image_index[self.current_frame]
        wait_semaphores = None if self.offscreen else c_array_list([self.image_available_semaphores[self.current_frame]], VkSemaphore)

        submit_ids = self.submit_command_buffer_ids[self.current_frame]
        frame_commands = self.command_buffers[self.current_frame]
        command_wrappers = [frame_commands[cmd_id] for cmd_id in submit_ids if cmd_id in frame_commands]
        if not command_wrappers and not self.offscreen:
            command_wrapper = self._prepare_present_layout_transition(image_idx)
            if command_wrapper is not None:
                command_wrappers.append(command_wrapper)
        command_count = len(command_wrappers)
        cmd_buffers = c_array_list([cb.command_buffer for cb in command_wrappers], VkCommandBuffer) if command_count else None

        signal_list = [self.render_finished_semaphores[image_idx]]
        submit_pnext, queue_fence, timeline_signal_value = self._build_submit_sync(signal_list)

        signal_semaphores = c_array_list(signal_list, VkSemaphore)
        submit_create = VkSubmitInfo(
            sType=VK_STRUCTURE_TYPE_SUBMIT_INFO,
            pNext=submit_pnext,
            waitSemaphoreCount=0 if self.offscreen else 1,
            pWaitSemaphores=wait_semaphores,
            pWaitDstStageMask=None if self.offscreen else self._wait_stages,
            commandBufferCount=command_count,
            pCommandBuffers=cmd_buffers,
            signalSemaphoreCount=len(signal_list),
            pSignalSemaphores=signal_semaphores)

        submit_array = c_array_list([submit_create], VkSubmitInfo)

        self.vkQueueSubmit(self.device.graphics_queue.vk_queue,
                  1, submit_array, queue_fence)

        self._on_submit_complete(image_idx, timeline_signal_value)
        if not self.offscreen:
            image_layouts = getattr(self.swapchain, "image_layouts", None)
            if image_layouts is not None and image_idx < len(image_layouts):
                image_layouts[image_idx] = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR

        if self.offscreen:
            self.last_presented_image_index = image_idx
            self._has_acquired_image = False
            return True

        swapchains = c_array_list([self.swapchain.swapchain], VkSwapchainKHR)
        image_indices = c_array_list([image_idx], c_uint32)

        present_create = VkPresentInfoKHR(
            sType=VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
            waitSemaphoreCount=1,
            pWaitSemaphores=signal_semaphores,
            swapchainCount=1,
            pSwapchains=swapchains,
            pImageIndices=image_indices)

        try:
            self.vkQueuePresentKHR(self.device.present_queue.vk_queue, byref(present_create))
        except VulkanOutOfDateKHRException:
            self._has_acquired_image = False
            return False

        self.last_presented_image_index = image_idx

        self._has_acquired_image = False
        return True

    def _prepare_present_layout_transition(self, image_idx: int) -> CommandBuffer | None:
        """Record a present-layout transition for an acquired image when no render pass ran."""
        image_layouts = self.swapchain.image_layouts
        old_layout = VK_IMAGE_LAYOUT_UNDEFINED
        if image_layouts is not None and image_idx < len(image_layouts):
            old_layout = image_layouts[image_idx]

        if old_layout == VK_IMAGE_LAYOUT_PRESENT_SRC_KHR:
            return None

        command_wrapper = self.command_buffers[self.current_frame].get(0)
        if command_wrapper is None:
            self.get_command_buffer()
            command_wrapper = self.command_buffers[self.current_frame][self.counter - 1]

        command_wrapper.reset()
        command_wrapper.begin()
        barrier = VkImageMemoryBarrier(
            sType=VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            srcAccessMask=VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT if old_layout != VK_IMAGE_LAYOUT_UNDEFINED else 0,
            dstAccessMask=0,
            oldLayout=old_layout,
            newLayout=VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
            srcQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED,
            dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED,
            image=self.swapchain.swapchain_images[image_idx],
            subresourceRange=VkImageSubresourceRange(
                aspectMask=VK_IMAGE_ASPECT_COLOR_BIT,
                baseMipLevel=0,
                levelCount=1,
                baseArrayLayer=0,
                layerCount=1,
            ),
        )
        barrier_array = c_array_list([barrier], VkImageMemoryBarrier)
        DeviceFunc.vkCmdPipelineBarrier(
            command_wrapper.command_buffer,
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
            0,
            0,
            None,
            0,
            None,
            1,
            barrier_array,
        )
        command_wrapper.end()
        return command_wrapper

    def delete(self) -> None:
        device = self.device
        vk_device = device.vk_device

        if vk_device:
            for semaphore in self.image_available_semaphores:
                device.vkDestroySemaphore(vk_device, semaphore, None)
        self.image_available_semaphores.clear()

        if vk_device:
            for semaphore in self.render_finished_semaphores:
                device.vkDestroySemaphore(vk_device, semaphore, None)
        self.render_finished_semaphores.clear()

        if vk_device:
            self._destroy_sync_primitives()
        else:
            self.fences.clear()
            self.timeline_semaphore = None

        if self.command_pool:
            self.command_pool.delete()
            self.command_pool = None
            self.command_buffers.clear()
            self.submit_command_buffer_ids.clear()
        self.device = None


class FrameSyncBinary(_FrameSyncBase):
    timeline_enabled = False

    def _initialize_sync_primitives(self, fence_info: VkFenceCreateInfo) -> None:
        self.fences = [create_fence(self.device, fence_info) for _ in range(self.frames_in_flight)]
        self.timeline_semaphore = None

    def _wait_for_current_frame(self, vk_device) -> None:
        fence_array = c_array_list([self.fences[self.current_frame]], VkFence)
        self.vkWaitForFences(vk_device, 1, fence_array, VK_TRUE, UINT64_MAX)

    def _on_prepare_offscreen(self, vk_device) -> None:
        fence_array = c_array_list([self.fences[self.current_frame]], VkFence)
        self.vkResetFences(vk_device, 1, fence_array)

    def _on_image_acquired(self, vk_device, image_idx: int) -> None:
        # Reset only after a successful acquire; otherwise no submit happens and
        # this fence would remain unsignaled indefinitely.
        fence_array = c_array_list([self.fences[self.current_frame]], VkFence)
        self.vkResetFences(vk_device, 1, fence_array)

        # If this image is already in flight from another frame, wait for that frame to complete.
        image_fence = self.image_fences[image_idx]
        if image_fence is not None and image_fence != self.fences[self.current_frame]:
            image_fence_array = c_array_list([image_fence], VkFence)
            self.vkWaitForFences(vk_device, 1, image_fence_array, VK_TRUE, UINT64_MAX)

        self.image_fences[image_idx] = self.fences[self.current_frame]

    def _build_submit_sync(self, signal_list: list[VkSemaphore]) -> tuple[Any, VkFence, int | None]:
        return None, self.fences[self.current_frame], None

    def _on_submit_complete(self, image_idx: int, timeline_signal_value: int | None) -> None:  # noqa: ARG002
        return

    def _destroy_sync_primitives(self) -> None:
        for fence in self.fences:
            self.device.vkDestroyFence(self.device.vk_device, fence, None)
        self.fences.clear()

        if self.timeline_semaphore:
            self.device.vkDestroySemaphore(self.device.vk_device, self.timeline_semaphore, None)
            self.timeline_semaphore = None


class FrameSyncTimeline(_FrameSyncBase):
    timeline_enabled = True

    def __init__(self, device: VulkanLogicalDevice,
                 swapchain: VulkanSwapchain | VulkanOffscreenSwapchain,
                 command_pool: CommandPool,
                 frames_in_flight: int,
                 stats: bool = True,
                 wait_semaphores: Callable[..., Any] | None = None,
                 wait_semaphore_fn_name: str | None = None):
        self.vkWaitSemaphores = wait_semaphores
        self._wait_semaphore_fn_name = wait_semaphore_fn_name
        self.timeline_value = 0
        self.frame_sync_values = [0 for _ in range(frames_in_flight)]
        super().__init__(device, swapchain, command_pool, frames_in_flight, stats=stats)

    def _wait_timeline(self, value: int) -> None:
        if value <= 0 or self.timeline_semaphore is None or self.vkWaitSemaphores is None:
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

    def _initialize_sync_primitives(self, fence_info: VkFenceCreateInfo) -> None:  # noqa: ARG002
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
        self.timeline_semaphore = create_semaphore(self.device, timeline_semaphore_info)
        self.fences = []

    def _wait_for_current_frame(self, vk_device) -> None:  # noqa: ARG002
        frame_value = self.frame_sync_values[self.current_frame]
        if frame_value:
            self._wait_timeline(frame_value)

    def _on_prepare_offscreen(self, vk_device) -> None:  # noqa: ARG002
        return

    def _on_image_acquired(self, vk_device, image_idx: int) -> None:  # noqa: ARG002
        image_value = self.image_sync_values[image_idx]
        if image_value and image_value != self.frame_sync_values[self.current_frame]:
            self._wait_timeline(image_value)

    def _build_submit_sync(self, signal_list: list[VkSemaphore]) -> tuple[Any, VkFence, int | None]:
        assert self.timeline_semaphore is not None
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
        return submit_pnext, VK_NULL_HANDLE, timeline_signal_value

    def _on_submit_complete(self, image_idx: int, timeline_signal_value: int | None) -> None:
        if timeline_signal_value is None:
            return
        self.timeline_value = timeline_signal_value
        self.frame_sync_values[self.current_frame] = timeline_signal_value
        self.image_sync_values[image_idx] = timeline_signal_value

    def _destroy_sync_primitives(self) -> None:
        if self.timeline_semaphore:
            self.device.vkDestroySemaphore(self.device.vk_device, self.timeline_semaphore, None)
            self.timeline_semaphore = None
        self.fences.clear()

    def wait_for_last_submission(self) -> None:
        self._wait_timeline(self.timeline_value)



FrameSync = _FrameSyncBase


def create_frame_sync(device: VulkanLogicalDevice,
                      swapchain: VulkanSwapchain | VulkanOffscreenSwapchain,
                      command_pool: CommandPool,
                      frames_in_flight: int,
                      stats: bool = True) -> _FrameSyncBase:
    offscreen = not device.supports_presentation()
    wait_semaphores, wait_fn_name = _resolve_wait_semaphores(device)
    use_timeline = (
        device.timeline_semaphore_enabled
        and wait_semaphores is not None
        and not offscreen
    )

    if _debug_api:
        if offscreen:
            print("(Vulkan) Frame sync running in offscreen mode; using fences.")
        elif use_timeline:
            print(f"(Vulkan) Frame sync using timeline semaphore path (wait_fn={wait_fn_name}).")
        elif device.timeline_semaphore_enabled:
            print("(Vulkan) Timeline semaphore feature detected, but no wait function is available; using fences.")
        else:
            print("(Vulkan) Frame sync using fence path.")

    if use_timeline:
        return FrameSyncTimeline(
            device,
            swapchain,
            command_pool,
            frames_in_flight,
            stats=stats,
            wait_semaphores=wait_semaphores,
            wait_semaphore_fn_name=wait_fn_name,
        )

    return FrameSyncBinary(
        device,
        swapchain,
        command_pool,
        frames_in_flight,
        stats=stats,
    )
