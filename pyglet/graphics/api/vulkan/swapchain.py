from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING
from ctypes import pointer, byref

from pyglet.enums import ComponentFormat
from pyglet.graphics.api.vulkan import c_array_list
from pyglet.math import clamp
from pyglet.libs.shared.vulkan_lib import DeviceFunc

from pyglet.libs.shared.vulkan_lib.func_helpers import GetPhysicalDeviceSurfaceCapabilitiesKHR, GetPhysicalDeviceSurfaceFormatsKHR, \
    GetPhysicalDeviceSurfacePresentModesKHR, GetSwapchainImagesKHR
from pyglet.libs.shared.vulkan_lib.vulkan_core import VK_FORMAT_B8G8R8A8_SRGB, \
    VK_FORMAT_R8G8B8A8_SRGB, \
    VK_COLOR_SPACE_SRGB_NONLINEAR_KHR, VK_PRESENT_MODE_FIFO_KHR, \
    VK_PRESENT_MODE_MAILBOX_KHR, VK_PRESENT_MODE_IMMEDIATE_KHR, VkExtent2D, VkImageViewCreateInfo, \
    VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, VK_IMAGE_VIEW_TYPE_2D, VkComponentMapping, VkImageSubresourceRange, \
    VK_IMAGE_ASPECT_COLOR_BIT, VkSwapchainCreateInfoKHR, VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR, \
    VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, VK_SHARING_MODE_EXCLUSIVE, VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR, VK_TRUE, \
    VK_IMAGE_USAGE_TRANSFER_SRC_BIT, \
    VkSwapchainKHR, VkFramebufferCreateInfo, VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO, VK_SUCCESS, VkImageView, \
    VkFramebuffer

if TYPE_CHECKING:
    from pyglet.window import Window
    from pyglet.graphics.api.vulkan.renderpass import RenderPass
    from pyglet.graphics.api.vulkan.instance import VulkanGlobal, VulkanSurface
    from pyglet.graphics.api.vulkan.devices import VulkanLogicalDevice, VulkanPhysicalGraphicsDevice, VulkanDevices



class VulkanSwapchain:
    image_views: list[VkImageView]

    def __init__(self, instance: VulkanGlobal,
                 logical: VulkanLogicalDevice,
                 physical: VulkanPhysicalGraphicsDevice,
                 surface: VulkanSurface,
                 window: Window):
        self.swapchain = None  # Initialized later
        self.swapchain_images = None  # Initialized later
        self.logical = logical
        self.physical = physical
        self.surface = surface
        self.window = window

        self.image_views = []  # Initialized as an empty list
        self.framebuffers = [] # Created after renderpass.

        self.surface_format = None  # Set during swapchain creation
        self.present_mode = None  # Set during swapchain creation
        self.extent = None  # Set during swapchain creation
        self.instance = instance

        self.width = window.width
        self.height = window.height

        self.create_swapchain()
        self.create_image_views()

    def recreate(self, width: int, height: int, renderpass: RenderPass):
        if width != self.width or height != self.height:
            self.width = width
            self.height = height

            self.delete()

            self.create_swapchain()
            self.create_image_views()
            self.create_framebuffers(renderpass)

    def create_framebuffers(self, renderpass: RenderPass):
        """Create framebuffers for each image view."""
        assert self.extent is not None
        for image in self.image_views:
            imageviews = [image]
            imageview_array = c_array_list(imageviews, VkImageView)
            framebuffer_create = VkFramebufferCreateInfo(
                sType=VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
                flags=0,
                renderPass=renderpass.vk_renderpass,
                attachmentCount=len(imageviews),
                pAttachments=imageview_array,
                width=self.extent.width,
                height=self.extent.height,
                layers=1)

            framebuffer = VkFramebuffer()
            DeviceFunc.vkCreateFramebuffer(self.logical.vk_device, byref(framebuffer_create), None, byref(framebuffer))
            self.framebuffers.append(framebuffer)

        print("FRAME BUFFERS", self.framebuffers)

    def create_swapchain(self):
        surface_capabilities = GetPhysicalDeviceSurfaceCapabilitiesKHR(
            self.physical.vk_device,
            self.surface.vk_surface,
        )
        surface_formats = GetPhysicalDeviceSurfaceFormatsKHR(
            self.physical.vk_device,
            self.surface.vk_surface,
        )
        surface_present_modes = GetPhysicalDeviceSurfacePresentModesKHR(
            self.physical.vk_device,
            self.surface.vk_surface,
        )

        if len(surface_formats) == 0 and len(surface_present_modes) == 0:
            print("Swap chain unavailable.")

        self.surface_format = self.choose_surface_format(surface_formats)
        self.present_mode = self.choose_present_mode(surface_present_modes)
        self.extent = self.choose_swap_extent(surface_capabilities)

        image_count = surface_capabilities.minImageCount + 1
        if 0 < surface_capabilities.maxImageCount < image_count:
            image_count = surface_capabilities.maxImageCount

        swapchain_create = VkSwapchainCreateInfoKHR(
            sType=VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
            surface=self.surface.vk_surface,
            minImageCount=image_count,
            imageFormat=self.surface_format.format,
            imageColorSpace=self.surface_format.colorSpace,
            imageExtent=self.extent,
            imageArrayLayers=1,  # For stereoscopic 3D.
            imageUsage=VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
            imageSharingMode=VK_SHARING_MODE_EXCLUSIVE,
            preTransform=surface_capabilities.currentTransform,
            compositeAlpha=VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
            presentMode=self.present_mode,
            clipped=VK_TRUE,
            #oldSwapchain=None,
        )

        self.swapchain = VkSwapchainKHR()

        if result := DeviceFunc.vkCreateSwapchainKHR(
                self.logical.vk_device, pointer(swapchain_create), None, byref(self.swapchain)) != VK_SUCCESS:
            raise Exception(result)

        self.swapchain_images = GetSwapchainImagesKHR(self.logical.vk_device, self.swapchain)

    def choose_surface_format(self, formats)-> int:
        for f in formats:
            if f.format == VK_FORMAT_B8G8R8A8_SRGB and f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR:
                return f
        return formats[0]

    def choose_present_mode(self, present_modes) -> int:
        # VSync Enabled. Blocks the application when the queue is full.
        # Vsync. (Triple buffering?) Instead of blocking, it replaces the oldest image with the most recent image.
        if VK_PRESENT_MODE_MAILBOX_KHR in present_modes:
            return VK_PRESENT_MODE_MAILBOX_KHR

        if VK_PRESENT_MODE_FIFO_KHR in present_modes:
            return VK_PRESENT_MODE_FIFO_KHR

        # No vsync
        return VK_PRESENT_MODE_IMMEDIATE_KHR

    def choose_swap_extent(self, capabilities) -> VkExtent2D:
        if capabilities.currentExtent.width != 0xFFFFFFFF:  # UINT32_MAX
            return VkExtent2D(capabilities.currentExtent.width, capabilities.currentExtent.height)

        return VkExtent2D(
            width=clamp(self.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width),
            height=clamp(self.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height),
        )

    def create_image_views(self):
        for image in self.swapchain_images:
            view_create = VkImageViewCreateInfo(
                sType=VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                image=image,
                viewType=VK_IMAGE_VIEW_TYPE_2D,
                format=self.surface_format.format,
                components=VkComponentMapping(),
                subresourceRange=VkImageSubresourceRange(
                    aspectMask=VK_IMAGE_ASPECT_COLOR_BIT,
                    baseMipLevel=0,
                    levelCount=1,
                    baseArrayLayer=0,
                    layerCount=1,
                ),
            )

            image_view = VkImageView()
            self.logical.vkCreateImageView(self.logical.vk_device, byref(view_create), None, byref(image_view))
            self.image_views.append(image_view)

    def __del__(self):
        self.delete()

    def delete(self):
        """Destroy resources involved in the swapchain."""
        for framebuffer in self.framebuffers:
            DeviceFunc.vkDestroyFramebuffer(self.logical.vk_device, framebuffer, None)
        self.framebuffers.clear()

        for view in self.image_views:
            DeviceFunc.vkDestroyImageView(self.logical.vk_device, view, None)
        self.image_views.clear()

        if self.swapchain:
            DeviceFunc.vkDestroySwapchainKHR(self.logical.vk_device, self.swapchain, None)
            self.swapchain = None


class VulkanOffscreenSwapchain:
    """Swapchain-like offscreen render target for headless Vulkan rendering.

    This mimics the subset of swapchain attributes currently expected by the
    Vulkan backend (extent, images, views, framebuffers, format), but renders
    into a single VkImage instead of a presentable surface.
    """

    image_views: list[VkImageView]

    def __init__(self, logical: VulkanLogicalDevice, devices: VulkanDevices, window: Window):
        self.swapchain = None
        self.logical = logical
        self.devices = devices
        self.window = window
        self.width = int(window.width)
        self.height = int(window.height)
        self.extent = VkExtent2D(width=self.width, height=self.height)
        self.surface_format = SimpleNamespace(
            format=VK_FORMAT_R8G8B8A8_SRGB,
            colorSpace=VK_COLOR_SPACE_SRGB_NONLINEAR_KHR,
        )
        self.present_mode = VK_PRESENT_MODE_FIFO_KHR

        self.offscreen_image = None
        self.offscreen_image_view = None
        self.swapchain_images = []
        self.image_views = []
        self.framebuffers = []

        self._create_image_resources()

    def _create_image_resources(self):
        from pyglet.graphics.api.vulkan.texture import VulkanImage, VulkanImageView  # noqa: PLC0415

        self.offscreen_image = VulkanImage(
            width=self.width,
            height=self.height,
            internal_format=ComponentFormat.RGBA,
            usage=VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
        )
        self.offscreen_image.create(self.devices)

        self.offscreen_image_view = VulkanImageView(self.devices, self.offscreen_image)
        self.swapchain_images = [self.offscreen_image.vk_image]
        self.image_views = [self.offscreen_image_view.vk_imageview]

    def recreate(self, width: int, height: int, renderpass: RenderPass):
        width = int(width)
        height = int(height)
        if width == self.width and height == self.height:
            return

        self.width = width
        self.height = height
        self.extent = VkExtent2D(width=self.width, height=self.height)

        self.delete()
        self._create_image_resources()
        self.create_framebuffers(renderpass)

    def create_framebuffers(self, renderpass: RenderPass):
        assert self.extent is not None
        for image in self.image_views:
            imageviews = [image]
            imageview_array = c_array_list(imageviews, VkImageView)
            framebuffer_create = VkFramebufferCreateInfo(
                sType=VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
                flags=0,
                renderPass=renderpass.vk_renderpass,
                attachmentCount=len(imageviews),
                pAttachments=imageview_array,
                width=self.extent.width,
                height=self.extent.height,
                layers=1,
            )

            framebuffer = VkFramebuffer()
            DeviceFunc.vkCreateFramebuffer(self.logical.vk_device, byref(framebuffer_create), None, byref(framebuffer))
            self.framebuffers.append(framebuffer)

    def delete(self):
        for framebuffer in self.framebuffers:
            DeviceFunc.vkDestroyFramebuffer(self.logical.vk_device, framebuffer, None)
        self.framebuffers.clear()

        if self.offscreen_image_view is not None:
            self.offscreen_image_view.delete()
            self.offscreen_image_view = None
        self.image_views.clear()

        if self.offscreen_image is not None:
            self.offscreen_image.delete()
            self.offscreen_image = None
        self.swapchain_images.clear()

    # def flip(self):
    #     try:
    #         image_index = self.instance.func.vkAcquireNextImageKHR(
    #             self.logical.vk_device,
    #             self.swapchain,
    #             UINT64_MAX,
    #             semaphore_image_available,
    #             None,
    #         )
    #     except VkNotReady:
    #         print('not ready')
    #         return
    #
    #     submit_create.pCommandBuffers[0] = command_buffers[image_index]
    #     vkQueueSubmit(config.vk_queue.graphics_queue, 1, submit_list, None)
    #
    #     present_create.pImageIndices[0] = image_index
    #     vkQueuePresentKHR(config.vk_queue.present_queue, present_create)
