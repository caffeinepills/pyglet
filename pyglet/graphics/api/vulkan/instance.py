from __future__ import annotations

import ctypes
from typing import TYPE_CHECKING, Any

import pyglet
from pyglet.config import VulkanUserConfig
from pyglet.graphics.api.base import (
    BackendGlobalObject,
    SurfaceContext,
    UBOMatrixTransformations,
    NullContext,
)
from pyglet.graphics.api.vulkan import DeviceFunc
from pyglet.graphics.api.vulkan.commands import CommandBuffer, CommandPool
from pyglet.graphics.api.vulkan.descriptor import DescriptorManager
from pyglet.graphics.api.vulkan.devices import VulkanDevices
from pyglet.graphics.api.vulkan.vk_info import VulkanInfo
from pyglet.graphics.api.vulkan.pipeline import GraphicsPipelineManager
from pyglet.graphics.api.vulkan.renderpass import RenderPass, RenderPassManager
from pyglet.graphics.shader import Shader, ShaderProgram
from pyglet.graphics.api.vulkan.swapchain import VulkanSwapchain
from pyglet.graphics.api.vulkan.sync import DeferredResourceRemoval, FrameSync
from pyglet.libs.shared.vulkan_lib import InstanceFunc, c_array_list, set_instance_functions, vulkan_core
from pyglet.libs.shared.vulkan_lib.func_helpers import CreateInstance, EnumerateInstanceLayerProperties
from pyglet.libs.shared.vulkan_lib.vulkan_core import (
    VK_API_VERSION_1_0,
    VK_DEBUG_REPORT_ERROR_BIT_EXT,
    VK_DEBUG_REPORT_WARNING_BIT_EXT,
    VK_MAKE_API_VERSION,
    VK_MAKE_VERSION,
    VK_STRUCTURE_TYPE_APPLICATION_INFO,
    VK_STRUCTURE_TYPE_DEBUG_REPORT_CALLBACK_CREATE_INFO_EXT,
    VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
    VK_STRUCTURE_TYPE_METAL_SURFACE_CREATE_INFO_EXT,
    VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
    VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR,
    VK_STRUCTURE_TYPE_XLIB_SURFACE_CREATE_INFO_KHR,
    VK_SUBPASS_CONTENTS_INLINE,
    VK_SUCCESS,
    PFN_vkDebugReportCallbackEXT,
    VkApplicationInfo,
    VkClearColorValue,
    VkClearValue,
    VkDebugReportCallbackCreateInfoEXT,
    VkDebugReportCallbackEXT,
    VkInstanceCreateInfo,
    VkOffset2D,
    VkRect2D,
    VkRenderPassBeginInfo,
    VkSurfaceKHR,
)
from pyglet.graphics.shader import ShaderType, UniformBlockDesc
from pyglet.math import Mat4

if TYPE_CHECKING:
    from ctypes import *

    from pyglet.config.vulkan import VulkanSurfaceConfig
    from pyglet.window import Window
    from pyglet.window.cocoa import CocoaWindow
    from pyglet.window.xlib import XlibWindow

_debug_api = pyglet.options.debug_api


class VulkanMatrices(UBOMatrixTransformations):
    # Create a default ShaderProgram, so the Window instance can
    # update the `WindowBlock` UBO shared by all default shaders.

    def __init__(self, window: Window, backend: VulkanGlobal):

        self._default_program = pyglet.graphics.get_default_shader()
        if "WindowBlock" not in self._default_program.uniform_blocks:
            self._default_program.set_uniform_blocks(WindowBlock)
        self.ubo = self._default_program.uniform_blocks["WindowBlock"].create_ubo()

        # Change to work with this:
        #self.ubo = self._default_program.uniform_blocks['WindowBlock'].create_ubo()

        self._viewport = (0, 0, *window.get_framebuffer_size())

        width, height = window.get_size()

        super().__init__(window, Mat4.orthogonal_projection(0, width, 0, height, -255, 255), Mat4(), Mat4())

        with self.ubo as ubo_data:
            ubo_data.projection = tuple(self._projection)
            ubo_data.view = tuple(self._view)

        # with self.ubo as window_block:
        #     window_block.view[:] = self._view
        #     window_block.projection[:] = self._projection
        #     #window_block.model[:] = self._model

    @property
    def projection(self) -> Mat4:
        return self._projection

    @projection.setter
    def projection(self, projection: Mat4):
        #with self.ubo as window_block:
        #    window_block.projection[:] = projection

        self._projection = projection

    @property
    def view(self) -> Mat4:
        return self._view

    @view.setter
    def view(self, view: Mat4):
        #with self.ubo as window_block:
        #    window_block.view[:] = view

        self._view = view

    @property
    def model(self) -> Mat4:
        return self._model

    @model.setter
    def model(self, model: Mat4):
        #with self.ubo as window_block:
        #    window_block.model[:] = model

        self._model = model


class VulkanSurface:
    def __init__(self, vk_surface: VkSurface | VkSurfaceKHR, global_vulkan: VulkanGlobal):
        self.vk_surface = vk_surface
        self.global_vulkan = global_vulkan

    def delete(self):
        if self.vk_surface:
            InstanceFunc.vkDestroySurfaceKHR(self.global_vulkan.instance.vk_instance, self.vk_surface, None)
            self.vk_surface = None

    @classmethod
    def get_surface(cls, global_vulkan: VulkanGlobal, window: Window):
        extensions = global_vulkan.instance.extensions
        print("EXTENSIONS", extensions)
        if b'VK_KHR_win32_surface' in extensions:
            return cls.create_win32_surface(global_vulkan, window)

        if b'VK_KHR_xlib_surface' in extensions:
            return cls.create_xlib_surface(global_vulkan, window)

        if b'VK_EXT_metal_surface' in extensions:
            return cls.create_metal_surface(global_vulkan, window)

        raise Exception("Surface is currently not supported.")

    @classmethod
    def create_metal_surface(cls, global_vulkan: VulkanGlobal, window: CocoaWindow):
        from pyglet.libs.shared.vulkan_lib.vulkan_metal import VkMetalSurfaceCreateInfoEXT

        vkCreateMetalSurfaceEXT = global_vulkan.instance.vkCreateMetalSurfaceEXT

        create_info = VkMetalSurfaceCreateInfoEXT(
            sType=VK_STRUCTURE_TYPE_METAL_SURFACE_CREATE_INFO_EXT,
            pLayer=window._metal_layer.ptr)

        vk_surface = VkSurfaceKHR()
        result = vkCreateMetalSurfaceEXT(global_vulkan.instance.vk_instance, ctypes.byref(create_info), None, ctypes.byref(vk_surface))
        if result != VK_SUCCESS:
            raise RuntimeError("Failed to create Metal surface.")

        return cls(vk_surface, global_vulkan)

    @classmethod
    def create_xlib_surface(cls, global_vulkan: VulkanGlobal, window: XlibWindow):
        from pyglet.libs.shared.vulkan_lib.vulkan_xlib import VkXlibSurfaceCreateInfoKHR

        vkCreateXlibSurfaceKHR = global_vulkan.instance.vkCreateXlibSurfaceKHR

        surface_create = VkXlibSurfaceCreateInfoKHR(
            sType=VK_STRUCTURE_TYPE_XLIB_SURFACE_CREATE_INFO_KHR,
            dpy=window._x_display,
            window=window._x_window,
            flags=0)

        vk_surface = VkSurfaceKHR()
        result = vkCreateXlibSurfaceKHR(global_vulkan.instance.vk_instance, surface_create, None, ctypes.byref(vk_surface))
        if result != VK_SUCCESS:
            raise Exception(result)

        return cls(vk_surface, global_vulkan)

    @classmethod
    def create_win32_surface(cls, global_vulkan: VulkanGlobal, window: Window):
        from pyglet.libs.shared.vulkan_lib.vulkan_win32 import VkWin32SurfaceCreateInfoKHR

        vkCreateWin32SurfaceKHR = global_vulkan.instance.vkCreateWin32SurfaceKHR
        surface_create = VkWin32SurfaceCreateInfoKHR(
            sType=VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR,
            hinstance=window._window_class.hInstance,
            hwnd=window._hwnd,
            flags=0)

        vk_surface = VkSurfaceKHR()
        result = vkCreateWin32SurfaceKHR(global_vulkan.instance.vk_instance, surface_create, None, ctypes.byref(vk_surface))
        if result != VK_SUCCESS:
            raise Exception(result)

        return cls(vk_surface, global_vulkan)


# Allow multiple frames to prevent waiting for a frame to finish.
# The more frames, the greater latency which could cause extra issues.
# If set to 1, you will need to call vkDeviceWaitIdle, which will force the frame to wait.
NUM_FRAMES_IN_FLIGHT = 2

class WindowBlock(UniformBlockDesc):
    stages = ("vertex",)
    set_num = 0
    bind_num = 0
    uniforms = (
        ("mat4", "projection"),
        ("mat4", "view"),
    )


class VulkanSurfaceContext(SurfaceContext):
    frame_sync: FrameSync
    core: VulkanGlobal
    swapchain: VulkanSwapchain | None  # A vulkan swapchain can actually be none for headless.
    def __init__(self, global_ctx: VulkanGlobal, window: Window, config: VulkanSurfaceConfig, devices: VulkanDevices) -> None:
        self.devices = devices
        self.instance = global_ctx.instance
        super().__init__(global_ctx, window, config)
        self._info = VulkanInfo()

        self.surface = None
        self.swapchain = None
        self.renderpass = None
        self.descriptor_pool = None
        self.pipeline = None
        self.command_buffers = None

        # Can technically have multiple logical devices and surfaces, but we'll just use 1.
        self.surface = VulkanSurface.get_surface(self.core, window)

        # Create the Logical Device once we have a surface to get a presentation queue.
        if self.devices.logical_device.vk_device is None:
            self.devices.logical_device.create(self.surface)
        self.logical_device = self.devices.logical_device
        self.core.command_pool.create()
        self._info.query(self)

        # TODO: Handle no swapchain for headless environments and swapchain support with VK_EXT_headless_surface
        assert self.logical_device.supports_presentation() is True, "Device does not support presentation."

        # Swapchain and image views.
        assert self.surface
        self.swapchain = VulkanSwapchain(self.core, self.logical_device, self.devices.physical_device,
                                         self.surface, self.window)

        # Renderpass
        self.renderpass = RenderPass(self.logical_device, self.swapchain.surface_format.format)

        # Framebuffers
        self.swapchain.create_framebuffers(self.renderpass)

        # Frame Syncing
        self.frame_sync = FrameSync(
            self.logical_device,
            self.swapchain,
            self.core.descriptor_mgr,
            CommandPool(self.logical_device),
            NUM_FRAMES_IN_FLIGHT,
        )

        self.default_cb_id = 0

        self.vkAcquireNextImageKHR = self.devices.logical_device.vkAcquireNextImageKHR
        self.vkQueuePresentKHR = self.devices.logical_device.vkQueuePresentKHR

    def set_clear_color(self, r: float, g: float, b: float, a: float) -> None:
        self.clear_color = (r, g, b, a)

    @property
    def info(self) -> VulkanInfo:
        return self._info

    def get_info(self) -> VulkanInfo:
        return self.info

    def resized(self, width, height):
        return
        DeviceFunc.vkDeviceWaitIdle(self.logical_device.vk_device)

        self.swapchain.recreate(width, height, self.renderpass)

    def recreate(self, width: int, height: int):
        """Recreate the swapchain resources when the window changes."""
        self.swapchain.recreate(width, height, self.renderpass)

    def attach(self, window: Window) -> None:
        assert self.window is window

    def clear(self) -> None:
        """Clear not necessary with Vulkan. Add exception or warning?"""

    def set_vsync(self, vsync):
        print("Setting vsync... to do")

    def set_current(self):
        self.core.set_current_context(self)
        # Process deferred resource removal.
        self.core.resource_removal.process()

    def _record_command_buffer(self, command_buffer: CommandBuffer, image_index: int):
        with command_buffer as vk_command_buffer:
            render_area = VkRect2D(offset=VkOffset2D(x=0, y=0),
                                   extent=self.swapchain.extent)
            color = VkClearColorValue(float32=(ctypes.c_float * 4)(*self.clear_color))
            clear_value = VkClearValue(color=color)

            cb_array = c_array_list([clear_value], VkClearValue)

            render_pass_begin_create = VkRenderPassBeginInfo(
                sType=VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
                renderPass=self.renderpass.vk_renderpass,
                framebuffer=self.swapchain.framebuffers[image_index],
                renderArea=render_area,
                clearValueCount=1,
                pClearValues=cb_array)

            self.logical_device.vkCmdBeginRenderPass(vk_command_buffer, render_pass_begin_create, VK_SUBPASS_CONTENTS_INLINE)
            self.logical_device.vkCmdEndRenderPass(vk_command_buffer)

    def before_draw(self):
        self.set_current()
        return self.frame_sync.before_draw()

    def flip(self):
        self.frame_sync.flip()

    def delete(self):
        if self.logical_device and self.logical_device.vk_device:
            self.logical_device.vkDeviceWaitIdle(self.logical_device.vk_device)
            self.core.resource_removal.process()

        matrices = getattr(self.window, "_matrices", None)
        if matrices and getattr(matrices, "ubo", None):
            matrices.ubo.delete()
            matrices.ubo = None

        if self.swapchain:
            self.swapchain.delete()
            self.swapchain = None

        if self.pipeline:
            self.pipeline.delete()
            self.pipeline = None

        if self.descriptor_pool:
            self.descriptor_pool.destroy()
            self.descriptor_pool = None

        if self.renderpass:
            self.renderpass.delete()
            self.renderpass = None

        if self.frame_sync:
            self.frame_sync.delete()
            self.frame_sync = None
        #
        # if self.command_pool:
        #     self.command_pool.delete()
        #     self.command_pool = None

        self.command_buffers = None

        if self.surface:
            self.surface.delete()
            self.surface = None

    def destroy(self):
        """Destroyed by window close."""
        if self.logical_device and self.logical_device.vk_device:
            self.logical_device.vkDeviceWaitIdle(self.logical_device.vk_device)
        self.delete()
        if self.window in self.core.windows:
            del self.core.windows[self.window]
        self.core.clear_current_context(self)


class VulkanInstanceFuncs:
    """Container for Vulkan instance functions for type hinting."""

    vkDestroyInstance: InstanceFunc.vkDestroyInstance
    vkDestroyInstance: InstanceFunc.vkDestroyInstance
    vkEnumeratePhysicalDevices: InstanceFunc.vkEnumeratePhysicalDevices
    vkGetPhysicalDeviceFeatures: InstanceFunc.vkGetPhysicalDeviceFeatures
    vkGetPhysicalDeviceFormatProperties: InstanceFunc.vkGetPhysicalDeviceFormatProperties
    vkGetPhysicalDeviceImageFormatProperties: InstanceFunc.vkGetPhysicalDeviceImageFormatProperties
    vkGetPhysicalDeviceProperties: InstanceFunc.vkGetPhysicalDeviceProperties
    vkGetPhysicalDeviceQueueFamilyProperties: InstanceFunc.vkGetPhysicalDeviceQueueFamilyProperties
    vkGetPhysicalDeviceMemoryProperties: InstanceFunc.vkGetPhysicalDeviceMemoryProperties
    vkGetInstanceProcAddr: InstanceFunc.vkGetInstanceProcAddr
    vkCreateDevice: InstanceFunc.vkCreateDevice
    vkEnumerateDeviceExtensionProperties: InstanceFunc.vkEnumerateDeviceExtensionProperties
    vkEnumerateDeviceLayerProperties: InstanceFunc.vkEnumerateDeviceLayerProperties
    vkGetPhysicalDeviceSparseImageFormatProperties: InstanceFunc.vkGetPhysicalDeviceSparseImageFormatProperties
    vkEnumeratePhysicalDeviceGroups: InstanceFunc.vkEnumeratePhysicalDeviceGroups
    vkGetPhysicalDeviceFeatures2: InstanceFunc.vkGetPhysicalDeviceFeatures2
    vkGetPhysicalDeviceProperties2: InstanceFunc.vkGetPhysicalDeviceProperties2
    vkGetPhysicalDeviceFormatProperties2: InstanceFunc.vkGetPhysicalDeviceFormatProperties2
    vkGetPhysicalDeviceImageFormatProperties2: InstanceFunc.vkGetPhysicalDeviceImageFormatProperties2
    vkGetPhysicalDeviceQueueFamilyProperties2: InstanceFunc.vkGetPhysicalDeviceQueueFamilyProperties2
    vkGetPhysicalDeviceMemoryProperties2: InstanceFunc.vkGetPhysicalDeviceMemoryProperties2
    vkGetPhysicalDeviceSparseImageFormatProperties2: InstanceFunc.vkGetPhysicalDeviceSparseImageFormatProperties2
    vkGetPhysicalDeviceExternalBufferProperties: InstanceFunc.vkGetPhysicalDeviceExternalBufferProperties
    vkGetPhysicalDeviceExternalFenceProperties: InstanceFunc.vkGetPhysicalDeviceExternalFenceProperties
    vkGetPhysicalDeviceExternalSemaphoreProperties: InstanceFunc.vkGetPhysicalDeviceExternalSemaphoreProperties
    vkGetPhysicalDeviceToolProperties: InstanceFunc.vkGetPhysicalDeviceToolProperties
    vkDestroySurfaceKHR: InstanceFunc.vkDestroySurfaceKHR
    vkGetPhysicalDeviceSurfaceSupportKHR: InstanceFunc.vkGetPhysicalDeviceSurfaceSupportKHR
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR: InstanceFunc.vkGetPhysicalDeviceSurfaceCapabilitiesKHR
    vkGetPhysicalDeviceSurfaceFormatsKHR: InstanceFunc.vkGetPhysicalDeviceSurfaceFormatsKHR
    vkGetPhysicalDeviceSurfacePresentModesKHR: InstanceFunc.vkGetPhysicalDeviceSurfacePresentModesKHR
    vkGetPhysicalDevicePresentRectanglesKHR: InstanceFunc.vkGetPhysicalDevicePresentRectanglesKHR
    vkGetPhysicalDeviceDisplayPropertiesKHR: InstanceFunc.vkGetPhysicalDeviceDisplayPropertiesKHR
    vkGetPhysicalDeviceDisplayPlanePropertiesKHR: InstanceFunc.vkGetPhysicalDeviceDisplayPlanePropertiesKHR
    vkGetDisplayPlaneSupportedDisplaysKHR: InstanceFunc.vkGetDisplayPlaneSupportedDisplaysKHR
    vkGetDisplayModePropertiesKHR: InstanceFunc.vkGetDisplayModePropertiesKHR
    vkCreateDisplayModeKHR: InstanceFunc.vkCreateDisplayModeKHR
    vkGetDisplayPlaneCapabilitiesKHR: InstanceFunc.vkGetDisplayPlaneCapabilitiesKHR
    vkCreateDisplayPlaneSurfaceKHR: InstanceFunc.vkCreateDisplayPlaneSurfaceKHR
    vkGetPhysicalDeviceVideoCapabilitiesKHR: InstanceFunc.vkGetPhysicalDeviceVideoCapabilitiesKHR
    vkGetPhysicalDeviceVideoFormatPropertiesKHR: InstanceFunc.vkGetPhysicalDeviceVideoFormatPropertiesKHR
    vkGetPhysicalDeviceFeatures2KHR: InstanceFunc.vkGetPhysicalDeviceFeatures2KHR
    vkGetPhysicalDeviceProperties2KHR: InstanceFunc.vkGetPhysicalDeviceProperties2KHR
    vkGetPhysicalDeviceFormatProperties2KHR: InstanceFunc.vkGetPhysicalDeviceFormatProperties2KHR
    vkGetPhysicalDeviceImageFormatProperties2KHR: InstanceFunc.vkGetPhysicalDeviceImageFormatProperties2KHR
    vkGetPhysicalDeviceQueueFamilyProperties2KHR: InstanceFunc.vkGetPhysicalDeviceQueueFamilyProperties2KHR
    vkGetPhysicalDeviceMemoryProperties2KHR: InstanceFunc.vkGetPhysicalDeviceMemoryProperties2KHR
    vkGetPhysicalDeviceSparseImageFormatProperties2KHR: InstanceFunc.vkGetPhysicalDeviceSparseImageFormatProperties2KHR
    vkEnumeratePhysicalDeviceGroupsKHR: InstanceFunc.vkEnumeratePhysicalDeviceGroupsKHR
    vkGetPhysicalDeviceExternalBufferPropertiesKHR: InstanceFunc.vkGetPhysicalDeviceExternalBufferPropertiesKHR
    vkGetPhysicalDeviceExternalSemaphorePropertiesKHR: InstanceFunc.vkGetPhysicalDeviceExternalSemaphorePropertiesKHR
    vkGetPhysicalDeviceExternalFencePropertiesKHR: InstanceFunc.vkGetPhysicalDeviceExternalFencePropertiesKHR
    vkEnumeratePhysicalDeviceQueueFamilyPerformanceQueryCountersKHR: InstanceFunc.vkEnumeratePhysicalDeviceQueueFamilyPerformanceQueryCountersKHR
    vkGetPhysicalDeviceQueueFamilyPerformanceQueryPassesKHR: InstanceFunc.vkGetPhysicalDeviceQueueFamilyPerformanceQueryPassesKHR
    vkGetPhysicalDeviceSurfaceCapabilities2KHR: InstanceFunc.vkGetPhysicalDeviceSurfaceCapabilities2KHR
    vkGetPhysicalDeviceSurfaceFormats2KHR: InstanceFunc.vkGetPhysicalDeviceSurfaceFormats2KHR
    vkGetPhysicalDeviceDisplayProperties2KHR: InstanceFunc.vkGetPhysicalDeviceDisplayProperties2KHR
    vkGetPhysicalDeviceDisplayPlaneProperties2KHR: InstanceFunc.vkGetPhysicalDeviceDisplayPlaneProperties2KHR
    vkGetDisplayModeProperties2KHR: InstanceFunc.vkGetDisplayModeProperties2KHR
    vkGetDisplayPlaneCapabilities2KHR: InstanceFunc.vkGetDisplayPlaneCapabilities2KHR
    vkGetPhysicalDeviceFragmentShadingRatesKHR: InstanceFunc.vkGetPhysicalDeviceFragmentShadingRatesKHR
    vkGetPhysicalDeviceVideoEncodeQualityLevelPropertiesKHR: InstanceFunc.vkGetPhysicalDeviceVideoEncodeQualityLevelPropertiesKHR
    vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR: InstanceFunc.vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR
    vkGetPhysicalDeviceCalibrateableTimeDomainsKHR: InstanceFunc.vkGetPhysicalDeviceCalibrateableTimeDomainsKHR
    vkCreateDebugReportCallbackEXT: InstanceFunc.vkCreateDebugReportCallbackEXT
    vkDestroyDebugReportCallbackEXT: InstanceFunc.vkDestroyDebugReportCallbackEXT
    vkDebugReportMessageEXT: InstanceFunc.vkDebugReportMessageEXT
    vkGetPhysicalDeviceExternalImageFormatPropertiesNV: InstanceFunc.vkGetPhysicalDeviceExternalImageFormatPropertiesNV
    vkReleaseDisplayEXT: InstanceFunc.vkReleaseDisplayEXT
    vkGetPhysicalDeviceSurfaceCapabilities2EXT: InstanceFunc.vkGetPhysicalDeviceSurfaceCapabilities2EXT
    vkCreateDebugUtilsMessengerEXT: InstanceFunc.vkCreateDebugUtilsMessengerEXT
    vkDestroyDebugUtilsMessengerEXT: InstanceFunc.vkDestroyDebugUtilsMessengerEXT
    vkSubmitDebugUtilsMessageEXT: InstanceFunc.vkSubmitDebugUtilsMessageEXT
    vkGetPhysicalDeviceMultisamplePropertiesEXT: InstanceFunc.vkGetPhysicalDeviceMultisamplePropertiesEXT
    vkGetPhysicalDeviceCalibrateableTimeDomainsEXT: InstanceFunc.vkGetPhysicalDeviceCalibrateableTimeDomainsEXT
    vkGetPhysicalDeviceToolPropertiesEXT: InstanceFunc.vkGetPhysicalDeviceToolPropertiesEXT
    vkGetPhysicalDeviceCooperativeMatrixPropertiesNV: InstanceFunc.vkGetPhysicalDeviceCooperativeMatrixPropertiesNV
    vkGetPhysicalDeviceSupportedFramebufferMixedSamplesCombinationsNV: InstanceFunc.vkGetPhysicalDeviceSupportedFramebufferMixedSamplesCombinationsNV
    vkCreateHeadlessSurfaceEXT: InstanceFunc.vkCreateHeadlessSurfaceEXT
    vkAcquireDrmDisplayEXT: InstanceFunc.vkAcquireDrmDisplayEXT
    vkGetDrmDisplayEXT: InstanceFunc.vkGetDrmDisplayEXT
    vkGetInstanceProcAddrLUNARG: InstanceFunc.vkGetInstanceProcAddrLUNARG
    vkGetPhysicalDeviceOpticalFlowImageFormatsNV: InstanceFunc.vkGetPhysicalDeviceOpticalFlowImageFormatsNV
    vkGetPhysicalDeviceCooperativeMatrixFlexibleDimensionsPropertiesNV: InstanceFunc.vkGetPhysicalDeviceCooperativeMatrixFlexibleDimensionsPropertiesNV

class VulkanInstance(VulkanInstanceFuncs):
    """Class managing the Vulkan instance."""

    def __init__(self, config: VulkanUserConfig | None = None) -> None:
        super().__init__()
        self._debug_callback_ptr = None
        self._debug_callback_handle = None
        self.modules = [vulkan_core]
        self.api_version =VK_MAKE_API_VERSION(0, config.major_version, config.minor_version, 0)

        if _debug_api:
            print(f"(Vulkan) Requested API version={config.major_version}.{config.minor_version}, resolved apiVersion={self.api_version}.")

        self.available_layers = EnumerateInstanceLayerProperties()
        self.extensions = [b'VK_KHR_surface', b'VK_EXT_debug_report']

        # Only include for debug API?
        self.layers = [b'VK_LAYER_KHRONOS_validation'] if self.is_validation_layer_available() else []

        if pyglet.compat_platform == "win32":
            from pyglet.libs.shared.vulkan_lib import vulkan_win32
            self.modules.append(vulkan_win32)
            self.extensions.append(b'VK_KHR_win32_surface')
        elif pyglet.compat_platform == "linux":
            from pyglet.libs.shared.vulkan_lib import vulkan_xlib
            self.modules.append(vulkan_xlib)
            self.extensions.append(b'VK_KHR_xlib_surface')
        elif pyglet.compat_platform == "wayland":
            self.extensions.append(b'VK_KHR_wayland_surface')
        elif pyglet.compat_platform == "darwin":
            from pyglet.libs.shared.vulkan_lib import vulkan_metal
            self.modules.append(vulkan_metal)
            self.extensions.append(b'VK_EXT_metal_surface')
        else:
            raise Exception("Platform not supported")

        app_info = VkApplicationInfo(
            sType=VK_STRUCTURE_TYPE_APPLICATION_INFO,
            pApplicationName=b"User Application",
            applicationVersion=VK_MAKE_VERSION(1, 0, 0),
            pEngineName=b"Pyglet",
            engineVersion=VK_MAKE_VERSION(1, 0, 0),
            apiVersion=self.api_version,
        )

        create_info = VkInstanceCreateInfo(
            sType=VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
            flags=0,
            pApplicationInfo=ctypes.pointer(app_info),
            enabledExtensionCount=len(self.extensions),
            ppEnabledExtensionNames=c_array_list(self.extensions, ctypes.c_char_p),
            enabledLayerCount=len(self.layers),
            ppEnabledLayerNames=c_array_list(self.layers, ctypes.c_char_p),
        )

        self.vk_instance = CreateInstance(create_info)

        funcs = set_instance_functions(self.vk_instance, self.modules)
        for func in funcs:
           name, fn = func
           setattr(self, name, fn)

        if _debug_api:
            if b'VK_LAYER_KHRONOS_validation' not in self.layers:
                print("Warning: The validation layer is not available.")
            else:
                print("Validation layer is available.")
            self._setup_debug_callback()

    def is_validation_layer_available(self) -> bool:
        return b'VK_LAYER_KHRONOS_validation' in [l.layerName for l in self.available_layers]

    def _setup_debug_callback(self) -> None:
        self._debug_callback_ptr = PFN_vkDebugReportCallbackEXT(self._debug_callback)
        debug_create = VkDebugReportCallbackCreateInfoEXT(
            sType=VK_STRUCTURE_TYPE_DEBUG_REPORT_CALLBACK_CREATE_INFO_EXT,
            flags=VK_DEBUG_REPORT_ERROR_BIT_EXT | VK_DEBUG_REPORT_WARNING_BIT_EXT,
            pfnCallback=self._debug_callback_ptr,
        )

        self._debug_callback_handle = VkDebugReportCallbackEXT()
        InstanceFunc.vkCreateDebugReportCallbackEXT(self.vk_instance, ctypes.pointer(debug_create), None, ctypes.byref(self._debug_callback_handle))

    def _debug_callback(self,
            flags: int,  # VkDebugReportFlagsEXT
            object_type: int,  # VkDebugReportObjectTypeEXT
            obj: int,  # Vulkan object (handle)
            location: int,  # Location in Vulkan API
            message_code: int,  # Message code
            layer_prefix: bytes | None,  # Layer name prefix
            message: bytes | None,  # Debug message
            user_data: Any | None,  # User data pointer (void*)
    ) -> int:  # Return value is VkBool32 (int)
        print(f"(Vulkan DEBUG) [{layer_prefix.decode()}]: {message.decode()}")
        return 0

    def delete(self) -> None:
        if self._debug_callback_handle:
            self.vkDestroyDebugReportCallbackEXT(self.vk_instance, self._debug_callback_handle, None)
            self._debug_callback_handle = None
        if self.vk_instance:
            self.vkDestroyInstance(self.vk_instance, None)
            self.vk_instance = None



class ObjectSpace:
    """A container to store shared objects that are to be removed."""

    def __init__(self) -> None:
        """Initialize the context object space."""


class VulkanGlobal(BackendGlobalObject):
    windows: dict[Window, VulkanSurfaceContext]

    def __init__(self):
        super().__init__()
        self.cached_programs = {}
        self.debug_callback_handle = None
        self._object_space = ObjectSpace()
        self.user_config = VulkanUserConfig()

        self.instance = VulkanInstance(self.user_config)

        self.devices = VulkanDevices(self, self.user_config)

        assert self.devices.logical_device is not None

        self.descriptor_mgr = DescriptorManager(self.devices.logical_device)
        self.renderpass_mgr = RenderPassManager(self.devices.logical_device)
        self.pipeline_mgr = GraphicsPipelineManager(self.devices, self.descriptor_mgr, self.cached_programs, self.renderpass_mgr)
        self.command_pool = CommandPool(self.devices.logical_device)
        self.resource_removal = DeferredResourceRemoval(self.devices.logical_device)

    @property
    def object_space(self) -> ObjectSpace:
        return self._object_space

    def post_init(self) -> None:
        pass

    def initialize_matrices(self, window: Window) -> VulkanMatrices:
        return VulkanMatrices(window, self)

    def set_viewport(self, window, x: int, y: int, width: int, height: int) -> None:
        pass

    def get_default_configs(self):
        return [
            VulkanUserConfig(),
        ]

    def get_config(self, **kwargs: bool | int | str | None) -> VulkanUserConfig:
        return VulkanUserConfig(**kwargs)

    def get_default_batch(self):
        context = self.resolve_context()
        if not hasattr(context, "default_batch"):
            context.default_batch = pyglet.graphics.Batch()

        return context.default_batch

    @property
    def current_window(self) -> VulkanSurfaceContext:
        ctx = self.current_context
        if not isinstance(ctx, NullContext):
            return ctx
        if self.windows:
            return next(iter(self.windows.values()))
        msg = "No Vulkan surface context is available."
        raise RuntimeError(msg)

    def get_surface_context(self, window: Window, config: VulkanSurfaceConfig,
                            shared: VulkanInstance) -> SurfaceContext:
        context = self.windows[window] = VulkanSurfaceContext(self, window, config, self.devices)
        self._have_context = True
        return context

    def get_cached_shader(self, name: str, *sources: tuple[str, str]) -> ShaderProgram:
        """Create a ShaderProgram.

        .. note:: This method is cached. Given the same shader sources, the
                  same ShaderProgram instance will be returned. For more
                  control over the ShaderProgram lifecycle, it is recommended
                  to manually create Shaders and link ShaderPrograms.

        .. versionadded:: 2.0.10
        """
        assert self.instance
        assert isinstance(name, str), "First argument must be a string name for the shader."
        if program := self.cached_programs.get(name):
            return program

        shaders = (Shader(src, srctype) for (src, srctype) in sources)
        program = ShaderProgram(*shaders)
        self.cached_programs[name] = program
        return program

    def get_shader(self, name: str) -> ShaderProgram:
        assert self.instance
        assert isinstance(name, str), "First argument must be a string name for the shader."
        return self.cached_programs[name]
        #
        # shaders = (Shader(src, srctype) for (src, srctype) in sources)
        # program = ShaderProgram(*shaders)
        # self.cached_programs[name] = program
        # return program

    def create_shader_program(self, name: str, *sources: Shader) -> ShaderProgram:
        print("CREATING NAME!", name, self.cached_programs)
        if name in self.cached_programs:
            msg = f"Shader name: {name} already exists."
            raise Exception(msg)
        shaders = (Shader(src, srctype) for (src, srctype) in sources)
        program = ShaderProgram(*shaders)
        program._id = name
        self.cached_programs[name] = program
        print("CACHED!", self.cached_programs)
        return program

    def create_shader(self, source_string: str, shader_type: ShaderType) -> Shader:
        return Shader(source_string, shader_type)

    def wait_idle(self):
        """Wait until the application device is idle.

        Blocking, so only call in specific circumstances when you need all resources to be idle.
        """
        logical = self.devices.logical_device
        if logical is None:
            return

        vk_device = getattr(logical, "vk_device", None)
        wait_idle = getattr(logical, "vkDeviceWaitIdle", None)
        if vk_device is None or wait_idle is None:
            return

        wait_idle(vk_device)

    @staticmethod
    def _delete_tracked_resources() -> None:
        """Delete internally tracked Vulkan resources prior to manager teardown."""
        # To prevent any possible circular imports.
        from pyglet.graphics.api.vulkan.buffer import VulkanUniformBufferObject  # noqa: PLC0415
        from pyglet.graphics.api.vulkan.draw import VulkanBatch  # noqa: PLC0415
        from pyglet.graphics.api.vulkan.shader import VulkanShaderProgram  # noqa: PLC0415
        from pyglet.graphics.api.vulkan.texture import VulkanTexture, VulkanSampler  # noqa: PLC0415

        VulkanBatch._delete_tracked_instances()  # noqa: SLF001
        VulkanUniformBufferObject._delete_tracked_instances()  # noqa: SLF001
        VulkanShaderProgram._delete_tracked_instances()  # noqa: SLF001
        VulkanTexture._delete_tracked_instances()  # noqa: SLF001
        VulkanSampler._delete_tracked_shared_samplers()  # noqa: SLF001

    def delete(self):
        self.wait_idle()
        self._delete_tracked_resources()

        """Initialize a full cleanup."""
        for window in self.windows.values():
            window.delete()
        self.windows.clear()

        # Shader programs are deleted by _delete_tracked_resources().
        self.cached_programs.clear()

        self.resource_removal.delete()

        self.pipeline_mgr.delete()
        self.descriptor_mgr.delete()
        self.command_pool.delete()

        # Finally destroy the logical device after everything is cleaned up.
        if self.devices.logical_device:
            self.devices.logical_device.delete()
            self.devices.logical_device = None

        self.instance.delete()


