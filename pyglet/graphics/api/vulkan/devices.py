from __future__ import annotations

import ctypes
from ctypes import POINTER, byref, c_float, c_uint32, pointer

from pyglet.graphics.api.vulkan import DeviceFunc
from pyglet.libs.shared.vulkan_lib import (
    InstanceFunc,
    c_array_list,
    c_str_array_list,
    set_device_functions,
)
from pyglet.libs.shared.vulkan_lib.func_helpers import EnumeratePhysicalDevices, \
    EnumerateDeviceExtensionProperties, CreateDevice, \
    GetPhysicalDeviceQueueFamilyProperties, GetPhysicalDeviceSurfaceSupportKHR
from pyglet.libs.shared.vulkan_lib.vulkan_core import VK_KHR_SWAPCHAIN_EXTENSION_NAME, VkDeviceQueueCreateInfo, \
    VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, \
    VkDeviceCreateInfo, VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, VK_QUEUE_GRAPHICS_BIT, \
    VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU, VkPhysicalDeviceMemoryProperties, \
    VkPhysicalDeviceFeatures, VK_QUEUE_TRANSFER_BIT, VK_QUEUE_COMPUTE_BIT, VkPhysicalDevice, VkPhysicalDeviceProperties, \
    VK_SUCCESS, VkQueue, VkMemoryPropertyFlagBits, VkDevice, VkPhysicalDeviceFeatures2, \
    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, VkPhysicalDeviceDescriptorIndexingFeatures, \
    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES, VkPhysicalDeviceTimelineSemaphoreFeatures, \
    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES, VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME, \
    VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME, VK_API_VERSION_1_2

import pyglet
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pyglet.graphics.api import VulkanGlobal

_debug_api = pyglet.options.debug_api

if TYPE_CHECKING:
    from pyglet.graphics.api.vulkan.instance import VulkanSurface, VulkanInstance


class QueueFamily:
    indice: int
    vk_queue: VkQueue

    __slots__ = ('indice', 'vk_queue')

    def __init__(self, device: VulkanLogicalDevice, indice: int, queue_index: int = 0) -> None:  # noqa: D107
        self.indice = indice

        self.vk_queue = VkQueue()
        device.vkGetDeviceQueue(device.vk_device, indice, queue_index, byref(self.vk_queue))


class VulkanLogicalDevice:
    """Create the logical device and retrieve the graphics and presentation queues.

    A logical device is an abstraction representing the physical device (GPU)
    and is responsible for managing communication between the application and
    the physical GPU.

    The graphics queue handles rendering commands, while the present queue is
    responsible for presenting rendered images to the screen.
    """
    vk_device: VkDevice | None
    compute_queue: QueueFamily | None
    transfer_queue: QueueFamily | None
    graphics_queue: QueueFamily | None
    present_queue: QueueFamily | None

    def __init__(self, instance: VulkanInstance, physical: VulkanPhysicalGraphicsDevice):
        super().__init__()
        self.instance = instance
        self.physical = physical
        self.api_version = int(self.physical.properties.apiVersion)
        self.device_extensions = []
        self.descriptor_indexing_enabled = False
        self.descriptor_binding_partially_bound = False
        self.descriptor_binding_uniform_buffer_update_after_bind = False
        self.descriptor_binding_sampled_image_update_after_bind = False
        self.timeline_semaphore_enabled = False

        self.graphics_queue = None
        self.present_queue = None
        self.transfer_queue = None
        self.compute_queue = None

        if not pyglet.options.headless or self.instance.headless_surface_enabled:
            self.device_extensions.append(VK_KHR_SWAPCHAIN_EXTENSION_NAME)

        self._configure_optional_features()

        if self.device_extensions:
            assert self.physical.have_extensions(*self.device_extensions) is True, \
                f"Your device does not support these extensions: {self.device_extensions}."

        #self.vk_device = self._create_device()
        self.vk_device = None

    def supports_presentation(self):
        return VK_KHR_SWAPCHAIN_EXTENSION_NAME in self.device_extensions

    # def _create_device(self) -> VkDevice:
    #     """Create a logical device."""
    #     # Find queue families, excluding presentation if surface is not provided
    #     graphics_indice, present_indice, transfer_indice, compute_indice = self.find_queue_families()
    #
    #     print(f"Queue Indices chosen {graphics_indice=}, {present_indice=}, {transfer_indice=}, {compute_indice=}")
    #
    #     unique_queue_families = {graphics_indice, transfer_indice, compute_indice}
    #     if present_indice is not None:
    #         unique_queue_families.add(present_indice)
    #
    #     queues_create = [VkDeviceQueueCreateInfo(
    #         sType=VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
    #         queueFamilyIndex=queue_family,
    #         queueCount=1,
    #         pQueuePriorities=c_array_list([1.0], c_float),  # High priority for all queues
    #     ) for queue_family in unique_queue_families]
    #
    #     device_create = VkDeviceCreateInfo(
    #         sType=VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
    #         pQueueCreateInfos=c_array_list(queues_create, VkDeviceQueueCreateInfo),
    #         queueCreateInfoCount=len(queues_create),
    #         pEnabledFeatures=pointer(self.physical.features),
    #         enabledExtensionCount=len(self.device_extensions),
    #         ppEnabledExtensionNames=c_str_array_list(self.device_extensions),
    #     )
    #
    #     return CreateDevice(self.physical.vk_device, device_create)

    def create(self, surface: VulkanSurface | None) -> None:
        """Create a logical device.

        A surface is required only when presentation/swapchain support is enabled.
        """
        assert self.vk_device is None, "The VK Device already exists."
        if self.supports_presentation() and surface is None:
            msg = "A Vulkan surface is required when presentation support is enabled."
            raise RuntimeError(msg)
        graphics_indice, present_indice, transfer_indice, compute_indice = self.find_queue_families(surface)

        print(f"Queue Indices chosen {graphics_indice=}, {present_indice=}, {transfer_indice=}, {compute_indice}")

        unique_queue_families = {graphics_indice, present_indice, transfer_indice, compute_indice}

        queues_create = [VkDeviceQueueCreateInfo(
                sType=VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                queueFamilyIndex=queue_family,
                queueCount=1,
                pQueuePriorities=c_array_list([1.0], c_float),  # High priority for all queues
            ) for queue_family in unique_queue_families if queue_family is not None]

        device_pnext = None
        descriptor_indexing_features = None
        timeline_features = None
        if self.descriptor_indexing_enabled:
            descriptor_indexing_features = VkPhysicalDeviceDescriptorIndexingFeatures(
                sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES,
                pNext=None,
                descriptorBindingPartiallyBound=self.descriptor_binding_partially_bound,
                descriptorBindingUniformBufferUpdateAfterBind=self.descriptor_binding_uniform_buffer_update_after_bind,
                descriptorBindingSampledImageUpdateAfterBind=self.descriptor_binding_sampled_image_update_after_bind,
            )
            device_pnext = ctypes.cast(pointer(descriptor_indexing_features), POINTER(None))

        if self.timeline_semaphore_enabled:
            timeline_features = VkPhysicalDeviceTimelineSemaphoreFeatures(
                sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES,
                pNext=device_pnext,
                timelineSemaphore=True,
            )
            device_pnext = ctypes.cast(pointer(timeline_features), POINTER(None))

        device_create = VkDeviceCreateInfo(
            sType=VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            pNext=device_pnext,
            pQueueCreateInfos=c_array_list(queues_create, VkDeviceQueueCreateInfo),
            queueCreateInfoCount=len(queues_create),
            pEnabledFeatures=pointer(self.physical.features),
            enabledExtensionCount=len(self.device_extensions),
            ppEnabledExtensionNames=c_str_array_list(self.device_extensions),
        )

        self.vk_device = CreateDevice(self.physical.vk_device, device_create)

        # Loaded into a function module and to this instance.
        funcs = set_device_functions(self.vk_device, self.instance.modules)
        for func_info in funcs:
            fn_name, fn = func_info
            setattr(self, fn_name, fn)

        self.graphics_queue = QueueFamily(self, graphics_indice)
        self.present_queue = QueueFamily(self, present_indice) if present_indice is not None else None
        self.transfer_queue = QueueFamily(self, transfer_indice)
        self.compute_queue = QueueFamily(self, compute_indice)

    @staticmethod
    def _is_callable_loaded(func) -> bool:
        return callable(func) and getattr(func, "__name__", "") != "MissingFunction"

    def _get_features2_getter(self):
        getter = getattr(self.instance, "vkGetPhysicalDeviceFeatures2", None)
        if self._is_callable_loaded(getter):
            return getter

        getter_khr = getattr(self.instance, "vkGetPhysicalDeviceFeatures2KHR", None)
        if self._is_callable_loaded(getter_khr):
            return getter_khr

        return None

    def _configure_optional_features(self) -> None:
        instance_api_version = int(getattr(self.instance, "api_version", 0))
        api_is_1_2_or_higher = (
            self.api_version >= VK_API_VERSION_1_2
            and instance_api_version >= VK_API_VERSION_1_2
        )
        has_descriptor_indexing_ext = self.physical.have_extension(VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME)
        has_timeline_semaphore_ext = self.physical.have_extension(VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME)

        wants_descriptor_indexing = api_is_1_2_or_higher or has_descriptor_indexing_ext
        wants_timeline = api_is_1_2_or_higher or has_timeline_semaphore_ext
        if not wants_descriptor_indexing and not wants_timeline:
            if _debug_api:
                print("(Vulkan) Descriptor indexing and timeline semaphore are unavailable for this API/driver.")
            return

        get_features2 = self._get_features2_getter()
        if get_features2 is None:
            if _debug_api:
                print("(Vulkan) vkGetPhysicalDeviceFeatures2* unavailable; optional descriptor/timeline features disabled.")
            return

        features2 = VkPhysicalDeviceFeatures2(
            sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
            pNext=None,
        )

        descriptor_features = None
        timeline_features = None
        pnext_chain = None

        if wants_descriptor_indexing:
            descriptor_features = VkPhysicalDeviceDescriptorIndexingFeatures(
                sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES,
                pNext=None,
            )
            pnext_chain = ctypes.cast(pointer(descriptor_features), POINTER(None))

        if wants_timeline:
            timeline_features = VkPhysicalDeviceTimelineSemaphoreFeatures(
                sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES,
                pNext=pnext_chain,
            )
            pnext_chain = ctypes.cast(pointer(timeline_features), POINTER(None))

        features2.pNext = pnext_chain
        get_features2(self.physical.vk_device, byref(features2))

        if descriptor_features is not None:
            self.descriptor_binding_uniform_buffer_update_after_bind = bool(
                descriptor_features.descriptorBindingUniformBufferUpdateAfterBind
            )
            self.descriptor_binding_sampled_image_update_after_bind = bool(
                descriptor_features.descriptorBindingSampledImageUpdateAfterBind
            )
            self.descriptor_binding_partially_bound = bool(descriptor_features.descriptorBindingPartiallyBound)

            self.descriptor_indexing_enabled = bool(
                self.descriptor_binding_uniform_buffer_update_after_bind
                and self.descriptor_binding_sampled_image_update_after_bind
            ) or self.descriptor_binding_partially_bound

            if self.descriptor_indexing_enabled and not api_is_1_2_or_higher and has_descriptor_indexing_ext:
                self.device_extensions.append(VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME)

            if _debug_api:
                if self.descriptor_indexing_enabled:
                    mode = "core Vulkan 1.2+" if api_is_1_2_or_higher else "VK_EXT_descriptor_indexing"
                    print(
                        "(Vulkan) Descriptor indexing enabled "
                        f"({mode}); partially_bound={self.descriptor_binding_partially_bound}, "
                        "update_after_bind_ubo="
                        f"{self.descriptor_binding_uniform_buffer_update_after_bind}, "
                        "update_after_bind_sampler="
                        f"{self.descriptor_binding_sampled_image_update_after_bind}."
                    )
                elif wants_descriptor_indexing:
                    print("(Vulkan) Descriptor indexing requested, but required device features are unavailable.")

        if timeline_features is not None:
            self.timeline_semaphore_enabled = bool(timeline_features.timelineSemaphore)
            if self.timeline_semaphore_enabled and not api_is_1_2_or_higher and has_timeline_semaphore_ext:
                self.device_extensions.append(VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME)

            if _debug_api:
                if self.timeline_semaphore_enabled:
                    mode = "core Vulkan 1.2+" if api_is_1_2_or_higher else "VK_KHR_timeline_semaphore"
                    print(f"(Vulkan) Timeline semaphore enabled ({mode}).")
                elif wants_timeline:
                    print("(Vulkan) Timeline semaphore requested, but timelineSemaphore feature is unavailable.")

    def find_queue_families(self, surface: VulkanSurface | None):
        """Finds the queue families that support graphics, presentation, transfer, and compute operations.

        Falls back to general-purpose queues if dedicated ones are not available.
        """
        queue_families = GetPhysicalDeviceQueueFamilyProperties(self.physical.vk_device)
        queue_family_graphics_index = None
        queue_family_present_index = None
        queue_family_transfer_index = None
        queue_family_compute_index = None
        fallback_transfer_index = None
        fallback_compute_index = None

        for i, queue_family in enumerate(queue_families):
            #print("Queue Family Index:", i, "Queue Flags:", queue_family.queueFlags, "Queue Count:",
            #      queue_family.queueCount)

            # Graphics queue
            if queue_family.queueCount > 0 and queue_family.queueFlags & VK_QUEUE_GRAPHICS_BIT:
                if queue_family_graphics_index is None:
                    queue_family_graphics_index = i

            # Present queue
            if VK_KHR_SWAPCHAIN_EXTENSION_NAME in self.device_extensions:
                support_present = GetPhysicalDeviceSurfaceSupportKHR(self.physical.vk_device, i, surface.vk_surface)
                if queue_family.queueCount > 0 and support_present:
                    if queue_family_present_index is None:
                        queue_family_present_index = i

            # Dedicated transfer queue
            if (queue_family.queueCount > 0 and
                    queue_family.queueFlags & VK_QUEUE_TRANSFER_BIT and
                    not (
                            queue_family.queueFlags & VK_QUEUE_GRAPHICS_BIT or queue_family.queueFlags & VK_QUEUE_COMPUTE_BIT)):
                if queue_family_transfer_index is None:
                    queue_family_transfer_index = i

            # Fallback transfer queue
            if queue_family.queueCount > 0 and queue_family.queueFlags & VK_QUEUE_TRANSFER_BIT:
                if fallback_transfer_index is None:
                    fallback_transfer_index = i

            # Dedicated compute queue
            if (queue_family.queueCount > 0 and
                    queue_family.queueFlags & VK_QUEUE_COMPUTE_BIT and
                    not (queue_family.queueFlags & VK_QUEUE_GRAPHICS_BIT)):
                if queue_family_compute_index is None:
                    queue_family_compute_index = i

            # Fallback compute queue
            if queue_family.queueCount > 0 and queue_family.queueFlags & VK_QUEUE_COMPUTE_BIT:
                if fallback_compute_index is None:
                    fallback_compute_index = i

        # Assign fallback indices if no dedicated transfer or compute queue is found
        if queue_family_transfer_index is None:
            queue_family_transfer_index = fallback_transfer_index
        if queue_family_compute_index is None:
            queue_family_compute_index = fallback_compute_index

        # Ensure essential queues are found
        if queue_family_graphics_index is None:
            raise Exception("Failed to find suitable queue families for rendering")
        if queue_family_transfer_index is None:
            raise Exception("Failed to find a suitable transfer queue.")
        if queue_family_compute_index is None:
            raise Exception("Failed to find a suitable compute queue.")
        if queue_family_present_index is None and VK_KHR_SWAPCHAIN_EXTENSION_NAME in self.device_extensions:
            raise Exception("Failed to find a suitable queue family for presenting")

        return (queue_family_graphics_index, queue_family_present_index, queue_family_transfer_index,
                queue_family_compute_index)

    def __del__(self) -> None:
        self.delete()

    def delete(self):
        """Clean up the Vulkan device and release resources.

        The logical device represents the physical device, and deleting it releases
        the allocated resources. This method also resets the graphics and presentation
        queues.
        """
        if self.vk_device:
            self.vkDestroyDevice(self.vk_device, None)
            self.vk_device = None
        self.graphics_queue = None
        self.present_queue = None
        self.transfer_queue = None
        self.compute_queue = None


def get_win32_presentation_support(physical_device: VulkanPhysicalGraphicsDevice, queue_family_index: int):
    vk_bool = InstanceFunc.vkGetPhysicalDeviceWin32PresentationSupportKHR(physical_device.vk_device, queue_family_index)
    return bool(vk_bool.value)

def get_xlib_presentation_support(physical_device: VulkanPhysicalGraphicsDevice, queue_family_index: int):
    pass

_os_to_presentation = {
    "win32": get_win32_presentation_support,
    "linux": get_xlib_presentation_support,
}

class VulkanPhysicalGraphicsDevice:
    """A physical graphics device."""
    _extensions: dict[str, int]
    memory_properties: VkPhysicalDeviceMemoryProperties
    features: VkPhysicalDeviceFeatures

    def __init__(self, instance: VulkanInstance, vk_device: VkPhysicalDevice):
        self.instance = instance

        self.vk_device = vk_device

        self.properties = VkPhysicalDeviceProperties()
        instance.vkGetPhysicalDeviceProperties(self.vk_device, byref(self.properties))

        self.name = self.properties.deviceName.decode('utf-8')

        self.features = VkPhysicalDeviceFeatures()
        instance.vkGetPhysicalDeviceFeatures(self.vk_device, byref(self.features))

        self._extensions = {}
        self._enumerate_extensions()

        self.memory_properties = VkPhysicalDeviceMemoryProperties()
        instance.vkGetPhysicalDeviceMemoryProperties(self.vk_device, byref(self.memory_properties))
        #print("Memory", self.memory_properties)

    def have_extension(self, name: str, version: int | None=None) -> bool:
        """Determine if the extension is supported.

        If a version number is included, will ensure the version matches.
        """
        if version is None:
            return name in self._extensions

        return bool(name in self._extensions and version == self._extensions[name])

    def have_extensions(self, *names: str) -> bool:
        return all(name in self._extensions for name in names)

    def get_extension_version(self, name: str) -> int | None:
        """Get the extension version."""
        if name in self._extensions:
            return self._extensions[name]

        return None

    def get_extensions(self) -> set[str]:
        return set(self._extensions.keys())

    def _enumerate_extensions(self) -> None:
        extensions = EnumerateDeviceExtensionProperties(self.instance, self.vk_device, None)
        for ext in extensions:
            self._extensions[ext.extensionName.decode('utf-8')] = ext.specVersion

    def delete(self):
        if self.vk_device:
            DeviceFunc.vkDestroyDevice(self.vk_device, None)
            self.vk_device = None

    @property
    def is_discrete(self):
        return self.properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU

    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.name}')"


class VulkanDevices:
    """Manages the Vulkan devices.

    Including physical and logical device creation,
    queue retrieval, and applying user configuration settings such as multisampling
    and depth size.
    """
    logical_device: VulkanLogicalDevice | None
    physical_device: VulkanPhysicalGraphicsDevice
    physical_devices: list[VulkanPhysicalGraphicsDevice]

    def __init__(self, vulkan_global: VulkanGlobal, config):
        self.config = config
        self.instance = vulkan_global.instance
        self.physical_devices = self.get_physical_devices()

        # Choose a discrete graphics card as the default we will use.
        # Possible to use multiple GPU's, but will just use one for now.
        self.physical_device = sorted(self.physical_devices, key=lambda dev: dev.is_discrete is True)[0]

        self.logical_device = VulkanLogicalDevice(vulkan_global.instance, self.physical_device)

        # Apply user configuration settings such as multisampling, depth size, etc.
        self.apply_config_settings()

    def find_memory_type(self, type_filter: int, properties: VkMemoryPropertyFlagBits) -> int:
        """Get a compatible memory type from the physical device.

        Args:
            type_filter:
                A bitmask representing the types of memory the resource can be allocated
                from (e.g., device-local, host-visible).

            properties:
                The memory property flags that need to match:

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
        mem_properties = self.physical_device.memory_properties

        for i in range(mem_properties.memoryTypeCount):
            if (type_filter & (1 << i)) and (
                    mem_properties.memoryTypes[i].propertyFlags & properties) == properties:
                return i

        raise Exception("Failed to find suitable memory type")

    def apply_config_settings(self):
        """Apply user-defined configuration settings (e.g., multisampling and depth size)
        from the VulkanConfig object to the Vulkan pipeline or rendering system.
        """
        # if 'samples' in self.config.settings:
        #     print(f"Applying multisampling: {self.config.settings['samples']} samples")
        # if 'depth_size' in self.config.settings:
        #     print(f"Setting depth buffer size: {self.config.settings['depth_size']} bits")

    def get_physical_devices(self) -> list[VulkanPhysicalGraphicsDevice]:
        """Select a physical device (GPU) based on the user's provided GPU name.
        If no name is provided, the first available GPU is selected.
        """
        physical_devices = EnumeratePhysicalDevices(self.instance)
        return [VulkanPhysicalGraphicsDevice(self.instance, device) for device in physical_devices]
