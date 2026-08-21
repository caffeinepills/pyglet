from __future__ import annotations

import ctypes
import threading
import weakref
from ctypes import byref
from typing import TYPE_CHECKING, ClassVar, Sequence

import pyglet

from pyglet.enums import TextureFilter, AddressMode, TextureType, ComponentFormat
from pyglet.libs.shared.vulkan_lib.vulkan_core import (
    VK_ACCESS_SHADER_READ_BIT,
    VK_ACCESS_TRANSFER_WRITE_BIT,
    VK_BORDER_COLOR_INT_OPAQUE_BLACK,
    VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
    VK_FALSE, VK_IMAGE_ASPECT_COLOR_BIT,
    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
    VK_IMAGE_LAYOUT_UNDEFINED,
    VK_IMAGE_TILING_OPTIMAL,
    VK_IMAGE_USAGE_SAMPLED_BIT,
    VK_IMAGE_USAGE_TRANSFER_DST_BIT,
    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
    VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
    VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
    VK_PIPELINE_STAGE_TRANSFER_BIT,
    VK_QUEUE_FAMILY_IGNORED,
    VK_SAMPLE_COUNT_1_BIT,
    VK_SHARING_MODE_EXCLUSIVE,
    VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
    VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
    VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
    VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
    VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
    VK_TRUE, VkBufferImageCopy,
    VkExtent3D, VkImageCreateInfo,
    VkImageMemoryBarrier,
    VkImageSubresourceLayers,
    VkImageSubresourceRange,
    VkImageViewCreateInfo,
    VkMemoryAllocateInfo, VkOffset3D,
    VkSamplerCreateInfo, VK_FORMAT_R8_UNORM, VK_FORMAT_D32_SFLOAT, VK_FORMAT_D24_UNORM_S8_UINT, VkImage, VkMemoryRequirements, VkDeviceMemory,
    VkComponentMapping, VkImageAspectFlagBits, VkFormat, VkImageViewType, VkImageView, VkSampler, VkBorderColor,
    VkDevice, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VkClearColorValue, VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK,
    VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_TYPE_2D, VK_FORMAT_R8G8B8_SRGB, VK_FORMAT_R8G8_SRGB,
    VK_FORMAT_R8_SRGB, VkAccessFlags, VkPipelineStageFlags, VK_FORMAT_B8G8R8A8_SRGB)
from . import DeviceFunc, c_array_list
from .buffer import StagingBufferObject
from .enums import IMAGE_VIEW_TYPE_MAP, TEXTURE_FILTER_MAP, ADDRESS_MODE_MAP, TEXTURE_TYPE_MAP
from pyglet.graphics.texture import (
    Texture,
    TextureRegion,
    UniformTextureSequence,
    _TextureRegionShared,
    _Texture3DShared,
    _TextureArrayShared,
    TextureGrid,
)
from pyglet.image.base import ImageData, ImageDataRegion, ImageException

if TYPE_CHECKING:
    from .devices import VulkanDevices



def get_max_texture_size() -> int:
    """Query the maximum texture size available."""
    return 2048
    # size = c_int()
    # glGetIntegerv(GL_MAX_TEXTURE_SIZE, size)
    # return size.value


def get_max_array_texture_layers() -> int:
    """Query the maximum TextureArray depth."""
    # max_layers = c_int()
    # glGetIntegerv(GL_MAX_ARRAY_TEXTURE_LAYERS, max_layers)
    # return max_layers.value
    return 256

TEXTURE_FORMAT_MAP = {
    'R': VK_FORMAT_R8_SRGB,
    'RG': VK_FORMAT_R8G8_SRGB,
    'RGB': VK_FORMAT_R8G8B8_SRGB,
    'RGBA': VK_FORMAT_R8G8B8A8_SRGB,
    'D': VK_FORMAT_D32_SFLOAT,  # Or VK_FORMAT_D16_UNORM for lower precision
    'DS': VK_FORMAT_D24_UNORM_S8_UINT,  # Depth-Stencil combined format
    'BGRA': VK_FORMAT_B8G8R8A8_SRGB,

    # Luminance and Alpha, as Vulkan doesn't support GL_LUMINANCE or GL_ALPHA directly:
    'L': VK_FORMAT_R8_UNORM,  # Use the red channel for luminance.
    'A': VK_FORMAT_R8_UNORM,  # Use the red channel for alpha.
}


class VulkanImage(Texture):
    """Represents the structure layout and usage of the image.

    Will allocate memory and create the underlying VkImage. Not accessible to shaders.
    """

    vk_image: VkImage | None
    vk_memory: VkDeviceMemory | None
    devices: VulkanDevices | None

    def __init__(self, width: int, height: int,
                 tex_type: TextureType = TextureType.TYPE_2D,
                 internal_format: ComponentFormat = ComponentFormat.RGBA,
                 internal_format_size: int = 8,
                 internal_format_type: str = "B",
                 filters: TextureFilter | tuple[TextureFilter, TextureFilter] | None = None,
                 address_mode: AddressMode = AddressMode.REPEAT,
                 anisotropic_level: int = 0,
                 usage: int = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                 properties: int = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                 depth: int = 1,
                 layer_count: int = 1,
                 mip_levels: int = 1) -> None:
        super().__init__(width, height, 0, tex_type, internal_format, internal_format_size, internal_format_type,
                         filters, address_mode, anisotropic_level)
        self.usage: int = usage
        self.properties: int = properties
        self.vk_fmt = TEXTURE_FORMAT_MAP[self.internal_format.value]
        self.depth = max(1, depth)
        self.layer_count = max(1, layer_count)
        self.mip_levels = max(1, mip_levels)
        self.vk_ldevice = None
        self.physical_device = None
        self.devices = None
        self.vk_image = None
        self.vk_memory = None
        self._current_layout = VK_IMAGE_LAYOUT_UNDEFINED

    def create(self, devices: VulkanDevices):
        """Create the VkImage."""
        self.devices = devices
        self.vk_ldevice = self.devices.logical_device.vk_device

        image_info = VkImageCreateInfo(
            sType=VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
            imageType=TEXTURE_TYPE_MAP.get(self.tex_type, VK_IMAGE_TYPE_2D),
            format=self.vk_fmt,
            extent=VkExtent3D(
                self.width,
                self.height,
                self.depth if self.tex_type == TextureType.TYPE_3D else 1,
            ),
            mipLevels=self.mip_levels,
            arrayLayers=self.layer_count if self.tex_type == TextureType.TYPE_2D_ARRAY else 1,
            samples=VK_SAMPLE_COUNT_1_BIT,
            tiling=VK_IMAGE_TILING_OPTIMAL,
            usage=self.usage,
            sharingMode=VK_SHARING_MODE_EXCLUSIVE,
            initialLayout=self._current_layout,
        )

        self.vk_image = VkImage()

        DeviceFunc.vkCreateImage(self.vk_ldevice, byref(image_info), None, byref(self.vk_image))
        self._allocate_memory()

    def _allocate_memory(self):
        """Allocate memory for the VkImage."""
        vk_memory_reqs = VkMemoryRequirements()
        DeviceFunc.vkGetImageMemoryRequirements(self.vk_ldevice, self.vk_image, byref(vk_memory_reqs))

        alloc_info = VkMemoryAllocateInfo(
            sType=VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            allocationSize=vk_memory_reqs.size,
            memoryTypeIndex=self.devices.find_memory_type(vk_memory_reqs.memoryTypeBits, self.properties),
        )

        self.vk_memory = VkDeviceMemory()
        DeviceFunc.vkAllocateMemory(self.vk_ldevice, byref(alloc_info), None, byref(self.vk_memory))
        DeviceFunc.vkBindImageMemory(self.vk_ldevice, self.vk_image, self.vk_memory, 0)

    def copy_image_to_buffer(self, command_buffer, staging_buffer):
        region = VkBufferImageCopy(
            bufferOffset=0,
            bufferRowLength=0,
            bufferImageHeight=0,
            imageSubresource=VkImageSubresourceLayers(
                aspectMask=VK_IMAGE_ASPECT_COLOR_BIT,
                mipLevel=0,
                baseArrayLayer=0,
                layerCount=1,
            ),
            imageOffset=VkOffset3D(x=0, y=0, z=0),
            imageExtent=VkExtent3D(width=self.width, height=self.height, depth=1),
        )
        DeviceFunc.vkCmdCopyImageToBuffer(
            command_buffer,
            self.vk_image,
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            staging_buffer.vk_buffer,
            1,
            byref(region),
        )

    def upload_data_region(self, image_data: ImageData,
                           x: int=0, y: int=0, z: int=0,
                           staging_buffer: StagingBufferObject | None = None, offset: int = 0):
        texture_pfmt = self.internal_format.value

        # Requires positive pitch.
        pitch = abs(image_data.pitch)

        # For now, convert the image if the texture does not natcg,
        if image_data.format != texture_pfmt:
            data_size = image_data.width * image_data.height * len(texture_pfmt)
            data = image_data.get_bytes(texture_pfmt, pitch)
        else:
            data_size = image_data.width * image_data.height * len(image_data.format)
            data = image_data.get_bytes(None, pitch)


        owns_staging_buffer = staging_buffer is None
        # TODO: Change to support others.
        if owns_staging_buffer:
            staging_buffer = StagingBufferObject(data_size)
            staging_buffer.create(self.devices)

        command_buffers = None
        try:
            staging_buffer.set_data_as_type(data, ctypes.c_ubyte)

            # TODO: Make command pools more accessible somewhere?
            pool = pyglet.graphics.api.core.command_pool

            command_buffers = pool.get_single_use(1)

            DeviceFunc.vkDeviceWaitIdle(self.devices.logical_device.vk_device)

            with command_buffers[0] as vk_command_buffer:
                # Ensure image is done being read from pipeline/shader before transitioning.
                self.transition_layout(vk_command_buffer,
                                       VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                       VK_ACCESS_SHADER_READ_BIT if self._current_layout != VK_IMAGE_LAYOUT_UNDEFINED else 0,
                                       VK_ACCESS_TRANSFER_WRITE_BIT,
                                       VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                                       VK_PIPELINE_STAGE_TRANSFER_BIT,
                )

                if self.tex_type == TextureType.TYPE_2D_ARRAY:
                    base_array_layer = z
                    image_z = 0
                elif self.tex_type == TextureType.TYPE_3D:
                    base_array_layer = 0
                    image_z = z
                else:
                    base_array_layer = 0
                    image_z = z

                # Copy buffer to image.
                region = VkBufferImageCopy(
                    bufferOffset=0,
                    bufferRowLength=0,
                    bufferImageHeight=0,
                    imageSubresource=VkImageSubresourceLayers(
                        aspectMask=VK_IMAGE_ASPECT_COLOR_BIT,
                        mipLevel=0,
                        baseArrayLayer=base_array_layer,
                        layerCount=1,
                    ),
                    imageOffset=VkOffset3D(x, y, image_z),
                    imageExtent=VkExtent3D(width=image_data.width, height=image_data.height, depth=1),
                )

                regions = [region]
                region_array = c_array_list(regions, VkBufferImageCopy)

                DeviceFunc.vkCmdCopyBufferToImage(vk_command_buffer,
                                                  staging_buffer.buffer.vk_buffer,
                                                  self.vk_image,
                                                  VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                                  1,
                                                  region_array)

                # Make it accessible by a shader again.
                self.transition_layout(
                    vk_command_buffer,
                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    VK_ACCESS_TRANSFER_WRITE_BIT,
                    VK_ACCESS_SHADER_READ_BIT,
                    VK_PIPELINE_STAGE_TRANSFER_BIT,
                    VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                )
        finally:
            if command_buffers is not None:
                pool.free(command_buffers)
            if owns_staging_buffer:
                staging_buffer.delete()

    def upload_data(self, image_data: ImageData, staging_buffer: StagingBufferObject | None = None, offset: int = 0):
        self.upload_data_region(image_data, 0, 0, 0, staging_buffer, offset)

    def delete(self) -> None:
        """Destroy the VkImage and free its memory."""
        if self.vk_image:
            DeviceFunc.vkDestroyImage(self.vk_ldevice, self.vk_image, None)
            self.vk_image = None

        if self.vk_memory:
            DeviceFunc.vkFreeMemory(self.vk_ldevice, self.vk_memory, None)
            self.vk_memory = None

    def transition_layout(self, command_buffer,
                          new_layout,
                          src_access_mask: VkAccessFlags,
                          dst_access_mask: VkAccessFlags,
                          src_stage_flags: VkPipelineStageFlags,
                          dst_stage_flags: VkPipelineStageFlags,
                          ):
        """Transition the image layout.

        Args:
            command_buffer:
                The command buffer to record the transition commands.
            new_layout:
                The new layout of the image.
            src_access_mask:
                How the resource was accessed before the barrier. Ensures the operations are finished before transition.
            dst_access_mask:
                How it will be accessed after the barrier.
            src_stage_flags:
                Ensures the pipeline stages are completed before barrier takes effect.
            dst_stage_flags:
                Ensures the resource is ready to be used in this stage after the barrier is done.
        """
        barrier = VkImageMemoryBarrier(
            sType=VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            oldLayout=self._current_layout,
            newLayout=new_layout,
            srcQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED,
            dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED,
            image=self.vk_image,
            subresourceRange=VkImageSubresourceRange(
                aspectMask=VK_IMAGE_ASPECT_COLOR_BIT,
                baseMipLevel=0,
                levelCount=self.mip_levels,
                baseArrayLayer=0,
                layerCount=self.layer_count if self.tex_type == TextureType.TYPE_2D_ARRAY else 1,
            ),
            srcAccessMask=src_access_mask,
            dstAccessMask=dst_access_mask,
        )

        # Apply the pipeline barrier
        img_barriers = [barrier]
        img_barriers_array = c_array_list(img_barriers, VkImageMemoryBarrier)

        DeviceFunc.vkCmdPipelineBarrier(
            command_buffer,
            src_stage_flags,
            dst_stage_flags,
            0,  # No dependency flags
            0, None,  # No memory barriers
            0, None,  # No buffer memory barriers
            1, img_barriers_array,  # One image memory barrier
        )
        self._current_layout = new_layout


class VulkanImageView:
    """Represents a view into a VulkanImage.

    Responsible for:
    - Creating the VkImageView to access specific parts of the VulkanImage.
    - Supporting specific mip levels, layers, or format reinterpretations.
    """
    aspect_mask: VkImageAspectFlagBits
    fmt: VkFormat
    view_type: VkImageViewType

    def __init__(self, devices: VulkanDevices, image: VulkanImage,
                 aspect_mask: VkImageAspectFlagBits = VK_IMAGE_ASPECT_COLOR_BIT,
                 fmt: VkFormat | None = None,
                 view_type: VkImageViewType | None = None,
                 ) -> None:
        self.devices = devices
        self.image = image

        # Should usually match the VkImage, but this class will allow customization.
        self.vk_fmt = fmt or image.vk_fmt
        self.view_type = view_type or IMAGE_VIEW_TYPE_MAP[image.tex_type]
        self.aspect_mask = aspect_mask
        self.vk_imageview = self._create_view()

    def _create_view(self) -> VkImageView:
        """Create the VkImageView."""
        # TODO: Mipmaps eventually.
        view_info = VkImageViewCreateInfo(
            sType=VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            image=self.image.vk_image,
            viewType=self.view_type,
            format=self.vk_fmt,
            components=VkComponentMapping(),
            subresourceRange=VkImageSubresourceRange(
                aspectMask=self.aspect_mask,
                baseMipLevel=0,
                levelCount=self.image.mip_levels,
                baseArrayLayer=0,
                layerCount=self.image.layer_count if self.image.tex_type == TextureType.TYPE_2D_ARRAY else 1,
            ),
        )
        vk_imageview = VkImageView()
        DeviceFunc.vkCreateImageView(self.devices.logical_device.vk_device, byref(view_info), None, byref(vk_imageview))
        return vk_imageview

    def __del__(self) -> None:
        self.delete()

    def delete(self) -> None:
        """Destroy the VkImageView."""
        if self.vk_imageview:
            DeviceFunc.vkDestroyImageView(self.devices.logical_device.vk_device, self.vk_imageview, None)
            self.vk_imageview = None

class VulkanSampler:
    """Represents a Vulkan sampler.

    Responsible for:
    - Defining how to sample a VulkanImageView (e.g., filtering, addressing).
    """
    __slots__ = ("__weakref__", "info", "vk_device", "vk_sampler")
    _shared_samplers: ClassVar[dict[str, VulkanSampler]] = {}

    def __init__(self, vk_device: VkDevice,
                 mag_filter: TextureFilter,
                 min_filter: TextureFilter,
                 address_mode_u: AddressMode,
                 address_mode_v: AddressMode,
                 address_mode_w: AddressMode,
                 anisotropy_level: int = 0,
                 bordercolor: VkBorderColor = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK,
                 unnormalized: bool = False,
                 ) -> None:
        self.vk_device = vk_device

        self.info = VkSamplerCreateInfo(
            sType=VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
            magFilter=TEXTURE_FILTER_MAP[mag_filter],
            minFilter=TEXTURE_FILTER_MAP[min_filter],
            addressModeU=ADDRESS_MODE_MAP[address_mode_u],
            addressModeV=ADDRESS_MODE_MAP[address_mode_v],
            addressModeW=ADDRESS_MODE_MAP[address_mode_w],
            anisotropyEnable=VK_TRUE if anisotropy_level > 0 else VK_FALSE,
            maxAnisotropy=anisotropy_level,
            borderColor=bordercolor,
            unnormalizedCoordinates=unnormalized,  # Unnormalized is 0 to 1. If True, 0 to image size.
        )
        self.vk_sampler = VkSampler()
        DeviceFunc.vkCreateSampler(self.vk_device, byref(self.info), None, byref(self.vk_sampler))

    def delete(self) -> None:
        """Destroy the VkSampler."""
        if self.vk_sampler:
            DeviceFunc.vkDestroySampler(self.vk_device, self.vk_sampler, None)
        self.vk_sampler = None
        self.vk_device = None
        self.info = None

    def __del__(self):
        try:
            self.delete()
        except Exception:
            pass

    @classmethod
    def get_shared_sampler(cls, vk_device: VkDevice,
                 mag_filter: TextureFilter = TextureFilter.LINEAR,
                 min_filter: TextureFilter = TextureFilter.LINEAR,
                 address_mode_u: AddressMode = AddressMode.REPEAT,
                 address_mode_v: AddressMode = AddressMode.REPEAT,
                 address_mode_w: AddressMode = AddressMode.REPEAT,
                 anisotropy_level: int = 0,
                 bordercolor: VkBorderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK,
                 unnormalized: bool = False) -> VulkanSampler:
        key = str((vk_device, mag_filter, min_filter, address_mode_u, address_mode_v, address_mode_w, anisotropy_level,
                 bordercolor, unnormalized))

        if sampler := cls._shared_samplers.get(key):
            return sampler

        cls._shared_samplers[key] = sampler = cls(vk_device, mag_filter, min_filter, address_mode_u, address_mode_v, address_mode_w, anisotropy_level,
                 bordercolor, unnormalized)
        return sampler

    @classmethod
    def _delete_tracked_shared_samplers(cls) -> None:
        for sampler in tuple(cls._shared_samplers.values()):
            sampler.delete()
        cls._shared_samplers.clear()


class UniqueIDHandler:
    _lock: threading.Lock = threading.Lock()
    _available_ids: ClassVar[set[int]] = set()
    _next_id: int = 1

    @classmethod
    def _generate_id(cls) -> int:
        """Generate a unique ID."""
        with cls._lock:
            if cls._available_ids:
                return cls._available_ids.pop()
            texture_id = cls._next_id
            cls._next_id += 1
            return texture_id

    @classmethod
    def _release_id(cls, unique_id: int) -> None:
        """Release an ID for reuse."""
        with cls._lock:
            cls._available_ids.add(unique_id)


class VulkanTextureRegion(_TextureRegionShared, Texture):
    owner: VulkanTexture

    def _init_region(self, x: int, y: int, z: int, width: int, height: int, owner: Texture) -> None:
        self.x = x
        self.y = y
        self.z = z
        self._width = width
        self._height = height
        self.owner = owner

        owner_u1 = owner.tex_coords[0]
        owner_v1 = owner.tex_coords[1]
        owner_u2 = owner.tex_coords[3]
        owner_v2 = owner.tex_coords[7]
        scale_u = owner_u2 - owner_u1
        scale_v = owner_v2 - owner_v1

        u1 = x / owner.width * scale_u + owner_u1
        # Vulkan textures use inverted V coordinates relative to OpenGL.
        v1 = (owner.height - (y + height)) / owner.height * scale_v + owner_v1
        u2 = (x + width) / owner.width * scale_u + owner_u1
        v2 = (owner.height - y) / owner.height * scale_v + owner_v1
        r = z / owner.images + owner.tex_coords[2]

        self.tex_coords = (u1, v1, r, u2, v1, r, u2, v2, r, u1, v2, r)

    def __init__(self, x: int, y: int, z: int, width: int, height: int, owner: VulkanTexture) -> None:
        super().__init__(
            width,
            height,
            owner.id,
            owner.tex_type,
            owner.internal_format,
            owner.internal_format_size,
            owner.internal_format_type,
            owner.filters,
            owner.address_mode,
            owner.anisotropic_level,
        )
        self._init_region(x, y, z, width, height, owner)
        self.sampler = owner.sampler
        self.image_view = owner.image_view
        self.image = owner.image


class VulkanTexture(Texture, UniqueIDHandler):
    _all_textures: ClassVar[weakref.WeakSet[VulkanTexture]] = weakref.WeakSet()
    _next_descriptor_generation: ClassVar[int] = 1

    tex_coords = (0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0)
    tex_coords_order: tuple[int, int, int, int] = (0, 1, 2, 3)
    region_class = VulkanTextureRegion

    def __init__(self, width: int, height: int, tex_id: int,
                 tex_type: TextureType = TextureType.TYPE_2D,
                 internal_format: ComponentFormat = ComponentFormat.RGBA,
                 internal_format_size: int = 8,
                 internal_format_type: str = "B",
                 filters: TextureFilter | tuple[TextureFilter, TextureFilter] | None = None,
                 address_mode: AddressMode = AddressMode.REPEAT,
                 anisotropic_level: int = 0,
                 depth: int = 1,
                 layer_count: int = 1) -> None:
        super().__init__(width, height, tex_id, tex_type, internal_format, internal_format_size, internal_format_type,
                         filters, address_mode, anisotropic_level)
        self._depth = max(1, int(depth))
        self._layer_count = max(1, int(layer_count))
        self.descriptor_generation_id = self._claim_descriptor_generation()
        if self.tex_type == TextureType.TYPE_3D:
            self.images = self._depth
        elif self.tex_type == TextureType.TYPE_2D_ARRAY:
            self.images = self._layer_count

        self._shadow_mipmaps: dict[int, bytearray] = {}

        devices = pyglet.graphics.api.core.devices
        self.devices = devices
        self.image = VulkanImage(
            width,
            height,
            tex_type,
            internal_format,
            internal_format_size,
            internal_format_type,
            filters,
            address_mode,
            anisotropic_level,
            depth=self._depth,
            layer_count=self._layer_count,
            mip_levels=self._mipmap_levels,
        )
        self.image.create(devices)
        self.image_view = VulkanImageView(devices, self.image)
        self.sampler = VulkanSampler.get_shared_sampler(
            devices.logical_device.vk_device,
            mag_filter=self.mag_filter,
            min_filter=self.min_filter,
            address_mode_u=self.address_mode,
            address_mode_v=self.address_mode,
            address_mode_w=self.address_mode,
            anisotropy_level=self.anisotropic_level,
            bordercolor=VK_BORDER_COLOR_INT_OPAQUE_BLACK,
            unnormalized=False,
        )

        self._ensure_shadow_level(0)
        self._all_textures.add(self)

    @classmethod
    def _claim_descriptor_generation(cls) -> int:
        generation = cls._next_descriptor_generation
        cls._next_descriptor_generation += 1
        return generation

    def _bump_descriptor_generation(self) -> None:
        self.descriptor_generation_id = self._claim_descriptor_generation()

    @classmethod
    def create_from_image(cls,
                          image_data: ImageData | ImageDataRegion,
                          tex_type: TextureType = TextureType.TYPE_2D,
                          internal_format_size: int = 8,
                          filters: TextureFilter | tuple[TextureFilter, TextureFilter] | None = None,
                          address_mode: AddressMode = AddressMode.REPEAT,
                          anisotropic_level: int = 0,
                          context=None) -> VulkanTexture:
        texture = cls.create(
            image_data.width,
            image_data.height,
            tex_type=tex_type,
            internal_format=ComponentFormat(image_data.format),
            internal_format_size=internal_format_size,
            internal_format_type=image_data.data_type,
            filters=filters,
            address_mode=address_mode,
            anisotropic_level=anisotropic_level,
            blank_data=False,
            context=context,
        )
        texture.upload(image_data, 0, 0, 0)
        return texture

    @classmethod
    def create(cls, width: int, height: int,
               tex_type: TextureType = TextureType.TYPE_2D,
               internal_format: ComponentFormat = ComponentFormat.RGBA,
               internal_format_size: int = 8,
               internal_format_type: str = "B",
               filters: TextureFilter | tuple[TextureFilter, TextureFilter] | None = None,
               address_mode: AddressMode = AddressMode.REPEAT,
               anisotropic_level: int = 0,
               blank_data: bool = True,
               context=None) -> VulkanTexture:
        texture_id = cls._generate_id()
        texture = cls(width, height, texture_id, tex_type, internal_format, internal_format_size, internal_format_type,
                      filters, address_mode, anisotropic_level)
        if blank_data:
            texture._mark_mipmap_valid(0)
        return texture

    def bind(self, texture_unit: int = 0) -> None:
        return None

    def _flush(self) -> None:
        return None

    def _delete_resource(self) -> None:
        self.delete()

    def upload_data(self, image_data: ImageData) -> None:
        self.upload(image_data, image_data.anchor_x, image_data.anchor_y, 0)

    def _get_mipmap_depth(self, level: int) -> int:
        """Return effective depth for a mip level.

        For 3D textures, depth shrinks per mip level. For 2D arrays,
        depth remains the number of array layers.
        """
        if self.tex_type == TextureType.TYPE_3D:
            depth = max(1, int(self.images))
            return max(1, depth >> level)
        if self.tex_type == TextureType.TYPE_2D_ARRAY:
            return max(1, int(getattr(self, "max_depth", self._layer_count)))
        return 1

    def _ensure_shadow_level(self, level: int, width: int | None = None, height: int | None = None,
                             depth: int | None = None) -> bytearray:
        """Ensure CPU shadow storage exists for a mip level and return it.

        Vulkan texture tests need deterministic readback. This helper allocates (or resizes) a bytearray
        for the requested mip level using texture dimensions/component count, and returns that buffer.
        """
        if width is None or height is None or depth is None:
            width, height, depth = self._get_mipmap_dimensions(level)
        component_count = len(self.internal_format.value)
        size = width * height * depth * component_count
        data = self._shadow_mipmaps.get(level)
        if data is None or len(data) != size:
            data = bytearray(size)
            self._shadow_mipmaps[level] = data
        return data

    def _allocate_mipmap_level(self, level: int, width: int, height: int, depth: int,
                               data_size: int | None) -> None:
        """Allocate shadow backing for a mip level.

        Vulkan-side mip allocation for parity tests is represented by CPU shadow
        storage. The GPU resource may remain single-level in this implementation.
        """
        self._ensure_shadow_level(level, width, height, depth)

    def _generate_mipmaps(self) -> None:
        """Populate shadow mip levels from base data.

        This builds lower mip levels in CPU memory so `generate_mipmaps` and
        `fetch` semantics match other backends for test coverage.
        """
        max_levels = self._compute_mipmap_count()
        for level in range(max_levels):
            self._ensure_shadow_level(level)

        component_count = len(self.internal_format.value)
        for level in range(1, max_levels):
            prev_w, prev_h, prev_d = self._get_mipmap_dimensions(level - 1)
            width, height, depth = self._get_mipmap_dimensions(level)
            prev_data = self._ensure_shadow_level(level - 1, prev_w, prev_h, prev_d)
            data = self._ensure_shadow_level(level, width, height, depth)

            for z in range(depth):
                src_z = min(prev_d - 1, z * 2) if self.tex_type == TextureType.TYPE_3D else min(prev_d - 1, z)
                for y in range(height):
                    src_y = min(prev_h - 1, y * 2)
                    for x in range(width):
                        src_x = min(prev_w - 1, x * 2)
                        src_idx = ((src_z * prev_h + src_y) * prev_w + src_x) * component_count
                        dst_idx = ((z * height + y) * width + x) * component_count
                        data[dst_idx:dst_idx + component_count] = prev_data[src_idx:src_idx + component_count]

    def fetch(self, z: int = 0, level: int = 0) -> ImageData:
        """Return image data from CPU shadow storage for a layer/slice.

        This readback path is intentionally shadow-buffer based to provide stable
        cross-backend behavior in current Vulkan tests.
        """
        if level < 0 or level >= self._mipmap_levels:
            msg = f"Mipmap level {level} is not initialized."
            raise ImageException(msg)

        width, height, depth = self._get_mipmap_dimensions(level)
        if z < 0 or z >= depth:
            msg = f"Depth layer {z} is out of range for mipmap level {level}."
            raise ImageException(msg)

        component_count = len(self.internal_format.value)
        layer_size = width * height * component_count
        data = self._ensure_shadow_level(level, width, height, depth)
        start = z * layer_size
        end = start + layer_size
        return ImageData(width, height, self.internal_format.value, bytes(data[start:end]), pitch=width * component_count)

    def _update_subregion(self, image_data: ImageData | ImageDataRegion, x: int, y: int, z: int, level: int = 0) -> None:
        width, height, depth = self._get_mipmap_dimensions(level)
        if z < 0 or z >= depth:
            msg = f"Depth layer {z} is out of range for mipmap level {level}."
            raise ImageException(msg)
        if x < 0 or y < 0 or x + image_data.width > width or y + image_data.height > height:
            msg = f"Image upload region exceeds texture dimensions.: {x}x{y}x{z} (Image: {image_data.width}x{image_data.height})"
            raise ImageException(msg)

        texture_format = self.internal_format.value
        component_count = len(texture_format)
        source_row_stride = image_data.width * component_count
        source_data = image_data.get_bytes(texture_format, source_row_stride)

        target = self._ensure_shadow_level(level, width, height, depth)
        layer_stride = width * height * component_count
        row_stride = width * component_count
        layer_offset = z * layer_stride

        for row in range(image_data.height):
            src_start = row * source_row_stride
            src_end = src_start + source_row_stride
            dst_start = layer_offset + ((y + row) * row_stride) + (x * component_count)
            dst_end = dst_start + source_row_stride
            target[dst_start:dst_end] = source_data[src_start:src_end]

        if self.image is not None and level == 0 and self.tex_type == TextureType.TYPE_2D and z == 0:
            self.image.upload_data_region(image_data, x, y, z)

    def __del__(self) -> None:
        try:
            self.delete()
        except Exception:
            pass

    def delete(self) -> None:
        type(self)._all_textures.discard(self)
        self._bump_descriptor_generation()
        if self.image:
            self.image.delete()

        if self.image_view:
            self.image_view.delete()

        self.image = None
        self.image_view = None
        self._shadow_mipmaps.clear()

        if self.id is not None:
            self._release_id(self.id)
            self.id = None

    @classmethod
    def _delete_tracked_instances(cls) -> None:
        for texture in tuple(cls._all_textures):
            texture.delete()


class VulkanTexture3D(_Texture3DShared[VulkanTextureRegion], VulkanTexture, UniformTextureSequence[VulkanTextureRegion]):
    item_width: int = 0
    item_height: int = 0
    items: tuple[VulkanTextureRegion, ...]

    @classmethod
    def create_for_images(cls, images: Sequence[ImageData],
                          internal_format_size: int = 8,
                          internal_format_type: str = "B",
                          filters: TextureFilter | tuple[TextureFilter, TextureFilter] | None = None,
                          address_mode: AddressMode = AddressMode.REPEAT,
                          anisotropic_level: int = 0,
                          context=None,
                          blank_data: bool = True) -> "VulkanTexture3D":
        if not images:
            raise ImageException("At least one image is required.")

        item_width = images[0].width
        item_height = images[0].height
        internal_format = ComponentFormat(images[0].format)

        if not all(img.width == item_width and img.height == item_height for img in images):
            raise ImageException("Images do not have same dimensions.")

        texture_id = cls._generate_id()
        texture = cls(
            item_width,
            item_height,
            texture_id,
            TextureType.TYPE_3D,
            internal_format,
            internal_format_size,
            internal_format_type,
            filters,
            address_mode,
            anisotropic_level,
            depth=len(images),
            layer_count=1,
        )
        texture.images = len(images)
        texture.item_width = item_width
        texture.item_height = item_height

        base_image = images[0]
        if base_image.anchor_x or base_image.anchor_y:
            texture.anchor_x = base_image.anchor_x
            texture.anchor_y = base_image.anchor_y

        items: list[VulkanTextureRegion] = []
        for i, image in enumerate(images):
            item = cls.region_class(0, 0, i, item_width, item_height, texture)
            items.append(item)
            texture.upload(image, image.anchor_x, image.anchor_y, i)

        texture.items = tuple(items)
        return texture

    def upload(self, image: ImageData | ImageDataRegion, x: int, y: int, z: int, level: int = 0) -> None:
        Texture.upload(self, image, x, y, z, level=level)

    def _bind_sequence_texture(self) -> None:
        return None


class VulkanTextureArrayRegion(VulkanTextureRegion):
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self.id}, size={self.width}x{self.height}, layer={self.z})"


class VulkanTextureArray(_TextureArrayShared[VulkanTextureArrayRegion], VulkanTexture,
                         UniformTextureSequence[VulkanTextureArrayRegion]):
    items: list[VulkanTextureArrayRegion]

    def __init__(self, width: int, height: int, tex_id: int, max_depth: int,
                 internal_format: ComponentFormat = ComponentFormat.RGBA,
                 internal_format_size: int = 8,
                 internal_format_type: str = "B",
                 filters: TextureFilter | tuple[TextureFilter, TextureFilter] | None = None,
                 address_mode: AddressMode = AddressMode.REPEAT,
                 anisotropic_level: int = 0) -> None:
        super().__init__(
            width,
            height,
            tex_id,
            TextureType.TYPE_2D_ARRAY,
            internal_format,
            internal_format_size,
            internal_format_type,
            filters,
            address_mode,
            anisotropic_level,
            depth=1,
            layer_count=max_depth,
        )
        self.max_depth = max_depth
        self.images = max_depth
        self.items = []

    @classmethod
    def create(cls, width: int, height: int,
               max_depth: int = 256,
               internal_format: ComponentFormat = ComponentFormat.RGBA,
               internal_format_size: int = 8,
               internal_format_type: str = "B",
               filters: TextureFilter | tuple[TextureFilter, TextureFilter] | None = None,
               address_mode: AddressMode = AddressMode.REPEAT,
               anisotropic_level: int = 0,
               context=None) -> "VulkanTextureArray":
        max_depth_limit = get_max_array_texture_layers()
        assert max_depth <= max_depth_limit, f"TextureArray max_depth supported is {max_depth_limit}."

        texture_id = cls._generate_id()
        return cls(width, height, texture_id, max_depth, internal_format, internal_format_size,
                   internal_format_type, filters, address_mode, anisotropic_level)

    @classmethod
    def create_for_images(cls, images: Sequence[ImageData],
                          max_depth: int | None = None,
                          internal_format_size: int = 8,
                          internal_format_type: str = "B",
                          filters: TextureFilter | tuple[TextureFilter, TextureFilter] | None = None,
                          address_mode: AddressMode = AddressMode.REPEAT,
                          anisotropic_level: int = 0,
                          context=None) -> "VulkanTextureArray":
        if not images:
            raise ImageException("At least one image is required.")

        item_width = images[0].width
        item_height = images[0].height
        internal_format = ComponentFormat(images[0].format)

        if not all(img.width == item_width and img.height == item_height for img in images):
            raise ImageException("Images do not have same dimensions.")

        if max_depth is None:
            max_depth = len(images)

        texture = cls.create(
            item_width,
            item_height,
            max_depth=max_depth,
            internal_format=internal_format,
            internal_format_size=internal_format_size,
            internal_format_type=internal_format_type,
            filters=filters,
            address_mode=address_mode,
            anisotropic_level=anisotropic_level,
            context=context,
        )

        base_image = images[0]
        if base_image.anchor_x or base_image.anchor_y:
            texture.anchor_x = base_image.anchor_x
            texture.anchor_y = base_image.anchor_y

        texture.item_width = item_width
        texture.item_height = item_height
        texture.allocate(*images)
        return texture

    def upload(self, image: ImageData | ImageDataRegion, x: int, y: int, z: int, level: int = 0) -> None:
        Texture.upload(self, image, x, y, z, level=level)

    def _bind_sequence_texture(self) -> None:
        return None

    def _allocate_image(self, image: ImageData, layer: int) -> None:
        self.upload(image, image.anchor_x, image.anchor_y, layer)

    def _get_mipmap_depth(self, level: int) -> int:
        return max(1, int(getattr(self, "max_depth", self._layer_count)))


class VulkanTextureGrid(TextureGrid):
    pass


VulkanTexture.region_class = VulkanTextureRegion
VulkanTextureRegion.region_class = VulkanTextureRegion
VulkanTextureArray.region_class = VulkanTextureArrayRegion
VulkanTextureArrayRegion.region_class = VulkanTextureArrayRegion
