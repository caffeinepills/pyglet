from __future__ import annotations

import ctypes
import threading
import weakref
from ctypes import byref
from typing import TYPE_CHECKING, ClassVar

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
    VK_FORMAT_R8_SRGB, VkAccessFlags, VkPipelineStageFlags)
from . import DeviceFunc, c_array_list
from .buffer import StagingBufferObject
from .enums import IMAGE_VIEW_TYPE_MAP, TEXTURE_FILTER_MAP, ADDRESS_MODE_MAP
from pyglet.graphics.texture import Texture, TextureRegion

if TYPE_CHECKING:
    from pyglet.image.base import ImageData
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

TEXTURE_FORMAT_MAP = {
    'R': VK_FORMAT_R8_SRGB,
    'RG': VK_FORMAT_R8G8_SRGB,
    'RGB': VK_FORMAT_R8G8B8_SRGB,
    'RGBA': VK_FORMAT_R8G8B8A8_SRGB,
    'D': VK_FORMAT_D32_SFLOAT,  # Or VK_FORMAT_D16_UNORM for lower precision
    'DS': VK_FORMAT_D24_UNORM_S8_UINT,  # Depth-Stencil combined format

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
                 properties: int = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) -> None:
        super().__init__(width, height, 0, tex_type, internal_format, internal_format_size, internal_format_type,
                         filters, address_mode, anisotropic_level)
        self.usage: int = usage
        self.properties: int = properties
        self.vk_fmt = TEXTURE_FORMAT_MAP[self.internal_format.value]
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
            # imageType can be mapped from self.tex_type when additional texture types are supported.
            imageType=VK_IMAGE_TYPE_2D,
            format=self.vk_fmt,
            extent=VkExtent3D(self.width, self.height, 1),
            mipLevels=1,
            arrayLayers=1,
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
                           x=0, y=0, z=0,
                           staging_buffer: StagingBufferObject | None = None, offset: int = 0):
        texture_pfmt = self.internal_format.value

        # Requires positive pitch.
        pitch = abs(image_data.pitch)

        # For now, convert the image if the texture does not natcg,
        if image_data.format != texture_pfmt:
            data_size = image_data.width * image_data.height * len(texture_pfmt)
            data = image_data.get_bytes(texture_pfmt,pitch)
        else:
            data_size = image_data.width * image_data.height * len(image_data.format)
            data = image_data.get_bytes(None, pitch)

        # TODO: Change to support others.
        if not staging_buffer:
            staging_buffer = StagingBufferObject(data_size)
            staging_buffer.create(self.devices)

        staging_buffer.set_data_as_type(data, ctypes.c_ubyte)

        # TODO: Make command pools more accessible somewhere?
        pool = pyglet.graphics.api.core.command_pool

        command_buffers = pool.get_single_use(1)

        DeviceFunc.vkDeviceWaitIdle(self.devices.logical_device.vk_device)

        with command_buffers[0] as vk_command_buffer:
            # Ensure image is done being read from pipeline/shader before transitioning.
            self.transition_layout(vk_command_buffer,
                                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                   VK_ACCESS_SHADER_READ_BIT if self._current_layout is not VK_IMAGE_LAYOUT_UNDEFINED else 0,  # Maybe more later,
                                   VK_ACCESS_TRANSFER_WRITE_BIT,
                                   VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                                   VK_PIPELINE_STAGE_TRANSFER_BIT,
            )

            # Copy buffer to image.
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
                imageOffset=VkOffset3D(x, y, z),
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

    def upload_data(self, image_data: ImageData, staging_buffer: StagingBufferObject | None = None, offset: int = 0):
        self.upload_data_region(image_data, 0, 0, 0, staging_buffer, offset)

        # #print("Data uploaded.")
        # staging_buffer = StagingBufferObject2(self.width * self.height * 4)
        # staging_buffer.create(self.devices)
        # #
        # # # Now verify.
        # #with command_buffers[3] as vk_command_buffer:
        # with command_buffers[3] as vk_command_buffer:
        #     self.copy_image_to_buffer(vk_command_buffer, staging_buffer)
        #    self.transition_image_to_transfer_src(vk_command_buffer)
        #
        # with command_buffers[4] as vk_command_buffer:
        #     self.copy_image_to_buffer(vk_command_buffer, staging_buffer)
        #
        # pool.free(command_buffers)
        # raise Exception("End")
        #
        # with staging_buffer.map_memory() as mem_ptr:
        #     data = ctypes.string_at(mem_ptr, self.width * self.height * 4)
        #     print("STAGING BUFFER DATA!", data[:1000])

    def clear_color(self):
        clear_color = VkClearColorValue(float32=(ctypes.c_float * 4)(0.5, 0.5, 0.5, 1.0))  # Clear to red
        image_subresource = VkImageSubresourceRange(
            aspectMask=VK_IMAGE_ASPECT_COLOR_BIT,
            baseMipLevel=0,
            levelCount=1,
            baseArrayLayer=0,
            layerCount=1,
        )

        DeviceFunc.vkCmdClearColorImage(
            vk_command_buffer,
            self.vk_image,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            byref(clear_color),
            1,
            byref(image_subresource),
        )

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
                levelCount=1,
                baseArrayLayer=0,
                layerCount=1,
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
                levelCount=1,
                baseArrayLayer=0,
                layerCount=1,
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


class VulkanTextureRegion(TextureRegion):
    owner: VulkanTexture
    def __init__(self, x: int, y: int, z: int, width: int, height: int, owner: VulkanTexture) -> None:
        super().__init__(x, y, z, width, height, owner)
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

        # Vulkan adjustment: flip the `y` coordinates
        u1 = x / owner.width * scale_u + owner_u1
        v1 = (owner.height - (y + height)) / owner.height * scale_v + owner_v1  # Flip y
        u2 = (x + width) / owner.width * scale_u + owner_u1
        v2 = (owner.height - y) / owner.height * scale_v + owner_v1  # Flip y
        r = z / owner.images + owner.tex_coords[2]

        self.tex_coords = (u1, v1, r, u2, v1, r, u2, v2, r, u1, v2, r)
        self.sampler = owner.sampler
        self.image_view = owner.image_view
        self.image = owner.image

    def get_image_data(self):
        image_data = self.owner.get_image_data(self.z)
        return image_data.get_region(self.x, self.y, self.width, self.height)

    def get_region(self, x: int, y: int, width: int, height: int) -> TextureRegion:
        x += self.x
        y += self.y
        region = self.region_class(x, y, self.z, width, height, self.owner)
        region._set_tex_coords_order(*self.tex_coords_order)
        return region

    def blit_into(self, source, x: int, y: int, z: int) -> None:
        assert source.width <= self._width and source.height <= self._height, f"{source} is larger than {self}"
        raise Exception

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}(id={self.id},"
                f" size={self.width}x{self.height}, owner={self.owner.width}x{self.owner.height})")

    def delete(self) -> None:
        """Deleting a TextureRegion has no effect. Operate on the owning texture instead."""

    def __del__(self):
        pass


class VulkanTexture(Texture, UniqueIDHandler):
    _all_textures: ClassVar[weakref.WeakSet] = weakref.WeakSet()

    """An image loaded into GPU memory."""
    tex_coords = (0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0)
    """12-tuple of float, named (u1, v1, r1, u2, v2, r2, ...).
    ``u, v, r`` give the 3D texture coordinates for vertices 1-4. The vertices
    are specified in the order bottom-left, bottom-right, top-right and top-left.
    """

    tex_coords_order: tuple[int, int, int, int] = (0, 1, 2, 3)
    """The default vertex winding order for a quad.
    This defaults to counter-clockwise, starting at the bottom-left.
    """
    region_class = VulkanTextureRegion

    # All attributes and methods as defined earlier

    def __init__(self, width: int, height: int, tex_id: int,
                 tex_type: TextureType = TextureType.TYPE_2D,
                 internal_format: ComponentFormat = ComponentFormat.RGBA,
                 internal_format_size: int = 8,
                 internal_format_type: str = "B",
                 filters: TextureFilter | tuple[TextureFilter, TextureFilter] | None = None,
                 address_mode: AddressMode = AddressMode.REPEAT,
                 anisotropic_level: int = 0) -> None:  # noqa: D107
        super().__init__(width, height, tex_id, tex_type, internal_format, internal_format_size, internal_format_type,
                         filters, address_mode, anisotropic_level)
        devices = pyglet.graphics.api.core.devices
        self.devices = devices
        self.image = VulkanImage(width, height, tex_type, internal_format, internal_format_size, internal_format_type,
                                 filters, address_mode, anisotropic_level)
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

        self.device = None  # Vulkan device (assume set externally)
        self.physical_device = None  # Vulkan physical device (assume set externally)
        self.command_pool = None  # Command pool (assume set externally)
        self.graphics_queue = None  # Graphics queue (assume set externally)
        self.texture_image = None  # Vulkan Image (assume created elsewhere)
        self.texture_image_memory = None  # Vulkan memory for the image (assume created elsewhere)

        self._all_textures.add(self)

    def upload_data(self, image_data: ImageData):
        self.image.upload_data(image_data)

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
        """Create a Texture.

        Create a Texture with the specified dimensions and attributes.

        Args:
            width:
                Width of texture in pixels.
            height:
                Height of texture in pixels.
            tex_type:
                The texture enum type.
            internal_format:
                Component format of the image data.
            internal_format_size:
                Byte size of the internal format.
            internal_format_type:
                Internal format type in struct format.
            filters:
                Texture format filter, passed as a list of min/mag filters, or a single filter to apply both.
            address_mode:
                Texture address mode.
            anisotropic_level:
                The maximum anisotropic level.
            blank_data:
                If True, initialize the texture data with all zeros. If False, do not pass initial data.
        """
        texture_id = cls._generate_id()
        return cls(width, height, texture_id, tex_type, internal_format, internal_format_size, internal_format_type,
                   filters, address_mode, anisotropic_level)

    def bind(self, texture_unit: int = 0):
        pass

    def _update_subregion(self, image_data, x: int, y: int, z: int, level: int = 0):
        """Uploads ImageData into a specific region (x, y, z) of the texture."""
        image_size = image_data.width * image_data.height * 4  # Assuming RGBA (4 bytes per pixel)
        self.image.upload_data_region(image_data, x, y, z)

    def _flush(self) -> None:
        """Flushes the texture buffer to the device."""
        pass

    def blit_sequence_into(self, image_data_sequence):
        """Uploads a sequence of (ImageData, x, y, z) into the texture in a single batch."""
        # Calculate total size needed for the staging buffer
        total_size = sum(image_data.width * image_data.height * 4 for image_data, _, _, _ in image_data_sequence)

        # Create a larger staging buffer
        staging_buffer, staging_buffer_memory = create_buffer(
            self.device,
            self.physical_device,
            total_size,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        )

        # Map staging buffer and copy all pixel data into it
        data_ptr = vkMapMemory(self.device, staging_buffer_memory, 0, total_size, 0)
        offset = 0
        buffer_image_copies = []

        for image_data, x, y, z in image_data_sequence:
            image_size = image_data.width * image_data.height * 4  # Assuming RGBA
            ctypes.memmove(ctypes.addressof(data_ptr.contents) + offset, image_data.pixels, image_size)

            # Define the region to copy data to for each ImageData
            buffer_image_copy = VkBufferImageCopy()
            buffer_image_copy.bufferOffset = offset
            buffer_image_copy.bufferRowLength = 0  # Tightly packed
            buffer_image_copy.bufferImageHeight = 0
            buffer_image_copy.imageSubresource = VkImageSubresourceLayers(
                aspectMask=VK_IMAGE_ASPECT_COLOR_BIT,
                mipLevel=0,
                baseArrayLayer=0,
                layerCount=1,
            )
            buffer_image_copy.imageOffset = VkOffset3D(x, y, z)
            buffer_image_copy.imageExtent = VkExtent3D(image_data.width, image_data.height, 1)

            buffer_image_copies.append(buffer_image_copy)

            offset += image_size

        vkUnmapMemory(self.device, staging_buffer_memory)

        # Transition image layout to transfer destination
        self._transition_image_layout(self.texture_image, VK_IMAGE_LAYOUT_UNDEFINED,
                                      VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)

        # Copy from staging buffer to image (all in one batch)
        command_buffer = self._begin_single_time_commands()

        vkCmdCopyBufferToImage(
            command_buffer,
            staging_buffer,
            self.texture_image,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            len(buffer_image_copies),
            buffer_image_copies,
        )

        self._end_single_time_commands(command_buffer)

        # Transition image layout to shader read
        self._transition_image_layout(self.texture_image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                      VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)

        # Cleanup
        vkDestroyBuffer(self.device, staging_buffer, None)
        vkFreeMemory(self.device, staging_buffer_memory, None)

    def __del__(self):
        self.delete()

    def delete(self):
        type(self)._all_textures.discard(self)
        if self.image:
            self.image.delete()

        if self.image_view:
            self.image_view.delete()

        self.image = None
        self.image_view = None
        if self.id is not None:
            self._release_id(self.id)
            self.id = None

    @classmethod
    def _delete_tracked_instances(cls) -> None:
        for texture in tuple(cls._all_textures):
            texture.delete()

class VulkanTexture3D(VulkanTexture):
    ...

class VulkanTextureArrayRegion(VulkanTexture):
    ...

class VulkanTextureArray(VulkanTexture):
    ...

class VulkanTextureGrid(VulkanTexture):
    ...
