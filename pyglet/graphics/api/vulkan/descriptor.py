from __future__ import annotations

from ctypes import byref
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Sequence

from pyglet.graphics.api.vulkan import c_array_list
from pyglet.libs.shared.vulkan_lib.vulkan_core import (
    VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
    VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
    VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
    VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
    VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
    VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
    VkDescriptorBufferInfo,
    VkDescriptorImageInfo,
    VkDescriptorPool,
    VkDescriptorPoolCreateInfo,
    VkDescriptorPoolSize,
    VkDescriptorSet,
    VkDescriptorSetAllocateInfo,
    VkDescriptorSetLayout,
    VkDescriptorSetLayoutBinding,
    VkDescriptorSetLayoutCreateInfo,
    VkWriteDescriptorSet, VK_SUCCESS, VkDescriptorSetLayoutCreateFlags, VkCommandBuffer,
    VK_PIPELINE_BIND_POINT_GRAPHICS, VkPipelineLayout)
from pyglet.graphics.state import State

# from vulkan import VkDescriptorPoolSize, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER

if TYPE_CHECKING:
    from pyglet.graphics.api.vulkan.texture import VulkanTexture
    from pyglet.graphics.api.vulkan.buffer import UniformBuffer
    from pyglet.graphics.api.vulkan.devices import VulkanLogicalDevice
    from pyglet.graphics.api.vulkan.shader import VulkanShaderProgram, stages_to_bits


# VK_DESCRIPTOR_TYPE_SAMPLER = 0
# VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER = 1
# VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE = 2
# VK_DESCRIPTOR_TYPE_STORAGE_IMAGE = 3
# VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER = 4
# VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER = 5
# VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER = 6
# VK_DESCRIPTOR_TYPE_STORAGE_BUFFER = 7
# VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC = 8
# VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC = 9
# VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT = 10

class DescriptorPool:
    def __init__(self, device: VulkanLogicalDevice):
        self.device = device
        self.frames_in_flight = 0
        self.max_sets = 0
        self.pool_sizes = []
        self.vk_descriptor_pool = None

    def create_pool(self, frames_in_flight: int, max_sets: int):
        self.frames_in_flight = frames_in_flight
        self.max_sets = max_sets
        assert len(self.pool_sizes) == 0, "Pool sizes should be empty."

        self.pool_sizes = [
            VkDescriptorPoolSize(
                type=VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                descriptorCount=frames_in_flight * max_sets
            ),
            VkDescriptorPoolSize(
                type=VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                descriptorCount=frames_in_flight * max_sets
            ),
        ]

        poolsize_descs = c_array_list(self.pool_sizes, VkDescriptorPoolSize)
        pool_info = VkDescriptorPoolCreateInfo(
            sType=VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            poolSizeCount=len(self.pool_sizes),
            pPoolSizes=poolsize_descs,
            maxSets=frames_in_flight * max_sets,
        )

        self.vk_descriptor_pool = VkDescriptorPool()
        result = self.device.vkCreateDescriptorPool(self.device.vk_device, pool_info, None,
                                                    byref(self.vk_descriptor_pool))
        if result != VK_SUCCESS:
            msg = f"Failed to create descriptor pool: {result}"
            raise RuntimeError(msg)

    def allocate_descriptor_sets(self, layouts: list[VkDescriptorSetLayout]) -> list[list[VkDescriptorSet]]:
        """Allocate a descriptor set from the provided layouts.

        Will return multiple for each frame in flight.
        """
        assert self.vk_descriptor_pool is not None
        assert self.frames_in_flight != 0
        layout_ct = len(layouts)

        fif_layouts = layouts * self.frames_in_flight

        set_layouts = c_array_list(fif_layouts, VkDescriptorSetLayout)

        alloc_info = VkDescriptorSetAllocateInfo(
            sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            descriptorPool=self.vk_descriptor_pool,
            descriptorSetCount=self.frames_in_flight * layout_ct,
            pSetLayouts=set_layouts,
        )

        print("ALLOCATED?")

        descriptor_sets = (VkDescriptorSet * (layout_ct * self.frames_in_flight))()
        self.device.vkAllocateDescriptorSets(self.device.vk_device, byref(alloc_info), descriptor_sets)

        # Group by layout element_count.
        return [descriptor_sets[i:i+layout_ct]
            for i in range(0, layout_ct * self.frames_in_flight, layout_ct)]

    def __del__(self) -> None:
        self.delete()

    def delete(self) -> None:
        """Destroy the descriptor pool, and all descriptor sets it has created."""
        if self.vk_descriptor_pool:
            self.device.vkDestroyDescriptorPool(self.device.vk_device, self.vk_descriptor_pool, None)
            self.vk_descriptor_pool = None
        print("Destroyed Descriptor Pool")


class DescriptorSetLayoutCache:
    # Layout is cached by the flag, tuple
    layout_cache: dict[tuple[VkDescriptorSetLayoutCreateFlags, tuple], VkDescriptorSetLayout]

    def __init__(self, device: VulkanLogicalDevice) -> None:
        self.device = device
        self.layout_cache = {}  # Cache for VkDescriptorSetLayout objects

    def get(self, layout_bindings: dict[str, VkDescriptorSetLayoutBinding],
            flags: VkDescriptorSetLayoutCreateFlags=0) -> VkDescriptorSetLayout:
        """Retrieve a cached VkDescriptorSetLayout or create a new one if not cached.

        flags = VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT - Possible addition later.
        """
        all_bindings: tuple[VkDescriptorSetLayoutBinding] = tuple(layout_bindings.values())
        key = (flags, self._create_cache_key(all_bindings))
        if key in self.layout_cache:
            return self.layout_cache[key]

        layout_info = VkDescriptorSetLayoutCreateInfo(
            sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            flags=flags,
            bindingCount=len(all_bindings),
            pBindings=c_array_list(all_bindings, VkDescriptorSetLayoutBinding),
        )

        layout = VkDescriptorSetLayout()
        self.device.vkCreateDescriptorSetLayout(self.device.vk_device, byref(layout_info), None, byref(layout))

        self.layout_cache[key] = layout
        return layout

    def _create_cache_key(self, layout_bindings: Sequence[VkDescriptorSetLayoutBinding]) -> tuple:
        """Create a hashable key based on the layout info."""
        bindings = [
            (
                binding.binding,
                binding.descriptorType,
                binding.descriptorCount,
                binding.stageFlags,
            )
            for binding in layout_bindings
        ]
        return tuple(bindings)

    def delete(self) -> None:
        """Destroy all cached layouts."""
        for layout in self.layout_cache.values():
            self.device.vkDestroyDescriptorSetLayout(self.device.vk_device, layout, None)
        self.layout_cache.clear()
        self.device = None


class DescriptorManager:
    """Manages allocation and caching of the descriptors across pipelines and windows."""
    def __init__(self, device: VulkanLogicalDevice):
        self.device = device
        self.layout_cache = DescriptorSetLayoutCache(self.device)
        self.descriptor_pool = DescriptorPool(self.device)
        self.frames_in_flight = 0
        self.descriptor_set_cache = {}
        self._key = {}

        # Todo: Push descriptors VK_KHR_push_descriptor or version 1.1. -
        #  This extension allows descriptors to be written into the command buffer.

        # Todo: Support for VK_EXT_descriptor_indexing or version 1.2 - Adds support for the advanced descriptor
        #  management features, including UPDATE_AFTER_BIND and partially bound descriptors.

    def create_pool(self, frames_in_flight: int, max_sets: int = 10) -> None:
        assert self.frames_in_flight == 0, "Pool was already created. Recreating is not yet finished."
        self.frames_in_flight = frames_in_flight
        self.descriptor_pool.create_pool(frames_in_flight, max_sets)

    def get_descriptor_set_layouts(self, program: VulkanShaderProgram) -> list[VkDescriptorSetLayout]:
        """Get the descriptor set layout for the given program.

        Eventually may need to support multiple set layouts. (Other than, set 0)
        """
        bindings = program.get_layout_bindings()

        return [self.layout_cache.get(bindings, flags=0)]

    def get_descriptor_sets(self, set_layout: list[VkDescriptorSetLayout], resource_states: list[State]) -> tuple[DescriptorSetObject, bool]:
        """Allocate the descriptor sets.

        Will return a descriptor set for each frame in flight.
        """
        key = tuple(resource_states)
        if key in self.descriptor_set_cache:
            return self.descriptor_set_cache[key], False

        self.descriptor_set_cache[key] = dset = DescriptorSetObject(self.device, self.descriptor_pool.allocate_descriptor_sets(set_layout))
        return dset, True

    def get_shared_descriptor_sets(self, set_layout: list[VkDescriptorSetLayout]) -> DescriptorSetObject:
        """Allocate a shared descriptor set.

        A shared descriptor set is meant to be used for frequently updated descriptors. Instead of resources being set
        to a specific descriptor, the descriptor set updates the pointer to a different resource every frame or
        very frequently. This can lower the need for lots of descriptor sets.
        """
        return DescriptorSetObject(self.device, self.descriptor_pool.allocate_descriptor_sets(set_layout))

    def delete(self) -> None:
        self.layout_cache.delete()
        self.descriptor_pool.delete()


class DescriptorSetObject:
    """Container object for managing multiple descriptor sets.

    Also manages the frames in flight as well.
    """
    set_count: int

    def __init__(self, device: VulkanLogicalDevice, descriptor_sets_frames: list[list[VkDescriptorSet]]):
        self.device = device
        # set[desc_set_num][frame_in_flight]
        self.descriptor_sets_frames = descriptor_sets_frames
        self.current_frame = 0

        self.set_count = len(descriptor_sets_frames[0])

        self.desc_arrays = [c_array_list(descriptor_sets, VkDescriptorSet)
                           for descriptor_sets in descriptor_sets_frames]
        self.bindings = []
        #self.name_to_binding = {}  # Mapping of resource name to (binding, type)

    def bind_to_pipeline(self, command_buffer: VkCommandBuffer, pipeline_layout: VkPipelineLayout, frame_idx: int) -> None:
        self.current_frame = frame_idx

        self.device.vkCmdBindDescriptorSets(
            command_buffer,
            VK_PIPELINE_BIND_POINT_GRAPHICS,
            pipeline_layout,
            0,  # Starting set index
            self.set_count,  # Number of descriptor sets
            self.desc_arrays[frame_idx],  # Array of descriptor sets
            0, None,
        )

    # def add_binding(self, name, binding, descriptor_type, descriptor_count, stage_flags):
    #     self.bindings.append(VkDescriptorSetLayoutBinding(
    #         binding=binding,
    #         descriptorType=descriptor_type,
    #         descriptorCount=descriptor_count,
    #         stageFlags=stage_flags,
    #         pImmutableSamplers=None,
    #     ))
    #     # Add the name-to-binding mapping
    #     self.name_to_binding[name] = (binding, descriptor_type)

    # def update_resource_by_name(self, name, resource_info):
    #     # Look up the binding information by name
    #     if name not in self.name_to_binding:
    #         raise ValueError(f"Resource '{name}' not found in descriptor set.")
    #
    #     binding, descriptor_type = self.name_to_binding[name]
    #
    #     # Use the descriptor type to determine how to update the resource
    #     if descriptor_type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
    #         self._update_buffer(binding, resource_info)
    #     elif descriptor_type == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
    #         self._update_image_sampler(binding, resource_info)

    def bind_texture(self, texture: VulkanTexture, binding: int, desc_set: int=0):
        image_info = VkDescriptorImageInfo(
            sampler=texture.sampler.vk_sampler,
            imageView=texture.image_view.vk_imageview,
            imageLayout=VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        )

        image_info_array = c_array_list([image_info], VkDescriptorImageInfo)

        write = VkWriteDescriptorSet(
            sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            dstSet=self.descriptor_sets_frames[self.current_frame][desc_set],
            dstBinding=binding,
            dstArrayElement=0,
            descriptorCount=1,
            descriptorType=VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            pImageInfo=image_info_array,
        )

        write_array = c_array_list([write], VkWriteDescriptorSet)
        self.device.vkUpdateDescriptorSets(self.device.vk_device, 1, write_array, 0, None)

    # Bind a uniform buffer to the descriptor set
    def bind_ubo(self, ubo: UniformBufferObject, binding=0, desc_set=0):
        buffer_info = VkDescriptorBufferInfo(
            buffer=ubo.buffer.buffer.vk_buffer,
            offset=0,
            range=ubo.buffer.size,
        )

        buff_info_array = c_array_list([buffer_info], VkDescriptorBufferInfo)

        write = VkWriteDescriptorSet(
            sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            dstSet=self.descriptor_sets_frames[self.current_frame][desc_set],
            dstBinding=binding,
            dstArrayElement=0,
            descriptorCount=1,
            descriptorType=VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            pBufferInfo=buff_info_array,
        )

        write_array = c_array_list([write], VkWriteDescriptorSet)
        self.device.vkUpdateDescriptorSets(self.device.vk_device, 1, write_array, 0, None)