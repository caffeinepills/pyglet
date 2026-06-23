from __future__ import annotations

import ctypes
import weakref
from dataclasses import dataclass
from ctypes import POINTER, byref
from typing import TYPE_CHECKING, Any, Sequence

import pyglet
from pyglet.graphics.api.vulkan import c_array_list
from pyglet.graphics.api.vulkan.frame_local import FrameLocalResource
from pyglet.libs.shared.vulkan_lib.vulkan_core import (
    VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT,
    VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT,
    VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT,
    VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT,
    VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
    VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
    VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO,
    VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
    VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
    VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
    VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
    VkDescriptorBindingFlags,
    VkDescriptorBufferInfo,
    VkDescriptorImageInfo,
    VkDescriptorPool,
    VkDescriptorPoolCreateInfo,
    VkDescriptorPoolSize,
    VkDescriptorSet,
    VkDescriptorSetAllocateInfo,
    VkDescriptorSetLayout,
    VkDescriptorSetLayoutBinding,
    VkDescriptorSetLayoutBindingFlagsCreateInfo,
    VkDescriptorSetLayoutCreateInfo,
    VkWriteDescriptorSet,
    VK_SUCCESS,
    VkDescriptorSetLayoutCreateFlags,
    VkCommandBuffer,
    VK_PIPELINE_BIND_POINT_GRAPHICS,
    VkPipelineLayout,
    VkDescriptorPoolCreateFlags,
)
from pyglet.graphics.state import State

_debug_api = pyglet.options.debug_api

# from vulkan import VkDescriptorPoolSize, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER

if TYPE_CHECKING:
    from pyglet.graphics.api.vulkan.state import DescriptorResourceState
    from pyglet.graphics.api.vulkan.texture import VulkanTexture
    from pyglet.graphics.api.vulkan.buffer import UniformBuffer, VulkanUniformBufferObject
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


@dataclass(frozen=True, slots=True)
class DescriptorBindingKey:
    binding: int
    descriptor_type: int
    descriptor_count: int
    stage_flags: int

    @classmethod
    def from_layout_binding(cls, layout_binding: VkDescriptorSetLayoutBinding) -> DescriptorBindingKey:
        return cls(
            binding=int(layout_binding.binding),
            descriptor_type=int(layout_binding.descriptorType),
            descriptor_count=int(layout_binding.descriptorCount),
            stage_flags=int(layout_binding.stageFlags),
        )


@dataclass(frozen=True, slots=True)
class DescriptorSetLayoutKey:
    flags: int
    bindings: tuple[DescriptorBindingKey, ...]
    binding_flags: tuple[int, ...] = ()

    @classmethod
    def from_layout_bindings(
        cls,
        layout_bindings: Sequence[VkDescriptorSetLayoutBinding],
        flags: int = 0,
        binding_flags: Sequence[int] | None = None,
    ) -> DescriptorSetLayoutKey:
        return cls(
            flags=int(flags),
            bindings=tuple(DescriptorBindingKey.from_layout_binding(binding) for binding in layout_bindings),
            binding_flags=tuple(binding_flags) if binding_flags else (),
        )


@dataclass(frozen=True, slots=True)
class DescriptorSetLayoutEntryKey:
    set_index: int
    layout_key: DescriptorSetLayoutKey


@dataclass(frozen=True, slots=True)
class DescriptorSetLayoutsKey:
    set_layouts: tuple[DescriptorSetLayoutEntryKey, ...]


@dataclass(frozen=True, slots=True)
class DescriptorSetLayouts:
    key: DescriptorSetLayoutsKey
    layouts: tuple[VkDescriptorSetLayout, ...]


@dataclass(frozen=True, slots=True)
class DescriptorResourceKey:
    set_index: int
    binding: int
    descriptor_type: int
    descriptor_generation_id: int


@dataclass(frozen=True, slots=True)
class UniformBindingKey(DescriptorResourceKey):
    buffer_id: int
    offset: int
    range_size: int


@dataclass(frozen=True, slots=True)
class SamplerKey(DescriptorResourceKey):
    texture_id: int
    sampler_id: int
    image_view_id: int


@dataclass(frozen=True, slots=True)
class DescriptorSetCacheKey:
    layout_key: DescriptorSetLayoutsKey
    resources: tuple[DescriptorResourceKey, ...]
    frame_index: int
    owner_key: int | None


class DescriptorPool:
    """Owns the Vulkan descriptor pool used by cached descriptor sets.

    The pool itself is device/global Vulkan state, not a frame-lifetime object.
    It is sized by ``frames_in_flight`` because ``DescriptorSetObject`` allocates
    one compatible descriptor-set tuple per frame slot. Individual descriptor
    sets are selected by frame index when commands are recorded.
    """

    flags: VkDescriptorPoolCreateFlags
    def __init__(self, device: VulkanLogicalDevice):
        self.device = device
        self.frames_in_flight = 0
        self.max_sets = 0
        self.pool_sizes = []
        self.vk_descriptor_pool = None
        self.flags = 0

    def create_pool(self, frames_in_flight: int, max_sets: int, flags: int = 0):
        if self.vk_descriptor_pool:
            if self.frames_in_flight != frames_in_flight or self.max_sets != max_sets or self.flags != flags:
                msg = ("Descriptor pool is already initialized with different settings "
                       f"(frames_in_flight={self.frames_in_flight}, max_sets={self.max_sets}, flags={self.flags}).")
                raise RuntimeError(msg)
            return

        self.frames_in_flight = frames_in_flight
        self.max_sets = max_sets
        self.flags = flags
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
            flags=flags,
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
        self.frames_in_flight = 0
        self.max_sets = 0
        self.flags = 0
        self.pool_sizes.clear()
        print("Destroyed Descriptor Pool")


class DescriptorSetLayoutCache:
    # Layout is cached by the flag, tuple
    layout_cache: dict[DescriptorSetLayoutKey, VkDescriptorSetLayout]

    def __init__(self, device: VulkanLogicalDevice) -> None:
        self.device = device
        self.layout_cache = {}  # Cache for VkDescriptorSetLayout objects

    def get(
        self,
        layout_key: DescriptorSetLayoutKey,
        layout_bindings: Sequence[VkDescriptorSetLayoutBinding],
    ) -> VkDescriptorSetLayout:
        """Retrieve a cached VkDescriptorSetLayout or create a new one if not cached.

        flags = VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT - Possible addition later.
        """
        if layout_key in self.layout_cache:
            return self.layout_cache[layout_key]

        layout_pnext = None
        binding_flags_info = None
        binding_flags = layout_key.binding_flags
        if binding_flags:
            binding_flags_arr = c_array_list(binding_flags, VkDescriptorBindingFlags)
            binding_flags_info = VkDescriptorSetLayoutBindingFlagsCreateInfo(
                sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO,
                pNext=None,
                bindingCount=len(binding_flags),
                pBindingFlags=binding_flags_arr,
            )
            layout_pnext = ctypes.cast(ctypes.pointer(binding_flags_info), POINTER(None))

        layout_info = VkDescriptorSetLayoutCreateInfo(
            sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            pNext=layout_pnext,
            flags=layout_key.flags,
            bindingCount=len(layout_bindings),
            pBindings=c_array_list(layout_bindings, VkDescriptorSetLayoutBinding),
        )

        layout = VkDescriptorSetLayout()
        self.device.vkCreateDescriptorSetLayout(self.device.vk_device, byref(layout_info), None, byref(layout))

        self.layout_cache[layout_key] = layout
        return layout

    def delete(self) -> None:
        """Destroy all cached layouts."""
        for layout in self.layout_cache.values():
            self.device.vkDestroyDescriptorSetLayout(self.device.vk_device, layout, None)
        self.layout_cache.clear()
        self.device = None


class DescriptorManager:
    """Manages Vulkan descriptor layouts, allocation, and descriptor-set caching.

    This manager is frame-aware but does not own frame synchronization. It uses
    the active frame index to select or update the descriptor sets that are safe
    for that frame slot. Completion and CPU-side reuse are handled by Vulkan
    ``FrameSync`` and the shared ``FrameResourceManager``.

    Descriptor layouts and pools are long-lived device resources. Descriptor
    set objects are cached structurally by layout, bound resources, owner, and
    frame index so UBO slice changes or texture changes get distinct descriptor
    writes without forcing the pool itself to be recreated per frame.
    """
    def __init__(self, device: VulkanLogicalDevice):
        self.device = device
        self.layout_cache = DescriptorSetLayoutCache(self.device)
        self.descriptor_pool = DescriptorPool(self.device)
        self.frames_in_flight = 0
        self.descriptor_set_cache: dict[DescriptorSetCacheKey, DescriptorSetObject] = {}
        self._ubo_generation_ids: weakref.WeakKeyDictionary[VulkanUniformBufferObject, int] = weakref.WeakKeyDictionary()
        self._ubo_generation_tokens: weakref.WeakKeyDictionary[VulkanUniformBufferObject, tuple[int, int]] = weakref.WeakKeyDictionary()

        if self._supports_update_after_bind():
            print("(Vulkan) Descriptor pool using UPDATE_AFTER_BIND (descriptor indexing path).")
        else:
            print("(Vulkan) Descriptor pool using legacy descriptor flags.")

        # Todo: Push descriptors VK_KHR_push_descriptor or version 1.1. -
        #  This extension allows descriptors to be written into the command buffer.

        # Todo: Support for VK_EXT_descriptor_indexing or version 1.2 - Adds support for the advanced descriptor
        #  management features, including UPDATE_AFTER_BIND and partially bound descriptors.

    def create_pool(self, frames_in_flight: int, max_sets: int = 10) -> None:
        if self.frames_in_flight:
            if self.frames_in_flight != frames_in_flight:
                msg = ("Descriptor pool already initialized with a different frames_in_flight value: "
                       f"{self.frames_in_flight} != {frames_in_flight}")
                raise RuntimeError(msg)
            return

        self.frames_in_flight = frames_in_flight
        pool_flags = 0
        if self._supports_update_after_bind():
            pool_flags |= VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT
        self.descriptor_pool.create_pool(frames_in_flight, max_sets, flags=pool_flags)

    def get_descriptor_set_layouts(self, program: VulkanShaderProgram) -> DescriptorSetLayouts:
        """Get descriptor set layouts for the given shader program, keyed structurally."""
        bindings_by_set = program.get_layout_bindings_by_set()
        if not bindings_by_set:
            return DescriptorSetLayouts(DescriptorSetLayoutsKey(tuple()), tuple())

        set_indices = sorted(bindings_by_set)
        if set_indices != list(range(set_indices[-1] + 1)):
            missing_sets = sorted(set(range(set_indices[-1] + 1)) - set(set_indices))
            msg = (
                "Descriptor sets must be contiguous and start from set 0. "
                f"Missing set indices: {missing_sets}"
            )
            raise ValueError(msg)

        layout_entries: list[DescriptorSetLayoutEntryKey] = []
        set_layouts: list[VkDescriptorSetLayout] = []

        for set_index in set_indices:
            ordered_bindings = tuple(
                bindings_by_set[set_index][binding]
                for binding in sorted(bindings_by_set[set_index])
            )
            layout_flags, binding_flags = self._descriptor_indexing_layout_flags(ordered_bindings)
            layout_key = DescriptorSetLayoutKey.from_layout_bindings(
                ordered_bindings,
                flags=layout_flags,
                binding_flags=binding_flags,
            )
            layout = self.layout_cache.get(layout_key, ordered_bindings)
            layout_entries.append(DescriptorSetLayoutEntryKey(set_index, layout_key))
            set_layouts.append(layout)

        return DescriptorSetLayouts(
            key=DescriptorSetLayoutsKey(tuple(layout_entries)),
            layouts=tuple(set_layouts),
        )

    def _supports_descriptor_indexing(self) -> bool:
        return False
        return self.device.descriptor_indexing_enabled

    def _supports_partially_bound(self) -> bool:
        return False
        return self.device.descriptor_binding_partially_bound

    def _supports_update_after_bind(self) -> bool:
        return False
        return bool(
            self._supports_descriptor_indexing()
            and self.device.descriptor_binding_uniform_buffer_update_after_bind
            and self.device.descriptor_binding_sampled_image_update_after_bind
        )

    def _descriptor_indexing_layout_flags(
            self, layout_bindings: tuple[VkDescriptorSetLayoutBinding, ...],
    ) -> tuple[int, list[int]]:
        """Build descriptor indexing flags for each binding if the device supports it."""
        if not self._supports_descriptor_indexing():
            return 0, []

        binding_flags: list[int] = []
        layout_flags = 0
        use_update_after_bind = self._supports_update_after_bind()
        use_partially_bound = self._supports_partially_bound()

        for binding in layout_bindings:
            flags = 0
            if use_update_after_bind:
                if binding.descriptorType == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
                    flags |= VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT
                elif binding.descriptorType == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
                    flags |= VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT

            if use_partially_bound:
                flags |= VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT

            if flags & VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT:
                layout_flags |= VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT

            binding_flags.append(flags)

        return layout_flags, binding_flags

    def get_descriptor_sets(self,
                            set_layouts: DescriptorSetLayouts,
                            resource_states: Sequence[DescriptorResourceState],
                            uniform_bindings: Sequence[tuple[VulkanUniformBufferObject, int, int]] | None = None,
                            resources: Sequence[DescriptorResourceKey] | None = None,
                            frame_index: int = 0,
                            owner: object | None = None) -> DescriptorSetObject:
        """Allocate the descriptor sets.

        Will return a descriptor set for each frame in flight.
        """
        active_frame = int(frame_index) % max(1, self.frames_in_flight)
        normalized_uniform_bindings = tuple(uniform_bindings or ())
        normalized_resources = tuple(resources or self.build_resource_keys(
            resource_states,
            normalized_uniform_bindings,
            active_frame,
        ))
        owner_key = id(owner) if owner is not None else None
        key = DescriptorSetCacheKey(
            set_layouts.key,
            normalized_resources,
            active_frame,
            owner_key,
        )
        if key in self.descriptor_set_cache:
            dset = self.descriptor_set_cache[key]
            dset.update_bound_resources(
                resource_states=resource_states,
                uniform_bindings=normalized_uniform_bindings,
                frame_idx=active_frame,
            )
            dset.validate_resources(normalized_resources)
            return dset

        if not set_layouts.layouts:
            msg = "Cannot allocate descriptor sets for an empty descriptor set layout list."
            raise RuntimeError(msg)

        self.descriptor_set_cache[key] = dset = DescriptorSetObject(
            self.device,
            self.descriptor_pool.allocate_descriptor_sets(list(set_layouts.layouts)),
            set_layouts_key=set_layouts.key,
        )
        dset.update_bound_resources(
            resource_states=resource_states,
            uniform_bindings=normalized_uniform_bindings,
            frame_idx=active_frame,
        )
        dset.validate_resources(normalized_resources)
        return dset

    def build_resource_keys(
        self,
        resource_states: Sequence[DescriptorResourceState],
        uniform_bindings: Sequence[tuple[VulkanUniformBufferObject, int, int]],
        frame_index: int,
    ) -> tuple[DescriptorResourceKey, ...]:
        keys: list[DescriptorResourceKey] = []
        keys.extend(
            self._get_uniform_binding_key(ubo, binding, set_index, frame_index)
            for ubo, binding, set_index in uniform_bindings
        )
        keys.extend(self._get_state_resource_key(state) for state in resource_states)
        return tuple(sorted(keys, key=self._resource_sort_key))

    @staticmethod
    def _resource_sort_key(resource: DescriptorResourceKey) -> tuple[int, int, int]:
        return resource.set_index, resource.binding, resource.descriptor_type

    @staticmethod
    def _ubo_descriptor_token(ubo: VulkanUniformBufferObject, buffer_id: int) -> tuple[int, int]:
        buffer_wrapper = getattr(ubo, "buffer", None)
        resource = getattr(buffer_wrapper, "buffer", None) if buffer_wrapper is not None else None
        return id(resource), int(buffer_id)

    def _get_ubo_generation_id(self, ubo: VulkanUniformBufferObject, buffer_id: int) -> int:
        token = self._ubo_descriptor_token(ubo, buffer_id)
        generation = self._ubo_generation_ids.get(ubo)
        if generation is None:
            generation = 1
        elif self._ubo_generation_tokens.get(ubo) != token:
            generation += 1
        self._ubo_generation_ids[ubo] = generation
        self._ubo_generation_tokens[ubo] = token
        return generation

    def _get_uniform_binding_key(
        self,
        ubo: VulkanUniformBufferObject,
        binding: int,
        set_index: int,
        frame_index: int,
    ) -> UniformBindingKey:
        buffer_id = int(ubo.id)
        frame_slice = ubo.get_frame_slice(frame_index)
        generation_id = self._get_ubo_generation_id(ubo, buffer_id)
        return UniformBindingKey(
            set_index=int(set_index),
            binding=int(binding),
            descriptor_type=VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            descriptor_generation_id=int(generation_id),
            buffer_id=buffer_id,
            offset=int(frame_slice.offset),
            range_size=int(frame_slice.size),
        )

    @staticmethod
    def _get_state_resource_key(state: DescriptorResourceState) -> DescriptorResourceKey:
        texture = getattr(state, "texture", None)
        binding = getattr(state, "binding", None)
        set_id = getattr(state, "set_id", None)
        if texture is None or binding is None or set_id is None:
            msg = f"Unsupported descriptor resource state for caching: {type(state).__name__}"
            raise RuntimeError(msg)
        return DescriptorManager._get_sampler_key(texture, binding, set_id)

    @staticmethod
    def _get_sampler_key(texture: VulkanTexture, binding: int, set_index: int) -> SamplerKey:
        owner = getattr(texture, "owner", texture)
        sampler = getattr(texture, "sampler", None) or getattr(owner, "sampler", None)
        image_view = getattr(texture, "image_view", None) or getattr(owner, "image_view", None)
        vk_sampler = getattr(sampler, "vk_sampler", None)
        vk_image_view = getattr(image_view, "vk_imageview", None)
        return SamplerKey(
            set_index=int(set_index),
            binding=int(binding),
            descriptor_type=VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            descriptor_generation_id=int(getattr(owner, "descriptor_generation_id", 0) or 0),
            texture_id=int(getattr(owner, "id", 0) or 0),
            sampler_id=int(getattr(vk_sampler, "value", 0) or 0),
            image_view_id=int(getattr(vk_image_view, "value", 0) or 0),
        )

    def get_shared_descriptor_sets(self, set_layouts: DescriptorSetLayouts) -> DescriptorSetObject:
        """Allocate a shared descriptor set.

        A shared descriptor set is meant to be used for frequently updated descriptors. Instead of resources being set
        to a specific descriptor, the descriptor set updates the pointer to a different resource every frame or
        very frequently. This can lower the need for lots of descriptor sets.
        """
        if not set_layouts.layouts:
            msg = "Cannot allocate descriptor sets for an empty descriptor set layout list."
            raise RuntimeError(msg)
        return DescriptorSetObject(
            self.device,
            self.descriptor_pool.allocate_descriptor_sets(list(set_layouts.layouts)),
            set_layouts_key=set_layouts.key,
        )

    def delete(self) -> None:
        self.layout_cache.delete()
        self.descriptor_pool.delete()
        self.frames_in_flight = 0
        self.descriptor_set_cache.clear()


@dataclass(slots=True)
class DescriptorFrameState:
    descriptor_sets: tuple[VkDescriptorSet, ...]
    descriptor_array: Any
    uploaded_version: int = -1


@dataclass(slots=True)
class DescriptorSetBindingGroup:
    """Used to group descriptor sets together."""
    device: VulkanLogicalDevice
    descriptor_set_count: int
    descriptor_set_array: Any
    first_set: int = 0

    @classmethod
    def from_descriptor_sets(
        cls,
        device: VulkanLogicalDevice,
        descriptor_sets: Sequence[VkDescriptorSet],
        first_set: int = 0,
    ) -> DescriptorSetBindingGroup:
        return cls(
            device=device,
            descriptor_set_count=len(descriptor_sets),
            descriptor_set_array=c_array_list(list(descriptor_sets), VkDescriptorSet),
            first_set=first_set,
        )

    def bind(self, command_buffer: VkCommandBuffer, pipeline_layout: VkPipelineLayout) -> None:
        self.device.vkCmdBindDescriptorSets(
            command_buffer,
            VK_PIPELINE_BIND_POINT_GRAPHICS,
            pipeline_layout,
            self.first_set,
            self.descriptor_set_count,
            self.descriptor_set_array,
            0,
            None,
        )


class DescriptorSetObject(FrameLocalResource[DescriptorFrameState]):
    """Frame-local wrapper around one logical descriptor-set binding group.

    A ``DescriptorSetObject`` contains one tuple of ``VkDescriptorSet`` handles
    per frame in flight. Callers treat it as one logical binding group and ask
    for the frame-specific tuple during command recording.

    This class uses ``FrameLocalResource`` for per-frame upload/version state.
    It does not determine when a submitted frame is complete; that remains the
    responsibility of Vulkan ``FrameSync`` and ``FrameResourceManager``.
    """
    set_count: int

    def __init__(
        self,
        device: VulkanLogicalDevice,
        descriptor_sets_frames: list[list[VkDescriptorSet]],
        set_layouts_key: DescriptorSetLayoutsKey | None = None,
    ):
        self.descriptor_sets_frames = descriptor_sets_frames
        FrameLocalResource.__init__(self, len(descriptor_sets_frames))
        self.device = device
        self.set_layouts_key = set_layouts_key
        self.set_count = len(self.frame_states[0].descriptor_sets)
        self.bindings: list[Any] = []
        self._resource_states: tuple[DescriptorResourceState, ...] = ()
        self._uniform_bindings: tuple[tuple[VulkanUniformBufferObject, int, int], ...] = ()
        #self.name_to_binding = {}  # Mapping of resource name to (binding, type)

    def _create_frame_states(self) -> list[DescriptorFrameState]:
        return [
            DescriptorFrameState(
                descriptor_sets=tuple(descriptor_sets),
                descriptor_array=c_array_list(descriptor_sets, VkDescriptorSet),
                uploaded_version=-1,
            )
            for descriptor_sets in self.descriptor_sets_frames
        ]

    def get_frame_descriptor_sets(self, frame_idx: int) -> tuple[VkDescriptorSet, ...]:
        resolved_frame = self.set_current_frame(frame_idx)
        self.upload_if_needed(resolved_frame)
        frame_state = self.get_frame_state(resolved_frame)
        return frame_state.descriptor_sets

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

    def _resolve_frame_idx(self, frame_idx: int) -> int:
        return self._normalize_frame_index(frame_idx)

    def update_bound_resources(
        self,
        resource_states: Sequence[DescriptorResourceState],
        uniform_bindings: Sequence[tuple[VulkanUniformBufferObject, int, int]],
        frame_idx: int,
    ) -> None:
        resolved_frame = self._resolve_frame_idx(frame_idx)
        normalized_resource_states = tuple(resource_states)
        normalized_uniform_bindings = tuple(uniform_bindings)

        if (
            normalized_resource_states != self._resource_states
            or normalized_uniform_bindings != self._uniform_bindings
        ):
            self._resource_states = normalized_resource_states
            self._uniform_bindings = normalized_uniform_bindings
            self.mark_data_updated()

        self.upload_if_needed(resolved_frame)

    def _upload_frame(self, frame_index: int) -> bool:
        for ubo, binding, set_index in self._uniform_bindings:
            self.update_ubo_binding(ubo, binding=binding, desc_set=set_index, frame_idx=frame_index)

        for state in self._resource_states:
            state.write_descriptor(self, frame_idx=frame_index)

        return True

    def validate_resources(self, resources: Sequence[DescriptorResourceKey]) -> None:
        if self.set_layouts_key is None:
            return

        bound_lookup = {(res.set_index, res.binding): res for res in resources}
        for set_entry in self.set_layouts_key.set_layouts:
            set_index = int(set_entry.set_index)
            for layout_binding in set_entry.layout_key.bindings:
                binding_index = int(layout_binding.binding)
                resource = bound_lookup.get((set_index, binding_index))
                if resource is None:
                    msg = (
                        "Descriptor validation failed: missing resource for "
                        f"set={set_index}, binding={binding_index}."
                    )
                    raise RuntimeError(msg)

                if int(resource.descriptor_type) != int(layout_binding.descriptor_type):
                    msg = (
                        "Descriptor validation failed: descriptor type mismatch for "
                        f"set={set_index}, binding={binding_index}; "
                        f"expected={int(layout_binding.descriptor_type)} "
                        f"actual={int(resource.descriptor_type)}."
                    )
                    raise RuntimeError(msg)

                if isinstance(resource, UniformBindingKey):
                    if resource.buffer_id == 0 or resource.range_size <= 0:
                        msg = (
                            "Descriptor validation failed: invalid uniform buffer resource for "
                            f"set={set_index}, binding={binding_index}."
                        )
                        raise RuntimeError(msg)
                elif isinstance(resource, SamplerKey):
                    if resource.texture_id == 0 or resource.sampler_id == 0 or resource.image_view_id == 0:
                        msg = (
                            "Descriptor validation failed: invalid sampled texture resource for "
                            f"set={set_index}, binding={binding_index}."
                        )
                        raise RuntimeError(msg)

    def update_sampled_texture_binding(
        self,
        texture: VulkanTexture,
        binding: int,
        desc_set: int = 0,
        frame_idx: int = 0,
    ) -> None:
        resolved_frame = self._resolve_frame_idx(frame_idx)
        image_info = VkDescriptorImageInfo(
            sampler=texture.sampler.vk_sampler,
            imageView=texture.image_view.vk_imageview,
            imageLayout=VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        )
        image_info_array = (VkDescriptorImageInfo * 1)(image_info)

        write = VkWriteDescriptorSet(
            sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            dstSet=self.get_frame_state(resolved_frame).descriptor_sets[desc_set],
            dstBinding=binding,
            dstArrayElement=0,
            descriptorCount=1,
            descriptorType=VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            pImageInfo=image_info_array,
        )

        write_array = (VkWriteDescriptorSet * 1)(write)
        self.device.vkUpdateDescriptorSets(self.device.vk_device, 1, write_array, 0, None)

    # Bind a uniform buffer to the descriptor set
    def update_ubo_binding(
        self,
        ubo: VulkanUniformBufferObject,
        binding: int = 0,
        desc_set: int = 0,
        frame_idx: int = 0,
    ) -> None:
        resolved_frame = self._resolve_frame_idx(frame_idx)
        ubo.upload_if_needed(resolved_frame)
        ubo._ensure_buffer_created()  # noqa: SLF001
        assert ubo.buffer is not None
        assert ubo.buffer.buffer is not None
        frame_slice = ubo.get_frame_slice(resolved_frame)
        dst_set = self.get_frame_state(resolved_frame).descriptor_sets[desc_set]
        buffer_info = VkDescriptorBufferInfo(
            buffer=ubo.buffer.buffer.vk_buffer,
            offset=frame_slice.offset,
            range=frame_slice.size,
        )
        buff_info_array = (VkDescriptorBufferInfo * 1)(buffer_info)

        write = VkWriteDescriptorSet(
            sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            dstSet=dst_set,
            dstBinding=binding,
            dstArrayElement=0,
            descriptorCount=1,
            descriptorType=VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            pBufferInfo=buff_info_array,
        )

        write_array = (VkWriteDescriptorSet * 1)(write)
        self.device.vkUpdateDescriptorSets(self.device.vk_device, 1, write_array, 0, None)
        ubo.set_frame_descriptor_set(dst_set, resolved_frame)
