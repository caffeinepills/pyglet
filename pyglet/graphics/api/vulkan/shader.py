from __future__ import annotations

import ctypes
import re
import weakref
import warnings
from dataclasses import dataclass
from typing import Sequence, TYPE_CHECKING, Any, BinaryIO, ClassVar, overload
from ctypes import byref, Structure

import pyglet
from pyglet.graphics.api.vulkan.buffer import UniformBuffer, VulkanUniformBufferObject
from pyglet.graphics.api.vulkan.spirv import COMPILATION_AVAILABLE, compile_shader, \
    INSPECTION_AVAILABLE, get_spirv_inspection_json, get_spirv_as_glsl, validate_spirv
from pyglet.libs.shared.vulkan_lib.vulkan_core import (
    VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
    VK_FORMAT_R16G16B16A16_SINT,
    VK_FORMAT_R16G16B16A16_UINT,
    VK_FORMAT_R16G16B16A16_UNORM,
    VK_FORMAT_R16G16B16_SINT,
    VK_FORMAT_R16G16B16_UINT,
    VK_FORMAT_R16G16B16_UNORM,
    VK_FORMAT_R16G16_SINT,
    VK_FORMAT_R16G16_UINT,
    VK_FORMAT_R16G16_UNORM,
    VK_FORMAT_R16_SINT,
    VK_FORMAT_R16_UINT,
    VK_FORMAT_R16_UNORM,
    VK_FORMAT_R32G32B32A32_SFLOAT,
    VK_FORMAT_R32G32B32A32_SINT,
    VK_FORMAT_R32G32B32A32_UINT,
    VK_FORMAT_R32G32B32_SFLOAT,
    VK_FORMAT_R32G32B32_SINT,
    VK_FORMAT_R32G32B32_UINT,
    VK_FORMAT_R32G32_SFLOAT,
    VK_FORMAT_R32G32_SINT,
    VK_FORMAT_R32G32_UINT,
    VK_FORMAT_R32_SFLOAT,
    VK_FORMAT_R32_SINT,
    VK_FORMAT_R32_UINT,
    VK_FORMAT_R8G8B8A8_SINT,
    VK_FORMAT_R8G8B8A8_SNORM,
    VK_FORMAT_R8G8B8A8_UINT,
    VK_FORMAT_R8G8B8A8_UNORM,
    VK_FORMAT_R8G8B8_SINT,
    VK_FORMAT_R8G8B8_SNORM,
    VK_FORMAT_R8G8B8_UINT,
    VK_FORMAT_R8G8B8_UNORM,
    VK_FORMAT_R8G8_SINT,
    VK_FORMAT_R8G8_SNORM,
    VK_FORMAT_R8G8_UINT,
    VK_FORMAT_R8G8_UNORM,
    VK_FORMAT_R8_SINT,
    VK_FORMAT_R8_SNORM,
    VK_FORMAT_R8_UINT,
    VK_FORMAT_R8_UNORM,
    VK_SHADER_STAGE_COMPUTE_BIT,
    VK_SHADER_STAGE_FRAGMENT_BIT,
    VK_SHADER_STAGE_GEOMETRY_BIT,
    VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT,
    VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT,
    VK_SHADER_STAGE_VERTEX_BIT,
    VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
    VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
    VkDescriptorSetLayoutBinding,
    VkPipelineShaderStageCreateInfo,
    VkShaderModuleCreateInfo, VkShaderModule, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
    VkPushConstantRange)
from pyglet.graphics.shader import Shader, ShaderProgram, ShaderType, UniformBlockDesc, UniformBlock, \
    Attribute, ShaderException, SampledTextureBinding, PushConstants, ShaderSource, \
    GraphicsAttribute, AttributeView

if TYPE_CHECKING:
    from pyglet.graphics.vertexdomain import InstanceIndexedVertexList, InstanceVertexList, VertexList, IndexedVertexList
    from pyglet.enums import GeometryMode
    from pyglet.customtypes import DataTypes, CTypesPointer
    from pyglet.graphics import Batch, Group
    from pyglet.graphics.api.vulkan.devices import VulkanLogicalDevice

_debug_api_shaders = pyglet.options.debug_api_shaders


_shader_type_to_stage: dict[ShaderType, int] = {
    'vertex': VK_SHADER_STAGE_VERTEX_BIT,
    'fragment': VK_SHADER_STAGE_FRAGMENT_BIT,
    'geometry': VK_SHADER_STAGE_GEOMETRY_BIT,
    'compute': VK_SHADER_STAGE_COMPUTE_BIT,
    'tesscontrol': VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT,
    'tessevaluation': VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT,
}

# Here temporarily to test with and without external libraries.
#INSPECTION_AVAILABLE = False
#COMPILATION_AVAILABLE = False

VULKAN_FORMAT_MAP: dict[tuple[int, DataTypes, bool], int] = {
    # Float formats (non-normalized only)
    (1, 'f', False): VK_FORMAT_R32_SFLOAT,
    (2, 'f', False): VK_FORMAT_R32G32_SFLOAT,
    (3, 'f', False): VK_FORMAT_R32G32B32_SFLOAT,
    (4, 'f', False): VK_FORMAT_R32G32B32A32_SFLOAT,

    # Signed int formats (non-normalized only)
    (1, 'i', False): VK_FORMAT_R32_SINT,
    (2, 'i', False): VK_FORMAT_R32G32_SINT,
    (3, 'i', False): VK_FORMAT_R32G32B32_SINT,
    (4, 'i', False): VK_FORMAT_R32G32B32A32_SINT,

    # Unsigned int formats (non-normalized only)
    (1, 'I', False): VK_FORMAT_R32_UINT,
    (2, 'I', False): VK_FORMAT_R32G32_UINT,
    (3, 'I', False): VK_FORMAT_R32G32B32_UINT,
    (4, 'I', False): VK_FORMAT_R32G32B32A32_UINT,

    # Signed short formats
    (1, 'h', False): VK_FORMAT_R16_SINT,
    (2, 'h', False): VK_FORMAT_R16G16_SINT,
    (3, 'h', False): VK_FORMAT_R16G16B16_SINT,
    (4, 'h', False): VK_FORMAT_R16G16B16A16_SINT,

    # Unsigned short formats (with and without normalization)
    (1, 'H', False): VK_FORMAT_R16_UINT,
    (2, 'H', False): VK_FORMAT_R16G16_UINT,
    (3, 'H', False): VK_FORMAT_R16G16B16_UINT,
    (4, 'H', False): VK_FORMAT_R16G16B16A16_UINT,

    (1, 'H', True): VK_FORMAT_R16_UNORM,
    (2, 'H', True): VK_FORMAT_R16G16_UNORM,
    (3, 'H', True): VK_FORMAT_R16G16B16_UNORM,
    (4, 'H', True): VK_FORMAT_R16G16B16A16_UNORM,

    # Signed byte formats (with and without normalization)
    (1, 'b', False): VK_FORMAT_R8_SINT,
    (2, 'b', False): VK_FORMAT_R8G8_SINT,
    (3, 'b', False): VK_FORMAT_R8G8B8_SINT,
    (4, 'b', False): VK_FORMAT_R8G8B8A8_SINT,

    (1, 'b', True): VK_FORMAT_R8_SNORM,
    (2, 'b', True): VK_FORMAT_R8G8_SNORM,
    (3, 'b', True): VK_FORMAT_R8G8B8_SNORM,
    (4, 'b', True): VK_FORMAT_R8G8B8A8_SNORM,

    # Unsigned byte formats (with and without normalization)
    (1, 'B', False): VK_FORMAT_R8_UINT,
    (2, 'B', False): VK_FORMAT_R8G8_UINT,
    (3, 'B', False): VK_FORMAT_R8G8B8_UINT,
    (4, 'B', False): VK_FORMAT_R8G8B8A8_UINT,

    (1, 'B', True): VK_FORMAT_R8_UNORM,
    (2, 'B', True): VK_FORMAT_R8G8_UNORM,
    (3, 'B', True): VK_FORMAT_R8G8B8_UNORM,
    (4, 'B', True): VK_FORMAT_R8G8B8A8_UNORM,
}

class VulkanShaderSource(ShaderSource):

    def validate(self) -> str:
        return ""



def get_vulkan_format(component_count: int, data_type: DataTypes, normalized: bool) -> int | None:
    """The appropriate Vulkan format for the given component element_count, data type, and normalization.

    There are two normalization types with Vulkan, but just one is added for parity with OpenGL.

    Args:
        component_count:
            Number of components (1 for scalar, 2 for vec2, etc.)
        data_type:
            Data type ('f' for float, 'i' for int, etc.)
        normalized:
            Whether the data is normalized to between 0 and 1.

    Returns:
        Corresponding Vulkan format integer.
    """
    key = (component_count, data_type, normalized)
    return VULKAN_FORMAT_MAP.get(key)


def _is_glsl_content(content: str) -> bool:
    """Simple implementation to check if a string is GLSL."""
    glsl_keywords = {'#version', 'layout', 'in', 'out'}
    return any(keyword in content for keyword in glsl_keywords)

class VulkanShader(Shader):
    @classmethod
    def supported_shaders(cls: type[Shader]) -> tuple[ShaderType, ...]:
        return 'vertex', 'fragment', 'compute', 'geometry', 'tesscontrol', 'tessevaluation'

    @staticmethod
    def get_string_class() -> type[ShaderSource]:
        pass

    compiled_data: bytes

    def __init__(self, source: str | BinaryIO, shader_type: ShaderType, entry_point: str="main") -> None:
        super().__init__(source, shader_type)
        self.device = None
        self.entry_point = entry_point
        self._shader_module = None
        self.info = None

        # If it's a file like object, let's get the data.
        if hasattr(source, 'read'):
            source = source.read()

        # If it's a source string, see if it's GLSL code?
        if isinstance(source, str) and _is_glsl_content(source):
            if COMPILATION_AVAILABLE:
                source = compile_shader(source, shader_type, entry_point=entry_point)
            else:
                msg = ("Compiling shaders is not available. Ensure shaderc is available in a system or environment "
                       "path.\nCheck pyglet.options.debug_lib for more information.")
                raise ShaderException(msg)

        self.compiled_data = source

        # Final check to make sure it is SPIR-V data.
        if not validate_spirv(self.compiled_data):
            raise ShaderException(
                "Source should be a string of GLSL code, SPIRV binary data, or a file-like object that returns either.")

        array_size = len(self.compiled_data) // 4
        uint32_array = (ctypes.c_uint32 * array_size).from_buffer_copy(self.compiled_data)

        self.create_info = VkShaderModuleCreateInfo(
            sType=VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            codeSize=len(self.compiled_data),
            pCode=ctypes.cast(uint32_array, ctypes.POINTER(ctypes.c_uint32)),
        )

    def get_source(self) -> str | None:
        if self.compiled_data and INSPECTION_AVAILABLE:
            return get_spirv_as_glsl(self.compiled_data)

        return None

    def attach(self, device: VulkanLogicalDevice) -> None:
        self.device = device
        """Attach a shader to a device."""

        self._shader_module = VkShaderModule()
        device.vkCreateShaderModule(device.vk_device, byref(self.create_info), None, byref(self._shader_module))

    def get_info(self) -> VkPipelineShaderStageCreateInfo:
        """Stage info needs to be generated for each pipeline being used.

        Need separate instances for each pipeline, should not be cached.
        """
        if not self.info:
            self.info = VkPipelineShaderStageCreateInfo(
                sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                stage=_shader_type_to_stage[self.type],
                module=self._shader_module,
                pName=self.entry_point.encode('utf-8'),
            )
        return self.info

    def __del__(self) -> None:
        self.delete()

    def delete(self) -> None:
        if self._shader_module:
            self.device.vkDestroyShaderModule(self.device.vk_device, self._shader_module, None)
            self._shader_module = None
            self.device = None

    def __repr__(self) -> str:
        return f"Shader(stage={self.type}, module={hex(self._shader_module.value) if self._shader_module else None})"

class VulkanAttribute(Attribute):
    vk_format: int

    def __init__(self, attribute: Attribute) -> None:
        super().__init__(
            attribute.fmt.name,
            attribute.location,
            attribute.fmt.components,
            attribute.fmt.data_type,
            attribute.fmt.normalized,
            attribute.fmt.divisor,
        )
        vk_format = get_vulkan_format(self.fmt.components, self.fmt.data_type, self.fmt.normalized)
        assert vk_format is not None, (
            "The attribute information was invalid. "
            f"({self.fmt.components}, {self.fmt.data_type}, {self.fmt.normalized})"
        )
        self.vk_format = vk_format

_reflection_to_type = {
    'vertex' : 'vert',
    'fragment': 'frag',
    'geometry': 'geom',
    'compute': 'comp',
    'tesscontrol': 'tesc',
    'tessevaluation': 'tese',
}

@dataclass
class _PushConstant:
    struct: ctypes.Structure
    constant_range: VkPushConstantRange
    size: int


class VulkanShaderProgram(ShaderProgram):
    _name_map: dict[str, Structure]
    device: VulkanLogicalDevice | None
    push_constants: list[_PushConstant]
    shaders: tuple[VulkanShader, ...]
    ubo: dict[str, UniformBlock]
    _live_programs: ClassVar[weakref.WeakSet] = weakref.WeakSet()

    def __init__(self, *shaders: VulkanShader) -> None:
        super().__init__(*shaders)
        self.device = None
        self.linked = False
        self.shaders = shaders
        self.ubo = {}
        self.push_constants = []
        self._name_map = {}
        self._live_programs.add(self)

        if INSPECTION_AVAILABLE:
            for shader in self.shaders:
                reflection_data = get_spirv_inspection_json(shader.compiled_data)
                self.load_spirv_reflection(reflection_data, shader.type)

    def load_spirv_reflection(self, reflection_data: dict, shader_type: str) -> None:
        print(f"Loading SPIR-V reflection for {shader_type.upper()}", reflection_data)
        attributes = []
        sampled_textures = []
        ubos = []
        push_constants = []

        # Handle entry points
        for entry_point in reflection_data.get("entryPoints", []):
            for shader in self.shaders:
                if _reflection_to_type[shader.type] == entry_point['mode']:
                    shader.entry_point = entry_point['name']
                    break

        # Process inputs (only for vertex shaders)
        if shader_type == "vertex":
            for input_data in reflection_data.get('inputs', []):
                data_type: DataTypes = input_data['type']
                components = int(data_type[-1]) if data_type.startswith('vec') else 1
                data_type = "f" if "float" in data_type or "vec" in data_type else "B"
                normalize = data_type == "B"  # Assume normalization for "B"... for now.
                attributes.append(Attribute(
                    name=input_data['name'],
                    location=input_data['location'],
                    components=components,
                    data_type=data_type,
                    normalize=normalize,
                ))

        # Extract uniform buffer objects (UBOs)
        for ubo_data in reflection_data.get('ubos', []):
            ubo_type_name = ubo_data['type']
            ubo_type = reflection_data['types'][ubo_type_name]
            uniforms = []
            for member in ubo_type['members']:
                uniforms.append((member['type'], member['name']))
            ubo_class = type(ubo_data['name'], (UniformBlockDesc,), {
                "stages": (shader_type,),  # Use shader_type to tag stages dynamically
                "set_num": ubo_data['set'],
                "bind_num": ubo_data['binding'],
                "uniforms": tuple(uniforms),
            })
            ubos.append(ubo_class)

        # Extract combined sampled textures
        for sampler_data in reflection_data.get('textures', []):
            sampled_textures.append(SampledTextureBinding(
                name=sampler_data['name'],
                desc_set=sampler_data['set'],
                binding=sampler_data['binding'],
                stages=(shader_type,),
            ))

        # Extract push constants:
        for pc_data in reflection_data.get('push_constants', []):
            if pc_data['push_constant'] is True:
                pc_type_name = pc_data['type']
                pc_type = reflection_data['types'][pc_type_name]
                pc_uniforms = []
                for member in pc_type['members']:
                    pc_uniforms.append((member['name'], member['type'] ))
                pc_class = PushConstants((shader_type,), tuple(pc_uniforms))
                push_constants.append(pc_class)

        # Apply attributes, UBOs, PCs, and sampled textures
        self.set_attributes(*attributes)
        self.set_uniform_blocks(*ubos)
        self.set_sampled_textures(*sampled_textures)
        self.set_push_constants(*push_constants)

    def set_shader_uniforms(self, push_constants: PushConstants):
        self.set_push_constants(push_constants)

    def set_sampled_textures(self, *sampled_textures: SampledTextureBinding) -> None:
        """Register sampled textures via base storage, with Vulkan stage merge on duplicates."""
        for sampled_texture in sampled_textures:
            existing = self._sampled_textures.get(sampled_texture.name)
            if existing is None:
                super().set_sampled_textures(sampled_texture)
                continue

            if (
                existing.desc_set != sampled_texture.desc_set
                or existing.binding != sampled_texture.binding
                or existing.count != sampled_texture.count
            ):
                msg = (
                    f"Sampled texture '{sampled_texture.name}' was declared with incompatible binding metadata. "
                    "Ensure descriptor set, binding, and count match across shader stages."
                )
                raise ShaderException(msg)

            merged_stages = tuple(dict.fromkeys((*existing.stages, *sampled_texture.stages)))
            self._sampled_textures[sampled_texture.name] = SampledTextureBinding(
                name=existing.name,
                desc_set=existing.desc_set,
                binding=existing.binding,
                count=existing.count,
                stages=merged_stages,
            )

    def set_push_constants(self, *push_constants: PushConstants):
        # We will map push constant names to the right structure. Ensure no overlap of variable names.
        for push_constant in push_constants:
            constants = [(uniform_type, name) for name, uniform_type in push_constant.constants]
            structure = create_structure(f'PushConstants{len(self.push_constants)}', constants)()

            size = ctypes.sizeof(structure)
            if size > 128:
                raise ShaderException("Push constants cannot exceed 128 bytes, use a Uniform Buffer instead.")

            constant_range = VkPushConstantRange(
                stageFlags=stages_to_bits(push_constant.stages),
                size=size,
            )

            for name in [name for name, _ in push_constant.constants]:
                if name in self._name_map:
                    raise Exception(f"Duplicate push constant: '{name}'. Ensure all push constant names are unique between all shaders in this program.")
                self._name_map[name] = structure

            self.push_constants.append(_PushConstant(
                structure,
                constant_range,
                size,
            ))

        if sum(pc.size for pc in self.push_constants) > 128:
            raise ShaderException("Push constants cannot exceed a total of 128 bytes, use a Uniform Buffer instead.")

    def set_attribute_format(self, name: str, data_type: DataTypes, normalize: bool):
        """Must be called before any Attribute Buffers are created."""
        self._attributes[name].set_data_type(data_type, normalize)

    def set_uniform_blocks(self, *ubos: UniformBlockDesc):
        """Generate a UniformBlock based on the UBODesc.

        We could probably adjust this to create it based on the Shader string eventually...
        """
        for ubo in ubos:
            name = ubo.__name__
            existing_block = self._uniform_blocks.get(name)
            if isinstance(existing_block, VulkanUniformBlock):
                if (
                    existing_block.index == ubo.set_num
                    and existing_block.binding == ubo.bind_num
                    and existing_block.uniforms == ubo.uniforms
                ):
                    merged_stages = tuple(dict.fromkeys((*existing_block._stages, *ubo.stages)))
                    existing_block._stages = merged_stages
                    existing_block._stage_bit = stages_to_bits(merged_stages)
                    existing_block.layout_binding.stageFlags = existing_block._stage_bit
                    continue

                existing_ubo = self.ubo.pop(name, None)
                if existing_ubo:
                    existing_ubo.delete()

            size = 0
            self._uniform_blocks[name] = self.get_uniform_block_cls()(
                self, name, ubo.set_num, size, ubo.bind_num, ubo.uniforms, len(ubo.uniforms),
                ubo.stages,
            )
            self.ubo[name] = self._uniform_blocks[name].create_ubo()

        #print("Loading", self._id, [get_spirv_as_glsl(shader.compiled_data) for shader in self.shaders])

    def _get_sampler_layout_bindings(self) -> dict[str, tuple[int, VkDescriptorSetLayoutBinding]]:
        return {
            name: (
                sampler.desc_set,
                VkDescriptorSetLayoutBinding(
                    binding=sampler.binding,
                    descriptorType=VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    descriptorCount=sampler.count,
                    stageFlags=stages_to_bits(sampler.stages),
                ),
            )
            for name, sampler in self.sampled_textures.items()
        }

    def _get_ubo_layout_bindings(self) -> dict[str, tuple[int, VkDescriptorSetLayoutBinding]]:
        return {
            name: (block.index, block.layout_binding)
            for name, block in self.uniform_blocks.items()
        }

    @staticmethod
    def _merge_layout_binding(
        bindings_by_set: dict[int, dict[int, VkDescriptorSetLayoutBinding]],
        set_index: int,
        resource_name: str,
        layout_binding: VkDescriptorSetLayoutBinding,
    ) -> None:
        set_bindings = bindings_by_set.setdefault(set_index, {})
        binding_index = int(layout_binding.binding)
        existing = set_bindings.get(binding_index)
        if existing is None:
            set_bindings[binding_index] = VkDescriptorSetLayoutBinding(
                binding=layout_binding.binding,
                descriptorType=layout_binding.descriptorType,
                descriptorCount=layout_binding.descriptorCount,
                stageFlags=layout_binding.stageFlags,
                pImmutableSamplers=layout_binding.pImmutableSamplers,
            )
            return

        if (
            int(existing.descriptorType) != int(layout_binding.descriptorType)
            or int(existing.descriptorCount) != int(layout_binding.descriptorCount)
        ):
            msg = (
                f"Descriptor binding collision in set {set_index}, binding {binding_index}. "
                f"Resource '{resource_name}' does not match existing descriptor type/count."
            )
            raise ShaderException(msg)

        existing.stageFlags |= layout_binding.stageFlags

    def get_layout_bindings_by_set(self) -> dict[int, dict[int, VkDescriptorSetLayoutBinding]]:
        """Descriptor layout bindings grouped by descriptor set and binding index."""
        resources: dict[str, tuple[int, VkDescriptorSetLayoutBinding]] = {}
        resources.update(self._get_ubo_layout_bindings())
        resources.update(self._get_sampler_layout_bindings())

        by_set: dict[int, dict[int, VkDescriptorSetLayoutBinding]] = {}
        for resource_name, (set_index, layout_binding) in resources.items():
            self._merge_layout_binding(by_set, set_index, resource_name, layout_binding)

        return {
            set_index: {
                binding_index: set_bindings[binding_index]
                for binding_index in sorted(set_bindings)
            }
            for set_index, set_bindings in sorted(by_set.items())
        }

    def get_layout_bindings(self) -> dict[str, VkDescriptorSetLayoutBinding]:
        """Defines the metadata on types of resources the shader will access.

        Tells Vulkan what (descriptorType) resources to expect from this shader, and where (binding) to link them.

        This layout may be discarded if a cached version already exists through the Descriptor Manager.
        """
        resources: dict[str, tuple[int, VkDescriptorSetLayoutBinding]] = {}
        resources.update(self._get_ubo_layout_bindings())
        resources.update(self._get_sampler_layout_bindings())

        set_indices = {set_index for set_index, _ in resources.values()}
        if len(set_indices) > 1:
            msg = "Shader defines multiple descriptor sets; use get_layout_bindings_by_set()."
            raise ShaderException(msg)

        resource_bindings = [layout_binding for _, layout_binding in resources.values()]
        all_bindings = [binding.binding for binding in resource_bindings]
        assert len(all_bindings) == len(set(all_bindings)), "Duplicate bindings detected: {}"
        return {name: layout_binding for name, (_, layout_binding) in resources.items()}

    def attach(self, device: VulkanLogicalDevice) -> None:
        """Attach the device to all shaders.

        Maybe a better way to do this.
        """
        assert self.device is None, "Device has already been attached."
        self.device = device
        for shader in self.shaders:
            shader.attach(device)
        print("SHADERS CREATED", self.id, self.shaders)

    def get_stages(self):
        """All stages involved with"""
        stages = [shader.get_info() for shader in self.shaders]
        assert None not in stages, "Stage information was not found."
        return stages

    def __del__(self):
        try:
            self.delete()
        except Exception:
            pass

    def delete(self):
        type(self)._live_programs.discard(self)
        if self.ubo:
            for ubo in self.ubo.values():
                ubo.delete()
        self.ubo.clear()
        self._uniform_blocks.clear()

        if self.shaders:
            for shader in self.shaders:
                shader.delete()
        self.shaders = None

    @classmethod
    def _delete_tracked_instances(cls) -> None:
        for program in tuple(cls._live_programs):
            program.delete()

    def cleanup(self) -> None:
        """On application exit, we want to remove all resources."""
        self.delete()

    def get_uniform_block_cls(self) -> type[VulkanUniformBlock]:
        return VulkanUniformBlock


# Helper function to map field types (like 'mat4' or 'vec4') to ctypes and return the size and alignment
def type_map(field_type: str, size: int = 1):
    """Maps GLSL types to ctypes types with size and alignment information.

    Args:
        field_type (str): GLSL type as a string (e.g., 'float', 'vec3', 'mat4').
        size (int): Number of elements if the field is an array.

    Returns:
        tuple: (ctypes type, total size in bytes, alignment in bytes)
    """
    type_mapping = {
        'float': (ctypes.c_float, 4, 4),  # Single float
        'int': (ctypes.c_int, 4, 4),  # Single integer
        'uint': (ctypes.c_uint, 4, 4),  # Single unsigned integer
        'bool': (ctypes.c_uint, 4, 4),  # GLSL bool is stored as uint

        'vec2': (ctypes.c_float * 2, 8, 8),  # Two floats, aligned to 8 bytes
        'vec3': (ctypes.c_float * 3, 12, 16),  # Three floats, padded to 16 bytes
        'vec4': (ctypes.c_float * 4, 16, 16),  # Four floats, aligned to 16 bytes

        'ivec2': (ctypes.c_int * 2, 8, 8),  # Two integers
        'ivec3': (ctypes.c_int * 3, 12, 16),  # Three integers, padded to 16 bytes
        'ivec4': (ctypes.c_int * 4, 16, 16),  # Four integers

        'uvec2': (ctypes.c_uint * 2, 8, 8),  # Two unsigned integers
        'uvec3': (ctypes.c_uint * 3, 12, 16),  # Three unsigned integers, padded to 16 bytes
        'uvec4': (ctypes.c_uint * 4, 16, 16),  # Four unsigned integers

        'mat2': (ctypes.c_float * 4, 32, 8),  # 2x2 matrix (4 floats), aligned to 8 bytes per column
        'mat3': (ctypes.c_float * 9, 48, 16),  # 3x3 matrix (9 floats), aligned to 16 bytes per column
        'mat4': (ctypes.c_float * 16, 64, 16),  # 4x4 matrix (16 floats), aligned to 16 bytes per column
    }

    base_type_info = type_mapping.get(field_type)
    if base_type_info is None:
        raise ValueError(f"Unknown field type: {field_type}")

    base_type, base_size, alignment = base_type_info

    if size == 1:
        return base_type, base_size, alignment

    # For arrays, the size of each element must align to the element's alignment
    total_size = base_size * size
    padded_size = ((total_size + alignment - 1) // alignment) * alignment
    return base_type * size, padded_size, alignment


# Helper function to parse type string (e.g., 'mat4[10]') and return base type and array size
def parse_type(type_str: str) -> tuple[str, int]:
    match = re.match(r'(\w+)(\[(\d+)\])?', type_str)
    if not match:
        msg = f"Invalid type format: {type_str}"
        raise ValueError(msg)

    base_type = match.group(1)
    array_size = int(match.group(3)) if match.group(3) else 1
    return base_type, array_size


class WindowBlock(ctypes.Structure):
    _fields_ = [
        ('projection', (ctypes.c_float * 16)),
        ('view', (ctypes.c_float * 16)),
    ]


    # def __init__(self, view_class: type[Structure], buffer_size: int, binding: int) -> None:
    #     """Initialize the Uniform Buffer Object with the specified Structure."""
    #     self.buffer = BufferObject(buffer_size)
    #     self.view = view_class()
    #     self._view_ptr = pointer(self.view)
    #     self.binding = binding


def stages_to_bits(stages: Sequence[ShaderType]) -> int:
    """Convert our Shader stages into the Vulkan equivalent bits."""
    stage_bit = 0
    for stage in stages:
        stage_bit |= _shader_type_to_stage[stage]
    return stage_bit


def create_structure(name: str, uniforms: Sequence[tuple[str, str]]) -> type[Structure]:
    """Dynamically generate and return a ctypes.Structure based on the uniforms."""
    fields = []
    current_offset = 0  # Track current byte offset in the structure

    for uniform in uniforms:
        field_type, field_name = uniform[0], uniform[1]

        # Parse the type and check for arrays
        base_type, size = parse_type(field_type)

        # Get the corresponding ctypes type, size, and alignment
        ctype_field, field_size, field_alignment = type_map(base_type, size)

        # Calculate padding to ensure correct alignment
        padding = (field_alignment - (current_offset % field_alignment)) % field_alignment
        if padding > 0:
            fields.append((f'_pad{len(fields)}', ctypes.c_byte * padding))
            current_offset += padding

        # Add the actual field
        fields.append((field_name, ctype_field))
        current_offset += field_size

    # Dynamically create a ctypes.Structure class with padding and alignment
    repr_fn = lambda self: str(dict(self._fields_))
    structure_class = type(f'{name}Struct', (ctypes.Structure,), {'_fields_': fields, '__repr__': repr_fn})
    return structure_class


class VulkanUniformBlock(UniformBlock):
    def __init__(self,
                 program: VulkanShaderProgram,
                 name: str,
                 index: int,
                 size: int,
                 binding: int,
                 uniforms: tuple[tuple[str, str]],
                 uniform_count: int,
                 stages: Sequence[ShaderType]) -> None:
        super().__init__(program, name, index, size, binding, uniforms, uniform_count)

        self._stages = stages

        # Add the stage bits on where this block is located.
        self._stage_bit = stages_to_bits(stages)

        self.layout_binding = VkDescriptorSetLayoutBinding(
            binding=self.binding,
            descriptorType=VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            descriptorCount=1,  # Size of array
            stageFlags=self._stage_bit,
            pImmutableSamplers=None,
        )

    def _introspect_uniforms(self):
        structure = create_structure(self.__class__.__name__, self.uniforms)
        size = ctypes.sizeof(structure)
        self.size = size
        return structure

    def create_ubo(
        self,
        *,
        copies_per_resource: int = 3,
        alignment: int | None = None,
        strict: bool = False,
    ) -> VulkanUniformBufferObject:
        """Create a new UniformBufferObject from this uniform block."""
        context = pyglet.graphics.api.core.resolve_context()
        return VulkanUniformBufferObject(
            context,
            view_class=self.view_cls,
            buffer_size=self.size,
            binding=self.binding,
            alignment=alignment,
            copies_per_resource=copies_per_resource,
            strict=strict,
            layout_binding=self.layout_binding,
        )

    def create_instance(self):
        """Create an instance of the dynamically generated structure."""
        structure = self.get_structure()
        return structure()

class VulkanComputeShaderProgram:
    ...




def get_default_shader() -> ShaderProgram:
    """Create and return the default sprite shader.

    This method allows the module to be imported without an OpenGL Context.
    """
    try:
        return pyglet.graphics.api.core.get_shader("default_graphics")
    except KeyError:
        load_package_shader = pyglet.graphics.api.core.load_package_shader
        program = pyglet.graphics.api.core.create_shader_program(
            "default_graphics",
            (load_package_shader("pyglet.graphics.api.vulkan.shaders", "primitives.vert.spv"), 'vertex'),
            (load_package_shader("pyglet.graphics.api.vulkan.shaders", "primitives.frag.spv"), 'fragment'),
        )
        if not program.is_defined:
            program.set_attributes(
                Attribute("position", location=0, components=3, data_type="f"),
                Attribute("colors", location=1, components=4, data_type="f"),
            )
            from pyglet.graphics.api.vulkan.instance import WindowBlock
            program.set_uniform_blocks(WindowBlock)

        return program
