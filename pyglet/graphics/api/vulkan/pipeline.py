from __future__ import annotations

import ctypes
from abc import abstractmethod
from dataclasses import dataclass
from enum import IntEnum, IntFlag
from typing import Sequence, TYPE_CHECKING, NewType, List, Any
from ctypes import byref, pointer

from pyglet.graphics.api.vulkan import c_array_list, DeviceFunc

from pyglet.graphics.api.vulkan.enums import geometry_map
from pyglet.enums import GeometryMode
from pyglet.libs.shared.vulkan_lib.vulkan_core import VK_SAMPLE_COUNT_1_BIT, VK_SAMPLE_COUNT_2_BIT, \
    VK_SAMPLE_COUNT_4_BIT, \
    VK_SAMPLE_COUNT_8_BIT, VK_SAMPLE_COUNT_16_BIT, VK_SAMPLE_COUNT_32_BIT, VK_SAMPLE_COUNT_64_BIT, \
    VkPipelineViewportStateCreateInfo, VkExtent2D, VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO, VkViewport, \
    VkRect2D, VkOffset2D, VkDescriptorBufferInfo, VkWriteDescriptorSet, VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, \
    VK_PIPELINE_BIND_POINT_GRAPHICS, VkDescriptorSet, VkDescriptorSetLayout, VkPipelineVertexInputStateCreateInfo, \
    VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO, VkPipelineLayoutCreateInfo, \
    VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, VkPipelineLayout, VkGraphicsPipelineCreateInfo, \
    VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, VkPipeline, VkPipelineColorBlendStateCreateInfo, \
    VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO, VkPipelineColorBlendAttachmentState, \
    VkPipelineDepthStencilStateCreateInfo, VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO, \
    VkPipelineInputAssemblyStateCreateInfo, VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO, \
    VkPipelineMultisampleStateCreateInfo, VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO, \
    VkPipelineRasterizationStateCreateInfo, VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO, \
    VkVertexInputBindingDescription, VkVertexInputAttributeDescription, VkPipelineShaderStageCreateInfo, \
    VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VkPushConstantRange, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, \
    VkDescriptorImageInfo, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_NULL_HANDLE, VK_SHADER_STAGE_FRAGMENT_BIT, \
    VkCommandBuffer

if TYPE_CHECKING:
    from pyglet.graphics import Group
    from pyglet.graphics.api.vulkan.texture import VulkanTexture
    from pyglet.graphics.api.vulkan.vertexdomain import VertexDomain
    from pyglet.graphics.api.vulkan.descriptor import DescriptorPool, DescriptorManager, DescriptorSetObject
    from pyglet.graphics.api.vulkan.renderpass import ColorAttachment, RenderPass, RenderPassManager
    from pyglet.graphics.api.vulkan.devices import VulkanLogicalDevice, VulkanDevices
    from pyglet.graphics.api.vulkan.shader import VulkanShaderProgram, VulkanUniformBufferObject, _PushConstant


class VkCompareOp(IntEnum):
    VK_COMPARE_OP_NEVER = 0
    VK_COMPARE_OP_LESS = 1
    VK_COMPARE_OP_EQUAL = 2
    VK_COMPARE_OP_LESS_OR_EQUAL = 3
    VK_COMPARE_OP_GREATER = 4
    VK_COMPARE_OP_NOT_EQUAL = 5
    VK_COMPARE_OP_GREATER_OR_EQUAL = 6
    VK_COMPARE_OP_ALWAYS = 7
    VK_COMPARE_OP__BEGIN_RANGE = 0
    VK_COMPARE_OP__END_RANGE = 7
    VK_COMPARE_OP__RANGE_SIZE = 8
    VK_COMPARE_OP__MAX_ENUM = 2147483647

class PipelineState:
    #info: Structure # Give type later.
    def _get_info(self):
        ...


class DepthStencilState(PipelineState):
    def __init__(self, depth_test_enable: bool,
                 depth_write_enable: bool,
                 depth_compare_op: VkCompareOp,
                 stencil_test_enable: bool = False):
        self.depth_test_enable = depth_test_enable
        self.depth_write_enable = depth_write_enable
        self.depth_compare_op = depth_compare_op
        self.stencil_test_enable = stencil_test_enable
        # max depth min depth optional.

        self.info = self._get_info()

    def _get_info(self):
        return VkPipelineDepthStencilStateCreateInfo(
            sType=VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
            depthTestEnable=self.depth_test_enable,
            depthWriteEnable=self.depth_write_enable,
            depthCompareOp=self.depth_compare_op.value,
            stencilTestEnable=self.stencil_test_enable,
        )

# Vertex Input State is retrieve from Domain?

class VertexAssemblyState(PipelineState):
    def __init__(self, mode: GeometryMode, primitive_restart_enable: bool = False):
        self.mode = geometry_map[mode]
        self.primitive_restart_enable = primitive_restart_enable
        self.info = self._get_info()

    def _get_info(self) -> VkPipelineInputAssemblyStateCreateInfo:
        return VkPipelineInputAssemblyStateCreateInfo(
            sType=VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
            topology=self.mode,
            primitiveRestartEnable=self.primitive_restart_enable,
        )


class ViewportState(PipelineState):
    def __init__(self, viewports: Sequence[VkViewport], scissors: Sequence[VkRect2D]):
        self.viewports = viewports
        self.scissors = scissors
        self.info = self._get_info()

    def _get_info(self) -> VkPipelineViewportStateCreateInfo:
        vp_array = c_array_list(self.viewports, VkViewport)
        scissor_array = c_array_list(self.scissors, VkRect2D)
        return VkPipelineViewportStateCreateInfo(
            sType=VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
            viewportCount=len(self.viewports),
            pViewports=vp_array,
            scissorCount=len(self.scissors),
            pScissors=scissor_array,
        )

    def __hash__(self) -> int:
        return hash((self.viewports, self.scissors))


bit_flag = {
    1:  VK_SAMPLE_COUNT_1_BIT,
    2:  VK_SAMPLE_COUNT_2_BIT,
    4:  VK_SAMPLE_COUNT_4_BIT,
    8:  VK_SAMPLE_COUNT_8_BIT,
    16: VK_SAMPLE_COUNT_16_BIT,
    32: VK_SAMPLE_COUNT_32_BIT,
    64: VK_SAMPLE_COUNT_64_BIT,
}

multisample_available = True  # Check: sampleRateShading available

class MultisampleState(PipelineState):
    def __init__(self, enabled=False, sample_count: int = VK_SAMPLE_COUNT_1_BIT):
        self.enabled = enabled
        self.sample_count = sample_count

        # Check if available?
        if multisample_available:
            if sample_count > 0:
                self.sample_count = sample_count
            else:
                self.enabled = False
        else:
            self.enabled = False

        self.info = self._get_info()

    def _get_info(self):
        return VkPipelineMultisampleStateCreateInfo(
            sType=VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
            rasterizationSamples=self.sample_count,
            sampleShadingEnable=self.enabled,
        )


# Provided by VK_VERSION_1_0
class VkPolygonMode(IntEnum):
    VK_POLYGON_MODE_FILL = 0
    VK_POLYGON_MODE_LINE = 1
    VK_POLYGON_MODE_POINT = 2

    # Provided by VK_NV_fill_rectangle
    VK_POLYGON_MODE_FILL_RECTANGLE_NV = 1000153000

# Provided by VK_VERSION_1_0
class VkCullModeFlagBits(IntFlag):
    VK_CULL_MODE_NONE  = 0
    VK_CULL_MODE_FRONT_BIT  = 0x00000001
    VK_CULL_MODE_BACK_BIT  = 0x00000002
    VK_CULL_MODE_FRONT_AND_BACK  = 0x00000003


# Provided by VK_VERSION_1_0
class VkFrontFace(IntEnum):
    VK_FRONT_FACE_COUNTER_CLOCKWISE = 0
    VK_FRONT_FACE_CLOCKWISE = 1


class RasterizationState(PipelineState):
    def __init__(self,
                 polygon_mode: VkPolygonMode = VkPolygonMode.VK_POLYGON_MODE_FILL,
                 cull_mode: VkCullModeFlagBits = VkCullModeFlagBits.VK_CULL_MODE_BACK_BIT,
                 front_face: VkFrontFace = VkFrontFace.VK_FRONT_FACE_CLOCKWISE,
                 depth_clamp_enable: bool = False,
                 line_width: float = 1.0,
                 ):
        self.polygon_mode = polygon_mode
        self.cull_mode = cull_mode
        self.front_face = front_face
        self.depth_clamp_enable = depth_clamp_enable
        self.line_width = line_width
        self.info = self._get_info()

    def _get_info(self) -> VkPipelineRasterizationStateCreateInfo:
        return VkPipelineRasterizationStateCreateInfo(
            sType=VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
            depthClampEnable=self.depth_clamp_enable,
            rasterizerDiscardEnable=False,
            polygonMode=self.polygon_mode,
            cullMode=self.cull_mode,
            frontFace=self.front_face,
            depthBiasEnable=False,
            lineWidth=self.line_width,
        )

    def __hash__(self) -> int:
        return hash((self.polygon_mode, self.cull_mode, self.front_face, self.depth_clamp_enable, self.line_width))

class ColorBlendState(PipelineState):
    def __init__(self, attachments: Sequence[ColorAttachment], logic_op_enable: bool = False):
        self.attachments = attachments
        self.logic_op_enable = logic_op_enable
        self.info = self.get_info()

    def get_info(self) -> VkPipelineColorBlendStateCreateInfo:

        attachments = [attachment.get_vk_blend_state() for attachment in self.attachments]
        attachment_array = c_array_list(attachments, VkPipelineColorBlendAttachmentState)
        return VkPipelineColorBlendStateCreateInfo(
            sType=VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
            logicOpEnable=self.logic_op_enable,
            attachmentCount=len(self.attachments),
            # Blend states of all attachments
            pAttachments=attachment_array,
        )

    def __hash__(self):
        print("(self.logic_op_enable, *self.attachments)", (self.logic_op_enable, *self.attachments))
        return hash((self.logic_op_enable, *self.attachments))



#
# class AttachmentState:
#     def __init__(self):
#         self.blend_color_src = BlendFactor.SRC_COLOR
#         self.blend_color_dst = BlendFactor.SRC_COLOR
#         self.blend_color_op = BlendOp.ADD
#         self.blend_alpha_src = BlendFactor.SRC_COLOR
#         self.blend_dst_dst = BlendFactor.SRC_COLOR
#         self.blend_alpha_op = BlendOp.ADD
#
#
#
# class GraphicsGroup:
#     attachments: dict[int, AttachmentState]
#
#     def __init__(self):
#         self.states = []
#         self._set_states = []
#         self._hash = 0
#         self.attachments = {}
#
#     def set_state(self):
#         for funcs_to_call in self.states:
#             funcs_to_call[0](*funcs_to_call[1:])
#     def set(self):
#         for state in self._set_states:
#             if state == "scissor":
#                 self.states.append(_gl_to_func[state[0]], *state[1:])
#
#     def set_scissor(self, x, y, width, height):
#         self._set_states.append(("scissor", x, y, width, height))
#
#     def set_color_blend_state(self, blend_src: BlendFactor, blend_dst: BlendFactor, blend_op: BlendOp, attachment_idx=0):
#         if attachment_idx not in self.attachments:
#             self.attachments[attachment_idx] = AttachmentState()
#
#         self.attachments[attachment_idx].blend_color_src = blend_src
#         self.attachments[attachment_idx].blend_color_dst = blend_dst
#         self.attachments[attachment_idx].blend_color_op = blend_op
#
#     def set_alpha_blend_state(self, blend_src: BlendFactor, blend_dst: BlendFactor, blend_op: BlendOp, attachment_idx=0):
#         if attachment_idx not in self.attachments:
#             self.attachments[attachment_idx] = AttachmentState()
#
#         self.attachments[attachment_idx].blend_alpha_src = blend_src
#         self.attachments[attachment_idx].blend_dst_dst = blend_dst
#         self.attachments[attachment_idx].blend_alpha_op = blend_op



class PipelineLayoutCache:
    def __init__(self, device: VulkanLogicalDevice) -> None:
        self.device = device
        self.cache = {}

    def get(self, push_constants: Sequence[_PushConstant],
            descriptor_set_layouts: Sequence[VkDescriptorSetLayout]) -> VkPipelineLayout:

        constant_ranges = [pc.constant_range for pc in push_constants]
        key = self._hash_info(constant_ranges, descriptor_set_layouts)
        if key not in self.cache:
            pipeline_layout_info = VkPipelineLayoutCreateInfo(
                sType=VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                setLayoutCount=len(descriptor_set_layouts),
                pSetLayouts=c_array_list(descriptor_set_layouts, VkDescriptorSetLayout),
                pushConstantRangeCount=len(constant_ranges),
                pPushConstantRanges=c_array_list(constant_ranges, VkPushConstantRange),
            )

            pipeline_layout = VkPipelineLayout()
            self.device.vkCreatePipelineLayout(
                self.device.vk_device, byref(pipeline_layout_info), None, byref(pipeline_layout),
            )
            self.cache[key] = pipeline_layout
        return self.cache[key]

    def delete_cache(self) -> None:
        for pl_layout in self.cache.values():
            self.device.vkDestroyPipelineLayout(self.device.vk_device, pl_layout, None)
        self.cache.clear()

    @staticmethod
    def _hash_info(constant_ranges: list[VkPushConstantRange], set_layouts: Sequence[VkDescriptorSetLayout]) -> int:
        # Convert layout info into a hashable format (e.g., tuple of descriptor set layouts, push constants)
        return hash((tuple([layout.value for layout in set_layouts]), tuple([(cr.stageFlags, cr.offset, cr.size) for cr in constant_ranges])))



class GraphicsPipelineManager:
    pipelines: dict[tuple, GraphicsPipeline]

    def __init__(self, devices: VulkanDevices,
                 descriptor_mgr: DescriptorManager,
                 shaders: dict[str, VulkanShaderProgram],  # Temp.
                 renderpass_mgr: RenderPassManager,
                 ):
        self.devices = devices
        self.descriptor_mgr = descriptor_mgr
        self.layout_cache = PipelineLayoutCache(devices.logical_device)
        self.pipelines = {}
        self.shaders = shaders
        self.renderpass_mgr = renderpass_mgr

    def delete(self):
        """Cleanup all pipeline resources."""
        self.layout_cache.delete_cache()

        for pipeline in self.pipelines.values():
            pipeline.delete()

        self.pipelines.clear()

        self.layout_cache = None

    def get_pipeline_from_group(self, group: Group,
                                renderpass: RenderPass,
                                geometry_mode: GeometryMode,
                                width: int, height: int,
                                vertex_domain: VertexDomain) -> GraphicsPipeline | None:
        if program_state := group._state_names.get("ShaderProgramState"):
            return self.get_pipeline(program_state.program, renderpass, geometry_mode, width, height, vertex_domain)
        return None

    def get_pipeline(self, shader_program: VulkanShaderProgram,
                     renderpass: RenderPass,
                     geometry_mode: GeometryMode,
                     width: int,
                     height: int,
                     domain: VertexDomain,
                     ):
        key = (shader_program, renderpass, geometry_mode, width, height, domain._hashable_attributes)
        print("KEY!", key)
        if key in self.pipelines:
            return self.pipelines[key]

        return self.create(shader_program, renderpass, geometry_mode, width, height, domain)

    def create(self, shader_program: VulkanShaderProgram,
               renderpass: RenderPass,
               geometry_mode: GeometryMode,
               width: int,
               height: int,
               domain: VertexDomain):

        key = (shader_program, renderpass, geometry_mode, width, height, domain._hashable_attributes)
        assert key not in self.pipelines
        #render_pass_key = pipeline_key[1]
        #renderpass = self.renderpass_mgr.get_renderpass(*render_pass_key)
        extent = VkExtent2D(width, height)
        pipeline = GraphicsPipeline(self.devices, self.devices.logical_device, shader_program,
                                    renderpass, geometry_mode, extent, self.descriptor_mgr, self.layout_cache, domain)
        pipeline.create()
        self.pipelines[key] = pipeline
        return pipeline

class _GraphicsPipelineBase:
    descriptor_set_layouts: list[VkDescriptorSetLayout]
    vk_pipeline: None
    program: VulkanShaderProgram

    def __init__(self, devices: VulkanDevices,
                 device: VulkanLogicalDevice,
                 program: VulkanShaderProgram,
                 render_pass: RenderPass,
                 geometry_mode: GeometryMode,
                 extent: VkExtent2D,
                 descriptor_mgr: DescriptorManager,
                 layout_cache: PipelineLayoutCache) -> None:
        self.devices = devices
        self.device = device
        self.render_pass = render_pass
        self.vk_pipeline = None
        self.bound = False  # remove.
        self.extent = extent
        self.geometry_mode = geometry_mode
        self.descriptor_mgr = descriptor_mgr
        self.descriptor_set_layouts = []
        self.current_frame = 0
        self.layout_cache = layout_cache
        self.program = program

        # Create the shader information when we attach a device.
        if not program.device:
            program.attach(device)

        self.viewport_state = ViewportState(
            [VkViewport(0.0, 0.0, float(extent.width), float(extent.height), 0.0, 1.0)],
            [VkRect2D(VkOffset2D(0, 0), extent)],
        )

        # It's important to keep the same order of color attachments throughout.
        self.color_blend_state = ColorBlendState(self.render_pass.color_attachments)
        self.rasterization_state = RasterizationState()
        self.vertex_assembly_state = VertexAssemblyState(self.geometry_mode, False)
        self.multisample_state = MultisampleState()

    def create(self) -> None:
        """Create the pipeline and pipeline layout."""
        self.vk_pipeline = self._create_pipeline()

    def bind(self, command_buffer: VkCommandBuffer) -> None:
        self.device.vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, self.vk_pipeline)

    @abstractmethod
    def get_vertex_input_state_info(self) -> VkPipelineVertexInputStateCreateInfo:
        ...

    def _create_pipeline(self):
        """Creates a Vulkan graphics pipeline."""
        self.descriptor_set_layouts = self.descriptor_mgr.get_descriptor_set_layouts(self.program)

        push_constants = self.program.push_constants
        self.pipeline_layout = self.layout_cache.get(push_constants, self.descriptor_set_layouts)

        stages = self.program.get_stages()
        vertex_input = self.get_vertex_input_state_info()

        pipeline_create_info = VkGraphicsPipelineCreateInfo(
            sType=VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
            stageCount=len(self.program.shaders),
            pStages=c_array_list(stages, VkPipelineShaderStageCreateInfo),
            pVertexInputState=pointer(vertex_input),
            pInputAssemblyState=pointer(self.vertex_assembly_state.info),
            pViewportState=pointer(self.viewport_state.info),
            pRasterizationState=pointer(self.rasterization_state.info),
            pMultisampleState=pointer(self.multisample_state.info),
            pColorBlendState=pointer(self.color_blend_state.info),
            layout=self.pipeline_layout,
            renderPass=self.render_pass.vk_renderpass,
            subpass=0,
        )

        self.pipeline_infos = [pipeline_create_info]
        pipeline_info_array = c_array_list(self.pipeline_infos, VkGraphicsPipelineCreateInfo)

        pipelines = (VkPipeline * len(self.pipeline_infos))()
        DeviceFunc.vkCreateGraphicsPipelines(self.device.vk_device, 0, len(self.pipeline_infos), pipeline_info_array, None, pipelines)
        return pipelines[0]

    def push_constants(self, command_buffer, push_constant_idx=0):
        if self.program.push_constants:
            pc = self.program.push_constants[push_constant_idx]
            self.device.vkCmdPushConstants(
                command_buffer,  # Command buffer
                self.pipeline_layout,  # Pipeline layout
                pc.constant_range.stageFlags,  # Shader stages
                0,  # Offset
                pc.constant_range.size,  # Size
                ctypes.byref(pc.struct),  # Pointer to the structure
            )

    def __del__(self):
        print("GC pipeline")
        self.delete()

    def delete(self) -> None:
        """Cleans up the Vulkan graphics pipeline and pipeline layout."""
        if self.vk_pipeline:
            print("- Start destroy pipeline")
            self.device.vkDestroyPipeline(self.device.vk_device, self.vk_pipeline, None)
            self.vk_pipeline = None
            print("Destroy pipeline")

        self.descriptor_set_layouts.clear()
        self.pipeline_layout = None


class EmptyVertexGraphicsPipeline(_GraphicsPipelineBase):

    def get_vertex_input_state_info(self) -> VkPipelineVertexInputStateCreateInfo:
        return VkPipelineVertexInputStateCreateInfo(
            sType=VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
            vertexBindingDescriptionCount=0,
            pVertexBindingDescriptions=None,
            vertexAttributeDescriptionCount=0,
            pVertexAttributeDescriptions=None,
        )

class GraphicsPipeline(_GraphicsPipelineBase):
    def __init__(self, devices: VulkanDevices, device: VulkanLogicalDevice, program: VulkanShaderProgram,
                 render_pass: RenderPass, geometry_mode: GeometryMode, extent: VkExtent2D, descriptor_mgr: DescriptorManager,
                 layout_cache: PipelineLayoutCache, vertex_domain: VertexDomain):
        super().__init__(devices, device, program, render_pass, geometry_mode, extent, descriptor_mgr, layout_cache)
        self.vertex_domain = vertex_domain

    def get_vertex_input_state_info(self) -> VkPipelineVertexInputStateCreateInfo:
        vertex_binding_descrip = self.vertex_domain.get_binding_descriptions()
        vbd_array = c_array_list(vertex_binding_descrip, VkVertexInputBindingDescription)

        print("BUILDING PIPELINE?", self.program)
        attribute_descrips = self.vertex_domain.get_attribute_descriptions()
        vad_array = c_array_list(attribute_descrips, VkVertexInputAttributeDescription)

        return VkPipelineVertexInputStateCreateInfo(
            sType=VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
            vertexBindingDescriptionCount=len(vertex_binding_descrip),
            pVertexBindingDescriptions=vbd_array,
            vertexAttributeDescriptionCount=len(attribute_descrips),
            pVertexAttributeDescriptions=vad_array,
        )