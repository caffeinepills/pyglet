from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence
from ctypes import byref

from pyglet.enums import BlendFactor, BlendOp
from pyglet.graphics.api.vulkan import DeviceFunc
from pyglet.graphics.api.vulkan.enums import BLEND_FACTOR_MAP, BLEND_OP_MAP
from pyglet.libs.shared.vulkan_lib import c_array_list
from pyglet.libs.shared.vulkan_lib.vulkan_core import (
    VK_ACCESS_COLOR_ATTACHMENT_READ_BIT,
    VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
    VK_ATTACHMENT_LOAD_OP_CLEAR,
    VK_ATTACHMENT_LOAD_OP_DONT_CARE,
    VK_ATTACHMENT_STORE_OP_DONT_CARE,
    VK_ATTACHMENT_STORE_OP_STORE,
    VK_COLOR_COMPONENT_A_BIT,
    VK_COLOR_COMPONENT_B_BIT,
    VK_COLOR_COMPONENT_G_BIT,
    VK_COLOR_COMPONENT_R_BIT,
    VK_DEPENDENCY_BY_REGION_BIT,
    VK_FORMAT_D32_SFLOAT,
    VK_FORMAT_R8G8B8A8_SRGB,
    VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
    VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
    VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
    VK_IMAGE_LAYOUT_UNDEFINED,
    VK_PIPELINE_BIND_POINT_GRAPHICS,
    VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
    VK_SAMPLE_COUNT_1_BIT,
    VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
    VK_SUBPASS_EXTERNAL,
    VkAttachmentDescription,
    VkAttachmentReference,
    VkPipelineColorBlendAttachmentState,
    VkRenderPassCreateInfo,
    VkSubpassDependency,
    VkSubpassDescription, VkRenderPass)

if TYPE_CHECKING:
    from pyglet.graphics.api.vulkan.devices import VulkanLogicalDevice


class Attachment:
    id: int | None
    ref: VkAttachmentReference
    def __init__(self,
                 fmt,  # Color or depth format (e.g., VK_FORMAT_B8G8R8A8_SRGB)
                 load_op=VK_ATTACHMENT_LOAD_OP_CLEAR,  # How the attachment should be treated at the start
                 store_op=VK_ATTACHMENT_STORE_OP_STORE,  # How the attachment should be treated at the end
                 initial_layout=VK_IMAGE_LAYOUT_UNDEFINED,
                 final_layout=VK_IMAGE_LAYOUT_PRESENT_SRC_KHR):
        """Initialize an attachment with default blend state and attachment description."""
        # Store the attachment format and other relevant properties
        self.id = None  # Will be set externally, e.g., index of attachment
        self.ref = None
        self.fmt = fmt
        assert isinstance(fmt, int), "Format should be int."
        self.desc = VkAttachmentDescription(
            flags=0,
            format=fmt,
            samples=VK_SAMPLE_COUNT_1_BIT,  # Typically no multisampling initially
            loadOp=load_op,
            storeOp=store_op,
            stencilLoadOp=VK_ATTACHMENT_LOAD_OP_DONT_CARE,  # Not used for color attachments
            stencilStoreOp=VK_ATTACHMENT_STORE_OP_DONT_CARE,  # Not used for color attachments
            initialLayout=initial_layout,  # Initial layout
            finalLayout=final_layout,  # Layout at the end of the render pass
        )

    def set_attachment_id(self, attachment_id: int):
        """Set the attachment ID (index) of this attachment.

        Children should also set ref.
        """
        self.id = attachment_id

    def get_attachment_description(self):
        """Return the attachment description."""
        return self.desc

    def __repr__(self):
        return f"{self.__class__.__name__}(id={self.id}, format={self.fmt})"


@dataclass
class BlendStateOptions:
    enabled: bool
    color_src: BlendFactor
    color_dst: BlendFactor
    color_op: BlendOp
    alpha_src: BlendFactor
    alpha_dst: BlendFactor
    alpha_op: BlendOp

    __slots__ = ("enabled", "color_src", "color_dst", "color_op", "alpha_src", "alpha_dst", "alpha_op")

    def get_vulkan_blend_state(self) -> VkPipelineColorBlendAttachmentState:
        return VkPipelineColorBlendAttachmentState(
            blendEnable=self.enabled,
            srcColorBlendFactor=BLEND_FACTOR_MAP[self.color_src],
            dstColorBlendFactor=BLEND_FACTOR_MAP[self.color_dst],  # No blending
            colorBlendOp=BLEND_OP_MAP[self.color_op],
            srcAlphaBlendFactor=BLEND_FACTOR_MAP[self.alpha_src],
            dstAlphaBlendFactor=BLEND_FACTOR_MAP[self.alpha_dst],
            alphaBlendOp=BLEND_OP_MAP[self.alpha_op],
            colorWriteMask=(
                    VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                    VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT
            ),
        )


class ColorAttachment(Attachment):

    def __init__(self, fmt: int = VK_FORMAT_R8G8B8A8_SRGB,
                 load_op=VK_ATTACHMENT_LOAD_OP_CLEAR,
                 store_op=VK_ATTACHMENT_STORE_OP_STORE,
                 initial_layout=VK_IMAGE_LAYOUT_UNDEFINED):
        super().__init__(fmt, load_op, store_op, initial_layout, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR)

        # self.blend_state = BlendStateOptions(
        #     enabled=False,
        #     color_src=BlendFactor.ONE,
        #     color_dst=BlendFactor.ZERO,
        #     color_op=BlendOp.ADD,
        #     alpha_src=BlendFactor.ONE,
        #     alpha_dst=BlendFactor.ZERO,
        #     alpha_op=BlendOp.ADD,
        # )

        self.blend_state = BlendStateOptions(
            enabled=True,
            color_src=BlendFactor.SRC_ALPHA,
            color_dst=BlendFactor.ONE_MINUS_SRC_ALPHA,
            color_op=BlendOp.ADD,
            alpha_src=BlendFactor.ONE,
            alpha_dst=BlendFactor.ZERO,
            alpha_op=BlendOp.ADD,
        )

    def get_vk_blend_state(self) -> VkPipelineColorBlendAttachmentState:
        return self.blend_state.get_vulkan_blend_state()

    def set_attachment_id(self, attachment_id):
        super().set_attachment_id(attachment_id)

        self.ref = VkAttachmentReference(
            attachment=attachment_id,
            layout=VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        )

    def set_blend_state(self, blend_state: BlendStateOptions) -> None:
        """This is the default blend state that Pyglet utilizes when blending is on."""
        self.blend_state = blend_state

    def __hash__(self):
        assert self.id
        return hash((self.id, self.blend_state))

class DepthAttachment(Attachment):

    def __init__(self, fmt: int = VK_FORMAT_D32_SFLOAT, load_op=VK_ATTACHMENT_LOAD_OP_CLEAR,
                 store_op=VK_ATTACHMENT_STORE_OP_STORE, initial_layout=VK_IMAGE_LAYOUT_UNDEFINED,
                 final_layout=VK_IMAGE_LAYOUT_PRESENT_SRC_KHR):
        super().__init__(fmt, load_op, store_op, initial_layout, final_layout)

    def set_attachment_id(self, attachment_id):
        super().set_attachment_id(attachment_id)

        self.ref = VkAttachmentReference(
            attachment=attachment_id,
            layout=VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        )

class Subpass:
    def __init__(self, color_attachments: Sequence[ColorAttachment],
                 depth_attachment: DepthAttachment | None = None):
        self.color_attachments = color_attachments
        self.depth_attachment = depth_attachment

        for idx, attachment in enumerate(color_attachments):
            attachment.set_attachment_id(idx)

        if self.depth_attachment:
            self.depth_attachment.set_attachment_id(len(color_attachments))

        color_refs = [attachment.ref for attachment in color_attachments]

        depth_ref = self.depth_attachment.ref if self.depth_attachment else None

        self.desc = VkSubpassDescription(
            pipelineBindPoint=VK_PIPELINE_BIND_POINT_GRAPHICS,
            colorAttachmentCount=len(self.color_attachments),
            pColorAttachments=c_array_list(color_refs, VkAttachmentReference),
            pDepthStencilAttachment=depth_ref,
            pInputAttachments=None,  # input attachments?
            pPreserveAttachments=None,  # preserve attachments?
        )


class RenderPassManager:
    def __init__(self, logical_device: VulkanLogicalDevice):
        self.logical_device = logical_device
        self.renderpasses = {}

    def get_renderpass(self, surface_fmt: int, color_attachments: int, depth: bool, subpasses: int=1) -> RenderPass:
        render_key = (surface_fmt, color_attachments, depth, subpasses)
        if render_key in self.renderpasses:
            return self.renderpasses[render_key]

        print("LOGICAL", self.logical_device)

        self.renderpasses[render_key] = RenderPass(self.logical_device, surface_fmt, depth)
        return self.renderpasses[render_key]


class RenderPass:
    def __init__(self, logical_device: VulkanLogicalDevice,
                 surface_format: int,
                 depth: bool=False,
                 sync: bool=False):
        print("surface", surface_format)
        self.device = logical_device
        self.attachments = []
        self.surface_format = surface_format
        self.color_attachments = [ColorAttachment(surface_format)]
        self.depth_attachment = None

        self.attachments.extend(self.color_attachments)
        if depth:
            self.depth_attachment = DepthAttachment(surface_format)
            self.attachments.append(self.depth_attachment)

        self.subpasses = [Subpass(self.color_attachments, self.depth_attachment)]
        self.dependencies = []

        self.vk_renderpass = None

        if sync:
            self.dependencies.append(self.create_external_dependency())

        self.vk_renderpass = self.create_render_pass()

    def key(self):
        return (self.surface_format, len(self.color_attachments), bool(self.depth_attachment), len(self.subpasses))

    def create_render_pass(self) -> VkRenderPass:
        """Creates a Vulkan render pass for color attachment."""
        attachment_descs = [attachment.desc for attachment in self.attachments]
        subpasses = [subpass.desc for subpass in self.subpasses]

        render_pass_info = VkRenderPassCreateInfo(
            sType=VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
            attachmentCount=len(self.attachments),
            pAttachments=c_array_list(attachment_descs, VkAttachmentDescription),
            subpassCount=len(subpasses),
            pSubpasses=c_array_list(subpasses, VkSubpassDescription),
            dependencyCount=len(self.dependencies),
            pDependencies=c_array_list(self.dependencies, VkSubpassDependency),
        )

        vk_renderpass = VkRenderPass()
        DeviceFunc.vkCreateRenderPass(self.device.vk_device, byref(render_pass_info), None, byref(vk_renderpass))
        return vk_renderpass

    def create_external_dependency(self):
        """Create a dependency from external operations to the first subpass."""
        # Ensure external operations complete before we start writing or reading from the attachments.
        return VkSubpassDependency(
            srcSubpass=VK_SUBPASS_EXTERNAL,  # External operations
            dstSubpass=0,
            srcStageMask=VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            srcAccessMask=0,
            dstStageMask=VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            dstAccessMask=VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
            dependencyFlags=VK_DEPENDENCY_BY_REGION_BIT,
        )

    def __del__(self):
        self.delete()

    def delete(self):
        """Cleans up the Vulkan render pass."""
        if self.vk_renderpass:
            self.device.vkDestroyRenderPass(self.device.vk_device, self.vk_renderpass, None)
            self.vk_renderpass = None
