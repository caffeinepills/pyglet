from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence, Any
from ctypes import byref

from pyglet.enums import BlendFactor, BlendOp
from pyglet.graphics.api.vulkan import DeviceFunc

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


@dataclass(frozen=True, slots=True)
class Attachment:
    fmt: int  # Color or depth format (e.g., VK_FORMAT_B8G8R8A8_SRGB)
    samples: int = VK_SAMPLE_COUNT_1_BIT
    load_op: int = VK_ATTACHMENT_LOAD_OP_CLEAR    # How the attachment should be treated at the start
    store_op: int = VK_ATTACHMENT_STORE_OP_STORE  # How the attachment should be treated at the end
    stencil_load_op: int = VK_ATTACHMENT_LOAD_OP_DONT_CARE
    stencil_store_op: int = VK_ATTACHMENT_STORE_OP_DONT_CARE
    initial_layout: int = VK_IMAGE_LAYOUT_UNDEFINED
    final_layout: int = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL

    def create_description(self) -> VkAttachmentDescription:
        return VkAttachmentDescription(
            flags=0,
            format=self.fmt,
            samples=self.samples,
            loadOp=self.load_op,
            storeOp=self.store_op,
            stencilLoadOp=self.stencil_load_op,
            stencilStoreOp=self.stencil_store_op,
            initialLayout=self.initial_layout,
            finalLayout=self.final_layout,
        )

    def ref(self, index: int) -> AttachmentRef:
        return AttachmentRef(
            attachment=index,
            layout=VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        )

@dataclass(frozen=True, slots=True)
class AttachmentRef:
    attachment: int
    layout: int

    def create_vk_ref(self) -> VkAttachmentReference:
        return VkAttachmentReference(
            attachment=self.attachment,
            layout=self.layout,
        )

@dataclass(frozen=True, slots=True)
class ColorAttachment(Attachment):
    final_layout: int = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL

    @classmethod
    def get_default(cls, fmt: int, *, offscreen: bool, clear_on_load: bool = True) -> ColorAttachment:
        """Defines Attachment specs for the color attachments."""
        final_layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL if offscreen else VK_IMAGE_LAYOUT_PRESENT_SRC_KHR
        if clear_on_load:
            initial_layout = VK_IMAGE_LAYOUT_UNDEFINED
            load_op = VK_ATTACHMENT_LOAD_OP_CLEAR
        else:
            initial_layout = VK_IMAGE_LAYOUT_UNDEFINED
            load_op = VK_ATTACHMENT_LOAD_OP_DONT_CARE

        return ColorAttachment(
            fmt=fmt,
            load_op=load_op,
            initial_layout=initial_layout,
            final_layout=final_layout,
        )


@dataclass(frozen=True, slots=True)
class DepthAttachment(Attachment):
    fmt: int = VK_FORMAT_D32_SFLOAT
    store_op: int = VK_ATTACHMENT_STORE_OP_DONT_CARE
    final_layout: int = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL

    @classmethod
    def get_default(cls, fmt: int = VK_FORMAT_D32_SFLOAT) -> DepthAttachment:
        """Defines Attachment specs for the color attachments."""
        return DepthAttachment(
            fmt=fmt,
            initial_layout=VK_IMAGE_LAYOUT_UNDEFINED,
            final_layout=VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        )

    def ref(self, index: int) -> AttachmentRef:
        return AttachmentRef(
            attachment=index,
            layout=VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        )



@dataclass(slots=True, frozen=True)
class Subpass:
    color_attachments: tuple[AttachmentRef, ...]
    depth_attachment: AttachmentRef | None = None
    input_attachments: tuple[AttachmentRef, ...] = ()
    resolve_attachments: tuple[AttachmentRef, ...] = ()
    preserve_attachments: tuple[int, ...] = ()

    @classmethod
    def from_attachments(cls, color_attachments: tuple[Attachment, ...],
                         depth_attachment: Attachment | None) -> Subpass:
        attachment_refs = [attachment.ref(idx) for idx, attachment in enumerate(color_attachments)]
        return cls(
            color_attachments=tuple(attachment_refs),
            depth_attachment=depth_attachment.ref(len(attachment_refs)) if depth_attachment else None,
        )

    def get_description(self) -> VkSubpassDescription:
        color_refs = [attach_ref.create_vk_ref() for attach_ref in self.color_attachments]
        depth_ref = self.depth_attachment.create_vk_ref() if self.depth_attachment else None
        return VkSubpassDescription(
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

@dataclass(frozen=True)
class RenderPassSpec:
    color_attachments: tuple[ColorAttachment, ...]
    depth_attachment: DepthAttachment | None = None
    sync: bool = False

class RenderPass:
    attachments: list[Attachment]

    def __init__(self, logical_device: VulkanLogicalDevice,
                 color_formats: Sequence[int],
                 depth_format: int | None=None,
                 sync: bool=False,
                 offscreen: bool=False,
                 clear_on_load: bool = True):
        self.device = logical_device

        color_attachments = tuple([
            ColorAttachment.get_default(fmt, offscreen=offscreen, clear_on_load=clear_on_load)
            for fmt in color_formats
        ])
        depth_attachment = DepthAttachment.get_default(fmt=depth_format) if depth_format else None
        self._key = RenderPassSpec(
            color_attachments=color_attachments,
            depth_attachment=depth_attachment,
            sync=sync,
        )

        self.attachments = [*color_attachments]
        if depth_format:
            assert depth_attachment is not None
            self.attachments.append(depth_attachment)

        self.subpasses = [Subpass.from_attachments(color_attachments, depth_attachment)]
        self.dependencies = []

        self.vk_renderpass = None

        if sync:
            self.dependencies.append(self.create_external_dependency())

        self.vk_renderpass = self.create_render_pass()

    @property
    def color_attachments(self) -> tuple[ColorAttachment, ...]:
        return self._key.color_attachments

    @property
    def depth_attachment(self) -> DepthAttachment | None:
        return self._key.depth_attachment

    def key(self):
        return (self._key, *self.subpasses)

    def create_render_pass(self) -> VkRenderPass:
        """Creates a Vulkan render pass for color attachment."""
        attachment_descs = [attachment.create_description() for attachment in self.attachments]
        subpasses = [subpass.get_description() for subpass in self.subpasses]

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
