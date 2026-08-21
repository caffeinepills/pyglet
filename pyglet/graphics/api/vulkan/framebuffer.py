"""Vulkan framebuffer abstractions with OpenGL-style compatibility.

Vulkan does not expose a global framebuffer bind state. This module provides
an API surface similar to :mod:`pyglet.graphics.api.gl.framebuffer` so code can
create/attach/delete framebuffer resources with similar ergonomics.
"""
from __future__ import annotations

from ctypes import byref
from typing import TYPE_CHECKING

import pyglet

from pyglet.enums import ComponentFormat, FramebufferAttachment, FramebufferTarget
from pyglet.graphics.api.vulkan import DeviceFunc, c_array_list
from pyglet.graphics.api.vulkan.buffer import StagingBufferObject2
from pyglet.image.base import ImageData
from pyglet.libs.shared.vulkan_lib.vulkan_core import (
    VK_ACCESS_MEMORY_READ_BIT,
    VK_ACCESS_TRANSFER_READ_BIT,
    VK_FORMAT_B8G8R8A8_SRGB,
    VK_FORMAT_B8G8R8A8_UNORM,
    VK_FORMAT_R8G8B8A8_SRGB,
    VK_FORMAT_R8G8B8A8_UNORM,
    VK_IMAGE_ASPECT_COLOR_BIT,
    VK_IMAGE_ASPECT_DEPTH_BIT,
    VK_IMAGE_ASPECT_STENCIL_BIT,
    VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
    VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
    VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
    VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
    VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
    VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
    VK_PIPELINE_STAGE_TRANSFER_BIT,
    VK_QUEUE_FAMILY_IGNORED,
    VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
    VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
    VkBufferImageCopy,
    VkExtent3D,
    VkFramebuffer,
    VkFramebufferCreateInfo,
    VkImageMemoryBarrier,
    VkImageSubresourceRange,
    VkImageSubresourceLayers,
    VkImageView,
    VkOffset3D,
)

if TYPE_CHECKING:
    from pyglet.graphics.api.vulkan.texture import VulkanTexture
    from pyglet.customtypes import DataTypes
    from pyglet.graphics.api.vulkan.instance import VulkanSurfaceContext
    from pyglet.graphics.api.vulkan.renderpass import RenderPass


_attachment_order = (
    FramebufferAttachment.COLOR0,
    FramebufferAttachment.COLOR1,
    FramebufferAttachment.COLOR2,
    FramebufferAttachment.COLOR3,
    FramebufferAttachment.COLOR4,
    FramebufferAttachment.COLOR5,
    FramebufferAttachment.COLOR6,
    FramebufferAttachment.COLOR7,
    FramebufferAttachment.COLOR8,
    FramebufferAttachment.COLOR9,
    FramebufferAttachment.COLOR10,
    FramebufferAttachment.COLOR11,
    FramebufferAttachment.COLOR12,
    FramebufferAttachment.COLOR13,
    FramebufferAttachment.COLOR14,
    FramebufferAttachment.COLOR15,
    FramebufferAttachment.DEPTH_STENCIL,
    FramebufferAttachment.DEPTH,
    FramebufferAttachment.STENCIL,
)

_color_attachment_order = _attachment_order[:16]


def _resolve_context(context: VulkanSurfaceContext | None) -> VulkanSurfaceContext:
    resolved = context or pyglet.graphics.api.core.current_context
    if not resolved or not hasattr(resolved, "logical_device"):
        msg = "A Vulkan context is required to create framebuffer resources."
        raise RuntimeError(msg)
    return resolved


def _resolve_framebuffer_image_index(frame_sync) -> int:
    if getattr(frame_sync, "_has_acquired_image", False):
        return frame_sync.current_image_index()

    for attr in ("last_presented_image_index", "last_acquired_image_index"):
        idx = getattr(frame_sync, attr, None)
        if idx is not None:
            return int(idx)

    return frame_sync.current_image_index()


def _swizzle_bgra_to_rgba(raw: bytes) -> bytes:
    rgba = bytearray(len(raw))
    rgba[0::4] = raw[2::4]
    rgba[1::4] = raw[1::4]
    rgba[2::4] = raw[0::4]
    rgba[3::4] = raw[3::4]
    return bytes(rgba)


def _get_view_from_resource(resource: VulkanTexture | VulkanRenderbuffer | VkImageView) -> VkImageView:
    view = getattr(getattr(resource, "image_view", None), "vk_imageview", None)
    if view:
        return view

    view = getattr(resource, "vk_imageview", None)
    if view:
        return view

    if isinstance(resource, VkImageView):
        return resource

    msg = "Attachment must expose a VkImageView (texture.image_view.vk_imageview or vk_imageview)."
    raise TypeError(msg)


def _get_attachment_size(resource) -> tuple[int, int]:
    width = int(getattr(resource, "width", 0) or 0)
    height = int(getattr(resource, "height", 0) or 0)
    return width, height


def _get_attachment_format(resource) -> int | None:
    for candidate in (
        getattr(getattr(resource, "image_view", None), "vk_fmt", None),
        getattr(getattr(resource, "image", None), "vk_fmt", None),
        getattr(resource, "vk_fmt", None),
    ):
        if candidate is not None:
            return int(candidate)
    return None


def _get_depth_attachment_key(attachments: dict[FramebufferAttachment, tuple[VkImageView, int, int, object]]) -> FramebufferAttachment | None:
    for depth_attachment in (FramebufferAttachment.DEPTH_STENCIL, FramebufferAttachment.DEPTH, FramebufferAttachment.STENCIL):
        if depth_attachment in attachments:
            return depth_attachment
    return None


def _resolve_render_pass_handle(render_pass, context: VulkanSurfaceContext):
    if render_pass is not None:
        return getattr(render_pass, "vk_renderpass", render_pass)
    ctx_renderpass = getattr(context, "renderpass", None)
    if ctx_renderpass is None:
        return None
    return getattr(ctx_renderpass, "vk_renderpass", ctx_renderpass)


def _resolve_render_pass_object(render_pass, context: VulkanSurfaceContext):
    if render_pass is not None:
        return render_pass
    return getattr(context, "renderpass", None)


def get_viewport() -> tuple[int, int, int, int]:
    """Get the current Vulkan viewport tuple (x, y, width, height)."""
    ctx = pyglet.graphics.api.core.current_window
    if hasattr(ctx.window, "viewport"):
        viewport = ctx.window.viewport
        if isinstance(viewport, tuple) and len(viewport) == 4:
            return viewport

    extent = ctx.swapchain.extent
    return 0, 0, int(extent.width), int(extent.height)


def get_screenshot() -> ImageData:
    """Read pixel data from the current Vulkan swapchain image into ImageData.

    This follows the OpenGL helper behavior closely, with Vulkan-specific
    deviations:
    1) the source is the most recent acquired/presented (or offscreen) image.
    2) swizzle conversion is applied for BGRA swapchain formats.
    """
    ctx = pyglet.graphics.api.core.current_window
    logical = ctx.logical_device
    swapchain = ctx.swapchain

    if swapchain is None:
        msg = "Vulkan screenshot capture requires an active swapchain."
        raise RuntimeError(msg)

    extent = swapchain.extent
    width = int(extent.width)
    height = int(extent.height)
    if width <= 0 or height <= 0:
        msg = "Cannot capture screenshot from a zero-sized framebuffer."
        raise RuntimeError(msg)

    logical.vkDeviceWaitIdle(logical.vk_device)

    frame_sync = ctx.frame_sync
    image_index = _resolve_framebuffer_image_index(frame_sync)
    image = swapchain.swapchain_images[image_index]
    source_layout = (
        VK_IMAGE_LAYOUT_PRESENT_SRC_KHR
        if logical.supports_presentation()
        else VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
    )

    pixel_size = 4
    byte_size = width * height * pixel_size
    staging = StagingBufferObject2(byte_size)
    staging.create(ctx.devices)

    pool = pyglet.graphics.api.core.command_pool
    command_buffer = pool.get_single_use(1)[0]

    try:
        with command_buffer as vk_command_buffer:
            to_transfer = VkImageMemoryBarrier(
                sType=VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                srcAccessMask=VK_ACCESS_MEMORY_READ_BIT,
                dstAccessMask=VK_ACCESS_TRANSFER_READ_BIT,
                oldLayout=source_layout,
                newLayout=VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                srcQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED,
                dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED,
                image=image,
                subresourceRange=VkImageSubresourceRange(
                    aspectMask=VK_IMAGE_ASPECT_COLOR_BIT,
                    baseMipLevel=0,
                    levelCount=1,
                    baseArrayLayer=0,
                    layerCount=1,
                ),
            )

            barrier_array = c_array_list([to_transfer], VkImageMemoryBarrier)
            DeviceFunc.vkCmdPipelineBarrier(
                vk_command_buffer,
                VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                0,
                0,
                None,
                0,
                None,
                1,
                barrier_array,
            )

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
                imageExtent=VkExtent3D(width=width, height=height, depth=1),
            )
            region_array = c_array_list([region], VkBufferImageCopy)
            DeviceFunc.vkCmdCopyImageToBuffer(
                vk_command_buffer,
                image,
                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                staging.buffer.vk_buffer,
                1,
                region_array,
            )

            to_present = VkImageMemoryBarrier(
                sType=VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                srcAccessMask=VK_ACCESS_TRANSFER_READ_BIT,
                dstAccessMask=VK_ACCESS_MEMORY_READ_BIT,
                oldLayout=VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                newLayout=source_layout,
                srcQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED,
                dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED,
                image=image,
                subresourceRange=VkImageSubresourceRange(
                    aspectMask=VK_IMAGE_ASPECT_COLOR_BIT,
                    baseMipLevel=0,
                    levelCount=1,
                    baseArrayLayer=0,
                    layerCount=1,
                ),
            )
            barrier_array = c_array_list([to_present], VkImageMemoryBarrier)
            DeviceFunc.vkCmdPipelineBarrier(
                vk_command_buffer,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                0,
                0,
                None,
                0,
                None,
                1,
                barrier_array,
            )

        raw = staging.get_bytes()
    finally:
        pool.free([command_buffer])
        staging.delete()

    vk_fmt = swapchain.surface_format.format
    if vk_fmt in (VK_FORMAT_B8G8R8A8_SRGB, VK_FORMAT_B8G8R8A8_UNORM):
        raw = _swizzle_bgra_to_rgba(raw)
    elif vk_fmt not in (VK_FORMAT_R8G8B8A8_SRGB, VK_FORMAT_R8G8B8A8_UNORM):
        # Most platforms expose BGRA/RGBA swapchain formats. Keep raw bytes for
        # other 4-channel layouts as a best-effort fallback.
        pass

    return ImageData(width, height, "RGBA", raw)


def get_max_color_attachments() -> int:
    """Return the max number of color attachments for the active Vulkan context."""
    return pyglet.graphics.api.core.current_window.info.MAX_COLOR_ATTACHMENTS


class VulkanRenderbuffer:
    """Vulkan renderbuffer compatibility class.

    Vulkan does not have a Renderbuffer, they just use Images and ImageViews.

    This class creates an attachment-only image view that can be
    attached to :class:`VulkanFramebuffer`.
    """

    def __init__(
        self,
        context: VulkanSurfaceContext | None,
        width: int,
        height: int,
        component_format: ComponentFormat,
        bit_size: int,
        data_type: DataTypes = "I",
        samples: int = 1,
    ) -> None:
        if samples != 1:
            msg = "VulkanRenderbuffer currently supports only samples=1."
            raise NotImplementedError(msg)

        self._context = _resolve_context(context)
        self._width = int(width)
        self._height = int(height)
        self._component_format = component_format
        self._bit_size = int(bit_size)
        self._data_type = data_type

        usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT
        aspect_mask = VK_IMAGE_ASPECT_COLOR_BIT
        if component_format == ComponentFormat.D:
            usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT
            aspect_mask = VK_IMAGE_ASPECT_DEPTH_BIT
        elif component_format == ComponentFormat.DS:
            usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT
            aspect_mask = VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT

        # Local import avoids a circular import at module load time.
        from pyglet.graphics.api.vulkan.texture import VulkanImage, VulkanImageView  # noqa: PLC0415

        self.image = VulkanImage(
            width=self._width,
            height=self._height,
            internal_format=component_format,
            internal_format_size=self._bit_size,
            internal_format_type=data_type,
            usage=usage,
        )
        self.image.create(self._context.devices)
        self.image_view = VulkanImageView(self._context.devices, self.image, aspect_mask=aspect_mask)

    @property
    def id(self) -> int:
        if self.image_view and self.image_view.vk_imageview:
            return int(self.image_view.vk_imageview.value)
        return 0

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    def bind(self) -> None:
        # Not a thing.
        return

    def unbind(self) -> None:
        return

    def delete(self) -> None:
        if self.image_view:
            self.image_view.delete()
            self.image_view = None
        if self.image:
            self.image.delete()
            self.image = None

    def __del__(self) -> None:
        try:
            self.delete()
        except Exception:
            pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self.id})"


class VulkanFramebuffer:
    """Vulkan framebuffer compatibility class with an OpenGL-style API."""

    _next_compat_id = 1

    def __init__(
        self,
        target: FramebufferTarget = FramebufferTarget.FRAMEBUFFER,
        context: VulkanSurfaceContext | None = None,
    ) -> None:
        """Create a compatibility framebuffer wrapper for Vulkan.

        Args:
            target:
                Binding target, not used for Vulkan.
                Only kept for parity with OpenGL API.
            context:
                Vulkan surface context that owns the logical device and active
                render state used to create/destroy the underlying
                ``VkFramebuffer``. A context is required because Vulkan
                resources are device-scoped.
        """
        self._context = _resolve_context(context)
        self.target = target
        self._render_pass = None
        self._id = 0

        self._framebuffer: VkFramebuffer | None = None
        self._attachments: dict[FramebufferAttachment, tuple[VkImageView, int, int, object]] = {}
        self._width = 0
        self._height = 0
        self._color_attachment_keys: tuple[FramebufferAttachment, ...] = ()
        self._depth_attachment_key: FramebufferAttachment | None = None
        self._owns_render_pass = False
        self._finalized = False

    @property
    def id(self) -> int:
        return int(self._id)

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    def _build_attachment_list(self) -> list[VkImageView]:
        renderpass_obj = _resolve_render_pass_object(self._render_pass, self._context)
        if renderpass_obj is None:
            return [self._attachments[attachment][0] for attachment in _attachment_order if attachment in self._attachments]

        color_keys = self._color_attachment_keys
        if not color_keys:
            color_count = len(getattr(renderpass_obj, "color_attachments", ()))
            color_keys = tuple(_color_attachment_order[:color_count])

        has_depth = self._depth_attachment_key is not None
        if not has_depth:
            has_depth = getattr(renderpass_obj, "depth_attachment", None) is not None
        ordered: list[VkImageView] = []

        for attachment in color_keys:
            ordered.append(self._attachments[attachment][0])

        if has_depth:
            depth_attachment = self._depth_attachment_key or FramebufferAttachment.DEPTH_STENCIL
            if depth_attachment not in self._attachments:
                depth_attachment = FramebufferAttachment.DEPTH
            ordered.append(self._attachments[depth_attachment][0])
        elif FramebufferAttachment.DEPTH_STENCIL in self._attachments:
            ordered.append(self._attachments[FramebufferAttachment.DEPTH_STENCIL][0])
        elif FramebufferAttachment.DEPTH in self._attachments:
            ordered.append(self._attachments[FramebufferAttachment.DEPTH][0])
        elif FramebufferAttachment.STENCIL in self._attachments:
            ordered.append(self._attachments[FramebufferAttachment.STENCIL][0])

        return ordered

    def _attachment_error(self) -> str | None:
        if not self._attachments:
            return "Framebuffer missing attachment."
        if self._width <= 0 or self._height <= 0:
            return "Framebuffer has invalid attachment dimensions."

        # Offscreen framebuffers create their own compatible render pass in
        # ``finalize``.  The window render pass may have different attachments.
        renderpass_obj = self._render_pass
        attached_color_keys = tuple(a for a in _color_attachment_order if a in self._attachments)
        expected_color_keys = self._color_attachment_keys or attached_color_keys
        if renderpass_obj is not None and not self._color_attachment_keys:
            color_count = len(getattr(renderpass_obj, "color_attachments", ()))
            if color_count:
                expected_color_keys = tuple(_color_attachment_order[:color_count])

        for attachment in expected_color_keys:
            if attachment not in self._attachments:
                return "Framebuffer missing required color attachment."

        if renderpass_obj is not None and len(attached_color_keys) > len(expected_color_keys):
            return "Framebuffer has unsupported color attachment count for render pass."

        for attachment, (_, _, _, resource) in self._attachments.items():
            if attachment not in _color_attachment_order:
                continue
            image = getattr(resource, "image", None)
            usage = getattr(image, "usage", None)
            if usage is not None and (usage & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT) == 0:
                return "Color attachment texture is missing VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT."
            if renderpass_obj is not None and attachment in expected_color_keys:
                idx = expected_color_keys.index(attachment)
                rp_color_attachments = getattr(renderpass_obj, "color_attachments", ())
                if idx < len(rp_color_attachments):
                    vk_fmt = _get_attachment_format(resource)
                    rp_fmt = getattr(rp_color_attachments[idx], "fmt", None)
                    if vk_fmt is not None and rp_fmt is not None and int(vk_fmt) != int(rp_fmt):
                        return "Framebuffer color attachment format does not match render pass format."

        has_depth = self._depth_attachment_key is not None
        if renderpass_obj is not None and not has_depth:
            has_depth = getattr(renderpass_obj, "depth_attachment", None) is not None
        has_depth_attachment = (
            FramebufferAttachment.DEPTH in self._attachments
            or FramebufferAttachment.DEPTH_STENCIL in self._attachments
            or FramebufferAttachment.STENCIL in self._attachments
        )
        if has_depth and not has_depth_attachment:
            return "Framebuffer missing required depth attachment."
        if renderpass_obj is not None and not has_depth and has_depth_attachment:
            return "Framebuffer has depth attachment, but current render pass has no depth attachment."

        if has_depth_attachment:
            depth_key = self._depth_attachment_key or _get_depth_attachment_key(self._attachments)
            if depth_key is not None:
                depth_resource = self._attachments[depth_key][3]
                image = getattr(depth_resource, "image", None)
                usage = getattr(image, "usage", None)
                if usage is not None and (usage & VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT) == 0:
                    return "Depth attachment texture is missing VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT."

        return None

    def _create_compatible_render_pass(self) -> None:
        if self._render_pass is not None:
            return

        color_keys = tuple(attachment for attachment in _color_attachment_order if attachment in self._attachments)
        depth_key = _get_depth_attachment_key(self._attachments)
        if not color_keys and depth_key is None:
            msg = "Framebuffer missing attachment."
            raise RuntimeError(msg)

        color_formats: tuple[int, ...] = tuple(
            _get_attachment_format(self._attachments[attachment][3]) for attachment in color_keys
        )
        if any(fmt is None for fmt in color_formats):
            msg = "Color attachment is missing a Vulkan format descriptor."
            raise RuntimeError(msg)

        depth_format = None
        if depth_key is not None:
            depth_format = _get_attachment_format(self._attachments[depth_key][3])
            if depth_format is None:
                msg = "Depth attachment is missing a Vulkan format descriptor."
                raise RuntimeError(msg)

        # Local import avoids circular import at module load time.
        from pyglet.graphics.api.vulkan.renderpass import RenderPass  # noqa: PLC0415

        self._color_attachment_keys = color_keys
        self._depth_attachment_key = depth_key
        self._render_pass = RenderPass(
            self._context.logical_device,
            color_formats=color_formats,
            depth_format=depth_format,
            offscreen=True,
        )
        self._owns_render_pass = True

    def _create_framebuffer(self) -> None:
        error = self._attachment_error()
        if error is not None:
            raise RuntimeError(error)

        render_pass_handle = _resolve_render_pass_handle(self._render_pass, self._context)
        attachments = self._build_attachment_list()
        view_array = c_array_list(attachments, VkImageView)

        framebuffer_create = VkFramebufferCreateInfo(
            sType=VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
            flags=0,
            renderPass=render_pass_handle,
            attachmentCount=len(attachments),
            pAttachments=view_array,
            width=self._width,
            height=self._height,
            layers=1,
        )

        framebuffer = VkFramebuffer()
        try:
            DeviceFunc.vkCreateFramebuffer(
                self._context.logical_device.vk_device,
                byref(framebuffer_create),
                None,
                byref(framebuffer),
            )
        except Exception as exc:
            msg = f"Failed to create Vulkan framebuffer: {exc}"
            raise RuntimeError(msg) from exc
        self._framebuffer = framebuffer
        self._id = self._framebuffer.value

    def _ensure_mutable(self) -> None:
        if self._finalized:
            msg = "Framebuffer attachments are immutable after finalize()."
            raise RuntimeError(msg)

    def finalize(self) -> None:
        """Create and freeze this framebuffer for subsequent binds."""
        if self._finalized:
            return
        if self._id == 0:
            self._id = type(self)._next_compat_id
            type(self)._next_compat_id += 1
        self._create_compatible_render_pass()
        self._create_framebuffer()
        self._finalized = True

    def bind(self) -> None:
        """Mark this finalized framebuffer as active for compatibility.

        Vulkan does not expose OpenGL-style global framebuffer binding.
        """
        if not self._finalized:
            msg = "Framebuffer must be finalized before it can be bound."
            raise RuntimeError(msg)

    def unbind(self) -> None:
        """Unbind."""

    def clear(self) -> None:
        """Compatibility no-op.

        Vulkan clear operations are applied during render pass begin.
        """
        return

    def delete(self) -> None:
        if self._framebuffer is not None:
            DeviceFunc.vkDestroyFramebuffer(self._context.logical_device.vk_device, self._framebuffer, None)
            self._framebuffer = None
        if self._owns_render_pass and self._render_pass is not None and hasattr(self._render_pass, "delete"):
            self._render_pass.delete()
            self._render_pass = None
            self._owns_render_pass = False
        self._color_attachment_keys = ()
        self._depth_attachment_key = None
        self._id = 0
        self._finalized = False

    def __del__(self) -> None:
        try:
            self.delete()
        except Exception:
            pass

    @property
    def is_complete(self) -> bool:
        return self._attachment_error() is None

    def get_status(self) -> str:
        error = self._attachment_error()
        if error:
            return error
        return "Framebuffer is complete."

    def attach_texture(
        self,
        texture: VulkanTexture,
        attachment: FramebufferAttachment = FramebufferAttachment.COLOR0,
        level: int = 0,
    ) -> None:
        self._ensure_mutable()
        if level != 0:
            msg = "VulkanFramebuffer currently supports only level=0 attachments."
            raise NotImplementedError(msg)

        view = _get_view_from_resource(texture)
        width, height = _get_attachment_size(texture)
        if self._width <= 0:
            self._width = width
        if self._height <= 0:
            self._height = height
        else:
            assert self._width == width, f"Width: {width} does not match current framebuffer width: {self._width}."
            assert self._height == height, f"Height: {height} does not match current framebuffer height: {self._height}."

        self._attachments[attachment] = (view, width, height, texture)

    def attach_texture_layer(
        self,
        texture,
        layer: int,
        level: int,
        attachment: FramebufferAttachment = FramebufferAttachment.COLOR0,
    ) -> None:
        self._ensure_mutable()
        if layer != 0:
            msg = "VulkanFramebuffer currently supports only layer=0 for attach_texture_layer."
            raise NotImplementedError(msg)
        self.attach_texture(texture, attachment=attachment, level=level)

    def attach_renderbuffer(
        self,
        renderbuffer: VulkanRenderbuffer,
        attachment: FramebufferAttachment = FramebufferAttachment.COLOR0,
    ) -> None:
        self._ensure_mutable()
        view = _get_view_from_resource(renderbuffer)
        width, height = _get_attachment_size(renderbuffer)
        if width <= 0:
            width = self._width
        if height <= 0:
            height = self._height
        self._attachments[attachment] = (view, width, height, renderbuffer)
        self._width = max(self._width, width)
        self._height = max(self._height, height)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self.id})"
