from __future__ import annotations


class Framebuffer:
    def __init__(self, device, render_pass, swapchain):
        self.device = device

        assert swapchain.image_views
        assert swapchain.extent
        self.framebuffers = self.create_framebuffers(render_pass.render_pass, swapchain.image_views, swapchain.extent)

    @property
    def count(self):
        return len(self.framebuffers)

    def create_framebuffers(self, render_pass, image_views, extent):
        """Create framebuffers for each image view.
        Each framebuffer corresponds to a swapchain image and the attachments of the render pass.
        """
        framebuffers = []
        for image_view in image_views:
            attachments = [image_view]

            framebuffer_info = VkFramebufferCreateInfo(
                sType=VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
                renderPass=render_pass,
                attachmentCount=len(attachments),
                pAttachments=attachments,
                width=extent.width,
                height=extent.height,
                layers=1,
            )
            framebuffer = vkCreateFramebuffer(self.device.logical_device, framebuffer_info, None)
            framebuffers.append(framebuffer)

        return framebuffers

    def delete(self):
        """Clean up the framebuffers.
        """
        for framebuffer in self.framebuffers:
            vkDestroyFramebuffer(self.device, framebuffer, None)
        print("DELETED?")
        self.framebuffers = []



class VulkanFramebuffer:
    def __init__(self, device, render_pass, width, height):
        """Initialize the VulkanFramebuffer without creating the framebuffer yet."""
        self.device = device
        self.render_pass = render_pass
        self.width = width
        self.height = height
        self.attachments = []  # List to store attachments (VkImageView)
        self.framebuffer = None  # Vulkan framebuffer object (will be created later)

    def attach_texture(self, image_view):
        """Attach a texture (VkImageView) to the framebuffer."""
        self.attachments.append(image_view)

    def create(self):
        """Create the Vulkan framebuffer using the current attachments."""
        if self.framebuffer:
            raise RuntimeError("Framebuffer has already been created.")

        if not self.attachments:
            raise RuntimeError("No attachments have been added to the framebuffer.")

        # Create the framebuffer with the current attachments
        framebuffer_info = vk.VkFramebufferCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
            renderPass=self.render_pass,
            attachmentCount=len(self.attachments),
            pAttachments=self.attachments,
            width=self.width,
            height=self.height,
            layers=1,  # Typically 1 layer for 2D framebuffers
        )

        self.framebuffer = vk.vkCreateFramebuffer(self.device, framebuffer_info, None)

    def clear(self):
        """Clear operation is handled via render pass begin info in Vulkan."""
        # Clear is done through VkClearValue and VkRenderPassBeginInfo.

    def delete(self):
        """Delete the Vulkan framebuffer."""
        if self.framebuffer:
            vk.vkDestroyFramebuffer(self.device, self.framebuffer, None)
            self.framebuffer = None

    def __del__(self):
        """Ensure resources are cleaned up when the object is destroyed."""
        if self.framebuffer:
            self.delete()

    def is_complete(self):
        """Check if the framebuffer has been created (considered 'complete')."""
        return self.framebuffer is not None

    def __repr__(self):
        return f"{self.__class__.__name__}(framebuffer={self.framebuffer})"


# Example usage:

# Assume we have a Vulkan logical device, render pass, and image views (textures)
def create_framebuffer_example(device, render_pass, color_attachment, depth_attachment):
    # Create the framebuffer object
    framebuffer = VulkanFramebuffer(device, render_pass, 800, 600)

    # Attach image views (VkImageView for color, depth attachments)
    framebuffer.attach_texture(color_attachment)
    framebuffer.attach_texture(depth_attachment)

    # Now that all attachments are added, create the framebuffer
    framebuffer.create()

    return framebuffer


class VulkanRenderbuffer:
    ...
