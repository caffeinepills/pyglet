from abc import ABC, abstractmethod
from typing import AnyStr, Any

from pyglet.graphics.api.vulkan.devices import VulkanLogicalDevice


class FrameObject(ABC):
    """Abstract base class for managing per-frame resources."""
    def __init__(self, device: VulkanLogicalDevice, frames_in_flight: int) -> None:
        self.device = device
        self.frames_in_flight = frames_in_flight
        self.current_frame = 0  # Tracks the current frame in flight
        self.resources = [self.create_resource() for _ in range(frames_in_flight)]
        self.dirty = [True] * frames_in_flight

    @abstractmethod
    def create_resource(self):
        """Create a resource for each frame. Subclasses must implement this method.

        :return: The resource for a frame (e.g., UBO, texture, etc.).
        """

    @abstractmethod
    def update(self, new_data: Any):
        """Update the resource for the next frame. Subclasses must implement this method.

        :param new_data: The new data to update the resource with.
        """

class BufferedObject(FrameObject):
    def __init__(self, device, frames_in_flight, initial_data, buffer_size=256):
        """
        Manage per-frame UBOs with logical state abstraction.

        :param device: The Vulkan logical device.
        :param frames_in_flight: Number of frames in flight.
        :param initial_data: Initial logical data for the object.
        :param buffer_size: Size of each UBO (default: 256 bytes).
        """
        self.data = initial_data  # Logical state of the object
        self.buffer_size = buffer_size
        super().__init__(device, frames_in_flight)

        # Initialize all UBOs with the initial data
        for resource in self.resources:
            self.update_ubo(resource, initial_data)

    def create_resource(self):
        """
        Create a UBO for each frame.
        """
        return allocate_buffer(
            self.device,
            size=self.buffer_size,
            usage=VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT
        )

    def update(self, new_data):
        """
        Update the logical data of the object and ensure the UBO for the next frame is updated.
        """
        self.data = new_data  # Update the logical state

        # Determine the UBO for the next frame
        next_frame = (self.current_frame + 1) % self.frames_in_flight
        next_ubo = self.resources[next_frame]

        # Update the UBO for the next frame
        self.update_ubo(next_ubo, new_data)

    def update_ubo(self, ubo, data):
        """
        Update the contents of a specific UBO with new data.

        :param ubo: The UBO to update.
        :param data: The data to write into the UBO.
        """
        vkMapMemory(self.device.vk_device, ubo.vk_device_memory, 0, len(data), 0, byref(mapped_memory))
        memcpy(mapped_memory, data, len(data))
        vkUnmapMemory(self.device.vk_device, ubo.vk_device_memory)

