from pyglet.libs.shared.vulkan_lib import *
from .shaders import *


def create_semaphore(logical_device):
    semaphore_create = VkSemaphoreCreateInfo(
        sType=VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
        flags=0)
    return vkCreateSemaphore(logical_device, semaphore_create, None)




def find_memory_type(physical_device, type_filter, properties):
    mem_properties = vkGetPhysicalDeviceMemoryProperties(physical_device)

    for i in range(mem_properties.memoryTypeCount):
        if (type_filter & (1 << i)) and (mem_properties.memoryTypes[i].propertyFlags & properties) == properties:
            return i

    raise Exception("Failed to find suitable memory type")

def create_staging_buffer(device, physical_device, command_pool, queue, data, size, usage):
    # Create a staging buffer in host-visible memory (CPU accessible)
    staging_buffer, staging_buffer_memory = create_buffer(
        device, physical_device, size,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
    )

    # Map memory and copy data to the staging buffer
    data_ptr = vkMapMemory(device, staging_buffer_memory, 0, size, 0)
    data_ptr[:] = data
    #ctypes.memmove(data_ptr, data, size)
    vkUnmapMemory(device, staging_buffer_memory)

    # Create the destination buffer in device-local memory (GPU accessible)
    buffer, buffer_memory = create_buffer(
        device, physical_device, size,
        usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
    )

    # Copy the data from the staging buffer to the destination buffer
    copy_buffer(device, command_pool, queue, staging_buffer, buffer, size)

    # Cleanup staging buffer
    vkDestroyBuffer(device, staging_buffer, None)
    vkFreeMemory(device, staging_buffer_memory, None)

    return buffer, buffer_memory

def copy_buffer(device, command_pool, queue, src_buffer, dst_buffer, size):
    command_buffer = begin_single_time_commands(device, command_pool)

    copy_region = VkBufferCopy(
        srcOffset=0,
        dstOffset=0,
        size=size
    )

    vkCmdCopyBuffer(command_buffer, src_buffer, dst_buffer, 1, [copy_region])

    end_single_time_commands(device, command_pool, queue, command_buffer)
    
def begin_single_time_commands(device, command_pool):
    alloc_info = VkCommandBufferAllocateInfo(
        sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        level=VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        commandPool=command_pool,
        commandBufferCount=1
    )
    command_buffer = vkAllocateCommandBuffers(device, alloc_info)[0]

    begin_info = VkCommandBufferBeginInfo(
        sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
    )
    vkBeginCommandBuffer(command_buffer, begin_info)

    return command_buffer

def end_single_time_commands(device, command_pool, queue, command_buffer):
    vkEndCommandBuffer(command_buffer)

    submit_info = VkSubmitInfo(
        sType=VK_STRUCTURE_TYPE_SUBMIT_INFO,
        commandBufferCount=1,
        pCommandBuffers=[command_buffer]
    )

    vkQueueSubmit(queue, 1, [submit_info], VK_NULL_HANDLE)
    vkQueueWaitIdle(queue)

    vkFreeCommandBuffers(device, command_pool, 1, [command_buffer])
    
def create_ubo_buffer(device, physical_device, ubo_data):
    # Get the size of the UBO structure
    buffer_size = ctypes.sizeof(ubo_data)

    # Create the UBO buffer (host-visible)
    ubo_buffer, ubo_buffer_memory = create_buffer(
        device, physical_device, buffer_size,
        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
    )

    return ubo_buffer, ubo_buffer_memory


def update_ubo(device, ubo_buffer_memory, ubo_data):
    # Map memory and copy UBO data into the mapped memory
    data_ptr = vkMapMemory(device, ubo_buffer_memory, 0, ctypes.sizeof(ubo_data), 0)

    # Copy the UBO data to the mapped memory
    # ctypes.memmove(data_ptr, ctypes.byref(ubo_data), ctypes.sizeof(ubo_data))
    data_ptr[:] = ubo_data
    print("UPDATING!", data_ptr[:])

    # Unmap the memory when done
    vkUnmapMemory(device, ubo_buffer_memory)