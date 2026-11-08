from __future__ import annotations

from pyglet.enums import (
    GeometryMode,
    BlendFactor,
    BlendOp,
    TextureType,
    TextureFilter,
    AddressMode,
    TextureWrapping,
    CompareOp,
)
from pyglet.libs.shared.vulkan_lib import vulkan_core as vk

geometry_map = {
    GeometryMode.POINTS: vk.VK_PRIMITIVE_TOPOLOGY_POINT_LIST,
    GeometryMode.LINES: vk.VK_PRIMITIVE_TOPOLOGY_LINE_LIST,
    GeometryMode.LINE_STRIP: vk.VK_PRIMITIVE_TOPOLOGY_LINE_STRIP,
    GeometryMode.TRIANGLES: vk.VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
    GeometryMode.TRIANGLE_STRIP: vk.VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP,
    GeometryMode.TRIANGLE_FAN: vk.VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN,
}



BLEND_FACTOR_MAP = {
    BlendFactor.ZERO: vk.VK_BLEND_FACTOR_ZERO,
    BlendFactor.ONE: vk.VK_BLEND_FACTOR_ONE,
    BlendFactor.SRC_COLOR: vk.VK_BLEND_FACTOR_SRC_COLOR,
    BlendFactor.ONE_MINUS_SRC_COLOR: vk.VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR,
    BlendFactor.DST_COLOR: vk.VK_BLEND_FACTOR_DST_COLOR,
    BlendFactor.ONE_MINUS_DST_COLOR: vk.VK_BLEND_FACTOR_ONE_MINUS_DST_COLOR,
    BlendFactor.SRC_ALPHA: vk.VK_BLEND_FACTOR_SRC_ALPHA,
    BlendFactor.ONE_MINUS_SRC_ALPHA: vk.VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
    BlendFactor.DST_ALPHA: vk.VK_BLEND_FACTOR_DST_ALPHA,
    BlendFactor.ONE_MINUS_DST_ALPHA: vk.VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA,
    BlendFactor.CONSTANT_COLOR: vk.VK_BLEND_FACTOR_CONSTANT_COLOR,
    BlendFactor.ONE_MINUS_CONSTANT_COLOR: vk.VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_COLOR,
    BlendFactor.CONSTANT_ALPHA: vk.VK_BLEND_FACTOR_CONSTANT_ALPHA,
    BlendFactor.ONE_MINUS_CONSTANT_ALPHA: vk.VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_ALPHA,
}

BLEND_OP_MAP = {
    BlendOp.ADD: vk.VK_BLEND_OP_ADD,
    BlendOp.SUBTRACT: vk.VK_BLEND_OP_SUBTRACT,
    BlendOp.REVERSE_SUBTRACT: vk.VK_BLEND_OP_REVERSE_SUBTRACT,
    BlendOp.MIN: vk.VK_BLEND_OP_MIN,
    BlendOp.MAX: vk.VK_BLEND_OP_MAX,
}

compare_op_map = {
    CompareOp.NEVER: vk.VK_COMPARE_OP_NEVER,
    CompareOp.LESS: vk.VK_COMPARE_OP_LESS,
    CompareOp.EQUAL: vk.VK_COMPARE_OP_EQUAL,
    CompareOp.LESS_OR_EQUAL: vk.VK_COMPARE_OP_LESS_OR_EQUAL,
    CompareOp.GREATER: vk.VK_COMPARE_OP_GREATER,
    CompareOp.NOT_EQUAL: vk.VK_COMPARE_OP_NOT_EQUAL,
    CompareOp.GREATER_OR_EQUAL: vk.VK_COMPARE_OP_GREATER_OR_EQUAL,
    CompareOp.ALWAYS: vk.VK_COMPARE_OP_ALWAYS,
}


TEXTURE_TYPE_MAP = {
    TextureType.TYPE_1D: vk.VK_IMAGE_TYPE_1D,
    TextureType.TYPE_2D: vk.VK_IMAGE_TYPE_2D,
    TextureType.TYPE_3D: vk.VK_IMAGE_TYPE_3D,
    TextureType.TYPE_CUBE_MAP: vk.VK_IMAGE_TYPE_2D,
    TextureType.TYPE_1D_ARRAY: vk.VK_IMAGE_TYPE_1D,
    TextureType.TYPE_2D_ARRAY: vk.VK_IMAGE_TYPE_2D,
    TextureType.TYPE_CUBE_MAP_ARRAY: vk.VK_IMAGE_TYPE_2D,
}

# Mapping for TextureType to Vulkan equivalents
IMAGE_VIEW_TYPE_MAP = {
    TextureType.TYPE_1D: vk.VK_IMAGE_VIEW_TYPE_1D,
    TextureType.TYPE_2D: vk.VK_IMAGE_VIEW_TYPE_2D,
    TextureType.TYPE_3D: vk.VK_IMAGE_VIEW_TYPE_3D,
    TextureType.TYPE_CUBE_MAP: vk.VK_IMAGE_VIEW_TYPE_CUBE,
    TextureType.TYPE_1D_ARRAY: vk.VK_IMAGE_VIEW_TYPE_1D_ARRAY,
    TextureType.TYPE_2D_ARRAY: vk.VK_IMAGE_VIEW_TYPE_2D_ARRAY,
    TextureType.TYPE_CUBE_MAP_ARRAY: vk.VK_IMAGE_VIEW_TYPE_CUBE_ARRAY,
}


# Mapping for TextureFilter to Vulkan equivalents
TEXTURE_FILTER_MAP = {
    TextureFilter.LINEAR: vk.VK_FILTER_LINEAR,
    TextureFilter.NEAREST: vk.VK_FILTER_NEAREST,
}

# Mapping for AddressMode to Vulkan equivalents
ADDRESS_MODE_MAP = {
    AddressMode.REPEAT: vk.VK_SAMPLER_ADDRESS_MODE_REPEAT,
    AddressMode.MIRRORED_REPEAT: vk.VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT,
    AddressMode.CLAMP_TO_EDGE: vk.VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
    AddressMode.CLAMP_TO_BORDER: vk.VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER,
}

# Mapping for TextureWrapping to Vulkan equivalents
TEXTURE_WRAPPING_MAP = {
    TextureWrapping.WRAP_S: "U",  # Vulkan uses U, V, W instead of S, T, R
    TextureWrapping.WRAP_T: "V",
    TextureWrapping.WRAP_R: "W",
}
