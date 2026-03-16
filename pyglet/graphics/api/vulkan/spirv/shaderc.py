from __future__ import annotations

import ctypes
from ctypes import c_char_p
from enum import Enum
from typing import TYPE_CHECKING

from pyglet.graphics.api.vulkan.spirv import ShaderCompilationError
from pyglet.libs.shared.spirv.lib_shaderc import (
    shaderc_compile_options_initialize,
    shaderc_compile_options_set_generate_debug_info,
    shaderc_compile_options_set_optimization_level,
    shaderc_compile_into_spv, shaderc_result_get_bytes,
    shaderc_result_get_length, shaderc_result_release,
    shaderc_result_get_error_message,
    shaderc_result_get_compilation_status,
    shaderc_compiler_release,
    shaderc_vertex_shader, shaderc_fragment_shader,
    shaderc_geometry_shader, shaderc_compute_shader,
    shaderc_tess_control_shader,
    shaderc_tess_evaluation_shader,
    shaderc_compiler_initialize,
    shaderc_optimization_level_performance, shaderc_optimization_level_zero, shaderc_optimization_level_size,

)

if TYPE_CHECKING:
    from pyglet.graphics.shader import ShaderType


_shader_type_sharderc_type: dict[ShaderType, int] = {
    'vertex': shaderc_vertex_shader,
    'fragment': shaderc_fragment_shader,
    'geometry': shaderc_geometry_shader,
    'compute': shaderc_compute_shader,
    'tesscontrol': shaderc_tess_control_shader,
    'tessevaluation': shaderc_tess_evaluation_shader,
}


class ShaderOptimization(Enum):
    NONE = "none"
    SIZE = "size"
    PERFORMANCE = "performance"


_OPTIMIZE_MAP = {
    ShaderOptimization.NONE: shaderc_optimization_level_zero,
    ShaderOptimization.SIZE: shaderc_optimization_level_size,
    ShaderOptimization.PERFORMANCE: shaderc_optimization_level_performance,
}

def compile_glsl_to_spirv(source_code: str | bytes, shader_type: ShaderType, entry_point: str = "main",
                          debug: bool=False, optimize: ShaderOptimization = ShaderOptimization.NONE) -> bytes:
    compiler = shaderc_compiler_initialize()
    if not compiler:
        raise ShaderCompilationError("Failed to initialize Shaderc compiler.")

    if isinstance(source_code, str):
        source_code = source_code.encode("utf-8")

    source = c_char_p(source_code)

    # Initialize options (could also be None for no options.)
    options = shaderc_compile_options_initialize()
    if debug:
        shaderc_compile_options_set_generate_debug_info(options)

    shaderc_compile_options_set_optimization_level(options, _OPTIMIZE_MAP[optimize])

    # Compile GLSL to SPIR-V
    result = shaderc_compile_into_spv(
        compiler,
        source,
        len(source_code),
        _shader_type_sharderc_type[shader_type],
        b"shader.glsl",
        entry_point.encode("utf-8"),  # Entry point
        options,
    )

    if not result:
        shaderc_compiler_release(compiler)
        raise ShaderCompilationError("Compilation failed and no result returned.")

    shaderc_result_get_compilation_status(result)

    error_message = shaderc_result_get_error_message(result).decode("utf-8")
    if error_message:
        shaderc_result_release(result)
        shaderc_compiler_release(compiler)
        msg = f"Compilation failed: {error_message}"
        raise ShaderCompilationError(msg)

    spirv_pointer = shaderc_result_get_bytes(result)
    spirv_length = shaderc_result_get_length(result)
    spirv_binary = ctypes.string_at(spirv_pointer, spirv_length)

    # Cleanup
    shaderc_result_release(result)
    shaderc_compiler_release(compiler)
    return spirv_binary
