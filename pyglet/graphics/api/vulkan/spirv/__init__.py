from __future__ import annotations

from typing import TYPE_CHECKING
import warnings
import struct
import pyglet

if TYPE_CHECKING:
    from pyglet.graphics.shader import ShaderType

class ShaderCompilationError(Exception):
    ...

INSPECTION_AVAILABLE = False
COMPILATION_AVAILABLE = False

_debug_lib = pyglet.options.debug_lib

try:
    from pyglet.graphics.api.vulkan.spirv.spirv_cross import get_json, get_glsl
    INSPECTION_AVAILABLE = True
except ImportError as err:
    if _debug_lib:
        warnings.warn(f"spirv_cross could not be loaded. Application inspection unavailable. {err}\n"
                      f"Verify you have the Vulkan SDK installed or the library in a path environment.")

try:
    from pyglet.graphics.api.vulkan.spirv.shaderc import compile_glsl_to_spirv
    COMPILATION_AVAILABLE = True
except ImportError as err:
    if _debug_lib:
        warnings.warn(f"shaderc lib could not be loaded. Application compilation unavailable. {err}\n"
                      f"Verify you have the Vulkan SDK installed or the library in a path environment.")
    compile_glsl_to_spirv = None



def compile_shader(shader_source: str, shader_type: ShaderType, output_filename: str | None=None,
                   entry_point: str="main") -> bytes:
    """Compiles GLSL shader source into SPIR-V using the shaderc_shared library.

    The library is included with the VulkanSDK and is useful for troubleshooting or development. You could potentially
    ship your application with this library, especially if you require dynamic shader generation.

    However, for cross-platform compatibility it may be best to compile your shaders into SPIR-V files. This can be
    done by providing a file location string to this function.
    """
    assert COMPILATION_AVAILABLE, "Can not compile shaders without a compilation method."

    spirv_binary = compile_glsl_to_spirv(shader_source, shader_type, entry_point=entry_point)

    if output_filename:
        with open(output_filename, 'wb') as f:
            f.write(spirv_binary)

    return spirv_binary

def get_spirv_inspection_json(spirv_binary: bytes) -> dict:
    """Inspects the SPIR-V binary data using the spirv_cross-shared library with SPIRV-Reflect output.

    The library is included with the VulkanSDK and is useful for troubleshooting or development. You could potentially
    ship your application with this library, especially if you require dynamic shader generation.

    However, for cross-platform compatibility it may be best to manually define your shader information on the
    ShaderProgram itself using Python methods.

    As a helper function, we have provided ShaderProgram.load_spirv_json function, to define your shaders using
    the output of this function. (Alternatively, you can use SPIRV-Reflect to get the same output). This may be an
    easier method than manually defining things or risking invalid input.
    """
    assert INSPECTION_AVAILABLE, "Can not inspect SPIR-V without an inspection method."
    return get_json(spirv_binary)

def get_spirv_as_glsl(spirv_binary: bytes) -> str:
    """Inspect's the SPIR-V binary data using the spirv_cross-shared library with SPIRV-Reflect output.

    The library is included with the VulkanSDK and is useful for troubleshooting or development.

    This is useful to verify the shader was converted properly, or to inspect what information the SPIR-V conversion
    may have optimized out.
    """
    assert INSPECTION_AVAILABLE, "Can not inspect SPIR-V without an inspection method."
    return get_glsl(spirv_binary)

def validate_spirv(data: bytes) -> bool:
    """Preliminary stage to verify if it is actually SPIR-V."""
    if len(data) % 4 != 0:
        if _debug_lib:
            warnings.warn("SPIR-V binary size is not aligned to 4 bytes.")
        return False

    # Check the magic number (first 4 bytes)
    magic_number = struct.unpack("I", data[:4])[0]
    if magic_number != 0x07230203:
        if _debug_lib:
            warnings.warn("Invalid SPIR-V magic number.")
        return False
    return True