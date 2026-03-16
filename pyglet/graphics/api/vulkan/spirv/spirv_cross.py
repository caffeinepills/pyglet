from __future__ import annotations

import json
import struct
from ctypes import byref, c_char_p

from pyglet.libs.shared.spirv.lib_spirv_cross import spvc_parsed_ir, spvc_context_parse_spirv, spvc_context, \
    spvc_context_create, spvc_context_destroy, spvc_compiler, spvc_context_create_compiler, spvc_backend, \
    spvc_compiler_compile, SPVC_BACKEND_GLSL, spvc_context_release_allocations, SPVC_BACKEND_JSON, SPVC_SUCCESS, SpvId

class SPIRVCross:
    def __init__(self):
        self.context = spvc_context()
        if result := spvc_context_create(byref(self.context)) != SPVC_SUCCESS:
            raise RuntimeError("Failed to create SPIRV-Cross context.", result)

    def clean(self):
        """Cleans, but doesn't terminate the context."""
        spvc_context_release_allocations(self.context)

    def get_json(self, data: bytes) -> dict:
        parsed_ir = self._get_parsed_ir(data)
        compiler = self._get_compiler(parsed_ir, SPVC_BACKEND_JSON)

        result = c_char_p()
        spvc_compiler_compile(compiler, byref(result))
        return json.loads(result.value.decode('utf-8'))

    def get_glsl(self, data: bytes, version=None):
        parsed_ir = self._get_parsed_ir(data)
        compiler = self._get_compiler(parsed_ir, SPVC_BACKEND_GLSL)

        result = c_char_p()
        spvc_compiler_compile(compiler, byref(result))
        return result.value.decode('utf-8')

    def _get_compiler(self, parsed_ir: spvc_parsed_ir, compiler_backend: int | spvc_backend):
        compiler = spvc_compiler()
        if result := (spvc_context_create_compiler(self.context, compiler_backend, parsed_ir, 0, byref(compiler))
                      != SPVC_SUCCESS):
            raise RuntimeError("Failed to create SPIRV-Cross compiler.", result)
        return compiler

    def _get_parsed_ir(self, data: bytes) -> spvc_parsed_ir:
        # Convert the raw bytes to a list of uint32_t, which is required for the parser.
        spirv_binary = list(struct.unpack(f"{len(data) // 4}I", data))

        parsed_ir = spvc_parsed_ir()
        if result := spvc_context_parse_spirv(
            self.context,
            (SpvId * len(spirv_binary))(*spirv_binary),
            len(spirv_binary),
            byref(parsed_ir),
        ) != SPVC_SUCCESS:
            raise RuntimeError("Failed to parse SPIR-V binary.", result)
        return parsed_ir

    def __del__(self) -> None:
        if self.context:
            spvc_context_destroy(self.context)
            self.context = None


_spirv_cross = SPIRVCross()


def get_json(spirv_data: bytes) -> dict:
    _data = _spirv_cross.get_json(spirv_data)
    _spirv_cross.clean()
    return _data

def get_glsl(spirv_data: bytes) -> str:
    _data = _spirv_cross.get_glsl(spirv_data)
    _spirv_cross.clean()
    return _data