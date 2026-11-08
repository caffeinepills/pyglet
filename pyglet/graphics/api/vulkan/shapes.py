from __future__ import annotations
import pyglet
from pyglet.enums import BlendFactor
from pyglet.graphics.api.vulkan.shader import WindowBlock
from pyglet.graphics.draw import Group

from typing import TYPE_CHECKING

from pyglet.graphics.shader import Attribute

if TYPE_CHECKING:
    from pyglet.graphics.api.vulkan.shader import VulkanShaderProgram

vertex_src = """#version 450 core
layout(location = 0) in vec2 position;
layout(location = 1) in vec3 translation;
layout(location = 2) in vec4 colors;
layout(location = 3) in float rotation;

layout(location = 0) out vec4 vertex_colors;

layout(set = 0, binding = 0) uniform WindowBlock {
    mat4 projection;
    mat4 view;
} window;

void main() {
    mat4 m_rotation = mat4(1.0);
    mat4 m_translate = mat4(1.0);

    m_translate[3][0] = translation.x;
    m_translate[3][1] = translation.y;
    m_rotation[0][0] =  cos(-radians(rotation));
    m_rotation[0][1] =  sin(-radians(rotation));
    m_rotation[1][0] = -sin(-radians(rotation));
    m_rotation[1][1] =  cos(-radians(rotation));

    gl_Position = window.projection * window.view * m_translate * m_rotation * vec4(position, translation.z, 1.0);
    vertex_colors = colors;
}
"""

# Fragment shader source code as a string
frag_src = """#version 450 core
layout(location = 0) in vec4 vertex_colors;
layout(location = 0) out vec4 final_color;

void main() {
    final_color = vertex_colors;
}
"""

def get_default_shader() -> VulkanShaderProgram:
    try:
        return pyglet.graphics.api.core.get_shader("default_shapes")
    except KeyError:
        load_package_shader = pyglet.graphics.api.core.load_package_shader
        program = pyglet.graphics.api.core.create_shader_program(
            "default_shapes",
            (load_package_shader("pyglet.graphics.api.vulkan.shaders", "shapes.vert.spv"), 'vertex'),
            (load_package_shader("pyglet.graphics.api.vulkan.shaders", "shapes.frag.spv"), 'fragment'),
        )
        if not program.is_defined:
            program.set_attributes(
                Attribute("position", location=0, components=3, data_type="f"),
                Attribute("translation", location=1, components=3, data_type="f"),
                Attribute("colors", location=2, components=4, data_type="f"),
                Attribute("rotation", location=3, components=1, data_type="f"),
            )
            program.set_uniform_blocks(WindowBlock)

        program.set_attribute_format("colors", data_type="B", normalize=True)
        return program



class _ShapeGroup(Group):
    """Shared Shape rendering Group.

    The group is automatically coalesced with other shape groups
    sharing the same parent group and blend parameters.
    """
    blend_src: int
    blend_dest: int

    def __init__(self, blend_src: BlendFactor, blend_dest: BlendFactor, program: VulkanShaderProgram,
                 parent: Group | None = None) -> None:
        """Create a Shape group.

        The group is created internally. Usually you do not
        need to explicitly create it.

        Args:
            blend_src:
                OpenGL blend source mode; for example, ``GL_SRC_ALPHA``.
            blend_dest:
                OpenGL blend destination mode; for example, ``GL_ONE_MINUS_SRC_ALPHA``.
            program:
                The ShaderProgram to use.
            parent:
                Optional parent group.
        """
        super().__init__(parent=parent)
        self.set_shader_program(program)
        self.set_blend(blend_src, blend_dest)
