from __future__ import annotations

import pyglet
from typing import TYPE_CHECKING, ClassVar

from pyglet.enums import BlendFactor
from pyglet.graphics.api.vulkan.instance import WindowBlock
from pyglet.graphics.draw import Group
from pyglet.graphics.shader import Attribute, SampledTextureBinding, PushConstants

if TYPE_CHECKING:
    from pyglet.graphics.api.vulkan.texture import VulkanTexture
    from pyglet.graphics.api.vulkan.shader import VulkanShaderProgram

layout_vertex_source = """#version 450
    layout(location = 0) in vec3 position;
    layout(location = 1) in vec4 colors;
    layout(location = 2) in vec3 tex_coords;
    layout(location = 3) in vec3 translation;
    layout(location = 4) in vec3 view_translation;
    layout(location = 5) in vec2 anchor;
    layout(location = 6) in float rotation;
    layout(location = 7) in float visible;

    layout(location = 0) out vec4 text_colors;
    layout(location = 1) out vec2 texture_coords;
    layout(location = 2) out vec4 vert_position;

    layout(set = 0, binding = 0) uniform WindowBlock {
        mat4 projection;
        mat4 view;
    } window;

    void main()
    {
        mat4 m_rotation = mat4(1.0);
        vec3 v_anchor = vec3(anchor.x, anchor.y, 0);
        mat4 m_anchor = mat4(1.0);
        mat4 m_translate = mat4(1.0);

        m_translate[3][0] = translation.x;
        m_translate[3][1] = translation.y;
        m_translate[3][2] = translation.z;
        m_rotation[0][0] =  cos(-radians(rotation));
        m_rotation[0][1] =  sin(-radians(rotation));
        m_rotation[1][0] = -sin(-radians(rotation));
        m_rotation[1][1] =  cos(-radians(rotation));

        gl_Position = window.projection * window.view * m_translate * m_anchor * m_rotation * vec4(position + view_translation + v_anchor, 1.0) * visible;

        vert_position = vec4(position + translation + view_translation + v_anchor, 1.0);
        text_colors = colors;
        texture_coords = tex_coords.xy;
    }
"""

layout_fragment_source = """#version 450
    layout(location = 0) in vec4 text_colors;
    layout(location = 1) in vec2 texture_coords;
    layout(location = 2) in vec4 vert_position;

    layout(location = 0) out vec4 final_colors;

    layout(push_constant) uniform PushConstants {
        bool scissor;
        vec4 scissor_area;
    } pc;

    layout(set = 0, binding = 2) uniform sampler2D text;

    void main()
    {
        final_colors = texture(text, texture_coords) * text_colors;
        if (pc.scissor == true) {
            if (vert_position.x < pc.scissor_area[0]) discard;                     // left
            if (vert_position.y < pc.scissor_area[1]) discard;                     // bottom
            if (vert_position.x > pc.scissor_area[0] + pc.scissor_area[2]) discard;   // right
            if (vert_position.y > pc.scissor_area[1] + pc.scissor_area[3]) discard;   // top
        }
    }
"""
layout_fragment_image_source = """#version 450
    layout(location = 0) in vec4 text_colors;
    layout(location = 1) in vec2 texture_coords;
    layout(location = 2) in vec4 vert_position;

    layout(location = 0) out vec4 final_colors;

    layout(push_constant) uniform PushConstants {
        bool scissor;
        vec4 scissor_area;
    } pc;

    layout(set = 0, binding = 2) uniform sampler2D layout_image;

    void main()
    {
        final_colors = texture(layout_image, texture_coords);
        if (pc.scissor == true) {
            if (vert_position.x < pc.scissor_area[0]) discard;                     // left
            if (vert_position.y < pc.scissor_area[1]) discard;                     // bottom
            if (vert_position.x > pc.scissor_area[0] + pc.scissor_area[2]) discard;   // right
            if (vert_position.y > pc.scissor_area[1] + pc.scissor_area[3]) discard;   // top
        }
    }
    """


def get_default_layout_shader() -> VulkanShaderProgram:
    """The default shader used for all glyphs in the layout."""
    try:
        return pyglet.graphics.api.core.get_shader("default_layout")
    except KeyError:
        load_package_shader = pyglet.graphics.api.core.load_package_shader
        program = pyglet.graphics.api.core.create_shader_program(
            "default_layout",
            (load_package_shader("pyglet.graphics.api.vulkan.shaders", "layout.vert.spv"), 'vertex'),
            (load_package_shader("pyglet.graphics.api.vulkan.shaders", "layout.frag.spv"), 'fragment'),
        )
        if not program.is_defined:
            program.set_attributes(
                Attribute("position", location=0, components=3, data_type="f"),
                Attribute("colors", location=1, components=4, data_type="f"),
                Attribute("tex_coords", location=2, components=3, data_type="f"),
                Attribute("translation", location=3, components=3, data_type="f"),
                Attribute("view_translation", location=4, components=3, data_type="f"),
                Attribute("anchor", location=5, components=2, data_type="f"),
                Attribute("rotation", location=6, components=1, data_type="f"),
                Attribute("visible", location=7, components=1, data_type="f"),
            )
            program.set_uniform_blocks(WindowBlock)
            program.set_push_constants(PushConstants(stages=('fragment', ), constants=[("scissor", "bool"), ("scissor_area", "vec4")]))
            program.set_sampled_textures(SampledTextureBinding("text", desc_set=0, binding=2))

        program.set_attribute_format("colors", data_type="B", normalize=True)
        return program


def get_default_image_layout_shader() -> VulkanShaderProgram:
    """The default shader used for an InlineElement image. Used for HTML Labels that insert images via <img> tag."""
    try:
        return pyglet.graphics.api.core.get_shader("default_layout_image")
    except KeyError:
        load_package_shader = pyglet.graphics.api.core.load_package_shader
        program = pyglet.graphics.api.core.create_shader_program(
            "default_layout_image",
            (load_package_shader("pyglet.graphics.api.vulkan.shaders", "layout.vert.spv"), 'vertex'),
            (load_package_shader("pyglet.graphics.api.vulkan.shaders", "layout_image.frag.spv"), 'fragment'),
        )
        if not program.is_defined:
            program.set_attributes(
                Attribute("position", location=0, components=3, data_type="f"),
                Attribute("colors", location=1, components=4, data_type="f"),
                Attribute("tex_coords", location=2, components=3, data_type="f"),
                Attribute("translation", location=3, components=3, data_type="f"),
                Attribute("view_translation", location=4, components=3, data_type="f"),
                Attribute("anchor", location=5, components=2, data_type="f"),
                Attribute("rotation", location=6, components=1, data_type="f"),
                Attribute("visible", location=7, components=1, data_type="f"),
            )
            program.set_uniform_blocks(WindowBlock)
            program.set_push_constants(PushConstants(stages=('fragment', ), constants=[("scissor", "bool"), ("scissor_area", "vec4")]))
            program.set_sampled_textures(SampledTextureBinding("layout_image", desc_set=0, binding=2))

        program.set_attribute_format("colors", data_type="B", normalize=True)
        return program


decoration_vertex_source = """#version 450 core
    layout(location = 0) in vec3 position;
    layout(location = 1) in vec4 colors;
    layout(location = 2) in vec3 tex_coords;
    layout(location = 3) in vec3 translation;
    layout(location = 4) in vec3 view_translation;
    layout(location = 5) in vec2 anchor;
    layout(location = 6) in float rotation;
    layout(location = 7) in float visible;

    layout(location = 0) out vec4 vert_colors;
    layout(location = 1) out vec4 vert_position;

    layout(set = 0, binding = 0) uniform WindowBlock {
        mat4 projection;
        mat4 view;
    } window;

    void main()
    {
        mat4 m_rotation = mat4(1.0);
        vec3 v_anchor = vec3(anchor.x, anchor.y, 0);
        mat4 m_anchor = mat4(1.0);
        mat4 m_translate = mat4(1.0);

        m_translate[3][0] = translation.x;
        m_translate[3][1] = translation.y;
        m_translate[3][2] = translation.z;
        m_rotation[0][0] =  cos(-radians(rotation));
        m_rotation[0][1] =  sin(-radians(rotation));
        m_rotation[1][0] = -sin(-radians(rotation));
        m_rotation[1][1] =  cos(-radians(rotation));

        gl_Position = window.projection * window.view * m_translate * m_anchor * m_rotation * vec4(position + view_translation + v_anchor, 1.0) * visible;

        vert_position = vec4(position + translation + view_translation + v_anchor, 1.0);
        vert_colors = colors;
    }
"""
decoration_fragment_source = """#version 450
    layout(location = 0) in vec4 vert_colors;
    layout(location = 1) in vec4 vert_position;

    layout(location = 0) out vec4 final_colors;

    layout(push_constant) uniform PushConstants {
        bool scissor;
        vec4 scissor_area;
    } pushConstants;

    void main()
    {
        if (pushConstants.scissor == true) {
            if (vert_position.x < pushConstants.scissor_area[0]) discard;                     // left
            if (vert_position.y < pushConstants.scissor_area[1]) discard;                     // bottom
            if (vert_position.x > pushConstants.scissor_area[0] + pushConstants.scissor_area[2]) discard;   // right
            if (vert_position.y > pushConstants.scissor_area[1] + pushConstants.scissor_area[3]) discard;   // top
        }
    }
"""


def get_default_decoration_shader() -> VulkanShaderProgram:
    """The default shader for underline and background decoration effects in the layout."""
    try:
        return pyglet.graphics.api.core.get_shader("default_layout_decoration")
    except KeyError:
        load_package_shader = pyglet.graphics.api.core.load_package_shader
        program = pyglet.graphics.api.core.create_shader_program(
            "default_layout_decoration",
            (load_package_shader("pyglet.graphics.api.vulkan.shaders", "layout_decoration.vert.spv"), 'vertex'),
            (load_package_shader("pyglet.graphics.api.vulkan.shaders", "layout_decoration.frag.spv"), 'fragment'),
        )
        print("DEFINED!", program.is_defined)
        if not program.is_defined:
            program.set_attributes(
                Attribute("position", location=0, components=3, data_type="f"),
                Attribute("colors", location=1, components=4, data_type="f"),
                Attribute("translation", location=3, components=3, data_type="f"),
                Attribute("view_translation", location=4, components=3, data_type="f"),
                Attribute("anchor", location=5, components=2, data_type="f"),
                Attribute("rotation", location=6, components=1, data_type="f"),
                Attribute("visible", location=7, components=1, data_type="f"),
            )
            program.set_uniform_blocks(WindowBlock)
            program.set_push_constants(PushConstants(stages=('fragment', ), constants=[("scissor", "bool"), ("scissor_area", "vec4")]))

        program.set_attribute_format("colors", data_type="B", normalize=True)
        return program

class TextLayoutGroup(Group):
    """Create a text layout rendering group.

    The group is created internally when a :py:class:`~pyglet.text.Label`
    is created; applications usually do not need to explicitly create it.
    """

    def __init__(self, texture: VulkanTexture, program: VulkanShaderProgram, order: int = 1,  # noqa: D107
                 parent: Group | None = None) -> None:
        super().__init__(order=order, parent=parent)
        print("SETTING TEXTURE IN GROUP", texture, id(self))
        self.texture = texture
        self.set_blend(BlendFactor.SRC_ALPHA, BlendFactor.ONE_MINUS_SRC_ALPHA)
        self.set_shader_program(program)
        self.set_texture(texture, 2)
        self.set_shader_uniform(program, "scissor", False)
        self.set_shader_uniform(program, "scissor_area", (0, 0, 0, 0))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(address={id(self)}, texture={self.texture})"

class TextDecorationGroup(Group):
    """Create a text decoration rendering group.

    The group is created internally when a :py:class:`~pyglet.text.Label`
    is created; applications usually do not need to explicitly create it.
    """

    def __init__(self, program: VulkanShaderProgram, order: int = 0,  # noqa: D107
                 parent: Group | None = None) -> None:
        super().__init__(order=order, parent=parent)
        self.set_blend(BlendFactor.SRC_ALPHA, BlendFactor.ONE_MINUS_SRC_ALPHA)
        self.set_shader_program(program)


# ====== SCROLLING TEXT

class ScrollableTextLayoutGroup(Group):
    """Default rendering group for :py:class:`~pyglet.text.layout.ScrollableTextLayout`.

    The group maintains internal state for specifying the viewable
    area, and for scrolling. Because the group has internal state
    specific to the text layout, the group is never shared.
    """
    scissor_area: ClassVar[tuple[int, int, int, int]] = 0, 0, 0, 0

    def __init__(self, texture: Texture, program: VulkanShaderProgram, order: int = 1,  # noqa: D107
                 parent: Group | None = None) -> None:

        super().__init__(order=order, parent=parent)
        self.texture = texture
        self.set_blend(BlendFactor.SRC_ALPHA, BlendFactor.ONE_MINUS_SRC_ALPHA)
        self.set_shader_program(program)
        self.set_texture(self.texture, 2)
        self.set_shader_uniform(program, "scissor", True)
        self.set_shader_uniform(program, "scissor_area", self.scissor_area)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.texture})"

    def __eq__(self, other: object) -> bool:
        return self is other

    def __hash__(self) -> int:
        return id(self)


class ScrollableTextDecorationGroup(Group):
    """Create a text decoration rendering group.

    The group is created internally when a :py:class:`~pyglet.text.Label`
    is created; applications usually do not need to explicitly create it.
    """

    scissor_area: ClassVar[tuple[int, int, int, int]] = 0, 0, 0, 0

    def __init__(self, program: VulkanShaderProgram, order: int = 0, parent: Group | None = None) -> None:  # noqa: D107
        super().__init__(order=order, parent=parent)
        self.program = program
        self.set_blend(BlendFactor.SRC_ALPHA, BlendFactor.ONE_MINUS_SRC_ALPHA)
        self.set_shader_program(program)
        self.set_shader_uniform(program,"scissor", True)
        self.set_shader_uniform(program,"scissor_area", self.scissor_area)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(scissor={self.scissor_area})"

    def __eq__(self, other: object) -> bool:
        return self is other

    def __hash__(self) -> int:
        return id(self)
