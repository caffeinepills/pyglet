from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import pyglet
from pyglet.graphics import Group
from pyglet.graphics.api.vulkan.instance import WindowBlock
from pyglet.graphics.shader import Attribute, Sampler

_is_pyglet_doc_run = hasattr(sys, 'is_pyglet_doc_run') and sys.is_pyglet_doc_run

if TYPE_CHECKING:
    from pyglet.graphics.texture import Texture
    from pyglet.enums import BlendFactor
    from pyglet.graphics.api.vulkan.shader import VulkanShaderProgram



def get_default_shader() -> VulkanShaderProgram:
    """Create and return the default sprite shader.

    This method allows the module to be imported without an OpenGL Context.
    """
    try:
        return pyglet.graphics.api.core.get_shader("default_sprite")
    except KeyError:
        load_package_shader = pyglet.graphics.api.core.load_package_shader
        program = pyglet.graphics.api.core.create_shader_program(
            "default_sprite",
            (load_package_shader("pyglet.graphics.api.vulkan.shaders", "sprite.vert.spv"), 'vertex'),
            (load_package_shader("pyglet.graphics.api.vulkan.shaders", "sprite.frag.spv"), 'fragment'),
        )
        if not program.is_defined:
            program.set_attributes(
                Attribute("translate", location=0, components=3, data_type="f"),
                Attribute("colors", location=1, components=4, data_type="f"),
                Attribute("tex_coords", location=2, components=3, data_type="f"),
                Attribute("scale", location=3, components=2, data_type="f"),
                Attribute("position", location=4, components=3, data_type="f"),
                Attribute("rotation", location=5, components=1, data_type="f"),
            )
            program.set_uniform_blocks(WindowBlock)
            program.set_samplers(Sampler("sprite_texture", desc_set=0, binding=2))

        program.set_attribute_format("colors", data_type="B", normalize=True)
        return program


def get_default_array_shader() -> VulkanShaderProgram:
    """Create and return the default array sprite shader.

    This method allows the module to be imported without an OpenGL Context.
    """
    load_package_shader = pyglet.graphics.api.core.load_package_shader
    return pyglet.graphics.api.core.get_cached_shader(
        "default_sprite",
        (load_package_shader("pyglet.graphics.api.vulkan.shaders", "sprite.vert.spv"), 'vertex'),
        (load_package_shader("pyglet.graphics.api.vulkan.shaders", "sprite_array.frag.spv"), 'fragment'),
    )

class SpriteGroup(Group):
    """Shared Sprite rendering Group."""

    def __init__(self, texture: Texture, blend_src: BlendFactor, blend_dest: BlendFactor,
                 program: VulkanShaderProgram, parent: Group | None = None) -> None:
        """Create a sprite group.

        The group is created internally when a :py:class:`~pyglet.sprite.Sprite`
        is created; applications usually do not need to explicitly create it.

        Args:
            texture:
                The (top-level) texture containing the sprite image.
            blend_src:
                Blend factor source mode; for example: ``SRC_ALPHA``.
            blend_dest:
                Blend factor source mode; for example: ``_ONE_MINUS_SRC_ALPHA``.
            program:
                A custom ShaderProgram.
            parent:
                Optional parent group.
        """
        super().__init__(parent=parent)
        self.texture = texture
        self.set_shader_program(program)
        self.set_blend(blend_src, blend_dest)
        self.set_texture(self.texture, 2)


