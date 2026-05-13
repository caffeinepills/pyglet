from __future__ import annotations

import sys
from typing import (
    TYPE_CHECKING,
    ClassVar,
    Protocol,
    Any,
)

import pyglet
from pyglet import graphics
from pyglet.enums import BlendFactor
from pyglet.text.layout import base, get_default_decoration_shader

if TYPE_CHECKING:
    from pyglet.customtypes import AnchorX, AnchorY
    from pyglet.font.base import Font, Glyph
    from pyglet.graphics import Batch
    from pyglet.graphics.shader import ShaderProgram
    from pyglet.image import Texture
    from pyglet.text.document import AbstractDocument
    from pyglet.text.layout.base import (
        _AbstractBox,
    )

_is_pyglet_doc_run = hasattr(sys, "is_pyglet_doc_run") and sys.is_pyglet_doc_run

layout_vertex_source = """#version 330 core
    in vec3 position;
    in vec4 colors;
    in vec3 tex_coords;
    in vec3 translation;
    in vec3 view_translation;
    in vec2 anchor;
    in vec2 scale;
    in float rotation;
    in float visible;

    out vec4 text_colors;
    out vec2 texture_coords;
    out vec4 vert_position;

    uniform WindowBlock
    {
        mat4 projection;
        mat4 view;
    } window;

    void main()
    {
        mat4 m_rotation = mat4(1.0);
        vec3 v_anchor = vec3(anchor.x, anchor.y, 0);
        mat4 m_anchor = mat4(1.0);
        mat4 m_translate = mat4(1.0);
        mat4 m_scale = mat4(1.0);

        m_translate[3][0] = translation.x;
        m_translate[3][1] = translation.y;
        m_translate[3][2] = translation.z;

        m_scale[0][0] = scale.x;
        m_scale[1][1] = scale.y;

        m_rotation[0][0] =  cos(-radians(rotation));
        m_rotation[0][1] =  sin(-radians(rotation));
        m_rotation[1][0] = -sin(-radians(rotation));
        m_rotation[1][1] =  cos(-radians(rotation));

        gl_Position = window.projection * window.view * m_translate * m_anchor * m_rotation * m_scale * vec4(position + view_translation + v_anchor, 1.0) * visible;

        vert_position = vec4(position + translation + view_translation + v_anchor, 1.0);
        text_colors = colors;
        texture_coords = tex_coords.xy;
    }

"""  # noqa: E501
msdf_layout_fragment_source = """#version 330 core
in vec4 text_colors;
in vec2 texture_coords;
in vec4 vert_position;

out vec4 final_colors;

uniform sampler2D text;
uniform bool scissor;
uniform vec4 scissor_area;
uniform float time;

// MSDF rendering controls
uniform float sdf_threshold;  // Typically 0.5
uniform float base_smoothing; // Unscaled smoothing
uniform float base_outline_thickness; // Unscaled outline thickness
uniform float weight;

// Scaling factor (computed per-character in your application)
uniform float scale;  // Computed as (rendered_glyph_size / sdf_glyph_size)

uniform vec4 outline_color; // Outline color

// Function to compute median of three values
float median(vec3 v) {
    return max(min(v.r, v.g), min(max(v.r, v.g), v.b));
}

void main()
{
    vec3 msdf_value = texture(text, texture_coords).rgb;

    float sdf_value = median(msdf_value);

    float outer_edge = sdf_threshold * weight;

    // Scale smoothing and outline thickness
    float smoothing = base_smoothing / scale;
    float outline_thickness = base_outline_thickness / scale;

    // Compute outline alpha
    float outline_alpha = smoothstep(
        outer_edge - outline_thickness - smoothing,
        outer_edge - outline_thickness + smoothing,
        sdf_value
    );

    // Compute fill alpha (regular text rendering)
    float fill_alpha = smoothstep(
        outer_edge - smoothing,
        outer_edge + smoothing,
        sdf_value
    );

    vec4 fill_color = vec4(text_colors.rgb, fill_alpha * text_colors.a);

    // Ignore outline color and just make a fun color
    vec3 col = 0.5 + 0.5 * cos(time * 2.0 + texture_coords.xyx + vec3(0, 2, 4));

    vec4 outline = vec4(outline_color.rgb * col, outline_alpha * outline_color.a);

    // Composite: Render the outline first, then overlay the fill
    final_colors = mix(outline, fill_color, fill_alpha);

    if (scissor == true) {
        if (vert_position.x < scissor_area[0]) discard;                     // left
        if (vert_position.y < scissor_area[1]) discard;                     // bottom
        if (vert_position.x > scissor_area[0] + scissor_area[2]) discard;   // right
        if (vert_position.y > scissor_area[1] + scissor_area[3]) discard;   // top
    }
}
"""

sdf_layout_fragment_source = """#version 330 core
in vec4 text_colors;
in vec2 texture_coords;
in vec4 vert_position;

out vec4 final_colors;

uniform sampler2D text;
uniform bool scissor;
uniform vec4 scissor_area;
uniform float time;

uniform float sdf_threshold;
uniform float base_smoothing;
uniform float base_outline_thickness;
uniform float weight;

// Scaling factor (computed per-character in your application)
uniform float scale;  // Computed as (rendered_glyph_size / sdf_glyph_size)

uniform vec4 outline_color; // Outline color

float contour(float dist, float edge, float width) {
  return clamp(smoothstep(edge - width, edge + width, dist), 0.0, 1.0);
}

void main()
{
    // Sample the SDF texture (assumed to be stored in red channel)
    float sdf_value = texture(text, texture_coords).r;

    float outer_edge = sdf_threshold * weight;

    // Scale smoothing and outline thickness
    float smoothing = base_smoothing / scale;
    float outline_thickness = base_outline_thickness / scale;

    // Compute outline alpha (shifts the edge outward)
    float outline_alpha = smoothstep(
        outer_edge - outline_thickness - smoothing,
        outer_edge - outline_thickness + smoothing,
        sdf_value
    );

    // Compute fill alpha (regular text rendering)
    float fill_alpha = smoothstep(
        outer_edge - smoothing,
        outer_edge + smoothing,
        sdf_value
    );

    // Blend outline and text colors
    vec4 fill_color = vec4(text_colors.rgb, fill_alpha * text_colors.a);

    vec3 col = 0.5 + 0.5*cos(time*2+texture_coords.xyx+vec3(0,2,4));

    vec4 outline = vec4(outline_color.rgb * col, outline_alpha * outline_color.a);

    // Composite: Render the outline first, then overlay the fill
    final_colors = mix(outline, fill_color, fill_alpha);

    // Optional scissoring
    if (scissor == true) {
        if (vert_position.x < scissor_area[0]) discard;                     // left
        if (vert_position.y < scissor_area[1]) discard;                     // bottom
        if (vert_position.x > scissor_area[0] + scissor_area[2]) discard;   // right
        if (vert_position.y > scissor_area[1] + scissor_area[3]) discard;   // top
    }
}
"""




layout_fragment_image_source = """#version 330 core
    in vec4 text_colors;
    in vec2 texture_coords;
    in vec4 vert_position;

    uniform sampler2D image_texture;

    out vec4 final_colors;

    uniform sampler2D text;
    uniform bool scissor;
    uniform vec4 scissor_area;

    void main()
    {
        final_colors = texture(image_texture, texture_coords.xy);
        if (scissor == true) {
            if (vert_position.x < scissor_area[0]) discard;                     // left
            if (vert_position.y < scissor_area[1]) discard;                     // bottom
            if (vert_position.x > scissor_area[0] + scissor_area[2]) discard;   // right
            if (vert_position.y > scissor_area[1] + scissor_area[3]) discard;   // top
        }
    }
"""

class DistFieldTextLayoutGroup(graphics.Group):
    """Create a text layout rendering group.

    The group is created internally when a :py:class:`~pyglet.text.Label`
    is created; applications usually do not need to explicitly create it.
    """

    def __init__(self, texture: Texture, program: ShaderProgram, order: int = 1,  # noqa: D107
                 parent: graphics.Group | None = None) -> None:
        super().__init__(order=order, parent=parent)
        self.texture = texture
        self.uniforms = {
            "scissor": False,
            "sdf_threshold": 0.5,
            "base_smoothing": 0.015,
            "base_outline_thickness": 0.2,
            "outline_color": (1.0, 1.0, 1.0, 1.0),
            "scale": 1.0,
            "weight": 1.0,
        }
        self.set_shader_program(program)
        self.set_blend(BlendFactor.SRC_ALPHA, BlendFactor.ONE_MINUS_SRC_ALPHA)
        self.set_texture(texture, 0)
        self.set_shader_uniforms(program, self.uniforms)

    @property
    def outline_thickness(self) -> float:
        return self.uniforms["base_outline_thickness"]

    @outline_thickness.setter
    def outline_thickness(self, value: float) -> None:
        self.uniforms["base_outline_thickness"] = value

    @property
    def smoothing(self) -> float:
        return self.uniforms["base_smoothing"]

    @smoothing.setter
    def smoothing(self, value: float) -> None:
        self.uniforms["base_smoothing"] = value

    @property
    def weight(self) -> float:
        return self.uniforms["weight"]

    @weight.setter
    def weight(self, value: float) -> None:
        self.uniforms["weight"] = value

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.texture})"


class _LayoutVertexList(Protocol):
    """Just a Protocol to add completion for VertexLists."""
    position: list
    colors: list
    translation: list
    view_translation: list
    anchor: list
    rotation: list
    visible: list
    count: int

    def delete(self) -> None: ...


def get_msdf_layout_shader() -> ShaderProgram:
    """The default shader used for all glyphs in the layout."""
    return pyglet.graphics.api.get_cached_shader("msdf_text_shader",
                                                 (layout_vertex_source, "vertex"),
                                                    (msdf_layout_fragment_source, "fragment"))

def get_sdf_layout_shader() -> ShaderProgram:
    """The default shader used for all glyphs in the layout."""
    return pyglet.graphics.api.get_cached_shader("sdf_text_shader", (layout_vertex_source, "vertex"),
                                                    (sdf_layout_fragment_source, "fragment"))


class _DistFieldGlyphBox(base._GlyphBox):  # noqa: SLF001
    owner: Texture
    font: Font
    glyphs: list[tuple[int, Glyph]]
    advance: int
    vertex_lists: list[_LayoutVertexList]

    def _get_layout_vertex_data(
        self,
        layout: SDFTextLayout,
        n_glyphs: int,
        vertices: list[int],
        tex_coords: list[float],
        colors: list[int],
        t_position: tuple[float, float, float],
        rotation: float,
        visible: bool,
        anchor_x: float,
        anchor_y: float,
    ) -> dict[str, tuple[str, Any]]:
        scale = layout.scale
        data: dict[str, tuple[str, Any]] = {
            "position": ("f", vertices),
            "translation": ("f", t_position * 4 * n_glyphs),
            "colors": ("Bn", colors),
            "view_translation": ("f", (0, 0, 0) * 4 * n_glyphs),
            "tex_coords": ("f", tex_coords),
            "rotation": ("f", ((rotation,) * 4) * n_glyphs),
            "visible": ("f", ((visible,) * 4) * n_glyphs),
            "anchor": ("f", ((anchor_x, anchor_y) * 4) * n_glyphs),
            "scale": ("f", ((scale, scale) * 4) * n_glyphs),
        }
        return data

    def update_scale(self, scale: float) -> None:
        scale_tuple = (scale, scale)
        for _vertex_list in self.vertex_lists:
            if hasattr(_vertex_list, "scale"):
                _vertex_list.scale[:] = scale_tuple * _vertex_list.count

    def __repr__(self) -> str:
        return f"_SDFGlyphBox({self.glyphs})"



class SDFTextLayout(base.TextLayout):  # noqa: D101
    _boxes: list[_DistFieldGlyphBox]
    group_class: ClassVar[type[DistFieldTextLayoutGroup]] = DistFieldTextLayoutGroup
    glyph_box_class: ClassVar[type[_AbstractBox]] = _DistFieldGlyphBox

    def __init__(self, document: AbstractDocument, x: float = 0, y: float = 0, z: float = 0, width: int | None = None,
                 height: int | None = None, anchor_x: AnchorX = 'left', anchor_y: AnchorY = 'bottom',
                 rotation: float = 0, multiline: bool = False, dpi: float | None = None, batch: Batch | None = None,
                 group: graphics.Group | None = None, program: ShaderProgram | None = None, wrap_lines: bool = True,
                 init_document: bool = True) -> None:
        self._scale = 1.0
        super().__init__(document, x, y, z, width, height, anchor_x, anchor_y, rotation, multiline, dpi, batch, group,
                         program or get_sdf_layout_shader(), wrap_lines, init_document)

    @property
    def scale(self) -> float:
        return self._scale

    @scale.setter
    def scale(self, value: float) -> None:
        self._scale = value
        for box in self._boxes:
            box.update_scale(self._scale)
