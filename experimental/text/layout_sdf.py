from __future__ import annotations

import sys
from typing import (
    TYPE_CHECKING,
    ClassVar,
    Protocol,
)

import pyglet
from pyglet import graphics
from pyglet.font.base import GlyphPosition
from pyglet.gl import (
    GL_BLEND,
    GL_DEPTH_ATTACHMENT,
    GL_DEPTH_COMPONENT,
    GL_LINES,
    GL_NEAREST,
    GL_ONE_MINUS_SRC_ALPHA,
    GL_SRC_ALPHA,
    GL_TEXTURE0,
    GL_TRIANGLES,
    glActiveTexture,
    glBindTexture,
    glBlendFunc,
    glDisable,
    glEnable,
)
from pyglet.text.layout import base

if TYPE_CHECKING:
    from pyglet.customtypes import AnchorX, AnchorY
    from pyglet.font.base import Font, Glyph
    from pyglet.graphics import Batch
    from pyglet.graphics.shader import ShaderProgram
    from pyglet.image import Texture
    from pyglet.text.document import AbstractDocument
    from pyglet.text.layout.base import (
        _AbstractBox,
        _LayoutContext,
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



decoration_vertex_source = """#version 330 core
    in vec3 position;
    in vec4 colors;
    in vec3 translation;
    in vec3 view_translation;
    in vec2 anchor;
    in float rotation;
    in float visible;

    out vec4 vert_colors;
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
"""  # noqa: E501



decoration_fragment_source = """#version 330 core
    in vec4 vert_colors;
    in vec4 vert_position;

    out vec4 final_colors;

    uniform bool scissor;
    uniform vec4 scissor_area;

    void main()
    {
        final_colors = vert_colors;
        if (scissor == true) {
            if (vert_position.x < scissor_area[0]) discard;                     // left
            if (vert_position.y < scissor_area[1]) discard;                     // bottom
            if (vert_position.x > scissor_area[0] + scissor_area[2]) discard;   // right
            if (vert_position.y > scissor_area[1] + scissor_area[3]) discard;   // top
        }
    }
"""

_empty_pos = GlyphPosition(0, 0, 0, 0)

class DistFieldTextLayoutGroup(graphics.Group):
    """Create a text layout rendering group.

    The group is created internally when a :py:class:`~pyglet.text.Label`
    is created; applications usually do not need to explicitly create it.
    """

    def __init__(self, texture: Texture, program: ShaderProgram, order: int = 1,  # noqa: D107
                 parent: graphics.Group | None = None) -> None:
        super().__init__(order=order, parent=parent)
        self.texture = texture
        self.program = program
        self.outline_thickness = 0.2
        self.smoothing = 0.015
        self.weight = 1.0

    def set_state(self) -> None:
        self.program.use()
        self.program["scissor"] = False
        self.program["sdf_threshold"] = 0.5
        self.program["base_smoothing"] = self.smoothing
        self.program["base_outline_thickness"] = self.outline_thickness
        self.program["outline_color"] = (1.0, 1.0, 1.0, 1.0)
        self.program["scale"] = 1.0
        self.program["weight"] = self.weight

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(self.texture.target, self.texture.id)

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    def unset_state(self) -> None:
        glDisable(GL_BLEND)
        self.program.stop()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.texture})"

    def __eq__(self, other: object) -> bool:
        return (other.__class__ is self.__class__ and
                self.parent is other.parent and
                self.program.id is other.program.id and
                self.order == other.order and
                self.texture.target == other.texture.target and
                self.texture.id == other.texture.id)

    def __hash__(self) -> int:
        return hash((id(self.parent), self.program.id, self.order, self.texture.target, self.texture.id))


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
    return pyglet.gl.current_context.create_program((layout_vertex_source, "vertex"),
                                                    (msdf_layout_fragment_source, "fragment"))

def get_sdf_layout_shader() -> ShaderProgram:
    """The default shader used for all glyphs in the layout."""
    return pyglet.gl.current_context.create_program((layout_vertex_source, "vertex"),
                                                    (sdf_layout_fragment_source, "fragment"))

def get_default_image_layout_shader() -> ShaderProgram:
    """The default shader used for an InlineElement image. Used for HTML Labels that insert images via <img> tag."""
    return pyglet.gl.current_context.create_program((layout_vertex_source, "vertex"),
                                                    (layout_fragment_image_source, "fragment"))


def get_default_decoration_shader() -> ShaderProgram:
    """The default shader for underline and background decoration effects in the layout."""
    return pyglet.gl.current_context.create_program((decoration_vertex_source, "vertex"),
                                                    (decoration_fragment_source, "fragment"))


class _DistFieldGlyphBox(base._GlyphBox):
    owner: Texture
    font: Font
    glyphs: list[tuple[int, Glyph]]
    advance: int
    vertex_lists: list[_LayoutVertexList]

    def place(self, layout: SDFTextLayout, i: int, x: float, y: float, z: float, line_x: float, line_y: float,
              rotation: float, visible: bool, anchor_x: float, anchor_y: float, context: _LayoutContext) -> None:
        # Creates the initial attributes and vertex lists of the glyphs.
        # line_x/line_y are calculated when lines shift. To prevent having to destroy and recalculate the layout
        # everytime we move this layout, we bake those into the vertices. This way the translate can be moved directly.
        assert self.glyphs
        assert not self.vertex_lists
        try:
            group = layout.group_cache[self.owner]
        except KeyError:
            group = layout.group_class(self.owner, layout.program, order=1, parent=layout.group)
            layout.group_cache[self.owner] = group

        n_glyphs = self.length
        vertices = []
        tex_coords = []
        baseline = 0
        x1 = line_x
        for start, end, baseline_ in context.baseline_iter.ranges(i, i + n_glyphs):
            baseline = layout._parse_distance(baseline_)  # noqa: SLF001
            assert len(self.glyphs[start - i:end - i]) == end - start
            for (kern, glyph, glyph_pos) in self.glyphs[start - i:end - i]:
                x1 += kern
                v0, v1, v2, v3 = glyph.vertices
                v0 += x1 + glyph_pos.x_offset
                v2 += x1 + glyph_pos.x_offset
                v1 += line_y + baseline + glyph_pos.y_offset
                v3 += line_y + baseline + glyph_pos.y_offset
                vertices.extend(map(round, [v0, v1, 0, v2, v1, 0, v2, v3, 0, v0, v3, 0]))
                t = glyph.tex_coords
                tex_coords.extend(t)
                x1 += glyph.advance + glyph_pos.x_advance
                v1 += glyph_pos.y_advance
                v3 += glyph_pos.y_advance

        # Text color
        colors = []
        for start, end, color in context.colors_iter.ranges(i, i + n_glyphs):
            if color is None:
                color = (0, 0, 0, 255)  # noqa: PLW2901
            if len(color) != 4:
                msg = f"Color requires 4 values (R, G, B, A). Value received: {color}"
                raise ValueError(msg)
            colors.extend(color * ((end - start) * 4))

        indices = []
        # Create indices for each glyph quad:
        for glyph_idx in range(n_glyphs):
            indices.extend([element + (glyph_idx * 4) for element in [0, 1, 2, 0, 2, 3]])

        t_position = (x, y, z)
        scale = layout.scale

        vertex_list = layout.program.vertex_list_indexed(n_glyphs * 4, GL_TRIANGLES, indices, layout.batch, group,
                                                         position=("f", vertices),
                                                         translation=("f", t_position * 4 * n_glyphs),
                                                         colors=("Bn", colors),
                                                         scale=("f", ((scale, scale) * 4) * n_glyphs),
                                                         tex_coords=("f", tex_coords),
                                                         rotation=("f", ((rotation,) * 4) * n_glyphs),
                                                         visible=("f", ((visible,) * 4) * n_glyphs),
                                                         anchor=("f", ((anchor_x, anchor_y) * 4) * n_glyphs))
        self._add_vertex_list(vertex_list, context)

        # Decoration (background color and underline)
        # -------------------------------------------
        # Should iterate over baseline too, but in practice any sensible
        # change in baseline will correspond with a change in font size,
        # and thus glyph run as well.  So we cheat and just use whatever
        # baseline was seen last.
        background_vertices = []
        background_colors = []
        underline_vertices = []
        underline_colors = []
        y1 = line_y + self.descent + baseline
        y2 = line_y + self.ascent + baseline
        x1 = line_x

        for start, end, decoration in context.decoration_iter.ranges(i, i + n_glyphs):
            bg, underline = decoration
            x2 = x1
            for (kern, glyph, glyph_pos) in self.glyphs[start - i:end - i]:
                x2 += glyph.advance + kern + glyph_pos.x_advance

            if bg is not None:
                if len(bg) != 4:
                    msg = f"Background color requires 4 values (R, G, B, A). Value received: {bg}"
                    raise ValueError(msg)

                background_vertices.extend([x1, y1, 0, x2, y1, 0, x2, y2, 0, x1, y2, 0])
                background_colors.extend(bg * 4)

            if underline is not None:
                if len(underline) != 4:
                    msg = f"Underline color requires 4 values (R, G, B, A). Value received: {underline}"
                    raise ValueError(msg)

                underline_vertices.extend([x1, line_y + baseline - 2, 0, x2, line_y + baseline - 2, 0])
                underline_colors.extend(underline * 2)

            x1 = x2

        if background_vertices:
            bg_count = len(background_vertices) // 3
            background_indices = [(0, 1, 2, 0, 2, 3)[i % 6] for i in range(bg_count * 3)]
            decoration_program = get_default_decoration_shader()
            background_list = decoration_program.vertex_list_indexed(bg_count, GL_TRIANGLES, background_indices,
                                                                     layout.batch, layout.background_decoration_group,
                                                                     position=("f", background_vertices),
                                                                     translation=("f", t_position * bg_count),
                                                                     colors=("Bn", background_colors),
                                                                     rotation=("f", (rotation,) * bg_count),
                                                                     visible=("f", (visible,) * bg_count),
                                                                     anchor=("f", (anchor_x, anchor_y) * bg_count))
            self._add_vertex_list(background_list, context)

        if underline_vertices:
            ul_count = len(underline_vertices) // 3
            decoration_program = get_default_decoration_shader()
            underline_list = decoration_program.vertex_list(ul_count, GL_LINES,
                                                            layout.batch, layout.foreground_decoration_group,
                                                            position=("f", underline_vertices),
                                                            translation=("f", t_position * ul_count),
                                                            colors=("Bn", underline_colors),
                                                            rotation=("f", (rotation,) * ul_count),
                                                            visible=("f", (visible,) * ul_count),
                                                            anchor=("f", (anchor_x, anchor_y) * ul_count))
            self._add_vertex_list(underline_list, context)

    def update_scale(self, scale: float) -> None:
        scale_tuple = (scale, scale)
        for _vertex_list in self.vertex_lists:
            _vertex_list.scale[:] = scale_tuple * _vertex_list.count

    def update_visibility(self, visible: bool) -> None:
        visible_tuple = (visible,)
        for _vertex_list in self.vertex_lists:
            _vertex_list.visible[:] = visible_tuple * _vertex_list.count

    def __repr__(self) -> str:
        return f"_SDFGlyphBox({self.glyphs})"



class SDFTextLayout(base.TextLayout):  # noqa: D101
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

    def _initialize_groups(self) -> None:
        decoration_shader = get_default_decoration_shader()
        self.background_decoration_group = self.decoration_class(decoration_shader, order=0, parent=self._user_group)
        self.foreground_decoration_group = self.decoration_class(decoration_shader, order=2, parent=self._user_group)

    @property
    def scale(self) -> float:
        return self._scale

    @scale.setter
    def scale(self, value: float) -> None:
        self._scale = value
        for box in self._boxes:
            box.update_scale(self._scale)