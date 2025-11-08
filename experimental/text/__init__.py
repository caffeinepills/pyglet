from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pyglet
from experimental.text.layout_sdf import SDFTextLayout
from pyglet.font import UserDefinedFontBase
from pyglet.font.base import Font, FontException, Glyph, GlyphPosition, get_grapheme_clusters
from pyglet.text import decode_text

if TYPE_CHECKING:
    from pyglet.graphics import Batch, Group, ShaderProgram
    from pyglet.customtypes import AnchorX, AnchorY, ContentVAlign
    from pathlib import Path
    from pyglet.text.document import AbstractDocument

    from pyglet.image import Texture


class DistFieldFont(UserDefinedFontBase):
    """An example of a Single Distance Field font being rendered.

    Requires the msdf-atlas-gen utility: https://github.com/Chlumsky/msdf-atlas-gen

    You will need to pass your font file to create an atlas and JSON data. For example::
        msdf-atlas-gen -font "YourFont.ttf" -imageout "atlas.png" -json "atlas.json" -type msdf -font-size 32
    """
    def __init__(self, name: str, sdf_filename: str, size: int,
            weight: str = "normal", italic: bool = False, stretch: bool = False, dpi: int = 96, locale: str | None = None,) -> None:
        """Initialize the font object.

        Args:
            name:
                A unique name for the font, must not collide with any other system or loaded font.
            sdf_filename:
                The filename path of the SDF-atlas image. The JSON file must be named the same.
            size:
                This is not the size of your font, but an identifier when passing to a label to match this
                font object. Use label.scale to change the actual size of the font.
            weight:
                This is not the weight of your font, but an identifier when passing to a label to match this
                font object.
            italic:
                This is not the italic of your font, but an identifier when passing to a label to match this
                font object.
            stretch:
                This is not the stretch of your font, but an identifier when passing to a label to match this
                font object.
            dpi:
                This is not the DPI of your font, but an identifier when passing to a label to match this
                font object.
            locale:
                This is not the locale of your font, but an identifier when passing to a label to match this
                font object.
        """
        self.atlas, self.atlas_info, self.data = self._load_sdf_atlas(
            f"{sdf_filename}.png", f"{sdf_filename}.json"
        )

        metrics = self.data.get("metrics", {})
        if not metrics:
            raise FontException("JSON file does not contain font metrics.")

        raster_size = self.data["atlas"]["size"]
        ascent = metrics.get("ascender") * raster_size
        descent = metrics.get("descender") * raster_size

        empty_glyph = self.atlas.get_region(0, 0, 1, 1)
        self.empty_glyph = Glyph(
            empty_glyph.x, empty_glyph.y, 0, empty_glyph.width, empty_glyph.height, empty_glyph.owner
        )
        self.empty_glyph.set_bearings(0, 0, -1)

        self.space_info = self.atlas_info.get(ord(" "))
        self.space_glyph = Glyph(
            empty_glyph.x, empty_glyph.y, 0, empty_glyph.width, empty_glyph.height, empty_glyph.owner
        )
        self.space_glyph.set_bearings(0, 0, self.space_info["advance"] * raster_size)

        super().__init__(name, " ", size, ascent, descent, weight, italic, stretch, dpi, locale)
        self.pixel_size = raster_size

    def _load_sdf_atlas(self, image_path: str | Path, json_path: str | Path) -> tuple[Texture, dict, dict]:
        """Load an SDF atlas image and its JSON mapping file from msdf-atlas-gen.

        The JSON format expected is msdf-atlas-gen.

        Args:
            image_path:
                Path to the atlas image file.
            json_path:
                Path to the JSON mapping file.

        Returns:
            dict: A mapping from each glyph's Unicode codepoint (int) to a dictionary containing:
                - 'advance': the glyph's advance value.
                - 'planeBounds': the optional plane bounds (if available).
                - 'atlasBounds': the raw atlas bounds from the JSON.
                - 'region': a pyglet.image.TextureRegion extracted from the image (or None if not defined).
        """
        atlas_image = pyglet.resource.image(image_path)
        atlas_json = pyglet.resource.file(json_path, 'r')
        data = json.load(atlas_json)

        glyph_data = {}
        for glyph in data.get("glyphs", []):
            codepoint = glyph.get("unicode")
            entry = {
                "advance": glyph.get("advance"),
                "planeBounds": glyph.get("planeBounds"),
                "atlasBounds": glyph.get("atlasBounds"),
                "region": None,  # Default if no atlasBounds available.
            }
            if "atlasBounds" in glyph:
                bounds = glyph["atlasBounds"]
                x = bounds["left"]
                y = bounds["bottom"]
                width = bounds["right"] - bounds["left"]
                height = bounds["top"] - bounds["bottom"]
                entry["region"] = atlas_image.get_region(x, y, width, height)

            glyph_data[codepoint] = entry

        return atlas_image, glyph_data, data

    def get_glyphs(self, text: str) -> tuple[list[Glyph], list[GlyphPosition]]:
        glyphs = []

        for c in get_grapheme_clusters(str(text)):
            # Get the glyph for 'c'.  Hide tabs (Windows and Linux render boxes)
            if c == "\t":
                c = " "  # noqa: PLW2901
            sdf_info = self.atlas_info.get(ord(c))
            if not sdf_info:
                # Go to default char if one is not found.
                glyph = self.empty_glyph
            else:
                if c == " ":
                    glyph = self.space_glyph
                else:
                    sdf_image = sdf_info["region"]
                    sdf_advance = sdf_info["advance"]
                    sdf_plane = sdf_info["planeBounds"]
                    glyph = Glyph(sdf_image.x, sdf_image.y, 0, sdf_image.width, sdf_image.height, sdf_image.owner)
                    glyph.set_bearings(-sdf_plane["bottom"] * self.pixel_size, sdf_plane["left"] * self.pixel_size, sdf_advance * self.pixel_size)

            glyphs.append(glyph)

        return glyphs, [GlyphPosition(0, 0, 0, 0)] * len(glyphs)


class DocumentSDFLabel(SDFTextLayout):
    """Base label class.

    A label is a layout that exposes convenience methods for manipulating the
    associated document.
    """

    def __init__(
            self, document: AbstractDocument,
            x: float = 0.0, y: float = 0.0, z: float = 0.0,
            width: int | None = None, height: int | None = None,
            anchor_x: AnchorX = "left", anchor_y: AnchorY = "baseline", rotation: float = 0.0,
            multiline: bool = False, dpi: int | None = None,
            batch: Batch | None = None, group: Group | None = None,
            program: ShaderProgram | None = None,
            init_document: bool = True,
    ) -> None:
        """Create a label for a given document.

        Args:
            document: Document to attach to the layout.
            x: X coordinate of the label.
            y: Y coordinate of the label.
            z: Z coordinate of the label.
            width: Width of the label in pixels, or ``None``
            height:  Height of the label in pixels, or ``None``
            anchor_x:
                Anchor point of the X coordinate: one of
                ``"left"``, `"center"`` or ``"right"``.
            anchor_y:
                Anchor point of the Y coordinate: one of
                ``"bottom"``, ``"baseline"``, ``"center"`` or ``"top"``.
            rotation:
                The amount to rotate the label in degrees. A
                positive amount will be a clockwise rotation, negative
                values will result in counter-clockwise rotation.
            multiline:
                If ``True``, the label will be word-wrapped and
                accept newline characters. You must also set the width
                of the label.
            dpi: Resolution of the fonts in this layout. Defaults to 96.
            batch: Optional graphics batch to add the label to.
            group: Optional graphics group to use.
            program: Optional graphics shader to use. Will affect all glyphs.
            init_document:
                If ``True``, the document will be initialized. If you
                are passing an already-initialized document, then you can
                avoid duplicating work by setting this to ``False``.
        """
        super().__init__(document, x, y, z, width, height, anchor_x, anchor_y, rotation,
                         multiline, dpi, batch, group, program, init_document=init_document)

    @property
    def text(self) -> str:
        """The text of the label."""
        return self.document.text

    @text.setter
    def text(self, text: str) -> None:
        self.document.text = text

    @property
    def color(self) -> tuple[int, int, int, int]:
        """Text color.

        Color is a 4-tuple of RGBA components, each in range [0, 255].
        """
        return self.document.get_style("color")

    @color.setter
    def color(self, color: tuple[int, int, int, int]) -> None:
        r, g, b, *a = color
        color = r, g, b, a[0] if a else 255
        self.document.set_style(0, len(self.document.text), {"color": color})

    @property
    def opacity(self) -> int:
        """Blend opacity.

        This property sets the alpha component of the colour of the label's
        vertices.  With the default blend mode, this allows the layout to be
        drawn with fractional opacity, blending with the background.

        An opacity of 255 (the default) has no effect.  An opacity of 128 will
        make the label appear semi-translucent.
        """
        return self.color[3]

    @opacity.setter
    def opacity(self, alpha: int) -> None:
        if alpha != self.color[3]:
            self.color = list(map(int, (*self.color[:3], alpha)))

    @property
    def font(self) -> Font:
        """Font object instance.

        This is the backend specific font object being used at the start of the Label.

        (Read Only)
        """
        return self.document.get_font(0)

    @property
    def font_name(self) -> str | list[str]:
        """Font family name.

        The font name, as passed to :py:func:`pyglet.font.load`.  A list of names can
        optionally be given: the first matching font will be used.
        """
        return self.document.get_style("font_name")

    @font_name.setter
    def font_name(self, font_name: str | list[str]) -> None:
        self.document.set_style(0, len(self.document.text), {"font_name": font_name})

    @property
    def font_size(self) -> float:
        """Font size, in points."""
        return self.document.get_style("font_size")

    @font_size.setter
    def font_size(self, font_size: float) -> None:
        self.document.set_style(0, len(self.document.text), {"font_size": font_size})

    @property
    def weight(self) -> str:
        """The font weight (boldness or thickness), as a string.

        See the :py:class:`~Weight` enum for valid cross-platform
        string values.
        """
        return self.document.get_style("weight")

    @weight.setter
    def weight(self, weight: str) -> None:
        self.document.set_style(0, len(self.document.text), {"weight": str(weight)})

    @property
    def italic(self) -> bool | str:
        """Italic font style."""
        return self.document.get_style("italic")

    @italic.setter
    def italic(self, italic: bool | str) -> None:
        self.document.set_style(0, len(self.document.text), {"italic": italic})

    def get_style(self, name: str) -> Any:
        """Get a document style value by name.

        If the document has more than one value of the named style,
        `pyglet.text.document.STYLE_INDETERMINATE` is returned.

        Args:
            name:
                Style name to query.  See documentation from `pyglet.text.layout` for known style names.
        """
        return self.document.get_style_range(name, 0, len(self.document.text))

    def set_style(self, name: str, value: Any) -> None:
        """Set a document style value by name over the whole document.

        Args:
            name:
                Name of the style to set.  See documentation for
                `pyglet.text.layout` for known style names.
            value:
                Value of the style.
        """
        self.document.set_style(0, len(self.document.text), {name: value})

    def __del__(self) -> None:
        self.delete()



class DistFieldLabel(DocumentSDFLabel):
    """Plain text label."""

    def __init__(
            self, text: str = "",
            x: float = 0.0, y: float = 0.0, z: float = 0.0,
            width: int | None = None, height: int | None = None,
            anchor_x: AnchorX = "left", anchor_y: AnchorY = "baseline", rotation: float = 0.0,
            multiline: bool = False, dpi: int | None = None,
            font_name: str | None = None, font_size: float | None = None,
            weight: str = "normal", italic: bool | str = False, stretch: bool | str = False,
            color: tuple[int, int, int, int] | tuple[int, int, int] = (255, 255, 255, 255),
            align: ContentVAlign = "left",
            batch: Batch | None = None, group: Group | None = None,
            program: ShaderProgram | None = None,
    ) -> None:
        """Create a plain text label.

        Args:
            text:
                Text to display.
            x:
                X coordinate of the label.
            y:
                Y coordinate of the label.
            z:
                Z coordinate of the label.
            width:
                Width of the label in pixels, or None
            height:
                Height of the label in pixels, or None
            anchor_x:
                Anchor point of the X coordinate: one of ``"left"``,
                ``"center"`` or ``"right"``.
            anchor_y:
                Anchor point of the Y coordinate: one of ``"bottom"``,
                ``"baseline"``, ``"center"`` or ``"top"``.
            rotation:
                The amount to rotate the label in degrees. A positive amount
                will be a clockwise rotation, negative values will result in
                counter-clockwise rotation.
            multiline:
                If True, the label will be word-wrapped and accept newline
                characters.  You must also set the width of the label.
            dpi:
                Resolution of the fonts in this layout.  Defaults to 96.
            font_name:
                Font family name(s).  If more than one name is given, the
                first matching name is used.
            font_size:
                Font size, in points.
            weight:
                The 'weight' of the font (boldness). See the :py:class:`~Weight`
                enum for valid cross-platform weight names.
            italic:
                Italic font style.
            stretch:
                 Stretch font style.
            color:
                Font color as RGBA or RGB components, each within
                ``0 <= component <= 255``.
            align:
                Horizontal alignment of text on a line, only applies if
                a width is supplied. One of ``"left"``, ``"center"``
                or ``"right"``.
            batch:
                Optional graphics batch to add the label to.
            group:
                Optional graphics group to use.
            program:
                Optional graphics shader to use. Will affect all glyphs.
        """
        doc = decode_text(text)
        r, g, b, *a = color
        rgba = r, g, b, a[0] if a else 255

        super().__init__(doc, x, y, z, width, height, anchor_x, anchor_y, rotation,
                         multiline, dpi, batch, group, program, init_document=False)

        self.document.set_style(0, len(self.document.text), {
            "font_name": font_name,
            "font_size": font_size,
            "weight": weight,
            "italic": italic,
            "stretch": stretch,
            "color": rgba,
            "align": align,
        })