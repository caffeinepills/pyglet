import pytest

from pyglet.graphics.atlas import TextureArrayBin
from pyglet.graphics.texture import TextureArraySizeExceeded
from pyglet.image import ImageData
from tests.annotations import GraphicsAPIGroups, require_graphics_api


pytestmark = require_graphics_api(GraphicsAPIGroups.GL3)


def _solid_rgba_image(width: int, height: int, color: tuple[int, int, int, int]) -> ImageData:
    return ImageData(width, height, "RGBA", bytes(color) * (width * height))


def test_texture_array_bin_creates_new_array_when_depth_full(gl3_context):
    gl3_context.switch_to()

    array_bin = TextureArrayBin(texture_width=4, texture_height=4, max_depth=2)
    image = _solid_rgba_image(4, 4, (1, 2, 3, 255))

    region_0 = array_bin.add(image)
    region_1 = array_bin.add(image)

    assert len(array_bin.arrays) == 1
    assert region_0.z == 0
    assert region_1.z == 1

    region_2 = array_bin.add(image)
    assert len(array_bin.arrays) == 2
    assert region_2.z == 0
    assert region_2.owner is array_bin.arrays[1]


def test_texture_array_bin_raises_for_oversized_image(gl3_context):
    gl3_context.switch_to()

    array_bin = TextureArrayBin(texture_width=4, texture_height=4, max_depth=2)

    with pytest.raises(TextureArraySizeExceeded):
        array_bin.add(_solid_rgba_image(5, 4, (1, 2, 3, 255)))
