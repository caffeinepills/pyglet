import pytest

import pyglet

from pyglet.graphics.atlas import AllocatorException, TextureAtlas, TextureBin
from pyglet.image import ImageData


def _solid_rgba_image(width: int, height: int, color: tuple[int, int, int, int]) -> ImageData:
    return ImageData(width, height, "RGBA", bytes(color) * (width * height))


def test_texture_atlas_add_with_border_and_upload(gl3_context):
    gl3_context.switch_to()

    atlas = TextureAtlas(width=8, height=8)
    image = _solid_rgba_image(2, 2, (255, 0, 0, 255))

    region = atlas.add(image, border=1)

    assert (region.x, region.y) == (1, 1)
    assert (region.width, region.height) == (2, 2)
    assert atlas.allocator.used_area == 16

    fetched = bytes(region.get_image_data().get_bytes("RGBA", region.width * 4))
    assert fetched == bytes([255, 0, 0, 255]) * 4


def test_texture_atlas_raises_when_no_space(gl3_context):
    gl3_context.switch_to()

    atlas = TextureAtlas(width=4, height=4)
    atlas.add(_solid_rgba_image(4, 4, (1, 2, 3, 255)))

    with pytest.raises(AllocatorException):
        atlas.add(_solid_rgba_image(1, 1, (9, 8, 7, 255)))


def test_texture_bin_creates_new_atlas_when_full(gl3_context):
    gl3_context.switch_to()

    texture_bin = TextureBin(texture_width=64, texture_height=64)
    image = _solid_rgba_image(64, 64, (10, 20, 30, 255))

    region_a = texture_bin.add(image)
    assert len(texture_bin.atlases) == 1

    region_b = texture_bin.add(image)
    assert len(texture_bin.atlases) == 2
    assert region_a.owner is texture_bin.atlases[0].texture
    assert region_b.owner is texture_bin.atlases[1].texture
