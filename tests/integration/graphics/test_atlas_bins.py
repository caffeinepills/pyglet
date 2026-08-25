import pytest

import pyglet

from pyglet.graphics.atlas import AllocatorException, TextureAtlas, TextureBin, TextureArrayBin
from pyglet.graphics.texture import TextureArraySizeExceeded
from pyglet.image import ImageData
from tests.annotations import GraphicsAPIGroups, skip_graphics_api


def _solid_rgba_image(width: int, height: int, color: tuple[int, int, int, int]) -> ImageData:
    return ImageData(width, height, "RGBA", bytes(color) * (width * height))


def test_texture_atlas_add_with_border_and_upload(test_window):
    test_window.switch_to()

    atlas = TextureAtlas(width=8, height=8)
    image = _solid_rgba_image(2, 2, (255, 0, 0, 255))

    region = atlas.add(image, border=1)

    assert (region.x, region.y) == (1, 1)
    assert (region.width, region.height) == (2, 2)
    assert atlas.allocator.used_area == 16

    fetched = bytes(region.get_image_data().get_bytes("RGBA", region.width * 4))
    assert fetched == bytes([255, 0, 0, 255]) * 4


def test_texture_atlas_raises_when_no_space(test_window):
    test_window.switch_to()

    atlas = TextureAtlas(width=4, height=4)
    atlas.add(_solid_rgba_image(4, 4, (1, 2, 3, 255)))

    with pytest.raises(AllocatorException):
        atlas.add(_solid_rgba_image(1, 1, (9, 8, 7, 255)))


def test_texture_bin_creates_new_atlas_when_full(test_window):
    test_window.switch_to()

    texture_bin = TextureBin(texture_width=64, texture_height=64)
    image = _solid_rgba_image(64, 64, (10, 20, 30, 255))

    region_a = texture_bin.add(image)
    assert len(texture_bin.atlases) == 1

    region_b = texture_bin.add(image)
    assert len(texture_bin.atlases) == 2
    assert region_a.owner is texture_bin.atlases[0].texture
    assert region_b.owner is texture_bin.atlases[1].texture


@skip_graphics_api(GraphicsAPIGroups.GL2)
def test_texture_array_bin_creates_new_array_when_depth_full(test_window):
    test_window.switch_to()

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


@skip_graphics_api(GraphicsAPIGroups.GL2)
def test_texture_array_bin_binds_existing_array_before_upload(test_window):
    test_window.switch_to()

    array_bin = TextureArrayBin(texture_width=4, texture_height=4, max_depth=2)
    image_a = _solid_rgba_image(4, 4, (1, 2, 3, 255))
    image_b = _solid_rgba_image(4, 4, (4, 5, 6, 255))
    image_c = _solid_rgba_image(4, 4, (7, 8, 9, 255))
    image_d = _solid_rgba_image(4, 4, (10, 11, 12, 255))

    array_bin.add(image_a)
    array_bin.add(image_b)
    array_bin.add(image_c)

    # Bind a different array before uploading into the existing second array.
    array_bin.arrays[0].bind()
    region_d = array_bin.add(image_d)

    assert region_d.owner is array_bin.arrays[1]
    fetched = bytes(region_d.get_image_data().get_bytes("RGBA", region_d.width * 4))
    assert fetched == bytes([10, 11, 12, 255]) * 16


@skip_graphics_api(GraphicsAPIGroups.GL2)
def test_texture_array_bin_raises_for_oversized_image(test_window):
    test_window.switch_to()

    array_bin = TextureArrayBin(texture_width=4, texture_height=4, max_depth=2)

    with pytest.raises(TextureArraySizeExceeded):
        array_bin.add(_solid_rgba_image(5, 4, (1, 2, 3, 255)))
