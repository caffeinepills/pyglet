import unittest

from pyglet.graphics import Texture3D
from pyglet.graphics.texture import TextureArray
from pyglet.image import ImageData
from pyglet.window import Window
from tests.annotations import GraphicsAPIGroups, require_graphics_api


pytestmark = require_graphics_api(GraphicsAPIGroups.GL3)


def colorbyte(color):
    return bytes((color,))


class TestTextureMipmapsArrays(unittest.TestCase):
    def setUp(self):
        self.w = Window(visible=False)

    def tearDown(self) -> None:
        self.w.close()

    def create_image(self, width, height, color):
        pixel = colorbyte(color) * 3 + colorbyte(255)
        data = pixel * (width * height)
        return ImageData(width, height, 'RGBA', data)

    def test_texture3d_mipmap_depth(self):
        images = [self.create_image(8, 4, i + 1) for i in range(4)]
        texture = Texture3D.create_for_images(images)
        self.assertEqual(texture._get_mipmap_depth(0), 4)
        self.assertEqual(texture._get_mipmap_depth(1), 2)
        self.assertEqual(texture._get_mipmap_depth(2), 1)
        texture.init_mipmaps()
        self.assertEqual(texture.mipmap_count, 4)

        texture.upload(self.create_image(8, 4, 7), 0, 0, 0, level=0)
        texture.upload(self.create_image(4, 2, 8), 0, 0, 0, level=1)
        self.assertIn(0, texture.valid_mipmaps)
        self.assertIn(1, texture.valid_mipmaps)

    def test_texture_array_mipmap_depth(self):
        images = [self.create_image(8, 4, i + 1) for i in range(4)]
        texture = TextureArray.create_for_images(images)
        self.assertEqual(texture._get_mipmap_depth(0), 4)
        self.assertEqual(texture._get_mipmap_depth(2), 4)
        texture.init_mipmaps()
        self.assertEqual(texture.mipmap_count, 4)

        texture.upload(self.create_image(8, 4, 7), 0, 0, 0, level=0)
        texture.upload(self.create_image(4, 2, 8), 0, 0, 0, level=1)
        self.assertIn(0, texture.valid_mipmaps)
        self.assertIn(1, texture.valid_mipmaps)
