import os.path
from contextlib import nullcontext
from typing import BinaryIO

from pyglet.image import ImageData
from pyglet.image.codecs import ImageDecodeException, ImageDecoder, ImageEncodeException, ImageEncoder

from PIL import Image


class PILImageDecoder(ImageDecoder):
    def get_file_extensions(self):
        # Only most common ones shown here
        return ['.bmp', '.cur', '.gif', '.ico', '.jpg', '.jpeg', '.pcx', '.png',
                '.tga', '.tif', '.tiff', '.xbm', '.xpm']

    # def get_animation_file_extensions(self):
    #     return ['.gif', '.ani']

    def decode(self, filename: str, file: BinaryIO | None) -> ImageData:
        with (open(filename, 'rb') if file is None else nullcontext(file)) as stream:
            try:
                with Image.open(stream) as source:
                    # Convert bitmap and palette images to component
                    image = source.convert() if source.mode in ('1', 'P') else source
                    mode = image.mode
                    if mode not in ('L', 'LA', 'RGB', 'RGBA'):
                        raise ImageDecodeException('Unsupported mode "%s"' % mode)
                    width, height = image.size
                    # Export packed rows bottom-up without allocating a transposed image.
                    pixels = image.tobytes('raw', mode, 0, -1)
            except ImageDecodeException:
                raise
            except Exception as e:
                raise ImageDecodeException('PIL cannot read %r: %s' % (filename or file, e)) from e

        return ImageData(width, height, mode, pixels)


class PILImageEncoder(ImageEncoder):
    def get_file_extensions(self):
        # Most common only
        return ['.bmp', '.eps', '.gif', '.jpg', '.jpeg',
                '.pcx', '.png', '.ppm', '.tiff', '.xbm']

    def encode(self, image, filename, file):
        # File format is guessed from filename extension, otherwise defaults to PNG.
        pil_format = (filename and os.path.splitext(filename)[1][1:]) or 'png'

        if pil_format.lower() == 'jpg':
            pil_format = 'JPEG'

        image = image.get_image_data()
        fmt = image.format
        if fmt != 'RGB':
            # Only save in RGB or RGBA formats.
            fmt = 'RGBA'
        pitch = -(image.width * len(fmt))

        # fromstring is deprecated, replaced by frombytes in Pillow (PIL fork)
        # (1.1.7) PIL still uses it
        try:
            image_from_fn = getattr(Image, "frombytes")
        except AttributeError:
            image_from_fn = getattr(Image, "fromstring")
        pil_image = image_from_fn(fmt, (image.width, image.height), image.get_bytes(fmt, pitch))

        try:
            pil_image.save(file, pil_format)
        except Exception as e:
            raise ImageEncodeException(e)


def get_decoders():
    return [PILImageDecoder()]


def get_encoders():
    return [PILImageEncoder()]
