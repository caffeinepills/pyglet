"""macOS image decoding through ImageIO and Core Graphics."""
from __future__ import annotations

from contextlib import contextmanager
from ctypes import Array, c_ubyte, c_void_p
from os import fsencode
from pathlib import Path
from typing import TYPE_CHECKING, BinaryIO

from pyglet.image import Animation, AnimationFrame, ImageData
from pyglet.image.codecs import ImageDecodeException, ImageDecoder
from pyglet.libs.darwin.cocoapy import NSMakeRect, cf, cg
from pyglet.libs.darwin.cocoapy import cfnumber_to_number, kCFAllocatorNull, kCGBlendModeCopy
from pyglet.libs.darwin.cocoapy import kCGImageAlphaPremultipliedLast
from pyglet.libs.darwin.cocoapy import kCGImagePropertyGIFDelayTime
from pyglet.libs.darwin.cocoapy import kCGImagePropertyGIFDictionary

if TYPE_CHECKING:
    from collections.abc import Iterator

    from pyglet.image.codecs import ImageEncoder


class ImageIODecoder(ImageDecoder):
    """Decode images with Image I/O and render pixels through Core Graphics."""

    def get_file_extensions(self) -> list[str]:
        # Image I/O can decode many more formats, but these are the most common.
        return [ '.bmp', '.cur', '.gif', '.ico', '.jp2', '.jpg', '.jpeg',
                 '.pcx', '.png', '.tga', '.tif', '.tiff', '.xbm', '.xpm' ]

    def get_animation_file_extensions(self) -> list[str]:
        return ['.gif']

    @staticmethod
    @contextmanager
    def _open_source(filename: str, file: BinaryIO | None) -> Iterator[int]:
        if file is None:
            path = fsencode(filename)
            url = cf.CFURLCreateFromFileSystemRepresentation(None, path, len(path), False)
            if url:
                try:
                    source = cg.CGImageSourceCreateWithURL(url, None)
                finally:
                    cf.CFRelease(url)
                if source:
                    try:
                        yield source
                    finally:
                        cf.CFRelease(source)
                    return

        # A failed URL open falls back to Python's file handling, preserving its I/O errors.
        # Keep the input bytes alive until both the source and CFData are released.
        file_bytes = Path(filename).read_bytes() if file is None else file.read()
        data = cf.CFDataCreateWithBytesNoCopy(None, file_bytes, len(file_bytes), kCFAllocatorNull)
        if not data:
            message = f"Could not allocate image data for {filename!r}."
            raise ImageDecodeException(message)
        try:
            source = cg.CGImageSourceCreateWithData(data, None)
            if not source:
                message = f"Could not open image {filename!r}."
                raise ImageDecodeException(message)
            try:
                yield source
            finally:
                cf.CFRelease(source)
        finally:
            cf.CFRelease(data)

    def _decode_frame(self, source: int, index: int) -> ImageData:
        image = cg.CGImageSourceCreateImageAtIndex(source, index, None)
        if not image:
            message = f"Could not decode image at index {index}."
            raise ImageDecodeException(message)

        try:
            width: int = cg.CGImageGetWidth(image)
            height: int = cg.CGImageGetHeight(image)
            pitch = 4 * width
            buffer: Array[c_ubyte] = (c_ubyte * (height * pitch))()

            # Normalize to 8-bit premultiplied RGBA for all input formats.
            color_space = cg.CGColorSpaceCreateDeviceRGB()
            if not color_space:
                raise ImageDecodeException("Could not create RGB color space.")
            try:
                # Create a bitmap context for the RGBA formatted data.
                # Note that premultiplied alpha is required:
                # http://developer.apple.com/library/mac/#qa/qa1037/_index.html
                bitmap = cg.CGBitmapContextCreate(buffer, width, height, 8, pitch, color_space,
                                                  kCGImageAlphaPremultipliedLast)
                if not bitmap:
                    raise ImageDecodeException("Could not create image bitmap.")
                try:
                    # The image covers the entire new bitmap, so blending is unnecessary.
                    cg.CGContextSetBlendMode(bitmap, kCGBlendModeCopy)
                    cg.CGContextDrawImage(bitmap, NSMakeRect(0, 0, width, height), image)
                finally:
                    cg.CGContextRelease(bitmap)
            finally:
                cg.CGColorSpaceRelease(color_space)
        finally:
            cg.CGImageRelease(image)

        return ImageData(width, height, 'RGBA', buffer, -pitch)

    def decode(self, filename: str, file: BinaryIO | None) -> ImageData:
        with self._open_source(filename, file) as source:
            return self._decode_frame(source, 0)

    def decode_animation(self, filename: str, file: BinaryIO | None) -> Animation:
        # If file is not an animated GIF, it will be loaded as a single-frame animation.
        frames: list[AnimationFrame] = []
        with self._open_source(filename, file) as source:
            count: int = cg.CGImageSourceGetCount(source)
            if not count:
                message = f"No image frames found in {filename!r}."
                raise ImageDecodeException(message)

            for index in range(count):
                duration = 0.1
                props = cg.CGImageSourceCopyPropertiesAtIndex(source, index, None)
                if props:
                    try:
                        gif_props = cf.CFDictionaryGetValue(props, kCGImagePropertyGIFDictionary)
                        if gif_props:
                            delay = cf.CFDictionaryGetValue(gif_props, kCGImagePropertyGIFDelayTime)
                            if delay:
                                duration = cfnumber_to_number(c_void_p(delay))
                    finally:
                        cf.CFRelease(props)
                image = self._decode_frame(source, index)
                frames.append(AnimationFrame(image, duration))

        return Animation(frames)


def get_decoders() -> list[ImageIODecoder]:
    """Return the ImageIO decoder."""
    return [ImageIODecoder()]


def get_encoders() -> list[ImageEncoder]:
    """Return the available ImageIO encoders."""
    return []
