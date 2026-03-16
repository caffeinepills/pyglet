from __future__ import annotations
import re

from pyglet.enums import TextureType
from pyglet.image.base import ImageData, ImageException, TextureBase, TextureDescriptor, TextureInternalFormat, \
    AbstractImage
from pyglet.util import asbytes


class VulkanImageData(AbstractImage):
    # !!! THIS SHOULD PROBABLY BE BACKEND GENERIC.
    """An image represented as a string of unsigned bytes."""

    _swap1_pattern = re.compile(asbytes('(.)'), re.DOTALL)
    _swap2_pattern = re.compile(asbytes('(.)(.)'), re.DOTALL)
    _swap3_pattern = re.compile(asbytes('(.)(.)(.)'), re.DOTALL)
    _swap4_pattern = re.compile(asbytes('(.)(.)(.)(.)'), re.DOTALL)

    _current_texture = None
    _current_mipmap_texture = None

    def __init__(self, width: int, height: int, fmt: str, data: bytes, pitch: int | None = None):
        """Initialise image data.

        Args:
            width:
                Width of image data
            height:
                Height of image data
            fmt:
                A valid format string, such as 'RGB', 'RGBA', 'ARGB', etc.
            data:
                A sequence of bytes containing the raw image data.
            pitch:
                If specified, the number of bytes per row.  Negative values
                indicate a top-to-bottom arrangement.  Defaults to
                ``width * len(format)``.
        """
        super().__init__(width, height)

        self._current_format = self._desired_format = fmt.upper()
        self._current_data = data
        self.pitch = pitch or width * len(fmt)
        self._current_pitch = self.pitch
        self.mipmap_images = []

    def __getstate__(self):
        return {
            'width': self.width,
            'height': self.height,
            '_current_data': self.get_bytes(self._current_format, self._current_pitch),
            '_current_format': self._current_format,
            '_desired_format': self._desired_format,
            '_current_pitch': self._current_pitch,
            'pitch': self.pitch,
            'mipmap_images': self.mipmap_images,
        }

    def get_image_data(self) -> VulkanImageData:
        return self

    @property
    def format(self) -> str:
        """Format string of the data. Read-write."""
        return self._desired_format

    @format.setter
    def format(self, fmt: str):
        self._desired_format = fmt.upper()
        self._current_texture = None

    def get_bytes(self, fmt: str | None = None, pitch: int | None = None) -> bytes:
        """Get the byte data of the image

        This method returns the raw byte data of the image, with optional conversion.
        To convert the data into another format, you can provide ``fmt`` and ``pitch``
        arguments. For example, if the image format is ``RGBA``, and you wish to get
        the byte data in ``RGB`` format::

            rgb_pitch = my_image.width // len('RGB')
            rgb_img_bytes = my_image.get_bytes(fmt='RGB', pitch=rgb_pitch)

        The image ``pitch`` may be negative, so be sure to check that when converting
        to another format. Switching the sign of the ``pitch`` will cause the image
        to appear "upside-down".

        Args:
            fmt:
                If provided, get the data in another format.
            pitch:
                The number of bytes per row. This generally means the length
                of the format string * the number of pixels per row.
                Negative values indicate a top-to-bottom arrangement.

        Note:
             Conversion to another format is done on the CPU, and can be
             somewhat costly for larger images. Consider performing conversion
             at load time for framerate sensitive applictions.
        """
        fmt = fmt or self._desired_format
        pitch = pitch or self._current_pitch

        if fmt == self._current_format and pitch == self._current_pitch:
            return self._current_data
        return self._convert(fmt, pitch)

    def set_bytes(self, fmt: str, pitch: int, data: bytes) -> None:
        """Set the byte data of the image.

        Args:
            fmt:
                The format string of the supplied data.
                For example: "RGB" or "RGBA"
            pitch:
                The number of bytes per row. This generally means the length
                of the format string * the number of pixels per row.
                Negative values indicate a top-to-bottom arrangement.
            data:
                Image data as bytes.
        """
        self._current_format = fmt
        self._current_pitch = pitch
        self._current_data = data
        self._current_texture = None
        self._current_mipmap_texture = None

    def get_data(self, fmt: str | None = None, pitch: int | None = None) -> bytes:
        """Get the byte data of the image.

        Warning:
            This method is deprecated and will be removed in the next version.
            Use :py:meth:`~get_bytes` instead.
        """
        return self.get_bytes(fmt, pitch)

    def set_data(self, fmt: str, pitch: int, data: bytes) -> None:
        """Set the byte data of the image.

        Warning:
            This method is deprecated and will be removed in the next version.
            Use :py:meth:`~set_bytes` instead.
        """
        self.set_bytes(fmt, pitch, data)

    def set_mipmap_image(self, level: int, image: AbstractImage) -> None:
        """Set a user-defined mipmap image for a particular level.

        These mipmap images will be applied to textures obtained via
        :py:meth:`~get_mipmapped_texture`, instead of automatically
        generated images for each level.

        Args:
            level:
                Mipmap level to set image at, must be >= 1.
            image:
                Image to set.  Must have correct dimensions for that mipmap
                level (i.e., width >> level, height >> level)
        """
        if level == 0:
            msg = 'Cannot set mipmap image at level 0 (it is this image)'
            raise ImageException(msg)

        # Check dimensions of mipmap
        width, height = self.width, self.height
        for i in range(level):
            width >>= 1
            height >>= 1
        if width != image.width or height != image.height:
            raise ImageException(f"Mipmap image has wrong dimensions for level {level}")

        # Extend mipmap_images list to required level
        self.mipmap_images += [None] * (level - len(self.mipmap_images))
        self.mipmap_images[level - 1] = image

    def create_texture(self, cls: type[TextureBase]) -> TextureBase:
        """Given a texture class, create a texture containing this image."""
        #internalformat = self._get_internalformat(self._desired_format)
        descriptor = TextureDescriptor(tex_type=TextureType.TYPE_2D)
        texture = cls.create(self.width, self.height, descriptor, blank_data=False)
        if self.anchor_x or self.anchor_y:
            texture.anchor_x = self.anchor_x
            texture.anchor_y = self.anchor_y

        texture.upload_data(self)

        #self.blit_to_texture(texture.target, texture.level, self.anchor_x, self.anchor_y, 0, None)

        return texture

    def get_texture(self) -> TextureBase:
        if not self._current_texture:
            from pyglet.graphics.api.vulkan.texture import VulkanTexture
            self._current_texture = self.create_texture(VulkanTexture)
        return self._current_texture

    def get_mipmapped_texture(self) -> TextureBase:
        ...

    def get_region(self, x: int, y: int, width: int, height: int) -> GLImageDataRegion:
        """Retrieve a rectangular region of this image data."""
        return GLImageDataRegion(x, y, width, height, self)

    def blit(self, x: int, y: int, z: int = 0, width: int | None = None, height: int | None = None) -> None:
        self.get_texture().blit(x, y, z, width, height)

    def blit_into(self, source, x: int, y: int, z: int) -> None:
        raise NotImplementedError(f"Not implemented for {self}")

    def blit_to_texture(self, target: int, level: int, x: int, y: int, z: int, internalformat: int = None):
        """Draw this image to the currently bound texture at ``target``.

        This image's anchor point will be aligned to the given ``x`` and ``y``
        coordinates.  If the currently bound texture is a 3D texture, the ``z``
        parameter gives the image slice to blit into.

        If ``internalformat`` is specified, ``glTexImage`` is used to initialise
        the texture; otherwise, ``glTexSubImage`` is used to update a region.
        """
        x -= self.anchor_x
        y -= self.anchor_y

        data_format = self.format
        data_pitch = abs(self._current_pitch)

        # Determine pixel format from format string
        fmt, gl_type = self._get_gl_format_and_type(data_format)

        if fmt is None:
            # Need to convert data to a standard form
            data_format = {
                1: 'R',
                2: 'RG',
                3: 'RGB',
                4: 'RGBA',
            }.get(len(data_format))
            fmt, gl_type = self._get_gl_format_and_type(data_format)

        # Get data in required format (hopefully will be the same format it's already
        # in, unless that's an obscure format, upside-down or the driver is old).
        data = self._convert(data_format, data_pitch)

        if data_pitch & 0x1:
            align = 1
        elif data_pitch & 0x2:
            align = 2
        else:
            align = 4
        row_length = data_pitch // len(data_format)

        glPixelStorei(GL_UNPACK_ALIGNMENT, align)
        glPixelStorei(GL_UNPACK_ROW_LENGTH, row_length)
        self._apply_region_unpack()

        if target == GL_TEXTURE_3D or target == GL_TEXTURE_2D_ARRAY:
            assert not internalformat
            glTexSubImage3D(target, level,
                            x, y, z,
                            self.width, self.height, 1,
                            fmt, gl_type,
                            data)
        elif internalformat:
            glTexImage2D(target, level,
                         internalformat,
                         self.width, self.height,
                         0,
                         fmt, gl_type,
                         data)
        else:
            glTexSubImage2D(target, level,
                            x, y,
                            self.width, self.height,
                            fmt, gl_type,
                            data)

        # Unset GL_UNPACK_ROW_LENGTH:
        glPixelStorei(GL_UNPACK_ROW_LENGTH, 0)
        self._default_region_unpack()

        # Flush image upload before data get GC'd:
        glFlush()

    def _apply_region_unpack(self):
        # Not needed on full images
        pass

    def _default_region_unpack(self):
        # Not needed on full images
        pass

    def _convert(self, fmt: str, pitch: int) -> bytes:
        """Return data in the desired format.

        This method does not alter this instance's current format or pitch.
        """
        if fmt == self._current_format and pitch == self._current_pitch:
            return self._current_data

        self._ensure_bytes()
        data = self._current_data
        current_pitch = self._current_pitch
        current_format = self._current_format
        sign_pitch = current_pitch // abs(current_pitch)
        if fmt != self._current_format:
            # Create replacement string, e.g. r'\4\1\2\3' to convert RGBA to ARGB
            repl = asbytes('')
            for c in fmt:
                try:
                    idx = current_format.index(c) + 1
                except ValueError:
                    idx = 1
                repl += asbytes(r'\%d' % idx)

            if len(current_format) == 1:
                swap_pattern = self._swap1_pattern
            elif len(current_format) == 2:
                swap_pattern = self._swap2_pattern
            elif len(current_format) == 3:
                swap_pattern = self._swap3_pattern
            elif len(current_format) == 4:
                swap_pattern = self._swap4_pattern
            else:
                raise ImageException('Current image format is wider than 32 bits.')

            packed_pitch = self.width * len(current_format)
            if abs(self._current_pitch) != packed_pitch:
                # Pitch is wider than pixel data, need to go row-by-row.
                new_pitch = abs(self._current_pitch)
                rows = [data[i:i + new_pitch] for i in range(0, len(data), new_pitch)]
                rows = [swap_pattern.sub(repl, r[:packed_pitch]) for r in rows]
                data = b''.join(rows)
            else:
                # Rows are tightly packed, apply regex over whole image.
                data = swap_pattern.sub(repl, data)

            # After conversion, rows will always be tightly packed
            current_pitch = sign_pitch * (len(fmt) * self.width)

        if pitch != current_pitch:
            diff = abs(current_pitch) - abs(pitch)
            if diff > 0:
                # New pitch is shorter than old pitch, chop bytes off each row
                new_pitch = abs(pitch)
                rows = [data[i:i + new_pitch - diff] for i in range(0, len(data), new_pitch)]
                data = b''.join(rows)

            elif diff < 0:
                # New pitch is longer than old pitch, add '0' bytes to each row
                new_pitch = abs(current_pitch)
                padding = bytes(1) * -diff
                rows = [data[i:i + new_pitch] + padding for i in range(0, len(data), new_pitch)]
                data = b''.join(rows)

            if current_pitch * pitch < 0:
                # Pitch differs in sign, swap row order
                new_pitch = abs(pitch)
                rows = [data[i:i + new_pitch] for i in range(0, len(data), new_pitch)]
                rows.reverse()
                data = b''.join(rows)

        return data

    def _ensure_bytes(self) -> None:
        if type(self._current_data) is not bytes:
            self._current_data = asbytes(self._current_data)

    @staticmethod
    def _get_gl_format_and_type(fmt):
        if fmt == 'R':
            return GL_RED, GL_UNSIGNED_BYTE
        elif fmt == 'RG':
            return GL_RG, GL_UNSIGNED_BYTE
        elif fmt == 'RGB':
            return GL_RGB, GL_UNSIGNED_BYTE
        elif fmt == 'BGR':
            return GL_BGR, GL_UNSIGNED_BYTE
        elif fmt == 'RGBA':
            return GL_RGBA, GL_UNSIGNED_BYTE
        elif fmt == 'BGRA':
            return GL_BGRA, GL_UNSIGNED_BYTE

        elif fmt == 'L':
            return GL_LUMINANCE, GL_UNSIGNED_BYTE
        elif fmt == 'A':
            return GL_ALPHA, GL_UNSIGNED_BYTE

        return None, None

    @staticmethod
    def _get_internalformat(fmt):
        if fmt == 'R':
            return GL_RED
        elif fmt == 'RG':
            return GL_RG
        elif fmt == 'RGB':
            return GL_RGB
        elif fmt == 'RGBA':
            return GL_RGBA
        elif fmt == 'D':
            return GL_DEPTH_COMPONENT
        elif fmt == 'DS':
            return GL_DEPTH_STENCIL

        elif fmt == 'L':
            return GL_LUMINANCE
        elif fmt == 'A':
            return GL_ALPHA

        return GL_RGBA


class VulkanImageDataRegion(VulkanImageData):
    def __init__(self, x, y, width, height, image_data):
        super().__init__(width, height,
                         image_data._current_format,
                         image_data._current_data,
                         image_data._current_pitch)
        self.x = x
        self.y = y

    def __getstate__(self):
        return {
            'width': self.width,
            'height': self.height,
            '_current_data': self.get_bytes(self._current_format, self._current_pitch),
            '_current_format': self._current_format,
            '_desired_format': self._desired_format,
            '_current_pitch': self._current_pitch,
            'pitch': self.pitch,
            'mipmap_images': self.mipmap_images,
            'x': self.x,
            'y': self.y,
        }

    def get_bytes(self, fmt=None, pitch=None):
        x1 = len(self._current_format) * self.x
        x2 = len(self._current_format) * (self.x + self.width)

        self._ensure_bytes()
        data = self._convert(self._current_format, abs(self._current_pitch))
        new_pitch = abs(self._current_pitch)
        rows = [data[i:i + new_pitch] for i in range(0, len(data), new_pitch)]
        rows = [row[x1:x2] for row in rows[self.y:self.y + self.height]]
        self._current_data = b''.join(rows)
        self._current_pitch = self.width * len(self._current_format)
        self._current_texture = None
        self.x = 0
        self.y = 0

        fmt = fmt or self._desired_format
        pitch = pitch or self._current_pitch
        return super().get_bytes(fmt, pitch)

    def set_bytes(self, fmt, pitch, data):
        self.x = 0
        self.y = 0
        super().set_bytes(fmt, pitch, data)

    def _apply_region_unpack(self):
        ...

    def _default_region_unpack(self):
        ...

    def get_region(self, x, y, width, height):
        x += self.x
        y += self.y
        return super().get_region(x, y, width, height)