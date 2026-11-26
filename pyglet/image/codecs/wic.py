from __future__ import annotations

import os
from ctypes import Array, POINTER, byref, cast, memmove, sizeof
from typing import Any, BinaryIO, Sequence

from pyglet.enums import PixelFormat
from pyglet.image import ImageData, get_default_decode_policy
from pyglet.image.codecs import ImageDecodeException, ImageDecoder, ImageEncodeException, ImageEncoder
from pyglet.libs.win32.wincodec import (
    CLSID_WICImagingFactory1,
    CLSID_WICImagingFactory2,
    GUID_ContainerFormatBmp,
    GUID_ContainerFormatJpeg,
    GUID_ContainerFormatPng,
    GUID_ContainerFormatTiff,
    GUID_ContainerFormatWmp,
    GUID_WICPixelFormat24bppBGR,
    GUID_WICPixelFormat24bppRGB,
    GUID_WICPixelFormat32bppBGRA,
    GUID_WICPixelFormat32bppRGBA,
    IID_IWICImagingFactory1,
    IID_IWICImagingFactory2,
    IPropertyBag2,
    IWICBitmap,
    IWICBitmapDecoder,
    IWICBitmapEncoder,
    IWICBitmapFlipRotator,
    IWICBitmapFrameDecode,
    IWICBitmapFrameEncode,
    IWICBitmapSource,
    IWICFormatConverter,
    IWICImagingFactory,
    IWICImagingFactory2,
    IWICMetadataQueryReader,
    IWICStream,
    WICBitmapCacheOnDemand,
    WICBitmapDitherTypeNone,
    WICBitmapEncoderNoCache,
    WICBitmapPaletteTypeCustom,
    WICBitmapTransformFlipVertical,
    WICDecodeMetadataCacheOnDemand,
)
from pyglet.libs.win32 import _kernel32 as kernel32
from pyglet.libs.win32 import _ole32 as ole32
from pyglet.libs.win32 import com
from pyglet.libs.win32.constants import (
    CLSCTX_INPROC_SERVER,
    GENERIC_READ,
    GENERIC_WRITE,
    GMEM_MOVEABLE,
    STREAM_SEEK_SET,
)
from pyglet.libs.win32.types import BOOL, BYTE, PROPVARIANT, STATSTG, UINT, ULONG, IStream


def _create_factory() -> IWICImagingFactory2 | IWICImagingFactory | None:
    """Get a WIC Factory.

    Factory 2 should be available since Windows 7 with a Platform Update, but be on the safe side.
    """
    try:
        factory = IWICImagingFactory2()
        ole32.CoCreateInstance(CLSID_WICImagingFactory2,
                               None,
                               CLSCTX_INPROC_SERVER,
                               IID_IWICImagingFactory2,
                               byref(factory))
        return factory
    except OSError:
        pass

    try:
        factory = IWICImagingFactory()
        ole32.CoCreateInstance(CLSID_WICImagingFactory1,
                               None,
                               CLSCTX_INPROC_SERVER,
                               IID_IWICImagingFactory1,
                               byref(factory))
        return factory
    except OSError:
        pass

    return None

_factory = _create_factory()
if not _factory:
    raise ImportError("Could not create WIC factory.")

def get_factory() -> IWICImagingFactory2 | IWICImagingFactory:
    """Retrieve the current WIC factory.

    WIC interfaces with many other libraries such as DirectWrite and Direct2D.
    """
    assert _factory is not None
    return _factory

def _get_bitmap_frame(bitmap_decoder: IWICBitmapDecoder, frame_index: int) -> IWICBitmapFrameDecode:
    bitmap = IWICBitmapFrameDecode()
    bitmap_decoder.GetFrame(frame_index, byref(bitmap))
    return bitmap

_guid_null = com.GUID()
_max_uint = (1 << 32) - 1
_rgb_formats = {
    "RGB": GUID_WICPixelFormat24bppRGB,
    "BGR": GUID_WICPixelFormat24bppBGR,
    "RGBA": GUID_WICPixelFormat32bppRGBA,
    "BGRA": GUID_WICPixelFormat32bppBGRA,
}


def get_bitmap(width: int, height: int, target_fmt: com.GUID=GUID_WICPixelFormat32bppBGRA) -> IWICBitmap:
    """Create a WIC Bitmap.

    Caller is responsible for releasing ``IWICBitmap``.
    """
    bitmap = IWICBitmap()
    _factory.CreateBitmap(width, height,
                      target_fmt,
                      WICBitmapCacheOnDemand,
                      byref(bitmap))
    return bitmap


def extract_image_data(bitmap: IWICBitmapSource, target_fmt: com.GUID | None = None) -> ImageData:
    """Extract image data from a WIC bitmap source in the requested format.

    .. note:: ``bitmap`` is released before this function returns.
    """
    if target_fmt is None:
        pixel_format = get_default_decode_policy().preferred_format or PixelFormat.BGRA8
        target_fmt = (
            GUID_WICPixelFormat32bppRGBA if pixel_format == PixelFormat.RGBA8 else GUID_WICPixelFormat32bppBGRA
        )
    if target_fmt == GUID_WICPixelFormat32bppRGBA:
        fmt = "RGBA"
    elif target_fmt == GUID_WICPixelFormat24bppBGR:
        fmt = "BGR"
    else:
        fmt = "BGRA"

    converter = IWICFormatConverter()
    flipper = IWICBitmapFlipRotator()
    source = bitmap
    try:
        native_width = UINT()
        native_height = UINT()
        bitmap.GetSize(byref(native_width), byref(native_height))
        width = native_width.value
        height = native_height.value

        pf = com.GUID()
        bitmap.GetPixelFormat(byref(pf))
        if pf != target_fmt:
            _factory.CreateFormatConverter(byref(converter))
            conversion_possible = BOOL()
            converter.CanConvert(pf, target_fmt, byref(conversion_possible))
            if not conversion_possible:
                target_fmt = GUID_WICPixelFormat24bppBGR
                fmt = 'BGR'
            converter.Initialize(bitmap, target_fmt, WICBitmapDitherTypeNone, None, 0, WICBitmapPaletteTypeCustom)
            source = converter

        # Return positive pitch so texture uploads can use the pixels directly.
        _factory.CreateBitmapFlipRotator(byref(flipper))
        flipper.Initialize(source, WICBitmapTransformFlipVertical)
        stride = len(fmt) * width
        buffer_size = stride * height
        buffer = (BYTE * buffer_size)()
        flipper.CopyPixels(None, stride, buffer_size, byref(buffer))
    finally:
        for obj in (flipper, converter, bitmap):
            if obj:
                obj.Release()

    return ImageData(width, height, fmt, buffer)


class WICDecoder(ImageDecoder):
    """Windows Imaging Component implementation for image decoding.

    This decoder is a replacement for GDI/GDI+ starting with Windows 7.
    """
    def __init__(self) -> None:  # noqa: D107
        super(ImageDecoder, self).__init__()
        self._factory = get_factory()

    def get_file_extensions(self) -> list[str]:
        return ['.bmp', '.jpg', '.jpeg', '.png', '.tif', '.tiff', '.ico', '.jxr', '.hdp', '.wdp']

    @staticmethod
    def _create_decode_stream(data: bytes) -> IStream:
        if len(data) <= _max_uint:
            stream = IWICStream()
            _factory.CreateStream(byref(stream))
            try:
                stream.InitializeFromMemory(cast(data, POINTER(BYTE)), len(data))
            except OSError:
                stream.Release()
                raise
            return stream

        # IWICStream takes a DWORD length. Keep the HGLOBAL path for larger inputs.
        stream = IStream()
        # Create a HGLOBAL with image data
        hglob = kernel32.GlobalAlloc(GMEM_MOVEABLE, len(data))
        if not hglob:
            raise OSError("Could not allocate a WIC stream buffer")
        try:
            ptr = kernel32.GlobalLock(hglob)
            if not ptr:
                raise OSError("Could not lock the WIC stream buffer")
            try:
                memmove(ptr, data, len(data))
            finally:
                kernel32.GlobalUnlock(hglob)
            ole32.CreateStreamOnHGlobal(hglob, True, byref(stream))
        except BaseException:
            kernel32.GlobalFree(hglob)
            raise
        return stream

    def _load_bitmap_decoder_from_filename(self, filename: str) -> IWICBitmapDecoder:
        """Create a decoder directly from a path, without copying it through Python memory."""
        decoder = IWICBitmapDecoder()
        self._factory.CreateDecoderFromFilename(
            filename, _guid_null, GENERIC_READ, WICDecodeMetadataCacheOnDemand, byref(decoder),
        )
        return decoder

    def decode(self, filename: str, file: BinaryIO | None) -> ImageData:
        if file is None:
            bitmap_decoder = self._load_bitmap_decoder_from_filename(filename)
            try:
                return extract_image_data(_get_bitmap_frame(bitmap_decoder, 0))
            finally:
                bitmap_decoder.Release()

        # WIC borrows this buffer until the decoder and stream are released.
        data = file.read()
        stream = None
        bitmap_decoder = IWICBitmapDecoder()
        try:
            try:
                stream = self._create_decode_stream(data)
                self._factory.CreateDecoderFromStream(
                    stream, None, WICDecodeMetadataCacheOnDemand, byref(bitmap_decoder),
                )
            except OSError as exc:
                raise ImageDecodeException(f"WIC cannot load {filename or file!r}") from exc
            return extract_image_data(_get_bitmap_frame(bitmap_decoder, 0))
        finally:
            if bitmap_decoder:
                bitmap_decoder.Release()
            if stream:
                stream.Release()

    @staticmethod
    def get_property_value(reader: IWICMetadataQueryReader, metadata_name: str) -> int:
        """Uses a metadata name and reader to return a single value.

        Can be used to get metadata from images.

        Handles cleanup of PROPVARIANT.

        Returns:
            0 on failure.
        """
        try:
            prop = PROPVARIANT()
            reader.GetMetadataByName(metadata_name, byref(prop))
            value = prop.llVal
            ole32.PropVariantClear(byref(prop))
        except OSError:
            value = 0

        return value


def get_decoders() -> Sequence[ImageDecoder]:  # noqa: D103
    return [WICDecoder()]


extension_to_container = {
    '.bmp': GUID_ContainerFormatBmp,
    '.jpg': GUID_ContainerFormatJpeg,
    '.jpeg': GUID_ContainerFormatJpeg,
    '.tif': GUID_ContainerFormatTiff,
    '.tiff': GUID_ContainerFormatTiff,
    '.wmp': GUID_ContainerFormatWmp,
    '.jxr': GUID_ContainerFormatWmp,
    '.wdp': GUID_ContainerFormatWmp,
    '.png': GUID_ContainerFormatPng,
}


def _check_pixel_buffer(data: bytes | Array[Any], size: int) -> None:
    available = len(data) if isinstance(data, bytes) else sizeof(data)
    if not 0 <= size <= min(_max_uint, available):
        raise ImageEncodeException("Invalid WIC pixel buffer size")


def _prepare_encoder_pixels(image: ImageData, fmt: str, pitch: int) -> bytes | Array[Any]:
    source_fmt = image._current_format  # noqa: SLF001
    source_guid = _rgb_formats.get(source_fmt)
    top_down = image._current_pitch < 0  # noqa: SLF001
    if image.data_type != "B" or source_guid is None or (source_fmt == fmt and top_down):
        return image.get_bytes(fmt, -pitch)

    source_pitch = image.width * len(source_fmt)
    source_size = source_pitch * image.height
    data = image.get_bytes(source_fmt, -source_pitch if top_down else source_pitch)
    _check_pixel_buffer(data, source_size)
    size = pitch * image.height
    if not 0 <= size <= _max_uint:
        raise ImageEncodeException("Invalid WIC pixel buffer size")

    bitmap = IWICBitmap()
    converter = IWICFormatConverter()
    flipper = IWICBitmapFlipRotator()
    source: IWICBitmapSource = bitmap
    try:
        # Borrow ctypes arrays via byref to avoid a reference cycle in cast.
        source_pointer = cast(data if isinstance(data, bytes) else byref(data), POINTER(BYTE))
        _factory.CreateBitmapFromMemory(
            image.width, image.height, source_guid, source_pitch, source_size, source_pointer, byref(bitmap),
        )
        if source_fmt != fmt:
            _factory.CreateFormatConverter(byref(converter))
            converter.Initialize(source, _rgb_formats[fmt], WICBitmapDitherTypeNone, None, 0, WICBitmapPaletteTypeCustom)
            source = converter
        if not top_down:
            _factory.CreateBitmapFlipRotator(byref(flipper))
            flipper.Initialize(source, WICBitmapTransformFlipVertical)
            source = flipper
        pixels = (BYTE * size)()
        source.CopyPixels(None, pitch, size, byref(pixels))
        return pixels
    finally:
        for obj in (flipper, converter, bitmap):
            if obj:
                obj.Release()


class WICEncoder(ImageEncoder):  # noqa: D101
    def get_file_extensions(self) -> Sequence[str]:
        return list(extension_to_container)

    def encode(self, image: ImageData, filename: str, file: BinaryIO | None) -> None:
        image = image.get_image_data()

        wicstream = IWICStream()
        encoder = IWICBitmapEncoder()
        frame = IWICBitmapFrameEncode()
        property_bag = IPropertyBag2()

        ext = (filename and os.path.splitext(filename)[1]) or '.png'

        # Choose container based on extension. Default to PNG.
        container = extension_to_container.get(ext, GUID_ContainerFormatPng)

        istream = None
        try:
            _factory.CreateStream(byref(wicstream))
            # JPEG has no alpha channel. Other codecs can use BGRA for alpha.
            fmt = 'BGR' if container == GUID_ContainerFormatJpeg or len(image.format) == 3 else 'BGRA'
            default_format = _rgb_formats[fmt]
            pitch = image.width * len(fmt)
            size = pitch * image.height

            if file:
                istream = IStream()
                ole32.CreateStreamOnHGlobal(None, True, byref(istream))
                wicstream.InitializeFromIStream(istream)
            else:
                wicstream.InitializeFromFilename(filename, GENERIC_WRITE)

            _factory.CreateEncoder(container, None, byref(encoder))
            encoder.Initialize(wicstream, WICBitmapEncoderNoCache)
            encoder.CreateNewFrame(byref(frame), byref(property_bag))
            frame.Initialize(property_bag)
            frame.SetSize(image.width, image.height)

            # SetPixelFormat is an in/out parameter. Never let WIC overwrite
            # the module-level GUID constants shared by the decoder.
            pixel_format = default_format.copy()
            frame.SetPixelFormat(byref(pixel_format))

            image_data = _prepare_encoder_pixels(image, fmt, pitch)
            _check_pixel_buffer(image_data, size)
            data = cast(image_data if isinstance(image_data, bytes) else byref(image_data), POINTER(BYTE))
            if pixel_format == default_format:
                frame.WritePixels(image.height, pitch, size, data)
            else:
                # Let the encoder convert the pixels if it selected a different format.
                bitmap = IWICBitmap()
                try:
                    _factory.CreateBitmapFromMemory(
                        image.width, image.height, default_format, pitch, size, data, byref(bitmap),
                    )
                    frame.WriteSource(bitmap, None)
                finally:
                    if bitmap:
                        bitmap.Release()
            frame.Commit()
            encoder.Commit()

            if file and istream:
                sts = STATSTG()
                istream.Stat(byref(sts), 0)
                stream_size = sts.cbSize
                istream.Seek(0, STREAM_SEEK_SET, None)

                buf = (BYTE * stream_size)()
                written = ULONG()
                istream.Read(byref(buf), stream_size, byref(written))

                if written.value == stream_size:
                    file.write(buf)
                else:
                    msg = f"WIC could not read all encoded data for {file!r}"
                    raise ImageEncodeException(msg)
        finally:
            if istream:
                istream.Release()
            if frame:
                frame.Release()
            if property_bag:
                property_bag.Release()
            if encoder:
                encoder.Release()
            if wicstream:
                wicstream.Release()


def get_encoders() -> Sequence[ImageEncoder]:  # noqa: D103
    return [WICEncoder()]
