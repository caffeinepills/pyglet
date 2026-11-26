"""Windows Imaging Component integration tests."""
from __future__ import annotations

from ctypes import c_ubyte
from io import BytesIO

import pytest

from tests.annotations import Platform, require_platform


pytestmark = require_platform(Platform.WINDOWS)


PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _assert_round_trip(encoded: bytes, expected: bytes) -> None:
    from pyglet.image.codecs.wic import WICDecoder  # noqa: PLC0415

    decoded = WICDecoder().decode("round_trip.png", BytesIO(encoded))
    assert decoded.width == 2
    assert decoded.height == 3
    assert decoded.get_bytes("RGBA", 8) == expected


def _assert_file_round_trip(filename, expected: bytes) -> None:
    from pyglet.image.codecs.wic import WICDecoder  # noqa: PLC0415

    decoded = WICDecoder().decode(str(filename), None)
    assert decoded.width == 2
    assert decoded.height == 3
    assert decoded.get_bytes("RGBA", 8) == expected


def test_wic_encodes_rgba_png_to_stream_and_filename(tmp_path):
    """WIC preserves RGBA pixels when producing PNG bytes or a PNG file."""
    from pyglet.image import ImageData  # noqa: PLC0415
    from pyglet.image.codecs.wic import WICEncoder  # noqa: PLC0415

    # Each row is deliberately distinct, which verifies that the negative
    # pitch supplied to WIC is correctly restored by its decoder.
    pixels = bytes((
        255, 0, 0, 255, 0, 255, 0, 128,
        0, 0, 255, 64, 255, 255, 0, 32,
        255, 0, 255, 16, 0, 255, 255, 0,
    ))
    source = ImageData(2, 3, "RGBA", pixels, 8)
    encoder = WICEncoder()

    stream = BytesIO()
    encoder.encode(source, "round_trip.png", stream)
    stream_bytes = stream.getvalue()
    assert stream_bytes.startswith(PNG_SIGNATURE)
    _assert_round_trip(stream_bytes, pixels)

    filename = tmp_path / "round_trip.png"
    encoder.encode(source, str(filename), None)
    file_bytes = filename.read_bytes()
    assert file_bytes.startswith(PNG_SIGNATURE)
    _assert_file_round_trip(filename, pixels)


@pytest.mark.parametrize(("fmt", "pitch", "as_array"), [
    ("RGB", 8, False),
    ("RGB", -8, False),
    ("RGBA", -8, False),
    ("BGRA", 8, True),
    ("BGRA", -8, True),
])
def test_wic_encodes_pixel_layouts(fmt: str, pitch: int, as_array: bool) -> None:
    """Preserve channels, row order, and padding for common encoder inputs."""
    from pyglet.image import ImageData  # noqa: PLC0415
    from pyglet.image.codecs.wic import WICDecoder, WICEncoder  # noqa: PLC0415

    rows = {
        "RGB": (bytes((255, 1, 2, 3, 254, 4, 17, 18)), bytes((5, 6, 253, 252, 7, 8, 19, 20))),
        "RGBA": (bytes((255, 1, 2, 9, 3, 254, 4, 10)), bytes((5, 6, 253, 0, 252, 7, 8, 12))),
        "BGRA": (bytes((2, 1, 255, 9, 4, 254, 3, 10)), bytes((253, 6, 5, 0, 8, 7, 252, 12))),
    }[fmt]
    pixels = b"".join(rows if pitch > 0 else reversed(rows))
    data = (c_ubyte * len(pixels)).from_buffer_copy(pixels) if as_array else pixels
    image = ImageData(2, 2, fmt, data, pitch)
    output = BytesIO()
    WICEncoder().encode(image, "layout.png", output)

    # Decode from the caller's current stream position.
    stream = BytesIO(b"prefix" + output.getvalue())
    stream.seek(len(b"prefix"))
    decoded = WICDecoder().decode("layout.png", stream)
    alpha = (255, 255, 255, 255) if fmt == "RGB" else (9, 10, 0, 12)
    assert decoded.get_bytes("RGBA", 8) == bytes((
        255, 1, 2, alpha[0], 3, 254, 4, alpha[1],
        5, 6, 253, alpha[2], 252, 7, 8, alpha[3],
    ))


def test_wic_jpeg_discards_alpha_without_changing_colors() -> None:
    """Native RGBA conversion matches encoding the same colors without alpha."""
    from pyglet.image import ImageData  # noqa: PLC0415
    from pyglet.image.codecs.wic import WICEncoder  # noqa: PLC0415

    pixels = bytes((255, 1, 2, 0, 3, 254, 4, 85, 5, 6, 253, 170, 252, 7, 8, 255))
    source = ImageData(2, 2, "RGBA", pixels)
    bgr = ImageData(2, 2, "BGR", source.get_bytes("BGR", -6), -6)
    encoder = WICEncoder()
    actual, expected = BytesIO(), BytesIO()
    encoder.encode(source, "test.jpg", actual)
    encoder.encode(bgr, "test.jpg", expected)
    assert actual.getvalue() == expected.getvalue()


def test_wic_encodes_rgba_as_bmp() -> None:
    from pyglet.image import ImageData  # noqa: PLC0415
    from pyglet.image.codecs.wic import WICDecoder, WICEncoder  # noqa: PLC0415

    pixels = bytes((255, 1, 2, 0, 3, 254, 4, 85, 5, 6, 253, 170, 252, 7, 8, 255))
    source = ImageData(2, 2, "RGBA", pixels)
    output = BytesIO()
    WICEncoder().encode(source, "test.bmp", output)
    output.seek(0)
    assert WICDecoder().decode("test.bmp", output).get_bytes("RGB", 6) == source.get_bytes("RGB", 6)


@pytest.mark.parametrize("payload", [b"", b"invalid image"])
def test_wic_rejects_invalid_stream(payload: bytes) -> None:
    from pyglet.image.codecs import ImageDecodeException  # noqa: PLC0415
    from pyglet.image.codecs.wic import WICDecoder  # noqa: PLC0415

    with pytest.raises(ImageDecodeException, match="invalid.png"):
        WICDecoder().decode("invalid.png", BytesIO(payload))
