import os
from io import BytesIO
from pathlib import Path
from unittest import mock

import pyglet
import pytest

from tests.annotations import Platform, require_platform


test_data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))


def get_test_data_file(path, file_name):
    """Get a file from the test data directory in an OS independent way."""
    return os.path.join(test_data_path, path, file_name)


@pytest.mark.parametrize("path_type", [str, Path])
@pytest.mark.parametrize("streaming", [False, True])
def test_load_audio_from_disk_path(path_type, streaming: bool) -> None:
    file_path = path_type(get_test_data_file("media", "alert.wav"))
    source = pyglet.media.load_audio(file_path, streaming=streaming)

    assert source.audio_format is not None
    assert source.duration > 0


def test_load_audio_from_open_file_object():
    file_path = get_test_data_file("media", "alert.wav")
    with open(file_path, "rb") as file_object:
        source = pyglet.media.load_audio(file_path, file=file_object, streaming=False)

    assert source.audio_format is not None
    assert source.duration > 0


@require_platform(Platform.WINDOWS)
@pytest.mark.parametrize("use_wic_stream", [True, False])
def test_wmf_reads_and_seeks_after_input_file_closes(use_wic_stream: bool) -> None:
    from pyglet.image.codecs import wic  # noqa: PLC0415
    from pyglet.media.codecs import wmf  # noqa: PLC0415

    path = Path(get_test_data_file("media", "alert.wav"))
    decoder = wmf.WMFDecoder()
    reference = pyglet.media.load_audio(path, decoder=decoder)
    expected = reference.get_audio_data(4096)
    assert expected is not None

    if use_wic_stream:
        guard = mock.patch.object(wmf.kernel32, "GlobalAlloc", side_effect=AssertionError("Unexpected input copy"))
    else:
        guard = mock.patch.object(wic, "get_factory", side_effect=ImportError("WIC unavailable"))

    prefix = b"skip this prefix"
    with guard, BytesIO(prefix + path.read_bytes()) as file:
        file.seek(len(prefix))
        source = pyglet.media.load_audio(path, file=file, decoder=decoder)

    sample = source.get_audio_data(4096)
    assert sample is not None
    assert sample.data == expected.data
    source.seek(0.0)
    sample = source.get_audio_data(4096)
    assert sample is not None
    assert sample.data == expected.data


def test_wav_uses_wave_decoder():
    from pyglet.media import codecs as media_codecs
    from pyglet.media.codecs.wave import WaveDecoder

    file_path = get_test_data_file("media", "alert.wav")
    decoder = next(
        decoder
        for decoder in media_codecs.get_decoders(file_path, media_capabilities="audio")
        if isinstance(decoder, WaveDecoder)
    )

    with mock.patch.object(decoder, "decode", wraps=decoder.decode) as decode_mock:
        source = pyglet.media.load_audio(file_path, streaming=False)

    decode_mock.assert_called_once()
    assert source.audio_format is not None
    assert source.duration > 0
