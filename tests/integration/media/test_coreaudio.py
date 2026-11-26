"""macOS Core Audio decoder integration tests."""

from __future__ import annotations

from ctypes import string_at
from gc import collect
from io import BytesIO, UnsupportedOperation
from struct import pack
from typing import TYPE_CHECKING
from wave import open as open_wave
from weakref import ref

import pytest

from tests.annotations import Platform, require_platform

if TYPE_CHECKING:
    from ctypes import Array, c_char
    from pathlib import Path

    from pyglet.media.codecs import AudioData

pytestmark = require_platform(Platform.OSX)

PCM = pack("<34h", *range(34))


@pytest.fixture
def wav_file(tmp_path: Path) -> Path:
    path = tmp_path / "stereo.wav"
    with open_wave(str(path), "wb") as writer:
        writer.setnchannels(2)
        writer.setsampwidth(2)
        writer.setframerate(48000)
        writer.writeframes(PCM)
    return path


class TrackedStream(BytesIO):
    readinto_calls: int
    supports_readinto: bool

    def __init__(self, data: bytes, supports_readinto: bool) -> None:
        super().__init__(data)
        self.readinto_calls = 0
        self.supports_readinto = supports_readinto

    def readinto(self, buffer: Array[c_char]) -> int:
        self.readinto_calls += 1
        if not self.supports_readinto:
            raise UnsupportedOperation
        return super().readinto(buffer)


def _assert_packet(packet: AudioData | None, expected: bytes) -> None:
    assert packet is not None
    assert packet.length == len(expected)
    assert bytes(packet.data) == expected
    assert string_at(packet.pointer, packet.length) == expected


@pytest.mark.parametrize("input_kind", ["path", "readinto", "read"])
def test_coreaudio_variable_reads(wav_file: Path, input_kind: str) -> None:
    from pyglet.media import load_audio  # noqa: PLC0415
    from pyglet.media.codecs.coreaudio import CoreAudioDecoder  # noqa: PLC0415

    stream = None if input_kind == "path" else TrackedStream(wav_file.read_bytes(), input_kind == "readinto")
    source = load_audio(wav_file, file=stream, decoder=CoreAudioDecoder())
    assert source.get_audio_data(3) is None  # Less than a complete stereo frame.

    first_packet = source.get_audio_data(16)
    _assert_packet(source.get_audio_data(4), PCM[16:20])
    _assert_packet(source.get_audio_data(28), PCM[20:48])
    _assert_packet(source.get_audio_data(29), PCM[48:])
    assert source.get_audio_data(16) is None

    source.seek_to_frame(2)
    _assert_packet(source.get_audio_data(8), PCM[8:16])
    del source

    _assert_packet(first_packet, PCM[:16])
    if stream is not None:
        assert stream.readinto_calls
        assert stream.closed


def test_coreaudio_keeps_stream_alive_after_caller_drops_reference() -> None:
    from pyglet.media import load_audio  # noqa: PLC0415
    from pyglet.media.codecs.coreaudio import CoreAudioDecoder  # noqa: PLC0415

    pcm = PCM * 32768
    stream = TrackedStream(b"", supports_readinto=True)
    with open_wave(stream, "wb") as writer:
        writer.setparams((2, 2, 48000, 0, "NONE", ""))
        writer.writeframes(pcm)
    stream.seek(0)

    source = load_audio("stereo.wav", file=stream, decoder=CoreAudioDecoder())
    stream_ref = ref(stream)
    reads_before_gc = stream.readinto_calls
    del stream
    collect()
    assert stream_ref() is not None

    # Seek beyond the start of the file so the read exercises the native callback.
    offset = len(pcm) - 4096
    source.seek_to_frame(offset // 4)
    _assert_packet(source.get_audio_data(4096), pcm[offset:])

    retained_stream = stream_ref()
    assert retained_stream is not None
    assert retained_stream.readinto_calls > reads_before_gc


@pytest.mark.parametrize("from_memory", [False, True])
def test_coreaudio_static_load_uses_only_decoded_pcm(wav_file: Path, from_memory: bool) -> None:
    from pyglet.media import load_audio  # noqa: PLC0415
    from pyglet.media.codecs.coreaudio import CoreAudioDecoder  # noqa: PLC0415

    stream = BytesIO(wav_file.read_bytes()) if from_memory else None
    source = load_audio(wav_file, file=stream, streaming=False, decoder=CoreAudioDecoder())
    queued = source.get_queue_source()
    assert queued is not None
    _assert_packet(queued.get_audio_data(len(PCM) + 4), PCM)
