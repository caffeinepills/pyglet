"""Decode macOS audio files through Core Audio."""

from __future__ import annotations

from ctypes import addressof, byref, c_char, c_long, c_uint32, memmove, sizeof
from io import UnsupportedOperation
from typing import TYPE_CHECKING

from pyglet.libs.darwin import cf, CFSTR
from pyglet.libs.darwin.coreaudio import kCFURLPOSIXPathStyle, AudioStreamBasicDescription, ca, ExtAudioFileRef, \
    kExtAudioFileProperty_FileDataFormat, kAudioFormatLinearPCM, kAudioFormatFlagIsSignedInteger, \
    kAudioFormatFlagIsPacked, kExtAudioFileProperty_ClientDataFormat, AudioFile_ReadProc, \
    AudioFile_GetSizeProc, AudioBufferList, kExtAudioFileProperty_FileLengthFrames, AudioFileID, err_check, \
    kAudioFileUnspecifiedError
from pyglet.media import StreamingSource, StaticSource
from pyglet.media.codecs import AudioFormat, MediaDecoder, AudioData


if TYPE_CHECKING:
    from ctypes import _Pointer
    from typing import BinaryIO


class MemoryFileObject:
    """Keep the native callbacks for reading a seekable Python stream alive."""

    getsize_func: AudioFile_GetSizeProc
    read_func: AudioFile_ReadProc

    def __init__(self, file: BinaryIO) -> None:
        """Prepare callbacks for the supplied stream."""
        if not getattr(file, 'seek', None) or not getattr(file, 'tell', None):
            raise Exception("File object does not support seeking.")

        file.seek(0, 2)
        file_size = file.tell()
        file.seek(0)
        readinto = getattr(file, 'readinto', None)

        def read_data_cb(_ref: int | None, offset: int, requested_length: int, buffer: int,
                         actual_count: _Pointer[c_uint32]) -> int:
            actual_count[0] = 0
            if not requested_length:
                return 0

            try:
                file.seek(offset)
                if callable(readinto):
                    try:
                        # Core Audio owns the destination for the duration of this callback.
                        destination = (c_char * requested_length).from_address(buffer)
                        count = readinto(destination) or 0
                    except (UnsupportedOperation, NotImplementedError):
                        file.seek(offset)
                    else:
                        if not 0 <= count <= requested_length:
                            return kAudioFileUnspecifiedError
                        actual_count[0] = count
                        return 0

                data = file.read(requested_length)
                count = len(data)
                if count > requested_length:
                    return kAudioFileUnspecifiedError
                memmove(buffer, data, count)
                actual_count[0] = count
            except Exception:  # noqa: BLE001
                # Python exceptions must not escape through the native callback.
                return kAudioFileUnspecifiedError

            return 0

        def getsize_cb(_ref: int | None) -> int:
            return file_size

        self.getsize_func = AudioFile_GetSizeProc(getsize_cb)
        self.read_func = AudioFile_ReadProc(read_data_cb)


class CoreAudioSource(StreamingSource):
    """Stream decoded PCM from an extended audio file."""

    _bl: AudioBufferList
    _file: BinaryIO | None
    _file_obj: MemoryFileObject | None

    def __init__(self, filename: str, file: BinaryIO | None = None) -> None:
        """Open an audio file and configure its PCM output."""
        self._file = file
        self._deleted = False
        self._file_obj = None
        self._audfile = None
        self._audref = None

        audref = ExtAudioFileRef()
        if file is None:
            fn_str = CFSTR(filename)
            url_ref = cf.CFURLCreateWithFileSystemPath(None, fn_str, kCFURLPOSIXPathStyle, False)
            try:
                err_check(ca.ExtAudioFileOpenURL(url_ref, byref(audref)))
            finally:
                cf.CFRelease(url_ref)
                cf.CFRelease(fn_str)
        else:
            self._file_obj = MemoryFileObject(file)

            self._audfile = AudioFileID()

            err_check(ca.AudioFileOpenWithCallbacks(
                None, self._file_obj.read_func, None, self._file_obj.getsize_func, None,
                0,
                byref(self._audfile)),
            )

            err_check(ca.ExtAudioFileWrapAudioFileID(self._audfile, False, byref(audref)))

        self._audref = audref

        format_info = AudioStreamBasicDescription()
        size = c_uint32(sizeof(format_info))
        err_check(ca.ExtAudioFileGetProperty(self._audref,
                                             kExtAudioFileProperty_FileDataFormat,
                                             byref(size),
                                             byref(format_info)))

        self.convert_desc = self.convert_format(format_info)

        err_check(ca.ExtAudioFileSetProperty(
            self._audref,
            kExtAudioFileProperty_ClientDataFormat,
            sizeof(self.convert_desc),
            byref(self.convert_desc),
        ))

        length = c_long()
        size = c_uint32(sizeof(format_info))
        # File length.
        err_check(ca.ExtAudioFileGetProperty(
            self._audref,
            kExtAudioFileProperty_FileLengthFrames,
            byref(size),
            byref(length),
        ))

        self.audio_format = AudioFormat(channels=self.convert_desc.mChannelsPerFrame,
                                        sample_size=self.convert_desc.mBitsPerChannel,
                                        sample_rate=int(self.convert_desc.mSampleRate))

        self._num_frames = length.value
        self._bytes_per_frame = self.convert_desc.mBytesPerFrame
        self._duration = self._num_frames / self.convert_desc.mSampleRate
        self._duration_per_frame = self._duration / self._num_frames
        self._bl = AudioBufferList()
        self._bl.mNumberBuffers = 1
        self._bl.mBuffers[0].mNumberChannels = self.convert_desc.mChannelsPerFrame

    @staticmethod
    def convert_format(original_desc, bitdepth=16):
        adesc = AudioStreamBasicDescription()
        adesc.mSampleRate = original_desc.mSampleRate
        adesc.mFormatID = kAudioFormatLinearPCM
        adesc.mFormatFlags = kAudioFormatFlagIsSignedInteger | kAudioFormatFlagIsPacked
        adesc.mChannelsPerFrame = original_desc.mChannelsPerFrame
        adesc.mBitsPerChannel = bitdepth
        adesc.mBytesPerPacket = original_desc.mChannelsPerFrame * adesc.mBitsPerChannel // 8
        adesc.mFramesPerPacket = 1
        adesc.mBytesPerFrame = adesc.mBytesPerPacket
        return adesc

    def __del__(self) -> None:
        try:
            try:
                if self._audref:
                    err_check(ca.ExtAudioFileDispose(self._audref))
                    self._audref = None
            finally:
                if self._audfile:
                    err_check(ca.AudioFileClose(self._audfile))
                    self._audfile = None
        finally:
            if self._file is not None:
                self._file.close()
                self._file = None

    def get_audio_data(self, num_bytes: int) -> AudioData | None:
        # AudioBuffer.mDataByteSize is a UInt32, and the request must fit whole frames.
        frame_count = min(num_bytes, 0xFFFFFFFF) // self._bytes_per_frame
        if frame_count <= 0:
            return None

        capacity = frame_count * self._bytes_per_frame
        buffer = (c_char * capacity)()
        output = self._bl.mBuffers[0]
        output.mDataByteSize = capacity
        output.mData = addressof(buffer)

        frames = c_uint32(frame_count)
        err_check(ca.ExtAudioFileRead(self._audref, byref(frames), byref(self._bl)))
        size = frames.value * self._bytes_per_frame
        if not size:
            return None

        # A short read exposes only initialized PCM while retaining the allocation.
        return AudioData(buffer if size == capacity else (c_char * size).from_buffer(buffer), size)

    def seek(self, timestamp: float) -> None:
        timestamp = max(0.0, min(timestamp, self._duration))
        position = int(timestamp / self._duration_per_frame)
        self.seek_to_frame(position)

    def seek_to_frame(self, frame: int) -> None:
        """Seek to an exact decoded PCM frame."""
        ca.ExtAudioFileSeek(self._audref, max(0, min(frame, self._num_frames)))


#########################################
#   Decoder class:
#########################################

class CoreAudioDecoder(MediaDecoder):

    def get_file_extensions(self) -> tuple[str, ...]:
        return '.aac', '.ac3', '.aif', '.aiff', '.aifc', '.caf', '.mp3', '.mp4', '.m4a', '.snd', '.au', '.sd2', '.wav'

    def decode(self, filename, file, streaming=True, **kwargs):
        if streaming:
            return CoreAudioSource(filename, file)
        return StaticSource(CoreAudioSource(filename, file))


def get_decoders():
    return [CoreAudioDecoder()]


def get_encoders():
    return []
