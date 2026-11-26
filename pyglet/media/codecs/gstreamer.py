"""Multi-format decoder using Gstreamer.
"""
from __future__ import annotations

import queue
import atexit
import weakref

from threading import Event, Thread
from typing import BinaryIO, cast

from pyglet.util import DecodeException
from .base import StreamingSource, AudioData, AudioFormat, StaticSource
from . import MediaEncoder, MediaDecoder

try:
    import gi
    gi.require_version('Gst', '1.0')
    from gi.repository import Gst, GLib
except (ValueError, ImportError) as e:
    raise ImportError(e)


class GStreamerDecodeException(DecodeException):
    pass


class _GLibMainLoopThread(Thread):
    """A background Thread for a GLib MainLoop"""
    mainloop: GLib.MainLoop

    def __init__(self) -> None:
        super().__init__(daemon=True)
        self.mainloop = GLib.MainLoop.new(None, False)
        self.start()

    def run(self) -> None:
        self.mainloop.run()


class _MessageHandler:
    """Message Handler class for GStreamer Sources.
    
    This separate class holds a weak reference to the
    Source, preventing garbage collection issues. 
    
    """
    source: GStreamerSource

    def __init__(self, source: GStreamerSource) -> None:
        self.source = weakref.proxy(source)

    def message(self, bus: Gst.Bus, message: Gst.Message) -> None:
        """The main message callback"""
        if message.type == Gst.MessageType.EOS:

            self.source._queue_packet(self.source.sentinel)
            if not self.source.caps:
                raise GStreamerDecodeException("Appears to be an unsupported file")

        elif message.type == Gst.MessageType.ERROR:
            raise GStreamerDecodeException(message.parse_error())

    def notify_caps(self, pad: Gst.Pad, *args: object) -> None:
        """notify::caps callback"""
        self.source.caps = True
        info = pad.get_current_caps().get_structure(0)

        self.source._duration = pad.get_peer().query_duration(Gst.Format.TIME).duration / Gst.SECOND
        channels = info.get_int('channels')[1]
        sample_rate = info.get_int('rate')[1]
        sample_size = int("".join(filter(str.isdigit, info.get_string('format'))))

        self.source.audio_format = AudioFormat(channels=channels, sample_size=sample_size, sample_rate=sample_rate)

        # Allow GStreamerSource.__init__ to complete:
        self.source.is_ready.set()

    def pad_added(self, element: Gst.Element, pad: Gst.Pad) -> None:
        """pad-added callback"""
        name = pad.query_caps(None).to_string()
        if name.startswith('audio/x-raw'):
            nextpad = self.source.converter.get_static_pad('sink')
            if not nextpad.is_linked():
                self.source.pads = True
                pad.link(nextpad)

    def no_more_pads(self, element: Gst.Element) -> None:
        """Finished Adding pads"""
        if not self.source.pads:
            raise GStreamerDecodeException('No Streams Found')

    def new_sample(self, sink: Gst.Element) -> Gst.FlowReturn:
        """new-sample callback"""
        sample = sink.emit('pull-sample')
        if sample is None:
            return Gst.FlowReturn.FLUSHING
        buffer = sample.get_buffer()
        # Extract a copy of the memory in the buffer:
        mem = buffer.extract_dup(0, buffer.get_size())
        return Gst.FlowReturn.OK if self.source._queue_packet(mem) else Gst.FlowReturn.FLUSHING

    def need_data(self, appsrc: Gst.Element, length: int) -> None:
        """Feed the encoded bytes requested by appsrc in random-access mode."""
        data = self.source._input_data
        start = self.source._input_offset
        if start >= len(data):
            appsrc.emit("end-of-stream")
            return

        end = min(start + length, len(data))
        buffer = Gst.Buffer.new_wrapped(data[start:end])
        buffer.offset = start
        buffer.offset_end = end
        self.source._input_offset = end
        appsrc.emit("push-buffer", buffer)

    def seek_data(self, appsrc: Gst.Element, offset: int) -> bool:
        """Reposition appsrc within the owned encoded bytes."""
        if not 0 <= offset <= len(self.source._input_data):
            return False
        self.source._input_offset = offset
        return True

    @staticmethod
    def unknown_type(uridecodebin: Gst.Element, decodebin: Gst.Element, caps: Gst.Caps) -> None:
        """unknown-type callback for unreadable files"""
        streaminfo = caps.to_string()
        if not streaminfo.startswith('audio/'):
            return
        raise GStreamerDecodeException(streaminfo)


class GStreamerSource(StreamingSource):
    _pipeline: Gst.Pipeline
    _input: Gst.Element
    _input_data: bytes
    _input_offset: int
    _bus_handler: int | None
    decoder: Gst.Element
    converter: Gst.Element
    appsink: Gst.Element
    caps_handler: int | None
    pads: bool
    caps: bool
    queue: queue.Queue[bytes | object]
    _finished: Event
    _stopping: Event
    is_ready: Event

    source_instances: weakref.WeakSet[GStreamerSource] = weakref.WeakSet()
    sentinel = object()

    def __init__(self, filename: str, file: BinaryIO | None = None) -> None:
        self._pipeline = Gst.Pipeline()
        self._stopping = Event()
        self._finished = Event()
        self.is_ready = Event()
        self.queue = queue.Queue(5)
        self.caps_handler = None
        self._bus_handler = None
        self._input_data = b""
        self._input_offset = 0

        msg_handler = _MessageHandler(self)

        # Create the major parts of the pipeline:
        self._input = Gst.ElementFactory.make("filesrc" if file is None else "appsrc", None)
        self.decoder = Gst.ElementFactory.make("decodebin", None)
        self.converter = Gst.ElementFactory.make("audioconvert", None)
        self.appsink = Gst.ElementFactory.make("appsink", None)
        if not all((self._input, self.decoder, self.converter, self.appsink)):
            raise GStreamerDecodeException("Could not initialize GStreamer.")

        if file is None:
            self._input.set_property("location", filename)
        else:
            # Own the input bytes so the caller can close the file after loading.
            file.seek(0)
            self._input_data = file.read()
            self._input.set_property("size", len(self._input_data))
            self._input.set_property("format", Gst.Format.BYTES)
            Gst.util_set_object_arg(self._input, "stream-type", "random-access")
            self._input.connect("need-data", msg_handler.need_data)
            self._input.connect("seek-data", msg_handler.seek_data)

        # Set callbacks for EOS and error messages:
        bus = self._pipeline.get_bus()
        self._bus_handler = bus.connect("message", msg_handler.message)
        bus.add_signal_watch()

        # Set decoder callback handlers:
        self.decoder.connect("pad-added", msg_handler.pad_added)
        self.decoder.connect("no-more-pads", msg_handler.no_more_pads)
        self.decoder.connect("unknown-type", msg_handler.unknown_type)

        # Set the sink's capabilities and behavior:
        self.appsink.set_property('caps', Gst.Caps.from_string('audio/x-raw,format=S16LE,layout=interleaved'))
        self.appsink.set_property('drop', False)
        self.appsink.set_property('sync', False)
        self.appsink.set_property('max-buffers', 0)     # unlimited
        self.appsink.set_property('emit-signals', True)
        # The callback to receive decoded data:
        self.appsink.connect("new-sample", msg_handler.new_sample)

        # Add all components to the pipeline:
        self._pipeline.add(self._input)
        self._pipeline.add(self.decoder)
        self._pipeline.add(self.converter)
        self._pipeline.add(self.appsink)
        # Link together necessary components:
        self._input.link(self.decoder)
        self.decoder.link(self.converter)
        self.converter.link(self.appsink)

        # Callback to notify once the sink is ready:
        self.caps_handler = self.appsink.get_static_pad("sink").connect("notify::caps", msg_handler.notify_caps)

        # Set by callbacks:
        self.pads = False
        self.caps = False
        state = self._pipeline.set_state(Gst.State.PLAYING)
        if state == Gst.StateChangeReturn.FAILURE or not self.is_ready.wait(timeout=1):
            self.delete()
            raise GStreamerDecodeException('Initialization Error')

        GStreamerSource.source_instances.add(self)

    def __del__(self) -> None:
        if hasattr(self, '_pipeline'):
            self.delete()

    def delete(self) -> None:
        if self._stopping.is_set():
            return
        self._stopping.set()
        if self.caps_handler is not None:
            self.appsink.get_static_pad("sink").disconnect(self.caps_handler)
        self._pipeline.set_state(Gst.State.NULL)
        if self._bus_handler is not None:
            bus = self._pipeline.get_bus()
            bus.disconnect(self._bus_handler)
            bus.remove_signal_watch()
        self._input_data = b""
        try:
            while True:
                self.queue.get_nowait()
        except queue.Empty:
            pass

    def _queue_packet(self, packet: bytes | object) -> bool:
        # A full queue must not keep a streaming callback alive during shutdown.
        while not self._stopping.is_set():
            try:
                self.queue.put(packet, timeout=0.1)
            except queue.Full:
                continue
            return True
        return False

    def get_audio_data(self, num_bytes: int) -> AudioData | None:
        if self._finished.is_set():
            return None

        data = bytes()
        while len(data) < num_bytes:
            packet = self.queue.get()
            if packet == self.sentinel:
                self._finished.set()
                break
            data += cast(bytes, packet)

        if not data:
            return None

        return AudioData(data, len(data))

    def seek(self, timestamp: float) -> None:
        self._seek(timestamp)

    def seek_to_frame(self, frame: int) -> None:
        """Seek to a decoded PCM frame."""
        self._seek(max(0, frame) / self.audio_format.sample_rate)

    def _seek(self, timestamp: float) -> None:
        # First clear any data in the queue:
        while not self.queue.empty():
            self.queue.get_nowait()

        self._pipeline.seek_simple(Gst.Format.TIME,
                                   Gst.SeekFlags.FLUSH | Gst.SeekFlags.ACCURATE,
                                   timestamp * Gst.SECOND)
        self._finished.clear()


def _cleanup() -> None:
    # At exist, ensure any remaining Source instances are cleaned up.
    # If this is not done, GStreamer may hang due to dangling callbacks.
    for src in GStreamerSource.source_instances:
        src.delete()


atexit.register(_cleanup)


#########################################
#   Decoder class:
#########################################

class GStreamerDecoder(MediaDecoder):
    _glib_loop: _GLibMainLoopThread

    def __init__(self) -> None:
        Gst.init(None)
        self._glib_loop = _GLibMainLoopThread()

    def get_file_extensions(self) -> tuple[str, ...]:
        return '.mp3', '.flac', '.ogg', '.m4a'

    def decode(self, filename: str, file: BinaryIO | None, streaming: bool = True, *args: object,
               **kwargs: object) -> GStreamerSource | StaticSource:

        if not any(filename.endswith(ext) for ext in self.get_file_extensions()):
            # Refuse to decode anything not specifically listed in the supported
            # file extensions list. This decoder does not yet support video, but
            # it would still decode it and return only the Audio track. This is
            # not desired, since the other decoders will not get a turn. Instead
            # we bail out and let pyglet pass it to the next codec (FFmpeg).
            raise GStreamerDecodeException('Unsupported format.')

        if streaming:
            return GStreamerSource(filename, file)
        else:
            return StaticSource(GStreamerSource(filename, file))


def get_decoders() -> list[GStreamerDecoder]:
    return [GStreamerDecoder()]


def get_encoders() -> list[MediaEncoder]:
    return []
