from __future__ import annotations

import ctypes
import mmap
import os
import threading
import urllib.parse
import warnings
from ctypes import create_string_buffer
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Sequence

import pyglet

from pyglet.event import EventDispatcher
from pyglet.libs.egl import egl
from pyglet.libs.linux.wayland import xkbcommon
from pyglet.window import (
    BaseWindow,
    DefaultMouseCursor,
    ImageMouseCursor,
    MouseCursorException,
    _PlatformEventHandler,
    _ViewEventHandler,
    key,
    mouse,
)
from pyglet.libs.linux.wayland.client import Client, Interface
from pyglet.libs.linux.wayland.wayland_egl import *

if TYPE_CHECKING:
    from pyglet.display.base import Screen

# Platform event data is single item, so use platform event handler directly.
WaylandEventHandler = _PlatformEventHandler
ViewEventHandler = _ViewEventHandler

_FALLBACK_BORDER_SIZE = 4
_FALLBACK_CAPTION_HEIGHT = 24
_FALLBACK_DECORATION_COLOR = b'\xe0\xe0\xe0\xff'  # XRGB8888, little endian
_FALLBACK_CLOSE_BUTTON_COLOR = b'\x4a\x46\xe5\xff'  # XRGB8888, little endian


def _create_close_button_pixels() -> bytes:
    """Return a red 24px button with a simple white close glyph."""
    size = _FALLBACK_CAPTION_HEIGHT
    pixels = bytearray(_FALLBACK_CLOSE_BUTTON_COLOR * size * size)
    for y in range(size):
        for x in range(size):
            if abs(x - y) <= 1 or abs(x - (size - y - 1)) <= 1:
                offset = (y * size + x) * 4
                pixels[offset:offset + 4] = b'\xff\xff\xff\xff'
    return bytes(pixels)


@dataclass
class _FallbackDecorationEdge:
    surface: Interface
    subsurface: Interface
    viewport: Interface
    name: str


@dataclass
class _WaylandShmBuffer:
    """Own the client resources backing one ``wl_buffer``."""

    buffer: Interface
    pool: Interface
    memory: mmap.mmap
    width: int
    height: int
    _destroyed: bool = field(default=False, init=False)

    @classmethod
    def create(cls, client: Client, shm: Interface, pixels: bytes, width: int, height: int,
               pixel_format: int) -> _WaylandShmBuffer:
        """Allocate and populate an SHM buffer, closing partial resources on failure."""
        fd = os.memfd_create('pyglet-wayland-shm', os.MFD_CLOEXEC)
        memory = None
        pool = None
        try:
            os.ftruncate(fd, len(pixels))
            memory = mmap.mmap(fd, len(pixels), access=mmap.ACCESS_WRITE)
            memory[:] = pixels
            pool = shm.create_pool(next(client.oid_pool), fd, len(pixels))
            buffer = pool.create_buffer(next(client.oid_pool), 0, width, height, width * 4, pixel_format)
            return cls(buffer, pool, memory, width, height)
        except Exception:
            if pool is not None:
                pool.destroy()
            if memory is not None:
                memory.close()
            raise
        finally:
            os.close(fd)

    def destroy(self) -> None:
        """Release the Wayland buffer, its pool, and its mapped storage once."""
        if self._destroyed:
            return
        self._destroyed = True
        try:
            self.buffer.destroy()
        finally:
            try:
                self.pool.destroy()
            finally:
                self.memory.close()


@dataclass
class _WaylandImageCursor:
    surface: Interface
    buffer: _WaylandShmBuffer
    hot_x: int
    hot_y: int

    def destroy(self) -> None:
        try:
            self.surface.destroy()
        finally:
            self.buffer.destroy()


@dataclass
class _FallbackDecorations:
    """State owned by the minimal client-side decoration fallback."""

    subcompositor: Interface | None = None
    viewporter: Interface | None = None
    background_buffer: _WaylandShmBuffer | None = None
    close_button_buffer: _WaylandShmBuffer | None = None
    edges: dict[str, _FallbackDecorationEdge] = field(default_factory=dict)
    pointer_edge: _FallbackDecorationEdge | None = None
    pointer_x: int = 0
    pointer_y: int = 0


class _FallbackDecorationManager:
    """Own the minimal client-side decoration surfaces and their input.

    Currently only create a top bar to drag, edges for resizing, and a close button.
    """

    def __init__(self, window: WaylandWindow) -> None:
        self._window = window
        self._decorations = _FallbackDecorations()

    @property
    def pointer_edge(self) -> _FallbackDecorationEdge | None:
        return self._decorations.pointer_edge

    def create(self) -> None:
        window = self._window
        decorations = self._decorations
        if decorations.edges or window._shadow or window._fullscreen or window._style is not None:
            return

        viewporter_protocol = window.client.protocol_dict.get('viewporter')
        if not viewporter_protocol or not window.client.globals.get('wp_viewporter'):
            return
        if not window.client.globals.get('wl_subcompositor') or not window.client.globals.get('wl_shm'):
            return

        wayland_protocol = window.client.protocol_dict['wayland']
        decorations.subcompositor = wayland_protocol.bind_interface('wl_subcompositor')
        decorations.viewporter = viewporter_protocol.bind_interface('wp_viewporter')
        decorations.background_buffer = window._create_wayland_buffer(
            _FALLBACK_DECORATION_COLOR, 1, 1, pixel_format=1,
        )
        decorations.close_button_buffer = window._create_wayland_buffer(
            _create_close_button_pixels(), _FALLBACK_CAPTION_HEIGHT, _FALLBACK_CAPTION_HEIGHT, pixel_format=1,
        )

        for name in ('top', 'left', 'right', 'bottom', 'close'):
            surface = window.wl_compositor.create_surface(next(window.client.oid_pool))
            subsurface = decorations.subcompositor.get_subsurface(
                next(window.client.oid_pool), surface, window.wl_surface,
            )
            viewport = decorations.viewporter.get_viewport(next(window.client.oid_pool), surface)
            subsurface.set_desync()
            buffer = decorations.close_button_buffer if name == 'close' else decorations.background_buffer
            surface.attach(buffer.buffer, 0, 0)
            surface.damage(0, 0, 1, 1)
            surface.commit()
            decorations.edges[name] = _FallbackDecorationEdge(surface, subsurface, viewport, name)

        self.update()

    def update(self) -> None:
        if not self._decorations.edges:
            return

        window = self._window
        dimensions = {
            'top': (0, -_FALLBACK_CAPTION_HEIGHT, window._width, _FALLBACK_CAPTION_HEIGHT),
            'left': (-_FALLBACK_BORDER_SIZE, -_FALLBACK_CAPTION_HEIGHT,
                     _FALLBACK_BORDER_SIZE, window._height + _FALLBACK_CAPTION_HEIGHT),
            'right': (window._width, -_FALLBACK_CAPTION_HEIGHT,
                      _FALLBACK_BORDER_SIZE, window._height + _FALLBACK_CAPTION_HEIGHT),
            'bottom': (-_FALLBACK_BORDER_SIZE, window._height,
                       window._width + _FALLBACK_BORDER_SIZE * 2, _FALLBACK_BORDER_SIZE),
            'close': (window._width - _FALLBACK_CAPTION_HEIGHT, -_FALLBACK_CAPTION_HEIGHT,
                      _FALLBACK_CAPTION_HEIGHT, _FALLBACK_CAPTION_HEIGHT),
        }
        for name, edge in self._decorations.edges.items():
            x, y, width, height = dimensions[name]
            edge.subsurface.set_position(x, y)
            edge.viewport.set_destination(width, height)
            edge.surface.commit()

    def destroy(self) -> None:
        decorations = self._decorations
        for edge in decorations.edges.values():
            edge.viewport.destroy()
            edge.subsurface.destroy()
            edge.surface.destroy()
        decorations.edges.clear()
        decorations.pointer_edge = None
        for buffer in (decorations.background_buffer, decorations.close_button_buffer):
            if buffer:
                buffer.destroy()
        decorations.background_buffer = None
        decorations.close_button_buffer = None

    def pointer_enter(self, surface, x: int, y: int) -> bool:
        self._decorations.pointer_edge = next(
            (edge for edge in self._decorations.edges.values()
             if ctypes.cast(surface, ctypes.c_void_p).value == ctypes.cast(edge.surface._proxy, ctypes.c_void_p).value),
            None,
        )
        if self._decorations.pointer_edge is None:
            return False
        self._decorations.pointer_x = x
        self._decorations.pointer_y = y
        return True

    def pointer_motion(self, x: int, y: int) -> bool:
        if self._decorations.pointer_edge is None:
            return False
        self._decorations.pointer_x = x
        self._decorations.pointer_y = y
        return True

    def pointer_leave(self) -> bool:
        if self._decorations.pointer_edge is None:
            return False
        self._decorations.pointer_edge = None
        return True

    def handle_button(self, serial: int) -> None:
        edge = self._decorations.pointer_edge
        if edge is None:
            return

        window = self._window
        if edge.name == 'close':
            window.dispatch_event('on_close')
            return

        x, y = self._decorations.pointer_x, self._decorations.pointer_y
        if edge.name == 'top' and y >= _FALLBACK_BORDER_SIZE:
            window.xdg_toplevel.move(window.wl_seat, serial)
            return
        if not window._resizable:
            return

        resize_edge = {
            'top': 1,
            'left': 5 if y < _FALLBACK_BORDER_SIZE else 4,
            'right': 6 if y < _FALLBACK_BORDER_SIZE else 8,
            'bottom': 9 if x < _FALLBACK_BORDER_SIZE else 10 if x > window._width + _FALLBACK_BORDER_SIZE else 2,
        }[edge.name]
        window.xdg_toplevel.resize(window.wl_seat, serial, resize_edge)


# symbol,ctrl -> motion mapping
_motion_map: dict[tuple[int, bool], int] = {
    (key.UP, False): key.MOTION_UP,
    (key.RIGHT, False): key.MOTION_RIGHT,
    (key.DOWN, False): key.MOTION_DOWN,
    (key.LEFT, False): key.MOTION_LEFT,
    (key.RIGHT, True): key.MOTION_NEXT_WORD,
    (key.LEFT, True): key.MOTION_PREVIOUS_WORD,
    (key.HOME, False): key.MOTION_BEGINNING_OF_LINE,
    (key.END, False): key.MOTION_END_OF_LINE,
    (key.PAGEUP, False): key.MOTION_PREVIOUS_PAGE,
    (key.PAGEDOWN, False): key.MOTION_NEXT_PAGE,
    (key.HOME, True): key.MOTION_BEGINNING_OF_FILE,
    (key.END, True): key.MOTION_END_OF_FILE,
    (key.BACKSPACE, False): key.MOTION_BACKSPACE,
    (key.DELETE, False): key.MOTION_DELETE,
    (key.A, True): key.MOTION_SELECT_ALL,
    (key.C, True): key.MOTION_COPY,
    (key.V, True): key.MOTION_PASTE,
}

_modifier_map: dict[int, int] = {
    key.LCTRL: key.MOD_CTRL, key.RCTRL: key.MOD_CTRL,
    key.LSHIFT: key.MOD_SHIFT, key.RSHIFT: key.MOD_SHIFT,
    key.LALT: key.MOD_ALT, key.RALT: key.MOD_ALT,
    key.LWINDOWS: key.MOD_WINDOWS, key.RWINDOWS: key.MOD_WINDOWS,
    key.CAPSLOCK: key.MOD_CAPSLOCK,
    key.NUMLOCK: key.MOD_NUMLOCK,
    key.SCROLLLOCK: key.MOD_SCROLLLOCK,
}


class WaylandWindow(BaseWindow):

    egl_display_connection = None
    egl_window = None
    egl_surface = None

    client: Client = None
    wl_compositor: Interface
    wl_surface: Interface
    xdg_wm_base: Interface
    wdg_surface: Interface
    xdg_toplevel: Interface | None
    xdg_decoration_manager: Interface | None = None
    xdg_toplevel_decoration: Interface | None = None
    wl_pointer: Interface | None = None
    wl_keyboard: Interface | None = None

    def __init__(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
        self._mouse_buttons = 0
        self._key_modifiers = 0
        self._has_keyboard_focus = False
        self.egl_window = None
        self._buffer_scale = 1.0
        self._viewport = None
        self._fractional_scale = None
        self._relative_pointer = None
        self._locked_pointer = None
        self.xdg_toplevel = None
        self._pointer_enter_serial: int | None = None
        self._wayland_image_cursor: _WaylandImageCursor | None = None
        self._wayland_shm: Interface | None = None
        self._xdg_activation = None
        self._xdg_activation_token = None
        self._data_offers: dict[int, Interface] = {}
        self._data_offer_mime_types: dict[int, set[str]] = {}
        self._file_drag_offer: Interface | None = None
        self._file_drag_active = False
        self._file_drag_position = (0, 0)
        self._fallback_decorations = _FallbackDecorationManager(self)
        self._entered_outputs = {}

        self.xkb_context = xkbcommon.xkb_context_new(xkbcommon.enum_xkb_context_flags(xkbcommon.XKB_CONTEXT_NO_FLAGS))
        self.xkb_keymap = None
        self.xkb_state_withmod = None   # updated when modifiers are pressed
        self.xkb_state_default = None   # never gets updated (for reference)
        self._text_buffer = create_string_buffer(4)
        super().__init__(*args, **kwargs)

    def _recreate(self, changes: Sequence[str]) -> None:
        if 'fullscreen' not in changes:
            return

        if self._fullscreen:
            self._fallback_decorations.destroy()
            self.xdg_toplevel.set_fullscreen(self.screen.wl_output)
        else:
            self.xdg_toplevel.unset_fullscreen()
            self._fallback_decorations.create()

    def switch_to(self) -> None:
        if self.context:
            self.context.set_current()

    def set_caption(self, caption: str) -> None:
        self._caption = caption
        if self.xdg_toplevel:
            self.xdg_toplevel.set_title(caption)

    def set_minimum_size(self, width: int, height: int) -> None:
        super().set_minimum_size(width, height)
        if self.xdg_toplevel:
            self.xdg_toplevel.set_min_size(width, height)

    def set_maximum_size(self, width: int, height: int) -> None:
        super().set_maximum_size(width, height)
        if self.xdg_toplevel:
            self.xdg_toplevel.set_max_size(width, height)

    def set_size(self, width: int, height: int) -> None:
        super().set_size(width, height)
        if self.egl_window:
            wl_egl_window_resize(self.egl_window, *self.get_framebuffer_size(), 0, 0)
        if self._viewport:
            self._viewport.set_destination(width, height)
        self._fallback_decorations.update()

    def get_size(self) -> tuple[int, int]:
        return self._width, self._height

    def get_current_screen(self) -> Screen:
        """Return the most recently entered output still occupied by the surface.

        Wayland reports output membership without overlap areas. Before any
        output is entered, or after all outputs are left, use the configured
        fullscreen target.
        """
        for output in reversed(tuple(self._entered_outputs)):
            screen = self.display._output_screens.get(output)
            if screen is not None:
                return screen
        return self._screen

    def get_framebuffer_size(self) -> tuple[int, int]:
        return round(self._width * self._buffer_scale), round(self._height * self._buffer_scale)

    def set_location(self, x: int, y: int) -> None:
        # Wayland does not allow setting window positions.
        warnings.warn('Wayland compositors control window placement; set_location() is ignored.', stacklevel=2)

    def get_location(self) -> tuple[int, int]:
        # Wayland deliberately does not expose global window coordinates.
        return 0, 0

    def activate(self) -> None:
        activation_protocol = self.client.protocol_dict.get('xdg_activation_v1')
        if not activation_protocol or not self.client.globals.get('xdg_activation_v1'):
            return

        self._xdg_activation = activation_protocol.bind_interface('xdg_activation_v1')
        token = self._xdg_activation.get_activation_token(next(self.client.oid_pool))
        token.set_surface(self.wl_surface)
        token.set_app_id(self._caption)
        token.set_handler('done', self.xdg_activation_token_done_handler)
        token.commit()
        self._xdg_activation_token = token

    def set_visible(self, visible: bool = True) -> None:
        # xdg-shell has no request to hide an existing toplevel.  Keep the
        # public state in sync; a later buffer commit makes it visible again.
        super().set_visible(visible)

    def minimize(self) -> None:
        self.xdg_toplevel.set_minimized()

    def maximize(self) -> None:
        self.xdg_toplevel.set_maximized()

    def set_vsync(self, vsync: bool) -> None:
        super().set_vsync(vsync)
        if self.context:
            self.context.set_vsync(vsync)

    def set_mouse_cursor_platform_visible(self, platform_visible: bool | None = None) -> None:
        if self.wl_pointer is None or self._pointer_enter_serial is None:
            return

        if platform_visible is None:
            platform_visible = self._mouse_visible and (
                not self._mouse_cursor.api_drawable
                or isinstance(self._mouse_cursor, ImageMouseCursor) and self._mouse_cursor.hw_drawable
            )

        if not platform_visible:
            self.wl_pointer.set_cursor(self._pointer_enter_serial, None, 0, 0)
        elif isinstance(self._mouse_cursor, ImageMouseCursor) and self._mouse_cursor.hw_drawable:
            cursor = self._get_wayland_image_cursor(self._mouse_cursor)
            self.wl_pointer.set_cursor(self._pointer_enter_serial, cursor.surface, cursor.hot_x, cursor.hot_y)
            cursor.surface.attach(cursor.buffer.buffer, 0, 0)
            cursor.surface.damage(0, 0, cursor.buffer.width, cursor.buffer.height)
            cursor.surface.commit()

    def close(self) -> None:
        if not self.context:
            return

        try:
            self._fallback_decorations.destroy()
        finally:
            try:
                if self._wayland_image_cursor is not None:
                    self._wayland_image_cursor.destroy()
                    self._wayland_image_cursor = None
            finally:
                super().close()

    def set_exclusive_mouse(self, exclusive: bool = True) -> None:
        if exclusive == self._mouse_exclusive:
            return
        if self.wl_pointer is None:
            warnings.warn('Wayland pointer is not available for exclusive mouse mode.', stacklevel=2)
            return

        if exclusive:
            relative_protocol = self.client.protocol_dict.get('relative_pointer_unstable_v1')
            constraints_protocol = self.client.protocol_dict.get('pointer_constraints_unstable_v1')
            if not relative_protocol or not constraints_protocol or not self.client.globals.get('zwp_relative_pointer_manager_v1') \
                    or not self.client.globals.get('zwp_pointer_constraints_v1'):
                raise MouseCursorException('Wayland compositor does not support relative-pointer and pointer-constraints.')
            relative_manager = relative_protocol.bind_interface('zwp_relative_pointer_manager_v1')
            constraints_manager = constraints_protocol.bind_interface('zwp_pointer_constraints_v1')
            self._relative_pointer = relative_manager.get_relative_pointer(next(self.client.oid_pool), self.wl_pointer)
            self._relative_pointer.set_handler('relative_motion', self.zwp_relative_pointer_relative_motion_handler)
            self._locked_pointer = constraints_manager.lock_pointer(
                next(self.client.oid_pool), self.wl_surface, self.wl_pointer, None, 2,
            )
            self._locked_pointer.set_handlers(
                locked=self.zwp_locked_pointer_locked_handler,
                unlocked=self.zwp_locked_pointer_unlocked_handler,
            )
            self.set_mouse_cursor_platform_visible(False)
        else:
            if self._locked_pointer:
                self._locked_pointer.destroy()
                self._locked_pointer = None
            if self._relative_pointer:
                self._relative_pointer.destroy()
                self._relative_pointer = None
            self.set_mouse_cursor_platform_visible()
        super().set_exclusive_mouse(exclusive)

    def set_exclusive_keyboard(self, exclusive: bool = True) -> None:
        pass

    def get_system_mouse_cursor(self, name: str) -> DefaultMouseCursor:
        if name == self.CURSOR_DEFAULT:
            return DefaultMouseCursor()
        msg = f'Wayland system cursor "{name}" requires libwayland-cursor, which is not yet bound by pyglet.'
        raise MouseCursorException(msg)

    def set_icon(self, *images) -> None:  # noqa: ANN002
        warnings.warn(
            'xdg-shell has no window-icon request. Set the icon through the application desktop file instead.',
            stacklevel=2,
        )

    def dispatch_events(self) -> None:
        while self._event_queue:
            EventDispatcher.dispatch_event(self, *self._event_queue.popleft())

    def dispatch_pending_events(self) -> None:
        self.dispatch_events()

    def _create(self) -> None:
        self.egl_display_connection = self.display.display_connection

        self._buffer_scale = max(1, int(self._screen.get_scale()))
        self._dpi = 96 * self._buffer_scale
        self._scale = 1.0

        self.client = self.display.client
        self.client.sync()

        self.wl_compositor = self.client.protocol_dict['wayland'].bind_interface('wl_compositor')

        self.wl_surface = self.wl_compositor.create_surface(next(self.client.oid_pool))
        self._entered_outputs.clear()
        self.wl_surface.set_handlers(
            preferred_buffer_scale=self.wl_surface_preferred_buffer_scale_handler,
            enter=self.wl_surface_enter_handler,
            leave=self.wl_surface_leave_handler,
        )
        self.wl_surface.set_buffer_scale(self._buffer_scale)

        fractional_protocol = self.client.protocol_dict.get('fractional_scale_v1')
        viewporter_protocol = self.client.protocol_dict.get('viewporter')
        if fractional_protocol and viewporter_protocol and self.client.globals.get('wp_fractional_scale_manager_v1') \
                and self.client.globals.get('wp_viewporter'):
            fractional_manager = fractional_protocol.bind_interface('wp_fractional_scale_manager_v1')
            viewporter = viewporter_protocol.bind_interface('wp_viewporter')
            self._viewport = viewporter.get_viewport(next(self.client.oid_pool), self.wl_surface)
            self._viewport.set_destination(self._width, self._height)
            self._fractional_scale = fractional_manager.get_fractional_scale(next(self.client.oid_pool), self.wl_surface)
            self._fractional_scale.set_handler('preferred_scale', self.wp_fractional_scale_preferred_scale_handler)
            self.wl_surface.set_buffer_scale(1)

        self.xdg_wm_base = self.client.protocol_dict['xdg_shell'].bind_interface('xdg_wm_base')
        self.xdg_wm_base.set_handler('ping', self.xdg_base_ping_handler)

        self.xdg_surface = self.xdg_wm_base.get_xdg_surface(next(self.client.oid_pool), self.wl_surface)
        self.xdg_surface.set_handler('configure', self.xdg_surface_configure_handler)

        self.xdg_toplevel = self.xdg_surface.get_toplevel(next(self.client.oid_pool))
        self.xdg_toplevel.set_handlers(
            configure=self.xdg_toplevel_configure_handler,
            configure_bounds=self.xdg_toplevel_configure_bounds,
            close=self.xdg_toplevel_close_handler,
        )
        self.xdg_toplevel.set_parent(None)
        self.xdg_toplevel.set_title(self._caption)
        self.xdg_toplevel.set_app_id(self._caption)

        # A Wayland compositor only draws a title bar and its controls when
        # the client and compositor negotiate server-side decorations.  The
        # extension is optional, so retain the undecorated fallback on
        # compositors that do not advertise it.
        decoration_protocol = self.client.protocol_dict.get('xdg_decoration_unstable_v1')
        if decoration_protocol and self.client.globals.get('zxdg_decoration_manager_v1'):
            self.xdg_decoration_manager = decoration_protocol.bind_interface('zxdg_decoration_manager_v1')
            self.xdg_toplevel_decoration = self.xdg_decoration_manager.get_toplevel_decoration(
                next(self.client.oid_pool), self.xdg_toplevel,
            )
            self.xdg_toplevel_decoration.set_handler('configure', self.xdg_toplevel_decoration_configure_handler)
            self.xdg_toplevel_decoration.set_mode(2)  # zxdg_toplevel_decoration_v1.mode.server_side
        else:
            self._fallback_decorations.create()

        if self.client.globals.get('wl_seat'):
            self.wl_seat = self.client.protocol_dict['wayland'].bind_interface('wl_seat')
            self.wl_seat.set_handler('capabilities', self.wl_seat_capabilities_handler)
            self.client.sync()

            if self._file_drops and self.client.globals.get('wl_data_device_manager'):
                self.wl_data_device_manager = self.client.protocol_dict['wayland'].bind_interface('wl_data_device_manager')
                self.wl_data_device = self.wl_data_device_manager.get_data_device(
                    next(self.client.oid_pool), self.wl_seat,
                )
                self.wl_data_device.set_handlers(
                    data_offer=self.wl_data_device_data_offer_handler,
                    enter=self.wl_data_device_enter_handler,
                    leave=self.wl_data_device_leave_handler,
                    motion=self.wl_data_device_motion_handler,
                    drop=self.wl_data_device_drop_handler,
                )

        if not self.egl_surface and not self._shadow:
            # An EGL window needs to be created from a Wayland surface,
            self.egl_window = wl_egl_window_create(self.wl_surface._proxy, *self.get_framebuffer_size())

            if not self.egl_window:
                err = egl.eglGetError()
                msg = f"Failed to create EGL wayland window. Error: 0x{err:04x}"
                raise Exception(msg)

            self._assign_config()

            self.egl_surface = egl.eglCreateWindowSurface(
                self.egl_display_connection,
                self.config._egl_config,
                self.egl_window,
                None,
            )

            if not self.egl_surface:
                err = egl.eglGetError()
                msg = f"Failed to create egl surface. Error: 0x{err:04x}"
                raise Exception(msg)

            # self.canvas = WaylandCanvas(self.display, self.egl_surface)
            self.context.attach(self)
            self.set_vsync(self._vsync)
            self.dispatch_event('_on_internal_resize', self._width, self._height)

    # Start Wayland event handlers

    def wl_surface_enter_handler(self, output) -> None:
        output_id = ctypes.cast(output, ctypes.c_void_p).value
        self._entered_outputs.pop(output_id, None)
        self._entered_outputs[output_id] = None

    def wl_surface_leave_handler(self, output) -> None:
        self._entered_outputs.pop(ctypes.cast(output, ctypes.c_void_p).value, None)

    def wl_seat_capabilities_handler(self, capabilities) -> None:
        """Create input objects only for capabilities offered."""
        if capabilities & 1 and self.wl_pointer is None:  # wl_seat.capability.pointer
            self.wl_pointer = self.wl_seat.get_pointer(next(self.client.oid_pool))
            self.wl_pointer.set_handlers(
                motion=self.wl_pointer_motion_handler,
                button=self.wl_pointer_button_handler,
                axis=self.wl_pointer_axis_handler,
                axis_value120=self.wl_pointer_axis_value120_handler,
                enter=self.wl_pointer_enter_handler,
                leave=self.wl_pointer_leave_handler,
            )
        if capabilities & 2 and self.wl_keyboard is None:  # wl_seat.capability.keyboard
            self.wl_keyboard = self.wl_seat.get_keyboard(next(self.client.oid_pool))
            self.wl_keyboard.set_handlers(
                keymap=self.wl_keyboard_keymap_handler,
                modifiers=self.wl_keyboard_modifiers_handler,
                key=self.wl_keyboard_key_handler,
                enter=self.wl_keyboard_enter_handler,
                leave=self.wl_keyboard_leave_handler,
            )

    def wl_data_device_data_offer_handler(self, offer: Interface) -> None:
        self._data_offers[offer.oid] = offer
        self._data_offer_mime_types[offer.oid] = set()
        offer.set_handler('offer', lambda mime_type: self.wl_data_offer_offer_handler(offer, mime_type))

    def wl_data_offer_offer_handler(self, offer: Interface, mime_type: bytes | str) -> None:
        if isinstance(mime_type, bytes):
            mime_type = mime_type.decode()
        self._data_offer_mime_types[offer.oid].add(mime_type)

    def _get_data_offer(self, offer_proxy) -> Interface | None:
        if not offer_proxy:
            return None
        offer_address = ctypes.cast(offer_proxy, ctypes.c_void_p).value
        return next(
            (offer for offer in self._data_offers.values()
             if ctypes.cast(offer._proxy, ctypes.c_void_p).value == offer_address),
            None,
        )

    def wl_data_device_enter_handler(self, serial, surface, x, y, offer_proxy) -> None:
        if ctypes.cast(surface, ctypes.c_void_p).value != ctypes.cast(self.wl_surface._proxy, ctypes.c_void_p).value:
            return

        self._file_drag_offer = self._get_data_offer(offer_proxy)
        self._file_drag_position = int(x / 256), self._height - int(y / 256)
        self._file_drag_active = False
        if self._file_drag_offer and 'text/uri-list' in self._data_offer_mime_types[self._file_drag_offer.oid]:
            self._file_drag_offer.accept(serial, 'text/uri-list')
            self._file_drag_active = True
            self.dispatch_event('on_file_drag_enter', *self._file_drag_position, [])

    def wl_data_device_motion_handler(self, time, x, y) -> None:
        self._file_drag_position = int(x / 256), self._height - int(y / 256)
        if self._file_drag_active:
            self.dispatch_event('on_file_drag', *self._file_drag_position, [])

    def wl_data_device_leave_handler(self) -> None:
        if self._file_drag_active:
            self.dispatch_event('on_file_drag_exit')
        self._file_drag_offer = None
        self._file_drag_active = False

    def wl_data_device_drop_handler(self) -> None:
        offer = self._file_drag_offer
        if offer is None or not self._file_drag_active:
            return

        read_fd, write_fd = os.pipe()
        offer.receive('text/uri-list', write_fd)
        os.close(write_fd)
        threading.Thread(
            target=self._read_file_drop,
            args=(read_fd, offer, self._file_drag_position),
            daemon=True,
        ).start()

    def _read_file_drop(self, read_fd: int, offer: Interface, position: tuple[int, int]) -> None:
        with os.fdopen(read_fd, 'rb') as data_stream:
            data = data_stream.read().decode(errors='replace')
        paths = [
            urllib.parse.unquote(urllib.parse.urlsplit(uri).path)
            for uri in data.splitlines()
            if uri and not uri.startswith('#') and urllib.parse.urlsplit(uri).scheme == 'file'
        ]
        if paths:
            self.dispatch_event('on_file_drop', *position, paths)
        if self._file_drag_offer is offer:
            self.wl_data_device_leave_handler()

    def _create_wayland_buffer(self, pixels: bytes, width: int, height: int,
                               pixel_format: int = 0) -> _WaylandShmBuffer:
        """Create an ARGB/XRGB shared-memory buffer for a Wayland surface."""
        if self._wayland_shm is None:
            self._wayland_shm = self.client.protocol_dict['wayland'].bind_interface('wl_shm')
        return _WaylandShmBuffer.create(
            self.client, self._wayland_shm, pixels, width, height, pixel_format,
        )

    def _get_wayland_image_cursor(self, cursor: ImageMouseCursor) -> _WaylandImageCursor:
        if self._wayland_image_cursor:
            self._wayland_image_cursor.destroy()

        image = cursor.texture.get_image_data()
        width, height = image.width, image.height
        rgba = image.get_bytes('RGBA', -width * 4)
        pixels = bytearray(width * height * 4)
        for index in range(0, len(rgba), 4):
            red, green, blue, alpha = rgba[index:index + 4]
            pixels[index:index + 4] = bytes((
                blue * alpha // 255,
                green * alpha // 255,
                red * alpha // 255,
                alpha,
            ))

        surface = self.wl_compositor.create_surface(next(self.client.oid_pool))
        buffer = self._create_wayland_buffer(bytes(pixels), width, height)
        self._wayland_image_cursor = _WaylandImageCursor(
            surface,
            buffer,
            min(max(0, int(cursor.hot_x)), width - 1),
            min(max(0, height - int(cursor.hot_y)), height - 1),
        )
        return self._wayland_image_cursor

    def xdg_base_ping_handler(self, serial):
        """Keep-alive response to the ping event"""
        self.xdg_wm_base.pong(serial)

    def wl_surface_preferred_buffer_scale_handler(self, factor):
        if getattr(self, '_fractional_scale', None) is not None:
            return
        factor = max(1, factor)
        if factor == self._buffer_scale:
            return

        self._buffer_scale = factor
        self._dpi = 96 * factor
        self.wl_surface.set_buffer_scale(factor)
        if self.egl_window:
            wl_egl_window_resize(self.egl_window, *self.get_framebuffer_size(), 0, 0)
        self.dispatch_event('_on_internal_scale', factor, self._dpi)

    def wp_fractional_scale_preferred_scale_handler(self, scale_120) -> None:
        factor = max(1.0, scale_120 / 120.0)
        if factor == self._buffer_scale:
            return

        self._buffer_scale = factor
        self._dpi = 96 * factor
        self.wl_surface.set_buffer_scale(1)
        if self._viewport:
            self._viewport.set_destination(self._width, self._height)
        if self.egl_window:
            wl_egl_window_resize(self.egl_window, *self.get_framebuffer_size(), 0, 0)
        self.dispatch_event('_on_internal_scale', factor, self._dpi)

    def zwp_relative_pointer_relative_motion_handler(self, utime_hi, utime_lo, dx, dy, dx_unaccel, dy_unaccel):
        dx /= 256.0
        dy /= 256.0
        self.dispatch_event('on_mouse_motion', self._mouse_x, self._mouse_y, dx, -dy)

    def zwp_locked_pointer_locked_handler(self) -> None:
        pass

    def zwp_locked_pointer_unlocked_handler(self) -> None:
        pass

    def xdg_toplevel_configure_handler(self, width, height, states):
        print(" --> xdg_toplevel configure event", width, height, states)
        # A zero size means the compositor leaves the client to choose its
        # size.  Non-zero dimensions must be applied before the next buffer
        # commit; this is how maximize and interactive resizing take effect.
        if width and height:
            self._width, self._height = width, height
            if self.egl_window:
                wl_egl_window_resize(self.egl_window, *self.get_framebuffer_size(), 0, 0)
            if self._viewport:
                self._viewport.set_destination(width, height)
            self._fallback_decorations.update()
            self.dispatch_event('_on_internal_resize', width, height)

    def xdg_toplevel_configure_bounds(self, width, height):
        print(" --> xdg_toplevel_configure_bounds event", width, height)

    def xdg_toplevel_decoration_configure_handler(self, mode):
        """Record the compositor-selected decoration mode.

        The compositor can reject server-side decorations and choose
        client-side ones instead.  The ``xdg_surface.configure`` event that
        follows is acknowledged by ``xdg_surface_configure_handler``.
        """
        self._decoration_mode = mode
        if mode == 1:  # zxdg_toplevel_decoration_v1.mode.client_side
            self._fallback_decorations.create()

    def xdg_toplevel_close_handler(self):
        self.dispatch_event('on_close')

    def xdg_activation_token_done_handler(self, token):
        """Activate the toplevel once the compositor has issued a valid token."""
        self._xdg_activation.activate(token, self.wl_surface)
        self._xdg_activation_token = None

    def xdg_surface_configure_handler(self, *args):
        print(" --> xdg_surface configure event", args)
        self.xdg_surface.ack_configure(args[0])

    def wl_pointer_button_handler(self, serial, time, button, state):
        if self._fallback_decorations.pointer_edge:
            if button == 0x110 and self.wl_pointer.enums['button_state'][state].name == 'pressed':
                self._fallback_decorations.handle_button(serial)
            return

        mouse_button = {0x110: mouse.LEFT,
                        0x111: mouse.RIGHT,
                        0x112: mouse.MIDDLE,
                        0x113: mouse.MOUSE4,
                        0x114: mouse.MOUSE5}[button]

        if self.wl_pointer.enums['button_state'][state].name == 'pressed':
            self._mouse_buttons |= mouse_button
            self.dispatch_event('on_mouse_press', self._mouse_x, self._mouse_y, mouse_button, self._key_modifiers)

        elif self.wl_pointer.enums['button_state'][state].name == 'released':
            self._mouse_buttons &= ~mouse_button
            self.dispatch_event('on_mouse_release', self._mouse_x, self._mouse_y, mouse_button, self._key_modifiers)

    def wl_pointer_motion_handler(self, time, surface_x, surface_y):
        if self._fallback_decorations.pointer_motion(surface_x, surface_y):
            return

        x, y = self._surface_to_window_coordinates(surface_x, surface_y)
        dx = x - self._mouse_x
        dy = y - self._mouse_y
        self._mouse_x = x
        self._mouse_y = y

        if self._mouse_buttons:
            self.dispatch_event('on_mouse_drag', x, y, dx, dy, self._mouse_buttons, self._key_modifiers)
        else:
            self.dispatch_event('on_mouse_motion', x, y, dx, dy)

    def wl_pointer_axis_value120_handler(self, axis, value120):
        # 0: vertical axis, 1: horizontal axis
        axis = self.wl_pointer.enums['axis'][axis].value
        scroll_x, scroll_y = {0: (0, -1), 1: (-1, 0)}[axis]
        scroll_x *= (value120 / 120)
        scroll_y *= (value120 / 120)
        self.dispatch_event('on_mouse_scroll', self._mouse_x, self._mouse_y, scroll_x, scroll_y)

    def wl_pointer_axis_handler(self, time, axis, value):
        """Handle scrolling on wl_pointer versions before axis_value120."""
        axis = self.wl_pointer.enums['axis'][axis].value
        value /= 256.0  # wl_fixed_t uses 24.8 fixed-point values.
        scroll_x, scroll_y = {0: (0, -value), 1: (-value, 0)}[axis]
        self.dispatch_event('on_mouse_scroll', self._mouse_x, self._mouse_y, scroll_x, scroll_y)

    # def wl_pointer_frame_handler(self):
    #     print("end of pointer frame")

    def wl_pointer_enter_handler(self, serial, surface, surface_x, surface_y):
        if self._fallback_decorations.pointer_enter(surface, surface_x, surface_y):
            return
        if not self._is_main_surface(surface):
            return

        self._pointer_enter_serial = serial
        self._mouse_in_window = True
        self.set_mouse_cursor_platform_visible()
        self.dispatch_event('on_mouse_enter', *self._surface_to_window_coordinates(surface_x, surface_y))

    def wl_pointer_leave_handler(self, serial, surface):
        if self._fallback_decorations.pointer_leave():
            return
        if not self._is_main_surface(surface):
            return

        self._pointer_enter_serial = None
        self._mouse_in_window = False
        self.dispatch_event('on_mouse_leave', self._mouse_x, self._mouse_y)

    def wl_keyboard_enter_handler(self, serial, surface, keys):
        if self._is_main_surface(surface):
            self._has_keyboard_focus = True

    def wl_keyboard_leave_handler(self, serial, surface):
        if self._is_main_surface(surface):
            self._has_keyboard_focus = False

    def _is_main_surface(self, surface) -> bool:
        """Return whether a Wayland callback targets this window's content surface."""
        return ctypes.cast(surface, ctypes.c_void_p).value == ctypes.cast(self.wl_surface._proxy, ctypes.c_void_p).value

    def _surface_to_window_coordinates(self, x: int, y: int) -> tuple[float, float]:
        """Convert top-left Wayland surface coordinates to pyglet coordinates."""
        return x / self._scale, self.height - y / self._scale

    def wl_keyboard_keymap_handler(self, fmt, fd, size):
        # Does not work reliably without mapping:
        mmap_obj = mmap.mmap(fd, size, prot=mmap.PROT_READ, flags=mmap.MAP_PRIVATE)
        data = ctypes.create_string_buffer(mmap_obj.read())
        os.close(fd)

        # Note: There is only one format, so no need to check:
        # fmt = self.wl_keyboard.enums['keymap_format'][fmt].name
        fmt = xkbcommon.enum_xkb_keymap_format(xkbcommon.XKB_KEYMAP_FORMAT_TEXT_V1)
        flags = xkbcommon.enum_xkb_keymap_compile_flags(xkbcommon.XKB_KEYMAP_COMPILE_NO_FLAGS)
        if xkb_keymap := xkbcommon.xkb_keymap_new_from_string(self.xkb_context, data, fmt, flags):
            self.xkb_keymap = xkb_keymap
            self.xkb_state_withmod = xkbcommon.xkb_state_new(self.xkb_keymap)
            self.xkb_state_default = xkbcommon.xkb_state_new(self.xkb_keymap)

    def wl_keyboard_modifiers_handler(self, serial, mods_depressed, mods_latched, mods_locked, group):
        xkbcommon.xkb_state_update_mask(
            self.xkb_state_withmod, mods_depressed, mods_latched, mods_locked, group, group, group)

    def wl_keyboard_key_handler(self, serial, time, keycode, state):
        keycode += 8    # xkbcommon expects +8
        symbol = xkbcommon.xkb_state_key_get_one_sym(self.xkb_state_default, keycode)
        modifier = _modifier_map.get(symbol, 0)

        # released	0, pressed	1, repeated  2
        state_name = self.wl_keyboard.enums['key_state'][state].name

        if state_name == 'pressed':
            _size = xkbcommon.xkb_state_key_get_utf8(self.xkb_state_withmod, keycode, self._text_buffer, 4)
            character = self._text_buffer[:_size].decode()

            self.dispatch_event('on_text', character)
            self.dispatch_event('on_key_press', symbol, self._key_modifiers)
            self._key_modifiers |= modifier

            if self._key_modifiers & key.MOD_ALT:
                return

            ctrl = self._key_modifiers & key.MOD_CTRL != 0
            shft = self._key_modifiers & key.MOD_SHIFT != 0
            if motion := _motion_map.get((symbol, ctrl), None):
                if shft:
                    self.dispatch_event('on_text_motion_select', motion)
                else:
                    self.dispatch_event('on_text_motion', motion)

        if state_name == 'released':
            self.dispatch_event('on_key_release', symbol, self._key_modifiers)
            self._key_modifiers &= ~modifier

    # End Wayland event handlers


__all__ = ['WaylandWindow']
