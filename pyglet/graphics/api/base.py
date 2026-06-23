from __future__ import annotations

import atexit
import os
import weakref
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generic, Protocol, TypeVar, get_type_hints, Sequence, Callable, NoReturn

from pyglet.graphics import GraphicsIntegrationError, GraphicsBackendError
from pyglet.util import debug_print

if TYPE_CHECKING:
    from pyglet.config import SurfaceConfig
    from pyglet.graphics.api.gl import ObjectSpace
    from pyglet.graphics.buffer import BufferRange
    from pyglet.window import Window

_debug_print = debug_print("debug_api")

class BackendGlobalObject(ABC):  # Temp name for now.
    """A container for backend resources and information that are required throughout the code.

    Meant to be accessed from a global level.
    """
    windows: weakref.WeakKeyDictionary[Window, SurfaceContext]
    current_context: SurfaceContext | NullContext
    _have_context: bool = False

    def __init__(self) -> None:
        self.windows = weakref.WeakKeyDictionary()
        self._current_window = None
        self.current_context = NullContext()

    @property
    def have_context(self) -> bool:
        return self._have_context

    def set_current_context(self, context: SurfaceContext) -> None:
        """Mark the provided surface context as active for this backend."""
        self.current_context = context
        self._current_window = context.window

    def clear_current_context(self, context: SurfaceContext | None = None) -> None:
        """Clear the active context, optionally only if it matches `context`."""
        if context is None or self.current_context is context:
            self.current_context = NullContext()
            self._current_window = None

    def resolve_context(self, context: SurfaceContext | None = None) -> SurfaceContext:
        """Return an explicit context or the backend's active context."""
        resolved = context if context is not None else self.current_context
        if isinstance(resolved, NullContext):
            msg = ("A rendering Context has not yet been created, or has already been deleted. Please ensure "
                   "a context exists (create a Window) before attempting to perform any GPU related activities.")
            raise RuntimeError(msg)
        return resolved

    @property
    @abstractmethod
    def object_space(self) -> ObjectSpace:
        ...

    @abstractmethod
    def get_surface_context(
            self,
            window: Window,
            config: SurfaceConfig,
            shared: SurfaceContext | None) -> SurfaceContext:
        """After a window is created, this will be called.

        This must return a BackendWindowObject object.
        """

    @abstractmethod
    def get_default_configs(self) -> Sequence:
        """Configs to use if none specified."""

    @abstractmethod
    def set_viewport(self, window, x: int, y: int, width: int, height: int) -> None:
        """Set the global viewport."""


class BackendRenderer(ABC):
    """Backend-specific rendering helpers used by draw-context state transitions."""

    def __init__(self, surface_ctx: SurfaceContext) -> None:
        self.surface_ctx = surface_ctx

    @abstractmethod
    def set_viewport(self, x: int, y: int, width: int, height: int) -> None:
        """Set the active viewport."""

    @abstractmethod
    def set_scissor(self, scissor: Any | None) -> None:
        """Set or clear the active scissor rectangle."""

    @abstractmethod
    def set_clear_color(self, r: float, g: float, b: float, a: float) -> None:
        """Set the active clear color."""

    @staticmethod
    def load_package_shader(package, resource_name):
        """Reads a binary resource from the given package or subpackage without external dependencies.

        Args:
            package: The full package path (e.g., 'pyglet.graphics.api.vulkan.shaders').
            resource_name: The resource filename (e.g., 'primitives.vert.spv').

        Returns:
            The binary contents of the resource.
        """
        # Dynamically resolve the package's directory
        package_path = os.path.dirname(__import__(package, fromlist=['']).__file__)
        resource_path = os.path.join(package_path, resource_name)

        # Read the file in binary mode
        with open(resource_path, 'rb') as file:
            return file.read()


class UnavailableBackendError(GraphicsBackendError):
    """An exception that occurs when a backend is not yet available."""
    def __init__(self) -> None:  # noqa: D107
        super().__init__(
            "A backend is not currently available. Ensure the backend choice is correct."
            "If your environment variable is set to no backend, then backend-required resources may have been imported."
        )


class NullBackend(BackendGlobalObject):  # noqa: D101
    def _raise_no_backend(self) -> NoReturn:
        raise UnavailableBackendError

    @property
    def object_space(self) -> ObjectSpace:
        self._raise_no_backend()

    def get_surface_context(self, window: Window, config: SurfaceConfig,
                            shared: SurfaceContext | None = None) -> SurfaceContext:
        self._raise_no_backend()

    def get_default_configs(self) -> Sequence:
        self._raise_no_backend()

    def set_viewport(self, window, x: int, y: int, width: int, height: int) -> None:
        self._raise_no_backend()


class SurfaceInfo(ABC):
    """Base backend capability info shared by all rendering APIs.

    Individual backends are expected to subclass this and fill values in ``query``.
    """
    extensions: set[str]
    vendor: str
    renderer: str
    version: str
    shading_language_version: str
    major_version: int
    minor_version: int
    api: str
    was_queried: bool

    # Common capability limits shared by backends these should be automatically queried by the API.
    MAX_ARRAY_TEXTURE_LAYERS: int
    MAX_TEXTURE_SIZE: int
    MAX_COLOR_ATTACHMENTS: int
    MAX_SAMPLES: int
    MAX_COLOR_TEXTURE_SAMPLES: int
    MAX_TEXTURE_IMAGE_UNITS: int
    MAX_COMBINED_TEXTURE_IMAGE_UNITS: int
    MAX_UNIFORM_BUFFER_BINDINGS: int
    MAX_UNIFORM_BUFFER_OFFSET_ALIGNMENT: int
    MAX_UNIFORM_BLOCK_SIZE: int
    MAX_VERTEX_ATTRIBS: int

    def __init__(self) -> None:
        self.extensions = set()
        self.vendor = ""
        self.renderer = ""
        self.version = "0.0"
        self.shading_language_version = ""
        self.major_version = 0
        self.minor_version = 0
        self.api = "unknown"
        self.was_queried = False

        self.MAX_ARRAY_TEXTURE_LAYERS = 0
        self.MAX_TEXTURE_SIZE = 0
        self.MAX_COLOR_ATTACHMENTS = 0
        self.MAX_SAMPLES = 0
        self.MAX_COLOR_TEXTURE_SAMPLES = 0
        self.MAX_TEXTURE_IMAGE_UNITS = 0
        self.MAX_COMBINED_TEXTURE_IMAGE_UNITS = 0
        self.MAX_UNIFORM_BUFFER_BINDINGS = 0
        self.MAX_UNIFORM_BUFFER_OFFSET_ALIGNMENT = 0
        self.MAX_UNIFORM_BLOCK_SIZE = 0
        self.MAX_VERTEX_ATTRIBS = 0

    @property
    def opengl_api(self) -> str:
        """Compatibility alias for legacy OpenGL naming."""
        return self.api

    @opengl_api.setter
    def opengl_api(self, value: str) -> None:
        self.api = value

    @abstractmethod
    def query(self, *_args: Any, **_kwargs: Any) -> None:
        """Populate backend info from an active context/device."""

    def have_extension(self, extension: str) -> bool:
        """Determine if an extension is available."""
        return extension in self.extensions

    def get_extensions(self) -> set[str]:
        """Get a set of available extensions."""
        return self.extensions

    def get_version(self) -> tuple[int, int]:
        """Get the current major and minor API version."""
        return self.major_version, self.minor_version

    def get_version_string(self) -> str:
        """Get the backend version string."""
        return self.version

    def have_version(self, major: int, minor: int = 0) -> bool:
        """Determine if a version is supported."""
        if not self.major_version and not self.minor_version:
            return False

        return (self.major_version > major or
                (self.major_version == major and self.minor_version >= minor) or
                (self.major_version == major and self.minor_version == minor))

    def get_renderer(self) -> str:
        """Determine the renderer string."""
        return self.renderer

    def get_vendor(self) -> str:
        """Determine the vendor string."""
        return self.vendor

    def get_opengl_api(self) -> str:
        """Compatibility alias for existing OpenGL callers."""
        return self.api


BackendFrameContextT = TypeVar("BackendFrameContextT")


@dataclass
class FrameContext(Generic[BackendFrameContextT]):
    """Per-frame data for one reusable frame slot."""

    backend_ctx: BackendFrameContextT
    frame_index: int = 0
    slot_index: int = 0
    fence: Any | None = None
    buffer_ranges: list[BufferRange] = field(default_factory=list)

    @property
    def has_tracked_uses(self) -> bool:
        return bool(self.buffer_ranges)



class FrameResourceManager(Generic[BackendFrameContextT]):
    """Tracks CPU-side resource reuse against submitted GPU frames.

    This manager is backend-neutral. It does not submit work, allocate command
    buffers, or own API synchronization primitives. Instead, it records which
    CPU-visible resources were used by a frame slot and asks the active
    ``SurfaceContext`` for a completion handle when those resources are
    submitted.

    The shared use case is ring-buffer backed uniform data: a ``BufferRange``
    cannot be overwritten until the frame that used it is complete.

    Backend-specific systems that are already explicitly frame-local, such as
    Vulkan command buffers or descriptor-set arrays, should keep their own
    frame-indexed storage.
    """
    slots: list[FrameContext[BackendFrameContextT]]

    def __init__(self, surface_ctx: SurfaceContext[BackendFrameContextT], slots: int = 3) -> None:
        self.surface_ctx = surface_ctx
        self.slots = [
            FrameContext(
                backend_ctx=surface_ctx.create_backend_frame_context(slot_index),
                slot_index=slot_index,
            )
            for slot_index in range(max(1, int(slots)))
        ]
        self._active_slot = self.slots[0]

    @property
    def active_slot(self) -> FrameContext[BackendFrameContextT]:
        return self._active_slot

    def _release_slot(self, slot: FrameContext[BackendFrameContextT]) -> None:
        if slot.fence is not None:
            self.surface_ctx.delete_frame_fence(slot.fence)
            slot.fence = None
        for buffer_range in slot.buffer_ranges:
            buffer_range.frame_use_count = max(0, buffer_range.frame_use_count - 1)
            buffer_range.in_use = buffer_range.frame_use_count > 0
        slot.buffer_ranges.clear()

    def frame_begin(self, frame_index: int) -> None:
        slot_index = frame_index % len(self.slots)
        slot = self.slots[slot_index]

        if slot.fence is not None:
            if not self.surface_ctx.poll_frame_fence(slot.fence):
                # The frame resources are still in use by the GPU after we've wrapped around.
                # At this point the CPU has caught up to the GPU. Wait for frame to become available.
                assert _debug_print(f"CPU caught up to GPU on frame slot: {slot_index}, frame index: {frame_index}.")
                self.surface_ctx.wait_frame_fence(slot.fence)
            self._release_slot(slot)
        elif slot.has_tracked_uses:
            # Some backends may not expose a completion primitive. In that case
            # resource tracking degrades to frame-slot reuse without blocking.
            self._release_slot(slot)

        slot.frame_index = frame_index
        self._active_slot = slot

    def allocate_buffer_range(self, buffer_range: BufferRange) -> None:
        """Mark a newly selected ring-buffer range as used by the active frame."""
        if buffer_range.in_use:
            msg = "Trying to allocate a buffer range that is already in use by a submitted frame."
            raise GraphicsIntegrationError(msg)

        self.use_buffer_range(buffer_range)

    def use_buffer_range(self, buffer_range: BufferRange) -> None:
        """Mark an existing ring-buffer range as referenced by the active frame."""
        if any(active_range is buffer_range for active_range in self._active_slot.buffer_ranges):
            return

        buffer_range.frame_use_count += 1
        buffer_range.in_use = True
        self._active_slot.buffer_ranges.append(buffer_range)

    def allocate_ubo(self, ubo_range: BufferRange) -> None:
        """Compatibility alias for UBO-backed ring-buffer allocation."""
        self.allocate_buffer_range(ubo_range)

    def use_ubo(self, ubo_range: BufferRange) -> None:
        """Compatibility alias for UBO-backed ring-buffer usage."""
        self.use_buffer_range(ubo_range)

    def frame_submit(self) -> None:
        if not self._active_slot.has_tracked_uses:
            return
        self._active_slot.fence = self.surface_ctx.create_frame_fence()

    def delete(self) -> None:
        for slot in self.slots:
            self._release_slot(slot)


class SurfaceContext(Generic[BackendFrameContextT], ABC):  # Temp name for now.
    """A container for backend resources and information that are tied to a specific Window.

    In OpenGL this would be something like an OpenGL Context, or in Vulkan, a Surface.
    """
    clear_color = (0.0, 0.0, 0.0, 1.0)
    renderer: BackendRenderer
    frame_resources: FrameResourceManager[BackendFrameContextT]

    def __init__(self, global_ctx: BackendGlobalObject, window: Window, config: Any,
                 frames_in_flight: int = 3) -> None:
        self.core = global_ctx
        self.window = window
        self.config = config
        self.frames_in_flight = max(1, int(frames_in_flight))
        self.frame_index = 0
        self._frame_active = False
        self.frame_resources = FrameResourceManager[BackendFrameContextT](self, self.frames_in_flight)

    @property
    @abstractmethod
    def info(self) -> SurfaceInfo:
        """Backend and hardware capability information for this context."""

    def get_info(self) -> SurfaceInfo:
        """Backend and hardware capability information for this context.

        Deprecated, use the `info` property instead.

        .. deprecated:: 3.0
        """
        return self.info

    @abstractmethod
    def set_clear_color(self, r: float, g: float, b: float, a: float) -> None:
        """Sets the clear color for the current Window.

        Default value is black.
        """
        # Backends need to implement setting this value.

    @abstractmethod
    def attach(self, window: Window) -> None:
        """Attaches the specified Window into the backend context.

        This function is called automatically when the operating system Window has been created.
        """

    @property
    def frame_active(self) -> bool:
        return self._frame_active

    @property
    def active_frame(self) -> int:
        """Current reusable frame slot for frame-local backend resources."""
        return self.frame_index % self.frames_in_flight

    @property
    def frame_context(self) -> FrameContext[BackendFrameContextT]:
        """Active frame-local data for this context."""
        return self.frame_resources.active_slot

    @abstractmethod
    def create_backend_frame_context(self, _slot_index: int) -> BackendFrameContextT:
        """Create backend-specific data attached to one reusable frame context."""

    @abstractmethod
    def destroy(self) -> None:
        """Destroys the graphical context."""

    def frame_begin(self) -> None:
        """Begin a frame draw/update scope for this context."""
        self.frame_resources.frame_begin(self.frame_index)
        self._frame_active = True

    def frame_submit(self) -> None:
        """Submit queued GPU work for the current frame."""
        self.frame_resources.frame_submit()

    def create_frame_fence(self) -> Any | None:
        """Create a backend completion handle for submitted frame resources.

        The returned object is owned by ``FrameResourceManager`` unless the
        backend documents otherwise in ``delete_frame_fence``.
        """
        return None

    def poll_frame_fence(self, fence: Any) -> bool:
        """Return whether a submitted frame fence has completed."""
        return True

    def wait_frame_fence(self, fence: Any) -> None:
        """Waits until a submitted frame fence has completed.

        This is a blocking operation.
        """

    def delete_frame_fence(self, fence: Any) -> None:
        """Release a completion handle returned by :meth:`create_frame_fence`."""

    def frame_end(self) -> None:
        """Finalize the current frame, present, and advance frame index."""
        if not self._frame_active:
            return
        try:
            self.frame_submit()
            self.present()
            self.frame_index += 1
        finally:
            self._frame_active = False

    @abstractmethod
    def present(self) -> None:
        """Present rendered frame output."""

    @abstractmethod
    def clear(self) -> None:
        """Clears the framebuffer."""


class UnavailableContextError(GraphicsBackendError):
    """An exception that occurs when a backend context is not yet available."""
    def __init__(self) -> None:  # noqa: D107
        super().__init__(
            "A Rendering Context has not yet been created, or has already been deleted. Please ensure "
            "a context exists (create a Window) before attempting to create objects with GPU resources.",
        )


class NullContext:
    """Used to represent a Null state before a Context is created.

    If a user tries to perform any context specific operations _before_
    any Context has been created (or if it has been deleted), this object
    will intercept any access attempts and provide a useful exception.

    .. note:: To simplify comparisons, this class behaves similarly to a NoneType.
              It is NOT actually a NoneType however, so it should not be used as such
              (for example as an argument to a ctypes call).
    """

    @staticmethod
    def _raise_no_context() -> NoReturn:
        raise UnavailableContextError

    def __getattribute__(self, item):
        object.__getattribute__(self, "_raise_no_context")()

    def __enter__(self) -> NoReturn:
        object.__getattribute__(self, "_raise_no_context")()

    def __exit__(self, *_args: Any) -> None:
        return None

    def __eq__(self, other):
        return other is None

    def __bool__(self):
        return False

    def __hash__(self):
        return hash(None)


class VerifiedGraphicsConfig:
    """Determines if this config is complete and able to create a Window resource context.

    This is kept separate as the original user config may be re-used for multiple windows.

    .. note:: Keep any non-backend attributes as private. Create a property if public use is wanted.
    """

    def __init__(self, window: Window, config: GraphicsConfig) -> None:  # noqa: D107
        self._window = window
        self._config = config

    @property
    def is_finalized(self) -> bool:
        return True

    @abstractmethod
    def apply_format(self) -> None:
        """Commit the selected finalized attributes to the backend."""

    @property
    def attributes(self) -> dict[str, Any]:
        return {attrib: value for attrib, value in self.__dict__.items() if attrib[0] != '_'}

    def __repr__(self) -> str:
        return f"Attributes({self.attributes})"


class GraphicsConfig:
    """User requested values for the graphics configuration.

    A Config stores the user preferences for attributes such as the size of the colour and depth buffers,
    double buffering, multisampling, anisotropic, and so on.

    Different platform and architectures only support certain attributes. Attributes are defined within the
    class attribute namespace.

    These are merely a suggestion to the backends; there is no guarantee that a platform or driver accept the
    combination of attributes together.
    """

    def __init__(self, **kwargs: float | str) -> None:
        """Create a template config with the given attributes.

        Specify attributes as keyword arguments, for example::

            template = UserConfig(double_buffer=True)

        """
        self._finalized_config: VerifiedGraphicsConfig | None = None
        self._attributes = {}
        self._user_set_attributes = set()

        # Fetch the type hints dynamically from the class itself
        attribute_types = get_type_hints(self)

        # Initialize defaults or user-specified values
        for attr_name, attr_type in attribute_types.items():
            default_value = getattr(self, attr_name, None)

            # Set user-provided values or fallback to default
            if attr_name in kwargs:
                user_value = kwargs[attr_name]
                if isinstance(user_value, attr_type):
                    self._attributes[attr_name] = user_value
                    self._user_set_attributes.add(attr_name)  # Track user-set attributes
                    setattr(self, attr_name, user_value)
                else:
                    msg = f"Incorrect type for {attr_name}. Expected {attr_type.__name__}."
                    raise TypeError(msg)
            else:
                self._attributes[attr_name] = default_value
                setattr(self, attr_name, default_value)

    @abstractmethod
    def match(self, window: Window) -> VerifiedGraphicsConfig | None:
        """Matches this config to a given platform Window.

        The subclassed platform config must handle setting the FinalizedAttributes values.

        Returns:
             `True` or `False` the given window matches the OpenGL configuration.
        """

    @property
    def is_finalized(self) -> bool:
        return False

    @classmethod
    def available_attributes(cls) -> dict:
        """Return a list of available attribute names, types, and their default values."""
        attributes = {}
        attribute_types = get_type_hints(cls)
        for attr_name, attr_type in attribute_types.items():
            default_value = getattr(cls, attr_name, None)
            attributes[attr_name] = (attr_type, default_value)
        return attributes

    @property
    def user_set_attributes(self) -> set[str]:
        """Return a set of attribute names that were explicitly set by the user."""
        return self._user_set_attributes


class GraphicsResource(Protocol):  # noqa: D101
    def delete(self) -> None:
        ...


class ResourceManagement:
    """A manager to handle the freeing of resources for an API.

    In some graphical API's, the order in which you free resources can be very specific.
    """
    managers: list[GraphicsResource]

    def __init__(self) -> None:  # noqa: D107
        self._func = None
        self.managers = []
        atexit.register(self.on_exit_cleanup)

    def set_pre_cleanup_func(self, func: Callable) -> None:
        """Register a function to be called before cleanup.

        Some API's may need to enforce a sync before cleanup.
        """
        self._func = func

    def register_manager(self, resource: GraphicsResource) -> None:
        """A manager handles multiple resources, these will be called in reverse order on cleanup."""
        self.managers.append(resource)

    def register_resource(self, resource: GraphicsResource) -> None:
        """Compatibility no-op.

        Vulkan now owns explicit shutdown ordering internally, so this
        registration hook is no longer required for backend cleanup.
        """
        _ = resource

    def on_exit_cleanup(self) -> None:
        """Cleans up all graphical resources that have been registered on application exit."""
        self.cleanup_all()

    def cleanup_all(self) -> None:
        """Cleans up all graphical resources that have been registered.

        Managers are called last, and in reverse order of registered.
        """
        if self._func:
            self._func()

        for resource in reversed(self.managers):
            resource.delete()
