from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, Protocol, TypeVar


class FrameStateWithUploadVersion(Protocol):
    uploaded_version: int


FrameStateT = TypeVar("FrameStateT", bound=FrameStateWithUploadVersion)


class FrameLocalResource(ABC, Generic[FrameStateT]):
    """Base helper for objects with one state record per frame in flight."""

    frames_in_flight: int
    current_frame: int
    version: int
    frame_states: list[FrameStateT]

    def __init__(self, frames_in_flight: int) -> None:
        self.frames_in_flight = max(1, int(frames_in_flight))
        self.current_frame = 0
        self.version = 0
        self.frame_states = self._create_frame_states()
        if len(self.frame_states) != self.frames_in_flight:
            msg = (
                f"{self.__class__.__name__} must create exactly {self.frames_in_flight} "
                f"frame state objects, got {len(self.frame_states)}."
            )
            raise RuntimeError(msg)
        self.invalidate_uploads()

    @abstractmethod
    def _create_frame_states(self) -> list[FrameStateT]:
        """Create one state object per frame in flight."""
        raise NotImplementedError

    def _normalize_frame_index(self, frame_index: int) -> int:
        return int(frame_index) % self.frames_in_flight

    def set_current_frame(self, frame_index: int) -> int:
        resolved = self._normalize_frame_index(frame_index)
        self.current_frame = resolved
        return resolved

    def mark_data_updated(self) -> None:
        self.version += 1

    def upload_current_if_needed(self) -> bool:
        return self.upload_if_needed(self.current_frame)

    def upload_all_if_needed(self) -> bool:
        uploaded = False
        for frame_index in range(self.frames_in_flight):
            uploaded = self.upload_if_needed(frame_index) or uploaded
        return uploaded

    def get_frame_state(self, frame_index: int) -> FrameStateT:
        return self.frame_states[self._normalize_frame_index(frame_index)]

    def is_uploaded_for_frame(self, frame_index: int) -> bool:
        resolved = self._normalize_frame_index(frame_index)
        return self.get_frame_state(resolved).uploaded_version == self.version

    def upload_if_needed(self, frame_index: int) -> bool:
        resolved = self._normalize_frame_index(frame_index)
        frame_state = self.get_frame_state(resolved)
        if frame_state.uploaded_version == self.version:
            return False
        if not self._upload_frame(resolved):
            return False
        frame_state.uploaded_version = self.version
        return True

    @abstractmethod
    def _upload_frame(self, frame_index: int) -> bool:
        """Upload host-side resource data into one frame-local GPU resource."""
        raise NotImplementedError

    def invalidate_uploads(self) -> None:
        for state in self.frame_states:
            state.uploaded_version = -1
