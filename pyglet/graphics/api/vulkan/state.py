from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Generator, TYPE_CHECKING

from pyglet.enums import BlendFactor, BlendOp, CompareOp
from pyglet.graphics.api.vulkan import DeviceFunc
from pyglet.libs.shared.vulkan_lib.vulkan_core import VkOffset2D, VkRect2D, VkExtent2D

from pyglet.graphics.state import State

if TYPE_CHECKING:
    from pyglet.graphics import Group, Texture
    from pyglet.customtypes import ScissorProtocol
    from pyglet.image.base import TextureBase
    from pyglet.graphics.api.vulkan.shader import VulkanShaderProgram
    from pyglet.graphics.api.vulkan.descriptor import DescriptorSetObject
    from pyglet.graphics.api.vulkan.texture import VulkanTexture
    from pyglet.graphics.api.vulkan.instance import VulkanSurfaceContext


class DescriptorResourceState(State):
    """Vulkan-only state that writes resources into descriptor sets."""
    group_hash: bool = True

    def write_descriptor(self, current_desc: DescriptorSetObject) -> None:
        raise NotImplementedError


@dataclass(frozen=True)
class TextureState(DescriptorResourceState):  # noqa: D101
    texture: VulkanTexture
    binding: int = 0
    set_id: int = 0

    @classmethod
    def from_texture(cls, texture: VulkanTexture, binding: int, set_id: int) -> TextureState:
        return cls(texture,
                   binding=binding,
                   set_id=set_id)

    def write_descriptor(self, current_desc: DescriptorSetObject) -> None:
        current_desc.bind_texture(self.texture, self.binding, self.set_id)


@dataclass(frozen=True)
class ShaderProgramState(State):
    program: VulkanShaderProgram

    sets_state: bool = False
    unsets_state: bool = False


@dataclass(frozen=True)
class RenderPassState(State):
    renderpass: Any  # Renderpass for Vulkan.


@dataclass(frozen=True)
class RenderAreaState(State):
    width: int
    height: int


@dataclass(frozen=True)
class ScissorState(State):
    spo: ScissorProtocol

    sets_state: bool = True

    def set_state(self, ctx: VulkanSurfaceContext) -> None:
        cb = ctx.frame_sync.get_current_command_buffer(ctx.default_cb_id).command_buffer
        rect = VkRect2D(
            offset=VkOffset2D(x=int(self.spo.x), y=int(self.spo.y)),
            extent=VkExtent2D(max(0, int(self.spo.width)), max(0, int(self.spo.height))),
        )
        rects = (VkRect2D * 1)(rect)
        DeviceFunc.vkCmdSetScissor(cb, 0, 1, rects)

@dataclass(frozen=True)
class BlendStateEnable(State):
    ...


@dataclass(frozen=True)
class BlendState(State):
    src: BlendFactor
    dst: BlendFactor
    op: BlendOp = BlendOp.ADD

    #sets_state: bool = True
    #dependents: bool = True



@dataclass(frozen=True)
class DepthTestState(State):
    func: int

@dataclass(frozen=True)
class DepthWriteState(State):
    flag: int

    sets_state: bool = False

@dataclass(frozen=True)
class StencilFuncState(State):
    func: Callable
    ref: int
    mask: int

@dataclass(frozen=True)
class StencilOpState(State):
    fail: int
    zfail: int
    zpass: int

@dataclass(frozen=True)
class PolygonModeState(State):
    face: int
    mode: int

@dataclass(frozen=True)
class ViewportState(State):
    x: float
    y: float
    width: float
    height: float

@dataclass(frozen=True)
class UniformBufferState(State):
    name: str
    binding: int

@dataclass(frozen=True)
class ShaderUniformState(State):
    """These are just push constants, since an equivalent uniform from OpenGL does not exist in Vulkan.

    Does not take multiple push constants into account yet.
    """
    program: VulkanShaderProgram
    data: dict[str, Any]

    sets_state: bool = True

    def set_state(self, ctx) -> None:
        pc = self.program.push_constants[0]
        for name, value in self.data.items():
            #self.program[name] = value
            setattr(pc.struct, name, value)

    def __hash__(self) -> int:
        return id(self)

    def __eq__(self, other: State) -> bool:
        return False



@dataclass(frozen=True)
class DepthTestStateEnable(State):
    sets_state: bool = False
    unsets_state: bool = False


@dataclass(frozen=True)
class DepthBufferComparison(State):
    func: CompareOp

    sets_state: bool = False
    parents: bool = False

    def generate_parent_states(self) -> Generator[State, None, None]:
        yield DepthTestStateEnable()
