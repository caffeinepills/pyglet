from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Generator, TYPE_CHECKING, Sequence

from pyglet.enums import BlendFactor, BlendOp

from pyglet.graphics.state import State

if TYPE_CHECKING:
    from pyglet.graphics import Group, Texture
    from pyglet.image.base import TextureBase
    from pyglet.graphics.api.vulkan.shader import VulkanShaderProgram
    from pyglet.graphics.api.vulkan.descriptor import DescriptorSetObject
    from pyglet.graphics.api.vulkan.texture import VulkanTexture



@dataclass(frozen=True)
class TextureState(State):  # noqa: D101
    texture: VulkanTexture
    binding: int = 0
    set_id: int = 0

    resolves_state: bool = True

    @classmethod
    def from_texture(cls, texture: VulkanTexture, binding: int, set_id: int) -> TextureState:
        return cls(texture,
                   binding=binding,
                   set_id=set_id)

    def resolve_state(self, current_desc: DescriptorSetObject) -> None:
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
    x: int
    y: int
    width: int
    height: int

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
    sets_state: bool = True
    unsets_state: bool = True

    def set_state(self, ctx: OpenGLSurfaceContext) -> None:
        ctx.glEnable(GL_DEPTH_TEST)

    def unset_state(self, ctx: OpenGLSurfaceContext) -> None:
        ctx.glDisable(GL_DEPTH_TEST)


@dataclass(frozen=True)
class DepthBufferComparison(State):
    func: CompareOp

    sets_state: bool = True
    parents: bool = True

    def generate_parent_states(self) -> Generator[State, None, None]:
        yield DepthTestStateEnable()

    def set_state(self, ctx: OpenGLSurfaceContext) -> None:
        ctx.glDepthFunc(compare_op_map[self.func])
