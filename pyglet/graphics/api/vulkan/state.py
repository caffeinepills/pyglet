from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable, Generator, TYPE_CHECKING

from pyglet.enums import BlendFactor, BlendOp, CompareOp, CullFace, FrontFace, StencilOp
from pyglet.graphics.api.vulkan import DeviceFunc
from pyglet.libs.shared.vulkan_lib.vulkan_core import VkOffset2D, VkRect2D, VkExtent2D

from pyglet.graphics.state import State, ViewportProtocol, _BaseViewportState, _BaseScissorState

if TYPE_CHECKING:
    from pyglet.graphics.draw import DrawContext
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

    def write_descriptor(self, current_desc: DescriptorSetObject, frame_idx: int) -> None:
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

    def write_descriptor(self, current_desc: DescriptorSetObject, frame_idx: int) -> None:
        current_desc.update_sampled_texture_binding(self.texture, self.binding, self.set_id, frame_idx=frame_idx)


@dataclass(frozen=True)
class ShaderProgramState(State):
    program: VulkanShaderProgram

    sets_state: bool = False
    unsets_state: bool = False


@dataclass(frozen=True)
class MultiTextureSamplerState(State):
    program: VulkanShaderProgram
    textures: tuple[TextureState, ...]

    @classmethod
    def from_textures(cls, program: VulkanShaderProgram, textures: dict[str, Texture],
                      first_texture_unit: int = 0, set_id: int = 0) -> MultiTextureSamplerState:
        bindings = []
        for index, (name, texture) in enumerate(textures.items(), first_texture_unit):
            metadata = program.sampled_textures.get(name)
            bindings.append(TextureState(texture, metadata.binding if metadata else index,
                                         metadata.desc_set if metadata else set_id))
        return cls(program, tuple(bindings))

@dataclass(frozen=True)
class RenderPassState(State):
    renderpass: Any  # Renderpass for Vulkan.


@dataclass(frozen=True)
class RenderAreaState(State):
    width: int
    height: int


@dataclass(frozen=True, eq=False)
class ScissorState(_BaseScissorState):
    scissor: ScissorProtocol
    owned_by_camera: bool = False
    sets_state: bool = True
    unsets_state: bool = True
    enforced_state: bool = True

    def apply_to_backend(self, ctx: DrawContext) -> None:
        ctx.apply_scissor()

@dataclass(frozen=True)
class BlendStateEnable(State):
    ...


@dataclass(frozen=True)
class BlendState(State):
    src: BlendFactor
    dst: BlendFactor
    op: BlendOp = BlendOp.ADD
    src_alpha: BlendFactor | None = field(default=None, kw_only=True)
    dst_alpha: BlendFactor | None = field(default=None, kw_only=True)
    op_alpha: BlendOp | None = field(default=None, kw_only=True)

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
    func: CompareOp
    ref: int
    mask: int

@dataclass(frozen=True)
class StencilOpState(State):
    fail: StencilOp
    z_fail: StencilOp
    z_pass: StencilOp

@dataclass(frozen=True)
class PolygonModeState(State):
    face: int
    mode: int

@dataclass(frozen=True, eq=False)
class ViewportState(_BaseViewportState):
    """Mutable viewport state shared with the other backends."""

    viewport: ViewportProtocol

    sets_state: bool = True
    unsets_state: bool = True
    enforced_state: bool = True

    def apply_to_backend(self, ctx: DrawContext) -> None:
        ctx.apply_viewport()

@dataclass(frozen=True)
class UniformBufferState(State):
    region: Any
    binding_index: int | None = None

@dataclass(frozen=True)
class ShaderUniformState(State):
    """These are just push constants, since an equivalent uniform from OpenGL does not exist in Vulkan.

    Does not take multiple push constants into account yet.
    """
    program: VulkanShaderProgram
    data: dict[str, Any]

    sets_state: bool = True

    def set_state(self, ctx: DrawContext) -> None:
        for name, value in self.data.items():
            for pc in self.program.push_constants:
                if hasattr(pc.struct, name):
                    target = getattr(pc.struct, name)
                    if hasattr(target, "__len__"):
                        target[:] = value
                    else:
                        setattr(pc.struct, name, value)
                    break

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


@dataclass(frozen=True)
class DepthRangeState(State):
    near: float
    far: float

    sets_state: bool = False


@dataclass(frozen=True)
class PolygonOffsetState(State):
    factor: float
    units: float

    sets_state: bool = False


@dataclass(frozen=True)
class SampleCoverageState(State):
    value: float
    invert: bool = False

    sets_state: bool = False


@dataclass(frozen=True)
class CullFaceState(State):
    face: CullFace

    sets_state: bool = False


@dataclass(frozen=True)
class FrontFaceState(State):
    face: FrontFace

    sets_state: bool = False


@dataclass(frozen=True)
class StencilTestState(State):
    enabled: bool

    sets_state: bool = False


@dataclass(frozen=True)
class StencilMaskState(State):
    mask: int

    sets_state: bool = False


@dataclass(frozen=True)
class ColorMaskState(State):
    red: bool
    green: bool
    blue: bool
    alpha: bool

    sets_state: bool = False


@dataclass(frozen=True)
class StencilClearState(State):
    sets_state: bool = True

    def set_state(self, ctx: DrawContext) -> None:
        from pyglet.libs.shared.vulkan_lib import vulkan_core as vk
        depth = ctx.backend_ctx.renderpass.depth_attachment
        if depth is None or depth.fmt not in (vk.VK_FORMAT_D24_UNORM_S8_UINT, vk.VK_FORMAT_D32_SFLOAT_S8_UINT,
                                             vk.VK_FORMAT_D16_UNORM_S8_UINT):
            return
        attachment = vk.VkClearAttachment(aspectMask=vk.VK_IMAGE_ASPECT_STENCIL_BIT,
                                          clearValue=vk.VkClearValue(depthStencil=vk.VkClearDepthStencilValue(1.0, 0)))
        rect = vk.VkClearRect(rect=vk.VkRect2D(vk.VkOffset2D(0, 0), ctx.backend_ctx.extent),
                              baseArrayLayer=0, layerCount=1)
        from ctypes import byref
        DeviceFunc.vkCmdClearAttachments(ctx.backend_ctx.command_buffer, 1, byref(attachment), 1, byref(rect))
