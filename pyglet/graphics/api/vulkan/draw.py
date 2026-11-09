from __future__ import annotations


import contextlib
import weakref
from dataclasses import dataclass, field
from typing import Callable, Generator, Sequence, Any, TYPE_CHECKING, ClassVar
from ctypes import byref, c_float
import pyglet
from pyglet.graphics.vertexdomain import VertexDomain
from pyglet.libs.shared.vulkan_lib.vulkan_core import VkRenderPassBeginInfo, \
    VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO, VkOffset2D, VkRect2D, VkClearColorValue, VkClearValue, \
    VkClearAttachment, VkClearRect, VK_SUBPASS_CONTENTS_INLINE, VK_IMAGE_ASPECT_COLOR_BIT

from pyglet.graphics.draw import Batch, _BucketBatch, BatchDrawOptions, DrawPass, DrawContext, Group
from pyglet.graphics.shader import Attribute
from pyglet.graphics.api.vulkan.state import DescriptorResourceState
from pyglet.graphics.api.vulkan import vertexdomain, c_array_list, DeviceFunc

_debug_graphics_batch = pyglet.options.debug_graphics_batch

if TYPE_CHECKING:
    from pyglet.graphics.api.vulkan.buffer import VulkanUniformBufferObject
    from pyglet.graphics.api.vulkan.vertexdomain import VertexList, IndexedVertexList
    from pyglet.graphics.api.vulkan.pipeline import GraphicsPipeline
    from pyglet.graphics.api.vulkan.descriptor import (
        DescriptorSetObject,
        DescriptorSetLayoutsKey,
        DescriptorResourceKey,
        DescriptorSetBindingGroup,
    )
    from pyglet.enums import GeometryMode
    from pyglet.graphics.shader import ShaderProgram
    from pyglet.graphics.vertexdomain import VertexGroupBucket
    from pyglet.graphics.api.vulkan.instance import VulkanSurfaceContext, VulkanFrameContext



# Default Shader source:

_vertex_source: str = """#version 330 core
    in vec3 position;
    in vec4 colors;
    in vec3 tex_coords;
    out vec4 vertex_colors;
    out vec3 texture_coords;

    uniform WindowBlock
    {
        mat4 projection;
        mat4 view;
    } window;

    void main()
    {
        gl_Position = window.projection * window.view * vec4(position, 1.0);

        vertex_colors = colors;
        texture_coords = tex_coords;
    }
"""

_fragment_source: str = """#version 330 core
    in vec4 vertex_colors;
    in vec3 texture_coords;
    out vec4 final_colors;

    uniform sampler2D our_texture;

    void main()
    {
        final_colors = texture(our_texture, texture_coords.xy) + vertex_colors;
    }
"""

# Default blit source
_blit_vertex_source: str = """#version 330 core
    in vec3 position;
    in vec3 tex_coords;
    out vec3 texture_coords;

    uniform WindowBlock
    {
        mat4 projection;
        mat4 view;
    } window;

    void main()
    {
        gl_Position = window.projection * window.view * vec4(position, 1.0);

        texture_coords = tex_coords;
    }
"""

_blit_fragment_source: str = """#version 330 core
    in vec3 texture_coords;
    out vec4 final_colors;

    uniform sampler2D our_texture;

    void main()
    {
        final_colors = texture(our_texture, texture_coords.xy);
    }
"""


def get_default_batch() -> Batch:
    """Batch used globally for objects that have no Batch specified."""
    return pyglet.graphics.api.core.get_default_batch()


def get_default_blit_shader() -> ShaderProgram:
    """A default basic shader for blitting, provides no blending."""
    return pyglet.graphics.api.core.get_cached_shader(
        "default_blit",
        (_blit_vertex_source, 'vertex'),
        (_blit_fragment_source, 'fragment'),
    )




_domain_class_map: dict[tuple[bool, bool], type[vertexdomain.VulkanVertexDomain]] = {
    # Indexed, Instanced : Domain
    (False, False): vertexdomain.VulkanVertexDomain,
    (True, False): vertexdomain.VulkanIndexedVertexDomain,
    (False, True): vertexdomain.VulkanInstancedVertexDomain,
    (True, True): vertexdomain.VulkanInstancedIndexedVertexDomain,
}

# States that are used for a pipeline match.
_pipeline_hash = [
    "shader_program", "blend_func", "blend_op", "depth_test", "depth_write", "stencil_func",
    "stencil_op", "polygon_mode", "viewport",
]

# These states require the pipeline to be in dynamic mode, but become callable.
_dynamic_state = [
    "blend_func", "blend_op", "scissor", "viewport",
]


def get_group_resource_states(group: Group):
    states = getattr(group, "states", getattr(group, "_states", ()))
    return [state for state in states if isinstance(state, DescriptorResourceState)]


def get_window_camera_ubo(window) -> VulkanUniformBufferObject | None:
    matrices = getattr(window, "_matrices", None)
    window_ubo = getattr(matrices, "ubo", None)
    if window_ubo is not None:
        return window_ubo

    camera = getattr(window, "camera", None)
    storage = getattr(camera, "view_storage", None)
    return getattr(storage, "ubo", getattr(storage, "_ubo", None))


@dataclass
class VulkanBackendDrawContext:
    """Temporary Vulkan draw data."""

    frame_ctx: VulkanFrameContext
    groups: list[Group] = field(default_factory=list)
    renderpass: Any = None
    extent: Any = None

    @property
    def command_buffer(self):
        return self.frame_ctx.command_buffer


class VulkanBatch(_BucketBatch):
    """Manage a collection of drawables for batched rendering.

    Many drawable pyglet objects accept an optional `Batch` argument in their
    constructors. By giving a `Batch` to multiple objects, you can tell pyglet
    that you expect to draw all of these objects at once, so it can optimise its
    use of OpenGL. Hence, drawing a `Batch` is often much faster than drawing
    each contained drawable separately.

    The following example creates a batch, adds two sprites to the batch, and
    then draws the entire batch::

        batch = pyglet.graphics.Batch()
        car = pyglet.sprite.Sprite(car_image, batch=batch)
        boat = pyglet.sprite.Sprite(boat_image, batch=batch)

        def on_draw():
            batch.draw()

    While any drawables can be added to a `Batch`, only those with the same
    draw mode, shader program, and group can be optimised together.

    Internally, a `Batch` manages a set of VertexDomains along with
    information about how the domains are to be drawn. To implement batching on
    a custom drawable, get your vertex domains from the given batch instead of
    setting them up yourself.
    """
    _context: VulkanSurfaceContext
    _draw_list: list[Callable]
    top_groups: list[Group]
    group_children: dict[Group, list[Group]]
    group_map: dict[Group, dict[tuple, VertexDomain]]
    pipeline_mgr: Any
    descriptor_mgr: Any
    _live_batches: ClassVar[weakref.WeakSet] = weakref.WeakSet()
    _domain_class_map: dict[tuple[bool, bool], type[VertexDomain]] = _domain_class_map

    def __init__(self, context: VulkanSurfaceContext | None = None, initial_count: int = 32) -> None:
        """Create a graphics batch."""
        # Mapping to find domain.
        # group -> (attributes, mode, indexed) -> domain
        resolved_context = context or pyglet.graphics.api.core.current_window
        super().__init__(resolved_context, initial_count)
        self.pipelines = []

        self._devices = pyglet.graphics.api.core.devices
        self.pipeline_mgr = pyglet.graphics.api.core.pipeline_mgr
        self.descriptor_mgr = pyglet.graphics.api.core.descriptor_mgr
        self._instance_count = 0
        self._live_batches.add(self)

    def _create_backend_draw_context(self) -> VulkanBackendDrawContext:
        return VulkanBackendDrawContext(self._context.frame_context.backend_ctx)

    def _create_draw_context(
        self,
        draw_pass: BatchDrawOptions | DrawPass,
    ) -> DrawContext[VulkanSurfaceContext, VulkanBackendDrawContext]:
        return DrawContext(
            surface_ctx=self._context,
            backend_ctx=self._create_backend_draw_context(),
            frame_context=self._context.frame_context,
            draw_pass=draw_pass.resolve(self._context),
            renderer=self._context.renderer,
        )

    def delete(self) -> None:
        type(self)._live_batches.discard(self)
        for domain in self._domain_registry.values():
            delete = getattr(domain, "delete", None)
            if delete is not None:
                delete()
        self._domain_registry.clear()
        for group in tuple(self.group_map):
            group._assigned_batches.discard(self)  # noqa: SLF001
        self.group_map.clear()
        self.group_children.clear()
        self.top_groups.clear()
        self._draw_list.clear()
        self._pass_draw_lists.clear()
        self._pass_registrations.clear()
        self._passes.clear()

    def __del__(self) -> None:
        try:
            self.delete()
        except Exception:
            pass

    @classmethod
    def _delete_tracked_instances(cls) -> None:
        for batch in tuple(cls._live_batches):
            batch.delete()

    @classmethod
    def _delete_context_instances(cls, context: VulkanSurfaceContext) -> None:
        for batch in tuple(cls._live_batches):
            if batch._context is context:
                batch.delete()

    def invalidate(self) -> None:
        """Force the batch to update the draw list.

        This method can be used to force the batch to re-compute the draw list
        when the ordering of groups has changed.

        .. versionadded:: 1.2
        """
        self._draw_list_dirty = True

    update_shader = Batch.update_shader

    migrate = Batch.migrate

    get_domain = Batch.get_domain

    def _cleanup_group(self, group: Group) -> None:
        del self.group_map[group]
        group._assigned_batches.remove(self)  # noqa: SLF001
        if group.parent:
            self.group_children[group.parent].remove(group)
        try:
            del self.group_children[group]
        except KeyError:
            pass
        try:
            self.top_groups.remove(group)
        except ValueError:
            pass

        for domain in self._domain_registry.values():
            if domain.has_bucket(group):
                del domain._vertex_buckets[group]  # noqa: SLF001

    def _add_group(self, group: Group) -> None:
        group._ensure_state_cache()
        self.group_map[group] = {}
        if group.parent is None:
            self.top_groups.append(group)
        else:
            if group.parent not in self.group_map:
                self._add_group(group.parent)
            if group.parent not in self.group_children:
                self.group_children[group.parent] = []
            self.group_children[group.parent].append(group)

        group._assigned_batches.add(self)  # noqa: SLF001
        self._draw_list_dirty = True

    def _get_program_uniform_bindings(
        self,
        pipeline: GraphicsPipeline,
        draw_ctx: DrawContext,
    ) -> list[tuple[VulkanUniformBufferObject, int, int]]:
        program = pipeline.program
        camera = draw_ctx.active_camera if draw_ctx.camera_stack else None
        storage = camera.view.storage if camera is not None else None
        window_ubo = getattr(storage, "ubo", getattr(storage, "_ubo", None))
        uniform_bindings: list[tuple[VulkanUniformBufferObject, int, int]] = []

        for block_name, uniform_block in program.uniform_blocks.items():
            if block_name == "WindowBlock":
                if window_ubo is not None:
                    uniform_bindings.append((window_ubo, uniform_block.binding, uniform_block.index))
                continue

            ubo = program.ubo.get(block_name)
            if ubo is not None:
                uniform_bindings.append((ubo, uniform_block.binding, uniform_block.index))

        return uniform_bindings

    @staticmethod
    def _upload_uniform_bindings_if_needed(
        uniform_bindings: Sequence[tuple[VulkanUniformBufferObject, int, int]],
        frame_index: int,
    ) -> None:
        for ubo, _, _ in uniform_bindings:
            ubo.upload_if_needed(frame_index)

    @staticmethod
    def _build_descriptor_binding_group(
        descriptor_sets: Sequence[DescriptorSetObject],
        frame_index: int,
    ) -> DescriptorSetBindingGroup | None:
        from pyglet.graphics.api.vulkan.descriptor import DescriptorSetBindingGroup

        merged_sets: list[Any] = []
        device = None
        for descriptor_set in descriptor_sets:
            device = descriptor_set.device
            merged_sets.extend(descriptor_set.get_frame_descriptor_sets(frame_index))

        if not merged_sets or device is None:
            return None

        return DescriptorSetBindingGroup.from_descriptor_sets(device, merged_sets)

    @staticmethod
    def _clear_color_attachment(
        command_buffer,
        render_area: VkRect2D,
        clear_value: VkClearValue,
    ) -> None:
        clear_attachment = VkClearAttachment(
            aspectMask=VK_IMAGE_ASPECT_COLOR_BIT,
            colorAttachment=0,
            clearValue=clear_value,
        )
        clear_rect = VkClearRect(
            rect=render_area,
            baseArrayLayer=0,
            layerCount=1,
        )
        clear_attachments = c_array_list([clear_attachment], VkClearAttachment)
        clear_rects = c_array_list([clear_rect], VkClearRect)
        DeviceFunc.vkCmdClearAttachments(command_buffer, 1, clear_attachments, 1, clear_rects)


    def _enter_group(self, group: Group, ctx: DrawContext) -> None:
        ctx.backend_ctx.groups.append(group)
        group.set_state_all(ctx)

    def _leave_group(self, group: Group, ctx: DrawContext) -> None:
        try:
            group.unset_state_all(ctx)
        finally:
            ctx.backend_ctx.groups.pop()

    def _draw_domain(self, ctx: DrawContext, domain: VertexDomain, mode: GeometryMode,
                     bucket: VertexGroupBucket | None = None,
                     vertex_list: VertexList | IndexedVertexList | None = None) -> None:
        from types import SimpleNamespace
        states = {}
        for group in ctx.backend_ctx.groups:
            states.update(group._state_names)
        program = states["ShaderProgramState"].program
        depth_range = states.get("DepthRangeState")
        ctx.renderer.set_depth_range(depth_range.near if depth_range else 0.0,
                                     depth_range.far if depth_range else 1.0)
        ctx.active_shader_program = program
        domain.get_vertex_input_binding(program)
        pipeline = self.pipeline_mgr.get_pipeline_from_group(
            SimpleNamespace(_state_names=states), ctx.backend_ctx.renderpass, mode,
            int(ctx.backend_ctx.extent.width), int(ctx.backend_ctx.extent.height), domain,
        )
        command_buffer = ctx.backend_ctx.command_buffer
        pipeline.bind(command_buffer)
        uniform_bindings = self._get_program_uniform_bindings(pipeline, ctx)
        # Explicit group UBO bindings override camera/program defaults.
        for state in states.values():
            if type(state).__name__ == "UniformBufferState":
                region = state.region
                region.bind(binding_index=state.binding_index)
                binding = state.binding_index if state.binding_index is not None else region._ubo.binding
                uniform_bindings = [entry for entry in uniform_bindings if entry[1:] != (binding, 0)]
                uniform_bindings.append((region._ubo, binding, 0))
        resources = []
        for state in states.values():
            if isinstance(state, DescriptorResourceState):
                resources.append(state)
            elif type(state).__name__ == "MultiTextureSamplerState":
                resources.extend(state.textures)
        frame_index = self._context.active_frame
        self._upload_uniform_bindings_if_needed(uniform_bindings, frame_index)
        layouts = pipeline.descriptor_set_layouts_info
        if layouts.layouts:
            descriptors = self.descriptor_mgr.get_descriptor_sets(
                layouts, resources, uniform_bindings=uniform_bindings,
                frame_index=frame_index, owner=self._context,
            )
            binding = self._build_descriptor_binding_group((descriptors,), frame_index)
            binding.bind(command_buffer, pipeline.pipeline_layout)
        # Push-constant structures are shared by programs; apply the effective
        # uniforms again for every draw, including after leaving a child scope.
        uniforms = states.get("ShaderUniformState")
        if uniforms is not None:
            uniforms.set_state(ctx)
        pipeline.push_constants(command_buffer, 0)
        if vertex_list is None:
            domain.draw_buckets(command_buffer, [bucket])
        else:
            domain._draw_subset_command(command_buffer, mode, vertex_list)

    def _compile_draw_list(self, draw_list: list) -> list[Callable]:
        from functools import partial
        calls = []
        for domain, mode, group in draw_list:
            if domain is None:
                calls.append(partial(self._enter_group if mode == "set" else self._leave_group, group))
            else:
                bucket = domain.get_drawable_bucket(group)
                calls.append(lambda ctx, domain=domain, mode=mode, bucket=bucket:
                             self._draw_domain(ctx, domain, mode, bucket))
        return calls

    def _compile_pass_draw_list(self, registrations: list) -> list[Callable]:
        from functools import partial
        return [partial(self._draw_registration, registration) for registration in registrations]

    def _draw_registration(self, registration: Any, ctx: DrawContext) -> None:
        chain = []
        group = registration.group
        while group is not None:
            if not group.visible:
                return
            chain.append(group)
            group = group.parent
        for group in reversed(chain):
            self._enter_group(group, ctx)
        try:
            self._draw_domain(ctx, registration.domain, registration.mode, registration.bucket)
        finally:
            for group in chain:
                self._leave_group(group, ctx)

    def _record_pass(self, draw_pass: BatchDrawOptions | DrawPass, calls: Sequence[Callable]) -> None:
        from pyglet.libs.shared.vulkan_lib.vulkan_core import VkExtent2D
        ctx = self._create_draw_context(draw_pass)
        frame = self._context.frame_context.backend_ctx
        clear_claimed = frame.clear_claimed
        command, should_clear = frame.begin_primary_command_buffer()
        target = ctx.draw_pass.framebuffer
        if target is None:
            renderpass = self._context.renderpass
            framebuffer = self._context.swapchain.framebuffers[self._context.frame_sync.current_image_index()]
            extent = self._context.swapchain.extent
        else:
            if target._context is not self._context:
                raise ValueError("DrawPass framebuffer belongs to another context.")
            target.finalize()
            renderpass, framebuffer = target._render_pass, target.handle
            extent = VkExtent2D(target.width, target.height)
            should_clear = False
            frame.clear_claimed = clear_claimed
        target_key = framebuffer.value
        if target_key in frame.rendered_targets:
            renderpass = renderpass.preserving_pass()
            from pyglet.libs.shared.vulkan_lib import vulkan_core as vk
            barrier = vk.VkMemoryBarrier(
                sType=vk.VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                srcAccessMask=vk.VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | vk.VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                dstAccessMask=vk.VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | vk.VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT
                              | vk.VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | vk.VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
            )
            stages = (vk.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT
                      | vk.VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | vk.VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT)
            DeviceFunc.vkCmdPipelineBarrier(command.command_buffer, stages, stages, 0,
                                           1, byref(barrier), 0, None, 0, None)
        frame.rendered_targets.add(target_key)
        ctx.backend_ctx.renderpass = renderpass
        ctx.backend_ctx.extent = extent
        self._context.renderer.render_extent = extent
        area = VkRect2D(offset=VkOffset2D(0, 0), extent=extent)
        begin = VkRenderPassBeginInfo(
            sType=VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
            renderPass=renderpass.vk_renderpass, framebuffer=framebuffer, renderArea=area,
        )
        from pyglet.libs.shared.vulkan_lib.vulkan_core import VkClearDepthStencilValue
        clear_values = [VkClearValue(color=VkClearColorValue(float32=(c_float * 4)(*ctx.draw_pass.clear_color)))
                        for _ in renderpass.color_attachments]
        if renderpass.depth_attachment is not None:
            clear_values.append(VkClearValue(depthStencil=VkClearDepthStencilValue(1.0, 0)))
        begin.clearValueCount = len(clear_values)
        begin.pClearValues = c_array_list(clear_values, VkClearValue)
        DeviceFunc.vkCmdBeginRenderPass(command.command_buffer, byref(begin), VK_SUBPASS_CONTENTS_INLINE)
        try:
            if should_clear:
                color = VkClearValue(color=VkClearColorValue(float32=(c_float * 4)(*ctx.draw_pass.clear_color)))
                self._clear_color_attachment(command.command_buffer, area, color)
            # Establish dynamic defaults even for camera=None.
            self._context.renderer.set_viewport(0, 0, int(extent.width), int(extent.height))
            self._context.renderer.set_scissor(None)
            ctx.begin()
            if not ctx.camera_stack:
                ctx.apply_viewport()
                ctx.apply_scissor()
            for call in calls:
                call(ctx)
        finally:
            DeviceFunc.vkCmdEndRenderPass(command.command_buffer)

    def _draw_default_pass(self) -> None:
        self._record_pass(BatchDrawOptions(), self._draw_list)

    def _draw_registered_pass(self, draw_pass: DrawPass) -> None:
        self._record_pass(draw_pass, self._pass_draw_lists[draw_pass])

    @contextlib.contextmanager
    def draw_with_options(self) -> Generator[BatchDrawOptions, Any, None]:
        options = BatchDrawOptions()
        try:
            yield options
        finally:
            self._update_draw_list()
            self._record_pass(options, self._draw_list)
            self.delete_empty_domains()

    def draw_subset(self, vertex_lists: Sequence[VertexList | IndexedVertexList]) -> None:
        self._draw_vertex_lists(vertex_lists)

    def _draw_vertex_lists(self, vertex_lists: Sequence[VertexList | IndexedVertexList],
                           mode: GeometryMode | None = None) -> None:
        from functools import partial
        calls = []
        def visit(group: Group) -> None:
            calls.append(partial(self._enter_group, group))
            for vertex_list in vertex_lists:
                if vertex_list.domain.batch is not self:
                    raise ValueError("VertexList belongs to another Batch.")
                if vertex_list.group is group:
                    calls.append(lambda ctx, vertex_list=vertex_list:
                                 self._draw_domain(ctx, vertex_list.domain, mode or vertex_list.mode,
                                                   vertex_list=vertex_list))
            for child in sorted(self.group_children.get(group, ())):
                if child.visible:
                    visit(child)
            calls.append(partial(self._leave_group, group))
        for group in sorted(self.top_groups):
            if group.visible:
                visit(group)
        self._record_pass(BatchDrawOptions(), calls)
