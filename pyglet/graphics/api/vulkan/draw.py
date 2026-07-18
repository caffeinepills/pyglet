from __future__ import annotations


import contextlib
import weakref
from dataclasses import dataclass
from typing import Callable, Generator, Sequence, Any, TYPE_CHECKING, ClassVar
from ctypes import byref, c_float
import pyglet
from pyglet.libs.shared.vulkan_lib.vulkan_core import VkRenderPassBeginInfo, \
    VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO, VkOffset2D, VkRect2D, VkClearColorValue, VkClearValue, \
    VkClearAttachment, VkClearRect, VK_SUBPASS_CONTENTS_INLINE, VK_IMAGE_ASPECT_COLOR_BIT

from pyglet.graphics.draw import _DomainKey, Batch, BatchDrawOptions, DrawContext, Group
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
    from pyglet.graphics import GeometryMode
    from pyglet.graphics.api.gl2.shader import ShaderProgram
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

# States that need to be called before the vertex draw.
_callable_states = {
    "shader_uniform": "push_constant",
    "shader_uniforms": "push_constants",
    "texture": "bind_texture",
    "textures": "bind_textures",
    "ubo": "bind_ubo",
    "ubos": "bind_ubos",
}


def get_group_resource_states(group: Group):
    states = getattr(group, "states", getattr(group, "_states", ()))
    return [state for state in states if isinstance(state, DescriptorResourceState)]


def get_window_camera_ubo(window) -> VulkanUniformBufferObject | None:
    matrices = getattr(window, "_matrices", None)
    window_ubo = getattr(matrices, "ubo", None)
    if window_ubo is not None:
        return window_ubo

    camera = getattr(window, "default_camera", None)
    storage = getattr(camera, "view_storage", None)
    return getattr(storage, "_ubo", None)


@dataclass
class VulkanBackendDrawContext:
    """Temporary Vulkan draw data."""

    frame_ctx: VulkanFrameContext

    @property
    def command_buffer(self):
        return self.frame_ctx.command_buffer


class VulkanBatch(Batch):
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
    _context: VulkanSurfaceContext
    top_groups: list[Group]
    group_children: dict[Group, list[Group]]
    group_map: dict[Group, dict[_DomainKey, vertexdomain.VertexDomain]]
    _live_batches: ClassVar[weakref.WeakSet] = weakref.WeakSet()

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
        draw_pass: BatchDrawOptions,
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
            domain.delete()
        self._domain_registry.clear()
        for group in tuple(self.group_map):
            group._assigned_batches.discard(self)  # noqa: SLF001
        self.group_map.clear()
        self.group_children.clear()
        self.top_groups.clear()
        self._draw_list.clear()

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

    def update_shader(self, vertex_list: VertexList | IndexedVertexList, mode: GeometryMode, group: Group,
                      program: ShaderProgram) -> bool:
        """Migrate a vertex list to another domain that has the specified shader attributes.

        The results are undefined if `mode` is not correct or if `vertex_list`
        does not belong to this batch (they are not checked and will not
        necessarily throw an exception immediately).

        Args:
            vertex_list:
                A vertex list currently belonging to this batch.
            mode:
                The current GL drawing mode of the vertex list.
            group:
                The new group to migrate to.
            program:
                The new shader program to migrate to.

        Returns:
            False if the domain's no longer match. The caller should handle this scenario.
        """
        attributes = self._normalized_shader_attributes(program, vertex_list.initial_attribs)

        if missing := [name for name in vertex_list.initial_attribs if name not in attributes]:
            if _debug_graphics_batch:
                import warnings

                warnings.warn(f"Missing required shader attributes for update: {missing}")
            return False

        drawable_attributes = {name: attributes[name] for name in vertex_list.initial_attribs}
        domain = self.get_domain(vertex_list.indexed, vertex_list.instanced, mode, group, drawable_attributes)

        # TODO: Allow migration if we can restore original vertices somehow. Much faster.
        # If the domain's don't match, we need to re-create the vertex list. Tell caller no match.
        if domain != vertex_list.domain:
            return False

        if vertex_list.group != group:
            vertex_list.update_group(group)
            self._draw_list_dirty = True

        return True

    def migrate(self, vertex_list: VertexList | IndexedVertexList, mode: GeometryMode, group: Group,
                batch: VulkanBatch) -> None:
        """Migrate a vertex list to another batch and/or group.

        `vertex_list` and `mode` together identify the vertex list to migrate.
        `group` and `batch` are new owners of the vertex list after migration.

        The results are undefined if `mode` is not correct or if `vertex_list`
        does not belong to this batch (they are not checked and will not
        necessarily throw an exception immediately).

        ``batch`` can remain unchanged if only a group change is desired.

        Args:
            vertex_list:
                A vertex list currently belonging to this batch.
            mode:
                The current GL drawing mode of the vertex list.
            group:
                The new group to migrate to.
            batch:
                The batch to migrate to (or the current batch).

        """
        attributes = vertex_list.domain.attribute_meta
        domain = batch.get_domain(vertex_list.indexed, vertex_list.instanced, mode, group, attributes)
        if domain != vertex_list.domain:
            vertex_list.migrate(domain, group)
        else:
            vertex_list.update_group(group)
            self._draw_list_dirty = True

    def get_domain(self, indexed: bool, instanced: bool, mode: GeometryMode, group: Group,
                   attributes: dict[str, Any]) -> (
            vertexdomain.VertexDomain | vertexdomain.IndexedVertexDomain | vertexdomain.InstancedVertexDomain |
            vertexdomain.InstancedIndexedVertexDomain):
        """Get, or create, the vertex domain corresponding to the given arguments.

        mode is the render mode such as GL_LINES or GL_TRIANGLES
        """
        # Group map is only used for group-tree lookup; domains are shared
        # across groups so transient text group replacement can reuse buffers.
        if group not in self.group_map:
            self._add_group(group)

        key = _DomainKey(indexed, instanced, mode, self._attributes_key(attributes))

        try:
            domain = self._domain_registry[key]
        except KeyError:
            # Create domain
            domain = _domain_class_map[(indexed, instanced)](self._context, self.initial_count, attributes)
            self._domain_registry[key] = domain
            self._draw_list_dirty = True

        return domain

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
    ) -> list[tuple[VulkanUniformBufferObject, int, int]]:
        program = pipeline.program
        window_ubo = get_window_camera_ubo(self._context.window)
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

    def _update_draw_list(self, draw_ctx: DrawContext) -> None:
        """Visit group tree in preorder and create a list of bound methods to call."""
        current_pipeline: GraphicsPipeline | None = None
        current_desc_sets: tuple[DescriptorSetObject, ...] = ()
        current_desc_binding: DescriptorSetBindingGroup | None = None
        current_descriptor_key: tuple[
            DescriptorSetLayoutsKey,
            tuple[DescriptorResourceKey, ...],
            int,
        ] | None = None

        frame_sync = self._context.frame_sync
        current_cb, should_clear = self._context.frame_context.backend_ctx.begin_primary_command_buffer()

        vk_command_buffer = current_cb.command_buffer

        render_area = VkRect2D(offset=VkOffset2D(x=0, y=0),
                               extent=self._context.swapchain.extent)
        color = VkClearColorValue(float32=(c_float * 4)(*draw_ctx.draw_pass.clear_color))
        clear_value = VkClearValue(color=color)

        render_pass_begin_create = VkRenderPassBeginInfo(
            sType=VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
            renderPass=self._context.renderpass.vk_renderpass,
            framebuffer=self._context.swapchain.framebuffers[frame_sync.current_image_index()],
            renderArea=render_area,
            clearValueCount=0,
            pClearValues=None,
        )

        DeviceFunc.vkCmdBeginRenderPass(vk_command_buffer, byref(render_pass_begin_create), VK_SUBPASS_CONTENTS_INLINE)
        if should_clear:
            self._clear_color_attachment(vk_command_buffer, render_area, clear_value)

        draw_ctx.begin()

        def set_default_scissor() -> None:
            rect = VkRect2D(
                offset=VkOffset2D(x=0, y=0),
                extent=self._context.swapchain.extent,
            )
            rects = c_array_list([rect], VkRect2D)
            DeviceFunc.vkCmdSetScissor(vk_command_buffer, 0, 1, rects)

        def visit(group: Group) -> list:
            nonlocal current_pipeline
            nonlocal current_desc_sets
            nonlocal current_desc_binding
            nonlocal current_descriptor_key

            draw_list = []

            drawable_domains = [
                (domain_key, domain, bucket)
                for domain_key, domain in self._domain_registry.items()
                if (bucket := domain.get_drawable_bucket(group))
            ]

            for domain_key, domain, bucket in drawable_domains:
                mode = domain_key.mode
                set_default_scissor()

                pipeline = self.pipeline_mgr.get_pipeline_from_group(group,
                                                                     self._context.renderpass,
                                                                     mode,
                                                                     self._context.window.width,
                                                                     self._context.window.height,
                                                                     domain)
                pipeline_changed = current_pipeline is not pipeline
                if pipeline_changed:
                    pipeline.bind(vk_command_buffer)

                    #self.descriptor_mgr.get
                    current_pipeline = pipeline

                group_resources = tuple(get_group_resource_states(group))
                uniform_bindings = self._get_program_uniform_bindings(pipeline)
                descriptor_set_layouts_info = pipeline.descriptor_set_layouts_info
                assert descriptor_set_layouts_info is not None
                frame_index = self._context.active_frame
                self._upload_uniform_bindings_if_needed(uniform_bindings, frame_index)
                descriptor_resources = self.descriptor_mgr.build_resource_keys(
                    group_resources,
                    uniform_bindings,
                    frame_index,
                )
                descriptor_key = (
                    descriptor_set_layouts_info.key,
                    descriptor_resources,
                    frame_index,
                )
                descriptor_changed = current_descriptor_key != descriptor_key

                if descriptor_changed:
                    if descriptor_set_layouts_info.layouts:
                        # Get or create the descriptor set
                        descriptor_set = self.descriptor_mgr.get_descriptor_sets(
                            descriptor_set_layouts_info,
                            list(group_resources),
                            uniform_bindings=uniform_bindings,
                            resources=descriptor_resources,
                            frame_index=frame_index,
                            owner=self._context,
                        )
                        current_desc_sets = (descriptor_set,)
                    else:
                        current_desc_sets = ()
                        current_desc_binding = None
                    current_descriptor_key = descriptor_key

                if current_desc_sets and (descriptor_changed or pipeline_changed):
                    current_desc_binding = self._build_descriptor_binding_group(
                        current_desc_sets,
                        frame_index,
                    )
                    if current_desc_binding is not None:
                        current_desc_binding.bind(vk_command_buffer, current_pipeline.pipeline_layout)

                for state in getattr(group, "states", getattr(group, "_states", ())):
                    if state.sets_state:
                        state.set_state(draw_ctx)

                pipeline.push_constants(vk_command_buffer, 0)

                domain.draw_buckets(vk_command_buffer, [bucket])

            # Sort and visit child groups of this group
            children = self.group_children.get(group)
            if children:
                children.sort()
                for child in list(children):
                    if child.visible:
                        draw_list.extend(visit(child))

            if children or drawable_domains:
                return [*draw_list]

            # Remove unused group from batch
            self._cleanup_group(group)

            return []

        self._draw_list = []

        self.top_groups.sort()
        #print("SELF TOP", self.top_groups)
        for top_group in list(self.top_groups):
            if top_group.visible:
                self._draw_list.extend(visit(top_group))

        # TODO: Make batches secondary command buffers, so they can be pre-recorded.
        #self._draw_list_dirty = True

        #print(self._draw_list)

        #print("DOMAIN!", self.group_map, self)

        #
        # if _debug_graphics_batch:
        #     self._dump_draw_list()

        DeviceFunc.vkCmdEndRenderPass(vk_command_buffer)

    def _dump_draw_list(self) -> None:
        def dump(group: Group, indent: str = '') -> None:
            print(indent, 'Begin group', group)
            for domain in self._domain_registry.values():
                bucket = domain.get_drawable_bucket(group)
                if bucket is None:
                    continue
                print(indent, '  ', domain)
                for start, size in bucket.merged_ranges:
                    print(indent, '    ', 'Region %d size %d:' % (start, size))
                    for key in domain.attrib_name_buffers:
                        buffer = domain.vertex_buffers.attrib_name_buffers[key]
                        print(indent, '      ', end=' ')
                        try:
                            region = buffer.get_region(start, size)
                            print(key, list(region))
                        except:  # noqa: E722
                            print(key, '(unmappable)')
            for child in self.group_children.get(group, ()):
                dump(child, indent + '  ')
            print(indent, 'End group', group)

        print(f'Draw list for {self!r}:')
        for group in self.top_groups:
            dump(group)

    def draw(self) -> None:
        """Draw the batch."""
        draw_ctx = self._create_draw_context(BatchDrawOptions())
        # Vulkan command buffers capture the current swapchain framebuffer.
        # Re-record each draw so the commands target the image acquired for this frame.
        self._update_draw_list(draw_ctx)

        #for func in self._draw_list:
            #print("FUNC!", func)
            #func()

    @contextlib.contextmanager
    def draw_with_options(self) -> Generator[BatchDrawOptions, Any, None]:
        draw_options = BatchDrawOptions()
        try:
            yield draw_options
        finally:
            draw_ctx = self._create_draw_context(draw_options)
            # Vulkan command buffers capture the current swapchain framebuffer.
            # Re-record each draw so the commands target the image acquired for this frame.
            self._update_draw_list(draw_ctx)

    def draw_subset(self, vertex_lists: Sequence[VertexList | IndexedVertexList]) -> None:
        """Draw only some vertex lists in the batch.

        The use of this method is highly discouraged, as it is quite
        inefficient.  Usually an application can be redesigned so that batches
        can always be drawn in their entirety, using `draw`.

        The given vertex lists must belong to this batch; behaviour is
        undefined if this condition is not met.

        Args:
            vertex_lists:
                Vertex lists to draw.

        """

        if not vertex_lists:
            return

        draw_ctx = self._create_draw_context(BatchDrawOptions())

        selected_by_domain: dict[Any, list[VertexList | IndexedVertexList]] = {}
        for vertex_list in vertex_lists:
            selected_by_domain.setdefault(vertex_list.domain, []).append(vertex_list)

        current_pipeline: GraphicsPipeline | None = None
        current_desc_sets: tuple[DescriptorSetObject, ...] = ()
        current_desc_binding: DescriptorSetBindingGroup | None = None
        current_descriptor_key: tuple[
            DescriptorSetLayoutsKey,
            tuple[DescriptorResourceKey, ...],
            int,
        ] | None = None

        frame_sync = self._context.frame_sync
        current_cb, should_clear = self._context.frame_context.backend_ctx.begin_primary_command_buffer()
        vk_command_buffer = current_cb.command_buffer

        render_area = VkRect2D(offset=VkOffset2D(x=0, y=0), extent=self._context.swapchain.extent)
        color = VkClearColorValue(float32=(c_float * 4)(*draw_ctx.draw_pass.clear_color))
        clear_value = VkClearValue(color=color)
        render_pass_begin_create = VkRenderPassBeginInfo(
            sType=VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
            renderPass=self._context.renderpass.vk_renderpass,
            framebuffer=self._context.swapchain.framebuffers[frame_sync.current_image_index()],
            renderArea=render_area,
            clearValueCount=0,
            pClearValues=None,
        )

        DeviceFunc.vkCmdBeginRenderPass(vk_command_buffer, byref(render_pass_begin_create), VK_SUBPASS_CONTENTS_INLINE)
        if should_clear:
            self._clear_color_attachment(vk_command_buffer, render_area, clear_value)

        draw_ctx.begin()

        def set_default_scissor() -> None:
            rect = VkRect2D(
                offset=VkOffset2D(x=0, y=0),
                extent=self._context.swapchain.extent,
            )
            rects = c_array_list([rect], VkRect2D)
            DeviceFunc.vkCmdSetScissor(vk_command_buffer, 0, 1, rects)

        def visit(group: Group) -> None:
            nonlocal current_pipeline
            nonlocal current_desc_sets
            nonlocal current_desc_binding
            nonlocal current_descriptor_key

            for domain_key, domain in self._domain_registry.items():
                selected_lists = [
                    vertex_list for vertex_list in selected_by_domain.get(domain, ())
                    if vertex_list.group is group
                ]
                if not selected_lists:
                    continue

                mode = domain_key.mode
                set_default_scissor()

                pipeline = self.pipeline_mgr.get_pipeline_from_group(
                    group,
                    self._context.renderpass,
                    mode,
                    self._context.window.width,
                    self._context.window.height,
                    domain,
                )
                pipeline_changed = current_pipeline is not pipeline
                if pipeline_changed:
                    pipeline.bind(vk_command_buffer)
                    current_pipeline = pipeline

                group_resources = tuple(get_group_resource_states(group))
                uniform_bindings = self._get_program_uniform_bindings(pipeline)
                descriptor_set_layouts_info = pipeline.descriptor_set_layouts_info
                assert descriptor_set_layouts_info is not None
                frame_index = self._context.active_frame
                self._upload_uniform_bindings_if_needed(uniform_bindings, frame_index)
                descriptor_resources = self.descriptor_mgr.build_resource_keys(
                    group_resources,
                    uniform_bindings,
                    frame_index,
                )
                descriptor_key = (
                    descriptor_set_layouts_info.key,
                    descriptor_resources,
                    frame_index,
                )
                descriptor_changed = current_descriptor_key != descriptor_key

                if descriptor_changed:
                    if descriptor_set_layouts_info.layouts:
                        descriptor_set = self.descriptor_mgr.get_descriptor_sets(
                            descriptor_set_layouts_info,
                            list(group_resources),
                            uniform_bindings=uniform_bindings,
                            resources=descriptor_resources,
                            frame_index=frame_index,
                            owner=self._context,
                        )
                        current_desc_sets = (descriptor_set,)
                    else:
                        current_desc_sets = ()
                        current_desc_binding = None
                    current_descriptor_key = descriptor_key

                if current_desc_sets and (descriptor_changed or pipeline_changed):
                    current_desc_binding = self._build_descriptor_binding_group(
                        current_desc_sets,
                        frame_index,
                    )
                    if current_desc_binding is not None:
                        current_desc_binding.bind(vk_command_buffer, current_pipeline.pipeline_layout)

                for state in getattr(group, "states", getattr(group, "_states", ())):
                    if state.sets_state:
                        state.set_state(draw_ctx)

                pipeline.push_constants(vk_command_buffer, 0)

                for selected in selected_lists:
                    domain._draw_subset_command(vk_command_buffer, mode, selected)  # noqa: SLF001

            children = self.group_children.get(group)
            if children:
                children.sort()
                for child in children:
                    if child.visible:
                        visit(child)

        self.top_groups.sort()
        for top_group in self.top_groups:
            if top_group.visible:
                visit(top_group)

        DeviceFunc.vkCmdEndRenderPass(vk_command_buffer)


