from __future__ import annotations


from typing import Callable, Sequence, Any, TYPE_CHECKING
from ctypes import byref, c_float
import pyglet
from pyglet.graphics.api import resource_manager
from pyglet.libs.shared.vulkan_lib.vulkan_core import VkRenderPassBeginInfo, \
    VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO, VkOffset2D, VkRect2D, VkClearColorValue, VkClearValue, \
    VK_SUBPASS_CONTENTS_INLINE

from pyglet.graphics.draw import _DomainKey, Batch, Group
from pyglet.graphics.shader import Attribute
from pyglet.graphics.state import State, TextureState, UniformBufferState
from pyglet.graphics.api.vulkan import vertexdomain, c_array_list, DeviceFunc

_debug_graphics_batch = pyglet.options.debug_graphics_batch

if TYPE_CHECKING:
    from pyglet.graphics.api.vulkan.vertexdomain import VertexList, IndexedVertexList
    from pyglet.graphics.api.vulkan.pipeline import GraphicsPipeline
    from pyglet.graphics.api.vulkan.descriptor import DescriptorSetObject
    from pyglet.graphics import GeometryMode
    from pyglet.graphics.api.gl2.shader import ShaderProgram



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




_domain_class_map: dict[tuple[bool, bool], type[vertexdomain.VertexDomain]] = {
    # Indexed, Instanced : Domain
    (False, False): vertexdomain.VertexDomain,
    (True, False): vertexdomain.IndexedVertexDomain,
   # (False, True): vertexdomain.InstancedVertexDomain,
    #(True, True): vertexdomain.InstancedIndexedVertexDomain,
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


class DrawListManager:
    def __init__(self):
        self.pipeline_mgr = pyglet.graphics.api.core.pipeline_mgr
        self.descriptor_mgr = pyglet.graphics.api.core.descriptor_mgr
        self.optimized_draw_list = []  # Optimized draw list

    def process_groups(self, groups):
        """Process all groups in bulk to create an optimized draw list.
        """
        # Step 2: Generate pipelines and descriptor sets
        current_pipeline = None
        current_descriptor_set = None

        for group in groups:
            # Get or create the pipeline
            pipeline = self.pipeline_mgr.get_pipeline_from_group(group)
            if pipeline != current_pipeline:
                self.optimized_draw_list.append({"bind_pipeline": pipeline})
                current_pipeline = pipeline

            # Get or create the descriptor set
            resources = get_group_resource_states(group)
            if resources:
                descriptor_set = self.descriptor_mgr.get_descriptor_sets(resources)
                if descriptor_set != current_descriptor_set:
                    self.optimized_draw_list.append({"bind_descriptor_set": descriptor_set})
                    current_descriptor_set = descriptor_set

            # Add the draw calls
            for draw_call in group.draw_calls:
                self.optimized_draw_list.append({
                    "draw_call": {
                        "vertices": draw_call["vertices"],
                        "indices": draw_call["indices"],
                    },
                })



    def get_descriptor_set(self, resources):
        """Determine or reuse a Vulkan descriptor set for the group.
        """
        resource_key = sha256(str(resources).encode()).hexdigest()
        if resource_key not in self.descriptor_set_cache:
            # Simulate descriptor set creation
            self.descriptor_set_cache[resource_key] = f"DescriptorSet_{resource_key[:6]}"
        return self.descriptor_set_cache[resource_key]

#_dl_manager = DrawListManager()


def get_group_resource_states(group: Group):
    ubo = []
    texture = []
    for state in group._states:
        if isinstance(state, UniformBufferState):
            ubo.append(state)
        elif isinstance(state, TextureState):
            texture.append(state)
    return ubo + texture

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
    _draw_list: list[Callable]
    top_groups: list[Group]
    group_children: dict[Group, list[Group]]
    group_map: dict[Group, dict[_DomainKey, vertexdomain.VertexDomain]]

    def __init__(self) -> None:
        """Create a graphics batch."""
        # Mapping to find domain.
        # group -> (attributes, mode, indexed) -> domain
        super().__init__()
        self.pipelines = []

        # Context this batch was created in.
        self._window_ctx = pyglet.graphics.api.core.current_window
        self._devices = pyglet.graphics.api.core.devices
        self.pipeline_mgr = pyglet.graphics.api.core.pipeline_mgr
        self.descriptor_mgr = pyglet.graphics.api.core.descriptor_mgr
        resource_manager.register_resource(self)

    def delete(self) -> None:
        for group in self.group_map.values():
            for domain in group.values():
                print("DOMAIN", domain)
                domain.delete()

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
        # No new attributes.
        attributes = program.attributes.copy()

        # Formats may differ (normalization) than what is declared in the shader.
        # Make those adjustments and attempt to get a domain.
        for a_name in attributes:
            if (a_name in vertex_list.initial_attribs and
                    vertex_list.initial_attribs[a_name]['format'] != attributes[a_name]['format']):
                attributes[a_name]['format'] = vertex_list.initial_attribs[a_name]['format']

        domain = self.get_domain(vertex_list.indexed, vertex_list.instanced, mode, group, attributes)

        # TODO: Allow migration if we can restore original vertices somehow. Much faster.
        # If the domain's don't match, we need to re-create the vertex list. Tell caller no match.
        if domain != vertex_list.domain:
            return False

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
        vertex_list.migrate(domain)

    def get_domain(self, indexed: bool, instanced: bool, mode: GeometryMode, group: Group,
                   attributes: dict[str, Any]) -> (
            vertexdomain.VertexDomain | vertexdomain.IndexedVertexDomain | vertexdomain.InstancedVertexDomain |
            vertexdomain.InstancedIndexedVertexDomain):
        """Get, or create, the vertex domain corresponding to the given arguments.

        mode is the render mode such as GL_LINES or GL_TRIANGLES
        """
        # Batch group
        if group not in self.group_map:
            self._add_group(group)

        domain_map = self.group_map[group]

        # If instanced, ensure a separate domain, as multiple instance sources can match the key.
        if instanced:
            self._instance_count += 1
            key = (indexed, self._instance_count, mode, str(attributes))
        else:
            # Find domain given formats, indices and mode
            key = (indexed, 0, mode, str(attributes))

        try:
            domain = domain_map[key]
        except KeyError:
            # Create domain
            domain = _domain_class_map[(indexed, instanced)](self._window_ctx, self.initial_count, attributes)
            domain_map[key] = domain
            self._draw_list_dirty = True

        return domain

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

    def _update_draw_list(self) -> None:
        """Visit group tree in preorder and create a list of bound methods to call."""
        current_pipeline: GraphicsPipeline | None = None
        current_desc_set: DescriptorSetObject | None = None
        current_resources: list[State] | None = None

        frame_sync = self._window_ctx.frame_sync

        current_cb = frame_sync.get_current_command_buffer(self._window_ctx.default_cb_id)
        current_cb.reset()

        vk_command_buffer = current_cb.command_buffer

        render_area = VkRect2D(offset=VkOffset2D(x=0, y=0),
                               extent=self._window_ctx.swapchain.extent)
        color = VkClearColorValue(float32=(c_float * 4)(*self._window_ctx.clear_color))
        clear_value = VkClearValue(color=color)

        cb_array = c_array_list([clear_value], VkClearValue)

        render_pass_begin_create = VkRenderPassBeginInfo(
            sType=VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
            renderPass=self._window_ctx.renderpass.vk_renderpass,
            framebuffer=self._window_ctx.swapchain.framebuffers[frame_sync.current_image_index()],
            renderArea=render_area,
            clearValueCount=1,
            pClearValues=cb_array,
        )

        current_cb.begin()

        DeviceFunc.vkCmdBeginRenderPass(vk_command_buffer, byref(render_pass_begin_create), VK_SUBPASS_CONTENTS_INLINE)

        def visit(group: Group) -> list:
            nonlocal current_pipeline
            nonlocal current_desc_set
            nonlocal current_resources

            draw_list = []

            # Draw domains using this group
            domain_map = self.group_map[group]

            # indexed, instanced, mode, program, str(attributes))
            for (indexed, instanced, mode, formats), domain in list(domain_map.items()):
                # Remove unused domains from batch
                if domain.is_empty:
                    del domain_map[(indexed, instanced, mode, formats)]
                    continue

                #print("DOMAIN!", domain)

                pipeline = self.pipeline_mgr.get_pipeline_from_group(group,
                                                                     self._window_ctx.renderpass,
                                                                     mode,
                                                                     self._window_ctx.window.width,
                                                                     self._window_ctx.window.height,
                                                                     domain)
                if current_pipeline is not pipeline:
                    pipeline.bind(vk_command_buffer)

                    #self.descriptor_mgr.get
                    current_pipeline = pipeline

                group_resources = get_group_resource_states(group)
                # If the descriptor set is not the same, change it.
                if current_resources != group_resources:
                    # Get or create the descriptor set
                    current_desc_set, created = self.descriptor_mgr.get_descriptor_sets(pipeline.descriptor_set_layouts, group_resources)

                    # Bind first so DescriptorSetObject.current_frame matches the frame being recorded.
                    current_desc_set.bind_to_pipeline(vk_command_buffer,
                                                      current_pipeline.pipeline_layout,
                                                      frame_sync.current_frame)

                    # Update the per-frame window UBO binding after selecting the frame index above.
                    current_desc_set.bind_ubo(self._window_ctx.window._matrices.ubo, 0)

                    for state in group._states:
                        if state.resolves_state:
                            state.resolve_state(current_desc_set)
                        if state.sets_state:
                            state.set_state(None)

                    current_resources = group_resources

                pipeline.push_constants(vk_command_buffer, 0)

                domain.draw(vk_command_buffer)

            # Sort and visit child groups of this group
            children = self.group_children.get(group)
            if children:
                children.sort()
                for child in list(children):
                    if child.visible:
                        draw_list.extend(visit(child))

            if children or domain_map:
                return [*draw_list]

            # Remove unused group from batch
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

        current_cb.end()

    def _dump_draw_list(self) -> None:
        def dump(group: Group, indent: str = '') -> None:
            print(indent, 'Begin group', group)
            domain_map = self.group_map[group]
            for domain in domain_map.values():
                print(indent, '  ', domain)
                for start, size in zip(*domain.vertex_buffers.allocator.get_allocated_regions()):
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
        if self._draw_list_dirty:
            self._update_draw_list()

        #for func in self._draw_list:
            #print("FUNC!", func)
            #func()

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

        # Horrendously inefficient.
        def visit(group: Group) -> None:
            group.set_state()

            # Draw domains using this group
            domain_map = self.group_map[group]
            for (_, _, mode, _), domain in domain_map.items():
                for alist in vertex_lists:
                    if alist.domain is domain:
                        alist.draw(mode)

            # Sort and visit child groups of this group
            children = self.group_children.get(group)
            if children:
                children.sort()
                for child in children:
                    if child.visible:
                        visit(child)

            group.unset_state()

        self.top_groups.sort()
        for top_group in self.top_groups:
            if top_group.visible:
                visit(top_group)


