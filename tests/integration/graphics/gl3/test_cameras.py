import pyglet
from tests.annotations import GraphicsAPIGroups, require_graphics_api


pytestmark = require_graphics_api(GraphicsAPIGroups.GL3)


class DummyDrawContext:
    active_shader_program = None


def test_camera_ubo_views_do_not_overwrite_stable_parent_range(gl3_context, monkeypatch):
    gl3_context.switch_to()
    ctx = gl3_context.context
    ctx.frame_resources.delete()

    monkeypatch.setattr(ctx, "create_frame_fence", object)
    monkeypatch.setattr(ctx, "poll_frame_fence", lambda _fence: True)
    monkeypatch.setattr(ctx, "delete_frame_fence", lambda _fence: None)

    camera = pyglet.window.camera.Camera2D(gl3_context)
    outer = camera.create_view(inherit=True)
    inner = outer.create_view(inherit=True)
    draw_context = DummyDrawContext()

    try:
        for frame_index, outer_y, inner_x in (
            (0, 0.0, 0.0),
            (1, 75.0, 35.0),
            (2, 150.0, 70.0),
            (3, 225.0, 105.0),
            (4, 300.0, 140.0),
        ):
            ctx.frame_resources.frame_begin(frame_index)
            outer.position = (0.0, outer_y)
            inner.position = (inner_x, 0.0)

            camera.begin(draw_context=draw_context)
            outer.begin(draw_context=draw_context)
            inner.begin(draw_context=draw_context)
            ctx.frame_resources.frame_submit()

        root_storage = camera.view.storage
        outer_storage = outer.storage
        inner_storage = inner.storage

        assert root_storage._current_binding.offset != outer_storage._current_binding.offset  # noqa: SLF001
        assert root_storage._current_binding.offset != inner_storage._current_binding.offset  # noqa: SLF001
        assert outer_storage._current_binding.offset != inner_storage._current_binding.offset  # noqa: SLF001
    finally:
        ctx.frame_resources.delete()
