from pyglet.shapes import Circle
from pyglet.graphics import Group, Batch
from tests.annotations import GraphicsAPIGroups, require_graphics_api


pytestmark = require_graphics_api(GraphicsAPIGroups.GL3 + GraphicsAPIGroups.GL2)


def test_batch_migration(gl3_context):
    batch = Batch()
    group = Group(order=10)
    shape = Circle(100, 100, 50, batch=batch, group=group)
    assert shape.batch == batch
    assert shape.group == group

    new_batch = Batch()
    shape.batch = new_batch
    assert shape.batch == new_batch


def test_group_migration(gl3_context):
    batch = Batch()
    group = Group(order=10)
    shape = Circle(100, 100, 50, batch=batch, group=group)
    assert shape.batch == batch
    assert shape.group == group

    new_group = Group()
    shape.group = new_group
    assert shape.group == new_group
