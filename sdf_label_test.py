import pyglet
from pyglet.font.base import SDFFont

class Camera:
    """ A simple 2D camera that contains the speed and offset."""

    def __init__(self, window: pyglet.window.Window, scroll_speed=1, min_zoom=0.25, max_zoom=4):
        assert min_zoom <= max_zoom, "Minimum zoom must not be greater than maximum zoom"
        self._window = window
        self.scroll_speed = scroll_speed
        self.max_zoom = max_zoom
        self.min_zoom = min_zoom
        self.offset_x = 0
        self.offset_y = 0
        self._zoom = max(min(1, self.max_zoom), self.min_zoom)

    @property
    def zoom(self):
        return self._zoom

    @zoom.setter
    def zoom(self, value):
        """ Here we set zoom, clamp value to minimum of min_zoom and max of max_zoom."""
        # Calculate the center of the view before zooming
        center_x = self.offset_x + self._window.width / (2 * self._zoom)
        center_y = self.offset_y + self._window.height / (2 * self._zoom)

        # Set the new zoom level
        self._zoom = max(min(value, self.max_zoom), self.min_zoom)

        # Adjust the offset to keep the center of the view in the same position
        self.offset_x = center_x - self._window.width / (2 * self._zoom)
        self.offset_y = center_y - self._window.height / (2 * self._zoom)

    @property
    def position(self):
        """Query the current offset."""
        return self.offset_x, self.offset_y

    @position.setter
    def position(self, value):
        """Set the scroll offset directly."""
        self.offset_x, self.offset_y = value

    def move(self, axis_x, axis_y):
        """ Move axis direction with scroll_speed.
            Example: Move left -> move(-1, 0)
         """
        self.offset_x += self.scroll_speed * axis_x
        self.offset_y += self.scroll_speed * axis_y

    def begin(self):
        # Set the current camera offset so you can draw your scene.

        # Translate using the offset.
        view_matrix = self._window.view.translate((-self.offset_x * self._zoom, -self.offset_y * self._zoom, 0))
        # Scale by zoom level.
        view_matrix = view_matrix.scale((self._zoom, self._zoom, 1))

        self._window.view = view_matrix

    def end(self):
        # Since this is a matrix, you will need to reverse the translate after rendering otherwise
        # it will multiply the current offset every draw update pushing it further and further away.

        # Reverse scale, since that was the last transform.
        view_matrix = self._window.view.scale((1 / self._zoom, 1 / self._zoom, 1))
        # Reverse translate.
        view_matrix = view_matrix.translate((self.offset_x * self._zoom, self.offset_y * self._zoom, 0))

        self._window.view = view_matrix

    def __enter__(self):
        self.begin()

    def __exit__(self, exception_type, exception_value, traceback):
        self.end()


class CustomFrame(pyglet.gui.Frame):
    def on_mouse_drag(self, x: int, y: int, dx: int, dy: int, buttons: int, modifiers: int) -> bool:
        """Pass the event to any widgets that are currently active."""
        handled = False
        for widget in self._active_widgets:
            if widget.on_mouse_drag(x, y, dx, dy, buttons, modifiers):
                handled = True
        self._mouse_pos = x, y
        return handled

class CustomSlider(pyglet.gui.Slider):
    """A Slider that can return status of mouse dragging."""
    def on_mouse_drag(self, x: int, y: int, dx: int, dy: int, buttons: int, modifiers: int) -> bool:
        if not self.enabled:
            return False
        if self._in_update:
            self._update_knob(x)
            return True
        return False

pyglet.resource.path.append('examples/gui/')
pyglet.resource.reindex()

window = pyglet.window.Window()
batch = pyglet.graphics.Batch()
ui_batch = pyglet.graphics.Batch()

sdf_font = SDFFont(None, "segoe_ui_msdf")
label = pyglet.text.SDFLabel(sdf_font, 'Hello World',
                          font_size=32,
                          x=window.width // 2,
                          y=window.height // 2,
                          anchor_x='center',
                          anchor_y='center',
                          batch=batch)

@window.event
def on_key_press(symbol, modifiers):
    if symbol == pyglet.window.key.R:
        label.scale = 1
    elif symbol == pyglet.window.key.F:
        label.color = (0, 255, 0, 255)

@window.event
def on_mouse_scroll(x, y, scroll_x, scroll_y):
    if scroll_y > 0:
        label.scale *= 1.25
        #camera.zoom *= 2
    elif scroll_y < 0:
        label.scale /= 1.25
        #camera.zoom /= 2

@window.event
def on_mouse_drag(x, y, dx, dy, buttons, modifiers):
    camera.offset_x -= dx
    camera.offset_y -= dy

bar = pyglet.resource.image('bar.png')
knob = pyglet.resource.image('knob.png')

def slider_handler(widget, value):
    new_value = value / 100
    slider_label.text = f"Outline Size: {new_value}"
    for group in label.group_cache.values():
        group.outline_thickness = new_value
    return True


frame = CustomFrame(window, order=4)

slider = CustomSlider(100, 200, bar, knob, edge=5, batch=ui_batch)
slider.set_handler('on_change', slider_handler)
frame.add_widget(slider)
slider_label = pyglet.text.Label("Outline Size: 0.0", x=300, y=200, batch=ui_batch, color=(255, 255, 255, 255))

def smooth_handler(widget, value):
    new_value = value / 500
    smooth_label.text = f"Smoothing: {new_value}"
    for group in label.group_cache.values():
        group.smoothing = new_value
    return True

slider2 = CustomSlider(100, 220, bar, knob, edge=5, batch=ui_batch)
slider2.set_handler('on_change', smooth_handler)
frame.add_widget(slider2)
smooth_label = pyglet.text.Label("Smoothing: 0.0", x=300, y=220, batch=ui_batch, color=(255,255, 255, 255))

def weight_handler(widget, value):
    new_value = value / 50
    weight_label.text = f"Weight: {new_value}"
    for group in label.group_cache.values():
        group.weight = new_value
    return True

slider3 = CustomSlider(100, 240, bar, knob, edge=5, batch=ui_batch)
slider3.set_handler('on_change', weight_handler)
frame.add_widget(slider3)
weight_label = pyglet.text.Label("Weight: 1.0", x=300, y=240, batch=ui_batch, color=(255, 255, 255, 255))


from pyglet.text.layout.sdf import get_default_layout_shader

shader = get_default_layout_shader()
elapsed = 0
def update(dt):
    global elapsed
    elapsed += dt
    with shader:
        try:
            shader["time"] = elapsed
        except:
            pass

pyglet.clock.schedule_interval(update, 1/60.0)


camera = Camera(window)
@window.event
def on_draw():
    window.clear()
    with camera:
        batch.draw()

    ui_batch.draw()

#pyglet.gl.glClearColor(0.8, 0.8, 0.8, 1.0)



pyglet.app.run()
