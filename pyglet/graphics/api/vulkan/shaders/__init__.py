


# Blank.
# Only to display a default black screen if started empty. Vulkan requires at least one pipeline and shader, just
# to clear the screen. Otherwise a white window for Win32 will appear.
# vertex_src = """#version 450
#
# void main() {
#     gl_Position = vec4(0.0, 0.0, 0.0, 1.0);
# }
# """
#
# # Fragment shader source code as a string
# frag_src = """#version 450
#
# layout(location = 0) out vec4 color;
#
# void main() {
#     color = vec4(0.0, 0.0, 0.0, 1.0);
# }
#"""

# Primitives:
# vertex_src = """#version 450
#
# layout(location = 0) in vec3 position;
# layout(location = 1) in vec4 colors;
#
# layout(set = 0, binding = 0) uniform WindowBlock {
#     mat4 projection;
#     mat4 view;
# } window;
#
# layout(location = 1) out vec4 vertex_colors;
#
# void main() {
#     gl_Position = window.projection * window.view * vec4(position, 1.0);
#     vertex_colors = colors;
# }
# """
#
# # Fragment shader source code as a string
# frag_src = """#version 450
#
# layout(location = 0) out vec4 color;  // Attachment 0
# layout(location = 1) in vec4 vertex_colors;
#
# void main() {
#     color = vertex_colors;
# }
# """



# Layouts
# vertex_src = """#version 450
# layout(location = 0) in vec3 position;
# layout(location = 1) in vec4 colors;
# layout(location = 2) in vec3 tex_coords;
# layout(location = 3) in vec3 translation;
# layout(location = 4) in vec3 view_translation;
# layout(location = 5) in vec2 anchor;
# layout(location = 6) in float rotation;
# layout(location = 7) in float visible;
#
# layout(location = 0) out vec4 text_colors;
# layout(location = 1) out vec2 texture_coords;
# layout(location = 2) out vec4 vert_position;
#
# layout(set = 0, binding = 0) uniform WindowBlock {
#     mat4 projection;
#     mat4 view;
# } window;
#
# void main()
# {
#     mat4 m_rotation = mat4(1.0);
#     vec3 v_anchor = vec3(anchor.x, anchor.y, 0);
#     mat4 m_anchor = mat4(1.0);
#     mat4 m_translate = mat4(1.0);
#
#     m_translate[3][0] = translation.x;
#     m_translate[3][1] = translation.y;
#     m_translate[3][2] = translation.z;
#     m_rotation[0][0] =  cos(-radians(rotation));
#     m_rotation[0][1] =  sin(-radians(rotation));
#     m_rotation[1][0] = -sin(-radians(rotation));
#     m_rotation[1][1] =  cos(-radians(rotation));
#
#     gl_Position = window.projection * window.view * m_translate * m_anchor * m_rotation * vec4(position + view_translation + v_anchor, 1.0) * visible;
#
#     vert_position = vec4(position + translation + view_translation + v_anchor, 1.0);
#     text_colors = colors;
#     texture_coords = tex_coords.xy;
# }
# """
#
# # Fragment shader source code as a string
# frag_src = """#version 450
# layout(location = 0) in vec4 text_colors;
# layout(location = 1) in vec2 texture_coords;
# layout(location = 2) in vec4 vert_position;
#
# layout(location = 0) out vec4 final_colors;
#
# layout(push_constant) uniform PushConstants {
#     bool scissor;
#     vec4 scissor_area;
# } pc_scissor;
#
# void main()
# {
#     final_colors = vec4(text_colors.rgb, texture(text, texture_coords).a * text_colors.a);
#     if (pc_scissor.scissor == true) {
#         if (vert_position.x < pc_scissor.scissor_area[0]) discard;                     // left
#         if (vert_position.y < pc_scissor.scissor_area[1]) discard;                     // bottom
#         if (vert_position.x > pc_scissor.scissor_area[0] + pc_scissor.scissor_area[2]) discard;   // right
#         if (vert_position.y > pc_scissor.scissor_area[1] + pc_scissor.scissor_area[3]) discard;   // top
#     }
# }
# """

# Layout decoration
# vertex_src = """#version 450
# layout(location = 0) in vec3 position;
# layout(location = 1) in vec4 colors;
# layout(location = 3) in vec3 translation;
# layout(location = 4) in vec3 view_translation;
# layout(location = 5) in vec2 anchor;
# layout(location = 6) in float rotation;
# layout(location = 7) in float visible;
#
# layout(location = 0) out vec4 vert_colors;
# layout(location = 2) out vec4 vert_position;
#
# layout(set = 0, binding = 0) uniform WindowBlock {
#     mat4 projection;
#     mat4 view;
# } window;
#
# void main()
# {
#     mat4 m_rotation = mat4(1.0);
#     vec3 v_anchor = vec3(anchor.x, anchor.y, 0);
#     mat4 m_anchor = mat4(1.0);
#     mat4 m_translate = mat4(1.0);
#
#     m_translate[3][0] = translation.x;
#     m_translate[3][1] = translation.y;
#     m_translate[3][2] = translation.z;
#     m_rotation[0][0] =  cos(-radians(rotation));
#     m_rotation[0][1] =  sin(-radians(rotation));
#     m_rotation[1][0] = -sin(-radians(rotation));
#     m_rotation[1][1] =  cos(-radians(rotation));
#
#     gl_Position = window.projection * window.view * m_translate * m_anchor * m_rotation * vec4(position + view_translation + v_anchor, 1.0) * visible;
#
#     vert_position = vec4(position + translation + view_translation + v_anchor, 1.0);
#     vert_colors = colors;
# }
# """
#
# # Fragment shader source code as a string
# frag_src = """#version 450
# layout(location = 0) in vec4 vert_colors;
# layout(location = 2) in vec4 vert_position;
#
# layout(location = 0) out vec4 final_colors;
#
# layout(push_constant) uniform PushConstants {
#     bool scissor;
#     vec4 scissor_area;
# } pc_scissor;
#
# void main()
# {
#     final_colors = vert_colors;
#     if (pc_scissor.scissor == true) {
#         if (vert_position.x < pc_scissor.scissor_area[0]) discard;                     // left
#         if (vert_position.y < pc_scissor.scissor_area[1]) discard;                     // bottom
#         if (vert_position.x > pc_scissor.scissor_area[0] + pc_scissor.scissor_area[2]) discard;   // right
#         if (vert_position.y > pc_scissor.scissor_area[1] + pc_scissor.scissor_area[3]) discard;   // top
#     }
# }
# """