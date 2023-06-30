from OpenGL.GL import *
from glfw.GLFW import *
import glm
import ctypes
import numpy as np
import copy

# 투영 설정
g_is_perspective = True

# rendering mode 설정
g_line_rendering = True

# animating mode 설정
g_is_animating = False
g_idx = 0
g_pressed_time = 0

# parsing 정보 저장
g_root = None
g_node_list = []
g_frames = []
g_frame_time = 0

# parsing copy
g_root_copy = None
g_node_list_copy = []

# 초기 각도 및 거리 설정
g_cam_azimuth = np.radians(45.)
g_cam_elevation = np.radians(45.)
g_cam_distance = 5.

# lookAt 함수의 인자
g_eye = glm.vec3(g_cam_distance * np.cos(g_cam_elevation) * np.sin(g_cam_azimuth), g_cam_distance * np.sin(g_cam_elevation), g_cam_distance *
                 np.cos(g_cam_elevation) * np.cos(g_cam_azimuth))
g_at = glm.vec3(0.0, 0.0, 0.0)
g_up_vector = glm.vec3(0, 1, 0)

# 마우스 이전 위치
g_prev_xpos = 0.
g_prev_ypos = 0.

g_vertex_shader_src = '''
# version 330 core

layout (location = 0) in vec3 vin_pos;
layout (location = 1) in vec3 vin_color;

out vec4 vout_color;

uniform mat4 MVP;

void main()
{
    // 3D points in homogeneous coordinates
    vec4 p3D_in_hcoord = vec4(vin_pos.xyz, 1.0);

    gl_Position = MVP * p3D_in_hcoord;

    vout_color = vec4(vin_color, 1.);
    //vout_color = vec4(1., 1., 1., 1.);
}
'''

g_fragment_shader_src = '''
# version 330 core

in vec4 vout_color;

out vec4 FragColor;

void main()
{
    FragColor = vout_color; 
}
'''

g_vertex_shader_lighting = '''
#version 330 core

layout (location = 0) in vec3 vin_pos; 
layout (location = 1) in vec3 vin_normal; 

out vec3 vout_surface_pos;
out vec3 vout_normal;

uniform mat4 MVP;
uniform mat4 M;

void main()
{
    vec4 p3D_in_hcoord = vec4(vin_pos.xyz, 1.0);
    gl_Position = MVP * p3D_in_hcoord;

    vout_surface_pos = vec3(M * vec4(vin_pos, 1));
    vout_normal = normalize( mat3(inverse(transpose(M)) ) * vin_normal);
}
'''

g_fragment_shader_lighting = '''
#version 330 core

in vec3 vout_surface_pos;
in vec3 vout_normal;

out vec4 FragColor;

uniform vec3 view_pos;

void main()
{
    // light and material properties
    vec3 light_pos = vec3(3,2,4);
    vec3 light_color = vec3(1,1,1);
    vec3 material_color = vec3(0,1,0);
    float material_shininess = 32.0;

    // light components
    vec3 light_ambient = 0.1*light_color;
    vec3 light_diffuse = light_color;
    vec3 light_specular = light_color;

    // material components
    vec3 material_ambient = material_color;
    vec3 material_diffuse = material_color;
    vec3 material_specular = light_color;  // for non-metal material

    // ambient
    vec3 ambient = light_ambient * material_ambient;

    // for diffiuse and specular
    vec3 normal = normalize(vout_normal);
    vec3 surface_pos = vout_surface_pos;
    vec3 light_dir = normalize(light_pos - surface_pos);

    // diffuse
    float diff = max(dot(normal, light_dir), 0);
    vec3 diffuse = diff * light_diffuse * material_diffuse;

    // specular
    vec3 view_dir = normalize(view_pos - surface_pos);
    vec3 reflect_dir = reflect(-light_dir, normal);
    float spec = pow( max(dot(view_dir, reflect_dir), 0.0), material_shininess);
    vec3 specular = spec * light_specular * material_specular;

    vec3 color = ambient + diffuse + specular;
    FragColor = vec4(color, 1.);
}
'''


class Node:
    def __init__(self, parent, link_transform_from_parent, shape_transform, color, channels, offset):
        # hierarchy
        self.parent = parent
        self.children = []
        if parent is not None:
            parent.children.append(self)
        self.offset = offset
        self.channels = channels

        # transform
        self.link_transform_from_parent = link_transform_from_parent
        self.joint_transform = glm.mat4()
        self.global_transform = glm.mat4()

        # shape
        self.shape_transform = shape_transform
        self.color = color

    def set_joint_transform(self, joint_transform):
        self.joint_transform = joint_transform

    def update_tree_global_transform(self):
        if self.parent is not None:
            self.global_transform = self.parent.get_global_transform(
            ) * self.link_transform_from_parent * self.joint_transform
        else:
            self.global_transform = self.link_transform_from_parent * self.joint_transform

        for child in self.children:
            child.update_tree_global_transform()

    def get_global_transform(self):
        return self.global_transform

    def get_shape_transform(self):
        return self.shape_transform

    def get_color(self):
        return self.color


def load_shaders(vertex_shader_source, fragment_shader_source):
    # build and compile our shader program
    # ------------------------------------

    # vertex shader
    # create an empty shader object
    vertex_shader = glCreateShader(GL_VERTEX_SHADER)
    # provide shader source code
    glShaderSource(vertex_shader, vertex_shader_source)
    # compile the shader object
    glCompileShader(vertex_shader)

    # -1 for shader compile errors
    success = glGetShaderiv(vertex_shader, GL_COMPILE_STATUS)
    if (not success):
        infoLog = glGetShaderInfoLog(vertex_shader)
        print("ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" + infoLog.decode())

    # fragment shader
    # create an empty shader object
    fragment_shader = glCreateShader(GL_FRAGMENT_SHADER)
    # provide shader source code
    glShaderSource(fragment_shader, fragment_shader_source)
    # compile the shader object
    glCompileShader(fragment_shader)

    # -1 for shader compile errors
    success = glGetShaderiv(fragment_shader, GL_COMPILE_STATUS)
    if (not success):
        infoLog = glGetShaderInfoLog(fragment_shader)
        print("ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n" + infoLog.decode())

    # link shaders
    # create an empty program object
    shader_program = glCreateProgram()
    # attach the shader objects to the program object
    glAttachShader(shader_program, vertex_shader)
    glAttachShader(shader_program, fragment_shader)
    glLinkProgram(shader_program)                    # link the program object

    # -1 for linking errors
    success = glGetProgramiv(shader_program, GL_LINK_STATUS)
    if (not success):
        infoLog = glGetProgramInfoLog(shader_program)
        print("ERROR::SHADER::PROGRAM::LINKING_FAILED\n" + infoLog.decode())

    glDeleteShader(vertex_shader)
    glDeleteShader(fragment_shader)

    return shader_program    # return the shader program


def scroll_callback(window, xoffset, yoffset):  # zoom in/out with mouse wheel
    global g_cam_distance, g_eye
    g_cam_distance -= yoffset * .1

    if g_cam_distance < 0:
        g_cam_distance = 0

    # g_eye update
    g_eye = glm.vec3(g_cam_distance * np.cos(g_cam_elevation) * np.sin(g_cam_azimuth), g_cam_distance * np.sin(g_cam_elevation), g_cam_distance *
                     np.cos(g_cam_elevation) * np.cos(g_cam_azimuth))


def cursor_pos_callback(window, xpos, ypos):  # orbit camera
    global g_cam_azimuth, g_cam_elevation, g_cam_distance, g_prev_xpos, g_prev_ypos, g_eye, g_at, g_up_vector
    if glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS:
        sensitivity = .1

        # 이전 마우스 위치와 현재 마우스 위치의 차이를 계산
        delta_x = g_prev_xpos - xpos
        delta_y = ypos - g_prev_ypos

        # 각도 update
        g_cam_azimuth += np.radians(delta_x) * sensitivity
        g_cam_elevation += np.radians(delta_y) * sensitivity

        # elevation 값 범위 제한
        if g_cam_elevation >= 2 * np.pi:
            g_cam_elevation -= 2 * np.pi
        if g_cam_elevation <= -2 * np.pi:
            g_cam_elevation += 2 * np.pi

        # g_up_vector update
        if g_cam_elevation >= np.pi / 2 and g_cam_elevation < 3 * np.pi / 2:
            g_up_vector = glm.vec3(0, -1, 0)
        elif (g_cam_elevation >= 0 and g_cam_elevation < np.pi / 2) or g_cam_elevation >= 3 * np.pi / 2:
            g_up_vector = glm.vec3(0, 1, 0)

        if g_cam_elevation <= - np.pi / 2 and g_cam_elevation > -3 * np.pi / 2:
            g_up_vector = glm.vec3(0, -1, 0)
        elif (g_cam_elevation <= 0 and g_cam_elevation > -np.pi / 2) or g_cam_elevation <= -3 * np.pi / 2:
            g_up_vector = glm.vec3(0, 1, 0)

        # 카메라 위치 x좌표 update
        g_eye[0] = g_cam_distance * \
            np.cos(g_cam_elevation) * np.sin(g_cam_azimuth)
        # 카메라 위치 y좌표 update
        g_eye[1] = np.sin(g_cam_elevation) * g_cam_distance
        # 카메라 위치 z좌표 update
        g_eye[2] = g_cam_distance * \
            np.cos(g_cam_elevation) * np.cos(g_cam_azimuth)

        # pan 고려
        g_eye += g_at

    elif glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS:
        panning(window, xpos, ypos)

    g_prev_xpos = xpos
    g_prev_ypos = ypos


def panning(window, xpos, ypos):  # pan camera
    global g_eye, g_at, g_up_vector, g_prev_xpos, g_prev_ypos
    if glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS:
        sensitivity = .01
        delta_x = g_prev_xpos - xpos
        delta_y = ypos - g_prev_ypos

        cam_w = glm.normalize(g_eye - g_at)  # backward
        cam_u = glm.normalize(glm.cross(g_up_vector, cam_w))  # right
        cam_v = glm.normalize(glm.cross(cam_w, cam_u))  # up

        # pan 계산
        pan = cam_u * delta_x * sensitivity + \
            cam_v * delta_y * sensitivity

        # at, eye update
        g_at += pan
        g_eye += pan


def key_callback(window, key, scancode, action, mods):
    global g_cam_azimuth, g_is_perspective, g_line_rendering, g_is_animating, g_node_list, g_node_list_copy, g_root, g_root_copy, g_idx, g_pressed_time
    if key == GLFW_KEY_ESCAPE and action == GLFW_PRESS:
        glfwSetWindowShouldClose(window, GLFW_TRUE)
    else:
        if action == GLFW_PRESS or action == GLFW_REPEAT:
            if key == GLFW_KEY_V:
                g_is_perspective = not g_is_perspective
            elif key == GLFW_KEY_1:
                g_line_rendering = True
            elif key == GLFW_KEY_2:
                g_line_rendering = False
            elif key == GLFW_KEY_SPACE:
                g_is_animating = not g_is_animating
                g_node_list = copy.deepcopy(g_node_list_copy)
                g_root = g_node_list[0]
                g_idx = 0
                g_pressed_time = glfwGetTime()


def drop_callback(window, paths):
    for path in paths:
        print("File name: ", path)
        if (path.endswith(".bvh")):
            print_bvh_info(path)
            parse_bvh_skeleton(path)
            parse_bvh_motion(path)


def print_bvh_info(path):
    global g_frame_time

    with open(path, 'r') as file:
        data = file.read()

    # Number of frames 추출
    num_frames_start = data.index("Frames:") + len("Frames:")
    num_frames_end = data.index("\n", num_frames_start)
    num_frames = int(data[num_frames_start:num_frames_end])

    # FPS 추출
    frame_time_start = data.index("Frame Time:") + len("Frame Time:")
    frame_time_end = data.index("\n", frame_time_start)
    frame_time = float(data[frame_time_start:frame_time_end])
    fps = int(1 / frame_time)
    g_frame_time = frame_time

    # Number of joints 추출
    num_joints = data.count("JOINT") + 1

    # 필요한 정보 출력
    print("Number of frames:", num_frames)
    print("FPS:", fps)
    print("Number of joints:", num_joints)
    print("List of all joint names: ")


def parse_bvh_skeleton(path):
    global g_node_list, g_root, g_node_list_copy, g_root_copy

    g_root = None
    g_node_list.clear()

    stack = []
    offset = []

    end_site = 0

    for line in open(path, "r"):
        tokens = line.strip().split()
        if tokens[0] == 'ROOT':
            name = tokens[1]
            print(name)
        elif tokens[0] == 'JOINT':
            name = tokens[1]
            print(name)
        elif tokens[0] == 'End':
            end_site = 1
        elif tokens[0] == 'OFFSET':
            offset.clear()
            offset.append(float(tokens[1]))
            offset.append(float(tokens[2]))
            offset.append(float(tokens[3]))

            if end_site == 1:
                end_site = 0
                position = glm.vec3(offset[0], offset[1], offset[2])
                node = Node(stack[-1], glm.translate(glm.vec3(offset[0], offset[1], offset[2])),
                            glm.mat4(), glm.vec3(0, 0, 1), None, position)
                stack.append(node)
                g_node_list.append(node)
        elif tokens[0] == 'CHANNELS':
            if tokens[1] == '6':
                channel = tokens[2:]
                position = glm.vec3(offset[0], offset[1], offset[2])
                root = Node(None, glm.mat4(), glm.mat4(),
                            glm.vec3(0, 0, 1), channel, position)
                stack.append(root)
                g_node_list.append(root)
                g_root = root
            elif tokens[1] == '3':
                channel = tokens[2:]
                position = glm.vec3(offset[0], offset[1], offset[2])
                node = Node(stack[-1], glm.translate(glm.vec3(offset[0], offset[1], offset[2])),
                            glm.mat4(), glm.vec3(0, 0, 1), channel, position)
                stack.append(node)
                g_node_list.append(node)
        elif tokens[0] == '}':
            stack.pop()

    g_node_list_copy = copy.deepcopy(g_node_list)
    g_root_copy = g_node_list_copy[0]


def parse_bvh_motion(path):
    global g_frames

    g_frames.clear()

    data = read_motion_data(path)
    g_frames = data[2:]


def read_motion_data(path):
    motion_data = []

    with open(path, 'r') as file:
        is_motion_section = False

        for line in file:
            line = line.strip()

            if line == "MOTION":
                is_motion_section = True
                continue

            if is_motion_section:
                f_data = line.split()
                motion_data.append(f_data)

    return motion_data


def prepare_vao_frame():
    # prepare vertex data (in main memory)
    vertices = glm.array(glm.float32,
                         # position        # color
                         -15.0, 0.0, 0.0,  1.0, 0.0, 0.0,  # x-axis start
                         15.0, 0.0, 0.0,  1.0, 0.0, 0.0,  # x-axis end
                         0.0, 0.0, 0.0,  0.0, 1.0, 0.0,  # y-axis start
                         0.0, 0.0, 0.0,  0.0, 1.0, 0.0,  # y-axis end
                         0.0, 0.0, -15.0,  0.0, 0.0, 1.0,  # z-axis start
                         0.0, 0.0, 15.0,  0.0, 0.0, 1.0,  # z-axis end
                         )

    # create and activate VAO (vertex array object)
    # create a vertex array object ID and store it to VAO variable
    VAO = glGenVertexArrays(1)
    glBindVertexArray(VAO)      # activate VAO

    # create and activate VBO (vertex buffer object)
    # create a buffer object ID and store it to VBO variable
    VBO = glGenBuffers(1)
    # activate VBO as a vertex buffer object
    glBindBuffer(GL_ARRAY_BUFFER, VBO)

    # copy vertex data to VBO
    # allocate GPU memory for and copy vertex data to the currently bound vertex buffer
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes,
                 vertices.ptr, GL_STATIC_DRAW)

    # configure vertex positions
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 *
                          glm.sizeof(glm.float32), None)
    glEnableVertexAttribArray(0)

    # configure vertex colors
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 *
                          glm.sizeof(glm.float32), ctypes.c_void_p(3*glm.sizeof(glm.float32)))
    glEnableVertexAttribArray(1)

    return VAO


def prepare_vao_xz():
    # prepare vertex data (in main memory)
    vertices = glm.array(glm.float32,
                         # position        # color
                         -15.0, 0.0, 0.0,  1.0, 1.0, 1.0,  # x-axis start
                         15.0, 0.0, 0.0,  1.0, 1.0, 1.0,  # x-axis end
                         0.0, 0.0, -15.0,  1.0, 1.0, 1.0,  # z-axis start
                         0.0, 0.0, 15.0,  1.0, 1.0, 1.0,  # z-axis end
                         )

    # create and activate VAO (vertex array object)
    # create a vertex array object ID and store it to VAO variable
    VAO = glGenVertexArrays(1)
    glBindVertexArray(VAO)      # activate VAO

    # create and activate VBO (vertex buffer object)
    # create a buffer object ID and store it to VBO variable
    VBO = glGenBuffers(1)
    # activate VBO as a vertex buffer object
    glBindBuffer(GL_ARRAY_BUFFER, VBO)

    # copy vertex data to VBO
    # allocate GPU memory for and copy vertex data to the currently bound vertex buffer
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes,
                 vertices.ptr, GL_STATIC_DRAW)

    # configure vertex positions
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 *
                          glm.sizeof(glm.float32), None)
    glEnableVertexAttribArray(0)

    # configure vertex colors
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 *
                          glm.sizeof(glm.float32), ctypes.c_void_p(3*glm.sizeof(glm.float32)))
    glEnableVertexAttribArray(1)

    return VAO


def prepare_vao_line(child_node):
    # prepare vertex data (in main memory)

    child_point = child_node.offset

    vertices = glm.array(glm.float32,
                         # position                                    # color
                         0, 0, 0,  1.0, 1.0, 1.0,  # y-axis start
                         child_point.x, child_point.y, child_point.z,  1.0, 1.0, 1.0,  # y-axis end
                         )

    # create and activate VAO (vertex array object)
    # create a vertex array object ID and store it to VAO variable
    VAO = glGenVertexArrays(1)
    glBindVertexArray(VAO)      # activate VAO

    # create and activate VBO (vertex buffer object)
    # create a buffer object ID and store it to VBO variable
    VBO = glGenBuffers(1)
    # activate VBO as a vertex buffer object
    glBindBuffer(GL_ARRAY_BUFFER, VBO)

    # copy vertex data to VBO
    # allocate GPU memory for and copy vertex data to the currently bound vertex buffer
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes,
                 vertices.ptr, GL_STATIC_DRAW)

    # configure vertex positions
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 *
                          glm.sizeof(glm.float32), None)
    glEnableVertexAttribArray(0)

    # configure vertex colors
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 *
                          glm.sizeof(glm.float32), ctypes.c_void_p(3*glm.sizeof(glm.float32)))
    glEnableVertexAttribArray(1)

    return VAO


def prepare_vao_box():
    # prepare vertex data (in main memory)
    # 8 vertices

    vertices = glm.array(glm.float32,
                         # position      normal
                         -1,  1,  1, -0.577,  0.577,  0.577,  # v0
                         1,  1,  1,  0.816,  0.408,  0.408,  # v1
                         1, -1,  1,  0.408, -0.408,  0.816,  # v2
                         -1, -1,  1, -0.408, -0.816,  0.408,  # v3
                         -1,  1, -1, -0.408,  0.408, -0.816,  # v4
                         1,  1, -1,  0.408,  0.816, -0.408,  # v5
                         1, -1, -1,  0.577, -0.577, -0.577,  # v6
                         -1, -1, -1, -0.816, -0.408, -0.408,  # v7
                         )

    # prepare index data
    # 12 triangles
    indices = glm.array(glm.uint32,
                        0, 2, 1,
                        0, 3, 2,
                        4, 5, 6,
                        4, 6, 7,
                        0, 1, 5,
                        0, 5, 4,
                        3, 6, 2,
                        3, 7, 6,
                        1, 2, 6,
                        1, 6, 5,
                        0, 7, 3,
                        0, 4, 7,
                        )

    # create and activate VAO (vertex array object)
    # create a vertex array object ID and store it to VAO variable
    VAO = glGenVertexArrays(1)
    glBindVertexArray(VAO)      # activate VAO

    # create and activate VBO (vertex buffer object)
    # create a buffer object ID and store it to VBO variable
    VBO = glGenBuffers(1)
    # activate VBO as a vertex buffer object
    glBindBuffer(GL_ARRAY_BUFFER, VBO)

    # create and activate EBO (element buffer object)
    # create a buffer object ID and store it to EBO variable
    EBO = glGenBuffers(1)
    # activate EBO as an element buffer object
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO)

    # copy vertex data to VBO
    # allocate GPU memory for and copy vertex data to the currently bound vertex buffer
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes,
                 vertices.ptr, GL_STATIC_DRAW)

    # copy index data to EBO
    # allocate GPU memory for and copy index data to the currently bound element buffer
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes,
                 indices.ptr, GL_STATIC_DRAW)

    # configure vertex positions
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 *
                          glm.sizeof(glm.float32), None)
    glEnableVertexAttribArray(0)

    # configure vertex normals
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 *
                          glm.sizeof(glm.float32), ctypes.c_void_p(3*glm.sizeof(glm.float32)))
    glEnableVertexAttribArray(1)

    return VAO


def draw_line(vao, MVP, MVP_loc):
    glBindVertexArray(vao)
    glUniformMatrix4fv(MVP_loc, 1, GL_FALSE, glm.value_ptr(MVP))
    glDrawArrays(GL_LINES, 0, 2)


def draw_node(vao, node, M, MVP, MVP_loc, M_loc, view_pos_loc, view_pos):
    glBindVertexArray(vao)
    glUniformMatrix4fv(MVP_loc, 1, GL_FALSE, glm.value_ptr(MVP))
    glUniformMatrix4fv(M_loc, 1, GL_FALSE, glm.value_ptr(M))
    glUniform3f(view_pos_loc, view_pos.x, view_pos.y, view_pos.z)
    glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_INT, None)


def translate_mat(chan, dis):
    if chan.upper() == 'XPOSITION':
        return glm.translate(glm.vec3(dis, 0, 0))
    elif chan.upper() == 'YPOSITION':
        return glm.translate(glm.vec3(0, dis, 0))
    elif chan.upper() == 'ZPOSITION':
        return glm.translate(glm.vec3(0, 0, dis))


def rotate_mat(chan, angle):
    angle = np.radians(angle)

    if chan.upper() == 'XROTATION':
        return glm.rotate(angle, glm.vec3(1, 0, 0))
    elif chan.upper() == 'YROTATION':
        return glm.rotate(angle, glm.vec3(0, 1, 0))
    elif chan.upper() == 'ZROTATION':
        return glm.rotate(angle, glm.vec3(0, 0, 1))


def root_translate_mat(node, chan, idx):
    translate1 = translate_mat(str(chan[0]), float(g_frames[idx][0]))
    translate2 = translate_mat(str(chan[1]), float(g_frames[idx][1]))
    translate3 = translate_mat(str(chan[2]), float(g_frames[idx][2]))
    return (translate1 * translate2 * translate3)


def node_rotate_mat(node, chan, idx, off):
    rotate1 = rotate_mat(str(chan[0]), float(g_frames[idx][6+off]))
    rotate2 = rotate_mat(str(chan[1]), float(g_frames[idx][7+off]))
    rotate3 = rotate_mat(str(chan[2]), float(g_frames[idx][8+off]))
    return (rotate1 * rotate2 * rotate3)


def main():
    global g_idx

    # initialize glfw
    if not glfwInit():
        return
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3)   # OpenGL 3.3
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3)
    # Do not allow legacy OpenGl API calls
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE)
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE)  # for macOS

    # create a window and OpenGL context
    window = glfwCreateWindow(1200, 1200, '2021097356', None, None)
    if not window:
        glfwTerminate()
        return
    glfwMakeContextCurrent(window)

    # register event callbacks
    glfwSetKeyCallback(window, key_callback)
    glfwSetScrollCallback(window, scroll_callback)
    glfwSetCursorPosCallback(window, cursor_pos_callback)
    glfwSetDropCallback(window, drop_callback)

    # load shaders
    shader_program = load_shaders(g_vertex_shader_src, g_fragment_shader_src)
    shader_lighting = load_shaders(
        g_vertex_shader_lighting, g_fragment_shader_lighting)

    # get uniform locations
    MVP_loc = glGetUniformLocation(shader_program, 'MVP')
    MVP_lighting_loc = glGetUniformLocation(shader_lighting, 'MVP')
    M_loc = glGetUniformLocation(shader_lighting, 'M')
    view_pos_loc = glGetUniformLocation(shader_lighting, 'view_pos')

    # prepare vaos
    vao_frame = prepare_vao_frame()
    vao_xz = prepare_vao_xz()

    glViewport(0, 0, 1200, 1200)

    g_idx = 0

    # scale size
    size = 0.04

    # loop until the user closes the window
    while not glfwWindowShouldClose(window):
        # render

        # enable depth test (we'll see details later)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glEnable(GL_DEPTH_TEST)

        glUseProgram(shader_program)

        # projection matrix
        global g_is_perspective
        if g_is_perspective:
            P = glm.perspective(45, 1, 1, 10000)
        else:
            P = glm.ortho(-.5 * g_cam_distance, .5 * g_cam_distance, -
                          .5 * g_cam_distance, .5 * g_cam_distance, -100, 100)

        # view matrix
        # rotate camera position with g_cam_azimuth / move camera up & down with g_cam_height
        view_pos = g_eye
        V = glm.lookAt(view_pos, g_at, g_up_vector)

        # current frame: P*V*I (now this is the world frame)
        I = glm.mat4()
        MVP = P*V*I
        glUniformMatrix4fv(MVP_loc, 1, GL_FALSE, glm.value_ptr(MVP))

        # draw current frame
        glBindVertexArray(vao_frame)
        glDrawArrays(GL_LINES, 0, 6)

        if g_root != None:
            g_root.update_tree_global_transform()

        # line rendering
        if g_line_rendering == True:
            if g_is_animating == False:
                for node in g_node_list:
                    MVP = P*V*node.global_transform
                    for node_child in node.children:
                        vao_line = prepare_vao_line(node_child)
                        draw_line(vao_line, MVP, MVP_loc)
            elif g_is_animating == True:
                # frame time이 지나면 frame을 바꿈
                g_idx = int((glfwGetTime() - g_pressed_time)
                            * int(1 / g_frame_time)) % len(g_frames)

                # root
                g_root.set_joint_transform(root_translate_mat(g_root, g_root.channels, g_idx) * rotate_mat(g_root.channels[3], float(
                    g_frames[g_idx][3])) * rotate_mat(g_root.channels[4], float(g_frames[g_idx][4])) * rotate_mat(g_root.channels[5], float(g_frames[g_idx][5])))

                # child
                off = 0
                for node in g_node_list:
                    # root는 이미 계산했으므로 skip
                    if node.parent == None:
                        continue
                    chan = node.channels

                    # end skip
                    if chan is None:
                        continue

                    node.set_joint_transform(
                        node_rotate_mat(node, chan, g_idx, off))
                    off += 3

                for node in g_node_list:
                    MVP = P*V*node.global_transform
                    for node_child in node.children:
                        vao_line = prepare_vao_line(node_child)
                        draw_line(vao_line, MVP, MVP_loc)

        # box rendering
        elif g_line_rendering == False:
            glUseProgram(shader_lighting)

            if g_is_animating == False:
                for node in g_node_list:
                    if node.parent == None:
                        continue
                    M = node.parent.global_transform
                    MVP = P*V*M

                    vao_box = prepare_vao_box()
                    v = glm.vec3(0, 0, 1)

                    offset = node.offset
                    dist = np.sqrt(np.dot(offset, offset))

                    transform_func = glm.mat4()

                    if not dist == 0:
                        new_frame = np.cross(v, offset/dist)
                        dist2 = np.sqrt(np.dot(new_frame, new_frame))
                        angle = np.arcsin(dist2)
                        transform_func *= glm.rotate(angle, new_frame)

                        transform_func *= glm.translate(
                            glm.vec3(0, 0, dist/2))
                        transform_func *= glm.scale(glm.vec3(size,
                                                    size, dist/2))

                        MVP *= transform_func
                        M *= transform_func

                        draw_node(vao_box, node, M, MVP, MVP_lighting_loc,
                                  M_loc, view_pos_loc, view_pos)
            elif g_is_animating == True:
                g_idx = int((glfwGetTime() - g_pressed_time)
                            * int(1 / g_frame_time)) % len(g_frames)

                g_root.set_joint_transform(
                    root_translate_mat(g_root, g_root.channels, g_idx) * rotate_mat(g_root.channels[3], float(
                        g_frames[g_idx][3])) * rotate_mat(g_root.channels[4], float(g_frames[g_idx][4])) * rotate_mat(g_root.channels[5], float(g_frames[g_idx][5])))

                off = 0
                for node in g_node_list:
                    # root node skip
                    if node.parent == None:
                        continue

                    chan = node.channels

                    # end effectors skip
                    if chan is None:
                        continue

                    # set joint transform
                    node.set_joint_transform(
                        node_rotate_mat(node, chan, g_idx, off))
                    off += 3

                for node in g_node_list:
                    if node.parent == None:
                        continue
                    M = node.parent.global_transform
                    MVP = P*V*M

                    vao_box = prepare_vao_box()
                    v = glm.vec3(0, 0, 1)

                    offset = node.offset
                    dist = np.sqrt(np.dot(offset, offset))

                    transform_func = glm.mat4()

                    if not dist == 0:
                        new_frame = np.cross(v, offset/dist)
                        dist2 = np.sqrt(np.dot(new_frame, new_frame))
                        angle = np.arcsin(dist2)
                        transform_func *= glm.rotate(angle, new_frame)

                        transform_func *= glm.translate(
                            glm.vec3(0, 0, dist/2))
                        transform_func *= glm.scale(glm.vec3(size,
                                                    size, dist/2))

                        MVP *= transform_func
                        M *= transform_func

                        draw_node(vao_box, node, M, MVP, MVP_lighting_loc,
                                  M_loc, view_pos_loc, view_pos)

        glUseProgram(shader_program)
        # draw x grid
        for i in range(-15, 16):
            T = glm.translate(glm.vec3(0, 0, i))
            M = T

            MVP = P*V*M
            glUniformMatrix4fv(MVP_loc, 1, GL_FALSE, glm.value_ptr(MVP))

            glBindVertexArray(vao_xz)
            glDrawArrays(GL_LINES, 0, 2)

        # draw z grid
        for i in range(-15, 16):
            T = glm.translate(glm.vec3(i, 0, 0))
            M = T

            MVP = P*V*M
            glUniformMatrix4fv(MVP_loc, 1, GL_FALSE, glm.value_ptr(MVP))

            glBindVertexArray(vao_xz)
            glDrawArrays(GL_LINES, 2, 4)

        # swap front and back buffers
        glfwSwapBuffers(window)

        # poll events
        glfwPollEvents()

    # terminate glfw
    glfwTerminate()


if __name__ == "__main__":
    main()
