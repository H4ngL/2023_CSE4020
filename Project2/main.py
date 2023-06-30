from OpenGL.GL import *
from glfw.GLFW import *
import glm
import ctypes
import numpy as np
import os

# 투영 설정
g_is_perspective = True

# rendering mode 설정
g_single_rendering = True

# solid/wireframe mode 설정
g_solid_mode = True

# 초기 각도 및 거리 설정
g_cam_azimuth = np.radians(45.)
g_cam_elevation = np.radians(45.)
g_cam_distance = 30.

# lookAt 함수의 인자
g_eye = glm.vec3(g_cam_distance * np.cos(g_cam_elevation) * np.sin(g_cam_azimuth), g_cam_distance * np.sin(g_cam_elevation), g_cam_distance *
                 np.cos(g_cam_elevation) * np.cos(g_cam_azimuth))
g_at = glm.vec3(0.0, 0.0, 0.0)
g_up_vector = glm.vec3(0, 1, 0)

# 마우스 이전 위치
g_prev_xpos = 0.
g_prev_ypos = 0.

# vertex positions, vertex normals, faces information 저장
g_vpos = []
g_vnormal = []
g_vertices = glm.array(glm.float32)

g_vertex_shader_src = '''
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

g_vertex_shader_src_grid = '''
# version 330 core

layout (location = 0) in vec3 vin_pos;
layout (location = 1) in vec3 vin_color;

out vec4 vout_color;

uniform mat4 MVP_grid;

void main()
{
    // 3D points in homogeneous coordinates
    vec4 p3D_in_hcoord = vec4(vin_pos.xyz, 1.0);

    gl_Position = MVP_grid * p3D_in_hcoord;

    vout_color = vec4(vin_color, 1.);
}
'''


g_fragment_shader_src = '''
#version 330 core

in vec3 vout_surface_pos;
in vec3 vout_normal;

out vec4 FragColor;

uniform vec3 view_pos;
uniform vec3 color;

void main()
{
    // light and material properties
    vec3 light_pos = vec3(3,10,0);
    vec3 light_pos2 = vec3(5,-10,-5);
    vec3 light_color = vec3(1,1,1);
    vec3 material_color = color;
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
    vec3 light_dir2 = normalize(light_pos2 - surface_pos);

    // diffuse
    float diff = max(dot(normal, light_dir), 0);
    vec3 diffuse = diff * light_diffuse * material_diffuse;
    float diff2 = max(dot(normal, light_dir2), 0);
    vec3 diffuse2 = diff2 * light_diffuse * material_diffuse;
    vec3 total_diffuse = diffuse + diffuse2;

    // specular
    vec3 view_dir = normalize(view_pos - surface_pos);
    vec3 reflect_dir = reflect(-light_dir, normal);
    float spec = pow( max(dot(view_dir, reflect_dir), 0.0), material_shininess);
    vec3 specular = spec * light_specular * material_specular;
    
    vec3 reflect_dir2 = reflect(-light_dir2, normal);
    float spec2 = pow( max(dot(view_dir, reflect_dir2), 0.0), material_shininess);
    vec3 specular2 = spec2 * light_specular * material_specular;

    vec3 color = ambient + total_diffuse + specular + specular2;
    FragColor = vec4(color, 1.);
}
'''

g_fragment_shader_src_grid = '''
# version 330 core

in vec4 vout_color;

out vec4 FragColor;

void main()
{

    FragColor = vout_color;

}
'''


class Node:
    def __init__(self, parent, shape_transform, color):
        # hierarchy
        self.parent = parent
        self.children = []
        if parent is not None:
            parent.children.append(self)

        # transform
        self.transform = glm.mat4()
        self.global_transform = glm.mat4()

        # shape
        self.shape_transform = shape_transform
        self.color = color

    def set_transform(self, transform):
        self.transform = transform

    def update_tree_global_transform(self):
        if self.parent is not None:
            self.global_transform = self.parent.get_global_transform() * self.transform
        else:
            self.global_transform = self.transform

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

    # check for shader compile errors
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

    # check for shader compile errors
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

    # check for linking errors
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
    global g_is_perspective, g_single_rendering, g_solid_mode
    if key == GLFW_KEY_ESCAPE and action == GLFW_PRESS:
        glfwSetWindowShouldClose(window, GLFW_TRUE)
    else:
        if action == GLFW_PRESS or action == GLFW_REPEAT:
            if key == GLFW_KEY_V:
                g_is_perspective = not g_is_perspective
            elif key == GLFW_KEY_H:
                g_single_rendering = False
            elif key == GLFW_KEY_Z:
                g_solid_mode = not g_solid_mode


def render_mode():
    global g_solid_mode
    if g_solid_mode:
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
    else:
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)


def drop_callback(window, paths):
    global g_single_rendering

    if g_single_rendering == False:
        g_single_rendering = True

    for path in paths:
        print("Dropped file:", path)
        if path.endswith(".obj"):
            parse_obj_file_line_by_line(path)


def parse_obj_file_line_by_line(path):
    global g_vpos, g_vnormal, g_vertices

    init_arr()

    for line in open(path, "r"):
        if line.startswith("v "):
            vertex = np.array(list(map(float, line.strip().split()[1:])))
            vertex = vertex.astype(np.float32)
            g_vpos.append(vertex)
        elif line.startswith("vn "):
            normal = np.array(list(map(float, line.strip().split()[1:])))
            normal = normal.astype(np.float32)
            g_vnormal.append(normal)
        elif line.startswith("f "):
            parse_f_line(line)

    g_vertices = glm.array(g_vertices, dtype=glm.float32)


def parse_f_line(line):
    global g_vertices

    split_line = line.strip().split()[1:]
    side = len(split_line)

    if side == 3:
        for token in split_line:
            nums = np.array(token.split('/'))
            length = len(nums)

            add_f_line(length, nums)
    elif side >= 4:
        must_have_vertex = split_line[0]
        must_have_vertex = np.array(must_have_vertex.split('/'))
        length = len(must_have_vertex)

        for i in range(1, side - 1):
            add_f_line(length, must_have_vertex)
            add_f_line(length, np.array(split_line[i].split('/')))
            add_f_line(length, np.array(split_line[i + 1].split('/')))


def add_f_line(length, nums):
    global g_vertices

    if length == 1:
        num = int(nums[0]) - 1
        g_vertices = np.append(g_vertices, np.array(
            g_vpos[num]).astype(np.float32))
    elif length == 2:
        num = int(nums[0]) - 1
        g_vertices = np.append(g_vertices, np.array(
            g_vpos[num]).astype(np.float32))
    elif length == 3:
        num1 = int(nums[0]) - 1
        num2 = int(nums[2]) - 1
        g_vertices = np.append(g_vertices, np.array(
            g_vpos[num1]).astype(np.float32))
        g_vertices = np.append(g_vertices, np.array(
            g_vnormal[num2]).astype(np.float32))


def init_arr():
    global g_vpos, g_vnormal, g_vertices

    g_vpos = []
    g_vnormal = []
    g_vertices = glm.array(glm.float32)


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


def prepare_vao_obj():
    # create and activate VAO
    VAO = glGenVertexArrays(1)
    glBindVertexArray(VAO)

    # create and activate VBO
    VBO = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, VBO)

    # copy vertex data to VBO
    glBufferData(GL_ARRAY_BUFFER, g_vertices.nbytes,
                 g_vertices.ptr, GL_STATIC_DRAW)

    # configure vertex positions
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 *
                          glm.sizeof(glm.float32), None)
    glEnableVertexAttribArray(0)

    # configure vertex normals
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 *
                          glm.sizeof(glm.float32), ctypes.c_void_p(3*glm.sizeof(glm.float32)))
    glEnableVertexAttribArray(1)

    return VAO


lake = []
mother_duck = []
baby_duck = []
lotus = []
lotus_leaf = []
butterfly = []


def prepare_vao_lake():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    data_path = os.path.join(dir_path, 'Animating_obj', 'lake.obj')

    parse_obj_file_line_by_line(data_path)

    lake.append(g_vertices)

    init_arr()

    # create and activate VAO
    VAO = glGenVertexArrays(1)
    glBindVertexArray(VAO)

    # create and activate VBO
    VBO = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, VBO)

    # copy vertex data to VBO
    glBufferData(GL_ARRAY_BUFFER, lake[0].nbytes,
                 lake[0].ptr, GL_STATIC_DRAW)

    # configure vertex positions
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 *
                          glm.sizeof(glm.float32), None)
    glEnableVertexAttribArray(0)

    # configure vertex normals
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 *
                          glm.sizeof(glm.float32), ctypes.c_void_p(3*glm.sizeof(glm.float32)))
    glEnableVertexAttribArray(1)

    return VAO


def prepare_vao_mother_duck():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    data_path = os.path.join(dir_path, 'Animating_obj', 'mother_duck.obj')

    parse_obj_file_line_by_line(data_path)

    mother_duck.append(g_vertices)

    init_arr()

    # create and activate VAO
    VAO = glGenVertexArrays(1)
    glBindVertexArray(VAO)

    # create and activate VBO
    VBO = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, VBO)

    # copy vertex data to VBO
    glBufferData(GL_ARRAY_BUFFER, mother_duck[0].nbytes,
                 mother_duck[0].ptr, GL_STATIC_DRAW)

    # configure vertex positions
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 *
                          glm.sizeof(glm.float32), None)
    glEnableVertexAttribArray(0)

    # configure vertex normals
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 *
                          glm.sizeof(glm.float32), ctypes.c_void_p(3*glm.sizeof(glm.float32)))
    glEnableVertexAttribArray(1)

    return VAO


def prepare_vao_baby_duck():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    data_path = os.path.join(dir_path, 'Animating_obj', 'baby_duck.obj')

    parse_obj_file_line_by_line(data_path)

    baby_duck.append(g_vertices)

    init_arr()

    # create and activate VAO
    VAO = glGenVertexArrays(1)
    glBindVertexArray(VAO)

    # create and activate VBO
    VBO = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, VBO)

    # copy vertex data to VBO
    glBufferData(GL_ARRAY_BUFFER, baby_duck[0].nbytes,
                 baby_duck[0].ptr, GL_STATIC_DRAW)

    # configure vertex positions
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 *
                          glm.sizeof(glm.float32), None)
    glEnableVertexAttribArray(0)

    # configure vertex normals
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 *
                          glm.sizeof(glm.float32), ctypes.c_void_p(3*glm.sizeof(glm.float32)))
    glEnableVertexAttribArray(1)

    return VAO


def prepare_vao_lotus():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    data_path = os.path.join(dir_path, 'Animating_obj', 'lotus.obj')

    parse_obj_file_line_by_line(data_path)

    # create and activate VAO
    VAO = glGenVertexArrays(1)
    glBindVertexArray(VAO)

    lotus.append(g_vertices)

    init_arr()

    # create and activate VBO
    VBO = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, VBO)

    # copy vertex data to VBO
    glBufferData(GL_ARRAY_BUFFER, lotus[0].nbytes,
                 lotus[0].ptr, GL_STATIC_DRAW)

    # configure vertex positions
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 *
                          glm.sizeof(glm.float32), None)
    glEnableVertexAttribArray(0)

    # configure vertex normals
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 *
                          glm.sizeof(glm.float32), ctypes.c_void_p(3*glm.sizeof(glm.float32)))
    glEnableVertexAttribArray(1)

    return VAO


def prepare_vao_lotus_leaf():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    data_path = os.path.join(dir_path, 'Animating_obj', 'lotus_leaf.obj')

    parse_obj_file_line_by_line(data_path)

    # create and activate VAO
    VAO = glGenVertexArrays(1)
    glBindVertexArray(VAO)

    lotus_leaf.append(g_vertices)

    init_arr()

    # create and activate VBO
    VBO = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, VBO)

    # copy vertex data to VBO
    glBufferData(GL_ARRAY_BUFFER, lotus_leaf[0].nbytes,
                 lotus_leaf[0].ptr, GL_STATIC_DRAW)

    # configure vertex positions
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 *
                          glm.sizeof(glm.float32), None)
    glEnableVertexAttribArray(0)

    # configure vertex normals
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 *
                          glm.sizeof(glm.float32), ctypes.c_void_p(3*glm.sizeof(glm.float32)))
    glEnableVertexAttribArray(1)

    return VAO


def prepare_vao_butterfly():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    data_path = os.path.join(dir_path, 'Animating_obj', 'butterfly.obj')

    parse_obj_file_line_by_line(data_path)

    # create and activate VAO
    VAO = glGenVertexArrays(1)
    glBindVertexArray(VAO)

    butterfly.append(g_vertices)

    init_arr()

    # create and activate VBO
    VBO = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, VBO)

    # copy vertex data to VBO
    glBufferData(GL_ARRAY_BUFFER, butterfly[0].nbytes,
                 butterfly[0].ptr, GL_STATIC_DRAW)

    # configure vertex positions
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 *
                          glm.sizeof(glm.float32), None)
    glEnableVertexAttribArray(0)

    # configure vertex normals
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 *
                          glm.sizeof(glm.float32), ctypes.c_void_p(3*glm.sizeof(glm.float32)))
    glEnableVertexAttribArray(1)

    return VAO


def draw_obj(vao, MVP, MVP_loc, length):
    glBindVertexArray(vao)
    glUniformMatrix4fv(MVP_loc, 1, GL_FALSE, glm.value_ptr(MVP))
    glDrawArrays(GL_TRIANGLES, 0, length)


def draw_node(vao, node, VP, MVP_loc, color_loc, length):
    MVP = VP * node.get_global_transform() * node.get_shape_transform()
    color = node.get_color()

    glBindVertexArray(vao)
    glUniformMatrix4fv(MVP_loc, 1, GL_FALSE, glm.value_ptr(MVP))
    glUniform3f(color_loc, color.r, color.g, color.b)
    glDrawArrays(GL_TRIANGLES, 0, length)


def main():
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
    shader_for_grid = load_shaders(
        g_vertex_shader_src_grid, g_fragment_shader_src_grid)

    # get uniform locations
    MVP_loc = glGetUniformLocation(shader_program, 'MVP')
    MVP_grid_loc = glGetUniformLocation(shader_for_grid, 'MVP_grid')
    M_loc = glGetUniformLocation(shader_program, 'M')
    view_pos_loc = glGetUniformLocation(shader_program, 'view_pos')
    color_loc = glGetUniformLocation(shader_program, 'color')

    # prepare vaos
    vao_frame = prepare_vao_frame()
    vao_xz = prepare_vao_xz()
    vao_lake = prepare_vao_lake()
    vao_mother_duck = prepare_vao_mother_duck()
    vao_baby_duck = prepare_vao_baby_duck()
    vao_lotus = prepare_vao_lotus()
    vao_lotus_leaf = prepare_vao_lotus_leaf()
    vao_butterfly = prepare_vao_butterfly()

    lake_node = Node(None, glm.scale((1, 1, 1)), glm.vec3(0.1, 0.6, 0.9))
    mother_duck_node = Node(lake_node, glm.translate(
        (8.5, 6, -8.5)) * glm.scale((0.04, 0.04, 0.04)), glm.vec3(1, 1, 1))
    baby_duck_node = Node(mother_duck_node, glm.rotate(-240, (0, 1, 0)) * glm.translate((0, 6, 0)) * glm.scale(
        (5., 5., 5.)), glm.vec3(1, 1, 0))
    baby_duck_node2 = Node(mother_duck_node, glm.rotate(-240, (0, 1, 0)) * glm.translate((0, 6, 0)) * glm.scale(
        (5., 5., 5.)), glm.vec3(1, 1, 0))
    baby_duck_node3 = Node(mother_duck_node, glm.rotate(-240, (0, 1, 0)) * glm.translate((0, 6, 0)) * glm.scale(
        (5., 5., 5.)), glm.vec3(1, 1, 0))
    lotus_node = Node(lake_node,  glm.translate((1, 6, 0)) * glm.rotate(np.radians(-90), (1, 0, 0)) * glm.scale(
        (.5, .5, .5)), glm.vec3(1, .7, .8))
    lotus_leaf_node = Node(lotus_node, glm.rotate(np.radians(-120), (0, 1, 0)) * glm.translate(
        (10.5, 6.2, 6.5)) * glm.scale((0.2, 0.2, -0.2)), glm.vec3(0.1, 0.9, 0.1))
    butterfly_node = Node(lotus_node, glm.scale(
        (.5, .5, .5)), glm.vec3(0.72, 0.95, 1.0))

    glViewport(0, 0, 1200, 1200)

    # loop until the user closes the window
    while not glfwWindowShouldClose(window):
        # render

        # enable depth test (we'll see details later)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glEnable(GL_DEPTH_TEST)

        glUseProgram(shader_program)

        # change rendering mode
        render_mode()

        # projection matrix
        global g_is_perspective
        if g_is_perspective:
            P = glm.perspective(45, 1, 1, 10000)
        else:
            P = glm.ortho(-.5 * g_cam_distance, .5 * g_cam_distance, -
                          .5 * g_cam_distance, .5 * g_cam_distance, -10000, 10000)

        # view matrix
        # rotate camera position with g_cam_azimuth / move camera up & down with g_cam_height
        view_pos = g_eye
        V = glm.lookAt(view_pos, g_at, g_up_vector)

        M = glm.mat4()
        MVP = P*V*M
        glUniformMatrix4fv(MVP_loc, 1, GL_FALSE, glm.value_ptr(MVP))
        glUniformMatrix4fv(M_loc, 1, GL_FALSE, glm.value_ptr(M))
        glUniform3f(view_pos_loc, view_pos.x, view_pos.y, view_pos.z)

        # current frame: P*V*I (now this is the world frame)
        glUseProgram(shader_for_grid)
        I = glm.mat4()
        MVP_grid = P*V*I
        glUniformMatrix4fv(MVP_grid_loc, 1, GL_FALSE, glm.value_ptr(MVP_grid))

        glBindVertexArray(vao_frame)
        glDrawArrays(GL_LINES, 0, 6)

        for i in range(-15, 16):
            T = glm.translate(glm.vec3(0, 0, i))

            MVP_grid = P*V*T
            glUniformMatrix4fv(MVP_grid_loc, 1, GL_FALSE,
                               glm.value_ptr(MVP_grid))

            glBindVertexArray(vao_xz)
            glDrawArrays(GL_LINES, 0, 2)

        # draw z grid
        for i in range(-15, 16):
            T = glm.translate(glm.vec3(i, 0, 0))

            MVP_grid = P*V*T
            glUniformMatrix4fv(MVP_grid_loc, 1, GL_FALSE,
                               glm.value_ptr(MVP_grid))

            glBindVertexArray(vao_xz)
            glDrawArrays(GL_LINES, 2, 4)

        glUseProgram(shader_program)
        vao_obj = prepare_vao_obj()

        # draw obj
        if g_single_rendering:
            color = glm.vec3(1, 1, 1)
            glUniform3f(color_loc, color.r, color.g, color.b)
            draw_obj(vao_obj, MVP, MVP_loc, len(g_vertices))
        else:
            t = glfwGetTime()

            # set local transformations of each node
            lake_node.set_transform(glm.translate((0, glm.sin(t * 0.5), 0)))

            r = 5
            theta = t * .4

            x = r * np.cos(theta)
            z = r * np.sin(theta)

            mother_duck_node.set_transform(glm.translate((
                x, 0, z,)) * glm.rotate(-t * .4, (0, 1, 0)) * glm.translate((0, glm.sin(t * 8)*.1 - .1, 0)))
            baby_duck_node.set_transform(glm.translate(
                (-1, 0, -2)) * glm.translate((0, glm.sin(t * 2)*.025 - .025, 0)))
            baby_duck_node2.set_transform(glm.translate(
                (.7, 0, -2.5)) * glm.translate((0, glm.sin(t * 2)*.025 - .025, 0)))
            baby_duck_node3.set_transform(glm.translate(
                (-.2, 0, -4)) * glm.translate((0, glm.sin(t * 2)*.025 - .025, 0)))
            lotus_node.set_transform(glm.rotate(t * .2, (0, 1, 0)))
            lotus_leaf_node.set_transform(glm.translate((1, 0, 1)))

            angle = t * 50
            rotate_x = np.sin(np.radians(angle)) * 45
            rotate_y = np.sin(np.radians(angle)) * 45

            rotate_matrix = glm.rotate(glm.radians(rotate_y), glm.vec3(
                0, 1, 0)) * glm.rotate(glm.radians(rotate_x), glm.vec3(1, 0, 0))
            butterfly_node.set_transform(
                glm.translate((1, 7, 0)) * rotate_matrix)

            # recursively update global transformations of all nodes
            lake_node.update_tree_global_transform()

            # draw nodes
            draw_node(vao_lake, lake_node, P*V,
                      MVP_loc, color_loc, len(lake[0]))
            draw_node(vao_mother_duck, mother_duck_node, P*V,
                      MVP_loc, color_loc, len(mother_duck[0]))
            draw_node(vao_baby_duck, baby_duck_node, P*V,
                      MVP_loc, color_loc, len(baby_duck[0]))
            draw_node(vao_baby_duck, baby_duck_node2, P*V,
                      MVP_loc, color_loc, len(baby_duck[0]))
            draw_node(vao_baby_duck, baby_duck_node3, P*V,
                      MVP_loc, color_loc, len(baby_duck[0]))
            draw_node(vao_lotus, lotus_node, P*V,
                      MVP_loc, color_loc, len(lotus[0]))
            draw_node(vao_lotus_leaf, lotus_leaf_node, P*V,
                      MVP_loc, color_loc, len(lotus_leaf[0]))
            draw_node(vao_butterfly, butterfly_node, P*V,
                      MVP_loc, color_loc, len(butterfly[0]))

        # swap front and back buffers
        glfwSwapBuffers(window)

        # poll events
        glfwPollEvents()

    # terminate glfw
    glfwTerminate()


if __name__ == "__main__":
    main()
