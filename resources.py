# coding=utf-8
""" Módulo que contiene las clases y objetos relacionados al jugador, entidades, cámara y
mecánicas de la aplicación, en resumen, todo lo que no tiene que ver con el apartado de modelos
geométricos ni la parte gráfica """

from OpenGL.GL import *
from sys import path_importer_cache
import glfw
import numpy as np
import grafica.transformations as tr
import grafica.easy_shaders as es
import grafica.scene_graph as sg
import grafica.basic_shapes as bs
import shapes3d as sh
from numpy import random as rd

# CLASE CON LOS ELEMENTOS DE LA MESA
class Bola:
    """ Las bolas de billar, la clase permite manejar la posición y velocidad de cada bola, calcular sus colisiones y físicas."""
    def __init__(self, pipeline, position):
        # Figura:
        self.model = None
        self.diam = 0.051 # Diametro de las bolas (Para todas las bolas de número, la bola blanca mide 0.048)
        self.pipeline = pipeline # Pipeline de la bola
        self.position = position
        self.velocity = 0 # Las pelotas siempre comienzan con velocidad 0

    def setModel(self, model):
        # Define la gpushape de la bola
        self.model = model

    def move(self, pos, vel):
        # Se modifican la velocidad y posición
        self.position = pos
        self.velocity = vel

    def draw(self):
        # Dibujar la bola
        glUniformMatrix4fv(glGetUniformLocation(self.pipeline.shaderProgram, "model"), 1, GL_TRUE, tr.matmul([
            tr.translate(self.position[0], self.position[1], self.position[2]),
            tr.uniformScale(self.diam/2)
        ])
        )
        self.pipeline.drawCall(self.model)

class MESA:
    """ La mesa de billar, clase que permite calcular el tamaño de esta, su colision con los bordes, y detectar cuando una bola entre en los bolsillos para considerar el puntaje"""
    def __init__(self, pipeline):
        # Figura
        self.pipeline = pipeline
        # La posición está fija en el 0,0,0
        self.tamaño = (2.12, 1.06)
        self.amortiguador = 0.0365 # Grosor de los amortiguadores
        self.bolsillos = 0.06 # Diametro de los bolsillos

        self.model = sh.createNormalTexTable(pipeline, self.tamaño[0], self.tamaño[1], self.amortiguador, self.bolsillos)

    def draw(self):
        # Dibujar la mesa completa
        sg.drawSceneGraphNode(self.model, self.pipeline, "model")


# Cámara en tercera persona
class ThirdCamera:
    def __init__(self, x, y , z):
        self.at = np.array([x, y, z+0.1])
        self.theta = -np.pi/2
        self.eye = np.array([x, y - 3.0, z + 0.1])
        self.up = np.array([0, 0, 1])

    # Determina el ángulo theta
    def set_theta(self, theta):
        self.theta = theta

    # Actualiza la matriz de vista y la retorna
    def update_view(self):
        self.eye[0] = 3 * np.cos(self.theta) + self.at[0]
        self.eye[1] = 3 * np.sin(self.theta) + self.at[1]
        self.eye[2] = self.at[2]

        viewMatrix = tr.lookAt(
            self.eye,
            self.at,
            self.up
        )
        return viewMatrix

class FirstCamera:
    def __init__(self, x, y, z):
        self.at = np.array([x, y + 3.0, z + 0.0])
        self.theta = -np.pi/2
        self.phi = np.pi/2
        self.eye = np.array([x, y, z + 0.0])
        self.up = np.array([0, 0, 1])

    # Determina el ángulo theta
    def set_theta(self, theta):
        self.theta = theta + np.pi
    
    def set_phi(self, phi):
        self.phi = phi

    # Actualiza la matriz de vista y la retorna
    def update_view(self):
        self.at[0] = 3*np.cos(self.theta) * np.sin(self.phi) + self.eye[0]
        self.at[1] = 3*np.sin(self.theta) * np.sin(self.phi) + self.eye[1]
        self.at[2] = 3*np.cos(self.phi) + self.eye[2]

        viewMatrix = tr.lookAt(
            self.eye,
            self.at,
            self.up
        )
        return viewMatrix
        

    
# Clase del controlador, tiene distintos parámetros que son utilizados para albergar la información de lo que ocurre
# en la aplicación.
class Controller:
    def __init__(self, width, height):
        self.fillPolygon = True
        self.waterEffect = False
        self.width = width
        self.height = height

        self.is_a_pressed = True
        self.is_t_pressed = True

        self.camera = FirstCamera(0, 0, 2.5)
        self.camara = 1

        self.light = 3

        self.reset = False
        self.empezar = False

        self.leftClickOn = False
        self.rightClickOn = False
        self.mousePos = (0.0, 0.0)

    # Función que retorna la cámara que se está utilizando
    def get_camera(self):
        return self.camera

    # Función que entrega la posición del vector at
    def getAtCamera(self):
        return self.camera.at
    
    # Función que obtiene la posición del vector eye
    def getEyeCamera(self):
        return self.camera.eye

    # Función que entrega el ángulo theta
    def getThetaCamera(self):
        return self.camera.theta

    # Función que detecta que tecla se está presionando
    def on_key(self, window, key, scancode, action, mods):
        
        if action == glfw.PRESS:
            if key == glfw.KEY_SPACE:
                self.fillPolygon = not self.fillPolygon

            if key == glfw.KEY_ESCAPE:
                glfw.set_window_should_close(window, True)

            if key == glfw.KEY_LEFT_CONTROL:
                self.waterEffect = not self.waterEffect

            if key == glfw.KEY_A:
                self.is_a_pressed = not self.is_a_pressed
            
            if key == glfw.KEY_T:
                self.is_t_pressed = not self.is_t_pressed

            if key == glfw.KEY_1:
                self.light = 1
        
            if key == glfw.KEY_2:
                self.light = 2

            if key == glfw.KEY_3:
                self.light = 3

            if key == glfw.KEY_4:
                self.light = 4
            
            if key == glfw.KEY_UP:
                self.reset = True
                self.empezar = True

    # Función que obtiene las coordenadas de la posición del mouse y las traduce en coordenadas de openGL
    def cursor_pos_callback(self, window, x, y):
        mousePosX = 2 * (x - self.width/2) / self.width

        if y<0:
            glfw.set_cursor_pos(window, x, 0)
        elif y>self.height:
            glfw.set_cursor_pos(window, x, self.height)


        mousePosY = 2 * (y - self.height/2) / self.height

        self.mousePos = (mousePosX, mousePosY)

    # Función que identifica si los botones del mouse son presionados
    def mouse_button_callback(self, window, button, action, mods):
        """
        glfw.MOUSE_BUTTON_1: left click
        glfw.MOUSE_BUTTON_2: right click
        glfw.MOUSE_BUTTON_3: scroll click
        """

        if (action == glfw.PRESS or action == glfw.REPEAT):
            if (button == glfw.MOUSE_BUTTON_1):
                self.leftClickOn = True

            if (button == glfw.MOUSE_BUTTON_2):
                self.rightClickOn = True

            if (button == glfw.MOUSE_BUTTON_3):
                pass

        elif (action ==glfw.RELEASE):
            if (button == glfw.MOUSE_BUTTON_1):
                self.leftClickOn = False
            if (button == glfw.MOUSE_BUTTON_2):
                self.rightClickOn = False

    #Funcion que recibe el input para manejar la camara y el tipo de esta, incluye ek movimiento del personaje
    def update_camera(self, delta):
        # Selecciona la cámara a utilizar
        if self.is_a_pressed and self.camara != 1:
            x = self.camera.at[0]
            y = self.camera.at[1]
            z = self.camera.at[2]-0.1
            self.camera = FirstCamera(x, y, z)
            self.camara = 1
        elif not self.is_a_pressed and self.camara != 3:
            x = self.camera.eye[0]
            y = self.camera.eye[1]
            z = self.camera.eye[2]
            self.camera = ThirdCamera(x, y, z)
            self.camara = 3

        direction = self.camera.at - self.camera.eye
        theta = -self.mousePos[0] * 2 * np.pi - np.pi/2

        mouseY = self.mousePos[1]
        phi = mouseY * (np.pi/2-0.01) + np.pi/2

        if self.camara == 3:
            if self.leftClickOn:
                self.camera.at += direction * delta

            if self.rightClickOn:
                self.camera.at -= direction * delta

        elif self.camara == 1:
            if self.leftClickOn:
                self.camera.eye += direction * delta

            if self.rightClickOn:
                self.camera.eye -= direction * delta
            self.camera.set_phi(phi)

        self.camera.set_theta(theta)

    def collision(self, cargas):
        # Funcion para detectar las colisiones con las cargas

        # Se recorren las cargas 
        for carga in cargas:
            # si la distancia a la carga es menor que la suma de los radios ha ocurrido en la colision
            if (self.radio+carga.radio)**2 > ((self.pos[0]- carga.pos[0])**2 + (self.pos[1]-carga.pos[1])**2):
                self.reset = True
                return

# Clase iluminación, crea los parámetros y las funciones para inicializar los shaders con normales.
class Iluminacion:
    def __init__(self):
        # Características de la luz por defecto
        self.LightPower = 0.6
        self.lightConcentration =20
        self.lightShininess = 1
        self.constantAttenuation = 0.01
        self.linearAttenuation = 0.03
        self.quadraticAttenuation = 0.04
        self.pipeline = None

    # fija los nuevos datos para la luz
    def setLight(self, Power, Concentration, Shininess, Attenuation):
        self.LightPower = Power
        self.lightConcentration = Concentration
        self.lightShininess = Shininess
        self.constantAttenuation = Attenuation[0]
        self.linearAttenuation = Attenuation[1]
        self.quadraticAttenuation = Attenuation[2]

    # Actualiza los parámetros de la luz al shader
    def updateLight(self, Pipeline, Pos, Direction, Camera):
        # Se guardan los variables
        LightPower = self.LightPower
        lightShininess = self.lightShininess
        lightConcentration = self.lightConcentration
        constantAttenuation = self.constantAttenuation
        linearAttenuation = self.linearAttenuation
        quadraticAttenuation = self.quadraticAttenuation
        # Se activa el program
        glUseProgram(Pipeline.shaderProgram)
        self.pipeline = Pipeline # Se guarda el shader utilizado
        # Se envían los uniforms
        glUniform3fv(glGetUniformLocation(Pipeline.shaderProgram, "lightPos"), 1, Pos)
        glUniform3f(glGetUniformLocation(Pipeline.shaderProgram, "La"), 0.35, 0.35, 0.35)
        glUniform3f(glGetUniformLocation(Pipeline.shaderProgram, "Ld"), LightPower, LightPower, LightPower)
        glUniform3f(glGetUniformLocation(Pipeline.shaderProgram, "Ls"), lightShininess, lightShininess, lightShininess)

        glUniform3f(glGetUniformLocation(Pipeline.shaderProgram, "viewPosition"), Camera[0], Camera[1], Camera[2])
        glUniform1ui(glGetUniformLocation(Pipeline.shaderProgram, "shininess"), 100)
        glUniform1ui(glGetUniformLocation(Pipeline.shaderProgram, "concentration"), lightConcentration)
        glUniform3f(glGetUniformLocation(Pipeline.shaderProgram, "lightDirection"), Direction[0], Direction[1], Direction[2])

        glUniform1f(glGetUniformLocation(Pipeline.shaderProgram, "constantAttenuation"), constantAttenuation)
        glUniform1f(glGetUniformLocation(Pipeline.shaderProgram, "linearAttenuation"), linearAttenuation)
        glUniform1f(glGetUniformLocation(Pipeline.shaderProgram, "quadraticAttenuation"), quadraticAttenuation)

    def addLight(self, i, pos, r, g, b):
        " Agregar las luces al shader, donde i indica que luz corresponde (del 0 al 3), pos es la posición de esta luz y r g b sus colores"
        strPos = "lightPos" + str(i)
        strLa = "La" + str(i)
        strLd = "Ld" + str(i)
        strLs = "Ls" + str(i)
        Pipeline = self.pipeline
        # Se envían los uniforms
        glUniform3fv(glGetUniformLocation(Pipeline.shaderProgram, strPos), 1, pos)
        glUniform3f(glGetUniformLocation(Pipeline.shaderProgram, strLa), 0.1, 0.1, 0.1)
        glUniform3f(glGetUniformLocation(Pipeline.shaderProgram, strLd), r*0.7, g*0.7, b*0.7)
        glUniform3f(glGetUniformLocation(Pipeline.shaderProgram, strLs), r, g, b)      


# Funciones


def calculateNormal(mesh):
    """ om.mesh() -> 
    Recibe el mesh, utilizando la estructura halfedge calcula la normal de las caras
    adyacentes y los promedia para conseguir su norma.
    Por ahora no tiene en cuenta los cálculos previamente hechos :s"""

    # Se activa la propiedad de agregar normales en las caras, sin embargo, no se utilizará el método de openmesh para
    # calcular dichas normales, sino se implementará una función propia para utilizar la estructura half-edge y simplemente
    # utiizar dicho espacio para guardar el vector normal resultante.
    mesh.request_face_normals()
    # Se calcula la normal de cada cara
    for face in mesh.faces():
        vertices = list(mesh.fv(face)) # Se obtiene los vértices de la cara
        P0 = np.array(mesh.point(vertices[0]))    # Se obtiene la coordenada del vértice 1
        P1 = np.array(mesh.point(vertices[1]))    # Se obtiene la coordenada del vértice 2
        P2 = np.array(mesh.point(vertices[2]))    # Se obtiene la coordenada del vértice 3
        dir1 = P0 - P1          # Calcula el vector que va desde el primer vértice al segundo
        dir2 = P2 - P1          # Calcula el vector que va desde el tercer vértice al segundo
        cruz = np.cross(dir2, dir1)     # Obtiene la normal de la cara
        mesh.set_normal(face, cruz/np.linalg.norm(cruz))    # Se guarda la normal normalizada como atributo en la cara 



def readFaceVertex(faceDescription):

    aux = faceDescription.split('/')

    assert len(aux[0]), "Vertex index has not been defined."

    faceVertex = [int(aux[0]), None, None]

    assert len(aux) == 2, "Only faces where its vertices require 3 indices are defined."

    if len(aux[1]) != 0:
        faceVertex[1] = int(aux[1])

    #if len(aux[2]) != 0:
     #   faceVertex[2] = int(aux[2])

    return faceVertex



def readOBJ(filename, color = None):

    vertices = []
    normals = []
    textCoords= []
    faces = []

    with open(filename, 'r') as file:
        for line in file.readlines():
            aux = line.strip().split(' ')
            
            if aux[0] == 'v':
                vertices += [[float(coord) for coord in aux[1:]]]

            elif aux[0] == 'vn':
                normals += [[float(coord) for coord in aux[1:]]]

            elif aux[0] == 'vt':
                assert len(aux[1:]) == 2, "Texture coordinates with different than 2 dimensions are not supported"
                textCoords += [[float(coord) for coord in aux[1:]]]

            elif aux[0] == 'f':
                N = len(aux)                
                faces += [[readFaceVertex(faceVertex) for faceVertex in aux[1:4]]]
                for i in range(3, N-1):
                    faces += [[readFaceVertex(faceVertex) for faceVertex in [aux[i], aux[i+1], aux[1]]]]

        vertexData = []
        indices = []
        index = 0

        # Per previous construction, each face is a triangle
        if color is None:
            for face in faces:

                # Checking each of the triangle vertices
                for i in range(0,3):
                    vertex = vertices[face[i][0]-1]
                    texture = textCoords[face[i][1]-1]
                    normal = normals[face[i][2]-1]

                    vertexData += [
                        vertex[0], vertex[1], vertex[2],
                        texture[0], texture[1],
                        normal[0], normal[1], normal[2]
                    ]

                # Connecting the 3 vertices to create a triangle
                indices += [index, index + 1, index + 2]
                index += 3        

            return bs.Shape(vertexData, indices)

        if color==1:
            for face in faces:

                # Checking each of the triangle vertices
                for i in range(0,3):
                    vertex = vertices[face[i][0]-1]
                    texture = textCoords[face[i][1]-1]

                    vertexData += [
                        vertex[0], vertex[1], vertex[2],
                        texture[0], texture[1],
                    ]

                # Connecting the 3 vertices to create a triangle
                indices += [index, index + 1, index + 2]
                index += 3        

            return bs.Shape(vertexData, indices)

        for face in faces:

            # Checking each of the triangle vertices
            for i in range(0,3):
                vertex = vertices[face[i][0]-1]
                normal = normals[face[i][2]-1]

                vertexData += [
                    vertex[0], vertex[1], vertex[2],
                    color[0], color[1], color[2],
                    normal[0], normal[1], normal[2]
                ]

            # Connecting the 3 vertices to create a triangle
            indices += [index, index + 1, index + 2]
            index += 3        

        return bs.Shape(vertexData, indices)
