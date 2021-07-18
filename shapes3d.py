"""Funciones para crear distintas figuras y escenas en 3D """

from numpy.lib.function_base import append
from grafica.gpu_shape import GPUShape
import openmesh as om
import numpy as np
import numpy.random as rd
from OpenGL.GL import *
import grafica.basic_shapes as bs
import grafica.easy_shaders as es
import grafica.transformations as tr
import grafica.scene_graph as sg
import sys, os.path
from resources import *
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
thisFilePath = os.path.abspath(__file__)
thisFolderPath = os.path.dirname(thisFilePath)
assetsDirectory = os.path.join(thisFolderPath, "sprites")
waterPath = os.path.join(assetsDirectory, "water.png")
displacementPath = os.path.join(assetsDirectory, "displacement.png")

boat1 = os.path.join(assetsDirectory, "boat1.obj")
boat2 = os.path.join(assetsDirectory, "boat2.obj")
boat3 = os.path.join(assetsDirectory, "boat3.obj")
wood1 = os.path.join(assetsDirectory, "wood1.jpg")
norm1 = os.path.join(assetsDirectory, "wood1_NRM.jpg")
wood2 = os.path.join(assetsDirectory, "wood2.jpg")
norm2 = os.path.join(assetsDirectory, "wood2_NRM.jpg")
wood3 = os.path.join(assetsDirectory, "wood3.jpg")
norm3 = os.path.join(assetsDirectory, "wood3_NRM.jpg")

# Convenience function to ease initialization
def createGPUShape(pipeline, shape):
    gpuShape = es.GPUShape().initBuffers()
    pipeline.setupVAO(gpuShape)
    gpuShape.fillBuffers(shape.vertices, shape.indices, GL_STATIC_DRAW)
    return gpuShape

def createTextureGPUShape(shape, pipeline, path, sWrapMode=GL_CLAMP_TO_EDGE, tWrapMode=GL_CLAMP_TO_EDGE, minFilterMode=GL_NEAREST, maxFilterMode=GL_NEAREST, mode=GL_STATIC_DRAW):
    # Funcion Conveniente para facilitar la inicializacion de un GPUShape con texturas
    gpuShape = es.GPUShape().initBuffers()
    pipeline.setupVAO(gpuShape)
    gpuShape.fillBuffers(shape.vertices, shape.indices, mode)
    gpuShape.texture = es.textureSimpleSetup(
        path, sWrapMode, tWrapMode, minFilterMode, maxFilterMode)
    return gpuShape

def createMultipleTextureGPUShape(shape, pipeline, paths, sWrapMode=GL_CLAMP_TO_EDGE, tWrapMode=GL_CLAMP_TO_EDGE, minFilterMode=GL_NEAREST, maxFilterMode=GL_NEAREST, mode=GL_STATIC_DRAW):
    # Funcion Conveniente para facilitar la inicializacion de un GPUShape con texturas
    Cantidad = len(paths)
    gpuShape = es.GPUShapeMulti(Cantidad).initBuffers()
    pipeline.setupVAO(gpuShape)
    gpuShape.fillBuffers(shape.vertices, shape.indices, mode)
    for i in range(Cantidad):
        gpuShape.texture.append( es.textureSimpleSetup(
            paths[i], sWrapMode, tWrapMode, minFilterMode, maxFilterMode))
    return gpuShape

def generateT(t):
    "Entrega un vector tiempo"   
    return np.array([[1, t, t**2, t**3]]).T

def Curve(typeCurve, V1, V2, V3, V4, N):
    """ str np.array np.array np.array np.array int -> np.ndarray((N,3))
    Función que crea una curva con los 4 vectores claves para la parametrización y entrega la curva en cuestión.
    Las curvas compatibles son:
    "Hermite", curva que recibe P1, P2, T1, T2, punto inicial y final y sus tengentes.
    "Bezier", curva que recibe P1, P2, P3, P4, punto inicial, intermedio 1 y y2 y la final.
    "CatmullRom", curva que recibe P0, P1, P2, P3, punto anterior, inicial, final, y después"""
    # Se crean N puntos a evaluar entre 0 y 1
    ts = np.linspace(0.0, 1.0, N)

    # Genera la matriz que contiene los puntos y tangentes claves de la curva
    G = np.concatenate((V1, V2, V3, V4), axis=1)
    
    # Se crera la matriz de la curva constante que da la forma:
    if typeCurve == "Hermite":
        M = np.array([[1, 0, -3, 2], [0, 0, 3, -2], [0, 1, -2, 1], [0, 0, -1, 1]])
    elif typeCurve == "Bezier":
        M = np.array([[1, -3, 3, -1], [0, 3, -6, 3], [0, 0, 3, -3], [0, 0, 0, 1]])
    elif typeCurve == "CatmullRom":
        M = np.array([[0, -0.5, 1, -0.5], [1, 0, -2.5, 1.5], [0, 0.5, 2, -1.5], [0, 0, -0.5, 0.5]])
    else:
        print ("No se reconoce la curva para los vectores:", V1, V2, V3, V4)
        assert False

    #Se crea la matriz de la curva
    M = np.matmul(G, M)
    
    # Se crea la curva:
    curve = np.ndarray((N,3))

    # Se evalua cada punto de la curva
    for i in range(len(ts)):
        T = generateT(ts[i])
        curve[i, 0:3] = np.matmul(M, T).T

    return curve

def Curveposition(typeCurve, V1, V2, V3, V4, t):
    """ Función similar a la anterior, salvo que en vez de generar todas las posiciones de la curva, genera la posición del instante de tiempo pedido
    Las curvas compatibles son:
    "Hermite", curva que recibe P1, P2, T1, T2, punto inicial y final y sus tengentes.
    "Bezier", curva que recibe P1, P2, P3, P4, punto inicial, intermedio 1 y y2 y la final.
    "CatmullRom", curva que recibe P0, P1, P2, P3, punto anterior, inicial, final, y después"""
    # Genera la matriz que contiene los puntos y tangentes claves de la curva
    G = np.concatenate((V1, V2, V3, V4), axis=1)
    
    # Se crera la matriz de la curva constante que da la forma:
    if typeCurve == "Hermite":
        M = np.array([[1, 0, -3, 2], [0, 0, 3, -2], [0, 1, -2, 1], [0, 0, -1, 1]])
    elif typeCurve == "Bezier":
        M = np.array([[1, -3, 3, -1], [0, 3, -6, 3], [0, 0, 3, -3], [0, 0, 0, 1]])
    elif typeCurve == "CatmullRom":
        M = np.array([[0, -0.5, 1, -0.5], [1, 0, -2.5, 1.5], [0, 0.5, 2, -1.5], [0, 0, -0.5, 0.5]])
    else:
        print ("No se reconoce la curva para los vectores:", V1, V2, V3, V4)
        assert False
    
    # se evalua en el tiempo correspondiente
    T = generateT(t)
    curve = np.matmul(G, M)
    return np.matmul(curve, T).T



######################################################
# Todo esto era para la tarea de cuevas
"""
######## CREANDO UNA MALLA FRACTAL #####
def fractalMesh(mesh, n):
    k = 0
    while k<n:
        newMesh = om.TriMesh()
        for face in mesh.faces():
            vertexs = list(mesh.fv(face))
            vertex1 = np.array(mesh.point(vertexs[0]))
            vertex2 = np.array(mesh.point(vertexs[1]))
            vertex3 = np.array(mesh.point(vertexs[2]))

            vertex12 = (vertex1 + vertex2)/2
            vertex23 = (vertex2 + vertex3)/2
            vertex13 = (vertex1 + vertex3)/2

            vertex12[2] = (rd.rand()-0.5)*0.1 + vertex12[2]
            vertex23[2] = (rd.rand()-0.5)*0.1 + vertex23[2]
            vertex13[2] = (rd.rand()-0.5)*0.1 + vertex13[2]

            v1 = newMesh.add_vertex(vertex1)
            v12 = newMesh.add_vertex(vertex12)
            v2 = newMesh.add_vertex(vertex2)
            v23 = newMesh.add_vertex(vertex23)
            v3 = newMesh.add_vertex(vertex3)
            v13 = newMesh.add_vertex(vertex13)

            newMesh.add_face(v1, v12, v13)
            newMesh.add_face(v12, v13, v23)
            newMesh.add_face(v13, v23, v3)
            newMesh.add_face(v12, v2, v23)
            #newMesh.add_face(v1, v2, v3)

        mesh = newMesh
        k+=1
        del newMesh

    return mesh


def caveMesh(matriz):
    Se crea las 2 mallas de polígonos correspondiente al suelo y el techo, por conveniencia, se utilizarán celdas
    de 5x5 metros cuadrados. (Considerando que los ejes se encontraran efectivamente en metros)
    De esta manera, Lara será capaz de moverse por la celda.
    suelo = om.TriMesh()
    techo = om.TriMesh()
    # Se obtienen las dimensiones de la matriz
    (N, M, k) = matriz.shape
    n= N//2
    m= M//2
    # Se crean arreglos que corresponderan al eje x e y de la cueva, de N+1 y M+1 vértices cada uno, de modo que
    # cada celda de la matriz sea generada por un cuadrado de 4 vértices
    if N%2!=0:
        xs = np.linspace(-3*N-n, 3*N+n, N*7)
    else:
        xs = np.linspace(-3*N-n, 3*N+n-1, N*7)
    if M%2!=0:
        ys = np.linspace(-3*M-m, 3*M+m, M*7)
    else:
        ys = np.linspace(-3*M-m, 3*M+m-1, M*7)

    

    # largo de arregles
    lenXS = len(xs)-1
    lenYS = len(ys)-1

    # Se generan los vértices de la malla, utilizando las alturas dadas
    for i in range(lenXS):
        x = xs[i]
        im = i//7   # Transforma el índice en su correspondiente celda de la matriz
        a = False
        for j in range(lenYS):
            y = ys[j]
            jm = j//7 # Transforma el índice en su correspondiente celda de la matriz
            b = False
            z0 = matriz[im][jm][0]
            z1 = matriz[im][jm][1]
            if (i+1)//7 != im:
                Im = im+1
                z0 = (z0 + matriz[Im][jm][0])/2
                z1 = (z1 + matriz[Im][jm][1])/2
                a = True
            if (j+1)//7 != jm:
                Jm = jm+1
                z0 = (z0 + matriz[im][Jm][0])/2
                z1 = (z1 + matriz[im][Jm][1])/2
                b = True
            if a and b:
                z0 = (matriz[im][jm][0] + matriz[im][Jm][0] + matriz[Im][Jm][0] + matriz[Im][jm][0])/4
                z1 = (matriz[im][jm][1] + matriz[im][Jm][1] + matriz[Im][Jm][1] + matriz[Im][jm][1])/4
            
            # Condición borde, para cerrar las paredes y que no se pueda salir al vacío
            if i==0 or j==0 or i==lenXS-1 or j==lenYS-1:
                z1 = z0

            # Agregamos el vértice a la malla correspondiente
            suelo.add_vertex([x, y, z0])
            techo.add_vertex([x, y, z1])

    # Se calcula el índice de cada punto (i, j) de la forma:
    index = lambda i, j: i*lenYS + j
    # Obtenemos los vertices de cada malla, y agregamos las caras
    vertexsuelo = list(suelo.vertices())
    vertextecho = list(techo.vertices())

    # Creamos las caras para esta malla (Y usar esta orientación para los factoriales)
    for i in range(lenXS-1):
        for j in range(lenYS-1):
            # los índices:
            isw = index(i,j)
            ise = index(i+1,j)
            ine = index(i+1,j+1)
            inw = index(i,j+1)
            # Identificar vértices
            Vsw = vertexsuelo[isw]
            Vse = vertexsuelo[ise]
            Vne = vertexsuelo[ine]
            Vnw = vertexsuelo[inw]
            # Se agregan las caras
            suelo.add_face(Vsw, Vse, Vne)
            suelo.add_face(Vne, Vnw, Vsw)
            # Identificar vértices
            Vsw = vertextecho[isw]
            Vse = vertextecho[ise]
            Vne = vertextecho[ine]
            Vnw = vertextecho[inw]
            # Se agregan las caras
            techo.add_face(Vsw, Vse, Vne)
            techo.add_face(Vne, Vnw, Vsw)

    # No alcancé a arreglar esto a tiempo para la entrega de esta tarea
    # Se aplican fractales a la malla
    fractal = 0
    sueloMesh = fractalMesh(suelo, fractal)
    techoMesh = fractalMesh(techo, fractal)
    lenXS += (lenXS-1)*(2**fractal -1)
    lenYS += (lenYS-1)*(2**fractal -1)
    
    index = lambda i, j: i*lenYS + j

    # Obtenemos los vertices de cada malla, y agregamos las caras
    vertexsuelo = list(sueloMesh.vertices())
    vertextecho = list(techoMesh.vertices())

    # Se generan los nuevos mesh que contienen las texturas (Se rehace ya que cada hay vértices que contienen
    # 4 coordenadas de texturas)
    sueloMeshtex = om.TriMesh()
    techoMeshtex = om.TriMesh()
    sueloMeshtex.request_vertex_texcoords2D()
    techoMeshtex.request_vertex_texcoords2D()

    indexMat = 7 + 6*(2**fractal -1)

    # Se crean las caras para cada cuadrado de la celda
    for i in range(lenXS-1):
        im = (i+1)//indexMat
        for j in range(lenYS-1):
            jm = (j+1)//indexMat
            # los índices:
            isw = index(i,j)
            ise = index(i+1,j)
            ine = index(i+1,j+1)
            inw = index(i,j+1)
            # Coordenadas de texturas
            indice = matriz[im][jm][2]
            tx = 1/12 * (indice)
            tX = 1/12 * (indice+1)
            tsw = [tx, 1]
            tse = [tX, 1]
            tne = [tX, 0]
            tnw = [tx, 0]
            # Identificar vértices
            vsw = vertexsuelo[isw]
            vse = vertexsuelo[ise]
            vne = vertexsuelo[ine]
            vnw = vertexsuelo[inw]
            # Agregar vertices a la nueva malla
            Vsw = sueloMeshtex.add_vertex(sueloMesh.point(vsw).tolist())
            Vse = sueloMeshtex.add_vertex(sueloMesh.point(vse).tolist())
            Vne = sueloMeshtex.add_vertex(sueloMesh.point(vne).tolist())
            Vnw = sueloMeshtex.add_vertex(sueloMesh.point(vnw).tolist())
            # Agregar las coordenadas de texturas a los vertices
            sueloMeshtex.set_texcoord2D(Vsw, tsw)
            sueloMeshtex.set_texcoord2D(Vse, tse)
            sueloMeshtex.set_texcoord2D(Vne, tne)
            sueloMeshtex.set_texcoord2D(Vnw, tnw)
            # Se agregan las caras
            sueloMeshtex.add_face(Vsw, Vse, Vne)
            sueloMeshtex.add_face(Vne, Vnw, Vsw)

            # Identificar vértices
            vsw = vertextecho[isw]
            vse = vertextecho[ise]
            vne = vertextecho[ine]
            vnw = vertextecho[inw]
            # Agregar vertices a la nueva malla
            Vsw = techoMeshtex.add_vertex(techoMesh.point(vsw).tolist())
            Vse = techoMeshtex.add_vertex(techoMesh.point(vse).tolist())
            Vne = techoMeshtex.add_vertex(techoMesh.point(vne).tolist())
            Vnw = techoMeshtex.add_vertex(techoMesh.point(vnw).tolist())
            # Coordenadas de texturas
            indice = matriz[im][jm][3]
            tx = 1/12 * (indice)
            tX = 1/12 * (indice+1)
            tsw = [tx, 1]
            tse = [tX, 1]
            tne = [tX, 0]
            tnw = [tx, 0]
            # Agregar las coordenadas de texturas a los vertices
            techoMeshtex.set_texcoord2D(Vsw, tsw)
            techoMeshtex.set_texcoord2D(Vse, tse)
            techoMeshtex.set_texcoord2D(Vne, tne)
            techoMeshtex.set_texcoord2D(Vnw, tnw)
            # Se agregan las caras
            techoMeshtex.add_face(Vsw, Vse, Vne)
            techoMeshtex.add_face(Vne, Vnw, Vsw)

    del sueloMesh
    del techoMesh
    # Se entregan las mallas
    return (sueloMeshtex, techoMeshtex, lenXS, lenYS)

def get_vertexs_and_indexes(mesh, orientation):
    # Obtenemos las caras de la malla
    faces = mesh.faces()

    # orientation indica si las normales deben apuntar abajo(-1) o arriba(1)
    assert orientation==1 or orientation==-1, "La orientación debe ser indicada con 1 o -1"

    # Creamos una lista para los vertices e indices
    vertexs = []
    indexes = []

    # Obtenemos los vertices y los recorremos
    for vertex in mesh.vertices():
        point = mesh.point(vertex).tolist()
        vertexs += point
        # Agregamos las coordenadas de a textura y su índice
        vertexs += mesh.texcoord2D(vertex).tolist()
        # Agregamos la norma
        normal = calculateNormal(mesh)
        normal = orientation * normal

        vertexs += [normal[0], normal[1], normal[2]]

    for face in faces:
        # Obtenemos los vertices de la cara
        face_indexes = mesh.fv(face)
        for vertex in face_indexes:
            # Obtenemos el numero de indice y lo agregamos a la lista
            indexes += [vertex.idx()]

    return vertexs, indexes

def createCave(pipeline, Matriz):
    # Creamos las mallas
    meshs = caveMesh(Matriz)
    # obtenemos los vértices e índices del suelo y del techo
    sVertices, sIndices = get_vertexs_and_indexes(meshs[0],1)
    tVertices, tIndices = get_vertexs_and_indexes(meshs[1],-1)
    sueloShape = bs.Shape(sVertices, sIndices)
    techoShape = bs.Shape(tVertices, tIndices)

    suelo = mallaTam(meshs[0], meshs[2], meshs[3])
    techo = meshs[1]

    gpuSuelo = createTextureGPUShape(sueloShape, pipeline, texturasPath)
    gpuTecho = createTextureGPUShape(techoShape, pipeline, texturasPath)

    return gpuSuelo, gpuTecho, suelo, techo

"""

########## Curva Nonuniform splines ##############################
class CatmullRom:
    """ Crear una curva catmull rom"""
    def __init__(self, posiciones, velocidad=1):
        """ Crea una curva catmull rom, en base a curvas de Hermite que van desde 0 a 1 cada una, descartando el primer y último nodo posición, manteniendo una continuidad C1"""
        self.posiciones = posiciones
        self.nodos = posiciones.shape[0]
        self.MCR = np.array([[0, -0.5, 1, -0.5], [1, 0, -2.5, 1.5], [0, 0.5, 2, -1.5], [0, 0, -0.5, 0.5]]) # Matriz que contiene la formulación de la curva
        self.tiempo = self.nodos-3
        self.vertices = None
        self.puntos = None
        self.Actual = 0.5
        self.velocidad = velocidad
        self.avanzar = False
        self.radio = 4 # Radio que tendrá el tobogán, se incluye acá para que pueda ser utilizado en la cámara también.
        self.bote = 1

    def getPosition(self, tiempo):
        """ Calcula la posición de la curva grande, estimando entre que nodos  está y calcular la curva de HERMITE que describe entre los nodos que se encuentra la posición en el tiempo"""
        assert tiempo<self.tiempo, "El tiempo a evaluar debe estar dentro del rango parametrizado del tiempo"
        T = int(tiempo)
        i = T+1
        t = tiempo-T     # Tiempo entre 0 y 1
        pos = self.posiciones
        G = np.concatenate((np.array([pos[i-1]]).T, np.array([pos[i]]).T, np.array([pos[i+1]]).T, np.array([pos[i+2]]).T), axis=1)
        matriz = np.matmul(G, self.MCR)
        T = generateT(t)
        return np.matmul(matriz, T).T[0]

    def drawGraph(self, pipeline, N):
        "Dibuja un gráfico en 2D, utilizando el pipeline entregado en formato Lines. Se utilizó para ver si funcionaba la estructura"
        dt = self.tiempo/N
        vertices = []
        indices = range(N)
        for i in range(N):
            t = i * dt
            pos = self.getPosition(t)
            vertices += [pos[0], pos[1], pos[2], 1, 0, 0] # posición rojo
        shape = bs.Shape(vertices, indices)
        gpu = createGPUShape(pipeline, shape)
        return gpu
    
    def createCurve(self, N):
        "Crea la curva en cuestión, creando un total de N puntos"
        dt = self.tiempo/N
        self.puntos = N
        vertices = []
        for i in range(N):
            t = i * dt
            pos = self.getPosition(t)
            vertices.append(pos)
        self.vertices = vertices

    def getvertice(self, i):
        " Entrega el vertice correspondiente a la iteración"
        if self.vertices is None:
            return AssertionError
        return self.vertices[i]
    
    def camera(self, delta, controller):
        """Entrega las posiciones de la camara  y hacia donde debe mirar en todo momento, una vez ya creada la curva con "createCurve" """
        if controller.reset:
            self.Actual = 0.5
            self.avanzar = True
        avance = delta * self.velocidad
        TiempoMax = self.tiempo-3
        tiempo = self.Actual + 0.5
        if tiempo+2*avance>= TiempoMax:
            self.avanzar = False

        eye = self.getPosition(self.Actual)
        at = self.getPosition(tiempo)
        at[2] -= self.radio-1
        if self.avanzar:
            self.Actual += avance
        return eye, at

    def boat(self, delta, controller):
        """Entrega las posiciones del bote  y hacia donde debe mirar en todo momento, una vez ya creada la curva con "createCurve" """
        if controller.reset:
            self.bote = 1
            controller.reset = False
            
        avance = delta * self.velocidad
        TiempoMax = self.tiempo-0.2
        tiempo = self.bote
        tiempo1 = tiempo + 0.001

        pos = self.getPosition(tiempo)
        dir = self.getPosition(tiempo1)
        theta, alpha = orientacion(pos, dir)

        if tiempo1 + avance<TiempoMax and controller.empezar:
            self.bote += avance
        return pos, theta, alpha, dir


def createRandomColorNormalToroid(N):
    vertices = []
    indices = []

    Rcolor = rd.rand()
    Gcolor = rd.rand()
    Bcolor = rd.rand()

    dalpha = 2 * np.pi /(N-1)
    dbeta = 2 * np.pi /(N-1)
    R=0.3
    r = 0.2
    c = 0
    for i in range(N-1):
        beta = i * dbeta
        beta2= (i+1) * dbeta
        for j in range(N-1):
            alpha = j * dalpha
            alpha2 = (j+1) * dalpha

            v0 = [(R + r*np.cos(alpha))*np.cos(beta), (R+r*np.cos(alpha))*np.sin(beta), r*np.sin(alpha)]
            v1 = [(R + r*np.cos(alpha2))*np.cos(beta), (R+r*np.cos(alpha2))*np.sin(beta), r*np.sin(alpha2)]
            v2 = [(R + r*np.cos(alpha2))*np.cos(beta2), (R+r*np.cos(alpha2))*np.sin(beta2), r*np.sin(alpha2)]
            v3 = [(R + r*np.cos(alpha))*np.cos(beta2), (R+r*np.cos(alpha))*np.sin(beta2), r*np.sin(alpha)]

            n0 = [np.cos(alpha) * np.cos(beta), np.cos(alpha) * np.sin(beta), np.sin(alpha)]
            n1 = [np.cos(alpha2) * np.cos(beta), np.cos(alpha2) * np.sin(beta), np.sin(alpha2)]
            n2 = [np.cos(alpha2) * np.cos(beta2), np.cos(alpha2) * np.sin(beta2), np.sin(alpha2)]
            n3 = [np.cos(alpha) * np.cos(beta2), np.cos(alpha) * np.sin(beta2), np.sin(alpha)]

            vertices += [v0[0], v0[1], v0[2], Rcolor, Gcolor, Bcolor, n0[0], n0[1], n0[2]]
            vertices += [v1[0], v1[1], v1[2], Rcolor, Gcolor, Bcolor, n1[0], n1[1], n1[2]]
            vertices += [v2[0], v2[1], v2[2], Rcolor, Gcolor, Bcolor, n2[0], n2[1], n2[2]]
            vertices += [v3[0], v3[1], v3[2], Rcolor, Gcolor, Bcolor, n3[0], n3[1], n3[2]]
            indices += [ c + 0, c + 1, c +2 ]
            indices += [ c + 2, c + 3, c + 0 ]
            c += 4
    return bs.Shape(vertices, indices)

def createtoroidNode(gpu, pos, phi, alpha):
    toroidNode = sg.SceneGraphNode("toroid")
    toroidNode.transform =tr.matmul([
        tr.translate(pos[0], pos[1], pos[2]),
        tr.rotationY(-phi), 
        tr.rotationX(np.pi/2-alpha)
    ])
    toroidNode.childs = [gpu]
    return toroidNode

def createToroidsNode(pipeline, curve, N):
    toroids = []
    TiempoMax = curve.tiempo-3
    tiempoin = 1
    tiempo = TiempoMax - tiempoin
    dt = tiempo/(N-1)

    for i in range(N):
        t = i*dt
        pos = curve.getPosition(t)
        dir = curve.getPosition(t+0.1)
        theta, alpha = orientacion(pos, dir)
        phi = np.pi*(rd.rand()*0.5 - 0.2)
        theta = np.pi*(rd.rand()-0.5)
        adaptarPos(pos, curve.radio, phi, theta)
        toroid = createGPUShape(pipeline, createRandomColorNormalToroid(15))
        toroids.append(createtoroidNode(toroid, pos, phi, alpha))




    scaledToroid = sg.SceneGraphNode("sc_toroid")
    scaledToroid.childs = toroids

    return scaledToroid



def createSlide(curva, N):
    "Recibe la curva con las posiciones correspondientes y un número de discretización de puntos para la malla"
    isinstance(curva, CatmullRom)
    Tobogan = om.TriMesh()
    ToboganTex = om.TriMesh()
    ToboganTex.request_vertex_texcoords2D()
    curva.createCurve(N)
    radio = curva.radio # radio del tobogán
    puntosCilindro = 40 # número de puntos en un cilindro
    dtheta = 2 * np.pi/(puntosCilindro-1)
    
    cilindro = []
    for j in range(puntosCilindro-1):
            theta = j * dtheta
            sin = np.sin(theta)
            cos = np.cos(theta)
            cilindro.append(np.array([0, radio * cos, radio * sin, 1]))

    for i in range(N-1):
        pos0 = curva.getvertice(i)
        pos1 = curva.getvertice(i+1)
        theta, alpha = orientacion(pos0, pos1)
        matriZY = tr.rotationY(-alpha)
        #matriZY = np.identity(4)
        matriZZ = tr.rotationZ(theta)
        matriz = np.matmul(matriZZ, matriZY)
        for j in range(puntosCilindro-1):
            P = np.ones(4)
            P[0:3] = np.matmul(matriz, cilindro[j])[0:3] + pos0
            vertice = np.array([P[0], P[1], P[2]])
            Tobogan.add_vertex(vertice)
    
     # Se calcula el índice de cada punto (i, j) de la forma:
    index = lambda i, j: i*(puntosCilindro-1) + j
    # Obtenemos los vertices de cada malla, y agregamos las caras
    vertex = list(Tobogan.vertices())

    # Creamos las caras para esta malla (Y usar esta orientación para los factoriales)
    for i in range(N-2):
        for j in range(puntosCilindro-1):
            # los índices:
            isw = index(i,j)
            ise = index(i+1,j)
            if j+1==puntosCilindro-1:
                ine = index(i+1, 0)
                inw = index(i, 0)
            else:
                ine = index(i+1,j+1)
                inw = index(i,j+1)
            # Identificar vértices
            Vsw = vertex[isw]
            Vse = vertex[ise]
            Vne = vertex[ine]
            Vnw = vertex[inw]
            # Se agregan las caras
            Tobogan.add_face(Vsw, Vse, Vne)
            Tobogan.add_face(Vne, Vnw, Vsw)

    # Agregamos vértices y caras a la malla poligonal de la parte inferior, que contendrá la textura del agua en movimiento
    for i in range(N-2):
        for j in range(puntosCilindro//2+1, puntosCilindro-2):
            # los índices:
            isw = index(i,j)
            ise = index(i+1,j) 
            ine = index(i+1,j+1)
            inw = index(i,j+1)
            # Identificar vértices
            Vsw = vertex[isw]
            Vse = vertex[ise]
            Vne = vertex[ine]
            Vnw = vertex[inw]
            # Se agregan a la nueva malla repitiendo vértices para las distintas coordenadas de texturas
            vsw = ToboganTex.add_vertex(Tobogan.point(Vsw))
            vse = ToboganTex.add_vertex(Tobogan.point(Vse))
            vne = ToboganTex.add_vertex(Tobogan.point(Vne))
            vnw = ToboganTex.add_vertex(Tobogan.point(Vnw))
            # Se agregan las coordenadas de texturas
            ToboganTex.set_texcoord2D(vsw, [0, 1])
            ToboganTex.set_texcoord2D(vse, [1, 1])
            ToboganTex.set_texcoord2D(vne, [1, 0])
            ToboganTex.set_texcoord2D(vnw, [0, 0])
            # Se agregan las caras
            ToboganTex.add_face(vsw, vse, vne)
            ToboganTex.add_face(vne, vnw, vsw)
    return Tobogan, ToboganTex

def get_vertexs_and_indexesTobogan(mesh, boolTex = False):
    # Obtenemos las caras de la malla
    faces = mesh.faces()

    calculateNormal(mesh) # Se calculan las normales de las caras

    # Creamos una lista para los vertices e indices
    vertexs = []
    indexes = []

    # Obtenemos los vertices y los recorremos
    for vertex in mesh.vertices():
        point = mesh.point(vertex).tolist()
        vertexs += point
        if boolTex:
            texcoord = mesh.texcoord2D(vertex).tolist()
            vertexs +=  texcoord
        else:
            vertexs += [0, 1, 0]
        
        normal = np.array([0, 0, 0])            # vector que promediará las normales de las caras adyacentes
        outHalfEdge = mesh.halfedge_handle(vertex)  #Se obtiene el half edge de salida
        OutHalfEdge = outHalfEdge
        k = True # Se crea una variable que sirve para indicar si seguimos dentro de las caras vecinas
        while k:
            face = mesh.face_handle(outHalfEdge)        # Obtiene la cara ligada al half edge
            nextHalfEdge = mesh.next_halfedge_handle(outHalfEdge)   # Obtiene el siguiente half edge 
            if mesh.face_handle(nextHalfEdge) != face:    # Revisa que el siguiente half edge está ligado a la misma cara
                k = False   # No lo está
            else:
                inHalfEdge = mesh.next_halfedge_handle(nextHalfEdge)    # Obtiene el siguiente half edge que apuntará al vértice nuevamente
                outHalfEdge = mesh.opposite_halfedge_handle(inHalfEdge) # Se pasa al half edge opuesto que va en salida
                if outHalfEdge == OutHalfEdge: k = False    # Volvemos al half edge del inicio
                Normal = np.array(mesh.normal(face)) # Se obtiene la normal calculada en la cara
                normal = normal + Normal    # Se suman las normales
        normal = normal/np.linalg.norm(normal)    # Se obtiene el promedio de las normales
        
        vertexs += [normal[0], normal[1], normal[2]]

    for face in faces:
        # Obtenemos los vertices de la cara
        face_indexes = mesh.fv(face)
        for vertex in face_indexes:
            # Obtenemos el numero de indice y lo agregamos a la lista
            indexes += [vertex.idx()]

    return vertexs, indexes

def createTobogan(pipeline, mesh):
    # obtenemos los vértices e índices del tobogán
    Vertices, Indices = get_vertexs_and_indexesTobogan(mesh)
    Shape = bs.Shape(Vertices, Indices)

    gpuShape = createGPUShape(pipeline, Shape)

    return gpuShape

def createTexTobogan(pipeline, mesh):
    # Se obitenen los vertices e índices del tobogán con texturas
    Vertices, Indices = get_vertexs_and_indexesTobogan(mesh, True)
    Shape = bs.Shape(Vertices, Indices)

    gpuShape = createMultipleTextureGPUShape(Shape, pipeline, [waterPath, displacementPath], minFilterMode=GL_LINEAR, maxFilterMode=GL_LINEAR)

    return gpuShape