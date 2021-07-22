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

texBola1 = os.path.join(assetsDirectory, "bola1.png")

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

def createNormalTexSphere(Nphi, Ntheta):
    """ int int -> bs.shape
    Crea una esfera de Nphi puntos discretizados para la rotación 0 a 2pi y Ntheta para la rotación de 0 a pi, con radio 1. Donde tiene incluida sus normales y texturas."""
    vertices = []
    indices = []
    dphi = 2 * np.pi/Nphi
    dtheta = np.pi/Ntheta
    contador = 0
    for i in range(Nphi):
        phi = i * dphi
        phi1 = (i+1) * dphi
        for j in range(Ntheta):
            theta = j * dtheta
            theta1 = (j+1) * dtheta

            v0 = [np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)]
            v1 = [np.sin(theta1)*np.cos(phi), np.sin(theta1)*np.sin(phi), np.cos(theta1)]
            v2 = [np.sin(theta1)*np.cos(phi1), np.sin(theta1)*np.sin(phi1), np.cos(theta1)]
            v3 = [np.sin(theta)*np.cos(phi1), np.sin(theta)*np.sin(phi1), np.cos(theta)]

            t0 = [phi/(2 * np.pi), theta/(np.pi)]
            t1 = [phi/(2 * np.pi), theta1/(np.pi)]
            t2 = [phi1/(2 * np.pi), theta1/(np.pi)]
            t3 = [phi1/(2 * np.pi), theta/(np.pi)]

            vertices += [v0[0], v0[1], v0[2], t0[0], t0[1], v0[0], v0[1], v0[2]]
            vertices += [v1[0], v1[1], v1[2], t1[0], t1[1], v1[0], v1[1], v1[2]]
            vertices += [v2[0], v2[1], v2[2], t2[0], t2[1], v2[0], v2[1], v2[2]]
            vertices += [v3[0], v3[1], v3[2], t3[0], t3[1], v3[0], v3[1], v3[2]]
            indices += [ contador + 0, contador + 1, contador +2 ]
            indices += [ contador + 2, contador + 3, contador + 0 ]
            contador += 4

    return bs.Shape(vertices, indices)


def createNormalTexTable(pipeline, largo, ancho, grosor, diam):
    """num num num num -> bs.Shape
    Crea una mesa de largo (eje X) y ancho (eje Y) personalizado, donde presentará 6 bolsillos en las esquinas y centro del lado largo, con un diametro especificado
    y además presentará amortiguadores con grosor entregado en los bordes de la mesa. Tendrá una altura específica y el centro se ubicará justo en el 0,0"""
    altura = 0.8 #Altura de la mesa
    dx = largo/2
    dy = ancho/2

    

    # Tela Verde
    def TelaVerde(pipeline, dx, dy, altura):
        vertices = []
        indices = []
        vertices += [dx, dy, altura, 0.3, 1 , 0.3, 0, 0, 1]
        vertices += [dx, -dy, altura, 0.3, 1 , 0.3, 0, 0, 1]
        vertices += [-dx, -dy, altura, 0.3, 1 , 0.3, 0, 0, 1]
        vertices += [-dx, dy, altura, 0.3, 1 , 0.3, 0, 0, 1]

        indices += [0, 1, 2, 2, 3, 0]

        return createGPUShape(pipeline, bs.Shape(vertices, indices))

    tela = sg.SceneGraphNode("Tela")
    tela.childs = [TelaVerde(pipeline, dx-3/2*diam, dy-3/2*diam, altura)]

    # Bordes
    def Bordes(pipeline, dx, dy, radio, altura):
        vertices = []
        indices = []

        color = [0.2, 0.1, 0.1]

        vertices += [dx, dy, altura, color[0], color[1], color[2], 0, 0, 1]
        vertices += [dx, -dy, altura, color[0], color[1], color[2], 0, 0, 1]
        vertices += [-dx, -dy, altura, color[0], color[1], color[2], 0, 0, 1]
        vertices += [-dx, dy, altura, color[0], color[1], color[2], 0, 0, 1]
        vertices += [dx-radio, dy-radio, altura, color[0], color[1], color[2], 0, 0, 1]
        vertices += [dx-radio, -dy+radio, altura, color[0], color[1], color[2], 0, 0, 1]
        vertices += [-dx+radio, -dy+radio, altura, color[0], color[1], color[2], 0, 0, 1]
        vertices += [-dx+radio, dy-radio, altura, color[0], color[1], color[2], 0, 0, 1]

        indices += [0, 4, 5, 0, 1, 5,
                    1, 5, 6, 1, 2, 6,
                    2, 6, 7, 2, 3, 7,
                    3, 7, 4, 3, 0, 4]

        altura1 = altura + radio*2
        vertices += [dx, dy, altura1, color[0], color[1], color[2], 0, 0, 1]
        vertices += [dx, -dy, altura1, color[0], color[1], color[2], 0, 0, 1]
        vertices += [-dx, -dy, altura1, color[0], color[1], color[2], 0, 0, 1]
        vertices += [-dx, dy, altura1, color[0], color[1], color[2], 0, 0, 1]
        vertices += [dx-radio, dy-radio, altura1, color[0], color[1], color[2], 0, 0, 1]
        vertices += [dx-radio, -dy+radio, altura1, color[0], color[1], color[2], 0, 0, 1]
        vertices += [-dx+radio, -dy+radio, altura1, color[0], color[1], color[2], 0, 0, 1]
        vertices += [-dx+radio, dy-radio, altura1, color[0], color[1], color[2], 0, 0, 1]

        indices += [8, 12, 13, 8, 9, 13,
                    9, 13, 14, 9, 10, 14,
                    10, 14, 15, 10, 11, 15,
                    11, 15, 12, 11, 8, 12]

        # índice en 16
        
        vertices += [dx-radio, dy-radio, altura, color[0], color[1], color[2], -1, 0, 0]
        vertices += [dx-radio, -dy+radio, altura, color[0], color[1], color[2], -1, 0, 0]
        vertices += [dx-radio, dy-radio, altura1, color[0], color[1], color[2], -1, 0, 0]
        vertices += [dx-radio, -dy+radio, altura1, color[0], color[1], color[2], -1, 0, 0]

        indices += [16, 17, 18, 17, 18, 19]

        vertices += [dx-radio, dy-radio, altura, color[0], color[1], color[2], 0, -1, 0]
        vertices += [-dx+radio, dy-radio, altura, color[0], color[1], color[2], 0, -1, 0]
        vertices += [dx-radio, dy-radio, altura1, color[0], color[1], color[2], 0, -1, 0]
        vertices += [-dx+radio, dy-radio, altura1, color[0], color[1], color[2], 0, -1, 0]

        indices += [20, 21, 22, 21, 22, 23]

        vertices += [-dx+radio, -dy+radio, altura, color[0], color[1], color[2], 1, 0, 0]
        vertices += [-dx+radio, dy-radio, altura, color[0], color[1], color[2], 1, 0, 0]
        vertices += [-dx+radio, -dy+radio, altura1, color[0], color[1], color[2], 1, 0, 0]
        vertices += [-dx+radio, dy-radio, altura1, color[0], color[1], color[2], 1, 0, 0]

        indices += [24, 25, 26, 25, 26, 27]

        vertices += [-dx+radio, -dy+radio, altura, color[0], color[1], color[2], 0, 1, 0]
        vertices += [dx-radio, -dy+radio, altura, color[0], color[1], color[2], 0, 1, 0]
        vertices += [-dx+radio, -dy+radio, altura1, color[0], color[1], color[2], 0, 1, 0]
        vertices += [dx-radio, -dy+radio, altura1, color[0], color[1], color[2], 0, 1, 0]

        indices += [28, 29, 30, 29, 30, 31]

        vertices += [dx, dy, altura, color[0], color[1], color[2], 1, 0, 0]
        vertices += [dx, -dy, altura, color[0], color[1], color[2], 1, 0, 0]
        vertices += [dx, dy, altura1, color[0], color[1], color[2], 1, 0, 0]
        vertices += [dx, -dy, altura1, color[0], color[1], color[2], 1, 0, 0]

        indices += [32, 33, 34, 33, 34, 35]

        vertices += [dx, dy, altura, color[0], color[1], color[2], 0, 1, 0]
        vertices += [-dx, dy, altura, color[0], color[1], color[2], 0, 1, 0]
        vertices += [dx, dy, altura1, color[0], color[1], color[2], 0, 1, 0]
        vertices += [-dx, dy, altura1, color[0], color[1], color[2], 0, 1, 0]

        indices += [36, 37, 38, 37, 38, 39]

        vertices += [-dx, -dy, altura, color[0], color[1], color[2], -1, 0, 0]
        vertices += [-dx, dy, altura, color[0], color[1], color[2], -1, 0, 0]
        vertices += [-dx, -dy, altura1, color[0], color[1], color[2], -1, 0, 0]
        vertices += [-dx, dy, altura1, color[0], color[1], color[2], -1, 0, 0]

        indices += [40, 41, 42, 41, 42, 43]

        vertices += [-dx, -dy, altura, color[0], color[1], color[2], 0, -1, 0]
        vertices += [dx, -dy, altura, color[0], color[1], color[2], 0, -1, 0]
        vertices += [-dx, -dy, altura1, color[0], color[1], color[2], 0, -1, 0]
        vertices += [dx, -dy, altura1, color[0], color[1], color[2], 0, -1, 0]

        indices += [44, 45, 46, 45, 46, 47]

        return createGPUShape(pipeline, bs.Shape(vertices, indices))

    bordes = sg.SceneGraphNode("Bordes")
    bordes.childs = [Bordes(pipeline, dx, dy, diam/2, altura)]


    # Bolsillos:

    #Ubiucación de los centros de cada bolsillo
    p0, p1, p2, p3, p4, p5 = (dx-diam,dy-diam), (dx-diam, -dy+diam), (-dx+diam, -dy+diam), (-dx+diam, dy-diam), (0, dy-diam), (0, -dy+diam)

    def Bolsillo(pipeline, radio, n):
        vertices = [0, 0, 0, 0.01, 0.01, 0.01, 0, 0, 1,
                    radio, 0, 0, 0.01, 0.01, 0.01, 0, 0, 1,
                    radio, 0, 0, 0.01, 0.01, 0.01, -1, 0, 0,
                    radio, 0, radio*2, 0.01, 0.01, 0.01, -1, 0, 0]
        indices = []
        dtheta = 2*np.pi/n
        for i in range(1,n+1):
            theta = i*dtheta
            rx = radio*np.cos(theta)
            ry = radio*np.sin(theta)

            vertices += [rx, ry, 0, 0.01, 0.01, 0.01, 0, 0, 1]
            vertices += [rx, ry, 0, 0.01, 0.01, 0.01, radio*np.cos(theta+np.pi), radio*np.sin(theta+np.pi), 0]
            vertices += [rx, ry, radio*2, 0.01, 0.01, 0.01, radio*np.cos(theta+np.pi), radio*np.sin(theta+np.pi), 0]

            j = 3*i
            indices += [0, j-2, j+1,
                        j-1, j, j+2,
                        j, j+2, j+3] 

        return createGPUShape(pipeline, bs.Shape(vertices, indices))
    
    altura += 0.001 - diam

    bolsillo1 = sg.SceneGraphNode("bolsillo1")
    bolsillo1.transform = tr.translate(p0[0], p0[1], altura)
    bolsillo1.childs = [Bolsillo(pipeline, diam/2, 30)]

    bolsillo2 = sg.SceneGraphNode("bolsillo2")
    bolsillo2.transform = tr.translate(p1[0], p1[1], altura)
    bolsillo2.childs = [Bolsillo(pipeline, diam/2, 30)]

    bolsillo3 = sg.SceneGraphNode("bolsillo3")
    bolsillo3.transform = tr.translate(p2[0], p2[1], altura)
    bolsillo3.childs = [Bolsillo(pipeline, diam/2, 30)]

    bolsillo4 = sg.SceneGraphNode("bolsillo4")
    bolsillo4.transform = tr.translate(p3[0], p3[1], altura)
    bolsillo4.childs = [Bolsillo(pipeline, diam/2, 30)]

    bolsillo5 = sg.SceneGraphNode("bolsillo5")
    bolsillo5.transform = tr.translate(p4[0], p4[1], altura)
    bolsillo5.childs = [Bolsillo(pipeline, diam/2, 30)]

    bolsillo6 = sg.SceneGraphNode("bolsillo6")
    bolsillo6.transform = tr.translate(p5[0], p5[1], altura)
    bolsillo6.childs = [Bolsillo(pipeline, diam/2, 30)]

    bolsillos = sg.SceneGraphNode("Bolsillos")
    bolsillos.childs = [bolsillo1, bolsillo2, bolsillo3, bolsillo4, bolsillo5, bolsillo6]




    mesa = sg.SceneGraphNode("Mesa")
    mesa.childs = [tela, bolsillos, bordes]




    return mesa



