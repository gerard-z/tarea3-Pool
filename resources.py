# coding=utf-8
""" Módulo que contiene las clases y objetos relacionados al jugador, entidades, cámara y
mecánicas de la aplicación, en resumen, todo lo que no tiene que ver con el apartado de modelos
geométricos ni la parte gráfica """

from OpenGL.GL import *
from sys import path_importer_cache
import glfw
import numpy as np
from numpy.linalg import linalg
import grafica.transformations as tr
import grafica.easy_shaders as es
import grafica.scene_graph as sg
import grafica.basic_shapes as bs
import shapes3d as sh
import grafica.odeResolver as ode
from numpy import random as rd

# CLASE CON LOS ELEMENTOS DE LA MESA
class Bola:
    """ Las bolas de billar, la clase permite manejar la posición y velocidad de cada bola, calcular sus colisiones y físicas."""
    def __init__(self, pipeline, position, shadowpipeline):
        # Figura:
        self.model = None
        self.diam = 0.051 # Diametro de las bolas (Para todas las bolas de número, la bola blanca mide 0.048)
        self.pipeline = pipeline # Pipeline de la bola
        self.shadowpipeline = shadowpipeline
        self.position = position
        self.velocity = np.array([0., 0., 0.]) # Las pelotas siempre comienzan con velocidad 0
        self.sombra = sh.createShadowQuad(shadowpipeline)

    def setModel(self, model):
        # Define la gpushape de la bola
        self.model = model

    def move(self, ROCE):
        "Movimiento con roce de las bolas"
        # Se modifican la velocidad y posición
        def roce(t, y0):
            # calcula posicion y velocidad con respecto al roce
            R = ROCE*np.array([1., 1.])
            ax = 0
            ay = 0
            Ax = False
            Ay = False
            if np.abs(y0[1][0])<0.01:
                Ax = True
            if np.abs(y0[1][1])<0.01:
                Ay = True
            if Ax and Ay:
                y0[1][0] = 0
                ax = y0[1][0]
                y0[1][1] = 0
                ay = y0[1][1]
        
            ax = y0[1][0]/R[0]
            ay = y0[1][1]/R[1]
            return np.array([y0[1], np.array([-ax, -ay]) ])

        y0 = np.array([self.position[0:2], self.velocity[0:2]])
        yfinal = ode.euler(0.01, 0, y0, roce)
        self.position[0:2] = yfinal[0]
        self.velocity[0:2] = yfinal[1]
            

    def interactionTable(self, mesa, COEF):
        """ Se le entrega la mesa donde está jugando, para detectar colisiones con la mesa y la gravedad ligada a esta, utilizando la aproximación por euler"""
        assert isinstance(mesa, MESA)
        tamaño = mesa.tamaño
        altura = 0.8
        diam = mesa.bolsillos # diámetro de los bolsillos

        X, Y = tamaño[0], tamaño[1]
        dx, dy = X/2, Y/2

        # detección de estar dentro del bolsillo (para caer posteriormente)
        def sobreBolsillo(x, y, pos, radio):
            diferencia = pos-np.array([x, y, pos[2]])
            distancia = np.linalg.norm(diferencia)
            return distancia <= radio
        
        # Calcula la gravedad
        def gravedad(t, y0):
            return np.array([y0[1], -1])

        y0 = np.array([self.position[2], self.velocity[2]])
        
        yfinal = ode.euler(0.01, 0, y0, gravedad) # Se utiliza tiempo 0, porque en realidad el tiempo es independiente de la gravedad.
        
        self.position[2] = yfinal[0]
        self.velocity[2] = yfinal[1]
        
        pos = self.position
        sobrebolsillodx = sobreBolsillo(dx-diam, dy-diam, pos, diam) or sobreBolsillo(dx-diam, -dy+diam, pos, diam)
        sobrebolsillo0 = sobreBolsillo(0, dy-diam, pos, diam) or sobreBolsillo(0, -dy+diam, pos, diam)
        sobrebolsillodx2 = sobreBolsillo(-dx+diam, dy-diam, pos, diam) or sobreBolsillo(-dx+diam, -dy+diam, pos, diam)

        posbajo = self.position[2]-self.diam/1.9

        if sobrebolsillodx or sobrebolsillo0 or sobrebolsillodx2:
            if posbajo <= altura - diam:
                self.velocity[2] = 0
        elif posbajo <= altura:
            self.velocity[2] = 0

        #Colisiones con los bordes de madera, que tienen la mitad del coeficiente de restitución (Utiliza colisión esfera/plano)
        if pos[0] +self.diam/2 > dx and self.velocity[0]>0:
            self.velocity[0] = -abs(self.velocity[0])*COEF/2

        if pos[0] < -dx + self.diam/2 and self.velocity[0]<0:
            self.velocity[0] = abs(self.velocity[0])*COEF/2

        if pos[1] > dy - self.diam/2 and self.velocity[1]>0:
            self.velocity[1] = -abs(self.velocity[1])*COEF/2

        if pos[1] < -dy + self.diam/2 and self.velocity[1]<0:
            self.velocity[1] = abs(self.velocity[1])*COEF/2

        # Colisiones con los amortiguadores (primera aproximación de una caja para las bolas con cajas de amortiguadores y luego se procesa según la forma)
        radio = diam/2
        recorte = diam + radio
        recorte1 = 2*diam + radio
        sdiam = self.diam
        
        def colisionBorde(componente, s):
            "Primera aproximación por plano, verifica si pasa por el plano para estar dentro de un amortiguador"
            if s>0:
                return pos[componente]+sdiam/2>s
            else:
                return pos[componente]-sdiam/2<s

        def colisionDiag(componente, s, operador):
            "Segunda aproximación, verifica si la posicion de la pelota es tal que debe considerar la colision con los bordes diagonales del amortiguador"
            if operador:
                return pos[componente]>s
            else:
                return pos[componente]<s

        def colisionPrecisa(orientacion, p1, p2):
            """Determina si colisiona con los bordes diagonales, la orientacion es True si la normal hacia el exterior del amortiguador coincide
            con la regla de la mano derecha rotando al vector p1-p2. Donde siempre se tendrá que p1 es el punto que está en contacto con el borde de la mesa"""
            tangente = p1-p2
            tangente /= np.linalg.norm(tangente)
            vector = pos[0:2]-p2
            proyeccion = np.dot(vector, tangente) * tangente
            punto = p2+proyeccion
            if np.linalg.norm(pos[0:2]-punto)<sdiam/2: #Existe una colision
                normal = np.array([-tangente[1], tangente[0]])
                if not orientacion: normal = -normal
                v = self.velocity[0:2]
                if not np.dot(v, normal)>0.0:
                    vn = np.dot(v, normal) * normal*COEF
                    vt = np.dot(v, tangente) * tangente

                    self.velocity[0:2] = -vn+vt

        #Amortiguador lateral derecho
        if colisionBorde(0, dx-recorte):
            A = colisionDiag(1, dy-recorte1, True)
            B = colisionDiag(1, -dy+recorte1, False)
            if A:
                colisionPrecisa(True, np.array([dx-radio, dy-recorte]), np.array([dx-recorte, dy-recorte1]))
            elif B:
                colisionPrecisa(False, np.array([dx-radio, -dy+recorte]), np.array([dx-recorte, -dy+recorte1]))
            elif self.velocity[0]>0:
                self.velocity[0] = -abs(self.velocity[0])*COEF

        #Amortiguador lateral izquierdo
        if colisionBorde(0, -dx+recorte):
            A = colisionDiag(1, dy-recorte1, True)
            B = colisionDiag(1, -dy+recorte1, False)
            if A:
                colisionPrecisa(True, np.array([-dx+radio, dy-recorte]), np.array([-dx+recorte, dy-recorte1]))
            elif B:
                colisionPrecisa(False, np.array([-dx+radio, -dy+recorte]), np.array([-dx+recorte, -dy+recorte1]))
            elif self.velocity[0]<0:
                self.velocity[0] = abs(self.velocity[0])*COEF
            
        #Amortiguador horizontal superior derecho
        if colisionBorde(1, dy-recorte):
            A = colisionDiag(0, dx-recorte1, True)
            B = colisionDiag(0, diam, False)
            if B:
                colisionPrecisa(True, np.array([radio, dy-radio]), np.array([diam, dy-recorte]))
            elif A:
                colisionPrecisa(False, np.array([dx-recorte, dy-radio]), np.array([dx-recorte1, dy-recorte]))
            elif self.velocity[1]>0:
                self.velocity[1] = -abs(self.velocity[1])*COEF
            
        #Amortiguador horizontal superior izquierdo
        if colisionBorde(1, dy-recorte):
            A = colisionDiag(0, -diam, True)
            B = colisionDiag(0, -dx+recorte1, False)
            if A:
                colisionPrecisa(False, np.array([-radio, dy-radio]), np.array([-diam, dy-recorte]))
            elif B:
                colisionPrecisa(True, np.array([-dx+recorte, dy-radio]), np.array([-dx+recorte1, dy-recorte]))
            elif self.velocity[1]>0:
                self.velocity[1] = -abs(self.velocity[1])*COEF
        
        #Amortiguador horizontal inferior derecho
        if colisionBorde(1, -dy+recorte):
            A = colisionDiag(0, dx-recorte1, True)
            B = colisionDiag(0, diam, False)
            if B:
                colisionPrecisa(False, np.array([radio, -dy+radio]), np.array([diam, -dy+recorte]))
            elif A:
                colisionPrecisa(True, np.array([dx-recorte, -dy+radio]), np.array([dx-recorte1, -dy+recorte]))
            elif self.velocity[1]<0:
                self.velocity[1] = abs(self.velocity[1])*COEF
            
        #Amortiguador horizontal inferior izquierdo
        if colisionBorde(1, -dy+recorte):
            A = colisionDiag(0, -diam, True)
            B = colisionDiag(0, -dx+recorte1, False)
            if A:
                colisionPrecisa(True, np.array([-radio, -dy+radio]), np.array([-diam, -dy+recorte]))
            elif B:
                colisionPrecisa(False, np.array([-dx+recorte, -dy+radio]), np.array([-dx+recorte1, -dy+recorte]))
            elif self.velocity[1]<0:
                self.velocity[1] = abs(self.velocity[1])*COEF






    def draw(self, projection, viewMatrix):
        # Dibujar la bola
        glUseProgram(self.pipeline.shaderProgram)
        glUniformMatrix4fv(glGetUniformLocation(self.pipeline.shaderProgram, "model"), 1, GL_TRUE, tr.matmul([
            tr.translate(self.position[0], self.position[1], self.position[2]),
            tr.uniformScale(self.diam/2)
        ])
        )
        self.pipeline.drawCall(self.model)

        glUseProgram(self.shadowpipeline.shaderProgram)
        glUniformMatrix4fv(glGetUniformLocation(self.shadowpipeline.shaderProgram, "projection"), 1, GL_TRUE, projection)
        glUniformMatrix4fv(glGetUniformLocation(self.shadowpipeline.shaderProgram, "view"), 1, GL_TRUE, viewMatrix)
        glUniformMatrix4fv(glGetUniformLocation(self.shadowpipeline.shaderProgram, "model"), 1, GL_TRUE, tr.matmul([
            tr.translate(self.position[0], self.position[1], self.position[2]-self.diam/2+0.001),
            tr.uniformScale(self.diam/2)
        ])
        )
        self.pipeline.drawCall(self.sombra)

class MESA:
    """ La mesa de billar, clase que permite calcular el tamaño de esta, su colision con los bordes, y detectar cuando una bola entre en los bolsillos para considerar el puntaje"""
    def __init__(self, pipeline):
        # Figura
        self.pipeline = pipeline
        # La posición está fija en el 0,0,0
        self.tamaño = (2.12, 1.06)
        self.amortiguador = 0.0365 # Grosor de los amortiguadores
        self.bolsillos = 0.06 # Diametro de los bolsillos

        self.model = sh.createNormalColorTable(pipeline, self.tamaño[0], self.tamaño[1], self.amortiguador, self.bolsillos)

    def draw(self):
        # Dibujar la mesa completa
        sg.drawSceneGraphNode(self.model, self.pipeline, "model")

class TACO:
    """ Taco que utilizará el jugador para golpear la pelota."""
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.position = None
        self.positionTemp = None
        self.model = sh.createNormalColorCuestrick(pipeline)
        self.phi = 0
        self.theta = 0

    def orientation(self, controller):
        """ Orienta al taco según la ubicación de las cámaras"""
        assert isinstance(controller, Controller)
        at = controller.getAtCamera()
        eye = controller.getEyeCamera()
        if controller.camara1:
            theta = controller.camera.theta # rotacionZ
            phi = controller.camera.phi #rotacionY
            if controller.rightClickOn and controller.disponible:
                self.phi = -phi-np.pi/2
            else:
                self.phi = -np.pi/2
            self.theta = theta+np.pi
            rx = 0.2*np.cos(theta)*np.sin(phi)
            ry = 0.2*np.sin(theta)*np.sin(phi)
            self.position = np.array([eye[0]-rx, eye[1]-ry, eye[2]-0.1])
        elif controller.camara3 and controller.rightClickOn and controller.disponible:
            theta = controller.camera.theta # rotacionZ
            self.phi = np.pi/50
            self.theta = theta + np.pi
            rx = 0.3*np.cos(theta)
            ry = 0.3*np.sin(theta)
            self.position = np.array([eye[0]+rx, eye[1]+ry, eye[2]-0.2])
        else:
            self.theta = 0
            self.phi = 0
            self.position = np.array([-0.1, 0.51, 0.87])


    def draw(self):
        # Dibujar el taco
        glUniformMatrix4fv(glGetUniformLocation(self.pipeline.shaderProgram, "model"), 1, GL_TRUE, tr.matmul([
            tr.translate(self.position[0], self.position[1], self.position[2]),
            tr.rotationZ(self.theta), tr.rotationY(self.phi)
        ])
        )
        self.pipeline.drawCall(self.model)




# Cámara en tercera persona
class ThirdCamera:
    def __init__(self, x, y , z):
        self.at = np.array([x, y, z])
        self.theta = -np.pi/2
        self.eye = np.array([x, y - 0.6, z + 0.3])
        self.up = np.array([0, 0, 1])

    # Determina el ángulo theta
    def set_theta(self, theta):
        self.theta = theta

    # Actualiza el objetivo a mirar
    def updateAt(self, x, y, z):
        self.at = np.array([x, y, z])

    # Actualiza la matriz de vista y la retorna
    def update_view(self):
        self.eye[0] = 0.6 * np.cos(self.theta) + self.at[0]
        self.eye[1] = 0.6 * np.sin(self.theta) + self.at[1]

        viewMatrix = tr.lookAt(
            self.eye,
            self.at,
            self.up
        )
        return viewMatrix

class SecondCamera:
    def __init__(self):
        self.at = np.array([0, 0, -1])
        self.eye = np.array([0., 0., 2.7])
        self.up = np.array([0, 1, 0])


    # Actualiza la matriz de vista y la retorna
    def update_view(self):
        viewMatrix = tr.lookAt(
            self.eye,
            self.at,
            self.up
        )
        return viewMatrix

class FirstCamera:
    def __init__(self, x, y, z):
        self.at = np.array([x, y + 1.0, z - 0.4])
        self.theta = 0
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
        self.at[0] = np.cos(self.theta) * np.sin(self.phi) + self.eye[0]
        self.at[1] = np.sin(self.theta) * np.sin(self.phi) + self.eye[1]
        self.at[2] = np.cos(self.phi) + self.eye[2]

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
        self.width = width
        self.height = height

        self.camera = SecondCamera()
        self.camara = 2
        self.camara1 = False
        self.camara2 = True
        self.camara3 = False

        self.light = 3

        self.selector = 0%16

        self.w = False
        self.s = False
        self.a = False
        self.d = False

        self.reset = False
        self.empezar = False
        self.disponible = True

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

            if key == glfw.KEY_1:
                self.camara = 1
        
            if key == glfw.KEY_2:
                self.camara = 2

            if key == glfw.KEY_3:
                self.camara = 3

            if key == glfw.KEY_Z:
                self.light = 1
        
            if key == glfw.KEY_X:
                self.light = 2

            if key == glfw.KEY_C:
                self.light = 3

            if key == glfw.KEY_V:
                self.light = 4

            if key == glfw.KEY_Q:
                self.selector = (self.selector-1)%16
            
            if key == glfw.KEY_E:
                self.selector = (self.selector+1)%16
            
            if key == glfw.KEY_UP:
                self.reset = True
                self.empezar = True

            if key == glfw.KEY_W:
                self.w = True

            if key == glfw.KEY_A:
                self.a = True

            if key == glfw.KEY_S:
                self.s = True

            if key == glfw.KEY_D:
                self.d = True
        
        elif action == glfw.RELEASE:
            if key == glfw.KEY_W:
                self.w = False

            if key == glfw.KEY_A:
                self.a = False

            if key == glfw.KEY_S:
                self.s = False

            if key == glfw.KEY_D:
                self.d = False

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
    def update_camera(self, delta, mesa, Bolas):
        dx, dy = mesa.tamaño[0]/2, mesa.tamaño[1]/2
        bolavista = Bolas[self.selector]
        # Selecciona la cámara a utilizar
        if self.camara==1 and not self.camara1:
            x = 0
            y = -dy-1
            z = 1.2
            self.camera = FirstCamera(x, y, z)
            self.camara1 = True
            self.camara2 = False
            self.camara3 = False
        elif self.camara==2 and not self.camara2:
            self.camera = SecondCamera()
            self.camara1 = False
            self.camara2 = True
            self.camara3 = False
        elif self.camara==3 and not self.camara3:
            pos = bolavista.position
            x = pos[0]
            y = pos[1]
            z = pos[2]
            self.camera = ThirdCamera(x, y, z)
            self.camara1 = False
            self.camara2=  False
            self.camara3 = True

        if self.camara1 or self.camara3:
            direction = self.camera.at[0:2] - self.camera.eye[0:2]
            direction = np.array([direction[0], direction[1], 0])
            direction /= np.linalg.norm(direction)
            rotatedir = np.array([-direction[1], direction[0], 0])
            theta = -self.mousePos[0] * 2 * np.pi - np.pi/2

            mouseY = self.mousePos[1]
            phi = mouseY * (np.pi/2-0.01) + np.pi/2

            if self.camara == 1:
                if self.w:
                    self.camera.eye += direction * delta

                if self.s:
                    self.camera.eye -= direction * delta

                if self.a:
                    self.camera.eye += rotatedir * delta

                if self.d:
                    self.camera.eye -= rotatedir * delta
                self.camera.set_phi(phi)

            if self.camara == 3:
                pos = bolavista.position
                x = pos[0]
                y = pos[1]
                z = pos[2]
                self.camera.updateAt(x, y, z)

            self.camera.set_theta(theta)
    
    def golpear(self, Bolas):
        """ Función que permite indicar que se va a golpear una pelota"""
        if self.camara == 3:
            bolavista = Bolas[self.selector]
            assert isinstance(bolavista, Bola)
            if self.disponible:
                if self.rightClickOn:
                    if self.leftClickOn:
                        self.disponible = False
                        theta = self.camera.theta + np.pi
                        v = np.array([2., 0., 0., 1.])
                        v = np.matmul(tr.rotationZ(theta),v)
                        bolavista.velocity = v[0:3]
        if self.camara == 1 and self.disponible:
            if self.rightClickOn:
                pos = self.camera.eye
                theta = self.camera.theta + np.pi
                phi = -self.camera.phi-np.pi/2
                puntoSelector = np.array([0.6, 0., 0., 1.])
                rx = 0.2*np.cos(theta)*np.sin(phi)
                ry = 0.2*np.sin(theta)*np.sin(phi)
                model = tr.matmul([tr.translate(pos[0]-rx, pos[1]-ry, pos[2]-0.2), tr.rotationZ(theta), tr.rotationY(phi)])
                puntoSelector = np.matmul(model, puntoSelector)[0:3]
                n = len(Bolas)
                print(puntoSelector)
                for i in range(n):
                    diferencia = puntoSelector-Bolas[i].position
                    distancia = np.linalg.norm(diferencia)
                    if distancia<=Bolas[i].diam/2:
                        print(i)
                        self.selector = i
                diferencia = puntoSelector-Bolas[self.selector].position
                distancia = np.linalg.norm(diferencia)
                if distancia<=Bolas[self.selector].diam/2 and self.leftClickOn:
                    self.disponible = False
                    v = np.array([2., 0., 0., 1.])
                    v = np.matmul(tr.rotationZ(theta-np.pi),v)
                    Bolas[self.selector].velocity = v[0:3]
                    


    def enMovimiento(self, Bolas):
        """ Si las bolas están en movimiento, bloquear golpear hasta que se queden quietas"""
        if not self.disponible:
            c=0
            n = len(Bolas)
            for bola in Bolas:
                if np.linalg.norm(bola.velocity) == 0:
                    c += 1
            if c==n:
                self.disponible = True




# Clase iluminación, crea los parámetros y las funciones para inicializar los shaders con normales.
class Iluminacion:
    def __init__(self):
        # Características de la luz por defecto
        self.LightPower = None
        self.lightConcentration = None
        self.lightShininess = None
        self.constantAttenuation = None
        self.linearAttenuation = None
        self.quadraticAttenuation = None
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


def colisionEsferas(bola1, bola2):
    """ Determina si las bolas están chocando entre sí, primero con una aproximación por cajas y luego entre circulos, en el plano 2D"""
    assert isinstance(bola1, Bola)
    assert isinstance(bola2, Bola)
    pos1 = bola1.position
    pos2 = bola2.position
    radio1 = bola1.diam/2
    radio2 = bola2.diam/2

    ColisionX = False
    ColisionY = False

    #Colision por cajas
    if pos1[0]-radio1<pos2[0]+radio2 and pos1[0]+radio1>pos2[0]-radio2: ColisionX = True
    if pos1[1]-radio1<pos2[1]+radio2 and pos1[1]+radio1>pos2[1]-radio2: ColisionY = True

    # Colision exacta
    if ColisionX and ColisionY:
        distancia = np.linalg.norm(pos2-pos1)
        distanciaColision = radio1+ radio2
        return distancia<distanciaColision
    return False

def GolpeEsferas(bola1, bola2, COEF): # Decidú que el golpe entre las pelotas sea elástico, esto debido a que la fuerza de roce y los choques inelásticos con los amortiguadores y borde es suficiente para detener el movimiento
    """ Habiendo un impacto entre las bolas de billar, se calcula sus nuevas velocidades considerando el coeficiente de restitución"""
    assert isinstance(bola1, Bola)
    assert isinstance(bola2, Bola)
    pos1 = bola1.position[0:2]
    pos2 = bola2.position[0:2]

    normal = pos2- pos1
    normal /= np.linalg.norm(normal)

    v1 = bola1.velocity[0:2]
    v2 = bola2.velocity[0:2]

    if not (np.dot(v1,normal)<0 and np.dot(v2, normal)>0):
        tangente = np.array([-normal[1], normal[0]])

        v1n = np.dot(v1, normal) * normal
        v1t = np.dot(v1, tangente) * tangente

        v2n = np.dot(v2, normal) * normal
        v2t = np.dot(v2, tangente) * tangente

        bola1.velocity[0:2] = v2n + v1t
        bola2.velocity[0:2] = v1n + v2t