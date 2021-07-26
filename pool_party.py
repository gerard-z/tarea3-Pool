""" Simulación 3D de una mesa de pool, implementada con las reglas de Blackball. Con medidas reales, considerando los ejes en metros."""

import glfw
from OpenGL.GL import *
import copy
import OpenGL.GL.shaders
import numpy as np
import grafica.transformations as tr
import grafica.basic_shapes as bs
import grafica.easy_shaders as es
import grafica.performance_monitor as pm
import grafica.scene_graph as sg
import grafica.newLightShaders as nl
from shapes3d import *
from resources import *
import sys
import json

" Se consideran que los ejes están en metros, por lo tanto las bolas que miden 51mm, tendrán un grosor de 0.051"

thisFilePath = os.path.abspath(__file__)
thisFolderPath = os.path.dirname(thisFilePath)
jason = os.path.join(thisFolderPath, "config.json")
#jason = str(sys.argv[1])
with open(jason,"r") as config:
    data = json.load(config)


ROCE = data["factor de friccion"]
COEF = data["coeficiente de restitucion"]


if __name__ == "__main__":

    # Initialize glfw
    if not glfw.init():
        glfw.set_window_should_close(window, True)

    width = 800
    height = 800
    title = "POOL"

    glfw.window_hint(glfw.SAMPLES, 4)
    window = glfw.create_window(width, height, title, None, None)

    if not window:
        glfw.terminate()
        glfw.set_window_should_close(window, True)

    glfw.make_context_current(window)

    controller = Controller(width, height)
    # Conectando las funciones: on_key, cursor_pos_callback, mouse_button_callback y scroll_callback del controlador al teclado y mouse
    glfw.set_key_callback(window, controller.on_key)

    glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_DISABLED)  # Deja la cámara centrada en la ventana con movimiento ilimitado
    glfw.set_cursor_pos(window, width/2, height/2)                  # Fija el mouse en el centro de la ventana
    glfw.set_cursor_pos_callback(window, controller.cursor_pos_callback)
    glfw.set_mouse_button_callback(window, controller.mouse_button_callback)


    # Diferentes shader 3D que consideran la iluminación de la linterna
    phongPipeline = nl.SimplePhongSpotlightShaderProgram()
    phongTexPipeline = nl.SimplePhongTextureSpotlightShaderProgram()
    phongOBJPipeline = nl.NormalPhongTextureSpotlightShaderProgram()

    # Este shader 3D no considera la iluminación de la linterna
    mvpPipeline = es.SimpleModelViewProjectionShaderProgram()
    texPipeline = es.SimpleTextureModelViewProjectionShaderProgram()

    # Este shader es uno en 2D
    pipeline2D = es.SimpleTransformShaderProgram()

    # Setting up the clear screen color
    glClearColor(0.65, 0.65, 0.65, 1.0)

    # As we work in 3D, we need to check which part is in front,
    # and which one is at the back
    glEnable(GL_DEPTH_TEST)

    # CREANDO GPU
    mesa = MESA(phongPipeline)
    taco = TACO(phongPipeline)

    bolaShape = createNormalTexSphere(40, 20)
    gpuBola1 = createTextureGPUShape(bolaShape, phongTexPipeline, texBola1, minFilterMode=GL_LINEAR)
    gpuBola2 = copy.deepcopy(gpuBola1)
    gpuBola2.texture = es.textureSimpleSetup(texBola2, GL_REPEAT, GL_REPEAT, GL_LINEAR, GL_LINEAR)
    gpuBola3 = copy.deepcopy(gpuBola1)
    gpuBola3.texture = es.textureSimpleSetup(texBola3, GL_REPEAT, GL_REPEAT, GL_LINEAR, GL_LINEAR)
    gpuBola4 = copy.deepcopy(gpuBola1)
    gpuBola4.texture = es.textureSimpleSetup(texBola4, GL_REPEAT, GL_REPEAT, GL_LINEAR, GL_LINEAR)
    gpuBola5 = copy.deepcopy(gpuBola1)
    gpuBola5.texture = es.textureSimpleSetup(texBola5, GL_REPEAT, GL_REPEAT, GL_LINEAR, GL_LINEAR)
    gpuBola6 = copy.deepcopy(gpuBola1)
    gpuBola6.texture = es.textureSimpleSetup(texBola6, GL_REPEAT, GL_REPEAT, GL_LINEAR, GL_LINEAR)
    gpuBola7 = copy.deepcopy(gpuBola1)
    gpuBola7.texture = es.textureSimpleSetup(texBola7, GL_REPEAT, GL_REPEAT, GL_LINEAR, GL_LINEAR)
    gpuBola8 = copy.deepcopy(gpuBola1)
    gpuBola8.texture = es.textureSimpleSetup(texBola8, GL_REPEAT, GL_REPEAT, GL_LINEAR, GL_LINEAR)
    gpuBola9 = copy.deepcopy(gpuBola1)
    gpuBola9.texture = es.textureSimpleSetup(texBola9, GL_REPEAT, GL_REPEAT, GL_LINEAR, GL_LINEAR)
    gpuBola10 = copy.deepcopy(gpuBola1)
    gpuBola10.texture = es.textureSimpleSetup(texBola10, GL_REPEAT, GL_REPEAT, GL_LINEAR, GL_LINEAR)
    gpuBola11 = copy.deepcopy(gpuBola1)
    gpuBola11.texture = es.textureSimpleSetup(texBola11, GL_REPEAT, GL_REPEAT, GL_LINEAR, GL_LINEAR)
    gpuBola12 = copy.deepcopy(gpuBola1)
    gpuBola12.texture = es.textureSimpleSetup(texBola12, GL_REPEAT, GL_REPEAT, GL_LINEAR, GL_LINEAR)
    gpuBola13 = copy.deepcopy(gpuBola1)
    gpuBola13.texture = es.textureSimpleSetup(texBola13, GL_REPEAT, GL_REPEAT, GL_LINEAR, GL_LINEAR)
    gpuBola14 = copy.deepcopy(gpuBola1)
    gpuBola14.texture = es.textureSimpleSetup(texBola14, GL_REPEAT, GL_REPEAT, GL_LINEAR, GL_LINEAR)
    gpuBola15 = copy.deepcopy(gpuBola1)
    gpuBola15.texture = es.textureSimpleSetup(texBola15, GL_REPEAT, GL_REPEAT, GL_LINEAR, GL_LINEAR)
    gpuBola0 = copy.deepcopy(gpuBola1)
    gpuBola0.texture = es.textureSimpleSetup(texBolaBlanca, GL_REPEAT, GL_REPEAT, GL_LINEAR, GL_LINEAR)

    gpuCirculo1 = createColorCircle(pipeline2D, 1, 0, 0, -0.95)
    gpuCirculo2 = createColorCircle(pipeline2D, 1, 0, 0, -0.9)
    gpuCirculo3 = createColorCircle(pipeline2D, 1, 0, 0, -0.85)
    gpuCirculo4 = createColorCircle(pipeline2D, 1, 0, 0, -0.8)
    gpuCirculo5 = createColorCircle(pipeline2D, 1, 0, 0, -0.75)
    camara = createCamaraIcon(pipeline2D)

    bola0 = Bola(phongTexPipeline, np.array([-0.5, 0., 0.86]), texPipeline)
    bola0.diam = 0.048
    bola0.setModel(gpuBola0)
    bola1 = Bola(phongTexPipeline, np.array([0.5, 0.102, 0.86]), texPipeline)
    bola1.setModel(gpuBola1)
    bola2 = Bola(phongTexPipeline, np.array([0.5, 0.051, 0.86]), texPipeline)
    bola2.setModel(gpuBola2)
    bola3 = Bola(phongTexPipeline, np.array([0.5, 0., 0.86]), texPipeline)
    bola3.setModel(gpuBola3)
    bola4 = Bola(phongTexPipeline, np.array([0.5, -0.051, 0.86]), texPipeline)
    bola4.setModel(gpuBola4)
    bola5 = Bola(phongTexPipeline, np.array([0.5, -0.102, 0.86]), texPipeline)
    bola5.setModel(gpuBola5)
    bola6 = Bola(phongTexPipeline, np.array([ 0.449, 0.0765, 0.86]), texPipeline)
    bola6.setModel(gpuBola6)
    bola7 = Bola(phongTexPipeline, np.array([0.449, 0.0255, 0.86]), texPipeline)
    bola7.setModel(gpuBola7)
    bola8 = Bola(phongTexPipeline, np.array([0.449, -0.0255, 0.86]), texPipeline)
    bola8.setModel(gpuBola8)
    bola9 = Bola(phongTexPipeline, np.array([0.449, -0.0765, 0.86]), texPipeline)
    bola9.setModel(gpuBola9)
    bola10 = Bola(phongTexPipeline, np.array([0.398, 0.051, 0.86]), texPipeline)
    bola10.setModel(gpuBola10)
    bola11 = Bola(phongTexPipeline, np.array([0.398, 0., 0.86]), texPipeline)
    bola11.setModel(gpuBola11)
    bola12 = Bola(phongTexPipeline, np.array([0.398, -0.051, 0.86]), texPipeline)
    bola12.setModel(gpuBola12)
    bola13 = Bola(phongTexPipeline, np.array([0.347, 0.0255, 0.86]), texPipeline)
    bola13.setModel(gpuBola13)
    bola14 = Bola(phongTexPipeline, np.array([0.347, -0.0255, 0.86]), texPipeline)
    bola14.setModel(gpuBola14)
    bola15 = Bola(phongTexPipeline, np.array([0.296, 0., 0.86]), texPipeline)
    bola15.setModel(gpuBola15)

    Bolas = [bola0, bola1, bola2, bola3, bola4, bola5, bola6, bola7, bola8, bola9, bola10, bola11, bola12, bola13, bola14, bola15]

    # iluminación
    lightPos = np.array([0., 0., 5])
    lightDirection = np.array([0, 0, -1])


    perfMonitor = pm.PerformanceMonitor(glfw.get_time(), 0.5)
    # glfw will swap buffers as soon as possible
    glfw.swap_interval(0)
    
    # Variables últies
    t0 = glfw.get_time()
    light = Iluminacion()

    # Application loop
    while not glfw.window_should_close(window):
        # Rendimiento en la pantalla
        perfMonitor.update(glfw.get_time())
        glfw.set_window_title(window, title + str(perfMonitor))
        
        # Variables del tiempo
        t1 = glfw.get_time()
        delta = t1 -t0
        t0 = t1

        # Using GLFW to check for input events
        glfw.poll_events()

        # Juego sin bola
        if controller.eliminados == 16:
            controller.camra = 2
            controller.selector = 0

        # ALgunos parámetros de movimiento
        if (glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS):
            delta *= 4
        if (glfw.get_key(window, glfw.KEY_LEFT_CONTROL) == glfw.PRESS):
            delta /= 4
        if (glfw.get_key(window, glfw.KEY_LEFT) == glfw.PRESS) and phi<=np.pi*0.3:
            phi += delta
        if (glfw.get_key(window, glfw.KEY_RIGHT) == glfw.PRESS) and phi>=-np.pi*0.2:
            phi -= delta
        # Definimos la cámara de la aplicación
        controller.update_camera(delta, mesa, Bolas)
        camera = controller.get_camera()

        viewMatrix = camera.update_view()

        #Eliminado del juego
        i = 0
        for bola in Bolas:
            if bola.borrar:
                controller.eliminados +=1
                if controller.eliminados == 16:
                    controller.selector = 0
                    controller.camara = 2
                else:
                    if controller.selector >= i:
                        controller.selector = (controller.selector-1)%(16-controller.eliminados)
                Bolas.remove(bola)
            i +=1

        # Físicas
        for bola in Bolas:
            bola.interactionTable(mesa, COEF)
            bola.move(ROCE, delta)

        controller.golpear(Bolas)
        controller.enMovimiento(Bolas)

        N = len(Bolas)
        for i in range(N):
            b1 = Bolas[i]
            for j in range(i+1, N):
                b2 = Bolas[j]
                if colisionEsferas(b1, b2):
                    GolpeEsferas(b1, b2, COEF)

        pos = controller.getEyeCamera()
        dir = controller.getAtCamera()

        # Orientacion de dibujos
        taco.orientation(controller)


        # definiendo parámetros del foco
        if controller.light==1:
            light.setLight(0.2, 2, 0.7, [0.01, 0.02, 0.03])

        elif controller.light==2:
            light.setLight(0.3, 2, 0.7, [0.01, 0.02, 0.03])

        elif controller.light==4:
            light.setLight(0, 1, 0, [0.01, 0.02, 0.03])

        else:
            light.setLight(0.5, 2, 0.7, [0.01, 0.02, 0.03])

        # Setting up the projection transform
        projection = tr.perspective(60, float(width) / float(height), 0.1, 100)

        # Clearing the screen in both, color and depth
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        # Filling or not the shapes depending on the controller state
        if (controller.fillPolygon):
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        else:
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)

        # Shaders para mider la fuerza de lanzamiento
        if controller.rightClickOn:
            fuerza = controller.Fuerza
            glUseProgram(pipeline2D.shaderProgram)
            glUniformMatrix4fv(glGetUniformLocation(pipeline2D.shaderProgram, "transform"), 1, GL_TRUE, tr.identity())
            pipeline2D.drawCall(gpuCirculo1)
            if fuerza>1:
                pipeline2D.drawCall(gpuCirculo2)
            if fuerza>2:
                pipeline2D.drawCall(gpuCirculo3)
            if fuerza>3:
                pipeline2D.drawCall(gpuCirculo4)
            if fuerza>4:
                pipeline2D.drawCall(gpuCirculo5)
            if controller.cinematica and controller.camara3:
                sg.drawSceneGraphNode(camara, pipeline2D, "transform")

        # Shader de iluminación para objetos sin texturas
        light.updateLight(phongPipeline, lightPos, lightDirection, lightPos)
        # Enviar matrices de transformaciones
        glUniformMatrix4fv(glGetUniformLocation(phongPipeline.shaderProgram, "projection"), 1, GL_TRUE, projection)
        glUniformMatrix4fv(glGetUniformLocation(phongPipeline.shaderProgram, "view"), 1, GL_TRUE, viewMatrix)

        # Iluminación del material
        glUniform3f(glGetUniformLocation(phongPipeline.shaderProgram, "Ka"), 0.2, 0.2, 0.2)
        glUniform3f(glGetUniformLocation(phongPipeline.shaderProgram, "Kd"), 0.4, 0.4, 0.4)
        glUniform3f(glGetUniformLocation(phongPipeline.shaderProgram, "Ks"), 0.6, 0.6, 0.6)

        #Draw
        glUniformMatrix4fv(glGetUniformLocation(phongPipeline.shaderProgram, "model"), 1, GL_TRUE, tr.matmul([tr.translate(0, 0 ,0), tr.uniformScale(1)]))
        taco.draw()
        glUniform1ui(glGetUniformLocation(phongPipeline.shaderProgram, "shininess"), 70)
        glUniform1ui(glGetUniformLocation(phongPipeline.shaderProgram, "concentration"), 1)
        mesa.draw()
        
        # Shader de texturas para dar efecto de movimiento de agua
        light.updateLight(phongTexPipeline, lightPos, lightDirection, controller.getEyeCamera())
        # Enviar matrices de transformaciones
        glUniformMatrix4fv(glGetUniformLocation(phongTexPipeline.shaderProgram, "projection"), 1, GL_TRUE, projection)
        glUniformMatrix4fv(glGetUniformLocation(phongTexPipeline.shaderProgram, "view"), 1, GL_TRUE, viewMatrix)

        # Iluminación del material
        glUniform3f(glGetUniformLocation(phongTexPipeline.shaderProgram, "Ka"), 0.2, 0.2, 0.2)
        glUniform3f(glGetUniformLocation(phongTexPipeline.shaderProgram, "Kd"), 0.5, 0.5, 0.5)
        glUniform3f(glGetUniformLocation(phongTexPipeline.shaderProgram, "Ks"), 1.0, 1.0, 1.0)

        # Drawing
        for bola in Bolas:
            bola.draw(projection, viewMatrix)

        # Shader de iluminación para objetos con texturas para color y normal
        light.updateLight(phongOBJPipeline, lightPos, lightDirection, lightPos)
        # Enviar matrices de transformaciones
        glUniformMatrix4fv(glGetUniformLocation(phongOBJPipeline.shaderProgram, "projection"), 1, GL_TRUE, projection)
        glUniformMatrix4fv(glGetUniformLocation(phongOBJPipeline.shaderProgram, "view"), 1, GL_TRUE, viewMatrix)

        # Iluminación del material
        glUniform3f(glGetUniformLocation(phongOBJPipeline.shaderProgram, "Ka"), 0.2, 0.2, 0.2)
        glUniform3f(glGetUniformLocation(phongOBJPipeline.shaderProgram, "Kd"), 0.5, 0.5, 0.5)
        glUniform3f(glGetUniformLocation(phongOBJPipeline.shaderProgram, "Ks"), 1.0, 1.0, 1.0)
        
        
        # Once the drawing is rendered, buffers are swap so an uncomplete drawing is never seen.
        glfw.swap_buffers(window)

    for bola in Bolas:
        bola.model.clear()
    mesa.model.clear()
    taco.model.clear()

    glfw.terminate()