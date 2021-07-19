""" Segunda parte, creación del tobogán, modelando un bote con OBJ, donde el tobogán tiene agua y obstáculos de por medio """

import glfw
from OpenGL.GL import *
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

" Se consideran que los ejes están en metros, por lo tanto las bolas que miden 51mm, tendrán un grosor de 0.051"


velocidad = 1
N = 10

if __name__ == "__main__":

    # Initialize glfw
    if not glfw.init():
        glfw.set_window_should_close(window, True)

    width = 800
    height = 800
    title = "POOL"

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

    # Este shader es uno en 2D
    pipeline2D = es.SimpleTransformShaderProgram()

    # Setting up the clear screen color
    glClearColor(0.65, 0.65, 0.65, 1.0)

    # As we work in 3D, we need to check which part is in front,
    # and which one is at the back
    glEnable(GL_DEPTH_TEST)

    # CREANDO GPU
    bolaShape = createNormalTexSphere(40, 20)
    gpuBola1 = createTextureGPUShape(bolaShape, phongTexPipeline, texBola1)

    toroidShape = createRandomColorNormalToroid(20)
    gpuToroid = createGPUShape(phongPipeline, toroidShape)


    bola1 = Bola(phongTexPipeline, np.array([0, 0, 0]))
    bola1.setModel(gpuBola1)

    perfMonitor = pm.PerformanceMonitor(glfw.get_time(), 0.5)
    # glfw will swap buffers as soon as possible
    glfw.swap_interval(0)
    
    # Variables últies
    t0 = glfw.get_time()
    light = Iluminacion()
    T = t0

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
        controller.update_camera(delta)
        camera = controller.get_camera()

        viewMatrix = camera.update_view()

        T += 3*delta * velocidad

        pos = controller.getEyeCamera()
        dir = controller.getAtCamera()

        # iluminación
        lightPos = np.array(pos)
        lightDirection = dir - pos


        # definiendo parámetros del foco
        if controller.light==1:
            light.setLight(0.6, 30, 1, [0.01, 0.03, 0.04])

        elif controller.light==2:
            light.setLight(0.8, 15, 1, [0.01, 0.02, 0.03])

        elif controller.light==4:
            light.setLight(0, 1, 0, [0.01, 0.03, 0.05])

        else:
            light.setLight(1, 6, 1, [0.01, 0.01, 0.01])

        # Setting up the projection transform
        projection = tr.perspective(60, float(width) / float(height), 0.1, 100)

        # Clearing the screen in both, color and depth
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Filling or not the shapes depending on the controller state
        if (controller.fillPolygon):
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        else:
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
        
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
        #bola1.draw()
        glUniformMatrix4fv(glGetUniformLocation(phongTexPipeline.shaderProgram, "model"), 1, GL_TRUE, tr.identity())
        phongTexPipeline.drawCall(gpuBola1)

        # Shader de iluminación para objetos sin texturas
        light.updateLight(phongPipeline, lightPos, lightDirection, lightPos)
        # Enviar matrices de transformaciones
        glUniformMatrix4fv(glGetUniformLocation(phongPipeline.shaderProgram, "projection"), 1, GL_TRUE, projection)
        glUniformMatrix4fv(glGetUniformLocation(phongPipeline.shaderProgram, "view"), 1, GL_TRUE, viewMatrix)

        # Iluminación del material
        glUniform3f(glGetUniformLocation(phongPipeline.shaderProgram, "Ka"), 0.2, 0.2, 0.2)
        glUniform3f(glGetUniformLocation(phongPipeline.shaderProgram, "Kd"), 0.5, 0.5, 0.5)
        glUniform3f(glGetUniformLocation(phongPipeline.shaderProgram, "Ks"), 1.0, 1.0, 1.0)

        #Draw
        glUniformMatrix4fv(glGetUniformLocation(phongPipeline.shaderProgram, "model"), 1, GL_TRUE, tr.matmul([tr.translate(10, 10 ,10), tr.uniformScale(10)]))
        phongPipeline.drawCall(gpuToroid)



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

    bola1.model.clear()
    gpuToroid.clear()

    glfw.terminate()